/**
 * Regions — connectivity within and between areas, on a platform with no atlas.
 *
 * The question a neuropil view exists to answer is *where does this area send
 * its output, and how much of its wiring never leaves home*. Answering it
 * normally starts from an atlas: a segmentation someone else drew, in which
 * every cell already carries the name of the structure it sits in.
 *
 * NeuroForge has no such segmentation. A circuit here is whatever the user or
 * the AI builder placed, so there is nothing to look a neuropil name up in.
 * Rather than invent one, the regions here are *derived*, by the only two
 * groupings the document actually supports:
 *
 *   - `gridRegions` cuts the bounding box of the cells into an N×N×N lattice.
 *     A lattice cell is a volume of space, not a brain area, and everything
 *     presented from it has to say so.
 *   - `populationRegions` treats each population as a region, which is the more
 *     meaningful grouping whenever a circuit was built from populations —
 *     those are the groups that were wired as units.
 *
 * Everything is measured off the live simulation buffers rather than the
 * document, for the same reason `graph-metrics` and `pathfinding` are: the
 * engine drops synapses whose endpoints were deleted and plasticity rewrites
 * weights every step, so anything derived from `circuit.synapses` would describe
 * a network nobody is watching.
 *
 * Storage is flat typed arrays throughout — a region index per slot, one CSR
 * membership list, one row-major matrix — with no per-cell and no per-pair
 * objects, which is what keeps a 50 000-cell partition affordable to cut, to
 * cross-tabulate and to re-rank on the main thread.
 */

import type { Circuit, SimulationBuffers } from '@neuroforge/shared';
import { identityColorHex } from '@neuroforge/shared';

import { graphSignature } from './graph-metrics';

/** How a partition was derived. Neither is anatomy; both must be labelled as such. */
export type RegionKind = 'grid' | 'population';

/** One division is the whole box, which is not a partition. */
export const MIN_GRID_DIVISIONS = 2;

/**
 * Ceiling on the lattice resolution. The cross-tabulation is regions², so the
 * cost of another division is cubic in the divisions and sextic in the matrix:
 * 8³ is 512 cells and a 262 144-entry matrix, which is already finer than any
 * panel has pixels to draw it in.
 */
export const MAX_GRID_DIVISIONS = 8;

/** Matches the largest lattice, and bounds the population partition too. */
export const MAX_REGIONS = MAX_GRID_DIVISIONS ** 3;

/** Members sampled for a region's colour strip. */
const DEFAULT_SWATCH_SAMPLES = 6;

/**
 * Cells a region needs before its locality index is allowed to rank it. A
 * single cell with one autapse is 100% locally connected and means nothing;
 * without a floor it would win the headline every time.
 */
export const DEFAULT_LOCALITY_FLOOR = 2;

/** A derived region: one lattice cell, or one population. */
export interface Region {
  /** Position in `RegionSet.regions`. */
  index: number;
  /**
   * Stable across rebuilds of the same partition of the same document, so a
   * selection or a hover can survive a recompute.
   */
  id: string;
  label: string;
  /** Live cells assigned to it, counted off the buffers. */
  size: number;
  /** Lattice coordinate, or -1 on a population region. */
  ix: number;
  iy: number;
  iz: number;
  /** Index into `circuit.populations`, or -1 for a lattice cell / the unassigned bucket. */
  population: number;
}

export interface RegionSet {
  kind: RegionKind;
  /** Identifies the network and the partition; see `regionSignature`. */
  signature: string;
  computeMs: number;
  /** Lattice divisions per axis; 0 for a population partition. */
  divisions: number;
  /** Neuron slots covered, parallel to the live buffers. */
  count: number;
  regions: readonly Region[];
  /** Region index per slot, or -1 when the slot belongs to no region. */
  regionOf: Int32Array;
  /** CSR offsets into `members`, length `regions.length + 1`. */
  memberStart: Uint32Array;
  /** Slots grouped by region, ascending within each region. */
  members: Uint32Array;
  /** Slots that landed in a region. */
  assigned: number;
  /** Slots that did not: an unplaceable position, or a folded-away population. */
  unassigned: number;
  /** Populations dropped by `MAX_REGIONS`; their cells count as unassigned. */
  omittedRegions: number;
  /** World box of the cells that were placed. Zeroed when none were. */
  min: Float32Array;
  max: Float32Array;
}

/**
 * Cheap identity for a partition. The divisions and the population count are
 * part of it because they define what a region *is*: changing either re-cuts
 * the whole thing even when not one synapse moved.
 */
export function regionSignature(
  buffers: SimulationBuffers,
  kind: RegionKind,
  divisions: number,
  populations: number,
): string {
  return `${kind}:${divisions}:${populations}:${graphSignature(buffers)}`;
}

function clampDivisions(divisions: number): number {
  const value = Math.floor(divisions);
  if (!Number.isFinite(value) || value < MIN_GRID_DIVISIONS) return MIN_GRID_DIVISIONS;
  return value > MAX_GRID_DIVISIONS ? MAX_GRID_DIVISIONS : value;
}

function emptyRegionSet(
  kind: RegionKind,
  divisions: number,
  signature: string,
  computeMs: number,
  count: number,
): RegionSet {
  const regionOf = new Int32Array(count);
  regionOf.fill(-1);
  return {
    kind,
    signature,
    computeMs,
    divisions,
    count,
    regions: [],
    regionOf,
    memberStart: new Uint32Array(1),
    members: new Uint32Array(0),
    assigned: 0,
    unassigned: count,
    omittedRegions: 0,
    min: new Float32Array(3),
    max: new Float32Array(3),
  };
}

/**
 * Fill the CSR membership list from a finished region assignment.
 *
 * `sizes` must already agree with `regionOf`; both are produced by the same
 * counting pass in each partitioner, so this only does the prefix sum and the
 * scatter.
 */
function buildMembership(
  regionOf: Int32Array,
  count: number,
  sizes: readonly number[],
): { memberStart: Uint32Array; members: Uint32Array } {
  const size = sizes.length;
  const memberStart = new Uint32Array(size + 1);
  for (let r = 0; r < size; r += 1) memberStart[r + 1] = memberStart[r] + sizes[r];
  const members = new Uint32Array(memberStart[size]);
  const cursor = new Uint32Array(size);
  for (let i = 0; i < count; i += 1) {
    const r = regionOf[i];
    if (r < 0) continue;
    members[memberStart[r] + cursor[r]] = i;
    cursor[r] += 1;
  }
  return { memberStart, members };
}

/* ------------------------------------------------------------------- grid -- */

/**
 * Cut the bounding box of the placed cells into an N×N×N lattice and keep the
 * cells that hold something.
 *
 * Empty lattice cells are dropped rather than carried as zero rows: a 6³
 * lattice over a circuit shaped like a column is mostly air, and 200 empty rows
 * would dominate both the list and the matrix while saying nothing.
 *
 * Ids are the lattice coordinate, so they survive a recompute — a region keeps
 * its identity as long as the division count and the extent of the network do.
 * Positions are read at the moment of the pass; a layout still relaxing moves
 * cells across the cuts, which is why the panel offers a re-cut rather than
 * pretending the partition is permanent.
 */
export function gridRegions(buffers: SimulationBuffers, divisions: number): RegionSet {
  const started = performance.now();
  const neurons = buffers.neurons;
  const n = neurons.count;
  const d = clampDivisions(divisions);
  const signature = regionSignature(buffers, 'grid', d, 0);

  if (n === 0) return emptyRegionSet('grid', d, signature, performance.now() - started, 0);

  const position = neurons.position;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let located = 0;

  for (let i = 0; i < n; i += 1) {
    const p = i * 3;
    const x = position[p];
    const y = position[p + 1];
    const z = position[p + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
    located += 1;
  }

  if (located === 0) return emptyRegionSet('grid', d, signature, performance.now() - started, n);

  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const spanZ = maxZ - minZ;
  // A degenerate axis — a sheet, a line, a single point — collapses to one
  // division instead of dividing by zero. The lattice cells it never reaches
  // are dropped by the compaction below like any other empty cell.
  const scaleX = spanX > 0 ? d / spanX : 0;
  const scaleY = spanY > 0 ? d / spanY : 0;
  const scaleZ = spanZ > 0 ? d / spanZ : 0;

  const lattice = d * d * d;
  const counts = new Uint32Array(lattice);
  const cellOf = new Int32Array(n);
  cellOf.fill(-1);

  for (let i = 0; i < n; i += 1) {
    const p = i * 3;
    const x = position[p];
    const y = position[p + 1];
    const z = position[p + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    // The cell on the far face lands exactly on `d` and is pulled back into the
    // last division, which is what stops a whole face becoming unassigned.
    const ix = scaleX > 0 ? Math.min(d - 1, Math.floor((x - minX) * scaleX)) : 0;
    const iy = scaleY > 0 ? Math.min(d - 1, Math.floor((y - minY) * scaleY)) : 0;
    const iz = scaleZ > 0 ? Math.min(d - 1, Math.floor((z - minZ) * scaleZ)) : 0;
    const cell = (ix * d + iy) * d + iz;
    cellOf[i] = cell;
    counts[cell] += 1;
  }

  const indexOfCell = new Int32Array(lattice);
  indexOfCell.fill(-1);
  const regions: Region[] = [];
  const sizes: number[] = [];

  for (let cell = 0; cell < lattice; cell += 1) {
    const size = counts[cell];
    if (size === 0) continue;
    const iz = cell % d;
    const iy = ((cell - iz) / d) % d;
    const ix = Math.floor(cell / (d * d));
    indexOfCell[cell] = regions.length;
    regions.push({
      index: regions.length,
      id: `grid:${d}:${ix}:${iy}:${iz}`,
      label: `X${ix + 1}·Y${iy + 1}·Z${iz + 1}`,
      size,
      ix,
      iy,
      iz,
      population: -1,
    });
    sizes.push(size);
  }

  const regionOf = new Int32Array(n);
  let unassigned = 0;
  for (let i = 0; i < n; i += 1) {
    const cell = cellOf[i];
    if (cell < 0) {
      regionOf[i] = -1;
      unassigned += 1;
      continue;
    }
    regionOf[i] = indexOfCell[cell];
  }

  const { memberStart, members } = buildMembership(regionOf, n, sizes);
  const min = new Float32Array(3);
  const max = new Float32Array(3);
  min[0] = minX;
  min[1] = minY;
  min[2] = minZ;
  max[0] = maxX;
  max[1] = maxY;
  max[2] = maxZ;

  return {
    kind: 'grid',
    signature,
    computeMs: performance.now() - started,
    divisions: d,
    count: n,
    regions,
    regionOf,
    memberStart,
    members,
    assigned: members.length,
    unassigned,
    omittedRegions: 0,
    min,
    max,
  };
}

/* ------------------------------------------------------------ populations -- */

/**
 * Treat every population that has live members as a region.
 *
 * Takes the buffers as well as the document because membership is read from the
 * buffers' population column rather than from `population.members`: the engine
 * is what decided which cell occupies which slot, and every other array here is
 * slot-indexed. A document whose populations have drifted from the running
 * network — an import mid-flight, an edit the engine has not reloaded yet —
 * would otherwise produce a partition that indexes cells nobody is simulating.
 *
 * Cells belonging to no population are collected into one trailing region
 * rather than dropped, because their wiring is part of every projection the
 * named populations take part in.
 */
export function populationRegions(circuit: Circuit, buffers: SimulationBuffers): RegionSet {
  const started = performance.now();
  const neurons = buffers.neurons;
  const n = neurons.count;
  const populations = circuit.populations;
  const popCount = Math.min(populations.length, 0xfffe);
  const signature = regionSignature(buffers, 'population', 0, populations.length);

  if (n === 0) return emptyRegionSet('population', 0, signature, performance.now() - started, 0);

  // One bucket per population plus a trailing one for the unaffiliated.
  const buckets = popCount + 1;
  const counts = new Uint32Array(buckets);
  const bucketOf = new Int32Array(n);
  for (let i = 0; i < n; i += 1) {
    const p = neurons.population[i];
    const bucket = p >= popCount ? popCount : p;
    bucketOf[i] = bucket;
    counts[bucket] += 1;
  }

  const occupied: number[] = [];
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    if (counts[bucket] > 0) occupied.push(bucket);
  }

  // Far beyond anything a document produces in practice, but a partition with
  // more rows than the matrix can hold has to lose some of them deliberately
  // rather than allocate a gigabyte of cross-tabulation.
  let kept = occupied;
  let omittedRegions = 0;
  if (occupied.length > MAX_REGIONS) {
    const largest = [...occupied].sort((a, b) => counts[b] - counts[a]).slice(0, MAX_REGIONS);
    const survives = new Set<number>(largest);
    kept = occupied.filter((bucket) => survives.has(bucket));
    omittedRegions = occupied.length - kept.length;
  }

  const indexOfBucket = new Int32Array(buckets);
  indexOfBucket.fill(-1);
  const regions: Region[] = [];
  const sizes: number[] = [];

  for (const bucket of kept) {
    const unaffiliated = bucket === popCount;
    const population = unaffiliated ? null : populations[bucket];
    const name = population === null ? '' : population.name;
    indexOfBucket[bucket] = regions.length;
    regions.push({
      index: regions.length,
      id: unaffiliated ? 'pop:unassigned' : `pop:${population?.id ?? bucket}`,
      label: unaffiliated ? 'Unaffiliated' : name.length > 0 ? name : `Population ${bucket + 1}`,
      size: counts[bucket],
      ix: -1,
      iy: -1,
      iz: -1,
      population: unaffiliated ? -1 : bucket,
    });
    sizes.push(counts[bucket]);
  }

  const regionOf = new Int32Array(n);
  let unassigned = 0;
  for (let i = 0; i < n; i += 1) {
    const r = indexOfBucket[bucketOf[i]];
    regionOf[i] = r;
    if (r < 0) unassigned += 1;
  }

  const { memberStart, members } = buildMembership(regionOf, n, sizes);

  const position = neurons.position;
  const min = new Float32Array(3);
  const max = new Float32Array(3);
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let located = 0;
  for (let i = 0; i < n; i += 1) {
    const p = i * 3;
    const x = position[p];
    const y = position[p + 1];
    const z = position[p + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
    located += 1;
  }
  if (located > 0) {
    min[0] = minX;
    min[1] = minY;
    min[2] = minZ;
    max[0] = maxX;
    max[1] = maxY;
    max[2] = maxZ;
  }

  return {
    kind: 'population',
    signature,
    computeMs: performance.now() - started,
    divisions: 0,
    count: n,
    regions,
    regionOf,
    memberStart,
    members,
    assigned: members.length,
    unassigned,
    omittedRegions,
    min,
    max,
  };
}

/* ----------------------------------------------------------------- matrix -- */

export interface RegionMatrix {
  /** The partition this was cross-tabulated from. */
  signature: string;
  computeMs: number;
  /** Rows and columns; equals `RegionSet.regions.length`. */
  size: number;
  /** Row-major size×size synapse counts. Rows are presynaptic. */
  count: Uint32Array;
  /** Row-major summed peak conductance (nS). */
  weight: Float64Array;
  /** The diagonal, lifted out: synapses with both ends in the same region. */
  recurrentCount: Uint32Array;
  recurrentWeight: Float64Array;
  /** Row and column marginals, the diagonal included. */
  outCount: Uint32Array;
  inCount: Uint32Array;
  outWeight: Float64Array;
  inWeight: Float64Array;
  /** Synapses that landed in a cell of the matrix. */
  synapses: number;
  /** Enabled synapses with at least one end in no region. */
  external: number;
  disabled: number;
  /** Synapses skipped because an endpoint is out of range. */
  dangling: number;
  maxCount: number;
  maxWeight: number;
  /**
   * The largest cell away from the diagonal. A connectome's recurrent totals
   * routinely dwarf every projection, so a colour scale anchored on `maxCount`
   * alone paints the whole off-diagonal black.
   */
  maxOffDiagonalCount: number;
  maxOffDiagonalWeight: number;
  /** Synapses on the diagonal, i.e. Σ recurrentCount. */
  recurrent: number;
}

/**
 * Cross-tabulate every live synapse by the regions of its two endpoints.
 *
 * One linear pass over the synapse list, writing into a row-major size×size
 * pair of arrays. Recurrent totals are the diagonal and are copied out
 * separately because that is the quantity the locality readouts are built on
 * and a strided read of a 262 144-entry matrix is not free.
 */
export function regionMatrix(regions: RegionSet, buffers: SimulationBuffers): RegionMatrix {
  const started = performance.now();
  const size = regions.regions.length;
  const synapses = buffers.synapses;
  const total = synapses.count;
  const n = Math.min(regions.count, buffers.neurons.count);
  const regionOf = regions.regionOf;

  const count = new Uint32Array(size * size);
  const weight = new Float64Array(size * size);
  const recurrentCount = new Uint32Array(size);
  const recurrentWeight = new Float64Array(size);
  const outCount = new Uint32Array(size);
  const inCount = new Uint32Array(size);
  const outWeight = new Float64Array(size);
  const inWeight = new Float64Array(size);

  let kept = 0;
  let external = 0;
  let disabled = 0;
  let dangling = 0;

  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) {
      disabled += 1;
      continue;
    }
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) {
      dangling += 1;
      continue;
    }
    const r = regionOf[pre];
    const c = regionOf[post];
    if (r < 0 || c < 0) {
      external += 1;
      continue;
    }
    const raw = synapses.weight[s];
    // A non-finite conductance is a broken synapse, not a strong one; it still
    // counts as wiring but must not poison the summed weight of a whole row.
    const w = Number.isFinite(raw) ? raw : 0;

    const at = r * size + c;
    count[at] += 1;
    weight[at] += w;
    outCount[r] += 1;
    inCount[c] += 1;
    outWeight[r] += w;
    inWeight[c] += w;
    kept += 1;
  }

  let maxCount = 0;
  let maxWeight = 0;
  let maxOffDiagonalCount = 0;
  let maxOffDiagonalWeight = 0;
  let recurrent = 0;

  for (let r = 0; r < size; r += 1) {
    const row = r * size;
    for (let c = 0; c < size; c += 1) {
      const cells = count[row + c];
      const mass = weight[row + c];
      if (cells > maxCount) maxCount = cells;
      if (mass > maxWeight) maxWeight = mass;
      if (r === c) continue;
      if (cells > maxOffDiagonalCount) maxOffDiagonalCount = cells;
      if (mass > maxOffDiagonalWeight) maxOffDiagonalWeight = mass;
    }
    recurrentCount[r] = count[row + r];
    recurrentWeight[r] = weight[row + r];
    recurrent += count[row + r];
  }

  return {
    signature: regions.signature,
    computeMs: performance.now() - started,
    size,
    count,
    weight,
    recurrentCount,
    recurrentWeight,
    outCount,
    inCount,
    outWeight,
    inWeight,
    synapses: kept,
    external,
    disabled,
    dangling,
    maxCount,
    maxWeight,
    maxOffDiagonalCount,
    maxOffDiagonalWeight,
    recurrent,
  };
}

/* ------------------------------------------------------------------ stats -- */

/**
 * Per-region statistics, held as parallel typed columns rather than as an
 * object per region.
 *
 * A table rather than a `RegionStats` per call because every column here comes
 * out of the same two passes: computing one region in isolation would walk the
 * whole synapse list for that region alone, which at 512 regions is 512 passes
 * over half a million synapses.
 */
export interface RegionStats {
  signature: string;
  computeMs: number;
  /** Number of regions; every column below has this length. */
  size: number;
  /** Live cells per region. */
  cells: Uint32Array;
  /** Members whose polarity is inhibitory. */
  inhibitory: Uint32Array;
  /** Mean firing rate at the moment of the pass (Hz). See `meanRateByRegion`. */
  meanRate: Float32Array;
  /** Synapses arriving at the region's cells, per cell. */
  meanInDegree: Float32Array;
  /** Synapses leaving the region's cells, per cell. */
  meanOutDegree: Float32Array;
  /**
   * Share of the connections touching this region that have both ends inside
   * it. NaN when nothing touches the region, where the quantity is undefined
   * rather than zero.
   */
  locality: Float32Array;
  /** Synapses with both ends inside the region, self-loops included. */
  internal: Uint32Array;
  /** Synapses leaving for another region, or for a cell in no region. */
  outgoing: Uint32Array;
  /** Synapses arriving from elsewhere. */
  incoming: Uint32Array;
  /** Centre of mass, 3 floats per region. */
  centroid: Float32Array;
  /** Bounding-box dimensions, 3 floats per region. */
  extent: Float32Array;
  /** RMS distance of the members from the centroid, in world units. */
  radius: Float32Array;
}

/**
 * Measure every region in three passes: one over the cells for position,
 * polarity and rate, one over the synapses for degree and locality, and one
 * more over the cells for the spread about the centroid the first pass could
 * not know yet.
 *
 * The locality index is the number this whole view exists to produce. It is
 * `internal / (internal + outgoing + incoming)`: of every connection with an
 * end in this region, the share that never leaves. A projection to a cell in no
 * region counts as leaving, because it does.
 */
export function regionStats(regions: RegionSet, buffers: SimulationBuffers): RegionStats {
  const started = performance.now();
  const size = regions.regions.length;
  const neurons = buffers.neurons;
  const synapses = buffers.synapses;
  const n = Math.min(regions.count, neurons.count);
  const regionOf = regions.regionOf;

  const cells = new Uint32Array(size);
  const inhibitory = new Uint32Array(size);
  const meanRate = new Float32Array(size);
  const meanInDegree = new Float32Array(size);
  const meanOutDegree = new Float32Array(size);
  const locality = new Float32Array(size);
  const internal = new Uint32Array(size);
  const outgoing = new Uint32Array(size);
  const incoming = new Uint32Array(size);
  const centroid = new Float32Array(size * 3);
  const extent = new Float32Array(size * 3);
  const radius = new Float32Array(size);

  if (size === 0) {
    return {
      signature: regions.signature,
      computeMs: performance.now() - started,
      size,
      cells,
      inhibitory,
      meanRate,
      meanInDegree,
      meanOutDegree,
      locality,
      internal,
      outgoing,
      incoming,
      centroid,
      extent,
      radius,
    };
  }

  const position = neurons.position;
  const sumX = new Float64Array(size);
  const sumY = new Float64Array(size);
  const sumZ = new Float64Array(size);
  const rateSum = new Float64Array(size);
  const placed = new Uint32Array(size);
  const boxMin = new Float64Array(size * 3).fill(Infinity);
  const boxMax = new Float64Array(size * 3).fill(-Infinity);

  for (let i = 0; i < n; i += 1) {
    const r = regionOf[i];
    if (r < 0) continue;
    cells[r] += 1;
    if (neurons.polarity[i] === 1) inhibitory[r] += 1;
    const rate = neurons.rate[i];
    if (Number.isFinite(rate)) rateSum[r] += rate;

    const p = i * 3;
    const x = position[p];
    const y = position[p + 1];
    const z = position[p + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    sumX[r] += x;
    sumY[r] += y;
    sumZ[r] += z;
    placed[r] += 1;
    const b = r * 3;
    if (x < boxMin[b]) boxMin[b] = x;
    if (y < boxMin[b + 1]) boxMin[b + 1] = y;
    if (z < boxMin[b + 2]) boxMin[b + 2] = z;
    if (x > boxMax[b]) boxMax[b] = x;
    if (y > boxMax[b + 1]) boxMax[b + 1] = y;
    if (z > boxMax[b + 2]) boxMax[b + 2] = z;
  }

  for (let r = 0; r < size; r += 1) {
    const members = placed[r];
    const b = r * 3;
    if (members > 0) {
      centroid[b] = sumX[r] / members;
      centroid[b + 1] = sumY[r] / members;
      centroid[b + 2] = sumZ[r] / members;
      extent[b] = boxMax[b] - boxMin[b];
      extent[b + 1] = boxMax[b + 1] - boxMin[b + 1];
      extent[b + 2] = boxMax[b + 2] - boxMin[b + 2];
    }
    meanRate[r] = cells[r] > 0 ? rateSum[r] / cells[r] : 0;
  }

  const spread = new Float64Array(size);
  for (let i = 0; i < n; i += 1) {
    const r = regionOf[i];
    if (r < 0 || placed[r] === 0) continue;
    const p = i * 3;
    const x = position[p];
    const y = position[p + 1];
    const z = position[p + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    const b = r * 3;
    const dx = x - centroid[b];
    const dy = y - centroid[b + 1];
    const dz = z - centroid[b + 2];
    spread[r] += dx * dx + dy * dy + dz * dz;
  }
  for (let r = 0; r < size; r += 1) {
    radius[r] = placed[r] > 0 ? Math.sqrt(spread[r] / placed[r]) : 0;
  }

  const outDegree = new Uint32Array(size);
  const inDegree = new Uint32Array(size);
  const total = synapses.count;

  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    const r = regionOf[pre];
    const c = regionOf[post];
    if (r >= 0) {
      outDegree[r] += 1;
      if (r === c) internal[r] += 1;
      else outgoing[r] += 1;
    }
    if (c >= 0) {
      inDegree[c] += 1;
      if (r !== c) incoming[c] += 1;
    }
  }

  for (let r = 0; r < size; r += 1) {
    const members = cells[r];
    meanInDegree[r] = members > 0 ? inDegree[r] / members : 0;
    meanOutDegree[r] = members > 0 ? outDegree[r] / members : 0;
    const incident = internal[r] + outgoing[r] + incoming[r];
    locality[r] = incident > 0 ? internal[r] / incident : Number.NaN;
  }

  return {
    signature: regions.signature,
    computeMs: performance.now() - started,
    size,
    cells,
    inhibitory,
    meanRate,
    meanInDegree,
    meanOutDegree,
    locality,
    internal,
    outgoing,
    incoming,
    centroid,
    extent,
    radius,
  };
}

/**
 * Mean firing rate per region, read straight off the live buffers.
 *
 * Kept out of the stats pass deliberately: rates move every step and the stats
 * are cached against the topology, so folding one into the other would either
 * freeze the rates or throw the whole table away sixty times a second. `out`
 * must hold at least `regions.regions.length` entries and is overwritten.
 */
export function meanRateByRegion(
  regions: RegionSet,
  buffers: SimulationBuffers,
  out: Float32Array,
): void {
  const size = regions.regions.length;
  if (size === 0) return;
  out.fill(0, 0, size);
  const counts = new Uint32Array(size);
  const neurons = buffers.neurons;
  // The engine can be reloaded underneath a partition; only the prefix both
  // agree on is meaningful, and a shorter one heals on the next re-cut.
  const n = Math.min(regions.count, neurons.count);
  for (let i = 0; i < n; i += 1) {
    const r = regions.regionOf[i];
    if (r < 0) continue;
    const rate = neurons.rate[i];
    if (!Number.isFinite(rate)) continue;
    out[r] += rate;
    counts[r] += 1;
  }
  for (let r = 0; r < size; r += 1) {
    out[r] = counts[r] > 0 ? out[r] / counts[r] : 0;
  }
}

/* ----------------------------------------------------------------- ranking -- */

export interface LocalityExtremes {
  /** Region index with the largest locality index, or -1 when none qualifies. */
  most: number;
  /** Region index with the smallest, or -1. */
  least: number;
  /** Regions that met the floor and had at least one incident connection. */
  considered: number;
}

/**
 * The most and least locally-connected regions — the answer this view exists to
 * give. Regions below `floor` cells are ignored: locality over a handful of
 * cells is dominated by whether they happen to be wired to each other, and a
 * singleton with one autapse would otherwise take the top spot every time.
 *
 * Ties resolve towards the lower region index, because the scan runs in region
 * order and only a strictly better value displaces the incumbent.
 */
export function localityExtremes(
  stats: RegionStats,
  floor = DEFAULT_LOCALITY_FLOOR,
): LocalityExtremes {
  let most = -1;
  let least = -1;
  let bestValue = -Infinity;
  let worstValue = Infinity;
  let considered = 0;

  for (let r = 0; r < stats.size; r += 1) {
    if (stats.cells[r] < floor) continue;
    const value = stats.locality[r];
    if (!Number.isFinite(value)) continue;
    considered += 1;
    if (value > bestValue) {
      bestValue = value;
      most = r;
    }
    if (value < worstValue) {
      worstValue = value;
      least = r;
    }
  }

  return { most, least, considered };
}

/* ---------------------------------------------------------------- members -- */

/** The slots belonging to one region, as a view into the CSR list. */
export function regionMembers(regions: RegionSet, index: number): Uint32Array {
  if (index < 0 || index >= regions.regions.length) return new Uint32Array(0);
  return regions.members.subarray(regions.memberStart[index], regions.memberStart[index + 1]);
}

/**
 * The cells that actually take part in one projection: the presynaptic cells of
 * `source` that reach `target`, together with the postsynaptic cells of
 * `target` they reach.
 *
 * Not the same as the union of the two regions — most of a region takes no part
 * in most of its projections, and selecting everything at both ends would hide
 * exactly the cells the click was asking about. With `source === target` this
 * returns the cells carrying the region's recurrent wiring.
 */
export function projectionMembers(
  regions: RegionSet,
  buffers: SimulationBuffers,
  source: number,
  target: number,
): Uint32Array {
  const size = regions.regions.length;
  if (source < 0 || source >= size || target < 0 || target >= size) return new Uint32Array(0);

  const n = Math.min(regions.count, buffers.neurons.count);
  const synapses = buffers.synapses;
  const total = synapses.count;
  const regionOf = regions.regionOf;
  const marked = new Uint8Array(n);
  let found = 0;

  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    if (regionOf[pre] !== source || regionOf[post] !== target) continue;
    if (marked[pre] === 0) {
      marked[pre] = 1;
      found += 1;
    }
    if (marked[post] === 0) {
      marked[post] = 1;
      found += 1;
    }
  }

  const out = new Uint32Array(found);
  let write = 0;
  for (let i = 0; i < n && write < found; i += 1) {
    if (marked[i] === 1) {
      out[write] = i;
      write += 1;
    }
  }
  return out;
}

/* --------------------------------------------------------------- swatches -- */

/**
 * A colour strip per region, sampled from the identity colours of its members.
 *
 * A derived region has no colour of its own — there is no atlas to take one
 * from, and a hash of the region id would be a colour the 3D scene has never
 * heard of. Sampling the members instead means every segment of the strip is
 * literally `identityColorHex(seed)` of a cell that is drawn in the viewport, so
 * a row in the list and a cluster of glyphs in the scene are the same object.
 *
 * Members are evenly spaced through the region rather than taken from the
 * front, which for a spatial partition means the strip spans the cell rather
 * than describing one corner of it.
 */
export function regionSwatches(
  regions: RegionSet,
  buffers: SimulationBuffers,
  samples = DEFAULT_SWATCH_SAMPLES,
): string[][] {
  const wanted = Math.max(1, Math.floor(samples));
  const seed = buffers.neurons.seed;
  const n = buffers.neurons.count;
  const strips: string[][] = [];

  for (let r = 0; r < regions.regions.length; r += 1) {
    const begin = regions.memberStart[r];
    const size = regions.memberStart[r + 1] - begin;
    const take = Math.min(wanted, size);
    const strip: string[] = [];
    for (let k = 0; k < take; k += 1) {
      const slot = regions.members[begin + Math.floor((k * size) / take)];
      if (slot < n) strip.push(identityColorHex(seed[slot]));
    }
    strips.push(strip);
  }
  return strips;
}
