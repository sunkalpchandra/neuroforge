/**
 * Connectivity fingerprints — cell typing without labels.
 *
 * A connectome does not come with cell types written on it. What it does come
 * with is wiring, and the standard way to recover a type from wiring is the
 * *connectivity fingerprint*: the vector of synaptic weight a cell sends to, and
 * receives from, every population in the network. Two cells that talk to the
 * same places in the same proportions are the same kind of cell, whatever the
 * user happened to name them.
 *
 * Everything here is measured off the live simulation buffers rather than the
 * document, for the same reason `graph-metrics` is: the engine drops synapses
 * whose endpoints were deleted and plasticity rewrites weights as the circuit
 * runs, so a fingerprint built from `circuit.synapses` would describe a network
 * nobody is watching.
 *
 * Storage is one dense row-major Float32Array of `count × dim` — no per-neuron
 * and no per-pair objects — which is what keeps a 20k-cell network affordable to
 * fingerprint, rank and cluster on the main thread.
 */

import type { SimulationBuffers } from '@neuroforge/shared';
import { Rng } from '@neuroforge/math';

import { graphSignature } from './graph-metrics';

/** `populationIndex` of the bucket holding cells that belong to no population. */
export const GROUP_UNASSIGNED = -1;
/** `populationIndex` of the bucket populations past `MAX_FINGERPRINT_GROUPS` fold into. */
export const GROUP_OVERFLOW = -2;

/**
 * Ceiling on the number of fingerprint dimensions.
 *
 * The fingerprint matrix is `neurons × 2 · groups` floats, so the group count
 * multiplies the whole allocation. A document with a handful of populations —
 * which is every document in practice — is nowhere near this; a pathological one
 * folds its smallest populations into a single overflow bucket rather than
 * quietly allocating hundreds of megabytes.
 */
export const MAX_FINGERPRINT_GROUPS = 48;

/** Model code shared by every member of a group, or this when they disagree. */
export const MODEL_MIXED = -1;

/** One column pair of the fingerprint: everything a cell sends to and gets from a group. */
export interface FingerprintGroup {
  /** Index into `circuit.populations`, or GROUP_UNASSIGNED / GROUP_OVERFLOW. */
  populationIndex: number;
  /** Populations this row stands for; greater than one only for the overflow bucket. */
  populations: number;
  /** Live members, counted off the buffers rather than off `population.members`. */
  size: number;
  /** Members whose polarity is inhibitory. */
  inhibitory: number;
  /** MODEL_CODE shared by every member, or MODEL_MIXED. */
  model: number;
  meanInDegree: number;
  meanOutDegree: number;
  /** Summed peak conductance arriving at / leaving this group (nS). */
  inWeight: number;
  outWeight: number;
}

export interface Fingerprints {
  /** Identifies the network this was built from; see `fingerprintSignature`. */
  signature: string;
  computeMs: number;

  /** Neuron slots covered, parallel to the live buffers. */
  count: number;
  /** `2 · groups.length`: the out-block followed by the in-block. */
  dim: number;
  groups: readonly FingerprintGroup[];

  /** Group index per slot. Every live slot belongs to exactly one group. */
  groupOf: Int32Array;
  /** Row-major `count × dim`. Each row of a connected cell has unit L2 norm. */
  data: Float32Array;

  inDegree: Uint32Array;
  outDegree: Uint32Array;
  /** Summed peak conductance per cell (nS), before normalisation. */
  inWeight: Float32Array;
  outWeight: Float32Array;

  /** 1 where the cell has at least one synapse, so its row has a direction. */
  connected: Uint8Array;
  connectedCount: number;
  /** Synapses that contributed: enabled, finite weight, endpoints still live. */
  synapses: number;
}

/**
 * Cheap identity for a fingerprint. The population count is part of it because
 * the groups are the fingerprint's axes: adding a population re-bases every
 * vector even when not one synapse changed.
 */
export function fingerprintSignature(
  buffers: SimulationBuffers,
  populationCount: number,
): string {
  return `${graphSignature(buffers)}:${populationCount}`;
}

/* ------------------------------------------------------------------- build -- */

interface GroupPlan {
  /** Compact group index per raw bucket; `popCount` is the unassigned bucket. */
  indexOf: Int32Array;
  populationIndex: Int32Array;
  populations: Uint32Array;
  count: number;
}

/**
 * Decide which populations get their own fingerprint dimension.
 *
 * Empty populations are dropped: a dimension no cell belongs to is a column of
 * zeros that costs memory and changes no distance. When more populations carry
 * members than there are dimensions to spare, the largest keep their own column
 * and the tail shares one, so the axes that dominate the wiring survive intact.
 */
function planGroups(population: Uint16Array, n: number, popCount: number): GroupPlan {
  const buckets = popCount + 1;
  const counts = new Uint32Array(buckets);
  for (let i = 0; i < n; i += 1) {
    const p = population[i];
    counts[p < popCount ? p : popCount] += 1;
  }

  const occupied: number[] = [];
  for (let b = 0; b < buckets; b += 1) if (counts[b] > 0) occupied.push(b);

  const indexOf = new Int32Array(buckets).fill(-1);
  if (occupied.length === 0) {
    return {
      indexOf,
      populationIndex: new Int32Array(0),
      populations: new Uint32Array(0),
      count: 0,
    };
  }

  // Under the ceiling every occupied bucket keeps its own column, in population
  // order, which is the order the panel lists them in.
  if (occupied.length <= MAX_FINGERPRINT_GROUPS) {
    const populationIndex = new Int32Array(occupied.length);
    const populations = new Uint32Array(occupied.length);
    occupied.forEach((bucket, at) => {
      indexOf[bucket] = at;
      populationIndex[at] = bucket === popCount ? GROUP_UNASSIGNED : bucket;
      populations[at] = 1;
    });
    return { indexOf, populationIndex, populations, count: occupied.length };
  }

  const keep = new Set<number>(
    [...occupied].sort((a, b) => counts[b] - counts[a]).slice(0, MAX_FINGERPRINT_GROUPS - 1),
  );
  const kept = occupied.filter((bucket) => keep.has(bucket));
  const folded = occupied.length - kept.length;

  const total = kept.length + 1;
  const populationIndex = new Int32Array(total);
  const populations = new Uint32Array(total);
  kept.forEach((bucket, at) => {
    indexOf[bucket] = at;
    populationIndex[at] = bucket === popCount ? GROUP_UNASSIGNED : bucket;
    populations[at] = 1;
  });
  const overflow = kept.length;
  populationIndex[overflow] = GROUP_OVERFLOW;
  populations[overflow] = folded;
  for (const bucket of occupied) if (!keep.has(bucket)) indexOf[bucket] = overflow;

  return { indexOf, populationIndex, populations, count: total };
}

function emptyFingerprints(signature: string, computeMs: number, count: number): Fingerprints {
  return {
    signature,
    computeMs,
    count,
    dim: 0,
    groups: [],
    groupOf: new Int32Array(count),
    data: new Float32Array(0),
    inDegree: new Uint32Array(count),
    outDegree: new Uint32Array(count),
    inWeight: new Float32Array(count),
    outWeight: new Float32Array(count),
    connected: new Uint8Array(count),
    connectedCount: 0,
    synapses: 0,
  };
}

/**
 * Build every cell's connectivity fingerprint in one pass over the synapses.
 *
 * The out-block and the in-block are each normalised to unit length before the
 * row as a whole is, so a cell's input profile and its output profile weigh the
 * same however lopsided its wiring is. Normalising the row in one go instead
 * would let a cell with two hundred afferents and one efferent be typed almost
 * entirely by what it listens to, which is exactly the distinction the second
 * half of the vector exists to make.
 *
 * Cells with no synapses keep an all-zero row. They have no direction to
 * compare, so they are excluded from ranking and clustering rather than being
 * silently declared identical to one another.
 */
export function buildFingerprints(
  buffers: SimulationBuffers,
  populationCount: number,
): Fingerprints {
  const started = performance.now();
  const signature = fingerprintSignature(buffers, populationCount);
  const neurons = buffers.neurons;
  const synapses = buffers.synapses;
  const n = neurons.count;
  const popCount = Math.max(0, Math.min(0xfffe, Math.floor(populationCount)));

  if (n === 0) return emptyFingerprints(signature, performance.now() - started, 0);

  const plan = planGroups(neurons.population, n, popCount);
  const g = plan.count;
  if (g === 0) return emptyFingerprints(signature, performance.now() - started, n);

  const dim = g * 2;
  const groupOf = new Int32Array(n);
  for (let i = 0; i < n; i += 1) {
    const p = neurons.population[i];
    groupOf[i] = plan.indexOf[p < popCount ? p : popCount];
  }

  const data = new Float32Array(n * dim);
  const inDegree = new Uint32Array(n);
  const outDegree = new Uint32Array(n);
  const inWeight = new Float32Array(n);
  const outWeight = new Float32Array(n);

  let kept = 0;
  const total = synapses.count;
  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    const weight = synapses.weight[s];
    if (!Number.isFinite(weight)) continue;
    // Magnitude only: which population a partner belongs to already carries the
    // sign, and a signed sum would let an excitatory and an inhibitory synapse
    // onto the same group cancel into "no connection at all".
    const w = Math.abs(weight);

    data[pre * dim + groupOf[post]] += w;
    data[post * dim + g + groupOf[pre]] += w;
    outDegree[pre] += 1;
    inDegree[post] += 1;
    outWeight[pre] += w;
    inWeight[post] += w;
    kept += 1;
  }

  const connected = new Uint8Array(n);
  let connectedCount = 0;
  for (let i = 0; i < n; i += 1) {
    const row = i * dim;
    let outSq = 0;
    let inSq = 0;
    for (let d = 0; d < g; d += 1) {
      const out = data[row + d];
      const inc = data[row + g + d];
      outSq += out * out;
      inSq += inc * inc;
    }
    if (outSq === 0 && inSq === 0) continue;

    // Each present block is scaled to 1/sqrt(blocks), so the row lands on the
    // unit sphere whether the cell has one direction of wiring or both.
    const blocks = (outSq > 0 ? 1 : 0) + (inSq > 0 ? 1 : 0);
    const share = 1 / Math.sqrt(blocks);
    const outScale = outSq > 0 ? share / Math.sqrt(outSq) : 0;
    const inScale = inSq > 0 ? share / Math.sqrt(inSq) : 0;
    for (let d = 0; d < g; d += 1) {
      data[row + d] *= outScale;
      data[row + g + d] *= inScale;
    }
    connected[i] = 1;
    connectedCount += 1;
  }

  /* -- group aggregates ---------------------------------------------------- */

  const sizes = new Uint32Array(g);
  const inhibitory = new Uint32Array(g);
  const models = new Int32Array(g).fill(-2);
  const inDegreeSum = new Float64Array(g);
  const outDegreeSum = new Float64Array(g);
  const inWeightSum = new Float64Array(g);
  const outWeightSum = new Float64Array(g);

  for (let i = 0; i < n; i += 1) {
    const index = groupOf[i];
    sizes[index] += 1;
    if (neurons.polarity[i] === 1) inhibitory[index] += 1;
    const model = neurons.model[i];
    if (models[index] === -2) models[index] = model;
    else if (models[index] !== model) models[index] = MODEL_MIXED;
    inDegreeSum[index] += inDegree[i];
    outDegreeSum[index] += outDegree[i];
    inWeightSum[index] += inWeight[i];
    outWeightSum[index] += outWeight[i];
  }

  const groups: FingerprintGroup[] = [];
  for (let index = 0; index < g; index += 1) {
    const size = sizes[index];
    groups.push({
      populationIndex: plan.populationIndex[index],
      populations: plan.populations[index],
      size,
      inhibitory: inhibitory[index],
      model: models[index] === -2 ? MODEL_MIXED : models[index],
      meanInDegree: size > 0 ? inDegreeSum[index] / size : 0,
      meanOutDegree: size > 0 ? outDegreeSum[index] / size : 0,
      inWeight: inWeightSum[index],
      outWeight: outWeightSum[index],
    });
  }

  return {
    signature,
    computeMs: performance.now() - started,
    count: n,
    dim,
    groups,
    groupOf,
    data,
    inDegree,
    outDegree,
    inWeight,
    outWeight,
    connected,
    connectedCount,
    synapses: kept,
  };
}

/**
 * Mean firing rate per group, read straight off the live buffers.
 *
 * Kept out of `Fingerprints` deliberately: rates move every step and the
 * fingerprint is cached against the topology, so folding one into the other
 * would either freeze the rates or throw away the fingerprint sixty times a
 * second. `out` must hold at least `groups.length` entries and is overwritten.
 */
export function meanRateByGroup(
  buffers: SimulationBuffers,
  prints: Fingerprints,
  out: Float32Array,
): void {
  const g = prints.groups.length;
  if (g === 0) return;
  out.fill(0, 0, g);
  const counts = new Uint32Array(g);
  const neurons = buffers.neurons;
  // The engine can be reloaded underneath a fingerprint; only the prefix both
  // agree on is meaningful, and a shorter one heals on the next rebuild.
  const n = Math.min(prints.count, neurons.count);
  for (let i = 0; i < n; i += 1) {
    const index = prints.groupOf[i];
    out[index] += neurons.rate[i];
    counts[index] += 1;
  }
  for (let index = 0; index < g; index += 1) {
    out[index] = counts[index] > 0 ? out[index] / counts[index] : 0;
  }
}

/* ---------------------------------------------------------------- ranking -- */

export interface SimilarCell {
  slot: number;
  /** Cosine similarity of the fingerprints, 0..1; components are non-negative. */
  score: number;
  group: number;
  inDegree: number;
  outDegree: number;
}

/** Upper bound on `k`, so a stray argument cannot ask for a list nothing renders. */
const MAX_SIMILAR = 256;

/**
 * The `k` cells whose fingerprint points most nearly the same way as `slot`'s.
 *
 * Rows are unit vectors, so the cosine is a plain dot product and the whole scan
 * is one pass of `count × dim` multiply-adds with a fixed-size insertion list on
 * top — no pair objects, no sort of the network. Ties resolve towards the lower
 * slot because the scan runs in slot order and only a strictly better score
 * displaces an entry.
 */
export function similarCells(prints: Fingerprints, slot: number, k: number): SimilarCell[] {
  const { count, dim, data, connected } = prints;
  if (dim === 0 || slot < 0 || slot >= count || connected[slot] === 0) return [];

  const wanted = Math.max(1, Math.min(Math.floor(k), MAX_SIMILAR, Math.max(1, count - 1)));
  const bestSlots = new Int32Array(wanted);
  const bestScores = new Float32Array(wanted);
  let length = 0;

  const query = slot * dim;
  for (let i = 0; i < count; i += 1) {
    if (i === slot || connected[i] === 0) continue;
    const row = i * dim;
    let dot = 0;
    for (let d = 0; d < dim; d += 1) dot += data[query + d] * data[row + d];
    if (dot <= 0) continue;
    if (length === wanted && dot <= bestScores[length - 1]) continue;

    let at = length < wanted ? length : wanted - 1;
    if (length < wanted) length += 1;
    while (at > 0 && bestScores[at - 1] < dot) {
      bestScores[at] = bestScores[at - 1];
      bestSlots[at] = bestSlots[at - 1];
      at -= 1;
    }
    bestScores[at] = dot;
    bestSlots[at] = i;
  }

  const result: SimilarCell[] = [];
  for (let i = 0; i < length; i += 1) {
    const other = bestSlots[i];
    result.push({
      slot: other,
      // Floating-point accumulation can push a perfect match a hair past one.
      score: Math.min(1, bestScores[i]),
      group: prints.groupOf[other],
      inDegree: prints.inDegree[other],
      outDegree: prints.outDegree[other],
    });
  }
  return result;
}

/* -------------------------------------------------------------- clustering -- */

export interface ClusterOptions {
  /** Fixed by default, so the same circuit always produces the same clusters. */
  seed?: number;
  maxIterations?: number;
}

export interface ClusterRun {
  k: number;
  seed: number;
  /** Cluster index per slot; -1 for cells excluded for having no synapses. */
  assignment: Int32Array;
  sizes: Uint32Array;
  /** Clusters that ended up with at least one member. */
  occupied: number;
  /** Lloyd iterations actually run. */
  iterations: number;
  maxIterations: number;
  /** False when the run hit `maxIterations` with assignments still moving. */
  converged: boolean;
  /** Members that changed cluster on the final iteration. */
  unsettled: number;
  /** Summed cosine distance from each cell to its centroid; lower is tighter. */
  inertia: number;
  members: number;
  excluded: number;
  /** Row-major `k × groups`: how many cells of each population each cluster holds. */
  contingency: Uint32Array;
  /** Group index each cluster is mostly made of, or -1 when the cluster is empty. */
  dominant: Int32Array;
  dominantShare: Float32Array;
  /** Share of cells whose cluster's dominant population is their own. NaN when empty. */
  purity: number;
  /**
   * Normalised mutual information between the clustering and the user's
   * populations, 0..1. NaN when either side has fewer than two occupied
   * categories, where the quantity is undefined rather than zero.
   */
  nmi: number;
  computeMs: number;
}

const DEFAULT_CLUSTER_SEED = 0x9e3779b9;
const DEFAULT_MAX_ITERATIONS = 64;

/**
 * Spherical k-means over the fingerprints, driven one iteration at a time.
 *
 * Iterations are exposed rather than run in a loop so a caller can yield to the
 * event loop between them and stop early: a full Lloyd pass is
 * `members × dim × k` multiply-adds, which at twenty thousand cells is long
 * enough to drop a frame and far too long to be uninterruptible.
 *
 * The rows are unit vectors, so squared Euclidean distance is `2(1 − cos)` and
 * the nearest centroid is simply the one with the largest dot product. Centroids
 * are re-normalised after every update, which keeps that equivalence exact
 * instead of letting the means drift inside the sphere.
 */
export class FingerprintClustering {
  readonly k: number;
  readonly seed: number;
  readonly maxIterations: number;
  /** Slots being clustered: everything with at least one synapse. */
  readonly points: Int32Array;

  private readonly prints: Fingerprints;
  private readonly centroids: Float32Array;
  private readonly assignment: Int32Array;
  private readonly affinity: Float32Array;
  private readonly sums: Float32Array;
  private readonly counts: Uint32Array;

  private iteration = 0;
  private moved = 0;
  private settled = false;
  private finished = false;
  private cost = 0;
  private elapsed = 0;

  constructor(prints: Fingerprints, k: number, options: ClusterOptions = {}) {
    this.prints = prints;
    this.seed = options.seed ?? DEFAULT_CLUSTER_SEED;
    this.maxIterations = Math.max(1, Math.floor(options.maxIterations ?? DEFAULT_MAX_ITERATIONS));

    const started = performance.now();
    const members: number[] = [];
    for (let i = 0; i < prints.count; i += 1) if (prints.connected[i] === 1) members.push(i);
    this.points = Int32Array.from(members);

    const m = this.points.length;
    this.k = Math.max(1, Math.min(Math.floor(k), m));
    const dim = prints.dim;

    this.centroids = new Float32Array(this.k * dim);
    this.assignment = new Int32Array(m).fill(-1);
    this.affinity = new Float32Array(m);
    this.sums = new Float32Array(this.k * dim);
    this.counts = new Uint32Array(this.k);

    if (m === 0 || dim === 0) {
      this.finished = true;
      this.settled = true;
    } else {
      this.seedCentroids();
    }
    this.elapsed = performance.now() - started;
  }

  /** Iterations run so far, and whether the run is over. */
  get progress(): { iterations: number; moved: number; done: boolean } {
    return { iterations: this.iteration, moved: this.moved, done: this.finished };
  }

  get done(): boolean {
    return this.finished;
  }

  private dot(row: number, centroid: number): number {
    const { data, dim } = this.prints;
    let sum = 0;
    for (let d = 0; d < dim; d += 1) sum += data[row + d] * this.centroids[centroid + d];
    return sum;
  }

  private copyPointInto(point: number, centroid: number): void {
    const { data, dim } = this.prints;
    const row = this.points[point] * dim;
    for (let d = 0; d < dim; d += 1) this.centroids[centroid + d] = data[row + d];
  }

  /**
   * k-means++ seeding. Each centre after the first is drawn with probability
   * proportional to the squared distance to the nearest centre already chosen,
   * which is what stops Lloyd from converging onto a split of one dense lobe
   * while leaving a whole population unrepresented.
   */
  private seedCentroids(): void {
    const rng = new Rng(this.seed);
    const { dim } = this.prints;
    const m = this.points.length;
    const dist2 = new Float32Array(m).fill(Infinity);

    this.copyPointInto(rng.int(m), 0);

    for (let c = 1; c < this.k; c += 1) {
      const previous = (c - 1) * dim;
      let total = 0;
      for (let p = 0; p < m; p += 1) {
        const d2 = Math.max(0, 2 * (1 - this.dot(this.points[p] * dim, previous)));
        if (d2 < dist2[p]) dist2[p] = d2;
        total += dist2[p];
      }
      // Every remaining point sits exactly on a chosen centre; there is no
      // spread left to sample from, so fall back to a uniform draw.
      if (!(total > 0)) {
        this.copyPointInto(rng.int(m), c * dim);
        continue;
      }
      let target = rng.next() * total;
      let pick = m - 1;
      for (let p = 0; p < m; p += 1) {
        target -= dist2[p];
        if (target <= 0) {
          pick = p;
          break;
        }
      }
      this.copyPointInto(pick, c * dim);
    }
  }

  /** Run one assignment-and-update pass. Returns true once the run is over. */
  step(): boolean {
    if (this.finished) return true;
    const started = performance.now();
    const { data, dim } = this.prints;
    const m = this.points.length;

    let moved = 0;
    let cost = 0;
    for (let p = 0; p < m; p += 1) {
      const row = this.points[p] * dim;
      let best = 0;
      let bestDot = -Infinity;
      for (let c = 0; c < this.k; c += 1) {
        const centroid = c * dim;
        let sum = 0;
        for (let d = 0; d < dim; d += 1) sum += data[row + d] * this.centroids[centroid + d];
        if (sum > bestDot) {
          bestDot = sum;
          best = c;
        }
      }
      if (this.assignment[p] !== best) {
        this.assignment[p] = best;
        moved += 1;
      }
      this.affinity[p] = bestDot;
      cost += 1 - bestDot;
    }

    this.moved = moved;
    this.cost = cost;
    this.iteration += 1;

    if (moved === 0) {
      this.settled = true;
      this.finished = true;
    } else if (this.iteration >= this.maxIterations) {
      this.finished = true;
    } else {
      this.updateCentroids();
    }

    this.elapsed += performance.now() - started;
    return this.finished;
  }

  /**
   * Move each centroid to the normalised mean of its members. A cluster that
   * lost every member is re-seeded onto the worst-explained cell rather than
   * left pointing at nothing, which is what keeps `k` clusters meaning `k`.
   */
  private updateCentroids(): void {
    const { data, dim } = this.prints;
    const m = this.points.length;
    this.sums.fill(0);
    this.counts.fill(0);

    for (let p = 0; p < m; p += 1) {
      const row = this.points[p] * dim;
      const target = this.assignment[p] * dim;
      for (let d = 0; d < dim; d += 1) this.sums[target + d] += data[row + d];
      this.counts[this.assignment[p]] += 1;
    }

    for (let c = 0; c < this.k; c += 1) {
      const centroid = c * dim;
      if (this.counts[c] === 0) {
        let worst = 0;
        let worstDot = Infinity;
        for (let p = 0; p < m; p += 1) {
          if (this.affinity[p] < worstDot) {
            worstDot = this.affinity[p];
            worst = p;
          }
        }
        this.copyPointInto(worst, centroid);
        // Claiming it stops a second empty cluster from seizing the same cell.
        this.affinity[worst] = Infinity;
        continue;
      }
      let norm = 0;
      for (let d = 0; d < dim; d += 1) {
        const value = this.sums[centroid + d];
        norm += value * value;
      }
      norm = Math.sqrt(norm);
      if (norm === 0) continue;
      for (let d = 0; d < dim; d += 1) this.centroids[centroid + d] = this.sums[centroid + d] / norm;
    }
  }

  /** Assemble the result. Safe to call before the run finishes. */
  result(): ClusterRun {
    const started = performance.now();
    const { count, groups, groupOf, connected } = this.prints;
    const g = groups.length;
    const m = this.points.length;

    const assignment = new Int32Array(count).fill(-1);
    const sizes = new Uint32Array(this.k);
    const contingency = new Uint32Array(this.k * Math.max(1, g));
    for (let p = 0; p < m; p += 1) {
      const slot = this.points[p];
      const cluster = this.assignment[p];
      if (cluster < 0) continue;
      assignment[slot] = cluster;
      sizes[cluster] += 1;
      if (g > 0) contingency[cluster * g + groupOf[slot]] += 1;
    }

    const dominant = new Int32Array(this.k).fill(-1);
    const dominantShare = new Float32Array(this.k);
    let matched = 0;
    let occupied = 0;
    for (let c = 0; c < this.k; c += 1) {
      if (sizes[c] === 0) continue;
      occupied += 1;
      let best = -1;
      let bestCount = 0;
      for (let index = 0; index < g; index += 1) {
        const value = contingency[c * g + index];
        if (value > bestCount) {
          bestCount = value;
          best = index;
        }
      }
      dominant[c] = best;
      dominantShare[c] = bestCount / sizes[c];
      matched += bestCount;
    }

    let excluded = 0;
    for (let i = 0; i < count; i += 1) if (connected[i] === 0) excluded += 1;

    return {
      k: this.k,
      seed: this.seed,
      assignment,
      sizes,
      occupied,
      iterations: this.iteration,
      maxIterations: this.maxIterations,
      converged: this.settled,
      unsettled: this.moved,
      inertia: this.cost,
      members: m,
      excluded,
      contingency,
      dominant,
      dominantShare,
      purity: m > 0 ? matched / m : Number.NaN,
      nmi: normalisedMutualInformation(contingency, sizes, g, m),
      computeMs: this.elapsed + (performance.now() - started),
    };
  }
}

/**
 * Agreement between the clustering and the populations the user assigned, on a
 * scale where 1 means each determines the other exactly and 0 means knowing one
 * tells you nothing about the other. Undefined — reported as NaN — when either
 * side has only one occupied category, because a partition into one part carries
 * no information for the other to share.
 */
function normalisedMutualInformation(
  contingency: Uint32Array,
  sizes: Uint32Array,
  g: number,
  total: number,
): number {
  if (g === 0 || total === 0) return Number.NaN;
  const k = sizes.length;

  const groupTotals = new Float64Array(g);
  let occupiedClusters = 0;
  for (let c = 0; c < k; c += 1) {
    if (sizes[c] > 0) occupiedClusters += 1;
    for (let index = 0; index < g; index += 1) groupTotals[index] += contingency[c * g + index];
  }
  let occupiedGroups = 0;
  for (let index = 0; index < g; index += 1) if (groupTotals[index] > 0) occupiedGroups += 1;
  if (occupiedClusters < 2 || occupiedGroups < 2) return Number.NaN;

  let hCluster = 0;
  for (let c = 0; c < k; c += 1) {
    if (sizes[c] === 0) continue;
    const p = sizes[c] / total;
    hCluster -= p * Math.log(p);
  }
  let hGroup = 0;
  for (let index = 0; index < g; index += 1) {
    if (groupTotals[index] === 0) continue;
    const p = groupTotals[index] / total;
    hGroup -= p * Math.log(p);
  }

  let mutual = 0;
  for (let c = 0; c < k; c += 1) {
    if (sizes[c] === 0) continue;
    for (let index = 0; index < g; index += 1) {
      const joint = contingency[c * g + index];
      if (joint === 0) continue;
      const pJoint = joint / total;
      mutual += pJoint * Math.log(pJoint / ((sizes[c] / total) * (groupTotals[index] / total)));
    }
  }

  const denominator = hCluster + hGroup;
  if (denominator <= 0) return Number.NaN;
  return Math.max(0, Math.min(1, (2 * mutual) / denominator));
}

/**
 * Cluster every connected cell by its fingerprint and report the result.
 *
 * Synchronous and blocking; drive `FingerprintClustering` directly when the run
 * has to stay interruptible.
 */
export function clusterByFingerprint(
  prints: Fingerprints,
  k: number,
  options: ClusterOptions = {},
): ClusterRun {
  const run = new FingerprintClustering(prints, k, options);
  let done = run.done;
  while (!done) done = run.step();
  return run.result();
}
