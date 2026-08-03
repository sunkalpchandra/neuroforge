/**
 * Turning population and projection specs into concrete neurons and synapses.
 *
 * Both builders are pure: they read a spec (and, for projections, the document
 * they will be wired into) and return records. Nothing here touches the store.
 *
 * Everything is reproducible. Positions come from the analytic layouts in
 * `@neuroforge/physics`, which are seeded; connectivity and the weight/delay
 * draws come from two independent `Rng` streams derived from the spec, so
 * re-running a spec rebuilds the same network, and changing a weight does not
 * rewire the topology.
 */

import type {
  ConnectivityRule,
  MorphologyArchetype,
  Neuron,
  NeuronId,
  NeuronModelKind,
  NeuronParams,
  NeuronPolarity,
  PlasticityKind,
  Population,
  PopulationId,
  PopulationLayout,
  ReceptorKind,
  Synapse,
  Vec3,
  Circuit,
} from '@neuroforge/shared';
import {
  DEFAULT_PLASTICITY,
  DEFAULT_STP,
  RECEPTOR_DEFAULTS,
  archetypeForPolarity,
  newNeuronId,
  newPopulationId,
  newSynapseId,
} from '@neuroforge/shared';
import { Rng, hashSeed } from '@neuroforge/math';
import {
  fibonacciSphere,
  layoutColumn,
  layoutDisc,
  layoutGrid,
  layoutSphere,
} from '@neuroforge/physics';

import {
  MIN_DELAY_MS,
  arcFor,
  cloneParams,
  defaultDrive,
  mergeParams,
  morphologyFor,
  receptorForPolarity,
  stringSeed,
} from './entities';

export interface PopulationSpec {
  name: string;
  size: number;
  polarity: NeuronPolarity;
  model: NeuronModelKind;
  params?: Partial<NeuronParams>;
  layout: PopulationLayout;
  origin?: Vec3;
  archetype?: MorphologyArchetype;
  color?: string | null;
}

export interface ProjectionSpec {
  name: string;
  source: PopulationId;
  target: PopulationId;
  rule: ConnectivityRule;
  receptor?: ReceptorKind;
  weightMean?: number;
  weightJitter?: number;
  delayMean?: number;
  delayJitter?: number;
  plasticity?: PlasticityKind;
}

/**
 * Fallbacks for the optional statistics on a `ProjectionSpec`.
 *
 * Exported because the `Projection` record the store files alongside the
 * synapses is provenance: it has to say what was actually drawn. Resolving the
 * same spec against two different sets of defaults would make the record
 * describe a network that was never built, and re-wiring from it would silently
 * produce a different one.
 */
export const DEFAULT_WEIGHT_MEAN = 1;
export const DEFAULT_WEIGHT_JITTER = 0.25;
export const DEFAULT_DELAY_MEAN = 1.5;
export const DEFAULT_DELAY_JITTER = 0.5;

/** Populations larger than this are refused rather than allowed to hang the tab. */
const MAX_POPULATION_SIZE = 200000;

const ORIGIN: Vec3 = { x: 0, y: 0, z: 0 };

/* ------------------------------------------------------------------ layout -- */

function layoutSeed(layout: PopulationLayout): number {
  return 'seed' in layout ? layout.seed : 0;
}

/**
 * Replace any coordinate a layout could not produce.
 *
 * `layoutColumn` in `@neuroforge/physics` returns NaN for almost exactly half
 * its slots. Its internal `hash01` ends on `h ^= h >>> 16`, and XOR yields a
 * *signed* int32, so the function returns a uniform value in [-0.5, 0.5) rather
 * than [0, 1); `layoutColumn` takes the square root of it to get a radius, and
 * every negative draw becomes NaN. A NaN position is not cosmetic — it poisons
 * the spatial hash, every bounding box, and every distance-based connectivity
 * rule downstream — so the slots the layout failed to place are resampled from
 * the slots it managed, jittered within a fraction of their bounding box.
 *
 * What that buys is finiteness and a plausible arrangement, and no more. It
 * cannot recover the layout's intent, because the surviving samples are drawn
 * from the same truncated distribution: a column asked for radius r comes back
 * at r*sqrt(1/2), about 71% of the requested width, and the golden-angle spiral
 * survives only in the half of the slots that were placed. Reconstructing either
 * would mean reimplementing the layout's formula here, which is the wrong place
 * for it. The real fix is one character upstream — `h = (h ^ (h >>> 16)) >>> 0`
 * — after which this function finds nothing to repair and does nothing.
 *
 * `layoutSphere` and `layoutDisc` use the same hash only for jitter, so they
 * produce finite but skewed offsets that nothing here can detect.
 */
function repairPositions(out: Float32Array, size: number, seed: number): void {
  const placed: number[] = [];
  const missing: number[] = [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  for (let i = 0; i < size; i += 1) {
    const p = i * 3;
    if (Number.isFinite(out[p]) && Number.isFinite(out[p + 1]) && Number.isFinite(out[p + 2])) {
      placed.push(i);
      if (out[p] < minX) minX = out[p];
      if (out[p] > maxX) maxX = out[p];
      if (out[p + 1] < minY) minY = out[p + 1];
      if (out[p + 1] > maxY) maxY = out[p + 1];
      if (out[p + 2] < minZ) minZ = out[p + 2];
      if (out[p + 2] > maxZ) maxZ = out[p + 2];
    } else {
      missing.push(i);
    }
  }
  if (missing.length === 0) return;

  const rng = new Rng(hashSeed(seed, 0x2eba1e));

  if (placed.length === 0) {
    const point = { x: 0, y: 0, z: 0 };
    for (const i of missing) {
      fibonacciSphere(i, size, 1, point);
      out[i * 3] = point.x;
      out[i * 3 + 1] = point.y;
      out[i * 3 + 2] = point.z;
    }
    return;
  }

  const spreadX = (maxX - minX) * 0.06;
  const spreadY = (maxY - minY) * 0.06;
  const spreadZ = (maxZ - minZ) * 0.06;

  for (const i of missing) {
    const donor = placed[rng.int(placed.length)] * 3;
    const p = i * 3;
    if (!Number.isFinite(out[p])) out[p] = out[donor] + rng.range(-spreadX, spreadX);
    if (!Number.isFinite(out[p + 1])) out[p + 1] = out[donor + 1] + rng.range(-spreadY, spreadY);
    if (!Number.isFinite(out[p + 2])) out[p + 2] = out[donor + 2] + rng.range(-spreadZ, spreadZ);
  }
}

/** Write `size` xyz triples for a layout into a flat position buffer. */
function layoutPositions(layout: PopulationLayout, size: number, seed: number): Float32Array {
  const out = new Float32Array(Math.max(1, size) * 3);
  switch (layout.kind) {
    case 'grid':
      layoutGrid(size, layout.columns, layout.rows, layout.layers, layout.spacing, out, 0);
      break;
    case 'sphere':
      layoutSphere(size, layout.radius, layout.jitter, layout.seed, out, 0);
      break;
    case 'disc':
      layoutDisc(size, layout.radius, layout.thickness, layout.seed, out, 0);
      break;
    case 'column':
      layoutColumn(size, layout.radius, layout.height, layout.seed, out, 0);
      break;
    case 'explicit': {
      const source = layout.positions;
      for (let i = 0; i < size; i += 1) {
        const point = source.length === 0 ? ORIGIN : source[i % source.length];
        out[i * 3] = point.x;
        out[i * 3 + 1] = point.y;
        out[i * 3 + 2] = point.z;
      }
      break;
    }
  }
  repairPositions(out, size, seed);
  return out;
}

/* ------------------------------------------------------------- populations -- */

/** Build neurons for a population spec without touching the store. */
export function instantiatePopulation(spec: PopulationSpec): {
  population: Population;
  neurons: Neuron[];
} {
  const size = Math.max(0, Math.min(MAX_POPULATION_SIZE, Math.floor(spec.size)));
  const polarity = spec.polarity;
  const archetype = spec.archetype ?? archetypeForPolarity(polarity);
  const params = mergeParams(spec.model, spec.params);
  const drive = defaultDrive(spec.model, polarity);
  const origin = spec.origin ?? ORIGIN;

  const baseSeed = hashSeed(stringSeed(spec.name), layoutSeed(spec.layout), size);
  const positions = layoutPositions(spec.layout, size, baseSeed);
  const variation = new Rng(hashSeed(baseSeed, 0x9e37));

  const id = newPopulationId();
  const neurons: Neuron[] = new Array<Neuron>(size);
  const members: NeuronId[] = new Array<NeuronId>(size);

  for (let i = 0; i < size; i += 1) {
    const neuronId = newNeuronId();
    members[i] = neuronId;
    neurons[i] = {
      id: neuronId,
      label: size === 1 ? spec.name : `${spec.name} ${i + 1}`,
      position: {
        x: positions[i * 3] + origin.x,
        y: positions[i * 3 + 1] + origin.y,
        z: positions[i * 3 + 2] + origin.z,
      },
      params: cloneParams(params),
      polarity,
      morphology: morphologyFor(archetype, hashSeed(baseSeed, i), variation.range(-0.12, 0.12)),
      population: id,
      bias: drive.bias,
      noise: drive.noise,
      enabled: true,
    };
  }

  const population: Population = {
    id,
    name: spec.name,
    size,
    polarity,
    params: cloneParams(params),
    morphology: morphologyFor(archetype, baseSeed),
    layout: spec.layout,
    origin: { x: origin.x, y: origin.y, z: origin.z },
    color: spec.color ?? null,
    members,
    collapsed: false,
  };

  return { population, neurons };
}

/* -------------------------------------------------------------- projection -- */

/**
 * Draw `want` distinct entries from `pool` and write them into `out`, skipping
 * `exclude`.
 *
 * A partial Fisher-Yates shuffle: each step swaps one unvisited entry into
 * position and never revisits it, so the same partner cannot be drawn twice. The
 * swaps are undone before returning, which leaves `pool` reusable for the next
 * neuron without an O(n) rebuild per call.
 */
function sampleWithoutReplacement(
  pool: number[],
  want: number,
  exclude: number,
  rng: Rng,
  out: number[],
): void {
  out.length = 0;
  const n = pool.length;
  const swaps: number[] = [];
  let cursor = 0;
  while (out.length < want && cursor < n) {
    const pick = cursor + rng.int(n - cursor);
    if (pick !== cursor) {
      const held = pool[cursor];
      pool[cursor] = pool[pick];
      pool[pick] = held;
      swaps.push(cursor, pick);
    }
    if (pool[cursor] !== exclude) out.push(pool[cursor]);
    cursor += 1;
  }
  for (let i = swaps.length - 2; i >= 0; i -= 2) {
    const a = swaps[i];
    const b = swaps[i + 1];
    const held = pool[a];
    pool[a] = pool[b];
    pool[b] = held;
  }
}

function squaredDistance(a: Vec3, b: Vec3): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

function membersOf(
  circuit: Circuit,
  population: PopulationId,
  byId: ReadonlyMap<string, Neuron>,
): Neuron[] {
  const record = circuit.populations.find((candidate) => candidate.id === population);
  if (record === undefined) return [];
  const out: Neuron[] = [];
  for (const memberId of record.members) {
    const neuron = byId.get(memberId);
    if (neuron !== undefined) out.push(neuron);
  }
  return out;
}

/**
 * Expand a connectivity rule into the ordered list of (source index, target
 * index) pairs it describes.
 *
 * Self-connections are rejected everywhere except where the rule explicitly
 * opts into them, which matters only when a population projects onto itself.
 */
function expandRule(
  rule: ConnectivityRule,
  sources: readonly Neuron[],
  targets: readonly Neuron[],
  rng: Rng,
  emit: (sourceIndex: number, targetIndex: number) => void,
): void {
  const sameNeuron = (s: number, t: number): boolean => sources[s].id === targets[t].id;

  switch (rule.kind) {
    case 'all-to-all': {
      for (let s = 0; s < sources.length; s += 1) {
        for (let t = 0; t < targets.length; t += 1) {
          if (!rule.selfConnections && sameNeuron(s, t)) continue;
          emit(s, t);
        }
      }
      return;
    }

    case 'random': {
      const probability = rule.probability;
      if (probability <= 0) return;
      for (let s = 0; s < sources.length; s += 1) {
        for (let t = 0; t < targets.length; t += 1) {
          if (!rule.selfConnections && sameNeuron(s, t)) continue;
          if (rng.next() < probability) emit(s, t);
        }
      }
      return;
    }

    case 'one-to-one': {
      const count = Math.min(sources.length, targets.length);
      for (let i = 0; i < count; i += 1) {
        if (sameNeuron(i, i)) continue;
        emit(i, i);
      }
      return;
    }

    case 'gaussian': {
      const sigma = Math.abs(rule.sigma);
      if (sigma <= 0) return;
      const denominator = 2 * sigma * sigma;
      const peak = rule.maxProbability;
      for (let s = 0; s < sources.length; s += 1) {
        for (let t = 0; t < targets.length; t += 1) {
          if (sameNeuron(s, t)) continue;
          const d2 = squaredDistance(sources[s].position, targets[t].position);
          const probability = peak * Math.exp(-d2 / denominator);
          if (rng.next() < probability) emit(s, t);
        }
      }
      return;
    }

    case 'distance-threshold': {
      const radius2 = rule.radius * rule.radius;
      for (let s = 0; s < sources.length; s += 1) {
        for (let t = 0; t < targets.length; t += 1) {
          if (sameNeuron(s, t)) continue;
          if (squaredDistance(sources[s].position, targets[t].position) > radius2) continue;
          if (rng.next() < rule.probability) emit(s, t);
        }
      }
      return;
    }

    case 'fixed-in-degree': {
      const degree = Math.max(0, Math.floor(rule.degree));
      if (degree === 0 || sources.length === 0) return;
      const pool: number[] = [];
      for (let s = 0; s < sources.length; s += 1) pool.push(s);
      const sourceIndexById = new Map<string, number>();
      for (let s = 0; s < sources.length; s += 1) sourceIndexById.set(sources[s].id, s);
      const picked: number[] = [];
      for (let t = 0; t < targets.length; t += 1) {
        const exclude = sourceIndexById.get(targets[t].id) ?? -1;
        sampleWithoutReplacement(pool, degree, exclude, rng, picked);
        for (let i = 0; i < picked.length; i += 1) emit(picked[i], t);
      }
      return;
    }

    case 'fixed-out-degree': {
      const degree = Math.max(0, Math.floor(rule.degree));
      if (degree === 0 || targets.length === 0) return;
      const pool: number[] = [];
      for (let t = 0; t < targets.length; t += 1) pool.push(t);
      const targetIndexById = new Map<string, number>();
      for (let t = 0; t < targets.length; t += 1) targetIndexById.set(targets[t].id, t);
      const picked: number[] = [];
      for (let s = 0; s < sources.length; s += 1) {
        const exclude = targetIndexById.get(sources[s].id) ?? -1;
        sampleWithoutReplacement(pool, degree, exclude, rng, picked);
        for (let i = 0; i < picked.length; i += 1) emit(s, picked[i]);
      }
      return;
    }
  }
}

/** Expand a projection spec into concrete synapses against a document. */
export function instantiateProjection(spec: ProjectionSpec, circuit: Circuit): Synapse[] {
  const byId = new Map<string, Neuron>();
  for (const neuron of circuit.neurons) byId.set(neuron.id, neuron);

  const sources = membersOf(circuit, spec.source, byId);
  const targets = membersOf(circuit, spec.target, byId);
  if (sources.length === 0 || targets.length === 0) return [];

  const ruleSeed = 'seed' in spec.rule ? spec.rule.seed : 0;
  const base = hashSeed(stringSeed(spec.name), ruleSeed, sources.length, targets.length);
  // Two streams: topology stays fixed when only the weight statistics change.
  const topology = new Rng(hashSeed(base, 1));
  const values = new Rng(hashSeed(base, 2));

  const weightMean = spec.weightMean ?? DEFAULT_WEIGHT_MEAN;
  const weightJitter = Math.abs(spec.weightJitter ?? DEFAULT_WEIGHT_JITTER);
  const delayMean = spec.delayMean ?? DEFAULT_DELAY_MEAN;
  const delayJitter = Math.abs(spec.delayJitter ?? DEFAULT_DELAY_JITTER);
  const plasticityKind = spec.plasticity ?? 'static';

  const synapses: Synapse[] = [];
  expandRule(spec.rule, sources, targets, topology, (sourceIndex, targetIndex) => {
    const source = sources[sourceIndex];
    const target = targets[targetIndex];
    const receptor = spec.receptor ?? receptorForPolarity(source.polarity);
    const distance = Math.sqrt(squaredDistance(source.position, target.position));

    synapses.push({
      id: newSynapseId(),
      source: source.id,
      target: target.id,
      receptor,
      weight: Math.max(0, weightMean + values.range(-weightJitter, weightJitter)),
      delay: Math.max(MIN_DELAY_MS, delayMean + values.range(-delayJitter, delayJitter)),
      kinetics: { ...RECEPTOR_DEFAULTS[receptor] },
      plasticity: { ...DEFAULT_PLASTICITY, kind: plasticityKind },
      stp: { ...DEFAULT_STP },
      releaseProbability: 1,
      arc: arcFor(distance),
      enabled: true,
    });
  });

  return synapses;
}
