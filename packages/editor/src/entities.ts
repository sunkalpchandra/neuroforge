/**
 * Factories for the document's leaf records, and the resting-drive table that
 * decides what a freshly created neuron actually does.
 *
 * Everything here is pure and allocation-local: no store access, no document
 * mutation. The store and the population builder share these so a neuron placed
 * by hand and a neuron produced by a population spec are indistinguishable.
 */

import type {
  Morphology,
  MorphologyArchetype,
  Neuron,
  NeuronId,
  NeuronModelKind,
  NeuronParams,
  NeuronPolarity,
  ReceptorKind,
  Synapse,
  Vec3,
} from '@neuroforge/shared';
import {
  DEFAULT_PLASTICITY,
  DEFAULT_STP,
  RECEPTOR_DEFAULTS,
  archetypeForPolarity,
  defaultMorphology,
  defaultParams,
  newSynapseId,
} from '@neuroforge/shared';

/**
 * Resting drive per membrane model: a sub-rheobase bias current paired with a
 * noise amplitude.
 *
 * Both numbers come from the same construction. The bias is set so the mean
 * input alone leaves the neuron roughly 8 mV below the point where the model's
 * fast subsystem loses its stable fixed point, and the noise is set so the
 * membrane fluctuates with a standard deviation near 3.5 mV. (For an input
 * modelled as white noise of amplitude `s` pA, the integrator injects
 * `s / sqrt(dt)` per step, which gives the membrane a stationary standard
 * deviation of `s * sqrt(tau_m / 2) / cm`.)
 *
 * That places every neuron a bit over two standard deviations below threshold:
 * firing is sparse, irregular and continuous, and — the part that matters — it
 * cannot run away, because the *mean* drive never reaches rheobase. All activity
 * beyond that has to be paid for by synaptic input.
 */
const MODEL_DRIVE: Record<NeuronModelKind, { readonly bias: number; readonly noise: number }> = {
  // Rheobase 100 pA (I_model = 4 at iScale 0.04); tau_m ~ 3 ms near rest.
  izhikevich: { bias: 55, noise: 78 },
  // Rheobase gL*(vThresh-eL) = 200 pA; tau_m = 20 ms.
  lif: { bias: 116, noise: 210 },
  // Rheobase ~ gL*(vT-eL) = 606 pA; tau_m = 9.4 ms.
  adex: { bias: 354, noise: 440 },
  // Rheobase ~ 620 pA for the classic channel densities at cm = 100 pF.
  'hodgkin-huxley': { bias: 470, noise: 260 },
  // Rheobase ~ 88 pA for the Rinzel parameterisation at cm = 20 pF.
  'morris-lecar': { bias: 58, noise: 30 },
};

/**
 * Interneurons sit closer to threshold and see a little more noise, which is
 * what makes them track the excitatory population rather than lead it.
 */
const INHIBITORY_BIAS_SCALE = 1.13;
const INHIBITORY_NOISE_SCALE = 1.05;

/** Bias and noise currents (pA) for a new neuron of this model and polarity. */
export function defaultDrive(
  model: NeuronModelKind,
  polarity: NeuronPolarity,
): { bias: number; noise: number } {
  const base = MODEL_DRIVE[model];
  if (polarity === 'inhibitory') {
    return {
      bias: round(base.bias * INHIBITORY_BIAS_SCALE, 2),
      noise: round(base.noise * INHIBITORY_NOISE_SCALE, 2),
    };
  }
  return { bias: base.bias, noise: base.noise };
}

/** Receptor a source of this polarity releases onto its targets. */
export function receptorForPolarity(polarity: NeuronPolarity): ReceptorKind {
  return polarity === 'inhibitory' ? 'gabaa' : 'ampa';
}

/**
 * Peak conductance (nS) that produces a postsynaptic response of roughly 4 mV in
 * the default models. GABA-A needs more of it than AMPA because its reversal
 * potential sits only a few millivolts below rest, so its driving force near
 * rest is small — most of its authority appears exactly when the target is
 * depolarised, which is what makes it a brake rather than a clamp.
 */
const DEFAULT_WEIGHT: Record<ReceptorKind, number> = {
  ampa: 0.85,
  nmda: 0.35,
  gabaa: 1.8,
  gabab: 0.9,
  gap: 0.4,
};

function defaultWeight(receptor: ReceptorKind): number {
  return DEFAULT_WEIGHT[receptor];
}

/** Shortest delay the delay queue can represent without rounding to zero. */
export const MIN_DELAY_MS = 0.1;

/** Spline sag for an axon spanning `distance` world units. */
export function arcFor(distance: number): number {
  return clamp(distance * 0.14, 0.4, 6);
}

export function distanceBetween(a: Vec3, b: Vec3): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function round(value: number, digits: number): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

/** FNV-1a over a string, used to derive reproducible seeds from names and ids. */
export function stringSeed(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** A fresh, independently mutable copy of a parameter block. */
export function cloneParams(params: NeuronParams): NeuronParams {
  return { ...params } as NeuronParams;
}

/**
 * Overlay a partial parameter patch onto a model's defaults. Keys that do not
 * belong to the model are ignored rather than smuggled in, so a patch written
 * for one model cannot corrupt another.
 */
export function mergeParams(model: NeuronModelKind, patch?: Partial<NeuronParams>): NeuronParams {
  const merged = defaultParams(model) as unknown as Record<string, unknown>;
  if (patch !== undefined) {
    const source = patch as unknown as Record<string, unknown>;
    for (const key of Object.keys(source)) {
      if (key === 'kind') continue;
      const value = source[key];
      if (value === undefined || !(key in merged)) continue;
      if (typeof value === 'number' && Number.isFinite(value)) merged[key] = value;
    }
  }
  return merged as unknown as NeuronParams;
}

/**
 * Per-neuron morphology with a deterministic seed and a little scale variation,
 * so a population reads as a population of individuals rather than a stamp.
 */
export function morphologyFor(
  archetype: MorphologyArchetype,
  seed: number,
  scaleJitter = 0,
): Morphology {
  const morphology = defaultMorphology(archetype, seed >>> 0);
  if (scaleJitter !== 0) {
    morphology.scale = round(morphology.scale * (1 + scaleJitter), 4);
  }
  return morphology;
}

export function makeNeuron(id: NeuronId, partial: Partial<Neuron> = {}): Neuron {
  const params = partial.params !== undefined ? cloneParams(partial.params) : defaultParams('izhikevich');
  const polarity = partial.polarity ?? 'excitatory';
  const archetype = partial.morphology?.archetype ?? archetypeForPolarity(polarity);
  const drive = defaultDrive(params.kind, polarity);
  const position = partial.position;

  return {
    id,
    label: partial.label ?? '',
    position:
      position === undefined ? { x: 0, y: 0, z: 0 } : { x: position.x, y: position.y, z: position.z },
    params,
    polarity,
    morphology:
      partial.morphology === undefined
        ? morphologyFor(archetype, stringSeed(id))
        : { ...partial.morphology },
    population: partial.population ?? null,
    bias: partial.bias ?? drive.bias,
    noise: partial.noise ?? drive.noise,
    enabled: partial.enabled ?? true,
  };
}

export function makeSynapse(
  source: NeuronId,
  target: NeuronId,
  sourcePolarity: NeuronPolarity,
  partial: Partial<Synapse> = {},
): Synapse {
  const receptor = partial.receptor ?? receptorForPolarity(sourcePolarity);
  return {
    id: partial.id ?? newSynapseId(),
    source,
    target,
    receptor,
    weight: partial.weight ?? defaultWeight(receptor),
    delay: Math.max(MIN_DELAY_MS, partial.delay ?? 1.2),
    kinetics: partial.kinetics === undefined ? { ...RECEPTOR_DEFAULTS[receptor] } : { ...partial.kinetics },
    plasticity:
      partial.plasticity === undefined ? { ...DEFAULT_PLASTICITY } : { ...partial.plasticity },
    stp: partial.stp === undefined ? { ...DEFAULT_STP } : { ...partial.stp },
    releaseProbability: partial.releaseProbability ?? 1,
    arc: partial.arc ?? 1.5,
    enabled: partial.enabled ?? true,
  };
}

/**
 * Projection ids are plain strings in the schema, so they are minted here rather
 * than by `@neuroforge/shared`. Same shape as the branded minters: a base-36
 * time prefix so ids sort chronologically, plus six characters of entropy.
 */
export function newProjectionId(): string {
  const alphabet = '0123456789abcdefghijklmnopqrstuvwxyz';
  const bytes = new Uint8Array(6);
  if (typeof globalThis.crypto !== 'undefined' && globalThis.crypto.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  }
  let suffix = '';
  for (let i = 0; i < bytes.length; i += 1) suffix += alphabet[bytes[i] % alphabet.length];
  return `pj_${Date.now().toString(36)}${suffix}`;
}
