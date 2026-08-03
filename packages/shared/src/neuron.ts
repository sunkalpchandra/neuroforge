import type { NeuronId, PopulationId } from './ids';
import type { Vec3 } from './geometry';

/**
 * The five supported membrane models. Each is a distinct ODE system; the
 * discriminated union below lets a neuron switch model at runtime while
 * keeping its parameters strongly typed.
 *
 * All voltages are in millivolts (mV), all times in milliseconds (ms), all
 * currents in picoamps (pA), capacitance in picofarads (pF), conductance in
 * nanosiemens (nS). This unit system is consistent across the simulation core,
 * the WGSL compute shaders, and every exporter.
 */
export type NeuronModelKind =
  | 'lif'
  | 'izhikevich'
  | 'hodgkin-huxley'
  | 'adex'
  | 'morris-lecar';

export const NEURON_MODEL_KINDS: readonly NeuronModelKind[] = [
  'lif',
  'izhikevich',
  'hodgkin-huxley',
  'adex',
  'morris-lecar',
] as const;

export const NEURON_MODEL_LABELS: Record<NeuronModelKind, string> = {
  lif: 'Leaky Integrate-and-Fire',
  izhikevich: 'Izhikevich',
  'hodgkin-huxley': 'Hodgkin–Huxley',
  adex: 'Adaptive Exponential IF',
  'morris-lecar': 'Morris–Lecar',
};

/** Number of state variables each model integrates. Drives GPU buffer strides. */
export const MODEL_STATE_WIDTH: Record<NeuronModelKind, number> = {
  lif: 2, // v, w(unused placeholder for stride uniformity)
  izhikevich: 2, // v, u
  'hodgkin-huxley': 4, // v, m, h, n
  adex: 2, // v, w
  'morris-lecar': 2, // v, w
};

/** Leaky integrate-and-fire. */
export interface LifParams {
  readonly kind: 'lif';
  /** Membrane capacitance (pF). */
  cm: number;
  /** Leak conductance (nS). */
  gL: number;
  /** Leak reversal / resting potential (mV). */
  eL: number;
  /** Spike threshold (mV). */
  vThresh: number;
  /** Post-spike reset potential (mV). */
  vReset: number;
  /** Absolute refractory period (ms). */
  tRefract: number;
}

/** Izhikevich two-variable model. dv = 0.04v^2 + 5v + 140 - u + I. */
export interface IzhikevichParams {
  readonly kind: 'izhikevich';
  /** Recovery time scale. */
  a: number;
  /** Recovery sensitivity to v. */
  b: number;
  /** Post-spike reset of v (mV). */
  c: number;
  /** Post-spike increment of u. */
  d: number;
  /** Peak / cutoff voltage (mV). */
  vPeak: number;
  /** Input current scale (pA -> model units). */
  iScale: number;
}

/** Hodgkin–Huxley with classic squid-axon channel kinetics. */
export interface HodgkinHuxleyParams {
  readonly kind: 'hodgkin-huxley';
  /** Membrane capacitance (pF). */
  cm: number;
  /** Max sodium conductance (nS). */
  gNa: number;
  /** Max potassium conductance (nS). */
  gK: number;
  /** Leak conductance (nS). */
  gL: number;
  /** Sodium reversal (mV). */
  eNa: number;
  /** Potassium reversal (mV). */
  eK: number;
  /** Leak reversal (mV). */
  eL: number;
  /** Voltage at which a spike is registered on the rising edge (mV). */
  vDetect: number;
  /** Temperature coefficient applied to all rate constants. */
  q10: number;
}

/** Adaptive exponential integrate-and-fire (Brette & Gerstner). */
export interface AdExParams {
  readonly kind: 'adex';
  cm: number;
  gL: number;
  eL: number;
  /** Sharpness of the exponential spike upstroke (mV). */
  deltaT: number;
  /** Effective threshold where the exponential takes over (mV). */
  vT: number;
  /** Numerical cutoff voltage (mV). */
  vPeak: number;
  vReset: number;
  /** Subthreshold adaptation conductance (nS). */
  a: number;
  /** Spike-triggered adaptation increment (pA). */
  b: number;
  /** Adaptation time constant (ms). */
  tauW: number;
  tRefract: number;
}

/** Morris–Lecar Ca/K oscillator. */
export interface MorrisLecarParams {
  readonly kind: 'morris-lecar';
  cm: number;
  gCa: number;
  gK: number;
  gL: number;
  eCa: number;
  eK: number;
  eL: number;
  /** Half-activation voltage for Ca (mV). */
  v1: number;
  /** Ca activation slope (mV). */
  v2: number;
  /** Half-activation voltage for K (mV). */
  v3: number;
  /** K activation slope (mV). */
  v4: number;
  /** Reference rate for w kinetics (1/ms). */
  phi: number;
  vDetect: number;
}

export type NeuronParams =
  | LifParams
  | IzhikevichParams
  | HodgkinHuxleyParams
  | AdExParams
  | MorrisLecarParams;

/** Extract the params type for a given model kind. */
export type ParamsFor<K extends NeuronModelKind> = Extract<NeuronParams, { kind: K }>;

/** Excitatory or inhibitory. Determines default synapse reversal potential. */
export type NeuronPolarity = 'excitatory' | 'inhibitory';

/**
 * Morphology descriptor. Dendrites and the axon are generated procedurally from
 * these numbers by the renderer, so a neuron is a few dozen bytes rather than a
 * mesh. Two neurons with the same seed and descriptor look identical.
 */
export interface Morphology {
  /** Deterministic seed for procedural branching. */
  seed: number;
  /** Soma radius in world units. */
  somaRadius: number;
  /** Number of primary dendritic trunks leaving the soma. */
  dendriteCount: number;
  /** Recursive branch depth per trunk. */
  dendriteDepth: number;
  /** Total dendritic extent in world units. */
  dendriteLength: number;
  /** Fraction by which each child branch shortens (0..1). */
  dendriteTaper: number;
  /** Half-angle of the branching cone in radians. */
  dendriteSpread: number;
  /** Axon length in world units. */
  axonLength: number;
  /** Number of terminal boutons at the axon tip. */
  axonTerminals: number;
  /** Global scale multiplier applied to the whole glyph. */
  scale: number;
  /** Archetype, used to pick sensible procedural defaults. */
  archetype: MorphologyArchetype;
}

export type MorphologyArchetype =
  | 'pyramidal'
  | 'basket'
  | 'granule'
  | 'purkinje'
  | 'stellate'
  | 'motor'
  | 'bipolar';

export const MORPHOLOGY_ARCHETYPES: readonly MorphologyArchetype[] = [
  'pyramidal',
  'basket',
  'granule',
  'purkinje',
  'stellate',
  'motor',
  'bipolar',
] as const;

/** A single neuron in the circuit graph. */
export interface Neuron {
  readonly id: NeuronId;
  /** Human-facing label; may be empty. */
  label: string;
  /** World-space position. */
  position: Vec3;
  params: NeuronParams;
  polarity: NeuronPolarity;
  morphology: Morphology;
  /** Owning population, if this neuron was created as part of one. */
  population: PopulationId | null;
  /** Constant injected current (pA). */
  bias: number;
  /** Per-neuron noise amplitude (pA, standard deviation of the input). */
  noise: number;
  /** Excluded from integration when false; still rendered, dimmed. */
  enabled: boolean;
}
