import type { NeuronId, SynapseId } from './ids';

/**
 * Synaptic receptor kinetics. Each maps to a conductance waveform with its own
 * time constants and reversal potential.
 */
export type ReceptorKind = 'ampa' | 'nmda' | 'gabaa' | 'gabab' | 'gap';

export const RECEPTOR_KINDS: readonly ReceptorKind[] = [
  'ampa',
  'nmda',
  'gabaa',
  'gabab',
  'gap',
] as const;

export const RECEPTOR_LABELS: Record<ReceptorKind, string> = {
  ampa: 'AMPA',
  nmda: 'NMDA',
  gabaa: 'GABA-A',
  gabab: 'GABA-B',
  gap: 'Gap junction',
};

/** Default kinetics per receptor: rise/decay in ms, reversal in mV. */
export interface ReceptorKinetics {
  tauRise: number;
  tauDecay: number;
  eRev: number;
  /** NMDA-style voltage dependence via Mg block; 0 disables it. */
  mgBlock: number;
}

export const RECEPTOR_DEFAULTS: Record<ReceptorKind, ReceptorKinetics> = {
  ampa: { tauRise: 0.4, tauDecay: 2.0, eRev: 0, mgBlock: 0 },
  nmda: { tauRise: 2.0, tauDecay: 100.0, eRev: 0, mgBlock: 1.0 },
  gabaa: { tauRise: 0.5, tauDecay: 6.0, eRev: -70, mgBlock: 0 },
  gabab: { tauRise: 10.0, tauDecay: 150.0, eRev: -90, mgBlock: 0 },
  gap: { tauRise: 0.05, tauDecay: 0.05, eRev: 0, mgBlock: 0 },
};

/** Plasticity rule applied to this synapse's weight. */
export type PlasticityKind = 'static' | 'stdp' | 'triplet-stdp' | 'hebbian' | 'oja';

export const PLASTICITY_KINDS: readonly PlasticityKind[] = [
  'static',
  'stdp',
  'triplet-stdp',
  'hebbian',
  'oja',
] as const;

export const PLASTICITY_LABELS: Record<PlasticityKind, string> = {
  static: 'Static',
  stdp: 'Pair STDP',
  'triplet-stdp': 'Triplet STDP',
  hebbian: 'Hebbian',
  oja: "Oja's rule",
};

/**
 * Spike-timing-dependent plasticity configuration. The triplet rule uses the
 * slow trace constants; the pair rule ignores them.
 */
export interface PlasticityConfig {
  kind: PlasticityKind;
  /** Potentiation amplitude. */
  aPlus: number;
  /** Depression amplitude. */
  aMinus: number;
  /** Pre-before-post time constant (ms). */
  tauPlus: number;
  /** Post-before-pre time constant (ms). */
  tauMinus: number;
  /** Slow presynaptic trace constant for the triplet rule (ms). */
  tauX: number;
  /** Slow postsynaptic trace constant for the triplet rule (ms). */
  tauY: number;
  /** Hard lower bound on weight. */
  wMin: number;
  /** Hard upper bound on weight. */
  wMax: number;
  /** Global learning-rate multiplier. */
  learningRate: number;
}

export const DEFAULT_PLASTICITY: PlasticityConfig = {
  kind: 'static',
  aPlus: 0.008,
  aMinus: 0.009,
  tauPlus: 16.8,
  tauMinus: 33.7,
  tauX: 101,
  tauY: 125,
  wMin: 0,
  wMax: 4,
  learningRate: 1,
};

/** Short-term plasticity (Tsodyks–Markram) — facilitation and depression. */
export interface ShortTermPlasticity {
  enabled: boolean;
  /** Utilisation of synaptic efficacy (0..1). */
  u: number;
  /** Recovery time constant for depression (ms). */
  tauRec: number;
  /** Facilitation time constant (ms). */
  tauFacil: number;
}

export const DEFAULT_STP: ShortTermPlasticity = {
  enabled: false,
  u: 0.5,
  tauRec: 800,
  tauFacil: 0,
};

/** A directed connection between two neurons. */
export interface Synapse {
  readonly id: SynapseId;
  readonly source: NeuronId;
  readonly target: NeuronId;
  receptor: ReceptorKind;
  /** Peak conductance (nS). Sign is carried by the receptor reversal, not this. */
  weight: number;
  /** Axonal conduction delay (ms). */
  delay: number;
  kinetics: ReceptorKinetics;
  plasticity: PlasticityConfig;
  stp: ShortTermPlasticity;
  /** Release probability per presynaptic spike (0..1). */
  releaseProbability: number;
  /** Spline sag applied when rendering the axon, in world units. */
  arc: number;
  enabled: boolean;
}

/**
 * Connectivity descriptor used when the AI builder or the population tool wires
 * two groups together. Expanded into concrete Synapse records at build time.
 */
export type ConnectivityRule =
  | { kind: 'all-to-all'; selfConnections: boolean }
  | { kind: 'random'; probability: number; seed: number; selfConnections: boolean }
  | { kind: 'one-to-one' }
  | { kind: 'gaussian'; sigma: number; maxProbability: number; seed: number }
  | { kind: 'distance-threshold'; radius: number; probability: number; seed: number }
  | { kind: 'fixed-in-degree'; degree: number; seed: number }
  | { kind: 'fixed-out-degree'; degree: number; seed: number };
