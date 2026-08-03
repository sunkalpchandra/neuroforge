import type { NeuronModelKind } from './neuron';
import type { ReceptorKind } from './synapse';
import type { PlasticityKind } from './synapse';

/**
 * Structure-of-arrays storage for the live simulation.
 *
 * Every field is a flat typed array indexed by a dense integer slot, not by
 * NeuronId. The editor owns the id -> slot mapping; the integrator and the
 * renderer only ever see slots. This layout is what makes the whole thing fast:
 * the same ArrayBuffers are handed to WASM as linear memory views, uploaded to
 * WebGPU storage buffers without a copy, and read by InstancedMesh attribute
 * updaters. Nothing is boxed, nothing is garbage collected per frame.
 *
 * Growth is amortised: capacity doubles and the arrays are reallocated, which is
 * the only moment references become stale. Consumers must re-read the arrays
 * after any structural edit rather than caching them across frames.
 */

/** Numeric encoding of NeuronModelKind stored in the model column. */
export const MODEL_CODE: Record<NeuronModelKind, number> = {
  lif: 0,
  izhikevich: 1,
  'hodgkin-huxley': 2,
  adex: 3,
  'morris-lecar': 4,
};

export const MODEL_FROM_CODE: readonly NeuronModelKind[] = [
  'lif',
  'izhikevich',
  'hodgkin-huxley',
  'adex',
  'morris-lecar',
];

export const RECEPTOR_CODE: Record<ReceptorKind, number> = {
  ampa: 0,
  nmda: 1,
  gabaa: 2,
  gabab: 3,
  gap: 4,
};

export const RECEPTOR_FROM_CODE: readonly ReceptorKind[] = [
  'ampa',
  'nmda',
  'gabaa',
  'gabab',
  'gap',
];

export const PLASTICITY_CODE: Record<PlasticityKind, number> = {
  static: 0,
  stdp: 1,
  'triplet-stdp': 2,
  hebbian: 3,
  oja: 4,
};

/**
 * Floats reserved per neuron for model parameters. Morris-Lecar needs 13; 16 is
 * the next power of two, which keeps every neuron's parameter block aligned to a
 * 64-byte boundary for GPU access.
 */
export const NEURON_PARAM_STRIDE = 16;

/** Floats reserved per synapse for plasticity parameters. */
export const SYNAPSE_PARAM_STRIDE = 12;

/**
 * Parameter slot indices. The integrator, the WGSL shaders and the Rust core all
 * agree on these offsets; changing one means changing all three.
 */
export const PARAM_SLOT = {
  /** LIF: cm, gL, eL, vThresh, vReset, tRefract */
  LIF_CM: 0,
  LIF_GL: 1,
  LIF_EL: 2,
  LIF_VTHRESH: 3,
  LIF_VRESET: 4,
  LIF_TREFRACT: 5,

  /** Izhikevich: a, b, c, d, vPeak, iScale */
  IZH_A: 0,
  IZH_B: 1,
  IZH_C: 2,
  IZH_D: 3,
  IZH_VPEAK: 4,
  IZH_ISCALE: 5,

  /** Hodgkin-Huxley: cm, gNa, gK, gL, eNa, eK, eL, vDetect, q10 */
  HH_CM: 0,
  HH_GNA: 1,
  HH_GK: 2,
  HH_GL: 3,
  HH_ENA: 4,
  HH_EK: 5,
  HH_EL: 6,
  HH_VDETECT: 7,
  HH_Q10: 8,

  /** AdEx: cm, gL, eL, deltaT, vT, vPeak, vReset, a, b, tauW, tRefract */
  ADEX_CM: 0,
  ADEX_GL: 1,
  ADEX_EL: 2,
  ADEX_DELTAT: 3,
  ADEX_VT: 4,
  ADEX_VPEAK: 5,
  ADEX_VRESET: 6,
  ADEX_A: 7,
  ADEX_B: 8,
  ADEX_TAUW: 9,
  ADEX_TREFRACT: 10,

  /** Morris-Lecar: cm, gCa, gK, gL, eCa, eK, eL, v1, v2, v3, v4, phi, vDetect */
  ML_CM: 0,
  ML_GCA: 1,
  ML_GK: 2,
  ML_GL: 3,
  ML_ECA: 4,
  ML_EK: 5,
  ML_EL: 6,
  ML_V1: 7,
  ML_V2: 8,
  ML_V3: 9,
  ML_V4: 10,
  ML_PHI: 11,
  ML_VDETECT: 12,
} as const;

/** Synapse plasticity parameter slots. */
export const SYN_PARAM_SLOT = {
  A_PLUS: 0,
  A_MINUS: 1,
  TAU_PLUS: 2,
  TAU_MINUS: 3,
  TAU_X: 4,
  TAU_Y: 5,
  W_MIN: 6,
  W_MAX: 7,
  LEARNING_RATE: 8,
  STP_U: 9,
  STP_TAU_REC: 10,
  STP_TAU_FACIL: 11,
} as const;

/** Per-neuron mutable state and static columns. */
export interface NeuronBuffers {
  capacity: number;
  count: number;

  /** World position, 3 floats per neuron. */
  position: Float32Array;
  /** Membrane potential (mV). */
  v: Float32Array;
  /** Recovery (Izhikevich u), adaptation (AdEx w) or K activation (ML w). */
  w: Float32Array;
  /** Hodgkin-Huxley sodium activation gate. */
  gateM: Float32Array;
  /** Hodgkin-Huxley sodium inactivation gate. */
  gateH: Float32Array;
  /** Hodgkin-Huxley potassium activation gate. */
  gateN: Float32Array;
  /** Intracellular calcium concentration, arbitrary units, used for display. */
  calcium: Float32Array;
  /** Total synaptic current accumulated this step (pA). */
  iSyn: Float32Array;
  /** Externally injected current from stimuli (pA). */
  iExt: Float32Array;
  /** Constant per-neuron bias current (pA). */
  bias: Float32Array;
  /** Per-neuron noise amplitude (pA). */
  noise: Float32Array;
  /** 1 on the step the neuron spiked, else 0. */
  spike: Uint8Array;
  /** Simulation time of the most recent spike (ms); -Infinity if never. */
  lastSpike: Float32Array;
  /** Simulation time until which the neuron is refractory (ms). */
  refractoryUntil: Float32Array;
  /** Visual spike envelope in 0..1, decays after each spike. Renderer-owned. */
  flash: Float32Array;
  /** Exponentially-smoothed firing rate (Hz). */
  rate: Float32Array;
  /** Total spikes since the last reset. */
  spikeCount: Uint32Array;
  /** MODEL_CODE value. */
  model: Uint8Array;
  /** 0 = excitatory, 1 = inhibitory. */
  polarity: Uint8Array;
  /** 0 = excluded from integration. */
  enabled: Uint8Array;
  /** Packed model parameters, NEURON_PARAM_STRIDE floats per neuron. */
  params: Float32Array;
  /** Render scale multiplier per neuron. */
  scale: Float32Array;
  /** Procedural morphology seed. */
  seed: Uint32Array;
  /** Morphology archetype index, drives which glyph variant is instanced. */
  archetype: Uint8Array;
  /** Index into the population list, or 0xffff for none. */
  population: Uint16Array;
  /** Bitfield: 1 = selected, 2 = hovered, 4 = probed, 8 = ghosted. */
  flags: Uint8Array;
}

/** Per-synapse state. */
export interface SynapseBuffers {
  capacity: number;
  count: number;

  /** Presynaptic neuron slot. */
  pre: Uint32Array;
  /** Postsynaptic neuron slot. */
  post: Uint32Array;
  /** Peak conductance (nS). */
  weight: Float32Array;
  /** Conduction delay (ms). */
  delay: Float32Array;
  /** Rising conductance state variable. */
  gRise: Float32Array;
  /** Decaying conductance state variable; g = gDecay - gRise. */
  gDecay: Float32Array;
  /** Rise time constant (ms). */
  tauRise: Float32Array;
  /** Decay time constant (ms). */
  tauDecay: Float32Array;
  /** Reversal potential (mV). */
  eRev: Float32Array;
  /** Magnesium block strength; 0 disables the NMDA nonlinearity. */
  mgBlock: Float32Array;
  /** Presynaptic eligibility trace. */
  preTrace: Float32Array;
  /** Postsynaptic eligibility trace. */
  postTrace: Float32Array;
  /** Slow presynaptic trace for the triplet rule. */
  preTraceSlow: Float32Array;
  /** Slow postsynaptic trace for the triplet rule. */
  postTraceSlow: Float32Array;
  /** Tsodyks-Markram available resources (0..1). */
  stpR: Float32Array;
  /** Tsodyks-Markram utilisation (0..1). */
  stpU: Float32Array;
  /** Release probability per spike (0..1). */
  releaseProb: Float32Array;
  /** RECEPTOR_CODE value. */
  receptor: Uint8Array;
  /** PLASTICITY_CODE value. */
  plasticity: Uint8Array;
  /** 0 = excluded from integration. */
  enabled: Uint8Array;
  /** Packed plasticity parameters, SYNAPSE_PARAM_STRIDE floats per synapse. */
  params: Float32Array;
  /** Visual travel envelope 0..1 for the impulse moving along this axon. */
  activity: Float32Array;
  /** Spline sag in world units. */
  arc: Float32Array;
}

/**
 * A delivered-spike queue. Presynaptic spikes are pushed with an absolute
 * arrival time and popped once simulation time passes it, which is how axonal
 * delay is implemented without per-synapse ring buffers.
 *
 * Implemented as a bucketed calendar queue: `buckets` slots, each a growable
 * list of synapse indices, indexed by arrival time modulo the horizon.
 */
export interface DelayQueue {
  /** Number of time buckets. */
  buckets: number;
  /** Bucket width in ms; buckets * resolution is the maximum representable delay. */
  resolution: number;
  /** Flat storage of synapse indices, `stride` entries per bucket. */
  entries: Uint32Array;
  /** Occupancy count per bucket. */
  counts: Uint32Array;
  /** Capacity of a single bucket. */
  stride: number;
  /** Amplitude carried by each queued event, parallel to `entries`. */
  amplitude: Float32Array;
}

/** Ring buffer of recent spike events, consumed by the particle system and probes. */
export interface SpikeLog {
  /** Neuron slot that fired. */
  neuron: Uint32Array;
  /** Simulation time of the spike (ms). */
  time: Float32Array;
  capacity: number;
  /** Write cursor; total events written since reset. */
  head: number;
}

/** Everything the integrator mutates each step. */
export interface SimulationBuffers {
  neurons: NeuronBuffers;
  synapses: SynapseBuffers;
  delays: DelayQueue;
  spikes: SpikeLog;
  /** Current simulation time (ms). */
  time: number;
  /** Steps executed since reset. */
  step: number;
}

const DEFAULT_NEURON_CAPACITY = 1024;
const DEFAULT_SYNAPSE_CAPACITY = 4096;

export function allocateNeuronBuffers(capacity = DEFAULT_NEURON_CAPACITY): NeuronBuffers {
  const cap = Math.max(1, capacity);
  const lastSpike = new Float32Array(cap);
  lastSpike.fill(-Infinity);
  const population = new Uint16Array(cap);
  population.fill(0xffff);
  const scale = new Float32Array(cap);
  scale.fill(1);
  const enabled = new Uint8Array(cap);
  enabled.fill(1);
  return {
    capacity: cap,
    count: 0,
    position: new Float32Array(cap * 3),
    v: new Float32Array(cap),
    w: new Float32Array(cap),
    gateM: new Float32Array(cap),
    gateH: new Float32Array(cap),
    gateN: new Float32Array(cap),
    calcium: new Float32Array(cap),
    iSyn: new Float32Array(cap),
    iExt: new Float32Array(cap),
    bias: new Float32Array(cap),
    noise: new Float32Array(cap),
    spike: new Uint8Array(cap),
    lastSpike,
    refractoryUntil: new Float32Array(cap),
    flash: new Float32Array(cap),
    rate: new Float32Array(cap),
    spikeCount: new Uint32Array(cap),
    model: new Uint8Array(cap),
    polarity: new Uint8Array(cap),
    enabled,
    params: new Float32Array(cap * NEURON_PARAM_STRIDE),
    scale,
    seed: new Uint32Array(cap),
    archetype: new Uint8Array(cap),
    population,
    flags: new Uint8Array(cap),
  };
}

export function allocateSynapseBuffers(capacity = DEFAULT_SYNAPSE_CAPACITY): SynapseBuffers {
  const cap = Math.max(1, capacity);
  const enabled = new Uint8Array(cap);
  enabled.fill(1);
  const stpR = new Float32Array(cap);
  stpR.fill(1);
  const releaseProb = new Float32Array(cap);
  releaseProb.fill(1);
  return {
    capacity: cap,
    count: 0,
    pre: new Uint32Array(cap),
    post: new Uint32Array(cap),
    weight: new Float32Array(cap),
    delay: new Float32Array(cap),
    gRise: new Float32Array(cap),
    gDecay: new Float32Array(cap),
    tauRise: new Float32Array(cap),
    tauDecay: new Float32Array(cap),
    eRev: new Float32Array(cap),
    mgBlock: new Float32Array(cap),
    preTrace: new Float32Array(cap),
    postTrace: new Float32Array(cap),
    preTraceSlow: new Float32Array(cap),
    postTraceSlow: new Float32Array(cap),
    stpR,
    stpU: new Float32Array(cap),
    releaseProb,
    receptor: new Uint8Array(cap),
    plasticity: new Uint8Array(cap),
    enabled,
    params: new Float32Array(cap * SYNAPSE_PARAM_STRIDE),
    activity: new Float32Array(cap),
    arc: new Float32Array(cap),
  };
}

export function allocateDelayQueue(
  buckets = 256,
  resolution = 0.25,
  stride = 64,
): DelayQueue {
  return {
    buckets,
    resolution,
    stride,
    entries: new Uint32Array(buckets * stride),
    counts: new Uint32Array(buckets),
    amplitude: new Float32Array(buckets * stride),
  };
}

export function allocateSpikeLog(capacity = 65536): SpikeLog {
  return {
    neuron: new Uint32Array(capacity),
    time: new Float32Array(capacity),
    capacity,
    head: 0,
  };
}

export function allocateSimulationBuffers(
  neuronCapacity = DEFAULT_NEURON_CAPACITY,
  synapseCapacity = DEFAULT_SYNAPSE_CAPACITY,
): SimulationBuffers {
  return {
    neurons: allocateNeuronBuffers(neuronCapacity),
    synapses: allocateSynapseBuffers(synapseCapacity),
    delays: allocateDelayQueue(),
    spikes: allocateSpikeLog(),
    time: 0,
    step: 0,
  };
}

/** Copy a typed array into a larger one of the same kind. */
function grow<T extends Float32Array | Uint32Array | Uint8Array | Uint16Array>(
  src: T,
  nextLength: number,
): T {
  const Ctor = src.constructor as new (length: number) => T;
  const next = new Ctor(nextLength);
  next.set(src as unknown as ArrayLike<number> & T);
  return next;
}

/**
 * Grow neuron storage to at least `required` slots. Returns the same object with
 * every column replaced, so any cached array reference held by a caller is stale
 * after this returns.
 */
export function growNeuronBuffers(b: NeuronBuffers, required: number): NeuronBuffers {
  if (required <= b.capacity) return b;
  let cap = b.capacity;
  while (cap < required) cap *= 2;
  const oldCap = b.capacity;

  b.position = grow(b.position, cap * 3);
  b.v = grow(b.v, cap);
  b.w = grow(b.w, cap);
  b.gateM = grow(b.gateM, cap);
  b.gateH = grow(b.gateH, cap);
  b.gateN = grow(b.gateN, cap);
  b.calcium = grow(b.calcium, cap);
  b.iSyn = grow(b.iSyn, cap);
  b.iExt = grow(b.iExt, cap);
  b.bias = grow(b.bias, cap);
  b.noise = grow(b.noise, cap);
  b.spike = grow(b.spike, cap);
  b.lastSpike = grow(b.lastSpike, cap);
  b.lastSpike.fill(-Infinity, oldCap);
  b.refractoryUntil = grow(b.refractoryUntil, cap);
  b.flash = grow(b.flash, cap);
  b.rate = grow(b.rate, cap);
  b.spikeCount = grow(b.spikeCount, cap);
  b.model = grow(b.model, cap);
  b.polarity = grow(b.polarity, cap);
  b.enabled = grow(b.enabled, cap);
  b.enabled.fill(1, oldCap);
  b.params = grow(b.params, cap * NEURON_PARAM_STRIDE);
  b.scale = grow(b.scale, cap);
  b.scale.fill(1, oldCap);
  b.seed = grow(b.seed, cap);
  b.archetype = grow(b.archetype, cap);
  b.population = grow(b.population, cap);
  b.population.fill(0xffff, oldCap);
  b.flags = grow(b.flags, cap);
  b.capacity = cap;
  return b;
}

/** Grow synapse storage to at least `required` slots. */
export function growSynapseBuffers(b: SynapseBuffers, required: number): SynapseBuffers {
  if (required <= b.capacity) return b;
  let cap = b.capacity;
  while (cap < required) cap *= 2;
  const oldCap = b.capacity;

  b.pre = grow(b.pre, cap);
  b.post = grow(b.post, cap);
  b.weight = grow(b.weight, cap);
  b.delay = grow(b.delay, cap);
  b.gRise = grow(b.gRise, cap);
  b.gDecay = grow(b.gDecay, cap);
  b.tauRise = grow(b.tauRise, cap);
  b.tauDecay = grow(b.tauDecay, cap);
  b.eRev = grow(b.eRev, cap);
  b.mgBlock = grow(b.mgBlock, cap);
  b.preTrace = grow(b.preTrace, cap);
  b.postTrace = grow(b.postTrace, cap);
  b.preTraceSlow = grow(b.preTraceSlow, cap);
  b.postTraceSlow = grow(b.postTraceSlow, cap);
  b.stpR = grow(b.stpR, cap);
  b.stpR.fill(1, oldCap);
  b.stpU = grow(b.stpU, cap);
  b.releaseProb = grow(b.releaseProb, cap);
  b.releaseProb.fill(1, oldCap);
  b.receptor = grow(b.receptor, cap);
  b.plasticity = grow(b.plasticity, cap);
  b.enabled = grow(b.enabled, cap);
  b.enabled.fill(1, oldCap);
  b.params = grow(b.params, cap * SYNAPSE_PARAM_STRIDE);
  b.activity = grow(b.activity, cap);
  b.arc = grow(b.arc, cap);
  b.capacity = cap;
  return b;
}

/** Selection / hover flag bits stored in NeuronBuffers.flags. */
export const NEURON_FLAG = {
  SELECTED: 1,
  HOVERED: 2,
  PROBED: 4,
  GHOSTED: 8,
} as const;
