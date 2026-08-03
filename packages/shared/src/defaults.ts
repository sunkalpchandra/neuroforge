import { NEURON_PARAM_STRIDE, PARAM_SLOT, SYNAPSE_PARAM_STRIDE, SYN_PARAM_SLOT } from './buffers';
import type {
  AdExParams,
  HodgkinHuxleyParams,
  IzhikevichParams,
  LifParams,
  MorphologyArchetype,
  Morphology,
  MorrisLecarParams,
  NeuronModelKind,
  NeuronParams,
  NeuronPolarity,
} from './neuron';
import type { PlasticityConfig, ShortTermPlasticity } from './synapse';

/**
 * Biophysically reasonable defaults for each model. The LIF, AdEx and HH values
 * are the standard cortical parameterisations found in the literature; the
 * Izhikevich defaults produce regular spiking.
 */

export const DEFAULT_LIF: LifParams = {
  kind: 'lif',
  cm: 200,
  gL: 10,
  eL: -70,
  vThresh: -50,
  vReset: -58,
  tRefract: 2,
};

export const DEFAULT_IZHIKEVICH: IzhikevichParams = {
  kind: 'izhikevich',
  a: 0.02,
  b: 0.2,
  c: -65,
  d: 8,
  vPeak: 30,
  iScale: 0.04,
};

export const DEFAULT_HODGKIN_HUXLEY: HodgkinHuxleyParams = {
  kind: 'hodgkin-huxley',
  cm: 100,
  gNa: 12000,
  gK: 3600,
  gL: 30,
  eNa: 50,
  eK: -77,
  eL: -54.4,
  vDetect: -20,
  q10: 1,
};

export const DEFAULT_ADEX: AdExParams = {
  kind: 'adex',
  cm: 281,
  gL: 30,
  eL: -70.6,
  deltaT: 2,
  vT: -50.4,
  vPeak: 20,
  vReset: -70.6,
  a: 4,
  b: 80.5,
  tauW: 144,
  tRefract: 2,
};

export const DEFAULT_MORRIS_LECAR: MorrisLecarParams = {
  kind: 'morris-lecar',
  cm: 20,
  gCa: 4.4,
  gK: 8,
  gL: 2,
  eCa: 120,
  eK: -84,
  eL: -60,
  v1: -1.2,
  v2: 18,
  v3: 2,
  v4: 30,
  phi: 0.04,
  vDetect: 0,
};

const DEFAULT_PARAMS_BY_KIND: Record<NeuronModelKind, NeuronParams> = {
  lif: DEFAULT_LIF,
  izhikevich: DEFAULT_IZHIKEVICH,
  'hodgkin-huxley': DEFAULT_HODGKIN_HUXLEY,
  adex: DEFAULT_ADEX,
  'morris-lecar': DEFAULT_MORRIS_LECAR,
};

/** A fresh, mutable copy of the defaults for a model. */
export function defaultParams(kind: NeuronModelKind): NeuronParams {
  return { ...DEFAULT_PARAMS_BY_KIND[kind] } as NeuronParams;
}

/**
 * Well-known Izhikevich presets. These are the parameter sets from the original
 * 2003 paper that produce each named firing pattern.
 */
export const IZHIKEVICH_PRESETS: Record<string, Omit<IzhikevichParams, 'kind'>> = {
  'regular-spiking': { a: 0.02, b: 0.2, c: -65, d: 8, vPeak: 30, iScale: 0.04 },
  'intrinsically-bursting': { a: 0.02, b: 0.2, c: -55, d: 4, vPeak: 30, iScale: 0.04 },
  chattering: { a: 0.02, b: 0.2, c: -50, d: 2, vPeak: 30, iScale: 0.04 },
  'fast-spiking': { a: 0.1, b: 0.2, c: -65, d: 2, vPeak: 30, iScale: 0.04 },
  'low-threshold-spiking': { a: 0.02, b: 0.25, c: -65, d: 2, vPeak: 30, iScale: 0.04 },
  'thalamo-cortical': { a: 0.02, b: 0.25, c: -65, d: 0.05, vPeak: 30, iScale: 0.04 },
  resonator: { a: 0.1, b: 0.26, c: -65, d: 2, vPeak: 30, iScale: 0.04 },
};

/**
 * Morphology presets per archetype. Chosen so that the procedural generator
 * produces recognisably different silhouettes at a glance.
 */
const MORPHOLOGY_PRESETS: Record<MorphologyArchetype, Omit<Morphology, 'seed' | 'archetype'>> = {
  pyramidal: {
    somaRadius: 1,
    dendriteCount: 5,
    dendriteDepth: 4,
    dendriteLength: 7.5,
    dendriteTaper: 0.68,
    dendriteSpread: 0.42,
    axonLength: 9,
    axonTerminals: 4,
    scale: 1,
  },
  basket: {
    somaRadius: 0.86,
    dendriteCount: 7,
    dendriteDepth: 3,
    dendriteLength: 5,
    dendriteTaper: 0.6,
    dendriteSpread: 0.95,
    axonLength: 6,
    axonTerminals: 8,
    scale: 0.92,
  },
  granule: {
    somaRadius: 0.6,
    dendriteCount: 3,
    dendriteDepth: 3,
    dendriteLength: 4,
    dendriteTaper: 0.62,
    dendriteSpread: 0.35,
    axonLength: 11,
    axonTerminals: 3,
    scale: 0.75,
  },
  purkinje: {
    somaRadius: 1.15,
    dendriteCount: 2,
    dendriteDepth: 6,
    dendriteLength: 10,
    dendriteTaper: 0.74,
    dendriteSpread: 0.55,
    axonLength: 8,
    axonTerminals: 2,
    scale: 1.15,
  },
  stellate: {
    somaRadius: 0.78,
    dendriteCount: 8,
    dendriteDepth: 3,
    dendriteLength: 5.5,
    dendriteTaper: 0.64,
    dendriteSpread: 1.25,
    axonLength: 5,
    axonTerminals: 5,
    scale: 0.88,
  },
  motor: {
    somaRadius: 1.3,
    dendriteCount: 6,
    dendriteDepth: 4,
    dendriteLength: 8,
    dendriteTaper: 0.7,
    dendriteSpread: 0.85,
    axonLength: 16,
    axonTerminals: 3,
    scale: 1.2,
  },
  bipolar: {
    somaRadius: 0.7,
    dendriteCount: 2,
    dendriteDepth: 3,
    dendriteLength: 6,
    dendriteTaper: 0.66,
    dendriteSpread: 0.18,
    axonLength: 10,
    axonTerminals: 2,
    scale: 0.85,
  },
};

export function defaultMorphology(
  archetype: MorphologyArchetype = 'pyramidal',
  seed = 1,
): Morphology {
  return { ...MORPHOLOGY_PRESETS[archetype], archetype, seed };
}

/** The archetype conventionally paired with each polarity. */
export function archetypeForPolarity(polarity: NeuronPolarity): MorphologyArchetype {
  return polarity === 'inhibitory' ? 'basket' : 'pyramidal';
}

/**
 * Write a model's parameters into the packed SoA parameter block.
 *
 * This is the single place that knows the memory layout, and it is shared by the
 * TypeScript integrator, the WASM bridge and the WGSL upload path so the three
 * can never disagree about slot ordering.
 */
export function packNeuronParams(
  params: NeuronParams,
  out: Float32Array,
  slot: number,
): void {
  const base = slot * NEURON_PARAM_STRIDE;
  out.fill(0, base, base + NEURON_PARAM_STRIDE);
  switch (params.kind) {
    case 'lif':
      out[base + PARAM_SLOT.LIF_CM] = params.cm;
      out[base + PARAM_SLOT.LIF_GL] = params.gL;
      out[base + PARAM_SLOT.LIF_EL] = params.eL;
      out[base + PARAM_SLOT.LIF_VTHRESH] = params.vThresh;
      out[base + PARAM_SLOT.LIF_VRESET] = params.vReset;
      out[base + PARAM_SLOT.LIF_TREFRACT] = params.tRefract;
      break;
    case 'izhikevich':
      out[base + PARAM_SLOT.IZH_A] = params.a;
      out[base + PARAM_SLOT.IZH_B] = params.b;
      out[base + PARAM_SLOT.IZH_C] = params.c;
      out[base + PARAM_SLOT.IZH_D] = params.d;
      out[base + PARAM_SLOT.IZH_VPEAK] = params.vPeak;
      out[base + PARAM_SLOT.IZH_ISCALE] = params.iScale;
      break;
    case 'hodgkin-huxley':
      out[base + PARAM_SLOT.HH_CM] = params.cm;
      out[base + PARAM_SLOT.HH_GNA] = params.gNa;
      out[base + PARAM_SLOT.HH_GK] = params.gK;
      out[base + PARAM_SLOT.HH_GL] = params.gL;
      out[base + PARAM_SLOT.HH_ENA] = params.eNa;
      out[base + PARAM_SLOT.HH_EK] = params.eK;
      out[base + PARAM_SLOT.HH_EL] = params.eL;
      out[base + PARAM_SLOT.HH_VDETECT] = params.vDetect;
      out[base + PARAM_SLOT.HH_Q10] = params.q10;
      break;
    case 'adex':
      out[base + PARAM_SLOT.ADEX_CM] = params.cm;
      out[base + PARAM_SLOT.ADEX_GL] = params.gL;
      out[base + PARAM_SLOT.ADEX_EL] = params.eL;
      out[base + PARAM_SLOT.ADEX_DELTAT] = params.deltaT;
      out[base + PARAM_SLOT.ADEX_VT] = params.vT;
      out[base + PARAM_SLOT.ADEX_VPEAK] = params.vPeak;
      out[base + PARAM_SLOT.ADEX_VRESET] = params.vReset;
      out[base + PARAM_SLOT.ADEX_A] = params.a;
      out[base + PARAM_SLOT.ADEX_B] = params.b;
      out[base + PARAM_SLOT.ADEX_TAUW] = params.tauW;
      out[base + PARAM_SLOT.ADEX_TREFRACT] = params.tRefract;
      break;
    case 'morris-lecar':
      out[base + PARAM_SLOT.ML_CM] = params.cm;
      out[base + PARAM_SLOT.ML_GCA] = params.gCa;
      out[base + PARAM_SLOT.ML_GK] = params.gK;
      out[base + PARAM_SLOT.ML_GL] = params.gL;
      out[base + PARAM_SLOT.ML_ECA] = params.eCa;
      out[base + PARAM_SLOT.ML_EK] = params.eK;
      out[base + PARAM_SLOT.ML_EL] = params.eL;
      out[base + PARAM_SLOT.ML_V1] = params.v1;
      out[base + PARAM_SLOT.ML_V2] = params.v2;
      out[base + PARAM_SLOT.ML_V3] = params.v3;
      out[base + PARAM_SLOT.ML_V4] = params.v4;
      out[base + PARAM_SLOT.ML_PHI] = params.phi;
      out[base + PARAM_SLOT.ML_VDETECT] = params.vDetect;
      break;
  }
}

/** Read a packed parameter block back into a typed params object. */
export function unpackNeuronParams(
  kind: NeuronModelKind,
  src: Float32Array,
  slot: number,
): NeuronParams {
  const b = slot * NEURON_PARAM_STRIDE;
  switch (kind) {
    case 'lif':
      return {
        kind: 'lif',
        cm: src[b + PARAM_SLOT.LIF_CM],
        gL: src[b + PARAM_SLOT.LIF_GL],
        eL: src[b + PARAM_SLOT.LIF_EL],
        vThresh: src[b + PARAM_SLOT.LIF_VTHRESH],
        vReset: src[b + PARAM_SLOT.LIF_VRESET],
        tRefract: src[b + PARAM_SLOT.LIF_TREFRACT],
      };
    case 'izhikevich':
      return {
        kind: 'izhikevich',
        a: src[b + PARAM_SLOT.IZH_A],
        b: src[b + PARAM_SLOT.IZH_B],
        c: src[b + PARAM_SLOT.IZH_C],
        d: src[b + PARAM_SLOT.IZH_D],
        vPeak: src[b + PARAM_SLOT.IZH_VPEAK],
        iScale: src[b + PARAM_SLOT.IZH_ISCALE],
      };
    case 'hodgkin-huxley':
      return {
        kind: 'hodgkin-huxley',
        cm: src[b + PARAM_SLOT.HH_CM],
        gNa: src[b + PARAM_SLOT.HH_GNA],
        gK: src[b + PARAM_SLOT.HH_GK],
        gL: src[b + PARAM_SLOT.HH_GL],
        eNa: src[b + PARAM_SLOT.HH_ENA],
        eK: src[b + PARAM_SLOT.HH_EK],
        eL: src[b + PARAM_SLOT.HH_EL],
        vDetect: src[b + PARAM_SLOT.HH_VDETECT],
        q10: src[b + PARAM_SLOT.HH_Q10],
      };
    case 'adex':
      return {
        kind: 'adex',
        cm: src[b + PARAM_SLOT.ADEX_CM],
        gL: src[b + PARAM_SLOT.ADEX_GL],
        eL: src[b + PARAM_SLOT.ADEX_EL],
        deltaT: src[b + PARAM_SLOT.ADEX_DELTAT],
        vT: src[b + PARAM_SLOT.ADEX_VT],
        vPeak: src[b + PARAM_SLOT.ADEX_VPEAK],
        vReset: src[b + PARAM_SLOT.ADEX_VRESET],
        a: src[b + PARAM_SLOT.ADEX_A],
        b: src[b + PARAM_SLOT.ADEX_B],
        tauW: src[b + PARAM_SLOT.ADEX_TAUW],
        tRefract: src[b + PARAM_SLOT.ADEX_TREFRACT],
      };
    case 'morris-lecar':
      return {
        kind: 'morris-lecar',
        cm: src[b + PARAM_SLOT.ML_CM],
        gCa: src[b + PARAM_SLOT.ML_GCA],
        gK: src[b + PARAM_SLOT.ML_GK],
        gL: src[b + PARAM_SLOT.ML_GL],
        eCa: src[b + PARAM_SLOT.ML_ECA],
        eK: src[b + PARAM_SLOT.ML_EK],
        eL: src[b + PARAM_SLOT.ML_EL],
        v1: src[b + PARAM_SLOT.ML_V1],
        v2: src[b + PARAM_SLOT.ML_V2],
        v3: src[b + PARAM_SLOT.ML_V3],
        v4: src[b + PARAM_SLOT.ML_V4],
        phi: src[b + PARAM_SLOT.ML_PHI],
        vDetect: src[b + PARAM_SLOT.ML_VDETECT],
      };
  }
}

/** Write plasticity + STP parameters into the packed synapse parameter block. */
export function packSynapseParams(
  plasticity: PlasticityConfig,
  stp: ShortTermPlasticity,
  out: Float32Array,
  slot: number,
): void {
  const b = slot * SYNAPSE_PARAM_STRIDE;
  out[b + SYN_PARAM_SLOT.A_PLUS] = plasticity.aPlus;
  out[b + SYN_PARAM_SLOT.A_MINUS] = plasticity.aMinus;
  out[b + SYN_PARAM_SLOT.TAU_PLUS] = plasticity.tauPlus;
  out[b + SYN_PARAM_SLOT.TAU_MINUS] = plasticity.tauMinus;
  out[b + SYN_PARAM_SLOT.TAU_X] = plasticity.tauX;
  out[b + SYN_PARAM_SLOT.TAU_Y] = plasticity.tauY;
  out[b + SYN_PARAM_SLOT.W_MIN] = plasticity.wMin;
  out[b + SYN_PARAM_SLOT.W_MAX] = plasticity.wMax;
  out[b + SYN_PARAM_SLOT.LEARNING_RATE] = plasticity.learningRate;
  out[b + SYN_PARAM_SLOT.STP_U] = stp.enabled ? stp.u : 0;
  out[b + SYN_PARAM_SLOT.STP_TAU_REC] = stp.tauRec;
  out[b + SYN_PARAM_SLOT.STP_TAU_FACIL] = stp.tauFacil;
}

/** Resting potential for a model, used to initialise v on reset. */
export function restingPotential(params: NeuronParams): number {
  switch (params.kind) {
    case 'lif':
      return params.eL;
    case 'izhikevich':
      return params.c;
    case 'hodgkin-huxley':
      return -65;
    case 'adex':
      return params.eL;
    case 'morris-lecar':
      return -60.9;
  }
}

/** Voltage range a model spans, used to normalise colour ramps and plots. */
export function voltageRange(kind: NeuronModelKind): { min: number; max: number } {
  switch (kind) {
    case 'hodgkin-huxley':
      return { min: -90, max: 45 };
    case 'izhikevich':
      return { min: -80, max: 35 };
    case 'morris-lecar':
      return { min: -70, max: 45 };
    case 'adex':
      return { min: -80, max: 25 };
    case 'lif':
      return { min: -80, max: -40 };
  }
}
