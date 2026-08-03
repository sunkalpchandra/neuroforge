import type { ShaderBinding } from '../types';
import {
  WGSL_NEURON_STRUCTS,
  WGSL_PRELUDE,
  WGSL_SYNAPSE_STRUCTS,
} from './common';

export { SYNAPSE_CURRENT_SCALE } from './common';

/**
 * Packed synapse parameter slots, transcribed from `SYN_PARAM_SLOT` in
 * `@neuroforge/shared/buffers`, plus the receptor and plasticity codes from
 * `RECEPTOR_CODE` and `PLASTICITY_CODE`.
 */
const SYNAPSE_PARAM_SLOTS = /* wgsl */ `
const SYNAPSE_PARAM_STRIDE : u32 = 12u;

const S_A_PLUS : u32 = 0u;
const S_A_MINUS : u32 = 1u;
const S_TAU_PLUS : u32 = 2u;
const S_TAU_MINUS : u32 = 3u;
const S_TAU_X : u32 = 4u;
const S_TAU_Y : u32 = 5u;
const S_W_MIN : u32 = 6u;
const S_W_MAX : u32 = 7u;
const S_LEARNING_RATE : u32 = 8u;
const S_STP_U : u32 = 9u;
const S_STP_TAU_REC : u32 = 10u;
const S_STP_TAU_FACIL : u32 = 11u;

const RECEPTOR_AMPA : u32 = 0u;
const RECEPTOR_NMDA : u32 = 1u;
const RECEPTOR_GABAA : u32 = 2u;
const RECEPTOR_GABAB : u32 = 3u;
const RECEPTOR_GAP : u32 = 4u;

const PLASTICITY_STATIC : u32 = 0u;
const PLASTICITY_STDP : u32 = 1u;
const PLASTICITY_TRIPLET : u32 = 2u;
const PLASTICITY_HEBBIAN : u32 = 3u;
const PLASTICITY_OJA : u32 = 4u;
`;

/**
 * Conductance update and current accumulation.
 *
 * One invocation per synapse. Conductance is the difference of two exponentials
 * that share an amplitude and decay with their own time constants, so a release
 * adds the same increment to both components and the observed waveform rises
 * with tauRise and falls with tauDecay. The increment is normalised so that
 * `weight` is the peak conductance in nS rather than an amplitude that shifts
 * whenever the kinetics are edited.
 *
 * `arrival[i]` is written by the delay stage: it carries the release amplitude
 * that reaches this synapse on this step (0 when nothing arrives, 1 for a
 * released spike, a fraction for a partial or probabilistic release). This
 * kernel consumes it and clears the slot.
 *
 * Postsynaptic current is accumulated with atomics into a fixed-point i32 buffer
 * because WGSL has no atomic float; see SYNAPSE_CURRENT_SCALE for the factor.
 */
export const SYNAPSE_PROPAGATE_WGSL = /* wgsl */ `
${WGSL_PRELUDE}
${WGSL_NEURON_STRUCTS}
${WGSL_SYNAPSE_STRUCTS}
${SYNAPSE_PARAM_SLOTS}

struct PropagateUniforms {
  @align(16) dt : f32,
  time : f32,
  gain : f32,
  mgConcentration : f32,
  activityTau : f32,
  count : u32,
  seed : u32,
  step : u32,
}

@group(0) @binding(0) var<uniform> uni : PropagateUniforms;
@group(0) @binding(1) var<storage, read> synapses : array<SynapseStatic>;
@group(0) @binding(2) var<storage, read> weights : array<f32>;
@group(0) @binding(3) var<storage, read> params : array<f32>;
@group(0) @binding(4) var<storage, read_write> dynamics : array<SynapseDynamic>;
@group(0) @binding(5) var<storage, read_write> arrival : array<f32>;
@group(0) @binding(6) var<storage, read_write> iSyn : array<atomic<i32>>;
@group(0) @binding(7) var<storage, read> neurons : array<NeuronDynamic>;

// The rise time is held below the decay time by this fraction. The two are equal
// for gap junctions and can be made equal by hand for any receptor, and the
// normalisation below divides by their difference, so without the separation the
// peak estimate loses all its significant digits.
const MAX_RISE_FRACTION : f32 = 0.9;

/**
 * Reciprocal of the peak of exp(-t/tauDecay) - exp(-t/tauRise).
 *
 * The peak occurs where the derivative vanishes, at
 * t = tauDecay tauRise / (tauDecay - tauRise) * ln(tauDecay / tauRise).
 */
fn dualExponentialNorm(tauRise : f32, tauDecay : f32) -> f32 {
  let tPeak = (tauDecay * tauRise / (tauDecay - tauRise)) * log(tauDecay / tauRise);
  let peak = exp(-tPeak / tauDecay) - exp(-tPeak / tauRise);
  return 1.0 / max(peak, 1.0e-6);
}

/**
 * Jahr-Stevens magnesium block, a sigmoid in postsynaptic voltage.
 *
 * Returns the fraction of NMDA channels not blocked. mgBlock scales the block
 * strength and disables the nonlinearity entirely at 0, which is how every
 * non-NMDA receptor is configured.
 */
fn magnesiumBlock(mgBlock : f32, mgConcentration : f32, vPost : f32) -> f32 {
  if (mgBlock <= 0.0) {
    return 1.0;
  }
  return 1.0 / (1.0 + mgBlock * (mgConcentration / 3.57) * safeExp(-0.062 * vPost));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= uni.count) {
    return;
  }

  let dt = uni.dt;
  let syn = synapses[index];
  var state = dynamics[index];
  let enabled = metaEnabled(syn.meta);

  let tauDecay = max(syn.tauDecay, MIN_TAU);
  let tauRise = clamp(syn.tauRise, MIN_TAU, tauDecay * MAX_RISE_FRACTION);

  state.gDecay = state.gDecay * exp(-dt / tauDecay);
  state.gRise = state.gRise * exp(-dt / tauRise);
  state.activity = state.activity * exp(-dt / max(uni.activityTau, MIN_TAU));

  // Short-term plasticity. packSynapseParams writes a utilisation of 0 when STP
  // is disabled, so that single test switches the whole mechanism off.
  let paramBase = index * SYNAPSE_PARAM_STRIDE;
  let stpUtilisation = params[paramBase + S_STP_U];
  let tauRecovery = params[paramBase + S_STP_TAU_REC];
  let tauFacilitation = params[paramBase + S_STP_TAU_FACIL];
  let stpActive = stpUtilisation > 0.0;

  if (stpActive) {
    state.stpR = state.stpR + (1.0 - state.stpR) * relaxFactor(dt, tauRecovery);
    // Utilisation relaxes back to the baseline the synapse was configured with,
    // not to zero: the baseline is the release probability of a rested synapse,
    // and facilitation is the excess above it that a burst builds up.
    if (tauFacilitation > MIN_TAU) {
      state.stpU = relax(state.stpU, stpUtilisation, dt, tauFacilitation);
    } else {
      state.stpU = stpUtilisation;
    }
  }

  let release = arrival[index];
  arrival[index] = 0.0;

  if (enabled && release > 0.0) {
    var efficacy = 1.0;
    if (stpActive) {
      if (tauFacilitation > MIN_TAU) {
        state.stpU = state.stpU + stpUtilisation * (1.0 - state.stpU);
      } else {
        state.stpU = stpUtilisation;
      }
      // The vesicles this release consumes.
      let released = state.stpU * state.stpR;
      state.stpR = max(state.stpR - released, 0.0);
      // Dividing by the baseline utilisation is what makes the stored weight
      // mean the peak conductance of a rested synapse. Without it a depressing
      // synapse would deliver only its utilisation fraction of the weight the
      // editor shows, which for the usual U in 0.2..0.5 is a two- to five-fold
      // shortfall against the CPU reference.
      efficacy = released / max(stpUtilisation, EPSILON);
    }
    let amplitude =
      weights[index] * uni.gain * efficacy * release * dualExponentialNorm(tauRise, tauDecay);
    state.gDecay = state.gDecay + amplitude;
    state.gRise = state.gRise + amplitude;
    state.activity = 1.0;
  }

  if (enabled) {
    let vPost = neurons[syn.post].v;
    var current = 0.0;
    if (metaByte(syn.meta, META_KIND_SHIFT) == RECEPTOR_GAP) {
      // A gap junction is an ohmic bridge between two membranes, not a released
      // transmitter: it conducts continuously and its driving force is the
      // voltage difference across the junction, not a fixed reversal potential.
      current = weights[index] * uni.gain * (neurons[syn.pre].v - vPost);
    } else {
      let g = max(state.gDecay - state.gRise, 0.0);
      current = g * magnesiumBlock(syn.mgBlock, uni.mgConcentration, vPost) * (syn.eRev - vPost);
    }
    if (current != 0.0) {
      atomicAdd(&iSyn[syn.post], toFixedCurrent(current));
    }
  }

  dynamics[index] = state;
}
`;

/**
 * Trace decay and weight update.
 *
 * One invocation per synapse, dispatched after the integrator has published this
 * step's spike flags. Every trace decays first, then the weight is updated using
 * the traces as they stand *before* this step's own spikes are added to them,
 * which is what makes the pair rule antisymmetric and the triplet rule causal.
 *
 * The packed parameter block carries a single potentiation and a single
 * depression amplitude, so the triplet rule ties its third-order amplitudes to
 * the second-order ones: A3+ = A2+ and A3- = A2-. The slow traces are small at
 * low firing rates, so the rule degenerates to exactly pair STDP there and
 * develops the characteristic rate dependence only when the pre- and
 * postsynaptic cells are both firing fast.
 */
export const SYNAPSE_STDP_WGSL = /* wgsl */ `
${WGSL_PRELUDE}
${WGSL_NEURON_STRUCTS}
${WGSL_SYNAPSE_STRUCTS}
${SYNAPSE_PARAM_SLOTS}

struct StdpUniforms {
  @align(16) dt : f32,
  time : f32,
  learningRate : f32,
  count : u32,
}

@group(0) @binding(0) var<uniform> uni : StdpUniforms;
@group(0) @binding(1) var<storage, read> synapses : array<SynapseStatic>;
@group(0) @binding(2) var<storage, read> params : array<f32>;
@group(0) @binding(3) var<storage, read_write> weights : array<f32>;
@group(0) @binding(4) var<storage, read_write> traces : array<SynapseTraces>;
@group(0) @binding(5) var<storage, read> outputs : array<NeuronOutput>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= uni.count) {
    return;
  }

  let syn = synapses[index];
  let rule = metaByte(syn.meta, META_SUBKIND_SHIFT);
  if (!metaEnabled(syn.meta) || rule == PLASTICITY_STATIC) {
    return;
  }

  let dt = uni.dt;
  let base = index * SYNAPSE_PARAM_STRIDE;
  let aPlus = params[base + S_A_PLUS];
  let aMinus = params[base + S_A_MINUS];
  let wMin = params[base + S_W_MIN];
  let wMax = params[base + S_W_MAX];
  let rate = params[base + S_LEARNING_RATE] * uni.learningRate;

  var trace = traces[index];
  trace.preTrace = trace.preTrace * exp(-dt / max(params[base + S_TAU_PLUS], MIN_TAU));
  trace.postTrace = trace.postTrace * exp(-dt / max(params[base + S_TAU_MINUS], MIN_TAU));
  trace.preTraceSlow = trace.preTraceSlow * exp(-dt / max(params[base + S_TAU_X], MIN_TAU));
  trace.postTraceSlow = trace.postTraceSlow * exp(-dt / max(params[base + S_TAU_Y], MIN_TAU));

  let preSpike = outputs[syn.pre].spike != 0u;
  let postSpike = outputs[syn.post].spike != 0u;
  var w = weights[index];

  switch (rule) {
    case PLASTICITY_STDP, PLASTICITY_TRIPLET: {
      let triplet = select(0.0, 1.0, rule == PLASTICITY_TRIPLET);
      if (preSpike) {
        w = w - rate * aMinus * trace.postTrace * (1.0 + triplet * trace.preTraceSlow);
      }
      if (postSpike) {
        w = w + rate * aPlus * trace.preTrace * (1.0 + triplet * trace.postTraceSlow);
      }
    }

    // Rate-coded correlation between the two traces, integrated over the step.
    case PLASTICITY_HEBBIAN: {
      w = w + rate * aPlus * trace.preTrace * trace.postTrace * dt;
    }

    // Hebbian growth with Oja's normalising decay, which bounds the weight
    // without relying on the hard clamp.
    case PLASTICITY_OJA: {
      w = w + rate * aPlus * trace.postTrace * (trace.preTrace - trace.postTrace * w) * dt;
    }

    default: {
    }
  }

  if (preSpike) {
    trace.preTrace = trace.preTrace + 1.0;
    trace.preTraceSlow = trace.preTraceSlow + 1.0;
  }
  if (postSpike) {
    trace.postTrace = trace.postTrace + 1.0;
    trace.postTraceSlow = trace.postTraceSlow + 1.0;
  }

  traces[index] = trace;
  weights[index] = clamp(w, wMin, max(wMax, wMin));
}
`;

export const SYNAPSE_PROPAGATE_BINDINGS: readonly ShaderBinding[] = [
  { binding: 0, name: 'uni', type: 'uniform' },
  { binding: 1, name: 'synapses', type: 'read-only-storage' },
  { binding: 2, name: 'weights', type: 'read-only-storage' },
  { binding: 3, name: 'params', type: 'read-only-storage' },
  { binding: 4, name: 'dynamics', type: 'storage' },
  { binding: 5, name: 'arrival', type: 'storage' },
  { binding: 6, name: 'iSyn', type: 'storage' },
  { binding: 7, name: 'neurons', type: 'read-only-storage' },
];

export const SYNAPSE_STDP_BINDINGS: readonly ShaderBinding[] = [
  { binding: 0, name: 'uni', type: 'uniform' },
  { binding: 1, name: 'synapses', type: 'read-only-storage' },
  { binding: 2, name: 'params', type: 'read-only-storage' },
  { binding: 3, name: 'weights', type: 'storage' },
  { binding: 4, name: 'traces', type: 'storage' },
  { binding: 5, name: 'outputs', type: 'read-only-storage' },
];
