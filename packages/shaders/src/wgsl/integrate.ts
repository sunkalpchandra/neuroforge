import type { ShaderBinding } from '../types';
import { WGSL_NEURON_STRUCTS, WGSL_PRELUDE } from './common';

/**
 * Packed neuron parameter slots.
 *
 * These are a literal transcription of `PARAM_SLOT` in
 * `@neuroforge/shared/buffers`, with `NEURON_PARAM_STRIDE` floats per neuron.
 * The CPU integrator, the WASM core and this kernel all index the same table;
 * every slot index below appears in exactly one place so a change to the shared
 * table is a one-line change here.
 */
const NEURON_PARAM_SLOTS = /* wgsl */ `
const NEURON_PARAM_STRIDE : u32 = 16u;

const MODEL_LIF : u32 = 0u;
const MODEL_IZHIKEVICH : u32 = 1u;
const MODEL_HODGKIN_HUXLEY : u32 = 2u;
const MODEL_ADEX : u32 = 3u;
const MODEL_MORRIS_LECAR : u32 = 4u;

const P_LIF_CM : u32 = 0u;
const P_LIF_GL : u32 = 1u;
const P_LIF_EL : u32 = 2u;
const P_LIF_VTHRESH : u32 = 3u;
const P_LIF_VRESET : u32 = 4u;
const P_LIF_TREFRACT : u32 = 5u;

const P_IZH_A : u32 = 0u;
const P_IZH_B : u32 = 1u;
const P_IZH_C : u32 = 2u;
const P_IZH_D : u32 = 3u;
const P_IZH_VPEAK : u32 = 4u;
const P_IZH_ISCALE : u32 = 5u;

const P_HH_CM : u32 = 0u;
const P_HH_GNA : u32 = 1u;
const P_HH_GK : u32 = 2u;
const P_HH_GL : u32 = 3u;
const P_HH_ENA : u32 = 4u;
const P_HH_EK : u32 = 5u;
const P_HH_EL : u32 = 6u;
const P_HH_VDETECT : u32 = 7u;
const P_HH_Q10 : u32 = 8u;

const P_ADEX_CM : u32 = 0u;
const P_ADEX_GL : u32 = 1u;
const P_ADEX_EL : u32 = 2u;
const P_ADEX_DELTAT : u32 = 3u;
const P_ADEX_VT : u32 = 4u;
const P_ADEX_VPEAK : u32 = 5u;
const P_ADEX_VRESET : u32 = 6u;
const P_ADEX_A : u32 = 7u;
const P_ADEX_B : u32 = 8u;
const P_ADEX_TAUW : u32 = 9u;
const P_ADEX_TREFRACT : u32 = 10u;

const P_ML_CM : u32 = 0u;
const P_ML_GCA : u32 = 1u;
const P_ML_GK : u32 = 2u;
const P_ML_GL : u32 = 3u;
const P_ML_ECA : u32 = 4u;
const P_ML_EK : u32 = 5u;
const P_ML_EL : u32 = 6u;
const P_ML_V1 : u32 = 7u;
const P_ML_V2 : u32 = 8u;
const P_ML_V3 : u32 = 9u;
const P_ML_V4 : u32 = 10u;
const P_ML_PHI : u32 = 11u;
const P_ML_VDETECT : u32 = 12u;
`;

/**
 * Classic squid-axon rate functions, in the modern convention where the argument
 * is the absolute membrane potential in mV rather than a displacement from rest.
 *
 * `alphaM` and `alphaN` are 0/0 at V = -40 mV and V = -55 mV respectively. Both
 * are removable: the numerator and the denominator are each linear in the shifted
 * voltage there, so the ratio tends to 0.1*x / (x/10) = 1 for alphaM and
 * 0.01*x / (x/10) = 0.1 for alphaN. Evaluating the quotient directly at those
 * voltages yields a NaN that then destroys the gate for the rest of the run, so
 * the limit is substituted whenever the denominator collapses.
 */
const HH_RATES = /* wgsl */ `
fn hhAlphaM(v : f32) -> f32 {
  let x = v + 40.0;
  let denom = 1.0 - safeExp(-x * 0.1);
  if (abs(denom) < 1.0e-6) {
    return 1.0;
  }
  return 0.1 * x / denom;
}

fn hhBetaM(v : f32) -> f32 {
  return 4.0 * safeExp(-(v + 65.0) / 18.0);
}

fn hhAlphaH(v : f32) -> f32 {
  return 0.07 * safeExp(-(v + 65.0) / 20.0);
}

fn hhBetaH(v : f32) -> f32 {
  return 1.0 / (1.0 + safeExp(-(v + 35.0) * 0.1));
}

fn hhAlphaN(v : f32) -> f32 {
  let x = v + 55.0;
  let denom = 1.0 - safeExp(-x * 0.1);
  if (abs(denom) < 1.0e-6) {
    return 0.1;
  }
  return 0.01 * x / denom;
}

fn hhBetaN(v : f32) -> f32 {
  return 0.125 * safeExp(-(v + 65.0) / 80.0);
}
`;

/**
 * Compute shader integrating all five membrane models. Workgroup size 64.
 *
 * One invocation owns one neuron for the whole step, so nothing but the spike
 * log cursor is contended. The kernel consumes and clears `iSyn`, which means
 * `SYNAPSE_PROPAGATE_WGSL` must be dispatched before it within a step and the
 * host never has to clear the accumulator itself.
 *
 * `uni.integrator` selects forward Euler (0) or exponential Euler (1) for every
 * linear relaxation. `SimulationSettings.integrator` values `rk2` and `rk4` have
 * no GPU counterpart and should be uploaded as exponential Euler, which is both
 * stabler and cheaper than either at the timesteps this app runs.
 */
export const NEURON_INTEGRATE_WGSL = /* wgsl */ `
${WGSL_PRELUDE}
${WGSL_NEURON_STRUCTS}
${NEURON_PARAM_SLOTS}
${HH_RATES}

struct IntegrateUniforms {
  @align(16) dt : f32,
  time : f32,
  noiseScale : f32,
  flashTau : f32,
  rateTau : f32,
  calciumTau : f32,
  calciumGain : f32,
  count : u32,
  seed : u32,
  step : u32,
  spikeLogCapacity : u32,
  integrator : u32,
}

@group(0) @binding(0) var<uniform> uni : IntegrateUniforms;
@group(0) @binding(1) var<storage, read> params : array<f32>;
@group(0) @binding(2) var<storage, read> statics : array<NeuronStatic>;
@group(0) @binding(3) var<storage, read_write> dynamics : array<NeuronDynamic>;
@group(0) @binding(4) var<storage, read_write> outputs : array<NeuronOutput>;
@group(0) @binding(5) var<storage, read_write> iSyn : array<i32>;
@group(0) @binding(6) var<storage, read_write> spikeLog : array<atomic<u32>>;

// The AdEx upstroke is unbounded; clamping its argument keeps the exponential
// finite while still overshooting vPeak by orders of magnitude, so the spike is
// detected on exactly the step it would have been without the clamp.
const ADEX_EXP_ARG_MAX : f32 = 20.0;

// Hodgkin-Huxley and Morris-Lecar have no reset, so a spike is an upward
// crossing of vDetect. A single action potential can wobble across that level
// more than once while the noise term is active, which would log one spike as
// several; two crossings closer together than this are the same event. Matches
// EDGE_REFRACTORY in the reference CPU integrator.
const EDGE_REFRACTORY : f32 = 1.0;

/** Push a spike into the ring the particle system and the probes read. */
fn logSpike(neuron : u32) {
  let ticket = atomicAdd(&spikeLog[0], 1u);
  if (uni.spikeLogCapacity > 0u) {
    atomicStore(&spikeLog[1u + (ticket % uni.spikeLogCapacity)], neuron);
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= uni.count) {
    return;
  }

  let dt = uni.dt;
  let mode = uni.integrator;
  let info = statics[index];
  var state = dynamics[index];
  var outState = outputs[index];

  // Synaptic current is delivered in fixed point by the propagate pass. Reading
  // it clears the accumulator for the next step.
  let iSynPa = f32(iSyn[index]) * INV_CURRENT_SCALE;
  iSyn[index] = 0;
  outState.spike = 0u;

  let flashDecay = exp(-dt / max(uni.flashTau, MIN_TAU));
  let rateDecay = exp(-dt / max(uni.rateTau, MIN_TAU));

  if (!metaEnabled(info.meta)) {
    outState.flash = outState.flash * flashDecay;
    outState.rate = outState.rate * rateDecay;
    outputs[index] = outState;
    return;
  }

  var noiseCurrent = 0.0;
  let noiseAmplitude = info.noise + uni.noiseScale;
  if (noiseAmplitude > 0.0) {
    // White noise is scaled by 1/sqrt(dt) so the variance a membrane accumulates
    // over a millisecond is independent of the timestep. Without it, halving dt
    // quietly halves the effective noise, and the GPU backend would disagree
    // with the CPU reference by sqrt(1/dt) - a factor of 3.2 at the default
    // dt = 0.1 ms.
    let whiteNoiseScale = inverseSqrt(max(dt, MIN_TAU));
    noiseCurrent =
      noiseAmplitude * whiteNoiseScale * randomNormal(hashCombine(index ^ uni.seed, uni.step));
  }
  let current = iSynPa + info.iExt + info.bias + noiseCurrent;

  let base = index * NEURON_PARAM_STRIDE;
  let vPrev = state.v;
  var spiked = false;

  switch (metaByte(info.meta, META_KIND_SHIFT)) {
    // cm dv/dt = -gL (v - eL) + I, integrated as a relaxation toward
    // eL + I/gL with time constant cm/gL, held at reset while refractory.
    case MODEL_LIF: {
      let cm = params[base + P_LIF_CM];
      let gL = max(params[base + P_LIF_GL], MIN_CONDUCTANCE);
      let eL = params[base + P_LIF_EL];
      let vThresh = params[base + P_LIF_VTHRESH];
      let vReset = params[base + P_LIF_VRESET];
      let tRefract = params[base + P_LIF_TREFRACT];

      if (uni.time < state.refractoryUntil) {
        state.v = vReset;
      } else {
        state.v = integrateLinear(state.v, eL + current / gL, dt, cm / gL, mode);
        if (state.v >= vThresh) {
          state.v = vReset;
          state.refractoryUntil = uni.time + tRefract;
          spiked = true;
        }
      }
    }

    // dv/dt = 0.04 v^2 + 5 v + 140 - u + I, du/dt = a (b v - u).
    // The voltage takes two half steps, as in the 2003 paper, because the
    // quadratic is stiff near the upstroke and a single full step at dt = 0.5 ms
    // already misplaces the peak.
    case MODEL_IZHIKEVICH: {
      let a = params[base + P_IZH_A];
      let b = params[base + P_IZH_B];
      let c = params[base + P_IZH_C];
      let d = params[base + P_IZH_D];
      let vPeak = params[base + P_IZH_VPEAK];
      let drive = current * params[base + P_IZH_ISCALE];

      let halfDt = dt * 0.5;
      var v = state.v;
      v = min(v + halfDt * (0.04 * v * v + 5.0 * v + 140.0 - state.w + drive), vPeak);
      v = min(v + halfDt * (0.04 * v * v + 5.0 * v + 140.0 - state.w + drive), vPeak);

      state.w = integrateLinear(state.w, b * v, dt, 1.0 / max(a, EPSILON), mode);
      state.v = v;

      if (state.v >= vPeak) {
        state.v = c;
        state.w = state.w + d;
        spiked = true;
      }
    }

    // Full m/h/n kinetics. Gates use the exact exponential update for their own
    // linear relaxation, then the voltage relaxes toward the conductance-weighted
    // reversal with time constant cm/gTotal.
    case MODEL_HODGKIN_HUXLEY: {
      let cm = params[base + P_HH_CM];
      let gNa = params[base + P_HH_GNA];
      let gK = params[base + P_HH_GK];
      let gL = params[base + P_HH_GL];
      let eNa = params[base + P_HH_ENA];
      let eK = params[base + P_HH_EK];
      let eL = params[base + P_HH_EL];
      let vDetect = params[base + P_HH_VDETECT];
      // q10 scales every rate constant; 1 is the 6.3 C reference the classic
      // parameter set was measured at.
      let q10 = max(params[base + P_HH_Q10], MIN_TAU);

      let v = state.v;
      state.m = gateStep(state.m, hhAlphaM(v) * q10, hhBetaM(v) * q10, dt);
      state.h = gateStep(state.h, hhAlphaH(v) * q10, hhBetaH(v) * q10, dt);
      state.n = gateStep(state.n, hhAlphaN(v) * q10, hhBetaN(v) * q10, dt);

      let gNaOpen = gNa * state.m * state.m * state.m * state.h;
      let gKOpen = gK * state.n * state.n * state.n * state.n;
      let gTotal = max(gNaOpen + gKOpen + gL, MIN_CONDUCTANCE);
      let vInf = (gNaOpen * eNa + gKOpen * eK + gL * eL + current) / gTotal;
      state.v = integrateLinear(v, vInf, dt, cm / gTotal, mode);

      spiked = vPrev < vDetect
        && state.v >= vDetect
        && uni.time - state.lastSpike > EDGE_REFRACTORY;
    }

    // cm dv/dt = -gL (v - eL) + gL dT exp((v - vT)/dT) - w + I,
    // tauW dw/dt = a (v - eL) - w.
    // The exponential term and the adaptation current are frozen across the step
    // and folded into the steady state, leaving the leak to be relaxed exactly.
    case MODEL_ADEX: {
      let cm = params[base + P_ADEX_CM];
      let gL = max(params[base + P_ADEX_GL], MIN_CONDUCTANCE);
      let eL = params[base + P_ADEX_EL];
      let deltaT = params[base + P_ADEX_DELTAT];
      let vT = params[base + P_ADEX_VT];
      let vPeak = params[base + P_ADEX_VPEAK];
      let vReset = params[base + P_ADEX_VRESET];
      let a = params[base + P_ADEX_A];
      let b = params[base + P_ADEX_B];
      let tauW = max(params[base + P_ADEX_TAUW], MIN_TAU);
      let tRefract = params[base + P_ADEX_TREFRACT];

      let vBefore = state.v;
      if (uni.time < state.refractoryUntil) {
        state.v = vReset;
      } else {
        var upstroke = 0.0;
        if (deltaT > MIN_TAU) {
          upstroke = gL * deltaT * exp(min((vBefore - vT) / deltaT, ADEX_EXP_ARG_MAX));
        }
        let vInf = eL + (upstroke - state.w + current) / gL;
        state.v = min(integrateLinear(vBefore, vInf, dt, cm / gL, mode), vPeak);
      }

      state.w = integrateLinear(state.w, a * (vBefore - eL), dt, tauW, mode);

      if (state.v >= vPeak) {
        state.v = vReset;
        state.w = state.w + b;
        state.refractoryUntil = uni.time + tRefract;
        spiked = true;
      }
    }

    // m_inf and w_inf are instantaneous sigmoids of v; w relaxes toward w_inf
    // with tau_w = 1 / (phi cosh((v - v3) / 2 v4)).
    case MODEL_MORRIS_LECAR: {
      let cm = params[base + P_ML_CM];
      let gCa = params[base + P_ML_GCA];
      let gK = params[base + P_ML_GK];
      let gL = params[base + P_ML_GL];
      let eCa = params[base + P_ML_ECA];
      let eK = params[base + P_ML_EK];
      let eL = params[base + P_ML_EL];
      let v1 = params[base + P_ML_V1];
      let v2 = max(params[base + P_ML_V2], MIN_TAU);
      let v3 = params[base + P_ML_V3];
      let v4 = max(params[base + P_ML_V4], MIN_TAU);
      let phi = max(params[base + P_ML_PHI], EPSILON);
      let vDetect = params[base + P_ML_VDETECT];

      let v = state.v;
      let mInf = 0.5 * (1.0 + tanh((v - v1) / v2));
      let wInf = 0.5 * (1.0 + tanh((v - v3) / v4));
      let tauW = 1.0 / (phi * cosh(clamp((v - v3) / (2.0 * v4), -30.0, 30.0)));
      state.w = integrateLinear(state.w, wInf, dt, tauW, mode);

      let gCaOpen = gCa * mInf;
      let gKOpen = gK * state.w;
      let gTotal = max(gCaOpen + gKOpen + gL, MIN_CONDUCTANCE);
      let vInf = (gCaOpen * eCa + gKOpen * eK + gL * eL + current) / gTotal;
      state.v = integrateLinear(v, vInf, dt, cm / gTotal, mode);

      spiked = vPrev < vDetect
        && state.v >= vDetect
        && uni.time - state.lastSpike > EDGE_REFRACTORY;
    }

    default: {
    }
  }

  state.v = clamp(state.v, -V_LIMIT, V_LIMIT);
  state.calcium = state.calcium * exp(-dt / max(uni.calciumTau, MIN_TAU));
  outState.flash = outState.flash * flashDecay;
  outState.rate = outState.rate * rateDecay;

  if (spiked) {
    outState.spike = 1u;
    outState.spikeCount = outState.spikeCount + 1u;
    outState.flash = 1.0;
    // An exponential spike kernel of unit area integrates to one event, so an
    // increment of 1/tau expressed per second reads directly as a rate in Hz.
    outState.rate = outState.rate + 1000.0 / max(uni.rateTau, MIN_TAU);
    state.lastSpike = uni.time;
    state.calcium = state.calcium + uni.calciumGain;
    logSpike(index);
  }

  dynamics[index] = state;
  outputs[index] = outState;
}
`;

/** Bind group layout that NEURON_INTEGRATE_WGSL expects, as a plain description. */
export const NEURON_INTEGRATE_BINDINGS: readonly ShaderBinding[] = [
  { binding: 0, name: 'uni', type: 'uniform' },
  { binding: 1, name: 'params', type: 'read-only-storage' },
  { binding: 2, name: 'statics', type: 'read-only-storage' },
  { binding: 3, name: 'dynamics', type: 'storage' },
  { binding: 4, name: 'outputs', type: 'storage' },
  { binding: 5, name: 'iSyn', type: 'storage' },
  { binding: 6, name: 'spikeLog', type: 'storage' },
];
