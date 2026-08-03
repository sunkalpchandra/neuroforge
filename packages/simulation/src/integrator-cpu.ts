import {
  MODEL_CODE,
  PARAM_SLOT,
  PLASTICITY_CODE,
  RECEPTOR_CODE,
  SYNAPSE_PARAM_STRIDE,
  SYN_PARAM_SLOT,
  NEURON_PARAM_STRIDE,
} from '@neuroforge/shared';
import type { SimulationBuffers, SimulationSettings } from '@neuroforge/shared';
import { Rng } from '@neuroforge/math';

import { Adjacency } from './adjacency';
import { createCursor, drain, maxDelay, schedule } from './delay-queue';
import type { DelayCursor } from './delay-queue';
import {
  hhSteadyState,
  mlSteadyState,
  stepAdEx,
  stepHodgkinHuxley,
  stepIzhikevich,
  stepLif,
  stepMorrisLecar,
} from './models';
import type { Integrator, StepResult } from './types';

/** Time constant of the displayed firing-rate estimate (ms). */
const RATE_TAU = 120;

/** Time constant of the visual spike envelope (ms). */
const FLASH_TAU = 45;

/** Calcium indicator time constant (ms), chosen to look like a GCaMP trace. */
const CALCIUM_TAU = 220;

/** Below this conductance a synapse contributes nothing worth the arithmetic. */
const CONDUCTANCE_EPSILON = 1e-7;

/** Minimum interval between detected spikes for edge-detected models (ms). */
const EDGE_REFRACTORY = 1;

/**
 * Peak-normalisation for a dual-exponential conductance.
 *
 * Injecting the weight directly into both state variables would make the actual
 * peak conductance depend on the ratio of the time constants, so changing a
 * synapse's kinetics would silently change its strength. This scales the
 * injection so the peak of (gDecay - gRise) equals the weight.
 */
function dualExponentialNorm(tauRise: number, tauDecay: number): number {
  if (tauDecay <= 0) return 1;
  if (tauRise <= 0 || Math.abs(tauDecay - tauRise) < 1e-6) return 1;
  const tPeak = ((tauRise * tauDecay) / (tauDecay - tauRise)) * Math.log(tauDecay / tauRise);
  const peak = Math.exp(-tPeak / tauDecay) - Math.exp(-tPeak / tauRise);
  return Math.abs(peak) < 1e-9 ? 1 : 1 / peak;
}

/**
 * Jahr-Stevens magnesium block. NMDA conductance is gated by postsynaptic
 * depolarisation, which is what makes an NMDA synapse a coincidence detector
 * rather than a slow AMPA synapse.
 */
function magnesiumBlock(v: number, strength: number): number {
  return 1 / (1 + strength * 0.28 * Math.exp(-0.062 * v));
}

/**
 * Whether a synapse's kinetics collapse to a single exponential.
 *
 * The dual-exponential waveform is the difference of two decaying states, so
 * equal time constants make it identically zero for every input — the synapse
 * would transmit nothing at all rather than transmitting sharply. In that case
 * the rise is treated as instantaneous and only the decay state is driven.
 */
function isSingleExponential(tauRise: number, tauDecay: number): boolean {
  return tauRise <= 0 || Math.abs(tauDecay - tauRise) < 1e-6;
}

/**
 * The reference integrator.
 *
 * This is the numerical ground truth for the product: the WASM and WebGPU
 * backends are validated against it, and where they disagree this one is right.
 * It is written for clarity of the numerics first and speed second, but it still
 * avoids all allocation inside the step loop — every scratch value is a field.
 */
export class CpuIntegrator implements Integrator {
  readonly backend = 'cpu' as const;

  private rng: Rng;
  private adjacency = new Adjacency();
  private cursor: DelayCursor | null = null;

  /** Previous membrane voltage, for rising-edge spike detection. */
  private previousV = new Float32Array(0);

  /** Events discarded because a delay bucket was full, since the last reset. */
  private droppedEvents = 0;

  private hhScratch = { v: 0, m: 0, h: 0, n: 0 };
  private pairScratch = { v: 0, w: 0 };
  private izhScratch = { v: 0, u: 0 };

  constructor(seed = 0x9e3779b9) {
    this.rng = new Rng(seed);
  }

  invalidate(): void {
    this.adjacency.invalidate();
  }

  get dropped(): number {
    return this.droppedEvents;
  }

  private ensureScratch(capacity: number): void {
    if (this.previousV.length >= capacity) return;
    const next = new Float32Array(capacity);
    next.set(this.previousV);
    this.previousV = next;
  }

  step(buffers: SimulationBuffers, settings: SimulationSettings, steps: number): StepResult {
    const started = performance.now();
    const { neurons, synapses, delays, spikes } = buffers;
    const neuronCount = neurons.count;
    if (neuronCount === 0 || steps <= 0) {
      return { steps: 0, spikes: 0, simMs: 0 };
    }

    this.adjacency.ensure(buffers);
    this.ensureScratch(neurons.capacity);
    if (this.cursor === null) this.cursor = createCursor(buffers.time, delays);

    const dt = settings.dt;
    const gain = settings.gain;
    const horizon = maxDelay(delays);
    const plasticityOn = settings.plasticityEnabled;
    const globalNoise = settings.noise;

    // White noise scales as 1/sqrt(dt) so that the variance a neuron accumulates
    // over a millisecond does not depend on the timestep. Without this, halving
    // dt would halve the effective noise and quietly change the answer.
    const noiseScale = 1 / Math.sqrt(dt);

    const synCount = synapses.count;
    let totalSpikes = 0;

    for (let s = 0; s < steps; s += 1) {
      buffers.time += dt;
      const time = buffers.time;

      // ---- 1. Deliver synaptic events whose conduction delay has elapsed ----
      drain(delays, this.cursor, time, (synapse, amplitude) => {
        synapses.gDecay[synapse] += amplitude;
        if (!isSingleExponential(synapses.tauRise[synapse], synapses.tauDecay[synapse])) {
          synapses.gRise[synapse] += amplitude;
        }
        synapses.activity[synapse] = 1;
      });

      // ---- 2. Decay conductances and accumulate postsynaptic current --------
      neurons.iSyn.fill(0, 0, neuronCount);

      for (let i = 0; i < synCount; i += 1) {
        if (synapses.enabled[i] === 0) continue;

        // Electrical coupling is continuous, not event-driven: current flows
        // whenever the two membranes differ in potential, with no presynaptic
        // spike required. Routing gap junctions through the conductance state
        // machine would make them silent between spikes and, with equal rise
        // and decay constants, silent during them as well.
        if (synapses.receptor[i] === RECEPTOR_CODE.gap) {
          const pre = synapses.pre[i];
          const post = synapses.post[i];
          if (pre < neuronCount && post < neuronCount) {
            const delta = neurons.v[pre] - neurons.v[post];
            neurons.iSyn[post] += synapses.weight[i] * delta * gain;
            synapses.activity[i] = Math.min(1, Math.abs(delta) * 0.02);
          }
          continue;
        }

        const tauRise = synapses.tauRise[i];
        const tauDecay = synapses.tauDecay[i];
        let gRise = synapses.gRise[i];
        let gDecay = synapses.gDecay[i];

        if (tauRise > 0) gRise *= Math.exp(-dt / tauRise);
        else gRise = 0;
        if (tauDecay > 0) gDecay *= Math.exp(-dt / tauDecay);
        else gDecay = 0;

        synapses.gRise[i] = gRise;
        synapses.gDecay[i] = gDecay;

        // Short-term plasticity resources recover toward 1 between spikes.
        const pBase = i * SYNAPSE_PARAM_STRIDE;
        const tauRec = synapses.params[pBase + SYN_PARAM_SLOT.STP_TAU_REC];
        const stpU = synapses.params[pBase + SYN_PARAM_SLOT.STP_U];
        if (stpU > 0 && tauRec > 0) {
          const r = synapses.stpR[i];
          synapses.stpR[i] = 1 + (r - 1) * Math.exp(-dt / tauRec);
          const tauFacil = synapses.params[pBase + SYN_PARAM_SLOT.STP_TAU_FACIL];
          if (tauFacil > 0) {
            synapses.stpU[i] = stpU + (synapses.stpU[i] - stpU) * Math.exp(-dt / tauFacil);
          }
        }

        const g = gDecay - gRise;
        synapses.activity[i] *= Math.exp(-dt / 60);

        if (g <= CONDUCTANCE_EPSILON) continue;

        const post = synapses.post[i];
        if (post >= neuronCount) continue;
        const vPost = neurons.v[post];

        const mg = synapses.mgBlock[i];
        const effective = mg > 0 ? g * magnesiumBlock(vPost, mg) : g;

        neurons.iSyn[post] += effective * (synapses.eRev[i] - vPost) * gain;
      }

      // ---- 3. Integrate membranes and detect spikes ------------------------
      for (let i = 0; i < neuronCount; i += 1) {
        if (neurons.enabled[i] === 0) {
          neurons.spike[i] = 0;
          continue;
        }

        const pBase = i * NEURON_PARAM_STRIDE;
        const noiseAmp = neurons.noise[i] + globalNoise;
        let current = neurons.iSyn[i] + neurons.iExt[i] + neurons.bias[i];
        if (noiseAmp > 0) current += this.rng.normal(0, noiseAmp) * noiseScale;

        const model = neurons.model[i];
        const vBefore = neurons.v[i];
        let fired = false;

        switch (model) {
          case MODEL_CODE.lif: {
            if (time < neurons.refractoryUntil[i]) {
              neurons.v[i] = neurons.params[pBase + PARAM_SLOT.LIF_VRESET];
              break;
            }
            const v = stepLif(vBefore, current, dt, neurons.params, pBase);
            if (v >= neurons.params[pBase + PARAM_SLOT.LIF_VTHRESH]) {
              neurons.v[i] = neurons.params[pBase + PARAM_SLOT.LIF_VRESET];
              neurons.refractoryUntil[i] = time + neurons.params[pBase + PARAM_SLOT.LIF_TREFRACT];
              fired = true;
            } else {
              neurons.v[i] = v;
            }
            break;
          }

          case MODEL_CODE.izhikevich: {
            this.izhScratch.v = vBefore;
            this.izhScratch.u = neurons.w[i];
            stepIzhikevich(this.izhScratch, current, dt, neurons.params, pBase);
            if (this.izhScratch.v >= neurons.params[pBase + PARAM_SLOT.IZH_VPEAK]) {
              neurons.v[i] = neurons.params[pBase + PARAM_SLOT.IZH_C];
              neurons.w[i] = this.izhScratch.u + neurons.params[pBase + PARAM_SLOT.IZH_D];
              fired = true;
            } else {
              neurons.v[i] = this.izhScratch.v;
              neurons.w[i] = this.izhScratch.u;
            }
            break;
          }

          case MODEL_CODE['hodgkin-huxley']: {
            this.hhScratch.v = vBefore;
            this.hhScratch.m = neurons.gateM[i];
            this.hhScratch.h = neurons.gateH[i];
            this.hhScratch.n = neurons.gateN[i];
            stepHodgkinHuxley(this.hhScratch, current, dt, neurons.params, pBase);
            neurons.v[i] = this.hhScratch.v;
            neurons.gateM[i] = this.hhScratch.m;
            neurons.gateH[i] = this.hhScratch.h;
            neurons.gateN[i] = this.hhScratch.n;

            // Hodgkin-Huxley has no reset: a spike is the upward crossing of the
            // detection voltage, which must be edge-triggered or one action
            // potential registers as dozens across its width.
            const detect = neurons.params[pBase + PARAM_SLOT.HH_VDETECT];
            if (
              this.hhScratch.v >= detect &&
              this.previousV[i] < detect &&
              time - neurons.lastSpike[i] > EDGE_REFRACTORY
            ) {
              fired = true;
            }
            break;
          }

          case MODEL_CODE.adex: {
            if (time < neurons.refractoryUntil[i]) {
              neurons.v[i] = neurons.params[pBase + PARAM_SLOT.ADEX_VRESET];
              break;
            }
            this.pairScratch.v = vBefore;
            this.pairScratch.w = neurons.w[i];
            stepAdEx(this.pairScratch, current, dt, neurons.params, pBase);
            if (this.pairScratch.v >= neurons.params[pBase + PARAM_SLOT.ADEX_VPEAK]) {
              neurons.v[i] = neurons.params[pBase + PARAM_SLOT.ADEX_VRESET];
              neurons.w[i] = this.pairScratch.w + neurons.params[pBase + PARAM_SLOT.ADEX_B];
              neurons.refractoryUntil[i] = time + neurons.params[pBase + PARAM_SLOT.ADEX_TREFRACT];
              fired = true;
            } else {
              neurons.v[i] = this.pairScratch.v;
              neurons.w[i] = this.pairScratch.w;
            }
            break;
          }

          case MODEL_CODE['morris-lecar']: {
            this.pairScratch.v = vBefore;
            this.pairScratch.w = neurons.w[i];
            stepMorrisLecar(this.pairScratch, current, dt, neurons.params, pBase);
            neurons.v[i] = this.pairScratch.v;
            neurons.w[i] = this.pairScratch.w;
            const detect = neurons.params[pBase + PARAM_SLOT.ML_VDETECT];
            if (
              this.pairScratch.v >= detect &&
              this.previousV[i] < detect &&
              time - neurons.lastSpike[i] > EDGE_REFRACTORY
            ) {
              fired = true;
            }
            break;
          }

          default:
            break;
        }

        this.previousV[i] = neurons.v[i];

        // Calcium tracks spiking with a slow decay; it is a display signal, not
        // a term in any equation, which is why it is integrated here rather than
        // inside the membrane models.
        neurons.calcium[i] *= Math.exp(-dt / CALCIUM_TAU);
        neurons.flash[i] *= Math.exp(-dt / FLASH_TAU);
        neurons.rate[i] += ((fired ? 1000 / dt : 0) - neurons.rate[i]) * (dt / RATE_TAU);

        if (!fired) {
          neurons.spike[i] = 0;
          continue;
        }

        neurons.spike[i] = 1;
        neurons.lastSpike[i] = time;
        neurons.spikeCount[i] += 1;
        neurons.flash[i] = 1;
        neurons.calcium[i] += 1;
        totalSpikes += 1;

        const slot = spikes.head % spikes.capacity;
        spikes.neuron[slot] = i;
        spikes.time[slot] = time;
        spikes.head += 1;

        this.emit(buffers, i, time, horizon, plasticityOn);
      }

      // ---- 4. Postsynaptic plasticity and trace decay ----------------------
      if (plasticityOn) this.decayTraces(buffers, dt);
    }

    return {
      steps,
      spikes: totalSpikes,
      simMs: performance.now() - started,
    };
  }

  /**
   * Schedule a spike down every outgoing synapse and apply the presynaptic half
   * of the plasticity rules.
   */
  private emit(
    buffers: SimulationBuffers,
    neuron: number,
    time: number,
    horizon: number,
    plasticityOn: boolean,
  ): void {
    const { synapses } = buffers;
    const end = this.adjacency.outEnd(neuron);

    for (let k = this.adjacency.outBegin(neuron); k < end; k += 1) {
      const s = this.adjacency.outAt(k);
      if (synapses.enabled[s] === 0) continue;
      // Electrical synapses carry current continuously and are handled in the
      // conductance pass; queueing a spike event for them would accumulate
      // conductance nothing ever consumes.
      if (synapses.receptor[s] === RECEPTOR_CODE.gap) continue;

      // Stochastic vesicle release.
      const p = synapses.releaseProb[s];
      if (p < 1 && this.rng.next() > p) continue;

      let amplitude = synapses.weight[s];

      const pBase = s * SYNAPSE_PARAM_STRIDE;
      const baseU = synapses.params[pBase + SYN_PARAM_SLOT.STP_U];
      if (baseU > 0) {
        const tauFacil = synapses.params[pBase + SYN_PARAM_SLOT.STP_TAU_FACIL];
        let u = tauFacil > 0 ? synapses.stpU[s] : baseU;
        if (tauFacil > 0) {
          u += baseU * (1 - u);
          synapses.stpU[s] = u;
        }
        const r = synapses.stpR[s];
        const released = u * r;
        synapses.stpR[s] = r - released;
        // Divide by the baseline utilisation so that a synapse at full resources
        // delivers its nominal weight rather than a fraction of it.
        amplitude *= released / baseU;
      }

      amplitude *= dualExponentialNorm(synapses.tauRise[s], synapses.tauDecay[s]);

      const delay = Math.min(synapses.delay[s], horizon);
      if (!schedule(buffers.delays, s, amplitude, time + delay)) {
        this.droppedEvents += 1;
      }

      if (plasticityOn && synapses.plasticity[s] !== PLASTICITY_CODE.static) {
        this.onPresynapticSpike(buffers, s);
      }
    }

    if (!plasticityOn) return;

    const inEnd = this.adjacency.inEnd(neuron);
    for (let k = this.adjacency.inBegin(neuron); k < inEnd; k += 1) {
      const s = this.adjacency.inAt(k);
      if (synapses.enabled[s] === 0) continue;
      if (synapses.plasticity[s] === PLASTICITY_CODE.static) continue;
      this.onPostsynapticSpike(buffers, s);
    }
  }

  /** Pre-before-post is depression: the postsynaptic trace is what is read. */
  private onPresynapticSpike(buffers: SimulationBuffers, s: number): void {
    const { synapses } = buffers;
    const p = s * SYNAPSE_PARAM_STRIDE;
    const rule = synapses.plasticity[s];
    const lr = synapses.params[p + SYN_PARAM_SLOT.LEARNING_RATE];
    const aMinus = synapses.params[p + SYN_PARAM_SLOT.A_MINUS];

    let delta = -aMinus * synapses.postTrace[s];
    if (rule === PLASTICITY_CODE['triplet-stdp']) {
      delta -= aMinus * synapses.postTrace[s] * synapses.preTraceSlow[s];
    }

    synapses.preTrace[s] += 1;
    synapses.preTraceSlow[s] += 1;
    this.applyWeightDelta(buffers, s, delta * lr);
  }

  /** Post-after-pre is potentiation: the presynaptic trace is what is read. */
  private onPostsynapticSpike(buffers: SimulationBuffers, s: number): void {
    const { synapses } = buffers;
    const p = s * SYNAPSE_PARAM_STRIDE;
    const rule = synapses.plasticity[s];
    const lr = synapses.params[p + SYN_PARAM_SLOT.LEARNING_RATE];
    const aPlus = synapses.params[p + SYN_PARAM_SLOT.A_PLUS];

    let delta = aPlus * synapses.preTrace[s];
    if (rule === PLASTICITY_CODE['triplet-stdp']) {
      delta += aPlus * synapses.preTrace[s] * synapses.postTraceSlow[s];
    } else if (rule === PLASTICITY_CODE.hebbian) {
      delta = aPlus * synapses.preTrace[s];
    } else if (rule === PLASTICITY_CODE.oja) {
      // Oja subtracts a decay proportional to the square of the postsynaptic
      // activity, which normalises the weight vector instead of letting it grow
      // without bound the way plain Hebbian learning does.
      const y = synapses.postTrace[s];
      delta = aPlus * (synapses.preTrace[s] - y * y * synapses.weight[s]);
    }

    synapses.postTrace[s] += 1;
    synapses.postTraceSlow[s] += 1;
    this.applyWeightDelta(buffers, s, delta * lr);
  }

  private applyWeightDelta(buffers: SimulationBuffers, s: number, delta: number): void {
    const { synapses } = buffers;
    const p = s * SYNAPSE_PARAM_STRIDE;
    const wMin = synapses.params[p + SYN_PARAM_SLOT.W_MIN];
    const wMax = synapses.params[p + SYN_PARAM_SLOT.W_MAX];
    const next = synapses.weight[s] + delta;
    synapses.weight[s] = next < wMin ? wMin : next > wMax ? wMax : next;
  }

  private decayTraces(buffers: SimulationBuffers, dt: number): void {
    const { synapses } = buffers;
    const count = synapses.count;
    for (let s = 0; s < count; s += 1) {
      if (synapses.plasticity[s] === PLASTICITY_CODE.static) continue;
      const p = s * SYNAPSE_PARAM_STRIDE;
      const tauPlus = synapses.params[p + SYN_PARAM_SLOT.TAU_PLUS];
      const tauMinus = synapses.params[p + SYN_PARAM_SLOT.TAU_MINUS];
      const tauX = synapses.params[p + SYN_PARAM_SLOT.TAU_X];
      const tauY = synapses.params[p + SYN_PARAM_SLOT.TAU_Y];
      if (tauPlus > 0) synapses.preTrace[s] *= Math.exp(-dt / tauPlus);
      if (tauMinus > 0) synapses.postTrace[s] *= Math.exp(-dt / tauMinus);
      if (tauX > 0) synapses.preTraceSlow[s] *= Math.exp(-dt / tauX);
      if (tauY > 0) synapses.postTraceSlow[s] *= Math.exp(-dt / tauY);
    }
  }

  /**
   * Return every neuron to its resting state without changing topology.
   * Gating variables are set to their steady values at rest rather than to zero,
   * because a Hodgkin-Huxley neuron starting with all gates closed fires a
   * spurious onset spike as they equilibrate.
   */
  reset(buffers: SimulationBuffers): void {
    const { neurons, synapses, delays, spikes } = buffers;
    const count = neurons.count;
    this.ensureScratch(neurons.capacity);

    for (let i = 0; i < count; i += 1) {
      const pBase = i * NEURON_PARAM_STRIDE;
      let rest: number;
      switch (neurons.model[i]) {
        case MODEL_CODE.lif:
          rest = neurons.params[pBase + PARAM_SLOT.LIF_EL];
          neurons.w[i] = 0;
          break;
        case MODEL_CODE.izhikevich:
          rest = neurons.params[pBase + PARAM_SLOT.IZH_C];
          neurons.w[i] = neurons.params[pBase + PARAM_SLOT.IZH_B] * rest;
          break;
        case MODEL_CODE['hodgkin-huxley']: {
          rest = -65;
          const gates = hhSteadyState(rest);
          neurons.gateM[i] = gates.m;
          neurons.gateH[i] = gates.h;
          neurons.gateN[i] = gates.n;
          neurons.w[i] = 0;
          break;
        }
        case MODEL_CODE.adex:
          rest = neurons.params[pBase + PARAM_SLOT.ADEX_EL];
          neurons.w[i] = 0;
          break;
        case MODEL_CODE['morris-lecar']:
          rest = neurons.params[pBase + PARAM_SLOT.ML_EL];
          neurons.w[i] = mlSteadyState(
            rest,
            neurons.params[pBase + PARAM_SLOT.ML_V3],
            neurons.params[pBase + PARAM_SLOT.ML_V4],
          );
          break;
        default:
          rest = -70;
          break;
      }
      neurons.v[i] = rest;
      this.previousV[i] = rest;
      neurons.iSyn[i] = 0;
      neurons.iExt[i] = 0;
      neurons.spike[i] = 0;
      neurons.lastSpike[i] = -Infinity;
      neurons.refractoryUntil[i] = 0;
      neurons.flash[i] = 0;
      neurons.rate[i] = 0;
      neurons.calcium[i] = 0;
      neurons.spikeCount[i] = 0;
    }

    const synCount = synapses.count;
    for (let s = 0; s < synCount; s += 1) {
      synapses.gRise[s] = 0;
      synapses.gDecay[s] = 0;
      synapses.preTrace[s] = 0;
      synapses.postTrace[s] = 0;
      synapses.preTraceSlow[s] = 0;
      synapses.postTraceSlow[s] = 0;
      synapses.stpR[s] = 1;
      synapses.stpU[s] = 0;
      synapses.activity[s] = 0;
    }

    delays.counts.fill(0);
    spikes.head = 0;
    buffers.time = 0;
    buffers.step = 0;
    this.cursor = createCursor(0, delays);
    this.droppedEvents = 0;
    this.adjacency.invalidate();
  }

  dispose(): void {
    this.previousV = new Float32Array(0);
    this.cursor = null;
  }
}
