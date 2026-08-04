import {
  DEFAULT_LIF,
  DEFAULT_SIMULATION_SETTINGS,
  MODEL_CODE,
  RECEPTOR_CODE,
  RECEPTOR_DEFAULTS,
  allocateSimulationBuffers,
  packNeuronParams,
} from '@neuroforge/shared';

import { CpuIntegrator } from './integrator-cpu';
import type { Integrator } from './types';

/**
 * Prove a backend works before trusting it with the user's network.
 *
 * A backend that silently produces nothing is worse than one that fails to load:
 * the app looks alive — time advances, frames render — while every readout says
 * the network is dead, and there is nothing on screen to suggest the compute
 * path is at fault rather than the circuit. Loading successfully is not evidence
 * of working, so each candidate integrates a small network whose behaviour the
 * reference implementation defines, and is rejected unless it agrees.
 *
 * The probe is deliberately tiny and runs once at startup; the cost is
 * negligible against being wrong about which backend is live.
 */

/** Simulated milliseconds the probe runs for. */
const PROBE_MS = 200;
const PROBE_DT = 0.05;

/** Driver plus two followers, wired strongly enough that silence is unambiguous. */
function buildProbe() {
  const buffers = allocateSimulationBuffers(8, 8);
  buffers.neurons.count = 3;
  buffers.synapses.count = 2;

  for (let i = 0; i < 3; i += 1) {
    buffers.neurons.model[i] = MODEL_CODE.lif;
    packNeuronParams(DEFAULT_LIF, buffers.neurons.params, i);
  }
  // Well above the LIF rheobase of gL*(vThresh - eL) = 200 pA, so the driver
  // must fire on any correct implementation.
  buffers.neurons.bias[0] = 400;

  const kinetics = RECEPTOR_DEFAULTS.ampa;
  for (let s = 0; s < 2; s += 1) {
    buffers.synapses.pre[s] = 0;
    buffers.synapses.post[s] = s + 1;
    buffers.synapses.weight[s] = 120;
    buffers.synapses.delay[s] = 1;
    buffers.synapses.tauRise[s] = kinetics.tauRise;
    buffers.synapses.tauDecay[s] = kinetics.tauDecay;
    buffers.synapses.eRev[s] = kinetics.eRev;
    buffers.synapses.receptor[s] = RECEPTOR_CODE.ampa;
    buffers.synapses.enabled[s] = 1;
  }
  return buffers;
}

export interface BackendProbe {
  ok: boolean;
  reason: string;
  spikes: number;
  referenceSpikes: number;
}

/**
 * Run the probe through `candidate` and compare against the CPU reference.
 *
 * Only gross disagreement is rejected. The backends integrate in different
 * precisions and orders, so demanding identical spike counts would reject a
 * correct implementation over one boundary-straddling spike; producing none at
 * all, or an order of magnitude too many, is what this is looking for.
 */
export function probeBackend(candidate: Integrator): BackendProbe {
  const settings = { ...DEFAULT_SIMULATION_SETTINGS, dt: PROBE_DT, noise: 0 };
  const steps = Math.round(PROBE_MS / PROBE_DT);

  const reference = new CpuIntegrator(1234);
  const referenceBuffers = buildProbe();
  reference.reset(referenceBuffers);
  const referenceSpikes = reference.step(referenceBuffers, settings, steps).spikes;
  reference.dispose();

  let spikes: number;
  try {
    const buffers = buildProbe();
    candidate.invalidate?.();
    candidate.reset(buffers);
    spikes = candidate.step(buffers, settings, steps).spikes;

    // A backend that reports spikes but never writes them into the buffers is
    // just as broken from the renderer's point of view, so the columns the scene
    // and the readouts actually read are checked too.
    let rateSum = 0;
    for (let i = 0; i < buffers.neurons.count; i += 1) rateSum += buffers.neurons.rate[i];
    if (spikes > 0 && rateSum <= 0) {
      return {
        ok: false,
        reason: 'reported spikes but left the firing-rate column empty',
        spikes,
        referenceSpikes,
      };
    }
    if (!Number.isFinite(buffers.neurons.v[0])) {
      return { ok: false, reason: 'produced a non-finite membrane voltage', spikes, referenceSpikes };
    }
  } catch (error) {
    return {
      ok: false,
      reason: `threw during the probe: ${error instanceof Error ? error.message : String(error)}`,
      spikes: 0,
      referenceSpikes,
    };
  }

  if (referenceSpikes === 0) {
    // The reference itself is the definition of correct; if it is silent the
    // probe is not measuring anything and the candidate gets the benefit.
    return { ok: true, reason: 'reference produced no spikes; probe inconclusive', spikes, referenceSpikes };
  }
  if (spikes === 0) {
    return { ok: false, reason: 'produced no spikes on a network the reference fires', spikes, referenceSpikes };
  }
  const ratio = spikes / referenceSpikes;
  if (ratio < 0.5 || ratio > 2) {
    return {
      ok: false,
      reason: `fired ${spikes} times where the reference fired ${referenceSpikes}`,
      spikes,
      referenceSpikes,
    };
  }

  return { ok: true, reason: 'agrees with the reference', spikes, referenceSpikes };
}
