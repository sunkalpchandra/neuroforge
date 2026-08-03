'use client';

import { useEffect, useState } from 'react';
import type { NeuronModelKind } from '@neuroforge/shared';
import { MODEL_FROM_CODE } from '@neuroforge/shared';

import { getEngine, getProbes } from '@/lib/runtime';

/** A snapshot of one neuron's live state, sampled off the simulation buffers. */
export interface LiveNeuron {
  slot: number;
  model: NeuronModelKind;
  voltage: number;
  adaptation: number;
  calcium: number;
  synapticCurrent: number;
  externalCurrent: number;
  rate: number;
  spikeCount: number;
  lastSpike: number;
  refractoryUntil: number;
  gateM: number;
  gateH: number;
  gateN: number;
  flash: number;
  /** Recent membrane trace, oldest first. */
  trace: Float32Array;
  traceCount: number;
}

const EMPTY_TRACE = new Float32Array(0);

/**
 * Poll one neuron's state at a fixed rate.
 *
 * The buffers change every frame, but a panel that re-rendered every frame
 * would cost more than the simulation it displays. Fifteen hertz is fast enough
 * to read as live and slow enough to be free.
 */
export function useLiveNeuron(slot: number | null, hz = 15): LiveNeuron | null {
  const [state, setState] = useState<LiveNeuron | null>(null);

  useEffect(() => {
    if (slot === null || slot < 0) {
      setState(null);
      return;
    }

    const probes = getProbes();
    probes.track(slot, 'voltage');

    const sample = () => {
      const engine = getEngine();
      const { neurons } = engine.buffers;
      if (slot >= neurons.count) {
        setState(null);
        return;
      }
      const trace = probes.read(slot, 'voltage');
      setState({
        slot,
        model: MODEL_FROM_CODE[neurons.model[slot]] ?? 'lif',
        voltage: neurons.v[slot],
        adaptation: neurons.w[slot],
        calcium: neurons.calcium[slot],
        synapticCurrent: neurons.iSyn[slot],
        externalCurrent: neurons.iExt[slot] + neurons.bias[slot],
        rate: neurons.rate[slot],
        spikeCount: neurons.spikeCount[slot],
        lastSpike: neurons.lastSpike[slot],
        refractoryUntil: neurons.refractoryUntil[slot],
        gateM: neurons.gateM[slot],
        gateH: neurons.gateH[slot],
        gateN: neurons.gateN[slot],
        flash: neurons.flash[slot],
        trace: trace?.values ?? EMPTY_TRACE,
        traceCount: trace?.count ?? 0,
      });
    };

    sample();
    const id = setInterval(sample, Math.max(16, 1000 / hz));
    return () => {
      clearInterval(id);
      probes.untrack(slot, 'voltage');
    };
  }, [slot, hz]);

  return state;
}
