import type { SimulationBuffers } from '@neuroforge/shared';

/** Signals a probe can record. */
export type ProbeSignal =
  | 'voltage'
  | 'current'
  | 'conductance'
  | 'calcium'
  | 'adaptation'
  | 'spikes';

interface Trace {
  values: Float32Array;
  times: Float32Array;
  /** Total samples written; the ring wraps once this exceeds capacity. */
  written: number;
}

function traceKey(slot: number, signal: string): string {
  return `${slot}:${signal}`;
}

/**
 * Ring buffers holding recent traces for the inspector plots.
 *
 * Sampling happens once per rendered frame rather than once per integration
 * substep. At a 0.1 ms timestep a substep-rate trace would be ten thousand
 * samples per simulated second per neuron, which no plot can show and no
 * browser should hold.
 */
export class ProbeRecorder {
  private traces = new Map<string, Trace>();
  private readonly capacity: number;

  constructor(capacity = 2048) {
    this.capacity = Math.max(16, capacity);
  }

  track(slot: number, signal: string): void {
    const key = traceKey(slot, signal);
    if (this.traces.has(key)) return;
    this.traces.set(key, {
      values: new Float32Array(this.capacity),
      times: new Float32Array(this.capacity),
      written: 0,
    });
  }

  untrack(slot: number, signal: string): void {
    this.traces.delete(traceKey(slot, signal));
  }

  get tracked(): number {
    return this.traces.size;
  }

  /** Append one sample to every tracked trace. */
  sample(buffers: SimulationBuffers): void {
    if (this.traces.size === 0) return;
    const { neurons } = buffers;
    const time = buffers.time;

    for (const [key, trace] of this.traces) {
      const separator = key.indexOf(':');
      const slot = Number.parseInt(key.slice(0, separator), 10);
      if (slot < 0 || slot >= neurons.count) continue;
      const signal = key.slice(separator + 1) as ProbeSignal;

      let value: number;
      switch (signal) {
        case 'voltage':
          value = neurons.v[slot];
          break;
        case 'current':
          value = neurons.iSyn[slot] + neurons.iExt[slot] + neurons.bias[slot];
          break;
        case 'conductance':
          // Reported as the instantaneous synaptic drive divided by the driving
          // force, which is the quantity an experimenter would call conductance.
          value = neurons.iSyn[slot];
          break;
        case 'calcium':
          value = neurons.calcium[slot];
          break;
        case 'adaptation':
          value = neurons.w[slot];
          break;
        case 'spikes':
          value = neurons.spike[slot];
          break;
        default:
          value = 0;
          break;
      }

      const index = trace.written % this.capacity;
      trace.values[index] = value;
      trace.times[index] = time;
      trace.written += 1;
    }
  }

  /**
   * Read a trace in chronological order.
   *
   * The ring is unrolled into a fresh pair of arrays so callers never have to
   * reason about the write cursor. Plots re-read on every frame they are
   * visible, which is a few thousand floats — cheap next to the copy-free
   * alternative's complexity.
   */
  read(slot: number, signal: string): { values: Float32Array; times: Float32Array; count: number } | null {
    const trace = this.traces.get(traceKey(slot, signal));
    if (!trace) return null;

    const count = Math.min(trace.written, this.capacity);
    const values = new Float32Array(count);
    const times = new Float32Array(count);
    const start = trace.written > this.capacity ? trace.written % this.capacity : 0;

    for (let i = 0; i < count; i += 1) {
      const index = (start + i) % this.capacity;
      values[i] = trace.values[index];
      times[i] = trace.times[index];
    }

    return { values, times, count };
  }

  clear(): void {
    for (const trace of this.traces.values()) {
      trace.written = 0;
      trace.values.fill(0);
      trace.times.fill(0);
    }
  }

  reset(): void {
    this.traces.clear();
  }
}
