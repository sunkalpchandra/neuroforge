import type { ComputeBackend, SimulationBuffers, SimulationSettings } from '@neuroforge/shared';

/** What one call to `step` actually did. */
export interface StepResult {
  /** Substeps executed. May be fewer than requested if the backend bailed out. */
  steps: number;
  /** Spikes emitted across all substeps. */
  spikes: number;
  /** Wall-clock milliseconds spent integrating. */
  simMs: number;
}

export interface Integrator {
  readonly backend: ComputeBackend;

  /** Advance the network by `steps` substeps of `settings.dt`. */
  step(buffers: SimulationBuffers, settings: SimulationSettings, steps: number): StepResult;

  /** Return every neuron to rest without changing topology. */
  reset(buffers: SimulationBuffers): void;

  /**
   * Discard any cached derivation of the network's topology.
   *
   * Backends cache adjacency, GPU bind groups and WASM-side mirrors of the
   * synapse list. The engine calls this after any structural edit; a backend
   * with nothing cached may leave it undefined.
   */
  invalidate?(): void;

  dispose(): void;
}

export const EMPTY_STEP_RESULT: StepResult = { steps: 0, spikes: 0, simMs: 0 };
