/** Runtime performance and capability reporting, surfaced in the status bar. */

export type ComputeBackend = 'webgpu' | 'wasm' | 'cpu';

export interface BackendCapabilities {
  webgpu: boolean;
  wasm: boolean;
  /** True when SharedArrayBuffer + cross-origin isolation allow threaded WASM. */
  threads: boolean;
  /** Maximum storage buffer binding size reported by the adapter, in bytes. */
  maxStorageBufferBindingSize: number;
  /** Maximum invocations per compute workgroup. */
  maxComputeInvocations: number;
  adapterName: string;
  /** Whether the WebGPU adapter reports a discrete GPU. */
  discreteGpu: boolean;
}

export interface FrameStats {
  /** Milliseconds spent in the last rendered frame. */
  frameMs: number;
  /** Smoothed frames per second. */
  fps: number;
  /** Milliseconds spent integrating the network in the last frame. */
  simMs: number;
  /** Number of integration substeps executed in the last frame. */
  substeps: number;
  /** Draw calls issued in the last frame. */
  drawCalls: number;
  /** Triangles submitted in the last frame. */
  triangles: number;
  /** Live GPU particles. */
  particles: number;
  /** Spikes delivered in the last frame. */
  spikes: number;
  /** Mean network firing rate in Hz. */
  meanRate: number;
  /** Simulated milliseconds elapsed. */
  simTime: number;
  /** Ratio of simulated time to wall-clock time. */
  realtimeFactor: number;
  neurons: number;
  synapses: number;
  backend: ComputeBackend;
}

export const EMPTY_FRAME_STATS: FrameStats = {
  frameMs: 0,
  fps: 0,
  simMs: 0,
  substeps: 0,
  drawCalls: 0,
  triangles: 0,
  particles: 0,
  spikes: 0,
  meanRate: 0,
  simTime: 0,
  realtimeFactor: 0,
  neurons: 0,
  synapses: 0,
  backend: 'cpu',
};

/** Exponential moving average helper used by the stats collectors. */
export function ema(previous: number, sample: number, alpha = 0.1): number {
  return previous + (sample - previous) * alpha;
}
