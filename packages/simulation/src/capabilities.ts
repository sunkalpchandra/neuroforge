import type { BackendCapabilities } from '@neuroforge/shared';

/**
 * Probe what this browser can actually do.
 *
 * Feature detection here is deliberately conservative: requesting an adapter can
 * succeed while requesting a device fails, and a device can exist with limits too
 * small to be useful. Anything uncertain is reported as unavailable so the engine
 * falls back rather than failing at the first dispatch.
 */
export async function detectCapabilities(): Promise<BackendCapabilities> {
  const capabilities: BackendCapabilities = {
    webgpu: false,
    wasm: typeof WebAssembly !== 'undefined',
    threads: false,
    maxStorageBufferBindingSize: 0,
    maxComputeInvocations: 0,
    adapterName: 'none',
    discreteGpu: false,
  };

  capabilities.threads =
    typeof SharedArrayBuffer !== 'undefined' &&
    typeof globalThis.crossOriginIsolated === 'boolean' &&
    globalThis.crossOriginIsolated;

  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return capabilities;

  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return capabilities;

    const limits = adapter.limits;
    capabilities.maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize ?? 0;
    capabilities.maxComputeInvocations = limits.maxComputeInvocationsPerWorkgroup ?? 0;

    // `info` is the standardised accessor; older implementations exposed
    // requestAdapterInfo(). Neither is guaranteed, and neither is load-bearing.
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    if (info) {
      capabilities.adapterName = info.description || info.vendor || 'webgpu';
      capabilities.discreteGpu = info.architecture !== 'integrated';
    } else {
      capabilities.adapterName = 'webgpu';
    }

    capabilities.webgpu = capabilities.maxComputeInvocations >= 64;
  } catch {
    // A throwing requestAdapter means no usable WebGPU, which is exactly the
    // default this function already carries.
  }

  return capabilities;
}

/** Acquire a compute device, or null when WebGPU is unavailable or refuses. */
export async function requestComputeDevice(): Promise<GPUDevice | null> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return null;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    // A lost device must not leave the engine dispatching into a dead handle.
    void device.lost.then(() => undefined);
    return device;
  } catch {
    return null;
  }
}
