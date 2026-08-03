import * as THREE from 'three';
import { COLORS, DEFAULT_RENDER_SETTINGS, hexToInt } from '@neuroforge/shared';

/**
 * Renderer and compute device acquisition.
 *
 * These are two independent decisions. Rendering is WebGL2 because the entire
 * post chain the product's look depends on is a WebGL effect library, and it
 * runs everywhere. Compute is WebGPU when the browser has it, obtained straight
 * from `navigator.gpu` with no canvas involved, because that is where thousands
 * of neurons integrated in parallel actually pays. A machine with WebGPU compute
 * and WebGL2 rendering is the expected configuration, not a fallback.
 *
 * Nothing here throws. A missing adapter, a refused device or a context that
 * cannot be created all degrade instead.
 */

export interface RendererHandle {
  renderer: THREE.WebGLRenderer | object;
  backend: 'webgpu' | 'webgl2';
  device: GPUDevice | null;
}

const MAX_PIXEL_RATIO = 2;

function tuneRenderer(renderer: THREE.WebGLRenderer): void {
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = DEFAULT_RENDER_SETTINGS.exposure;
  renderer.setClearColor(hexToInt(COLORS.background), 1);
  renderer.autoClear = true;
  renderer.shadowMap.enabled = false;
  renderer.info.autoReset = true;
  if (typeof window !== 'undefined') {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
  }
}

function createWebGLRenderer(canvas: HTMLCanvasElement): THREE.WebGLRenderer | null {
  const attempts: THREE.WebGLRendererParameters[] = [
    {
      canvas,
      antialias: true,
      alpha: false,
      depth: true,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
      failIfMajorPerformanceCaveat: false,
    },
    // Second pass drops multisampling and the performance hint, which is what a
    // software or heavily contended context will actually grant.
    { canvas, antialias: false, alpha: false, depth: true, stencil: false },
  ];

  for (const parameters of attempts) {
    try {
      const renderer = new THREE.WebGLRenderer(parameters);
      tuneRenderer(renderer);
      return renderer;
    } catch (error) {
      console.warn('[neuroforge/renderer] WebGL context creation failed', error);
    }
  }
  return null;
}

async function requestComputeDevice(): Promise<GPUDevice | null> {
  if (typeof navigator === 'undefined') return null;
  const gpu = navigator.gpu;
  if (!gpu) return null;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    // A lost device must not surface as an unhandled rejection; the simulation
    // discovers the loss through its own dispatches.
    void device.lost.then(() => undefined);
    return device;
  } catch {
    return null;
  }
}

/**
 * A tuned WebGL2 renderer for the scene plus, separately, a WebGPU device for
 * the integrator. `device` is null when the browser has no WebGPU; `backend`
 * always describes the render path.
 */
export async function createRenderer(canvas: HTMLCanvasElement): Promise<RendererHandle> {
  const [renderer, device] = await Promise.all([
    Promise.resolve(createWebGLRenderer(canvas)),
    requestComputeDevice(),
  ]);

  return {
    renderer: renderer ?? {},
    backend: 'webgl2',
    device,
  };
}
