import * as THREE from 'three';
import { allocateSimulationBuffers, DEFAULT_RENDER_SETTINGS } from '@neuroforge/shared';
import { GlyphLibrary, NeuronField, AxonField, SpikeParticles, SelectionOverlay, InfiniteGrid, CameraRig } from '@neuroforge/renderer';

/* ------------------------------------------------------------------ */
/* 1. GlyphLibrary eviction while a NeuronField still references it    */
/* ------------------------------------------------------------------ */
console.log('--- library eviction ---');
{
  const disposed = new Set<THREE.BufferGeometry>();
  const origDispose = THREE.BufferGeometry.prototype.dispose;
  THREE.BufferGeometry.prototype.dispose = function patched(this: THREE.BufferGeometry) {
    disposed.add(this);
    return origDispose.call(this);
  };

  const buffers = allocateSimulationBuffers(256, 256);
  const n = buffers.neurons;
  n.count = 210;
  for (let i = 0; i < n.count; i += 1) {
    n.archetype[i] = i % 7;
    n.seed[i] = i;                       // spread across all 3 variants
    n.position[i * 3] = i * 4;
  }

  const small = new GlyphLibrary(4);
  const f = new NeuronField(small);
  f.rebuild(buffers);

  let live = 0;
  let dead = 0;
  for (const child of f.children) {
    const g = (child as THREE.Mesh).geometry;
    const pos = g.attributes.position;
    // find the source geometry this instanced wrapper shares its attributes with
    let isDead = false;
    for (const d of disposed) {
      if (d.attributes.position === pos) { isDead = true; break; }
    }
    if (isDead) dead += 1; else live += 1;
  }
  console.log(`groups=${f.children.length} sharing-live-source=${live} sharing-DISPOSED-source=${dead}`);
  f.dispose();
  small.dispose();
  THREE.BufferGeometry.prototype.dispose = origDispose;
}

/* ------------------------------------------------------------------ */
/* 2. Per-frame allocation, measured per subsystem                     */
/* ------------------------------------------------------------------ */
console.log('--- allocation per subsystem ---');
{
  const N = 4000;
  const S = 16000;
  const buffers = allocateSimulationBuffers(N, S);
  const n = buffers.neurons;
  n.count = N;
  for (let i = 0; i < N; i += 1) {
    n.position[i * 3] = Math.sin(i) * 40;
    n.position[i * 3 + 1] = Math.cos(i * 1.7) * 20;
    n.position[i * 3 + 2] = Math.sin(i * 0.3) * 40;
    n.archetype[i] = i % 7;
    n.seed[i] = i * 2654435761;
  }
  const s = buffers.synapses;
  s.count = S;
  for (let i = 0; i < S; i += 1) {
    s.pre[i] = i % N;
    s.post[i] = (i * 7 + 3) % N;
    s.receptor[i] = i % 5;
  }
  n.flags[3] = 1;
  n.flags[9] = 2;

  const library = new GlyphLibrary();
  const field = new NeuronField(library);
  const axons = new AxonField();
  const particles = new SpikeParticles();
  const overlay = new SelectionOverlay();
  const grid = new InfiniteGrid();
  const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 5000);
  camera.position.set(0, 34, 96);
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  const settings = { ...DEFAULT_RENDER_SETTINGS };

  field.rebuild(buffers);
  axons.rebuild(buffers);
  // keep a steady stream of spikes so the particle pool stays saturated
  const spike = (f: number): void => {
    for (let i = 0; i < N; i += 1) {
      n.spike[i] = (i + f) % 211 === 0 ? 1 : 0;
      if (n.spike[i]) {
        buffers.spikes.neuron[buffers.spikes.head % buffers.spikes.capacity] = i;
        buffers.spikes.time[buffers.spikes.head % buffers.spikes.capacity] = buffers.time;
        buffers.spikes.head += 1;
      }
    }
    buffers.time += 16;
    buffers.step += 1;
  };

  for (let f = 0; f < 60; f += 1) {
    spike(f);
    particles.emitFromSpikes(buffers);
    field.update(buffers, 0.016, settings);
    axons.update(buffers, 0.016, settings);
    particles.update(0.016, settings);
    overlay.update(buffers, 0.016);
    grid.update(camera, settings);
  }

  const measure = (label: string, fn: (f: number) => void): void => {
    global.gc?.(); global.gc?.();
    const before = process.memoryUsage().heapUsed;
    const t0 = performance.now();
    for (let f = 0; f < 500; f += 1) fn(f);
    const t1 = performance.now();
    global.gc?.();
    const after = process.memoryUsage().heapUsed;
    console.log(
      `${label.padEnd(22)} ${((t1 - t0) / 500).toFixed(4)}ms/frame   heap ${(((after - before) / 500)).toFixed(1)} B/frame`,
    );
  };

  measure('spike-log only', (f) => { spike(f); });
  measure('emitFromSpikes', (f) => { spike(f); particles.emitFromSpikes(buffers); });
  measure('NeuronField.update', () => field.update(buffers, 0.016, settings));
  measure('AxonField.update', () => axons.update(buffers, 0.016, settings));
  measure('SpikeParticles.update', () => particles.update(0.016, settings));
  measure('SelectionOverlay.update', () => overlay.update(buffers, 0.016));
  measure('InfiniteGrid.update', () => grid.update(camera, settings));

  field.dispose(); axons.dispose(); particles.dispose(); overlay.dispose(); grid.dispose(); library.dispose();
}

/* ------------------------------------------------------------------ */
/* 3. raycast accuracy on a well-separated grid                        */
/* ------------------------------------------------------------------ */
console.log('--- raycast, sparse grid ---');
{
  const N = 512;
  const buffers = allocateSimulationBuffers(N, 16);
  const n = buffers.neurons;
  n.count = N;
  for (let i = 0; i < N; i += 1) {
    n.position[i * 3] = (i % 8) * 20 - 70;
    n.position[i * 3 + 1] = (Math.floor(i / 8) % 8) * 20 - 70;
    n.position[i * 3 + 2] = Math.floor(i / 64) * 20 - 70;
    n.archetype[i] = i % 7;
    n.seed[i] = i;
  }
  const library = new GlyphLibrary();
  const field = new NeuronField(library);
  field.rebuild(buffers);
  const ray = new THREE.Raycaster();
  let exact = 0;
  let miss = 0;
  for (let i = 0; i < N; i += 1) {
    const t = new THREE.Vector3(n.position[i * 3], n.position[i * 3 + 1], n.position[i * 3 + 2]);
    // come in from an oblique angle so the ray does not skewer a whole column
    const o = t.clone().add(new THREE.Vector3(140, 90, 210));
    ray.set(o, t.clone().sub(o).normalize());
    const slot = field.raycastSlot(ray);
    if (slot === i) exact += 1;
    else if (slot === -1) miss += 1;
  }
  console.log(`sparse grid: exact ${exact}/${N}, missed ${miss}`);
  field.dispose(); library.dispose();
}

/* ------------------------------------------------------------------ */
/* 4. camera: look around in fly mode, then round-trip the state       */
/* ------------------------------------------------------------------ */
console.log('--- camera fly-mode state round trip ---');
{
  type Listener = (e: unknown) => void;
  const makeDom = (): { dom: HTMLElement; fire: (t: string, e: unknown) => void } => {
    const listeners = new Map<string, Listener[]>();
    const dom = {
      style: {} as Record<string, string>,
      addEventListener: (t: string, fn: Listener) => {
        const list = listeners.get(t) ?? [];
        list.push(fn);
        listeners.set(t, list);
      },
      removeEventListener: () => undefined,
      setPointerCapture: () => undefined,
      releasePointerCapture: () => undefined,
      hasPointerCapture: () => false,
      requestPointerLock: () => undefined,
    } as unknown as HTMLElement;
    return {
      dom,
      fire: (t, e) => { for (const fn of listeners.get(t) ?? []) fn(e); },
    };
  };

  for (const mode of ['fly', 'first-person'] as const) {
    const { dom, fire } = makeDom();
    const cam = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 5000);
    cam.position.set(0, 34, 96);
    const rig = new CameraRig(cam, dom);
    rig.mode = mode;
    for (let f = 0; f < 20; f += 1) rig.update(0.016);

    // drag to look around
    fire('pointerdown', { pointerId: 1, button: 0, clientX: 0, clientY: 0, preventDefault() {} });
    for (let k = 1; k <= 40; k += 1) {
      fire('pointermove', { pointerId: 1, clientX: k * 6, clientY: k * 2, shiftKey: false });
      rig.update(0.016);
    }
    fire('pointerup', { pointerId: 1 });
    for (let f = 0; f < 200; f += 1) rig.update(0.016);

    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);
    const state = rig.getState();

    const cam2 = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 5000);
    const rig2 = new CameraRig(cam2, makeDom().dom);
    rig2.setState(state, true);
    rig2.update(0);
    const dir2 = new THREE.Vector3();
    cam2.getWorldDirection(dir2);

    console.log(
      `${mode}: posErr ${cam2.position.distanceTo(cam.position).toFixed(4)}  dirErr ${dir2.distanceTo(dir).toFixed(4)}`,
    );
    console.log(`   before dir ${dir.toArray().map((v) => v.toFixed(3)).join(',')}  after dir ${dir2.toArray().map((v) => v.toFixed(3)).join(',')}`);
    rig.dispose(); rig2.dispose();
  }
}

/* ------------------------------------------------------------------ */
/* 5. camera: fly -> orbit after flying somewhere                      */
/* ------------------------------------------------------------------ */
console.log('--- camera fly then back to orbit ---');
{
  const listeners = new Map<string, ((e: unknown) => void)[]>();
  const dom = {
    style: {} as Record<string, string>,
    addEventListener: (t: string, fn: (e: unknown) => void) => {
      const l = listeners.get(t) ?? []; l.push(fn); listeners.set(t, l);
    },
    removeEventListener: () => undefined,
    setPointerCapture: () => undefined,
    releasePointerCapture: () => undefined,
    hasPointerCapture: () => false,
    requestPointerLock: () => undefined,
  } as unknown as HTMLElement;
  const fire = (t: string, e: unknown): void => { for (const fn of listeners.get(t) ?? []) fn(e); };

  const cam = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 5000);
  cam.position.set(0, 34, 96);
  const rig = new CameraRig(cam, dom);
  rig.mode = 'fly';
  fire('keydown', { key: 'w', target: null });
  for (let f = 0; f < 120; f += 1) rig.update(0.016);
  fire('keyup', { key: 'w', target: null });
  for (let f = 0; f < 60; f += 1) rig.update(0.016);
  const flown = cam.position.clone();
  const dirBefore = new THREE.Vector3();
  cam.getWorldDirection(dirBefore);
  console.log('flew to', flown.toArray().map((v) => v.toFixed(2)).join(','));
  rig.mode = 'orbit';
  rig.update(0);
  const dirAfter = new THREE.Vector3();
  cam.getWorldDirection(dirAfter);
  console.log(
    `fly->orbit: posJump ${cam.position.distanceTo(flown).toFixed(4)} dirJump ${dirAfter.distanceTo(dirBefore).toFixed(4)} target ${JSON.stringify(rig.getState().target)}`,
  );
  rig.dispose();
}

console.log('DONE2');
