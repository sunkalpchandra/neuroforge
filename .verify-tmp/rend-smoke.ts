import * as THREE from 'three';
import {
  allocateSimulationBuffers,
  DEFAULT_RENDER_SETTINGS,
  MORPHOLOGY_ARCHETYPES,
  defaultMorphology,
  NEURON_FLAG,
} from '@neuroforge/shared';
import {
  buildSomaGeometry,
  buildDendriteGeometry,
  buildAxonGeometry,
  GlyphLibrary,
  NeuronField,
  AxonField,
  SpikeParticles,
  InfiniteGrid,
  SelectionOverlay,
  CameraRig,
} from '@neuroforge/renderer';

function checkFinite(label: string, g: THREE.BufferGeometry): void {
  for (const name of Object.keys(g.attributes)) {
    const a = g.attributes[name].array as Float32Array;
    for (let i = 0; i < a.length; i += 1) {
      if (!Number.isFinite(a[i])) {
        console.log(`FAIL ${label}.${name} non-finite at ${i}`);
        return;
      }
    }
  }
  const idx = g.index;
  if (idx) {
    const vcount = g.attributes.position.count;
    for (let i = 0; i < idx.count; i += 1) {
      const v = idx.getX(i);
      if (v < 0 || v >= vcount) {
        console.log(`FAIL ${label}.index out of range ${v} / ${vcount}`);
        return;
      }
    }
  }
}

function extent(g: THREE.BufferGeometry): string {
  g.computeBoundingBox();
  const b = g.boundingBox!;
  return `${(b.max.x - b.min.x).toFixed(1)}x${(b.max.y - b.min.y).toFixed(1)}x${(b.max.z - b.min.z).toFixed(1)}`;
}

console.log('--- glyph geometry ---');
for (const arch of MORPHOLOGY_ARCHETYPES) {
  const m = defaultMorphology(arch, 12345);
  const d = buildDendriteGeometry(m);
  const a = buildAxonGeometry(m);
  checkFinite(`${arch}.dendrites`, d);
  checkFinite(`${arch}.axon`, a);
  console.log(
    `${arch.padEnd(10)} dend ${extent(d).padEnd(20)} v${String(d.attributes.position.count).padEnd(6)} axon ${extent(a).padEnd(20)} v${a.attributes.position.count}`,
  );
}

console.log('--- degenerate morphologies ---');
const degenerate = [
  { ...defaultMorphology('pyramidal', 7), dendriteLength: 0, axonLength: 0 },
  { ...defaultMorphology('purkinje', 7), scale: 0 },
  { ...defaultMorphology('granule', 7), dendriteCount: 0, dendriteDepth: 0, axonTerminals: 0 },
  { ...defaultMorphology('motor', 7), dendriteTaper: 0, dendriteSpread: 0 },
  { ...defaultMorphology('bipolar', 7), dendriteTaper: 1 },
  { ...defaultMorphology('stellate', 7), dendriteDepth: 12, dendriteCount: 20 },
];
for (const m of degenerate) {
  checkFinite(`degenerate.${m.archetype}.d`, buildDendriteGeometry(m));
  checkFinite(`degenerate.${m.archetype}.a`, buildAxonGeometry(m));
}
checkFinite('soma', buildSomaGeometry());
console.log('degenerate ok');

console.log('--- determinism ---');
{
  const m = defaultMorphology('pyramidal', 999);
  const a = buildDendriteGeometry(m).attributes.position.array as Float32Array;
  const b = buildDendriteGeometry(m).attributes.position.array as Float32Array;
  let same = a.length === b.length;
  if (same) for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) { same = false; break; }
  console.log('deterministic:', same);
}

console.log('--- fields ---');
const N = 2000;
const S = 8000;
const buffers = allocateSimulationBuffers(N, S);
const n = buffers.neurons;
n.count = N;
for (let i = 0; i < N; i += 1) {
  n.position[i * 3] = Math.sin(i) * 40;
  n.position[i * 3 + 1] = Math.cos(i * 1.7) * 20;
  n.position[i * 3 + 2] = Math.sin(i * 0.3) * 40;
  n.v[i] = -65 + (i % 40);
  n.archetype[i] = i % 7;
  n.seed[i] = i * 2654435761;
  n.polarity[i] = i % 5 === 0 ? 1 : 0;
  n.scale[i] = 1;
}
n.flags[3] = NEURON_FLAG.SELECTED;
n.flags[9] = NEURON_FLAG.HOVERED;
const s = buffers.synapses;
s.count = S;
for (let i = 0; i < S; i += 1) {
  s.pre[i] = i % N;
  s.post[i] = (i * 7 + 3) % N;
  s.weight[i] = (i % 10) * 0.5;
  s.receptor[i] = i % 5;
  s.arc[i] = 0;
  s.activity[i] = 0;
}

const library = new GlyphLibrary();
const field = new NeuronField(library);
const axons = new AxonField();
const particles = new SpikeParticles();
const grid = new InfiniteGrid();
const overlay = new SelectionOverlay();

field.rebuild(buffers);
axons.rebuild(buffers);
console.log('neuron draw calls:', field.children.length);

const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 5000);
camera.position.set(0, 34, 96);
camera.updateMatrixWorld();
camera.updateProjectionMatrix();

const settings = { ...DEFAULT_RENDER_SETTINGS };

for (let f = 0; f < 30; f += 1) {
  buffers.step += 1;
  buffers.time += 16;
  for (let i = 0; i < N; i += 1) {
    n.spike[i] = (i + f) % 97 === 0 ? 1 : 0;
    if (n.spike[i]) {
      buffers.spikes.neuron[buffers.spikes.head % buffers.spikes.capacity] = i;
      buffers.spikes.time[buffers.spikes.head % buffers.spikes.capacity] = buffers.time;
      buffers.spikes.head += 1;
    }
  }
  particles.emitFromSpikes(buffers);
  field.update(buffers, 0.016, settings);
  axons.update(buffers, 0.016, settings);
  particles.update(0.016, settings);
  overlay.update(buffers, 0.016);
  grid.update(camera, settings);
}
console.log('live particles:', particles.liveCount);

if (global.gc) global.gc();
const before = process.memoryUsage().heapUsed;
for (let f = 0; f < 300; f += 1) {
  buffers.step += 1;
  particles.emitFromSpikes(buffers);
  field.update(buffers, 0.016, settings);
  axons.update(buffers, 0.016, settings);
  particles.update(0.016, settings);
  overlay.update(buffers, 0.016);
  grid.update(camera, settings);
}
const after = process.memoryUsage().heapUsed;
console.log('heap delta over 300 frames (KB):', ((after - before) / 1024).toFixed(1));

function probe(label: string, g: THREE.BufferGeometry): void {
  for (const name of Object.keys(g.attributes)) {
    const arr = g.attributes[name].array as Float32Array;
    for (let i = 0; i < arr.length; i += 1) {
      if (!Number.isFinite(arr[i])) {
        console.log(`FAIL ${label}.${name} non-finite`);
        return;
      }
    }
  }
}
for (const child of field.children) probe('neuronfield', (child as THREE.Mesh).geometry);
probe('axonfield', (axons.children[0] as THREE.Mesh).geometry);
probe('particles', particles.geometry);
probe('overlay', (overlay.children[0] as THREE.Mesh).geometry);
console.log('field attributes finite');

const t0 = performance.now();
for (let f = 0; f < 100; f += 1) field.update(buffers, 0.016, settings);
const t1 = performance.now();
for (let f = 0; f < 100; f += 1) axons.update(buffers, 0.016, settings);
const t2 = performance.now();
console.log(`neuronfield ${((t1 - t0) / 100).toFixed(3)}ms  axonfield ${((t2 - t1) / 100).toFixed(3)}ms`);

console.log('--- raycast ---');
{
  const ray = new THREE.Raycaster();
  let hits = 0;
  let correct = 0;
  for (let i = 0; i < 50; i += 1) {
    const target = new THREE.Vector3(
      n.position[i * 3],
      n.position[i * 3 + 1],
      n.position[i * 3 + 2],
    );
    const origin = target.clone().add(new THREE.Vector3(0, 0, 300));
    ray.set(origin, target.clone().sub(origin).normalize());
    const slot = field.raycastSlot(ray);
    if (slot >= 0) hits += 1;
    if (slot === i) correct += 1;
  }
  console.log(`raycast hits ${hits}/50 exact ${correct}/50`);
}

console.log('--- camera ---');
{
  const style: Record<string, string> = {};
  const dom = {
    style,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    setPointerCapture: () => undefined,
    releasePointerCapture: () => undefined,
    hasPointerCapture: () => false,
    requestPointerLock: () => undefined,
  } as unknown as HTMLElement;
  const cam = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 5000);
  cam.position.set(0, 34, 96);
  const rig = new CameraRig(cam, dom);
  for (let f = 0; f < 120; f += 1) rig.update(0.016);
  console.log('orbit pos', cam.position.toArray().map((v) => v.toFixed(3)).join(','));

  for (const mode of ['fly', 'first-person', 'cinematic', 'orbit'] as const) {
    const bx = cam.position.x, by = cam.position.y, bz = cam.position.z;
    const bdir = new THREE.Vector3();
    cam.getWorldDirection(bdir);
    rig.mode = mode;
    rig.update(0.016);
    const jump = Math.hypot(cam.position.x - bx, cam.position.y - by, cam.position.z - bz);
    const adir = new THREE.Vector3();
    cam.getWorldDirection(adir);
    console.log(`  -> ${mode}: pos jump ${jump.toFixed(4)} dir jump ${adir.distanceTo(bdir).toFixed(4)}`);
  }

  for (const mode of ['orbit', 'fly', 'first-person', 'cinematic'] as const) {
    rig.mode = mode;
    for (let f = 0; f < 40; f += 1) rig.update(0.016);
    const st = rig.getState();
    const dir = new THREE.Vector3();
    cam.getWorldDirection(dir);

    const cam2 = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 5000);
    const rig2 = new CameraRig(cam2, dom);
    rig2.setState(st, true);
    rig2.update(0);
    const dir2 = new THREE.Vector3();
    cam2.getWorldDirection(dir2);
    const posErr = cam2.position.distanceTo(cam.position);
    const dirErr = dir2.distanceTo(dir);
    console.log(`  roundtrip ${mode}: posErr ${posErr.toFixed(4)} dirErr ${dirErr.toFixed(4)}`);
    rig2.dispose();
  }

  rig.mode = 'orbit';
  rig.frame(new THREE.Vector3(-50, -20, -50), new THREE.Vector3(50, 20, 50), 0.5);
  for (let f = 0; f < 120; f += 1) rig.update(0.016);
  console.log('framed dist', cam.position.distanceTo(new THREE.Vector3(0, 0, 0)).toFixed(2));
  rig.dispose();
}

console.log('--- structural churn ---');
{
  for (let round = 0; round < 5; round += 1) {
    n.count = 500 + round * 300;
    s.count = 1000 + round * 900;
    field.rebuild(buffers);
    axons.rebuild(buffers);
    for (let f = 0; f < 5; f += 1) {
      field.update(buffers, 0.016, settings);
      axons.update(buffers, 0.016, settings);
      overlay.update(buffers, 0.016);
    }
  }
  console.log('churn ok, draw calls', field.children.length);
}

console.log('--- small glyph library ---');
{
  const small = new GlyphLibrary(4);
  const f2 = new NeuronField(small);
  n.count = N;
  f2.rebuild(buffers);
  const sources = new Set<THREE.BufferGeometry>();
  for (const child of f2.children) {
    const g = (child as THREE.Mesh).geometry;
    if (!g.attributes.position) console.log('FAIL: group lost its position attribute');
    sources.add(g);
  }
  console.log('small-library groups:', f2.children.length, 'geometries with position:',
    [...sources].filter((g) => !!g.attributes.position).length);
  f2.dispose();
  small.dispose();
}

field.dispose();
axons.dispose();
particles.dispose();
grid.dispose();
overlay.dispose();
library.dispose();
console.log('DONE');
