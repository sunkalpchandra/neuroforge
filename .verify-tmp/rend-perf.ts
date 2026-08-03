import { allocateSimulationBuffers, DEFAULT_RENDER_SETTINGS } from '@neuroforge/shared';
import { AxonField, GlyphLibrary, NeuronField } from '@neuroforge/renderer';
import { axonSag, chordLength } from '@neuroforge/renderer/axon-spline';

const N = 50000;
const S = 200000;
const buffers = allocateSimulationBuffers(N, S);
const n = buffers.neurons;
n.count = N;
for (let i = 0; i < N; i += 1) {
  n.position[i * 3] = Math.sin(i) * 400;
  n.position[i * 3 + 1] = Math.cos(i * 1.7) * 200;
  n.position[i * 3 + 2] = Math.sin(i * 0.3) * 400;
  n.archetype[i] = i % 7;
  n.seed[i] = i * 2654435761;
}
const s = buffers.synapses;
s.count = S;
for (let i = 0; i < S; i += 1) {
  s.pre[i] = i % N;
  s.post[i] = (i * 7 + 3) % N;
  s.receptor[i] = i % 5;
  s.weight[i] = (i % 10) * 0.4;
}

const library = new GlyphLibrary();
const field = new NeuronField(library);
const axons = new AxonField();
const settings = { ...DEFAULT_RENDER_SETTINGS };

const t0 = performance.now();
field.rebuild(buffers);
const t1 = performance.now();
axons.rebuild(buffers);
const t2 = performance.now();
console.log(`rebuild: neurons ${(t1 - t0).toFixed(1)}ms  axons ${(t2 - t1).toFixed(1)}ms`);

for (let f = 0; f < 20; f += 1) {
  field.update(buffers, 0.016, settings);
  axons.update(buffers, 0.016, settings);
}

const bench = (label: string, fn: () => void, iters = 40): void => {
  const a = performance.now();
  for (let i = 0; i < iters; i += 1) fn();
  const b = performance.now();
  console.log(`${label.padEnd(28)} ${((b - a) / iters).toFixed(2)} ms/frame`);
};

bench('NeuronField.update 50k', () => field.update(buffers, 0.016, settings));
bench('AxonField.update 200k', () => axons.update(buffers, 0.016, settings));

// Baseline: the same inner loop written inline, no cross-module calls.
const start = new Float32Array(S * 3);
const end = new Float32Array(S * 3);
const sag = new Float32Array(S);
const activity = new Float32Array(S);
function inline(): void {
  const position = n.position;
  const limit = n.count;
  for (let i = 0; i < S; i += 1) {
    const pre = s.pre[i];
    const post = s.post[i];
    if (pre >= limit || post >= limit) continue;
    const a = pre * 3;
    const b = post * 3;
    const ax = position[a];
    const ay = position[a + 1];
    const az = position[a + 2];
    const bx = position[b];
    const by = position[b + 1];
    const bz = position[b + 2];
    start[i * 3] = ax; start[i * 3 + 1] = ay; start[i * 3 + 2] = az;
    end[i * 3] = bx; end[i * 3 + 1] = by; end[i * 3 + 2] = bz;
    const dx = bx - ax, dy = by - ay, dz = bz - az;
    const chord = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const arc = s.arc[i];
    sag[i] = arc !== 0 ? arc : chord * 0.15;
    activity[i] = s.enabled[i] === 0 ? 0 : s.activity[i];
  }
}
bench('inline equivalent 200k', inline);

function viaHelpers(): void {
  const position = n.position;
  const limit = n.count;
  for (let i = 0; i < S; i += 1) {
    const pre = s.pre[i];
    const post = s.post[i];
    if (pre >= limit || post >= limit) continue;
    const a = pre * 3;
    const b = post * 3;
    const ax = position[a];
    const ay = position[a + 1];
    const az = position[a + 2];
    const bx = position[b];
    const by = position[b + 1];
    const bz = position[b + 2];
    start[i * 3] = ax; start[i * 3 + 1] = ay; start[i * 3 + 2] = az;
    end[i * 3] = bx; end[i * 3 + 1] = by; end[i * 3 + 2] = bz;
    sag[i] = axonSag(s.arc[i], chordLength(ax, ay, az, bx, by, bz));
    activity[i] = s.enabled[i] === 0 ? 0 : s.activity[i];
  }
}
bench('cross-module helpers 200k', viaHelpers);

field.dispose(); axons.dispose(); library.dispose();
console.log('DONE-PERF');
