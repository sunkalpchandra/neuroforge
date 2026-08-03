import { createJiti } from 'jiti';
import { fileURLToPath } from 'node:url';
const root = '/Users/sunkalp/neuroforge';
const jiti = createJiti(fileURLToPath(import.meta.url), {
  alias: {
    '@neuroforge/shared': `${root}/packages/shared/src/index.ts`,
    '@neuroforge/math': `${root}/packages/math/src/index.ts`,
    '@neuroforge/simulation': `${root}/packages/simulation/src/index.ts`,
  },
  interopDefault: true, fsCache: false, moduleCache: false,
});
const S = await jiti.import('@neuroforge/shared');
const { SimulationEngine } = await jiti.import('@neuroforge/simulation');

const n = {
  id: 'a', label: 'a', position: {x:0,y:0,z:0}, params: {...S.DEFAULT_LIF},
  polarity: 'excitatory', morphology: S.defaultMorphology('pyramidal', 1),
  population: null, bias: 0, noise: 0, enabled: true,
};
const c = {
  id:'c', name:'n', description:'', version:1, createdAt:0, updatedAt:0,
  neurons:[n], synapses:[], populations:[], projections:[], stimuli:[], probes:[],
  simulation: {...S.DEFAULT_SIMULATION_SETTINGS, dt: 0.02, noise: 0},
  camera: {...S.DEFAULT_CAMERA}, render: {...S.DEFAULT_RENDER_SETTINGS}, tags: [],
};
const e = new SimulationEngine();
e.load(c);
const b = e.buffers;
console.log('params cm,gL,eL,vTh,vRe,tRef =', Array.from(b.neurons.params.slice(0,6)));
console.log('settings dt', e.settings.dt, 'noise', e.settings.noise, 'gain', e.settings.gain);
b.neurons.bias[0] = 0;
for (let i=0;i<5000;i++) e.stepOnce();
console.log('baseline v =', b.neurons.v[0]);
b.neurons.bias[0] = -30;
const rec = new Float32Array(15000);
for (let i=0;i<15000;i++){ e.stepOnce(); rec[i]=b.neurons.v[0]; }
console.log('final v =', b.neurons.v[0], ' rec[14999]=', rec[14999], ' iExt', b.neurons.iExt[0], ' iSyn', b.neurons.iSyn[0], ' bias', b.neurons.bias[0]);
// analytic
const tau = 200/10, vInf = -70 + (-30)/10;
console.log('analytic vInf', vInf);
for (const t of [1,5,10,20,40,80]) {
  const i = Math.round(t/0.02)-1;
  console.log(` t=${t} sim=${rec[i].toFixed(6)} exact=${(vInf + (-70-vInf)*Math.exp(-t/tau)).toFixed(6)}`);
}
e.dispose();
