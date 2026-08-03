import { createJiti } from 'jiti';
import { fileURLToPath } from 'node:url';
const root = '/Users/sunkalp/neuroforge';
const jiti = createJiti(fileURLToPath(import.meta.url), {
  alias: {
    '@neuroforge/shared': `${root}/packages/shared/src/index.ts`,
    '@neuroforge/math': `${root}/packages/math/src/index.ts`,
    '@neuroforge/simulation': `${root}/packages/simulation/src/index.ts`,
  }, interopDefault: true, fsCache: false, moduleCache: true,
});
const S = await jiti.import('@neuroforge/shared');
const P = await jiti.import(`${root}/apps/web/src/lib/experiments/protocols.ts`);
const n = (id, p, o={}) => ({ id, label: id, position:{x:0,y:0,z:0}, params:{...p},
  polarity:'excitatory', morphology:S.defaultMorphology('pyramidal',1), population:null,
  bias:0, noise:0, enabled:true, ...o });
const sy = (id,a,b,o={}) => ({ id, source:a, target:b, receptor:'ampa', weight:2, delay:1,
  kinetics:{...S.RECEPTOR_DEFAULTS.ampa}, plasticity:{...S.DEFAULT_PLASTICITY},
  stp:{...S.DEFAULT_STP}, releaseProbability:1, arc:0, enabled:true, ...o });
const circ = (ns,ss) => ({ id:'c',name:'n',description:'',version:1,createdAt:0,updatedAt:0,
  neurons:ns, synapses:ss, populations:[], projections:[], stimuli:[], probes:[],
  simulation:{...S.DEFAULT_SIMULATION_SETTINGS}, camera:{...S.DEFAULT_CAMERA},
  render:{...S.DEFAULT_RENDER_SETTINGS}, tags:[] });

const base = circ([n('a', S.DEFAULT_LIF)], []);
const pair = circ([n('p', S.DEFAULT_LIF), n('q', S.DEFAULT_LIF)],
  [sy('s','p','q',{ stp:{enabled:true,u:0.5,tauRec:400,tauFacil:0} })]);

async function timed(name, fn) {
  let worst = 0, prev = performance.now(), conds = 0;
  const r = await fn({ onProgress: () => {
    const now = performance.now();
    if (conds > 0) worst = Math.max(worst, now - prev);
    prev = now; conds += 1;
  }});
  const now = performance.now();
  worst = Math.max(worst, now - prev);
  console.log(name.padEnd(28), 'conds', String(conds).padStart(4), ' worst condition', worst.toFixed(1), 'ms', ' sim', r.meta.simulatedMs.toFixed(0), 'ms', ' wall', r.meta.elapsedMs.toFixed(0), 'ms');
}

await timed('FI default dt=0.1', (o) => P.runFiCurve(base,'a',{fromPa:0,toPa:500,stepPa:25,settleMs:200,measureMs:500,dt:0.1},o));
await timed('FI dt=0.005 128 levels', (o) => P.runFiCurve(base,'a',{fromPa:0,toPa:500,stepPa:0.1,settleMs:200,measureMs:500,dt:0.005},o));
await timed('IV default', (o) => P.runIvCurve(base,'a',{fromPa:-100,toPa:60,stepPa:10,settleMs:200,measureMs:100,dt:0.1},o));
await timed('TAU dt=0.005 step 300', (o) => P.runMembraneTau(base,'a',{amplitudePa:-50,baselineMs:100,stepMs:300,dt:0.005},o));
await timed('ADAPT 1 s', (o) => P.runAdaptation(base,'a',{amplitudePa:400,settleMs:100,durationMs:1000,dt:0.1},o));
await timed('RHEO tol 0.5', (o) => P.runRheobase(base,'a',{lowPa:0,highPa:1000,tolerancePa:0.5,windowMs:500,dt:0.1},o));
await timed('PPR default', (o) => P.runPairedPulse(pair,{synapseId:'s',fromMs:10,toMs:200,stepMs:10,windowMs:150,trials:1,dt:0.1},o));
await timed('PPR dt=0.005', (o) => P.runPairedPulse(pair,{synapseId:'s',fromMs:10,toMs:200,stepMs:20,windowMs:150,trials:1,dt:0.005},o));
