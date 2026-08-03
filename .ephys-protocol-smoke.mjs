import { createJiti } from 'jiti';
import { fileURLToPath } from 'node:url';

const root = '/Users/sunkalp/neuroforge';
const pkg = (name) => `${root}/packages/${name}/src/index.ts`;

const jiti = createJiti(fileURLToPath(import.meta.url), {
  alias: {
    '@neuroforge/shared': pkg('shared'),
    '@neuroforge/math': pkg('math'),
    '@neuroforge/simulation': pkg('simulation'),
  },
  interopDefault: true,
  fsCache: false,
  moduleCache: false,
});

const shared = await jiti.import('@neuroforge/shared');
const P = await jiti.import(`${root}/apps/web/src/lib/experiments/protocols.ts`);

const {
  DEFAULT_LIF,
  DEFAULT_ADEX,
  DEFAULT_IZHIKEVICH,
  DEFAULT_SIMULATION_SETTINGS,
  DEFAULT_CAMERA,
  DEFAULT_RENDER_SETTINGS,
  DEFAULT_PLASTICITY,
  DEFAULT_STP,
  RECEPTOR_DEFAULTS,
  defaultMorphology,
} = shared;

function neuron(id, params, overrides = {}) {
  return {
    id,
    label: id,
    position: { x: 0, y: 0, z: 0 },
    params: { ...params },
    polarity: 'excitatory',
    morphology: defaultMorphology('pyramidal', 7),
    population: null,
    bias: 0,
    noise: 0,
    enabled: true,
    ...overrides,
  };
}

function synapse(id, source, target, overrides = {}) {
  return {
    id,
    source,
    target,
    receptor: 'ampa',
    weight: 2,
    delay: 1,
    kinetics: { ...RECEPTOR_DEFAULTS.ampa },
    plasticity: { ...DEFAULT_PLASTICITY },
    stp: { ...DEFAULT_STP },
    releaseProbability: 1,
    arc: 0,
    enabled: true,
    ...overrides,
  };
}

function circuitOf(neurons, synapses) {
  return {
    id: 'circ',
    name: 'smoke',
    description: '',
    version: 1,
    createdAt: 0,
    updatedAt: 0,
    neurons,
    synapses,
    populations: [],
    projections: [],
    stimuli: [],
    probes: [],
    simulation: { ...DEFAULT_SIMULATION_SETTINGS },
    camera: { ...DEFAULT_CAMERA },
    render: { ...DEFAULT_RENDER_SETTINGS },
    tags: [],
  };
}

const base = circuitOf(
  [neuron('lif-1', DEFAULT_LIF), neuron('adex-1', DEFAULT_ADEX), neuron('izh-1', DEFAULT_IZHIKEVICH)],
  [],
);

const line = (...args) => console.log(...args);

/* -------------------------------------------------------------------- F-I */
{
  const r = await P.runFiCurve(base, 'lif-1', {
    fromPa: 0, toPa: 500, stepPa: 25, settleMs: 200, measureMs: 500, dt: 0.05,
  });
  line('FI  lif  rheobase', r.rheobasePa, '(analytic 200)  gain', r.gainHzPerPa?.toFixed(4),
    'r2', r.fit?.r2.toFixed(4), 'peak', r.maxRateHz.toFixed(2));
  const at300 = r.points.find((p) => Math.abs(p.currentPa - 300) < 1e-6);
  const { cm, gL, eL, vThresh, vReset, tRefract } = DEFAULT_LIF;
  const tau = cm / gL;
  const I = 300;
  const isi = tRefract + tau * Math.log((I - gL * (vReset - eL)) / (I - gL * (vThresh - eL)));
  line('    @300pA measured', at300.rateHz.toFixed(3), 'Hz  analytic', (1000 / isi).toFixed(3), 'Hz',
    ' latency', at300.latencyMs?.toFixed(2), 'ms');
}

/* -------------------------------------------------------------------- I-V */
{
  const r = await P.runIvCurve(base, 'lif-1', {
    fromPa: -100, toPa: 60, stepPa: 10, settleMs: 300, measureMs: 100, dt: 0.05,
  });
  line('IV  lif  Rin', r.inputResistanceMohm?.toFixed(4), 'MOhm (analytic 100)  V0',
    r.interceptMv?.toFixed(4), 'mV (analytic -70)  excluded', r.excluded, ' r2', r.fit?.r2.toFixed(6));
}

/* -------------------------------------------------------------------- tau */
for (const id of ['lif-1', 'adex-1', 'izh-1']) {
  try {
    const r = await P.runMembraneTau(base, id, {
      amplitudePa: -30, baselineMs: 100, stepMs: 300, dt: 0.02,
    });
    line(`TAU ${id.padEnd(6)} fitted`, r.tauMs.toFixed(4), ' analytic', r.analyticTauMs?.toFixed(4) ?? 'n/a',
      ' err%', r.errorPercent?.toFixed(3) ?? 'n/a', ' r2', r.fitR2.toFixed(6), ' defl', r.deflectionMv.toFixed(3),
      ' samples', r.traceT.length);
  } catch (e) {
    line(`TAU ${id.padEnd(6)}`, e.name, '-', e.message.slice(0, 80));
  }
}

/* ------------------------------------------------------------- adaptation */
{
  const r = await P.runAdaptation(base, 'adex-1', {
    amplitudePa: 800, settleMs: 100, durationMs: 1200, dt: 0.02,
  });
  line('ADP adex spikes', r.spikeCount, ' index', r.adaptationIndex?.toFixed(3),
    ' inst', r.instantaneousHz?.toFixed(2), ' steady', r.steadyHz?.toFixed(2),
    ' isis', r.isisMs.length, ' first3', r.isisMs.slice(0, 3).map((v) => v.toFixed(2)).join(','));
  const r2 = await P.runAdaptation(base, 'lif-1', {
    amplitudePa: 400, settleMs: 100, durationMs: 800, dt: 0.02,
  });
  line('ADP lif  spikes', r2.spikeCount, ' index', r2.adaptationIndex?.toFixed(6), '(expect 1)');
  const r3 = await P.runAdaptation(base, 'lif-1', {
    amplitudePa: 1, settleMs: 50, durationMs: 200, dt: 0.05,
  });
  line('ADP silent spikes', r3.spikeCount, ' index', r3.adaptationIndex, ' steady', r3.steadyHz);
}

/* --------------------------------------------------------------- rheobase */
{
  const r = await P.runRheobase(base, 'lif-1', {
    lowPa: 0, highPa: 1000, tolerancePa: 0.01, windowMs: 3000, dt: 0.02,
  });
  line('RHE lif ', r.rheobasePa.toFixed(4), 'pA (analytic 200)  iters', r.iterations,
    ' probes', r.probes.length, ' latency', r.latencyMs?.toFixed(1), ' bracket',
    (r.bracketHighPa - r.bracketLowPa).toFixed(5));
  const r2 = await P.runRheobase(base, 'izh-1', {
    lowPa: 0, highPa: 1000, tolerancePa: 0.5, windowMs: 1000, dt: 0.02,
  });
  line('RHE izh ', r2.rheobasePa.toFixed(3), 'pA  iters', r2.iterations);
  const r3 = await P.runRheobase(base, 'lif-1', {
    lowPa: 300, highPa: 900, tolerancePa: 1, windowMs: 500, dt: 0.05,
  });
  line('RHE bounded-below', r3.boundedBelow, r3.rheobasePa);
}

/* ------------------------------------------------------------ paired pulse */
{
  const pre = neuron('pre', DEFAULT_LIF);
  const post = neuron('post', DEFAULT_LIF);

  const noStp = circuitOf([pre, post], [synapse('s1', 'pre', 'post')]);
  const r0 = await P.runPairedPulse(noStp, {
    synapseId: 's1', fromMs: 20, toMs: 100, stepMs: 20, windowMs: 120, trials: 1, dt: 0.05,
  });
  line('PPR static ', r0.points.map((p) => `${p.intervalMs}:${p.ratio === null ? 'x' : p.ratio.toFixed(5)}`).join(' '),
    ' stim', r0.stimulusPa, 'pA  peak1', r0.points[0].peak1Ns.toFixed(5));

  const depressing = circuitOf([pre, post], [
    synapse('s1', 'pre', 'post', { stp: { enabled: true, u: 0.5, tauRec: 400, tauFacil: 0 } }),
  ]);
  const r1 = await P.runPairedPulse(depressing, {
    synapseId: 's1', fromMs: 20, toMs: 200, stepMs: 45, windowMs: 150, trials: 1, dt: 0.05,
  });
  line('PPR depressing (u=0.5, tauRec=400):');
  for (const p of r1.points) {
    // Tsodyks-Markram: R1 = 1 -> release1 = u. R2 = 1 - u recovered over dt.
    const R2 = 1 - 0.5 * Math.exp(-p.intervalMs / 400);
    line('   ', String(p.intervalMs).padStart(4), 'ms  measured', p.ratio?.toFixed(5),
      '  analytic', R2.toFixed(5), '  evoked', p.evoked, p.failure ?? '');
  }

  const facilitating = circuitOf([pre, post], [
    synapse('s1', 'pre', 'post', { stp: { enabled: true, u: 0.15, tauRec: 100, tauFacil: 500 } }),
  ]);
  const r2 = await P.runPairedPulse(facilitating, {
    synapseId: 's1', fromMs: 20, toMs: 200, stepMs: 60, windowMs: 150, trials: 1, dt: 0.05,
  });
  line('PPR facil  ', r2.points.map((p) => `${p.intervalMs}:${p.ratio === null ? 'x' : p.ratio.toFixed(4)}`).join(' '));

  const shortIsi = await P.runPairedPulse(depressing, {
    synapseId: 's1', fromMs: 1, toMs: 4, stepMs: 1, windowMs: 60, trials: 1, dt: 0.05,
  });
  line('PPR refractory intervals:', shortIsi.points.map((p) => `${p.intervalMs}:${p.ratio === null ? `x(${p.failure})` : p.ratio.toFixed(3)}`).join(' | '));

  const stochastic = circuitOf([pre, post], [
    synapse('s1', 'pre', 'post', { releaseProbability: 0.5, stp: { enabled: true, u: 0.5, tauRec: 400, tauFacil: 0 } }),
  ]);
  const r3 = await P.runPairedPulse(stochastic, {
    synapseId: 's1', fromMs: 50, toMs: 50, stepMs: 10, windowMs: 120, trials: 16, dt: 0.05,
  });
  line('PPR p=0.5 trials=16 ->', r3.points.map((p) => `${p.intervalMs}:${p.ratio?.toFixed(3)} (${p.evoked} evoked)`).join(' '));

  const autapse = circuitOf([pre], [synapse('s1', 'pre', 'pre', { delay: 3 })]);
  const r4 = await P.runPairedPulse(autapse, {
    synapseId: 's1', fromMs: 40, toMs: 40, stepMs: 10, windowMs: 100, trials: 1, dt: 0.05,
  });
  line('PPR autapse ratio', r4.points[0].ratio?.toFixed(4), ' peak1', r4.points[0].peak1Ns.toFixed(4));
}

/* ------------------------------------------------------------ abort + errors */
{
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 4);
  try {
    await P.runFiCurve(base, 'lif-1',
      { fromPa: 0, toPa: 1000, stepPa: 8, settleMs: 200, measureMs: 500, dt: 0.05 },
      { signal: controller.signal });
    line('ABORT FAILED: completed');
  } catch (e) {
    line('ABORT', e.name, '-', e.message);
  }

  let ticks = 0;
  await P.runIvCurve(base, 'lif-1',
    { fromPa: -50, toPa: -10, stepPa: 10, settleMs: 20, measureMs: 20, dt: 0.1 },
    { onProgress: () => { ticks += 1; } });
  line('PROGRESS ticks', ticks, '(expect 5)');

  const cases = [
    ['tau positive', () => P.runMembraneTau(base, 'lif-1', { amplitudePa: 50, baselineMs: 50, stepMs: 100, dt: 0.05 })],
    ['tau tiny', () => P.runMembraneTau(base, 'lif-1', { amplitudePa: -0.001, baselineMs: 50, stepMs: 100, dt: 0.05 })],
    ['rheo no spike', () => P.runRheobase(base, 'lif-1', { lowPa: 0, highPa: 10, tolerancePa: 1, windowMs: 200, dt: 0.05 })],
    ['ppr missing syn', () => P.runPairedPulse(base, { synapseId: 'nope', fromMs: 10, toMs: 20, stepMs: 10, windowMs: 50, trials: 1, dt: 0.05 })],
    ['missing neuron', () => P.runFiCurve(base, 'ghost', { fromPa: 0, toPa: 10, stepPa: 5, settleMs: 10, measureMs: 10, dt: 0.1 })],
  ];
  for (const [what, fn] of cases) {
    try {
      await fn();
      line(`ERR ${what}: NO THROW`);
    } catch (e) {
      line(`ERR ${what.padEnd(16)}`, e.name, '-', e.message.slice(0, 95));
    }
  }

  const gapCircuit = circuitOf([neuron('a', DEFAULT_LIF), neuron('b', DEFAULT_LIF)],
    [synapse('g', 'a', 'b', { receptor: 'gap', kinetics: { ...RECEPTOR_DEFAULTS.gap } })]);
  try {
    await P.runPairedPulse(gapCircuit, { synapseId: 'g', fromMs: 10, toMs: 20, stepMs: 10, windowMs: 50, trials: 1, dt: 0.05 });
    line('ERR gap: NO THROW');
  } catch (e) {
    line('ERR gap             ', e.name, '-', e.message.slice(0, 95));
  }

  const hotCircuit = circuitOf([neuron('a', DEFAULT_LIF, { bias: 400 }), neuron('b', DEFAULT_LIF)],
    [synapse('s', 'a', 'b')]);
  try {
    await P.runPairedPulse(hotCircuit, { synapseId: 's', fromMs: 20, toMs: 20, stepMs: 10, windowMs: 60, trials: 1, dt: 0.05 });
    line('ERR spontaneous: NO THROW');
  } catch (e) {
    line('ERR spontaneous     ', e.name, '-', e.message.slice(0, 95));
  }
}

/* ---------------------------------------------------------- holding bias */
{
  const held = circuitOf([neuron('h', DEFAULT_LIF, { bias: 100 })], []);
  const r = await P.runRheobase(held, 'h', { lowPa: 0, highPa: 500, tolerancePa: 0.05, windowMs: 3000, dt: 0.02 });
  line('HOLD bias 100pA -> rheobase', r.rheobasePa.toFixed(3), 'pA (expect ~100 command)  meta.holding', r.meta.holdingPa);
}

/* ------------------------------------------------------------------- misc */
{
  line('sweepCount 0..500 step .01 ->', P.sweepCount(0, 500, 0.01, P.MAX_LEVELS), 'cap', P.MAX_LEVELS);
  line('sweepCount degenerate     ->', P.sweepCount(5, 5, 1, P.MAX_LEVELS));
  line('sweepCount reversed       ->', P.sweepCount(100, 0, 25, P.MAX_LEVELS));
  const r = await P.runFiCurve(base, 'lif-1', { fromPa: 0, toPa: 200, stepPa: 100, settleMs: 50, measureMs: 100, dt: 0.1 });
  line('--- CSV ---');
  line(r.csv);
}
