import { createJiti } from 'jiti';
import { fileURLToPath } from 'node:url';
import { readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
globalThis.React = React;

const root = '/Users/sunkalp/neuroforge';
const src = `${root}/apps/web/src/components/experiments/ephys-panel.tsx`;
const probe = `${root}/apps/web/src/components/experiments/.probe-ephys.tsx`;
writeFileSync(probe, readFileSync(src, 'utf8') +
  '\nexport { FiView, IvView, TauView, AdaptationView, PprView, RheobaseView, Plot, SpikeStrip, niceTicks };\n');

const jiti = createJiti(fileURLToPath(import.meta.url), {
  alias: {
    '@neuroforge/shared': `${root}/packages/shared/src/index.ts`,
    '@neuroforge/math': `${root}/packages/math/src/index.ts`,
    '@neuroforge/simulation': `${root}/packages/simulation/src/index.ts`,
    '@neuroforge/editor': `${root}/packages/editor/src/index.ts`,
    '@neuroforge/ui': `${root}/packages/ui/src/index.ts`,
    '@/lib/format': `${root}/apps/web/src/lib/format.ts`,
    '@/lib/experiments/protocols': `${root}/apps/web/src/lib/experiments/protocols.ts`,
  },
  interopDefault: true, fsCache: false, moduleCache: true, jsx: true,
});

try {
  const S = await jiti.import('@neuroforge/shared');
  const P = await jiti.import('@/lib/experiments/protocols');
  const V = await jiti.import(probe);

  const neuron = (id, params, o = {}) => ({
    id, label: id, position: {x:0,y:0,z:0}, params: {...params},
    polarity: 'excitatory', morphology: S.defaultMorphology('pyramidal', 4242),
    population: null, bias: 0, noise: 0, enabled: true, ...o,
  });
  const syn = (id, a, b, o = {}) => ({
    id, source: a, target: b, receptor: 'ampa', weight: 2, delay: 1,
    kinetics: {...S.RECEPTOR_DEFAULTS.ampa}, plasticity: {...S.DEFAULT_PLASTICITY},
    stp: {...S.DEFAULT_STP}, releaseProbability: 1, arc: 0, enabled: true, ...o,
  });
  const circ = (ns, ss) => ({
    id:'c', name:'n', description:'', version:1, createdAt:0, updatedAt:0,
    neurons: ns, synapses: ss, populations: [], projections: [], stimuli: [], probes: [],
    simulation: {...S.DEFAULT_SIMULATION_SETTINGS}, camera: {...S.DEFAULT_CAMERA},
    render: {...S.DEFAULT_RENDER_SETTINGS}, tags: [],
  });

  const colour = S.identityColorHex(4242);
  const base = circ([neuron('a', S.DEFAULT_LIF), neuron('x', S.DEFAULT_ADEX)], []);
  const render = (name, el) => {
    const html = renderToStaticMarkup(el);
    const svgs = (html.match(/<svg/g) || []).length;
    const texts = (html.match(/<text/g) || []).length;
    const nan = /NaN|Infinity|undefined|\bnull\b/.test(html);
    console.log(`${name.padEnd(22)} len ${String(html.length).padStart(6)}  svg ${svgs}  text ${texts}  ${nan ? 'BAD-TOKEN' : 'clean'}  colour ${html.includes(colour)}`);
    if (nan) {
      const m = html.match(/.{60}(NaN|Infinity|undefined|null).{60}/);
      console.log('   >>', m && m[0]);
    }
    return html;
  };

  render('FI', React.createElement(V.FiView, { color: colour,
    result: await P.runFiCurve(base, 'a', { fromPa: 0, toPa: 500, stepPa: 50, settleMs: 100, measureMs: 300, dt: 0.1 }) }));
  render('FI silent', React.createElement(V.FiView, { color: colour,
    result: await P.runFiCurve(base, 'a', { fromPa: 0, toPa: 50, stepPa: 25, settleMs: 50, measureMs: 100, dt: 0.1 }) }));
  render('FI one level', React.createElement(V.FiView, { color: colour,
    result: await P.runFiCurve(base, 'a', { fromPa: 400, toPa: 400, stepPa: 25, settleMs: 50, measureMs: 100, dt: 0.1 }) }));

  render('IV', React.createElement(V.IvView, { color: colour,
    result: await P.runIvCurve(base, 'a', { fromPa: -100, toPa: 60, stepPa: 20, settleMs: 200, measureMs: 100, dt: 0.1 }) }));
  render('IV with spiking', React.createElement(V.IvView, { color: colour,
    result: await P.runIvCurve(base, 'a', { fromPa: -50, toPa: 400, stepPa: 50, settleMs: 200, measureMs: 100, dt: 0.1 }) }));

  render('TAU lif', React.createElement(V.TauView, { color: colour,
    result: await P.runMembraneTau(base, 'a', { amplitudePa: -40, baselineMs: 60, stepMs: 250, dt: 0.05 }) }));
  render('TAU adex', React.createElement(V.TauView, { color: colour,
    result: await P.runMembraneTau(base, 'x', { amplitudePa: -40, baselineMs: 60, stepMs: 250, dt: 0.05 }) }));

  render('ADAPT adex', React.createElement(V.AdaptationView, { color: colour,
    result: await P.runAdaptation(base, 'x', { amplitudePa: 800, settleMs: 60, durationMs: 900, dt: 0.05 }) }));
  render('ADAPT silent', React.createElement(V.AdaptationView, { color: colour,
    result: await P.runAdaptation(base, 'a', { amplitudePa: 1, settleMs: 30, durationMs: 200, dt: 0.1 }) }));
  render('ADAPT one spike', React.createElement(V.AdaptationView, { color: colour,
    result: await P.runAdaptation(base, 'a', { amplitudePa: 210, settleMs: 30, durationMs: 60, dt: 0.1 }) }));

  const pair = circ([neuron('p', S.DEFAULT_LIF), neuron('q', S.DEFAULT_LIF, { label: 'Target Q' })],
    [syn('s', 'p', 'q', { stp: { enabled: true, u: 0.45, tauRec: 350, tauFacil: 0 } })]);
  render('PPR depressing', React.createElement(V.PprView, { color: colour,
    result: await P.runPairedPulse(pair, { synapseId: 's', fromMs: 10, toMs: 150, stepMs: 35, windowMs: 120, trials: 1, dt: 0.1 }) }));
  const flat = circ([neuron('p', S.DEFAULT_LIF), neuron('q', S.DEFAULT_LIF)], [syn('s', 'p', 'q')]);
  render('PPR no STP', React.createElement(V.PprView, { color: colour,
    result: await P.runPairedPulse(flat, { synapseId: 's', fromMs: 20, toMs: 60, stepMs: 20, windowMs: 100, trials: 1, dt: 0.1 }) }));
  const stoch = circ([neuron('p', S.DEFAULT_LIF), neuron('q', S.DEFAULT_LIF)],
    [syn('s', 'p', 'q', { releaseProbability: 0.4 })]);
  render('PPR stochastic', React.createElement(V.PprView, { color: colour,
    result: await P.runPairedPulse(stoch, { synapseId: 's', fromMs: 30, toMs: 30, stepMs: 10, windowMs: 100, trials: 3, dt: 0.1 }) }));
  render('PPR all-fail', React.createElement(V.PprView, { color: colour,
    result: await P.runPairedPulse(flat, { synapseId: 's', fromMs: 0.5, toMs: 1.5, stepMs: 0.5, windowMs: 50, trials: 1, dt: 0.1 }) }));

  render('RHEO', React.createElement(V.RheobaseView, { color: colour,
    result: await P.runRheobase(base, 'a', { lowPa: 0, highPa: 800, tolerancePa: 1, windowMs: 800, dt: 0.05 }) }));
  render('RHEO bounded below', React.createElement(V.RheobaseView, { color: colour,
    result: await P.runRheobase(base, 'a', { lowPa: 400, highPa: 800, tolerancePa: 1, windowMs: 800, dt: 0.05 }) }));

  // Plot edge cases
  render('Plot empty', React.createElement(V.Plot, { series: [], xLabel: 'x', yLabel: 'y', ariaLabel: 'empty' }));
  render('Plot single point', React.createElement(V.Plot, {
    series: [{ id: 'a', points: [{ x: 3, y: 7 }], color: colour, dots: true }], xLabel: 'x', yLabel: 'y', ariaLabel: 'one' }));
  render('Plot flat line', React.createElement(V.Plot, {
    series: [{ id: 'a', points: [{ x: 0, y: 5 }, { x: 10, y: 5 }], color: colour, line: true }], xLabel: 'x', yLabel: 'y', ariaLabel: 'flat' }));
  render('Plot tiny range', React.createElement(V.Plot, {
    series: [{ id: 'a', points: [{ x: 0, y: 1e-9 }, { x: 1e-9, y: 2e-9 }], color: colour, line: true, dots: true }], xLabel: 'x', yLabel: 'y', ariaLabel: 'tiny' }));
  render('Plot negatives', React.createElement(V.Plot, {
    series: [{ id: 'a', points: [{ x: -500, y: -0.004 }, { x: 500, y: 12345 }], color: colour, line: true, dots: true }], xLabel: 'x', yLabel: 'y', ariaLabel: 'neg' }));
  render('SpikeStrip zero', React.createElement(V.SpikeStrip, { times: [], durationMs: 0, color: colour }));

  console.log('ticks 0..500:', V.niceTicks(0, 500, 4));
  console.log('ticks -70..-73:', V.niceTicks(-73, -70, 4));
  console.log('ticks equal:', V.niceTicks(5, 5, 4));
} finally {
  unlinkSync(probe);
}
