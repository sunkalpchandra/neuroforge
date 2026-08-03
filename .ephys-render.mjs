import { createJiti } from 'jiti';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

globalThis.React = React;

const root = '/Users/sunkalp/neuroforge';
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
  interopDefault: true, fsCache: false, moduleCache: true,
  jsx: true,
});

const S = await jiti.import('@neuroforge/shared');
const { useEditor } = await jiti.import('@neuroforge/editor');
const mod = await jiti.import(`${root}/apps/web/src/components/experiments/ephys-panel.tsx`);
const { EphysPanel } = mod;
console.log('export EphysPanel:', typeof EphysPanel);

function neuron(id, params, overrides = {}) {
  return {
    id, label: id, position: {x:0,y:0,z:0}, params: {...params},
    polarity: 'excitatory', morphology: S.defaultMorphology('pyramidal', 1234),
    population: null, bias: 0, noise: 0, enabled: true, ...overrides,
  };
}
const syn = (id, source, target, o = {}) => ({
  id, source, target, receptor: 'ampa', weight: 2, delay: 1,
  kinetics: {...S.RECEPTOR_DEFAULTS.ampa}, plasticity: {...S.DEFAULT_PLASTICITY},
  stp: {...S.DEFAULT_STP}, releaseProbability: 1, arc: 0, enabled: true, ...o,
});

const circuit = {
  id: 'c', name: 'n', description: '', version: 1, createdAt: 0, updatedAt: 0,
  neurons: [neuron('a', S.DEFAULT_LIF, { label: 'Pyr A' }), neuron('b', S.DEFAULT_LIF), neuron('c', S.DEFAULT_ADEX)],
  synapses: [syn('s1','a','b'), syn('s2','a','c',{ receptor:'gap', kinetics:{...S.RECEPTOR_DEFAULTS.gap} }), syn('s3','a','b',{ enabled:false })],
  populations: [], projections: [], stimuli: [], probes: [],
  simulation: {...S.DEFAULT_SIMULATION_SETTINGS}, camera: {...S.DEFAULT_CAMERA},
  render: {...S.DEFAULT_RENDER_SETTINGS}, tags: [],
};

// No selection
// zustand v5 renders from getInitialState() on the server, so the harness
// overrides that rather than calling setState.
const initialState = useEditor.getInitialState();
const setStore = (patch) => {
  useEditor.setState(patch);
  Object.assign(initialState, patch);
};
setStore({ circuit, selection: [] });
let html = renderToStaticMarkup(React.createElement(EphysPanel, { onClose: () => {} }));
console.log('empty-state ok, has prompt:', html.includes('No cell to record from'), 'len', html.length);

// Closed
console.log('closed renders null:', renderToStaticMarkup(React.createElement(EphysPanel, { open: false })) === '');

// Selected — every protocol
setStore({ circuit, selection: ['a'] });
for (const p of ['fi','iv','tau','adaptation','ppr','rheobase']) {
  const Harness = () => {
    const [, force] = React.useState(0);
    return React.createElement(EphysPanel, { onClose: () => force(1) });
  };
  // Drive the protocol select by rendering with an initial state override is not
  // possible from outside, so just render the default and confirm no throw.
  html = renderToStaticMarkup(React.createElement(EphysPanel, {}));
  if (p === 'fi') {
    console.log('selected render len', html.length,
      '| swatch', html.includes(S.identityColorHex(1234)),
      '| run label', html.includes('Run F–I curve'),
      '| cond', /\d+ cond/.test(html),
      '| isolation note', html.includes('characterised alone'));
  }
}

// Multi-selection note
setStore({ circuit, selection: ['a','b'] });
html = renderToStaticMarkup(React.createElement(EphysPanel, {}));
console.log('multi-select note:', html.includes('cells are selected'));

// Cell with no outgoing synapses -> ppr blocked path is only reachable after a
// protocol switch, but the render must still be clean.
setStore({ circuit, selection: ['c'] });
html = renderToStaticMarkup(React.createElement(EphysPanel, {}));
console.log('adex cell render len', html.length, '| model', html.includes('adex'));

setStore({ circuit, selection: ['a'] });
const dump = renderToStaticMarkup(React.createElement(EphysPanel, {}));
console.log('---- HTML ----');
console.log(dump.replace(/></g, '>\n<').slice(0, 4000));
