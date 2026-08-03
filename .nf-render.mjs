import { createJiti } from 'jiti';
import { renderToStaticMarkup } from 'react-dom/server';
import React from 'react';
const root = '/Users/sunkalp/neuroforge/';
const alias = {};
for (const p of ['shared','math','physics','simulation','editor','ui','io','ai','renderer','shaders']) {
  alias['@neuroforge/' + p] = root + 'packages/' + p + '/src/index.ts';
}
alias['@'] = root + 'apps/web/src';
const jiti = createJiti(root + 'noop.js', { alias, interopDefault: true, jsx: { runtime: 'automatic' } });
const mod = await jiti.import(root + 'apps/web/src/components/analysis/cell-types-panel.tsx');
console.log('exports:', Object.keys(mod));
const html = renderToStaticMarkup(React.createElement(mod.CellTypesPanel, { onClose: () => {} }));
console.log('length', html.length);
console.log(html.slice(0, 700));
console.log('closed=', html.includes('Close cell types panel'), 'title=', html.includes('Cell types'));
const hidden = renderToStaticMarkup(React.createElement(mod.CellTypesPanel, { open: false }));
console.log('closed render length', hidden.length);
