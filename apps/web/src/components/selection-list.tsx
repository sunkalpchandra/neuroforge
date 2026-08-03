'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Crosshair, ListX, X } from 'lucide-react';
import { IconButton, Tooltip } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { identityColorHex } from '@neuroforge/shared';
import type { NeuronId } from '@neuroforge/shared';

import { getEngine } from '@/lib/runtime';
import { fixed } from '@/lib/format';

/** Rows rendered before the list is truncated with a count. */
const MAX_ROWS = 300;

/**
 * The selected-cell list.
 *
 * Neuroglancer's segment list is the reason a connectome is traceable: it ties a
 * colour you can see in the scene to an identity you can act on. The swatch here
 * is the identical colour the renderer tints that cell with, both derived from
 * the neuron's procedural seed, so the correspondence is exact rather than
 * approximate.
 */
export function SelectionList() {
  const selection = useEditor((s) => s.selection);
  const neurons = useEditor((s) => s.circuit.neurons);
  const select = useEditor((s) => s.select);
  const clearSelection = useEditor((s) => s.clearSelection);
  const [rates, setRates] = useState<Float32Array>(() => new Float32Array(0));

  const rows = useMemo(() => {
    if (selection.length === 0) return [];
    const index = new Map<string, (typeof neurons)[number]>();
    for (const neuron of neurons) index.set(neuron.id, neuron);
    return selection
      .slice(0, MAX_ROWS)
      .map((id) => index.get(id))
      .filter((n): n is (typeof neurons)[number] => n !== undefined);
  }, [selection, neurons]);

  // Live firing rates are read off the simulation buffers on a timer rather than
  // per frame: this list can be hundreds of rows and none of them need 144 Hz.
  useEffect(() => {
    if (rows.length === 0) {
      setRates(new Float32Array(0));
      return;
    }
    const sample = () => {
      const engine = getEngine();
      const next = new Float32Array(rows.length);
      for (let i = 0; i < rows.length; i += 1) {
        const slot = engine.slotOf(rows[i].id);
        next[i] = slot >= 0 ? engine.buffers.neurons.rate[slot] : 0;
      }
      setRates(next);
    };
    sample();
    const id = setInterval(sample, 200);
    return () => clearInterval(id);
  }, [rows]);

  const focus = useCallback(
    (id: NeuronId) => {
      select([id]);
    },
    [select],
  );

  const remove = useCallback(
    (id: NeuronId) => {
      select(selection.filter((s) => s !== id));
    },
    [select, selection],
  );

  if (selection.length === 0) return null;

  return (
    <div className="nf-glass pointer-events-auto absolute right-3 bottom-3 flex max-h-[42vh] w-[264px] flex-col rounded-panel">
      <div className="flex items-center gap-1.5 border-b border-hairline px-2 py-1.5">
        <Crosshair className="size-3 text-ink-faint" />
        <span className="text-[10px] tracking-[0.08em] text-ink-faint uppercase">Selected</span>
        <span className="nf-numeric text-[10px] text-accent">{selection.length}</span>
        <div className="flex-1" />
        <Tooltip content="Clear selection" shortcut="Esc">
          <IconButton label="Clear selection" size="sm" onClick={clearSelection}>
            <ListX />
          </IconButton>
        </Tooltip>
      </div>

      <ul className="min-h-0 flex-1 overflow-y-auto py-0.5">
        {rows.map((neuron, i) => (
          <li key={neuron.id}>
            <div className="group flex items-center gap-1.5 px-2 py-[3px] transition-colors hover:bg-panel-raised">
              <span
                aria-hidden
                className="size-2.5 shrink-0 rounded-[2px]"
                style={{
                  backgroundColor: identityColorHex(neuron.morphology.seed),
                  boxShadow: `0 0 6px ${identityColorHex(neuron.morphology.seed)}55`,
                }}
              />
              <button
                type="button"
                onClick={() => focus(neuron.id)}
                className="min-w-0 flex-1 truncate text-left text-[11px] text-ink"
              >
                {neuron.label || neuron.id.slice(0, 10)}
              </button>
              <span className="nf-numeric shrink-0 text-[10px] text-ink-faint">
                {fixed(rates[i] ?? 0, 1)} Hz
              </span>
              <button
                type="button"
                aria-label={`Remove ${neuron.label || neuron.id} from selection`}
                onClick={() => remove(neuron.id)}
                className="shrink-0 text-ink-faint opacity-0 transition-opacity group-hover:opacity-100 hover:text-danger focus-visible:opacity-100"
              >
                <X className="size-3" />
              </button>
            </div>
          </li>
        ))}
        {selection.length > rows.length ? (
          <li className="px-2 py-1 text-[10px] text-ink-faint">
            +{selection.length - rows.length} more selected
          </li>
        ) : null}
      </ul>
    </div>
  );
}
