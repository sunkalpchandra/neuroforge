'use client';

import { useCallback, useEffect } from 'react';
import { ToastViewport } from '@neuroforge/ui';
import { buildShortcuts, useEditor } from '@neuroforge/editor';
import { Autosaver } from '@neuroforge/io';
import type { NeuronId } from '@neuroforge/shared';

import { Inspector } from './inspector/inspector';
import { StatusBar } from './status-bar';
import { TopBar } from './top-bar';
import { Viewport } from './viewport';
import { getEngine } from '@/lib/runtime';

/** Match a KeyboardEvent against a shortcut descriptor like "Mod+Shift+Z". */
function matches(event: KeyboardEvent, keys: string): boolean {
  const parts = keys.toLowerCase().split('+');
  const key = parts[parts.length - 1];
  const wantMod = parts.includes('mod');
  const wantShift = parts.includes('shift');
  const wantAlt = parts.includes('alt');

  // Mod is Cmd on macOS and Ctrl elsewhere; accepting either would make Ctrl+Z
  // fire alongside Cmd+Z on a Mac and undo twice. navigator.platform is
  // deprecated, so this reads the user agent instead.
  const isMac =
    typeof navigator !== 'undefined' && /Mac|iPhone|iPad|iPod/.test(navigator.userAgent);
  const mod = isMac ? event.metaKey : event.ctrlKey;

  if (wantMod !== mod) return false;
  if (wantShift !== event.shiftKey) return false;
  if (wantAlt !== event.altKey) return false;

  const pressed = event.key.toLowerCase();
  if (key === 'space') return pressed === ' ';
  if (key === 'delete') return pressed === 'delete' || pressed === 'backspace';
  return pressed === key;
}

/**
 * The entire application.
 *
 * There is no routing: every surface is a panel composited over a canvas that
 * never unmounts, which is what keeps the simulation and the GPU resources alive
 * across every interaction.
 */
export function Workspace() {
  const circuit = useEditor((s) => s.circuit);
  const select = useEditor((s) => s.select);
  const clearSelection = useEditor((s) => s.clearSelection);

  // The renderer works in dense slots; the document works in ids. The engine
  // owns the mapping, so translation happens here rather than in either of them.
  const onPick = useCallback(
    (slot: number) => {
      if (slot < 0) {
        clearSelection();
        return;
      }
      const id = getEngine().idOf(slot);
      if (id) select([id as NeuronId]);
    },
    [clearSelection, select],
  );

  // Push the document into the engine whenever its structure changes. The
  // engine rebuilds its buffers, so this must not run on every parameter tweak —
  // the identity of the neuron and synapse arrays is the structural signal.
  useEffect(() => {
    getEngine().load(circuit);
    getEngine().play();
  }, [circuit.neurons, circuit.synapses, circuit.id]);

  useEffect(() => {
    const shortcuts = buildShortcuts();
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      // Never steal keys from a field the user is typing into.
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable)
      ) {
        return;
      }
      for (const shortcut of shortcuts) {
        if (matches(event, shortcut.keys)) {
          event.preventDefault();
          shortcut.run();
          return;
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    const autosaver = new Autosaver(1500);
    autosaver.start(
      () => useEditor.getState().circuit,
      () => undefined,
    );
    const unsubscribe = useEditor.subscribe(() => autosaver.touch());
    return () => {
      unsubscribe();
      void autosaver.flush();
      autosaver.stop();
    };
  }, []);

  return (
    <div className="flex h-dvh w-dvw flex-col overflow-hidden bg-bg">
      <TopBar />
      <main className="relative flex-1 overflow-hidden">
        <Viewport render={circuit.render} onPick={onPick} />
        {/* Panels float over the canvas and are pointer-transparent where they
            are empty, so dragging the scene never snags on panel gutters. */}
        <div className="pointer-events-none absolute inset-0">
          <Inspector />
        </div>
      </main>
      <StatusBar />
      <ToastViewport position="bottom-right" />
    </div>
  );
}
