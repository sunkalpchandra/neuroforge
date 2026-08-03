'use client';

import { useCallback, useEffect } from 'react';
import { ToastViewport, pushToast } from '@neuroforge/ui';
import { buildShortcuts, useEditor } from '@neuroforge/editor';
import { Autosaver, createSnapshot } from '@neuroforge/io';
import type { NeuronId } from '@neuroforge/shared';

import { AiBuilder } from './builder/ai-builder';
import { CommandPalette } from './command-palette';
import { NetworkAnalysis } from './analysis/network-analysis';
import { SearchPanel } from './search/search-panel';
import { RasterPlot } from './analysis/raster-plot';
import { LibraryPanel } from './library/library-panel';
import { SelectionList } from './selection-list';
import { ViewControls } from './view-controls';
import { Inspector } from './inspector/inspector';
import { StatusBar } from './status-bar';
import { TopBar } from './top-bar';
import { Viewport } from './viewport';
import { boundsOf, getEngine, requestCameraFrame, syncSelectionFlags } from '@/lib/runtime';

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
  const selection = useEditor((s) => s.selection);
  const hovered = useEditor((s) => s.hovered);

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
    // load() zeroes the flag column, so the selection has to be republished.
    syncSelectionFlags(selection, hovered);
  }, [circuit.neurons, circuit.synapses, circuit.id, selection, hovered]);

  useEffect(() => {
    syncSelectionFlags(selection, hovered);
  }, [selection, hovered]);

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


  // buildShortcuts() dispatches these four as window events rather than acting
  // directly, because the actions live outside the document the editor owns.
  // Nothing was listening, so Space, R, F and Mod+S were inert.
  useEffect(() => {
    const onAction = (event: Event) => {
      const detail = (event as CustomEvent<{ action?: string }>).detail;
      switch (detail?.action) {
        case 'play-pause': {
          const engine = getEngine();
          if (engine.running) engine.pause();
          else engine.play();
          break;
        }
        case 'reset':
          getEngine().reset();
          break;
        case 'frame-selection': {
          const bounds = boundsOf(useEditor.getState().selection);
          if (bounds) requestCameraFrame(bounds);
          break;
        }
        case 'snapshot': {
          const current = useEditor.getState().circuit;
          void createSnapshot(current, `Manual — ${current.name}`, false)
            .then(() => pushToast({ tone: 'success', title: 'Snapshot saved' }))
            .catch((error: unknown) =>
              pushToast({
                tone: 'danger',
                title: 'Snapshot failed',
                description: error instanceof Error ? error.message : String(error),
              }),
            );
          break;
        }
        default:
          break;
      }
    };
    window.addEventListener('neuroforge:shortcut', onAction);
    return () => window.removeEventListener('neuroforge:shortcut', onAction);
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
          <AiBuilder />
          <Inspector />
          <ViewControls />
          <NetworkAnalysis />
          <SearchPanel />
          <SelectionList />
          <LibraryPanel />
        </div>
      </main>
      {/* Docked rather than floated: the raster is read against the scene while
          the simulation runs, so it takes its own band of the column instead of
          covering the cells it is reporting on. */}
      <RasterPlot />
      <StatusBar />
      {/* Outside the panel layer: it portals to the body and owns the whole
          viewport while it is open, so it must not inherit pointer-events-none. */}
      <CommandPalette />
      <ToastViewport position="bottom-right" />
    </div>
  );
}
