/**
 * The keyboard map.
 *
 * `buildShortcuts` returns descriptors only — matching a key event against
 * `keys` and deciding whether the focus is in a text field is the app's job.
 * Each descriptor's `run` is a real action, not a hook: document edits go
 * straight to the store, and the four actions that belong to systems the editor
 * does not own (the transport, the camera and version history) are published as
 * a `neuroforge:shortcut` CustomEvent on `window`.
 *
 * Event contract, for whoever wires it up:
 *
 *   window.addEventListener('neuroforge:shortcut', (event) => {
 *     switch (event.detail.action) {
 *       case 'play-pause':      // toggle SimulationEngine.play()/pause()
 *       case 'reset':           // SimulationEngine.reset()
 *       case 'frame-selection': // CameraRig.frame() over the current selection
 *       case 'snapshot':        // io.createSnapshot()
 *     }
 *   });
 *
 * `keys` uses a canonical shape: `Mod` is Cmd on macOS and Ctrl elsewhere,
 * modifiers are ordered `Mod`, `Shift`, `Alt`, and the final segment is the
 * KeyboardEvent `key` value with single letters upper-cased.
 */

import type { Tool } from './store';
import { useEditor } from './store';

export interface Shortcut {
  keys: string;
  label: string;
  group: string;
  run(): void;
}

const SHORTCUT_EVENT = 'neuroforge:shortcut';

type ExternalAction = 'play-pause' | 'reset' | 'frame-selection' | 'snapshot';

function emit(action: ExternalAction): void {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(SHORTCUT_EVENT, { detail: { action } }));
}

const GROUP = {
  transport: 'Transport',
  tools: 'Tools',
  edit: 'Edit',
  selection: 'Selection',
  view: 'View',
  file: 'File',
} as const;

/** The five tools the number row reaches, in order. */
const NUMBERED_TOOLS: readonly { tool: Tool; label: string }[] = [
  { tool: 'select', label: 'Select tool' },
  { tool: 'place', label: 'Place tool' },
  { tool: 'connect', label: 'Connect tool' },
  { tool: 'erase', label: 'Erase tool' },
  { tool: 'probe', label: 'Probe tool' },
];

function deleteSelection(): void {
  const { selection, selectedSynapses, removeNeurons, removeSynapses } = useEditor.getState();
  // Synapses first: removing neurons already drops the synapses attached to
  // them, and doing it in this order keeps both edits from fighting over the
  // same records.
  if (selectedSynapses.length > 0) removeSynapses(selectedSynapses);
  if (selection.length > 0) removeNeurons(selection);
}

function toggleGrid(): void {
  const state = useEditor.getState();
  state.setRenderSettings({ gridVisible: !state.circuit.render.gridVisible });
}

function escape(): void {
  const state = useEditor.getState();
  if (state.commandPaletteOpen) {
    state.togglePanel('commandPalette', false);
    return;
  }
  state.clearSelection();
}

export function buildShortcuts(): Shortcut[] {
  const shortcuts: Shortcut[] = [
    {
      keys: 'Space',
      label: 'Play / pause',
      group: GROUP.transport,
      run: () => emit('play-pause'),
    },
    {
      keys: 'R',
      label: 'Reset simulation',
      group: GROUP.transport,
      run: () => emit('reset'),
    },
  ];

  NUMBERED_TOOLS.forEach((entry, index) => {
    shortcuts.push({
      keys: String(index + 1),
      label: entry.label,
      group: GROUP.tools,
      run: () => useEditor.getState().setTool(entry.tool),
    });
  });

  shortcuts.push(
    {
      keys: 'Mod+Z',
      label: 'Undo',
      group: GROUP.edit,
      run: () => useEditor.getState().undo(),
    },
    {
      keys: 'Mod+Shift+Z',
      label: 'Redo',
      group: GROUP.edit,
      run: () => useEditor.getState().redo(),
    },
    {
      keys: 'Mod+A',
      label: 'Select all',
      group: GROUP.selection,
      run: () => useEditor.getState().selectAll(),
    },
  );

  // Both erase keys are one command with two bindings, so they cannot drift.
  for (const key of ['Delete', 'Backspace']) {
    shortcuts.push({
      keys: key,
      label: 'Delete selection',
      group: GROUP.edit,
      run: deleteSelection,
    });
  }

  shortcuts.push(
    {
      keys: 'F',
      label: 'Frame selection',
      group: GROUP.view,
      run: () => emit('frame-selection'),
    },
    {
      keys: 'G',
      label: 'Toggle grid',
      group: GROUP.view,
      run: toggleGrid,
    },
    {
      keys: 'Mod+K',
      label: 'Command palette',
      group: GROUP.view,
      run: () => useEditor.getState().togglePanel('commandPalette'),
    },
    {
      keys: 'Mod+S',
      label: 'Snapshot',
      group: GROUP.file,
      run: () => emit('snapshot'),
    },
    {
      keys: 'Escape',
      label: 'Clear selection',
      group: GROUP.selection,
      run: escape,
    },
  );

  return shortcuts;
}
