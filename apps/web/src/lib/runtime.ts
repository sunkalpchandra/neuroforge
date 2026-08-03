import { ProbeRecorder, SimulationEngine } from '@neuroforge/simulation';
import { ForceLayout } from '@neuroforge/physics';
import type { FrameStats } from '@neuroforge/shared';
import { EMPTY_FRAME_STATS, NEURON_FLAG } from '@neuroforge/shared';

/**
 * The long-lived, non-React runtime.
 *
 * The simulation engine, the probe recorder and the layout solver all own large
 * typed arrays and must survive every re-render, Fast Refresh cycle and panel
 * toggle. Putting them in React state would mean reallocating a hundred
 * thousand neurons whenever a slider moves, so they live here as a module
 * singleton and React subscribes to their statistics instead of owning them.
 *
 * Construction is lazy because this module is imported by server-rendered code
 * during the static export, where `performance` and `navigator` exist but no
 * simulation should start.
 */

let engine: SimulationEngine | null = null;
let probes: ProbeRecorder | null = null;
let layout: ForceLayout | null = null;

export function getEngine(): SimulationEngine {
  if (engine === null) engine = new SimulationEngine();
  return engine;
}

export function getProbes(): ProbeRecorder {
  if (probes === null) probes = new ProbeRecorder(2048);
  return probes;
}

export function getLayout(): ForceLayout {
  if (layout === null) layout = new ForceLayout();
  return layout;
}

/**
 * Publish the editor's selection into the simulation buffers.
 *
 * The document tracks selection as ids; the renderer reads a per-neuron flag
 * bitfield, because a shader cannot look anything up in a Set. Nothing bridged
 * the two, so the selection halo and the dimming of unselected cells had no
 * input and never engaged. This is that bridge, called whenever the selection
 * changes rather than per frame.
 */
export function syncSelectionFlags(
  selection: readonly string[],
  hovered: string | null,
): void {
  const engine = getEngine();
  const { neurons } = engine.buffers;
  const count = neurons.count;
  if (count === 0) return;

  // Clearing only the bits this function owns leaves PROBED and GHOSTED, which
  // are set elsewhere, intact.
  const KEEP = ~(NEURON_FLAG.SELECTED | NEURON_FLAG.HOVERED);
  for (let i = 0; i < count; i += 1) neurons.flags[i] &= KEEP;

  for (const id of selection) {
    const slot = engine.slotOf(id);
    if (slot >= 0 && slot < count) neurons.flags[slot] |= NEURON_FLAG.SELECTED;
  }

  if (hovered !== null) {
    const slot = engine.slotOf(hovered);
    if (slot >= 0 && slot < count) neurons.flags[slot] |= NEURON_FLAG.HOVERED;
  }
}

/* ------------------------------------------------------------------ stats -- */

type StatsListener = () => void;

const listeners = new Set<StatsListener>();
let snapshot: FrameStats = EMPTY_FRAME_STATS;

/**
 * Statistics are republished at a fixed cadence rather than every frame.
 *
 * useSyncExternalStore re-renders every subscriber whenever the snapshot
 * identity changes. At 144 Hz that would re-render the entire status bar 144
 * times a second to move a number by a tenth, which costs more than the
 * simulation it is reporting on.
 */
const STATS_INTERVAL_MS = 100;
let lastPublish = 0;

export function publishStats(now: number): void {
  if (now - lastPublish < STATS_INTERVAL_MS) return;
  lastPublish = now;
  const next = getEngine().stats;
  // The engine replaces its stats object on every change, so identity is a
  // sufficient and cheap staleness check.
  if (next === snapshot) return;
  snapshot = next;
  for (const listener of listeners) listener();
}

export function subscribeStats(listener: StatsListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getStatsSnapshot(): FrameStats {
  return snapshot;
}

/** Server render has no engine; return the empty stats rather than constructing one. */
export function getStatsServerSnapshot(): FrameStats {
  return EMPTY_FRAME_STATS;
}

/** Release every runtime resource. Used by tests and by hard document resets. */
export function disposeRuntime(): void {
  engine?.dispose();
  engine = null;
  probes?.reset();
  probes = null;
  layout = null;
  snapshot = EMPTY_FRAME_STATS;
  lastPublish = 0;
}
