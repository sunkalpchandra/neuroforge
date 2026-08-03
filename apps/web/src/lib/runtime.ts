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

/* ------------------------------------------------------------- camera ----- */

export interface FrameRequest {
  min: [number, number, number];
  max: [number, number, number];
}

let pendingFrame: FrameRequest | null = null;

/**
 * Ask the camera to ease onto a bounding box.
 *
 * The rig is owned by the R3F scene and the request comes from the keyboard map
 * outside it, so it is left here for the render loop to pick up rather than
 * reaching across the boundary. Only the most recent request survives: framing
 * twice before a frame renders should land on the second target, not animate
 * through the first.
 */
export function requestCameraFrame(request: FrameRequest): void {
  pendingFrame = request;
}

/** Take the pending frame request, if any. Called once per rendered frame. */
export function consumeCameraFrame(): FrameRequest | null {
  const request = pendingFrame;
  pendingFrame = null;
  return request;
}

/**
 * Bounding box of a set of neurons, read from the live positions.
 * Returns null when nothing in `ids` resolves to a live slot.
 */
export function boundsOf(ids: readonly string[]): FrameRequest | null {
  const engine = getEngine();
  const { neurons } = engine.buffers;
  const position = neurons.position;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let found = 0;

  const visit = (slot: number): void => {
    const p = slot * 3;
    const x = position[p];
    const y = position[p + 1];
    const z = position[p + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
    found += 1;
  };

  if (ids.length === 0) {
    // Framing with an empty selection frames the whole network, which is what
    // every 3D tool does and what a user pressing F on nothing expects.
    for (let i = 0; i < neurons.count; i += 1) visit(i);
  } else {
    for (const id of ids) {
      const slot = engine.slotOf(id);
      if (slot >= 0 && slot < neurons.count) visit(slot);
    }
  }

  if (found === 0) return null;
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
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
