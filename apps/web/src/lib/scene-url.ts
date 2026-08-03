'use client';

import { COLOR_MODES } from '@neuroforge/shared';
import type { CameraState, ColorMode, RenderSettings } from '@neuroforge/shared';

import type { DockTab } from './dock-store';

/**
 * Scene state encoded into the URL fragment.
 *
 * Neuroglancer's defining affordance is that a view is a link: the camera, the
 * selected segments and every display setting live in the URL, so a finding can
 * be sent to a colleague rather than described to them. This is that, adapted —
 * the circuit itself is far too large for a URL, so a link carries the *view* and
 * names the document it belongs to, and says so plainly when opened against a
 * different one rather than silently restoring a nonsensical selection.
 *
 * The fragment is used rather than the query string because a static host never
 * sees it, so no link ever causes a page fetch that 404s.
 */

/** Version prefix, so a future format change can be detected rather than crash. */
const VERSION = 1;

/**
 * Cap on encoded selection size. Ids run ~14 characters; a few hundred keeps the
 * link inside what mail clients and chat apps forward without truncating.
 */
const MAX_SELECTION = 400;

export interface SceneState {
  circuitId: string;
  camera: Pick<CameraState, 'position' | 'target' | 'fov' | 'mode'>;
  selection: readonly string[];
  /** True when the selection was longer than MAX_SELECTION and was clipped. */
  selectionTruncated: boolean;
  render: Pick<
    RenderSettings,
    | 'colorMode'
    | 'dimUnselected'
    | 'saturation'
    | 'bloomIntensity'
    | 'showDendrites'
    | 'showAxons'
    | 'showSynapses'
    | 'showParticles'
    | 'gridVisible'
    | 'depthOfField'
    | 'ambientOcclusion'
  >;
  docks: { left: DockTab | null; right: DockTab | null; bottom: DockTab | null };
}

/** Compact wire form; short keys because every byte is a URL character. */
interface Wire {
  v: number;
  c: string;
  p: [number, number, number];
  t: [number, number, number];
  f: number;
  m: string;
  s: string[];
  st: 0 | 1;
  r: {
    cm: string;
    d: number;
    sa: number;
    b: number;
    fl: number;
    dof: 0 | 1;
    ao: 0 | 1;
  };
  dk: [string | null, string | null, string | null];
}

/** Layer visibility packed into one integer, five booleans being five characters otherwise. */
const LAYER_BITS = {
  dendrites: 1,
  axons: 2,
  synapses: 4,
  particles: 8,
  grid: 16,
} as const;

function round(value: number, places: number): number {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}

function toBase64Url(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(text: string): string | null {
  try {
    const padded = text.replace(/-/g, '+').replace(/_/g, '/');
    const binary = atob(padded + '='.repeat((4 - (padded.length % 4)) % 4));
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
}

export function encodeScene(state: SceneState): string {
  const layers =
    (state.render.showDendrites ? LAYER_BITS.dendrites : 0) |
    (state.render.showAxons ? LAYER_BITS.axons : 0) |
    (state.render.showSynapses ? LAYER_BITS.synapses : 0) |
    (state.render.showParticles ? LAYER_BITS.particles : 0) |
    (state.render.gridVisible ? LAYER_BITS.grid : 0);

  const selection = state.selection.slice(0, MAX_SELECTION);

  const wire: Wire = {
    v: VERSION,
    c: state.circuitId,
    // Three decimals is well under a pixel at any camera distance this app
    // reaches, and halves the length of the encoded camera.
    p: [
      round(state.camera.position.x, 3),
      round(state.camera.position.y, 3),
      round(state.camera.position.z, 3),
    ],
    t: [
      round(state.camera.target.x, 3),
      round(state.camera.target.y, 3),
      round(state.camera.target.z, 3),
    ],
    f: round(state.camera.fov, 2),
    m: state.camera.mode,
    s: [...selection],
    st: state.selection.length > MAX_SELECTION ? 1 : 0,
    r: {
      cm: state.render.colorMode,
      d: round(state.render.dimUnselected, 3),
      sa: round(state.render.saturation, 3),
      b: round(state.render.bloomIntensity, 3),
      fl: layers,
      dof: state.render.depthOfField ? 1 : 0,
      ao: state.render.ambientOcclusion ? 1 : 0,
    },
    dk: [state.docks.left, state.docks.right, state.docks.bottom],
  };

  return toBase64Url(JSON.stringify(wire));
}

function isVec3(value: unknown): value is [number, number, number] {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((v) => typeof v === 'number' && Number.isFinite(v))
  );
}

const CAMERA_MODES = new Set(['orbit', 'fly', 'first-person', 'cinematic']);

/**
 * Decode a fragment back into scene state.
 *
 * Returns null rather than a partial state for anything malformed: a link is
 * untrusted input, and half-restoring a view is more confusing than ignoring it.
 */
export function decodeScene(fragment: string): SceneState | null {
  const json = fromBase64Url(fragment);
  if (json === null) return null;

  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    return null;
  }
  if (typeof raw !== 'object' || raw === null) return null;

  const w = raw as Partial<Wire>;
  if (w.v !== VERSION) return null;
  if (typeof w.c !== 'string') return null;
  if (!isVec3(w.p) || !isVec3(w.t)) return null;
  if (typeof w.f !== 'number' || !Number.isFinite(w.f)) return null;
  if (typeof w.m !== 'string' || !CAMERA_MODES.has(w.m)) return null;
  if (!Array.isArray(w.s) || w.s.some((id) => typeof id !== 'string')) return null;
  if (typeof w.r !== 'object' || w.r === null) return null;

  const r = w.r;
  const colorMode = (COLOR_MODES as readonly string[]).includes(r.cm)
    ? (r.cm as ColorMode)
    : 'identity';
  const layers = typeof r.fl === 'number' ? r.fl : 0;
  const clamp01 = (value: unknown, fallback: number): number =>
    typeof value === 'number' && Number.isFinite(value)
      ? Math.min(1, Math.max(0, value))
      : fallback;

  const dockAt = (index: number): DockTab | null => {
    const value = Array.isArray(w.dk) ? w.dk[index] : null;
    return typeof value === 'string' ? (value as DockTab) : null;
  };

  return {
    circuitId: w.c,
    camera: {
      position: { x: w.p[0], y: w.p[1], z: w.p[2] },
      target: { x: w.t[0], y: w.t[1], z: w.t[2] },
      fov: Math.min(170, Math.max(5, w.f)),
      mode: w.m as CameraState['mode'],
    },
    selection: w.s.slice(0, MAX_SELECTION),
    selectionTruncated: w.st === 1,
    render: {
      colorMode,
      dimUnselected: clamp01(r.d, 0.18),
      saturation:
        typeof r.sa === 'number' && Number.isFinite(r.sa)
          ? Math.min(2, Math.max(0, r.sa))
          : 1,
      bloomIntensity:
        typeof r.b === 'number' && Number.isFinite(r.b) ? Math.max(0, r.b) : 0.3,
      showDendrites: (layers & LAYER_BITS.dendrites) !== 0,
      showAxons: (layers & LAYER_BITS.axons) !== 0,
      showSynapses: (layers & LAYER_BITS.synapses) !== 0,
      showParticles: (layers & LAYER_BITS.particles) !== 0,
      gridVisible: (layers & LAYER_BITS.grid) !== 0,
      depthOfField: r.dof === 1,
      ambientOcclusion: r.ao === 1,
    },
    docks: { left: dockAt(0), right: dockAt(1), bottom: dockAt(2) },
  };
}

/** Read the scene out of the current location, if there is one. */
export function readSceneFromLocation(): SceneState | null {
  if (typeof window === 'undefined') return null;
  const hash = window.location.hash;
  if (!hash.startsWith('#!')) return null;
  return decodeScene(hash.slice(2));
}

/**
 * Write the scene into the address bar without adding a history entry.
 *
 * replaceState rather than pushState: orbiting a camera should not fill the back
 * button with a hundred intermediate views.
 */
export function writeSceneToLocation(state: SceneState): string {
  const fragment = `#!${encodeScene(state)}`;
  if (typeof window !== 'undefined') {
    window.history.replaceState(null, '', fragment);
  }
  return fragment;
}

export function sceneLink(state: SceneState): string {
  if (typeof window === 'undefined') return '';
  const { origin, pathname, search } = window.location;
  return `${origin}${pathname}${search}#!${encodeScene(state)}`;
}
