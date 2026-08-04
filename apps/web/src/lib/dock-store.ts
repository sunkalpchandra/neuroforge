'use client';

import { create } from 'zustand';

export type DockSide = 'left' | 'right' | 'bottom';

/** Registered tab identifiers, kept as a union so a typo cannot open nothing. */
export type DockTab =
  | 'builder'
  | 'search'
  | 'cell-types'
  | 'inspector'
  | 'analysis'
  | 'pathways'
  | 'connectivity'
  | 'query'
  | 'graph'
  | 'stats'
  | 'regions'
  | 'ephys'
  | 'network-experiments'
  | 'library'
  | 'raster';

interface SideState {
  /** Active tab, or null when the side is collapsed to its rail. */
  active: DockTab | null;
  /** Width for the vertical docks, height for the bottom one, in CSS pixels. */
  size: number;
}

export interface DockState {
  left: SideState;
  right: SideState;
  bottom: SideState;
  /** Toggle a tab: opening it, or collapsing the side if it is already active. */
  toggle: (side: DockSide, tab: DockTab) => void;
  open: (side: DockSide, tab: DockTab) => void;
  collapse: (side: DockSide) => void;
  resize: (side: DockSide, size: number) => void;
}

/** Bounds per side, so a drag cannot leave a dock unusably small or eat the canvas. */
const LIMITS: Record<DockSide, { min: number; max: number }> = {
  left: { min: 240, max: 560 },
  right: { min: 260, max: 620 },
  bottom: { min: 120, max: 480 },
};

function clampSize(side: DockSide, size: number): number {
  const { min, max } = LIMITS[side];
  return size < min ? min : size > max ? max : size;
}

/**
 * Which panel is docked where.
 *
 * Panels were previously absolutely positioned over the canvas, which works for
 * three of them and collapses at a dozen: they stack on the same corners and
 * cover each other. Docking makes the count survivable — one panel visible per
 * side, chosen from a rail — and keeps the centre of the viewport clear, which is
 * where the connectome is.
 */
export const useDock = create<DockState>((set) => ({
  left: { active: 'builder', size: 300 },
  right: { active: 'inspector', size: 340 },
  bottom: { active: null, size: 200 },

  toggle: (side, tab) =>
    set((state) => ({
      [side]: {
        ...state[side],
        active: state[side].active === tab ? null : tab,
      },
    }) as Pick<DockState, DockSide>),

  open: (side, tab) =>
    set((state) => ({ [side]: { ...state[side], active: tab } }) as Pick<DockState, DockSide>),

  collapse: (side) =>
    set((state) => ({ [side]: { ...state[side], active: null } }) as Pick<DockState, DockSide>),

  resize: (side, size) =>
    set((state) => ({
      [side]: { ...state[side], size: clampSize(side, size) },
    }) as Pick<DockState, DockSide>),
}));
