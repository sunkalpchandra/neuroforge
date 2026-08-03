/**
 * Design tokens. These are the single source of truth: the Tailwind theme block
 * in globals.css mirrors these values, and the renderer reads the numeric forms
 * directly so the 3D scene and the 2D chrome stay in agreement.
 */

export const COLORS = {
  background: '#07090B',
  panel: '#101418',
  panelRaised: '#151A20',
  border: 'rgba(255,255,255,0.05)',
  borderStrong: 'rgba(255,255,255,0.10)',
  text: '#F5F7FA',
  textMuted: '#8A93A0',
  textFaint: '#5A626D',
  accent: '#4FD1FF',
  secondary: '#B66BFF',
  success: '#4ADE80',
  warning: '#FBBF24',
  danger: '#FB7185',
} as const;

export type ColorToken = keyof typeof COLORS;

/** Linear-space RGB triples for the renderer, derived from COLORS. */
export const COLOR_RGB: Record<ColorToken, readonly [number, number, number]> = {
  background: srgbToLinear('#07090B'),
  panel: srgbToLinear('#101418'),
  panelRaised: srgbToLinear('#151A20'),
  border: [1, 1, 1],
  borderStrong: [1, 1, 1],
  text: srgbToLinear('#F5F7FA'),
  textMuted: srgbToLinear('#8A93A0'),
  textFaint: srgbToLinear('#5A626D'),
  accent: srgbToLinear('#4FD1FF'),
  secondary: srgbToLinear('#B66BFF'),
  success: srgbToLinear('#4ADE80'),
  warning: srgbToLinear('#FBBF24'),
  danger: srgbToLinear('#FB7185'),
};

/** Convert a #rrggbb string to a linear-space RGB triple. */
export function srgbToLinear(hex: string): [number, number, number] {
  const int = Number.parseInt(hex.replace('#', ''), 16);
  const r = ((int >> 16) & 0xff) / 255;
  const g = ((int >> 8) & 0xff) / 255;
  const b = (int & 0xff) / 255;
  const t = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return [t(r), t(g), t(b)];
}

/** Convert a #rrggbb string to a packed 0xRRGGBB integer. */
export function hexToInt(hex: string): number {
  return Number.parseInt(hex.replace('#', ''), 16);
}

/**
 * Motion tokens. Every animation in the product uses one of these; nothing
 * invents its own timing. Durations are in seconds to match Motion's API.
 */
export const MOTION = {
  /** Micro-interactions: hover, focus ring, icon state. */
  instant: 0.12,
  /** The house default called for in the brief. */
  base: 0.2,
  /** Panel open/close, tab switches. */
  panel: 0.28,
  /** Large surface transitions. */
  slow: 0.42,
} as const;

/** Spring presets used with Motion's `type: 'spring'`. */
export const SPRING = {
  /** Snappy, minimal overshoot — buttons, toggles. */
  crisp: { type: 'spring', stiffness: 520, damping: 38, mass: 0.8 },
  /** Default panel motion. */
  smooth: { type: 'spring', stiffness: 320, damping: 34, mass: 1 },
  /** Heavier surfaces, drawers. */
  soft: { type: 'spring', stiffness: 180, damping: 28, mass: 1.2 },
  /** Camera and world-space interpolation. */
  cinematic: { type: 'spring', stiffness: 90, damping: 22, mass: 1.4 },
} as const;

/** Standard cubic-bezier easings. */
export const EASE = {
  out: [0.16, 1, 0.3, 1],
  inOut: [0.65, 0, 0.35, 1],
  in: [0.55, 0, 1, 0.45],
} as const;

/** Z-index scale, kept explicit so overlays never fight. */
export const LAYER = {
  canvas: 0,
  overlay: 10,
  panel: 20,
  toolbar: 30,
  popover: 40,
  modal: 50,
  toast: 60,
  tooltip: 70,
} as const;

/** Colour ramp used to map membrane voltage to a hue in the scene. */
export const VOLTAGE_RAMP: readonly { stop: number; color: string }[] = [
  { stop: -90, color: '#1B2A4A' },
  { stop: -70, color: '#2E4A7D' },
  { stop: -55, color: '#4FD1FF' },
  { stop: -40, color: '#8AE0FF' },
  { stop: -20, color: '#B66BFF' },
  { stop: 0, color: '#F2D4FF' },
  { stop: 30, color: '#FFFFFF' },
];

/** Accent colours for the two polarities. */
export const POLARITY_COLORS = {
  excitatory: '#4FD1FF',
  inhibitory: '#B66BFF',
} as const;

/** Receptor colours used for synapse rendering and legends. */
export const RECEPTOR_COLORS = {
  ampa: '#4FD1FF',
  nmda: '#6EE7F9',
  gabaa: '#B66BFF',
  gabab: '#8B5CF6',
  gap: '#4ADE80',
} as const;
