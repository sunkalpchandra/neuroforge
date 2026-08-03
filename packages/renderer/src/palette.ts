import * as THREE from 'three';
import { COLORS, RECEPTOR_COLORS, RECEPTOR_FROM_CODE, srgbToLinear } from '@neuroforge/shared';

/**
 * The whole scene is authored in linear light: every custom program in
 * `@neuroforge/shaders` writes `gl_FragColor` without a tone-mapping or
 * colour-space chunk, because the encode happens once at the end of the
 * post-processing chain. Handing a material an sRGB triple would therefore
 * brighten it twice, so every colour that reaches a uniform comes through here.
 */
export function linearColor(hex: string): THREE.Color {
  const [r, g, b] = srgbToLinear(hex);
  return new THREE.Color(r, g, b);
}

export const BACKGROUND = COLORS.background;

/** Linear RGB triples indexed by `RECEPTOR_CODE`. */
export const RECEPTOR_LINEAR: Float32Array = buildReceptorTable();

function buildReceptorTable(): Float32Array {
  const table = new Float32Array(RECEPTOR_FROM_CODE.length * 3);
  for (let i = 0; i < RECEPTOR_FROM_CODE.length; i += 1) {
    const [r, g, b] = srgbToLinear(RECEPTOR_COLORS[RECEPTOR_FROM_CODE[i]]);
    table[i * 3] = r;
    table[i * 3 + 1] = g;
    table[i * 3 + 2] = b;
  }
  return table;
}

/** Clamped receptor lookup; an out-of-range code falls back to AMPA. */
export function receptorOffset(code: number): number {
  const count = RECEPTOR_FROM_CODE.length;
  const index = code >= 0 && code < count ? code : 0;
  return index * 3;
}

/**
 * Device-pixel dimensions of the drawing buffer.
 *
 * Two of the shader programs need it (ribbon minimum pixel width, point sprite
 * attenuation) but neither `update()` signature carries a renderer, so it is
 * read from the window and cached. Outside a browser it degrades to a sane
 * 1080p guess rather than producing NaN uniforms.
 */
export function viewportWidth(): number {
  if (typeof window === 'undefined') return 1920;
  return Math.max(1, window.innerWidth * Math.min(window.devicePixelRatio || 1, 2));
}

export function viewportHeight(): number {
  if (typeof window === 'undefined') return 1080;
  return Math.max(1, window.innerHeight * Math.min(window.devicePixelRatio || 1, 2));
}
