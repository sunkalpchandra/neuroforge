import type { UniformTable } from '../types';
import { FOG_GLSL } from './common';

/**
 * Additive point sprites for spike impulses.
 *
 * Per-vertex attributes alongside the `position` Three injects:
 *   aLife     float  1 at birth falling to 0 at death
 *   aSize     float  per-particle size multiplier
 *   aColor    vec3   linear-space tint
 *   aVelocity vec3   world units per second, used only to orient the streak
 *
 * `uScale` follows Three's own points convention: half the drawing buffer height
 * in device pixels. Size attenuation then matches `PointsMaterial`, so a particle
 * and a Three-native point of the same size agree at every depth.
 */
export const PARTICLE_VERTEX_GLSL = /* glsl */ `
attribute float aLife;
attribute float aSize;
attribute vec3 aColor;
attribute vec3 aVelocity;

uniform float uSize;
uniform float uScale;
uniform float uStretch;
uniform float uFadeIn;

varying vec3 vColor;
varying float vLife;
varying float vViewDepth;
varying vec2 vStreak;
varying float vStreakAmount;

void main() {
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  float life = clamp(aLife, 0.0, 1.0);

  // Ramp in over the first instant of the particle's life so emission does not
  // pop, then let the remaining lifetime carry the decay.
  float age = 1.0 - life;
  float envelope = life * smoothstep(0.0, max(uFadeIn, 1e-4), age);

  gl_PointSize = max(uSize * aSize * envelope * (uScale / max(-viewPosition.z, 1e-3)), 1.0);

  // A point sprite cannot be stretched by the rasteriser, so the direction of
  // travel is handed to the fragment stage and the sprite is elongated there.
  vec3 viewVelocity = mat3(modelViewMatrix) * aVelocity;
  vStreak = viewVelocity.xy + vec2(1e-6, 0.0);
  vStreakAmount = clamp(length(viewVelocity.xy) * uStretch, 0.0, 3.0);

  vColor = aColor;
  vLife = envelope;
  vViewDepth = -viewPosition.z;

  gl_Position = projectionMatrix * viewPosition;
}
`;

const PARTICLE_FRAGMENT_BODY = /* glsl */ `
uniform vec3 uCoreColor;
uniform vec3 uFogColor;
uniform float uSoftness;
uniform float uIntensity;
uniform float uFogDensity;

varying vec3 vColor;
varying float vLife;
varying float vViewDepth;
varying vec2 vStreak;
varying float vStreakAmount;

void main() {
  vec2 uv = gl_PointCoord * 2.0 - 1.0;

  // Rotate into the frame of the direction of travel and stretch along it, which
  // turns the round sprite into a motion-blurred comet without extra geometry.
  vec2 forward = normalize(vStreak);
  vec2 sideways = vec2(-forward.y, forward.x);
  vec2 local = vec2(dot(uv, forward) / (1.0 + vStreakAmount), dot(uv, sideways));

  float r2 = dot(local, local);
  if (r2 > 1.0) {
    discard;
  }

  float falloff = exp(-r2 * max(uSoftness, 0.0));
  float core = pow(falloff, 6.0);
  vec3 color = mix(vColor, uCoreColor, core) * uIntensity * falloff * vLife;
  color = applyFog(color, vViewDepth, uFogDensity, uFogColor);

  gl_FragColor = vec4(color, falloff * vLife);
}
`;

/** Self-contained: already includes FOG_GLSL. */
export const PARTICLE_FRAGMENT_GLSL = [FOG_GLSL, PARTICLE_FRAGMENT_BODY].join('\n');

/** Uniform names and types for the spike particle material. */
export const PARTICLE_UNIFORMS = {
  uSize: 'float',
  uScale: 'float',
  uStretch: 'float',
  uFadeIn: 'float',
  uCoreColor: 'vec3',
  uFogColor: 'vec3',
  uSoftness: 'float',
  uIntensity: 'float',
  uFogDensity: 'float',
} as const satisfies UniformTable;
