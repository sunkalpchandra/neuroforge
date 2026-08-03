import type { UniformTable } from '../types';
import { FOG_GLSL } from './common';

/**
 * Infinite analytic grid, drawn on one screen-covering quad.
 *
 * The mesh carries no grid geometry at all: it is a `PlaneGeometry(2, 2)` whose
 * `position.xy` is already in normalised device coordinates. The vertex stage
 * unprojects the near and far plane points of each corner, the fragment stage
 * intersects the interpolated ray with the ground plane, and the lines are
 * evaluated from the hit point. Unprojecting both planes rather than starting
 * the ray at the camera keeps it correct under an orthographic projection too.
 */
export const GRID_VERTEX_GLSL = /* glsl */ `
uniform mat4 uInverseProjectionMatrix;
uniform mat4 uInverseViewMatrix;

varying vec3 vRayOrigin;
varying vec3 vRayDirection;

vec3 unproject(vec2 ndc, float depth) {
  vec4 view = uInverseProjectionMatrix * vec4(ndc, depth, 1.0);
  return (uInverseViewMatrix * vec4(view.xyz / view.w, 1.0)).xyz;
}

void main() {
  vec2 ndc = position.xy;
  vec3 nearPoint = unproject(ndc, -1.0);
  vec3 farPoint = unproject(ndc, 1.0);

  vRayOrigin = nearPoint;
  vRayDirection = farPoint - nearPoint;

  gl_Position = vec4(ndc, 1.0, 1.0);
}
`;

const GRID_FRAGMENT_BODY = /* glsl */ `
uniform mat4 projectionMatrix;

uniform vec3 uGridColor;
uniform vec3 uMajorColor;
uniform vec3 uAxisXColor;
uniform vec3 uAxisZColor;
uniform vec3 uFogColor;
uniform float uHeight;
uniform float uCellSize;
uniform float uMajorEvery;
uniform float uLineWidth;
uniform float uMajorWidth;
uniform float uAxisWidth;
uniform float uMinorOpacity;
uniform float uMajorOpacity;
uniform float uAxisOpacity;
uniform float uFadeStart;
uniform float uFadeEnd;
uniform float uOpacity;
uniform float uFogDensity;

varying vec3 vRayOrigin;
varying vec3 vRayDirection;

/**
 * Coverage of a line whose centre is distancePixels away, in pixels.
 *
 * Because the distance is measured in pixels the line keeps the same apparent
 * thickness at every zoom level and every grazing angle, which is the whole
 * point of doing this analytically instead of drawing geometry.
 */
float lineCoverage(float distancePixels, float halfWidthPixels) {
  float w = max(halfWidthPixels, 0.5);
  return 1.0 - smoothstep(w - 0.5, w + 0.5, distancePixels);
}

/** Distance in pixels from a grid-cell coordinate to the nearest cell boundary. */
float cellDistancePixels(vec2 coord, vec2 derivative) {
  vec2 d = abs(fract(coord - 0.5) - 0.5) / max(derivative, vec2(1e-8));
  return min(d.x, d.y);
}

void main() {
  vec3 origin = vRayOrigin;
  vec3 direction = normalize(vRayDirection);

  // The ray is intersected unconditionally, with the denominator nudged off
  // zero, so that neighbouring fragments still produce usable derivatives at the
  // horizon; the ones that missed the plane are discarded at the very end.
  float denominator = direction.y;
  float safeDenominator = (denominator >= 0.0 ? 1.0 : -1.0) * max(abs(denominator), 1e-6);
  float t = (uHeight - origin.y) / safeDenominator;
  vec3 hit = origin + direction * max(t, 0.0);
  vec2 p = hit.xz;

  float cell = max(uCellSize, 1e-4);
  float majorCell = cell * max(uMajorEvery, 1.0);

  vec2 minorCoord = p / cell;
  vec2 minorDerivative = fwidth(minorCoord);
  float minor = lineCoverage(cellDistancePixels(minorCoord, minorDerivative), uLineWidth);

  // Once a cell shrinks below a few pixels the minor lines can no longer be
  // resolved and would beat against the pixel grid. Fading them out there is
  // what keeps the grid free of moire at any distance.
  float pixelsPerCell = 1.0 / max(max(minorDerivative.x, minorDerivative.y), 1e-8);
  minor *= smoothstep(2.0, 10.0, pixelsPerCell);

  vec2 majorCoord = p / majorCell;
  float major = lineCoverage(cellDistancePixels(majorCoord, fwidth(majorCoord)), uMajorWidth);

  vec2 axisDerivative = fwidth(p);
  float axisAlongZ = lineCoverage(abs(p.x) / max(axisDerivative.x, 1e-8), uAxisWidth);
  float axisAlongX = lineCoverage(abs(p.y) / max(axisDerivative.y, 1e-8), uAxisWidth);

  vec3 color = mix(uGridColor, uMajorColor, major);
  color = mix(color, uAxisZColor, axisAlongZ);
  color = mix(color, uAxisXColor, axisAlongX);

  float alpha = max(minor * uMinorOpacity, major * uMajorOpacity);
  alpha = max(alpha, max(axisAlongX, axisAlongZ) * uAxisOpacity);

  float distanceToHit = length(hit - origin);
  float fade = 1.0 - clamp((distanceToHit - uFadeStart) / max(uFadeEnd - uFadeStart, 1e-4), 0.0, 1.0);
  // Suppress the last degree or two before the horizon, where a whole span of
  // grid collapses into one row of pixels.
  float grazing = clamp(abs(direction.y) * 6.0, 0.0, 1.0);
  alpha *= fade * grazing * uOpacity;

  if (t <= 0.0 || alpha < 0.002) {
    discard;
  }

  color = applyFog(color, distanceToHit, uFogDensity, uFogColor);

  // Written so the plane occludes and is occluded correctly rather than sitting
  // flat on the far plane. Assumes the default [-1,1] clip depth convention.
  vec4 clip = projectionMatrix * viewMatrix * vec4(hit, 1.0);
  gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);

  gl_FragColor = vec4(color, alpha);
}
`;

/** Self-contained: already includes FOG_GLSL. */
export const GRID_FRAGMENT_GLSL = [FOG_GLSL, GRID_FRAGMENT_BODY].join('\n');

/** Uniform names and types for the infinite grid material. */
export const GRID_UNIFORMS = {
  uInverseProjectionMatrix: 'mat4',
  uInverseViewMatrix: 'mat4',
  uGridColor: 'vec3',
  uMajorColor: 'vec3',
  uAxisXColor: 'vec3',
  uAxisZColor: 'vec3',
  uFogColor: 'vec3',
  uHeight: 'float',
  uCellSize: 'float',
  uMajorEvery: 'float',
  uLineWidth: 'float',
  uMajorWidth: 'float',
  uAxisWidth: 'float',
  uMinorOpacity: 'float',
  uMajorOpacity: 'float',
  uAxisOpacity: 'float',
  uFadeStart: 'float',
  uFadeEnd: 'float',
  uOpacity: 'float',
  uFogDensity: 'float',
} as const satisfies UniformTable;
