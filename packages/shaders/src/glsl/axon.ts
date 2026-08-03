import type { UniformTable } from '../types';
import { FOG_GLSL } from './common';

/**
 * Evaluated identically here, in the particle compute kernels and by `sampleArc`
 * in `@neuroforge/math`: a quadratic Bezier from the presynaptic soma to the
 * postsynaptic soma whose control point is the midpoint displaced by `sag` along
 * world up, projected perpendicular to the chord and renormalised. Keeping one
 * definition is what stops the travelling impulse from drifting off the ribbon
 * it is supposed to be riding.
 */
const AXON_SPLINE_GLSL = /* glsl */ `
const float SPLINE_LENGTH_EPSILON = 1e-9;
const float SPLINE_PARALLEL_EPSILON = 1e-6;

vec3 axonControl(vec3 a, vec3 b, float sag) {
  vec3 chord = b - a;
  float chordLength = length(chord);
  vec3 normal = vec3(0.0, 1.0, 0.0);
  if (chordLength > SPLINE_LENGTH_EPSILON) {
    vec3 u = chord / chordLength;
    vec3 reference = vec3(0.0, 1.0, 0.0);
    if (u.y * u.y > 1.0 - SPLINE_PARALLEL_EPSILON) {
      reference = abs(u.x) <= abs(u.z) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    }
    vec3 perpendicular = reference - u * dot(u, reference);
    float perpendicularLength = length(perpendicular);
    if (perpendicularLength > SPLINE_LENGTH_EPSILON) {
      normal = perpendicular / perpendicularLength;
    }
  }
  return mix(a, b, 0.5) + normal * sag;
}

vec3 axonPoint(vec3 a, vec3 b, float sag, float t) {
  vec3 c = axonControl(a, b, sag);
  float s = 1.0 - t;
  return s * s * a + 2.0 * s * t * c + t * t * b;
}

vec3 axonTangent(vec3 a, vec3 b, float sag, float t) {
  vec3 c = axonControl(a, b, sag);
  return 2.0 * (1.0 - t) * (c - a) + 2.0 * t * (b - c);
}
`;

/**
 * Instanced ribbon along an axon spline.
 *
 * The base geometry is a flat strip: `position.x` is the parameter along the
 * spline in 0..1 and `position.y` is the side in -1..1. One instance is one
 * axon. The strip is widened in view space along the axis perpendicular to both
 * the tangent and the view direction, so the ribbon always turns its face to the
 * camera and reads as a tube without any tube geometry.
 *
 * Per-instance attributes:
 *   instanceStart    vec3   presynaptic soma position
 *   instanceEnd      vec3   postsynaptic soma position
 *   instanceColor    vec3   receptor or polarity colour, linear space
 *   instanceSag      float  spline sag in world units
 *   instanceWidth    float  per-axon width multiplier
 *   instanceActivity float  0..1 travel envelope from SynapseBuffers.activity
 *   instancePulse    float  per-axon phase offset for the impulse
 */
export const AXON_VERTEX_GLSL = [
  AXON_SPLINE_GLSL,
  /* glsl */ `
attribute vec3 instanceStart;
attribute vec3 instanceEnd;
attribute vec3 instanceColor;
attribute float instanceSag;
attribute float instanceWidth;
attribute float instanceActivity;
attribute float instancePulse;

uniform float uWidth;
uniform float uTaper;
uniform float uMinPixelWidth;
uniform vec2 uResolution;

varying vec3 vViewPosition;
varying vec3 vColor;
varying float vT;
varying float vSide;
varying float vActivity;
varying float vPulse;

void main() {
  float t = clamp(position.x, 0.0, 1.0);
  float side = position.y;

  vec3 world = axonPoint(instanceStart, instanceEnd, instanceSag, t);
  vec3 tangent = axonTangent(instanceStart, instanceEnd, instanceSag, t);

  mat4 modelView = viewMatrix * modelMatrix;
  vec4 viewPosition = modelView * vec4(world, 1.0);
  vec3 viewTangent = normalize(mat3(modelView) * tangent);

  // In view space the camera looks down -z, so the screen-plane normal of the
  // ribbon is the tangent crossed with the view axis. The fallback covers an
  // axon pointing straight at the camera, where that cross product vanishes.
  vec3 axis = cross(viewTangent, vec3(0.0, 0.0, 1.0));
  float axisLength = length(axis);
  vec3 ribbonNormal = axisLength > 1e-4 ? axis / axisLength : vec3(1.0, 0.0, 0.0);

  float width = uWidth * instanceWidth * mix(1.0, uTaper, t) * (0.85 + 0.35 * instanceActivity);

  // A perspective camera sees this much world space per pixel at this depth;
  // holding the ribbon above a minimum pixel width keeps distant axons visible
  // instead of dissolving into aliasing.
  float worldPerPixel = 2.0 * abs(viewPosition.z)
    / max(uResolution.y * projectionMatrix[1][1], 1e-4);
  width = max(width, uMinPixelWidth * worldPerPixel);

  viewPosition.xyz += ribbonNormal * side * width * 0.5;

  vViewPosition = -viewPosition.xyz;
  vColor = instanceColor;
  vT = t;
  vSide = side;
  vActivity = clamp(instanceActivity, 0.0, 1.0);
  vPulse = instancePulse;

  gl_Position = projectionMatrix * viewPosition;
}
`,
].join('\n');

const AXON_FRAGMENT_BODY = /* glsl */ `
uniform vec3 uPulseColor;
uniform vec3 uFogColor;
uniform float uPulse;
uniform float uPulseWidth;
uniform float uPulseTrail;
uniform float uPulseIntensity;
uniform float uBaseIntensity;
uniform float uActivityIntensity;
uniform float uCoreSoftness;
uniform float uOpacity;
uniform float uFogDensity;

varying vec3 vViewPosition;
varying vec3 vColor;
varying float vT;
varying float vSide;
varying float vActivity;
varying float vPulse;

void main() {
  // |side| runs 0 at the ribbon centre to 1 at its edge; a gaussian across it
  // gives the flat strip the falloff of a round tube.
  float edge = abs(vSide);
  float core = exp(-edge * edge * uCoreSoftness);

  // The head position is a uniform so every axon advances on the same clock;
  // the per-instance phase keeps them from firing in lockstep.
  float head = fract(uPulse + vPulse);
  float offset = vT - head;
  float width = max(uPulseWidth, 1e-4);
  float glow = exp(-(offset * offset) / (width * width));

  // A comet tail trailing behind the head, i.e. at parameters it has passed.
  float behind = max(-offset, 0.0);
  float tail = exp(-behind / max(uPulseTrail, 1e-4)) * (1.0 - step(0.0, offset));
  float impulse = max(glow, tail * 0.65);

  vec3 color = vColor * (uBaseIntensity + uActivityIntensity * vActivity);
  color += uPulseColor * impulse * vActivity * uPulseIntensity;

  float alpha = uOpacity * core * (0.35 + 0.65 * vActivity + impulse);
  // Fade both ends so the ribbon meets the somata instead of stopping at a cap.
  alpha *= smoothstep(0.0, 0.04, vT) * smoothstep(1.0, 0.94, vT);

  color = applyFog(color, length(vViewPosition), uFogDensity, uFogColor);

  gl_FragColor = vec4(color, clamp(alpha, 0.0, 1.0));
}
`;

/** Self-contained: already includes FOG_GLSL. */
export const AXON_FRAGMENT_GLSL = [FOG_GLSL, AXON_FRAGMENT_BODY].join('\n');

/** Uniform names and types for the axon material. */
export const AXON_UNIFORMS = {
  uWidth: 'float',
  uTaper: 'float',
  uMinPixelWidth: 'float',
  uResolution: 'vec2',
  uPulseColor: 'vec3',
  uFogColor: 'vec3',
  uPulse: 'float',
  uPulseWidth: 'float',
  uPulseTrail: 'float',
  uPulseIntensity: 'float',
  uBaseIntensity: 'float',
  uActivityIntensity: 'float',
  uCoreSoftness: 'float',
  uOpacity: 'float',
  uFogDensity: 'float',
} as const satisfies UniformTable;
