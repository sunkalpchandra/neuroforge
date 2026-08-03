import type { UniformTable } from '../types';
import { FOG_GLSL, VOLTAGE_RAMP_GLSL } from './common';

/**
 * Per-instance attributes the neuron material expects on an
 * `InstancedBufferGeometry`, alongside the `position`/`normal` Three injects:
 *
 *   instanceOffset   vec3   world-space soma centre
 *   instanceScale    float  per-neuron radius multiplier
 *   instanceVoltage  float  membrane potential in mV
 *   instanceFlash    float  spike envelope in 0..1
 *   instancePolarity float  0 excitatory, 1 inhibitory
 *   instanceFlags    float  NEURON_FLAG bits as a small integer
 */
export const NEURON_VERTEX_GLSL = /* glsl */ `
attribute vec3 instanceOffset;
attribute float instanceScale;
attribute float instanceVoltage;
attribute float instanceFlash;
attribute float instancePolarity;
attribute float instanceFlags;

uniform float uNeuronScale;
uniform float uSpikeSwell;

varying vec3 vViewPosition;
varying vec3 vNormalView;
varying float vVoltage;
varying float vFlash;
varying float vPolarity;
varying float vFlags;

void main() {
  float flash = clamp(instanceFlash, 0.0, 1.0);
  float scale = instanceScale * uNeuronScale * (1.0 + uSpikeSwell * flash);

  vec4 worldPosition = modelMatrix * vec4(position * scale + instanceOffset, 1.0);
  vec4 viewPosition = viewMatrix * worldPosition;

  // Instancing only translates and scales uniformly, so the normal matrix built
  // from modelViewMatrix still rotates instance normals correctly.
  vNormalView = normalize(normalMatrix * normal);
  vViewPosition = -viewPosition.xyz;
  vVoltage = instanceVoltage;
  vFlash = flash;
  vPolarity = instancePolarity;
  vFlags = instanceFlags;

  gl_Position = projectionMatrix * viewPosition;
}
`;

const NEURON_FRAGMENT_BODY = /* glsl */ `
uniform vec3 uExcitatoryColor;
uniform vec3 uInhibitoryColor;
uniform vec3 uLightDirection;
uniform vec3 uLightColor;
uniform vec3 uFillColor;
uniform vec3 uAmbientColor;
uniform vec3 uRimColor;
uniform vec3 uSelectionColor;
uniform vec3 uHoverColor;
uniform vec3 uFogColor;
uniform float uRimPower;
uniform float uRimStrength;
uniform float uEmissiveStrength;
uniform float uRoughness;
uniform float uOpacity;
uniform float uGhostOpacity;
uniform float uVoltageColoring;
uniform float uTranslucency;
uniform float uFogDensity;

varying vec3 vViewPosition;
varying vec3 vNormalView;
varying float vVoltage;
varying float vFlash;
varying float vPolarity;
varying float vFlags;

const float PI = 3.141592653589793;

/**
 * Extract one bit from a small non-negative integer carried in a float.
 *
 * The epsilon absorbs the rounding a varying picks up while being interpolated
 * across a triangle whose three vertices all carry the same value; without it an
 * exact 3.0 can arrive as 2.9999998 and lose its low bit.
 */
float flagBit(float flags, float bitValue) {
  return mod(floor(flags / bitValue + 0.001), 2.0);
}

void main() {
  vec3 N = normalize(vNormalView);
  vec3 V = normalize(vViewPosition);
  vec3 L = normalize(uLightDirection);
  float ndv = clamp(dot(N, V), 0.0, 1.0);
  float ndl = dot(N, L);

  vec3 polarityColor = mix(uExcitatoryColor, uInhibitoryColor, clamp(vPolarity, 0.0, 1.0));
  vec3 baseColor = mix(polarityColor, voltageRamp(vVoltage), clamp(uVoltageColoring, 0.0, 1.0));

  // Wrapped diffuse stands in for light scattering through the membrane: the
  // terminator softens and the shadowed side keeps a little transmitted colour.
  float translucency = clamp(uTranslucency, 0.0, 1.0);
  float wrapped = clamp((ndl + translucency) / (1.0 + translucency), 0.0, 1.0);
  float diffuse = mix(max(ndl, 0.0), wrapped, translucency);

  // GGX specular with a Schlick fresnel at a dielectric F0.
  vec3 H = normalize(L + V);
  float ndh = max(dot(N, H), 0.0);
  float vdh = clamp(dot(V, H), 0.0, 1.0);
  float ndlSat = max(ndl, 0.0);
  float roughness = clamp(uRoughness, 0.045, 1.0);
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;
  float d = ndh * ndh * (alpha2 - 1.0) + 1.0;
  // The floor here only exists to rule out 0/0; it must not bind. Because
  // roughness is clamped above, d is at least alpha2 = 4.1e-6 and PI*d*d is at
  // least 5e-11, so 1e-12 never engages. A larger floor would cap the highlight
  // of every surface smoother than roughness 0.274 and invert the response,
  // making a mirror-smooth neuron duller than a matte one.
  float distribution = alpha2 / max(PI * d * d, 1e-12);
  float fresnel = 0.04 + 0.96 * pow(1.0 - vdh, 5.0);

  // Height-correlated Smith visibility. This term carries the 1/(4 ndl ndv)
  // denominator of the microfacet BRDF, so the specular lobe integrates to at
  // most the incident light. Distribution alone peaks at 1/(pi alpha2), which is
  // ~8e4 at the minimum roughness above: without the visibility term and the
  // ndl factor the highlight is not a highlight but a hole punched through the
  // bloom threshold.
  float lambdaV = ndlSat * sqrt(ndv * ndv * (1.0 - alpha2) + alpha2);
  float lambdaL = ndv * sqrt(ndlSat * ndlSat * (1.0 - alpha2) + alpha2);
  float visibility = 0.5 / max(lambdaV + lambdaL, 1e-5);
  vec3 specular = uLightColor * distribution * visibility * fresnel * ndlSat;

  float rim = pow(1.0 - ndv, max(uRimPower, 0.5)) * uRimStrength;

  vec3 color = baseColor * (uAmbientColor
    + uLightColor * diffuse
    + uFillColor * max(-ndl, 0.0) * 0.35);
  color += specular;
  color += uRimColor * rim * (0.35 + 0.65 * vFlash);

  // The envelope is squared so that resting glow stays under the bloom threshold
  // and only an actual spike blooms.
  color += baseColor * uEmissiveStrength * vFlash * vFlash;

  float selected = flagBit(vFlags, 1.0);
  float hovered = flagBit(vFlags, 2.0);
  float probed = flagBit(vFlags, 4.0);
  float ghosted = flagBit(vFlags, 8.0);
  color += uSelectionColor * selected * (0.25 + 0.75 * rim);
  color += uHoverColor * hovered * rim * 0.6;
  color += uSelectionColor * probed * 0.08;

  float depth = length(vViewPosition);
  color = applyFog(color, depth, uFogDensity, uFogColor);

  // Grazing angles and spikes thicken the silhouette, which is what sells the
  // membrane as a shell with something inside it rather than a solid ball.
  float opacity = uOpacity * mix(1.0, clamp(uGhostOpacity, 0.0, 1.0), ghosted);
  opacity = clamp(opacity + rim * 0.35 + vFlash * 0.4, 0.0, 1.0);

  gl_FragColor = vec4(color, opacity);
}
`;

/** Self-contained: already includes VOLTAGE_RAMP_GLSL and FOG_GLSL. */
export const NEURON_FRAGMENT_GLSL = [
  VOLTAGE_RAMP_GLSL,
  FOG_GLSL,
  NEURON_FRAGMENT_BODY,
].join('\n');

/** Uniform names and types for the neuron material. `uLightDirection` is in view space. */
export const NEURON_UNIFORMS = {
  uNeuronScale: 'float',
  uSpikeSwell: 'float',
  uExcitatoryColor: 'vec3',
  uInhibitoryColor: 'vec3',
  uLightDirection: 'vec3',
  uLightColor: 'vec3',
  uFillColor: 'vec3',
  uAmbientColor: 'vec3',
  uRimColor: 'vec3',
  uSelectionColor: 'vec3',
  uHoverColor: 'vec3',
  uFogColor: 'vec3',
  uRimPower: 'float',
  uRimStrength: 'float',
  uEmissiveStrength: 'float',
  uRoughness: 'float',
  uOpacity: 'float',
  uGhostOpacity: 'float',
  uVoltageColoring: 'float',
  uTranslucency: 'float',
  uFogDensity: 'float',
} as const satisfies UniformTable;
