/**
 * Shared GLSL chunks.
 *
 * Every chunk is wrapped in an include guard, so concatenating them in any order
 * and any number of times is safe. The program strings exported by the other
 * modules in this directory already prepend the chunks they need; the guards are
 * what makes it harmless for a caller to prepend them again.
 *
 * These are written for `THREE.ShaderMaterial` at its default GLSL version:
 * no `#version`, no precision header, and no redeclaration of the attributes,
 * varyings-turned-in/out or matrices Three injects. Three compiles that prefix
 * as `#version 300 es` with `attribute`/`varying`/`gl_FragColor` aliased onto
 * the ES 3.00 spellings, so `fwidth`, `gl_FragDepth` and integer arithmetic are
 * all available even though the source reads like ES 1.00.
 */

/**
 * `vec3 voltageRamp(float v)` — membrane potential in mV to a linear-space RGB.
 *
 * This is VOLTAGE_RAMP from `@neuroforge/shared/theme` evaluated as a chain of
 * mixes. Each mix is driven by a clamped linear ramp between adjacent stops, so
 * the term for a stop the voltage has already passed saturates at 1 and simply
 * replaces the accumulated colour: the chain is exactly the piecewise-linear
 * interpolation of the table, with no branches and no lookup texture.
 *
 * The stops are interpolated in sRGB, which is the space they were chosen in,
 * and converted to linear at the end because the renderer works in linear light.
 */
export const VOLTAGE_RAMP_GLSL = /* glsl */ `
#ifndef NF_VOLTAGE_RAMP_INCLUDED
#define NF_VOLTAGE_RAMP_INCLUDED

vec3 nfSrgbToLinear(vec3 c) {
  vec3 low = c / 12.92;
  vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));
  return mix(low, high, step(vec3(0.04045), c));
}

float nfRampStep(float v, float lo, float hi) {
  return clamp((v - lo) / (hi - lo), 0.0, 1.0);
}

vec3 voltageRamp(float v) {
  vec3 c = vec3(0.105882, 0.164706, 0.290196);                                  // -90 mV #1B2A4A
  c = mix(c, vec3(0.180392, 0.290196, 0.490196), nfRampStep(v, -90.0, -70.0));  // -70 mV #2E4A7D
  c = mix(c, vec3(0.309804, 0.819608, 1.000000), nfRampStep(v, -70.0, -55.0));  // -55 mV #4FD1FF
  c = mix(c, vec3(0.541176, 0.878431, 1.000000), nfRampStep(v, -55.0, -40.0));  // -40 mV #8AE0FF
  c = mix(c, vec3(0.713725, 0.419608, 1.000000), nfRampStep(v, -40.0, -20.0));  // -20 mV #B66BFF
  c = mix(c, vec3(0.949020, 0.831373, 1.000000), nfRampStep(v, -20.0, 0.0));    //   0 mV #F2D4FF
  c = mix(c, vec3(1.000000, 1.000000, 1.000000), nfRampStep(v, 0.0, 30.0));     //  30 mV #FFFFFF
  return nfSrgbToLinear(c);
}

#endif
`;

/**
 * `vec3 applyFog(vec3 color, float depth, float density, vec3 fogColor)`.
 *
 * Exponential-squared fog, matching `RenderSettings.fogDensity`. The squared
 * form keeps near geometry completely clear and then falls off quickly, which
 * reads as atmospheric depth rather than as a uniform wash.
 */
export const FOG_GLSL = /* glsl */ `
#ifndef NF_FOG_INCLUDED
#define NF_FOG_INCLUDED

float fogFactor(float depth, float density) {
  float d = depth * density;
  return 1.0 - exp2(-d * d * 1.442695);
}

vec3 applyFog(vec3 color, float depth, float density, vec3 fogColor) {
  return mix(color, fogColor, clamp(fogFactor(depth, density), 0.0, 1.0));
}

#endif
`;

/**
 * Hash and simplex helpers.
 *
 * `hash11`/`hash13`/`hash33` are the standard low-bias fract-multiply hashes;
 * `snoise` is the gradient-lookup-free simplex construction, which is what makes
 * it cheap enough to call per fragment.
 */
export const NOISE_GLSL = /* glsl */ `
#ifndef NF_NOISE_INCLUDED
#define NF_NOISE_INCLUDED

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float hash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 hash33(vec3 p3) {
  p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}

vec3 nfMod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 nfMod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 nfPermute(vec4 x) { return nfMod289(((x * 34.0) + 1.0) * x); }
vec4 nfTaylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = nfMod289(i);
  vec4 p = nfPermute(nfPermute(nfPermute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0))
    + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = nfTaylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

float fbm3(vec3 p, int octaves, float lacunarity, float gain) {
  float amplitude = 0.5;
  float total = 0.0;
  float normalisation = 0.0;
  vec3 q = p;
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    total += snoise(q) * amplitude;
    normalisation += amplitude;
    q *= lacunarity;
    amplitude *= gain;
  }
  return total / max(normalisation, 1e-5);
}

#endif
`;
