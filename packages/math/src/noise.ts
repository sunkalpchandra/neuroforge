import { clamp } from './internal';
import { Rng } from './rng';

/**
 * 3D simplex noise (Perlin's simplectic grid, Gustavson's formulation) plus the
 * fbm and curl fields built on it.
 *
 * Simplex rather than classic Perlin because the axis-aligned artefacts of a
 * cubic lattice are visible in dendrite jitter and in the drifting particle
 * field, and because the 3D case costs 4 gradient evaluations instead of 8.
 */

/** Skew/unskew factors for 3 dimensions: (sqrt(4)-1)/3 and (1 - 1/sqrt(4))/3. */
const F3 = 1 / 3;
const G3 = 1 / 6;

/**
 * The 12 midpoints of the edges of a cube. Using a gradient set of uniform
 * length keeps the noise isotropic; 12 is the standard 3D table.
 */
const GRAD3 = new Int8Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0, 1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1, 0, 1, 1, 0, -1,
  1, 0, 1, -1, 0, -1, -1,
]);

/**
 * Empirical scale that maps the raw kernel sum into roughly [-1,1]. The bound is
 * not exactly 1, so the result is clamped: callers are promised [-1,1].
 */
const SIMPLEX_SCALE = 32;

const PERM = new Uint8Array(512);
const PERM_MOD_12 = new Uint8Array(512);

{
  // A fixed shuffle rather than a literal table: it is a valid permutation by
  // construction and, being seeded, is identical on every run and every build.
  const source: number[] = new Array(256);
  for (let i = 0; i < 256; i += 1) source[i] = i;
  new Rng(0x1a2b3c4d).shuffle(source);
  for (let i = 0; i < 512; i += 1) {
    const value = source[i & 255];
    PERM[i] = value;
    PERM_MOD_12[i] = value % 12;
  }
}

function dotGrad(gi: number, x: number, y: number, z: number): number {
  return GRAD3[gi] * x + GRAD3[gi + 1] * y + GRAD3[gi + 2] * z;
}

function contribution(gi: number, x: number, y: number, z: number): number {
  let t = 0.6 - x * x - y * y - z * z;
  if (t <= 0) return 0;
  t *= t;
  return t * t * dotGrad(gi, x, y, z);
}

export function simplex3(x: number, y: number, z: number): number {
  // Skew the input onto the simplectic lattice and find the containing cell.
  const s = (x + y + z) * F3;
  const i = Math.floor(x + s);
  const j = Math.floor(y + s);
  const k = Math.floor(z + s);
  const t = (i + j + k) * G3;
  const x0 = x - (i - t);
  const y0 = y - (j - t);
  const z0 = z - (k - t);

  // Rank the coordinates to pick which of the six tetrahedra we are in.
  let i1: number;
  let j1: number;
  let k1: number;
  let i2: number;
  let j2: number;
  let k2: number;
  if (x0 >= y0) {
    if (y0 >= z0) {
      i1 = 1;
      j1 = 0;
      k1 = 0;
      i2 = 1;
      j2 = 1;
      k2 = 0;
    } else if (x0 >= z0) {
      i1 = 1;
      j1 = 0;
      k1 = 0;
      i2 = 1;
      j2 = 0;
      k2 = 1;
    } else {
      i1 = 0;
      j1 = 0;
      k1 = 1;
      i2 = 1;
      j2 = 0;
      k2 = 1;
    }
  } else if (y0 < z0) {
    i1 = 0;
    j1 = 0;
    k1 = 1;
    i2 = 0;
    j2 = 1;
    k2 = 1;
  } else if (x0 < z0) {
    i1 = 0;
    j1 = 1;
    k1 = 0;
    i2 = 0;
    j2 = 1;
    k2 = 1;
  } else {
    i1 = 0;
    j1 = 1;
    k1 = 0;
    i2 = 1;
    j2 = 1;
    k2 = 0;
  }

  const x1 = x0 - i1 + G3;
  const y1 = y0 - j1 + G3;
  const z1 = z0 - k1 + G3;
  const x2 = x0 - i2 + 2 * G3;
  const y2 = y0 - j2 + 2 * G3;
  const z2 = z0 - k2 + 2 * G3;
  const x3 = x0 - 1 + 3 * G3;
  const y3 = y0 - 1 + 3 * G3;
  const z3 = z0 - 1 + 3 * G3;

  const ii = i & 255;
  const jj = j & 255;
  const kk = k & 255;
  const gi0 = PERM_MOD_12[ii + PERM[jj + PERM[kk]]] * 3;
  const gi1 = PERM_MOD_12[ii + i1 + PERM[jj + j1 + PERM[kk + k1]]] * 3;
  const gi2 = PERM_MOD_12[ii + i2 + PERM[jj + j2 + PERM[kk + k2]]] * 3;
  const gi3 = PERM_MOD_12[ii + 1 + PERM[jj + 1 + PERM[kk + 1]]] * 3;

  const n =
    contribution(gi0, x0, y0, z0) +
    contribution(gi1, x1, y1, z1) +
    contribution(gi2, x2, y2, z2) +
    contribution(gi3, x3, y3, z3);

  return clamp(SIMPLEX_SCALE * n, -1, 1);
}

const MAX_OCTAVES = 12;

/** Fractal sum of simplex octaves, normalised back into [-1,1]. */
export function fbm3(
  x: number,
  y: number,
  z: number,
  octaves = 5,
  lacunarity = 2,
  gain = 0.5,
): number {
  const count = Math.min(MAX_OCTAVES, Math.max(1, Math.floor(octaves)));
  let amplitude = 1;
  let frequency = 1;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < count; o += 1) {
    sum += amplitude * simplex3(x * frequency, y * frequency, z * frequency);
    norm += Math.abs(amplitude);
    amplitude *= gain;
    frequency *= lacunarity;
  }
  return norm > 0 ? clamp(sum / norm, -1, 1) : 0;
}

/**
 * Offsets that decorrelate the three components of the vector potential. Large
 * and irrational-looking so the three fbm fields never sample the same lattice
 * cells in step with each other.
 */
const POTENTIAL_OFFSET = new Float64Array([
  0, 0, 0, 31.416, -47.853, 12.793, -233.14, 17.71, -91.31,
]);

function potential(axis: number, x: number, y: number, z: number): number {
  const o = axis * 3;
  return fbm3(x + POTENTIAL_OFFSET[o], y + POTENTIAL_OFFSET[o + 1], z + POTENTIAL_OFFSET[o + 2]);
}

const DEFAULT_CURL_EPSILON = 1e-3;

/**
 * Curl of the fbm vector potential, evaluated by central differences.
 *
 * The result is divergence-free by construction, which is what makes it usable
 * as an advection field: particles neither bunch up nor thin out.
 */
export function curlNoise3(
  x: number,
  y: number,
  z: number,
  epsilon = DEFAULT_CURL_EPSILON,
): { x: number; y: number; z: number } {
  const h = epsilon > 0 ? epsilon : DEFAULT_CURL_EPSILON;
  const inv = 1 / (2 * h);

  const dp2dy = (potential(2, x, y + h, z) - potential(2, x, y - h, z)) * inv;
  const dp1dz = (potential(1, x, y, z + h) - potential(1, x, y, z - h)) * inv;
  const dp0dz = (potential(0, x, y, z + h) - potential(0, x, y, z - h)) * inv;
  const dp2dx = (potential(2, x + h, y, z) - potential(2, x - h, y, z)) * inv;
  const dp1dx = (potential(1, x + h, y, z) - potential(1, x - h, y, z)) * inv;
  const dp0dy = (potential(0, x, y + h, z) - potential(0, x, y - h, z)) * inv;

  return {
    x: dp2dy - dp1dz,
    y: dp0dz - dp2dx,
    z: dp1dx - dp0dy,
  };
}
