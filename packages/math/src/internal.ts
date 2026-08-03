/**
 * Numeric primitives shared by more than one module in this package.
 *
 * Nothing here is part of the public contract; `index.ts` does not re-export it.
 */

export const TAU = Math.PI * 2;

export function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

export function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

/**
 * SplitMix32 finaliser. A full-avalanche 32-bit mixer: every input bit affects
 * every output bit, which is what makes it safe to derive independent generator
 * states from a single counter.
 */
export function mix32(x: number): number {
  let h = x | 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x21f0aaad);
  h ^= h >>> 15;
  h = Math.imul(h, 0x735a2d97);
  h ^= h >>> 15;
  return h >>> 0;
}

const BIT_BUFFER = new ArrayBuffer(8);
const BIT_F64 = new Float64Array(BIT_BUFFER);
const BIT_U32 = new Uint32Array(BIT_BUFFER);

/**
 * Fold an arbitrary number into a 32-bit word. Integers inside the 32-bit range
 * map to themselves so that `new Rng(7)` is the generator a reader expects;
 * anything else is folded from its IEEE-754 bit pattern, which keeps fractional
 * and out-of-range seeds distinguishable instead of collapsing them to 0.
 */
export function floatToWord(value: number): number {
  if (Number.isInteger(value) && value >= -2147483648 && value <= 4294967295) {
    return value >>> 0;
  }
  BIT_F64[0] = value;
  return (BIT_U32[0] ^ BIT_U32[1]) >>> 0;
}

/** Smallest power of two >= n, for n in [0, 2^31]. */
export function nextPowerOfTwo(n: number): number {
  if (!(n > 1)) return 1;
  let v = Math.ceil(n) - 1;
  v |= v >>> 1;
  v |= v >>> 2;
  v |= v >>> 4;
  v |= v >>> 8;
  v |= v >>> 16;
  return v + 1;
}

export function isPowerOfTwo(n: number): boolean {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

/** Lanczos g=7, n=9 coefficients. */
const LANCZOS = [
  0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531,
  -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019572e-6,
  1.5056327351493116e-7,
];

const LOG_SQRT_TAU = 0.5 * Math.log(TAU);

/**
 * Natural log of the gamma function. Used for log(k!) in the Poisson rejection
 * sampler, where k can be in the thousands and a direct factorial overflows.
 */
export function logGamma(x: number): number {
  if (x < 0.5) {
    // Reflection formula: Gamma(x)Gamma(1-x) = pi / sin(pi x).
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
  }
  const z = x - 1;
  const t = z + 7.5;
  let a = LANCZOS[0];
  for (let i = 1; i < 9; i += 1) a += LANCZOS[i] / (z + i);
  return LOG_SQRT_TAU + (z + 0.5) * Math.log(t) - t + Math.log(a);
}
