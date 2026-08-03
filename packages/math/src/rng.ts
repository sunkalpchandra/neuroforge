import { TAU, floatToWord, logGamma, mix32 } from './internal';

/**
 * Deterministic pseudo-random generator.
 *
 * The core is xoshiro128**, a four-word 32-bit generator with a period of
 * 2^128-1 that passes BigCrush. It is used instead of an LCG because the whole
 * product depends on reproducibility: the same seed must reproduce the same
 * network, the same noise currents and the same spike train, and low-order bit
 * correlations in a naive LCG show up as visible structure in procedural
 * morphology and as spurious synchrony in Poisson stimuli.
 *
 * State is seeded through SplitMix32 so that consecutive integer seeds produce
 * decorrelated streams. There is no global state; every stochastic process owns
 * its own instance or a `fork()` of one.
 */

const TWO_POW_32 = 4294967296;
const UINT32_SCALE = 1 / TWO_POW_32;
const GOLDEN_GAMMA = 0x9e3779b9 | 0;

/** Above this mean, Knuth's product method needs O(lambda) draws; PTRS is O(1). */
const POISSON_PTRS_THRESHOLD = 30;

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

export class Rng {
  #s0 = 0;
  #s1 = 0;
  #s2 = 0;
  #s3 = 0;
  #seed: number;
  /** Second Box-Muller variate, kept so each normal() costs half a transform. */
  #spare = 0;
  #hasSpare = false;

  constructor(seed: number) {
    this.#seed = floatToWord(seed);
    this.#seedState(this.#seed);
  }

  get seed(): number {
    return this.#seed;
  }

  /** Uniform in [0,1) with 32 bits of resolution. */
  next(): number {
    return this.#nextUint32() * UINT32_SCALE;
  }

  /** Uniform integer in [0, maxExclusive), rejection-sampled so it is unbiased. */
  int(maxExclusive: number): number {
    const m = Math.floor(maxExclusive);
    if (!(m > 0)) return 0;
    if (m > TWO_POW_32) {
      // Beyond the generator's word size; falls back to float resolution.
      return Math.floor(this.next() * m);
    }
    const limit = TWO_POW_32 - (TWO_POW_32 % m);
    let r = this.#nextUint32();
    while (r >= limit) r = this.#nextUint32();
    return r % m;
  }

  range(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  /** Box-Muller normal deviate; the discarded sine term is cached for the next call. */
  normal(mean = 0, stdDev = 1): number {
    if (this.#hasSpare) {
      this.#hasSpare = false;
      return mean + stdDev * this.#spare;
    }
    const radius = Math.sqrt(-2 * Math.log(this.#nextOpenUnit()));
    const theta = TAU * this.next();
    this.#spare = radius * Math.sin(theta);
    this.#hasSpare = true;
    return mean + stdDev * radius * Math.cos(theta);
  }

  /** Exponential with the given rate; mean is 1/rate. */
  exponential(rate: number): number {
    if (!(rate > 0)) return Infinity;
    return -Math.log(this.#nextOpenUnit()) / rate;
  }

  poisson(lambda: number): number {
    if (!(lambda > 0)) return 0;
    if (lambda < POISSON_PTRS_THRESHOLD) {
      // Knuth: multiply uniforms until the product drops below exp(-lambda).
      const limit = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k += 1;
        p *= this.next();
      } while (p > limit);
      return k - 1;
    }
    return this.#poissonPtrs(lambda);
  }

  bool(probability: number): boolean {
    return this.next() < probability;
  }

  pick<T>(items: readonly T[]): T {
    if (items.length === 0) {
      throw new RangeError('Rng.pick requires a non-empty array');
    }
    return items[this.int(items.length)];
  }

  /** Fisher-Yates, in place. */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i -= 1) {
      const j = this.int(i + 1);
      const swap = items[i];
      items[i] = items[j];
      items[j] = swap;
    }
    return items;
  }

  /** Uniform on the unit sphere (Archimedes' equal-area cylinder projection). */
  onSphere(): { x: number; y: number; z: number } {
    const z = this.next() * 2 - 1;
    const theta = TAU * this.next();
    const r = Math.sqrt(Math.max(0, 1 - z * z));
    return { x: r * Math.cos(theta), y: r * Math.sin(theta), z };
  }

  /** Uniform inside the unit ball; the cube root keeps the density constant. */
  inSphere(): { x: number; y: number; z: number } {
    const dir = this.onSphere();
    const r = Math.cbrt(this.next());
    dir.x *= r;
    dir.y *= r;
    dir.z *= r;
    return dir;
  }

  /**
   * A generator whose stream is independent of this one. Derived purely from the
   * current state and the salt, so forking is reproducible and does not disturb
   * the parent sequence: the same parent state and salt always yield the same
   * child.
   */
  fork(salt: number): Rng {
    const s = floatToWord(salt);
    let acc = mix32(this.#s0 ^ Math.imul(s, 0x9e3779b1));
    acc = mix32(acc ^ this.#s1 ^ 0x85ebca6b);
    acc = mix32(acc ^ this.#s2 ^ Math.imul(s, 0xc2b2ae35));
    acc = mix32(acc ^ this.#s3 ^ this.#seed);
    return new Rng(acc);
  }

  /** Restart the stream, optionally under a new seed. */
  reset(seed?: number): void {
    if (seed !== undefined) this.#seed = floatToWord(seed);
    this.#seedState(this.#seed);
  }

  #seedState(seed: number): void {
    let s = seed | 0;
    s = (s + GOLDEN_GAMMA) | 0;
    this.#s0 = mix32(s);
    s = (s + GOLDEN_GAMMA) | 0;
    this.#s1 = mix32(s);
    s = (s + GOLDEN_GAMMA) | 0;
    this.#s2 = mix32(s);
    s = (s + GOLDEN_GAMMA) | 0;
    this.#s3 = mix32(s);
    // The all-zero state is a fixed point of the recurrence.
    if ((this.#s0 | this.#s1 | this.#s2 | this.#s3) === 0) this.#s3 = 0x9e3779b9;
    this.#hasSpare = false;
  }

  #nextUint32(): number {
    const s0 = this.#s0;
    const s1 = this.#s1;
    const result = Math.imul(rotl(Math.imul(s1, 5), 7), 9) >>> 0;
    const t = (s1 << 9) | 0;
    const s2 = (this.#s2 ^ s0) | 0;
    const s3 = (this.#s3 ^ s1) | 0;
    this.#s1 = (s1 ^ s2) >>> 0;
    this.#s0 = (s0 ^ s3) >>> 0;
    this.#s2 = (s2 ^ t) >>> 0;
    this.#s3 = rotl(s3, 11);
    return result;
  }

  /** Uniform in (0,1); used where a zero would produce log(0). */
  #nextOpenUnit(): number {
    return (this.#nextUint32() + 0.5) * UINT32_SCALE;
  }

  /**
   * Hoermann's PTRS (transformed rejection with squeeze). Constant time in
   * lambda, which matters because Poisson stimuli are sampled per neuron per
   * step at rates that can reach hundreds of events.
   */
  #poissonPtrs(lambda: number): number {
    const b = 0.931 + 2.53 * Math.sqrt(lambda);
    const a = -0.059 + 0.02483 * b;
    const invAlpha = 1.1239 + 1.1328 / (b - 3.4);
    const vr = 0.9277 - 3.6224 / (b - 2);
    const logLambda = Math.log(lambda);
    for (;;) {
      const u = this.next() - 0.5;
      const v = this.next();
      const us = 0.5 - Math.abs(u);
      const k = Math.floor(((2 * a) / us + b) * u + lambda + 0.43);
      if (us >= 0.07 && v <= vr) return k;
      if (!(k >= 0)) continue;
      if (us < 0.013 && v > us) continue;
      const accept = Math.log((v * invAlpha) / (a / (us * us) + b));
      if (accept <= k * logLambda - lambda - logGamma(k + 1)) return k;
    }
  }
}

/**
 * Combine any number of values into a 32-bit seed. Order matters, so
 * `hashSeed(circuitSeed, populationIndex, neuronIndex)` gives every neuron a
 * distinct, reproducible stream.
 */
export function hashSeed(...values: number[]): number {
  let h = 0x811c9dc5 | 0;
  for (let i = 0; i < values.length; i += 1) {
    h = (Math.imul(h ^ floatToWord(values[i]), 0x9e3779b1) + i) | 0;
    h = mix32(h) | 0;
  }
  return mix32(h ^ values.length);
}
