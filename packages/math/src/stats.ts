import { TAU, isPowerOfTwo, nextPowerOfTwo } from './internal';

/**
 * Spike-train statistics and the spectral tools the inspector and the AI builder
 * use to decide whether a circuit is actually oscillating in the band it was
 * asked for.
 */

/**
 * Neumaier compensated summation. Traces are tens of thousands of samples long
 * and are stored as float32, so the naive running sum loses real precision here.
 */
function sumRange(values: ArrayLike<number>, n: number): number {
  let sum = 0;
  let compensation = 0;
  for (let i = 0; i < n; i += 1) {
    const value = values[i];
    const t = sum + value;
    compensation += Math.abs(sum) >= Math.abs(value) ? sum - t + value : value - t + sum;
    sum = t;
  }
  return sum + compensation;
}

function effectiveCount(values: ArrayLike<number>, count?: number): number {
  const n = count === undefined ? values.length : Math.floor(count);
  if (!(n > 0)) return 0;
  return n < values.length ? n : values.length;
}

export function mean(values: ArrayLike<number>, count?: number): number {
  const n = effectiveCount(values, count);
  if (n === 0) return 0;
  return sumRange(values, n) / n;
}

/**
 * Sample standard deviation (Bessel corrected). Everything measured here is a
 * finite sample of an ongoing process, not a whole population, so n-1 is the
 * right divisor; it also keeps `coefficientOfVariation` comparable with the
 * values reported in the literature.
 */
export function stdDev(values: ArrayLike<number>, count?: number): number {
  return Math.sqrt(variance(values, count));
}

function variance(values: ArrayLike<number>, count?: number): number {
  const n = effectiveCount(values, count);
  if (n < 2) return 0;
  const mu = sumRange(values, n) / n;
  let sum = 0;
  let compensation = 0;
  for (let i = 0; i < n; i += 1) {
    const d = values[i] - mu;
    const term = d * d;
    const t = sum + term;
    compensation += Math.abs(sum) >= term ? sum - t + term : term - t + sum;
    sum = t;
  }
  return (sum + compensation) / (n - 1);
}

/**
 * Consecutive differences of a spike-time list. Returns the number written,
 * which is `count - 1` unless `out` runs out of room.
 */
export function interSpikeIntervals(
  times: ArrayLike<number>,
  count: number,
  out: Float32Array,
): number {
  const n = Math.min(Math.floor(count), times.length);
  if (n < 2) return 0;
  const limit = Math.min(n - 1, out.length);
  for (let i = 0; i < limit; i += 1) {
    out[i] = times[i + 1] - times[i];
  }
  return limit;
}

/**
 * CV of the inter-spike intervals: 0 for a metronome, 1 for a Poisson process,
 * above 1 for bursting.
 */
export function coefficientOfVariation(intervals: ArrayLike<number>, count: number): number {
  const n = effectiveCount(intervals, count);
  if (n < 2) return 0;
  const mu = sumRange(intervals, n) / n;
  if (!(mu > 0)) return 0;
  return Math.sqrt(variance(intervals, n)) / mu;
}

/** Variance-to-mean ratio of spike counts; 1 for a Poisson process. */
export function fanoFactor(counts: ArrayLike<number>, count: number): number {
  const n = effectiveCount(counts, count);
  if (n < 2) return 0;
  const mu = sumRange(counts, n) / n;
  if (!(mu > 0)) return 0;
  return variance(counts, n) / mu;
}

interface Twiddles {
  cos: Float64Array;
  sin: Float64Array;
}

const twiddleCache = new Map<number, Twiddles>();
const MAX_CACHED_TWIDDLES = 6;

/**
 * Twiddle factors are tabulated per transform size rather than advanced by
 * repeated complex multiplication, which would accumulate rotation error across
 * the outer stages of a long transform.
 */
function twiddlesFor(n: number): Twiddles {
  const cached = twiddleCache.get(n);
  if (cached !== undefined) return cached;
  const half = n >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let i = 0; i < half; i += 1) {
    const angle = (-TAU * i) / n;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }
  const table: Twiddles = { cos, sin };
  if (twiddleCache.size >= MAX_CACHED_TWIDDLES) {
    const oldest = twiddleCache.keys().next().value;
    if (oldest !== undefined) twiddleCache.delete(oldest);
  }
  twiddleCache.set(n, table);
  return table;
}

/** In-place radix-2 decimation-in-time FFT. Both arrays must be the same power-of-two length. */
export function fft(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if (imag.length !== n) {
    throw new RangeError(`fft needs matching arrays, got ${n} and ${imag.length}`);
  }
  if (!isPowerOfTwo(n)) {
    throw new RangeError(`fft needs a power-of-two length, got ${n}`);
  }
  if (n === 1) return;

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; (j & bit) !== 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      const ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
  }

  const { cos, sin } = twiddlesFor(n);
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const stride = n / len;
    for (let base = 0; base < n; base += len) {
      for (let k = 0; k < half; k += 1) {
        const w = k * stride;
        const wr = cos[w];
        const wi = sin[w];
        const a = base + k;
        const b = a + half;
        const xr = real[b];
        const xi = imag[b];
        const tr = xr * wr - xi * wi;
        const ti = xr * wi + xi * wr;
        real[b] = real[a] - tr;
        imag[b] = imag[a] - ti;
        real[a] += tr;
        imag[a] += ti;
      }
    }
  }
}

let fftReal = new Float32Array(0);
let fftImag = new Float32Array(0);

/**
 * Single-sided power spectrum of a real signal, Hann windowed and zero padded up
 * to the next power of two. Returns the bin width in Hz.
 *
 * Normalisation is amplitude correct: a pure tone of amplitude A peaks at
 * A^2 / 2 regardless of record length, window or padding, so thresholds set in
 * the UI stay meaningful when the trace length changes.
 */
export function powerSpectrum(
  signal: Float32Array,
  sampleRateHz: number,
  out: Float32Array,
): number {
  const m = signal.length;
  if (m === 0 || !(sampleRateHz > 0)) {
    out.fill(0);
    return 0;
  }

  const n = nextPowerOfTwo(Math.max(2, m));
  if (fftReal.length !== n) {
    fftReal = new Float32Array(n);
    fftImag = new Float32Array(n);
  }
  const re = fftReal;
  const im = fftImag;
  re.fill(0);
  im.fill(0);

  const denom = m > 1 ? m - 1 : 1;
  let windowSum = 0;
  for (let i = 0; i < m; i += 1) {
    const w = 0.5 - 0.5 * Math.cos((TAU * i) / denom);
    re[i] = signal[i] * w;
    windowSum += w;
  }

  fft(re, im);

  const nyquist = n >> 1;
  const bins = nyquist + 1;
  const scale = windowSum > 0 ? 1 / (windowSum * windowSum) : 0;
  const limit = Math.min(bins, out.length);
  for (let k = 0; k < limit; k += 1) {
    const power = re[k] * re[k] + im[k] * im[k];
    // Bin 0 and the Nyquist bin have no mirror image to fold in.
    out[k] = k === 0 || k === nyquist ? power * scale : 2 * power * scale;
  }
  if (out.length > limit) out.fill(0, limit);

  return sampleRateHz / n;
}

/** Log floor for the peak interpolation; keeps empty bins from producing -Infinity. */
const LOG_FLOOR = 1e-30;

/**
 * Peak frequency inside [minHz, maxHz], refined by fitting a parabola through
 * the log magnitude of the peak bin and its two neighbours.
 *
 * The refinement matters: at a 4 Hz bin width, reporting bin centres would make
 * a 38 Hz rhythm and a 42 Hz rhythm indistinguishable, and the builder verifies
 * band placement from this number.
 */
export function dominantFrequency(
  spectrum: Float32Array,
  binHz: number,
  minHz: number,
  maxHz: number,
): number {
  const bins = spectrum.length;
  if (bins === 0 || !(binHz > 0)) return 0;
  // Bin 0 is DC; a "dominant frequency" of zero is never a useful answer.
  const first = Math.max(1, Math.ceil(minHz / binHz));
  const last = Math.min(bins - 1, Math.floor(maxHz / binHz));
  if (first > last) return 0;

  let peak = first;
  let peakValue = spectrum[first];
  for (let k = first + 1; k <= last; k += 1) {
    if (spectrum[k] > peakValue) {
      peakValue = spectrum[k];
      peak = k;
    }
  }

  let offset = 0;
  if (peak > 0 && peak + 1 < bins) {
    const y0 = Math.log(Math.max(spectrum[peak - 1], LOG_FLOOR));
    const y1 = Math.log(Math.max(spectrum[peak], LOG_FLOOR));
    const y2 = Math.log(Math.max(spectrum[peak + 1], LOG_FLOOR));
    const curvature = y0 - 2 * y1 + y2;
    if (curvature < 0) {
      offset = (0.5 * (y0 - y2)) / curvature;
      if (offset > 0.5) offset = 0.5;
      else if (offset < -0.5) offset = -0.5;
    }
  }
  return (peak + offset) * binHz;
}

/**
 * Normalised cross-correlation for lags in [-maxLag, maxLag], written to
 * `out[lag + maxLag]`. Correlating a signal with itself gives exactly 1 at zero
 * lag, so the values are directly comparable between neuron pairs.
 */
export function crossCorrelation(
  a: Float32Array,
  b: Float32Array,
  maxLag: number,
  out: Float32Array,
): void {
  const n = Math.min(a.length, b.length);
  const lags = Math.max(0, Math.floor(maxLag));
  const total = 2 * lags + 1;
  const limit = Math.min(total, out.length);
  if (limit <= 0) return;
  if (n === 0) {
    out.fill(0, 0, limit);
    return;
  }

  const meanA = sumRange(a, n) / n;
  const meanB = sumRange(b, n) / n;
  let energyA = 0;
  let energyB = 0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    energyA += da * da;
    energyB += db * db;
  }
  const norm = Math.sqrt(energyA * energyB);

  for (let lag = -lags; lag <= lags; lag += 1) {
    const slot = lag + lags;
    if (slot >= limit) break;
    if (norm <= 0) {
      out[slot] = 0;
      continue;
    }
    const start = lag < 0 ? -lag : 0;
    const end = lag > 0 ? n - lag : n;
    let sum = 0;
    for (let i = start; i < end; i += 1) {
      sum += (a[i] - meanA) * (b[i + lag] - meanB);
    }
    out[slot] = sum / norm;
  }
  if (out.length > limit) out.fill(0, limit);
}
