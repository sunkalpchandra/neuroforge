import { sampleArc } from '@neuroforge/math';

/**
 * CPU-side evaluation of the axon spline.
 *
 * The ribbon material, the particle advection and `sampleArc` in
 * `@neuroforge/math` must agree exactly about where a synapse's curve runs, or
 * the travelling impulse drifts off the ribbon it is supposed to be riding. So
 * the control point is not re-derived here: it is recovered from `sampleArc`
 * itself, which makes the maths package the single definition.
 */

const SAMPLES = 9;
const samples = new Float32Array(SAMPLES * 3);
const midpoint = new Float32Array(9);

/** Bow applied to a synapse whose `arc` column is still zero. */
const DEFAULT_SAG_RATIO = 0.15;

export function chordLength(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * A flat bundle of perfectly straight axons reads as a plate rather than as
 * wiring, so a synapse that has never been given an explicit arc gets one
 * proportional to its span.
 */
export function axonSag(arc: number, chord: number): number {
  return arc !== 0 ? arc : chord * DEFAULT_SAG_RATIO;
}

/**
 * Quadratic control point of the arc, written into `out`.
 *
 * `sampleArc` evaluates a quadratic Bezier, so its midpoint sample inverts to
 * the control point exactly: P(0.5) = (a + 2c + b) / 4.
 */
export function controlPoint(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  sag: number,
  out: Float32Array,
  outOffset: number,
): void {
  sampleArc(ax, ay, az, bx, by, bz, sag, 3, midpoint, 0);
  out[outOffset] = 2 * midpoint[3] - 0.5 * (ax + bx);
  out[outOffset + 1] = 2 * midpoint[4] - 0.5 * (ay + by);
  out[outOffset + 2] = 2 * midpoint[5] - 0.5 * (az + bz);
}

/** Arc length of the curve, so impulses travel at a speed in world units. */
export function arcLength(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  sag: number,
): number {
  sampleArc(ax, ay, az, bx, by, bz, sag, SAMPLES, samples, 0);
  let total = 0;
  for (let i = 1; i < SAMPLES; i += 1) {
    const p = (i - 1) * 3;
    const q = i * 3;
    const dx = samples[q] - samples[p];
    const dy = samples[q + 1] - samples[p + 1];
    const dz = samples[q + 2] - samples[p + 2];
    total += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return total;
}

/** Point on the curve at parameter t, given a precomputed control point. */
export function pointAt(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  control: Float32Array,
  controlOffset: number,
  t: number,
  out: Float32Array,
  outOffset: number,
): void {
  const s = 1 - t;
  const w0 = s * s;
  const w1 = 2 * s * t;
  const w2 = t * t;
  out[outOffset] = w0 * ax + w1 * control[controlOffset] + w2 * bx;
  out[outOffset + 1] = w0 * ay + w1 * control[controlOffset + 1] + w2 * by;
  out[outOffset + 2] = w0 * az + w1 * control[controlOffset + 2] + w2 * bz;
}

/** Unnormalised derivative of the curve at parameter t. */
export function tangentAt(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  control: Float32Array,
  controlOffset: number,
  t: number,
  out: Float32Array,
  outOffset: number,
): void {
  const s = 1 - t;
  out[outOffset] = 2 * s * (control[controlOffset] - ax) + 2 * t * (bx - control[controlOffset]);
  out[outOffset + 1] =
    2 * s * (control[controlOffset + 1] - ay) + 2 * t * (by - control[controlOffset + 1]);
  out[outOffset + 2] =
    2 * s * (control[controlOffset + 2] - az) + 2 * t * (bz - control[controlOffset + 2]);
}
