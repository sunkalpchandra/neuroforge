/**
 * Curve sampling and arc-length reparameterisation.
 *
 * Every routine works on flat xyz triples so the results can be uploaded
 * straight into a vertex buffer without a copy or a conversion pass.
 */

/** Below this the chord is treated as degenerate and the arc becomes a point. */
const LENGTH_EPSILON = 1e-9;

/**
 * If the squared cosine between the chord and world up exceeds this, the
 * Gram-Schmidt step against up would lose all its precision.
 */
const PARALLEL_EPSILON = 1e-6;

/** Uniform Catmull-Rom through p1 and p2, one scalar component at a time. */
export function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    0.5 *
    (2 * p1 +
      (p2 - p0) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (3 * p1 - p0 - 3 * p2 + p3) * t3)
  );
}

/** Cubic Bezier in Bernstein form. */
export function cubicBezier(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const mt = 1 - t;
  const mt2 = mt * mt;
  const t2 = t * t;
  return mt2 * mt * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t2 * t * p3;
}

/**
 * Sample a quadratic arc from a to b, bulging by `sag` along the direction
 * perpendicular to the chord that lies closest to world up.
 *
 * Writes exactly `3 * samples` floats starting at `offset`; the first sample is
 * a and the last is b.
 */
export function sampleArc(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
  sag: number,
  samples: number,
  out: Float32Array,
  offset = 0,
): void {
  const count = Math.floor(samples);
  if (count <= 0) return;
  const base = Math.floor(offset);
  if (base < 0 || base + count * 3 > out.length) {
    throw new RangeError(
      `sampleArc needs ${count * 3} floats at offset ${base}, out holds ${out.length}`,
    );
  }

  const dx = bx - ax;
  const dy = by - ay;
  const dz = bz - az;
  const length = Math.sqrt(dx * dx + dy * dy + dz * dz);

  let nx = 0;
  let ny = 1;
  let nz = 0;
  if (length > LENGTH_EPSILON) {
    const ux = dx / length;
    const uy = dy / length;
    const uz = dz / length;
    // Reference direction: world up, unless the chord is nearly parallel to it,
    // in which case the world axis least aligned with the chord is used instead
    // so the projection below stays well conditioned.
    let rx = 0;
    let ry = 1;
    let rz = 0;
    if (uy * uy > 1 - PARALLEL_EPSILON) {
      if (Math.abs(ux) <= Math.abs(uz)) {
        rx = 1;
        ry = 0;
      } else {
        rz = 1;
        ry = 0;
      }
    }
    const dot = ux * rx + uy * ry + uz * rz;
    const px = rx - ux * dot;
    const py = ry - uy * dot;
    const pz = rz - uz * dot;
    const plen = Math.sqrt(px * px + py * py + pz * pz);
    if (plen > LENGTH_EPSILON) {
      nx = px / plen;
      ny = py / plen;
      nz = pz / plen;
    }
  }

  const cx = (ax + bx) * 0.5 + nx * sag;
  const cy = (ay + by) * 0.5 + ny * sag;
  const cz = (az + bz) * 0.5 + nz * sag;

  const denom = count > 1 ? count - 1 : 1;
  for (let i = 0; i < count; i += 1) {
    const t = i / denom;
    const mt = 1 - t;
    const w0 = mt * mt;
    const w1 = 2 * mt * t;
    const w2 = t * t;
    const o = base + i * 3;
    out[o] = w0 * ax + w1 * cx + w2 * bx;
    out[o + 1] = w0 * ay + w1 * cy + w2 * by;
    out[o + 2] = w0 * az + w1 * cz + w2 * bz;
  }
}

/**
 * Cumulative chord length of a sampled polyline: `table[i]` is the distance from
 * the first point to point i, so `table[count-1]` is the total length.
 */
export function buildArcLengthTable(points: Float32Array, count: number): Float32Array {
  const n = Math.max(0, Math.min(Math.floor(count), Math.floor(points.length / 3)));
  const table = new Float32Array(n);
  let total = 0;
  for (let i = 1; i < n; i += 1) {
    const p = (i - 1) * 3;
    const q = i * 3;
    const dx = points[q] - points[p];
    const dy = points[q + 1] - points[p + 1];
    const dz = points[q + 2] - points[p + 2];
    total += Math.sqrt(dx * dx + dy * dy + dz * dz);
    table[i] = total;
  }
  return table;
}

/**
 * Position at a given distance along the polyline, by binary search over the
 * cumulative table.
 *
 * This is what makes an impulse travel at a constant speed along an axon:
 * stepping the curve parameter uniformly would visibly accelerate through the
 * flat part of an arc and stall through the bend.
 */
export function sampleAtDistance(
  points: Float32Array,
  table: Float32Array,
  count: number,
  distance: number,
  out: { x: number; y: number; z: number },
): void {
  const n = Math.max(0, Math.min(Math.floor(count), table.length, Math.floor(points.length / 3)));
  if (n === 0) {
    out.x = 0;
    out.y = 0;
    out.z = 0;
    return;
  }
  if (n === 1) {
    out.x = points[0];
    out.y = points[1];
    out.z = points[2];
    return;
  }

  const total = table[n - 1];
  // The comparison order also maps NaN onto the start of the curve.
  let d = distance > 0 ? distance : 0;
  if (d > total) d = total;

  // Largest index whose cumulative distance is still <= d.
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    if (table[mid] <= d) lo = mid;
    else hi = mid - 1;
  }
  const i = lo < n - 1 ? lo : n - 2;

  const span = table[i + 1] - table[i];
  const t = span > 0 ? (d - table[i]) / span : 0;
  const p = i * 3;
  const q = p + 3;
  out.x = points[p] + (points[q] - points[p]) * t;
  out.y = points[p + 1] + (points[q + 1] - points[p + 1]) * t;
  out.z = points[p + 2] + (points[q + 2] - points[p + 2]) * t;
}
