import { SpringScalar, SpringVec3, damp, dampAngle, easeInOutQuart, easeOutBack, easeOutElastic, easeOutExpo } from './easing';
import { curlNoise3, fbm3, simplex3 } from './noise';
import { Rng, hashSeed } from './rng';
import { SpatialHash } from './spatial';
import { buildArcLengthTable, catmullRom, cubicBezier, sampleArc, sampleAtDistance } from './spline';
import {
  coefficientOfVariation,
  crossCorrelation,
  dominantFrequency,
  fanoFactor,
  fft,
  interSpikeIntervals,
  mean,
  powerSpectrum,
  stdDev,
} from './stats';

/**
 * Self-check for the invariants this package promises. Everything here is
 * deterministic: the generators are seeded, so a run either passes on every
 * machine or fails on every machine.
 *
 * Returns the names of the invariants that failed; an empty array means the
 * package is behaving.
 */
export function verifyMath(): string[] {
  const failures: string[] = [];
  const check = (name: string, ok: boolean): void => {
    if (!ok) failures.push(name);
  };
  const near = (a: number, b: number, tolerance: number): boolean =>
    Number.isFinite(a) && Math.abs(a - b) <= tolerance;

  checkRng(check, near);
  checkNoise(check, near);
  checkSpline(check, near);
  checkEasing(check, near);
  checkSpatial(check);
  checkStats(check, near);

  return failures;
}

type Check = (name: string, ok: boolean) => void;
type Near = (a: number, b: number, tolerance: number) => boolean;

const SAMPLES = 100000;

function checkRng(check: Check, near: Near): void {
  const a = new Rng(0xc0ffee);
  const b = new Rng(0xc0ffee);
  let identical = true;
  for (let i = 0; i < 4096; i += 1) {
    if (a.next() !== b.next()) identical = false;
  }
  check('rng.determinism', identical);

  a.reset();
  b.reset(0xc0ffee);
  let resetMatches = true;
  for (let i = 0; i < 1024; i += 1) {
    if (a.next() !== b.next()) resetMatches = false;
  }
  check('rng.reset-restores-stream', resetMatches);
  check('rng.seed-exposed', a.seed === 0xc0ffee);

  const parent = new Rng(7);
  const fork1 = parent.fork(1);
  const fork2 = parent.fork(1);
  const fork3 = parent.fork(2);
  check('rng.fork-determinism', fork1.next() === fork2.next());
  check('rng.fork-salt-distinct', fork1.next() !== fork3.next());

  // Parent and child must not be measurably correlated.
  const streamLength = 20000;
  const parentStream = new Float32Array(streamLength);
  const childStream = new Float32Array(streamLength);
  const source = new Rng(31337);
  const child = source.fork(0x5eed);
  for (let i = 0; i < streamLength; i += 1) {
    parentStream[i] = source.next();
    childStream[i] = child.next();
  }
  const correlation = new Float32Array(1);
  crossCorrelation(parentStream, childStream, 0, correlation);
  check('rng.fork-independence', Math.abs(correlation[0]) < 0.03);

  const uniform = new Rng(11);
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < SAMPLES; i += 1) {
    const v = uniform.next();
    if (v < 0 || v >= 1) {
      check('rng.uniform-range', false);
      break;
    }
    sum += v;
    sumSq += v * v;
  }
  const uniformMean = sum / SAMPLES;
  const uniformVariance = sumSq / SAMPLES - uniformMean * uniformMean;
  check('rng.uniform-mean', near(uniformMean, 0.5, 0.005));
  check('rng.uniform-variance', near(uniformVariance, 1 / 12, 0.002));

  const gaussian = new Rng(12);
  let gSum = 0;
  let gSumSq = 0;
  for (let i = 0; i < SAMPLES; i += 1) {
    const v = gaussian.normal(0, 1);
    gSum += v;
    gSumSq += v * v;
  }
  const gMean = gSum / SAMPLES;
  const gVariance = gSumSq / SAMPLES - gMean * gMean;
  check('rng.normal-mean', near(gMean, 0, 0.02));
  check('rng.normal-variance', near(gVariance, 1, 0.02));

  const shifted = new Rng(13);
  let sSum = 0;
  for (let i = 0; i < SAMPLES; i += 1) sSum += shifted.normal(-70, 4);
  check('rng.normal-shifted-mean', near(sSum / SAMPLES, -70, 0.1));

  const exponential = new Rng(14);
  let eSum = 0;
  for (let i = 0; i < SAMPLES; i += 1) eSum += exponential.exponential(2.5);
  check('rng.exponential-mean', near(eSum / SAMPLES, 0.4, 0.01));

  check('rng.poisson-small', checkPoisson(new Rng(15), 4, 0.1, 0.3, near));
  check('rng.poisson-large', checkPoisson(new Rng(16), 120, 1, 12, near));

  const buckets = new Uint32Array(6);
  const dice = new Rng(17);
  const rolls = 60000;
  let inRange = true;
  for (let i = 0; i < rolls; i += 1) {
    const roll = dice.int(6);
    if (roll < 0 || roll > 5) inRange = false;
    else buckets[roll] += 1;
  }
  let uniformBuckets = inRange;
  for (let i = 0; i < 6; i += 1) {
    if (Math.abs(buckets[i] - rolls / 6) > 500) uniformBuckets = false;
  }
  check('rng.int-uniformity', uniformBuckets);
  check('rng.int-degenerate', dice.int(0) === 0 && dice.int(-3) === 0);

  const items: number[] = [];
  for (let i = 0; i < 100; i += 1) items.push(i);
  const shuffled = new Rng(18).shuffle(items.slice());
  const seen = new Uint8Array(100);
  let permutation = shuffled.length === 100;
  let moved = false;
  for (let i = 0; i < shuffled.length; i += 1) {
    const v = shuffled[i];
    if (!Number.isInteger(v) || v < 0 || v > 99 || seen[v] === 1) permutation = false;
    else seen[v] = 1;
    if (v !== i) moved = true;
  }
  check('rng.shuffle-permutation', permutation && moved);

  const sphere = new Rng(19);
  let onUnit = true;
  let insideBall = true;
  let radiusSum = 0;
  const trials = 20000;
  for (let i = 0; i < trials; i += 1) {
    const s = sphere.onSphere();
    if (!near(Math.hypot(s.x, s.y, s.z), 1, 1e-6)) onUnit = false;
    const p = sphere.inSphere();
    const r = Math.hypot(p.x, p.y, p.z);
    if (r > 1) insideBall = false;
    radiusSum += r;
  }
  check('rng.on-sphere-unit', onUnit);
  check('rng.in-sphere-bounded', insideBall);
  check('rng.in-sphere-density', near(radiusSum / trials, 0.75, 0.02));

  check('rng.bool-bounds', !new Rng(20).bool(0) && new Rng(20).bool(1));
  check('rng.pick-member', [4, 8, 15].includes(new Rng(21).pick([4, 8, 15])));

  check('hashSeed.determinism', hashSeed(1, 2, 3) === hashSeed(1, 2, 3));
  check('hashSeed.order-sensitive', hashSeed(1, 2, 3) !== hashSeed(3, 2, 1));
  const h = hashSeed(9, 0.5, -3);
  check('hashSeed.uint32', Number.isInteger(h) && h >= 0 && h <= 0xffffffff);
}

function checkPoisson(rng: Rng, lambda: number, meanTol: number, varTol: number, near: Near): boolean {
  const n = 20000;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i += 1) {
    const k = rng.poisson(lambda);
    if (!Number.isInteger(k) || k < 0) return false;
    sum += k;
    sumSq += k * k;
  }
  const m = sum / n;
  const v = sumSq / n - m * m;
  return near(m, lambda, meanTol) && near(v, lambda, varTol);
}

function checkNoise(check: Check, near: Near): void {
  const rng = new Rng(2024);
  let inRange = true;
  let continuous = true;
  let sum = 0;
  let sumSq = 0;
  const n = 20000;
  for (let i = 0; i < n; i += 1) {
    const x = rng.range(-500, 500);
    const y = rng.range(-500, 500);
    const z = rng.range(-500, 500);
    const v = simplex3(x, y, z);
    if (!(v >= -1 && v <= 1)) inRange = false;
    sum += v;
    sumSq += v * v;
    // A lattice or permutation bug shows up as a discontinuity here.
    const delta = Math.abs(simplex3(x + 1e-3, y, z) - v);
    if (!(delta < 0.05)) continuous = false;
  }
  check('simplex3.range', inRange);
  check('simplex3.continuity', continuous);
  const noiseMean = sum / n;
  check('simplex3.zero-mean', near(noiseMean, 0, 0.05));
  check('simplex3.non-degenerate', sumSq / n - noiseMean * noiseMean > 0.02);
  check('simplex3.lattice-nodes-vanish', Math.abs(simplex3(0, 0, 0)) < 1e-9);

  let fbmInRange = true;
  for (let i = 0; i < 4000; i += 1) {
    const v = fbm3(rng.range(-100, 100), rng.range(-100, 100), rng.range(-100, 100));
    if (!(v >= -1 && v <= 1)) fbmInRange = false;
  }
  check('fbm3.range', fbmInRange);

  let curlFinite = true;
  let curlMagnitude = 0;
  for (let i = 0; i < 500; i += 1) {
    const c = curlNoise3(rng.range(-50, 50), rng.range(-50, 50), rng.range(-50, 50));
    if (!Number.isFinite(c.x) || !Number.isFinite(c.y) || !Number.isFinite(c.z)) curlFinite = false;
    curlMagnitude += Math.hypot(c.x, c.y, c.z);
  }
  check('curlNoise3.finite', curlFinite);
  check('curlNoise3.non-zero', curlMagnitude / 500 > 1e-4);
}

function checkSpline(check: Check, near: Near): void {
  check('catmullRom.endpoints', catmullRom(0, 3, 7, 9, 0) === 3 && catmullRom(0, 3, 7, 9, 1) === 7);
  check(
    'cubicBezier.endpoints',
    cubicBezier(2, 5, 8, 11, 0) === 2 && cubicBezier(2, 5, 8, 11, 1) === 11,
  );
  check('cubicBezier.linear', near(cubicBezier(0, 1, 2, 3, 0.5), 1.5, 1e-12));

  const count = 128;
  const points = new Float32Array(count * 3 + 6);
  sampleArc(0, 0, 0, 10, 0, 0, 4, count, points, 6);
  check(
    'sampleArc.offset-untouched',
    points[0] === 0 && points[1] === 0 && points[2] === 0 && points[5] === 0,
  );
  const body = points.subarray(6);
  check(
    'sampleArc.endpoints',
    near(body[0], 0, 1e-5) &&
      near(body[1], 0, 1e-5) &&
      near(body[(count - 1) * 3], 10, 1e-5) &&
      near(body[(count - 1) * 3 + 1], 0, 1e-5),
  );
  // The control point sits `sag` above the chord, so the curve peaks at sag/2.
  const midIndex = ((count - 1) >> 1) * 3;
  check('sampleArc.sag-direction', body[midIndex + 1] > 1.9 && body[midIndex + 1] < 2.1);

  // A chord parallel to world up must still produce a finite perpendicular.
  const vertical = new Float32Array(count * 3);
  sampleArc(0, 0, 0, 0, 10, 0, 3, count, vertical);
  let verticalFinite = true;
  let verticalBulge = 0;
  for (let i = 0; i < count * 3; i += 1) {
    if (!Number.isFinite(vertical[i])) verticalFinite = false;
  }
  for (let i = 0; i < count; i += 1) {
    verticalBulge = Math.max(verticalBulge, Math.hypot(vertical[i * 3], vertical[i * 3 + 2]));
  }
  check('sampleArc.vertical-stable', verticalFinite && near(verticalBulge, 1.5, 0.05));

  const table = buildArcLengthTable(body, count);
  const total = table[count - 1];
  check('buildArcLengthTable.monotonic', isMonotonic(table, count) && total > 10);

  const out = { x: 0, y: 0, z: 0 };
  sampleAtDistance(body, table, count, 0, out);
  check('sampleAtDistance.start', near(out.x, 0, 1e-5) && near(out.y, 0, 1e-5));
  sampleAtDistance(body, table, count, total * 2, out);
  check('sampleAtDistance.clamped-end', near(out.x, 10, 1e-5) && near(out.y, 0, 1e-5));

  const steps = 64;
  const expected = total / steps;
  let previousX = 0;
  let previousY = 0;
  let previousZ = 0;
  let worst = 0;
  for (let i = 0; i <= steps; i += 1) {
    sampleAtDistance(body, table, count, (total * i) / steps, out);
    if (i > 0) {
      const d = Math.hypot(out.x - previousX, out.y - previousY, out.z - previousZ);
      worst = Math.max(worst, Math.abs(d - expected));
    }
    previousX = out.x;
    previousY = out.y;
    previousZ = out.z;
  }
  check('sampleAtDistance.constant-speed', worst <= expected * 0.02);
}

function isMonotonic(table: Float32Array, count: number): boolean {
  if (table[0] !== 0) return false;
  for (let i = 1; i < count; i += 1) {
    if (!(table[i] >= table[i - 1])) return false;
  }
  return true;
}

function checkEasing(check: Check, near: Near): void {
  const curves: ReadonlyArray<readonly [string, (t: number) => number]> = [
    ['easeOutExpo', easeOutExpo],
    ['easeInOutQuart', easeInOutQuart],
    ['easeOutBack', easeOutBack],
    ['easeOutElastic', easeOutElastic],
  ];
  for (const [name, fn] of curves) {
    check(`${name}.endpoints`, near(fn(0), 0, 1e-9) && near(fn(1), 1, 1e-9));
    let finite = true;
    for (let i = -5; i <= 15; i += 1) {
      if (!Number.isFinite(fn(i / 10))) finite = false;
    }
    check(`${name}.finite-outside-unit`, finite);
  }
  check('easeInOutQuart.symmetry', near(easeInOutQuart(0.5), 0.5, 1e-12));

  const once = damp(0, 1, 4, 0.5);
  const twice = damp(damp(0, 1, 4, 0.25), 1, 4, 0.25);
  check('damp.framerate-independent', near(once, twice, 1e-12));
  check('damp.zero-dt', damp(0.25, 1, 4, 0) === 0.25);

  const wrapped = dampAngle(3.0, -3.0, 10, 0.05);
  // The short way from 3.0 to -3.0 goes forwards through pi, not backwards.
  check('dampAngle.shortest-path', wrapped > 3.0 || wrapped < -3.0);
  check('dampAngle.bounded', Math.abs(dampAngle(3.1, -3.1, 40, 1)) <= Math.PI + 1e-9);

  const spring = new SpringScalar(0);
  spring.target = 1;
  check('SpringScalar.target-roundtrip', spring.target === 1);
  for (let i = 0; i < 600; i += 1) spring.step(1 / 60);
  check('SpringScalar.converges', near(spring.value, 1, 1e-3) && spring.settled);

  const stiff = new SpringScalar(0, 900, 40, 1);
  stiff.target = 5;
  let stable = true;
  for (let i = 0; i < 40; i += 1) {
    const v = stiff.step(0.25);
    if (!Number.isFinite(v) || Math.abs(v) > 50) stable = false;
  }
  check('SpringScalar.large-dt-stable', stable && near(stiff.value, 5, 0.01));

  const jumped = new SpringScalar(0);
  jumped.target = 10;
  jumped.step(0.1);
  jumped.jump(-2);
  check('SpringScalar.jump', jumped.value === -2 && jumped.target === -2 && jumped.settled);

  const vec = new SpringVec3(0, 0, 0);
  vec.setTarget(1, -2, 3);
  for (let i = 0; i < 600; i += 1) vec.step(1 / 60);
  check(
    'SpringVec3.converges',
    near(vec.x, 1, 1e-3) && near(vec.y, -2, 1e-3) && near(vec.z, 3, 1e-3) && vec.settled,
  );
}

function checkSpatial(check: Check): void {
  const rng = new Rng(4242);
  const count = 400;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i += 1) positions[i] = rng.range(-25, 25);

  const grid = new SpatialHash(4);
  grid.rebuild(positions, count);
  const out = new Uint32Array(count);
  const seen = new Uint8Array(count);

  let radiusMatches = true;
  for (let q = 0; q < 24 && radiusMatches; q += 1) {
    const x = rng.range(-30, 30);
    const y = rng.range(-30, 30);
    const z = rng.range(-30, 30);
    const radius = rng.range(0.5, 12);
    const hits = grid.queryRadius(x, y, z, radius, out);
    let expected = 0;
    for (let i = 0; i < count; i += 1) {
      if (distance2(positions, i, x, y, z) <= radius * radius) expected += 1;
    }
    if (hits !== expected) radiusMatches = false;
    seen.fill(0);
    for (let i = 0; i < hits; i += 1) {
      const index = out[i];
      if (index >= count || seen[index] === 1) radiusMatches = false;
      else seen[index] = 1;
      if (distance2(positions, index, x, y, z) > radius * radius) radiusMatches = false;
    }
  }
  check('SpatialHash.queryRadius-matches-brute-force', radiusMatches);

  const small = new Uint32Array(3);
  const dense = grid.queryRadius(0, 0, 0, 60, small);
  check('SpatialHash.queryRadius-respects-out-capacity', dense === 3);

  let nearestMatches = true;
  for (let q = 0; q < 24 && nearestMatches; q += 1) {
    const x = rng.range(-30, 30);
    const y = rng.range(-30, 30);
    const z = rng.range(-30, 30);
    const maxRadius = rng.range(1, 10);
    const got = grid.nearest(x, y, z, maxRadius);
    let expected = -1;
    let best = maxRadius * maxRadius;
    for (let i = 0; i < count; i += 1) {
      const d2 = distance2(positions, i, x, y, z);
      if (d2 <= best) {
        best = d2;
        expected = i;
      }
    }
    if (got !== expected) nearestMatches = false;
  }
  check('SpatialHash.nearest-matches-brute-force', nearestMatches);

  let rayMatches = true;
  for (let q = 0; q < 24 && rayMatches; q += 1) {
    const ox = rng.range(-60, 60);
    const oy = rng.range(-60, 60);
    const oz = rng.range(-60, 60);
    const target = rng.int(count);
    const dx = positions[target * 3] - ox;
    const dy = positions[target * 3 + 1] - oy;
    const dz = positions[target * 3 + 2] - oz;
    const length = Math.hypot(dx, dy, dz);
    const rx = dx / length;
    const ry = dy / length;
    const rz = dz / length;
    const threshold = 1.5;
    const got = grid.raycast(ox, oy, oz, dx, dy, dz, threshold);

    let expected = -1;
    let bestAlong = Infinity;
    for (let i = 0; i < count; i += 1) {
      const vx = positions[i * 3] - ox;
      const vy = positions[i * 3 + 1] - oy;
      const vz = positions[i * 3 + 2] - oz;
      let along = vx * rx + vy * ry + vz * rz;
      if (along < 0) along = 0;
      const ex = vx - rx * along;
      const ey = vy - ry * along;
      const ez = vz - rz * along;
      if (ex * ex + ey * ey + ez * ez <= threshold * threshold && along < bestAlong) {
        bestAlong = along;
        expected = i;
      }
    }
    if (got !== expected) rayMatches = false;
  }
  check('SpatialHash.raycast-matches-brute-force', rayMatches);

  const empty = new SpatialHash(2);
  check(
    'SpatialHash.empty-is-safe',
    empty.queryRadius(0, 0, 0, 5, out) === 0 &&
      empty.nearest(0, 0, 0, 5) === -1 &&
      empty.raycast(0, 0, 0, 0, 0, 1, 1) === -1,
  );

  empty.rebuild(positions, count);
  empty.rebuild(positions, 0);
  check('SpatialHash.rebuild-to-empty', empty.nearest(0, 0, 0, 100) === -1);
}

function distance2(positions: Float32Array, index: number, x: number, y: number, z: number): number {
  const p = index * 3;
  const dx = positions[p] - x;
  const dy = positions[p + 1] - y;
  const dz = positions[p + 2] - z;
  return dx * dx + dy * dy + dz * dz;
}

function checkStats(check: Check, near: Near): void {
  const values = new Float32Array([2, 4, 4, 4, 5, 5, 7, 9]);
  check('mean.known', near(mean(values), 5, 1e-6));
  check('mean.partial', near(mean(values, 4), 3.5, 1e-6));
  check('stdDev.known', near(stdDev(values), Math.sqrt(32 / 7), 1e-6));
  check('stdDev.degenerate', stdDev(new Float32Array([3])) === 0);

  const times = new Float32Array([0, 10, 20, 30, 40]);
  const intervals = new Float32Array(8);
  const written = interSpikeIntervals(times, 5, intervals);
  check('interSpikeIntervals.count', written === 4);
  check('interSpikeIntervals.values', intervals[0] === 10 && intervals[3] === 10);
  check('coefficientOfVariation.regular', coefficientOfVariation(intervals, written) === 0);

  const jittered = new Float32Array([8, 12, 8, 12]);
  check('coefficientOfVariation.jittered', near(coefficientOfVariation(jittered, 4), 0.2309, 1e-3));
  check('fanoFactor.constant', fanoFactor(new Float32Array([2, 2, 2, 2]), 4) === 0);
  check('fanoFactor.known', near(fanoFactor(new Float32Array([1, 3, 1, 3]), 4), 2 / 3, 1e-6));

  // Forward transform of a tone that lands exactly on bin 8.
  const n = 256;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  const original = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const v = Math.cos((2 * Math.PI * 8 * i) / n);
    real[i] = v;
    original[i] = v;
  }
  fft(real, imag);
  let peakBin = 0;
  let peakValue = -1;
  for (let k = 0; k < n / 2; k += 1) {
    const p = real[k] * real[k] + imag[k] * imag[k];
    if (p > peakValue) {
      peakValue = p;
      peakBin = k;
    }
  }
  check('fft.tone-lands-on-its-bin', peakBin === 8);
  check('fft.tone-amplitude', near(Math.sqrt(peakValue) / (n / 2), 1, 1e-3));

  // Round trip: conjugate, transform, conjugate, scale.
  for (let i = 0; i < n; i += 1) imag[i] = -imag[i];
  fft(real, imag);
  let worst = 0;
  for (let i = 0; i < n; i += 1) {
    worst = Math.max(worst, Math.abs(real[i] / n - original[i]));
  }
  check('fft.round-trip', worst < 1e-4);

  let rejectsNonPowerOfTwo = false;
  try {
    fft(new Float32Array(6), new Float32Array(6));
  } catch {
    rejectsNonPowerOfTwo = true;
  }
  check('fft.rejects-non-power-of-two', rejectsNonPowerOfTwo);

  // On-bin tone: amplitude normalisation must be exact.
  const rate = 1000;
  const length = 1024;
  const onBinHz = (40 * rate) / length;
  const amplitude = 1.5;
  const signal = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    signal[i] = amplitude * Math.sin((2 * Math.PI * onBinHz * i) / rate);
  }
  const spectrum = new Float32Array(length / 2 + 1);
  const binHz = powerSpectrum(signal, rate, spectrum);
  check('powerSpectrum.bin-width', near(binHz, rate / length, 1e-9));
  check('powerSpectrum.amplitude-normalised', near(spectrum[40], (amplitude * amplitude) / 2, 0.02));
  check('powerSpectrum.dc-rejected', spectrum[0] < 1e-3);
  check('powerSpectrum.dominant', near(dominantFrequency(spectrum, binHz, 1, 200), onBinHz, 0.2));

  // Off-bin tone in a coarse spectrum: only sub-bin interpolation can find it.
  const coarseLength = 256;
  const coarse = new Float32Array(coarseLength);
  const offBinHz = 41.5;
  for (let i = 0; i < coarseLength; i += 1) {
    coarse[i] = Math.sin((2 * Math.PI * offBinHz * i) / rate);
  }
  const coarseSpectrum = new Float32Array(coarseLength / 2 + 1);
  const coarseBinHz = powerSpectrum(coarse, rate, coarseSpectrum);
  const estimated = dominantFrequency(coarseSpectrum, coarseBinHz, 5, 200);
  check('dominantFrequency.sub-bin-accuracy', near(estimated, offBinHz, 0.5));
  check('dominantFrequency.beats-bin-centre', coarseBinHz > 3 && Math.abs(estimated - offBinHz) < coarseBinHz / 4);
  const outOfBand = dominantFrequency(coarseSpectrum, coarseBinHz, 100, 200);
  check('dominantFrequency.band-limited', outOfBand >= 100 - coarseBinHz && outOfBand <= 200 + coarseBinHz);

  // Non-power-of-two records must be zero padded internally.
  const oddLength = 300;
  const odd = new Float32Array(oddLength);
  for (let i = 0; i < oddLength; i += 1) odd[i] = Math.sin((2 * Math.PI * 40 * i) / rate);
  const oddSpectrum = new Float32Array(512);
  const oddBinHz = powerSpectrum(odd, rate, oddSpectrum);
  check('powerSpectrum.zero-pads', near(oddBinHz, rate / 512, 1e-9));
  check('powerSpectrum.padded-dominant', near(dominantFrequency(oddSpectrum, oddBinHz, 5, 200), 40, 1));

  const wave = new Float32Array(512);
  const lagged = new Float32Array(512);
  for (let i = 0; i < 512; i += 1) {
    wave[i] = Math.sin((2 * Math.PI * i) / 64);
    lagged[i] = Math.sin((2 * Math.PI * (i + 5)) / 64);
  }
  const maxLag = 32;
  const correlogram = new Float32Array(2 * maxLag + 1);
  crossCorrelation(wave, wave, maxLag, correlogram);
  check('crossCorrelation.self-at-zero-lag', near(correlogram[maxLag], 1, 1e-4));
  crossCorrelation(wave, lagged, maxLag, correlogram);
  let bestLag = -maxLag;
  let best = -Infinity;
  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    if (correlogram[lag + maxLag] > best) {
      best = correlogram[lag + maxLag];
      bestLag = lag;
    }
  }
  check('crossCorrelation.recovers-shift', bestLag === -5);
  check('crossCorrelation.bounded', best <= 1 + 1e-6);
}
