import * as THREE from 'three';
import { Rng, fbm3, hashSeed, simplex3 } from '@neuroforge/math';
import { defaultMorphology } from '@neuroforge/shared';
import type { Morphology, MorphologyArchetype } from '@neuroforge/shared';
import { MeshSink, emitSphere, emitTube, unitSphere } from './mesh-builder';

/**
 * Procedural neuron glyphs.
 *
 * A neuron is a few dozen bytes of `Morphology` on disk and a few thousand
 * triangles on the GPU, generated here. Nothing is modelled by hand and nothing
 * is a primitive: the soma is a noise-deformed icosphere, the arbors are swept
 * tapered tubes along recursively branched paths, and the whole thing is
 * deterministic in `morphology.seed`.
 *
 * Every vertex carries `aBranchT`, the normalised path distance from the soma,
 * which is what lets a material animate a signal travelling outward.
 */

const TAU = Math.PI * 2;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * Hard ceiling on branches per arbor. Purkinje at depth 6 would otherwise
 * generate 127 branches per trunk; the budget bounds the geometry without
 * making the silhouette any less characteristic.
 */
const MAX_BRANCHES = 240;

const MAX_PATH_POINTS = 16;

/**
 * Path scratch shared by every branch.
 *
 * Safe under recursion because a branch emits its tube before it spawns any
 * child and carries only scalars — its end point, end direction and end
 * parameter — across the recursive call.
 */
const pathScratch = new Float32Array(MAX_PATH_POINTS * 3);
const radiusScratch = new Float32Array(MAX_PATH_POINTS);
const branchTScratch = new Float32Array(MAX_PATH_POINTS);
const endDir = new Float32Array(3);
const dirScratch = new Float32Array(3);

/** Frequency of the field that bends branch paths, in glyph-local units. */
const WANDER_FREQUENCY = 0.55;

interface GrowthConfig {
  children: number;
  spread: number;
  lengthTaper: number;
  radiusTaper: number;
  curl: number;
  droop: number;
  /** Multiplier applied to every generated z component; < 1 flattens the arbor. */
  flatten: number;
  /** Terminal tip radius as a fraction of the branch's start radius. */
  tipRadius: number;
  invTotal: number;
  boutons: number;
  boutonRadius: number;
  budget: { left: number };
}

interface TrunkSpec {
  dx: number;
  dy: number;
  dz: number;
  lengthScale: number;
  radiusScale: number;
  depth: number;
}

function normalizeInto(x: number, y: number, z: number, out: Float32Array): void {
  const length = Math.sqrt(x * x + y * y + z * z);
  if (length > 1e-9) {
    out[0] = x / length;
    out[1] = y / length;
    out[2] = z / length;
  } else {
    out[0] = 0;
    out[1] = 1;
    out[2] = 0;
  }
}

/** Unit vector perpendicular to (x,y,z), chosen for numerical stability. */
function perpendicularInto(x: number, y: number, z: number, out: Float32Array): void {
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  let rx = 0;
  let ry = 0;
  let rz = 0;
  if (ax <= ay && ax <= az) rx = 1;
  else if (ay <= az) ry = 1;
  else rz = 1;
  normalizeInto(ry * z - rz * y, rz * x - rx * z, rx * y - ry * x, out);
}

function fibonacciDirection(index: number, count: number, out: Float32Array): void {
  const y = count <= 1 ? 0 : 1 - (2 * index) / (count - 1);
  const radius = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = index * GOLDEN_ANGLE;
  out[0] = Math.cos(theta) * radius;
  out[1] = y;
  out[2] = Math.sin(theta) * radius;
}

/**
 * Walk a branch path into the shared scratch buffers.
 *
 * The direction is bent every step by a divergence-free-ish sample of simplex
 * noise projected perpendicular to the current heading, so branches wander
 * without ever folding back on themselves, and identical seeds wander
 * identically.
 */
function walkPath(
  rng: Rng,
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  length: number,
  startRadius: number,
  endRadius: number,
  tStart: number,
  invTotal: number,
  points: number,
  curl: number,
  droop: number,
  flatten: number,
): void {
  const count = Math.max(2, Math.min(MAX_PATH_POINTS, points));
  const step = length / (count - 1);
  const jitterX = rng.range(-90, 90);
  const jitterY = rng.range(-90, 90);
  const jitterZ = rng.range(-90, 90);

  let px = ox;
  let py = oy;
  let pz = oz;
  normalizeInto(dx, dy, dz, dirScratch);
  let ux = dirScratch[0];
  let uy = dirScratch[1];
  let uz = dirScratch[2];

  pathScratch[0] = px;
  pathScratch[1] = py;
  pathScratch[2] = pz;
  radiusScratch[0] = startRadius;
  branchTScratch[0] = tStart;

  for (let i = 1; i < count; i += 1) {
    const qx = px * WANDER_FREQUENCY + jitterX;
    const qy = py * WANDER_FREQUENCY + jitterY;
    const qz = pz * WANDER_FREQUENCY + jitterZ;
    const wx = simplex3(qx, qy, qz);
    const wy = simplex3(qx + 37.1, qy - 19.7, qz + 5.3);
    const wz = simplex3(qx - 11.9, qy + 43.5, qz - 27.4);
    const along = wx * ux + wy * uy + wz * uz;
    ux += (wx - ux * along) * curl;
    uy += (wy - uy * along) * curl - droop;
    uz += (wz - uz * along) * curl;
    uz *= flatten;
    normalizeInto(ux, uy, uz, dirScratch);
    ux = dirScratch[0];
    uy = dirScratch[1];
    uz = dirScratch[2];

    px += ux * step;
    py += uy * step;
    pz += uz * step;
    const t = i / (count - 1);
    const o = i * 3;
    pathScratch[o] = px;
    pathScratch[o + 1] = py;
    pathScratch[o + 2] = pz;
    radiusScratch[i] = startRadius + (endRadius - startRadius) * t;
    branchTScratch[i] = tStart + length * t * invTotal;
  }

  endDir[0] = ux;
  endDir[1] = uy;
  endDir[2] = uz;
}

function radialFor(level: number): number {
  return level === 0 ? 7 : level === 1 ? 5 : 4;
}

function pointsFor(level: number): number {
  return level === 0 ? 6 : level === 1 ? 4 : 3;
}

/** Emit `count` short twigs from a tip, each closed by a bouton. */
function emitTerminals(
  sink: MeshSink,
  rng: Rng,
  cfg: GrowthConfig,
  x: number,
  y: number,
  z: number,
  dx: number,
  dy: number,
  dz: number,
  count: number,
  radius: number,
  twigLength: number,
  spread: number,
  tStart: number,
): void {
  const e1 = new Float32Array(3);
  perpendicularInto(dx, dy, dz, e1);
  const e2x = dy * e1[2] - dz * e1[1];
  const e2y = dz * e1[0] - dx * e1[2];
  const e2z = dx * e1[1] - dy * e1[0];
  const rollBase = rng.next() * TAU;

  for (let i = 0; i < count; i += 1) {
    const roll = rollBase + (i / count) * TAU + rng.range(-0.3, 0.3);
    const angle = spread * (0.45 + 0.7 * rng.next());
    const ca = Math.cos(angle);
    const sa = Math.sin(angle);
    const cr = Math.cos(roll);
    const sr = Math.sin(roll);
    const bx = dx * ca + (e1[0] * cr + e2x * sr) * sa;
    const by = dy * ca + (e1[1] * cr + e2y * sr) * sa;
    const bz = (dz * ca + (e1[2] * cr + e2z * sr) * sa) * cfg.flatten;

    const tip = radius * 0.55;
    walkPath(
      rng,
      x,
      y,
      z,
      bx,
      by,
      bz,
      twigLength,
      radius,
      tip,
      tStart,
      cfg.invTotal,
      3,
      cfg.curl * 0.5,
      0,
      cfg.flatten,
    );
    emitTube(sink, pathScratch, radiusScratch, branchTScratch, 3, 4, false);
    emitSphere(
      sink,
      pathScratch[6],
      pathScratch[7],
      pathScratch[8],
      cfg.boutonRadius,
      0,
      branchTScratch[2],
    );
  }
}

/**
 * Grow one branch and, unless it is terminal, its children.
 *
 * `level` counts down from the trunk; `maxDepth` is how many further splits are
 * allowed, so a pyramidal apical trunk can branch deeper than its own basal
 * skirt without needing a second configuration.
 */
function growBranch(
  sink: MeshSink,
  rng: Rng,
  cfg: GrowthConfig,
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  length: number,
  radius: number,
  tStart: number,
  level: number,
  maxDepth: number,
): void {
  if (cfg.budget.left <= 0) return;
  cfg.budget.left -= 1;

  const terminal = level >= maxDepth;
  const endRadius = terminal ? radius * cfg.tipRadius : radius * cfg.radiusTaper;
  const points = pointsFor(level);

  walkPath(
    rng,
    ox,
    oy,
    oz,
    dx,
    dy,
    dz,
    length,
    radius,
    endRadius,
    tStart,
    cfg.invTotal,
    points,
    cfg.curl,
    cfg.droop,
    cfg.flatten,
  );

  const last = (points - 1) * 3;
  const ex = pathScratch[last];
  const ey = pathScratch[last + 1];
  const ez = pathScratch[last + 2];
  const fx = endDir[0];
  const fy = endDir[1];
  const fz = endDir[2];
  const tEnd = branchTScratch[points - 1];

  emitTube(
    sink,
    pathScratch,
    radiusScratch,
    branchTScratch,
    points,
    radialFor(level),
    terminal && cfg.boutons === 0,
  );

  if (terminal) {
    if (cfg.boutons > 0) {
      emitTerminals(
        sink,
        rng,
        cfg,
        ex,
        ey,
        ez,
        fx,
        fy,
        fz,
        cfg.boutons,
        endRadius,
        length * 0.22,
        1.0,
        tEnd,
      );
    }
    return;
  }

  const e1 = new Float32Array(3);
  perpendicularInto(fx, fy, fz, e1);
  const e2x = fy * e1[2] - fz * e1[1];
  const e2y = fz * e1[0] - fx * e1[2];
  const e2z = fx * e1[1] - fy * e1[0];
  const rollBase = rng.next() * TAU;

  for (let c = 0; c < cfg.children; c += 1) {
    const roll = rollBase + (c / cfg.children) * TAU + rng.range(-0.35, 0.35);
    const angle = cfg.spread * (0.55 + 0.65 * rng.next());
    const ca = Math.cos(angle);
    const sa = Math.sin(angle);
    const cr = Math.cos(roll);
    const sr = Math.sin(roll);
    const bx = fx * ca + (e1[0] * cr + e2x * sr) * sa;
    const by = fy * ca + (e1[1] * cr + e2y * sr) * sa;
    const bz = (fz * ca + (e1[2] * cr + e2z * sr) * sa) * cfg.flatten;
    growBranch(
      sink,
      rng,
      cfg,
      ex,
      ey,
      ez,
      bx,
      by,
      bz,
      length * cfg.lengthTaper * (0.85 + 0.3 * rng.next()),
      endRadius,
      tEnd,
      level + 1,
      maxDepth,
    );
  }
}

/**
 * Sum of `taper^i` over a chain of `depth + 1` branches.
 *
 * `Morphology.dendriteLength` is the *total* extent of the arbor, not the length
 * of its first branch, so the trunk has to be divided by this before it is grown
 * or a deep archetype like purkinje would end up three times its stated size.
 */
function taperChain(taper: number, depth: number): number {
  let total = 0;
  let segment = 1;
  for (let i = 0; i <= depth; i += 1) {
    total += segment;
    segment *= taper;
  }
  return Math.max(total, 1e-4);
}

/** Per-archetype dendritic silhouette. This is what makes the seven readable. */
function dendriteTrunks(m: Morphology, rng: Rng): TrunkSpec[] {
  const count = Math.max(1, Math.round(m.dendriteCount));
  const depth = Math.max(1, Math.round(m.dendriteDepth));
  const trunks: TrunkSpec[] = [];
  const dir = new Float32Array(3);

  switch (m.archetype) {
    case 'pyramidal': {
      // One dominant apical trunk plus a shallow basal skirt.
      normalizeInto(rng.range(-0.14, 0.14), 1, rng.range(-0.14, 0.14), dir);
      trunks.push({
        dx: dir[0],
        dy: dir[1],
        dz: dir[2],
        lengthScale: 1.55,
        radiusScale: 1.4,
        depth,
      });
      const basal = count - 1;
      for (let i = 0; i < basal; i += 1) {
        const angle = (i / Math.max(1, basal)) * TAU + rng.range(-0.3, 0.3);
        normalizeInto(
          Math.cos(angle),
          -0.55 + rng.range(-0.12, 0.12),
          Math.sin(angle),
          dir,
        );
        trunks.push({
          dx: dir[0],
          dy: dir[1],
          dz: dir[2],
          lengthScale: 0.46,
          radiusScale: 0.78,
          depth: Math.max(1, depth - 1),
        });
      }
      break;
    }
    case 'basket': {
      // Dense, short and radial; the arbor is a ball, not a tree.
      for (let i = 0; i < count; i += 1) {
        fibonacciDirection(i, count, dir);
        normalizeInto(
          dir[0] + rng.range(-0.22, 0.22),
          dir[1] + rng.range(-0.22, 0.22),
          dir[2] + rng.range(-0.22, 0.22),
          dir,
        );
        trunks.push({
          dx: dir[0],
          dy: dir[1],
          dz: dir[2],
          lengthScale: 0.8 + rng.range(0, 0.25),
          radiusScale: 0.9,
          depth,
        });
      }
      break;
    }
    case 'granule': {
      // A small upward claw; everything else about a granule cell is its axon.
      for (let i = 0; i < count; i += 1) {
        const theta = rng.range(0.12, 0.5);
        const phi = (i / count) * TAU + rng.range(-0.2, 0.2);
        normalizeInto(
          Math.sin(theta) * Math.cos(phi),
          Math.cos(theta),
          Math.sin(theta) * Math.sin(phi),
          dir,
        );
        trunks.push({
          dx: dir[0],
          dy: dir[1],
          dz: dir[2],
          lengthScale: 0.95,
          radiusScale: 0.85,
          depth,
        });
      }
      break;
    }
    case 'purkinje': {
      // Two trunks rising in one plane; `flatten` keeps the whole fan flat.
      for (let i = 0; i < count; i += 1) {
        const tilt = (i - (count - 1) * 0.5) * 0.42 + rng.range(-0.06, 0.06);
        normalizeInto(Math.sin(tilt), Math.cos(tilt), 0, dir);
        trunks.push({
          dx: dir[0],
          dy: dir[1],
          dz: dir[2],
          lengthScale: 1,
          radiusScale: 1.15,
          depth,
        });
      }
      break;
    }
    case 'stellate': {
      // Symmetric and unjittered, so the spikes read as a star.
      for (let i = 0; i < count; i += 1) {
        fibonacciDirection(i, count, dir);
        trunks.push({
          dx: dir[0],
          dy: dir[1],
          dz: dir[2],
          lengthScale: 1,
          radiusScale: 1,
          depth,
        });
      }
      break;
    }
    case 'motor': {
      // Large and radial, but biased off the axis the very long axon occupies.
      for (let i = 0; i < count; i += 1) {
        fibonacciDirection(i, count, dir);
        normalizeInto(
          dir[0] + rng.range(-0.15, 0.15),
          dir[1] * 0.7 + 0.3,
          dir[2] + rng.range(-0.15, 0.15),
          dir,
        );
        trunks.push({
          dx: dir[0],
          dy: dir[1],
          dz: dir[2],
          lengthScale: 0.95,
          radiusScale: 1.25,
          depth,
        });
      }
      break;
    }
    case 'bipolar': {
      // Exactly two opposed processes, by definition of the archetype.
      normalizeInto(rng.range(-0.05, 0.05), 1, rng.range(-0.05, 0.05), dir);
      trunks.push({
        dx: dir[0],
        dy: dir[1],
        dz: dir[2],
        lengthScale: 1.15,
        radiusScale: 1,
        depth,
      });
      trunks.push({
        dx: -dir[0],
        dy: -dir[1],
        dz: -dir[2],
        lengthScale: 0.6,
        radiusScale: 0.85,
        depth: Math.max(1, depth - 1),
      });
      break;
    }
  }
  return trunks;
}

interface DendriteStyle {
  children: number;
  curl: number;
  droop: number;
  flatten: number;
  tipRadius: number;
  trunkRadius: number;
}

const DENDRITE_STYLE: Record<MorphologyArchetype, DendriteStyle> = {
  pyramidal: { children: 2, curl: 0.5, droop: 0.02, flatten: 1, tipRadius: 0.16, trunkRadius: 0.24 },
  basket: { children: 3, curl: 0.85, droop: 0, flatten: 1, tipRadius: 0.2, trunkRadius: 0.22 },
  granule: { children: 2, curl: 0.45, droop: 0, flatten: 1, tipRadius: 0.18, trunkRadius: 0.26 },
  purkinje: { children: 2, curl: 0.55, droop: 0, flatten: 0.08, tipRadius: 0.14, trunkRadius: 0.26 },
  stellate: { children: 2, curl: 0.1, droop: 0, flatten: 1, tipRadius: 0.04, trunkRadius: 0.22 },
  motor: { children: 2, curl: 0.45, droop: 0.02, flatten: 1, tipRadius: 0.16, trunkRadius: 0.28 },
  bipolar: { children: 1, curl: 0.3, droop: 0, flatten: 1, tipRadius: 0.12, trunkRadius: 0.24 },
};

interface AxonPlan {
  dx: number;
  dy: number;
  dz: number;
  curl: number;
  droop: number;
  flatten: number;
  /** Fractions along the shaft at which a collateral leaves it. */
  collaterals: readonly number[];
  /** Split the tip into two opposed horizontal fibres, as granule cells do. */
  bifurcate: boolean;
  radius: number;
  terminalSpread: number;
}

const AXON_PLAN: Record<MorphologyArchetype, AxonPlan> = {
  pyramidal: {
    dx: 0.12,
    dy: -1,
    dz: 0.05,
    curl: 0.22,
    droop: 0.06,
    flatten: 1,
    collaterals: [0.45, 0.7],
    bifurcate: false,
    radius: 0.13,
    terminalSpread: 0.75,
  },
  basket: {
    dx: 0.1,
    dy: -1,
    dz: -0.12,
    curl: 0.4,
    droop: 0.04,
    flatten: 1,
    collaterals: [0.3, 0.55, 0.78],
    bifurcate: false,
    radius: 0.12,
    terminalSpread: 1.25,
  },
  granule: {
    dx: 0,
    dy: 1,
    dz: 0,
    curl: 0.06,
    droop: 0,
    flatten: 1,
    collaterals: [],
    bifurcate: true,
    radius: 0.075,
    terminalSpread: 0.5,
  },
  purkinje: {
    dx: 0.05,
    dy: -1,
    dz: 0,
    curl: 0.15,
    droop: 0.05,
    flatten: 0.3,
    collaterals: [0.55],
    bifurcate: false,
    radius: 0.12,
    terminalSpread: 0.6,
  },
  stellate: {
    dx: 0.25,
    dy: -1,
    dz: 0.2,
    curl: 0.35,
    droop: 0.05,
    flatten: 1,
    collaterals: [0.5],
    bifurcate: false,
    radius: 0.1,
    terminalSpread: 0.9,
  },
  motor: {
    dx: 0.02,
    dy: -1,
    dz: 0.02,
    curl: 0.05,
    droop: 0.02,
    flatten: 1,
    collaterals: [],
    bifurcate: false,
    radius: 0.17,
    terminalSpread: 0.55,
  },
  bipolar: {
    dx: 0,
    dy: -1,
    dz: 0,
    curl: 0.08,
    droop: 0,
    flatten: 1,
    collaterals: [],
    bifurcate: false,
    radius: 0.09,
    terminalSpread: 0.4,
  },
};

interface SomaShape {
  x: number;
  y: number;
  z: number;
  /** Linear radius modulation with height: negative narrows the top. */
  taper: number;
  /** Amplitude of the low-frequency deformation. */
  lumps: number;
}

const SOMA_SHAPE: Record<MorphologyArchetype, SomaShape> = {
  pyramidal: { x: 0.92, y: 1.2, z: 0.92, taper: -0.24, lumps: 0.16 },
  basket: { x: 1.04, y: 0.98, z: 1.02, taper: -0.04, lumps: 0.2 },
  granule: { x: 1, y: 1.02, z: 1, taper: -0.02, lumps: 0.13 },
  purkinje: { x: 0.94, y: 1.26, z: 0.94, taper: -0.18, lumps: 0.15 },
  stellate: { x: 1.02, y: 1, z: 1.02, taper: 0, lumps: 0.22 },
  motor: { x: 1.08, y: 0.96, z: 1.08, taper: -0.08, lumps: 0.18 },
  bipolar: { x: 0.74, y: 1.55, z: 0.74, taper: 0, lumps: 0.1 },
};

/** Smooth vertex normals for an indexed triangle soup. */
function computeIndexedNormals(positions: Float32Array, index: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < index.length; i += 3) {
    const a = index[i] * 3;
    const b = index[i + 1] * 3;
    const c = index[i + 2] * 3;
    const abx = positions[b] - positions[a];
    const aby = positions[b + 1] - positions[a + 1];
    const abz = positions[b + 2] - positions[a + 2];
    const acx = positions[c] - positions[a];
    const acy = positions[c + 1] - positions[a + 1];
    const acz = positions[c + 2] - positions[a + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    normals[a] += nx;
    normals[a + 1] += ny;
    normals[a + 2] += nz;
    normals[b] += nx;
    normals[b + 1] += ny;
    normals[b + 2] += nz;
    normals[c] += nx;
    normals[c + 1] += ny;
    normals[c + 2] += nz;
  }
  for (let i = 0; i < normals.length; i += 3) {
    const length = Math.sqrt(
      normals[i] * normals[i] + normals[i + 1] * normals[i + 1] + normals[i + 2] * normals[i + 2],
    );
    if (length > 1e-9) {
      normals[i] /= length;
      normals[i + 1] /= length;
      normals[i + 2] /= length;
    } else {
      normals[i + 1] = 1;
    }
  }
  return normals;
}

/** Soma for a specific descriptor; the exported builder is the default case. */
export function createSomaGeometry(m: Morphology, detail: number): THREE.BufferGeometry {
  const level = Math.max(0, Math.min(4, Math.floor(detail)));
  const sphere = unitSphere(level);
  const source = sphere.position;
  const shape = SOMA_SHAPE[m.archetype];
  const radius = m.somaRadius * m.scale;
  const phase = ((hashSeed(m.seed, 0x50a1) >>> 0) / 0xffffffff) * 60 - 30;
  const positions = new Float32Array(source.length);

  for (let i = 0; i < source.length; i += 3) {
    const taper = 1 + shape.taper * source[i + 1];
    const x = source[i] * shape.x * taper;
    const y = source[i + 1] * shape.y;
    const z = source[i + 2] * shape.z * taper;
    // Two octaves at very different scales: broad lobes that give the cell body
    // an asymmetric silhouette, plus a fine ripple that breaks up the terminator.
    const broad = fbm3(x * 1.35 + phase, y * 1.35 - phase, z * 1.35 + phase * 0.5, 3, 2.1, 0.55);
    const fine = simplex3(x * 4.2 - phase, y * 4.2 + phase * 0.25, z * 4.2 + phase);
    const bump = 1 + shape.lumps * broad + shape.lumps * 0.28 * fine;
    positions[i] = x * radius * bump;
    positions[i + 1] = y * radius * bump;
    positions[i + 2] = z * radius * bump;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute(
    'normal',
    new THREE.BufferAttribute(computeIndexedNormals(positions, sphere.index), 3),
  );
  geometry.setAttribute(
    'aBranchT',
    new THREE.BufferAttribute(new Float32Array(source.length / 3), 1),
  );
  geometry.setIndex(new THREE.BufferAttribute(sphere.index.slice(), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

/**
 * Organic cell body: an icosphere pushed around by low-frequency noise, never a
 * smooth ball. `detail` is the icosphere subdivision level; 2 gives 162 vertices,
 * which is enough for the silhouette to read as irregular at working zoom.
 */
export function buildSomaGeometry(detail = 2): THREE.BufferGeometry {
  return createSomaGeometry(defaultMorphology('pyramidal', 1), detail);
}

/** Recursively branched dendritic arbor as one merged geometry. */
export function buildDendriteGeometry(morphology: Morphology): THREE.BufferGeometry {
  const sink = new MeshSink();
  const rng = new Rng(hashSeed(morphology.seed, 0xdead1e));
  const style = DENDRITE_STYLE[morphology.archetype];
  const scale = morphology.scale;
  const somaRadius = morphology.somaRadius * scale;
  const trunks = dendriteTrunks(morphology, rng);

  let longest = 1;
  let deepest = 1;
  for (const trunk of trunks) {
    if (trunk.lengthScale > longest) longest = trunk.lengthScale;
    if (trunk.depth > deepest) deepest = trunk.depth;
  }
  const taper = Math.min(0.95, Math.max(0.2, morphology.dendriteTaper));
  // A zero-extent arbor is legal input; the floor keeps the reciprocal below
  // finite rather than turning every branch parameter into a NaN.
  const extent = Math.max(morphology.dendriteLength * scale, 1e-4);
  const trunkLength = extent / taperChain(taper, deepest);
  const cfg: GrowthConfig = {
    children: style.children,
    spread: morphology.dendriteSpread,
    lengthTaper: taper,
    radiusTaper: Math.pow(taper, 1.35),
    curl: style.curl,
    droop: style.droop,
    flatten: style.flatten,
    tipRadius: style.tipRadius,
    invTotal: 1 / (extent * longest),
    boutons: 0,
    boutonRadius: 0,
    budget: { left: MAX_BRANCHES },
  };

  for (const trunk of trunks) {
    const start = somaRadius * 0.82;
    growBranch(
      sink,
      rng,
      cfg,
      trunk.dx * start,
      trunk.dy * start,
      trunk.dz * start,
      trunk.dx,
      trunk.dy,
      trunk.dz,
      trunkLength * trunk.lengthScale,
      somaRadius * style.trunkRadius * trunk.radiusScale,
      0,
      0,
      trunk.depth,
    );
  }

  return sink.toGeometry();
}

/** Long, thin, sparsely branched axon closed by `axonTerminals` boutons. */
export function buildAxonGeometry(morphology: Morphology): THREE.BufferGeometry {
  const sink = new MeshSink();
  const rng = new Rng(hashSeed(morphology.seed, 0xa204));
  const plan = AXON_PLAN[morphology.archetype];
  const scale = morphology.scale;
  const somaRadius = morphology.somaRadius * scale;
  const shaftLength = morphology.axonLength * scale;
  const terminals = Math.max(1, Math.round(morphology.axonTerminals));

  const startRadius = somaRadius * plan.radius * 1.6;
  const endRadius = startRadius * 0.42;
  const boutonRadius = Math.max(endRadius * 1.9, somaRadius * 0.07);
  const twig = shaftLength * 0.12;
  const arms = plan.bifurcate ? shaftLength * 0.55 : 0;
  const invTotal = 1 / (shaftLength + arms + twig + 1e-4);

  const cfg: GrowthConfig = {
    children: 1,
    spread: 0.7,
    lengthTaper: 0.6,
    radiusTaper: 0.7,
    curl: plan.curl,
    droop: plan.droop,
    flatten: plan.flatten,
    tipRadius: 0.5,
    invTotal,
    boutons: 1,
    boutonRadius,
    budget: { left: MAX_BRANCHES },
  };

  const points = Math.max(6, Math.min(MAX_PATH_POINTS, Math.round(shaftLength * 1.1)));
  normalizeInto(plan.dx, plan.dy, plan.dz, dirScratch);
  const ax = dirScratch[0];
  const ay = dirScratch[1];
  const az = dirScratch[2];
  const attach = somaRadius * 0.85;
  walkPath(
    rng,
    ax * attach,
    ay * attach,
    az * attach,
    ax,
    ay,
    az,
    shaftLength,
    startRadius,
    endRadius,
    0,
    invTotal,
    points,
    plan.curl,
    plan.droop,
    plan.flatten,
  );
  emitTube(sink, pathScratch, radiusScratch, branchTScratch, points, 6, false);

  // The shaft's samples have to be copied out before any child branch reuses
  // the shared path scratch.
  const shaft = pathScratch.slice(0, points * 3);
  const shaftT = branchTScratch.slice(0, points);
  const shaftRadius = radiusScratch.slice(0, points);
  const tipX = shaft[(points - 1) * 3];
  const tipY = shaft[(points - 1) * 3 + 1];
  const tipZ = shaft[(points - 1) * 3 + 2];
  const fx = endDir[0];
  const fy = endDir[1];
  const fz = endDir[2];
  const tipT = shaftT[points - 1];

  for (const fraction of plan.collaterals) {
    const at = Math.max(1, Math.min(points - 2, Math.round(fraction * (points - 1))));
    const e1 = new Float32Array(3);
    perpendicularInto(ax, ay, az, e1);
    const roll = rng.next() * TAU;
    const e2x = ay * e1[2] - az * e1[1];
    const e2y = az * e1[0] - ax * e1[2];
    const e2z = ax * e1[1] - ay * e1[0];
    const cr = Math.cos(roll);
    const sr = Math.sin(roll);
    const ca = Math.cos(1.15);
    const sa = Math.sin(1.15);
    growBranch(
      sink,
      rng,
      cfg,
      shaft[at * 3],
      shaft[at * 3 + 1],
      shaft[at * 3 + 2],
      ax * ca + (e1[0] * cr + e2x * sr) * sa,
      ay * ca + (e1[1] * cr + e2y * sr) * sa,
      (az * ca + (e1[2] * cr + e2z * sr) * sa) * plan.flatten,
      shaftLength * 0.3,
      shaftRadius[at] * 0.6,
      shaftT[at],
      1,
      1,
    );
  }

  if (plan.bifurcate) {
    // The granule cell's T: two opposed parallel fibres, each carrying half the
    // terminals, which is the single most recognisable thing about its axon.
    const e1 = new Float32Array(3);
    perpendicularInto(fx, fy, fz, e1);
    const half = Math.max(1, Math.ceil(terminals / 2));
    for (let side = 0; side < 2; side += 1) {
      const sign = side === 0 ? 1 : -1;
      const bx = e1[0] * sign;
      const by = e1[1] * sign + 0.06;
      const bz = e1[2] * sign;
      walkPath(
        rng,
        tipX,
        tipY,
        tipZ,
        bx,
        by,
        bz,
        shaftLength * 0.55,
        endRadius,
        endRadius * 0.7,
        tipT,
        invTotal,
        5,
        0.08,
        0,
        1,
      );
      emitTube(sink, pathScratch, radiusScratch, branchTScratch, 5, 4, false);
      const armX = pathScratch[12];
      const armY = pathScratch[13];
      const armZ = pathScratch[14];
      const armT = branchTScratch[4];
      const gx = endDir[0];
      const gy = endDir[1];
      const gz = endDir[2];
      emitTerminals(
        sink,
        rng,
        cfg,
        armX,
        armY,
        armZ,
        gx,
        gy,
        gz,
        half,
        endRadius * 0.7,
        twig,
        plan.terminalSpread,
        armT,
      );
    }
  } else {
    emitTerminals(
      sink,
      rng,
      cfg,
      tipX,
      tipY,
      tipZ,
      fx,
      fy,
      fz,
      terminals,
      endRadius,
      twig,
      plan.terminalSpread,
      tipT,
    );
  }

  return sink.toGeometry();
}
