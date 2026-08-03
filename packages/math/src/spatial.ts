import { nextPowerOfTwo } from './internal';

/**
 * Uniform spatial hash over structure-of-arrays positions.
 *
 * The grid is a flat open-addressed table of occupied cells plus a CSR-style
 * item list, all in typed arrays. There is no Map, no string key and no
 * allocation on any query path, because these queries run per frame while the
 * pointer moves over a network of tens of thousands of neurons.
 *
 * `rebuild` keeps a reference to the caller's position array rather than copying
 * it. Positions may change in place between rebuilds — the grid will simply get
 * stale — but the array itself must not be reallocated without a rebuild.
 */

/** Keeps cell indices well inside int32 even after padding arithmetic. */
const CELL_LIMIT = 1 << 28;

const MIN_TABLE_SIZE = 16;

const SQRT3 = Math.sqrt(3);

const EMPTY_POSITIONS = new Float32Array(0);

/**
 * Cell index for a world coordinate. Clamped rather than wrapped so that far
 * away or non-finite coordinates land in a corner cell instead of aliasing on
 * top of real data.
 */
function cellCoord(value: number, invCellSize: number): number {
  const c = Math.floor(value * invCellSize);
  if (c >= CELL_LIMIT) return CELL_LIMIT;
  if (c > -CELL_LIMIT) return c;
  return -CELL_LIMIT;
}

function hashCell(x: number, y: number, z: number): number {
  let h = Math.imul(x, 0x8da6b343) ^ Math.imul(y, 0xd8163841) ^ Math.imul(z, 0xcb1ab31f);
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12;
  return h;
}

/**
 * Scratch for the ray walk. Module scoped and never read outside the call that
 * writes it; JavaScript is single threaded and none of these routines re-enter.
 */
const rayOrigin = new Float64Array(3);
const rayDir = new Float64Array(3);
const ddaCell = new Int32Array(3);
const ddaStep = new Int32Array(3);
const ddaTMax = new Float64Array(3);
const ddaTDelta = new Float64Array(3);
const clipInterval = { enter: 0, exit: 0 };

/** Narrows `clipInterval` by one slab. Returns false once the interval is empty. */
function clipSlab(origin: number, dir: number, lo: number, hi: number): boolean {
  if (dir === 0) return origin >= lo && origin <= hi;
  const inv = 1 / dir;
  let t1 = (lo - origin) * inv;
  let t2 = (hi - origin) * inv;
  if (t1 > t2) {
    const swap = t1;
    t1 = t2;
    t2 = swap;
  }
  if (t1 > clipInterval.enter) clipInterval.enter = t1;
  if (t2 < clipInterval.exit) clipInterval.exit = t2;
  return clipInterval.enter <= clipInterval.exit;
}

export class SpatialHash {
  #cellSize: number;
  #invCellSize: number;
  #positions: Float32Array = EMPTY_POSITIONS;
  #count = 0;

  #tableSize = 0;
  #mask = 0;
  #keyX = new Int32Array(0);
  #keyY = new Int32Array(0);
  #keyZ = new Int32Array(0);
  #used = new Uint8Array(0);
  #bucketStart = new Uint32Array(0);
  #bucketEnd = new Uint32Array(0);
  #items = new Uint32Array(0);
  #bucketOf = new Uint32Array(0);

  #minCellX = 0;
  #minCellY = 0;
  #minCellZ = 0;
  #maxCellX = 0;
  #maxCellY = 0;
  #maxCellZ = 0;

  /** Per-query best hit. Written and read only within a single query call. */
  #bestIndex = -1;
  #bestPrimary = Infinity;
  #bestSecondary = Infinity;

  constructor(cellSize: number) {
    if (!(cellSize > 0) || !Number.isFinite(cellSize)) {
      throw new RangeError(`SpatialHash cell size must be positive and finite, got ${cellSize}`);
    }
    this.#cellSize = cellSize;
    this.#invCellSize = 1 / cellSize;
  }

  rebuild(positions: Float32Array, count: number): void {
    const n = Math.max(0, Math.min(Math.floor(count), Math.floor(positions.length / 3)));
    this.#positions = positions;
    this.#count = n;
    if (n === 0) {
      this.#tableSize = 0;
      this.#mask = 0;
      this.#minCellX = 0;
      this.#minCellY = 0;
      this.#minCellZ = 0;
      this.#maxCellX = 0;
      this.#maxCellY = 0;
      this.#maxCellZ = 0;
      return;
    }

    // Load factor <= 0.5 keeps linear probing short and guarantees the insert
    // loop always finds a free slot.
    const table = nextPowerOfTwo(Math.max(MIN_TABLE_SIZE, n * 2));
    if (table !== this.#tableSize) {
      this.#tableSize = table;
      this.#keyX = new Int32Array(table);
      this.#keyY = new Int32Array(table);
      this.#keyZ = new Int32Array(table);
      this.#used = new Uint8Array(table);
      this.#bucketStart = new Uint32Array(table);
      this.#bucketEnd = new Uint32Array(table);
    } else {
      this.#used.fill(0);
      this.#bucketStart.fill(0);
    }
    this.#mask = table - 1;
    if (this.#items.length < n) {
      this.#items = new Uint32Array(n);
      this.#bucketOf = new Uint32Array(n);
    }

    const inv = this.#invCellSize;
    const counts = this.#bucketStart;
    let minX = CELL_LIMIT;
    let minY = CELL_LIMIT;
    let minZ = CELL_LIMIT;
    let maxX = -CELL_LIMIT;
    let maxY = -CELL_LIMIT;
    let maxZ = -CELL_LIMIT;

    for (let i = 0; i < n; i += 1) {
      const p = i * 3;
      const cx = cellCoord(positions[p], inv);
      const cy = cellCoord(positions[p + 1], inv);
      const cz = cellCoord(positions[p + 2], inv);
      const bucket = this.#insert(cx, cy, cz);
      this.#bucketOf[i] = bucket;
      counts[bucket] += 1;
      if (cx < minX) minX = cx;
      if (cy < minY) minY = cy;
      if (cz < minZ) minZ = cz;
      if (cx > maxX) maxX = cx;
      if (cy > maxY) maxY = cy;
      if (cz > maxZ) maxZ = cz;
    }

    this.#minCellX = minX;
    this.#minCellY = minY;
    this.#minCellZ = minZ;
    this.#maxCellX = maxX;
    this.#maxCellY = maxY;
    this.#maxCellZ = maxZ;

    // Counting sort: turn per-bucket counts into offsets, then scatter.
    let offset = 0;
    const ends = this.#bucketEnd;
    for (let b = 0; b < table; b += 1) {
      const c = counts[b];
      counts[b] = offset;
      ends[b] = offset;
      offset += c;
    }
    const items = this.#items;
    for (let i = 0; i < n; i += 1) {
      const b = this.#bucketOf[i];
      items[ends[b]] = i;
      ends[b] += 1;
    }
  }

  /** Returns the number of hits written into `out`; stops early if `out` fills. */
  queryRadius(x: number, y: number, z: number, radius: number, out: Uint32Array): number {
    if (this.#count === 0 || out.length === 0 || !(radius >= 0)) return 0;
    const inv = this.#invCellSize;
    const r2 = radius * radius;

    const x0 = Math.max(cellCoord(x - radius, inv), this.#minCellX);
    const x1 = Math.min(cellCoord(x + radius, inv), this.#maxCellX);
    const y0 = Math.max(cellCoord(y - radius, inv), this.#minCellY);
    const y1 = Math.min(cellCoord(y + radius, inv), this.#maxCellY);
    const z0 = Math.max(cellCoord(z - radius, inv), this.#minCellZ);
    const z1 = Math.min(cellCoord(z + radius, inv), this.#maxCellZ);

    const pos = this.#positions;
    const items = this.#items;
    const starts = this.#bucketStart;
    const ends = this.#bucketEnd;
    const limit = out.length;
    let found = 0;

    for (let cx = x0; cx <= x1; cx += 1) {
      for (let cy = y0; cy <= y1; cy += 1) {
        for (let cz = z0; cz <= z1; cz += 1) {
          const bucket = this.#find(cx, cy, cz);
          if (bucket < 0) continue;
          const end = ends[bucket];
          for (let e = starts[bucket]; e < end; e += 1) {
            const index = items[e];
            const p = index * 3;
            const dx = pos[p] - x;
            const dy = pos[p + 1] - y;
            const dz = pos[p + 2] - z;
            if (dx * dx + dy * dy + dz * dz <= r2) {
              out[found] = index;
              found += 1;
              if (found >= limit) return found;
            }
          }
        }
      }
    }
    return found;
  }

  /** Slot index of the closest position within `maxRadius`, or -1. */
  nearest(x: number, y: number, z: number, maxRadius: number): number {
    if (this.#count === 0 || !(maxRadius >= 0)) return -1;
    const inv = this.#invCellSize;
    const cs = this.#cellSize;
    const cx = cellCoord(x, inv);
    const cy = cellCoord(y, inv);
    const cz = cellCoord(z, inv);

    this.#bestIndex = -1;
    this.#bestPrimary = maxRadius * maxRadius;

    const span = Math.max(
      Math.max(cx - this.#minCellX, this.#maxCellX - cx),
      Math.max(
        Math.max(cy - this.#minCellY, this.#maxCellY - cy),
        Math.max(cz - this.#minCellZ, this.#maxCellZ - cz),
      ),
    );
    const reach = Math.floor(maxRadius * inv) + 1;
    const maxRing = Math.min(span, reach);

    for (let ring = 0; ring <= maxRing; ring += 1) {
      // Anything in a cell `ring` shells out is at least (ring-1) cells away, so
      // once that floor exceeds the current best no later shell can improve it.
      if (this.#bestIndex >= 0 && ring > 1) {
        const floorDistance = (ring - 1) * cs;
        if (floorDistance * floorDistance > this.#bestPrimary) break;
      }
      for (let ix = cx - ring; ix <= cx + ring; ix += 1) {
        const onX = ix === cx - ring || ix === cx + ring;
        for (let iy = cy - ring; iy <= cy + ring; iy += 1) {
          const onShell = onX || iy === cy - ring || iy === cy + ring;
          for (let iz = cz - ring; iz <= cz + ring; iz += 1) {
            if (!onShell && iz !== cz - ring && iz !== cz + ring) continue;
            this.#testCellNearest(ix, iy, iz, x, y, z);
          }
        }
      }
    }
    return this.#bestIndex;
  }

  /**
   * First slot within `threshold` of the ray, measured perpendicular to it and
   * ordered by distance along the ray. The direction need not be normalised.
   */
  raycast(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    threshold: number,
  ): number {
    if (this.#count === 0) return -1;
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!(length > 0)) return -1;
    const rx = dx / length;
    const ry = dy / length;
    const rz = dz / length;
    const pick = threshold > 0 ? threshold : 0;
    const cs = this.#cellSize;
    const inv = this.#invCellSize;

    clipInterval.enter = 0;
    clipInterval.exit = Infinity;
    if (!clipSlab(ox, rx, this.#minCellX * cs - pick, (this.#maxCellX + 1) * cs + pick)) return -1;
    if (!clipSlab(oy, ry, this.#minCellY * cs - pick, (this.#maxCellY + 1) * cs + pick)) return -1;
    if (!clipSlab(oz, rz, this.#minCellZ * cs - pick, (this.#maxCellZ + 1) * cs + pick)) return -1;
    const tEnter = clipInterval.enter;
    const tExit = clipInterval.exit;

    this.#bestIndex = -1;
    this.#bestPrimary = Infinity;
    this.#bestSecondary = Infinity;

    rayOrigin[0] = ox;
    rayOrigin[1] = oy;
    rayOrigin[2] = oz;
    rayDir[0] = rx;
    rayDir[1] = ry;
    rayDir[2] = rz;
    for (let a = 0; a < 3; a += 1) {
      const entry = rayOrigin[a] + rayDir[a] * tEnter;
      const cell = cellCoord(entry, inv);
      const d = rayDir[a];
      ddaCell[a] = cell;
      if (d > 0) {
        ddaStep[a] = 1;
        ddaTMax[a] = tEnter + ((cell + 1) * cs - entry) / d;
        ddaTDelta[a] = cs / d;
      } else if (d < 0) {
        ddaStep[a] = -1;
        ddaTMax[a] = tEnter + (cell * cs - entry) / d;
        ddaTDelta[a] = cs / -d;
      } else {
        ddaStep[a] = 0;
        ddaTMax[a] = Infinity;
        ddaTDelta[a] = Infinity;
      }
    }

    // A hit can be up to `pick` away from the ray, so cells that far to the side
    // of the walked cell have to be tested too.
    const pad = Math.ceil(pick * inv);
    const threshold2 = pick * pick;
    const lookBehind = (pad + 1) * cs * SQRT3;
    const maxSteps =
      this.#maxCellX -
      this.#minCellX +
      (this.#maxCellY - this.#minCellY) +
      (this.#maxCellZ - this.#minCellZ) +
      6 * (pad + 1) +
      3;

    let travel = tEnter;
    for (let step = 0; step <= maxSteps; step += 1) {
      if (!(travel <= tExit)) break;
      if (this.#bestIndex >= 0 && travel - lookBehind > this.#bestPrimary) break;

      const cx = ddaCell[0];
      const cy = ddaCell[1];
      const cz = ddaCell[2];
      for (let ix = cx - pad; ix <= cx + pad; ix += 1) {
        for (let iy = cy - pad; iy <= cy + pad; iy += 1) {
          for (let iz = cz - pad; iz <= cz + pad; iz += 1) {
            this.#testCellRay(ix, iy, iz, ox, oy, oz, rx, ry, rz, threshold2);
          }
        }
      }

      let axis = 0;
      if (ddaTMax[1] < ddaTMax[axis]) axis = 1;
      if (ddaTMax[2] < ddaTMax[axis]) axis = 2;
      travel = ddaTMax[axis];
      ddaCell[axis] += ddaStep[axis];
      ddaTMax[axis] += ddaTDelta[axis];
    }

    return this.#bestIndex;
  }

  #insert(cx: number, cy: number, cz: number): number {
    const mask = this.#mask;
    const used = this.#used;
    const keyX = this.#keyX;
    const keyY = this.#keyY;
    const keyZ = this.#keyZ;
    let b = hashCell(cx, cy, cz) & mask;
    for (;;) {
      if (used[b] === 0) {
        used[b] = 1;
        keyX[b] = cx;
        keyY[b] = cy;
        keyZ[b] = cz;
        return b;
      }
      if (keyX[b] === cx && keyY[b] === cy && keyZ[b] === cz) return b;
      b = (b + 1) & mask;
    }
  }

  #find(cx: number, cy: number, cz: number): number {
    const mask = this.#mask;
    const used = this.#used;
    const keyX = this.#keyX;
    const keyY = this.#keyY;
    const keyZ = this.#keyZ;
    let b = hashCell(cx, cy, cz) & mask;
    for (;;) {
      if (used[b] === 0) return -1;
      if (keyX[b] === cx && keyY[b] === cy && keyZ[b] === cz) return b;
      b = (b + 1) & mask;
    }
  }

  #testCellNearest(cx: number, cy: number, cz: number, x: number, y: number, z: number): void {
    const bucket = this.#find(cx, cy, cz);
    if (bucket < 0) return;
    const pos = this.#positions;
    const items = this.#items;
    const end = this.#bucketEnd[bucket];
    for (let e = this.#bucketStart[bucket]; e < end; e += 1) {
      const index = items[e];
      const p = index * 3;
      const dx = pos[p] - x;
      const dy = pos[p + 1] - y;
      const dz = pos[p + 2] - z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < this.#bestPrimary || (this.#bestIndex < 0 && d2 <= this.#bestPrimary)) {
        this.#bestPrimary = d2;
        this.#bestIndex = index;
      }
    }
  }

  #testCellRay(
    cx: number,
    cy: number,
    cz: number,
    ox: number,
    oy: number,
    oz: number,
    rx: number,
    ry: number,
    rz: number,
    threshold2: number,
  ): void {
    const bucket = this.#find(cx, cy, cz);
    if (bucket < 0) return;
    const pos = this.#positions;
    const items = this.#items;
    const end = this.#bucketEnd[bucket];
    for (let e = this.#bucketStart[bucket]; e < end; e += 1) {
      const index = items[e];
      const p = index * 3;
      const vx = pos[p] - ox;
      const vy = pos[p + 1] - oy;
      const vz = pos[p + 2] - oz;
      // Points behind the origin are measured from the origin itself, so a
      // camera sitting inside a neuron still picks it.
      let along = vx * rx + vy * ry + vz * rz;
      if (along < 0) along = 0;
      const ex = vx - rx * along;
      const ey = vy - ry * along;
      const ez = vz - rz * along;
      const perp2 = ex * ex + ey * ey + ez * ez;
      if (perp2 > threshold2) continue;
      if (
        along < this.#bestPrimary ||
        (along === this.#bestPrimary && perp2 < this.#bestSecondary)
      ) {
        this.#bestIndex = index;
        this.#bestPrimary = along;
        this.#bestSecondary = perp2;
      }
    }
  }
}
