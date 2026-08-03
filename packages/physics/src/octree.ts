/**
 * A Barnes-Hut octree stored entirely in flat typed arrays.
 *
 * There is no node object and no allocation during a build beyond the occasional
 * capacity doubling, because this runs every relaxation step over every neuron
 * and per-node garbage would dominate the cost of the physics itself.
 *
 * Nodes are addressed by integer index. A node is a leaf when `body[node] >= 0`
 * (it holds exactly that one body) and internal when it has children. An empty
 * node has `body === EMPTY` and no children.
 */

const EMPTY = -1;
const NO_CHILD = -1;

/**
 * Coincident or near-coincident points would subdivide forever. Past this depth
 * a node stops splitting and simply accumulates mass, which is correct for the
 * force approximation and terminates.
 */
const MAX_DEPTH = 24;

export class Octree {
  private capacity: number;
  private nodeCount = 0;

  /** Child node indices, 8 per node. */
  private children: Int32Array;
  /**
   * Bitmask of which octants have been created, one bit per child. Checking a
   * single child slot is not sufficient to tell a leaf from an internal node —
   * an internal node may have children in octants 1..7 and none in octant 0.
   */
  private childMask: Uint8Array;
  /** Body index held by a leaf, or EMPTY. */
  private body: Int32Array;
  /** Accumulated mass (body count) in the subtree. */
  private mass: Float32Array;
  /** Centre of mass of the subtree. */
  private comX: Float32Array;
  private comY: Float32Array;
  private comZ: Float32Array;
  /** Node cube centre and half-extent. */
  private cx: Float32Array;
  private cy: Float32Array;
  private cz: Float32Array;
  private half: Float32Array;
  private depth: Int32Array;

  /** Explicit traversal stack, reused across queries. */
  private stack: Int32Array;

  /**
   * Positions for the build currently in progress. Insertion needs to read the
   * coordinates of a leaf's existing occupant in order to push it down a level,
   * and the centre-of-mass columns are not populated until `summarise` runs.
   */
  private buildPositions: Float32Array | null = null;

  constructor(capacity = 4096) {
    this.capacity = Math.max(64, capacity);
    this.children = new Int32Array(this.capacity * 8);
    this.childMask = new Uint8Array(this.capacity);
    this.body = new Int32Array(this.capacity);
    this.mass = new Float32Array(this.capacity);
    this.comX = new Float32Array(this.capacity);
    this.comY = new Float32Array(this.capacity);
    this.comZ = new Float32Array(this.capacity);
    this.cx = new Float32Array(this.capacity);
    this.cy = new Float32Array(this.capacity);
    this.cz = new Float32Array(this.capacity);
    this.half = new Float32Array(this.capacity);
    this.depth = new Int32Array(this.capacity);
    this.stack = new Int32Array(MAX_DEPTH * 8 + 64);
  }

  private grow(required: number): void {
    if (required <= this.capacity) return;
    let cap = this.capacity;
    while (cap < required) cap *= 2;

    const children = new Int32Array(cap * 8);
    children.set(this.children);
    this.children = children;

    const childMask = new Uint8Array(cap);
    childMask.set(this.childMask);
    this.childMask = childMask;

    const copy = <T extends Int32Array | Float32Array>(src: T): T => {
      const Ctor = src.constructor as new (n: number) => T;
      const next = new Ctor(cap);
      next.set(src as unknown as ArrayLike<number> & T);
      return next;
    };
    this.body = copy(this.body);
    this.mass = copy(this.mass);
    this.comX = copy(this.comX);
    this.comY = copy(this.comY);
    this.comZ = copy(this.comZ);
    this.cx = copy(this.cx);
    this.cy = copy(this.cy);
    this.cz = copy(this.cz);
    this.half = copy(this.half);
    this.depth = copy(this.depth);
    this.capacity = cap;
  }

  private allocate(centerX: number, centerY: number, centerZ: number, half: number, depth: number): number {
    this.grow(this.nodeCount + 1);
    const node = this.nodeCount;
    this.nodeCount += 1;
    const base = node * 8;
    for (let i = 0; i < 8; i += 1) this.children[base + i] = NO_CHILD;
    this.childMask[node] = 0;
    this.body[node] = EMPTY;
    this.mass[node] = 0;
    this.comX[node] = 0;
    this.comY[node] = 0;
    this.comZ[node] = 0;
    this.cx[node] = centerX;
    this.cy[node] = centerY;
    this.cz[node] = centerZ;
    this.half[node] = half;
    this.depth[node] = depth;
    return node;
  }

  /** Octant index for a point relative to a node centre. */
  private octant(node: number, x: number, y: number, z: number): number {
    return (x >= this.cx[node] ? 1 : 0) | (y >= this.cy[node] ? 2 : 0) | (z >= this.cz[node] ? 4 : 0);
  }

  /**
   * Rebuild the tree over `count` bodies read from a flat xyz position array.
   * Bodies with non-finite coordinates are skipped rather than poisoning the
   * bounds, which otherwise collapses the whole tree to a single degenerate node.
   */
  build(positions: Float32Array, count: number): void {
    this.nodeCount = 0;
    if (count <= 0) return;

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;

    for (let i = 0; i < count; i += 1) {
      const p = i * 3;
      const x = positions[p];
      const y = positions[p + 1];
      const z = positions[p + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }

    if (!Number.isFinite(minX)) return;

    const centerX = (minX + maxX) * 0.5;
    const centerY = (minY + maxY) * 0.5;
    const centerZ = (minZ + maxZ) * 0.5;
    // A hair of padding keeps points exactly on the boundary inside the root.
    const half = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5 + 1e-3;

    this.grow(count * 2);
    this.allocate(centerX, centerY, centerZ, half, 0);
    this.buildPositions = positions;

    for (let i = 0; i < count; i += 1) {
      const p = i * 3;
      const x = positions[p];
      const y = positions[p + 1];
      const z = positions[p + 2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      this.insert(0, i, x, y, z);
    }

    this.summarise(positions);
    this.buildPositions = null;
  }

  private insert(root: number, bodyIndex: number, x: number, y: number, z: number): void {
    let node = root;

    for (;;) {
      const occupant = this.body[node];

      if (occupant === EMPTY && this.childMask[node] === 0) {
        // Empty leaf: claim it.
        this.body[node] = bodyIndex;
        return;
      }

      if (occupant !== EMPTY) {
        // Occupied leaf. At the depth limit, stop splitting and let both bodies
        // share the node; the mass summary below still accounts for them.
        if (this.depth[node] >= MAX_DEPTH) {
          this.mass[node] += 1;
          return;
        }
        this.body[node] = EMPTY;
        this.pushDown(node, occupant);
      }

      const oct = this.octant(node, x, y, z);
      let child = this.children[node * 8 + oct];
      if (child === NO_CHILD) {
        child = this.makeChild(node, oct);
      }
      node = child;
    }
  }

  /** Move an existing leaf occupant one level down into its own octant. */
  private pushDown(node: number, occupant: number): void {
    const positions = this.buildPositions;
    if (positions === null) return;
    const p = occupant * 3;
    const ox = positions[p];
    const oy = positions[p + 1];
    const oz = positions[p + 2];
    const oct = this.octant(node, ox, oy, oz);
    let child = this.children[node * 8 + oct];
    if (child === NO_CHILD) child = this.makeChild(node, oct);
    this.insert(child, occupant, ox, oy, oz);
  }

  private makeChild(node: number, oct: number): number {
    const h = this.half[node] * 0.5;
    const nx = this.cx[node] + ((oct & 1) !== 0 ? h : -h);
    const ny = this.cy[node] + ((oct & 2) !== 0 ? h : -h);
    const nz = this.cz[node] + ((oct & 4) !== 0 ? h : -h);
    const child = this.allocate(nx, ny, nz, h, this.depth[node] + 1);
    this.children[node * 8 + oct] = child;
    this.childMask[node] |= 1 << oct;
    return child;
  }

  /**
   * Post-order accumulation of mass and centre of mass. Node indices are always
   * greater than their parent's because allocation is monotonic, so iterating
   * backwards visits every child before its parent without recursion.
   */
  private summarise(positions: Float32Array): void {
    for (let node = this.nodeCount - 1; node >= 0; node -= 1) {
      const occupant = this.body[node];
      if (occupant !== EMPTY) {
        const p = occupant * 3;
        this.mass[node] = 1;
        this.comX[node] = positions[p];
        this.comY[node] = positions[p + 1];
        this.comZ[node] = positions[p + 2];
        continue;
      }

      let m = 0;
      let sx = 0;
      let sy = 0;
      let sz = 0;
      const base = node * 8;
      for (let i = 0; i < 8; i += 1) {
        const child = this.children[base + i];
        if (child === NO_CHILD) continue;
        const cm = this.mass[child];
        if (cm === 0) continue;
        m += cm;
        sx += this.comX[child] * cm;
        sy += this.comY[child] * cm;
        sz += this.comZ[child] * cm;
      }
      this.mass[node] = m;
      if (m > 0) {
        this.comX[node] = sx / m;
        this.comY[node] = sy / m;
        this.comZ[node] = sz / m;
      }
    }
  }

  /**
   * Accumulate the repulsive force on one body into `force`.
   *
   * A subtree is treated as a single point mass when its width divided by the
   * distance to it falls below `theta`. Force follows an inverse-square law with
   * a softening term, which keeps two coincident neurons from launching each
   * other to infinity.
   */
  repulsion(
    bodyIndex: number,
    x: number,
    y: number,
    z: number,
    strength: number,
    theta: number,
    softening: number,
    force: { x: number; y: number; z: number },
  ): void {
    force.x = 0;
    force.y = 0;
    force.z = 0;
    if (this.nodeCount === 0) return;

    const stack = this.stack;
    let top = 0;
    stack[top++] = 0;
    const theta2 = theta * theta;
    const soft2 = softening * softening;

    while (top > 0) {
      const node = stack[--top];
      const m = this.mass[node];
      if (m === 0) continue;
      if (this.body[node] === bodyIndex) continue;

      const dx = x - this.comX[node];
      const dy = y - this.comY[node];
      const dz = z - this.comZ[node];
      const dist2 = dx * dx + dy * dy + dz * dz;
      const width = this.half[node] * 2;

      const isLeaf = this.body[node] !== EMPTY;
      // width^2 / dist^2 < theta^2 is the Barnes-Hut criterion without a sqrt.
      if (isLeaf || width * width < theta2 * dist2) {
        const soft = dist2 + soft2;
        const inv = strength * m / (soft * Math.sqrt(soft));
        force.x += dx * inv;
        force.y += dy * inv;
        force.z += dz * inv;
        continue;
      }

      const base = node * 8;
      for (let i = 0; i < 8; i += 1) {
        const child = this.children[base + i];
        if (child === NO_CHILD) continue;
        if (top >= stack.length) {
          const bigger = new Int32Array(stack.length * 2);
          bigger.set(stack);
          this.stack = bigger;
          return this.repulsion(bodyIndex, x, y, z, strength, theta, softening, force);
        }
        stack[top++] = child;
      }
    }
  }

  get nodes(): number {
    return this.nodeCount;
  }
}
