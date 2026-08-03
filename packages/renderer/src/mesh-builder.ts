import * as THREE from 'three';

/**
 * Triangle accumulator for procedural glyphs.
 *
 * Everything a neuron is made of — soma, every dendritic segment, every axonal
 * bouton — lands in one of these and comes out as a single indexed
 * `BufferGeometry`. That is the point: a neuron is three draw calls, not three
 * hundred, and instancing a hundred thousand of them is then only a matter of
 * per-instance attributes.
 */
export class MeshSink {
  readonly position: number[] = [];
  readonly normal: number[] = [];
  readonly branchT: number[] = [];
  readonly index: number[] = [];

  get vertexCount(): number {
    return this.position.length / 3;
  }

  pushVertex(
    x: number,
    y: number,
    z: number,
    nx: number,
    ny: number,
    nz: number,
    t: number,
  ): number {
    const at = this.position.length / 3;
    this.position.push(x, y, z);
    this.normal.push(nx, ny, nz);
    // Ordered so that a NaN parameter — which a zero-extent morphology can
    // produce — lands at the soma rather than propagating into the attribute.
    this.branchT.push(t > 0 ? (t < 1 ? t : 1) : 0);
    return at;
  }

  pushTriangle(a: number, b: number, c: number): void {
    this.index.push(a, b, c);
  }

  /**
   * The `aBranchT` attribute is the contract between geometry and material:
   * normalised distance from the soma, so a shader can drive a wave outward
   * along the arbor without knowing anything about how it was built.
   */
  toGeometry(): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(this.position, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(this.normal, 3));
    geometry.setAttribute('aBranchT', new THREE.Float32BufferAttribute(this.branchT, 1));
    geometry.setIndex(this.index);
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    return geometry;
  }
}

const EPSILON = 1e-9;

/** Scratch frame for `emitTube`; single-threaded and never live across a call. */
const frame = new Float64Array(9);

/**
 * Writes a unit vector perpendicular to (tx,ty,tz) into the first three slots of
 * `frame`. Crossing against the world axis *least* aligned with the tangent is
 * what keeps the result well conditioned for every tangent direction.
 */
function seedNormal(tx: number, ty: number, tz: number): void {
  const ax = Math.abs(tx);
  const ay = Math.abs(ty);
  const az = Math.abs(tz);
  let rx = 0;
  let ry = 0;
  let rz = 0;
  if (ax <= ay && ax <= az) rx = 1;
  else if (ay <= az) ry = 1;
  else rz = 1;
  const nx = ry * tz - rz * ty;
  const ny = rz * tx - rx * tz;
  const nz = rx * ty - ry * tx;
  const length = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  frame[0] = nx / length;
  frame[1] = ny / length;
  frame[2] = nz / length;
}

/**
 * Sweep a tapered tube along a polyline and append it to `sink`.
 *
 * Cross-sections are carried along the path by parallel transport rather than
 * rebuilt from a fixed reference axis, which is what stops a curving dendrite
 * from visibly twisting about its own centreline. Vertex normals are tilted by
 * the local taper slope, so a sharply tapering branch shades as a cone instead
 * of as a cylinder.
 */
export function emitTube(
  sink: MeshSink,
  path: Float32Array,
  radii: Float32Array,
  branchT: Float32Array,
  count: number,
  radial: number,
  capTip: boolean,
): void {
  if (count < 2 || radial < 3) return;
  const base = sink.vertexCount;

  let seeded = false;
  let tx = 0;
  let ty = 0;
  let tz = 1;
  let nx = 1;
  let ny = 0;
  let nz = 0;

  for (let i = 0; i < count; i += 1) {
    const p = i * 3;
    // The final ring has no forward neighbour, so it keeps the tangent of the
    // one before it and measures its span backwards instead.
    const forward = i < count - 1;
    const ax = forward ? path[p + 3] - path[p] : path[p] - path[p - 3];
    const ay = forward ? path[p + 4] - path[p + 1] : path[p + 1] - path[p - 2];
    const az = forward ? path[p + 5] - path[p + 2] : path[p + 2] - path[p - 1];
    const segment = Math.sqrt(ax * ax + ay * ay + az * az);
    if (forward && segment > EPSILON) {
      tx = ax / segment;
      ty = ay / segment;
      tz = az / segment;
    }

    if (!seeded) {
      seedNormal(tx, ty, tz);
      nx = frame[0];
      ny = frame[1];
      nz = frame[2];
      seeded = true;
    }

    const dot = nx * tx + ny * ty + nz * tz;
    let px = nx - tx * dot;
    let py = ny - ty * dot;
    let pz = nz - tz * dot;
    let plen = Math.sqrt(px * px + py * py + pz * pz);
    if (plen <= EPSILON) {
      // The transported normal collapsed onto the tangent, which happens only
      // when the path doubles back on itself; reseeding is the sane recovery.
      seedNormal(tx, ty, tz);
      px = frame[0];
      py = frame[1];
      pz = frame[2];
      plen = 1;
    }
    nx = px / plen;
    ny = py / plen;
    nz = pz / plen;

    const bx = ty * nz - tz * ny;
    const by = tz * nx - tx * nz;
    const bz = tx * ny - ty * nx;

    frame[0] = nx;
    frame[1] = ny;
    frame[2] = nz;
    frame[3] = bx;
    frame[4] = by;
    frame[5] = bz;
    frame[6] = tx;
    frame[7] = ty;
    frame[8] = tz;

    const radius = radii[i];
    const nextRadius = i < count - 1 ? radii[i + 1] : radius;
    const slope = segment > EPSILON ? (nextRadius - radius) / segment : 0;
    const t = branchT[i];

    for (let j = 0; j < radial; j += 1) {
      const angle = (j / radial) * Math.PI * 2;
      const ca = Math.cos(angle);
      const sa = Math.sin(angle);
      const dx = frame[0] * ca + frame[3] * sa;
      const dy = frame[1] * ca + frame[4] * sa;
      const dz = frame[2] * ca + frame[5] * sa;
      let vnx = dx - frame[6] * slope;
      let vny = dy - frame[7] * slope;
      let vnz = dz - frame[8] * slope;
      const vnl = Math.sqrt(vnx * vnx + vny * vny + vnz * vnz) || 1;
      vnx /= vnl;
      vny /= vnl;
      vnz /= vnl;
      sink.pushVertex(
        path[p] + dx * radius,
        path[p + 1] + dy * radius,
        path[p + 2] + dz * radius,
        vnx,
        vny,
        vnz,
        t,
      );
    }
  }

  for (let i = 0; i < count - 1; i += 1) {
    const ring = base + i * radial;
    const next = ring + radial;
    for (let j = 0; j < radial; j += 1) {
      const jn = (j + 1) % radial;
      sink.pushTriangle(ring + j, ring + jn, next + j);
      sink.pushTriangle(ring + jn, next + jn, next + j);
    }
  }

  if (capTip) {
    const last = (count - 1) * 3;
    const tip = radii[count - 1] * 1.8;
    const apex = sink.pushVertex(
      path[last] + tx * tip,
      path[last + 1] + ty * tip,
      path[last + 2] + tz * tip,
      tx,
      ty,
      tz,
      branchT[count - 1],
    );
    const ring = base + (count - 1) * radial;
    for (let j = 0; j < radial; j += 1) {
      sink.pushTriangle(ring + j, ring + ((j + 1) % radial), apex);
    }
  }
}

interface UnitSphere {
  position: Float32Array;
  index: Uint32Array;
}

const SPHERE_CACHE = new Map<number, UnitSphere>();

const PHI = (1 + Math.sqrt(5)) / 2;

const ICOSAHEDRON_VERTICES = [
  -1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI, 0, 0, -1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI,
  PHI, 0, -1, PHI, 0, 1, -PHI, 0, -1, -PHI, 0, 1,
];

const ICOSAHEDRON_FACES = [
  0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
  3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
];

/**
 * Unit icosphere by recursive edge midpoint subdivision, cached per detail
 * level. Icosahedral rather than UV because a UV sphere concentrates vertices
 * at its poles, and the poles are exactly where a deformed soma reads worst.
 */
export function unitSphere(detail: number): UnitSphere {
  const level = Math.max(0, Math.min(4, Math.floor(detail)));
  const cached = SPHERE_CACHE.get(level);
  if (cached) return cached;

  const positions: number[] = [];
  for (let i = 0; i < ICOSAHEDRON_VERTICES.length; i += 3) {
    const x = ICOSAHEDRON_VERTICES[i];
    const y = ICOSAHEDRON_VERTICES[i + 1];
    const z = ICOSAHEDRON_VERTICES[i + 2];
    const length = Math.sqrt(x * x + y * y + z * z);
    positions.push(x / length, y / length, z / length);
  }
  let faces = ICOSAHEDRON_FACES.slice();

  for (let step = 0; step < level; step += 1) {
    const midpoints = new Map<number, number>();
    const next: number[] = [];
    for (let f = 0; f < faces.length; f += 3) {
      const a = faces[f];
      const b = faces[f + 1];
      const c = faces[f + 2];
      const ab = midpoint(positions, midpoints, a, b);
      const bc = midpoint(positions, midpoints, b, c);
      const ca = midpoint(positions, midpoints, c, a);
      next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    faces = next;
  }

  const sphere: UnitSphere = {
    position: new Float32Array(positions),
    index: new Uint32Array(faces),
  };
  SPHERE_CACHE.set(level, sphere);
  return sphere;
}

function midpoint(
  positions: number[],
  cache: Map<number, number>,
  a: number,
  b: number,
): number {
  const lo = a < b ? a : b;
  const hi = a < b ? b : a;
  const key = lo * 0x10000 + hi;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;
  const x = (positions[a * 3] + positions[b * 3]) * 0.5;
  const y = (positions[a * 3 + 1] + positions[b * 3 + 1]) * 0.5;
  const z = (positions[a * 3 + 2] + positions[b * 3 + 2]) * 0.5;
  const length = Math.sqrt(x * x + y * y + z * z) || 1;
  const index = positions.length / 3;
  positions.push(x / length, y / length, z / length);
  cache.set(key, index);
  return index;
}

/** Append a sphere; used for axon terminal boutons. */
export function emitSphere(
  sink: MeshSink,
  cx: number,
  cy: number,
  cz: number,
  radius: number,
  detail: number,
  t: number,
): void {
  const sphere = unitSphere(detail);
  const base = sink.vertexCount;
  const source = sphere.position;
  for (let i = 0; i < source.length; i += 3) {
    sink.pushVertex(
      cx + source[i] * radius,
      cy + source[i + 1] * radius,
      cz + source[i + 2] * radius,
      source[i],
      source[i + 1],
      source[i + 2],
      t,
    );
  }
  const index = sphere.index;
  for (let i = 0; i < index.length; i += 3) {
    sink.pushTriangle(base + index[i], base + index[i + 1], base + index[i + 2]);
  }
}
