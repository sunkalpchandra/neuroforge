/**
 * Analytic spatial arrangements used when a population is instantiated.
 *
 * Every function writes 3 floats per element into `out` starting at `offset`,
 * matching the SoA position column layout. They are deterministic: the same
 * seed always produces the same arrangement, which is what makes a circuit
 * reproducible from its document alone.
 */

/**
 * A small deterministic hash used for jitter. Kept local so the physics package
 * has no dependency on @neuroforge/math — layout must work before the math
 * package is loaded, and duplicating twelve lines is cheaper than a cycle.
 */
function hash01(seed: number, index: number, salt: number): number {
  let h = (seed ^ (index * 0x9e3779b1) ^ (salt * 0x85ebca6b)) >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  h ^= h >>> 16;
  return h / 4294967296;
}

/** Symmetric jitter in [-amount, amount]. */
function jitter(seed: number, index: number, salt: number, amount: number): number {
  return (hash01(seed, index, salt) * 2 - 1) * amount;
}

/**
 * Point `index` of a `count`-point Fibonacci sphere. This gives a far more even
 * distribution than sampling spherical angles uniformly, which clusters at the
 * poles.
 */
export function fibonacciSphere(
  index: number,
  count: number,
  radius: number,
  out: { x: number; y: number; z: number },
): void {
  if (count <= 1) {
    out.x = 0;
    out.y = 0;
    out.z = 0;
    return;
  }
  const golden = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - (index / (count - 1)) * 2;
  const r = Math.sqrt(Math.max(0, 1 - y * y));
  const theta = golden * index;
  out.x = Math.cos(theta) * r * radius;
  out.y = y * radius;
  out.z = Math.sin(theta) * r * radius;
}

/**
 * A rectangular lattice. Extra elements beyond columns*rows*layers continue the
 * pattern rather than being dropped, so a caller that miscounts still gets every
 * neuron placed somewhere sensible.
 */
export function layoutGrid(
  count: number,
  columns: number,
  rows: number,
  layers: number,
  spacing: number,
  out: Float32Array,
  offset = 0,
): void {
  const cols = Math.max(1, Math.floor(columns));
  const rws = Math.max(1, Math.floor(rows));
  const perLayer = cols * rws;
  const halfX = ((cols - 1) * spacing) / 2;
  const halfY = ((rws - 1) * spacing) / 2;
  const effectiveLayers = Math.max(1, Math.floor(layers));
  const halfZ = ((effectiveLayers - 1) * spacing) / 2;

  for (let i = 0; i < count; i += 1) {
    const layer = Math.floor(i / perLayer);
    const withinLayer = i - layer * perLayer;
    const row = Math.floor(withinLayer / cols);
    const col = withinLayer - row * cols;
    const p = offset + i * 3;
    out[p] = col * spacing - halfX;
    out[p + 1] = row * spacing - halfY;
    out[p + 2] = layer * spacing - halfZ;
  }
}

/** Points on a sphere shell, evenly distributed, optionally perturbed. */
export function layoutSphere(
  count: number,
  radius: number,
  jitterAmount: number,
  seed: number,
  out: Float32Array,
  offset = 0,
): void {
  const scratch = { x: 0, y: 0, z: 0 };
  for (let i = 0; i < count; i += 1) {
    fibonacciSphere(i, count, radius, scratch);
    const p = offset + i * 3;
    out[p] = scratch.x + jitter(seed, i, 1, jitterAmount);
    out[p + 1] = scratch.y + jitter(seed, i, 2, jitterAmount);
    out[p + 2] = scratch.z + jitter(seed, i, 3, jitterAmount);
  }
}

/**
 * A flat disc of the given radius and thickness, lying in the XZ plane.
 * Radial positions use a sqrt distribution so density is uniform per unit area
 * rather than crowding the centre.
 */
export function layoutDisc(
  count: number,
  radius: number,
  thickness: number,
  seed: number,
  out: Float32Array,
  offset = 0,
): void {
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i += 1) {
    const r = radius * Math.sqrt((i + 0.5) / Math.max(1, count));
    const theta = golden * i;
    const p = offset + i * 3;
    out[p] = Math.cos(theta) * r;
    out[p + 1] = jitter(seed, i, 4, thickness * 0.5);
    out[p + 2] = Math.sin(theta) * r;
  }
}

/**
 * A cortical column: a vertical cylinder of the given radius and height, with
 * elements distributed evenly up the Y axis and spiralled in the XZ plane.
 */
export function layoutColumn(
  count: number,
  radius: number,
  height: number,
  seed: number,
  out: Float32Array,
  offset = 0,
): void {
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i += 1) {
    const t = count <= 1 ? 0.5 : i / (count - 1);
    const r = radius * Math.sqrt(hash01(seed, i, 5));
    const theta = golden * i;
    const p = offset + i * 3;
    out[p] = Math.cos(theta) * r;
    out[p + 1] = (t - 0.5) * height;
    out[p + 2] = Math.sin(theta) * r;
  }
}
