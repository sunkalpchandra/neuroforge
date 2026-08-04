/**
 * Evaluate a query AST against the running network.
 *
 * The result is a `Uint8Array` mask over neuron slots rather than a list of ids.
 * A mask is what every consumer actually wants — the renderer needs a per-slot
 * flag, the selection needs a filter, the panel needs a count — and it makes the
 * boolean operators free: an AND is a byte-wise loop over two arrays, not a set
 * intersection over fifty thousand boxed strings.
 *
 * Everything expensive is built once per evaluation and shared by every clause:
 *
 * - Adjacency. `buildPathGraph` folds the live synapse list into deduplicated
 *   forward CSR; the reverse rows are derived from it in one linear pass. A
 *   query with six connectivity clauses builds the graph once, not six times.
 * - Degrees, conductance sums and receptor masks, from a single pass over the
 *   synapse columns.
 * - Connectivity fingerprints, shared with the rest of the app through the
 *   runtime cache whenever the buffers being queried are the live ones.
 *
 * Per-cell object allocation is avoided throughout. Masks come from a pool and
 * are handed back when a clause is done with them, and string fields read into
 * one shared scratch array rather than returning one. The only per-cell
 * allocation left is the folded form of a label, which text matching memoises on
 * first use so a query pays for it once however many clauses fold that label.
 *
 * AND short-circuits twice over: the cheaper side runs first, and if it matches
 * nothing the other side never runs at all; when both do run, the intersection
 * is written into whichever mask holds fewer matches.
 */

import type { Circuit, SimulationBuffers } from '@neuroforge/shared';
import { MORPHOLOGY_ARCHETYPES } from '@neuroforge/shared';

import type { PathGraph } from '../pathfinding';
import { buildPathGraph, isSlot } from '../pathfinding';
import type { Fingerprints } from '../similarity';
import { buildFingerprints } from '../similarity';
import { getEngine, getFingerprints } from '../runtime';

import type { FieldSource, NumericField, QueryField, StringField } from './fields';
import { MAX_FIELD_VALUES, fieldPresent, findField } from './fields';
import type {
  BooleanCompareNode,
  ConnectivityNode,
  NumericCompareNode,
  PathwaysNode,
  PresenceNode,
  QueryNode,
  SimilarityNode,
  StringCompareNode,
} from './parser';

/**
 * Cosine a connectivity fingerprint must reach to count as similar wiring.
 *
 * Rows are unit vectors with non-negative components, so the cosine is a share
 * of wiring in common: 0.6 is "most of where this cell talks, that one talks
 * too". Lower and a query for one cell type returns half the network; higher and
 * it returns only exact duplicates.
 */
export const SIMILAR_CONNECTIVITY_THRESHOLD = 0.6;

/**
 * Largest normalised morphology distance still counted as the same shape.
 *
 * Distance is the RMS difference across seven descriptors, each scaled by its
 * range over the document, with a mismatched archetype contributing a full unit.
 * At 0.15 two cells must agree on archetype and sit within about a seventh of
 * the population's spread on every measurement.
 */
export const SIMILAR_SHAPE_DISTANCE = 0.15;

/**
 * Cells a similarity operator will compare against.
 *
 * `~c "Pm"` against a label carried by four thousand cells would be four
 * thousand full passes over the fingerprint matrix. The seed set is capped and
 * the result says so, which is honest about the answer being partial rather
 * than freezing the tab to be complete.
 */
export const MAX_SIMILARITY_SEEDS = 16;

export interface QueryResult {
  /** One byte per neuron slot: 1 where the cell matched. */
  mask: Uint8Array;
  /** Cells matched. */
  count: number;
  /** Slots considered, i.e. `buffers.neurons.count`. */
  total: number;
  computeMs: number;
  /**
   * Non-fatal notes: values that named no cell, seed sets that were capped.
   * A query that produced these still returned a real answer.
   */
  warnings: readonly string[];
}

/* ------------------------------------------------------------ text shapes -- */

const LETTER = /\p{L}/u;

function isDigit(char: string): boolean {
  return char >= '0' && char <= '9';
}

/**
 * Canonical form for the `{like}` operator.
 *
 * A name is reduced to its alphabetic and numeric parts: letters lowercased,
 * numbers stripped of leading zeros, punctuation and spacing dropped, and the
 * parts sorted so their order stops mattering. "Pm2", "pm-2" and "PM 02" all
 * become the same string, which is the equivalence Codex documents and the
 * reason a loose match is usable on names nobody agrees how to spell.
 */
export function likeKey(text: string): string {
  const parts: string[] = [];
  const n = text.length;
  let at = 0;
  while (at < n) {
    const char = text[at];
    if (isDigit(char)) {
      let end = at;
      while (end < n && isDigit(text[end])) end += 1;
      const digits = text.slice(at, end);
      let lead = 0;
      while (lead < digits.length - 1 && digits[lead] === '0') lead += 1;
      parts.push(digits.slice(lead));
      at = end;
      continue;
    }
    if (LETTER.test(char)) {
      let end = at;
      while (end < n && LETTER.test(text[end])) end += 1;
      parts.push(text.slice(at, end).toLowerCase());
      at = end;
      continue;
    }
    at += 1;
  }
  parts.sort();
  // Parts hold only letters and digits, so | can never occur inside one and
  // ["ab","c"] can never collide with ["a","bc"].
  return parts.join('|');
}

/* ----------------------------------------------------------------- masks --- */

/**
 * Recycles mask arrays across clauses. Evaluation is depth-first and
 * single-threaded, so a mask released by one clause is always free for the
 * next; a fifty-thousand-cell query with a dozen clauses ends up allocating
 * three or four arrays rather than a dozen.
 */
class MaskPool {
  private readonly free: Uint8Array[] = [];
  private readonly size: number;

  constructor(size: number) {
    this.size = size;
  }

  acquire(): Uint8Array {
    const mask = this.free.pop();
    if (mask === undefined) return new Uint8Array(this.size);
    mask.fill(0);
    return mask;
  }

  release(mask: Uint8Array): void {
    this.free.push(mask);
  }
}

function countMask(mask: Uint8Array, n: number): number {
  let total = 0;
  for (let i = 0; i < n; i += 1) total += mask[i];
  return total;
}

/* --------------------------------------------------------------- context --- */

/**
 * Notes gathered while evaluating. Deduplicated, because a value that names no
 * cell usually names no cell in several clauses at once and one line about it is
 * enough for the panel to show.
 */
class WarningLog {
  private readonly seen = new Set<string>();

  add(message: string): void {
    this.seen.add(message);
  }

  unresolved(value: string): void {
    this.seen.add(`no cell matches "${value}"`);
  }

  list(): readonly string[] {
    return [...this.seen];
  }
}

interface Evaluation {
  source: FieldSource;
  n: number;
  pool: MaskPool;
  scratch: string[];
  warnings: WarningLog;

  graph: PathGraph | null;
  inStart: Uint32Array | null;
  inTarget: Uint32Array | null;
  prints: Fingerprints | null;
  shape: ShapeFeatures | null;

  seeds: Map<string, Int32Array>;
  likeKeys: Map<string, string>;
  lowered: Map<string, string>;

  /** Reused visitation stamps, so set operations need no O(n) clear. */
  stamp: Int32Array;
  stampToken: number;
  counter: Uint32Array;
}

function memoLike(ev: Evaluation, text: string): string {
  const cached = ev.likeKeys.get(text);
  if (cached !== undefined) return cached;
  const key = likeKey(text);
  ev.likeKeys.set(text, key);
  return key;
}

function memoLower(ev: Evaluation, text: string): string {
  const cached = ev.lowered.get(text);
  if (cached !== undefined) return cached;
  const lower = text.toLowerCase();
  ev.lowered.set(text, lower);
  return lower;
}

function nextStamp(ev: Evaluation): number {
  if (ev.stampToken >= 0x7ffffffe) {
    ev.stamp.fill(0);
    ev.stampToken = 0;
  }
  ev.stampToken += 1;
  return ev.stampToken;
}

/**
 * Derive everything a field accessor needs that no buffer column holds.
 *
 * One pass over the synapse list. Degrees count synapses rather than distinct
 * partners, which is what the connectome metrics panel reports, so the number a
 * query filters on is the number the panel shows.
 */
function buildFieldSource(buffers: SimulationBuffers, circuit: Circuit): FieldSource {
  const neurons = buffers.neurons;
  const synapses = buffers.synapses;
  const n = neurons.count;

  const inDegree = new Uint32Array(n);
  const outDegree = new Uint32Array(n);
  const weightIn = new Float32Array(n);
  const weightOut = new Float32Array(n);
  const receptorMask = new Uint8Array(n);

  const total = synapses.count;
  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    const weight = synapses.weight[s];
    outDegree[pre] += 1;
    inDegree[post] += 1;
    if (Number.isFinite(weight)) {
      weightOut[pre] += weight;
      weightIn[post] += weight;
    }
    receptorMask[pre] |= 1 << synapses.receptor[s];
  }

  return { buffers, circuit, count: n, inDegree, outDegree, weightIn, weightOut, receptorMask };
}

/** Forward adjacency, built at most once per evaluation. */
function graphOf(ev: Evaluation): PathGraph {
  if (ev.graph === null) ev.graph = buildPathGraph(ev.source.buffers);
  return ev.graph;
}

/** Reverse rows of the same adjacency, derived from the forward CSR in one pass. */
function reverseOf(ev: Evaluation): { start: Uint32Array; target: Uint32Array } {
  if (ev.inStart !== null && ev.inTarget !== null) {
    return { start: ev.inStart, target: ev.inTarget };
  }
  const graph = graphOf(ev);
  const n = graph.n;
  const start = new Uint32Array(n + 1);
  for (let e = 0; e < graph.edges; e += 1) start[graph.outTarget[e] + 1] += 1;
  for (let i = 0; i < n; i += 1) start[i + 1] += start[i];

  const target = new Uint32Array(graph.edges);
  const cursor = new Uint32Array(n);
  for (let u = 0; u < n; u += 1) {
    const end = graph.outStart[u + 1];
    for (let e = graph.outStart[u]; e < end; e += 1) {
      const v = graph.outTarget[e];
      target[start[v] + cursor[v]] = u;
      cursor[v] += 1;
    }
  }
  ev.inStart = start;
  ev.inTarget = target;
  return { start, target };
}

/**
 * Connectivity fingerprints for the network being queried.
 *
 * When the buffers are the live ones the runtime's cache answers, so the query
 * panel, the inspector's similar-cells list and the cell-type clustering share a
 * single build per topology. A synthetic set of buffers — the self-check uses
 * one — is fingerprinted on its own.
 */
function fingerprintsOf(ev: Evaluation): Fingerprints {
  if (ev.prints !== null) return ev.prints;
  const buffers = ev.source.buffers;
  const populations = ev.source.circuit.populations.length;
  ev.prints =
    buffers === getEngine().buffers
      ? getFingerprints(populations)
      : buildFingerprints(buffers, populations);
  return ev.prints;
}

/* ---------------------------------------------------------------- shapes --- */

/** Morphology descriptors scaled onto 0..1 by their spread over the document. */
interface ShapeFeatures {
  /** Row-major `count × SHAPE_NUMERIC`, each column normalised by its range. */
  data: Float32Array;
  /** Archetype index per slot, or `NO_ARCHETYPE` when the slot has no record. */
  archetype: Uint8Array;
}

const SHAPE_NUMERIC = 6;
/** Numeric descriptors plus the archetype, which contributes one categorical unit. */
const SHAPE_DIMS = SHAPE_NUMERIC + 1;
const NO_ARCHETYPE = 0xff;

function shapeOf(ev: Evaluation): ShapeFeatures {
  if (ev.shape !== null) return ev.shape;
  const n = ev.n;
  const neurons = ev.source.circuit.neurons;
  const data = new Float32Array(n * SHAPE_NUMERIC);
  const archetype = new Uint8Array(n).fill(NO_ARCHETYPE);

  const min = new Float64Array(SHAPE_NUMERIC).fill(Number.POSITIVE_INFINITY);
  const max = new Float64Array(SHAPE_NUMERIC).fill(Number.NEGATIVE_INFINITY);

  const limit = Math.min(n, neurons.length);
  for (let slot = 0; slot < limit; slot += 1) {
    const morphology = neurons[slot].morphology;
    const row = slot * SHAPE_NUMERIC;
    data[row] = morphology.somaRadius;
    data[row + 1] = morphology.dendriteCount;
    data[row + 2] = morphology.dendriteDepth;
    data[row + 3] = morphology.dendriteLength;
    data[row + 4] = morphology.dendriteSpread;
    data[row + 5] = morphology.axonLength;
    for (let d = 0; d < SHAPE_NUMERIC; d += 1) {
      const value = data[row + d];
      if (!Number.isFinite(value)) continue;
      if (value < min[d]) min[d] = value;
      if (value > max[d]) max[d] = value;
    }
    const index = MORPHOLOGY_ARCHETYPES.indexOf(morphology.archetype);
    archetype[slot] = index >= 0 ? index : NO_ARCHETYPE;
  }

  // A descriptor every cell agrees on carries no information, so it is flattened
  // to zero rather than blowing up on a zero-width range.
  const scale = new Float64Array(SHAPE_NUMERIC);
  for (let d = 0; d < SHAPE_NUMERIC; d += 1) {
    const span = max[d] - min[d];
    scale[d] = Number.isFinite(span) && span > 0 ? 1 / span : 0;
  }
  for (let slot = 0; slot < limit; slot += 1) {
    const row = slot * SHAPE_NUMERIC;
    for (let d = 0; d < SHAPE_NUMERIC; d += 1) {
      const value = data[row + d];
      data[row + d] = Number.isFinite(value) ? (value - min[d]) * scale[d] : 0;
    }
  }

  ev.shape = { data, archetype };
  return ev.shape;
}

/** RMS distance across the descriptors; a mismatched archetype costs a full unit. */
function shapeDistance(shape: ShapeFeatures, a: number, b: number): number {
  let sum = shape.archetype[a] === shape.archetype[b] ? 0 : 1;
  const rowA = a * SHAPE_NUMERIC;
  const rowB = b * SHAPE_NUMERIC;
  for (let d = 0; d < SHAPE_NUMERIC; d += 1) {
    const delta = shape.data[rowA + d] - shape.data[rowB + d];
    sum += delta * delta;
  }
  return Math.sqrt(sum / SHAPE_DIMS);
}

/* ------------------------------------------------------------ seed lookup -- */

/**
 * Resolve a connectivity or similarity value to the cells it names.
 *
 * Three passes in order of decreasing confidence: an exact id, then an exact
 * label ignoring case, then a `{like}` match on the label. Stopping at the first
 * pass that finds anything means a precise name is never widened by a loose one,
 * while a name the user half-remembers still lands.
 */
function resolveSeeds(ev: Evaluation, value: string): Int32Array {
  const cached = ev.seeds.get(value);
  if (cached !== undefined) return cached;

  const neurons = ev.source.circuit.neurons;
  const limit = Math.min(ev.n, neurons.length);
  const hits: number[] = [];

  for (let slot = 0; slot < limit; slot += 1) {
    if (neurons[slot].id === value) hits.push(slot);
  }

  if (hits.length === 0) {
    const wanted = memoLower(ev, value);
    for (let slot = 0; slot < limit; slot += 1) {
      const label = neurons[slot].label;
      if (label !== '' && memoLower(ev, label) === wanted) hits.push(slot);
    }
  }

  if (hits.length === 0) {
    const wanted = memoLike(ev, value);
    if (wanted !== '') {
      for (let slot = 0; slot < limit; slot += 1) {
        const label = neurons[slot].label;
        if (label !== '' && memoLike(ev, label) === wanted) hits.push(slot);
      }
    }
  }

  const resolved = Int32Array.from(hits);
  ev.seeds.set(value, resolved);
  return resolved;
}

/* ------------------------------------------------------------- field help -- */

function stringFieldOf(name: string): StringField | null {
  const field = findField(name);
  return field !== null && field.type === 'string' ? field : null;
}

function numericFieldOf(name: string): NumericField | null {
  const field = findField(name);
  return field !== null && field.type === 'numeric' ? field : null;
}

function fieldOf(name: string): QueryField | null {
  return findField(name);
}

/* ------------------------------------------------------------- predicates -- */

function evalStringCompare(ev: Evaluation, node: StringCompareNode, mask: Uint8Array): void {
  const field = stringFieldOf(node.field);
  // The parser guarantees the field exists and is text; a null here can only
  // mean an AST from an older grammar, which matches nothing rather than throws.
  if (field === null) return;

  const op = node.op;
  const negated = op === 'not_equal' || op === 'not_contains' || op === 'not_in';
  const source = ev.source;
  const scratch = ev.scratch;

  const needles: string[] = [];
  for (const value of node.values) {
    needles.push(op === 'like' ? memoLike(ev, value) : memoLower(ev, value));
  }
  const count = needles.length;

  for (let slot = 0; slot < ev.n; slot += 1) {
    const values = field.readInto(source, slot, scratch);
    let hit = false;
    for (let v = 0; v < values && !hit; v += 1) {
      const raw = scratch[v];
      const subject = op === 'like' ? memoLike(ev, raw) : memoLower(ev, raw);
      for (let k = 0; k < count; k += 1) {
        const needle = needles[k];
        const matched =
          op === 'starts_with'
            ? subject.startsWith(needle)
            : op === 'ends_with'
              ? subject.endsWith(needle)
              : op === 'contains' || op === 'not_contains'
                ? subject.includes(needle)
                : subject === needle;
        if (matched) {
          hit = true;
          break;
        }
      }
    }
    mask[slot] = hit !== negated ? 1 : 0;
  }
}

function evalNumericCompare(ev: Evaluation, node: NumericCompareNode, mask: Uint8Array): void {
  const field = numericFieldOf(node.field);
  if (field === null) return;
  const source = ev.source;
  const op = node.op;
  const values = node.values;
  const first = values.length > 0 ? values[0] : Number.NaN;
  const second = values.length > 1 ? values[1] : Number.NaN;

  for (let slot = 0; slot < ev.n; slot += 1) {
    const value = field.read(source, slot);
    let hit: boolean;
    switch (op) {
      case 'gt':
        hit = value > first;
        break;
      case 'gte':
        hit = value >= first;
        break;
      case 'lt':
        hit = value < first;
        break;
      case 'lte':
        hit = value <= first;
        break;
      case 'between':
        hit = value >= first && value <= second;
        break;
      case 'equal':
        hit = value === first;
        break;
      case 'not_equal':
        hit = value !== first;
        break;
      default: {
        // in / not_in
        let found = false;
        for (let k = 0; k < values.length && !found; k += 1) found = value === values[k];
        hit = op === 'in' ? found : !found;
        break;
      }
    }
    mask[slot] = hit ? 1 : 0;
  }
}

function evalBooleanCompare(ev: Evaluation, node: BooleanCompareNode, mask: Uint8Array): void {
  const field = fieldOf(node.field);
  if (field === null || field.type !== 'boolean') return;
  const source = ev.source;
  const wanted = node.op === 'equal' ? node.value : !node.value;
  for (let slot = 0; slot < ev.n; slot += 1) {
    mask[slot] = field.read(source, slot) === wanted ? 1 : 0;
  }
}

function evalPresence(ev: Evaluation, node: PresenceNode, mask: Uint8Array): void {
  const field = fieldOf(node.field);
  if (field === null) return;
  const wanted = node.op === 'has';
  for (let slot = 0; slot < ev.n; slot += 1) {
    mask[slot] = fieldPresent(field, ev.source, slot, ev.scratch) === wanted ? 1 : 0;
  }
}

function evalConnectivity(ev: Evaluation, node: ConnectivityNode, mask: Uint8Array): void {
  // A name that resolves to nothing has already been reported by `warnNames`.
  const seeds = resolveSeeds(ev, node.value);
  if (seeds.length === 0) return;
  const graph = graphOf(ev);
  const reverse = reverseOf(ev);

  if (node.op === 'upstream' || node.op === 'downstream') {
    const start = node.op === 'upstream' ? reverse.start : graph.outStart;
    const target = node.op === 'upstream' ? reverse.target : graph.outTarget;
    for (let i = 0; i < seeds.length; i += 1) {
      const seed = seeds[i];
      if (!isSlot(graph, seed)) continue;
      const end = start[seed + 1];
      for (let e = start[seed]; e < end; e += 1) mask[target[e]] = 1;
    }
    return;
  }

  if (node.op === 'reciprocal') {
    for (let i = 0; i < seeds.length; i += 1) {
      const seed = seeds[i];
      if (!isSlot(graph, seed)) continue;
      const token = nextStamp(ev);
      const outEnd = graph.outStart[seed + 1];
      for (let e = graph.outStart[seed]; e < outEnd; e += 1) ev.stamp[graph.outTarget[e]] = token;
      const inEnd = reverse.start[seed + 1];
      for (let e = reverse.start[seed]; e < inEnd; e += 1) {
        const u = reverse.target[e];
        if (ev.stamp[u] === token) mask[u] = 1;
      }
    }
    return;
  }

  // upstream_all / downstream_all: a cell must answer every seed, so the seeds
  // are counted per candidate and only a full house survives. The rows are
  // deduplicated, so one seed can contribute at most one count per candidate.
  const start = node.op === 'upstream_all' ? reverse.start : graph.outStart;
  const target = node.op === 'upstream_all' ? reverse.target : graph.outTarget;
  ev.counter.fill(0, 0, ev.n);
  let counted = 0;
  for (let i = 0; i < seeds.length; i += 1) {
    const seed = seeds[i];
    if (!isSlot(graph, seed)) continue;
    counted += 1;
    const end = start[seed + 1];
    for (let e = start[seed]; e < end; e += 1) ev.counter[target[e]] += 1;
  }
  if (counted === 0) return;
  for (let slot = 0; slot < ev.n; slot += 1) mask[slot] = ev.counter[slot] === counted ? 1 : 0;
}

/**
 * Every cell lying on a shortest route from a source cell to a target cell.
 *
 * A single BFS from each end answers this for all routes at once: `v` is on some
 * shortest path exactly when `dist(source, v) + dist(v, target)` equals the
 * shortest distance itself. Enumerating the routes instead would be exponential
 * in a recurrent network and would answer the same question.
 *
 * When either end names several cells the search runs multi-source, so the
 * answer is the union over the globally shortest source-to-target routes.
 */
function evalPathways(ev: Evaluation, node: PathwaysNode, mask: Uint8Array): void {
  const sources = resolveSeeds(ev, node.source);
  const targets = resolveSeeds(ev, node.target);
  if (sources.length === 0 || targets.length === 0) return;

  const graph = graphOf(ev);
  const reverse = reverseOf(ev);
  const n = ev.n;

  const forward = bfsLayers(n, graph.outStart, graph.outTarget, sources, graph);
  const backward = bfsLayers(n, reverse.start, reverse.target, targets, graph);

  let best = -1;
  for (let i = 0; i < targets.length; i += 1) {
    const slot = targets[i];
    if (!isSlot(graph, slot)) continue;
    const distance = forward[slot];
    if (distance >= 0 && (best < 0 || distance < best)) best = distance;
  }
  if (best < 0) return;

  for (let slot = 0; slot < n; slot += 1) {
    const a = forward[slot];
    const b = backward[slot];
    if (a >= 0 && b >= 0 && a + b === best) mask[slot] = 1;
  }
}

/** Multi-source BFS depth per slot; -1 where a slot is unreachable. */
function bfsLayers(
  n: number,
  start: Uint32Array,
  target: Uint32Array,
  sources: Int32Array,
  graph: PathGraph,
): Int32Array {
  const depth = new Int32Array(n).fill(-1);
  const queue = new Uint32Array(n);
  let tail = 0;
  for (let i = 0; i < sources.length; i += 1) {
    const slot = sources[i];
    if (!isSlot(graph, slot) || depth[slot] !== -1) continue;
    depth[slot] = 0;
    queue[tail] = slot;
    tail += 1;
  }
  let head = 0;
  while (head < tail) {
    const u = queue[head];
    head += 1;
    const next = depth[u] + 1;
    const end = start[u + 1];
    for (let e = start[u]; e < end; e += 1) {
      const v = target[e];
      if (depth[v] !== -1) continue;
      depth[v] = next;
      queue[tail] = v;
      tail += 1;
    }
  }
  return depth;
}

function evalSimilarity(ev: Evaluation, node: SimilarityNode, mask: Uint8Array): void {
  const all = resolveSeeds(ev, node.value);
  if (all.length === 0) return;
  const seeds = all.length > MAX_SIMILARITY_SEEDS ? all.subarray(0, MAX_SIMILARITY_SEEDS) : all;
  if (seeds.length < all.length) {
    ev.warnings.add(
      `"${node.value}" names ${all.length} cells; compared against the first ${MAX_SIMILARITY_SEEDS}`,
    );
  }

  if (node.op === 'similar_shape') {
    const shape = shapeOf(ev);
    const limit = Math.min(ev.n, ev.source.circuit.neurons.length);
    for (let i = 0; i < seeds.length; i += 1) {
      const seed = seeds[i];
      if (seed >= limit || shape.archetype[seed] === NO_ARCHETYPE) continue;
      for (let slot = 0; slot < limit; slot += 1) {
        if (mask[slot] === 1 || shape.archetype[slot] === NO_ARCHETYPE) continue;
        if (shapeDistance(shape, seed, slot) <= SIMILAR_SHAPE_DISTANCE) mask[slot] = 1;
      }
    }
    return;
  }

  const prints = fingerprintsOf(ev);
  if (prints.dim === 0) return;
  const groups = prints.dim / 2;
  // The out-block leads the row and the in-block follows it, so "similar
  // outputs" reads the first half and "similar inputs" the second.
  const offset = node.op === 'similar_connectivity_upstream' ? groups : 0;
  const width = node.op === 'similar_connectivity' ? prints.dim : groups;

  const data = prints.data;
  const dim = prints.dim;
  const limit = Math.min(ev.n, prints.count);

  for (let i = 0; i < seeds.length; i += 1) {
    const seed = seeds[i];
    if (seed >= limit || prints.connected[seed] === 0) continue;
    const seedRow = seed * dim + offset;
    let seedNorm = 0;
    for (let d = 0; d < width; d += 1) seedNorm += data[seedRow + d] * data[seedRow + d];
    // A cell with no wiring in the direction being compared has no fingerprint
    // to point anywhere, so it is skipped rather than declared similar to all.
    if (seedNorm === 0) continue;
    seedNorm = Math.sqrt(seedNorm);

    for (let slot = 0; slot < limit; slot += 1) {
      if (mask[slot] === 1 || prints.connected[slot] === 0) continue;
      const row = slot * dim + offset;
      let dot = 0;
      let norm = 0;
      for (let d = 0; d < width; d += 1) {
        const value = data[row + d];
        dot += data[seedRow + d] * value;
        norm += value * value;
      }
      if (dot <= 0 || norm === 0) continue;
      if (dot / (seedNorm * Math.sqrt(norm)) >= SIMILAR_CONNECTIVITY_THRESHOLD) mask[slot] = 1;
    }
  }
}

/* ---------------------------------------------------------------- walking -- */

/**
 * Rough price of a clause, used only to decide which side of an AND runs first.
 * The ordering matters far more than the numbers: putting a field scan before a
 * fingerprint pass is what lets the fingerprint pass be skipped entirely.
 */
function costOf(node: QueryNode): number {
  switch (node.kind) {
    case 'all':
      return 0;
    case 'and':
    case 'or':
      return costOf(node.left) + costOf(node.right);
    case 'not':
      return costOf(node.operand);
    case 'connectivity':
      return 4;
    case 'pathways':
      return 8;
    case 'similarity':
      return node.op === 'similar_shape' ? 12 : 20;
    default:
      return 1;
  }
}

/**
 * Resolve every cell name the query mentions, before any clause runs.
 *
 * Whether a name matches a cell is a fact about the query, not about the branch
 * it happens to sit in — but `and` skips its expensive side entirely whenever
 * the cheap side matched nothing, so a mistyped name over there would otherwise
 * produce a silent empty answer with nothing said about why. Resolving up front
 * makes the warnings independent of evaluation order, and `resolveSeeds`
 * memoises, so a clause that does run afterwards pays nothing for this.
 */
function warnNames(ev: Evaluation, node: QueryNode): void {
  switch (node.kind) {
    case 'and':
    case 'or':
      warnNames(ev, node.left);
      warnNames(ev, node.right);
      return;
    case 'not':
      warnNames(ev, node.operand);
      return;
    case 'connectivity':
    case 'similarity':
      if (resolveSeeds(ev, node.value).length === 0) ev.warnings.unresolved(node.value);
      return;
    case 'pathways':
      if (resolveSeeds(ev, node.source).length === 0) ev.warnings.unresolved(node.source);
      if (resolveSeeds(ev, node.target).length === 0) ev.warnings.unresolved(node.target);
      return;
    default:
      return;
  }
}

function evalNode(ev: Evaluation, node: QueryNode): Uint8Array {
  switch (node.kind) {
    case 'all': {
      const mask = ev.pool.acquire();
      mask.fill(1, 0, ev.n);
      return mask;
    }
    case 'not': {
      const mask = evalNode(ev, node.operand);
      for (let i = 0; i < ev.n; i += 1) mask[i] = mask[i] === 0 ? 1 : 0;
      return mask;
    }
    case 'or': {
      const left = evalNode(ev, node.left);
      const right = evalNode(ev, node.right);
      for (let i = 0; i < ev.n; i += 1) left[i] |= right[i];
      ev.pool.release(right);
      return left;
    }
    case 'and': {
      // Cheap side first: if it matches nothing, the expensive side is never run.
      const leftFirst = costOf(node.left) <= costOf(node.right);
      const first = evalNode(ev, leftFirst ? node.left : node.right);
      const firstCount = countMask(first, ev.n);
      if (firstCount === 0) return first;

      const second = evalNode(ev, leftFirst ? node.right : node.left);
      const secondCount = countMask(second, ev.n);
      // Intersect into whichever side holds fewer matches, so the surviving
      // array is the one already close to the answer.
      if (firstCount <= secondCount) {
        for (let i = 0; i < ev.n; i += 1) if (first[i] === 1 && second[i] === 0) first[i] = 0;
        ev.pool.release(second);
        return first;
      }
      for (let i = 0; i < ev.n; i += 1) if (second[i] === 1 && first[i] === 0) second[i] = 0;
      ev.pool.release(first);
      return second;
    }
    default: {
      const mask = ev.pool.acquire();
      switch (node.kind) {
        case 'string-compare':
          evalStringCompare(ev, node, mask);
          break;
        case 'numeric-compare':
          evalNumericCompare(ev, node, mask);
          break;
        case 'boolean-compare':
          evalBooleanCompare(ev, node, mask);
          break;
        case 'presence':
          evalPresence(ev, node, mask);
          break;
        case 'connectivity':
          evalConnectivity(ev, node, mask);
          break;
        case 'pathways':
          evalPathways(ev, node, mask);
          break;
        default:
          evalSimilarity(ev, node, mask);
          break;
      }
      return mask;
    }
  }
}

/* -------------------------------------------------------------------- api -- */

/**
 * Run a parsed query over the live buffers.
 *
 * `circuit` supplies the columns the SoA does not carry — labels, ids,
 * archetypes and population names — and slot `i` is `circuit.neurons[i]`, the
 * mapping the engine establishes when it loads a document. A document that has
 * moved ahead of the engine simply reports no value for the slots that no longer
 * line up, which heals on the next load rather than throwing mid-render.
 */
export function evaluateQuery(
  ast: QueryNode,
  buffers: SimulationBuffers,
  circuit: Circuit,
): QueryResult {
  const started = performance.now();
  const n = buffers.neurons.count;

  if (n === 0) {
    return {
      mask: new Uint8Array(0),
      count: 0,
      total: 0,
      computeMs: performance.now() - started,
      warnings: [],
    };
  }

  const warnings = new WarningLog();
  const ev: Evaluation = {
    source: buildFieldSource(buffers, circuit),
    n,
    pool: new MaskPool(n),
    scratch: new Array<string>(MAX_FIELD_VALUES).fill(''),
    warnings,
    graph: null,
    inStart: null,
    inTarget: null,
    prints: null,
    shape: null,
    seeds: new Map<string, Int32Array>(),
    likeKeys: new Map<string, string>(),
    lowered: new Map<string, string>(),
    stamp: new Int32Array(n),
    stampToken: 0,
    counter: new Uint32Array(n),
  };

  warnNames(ev, ast);
  const mask = evalNode(ev, ast);
  return {
    mask,
    count: countMask(mask, n),
    total: n,
    computeMs: performance.now() - started,
    warnings: warnings.list(),
  };
}

/** Slots that matched, as a dense list. Convenient for selection and iteration. */
export function matchedSlots(result: QueryResult): number[] {
  const slots: number[] = [];
  for (let slot = 0; slot < result.total; slot += 1) {
    if (result.mask[slot] === 1) slots.push(slot);
  }
  return slots;
}
