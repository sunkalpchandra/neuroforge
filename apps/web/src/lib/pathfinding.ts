/**
 * Signalling routes through the running connectome.
 *
 * Every query here answers the question a wiring diagram is actually consulted
 * for: *how does this cell reach that one*. All of it runs over the live
 * simulation buffers rather than the document, for the same reason the
 * connectome metrics do — the engine drops synapses whose endpoints were
 * deleted, and plasticity rewrites weights every step, so anything derived from
 * `circuit.synapses` would describe a network the user is no longer watching.
 *
 * The adjacency is CSR, built in one linear pass with no sort: parallel
 * synapses between the same ordered pair are folded together through a
 * per-source stamp array, which is what keeps the build O(E) rather than
 * O(E log E).
 *
 * Path enumeration is not a polynomial problem — counting the simple paths
 * between two nodes of a recurrent graph is #P-hard — so every search here
 * carries an explicit node-visit budget and reports whether it hit it. A
 * connectome panel that quietly wedges the tab is worse than one that says it
 * ran out of room.
 */

import type { ReceptorKind, SimulationBuffers } from '@neuroforge/shared';
import { RECEPTOR_FROM_CODE } from '@neuroforge/shared';

import { graphSignature } from './graph-metrics';

/**
 * Default ceiling on node visits for one query. A 100k-synapse recurrent
 * network is a few milliseconds of BFS, so this is generous for the searches
 * that terminate and decisive for the ones that would not.
 */
export const DEFAULT_VISIT_BUDGET = 400_000;

/** Longest hop horizon the census and the reachable set will accept. */
export const MAX_HOP_HORIZON = 8;

export interface PathSearchOptions {
  /**
   * Node visits allowed before the search gives up. The result reports
   * `truncated` when this bound stopped it, so a partial answer is never
   * presented as a complete one.
   */
  budget?: number;
}

/**
 * Forward CSR adjacency of the running network.
 *
 * One entry per ordered pair, not per synapse: parallel connections are
 * collapsed, their peak conductances summed — conductances in parallel add —
 * while the delay and receptor reported are those of the dominant synapse,
 * the one carrying the most of that conductance.
 */
export interface PathGraph {
  /** Topology identity, shared with the connectome metrics. */
  signature: string;
  /** Neuron slots covered; row indices run 0..n-1. */
  n: number;
  /** Distinct ordered pairs, i.e. the length of every edge column. */
  edges: number;
  /** Row offsets into the edge columns, length n + 1. */
  outStart: Uint32Array;
  outTarget: Uint32Array;
  /** Summed peak conductance of the pair (nS). */
  outWeight: Float32Array;
  /** Conduction delay of the dominant synapse (ms). */
  outDelay: Float32Array;
  /** RECEPTOR_CODE of the dominant synapse. */
  outReceptor: Uint8Array;
  /** Synapses folded into this edge. */
  outParallel: Uint32Array;
  /** Enabled synapses with live endpoints that entered the graph. */
  synapses: number;
  disabled: number;
  /** Synapses skipped because an endpoint is out of range. */
  dangling: number;
  buildMs: number;
}

export interface RouteHop {
  from: number;
  to: number;
  /** Summed peak conductance of the pair (nS). */
  weight: number;
  /** Conduction delay of the dominant synapse (ms). */
  delay: number;
  receptor: ReceptorKind;
  /** Synapses running in parallel between these two cells. */
  parallel: number;
}

export interface Route {
  /** Slot sequence, source first and target last. */
  nodes: readonly number[];
  hops: readonly RouteHop[];
  /** Synaptic steps; `nodes.length - 1`. */
  length: number;
  /** Summed conduction delay along the route (ms). */
  delay: number;
  /**
   * Σ ln(weight) over the hops. Kept in log space because a route of a few
   * dozen sub-unit conductances underflows a float64 product to exactly zero,
   * which would rank every long route identically at the bottom.
   */
  logWeight: number;
  /** Weakest conductance on the route (nS) — the bottleneck of the chain. */
  bottleneck: number;
}

export interface KShortestResult {
  /** Loopless routes, fewest hops first. */
  paths: readonly (readonly number[])[];
  requested: number;
  /** True when the visit budget stopped the search before `requested` routes. */
  truncated: boolean;
  visited: number;
  computeMs: number;
}

export interface PathCensus {
  /** Loopless paths of each length; the index is the hop count. */
  byLength: Uint32Array;
  total: number;
  maxHops: number;
  truncated: boolean;
  visited: number;
  computeMs: number;
}

export interface ReachResult {
  /** Hop depth of every slot, or -1 when it is not reached within `hops`. */
  depth: Int32Array;
  /** Cells first reached at each depth; index 0 is the source itself. */
  sizes: Uint32Array;
  /** Cells reached at or before each depth. */
  cumulative: Uint32Array;
  /** Cells reached, the source included. */
  total: number;
  hops: number;
  truncated: boolean;
  visited: number;
  computeMs: number;
}

/* ------------------------------------------------------------------ build -- */

/**
 * Fold the live synapse list into forward CSR adjacency.
 *
 * Allocates its result rather than reusing module scratch: a caller holds the
 * graph across React renders while another build may run, and a shared buffer
 * would rewrite a graph someone is still reading.
 */
export function buildPathGraph(buffers: SimulationBuffers): PathGraph {
  const started = performance.now();
  const signature = graphSignature(buffers);
  const neurons = buffers.neurons;
  const synapses = buffers.synapses;
  const n = neurons.count;
  const total = synapses.count;

  const rawStart = new Uint32Array(n + 1);
  let kept = 0;
  let disabled = 0;
  let dangling = 0;

  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) {
      disabled += 1;
      continue;
    }
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) {
      dangling += 1;
      continue;
    }
    rawStart[pre + 1] += 1;
    kept += 1;
  }

  for (let i = 0; i < n; i += 1) rawStart[i + 1] += rawStart[i];

  const rawSynapse = new Uint32Array(kept);
  const cursor = new Uint32Array(n);
  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    rawSynapse[rawStart[pre] + cursor[pre]] = s;
    cursor[pre] += 1;
  }

  const outStart = new Uint32Array(n + 1);
  const outTarget = new Uint32Array(kept);
  const outWeight = new Float32Array(kept);
  const outDelay = new Float32Array(kept);
  const outReceptor = new Uint8Array(kept);
  const outParallel = new Uint32Array(kept);
  /** Conductance of the dominant synapse per edge, which picks delay and receptor. */
  const dominant = new Float32Array(kept);

  // `mark` holds the source slot + 1 that last touched a target, so each row
  // gets a fresh namespace without an O(n) clear between rows.
  const mark = new Uint32Array(n);
  const edgeAt = new Uint32Array(n);

  let write = 0;
  for (let u = 0; u < n; u += 1) {
    outStart[u] = write;
    const token = u + 1;
    const end = rawStart[u + 1];
    for (let i = rawStart[u]; i < end; i += 1) {
      const s = rawSynapse[i];
      const v = synapses.post[s];
      const weight = synapses.weight[s];
      if (mark[v] !== token) {
        mark[v] = token;
        edgeAt[v] = write;
        outTarget[write] = v;
        outWeight[write] = weight;
        outDelay[write] = synapses.delay[s];
        outReceptor[write] = synapses.receptor[s];
        outParallel[write] = 1;
        dominant[write] = weight;
        write += 1;
        continue;
      }
      const e = edgeAt[v];
      outWeight[e] += weight;
      outParallel[e] += 1;
      if (weight > dominant[e]) {
        dominant[e] = weight;
        outDelay[e] = synapses.delay[s];
        outReceptor[e] = synapses.receptor[s];
      }
    }
  }
  outStart[n] = write;

  return {
    signature,
    n,
    edges: write,
    outStart,
    outTarget: write === kept ? outTarget : outTarget.slice(0, write),
    outWeight: write === kept ? outWeight : outWeight.slice(0, write),
    outDelay: write === kept ? outDelay : outDelay.slice(0, write),
    outReceptor: write === kept ? outReceptor : outReceptor.slice(0, write),
    outParallel: write === kept ? outParallel : outParallel.slice(0, write),
    synapses: kept,
    disabled,
    dangling,
    buildMs: performance.now() - started,
  };
}

export function isSlot(graph: PathGraph, slot: number): boolean {
  return Number.isInteger(slot) && slot >= 0 && slot < graph.n;
}

/** Edge index for an ordered pair, or -1 when the pair is not wired. */
export function edgeBetween(graph: PathGraph, from: number, to: number): number {
  if (!isSlot(graph, from)) return -1;
  const end = graph.outStart[from + 1];
  for (let e = graph.outStart[from]; e < end; e += 1) {
    if (graph.outTarget[e] === to) return e;
  }
  return -1;
}

export function outDegree(graph: PathGraph, slot: number): number {
  if (!isSlot(graph, slot)) return 0;
  return graph.outStart[slot + 1] - graph.outStart[slot];
}

/**
 * Attach weights, delays and receptors to a slot sequence.
 *
 * Returns null when a hop is not wired in this graph, which is how a route
 * computed against an older topology is rejected rather than reported with
 * invented numbers.
 */
export function describeRoute(graph: PathGraph, nodes: readonly number[]): Route | null {
  if (nodes.length === 0) return null;
  const hops: RouteHop[] = [];
  let delay = 0;
  let logWeight = 0;
  let bottleneck = Number.POSITIVE_INFINITY;

  for (let i = 0; i + 1 < nodes.length; i += 1) {
    const e = edgeBetween(graph, nodes[i], nodes[i + 1]);
    if (e < 0) return null;
    const weight = graph.outWeight[e];
    hops.push({
      from: nodes[i],
      to: nodes[i + 1],
      weight,
      delay: graph.outDelay[e],
      receptor: RECEPTOR_FROM_CODE[graph.outReceptor[e]],
      parallel: graph.outParallel[e],
    });
    delay += graph.outDelay[e];
    // Math.log(0) is -Infinity, which is the honest answer for a chain that
    // includes a synapse conducting nothing: the product really is zero.
    logWeight += Math.log(weight);
    if (weight < bottleneck) bottleneck = weight;
  }

  return {
    nodes,
    hops,
    length: hops.length,
    delay,
    logWeight: hops.length === 0 ? 0 : logWeight,
    bottleneck: hops.length === 0 ? 0 : bottleneck,
  };
}

/* ---------------------------------------------------------------- scratch -- */

let parentNode: Int32Array = new Int32Array(0);
let queue: Uint32Array = new Uint32Array(0);
let seen: Uint32Array = new Uint32Array(0);
let seenToken = 0;

let onPath: Uint8Array = new Uint8Array(0);

function ensureSearchScratch(n: number): void {
  if (parentNode.length >= n) return;
  parentNode = new Int32Array(n);
  queue = new Uint32Array(n);
  seen = new Uint32Array(n);
  // Fresh storage is zeroed, so the stamp sequence has to restart with it.
  seenToken = 0;
}

interface BfsOutcome {
  path: number[] | null;
  visited: number;
  truncated: boolean;
}

/**
 * Breadth-first search for one route, with optional node and edge exclusions.
 *
 * The exclusions are what make Yen's algorithm possible: a spur search must
 * avoid the root path's cells and the first edges of every route already
 * found. Both arrays are owned by the caller so the bans can be set and cleared
 * in O(banned) rather than O(n) per spur.
 */
function bfsPath(
  graph: PathGraph,
  source: number,
  target: number,
  banNode: Uint8Array | null,
  banEdge: Uint8Array | null,
  budget: number,
): BfsOutcome {
  ensureSearchScratch(graph.n);
  if (seenToken >= 0xfffffffe) {
    seen.fill(0);
    seenToken = 0;
  }
  seenToken += 1;
  const token = seenToken;

  let head = 0;
  let tail = 0;
  let visited = 0;

  seen[source] = token;
  parentNode[source] = -1;
  queue[tail] = source;
  tail += 1;

  while (head < tail) {
    const u = queue[head];
    head += 1;
    visited += 1;

    if (u === target) {
      const path: number[] = [];
      for (let at = u; at !== -1; at = parentNode[at]) path.push(at);
      path.reverse();
      return { path, visited, truncated: false };
    }
    if (visited >= budget) return { path: null, visited, truncated: true };

    const end = graph.outStart[u + 1];
    for (let e = graph.outStart[u]; e < end; e += 1) {
      if (banEdge !== null && banEdge[e] === 1) continue;
      const v = graph.outTarget[e];
      if (seen[v] === token) continue;
      if (banNode !== null && banNode[v] === 1) continue;
      seen[v] = token;
      parentNode[v] = u;
      queue[tail] = v;
      tail += 1;
    }
  }

  return { path: null, visited, truncated: false };
}

/* -------------------------------------------------------------- searches -- */

/**
 * Fewest-hop route from `source` to `target`, as a slot sequence including both
 * ends, or null when the target is not reachable.
 */
export function shortestPath(
  graph: PathGraph,
  source: number,
  target: number,
): number[] | null {
  if (!isSlot(graph, source) || !isSlot(graph, target)) return null;
  return bfsPath(graph, source, target, null, null, Number.POSITIVE_INFINITY).path;
}

function sameSequence(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

function sharesPrefix(path: readonly number[], root: readonly number[]): boolean {
  if (path.length < root.length) return false;
  for (let i = 0; i < root.length; i += 1) if (path[i] !== root[i]) return false;
  return true;
}

/** Order candidates by hops, then lexicographically so the result is stable. */
function compareCandidates(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) return a.length - b.length;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return 0;
}

/**
 * The `k` shortest loopless routes, by Yen's algorithm over BFS.
 *
 * Yen spends one search per node of the previous route per iteration, so the
 * work grows as k · length · (V + E). On a dense recurrent network that is
 * enough to stall a tab, which is why every spur search draws from a shared
 * visit budget and the result says when the budget, rather than the network,
 * ended the search.
 */
export function kShortestPaths(
  graph: PathGraph,
  source: number,
  target: number,
  k: number,
  options: PathSearchOptions = {},
): KShortestResult {
  const started = performance.now();
  const budget = options.budget ?? DEFAULT_VISIT_BUDGET;
  const requested = Math.max(1, Math.floor(k));
  const empty: KShortestResult = {
    paths: [],
    requested,
    truncated: false,
    visited: 0,
    computeMs: 0,
  };

  if (!isSlot(graph, source) || !isSlot(graph, target)) {
    return { ...empty, computeMs: performance.now() - started };
  }

  let visited = 0;
  let truncated = false;

  const first = bfsPath(graph, source, target, null, null, budget);
  visited += first.visited;
  if (first.path === null) {
    return {
      ...empty,
      truncated: first.truncated,
      visited,
      computeMs: performance.now() - started,
    };
  }

  const accepted: number[][] = [first.path];
  const candidates: number[][] = [];
  const banNode = new Uint8Array(graph.n);
  const banEdge = new Uint8Array(graph.edges);
  const bannedEdges: number[] = [];

  outer: while (accepted.length < requested) {
    const previous = accepted[accepted.length - 1];

    for (let i = 0; i + 1 < previous.length; i += 1) {
      const spur = previous[i];
      const root = previous.slice(0, i + 1);

      bannedEdges.length = 0;
      for (const path of accepted) {
        if (path.length <= i + 1 || !sharesPrefix(path, root)) continue;
        const e = edgeBetween(graph, path[i], path[i + 1]);
        if (e >= 0 && banEdge[e] === 0) {
          banEdge[e] = 1;
          bannedEdges.push(e);
        }
      }
      // The spur node itself stays open; everything earlier on the root is
      // closed, which is what keeps the spliced route loopless.
      for (let j = 0; j < i; j += 1) banNode[root[j]] = 1;

      const remaining = budget - visited;
      const spurResult =
        remaining > 0
          ? bfsPath(graph, spur, target, banNode, banEdge, remaining)
          : { path: null, visited: 0, truncated: true };
      visited += spurResult.visited;

      for (const e of bannedEdges) banEdge[e] = 0;
      for (let j = 0; j < i; j += 1) banNode[root[j]] = 0;

      if (spurResult.path !== null) {
        const whole = root.slice(0, i).concat(spurResult.path);
        const known =
          accepted.some((path) => sameSequence(path, whole)) ||
          candidates.some((path) => sameSequence(path, whole));
        if (!known) candidates.push(whole);
      }

      if (spurResult.truncated || visited >= budget) {
        truncated = true;
        break outer;
      }
    }

    if (candidates.length === 0) break;
    candidates.sort(compareCandidates);
    const best = candidates.shift();
    if (best === undefined) break;
    accepted.push(best);
  }

  // Falling short of `requested` without `truncated` means the network holds no
  // further loopless route, which is an answer rather than a shortfall.
  return {
    paths: accepted,
    requested,
    truncated,
    visited,
    computeMs: performance.now() - started,
  };
}

/**
 * Count the loopless routes of up to `maxHops` steps and their length spectrum.
 *
 * When `source` and `target` are the same cell this enumerates the loops
 * through it instead, which is the only sensible reading of "paths from a cell
 * to itself" and the interesting one in a recurrent network.
 *
 * Simple-path counting is #P-hard, so the depth-first walk is bounded by the
 * visit budget and says when it stopped early. The counts reported are always
 * exact for the part of the tree that was walked.
 */
export function allPathsWithinHops(
  graph: PathGraph,
  source: number,
  target: number,
  maxHops: number,
  options: PathSearchOptions = {},
): PathCensus {
  const started = performance.now();
  const budget = options.budget ?? DEFAULT_VISIT_BUDGET;
  const horizon = Math.max(1, Math.min(MAX_HOP_HORIZON, Math.floor(maxHops)));
  const byLength = new Uint32Array(horizon + 1);

  if (!isSlot(graph, source) || !isSlot(graph, target)) {
    return {
      byLength,
      total: 0,
      maxHops: horizon,
      truncated: false,
      visited: 0,
      computeMs: performance.now() - started,
    };
  }

  if (onPath.length < graph.n) onPath = new Uint8Array(graph.n);

  let total = 0;
  let visited = 0;
  let truncated = false;

  const walk = (u: number, depth: number): void => {
    visited += 1;
    if (visited >= budget) {
      truncated = true;
      return;
    }
    if (u === target && depth > 0) {
      byLength[depth] += 1;
      total += 1;
      return;
    }
    if (depth === horizon) return;

    const end = graph.outStart[u + 1];
    for (let e = graph.outStart[u]; e < end; e += 1) {
      const v = graph.outTarget[e];
      // The target is reachable even when it is the source of a loop census;
      // every other cell may appear at most once, which is what "loopless"
      // means here.
      if (onPath[v] === 1 && v !== target) continue;
      onPath[v] = 1;
      walk(v, depth + 1);
      onPath[v] = 0;
      if (truncated) return;
    }
  };

  onPath[source] = 1;
  walk(source, 0);
  onPath[source] = 0;

  return {
    byLength,
    total,
    maxHops: horizon,
    truncated,
    visited,
    computeMs: performance.now() - started,
  };
}

/**
 * The forward reachable set of a cell, layer by layer — its sphere of
 * influence. Depth 0 is the cell itself.
 */
export function reachableWithin(
  graph: PathGraph,
  source: number,
  hops: number,
  options: PathSearchOptions = {},
): ReachResult {
  const started = performance.now();
  const budget = options.budget ?? DEFAULT_VISIT_BUDGET;
  const horizon = Math.max(1, Math.min(MAX_HOP_HORIZON, Math.floor(hops)));
  const depth = new Int32Array(graph.n).fill(-1);
  const sizes = new Uint32Array(horizon + 1);
  const cumulative = new Uint32Array(horizon + 1);

  if (!isSlot(graph, source)) {
    return {
      depth,
      sizes,
      cumulative,
      total: 0,
      hops: horizon,
      truncated: false,
      visited: 0,
      computeMs: performance.now() - started,
    };
  }

  let current = new Uint32Array(graph.n);
  let buffer = new Uint32Array(graph.n);
  let frontierLength = 1;
  current[0] = source;
  depth[source] = 0;
  sizes[0] = 1;

  let total = 1;
  let visited = 1;
  let truncated = false;

  for (let level = 1; level <= horizon && frontierLength > 0 && !truncated; level += 1) {
    let nextLength = 0;
    for (let i = 0; i < frontierLength; i += 1) {
      const u = current[i];
      const end = graph.outStart[u + 1];
      for (let e = graph.outStart[u]; e < end; e += 1) {
        visited += 1;
        const v = graph.outTarget[e];
        if (depth[v] !== -1) continue;
        depth[v] = level;
        buffer[nextLength] = v;
        nextLength += 1;
        total += 1;
      }
      if (visited >= budget) {
        // The layer is left incomplete rather than the next one being built on
        // a partial frontier, which would misreport later depths as small.
        truncated = true;
        break;
      }
    }
    sizes[level] = nextLength;
    const swap = current;
    current = buffer;
    buffer = swap;
    frontierLength = nextLength;
  }

  let running = 0;
  for (let level = 0; level <= horizon; level += 1) {
    running += sizes[level];
    cumulative[level] = running;
  }

  return {
    depth,
    sizes,
    cumulative,
    total,
    hops: horizon,
    truncated,
    visited,
    computeMs: performance.now() - started,
  };
}

/** Slots reached at or before `hops`, the source included. */
export function reachedSlots(reach: ReachResult, hops: number): number[] {
  const limit = Math.max(0, Math.min(reach.hops, Math.floor(hops)));
  const slots: number[] = [];
  for (let slot = 0; slot < reach.depth.length; slot += 1) {
    const at = reach.depth[slot];
    if (at >= 0 && at <= limit) slots.push(slot);
  }
  return slots;
}
