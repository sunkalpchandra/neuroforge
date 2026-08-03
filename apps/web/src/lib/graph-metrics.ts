/**
 * Connectome statistics over the live simulation buffers.
 *
 * These metrics describe the network that is actually running, not the document
 * that produced it: the engine drops synapses whose endpoints were deleted, and
 * plasticity rewrites weights every step, so anything derived from
 * `circuit.synapses` would drift away from what the user is watching.
 *
 * Everything is computed from CSR adjacency built once per call. A nested scan
 * over the synapse list would be O(V·E) — at 100k synapses that is minutes, not
 * milliseconds. All working storage lives at module scope and is only ever
 * reallocated when it has to grow, so a recompute allocates nothing beyond the
 * result object and its histograms.
 */

import type { ReceptorKind, SimulationBuffers } from '@neuroforge/shared';
import { RECEPTOR_FROM_CODE } from '@neuroforge/shared';

/** Mean synaptic weight and share of the wiring for one receptor kind. */
export interface ReceptorStat {
  receptor: ReceptorKind;
  count: number;
  /** Mean peak conductance across synapses of this kind (nS). */
  meanWeight: number;
  /** Summed peak conductance (nS). */
  totalConductance: number;
}

/** One of the most highly connected cells in the network. */
export interface HubNeuron {
  slot: number;
  /** Morphology seed — the key `identityColor` is derived from. */
  seed: number;
  inDegree: number;
  outDegree: number;
  degree: number;
  inhibitory: boolean;
}

export interface GraphMetrics {
  /** Identifies the topology this was computed from; see `graphSignature`. */
  signature: string;
  computeMs: number;

  neurons: number;
  disabledNeurons: number;
  /** Synapses that participate in the graph: enabled and with valid endpoints. */
  synapses: number;
  disabledSynapses: number;
  selfLoops: number;
  /** Distinct ordered pairs, self-loops excluded. */
  uniqueEdges: number;
  /** uniqueEdges / (n·(n−1)). */
  density: number;

  /** Mean in-degree, which for a directed graph equals the mean out-degree. */
  meanDegree: number;
  maxInDegree: number;
  maxOutDegree: number;
  maxDegree: number;
  /** Count of neurons at each in-degree; index is the degree itself. */
  inHistogram: Uint32Array;
  outHistogram: Uint32Array;

  components: number;
  largestComponent: number;
  isolated: number;

  /** Share of distinct connections that are answered by one in the other direction. */
  reciprocity: number;
  /** Mean local clustering coefficient of the underlying undirected graph. */
  clustering: number;
  /** False when `clustering` is a sampled estimate rather than a full pass. */
  clusteringExact: boolean;
  clusteringSamples: number;

  excitatoryNeurons: number;
  inhibitoryNeurons: number;
  /** Classified by the polarity of the presynaptic cell. */
  excitatorySynapses: number;
  inhibitorySynapses: number;
  excitatoryConductance: number;
  inhibitoryConductance: number;
  meanWeight: number;

  receptors: readonly ReceptorStat[];
  hubs: readonly HubNeuron[];
}

/* ----------------------------------------------------------------- scratch -- */

let outStart: Uint32Array = new Uint32Array(1);
let outIdx: Uint32Array = new Uint32Array(1);
let inStart: Uint32Array = new Uint32Array(1);
let inIdx: Uint32Array = new Uint32Array(1);
let cursor: Uint32Array = new Uint32Array(1);

let simpleOutStart: Uint32Array = new Uint32Array(1);
let simpleOutIdx: Uint32Array = new Uint32Array(1);
let simpleInStart: Uint32Array = new Uint32Array(1);
let simpleInIdx: Uint32Array = new Uint32Array(1);
let undStart: Uint32Array = new Uint32Array(1);
let undIdx: Uint32Array = new Uint32Array(1);

let parents: Int32Array = new Int32Array(1);
let componentSize: Uint32Array = new Uint32Array(1);
let marks: Uint32Array = new Uint32Array(1);
let markToken = 0;

let hubSlots: Int32Array = new Int32Array(1);
let hubDegrees: Int32Array = new Int32Array(1);

const RECEPTOR_KIND_COUNT = RECEPTOR_FROM_CODE.length;
const receptorCounts = new Uint32Array(RECEPTOR_KIND_COUNT);
const receptorWeights = new Float64Array(RECEPTOR_KIND_COUNT);

/**
 * Ceiling on the triangle-counting work for the clustering coefficient. Local
 * clustering costs the square of a node's degree, and a connectome's degree
 * distribution is heavy-tailed, so an exact pass over a hub-dominated network
 * would block the main thread for seconds. Past this budget the coefficient is
 * estimated over a deterministic sample instead.
 */
const CLUSTERING_BUDGET = 12_000_000;

const DEFAULT_HUB_COUNT = 12;

/** Golden-ratio stride: visits nodes in an order uncorrelated with population blocks. */
const PHI_CONJUGATE = 0.618033988749895;

function ensureU32(array: Uint32Array, length: number): Uint32Array {
  return array.length >= length ? array : new Uint32Array(length);
}

function ensureI32(array: Int32Array, length: number): Int32Array {
  return array.length >= length ? array : new Int32Array(length);
}

function gcd(a: number, b: number): number {
  let x = a;
  let y = b;
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x;
}

/** Union-find root with path halving. */
function findRoot(node: number): number {
  let i = node;
  while (parents[i] !== i) {
    parents[i] = parents[parents[i]];
    i = parents[i];
  }
  return i;
}

/** Membership test in a sorted CSR row. */
function rowContains(values: Uint32Array, from: number, to: number, target: number): boolean {
  let lo = from;
  let hi = to - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const value = values[mid];
    if (value === target) return true;
    if (value < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return false;
}

/**
 * Local clustering coefficient of one node in the undirected simple graph.
 *
 * Neighbours are stamped with a monotonic token rather than cleared between
 * nodes, which is what keeps this allocation-free and removes the O(V) clear
 * that would otherwise dominate for low-degree nodes.
 */
function localClustering(node: number): number {
  const begin = undStart[node];
  const end = undStart[node + 1];
  const k = end - begin;
  if (k < 2) return 0;

  markToken += 1;
  for (let i = begin; i < end; i += 1) marks[undIdx[i]] = markToken;

  let links = 0;
  for (let i = begin; i < end; i += 1) {
    const neighbour = undIdx[i];
    const nb = undStart[neighbour];
    const ne = undStart[neighbour + 1];
    for (let j = nb; j < ne; j += 1) {
      const other = undIdx[j];
      if (other !== node && marks[other] === markToken) links += 1;
    }
  }
  // `links` counts each triangle edge from both endpoints, so it is already 2T.
  return links / (k * (k - 1));
}

/* ------------------------------------------------------------------- api ---- */

/**
 * Cheap identity for the topology. Only structural edits change it, so a caller
 * can poll this to decide whether a recompute is worth doing.
 */
export function graphSignature(buffers: SimulationBuffers): string {
  return `${buffers.neurons.count}:${buffers.synapses.count}`;
}

function emptyMetrics(signature: string, computeMs: number, neurons: number): GraphMetrics {
  return {
    signature,
    computeMs,
    neurons,
    disabledNeurons: 0,
    synapses: 0,
    disabledSynapses: 0,
    selfLoops: 0,
    uniqueEdges: 0,
    density: 0,
    meanDegree: 0,
    maxInDegree: 0,
    maxOutDegree: 0,
    maxDegree: 0,
    inHistogram: new Uint32Array(1),
    outHistogram: new Uint32Array(1),
    components: 0,
    largestComponent: 0,
    isolated: neurons,
    reciprocity: 0,
    clustering: 0,
    clusteringExact: true,
    clusteringSamples: 0,
    excitatoryNeurons: 0,
    inhibitoryNeurons: 0,
    excitatorySynapses: 0,
    inhibitorySynapses: 0,
    excitatoryConductance: 0,
    inhibitoryConductance: 0,
    meanWeight: 0,
    receptors: [],
    hubs: [],
  };
}

/**
 * Compute every metric in one pass set. Synchronous and blocking: at 100k
 * synapses this is a few tens of milliseconds, which is fine on demand and
 * unacceptable per frame.
 */
export function computeGraphMetrics(
  buffers: SimulationBuffers,
  hubCount = DEFAULT_HUB_COUNT,
): GraphMetrics {
  const started = performance.now();
  const signature = graphSignature(buffers);
  const neurons = buffers.neurons;
  const synapses = buffers.synapses;
  const n = neurons.count;
  const total = synapses.count;

  if (n === 0) return emptyMetrics(signature, performance.now() - started, 0);

  let disabledNeurons = 0;
  let excitatoryNeurons = 0;
  let inhibitoryNeurons = 0;
  for (let i = 0; i < n; i += 1) {
    if (neurons.enabled[i] === 0) disabledNeurons += 1;
    if (neurons.polarity[i] === 1) inhibitoryNeurons += 1;
    else excitatoryNeurons += 1;
  }

  /* -- counting pass ------------------------------------------------------- */

  outStart = ensureU32(outStart, n + 1);
  inStart = ensureU32(inStart, n + 1);
  outStart.fill(0, 0, n + 1);
  inStart.fill(0, 0, n + 1);

  receptorCounts.fill(0);
  receptorWeights.fill(0);

  let kept = 0;
  let disabledSynapses = 0;
  let selfLoops = 0;
  let weightSum = 0;
  let excitatorySynapses = 0;
  let inhibitorySynapses = 0;
  let excitatoryConductance = 0;
  let inhibitoryConductance = 0;

  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) {
      disabledSynapses += 1;
      continue;
    }
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;

    if (pre === post) selfLoops += 1;
    outStart[pre + 1] += 1;
    inStart[post + 1] += 1;
    kept += 1;

    const weight = synapses.weight[s];
    weightSum += weight;

    const receptor = synapses.receptor[s];
    if (receptor < RECEPTOR_KIND_COUNT) {
      receptorCounts[receptor] += 1;
      receptorWeights[receptor] += weight;
    }

    if (neurons.polarity[pre] === 1) {
      inhibitorySynapses += 1;
      inhibitoryConductance += weight;
    } else {
      excitatorySynapses += 1;
      excitatoryConductance += weight;
    }
  }

  if (kept === 0) {
    const empty = emptyMetrics(signature, performance.now() - started, n);
    // Every neuron is its own singleton component and sits at degree zero.
    empty.inHistogram[0] = n;
    empty.outHistogram[0] = n;
    return {
      ...empty,
      disabledNeurons,
      disabledSynapses,
      excitatoryNeurons,
      inhibitoryNeurons,
      components: n,
      largestComponent: 1,
    };
  }

  for (let i = 0; i < n; i += 1) {
    outStart[i + 1] += outStart[i];
    inStart[i + 1] += inStart[i];
  }

  /* -- fill CSR ------------------------------------------------------------ */

  outIdx = ensureU32(outIdx, kept);
  inIdx = ensureU32(inIdx, kept);
  cursor = ensureU32(cursor, n);

  cursor.fill(0, 0, n);
  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    outIdx[outStart[pre] + cursor[pre]] = post;
    cursor[pre] += 1;
  }

  cursor.fill(0, 0, n);
  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    inIdx[inStart[post] + cursor[post]] = pre;
    cursor[post] += 1;
  }

  /* -- degrees, histograms and hubs ---------------------------------------- */

  let maxInDegree = 0;
  let maxOutDegree = 0;
  let maxDegree = 0;
  for (let u = 0; u < n; u += 1) {
    const outDeg = outStart[u + 1] - outStart[u];
    const inDeg = inStart[u + 1] - inStart[u];
    if (outDeg > maxOutDegree) maxOutDegree = outDeg;
    if (inDeg > maxInDegree) maxInDegree = inDeg;
    const degree = outDeg + inDeg;
    if (degree > maxDegree) maxDegree = degree;
  }

  const inHistogram = new Uint32Array(maxInDegree + 1);
  const outHistogram = new Uint32Array(maxOutDegree + 1);

  const wanted = Math.max(1, Math.min(hubCount, n));
  hubSlots = ensureI32(hubSlots, wanted);
  hubDegrees = ensureI32(hubDegrees, wanted);
  let hubLength = 0;

  for (let u = 0; u < n; u += 1) {
    const outDeg = outStart[u + 1] - outStart[u];
    const inDeg = inStart[u + 1] - inStart[u];
    inHistogram[inDeg] += 1;
    outHistogram[outDeg] += 1;

    const degree = outDeg + inDeg;
    if (degree === 0) continue;
    if (hubLength === wanted && degree <= hubDegrees[hubLength - 1]) continue;

    // Insertion into a fixed-size descending list; `wanted` is a dozen, so this
    // is cheaper than sorting every neuron by degree.
    let at = hubLength < wanted ? hubLength : wanted - 1;
    if (hubLength < wanted) hubLength += 1;
    while (at > 0 && hubDegrees[at - 1] < degree) {
      hubDegrees[at] = hubDegrees[at - 1];
      hubSlots[at] = hubSlots[at - 1];
      at -= 1;
    }
    hubDegrees[at] = degree;
    hubSlots[at] = u;
  }

  const hubs: HubNeuron[] = [];
  for (let i = 0; i < hubLength; i += 1) {
    const slot = hubSlots[i];
    hubs.push({
      slot,
      seed: neurons.seed[slot],
      inDegree: inStart[slot + 1] - inStart[slot],
      outDegree: outStart[slot + 1] - outStart[slot],
      degree: hubDegrees[i],
      inhibitory: neurons.polarity[slot] === 1,
    });
  }

  /* -- simple (deduplicated) adjacency ------------------------------------- */

  for (let u = 0; u < n; u += 1) {
    const begin = outStart[u];
    const end = outStart[u + 1];
    if (end - begin > 1) outIdx.subarray(begin, end).sort();
    const inBegin = inStart[u];
    const inEnd = inStart[u + 1];
    if (inEnd - inBegin > 1) inIdx.subarray(inBegin, inEnd).sort();
  }

  simpleOutStart = ensureU32(simpleOutStart, n + 1);
  simpleInStart = ensureU32(simpleInStart, n + 1);
  simpleOutIdx = ensureU32(simpleOutIdx, kept);
  simpleInIdx = ensureU32(simpleInIdx, kept);

  let write = 0;
  for (let u = 0; u < n; u += 1) {
    simpleOutStart[u] = write;
    const end = outStart[u + 1];
    let previous = -1;
    for (let i = outStart[u]; i < end; i += 1) {
      const v = outIdx[i];
      if (v === u || v === previous) continue;
      simpleOutIdx[write] = v;
      write += 1;
      previous = v;
    }
  }
  simpleOutStart[n] = write;
  const uniqueEdges = write;

  write = 0;
  for (let u = 0; u < n; u += 1) {
    simpleInStart[u] = write;
    const end = inStart[u + 1];
    let previous = -1;
    for (let i = inStart[u]; i < end; i += 1) {
      const v = inIdx[i];
      if (v === u || v === previous) continue;
      simpleInIdx[write] = v;
      write += 1;
      previous = v;
    }
  }
  simpleInStart[n] = write;

  /* -- reciprocity --------------------------------------------------------- */

  let reciprocated = 0;
  for (let u = 0; u < n; u += 1) {
    const end = simpleOutStart[u + 1];
    for (let i = simpleOutStart[u]; i < end; i += 1) {
      const v = simpleOutIdx[i];
      if (rowContains(simpleOutIdx, simpleOutStart[v], simpleOutStart[v + 1], u)) {
        reciprocated += 1;
      }
    }
  }
  const reciprocity = uniqueEdges > 0 ? reciprocated / uniqueEdges : 0;

  /* -- undirected view ----------------------------------------------------- */

  undStart = ensureU32(undStart, n + 1);
  undIdx = ensureU32(undIdx, uniqueEdges + simpleInStart[n]);

  write = 0;
  for (let u = 0; u < n; u += 1) {
    undStart[u] = write;
    let i = simpleOutStart[u];
    const iEnd = simpleOutStart[u + 1];
    let j = simpleInStart[u];
    const jEnd = simpleInStart[u + 1];
    // Both rows are sorted and self-free, so the union is a linear merge.
    while (i < iEnd || j < jEnd) {
      const a = i < iEnd ? simpleOutIdx[i] : 0xffffffff;
      const b = j < jEnd ? simpleInIdx[j] : 0xffffffff;
      const v = a <= b ? a : b;
      if (a === v) i += 1;
      if (b === v) j += 1;
      undIdx[write] = v;
      write += 1;
    }
  }
  undStart[n] = write;

  /* -- weakly connected components ----------------------------------------- */

  parents = ensureI32(parents, n);
  componentSize = ensureU32(componentSize, n);
  for (let u = 0; u < n; u += 1) parents[u] = u;

  for (let u = 0; u < n; u += 1) {
    const end = simpleOutStart[u + 1];
    for (let i = simpleOutStart[u]; i < end; i += 1) {
      const a = findRoot(u);
      const b = findRoot(simpleOutIdx[i]);
      if (a !== b) parents[b] = a;
    }
  }

  componentSize.fill(0, 0, n);
  for (let u = 0; u < n; u += 1) componentSize[findRoot(u)] += 1;

  let components = 0;
  let largestComponent = 0;
  let isolated = 0;
  for (let u = 0; u < n; u += 1) {
    const size = componentSize[u];
    if (size > 0) {
      components += 1;
      if (size > largestComponent) largestComponent = size;
    }
    if (undStart[u + 1] === undStart[u]) isolated += 1;
  }

  /* -- clustering coefficient ---------------------------------------------- */

  marks = ensureU32(marks, n);
  if (markToken > 0xffffffff - n - 1) {
    marks.fill(0);
    markToken = 0;
  }

  let workEstimate = 0;
  for (let u = 0; u < n; u += 1) {
    const k = undStart[u + 1] - undStart[u];
    workEstimate += k * k;
  }

  let clusteringSum = 0;
  let clusteringSamples = 0;
  let clusteringExact = true;

  if (workEstimate <= CLUSTERING_BUDGET) {
    for (let u = 0; u < n; u += 1) {
      clusteringSum += localClustering(u);
      clusteringSamples += 1;
    }
  } else {
    clusteringExact = false;
    let step = Math.max(1, Math.floor(n * PHI_CONJUGATE)) % n;
    if (step === 0) step = 1;
    while (gcd(step, n) !== 1) {
      step += 1;
      if (step >= n) step = 1;
    }
    let node = 0;
    let spent = 0;
    for (let t = 0; t < n; t += 1) {
      const k = undStart[node + 1] - undStart[node];
      const cost = k * k + 1;
      // Nodes whose own cost exceeds the whole budget are skipped rather than
      // allowed to consume the sample; they are the rare super-hubs.
      if (cost <= CLUSTERING_BUDGET && spent + cost <= CLUSTERING_BUDGET) {
        clusteringSum += localClustering(node);
        clusteringSamples += 1;
        spent += cost;
      }
      node = (node + step) % n;
      if (spent >= CLUSTERING_BUDGET) break;
    }
  }

  const clustering = clusteringSamples > 0 ? clusteringSum / clusteringSamples : 0;

  /* -- assemble ------------------------------------------------------------ */

  const receptors: ReceptorStat[] = [];
  for (let code = 0; code < RECEPTOR_KIND_COUNT; code += 1) {
    const count = receptorCounts[code];
    if (count === 0) continue;
    receptors.push({
      receptor: RECEPTOR_FROM_CODE[code],
      count,
      meanWeight: receptorWeights[code] / count,
      totalConductance: receptorWeights[code],
    });
  }
  receptors.sort((a, b) => b.count - a.count);

  return {
    signature,
    computeMs: performance.now() - started,
    neurons: n,
    disabledNeurons,
    synapses: kept,
    disabledSynapses,
    selfLoops,
    uniqueEdges,
    density: n > 1 ? uniqueEdges / (n * (n - 1)) : 0,
    meanDegree: kept / n,
    maxInDegree,
    maxOutDegree,
    maxDegree,
    inHistogram,
    outHistogram,
    components,
    largestComponent,
    isolated,
    reciprocity,
    clustering,
    clusteringExact,
    clusteringSamples,
    excitatoryNeurons,
    inhibitoryNeurons,
    excitatorySynapses,
    inhibitorySynapses,
    excitatoryConductance,
    inhibitoryConductance,
    meanWeight: weightSum / kept,
    receptors,
    hubs,
  };
}
