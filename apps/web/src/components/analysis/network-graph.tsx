'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Maximize2,
  PinOff,
  RefreshCw,
  RotateCw,
  TriangleAlert,
  Waypoints,
  X,
} from 'lucide-react';
import {
  Badge,
  Button,
  EmptyState,
  IconButton,
  Panel,
  PanelHeader,
  SegmentedControl,
  Slider,
  Spinner,
  Switch,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import type { SegmentedOption } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import {
  RECEPTOR_COLORS,
  RECEPTOR_FROM_CODE,
  RECEPTOR_LABELS,
  identityColorHex,
} from '@neuroforge/shared';
import type {
  Neuron,
  NeuronId,
  ReceptorKind,
  SimulationBuffers,
} from '@neuroforge/shared';
import type { SimulationEngine } from '@neuroforge/simulation';

import { getEngine } from '@/lib/runtime';
import { compact, fixed, grouped } from '@/lib/format';
import { graphSignature } from '@/lib/graph-metrics';
import { buildPathGraph } from '@/lib/pathfinding';
import type { PathGraph } from '@/lib/pathfinding';

/* --------------------------------------------------------------- constants -- */

/**
 * Ceiling on the drawn node count.
 *
 * Two hops out of a single cell in a recurrent connectome routinely reaches
 * thousands of partners, and a force layout over thousands of nodes is neither
 * affordable nor legible — it is a hairball with a frame budget. Past this the
 * view keeps the most strongly connected partners and says how many it dropped.
 */
const MAX_GRAPH_NODES = 400;

/**
 * Ceiling on the cells the traversal will *discover* per level, which is a
 * larger number than it will draw: the extra discovery exists only so the panel
 * can report the true size of the neighbourhood it truncated. Past this the
 * count becomes a lower bound and says so, rather than costing a sort of the
 * whole network on every step of the weight slider.
 */
const DISCOVERY_LIMIT = 20_000;

/** Above this many nodes a per-node label is narrower than the name it carries. */
const LABEL_LIMIT = 60;

/** Safety net for engine reloads that did not come through this component's deps. */
const SIGNATURE_POLL_MS = 500;

/**
 * Cost above which a document edit that left the neuron and synapse counts alone
 * stops triggering an automatic rebuild. Adding cells is a discrete act and
 * always worth the pass; dragging a weight slider republishes the document
 * continuously and must not pay for a CSR build sixty times a second.
 */
const AUTO_BUILD_BUDGET_MS = 16;

/* -- surface ---------------------------------------------------------------- */

/** The near-black the viewport clears to, so the canvas and the scene agree. */
const SURFACE_CSS = '#05070a';
const LABEL_CSS = '#8a93a0';
const FAINT_CSS = '#5a626d';
const ACCENT_CSS = '#4fd1ff';
const INK_CSS = '#f5f7fa';
/** Pinned rings; amber reads as "held in place" and collides with no receptor hue. */
const PIN_CSS = '#fbbf24';

/**
 * Tint the renderer gives a cell belonging to no population, and the offsets it
 * hashes a population index with. Copied from the connectivity matrix so a group
 * node here, a row there and a block of glyphs in the scene are one colour.
 */
const UNASSIGNED_CSS = '#7c8189';
const POPULATION_HUE_SALT = 0x9e37;
const POPULATION_HUE_STRIDE = 2654435761;

/* -- layout ----------------------------------------------------------------- */

/**
 * Rest length of an average edge, in layout units.
 *
 * The absolute scale of these constants is arbitrary — the view is fitted to
 * whatever extent the layout settles at — so what they are tuned against is the
 * ratio of the distance between connected cells to the distance between
 * unconnected ones. At these values a graph of two communities settles with its
 * cross-community distances a little over twice its within-community ones,
 * which is the property that makes a module visible as a module.
 */
const IDEAL_LENGTH = 44;
/** Coulomb constant for the node-node repulsion. */
const REPULSION = 9000;
/** Repulsion cutoff. Beyond this the grid stops looking and gravity takes over. */
const REPULSION_RANGE = 190;
const REPULSION_RANGE2 = REPULSION_RANGE * REPULSION_RANGE;
const INV_REPULSION_RANGE2 = 1 / REPULSION_RANGE2;
/** Hooke constant per edge, before the degree normalisation applied per spring. */
const SPRING = 0.6;
/** Pull toward the origin; this is what keeps disconnected components in frame. */
const GRAVITY = 0.015;
const VELOCITY_DECAY = 0.86;
const MAX_SPEED = 26;
/** Squared separation below which two nodes are nudged apart deterministically. */
const MIN_SEPARATION2 = 1e-4;

/** Cooling schedule: every force is scaled by alpha, which decays each tick. */
const ALPHA_DECAY = 0.985;
const ALPHA_MIN = 0.008;
/** Mean squared speed below which the layout is called settled. */
const SETTLE_ENERGY = 0.02;
/** Ticks before the energy test is trusted; the first few are quiet by accident. */
const MIN_STEPS = 24;
/** Hard ceiling per heating episode, so a pathological graph still stops. */
const MAX_STEPS = 460;
/** Solver ticks per animation frame. */
const STEPS_PER_FRAME = 4;
/** Alpha a drag reheats to. */
const REHEAT_ALPHA = 0.45;
/** Alpha a filter change reheats to, when the node set itself did not change. */
const NUDGE_ALPHA = 0.12;

/** Cells per axis in the repulsion grid; caps its allocation at ~9k buckets. */
const GRID_MAX = 96;

/* -- view ------------------------------------------------------------------- */

const MIN_SCALE = 0.06;
const MAX_SCALE = 8;
const ZOOM_RATE = 0.0016;
const ZOOM_STEP = 1.25;
/** Room left around the graph when the view is fitted, in CSS px. */
const FIT_PAD = 34;
/** Pointer travel past which a press is a drag rather than a click. */
const CLICK_SLOP = 3;
/** How far the view eases toward the fit target each tick while the solver runs. */
const FIT_EASE = 0.25;

const NODE_MIN_RADIUS = 3.2;
const NODE_MAX_RADIUS = 11;
/** Node radii scale with the zoom, but never out of usefulness. */
const MIN_DRAWN_RADIUS = 2.2;
const MAX_DRAWN_RADIUS = 34;
/** Perpendicular bow, as a fraction of the edge length. Separates reciprocal pairs. */
const EDGE_CURVE = 0.13;
const TAU = Math.PI * 2;
/** Golden angle; seeds new nodes on a phyllotaxis spiral rather than a ring. */
const GOLDEN_ANGLE = 2.399963229728653;

type HopDepth = '1' | '2';

const HOP_OPTIONS: readonly SegmentedOption<HopDepth>[] = [
  { value: '1', label: '1 hop', title: 'The selection and the cells it is directly wired to' },
  {
    value: '2',
    label: '2 hops',
    title: 'The selection, its partners, and their partners in turn',
  },
];

/* ------------------------------------------------------------------- types -- */

type GraphMode = 'neighbourhood' | 'population' | 'network';

interface Vec2 {
  x: number;
  y: number;
}

interface GraphNode {
  kind: 'cell' | 'group';
  /** Stable identity across rebuilds: the buffer slot, or −2−population index. */
  key: number;
  /** Buffer slot for a cell node; −1 for a group node. */
  slot: number;
  /** Population index for a group node; −1 for the unassigned bucket and for cells. */
  group: number;
  /** identityColorHex of the cell's morphology seed, or the population's tint. */
  color: string;
  /** Cells this node stands for; 1 for a cell node. */
  members: number;
  inhibitory: boolean;
  /** Hops from the seed set; 0 for a seed, −1 outside the neighbourhood view. */
  depth: number;
  /** Degrees inside the drawn graph. */
  inDegree: number;
  outDegree: number;
  /** Degrees in the whole running network, before any filter. */
  netIn: number;
  netOut: number;
  /** Summed peak conductance on the drawn incident edges (nS). */
  strength: number;
  /** 0..1, drives the drawn radius. */
  size01: number;
}

interface GraphEdge {
  /** Node indices. `from === to` is an autapse, or recurrence inside a population. */
  from: number;
  to: number;
  /** Summed peak conductance (nS). */
  weight: number;
  /**
   * The connection's receptor.
   *
   * For a cell edge this is the receptor of the heaviest synapse between the
   * pair, which is how `buildPathGraph` labels a folded connection and how the
   * scene and the pathways panel colour the same wiring — a connection must not
   * change hue depending on which panel is looking at it. For a population edge
   * it is the receptor whose pairs carry the most summed conductance, each pair
   * counted under the label above.
   */
  receptor: ReceptorKind;
  /** Ordered cell pairs folded into this edge; 1 for a cell edge. */
  pairs: number;
  synapses: number;
}

interface GraphView {
  mode: GraphMode;
  buildMs: number;
  nodes: readonly GraphNode[];
  edges: readonly GraphEdge[];
  /** Node index per neuron slot, or −1; only built for the population view. */
  slotNode: Int32Array | null;
  hops: number;
  /**
   * What the view could have drawn before the cap, in whatever unit it draws:
   * cells in the hop horizon including the seeds (neighbourhood), cells in the
   * network (network), occupied population buckets (population).
   */
  reached: number;
  /** False when the discovery limit stopped the count short of the true figure. */
  reachedExact: boolean;
  /** `reached` less what was drawn: what the cap left out. */
  omitted: number;
  /** Selected ids that no longer resolve to a live slot. */
  missingSeeds: number;
  /** Heaviest drawn edge (nS); normalises every thickness and rest length. */
  maxWeight: number;
  /** Edges whose two endpoints are the same node. */
  selfEdges: number;
  /** Ordered cell pairs and synapses the drawn edges stand for. */
  pairs: number;
  synapses: number;
  neurons: number;
}

/**
 * The transpose of a `PathGraph`, plus the per-slot columns the view needs.
 *
 * The columns are copied out of the live buffers rather than read through them:
 * the engine replaces every typed array on a reload, and a diagram half-built
 * from old polarities and new adjacency would draw excitation as inhibition.
 */
interface Topology {
  graph: PathGraph;
  /** Row offsets into `inEdge`, length n + 1. */
  inStart: Uint32Array;
  /** Forward-edge index of each incoming edge. */
  inEdge: Uint32Array;
  /** Source slot of each forward edge. */
  edgeSource: Uint32Array;
  polarity: Uint8Array;
  seed: Uint32Array;
  population: Uint16Array;
  /** Heaviest connection in the network (nS); the weight threshold's scale. */
  maxWeight: number;
  buildMs: number;
}

interface Filters {
  /** Minimum summed pair conductance for an edge to count (nS). */
  minWeight: number;
  /** False drops every edge whose presynaptic cell is inhibitory. */
  inhibitory: boolean;
}

/* ------------------------------------------------------------------- build -- */

/** Fold the network into forward CSR, transpose it, and copy the slot columns. */
function indexTopology(buffers: SimulationBuffers): Topology {
  const started = performance.now();
  const graph = buildPathGraph(buffers);
  const n = graph.n;
  const edges = graph.edges;

  const inStart = new Uint32Array(n + 1);
  const inEdge = new Uint32Array(edges);
  const edgeSource = new Uint32Array(edges);

  for (let e = 0; e < edges; e += 1) inStart[graph.outTarget[e] + 1] += 1;
  for (let i = 0; i < n; i += 1) inStart[i + 1] += inStart[i];

  const cursor = new Uint32Array(n);
  let maxWeight = 0;
  for (let u = 0; u < n; u += 1) {
    const end = graph.outStart[u + 1];
    for (let e = graph.outStart[u]; e < end; e += 1) {
      const v = graph.outTarget[e];
      edgeSource[e] = u;
      inEdge[inStart[v] + cursor[v]] = e;
      cursor[v] += 1;
      if (graph.outWeight[e] > maxWeight) maxWeight = graph.outWeight[e];
    }
  }

  const neurons = buffers.neurons;
  return {
    graph,
    inStart,
    inEdge,
    edgeSource,
    polarity: neurons.polarity.slice(0, n),
    seed: neurons.seed.slice(0, n),
    population: neurons.population.slice(0, n),
    maxWeight,
    buildMs: performance.now() - started,
  };
}

/**
 * Whether an edge survives the filters.
 *
 * Inhibition is judged by the polarity of the presynaptic cell rather than by
 * the receptor, which is how the rest of the app splits E from I. A GABA-A
 * synapse made by an excitatory cell is a wiring mistake, not an inhibitory
 * connection, and folding it into the inhibitory bucket would hide the mistake.
 */
function edgePasses(topology: Topology, edge: number, filters: Filters): boolean {
  if (topology.graph.outWeight[edge] < filters.minWeight) return false;
  if (!filters.inhibitory && topology.polarity[topology.edgeSource[edge]] === 1) return false;
  return true;
}

interface Neighbourhood {
  /** Admitted slots: the seeds, then partners by descending connection strength. */
  slots: Int32Array;
  /** Hop distance per admitted slot, parallel to `slots`. */
  depth: Int32Array;
  /** Cells within the horizon before the cap. */
  reached: number;
  /** False when the discovery limit stopped the count short. */
  exact: boolean;
}

/**
 * Grow the drawn set outward from the seeds, strongest partner first.
 *
 * The traversal is undirected — "who does this cell talk to" means afferents as
 * well as efferents — but it walks the directed CSR and its transpose rather
 * than materialising an undirected copy of the network.
 *
 * Every candidate at a level is discovered before any is admitted, and the ones
 * that fit are those carrying the most conductance to the part of the graph
 * already drawn. Truncating in traversal order instead would show whichever
 * partners happened to sit at low slot indices, which is an artefact of document
 * order and of nothing else.
 */
function growNeighbourhood(
  topology: Topology,
  seeds: readonly number[],
  hops: number,
  filters: Filters,
  cap: number,
): Neighbourhood {
  const graph = topology.graph;
  const n = graph.n;
  const seen = new Int32Array(n).fill(-1);
  const queued = new Uint8Array(n);
  const strength = new Float32Array(n);

  const admitted: number[] = [];
  const admittedDepth: number[] = [];
  let frontier: number[] = [];
  let reached = 0;
  let exact = true;

  for (const slot of seeds) {
    if (slot < 0 || slot >= n || seen[slot] >= 0) continue;
    seen[slot] = 0;
    reached += 1;
    frontier.push(slot);
    if (admitted.length < cap) {
      admitted.push(slot);
      admittedDepth.push(0);
    }
  }

  for (let level = 1; level <= hops && frontier.length > 0; level += 1) {
    const candidates: number[] = [];

    const consider = (v: number, weight: number): void => {
      if (seen[v] >= 0) return;
      if (queued[v] === 0) {
        // Zero-weight synapses exist, so membership cannot be inferred from the
        // accumulated strength; the stamp is what keeps candidates distinct.
        if (candidates.length >= DISCOVERY_LIMIT) {
          exact = false;
          return;
        }
        queued[v] = 1;
        candidates.push(v);
      }
      strength[v] += weight;
    };

    for (const u of frontier) {
      const outEnd = graph.outStart[u + 1];
      for (let e = graph.outStart[u]; e < outEnd; e += 1) {
        if (!edgePasses(topology, e, filters)) continue;
        consider(graph.outTarget[e], graph.outWeight[e]);
      }
      const inEnd = topology.inStart[u + 1];
      for (let i = topology.inStart[u]; i < inEnd; i += 1) {
        const e = topology.inEdge[i];
        if (!edgePasses(topology, e, filters)) continue;
        consider(topology.edgeSource[e], graph.outWeight[e]);
      }
    }

    // Stamped before ranking, so a cell found twice at this level is one
    // candidate and the next level does not rediscover this one.
    for (const v of candidates) seen[v] = level;
    reached += candidates.length;

    // Descending strength, ties to the lower slot so the view is reproducible.
    candidates.sort((a, b) => strength[b] - strength[a] || a - b);
    for (const v of candidates) {
      if (admitted.length >= cap) break;
      admitted.push(v);
      admittedDepth.push(level);
    }
    for (const v of candidates) strength[v] = 0;

    frontier = candidates;
  }

  return {
    slots: Int32Array.from(admitted),
    depth: Int32Array.from(admittedDepth),
    reached,
    exact,
  };
}

/** The `cap` cells with the highest total degree, in descending degree order. */
function topDegreeSlots(topology: Topology, cap: number): Int32Array {
  const graph = topology.graph;
  const n = graph.n;
  const degree = new Int32Array(n);
  for (let u = 0; u < n; u += 1) {
    degree[u] =
      graph.outStart[u + 1] - graph.outStart[u] + topology.inStart[u + 1] - topology.inStart[u];
  }
  const order = new Int32Array(n);
  for (let u = 0; u < n; u += 1) order[u] = u;
  // A full sort is O(V log V) once per rebuild, which next to the CSR build it
  // follows is noise, and far simpler than a partial selection that has to stay
  // stable across ties.
  order.sort((a, b) => degree[b] - degree[a] || a - b);
  return order.slice(0, Math.min(cap, n));
}

/** Running totals for one drawn edge while parallel connections are folded in. */
interface EdgeAccumulator {
  from: number;
  to: number;
  weight: number;
  pairs: number;
  synapses: number;
  /** Summed conductance per receptor, so the drawn hue is the dominant one. */
  byReceptor: Float64Array;
}

function finishEdges(accumulators: readonly EdgeAccumulator[]): GraphEdge[] {
  return accumulators.map((entry) => {
    let best = 0;
    for (let code = 1; code < entry.byReceptor.length; code += 1) {
      if (entry.byReceptor[code] > entry.byReceptor[best]) best = code;
    }
    return {
      from: entry.from,
      to: entry.to,
      weight: entry.weight,
      receptor: RECEPTOR_FROM_CODE[best],
      pairs: entry.pairs,
      synapses: entry.synapses,
    };
  });
}

/** Per-view and per-node statistics that are only knowable once the edges are. */
function summarise(
  header: Omit<
    GraphView,
    'nodes' | 'edges' | 'maxWeight' | 'selfEdges' | 'pairs' | 'synapses'
  >,
  nodes: GraphNode[],
  edges: readonly GraphEdge[],
  sizeByMembers: boolean,
): GraphView {
  let maxWeight = 0;
  let selfEdges = 0;
  let pairs = 0;
  let synapses = 0;

  for (const edge of edges) {
    if (edge.from === edge.to) selfEdges += 1;
    if (edge.weight > maxWeight) maxWeight = edge.weight;
    pairs += edge.pairs;
    synapses += edge.synapses;
    nodes[edge.from].outDegree += 1;
    nodes[edge.to].inDegree += 1;
    nodes[edge.from].strength += edge.weight;
    if (edge.to !== edge.from) nodes[edge.to].strength += edge.weight;
  }

  let maxSize = 0;
  for (const node of nodes) {
    const value = sizeByMembers ? node.members : node.inDegree + node.outDegree;
    if (value > maxSize) maxSize = value;
  }
  for (const node of nodes) {
    const value = sizeByMembers ? node.members : node.inDegree + node.outDegree;
    // Square root, so a node's *area* is proportional to what it measures; a
    // linear radius would draw a hundred-degree hub a hundred times a leaf.
    node.size01 = maxSize > 0 ? Math.sqrt(value / maxSize) : 0;
  }

  return { ...header, nodes, edges, maxWeight, selfEdges, pairs, synapses };
}

function cellNode(topology: Topology, slot: number, depth: number): GraphNode {
  return {
    kind: 'cell',
    key: slot,
    slot,
    group: -1,
    // The one colour rule the whole product obeys: a cell is this hue in the
    // chrome because it is this hue in the scene.
    color: identityColorHex(topology.seed[slot]),
    members: 1,
    inhibitory: topology.polarity[slot] === 1,
    depth,
    inDegree: 0,
    outDegree: 0,
    netIn: topology.inStart[slot + 1] - topology.inStart[slot],
    netOut: topology.graph.outStart[slot + 1] - topology.graph.outStart[slot],
    strength: 0,
    size01: 0,
  };
}

interface CellViewOptions {
  mode: GraphMode;
  slots: Int32Array;
  /** Hop distance per slot, or null when the view has no seed set. */
  depths: Int32Array | null;
  filters: Filters;
  hops: number;
  reached: number;
  reachedExact: boolean;
  omitted: number;
  missingSeeds: number;
  started: number;
}

/** The subgraph induced on a set of cells. */
function buildCellView(topology: Topology, options: CellViewOptions): GraphView {
  const graph = topology.graph;
  const n = graph.n;
  const slots = options.slots;
  const nodeOf = new Int32Array(n).fill(-1);
  const nodes: GraphNode[] = [];

  for (let i = 0; i < slots.length; i += 1) {
    const slot = slots[i];
    nodeOf[slot] = i;
    nodes.push(cellNode(topology, slot, options.depths === null ? -1 : options.depths[i]));
  }

  // The CSR already folds parallel synapses between an ordered pair, so an edge
  // here is one row of it and needs no accumulator.
  const edges: GraphEdge[] = [];
  for (let i = 0; i < slots.length; i += 1) {
    const u = slots[i];
    const end = graph.outStart[u + 1];
    for (let e = graph.outStart[u]; e < end; e += 1) {
      const target = nodeOf[graph.outTarget[e]];
      if (target < 0) continue;
      if (!edgePasses(topology, e, options.filters)) continue;
      edges.push({
        from: i,
        to: target,
        weight: graph.outWeight[e],
        receptor: RECEPTOR_FROM_CODE[graph.outReceptor[e]],
        pairs: 1,
        synapses: graph.outParallel[e],
      });
    }
  }

  return summarise(
    {
      mode: options.mode,
      buildMs: performance.now() - options.started,
      slotNode: null,
      hops: options.hops,
      reached: options.reached,
      reachedExact: options.reachedExact,
      omitted: options.omitted,
      missingSeeds: options.missingSeeds,
      neurons: n,
    },
    nodes,
    edges,
    false,
  );
}

/**
 * One node per population, one edge per ordered population pair.
 *
 * This is the view a connectome is read at before any cell is chosen: which
 * groups project where, and how strongly. It stays legible at any network size
 * because its node count is the number of populations rather than the number of
 * cells, which is exactly the property the cell views cannot have.
 */
function buildPopulationView(
  topology: Topology,
  populationCount: number,
  filters: Filters,
  started: number,
): GraphView {
  const graph = topology.graph;
  const n = graph.n;
  const buckets = populationCount + 1;
  const counts = new Uint32Array(buckets);
  const inhibitory = new Uint32Array(buckets);
  const netIn = new Float64Array(buckets);
  const netOut = new Float64Array(buckets);

  const bucketOf = (slot: number): number => {
    const p = topology.population[slot];
    return p >= populationCount ? populationCount : p;
  };

  for (let slot = 0; slot < n; slot += 1) {
    const bucket = bucketOf(slot);
    counts[bucket] += 1;
    if (topology.polarity[slot] === 1) inhibitory[bucket] += 1;
    netIn[bucket] += topology.inStart[slot + 1] - topology.inStart[slot];
    netOut[bucket] += graph.outStart[slot + 1] - graph.outStart[slot];
  }

  // The node cap holds here too. Populations are made one at a time and usually
  // number in the tens, but an imported or generated circuit can carry thousands,
  // and nothing downstream survives that: the pair index below is quadratic in
  // the node count, and a force layout over thousands of groups is the same
  // hairball the cell views cap to avoid. Largest first, so what survives is the
  // part of the network most of the cells are in; ties to the lower bucket, so
  // the view is reproducible.
  const occupied: number[] = [];
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    if (counts[bucket] > 0) occupied.push(bucket);
  }
  const kept =
    occupied.length <= MAX_GRAPH_NODES
      ? occupied
      : [...occupied]
          .sort((a, b) => counts[b] - counts[a] || a - b)
          .slice(0, MAX_GRAPH_NODES)
          .sort((a, b) => a - b);

  const nodeOfBucket = new Int32Array(buckets).fill(-1);
  const nodes: GraphNode[] = [];
  for (const bucket of kept) {
    const unassigned = bucket === populationCount;
    nodeOfBucket[bucket] = nodes.length;
    nodes.push({
      kind: 'group',
      key: unassigned ? -1 : -2 - bucket,
      slot: -1,
      group: unassigned ? -1 : bucket,
      color: unassigned
        ? UNASSIGNED_CSS
        : identityColorHex(bucket * POPULATION_HUE_STRIDE + POPULATION_HUE_SALT),
      members: counts[bucket],
      // A population is drawn as inhibitory only when every one of its cells is,
      // which is what calling a population inhibitory means; a mixed one is not.
      inhibitory: inhibitory[bucket] === counts[bucket],
      depth: -1,
      inDegree: 0,
      outDegree: 0,
      netIn: netIn[bucket],
      netOut: netOut[bucket],
      strength: 0,
      size01: 0,
    });
  }

  // −1 for a cell whose population the cap dropped, which is what keeps its
  // edges out of the aggregation below and out of the click-to-select mapping.
  const slotNode = new Int32Array(n);
  for (let slot = 0; slot < n; slot += 1) slotNode[slot] = nodeOfBucket[bucketOf(slot)];

  const count = nodes.length;
  const accumulators: EdgeAccumulator[] = [];
  const indexOf = new Int32Array(count * count).fill(-1);

  for (let u = 0; u < n; u += 1) {
    const from = slotNode[u];
    if (from < 0) continue;
    const end = graph.outStart[u + 1];
    for (let e = graph.outStart[u]; e < end; e += 1) {
      if (!edgePasses(topology, e, filters)) continue;
      const to = slotNode[graph.outTarget[e]];
      if (to < 0) continue;
      const at = from * count + to;
      let index = indexOf[at];
      if (index < 0) {
        index = accumulators.length;
        indexOf[at] = index;
        accumulators.push({
          from,
          to,
          weight: 0,
          pairs: 0,
          synapses: 0,
          byReceptor: new Float64Array(RECEPTOR_FROM_CODE.length),
        });
      }
      const entry = accumulators[index];
      const weight = graph.outWeight[e];
      entry.weight += weight;
      entry.pairs += 1;
      entry.synapses += graph.outParallel[e];
      entry.byReceptor[graph.outReceptor[e]] += weight;
    }
  }

  return summarise(
    {
      mode: 'population',
      buildMs: performance.now() - started,
      slotNode,
      hops: 0,
      reached: occupied.length,
      reachedExact: true,
      omitted: occupied.length - nodes.length,
      missingSeeds: 0,
      neurons: n,
    },
    nodes,
    finishEdges(accumulators),
    true,
  );
}

/**
 * Choose and build the view.
 *
 * With cells selected the panel answers "how is this wired"; with nothing
 * selected it answers "how is the network organised", which is a question about
 * populations rather than about cells. A document with no populations to compare
 * falls back to its most connected cells, because a single "unassigned" blob is
 * not an answer to anything.
 */
function buildGraphView(
  topology: Topology,
  seeds: readonly number[],
  missingSeeds: number,
  populationCount: number,
  hops: number,
  filters: Filters,
): GraphView {
  const started = performance.now();
  const n = topology.graph.n;

  if (seeds.length > 0 || n === 0) {
    const grown = growNeighbourhood(topology, seeds, hops, filters, MAX_GRAPH_NODES);
    return buildCellView(topology, {
      mode: 'neighbourhood',
      slots: grown.slots,
      depths: grown.depth,
      filters,
      hops,
      reached: grown.reached,
      reachedExact: grown.exact,
      omitted: Math.max(0, grown.reached - grown.slots.length),
      missingSeeds,
      started,
    });
  }

  let occupied = 0;
  if (populationCount > 0) {
    const present = new Uint8Array(populationCount + 1);
    for (let slot = 0; slot < n; slot += 1) {
      const p = topology.population[slot];
      present[p >= populationCount ? populationCount : p] = 1;
    }
    for (let bucket = 0; bucket <= populationCount; bucket += 1) occupied += present[bucket];
  }

  const slots = occupied >= 2 ? null : topDegreeSlots(topology, MAX_GRAPH_NODES);
  const fallback =
    slots === null
      ? buildPopulationView(topology, populationCount, filters, started)
      : buildCellView(topology, {
          mode: 'network',
          slots,
          depths: null,
          filters,
          hops: 0,
          reached: n,
          reachedExact: true,
          omitted: n - slots.length,
          missingSeeds: 0,
          started,
        });

  // Reaching here with seeds missing means the *whole* selection went stale, so
  // there was nothing left to grow a neighbourhood from. The panel has to say
  // so, or the sudden switch back to the overview reads as a bug.
  return missingSeeds === 0 ? fallback : { ...fallback, missingSeeds };
}

/* ------------------------------------------------------------------ solver -- */

/**
 * A compact 2D force layout.
 *
 * Deliberately not the 3D `ForceLayout` from @neuroforge/physics: that one lays
 * out anatomy in the scene, where the answer has to agree with a camera and a
 * third axis. This lays out a diagram, where the only requirements are that
 * connected cells end up near each other, that nothing overlaps, and that it
 * stops.
 *
 * Repulsion is short-range and bucketed into a uniform grid, so a pass is
 * O(n · neighbours) rather than O(n²) — at the 400-node cap the quadratic form
 * would be 160 000 pair tests a tick, four times a frame. What the cutoff gives
 * up is the long-range spreading that keeps disconnected components apart;
 * centring gravity supplies that instead, at O(n).
 */
class GraphLayout {
  readonly count: number;
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly pinned: Uint8Array;
  /** Node index by node key, for carrying positions across a rebuild. */
  readonly indexOfKey: Map<number, number>;
  /** Nodes that kept a position from the layout this one replaced. */
  readonly carried: number;

  alpha = 1;
  steps = 0;
  energy = Number.POSITIVE_INFINITY;

  private readonly vx: Float32Array;
  private readonly vy: Float32Array;
  private readonly fx: Float32Array;
  private readonly fy: Float32Array;
  private readonly mass: Float32Array;

  private readonly springA: Uint32Array;
  private readonly springB: Uint32Array;
  private readonly stiffness: Float32Array;
  private readonly rest: Float32Array;
  private readonly springs: number;

  private readonly cellOf: Uint32Array;
  private readonly gridItems: Uint32Array;
  private gridStart = new Uint32Array(0);
  private gridCursor = new Uint32Array(0);

  constructor(view: GraphView, previous: GraphLayout | null, pins: ReadonlyMap<number, Vec2>) {
    const nodes = view.nodes;
    const count = nodes.length;
    this.count = count;
    this.x = new Float32Array(count);
    this.y = new Float32Array(count);
    this.vx = new Float32Array(count);
    this.vy = new Float32Array(count);
    this.fx = new Float32Array(count);
    this.fy = new Float32Array(count);
    this.mass = new Float32Array(count);
    this.pinned = new Uint8Array(count);
    this.cellOf = new Uint32Array(count);
    this.gridItems = new Uint32Array(count);
    this.indexOfKey = new Map<number, number>();

    let carried = 0;
    for (let i = 0; i < count; i += 1) {
      const node = nodes[i];
      this.indexOfKey.set(node.key, i);
      this.mass[i] = 1 + 0.35 * Math.sqrt(node.inDegree + node.outDegree);

      const pin = pins.get(node.key);
      if (pin !== undefined) {
        this.x[i] = pin.x;
        this.y[i] = pin.y;
        this.pinned[i] = 1;
        carried += 1;
        continue;
      }
      if (previous !== null) {
        // A node that survived the last build keeps its position, so nudging the
        // weight threshold rearranges the diagram rather than restarting it.
        const at = previous.indexOfKey.get(node.key);
        if (at !== undefined) {
          this.x[i] = previous.x[at];
          this.y[i] = previous.y[at];
          carried += 1;
          continue;
        }
      }
      this.seedAt(i);
    }
    this.carried = carried;

    let springs = 0;
    for (const edge of view.edges) if (edge.from !== edge.to) springs += 1;
    this.springs = springs;
    this.springA = new Uint32Array(springs);
    this.springB = new Uint32Array(springs);
    this.stiffness = new Float32Array(springs);
    this.rest = new Float32Array(springs);

    const logMax = Math.log1p(view.maxWeight);
    let at = 0;
    for (const edge of view.edges) {
      if (edge.from === edge.to) continue;
      const a = edge.from;
      const b = edge.to;
      const degreeA = nodes[a].inDegree + nodes[a].outDegree;
      const degreeB = nodes[b].inDegree + nodes[b].outDegree;
      this.springA[at] = a;
      this.springB[at] = b;
      // Normalising by the lesser degree is what stops a hub from dragging its
      // whole fan into a knot: a leaf's one edge pulls hard, a hub's hundred
      // each pull a little, and what any node feels in total stays comparable.
      this.stiffness[at] = SPRING / Math.max(1, Math.min(degreeA, degreeB));
      const t = logMax > 0 ? Math.log1p(edge.weight) / logMax : 0;
      // Strong connections settle closer, so weight reads as proximity as well
      // as as thickness.
      this.rest[at] = IDEAL_LENGTH * (1.3 - 0.6 * t);
      at += 1;
    }
  }

  /** Phyllotaxis: even coverage without the ring artefacts of a polar seed. */
  private seedAt(i: number): void {
    const angle = i * GOLDEN_ANGLE;
    const radius = IDEAL_LENGTH * 0.9 * Math.sqrt(i + 0.5);
    this.x[i] = Math.cos(angle) * radius;
    this.y[i] = Math.sin(angle) * radius;
  }

  get settled(): boolean {
    if (this.count === 0) return true;
    if (this.steps >= MAX_STEPS) return true;
    if (this.alpha < ALPHA_MIN) return true;
    return this.steps >= MIN_STEPS && this.energy < SETTLE_ENERGY;
  }

  /** Whether this layout was built for exactly these nodes, in this order. */
  matches(view: GraphView): boolean {
    if (view.nodes.length !== this.count) return false;
    for (let i = 0; i < this.count; i += 1) {
      if (this.indexOfKey.get(view.nodes[i].key) !== i) return false;
    }
    return true;
  }

  adopt(previous: GraphLayout): void {
    this.alpha = previous.alpha;
    this.steps = previous.steps;
    this.energy = previous.energy;
  }

  reheat(alpha: number): void {
    this.alpha = Math.max(this.alpha, alpha);
    this.steps = 0;
    this.energy = Number.POSITIVE_INFINITY;
  }

  /** Throw every unpinned node back onto the spiral and start again. */
  reseed(): void {
    for (let i = 0; i < this.count; i += 1) {
      if (this.pinned[i] === 1) continue;
      this.seedAt(i);
      this.vx[i] = 0;
      this.vy[i] = 0;
    }
    this.alpha = 1;
    this.steps = 0;
    this.energy = Number.POSITIVE_INFINITY;
  }

  place(i: number, x: number, y: number): void {
    if (i < 0 || i >= this.count) return;
    this.x[i] = x;
    this.y[i] = y;
    this.vx[i] = 0;
    this.vy[i] = 0;
  }

  bounds(): { minX: number; minY: number; maxX: number; maxY: number } {
    if (this.count === 0) return { minX: -1, minY: -1, maxX: 1, maxY: 1 };
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < this.count; i += 1) {
      const x = this.x[i];
      const y = this.y[i];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    return { minX, minY, maxX, maxY };
  }

  step(): void {
    const count = this.count;
    if (count === 0) {
      this.energy = 0;
      return;
    }
    const alpha = this.alpha;
    this.fx.fill(0);
    this.fy.fill(0);
    this.repel();

    for (let e = 0; e < this.springs; e += 1) {
      const a = this.springA[e];
      const b = this.springB[e];
      const dx = this.x[b] - this.x[a];
      const dy = this.y[b] - this.y[a];
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < 1e-3) continue;
      const force = this.stiffness[e] * (d - this.rest[e]);
      const ux = (dx / d) * force;
      const uy = (dy / d) * force;
      this.fx[a] += ux;
      this.fy[a] += uy;
      this.fx[b] -= ux;
      this.fy[b] -= uy;
    }

    let energy = 0;
    for (let i = 0; i < count; i += 1) {
      if (this.pinned[i] === 1) {
        this.vx[i] = 0;
        this.vy[i] = 0;
        continue;
      }
      // Gravity is scaled by mass and the integration divides by it, so every
      // node is drawn to the centre at the same rate while a hub still resists
      // the repulsion that would otherwise push it out to the rim.
      const ax = this.fx[i] / this.mass[i] - GRAVITY * this.x[i];
      const ay = this.fy[i] / this.mass[i] - GRAVITY * this.y[i];
      let vx = (this.vx[i] + ax * alpha) * VELOCITY_DECAY;
      let vy = (this.vy[i] + ay * alpha) * VELOCITY_DECAY;
      const speed2 = vx * vx + vy * vy;
      if (speed2 > MAX_SPEED * MAX_SPEED) {
        const scale = MAX_SPEED / Math.sqrt(speed2);
        vx *= scale;
        vy *= scale;
      }
      this.vx[i] = vx;
      this.vy[i] = vy;
      this.x[i] += vx;
      this.y[i] += vy;
      energy += vx * vx + vy * vy;
    }

    this.energy = energy / count;
    this.steps += 1;
    this.alpha = alpha * ALPHA_DECAY;
  }

  /** Bucket every node, then test only the four forward-neighbouring cells. */
  private repel(): void {
    const count = this.count;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < count; i += 1) {
      const x = this.x[i];
      const y = this.y[i];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }

    // The cell is never narrower than the cutoff, so the 3x3 window around a
    // node always contains everything that can reach it. Widening the cell when
    // the graph outgrows the grid trades a few more pair tests for a bounded
    // allocation, which is the right way round.
    const divisor = Math.max(1, GRID_MAX - 1);
    const cell = Math.max(REPULSION_RANGE, (maxX - minX) / divisor, (maxY - minY) / divisor);
    const cols = Math.max(1, Math.min(GRID_MAX, Math.floor((maxX - minX) / cell) + 1));
    const rows = Math.max(1, Math.min(GRID_MAX, Math.floor((maxY - minY) / cell) + 1));
    const cells = cols * rows;

    if (this.gridStart.length < cells + 1) {
      this.gridStart = new Uint32Array(cells + 1);
      this.gridCursor = new Uint32Array(cells + 1);
    }
    const start = this.gridStart;
    start.fill(0, 0, cells + 1);

    for (let i = 0; i < count; i += 1) {
      const gx = Math.max(0, Math.min(cols - 1, Math.floor((this.x[i] - minX) / cell)));
      const gy = Math.max(0, Math.min(rows - 1, Math.floor((this.y[i] - minY) / cell)));
      const c = gy * cols + gx;
      this.cellOf[i] = c;
      start[c + 1] += 1;
    }
    for (let c = 0; c < cells; c += 1) start[c + 1] += start[c];
    this.gridCursor.set(start.subarray(0, cells));
    for (let i = 0; i < count; i += 1) {
      const c = this.cellOf[i];
      this.gridItems[this.gridCursor[c]] = i;
      this.gridCursor[c] += 1;
    }

    for (let gy = 0; gy < rows; gy += 1) {
      for (let gx = 0; gx < cols; gx += 1) {
        const c = gy * cols + gx;
        const to = start[c + 1];
        for (let a = start[c]; a < to; a += 1) {
          const i = this.gridItems[a];
          for (let b = a + 1; b < to; b += 1) this.pairRepel(i, this.gridItems[b]);
          // Four of the eight neighbours, which visits every pair exactly once.
          this.sweep(i, gx + 1, gy, cols, rows, start);
          this.sweep(i, gx - 1, gy + 1, cols, rows, start);
          this.sweep(i, gx, gy + 1, cols, rows, start);
          this.sweep(i, gx + 1, gy + 1, cols, rows, start);
        }
      }
    }
  }

  private sweep(
    i: number,
    gx: number,
    gy: number,
    cols: number,
    rows: number,
    start: Uint32Array,
  ): void {
    if (gx < 0 || gy < 0 || gx >= cols || gy >= rows) return;
    const c = gy * cols + gx;
    const to = start[c + 1];
    for (let b = start[c]; b < to; b += 1) this.pairRepel(i, this.gridItems[b]);
  }

  private pairRepel(i: number, j: number): void {
    let dx = this.x[j] - this.x[i];
    let dy = this.y[j] - this.y[i];
    let d2 = dx * dx + dy * dy;
    if (d2 >= REPULSION_RANGE2) return;
    if (d2 < MIN_SEPARATION2) {
      // Coincident nodes have no direction to separate along. The offset is
      // taken from the indices, so the same graph always resolves the same way.
      dx = ((i & 7) - 3.5) * 0.5 + 0.3;
      dy = ((j & 7) - 3.5) * 0.5 + 0.3;
      d2 = dx * dx + dy * dy;
    }
    const d = Math.sqrt(d2);
    // Subtracting the value at the cutoff lets the force reach zero at the grid
    // boundary instead of stepping off a cliff there, which would make a node
    // jitter as it crossed in and out of range.
    const force = REPULSION * this.mass[i] * this.mass[j] * (1 / d2 - INV_REPULSION_RANGE2);
    const ux = (dx / d) * force;
    const uy = (dy / d) * force;
    this.fx[i] -= ux;
    this.fy[i] -= uy;
    this.fx[j] += ux;
    this.fy[j] += uy;
  }
}

/* ------------------------------------------------------------------- paint -- */

interface Viewport {
  /** World point sitting at the centre of the canvas. */
  x: number;
  y: number;
  scale: number;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function fitViewport(
  bounds: { minX: number; minY: number; maxX: number; maxY: number },
  width: number,
  height: number,
): Viewport {
  const spanX = Math.max(bounds.maxX - bounds.minX, 1);
  const spanY = Math.max(bounds.maxY - bounds.minY, 1);
  const usableW = Math.max(1, width - FIT_PAD * 2);
  const usableH = Math.max(1, height - FIT_PAD * 2);
  return {
    x: (bounds.minX + bounds.maxX) / 2,
    y: (bounds.minY + bounds.maxY) / 2,
    scale: clamp(Math.min(usableW / spanX, usableH / spanY), MIN_SCALE, MAX_SCALE),
  };
}

function drawnRadius(node: GraphNode, scale: number): number {
  const base = NODE_MIN_RADIUS + (NODE_MAX_RADIUS - NODE_MIN_RADIUS) * node.size01;
  return clamp(base * scale, MIN_DRAWN_RADIUS, MAX_DRAWN_RADIUS);
}

function ellipsise(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let cut = text;
  while (cut.length > 1 && ctx.measureText(`${cut}…`).width > maxWidth) cut = cut.slice(0, -1);
  return `${cut}…`;
}

interface PaintOptions {
  view: GraphView;
  layout: GraphLayout;
  labels: ReadonlyMap<number, string>;
  viewport: Viewport;
  /** Node index under the pointer, or −1. */
  hover: number;
  showLabels: boolean;
  mono: string;
}

/**
 * One canvas, repainted whole.
 *
 * At the 400-node cap a full repaint is a few thousand path operations, well
 * under a frame, so the hover highlight is drawn in the same pass as the graph
 * rather than on a second canvas. That is what makes the focus effect possible
 * at all: dimming everything the pointer is *not* on cannot be done from an
 * overlay, and on a dense diagram it is the difference between a readable answer
 * and a mesh.
 */
function paintGraph(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  options: PaintOptions,
): void {
  const { view, layout, labels, viewport, hover, showLabels, mono } = options;
  const nodes = view.nodes;
  const count = nodes.length;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = SURFACE_CSS;
  ctx.fillRect(0, 0, width, height);
  if (count === 0 || layout.count !== count) return;

  const scale = viewport.scale;
  const halfW = width / 2;
  const halfH = height / 2;
  const sx = (i: number): number => (layout.x[i] - viewport.x) * scale + halfW;
  const sy = (i: number): number => (layout.y[i] - viewport.y) * scale + halfH;

  const radii = new Float32Array(count);
  for (let i = 0; i < count; i += 1) radii[i] = drawnRadius(nodes[i], scale);

  const focused = hover >= 0 && hover < count;
  const near = new Uint8Array(count);
  if (focused) {
    near[hover] = 1;
    for (const edge of view.edges) {
      if (edge.from === hover) near[edge.to] = 1;
      else if (edge.to === hover) near[edge.from] = 1;
    }
  }

  const logMax = Math.log1p(view.maxWeight);
  const strokeScale = clamp(scale, 0.55, 2);

  /* -- edges -------------------------------------------------------------- */

  ctx.lineCap = 'round';
  for (const edge of view.edges) {
    if (edge.from === edge.to) continue;
    const lit = !focused || edge.from === hover || edge.to === hover;
    const t = logMax > 0 ? Math.log1p(edge.weight) / logMax : 0;

    const ax = sx(edge.from);
    const ay = sy(edge.from);
    const bx = sx(edge.to);
    const by = sy(edge.to);
    const dx = bx - ax;
    const dy = by - ay;
    const length = Math.hypot(dx, dy);
    if (length < 1e-3) continue;

    // The bow is signed by node order, so A→B and B→A curve to opposite sides
    // and a reciprocal pair reads as two connections rather than one.
    const bow = EDGE_CURVE * length * (edge.from < edge.to ? 1 : -1);
    const cx = (ax + bx) / 2 - (dy / length) * bow;
    const cy = (ay + by) / 2 + (dx / length) * bow;

    // Both ends are trimmed back to the node rims, so the line neither starts
    // inside the source dot nor buries its arrowhead in the target one. The
    // trim is capped at a third of the span each: two adjacent hubs, or any
    // pair seen at a low enough zoom, are closer together than their own radii,
    // and an untrimmed-past-each-other line draws its arrow pointing backwards.
    const trim = length / 3;
    const headLen = Math.hypot(bx - cx, by - cy) || 1;
    const hx = (bx - cx) / headLen;
    const hy = (by - cy) / headLen;
    const tailLen = Math.hypot(cx - ax, cy - ay) || 1;
    const tailTrim = Math.min(radii[edge.from], trim);
    const headTrim = Math.min(radii[edge.to] + 1, trim);
    const startX = ax + ((cx - ax) / tailLen) * tailTrim;
    const startY = ay + ((cy - ay) / tailLen) * tailTrim;
    const endX = bx - hx * headTrim;
    const endY = by - hy * headTrim;

    ctx.globalAlpha = (0.22 + 0.52 * t) * (lit ? 1 : 0.16);
    ctx.strokeStyle = RECEPTOR_COLORS[edge.receptor];
    ctx.lineWidth = (0.6 + 2.2 * t) * strokeScale;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.quadraticCurveTo(cx, cy, endX, endY);
    ctx.stroke();

    const head = Math.min(clamp((3.6 + 3.4 * t) * strokeScale, 3.2, 11), length * 0.4);
    const px = -hy;
    const py = hx;
    ctx.globalAlpha = (0.5 + 0.45 * t) * (lit ? 1 : 0.16);
    ctx.fillStyle = RECEPTOR_COLORS[edge.receptor];
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - hx * head + px * head * 0.5, endY - hy * head + py * head * 0.5);
    ctx.lineTo(endX - hx * head - px * head * 0.5, endY - hy * head - py * head * 0.5);
    ctx.closePath();
    ctx.fill();
  }

  /* -- recurrence --------------------------------------------------------- */

  for (const edge of view.edges) {
    if (edge.from !== edge.to) continue;
    const lit = !focused || edge.from === hover;
    const t = logMax > 0 ? Math.log1p(edge.weight) / logMax : 0;
    const r = radii[edge.from];
    const loop = Math.max(3, r * 0.95);
    ctx.globalAlpha = (0.3 + 0.5 * t) * (lit ? 1 : 0.16);
    ctx.strokeStyle = RECEPTOR_COLORS[edge.receptor];
    ctx.lineWidth = (0.6 + 1.8 * t) * strokeScale;
    ctx.beginPath();
    ctx.arc(sx(edge.from) + r + loop * 0.6, sy(edge.from) - r - loop * 0.6, loop, 0, TAU);
    ctx.stroke();
  }

  /* -- nodes -------------------------------------------------------------- */

  for (let i = 0; i < count; i += 1) {
    const node = nodes[i];
    const x = sx(i);
    const y = sy(i);
    const r = radii[i];
    if (x < -r - 40 || y < -r - 40 || x > width + r + 40 || y > height + r + 40) continue;

    // Depth 2 is drawn back a little, so distance from the selection is legible
    // without spending a second colour channel on it.
    ctx.globalAlpha = focused && near[i] === 0 ? 0.3 : node.depth > 1 ? 0.78 : 1;
    ctx.fillStyle = node.color;
    ctx.beginPath();
    if (node.inhibitory) {
      // A diamond for inhibition and a disc for excitation: the polarity then
      // survives greyscale, colour blindness, and the identity hue the node is
      // obliged to carry.
      const d = r * 1.18;
      ctx.moveTo(x, y - d);
      ctx.lineTo(x + d, y);
      ctx.lineTo(x, y + d);
      ctx.lineTo(x - d, y);
      ctx.closePath();
    } else {
      ctx.arc(x, y, r, 0, TAU);
    }
    ctx.fill();
    ctx.strokeStyle = SURFACE_CSS;
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.globalAlpha = 1;
    if (node.depth === 0) {
      ctx.strokeStyle = ACCENT_CSS;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, r + 3.5, 0, TAU);
      ctx.stroke();
    }
    if (layout.pinned[i] === 1) {
      ctx.strokeStyle = PIN_CSS;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.arc(x, y, r + 2, 0, TAU);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (i === hover) {
      ctx.strokeStyle = INK_CSS;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(x, y, r + 5.5, 0, TAU);
      ctx.stroke();
    }
  }

  /* -- labels ------------------------------------------------------------- */

  ctx.globalAlpha = 1;
  ctx.font = `9px ${mono}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i < count; i += 1) {
    const emphasised = i === hover || nodes[i].depth === 0;
    if (!showLabels && !emphasised) continue;
    if (focused && near[i] === 0) continue;
    const label = labels.get(nodes[i].key);
    if (label === undefined || label.length === 0) continue;
    const x = sx(i);
    const y = sy(i);
    if (x < -60 || y < -20 || x > width + 60 || y > height + 20) continue;
    ctx.fillStyle = emphasised ? INK_CSS : LABEL_CSS;
    ctx.fillText(ellipsise(ctx, label, 92), x, y + radii[i] + 4);
  }

  // No scale bar: positions here are not measurements of anything. The zoom is
  // reported instead, because after a few wheel notches over a dense graph it is
  // the only way to know how far in you are.
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = FAINT_CSS;
  ctx.fillText(`${scale >= 1 ? scale.toFixed(2) : `1/${(1 / scale).toFixed(1)}`}×`, 8, height - 8);
}

/* --------------------------------------------------------------- component -- */

/** Prefer the document's own name for a cell, but only when it is provably that cell. */
function neuronLabel(
  neurons: readonly Neuron[],
  engine: SimulationEngine,
  slot: number,
): string {
  const id = engine.idOf(slot);
  // Slots are assigned in document order by `load`, so the positional lookup is
  // O(1) — but the buffers can be one edit ahead of this render, and naming the
  // wrong cell is worse than naming none, hence the identity check.
  const neuron = neurons[slot];
  if (neuron !== undefined && neuron.id === id && neuron.label.length > 0) return neuron.label;
  return `#${slot}`;
}

interface DragState {
  pointerId: number;
  kind: 'node' | 'pan';
  /** Node being dragged, or −1 for a background pan. */
  node: number;
  /** Pointer offset from the node centre, in world units. */
  offsetX: number;
  offsetY: number;
  /** Screen position and viewport centre at the press: the pan and the slop test. */
  startX: number;
  startY: number;
  originX: number;
  originY: number;
  moved: boolean;
  additive: boolean;
}

export interface NetworkGraphProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Overrides the default placement. */
  className?: string;
}

/**
 * The 2D wiring diagram — the connectome with the anatomy taken out.
 *
 * The scene answers where cells are; this answers what they are connected to,
 * which is a different question with a different geometry. Positions here carry
 * no anatomical meaning whatsoever: they are chosen by a force layout purely so
 * that connected cells sit near each other and an edge can be followed by eye.
 *
 * The layout runs on an animation frame until its kinetic energy settles and
 * then stops outright. A diagram that never stops moving is a battery drain and,
 * worse, unreadable — nothing can be traced between endpoints that are drifting.
 */
export function NetworkGraph({ open = true, onClose, className }: NetworkGraphProps) {
  const circuit = useEditor((s) => s.circuit);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [hops, setHops] = useState<HopDepth>('1');
  const [showInhibitory, setShowInhibitory] = useState(true);
  /** A fraction of the heaviest connection, so the setting survives an edit. */
  const [threshold, setThreshold] = useState(0);

  const [topology, setTopology] = useState<Topology | null>(null);
  const [pending, setPending] = useState(false);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [hover, setHover] = useState(-1);
  const [pinCount, setPinCount] = useState(0);
  const [settled, setSettled] = useState(true);
  const [surface, setSurface] = useState({ width: 0, height: 0, dpr: 1 });

  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const layoutRef = useRef<GraphLayout | null>(null);
  const viewportRef = useRef<Viewport>({ x: 0, y: 0, scale: 1 });
  const pinsRef = useRef(new Map<number, Vec2>());
  const dragRef = useRef<DragState | null>(null);
  const frameRef = useRef(0);
  const runningRef = useRef(false);
  const autoFitRef = useRef(true);
  const paintRef = useRef<() => void>(() => undefined);
  const unavailableRef = useRef(false);
  const busyRef = useRef(false);
  const buildFrameRef = useRef(0);
  const signatureRef = useRef('');
  const dirtyRef = useRef(true);
  const costRef = useRef<number | null>(null);
  const monoRef = useRef<{ canvas: HTMLCanvasElement | null; family: string }>({
    canvas: null,
    family: 'ui-monospace, monospace',
  });

  /* -- topology ----------------------------------------------------------- */

  const rebuild = useCallback(() => {
    if (busyRef.current) return;
    busyRef.current = true;
    setPending(true);
    // Deferring by a frame lets the busy state paint before the blocking pass,
    // and puts that pass after every effect in this commit — including the
    // ancestor one that reloads the engine, which runs after this component's
    // and would otherwise leave us indexing last edit's buffers.
    buildFrameRef.current = requestAnimationFrame(() => {
      try {
        const buffers = getEngine().buffers;
        // Stamped before the pass, so a build that throws cannot leave the poll
        // below retrying the same failing network twice a second.
        signatureRef.current = graphSignature(buffers);
        dirtyRef.current = false;
        const next = indexTopology(buffers);
        costRef.current = next.buildMs;
        setTopology(next);
        setStale(false);
        setError(null);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setPending(false);
        busyRef.current = false;
      }
    });
  }, []);

  // Copy-on-write means any edit to a cell or a synapse replaces one of these
  // arrays. Rewiring an endpoint or retuning a weight leaves the counts alone,
  // which the buffer signature cannot see, so the edit is flagged here.
  useEffect(() => {
    dirtyRef.current = true;
  }, [circuit.neurons, circuit.synapses]);

  useEffect(() => {
    if (!open) return;
    // The engine is loaded by an effect in an ancestor which commits after this
    // one; polling is what makes the panel correct regardless of that ordering,
    // and it also catches loads triggered from anywhere else.
    const poll = () => {
      if (busyRef.current) return;
      const resized = graphSignature(getEngine().buffers) !== signatureRef.current;
      if (!resized && !dirtyRef.current) return;
      if (resized || costRef.current === null || costRef.current <= AUTO_BUILD_BUDGET_MS) {
        rebuild();
      } else {
        setStale(true);
      }
    };
    poll();
    const id = setInterval(poll, SIGNATURE_POLL_MS);
    return () => {
      clearInterval(id);
      // A cancelled frame never reaches the `finally` above, so both the latch
      // and the spinner it drives have to be released here.
      cancelAnimationFrame(buildFrameRef.current);
      busyRef.current = false;
      setPending(false);
    };
  }, [open, rebuild]);

  /* -- view --------------------------------------------------------------- */

  const seeds = useMemo(() => {
    if (topology === null || selection.length === 0) {
      return { slots: [] as number[], missing: 0 };
    }
    const engine = getEngine();
    const slots: number[] = [];
    const seen = new Set<number>();
    let missing = 0;
    for (const id of selection) {
      const slot = engine.slotOf(id);
      if (slot < 0 || slot >= topology.graph.n) {
        missing += 1;
        continue;
      }
      if (seen.has(slot)) continue;
      seen.add(slot);
      slots.push(slot);
    }
    return { slots, missing };
  }, [topology, selection]);

  const minWeight = topology === null ? 0 : threshold * topology.maxWeight;
  // Read off the document once, so nothing below closes over `circuit` itself:
  // the store republishes it on every camera frame, and a memo that depended on
  // its identity would re-derive four hundred labels sixty times an orbit.
  const neurons = circuit.neurons;
  const populations = circuit.populations;
  const populationCount = populations.length;

  const view = useMemo(() => {
    if (topology === null) return null;
    return buildGraphView(topology, seeds.slots, seeds.missing, populationCount, Number(hops), {
      minWeight,
      inhibitory: showInhibitory,
    });
  }, [topology, seeds, populationCount, hops, minWeight, showInhibitory]);

  // Names live in the document; the diagram is built from the buffers. Only the
  // two collections that can actually rename a node are inputs here.
  const labels = useMemo(() => {
    const map = new Map<number, string>();
    if (view === null) return map;
    const engine = getEngine();
    for (const node of view.nodes) {
      if (node.kind === 'cell') {
        map.set(node.key, neuronLabel(neurons, engine, node.slot));
        continue;
      }
      if (node.group < 0) {
        map.set(node.key, 'Unassigned');
        continue;
      }
      const population = populations[node.group];
      const name = population === undefined ? '' : population.name;
      map.set(node.key, name.length > 0 ? name : `Population ${node.group + 1}`);
    }
    return map;
  }, [view, neurons, populations]);

  const showLabels = view !== null && view.nodes.length <= LABEL_LIMIT;

  /* -- surface ------------------------------------------------------------ */

  useEffect(() => {
    if (!open) return;
    const host = hostRef.current;
    if (host === null) return;

    const measure = () => {
      setSurface((current) => {
        const width = host.clientWidth;
        const height = host.clientHeight;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        if (current.width === width && current.height === height && current.dpr === dpr) {
          return current;
        }
        return { width, height, dpr };
      });
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(host);
    // A window resize is also how a drag to a second display announces itself,
    // which changes the device ratio without changing the element's CSS size.
    window.addEventListener('resize', measure);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [open]);

  /* -- paint -------------------------------------------------------------- */

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    const layout = layoutRef.current;
    if (canvas === null || view === null || layout === null) return;
    if (surface.width <= 0 || surface.height <= 0) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (ctx === null) {
      // Latched through a ref: this runs inside the solver loop, and an
      // unconditional setState here would schedule a render every frame.
      if (!unavailableRef.current) {
        unavailableRef.current = true;
        setUnavailable(true);
      }
      return;
    }
    if (unavailableRef.current) {
      unavailableRef.current = false;
      setUnavailable(false);
    }

    const deviceWidth = Math.max(2, Math.round(surface.width * surface.dpr));
    const deviceHeight = Math.max(2, Math.round(surface.height * surface.dpr));
    if (canvas.width !== deviceWidth || canvas.height !== deviceHeight) {
      canvas.width = deviceWidth;
      canvas.height = deviceHeight;
    }

    // The canvas carries `nf-numeric`, so its computed family is the same mono
    // face the rest of the chrome prints in. Resolved once per canvas element
    // rather than per call: `paint` runs on every frame of a solve, the face
    // cannot change between those frames, and `getComputedStyle` forces a style
    // recalculation each time it is asked.
    if (monoRef.current.canvas !== canvas) {
      monoRef.current = {
        canvas,
        family: getComputedStyle(canvas).fontFamily || 'ui-monospace, monospace',
      };
    }
    paintGraph(ctx, surface.width, surface.height, surface.dpr, {
      view,
      layout,
      labels,
      viewport: viewportRef.current,
      hover,
      showLabels,
      mono: monoRef.current.family,
    });
  }, [view, labels, surface, hover, showLabels]);

  useEffect(() => {
    paintRef.current = paint;
    paint();
  }, [paint]);

  /* -- solver loop -------------------------------------------------------- */

  // Held in a ref so the loop always re-enters the current closure: `tick`
  // captures the surface size, and a resize mid-solve must not leave the frames
  // that follow it fitting to the old one.
  const tickRef = useRef<() => void>(() => undefined);

  const tick = useCallback(() => {
    const layout = layoutRef.current;
    if (layout === null) {
      runningRef.current = false;
      frameRef.current = 0;
      setSettled(true);
      return;
    }
    for (let i = 0; i < STEPS_PER_FRAME && !layout.settled; i += 1) layout.step();

    const canFit = autoFitRef.current && surface.width > 0 && surface.height > 0;
    if (canFit) {
      const target = fitViewport(layout.bounds(), surface.width, surface.height);
      const viewport = viewportRef.current;
      // Eased rather than snapped: the graph expands as it relaxes, and a view
      // that jumped to match it every tick would read as camera shake.
      viewport.x += (target.x - viewport.x) * FIT_EASE;
      viewport.y += (target.y - viewport.y) * FIT_EASE;
      viewport.scale += (target.scale - viewport.scale) * FIT_EASE;
    }

    if (!layout.settled) {
      paintRef.current();
      frameRef.current = requestAnimationFrame(() => tickRef.current());
      return;
    }

    // Settled: land the view exactly on the fit rather than a quarter short of
    // it, paint once more, and stop. Nothing schedules another frame from here.
    runningRef.current = false;
    frameRef.current = 0;
    if (canFit) viewportRef.current = fitViewport(layout.bounds(), surface.width, surface.height);
    paintRef.current();
    setSettled(true);
  }, [surface.width, surface.height]);

  useEffect(() => {
    tickRef.current = tick;
  }, [tick]);

  const startLayout = useCallback(() => {
    if (runningRef.current || layoutRef.current === null) return;
    runningRef.current = true;
    setSettled(false);
    frameRef.current = requestAnimationFrame(() => tickRef.current());
  }, []);

  // Rebuild the solver whenever the drawn graph changes. A view over exactly the
  // same nodes — the usual result of nudging a filter — keeps its positions and
  // its temperature, so the diagram rearranges instead of lurching.
  useEffect(() => {
    if (view === null) {
      layoutRef.current = null;
      return;
    }
    const previous = layoutRef.current;
    const layout = new GraphLayout(view, previous, pinsRef.current);
    if (previous !== null && previous.matches(view)) {
      layout.adopt(previous);
      layout.reheat(NUDGE_ALPHA);
    } else {
      layout.reheat(1);
    }
    // Re-framing is reserved for a graph that is substantially a different one —
    // a new selection, a new mode. Thinning the current graph with the weight
    // slider carries almost every node over, and yanking the view back to fit
    // would undo whatever the user had zoomed in on to use the slider for.
    if (previous === null || layout.carried * 2 < view.nodes.length) {
      autoFitRef.current = true;
    }
    layoutRef.current = layout;
    startLayout();
  }, [view, startLayout]);

  useEffect(
    () => () => {
      cancelAnimationFrame(frameRef.current);
      runningRef.current = false;
    },
    [],
  );

  // Closing unmounts the canvas; reopening mounts a fresh, blank one. Nothing
  // else would repaint it — the paint effect only fires when one of its inputs
  // changes, and reopening at the same size changes none of them — and a layout
  // that was still relaxing when the panel closed would never be handed another
  // frame, leaving it frozen behind a "solving" badge that never clears.
  useEffect(() => {
    if (!open) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
      runningRef.current = false;
      return;
    }
    const layout = layoutRef.current;
    if (layout !== null && !layout.settled) startLayout();
    else paintRef.current();
  }, [open, startLayout]);

  // A node index only means anything against the view it was measured on.
  useEffect(() => {
    setHover(-1);
  }, [view]);

  /* -- viewport ----------------------------------------------------------- */

  const fit = useCallback(() => {
    const layout = layoutRef.current;
    if (layout === null || surface.width <= 0 || surface.height <= 0) return;
    autoFitRef.current = true;
    viewportRef.current = fitViewport(layout.bounds(), surface.width, surface.height);
    paintRef.current();
  }, [surface.width, surface.height]);

  const zoomAt = useCallback((px: number, py: number, factor: number) => {
    const viewport = viewportRef.current;
    const next = clamp(viewport.scale * factor, MIN_SCALE, MAX_SCALE);
    if (next === viewport.scale) return;
    const host = hostRef.current;
    if (host === null) return;
    autoFitRef.current = false;
    const offsetX = px - host.clientWidth / 2;
    const offsetY = py - host.clientHeight / 2;
    // Keep the world point under the pointer exactly where it is.
    const worldX = offsetX / viewport.scale + viewport.x;
    const worldY = offsetY / viewport.scale + viewport.y;
    viewport.x = worldX - offsetX / next;
    viewport.y = worldY - offsetY / next;
    viewport.scale = next;
    paintRef.current();
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (!open || host === null) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = host.getBoundingClientRect();
      // deltaMode is lines on Firefox and pages on some remote desktops; either
      // taken as pixels would zoom by orders of magnitude per notch.
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
      zoomAt(
        event.clientX - rect.left,
        event.clientY - rect.top,
        Math.exp(-event.deltaY * unit * ZOOM_RATE),
      );
    };
    // A passive listener cannot preventDefault and React registers onWheel as
    // one, so through React the page would scroll behind the zoom.
    host.addEventListener('wheel', onWheel, { passive: false });
    return () => host.removeEventListener('wheel', onWheel);
  }, [open, zoomAt]);

  /* -- picking ------------------------------------------------------------ */

  const pick = useCallback(
    (px: number, py: number): number => {
      const layout = layoutRef.current;
      const host = hostRef.current;
      if (view === null || layout === null || host === null) return -1;
      if (layout.count !== view.nodes.length) return -1;
      const viewport = viewportRef.current;
      const halfW = host.clientWidth / 2;
      const halfH = host.clientHeight / 2;
      // Backwards, so the node drawn last — the one on top — is the one picked.
      for (let i = view.nodes.length - 1; i >= 0; i -= 1) {
        const x = (layout.x[i] - viewport.x) * viewport.scale + halfW;
        const y = (layout.y[i] - viewport.y) * viewport.scale + halfH;
        const r = drawnRadius(view.nodes[i], viewport.scale) + 4;
        const dx = px - x;
        const dy = py - y;
        if (dx * dx + dy * dy <= r * r) return i;
      }
      return -1;
    },
    [view],
  );

  const toWorld = useCallback((px: number, py: number): Vec2 => {
    const host = hostRef.current;
    const viewport = viewportRef.current;
    const width = host === null ? 0 : host.clientWidth;
    const height = host === null ? 0 : host.clientHeight;
    return {
      x: (px - width / 2) / viewport.scale + viewport.x,
      y: (py - height / 2) / viewport.scale + viewport.y,
    };
  }, []);

  const selectNode = useCallback(
    (index: number, additive: boolean) => {
      if (view === null || index < 0 || index >= view.nodes.length) return;
      const node = view.nodes[index];
      const engine = getEngine();
      if (node.kind === 'cell') {
        const id = engine.idOf(node.slot);
        if (id !== null) select([id as NeuronId], additive);
        return;
      }
      const slotNode = view.slotNode;
      if (slotNode === null) return;
      // Selecting a population selects its cells; there is nothing else in the
      // document a population node could stand for.
      const ids: NeuronId[] = [];
      const limit = Math.min(slotNode.length, engine.buffers.neurons.count);
      for (let slot = 0; slot < limit; slot += 1) {
        if (slotNode[slot] !== index) continue;
        const id = engine.idOf(slot);
        if (id !== null) ids.push(id as NeuronId);
      }
      if (ids.length > 0) select(ids, additive);
    },
    [view, select],
  );

  /* -- pointer ------------------------------------------------------------ */

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const index = pick(px, py);
      const viewport = viewportRef.current;
      const layout = layoutRef.current;
      const onNode = index >= 0 && layout !== null;
      const world = onNode ? toWorld(px, py) : { x: 0, y: 0 };

      dragRef.current = {
        pointerId: event.pointerId,
        kind: onNode ? 'node' : 'pan',
        node: onNode ? index : -1,
        offsetX: onNode && layout !== null ? layout.x[index] - world.x : 0,
        offsetY: onNode && layout !== null ? layout.y[index] - world.y : 0,
        startX: px,
        startY: py,
        originX: viewport.x,
        originY: viewport.y,
        moved: false,
        additive: event.shiftKey,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [pick, toWorld],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const drag = dragRef.current;

      if (drag === null) {
        const index = pick(px, py);
        setHover((current) => (current === index ? current : index));
        return;
      }

      if (!drag.moved) {
        const dx = px - drag.startX;
        const dy = py - drag.startY;
        if (dx * dx + dy * dy < CLICK_SLOP * CLICK_SLOP) return;
        drag.moved = true;
      }

      if (drag.kind === 'pan') {
        autoFitRef.current = false;
        const viewport = viewportRef.current;
        viewport.x = drag.originX - (px - drag.startX) / viewport.scale;
        viewport.y = drag.originY - (py - drag.startY) / viewport.scale;
        paintRef.current();
        return;
      }

      const layout = layoutRef.current;
      if (layout === null) return;
      autoFitRef.current = false;
      const world = toWorld(px, py);
      layout.place(drag.node, world.x + drag.offsetX, world.y + drag.offsetY);
      layout.pinned[drag.node] = 1;
      // The rest of the diagram has to make room, so the solver is woken up.
      layout.reheat(REHEAT_ALPHA);
      startLayout();
      paintRef.current();
    },
    [pick, toWorld, startLayout],
  );

  const endDrag = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current;
      if (drag === null || drag.pointerId !== event.pointerId) return;
      dragRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }

      const layout = layoutRef.current;
      if (drag.kind !== 'node' || layout === null || view === null) return;
      if (!drag.moved) {
        selectNode(drag.node, drag.additive);
        return;
      }
      // Dropped where the user put it. The pin is what makes that position mean
      // something, and it survives every later rebuild of the same node.
      const node = view.nodes[drag.node];
      if (node !== undefined) {
        pinsRef.current.set(node.key, { x: layout.x[drag.node], y: layout.y[drag.node] });
        setPinCount(pinsRef.current.size);
      }
    },
    [view, selectNode],
  );

  const onDoubleClick = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const index = pick(event.clientX - rect.left, event.clientY - rect.top);
      const layout = layoutRef.current;
      if (index < 0 || layout === null || view === null) return;
      const node = view.nodes[index];
      if (node === undefined || layout.pinned[index] === 0) return;
      layout.pinned[index] = 0;
      pinsRef.current.delete(node.key);
      setPinCount(pinsRef.current.size);
      layout.reheat(REHEAT_ALPHA);
      startLayout();
    },
    [pick, view, startLayout],
  );

  const unpinAll = useCallback(() => {
    pinsRef.current.clear();
    setPinCount(0);
    const layout = layoutRef.current;
    if (layout === null) return;
    layout.pinned.fill(0);
    layout.reheat(REHEAT_ALPHA);
    startLayout();
  }, [startLayout]);

  const relayout = useCallback(() => {
    const layout = layoutRef.current;
    if (layout === null) return;
    autoFitRef.current = true;
    layout.reseed();
    startLayout();
  }, [startLayout]);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const host = hostRef.current;
      const viewport = viewportRef.current;
      const step = 40 / viewport.scale;
      switch (event.key) {
        case '0':
          event.preventDefault();
          fit();
          return;
        case '+':
        case '=':
          event.preventDefault();
          if (host !== null) zoomAt(host.clientWidth / 2, host.clientHeight / 2, ZOOM_STEP);
          return;
        case '-':
          event.preventDefault();
          if (host !== null) zoomAt(host.clientWidth / 2, host.clientHeight / 2, 1 / ZOOM_STEP);
          return;
        case 'ArrowLeft':
        case 'ArrowRight':
        case 'ArrowUp':
        case 'ArrowDown':
          event.preventDefault();
          autoFitRef.current = false;
          if (event.key === 'ArrowLeft') viewport.x -= step;
          if (event.key === 'ArrowRight') viewport.x += step;
          if (event.key === 'ArrowUp') viewport.y -= step;
          if (event.key === 'ArrowDown') viewport.y += step;
          paintRef.current();
          return;
        case 'Enter':
        case ' ':
          if (hover < 0) return;
          event.preventDefault();
          selectNode(hover, event.shiftKey);
          return;
        default:
          return;
      }
    },
    [fit, zoomAt, hover, selectNode],
  );

  if (!open) return null;

  /* ---------------------------------------------------------------- view -- */

  const placement =
    className ?? 'absolute top-3 bottom-3 left-1/2 w-[min(560px,calc(100vw-2rem))] -translate-x-1/2';

  const subtitle =
    view === null
      ? undefined
      : view.mode === 'neighbourhood'
        ? `${grouped(view.nodes.length)} cells · ${grouped(view.edges.length)} connections · ${
            view.hops
          } hop${view.hops === 1 ? '' : 's'}`
        : view.mode === 'population'
          ? `${grouped(view.nodes.length)} populations · ${grouped(view.edges.length)} projections`
          : `${grouped(view.nodes.length)} of ${grouped(view.neurons)} cells · ${grouped(
              view.edges.length,
            )} connections`;

  const header = (
    <PanelHeader
      title="Network graph"
      subtitle={subtitle}
      icon={<Waypoints />}
      actions={
        <>
          {error !== null ? (
            <Tooltip content={`The last build failed: ${error}`}>
              <Badge variant="danger" size="sm" tabIndex={0}>
                failed
              </Badge>
            </Tooltip>
          ) : stale ? (
            <Tooltip content="The circuit changed after this build, and re-indexing its adjacency costs more than a frame. Rebuild to bring the diagram back in step.">
              <Badge variant="warning" size="sm" tabIndex={0}>
                stale
              </Badge>
            </Tooltip>
          ) : !settled ? (
            <Tooltip content="The force layout is still relaxing. It stops of its own accord once it settles.">
              <Badge variant="accent" size="sm" dot tabIndex={0}>
                solving
              </Badge>
            </Tooltip>
          ) : view !== null ? (
            <Tooltip content="Time taken to derive this diagram from the indexed adjacency">
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(view.buildMs, 1)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Scatter the diagram and solve it again, leaving pinned nodes where they are">
            <IconButton label="Re-run layout" size="sm" onClick={relayout} disabled={view === null}>
              <RotateCw />
            </IconButton>
          </Tooltip>
          <Tooltip content="Re-index the adjacency from the running network">
            <IconButton label="Rebuild graph" size="sm" onClick={rebuild} disabled={pending}>
              <RefreshCw className={pending ? 'animate-spin' : undefined} />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close network graph panel" size="sm" onClick={onClose}>
              <X />
            </IconButton>
          ) : null}
        </>
      }
    />
  );

  const seeded = view !== null && view.mode === 'neighbourhood';

  const controls = (
    <div className="flex shrink-0 flex-col gap-1.5 border-b border-hairline px-3 py-2">
      <div className="flex items-center gap-2">
        <Tooltip
          content={
            seeded
              ? 'How far out of the selection the diagram reaches.'
              : 'Hop depth applies to a selection. Select cells — in the scene, the search panel or the hub list — and the diagram follows their wiring outward from there.'
          }
          side="top"
        >
          <span className="shrink-0 text-[9.5px] tracking-[0.08em] text-ink-faint uppercase">
            depth
          </span>
        </Tooltip>
        <SegmentedControl
          size="sm"
          value={hops}
          onChange={setHops}
          options={HOP_OPTIONS}
          aria-label="Hop depth"
          className={seeded ? undefined : 'opacity-50'}
        />
        <Toggle
          className="ml-auto"
          label="inhibitory"
          hint="Include connections made by inhibitory cells. Hiding them leaves the excitatory skeleton of the circuit, which is the only part a feedforward motif lives in."
          checked={showInhibitory}
          onChange={setShowInhibitory}
        />
      </div>
      <div className="flex items-center gap-2">
        <Tooltip
          content="Drop connections weaker than this. The one control that turns a hairball back into a diagram: raising it also stops weak partners from being reached at all."
          side="top"
        >
          <span className="shrink-0 text-[9.5px] tracking-[0.08em] text-ink-faint uppercase">
            min w
          </span>
        </Tooltip>
        <Slider
          className="min-w-0 flex-1"
          value={threshold}
          onChange={setThreshold}
          min={0}
          max={1}
          step={0.005}
          showValue
          formatValue={() => (minWeight <= 0 ? 'all' : `≥ ${fixed(minWeight, 2)} nS`)}
          aria-label="Minimum connection weight"
        />
      </div>
    </div>
  );

  if (view === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        {controls}
        {error === null ? (
          <EmptyState
            icon={<Spinner size={18} />}
            title="Building the graph"
            description="Folding the running network into a wiring diagram."
          />
        ) : (
          <EmptyState
            icon={<TriangleAlert className="text-danger" />}
            title="Could not build the graph"
            description={error}
            action={
              <Button size="sm" icon={<RefreshCw />} onClick={rebuild} loading={pending}>
                Try again
              </Button>
            }
          />
        )}
      </Panel>
    );
  }

  const notice =
    view.neurons === 0
      ? 'No neurons yet — place cells and wire them together, and their connectivity appears here.'
      : view.nodes.length === 0
        ? selection.length > 0
          ? 'The selected cells are no longer in the running network.'
          : 'Nothing to draw yet.'
        : unavailable
          ? 'This browser would not give the diagram a 2D canvas, so it cannot be drawn.'
          : null;

  const hovered = hover >= 0 && hover < view.nodes.length ? view.nodes[hover] : null;
  const receptors = new Set<ReceptorKind>();
  for (const edge of view.edges) receptors.add(edge.receptor);

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      {header}
      {controls}

      <div
        ref={hostRef}
        tabIndex={0}
        role="application"
        aria-label={`Network graph: ${view.nodes.length} nodes, ${view.edges.length} connections. Drag a node to pin it, click to select it, double-click to release it. Scroll to zoom, drag the background to pan, press 0 to fit.`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => {
          if (dragRef.current === null) setHover(-1);
        }}
        onDoubleClick={onDoubleClick}
        onKeyDown={onKeyDown}
        className={cn(
          'relative min-h-[240px] w-full flex-1 touch-none focus-visible:outline-1',
          hovered === null ? 'cursor-grab active:cursor-grabbing' : 'cursor-pointer',
        )}
      >
        <canvas
          ref={canvasRef}
          aria-hidden
          className="nf-numeric absolute inset-0 block size-full"
        />

        {/* The host captures the pointer to pan, which would retarget the
            release away from whichever button was pressed and swallow its
            click. These controls therefore keep their presses to themselves. */}
        <div
          className="absolute top-2 right-2 flex items-center gap-0.5"
          onPointerDown={(event) => event.stopPropagation()}
          onDoubleClick={(event) => event.stopPropagation()}
        >
          <Tooltip content="Fit the diagram to the panel" shortcut="0" side="left">
            <IconButton
              label="Fit graph to view"
              size="sm"
              className="bg-panel/70 backdrop-blur-sm"
              onClick={fit}
            >
              <Maximize2 />
            </IconButton>
          </Tooltip>
          {pinCount > 0 ? (
            <Tooltip content={`Release all ${pinCount} pinned nodes`} side="left">
              <IconButton
                label="Unpin all nodes"
                size="sm"
                className="bg-panel/70 text-warning backdrop-blur-sm"
                onClick={unpinAll}
              >
                <PinOff />
              </IconButton>
            </Tooltip>
          ) : null}
        </div>

        {notice !== null ? (
          // Opaque rather than translucent: a degenerate diagram painted behind
          // the message would read as data.
          <div
            className="pointer-events-none absolute inset-0 flex items-center justify-center px-6"
            style={{ backgroundColor: SURFACE_CSS }}
          >
            <span className="max-w-[40ch] text-center text-[10.5px] leading-relaxed text-ink-faint">
              {notice}
            </span>
          </div>
        ) : null}
      </div>

      {/* -- readout and legend -------------------------------------------- */}

      <div className="flex shrink-0 flex-col gap-1.5 border-t border-hairline px-3 py-2">
        <div className="flex min-h-[30px] flex-col justify-center gap-1">
          {hovered === null ? (
            <p className="text-[10.5px] leading-tight text-ink-faint">
              {view.mode === 'population'
                ? 'One node per population, sized by the cells it holds. Select cells to follow their wiring outward instead.'
                : view.mode === 'network'
                  ? 'The most connected cells in the network. Select cells to follow their wiring outward instead.'
                  : 'Drag a node to pin it, click to select it, double-click to release it.'}
            </p>
          ) : (
            <>
              <div className="flex items-center gap-1.5 text-[10.5px]">
                <span
                  aria-hidden
                  className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
                  style={{ backgroundColor: hovered.color }}
                />
                <span className="min-w-0 flex-1 truncate text-ink">
                  {labels.get(hovered.key) ?? '—'}
                </span>
                {hovered.inhibitory ? (
                  <span className="shrink-0 text-[8.5px] font-semibold tracking-[0.08em] text-secondary">
                    INH
                  </span>
                ) : null}
                {hovered.depth >= 0 ? (
                  <Badge variant="outline" size="sm" numeric>
                    hop {hovered.depth}
                  </Badge>
                ) : null}
                <span className="nf-numeric shrink-0 text-ink">
                  {fixed(hovered.strength, 2)} nS
                </span>
              </div>
              <div className="nf-numeric flex items-baseline gap-2 text-[9.5px] text-ink-faint">
                <span>
                  drawn {grouped(hovered.inDegree)}
                  <span className="text-ink-faint/60">/</span>
                  {grouped(hovered.outDegree)}
                </span>
                <span>
                  network {grouped(hovered.netIn)}
                  <span className="text-ink-faint/60">/</span>
                  {grouped(hovered.netOut)}
                </span>
                {hovered.kind === 'group' ? <span>{grouped(hovered.members)} cells</span> : null}
                <span className="ml-auto">in / out</span>
              </div>
            </>
          )}
        </div>

        {view.omitted > 0 ? (
          <p className="text-[9.5px] leading-snug text-warning">
            {view.mode === 'neighbourhood' ? (
              <>
                {view.reachedExact ? '' : 'At least '}
                {grouped(view.reached)} cells lie within {view.hops} hop
                {view.hops === 1 ? '' : 's'} — more than this view can draw legibly. The{' '}
                {grouped(view.nodes.length)} most strongly connected are shown. Raise the weight
                threshold or reduce the depth to see a whole neighbourhood.
              </>
            ) : view.mode === 'population' ? (
              <>
                Drawing the {grouped(view.nodes.length)} largest of {grouped(view.reached)}{' '}
                populations; {grouped(view.omitted)} are not shown. Select cells to follow their
                wiring outward instead.
              </>
            ) : (
              <>
                Drawing the {grouped(view.nodes.length)} most connected cells;{' '}
                {grouped(view.omitted)} are not shown. Group the network into populations, or
                select cells, for a complete view.
              </>
            )}
          </p>
        ) : null}
        {view.missingSeeds > 0 ? (
          <p className="text-[9.5px] leading-snug text-ink-faint">
            {grouped(view.missingSeeds)} selected{' '}
            {view.missingSeeds === 1 ? 'cell is' : 'cells are'} not in the running network
            {view.mode === 'neighbourhood'
              ? '.'
              : ', so there was nothing to grow a neighbourhood from — this is the overview instead.'}
          </p>
        ) : null}

        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[9px] text-ink-faint">
          {RECEPTOR_FROM_CODE.filter((receptor) => receptors.has(receptor)).map((receptor) => (
            <span key={receptor} className="flex items-center gap-1">
              <span
                aria-hidden
                className="h-[2px] w-3 rounded-full"
                style={{ backgroundColor: RECEPTOR_COLORS[receptor] }}
              />
              {RECEPTOR_LABELS[receptor]}
            </span>
          ))}
          <span className="ml-auto flex items-center gap-2.5">
            <span>◆ inhibitory</span>
            <span>size · {view.mode === 'population' ? 'cells' : 'degree'}</span>
            {view.selfEdges > 0 ? <span>{compact(view.selfEdges)} recurrent</span> : null}
            {view.mode === 'population' ? (
              <span className="nf-numeric">{compact(view.pairs)} pairs</span>
            ) : null}
            <span className="nf-numeric">{compact(view.synapses)} syn</span>
          </span>
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ parts -- */

interface ToggleProps {
  label: string;
  hint: string;
  checked: boolean;
  className?: string;
  onChange: (checked: boolean) => void;
}

function Toggle({ label, hint, checked, className, onChange }: ToggleProps) {
  return (
    <Tooltip content={hint} side="top">
      <label
        className={cn(
          'flex cursor-pointer items-center gap-1.5 text-[9.5px] tracking-[0.08em] text-ink-muted uppercase select-none hover:text-ink',
          className,
        )}
      >
        <Switch size="sm" checked={checked} onCheckedChange={onChange} aria-label={label} />
        {label}
      </label>
    </Tooltip>
  );
}
