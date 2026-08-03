'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Grid3x3, RefreshCw, TriangleAlert, X } from 'lucide-react';
import {
  Badge,
  Button,
  EmptyState,
  IconButton,
  Panel,
  PanelHeader,
  SegmentedControl,
  Spinner,
  Switch,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import type { SegmentedOption } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { identityColorHex } from '@neuroforge/shared';
import type { Circuit, NeuronId } from '@neuroforge/shared';
import type { SimulationEngine } from '@neuroforge/simulation';

import { getEngine } from '@/lib/runtime';
import { compact, fixed, grouped } from '@/lib/format';
import { graphSignature } from '@/lib/graph-metrics';

/* ------------------------------------------------------------- constants -- */

/** Rows shown by the by-neuron view when nothing is selected. */
const DEFAULT_NEURON_ROWS = 128;

/**
 * Ceiling on a by-neuron matrix. 512 rows is already 262 144 cells; past that
 * the grid is finer than the panel has pixels and the build stops being free.
 */
const MAX_NEURON_ROWS = 512;

/** Safety net for engine reloads that did not come through this component's deps. */
const SIGNATURE_POLL_MS = 500;

/**
 * Cost above which a document edit that left the neuron and synapse counts alone
 * stops triggering an automatic rebuild.
 *
 * Adding or deleting cells is a discrete act and always worth the pass. Dragging
 * a weight slider is not: it republishes the document continuously, and a build
 * longer than a frame would stall the pointer sixty times a second. Past this
 * budget the panel marks itself stale and waits to be asked. Changing the
 * grouping or the selection always rebuilds, because those change what the
 * matrix *is* rather than what it says.
 */
const AUTO_BUILD_BUDGET_MS = 16;

const SURFACE_CSS = '#0a0d11';
/** The field a cell with no connection sits on — the near-black of the viewport. */
const FIELD_CSS = '#05070a';
const GRID_CSS = 'rgba(255,255,255,0.05)';
const DIAGONAL_CSS = 'rgba(255,255,255,0.14)';
const HIGHLIGHT_CSS = 'rgba(255,255,255,0.08)';
const INK_MUTED_CSS = '#8a93a0';
const ACCENT_CSS = '#4fd1ff';

/**
 * Tint the renderer gives a cell that belongs to no population, converted from
 * the linear triple in `neuron-field.ts` to sRGB. Keeping the two in agreement
 * is what lets a grey row here and a grey glyph in the scene be the same thing.
 */
const UNASSIGNED_CSS = '#7c8189';

/** Offset the renderer applies before hashing a population index into a hue. */
const POPULATION_HUE_SALT = 0x9e37;
const POPULATION_HUE_STRIDE = 2654435761;

const PAD = 8;
/** Thickness of the identity-colour strip along each axis, in CSS px. */
const TICK = 5;
const TICK_GAP = 3;
/** Room reserved for a text label beside the tick strip. */
const LABEL_ROOM = 74;
/** Below this cell size a label cannot be told apart from its neighbour. */
const MIN_LABEL_CELL = 8;
/** Below this the grid lines cost more legibility than they add. */
const MIN_GRID_CELL = 6;

/* ------------------------------------------------------------------ ramps -- */

type Rgb = readonly [number, number, number];

/**
 * Magma. Monotonic in lightness from black, which is the property that makes a
 * heatmap readable: rank order in the data survives as rank order in perceived
 * brightness, and it does so under greyscale printing and for dichromats. It
 * also starts at the field colour, so an unconnected cell and a zero-weight one
 * agree rather than fight.
 */
const SEQUENTIAL: readonly Rgb[] = [
  [0, 0, 4],
  [24, 15, 61],
  [68, 15, 118],
  [114, 31, 129],
  [158, 47, 127],
  [205, 64, 113],
  [241, 96, 93],
  [253, 150, 104],
  [252, 253, 191],
];

/**
 * The two arms of the diverging scale, used when the excitatory / inhibitory
 * sign is shown. Both run out of the near-black field rather than through a
 * light neutral: on a black canvas a light midpoint would make *zero* the
 * brightest thing in the matrix. Hues are the app's own E and I accents, so the
 * matrix, the balance bars and the scene all say inhibition in the same violet.
 */
const DIVERGING_NEG: readonly Rgb[] = [
  [5, 7, 10],
  [58, 32, 86],
  [124, 72, 176],
  [182, 107, 255],
  [231, 204, 255],
];

const DIVERGING_POS: readonly Rgb[] = [
  [5, 7, 10],
  [23, 71, 94],
  [43, 143, 181],
  [79, 209, 255],
  [205, 241, 255],
];

const RAMP_STEPS = 256;

/**
 * Floor applied to every realised connection's position on the ramp, so the
 * weakest edge in the network is still separable from an absent one. Nothing
 * below it is reachable, which is why the legend gradient is floored too.
 */
const CELL_FLOOR = 0.08;

function sampleRamp(ramp: readonly Rgb[], t: number): Rgb {
  const clamped = t <= 0 ? 0 : t >= 1 ? 1 : t;
  const x = clamped * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(x));
  const f = x - i;
  const a = ramp[i];
  const b = ramp[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** Quantised ramp. A 512-row matrix is 262 144 cells; none of them may allocate. */
function bakeRamp(ramp: readonly Rgb[]): readonly string[] {
  const out = new Array<string>(RAMP_STEPS);
  for (let i = 0; i < RAMP_STEPS; i += 1) {
    const [r, g, b] = sampleRamp(ramp, i / (RAMP_STEPS - 1));
    out[i] = `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
  }
  return out;
}

const SEQUENTIAL_CSS = bakeRamp(SEQUENTIAL);
const NEGATIVE_CSS = bakeRamp(DIVERGING_NEG);
const POSITIVE_CSS = bakeRamp(DIVERGING_POS);

function rampIndex(t: number): number {
  const floored = CELL_FLOOR + (1 - CELL_FLOOR) * (t <= 0 ? 0 : t >= 1 ? 1 : t);
  return Math.round(floored * (RAMP_STEPS - 1));
}

function gradientStops(ramp: readonly string[], from: number, to: number, reversed: boolean): string {
  const stops: string[] = [];
  const count = 10;
  for (let i = 0; i <= count; i += 1) {
    const t = i / count;
    const colour = ramp[rampIndex(reversed ? 1 - t : t)];
    stops.push(`${colour} ${(from + (to - from) * t).toFixed(1)}%`);
  }
  return stops.join(',');
}

const SEQUENTIAL_GRADIENT = `linear-gradient(90deg, ${gradientStops(SEQUENTIAL_CSS, 0, 100, false)})`;
const DIVERGING_GRADIENT = `linear-gradient(90deg, ${gradientStops(
  NEGATIVE_CSS,
  0,
  50,
  true,
)},${gradientStops(POSITIVE_CSS, 50, 100, false)})`;

/* ------------------------------------------------------------------ types -- */

type Grouping = 'population' | 'neuron';
type Metric = 'total' | 'mean' | 'prob';

const GROUPING_OPTIONS: readonly SegmentedOption<Grouping>[] = [
  {
    value: 'population',
    label: 'population',
    title: 'One row and column per population, summarising every synapse between them',
  },
  {
    value: 'neuron',
    label: 'neuron',
    title: 'Raw adjacency for the selected cells — this is where motifs are visible',
  },
];

const METRIC_OPTIONS: readonly SegmentedOption<Metric>[] = [
  { value: 'total', label: 'Σw', title: 'Total peak conductance from the row group to the column group' },
  { value: 'mean', label: 'mean', title: 'Mean peak conductance per synapse' },
  { value: 'prob', label: 'p', title: 'Connection probability: realised ordered pairs over possible ones' },
];

interface MatrixGroup {
  label: string;
  /** The hue this group is drawn in by the renderer. */
  color: string;
  /** Cells the group covers; 1 in the by-neuron view. */
  size: number;
  /** Buffer slot in the by-neuron view, else -1. */
  slot: number;
  /** Index into `circuit.populations`, or -1 for the unassigned bucket. */
  population: number;
}

interface MatrixData {
  mode: Grouping;
  buildMs: number;
  groups: readonly MatrixGroup[];
  /** Group index per neuron slot, or -1 when the slot is outside the matrix. */
  slotGroup: Int32Array;
  /** Synapse count per ordered group pair. */
  edges: Uint32Array;
  /** Distinct ordered neuron pairs per group pair; the numerator of p. */
  pairs: Uint32Array;
  /** Summed peak conductance (nS), unsigned. */
  weight: Float64Array;
  /** Summed peak conductance signed by the polarity of the presynaptic cell. */
  signed: Float64Array;
  /** Enabled synapses that landed inside the matrix. */
  shownEdges: number;
  /** Enabled synapses with at least one endpoint outside it. */
  hiddenEdges: number;
  /** Cells the row cap left out of the by-neuron view. */
  omitted: number;
  neurons: number;
}

interface ValueField {
  values: Float64Array;
  /** Largest magnitude in the matrix. */
  max: number;
  /** Smallest non-zero magnitude; Infinity when the matrix is empty. */
  minPositive: number;
  negatives: boolean;
  /** '%' for probabilities and row-normalised shares, 'nS' for conductance. */
  unit: '%' | 'nS';
}

interface Geometry {
  /** Top-left of the matrix body, in CSS px. */
  x: number;
  y: number;
  side: number;
  cell: number;
  gutter: number;
  labels: boolean;
}

interface Cell {
  row: number;
  col: number;
}

/* ------------------------------------------------------------------ build -- */

/** Prefer the document's own name for a cell, but only when it is provably the same cell. */
function neuronLabel(circuit: Circuit, engine: SimulationEngine, slot: number): string {
  const id = engine.idOf(slot);
  // Slots are assigned in document order by `load`, so the positional lookup is
  // O(1) — but the buffers can be one edit ahead of this render, and naming the
  // wrong cell is worse than naming none, hence the identity check.
  const neuron = circuit.neurons[slot];
  if (neuron !== undefined && neuron.id === id && neuron.label.length > 0) return neuron.label;
  return `#${slot}`;
}

/**
 * Summarise the running network into an N x N matrix.
 *
 * Read off the live buffers rather than the document: the engine drops synapses
 * whose endpoints were deleted and plasticity rewrites weights as it runs, so a
 * matrix built from `circuit.synapses` would describe a network nobody is
 * watching. Everything is one linear pass over the synapse list, plus a CSR
 * build for the distinct-pair count that connection probability needs.
 */
function buildMatrix(
  engine: SimulationEngine,
  circuit: Circuit,
  mode: Grouping,
  selection: readonly NeuronId[],
): MatrixData {
  const started = performance.now();
  const buffers = engine.buffers;
  const neurons = buffers.neurons;
  const synapses = buffers.synapses;
  const n = neurons.count;

  const slotGroup = new Int32Array(n);
  slotGroup.fill(-1);
  const groups: MatrixGroup[] = [];
  let omitted = 0;

  if (mode === 'population') {
    const populationCount = circuit.populations.length;
    // One extra bucket at the end for cells that belong to no population.
    const counts = new Uint32Array(populationCount + 1);
    const bucketOf = (slot: number): number => {
      const p = neurons.population[slot];
      return p === 0xffff || p >= populationCount ? populationCount : p;
    };
    for (let i = 0; i < n; i += 1) counts[bucketOf(i)] += 1;

    const groupOfBucket = new Int32Array(populationCount + 1);
    groupOfBucket.fill(-1);
    for (let bucket = 0; bucket <= populationCount; bucket += 1) {
      if (counts[bucket] === 0) continue;
      const unassigned = bucket === populationCount;
      const name = unassigned ? '' : (circuit.populations[bucket]?.name ?? '');
      groupOfBucket[bucket] = groups.length;
      groups.push({
        label: unassigned
          ? 'Unassigned'
          : name.length > 0
            ? name
            : `Population ${bucket + 1}`,
        // Reproduces the renderer's population tint exactly, so a row here and a
        // group of glyphs in the scene are the same colour under `population`
        // colour mode.
        color: unassigned
          ? UNASSIGNED_CSS
          : identityColorHex(bucket * POPULATION_HUE_STRIDE + POPULATION_HUE_SALT),
        size: counts[bucket],
        slot: -1,
        population: unassigned ? -1 : bucket,
      });
    }
    for (let i = 0; i < n; i += 1) slotGroup[i] = groupOfBucket[bucketOf(i)];
  } else {
    const slots: number[] = [];
    if (selection.length > 0) {
      const seen = new Uint8Array(n);
      for (const id of selection) {
        const slot = engine.slotOf(id);
        if (slot < 0 || slot >= n || seen[slot] === 1) continue;
        seen[slot] = 1;
        if (slots.length >= MAX_NEURON_ROWS) {
          omitted += 1;
          continue;
        }
        slots.push(slot);
      }
      // Slot order is document order, which keeps a population's cells adjacent
      // and therefore keeps its block visible as a block.
      slots.sort((a, b) => a - b);
    } else {
      const limit = Math.min(n, DEFAULT_NEURON_ROWS);
      for (let i = 0; i < limit; i += 1) slots.push(i);
      omitted = n - limit;
    }

    for (let i = 0; i < slots.length; i += 1) {
      const slot = slots[i];
      slotGroup[slot] = i;
      const population = neurons.population[slot];
      groups.push({
        label: neuronLabel(circuit, engine, slot),
        color: identityColorHex(neurons.seed[slot]),
        size: 1,
        slot,
        population:
          population === 0xffff || population >= circuit.populations.length ? -1 : population,
      });
    }
  }

  const size = groups.length;
  const cells = size * size;
  const edges = new Uint32Array(cells);
  const pairs = new Uint32Array(cells);
  const weight = new Float64Array(cells);
  const signed = new Float64Array(cells);

  const total = synapses.count;
  let shownEdges = 0;
  let hiddenEdges = 0;

  for (let s = 0; s < total; s += 1) {
    if (synapses.enabled[s] === 0) continue;
    const pre = synapses.pre[s];
    const post = synapses.post[s];
    if (pre >= n || post >= n) continue;
    const gi = slotGroup[pre];
    const gj = slotGroup[post];
    if (gi < 0 || gj < 0) {
      hiddenEdges += 1;
      continue;
    }
    const at = gi * size + gj;
    edges[at] += 1;
    const w = synapses.weight[s];
    weight[at] += w;
    signed[at] += neurons.polarity[pre] === 1 ? -w : w;
    shownEdges += 1;
  }

  // Connection probability counts *pairs*, not synapses: two axons onto the same
  // dendrite are one realised connection. Deduplicating needs the out-adjacency,
  // so it is built here in CSR and walked once with a per-row marker.
  if (shownEdges > 0) {
    const start = new Uint32Array(n + 1);
    for (let s = 0; s < total; s += 1) {
      if (synapses.enabled[s] === 0) continue;
      const pre = synapses.pre[s];
      const post = synapses.post[s];
      if (pre >= n || post >= n) continue;
      if (slotGroup[pre] < 0 || slotGroup[post] < 0) continue;
      start[pre + 1] += 1;
    }
    for (let i = 0; i < n; i += 1) start[i + 1] += start[i];

    const targets = new Uint32Array(shownEdges);
    const cursor = new Uint32Array(n);
    for (let s = 0; s < total; s += 1) {
      if (synapses.enabled[s] === 0) continue;
      const pre = synapses.pre[s];
      const post = synapses.post[s];
      if (pre >= n || post >= n) continue;
      if (slotGroup[pre] < 0 || slotGroup[post] < 0) continue;
      targets[start[pre] + cursor[pre]] = post;
      cursor[pre] += 1;
    }

    // Stamped with the row index rather than cleared between rows, which keeps
    // this linear instead of O(rows · neurons).
    const seen = new Int32Array(n);
    seen.fill(-1);
    for (let u = 0; u < n; u += 1) {
      const gi = slotGroup[u];
      if (gi < 0) continue;
      const end = start[u + 1];
      for (let i = start[u]; i < end; i += 1) {
        const v = targets[i];
        if (seen[v] === u) continue;
        seen[v] = u;
        pairs[gi * size + slotGroup[v]] += 1;
      }
    }
  }

  return {
    mode,
    buildMs: performance.now() - started,
    groups,
    slotGroup,
    edges,
    pairs,
    weight,
    signed,
    shownEdges,
    hiddenEdges,
    omitted,
    neurons: n,
  };
}

/**
 * Project the matrix onto the quantity being drawn.
 *
 * Autapses are counted like any other connection, so the denominator of the
 * probability is every ordered pair including u = v. Excluding them from the
 * numerator but not the denominator — or the reverse — is the usual way this
 * statistic ends up lying about the diagonal.
 */
function deriveValues(
  data: MatrixData,
  metric: Metric,
  showSign: boolean,
  rowNormalise: boolean,
): ValueField {
  const size = data.groups.length;
  const values = new Float64Array(size * size);

  for (let i = 0; i < size; i += 1) {
    const rowSize = data.groups[i].size;
    for (let j = 0; j < size; j += 1) {
      const at = i * size + j;
      if (metric === 'prob') {
        const possible = rowSize * data.groups[j].size;
        values[at] = possible > 0 ? data.pairs[at] / possible : 0;
        continue;
      }
      const sum = showSign ? data.signed[at] : data.weight[at];
      const count = data.edges[at];
      values[at] = metric === 'mean' ? (count > 0 ? sum / count : 0) : sum;
    }
  }

  if (rowNormalise) {
    for (let i = 0; i < size; i += 1) {
      let total = 0;
      for (let j = 0; j < size; j += 1) total += Math.abs(values[i * size + j]);
      if (total <= 0) continue;
      for (let j = 0; j < size; j += 1) values[i * size + j] /= total;
    }
  }

  let max = 0;
  let minPositive = Infinity;
  let negatives = false;
  for (let k = 0; k < values.length; k += 1) {
    const value = values[k];
    if (value < 0) negatives = true;
    const magnitude = Math.abs(value);
    if (magnitude > max) max = magnitude;
    if (magnitude > 0 && magnitude < minPositive) minPositive = magnitude;
  }

  return {
    values,
    max,
    minPositive,
    negatives,
    unit: metric === 'prob' || rowNormalise ? '%' : 'nS',
  };
}

/* ----------------------------------------------------------------- canvas -- */

function layoutMatrix(width: number, height: number, count: number): Geometry {
  const bare = TICK + TICK_GAP;
  const labelled = bare + LABEL_ROOM;
  const sideFor = (gutter: number): number =>
    Math.min(width - gutter - PAD, height - gutter - PAD);

  let gutter = bare;
  let labels = false;
  if (count > 0 && sideFor(labelled) / count >= MIN_LABEL_CELL) {
    gutter = labelled;
    labels = true;
  }

  const side = Math.max(0, sideFor(gutter));
  return {
    x: gutter + Math.max(0, (width - gutter - PAD - side) / 2),
    y: gutter + Math.max(0, (height - gutter - PAD - side) / 2),
    side,
    cell: count > 0 ? side / count : 0,
    gutter,
    labels,
  };
}

/**
 * Cell boundaries along one axis. Snapped to whole pixels once a cell is wide
 * enough to survive it; below that the rounding would collapse neighbouring
 * columns onto each other and lose rows of the matrix entirely.
 */
function edgePositions(origin: number, side: number, count: number): Float64Array {
  const out = new Float64Array(count + 1);
  const cell = count > 0 ? side / count : 0;
  const snap = cell >= 2;
  for (let k = 0; k <= count; k += 1) {
    const position = origin + k * cell;
    out[k] = snap ? Math.round(position) : position;
  }
  return out;
}

function ellipsise(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let cut = text;
  while (cut.length > 1 && ctx.measureText(`${cut}…`).width > maxWidth) cut = cut.slice(0, -1);
  return `${cut}…`;
}

interface PaintOptions {
  data: MatrixData;
  field: ValueField;
  geometry: Geometry;
  log: boolean;
  diverging: boolean;
  mono: string;
}

function paintMatrix(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  options: PaintOptions,
): void {
  const { data, field, geometry, log, diverging, mono } = options;
  const count = data.groups.length;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = SURFACE_CSS;
  ctx.fillRect(0, 0, width, height);
  if (count === 0 || geometry.side <= 0) return;

  const cols = edgePositions(geometry.x, geometry.side, count);
  const rows = edgePositions(geometry.y, geometry.side, count);
  const left = cols[0];
  const top = rows[0];
  const right = cols[count];
  const bottom = rows[count];

  ctx.fillStyle = FIELD_CSS;
  ctx.fillRect(left, top, right - left, bottom - top);

  /* -- cells ------------------------------------------------------------- */

  const logMin = Math.log(field.minPositive);
  const logSpan = Math.log(field.max) - logMin;
  const usable = field.max > 0 && Number.isFinite(field.minPositive);

  if (usable) {
    for (let i = 0; i < count; i += 1) {
      const y = rows[i];
      const cellHeight = Math.max(rows[i + 1] - y, 0.5);
      for (let j = 0; j < count; j += 1) {
        const value = field.values[i * count + j];
        const magnitude = Math.abs(value);
        if (magnitude <= 0) continue;
        const t =
          log && logSpan > 0
            ? (Math.log(magnitude) - logMin) / logSpan
            : log
              ? 1
              : magnitude / field.max;
        const ramp = diverging ? (value < 0 ? NEGATIVE_CSS : POSITIVE_CSS) : SEQUENTIAL_CSS;
        ctx.fillStyle = ramp[rampIndex(t)];
        const x = cols[j];
        ctx.fillRect(x, y, Math.max(cols[j + 1] - x, 0.5), cellHeight);
      }
    }
  }

  /* -- grid, diagonal and axis strips ------------------------------------ */

  if (geometry.cell >= MIN_GRID_CELL) {
    ctx.fillStyle = GRID_CSS;
    for (let k = 1; k < count; k += 1) {
      ctx.fillRect(cols[k], top, 1, bottom - top);
      ctx.fillRect(left, rows[k], right - left, 1);
    }
  }

  if (count > 1) {
    // The autapse diagonal, marked because a connectome is read as blocks either
    // side of it and it is otherwise invisible in a sparse matrix.
    ctx.strokeStyle = DIAGONAL_CSS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(right, bottom);
    ctx.stroke();
  }

  const tickTop = top - TICK - TICK_GAP;
  const tickLeft = left - TICK - TICK_GAP;
  for (let k = 0; k < count; k += 1) {
    ctx.fillStyle = data.groups[k].color;
    ctx.fillRect(cols[k], tickTop, Math.max(cols[k + 1] - cols[k], 0.5), TICK);
    ctx.fillRect(tickLeft, rows[k], TICK, Math.max(rows[k + 1] - rows[k], 0.5));
  }

  /* -- labels ------------------------------------------------------------ */

  if (!geometry.labels) return;

  ctx.font = `9px ${mono}`;
  ctx.fillStyle = INK_MUTED_CSS;
  ctx.textBaseline = 'middle';

  // Labels are drawn outward from the tick strip, so the room they have is the
  // gutter less the strip, the gap and the 3px inset used below.
  const room = LABEL_ROOM - TICK_GAP - 1;
  for (let k = 0; k < count; k += 1) {
    const label = ellipsise(ctx, data.groups[k].label, room);
    ctx.textAlign = 'right';
    ctx.fillText(label, tickLeft - 3, (rows[k] + rows[k + 1]) / 2);

    // Column labels run up the page: the only orientation that fits a name above
    // a column narrow enough to be worth having a matrix for.
    ctx.save();
    ctx.translate((cols[k] + cols[k + 1]) / 2, tickTop - 3);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'left';
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
}

function paintOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  geometry: Geometry,
  count: number,
  hover: Cell | null,
): void {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (hover === null || count === 0 || geometry.side <= 0) return;
  if (hover.row >= count || hover.col >= count) return;

  const cols = edgePositions(geometry.x, geometry.side, count);
  const rows = edgePositions(geometry.y, geometry.side, count);
  const bandLeft = cols[0] - TICK - TICK_GAP;
  const bandTop = rows[0] - TICK - TICK_GAP;
  const right = cols[count];
  const bottom = rows[count];

  const x = cols[hover.col];
  const y = rows[hover.row];
  const w = Math.max(cols[hover.col + 1] - x, 1);
  const h = Math.max(rows[hover.row + 1] - y, 1);

  // The bands run back into the tick strips so the two identity colours the
  // readout names are the two the pointer is pointing at.
  ctx.fillStyle = HIGHLIGHT_CSS;
  ctx.fillRect(bandLeft, y, right - bandLeft, h);
  ctx.fillRect(x, bandTop, w, bottom - bandTop);

  ctx.strokeStyle = ACCENT_CSS;
  ctx.lineWidth = 1;
  ctx.strokeRect(
    Math.round(x) - 0.5,
    Math.round(y) - 0.5,
    Math.max(Math.round(w) + 1, 2),
    Math.max(Math.round(h) + 1, 2),
  );
}

function hitTest(geometry: Geometry, count: number, px: number, py: number): Cell | null {
  if (geometry.cell <= 0 || count === 0) return null;
  const col = Math.floor((px - geometry.x) / geometry.cell);
  const row = Math.floor((py - geometry.y) / geometry.cell);
  if (col < 0 || col >= count || row < 0 || row >= count) return null;
  return { row, col };
}

/* ------------------------------------------------------------- formatting -- */

function percent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '—';
  const scaled = value * 100;
  const precision = scaled !== 0 && Math.abs(scaled) < 0.1 ? 3 : digits;
  return `${scaled.toFixed(precision)}%`;
}

function formatValue(value: number, unit: '%' | 'nS'): string {
  if (!Number.isFinite(value)) return '—';
  if (unit === '%') return percent(value);
  return `${fixed(value, Math.abs(value) >= 100 ? 1 : 2)} nS`;
}

/* -------------------------------------------------------------- component -- */

const EMPTY_SELECTION: readonly NeuronId[] = [];

export interface ConnectivityMatrixProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Overrides the default placement. */
  className?: string;
}

/**
 * The connection-strength heatmap — the standard summary of a connectome.
 *
 * The matrix is painted into a canvas rather than laid out as DOM cells: the
 * by-neuron view is 16 384 cells at its default size and 262 144 at its cap, and
 * a node per cell would cost more to lay out than the whole simulation costs to
 * run. It is rebuilt only when the circuit or the grouping changes, never per
 * frame, and the hover overlay lives on its own canvas so following the pointer
 * never repaints the matrix underneath it.
 */
export function ConnectivityMatrix({ open = true, onClose, className }: ConnectivityMatrixProps) {
  const circuit = useEditor((s) => s.circuit);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [mode, setMode] = useState<Grouping>('population');
  const [metric, setMetric] = useState<Metric>('total');
  const [showSign, setShowSign] = useState(false);
  const [log, setLog] = useState(true);
  const [rowNormalise, setRowNormalise] = useState(false);

  const [data, setData] = useState<MatrixData | null>(null);
  const [pending, setPending] = useState(false);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<Cell | null>(null);
  const [surface, setSurface] = useState({ width: 0, height: 0, dpr: 1 });
  // Set when the browser refuses a 2D context, so the panel explains itself
  // rather than sitting there as an unexplained black rectangle.
  const [unavailable, setUnavailable] = useState(false);

  /**
   * The measured element is held as state rather than in a ref because it does
   * not exist yet on the commit that first runs the effect below.
   *
   * Until the first build lands, this component renders the loading panel,
   * which has no canvas host. A ref object would therefore still be null when
   * the sizing effect ran, and because `open` — its only other input — never
   * changes afterwards, the effect would never run again: the ResizeObserver
   * would never attach, the surface would stay at zero and the matrix would
   * never paint. A callback ref re-runs the effect on the commit that actually
   * attaches the node.
   */
  const [host, setHost] = useState<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const busyRef = useRef(false);
  const frameRef = useRef(0);
  const signatureRef = useRef('');
  /** Cost of the last completed build; null until one has run. */
  const costRef = useRef<number | null>(null);
  const hasDataRef = useRef(false);
  const viewRef = useRef<{ mode: Grouping; subset: readonly NeuronId[] } | null>(null);
  const documentRef = useRef<Pick<Circuit, 'neurons' | 'synapses' | 'populations'> | null>(null);
  const wantedModeRef = useRef<Grouping>(mode);

  // The by-neuron view is defined by the selection; the by-population view is
  // not, and must not rebuild every time the user clicks a cell.
  const subset = mode === 'neuron' ? selection : EMPTY_SELECTION;

  const build = useCallback(() => {
    // The latch below can only be held between scheduling the frame and running
    // it, so a request that arrives while it is held is absorbed by the pass
    // that has not run yet rather than dropped — which is what stops a grouping
    // change made in the same frame as an edit from being silently ignored.
    wantedModeRef.current = mode;
    if (busyRef.current) return;
    busyRef.current = true;
    setPending(true);
    // Deferring by a frame does two things: it lets the busy state paint before
    // the blocking pass, and it puts the pass after every effect in this commit
    // — including the ancestor one that reloads the engine, which runs *after*
    // this component's effects and would otherwise leave us reading last edit's
    // buffers.
    frameRef.current = requestAnimationFrame(() => {
      try {
        const engine = getEngine();
        const state = useEditor.getState();
        const wanted = wantedModeRef.current;
        // Stamped before the pass, so a build that throws cannot leave the poll
        // below retrying the same failing matrix twice a second.
        signatureRef.current = graphSignature(engine.buffers);
        const next = buildMatrix(
          engine,
          state.circuit,
          wanted,
          wanted === 'neuron' ? state.selection : EMPTY_SELECTION,
        );
        costRef.current = next.buildMs;
        hasDataRef.current = true;
        setData(next);
        setStale(false);
        setError(null);
      } catch (cause) {
        // The busy latch is what stops the poll from stampeding, so it has to be
        // released on the failing path too; without this one throw would leave
        // the panel spinning on a stale matrix forever.
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setPending(false);
        busyRef.current = false;
      }
    });
  }, [mode]);

  useEffect(() => {
    if (!open) return;

    const view = viewRef.current;
    const viewChanged = view === null || view.mode !== mode || view.subset !== subset;
    viewRef.current = { mode, subset };

    // The editor drafts copy-on-write, so any edit to a cell or a synapse
    // replaces one of these arrays; nothing else can change what the matrix
    // would say. Comparing them is what keeps reopening the panel, or a
    // re-render from an unrelated store field, off the rebuild path.
    const built = documentRef.current;
    const documentChanged =
      built === null ||
      built.neurons !== circuit.neurons ||
      built.synapses !== circuit.synapses ||
      built.populations !== circuit.populations;
    documentRef.current = {
      neurons: circuit.neurons,
      synapses: circuit.synapses,
      populations: circuit.populations,
    };

    if (viewChanged || !hasDataRef.current) {
      build();
    } else if (documentChanged) {
      if (costRef.current === null || costRef.current <= AUTO_BUILD_BUDGET_MS) build();
      else setStale(true);
    }

    // Engine reloads can also come from outside React — a document import, a
    // runtime reset — and those change nothing this effect depends on.
    const poll = () => {
      if (busyRef.current) return;
      if (graphSignature(getEngine().buffers) === signatureRef.current) return;
      build();
    };
    const id = setInterval(poll, SIGNATURE_POLL_MS);
    return () => {
      clearInterval(id);
      // A cancelled frame never reaches the `finally` above, so both the latch
      // and the spinner it drives have to be released here.
      cancelAnimationFrame(frameRef.current);
      busyRef.current = false;
      setPending(false);
    };
  }, [open, build, subset, circuit.neurons, circuit.synapses, circuit.populations]);

  // Indices are only meaningful against the matrix they were measured on.
  useEffect(() => {
    setHover(null);
  }, [data]);

  const field = useMemo(
    () =>
      data === null
        ? null
        : deriveValues(data, metric, showSign && metric !== 'prob', rowNormalise),
    [data, metric, showSign, rowNormalise],
  );

  const count = data === null ? 0 : data.groups.length;
  const geometry = useMemo(
    () => layoutMatrix(surface.width, surface.height, count),
    [surface.width, surface.height, count],
  );

  /* ------------------------------------------------------------- surface -- */

  const attachHost = useCallback((node: HTMLDivElement | null) => {
    setHost(node);
  }, []);

  useEffect(() => {
    if (!open || host === null) return;

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
  }, [open, host]);

  /* --------------------------------------------------------------- paint -- */

  const diverging = showSign && metric !== 'prob';

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!open || canvas === null || data === null || field === null) return;
    if (surface.width <= 0 || surface.height <= 0) return;

    const ctx = canvas.getContext('2d', { alpha: false });
    if (ctx === null) {
      setUnavailable(true);
      return;
    }
    setUnavailable(false);

    const deviceWidth = Math.max(2, Math.round(surface.width * surface.dpr));
    const deviceHeight = Math.max(2, Math.round(surface.height * surface.dpr));
    if (canvas.width !== deviceWidth || canvas.height !== deviceHeight) {
      canvas.width = deviceWidth;
      canvas.height = deviceHeight;
    }

    // The canvas carries `nf-numeric`, so its computed family is the same mono
    // face the rest of the chrome prints in.
    const mono = getComputedStyle(canvas).fontFamily || 'ui-monospace, monospace';
    paintMatrix(ctx, surface.width, surface.height, surface.dpr, {
      data,
      field,
      geometry,
      log,
      diverging,
      mono,
    });
  }, [open, data, field, geometry, log, diverging, surface]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!open || overlay === null || surface.width <= 0 || surface.height <= 0) return;
    const ctx = overlay.getContext('2d');
    if (ctx === null) return;

    const deviceWidth = Math.max(2, Math.round(surface.width * surface.dpr));
    const deviceHeight = Math.max(2, Math.round(surface.height * surface.dpr));
    if (overlay.width !== deviceWidth || overlay.height !== deviceHeight) {
      overlay.width = deviceWidth;
      overlay.height = deviceHeight;
    }
    paintOverlay(ctx, surface.width, surface.height, surface.dpr, geometry, count, hover);
  }, [open, geometry, count, hover, surface]);

  /* ------------------------------------------------------------ pointer -- */

  const trackPointer = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const hit = hitTest(geometry, count, event.clientX - rect.left, event.clientY - rect.top);
      setHover((current) => {
        if (current === null && hit === null) return current;
        if (current !== null && hit !== null && current.row === hit.row && current.col === hit.col) {
          return current;
        }
        return hit;
      });
    },
    [geometry, count],
  );

  const selectCell = useCallback(
    (cell: Cell) => {
      if (data === null) return;
      const engine = getEngine();
      const neurons = engine.buffers.neurons;
      const limit = Math.min(neurons.count, data.slotGroup.length);
      const ids: NeuronId[] = [];
      for (let slot = 0; slot < limit; slot += 1) {
        const group = data.slotGroup[slot];
        if (group !== cell.row && group !== cell.col) continue;
        const id = engine.idOf(slot);
        if (id !== null) ids.push(id as NeuronId);
      }
      if (ids.length > 0) select(ids);
    },
    [data, select],
  );

  const clickCell = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const hit = hitTest(geometry, count, event.clientX - rect.left, event.clientY - rect.top);
      if (hit !== null) selectCell(hit);
    },
    [geometry, count, selectCell],
  );

  const moveHover = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (count === 0) return;
      const current = hover ?? { row: 0, col: 0 };
      let next: Cell;
      switch (event.key) {
        case 'ArrowUp':
          next = { row: Math.max(0, current.row - 1), col: current.col };
          break;
        case 'ArrowDown':
          next = { row: Math.min(count - 1, current.row + 1), col: current.col };
          break;
        case 'ArrowLeft':
          next = { row: current.row, col: Math.max(0, current.col - 1) };
          break;
        case 'ArrowRight':
          next = { row: current.row, col: Math.min(count - 1, current.col + 1) };
          break;
        case 'Enter':
        case ' ':
          event.preventDefault();
          if (hover !== null) selectCell(hover);
          return;
        default:
          return;
      }
      event.preventDefault();
      setHover(next);
    },
    [count, hover, selectCell],
  );

  if (!open) return null;

  /* ---------------------------------------------------------------- view -- */

  // Edge-anchored with no transform, like every other panel. The `.nf-docked`
  // adapter neutralises a floating panel's position, inset and width when it is
  // mounted in a dock, but it cannot neutralise a transform — a centring
  // `-translate-x-1/2` survives docking and shifts the panel half its own width
  // out of the column, where the dock's `overflow-hidden` clips it.
  const placement = className ?? 'absolute top-3 bottom-3 right-3 w-[min(520px,calc(100vw-2rem))]';

  const header = (
    <PanelHeader
      title="Connectivity"
      subtitle={
        data === null
          ? undefined
          : `${grouped(count)} × ${grouped(count)} · ${grouped(data.shownEdges)} synapses`
      }
      icon={<Grid3x3 />}
      actions={
        <>
          {error !== null ? (
            <Tooltip content={`The last build failed: ${error}`}>
              <Badge variant="danger" size="sm" tabIndex={0}>
                failed
              </Badge>
            </Tooltip>
          ) : stale ? (
            <Tooltip content="The circuit changed after this build, and rebuilding it costs more than a frame. Rebuild to bring the matrix back in step.">
              <Badge variant="warning" size="sm" tabIndex={0}>
                stale
              </Badge>
            </Tooltip>
          ) : data !== null ? (
            <Tooltip content="Time taken by the last pass over the live buffers">
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(data.buildMs, 1)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Rebuild from the running network">
            <IconButton label="Rebuild matrix" size="sm" onClick={build} disabled={pending}>
              <RefreshCw className={pending ? 'animate-spin' : undefined} />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close connectivity panel" size="sm" onClick={onClose}>
              <X />
            </IconButton>
          ) : null}
        </>
      }
    />
  );

  const controls = (
    <div className="flex shrink-0 flex-col gap-1.5 border-b border-hairline px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="shrink-0 text-[9.5px] tracking-[0.08em] text-ink-faint uppercase">
          group
        </span>
        <SegmentedControl
          size="sm"
          value={mode}
          onChange={setMode}
          options={GROUPING_OPTIONS}
          aria-label="Matrix grouping"
        />
        <SegmentedControl
          size="sm"
          className="ml-auto"
          value={metric}
          onChange={setMetric}
          options={METRIC_OPTIONS}
          aria-label="Cell quantity"
        />
      </div>
      <div className="flex items-center gap-3">
        <Toggle
          label="signed"
          hint={
            metric === 'prob'
              ? 'A connection probability has no sign — switch to Σw or mean to colour by excitation and inhibition.'
              : 'Colour by net signed weight, inhibitory presynaptic cells counting negative. Switches the scale from sequential to diverging.'
          }
          // Reflects what the scale is actually doing, not the remembered
          // preference: with `p` selected the matrix has no sign to show.
          checked={diverging}
          disabled={metric === 'prob'}
          onChange={setShowSign}
        />
        <Toggle
          label="log"
          hint="Map magnitude logarithmically between the smallest and largest non-zero cell. Synaptic weight distributions are heavy-tailed, and on a linear scale almost every cell lands in the bottom decile of the ramp."
          checked={log}
          onChange={setLog}
        />
        <Toggle
          label="row 1.0"
          hint="Normalise each row so its magnitudes sum to one, which is how targeting patterns are compared between groups of very different size."
          checked={rowNormalise}
          onChange={setRowNormalise}
        />
      </div>
    </div>
  );

  if (data === null || field === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        {controls}
        {error === null ? (
          <EmptyState
            icon={<Spinner size={18} />}
            title="Building the matrix"
            description="Summarising every connection in the running network."
          />
        ) : (
          <EmptyState
            icon={<TriangleAlert className="text-danger" />}
            title="Could not build the matrix"
            description={error}
            action={
              <Button size="sm" icon={<RefreshCw />} onClick={build} loading={pending}>
                Try again
              </Button>
            }
          />
        )}
      </Panel>
    );
  }

  const notice =
    data.neurons === 0
      ? 'No neurons yet — place cells and wire them together, and their connectivity appears here.'
      : count === 0
        ? 'Nothing to group. Select the cells whose adjacency you want to read.'
        : data.mode === 'population' && count < 2
          ? 'This circuit has no populations to compare — group by neuron to read the raw adjacency.'
          : unavailable
            ? 'This browser would not give the matrix a 2D canvas, so it cannot be drawn.'
            : null;

  const hovered =
    hover !== null && hover.row < count && hover.col < count
      ? {
          cell: hover,
          source: data.groups[hover.row],
          target: data.groups[hover.col],
          at: hover.row * count + hover.col,
        }
      : null;

  const scaleLabel = diverging ? 'diverging · net E/I sign' : 'sequential · magnitude';
  const peak = formatValue(field.max, field.unit);
  const rangeLabel =
    field.max <= 0
      ? 'nothing connected'
      : diverging
        ? // Both arms of the diverging ramp are driven by the same magnitude, so
          // the bar runs from −max on the left, through the field colour at
          // zero, to +max on the right. Printing the magnitude range alone would
          // put one end's numbers on a two-ended bar.
          `−${peak} → +${peak}`
        : log && Number.isFinite(field.minPositive)
          ? `${formatValue(field.minPositive, field.unit)} → ${peak}`
          : `0 → ${peak}`;

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      {header}
      {controls}

      <div
        ref={attachHost}
        tabIndex={0}
        aria-label={`Connectivity matrix, ${count} by ${count}, grouped by ${data.mode}, ${grouped(
          data.shownEdges,
        )} synapses. Arrow keys move the cursor, enter selects the cells involved.`}
        onPointerMove={trackPointer}
        onPointerLeave={() => setHover(null)}
        onPointerDown={clickCell}
        onKeyDown={moveHover}
        className="relative min-h-[220px] w-full flex-1 cursor-crosshair touch-none focus-visible:outline-1"
      >
        <canvas ref={canvasRef} aria-hidden className="nf-numeric absolute inset-0 block size-full" />
        <canvas
          ref={overlayRef}
          aria-hidden
          className="pointer-events-none absolute inset-0 block size-full"
        />
        {notice !== null ? (
          // Opaque rather than translucent: a degenerate matrix painted behind
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

      {/* -- legend and readout ------------------------------------------- */}

      <div className="flex shrink-0 flex-col gap-1.5 border-t border-hairline px-3 py-2">
        <div className="flex items-center gap-2">
          <span
            aria-hidden
            className="h-2 min-w-0 flex-1 rounded-[2px] ring-1 ring-white/10"
            style={{ background: diverging ? DIVERGING_GRADIENT : SEQUENTIAL_GRADIENT }}
          />
          <span className="nf-numeric shrink-0 text-[9.5px] text-ink-faint">{rangeLabel}</span>
        </div>
        <div className="flex items-baseline justify-between gap-2 text-[9.5px] text-ink-faint">
          <span className="truncate">
            {scaleLabel} · {log ? 'log' : 'linear'}
            {rowNormalise ? ' · row-normalised' : ''}
          </span>
          <span className="shrink-0">rows source → columns target</span>
        </div>

        <div className="mt-0.5 flex min-h-[30px] flex-col justify-center gap-1">
          {hovered === null ? (
            <p className="text-[10.5px] leading-tight text-ink-faint">
              {/* Each clause is printed only when it has something to report:
                  the two are independent, and an unconnected network can omit
                  cells without hiding a single synapse. */}
              {data.hiddenEdges > 0 ? (
                <>{grouped(data.hiddenEdges)} synapses reach outside this matrix. </>
              ) : null}
              {data.omitted > 0 ? <>{grouped(data.omitted)} cells are not shown. </> : null}
              Hover a cell to read its connection statistics; click to select the cells involved.
            </p>
          ) : (
            <>
              <div className="flex items-center gap-1.5 text-[10.5px]">
                <Swatch color={hovered.source.color} />
                <span className="min-w-0 flex-1 truncate text-ink">{hovered.source.label}</span>
                <span aria-hidden className="shrink-0 text-ink-faint">
                  →
                </span>
                <Swatch color={hovered.target.color} />
                <span className="min-w-0 flex-1 truncate text-ink">{hovered.target.label}</span>
                <span className="nf-numeric shrink-0 text-ink">
                  {formatValue(field.values[hovered.at], field.unit)}
                </span>
              </div>
              <div className="nf-numeric flex items-baseline gap-2 text-[9.5px] text-ink-faint">
                <span>{grouped(data.edges[hovered.at])} syn</span>
                <span>
                  {grouped(data.pairs[hovered.at])}/
                  {compact(hovered.source.size * hovered.target.size)} pairs
                </span>
                <span>
                  μ{' '}
                  {data.edges[hovered.at] > 0
                    ? `${fixed(
                        (diverging ? data.signed[hovered.at] : data.weight[hovered.at]) /
                          data.edges[hovered.at],
                        2,
                      )} nS`
                    : '—'}
                </span>
                <span className="ml-auto">
                  p{' '}
                  {percent(
                    hovered.source.size * hovered.target.size > 0
                      ? data.pairs[hovered.at] / (hovered.source.size * hovered.target.size)
                      : 0,
                  )}
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ parts -- */

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
      style={{ backgroundColor: color }}
    />
  );
}

interface ToggleProps {
  label: string;
  hint: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}

function Toggle({ label, hint, checked, disabled = false, onChange }: ToggleProps) {
  return (
    <Tooltip content={hint} side="top">
      <label
        className={cn(
          'flex cursor-pointer items-center gap-1.5 text-[9.5px] tracking-[0.08em] uppercase select-none',
          disabled ? 'cursor-not-allowed text-ink-faint/60' : 'text-ink-muted hover:text-ink',
        )}
      >
        <Switch
          size="sm"
          checked={checked}
          onCheckedChange={onChange}
          disabled={disabled}
          aria-label={label}
        />
        {label}
      </label>
    </Tooltip>
  );
}
