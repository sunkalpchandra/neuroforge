'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { BarChart3, RefreshCw, X } from 'lucide-react';
import {
  Badge,
  EmptyState,
  IconButton,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  SegmentedControl,
  Select,
  SelectItem,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import {
  MORPHOLOGY_ARCHETYPES,
  NEURON_MODEL_KINDS,
  NEURON_MODEL_LABELS,
  POLARITY_COLORS,
  RECEPTOR_COLORS,
  RECEPTOR_KINDS,
  RECEPTOR_LABELS,
  identityColorHex,
} from '@neuroforge/shared';
import type {
  Neuron,
  NeuronId,
  NeuronModelKind,
  Population,
  Synapse,
} from '@neuroforge/shared';

import { PARAM_FIELDS, readParam } from '@/components/inspector/param-fields';
import type { ParamField } from '@/components/inspector/param-fields';
import { fixed, grouped } from '@/lib/format';
import { graphSignature } from '@/lib/graph-metrics';
import { getEngine } from '@/lib/runtime';

/**
 * Poll cadence for the buffer signature. Two integer reads; effectively free.
 *
 * The engine is loaded by an effect in an ancestor which may commit after this
 * panel mounts, so the firing rates are not readable on the first pass. Polling
 * is what makes the panel correct regardless of that ordering.
 */
const SIGNATURE_POLL_MS = 500;

/** Maximum columns in a histogram. More than this and a column is sub-pixel. */
const HIST_BINS = 40;

const CHART_W = 320;
const CHART_H = 54;
/** Rows left below the plot for the axis rule. */
const BASELINE = CHART_H - 2;
/** Rows left above the tallest column, so a peak bar does not touch the top. */
const PLOT_TOP = 4;
const PLOT_H = BASELINE - PLOT_TOP;

/** Cells listed when a bin is opened. Enough to recognise a tail, not a table. */
const FOCUS_ROW_LIMIT = 16;

/* -------------------------------------------------------------------- model -- */

type Scope = 'selection' | 'circuit';
type CountScale = 'linear' | 'log';

/** What one sample of a distribution is, which is what selecting a bar reaches. */
type SampleKind = 'neuron' | 'synapse';

interface Distribution {
  key: string;
  label: string;
  /** Physical unit; empty string for a dimensionless count. */
  unit: string;
  /** Caption for the value axis, unit included. */
  axis: string;
  hint: string;
  kind: SampleKind;
  /** Digits every readout of this quantity is rendered with. */
  precision: number;
  /** Samples only ever take whole values, so bins are integer-aligned. */
  integral: boolean;

  n: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  /**
   * Population standard deviation. The subset is enumerated in full rather than
   * sampled from a larger pool, so there is no degree of freedom to lose.
   */
  sd: number;
  p05: number;
  p95: number;

  /** One value per sample. */
  values: Float32Array;
  /** Document index of the neuron or synapse each sample was read from. */
  owners: Int32Array;

  binWidth: number;
  binCount: number;
  counts: Uint32Array;
  peak: number;
}

interface CategoryRow {
  key: string;
  label: string;
  color: string;
  count: number;
  share: number;
  /** Document indices of the members, matching the chart's `kind`. */
  members: Int32Array;
}

interface CategoryChart {
  key: string;
  label: string;
  hint: string;
  kind: SampleKind;
  total: number;
  rows: readonly CategoryRow[];
}

/** A membrane model present in the subset, and how many cells carry it. */
interface ModelGroup {
  kind: NeuronModelKind;
  label: string;
  cells: number;
}

interface ParamSeries {
  field: ParamField;
  distribution: Distribution;
  /** Every cell holds the same value, so there is no distribution to draw. */
  constant: boolean;
}

interface StatsModel {
  computeMs: number;
  scope: Scope;
  /** Cells in the subset. */
  cells: number;
  /** Synapses with at least one endpoint in the subset. */
  synapses: number;

  rate: Distribution;
  inDegree: Distribution;
  outDegree: Distribution;
  weight: Distribution;
  delay: Distribution;

  /** Membrane models present in the subset, most populous first. */
  models: readonly ModelGroup[];
  /** The model whose parameters were measured, or null when the subset is empty. */
  paramKind: NeuronModelKind | null;
  params: readonly ParamSeries[];

  categories: readonly CategoryChart[];
}

/**
 * A bin the user reached into: the members captured at the moment it was picked.
 *
 * Members are captured rather than recomputed because picking a bin also selects
 * its cells, and in `selection` scope that narrows the subset every distribution
 * is built from. Recomputing would dissolve the very bin the user just opened.
 */
interface Focus {
  /** Distribution or category chart the bin belongs to. */
  key: string;
  kind: SampleKind;
  label: string;
  unit: string;
  precision: number;
  /** Value range of the bin; NaN for a categorical row, which spans no range. */
  lo: number;
  hi: number;
  members: Int32Array;
  /** Value per member, parallel to `members`; null for a categorical row. */
  values: Float32Array | null;
}

/** One line of the opened-bin list. */
interface FocusRow {
  key: string;
  /** Identity colour of the cell, or of the presynaptic cell for a synapse. */
  color: string;
  /** Identity colour of the postsynaptic cell; null for a cell row. */
  toColor: string | null;
  label: string;
  detail: string | null;
  value: number | null;
  ids: readonly NeuronId[];
}

/* --------------------------------------------------------------------- math -- */

/**
 * Scratch used for the percentile sort.
 *
 * Percentiles need the samples in order, and every distribution in the panel
 * wants them. Sorting into a shared buffer that is only ever grown means a pass
 * over a 100k-cell circuit allocates one array rather than one per series.
 */
let sortScratch = new Float64Array(1);

function ensureScratch(length: number): Float64Array {
  if (sortScratch.length < length) sortScratch = new Float64Array(length);
  return sortScratch;
}

/** Linear-interpolated quantile of an ascending run of `n` values. */
function quantile(sorted: Float64Array, n: number, q: number): number {
  if (n === 0) return 0;
  if (n === 1) return sorted[0];
  const position = (n - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/** Bin a value falls in. Values outside the range clamp rather than escape. */
function binOf(distribution: Distribution, value: number): number {
  const { binCount, binWidth, min } = distribution;
  if (binCount <= 1 || binWidth <= 0) return 0;
  const index = Math.floor((value - min) / binWidth);
  if (index < 0) return 0;
  return index >= binCount ? binCount - 1 : index;
}

/** Lower edge of a bin, in value units. */
function binLow(distribution: Distribution, bin: number): number {
  return distribution.min + bin * distribution.binWidth;
}

/**
 * Upper edge of a bin. Integer-valued data owns whole numbers, so its bins are
 * closed on both sides; continuous bins are half-open and quote the next edge.
 */
function binHigh(distribution: Distribution, bin: number): number {
  const low = binLow(distribution, bin);
  return distribution.integral ? low + distribution.binWidth - 1 : low + distribution.binWidth;
}

function binLabel(distribution: Distribution, bin: number): string {
  const low = binLow(distribution, bin);
  const high = binHigh(distribution, bin);
  const unit = distribution.unit === '' ? '' : ` ${distribution.unit}`;
  if (distribution.integral) {
    if (distribution.binWidth <= 1) return `${grouped(low)}${unit}`;
    return `${grouped(low)}–${grouped(high)}${unit}`;
  }
  if (distribution.binCount <= 1) return `${fixed(low, distribution.precision)}${unit}`;
  return `${fixed(low, distribution.precision)}–${fixed(high, distribution.precision)}${unit}`;
}

interface Samples {
  values: Float32Array;
  owners: Int32Array;
}

/**
 * Read one number per subset member.
 *
 * Non-finite readings are dropped rather than binned: a neuron whose live slot
 * has not been allocated yet has no firing rate, and folding that in as a zero
 * would put a phantom spike at the bottom of the histogram. The two arrays stay
 * parallel because a dropped sample advances neither cursor.
 */
function collect(members: Int32Array, read: (index: number) => number): Samples {
  const values = new Float32Array(members.length);
  const owners = new Int32Array(members.length);
  let kept = 0;
  for (let i = 0; i < members.length; i += 1) {
    const index = members[i];
    const value = read(index);
    if (!Number.isFinite(value)) continue;
    values[kept] = value;
    owners[kept] = index;
    kept += 1;
  }
  return { values: values.subarray(0, kept), owners: owners.subarray(0, kept) };
}

interface DistributionSpec {
  key: string;
  label: string;
  unit: string;
  axis: string;
  hint: string;
  kind: SampleKind;
  precision: number;
  integral: boolean;
}

function buildDistribution(spec: DistributionSpec, samples: Samples): Distribution {
  const { values, owners } = samples;
  const n = values.length;

  if (n === 0) {
    return {
      ...spec,
      n: 0,
      min: 0,
      max: 0,
      mean: 0,
      median: 0,
      sd: 0,
      p05: 0,
      p95: 0,
      values,
      owners,
      binWidth: 0,
      binCount: 0,
      counts: new Uint32Array(0),
      peak: 0,
    };
  }

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const value = values[i];
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
  }
  const mean = sum / n;

  let squares = 0;
  for (let i = 0; i < n; i += 1) {
    const delta = values[i] - mean;
    squares += delta * delta;
  }
  const sd = Math.sqrt(squares / n);

  const sorted = ensureScratch(n).subarray(0, n);
  for (let i = 0; i < n; i += 1) sorted[i] = values[i];
  sorted.sort();

  let binWidth: number;
  let binCount: number;
  if (spec.integral) {
    // Whole-numbered data gets whole-numbered bins, so a column stands for an
    // exact run of degrees rather than a fractional slice of one.
    const span = max - min + 1;
    binWidth = Math.max(1, Math.ceil(span / HIST_BINS));
    binCount = Math.max(1, Math.ceil(span / binWidth));
  } else if (max === min) {
    binWidth = 0;
    binCount = 1;
  } else {
    binCount = HIST_BINS;
    binWidth = (max - min) / HIST_BINS;
  }

  const counts = new Uint32Array(binCount);
  const shape: Distribution = {
    ...spec,
    n,
    min,
    max,
    mean,
    median: quantile(sorted, n, 0.5),
    sd,
    p05: quantile(sorted, n, 0.05),
    p95: quantile(sorted, n, 0.95),
    values,
    owners,
    binWidth,
    binCount,
    counts,
    peak: 0,
  };

  for (let i = 0; i < n; i += 1) counts[binOf(shape, values[i])] += 1;
  let peak = 0;
  for (let i = 0; i < binCount; i += 1) if (counts[i] > peak) peak = counts[i];
  shape.peak = peak;
  return shape;
}

interface CategorySpec {
  key: string;
  label: string;
  hint: string;
  kind: SampleKind;
}

/**
 * Tally members into named buckets and materialise each bucket's membership.
 *
 * Two passes rather than an array of arrays: the first counts so the second can
 * write straight into an exactly-sized Int32Array, which is what keeps a
 * breakdown of a hundred thousand cells from allocating a hundred thousand
 * boxed numbers.
 */
function buildCategory(
  spec: CategorySpec,
  members: Int32Array,
  bucketOf: (index: number) => string,
  describe: (bucket: string) => { label: string; color: string },
  order: readonly string[] | null,
): CategoryChart {
  const counts = new Map<string, number>();
  for (let i = 0; i < members.length; i += 1) {
    const bucket = bucketOf(members[i]);
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }

  const keys =
    order === null
      ? [...counts.keys()].sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0))
      : order.filter((key) => counts.has(key));
  // A bucket outside a fixed order still has to appear, or the percentages lie.
  if (order !== null) {
    for (const key of counts.keys()) if (!keys.includes(key)) keys.push(key);
  }

  const slots = new Map<string, { row: number; cursor: number }>();
  const rows: CategoryRow[] = keys.map((key, row) => {
    const count = counts.get(key) ?? 0;
    slots.set(key, { row, cursor: 0 });
    const { label, color } = describe(key);
    return {
      key,
      label,
      color,
      count,
      share: members.length > 0 ? count / members.length : 0,
      members: new Int32Array(count),
    };
  });

  for (let i = 0; i < members.length; i += 1) {
    const index = members[i];
    const slot = slots.get(bucketOf(index));
    if (slot === undefined) continue;
    rows[slot.row].members[slot.cursor] = index;
    slot.cursor += 1;
  }

  return { ...spec, total: members.length, rows };
}

function titleCase(value: string): string {
  return value.length === 0 ? value : value[0].toUpperCase() + value.slice(1);
}

function percent(share: number, digits = 1): string {
  if (!Number.isFinite(share)) return '—';
  return `${(share * 100).toFixed(digits)}%`;
}

/** A value with its unit, at the precision the quantity is quoted in. */
function withUnit(value: number, unit: string, precision: number): string {
  const text = fixed(value, precision);
  return unit === '' ? text : `${text} ${unit}`;
}

/* ------------------------------------------------------------------ compute -- */

const EMPTY_SUBSET = new Int32Array(0);

/**
 * Every distribution in the panel, from one pass over the document plus one read
 * of the live rate column.
 *
 * Structure and parameters come from the document because that is what the
 * selection names and what the categorical breakdowns are defined in terms of —
 * an archetype and a population have no representation in the simulation
 * buffers. Firing rate is the one genuinely live quantity and is read from the
 * engine, per cell, through the id-to-slot map.
 */
function computeStats(
  neurons: readonly Neuron[],
  synapses: readonly Synapse[],
  populations: readonly Population[],
  selection: readonly NeuronId[],
  preferred: Scope,
  requestedModel: NeuronModelKind | null,
): StatsModel {
  const started = performance.now();
  const neuronCount = neurons.length;

  const indexOf = new Map<string, number>();
  for (let i = 0; i < neuronCount; i += 1) indexOf.set(neurons[i].id, i);

  /* -- subset ------------------------------------------------------------- */

  let subset = EMPTY_SUBSET;
  let scope: Scope = 'circuit';
  if (preferred === 'selection' && selection.length > 0) {
    const resolved = new Int32Array(selection.length);
    let kept = 0;
    for (const id of selection) {
      const index = indexOf.get(id);
      if (index === undefined) continue;
      resolved[kept] = index;
      kept += 1;
    }
    if (kept > 0) {
      subset = resolved.subarray(0, kept);
      scope = 'selection';
    }
  }
  if (scope === 'circuit') {
    subset = new Int32Array(neuronCount);
    for (let i = 0; i < neuronCount; i += 1) subset[i] = i;
  }

  const inSubset = new Uint8Array(neuronCount);
  for (let i = 0; i < subset.length; i += 1) inSubset[subset[i]] = 1;

  /* -- degrees and incident wiring ---------------------------------------- */

  // This is the counting pass of the CSR build in `graph-metrics`: one sweep
  // that resolves both endpoints and increments a per-node total. The prefix sum
  // and the index fill are skipped because nothing here walks neighbours — only
  // the row lengths are wanted, and materialising the adjacency to read them
  // back would double the memory for no answer.
  const inDegrees = new Uint32Array(neuronCount);
  const outDegrees = new Uint32Array(neuronCount);
  const incidentBuffer = new Int32Array(synapses.length);
  let incidentCount = 0;

  for (let s = 0; s < synapses.length; s += 1) {
    const synapse = synapses[s];
    if (!synapse.enabled) continue;
    const pre = indexOf.get(synapse.source);
    const post = indexOf.get(synapse.target);
    // A synapse referencing a deleted cell is dropped rather than counted
    // against slot zero, which is what the engine does when it loads.
    if (pre === undefined || post === undefined) continue;
    outDegrees[pre] += 1;
    inDegrees[post] += 1;
    if (inSubset[pre] === 1 || inSubset[post] === 1) {
      incidentBuffer[incidentCount] = s;
      incidentCount += 1;
    }
  }
  const incident = incidentBuffer.subarray(0, incidentCount);

  /* -- activity ------------------------------------------------------------ */

  const engine = getEngine();
  const live = engine.buffers.neurons;
  const rateColumn = live.rate;
  const liveCount = live.count;

  const rate = buildDistribution(
    {
      key: 'rate',
      label: 'Firing rate',
      unit: 'Hz',
      axis: 'firing rate (Hz)',
      hint: 'Exponentially smoothed spike rate read from the running network. Cells whose slot has not been allocated yet are excluded rather than counted as silent.',
      kind: 'neuron',
      precision: 2,
      integral: false,
    },
    collect(subset, (index) => {
      const slot = engine.slotOf(neurons[index].id);
      return slot >= 0 && slot < liveCount ? rateColumn[slot] : Number.NaN;
    }),
  );

  /* -- topology ------------------------------------------------------------ */

  const inDegree = buildDistribution(
    {
      key: 'in-degree',
      label: 'In-degree',
      unit: '',
      axis: 'afferent synapses per cell',
      hint: 'Enabled synapses arriving at each cell, counted over the whole circuit rather than only within the subset.',
      kind: 'neuron',
      precision: 0,
      integral: true,
    },
    collect(subset, (index) => inDegrees[index]),
  );

  const outDegree = buildDistribution(
    {
      key: 'out-degree',
      label: 'Out-degree',
      unit: '',
      axis: 'efferent synapses per cell',
      hint: 'Enabled synapses leaving each cell, counted over the whole circuit rather than only within the subset.',
      kind: 'neuron',
      precision: 0,
      integral: true,
    },
    collect(subset, (index) => outDegrees[index]),
  );

  const weight = buildDistribution(
    {
      key: 'weight',
      label: 'Synaptic weight',
      unit: 'nS',
      axis: 'peak conductance (nS)',
      hint: 'Peak conductance as written in the document. Plasticity rewrites the running values, which are not what this measures.',
      kind: 'synapse',
      precision: 2,
      integral: false,
    },
    collect(incident, (index) => synapses[index].weight),
  );

  const delay = buildDistribution(
    {
      key: 'delay',
      label: 'Conduction delay',
      unit: 'ms',
      axis: 'axonal delay (ms)',
      hint: 'Time between a presynaptic spike and its arrival at the postsynaptic cell.',
      kind: 'synapse',
      precision: 2,
      integral: false,
    },
    collect(incident, (index) => synapses[index].delay),
  );

  /* -- membrane parameters ------------------------------------------------- */

  const modelCounts = new Map<NeuronModelKind, number>();
  for (let i = 0; i < subset.length; i += 1) {
    const kind = neurons[subset[i]].params.kind;
    modelCounts.set(kind, (modelCounts.get(kind) ?? 0) + 1);
  }
  const models: ModelGroup[] = NEURON_MODEL_KINDS.filter((kind) => modelCounts.has(kind))
    .map((kind) => ({
      kind,
      label: NEURON_MODEL_LABELS[kind],
      cells: modelCounts.get(kind) ?? 0,
    }))
    .sort((a, b) => b.cells - a.cells);

  // The fields a circuit's cells actually carry, not the fields LIF happens to
  // have: a Hodgkin-Huxley network has channel conductances and no threshold,
  // and asking it for one produces a column of zeros that means nothing.
  const paramKind =
    requestedModel !== null && modelCounts.has(requestedModel)
      ? requestedModel
      : (models[0]?.kind ?? null);

  let params: ParamSeries[] = [];
  if (paramKind !== null) {
    const ofKind = new Int32Array(modelCounts.get(paramKind) ?? 0);
    let kept = 0;
    for (let i = 0; i < subset.length; i += 1) {
      const index = subset[i];
      if (neurons[index].params.kind !== paramKind) continue;
      ofKind[kept] = index;
      kept += 1;
    }
    params = PARAM_FIELDS[paramKind].map((field) => {
      const distribution = buildDistribution(
        {
          key: `param:${paramKind}:${field.key}`,
          label: field.label,
          unit: field.unit,
          axis: field.unit === '' ? field.label.toLowerCase() : `${field.label.toLowerCase()} (${field.unit})`,
          hint: field.hint,
          kind: 'neuron',
          precision: field.precision,
          integral: false,
        },
        collect(ofKind, (index) => readParam(neurons[index].params, field.key)),
      );
      return {
        field,
        distribution,
        constant: distribution.n > 0 && distribution.min === distribution.max,
      };
    });
  }

  /* -- categorical breakdowns ---------------------------------------------- */

  const populationById = new Map<string, Population>();
  for (const population of populations) populationById.set(population.id, population);

  const categories: CategoryChart[] = [
    buildCategory(
      {
        key: 'model',
        label: 'Model kind',
        hint: 'Membrane model integrating each cell.',
        kind: 'neuron',
      },
      subset,
      (index) => neurons[index].params.kind,
      (bucket) => ({
        label: NEURON_MODEL_LABELS[bucket as NeuronModelKind] ?? bucket,
        color: 'var(--color-accent)',
      }),
      NEURON_MODEL_KINDS,
    ),
    buildCategory(
      {
        key: 'polarity',
        label: 'Polarity',
        hint: 'Sign of the transmitter each cell releases.',
        kind: 'neuron',
      },
      subset,
      (index) => neurons[index].polarity,
      (bucket) => ({
        label: titleCase(bucket),
        color:
          bucket === 'inhibitory' ? POLARITY_COLORS.inhibitory : POLARITY_COLORS.excitatory,
      }),
      ['excitatory', 'inhibitory'],
    ),
    buildCategory(
      {
        key: 'archetype',
        label: 'Archetype',
        hint: 'Morphological class the procedural glyph is generated from.',
        kind: 'neuron',
      },
      subset,
      (index) => neurons[index].morphology.archetype,
      (bucket) => ({ label: titleCase(bucket), color: 'var(--color-accent)' }),
      MORPHOLOGY_ARCHETYPES,
    ),
    buildCategory(
      {
        key: 'population',
        label: 'Population',
        hint: 'Named group each cell was instantiated as part of.',
        kind: 'neuron',
      },
      subset,
      (index) => neurons[index].population ?? '',
      (bucket) => {
        const population = bucket === '' ? undefined : populationById.get(bucket);
        if (population === undefined) {
          return { label: 'Unassigned', color: 'var(--color-ink-faint)' };
        }
        // The population's own hue, from the same generator its cells are drawn
        // with, so a row here and a cluster in the scene are the same object.
        return {
          label: population.name,
          color: population.color ?? identityColorHex(population.morphology.seed),
        };
      },
      null,
    ),
    buildCategory(
      {
        key: 'receptor',
        label: 'Receptor',
        hint: 'Postsynaptic receptor kinetics of the incident synapses.',
        kind: 'synapse',
      },
      incident,
      (index) => synapses[index].receptor,
      (bucket) => ({
        label: RECEPTOR_LABELS[bucket as keyof typeof RECEPTOR_LABELS] ?? bucket,
        color: RECEPTOR_COLORS[bucket as keyof typeof RECEPTOR_COLORS] ?? 'var(--color-accent)',
      }),
      RECEPTOR_KINDS,
    ),
  ];

  return {
    computeMs: performance.now() - started,
    scope,
    cells: subset.length,
    synapses: incidentCount,
    rate,
    inDegree,
    outDegree,
    weight,
    delay,
    models,
    paramKind,
    params,
    categories,
  };
}

/* -------------------------------------------------------------------- panel -- */

const SCALE_OPTIONS = [
  { value: 'linear' as const, label: 'linear', title: 'Linear count axis' },
  { value: 'log' as const, label: 'log', title: 'Logarithmic count axis' },
];

export interface StatsPanelProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Replaces the default docking entirely. */
  className?: string;
}

/**
 * Distributions of every attribute, over the selection or the whole circuit.
 *
 * A connectome's interesting attributes are all heavy-tailed — a handful of hub
 * cells carry orders of magnitude more wiring than the median, and a handful of
 * synapses carry most of the conductance. A mean is close to useless against
 * that, which is why everything here is a shape with its percentiles marked, and
 * why a bar is a control: reaching into the tail and selecting the dozen cells
 * that live there is the question a statistics view exists to answer.
 *
 * The whole model is one `useMemo` over the document arrays and the selection.
 * Those arrays are sliced out of the store rather than taken from `circuit`,
 * because the store republishes the document on every camera orbit frame and
 * re-measuring a hundred thousand cells at 144 Hz would cost more than the
 * simulation being measured.
 */
export function StatsPanel({ open = true, onClose, className }: StatsPanelProps) {
  const neurons = useEditor((s) => s.circuit.neurons);
  const synapses = useEditor((s) => s.circuit.synapses);
  const populations = useEditor((s) => s.circuit.populations);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [preferred, setPreferred] = useState<Scope>('selection');
  const [scale, setScale] = useState<CountScale>('log');
  const [requestedModel, setRequestedModel] = useState<NeuronModelKind | null>(null);
  const [focus, setFocus] = useState<Focus | null>(null);
  /**
   * Bumped when the live buffers are worth re-reading. Firing rate is the only
   * quantity here that changes without the document changing, and it moves every
   * step — so it is sampled on demand rather than followed.
   */
  const [sample, setSample] = useState(0);

  useEffect(() => {
    if (!open) return;
    let last = '';
    const poll = () => {
      const signature = graphSignature(getEngine().buffers);
      if (signature === last) return;
      last = signature;
      setSample((value) => value + 1);
    };
    poll();
    const id = setInterval(poll, SIGNATURE_POLL_MS);
    return () => clearInterval(id);
  }, [open]);

  // An opened bin holds document indices. Any edit to the collections can move
  // what those indices name, so the capture is dropped rather than allowed to
  // point at whichever cell inherited the slot.
  useEffect(() => {
    setFocus(null);
  }, [neurons, synapses]);

  const stats = useMemo(() => {
    if (!open) return null;
    // `sample` is a dependency rather than a value: it is what makes a refresh
    // re-read the live rate column.
    void sample;
    return computeStats(neurons, synapses, populations, selection, preferred, requestedModel);
  }, [open, neurons, synapses, populations, selection, preferred, requestedModel, sample]);

  const focusRows = useMemo<readonly FocusRow[]>(() => {
    if (focus === null) return [];
    const limit = Math.min(focus.members.length, FOCUS_ROW_LIMIT);
    const rows: FocusRow[] = [];
    for (let i = 0; i < limit; i += 1) {
      const index = focus.members[i];
      const value = focus.values === null ? null : focus.values[i];
      if (focus.kind === 'neuron') {
        const neuron = neurons[index];
        if (neuron === undefined) continue;
        rows.push({
          key: neuron.id,
          color: identityColorHex(neuron.morphology.seed),
          toColor: null,
          label: neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8),
          detail: NEURON_MODEL_LABELS[neuron.params.kind],
          value,
          ids: [neuron.id],
        });
        continue;
      }
      const synapse = synapses[index];
      if (synapse === undefined) continue;
      const source = neurons.find((candidate) => candidate.id === synapse.source);
      const target = neurons.find((candidate) => candidate.id === synapse.target);
      if (source === undefined || target === undefined) continue;
      rows.push({
        key: synapse.id,
        color: identityColorHex(source.morphology.seed),
        toColor: identityColorHex(target.morphology.seed),
        label: source.label.length > 0 ? source.label : source.id.slice(0, 8),
        detail: target.label.length > 0 ? target.label : target.id.slice(0, 8),
        value,
        ids: [source.id, target.id],
      });
    }
    return rows;
  }, [focus, neurons, synapses]);

  /** Cells a set of members resolves to, deduplicated and in document order. */
  const neuronsOf = useCallback(
    (members: Int32Array, kind: SampleKind): NeuronId[] => {
      if (kind === 'neuron') {
        const ids: NeuronId[] = [];
        for (let i = 0; i < members.length; i += 1) {
          const neuron = neurons[members[i]];
          if (neuron !== undefined) ids.push(neuron.id);
        }
        return ids;
      }
      const seen = new Set<string>();
      const ids: NeuronId[] = [];
      for (let i = 0; i < members.length; i += 1) {
        const synapse = synapses[members[i]];
        if (synapse === undefined) continue;
        // Both endpoints: a synapse is not selectable on its own here, and the
        // pair is what "the cells in this bin" means for a wiring attribute.
        if (!seen.has(synapse.source)) {
          seen.add(synapse.source);
          ids.push(synapse.source);
        }
        if (!seen.has(synapse.target)) {
          seen.add(synapse.target);
          ids.push(synapse.target);
        }
      }
      return ids;
    },
    [neurons, synapses],
  );

  const pickBin = useCallback(
    (distribution: Distribution, bin: number) => {
      const size = distribution.counts[bin] ?? 0;
      if (size === 0) return;
      const members = new Int32Array(size);
      const values = new Float32Array(size);
      let kept = 0;
      for (let i = 0; i < distribution.n && kept < size; i += 1) {
        if (binOf(distribution, distribution.values[i]) !== bin) continue;
        members[kept] = distribution.owners[i];
        values[kept] = distribution.values[i];
        kept += 1;
      }
      setFocus({
        key: distribution.key,
        kind: distribution.kind,
        label: `${distribution.label} ${binLabel(distribution, bin)}`,
        unit: distribution.unit,
        precision: distribution.precision,
        lo: binLow(distribution, bin),
        hi: binHigh(distribution, bin),
        members,
        values,
      });
      select(neuronsOf(members, distribution.kind));
    },
    [neuronsOf, select],
  );

  const pickCategory = useCallback(
    (chart: CategoryChart, row: CategoryRow) => {
      if (row.count === 0) return;
      setFocus({
        key: chart.key,
        kind: chart.kind,
        label: `${chart.label} · ${row.label}`,
        unit: '',
        precision: 0,
        lo: Number.NaN,
        hi: Number.NaN,
        members: row.members,
        values: null,
      });
      select(neuronsOf(row.members, chart.kind));
    },
    [neuronsOf, select],
  );

  const clearFocus = useCallback(() => setFocus(null), []);

  if (!open) return null;

  const placement = className ?? 'absolute top-3 bottom-3 right-3 w-[360px]';

  const header = (
    <PanelHeader
      title="Statistics"
      subtitle={
        stats === null
          ? undefined
          : `${stats.scope === 'selection' ? 'Selection' : 'Whole circuit'} · ${grouped(
              stats.cells,
            )} cells · ${grouped(stats.synapses)} synapses`
      }
      icon={<BarChart3 />}
      actions={
        <>
          {stats !== null ? (
            <Tooltip content={`One pass over ${grouped(stats.cells)} cells and ${grouped(stats.synapses)} synapses`}>
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(stats.computeMs, 1)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Re-read the live firing rates and rebuild every distribution">
            <IconButton
              label="Recompute statistics"
              size="sm"
              onClick={() => setSample((value) => value + 1)}
            >
              <RefreshCw />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close statistics panel" size="sm" onClick={onClose}>
              <X />
            </IconButton>
          ) : null}
        </>
      }
    />
  );

  if (stats === null || neurons.length === 0) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        <EmptyState
          icon={<BarChart3 />}
          title="Nothing to measure"
          description="Place neurons and wire them together, then this panel reports the distribution of every attribute across the circuit or across whatever you have selected."
        />
      </Panel>
    );
  }

  const chartProps = {
    log: scale === 'log',
    focus,
    focusRows,
    onPick: pickBin,
    onClearFocus: clearFocus,
    onSelectRow: (row: FocusRow) => select(row.ids),
  };

  const model = stats.models.find((entry) => entry.kind === stats.paramKind) ?? null;
  const varying = stats.params.filter((series) => !series.constant && series.distribution.n > 0);
  const constants = stats.params.filter((series) => series.constant);

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      {header}

      <ScrollArea className="min-h-0 flex-1">
        <PanelSection
          label="Sample"
          flush
          aside={
            <SegmentedControl
              size="sm"
              value={stats.scope}
              onChange={setPreferred}
              options={[
                {
                  value: 'selection' as const,
                  label: 'selection',
                  title: 'Measure only the selected cells',
                  disabled: selection.length === 0,
                },
                { value: 'circuit' as const, label: 'circuit', title: 'Measure the whole circuit' },
              ]}
              aria-label="Statistics scope"
            />
          }
        >
          <p className="nf-numeric text-[10.5px] leading-relaxed text-ink-muted">
            n = <span className="text-ink">{grouped(stats.cells)}</span> cells ·{' '}
            <span className="text-ink">{grouped(stats.synapses)}</span> incident synapses ·
            computed in <span className="text-ink">{fixed(stats.computeMs, 2)} ms</span>
          </p>
          <p className="text-[9.5px] leading-relaxed text-ink-faint">
            {stats.scope === 'selection'
              ? 'Degrees count the whole circuit; synapse statistics cover every connection touching the selection.'
              : selection.length === 0
                ? 'Select cells in the scene to narrow every distribution to them.'
                : 'Measuring the whole circuit while a selection exists.'}
          </p>
          <MarkerLegend />
        </PanelSection>

        <PanelSection
          label="Activity"
          aside={
            stats.rate.n === 0 ? (
              <Badge variant="outline" size="sm">
                no live cells
              </Badge>
            ) : stats.rate.max === 0 ? (
              <Badge variant="warning" size="sm">
                not run
              </Badge>
            ) : null
          }
        >
          <SeriesChart dist={stats.rate} color="var(--color-success)" {...chartProps} />
        </PanelSection>

        <PanelSection
          label="Degree"
          aside={
            <Tooltip content="Count axis for every histogram in this panel. Degree distributions are heavy-tailed; on a linear axis the whole network collapses into the first column.">
              <SegmentedControl
                size="sm"
                value={scale}
                onChange={setScale}
                options={SCALE_OPTIONS}
                aria-label="Count axis scale"
              />
            </Tooltip>
          }
        >
          <SeriesChart dist={stats.inDegree} color="var(--color-accent)" {...chartProps} />
          <SeriesChart
            dist={stats.outDegree}
            color="var(--color-secondary)"
            className="mt-3"
            {...chartProps}
          />
        </PanelSection>

        <PanelSection label="Synapses">
          <SeriesChart dist={stats.weight} color="var(--color-accent)" {...chartProps} />
          <SeriesChart
            dist={stats.delay}
            color="var(--color-secondary)"
            className="mt-3"
            {...chartProps}
          />
        </PanelSection>

        <PanelSection
          label="Membrane parameters"
          aside={
            stats.models.length > 1 && stats.paramKind !== null ? (
              <Select
                size="sm"
                value={stats.paramKind}
                onValueChange={(value) => setRequestedModel(value as NeuronModelKind)}
                aria-label="Membrane model"
                contentClassName="min-w-[220px]"
              >
                {stats.models.map((entry) => (
                  <SelectItem
                    key={entry.kind}
                    value={entry.kind}
                    description={`${grouped(entry.cells)} cells`}
                  >
                    {entry.label}
                  </SelectItem>
                ))}
              </Select>
            ) : model !== null ? (
              <span className="nf-numeric text-[9.5px] text-ink-faint">
                {grouped(model.cells)} cells
              </span>
            ) : null
          }
        >
          {model === null ? (
            <p className="text-[10.5px] text-ink-faint">No cells in this sample.</p>
          ) : (
            <>
              <p className="text-[9.5px] leading-relaxed text-ink-faint">
                {model.label} — {grouped(model.cells)} of {grouped(stats.cells)} cells
                {stats.models.length > 1
                  ? `, ${grouped(stats.models.length - 1)} other model${
                      stats.models.length > 2 ? 's' : ''
                    } present`
                  : ''}
                .
              </p>
              {varying.length === 0 ? (
                <p className="text-[10.5px] text-ink-faint">
                  Every parameter is identical across these cells.
                </p>
              ) : (
                varying.map((series, index) => (
                  <SeriesChart
                    key={series.distribution.key}
                    dist={series.distribution}
                    color="var(--color-warning)"
                    className={index === 0 ? undefined : 'mt-3'}
                    {...chartProps}
                  />
                ))
              )}
              {constants.length > 0 ? (
                <div className="mt-2 flex flex-col gap-0.5 border-t border-hairline pt-2">
                  <span className="text-[9px] font-medium uppercase tracking-[0.07em] text-ink-faint">
                    Uniform across the sample
                  </span>
                  {constants.map((series) => (
                    <Tooltip key={series.distribution.key} content={series.field.hint} side="top">
                      <div
                        role="group"
                        tabIndex={0}
                        className="flex items-baseline gap-2 rounded-sm text-[10.5px] focus-visible:outline-1"
                      >
                        <span className="truncate text-ink-muted">{series.field.label}</span>
                        <span className="nf-numeric ml-auto shrink-0 text-ink">
                          {withUnit(
                            series.distribution.min,
                            series.field.unit,
                            series.field.precision,
                          )}
                        </span>
                      </div>
                    </Tooltip>
                  ))}
                </div>
              ) : null}
            </>
          )}
        </PanelSection>

        {stats.categories.map((chart) => (
          <PanelSection
            key={chart.key}
            label={chart.label}
            aside={
              <span className="nf-numeric text-[9.5px] text-ink-faint">
                {grouped(chart.total)} {chart.kind === 'neuron' ? 'cells' : 'synapses'}
              </span>
            }
          >
            <CategoryBars
              chart={chart}
              focus={focus}
              focusRows={focusRows}
              onPick={pickCategory}
              onClearFocus={clearFocus}
              onSelectRow={(row) => select(row.ids)}
            />
          </PanelSection>
        ))}
      </ScrollArea>
    </Panel>
  );
}

/* ------------------------------------------------------------------- pieces -- */

/** Key to the marker lines drawn over every histogram. */
function MarkerLegend() {
  return (
    <div className="flex items-center gap-3 text-[9px] text-ink-faint">
      <span className="flex items-center gap-1">
        <svg width={12} height={8} aria-hidden className="shrink-0">
          <line x1={6} y1={0} x2={6} y2={8} stroke="var(--color-ink)" strokeWidth={1.25} />
        </svg>
        median
      </span>
      <span className="flex items-center gap-1">
        <svg width={12} height={8} aria-hidden className="shrink-0">
          <line
            x1={6}
            y1={0}
            x2={6}
            y2={8}
            stroke="var(--color-warning)"
            strokeWidth={1.25}
            strokeDasharray="2 2"
          />
        </svg>
        mean
      </span>
      <span className="flex items-center gap-1">
        <svg width={12} height={8} aria-hidden className="shrink-0">
          <line x1={2} y1={6} x2={2} y2={8} stroke="var(--color-ink-faint)" strokeWidth={1.25} />
          <line x1={10} y1={6} x2={10} y2={8} stroke="var(--color-ink-faint)" strokeWidth={1.25} />
        </svg>
        5th / 95th
      </span>
    </div>
  );
}

interface SeriesChartProps {
  dist: Distribution;
  color: string;
  log: boolean;
  focus: Focus | null;
  focusRows: readonly FocusRow[];
  onPick: (dist: Distribution, bin: number) => void;
  onClearFocus: () => void;
  onSelectRow: (row: FocusRow) => void;
  className?: string;
}

/**
 * One histogram: the plot, its axes, the five-number readout, and the list of
 * whatever is inside the bin the user last reached into.
 */
function SeriesChart({
  dist,
  color,
  log,
  focus,
  focusRows,
  onPick,
  onClearFocus,
  onSelectRow,
  className,
}: SeriesChartProps) {
  const [cursor, setCursor] = useState(-1);
  const focused = focus !== null && focus.key === dist.key;

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<SVGSVGElement>) => {
      if (dist.binCount === 0) return;
      const current = cursor < 0 ? 0 : cursor;
      switch (event.key) {
        case 'ArrowRight':
          event.preventDefault();
          setCursor(Math.min(dist.binCount - 1, current + 1));
          return;
        case 'ArrowLeft':
          event.preventDefault();
          setCursor(Math.max(0, current - 1));
          return;
        case 'Home':
          event.preventDefault();
          setCursor(0);
          return;
        case 'End':
          event.preventDefault();
          setCursor(dist.binCount - 1);
          return;
        case 'Enter':
        case ' ':
          event.preventDefault();
          if (cursor >= 0) onPick(dist, cursor);
          return;
        default:
      }
    },
    [cursor, dist, onPick],
  );

  if (dist.n === 0) {
    return (
      <div className={className}>
        <div className="flex items-baseline justify-between">
          <span className="text-[10.5px] text-ink">{dist.label}</span>
          <span className="nf-numeric text-[9.5px] text-ink-faint">n=0</span>
        </div>
        <p className="mt-1 text-[9.5px] text-ink-faint">No samples in this scope.</p>
      </div>
    );
  }

  const slot = CHART_W / dist.binCount;
  const gap = slot > 5 ? 2 : slot > 2.5 ? 1 : 0;
  const barWidth = Math.max(0.75, slot - gap);
  const logPeak = Math.log1p(dist.peak);
  /**
   * Width of the value axis. For integer data the last bin runs to the end of a
   * whole column, which is wider than `max - min`; markers have to be placed on
   * the same axis the columns are drawn on or the median lands beside its bar.
   */
  const axisSpan = dist.binCount * dist.binWidth;
  const centred = dist.integral ? 0.5 * Math.min(1, dist.binWidth) : 0;

  const markerX = (value: number): number => {
    if (axisSpan <= 0) return CHART_W / 2;
    const t = (value - dist.min + centred) / axisSpan;
    return Math.min(CHART_W, Math.max(0, t)) * CHART_W === 0 ? 0 : Math.min(1, Math.max(0, t)) * CHART_W;
  };

  const heightOf = (count: number): number => {
    if (count <= 0 || dist.peak <= 0) return 0;
    const norm = log ? Math.log1p(count) / (logPeak || 1) : count / dist.peak;
    return Math.max(1.25, norm * PLOT_H);
  };

  const countNoun = dist.kind === 'neuron' ? 'cells' : 'synapses';
  const unitSuffix = dist.unit === '' ? '' : ` ${dist.unit}`;

  return (
    <div className={className}>
      <div className="flex items-baseline justify-between gap-2">
        <Tooltip content={dist.hint} side="top">
          <span
            role="group"
            tabIndex={0}
            className="truncate rounded-sm text-[10.5px] text-ink focus-visible:outline-1"
          >
            {dist.label}
          </span>
        </Tooltip>
        <span className="nf-numeric shrink-0 text-[9.5px] text-ink-faint">
          peak {grouped(dist.peak)} {countNoun}
        </span>
      </div>

      {/* preserveAspectRatio="none" keeps the vertical scale at 1:1 while the
          columns stretch to the panel width, which is what a histogram wants.
          Every label therefore lives in the HTML around the plot, not in it. */}
      <svg
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        width="100%"
        height={CHART_H}
        preserveAspectRatio="none"
        role="img"
        tabIndex={0}
        aria-label={`${dist.label} distribution over ${grouped(dist.n)} ${countNoun}, ${withUnit(
          dist.min,
          dist.unit,
          dist.precision,
        )} to ${withUnit(dist.max, dist.unit, dist.precision)}, median ${withUnit(
          dist.median,
          dist.unit,
          dist.precision,
        )}. Arrow keys move between bins, Enter selects the ${countNoun} in one.`}
        onKeyDown={handleKeyDown}
        className="mt-1 block rounded-sm focus-visible:outline-1"
      >
        <rect
          x={0}
          y={BASELINE}
          width={CHART_W}
          height={0.75}
          fill="var(--color-ink-faint)"
          opacity={0.45}
        />

        {Array.from({ length: dist.binCount }, (_, bin) => {
          const count = dist.counts[bin];
          const x = bin * slot;
          const low = binLow(dist, bin);
          const high = binHigh(dist, bin);
          const inFocus =
            focused && Number.isFinite(focus.lo) && low >= focus.lo - 1e-9 && high <= focus.hi + 1e-9;
          const height = heightOf(count);
          return (
            <g key={bin}>
              {/* A full-height hit target: the tail columns that matter most are
                  one pixel tall, and a one-pixel click target is not a control. */}
              <rect
                x={x}
                y={0}
                width={Math.max(slot, 1)}
                height={BASELINE}
                fill={cursor === bin ? 'var(--color-ink)' : 'transparent'}
                opacity={cursor === bin ? 0.06 : 0}
                className={count > 0 ? 'cursor-pointer' : undefined}
                onClick={() => {
                  if (count === 0) return;
                  setCursor(bin);
                  onPick(dist, bin);
                }}
              >
                <title>
                  {binLabel(dist, bin)}: {grouped(count)} {countNoun} ({percent(count / dist.n)})
                </title>
              </rect>
              {count > 0 ? (
                <rect
                  x={x + gap / 2}
                  y={BASELINE - height}
                  width={barWidth}
                  height={height}
                  fill={inFocus ? 'var(--color-ink)' : color}
                  opacity={inFocus ? 1 : 0.85}
                  pointerEvents="none"
                />
              ) : null}
            </g>
          );
        })}

        <line
          x1={markerX(dist.p05)}
          y1={BASELINE - 5}
          x2={markerX(dist.p05)}
          y2={BASELINE}
          stroke="var(--color-ink-faint)"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
          pointerEvents="none"
        />
        <line
          x1={markerX(dist.p95)}
          y1={BASELINE - 5}
          x2={markerX(dist.p95)}
          y2={BASELINE}
          stroke="var(--color-ink-faint)"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
          pointerEvents="none"
        />
        <line
          x1={markerX(dist.mean)}
          y1={PLOT_TOP - 2}
          x2={markerX(dist.mean)}
          y2={BASELINE}
          stroke="var(--color-warning)"
          strokeWidth={1}
          strokeDasharray="2 2"
          vectorEffect="non-scaling-stroke"
          pointerEvents="none"
        />
        <line
          x1={markerX(dist.median)}
          y1={PLOT_TOP - 2}
          x2={markerX(dist.median)}
          y2={BASELINE}
          stroke="var(--color-ink)"
          strokeWidth={1}
          opacity={0.7}
          vectorEffect="non-scaling-stroke"
          pointerEvents="none"
        />
      </svg>

      <div className="mt-0.5 flex items-baseline justify-between gap-2 text-[9px] text-ink-faint">
        <span className="nf-numeric shrink-0">{fixed(dist.min, dist.precision)}</span>
        <span className="truncate">
          {dist.axis} → {log ? `log ${countNoun}` : countNoun}
        </span>
        <span className="nf-numeric shrink-0">{fixed(dist.max, dist.precision)}</span>
      </div>

      <div className="nf-numeric mt-1 flex flex-wrap gap-x-2 gap-y-0.5 text-[9.5px] text-ink-faint">
        <span>
          n <span className="text-ink-muted">{grouped(dist.n)}</span>
        </span>
        <span>
          mean <span className="text-ink-muted">{fixed(dist.mean, dist.precision)}</span>
        </span>
        <span>
          med <span className="text-ink-muted">{fixed(dist.median, dist.precision)}</span>
        </span>
        <span>
          sd <span className="text-ink-muted">{fixed(dist.sd, dist.precision)}</span>
        </span>
        <span>
          range{' '}
          <span className="text-ink-muted">
            {fixed(dist.min, dist.precision)}–{fixed(dist.max, dist.precision)}
            {unitSuffix}
          </span>
        </span>
      </div>

      {focused ? (
        <FocusList
          focus={focus}
          rows={focusRows}
          onClear={onClearFocus}
          onSelectRow={onSelectRow}
        />
      ) : null}
    </div>
  );
}

interface FocusListProps {
  focus: Focus;
  rows: readonly FocusRow[];
  onClear: () => void;
  onSelectRow: (row: FocusRow) => void;
}

/**
 * What is actually inside the opened bin.
 *
 * Every swatch is `identityColorHex` of the cell's morphology seed, which is the
 * same hue the renderer draws it in — so a row here and a glyph in the scene are
 * recognisably the same cell.
 */
function FocusList({ focus, rows, onClear, onSelectRow }: FocusListProps) {
  const hidden = focus.members.length - rows.length;
  return (
    <div className="mt-1.5 rounded-control border border-hairline bg-black/20 px-1.5 py-1">
      <div className="flex items-center gap-1.5">
        <span className="truncate text-[9.5px] text-ink-muted">{focus.label}</span>
        <span className="nf-numeric ml-auto shrink-0 text-[9.5px] text-ink-faint">
          {grouped(focus.members.length)} {focus.kind === 'neuron' ? 'cells' : 'synapses'}
        </span>
        <IconButton label="Close bin contents" size="sm" onClick={onClear}>
          <X />
        </IconButton>
      </div>
      <ul className="mt-0.5 flex flex-col">
        {rows.map((row) => (
          <li key={row.key}>
            <button
              type="button"
              onClick={() => onSelectRow(row)}
              className={cn(
                'flex w-full items-center gap-1.5 rounded-control px-1 py-0.5 text-left text-[10px]',
                'transition-colors hover:bg-panel-raised focus-visible:bg-panel-raised',
              )}
            >
              <span
                aria-hidden
                className="size-2 shrink-0 rounded-[2px] ring-1 ring-white/15"
                style={{ backgroundColor: row.color }}
              />
              {row.toColor !== null ? (
                <>
                  <span aria-hidden className="shrink-0 text-ink-faint">
                    →
                  </span>
                  <span
                    aria-hidden
                    className="size-2 shrink-0 rounded-[2px] ring-1 ring-white/15"
                    style={{ backgroundColor: row.toColor }}
                  />
                </>
              ) : null}
              <span className="truncate text-ink">{row.label}</span>
              {row.detail !== null ? (
                <span className="truncate text-ink-faint">{row.detail}</span>
              ) : null}
              {row.value !== null ? (
                <span className="nf-numeric ml-auto shrink-0 text-ink-muted">
                  {withUnit(row.value, focus.unit, focus.precision)}
                </span>
              ) : null}
            </button>
          </li>
        ))}
      </ul>
      {hidden > 0 ? (
        <p className="px-1 pt-0.5 text-[9px] text-ink-faint">
          {grouped(hidden)} more — all {grouped(focus.members.length)} are selected in the scene.
        </p>
      ) : null}
    </div>
  );
}

interface CategoryBarsProps {
  chart: CategoryChart;
  focus: Focus | null;
  focusRows: readonly FocusRow[];
  onPick: (chart: CategoryChart, row: CategoryRow) => void;
  onClearFocus: () => void;
  onSelectRow: (row: FocusRow) => void;
}

/** Horizontal bars with counts and shares; each row selects its members. */
function CategoryBars({
  chart,
  focus,
  focusRows,
  onPick,
  onClearFocus,
  onSelectRow,
}: CategoryBarsProps) {
  const focused = focus !== null && focus.key === chart.key;
  const noun = chart.kind === 'neuron' ? 'cells' : 'synapses';

  if (chart.rows.length === 0) {
    return <p className="text-[10.5px] text-ink-faint">Nothing to break down.</p>;
  }

  const largest = chart.rows.reduce((max, row) => Math.max(max, row.count), 0);

  return (
    <>
      <ul className="flex flex-col">
        {chart.rows.map((row) => {
          const active = focused && focus.label.endsWith(`· ${row.label}`);
          return (
            <li key={row.key}>
              <button
                type="button"
                onClick={() => onPick(chart, row)}
                aria-pressed={active}
                className={cn(
                  'flex w-full flex-col gap-0.5 rounded-control px-1 py-1 text-left',
                  'transition-colors hover:bg-panel-raised focus-visible:bg-panel-raised',
                  active && 'bg-white/[0.07]',
                )}
              >
                <span className="flex items-baseline gap-1.5 text-[10.5px]">
                  <span
                    aria-hidden
                    className="size-2 shrink-0 translate-y-px rounded-[2px]"
                    style={{ backgroundColor: row.color }}
                  />
                  <span className="truncate text-ink">{row.label}</span>
                  <span className="nf-numeric ml-auto shrink-0 text-ink-faint">
                    {grouped(row.count)}
                  </span>
                  <span className="nf-numeric w-[46px] shrink-0 text-right text-ink">
                    {percent(row.share)}
                  </span>
                </span>
                {/* The track is scaled to the largest row rather than to the
                    total, so a breakdown with one dominant bucket still shows
                    the shape of its tail. */}
                <span className="block h-[3px] w-full overflow-hidden rounded-full bg-white/[0.05]">
                  <span
                    className="block h-full rounded-full"
                    style={{
                      width: `${largest > 0 ? (row.count / largest) * 100 : 0}%`,
                      backgroundColor: row.color,
                    }}
                  />
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <p className="nf-numeric px-1 text-[9px] text-ink-faint">
        {grouped(chart.total)} {noun} · {grouped(chart.rows.length)} categor
        {chart.rows.length === 1 ? 'y' : 'ies'}
      </p>
      {focused ? (
        <FocusList
          focus={focus}
          rows={focusRows}
          onClear={onClearFocus}
          onSelectRow={onSelectRow}
        />
      ) : null}
    </>
  );
}
