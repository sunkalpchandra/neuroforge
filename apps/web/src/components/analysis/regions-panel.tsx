'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Boxes, Info, RefreshCw, TriangleAlert, X } from 'lucide-react';
import {
  Badge,
  Button,
  EmptyState,
  IconButton,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  SegmentedControl,
  Separator,
  Slider,
  Spinner,
  Switch,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import type { SegmentedOption } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import type { Circuit, NeuronId } from '@neuroforge/shared';

import { getEngine } from '@/lib/runtime';
import { compact, fixed, grouped } from '@/lib/format';
import { graphSignature } from '@/lib/graph-metrics';
import {
  DEFAULT_LOCALITY_FLOOR,
  MAX_GRID_DIVISIONS,
  MIN_GRID_DIVISIONS,
  gridRegions,
  localityExtremes,
  meanRateByRegion,
  populationRegions,
  projectionMembers,
  regionMatrix,
  regionMembers,
  regionStats,
  regionSwatches,
} from '@/lib/regions';
import type { RegionKind, RegionMatrix, RegionSet, RegionStats } from '@/lib/regions';

/* -------------------------------------------------------------- constants -- */

/** Safety net for engine reloads that did not come through this component's deps. */
const SIGNATURE_POLL_MS = 500;

/** Firing rates move every step; the partition underneath them does not. */
const RATE_POLL_MS = 200;

/**
 * Cost above which a document edit that left the neuron and synapse counts
 * alone stops triggering an automatic re-cut.
 *
 * Adding or deleting cells is a discrete act and always worth the pass.
 * Dragging a weight slider is not: it republishes the document continuously,
 * and a pass longer than a frame would stall the pointer twice a second. Past
 * this budget the panel marks itself stale and waits to be asked. Changing the
 * partition always rebuilds, because that changes what a region *is*.
 */
const AUTO_BUILD_BUDGET_MS = 16;

const DEFAULT_DIVISIONS = 3;

/**
 * Rows the list renders before it stops. An 8³ lattice can leave 512 occupied
 * cells, and a list that long is a scroll gesture rather than a readout; the
 * chosen ordering decides which end of it survives the cut.
 */
const MAX_LIST_ROWS = 120;

/** The near-black of the viewport, which an unconnected cell sits on. */
const FIELD_CSS = '#05070a';
const SURFACE_CSS = '#0a0d11';
const GRID_CSS = 'rgba(255,255,255,0.05)';
const DIAGONAL_CSS = 'rgba(255,255,255,0.15)';
const HIGHLIGHT_CSS = 'rgba(255,255,255,0.07)';
const CURSOR_CSS = '#4fd1ff';
/** Drawn where a region has no member colour to sample, which only a race produces. */
const UNKNOWN_CSS = '#7c8189';

const PAD = 6;
/** Thickness of the identity-colour strip along each axis, in CSS px. */
const TICK = 5;
const TICK_GAP = 3;
/** Below this cell size the grid lines cost more legibility than they add. */
const MIN_GRID_CELL = 7;
/** A strip needs this many pixels per segment before the segments mean anything. */
const MIN_STRIP_SEGMENT = 2.5;

/* ------------------------------------------------------------------ ramp -- */

type Rgb = readonly [number, number, number];

/**
 * Magma — the same sequential ramp the connectivity matrix paints with. The two
 * views are read side by side and a second ramp would imply a second quantity.
 * It is monotonic in lightness from black, so rank order in the data survives as
 * rank order in perceived brightness, and it starts at the field colour, so an
 * unconnected pair and a zero-weight one agree rather than fight.
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

const RAMP_STEPS = 256;

/**
 * Floor applied to every realised projection's position on the ramp, so the
 * thinnest connection in the network is still separable from an absent one.
 * Nothing below it is reachable, which is why the legend gradient is floored too.
 */
const CELL_FLOOR = 0.09;

function bakeRamp(ramp: readonly Rgb[]): readonly string[] {
  const out = new Array<string>(RAMP_STEPS);
  for (let i = 0; i < RAMP_STEPS; i += 1) {
    const x = (i / (RAMP_STEPS - 1)) * (ramp.length - 1);
    const at = Math.min(ramp.length - 2, Math.floor(x));
    const f = x - at;
    const a = ramp[at];
    const b = ramp[at + 1];
    const r = Math.round(a[0] + (b[0] - a[0]) * f);
    const g = Math.round(a[1] + (b[1] - a[1]) * f);
    const bl = Math.round(a[2] + (b[2] - a[2]) * f);
    out[i] = `rgb(${r},${g},${bl})`;
  }
  return out;
}

const RAMP_CSS = bakeRamp(SEQUENTIAL);

function rampIndex(t: number): number {
  const clamped = t <= 0 ? 0 : t >= 1 ? 1 : t;
  return Math.round((CELL_FLOOR + (1 - CELL_FLOOR) * clamped) * (RAMP_STEPS - 1));
}

const RAMP_GRADIENT = `linear-gradient(90deg, ${Array.from({ length: 11 }, (_, i) => {
  const t = i / 10;
  return `${RAMP_CSS[rampIndex(t)]} ${(t * 100).toFixed(0)}%`;
}).join(',')})`;

/* ------------------------------------------------------------------ types -- */

type Metric = 'synapses' | 'weight';
type Ordering = 'partition' | 'size' | 'locality';

const KIND_OPTIONS: readonly SegmentedOption<RegionKind>[] = [
  {
    value: 'grid',
    label: 'grid',
    title: 'Cut the network’s bounding box into an N × N × N lattice and treat each occupied cell as a region',
  },
  {
    value: 'population',
    label: 'population',
    title: 'Treat each population as a region — the grouping the circuit was actually wired in',
  },
];

const METRIC_OPTIONS: readonly SegmentedOption<Metric>[] = [
  { value: 'synapses', label: 'syn', title: 'Number of synapses in each region pair' },
  { value: 'weight', label: 'Σw', title: 'Summed peak conductance of each region pair, in nS' },
];

const ORDER_OPTIONS: readonly SegmentedOption<Ordering>[] = [
  { value: 'partition', label: '#', title: 'Partition order: lattice order, or population order' },
  { value: 'size', label: 'n', title: 'Largest regions first' },
  { value: 'locality', label: 'loc', title: 'Most locally-connected regions first' },
];

interface RegionView {
  regions: RegionSet;
  matrix: RegionMatrix;
  stats: RegionStats;
  swatches: readonly (readonly string[])[];
  buildMs: number;
}

interface Cell {
  row: number;
  col: number;
}

interface Field {
  values: ArrayLike<number>;
  max: number;
  /** Smallest non-zero cell, which is where a log scale has to start. */
  minPositive: number;
  unit: string;
}

interface Geometry {
  origin: number;
  cell: number;
  plot: number;
}

const EMPTY_RATES = new Float32Array(0);
const EMPTY_ORDER: readonly number[] = [];

export interface RegionsPanelProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Overrides the default placement. */
  className?: string;
}

/* ------------------------------------------------------------------ panel -- */

/**
 * Connectivity within and between regions.
 *
 * The regions are derived rather than atlased — see `lib/regions.ts` — and the
 * panel says so on its face rather than letting a lattice cell read as a
 * neuropil. Everything is measured off the live simulation buffers, and the
 * cross-tabulation is re-cut only when the circuit or the partition changes,
 * never per frame.
 */
export function RegionsPanel({ open = true, onClose, className }: RegionsPanelProps) {
  const circuit = useEditor((s) => s.circuit);
  const select = useEditor((s) => s.select);

  const [kind, setKind] = useState<RegionKind>('grid');
  const [divisions, setDivisions] = useState(DEFAULT_DIVISIONS);
  const [metric, setMetric] = useState<Metric>('synapses');
  const [log, setLog] = useState(true);
  const [order, setOrder] = useState<Ordering>('partition');

  const [view, setView] = useState<RegionView | null>(null);
  const [pending, setPending] = useState(false);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<Cell | null>(null);
  const [surface, setSurface] = useState({ width: 0, height: 0, dpr: 1 });
  // Set when the browser refuses a 2D context, so the panel explains itself
  // rather than sitting there as an unexplained black square.
  const [unavailable, setUnavailable] = useState(false);

  /**
   * The measured element is held as state rather than in a ref because it does
   * not exist yet on the commit that first runs the sizing effect below.
   *
   * Until the first cut lands this component renders the loading panel, which
   * has no canvas host. A ref object would therefore still be null when that
   * effect ran, and because `open` — its only other input — never changes
   * afterwards, the effect would never run again: the ResizeObserver would never
   * attach, the surface would stay at zero and the matrix would never paint. A
   * callback ref re-runs the effect on the commit that attaches the node.
   */
  const [host, setHost] = useState<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const busyRef = useRef(false);
  const frameRef = useRef(0);
  const signatureRef = useRef('');
  /** Cost of the last completed pass; null until one has run. */
  const costRef = useRef<number | null>(null);
  const hasDataRef = useRef(false);
  const partitionRef = useRef<{ kind: RegionKind; divisions: number } | null>(null);
  const documentRef = useRef<Pick<Circuit, 'neurons' | 'synapses' | 'populations'> | null>(null);
  const wantedRef = useRef({ kind, divisions });

  const build = useCallback(() => {
    // The latch below is only held between scheduling the frame and running it,
    // so a request arriving in that window is absorbed by the pass that has not
    // run yet rather than dropped — which is what lets the divisions slider be
    // dragged without losing the value it settles on.
    wantedRef.current = { kind, divisions };
    if (busyRef.current) return;
    busyRef.current = true;
    setPending(true);
    // Deferring by a frame does two things: it lets the busy state paint before
    // the blocking pass, and it puts the pass after every effect in this commit
    // — including the ancestor one that reloads the engine, which runs after
    // this component's effects and would otherwise leave us cutting up last
    // edit's buffers.
    frameRef.current = requestAnimationFrame(() => {
      try {
        const buffers = getEngine().buffers;
        const wanted = wantedRef.current;
        // Stamped before the pass, so a cut that throws cannot leave the poll
        // below retrying the same failing partition twice a second.
        signatureRef.current = graphSignature(buffers);
        const started = performance.now();
        const regions =
          wanted.kind === 'grid'
            ? gridRegions(buffers, wanted.divisions)
            : populationRegions(useEditor.getState().circuit, buffers);
        const matrix = regionMatrix(regions, buffers);
        const stats = regionStats(regions, buffers);
        const swatches = regionSwatches(regions, buffers);
        const next: RegionView = {
          regions,
          matrix,
          stats,
          swatches,
          buildMs: performance.now() - started,
        };
        costRef.current = next.buildMs;
        hasDataRef.current = true;
        setView(next);
        setStale(false);
        setError(null);
      } catch (cause) {
        // The busy latch is what stops the poll from stampeding, so it has to be
        // released on the failing path too; without this one throw would leave
        // the panel spinning on a stale partition forever.
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setPending(false);
        busyRef.current = false;
      }
    });
  }, [kind, divisions]);

  useEffect(() => {
    if (!open) return;

    const previous = partitionRef.current;
    const partitionChanged =
      previous === null || previous.kind !== kind || previous.divisions !== divisions;
    partitionRef.current = { kind, divisions };

    // The editor drafts copy-on-write, so any edit to a cell, a synapse or a
    // population replaces one of these arrays; nothing else can change what the
    // partition would say. Comparing them keeps reopening the panel, or a
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

    if (partitionChanged || !hasDataRef.current) {
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
  }, [open, build, kind, divisions, circuit.neurons, circuit.synapses, circuit.populations]);

  // A cursor is only meaningful against the matrix it was measured on.
  useEffect(() => {
    setHover(null);
  }, [view]);

  const rates = useRegionRates(open ? view : null);

  const size = view === null ? 0 : view.regions.regions.length;

  const field = useMemo<Field | null>(() => {
    if (view === null) return null;
    const matrix = view.matrix;
    const values = metric === 'synapses' ? matrix.count : matrix.weight;
    const max = metric === 'synapses' ? matrix.maxCount : matrix.maxWeight;
    let minPositive = Infinity;
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      if (value > 0 && value < minPositive) minPositive = value;
    }
    return { values, max, minPositive, unit: metric === 'synapses' ? 'syn' : 'nS' };
  }, [view, metric]);

  const extremes = useMemo(
    () => (view === null ? null : localityExtremes(view.stats)),
    [view],
  );

  const ranked = useMemo(() => {
    if (view === null || size === 0) return EMPTY_ORDER;
    const stats = view.stats;
    const indices = Array.from({ length: size }, (_, i) => i);
    if (order === 'size') {
      indices.sort((a, b) => stats.cells[b] - stats.cells[a] || a - b);
    } else if (order === 'locality') {
      // A region nothing connects to has no locality at all; it sinks rather
      // than sorting as if it were perfectly isolated or perfectly local.
      const rank = (i: number): number =>
        Number.isFinite(stats.locality[i]) ? stats.locality[i] : -1;
      indices.sort((a, b) => rank(b) - rank(a) || a - b);
    }
    return indices;
  }, [view, size, order]);

  const geometry = useMemo(
    () => layoutMatrix(Math.min(surface.width, surface.height), size),
    [surface.width, surface.height, size],
  );

  /* --------------------------------------------------------------- surface -- */

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

  /* ----------------------------------------------------------------- paint -- */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!open || canvas === null || view === null || field === null) return;
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

    paintMatrix(ctx, surface.width, surface.height, surface.dpr, {
      size,
      field,
      geometry,
      log,
      swatches: view.swatches,
    });
  }, [open, view, field, geometry, log, size, surface]);

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
    paintCursor(ctx, surface.width, surface.height, surface.dpr, geometry, size, hover);
  }, [open, geometry, size, hover, surface]);

  /* --------------------------------------------------------------- pointer -- */

  const selectRegion = useCallback(
    (index: number) => {
      if (view === null) return;
      selectSlots(regionMembers(view.regions, index), select);
    },
    [view, select],
  );

  const selectProjection = useCallback(
    (cell: Cell) => {
      if (view === null) return;
      const slots = projectionMembers(view.regions, getEngine().buffers, cell.row, cell.col);
      selectSlots(slots, select);
    },
    [view, select],
  );

  const trackPointer = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      const hit = hitTest(geometry, size, event.clientX - rect.left, event.clientY - rect.top);
      setHover((current) => {
        if (current === null && hit === null) return current;
        if (current !== null && hit !== null && current.row === hit.row && current.col === hit.col) {
          return current;
        }
        return hit;
      });
    },
    [geometry, size],
  );

  const clickCell = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const hit = hitTest(geometry, size, event.clientX - rect.left, event.clientY - rect.top);
      if (hit !== null) selectProjection(hit);
    },
    [geometry, size, selectProjection],
  );

  const moveCursor = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (size === 0) return;
      const current = hover ?? { row: 0, col: 0 };
      let next: Cell;
      switch (event.key) {
        case 'ArrowUp':
          next = { row: Math.max(0, current.row - 1), col: current.col };
          break;
        case 'ArrowDown':
          next = { row: Math.min(size - 1, current.row + 1), col: current.col };
          break;
        case 'ArrowLeft':
          next = { row: current.row, col: Math.max(0, current.col - 1) };
          break;
        case 'ArrowRight':
          next = { row: current.row, col: Math.min(size - 1, current.col + 1) };
          break;
        case 'Enter':
        case ' ':
          event.preventDefault();
          if (hover !== null) selectProjection(hover);
          return;
        default:
          return;
      }
      event.preventDefault();
      setHover(next);
    },
    [size, hover, selectProjection],
  );

  if (!open) return null;

  /* ------------------------------------------------------------------ view -- */

  const placement = className ?? 'absolute top-3 bottom-3 left-3 w-[340px]';

  const header = (
    <PanelHeader
      title="Regions"
      subtitle={
        view === null
          ? undefined
          : `${grouped(size)} regions · ${grouped(view.regions.assigned)} cells`
      }
      icon={<Boxes />}
      actions={
        <>
          {error !== null ? (
            <Tooltip content={`The last pass failed: ${error}`}>
              <Badge variant="danger" size="sm" tabIndex={0}>
                failed
              </Badge>
            </Tooltip>
          ) : stale ? (
            <Tooltip content="The circuit changed after this pass, and re-cutting it costs more than a frame. Re-cut to bring the regions back in step.">
              <Badge variant="warning" size="sm" tabIndex={0}>
                stale
              </Badge>
            </Tooltip>
          ) : view !== null ? (
            <Tooltip content="Time taken by the last pass over the live buffers">
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(view.buildMs, 1)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Re-cut from the running network. A spatial partition is cut from the positions as they stand, so a layout still relaxing needs a fresh cut.">
            <IconButton label="Re-cut regions" size="sm" onClick={build} disabled={pending}>
              <RefreshCw className={pending ? 'animate-spin' : undefined} />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close regions panel" size="sm" onClick={onClose}>
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
          divide by
        </span>
        <SegmentedControl
          size="sm"
          className="ml-auto"
          value={kind}
          onChange={setKind}
          options={KIND_OPTIONS}
          aria-label="Region partitioning"
        />
      </div>
      {kind === 'grid' ? (
        <div className="flex items-center gap-2">
          <span className="w-[52px] shrink-0 text-[9.5px] tracking-[0.08em] text-ink-faint uppercase">
            lattice
          </span>
          <Slider
            className="min-w-0 flex-1"
            value={divisions}
            onChange={setDivisions}
            min={MIN_GRID_DIVISIONS}
            max={MAX_GRID_DIVISIONS}
            step={1}
            showValue
            formatValue={(value) => `${value}³`}
            aria-label="Lattice divisions per axis"
          />
        </div>
      ) : null}
    </div>
  );

  // The honest caption. It is not a tooltip and not a one-time hint: every
  // number below it is a spatial or organisational statistic wearing the
  // vocabulary of anatomy, and a reader arriving at a screenshot of this panel
  // has to be able to see that from the panel itself.
  const provenance = (
    <div className="flex shrink-0 items-start gap-1.5 border-b border-hairline px-3 py-1.5">
      <Info aria-hidden className="mt-[1px] size-3 shrink-0 text-ink-faint" />
      <p className="text-[9.5px] leading-snug text-ink-faint">
        <span className="text-ink-muted">Derived, not anatomical.</span>{' '}
        {kind === 'grid'
          ? 'This platform has no neuropil atlas, so a region here is one box of a lattice cut from the network’s own bounding box — a volume of space, not a named brain area.'
          : 'This platform has no neuropil atlas, so a region here is a population the circuit was built from — the group it was wired as, not a named brain area.'}
      </p>
    </div>
  );

  if (view === null || field === null || extremes === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        {controls}
        {provenance}
        {error === null ? (
          <EmptyState
            icon={<Spinner size={18} />}
            title="Cutting the network into regions"
            description="Assigning every cell, then cross-tabulating each connection by the regions at its two ends."
          />
        ) : (
          <EmptyState
            icon={<TriangleAlert className="text-danger" />}
            title="Could not build the regions"
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

  if (view.regions.count === 0) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        {controls}
        {provenance}
        <EmptyState
          icon={<Boxes />}
          title="No cells to divide"
          description="Place neurons and wire them together, then this panel reports what each part of the network keeps to itself and what it sends elsewhere."
        />
      </Panel>
    );
  }

  const { regions, matrix, stats, swatches } = view;
  const recurrentShare = matrix.synapses > 0 ? matrix.recurrent / matrix.synapses : Number.NaN;

  const notice =
    size === 0
      ? 'Every cell has an unplaceable position, so there is nothing to partition.'
      : kind === 'population' && circuit.populations.length === 0
        ? 'This circuit has no populations, so every cell falls in one bucket. Switch to the spatial lattice to divide it up.'
        : size < 2
          ? 'A single region has nothing to project to. Raise the lattice, or build the circuit from populations.'
          : unavailable
            ? 'This browser would not give the matrix a 2D canvas, so it cannot be drawn.'
            : null;

  const hovered =
    hover !== null && hover.row < size && hover.col < size
      ? {
          cell: hover,
          source: regions.regions[hover.row],
          target: regions.regions[hover.col],
          at: hover.row * size + hover.col,
        }
      : null;

  const rangeLabel =
    field.max <= 0
      ? 'nothing connected'
      : log && Number.isFinite(field.minPositive)
        ? `${formatValue(field.minPositive, field.unit)} → ${formatValue(field.max, field.unit)}`
        : `0 → ${formatValue(field.max, field.unit)}`;

  const listed = ranked.slice(0, MAX_LIST_ROWS);

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      {header}
      {controls}
      {provenance}

      <ScrollArea className="min-h-0 flex-1">
        {/* -- the headline this view exists to deliver ---------------------- */}

        <PanelSection
          label="Locality"
          flush
          aside={
            <Tooltip content="Share of all live synapses whose two ends land in the same region">
              <span className="nf-numeric text-[9.5px] text-ink-faint" tabIndex={0}>
                {percent(recurrentShare, 0)} recurrent
              </span>
            </Tooltip>
          }
        >
          <p className="text-[9.5px] leading-snug text-ink-faint">
            A region’s locality index is the share of the connections touching it that have both
            ends inside it. Regions under {DEFAULT_LOCALITY_FLOOR} cells are left out of the
            ranking — locality over a handful of cells says more about the cut than the wiring.
          </p>
          {extremes.considered === 0 ? (
            <p className="text-[10.5px] text-ink-faint">
              Nothing is wired yet, so no region has a locality index.
            </p>
          ) : (
            <div className="mt-0.5 flex flex-col">
              <ExtremeRow
                caption="most local"
                tone={MOST_LOCAL_TONE}
                index={extremes.most}
                regions={regions}
                stats={stats}
                swatches={swatches}
                onSelect={selectRegion}
              />
              <ExtremeRow
                caption="least local"
                tone={LEAST_LOCAL_TONE}
                index={extremes.least}
                regions={regions}
                stats={stats}
                swatches={swatches}
                onSelect={selectRegion}
              />
            </div>
          )}
        </PanelSection>

        <Separator />

        {/* -- the cross-tabulation ------------------------------------------ */}

        <PanelSection
          label="Region × region"
          flush
          aside={
            <SegmentedControl
              size="sm"
              value={metric}
              onChange={setMetric}
              options={METRIC_OPTIONS}
              aria-label="Matrix quantity"
            />
          }
        >
          <div
            ref={setHost}
            tabIndex={0}
            aria-label={`Region by region connection matrix, ${size} by ${size}, ${grouped(
              matrix.synapses,
            )} synapses. Arrow keys move the cursor, enter selects the cells in that projection.`}
            onPointerMove={trackPointer}
            onPointerLeave={() => setHover(null)}
            onPointerDown={clickCell}
            onKeyDown={moveCursor}
            className="relative aspect-square w-full cursor-crosshair touch-none rounded-[3px] focus-visible:outline-1"
          >
            <canvas ref={canvasRef} aria-hidden className="absolute inset-0 block size-full" />
            <canvas
              ref={overlayRef}
              aria-hidden
              className="pointer-events-none absolute inset-0 block size-full"
            />
            {notice !== null ? (
              // Opaque rather than translucent: a degenerate matrix painted
              // behind the message would read as data.
              <div
                className="pointer-events-none absolute inset-0 flex items-center justify-center px-5"
                style={{ backgroundColor: SURFACE_CSS }}
              >
                <span className="max-w-[34ch] text-center text-[10px] leading-relaxed text-ink-faint">
                  {notice}
                </span>
              </div>
            ) : null}
          </div>

          <div className="mt-1 flex items-center gap-2">
            <span
              aria-hidden
              className="h-1.5 min-w-0 flex-1 rounded-[2px] ring-1 ring-white/10"
              style={{ background: RAMP_GRADIENT }}
            />
            <span className="nf-numeric shrink-0 text-[9.5px] text-ink-faint">{rangeLabel}</span>
            <Tooltip content="Map magnitude logarithmically between the smallest and largest non-zero pair. Recurrent totals routinely dwarf every projection, and on a linear scale the whole off-diagonal lands in the bottom decile of the ramp.">
              <label className="flex shrink-0 cursor-pointer items-center gap-1 text-[9.5px] tracking-[0.08em] text-ink-muted uppercase select-none hover:text-ink">
                <Switch size="sm" checked={log} onCheckedChange={setLog} aria-label="log" />
                log
              </label>
            </Tooltip>
          </div>

          <div className="flex items-baseline justify-between gap-2 text-[9px] text-ink-faint">
            <span className="truncate">rows source → columns target · diagonal is recurrent</span>
            <span className="nf-numeric shrink-0">{grouped(matrix.synapses)} placed</span>
          </div>

          <div className="mt-0.5 flex min-h-[30px] flex-col justify-center gap-1">
            {hovered === null ? (
              <p className="text-[10px] leading-tight text-ink-faint">
                {matrix.external > 0 ? (
                  <>{grouped(matrix.external)} synapses reach a cell in no region. </>
                ) : null}
                Hover a cell to read one projection; click to select the neurons that carry it.
              </p>
            ) : (
              <>
                <div className="flex items-center gap-1.5 text-[10.5px]">
                  <Strip colors={swatches[hovered.cell.row]} />
                  <span className="min-w-0 flex-1 truncate text-ink">{hovered.source.label}</span>
                  <span aria-hidden className="shrink-0 text-ink-faint">
                    →
                  </span>
                  <Strip colors={swatches[hovered.cell.col]} />
                  <span className="min-w-0 flex-1 truncate text-ink">{hovered.target.label}</span>
                </div>
                <div className="nf-numeric flex items-baseline gap-2 text-[9.5px] text-ink-faint">
                  <span className="text-ink">{grouped(matrix.count[hovered.at])} syn</span>
                  <span>{fixed(matrix.weight[hovered.at], 1)} nS</span>
                  <span>
                    μ{' '}
                    {matrix.count[hovered.at] > 0
                      ? `${fixed(matrix.weight[hovered.at] / matrix.count[hovered.at], 2)} nS`
                      : '—'}
                  </span>
                  <span className="ml-auto">
                    {compact(hovered.source.size)}×{compact(hovered.target.size)} cells
                  </span>
                </div>
              </>
            )}
          </div>
        </PanelSection>

        <Separator />

        {/* -- the regions themselves ---------------------------------------- */}

        <PanelSection
          label="Regions"
          flush
          aside={
            <SegmentedControl
              size="sm"
              value={order}
              onChange={setOrder}
              options={ORDER_OPTIONS}
              aria-label="Region ordering"
            />
          }
        >
          {size === 0 ? (
            <p className="text-[10.5px] text-ink-faint">Nothing to divide.</p>
          ) : (
            <>
              {/* Column widths mirror RegionRow exactly, including the leading
                  swatch, so the headings sit over the values they name. */}
              <div className="flex items-baseline gap-1.5 px-1.5 text-[9px] tracking-[0.06em] text-ink-faint uppercase">
                <span aria-hidden className="w-3.5 shrink-0" />
                <span className="min-w-0 flex-1">region</span>
                <span className="w-10 shrink-0 text-right">cells</span>
                <span className="w-11 shrink-0 text-right">rate</span>
                <span className="w-9 shrink-0 text-right">local</span>
              </div>
              <ul className="flex flex-col">
                {listed.map((index) => (
                  <li key={regions.regions[index].id}>
                    <RegionRow
                      region={regions.regions[index]}
                      colors={swatches[index]}
                      cells={stats.cells[index]}
                      // The live sample when it is in step with the partition,
                      // and the snapshot taken by the pass until it is — which
                      // is what stops a re-cut from blanking the column.
                      rate={rates.length === size ? rates[index] : stats.meanRate[index]}
                      locality={stats.locality[index]}
                      hint={describeRegion(stats, index)}
                      onSelect={selectRegion}
                    />
                  </li>
                ))}
              </ul>
              {ranked.length > listed.length ? (
                <p className="px-1.5 pt-1 text-[9.5px] text-ink-faint">
                  {grouped(ranked.length - listed.length)} more regions past the list cap.
                </p>
              ) : null}
              {regions.unassigned > 0 ? (
                <p className="px-1.5 pt-1 text-[9.5px] text-ink-faint">
                  {grouped(regions.unassigned)} cells fell in no region
                  {regions.omittedRegions > 0
                    ? `, ${grouped(regions.omittedRegions)} populations past the region cap`
                    : ''}
                  .
                </p>
              ) : null}
            </>
          )}
        </PanelSection>
      </ScrollArea>
    </Panel>
  );
}

/* ------------------------------------------------------------------ rates -- */

/**
 * Mean firing rate per region, resampled off the live buffers.
 *
 * A sample identical to the last one is dropped rather than published: a paused
 * simulation must not re-render the list five times a second to show the same
 * numbers.
 */
function useRegionRates(view: RegionView | null): Float32Array {
  const [rates, setRates] = useState<Float32Array>(EMPTY_RATES);
  const regions = view === null ? null : view.regions;

  useEffect(() => {
    if (regions === null) return;
    const size = regions.regions.length;
    if (size === 0) return;
    const scratch = new Float32Array(size);
    let published: Float32Array | null = null;

    const sample = () => {
      meanRateByRegion(regions, getEngine().buffers, scratch);
      if (published !== null && published.length === size) {
        let same = true;
        for (let i = 0; i < size; i += 1) {
          if (published[i] !== scratch[i]) {
            same = false;
            break;
          }
        }
        if (same) return;
      }
      published = scratch.slice();
      setRates(published);
    };

    sample();
    const id = setInterval(sample, RATE_POLL_MS);
    return () => clearInterval(id);
  }, [regions]);

  return rates;
}

/* --------------------------------------------------------------- selection -- */

/** Publish a selection built from live slots. Returns how many resolved. */
function selectSlots(
  slots: ArrayLike<number>,
  select: (ids: readonly NeuronId[]) => void,
): number {
  const engine = getEngine();
  const count = engine.buffers.neurons.count;
  const ids: NeuronId[] = [];
  for (let i = 0; i < slots.length; i += 1) {
    const slot = slots[i];
    if (slot >= count) continue;
    const id = engine.idOf(slot);
    if (id !== null) ids.push(id as NeuronId);
  }
  if (ids.length > 0) select(ids);
  return ids.length;
}

/* ---------------------------------------------------------------- geometry -- */

function layoutMatrix(extent: number, size: number): Geometry {
  const origin = PAD + TICK + TICK_GAP;
  const plot = Math.max(0, extent - origin - PAD);
  return { origin, cell: size > 0 ? plot / size : 0, plot };
}

function hitTest(geometry: Geometry, size: number, x: number, y: number): Cell | null {
  const { origin, cell, plot } = geometry;
  if (size === 0 || cell <= 0) return null;
  if (x < origin || y < origin || x >= origin + plot || y >= origin + plot) return null;
  const col = Math.min(size - 1, Math.floor((x - origin) / cell));
  const row = Math.min(size - 1, Math.floor((y - origin) / cell));
  return { row, col };
}

/* ------------------------------------------------------------------ paint -- */

interface PaintOptions {
  size: number;
  field: Field;
  geometry: Geometry;
  log: boolean;
  swatches: readonly (readonly string[])[];
}

function normalise(value: number, field: Field, log: boolean): number {
  if (value <= 0 || field.max <= 0) return 0;
  if (!log) return value / field.max;
  const floor = Number.isFinite(field.minPositive) ? field.minPositive : value;
  const span = Math.log(field.max) - Math.log(floor);
  // A matrix whose non-zero cells all carry the same value has no spread to
  // stretch; every one of them sits at the top of the ramp, which is true.
  if (!(span > 0)) return 1;
  return (Math.log(value) - Math.log(floor)) / span;
}

/**
 * Paint one segment of a region's colour strip per sampled member.
 *
 * Below a couple of pixels per segment the samples are indistinguishable, so
 * the strip falls back to the first of them rather than drawing a smear.
 */
function paintStrip(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  colors: readonly string[],
  vertical: boolean,
): void {
  if (colors.length === 0) {
    ctx.fillStyle = UNKNOWN_CSS;
    ctx.fillRect(x, y, width, height);
    return;
  }
  const span = vertical ? height : width;
  const count = Math.max(1, Math.min(colors.length, Math.floor(span / MIN_STRIP_SEGMENT)));
  const step = span / count;
  for (let i = 0; i < count; i += 1) {
    ctx.fillStyle = colors[Math.floor((i * colors.length) / count)];
    // Half a pixel of overlap, so neighbouring segments never leave a seam of
    // background showing through at fractional cell sizes.
    if (vertical) ctx.fillRect(x, y + i * step, width, step + 0.5);
    else ctx.fillRect(x + i * step, y, step + 0.5, height);
  }
}

function paintMatrix(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  options: PaintOptions,
): void {
  const { size, field, geometry, log, swatches } = options;
  const { origin, cell, plot } = geometry;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = SURFACE_CSS;
  ctx.fillRect(0, 0, width, height);
  if (size === 0 || plot <= 0) return;

  ctx.fillStyle = FIELD_CSS;
  ctx.fillRect(origin, origin, plot, plot);

  // Cells first, then the rules on top: at fractional cell sizes a fill and a
  // stroke on the same edge fight, and the fill has to lose.
  //
  // Each fill overruns its cell slightly so neighbours never leave a seam of
  // background showing through at fractional sizes; the clip is what stops the
  // last row and column from overrunning the plot itself.
  const overlap = cell < 1.5 ? 0.5 : 1;
  ctx.save();
  ctx.beginPath();
  ctx.rect(origin, origin, plot, plot);
  ctx.clip();
  for (let row = 0; row < size; row += 1) {
    const base = row * size;
    const y = origin + row * cell;
    for (let col = 0; col < size; col += 1) {
      const value = field.values[base + col];
      if (value <= 0) continue;
      ctx.fillStyle = RAMP_CSS[rampIndex(normalise(value, field, log))];
      ctx.fillRect(origin + col * cell, y, cell + overlap, cell + overlap);
    }
  }
  ctx.restore();

  if (cell >= MIN_GRID_CELL) {
    ctx.strokeStyle = GRID_CSS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < size; i += 1) {
      const at = Math.round(origin + i * cell) + 0.5;
      ctx.moveTo(at, origin);
      ctx.lineTo(at, origin + plot);
      ctx.moveTo(origin, at);
      ctx.lineTo(origin + plot, at);
    }
    ctx.stroke();
  }

  // The recurrent diagonal, marked so it stays identifiable even where a region
  // keeps nothing to itself and its cell is empty.
  ctx.strokeStyle = DIAGONAL_CSS;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(origin, origin);
  ctx.lineTo(origin + plot, origin + plot);
  ctx.stroke();

  // Identity strips along both axes. Every segment is a real cell's colour, so
  // a row here and a cluster of glyphs in the viewport are the same object.
  for (let i = 0; i < size; i += 1) {
    const colors = swatches[i];
    const at = origin + i * cell;
    const span = Math.min(cell + overlap, origin + plot - at);
    paintStrip(ctx, PAD, at, TICK, span, colors, true);
    paintStrip(ctx, at, PAD, span, TICK, colors, false);
  }

  ctx.strokeStyle = GRID_CSS;
  ctx.lineWidth = 1;
  ctx.strokeRect(origin + 0.5, origin + 0.5, plot - 1, plot - 1);
}

function paintCursor(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  geometry: Geometry,
  size: number,
  hover: Cell | null,
): void {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (hover === null || size === 0 || geometry.cell <= 0) return;
  if (hover.row >= size || hover.col >= size) return;

  const { origin, cell, plot } = geometry;
  const x = origin + hover.col * cell;
  const y = origin + hover.row * cell;

  // The bands are what make a cell readable against its own row and column when
  // the grid is too fine to count squares along.
  ctx.fillStyle = HIGHLIGHT_CSS;
  ctx.fillRect(origin, y, plot, cell);
  ctx.fillRect(x, origin, cell, plot);

  ctx.strokeStyle = CURSOR_CSS;
  ctx.lineWidth = 1;
  ctx.strokeRect(
    Math.round(x) + 0.5,
    Math.round(y) + 0.5,
    Math.max(1, Math.round(cell) - 1),
    Math.max(1, Math.round(cell) - 1),
  );
}

/* ------------------------------------------------------------------ pieces -- */

function percent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function formatValue(value: number, unit: string): string {
  if (!Number.isFinite(value)) return '—';
  if (unit === 'syn') return `${compact(value)} syn`;
  return `${value >= 100 ? compact(value) : fixed(value, 1)} nS`;
}

/**
 * A region's colour, sampled from the identity colours of its own members.
 *
 * A derived region has no colour of its own; taking one from the cells inside
 * it is what keeps the chrome and the 3D scene in agreement.
 */
function Strip({ colors }: { colors: readonly string[] | undefined }) {
  const samples = colors === undefined || colors.length === 0 ? [UNKNOWN_CSS] : colors;
  return (
    <span
      aria-hidden
      className="flex h-2.5 w-3.5 shrink-0 overflow-hidden rounded-[2px] ring-1 ring-white/15"
    >
      {samples.map((color, i) => (
        <span key={`${color}-${i}`} className="h-full flex-1" style={{ backgroundColor: color }} />
      ))}
    </span>
  );
}

/** Text and rule colours for the two ends of the locality ranking. */
interface Tone {
  text: string;
  rule: string;
}

const MOST_LOCAL_TONE: Tone = { text: 'text-success', rule: 'bg-success' };
const LEAST_LOCAL_TONE: Tone = { text: 'text-warning', rule: 'bg-warning' };

/**
 * Everything the stats pass measured about one region, as one tooltip line.
 *
 * Built here rather than in the row so the row stays a renderer: the numbers
 * are all columns of the same table and formatting them together keeps their
 * units and precision consistent.
 */
function describeRegion(stats: RegionStats, index: number): string {
  const b = index * 3;
  const wiring =
    `${grouped(stats.internal[index])} synapses stay inside, ` +
    `${grouped(stats.outgoing[index])} leave, ${grouped(stats.incoming[index])} arrive`;
  const degree =
    `mean in/out degree ${fixed(stats.meanInDegree[index], 1)}/` +
    `${fixed(stats.meanOutDegree[index], 1)}`;
  const shape =
    `${fixed(stats.extent[b], 0)}×${fixed(stats.extent[b + 1], 0)}×${fixed(stats.extent[b + 2], 0)} ` +
    `units about (${fixed(stats.centroid[b], 0)}, ${fixed(stats.centroid[b + 1], 0)}, ` +
    `${fixed(stats.centroid[b + 2], 0)}), spread ${fixed(stats.radius[index], 1)}`;
  const balance =
    stats.cells[index] > 0
      ? `${percent(stats.inhibitory[index] / stats.cells[index], 0)} inhibitory`
      : 'no cells';
  return `${wiring} · ${degree} · ${balance} · ${shape}`;
}

interface ExtremeRowProps {
  caption: string;
  tone: Tone;
  index: number;
  regions: RegionSet;
  stats: RegionStats;
  swatches: readonly (readonly string[])[];
  onSelect: (index: number) => void;
}

function ExtremeRow({ caption, tone, index, regions, stats, swatches, onSelect }: ExtremeRowProps) {
  if (index < 0) {
    return (
      <div className="flex items-center gap-1.5 px-1.5 py-1 text-[10.5px] text-ink-faint">
        <span className={cn('w-[62px] shrink-0 text-[9px] tracking-[0.06em] uppercase', tone.text)}>
          {caption}
        </span>
        <span>no region qualifies</span>
      </div>
    );
  }

  const region = regions.regions[index];
  const locality = stats.locality[index];
  return (
    <Tooltip content={describeRegion(stats, index)} side="top">
      <button
        type="button"
        onClick={() => onSelect(index)}
        className={cn(
          'relative flex w-full items-center gap-1.5 overflow-hidden rounded-control px-1.5 py-1 text-left text-[10.5px] transition-colors',
          'hover:bg-panel-raised focus-visible:bg-panel-raised',
        )}
      >
        <span
          className={cn('w-[62px] shrink-0 text-[9px] tracking-[0.06em] uppercase', tone.text)}
        >
          {caption}
        </span>
        <Strip colors={swatches[index]} />
        <span className="min-w-0 flex-1 truncate text-ink">{region.label}</span>
        <span className="nf-numeric shrink-0 text-ink-faint">{compact(stats.cells[index])}</span>
        <span className={cn('nf-numeric w-9 shrink-0 text-right', tone.text)}>
          {percent(locality, 0)}
        </span>
        <span
          aria-hidden
          className={cn('absolute bottom-0 left-0 h-px opacity-60', tone.rule)}
          style={{ width: `${Math.max(0, Math.min(1, locality)) * 100}%` }}
        />
      </button>
    </Tooltip>
  );
}

interface RegionRowProps {
  region: RegionSet['regions'][number];
  colors: readonly string[] | undefined;
  cells: number;
  rate: number;
  locality: number;
  hint: string;
  onSelect: (index: number) => void;
}

function RegionRow({ region, colors, cells, rate, locality, hint, onSelect }: RegionRowProps) {
  return (
    <Tooltip content={hint} side="top">
      <button
        type="button"
        onClick={() => onSelect(region.index)}
        className={cn(
          'relative flex w-full items-center gap-1.5 overflow-hidden rounded-control px-1.5 py-1 text-left text-[10.5px] transition-colors',
          'hover:bg-panel-raised focus-visible:bg-panel-raised',
        )}
      >
        <Strip colors={colors} />
        <span className="min-w-0 flex-1 truncate text-ink">{region.label}</span>
        <span className="nf-numeric w-10 shrink-0 text-right text-ink-faint">
          {compact(cells)}
        </span>
        <span className="nf-numeric w-11 shrink-0 text-right text-ink-muted">
          {fixed(rate, 1)} Hz
        </span>
        <span
          className={cn(
            'nf-numeric w-9 shrink-0 text-right',
            Number.isFinite(locality) ? 'text-accent' : 'text-ink-faint',
          )}
        >
          {percent(locality, 0)}
        </span>
        <span
          aria-hidden
          className="absolute bottom-0 left-0 h-px bg-accent opacity-60"
          style={{
            width: `${Number.isFinite(locality) ? Math.max(0, Math.min(1, locality)) * 100 : 0}%`,
          }}
        />
      </button>
    </Tooltip>
  );
}
