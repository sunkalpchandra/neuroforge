'use client';

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Fingerprint, Play, RefreshCw, Square, TriangleAlert, X } from 'lucide-react';
import {
  Badge,
  Button,
  EmptyState,
  Field,
  IconButton,
  Meter,
  NumberField,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  SegmentedControl,
  Separator,
  Spinner,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { COLORS, MODEL_FROM_CODE, identityColorHex } from '@neuroforge/shared';
import type { NeuronId, NeuronModelKind } from '@neuroforge/shared';

import { getEngine } from '@/lib/runtime';
import { compact, fixed, grouped } from '@/lib/format';
import {
  FingerprintClustering,
  GROUP_OVERFLOW,
  GROUP_UNASSIGNED,
  MODEL_MIXED,
  buildFingerprints,
  fingerprintSignature,
  meanRateByGroup,
  similarCells,
} from '@/lib/similarity';
import type { ClusterRun, Fingerprints, SimilarCell } from '@/lib/similarity';

/** Poll cadence for the topology signature. Two integer reads; effectively free. */
const SIGNATURE_POLL_MS = 500;

/**
 * Cost above which an edit that left the counts alone stops triggering an
 * automatic rebuild. Placing a cell is a discrete act and always worth the pass;
 * dragging a weight slider republishes the document continuously, and past this
 * budget the panel says it is stale rather than stalling the pointer.
 */
const AUTO_REBUILD_BUDGET_MS = 16;

/** How often the live per-population firing rates are resampled. */
const RATE_POLL_MS = 200;

/** Milliseconds of k-means allowed per animation frame before yielding. */
const CLUSTER_FRAME_BUDGET_MS = 12;
const CLUSTER_MAX_ITERATIONS = 64;

const MIN_K = 2;
const MAX_K = 24;
const DEFAULT_K = 6;

/** Members sampled for a group's colour strip. Enough to read, cheap to mount. */
const STRIP_SAMPLES = 26;

const SIMILAR_COUNTS = [
  { value: '8', label: '8', title: '8 nearest cells' },
  { value: '16', label: '16', title: '16 nearest cells' },
  { value: '32', label: '32', title: '32 nearest cells' },
];

/** Row-width abbreviations; the full model names never fit a dense list. */
const MODEL_SHORT: Record<NeuronModelKind, string> = {
  lif: 'LIF',
  izhikevich: 'IZH',
  'hodgkin-huxley': 'HH',
  adex: 'AdEx',
  'morris-lecar': 'ML',
};

const PLACEMENT = 'absolute top-3 bottom-3 right-3 w-[360px]';

const EMPTY_RATES = new Float32Array(0);

/* ------------------------------------------------------------------ helpers -- */

function modelLabel(code: number): string {
  if (code === MODEL_MIXED) return 'mixed';
  const kind = MODEL_FROM_CODE[code];
  return kind === undefined ? '—' : MODEL_SHORT[kind];
}

/** Translucent form of a swatch colour, for matrix cells and row fills. */
function tint(color: string, fraction: number): string {
  const percent = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  return `color-mix(in srgb, ${color} ${percent}%, transparent)`;
}

/** What the chrome shows for one fingerprint dimension pair. */
interface GroupView {
  name: string;
  color: string;
  /** Long-form name for tooltips, which have room the row does not. */
  detail: string;
}

/**
 * Identity colours of a sample of members, evenly strided so the strip shows the
 * spread of a population rather than its first few cells. These are the exact
 * hues the scene draws those cells in, which is what makes a row here and a
 * cluster of glyphs out there recognisably the same object.
 */
function sampleStrips(
  labels: Int32Array,
  labelCount: number,
  sizes: ArrayLike<number>,
  seeds: Uint32Array,
  count: number,
): string[][] {
  const strips: string[][] = [];
  for (let index = 0; index < labelCount; index += 1) strips.push([]);
  if (labelCount === 0) return strips;

  const strides = new Int32Array(labelCount);
  for (let index = 0; index < labelCount; index += 1) {
    strides[index] = Math.max(1, Math.floor((sizes[index] ?? 0) / STRIP_SAMPLES));
  }
  const seen = new Int32Array(labelCount);

  for (let slot = 0; slot < count; slot += 1) {
    const index = labels[slot];
    if (index < 0 || index >= labelCount) continue;
    const rank = seen[index];
    seen[index] = rank + 1;
    const strip = strips[index];
    if (strip.length >= STRIP_SAMPLES) continue;
    if (rank % strides[index] !== 0) continue;
    strip.push(identityColorHex(seeds[slot]));
  }
  return strips;
}

/** Publish a selection built from live slots. Returns how many cells resolved. */
function selectSlots(
  count: number,
  test: (slot: number) => boolean,
  select: (ids: readonly NeuronId[]) => void,
): number {
  const engine = getEngine();
  const live = Math.min(count, engine.buffers.neurons.count);
  const ids: NeuronId[] = [];
  for (let slot = 0; slot < live; slot += 1) {
    if (!test(slot)) continue;
    const id = engine.idOf(slot);
    if (id !== null) ids.push(id as NeuronId);
  }
  select(ids);
  return ids.length;
}

/* -------------------------------------------------------------------- panel -- */

export interface CellTypesPanelProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Replaces the default docking entirely. */
  className?: string;
}

/**
 * Cell typing by connectivity.
 *
 * The three sections answer the same question at three scales: what groups exist
 * and how they behave, which individual cells are wired like the one in hand,
 * and what groups the wiring itself proposes when the user's labels are taken
 * away. The last of those is the interesting one — the contingency readout is
 * there so a disagreement between the clustering and the labels is visible
 * rather than glossed over.
 */
export function CellTypesPanel({ open = true, onClose, className }: CellTypesPanelProps) {
  // Sliced rather than taken whole: the store republishes the document on every
  // camera orbit frame, and this panel has no interest in the camera.
  const populations = useEditor((s) => s.circuit.populations);
  const documentNeurons = useEditor((s) => s.circuit.neurons);
  const documentSynapses = useEditor((s) => s.circuit.synapses);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [prints, setPrints] = useState<Fingerprints | null>(null);
  const [pending, setPending] = useState(false);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const populationCount = populations.length;
  const signatureRef = useRef('');
  const frameRef = useRef(0);
  const busyRef = useRef(false);
  const dirtyRef = useRef(true);
  /** Cost of the last completed pass; null until one has run. */
  const costRef = useRef<number | null>(null);

  const rebuild = useCallback(() => {
    if (busyRef.current) return;
    busyRef.current = true;
    setPending(true);
    // Yield one frame so the busy state paints before the blocking pass.
    frameRef.current = requestAnimationFrame(() => {
      try {
        const buffers = getEngine().buffers;
        signatureRef.current = fingerprintSignature(buffers, populationCount);
        dirtyRef.current = false;
        const next = buildFingerprints(buffers, populationCount);
        costRef.current = next.computeMs;
        setPrints(next);
        setStale(false);
        setError(null);
      } catch (cause) {
        // The busy latch below is what stops the poll from stampeding, so it has
        // to be released on the failing path too.
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setPending(false);
        busyRef.current = false;
      }
    });
  }, [populationCount]);

  // Rewiring an endpoint, retuning a weight and renaming a population all leave
  // the counts untouched, which the signature cannot see. Flagging the document
  // here is what keeps the fingerprints describing the network that exists.
  useEffect(() => {
    dirtyRef.current = true;
  }, [documentNeurons, documentSynapses, populations]);

  useEffect(() => {
    if (!open) return;
    // The engine is loaded by an effect in an ancestor, which commits after this
    // one; polling is what makes the panel correct regardless of that ordering.
    const poll = () => {
      if (busyRef.current) return;
      const resized =
        fingerprintSignature(getEngine().buffers, populationCount) !== signatureRef.current;
      if (!resized && !dirtyRef.current) return;
      if (resized || costRef.current === null || costRef.current <= AUTO_REBUILD_BUDGET_MS) {
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
      cancelAnimationFrame(frameRef.current);
      busyRef.current = false;
      setPending(false);
    };
  }, [open, populationCount, rebuild]);

  const groupViews = useMemo<GroupView[]>(() => {
    if (prints === null) return [];
    return prints.groups.map((group, index) => {
      if (group.populationIndex === GROUP_UNASSIGNED) {
        return {
          name: 'Unassigned',
          color: COLORS.textFaint,
          detail: 'Cells that belong to no population',
        };
      }
      if (group.populationIndex === GROUP_OVERFLOW) {
        return {
          name: `${grouped(group.populations)} smaller populations`,
          color: COLORS.textMuted,
          detail: `${grouped(group.populations)} populations folded into one fingerprint dimension`,
        };
      }
      const population = populations[group.populationIndex];
      if (population === undefined) {
        return {
          name: `Population ${index + 1}`,
          color: COLORS.textFaint,
          detail: 'This population left the document since the last rebuild',
        };
      }
      // The population carries its own morphology seed, so it gets a saturated
      // hue of its own from the same generator the cells are drawn with.
      return {
        name: population.name,
        color: population.color ?? identityColorHex(population.morphology.seed),
        detail: population.name,
      };
    });
  }, [prints, populations]);

  const selectGroup = useCallback(
    (index: number) => {
      if (prints === null) return;
      selectSlots(prints.count, (slot) => prints.groupOf[slot] === index, select);
    },
    [prints, select],
  );

  const selectCluster = useCallback(
    (run: ClusterRun, cluster: number) => {
      if (prints === null) return;
      selectSlots(prints.count, (slot) => run.assignment[slot] === cluster, select);
    },
    [prints, select],
  );

  if (!open) return null;

  const placement = className ?? PLACEMENT;

  const header = (
    <PanelHeader
      title="Cell types"
      subtitle={
        prints === null
          ? undefined
          : `${grouped(prints.count)} cells · ${grouped(prints.groups.length)} populations · ${
              prints.dim
            }-D fingerprint`
      }
      icon={<Fingerprint />}
      actions={
        <>
          {error !== null ? (
            <Tooltip content={`The last pass failed: ${error}`}>
              <Badge variant="danger" size="sm" tabIndex={0}>
                failed
              </Badge>
            </Tooltip>
          ) : stale ? (
            <Tooltip content="The circuit changed after this pass, and rebuilding the fingerprints costs more than a frame. Rebuild to bring them back in step.">
              <Badge variant="warning" size="sm" tabIndex={0}>
                stale
              </Badge>
            </Tooltip>
          ) : prints !== null ? (
            <Tooltip content="Time taken to fingerprint every cell in the running network">
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(prints.computeMs, 1)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Rebuild the fingerprints from the running network">
            <IconButton label="Rebuild fingerprints" size="sm" onClick={rebuild} disabled={pending}>
              <RefreshCw className={pending ? 'animate-spin' : undefined} />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close cell types panel" size="sm" onClick={onClose}>
              <X />
            </IconButton>
          ) : null}
        </>
      }
    />
  );

  if (prints === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        {error === null ? (
          <EmptyState
            icon={<Spinner size={18} />}
            title="Fingerprinting the network"
            description="Measuring what every cell connects to, in and out."
          />
        ) : (
          <EmptyState
            icon={<TriangleAlert className="text-danger" />}
            title="Could not fingerprint the network"
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

  if (prints.count === 0) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        <EmptyState
          icon={<Fingerprint />}
          title="No cells to type"
          description="Place neurons and wire them together. This panel then groups them by what they connect to rather than by what they are called."
        />
      </Panel>
    );
  }

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      {header}

      <ScrollArea className="min-h-0 flex-1">
        <PopulationsSection
          prints={prints}
          groups={groupViews}
          live={open}
          onSelect={selectGroup}
        />

        <Separator />

        <SimilarSection
          prints={prints}
          groups={groupViews}
          selection={selection}
          onSelect={select}
        />

        <Separator />

        <ClustersSection
          prints={prints}
          groups={groupViews}
          onSelectCluster={selectCluster}
        />
      </ScrollArea>
    </Panel>
  );
}

/* -------------------------------------------------------------- populations -- */

/**
 * Mean firing rate per population, resampled off the live buffers.
 *
 * A sample identical to the last one is dropped rather than published: a paused
 * simulation must not re-render the list five times a second to show the same
 * numbers.
 */
function useGroupRates(prints: Fingerprints, enabled: boolean): Float32Array {
  const [rates, setRates] = useState<Float32Array>(EMPTY_RATES);

  useEffect(() => {
    const size = prints.groups.length;
    if (!enabled || size === 0) return;
    const scratch = new Float32Array(size);
    let published: Float32Array | null = null;

    const sample = () => {
      meanRateByGroup(getEngine().buffers, prints, scratch);
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
  }, [prints, enabled]);

  return rates;
}

interface PopulationsSectionProps {
  prints: Fingerprints;
  groups: readonly GroupView[];
  live: boolean;
  onSelect: (index: number) => void;
}

function PopulationsSection({ prints, groups, live, onSelect }: PopulationsSectionProps) {
  const rates = useGroupRates(prints, live);

  const strips = useMemo(() => {
    const neurons = getEngine().buffers.neurons;
    const sizes = prints.groups.map((group) => group.size);
    // Clamped: a reload can leave the fingerprints one poll longer than the
    // buffers they were built from, and a seed read past the end is not a colour.
    const covered = Math.min(prints.count, neurons.count);
    return sampleStrips(prints.groupOf, prints.groups.length, sizes, neurons.seed, covered);
  }, [prints]);

  let peakRate = 0;
  for (let i = 0; i < rates.length; i += 1) if (rates[i] > peakRate) peakRate = rates[i];

  return (
    <PanelSection
      label="Populations"
      flush
      aside={
        <span className="nf-numeric text-[9.5px] text-ink-faint">
          {grouped(prints.synapses)} wired
        </span>
      }
    >
      {groups.length === 0 ? (
        <p className="text-[10.5px] text-ink-faint">Nothing to group.</p>
      ) : (
        <ul className="flex flex-col">
          {groups.map((view, index) => {
            const group = prints.groups[index];
            return (
              <li key={`${view.name}-${index}`}>
                <PopulationRow
                  index={index}
                  name={view.name}
                  detail={view.detail}
                  color={view.color}
                  size={group.size}
                  inhibitory={group.inhibitory}
                  model={group.model}
                  meanInDegree={group.meanInDegree}
                  meanOutDegree={group.meanOutDegree}
                  rate={index < rates.length ? rates[index] : Number.NaN}
                  peakRate={peakRate}
                  strip={strips[index]}
                  onSelect={onSelect}
                />
              </li>
            );
          })}
        </ul>
      )}
    </PanelSection>
  );
}

interface PopulationRowProps {
  index: number;
  name: string;
  detail: string;
  color: string;
  size: number;
  inhibitory: number;
  model: number;
  meanInDegree: number;
  meanOutDegree: number;
  rate: number;
  peakRate: number;
  strip: readonly string[];
  onSelect: (index: number) => void;
}

/**
 * One population. Memoised because the section re-renders at the rate poll and
 * only the rows whose mean rate actually moved need reconciling — which is also
 * why the click handler takes the index rather than closing over it.
 */
const PopulationRow = memo(function PopulationRow({
  index,
  name,
  detail,
  color,
  size,
  inhibitory,
  model,
  meanInDegree,
  meanOutDegree,
  rate,
  peakRate,
  strip,
  onSelect,
}: PopulationRowProps) {
  const polarity = inhibitory === 0 ? 'E' : inhibitory === size ? 'I' : 'E/I';
  const polarityTone =
    inhibitory === 0 ? 'text-accent' : inhibitory === size ? 'text-secondary' : 'text-ink-faint';

  return (
    <button
      type="button"
      onClick={() => onSelect(index)}
      title={`${detail} — select all ${grouped(size)} cells`}
      className={cn(
        'relative flex w-full flex-col gap-0.5 overflow-hidden rounded-control px-1.5 py-1 text-left',
        'transition-colors hover:bg-panel-raised focus-visible:bg-panel-raised',
      )}
    >
      <span className="flex w-full items-center gap-1.5">
        <span
          aria-hidden
          className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
          style={{ backgroundColor: color }}
        />
        <span className="min-w-0 flex-1 truncate text-[11px] text-ink">{name}</span>
        <span className={cn('shrink-0 text-[9px] font-semibold tracking-[0.06em]', polarityTone)}>
          {polarity}
        </span>
        <span className="nf-numeric w-8 shrink-0 text-right text-[9.5px] text-ink-faint">
          {modelLabel(model)}
        </span>
        <span className="nf-numeric w-11 shrink-0 text-right text-[10.5px] text-ink">
          {grouped(size)}
        </span>
      </span>

      <span className="flex w-full items-center gap-1.5 pl-4 text-[9.5px] text-ink-faint">
        <span className="nf-numeric shrink-0">
          in {fixed(meanInDegree, 1)}
          <span className="px-0.5 text-ink-faint/60">/</span>
          out {fixed(meanOutDegree, 1)}
        </span>
        <span className="min-w-0 flex-1" />
        <span
          className={cn('nf-numeric shrink-0', rate >= 0.05 ? 'text-ink-muted' : 'text-ink-faint')}
        >
          {fixed(rate, 2)} Hz
        </span>
      </span>

      {/* The exact hues these cells are drawn in, strided across the population. */}
      {strip.length > 0 ? (
        <span aria-hidden className="mt-0.5 flex h-[3px] w-full gap-px pl-4">
          {strip.map((hue, at) => (
            <span
              key={at}
              className="min-w-0 flex-1 rounded-[1px]"
              style={{ backgroundColor: hue }}
            />
          ))}
        </span>
      ) : null}

      <span
        aria-hidden
        className="absolute bottom-0 left-0 h-px opacity-70"
        style={{
          width: peakRate > 0 && Number.isFinite(rate) ? `${(rate / peakRate) * 100}%` : '0%',
          backgroundColor: color,
        }}
      />
    </button>
  );
});

/* ------------------------------------------------------------ similar cells -- */

interface SimilarRow extends SimilarCell {
  id: NeuronId | null;
  label: string;
  seed: number;
  inhibitory: boolean;
}

interface SimilarSectionProps {
  prints: Fingerprints;
  groups: readonly GroupView[];
  selection: readonly NeuronId[];
  onSelect: (ids: readonly NeuronId[], additive?: boolean) => void;
}

function SimilarSection({ prints, groups, selection, onSelect }: SimilarSectionProps) {
  const documentNeurons = useEditor((s) => s.circuit.neurons);
  const [count, setCount] = useState('16');

  // The first of the selection is the query cell; a multi-cell selection has no
  // single fingerprint to rank against, and the note below says so.
  const primary = selection.length > 0 ? selection[0] : null;
  // Resolved against `prints` as well as the id, because a reload renumbers
  // every slot and the ranking has to follow the network it was built from.
  const slot = useMemo(
    () => (primary === null ? -1 : getEngine().slotOf(primary)),
    [primary, prints],
  );

  const ranked = useMemo(() => {
    if (slot < 0 || slot >= prints.count) return [];
    return similarCells(prints, slot, Number(count));
  }, [prints, slot, count]);

  const rows = useMemo<SimilarRow[]>(() => {
    if (ranked.length === 0) return [];
    const engine = getEngine();
    const neurons = engine.buffers.neurons;
    const list: SimilarRow[] = ranked.map((entry) => {
      // `idOf` returns null exactly when the slot is past the live network, so
      // it doubles as the bounds check for the columns read beside it.
      const id = engine.idOf(entry.slot);
      return {
        ...entry,
        id: id === null ? null : (id as NeuronId),
        seed: id === null ? 0 : neurons.seed[entry.slot],
        inhibitory: id !== null && neurons.polarity[entry.slot] === 1,
        label: id === null ? `slot ${entry.slot}` : id.slice(0, 8),
      };
    });
    // One pass over the document rather than a lookup per row, which would be
    // O(rows · neurons) on a large circuit. It stops as soon as every row is
    // named, so it usually walks a short prefix.
    const index = new Map<string, number>();
    list.forEach((row, at) => {
      if (row.id !== null) index.set(row.id, at);
    });
    let remaining = index.size;
    for (const neuron of documentNeurons) {
      if (remaining === 0) break;
      const at = index.get(neuron.id);
      if (at === undefined) continue;
      remaining -= 1;
      if (neuron.label.length > 0) list[at].label = neuron.label;
    }
    return list;
  }, [ranked, documentNeurons]);

  const queryLabel = useMemo(() => {
    if (primary === null) return '';
    for (const neuron of documentNeurons) {
      if (neuron.id !== primary) continue;
      return neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8);
    }
    return primary.slice(0, 8);
  }, [primary, documentNeurons]);

  const aside = (
    <SegmentedControl
      size="sm"
      value={count}
      onChange={setCount}
      options={SIMILAR_COUNTS}
      aria-label="Number of similar cells to rank"
    />
  );

  if (primary === null) {
    return (
      <PanelSection label="Similar cells">
        <EmptyState
          compact
          icon={<Fingerprint size={14} />}
          title="No cell selected"
          description="Pick a cell in the scene or in any list. This section then ranks the whole network by how closely its wiring matches — the same comparison connectomics uses to find a cell type without a label."
        />
      </PanelSection>
    );
  }

  if (slot < 0 || slot >= prints.count) {
    return (
      <PanelSection label="Similar cells" aside={aside}>
        <p className="text-[10.5px] leading-snug text-ink-faint">
          The selected cell is not in the running network yet. It appears here once the engine has
          loaded the edit that created it.
        </p>
      </PanelSection>
    );
  }

  const queryColor = identityColorHex(getEngine().buffers.neurons.seed[slot]);
  const queryGroup = groups[prints.groupOf[slot]];
  const unconnected = prints.connected[slot] === 0;

  return (
    <PanelSection label="Similar cells" aside={aside}>
      <div className="mb-1 flex items-center gap-1.5 rounded-control bg-white/[0.03] px-1.5 py-1">
        <span
          aria-hidden
          className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
          style={{ backgroundColor: queryColor }}
        />
        <span className="min-w-0 flex-1 truncate text-[10.5px] text-ink">{queryLabel}</span>
        {queryGroup !== undefined ? (
          <span className="max-w-[38%] shrink-0 truncate text-[9.5px] text-ink-faint">
            {queryGroup.name}
          </span>
        ) : null}
        <span className="nf-numeric shrink-0 text-[9.5px] text-ink-faint">
          {grouped(prints.inDegree[slot])}
          <span className="text-ink-faint/60">/</span>
          {grouped(prints.outDegree[slot])}
        </span>
      </div>

      {selection.length > 1 ? (
        <p className="mb-1 text-[9.5px] leading-snug text-ink-faint">
          {grouped(selection.length)} cells are selected; the ranking is for the first of them.
        </p>
      ) : null}

      {unconnected ? (
        <p className="text-[10.5px] leading-snug text-ink-faint">
          This cell has no synapses, so it has no connectivity fingerprint. Wire it to something and
          it becomes comparable.
        </p>
      ) : rows.length === 0 ? (
        <p className="text-[10.5px] leading-snug text-ink-faint">
          No other cell shares a partner population with this one.
        </p>
      ) : (
        <ul className="flex flex-col">
          {rows.map((row) => {
            const colour = identityColorHex(row.seed);
            const group = groups[row.group];
            return (
              <li key={row.slot}>
                <button
                  type="button"
                  disabled={row.id === null}
                  onClick={(event) => {
                    if (row.id !== null) {
                      onSelect([row.id], event.shiftKey || event.metaKey || event.ctrlKey);
                    }
                  }}
                  title={`${row.label} — cosine ${fixed(row.score, 4)} · in ${grouped(
                    row.inDegree,
                  )} / out ${grouped(row.outDegree)}`}
                  className={cn(
                    'relative flex w-full items-center gap-1.5 overflow-hidden rounded-control px-1.5 py-1 text-left text-[10.5px]',
                    'transition-colors hover:bg-panel-raised focus-visible:bg-panel-raised',
                    'disabled:pointer-events-none disabled:opacity-50',
                  )}
                >
                  <span
                    aria-hidden
                    className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
                    style={{ backgroundColor: colour }}
                  />
                  <span className="min-w-0 flex-1 truncate text-ink">{row.label}</span>
                  {row.inhibitory ? (
                    <span className="shrink-0 text-[8.5px] font-semibold tracking-[0.08em] text-secondary">
                      INH
                    </span>
                  ) : null}
                  {group !== undefined ? (
                    <span className="max-w-[34%] shrink-0 truncate text-[9.5px] text-ink-faint">
                      {group.name}
                    </span>
                  ) : null}
                  <span className="nf-numeric w-9 shrink-0 text-right text-ink">
                    {fixed(row.score, 3)}
                  </span>
                  <span
                    aria-hidden
                    className="absolute bottom-0 left-0 h-px opacity-80"
                    style={{ width: `${row.score * 100}%`, backgroundColor: colour }}
                  />
                </button>
              </li>
            );
          })}
        </ul>
      )}

      <p className="mt-1 text-[9.5px] leading-snug text-ink-faint">
        Cosine similarity over the weight a cell sends to and receives from each population. Both
        halves are normalised, so a cell is compared on the shape of its wiring rather than on how
        much of it there is.
      </p>
    </PanelSection>
  );
}

/* ---------------------------------------------------------------- clusters -- */

interface ClusterProgress {
  iterations: number;
  moved: number;
}

interface ClustersSectionProps {
  prints: Fingerprints;
  groups: readonly GroupView[];
  onSelectCluster: (run: ClusterRun, cluster: number) => void;
}

function ClustersSection({ prints, groups, onSelectCluster }: ClustersSectionProps) {
  const [k, setK] = useState(DEFAULT_K);
  const [run, setRun] = useState<ClusterRun | null>(null);
  const [progress, setProgress] = useState<ClusterProgress | null>(null);

  /** Bumped by every cancel, so an abandoned run cannot publish over a newer one. */
  const generationRef = useRef(0);
  const frameRef = useRef(0);

  const cancel = useCallback(() => {
    generationRef.current += 1;
    cancelAnimationFrame(frameRef.current);
    setProgress(null);
  }, []);

  // A result describes the fingerprints it was computed from; once those are
  // rebuilt it is a statement about a network that no longer exists.
  useEffect(() => {
    cancel();
    setRun(null);
  }, [prints, cancel]);

  useEffect(() => cancel, [cancel]);

  const start = useCallback(() => {
    cancel();
    if (prints.connectedCount === 0) return;
    const generation = generationRef.current;
    setRun(null);
    setProgress({ iterations: 0, moved: 0 });

    let runner: FingerprintClustering | null = null;
    const tick = () => {
      if (generationRef.current !== generation) return;
      const deadline = performance.now() + CLUSTER_FRAME_BUDGET_MS;
      // k-means++ seeding is itself k passes over every fingerprint, so it runs
      // on this frame rather than in the click handler: the progress meter has
      // to paint before anything blocks.
      if (runner === null) {
        runner = new FingerprintClustering(prints, k, {
          maxIterations: CLUSTER_MAX_ITERATIONS,
        });
        // Seeding is `k` passes over every fingerprint and can consume the whole
        // budget by itself. Starting an iteration on top of it is what turns one
        // long frame into a visibly dropped one, so the first pass waits.
        if (!runner.done && performance.now() >= deadline) {
          frameRef.current = requestAnimationFrame(tick);
          return;
        }
      }
      const active: FingerprintClustering = runner;
      // At least one iteration per frame, so a network whose single pass is
      // longer than the budget still makes progress instead of spinning. Beyond
      // the first, another only starts when the budget can pay for one the
      // length of the last: checking the clock only afterwards is what lets a
      // twelve-millisecond budget spend twenty-four.
      let done = false;
      while (!done) {
        const started = performance.now();
        done = active.step();
        const finished = performance.now();
        if (finished + (finished - started) > deadline) break;
      }

      if (done) {
        setProgress(null);
        setRun(active.result());
        return;
      }
      const state = active.progress;
      setProgress({ iterations: state.iterations, moved: state.moved });
      frameRef.current = requestAnimationFrame(tick);
    };

    frameRef.current = requestAnimationFrame(tick);
  }, [cancel, k, prints]);

  const order = useMemo(() => {
    if (run === null) return [];
    const indices: number[] = [];
    for (let c = 0; c < run.k; c += 1) if (run.sizes[c] > 0) indices.push(c);
    indices.sort((a, b) => run.sizes[b] - run.sizes[a] || a - b);
    return indices;
  }, [run]);

  const strips = useMemo(() => {
    if (run === null) return [];
    const neurons = getEngine().buffers.neurons;
    const covered = Math.min(prints.count, neurons.count);
    return sampleStrips(run.assignment, run.k, run.sizes, neurons.seed, covered);
  }, [run, prints]);

  const running = progress !== null;
  const maxK = Math.max(MIN_K, Math.min(MAX_K, prints.connectedCount));

  return (
    <PanelSection
      label="Clusters"
      aside={
        run !== null ? (
          <>
            <Tooltip
              content={`Seeded at 0x${run.seed.toString(16)}, so the same circuit and the same k always produce the same clusters.`}
            >
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {grouped(run.iterations)} iter
              </Badge>
            </Tooltip>
            <Tooltip
              content={
                run.converged
                  ? `Assignments stopped moving after ${grouped(run.iterations)} iterations.`
                  : `Still moving ${grouped(run.unsettled)} cells when the ${grouped(
                      run.maxIterations,
                    )}-iteration cap was reached. Treat the split as provisional.`
              }
            >
              <Badge variant={run.converged ? 'success' : 'warning'} size="sm" tabIndex={0}>
                {run.converged ? 'converged' : 'capped'}
              </Badge>
            </Tooltip>
          </>
        ) : undefined
      }
    >
      <div className="flex items-end gap-1.5">
        <Field label="Clusters — k" orientation="column" className="min-w-0 flex-1">
          <NumberField
            value={k}
            onChange={setK}
            min={MIN_K}
            max={maxK}
            step={1}
            precision={0}
            defaultValue={DEFAULT_K}
            aria-label="Number of clusters"
            disabled={running}
          />
        </Field>
        {running ? (
          <Button size="sm" variant="secondary" icon={<Square />} onClick={cancel}>
            Stop
          </Button>
        ) : (
          <Button
            size="sm"
            variant="primary"
            icon={<Play />}
            onClick={start}
            disabled={prints.connectedCount === 0}
          >
            Run
          </Button>
        )}
      </div>

      {prints.connectedCount === 0 ? (
        <p className="mt-1 text-[10.5px] leading-snug text-ink-faint">
          Nothing is wired yet, so there are no fingerprints to cluster.
        </p>
      ) : null}

      {progress !== null ? (
        <div className="mt-1.5 flex flex-col gap-1">
          <Meter
            size="sm"
            tone="accent"
            value={progress.iterations / CLUSTER_MAX_ITERATIONS}
            label={<span className="text-[9.5px] text-ink-faint">k-means</span>}
            valueLabel={
              <span className="nf-numeric text-[9.5px] text-ink-muted">
                iter {grouped(progress.iterations)}/{grouped(CLUSTER_MAX_ITERATIONS)} ·{' '}
                {grouped(progress.moved)} moved
              </span>
            }
          />
        </div>
      ) : null}

      {run !== null ? (
        <>
          <div className="mt-1.5 grid grid-cols-3 gap-x-2 gap-y-1">
            <ClusterStat
              label="Clustered"
              value={grouped(run.members)}
              hint={`${grouped(run.excluded)} cells were excluded for having no synapses`}
            />
            <ClusterStat
              label="Purity"
              value={Number.isFinite(run.purity) ? `${fixed(run.purity * 100, 0)}%` : '—'}
              hint={
                Number.isFinite(run.purity)
                  ? "Share of cells whose cluster is mostly made of that cell's own population"
                  : 'Undefined: with fewer than two populations carrying cells, every cluster agrees with the only label there is.'
              }
            />
            <ClusterStat
              label="NMI"
              value={Number.isFinite(run.nmi) ? fixed(run.nmi, 3) : '—'}
              hint={
                Number.isFinite(run.nmi)
                  ? 'Normalised mutual information between the clustering and the population labels. 1 means each determines the other; 0 means they are independent.'
                  : 'Undefined: mutual information needs at least two occupied clusters and two occupied populations.'
              }
            />
          </div>

          <ul className="mt-1.5 flex flex-col">
            {order.map((cluster) => {
              const dominant = run.dominant[cluster];
              const view = dominant >= 0 ? groups[dominant] : undefined;
              const colour = view?.color ?? COLORS.textFaint;
              const share = run.dominantShare[cluster];
              return (
                <li key={cluster}>
                  <button
                    type="button"
                    onClick={() => onSelectCluster(run, cluster)}
                    title={`Cluster ${cluster + 1} — select all ${grouped(
                      run.sizes[cluster],
                    )} cells`}
                    className={cn(
                      'relative flex w-full flex-col gap-0.5 overflow-hidden rounded-control px-1.5 py-1 text-left',
                      'transition-colors hover:bg-panel-raised focus-visible:bg-panel-raised',
                    )}
                  >
                    <span className="flex w-full items-center gap-1.5 text-[10.5px]">
                      <span
                        aria-hidden
                        className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
                        style={{ backgroundColor: colour }}
                      />
                      <span className="nf-numeric shrink-0 text-ink-faint">
                        C{cluster + 1}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-ink">
                        {view?.name ?? 'no dominant population'}
                      </span>
                      <span className="nf-numeric shrink-0 text-ink-faint">
                        {fixed(share * 100, 0)}%
                      </span>
                      <span className="nf-numeric w-11 shrink-0 text-right text-ink">
                        {grouped(run.sizes[cluster])}
                      </span>
                    </span>
                    {strips[cluster] !== undefined && strips[cluster].length > 0 ? (
                      <span aria-hidden className="flex h-[3px] w-full gap-px pl-4">
                        {strips[cluster].map((hue, at) => (
                          <span
                            key={at}
                            className="min-w-0 flex-1 rounded-[1px]"
                            style={{ backgroundColor: hue }}
                          />
                        ))}
                      </span>
                    ) : null}
                    <span
                      aria-hidden
                      className="absolute bottom-0 left-0 h-px opacity-70"
                      style={{
                        width: `${
                          run.members > 0 ? (run.sizes[cluster] / run.members) * 100 : 0
                        }%`,
                        backgroundColor: colour,
                      }}
                    />
                  </button>
                </li>
              );
            })}
          </ul>

          <Contingency run={run} groups={groups} order={order} />
        </>
      ) : null}
    </PanelSection>
  );
}

interface ClusterStatProps {
  label: string;
  value: string;
  hint: string;
}

function ClusterStat({ label, value, hint }: ClusterStatProps) {
  return (
    <Tooltip content={hint} side="top">
      <div
        role="group"
        tabIndex={0}
        aria-label={`${label}: ${value}`}
        className="flex flex-col gap-0.5 rounded-sm focus-visible:outline-1"
      >
        <span className="truncate text-[9px] font-medium uppercase tracking-[0.07em] text-ink-faint">
          {label}
        </span>
        <span className="nf-numeric text-[11.5px] leading-none text-ink">{value}</span>
      </div>
    </Tooltip>
  );
}

interface ContingencyProps {
  run: ClusterRun;
  groups: readonly GroupView[];
  order: readonly number[];
}

/**
 * Clusters against populations.
 *
 * A cell is shaded by the share of its cluster that fell into that population,
 * so a diagonal-looking matrix means the wiring agrees with the labels and a
 * smeared one means it does not. That disagreement is the point of the readout:
 * it is where a label is doing less work than the user thinks.
 */
function Contingency({ run, groups, order }: ContingencyProps) {
  const g = groups.length;
  if (g === 0 || order.length === 0) return null;

  return (
    <div className="mt-2 flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[9px] font-semibold uppercase tracking-[0.09em] text-ink-muted">
          Cluster × population
        </span>
        <span className="nf-numeric text-[9px] text-ink-faint">cells</span>
      </div>
      <div className="-mx-1 overflow-x-auto px-1">
        <table className="w-max border-collapse">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-panel pr-1 text-left align-bottom">
                <span className="sr-only">Cluster</span>
              </th>
              {groups.map((view, index) => (
                <th key={index} className="px-px pb-1 align-bottom">
                  <Tooltip content={view.detail} side="top">
                    <span
                      tabIndex={0}
                      className="mx-auto block h-1.5 w-6 rounded-[1px] ring-1 ring-white/10"
                      style={{ backgroundColor: view.color }}
                    />
                  </Tooltip>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {order.map((cluster) => {
              const size = run.sizes[cluster];
              return (
                <tr key={cluster}>
                  <th className="sticky left-0 z-10 bg-panel pr-1 text-right">
                    <span className="nf-numeric text-[9px] font-normal text-ink-faint">
                      C{cluster + 1}
                    </span>
                  </th>
                  {groups.map((view, index) => {
                    const value = run.contingency[cluster * g + index];
                    const share = size > 0 ? value / size : 0;
                    return (
                      <td
                        key={index}
                        title={`Cluster ${cluster + 1} · ${view.name}: ${grouped(
                          value,
                        )} cells, ${fixed(share * 100, 0)}% of the cluster`}
                        className={cn(
                          'nf-numeric h-4 w-6 px-px text-center text-[9px]',
                          value === 0
                            ? 'text-ink-faint/40'
                            : share >= 0.5
                              ? 'text-ink'
                              : 'text-ink-muted',
                        )}
                        style={{
                          backgroundColor:
                            value === 0 ? 'transparent' : tint(view.color, 0.1 + share * 0.65),
                        }}
                      >
                        {value === 0 ? '·' : compact(value)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[9.5px] leading-snug text-ink-faint">
        Rows are connectivity clusters, columns the populations you assigned. A block-diagonal
        matrix means the wiring agrees with your labels; a smeared one means the two disagree, and
        the clusters are describing something the labels do not.
      </p>
    </div>
  );
}
