'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Network, RefreshCw, TriangleAlert, X } from 'lucide-react';
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
  Spinner,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { RECEPTOR_COLORS, RECEPTOR_LABELS, identityColorHex } from '@neuroforge/shared';
import type { NeuronId } from '@neuroforge/shared';

import { getEngine } from '@/lib/runtime';
import { fixed, grouped } from '@/lib/format';
import { computeGraphMetrics, graphSignature } from '@/lib/graph-metrics';
import type { GraphMetrics } from '@/lib/graph-metrics';

/** Poll cadence for the topology signature. Two integer reads; effectively free. */
const SIGNATURE_POLL_MS = 500;

/**
 * Cost above which an edit that left the neuron and synapse counts alone stops
 * triggering an automatic recompute.
 *
 * Adding or deleting cells is a discrete act and always worth the pass. Dragging
 * a weight slider is not: it republishes the document continuously, and a pass
 * longer than a frame would stall the pointer twice a second. Past this budget
 * the panel marks itself stale and waits to be asked instead of lying quietly.
 */
const AUTO_RECOMPUTE_BUDGET_MS = 16;

const HIST_BINS = 44;
const HIST_WIDTH = 316;
const HIST_HEIGHT = 54;
/** Rows left below the baseline for the axis rule. */
const HIST_BASELINE = HIST_HEIGHT - 2;

type Scale = 'lin' | 'log';

const SCALE_OPTIONS = [
  { value: 'lin' as const, label: 'linear', title: 'Linear count axis' },
  { value: 'log' as const, label: 'log', title: 'Logarithmic count axis' },
];

export interface NetworkAnalysisProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Overrides the default placement. */
  className?: string;
}

/**
 * Connectome statistics for the running network.
 *
 * Everything here is measured off the live simulation buffers rather than the
 * document, and only when the circuit changes or the user asks — the metrics are
 * a few tens of milliseconds of graph work at scale, which is fine on demand and
 * ruinous per frame. Past `AUTO_RECOMPUTE_BUDGET_MS` even the on-change pass is
 * withheld and the panel says it is stale instead, so a slider drag never pays
 * for a graph traversal it did not ask for.
 */
export function NetworkAnalysis({ open = true, onClose, className }: NetworkAnalysisProps) {
  const circuit = useEditor((s) => s.circuit);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [metrics, setMetrics] = useState<GraphMetrics | null>(null);
  const [scale, setScale] = useState<Scale>('log');
  const [pending, setPending] = useState(false);
  const [stale, setStale] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const signatureRef = useRef('');
  const frameRef = useRef(0);
  const busyRef = useRef(false);
  const dirtyRef = useRef(true);
  /** Cost of the last completed pass; null until one has run. */
  const costRef = useRef<number | null>(null);

  const recompute = useCallback(() => {
    if (busyRef.current) return;
    busyRef.current = true;
    setPending(true);
    // Yield one frame so the busy state paints before the blocking pass.
    frameRef.current = requestAnimationFrame(() => {
      try {
        const buffers = getEngine().buffers;
        signatureRef.current = graphSignature(buffers);
        dirtyRef.current = false;
        const next = computeGraphMetrics(buffers);
        costRef.current = next.computeMs;
        setMetrics(next);
        setStale(false);
        setError(null);
      } catch (cause) {
        // The busy latch below is what stops the poll from stampeding, so it has
        // to be released on the failing path too. Without this a single throw
        // would leave the panel spinning on a stale readout forever.
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setPending(false);
        busyRef.current = false;
      }
    });
  }, []);

  // The editor drafts copy-on-write, so any edit to a cell or a synapse replaces
  // these arrays. Rewiring an endpoint, disabling a cell and retuning a weight
  // all leave the counts untouched, which the buffer signature cannot see — the
  // readouts would go on describing a network that no longer exists. This effect
  // runs before the ancestor one that reloads the engine, so the flag it raises
  // is always cleared by a pass over buffers at least as new as the flag.
  useEffect(() => {
    dirtyRef.current = true;
  }, [circuit.neurons, circuit.synapses]);

  useEffect(() => {
    if (!open) return;
    // The engine is loaded by an effect in an ancestor, which commits after this
    // one; polling is what makes the panel correct regardless of that ordering,
    // and it also catches loads triggered from anywhere else.
    const poll = () => {
      if (busyRef.current) return;
      const resized = graphSignature(getEngine().buffers) !== signatureRef.current;
      if (!resized && !dirtyRef.current) return;
      if (resized || costRef.current === null || costRef.current <= AUTO_RECOMPUTE_BUDGET_MS) {
        recompute();
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
  }, [open, recompute]);

  const hubs = useMemo(() => {
    if (metrics === null || metrics.hubs.length === 0) return [];
    const engine = getEngine();
    const rows = metrics.hubs.map((hub) => {
      const id = engine.idOf(hub.slot);
      return {
        ...hub,
        id: id === null ? null : (id as NeuronId),
        label: id === null ? `slot ${hub.slot}` : id.slice(0, 8),
      };
    });
    // One pass over the document rather than a lookup per hub, which would be
    // O(hubs · neurons) on a hundred-thousand-cell circuit. The pass stops as
    // soon as the last hub is named, so on a large circuit it usually walks a
    // small prefix rather than the whole list.
    const index = new Map<string, number>();
    rows.forEach((row, at) => {
      if (row.id !== null) index.set(row.id, at);
    });
    let remaining = index.size;
    for (const neuron of circuit.neurons) {
      if (remaining === 0) break;
      const at = index.get(neuron.id);
      if (at === undefined) continue;
      remaining -= 1;
      if (neuron.label.length > 0) rows[at].label = neuron.label;
    }
    return rows;
  }, [metrics, circuit.neurons]);

  if (!open) return null;

  const placement = className ?? 'absolute top-3 bottom-3 left-3 w-[340px]';

  // One header for every state, so the recompute and close controls never move
  // or disappear as the panel goes from measuring to empty to populated.
  const header = (
    <PanelHeader
      title="Connectome"
      subtitle={
        metrics === null
          ? undefined
          : `${grouped(metrics.neurons)} cells · ${grouped(metrics.synapses)} synapses`
      }
      icon={<Network />}
      actions={
        <>
          {error !== null ? (
            <Tooltip content={`The last pass failed: ${error}`}>
              <Badge variant="danger" size="sm" tabIndex={0}>
                failed
              </Badge>
            </Tooltip>
          ) : stale ? (
            <Tooltip content="The circuit changed after this pass, and recomputing it costs more than a frame. Recompute to bring the readouts back in step.">
              <Badge variant="warning" size="sm" tabIndex={0}>
                stale
              </Badge>
            </Tooltip>
          ) : metrics !== null ? (
            <Tooltip content="Time taken by the last pass over the live buffers">
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(metrics.computeMs, 1)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Recompute from the running network">
            <IconButton label="Recompute metrics" size="sm" onClick={recompute} disabled={pending}>
              <RefreshCw className={pending ? 'animate-spin' : undefined} />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close connectome panel" size="sm" onClick={onClose}>
              <X />
            </IconButton>
          ) : null}
        </>
      }
    />
  );

  if (metrics === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        {error === null ? (
          <EmptyState
            icon={<Spinner size={18} />}
            title="Measuring the connectome"
            description="Building the adjacency of the running network."
          />
        ) : (
          <EmptyState
            icon={<TriangleAlert className="text-danger" />}
            title="Could not measure the network"
            description={error}
            action={
              <Button size="sm" icon={<RefreshCw />} onClick={recompute} loading={pending}>
                Try again
              </Button>
            }
          />
        )}
      </Panel>
    );
  }

  if (metrics.neurons === 0) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        <EmptyState
          icon={<Network />}
          title="No network to measure"
          description="Place neurons and wire them together, then this panel reports the graph statistics of the running circuit."
        />
      </Panel>
    );
  }

  const largestShare = metrics.neurons > 0 ? metrics.largestComponent / metrics.neurons : 0;
  const maxReceptorWeight = metrics.receptors.reduce((max, r) => Math.max(max, r.meanWeight), 0);
  const topHubDegree = hubs.length > 0 ? hubs[0].degree : 1;
  const selected = new Set<string>(selection);

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      {header}

      <ScrollArea className="min-h-0 flex-1">
        <PanelSection label="Topology" flush>
          <div className="grid grid-cols-2 gap-x-3 gap-y-2">
            <Stat
              label="Neurons"
              value={grouped(metrics.neurons)}
              hint={
                metrics.disabledNeurons > 0
                  ? `${grouped(metrics.disabledNeurons)} excluded from integration`
                  : 'All cells participate in integration'
              }
            />
            <Stat
              label="Synapses"
              value={grouped(metrics.synapses)}
              hint={
                metrics.disabledSynapses > 0
                  ? `${grouped(metrics.disabledSynapses)} disabled and excluded here`
                  : 'Enabled connections with live endpoints'
              }
            />
            <Stat
              label="Mean degree"
              value={fixed(metrics.meanDegree, 2)}
              hint="Mean in-degree, which equals the mean out-degree in a directed graph"
            />
            <Stat
              label="Max degree"
              value={grouped(metrics.maxDegree)}
              hint="Highest combined in + out degree in the network"
            />
            <Stat
              label="Max in / out"
              value={`${grouped(metrics.maxInDegree)} / ${grouped(metrics.maxOutDegree)}`}
              hint="Busiest single afferent and efferent fan"
            />
            <Stat
              label="Density"
              value={percent(metrics.density, metrics.density < 0.01 ? 3 : 2)}
              hint={`${grouped(metrics.uniqueEdges)} distinct ordered pairs out of ${grouped(
                metrics.neurons * (metrics.neurons - 1),
              )} possible`}
            />
            <Stat
              label="Reciprocity"
              value={percent(metrics.reciprocity)}
              tone="text-accent"
              hint="Share of distinct connections answered by one in the opposite direction"
            />
            <Stat
              label="Clustering"
              value={`${metrics.clusteringExact ? '' : '≈'}${fixed(metrics.clustering, 3)}`}
              tone="text-accent"
              hint={
                metrics.clusteringExact
                  ? 'Mean local clustering coefficient of the undirected graph'
                  : `Estimated over ${grouped(metrics.clusteringSamples)} sampled cells — an exact pass on this degree distribution would block the frame`
              }
            />
            <Stat
              label="Components"
              value={grouped(metrics.components)}
              hint="Weakly connected components, counting isolated cells"
            />
            <Stat
              label="Largest"
              value={`${grouped(metrics.largestComponent)} · ${percent(largestShare, 0)}`}
              hint="Size of the largest weakly connected component"
            />
            <Stat
              label="Isolated"
              value={grouped(metrics.isolated)}
              tone={metrics.isolated > 0 ? 'text-warning' : undefined}
              hint="Cells with no connection in either direction"
            />
            <Stat
              label="Self-loops"
              value={grouped(metrics.selfLoops)}
              hint="Autapses; excluded from density, reciprocity and clustering"
            />
          </div>
        </PanelSection>

        <Separator />

        <PanelSection
          label="Degree distribution"
          aside={
            <SegmentedControl
              size="sm"
              value={scale}
              onChange={setScale}
              options={SCALE_OPTIONS}
              aria-label="Count axis scale"
            />
          }
        >
          <DegreeChart
            title="In-degree"
            values={metrics.inHistogram}
            color="var(--color-accent)"
            log={scale === 'log'}
          />
          <DegreeChart
            title="Out-degree"
            values={metrics.outHistogram}
            color="var(--color-secondary)"
            log={scale === 'log'}
            className="mt-3"
          />
        </PanelSection>

        <Separator />

        <PanelSection
          label="Excitatory / inhibitory"
          aside={
            <div className="flex items-center gap-2 text-[9.5px] text-ink-faint">
              <LegendDot color="var(--color-accent)" label="exc" />
              <LegendDot color="var(--color-secondary)" label="inh" />
            </div>
          }
        >
          <BalanceBar
            label="Cells"
            excitatory={metrics.excitatoryNeurons}
            inhibitory={metrics.inhibitoryNeurons}
            format={grouped}
          />
          <BalanceBar
            label="Synapses"
            excitatory={metrics.excitatorySynapses}
            inhibitory={metrics.inhibitorySynapses}
            format={grouped}
            className="mt-2"
          />
          <BalanceBar
            label="Peak conductance"
            excitatory={metrics.excitatoryConductance}
            inhibitory={metrics.inhibitoryConductance}
            format={(value) => `${fixed(value, 0)} nS`}
            className="mt-2"
          />
        </PanelSection>

        <Separator />

        <PanelSection
          label="Receptors"
          aside={
            <span className="nf-numeric text-[10px] text-ink-faint">
              mean {fixed(metrics.meanWeight, 2)} nS
            </span>
          }
        >
          {metrics.receptors.length === 0 ? (
            <p className="text-[10.5px] text-ink-faint">No synapses to break down.</p>
          ) : (
            metrics.receptors.map((stat) => (
              <div key={stat.receptor} className="flex flex-col gap-1 py-0.5">
                <div className="flex items-baseline gap-1.5 text-[10.5px]">
                  <span
                    aria-hidden
                    className="size-2 shrink-0 translate-y-px rounded-[2px]"
                    style={{ backgroundColor: RECEPTOR_COLORS[stat.receptor] }}
                  />
                  <span className="truncate text-ink">{RECEPTOR_LABELS[stat.receptor]}</span>
                  <span className="nf-numeric ml-auto shrink-0 text-ink-faint">
                    n={grouped(stat.count)}
                  </span>
                  <span className="nf-numeric w-[52px] shrink-0 text-right text-ink">
                    {fixed(stat.meanWeight, 2)} nS
                  </span>
                </div>
                <div className="h-[3px] w-full overflow-hidden rounded-full bg-white/[0.05]">
                  <span
                    className="block h-full rounded-full"
                    style={{
                      width: `${
                        maxReceptorWeight > 0 ? (stat.meanWeight / maxReceptorWeight) * 100 : 0
                      }%`,
                      backgroundColor: RECEPTOR_COLORS[stat.receptor],
                    }}
                  />
                </div>
              </div>
            ))
          )}
        </PanelSection>

        <Separator />

        <PanelSection
          label="Hubs"
          aside={<span className="text-[9.5px] text-ink-faint">by total degree</span>}
        >
          {hubs.length === 0 ? (
            <p className="text-[10.5px] text-ink-faint">Nothing is connected yet.</p>
          ) : (
            <ul className="flex flex-col">
              {hubs.map((hub) => {
                const colour = identityColorHex(hub.seed);
                const active = hub.id !== null && selected.has(hub.id);
                return (
                  <li key={hub.slot}>
                    <button
                      type="button"
                      disabled={hub.id === null}
                      onClick={() => {
                        if (hub.id !== null) select([hub.id]);
                      }}
                      aria-pressed={active}
                      className={cn(
                        'relative flex w-full items-center gap-1.5 overflow-hidden rounded-control px-1.5 py-1 text-left text-[10.5px] transition-colors',
                        'hover:bg-panel-raised focus-visible:bg-panel-raised disabled:pointer-events-none disabled:opacity-50',
                        active && 'bg-white/[0.07]',
                      )}
                    >
                      {/* The swatch is the same hue the cell is drawn in, so a row
                          here and a glyph in the scene are the same object. */}
                      <span
                        aria-hidden
                        className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
                        style={{ backgroundColor: colour }}
                      />
                      <span className="truncate text-ink">{hub.label}</span>
                      {hub.inhibitory ? (
                        <span className="shrink-0 text-[8.5px] font-semibold tracking-[0.08em] text-secondary">
                          INH
                        </span>
                      ) : null}
                      <span className="nf-numeric ml-auto shrink-0 text-ink-faint">
                        {grouped(hub.inDegree)}
                        <span className="text-ink-faint/60">/</span>
                        {grouped(hub.outDegree)}
                      </span>
                      <span className="nf-numeric w-9 shrink-0 text-right text-ink">
                        {grouped(hub.degree)}
                      </span>
                      <span
                        aria-hidden
                        className="absolute bottom-0 left-0 h-px opacity-70"
                        style={{
                          width: `${(hub.degree / topHubDegree) * 100}%`,
                          backgroundColor: colour,
                        }}
                      />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </PanelSection>
      </ScrollArea>
    </Panel>
  );
}

/* ------------------------------------------------------------------ pieces -- */

function percent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

interface StatProps {
  label: string;
  value: string;
  hint: string;
  tone?: string;
}

function Stat({ label, value, hint, tone }: StatProps) {
  return (
    <Tooltip content={hint} side="top">
      <div
        role="group"
        tabIndex={0}
        aria-label={`${label}: ${value}`}
        className="flex flex-col gap-0.5 rounded-sm focus-visible:outline-1"
      >
        <span className="truncate text-[9.5px] font-medium tracking-[0.07em] text-ink-faint uppercase">
          {label}
        </span>
        <span className={cn('nf-numeric text-[12px] leading-none text-ink', tone)}>{value}</span>
      </div>
    </Tooltip>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span aria-hidden className="size-1.5 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

interface BalanceBarProps {
  label: string;
  excitatory: number;
  inhibitory: number;
  format: (value: number) => string;
  className?: string;
}

function BalanceBar({ label, excitatory, inhibitory, format, className }: BalanceBarProps) {
  const total = excitatory + inhibitory;
  const share = total > 0 ? excitatory / total : 0;
  return (
    <div className={cn('flex flex-col gap-1', className)}>
      <div className="flex items-baseline justify-between text-[10px]">
        <span className="text-ink-faint">{label}</span>
        <span className="nf-numeric text-ink-faint">
          <span className="text-accent">{format(excitatory)}</span>
          <span className="px-1">·</span>
          <span className="text-secondary">{format(inhibitory)}</span>
          <span className="pl-1.5 text-ink-faint">{percent(share, 0)} E</span>
        </span>
      </div>
      {/* Flex growth rather than percentage widths, so the 2px separator between
          the two fills never pushes the track into overflow. */}
      <div className="flex h-1.5 w-full gap-[2px] rounded-full bg-white/[0.05]">
        <span
          className="min-w-0 rounded-full bg-accent"
          style={{ flexGrow: excitatory, flexBasis: 0 }}
        />
        <span
          className="min-w-0 rounded-full bg-secondary"
          style={{ flexGrow: inhibitory, flexBasis: 0 }}
        />
      </div>
    </div>
  );
}

interface Bins {
  counts: Float64Array;
  binWidth: number;
  peak: number;
  maxDegree: number;
}

/**
 * Collapse a per-degree histogram into at most HIST_BINS columns. Degree
 * distributions in a connectome run to thousands of distinct degrees, which is
 * more columns than a 316-unit-wide chart has pixels.
 */
function binHistogram(values: Uint32Array): Bins {
  const length = Math.max(1, values.length);
  const binWidth = Math.max(1, Math.ceil(length / HIST_BINS));
  const binCount = Math.max(1, Math.ceil(length / binWidth));
  const counts = new Float64Array(binCount);
  let peak = 0;
  for (let degree = 0; degree < values.length; degree += 1) {
    const bin = (degree / binWidth) | 0;
    counts[bin] += values[degree];
  }
  for (let i = 0; i < binCount; i += 1) if (counts[i] > peak) peak = counts[i];
  return { counts, binWidth, peak, maxDegree: values.length - 1 };
}

interface DegreeChartProps {
  title: string;
  values: Uint32Array;
  color: string;
  log: boolean;
  className?: string;
}

function DegreeChart({ title, values, color, log, className }: DegreeChartProps) {
  const bins = useMemo(() => binHistogram(values), [values]);
  const { counts, binWidth, peak, maxDegree } = bins;
  const binCount = counts.length;

  const slot = HIST_WIDTH / binCount;
  const gap = slot > 4 ? 2 : slot > 2 ? 1 : 0;
  const barWidth = Math.max(0.75, slot - gap);
  const plotHeight = HIST_BASELINE - 2;
  const logPeak = Math.log1p(peak);

  return (
    <div className={className}>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[10px] text-ink">{title}</span>
        <span className="nf-numeric text-[9.5px] text-ink-faint">
          peak {grouped(peak)} cells
        </span>
      </div>
      {/* preserveAspectRatio="none" keeps the vertical scale at 1:1 while the
          columns stretch to the panel width, which is what a histogram wants. */}
      <svg
        viewBox={`0 0 ${HIST_WIDTH} ${HIST_HEIGHT}`}
        width="100%"
        height={HIST_HEIGHT}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${title} distribution, ${grouped(maxDegree)} maximum, peak ${grouped(
          peak,
        )} cells`}
        className="block"
      >
        <rect
          x={0}
          y={HIST_BASELINE}
          width={HIST_WIDTH}
          height={0.75}
          fill="var(--color-ink-faint)"
          opacity={0.45}
        />
        {Array.from({ length: binCount }, (_, i) => {
          const value = counts[i];
          if (value <= 0) return null;
          const norm =
            peak <= 0 ? 0 : log ? Math.log1p(value) / (logPeak || 1) : value / peak;
          const height = Math.max(1, norm * plotHeight);
          const from = i * binWidth;
          const to = Math.min(maxDegree, from + binWidth - 1);
          return (
            <rect
              key={i}
              x={i * slot + gap / 2}
              y={HIST_BASELINE - height}
              width={barWidth}
              height={height}
              fill={color}
              opacity={0.85}
              className="transition-opacity hover:opacity-100"
            >
              <title>
                {from === to ? `degree ${from}` : `degree ${from}–${to}`}: {grouped(value)} cells
              </title>
            </rect>
          );
        })}
      </svg>
      <div className="mt-0.5 flex items-baseline justify-between text-[9px] text-ink-faint">
        <span className="nf-numeric">0</span>
        <span>{log ? 'log count' : 'count'}</span>
        <span className="nf-numeric">{grouped(maxDegree)}</span>
      </div>
    </div>
  );
}
