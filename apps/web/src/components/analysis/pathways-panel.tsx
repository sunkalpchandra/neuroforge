'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowRightLeft,
  CircleSlash,
  Crosshair,
  Focus,
  RefreshCw,
  Search,
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
  PanelSection,
  ScrollArea,
  SegmentedControl,
  Separator,
  Spinner,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import type { SegmentedOption } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { RECEPTOR_COLORS, asNeuronId, identityColorHex } from '@neuroforge/shared';
import type { Neuron, NeuronId, ReceptorKind } from '@neuroforge/shared';

import { boundsOf, getEngine, requestCameraFrame } from '@/lib/runtime';
import { compact, fixed, grouped } from '@/lib/format';
import { graphSignature } from '@/lib/graph-metrics';
import {
  allPathsWithinHops,
  buildPathGraph,
  describeRoute,
  kShortestPaths,
  reachableWithin,
  reachedSlots,
  shortestPath,
} from '@/lib/pathfinding';
import type { PathCensus, PathGraph, ReachResult, Route } from '@/lib/pathfinding';

/** Cadence at which the panel checks whether the running network moved under it. */
const POLL_MS = 500;

/** Rows offered by the endpoint search. */
const MAX_SUGGESTIONS = 6;

/** Identity swatches drawn in an alternate-route strip before it elides. */
const MAX_STRIP = 8;

const PLACEMENT = 'absolute top-3 bottom-3 left-3 w-[348px]';

type RouteCount = '3' | '5' | '8';
type HopHorizon = '2' | '3' | '4' | '5';
type ReachDepth = '1' | '2' | '3' | '4';
type Rank = 'hops' | 'weight' | 'delay';

const ROUTE_COUNT_OPTIONS: readonly SegmentedOption<RouteCount>[] = [
  { value: '3', label: '3', title: 'Find three loopless routes' },
  { value: '5', label: '5', title: 'Find five loopless routes' },
  { value: '8', label: '8', title: 'Find eight loopless routes' },
];

const HOP_OPTIONS: readonly SegmentedOption<HopHorizon>[] = [
  { value: '2', label: '2', title: 'Count paths of up to two hops' },
  { value: '3', label: '3', title: 'Count paths of up to three hops' },
  { value: '4', label: '4', title: 'Count paths of up to four hops' },
  { value: '5', label: '5', title: 'Count paths of up to five hops' },
];

const REACH_OPTIONS: readonly SegmentedOption<ReachDepth>[] = [
  { value: '1', label: '1', title: 'Cells one synapse downstream' },
  { value: '2', label: '2', title: 'Cells up to two synapses downstream' },
  { value: '3', label: '3', title: 'Cells up to three synapses downstream' },
  { value: '4', label: '4', title: 'Cells up to four synapses downstream' },
];

const RANK_OPTIONS: readonly SegmentedOption<Rank>[] = [
  { value: 'hops', label: 'hops', title: 'Fewest synaptic steps first' },
  { value: 'weight', label: 'Σln w', title: 'Largest product of synaptic weights first' },
  { value: 'delay', label: 'ms', title: 'Shortest summed conduction delay first' },
];

/** Row-width receptor names; the full labels do not fit a hop connector. */
const RECEPTOR_SHORT: Record<ReceptorKind, string> = {
  ampa: 'AMPA',
  nmda: 'NMDA',
  gabaa: 'GABA-A',
  gabab: 'GABA-B',
  gap: 'GAP',
};

type Status =
  | 'empty'
  | 'unset'
  | 'source-gone'
  | 'target-gone'
  | 'same'
  | 'unreachable'
  | 'ok';

interface Analysis {
  signature: string;
  computeMs: number;
  neurons: number;
  edges: number;
  /** Enabled synapses with live endpoints that the edges were folded from. */
  synapses: number;
  disabledSynapses: number;
  /** Synapses skipped because an endpoint no longer resolves. */
  danglingSynapses: number;
  buildMs: number;
  status: Status;
  source: number;
  target: number;
  routes: readonly Route[];
  routesTruncated: boolean;
  /** Hops in the opposite direction when this one is unreachable, else null. */
  reverseHops: number | null;
  census: PathCensus | null;
  reach: ReachResult | null;
  /** Morphology seed per slot for every cell this analysis names. */
  seeds: ReadonlyMap<number, number>;
  /** Document id per slot, captured with the buffers the routes came from. */
  ids: ReadonlyMap<number, string>;
}

export interface PathwaysPanelProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Replaces the default docking entirely. */
  className?: string;
}

/**
 * Signalling routes between two cells.
 *
 * The question a connectomics user asks of a wiring diagram is not how dense it
 * is but how one cell reaches another, so this panel answers exactly that: the
 * ordered chain of synapses, what each one weighs and how long it takes, the
 * alternate routes that carry the same signal, how many paths exist at all, and
 * how far the source's influence spreads.
 *
 * Every search runs over the live buffers and is budgeted — path enumeration in
 * a recurrent network is unbounded work — and every route can be pushed into the
 * selection so the 3D scene lights the whole chain in the cells' own colours.
 */
export function PathwaysPanel({ open = true, onClose, className }: PathwaysPanelProps) {
  const circuit = useEditor((s) => s.circuit);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [sourceId, setSourceId] = useState<NeuronId | null>(null);
  const [targetId, setTargetId] = useState<NeuronId | null>(null);
  const [routeCount, setRouteCount] = useState<RouteCount>('5');
  const [censusHops, setCensusHops] = useState<HopHorizon>('3');
  const [reachDepth, setReachDepth] = useState<ReachDepth>('2');
  const [rank, setRank] = useState<Rank>('hops');
  const [pickedKey, setPickedKey] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);

  const signatureRef = useRef('');
  const dirtyRef = useRef(true);
  /** The graph is rebuilt per revision, not per control change. */
  const graphRef = useRef<{ revision: number; graph: PathGraph } | null>(null);

  // Copy-on-write means any edit to a cell or a synapse replaces these arrays.
  // Rewiring an endpoint or retuning a weight leaves the counts alone, which the
  // buffer signature cannot see, so the edit is flagged here and picked up by
  // the poll below — which runs after the ancestor effect that reloads the
  // engine, and therefore always reads buffers at least as new as the flag.
  useEffect(() => {
    dirtyRef.current = true;
  }, [circuit.neurons, circuit.synapses]);

  useEffect(() => {
    if (!open) return;
    const poll = () => {
      const signature = graphSignature(getEngine().buffers);
      if (!dirtyRef.current && signature === signatureRef.current) return;
      dirtyRef.current = false;
      setRevision((value) => value + 1);
    };
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setPending(true);

    // One frame of delay so the busy state paints before the blocking pass.
    const frame = requestAnimationFrame(() => {
      if (cancelled) return;
      try {
        const engine = getEngine();
        const buffers = engine.buffers;
        const signature = graphSignature(buffers);
        signatureRef.current = signature;

        const cached = graphRef.current;
        const graph =
          cached !== null && cached.revision === revision && cached.graph.signature === signature
            ? cached.graph
            : buildPathGraph(buffers);
        graphRef.current = { revision, graph };

        const started = performance.now();
        const source = sourceId === null ? -1 : engine.slotOf(sourceId);
        const target = targetId === null ? -1 : engine.slotOf(targetId);

        const seeds = new Map<number, number>();
        const ids = new Map<number, string>();
        const remember = (slot: number): void => {
          if (slot < 0 || slot >= graph.n || seeds.has(slot)) return;
          seeds.set(slot, buffers.neurons.seed[slot]);
          const id = engine.idOf(slot);
          if (id !== null) ids.set(slot, id);
        };
        remember(source);
        remember(target);

        let status: Status;
        if (graph.n === 0) status = 'empty';
        else if (sourceId === null || targetId === null) status = 'unset';
        else if (source < 0) status = 'source-gone';
        else if (target < 0) status = 'target-gone';
        else status = source === target ? 'same' : 'ok';

        const endpointsLive = source >= 0 && target >= 0;
        const census = endpointsLive
          ? allPathsWithinHops(graph, source, target, Number(censusHops))
          : null;
        const reach =
          source >= 0 ? reachableWithin(graph, source, Number(reachDepth)) : null;

        const routes: Route[] = [];
        let routesTruncated = false;
        let reverseHops: number | null = null;

        if (status === 'ok') {
          const found = kShortestPaths(graph, source, target, Number(routeCount));
          routesTruncated = found.truncated;
          for (const path of found.paths) {
            const route = describeRoute(graph, path);
            if (route === null) continue;
            routes.push(route);
            for (const slot of route.nodes) remember(slot);
          }
          if (routes.length === 0) {
            status = 'unreachable';
            // Whether the signal runs the other way is the first thing a user
            // wants to know about a dead end, and it is one more BFS.
            const reverse = shortestPath(graph, target, source);
            reverseHops = reverse === null ? null : reverse.length - 1;
          }
        }

        if (cancelled) return;
        setAnalysis({
          signature,
          computeMs: performance.now() - started,
          neurons: graph.n,
          edges: graph.edges,
          synapses: graph.synapses,
          disabledSynapses: graph.disabled,
          danglingSynapses: graph.dangling,
          buildMs: graph.buildMs,
          status,
          source,
          target,
          routes,
          routesTruncated,
          reverseHops,
          census,
          reach,
          seeds,
          ids,
        });
        setError(null);
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        if (!cancelled) setPending(false);
      }
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      setPending(false);
    };
  }, [open, revision, sourceId, targetId, routeCount, censusHops, reachDepth]);

  const sourceNeuron = useNeuron(circuit.neurons, sourceId);
  const targetNeuron = useNeuron(circuit.neurons, targetId);

  /**
   * Labels for every cell the analysis names, resolved in a single pass over the
   * document. A lookup per slot would be O(routes · neurons) on a large circuit.
   */
  const labels = useMemo(() => {
    const map = new Map<number, string>();
    if (analysis === null) return map;
    const slotOfId = new Map<string, number>();
    for (const [slot, id] of analysis.ids) {
      slotOfId.set(id, slot);
      map.set(slot, id.slice(0, 8));
    }
    let remaining = slotOfId.size;
    for (const neuron of circuit.neurons) {
      if (remaining === 0) break;
      const slot = slotOfId.get(neuron.id);
      if (slot === undefined) continue;
      remaining -= 1;
      if (neuron.label.length > 0) map.set(slot, neuron.label);
    }
    return map;
  }, [analysis, circuit.neurons]);

  const labelOf = useCallback(
    (slot: number): string => labels.get(slot) ?? `slot ${slot}`,
    [labels],
  );

  const colorOf = useCallback(
    (slot: number): string => identityColorHex(analysis?.seeds.get(slot) ?? 0),
    [analysis],
  );

  const ranked = useMemo(() => {
    const routes = analysis === null ? [] : [...analysis.routes];
    routes.sort((a, b) => {
      if (rank === 'weight') {
        // Compared rather than subtracted: a route through a zero-conductance
        // synapse has a log weight of -Infinity, and the difference of two of
        // those is NaN, which would scramble the whole ordering.
        if (a.logWeight !== b.logWeight) return b.logWeight > a.logWeight ? 1 : -1;
        return a.length - b.length;
      }
      if (rank === 'delay') {
        if (a.delay !== b.delay) return a.delay - b.delay;
        return a.length - b.length;
      }
      if (a.length !== b.length) return a.length - b.length;
      return a.delay - b.delay;
    });
    return routes;
  }, [analysis, rank]);

  const picked = useMemo(() => {
    if (ranked.length === 0) return null;
    if (pickedKey !== null) {
      const match = ranked.find((route) => routeKey(route) === pickedKey);
      if (match !== undefined) return match;
    }
    return ranked[0];
  }, [ranked, pickedKey]);

  const idsOfRoute = useCallback(
    (route: Route): NeuronId[] => {
      const out: NeuronId[] = [];
      if (analysis === null) return out;
      for (const slot of route.nodes) {
        const id = analysis.ids.get(slot);
        if (id !== undefined) out.push(asNeuronId(id));
      }
      return out;
    },
    [analysis],
  );

  const selectRoute = useCallback(
    (route: Route) => {
      setPickedKey(routeKey(route));
      const ids = idsOfRoute(route);
      if (ids.length > 0) select(ids);
    },
    [idsOfRoute, select],
  );

  const frameRoute = useCallback(
    (route: Route) => {
      const ids = idsOfRoute(route);
      if (ids.length === 0) return;
      const bounds = boundsOf(ids);
      if (bounds !== null) requestCameraFrame(bounds);
    },
    [idsOfRoute],
  );

  const selectReachable = useCallback(() => {
    const reach = analysis === null ? null : analysis.reach;
    if (reach === null) return;
    const engine = getEngine();
    const ids: NeuronId[] = [];
    for (const slot of reachedSlots(reach, Number(reachDepth))) {
      const id = engine.idOf(slot);
      if (id !== null) ids.push(asNeuronId(id));
    }
    if (ids.length > 0) select(ids);
  }, [analysis, reachDepth, select]);

  const swap = useCallback(() => {
    setPickedKey(null);
    setSourceId(targetId);
    setTargetId(sourceId);
  }, [sourceId, targetId]);

  const pickSource = useCallback((id: NeuronId | null) => {
    setPickedKey(null);
    setSourceId(id);
  }, []);

  const pickTarget = useCallback((id: NeuronId | null) => {
    setPickedKey(null);
    setTargetId(id);
  }, []);

  const refresh = useCallback(() => {
    graphRef.current = null;
    setRevision((value) => value + 1);
  }, []);

  if (!open) return null;

  const truncated =
    analysis !== null &&
    (analysis.routesTruncated ||
      (analysis.census?.truncated ?? false) ||
      (analysis.reach?.truncated ?? false));

  const header = (
    <PanelHeader
      title="Pathways"
      subtitle={
        sourceNeuron !== null && targetNeuron !== null
          ? `${nameOf(sourceNeuron)} → ${nameOf(targetNeuron)}`
          : 'Signalling routes between two cells'
      }
      icon={<Waypoints size={14} />}
      actions={
        <>
          {error !== null ? (
            <Tooltip content={`The last search failed: ${error}`}>
              <Badge variant="danger" size="sm" tabIndex={0}>
                failed
              </Badge>
            </Tooltip>
          ) : truncated ? (
            <Tooltip content="A search hit its visit budget and stopped early. The numbers below are exact for the part of the network that was walked; shorten the hop horizon for a complete answer.">
              <Badge variant="warning" size="sm" tabIndex={0}>
                budget
              </Badge>
            </Tooltip>
          ) : analysis !== null ? (
            <Tooltip content="Time taken by the last search over the running network">
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(analysis.computeMs, 1)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Search again against the running network">
            <IconButton label="Recompute pathways" size="sm" onClick={refresh} disabled={pending}>
              <RefreshCw className={pending ? 'animate-spin' : undefined} />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close pathways panel" size="sm" onClick={onClose}>
              <X />
            </IconButton>
          ) : null}
        </>
      }
    />
  );

  if (error !== null && analysis === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', className ?? PLACEMENT)}>
        {header}
        <EmptyState
          icon={<TriangleAlert className="text-danger" />}
          title="Could not search the network"
          description={error}
          action={
            <Button size="sm" icon={<RefreshCw />} onClick={refresh} loading={pending}>
              Try again
            </Button>
          }
        />
      </Panel>
    );
  }

  if (analysis === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', className ?? PLACEMENT)}>
        {header}
        <EmptyState
          icon={<Spinner size={18} />}
          title="Reading the wiring"
          description="Building the adjacency of the running network."
        />
      </Panel>
    );
  }

  if (analysis.status === 'empty') {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', className ?? PLACEMENT)}>
        {header}
        <EmptyState
          icon={<Waypoints />}
          title="No network to trace"
          description="Place cells and wire them together, then this panel finds the routes a signal can take between any two of them."
        />
      </Panel>
    );
  }

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', className ?? PLACEMENT)}>
      {header}

      <ScrollArea className="min-h-0 flex-1">
        <PanelSection
          label="Endpoints"
          flush
          aside={
            <Tooltip content="Swap the source and the target">
              <IconButton
                label="Swap source and target"
                size="sm"
                className="size-5"
                disabled={sourceId === null && targetId === null}
                onClick={swap}
              >
                <ArrowRightLeft size={11} />
              </IconButton>
            </Tooltip>
          }
        >
          <EndpointField
            role="Source"
            id={sourceId}
            neuron={sourceNeuron}
            missing={sourceId !== null && analysis.source < 0 && sourceNeuron === null}
            neurons={circuit.neurons}
            selection={selection}
            onPick={pickSource}
          />
          <EndpointField
            role="Target"
            id={targetId}
            neuron={targetNeuron}
            missing={targetId !== null && analysis.target < 0 && targetNeuron === null}
            neurons={circuit.neurons}
            selection={selection}
            onPick={pickTarget}
            className="mt-1.5"
          />
        </PanelSection>

        <Separator />

        <PanelSection
          label="Route"
          aside={
            analysis.status === 'ok' ? (
              <SegmentedControl<Rank>
                size="sm"
                value={rank}
                onChange={setRank}
                options={RANK_OPTIONS}
                aria-label="Rank routes by"
              />
            ) : undefined
          }
        >
          {analysis.status !== 'ok' || picked === null ? (
            <Notice
              status={analysis.status}
              reverseHops={analysis.reverseHops}
              sourceName={sourceNeuron === null ? null : nameOf(sourceNeuron)}
              targetName={targetNeuron === null ? null : nameOf(targetNeuron)}
              inDocument={
                analysis.status === 'source-gone' ? sourceNeuron !== null : targetNeuron !== null
              }
              onSwap={swap}
              onRecompute={refresh}
              onClearSource={() => pickSource(null)}
              onClearTarget={() => pickTarget(null)}
            />
          ) : (
            <>
              <div className="grid grid-cols-4 gap-x-2">
                <Stat
                  label="Hops"
                  value={grouped(picked.length)}
                  hint="Synaptic steps between the two cells on this route"
                />
                <Stat
                  label="Delay"
                  value={fixed(picked.delay, 2)}
                  unit="ms"
                  hint="Summed axonal conduction delay along the route"
                />
                <Stat
                  label="Σ ln w"
                  value={picked.logWeight === -Infinity ? '−∞' : fixed(picked.logWeight, 2)}
                  hint="Product of the synaptic weights, in log units. A product of dozens of sub-unit conductances underflows to zero in float64, so it is only ever reported as its logarithm."
                  tone="text-accent"
                />
                <Stat
                  label="Min w"
                  value={fixed(picked.bottleneck, 2)}
                  unit="nS"
                  hint="Weakest conductance on the route — its bottleneck"
                />
              </div>

              <div className="mt-2 flex items-center gap-1.5">
                <Button
                  size="sm"
                  variant="primary"
                  className="h-6 flex-1 px-2 text-[10.5px]"
                  onClick={() => selectRoute(picked)}
                >
                  Select all {picked.nodes.length} cells
                </Button>
                <Tooltip content="Frame the route in the viewport">
                  <IconButton
                    label="Frame the route"
                    size="sm"
                    className="size-6"
                    onClick={() => frameRoute(picked)}
                  >
                    <Focus size={12} />
                  </IconButton>
                </Tooltip>
              </div>

              <ol className="mt-2 flex flex-col">
                {picked.nodes.map((slot, index) => {
                  const hop = index < picked.hops.length ? picked.hops[index] : null;
                  const id = analysis.ids.get(slot);
                  return (
                    <li key={`${slot}-${index}`} className="flex flex-col">
                      <button
                        type="button"
                        disabled={id === undefined}
                        onClick={() => {
                          if (id !== undefined) select([asNeuronId(id)]);
                        }}
                        className={cn(
                          'flex w-full items-center gap-1.5 rounded-control py-[3px] pr-1 text-left',
                          'transition-colors hover:bg-white/[0.05] disabled:pointer-events-none',
                        )}
                      >
                        <span className="nf-numeric w-3 shrink-0 text-right text-[9px] text-ink-faint">
                          {index}
                        </span>
                        <span className="flex w-2.5 shrink-0 justify-center">
                          <span
                            aria-hidden
                            className="size-2 rounded-full ring-1 ring-white/20"
                            style={{
                              backgroundColor: colorOf(slot),
                              boxShadow: `0 0 6px ${colorOf(slot)}66`,
                            }}
                          />
                        </span>
                        <span className="min-w-0 flex-1 truncate text-[11px] text-ink">
                          {labelOf(slot)}
                        </span>
                        {index === 0 ? <Tag>src</Tag> : null}
                        {index === picked.nodes.length - 1 ? <Tag>dst</Tag> : null}
                      </button>
                      {hop !== null ? (
                        <span className="ml-[22px] flex items-center gap-1.5 border-l border-hairline-strong py-[3px] pl-2 text-[9.5px] text-ink-faint">
                          <span
                            aria-hidden
                            className="size-1.5 shrink-0 rounded-[1px]"
                            style={{ backgroundColor: RECEPTOR_COLORS[hop.receptor] }}
                          />
                          <span className="nf-numeric text-ink-muted">
                            {fixed(hop.weight, 2)} nS
                          </span>
                          <span aria-hidden>·</span>
                          <span className="nf-numeric text-ink-muted">
                            {fixed(hop.delay, 2)} ms
                          </span>
                          <span aria-hidden>·</span>
                          <span>{RECEPTOR_SHORT[hop.receptor]}</span>
                          {hop.parallel > 1 ? (
                            <Tooltip
                              content={`${hop.parallel} synapses run in parallel between these cells; their peak conductances are summed and the delay and receptor are those of the strongest.`}
                              side="top"
                            >
                              <span className="nf-numeric cursor-help text-ink-faint" tabIndex={0}>
                                ×{hop.parallel}
                              </span>
                            </Tooltip>
                          ) : null}
                        </span>
                      ) : null}
                    </li>
                  );
                })}
              </ol>
            </>
          )}
        </PanelSection>

        {analysis.status === 'ok' ? (
          <>
            <Separator />
            <PanelSection
              label="Alternate routes"
              aside={
                <SegmentedControl<RouteCount>
                  size="sm"
                  value={routeCount}
                  onChange={setRouteCount}
                  options={ROUTE_COUNT_OPTIONS}
                  aria-label="Routes to find"
                />
              }
            >
              <ul className="flex flex-col">
                {ranked.map((route, index) => {
                  const key = routeKey(route);
                  const active = picked !== null && routeKey(picked) === key;
                  return (
                    <li key={key}>
                      <button
                        type="button"
                        aria-pressed={active}
                        onClick={() => selectRoute(route)}
                        className={cn(
                          'flex w-full items-center gap-1.5 rounded-control px-1 py-1 text-left',
                          'transition-colors hover:bg-white/[0.05]',
                          active && 'bg-accent/12',
                        )}
                      >
                        <span className="nf-numeric w-3 shrink-0 text-[9px] text-ink-faint">
                          {index + 1}
                        </span>
                        <span className="flex min-w-0 flex-1 items-center gap-[2px]">
                          {route.nodes.slice(0, MAX_STRIP).map((slot, at) => (
                            <span
                              key={`${slot}-${at}`}
                              aria-hidden
                              className="h-2.5 w-[7px] shrink-0 rounded-[1px]"
                              style={{ backgroundColor: colorOf(slot) }}
                            />
                          ))}
                          {route.nodes.length > MAX_STRIP ? (
                            <span className="nf-numeric pl-0.5 text-[9px] text-ink-faint">
                              +{route.nodes.length - MAX_STRIP}
                            </span>
                          ) : null}
                        </span>
                        <span className="nf-numeric w-7 shrink-0 text-right text-[10px] text-ink">
                          {route.length}h
                        </span>
                        <span className="nf-numeric w-12 shrink-0 text-right text-[10px] text-ink-muted">
                          {fixed(route.delay, 1)}
                        </span>
                        <span className="nf-numeric w-11 shrink-0 text-right text-[10px] text-accent">
                          {route.logWeight === -Infinity ? '−∞' : fixed(route.logWeight, 1)}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
              <div className="mt-1 flex items-baseline justify-between text-[9px] text-ink-faint">
                <span>
                  {analysis.routesTruncated
                    ? 'Budget reached — more routes may exist'
                    : `${grouped(ranked.length)} loopless route${ranked.length === 1 ? '' : 's'} found`}
                </span>
                <span className="nf-numeric">hops · ms · Σln w</span>
              </div>
            </PanelSection>
          </>
        ) : null}

        {analysis.census !== null ? (
          <>
            <Separator />
            <PanelSection
              label={analysis.status === 'same' ? 'Loop spectrum' : 'Path spectrum'}
              aside={
                <SegmentedControl<HopHorizon>
                  size="sm"
                  value={censusHops}
                  onChange={setCensusHops}
                  options={HOP_OPTIONS}
                  aria-label="Hop horizon"
                />
              }
            >
              <Bars
                bins={Array.from({ length: analysis.census.maxHops }, (_, i) => {
                  const hops = i + 1;
                  const count = analysis.census === null ? 0 : analysis.census.byLength[hops];
                  return {
                    key: String(hops),
                    label: String(hops),
                    value: count,
                    title: `${grouped(count)} path${count === 1 ? '' : 's'} of ${hops} hop${
                      hops === 1 ? '' : 's'
                    }`,
                  };
                })}
                color="var(--color-accent)"
                caption={
                  analysis.status === 'same'
                    ? 'hops · loops by length, log scale'
                    : 'hops · paths by length, log scale'
                }
                total={`${compact(analysis.census.total)} total`}
              />
              {analysis.census.total === 0 && !analysis.census.truncated ? (
                <p className="mt-1 text-[9.5px] leading-snug text-ink-faint">
                  {analysis.status === 'same'
                    ? `Nothing this cell drives comes back to it within ${analysis.census.maxHops} hops.`
                    : `No path at all within ${analysis.census.maxHops} hops. Widen the horizon to look further.`}
                </p>
              ) : null}
              {analysis.census.truncated ? (
                <p className="mt-1 text-[9.5px] leading-snug text-warning">
                  Enumeration stopped at {grouped(analysis.census.visited)} visits. Counting the
                  simple paths of a recurrent network has no polynomial bound, so the totals
                  above are a floor, not the whole census.
                </p>
              ) : null}
            </PanelSection>
          </>
        ) : null}

        {analysis.reach !== null ? (
          <>
            <Separator />
            <PanelSection
              label="Sphere of influence"
              aside={
                <SegmentedControl<ReachDepth>
                  size="sm"
                  value={reachDepth}
                  onChange={setReachDepth}
                  options={REACH_OPTIONS}
                  aria-label="Reachable depth"
                />
              }
            >
              <Bars
                bins={Array.from({ length: analysis.reach.hops }, (_, i) => {
                  const depth = i + 1;
                  const count = analysis.reach === null ? 0 : analysis.reach.sizes[depth];
                  const total = analysis.reach === null ? 0 : analysis.reach.cumulative[depth];
                  return {
                    key: String(depth),
                    label: String(depth),
                    value: count,
                    title: `${grouped(count)} cell${count === 1 ? '' : 's'} first reached at ${depth} hop${
                      depth === 1 ? '' : 's'
                    }; ${grouped(total)} within`,
                  };
                })}
                color="var(--color-secondary)"
                caption="hops · cells first reached, log scale"
                total={`${compact(
                  analysis.reach.cumulative[analysis.reach.hops] - 1,
                )} downstream`}
              />
              <div className="mt-1.5 flex items-center gap-1.5">
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-6 flex-1 px-2 text-[10.5px]"
                  disabled={analysis.reach.total <= 1}
                  onClick={selectReachable}
                >
                  Select the {grouped(analysis.reach.total)} reached
                </Button>
                <span className="nf-numeric shrink-0 text-[9.5px] text-ink-faint">
                  {percent(
                    analysis.neurons > 0 ? analysis.reach.total / analysis.neurons : 0,
                    1,
                  )}{' '}
                  of net
                </span>
              </div>
            </PanelSection>
          </>
        ) : null}

        <Separator />

        <PanelSection
          label="Graph"
          aside={
            <span className="nf-numeric text-[9px] text-ink-faint">
              built in {fixed(analysis.buildMs, 1)} ms
            </span>
          }
        >
          <p className="text-[9.5px] leading-snug text-ink-faint">
            Searched {grouped(analysis.neurons)} cells and {grouped(analysis.edges)} connected pairs,
            folded from {grouped(analysis.synapses)} live synapses — parallel synapses between the
            same two cells count once, with their conductances summed.
            {analysis.disabledSynapses > 0
              ? ` ${grouped(analysis.disabledSynapses)} disabled synapses carry no signal and were left out.`
              : ''}
            {analysis.danglingSynapses > 0
              ? ` ${grouped(analysis.danglingSynapses)} synapses point at a cell the engine no longer holds and were skipped.`
              : ''}
          </p>
        </PanelSection>
      </ScrollArea>
    </Panel>
  );
}

/* ------------------------------------------------------------------ pieces -- */

function routeKey(route: Route): string {
  return route.nodes.join('-');
}

function nameOf(neuron: Neuron): string {
  return neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8);
}

function percent(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/** Resolve an id against the document without a lookup per render. */
function useNeuron(neurons: readonly Neuron[], id: NeuronId | null): Neuron | null {
  return useMemo(() => {
    if (id === null) return null;
    return neurons.find((neuron) => neuron.id === id) ?? null;
  }, [neurons, id]);
}

function Tag({ children }: { children: string }) {
  return (
    <span className="shrink-0 rounded-[2px] bg-white/[0.07] px-1 text-[8.5px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
      {children}
    </span>
  );
}

interface StatProps {
  label: string;
  value: string;
  unit?: string;
  hint: string;
  tone?: string;
}

function Stat({ label, value, unit, hint, tone }: StatProps) {
  return (
    <Tooltip content={hint} side="top">
      <div
        role="group"
        tabIndex={0}
        aria-label={`${label}: ${value}${unit === undefined ? '' : ` ${unit}`}`}
        className="flex flex-col gap-0.5 rounded-sm focus-visible:outline-1"
      >
        <span className="truncate text-[9px] font-medium uppercase tracking-[0.07em] text-ink-faint">
          {label}
        </span>
        <span className={cn('nf-numeric truncate text-[11.5px] leading-none text-ink', tone)}>
          {value}
          {unit !== undefined ? (
            <span className="pl-0.5 text-[9px] text-ink-faint">{unit}</span>
          ) : null}
        </span>
      </div>
    </Tooltip>
  );
}

interface NoticeProps {
  status: Status;
  reverseHops: number | null;
  sourceName: string | null;
  targetName: string | null;
  /** The endpoint the status complains about still exists in the document. */
  inDocument: boolean;
  onSwap: () => void;
  onRecompute: () => void;
  onClearSource: () => void;
  onClearTarget: () => void;
}

/** Every dead end gets its own sentence; a blank panel explains nothing. */
function Notice({
  status,
  reverseHops,
  sourceName,
  targetName,
  inDocument,
  onSwap,
  onRecompute,
  onClearSource,
  onClearTarget,
}: NoticeProps) {
  if (status === 'unset') {
    return (
      <p className="text-[10.5px] leading-snug text-ink-faint">
        Pick a source and a target above — from the current selection, or by searching for a
        label — and this panel traces the synaptic chain between them.
      </p>
    );
  }

  if (status === 'source-gone' || status === 'target-gone') {
    const which = status === 'source-gone' ? 'source' : 'target';
    // A cell the document still holds has simply not been loaded into the
    // engine yet, which is a different problem from a deleted one and takes a
    // different fix.
    return (
      <div className="flex flex-col gap-1.5">
        <p
          className={cn(
            'flex items-start gap-1.5 text-[10.5px] leading-snug',
            inDocument ? 'text-ink-muted' : 'text-danger',
          )}
        >
          <TriangleAlert
            size={12}
            className={cn('mt-px shrink-0', inDocument && 'text-warning')}
            aria-hidden
          />
          <span>
            {inDocument
              ? `The ${which} cell is in the circuit but has not reached the running network yet.`
              : `The ${which} cell is gone — it was deleted, or the circuit was replaced around it.`}
          </span>
        </p>
        <Button
          size="sm"
          variant="secondary"
          className="h-6 self-start px-2 text-[10.5px]"
          icon={inDocument ? <RefreshCw size={11} /> : undefined}
          onClick={
            inDocument ? onRecompute : status === 'source-gone' ? onClearSource : onClearTarget
          }
        >
          {inDocument ? 'Search again' : `Clear the ${which}`}
        </Button>
      </div>
    );
  }

  if (status === 'same') {
    return (
      <p className="text-[10.5px] leading-snug text-ink-faint">
        Source and target are the same cell, so there is no route to trace. The spectrum below
        counts the recurrent loops that leave {sourceName ?? 'it'} and come back instead.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      <p className="flex items-start gap-1.5 text-[10.5px] leading-snug text-ink-muted">
        <CircleSlash size={12} className="mt-px shrink-0 text-warning" aria-hidden />
        <span>
          No directed route from {sourceName ?? 'the source'} to {targetName ?? 'the target'}.
          Nothing this cell fires can reach the other through the current wiring.
        </span>
      </p>
      {reverseHops !== null ? (
        <Button
          size="sm"
          variant="secondary"
          icon={<ArrowRightLeft size={11} />}
          className="h-6 self-start px-2 text-[10.5px]"
          onClick={onSwap}
        >
          The reverse runs in {reverseHops} hop{reverseHops === 1 ? '' : 's'} — swap
        </Button>
      ) : null}
    </div>
  );
}

interface EndpointFieldProps {
  role: 'Source' | 'Target';
  id: NeuronId | null;
  neuron: Neuron | null;
  /** The id is set but the engine has no slot for it. */
  missing: boolean;
  neurons: readonly Neuron[];
  selection: readonly NeuronId[];
  onPick: (id: NeuronId | null) => void;
  className?: string;
}

/**
 * One end of the search: the current cell as a chip, or a label search when
 * nothing is chosen. The swatch is the hue the cell is drawn in, so the row and
 * the glyph in the scene are visibly the same object.
 */
function EndpointField({
  role,
  id,
  neuron,
  missing,
  neurons,
  selection,
  onPick,
  className,
}: EndpointFieldProps) {
  const [query, setQuery] = useState('');

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (needle === '') return [];
    const hits: Neuron[] = [];
    for (const candidate of neurons) {
      if (
        candidate.label.toLowerCase().includes(needle) ||
        candidate.id.toLowerCase().includes(needle)
      ) {
        hits.push(candidate);
        if (hits.length === MAX_SUGGESTIONS) break;
      }
    }
    return hits;
  }, [neurons, query]);

  const takeSelection = () => {
    if (selection.length === 0) return;
    setQuery('');
    onPick(selection[0]);
  };

  return (
    <div className={cn('flex flex-col gap-1', className)}>
      <div className="flex items-center gap-1.5">
        <span className="w-10 shrink-0 text-[9px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
          {role}
        </span>

        {id !== null ? (
          <span
            className={cn(
              'flex h-6 min-w-0 flex-1 items-center gap-1.5 rounded-control border px-1.5',
              missing ? 'border-danger/40 bg-danger/10' : 'border-hairline bg-panel-raised',
            )}
          >
            <span
              aria-hidden
              className="size-2 shrink-0 rounded-[2px] ring-1 ring-white/15"
              style={{
                backgroundColor:
                  neuron === null ? 'var(--color-ink-faint)' : identityColorHex(neuron.morphology.seed),
              }}
            />
            <span
              className={cn(
                'min-w-0 flex-1 truncate text-[11px]',
                missing ? 'text-danger' : 'text-ink',
              )}
            >
              {neuron === null ? id.slice(0, 12) : nameOf(neuron)}
            </span>
            {missing ? (
              <span className="shrink-0 text-[8.5px] font-semibold uppercase tracking-[0.08em] text-danger">
                gone
              </span>
            ) : null}
            <IconButton
              label={`Clear the ${role.toLowerCase()}`}
              size="sm"
              className="size-4 shrink-0"
              onClick={() => {
                setQuery('');
                onPick(null);
              }}
            >
              <X size={10} />
            </IconButton>
          </span>
        ) : (
          <span className="flex h-6 min-w-0 flex-1 items-center gap-1.5 rounded-control border border-hairline bg-panel-raised px-1.5">
            <Search size={11} aria-hidden className="shrink-0 text-ink-faint" />
            <input
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Escape' && query !== '') {
                  event.preventDefault();
                  event.stopPropagation();
                  setQuery('');
                } else if (event.key === 'Enter' && matches.length > 0) {
                  event.preventDefault();
                  setQuery('');
                  onPick(matches[0].id);
                }
              }}
              placeholder={`Search for the ${role.toLowerCase()}…`}
              aria-label={`Search for the ${role.toLowerCase()} cell`}
              spellCheck={false}
              autoComplete="off"
              className="min-w-0 flex-1 bg-transparent text-[11px] text-ink outline-none placeholder:text-ink-faint"
            />
          </span>
        )}

        <Tooltip content={`Use the selected cell as the ${role.toLowerCase()}`}>
          <IconButton
            label={`Use the selected cell as the ${role.toLowerCase()}`}
            size="sm"
            className="size-6 shrink-0"
            disabled={selection.length === 0}
            onClick={takeSelection}
          >
            <Crosshair size={12} />
          </IconButton>
        </Tooltip>
      </div>

      {id === null && query.trim() !== '' ? (
        matches.length === 0 ? (
          <p className="pl-[46px] text-[9.5px] text-ink-faint">No cell matches that.</p>
        ) : (
          <ul className="ml-[46px] flex flex-col overflow-hidden rounded-control border border-hairline bg-panel-raised">
            {matches.map((candidate) => (
              <li key={candidate.id}>
                <button
                  type="button"
                  onClick={() => {
                    setQuery('');
                    onPick(candidate.id);
                  }}
                  className="flex w-full items-center gap-1.5 px-1.5 py-[3px] text-left transition-colors hover:bg-white/[0.06]"
                >
                  <span
                    aria-hidden
                    className="size-2 shrink-0 rounded-[2px]"
                    style={{ backgroundColor: identityColorHex(candidate.morphology.seed) }}
                  />
                  <span className="min-w-0 flex-1 truncate text-[10.5px] text-ink-muted">
                    {nameOf(candidate)}
                  </span>
                  <span
                    className={cn(
                      'shrink-0 text-[9px] font-semibold',
                      candidate.polarity === 'inhibitory' ? 'text-secondary' : 'text-accent',
                    )}
                  >
                    {candidate.polarity === 'inhibitory' ? 'I' : 'E'}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )
      ) : null}
    </div>
  );
}

interface Bin {
  key: string;
  label: string;
  value: number;
  title: string;
}

interface BarsProps {
  bins: readonly Bin[];
  color: string;
  caption: string;
  total: string;
}

const BAR_HEIGHT = 40;

/**
 * Compact column chart.
 *
 * Log scaled, because both quantities it draws — paths by length and cells by
 * depth — grow geometrically with the horizon, and on a linear axis every column
 * but the last is a flat line against the baseline.
 */
function Bars({ bins, color, caption, total }: BarsProps) {
  const peak = bins.reduce((max, bin) => Math.max(max, bin.value), 0);
  const logPeak = Math.log1p(peak);
  const width = Math.max(1, bins.length) * 10;
  const slot = width / Math.max(1, bins.length);
  const barWidth = Math.max(1, slot - 2.4);

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${BAR_HEIGHT}`}
        width="100%"
        height={BAR_HEIGHT}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${caption}: ${bins.map((bin) => `${bin.label} → ${bin.value}`).join(', ')}`}
        className="block"
      >
        <rect
          x={0}
          y={BAR_HEIGHT - 0.6}
          width={width}
          height={0.6}
          fill="var(--color-ink-faint)"
          opacity={0.45}
        />
        {bins.map((bin, index) => {
          if (bin.value <= 0) return null;
          const norm = logPeak > 0 ? Math.log1p(bin.value) / logPeak : 1;
          const height = Math.max(1, norm * (BAR_HEIGHT - 2));
          return (
            <rect
              key={bin.key}
              x={index * slot + (slot - barWidth) / 2}
              y={BAR_HEIGHT - 0.6 - height}
              width={barWidth}
              height={height}
              fill={color}
              opacity={0.85}
              className="transition-opacity hover:opacity-100"
            >
              <title>{bin.title}</title>
            </rect>
          );
        })}
      </svg>
      <div className="mt-0.5 flex" aria-hidden>
        {bins.map((bin) => (
          <span
            key={bin.key}
            className="nf-numeric min-w-0 flex-1 text-center text-[9px] text-ink-muted"
          >
            {bin.value > 0 ? compact(bin.value) : '·'}
          </span>
        ))}
      </div>
      <div className="flex" aria-hidden>
        {bins.map((bin) => (
          <span
            key={bin.key}
            className="nf-numeric min-w-0 flex-1 text-center text-[8.5px] text-ink-faint"
          >
            {bin.label}
          </span>
        ))}
      </div>
      <div className="mt-0.5 flex items-baseline justify-between text-[9px] text-ink-faint">
        <span>{caption}</span>
        <span className="nf-numeric">{total}</span>
      </div>
    </div>
  );
}
