'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Ban,
  FlaskConical,
  Play,
  Scissors,
  Square,
  TrendingUp,
  TriangleAlert,
  Waves,
  X,
  Zap,
} from 'lucide-react';
import {
  Badge,
  Button,
  EmptyState,
  IconButton,
  Meter,
  NumberField,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  SegmentedControl,
  Select,
  SelectItem,
  Separator,
  Sparkline,
  Switch,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import type { SegmentedOption } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { identityColorHex } from '@neuroforge/shared';
import type { Circuit, NeuronId } from '@neuroforge/shared';

import { compact, fixed, grouped } from '@/lib/format';
import {
  FREQUENCY_BANDS,
  MIN_SPECTRUM_SPIKES,
  isAborted,
  parseTargetSpec,
  randomSlotSample,
  resolveTargetSlots,
  runLesion,
  runPerturbation,
  runRhythm,
  runTransfer,
  targetSpecValue,
} from '@/lib/experiments/network';
import type {
  CellRef,
  ExperimentProgress,
  ExperimentResult,
  LesionResult,
  LesionTarget,
  PerturbationResult,
  RhythmResult,
  SpectrumResult,
  TargetSpec,
  TransferResult,
} from '@/lib/experiments/network';

/* -------------------------------------------------------------- protocols -- */

type ProtocolKey = 'rhythm' | 'lesion' | 'transfer' | 'perturbation';

interface Protocol {
  key: ProtocolKey;
  name: string;
  blurb: string;
  icon: typeof Waves;
}

const PROTOCOLS: readonly Protocol[] = [
  {
    key: 'rhythm',
    name: 'Oscillation spectrum',
    blurb: 'Power spectrum of the population rate, plus the synchrony index.',
    icon: Waves,
  },
  {
    key: 'lesion',
    name: 'Lesion study',
    blurb: 'Ablate a set of cells and compare against a same-seed control.',
    icon: Scissors,
  },
  {
    key: 'transfer',
    name: 'Input–output transfer',
    blurb: 'Sweep a Poisson drive and read the downstream rate out.',
    icon: TrendingUp,
  },
  {
    key: 'perturbation',
    name: 'Perturbation sensitivity',
    blurb: 'One extra spike, two identical runs, and how fast they separate.',
    icon: Zap,
  },
];

/* ----------------------------------------------------------------- params -- */

interface RhythmState {
  durationMs: number;
  warmupMs: number;
  binMs: number;
  seed: number;
}

interface LesionState {
  target: LesionTarget;
  size: number;
  durationMs: number;
  warmupMs: number;
  binMs: number;
  seed: number;
}

interface TransferState {
  input: TargetSpec;
  output: TargetSpec;
  outputExcludesInput: boolean;
  minRateHz: number;
  maxRateHz: number;
  levels: number;
  amplitudePa: number;
  durationMs: number;
  warmupMs: number;
  seed: number;
}

interface PerturbationState {
  useSelection: boolean;
  amplitudePa: number;
  warmupMs: number;
  durationMs: number;
  sampleMs: number;
  seed: number;
}

/**
 * Defaults are sized so a first run finishes in a couple of seconds on a small
 * circuit while still resolving the delta band: a 1200 ms record puts the
 * frequency resolution below 1 Hz, which is the shortest record that can say
 * anything about a 1–4 Hz rhythm at all.
 */
const DEFAULT_RHYTHM: RhythmState = { durationMs: 1200, warmupMs: 200, binMs: 1, seed: 1 };

const DEFAULT_LESION: LesionState = {
  target: 'hubs',
  size: 8,
  durationMs: 800,
  warmupMs: 200,
  binMs: 1,
  seed: 1,
};

const DEFAULT_TRANSFER: TransferState = {
  input: { kind: 'selection' },
  output: { kind: 'all' },
  outputExcludesInput: true,
  minRateHz: 0,
  maxRateHz: 120,
  levels: 6,
  amplitudePa: 400,
  durationMs: 300,
  warmupMs: 120,
  seed: 1,
};

const DEFAULT_PERTURBATION: PerturbationState = {
  useSelection: true,
  amplitudePa: 6000,
  warmupMs: 250,
  durationMs: 350,
  sampleMs: 2,
  seed: 1,
};

const LESION_OPTIONS: readonly SegmentedOption<LesionTarget>[] = [
  { value: 'selection', label: 'selection', title: 'Ablate the cells selected in the scene' },
  { value: 'hubs', label: 'hubs', title: 'Ablate the most connected cells, by total degree' },
  { value: 'random', label: 'random', title: 'Ablate a deterministic random sample' },
];

interface ResultEntry {
  result: ExperimentResult;
  /** Document revision the measurement was taken at; see `revision` below. */
  revision: number;
}

type ResultSet = Partial<Record<ProtocolKey, ResultEntry>>;

type Status = 'idle' | 'running' | 'done' | 'cancelled' | 'error';

export interface NetworkExperimentsPanelProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Overrides the default placement. */
  className?: string;
}

/**
 * Population-level experiments over the current circuit.
 *
 * Nothing here touches the running simulation: every protocol builds its own
 * engine from a copy of the document, so a lesion study deletes nothing and a
 * sweep leaves the network on screen exactly where the user left it. Long runs
 * yield between slices and stop on the frame Cancel is pressed.
 */
export function NetworkExperimentsPanel({
  open = true,
  onClose,
  className,
}: NetworkExperimentsPanelProps) {
  const circuit = useEditor((s) => s.circuit);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [protocol, setProtocol] = useState<ProtocolKey>('rhythm');
  const [rhythm, setRhythm] = useState<RhythmState>(DEFAULT_RHYTHM);
  const [lesion, setLesion] = useState<LesionState>(DEFAULT_LESION);
  const [transfer, setTransfer] = useState<TransferState>(DEFAULT_TRANSFER);
  const [perturbation, setPerturbation] = useState<PerturbationState>(DEFAULT_PERTURBATION);

  const [status, setStatus] = useState<Status>('idle');
  const [progress, setProgress] = useState<ExperimentProgress>({ fraction: 0, label: '' });
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultSet>({});

  const abortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  // The editor drafts copy-on-write, so any edit to a cell or a synapse replaces
  // these arrays. A result measured before that edit describes a network that no
  // longer exists, and saying so is better than either discarding the work or
  // presenting it as current.
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    setRevision((previous) => previous + 1);
  }, [circuit.neurons, circuit.synapses]);

  /* ------------------------------------------------------------ previews -- */

  const inputSlots = useMemo(
    () =>
      protocol === 'transfer' ? resolveTargetSlots(circuit, transfer.input, selection) : [],
    [protocol, circuit, transfer.input, selection],
  );

  const outputSlots = useMemo(
    () =>
      protocol === 'transfer'
        ? resolveTargetSlots(
            circuit,
            transfer.output,
            selection,
            transfer.outputExcludesInput ? inputSlots : null,
          )
        : [],
    [protocol, circuit, transfer.output, transfer.outputExcludesInput, selection, inputSlots],
  );

  const lesionPreview = useMemo<readonly CellRef[]>(() => {
    if (protocol !== 'lesion') return [];
    if (lesion.target === 'selection') {
      return resolveTargetSlots(circuit, { kind: 'selection' }, selection).map((slot) =>
        cellOf(circuit, slot),
      );
    }
    if (lesion.target === 'random') {
      return randomSlotSample(circuit, lesion.size, lesion.seed).map((slot) =>
        cellOf(circuit, slot),
      );
    }
    // Hubs need a full adjacency pass, which is too expensive to run on every
    // keystroke; the run resolves them and the result lists what it hit.
    return [];
  }, [protocol, circuit, selection, lesion.target, lesion.size, lesion.seed]);

  const perturbTarget = useMemo<CellRef | null>(() => {
    if (protocol !== 'perturbation') return null;
    if (!perturbation.useSelection || selection.length === 0) return null;
    const slot = circuit.neurons.findIndex((neuron) => neuron.id === selection[0]);
    return slot < 0 ? null : cellOf(circuit, slot);
  }, [protocol, circuit, selection, perturbation.useSelection]);

  /* ---------------------------------------------------------------- cost -- */

  const dt = circuit.simulation.dt;
  const cost = useMemo(() => {
    switch (protocol) {
      case 'rhythm':
        return { simMs: rhythm.warmupMs + rhythm.durationMs, runs: 1 };
      case 'lesion':
        return { simMs: (lesion.warmupMs + lesion.durationMs) * 2, runs: 2 };
      case 'transfer':
        return {
          simMs: (transfer.warmupMs + transfer.durationMs) * Math.max(2, transfer.levels),
          runs: Math.max(2, transfer.levels),
        };
      default:
        return { simMs: (perturbation.warmupMs + perturbation.durationMs) * 2, runs: 2 };
    }
  }, [protocol, rhythm, lesion, transfer, perturbation]);

  const blocker = useMemo<string | null>(() => {
    if (circuit.neurons.length === 0) return 'Build a circuit first — there is nothing to measure.';
    if (protocol === 'lesion' && lesion.target === 'selection' && selection.length === 0) {
      return 'Select the cells to ablate, or ablate hubs or a random sample instead.';
    }
    if (protocol === 'transfer') {
      if (inputSlots.length === 0) {
        return transfer.input.kind === 'selection'
          ? 'Nothing is selected to drive. Select cells, or pick a population.'
          : 'The driven population is empty.';
      }
      if (outputSlots.length === 0) {
        return transfer.outputExcludesInput
          ? 'Every cell in the readout is also driven; there is nothing left to measure.'
          : 'The readout population is empty.';
      }
    }
    if (protocol === 'perturbation' && circuit.synapses.length === 0) {
      return 'A perturbation cannot propagate through a circuit with no synapses.';
    }
    return null;
  }, [
    circuit,
    protocol,
    lesion.target,
    selection,
    inputSlots,
    outputSlots,
    transfer.input.kind,
    transfer.outputExcludesInput,
  ]);

  /* ----------------------------------------------------------------- run -- */

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const run = useCallback(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus('running');
    setError(null);
    setProgress({ fraction: 0, label: 'Building the harness' });

    const ctx = {
      signal: controller.signal,
      onProgress: (next: ExperimentProgress) => {
        if (mountedRef.current && abortRef.current === controller) setProgress(next);
      },
    };

    const started = (async (): Promise<ExperimentResult> => {
      switch (protocol) {
        case 'rhythm':
          return runRhythm(circuit, rhythm, ctx);
        case 'lesion':
          return runLesion(circuit, { ...lesion, selection }, ctx);
        case 'transfer':
          return runTransfer(circuit, transfer, ctx, selection);
        default:
          return runPerturbation(
            circuit,
            {
              target: perturbation.useSelection && selection.length > 0 ? selection[0] : null,
              amplitudePa: perturbation.amplitudePa,
              warmupMs: perturbation.warmupMs,
              durationMs: perturbation.durationMs,
              sampleMs: perturbation.sampleMs,
              seed: perturbation.seed,
            },
            ctx,
          );
      }
    })();

    void started
      .then((result) => {
        // A superseded run must not publish over the one that replaced it.
        if (!mountedRef.current || abortRef.current !== controller) return;
        setResults((previous) => ({ ...previous, [protocol]: { result, revision } }));
        setStatus('done');
        abortRef.current = null;
      })
      .catch((cause: unknown) => {
        if (!mountedRef.current || abortRef.current !== controller) return;
        abortRef.current = null;
        if (isAborted(cause)) {
          setStatus('cancelled');
          return;
        }
        setError(cause instanceof Error ? cause.message : String(cause));
        setStatus('error');
      });
  }, [protocol, circuit, rhythm, lesion, transfer, perturbation, selection, revision]);

  const selectCell = useCallback(
    (cell: CellRef) => {
      if (cell.id !== null) select([cell.id]);
    },
    [select],
  );

  // A failure or a cancellation belongs to the protocol that produced it; the
  // badge and the error line would otherwise follow the user to a protocol they
  // never ran. A run in flight keeps its status, because it is still running.
  const chooseProtocol = useCallback((next: ProtocolKey) => {
    setProtocol(next);
    setStatus((current) => (current === 'running' ? current : 'idle'));
    setError(null);
  }, []);

  if (!open) return null;

  const running = status === 'running';
  const entry = results[protocol];
  const result = entry?.result;
  const stale = entry !== undefined && entry.revision !== revision;
  const placement = className ?? 'absolute top-3 right-3 bottom-3 w-[372px]';

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      <PanelHeader
        title="Experiments"
        subtitle={`${grouped(circuit.neurons.length)} cells · ${grouped(
          circuit.synapses.length,
        )} synapses · dt ${fixed(dt, 2)} ms`}
        icon={<FlaskConical />}
        actions={
          <>
            {running ? (
              <Badge variant="accent" size="sm" dot>
                running
              </Badge>
            ) : status === 'cancelled' ? (
              <Badge variant="warning" size="sm">
                cancelled
              </Badge>
            ) : status === 'error' ? (
              <Badge variant="danger" size="sm">
                failed
              </Badge>
            ) : stale ? (
              <Tooltip content="The circuit changed after this measurement. Run it again to bring the numbers back in step.">
                <Badge variant="warning" size="sm" tabIndex={0}>
                  stale
                </Badge>
              </Tooltip>
            ) : result !== undefined ? (
              <Tooltip content="Wall-clock time the last run of this protocol took">
                <Badge variant="outline" size="sm" numeric tabIndex={0}>
                  {formatSeconds(result.elapsedMs)}
                </Badge>
              </Tooltip>
            ) : null}
            {onClose ? (
              <IconButton label="Close experiments panel" size="sm" onClick={onClose}>
                <X />
              </IconButton>
            ) : null}
          </>
        }
      />

      <ScrollArea className="min-h-0 flex-1">
        <PanelSection label="Protocol" flush>
          {/* Toggle buttons rather than a radiogroup: a radiogroup owes the user
              arrow-key roving focus, and plain tab order is the honest thing to
              expose for four rows that are each independently reachable. */}
          <div role="group" aria-label="Experiment protocol" className="flex flex-col gap-px">
            {PROTOCOLS.map((entry) => {
              const Icon = entry.icon;
              const active = entry.key === protocol;
              return (
                <button
                  key={entry.key}
                  type="button"
                  aria-pressed={active}
                  onClick={() => chooseProtocol(entry.key)}
                  className={cn(
                    'relative flex w-full items-start gap-2 rounded-control px-1.5 py-1 text-left transition-colors',
                    'hover:bg-panel-raised focus-visible:bg-panel-raised',
                    active && 'bg-white/[0.07]',
                  )}
                >
                  {/* Accent rule on the active row, so the selection reads at a
                      glance without a control-sized affordance. */}
                  {active ? (
                    <span
                      aria-hidden
                      className="absolute inset-y-1 left-0 w-px rounded-full bg-accent"
                    />
                  ) : null}
                  <Icon
                    size={11}
                    aria-hidden
                    className={cn('mt-[3px] shrink-0', active ? 'text-accent' : 'text-ink-faint')}
                  />
                  <span className="flex min-w-0 flex-col">
                    <span
                      className={cn('truncate text-[11px]', active ? 'text-ink' : 'text-ink-muted')}
                    >
                      {entry.name}
                    </span>
                    <span className="text-[9.5px] leading-tight text-ink-faint">{entry.blurb}</span>
                  </span>
                  {results[entry.key] !== undefined ? (
                    <span
                      aria-hidden
                      title="Measured"
                      className="mt-[5px] ml-auto size-1.5 shrink-0 rounded-full bg-success/80"
                    />
                  ) : null}
                </button>
              );
            })}
          </div>
        </PanelSection>

        <Separator />

        <PanelSection label="Parameters">
          {protocol === 'rhythm' ? (
            <RhythmControls value={rhythm} onChange={setRhythm} disabled={running} />
          ) : protocol === 'lesion' ? (
            <LesionControls
              value={lesion}
              onChange={setLesion}
              disabled={running}
              preview={lesionPreview}
              selectionCount={selection.length}
              onSelectCell={selectCell}
            />
          ) : protocol === 'transfer' ? (
            <TransferControls
              value={transfer}
              onChange={setTransfer}
              disabled={running}
              circuit={circuit}
              inputCount={inputSlots.length}
              outputCount={outputSlots.length}
            />
          ) : (
            <PerturbationControls
              value={perturbation}
              onChange={setPerturbation}
              disabled={running}
              target={perturbTarget}
              onSelectCell={selectCell}
            />
          )}
        </PanelSection>

        <PanelSection>
          <div className="flex items-center gap-2">
            {running ? (
              <Button size="sm" variant="danger" icon={<Square />} onClick={cancel}>
                Cancel
              </Button>
            ) : (
              <Button
                size="sm"
                variant="primary"
                icon={<Play />}
                onClick={run}
                disabled={blocker !== null}
              >
                Run
              </Button>
            )}
            <Tooltip
              content={`${cost.runs} isolated run${cost.runs === 1 ? '' : 's'} of ${grouped(
                Math.round(cost.simMs / Math.max(1, cost.runs)),
              )} ms each, at ${fixed(dt, 2)} ms per step`}
            >
              <span
                tabIndex={0}
                className="nf-numeric ml-auto rounded-sm text-[10px] text-ink-faint focus-visible:outline-1"
              >
                {formatSeconds(cost.simMs)} simulated · {compact(Math.round(cost.simMs / dt))} steps
              </span>
            </Tooltip>
          </div>

          {running ? (
            <Meter
              className="mt-2"
              size="sm"
              value={progress.fraction}
              label={progress.label}
              valueLabel={`${(progress.fraction * 100).toFixed(0)}%`}
            />
          ) : null}

          {blocker !== null && !running ? (
            <p className="mt-2 flex items-start gap-1.5 text-[10px] leading-snug text-ink-faint">
              <Ban size={10} aria-hidden className="mt-[2px] shrink-0" />
              {blocker}
            </p>
          ) : null}

          {status === 'error' && error !== null ? (
            <p className="mt-2 flex items-start gap-1.5 text-[10px] leading-snug text-danger">
              <TriangleAlert size={10} aria-hidden className="mt-[2px] shrink-0" />
              {error}
            </p>
          ) : null}
        </PanelSection>

        <Separator />

        {stale ? (
          <p className="flex items-start gap-1.5 px-3 pt-2 text-[10px] leading-snug text-warning/90">
            <TriangleAlert size={10} aria-hidden className="mt-[2px] shrink-0" />
            The circuit changed after this measurement — everything below describes the network as
            it was.
          </p>
        ) : null}

        {result === undefined ? (
          <EmptyState
            compact
            icon={<Activity />}
            title="No measurement yet"
            description="Run the protocol and its results appear here. Every number is computed from a fresh simulation of this circuit."
          />
        ) : result.kind === 'rhythm' ? (
          <RhythmReport result={result} onSelectCell={selectCell} />
        ) : result.kind === 'lesion' ? (
          <LesionReport result={result} onSelectCell={selectCell} />
        ) : result.kind === 'transfer' ? (
          <TransferReport result={result} onSelectCell={selectCell} />
        ) : (
          <PerturbationReport result={result} onSelectCell={selectCell} />
        )}
      </ScrollArea>
    </Panel>
  );
}

/* ------------------------------------------------------------- parameters -- */

interface ControlProps<T> {
  value: T;
  onChange: (value: T) => void;
  disabled: boolean;
}

/** A compact numeric row; every parameter in this panel is one of these. */
function ParamRow({
  label,
  hint,
  disabled,
  children,
}: {
  label: string;
  hint: string;
  disabled: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        'flex min-w-0 items-center gap-2',
        disabled && 'pointer-events-none opacity-45',
      )}
    >
      {/* The tooltip hangs off the label rather than the row: the control is a
          drag-to-scrub field, and nothing else should be listening to pointers
          over it. */}
      <Tooltip content={hint} side="left">
        <span
          tabIndex={0}
          className="w-[76px] shrink-0 truncate rounded-sm text-[10.5px] text-ink-muted focus-visible:outline-1"
        >
          {label}
        </span>
      </Tooltip>
      <span className="ml-auto flex shrink-0 items-center">{children}</span>
    </div>
  );
}

function Lead({ children }: { children: React.ReactNode }) {
  return <p className="pb-1 text-[10px] leading-snug text-ink-faint">{children}</p>;
}

function RhythmControls({ value, onChange, disabled }: ControlProps<RhythmState>) {
  return (
    <div className="flex flex-col gap-1">
      <Lead>
        Bins the population spike train, transforms it, and reports which band the peak lands
        in — alongside the Golomb–Rinzel synchrony index over the same bins.
      </Lead>
      <ParamRow label="Record" hint="Window the spectrum is computed over. Longer records resolve closer peaks." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.durationMs}
          onChange={(next) => onChange({ ...value, durationMs: next })}
          min={100}
          max={20000}
          step={50}
          precision={0}
          unit="ms"
          aria-label="Recording window"
        />
      </ParamRow>
      <ParamRow label="Warm-up" hint="Stepped but not recorded, so the onset transient stays out of the spectrum." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.warmupMs}
          onChange={(next) => onChange({ ...value, warmupMs: next })}
          min={0}
          max={5000}
          step={25}
          precision={0}
          unit="ms"
          aria-label="Warm-up before recording"
        />
      </ParamRow>
      <ParamRow label="Bin" hint="Width of the population-rate bins; the sample rate of the transform is its reciprocal." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.binMs}
          onChange={(next) => onChange({ ...value, binMs: next })}
          min={0.25}
          max={10}
          step={0.25}
          precision={2}
          unit="ms"
          aria-label="Bin width"
        />
      </ParamRow>
      <ParamRow label="Seed" hint="Seeds the isolated engine's noise and release streams." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.seed}
          onChange={(next) => onChange({ ...value, seed: Math.round(next) })}
          min={0}
          max={99999}
          step={1}
          precision={0}
          aria-label="Run seed"
        />
      </ParamRow>
    </div>
  );
}

interface LesionControlProps extends ControlProps<LesionState> {
  preview: readonly CellRef[];
  selectionCount: number;
  onSelectCell: (cell: CellRef) => void;
}

function LesionControls({
  value,
  onChange,
  disabled,
  preview,
  selectionCount,
  onSelectCell,
}: LesionControlProps) {
  return (
    <div className="flex flex-col gap-1">
      <Lead>
        Runs a control and a lesioned network from the same seed and measures both over the
        surviving cells only, so the change is the ablation rather than a change of denominator.
      </Lead>
      <ParamRow label="Ablate" hint="Which cells to disable for the lesioned run." disabled={disabled}>
        <SegmentedControl
          size="sm"
          value={value.target}
          onChange={(next) => onChange({ ...value, target: next })}
          options={LESION_OPTIONS}
          aria-label="Cells to ablate"
        />
      </ParamRow>

      {value.target === 'selection' ? (
        <p className="text-[10px] text-ink-faint">
          {selectionCount === 0
            ? 'Nothing is selected.'
            : `${grouped(selectionCount)} selected cell${selectionCount === 1 ? '' : 's'} will be disabled.`}
        </p>
      ) : (
        <ParamRow
          label="Set size"
          hint={
            value.target === 'hubs'
              ? 'Cells with the highest combined in + out degree, resolved when the run starts.'
              : 'Size of the deterministic random sample; the seed below picks it.'
          }
          disabled={disabled}
        >
          <NumberField
            className="w-[96px]"
            value={value.size}
            onChange={(next) => onChange({ ...value, size: Math.round(next) })}
            min={1}
            max={4096}
            step={1}
            precision={0}
            unit="cells"
            aria-label="Number of cells to ablate"
          />
        </ParamRow>
      )}

      {preview.length > 0 ? <CellStrip cells={preview} onSelect={onSelectCell} /> : null}
      {value.target === 'hubs' ? (
        <p className="text-[9.5px] text-ink-faint">
          Hubs are resolved from the connectome when the run starts; the result lists exactly
          which cells were hit.
        </p>
      ) : null}

      <ParamRow label="Record" hint="Each of the two runs records this long, after its warm-up." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.durationMs}
          onChange={(next) => onChange({ ...value, durationMs: next })}
          min={100}
          max={20000}
          step={50}
          precision={0}
          unit="ms"
          aria-label="Recording window"
        />
      </ParamRow>
      <ParamRow label="Warm-up" hint="Discarded transient at the head of each run." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.warmupMs}
          onChange={(next) => onChange({ ...value, warmupMs: next })}
          min={0}
          max={5000}
          step={25}
          precision={0}
          unit="ms"
          aria-label="Warm-up before recording"
        />
      </ParamRow>
      <ParamRow
        label="Seed"
        hint="Control and lesion run from this same seed; that is what makes the two comparable at all."
        disabled={disabled}
      >
        <NumberField
          className="w-[96px]"
          value={value.seed}
          onChange={(next) => onChange({ ...value, seed: Math.round(next) })}
          min={0}
          max={99999}
          step={1}
          precision={0}
          aria-label="Run seed"
        />
      </ParamRow>
    </div>
  );
}

interface TransferControlProps extends ControlProps<TransferState> {
  circuit: Circuit;
  inputCount: number;
  outputCount: number;
}

function TransferControls({
  value,
  onChange,
  disabled,
  circuit,
  inputCount,
  outputCount,
}: TransferControlProps) {
  const options = useMemo(
    () => [
      { value: 'selection', label: 'Current selection' },
      { value: 'all', label: 'Every cell' },
      ...circuit.populations.map((population) => ({
        value: `population:${population.id}`,
        label: population.name.length > 0 ? population.name : population.id.slice(0, 8),
      })),
    ],
    [circuit.populations],
  );

  return (
    <div className="flex flex-col gap-1">
      <Lead>
        Delivers an independent Poisson event train to every driven cell and measures the
        readout population&rsquo;s firing rate. Every level runs from the same network seed and
        the same input stream, so a difference between two points is the rate and nothing else.
      </Lead>

      <ParamRow
        label="Driven"
        hint={`${grouped(inputCount)} cell${inputCount === 1 ? '' : 's'} receive the Poisson drive.`}
        disabled={disabled}
      >
        <Select
          className="w-[150px]"
          size="sm"
          value={targetSpecValue(value.input)}
          onValueChange={(next) => onChange({ ...value, input: parseTargetSpec(next) })}
          aria-label="Driven population"
        >
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </Select>
      </ParamRow>

      <ParamRow
        label="Readout"
        hint={`${grouped(outputCount)} cell${outputCount === 1 ? '' : 's'} are measured.`}
        disabled={disabled}
      >
        <Select
          className="w-[150px]"
          size="sm"
          value={targetSpecValue(value.output)}
          onValueChange={(next) => onChange({ ...value, output: parseTargetSpec(next) })}
          aria-label="Readout population"
        >
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </Select>
      </ParamRow>

      <div className="flex items-baseline gap-2 pb-0.5">
        <span className="w-[76px] shrink-0 text-[9.5px] text-ink-faint">
          {grouped(inputCount)} → {grouped(outputCount)} cells
        </span>
        <span className="nf-numeric ml-auto text-[9.5px] text-ink-faint">
          {value.levels} levels
        </span>
      </div>

      <ParamRow
        label="Exclude driven"
        hint="Keep the driven cells out of the readout, so the curve is transfer rather than echo."
        disabled={disabled}
      >
        <Switch
          size="sm"
          checked={value.outputExcludesInput}
          onCheckedChange={(next) => onChange({ ...value, outputExcludesInput: next })}
          aria-label="Exclude the driven cells from the readout"
        />
      </ParamRow>

      <ParamRow label="Rate" hint="Lowest and highest per-cell input rate in the sweep." disabled={disabled}>
        <span className="flex items-center gap-1">
          <NumberField
            className="w-[68px]"
            value={value.minRateHz}
            onChange={(next) => onChange({ ...value, minRateHz: next })}
            min={0}
            max={2000}
            step={5}
            precision={0}
            unit="Hz"
            aria-label="Lowest input rate"
          />
          <span aria-hidden className="text-[10px] text-ink-faint">
            →
          </span>
          <NumberField
            className="w-[68px]"
            value={value.maxRateHz}
            onChange={(next) => onChange({ ...value, maxRateHz: next })}
            min={1}
            max={4000}
            step={5}
            precision={0}
            unit="Hz"
            aria-label="Highest input rate"
          />
        </span>
      </ParamRow>

      <ParamRow label="Levels" hint="Conditions across the sweep; each is its own isolated run." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.levels}
          onChange={(next) => onChange({ ...value, levels: Math.round(next) })}
          min={2}
          max={24}
          step={1}
          precision={0}
          aria-label="Number of sweep levels"
        />
      </ParamRow>

      <ParamRow
        label="Event"
        hint="Charge of one input event, given as the amplitude of an equivalent 1 ms current pulse."
        disabled={disabled}
      >
        <NumberField
          className="w-[96px]"
          value={value.amplitudePa}
          onChange={(next) => onChange({ ...value, amplitudePa: next })}
          min={1}
          max={100000}
          step={25}
          precision={0}
          unit="pA"
          logarithmic
          aria-label="Input event amplitude"
        />
      </ParamRow>

      <ParamRow label="Record" hint="Recording window at each level, after that level's warm-up." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.durationMs}
          onChange={(next) => onChange({ ...value, durationMs: next })}
          min={50}
          max={5000}
          step={25}
          precision={0}
          unit="ms"
          aria-label="Recording window per level"
        />
      </ParamRow>
      <ParamRow label="Warm-up" hint="Time the drive runs before the readout is recorded." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.warmupMs}
          onChange={(next) => onChange({ ...value, warmupMs: next })}
          min={0}
          max={2000}
          step={25}
          precision={0}
          unit="ms"
          aria-label="Warm-up per level"
        />
      </ParamRow>
      <ParamRow
        label="Seed"
        hint="Shared by every level, so the event times at a low rate are a subset of those at a high one."
        disabled={disabled}
      >
        <NumberField
          className="w-[96px]"
          value={value.seed}
          onChange={(next) => onChange({ ...value, seed: Math.round(next) })}
          min={0}
          max={99999}
          step={1}
          precision={0}
          aria-label="Run seed"
        />
      </ParamRow>
    </div>
  );
}

interface PerturbationControlProps extends ControlProps<PerturbationState> {
  target: CellRef | null;
  onSelectCell: (cell: CellRef) => void;
}

function PerturbationControls({
  value,
  onChange,
  disabled,
  target,
  onSelectCell,
}: PerturbationControlProps) {
  return (
    <div className="flex flex-col gap-1">
      <Lead>
        Steps two identical networks in lockstep from one seed, injects a single extra spike
        into one cell of the second, and measures how fast the two trajectories separate.
      </Lead>

      <ParamRow
        label="Use selection"
        hint="On perturbs the first selected cell; off perturbs the highest-degree cell in the network."
        disabled={disabled}
      >
        <Switch
          size="sm"
          checked={value.useSelection}
          onCheckedChange={(next) => onChange({ ...value, useSelection: next })}
          aria-label="Perturb the first selected cell"
        />
      </ParamRow>
      <div className="flex items-center gap-1.5">
        <p className="min-w-0 flex-1 text-[9.5px] leading-snug text-ink-faint">
          {value.useSelection
            ? target === null
              ? 'Nothing is selected — the highest-degree cell will be used.'
              : 'The first selected cell receives the extra spike.'
            : 'The highest-degree cell receives the extra spike.'}
        </p>
        {target !== null ? <CellStrip cells={[target]} onSelect={onSelectCell} /> : null}
      </div>

      <ParamRow
        label="Pulse"
        hint="Amplitude of the 1 ms current pulse that evokes the extra spike. Too small and no spike is evoked; the result says so."
        disabled={disabled}
      >
        <NumberField
          className="w-[96px]"
          value={value.amplitudePa}
          onChange={(next) => onChange({ ...value, amplitudePa: next })}
          min={10}
          max={1000000}
          step={100}
          precision={0}
          unit="pA"
          logarithmic
          aria-label="Perturbation pulse amplitude"
        />
      </ParamRow>
      <ParamRow label="Warm-up" hint="Both runs settle for this long before the perturbation lands." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.warmupMs}
          onChange={(next) => onChange({ ...value, warmupMs: next })}
          min={0}
          max={5000}
          step={25}
          precision={0}
          unit="ms"
          aria-label="Warm-up before the perturbation"
        />
      </ParamRow>
      <ParamRow label="Observe" hint="How long the two runs are traced after the perturbation." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.durationMs}
          onChange={(next) => onChange({ ...value, durationMs: next })}
          min={50}
          max={5000}
          step={25}
          precision={0}
          unit="ms"
          aria-label="Observation window"
        />
      </ParamRow>
      <ParamRow label="Sample" hint="Interval between distance measurements along the trace." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.sampleMs}
          onChange={(next) => onChange({ ...value, sampleMs: next })}
          min={0.5}
          max={20}
          step={0.5}
          precision={1}
          unit="ms"
          aria-label="Sample interval"
        />
      </ParamRow>
      <ParamRow label="Seed" hint="Both runs use it; the injected pulse is the only difference between them." disabled={disabled}>
        <NumberField
          className="w-[96px]"
          value={value.seed}
          onChange={(next) => onChange({ ...value, seed: Math.round(next) })}
          min={0}
          max={99999}
          step={1}
          precision={0}
          aria-label="Run seed"
        />
      </ParamRow>
    </div>
  );
}

/* ---------------------------------------------------------------- reports -- */

interface ReportProps<T> {
  result: T;
  onSelectCell: (cell: CellRef) => void;
}

function RhythmReport({ result, onSelectCell }: ReportProps<RhythmResult>) {
  const { spectrum, synchrony, summary } = result;
  const quiet = summary.spikes < MIN_SPECTRUM_SPIKES;

  return (
    <>
      <PanelSection
        label="Rhythm"
        aside={
          spectrum.band !== null && !quiet ? (
            <Badge
              size="sm"
              variant="outline"
              numeric
              style={{ color: spectrum.band.color, borderColor: `${spectrum.band.color}55` }}
            >
              {spectrum.band.symbol} {spectrum.band.label}
            </Badge>
          ) : null
        }
      >
        <div className="grid grid-cols-2 gap-x-3 gap-y-2">
          <Stat
            label="Dominant"
            value={spectrum.dominantHz > 0 ? `${fixed(spectrum.dominantHz, 1)} Hz` : '—'}
            hint={`Peak of the population-rate spectrum between 1 and ${fixed(spectrum.searchMaxHz, 0)} Hz, interpolated between bins.`}
            color={quiet ? undefined : spectrum.band?.color}
          />
          <Stat
            label="Prominence"
            value={spectrum.prominence > 0 ? `${fixed(spectrum.prominence, 1)}×` : '—'}
            hint={`Peak power over the median power of the search band. A record this long with no rhythm in it reaches ${fixed(spectrum.flatProminence, 1)}× on its own, so a peak below that is noise.`}
            tone={
              spectrum.prominence > 0 && spectrum.prominence < spectrum.flatProminence
                ? 'text-warning'
                : 'text-accent'
            }
          />
          <Stat
            label="Synchrony"
            value={synchrony.index === null ? '—' : fixed(synchrony.index, 3)}
            hint={`Golomb–Rinzel χ² over ${grouped(synchrony.sampleSize)} sampled cells at a ${fixed(synchrony.binMs, 2)} ms bin. 1 is lockstep; an independent population of this size would sit near ${fixed(synchrony.asynchronousFloor, 3)}. χ = ${synchrony.chi === null ? '—' : fixed(synchrony.chi, 3)}.`}
            tone={
              synchrony.index === null
                ? undefined
                : synchrony.index > 0.5
                  ? 'text-warning'
                  : 'text-accent'
            }
          />
          <Stat
            label="Mean rate"
            value={`${fixed(summary.meanRateHz, 2)} Hz`}
            hint={`${grouped(summary.spikes)} spikes from ${grouped(summary.measured)} cells over ${formatSeconds(summary.durationMs)}.`}
          />
          <Stat
            label="Active"
            value={`${compact(summary.activeCells)}/${compact(summary.measured)}`}
            hint="Cells that fired at least once inside the recording window."
          />
          <Stat
            label="Resolution"
            value={`${fixed(spectrum.resolutionHz, 2)} Hz`}
            hint="Frequency resolution of this record before windowing; the Hann window widens the effective peak beyond it."
          />
        </div>
      </PanelSection>

      <PanelSection label="Power spectrum" aside={<BandKeyLegend />}>
        <SpectrumChart spectrum={spectrum} muted={quiet} />
        <BandTable spectrum={spectrum} />
      </PanelSection>

      <PanelSection
        label="Population rate"
        aside={
          <span className="nf-numeric text-[9.5px] text-ink-faint">
            {formatSeconds(summary.durationMs)} · Hz/cell
          </span>
        }
      >
        <Sparkline
          values={result.rate}
          width={320}
          height={38}
          color="var(--color-accent)"
          fill
          endpoint={false}
          strokeWidth={1}
          label={`Population rate over ${formatSeconds(summary.durationMs)}`}
          className="w-full"
        />
      </PanelSection>

      {result.busiest.length > 0 ? (
        <PanelSection label="Busiest cells">
          <ul className="flex flex-col">
            {result.busiest.map((cell) => (
              <li key={cell.slot}>
                <CellRow
                  cell={cell}
                  onSelect={onSelectCell}
                  trailing={`${fixed(cell.rateHz, 1)} Hz`}
                  secondary={`${grouped(cell.spikes)} spk`}
                />
              </li>
            ))}
          </ul>
        </PanelSection>
      ) : null}

      <WarningList warnings={result.warnings} />
    </>
  );
}

function LesionReport({ result, onSelectCell }: ReportProps<LesionResult>) {
  const { control, lesioned } = result;
  return (
    <>
      <PanelSection
        label="Lesion"
        aside={
          <span className="nf-numeric text-[9.5px] text-ink-faint">
            {grouped(result.ablated.length)} ablated · {grouped(result.survivors)} measured
          </span>
        }
      >
        <p className="mb-2 text-[10px] leading-snug text-ink-faint">
          Both runs use seed {result.summary.seed} and are measured over the same{' '}
          {grouped(result.survivors)} surviving cells, so the differences below are the lesion
          rather than a change of denominator.
        </p>
        <CompareRow
          label="Mean rate"
          unit="Hz"
          before={control.meanRateHz}
          after={lesioned.meanRateHz}
          digits={2}
        />
        <CompareRow
          label="Synchrony"
          unit=""
          before={control.synchrony}
          after={lesioned.synchrony}
          digits={3}
        />
        <CompareRow
          label="Dominant"
          unit="Hz"
          before={control.dominantHz}
          after={lesioned.dominantHz}
          digits={1}
        />
        <CompareRow
          label="Active cells"
          unit=""
          before={control.activeCells}
          after={lesioned.activeCells}
          digits={0}
        />
      </PanelSection>

      <PanelSection
        label="Spectra"
        aside={
          <div className="flex items-center gap-2 text-[9.5px] text-ink-faint">
            <LegendDot color="var(--color-ink-muted)" label="control" />
            <LegendDot color="var(--color-danger)" label="lesioned" />
          </div>
        }
      >
        <SpectrumChart
          spectrum={control.spectrum}
          overlay={lesioned.spectrum}
          overlayColor="var(--color-danger)"
          strokeColor="var(--color-ink-muted)"
          muted={control.spikes < MIN_SPECTRUM_SPIKES}
        />
        {/* Prominence sits next to each peak because a frequency that moved is
            only news when the peak it moved was above the noise floor. */}
        <div className="mt-1 flex items-baseline justify-between text-[9.5px] text-ink-faint">
          <span>
            control{' '}
            <span className="nf-numeric text-ink-muted">
              {control.dominantHz > 0 ? `${fixed(control.dominantHz, 1)} Hz` : '—'}
            </span>
            {control.band !== null ? ` · ${control.band.label} · ` : ' · '}
            <span className="nf-numeric">{fixed(control.spectrum.prominence, 1)}×</span>
          </span>
          <span>
            lesioned{' '}
            <span className="nf-numeric text-danger">
              {lesioned.dominantHz > 0 ? `${fixed(lesioned.dominantHz, 1)} Hz` : '—'}
            </span>
            {lesioned.band !== null ? ` · ${lesioned.band.label} · ` : ' · '}
            <span className="nf-numeric">{fixed(lesioned.spectrum.prominence, 1)}×</span>
          </span>
        </div>
        <p className="mt-1 text-[9.5px] leading-snug text-ink-faint">
          Prominence is the peak over the median power of the band; a record with no rhythm in it
          reaches {fixed(control.spectrum.flatProminence, 1)}× on its own.
        </p>
      </PanelSection>

      <PanelSection
        label="Ablated cells"
        aside={
          <span className="nf-numeric text-[9.5px] text-ink-faint">{result.target}</span>
        }
      >
        <CellStrip cells={result.ablated} onSelect={onSelectCell} limit={24} />
      </PanelSection>

      <WarningList warnings={result.warnings} />
    </>
  );
}

function TransferReport({ result, onSelectCell }: ReportProps<TransferResult>) {
  return (
    <>
      <PanelSection
        label="Transfer function"
        aside={
          <span className="nf-numeric text-[9.5px] text-ink-faint">
            {grouped(result.inputCells)} → {grouped(result.outputCells)}
          </span>
        }
      >
        <TransferChart result={result} />
        <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-2">
          <Stat
            label="Peak gain"
            value={`${fixed(result.maxSlope, 3)}`}
            hint="Steepest segment of the curve, in readout Hz per driven-cell input Hz."
          />
          <Stat
            label="At"
            value={`${fixed(result.maxSlopeAtHz, 1)} Hz`}
            hint="Input rate at which the curve is steepest."
          />
          <Stat
            label="Ceiling"
            value={`${fixed(result.ceilingHz, 2)} Hz`}
            hint="Readout rate at the highest drive level tested."
          />
          <Stat
            label="Event"
            value={`${grouped(result.amplitudePa)} pA`}
            hint="Charge of one input event, as the amplitude of an equivalent 1 ms pulse."
          />
        </div>
      </PanelSection>

      <PanelSection label="Levels">
        <div className="flex flex-col gap-px">
          <div className="flex items-baseline gap-2 pb-0.5 text-[9px] tracking-[0.08em] text-ink-faint uppercase">
            <span className="w-[54px] shrink-0">drive</span>
            <span className="w-[54px] shrink-0 text-right">delivered</span>
            <span className="flex-1 text-right">driven</span>
            <span className="w-[56px] shrink-0 text-right">readout</span>
          </div>
          {result.points.map((point, index) => (
            <div
              key={index}
              className="nf-numeric flex items-baseline gap-2 text-[10.5px]"
            >
              <span className="w-[54px] shrink-0 text-ink-faint">
                {fixed(point.requestedHz, 0)} Hz
              </span>
              <span className="w-[54px] shrink-0 text-right text-ink-muted">
                {fixed(point.deliveredHz, 1)}
              </span>
              <span className="flex-1 text-right text-secondary">
                {fixed(point.inputRateHz, 2)}
              </span>
              <span className="w-[56px] shrink-0 text-right text-accent">
                {fixed(point.outputRateHz, 2)}
              </span>
            </div>
          ))}
        </div>
      </PanelSection>

      <PanelSection label="Driven cells">
        <CellStrip cells={result.inputPreview} onSelect={onSelectCell} limit={8} />
        <p className="mt-1 text-[9.5px] text-ink-faint">
          {result.inputCells > result.inputPreview.length
            ? `First ${result.inputPreview.length} of ${grouped(result.inputCells)}.`
            : `${grouped(result.inputCells)} driven.`}{' '}
          Readout covers {grouped(result.outputCells)} cell
          {result.outputCells === 1 ? '' : 's'}.
        </p>
      </PanelSection>

      <WarningList warnings={result.warnings} />
    </>
  );
}

function PerturbationReport({ result, onSelectCell }: ReportProps<PerturbationResult>) {
  const color = identityColorHex(result.cell.colorSeed);
  const lambda = result.lambdaPerSecond;
  return (
    <>
      <PanelSection
        label="Divergence"
        aside={
          result.evoked ? (
            <Badge size="sm" variant="success">
              spike evoked
            </Badge>
          ) : (
            <Badge size="sm" variant="warning">
              no extra spike
            </Badge>
          )
        }
      >
        <DivergenceChart result={result} color={color} />
        <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-2">
          <Stat
            label="Growth"
            value={lambda === null ? '—' : `${fixed(lambda, 1)} /s`}
            hint={
              lambda === null
                ? 'The trace never grew over enough samples to fit a rate.'
                : `Least squares of log distance from ${fixed(result.fitFromMs, 1)} to ${fixed(result.fitToMs, 1)} ms. Positive means one spike reorganises the population; negative means the network forgets it.`
            }
            tone={lambda === null ? undefined : lambda > 0 ? 'text-danger' : 'text-success'}
          />
          <Stat
            label="Peak"
            value={`${fixed(result.peakDistance, 2)} mV`}
            hint="Largest RMS membrane-potential difference between the two runs."
          />
          <Stat
            label="Final"
            value={`${fixed(result.finalDistance, 2)} mV`}
            hint="Difference at the end of the observation window."
          />
          <Stat
            label="Touched"
            value={`${compact(result.divergedCells)}/${compact(result.neurons)}`}
            hint="Cells whose spike train differed from the control at any point."
          />
        </div>
        <p className="mt-2 text-[10px] leading-snug text-ink-faint">
          Both runs start from the same seed and differ only by the{' '}
          {grouped(result.amplitudePa)} pA pulse. Over the window the perturbed run fired{' '}
          <span className="nf-numeric text-ink">{signed(result.spikeDelta, 0)}</span> spikes
          relative to the control.
        </p>
      </PanelSection>

      <PanelSection label="Perturbed cell">
        <CellStrip cells={[result.cell]} onSelect={onSelectCell} />
      </PanelSection>

      <PanelSection
        label="Spike divergence"
        aside={
          <span className="nf-numeric text-[9.5px] text-ink-faint">share of cells per sample</span>
        }
      >
        <Sparkline
          values={result.spikeDivergence}
          count={result.samples}
          width={320}
          height={30}
          min={0}
          color="var(--color-secondary)"
          fill
          endpoint={false}
          strokeWidth={1}
          label="Share of cells whose spike output differed, over time"
          className="w-full"
        />
      </PanelSection>

      <WarningList warnings={result.warnings} />
    </>
  );
}

/* ------------------------------------------------------------- spectrum ---- */

const SPEC_W = 320;
const SPEC_H = 104;
const SPEC_PAD_L = 28;
const SPEC_PAD_R = 6;
const SPEC_PAD_T = 9;
const SPEC_AXIS_H = 14;
const SPEC_PLOT_W = SPEC_W - SPEC_PAD_L - SPEC_PAD_R;
const SPEC_PLOT_H = SPEC_H - SPEC_PAD_T - SPEC_AXIS_H;
/** Decades of power the log axis shows below the peak. */
const SPEC_DECADES = 4;

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

interface SpectrumGeometry {
  minHz: number;
  maxHz: number;
  peak: number;
  floor: number;
  x: (hz: number) => number;
  y: (power: number) => number;
}

function spectrumGeometry(spectrum: SpectrumResult, others: readonly SpectrumResult[]): SpectrumGeometry | null {
  if (!(spectrum.binHz > 0) || spectrum.searchMaxHz <= 1) return null;
  const minHz = 1;
  const maxHz = Math.max(4, spectrum.searchMaxHz);
  let peak = 0;
  for (const source of [spectrum, ...others]) {
    if (!(source.binHz > 0)) continue;
    const first = Math.max(1, Math.ceil(minHz / source.binHz));
    const last = Math.min(source.power.length - 1, Math.floor(maxHz / source.binHz));
    for (let k = first; k <= last; k += 1) if (source.power[k] > peak) peak = source.power[k];
  }
  if (!(peak > 0)) return null;
  const floor = peak / 10 ** SPEC_DECADES;
  const logSpan = Math.log10(maxHz / minHz);
  return {
    minHz,
    maxHz,
    peak,
    floor,
    x: (hz) => SPEC_PAD_L + clamp01(Math.log10(Math.max(hz, minHz) / minHz) / logSpan) * SPEC_PLOT_W,
    y: (power) =>
      SPEC_PAD_T +
      (1 - clamp01(Math.log10(Math.max(power, floor) / floor) / SPEC_DECADES)) * SPEC_PLOT_H,
  };
}

/**
 * Path through a spectrum, decimated onto the pixel grid.
 *
 * A log frequency axis crushes the top half of the transform into a handful of
 * columns, so bins are collapsed by taking the maximum in each column: averaging
 * would erase exactly the narrow peak the plot exists to show.
 */
function spectrumPath(spectrum: SpectrumResult, geometry: SpectrumGeometry): string {
  if (!(spectrum.binHz > 0)) return '';
  const first = Math.max(1, Math.ceil(geometry.minHz / spectrum.binHz));
  const last = Math.min(spectrum.power.length - 1, Math.floor(geometry.maxHz / spectrum.binHz));
  if (first > last) return '';

  let path = '';
  let column = -1;
  let columnMax = 0;
  let columnX = 0;
  for (let k = first; k <= last; k += 1) {
    const x = geometry.x(k * spectrum.binHz);
    const bucket = Math.round(x * 2);
    if (bucket !== column) {
      if (column >= 0) {
        path += `${path.length === 0 ? 'M' : 'L'}${columnX.toFixed(2)} ${geometry
          .y(columnMax)
          .toFixed(2)}`;
      }
      column = bucket;
      columnMax = spectrum.power[k];
      columnX = x;
    } else if (spectrum.power[k] > columnMax) {
      columnMax = spectrum.power[k];
    }
  }
  if (column >= 0) {
    path += `${path.length === 0 ? 'M' : 'L'}${columnX.toFixed(2)} ${geometry
      .y(columnMax)
      .toFixed(2)}`;
  }
  return path;
}

interface SpectrumChartProps {
  spectrum: SpectrumResult;
  overlay?: SpectrumResult;
  strokeColor?: string;
  overlayColor?: string;
  muted?: boolean;
}

function SpectrumChart({
  spectrum,
  overlay,
  strokeColor = 'var(--color-accent)',
  overlayColor,
  muted = false,
}: SpectrumChartProps) {
  const geometry = useMemo(
    () => spectrumGeometry(spectrum, overlay === undefined ? [] : [overlay]),
    [spectrum, overlay],
  );
  const primary = useMemo(
    () => (geometry === null ? '' : spectrumPath(spectrum, geometry)),
    [spectrum, geometry],
  );
  const secondary = useMemo(
    () => (geometry === null || overlay === undefined ? '' : spectrumPath(overlay, geometry)),
    [overlay, geometry],
  );

  if (geometry === null) {
    return (
      <div className="flex h-[76px] items-center justify-center rounded-control bg-white/[0.02] text-[10px] text-ink-faint">
        No measurable power in the population rate.
      </div>
    );
  }

  const baseline = SPEC_PAD_T + SPEC_PLOT_H;
  const peakX = geometry.x(spectrum.dominantHz);
  const showPeak = spectrum.dominantHz >= geometry.minHz && !muted;

  return (
    <svg
      viewBox={`0 0 ${SPEC_W} ${SPEC_H}`}
      width="100%"
      height={SPEC_H}
      role="img"
      aria-label={`Power spectrum of the population rate. Peak at ${fixed(
        spectrum.dominantHz,
        1,
      )} hertz${spectrum.band === null ? '' : `, in the ${spectrum.band.label} band`}.`}
      className="nf-numeric block"
    >
      {FREQUENCY_BANDS.map((band) => {
        const x0 = geometry.x(Math.max(band.lowHz, geometry.minHz));
        const x1 = geometry.x(Math.min(band.highHz, geometry.maxHz));
        if (x1 - x0 < 0.5) return null;
        const dominant = spectrum.band !== null && spectrum.band.key === band.key && !muted;
        return (
          <g key={band.key}>
            <rect
              x={x0}
              y={SPEC_PAD_T}
              width={x1 - x0}
              height={SPEC_PLOT_H}
              fill={band.color}
              opacity={dominant ? 0.14 : 0.055}
            />
            <rect x={x0} y={baseline} width={x1 - x0} height={2} fill={band.color} opacity={0.8} />
            {x1 - x0 > 20 ? (
              <text
                x={(x0 + x1) / 2}
                y={SPEC_H - 3}
                textAnchor="middle"
                fontSize={8}
                fill={dominant ? band.color : 'var(--color-ink-faint)'}
              >
                {band.symbol}
              </text>
            ) : null}
          </g>
        );
      })}

      {FREQUENCY_BANDS.map((band) => (
        <line
          key={`edge-${band.key}`}
          x1={geometry.x(band.lowHz)}
          x2={geometry.x(band.lowHz)}
          y1={SPEC_PAD_T}
          y2={baseline}
          stroke="var(--color-hairline)"
          strokeWidth={0.75}
        />
      ))}

      {secondary.length > 0 ? (
        <path
          d={secondary}
          fill="none"
          stroke={overlayColor ?? 'var(--color-danger)'}
          strokeWidth={1}
          strokeLinejoin="round"
          opacity={0.9}
        />
      ) : null}
      {primary.length > 0 ? (
        <path
          d={primary}
          fill="none"
          stroke={strokeColor}
          strokeWidth={1.1}
          strokeLinejoin="round"
          opacity={muted ? 0.35 : 1}
        />
      ) : null}

      {showPeak ? (
        <>
          <line
            x1={peakX}
            x2={peakX}
            y1={SPEC_PAD_T}
            y2={baseline}
            stroke={spectrum.band?.color ?? 'var(--color-ink)'}
            strokeWidth={0.75}
            strokeDasharray="2 2"
          />
          <text
            x={Math.min(peakX + 3, SPEC_W - SPEC_PAD_R - 34)}
            y={SPEC_PAD_T + 8}
            fontSize={8.5}
            fill={spectrum.band?.color ?? 'var(--color-ink)'}
          >
            {fixed(spectrum.dominantHz, 1)} Hz
          </text>
        </>
      ) : null}

      <text x={2} y={SPEC_PAD_T + 6} fontSize={8} fill="var(--color-ink-faint)">
        peak
      </text>
      <text x={2} y={baseline} fontSize={8} fill="var(--color-ink-faint)">
        −{SPEC_DECADES}
      </text>
      <text x={SPEC_PAD_L} y={SPEC_H - 3} fontSize={8} fill="var(--color-ink-faint)">
        1
      </text>
      <text
        x={SPEC_W - SPEC_PAD_R}
        y={SPEC_H - 3}
        textAnchor="end"
        fontSize={8}
        fill="var(--color-ink-faint)"
      >
        {fixed(geometry.maxHz, 0)} Hz
      </text>
    </svg>
  );
}

function BandKeyLegend() {
  return <span className="text-[9.5px] text-ink-faint">log power · log Hz</span>;
}

function BandTable({ spectrum }: { spectrum: SpectrumResult }) {
  return (
    <ul className="mt-1.5 flex flex-col gap-0.5">
      {spectrum.bands.map((entry) => {
        const dominant = spectrum.band !== null && spectrum.band.key === entry.band.key;
        return (
          <li key={entry.band.key} className="flex items-center gap-1.5 text-[10px]">
            <span
              aria-hidden
              className="size-2 shrink-0 rounded-[2px]"
              style={{ backgroundColor: entry.band.color }}
            />
            <span className={cn('w-[42px] shrink-0', dominant ? 'text-ink' : 'text-ink-muted')}>
              {entry.band.label}
            </span>
            <span className="nf-numeric w-[48px] shrink-0 text-[9.5px] text-ink-faint">
              {entry.band.lowHz}–{entry.band.highHz}
            </span>
            <span className="h-[3px] min-w-0 flex-1 overflow-hidden rounded-full bg-white/[0.05]">
              <span
                className="block h-full rounded-full"
                style={{
                  width: `${clamp01(entry.share) * 100}%`,
                  backgroundColor: entry.band.color,
                  opacity: dominant ? 1 : 0.7,
                }}
              />
            </span>
            <span className="nf-numeric w-[34px] shrink-0 text-right text-ink-faint">
              {(entry.share * 100).toFixed(0)}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/* ------------------------------------------------------------- transfer ---- */

const XFER_W = 320;
const XFER_H = 112;
const XFER_PAD_L = 30;
const XFER_PAD_R = 8;
const XFER_PAD_T = 8;
const XFER_AXIS_H = 14;
const XFER_PLOT_W = XFER_W - XFER_PAD_L - XFER_PAD_R;
const XFER_PLOT_H = XFER_H - XFER_PAD_T - XFER_AXIS_H;

function TransferChart({ result }: { result: TransferResult }) {
  const points = result.points;
  const maxX = points.reduce((max, point) => Math.max(max, point.deliveredHz), 0);
  const maxY = points.reduce((max, point) => Math.max(max, point.outputRateHz), 0);
  const spanX = maxX > 0 ? maxX : 1;
  const spanY = maxY > 0 ? maxY * 1.1 : 1;

  const x = (value: number): number => XFER_PAD_L + clamp01(value / spanX) * XFER_PLOT_W;
  const y = (value: number): number =>
    XFER_PAD_T + (1 - clamp01(value / spanY)) * XFER_PLOT_H;
  const baseline = XFER_PAD_T + XFER_PLOT_H;

  const path = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'}${x(point.deliveredHz).toFixed(2)} ${y(point.outputRateHz).toFixed(2)}`)
    .join('');

  return (
    <svg
      viewBox={`0 0 ${XFER_W} ${XFER_H}`}
      width="100%"
      height={XFER_H}
      role="img"
      aria-label={`Input–output transfer curve. Readout reaches ${fixed(
        result.ceilingHz,
        2,
      )} hertz at the highest drive.`}
      className="nf-numeric block"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
        <line
          key={fraction}
          x1={XFER_PAD_L}
          x2={XFER_W - XFER_PAD_R}
          y1={XFER_PAD_T + fraction * XFER_PLOT_H}
          y2={XFER_PAD_T + fraction * XFER_PLOT_H}
          stroke="var(--color-hairline)"
          strokeWidth={0.75}
        />
      ))}
      <line
        x1={XFER_PAD_L}
        x2={XFER_PAD_L}
        y1={XFER_PAD_T}
        y2={baseline}
        stroke="var(--color-hairline-strong)"
        strokeWidth={0.75}
      />

      {maxY > 0 ? (
        <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={1.25} strokeLinejoin="round" />
      ) : null}

      {points.map((point, index) => (
        <circle
          key={index}
          cx={x(point.deliveredHz)}
          cy={y(point.outputRateHz)}
          r={1.9}
          fill="var(--color-accent)"
        >
          <title>
            {`${fixed(point.deliveredHz, 1)} Hz in → ${fixed(point.outputRateHz, 2)} Hz out (${grouped(point.outputSpikes)} spikes)`}
          </title>
        </circle>
      ))}

      <text x={2} y={XFER_PAD_T + 6} fontSize={8} fill="var(--color-ink-faint)">
        {fixed(spanY, spanY < 10 ? 1 : 0)}
      </text>
      <text x={2} y={baseline} fontSize={8} fill="var(--color-ink-faint)">
        0
      </text>
      <text x={XFER_PAD_L} y={XFER_H - 3} fontSize={8} fill="var(--color-ink-faint)">
        0
      </text>
      <text
        x={XFER_W - XFER_PAD_R}
        y={XFER_H - 3}
        textAnchor="end"
        fontSize={8}
        fill="var(--color-ink-faint)"
      >
        {fixed(maxX, 0)} Hz in
      </text>
    </svg>
  );
}

/* ----------------------------------------------------------- divergence ---- */

const DIV_W = 320;
const DIV_H = 112;
const DIV_PAD_L = 30;
const DIV_PAD_R = 8;
const DIV_PAD_T = 8;
const DIV_AXIS_H = 14;
const DIV_PLOT_W = DIV_W - DIV_PAD_L - DIV_PAD_R;
const DIV_PLOT_H = DIV_H - DIV_PAD_T - DIV_AXIS_H;
const DIV_DECADES = 6;

function DivergenceChart({ result, color }: { result: PerturbationResult; color: string }) {
  const geometry = useMemo(() => {
    let peak = 0;
    for (let i = 0; i < result.samples; i += 1) {
      if (result.distance[i] > peak) peak = result.distance[i];
    }
    if (!(peak > 0)) return null;
    const floor = peak / 10 ** DIV_DECADES;
    const spanT = result.times[result.samples - 1] || 1;
    return { peak, floor, spanT };
  }, [result]);

  if (geometry === null) {
    return (
      <div className="flex h-[80px] items-center justify-center rounded-control bg-white/[0.02] text-[10px] text-ink-faint">
        The two runs never separated inside this window.
      </div>
    );
  }

  const x = (ms: number): number => DIV_PAD_L + clamp01(ms / geometry.spanT) * DIV_PLOT_W;
  const y = (value: number): number =>
    DIV_PAD_T +
    (1 - clamp01(Math.log10(Math.max(value, geometry.floor) / geometry.floor) / DIV_DECADES)) *
      DIV_PLOT_H;
  const baseline = DIV_PAD_T + DIV_PLOT_H;

  let path = '';
  for (let i = 0; i < result.samples; i += 1) {
    path += `${i === 0 ? 'M' : 'L'}${x(result.times[i]).toFixed(2)} ${y(result.distance[i]).toFixed(2)}`;
  }

  const fitted = result.lambdaPerSecond !== null && result.fitToMs > result.fitFromMs;

  return (
    <svg
      viewBox={`0 0 ${DIV_W} ${DIV_H}`}
      width="100%"
      height={DIV_H}
      role="img"
      aria-label={`Divergence between the control and perturbed runs, peaking at ${fixed(
        result.peakDistance,
        2,
      )} millivolts.`}
      className="nf-numeric block"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
        <line
          key={fraction}
          x1={DIV_PAD_L}
          x2={DIV_W - DIV_PAD_R}
          y1={DIV_PAD_T + fraction * DIV_PLOT_H}
          y2={DIV_PAD_T + fraction * DIV_PLOT_H}
          stroke="var(--color-hairline)"
          strokeWidth={0.75}
        />
      ))}

      {fitted ? (
        <rect
          x={x(result.fitFromMs)}
          y={DIV_PAD_T}
          width={Math.max(1, x(result.fitToMs) - x(result.fitFromMs))}
          height={DIV_PLOT_H}
          fill="var(--color-ink)"
          opacity={0.05}
        />
      ) : null}

      <path d={path} fill="none" stroke={color} strokeWidth={1.25} strokeLinejoin="round" />

      <text x={2} y={DIV_PAD_T + 6} fontSize={8} fill="var(--color-ink-faint)">
        {fixed(geometry.peak, geometry.peak < 10 ? 1 : 0)}
      </text>
      <text x={2} y={baseline} fontSize={8} fill="var(--color-ink-faint)">
        mV
      </text>
      <text x={DIV_PAD_L} y={DIV_H - 3} fontSize={8} fill="var(--color-ink-faint)">
        0
      </text>
      {fitted ? (
        <text
          x={x(result.fitFromMs) + 2}
          y={DIV_H - 3}
          fontSize={8}
          fill="var(--color-ink-faint)"
        >
          fit
        </text>
      ) : null}
      <text
        x={DIV_W - DIV_PAD_R}
        y={DIV_H - 3}
        textAnchor="end"
        fontSize={8}
        fill="var(--color-ink-faint)"
      >
        {fixed(geometry.spanT, 0)} ms
      </text>
    </svg>
  );
}

/* ----------------------------------------------------------------- pieces -- */

function cellOf(circuit: Circuit, slot: number): CellRef {
  const neuron = circuit.neurons[slot];
  if (neuron === undefined) return { slot, id: null, colorSeed: 0, label: `#${slot}` };
  return {
    slot,
    id: neuron.id as NeuronId,
    colorSeed: neuron.morphology.seed,
    label: neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8),
  };
}

function formatSeconds(ms: number): string {
  if (!Number.isFinite(ms)) return '—';
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function signed(value: number, digits: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return `±0${digits > 0 ? `.${'0'.repeat(digits)}` : ''}`;
  return `${value > 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
}

interface StatProps {
  label: string;
  value: string;
  hint: string;
  tone?: string;
  color?: string;
}

function Stat({ label, value, hint, tone, color }: StatProps) {
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
        <span
          className={cn('nf-numeric text-[12px] leading-none text-ink', tone)}
          style={color === undefined ? undefined : { color }}
        >
          {value}
        </span>
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

interface CompareRowProps {
  label: string;
  unit: string;
  before: number | null;
  after: number | null;
  digits: number;
}

/**
 * One before / after pair. Both bars share a scale so the change is read from
 * the geometry rather than from the numbers alone.
 */
function CompareRow({ label, unit, before, after, digits }: CompareRowProps) {
  const has = before !== null && after !== null;
  const scale = has ? Math.max(Math.abs(before), Math.abs(after)) : 0;
  const delta = has ? after - before : null;
  const relative = has && before !== 0 ? (after - before) / Math.abs(before) : null;

  return (
    <div className="flex flex-col gap-0.5 py-1">
      <div className="flex items-baseline gap-2 text-[10px]">
        <span className="text-ink-muted">{label}</span>
        <span className="nf-numeric ml-auto text-ink-faint">
          {before === null ? '—' : fixed(before, digits)}
        </span>
        <span aria-hidden className="text-ink-faint">
          →
        </span>
        <span className="nf-numeric w-[52px] text-right text-ink">
          {after === null ? '—' : fixed(after, digits)}
          {unit.length > 0 ? <span className="text-ink-faint"> {unit}</span> : null}
        </span>
        <span
          className={cn(
            'nf-numeric w-[52px] shrink-0 text-right text-[9.5px]',
            delta === null
              ? 'text-ink-faint'
              : delta > 0
                ? 'text-success'
                : delta < 0
                  ? 'text-danger'
                  : 'text-ink-faint',
          )}
        >
          {relative === null ? signed(delta ?? 0, digits) : `${signed(relative * 100, 0)}%`}
        </span>
      </div>
      <div className="flex flex-col gap-[2px]">
        <Bar value={has ? Math.abs(before) : 0} scale={scale} className="bg-ink-muted/70" />
        <Bar value={has ? Math.abs(after) : 0} scale={scale} className="bg-danger" />
      </div>
    </div>
  );
}

function Bar({ value, scale, className }: { value: number; scale: number; className: string }) {
  return (
    <span className="block h-[3px] w-full overflow-hidden rounded-full bg-white/[0.05]">
      <span
        className={cn('block h-full rounded-full', className)}
        style={{ width: `${scale > 0 ? clamp01(value / scale) * 100 : 0}%` }}
      />
    </span>
  );
}

interface CellRowProps {
  cell: CellRef;
  onSelect: (cell: CellRef) => void;
  trailing?: string;
  secondary?: string;
}

function CellRow({ cell, onSelect, trailing, secondary }: CellRowProps) {
  const color = identityColorHex(cell.colorSeed);
  return (
    <button
      type="button"
      disabled={cell.id === null}
      onClick={() => onSelect(cell)}
      className={cn(
        'flex w-full items-center gap-1.5 rounded-control px-1.5 py-1 text-left text-[10.5px] transition-colors',
        'hover:bg-panel-raised focus-visible:bg-panel-raised disabled:pointer-events-none disabled:opacity-50',
      )}
    >
      {/* The same hue the renderer draws this cell in, so a row here and a glyph
          in the scene are visibly the same object. */}
      <span
        aria-hidden
        className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-white/15"
        style={{ backgroundColor: color }}
      />
      <span className="truncate text-ink">{cell.label}</span>
      {secondary !== undefined ? (
        <span className="nf-numeric ml-auto shrink-0 text-ink-faint">{secondary}</span>
      ) : null}
      {trailing !== undefined ? (
        <span
          className={cn(
            'nf-numeric w-[52px] shrink-0 text-right text-ink',
            secondary === undefined && 'ml-auto',
          )}
        >
          {trailing}
        </span>
      ) : null}
    </button>
  );
}

function CellStrip({
  cells,
  onSelect,
  limit = 16,
}: {
  cells: readonly CellRef[];
  onSelect: (cell: CellRef) => void;
  limit?: number;
}) {
  if (cells.length === 0) return null;
  const shown = cells.slice(0, limit);
  return (
    <div className="flex flex-wrap items-center gap-1">
      {shown.map((cell) => (
        <Tooltip key={cell.slot} content={`${cell.label} — slot ${cell.slot}`}>
          <button
            type="button"
            disabled={cell.id === null}
            onClick={() => onSelect(cell)}
            aria-label={`Select ${cell.label}`}
            className="size-3 shrink-0 rounded-[3px] ring-1 ring-white/15 transition-transform hover:scale-110 focus-visible:outline-1 disabled:pointer-events-none"
            style={{ backgroundColor: identityColorHex(cell.colorSeed) }}
          />
        </Tooltip>
      ))}
      {cells.length > shown.length ? (
        <span className="nf-numeric text-[9.5px] text-ink-faint">
          +{grouped(cells.length - shown.length)}
        </span>
      ) : null}
    </div>
  );
}

function WarningList({ warnings }: { warnings: readonly string[] }) {
  if (warnings.length === 0) return null;
  return (
    <PanelSection label="Read this before believing it">
      <ul className="flex flex-col gap-1.5">
        {warnings.map((warning) => (
          <li key={warning} className="flex items-start gap-1.5 text-[10px] leading-snug text-warning/90">
            <TriangleAlert size={10} aria-hidden className="mt-[2px] shrink-0" />
            <span>{warning}</span>
          </li>
        ))}
      </ul>
    </PanelSection>
  );
}
