'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Camera,
  Check,
  ChevronRight,
  ClockArrowLeft,
  Copy,
  Database,
  Download,
  FolderOpen,
  Library,
  RefreshCw,
  RotateCcw,
  Save,
  Scissors,
  Trash2,
  TriangleAlert,
  X,
} from 'lucide-react';
import {
  Badge,
  Button,
  Dialog,
  EmptyState,
  Field,
  IconButton,
  NumberField,
  Panel,
  PanelHeader,
  ScrollArea,
  Spinner,
  Tab,
  TabPanel,
  Tabs,
  TabsList,
  Tooltip,
  cn,
  pushToast,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import {
  createSnapshot,
  db,
  deleteCircuit,
  downloadExport,
  exportCircuit,
  listCircuits,
  listSnapshots,
  loadCircuit as readStoredCircuit,
  pruneSnapshots,
  restoreSnapshot,
  saveCircuit,
} from '@neuroforge/io';
import type { ExportFormat, ExportResult } from '@neuroforge/io';
import { identityColorHex } from '@neuroforge/shared';
import type { Circuit, CircuitId, Neuron, Snapshot } from '@neuroforge/shared';

import { grouped, since } from '@/lib/format';

/* ------------------------------------------------------------------ export -- */

interface FormatDescriptor {
  format: ExportFormat;
  name: string;
  description: string;
  binary: boolean;
}

const FORMATS: readonly FormatDescriptor[] = [
  {
    format: 'json',
    name: 'NeuroForge JSON',
    description: 'The whole document — topology, per-neuron parameters, stimuli, camera and scene.',
    binary: false,
  },
  {
    format: 'brian2',
    name: 'Brian2',
    description: 'Runnable Python: one NeuronGroup per model, with equations, delays and monitors.',
    binary: false,
  },
  {
    format: 'nest',
    name: 'PyNEST',
    description: 'Runnable Python mapping each model onto a built-in NEST model, converting units.',
    binary: false,
  },
  {
    format: 'pytorch',
    name: 'PyTorch',
    description: 'A torch.nn.Module running the circuit as a recurrent spiking layer, trainable.',
    binary: false,
  },
  {
    format: 'python',
    name: 'NumPy reference',
    description: 'A dependency-free integration loop: the ground truth the other exports match.',
    binary: false,
  },
  {
    format: 'onnx',
    name: 'ONNX graph',
    description: 'An ONNX protobuf of the binned weight matrices and an unrolled LIF cell.',
    binary: true,
  },
];

const PREVIEW_LINES = 40;
const HEX_ROWS = 14;
const BYTES_PER_ROW = 16;

type ProbeStatus = 'measuring' | 'ready' | 'failed';

interface FormatProbe {
  status: ProbeStatus;
  bytes: number;
  lines: number;
  preview: string;
  truncated: boolean;
  binary: boolean;
  filename: string;
  error: string;
}

const MEASURING: FormatProbe = {
  status: 'measuring',
  bytes: 0,
  lines: 0,
  preview: '',
  truncated: false,
  binary: false,
  filename: '',
  error: '',
};

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Decimal units, because a file manager reports these files the same way. */
function byteSize(value: number): string {
  if (!Number.isFinite(value)) return '—';
  if (value < 1000) return `${value} B`;
  if (value < 1_000_000) return `${(value / 1000).toFixed(1)} kB`;
  return `${(value / 1_000_000).toFixed(2)} MB`;
}

/** Classic offset/hex/ASCII dump, which is the only readable preview of a protobuf. */
function hexPreview(bytes: Uint8Array): string {
  const limit = Math.min(bytes.length, HEX_ROWS * BYTES_PER_ROW);
  const rows: string[] = [];
  for (let offset = 0; offset < limit; offset += BYTES_PER_ROW) {
    const end = Math.min(offset + BYTES_PER_ROW, limit);
    let hex = '';
    let ascii = '';
    for (let i = 0; i < BYTES_PER_ROW; i += 1) {
      const index = offset + i;
      if (index < end) {
        const byte = bytes[index];
        hex += byte.toString(16).padStart(2, '0');
        ascii += byte >= 0x20 && byte < 0x7f ? String.fromCharCode(byte) : '.';
      } else {
        hex += '  ';
      }
      hex += i === 7 ? '  ' : ' ';
    }
    rows.push(`${offset.toString(16).padStart(8, '0')}  ${hex} |${ascii}`);
  }
  return rows.join('\n');
}

function probeFrom(result: ExportResult): FormatProbe {
  if (typeof result.content === 'string') {
    const lines = result.content.split('\n');
    return {
      status: 'ready',
      bytes: new TextEncoder().encode(result.content).length,
      lines: lines.length,
      preview: lines.slice(0, PREVIEW_LINES).join('\n'),
      truncated: lines.length > PREVIEW_LINES,
      binary: false,
      filename: result.filename,
      error: '',
    };
  }
  return {
    status: 'ready',
    bytes: result.content.byteLength,
    lines: 0,
    preview: hexPreview(result.content),
    truncated: result.content.byteLength > HEX_ROWS * BYTES_PER_ROW,
    binary: true,
    filename: result.filename,
    error: '',
  };
}

/* ------------------------------------------------------------------ colour -- */

const RIBBON_SWATCHES = 18;

/**
 * The identity colours of a document's cells, in the same hues the scene draws
 * them in. Sampled with a stride rather than truncated so the strip is a
 * fingerprint of the whole circuit, not of its first eighteen neurons.
 */
function CellRibbon({ neurons, className }: { neurons: readonly Neuron[]; className?: string }) {
  const colors = useMemo(() => {
    const total = neurons.length;
    if (total === 0) return [];
    const count = Math.min(RIBBON_SWATCHES, total);
    const stride = total / count;
    const out: string[] = [];
    for (let i = 0; i < count; i += 1) {
      out.push(identityColorHex(neurons[Math.floor(i * stride)].morphology.seed));
    }
    return out;
  }, [neurons]);

  if (colors.length === 0) {
    return <span className={cn('h-2.5 w-1 rounded-[1px] bg-white/10', className)} aria-hidden />;
  }

  return (
    <span
      className={cn('flex shrink-0 items-center gap-px', className)}
      role="img"
      aria-label={`${colors.length} cell colours`}
    >
      {colors.map((color, index) => (
        <span
          key={`${color}-${index}`}
          className="h-2.5 w-1 rounded-[1px]"
          style={{ background: color, boxShadow: `0 0 5px ${color}55` }}
        />
      ))}
    </span>
  );
}

/* ------------------------------------------------------------- confirmation -- */

interface Confirmation {
  title: string;
  description: string;
  confirmLabel: string;
  danger: boolean;
  prune: boolean;
  run: () => Promise<void>;
}

const DEFAULT_PRUNE_KEEP = 20;

/* -------------------------------------------------------------------- panel -- */

type LibraryTab = 'export' | 'snapshots' | 'circuits';

/**
 * Export, version history and the document store.
 *
 * Everything here talks to IndexedDB, which can fail outright (private windows,
 * a blocked upgrade, a quota refusal). Every read therefore ends in a settled
 * state — rows, or an error with a retry — and never in a spinner that outlives
 * the request that started it.
 */
export function LibraryPanel() {
  const open = useEditor((s) => s.libraryOpen);
  const circuit = useEditor((s) => s.circuit);
  const togglePanel = useEditor((s) => s.togglePanel);
  const loadIntoEditor = useEditor((s) => s.loadCircuit);

  const [tab, setTab] = useState<LibraryTab>('export');
  const [now, setNow] = useState(() => Date.now());

  const close = useCallback(() => togglePanel('library', false), [togglePanel]);

  useEffect(() => {
    if (!open) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 15_000);
    return () => clearInterval(timer);
  }, [open]);

  /* -- export ------------------------------------------------------------- */

  const [probes, setProbes] = useState<Partial<Record<ExportFormat, FormatProbe>>>({});
  const [expanded, setExpanded] = useState<ExportFormat | null>(null);
  const [copied, setCopied] = useState<ExportFormat | null>(null);
  const requested = useRef(new Set<ExportFormat>());
  const measureTimers = useRef(new Set<ReturnType<typeof setTimeout>>());
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelMeasurements = useCallback(() => {
    for (const timer of measureTimers.current) clearTimeout(timer);
    measureTimers.current.clear();
  }, []);

  useEffect(
    () => () => {
      for (const timer of measureTimers.current) clearTimeout(timer);
      measureTimers.current.clear();
      if (copyTimer.current !== null) clearTimeout(copyTimer.current);
    },
    [],
  );

  // A measurement describes one version of the document, so editing the circuit
  // invalidates every cached size and preview. Deferred measurements still in
  // flight are cancelled along with them: each closes over the circuit it was
  // scheduled for, so letting one land would print the previous version's size
  // and preview against the current document.
  useEffect(() => {
    cancelMeasurements();
    requested.current.clear();
    setProbes({});
    setCopied(null);
  }, [circuit, cancelMeasurements]);

  const measure = useCallback(
    (format: ExportFormat) => {
      if (requested.current.has(format)) return;
      requested.current.add(format);
      setProbes((prev) => ({ ...prev, [format]: MEASURING }));

      // Serialising a large circuit is not cheap, so the row is allowed to paint
      // its measuring state before the main thread is taken.
      const timer = setTimeout(() => {
        measureTimers.current.delete(timer);
        try {
          // Rendered outside the updater: React may invoke an updater twice, and
          // serialising the document twice is exactly what this defers.
          const probe = probeFrom(exportCircuit(circuit, format));
          setProbes((prev) => ({ ...prev, [format]: probe }));
        } catch (error) {
          const message = messageOf(error);
          setProbes((prev) => ({
            ...prev,
            [format]: { ...MEASURING, status: 'failed', error: message },
          }));
          pushToast({ tone: 'danger', title: `${format} export failed`, description: message });
        }
      }, 0);
      measureTimers.current.add(timer);
    },
    [circuit],
  );

  /** A failed measurement is cached like any other, so retrying has to evict it. */
  const remeasure = useCallback(
    (format: ExportFormat) => {
      requested.current.delete(format);
      measure(format);
    },
    [measure],
  );

  useEffect(() => {
    if (expanded) measure(expanded);
  }, [expanded, measure]);

  const download = useCallback(
    (descriptor: FormatDescriptor) => {
      try {
        const result = exportCircuit(circuit, descriptor.format);
        downloadExport(result);
        pushToast({ tone: 'success', title: 'Exported', description: result.filename });
      } catch (error) {
        pushToast({
          tone: 'danger',
          title: `${descriptor.name} export failed`,
          description: messageOf(error),
        });
      }
    },
    [circuit],
  );

  const copy = useCallback(
    async (format: ExportFormat) => {
      try {
        if (typeof navigator === 'undefined' || !navigator.clipboard) {
          throw new Error('The clipboard is only available over HTTPS or on localhost');
        }
        const result = exportCircuit(circuit, format);
        if (typeof result.content !== 'string') {
          throw new Error('This format is binary; download it instead');
        }
        await navigator.clipboard.writeText(result.content);
        setCopied(format);
        // Copying again before the tick fades restarts it rather than stacking a
        // second timer that would clear the newer confirmation early.
        if (copyTimer.current !== null) clearTimeout(copyTimer.current);
        copyTimer.current = setTimeout(() => {
          copyTimer.current = null;
          setCopied((current) => (current === format ? null : current));
        }, 1600);
      } catch (error) {
        pushToast({ tone: 'danger', title: 'Copy failed', description: messageOf(error) });
      }
    },
    [circuit],
  );

  /* -- snapshots ---------------------------------------------------------- */

  const circuitId: CircuitId = circuit.id;
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [snapshotsLoading, setSnapshotsLoading] = useState(false);
  const [snapshotsError, setSnapshotsError] = useState<string | null>(null);
  const [snapshotLabel, setSnapshotLabel] = useState('');
  const [taking, setTaking] = useState(false);
  const snapshotGeneration = useRef(0);

  /**
   * Returns the rows it committed, or null when the read failed or a newer read
   * superseded it — so a caller that needs to describe the result (how many
   * snapshots a trim removed) reads the same list that reached the screen
   * instead of issuing a second, unguarded query.
   */
  const refreshSnapshots = useCallback(async (): Promise<Snapshot[] | null> => {
    snapshotGeneration.current += 1;
    const generation = snapshotGeneration.current;
    setSnapshotsLoading(true);
    try {
      const rows = await listSnapshots(circuitId);
      if (generation !== snapshotGeneration.current) return null;
      setSnapshots(rows);
      setSnapshotsError(null);
      return rows;
    } catch (error) {
      if (generation !== snapshotGeneration.current) return null;
      const message = messageOf(error);
      setSnapshots([]);
      setSnapshotsError(message);
      pushToast({ tone: 'danger', title: 'Version history unavailable', description: message });
      return null;
    } finally {
      if (generation === snapshotGeneration.current) setSnapshotsLoading(false);
    }
  }, [circuitId]);

  const takeSnapshot = useCallback(async () => {
    if (taking) return;
    setTaking(true);
    const label =
      snapshotLabel.trim() ||
      `Manual ${new Date().toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })}`;
    try {
      const snapshot = await createSnapshot(circuit, label, false);
      setSnapshotLabel('');
      pushToast({
        tone: 'success',
        title: 'Snapshot taken',
        description: `${snapshot.label} — ${grouped(snapshot.neuronCount)} cells`,
      });
      await refreshSnapshots();
    } catch (error) {
      pushToast({ tone: 'danger', title: 'Snapshot failed', description: messageOf(error) });
    } finally {
      setTaking(false);
    }
  }, [circuit, refreshSnapshots, snapshotLabel, taking]);

  /* -- circuits ----------------------------------------------------------- */

  const [circuits, setCircuits] = useState<Circuit[]>([]);
  const [circuitsLoading, setCircuitsLoading] = useState(false);
  const [circuitsError, setCircuitsError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const circuitGeneration = useRef(0);

  const refreshCircuits = useCallback(async () => {
    circuitGeneration.current += 1;
    const generation = circuitGeneration.current;
    setCircuitsLoading(true);
    try {
      const rows = await listCircuits();
      if (generation !== circuitGeneration.current) return;
      setCircuits(rows);
      setCircuitsError(null);
    } catch (error) {
      if (generation !== circuitGeneration.current) return;
      const message = messageOf(error);
      setCircuits([]);
      setCircuitsError(message);
      pushToast({ tone: 'danger', title: 'Document store unavailable', description: message });
    } finally {
      if (generation === circuitGeneration.current) setCircuitsLoading(false);
    }
  }, []);

  const saveCurrent = useCallback(async () => {
    if (saving) return;
    setSaving(true);
    try {
      await saveCircuit(circuit);
      pushToast({ tone: 'success', title: 'Saved', description: circuit.name });
      await refreshCircuits();
    } catch (error) {
      pushToast({ tone: 'danger', title: 'Save failed', description: messageOf(error) });
    } finally {
      setSaving(false);
    }
  }, [circuit, refreshCircuits, saving]);

  useEffect(() => {
    if (!open) return;
    if (tab === 'snapshots') void refreshSnapshots();
    if (tab === 'circuits') void refreshCircuits();
  }, [open, tab, refreshSnapshots, refreshCircuits]);

  /* -- destructive actions ------------------------------------------------ */

  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [pruneKeep, setPruneKeep] = useState(DEFAULT_PRUNE_KEEP);
  const pruneKeepRef = useRef(DEFAULT_PRUNE_KEEP);

  const changePruneKeep = useCallback((value: number) => {
    const rounded = Math.max(1, Math.round(value));
    pruneKeepRef.current = rounded;
    setPruneKeep(rounded);
  }, []);

  const runConfirmation = useCallback(async () => {
    if (!confirmation || confirming) return;
    setConfirming(true);
    try {
      await confirmation.run();
    } catch (error) {
      pushToast({
        tone: 'danger',
        title: confirmation.confirmLabel + ' failed',
        description: messageOf(error),
      });
    } finally {
      setConfirming(false);
      setConfirmation(null);
    }
  }, [confirmation, confirming]);

  const askRestore = useCallback(
    (snapshot: Snapshot) => {
      setConfirmation({
        title: `Restore “${snapshot.label}”?`,
        description:
          'The open document is replaced by this version. Anything changed since it was taken is discarded.',
        confirmLabel: 'Restore',
        danger: true,
        prune: false,
        run: async () => {
          const restored = await restoreSnapshot(snapshot.id);
          if (!restored) {
            pushToast({
              tone: 'danger',
              title: 'Snapshot could not be read',
              description: 'The stored version is missing or was rejected by validation.',
            });
            await refreshSnapshots();
            return;
          }
          loadIntoEditor(restored);
          pushToast({
            tone: 'success',
            title: 'Restored',
            description: `${snapshot.label} — ${grouped(restored.neurons.length)} cells, ${grouped(
              restored.synapses.length,
            )} synapses`,
          });
          close();
        },
      });
    },
    [close, loadIntoEditor, refreshSnapshots],
  );

  const askDeleteSnapshot = useCallback(
    (snapshot: Snapshot) => {
      setConfirmation({
        title: `Delete “${snapshot.label}”?`,
        description: 'This version is removed permanently. The open document is not affected.',
        confirmLabel: 'Delete',
        danger: true,
        prune: false,
        run: async () => {
          // `pruneSnapshots` trims by age; removing one chosen version is a
          // direct table delete against the same store.
          await db.snapshots.delete(snapshot.id);
          pushToast({ tone: 'neutral', title: 'Snapshot deleted', description: snapshot.label });
          await refreshSnapshots();
        },
      });
    },
    [refreshSnapshots],
  );

  const askPrune = useCallback(() => {
    changePruneKeep(DEFAULT_PRUNE_KEEP);
    setConfirmation({
      title: 'Trim version history',
      description:
        'Automatic snapshots are dropped before manual ones, and the oldest go first.',
      confirmLabel: 'Trim',
      danger: true,
      prune: true,
      run: async () => {
        const before = snapshots.length;
        await pruneSnapshots(circuitId, pruneKeepRef.current);
        // Re-read through the guarded refresh so this cannot be overwritten by a
        // list request that was already in flight when the trim started.
        const rows = await refreshSnapshots();
        // Null means the re-read failed or was superseded; it has already said so.
        if (!rows) return;
        const removed = Math.max(0, before - rows.length);
        pushToast({
          tone: removed > 0 ? 'success' : 'neutral',
          title: removed > 0 ? `Removed ${grouped(removed)} snapshots` : 'Nothing to trim',
          description: `${grouped(rows.length)} kept`,
        });
      },
    });
  }, [changePruneKeep, circuitId, refreshSnapshots, snapshots.length]);

  const askOpenCircuit = useCallback(
    (stored: Circuit) => {
      setConfirmation({
        title: `Open “${stored.name}”?`,
        description:
          'The open document is replaced. Anything not yet written to the store is lost.',
        confirmLabel: 'Open',
        danger: false,
        prune: false,
        run: async () => {
          const loaded = await readStoredCircuit(stored.id);
          if (!loaded) {
            pushToast({
              tone: 'danger',
              title: 'Document could not be read',
              description: 'The stored record is missing or was rejected by validation.',
            });
            await refreshCircuits();
            return;
          }
          loadIntoEditor(loaded);
          pushToast({
            tone: 'success',
            title: 'Opened',
            description: `${loaded.name} — ${grouped(loaded.neurons.length)} cells`,
          });
          close();
        },
      });
    },
    [close, loadIntoEditor, refreshCircuits],
  );

  const askDeleteCircuit = useCallback(
    (stored: Circuit) => {
      setConfirmation({
        title: `Delete “${stored.name}”?`,
        description: 'The document and its entire version history are removed permanently.',
        confirmLabel: 'Delete',
        danger: true,
        prune: false,
        run: async () => {
          await deleteCircuit(stored.id);
          pushToast({ tone: 'neutral', title: 'Document deleted', description: stored.name });
          await refreshCircuits();
        },
      });
    },
    [refreshCircuits],
  );

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      // While a confirmation is up it owns Escape; dismissing it must not also
      // close the panel underneath.
      if (event.key !== 'Escape' || confirmation) return;
      event.stopPropagation();
      close();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, close, confirmation]);

  if (!open) return null;

  return (
    <>
      {/* The panel mounts inside a pointer-events-none overlay, so the scrim has
          to opt back in or clicks would fall through to the canvas. */}
      <div
        className="pointer-events-auto fixed inset-0 z-40 bg-bg/60 backdrop-blur-[2px]"
        onClick={close}
        aria-hidden
      />
      <Panel
        raised
        role="dialog"
        aria-modal="false"
        aria-label="Library"
        className={cn(
          'pointer-events-auto fixed top-1/2 left-1/2 z-50 -translate-x-1/2 -translate-y-1/2',
          'h-[min(640px,calc(100dvh-5rem))] w-[min(820px,calc(100vw-2rem))] flex-col',
        )}
      >
        <PanelHeader
          title="Library"
          subtitle={`${circuit.name} — ${grouped(circuit.neurons.length)} cells, ${grouped(
            circuit.synapses.length,
          )} synapses`}
          icon={<Library />}
          actions={
            <IconButton label="Close library" size="sm" onClick={close}>
              <X />
            </IconButton>
          }
        />

        <Tabs
          value={tab}
          onValueChange={(value) => setTab(value as LibraryTab)}
          className="flex min-h-0 flex-1 flex-col"
        >
          <TabsList className="px-3">
            <Tab value="export" icon={<Download size={12} />} badge={FORMATS.length}>
              Export
            </Tab>
            <Tab value="snapshots" icon={<ClockArrowLeft size={12} />}>
              Snapshots
            </Tab>
            <Tab value="circuits" icon={<Database size={12} />}>
              Circuits
            </Tab>
          </TabsList>

          {/* The inner wrapper carries the flex layout: TabPanel hides itself
              with `display: none`, which a `flex` on the same element fights. */}
          <TabPanel value="export">
            <div className="flex h-full min-h-0 flex-col">
              <div className="flex shrink-0 items-center gap-2 border-b border-hairline px-3 py-1.5">
                <CellRibbon neurons={circuit.neurons} />
                <span className="nf-numeric truncate text-[10px] tracking-[0.02em] text-ink-faint">
                  {grouped(circuit.neurons.length)} cells · size is measured on hover, expand for a
                  preview
                </span>
              </div>
              <ScrollArea className="min-h-0 flex-1">
                <ul>
                  {FORMATS.map((descriptor) => {
                    const probe = probes[descriptor.format];
                    const isOpen = expanded === descriptor.format;
                    return (
                      <li
                        key={descriptor.format}
                        className="border-b border-hairline last:border-b-0"
                        onPointerEnter={() => measure(descriptor.format)}
                      >
                        <div className="flex items-center gap-2 px-3 py-1.5 transition-colors hover:bg-white/[0.025]">
                          <button
                            type="button"
                            aria-expanded={isOpen}
                            onFocus={() => measure(descriptor.format)}
                            onClick={() => setExpanded(isOpen ? null : descriptor.format)}
                            className="flex min-w-0 flex-1 items-center gap-2 rounded-control text-left"
                          >
                            <ChevronRight
                              aria-hidden
                              className={cn(
                                'size-3 shrink-0 text-ink-faint transition-transform duration-150',
                                isOpen && 'rotate-90',
                              )}
                            />
                            <span className="w-[104px] shrink-0 truncate text-[11.5px] text-ink">
                              {descriptor.name}
                            </span>
                            <span className="min-w-0 flex-1 truncate text-[10.5px] text-ink-faint">
                              {descriptor.description}
                            </span>
                          </button>
                          <span className="nf-numeric w-[58px] shrink-0 text-right text-[10.5px] text-ink-muted">
                            {!probe ? (
                              <span className="text-ink-faint">—</span>
                            ) : probe.status === 'measuring' ? (
                              <Spinner size={11} label="Measuring export" />
                            ) : probe.status === 'failed' ? (
                              <TriangleAlert
                                className="inline size-3 text-danger"
                                aria-label="Export failed"
                              />
                            ) : (
                              byteSize(probe.bytes)
                            )}
                          </span>
                          <Tooltip content={`Download ${descriptor.name}`}>
                            <IconButton
                              label={`Download ${descriptor.name}`}
                              size="sm"
                              onClick={() => download(descriptor)}
                            >
                              <Download />
                            </IconButton>
                          </Tooltip>
                        </div>

                        {isOpen ? (
                          <div className="border-t border-hairline bg-black/25 px-3 py-2">
                            {!probe || probe.status === 'measuring' ? (
                              <div
                                role="status"
                                className="flex items-center gap-2 py-3 text-[11px] text-ink-faint"
                              >
                                <Spinner size={12} />
                                Rendering {descriptor.name}…
                              </div>
                            ) : probe.status === 'failed' ? (
                              <div className="flex items-start justify-between gap-3 py-2">
                                <p className="flex min-w-0 items-start gap-1.5 text-[11px] leading-snug text-danger">
                                  <TriangleAlert className="mt-px size-3 shrink-0" aria-hidden />
                                  {probe.error}
                                </p>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  icon={<RefreshCw />}
                                  onClick={() => remeasure(descriptor.format)}
                                >
                                  Try again
                                </Button>
                              </div>
                            ) : (
                              <>
                                <div className="mb-1.5 flex items-center justify-between gap-2">
                                  <span className="nf-numeric truncate text-[10px] text-ink-faint">
                                    {probe.filename} · {byteSize(probe.bytes)}
                                    {probe.binary
                                      ? ' · binary'
                                      : ` · ${grouped(probe.lines)} lines`}
                                  </span>
                                  {probe.binary ? (
                                    <span className="nf-numeric shrink-0 text-[10px] text-ink-faint">
                                      hex dump
                                    </span>
                                  ) : (
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      icon={
                                        copied === descriptor.format ? (
                                          <Check className="text-success" />
                                        ) : (
                                          <Copy />
                                        )
                                      }
                                      onClick={() => void copy(descriptor.format)}
                                    >
                                      {copied === descriptor.format ? 'Copied' : 'Copy'}
                                    </Button>
                                  )}
                                </div>
                                <pre className="max-h-[260px] overflow-auto rounded-control border border-hairline bg-bg/70 p-2 font-mono text-[10.5px] leading-[1.5] text-ink-muted">
                                  {probe.preview}
                                </pre>
                                {probe.truncated ? (
                                  <p className="mt-1 text-[10px] text-ink-faint">
                                    {probe.binary
                                      ? `First ${HEX_ROWS * BYTES_PER_ROW} bytes of ${byteSize(probe.bytes)}.`
                                      : `First ${PREVIEW_LINES} of ${grouped(probe.lines)} lines.`}
                                  </p>
                                ) : null}
                              </>
                            )}
                          </div>
                        ) : null}
                      </li>
                    );
                  })}
                </ul>
              </ScrollArea>
            </div>
          </TabPanel>

          <TabPanel value="snapshots">
            <div className="flex h-full min-h-0 flex-col">
              <div className="flex shrink-0 items-center gap-2 border-b border-hairline px-3 py-2">
                <input
                  value={snapshotLabel}
                  onChange={(event) => setSnapshotLabel(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      void takeSnapshot();
                    }
                  }}
                  placeholder="Label this version…"
                  aria-label="Snapshot label"
                  className="h-7 min-w-0 flex-1 rounded-control border border-hairline bg-bg px-2 text-[11.5px] text-ink outline-none placeholder:text-ink-faint focus-visible:border-accent"
                />
                <Button
                  size="sm"
                  variant="primary"
                  icon={<Camera />}
                  loading={taking}
                  onClick={() => void takeSnapshot()}
                >
                  Snapshot
                </Button>
                <Tooltip content="Trim version history">
                  <IconButton
                    label="Trim version history"
                    size="sm"
                    disabled={snapshots.length === 0}
                    onClick={askPrune}
                  >
                    <Scissors />
                  </IconButton>
                </Tooltip>
              </div>

              <ScrollArea className="min-h-0 flex-1">
                {snapshotsLoading && snapshots.length === 0 ? (
                  <LoadingRows label="Reading version history…" />
                ) : snapshotsError ? (
                  <FailureState
                    title="Version history unavailable"
                    description={snapshotsError}
                    onRetry={() => void refreshSnapshots()}
                  />
                ) : snapshots.length === 0 ? (
                  <EmptyState
                    icon={<ClockArrowLeft />}
                    title="No snapshots yet"
                    description="Take one before a risky edit; restoring is a single click and the current document is never overwritten silently."
                  />
                ) : (
                  <ul>
                    {snapshots.map((snapshot) => (
                      <li
                        key={snapshot.id}
                        className="flex items-center gap-2 border-b border-hairline px-3 py-1.5 transition-colors last:border-b-0 hover:bg-white/[0.025]"
                      >
                        <CellRibbon neurons={snapshot.circuit.neurons} />
                        <button
                          type="button"
                          onClick={() => askRestore(snapshot)}
                          className="flex min-w-0 flex-1 flex-col items-start gap-0.5 rounded-control text-left"
                        >
                          <span className="flex w-full min-w-0 items-center gap-1.5">
                            <span className="truncate text-[11.5px] text-ink">{snapshot.label}</span>
                            {snapshot.automatic ? (
                              <Badge variant="outline" size="sm">
                                auto
                              </Badge>
                            ) : null}
                          </span>
                          <span className="nf-numeric truncate text-[10px] text-ink-faint">
                            {grouped(snapshot.neuronCount)} cells · {grouped(snapshot.synapseCount)}{' '}
                            synapses · {since(snapshot.createdAt, now)}
                          </span>
                        </button>
                        <Tooltip content="Restore this version">
                          <IconButton
                            label={`Restore ${snapshot.label}`}
                            size="sm"
                            onClick={() => askRestore(snapshot)}
                          >
                            <RotateCcw />
                          </IconButton>
                        </Tooltip>
                        <Tooltip content="Delete this version">
                          <IconButton
                            label={`Delete ${snapshot.label}`}
                            size="sm"
                            variant="ghost"
                            onClick={() => askDeleteSnapshot(snapshot)}
                          >
                            <Trash2 />
                          </IconButton>
                        </Tooltip>
                      </li>
                    ))}
                  </ul>
                )}
              </ScrollArea>
            </div>
          </TabPanel>

          <TabPanel value="circuits">
            <div className="flex h-full min-h-0 flex-col">
              <div className="flex shrink-0 items-center gap-2 border-b border-hairline px-3 py-2">
                <span className="min-w-0 flex-1 truncate text-[10.5px] text-ink-faint">
                  Documents in this browser. Autosave keeps the open one current.
                </span>
                <Button
                  size="sm"
                  variant="primary"
                  icon={<Save />}
                  loading={saving}
                  onClick={() => void saveCurrent()}
                >
                  Save current
                </Button>
                <Tooltip content="Reload the list">
                  <IconButton
                    label="Reload the document list"
                    size="sm"
                    onClick={() => void refreshCircuits()}
                  >
                    <RefreshCw />
                  </IconButton>
                </Tooltip>
              </div>

              <ScrollArea className="min-h-0 flex-1">
                {circuitsLoading && circuits.length === 0 ? (
                  <LoadingRows label="Reading the document store…" />
                ) : circuitsError ? (
                  <FailureState
                    title="Document store unavailable"
                    description={circuitsError}
                    onRetry={() => void refreshCircuits()}
                  />
                ) : circuits.length === 0 ? (
                  <EmptyState
                    icon={<Database />}
                    title="Nothing saved yet"
                    description="Save the open circuit to keep it in this browser's IndexedDB."
                  />
                ) : (
                  <ul>
                    {circuits.map((stored) => {
                      const isCurrent = stored.id === circuitId;
                      return (
                        <li
                          key={stored.id}
                          className="flex items-center gap-2 border-b border-hairline px-3 py-1.5 transition-colors last:border-b-0 hover:bg-white/[0.025]"
                        >
                          <CellRibbon neurons={stored.neurons} />
                          <button
                            type="button"
                            disabled={isCurrent}
                            onClick={() => askOpenCircuit(stored)}
                            className="flex min-w-0 flex-1 flex-col items-start gap-0.5 rounded-control text-left disabled:cursor-default"
                          >
                            <span className="flex w-full min-w-0 items-center gap-1.5">
                              <span className="truncate text-[11.5px] text-ink">{stored.name}</span>
                              {isCurrent ? (
                                <Badge variant="accent" size="sm" dot>
                                  open
                                </Badge>
                              ) : null}
                              {stored.tags.slice(0, 2).map((tag) => (
                                <Badge key={tag} variant="outline" size="sm">
                                  {tag}
                                </Badge>
                              ))}
                            </span>
                            <span className="nf-numeric truncate text-[10px] text-ink-faint">
                              {grouped(stored.neurons.length)} cells · {grouped(
                                stored.synapses.length,
                              )}{' '}
                              synapses · saved {since(stored.updatedAt, now)}
                            </span>
                          </button>
                          <Tooltip content={isCurrent ? 'Already open' : 'Open this document'}>
                            <IconButton
                              label={`Open ${stored.name}`}
                              size="sm"
                              disabled={isCurrent}
                              onClick={() => askOpenCircuit(stored)}
                            >
                              <FolderOpen />
                            </IconButton>
                          </Tooltip>
                          <Tooltip
                            content={isCurrent ? 'Close it before deleting' : 'Delete this document'}
                          >
                            <IconButton
                              label={`Delete ${stored.name}`}
                              size="sm"
                              variant="ghost"
                              disabled={isCurrent}
                              onClick={() => askDeleteCircuit(stored)}
                            >
                              <Trash2 />
                            </IconButton>
                          </Tooltip>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </ScrollArea>
            </div>
          </TabPanel>
        </Tabs>
      </Panel>

      {confirmation ? (
        <Dialog
          open
          onOpenChange={(next) => {
            if (!next && !confirming) setConfirmation(null);
          }}
          title={confirmation.title}
          description={confirmation.description}
          size="sm"
          footer={
            <>
              <Button
                size="sm"
                variant="ghost"
                disabled={confirming}
                onClick={() => setConfirmation(null)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                variant={confirmation.danger ? 'danger' : 'primary'}
                loading={confirming}
                onClick={() => void runConfirmation()}
              >
                {confirmation.confirmLabel}
              </Button>
            </>
          }
        >
          {confirmation.prune ? (
            <Field
              label="Snapshots to keep"
              description={`History currently holds ${grouped(snapshots.length)}.`}
            >
              <NumberField
                value={pruneKeep}
                onChange={changePruneKeep}
                min={1}
                max={200}
                step={1}
                precision={0}
              />
            </Field>
          ) : null}
        </Dialog>
      ) : null}
    </>
  );
}

/* ------------------------------------------------------------------ states -- */

function LoadingRows({ label }: { label: string }) {
  return (
    <div
      role="status"
      className="flex items-center justify-center gap-2 py-10 text-[11px] text-ink-faint"
    >
      <Spinner size={13} />
      {label}
    </div>
  );
}

function FailureState({
  title,
  description,
  onRetry,
}: {
  title: string;
  description: string;
  onRetry: () => void;
}) {
  return (
    <EmptyState
      icon={<TriangleAlert className="text-warning" />}
      title={title}
      description={description}
      action={
        <Button size="sm" icon={<RefreshCw />} onClick={onRetry}>
          Try again
        </Button>
      }
    />
  );
}
