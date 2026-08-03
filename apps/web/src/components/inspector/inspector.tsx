'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, GitBranch, Sliders, Zap } from 'lucide-react';
import {
  Badge,
  EmptyState,
  Field,
  Meter,
  NumberField,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  Select,
  SelectItem,
  Separator,
  Sparkline,
  Switch,
  Tab,
  TabPanel,
  Tabs,
  TabsList,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import {
  NEURON_MODEL_KINDS,
  NEURON_MODEL_LABELS,
  RECEPTOR_LABELS,
  defaultParams,
  voltageRange,
  identityColorHex,
} from '@neuroforge/shared';
import type { NeuronId, NeuronModelKind } from '@neuroforge/shared';

import { getEngine, getFingerprints } from '@/lib/runtime';
import { fixed, grouped, millivolts } from '@/lib/format';
import { useLiveNeuron } from '@/hooks/use-live-neuron';
import { similarCells } from '@/lib/similarity';
import { PARAM_FIELDS, readParam, writeParam } from './param-fields';

const EMPTY_TRACE = new Float32Array(0);

/**
 * The neuron inspector.
 *
 * Every control writes through the editor's transaction system, so a parameter
 * drag is undoable as a single step rather than as one entry per pointer move,
 * and the engine picks the change up from the rebuilt buffers on the next frame.
 */
export function Inspector() {
  const open = useEditor((s) => s.inspectorOpen);
  const selection = useEditor((s) => s.selection);
  const circuit = useEditor((s) => s.circuit);
  const updateNeuron = useEditor((s) => s.updateNeuron);
  const select = useEditor((s) => s.select);

  const selectedId: NeuronId | null = selection.length > 0 ? selection[0] : null;
  const neuron = useMemo(
    () => (selectedId ? (circuit.neurons.find((n) => n.id === selectedId) ?? null) : null),
    [circuit.neurons, selectedId],
  );

  const slot = useMemo(
    () => (selectedId ? getEngine().slotOf(selectedId) : -1),
    // The slot mapping is rebuilt whenever the neuron array identity changes.
    [selectedId, circuit.neurons],
  );

  const live = useLiveNeuron(slot >= 0 ? slot : null);

  const connections = useMemo(() => {
    if (!selectedId) return { incoming: [], outgoing: [] };
    const incoming = circuit.synapses.filter((s) => s.target === selectedId);
    const outgoing = circuit.synapses.filter((s) => s.source === selectedId);
    return { incoming, outgoing };
  }, [circuit.synapses, selectedId]);

  // Cells whose connectivity fingerprint points the same way as this one's.
  // This is how connectomics identifies a cell type without a label, and it is
  // the question the inspector is otherwise unable to answer.
  const similar = useMemo(() => {
    if (slot < 0) return [];
    const prints = getFingerprints(circuit.populations.length);
    return similarCells(prints, slot, 8);
  }, [slot, circuit.populations.length, circuit.synapses]);

  const [labelDraft, setLabelDraft] = useState('');
  useEffect(() => {
    setLabelDraft(neuron?.label ?? '');
  }, [neuron?.id, neuron?.label]);

  const commitLabel = useCallback(() => {
    if (!neuron) return;
    const next = labelDraft.trim();
    if (next === neuron.label) return;
    updateNeuron(neuron.id, { label: next });
  }, [neuron, labelDraft, updateNeuron]);

  const changeModel = useCallback(
    (kind: string) => {
      if (!neuron) return;
      updateNeuron(neuron.id, { params: defaultParams(kind as NeuronModelKind) });
    },
    [neuron, updateNeuron],
  );

  const changeParam = useCallback(
    (key: string, value: number) => {
      if (!neuron) return;
      updateNeuron(neuron.id, { params: writeParam(neuron.params, key, value) });
    },
    [neuron, updateNeuron],
  );

  if (!open) return null;

  if (!neuron) {
    return (
      <Panel className="pointer-events-auto absolute top-3 right-3 bottom-3 w-[340px]">
        <PanelHeader title="Inspector" />
        <EmptyState
          icon={<Activity />}
          title="Nothing selected"
          description="Click a neuron in the scene to inspect its membrane state, parameters and connections."
        />
      </Panel>
    );
  }

  const fields = PARAM_FIELDS[neuron.params.kind];
  const range = voltageRange(neuron.params.kind);
  const normalisedV = live
    ? Math.min(1, Math.max(0, (live.voltage - range.min) / (range.max - range.min)))
    : 0;
  const isHH = neuron.params.kind === 'hodgkin-huxley';

  return (
    <Panel className="pointer-events-auto absolute top-3 right-3 bottom-3 flex w-[340px] flex-col">
      <PanelHeader
        title={neuron.label || 'Neuron'}
        subtitle={NEURON_MODEL_LABELS[neuron.params.kind]}
        actions={
          <Badge variant={neuron.polarity === 'inhibitory' ? 'secondary' : 'accent'} dot>
            {neuron.polarity === 'inhibitory' ? 'Inhibitory' : 'Excitatory'}
          </Badge>
        }
      />

      <Tabs defaultValue="state" className="flex min-h-0 flex-1 flex-col">
        <TabsList className="px-3">
          <Tab value="state" icon={<Activity />}>
            State
          </Tab>
          <Tab value="params" icon={<Sliders />}>
            Parameters
          </Tab>
          <Tab value="wiring" icon={<GitBranch />}>
            Wiring
          </Tab>
        </TabsList>

        <ScrollArea className="min-h-0 flex-1">
          <TabPanel value="state">
            <PanelSection label="Membrane" flush>
              <div className="mb-2 flex items-baseline justify-between">
                <span className="nf-numeric text-2xl text-accent">
                  {live ? millivolts(live.voltage) : '—'}
                </span>
                <span className="nf-numeric text-[11px] text-ink-faint">
                  {live ? `${fixed(live.rate, 1)} Hz` : '—'}
                </span>
              </div>
              <Sparkline
                values={live?.trace ?? EMPTY_TRACE}
                count={live?.traceCount ?? 0}
                min={range.min}
                max={range.max}
                height={72}
                color="#4FD1FF"
                fill
                endpoint
                label="Membrane potential over time"
                threshold={
                  neuron.params.kind === 'lif' ? readParam(neuron.params, 'vThresh') : undefined
                }
              />
              <Meter value={normalisedV} label="Membrane" className="mt-3" />
            </PanelSection>

            <Separator />

            <PanelSection label="Signals">
              <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-[11px]">
                <Readout label="Synaptic current" value={live ? `${fixed(live.synapticCurrent, 1)} pA` : '—'} />
                <Readout label="Injected current" value={live ? `${fixed(live.externalCurrent, 1)} pA` : '—'} />
                <Readout label="Adaptation" value={live ? fixed(live.adaptation, 2) : '—'} />
                <Readout label="Calcium" value={live ? fixed(live.calcium, 3) : '—'} />
                <Readout label="Spikes" value={live ? grouped(live.spikeCount) : '—'} />
                <Readout
                  label="Last spike"
                  value={
                    live && Number.isFinite(live.lastSpike) ? `${fixed(live.lastSpike, 1)} ms` : 'never'
                  }
                />
              </dl>
            </PanelSection>

            {isHH ? (
              <>
                <Separator />
                <PanelSection label="Ion channels">
                  <Meter value={live?.gateM ?? 0} label="m — Na activation" tone="accent" />
                  <Meter value={live?.gateH ?? 0} label="h — Na inactivation" tone="secondary" className="mt-2" />
                  <Meter value={live?.gateN ?? 0} label="n — K activation" tone="success" className="mt-2" />
                </PanelSection>
              </>
            ) : null}
          </TabPanel>

          <TabPanel value="params">
            <PanelSection label="Annotation" flush>
              <Field
                label="Label"
                description="Names are how a cell stays findable once the network is large."
              >
                <input
                  value={labelDraft}
                  onChange={(e) => setLabelDraft(e.target.value)}
                  onBlur={commitLabel}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.currentTarget.blur();
                    } else if (e.key === 'Escape') {
                      // Revert rather than commit, matching every other text field
                      // in the app and the user's expectation of Escape.
                      setLabelDraft(neuron.label);
                      e.currentTarget.blur();
                    }
                  }}
                  placeholder={neuron.id.slice(0, 12)}
                  className="w-full rounded-control border border-hairline bg-bg px-2 py-1 text-[12px] text-ink outline-none placeholder:text-ink-faint focus-visible:border-accent"
                />
              </Field>
            </PanelSection>

            <Separator />

            <PanelSection label="Model">
              <Field label="Membrane model" description="Switching resets parameters to that model's defaults.">
                <Select value={neuron.params.kind} onValueChange={changeModel}>
                  {NEURON_MODEL_KINDS.map((kind) => (
                    <SelectItem key={kind} value={kind}>
                      {NEURON_MODEL_LABELS[kind]}
                    </SelectItem>
                  ))}
                </Select>
              </Field>
            </PanelSection>

            <Separator />

            <PanelSection label="Parameters">
              {fields.map((field) => (
                <Field key={field.key} label={field.label} description={field.hint} className="mb-2">
                  <NumberField
                    value={readParam(neuron.params, field.key)}
                    onChange={(value) => changeParam(field.key, value)}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    precision={field.precision}
                    unit={field.unit}
                    logarithmic={field.log}
                  />
                </Field>
              ))}
            </PanelSection>

            <Separator />

            <PanelSection label="Drive">
              <Field label="Bias current" description="Constant current injected into this neuron.">
                <NumberField
                  value={neuron.bias}
                  onChange={(value) => updateNeuron(neuron.id, { bias: value })}
                  min={-2000}
                  max={2000}
                  step={5}
                  precision={0}
                  unit="pA"
                />
              </Field>
              <Field label="Noise" description="Standard deviation of the injected white-noise current." className="mt-2">
                <NumberField
                  value={neuron.noise}
                  onChange={(value) => updateNeuron(neuron.id, { noise: value })}
                  min={0}
                  max={500}
                  step={1}
                  precision={0}
                  unit="pA"
                />
              </Field>
              <Field label="Enabled" description="Disabled neurons stay visible but are excluded from integration." className="mt-2">
                <Switch
                  checked={neuron.enabled}
                  onCheckedChange={(checked) => updateNeuron(neuron.id, { enabled: checked })}
                />
              </Field>
            </PanelSection>
          </TabPanel>

          <TabPanel value="wiring">
            <PanelSection label={`Incoming — ${connections.incoming.length}`} flush>
              <SynapseList
                entries={connections.incoming.map((s) => ({
                  id: s.id,
                  peer: s.source,
                  receptor: RECEPTOR_LABELS[s.receptor],
                  weight: s.weight,
                  delay: s.delay,
                  plasticity: s.plasticity.kind,
                }))}
                circuit={circuit}
              />
            </PanelSection>

            <Separator />

            <PanelSection label="Similar cells">
              {similar.length === 0 ? (
                <p className="text-[11px] text-ink-faint">
                  This cell has no connections to compare.
                </p>
              ) : (
                <ul className="flex flex-col gap-0.5">
                  {similar.map((entry) => {
                    const id = getEngine().idOf(entry.slot);
                    const peer = id ? circuit.neurons.find((n) => n.id === id) : null;
                    return (
                      <li key={entry.slot}>
                        <button
                          type="button"
                          onClick={() => (id ? select([id as NeuronId]) : undefined)}
                          className="flex w-full items-center gap-1.5 rounded-control px-1.5 py-1 text-left transition-colors hover:bg-panel-raised"
                        >
                          <span
                            aria-hidden
                            className="size-2 shrink-0 rounded-[2px]"
                            style={{
                              backgroundColor: peer
                                ? identityColorHex(peer.morphology.seed)
                                : '#444',
                            }}
                          />
                          <span className="min-w-0 flex-1 truncate text-[11px] text-ink">
                            {peer?.label || id?.slice(0, 10) || `slot ${entry.slot}`}
                          </span>
                          {/* The bar makes a run of near-identical scores legible
                              in a way three decimal places never is. */}
                          <span className="h-1 w-10 shrink-0 overflow-hidden rounded-full bg-panel">
                            <span
                              className="block h-full bg-accent"
                              style={{ width: `${Math.max(2, entry.score * 100)}%` }}
                            />
                          </span>
                          <span className="nf-numeric w-8 shrink-0 text-right text-[10px] text-ink-faint">
                            {entry.score.toFixed(2)}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </PanelSection>

            <Separator />

            <PanelSection label={`Outgoing — ${connections.outgoing.length}`}>
              <SynapseList
                entries={connections.outgoing.map((s) => ({
                  id: s.id,
                  peer: s.target,
                  receptor: RECEPTOR_LABELS[s.receptor],
                  weight: s.weight,
                  delay: s.delay,
                  plasticity: s.plasticity.kind,
                }))}
                circuit={circuit}
              />
            </PanelSection>
          </TabPanel>
        </ScrollArea>
      </Tabs>
    </Panel>
  );
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-ink-faint">{label}</dt>
      <dd className="nf-numeric text-ink">{value}</dd>
    </div>
  );
}

interface SynapseEntry {
  id: string;
  peer: string;
  receptor: string;
  weight: number;
  delay: number;
  plasticity: string;
}

function SynapseList({
  entries,
  circuit,
}: {
  entries: SynapseEntry[];
  circuit: ReturnType<typeof useEditor.getState>['circuit'];
}) {
  const select = useEditor((s) => s.select);
  const labelOf = useCallback(
    (id: string) => circuit.neurons.find((n) => n.id === id)?.label || id.slice(0, 8),
    [circuit.neurons],
  );

  if (entries.length === 0) {
    return <p className="text-[11px] text-ink-faint">None.</p>;
  }

  // Long connection lists are the common case for a recurrent circuit; showing
  // all of them would make the panel unusable, so this caps and says so.
  const shown = entries.slice(0, 40);

  return (
    <ul className="flex flex-col gap-1">
      {shown.map((entry) => (
        <li key={entry.id}>
          <button
            type="button"
            onClick={() => select([entry.peer as NeuronId])}
            className="flex w-full items-center justify-between rounded-control px-2 py-1 text-left text-[11px] transition-colors hover:bg-panel-raised focus-visible:bg-panel-raised"
          >
            <span className="truncate text-ink">{labelOf(entry.peer)}</span>
            <span className="nf-numeric flex shrink-0 items-center gap-2 text-ink-faint">
              <span>{entry.receptor}</span>
              <span className="text-accent">{fixed(entry.weight, 2)}</span>
              <span>{fixed(entry.delay, 1)} ms</span>
              {entry.plasticity !== 'static' ? <Zap className="size-3 text-success" /> : null}
            </span>
          </button>
        </li>
      ))}
      {entries.length > shown.length ? (
        <li className="px-2 pt-1 text-[11px] text-ink-faint">
          +{entries.length - shown.length} more not shown
        </li>
      ) : null}
    </ul>
  );
}
