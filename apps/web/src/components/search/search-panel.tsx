'use client';

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ListFilter, Search, X } from 'lucide-react';
import {
  Button,
  EmptyState,
  Field,
  IconButton,
  NumberField,
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
  COLORS,
  NEURON_MODEL_KINDS,
  NEURON_MODEL_LABELS,
  POLARITY_COLORS,
  identityColorHex,
} from '@neuroforge/shared';
import type { Neuron, NeuronId, NeuronModelKind } from '@neuroforge/shared';

import { getEngine } from '@/lib/runtime';
import { fixed, grouped } from '@/lib/format';

/** Rows actually mounted. Beyond this the panel reports the remainder instead. */
const MAX_ROWS = 200;
/** Top of the firing-rate filter's domain; a maximum here means "no limit". */
const RATE_CEILING = 500;
const POLL_HZ = 10;
/** Population-filter key for neurons that belong to no population. */
const NO_POPULATION = 'unassigned';
/**
 * Default docking. Replaced wholesale by `className` rather than merged with it:
 * a caller that wants the right edge has to be able to drop `left-3`, and
 * `left-3 right-3` is not a conflict tailwind-merge resolves.
 */
const PLACEMENT = 'absolute top-3 bottom-3 left-3 w-[320px]';

const NO_KEYS: ReadonlySet<string> = new Set<string>();

type PolarityFilter = 'any' | 'excitatory' | 'inhibitory';
type ModelFilter = 'any' | NeuronModelKind;

const POLARITY_OPTIONS: readonly { value: PolarityFilter; label: string; title: string }[] = [
  { value: 'any', label: 'Any', title: 'Both polarities' },
  { value: 'excitatory', label: 'Exc', title: 'Excitatory only' },
  { value: 'inhibitory', label: 'Inh', title: 'Inhibitory only' },
];

/** Row-width abbreviations; the full names are far too long for a dense list. */
const MODEL_SHORT: Record<NeuronModelKind, string> = {
  lif: 'LIF',
  izhikevich: 'IZH',
  'hodgkin-huxley': 'HH',
  adex: 'AdEx',
  'morris-lecar': 'ML',
};

interface RateSnapshot {
  /**
   * Firing rate in Hz, indexed parallel to the neuron array it was sampled for.
   * `NaN` marks a cell the engine has not loaded yet, which must not read as a
   * measured zero.
   */
  values: Float32Array;
  /** Bumped on every sample so memos can key on the poll rather than on identity. */
  version: number;
}

const EMPTY_RATES: RateSnapshot = { values: new Float32Array(0), version: 0 };

/** The rate of a neuron by document index; `NaN` when it has not been sampled. */
function rateAt(values: Float32Array, index: number): number {
  return index < values.length ? values[index] : Number.NaN;
}

/**
 * Poll every neuron's firing rate off the live buffers.
 *
 * The engine is loaded from an effect in the workspace, and child effects run
 * before their parent's, so the id-to-slot map can lag the document by one
 * sample. A map is therefore only cached once it resolves every id against a
 * buffer of the matching size; an incomplete one is rebuilt on the next tick,
 * which makes the staleness self-healing rather than permanent.
 *
 * Sampling stops entirely while `enabled` is false, and a sample identical to
 * the last one is dropped rather than published: a paused simulation, or a
 * closed panel, must not re-render a two-hundred-row list ten times a second.
 */
function useLiveRates(neurons: readonly Neuron[], hz: number, enabled: boolean): RateSnapshot {
  const [snapshot, setSnapshot] = useState<RateSnapshot>(EMPTY_RATES);
  const cache = useRef<{
    neurons: readonly Neuron[] | null;
    slots: Int32Array;
    values: Float32Array;
    /** The array identity last handed to React, to detect a reallocation. */
    published: Float32Array | null;
  }>({ neurons: null, slots: new Int32Array(0), values: new Float32Array(0), published: null });

  useEffect(() => {
    if (!enabled) return;

    const sample = () => {
      const engine = getEngine();
      const buffers = engine.buffers.neurons;
      const count = neurons.length;
      const store = cache.current;

      if (store.slots.length !== count) {
        store.slots = new Int32Array(count);
        store.values = new Float32Array(count);
        store.neurons = null;
      }

      // Rebuilding when the engine's own count disagrees as well as when the
      // document changed catches a runtime that was disposed and reconstructed
      // underneath a document whose array identity never moved.
      if (store.neurons !== neurons || buffers.count !== count) {
        let complete = buffers.count === count;
        for (let i = 0; i < count; i += 1) {
          const slot = engine.slotOf(neurons[i].id);
          store.slots[i] = slot;
          if (slot < 0) complete = false;
        }
        store.neurons = complete ? neurons : null;
      }

      const { slots, values } = store;
      let changed = store.published !== values;
      for (let i = 0; i < count; i += 1) {
        const slot = slots[i];
        const next = slot >= 0 && slot < buffers.count ? buffers.rate[slot] : Number.NaN;
        // Object.is rather than !==, so an unresolved cell does not read as a
        // change on every single tick.
        if (!Object.is(next, values[i])) {
          values[i] = next;
          changed = true;
        }
      }

      if (!changed) return;
      store.published = values;
      setSnapshot((previous) => ({ values, version: previous.version + 1 }));
    };

    sample();
    const id = setInterval(sample, Math.max(16, Math.round(1000 / hz)));
    return () => clearInterval(id);
  }, [neurons, hz, enabled]);

  return snapshot;
}

interface PopulationChip {
  key: string;
  name: string;
  size: number;
  color: string;
}

export interface SearchPanelProps {
  /** Hidden entirely when false, so a toggle can mount it permanently. */
  open?: boolean;
  /** Supplying this renders a close control in the header. */
  onClose?: () => void;
  /** Replaces the default left-edge docking entirely. */
  className?: string;
}

/**
 * Cell query panel.
 *
 * Filters are combined with AND and evaluated in two passes: everything
 * structural is memoised against the document, and the firing-rate window is
 * applied on top of that result at the poll rate. A rate window that spans the
 * whole domain skips the second pass entirely, so a panel with no rate filter
 * costs nothing extra as the simulation runs.
 */
export function SearchPanel({ open = true, onClose, className }: SearchPanelProps) {
  const circuit = useEditor((s) => s.circuit);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);

  const [query, setQuery] = useState('');
  const [polarity, setPolarity] = useState<PolarityFilter>('any');
  const [model, setModel] = useState<ModelFilter>('any');
  const [populations, setPopulations] = useState<ReadonlySet<string>>(NO_KEYS);
  const [rateMin, setRateMin] = useState(0);
  const [rateMax, setRateMax] = useState(Number.POSITIVE_INFINITY);
  const [degreeMin, setDegreeMin] = useState(0);
  const [degreeMax, setDegreeMax] = useState(Number.POSITIVE_INFINITY);
  const [filtersOpen, setFiltersOpen] = useState(true);

  const rates = useLiveRates(circuit.neurons, POLL_HZ, open);

  const degrees = useMemo(() => {
    const map = new Map<string, number>();
    for (const synapse of circuit.synapses) {
      map.set(synapse.source, (map.get(synapse.source) ?? 0) + 1);
      map.set(synapse.target, (map.get(synapse.target) ?? 0) + 1);
    }
    return map;
  }, [circuit.synapses]);

  const degreeCeiling = useMemo(() => {
    let top = 0;
    for (const degree of degrees.values()) if (degree > top) top = degree;
    return Math.max(1, top);
  }, [degrees]);

  const known = useMemo(
    () => new Set<string>(circuit.populations.map((population) => population.id)),
    [circuit.populations],
  );

  /**
   * The chip a cell belongs to. An id left dangling by a deleted population
   * resolves to Unassigned rather than to a key no chip carries, so every cell
   * in the document is reachable through exactly one chip.
   */
  const chipKey = useCallback(
    (population: Neuron['population']): string =>
      population !== null && known.has(population) ? population : NO_POPULATION,
    [known],
  );

  const chips = useMemo<PopulationChip[]>(() => {
    // Counted off the neurons rather than read from `population.members`: the
    // number on a chip has to be the number of rows clicking it produces, and
    // membership lists are a separate record that a direct edit can outrun.
    const counts = new Map<string, number>();
    for (const neuron of circuit.neurons) {
      const key = chipKey(neuron.population);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const list: PopulationChip[] = circuit.populations.map((population) => ({
      key: population.id,
      name: population.name,
      size: counts.get(population.id) ?? 0,
      color: population.color ?? POLARITY_COLORS[population.polarity],
    }));
    const unassigned = counts.get(NO_POPULATION) ?? 0;
    if (unassigned > 0) {
      list.push({
        key: NO_POPULATION,
        name: 'Unassigned',
        size: unassigned,
        color: COLORS.textFaint,
      });
    }
    return list;
  }, [chipKey, circuit.neurons, circuit.populations]);

  const rateUpper = rateMax >= RATE_CEILING ? Number.POSITIVE_INFINITY : rateMax;
  const degreeUpper = degreeMax >= degreeCeiling ? Number.POSITIVE_INFINITY : degreeMax;
  const rateActive = rateMin > 0 || Number.isFinite(rateUpper);

  const structural = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const hits: number[] = [];
    for (let i = 0; i < circuit.neurons.length; i += 1) {
      const neuron = circuit.neurons[i];
      if (polarity !== 'any' && neuron.polarity !== polarity) continue;
      if (model !== 'any' && neuron.params.kind !== model) continue;
      if (populations.size > 0 && !populations.has(chipKey(neuron.population))) continue;
      const degree = degrees.get(neuron.id) ?? 0;
      if (degree < degreeMin || degree > degreeUpper) continue;
      if (
        needle !== '' &&
        !neuron.label.toLowerCase().includes(needle) &&
        !neuron.id.toLowerCase().includes(needle)
      ) {
        continue;
      }
      hits.push(i);
    }
    return hits;
  }, [
    chipKey,
    circuit.neurons,
    degrees,
    degreeMin,
    degreeUpper,
    model,
    polarity,
    populations,
    query,
  ]);

  const matches = useMemo(() => {
    if (!rateActive) return structural;
    const { values } = rates;
    // An unsampled cell has no known rate, so it satisfies no rate window:
    // every comparison against NaN is false.
    return structural.filter((index) => {
      const rate = rateAt(values, index);
      return rate >= rateMin && rate <= rateUpper;
    });
  }, [structural, rateActive, rateMin, rateUpper, rates]);

  const selected = useMemo(() => new Set<string>(selection), [selection]);

  const active =
    query.trim() !== '' ||
    polarity !== 'any' ||
    model !== 'any' ||
    populations.size > 0 ||
    rateMin > 0 ||
    Number.isFinite(rateUpper) ||
    degreeMin > 0 ||
    Number.isFinite(degreeUpper);

  const pick = useCallback(
    (id: NeuronId, additive: boolean) => {
      select([id], additive);
    },
    [select],
  );

  const selectMatches = useCallback(() => {
    if (matches.length === 0) return;
    select(matches.map((index) => circuit.neurons[index].id));
  }, [matches, circuit.neurons, select]);

  const reset = useCallback(() => {
    setQuery('');
    setPolarity('any');
    setModel('any');
    setPopulations(NO_KEYS);
    setRateMin(0);
    setRateMax(Number.POSITIVE_INFINITY);
    setDegreeMin(0);
    setDegreeMax(Number.POSITIVE_INFINITY);
  }, []);

  const togglePopulation = useCallback((key: string) => {
    setPopulations((previous) => {
      const next = new Set(previous);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  if (!open) return null;

  const shown = matches.slice(0, MAX_ROWS);
  const hidden = matches.length - shown.length;

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', className ?? PLACEMENT)}>
      <PanelHeader
        title="Cell search"
        subtitle={`${grouped(circuit.neurons.length)} cells · ${grouped(circuit.synapses.length)} synapses`}
        icon={<Search size={14} />}
        actions={
          <>
            <IconButton
              label="Toggle filters"
              size="sm"
              variant={filtersOpen ? 'secondary' : 'ghost'}
              aria-pressed={filtersOpen}
              onClick={() => setFiltersOpen((value) => !value)}
            >
              <ListFilter size={13} />
            </IconButton>
            {onClose ? (
              <IconButton label="Close cell search" size="sm" onClick={onClose}>
                <X size={13} />
              </IconButton>
            ) : null}
          </>
        }
      />

      <div className="flex h-8 shrink-0 items-center gap-1.5 border-b border-hairline px-2.5">
        <Search size={12} aria-hidden className="shrink-0 text-ink-faint" />
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Escape' && query !== '') {
              event.preventDefault();
              event.stopPropagation();
              setQuery('');
            } else if (event.key === 'Enter') {
              event.preventDefault();
              selectMatches();
            }
          }}
          placeholder="Label or id…"
          aria-label="Search cells by label or id"
          spellCheck={false}
          autoComplete="off"
          className="min-w-0 flex-1 bg-transparent text-[11.5px] text-ink outline-none placeholder:text-ink-faint"
        />
        {query !== '' ? (
          <IconButton
            label="Clear search text"
            size="sm"
            className="size-5"
            onClick={() => setQuery('')}
          >
            <X size={11} />
          </IconButton>
        ) : null}
      </div>

      {filtersOpen ? (
        <div className="max-h-[46%] shrink-0 overflow-y-auto overscroll-contain border-b border-hairline">
          <PanelSection label="Class" flush>
            <Field label="Polarity" orientation="column">
              <SegmentedControl<PolarityFilter>
                value={polarity}
                onChange={setPolarity}
                options={POLARITY_OPTIONS}
                size="sm"
                fullWidth
              />
            </Field>
            <Field label="Model" orientation="column" className="mt-1.5">
              <Select
                value={model}
                onValueChange={(value) => setModel(value as ModelFilter)}
                size="sm"
              >
                <SelectItem value="any">Any model</SelectItem>
                {NEURON_MODEL_KINDS.map((kind) => (
                  <SelectItem key={kind} value={kind}>
                    {NEURON_MODEL_LABELS[kind]}
                  </SelectItem>
                ))}
              </Select>
            </Field>
          </PanelSection>

          <PanelSection
            label="Populations"
            aside={
              populations.size > 0 ? (
                <button
                  type="button"
                  onClick={() => setPopulations(NO_KEYS)}
                  className="text-[10px] text-ink-faint transition-colors hover:text-ink"
                >
                  reset
                </button>
              ) : undefined
            }
          >
            {chips.length === 0 ? (
              <p className="text-[10.5px] text-ink-faint">
                No populations; every cell was placed individually.
              </p>
            ) : (
              <div className="flex flex-wrap gap-1">
                {chips.map((chip) => {
                  const on = populations.has(chip.key);
                  return (
                    <button
                      key={chip.key}
                      type="button"
                      aria-pressed={on}
                      onClick={() => togglePopulation(chip.key)}
                      className={cn(
                        'flex h-5 min-w-0 max-w-full items-center gap-1.5 rounded-full border px-1.5',
                        'text-[10px] transition-colors',
                        on
                          ? 'border-accent/45 bg-accent/12 text-ink'
                          : 'border-hairline bg-white/[0.03] text-ink-muted hover:border-hairline-strong hover:text-ink',
                      )}
                    >
                      <span
                        aria-hidden
                        className="size-1.5 shrink-0 rounded-full"
                        style={{ backgroundColor: chip.color }}
                      />
                      <span className="truncate">{chip.name}</span>
                      <span className="nf-numeric shrink-0 text-ink-faint">{chip.size}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </PanelSection>

          <PanelSection label="Ranges">
            <Field label="Firing rate — Hz" orientation="column">
              <div className="flex items-center gap-1.5">
                <NumberField
                  value={rateMin}
                  onChange={setRateMin}
                  min={0}
                  max={RATE_CEILING}
                  step={1}
                  precision={0}
                  defaultValue={0}
                  aria-label="Minimum firing rate"
                  className="min-w-0 flex-1"
                />
                <span aria-hidden className="shrink-0 text-[10px] text-ink-faint">
                  –
                </span>
                <NumberField
                  value={Math.min(rateMax, RATE_CEILING)}
                  onChange={(value) =>
                    setRateMax(value >= RATE_CEILING ? Number.POSITIVE_INFINITY : value)
                  }
                  min={0}
                  max={RATE_CEILING}
                  step={1}
                  precision={0}
                  defaultValue={RATE_CEILING}
                  aria-label="Maximum firing rate"
                  className="min-w-0 flex-1"
                />
              </div>
            </Field>
            <Field label="Degree — synapses" orientation="column" className="mt-1.5">
              <div className="flex items-center gap-1.5">
                <NumberField
                  value={degreeMin}
                  onChange={setDegreeMin}
                  min={0}
                  max={degreeCeiling}
                  step={1}
                  precision={0}
                  defaultValue={0}
                  aria-label="Minimum degree"
                  className="min-w-0 flex-1"
                />
                <span aria-hidden className="shrink-0 text-[10px] text-ink-faint">
                  –
                </span>
                <NumberField
                  value={Math.min(degreeMax, degreeCeiling)}
                  onChange={(value) =>
                    setDegreeMax(value >= degreeCeiling ? Number.POSITIVE_INFINITY : value)
                  }
                  min={0}
                  max={degreeCeiling}
                  step={1}
                  precision={0}
                  defaultValue={degreeCeiling}
                  aria-label="Maximum degree"
                  className="min-w-0 flex-1"
                />
              </div>
            </Field>
            <p className="mt-1 text-[10px] leading-snug text-ink-faint">
              Bounds are inclusive. A maximum left at the top of its range means no limit. Degree
              counts incoming and outgoing synapses together.
            </p>
          </PanelSection>
        </div>
      ) : null}

      <div className="flex h-8 shrink-0 items-center gap-1.5 border-b border-hairline px-2.5">
        <span className="nf-numeric text-[10px] text-ink-muted">
          {grouped(matches.length)}
          <span className="text-ink-faint">/{grouped(circuit.neurons.length)}</span>
        </span>
        {selection.length > 0 ? (
          <span className="nf-numeric text-[10px] text-accent">{grouped(selection.length)} sel</span>
        ) : null}
        <div className="flex-1" />
        <Tooltip content="Select every match" shortcut="Enter" side="top">
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-1.5 text-[10.5px]"
            disabled={matches.length === 0}
            onClick={selectMatches}
          >
            Select all
          </Button>
        </Tooltip>
        <Tooltip content="Clear every filter" side="top">
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-1.5 text-[10.5px]"
            disabled={!active}
            onClick={reset}
          >
            Clear
          </Button>
        </Tooltip>
      </div>

      <div
        aria-hidden
        className="flex h-5 shrink-0 items-center gap-2 border-b border-hairline px-3 text-[9px] font-semibold uppercase tracking-[0.09em] text-ink-faint"
      >
        <span className="size-2 shrink-0" />
        <span className="min-w-0 flex-1">Cell</span>
        <span className="w-9 shrink-0 text-right">Model</span>
        <span className="w-2 shrink-0 text-center">P</span>
        <span className="w-10 shrink-0 text-right">Hz</span>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        {shown.length === 0 ? (
          <EmptyState
            compact
            icon={<Search size={14} />}
            title="No matches"
            description={
              circuit.neurons.length === 0
                ? 'This circuit has no cells yet.'
                : 'No cell satisfies every active filter.'
            }
          />
        ) : (
          <ul className="flex flex-col py-0.5">
            {shown.map((index) => {
              const neuron = circuit.neurons[index];
              return (
                <li key={neuron.id}>
                  <ResultRow
                    id={neuron.id}
                    label={neuron.label !== '' ? neuron.label : neuron.id}
                    color={identityColorHex(neuron.morphology.seed)}
                    model={MODEL_SHORT[neuron.params.kind]}
                    inhibitory={neuron.polarity === 'inhibitory'}
                    rate={rateAt(rates.values, index)}
                    selected={selected.has(neuron.id)}
                    onSelect={pick}
                  />
                </li>
              );
            })}
            {hidden > 0 ? (
              <li className="px-3 py-1.5 text-[10px] leading-snug text-ink-faint">
                +{grouped(hidden)} more matched. Narrow the query to see them.
              </li>
            ) : null}
          </ul>
        )}
      </ScrollArea>
    </Panel>
  );
}

interface ResultRowProps {
  id: NeuronId;
  label: string;
  color: string;
  model: string;
  inhibitory: boolean;
  rate: number;
  selected: boolean;
  onSelect: (id: NeuronId, additive: boolean) => void;
}

/**
 * One result. Memoised because the panel re-renders at the poll rate and only
 * the rows whose rate actually moved need to be reconciled.
 */
const ResultRow = memo(function ResultRow({
  id,
  label,
  color,
  model,
  inhibitory,
  rate,
  selected,
  onSelect,
}: ResultRowProps) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      title={`${label} · ${id}`}
      onClick={(event) => onSelect(id, event.shiftKey || event.metaKey || event.ctrlKey)}
      className={cn(
        'flex h-6 w-full items-center gap-2 px-3 text-left',
        'transition-colors duration-[120ms] focus-visible:outline-none',
        selected ? 'bg-accent/12' : 'hover:bg-white/[0.05] focus-visible:bg-white/[0.07]',
      )}
    >
      <span
        aria-hidden
        className="size-2 shrink-0 rounded-[2px]"
        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}66` }}
      />
      <span
        className={cn('min-w-0 flex-1 truncate text-[11px]', selected ? 'text-ink' : 'text-ink-muted')}
      >
        {label}
      </span>
      <span className="nf-numeric w-9 shrink-0 text-right text-[9.5px] text-ink-faint">
        {model}
      </span>
      <span
        className={cn(
          'w-2 shrink-0 text-center text-[10px] font-semibold',
          inhibitory ? 'text-secondary' : 'text-accent',
        )}
      >
        {inhibitory ? 'I' : 'E'}
      </span>
      <span
        className={cn(
          'nf-numeric w-10 shrink-0 text-right text-[10px]',
          rate >= 0.05 ? 'text-ink' : 'text-ink-faint',
        )}
      >
        {fixed(rate, 1)}
      </span>
    </button>
  );
});
