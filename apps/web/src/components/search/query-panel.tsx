'use client';

import { memo, useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpen,
  BookmarkPlus,
  Braces,
  CornerDownLeft,
  Eye,
  EyeOff,
  ListPlus,
  SquareDashedMousePointer,
  Trash2,
  TriangleAlert,
  X,
} from 'lucide-react';
import {
  Badge,
  Button,
  EmptyState,
  IconButton,
  Kbd,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  Spinner,
  Tooltip,
  cn,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { getSetting, setSetting } from '@neuroforge/io';
import {
  DEFAULT_RENDER_SETTINGS,
  MORPHOLOGY_ARCHETYPES,
  NEURON_MODEL_KINDS,
  RECEPTOR_KINDS,
  identityColorHex,
} from '@neuroforge/shared';
import type { Neuron, NeuronId, NeuronModelKind } from '@neuroforge/shared';

import { getEngine } from '@/lib/runtime';
import { fixed, grouped } from '@/lib/format';
import { graphSignature } from '@/lib/graph-metrics';
import {
  QUERY_FIELDS,
  QUERY_OPERATORS,
  RESERVED_CHARS,
  findField,
  matchedSlots,
  runQuery,
  tokenize,
} from '@/lib/query';
import type {
  OperatorGroup,
  OperatorSpec,
  QueryError,
  QueryField,
  QueryResult,
} from '@/lib/query';

/**
 * Delay between the last keystroke and the query running.
 *
 * Evaluation walks every cell, and a connectivity clause walks the adjacency as
 * well; running that per keystroke would make a forty-character query cost forty
 * passes over the network. Two hundred milliseconds is under the threshold where
 * a count stops reading as live and over the gap between the keystrokes of
 * anyone actually typing.
 */
const DEBOUNCE_MS = 200;

/** Cadence at which the panel checks whether the running network moved under it. */
const SIGNATURE_POLL_MS = 500;

/** Firing rates are re-read at this rate, for the mounted rows only. */
const RATE_HZ = 10;

/** Rows actually mounted. Past this the list reports the remainder instead. */
const MAX_ROWS = 200;

/** Completions offered at once. A longer list is a menu, not a suggestion. */
const MAX_SUGGESTIONS = 9;

/**
 * Distinct labels collected for value completion.
 *
 * Every label in a hundred-thousand-cell connectome is not a menu, and this is a
 * convenience rather than an index: past this many the list is truncated and the
 * name gets typed instead.
 */
const MAX_LABEL_VALUES = 500;

/** `dimUnselected` used by Isolate. Not zero, so hidden cells still hint at context. */
const ISOLATE_DIM = 0.02;

/** Saved queries live under this key in the settings table. */
const SAVED_KEY = 'query.saved';

/** Saved queries kept; the oldest are dropped past this. */
const MAX_SAVED = 40;

/** Query text echoed back inside a message, before it is elided. */
const ECHO_CHARS = 60;

const PLACEMENT = 'absolute top-3 bottom-3 left-3 w-[368px]';

const EMPTY_SLOTS: readonly number[] = [];
const EMPTY_IDS: readonly NeuronId[] = [];
const EMPTY_RATES = new Float32Array(0);

/** Row-width abbreviations; the full model names are far too long for a dense list. */
const MODEL_SHORT: Record<NeuronModelKind, string> = {
  lif: 'LIF',
  izhikevich: 'IZH',
  'hodgkin-huxley': 'HH',
  adex: 'AdEx',
  'morris-lecar': 'ML',
};

const GROUP_LABELS: Record<OperatorGroup, string> = {
  text: 'Text',
  numeric: 'Numeric',
  connectivity: 'Connectivity',
  similarity: 'Similarity',
  presence: 'Presence',
  logic: 'Logic',
};

/**
 * The operator table sectioned by group, in the order the table declares them.
 *
 * Derived rather than transcribed, so an operator added to the language shows up
 * in the reference without anyone remembering to add it here as well.
 */
const OPERATOR_SECTIONS: readonly { group: OperatorGroup; operators: OperatorSpec[] }[] = (() => {
  const sections: { group: OperatorGroup; operators: OperatorSpec[] }[] = [];
  const index = new Map<OperatorGroup, OperatorSpec[]>();
  for (const spec of QUERY_OPERATORS) {
    let bucket = index.get(spec.group);
    if (bucket === undefined) {
      bucket = [];
      index.set(spec.group, bucket);
      sections.push({ group: spec.group, operators: bucket });
    }
    bucket.push(spec);
  }
  return sections;
})();

/* ------------------------------------------------------------- completion -- */

const RESERVED = new Set<string>(RESERVED_CHARS.split(''));
const WHITESPACE = /\s/;

/** Mirrors the lexer's notion of a bare word, so completion splits where it does. */
function isWordChar(char: string): boolean {
  if (WHITESPACE.test(char)) return false;
  if (char === '{' || char === '}' || char === '"' || char === "'") return false;
  return !RESERVED.has(char);
}

/** True for the characters a bracket operator name is spelled with. */
function isBracketNameChar(char: string): boolean {
  return (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || char === '_';
}

/** Wrap a value in quotes when writing it bare would lex as something else. */
function quoteValue(value: string): string {
  let needs = value === '';
  for (const char of value) {
    if (
      WHITESPACE.test(char) ||
      RESERVED.has(char) ||
      char === '{' ||
      char === '}' ||
      char === '"' ||
      char === "'" ||
      char === ','
    ) {
      needs = true;
      break;
    }
  }
  if (!needs) return value;
  return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
}

/** Shorten a query for use inside a sentence. */
function echo(source: string): string {
  const trimmed = source.trim();
  return trimmed.length <= ECHO_CHARS ? trimmed : `${trimmed.slice(0, ECHO_CHARS - 1)}…`;
}

/**
 * What the grammar allows at a point in the source.
 *
 * `term` is the start of a clause, `operator` follows a bare word, `value`
 * follows an operator that takes one, `field` follows a presence operator, and
 * `logical` follows a finished clause.
 */
type Expectation =
  | { kind: 'term' }
  | { kind: 'operator'; field: QueryField | null }
  | { kind: 'value'; spec: OperatorSpec; field: QueryField | null }
  | { kind: 'field' }
  | { kind: 'logical' };

/**
 * Walk the tokens before the caret and report what may follow them.
 *
 * This is a deliberately loose reading of the grammar in `parser.ts`: it tracks
 * the shape of the next term rather than building a tree, because a query being
 * typed is incomplete by definition and parsing it would fail on the last token
 * every single time. Anything this cannot follow returns null, which shows no
 * suggestions rather than wrong ones.
 */
function expectationAt(head: string): Expectation | null {
  const lexed = tokenize(head);
  if (!lexed.ok) return null;

  let state: Expectation = { kind: 'term' };
  for (const token of lexed.tokens) {
    if (token.kind === 'end') break;
    const spec = token.operator;

    switch (state.kind) {
      case 'term': {
        if (token.kind === 'word') {
          state = { kind: 'operator', field: token.quoted ? null : findField(token.value) };
          break;
        }
        if (token.kind === 'group-open') break;
        if (token.kind === 'operator' && spec !== null) {
          if (spec.shape === 'prefix') break;
          if (spec.shape === 'value') {
            state = { kind: 'value', spec, field: null };
            break;
          }
          if (spec.shape === 'attribute') {
            state = { kind: 'field' };
            break;
          }
        }
        return null;
      }
      case 'operator': {
        if (token.kind === 'operator' && spec !== null) {
          if (spec.shape === 'field') {
            state = { kind: 'value', spec, field: state.field };
            break;
          }
          if (spec.shape === 'pathway') {
            state = { kind: 'value', spec, field: null };
            break;
          }
        }
        return null;
      }
      case 'value':
      case 'field': {
        if (token.kind === 'word') {
          state = { kind: 'logical' };
          break;
        }
        return null;
      }
      case 'logical': {
        if (token.kind === 'group-close') break;
        if (token.kind === 'operator' && spec !== null && spec.shape === 'logical') {
          state = { kind: 'term' };
          break;
        }
        return null;
      }
    }
  }
  return state;
}

type SuggestionKind = 'field' | 'operator' | 'value';

interface Suggestion {
  /** Source text written in place of the token being edited. */
  insert: string;
  /** Primary text of the row. */
  title: string;
  /** Trailing monospace detail: the other spelling, the field type, the unit. */
  detail: string;
  /** One line under the title. */
  hint: string;
  kind: SuggestionKind;
  /** Everything the typed prefix is matched against, lowercased. */
  terms: readonly string[];
}

const NO_SUGGESTIONS: readonly Suggestion[] = [];

/** The token being edited and what may replace it. */
interface Completion {
  /** Offset where the replacement starts. */
  from: number;
  /** Offset where it ends, always the caret. */
  to: number;
  suggestions: readonly Suggestion[];
}

/** Values a field is known to take beyond whatever this document happens to hold. */
interface Vocabulary {
  populations: readonly string[];
  labels: readonly string[];
}

function fieldSuggestion(field: QueryField): Suggestion {
  const detail = field.type === 'numeric' && field.unit !== '' ? `num · ${field.unit}` : field.type;
  return {
    insert: field.name,
    title: field.name,
    detail,
    hint: field.description,
    kind: 'field',
    terms: [field.name, field.description.toLowerCase()],
  };
}

function operatorSuggestion(spec: OperatorSpec): Suggestion {
  return {
    insert: spec.symbol,
    title: spec.symbol,
    detail: spec.bracket,
    hint: `${spec.label} — ${spec.description}`,
    kind: 'operator',
    terms: [spec.symbol.toLowerCase(), spec.id, spec.bracket.toLowerCase(), spec.label.toLowerCase()],
  };
}

function bracketSuggestion(spec: OperatorSpec): Suggestion {
  return {
    insert: spec.bracket,
    title: spec.bracket,
    detail: spec.symbol,
    hint: `${spec.label} — ${spec.description}`,
    kind: 'operator',
    terms: [spec.id, spec.bracket.toLowerCase(), spec.label.toLowerCase()],
  };
}

function valueSuggestion(value: string, hint: string): Suggestion {
  return {
    insert: quoteValue(value),
    title: value,
    detail: '',
    hint,
    kind: 'value',
    terms: [value.toLowerCase()],
  };
}

/** Values worth offering after an operator, given what it is comparing. */
function valuesFor(expectation: Expectation, vocabulary: Vocabulary): Suggestion[] {
  if (expectation.kind !== 'value') return [];
  const { spec, field } = expectation;

  // The connectivity, similarity and pathway operators all resolve their operand
  // against a cell id or label, never against a population name.
  if (field === null) return vocabulary.labels.map((label) => valueSuggestion(label, 'Cell label'));

  if (field.type === 'boolean') {
    return [valueSuggestion('true', 'Matches'), valueSuggestion('false', 'Does not match')];
  }

  if (field.type === 'numeric') {
    if (spec.valueForm === 'range') {
      const unit = field.unit === '' ? '' : ` ${field.unit}`;
      return [valueSuggestion('5..20', `An inclusive range, low..high, in${unit === '' ? ' plain units' : unit}`)];
    }
    return [];
  }

  switch (field.name) {
    case 'model':
      return NEURON_MODEL_KINDS.map((kind) => valueSuggestion(kind, 'Membrane model'));
    case 'polarity':
      return [
        valueSuggestion('excitatory', 'Depolarises its targets'),
        valueSuggestion('inhibitory', 'Hyperpolarises its targets'),
      ];
    case 'archetype':
      return MORPHOLOGY_ARCHETYPES.map((archetype) =>
        valueSuggestion(archetype, 'Morphology archetype'),
      );
    case 'receptor':
      return RECEPTOR_KINDS.map((receptor) => valueSuggestion(receptor, 'Receptor kind'));
    case 'population':
      return vocabulary.populations.map((name) =>
        valueSuggestion(name, 'Population in this circuit'),
      );
    case 'label':
      return vocabulary.labels.map((label) => valueSuggestion(label, 'Cell label'));
    default:
      // `id` is the field left over, and a menu of opaque identifiers is noise
      // rather than help.
      return [];
  }
}

/**
 * Everything grammatical at this point, before the typed prefix narrows it.
 *
 * `bracketed` chooses the spelling, not the set. The two forms are the same
 * operators and the parser type-checks them identically, so the field-type
 * filter below has to apply to both: offering `{like}` after `rate` because the
 * user reached for the bracket form would put a guaranteed type error in the
 * menu, which is exactly what driving this off the operator table is meant to
 * make impossible.
 */
function candidatesFor(
  expectation: Expectation,
  vocabulary: Vocabulary,
  bracketed: boolean,
): Suggestion[] {
  const spell = bracketed ? bracketSuggestion : operatorSuggestion;

  switch (expectation.kind) {
    case 'term': {
      const operators = QUERY_OPERATORS.filter(
        (spec) => spec.shape === 'value' || spec.shape === 'attribute' || spec.shape === 'prefix',
      ).map(spell);
      // A field name is a bare word; inside `{…}` only an operator can appear.
      return bracketed ? operators : [...QUERY_FIELDS.map(fieldSuggestion), ...operators];
    }
    case 'operator': {
      const field = expectation.field;
      return QUERY_OPERATORS.filter((spec) => {
        // An unrecognised word can still be the left end of a pathway, but it can
        // never be compared, so no field operator is offered for one.
        if (spec.shape === 'pathway') return true;
        if (spec.shape !== 'field') return false;
        return field !== null && spec.fieldTypes.includes(field.type);
      }).map(spell);
    }
    case 'value':
      return bracketed ? [] : valuesFor(expectation, vocabulary);
    case 'field':
      return bracketed ? [] : QUERY_FIELDS.map(fieldSuggestion);
    case 'logical':
      return QUERY_OPERATORS.filter((spec) => spec.shape === 'logical').map(spell);
  }
}

/**
 * Rank candidates against what has been typed.
 *
 * A hit at the start of a name outranks one in the middle of it, which outranks
 * one in the prose; inside a tier the table's own order is kept, so the list
 * never reshuffles as a fourth character narrows it.
 */
function rank(candidates: readonly Suggestion[], prefix: string): Suggestion[] {
  if (prefix === '') return candidates.slice(0, MAX_SUGGESTIONS);
  const needle = prefix.toLowerCase();
  const scored: { suggestion: Suggestion; tier: number; at: number }[] = [];
  candidates.forEach((suggestion, at) => {
    let tier = Number.POSITIVE_INFINITY;
    suggestion.terms.forEach((term, index) => {
      if (term.startsWith(needle)) tier = Math.min(tier, index === 0 ? 0 : 1);
      else if (term.includes(needle)) tier = Math.min(tier, index === 0 ? 2 : 3);
    });
    if (Number.isFinite(tier)) scored.push({ suggestion, tier, at });
  });
  scored.sort((a, b) => (a.tier === b.tier ? a.at - b.at : a.tier - b.tier));
  return scored.slice(0, MAX_SUGGESTIONS).map((entry) => entry.suggestion);
}

/** The token under the caret and the completions that could replace it. */
function completionAt(source: string, caret: number, vocabulary: Vocabulary): Completion | null {
  const at = Math.max(0, Math.min(caret, source.length));

  // A bracket operator is being spelled when the caret sits in a run of name
  // characters opened by a single `{`; `{{` is the group opener, not an operator.
  let nameStart = at;
  while (nameStart > 0 && isBracketNameChar(source[nameStart - 1])) nameStart -= 1;
  const bracketed =
    nameStart > 0 &&
    source[nameStart - 1] === '{' &&
    !(nameStart > 1 && source[nameStart - 2] === '{');

  let from: number;
  let prefix: string;
  if (bracketed) {
    from = nameStart - 1;
    prefix = source.slice(nameStart, at);
  } else {
    let wordStart = at;
    while (wordStart > 0 && isWordChar(source[wordStart - 1])) wordStart -= 1;
    from = wordStart;
    prefix = source.slice(wordStart, at);
  }

  const expectation = expectationAt(source.slice(0, from));
  if (expectation === null) return null;

  const suggestions = rank(candidatesFor(expectation, vocabulary, bracketed), prefix);
  if (suggestions.length === 0) return null;
  return { from, to: at, suggestions };
}

/* ------------------------------------------------------------------ saved -- */

interface SavedQuery {
  /** Unique within the list; saving under an existing name replaces it. */
  name: string;
  text: string;
  createdAt: number;
}

/**
 * Coerce whatever the settings table holds into saved queries.
 *
 * The stored value is `unknown` by contract and may have been written by an
 * older build or edited by hand in devtools, so every field is checked rather
 * than asserted. A malformed entry is dropped: one bad row must not cost the
 * user the rest of them.
 */
function readSaved(raw: unknown): SavedQuery[] {
  if (!Array.isArray(raw)) return [];
  const out: SavedQuery[] = [];
  const seen = new Set<string>();
  for (const entry of raw) {
    if (typeof entry !== 'object' || entry === null) continue;
    const record = entry as Record<string, unknown>;
    const name = typeof record.name === 'string' ? record.name.trim() : '';
    const text = typeof record.text === 'string' ? record.text : '';
    const createdAt =
      typeof record.createdAt === 'number' && Number.isFinite(record.createdAt)
        ? record.createdAt
        : 0;
    if (name === '' || text.trim() === '' || seen.has(name)) continue;
    seen.add(name);
    out.push({ name, text, createdAt });
    if (out.length >= MAX_SAVED) break;
  }
  return out;
}

/* ----------------------------------------------------------------- samples -- */

interface Sample {
  text: string;
  note: string;
}

/**
 * Worked queries, in the spirit of Codex's own sample list.
 *
 * Every field and operator used here exists in this platform's tables, so each
 * one runs exactly as written. The connectivity and similarity examples need a
 * cell to point at and so are only offered once the document has a named one.
 */
const STATIC_SAMPLES: readonly Sample[] = [
  { text: 'polarity == inhibitory && rate > 5', note: 'Inhibitory cells firing above 5 Hz.' },
  { text: 'model << lif,adex', note: 'Cells on either the LIF or the AdEx membrane model.' },
  { text: 'degree >= 10', note: 'Hubs: ten or more synapses, incoming and outgoing together.' },
  { text: 'rate <> 5..20', note: 'Cells inside the 5–20 Hz band.' },
  { text: '!$ population', note: 'Cells placed individually, belonging to no population.' },
  {
    text: 'archetype == basket && spiking == true',
    note: 'Basket cells that fired on the most recent step.',
  },
  { text: 'enabled == false', note: 'Cells excluded from integration.' },
  {
    text: '{{model == lif || model == izhikevich}} && weight_in > 20',
    note: 'Grouping: either of two models, both strongly driven.',
  },
  {
    text: 'receptor == gabaa && out_degree > 4',
    note: 'Cells driving GABA-A synapses onto five or more targets.',
  },
  {
    text: '!! selected == true && voltage > -55',
    note: 'Depolarised cells that are not in the selection.',
  },
];

/* ------------------------------------------------------------------ rates -- */

interface RateSnapshot {
  /** Firing rate in Hz, parallel to the slots it was sampled for. */
  values: Float32Array;
  /** The exact list these values were read for, compared by identity. */
  slots: readonly number[];
  version: number;
}

const NO_RATES: RateSnapshot = { values: EMPTY_RATES, slots: EMPTY_SLOTS, version: 0 };

/**
 * Poll the firing rate of a fixed set of slots.
 *
 * Only the mounted rows are sampled, and a sample identical to the last one is
 * dropped rather than published — a paused simulation must not re-render two
 * hundred rows ten times a second. A slot outside the live buffers reads as NaN,
 * which prints as a dash rather than as a measured zero.
 *
 * The slot list is carried alongside the values and checked on the way out. A
 * new answer re-renders the rows one paint before the effect can resample them,
 * and without that check row `i` of the new match set would be labelled with the
 * rate of row `i` of the old one — a real number against the wrong cell, which
 * is worse than no number at all.
 */
function useSlotRates(slots: readonly number[], hz: number, enabled: boolean): Float32Array {
  const [snapshot, setSnapshot] = useState<RateSnapshot>(NO_RATES);
  const store = useRef<Float32Array>(EMPTY_RATES);

  useEffect(() => {
    if (!enabled) return;

    // The first sample for a given slot list always publishes, even when every
    // rate happens to be unchanged: the values still have to be re-labelled to
    // these slots before the rows are allowed to read them.
    let first = true;

    const sample = () => {
      const { neurons } = getEngine().buffers;
      const count = slots.length;
      let values = store.current;
      let changed = first;
      if (values.length !== count) {
        values = new Float32Array(count);
        store.current = values;
        changed = true;
      }
      for (let i = 0; i < count; i += 1) {
        const slot = slots[i];
        const next = slot >= 0 && slot < neurons.count ? neurons.rate[slot] : Number.NaN;
        // Object.is rather than !==, so an unresolved slot does not read as a
        // change on every single tick.
        if (!Object.is(next, values[i])) {
          values[i] = next;
          changed = true;
        }
      }
      first = false;
      if (!changed) return;
      setSnapshot((previous) => ({ values, slots, version: previous.version + 1 }));
    };

    sample();
    const id = setInterval(sample, Math.max(16, Math.round(1000 / hz)));
    return () => clearInterval(id);
  }, [slots, hz, enabled]);

  return snapshot.slots === slots ? snapshot.values : EMPTY_RATES;
}

/* ------------------------------------------------------------------ panel -- */

/** A finished evaluation, kept whole so the count and the rows never disagree. */
interface Evaluated {
  source: string;
  error: QueryError | null;
  result: QueryResult | null;
  slots: readonly number[];
}

interface RowModel {
  /** Index into the sampled-rate array, which is parallel to the mounted slots. */
  at: number;
  id: NeuronId;
  label: string;
  color: string;
  model: string;
  inhibitory: boolean;
}

export interface QueryPanelProps {
  /** Hidden entirely when false, so a host can mount it permanently. */
  open?: boolean;
  /** Supplying this renders a close control in the header. */
  onClose?: () => void;
  /** Replaces the default left-edge docking entirely. */
  className?: string;
}

/**
 * The structured query language, as a panel.
 *
 * A connectome stops being browsable somewhere around a few thousand cells, and
 * what replaces browsing is asking. This is the front end of `lib/query`: the
 * same field and operator tables the parser type-checks against drive the
 * autocomplete and the reference here, so nothing this panel offers can be
 * something the language then rejects.
 *
 * Parsing and evaluation are debounced off the keystroke and yielded a frame, so
 * typing never waits on a pass over the network; the previous answer stays on
 * screen, dimmed, until the next one lands.
 */
export function QueryPanel({ open = true, onClose, className }: QueryPanelProps) {
  const neurons = useEditor((s) => s.circuit.neurons);
  const synapses = useEditor((s) => s.circuit.synapses);
  const populations = useEditor((s) => s.circuit.populations);
  const dimUnselected = useEditor((s) => s.circuit.render.dimUnselected);
  const selection = useEditor((s) => s.selection);
  const select = useEditor((s) => s.select);
  const setRenderSettings = useEditor((s) => s.setRenderSettings);

  const [text, setText] = useState('');
  const [caret, setCaret] = useState(0);
  const [evaluated, setEvaluated] = useState<Evaluated | null>(null);
  const [pending, setPending] = useState(false);
  const [revision, setRevision] = useState(0);

  const [menuOpen, setMenuOpen] = useState(false);
  const [active, setActive] = useState(0);

  const [libraryOpen, setLibraryOpen] = useState(true);
  const [saved, setSaved] = useState<readonly SavedQuery[]>([]);
  const [savedError, setSavedError] = useState<string | null>(null);
  const [naming, setNaming] = useState(false);
  const [draftName, setDraftName] = useState('');

  const domId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const frameRef = useRef(0);
  const signatureRef = useRef('');
  const caretRef = useRef<number | null>(null);

  /* ------------------------------------------------------------ vocabulary */

  const vocabulary = useMemo<Vocabulary>(() => {
    const names: string[] = [];
    for (const population of populations) {
      if (population.name !== '' && !names.includes(population.name)) names.push(population.name);
    }
    const labels = new Set<string>();
    for (const neuron of neurons) {
      if (neuron.label === '' || labels.has(neuron.label)) continue;
      labels.add(neuron.label);
      if (labels.size >= MAX_LABEL_VALUES) break;
    }
    return { populations: names.sort(), labels: [...labels].sort() };
  }, [neurons, populations]);

  const samples = useMemo<readonly Sample[]>(() => {
    const list: Sample[] = [];
    const population = vocabulary.populations[0];
    if (population !== undefined) {
      list.push({
        text: `population == ${quoteValue(population)}`,
        note: `Every cell in ${population}.`,
      });
    }
    const label = vocabulary.labels[0];
    if (label !== undefined) {
      list.push({ text: `^^ ${quoteValue(label)}`, note: `Cells presynaptic to ${label}.` });
      list.push({ text: `~c ${quoteValue(label)}`, note: `Cells wired the way ${label} is.` });
    }
    return [...list, ...STATIC_SAMPLES];
  }, [vocabulary]);

  /* -------------------------------------------------------------- settings */

  useEffect(() => {
    let cancelled = false;
    getSetting<unknown>(SAVED_KEY, [])
      .then((raw) => {
        if (!cancelled) setSaved(readSaved(raw));
      })
      .catch((cause: unknown) => {
        if (!cancelled) setSavedError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // The list is published before the write resolves: a save that fails because
  // the browser has no IndexedDB must still leave the query usable for this
  // session, with the reason stated rather than the entry silently vanishing.
  const persist = useCallback((next: readonly SavedQuery[]) => {
    setSaved(next);
    setSetting(SAVED_KEY, next)
      .then(() => setSavedError(null))
      .catch((cause: unknown) => {
        setSavedError(cause instanceof Error ? cause.message : String(cause));
      });
  }, []);

  /* ------------------------------------------------------------ evaluation */

  // The engine is loaded by an effect in an ancestor, which commits after this
  // one, and a structural edit reallocates the buffers under whatever slots the
  // last answer named. Polling the signature is what keeps the two in step
  // regardless of that ordering.
  useEffect(() => {
    if (!open) return;
    const poll = () => {
      const signature = graphSignature(getEngine().buffers);
      if (signature === signatureRef.current) return;
      signatureRef.current = signature;
      setRevision((value) => value + 1);
    };
    poll();
    const id = setInterval(poll, SIGNATURE_POLL_MS);
    return () => clearInterval(id);
  }, [open]);

  /**
   * Selection only re-triggers evaluation for a query that reads it.
   *
   * `selected` is a queryable field, so a query naming it goes stale the moment
   * anything is picked — including by clicking a row in this very list. Every
   * other query is independent of the selection, and re-evaluating those on each
   * click would charge a pass over the connectome for a pointer event.
   */
  const selectionKey = text.includes('selected') ? selection : EMPTY_IDS;

  // Depending on the three collections rather than on `circuit` is deliberate:
  // the document is republished on every orbit frame with the same entity arrays,
  // and re-running the query per camera frame would spend a pass over the
  // connectome sixty times a second on a view change.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setPending(true);
    const timer = setTimeout(() => {
      // One frame of slack, so the keystroke that scheduled this has painted
      // before the blocking pass starts.
      frameRef.current = requestAnimationFrame(() => {
        if (cancelled) return;
        const circuit = useEditor.getState().circuit;
        const run = runQuery(text, getEngine().buffers, circuit);
        setEvaluated(
          run.ok
            ? { source: text, error: null, result: run.result, slots: matchedSlots(run.result) }
            : { source: text, error: run.error, result: null, slots: EMPTY_SLOTS },
        );
        setPending(false);
      });
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      cancelAnimationFrame(frameRef.current);
    };
  }, [text, open, revision, neurons, synapses, populations, selectionKey]);

  /* ----------------------------------------------------------- completions */

  const completion = useMemo(
    () => (open ? completionAt(text, caret, vocabulary) : null),
    [open, text, caret, vocabulary],
  );

  const suggestions = completion?.suggestions ?? NO_SUGGESTIONS;
  const showMenu = menuOpen && suggestions.length > 0;
  const activeIndex = Math.min(active, Math.max(0, suggestions.length - 1));

  // Restoring the caret has to happen after the new value is on the element, or
  // the browser drops it at the end of the text rather than after the token that
  // was just completed.
  useLayoutEffect(() => {
    const at = caretRef.current;
    if (at === null) return;
    caretRef.current = null;
    const input = inputRef.current;
    if (input === null) return;
    input.focus();
    input.setSelectionRange(at, at);
    setCaret(at);
  }, [text]);

  useLayoutEffect(() => {
    if (!showMenu) return;
    listRef.current?.children.item(activeIndex)?.scrollIntoView({ block: 'nearest' });
  }, [showMenu, activeIndex]);

  // The mirror carries the error underline and sits behind the input, so it has
  // to follow the input's horizontal scroll exactly or the underline drifts off
  // the token it belongs to.
  useLayoutEffect(() => {
    const input = inputRef.current;
    const mirror = mirrorRef.current;
    if (input !== null && mirror !== null) mirror.scrollLeft = input.scrollLeft;
  }, [text, caret]);

  const accept = useCallback(
    (index: number) => {
      if (completion === null) return;
      const suggestion = completion.suggestions[index];
      if (suggestion === undefined) return;
      const insert = `${suggestion.insert} `;
      caretRef.current = completion.from + insert.length;
      setText(text.slice(0, completion.from) + insert + text.slice(completion.to));
      setActive(0);
      setMenuOpen(true);
    },
    [completion, text],
  );

  /** Replace the whole query, from a sample, a saved entry or the clear button. */
  const load = useCallback((source: string) => {
    setText(source);
    setCaret(source.length);
    setActive(0);
    setMenuOpen(false);
  }, []);

  /* ------------------------------------------------------------- selection */

  const idsOf = useCallback((slots: readonly number[]): NeuronId[] => {
    // Read through the store rather than a captured array, so an action fired
    // after an edit resolves against the document as it is now.
    const source = useEditor.getState().circuit.neurons;
    const ids: NeuronId[] = [];
    for (const slot of slots) {
      const neuron: Neuron | undefined = source[slot];
      if (neuron !== undefined) ids.push(neuron.id);
    }
    return ids;
  }, []);

  const result = evaluated?.result ?? null;
  const matches = evaluated?.slots ?? EMPTY_SLOTS;
  const hasMatches = matches.length > 0;

  const selectMatches = useCallback(() => {
    if (matches.length === 0) return;
    select(idsOf(matches));
  }, [idsOf, matches, select]);

  const addMatches = useCallback(() => {
    if (matches.length === 0) return;
    select(idsOf(matches), true);
  }, [idsOf, matches, select]);

  const selectComplement = useCallback(() => {
    if (result === null) return;
    const complement: number[] = [];
    for (let slot = 0; slot < result.total; slot += 1) {
      if (result.mask[slot] === 0) complement.push(slot);
    }
    select(idsOf(complement));
  }, [idsOf, result, select]);

  const isolated = dimUnselected <= ISOLATE_DIM;

  const isolate = useCallback(() => {
    if (matches.length === 0) return;
    // The renderer fades everything outside the selection, so isolating is
    // exactly "select the matches, then turn that fade up".
    select(idsOf(matches));
    setRenderSettings({ dimUnselected: ISOLATE_DIM });
  }, [idsOf, matches, select, setRenderSettings]);

  const restoreDimming = useCallback(() => {
    setRenderSettings({ dimUnselected: DEFAULT_RENDER_SETTINGS.dimUnselected });
  }, [setRenderSettings]);

  /* ------------------------------------------------------------------ rows */

  const visible = useMemo(
    () => (matches.length > MAX_ROWS ? matches.slice(0, MAX_ROWS) : matches),
    [matches],
  );

  const rates = useSlotRates(visible, RATE_HZ, open && visible.length > 0);

  const rows = useMemo(() => {
    const list: RowModel[] = [];
    visible.forEach((slot, at) => {
      const neuron: Neuron | undefined = neurons[slot];
      if (neuron === undefined) return;
      list.push({
        at,
        id: neuron.id,
        label: neuron.label !== '' ? neuron.label : neuron.id,
        color: identityColorHex(neuron.morphology.seed),
        model: MODEL_SHORT[neuron.params.kind],
        inhibitory: neuron.polarity === 'inhibitory',
      });
    });
    return list;
  }, [visible, neurons]);

  const selected = useMemo(() => new Set<string>(selection), [selection]);

  const pick = useCallback(
    (id: NeuronId, additive: boolean) => {
      select([id], additive);
    },
    [select],
  );

  /* -------------------------------------------------------------- keyboard */

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (showMenu) {
        if (event.key === 'ArrowDown') {
          event.preventDefault();
          setActive((index) => (index + 1) % suggestions.length);
          return;
        }
        if (event.key === 'ArrowUp') {
          event.preventDefault();
          setActive((index) => (index - 1 + suggestions.length) % suggestions.length);
          return;
        }
        if (event.key === 'Tab' || event.key === 'Enter') {
          event.preventDefault();
          accept(activeIndex);
          return;
        }
        if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          setMenuOpen(false);
          return;
        }
      }
      if (event.key === 'Enter') {
        event.preventDefault();
        selectMatches();
        return;
      }
      // Escape is only swallowed when it did something here; otherwise the
      // workspace's own Escape — clear the selection — has to keep working.
      if (event.key === 'Escape' && text !== '') {
        event.preventDefault();
        event.stopPropagation();
        load('');
      }
    },
    [accept, activeIndex, load, selectMatches, showMenu, suggestions.length, text],
  );

  const onChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setText(value);
    setCaret(event.target.selectionStart ?? value.length);
    setActive(0);
    setMenuOpen(true);
  }, []);

  const onCaretMove = useCallback((event: React.SyntheticEvent<HTMLInputElement>) => {
    setCaret(event.currentTarget.selectionStart ?? event.currentTarget.value.length);
  }, []);

  const syncMirror = useCallback(() => {
    const input = inputRef.current;
    const mirror = mirrorRef.current;
    if (input !== null && mirror !== null) mirror.scrollLeft = input.scrollLeft;
  }, []);

  /* ---------------------------------------------------------------- saving */

  const commitSave = useCallback(() => {
    const name = draftName.trim();
    if (name === '' || text.trim() === '') return;
    const entry: SavedQuery = { name, text, createdAt: Date.now() };
    persist([entry, ...saved.filter((query) => query.name !== name)].slice(0, MAX_SAVED));
    setDraftName('');
    setNaming(false);
  }, [draftName, persist, saved, text]);

  const remove = useCallback(
    (name: string) => {
      persist(saved.filter((query) => query.name !== name));
    },
    [persist, saved],
  );

  useEffect(() => {
    if (naming) nameRef.current?.focus();
  }, [naming]);

  if (!open) return null;

  /* ---------------------------------------------------------------- render */

  const error = evaluated?.error ?? null;
  const total = result?.total ?? 0;
  const count = result?.count ?? 0;
  const stale = pending || (evaluated !== null && evaluated.source !== text);
  const blank = text.trim() === '';
  const everything = result !== null && total > 0 && count === total && !blank;
  const elided = matches.length - rows.length;
  const warnings = result?.warnings ?? [];
  const underline = describeUnderline(text, error);

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', className ?? PLACEMENT)}>
      <PanelHeader
        title="Query"
        subtitle={`${grouped(neurons.length)} cells · ${grouped(synapses.length)} synapses`}
        icon={<Braces size={14} />}
        actions={
          <>
            <Tooltip content="Samples, saved queries and the operator reference">
              <IconButton
                label="Toggle the query library"
                size="sm"
                variant={libraryOpen ? 'secondary' : 'ghost'}
                aria-pressed={libraryOpen}
                onClick={() => setLibraryOpen((value) => !value)}
              >
                <BookOpen size={13} />
              </IconButton>
            </Tooltip>
            {onClose ? (
              <IconButton label="Close the query panel" size="sm" onClick={onClose}>
                <X size={13} />
              </IconButton>
            ) : null}
          </>
        }
      />

      {/* ------------------------------------------------------------- input */}

      <div className="relative shrink-0 border-b border-hairline">
        <div className="flex items-center gap-1.5 px-2.5 py-[7px]">
          <Braces size={12} aria-hidden className="shrink-0 text-ink-faint" />
          <div className="relative min-w-0 flex-1">
            {/* Behind the input, glyph for glyph, carrying nothing but the
                underline. A text input cannot draw inside its own value, so a
                mirrored layer is how every editor does this. */}
            <div
              ref={mirrorRef}
              aria-hidden
              className="pointer-events-none absolute inset-0 overflow-hidden font-mono text-[11.5px] leading-[18px] whitespace-pre text-transparent"
            >
              {underline === null ? (
                text
              ) : (
                <>
                  {underline.before}
                  <span className="underline decoration-danger decoration-wavy decoration-2 underline-offset-[3px]">
                    {underline.span}
                  </span>
                  {underline.after}
                </>
              )}
            </div>
            <input
              ref={inputRef}
              type="text"
              role="combobox"
              aria-expanded={showMenu}
              aria-controls={showMenu ? `${domId}-suggestions` : undefined}
              aria-activedescendant={showMenu ? `${domId}-option-${activeIndex}` : undefined}
              aria-autocomplete="list"
              aria-label="Cell query"
              aria-invalid={error !== null}
              spellCheck={false}
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="off"
              placeholder="polarity == inhibitory && rate > 5"
              value={text}
              onChange={onChange}
              onSelect={onCaretMove}
              onKeyDown={onKeyDown}
              onScroll={syncMirror}
              onFocus={() => setMenuOpen(true)}
              onBlur={() => setMenuOpen(false)}
              className={cn(
                'relative block w-full border-0 bg-transparent p-0 font-mono text-[11.5px]',
                'leading-[18px] text-ink outline-none placeholder:text-ink-faint/70',
              )}
            />
          </div>
          {text !== '' ? (
            <IconButton
              label="Clear the query"
              size="sm"
              className="size-5"
              onClick={() => load('')}
            >
              <X size={11} />
            </IconButton>
          ) : null}
        </div>

        {showMenu ? (
          <div className="nf-glass-raised absolute top-full right-1.5 left-1.5 z-40 mt-1 rounded-control py-1">
            <div
              id={`${domId}-suggestions`}
              role="listbox"
              aria-label="Query completions"
              ref={listRef}
              className="max-h-[228px] overflow-y-auto overscroll-contain"
            >
              {suggestions.map((suggestion, index) => (
                <div
                  key={`${suggestion.kind}:${suggestion.insert}`}
                  id={`${domId}-option-${index}`}
                  role="option"
                  aria-selected={index === activeIndex}
                  // Pointer-down rather than click: the input's blur closes the
                  // menu, and a click would never land on a row that had gone.
                  onPointerDown={(event) => {
                    event.preventDefault();
                    accept(index);
                  }}
                  onPointerEnter={() => setActive(index)}
                  className={cn(
                    'flex cursor-pointer flex-col gap-0.5 px-2 py-1',
                    index === activeIndex ? 'bg-accent/12' : 'hover:bg-white/[0.05]',
                  )}
                >
                  <span className="flex items-baseline gap-1.5">
                    <span
                      aria-hidden
                      className={cn(
                        'w-[24px] shrink-0 text-[8.5px] font-semibold tracking-[0.08em] uppercase',
                        suggestion.kind === 'field'
                          ? 'text-accent'
                          : suggestion.kind === 'operator'
                            ? 'text-secondary'
                            : 'text-ink-faint',
                      )}
                    >
                      {suggestion.kind === 'field' ? 'fld' : suggestion.kind === 'operator' ? 'op' : 'val'}
                    </span>
                    <span className="nf-numeric min-w-0 truncate text-[11px] text-ink">
                      {suggestion.title}
                    </span>
                    {suggestion.detail !== '' ? (
                      <span className="nf-numeric ml-auto shrink-0 text-[9.5px] text-ink-faint">
                        {suggestion.detail}
                      </span>
                    ) : null}
                  </span>
                  <span className="truncate pl-[30px] text-[10px] leading-snug text-ink-faint">
                    {suggestion.hint}
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-1 flex items-center gap-1 border-t border-hairline px-2 pt-1 text-[9.5px] text-ink-faint">
              <Kbd keys="Tab" size="sm" />
              <span className="mr-1">accept</span>
              <Kbd keys="ArrowUp" size="sm" />
              <Kbd keys="ArrowDown" size="sm" />
              <span className="mr-1">move</span>
              <Kbd keys="Escape" size="sm" />
              <span>dismiss</span>
            </div>
          </div>
        ) : null}
      </div>

      {/* ------------------------------------------------------------ status */}

      <div className="flex min-h-[26px] shrink-0 items-start gap-1.5 border-b border-hairline px-2.5 py-1 text-[10.5px] leading-snug">
        {error !== null ? (
          <>
            <TriangleAlert size={11} aria-hidden className="mt-[3px] shrink-0 text-danger" />
            <span className="min-w-0 flex-1 text-danger">{error.message}</span>
            <span className="nf-numeric mt-px shrink-0 text-[9.5px] text-ink-faint">
              col {error.offset + 1}
            </span>
          </>
        ) : result === null ? (
          <span className="flex items-center gap-1.5 text-ink-faint">
            <Spinner size={11} />
            Reading the query…
          </span>
        ) : (
          <>
            <span className={cn('min-w-0 flex-1', stale ? 'text-ink-faint' : 'text-ink-muted')}>
              {blank ? (
                <>
                  Whole population — <span className="nf-numeric text-ink">{grouped(total)}</span>{' '}
                  cells. Add a clause, or load a sample below.
                </>
              ) : count === 0 ? (
                'No cell matches this query.'
              ) : everything ? (
                <>
                  Every cell matches — all{' '}
                  <span className="nf-numeric text-ink">{grouped(total)}</span>.
                </>
              ) : (
                <>
                  <span className="nf-numeric text-ink">{grouped(count)}</span> of{' '}
                  <span className="nf-numeric">{grouped(total)}</span> cells matched.
                </>
              )}
            </span>
            {stale ? (
              <Spinner size={11} className="mt-px shrink-0" />
            ) : (
              <Tooltip content="Time taken by the last pass over the running network" side="left">
                <Badge variant="outline" size="sm" numeric tabIndex={0} className="mt-px shrink-0">
                  {fixed(result.computeMs, 1)} ms
                </Badge>
              </Tooltip>
            )}
          </>
        )}
      </div>

      {warnings.length > 0 ? (
        <div className="shrink-0 border-b border-hairline px-2.5 py-1">
          {warnings.map((warning) => (
            <p key={warning} className="text-[10px] leading-snug text-warning">
              {warning}
            </p>
          ))}
        </div>
      ) : null}

      {/* ----------------------------------------------------------- library */}

      {libraryOpen ? (
        <div className="max-h-[46%] shrink-0 overflow-y-auto overscroll-contain border-b border-hairline">
          <PanelSection
            label="Samples"
            flush
            aside={<span className="text-[9.5px] text-ink-faint">click to load</span>}
          >
            <div className="flex flex-col">
              {samples.map((sample) => (
                <button
                  key={sample.text}
                  type="button"
                  onClick={() => load(sample.text)}
                  className="-mx-1 flex flex-col gap-px rounded-control px-1 py-1 text-left transition-colors hover:bg-white/[0.05] focus-visible:bg-white/[0.07]"
                >
                  <span className="nf-numeric truncate text-[10.5px] text-accent">
                    {sample.text}
                  </span>
                  <span className="truncate text-[10px] text-ink-faint">{sample.note}</span>
                </button>
              ))}
            </div>
          </PanelSection>

          <PanelSection
            label="Saved"
            aside={
              <Tooltip content="Save the current query">
                <IconButton
                  label="Save the current query"
                  size="sm"
                  className="size-5"
                  disabled={text.trim() === ''}
                  onClick={() => setNaming(true)}
                >
                  <BookmarkPlus size={12} />
                </IconButton>
              </Tooltip>
            }
          >
            {naming ? (
              <div className="mb-1 flex items-center gap-1.5">
                <input
                  ref={nameRef}
                  type="text"
                  value={draftName}
                  spellCheck={false}
                  autoComplete="off"
                  placeholder="Name this query…"
                  aria-label="Name for the saved query"
                  onChange={(event) => setDraftName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      commitSave();
                    } else if (event.key === 'Escape') {
                      event.preventDefault();
                      event.stopPropagation();
                      setNaming(false);
                      setDraftName('');
                    }
                  }}
                  className={cn(
                    'h-6 min-w-0 flex-1 rounded-control border border-hairline bg-white/[0.04] px-1.5',
                    'text-[11px] text-ink outline-none placeholder:text-ink-faint focus:border-accent/45',
                  )}
                />
                <Button
                  size="sm"
                  variant="primary"
                  className="h-6 px-2 text-[10.5px]"
                  disabled={draftName.trim() === ''}
                  onClick={commitSave}
                >
                  Save
                </Button>
              </div>
            ) : null}

            {savedError !== null ? (
              <p className="mb-1 text-[10px] leading-snug text-warning">
                Saved queries cannot be written in this browser: {savedError}
              </p>
            ) : null}

            {saved.length === 0 ? (
              <p className="text-[10.5px] text-ink-faint">
                Nothing saved yet. Write a query, then use the bookmark to keep it.
              </p>
            ) : (
              <div className="flex flex-col">
                {saved.map((query) => (
                  <div
                    key={query.name}
                    className="-mx-1 flex items-center gap-1.5 rounded-control px-1 transition-colors hover:bg-white/[0.05]"
                  >
                    <button
                      type="button"
                      onClick={() => load(query.text)}
                      title={query.text}
                      className="flex min-w-0 flex-1 flex-col gap-px py-1 text-left"
                    >
                      <span className="truncate text-[11px] text-ink">{query.name}</span>
                      <span className="nf-numeric truncate text-[10px] text-ink-faint">
                        {query.text}
                      </span>
                    </button>
                    <IconButton
                      label={`Delete the saved query ${query.name}`}
                      size="sm"
                      className="size-5 shrink-0"
                      onClick={() => remove(query.name)}
                    >
                      <Trash2 size={11} />
                    </IconButton>
                  </div>
                ))}
              </div>
            )}
          </PanelSection>

          <PanelSection
            label="Operators"
            collapsible
            defaultOpen={false}
            aside={
              <span className="nf-numeric text-[9.5px] text-ink-faint">
                {QUERY_OPERATORS.length}
              </span>
            }
          >
            <p className="mb-1 text-[10px] leading-snug text-ink-faint">
              Both spellings parse identically. Group terms with{' '}
              <span className="nf-numeric text-ink-muted">{'{{ }}'}</span>;{' '}
              <span className="nf-numeric text-ink-muted">&amp;&amp;</span> binds tighter than{' '}
              <span className="nf-numeric text-ink-muted">||</span>.
            </p>
            {OPERATOR_SECTIONS.map((section) => (
              <div key={section.group} className="mb-1.5 last:mb-0">
                <span className="mb-0.5 block text-[9px] font-semibold tracking-[0.09em] text-ink-faint uppercase">
                  {GROUP_LABELS[section.group]}
                </span>
                {section.operators.map((spec) => (
                  <button
                    key={spec.id}
                    type="button"
                    onClick={() => load(spec.example)}
                    title={`Load the example: ${spec.example}`}
                    className="-mx-1 flex w-[calc(100%+0.5rem)] flex-col gap-px rounded-control px-1 py-[3px] text-left transition-colors hover:bg-white/[0.05] focus-visible:bg-white/[0.07]"
                  >
                    <span className="flex items-baseline gap-1.5">
                      <span className="nf-numeric w-[42px] shrink-0 text-[10.5px] text-secondary">
                        {spec.symbol}
                      </span>
                      <span className="nf-numeric min-w-0 truncate text-[10px] text-ink-muted">
                        {spec.bracket}
                      </span>
                      <span className="ml-auto shrink-0 text-[9.5px] text-ink-faint">
                        {spec.label}
                      </span>
                    </span>
                    <span className="pl-[48px] text-[10px] leading-snug text-ink-faint">
                      {spec.description}
                    </span>
                  </button>
                ))}
              </div>
            ))}
          </PanelSection>

          <PanelSection
            label="Fields"
            collapsible
            defaultOpen={false}
            aside={
              <span className="nf-numeric text-[9.5px] text-ink-faint">{QUERY_FIELDS.length}</span>
            }
          >
            {QUERY_FIELDS.map((field) => (
              <div key={field.name} className="flex flex-col gap-px py-[2px]">
                <span className="flex items-baseline gap-1.5">
                  <span className="nf-numeric min-w-0 truncate text-[10.5px] text-accent">
                    {field.name}
                  </span>
                  <span className="ml-auto shrink-0 text-[9.5px] text-ink-faint">
                    {field.type === 'numeric' && field.unit !== ''
                      ? `numeric · ${field.unit}`
                      : field.type}
                  </span>
                </span>
                <span className="text-[10px] leading-snug text-ink-faint">{field.description}</span>
              </div>
            ))}
          </PanelSection>
        </div>
      ) : null}

      {/* ----------------------------------------------------------- actions */}

      <div className="flex h-8 shrink-0 items-center gap-1 border-b border-hairline px-2">
        {selection.length > 0 ? (
          <span className="nf-numeric shrink-0 px-0.5 text-[10px] text-accent">
            {grouped(selection.length)} sel
          </span>
        ) : null}
        <div className="flex-1" />
        <Tooltip content="Replace the selection with every match" shortcut="Enter" side="top">
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-1.5 text-[10.5px]"
            icon={<SquareDashedMousePointer size={11} />}
            disabled={!hasMatches}
            onClick={selectMatches}
          >
            Select
          </Button>
        </Tooltip>
        <Tooltip content="Add every match to the current selection" side="top">
          <IconButton
            label="Add the matches to the selection"
            size="sm"
            className="size-6"
            disabled={!hasMatches}
            onClick={addMatches}
          >
            <ListPlus size={12} />
          </IconButton>
        </Tooltip>
        <Tooltip content="Select every cell this query did not match" side="top">
          <IconButton
            label="Select the cells that did not match"
            size="sm"
            className="size-6"
            disabled={total === 0}
            onClick={selectComplement}
          >
            <CornerDownLeft size={12} className="rotate-180" />
          </IconButton>
        </Tooltip>
        {isolated ? (
          <Tooltip content="Bring the unmatched cells back to their normal opacity" side="top">
            <IconButton
              label="Restore the dimming of unselected cells"
              size="sm"
              className="size-6"
              variant="secondary"
              onClick={restoreDimming}
            >
              <Eye size={12} />
            </IconButton>
          </Tooltip>
        ) : (
          <Tooltip content="Select the matches and fade everything else in the scene" side="top">
            <IconButton
              label="Isolate the matches in the scene"
              size="sm"
              className="size-6"
              disabled={!hasMatches}
              onClick={isolate}
            >
              <EyeOff size={12} />
            </IconButton>
          </Tooltip>
        )}
      </div>

      <div
        aria-hidden
        className="flex h-5 shrink-0 items-center gap-2 border-b border-hairline px-3 text-[9px] font-semibold tracking-[0.09em] text-ink-faint uppercase"
      >
        <span className="size-2 shrink-0" />
        <span className="min-w-0 flex-1">Cell</span>
        <span className="w-9 shrink-0 text-right">Model</span>
        <span className="w-2 shrink-0 text-center">P</span>
        <span className="w-10 shrink-0 text-right">Hz</span>
      </div>

      {/* ----------------------------------------------------------- results */}

      <ScrollArea className="min-h-0 flex-1">
        {rows.length === 0 ? (
          <EmptyState
            compact
            icon={
              error !== null ? (
                <TriangleAlert size={14} className="text-danger" />
              ) : (
                <Braces size={14} />
              )
            }
            title={
              error !== null
                ? 'The query could not be read'
                : total === 0
                  ? 'Nothing to query'
                  : 'No matches'
            }
            description={
              error !== null
                ? `${error.message}. The underlined span in the box above is where it went wrong.`
                : total === 0
                  ? 'This circuit has no cells yet. Place some, or build a population, and every attribute becomes queryable.'
                  : `No cell satisfies ${echo(text)}. Loosen a clause, or use ~= for a name you are not sure how to spell.`
            }
          />
        ) : (
          <div className="flex flex-col py-0.5">
            {rows.map((row) => (
              <ResultRow
                key={row.id}
                id={row.id}
                label={row.label}
                color={row.color}
                model={row.model}
                inhibitory={row.inhibitory}
                rate={row.at < rates.length ? rates[row.at] : Number.NaN}
                selected={selected.has(row.id)}
                onSelect={pick}
              />
            ))}
            {elided > 0 ? (
              <p className="px-3 py-1.5 text-[10px] leading-snug text-ink-faint">
                +{grouped(elided)} more matched. Every action above still covers all{' '}
                {grouped(matches.length)}.
              </p>
            ) : null}
          </div>
        )}
      </ScrollArea>
    </Panel>
  );
}

/* ------------------------------------------------------------------ pieces -- */

/**
 * Split the source around the span an error names.
 *
 * An error at the very end of the input has zero width and nothing to sit under,
 * so a single space is underlined there instead — otherwise the class of error
 * that is hardest to spot, the missing operand, would be the one with no marker
 * at all.
 */
function describeUnderline(
  source: string,
  error: QueryError | null,
): { before: string; span: string; after: string } | null {
  if (error === null) return null;
  const start = Math.max(0, Math.min(error.offset, source.length));
  const end = Math.min(source.length, start + Math.max(1, error.length));
  const span = source.slice(start, end);
  return {
    before: source.slice(0, start),
    span: span === '' ? ' ' : span,
    after: source.slice(end),
  };
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
 * One matched cell. Memoised because the list re-renders at the poll rate and
 * only the rows whose firing rate actually moved need to be reconciled.
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
      {/* The swatch is the hue the cell is drawn in, so a row here and a glyph in
          the scene are unmistakably the same object. */}
      <span
        aria-hidden
        className="size-2 shrink-0 rounded-[2px]"
        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}66` }}
      />
      <span
        className={cn(
          'min-w-0 flex-1 truncate text-[11px]',
          selected ? 'text-ink' : 'text-ink-muted',
        )}
      >
        {label}
      </span>
      <span className="nf-numeric w-9 shrink-0 text-right text-[9.5px] text-ink-faint">{model}</span>
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
        {Number.isFinite(rate) ? fixed(rate, 1) : '—'}
      </span>
    </button>
  );
});
