import { NEURON_MODEL_LABELS, defaultParams } from '@neuroforge/shared';
import type {
  Circuit,
  ConnectivityRule,
  MorphologyArchetype,
  NeuronModelKind,
  NeuronParams,
  NeuronPolarity,
  PopulationLayout,
  ReceptorKind,
  StimulusPattern,
  Vec3,
} from '@neuroforge/shared';
import { clamp, hashText } from './coerce';
import {
  ARCHETYPE_WORDS,
  FIRING_WORDS,
  GENERIC_REGION,
  HEAD_NOUNS,
  LAYOUT_WORDS,
  MODEL_WORDS,
  PHRASE_BOUNDARIES,
  POLARITY_WORDS,
  RECEPTOR_WORDS,
  REGION_WORDS,
  REMOVAL_VERBS,
  RHYTHM_BANDS,
  RHYTHM_WORDS,
  bandForFrequency,
} from './lexicon';
import type { AnalyticLayout, FiringPattern, RegionProfile } from './lexicon';
import {
  IZHIKEVICH_ONLY_PATTERNS,
  interneuronParams,
  izhikevichParams,
  kindOnlyParams,
  layoutFor,
  layoutRadius,
  sustainedDrive,
} from './presets';
import { MAX_POPULATION_SIZE } from './schema';
import { PromptScan, isCountToken, parseCount, titleCase } from './text';
import type { AiPlan, CircuitAction, NamedProjectionSpec, PopulationSpec } from './types';

/** Default population sizes when the prompt gives no count. */
const DEFAULT_EXCITATORY_SIZE = 100;
/** Cortex runs at roughly four excitatory cells per interneuron. */
const INHIBITORY_FRACTION = 0.25;
const GAP_BETWEEN_POPULATIONS = 8;

const SPARSER_WORDS = ['sparser', 'sparse', 'thinner', 'weaker', 'looser', 'sparsely'];
const DENSER_WORDS = ['denser', 'dense', 'stronger', 'tighter', 'thicker', 'densely'];
const FASTER_WORDS = ['faster', 'quicker', 'speed-up', 'fast'];
const SLOWER_WORDS = ['slower', 'slow-down', 'slow'];
const RATE_WORDS = ['fire', 'fires', 'firing', 'spike', 'spikes', 'spiking', 'rate', 'rates', 'active', 'activity'];
const CLEAR_VERBS = ['clear', 'reset', 'wipe', 'empty', 'erase', 'blank', 'start-over', 'from-scratch'];
const CLEAR_OBJECTS = ['circuit', 'network', 'everything', 'all', 'scene', 'canvas', 'board', 'workspace', 'it', 'this'];
/** Verbs that ask for cells that already exist to be rebuilt with another model. */
const MODEL_SWITCH_VERBS = ['switch', 'convert', 'change', 'turn', 'replace', 'swap'];
const SWITCH_VERBS = [...MODEL_SWITCH_VERBS, 'everything', 'every'];
/**
 * Prepositions that introduce the model a switch moves to. A phrase in this
 * position names a target model, never cells to build.
 */
const MODEL_SWITCH_GOVERNORS: ReadonlySet<string> = new Set(['to', 'into']);
const RECURRENCE_WORDS = ['recurrent', 'recurrently', 'recurrence', 'circuit', 'network', 'microcircuit', 'loop', 'attractor'];
const STIMULUS_TRIGGERS = [
  'stimulate', 'stimulates', 'stimulated', 'stimulating', 'stimulus', 'stimuli',
  'inject', 'injects', 'injected', 'injecting', 'injection',
  'input', 'inputs', 'drive', 'drives', 'driven', 'driving',
];
const STIMULUS_PATTERN_WORDS = ['poisson', 'step', 'ramp', 'sine', 'sinusoidal', 'pulse-train', 'constant', 'tonic', 'train', 'pulses'];
/** Words that sit between a verb and the noun phrase it governs. */
const DETERMINERS: ReadonlySet<string> = new Set([
  'a', 'all', 'an', 'any', 'both', 'each', 'every', 'her', 'his', 'its', 'my', 'our', 'some',
  'that', 'the', 'their', 'these', 'this', 'those', 'your',
]);
/**
 * Verbs that unambiguously ask for new neurons. `make` is deliberately absent:
 * it is causative at least as often as it is creative ("make them fire faster").
 */
const CREATION_VERBS: ReadonlySet<string> = new Set([
  'add', 'append', 'build', 'create', 'generate', 'give', 'include', 'insert', 'instantiate',
  'place', 'put', 'spawn',
]);
/**
 * Head nouns that name a container rather than the cells in it, and so hand the
 * count to a following "of" phrase: "a thalamic relay population of 300 cells".
 */
const CONTAINER_NOUNS: ReadonlySet<string> = new Set(['population', 'populations']);
/** Ceiling on "N populations of …", so a typo cannot fill the document. */
const MAX_REPEATED_POPULATIONS = 8;
/**
 * Words inside an already-normalised phrase. Safe to share: `String.match` with a
 * global pattern resets `lastIndex` itself.
 */
const WORD_RE = /[a-z0-9]+(?:[-'][a-z0-9]+)*/g;
/** Words that say an edit applies to the whole circuit rather than one population. */
const GLOBAL_SCOPE_WORDS = ['everything', 'every', 'all', 'both', 'each', 'entire', 'whole'];
/** Words joining conjuncts that share a governing verb. */
const COORDINATORS: ReadonlySet<string> = new Set(['and', 'or', 'plus', 'with', 'along']);
/**
 * Tokens that end a clause, so the search for a governing verb does not reach
 * back into a previous sentence and inherit its verb.
 */
const SENTENCE_ENDS: ReadonlySet<string> = new Set(['.', ';', '!', '?', 'then']);
/**
 * Match strength at which a countless phrase with no creation verb is read as
 * naming a population the document already holds. One agreeing attribute is
 * enough here — with neither a count nor a creation verb the phrase is far more
 * likely to be pointing at existing cells than asking for new ones.
 */
const REFERENCE_SCORE = 2;

/**
 * How much a word naming an attribute says about a population: it counts for the
 * candidate that has that attribute and just as much against one that does not.
 * Without the negative half, "purkinje cells" would match a basket population on
 * the strength of them both being inhibitory.
 */
function agreement<T>(named: T | undefined, actual: T): number {
  if (named === undefined) return 0;
  return named === actual ? 2 : -2;
}

interface KnownPopulation {
  name: string;
  polarity: NeuronPolarity;
  model: NeuronModelKind;
  size: number;
  archetype: MorphologyArchetype;
  /** Mean bias current across the population, in pA. */
  bias: number;
  /** True when this plan creates it rather than the document already holding it. */
  planned: boolean;
}

interface KnownProjection {
  name: string;
  sourceName: string;
  targetName: string;
  weightMean: number;
  delayMean: number;
  /** Non-null when this plan creates the projection and can still edit its spec. */
  spec: NamedProjectionSpec | null;
}

interface PopulationDraft {
  name: string;
  size: number;
  polarity: NeuronPolarity;
  model: NeuronModelKind;
  archetype: MorphologyArchetype;
  pattern: FiringPattern;
  params: Partial<NeuronParams> | null;
  layout: PopulationLayout;
}

interface ConnectRequest {
  sourceText: string;
  targetText: string;
  tail: string;
}

interface StimulusRequest {
  pattern: StimulusPattern;
  /** The phrase naming the population to drive, or null when the prompt named none. */
  targetText: string | null;
}

/**
 * Deterministic offline planner. Runs with no network access and no API key, and
 * is the reference behaviour the hosted models are prompted to imitate.
 */
export function planLocally(prompt: string, circuit: Circuit): AiPlan {
  return new Planner(prompt, circuit).run();
}

class Planner {
  private readonly scan: PromptScan;
  private readonly circuit: Circuit;
  private readonly seed: number;
  private readonly actions: CircuitAction[] = [];
  private readonly warnings: string[] = [];
  private readonly populations: KnownPopulation[] = [];
  private readonly projections: KnownProjection[] = [];
  private readonly created: KnownPopulation[] = [];
  private readonly reservedPopulationNames = new Set<string>();
  private readonly reservedProjectionNames = new Set<string>();
  /** Words recognised before population phrases are cut, and claimed after. */
  private readonly deferredClaims: string[] = [];
  /** Existing populations the prompt pointed at rather than asked to create. */
  private readonly focus: KnownPopulation[] = [];
  private focusUsed = false;
  private region: RegionProfile = GENERIC_REGION;
  private layoutOverride: AnalyticLayout | null = null;
  private cleared = false;
  private seedCounter = 0;
  /** Right-hand edge of everything placed so far, in world X. */
  private placementCursor = 0;

  constructor(prompt: string, circuit: Circuit) {
    this.scan = new PromptScan(prompt);
    this.circuit = circuit;
    this.seed = hashText(this.scan.text);
  }

  run(): AiPlan {
    this.detectClear();
    // Region and layout words are recognised now but claimed later: claiming them
    // here would cut short the backwards walk that collects a population's
    // modifiers, since that walk stops at the first token another pass has taken.
    this.detectRegion();
    this.detectLayout();
    if (!this.cleared) this.seedFromCircuit();
    const connections = this.collectConnectRequests();
    const stimulus = this.collectStimulusRequest();
    this.createPopulations();
    this.flushDeferredClaims();
    this.applyConnectRequests(connections);
    this.autoWire();
    this.switchModel();
    this.applyRhythm();
    this.applyDensity();
    this.applySpeed();
    this.applyStimulus(stimulus);
    this.reportOrphans();
    this.reportUnusedFocus();
    this.reportLeftovers();
    return { summary: this.buildSummary(), actions: this.actions, warnings: this.warnings };
  }

  // ---------------------------------------------------------------- utilities

  private nextSeed(): number {
    this.seedCounter += 1;
    return (Math.imul(this.seed ^ this.seedCounter, 0x27d4eb2d) >>> 0) % 0x7fffffff;
  }

  private warn(message: string): void {
    if (!this.warnings.includes(message)) this.warnings.push(message);
  }

  /** Reserve a display name so a draft made before it is emitted still cannot collide. */
  private reserve(base: string, taken: Set<string>, live: readonly { name: string }[]): string {
    const used = (name: string): boolean =>
      taken.has(name.toLowerCase()) || live.some((item) => item.name.toLowerCase() === name.toLowerCase());
    let candidate = base;
    for (let n = 2; used(candidate); n += 1) candidate = `${base} ${n}`;
    taken.add(candidate.toLowerCase());
    return candidate;
  }

  private uniquePopulationName(base: string): string {
    return this.reserve(base, this.reservedPopulationNames, this.populations);
  }

  private uniqueProjectionName(base: string): string {
    return this.reserve(base, this.reservedProjectionNames, this.projections);
  }

  private findProjection(sourceName: string, targetName: string): KnownProjection | null {
    return (
      this.projections.find(
        (p) => p.sourceName === sourceName && p.targetName === targetName,
      ) ?? null
    );
  }

  private largestOf(
    pool: readonly KnownPopulation[],
    polarity: NeuronPolarity,
  ): KnownPopulation | null {
    let best: KnownPopulation | null = null;
    for (const population of pool) {
      if (population.polarity !== polarity) continue;
      if (best === null || population.size > best.size) best = population;
    }
    return best;
  }

  private largest(polarity: NeuronPolarity): KnownPopulation | null {
    return this.largestOf(this.populations, polarity);
  }

  /**
   * The population an unqualified reference means. A population this plan just
   * created wins over one the document already held, because "drive them" in
   * "add 200 excitatory neurons and drive them" is about the new cells.
   */
  private subjectPopulation(): KnownPopulation | null {
    return (
      this.largestOf(this.created, 'excitatory') ??
      this.largestOf(this.created, 'inhibitory') ??
      this.largest('excitatory') ??
      this.largest('inhibitory')
    );
  }

  /** Claim every occurrence of `word` in the prompt. */
  private claimWord(word: string): void {
    for (let i = this.scan.indexOf(word); i !== -1; i = this.scan.indexOf(word, i + 1)) {
      this.scan.claim(i);
    }
  }

  private claimWords(words: readonly string[]): void {
    for (const word of words) this.claimWord(word);
  }

  /** Index of the first unclaimed token from `words`, or -1. A claimed token already has a meaning. */
  private firstIndexOf(words: readonly string[]): number {
    for (let i = 0; i < this.scan.length; i += 1) {
      if (this.scan.isClaimed(i)) continue;
      if (words.includes(this.scan.word(i))) return i;
    }
    return -1;
  }

  private firstOf(words: readonly string[]): string | null {
    const index = this.firstIndexOf(words);
    return index === -1 ? null : this.scan.word(index);
  }

  /** The layout every population in this plan is built with. */
  private newLayout(size: number): PopulationLayout {
    return layoutFor(this.layoutOverride ?? this.region.layout, size, this.nextSeed());
  }

  /**
   * Claim the region and layout words. Deferred until after `createPopulations`
   * so the backwards walk that collects a population's modifiers can still cross
   * them — "add 100 cortical neurons" needs `cortical` to stay unclaimed long
   * enough for `neurons` to reach the count in front of it.
   */
  private flushDeferredClaims(): void {
    this.claimWords(this.deferredClaims);
    this.deferredClaims.length = 0;
  }

  // ------------------------------------------------------------------- phases

  private detectClear(): void {
    for (const verb of CLEAR_VERBS) {
      const index = this.scan.indexOf(verb);
      if (index === -1) continue;
      const standalone = verb === 'start-over' || verb === 'from-scratch';
      let objectIndex = -1;
      for (let i = index + 1; i <= index + 3 && i < this.scan.length; i += 1) {
        if (CLEAR_OBJECTS.includes(this.scan.word(i))) objectIndex = i;
      }
      if (!standalone && objectIndex === -1) continue;
      this.scan.claimRange(index, (objectIndex === -1 ? index : objectIndex) + 1);
      this.cleared = true;
      this.actions.push({ type: 'clear' });
      return;
    }
  }

  private detectRegion(): void {
    let matched: RegionProfile | null = null;
    for (const [word, profile] of REGION_WORDS) {
      if (!this.scan.has(word)) continue;
      if (matched === null) matched = profile;
      this.deferredClaims.push(word);
    }
    if (matched !== null) this.region = matched;
  }

  private detectLayout(): void {
    for (const [word, layout] of LAYOUT_WORDS) {
      if (!this.scan.has(word)) continue;
      if (this.layoutOverride === null) this.layoutOverride = layout;
      this.deferredClaims.push(word);
    }
  }

  private seedFromCircuit(): void {
    const biasTotals = new Map<string, { sum: number; count: number }>();
    for (const neuron of this.circuit.neurons) {
      if (neuron.population === null) continue;
      const entry = biasTotals.get(neuron.population);
      if (entry) {
        entry.sum += neuron.bias;
        entry.count += 1;
      } else {
        biasTotals.set(neuron.population, { sum: neuron.bias, count: 1 });
      }
    }
    const namesById = new Map<string, string>();
    for (const population of this.circuit.populations) {
      namesById.set(population.id, population.name);
      const bias = biasTotals.get(population.id);
      this.populations.push({
        name: population.name,
        polarity: population.polarity,
        model: population.params.kind,
        size: population.size,
        archetype: population.morphology.archetype,
        bias: bias && bias.count > 0 ? bias.sum / bias.count : 0,
        planned: false,
      });
    }
    for (const projection of this.circuit.projections) {
      const sourceName = namesById.get(projection.source);
      const targetName = namesById.get(projection.target);
      if (sourceName === undefined || targetName === undefined) continue;
      this.projections.push({
        name: projection.name,
        sourceName,
        targetName,
        weightMean: projection.weightMean,
        delayMean: projection.delayMean,
        spec: null,
      });
    }
  }

  private collectConnectRequests(): ConnectRequest[] {
    const pattern =
      /\b(?:connect|connects|connecting|wire|wires|wiring|link|links|linking|project|projects|projecting|innervate|innervates)\b\s+([\s\S]{1,70}?)\s+(?:to|onto|into|towards|toward|with)\s+([\s\S]{1,70}?)(?=\s+(?:with|at|using|via|and|so|then)\b|[.;!?]|$)/g;
    const requests: ConnectRequest[] = [];
    for (const match of this.scan.matchAll(pattern)) {
      const end = match.index + match[0].length;
      const stop = this.sentenceEnd(end);
      this.scan.claimChars(match.index, stop);
      requests.push({
        sourceText: match[1],
        targetText: match[2],
        tail: this.scan.text.slice(end, stop),
      });
    }
    return requests;
  }

  private sentenceEnd(from: number): number {
    const rest = this.scan.text.slice(from);
    const match = /[.;!?]/.exec(rest);
    return match === null ? this.scan.text.length : from + match.index;
  }

  private createPopulations(): void {
    const drafts: PopulationDraft[] = [];
    for (let head = 0; head < this.scan.length; head += 1) {
      const word = this.scan.word(head);
      if (!HEAD_NOUNS.has(word) || this.scan.isClaimed(head)) continue;

      let start = head;
      while (
        start > 0 &&
        !this.scan.isClaimed(start - 1) &&
        !PHRASE_BOUNDARIES.has(this.scan.word(start - 1)) &&
        !HEAD_NOUNS.has(this.scan.word(start - 1))
      ) {
        start -= 1;
      }

      // A container noun hands the size of the population to the phrase after
      // "of": "a thalamic relay population of 300 cells" is 300 cells, not a
      // default-sized relay population plus 300 more. Any count on the container
      // itself counts populations rather than neurons, so it is taken out of the
      // modifiers and used to repeat the draft. Extending is safe even when
      // nothing comes of it: the merged modifiers are a superset of the inner
      // phrase's own, so a phrase that drafts nothing still drafts nothing.
      let modifiers = this.scan.slice(start, head + 1);
      let repeat = 1;
      if (
        CONTAINER_NOUNS.has(word) &&
        this.scan.word(head + 1) === 'of' &&
        !this.scan.isClaimed(head + 1)
      ) {
        const inner = this.innerHeadNoun(head + 2);
        if (inner !== -1) {
          const containerCount = parseCount(modifiers);
          const tail = this.scan.slice(head + 1, inner + 1);
          if (containerCount === null) {
            modifiers = [...modifiers, ...tail];
          } else {
            repeat = Math.min(containerCount, MAX_REPEATED_POPULATIONS);
            if (containerCount > MAX_REPEATED_POPULATIONS) {
              this.warn(
                `Asking for ${containerCount} populations at once is more than the offline planner will build; ${MAX_REPEATED_POPULATIONS} were created.`,
              );
            }
            modifiers = [...modifiers.filter((token) => !isCountToken(token)), ...tail];
          }
          head = inner;
        }
      }

      // The verb governing the phrase can sit behind a determiner the backwards
      // walk already stopped at: "remove the basket cells" boundaries on "the".
      let verb = start - 1;
      while (verb >= 0 && DETERMINERS.has(this.scan.word(verb))) verb -= 1;
      // A conjunct inherits the verb governing the list it belongs to. In
      // "create 200 excitatory neurons and inhibitory basket cells" the second
      // phrase is governed by "create", not by "and"; without this it reads as
      // naming cells the circuit already has and no population is created.
      if (verb >= 0 && COORDINATORS.has(this.scan.word(verb))) {
        for (let i = verb - 1; i >= 0; i -= 1) {
          const word = this.scan.word(i);
          if (CREATION_VERBS.has(word) || REMOVAL_VERBS.has(word)) {
            verb = i;
            break;
          }
          if (SENTENCE_ENDS.has(word)) break;
        }
      }
      const governing = verb >= 0 ? this.scan.word(verb) : '';
      if (REMOVAL_VERBS.has(governing)) {
        this.scan.claimRange(verb, head + 1);
        this.warn(
          'Removing neurons is not something the offline planner can express; that part of the request was skipped.',
        );
        continue;
      }

      const count = parseCount(modifiers);

      // "switch everything to Izhikevich neurons" names the model to move to, not
      // cells to build. Left unclaimed, so `switchModel` reads it.
      if (
        count === null &&
        MODEL_SWITCH_GOVERNORS.has(governing) &&
        this.scan.hasAny(MODEL_SWITCH_VERBS) &&
        modifiers.some((token) => MODEL_WORDS[token] !== undefined)
      ) {
        continue;
      }

      // "Make the basket cells fire faster" names a population that already
      // exists; only a count or an explicit creation verb makes a phrase a
      // request for new neurons.
      if (count === null && !CREATION_VERBS.has(governing)) {
        const referenced = this.referencedPopulation(modifiers);
        if (referenced !== null) {
          this.scan.claimRange(start, head + 1);
          if (!this.focus.includes(referenced)) this.focus.push(referenced);
          continue;
        }
      }

      const draft = this.draftFromPhrase(modifiers, this.scan.word(head), drafts);
      if (draft === null) continue;
      this.scan.claimRange(start, head + 1);
      drafts.push(draft);
      // Each repeat is drafted rather than copied so it gets its own name and
      // its own layout seed, which is what keeps the copies from overlapping.
      for (let n = 1; n < repeat; n += 1) {
        const copy = this.draftFromPhrase(modifiers, this.scan.word(head), drafts);
        if (copy !== null) drafts.push(copy);
      }
    }
    if (drafts.length === 0) return;
    this.emitPopulations(drafts);
  }

  /**
   * The head noun of the phrase introduced by "of" after a container noun, or -1
   * when what follows is not a plain modifier list ending in one. The window is
   * short because a long run of words between the two nouns means the second is
   * a separate phrase rather than the container's contents.
   */
  private innerHeadNoun(from: number): number {
    const limit = Math.min(this.scan.length, from + 6);
    for (let i = from; i < limit; i += 1) {
      if (this.scan.isClaimed(i)) return -1;
      const token = this.scan.word(i);
      if (HEAD_NOUNS.has(token)) return i;
      if (PHRASE_BOUNDARIES.has(token)) return -1;
    }
    return -1;
  }

  private draftFromPhrase(
    modifiers: readonly string[],
    head: string,
    existing: readonly PopulationDraft[],
  ): PopulationDraft | null {
    let polarity: NeuronPolarity | null = null;
    let archetype: MorphologyArchetype | null = null;
    let model: NeuronModelKind | null = null;
    let pattern: FiringPattern | null = null;
    for (const token of modifiers) {
      polarity = POLARITY_WORDS[token] ?? polarity;
      archetype = ARCHETYPE_WORDS[token] ?? archetype;
      model = MODEL_WORDS[token] ?? model;
      pattern = FIRING_WORDS[token] ?? pattern;
    }
    const count = parseCount(modifiers);
    if (count === null && polarity === null && archetype === null && model === null && pattern === null) {
      return null;
    }
    if (head === 'interneuron' || head === 'interneurons') polarity = 'inhibitory';
    if (archetype === 'basket' || archetype === 'purkinje') polarity = polarity ?? 'inhibitory';
    const resolvedPolarity = polarity ?? this.region.defaultPolarity;
    const resolvedArchetype =
      archetype ??
      (resolvedPolarity === 'inhibitory'
        ? this.region.inhibitoryArchetype
        : this.region.excitatoryArchetype);
    const resolvedPattern =
      pattern ??
      (resolvedPolarity === 'inhibitory'
        ? this.region.inhibitoryPattern
        : this.region.excitatoryPattern);
    const resolvedModel = model ?? 'izhikevich';

    let size = count ?? this.inferSize(resolvedPolarity, existing);
    if (size > MAX_POPULATION_SIZE) {
      this.warn(
        `A population of ${size} neurons exceeds the ${MAX_POPULATION_SIZE} per-population limit; it was reduced to ${MAX_POPULATION_SIZE}.`,
      );
      size = MAX_POPULATION_SIZE;
    }
    size = Math.max(1, size);

    let params: Partial<NeuronParams> | null = null;
    if (resolvedModel === 'izhikevich') {
      params = izhikevichParams(resolvedPattern);
    } else if (pattern !== null) {
      if (IZHIKEVICH_ONLY_PATTERNS.includes(pattern)) {
        this.warn(
          `"${pattern}" is an Izhikevich firing pattern; the ${resolvedModel} population was left at its default parameters.`,
        );
      } else {
        params = interneuronParams(resolvedModel, pattern === 'fast-spiking');
      }
    }

    const label =
      this.region.label === ''
        ? `${titleCase(resolvedArchetype)} ${resolvedPolarity === 'inhibitory' ? 'Interneurons' : 'Neurons'}`
        : `${this.region.label} ${titleCase(resolvedArchetype)}`;

    return {
      name: this.uniquePopulationName(label),
      size,
      polarity: resolvedPolarity,
      model: resolvedModel,
      archetype: resolvedArchetype,
      pattern: resolvedPattern,
      params,
      layout: this.newLayout(size),
    };
  }

  private inferSize(polarity: NeuronPolarity, drafts: readonly PopulationDraft[]): number {
    if (polarity === 'excitatory') return DEFAULT_EXCITATORY_SIZE;
    const excitatory = drafts.find((d) => d.polarity === 'excitatory');
    const base = excitatory?.size ?? this.largest('excitatory')?.size ?? DEFAULT_EXCITATORY_SIZE;
    return Math.max(1, Math.round(base * INHIBITORY_FRACTION));
  }

  /** Lay the new populations out along X, clear of anything the document already holds. */
  private emitPopulations(drafts: readonly PopulationDraft[]): void {
    // A cleared document holds nothing to avoid, and the cursor has to survive
    // across calls: `applyRhythm` and the connect pass both emit populations
    // after `createPopulations` has already placed some.
    let cursor = this.placementCursor;
    if (!this.cleared) {
      for (const population of this.circuit.populations) {
        cursor = Math.max(cursor, population.origin.x + layoutRadius(population.layout));
      }
    }
    const clearing = cursor > 0;
    if (clearing) cursor += GAP_BETWEEN_POPULATIONS;
    const offsets: number[] = [];
    for (const draft of drafts) {
      const radius = layoutRadius(draft.layout);
      offsets.push(cursor + radius);
      cursor += 2 * radius + GAP_BETWEEN_POPULATIONS;
    }
    this.placementCursor = cursor;
    // Only a plan starting from an empty row centres its populations on the
    // origin. Recentring while something else is already placed would undo the
    // clearance the cursor just computed and drop the new cells on top of it.
    const centre = clearing
      ? 0
      : drafts.length > 1
        ? (offsets[0] + offsets[offsets.length - 1]) * 0.5
        : offsets[0];

    for (let i = 0; i < drafts.length; i += 1) {
      const draft = drafts[i];
      const origin: Vec3 = { x: Math.round((offsets[i] - centre) * 10) / 10, y: 0, z: 0 };
      const spec: PopulationSpec = {
        name: draft.name,
        size: draft.size,
        polarity: draft.polarity,
        model: draft.model,
        layout: draft.layout,
        origin,
        archetype: draft.archetype,
        color: null,
      };
      if (draft.params !== null) spec.params = draft.params;
      this.actions.push({ type: 'create-population', spec });
      const known: KnownPopulation = {
        name: draft.name,
        polarity: draft.polarity,
        model: draft.model,
        size: draft.size,
        archetype: draft.archetype,
        bias: 0,
        planned: true,
      };
      this.populations.push(known);
      this.created.push(known);
    }
  }

  // -------------------------------------------------------------- connections

  private applyConnectRequests(requests: readonly ConnectRequest[]): void {
    for (const request of requests) {
      const source = this.resolveOrCreate(request.sourceText);
      const target = this.resolveOrCreate(request.targetText);
      if (source === null || target === null) {
        const missing = source === null ? request.sourceText : request.targetText;
        this.warn(
          `Could not tell which population "${missing.trim()}" refers to, so that connection was skipped.`,
        );
        continue;
      }
      if (this.findProjection(source.name, target.name) !== null) {
        this.warn(
          `${source.name} already projects to ${target.name}, and the connectivity of a projection that exists cannot be re-drawn in place; that request was skipped.`,
        );
        continue;
      }
      // The lookahead that ends the target phrase does not fire on a trailing
      // "all-to-all", so a rule can land inside the target text rather than the
      // tail. Both are searched for the qualifiers.
      const qualifiers = `${request.targetText} ${request.tail}`;
      this.connect(
        source,
        target,
        this.ruleFromTail(qualifiers),
        this.receptorFromTail(qualifiers, source),
      );
    }
  }

  /** How strongly a phrase describes one population: by name, polarity, morphology or model. */
  private matchScore(
    candidate: KnownPopulation,
    words: readonly string[],
    phrase: string,
  ): number {
    const name = candidate.name.toLowerCase();
    let score = phrase.includes(name) ? 10 : 0;
    const nameWords = name.split(/[^a-z0-9]+/).filter((w) => w.length > 0);
    for (const word of words) {
      // Generated names end in the same head nouns every phrase uses, so
      // matching on one would make "the neurons" name "Pyramidal Neurons".
      if (!HEAD_NOUNS.has(word) && nameWords.includes(word)) score += 3;
      score += agreement(POLARITY_WORDS[word], candidate.polarity);
      score += agreement(ARCHETYPE_WORDS[word], candidate.archetype);
      score += agreement(MODEL_WORDS[word], candidate.model);
    }
    return score;
  }

  /**
   * The population a connect phrase names, building it when the phrase describes
   * cells the document does not have. "Connect 400 pyramidal cells to 100 basket
   * cells" asks for both populations as well as the projection, and the connect
   * clause claims that text before `createPopulations` ever sees it, so this is
   * the only place those phrases can be turned into populations.
   */
  private resolveOrCreate(phrase: string): KnownPopulation | null {
    const existing = this.resolvePopulation(phrase);
    if (existing !== null) return existing;
    const words = phrase.toLowerCase().match(WORD_RE) ?? [];
    let head = -1;
    for (let i = words.length - 1; i >= 0; i -= 1) {
      if (HEAD_NOUNS.has(words[i])) {
        head = i;
        break;
      }
    }
    if (head === -1) return null;
    const draft = this.draftFromPhrase(words.slice(0, head + 1), words[head], []);
    if (draft === null) return null;
    this.emitPopulations([draft]);
    return this.populations[this.populations.length - 1];
  }

  private resolvePopulation(phrase: string): KnownPopulation | null {
    const lower = phrase.toLowerCase();
    const words = lower.match(WORD_RE) ?? [];
    if (words.length === 0) return null;
    let best: KnownPopulation | null = null;
    let bestScore = 0;
    let tied = false;
    for (const candidate of this.populations) {
      const score = this.matchScore(candidate, words, lower);
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
        tied = false;
      } else if (score === bestScore && score > 0) {
        tied = true;
      }
    }
    if (best !== null && tied) {
      this.warn(
        `"${phrase.trim()}" matched more than one population equally well; ${best.name} was used.`,
      );
    }
    return bestScore > 0 ? best : null;
  }

  /**
   * The population a phrase points at, when it points at exactly one that the
   * document already holds. Populations this plan creates are excluded: a phrase
   * cannot be referring to something that did not exist when it was written.
   */
  private referencedPopulation(words: readonly string[]): KnownPopulation | null {
    const phrase = words.join(' ');
    let best: KnownPopulation | null = null;
    let bestScore = 0;
    let tied = false;
    for (const candidate of this.populations) {
      if (candidate.planned) continue;
      const score = this.matchScore(candidate, words, phrase);
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
        tied = false;
      } else if (score === bestScore && score > 0) {
        tied = true;
      }
    }
    return bestScore >= REFERENCE_SCORE && !tied ? best : null;
  }

  /**
   * The populations an unqualified edit applies to: the ones the prompt named,
   * or all of them when it named none.
   */
  private focusedPopulations(): readonly KnownPopulation[] {
    if (this.focus.length === 0) return this.populations;
    this.focusUsed = true;
    return this.focus;
  }

  /** Claim bare head nouns, which an edit that spans every population has addressed. */
  private claimHeadNouns(): void {
    for (let i = 0; i < this.scan.length; i += 1) {
      if (HEAD_NOUNS.has(this.scan.word(i))) this.scan.claim(i);
    }
  }

  private ruleFromTail(tail: string): ConnectivityRule {
    if (/\ball-to-all\b|\bfully-connected\b|\bevery\b/.test(tail)) {
      return { kind: 'all-to-all', selfConnections: false };
    }
    if (/\bone-to-one\b/.test(tail)) return { kind: 'one-to-one' };
    const percent = /(\d+(?:\.\d+)?)\s*(?:%|percent)/.exec(tail);
    if (percent !== null) {
      return {
        kind: 'random',
        probability: clamp(Number.parseFloat(percent[1]) / 100, 0, 1),
        seed: this.nextSeed(),
        selfConnections: false,
      };
    }
    const decimal = /\b(?:probability|chance)\s+(?:of\s+)?(\d*\.\d+)|\b(\d*\.\d+)\s+probability/.exec(tail);
    if (decimal !== null) {
      return {
        kind: 'random',
        probability: clamp(Number.parseFloat(decimal[1] ?? decimal[2]), 0, 1),
        seed: this.nextSeed(),
        selfConnections: false,
      };
    }
    const degree = /(\d+)\s+(?:inputs?|connections?)\s+(?:per|each)/.exec(tail);
    if (degree !== null) {
      return {
        kind: 'fixed-in-degree',
        degree: Math.max(1, Number.parseInt(degree[1], 10)),
        seed: this.nextSeed(),
      };
    }
    return { kind: 'random', probability: 0.1, seed: this.nextSeed(), selfConnections: false };
  }

  private receptorFromTail(tail: string, source: KnownPopulation): ReceptorKind {
    const words = tail.toLowerCase().match(/[a-z]+(?:-[a-z]+)*/g) ?? [];
    for (const word of words) {
      const receptor = RECEPTOR_WORDS[word];
      if (receptor !== undefined) return receptor;
    }
    return source.polarity === 'inhibitory' ? 'gabaa' : 'ampa';
  }

  private connect(
    source: KnownPopulation,
    target: KnownPopulation,
    rule: ConnectivityRule,
    receptor: ReceptorKind,
    overrides: { weightMean?: number; delayMean?: number } = {},
  ): KnownProjection {
    const inhibitory = receptor === 'gabaa' || receptor === 'gabab';
    const weightMean = overrides.weightMean ?? (inhibitory ? 4.5 : 1.2);
    const delayMean = overrides.delayMean ?? (inhibitory ? 1 : 1.5);
    const spec: NamedProjectionSpec = {
      name: this.uniqueProjectionName(`${source.name} -> ${target.name}`),
      sourceName: source.name,
      targetName: target.name,
      rule,
      receptor,
      weightMean,
      weightJitter: Math.round(weightMean * 0.25 * 100) / 100,
      delayMean,
      delayJitter: Math.round(delayMean * 0.3 * 100) / 100,
      plasticity: 'static',
    };
    this.actions.push({ type: 'connect-populations', spec });
    const projection: KnownProjection = {
      name: spec.name,
      sourceName: source.name,
      targetName: target.name,
      weightMean,
      delayMean,
      spec,
    };
    this.projections.push(projection);
    return projection;
  }

  /** Wire a freshly created excitatory/inhibitory pair into a working microcircuit. */
  private autoWire(): void {
    if (this.created.length === 0) return;
    const recurrent = this.region.recurrent || this.scan.hasAny(RECURRENCE_WORDS);
    if (!recurrent) return;
    this.claimWords(RECURRENCE_WORDS);

    const excitatory = this.created.filter((p) => p.polarity === 'excitatory');
    const inhibitory = this.created.filter((p) => p.polarity === 'inhibitory');
    const wire = (
      source: KnownPopulation,
      target: KnownPopulation,
      probability: number,
      receptor: ReceptorKind,
    ): void => {
      if (this.findProjection(source.name, target.name) !== null) return;
      this.connect(source, target, {
        kind: 'random',
        probability,
        seed: this.nextSeed(),
        selfConnections: false,
      }, receptor);
    };

    for (const source of excitatory) {
      if (source.size >= 8) wire(source, source, 0.08, 'ampa');
      for (const target of inhibitory) wire(source, target, 0.2, 'ampa');
    }
    for (const source of inhibitory) {
      for (const target of excitatory) wire(source, target, 0.3, 'gabaa');
      if (source.size >= 8) wire(source, source, 0.15, 'gabaa');
    }
  }

  private reportOrphans(): void {
    for (const population of this.created) {
      const touched = this.projections.some(
        (p) => p.sourceName === population.name || p.targetName === population.name,
      );
      if (!touched) {
        this.warn(
          `${population.name} was created but is not connected to anything; ask to connect it to make it part of the circuit.`,
        );
      }
    }
  }

  // ------------------------------------------------------------- global edits

  private switchModel(): void {
    let kind: NeuronModelKind | null = null;
    let index = -1;
    for (let i = 0; i < this.scan.length; i += 1) {
      if (this.scan.isClaimed(i)) continue;
      const candidate = MODEL_WORDS[this.scan.word(i)];
      if (candidate !== undefined) {
        kind = candidate;
        index = i;
        break;
      }
    }
    if (kind === null) return;
    if (!this.scan.hasAny(SWITCH_VERBS)) return;
    this.scan.claim(index);
    this.claimWords(SWITCH_VERBS);

    if (this.populations.length === 0) {
      this.warn(`There are no populations to switch to ${NEURON_MODEL_LABELS[kind]}.`);
      return;
    }
    const targets = this.focusedPopulations();
    if (targets === this.populations && this.scan.hasAny(GLOBAL_SCOPE_WORDS)) {
      this.claimWords(GLOBAL_SCOPE_WORDS);
      this.claimHeadNouns();
    }
    let changed = 0;
    for (const population of targets) {
      if (population.model === kind) continue;
      population.model = kind;
      changed += 1;
      this.actions.push({
        type: 'tune-population',
        name: population.name,
        params: defaultParams(kind),
      });
    }
    // Producing no actions because the request is already satisfied is a very
    // different thing from failing to understand it, and the generic
    // "not understood" fallback would report the wrong one.
    if (changed === 0) {
      const scope = targets === this.populations ? 'Every population' : targets[0].name;
      this.warn(`${scope} already uses ${NEURON_MODEL_LABELS[kind]}; nothing to change.`);
    }
  }

  private detectRhythm(): { hz: number; explicit: boolean } | null {
    // A frequency already claimed belongs to whatever claimed it — the rate of a
    // Poisson drive is not a rhythm target — so take the first free one.
    for (const match of this.scan.matchAll(/(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b/g)) {
      const end = match.index + match[0].length;
      if (this.scan.isRangeClaimed(match.index, end)) continue;
      this.scan.claimChars(match.index, end);
      this.claimWords(['frequency', 'band', 'rhythm', 'oscillate', 'at']);
      this.claimWords(RHYTHM_WORDS);
      return { hz: clamp(Number.parseFloat(match[1]), 0.1, 1000), explicit: true };
    }
    for (const band of RHYTHM_BANDS) {
      const word = this.firstOf(band.words);
      if (word === null) continue;
      this.claimWords(band.words);
      this.claimWords(['frequency', 'band', 'rhythm', 'oscillate', 'range']);
      this.claimWords(RHYTHM_WORDS);
      return { hz: band.targetHz, explicit: false };
    }
    if (this.scan.hasAny(RHYTHM_WORDS)) {
      this.claimWords(RHYTHM_WORDS);
      this.claimWords(['frequency', 'band', 'rhythm']);
      this.warn('No target frequency was given, so the gamma band (40 Hz) was assumed.');
      return { hz: 40, explicit: false };
    }
    return null;
  }

  private applyRhythm(): void {
    const target = this.detectRhythm();
    if (target === null) return;
    const hz = target.hz;
    const band = bandForFrequency(hz);
    const dt = hz >= 80 ? 0.025 : hz >= 30 ? 0.05 : 0.1;

    this.actions.push({
      type: 'set-simulation',
      patch: {
        dt,
        noise: Math.round(clamp(6 + hz * 0.15, 6, 40)),
        plasticityEnabled: false,
      },
    });

    const excitatory = this.largest('excitatory');
    if (excitatory === null) {
      this.warn(
        `There is no excitatory population to carry a ${band.name} rhythm, so only the integration settings were changed.`,
      );
      return;
    }

    let inhibitory = this.largest('inhibitory');
    if (inhibitory === null) {
      const size = Math.max(1, Math.round(excitatory.size * INHIBITORY_FRACTION));
      const draft: PopulationDraft = {
        name: this.uniquePopulationName(
          this.region.label === ''
            ? 'Basket Interneurons'
            : `${this.region.label} ${titleCase(this.region.inhibitoryArchetype)}`,
        ),
        size,
        polarity: 'inhibitory',
        model: excitatory.model,
        archetype: this.region.inhibitoryArchetype,
        pattern: 'fast-spiking',
        params: interneuronParams(excitatory.model, hz >= 30),
        layout: this.newLayout(size),
      };
      this.emitPopulations([draft]);
      inhibitory = this.populations[this.populations.length - 1];
      this.warn(
        `A ${band.name} rhythm needs recurrent inhibition, so a population of ${size} interneurons was added.`,
      );
    }

    const inhibitoryBias = Math.round(sustainedDrive(inhibitory.model, hz) * 0.3);
    inhibitory.bias = inhibitoryBias;
    this.actions.push({
      type: 'tune-population',
      name: inhibitory.name,
      params: interneuronParams(inhibitory.model, hz >= 30),
      bias: inhibitoryBias,
      noise: Math.round(clamp(hz * 0.2, 4, 30)),
    });

    const drive = sustainedDrive(excitatory.model, hz);
    excitatory.bias = drive;
    this.actions.push({
      type: 'tune-population',
      name: excitatory.name,
      params: kindOnlyParams(excitatory.model),
      bias: drive,
      noise: Math.round(clamp(drive * 0.06, 4, 60)),
    });

    const periodMs = 1000 / hz;
    const loopDelay = Math.round(clamp(periodMs * 0.08, 0.6, 12) * 100) / 100;
    const inhibitoryWeight = Math.round(clamp(4 + 0.06 * hz, 3, 16) * 100) / 100;
    const excitatoryWeight = Math.round(clamp(1 + 0.015 * hz, 0.8, 4) * 100) / 100;
    const inhibitoryReceptor: ReceptorKind = hz >= 20 ? 'gabaa' : 'gabab';
    if (inhibitoryReceptor === 'gabab') {
      this.warn(
        `Below 20 Hz the only inhibition slow enough is GABA-B (150 ms decay), so the rhythm will sit at the low end of the ${band.name} band.`,
      );
    }

    this.tuneOrConnect(excitatory, inhibitory, 0.25, 'ampa', excitatoryWeight, loopDelay);
    this.tuneOrConnect(inhibitory, excitatory, 0.35, inhibitoryReceptor, inhibitoryWeight, loopDelay);
  }

  /** Retune an existing projection towards the rhythm, or create it if it is missing. */
  private tuneOrConnect(
    source: KnownPopulation,
    target: KnownPopulation,
    probability: number,
    receptor: ReceptorKind,
    weightMean: number,
    delayMean: number,
  ): void {
    const existing = this.findProjection(source.name, target.name);
    if (existing === null) {
      this.connect(
        source,
        target,
        { kind: 'random', probability, seed: this.nextSeed(), selfConnections: false },
        receptor,
        { weightMean, delayMean },
      );
      return;
    }
    existing.weightMean = weightMean;
    existing.delayMean = delayMean;
    if (existing.spec !== null) {
      existing.spec.weightMean = weightMean;
      existing.spec.delayMean = delayMean;
      existing.spec.plasticity = 'static';
      return;
    }
    this.actions.push({
      type: 'tune-projection',
      name: existing.name,
      weightMean,
      delayMean,
      plasticity: 'static',
    });
  }

  private applyDensity(): void {
    const sparser = this.firstOf(SPARSER_WORDS);
    const denser = this.firstOf(DENSER_WORDS);
    if (sparser === null && denser === null) return;
    this.claimWords(SPARSER_WORDS);
    this.claimWords(DENSER_WORDS);
    const factor = sparser !== null ? 0.6 : 1.6;
    const label = sparser !== null ? 'sparser' : 'denser';

    let touched = 0;
    let rescaledWeights = 0;
    for (const projection of this.projections) {
      if (projection.spec !== null) {
        const rule = projection.spec.rule;
        if (rule.kind === 'random' || rule.kind === 'distance-threshold') {
          rule.probability = Math.round(clamp(rule.probability * factor, 0.001, 1) * 1000) / 1000;
          touched += 1;
          continue;
        }
        if (rule.kind === 'gaussian') {
          rule.maxProbability = Math.round(clamp(rule.maxProbability * factor, 0.001, 1) * 1000) / 1000;
          touched += 1;
          continue;
        }
      }
      const weightMean = Math.round(clamp(projection.weightMean * factor, 0, 1000) * 100) / 100;
      projection.weightMean = weightMean;
      this.actions.push({ type: 'tune-projection', name: projection.name, weightMean });
      touched += 1;
      rescaledWeights += 1;
    }
    if (rescaledWeights > 0) {
      this.warn(
        `The number of synapses in a projection that already exists cannot be changed in place, so "${label}" was applied to ${rescaledWeights} projection${rescaledWeights === 1 ? '' : 's'} as a x${factor} change in synaptic weight instead.`,
      );
    }
    if (touched === 0) {
      const gain = Math.round(clamp(this.circuit.simulation.gain * factor, 0, 100) * 100) / 100;
      this.actions.push({ type: 'set-simulation', patch: { gain } });
      this.warn(
        `There are no projections to make ${label}, so the global synaptic gain was scaled to ${gain} instead.`,
      );
    }
  }

  private applySpeed(): void {
    const faster = this.firstOf(FASTER_WORDS);
    const slower = this.firstOf(SLOWER_WORDS);
    if (faster === null && slower === null) return;
    this.claimWords(FASTER_WORDS);
    this.claimWords(SLOWER_WORDS);
    const speedingUp = faster !== null;

    if (this.scan.hasAny(RATE_WORDS)) {
      this.claimWords(RATE_WORDS);
      // The bare noun naming the cells — "the neurons" in "make the neurons fire
      // faster" — is the subject of this edit and so has been addressed.
      this.claimHeadNouns();
      if (this.populations.length === 0) {
        this.warn('There are no populations whose firing rate could be changed.');
        return;
      }
      const step = speedingUp ? 80 : -80;
      for (const population of this.focusedPopulations()) {
        const bias = Math.round(clamp(population.bias + step, -5000, 5000));
        population.bias = bias;
        this.actions.push({
          type: 'tune-population',
          name: population.name,
          params: kindOnlyParams(population.model),
          bias,
        });
      }
      return;
    }
    const speed = Math.round(clamp(this.circuit.simulation.speed * (speedingUp ? 2 : 0.5), 0.05, 20) * 100) / 100;
    this.actions.push({ type: 'set-simulation', patch: { speed } });
  }

  // ---------------------------------------------------------------- stimuli

  /**
   * Read the external-drive request out of the prompt and claim the words that
   * describe it.
   *
   * This runs before populations are drafted because a drive names its target
   * with the same grammar a new population uses — "…to the excitatory
   * population" would otherwise be cut as a population phrase and instantiated.
   * The target is only a phrase at this point; it is resolved in
   * `applyStimulus`, once every population the plan creates exists.
   */
  private collectStimulusRequest(): StimulusRequest | null {
    const triggerIndex = this.firstIndexOf(STIMULUS_TRIGGERS);
    if (triggerIndex === -1) return null;
    const trigger = this.scan.tokens[triggerIndex].start;
    const from = this.sentenceStart(trigger);
    const to = this.sentenceEnd(trigger);
    const sentence = this.scan.text.slice(from, to);

    let patternWord: string | null = null;
    let patternAt = Number.POSITIVE_INFINITY;
    for (const word of STIMULUS_PATTERN_WORDS) {
      const match = new RegExp(`\\b${word}\\b`).exec(sentence);
      if (match === null || match.index >= patternAt) continue;
      patternAt = match.index;
      patternWord = word;
    }
    const amplitudeMatch = /(\d+(?:\.\d+)?)\s*pa\b/.exec(sentence);
    // A bare "drive it" says nothing about what to inject; leave it to the
    // leftover report rather than inventing a waveform.
    if (patternWord === null && amplitudeMatch === null) return null;

    const amplitude =
      amplitudeMatch === null ? 150 : clamp(Number.parseFloat(amplitudeMatch[1]), -1e5, 1e5);
    const frequencyMatch = /(\d+(?:\.\d+)?)\s*(?:hz|hertz)\b/.exec(sentence);
    const frequency =
      frequencyMatch === null ? 10 : clamp(Number.parseFloat(frequencyMatch[1]), 0.01, 1000);

    // Claim from the earliest word that describes the drive, not from the
    // trigger, so "add a poisson input …" does not leave "poisson" behind.
    const start = Math.min(
      trigger,
      patternWord === null ? Number.POSITIVE_INFINITY : from + patternAt,
      amplitudeMatch === null ? Number.POSITIVE_INFINITY : from + amplitudeMatch.index,
    );
    this.scan.claimChars(start, to);

    const tail = this.scan.text.slice(start, to);
    const targetMatch =
      /\b(?:to|into|onto)\s+([\s\S]{1,60}?)(?=\s+(?:with|at|using|and)\b|[.;!?]|$)/.exec(tail);
    return {
      pattern: this.stimulusPattern(patternWord ?? 'constant', amplitude, frequency),
      targetText: targetMatch === null ? null : targetMatch[1],
    };
  }

  private applyStimulus(request: StimulusRequest | null): void {
    if (request === null) return;
    const named = request.targetText === null ? null : this.resolvePopulation(request.targetText);
    if (named === null && request.targetText !== null) {
      this.warn(
        `Could not tell which population "${request.targetText.trim()}" refers to, so the drive was attached to the largest one instead.`,
      );
    }
    const target = named ?? this.subjectPopulation();
    if (target === null) {
      this.warn('There is no population to attach a stimulus to, so the drive was skipped.');
      return;
    }
    this.actions.push({
      type: 'add-stimulus',
      targetPopulation: target.name,
      pattern: request.pattern,
      name: `${titleCase(request.pattern.kind)} drive`,
    });
  }

  private sentenceStart(from: number): number {
    const head = this.scan.text.slice(0, from);
    const match = /[.;!?][^.;!?]*$/.exec(head);
    return match === null ? 0 : match.index + 1;
  }

  private stimulusPattern(word: string, amplitude: number, frequency: number): StimulusPattern {
    switch (word) {
      case 'poisson':
        return { kind: 'poisson', rate: frequency, amplitude, seed: this.nextSeed() };
      case 'step':
        return { kind: 'step', amplitude, start: 0, duration: 1000 };
      case 'ramp':
        return { kind: 'ramp', from: 0, to: amplitude, start: 0, duration: 1000 };
      case 'sine':
      case 'sinusoidal':
        return { kind: 'sine', amplitude, frequency, offset: 0 };
      case 'pulse-train':
      case 'train':
      case 'pulses':
        return { kind: 'pulse-train', amplitude, frequency, width: 1, start: 0 };
      default:
        return { kind: 'constant', amplitude };
    }
  }

  // ----------------------------------------------------------------- reporting

  private reportUnusedFocus(): void {
    if (this.focus.length === 0 || this.focusUsed) return;
    const names = this.focus.map((population) => population.name).join(', ');
    this.warn(
      `${names} was read as a reference to cells the circuit already has, but the request did not say what to change about it.`,
    );
  }

  private reportLeftovers(): void {
    const leftover = this.scan.unclaimedWords(8);
    if (leftover.length === 0) return;
    const quoted = leftover.map((word) => `"${word}"`).join(', ');
    this.warn(`Part of the request was not understood and was left out of the plan: ${quoted}.`);
  }

  private buildSummary(): string {
    if (this.actions.length === 0) {
      // An earlier pass may already have said exactly why nothing was emitted —
      // that the projection exists, that the model is already in use. Adding the
      // generic fallback on top of a specific reason contradicts it, and reads as
      // though the planner failed to understand a request it understood fine.
      if (this.warnings.length === 0) {
        this.warn('Nothing in this request maps to a circuit edit that can be made offline.');
        return 'No changes: the offline planner could not turn this request into circuit edits.';
      }
      return 'No changes: every part of this request was already satisfied or could not be applied.';
    }
    const parts: string[] = [];
    if (this.cleared) parts.push('cleared the circuit');
    const created = this.created;
    if (created.length > 0) {
      const total = created.reduce((sum, p) => sum + p.size, 0);
      const detail = created.map((p) => `${p.size} ${p.name}`).join(', ');
      parts.push(`added ${total} neurons across ${created.length} population${created.length === 1 ? '' : 's'} (${detail})`);
    }
    const newProjections = this.projections.filter((p) => p.spec !== null).length;
    if (newProjections > 0) {
      parts.push(`wired ${newProjections} projection${newProjections === 1 ? '' : 's'}`);
    }
    const tunedPopulations = this.actions.filter((a) => a.type === 'tune-population').length;
    if (tunedPopulations > 0) {
      parts.push(`retuned ${tunedPopulations} population${tunedPopulations === 1 ? '' : 's'}`);
    }
    const tunedProjections = this.actions.filter((a) => a.type === 'tune-projection').length;
    if (tunedProjections > 0) {
      parts.push(`retuned ${tunedProjections} projection${tunedProjections === 1 ? '' : 's'}`);
    }
    if (this.actions.some((a) => a.type === 'set-simulation')) parts.push('updated the integration settings');
    if (this.actions.some((a) => a.type === 'add-stimulus')) parts.push('attached an external drive');
    if (this.actions.some((a) => a.type === 'set-render')) parts.push('updated the scene appearance');
    if (parts.length === 0) return 'Applied the requested changes.';
    const sentence = parts.join('; ');
    return `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}.`;
  }
}
