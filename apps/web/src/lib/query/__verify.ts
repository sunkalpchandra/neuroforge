/**
 * Self-check for the query language.
 *
 * Everything here runs against one hand-built network of six cells and eight
 * synapses whose answers were worked out by hand before the code was written,
 * which is the only way a test of a search language means anything: if the
 * expected sets came out of the implementation they would agree with any bug it
 * has. The fixture is small enough to reason about completely and rich enough to
 * separate every operator from every other:
 *
 *   slot  label    polarity     model        population  archetype
 *   0     "Pm 2"   excitatory   izhikevich   Alpha       pyramidal
 *   1     "PM 02"  excitatory   lif          Alpha       pyramidal
 *   2     "Pm3"    inhibitory   lif          Beta        basket
 *   3     "LC10"   excitatory   adex         Beta        granule
 *   4     "T4a"    excitatory   izhikevich   —           pyramidal
 *   5     ""       inhibitory   lif          —           purkinje
 *
 *   edges  0->2  1->2  2->3  3->4  2->0  4->5  0->3  2->1
 *
 *   out-neighbours  0:{2,3}  1:{2}  2:{0,1,3}  3:{4}  4:{5}  5:{}
 *   in-neighbours   0:{2}    1:{2}  2:{0,1}    3:{0,2}  4:{3}  5:{4}
 *
 * Slots 0 and 1 are wired identically once wiring is read at population
 * granularity — both send only to Beta and receive only from Beta — so their
 * connectivity fingerprints are the same vector, which is what makes the
 * similarity operators checkable by hand rather than by eye. Their morphology
 * descriptors are identical too, and slot 4's are deliberately near theirs, so
 * `{similar_shape}` has a boundary to be right about.
 *
 * Nothing here touches the editor store or the live engine buffers: the fixture
 * owns its own `SimulationBuffers`, so calling `verifyQuery()` from a running
 * session cannot disturb the user's document.
 *
 * Returns the names of the invariants that failed; empty means the language is
 * behaving.
 */

import type {
  Circuit,
  Morphology,
  MorphologyArchetype,
  Neuron,
  NeuronModelKind,
  NeuronPolarity,
  Population,
  ReceptorKind,
  SimulationBuffers,
  Synapse,
} from '@neuroforge/shared';
import {
  CIRCUIT_SCHEMA_VERSION,
  DEFAULT_CAMERA,
  DEFAULT_PLASTICITY,
  DEFAULT_RENDER_SETTINGS,
  DEFAULT_SIMULATION_SETTINGS,
  DEFAULT_STP,
  MODEL_CODE,
  NEURON_FLAG,
  RECEPTOR_CODE,
  RECEPTOR_DEFAULTS,
  allocateSimulationBuffers,
  asCircuitId,
  asNeuronId,
  asPopulationId,
  asSynapseId,
  defaultMorphology,
  defaultParams,
} from '@neuroforge/shared';

import { QUERY_FIELDS, findField } from './fields';
import { QUERY_OPERATORS, tokenize } from './lexer';
import type { QueryNode } from './parser';
import { MAX_QUERY_DEPTH, MAX_QUERY_TERMS, parseQuery } from './parser';
import { evaluateQuery, likeKey, matchedSlots } from './evaluate';

type Check = (name: string, ok: boolean) => void;

interface Fixture {
  circuit: Circuit;
  buffers: SimulationBuffers;
}

export function verifyQuery(): string[] {
  const failures: string[] = [];
  const check: Check = (name, ok) => {
    if (!ok) failures.push(name);
  };

  const fixture = buildFixture();

  checkTables(check);
  checkLexer(check);
  checkPrecedence(check);
  checkParseErrors(check);
  checkLike(check, fixture);
  checkStringOperators(check, fixture);
  checkNumericOperators(check, fixture);
  checkBooleanAndPresence(check, fixture);
  checkConnectivity(check, fixture);
  checkSimilarity(check, fixture);
  checkLogic(check, fixture);
  checkOperatorForms(check, fixture);

  return failures;
}

/* ---------------------------------------------------------------- fixture -- */

interface CellSpec {
  id: string;
  label: string;
  polarity: NeuronPolarity;
  model: NeuronModelKind;
  /** Index into the population list, or -1 for none. */
  population: number;
  archetype: MorphologyArchetype;
  somaRadius: number;
  dendriteCount: number;
  dendriteDepth: number;
  dendriteLength: number;
  dendriteSpread: number;
  axonLength: number;
  rate: number;
  voltage: number;
  spikes: number;
  spiking: boolean;
  bias: number;
  noise: number;
  enabled: boolean;
  selected: boolean;
}

const CELLS: readonly CellSpec[] = [
  {
    id: 'n0',
    label: 'Pm 2',
    polarity: 'excitatory',
    model: 'izhikevich',
    population: 0,
    archetype: 'pyramidal',
    somaRadius: 1,
    dendriteCount: 4,
    dendriteDepth: 3,
    dendriteLength: 10,
    dendriteSpread: 0.6,
    axonLength: 20,
    rate: 0,
    voltage: -65,
    spikes: 0,
    spiking: false,
    bias: 50,
    noise: 10,
    enabled: true,
    selected: false,
  },
  {
    id: 'n1',
    label: 'PM 02',
    polarity: 'excitatory',
    model: 'lif',
    population: 0,
    archetype: 'pyramidal',
    somaRadius: 1,
    dendriteCount: 4,
    dendriteDepth: 3,
    dendriteLength: 10,
    dendriteSpread: 0.6,
    axonLength: 20,
    rate: 6,
    voltage: -60,
    spikes: 10,
    spiking: false,
    bias: 55,
    noise: 20,
    enabled: true,
    selected: true,
  },
  {
    id: 'n2',
    label: 'Pm3',
    polarity: 'inhibitory',
    model: 'lif',
    population: 1,
    archetype: 'basket',
    somaRadius: 0.6,
    dendriteCount: 8,
    dendriteDepth: 2,
    dendriteLength: 4,
    dendriteSpread: 1.2,
    axonLength: 6,
    rate: 12,
    voltage: -55,
    spikes: 40,
    spiking: false,
    bias: 60,
    noise: 30,
    enabled: true,
    selected: false,
  },
  {
    id: 'n3',
    label: 'LC10',
    polarity: 'excitatory',
    model: 'adex',
    population: 1,
    archetype: 'granule',
    somaRadius: 0.4,
    dendriteCount: 2,
    dendriteDepth: 1,
    dendriteLength: 2,
    dendriteSpread: 0.3,
    axonLength: 30,
    rate: 25,
    voltage: -50,
    spikes: 100,
    spiking: true,
    bias: 65,
    noise: 40,
    enabled: true,
    selected: false,
  },
  {
    id: 'n4',
    label: 'T4a',
    polarity: 'excitatory',
    model: 'izhikevich',
    population: -1,
    archetype: 'pyramidal',
    somaRadius: 0.98,
    dendriteCount: 4,
    dendriteDepth: 3,
    dendriteLength: 9.6,
    dendriteSpread: 0.62,
    axonLength: 19.5,
    rate: 3,
    voltage: -70,
    spikes: 5,
    spiking: false,
    bias: 70,
    noise: 50,
    enabled: true,
    selected: false,
  },
  {
    id: 'n5',
    label: '',
    polarity: 'inhibitory',
    model: 'lif',
    population: -1,
    archetype: 'purkinje',
    somaRadius: 1.4,
    dendriteCount: 12,
    dendriteDepth: 5,
    dendriteLength: 16,
    dendriteSpread: 1.5,
    axonLength: 3,
    rate: 0,
    voltage: -65,
    spikes: 0,
    spiking: false,
    bias: 75,
    noise: 60,
    enabled: false,
    selected: false,
  },
];

/** Ordered pairs, presynaptic slot first. */
const EDGES: readonly (readonly [number, number])[] = [
  [0, 2],
  [1, 2],
  [2, 3],
  [3, 4],
  [2, 0],
  [4, 5],
  [0, 3],
  [2, 1],
];

/** One conductance for every synapse, so a weight sum is visibly not a count. */
const EDGE_WEIGHT = 0.5;

const POPULATION_NAMES: readonly string[] = ['Alpha', 'Beta'];

function morphologyFor(spec: CellSpec, seed: number): Morphology {
  return {
    ...defaultMorphology(spec.archetype, seed),
    somaRadius: spec.somaRadius,
    dendriteCount: spec.dendriteCount,
    dendriteDepth: spec.dendriteDepth,
    dendriteLength: spec.dendriteLength,
    dendriteSpread: spec.dendriteSpread,
    axonLength: spec.axonLength,
  };
}

function receptorFor(polarity: NeuronPolarity): ReceptorKind {
  return polarity === 'inhibitory' ? 'gabaa' : 'ampa';
}

function buildFixture(): Fixture {
  const neurons: Neuron[] = CELLS.map((spec, index) => ({
    id: asNeuronId(spec.id),
    label: spec.label,
    position: { x: index * 10, y: 0, z: -index },
    params: defaultParams(spec.model),
    polarity: spec.polarity,
    morphology: morphologyFor(spec, 1000 + index),
    population: spec.population >= 0 ? asPopulationId(`p${spec.population}`) : null,
    bias: spec.bias,
    noise: spec.noise,
    enabled: spec.enabled,
  }));

  const populations: Population[] = POPULATION_NAMES.map((name, index) => {
    const members = neurons.filter((_, slot) => CELLS[slot].population === index);
    const polarity = members.length > 0 ? CELLS[neurons.indexOf(members[0])].polarity : 'excitatory';
    return {
      id: asPopulationId(`p${index}`),
      name,
      size: members.length,
      polarity,
      params: defaultParams('izhikevich'),
      morphology: defaultMorphology('pyramidal', index + 1),
      layout: { kind: 'grid', columns: 2, rows: 1, layers: 1, spacing: 4 },
      origin: { x: 0, y: 0, z: 0 },
      color: null,
      members: members.map((neuron) => neuron.id),
      collapsed: false,
    };
  });

  const synapses: Synapse[] = EDGES.map(([pre, post], index) => {
    const receptor = receptorFor(CELLS[pre].polarity);
    return {
      id: asSynapseId(`s${index}`),
      source: neurons[pre].id,
      target: neurons[post].id,
      receptor,
      weight: EDGE_WEIGHT,
      delay: 1,
      kinetics: { ...RECEPTOR_DEFAULTS[receptor] },
      plasticity: { ...DEFAULT_PLASTICITY },
      stp: { ...DEFAULT_STP },
      releaseProbability: 1,
      arc: 1.5,
      enabled: true,
    };
  });

  const circuit: Circuit = {
    id: asCircuitId('verify-query'),
    name: 'Query self-check',
    description: 'Hand-built fixture for the query language invariants.',
    version: CIRCUIT_SCHEMA_VERSION,
    createdAt: 0,
    updatedAt: 0,
    neurons,
    synapses,
    populations,
    projections: [],
    stimuli: [],
    probes: [],
    simulation: { ...DEFAULT_SIMULATION_SETTINGS },
    camera: { ...DEFAULT_CAMERA },
    render: { ...DEFAULT_RENDER_SETTINGS },
    tags: [],
  };

  // The buffers are filled the same way the engine fills them, so slot i is
  // circuit.neurons[i] exactly as it is in the running application.
  const buffers = allocateSimulationBuffers(16, 32);
  const nb = buffers.neurons;
  nb.count = CELLS.length;
  for (let i = 0; i < CELLS.length; i += 1) {
    const spec = CELLS[i];
    nb.position[i * 3] = i * 10;
    nb.position[i * 3 + 1] = 0;
    nb.position[i * 3 + 2] = -i;
    nb.model[i] = MODEL_CODE[spec.model];
    nb.polarity[i] = spec.polarity === 'inhibitory' ? 1 : 0;
    nb.enabled[i] = spec.enabled ? 1 : 0;
    nb.flags[i] = spec.selected ? NEURON_FLAG.SELECTED : 0;
    nb.rate[i] = spec.rate;
    nb.v[i] = spec.voltage;
    nb.spikeCount[i] = spec.spikes;
    nb.spike[i] = spec.spiking ? 1 : 0;
    nb.bias[i] = spec.bias;
    nb.noise[i] = spec.noise;
    nb.seed[i] = 1000 + i;
    nb.population[i] = spec.population >= 0 ? spec.population : 0xffff;
  }

  const sb = buffers.synapses;
  sb.count = EDGES.length;
  for (let s = 0; s < EDGES.length; s += 1) {
    const [pre, post] = EDGES[s];
    sb.pre[s] = pre;
    sb.post[s] = post;
    sb.weight[s] = EDGE_WEIGHT;
    sb.delay[s] = 1;
    sb.enabled[s] = 1;
    sb.receptor[s] = RECEPTOR_CODE[receptorFor(CELLS[pre].polarity)];
  }

  return { circuit, buffers };
}

/* ----------------------------------------------------------------- helpers -- */

/** Slots a query matches, or null when it failed to parse. */
function slotsOf(fixture: Fixture, source: string): number[] | null {
  const parsed = parseQuery(source);
  if (!parsed.ok) return null;
  return matchedSlots(evaluateQuery(parsed.ast, fixture.buffers, fixture.circuit));
}

function sameSlots(actual: number[] | null, expected: readonly number[]): boolean {
  if (actual === null || actual.length !== expected.length) return false;
  for (let i = 0; i < expected.length; i += 1) {
    if (actual[i] !== expected[i]) return false;
  }
  return true;
}

/** Assert a query matches exactly these slots, in slot order. */
function expectSlots(
  check: Check,
  fixture: Fixture,
  name: string,
  source: string,
  expected: readonly number[],
): void {
  check(name, sameSlots(slotsOf(fixture, source), expected));
}

/** Assert a query is rejected, optionally at a particular offset. */
function expectError(check: Check, name: string, source: string, offset?: number): void {
  const parsed = parseQuery(source);
  if (parsed.ok) {
    check(name, false);
    return;
  }
  check(name, parsed.error.message.length > 0 && (offset === undefined || parsed.error.offset === offset));
}

/**
 * Structural equality over ASTs, ignoring source offsets. The two spellings of
 * an operator sit at different columns by construction, so comparing positions
 * would only ever prove that `{and}` is longer than `&&`.
 */
function sameTree(a: QueryNode, b: QueryNode): boolean {
  const withoutOffsets = (key: string, value: unknown): unknown =>
    key === 'offset' ? undefined : value;
  return JSON.stringify(a, withoutOffsets) === JSON.stringify(b, withoutOffsets);
}

function astOf(source: string): QueryNode | null {
  const parsed = parseQuery(source);
  return parsed.ok ? parsed.ast : null;
}

/* ------------------------------------------------------------------ tables -- */

function checkTables(check: Check): void {
  check(
    'table/bracket-form-matches-id',
    QUERY_OPERATORS.every((spec) => spec.bracket === `{${spec.id}}`),
  );

  const symbols = new Set(QUERY_OPERATORS.map((spec) => spec.symbol));
  check('table/symbols-unique', symbols.size === QUERY_OPERATORS.length);

  const ids = new Set(QUERY_OPERATORS.map((spec) => spec.id));
  check('table/ids-unique', ids.size === QUERY_OPERATORS.length);

  const names = new Set(QUERY_FIELDS.map((field) => field.name));
  check('table/field-names-unique', names.size === QUERY_FIELDS.length);
  check(
    'table/field-names-lowercase',
    QUERY_FIELDS.every((field) => field.name === field.name.toLowerCase()),
  );
  check(
    'table/every-field-resolves',
    QUERY_FIELDS.every((field) => findField(field.name) === field),
  );
  check('table/field-lookup-ignores-case', findField('LABEL') === findField('label'));

  // Every operator declares fieldTypes iff it takes a field on either side.
  check(
    'table/field-types-match-shape',
    QUERY_OPERATORS.every((spec) =>
      spec.shape === 'field' || spec.shape === 'attribute'
        ? spec.fieldTypes.length > 0
        : spec.fieldTypes.length === 0,
    ),
  );
}

/* ------------------------------------------------------------------- lexer -- */

function checkLexer(check: Check): void {
  // Longest match first, applied to every symbol in the table at once: each one
  // must lex to itself and not to a shorter operator that prefixes it.
  let allSymbols = true;
  let allBrackets = true;
  for (const spec of QUERY_OPERATORS) {
    const bySymbol = tokenize(`${spec.symbol} x`);
    if (!bySymbol.ok || bySymbol.tokens[0].operator?.id !== spec.id) allSymbols = false;
    const byBracket = tokenize(`${spec.bracket} x`);
    if (!byBracket.ok || byBracket.tokens[0].operator?.id !== spec.id) allBrackets = false;
  }
  check('lex/every-symbol-lexes-to-itself', allSymbols);
  check('lex/every-bracket-lexes-to-itself', allBrackets);

  const notEqual = tokenize('label != x');
  check(
    'lex/not-equal-is-one-token',
    notEqual.ok && notEqual.tokens.length === 4 && notEqual.tokens[1].operator?.id === 'not_equal',
  );

  const upstreamAll = tokenize('~UA x');
  check(
    'lex/upstream-all-beats-similar-upstream',
    upstreamAll.ok && upstreamAll.tokens[0].operator?.id === 'upstream_all',
  );

  const tight = tokenize('rate>=5');
  check(
    'lex/operators-need-no-surrounding-space',
    tight.ok &&
      tight.tokens.length === 4 &&
      tight.tokens[0].value === 'rate' &&
      tight.tokens[1].operator?.id === 'gte' &&
      tight.tokens[2].value === '5',
  );

  const quoted = tokenize('label == "Pm 2"');
  check(
    'lex/quoted-value-keeps-spaces',
    quoted.ok && quoted.tokens[2].value === 'Pm 2' && quoted.tokens[2].quoted,
  );

  const grouped = tokenize('{{a}}');
  check(
    'lex/double-braces-are-grouping',
    grouped.ok &&
      grouped.tokens[0].kind === 'group-open' &&
      grouped.tokens[2].kind === 'group-close',
  );

  const negative = tokenize('bias > -5');
  check('lex/negative-numbers-are-words', negative.ok && negative.tokens[2].value === '-5');
}

/* -------------------------------------------------------------- precedence -- */

function checkPrecedence(check: Check): void {
  const empty = astOf('');
  check('parse/empty-is-the-all-node', empty !== null && empty.kind === 'all');

  const blank = astOf('    ');
  check('parse/whitespace-is-the-all-node', blank !== null && blank.kind === 'all');

  const orAnd = astOf('rate > 1 || rate > 2 && rate > 3');
  check(
    'parse/or-binds-looser-than-and',
    orAnd !== null && orAnd.kind === 'or' && orAnd.right.kind === 'and',
  );

  const andOr = astOf('rate > 1 && rate > 2 || rate > 3');
  check(
    'parse/and-binds-tighter-on-the-left',
    andOr !== null && andOr.kind === 'or' && andOr.left.kind === 'and',
  );

  const chain = astOf('rate > 1 && rate > 2 && rate > 3');
  check(
    'parse/and-is-left-associative',
    chain !== null && chain.kind === 'and' && chain.left.kind === 'and',
  );

  const orChain = astOf('rate > 1 || rate > 2 || rate > 3');
  check(
    'parse/or-is-left-associative',
    orChain !== null && orChain.kind === 'or' && orChain.left.kind === 'or',
  );

  const grouped = astOf('{{rate > 1 || rate > 2}} && rate > 3');
  check(
    'parse/grouping-overrides-precedence',
    grouped !== null && grouped.kind === 'and' && grouped.left.kind === 'or',
  );

  const notAnd = astOf('!! rate > 1 && rate > 2');
  check(
    'parse/not-binds-tighter-than-and',
    notAnd !== null && notAnd.kind === 'and' && notAnd.left.kind === 'not',
  );

  const doubleNot = astOf('!! !! rate > 1');
  check(
    'parse/not-nests',
    doubleNot !== null && doubleNot.kind === 'not' && doubleNot.operand.kind === 'not',
  );

  const nested = astOf('{{ {{rate > 1}} }}');
  check('parse/groups-nest', nested !== null && nested.kind === 'numeric-compare');
}

/* ------------------------------------------------------------ parse errors -- */

function checkParseErrors(check: Check): void {
  expectError(check, 'error/string-operator-on-numeric-field', 'rate ~= foo');
  expectError(check, 'error/contains-on-numeric-field', 'degree >> 4');
  expectError(check, 'error/numeric-operator-on-string-field', 'label > 5');
  expectError(check, 'error/between-on-string-field', 'label <> 1..2');
  expectError(check, 'error/like-on-boolean-field', 'enabled ~= yes');
  expectError(check, 'error/unknown-field', 'nosuchfield == 1');
  expectError(check, 'error/unknown-bracket-operator', 'label {nope} x');
  expectError(check, 'error/missing-value', 'label ==');
  expectError(check, 'error/non-numeric-value', 'rate > abc');
  expectError(check, 'error/malformed-range', 'rate <> 5');
  expectError(check, 'error/trailing-logical-operator', 'label == a &&');
  expectError(check, 'error/leading-logical-operator', '&& label == a');
  expectError(check, 'error/bare-term-without-operator', 'pm2');
  expectError(check, 'error/two-terms-without-operator', 'label == a label == b');
  expectError(check, 'error/quoted-field-name', '"label" == a');
  expectError(check, 'error/empty-group', '{{}}');
  expectError(check, 'error/missing-field-after-has', '$$ 42');
  expectError(check, 'error/empty-list-item', 'model << lif,,adex');
  expectError(check, 'error/list-without-commas', 'model << lif adex');

  // Offsets have to point at the token that is wrong, so a query box can
  // underline it rather than the whole line.
  const unclosed = '{{label == a';
  expectError(check, 'error/unclosed-group', unclosed, unclosed.length);

  const typeMismatch = 'label == a && rate ~= x';
  expectError(check, 'error/offset-points-at-the-operator', typeMismatch, typeMismatch.indexOf('~='));

  const stray = 'label == a & b';
  expectError(check, 'error/stray-reserved-character', stray, stray.indexOf('&'));

  const unterminated = 'label == "abc';
  expectError(check, 'error/unterminated-quote', unterminated, unterminated.indexOf('"'));

  const suggestion = parseQuery('lable == a');
  check(
    'error/suggests-the-nearest-field',
    !suggestion.ok && suggestion.error.message.includes('label'),
  );

  // A query too big to walk by recursion has to come back as an error, not as a
  // stack overflow thrown out of a function documented never to throw.
  const longChain = new Array(MAX_QUERY_TERMS + 1).fill('rate > 5').join(' || ');
  expectError(check, 'error/too-many-terms', longChain);
  const deepGroups = `${'{{'.repeat(MAX_QUERY_DEPTH + 1)}rate > 5${'}}'.repeat(MAX_QUERY_DEPTH + 1)}`;
  expectError(check, 'error/too-deeply-nested', deepGroups);
  const deepNots = `${'!!'.repeat(MAX_QUERY_DEPTH + 1)}rate > 5`;
  expectError(check, 'error/too-many-prefixes', deepNots);
  check(
    'parse/a-query-at-the-ceiling-still-parses',
    parseQuery(new Array(MAX_QUERY_TERMS).fill('rate > 5').join(' || ')).ok,
  );
  check(
    'parse/nesting-at-the-ceiling-still-parses',
    parseQuery(`${'{{'.repeat(MAX_QUERY_DEPTH)}rate > 5${'}}'.repeat(MAX_QUERY_DEPTH)}`).ok,
  );

  // And the counterpart: things that must not be errors.
  check('parse/valid-query-parses', parseQuery('rate > 5 && polarity == inhibitory').ok);
  check('parse/quoted-value-parses', parseQuery('label == "Pm 2"').ok);
  check('parse/negative-number-parses', parseQuery('voltage < -60').ok);
  check('parse/decimal-range-parses', parseQuery('rate <> 0.5..1.5').ok);
  check('parse/reversed-range-is-accepted', parseQuery('rate <> 20..5').ok);
}

/* -------------------------------------------------------------------- like -- */

function checkLike(check: Check, fixture: Fixture): void {
  check(
    'like/case-punctuation-and-spacing-collapse',
    likeKey('Pm2') === likeKey('pm-2') && likeKey('pm-2') === likeKey('PM 02'),
  );
  check('like/leading-zeros-are-stripped', likeKey('LC010') === likeKey('lc10'));
  check('like/part-order-is-ignored', likeKey('LC 10 a') === likeKey('a 10 lc'));
  check('like/different-numbers-stay-different', likeKey('Pm2') !== likeKey('Pm3'));
  check('like/different-letters-stay-different', likeKey('Pm2') !== likeKey('Tm2'));
  check('like/parts-do-not-run-together', likeKey('ab c') !== likeKey('a bc'));
  check('like/no-alphanumerics-is-empty', likeKey('--/--') === '');

  expectSlots(check, fixture, 'like/matches-every-spelling', 'label ~= pm2', [0, 1]);
  expectSlots(check, fixture, 'like/matches-from-a-spaced-value', 'label ~= "PM 02"', [0, 1]);
  expectSlots(check, fixture, 'like/matches-from-a-hyphenated-value', 'label ~= pm-2', [0, 1]);
  expectSlots(check, fixture, 'like/does-not-over-match', 'label ~= pm3', [2]);
}

/* ---------------------------------------------------------- string queries -- */

function checkStringOperators(check: Check, fixture: Fixture): void {
  expectSlots(check, fixture, 'op/equal', 'polarity == inhibitory', [2, 5]);
  expectSlots(check, fixture, 'op/equal-ignores-case', 'polarity == INHIBITORY', [2, 5]);
  expectSlots(check, fixture, 'op/not-equal', 'label != Pm3', [0, 1, 3, 4, 5]);
  expectSlots(check, fixture, 'op/starts-with', 'label ^* Pm', [0, 1, 2]);
  expectSlots(check, fixture, 'op/ends-with', 'label ^$ 2', [0, 1]);
  expectSlots(check, fixture, 'op/contains', 'label >> M', [0, 1, 2]);
  expectSlots(check, fixture, 'op/not-contains', 'label !> M', [3, 4, 5]);
  expectSlots(check, fixture, 'op/in', 'model << lif,adex', [1, 2, 3, 5]);
  expectSlots(check, fixture, 'op/not-in', 'model !< lif,adex', [0, 4]);
  expectSlots(check, fixture, 'op/in-tolerates-spacing', 'model << lif, adex', [1, 2, 3, 5]);
  expectSlots(check, fixture, 'op/in-tolerates-spaced-commas', 'model << lif , adex', [1, 2, 3, 5]);
  expectSlots(check, fixture, 'op/in-with-a-quoted-item', 'label << "Pm 2",LC10', [0, 3]);
  expectSlots(check, fixture, 'op/id-field', 'id == n3', [3]);
  expectSlots(check, fixture, 'op/population-name', 'population == Alpha', [0, 1]);
  expectSlots(check, fixture, 'op/archetype', 'archetype == pyramidal', [0, 1, 4]);
  // Receptor is multi-valued and read off efferent synapses; only the
  // inhibitory cell with outputs drives GABA-A.
  expectSlots(check, fixture, 'op/receptor', 'receptor == gabaa', [2]);
  expectSlots(check, fixture, 'op/receptor-ampa', 'receptor == ampa', [0, 1, 3, 4]);
}

/* --------------------------------------------------------- numeric queries -- */

function checkNumericOperators(check: Check, fixture: Fixture): void {
  expectSlots(check, fixture, 'op/greater-than', 'rate > 6', [2, 3]);
  expectSlots(check, fixture, 'op/at-least', 'rate >= 6', [1, 2, 3]);
  expectSlots(check, fixture, 'op/less-than', 'voltage < -60', [0, 4, 5]);
  expectSlots(check, fixture, 'op/at-most', 'voltage <= -60', [0, 1, 4, 5]);
  expectSlots(check, fixture, 'op/between', 'rate <> 5..20', [1, 2]);
  expectSlots(check, fixture, 'op/between-is-inclusive', 'rate <> 6..12', [1, 2]);
  expectSlots(check, fixture, 'op/reversed-range-reads-the-same', 'rate <> 20..5', [1, 2]);
  expectSlots(check, fixture, 'op/numeric-equal', 'rate == 12', [2]);
  expectSlots(check, fixture, 'op/numeric-not-equal', 'rate != 0', [1, 2, 3, 4]);
  expectSlots(check, fixture, 'op/numeric-in', 'out_degree << 0,3', [2, 5]);
  expectSlots(check, fixture, 'op/numeric-not-in', 'out_degree !< 0,3', [0, 1, 3, 4]);

  // Degrees count synapses; the hand-derived tables are in the file header.
  expectSlots(check, fixture, 'op/degree', 'degree >= 3', [0, 2, 3]);
  expectSlots(check, fixture, 'op/in-degree', 'in_degree == 2', [2, 3]);
  expectSlots(check, fixture, 'op/out-degree', 'out_degree == 1', [1, 3, 4]);
  expectSlots(check, fixture, 'op/isolated-outputs', 'out_degree == 0', [5]);

  // Conductance sums, not counts: every synapse carries 0.5 nS.
  expectSlots(check, fixture, 'op/weight-out', 'weight_out > 1', [2]);
  expectSlots(check, fixture, 'op/weight-in', 'weight_in >= 1', [2, 3]);

  expectSlots(check, fixture, 'op/spike-count', 'spikes > 10', [2, 3]);
  expectSlots(check, fixture, 'op/bias', 'bias > 62', [3, 4, 5]);
  expectSlots(check, fixture, 'op/noise', 'noise <= 20', [0, 1]);
  expectSlots(check, fixture, 'op/position-x', 'x > 25', [3, 4, 5]);
  expectSlots(check, fixture, 'op/position-y', 'y == 0', [0, 1, 2, 3, 4, 5]);
  expectSlots(check, fixture, 'op/position-z', 'z < -3', [4, 5]);
}

/* ------------------------------------------------- booleans and presence -- */

function checkBooleanAndPresence(check: Check, fixture: Fixture): void {
  expectSlots(check, fixture, 'op/enabled-false', 'enabled == false', [5]);
  expectSlots(check, fixture, 'op/enabled-true', 'enabled == true', [0, 1, 2, 3, 4]);
  expectSlots(check, fixture, 'op/selected', 'selected == true', [1]);
  expectSlots(check, fixture, 'op/spiking', 'spiking == true', [3]);
  expectSlots(check, fixture, 'op/boolean-not-equal', 'enabled != true', [5]);
  expectSlots(check, fixture, 'op/boolean-accepts-yes', 'spiking == yes', [3]);

  expectSlots(check, fixture, 'op/has-population', '$$ population', [0, 1, 2, 3]);
  expectSlots(check, fixture, 'op/missing-population', '!$ population', [4, 5]);
  expectSlots(check, fixture, 'op/missing-label', '!$ label', [5]);
  expectSlots(check, fixture, 'op/has-label', '$$ label', [0, 1, 2, 3, 4]);
  // The one cell with no outputs drives no receptor anywhere.
  expectSlots(check, fixture, 'op/missing-receptor', '!$ receptor', [5]);
  // A boolean is never absent: false is a value.
  expectSlots(check, fixture, 'op/boolean-is-always-present', '$$ enabled', [0, 1, 2, 3, 4, 5]);
}

/* ---------------------------------------------------------- connectivity -- */

function checkConnectivity(check: Check, fixture: Fixture): void {
  // in-neighbours of slot 2 are {0,1}; out-neighbours are {0,1,3}.
  expectSlots(check, fixture, 'op/upstream', '^^ Pm3', [0, 1]);
  expectSlots(check, fixture, 'op/downstream', '!^ Pm3', [0, 1, 3]);
  // Both directions with slot 2 at once.
  expectSlots(check, fixture, 'op/reciprocal', '^v Pm3', [0, 1]);

  // "pm2" is a loose match, so it names slots 0 and 1 together. Cells
  // presynaptic to both are in(0) ∩ in(1) = {2}; postsynaptic to both are
  // out(0) ∩ out(1) = {2,3} ∩ {2} = {2}.
  expectSlots(check, fixture, 'op/upstream-all', '~UA pm2', [2]);
  expectSlots(check, fixture, 'op/downstream-all', '~DA pm2', [2]);
  // The plain forms take the union instead, which is a strictly larger answer.
  expectSlots(check, fixture, 'op/upstream-of-several', '^^ pm2', [2]);
  expectSlots(check, fixture, 'op/downstream-of-several', '!^ pm2', [2, 3]);

  // Shortest route from slot 0 to slot 4 is 0->3->4 at two hops; 0->2->3->4 is
  // three and so contributes nothing.
  expectSlots(check, fixture, 'op/pathways', '"Pm 2" => T4a', [0, 3, 4]);
  // Slot 5 has no outgoing edge, so nothing leads back from it.
  expectSlots(check, fixture, 'op/pathways-unreachable', 'T4a => "Pm 2"', []);
  expectSlots(check, fixture, 'op/pathways-single-hop', 'LC10 => T4a', [3, 4]);

  // An id resolves ahead of any label, and an unknown value matches nothing
  // while still returning a real answer.
  expectSlots(check, fixture, 'op/upstream-by-id', '^^ n2', [0, 1]);
  expectSlots(check, fixture, 'op/unresolved-value-matches-nothing', '^^ nosuchcell', []);

  const parsed = parseQuery('^^ nosuchcell');
  const result = parsed.ok
    ? evaluateQuery(parsed.ast, fixture.buffers, fixture.circuit)
    : null;
  check('op/unresolved-value-warns', result !== null && result.warnings.length > 0);

  // An `and` runs its cheap side first and skips the other side entirely when
  // that matched nothing. Whether a name exists is a fact about the query rather
  // than about the branch it sits in, so the warning has to survive that.
  const skipped = parseQuery('rate > 1000000 && ^^ nosuchcell');
  const skippedResult = skipped.ok
    ? evaluateQuery(skipped.ast, fixture.buffers, fixture.circuit)
    : null;
  check(
    'op/unresolved-value-warns-from-a-skipped-branch',
    skippedResult !== null && skippedResult.count === 0 && skippedResult.warnings.length > 0,
  );

  // And the counterpart: a query that resolved everything says nothing.
  const quiet = parseQuery('^^ Pm3');
  const quietResult = quiet.ok ? evaluateQuery(quiet.ast, fixture.buffers, fixture.circuit) : null;
  check('op/resolved-value-does-not-warn', quietResult !== null && quietResult.warnings.length === 0);
}

/* ------------------------------------------------------------- similarity -- */

function checkSimilarity(check: Check, fixture: Fixture): void {
  // Slots 0 and 1 send only to Beta and receive only from Beta, so their full
  // fingerprints are identical; nothing else reaches the 0.6 cosine.
  expectSlots(check, fixture, 'op/similar-connectivity', '~c "Pm 2"', [0, 1]);
  // Input profiles: slots 0, 1 and 4 all hear only from Beta, and slot 3 hears
  // from Alpha and Beta equally, which is cos 0.707.
  expectSlots(check, fixture, 'op/similar-connectivity-upstream', '~u "Pm 2"', [0, 1, 3, 4]);
  // Output profiles: only slots 0 and 1 project solely into Beta.
  expectSlots(check, fixture, 'op/similar-connectivity-downstream', '~d "Pm 2"', [0, 1]);
  // The three read the same network and must not agree by accident.
  check(
    'op/similarity-directions-differ',
    !sameSlots(slotsOf(fixture, '~u "Pm 2"'), slotsOf(fixture, '~c "Pm 2"') ?? []),
  );

  // Slot 1 is an exact copy of slot 0's morphology and slot 4 is within a
  // seventh of the document's spread on every descriptor; the other three
  // differ in archetype, which alone puts them past the threshold.
  expectSlots(check, fixture, 'op/similar-shape', '~~ "Pm 2"', [0, 1, 4]);
  check(
    'op/similar-shape-differs-from-wiring',
    !sameSlots(slotsOf(fixture, '~~ "Pm 2"'), slotsOf(fixture, '~c "Pm 2"') ?? []),
  );
  expectSlots(check, fixture, 'op/similar-shape-unresolved', '~~ nosuchcell', []);
}

/* ------------------------------------------------------------------ logic -- */

function checkLogic(check: Check, fixture: Fixture): void {
  const all = [0, 1, 2, 3, 4, 5];

  // Every cell is excitatory or inhibitory, and the lif cells are 1, 2 and 5.
  // `a || b && c` is `a || (b && c)`, which is every cell; grouping the or
  // instead narrows it to the lif cells.
  expectSlots(
    check,
    fixture,
    'logic/or-and-precedence-in-the-answer',
    'polarity == excitatory || polarity == inhibitory && model == lif',
    all,
  );
  expectSlots(
    check,
    fixture,
    'logic/grouping-changes-the-answer',
    '{{polarity == excitatory || polarity == inhibitory}} && model == lif',
    [1, 2, 5],
  );

  expectSlots(check, fixture, 'logic/not', '!! polarity == inhibitory', [0, 1, 3, 4]);
  expectSlots(check, fixture, 'logic/double-not-is-identity', '!! !! polarity == inhibitory', [2, 5]);
  expectSlots(check, fixture, 'logic/and', 'polarity == excitatory && rate > 1', [1, 3, 4]);
  expectSlots(check, fixture, 'logic/or', 'model == adex || model == lif', [1, 2, 3, 5]);

  // An empty query is everything, not nothing: an untouched filter box must not
  // hide the network.
  expectSlots(check, fixture, 'logic/empty-query-matches-everything', '', all);
  expectSlots(check, fixture, 'logic/whitespace-query-matches-everything', '   ', all);

  // A conjunction whose cheap side is empty must still be empty, and must not
  // depend on the expensive side having run.
  expectSlots(check, fixture, 'logic/and-with-empty-side', 'label == nothing && ~c "Pm 2"', []);
  expectSlots(check, fixture, 'logic/and-with-empty-side-reversed', '~c "Pm 2" && label == nothing', []);
  expectSlots(check, fixture, 'logic/or-with-empty-side', 'label == nothing || model == adex', [3]);

  // Combining the operator families is the point of the language.
  expectSlots(
    check,
    fixture,
    'logic/connectivity-with-attributes',
    '^^ Pm3 && polarity == excitatory',
    [0, 1],
  );
  expectSlots(
    check,
    fixture,
    'logic/pathways-with-attributes',
    '"Pm 2" => T4a && !$ population',
    [4],
  );

  const parsed = parseQuery('rate > 6');
  const result = parsed.ok ? evaluateQuery(parsed.ast, fixture.buffers, fixture.circuit) : null;
  check(
    'result/mask-covers-every-slot',
    result !== null && result.mask.length === 6 && result.total === 6 && result.count === 2,
  );
}

/* -------------------------------------------------- bracket / symbol forms -- */

function checkOperatorForms(check: Check, fixture: Fixture): void {
  const pairs: readonly (readonly [string, string])[] = [
    ['label {starts_with} Pm', 'label ^* Pm'],
    ['label {ends_with} 2', 'label ^$ 2'],
    ['label {like} pm2', 'label ~= pm2'],
    ['label {contains} M', 'label >> M'],
    ['label {not_contains} M', 'label !> M'],
    ['label {not_equal} Pm3', 'label != Pm3'],
    ['polarity {equal} inhibitory', 'polarity == inhibitory'],
    ['model {in} lif,adex', 'model << lif,adex'],
    ['model {not_in} lif,adex', 'model !< lif,adex'],
    ['rate {gt} 6', 'rate > 6'],
    ['rate {gte} 6', 'rate >= 6'],
    ['voltage {lt} -60', 'voltage < -60'],
    ['voltage {lte} -60', 'voltage <= -60'],
    ['rate {between} 5..20', 'rate <> 5..20'],
    ['{upstream} Pm3', '^^ Pm3'],
    ['{downstream} Pm3', '!^ Pm3'],
    ['{upstream_all} pm2', '~UA pm2'],
    ['{downstream_all} pm2', '~DA pm2'],
    ['{reciprocal} Pm3', '^v Pm3'],
    ['"Pm 2" {pathways} T4a', '"Pm 2" => T4a'],
    ['{similar_connectivity} "Pm 2"', '~c "Pm 2"'],
    ['{similar_connectivity_upstream} "Pm 2"', '~u "Pm 2"'],
    ['{similar_connectivity_downstream} "Pm 2"', '~d "Pm 2"'],
    ['{similar_shape} "Pm 2"', '~~ "Pm 2"'],
    ['{has} population', '$$ population'],
    ['{missing} label', '!$ label'],
    ['rate > 1 {and} rate > 2', 'rate > 1 && rate > 2'],
    ['rate > 1 {or} rate > 2', 'rate > 1 || rate > 2'],
    ['{not} rate > 1', '!! rate > 1'],
  ];

  let treesAgree = true;
  let answersAgree = true;
  for (const [bracket, symbol] of pairs) {
    const a = astOf(bracket);
    const b = astOf(symbol);
    if (a === null || b === null || !sameTree(a, b)) {
      treesAgree = false;
      continue;
    }
    if (!sameSlots(slotsOf(fixture, bracket), slotsOf(fixture, symbol) ?? [-1])) {
      answersAgree = false;
    }
  }
  check('forms/bracket-and-symbol-parse-alike', treesAgree);
  check('forms/bracket-and-symbol-answer-alike', answersAgree);

  // Every operator in the table is exercised by one of the pairs above, so a
  // new operator cannot be added without a matching invariant.
  const covered = new Set<string>();
  for (const [bracket] of pairs) {
    for (const spec of QUERY_OPERATORS) {
      if (bracket.includes(spec.bracket)) covered.add(spec.id);
    }
  }
  check('forms/every-operator-is-covered', covered.size === QUERY_OPERATORS.length);
}
