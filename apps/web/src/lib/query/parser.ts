/**
 * Precedence-climbing parser for the cell query language.
 *
 * Precedence, loosest first:
 *
 *   or  ->  `a || b`
 *   and ->  `a && b`
 *   not ->  `!! a`
 *   {{ }} grouping, which overrides all of it
 *
 * so `a || b && c` is `a || (b && c)` and `{{a || b}} && c` is not. Both binary
 * operators are left-associative, which is what the climbing loop below does by
 * recursing at `precedence + 1`.
 *
 * The parser is also the type checker. A field's declared type decides which
 * operators may be applied to it and how the right-hand side is read, so
 * `rate ~= foo` fails here with a message about `{like}` needing text, rather
 * than evaluating to an empty set that looks like a network with no such cell.
 * A query language that answers "nothing matched" when it means "you wrote that
 * wrong" is worse than no query language.
 *
 * Errors carry the offset and length of the offending token so a query box can
 * underline it, and every message names what was expected there.
 */

import type { QueryField } from './fields';
import { QUERY_FIELD_NAMES, findField } from './fields';
import type { OperatorSpec, Token } from './lexer';
import { tokenize } from './lexer';

/* -------------------------------------------------------------------- ast -- */

export type StringOperator =
  | 'equal'
  | 'not_equal'
  | 'like'
  | 'starts_with'
  | 'ends_with'
  | 'contains'
  | 'not_contains'
  | 'in'
  | 'not_in';

export type NumericOperator =
  | 'equal'
  | 'not_equal'
  | 'gt'
  | 'gte'
  | 'lt'
  | 'lte'
  | 'between'
  | 'in'
  | 'not_in';

export type ConnectivityOperator =
  | 'upstream'
  | 'downstream'
  | 'upstream_all'
  | 'downstream_all'
  | 'reciprocal';

export type SimilarityOperator =
  | 'similar_connectivity'
  | 'similar_connectivity_upstream'
  | 'similar_connectivity_downstream'
  | 'similar_shape';

/** An empty query. Matches every cell, which is what an empty filter means. */
export interface AllNode {
  kind: 'all';
}

export interface AndNode {
  kind: 'and';
  left: QueryNode;
  right: QueryNode;
}

export interface OrNode {
  kind: 'or';
  left: QueryNode;
  right: QueryNode;
}

export interface NotNode {
  kind: 'not';
  operand: QueryNode;
}

export interface StringCompareNode {
  kind: 'string-compare';
  op: StringOperator;
  field: string;
  /** One entry for the single-value operators, several for `{in}` / `{not_in}`. */
  values: readonly string[];
  offset: number;
}

export interface NumericCompareNode {
  kind: 'numeric-compare';
  op: NumericOperator;
  field: string;
  /** `[low, high]` for `{between}`, a list for `{in}`, one entry otherwise. */
  values: readonly number[];
  offset: number;
}

export interface BooleanCompareNode {
  kind: 'boolean-compare';
  op: 'equal' | 'not_equal';
  field: string;
  value: boolean;
  offset: number;
}

export interface PresenceNode {
  kind: 'presence';
  op: 'has' | 'missing';
  field: string;
  offset: number;
}

export interface ConnectivityNode {
  kind: 'connectivity';
  op: ConnectivityOperator;
  /** A cell label or id; may resolve to several cells. */
  value: string;
  offset: number;
}

export interface SimilarityNode {
  kind: 'similarity';
  op: SimilarityOperator;
  value: string;
  offset: number;
}

export interface PathwaysNode {
  kind: 'pathways';
  source: string;
  target: string;
  offset: number;
}

export type QueryNode =
  | AllNode
  | AndNode
  | OrNode
  | NotNode
  | StringCompareNode
  | NumericCompareNode
  | BooleanCompareNode
  | PresenceNode
  | ConnectivityNode
  | SimilarityNode
  | PathwaysNode;

export interface QueryError {
  message: string;
  /** Character offset into the source where the problem starts. */
  offset: number;
  /** Length of the offending span; zero-width at end of input. */
  length: number;
}

export type ParseResult = { ok: true; ast: QueryNode } | { ok: false; error: QueryError };

/* ------------------------------------------------------------- complexity -- */

/**
 * Leaf clauses one query may contain.
 *
 * Both this parser and the evaluator walk the tree by recursion, and a chain of
 * `a || b || c || …` builds a tree as deep as it is long. Past a few thousand
 * terms that recursion exhausts the JavaScript stack, and a `RangeError` thrown
 * out of a function documented never to throw reaches a query box as a blank
 * panel rather than a message. Bounding the term count bounds the depth of the
 * tree — a tree of T leaves is at most T deep — so one ceiling covers the
 * parser, `costOf` and `evalNode` at once.
 *
 * Five hundred clauses is far past anything written by hand and far short of
 * where a browser stack gives out, and asking for more is now an ordinary parse
 * error that names where the query got too big.
 */
export const MAX_QUERY_TERMS = 512;

/** Nesting of `{{ }}` groups and `!!` prefixes, bounded for the same reason. */
export const MAX_QUERY_DEPTH = 64;

/* ----------------------------------------------------------------- errors -- */

class ParseFailure extends Error {
  readonly error: QueryError;

  constructor(error: QueryError) {
    super(error.message);
    this.name = 'ParseFailure';
    this.error = error;
  }
}

function fail(message: string, offset: number, length: number): never {
  throw new ParseFailure({ message, offset, length: Math.max(0, length) });
}

/** How a token reads inside an error message. */
function describe(token: Token): string {
  switch (token.kind) {
    case 'end':
      return 'the end of the query';
    case 'operator':
      return `the operator ${token.operator?.bracket ?? token.text} (${token.text})`;
    case 'group-open':
      return '{{';
    case 'group-close':
      return '}}';
    default:
      return `"${token.value}"`;
  }
}

/** Article-correct type name for a message: "a string field", "a numeric field". */
function typeNames(types: readonly string[]): string {
  if (types.length === 1) return `a ${types[0]}`;
  if (types.length === 2) return `a ${types[0]} or ${types[1]}`;
  return `a ${types.slice(0, -1).join(', ')} or ${types[types.length - 1]}`;
}

/** Levenshtein distance, capped where it stops being a plausible typo. */
function editDistance(a: string, b: string): number {
  const rows = a.length + 1;
  const cols = b.length + 1;
  let previous = new Uint16Array(cols);
  let current = new Uint16Array(cols);
  for (let j = 0; j < cols; j += 1) previous[j] = j;
  for (let i = 1; i < rows; i += 1) {
    current[0] = i;
    for (let j = 1; j < cols; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      const deletion = previous[j] + 1;
      const insertion = current[j - 1] + 1;
      const substitution = previous[j - 1] + cost;
      current[j] = Math.min(deletion, insertion, substitution);
    }
    const swap = previous;
    previous = current;
    current = swap;
  }
  return previous[cols - 1];
}

/** The field name closest to a misspelling, when one is close enough to suggest. */
function nearestField(name: string): string | null {
  const needle = name.toLowerCase();
  let best: string | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of QUERY_FIELD_NAMES) {
    const distance = editDistance(needle, candidate);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }
  // Beyond a third of the name's length the "suggestion" is noise. Rounded up,
  // so a transposition — two edits, and the commonest typo there is — still
  // resolves on a name as short as "label".
  return best !== null && bestDistance <= Math.max(1, Math.ceil(needle.length / 3)) ? best : null;
}

/* ----------------------------------------------------------------- values -- */

/** Strict numeric literal: no empty string, no whitespace-only, no Infinity. */
function toNumber(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === '') return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

const TRUE_WORDS: ReadonlySet<string> = new Set(['true', 'yes', '1', 'on']);
const FALSE_WORDS: ReadonlySet<string> = new Set(['false', 'no', '0', 'off']);

function toBoolean(text: string): boolean | null {
  const lowered = text.trim().toLowerCase();
  if (TRUE_WORDS.has(lowered)) return true;
  if (FALSE_WORDS.has(lowered)) return false;
  return null;
}

/* ----------------------------------------------------------------- parser -- */

/** Binding power of the two binary logical operators; higher binds tighter. */
const PRECEDENCE = new Map<string, number>([
  ['or', 1],
  ['and', 2],
]);

class Parser {
  private readonly tokens: readonly Token[];
  private at = 0;
  /** Leaf clauses built so far, against `MAX_QUERY_TERMS`. */
  private terms = 0;
  /** Groups and prefixes currently open, against `MAX_QUERY_DEPTH`. */
  private depth = 0;

  constructor(tokens: readonly Token[]) {
    this.tokens = tokens;
  }

  /** Charge one leaf clause, failing on the token that went over. */
  private countTerm(token: Token): void {
    this.terms += 1;
    if (this.terms > MAX_QUERY_TERMS) {
      fail(
        `this query has more than ${MAX_QUERY_TERMS} clauses; narrow it, or use {in} with a list instead of a chain of ||`,
        token.offset,
        token.length,
      );
    }
  }

  /** Open a nesting level, failing on the token that went over. */
  private descend(token: Token): void {
    this.depth += 1;
    if (this.depth > MAX_QUERY_DEPTH) {
      fail(
        `this query nests more than ${MAX_QUERY_DEPTH} levels deep`,
        token.offset,
        token.length,
      );
    }
  }

  private peek(): Token {
    return this.tokens[this.at];
  }

  private next(): Token {
    const token = this.tokens[this.at];
    if (token.kind !== 'end') this.at += 1;
    return token;
  }

  /** Parse a whole query and insist the input is used up. */
  parse(): QueryNode {
    if (this.peek().kind === 'end') return { kind: 'all' };
    const node = this.parseExpression(1);
    const trailing = this.peek();
    if (trailing.kind !== 'end') {
      fail(
        `expected && or || but found ${describe(trailing)}`,
        trailing.offset,
        trailing.length,
      );
    }
    return node;
  }

  /**
   * Precedence climbing. `minPrecedence` is the loosest operator this call is
   * allowed to consume; recursing one level tighter on the right-hand side is
   * what makes both operators left-associative.
   */
  private parseExpression(minPrecedence: number): QueryNode {
    let left = this.parseUnary();
    for (;;) {
      const token = this.peek();
      if (token.kind !== 'operator' || token.operator === null) break;
      const precedence = PRECEDENCE.get(token.operator.id);
      if (precedence === undefined || precedence < minPrecedence) break;
      const spec = token.operator;
      this.next();
      const right = this.parseExpression(precedence + 1);
      left =
        spec.id === 'and'
          ? { kind: 'and', left, right }
          : { kind: 'or', left, right };
    }
    return left;
  }

  private parseUnary(): QueryNode {
    const token = this.peek();
    if (token.kind === 'operator' && token.operator !== null && token.operator.shape === 'prefix') {
      this.next();
      this.descend(token);
      const operand = this.parseUnary();
      this.depth -= 1;
      return { kind: 'not', operand };
    }
    return this.parsePrimary();
  }

  private parsePrimary(): QueryNode {
    const token = this.peek();

    if (token.kind === 'group-open') {
      this.next();
      if (this.peek().kind === 'group-close') {
        const close = this.peek();
        fail('expected a term inside {{ }}', close.offset, close.length);
      }
      this.descend(token);
      const inner = this.parseExpression(1);
      this.depth -= 1;
      const close = this.peek();
      if (close.kind !== 'group-close') {
        fail(`expected }} to close the group but found ${describe(close)}`, close.offset, close.length);
      }
      this.next();
      return inner;
    }

    // Everything below this point builds exactly one leaf of the tree.
    this.countTerm(token);

    if (token.kind === 'operator' && token.operator !== null) {
      const spec = token.operator;
      if (spec.shape === 'value') {
        this.next();
        return this.parseValueOperator(spec, token.offset);
      }
      if (spec.shape === 'attribute') {
        this.next();
        return this.parseAttributeOperator(spec, token.offset);
      }
      fail(
        `expected a term but found ${describe(token)}`,
        token.offset,
        token.length,
      );
    }

    if (token.kind !== 'word') {
      fail(`expected a term but found ${describe(token)}`, token.offset, token.length);
    }

    const atom = this.next();
    const operatorToken = this.peek();
    if (operatorToken.kind !== 'operator' || operatorToken.operator === null) {
      fail(
        `expected an operator after ${describe(atom)}; write something like ${atom.value} ~= <value>`,
        operatorToken.offset,
        operatorToken.length,
      );
    }

    const spec = operatorToken.operator;

    if (spec.shape === 'pathway') {
      this.next();
      const target = this.expectWord(spec);
      return { kind: 'pathways', source: atom.value, target: target.value, offset: atom.offset };
    }

    if (spec.shape !== 'field') {
      fail(
        `${spec.bracket} (${spec.symbol}) does not take a field on its left; write it before the value`,
        operatorToken.offset,
        operatorToken.length,
      );
    }

    if (atom.quoted) {
      fail(
        `expected a field name before ${spec.bracket} (${spec.symbol}), but ${describe(atom)} is quoted text`,
        atom.offset,
        atom.length,
      );
    }

    const field = findField(atom.value);
    if (field === null) {
      const suggestion = nearestField(atom.value);
      fail(
        suggestion === null
          ? `unknown field "${atom.value}"`
          : `unknown field "${atom.value}"; did you mean ${suggestion}?`,
        atom.offset,
        atom.length,
      );
    }

    if (!spec.fieldTypes.includes(field.type)) {
      fail(
        `${spec.bracket} (${spec.symbol}) needs ${typeNames(spec.fieldTypes)} field, but ${field.name} is ${field.type}`,
        operatorToken.offset,
        operatorToken.length,
      );
    }

    this.next();
    return this.parseFieldComparison(field, spec, atom.offset);
  }

  private parseValueOperator(spec: OperatorSpec, offset: number): QueryNode {
    const value = this.expectWord(spec);
    const connectivity = connectivityOperatorOf(spec);
    if (connectivity !== null) {
      return { kind: 'connectivity', op: connectivity, value: value.value, offset };
    }
    const similarity = similarityOperatorOf(spec);
    if (similarity !== null) {
      return { kind: 'similarity', op: similarity, value: value.value, offset };
    }
    // No other operator declares the `value` shape.
    return fail(`${spec.bracket} cannot be used here`, offset, spec.symbol.length);
  }

  private parseAttributeOperator(spec: OperatorSpec, offset: number): QueryNode {
    const token = this.peek();
    if (token.kind !== 'word' || token.quoted) {
      fail(
        `expected a field name after ${spec.bracket} (${spec.symbol}) but found ${describe(token)}`,
        token.offset,
        token.length,
      );
    }
    this.next();
    const field = findField(token.value);
    if (field === null) {
      const suggestion = nearestField(token.value);
      fail(
        suggestion === null
          ? `unknown field "${token.value}"`
          : `unknown field "${token.value}"; did you mean ${suggestion}?`,
        token.offset,
        token.length,
      );
    }
    return {
      kind: 'presence',
      op: spec.id === 'has' ? 'has' : 'missing',
      field: field.name,
      offset,
    };
  }

  private parseFieldComparison(
    field: QueryField,
    spec: OperatorSpec,
    offset: number,
  ): QueryNode {
    const token = this.expectWord(spec);

    if (spec.valueForm === 'range') {
      const parts = token.value.split('..');
      if (parts.length !== 2) {
        fail(
          `expected a range written low..high after ${spec.bracket} (${spec.symbol}) but found ${describe(token)}`,
          token.offset,
          token.length,
        );
      }
      const low = toNumber(parts[0]);
      const high = toNumber(parts[1]);
      if (low === null || high === null) {
        fail(
          `expected two numbers in the range low..high but found ${describe(token)}`,
          token.offset,
          token.length,
        );
      }
      // Accepting a reversed range silently would make `<> 20..5` mean "nothing"
      // when the user plainly meant the same window.
      const lo = Math.min(low, high);
      const hi = Math.max(low, high);
      return { kind: 'numeric-compare', op: 'between', field: field.name, values: [lo, hi], offset };
    }

    if (spec.valueForm === 'list') {
      // A list is one token when it is written tightly, but `lif, adex` reads
      // naturally and lexes as two. Adjacent words are pulled back together
      // whenever a comma sits on the seam, so the spacing a user would write by
      // habit is not a syntax error.
      let text = token.value;
      for (;;) {
        const next = this.peek();
        if (next.kind !== 'word') break;
        if (!text.endsWith(',') && !next.value.startsWith(',')) break;
        text += next.value;
        this.next();
      }
      const raw = text.split(',').map((part) => part.trim());
      if (raw.some((part) => part === '')) {
        fail(
          `expected a comma-separated list after ${spec.bracket} (${spec.symbol}) with no empty items`,
          token.offset,
          token.length,
        );
      }
      const op = spec.id === 'in' ? 'in' : 'not_in';
      if (field.type === 'numeric') {
        const values: number[] = [];
        for (const part of raw) {
          const value = toNumber(part);
          if (value === null) {
            fail(
              `expected numbers in the list after ${spec.bracket} (${spec.symbol}) but found "${part}"`,
              token.offset,
              token.length,
            );
          }
          values.push(value);
        }
        return { kind: 'numeric-compare', op, field: field.name, values, offset };
      }
      return { kind: 'string-compare', op, field: field.name, values: raw, offset };
    }

    if (field.type === 'numeric') {
      const value = toNumber(token.value);
      if (value === null) {
        fail(
          `expected a number after ${spec.bracket} (${spec.symbol}) but found ${describe(token)}`,
          token.offset,
          token.length,
        );
      }
      const op = numericOperatorOf(spec);
      if (op === null) {
        fail(
          `${spec.bracket} (${spec.symbol}) cannot compare the numeric field ${field.name}`,
          token.offset,
          token.length,
        );
      }
      return { kind: 'numeric-compare', op, field: field.name, values: [value], offset };
    }

    if (field.type === 'boolean') {
      const value = toBoolean(token.value);
      if (value === null) {
        fail(
          `expected true or false after ${spec.bracket} (${spec.symbol}) but found ${describe(token)}`,
          token.offset,
          token.length,
        );
      }
      return {
        kind: 'boolean-compare',
        op: spec.id === 'equal' ? 'equal' : 'not_equal',
        field: field.name,
        value,
        offset,
      };
    }

    const op = stringOperatorOf(spec);
    if (op === null) {
      fail(
        `${spec.bracket} (${spec.symbol}) cannot compare the text field ${field.name}`,
        token.offset,
        token.length,
      );
    }
    return { kind: 'string-compare', op, field: field.name, values: [token.value], offset };
  }

  private expectWord(spec: OperatorSpec): Token {
    const token = this.peek();
    if (token.kind !== 'word') {
      fail(
        `expected a value after ${spec.bracket} (${spec.symbol}) but found ${describe(token)}`,
        token.offset,
        token.length,
      );
    }
    if (token.value === '') {
      fail(
        `expected a value after ${spec.bracket} (${spec.symbol}) but the value is empty`,
        token.offset,
        token.length,
      );
    }
    this.next();
    return token;
  }
}

const STRING_OPERATORS: ReadonlySet<string> = new Set<string>([
  'equal',
  'not_equal',
  'like',
  'starts_with',
  'ends_with',
  'contains',
  'not_contains',
  'in',
  'not_in',
]);

const NUMERIC_OPERATORS: ReadonlySet<string> = new Set<string>([
  'equal',
  'not_equal',
  'gt',
  'gte',
  'lt',
  'lte',
  'between',
  'in',
  'not_in',
]);

const CONNECTIVITY_OPERATORS: ReadonlySet<string> = new Set<string>([
  'upstream',
  'downstream',
  'upstream_all',
  'downstream_all',
  'reciprocal',
]);

const SIMILARITY_OPERATORS: ReadonlySet<string> = new Set<string>([
  'similar_connectivity',
  'similar_connectivity_upstream',
  'similar_connectivity_downstream',
  'similar_shape',
]);

function stringOperatorOf(spec: OperatorSpec): StringOperator | null {
  return STRING_OPERATORS.has(spec.id) ? (spec.id as StringOperator) : null;
}

function numericOperatorOf(spec: OperatorSpec): NumericOperator | null {
  return NUMERIC_OPERATORS.has(spec.id) ? (spec.id as NumericOperator) : null;
}

function connectivityOperatorOf(spec: OperatorSpec): ConnectivityOperator | null {
  return CONNECTIVITY_OPERATORS.has(spec.id) ? (spec.id as ConnectivityOperator) : null;
}

function similarityOperatorOf(spec: OperatorSpec): SimilarityOperator | null {
  return SIMILARITY_OPERATORS.has(spec.id) ? (spec.id as SimilarityOperator) : null;
}

/**
 * Parse a query into an AST.
 *
 * An empty or whitespace-only source is not an error and not an empty result:
 * it produces the `all` node, so a query box that has not been typed into yet
 * shows the whole network rather than nothing.
 */
export function parseQuery(source: string): ParseResult {
  const lexed = tokenize(source);
  if (!lexed.ok) return { ok: false, error: lexed.error };
  try {
    return { ok: true, ast: new Parser(lexed.tokens).parse() };
  } catch (error) {
    if (error instanceof ParseFailure) return { ok: false, error: error.error };
    throw error;
  }
}
