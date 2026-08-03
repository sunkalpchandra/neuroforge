/**
 * Tokeniser and operator vocabulary for the cell query language.
 *
 * The grammar is FlyWire Codex's, which is the reason a connectome with a
 * hundred thousand cells is browsable at all: structured predicates over cell
 * attributes, connectivity and morphological similarity, combined with explicit
 * boolean operators. Every operator has two spellings — a bracket form that is
 * unambiguous and self-documenting (`{starts_with}`) and a symbol form that is
 * fast to type (`^*`) — and they parse to exactly the same node.
 *
 * Two lexing decisions carry the whole design:
 *
 * Symbols are matched longest-first against a table sorted by length, so `!=`
 * can never be read as `!` followed by `=`, and `~UA` can never be read as `~u`
 * followed by `A`. There is no backtracking anywhere in the lexer as a result.
 *
 * The nine characters that begin an operator — `= ! ~ ^ > < $ & |` — are
 * reserved, and a bare word stops at any of them. That is what lets
 * `rate>=5` lex without spaces. It also means a value containing one of those
 * characters must be quoted, and a stray one is an error naming its offset
 * rather than a word that silently means something else.
 */

import type { FieldType } from './fields';

export type OperatorId =
  | 'equal'
  | 'not_equal'
  | 'like'
  | 'starts_with'
  | 'ends_with'
  | 'contains'
  | 'not_contains'
  | 'in'
  | 'not_in'
  | 'gt'
  | 'gte'
  | 'lt'
  | 'lte'
  | 'between'
  | 'upstream'
  | 'downstream'
  | 'upstream_all'
  | 'downstream_all'
  | 'reciprocal'
  | 'pathways'
  | 'similar_connectivity'
  | 'similar_connectivity_upstream'
  | 'similar_connectivity_downstream'
  | 'similar_shape'
  | 'has'
  | 'missing'
  | 'and'
  | 'or'
  | 'not';

/**
 * How an operator sits in the source text.
 *
 * `field`     — `field OP value`, the ordinary attribute predicate.
 * `pathway`   — `value OP value`, the only binary operator over two cells.
 * `value`     — `OP value`, the connectivity and similarity predicates.
 * `attribute` — `OP field`, presence tests.
 * `logical`   — `expr OP expr`.
 * `prefix`    — `OP expr`.
 */
export type OperatorShape = 'field' | 'pathway' | 'value' | 'attribute' | 'logical' | 'prefix';

/** Shape of the right-hand side, independent of the field's type. */
export type ValueForm = 'none' | 'single' | 'list' | 'range';

/** Grouping in the help table, so the UI can section the operator list. */
export type OperatorGroup = 'text' | 'numeric' | 'connectivity' | 'similarity' | 'presence' | 'logic';

export interface OperatorSpec {
  id: OperatorId;
  /** Bracket spelling, e.g. `{starts_with}`. */
  bracket: string;
  /** Symbol spelling, e.g. `^*`. */
  symbol: string;
  shape: OperatorShape;
  valueForm: ValueForm;
  /** Field types this operator accepts; empty when it takes no field. */
  fieldTypes: readonly FieldType[];
  group: OperatorGroup;
  /** Short human name for the help table. */
  label: string;
  description: string;
  /** A worked example, shown beside the operator in the help panel. */
  example: string;
}

const STRING_ONLY: readonly FieldType[] = ['string'];
const NUMERIC_ONLY: readonly FieldType[] = ['numeric'];
const COMPARABLE: readonly FieldType[] = ['string', 'numeric'];
const ANY_FIELD: readonly FieldType[] = ['string', 'numeric', 'boolean'];
const NO_FIELD: readonly FieldType[] = [];

/**
 * Every operator, both spellings, in help-panel order.
 *
 * The bracket form is always `{` + the operator id + `}`, which is what keeps
 * this table and the `OperatorId` union impossible to drift apart.
 */
export const QUERY_OPERATORS: readonly OperatorSpec[] = [
  {
    id: 'equal',
    bracket: '{equal}',
    symbol: '==',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: ANY_FIELD,
    group: 'text',
    label: 'Equals',
    description: 'Exact match. Text comparison ignores case.',
    example: 'polarity == inhibitory',
  },
  {
    id: 'not_equal',
    bracket: '{not_equal}',
    symbol: '!=',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: ANY_FIELD,
    group: 'text',
    label: 'Not equal',
    description: 'Matches when no value of the field equals this one.',
    example: 'model != lif',
  },
  {
    id: 'like',
    bracket: '{like}',
    symbol: '~=',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: STRING_ONLY,
    group: 'text',
    label: 'Like',
    description:
      'Loose match: ignores case, punctuation, spacing, the order of the parts and leading zeros. "Pm2", "pm-2" and "PM 02" are the same name.',
    example: 'label ~= pm2',
  },
  {
    id: 'starts_with',
    bracket: '{starts_with}',
    symbol: '^*',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: STRING_ONLY,
    group: 'text',
    label: 'Starts with',
    description: 'Prefix match, ignoring case.',
    example: 'label ^* LC',
  },
  {
    id: 'ends_with',
    bracket: '{ends_with}',
    symbol: '^$',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: STRING_ONLY,
    group: 'text',
    label: 'Ends with',
    description: 'Suffix match, ignoring case.',
    example: 'label ^$ _R',
  },
  {
    id: 'contains',
    bracket: '{contains}',
    symbol: '>>',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: STRING_ONLY,
    group: 'text',
    label: 'Contains',
    description: 'Substring match, ignoring case.',
    example: 'population >> layer',
  },
  {
    id: 'not_contains',
    bracket: '{not_contains}',
    symbol: '!>',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: STRING_ONLY,
    group: 'text',
    label: 'Does not contain',
    description: 'Matches when no value of the field contains this substring.',
    example: 'label !> test',
  },
  {
    id: 'in',
    bracket: '{in}',
    symbol: '<<',
    shape: 'field',
    valueForm: 'list',
    fieldTypes: COMPARABLE,
    group: 'text',
    label: 'In',
    description: 'Matches any item of a comma-separated list.',
    example: 'model << lif,adex',
  },
  {
    id: 'not_in',
    bracket: '{not_in}',
    symbol: '!<',
    shape: 'field',
    valueForm: 'list',
    fieldTypes: COMPARABLE,
    group: 'text',
    label: 'Not in',
    description: 'Matches when no value of the field appears in the list.',
    example: 'archetype !< basket,stellate',
  },
  {
    id: 'gt',
    bracket: '{gt}',
    symbol: '>',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: NUMERIC_ONLY,
    group: 'numeric',
    label: 'Greater than',
    description: 'Strictly greater.',
    example: 'rate > 12',
  },
  {
    id: 'gte',
    bracket: '{gte}',
    symbol: '>=',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: NUMERIC_ONLY,
    group: 'numeric',
    label: 'At least',
    description: 'Greater than or equal.',
    example: 'degree >= 8',
  },
  {
    id: 'lt',
    bracket: '{lt}',
    symbol: '<',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: NUMERIC_ONLY,
    group: 'numeric',
    label: 'Less than',
    description: 'Strictly less.',
    example: 'voltage < -60',
  },
  {
    id: 'lte',
    bracket: '{lte}',
    symbol: '<=',
    shape: 'field',
    valueForm: 'single',
    fieldTypes: NUMERIC_ONLY,
    group: 'numeric',
    label: 'At most',
    description: 'Less than or equal.',
    example: 'out_degree <= 2',
  },
  {
    id: 'between',
    bracket: '{between}',
    symbol: '<>',
    shape: 'field',
    valueForm: 'range',
    fieldTypes: NUMERIC_ONLY,
    group: 'numeric',
    label: 'Between',
    description: 'Inclusive range, written low..high.',
    example: 'rate <> 5..20',
  },
  {
    id: 'upstream',
    bracket: '{upstream}',
    symbol: '^^',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'connectivity',
    label: 'Upstream of',
    description: 'Cells presynaptic to any cell matching this label or id.',
    example: '^^ "Pm 2"',
  },
  {
    id: 'downstream',
    bracket: '{downstream}',
    symbol: '!^',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'connectivity',
    label: 'Downstream of',
    description: 'Cells postsynaptic to any cell matching this label or id.',
    example: '!^ "Pm 2"',
  },
  {
    id: 'upstream_all',
    bracket: '{upstream_all}',
    symbol: '~UA',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'connectivity',
    label: 'Upstream of all',
    description: 'Cells presynaptic to every cell matching this label or id.',
    example: '~UA LC10',
  },
  {
    id: 'downstream_all',
    bracket: '{downstream_all}',
    symbol: '~DA',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'connectivity',
    label: 'Downstream of all',
    description: 'Cells postsynaptic to every cell matching this label or id.',
    example: '~DA LC10',
  },
  {
    id: 'reciprocal',
    bracket: '{reciprocal}',
    symbol: '^v',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'connectivity',
    label: 'Reciprocal with',
    description: 'Cells wired to a match in both directions at once.',
    example: '^v LC10',
  },
  {
    id: 'pathways',
    bracket: '{pathways}',
    symbol: '=>',
    shape: 'pathway',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'connectivity',
    label: 'Pathways',
    description:
      'Every cell lying on a shortest route from the left cell to the right one, both ends included.',
    example: '"Pm 2" => LC10',
  },
  {
    id: 'similar_connectivity',
    bracket: '{similar_connectivity}',
    symbol: '~c',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'similarity',
    label: 'Similar wiring',
    description: 'Cells whose full connectivity fingerprint points the same way as this one.',
    example: '~c LC10',
  },
  {
    id: 'similar_connectivity_upstream',
    bracket: '{similar_connectivity_upstream}',
    symbol: '~u',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'similarity',
    label: 'Similar inputs',
    description: 'Cells that listen to the same places as this one.',
    example: '~u LC10',
  },
  {
    id: 'similar_connectivity_downstream',
    bracket: '{similar_connectivity_downstream}',
    symbol: '~d',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'similarity',
    label: 'Similar outputs',
    description: 'Cells that project to the same places as this one.',
    example: '~d LC10',
  },
  {
    id: 'similar_shape',
    bracket: '{similar_shape}',
    symbol: '~~',
    shape: 'value',
    valueForm: 'single',
    fieldTypes: NO_FIELD,
    group: 'similarity',
    label: 'Similar shape',
    description:
      'Cells with a comparable morphology: same archetype, and close on soma radius, dendrite count, depth, length and spread, and axon length.',
    example: '~~ LC10',
  },
  {
    id: 'has',
    bracket: '{has}',
    symbol: '$$',
    shape: 'attribute',
    valueForm: 'none',
    fieldTypes: ANY_FIELD,
    group: 'presence',
    label: 'Has',
    description: 'Cells that carry a value for this field.',
    example: '$$ population',
  },
  {
    id: 'missing',
    bracket: '{missing}',
    symbol: '!$',
    shape: 'attribute',
    valueForm: 'none',
    fieldTypes: ANY_FIELD,
    group: 'presence',
    label: 'Missing',
    description: 'Cells with no value for this field.',
    example: '!$ label',
  },
  {
    id: 'and',
    bracket: '{and}',
    symbol: '&&',
    shape: 'logical',
    valueForm: 'none',
    fieldTypes: NO_FIELD,
    group: 'logic',
    label: 'And',
    description: 'Both sides must match. Binds tighter than or.',
    example: 'rate > 5 && polarity == inhibitory',
  },
  {
    id: 'or',
    bracket: '{or}',
    symbol: '||',
    shape: 'logical',
    valueForm: 'none',
    fieldTypes: NO_FIELD,
    group: 'logic',
    label: 'Or',
    description: 'Either side may match. Binds loosest.',
    example: 'model == lif || model == adex',
  },
  {
    id: 'not',
    bracket: '{not}',
    symbol: '!!',
    shape: 'prefix',
    valueForm: 'none',
    fieldTypes: NO_FIELD,
    group: 'logic',
    label: 'Not',
    description: 'Inverts the term that follows. Binds tighter than and.',
    example: '!! $$ population',
  },
];

const BY_ID = new Map<OperatorId, OperatorSpec>(QUERY_OPERATORS.map((spec) => [spec.id, spec]));

/** Bracket name (without the braces) to spec. */
const BY_BRACKET_NAME = new Map<string, OperatorSpec>(
  QUERY_OPERATORS.map((spec) => [spec.bracket.slice(1, -1), spec]),
);

/**
 * Symbols sorted longest first. Matching walks this list and takes the first
 * hit, which is what makes `!=` unreachable as `!` and `~UA` unreachable as
 * `~u`. Matching is case-sensitive: `~DA` and `~d` are different operators.
 */
const SYMBOLS_BY_LENGTH: readonly OperatorSpec[] = [...QUERY_OPERATORS].sort(
  (a, b) => b.symbol.length - a.symbol.length,
);

/** Characters that may begin a symbol operator, and so may not appear in a bare word. */
export const RESERVED_CHARS = '=!~^><$&|';

const RESERVED = new Set<string>(RESERVED_CHARS.split(''));

export function operatorById(id: OperatorId): OperatorSpec {
  const spec = BY_ID.get(id);
  // Every id in the union has a row; this is the compiler-shaped fallback.
  if (spec === undefined) throw new Error(`unknown operator id: ${id}`);
  return spec;
}

/* ------------------------------------------------------------------ token -- */

export type TokenKind = 'word' | 'operator' | 'group-open' | 'group-close' | 'end';

export interface Token {
  kind: TokenKind;
  /** Exact source text, quotes and braces included. */
  text: string;
  /** Decoded text of a word: quotes stripped, escapes resolved. */
  value: string;
  /** The operator this token is, for `operator` tokens. */
  operator: OperatorSpec | null;
  /** True when a word came from a quoted literal, so it is never a field name. */
  quoted: boolean;
  offset: number;
  length: number;
}

export interface LexError {
  message: string;
  offset: number;
  length: number;
}

export type LexResult = { ok: true; tokens: readonly Token[] } | { ok: false; error: LexError };

const WHITESPACE = /\s/;

function isWhitespace(char: string): boolean {
  return WHITESPACE.test(char);
}

/** True for a character that may sit inside an unquoted value or field name. */
function isWordChar(char: string): boolean {
  if (isWhitespace(char)) return false;
  if (char === '{' || char === '}' || char === '"' || char === "'") return false;
  return !RESERVED.has(char);
}

function token(
  kind: TokenKind,
  text: string,
  value: string,
  operator: OperatorSpec | null,
  quoted: boolean,
  offset: number,
): Token {
  return { kind, text, value, operator, quoted, offset, length: text.length };
}

/**
 * Split a query into tokens.
 *
 * Returns an error rather than throwing, because this runs on every keystroke of
 * a query box and a half-typed query is the normal case, not an exception.
 */
export function tokenize(source: string): LexResult {
  const tokens: Token[] = [];
  const n = source.length;
  let at = 0;

  while (at < n) {
    const char = source[at];

    if (isWhitespace(char)) {
      at += 1;
      continue;
    }

    if (source.startsWith('{{', at)) {
      tokens.push(token('group-open', '{{', '{{', null, false, at));
      at += 2;
      continue;
    }

    if (source.startsWith('}}', at)) {
      tokens.push(token('group-close', '}}', '}}', null, false, at));
      at += 2;
      continue;
    }

    if (char === '}') {
      return {
        ok: false,
        error: {
          message: 'unmatched }; a group closes with }}',
          offset: at,
          length: 1,
        },
      };
    }

    if (char === '{') {
      const close = source.indexOf('}', at + 1);
      if (close === -1) {
        return {
          ok: false,
          error: {
            message: 'unterminated bracket operator; expected a closing }',
            offset: at,
            length: n - at,
          },
        };
      }
      const name = source.slice(at + 1, close);
      const spec = BY_BRACKET_NAME.get(name);
      if (spec === undefined) {
        return {
          ok: false,
          error: {
            message: `unknown operator {${name}}`,
            offset: at,
            length: close + 1 - at,
          },
        };
      }
      const text = source.slice(at, close + 1);
      tokens.push(token('operator', text, spec.id, spec, false, at));
      at = close + 1;
      continue;
    }

    if (char === '"' || char === "'") {
      const quote = char;
      let cursor = at + 1;
      let value = '';
      let closed = false;
      while (cursor < n) {
        const c = source[cursor];
        if (c === '\\' && cursor + 1 < n) {
          value += source[cursor + 1];
          cursor += 2;
          continue;
        }
        if (c === quote) {
          closed = true;
          cursor += 1;
          break;
        }
        value += c;
        cursor += 1;
      }
      if (!closed) {
        return {
          ok: false,
          error: {
            message: `unterminated quoted value; expected a closing ${quote}`,
            offset: at,
            length: n - at,
          },
        };
      }
      tokens.push(token('word', source.slice(at, cursor), value, null, true, at));
      at = cursor;
      continue;
    }

    if (RESERVED.has(char)) {
      let matched: OperatorSpec | null = null;
      for (const spec of SYMBOLS_BY_LENGTH) {
        if (source.startsWith(spec.symbol, at)) {
          matched = spec;
          break;
        }
      }
      if (matched === null) {
        return {
          ok: false,
          error: {
            message: `unexpected character ${char}; quote the value if it is part of a name`,
            offset: at,
            length: 1,
          },
        };
      }
      tokens.push(token('operator', matched.symbol, matched.id, matched, false, at));
      at += matched.symbol.length;
      continue;
    }

    let cursor = at;
    while (cursor < n && isWordChar(source[cursor])) cursor += 1;
    const text = source.slice(at, cursor);
    tokens.push(token('word', text, text, null, false, at));
    at = cursor;
  }

  tokens.push(token('end', '', '', null, false, n));
  return { ok: true, tokens };
}
