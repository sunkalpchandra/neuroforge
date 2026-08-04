/**
 * The cell query language.
 *
 * A connectome is only browsable if you can ask it questions, and the questions
 * worth asking are structured: not "find the text pm2" but "find the inhibitory
 * cells upstream of Pm2 that fire above 10 Hz and are wired like LC10". This
 * module is that language — FlyWire Codex's grammar, extended with the numeric
 * comparisons this platform can answer because it has a simulation running
 * underneath the wiring.
 *
 *   parseQuery(source)                     source text -> typed AST, or an error
 *   evaluateQuery(ast, buffers, circuit)   AST -> a mask over neuron slots
 *   runQuery(source, buffers, circuit)     both, in one call
 *
 * `QUERY_FIELDS` and `QUERY_OPERATORS` are the same tables the parser type-checks
 * against, so a help panel or an autocomplete list built from them can never
 * offer something the parser rejects.
 *
 * Both operator spellings parse identically: `label {starts_with} LC` and
 * `label ^* LC` are the same query.
 */

import type { Circuit, SimulationBuffers } from '@neuroforge/shared';

import type { QueryError, QueryNode, ParseResult } from './parser';
import { parseQuery } from './parser';
import type { QueryResult } from './evaluate';
import { evaluateQuery } from './evaluate';

export type {
  FieldSource,
  FieldType,
  BooleanField,
  NumericField,
  QueryField,
  StringField,
} from './fields';
export {
  MAX_FIELD_VALUES,
  QUERY_FIELDS,
  QUERY_FIELD_NAMES,
  fieldPresent,
  findField,
} from './fields';

export type {
  LexError,
  LexResult,
  OperatorGroup,
  OperatorId,
  OperatorShape,
  OperatorSpec,
  Token,
  TokenKind,
  ValueForm,
} from './lexer';
export { QUERY_OPERATORS, RESERVED_CHARS, operatorById, tokenize } from './lexer';

export type {
  AllNode,
  AndNode,
  BooleanCompareNode,
  ConnectivityNode,
  ConnectivityOperator,
  NotNode,
  NumericCompareNode,
  NumericOperator,
  OrNode,
  ParseResult,
  PathwaysNode,
  PresenceNode,
  QueryError,
  QueryNode,
  SimilarityNode,
  SimilarityOperator,
  StringCompareNode,
  StringOperator,
} from './parser';
export { MAX_QUERY_DEPTH, MAX_QUERY_TERMS, parseQuery } from './parser';

export type { QueryResult } from './evaluate';
export {
  MAX_SIMILARITY_SEEDS,
  SIMILAR_CONNECTIVITY_THRESHOLD,
  SIMILAR_SHAPE_DISTANCE,
  evaluateQuery,
  likeKey,
  matchedSlots,
} from './evaluate';

export { verifyQuery } from './__verify';

/** Outcome of `runQuery`: either a parsed-and-evaluated answer, or where it broke. */
export type QueryRun =
  | { ok: true; ast: QueryNode; result: QueryResult }
  | { ok: false; error: QueryError };

/**
 * Parse and evaluate in one step.
 *
 * The parse result is returned alongside the mask because a panel that wants to
 * explain a query — highlight its clauses, offer a "select all matches" button
 * keyed on what was asked — needs the tree as well as the answer.
 */
export function runQuery(
  source: string,
  buffers: SimulationBuffers,
  circuit: Circuit,
): QueryRun {
  const parsed: ParseResult = parseQuery(source);
  if (!parsed.ok) return { ok: false, error: parsed.error };
  return { ok: true, ast: parsed.ast, result: evaluateQuery(parsed.ast, buffers, circuit) };
}
