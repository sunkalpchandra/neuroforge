/**
 * Prompt tokenisation and coverage tracking for the offline planner.
 *
 * Every handler marks the tokens it consumed. Whatever is left at the end that is
 * not a function word becomes a warning, which is what stops the planner from
 * silently ignoring half a sentence.
 */

export interface Token {
  readonly text: string;
  readonly start: number;
  readonly end: number;
}

/**
 * Multi-word domain terms rewritten to their hyphenated single-token form. Each
 * replacement preserves length, so character offsets stay valid.
 */
const COMPOUND_TERMS: readonly string[][] = [
  ['fast', 'spiking'],
  ['regular', 'spiking'],
  ['low', 'threshold'],
  ['intrinsically', 'bursting'],
  ['hodgkin', 'huxley'],
  ['morris', 'lecar'],
  ['integrate', 'and', 'fire'],
  ['adaptive', 'exponential'],
  ['leaky', 'integrator'],
  ['all', 'to', 'all'],
  ['one', 'to', 'one'],
  ['fully', 'connected'],
  ['gap', 'junction'],
  ['pulse', 'train'],
  ['medium', 'spiny'],
  ['thalamo', 'cortical'],
  ['gaba', 'a'],
  ['gaba', 'b'],
  ['dentate', 'gyrus'],
  ['start', 'over'],
  ['from', 'scratch'],
  ['speed', 'up'],
  ['slow', 'down'],
  ['sharp', 'wave'],
];

const COMPOUND_RULES: readonly { pattern: RegExp; replacement: string }[] = COMPOUND_TERMS.map(
  (words) => ({
    pattern: new RegExp(`\\b${words.join('[ -]')}\\b`, 'g'),
    replacement: words.join('-'),
  }),
);

const TOKEN_RE = /\d+(?:\.\d+)?%|\d+(?:\.\d+)?[a-z]*|[a-z]+(?:[-'][a-z]+)*\d*/g;

/** Lowercase, fold compound terms and strip digit-group separators. */
export function normalisePrompt(raw: string): string {
  let text = raw
    .toLowerCase()
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[–—]/g, ' ')
    .replace(/(\d),(?=\d{3}\b)/g, '');
  for (const rule of COMPOUND_RULES) text = text.replace(rule.pattern, rule.replacement);
  return text;
}

const NUMBER_WORDS: Record<string, number> = {
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  eleven: 11,
  twelve: 12,
  thirteen: 13,
  fourteen: 14,
  fifteen: 15,
  sixteen: 16,
  seventeen: 17,
  eighteen: 18,
  nineteen: 19,
  twenty: 20,
  thirty: 30,
  forty: 40,
  fifty: 50,
  sixty: 60,
  seventy: 70,
  eighty: 80,
  ninety: 90,
};

const MULTIPLIER_WORDS: Record<string, number> = {
  hundred: 100,
  thousand: 1000,
  dozen: 12,
};

/** A bare numeric literal, optionally with a `k` magnitude suffix. */
export function parseNumericToken(token: string): number | null {
  const match = /^(\d+(?:\.\d+)?)(k)?$/.exec(token);
  if (!match) return null;
  const value = Number.parseFloat(match[1]);
  if (!Number.isFinite(value)) return null;
  return match[2] === 'k' ? value * 1000 : value;
}

/** Any word or literal that could be part of a count. */
export function isCountToken(token: string): boolean {
  return (
    parseNumericToken(token) !== null ||
    NUMBER_WORDS[token] !== undefined ||
    MULTIPLIER_WORDS[token] !== undefined
  );
}

/**
 * Read a count out of a token run: digits, `1.5k`, number words and the
 * multipliers that follow them ("two hundred", "a thousand").
 */
export function parseCount(tokens: readonly string[]): number | null {
  let current = 0;
  let seen = false;
  for (const token of tokens) {
    const literal = parseNumericToken(token);
    if (literal !== null) {
      current = current === 0 ? literal : current + literal;
      seen = true;
      continue;
    }
    const word = NUMBER_WORDS[token];
    if (word !== undefined) {
      current += word;
      seen = true;
      continue;
    }
    const multiplier = MULTIPLIER_WORDS[token];
    if (multiplier !== undefined) {
      current = (current === 0 ? 1 : current) * multiplier;
      seen = true;
    }
  }
  return seen ? Math.round(current) : null;
}

/** Words that carry no instruction and so never justify a "not understood" warning. */
const FILLER_WORDS: ReadonlySet<string> = new Set([
  'a', 'about', 'above', 'add', 'adding', 'after', 'again', 'all', 'also', 'an', 'and', 'any',
  'apply', 'are', 'around', 'as', 'at', 'back', 'be', 'been', 'before', 'between', 'both', 'build',
  'building', 'built', 'but', 'by', 'can', 'change', 'circuit', 'circuits', 'configure', 'connect',
  'connected', 'connecting', 'connection', 'connections', 'construct', 'could', 'create', 'created',
  'creating', 'current', 'currently', 'default', 'design', 'do', 'does', 'each', 'else', 'every',
  'everything', 'few', 'first', 'for', 'from', 'generate', 'get', 'give', 'go', 'group', 'groups',
  'has', 'have', 'here', 'how', 'i', 'if', 'in', 'instead', 'into', 'is', 'it', 'its', 'just',
  'keep', 'let', 'like', 'link', 'made', 'make', 'making', 'many', 'me', 'model', 'more', 'most',
  'much', 'must', 'my', 'need', 'needs', 'net', 'network', 'networks', 'new', 'no', 'not', 'now',
  'of', 'off', 'on', 'one', 'ones', 'only', 'onto', 'or', 'other', 'our', 'out', 'over', 'place',
  'please', 'plus', 'population', 'populations', 'put', 'rest', 'same', 'set', 'setup', 'should',
  'similar', 'simulation', 'so', 'some', 'something', 'still', 'such', 'sure', 'take', 'than',
  'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too',
  'try', 'two', 'up', 'us', 'use', 'used', 'using', 'very', 'want', 'was', 'way', 'we', 'well',
  'were', 'what', 'when', 'where', 'which', 'while', 'who', 'why', 'will', 'wire', 'wired',
  'wiring', 'with', 'within', 'without', 'work', 'would', 'you', 'your',
]);

/** A tokenised prompt that remembers which tokens a handler has already claimed. */
export class PromptScan {
  readonly text: string;
  readonly tokens: readonly Token[];
  private readonly claimed: boolean[];

  constructor(raw: string) {
    this.text = normalisePrompt(raw);
    const tokens: Token[] = [];
    TOKEN_RE.lastIndex = 0;
    for (let m = TOKEN_RE.exec(this.text); m !== null; m = TOKEN_RE.exec(this.text)) {
      tokens.push({ text: m[0], start: m.index, end: m.index + m[0].length });
    }
    this.tokens = tokens;
    this.claimed = new Array<boolean>(tokens.length).fill(false);
  }

  get length(): number {
    return this.tokens.length;
  }

  word(index: number): string {
    return index >= 0 && index < this.tokens.length ? this.tokens[index].text : '';
  }

  /** Token texts in `[from, to)`, clamped to the prompt. */
  slice(from: number, to: number): string[] {
    const lo = Math.max(0, from);
    const hi = Math.min(this.tokens.length, to);
    const out: string[] = [];
    for (let i = lo; i < hi; i += 1) out.push(this.tokens[i].text);
    return out;
  }

  claim(index: number): void {
    if (index >= 0 && index < this.claimed.length) this.claimed[index] = true;
  }

  claimRange(from: number, to: number): void {
    for (let i = Math.max(0, from); i < Math.min(this.claimed.length, to); i += 1) {
      this.claimed[i] = true;
    }
  }

  /** Claim every token that overlaps the character range `[start, end)`. */
  claimChars(start: number, end: number): void {
    for (let i = 0; i < this.tokens.length; i += 1) {
      const token = this.tokens[i];
      if (token.end > start && token.start < end) this.claimed[i] = true;
    }
  }

  isClaimed(index: number): boolean {
    return index >= 0 && index < this.claimed.length && this.claimed[index];
  }

  /** True when any token overlapping the character range has already been claimed. */
  isRangeClaimed(start: number, end: number): boolean {
    for (let i = 0; i < this.tokens.length; i += 1) {
      const token = this.tokens[i];
      if (token.end > start && token.start < end && this.claimed[i]) return true;
    }
    return false;
  }

  /** Index of the first occurrence of `word`, or -1. */
  indexOf(word: string, from = 0): number {
    for (let i = Math.max(0, from); i < this.tokens.length; i += 1) {
      if (this.tokens[i].text === word) return i;
    }
    return -1;
  }

  has(word: string): boolean {
    return this.indexOf(word) !== -1;
  }

  /** Index of the first token in `words`, or -1. */
  indexOfAny(words: readonly string[]): number {
    for (let i = 0; i < this.tokens.length; i += 1) {
      if (words.includes(this.tokens[i].text)) return i;
    }
    return -1;
  }

  hasAny(words: readonly string[]): boolean {
    return this.indexOfAny(words) !== -1;
  }

  /**
   * Run `pattern` over the prompt text and claim the tokens each match covers.
   * The pattern must be global; matches are returned in order.
   */
  matchAll(pattern: RegExp): RegExpExecArray[] {
    const out: RegExpExecArray[] = [];
    pattern.lastIndex = 0;
    for (let m = pattern.exec(this.text); m !== null; m = pattern.exec(this.text)) {
      out.push(m);
      if (m[0].length === 0) pattern.lastIndex += 1;
    }
    return out;
  }

  /** Unclaimed tokens that look like they carried meaning, de-duplicated in order. */
  unclaimedWords(limit: number): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (let i = 0; i < this.tokens.length; i += 1) {
      if (this.claimed[i]) continue;
      const text = this.tokens[i].text;
      if (text.length < 3) continue;
      if (FILLER_WORDS.has(text)) continue;
      if (isCountToken(text)) continue;
      if (seen.has(text)) continue;
      seen.add(text);
      out.push(text);
      if (out.length >= limit) break;
    }
    return out;
  }
}

/** Title-case a hyphenated or spaced identifier for display in a plan. */
export function titleCase(text: string): string {
  return text
    .split(/[\s_]+/)
    .filter((part) => part.length > 0)
    .map((part) =>
      part
        .split('-')
        .map((piece) => (piece.length === 0 ? piece : piece[0].toUpperCase() + piece.slice(1)))
        .join('-'),
    )
    .join(' ');
}
