/**
 * The undo/redo system.
 *
 * A command carries only what it changed. `DiffCommand` — the one every built-in
 * edit produces — stores the previous and next value of each touched entity as
 * bare references into the copy-on-write history of the document, so recording
 * an edit costs O(edit), never O(document). Dragging a slider across a hundred
 * frames coalesces into a single entry holding two versions of one entity.
 */

import type { Circuit } from '@neuroforge/shared';

import type { CircuitDiff } from './draft';
import { applyDiff, isEmptyDiff, mergeDiff, runDraft } from './draft';

/** An undoable operation. `apply` and `revert` must be exact inverses. */
export interface Command {
  readonly label: string;
  apply(draft: Circuit): void;
  revert(draft: Circuit): void;
  /** Commands with the same mergeKey issued within the coalesce window merge. */
  readonly mergeKey?: string;
}

/** Longest chain of undoable edits retained before the oldest is dropped. */
export const DEFAULT_HISTORY_DEPTH = 200;

/** Two edits sharing a mergeKey closer together than this become one entry. */
export const DEFAULT_COALESCE_MS = 400;

/* --------------------------------------------------------------- commands -- */

/** A command expressed as a before/after diff of the entities it touched. */
export class DiffCommand implements Command {
  readonly label: string;
  readonly mergeKey?: string;
  readonly diff: CircuitDiff;

  constructor(label: string, diff: CircuitDiff, mergeKey?: string) {
    this.label = label;
    this.diff = diff;
    this.mergeKey = mergeKey;
  }

  apply(draft: Circuit): void {
    applyDiff(draft, this.diff, 'after');
  }

  revert(draft: Circuit): void {
    applyDiff(draft, this.diff, 'before');
  }
}

/**
 * Two commands that undo as one. Used when coalescing merges entries that are
 * not both diffs — an externally supplied `Command` cannot be composed
 * structurally, but it can be replayed in order and reverted in reverse.
 */
export class CompositeCommand implements Command {
  readonly label: string;
  readonly mergeKey?: string;
  readonly parts: readonly Command[];

  constructor(label: string, parts: readonly Command[], mergeKey?: string) {
    this.label = label;
    this.parts = parts;
    this.mergeKey = mergeKey;
  }

  apply(draft: Circuit): void {
    for (let i = 0; i < this.parts.length; i += 1) this.parts[i].apply(draft);
  }

  revert(draft: Circuit): void {
    for (let i = this.parts.length - 1; i >= 0; i -= 1) this.parts[i].revert(draft);
  }
}

function flatten(command: Command): readonly Command[] {
  return command instanceof CompositeCommand ? command.parts : [command];
}

/** Compose two commands into one entry with the same net effect. */
export function mergeCommands(first: Command, second: Command): Command {
  if (first instanceof DiffCommand && second instanceof DiffCommand) {
    return new DiffCommand(second.label, mergeDiff(first.diff, second.diff), second.mergeKey);
  }
  return new CompositeCommand(
    second.label,
    [...flatten(first), ...flatten(second)],
    second.mergeKey,
  );
}

/* ---------------------------------------------------------------- running -- */

/** Run a command forward against a document, returning the new document. */
export function applyCommand(circuit: Circuit, command: Command): Circuit {
  return runDraft(circuit, (draft) => command.apply(draft), false).circuit;
}

/** Run a command backward against a document, returning the new document. */
export function revertCommand(circuit: Circuit, command: Command): Circuit {
  return runDraft(circuit, (draft) => command.revert(draft), false).circuit;
}

export interface TransactionResult {
  readonly circuit: Circuit;
  /** Null when `mutate` changed nothing, so no history entry is warranted. */
  readonly command: Command | null;
}

/**
 * Run an arbitrary mutation and capture it as a command automatically. Only the
 * entities the mutation actually touched are recorded.
 */
export function createTransaction(
  circuit: Circuit,
  label: string,
  mutate: (draft: Circuit) => void,
  mergeKey?: string,
): TransactionResult {
  const result = runDraft(circuit, mutate, true);
  if (result.diff === null || isEmptyDiff(result.diff)) {
    return { circuit, command: null };
  }
  return { circuit: result.circuit, command: new DiffCommand(label, result.diff, mergeKey) };
}

/* ---------------------------------------------------------------- history -- */

export interface HistoryOptions {
  depth?: number;
  coalesceMs?: number;
  /** Injectable clock; the coalesce window is the only thing that reads it. */
  now?: () => number;
}

/**
 * A bounded stack of applied commands plus the redo stack they move to.
 *
 * The history never holds a document, only the commands, so its memory cost is
 * proportional to the edits made rather than to the size of the circuit.
 */
export class History {
  readonly #undo: Command[] = [];
  readonly #redo: Command[] = [];
  readonly #depth: number;
  readonly #window: number;
  readonly #now: () => number;
  #lastRecordedAt = Number.NEGATIVE_INFINITY;

  constructor(options: HistoryOptions = {}) {
    this.#depth = Math.max(1, Math.floor(options.depth ?? DEFAULT_HISTORY_DEPTH));
    this.#window = Math.max(0, options.coalesceMs ?? DEFAULT_COALESCE_MS);
    this.#now = options.now ?? Date.now;
  }

  get undoDepth(): number {
    return this.#undo.length;
  }

  get redoDepth(): number {
    return this.#redo.length;
  }

  clear(): void {
    this.#undo.length = 0;
    this.#redo.length = 0;
    this.#lastRecordedAt = Number.NEGATIVE_INFINITY;
  }

  /** Record a command that has already been applied to the document. */
  record(command: Command): void {
    this.#redo.length = 0;
    const at = this.#now();
    const top = this.#undo[this.#undo.length - 1];
    const coalesces =
      top !== undefined &&
      command.mergeKey !== undefined &&
      top.mergeKey === command.mergeKey &&
      at - this.#lastRecordedAt <= this.#window;

    if (coalesces) {
      this.#undo[this.#undo.length - 1] = mergeCommands(top, command);
    } else {
      this.#undo.push(command);
      while (this.#undo.length > this.#depth) this.#undo.shift();
    }
    this.#lastRecordedAt = at;
  }

  /**
   * Move the newest entry onto the redo stack and hand it back so the caller can
   * revert it. Returns null when there is nothing to undo.
   */
  takeUndo(): Command | null {
    const command = this.#undo.pop();
    if (command === undefined) return null;
    this.#redo.push(command);
    // An undo ends the coalescing run; the next edit starts a fresh entry.
    this.#lastRecordedAt = Number.NEGATIVE_INFINITY;
    return command;
  }

  /** Move the newest redo entry back onto the undo stack and hand it back. */
  takeRedo(): Command | null {
    const command = this.#redo.pop();
    if (command === undefined) return null;
    this.#undo.push(command);
    this.#lastRecordedAt = Number.NEGATIVE_INFINITY;
    return command;
  }
}
