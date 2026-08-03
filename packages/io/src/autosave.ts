/**
 * Debounced autosave driver.
 *
 * A burst of edits coalesces into a single write. Two writes for the same
 * document never overlap: each save chains onto the previous one for that key,
 * so IndexedDB can never see them out of order. The tab losing visibility, or
 * being put into the back/forward cache, flushes immediately — that is the last
 * moment a browser reliably gives us before the page can be discarded.
 */

import type { Circuit, CircuitId } from '@neuroforge/shared';

import { saveCircuit } from './persistence';

export const DEFAULT_AUTOSAVE_INTERVAL_MS = 1500;

/**
 * Continuous editing must not postpone a write forever, so the debounce is
 * capped at this multiple of the interval since the first unsaved edit.
 */
const MAX_WAIT_MULTIPLIER = 4;

export class Autosaver {
  readonly #interval: number;
  readonly #maxWait: number;
  /** In-flight write per circuit id; the value is the tail of that chain. */
  readonly #writes = new Map<CircuitId, Promise<void>>();

  #getCircuit: (() => Circuit) | null = null;
  #onSaved: ((at: number) => void) | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #running = false;
  #dirty = false;
  #firstTouchAt = 0;

  constructor(intervalMs: number = DEFAULT_AUTOSAVE_INTERVAL_MS) {
    this.#interval = Math.max(0, intervalMs);
    this.#maxWait = this.#interval * MAX_WAIT_MULTIPLIER;
  }

  /** True while a write is outstanding or an edit is waiting to be written. */
  get pending(): boolean {
    return this.#dirty || this.#writes.size > 0;
  }

  start(getCircuit: () => Circuit, onSaved: (at: number) => void): void {
    if (this.#running) this.stop();
    this.#getCircuit = getCircuit;
    this.#onSaved = onSaved;
    this.#running = true;
    this.#dirty = false;
    this.#firstTouchAt = 0;

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.#handleVisibilityChange);
    }
    if (typeof window !== 'undefined') {
      window.addEventListener('pagehide', this.#handlePageHide);
    }
  }

  /** Mark the document as changed and (re)arm the debounce. */
  touch(): void {
    if (!this.#running) return;
    const now = Date.now();
    if (!this.#dirty) this.#firstTouchAt = now;
    this.#dirty = true;
    const waited = now - this.#firstTouchAt;
    const delay = Math.max(0, Math.min(this.#interval, this.#maxWait - waited));
    this.#clearTimer();
    this.#timer = setTimeout(() => {
      this.#timer = null;
      void this.#write();
    }, delay);
  }

  /** Write any pending edit now and wait for every outstanding write to land. */
  async flush(): Promise<void> {
    this.#clearTimer();
    if (this.#dirty) await this.#write();
    while (this.#writes.size > 0) {
      await Promise.all([...this.#writes.values()]);
    }
  }

  /** Detach listeners and cancel the debounce, writing a final time if needed. */
  stop(): void {
    this.#clearTimer();
    const hadPendingEdit = this.#dirty;
    this.#running = false;

    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.#handleVisibilityChange);
    }
    if (typeof window !== 'undefined') {
      window.removeEventListener('pagehide', this.#handlePageHide);
    }

    if (hadPendingEdit) void this.#write();
    this.#getCircuit = null;
    this.#onSaved = null;
  }

  #clearTimer(): void {
    if (this.#timer !== null) {
      clearTimeout(this.#timer);
      this.#timer = null;
    }
  }

  readonly #handleVisibilityChange = (): void => {
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
      void this.flush();
    }
  };

  readonly #handlePageHide = (): void => {
    void this.flush();
  };

  async #write(): Promise<void> {
    const getCircuit = this.#getCircuit;
    if (!this.#dirty || !getCircuit) return;

    // The document is snapshotted the moment the write is queued; anything the
    // user types after this point sets the flag again and schedules the next one.
    const circuit = getCircuit();
    this.#dirty = false;
    this.#firstTouchAt = 0;

    const key = circuit.id;
    const previous = this.#writes.get(key) ?? Promise.resolve();
    const onSaved = this.#onSaved;
    const next = previous.then(async () => {
      try {
        await saveCircuit(circuit);
        if (onSaved) onSaved(Date.now());
      } catch (error) {
        // A failed autosave must not poison the chain or reject into a caller
        // that only asked for a debounce.
        console.error('[neuroforge/io] autosave failed', error);
      }
    });
    this.#writes.set(key, next);
    void next.then(() => {
      if (this.#writes.get(key) === next) this.#writes.delete(key);
    });
    await next;
  }
}
