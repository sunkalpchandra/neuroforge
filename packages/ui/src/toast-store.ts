'use client';

/**
 * A tiny external store for transient notifications.
 *
 * Deliberately not Zustand: the UI package must stay dependency-light and
 * consumable from anywhere, including modules that run before the app store
 * exists. `useSyncExternalStore` gives concurrent-safe subscriptions with no
 * library at all.
 */

export type ToastTone = 'neutral' | 'success' | 'warning' | 'danger' | 'accent';

export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface ToastOptions {
  title: string;
  description?: string;
  tone?: ToastTone;
  /** Milliseconds before auto-dismiss. `0` or `Infinity` pins the toast. */
  duration?: number;
  action?: ToastAction;
  /** Reuse an id to replace an existing toast in place, e.g. a progress update. */
  id?: string;
}

export interface ToastRecord {
  readonly id: string;
  readonly title: string;
  readonly description: string | undefined;
  readonly tone: ToastTone;
  readonly duration: number;
  readonly action: ToastAction | undefined;
  readonly createdAt: number;
}

/** Newest first; the viewport renders them in this order. */
const EMPTY: readonly ToastRecord[] = Object.freeze([]);
const MAX_VISIBLE = 5;
const DEFAULT_DURATION = 5000;

let toasts: readonly ToastRecord[] = EMPTY;
let counter = 0;

const listeners = new Set<() => void>();

function emit(next: readonly ToastRecord[]): void {
  toasts = next;
  for (const listener of listeners) listener();
}

export function subscribeToToasts(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getToastSnapshot(): readonly ToastRecord[] {
  return toasts;
}

export function getToastServerSnapshot(): readonly ToastRecord[] {
  return EMPTY;
}

/**
 * Queue a notification. Returns the toast id so the caller can dismiss or
 * replace it later.
 */
export function pushToast(options: ToastOptions | string): string {
  const input: ToastOptions = typeof options === 'string' ? { title: options } : options;
  counter += 1;
  const id = input.id ?? `toast-${counter}`;
  const record: ToastRecord = {
    id,
    title: input.title,
    description: input.description,
    tone: input.tone ?? 'neutral',
    duration: input.duration ?? DEFAULT_DURATION,
    action: input.action,
    createdAt: Date.now(),
  };

  const existing = toasts.findIndex((toast) => toast.id === id);
  if (existing >= 0) {
    const next = toasts.slice();
    next[existing] = record;
    emit(next);
    return id;
  }

  emit([record, ...toasts].slice(0, MAX_VISIBLE));
  return id;
}

export function dismissToast(id: string): void {
  const next = toasts.filter((toast) => toast.id !== id);
  if (next.length !== toasts.length) emit(next);
}

export function clearToasts(): void {
  if (toasts.length > 0) emit(EMPTY);
}
