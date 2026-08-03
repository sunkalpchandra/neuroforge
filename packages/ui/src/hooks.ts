'use client';

import { useCallback, useRef, useState, useSyncExternalStore } from 'react';
import type { Ref, RefCallback } from 'react';

/**
 * Merge several refs into one callback ref. React 19 ref cleanups are supported
 * for object refs by nulling them on detach, which is what React itself does.
 */
export function useMergedRefs<T>(...refs: readonly (Ref<T> | undefined)[]): RefCallback<T> {
  const stored = useRef<readonly (Ref<T> | undefined)[]>(refs);
  stored.current = refs;

  return useCallback((node: T | null) => {
    for (const ref of stored.current) {
      if (typeof ref === 'function') {
        ref(node);
      } else if (ref !== null && ref !== undefined) {
        (ref as { current: T | null }).current = node;
      }
    }
  }, []);
}

/**
 * A stable function identity that always calls the latest `handler`. Lets an
 * effect or an event listener registered once still see fresh props without
 * re-subscribing every render.
 */
export function useEventCallback<Args extends readonly unknown[], Result>(
  handler: (...args: Args) => Result,
): (...args: Args) => Result {
  const stored = useRef(handler);
  stored.current = handler;
  return useCallback((...args: Args) => stored.current(...args), []);
}

/** Keeps a mutable ref pointing at the most recent render's value. */
export function useLatest<T>(value: T): { readonly current: T } {
  const stored = useRef(value);
  stored.current = value;
  return stored;
}

/**
 * Controlled/uncontrolled state. `controlled` being `undefined` selects the
 * uncontrolled branch; passing a value pins the component to the caller.
 */
export function useControllableState<T>(
  controlled: T | undefined,
  defaultValue: T,
  onChange?: (value: T) => void,
): [T, (next: T) => void] {
  const [internal, setInternal] = useState<T>(defaultValue);
  const isControlled = controlled !== undefined;
  const value = isControlled ? controlled : internal;
  const changeRef = useLatest(onChange);
  const controlledRef = useLatest(isControlled);

  const setValue = useCallback(
    (next: T) => {
      if (!controlledRef.current) setInternal(next);
      changeRef.current?.(next);
    },
    [changeRef, controlledRef],
  );

  return [value, setValue];
}

const subscribeToNothing = (): (() => void) => (): void => undefined;

function detectMac(): boolean {
  if (typeof navigator === 'undefined') return false;
  const candidate = navigator as Navigator & { userAgentData?: { platform?: string } };
  const platform = candidate.userAgentData?.platform ?? candidate.platform ?? '';
  return /mac|iphone|ipad|ipod/i.test(platform);
}

const serverIsMac = (): boolean => false;

/**
 * Platform detection that is safe under SSR: the server and the first client
 * render both report `false`, then the store settles to the real value, so
 * hydration never mismatches.
 */
export function useIsMac(): boolean {
  return useSyncExternalStore(subscribeToNothing, detectMac, serverIsMac);
}
