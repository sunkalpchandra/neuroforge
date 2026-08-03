'use client';

import * as React from 'react';
import { cn } from './cn';
import { useIsMac } from './hooks';

/** Symbols that read better than their names on every platform. */
const UNIVERSAL_SYMBOLS: Record<string, string> = {
  shift: '⇧',
  enter: '↵',
  return: '↵',
  escape: 'Esc',
  esc: 'Esc',
  arrowup: '↑',
  arrowdown: '↓',
  arrowleft: '←',
  arrowright: '→',
  up: '↑',
  down: '↓',
  left: '←',
  right: '→',
  backspace: '⌫',
  delete: '⌦',
  space: 'Space',
  tab: '⇥',
  pageup: 'PgUp',
  pagedown: 'PgDn',
};

const MAC_SYMBOLS: Record<string, string> = {
  mod: '⌘',
  cmd: '⌘',
  meta: '⌘',
  command: '⌘',
  alt: '⌥',
  option: '⌥',
  ctrl: '⌃',
  control: '⌃',
};

const OTHER_SYMBOLS: Record<string, string> = {
  mod: 'Ctrl',
  cmd: 'Ctrl',
  meta: 'Win',
  command: 'Ctrl',
  alt: 'Alt',
  option: 'Alt',
  ctrl: 'Ctrl',
  control: 'Ctrl',
};

function renderKey(token: string, isMac: boolean): string {
  const key = token.trim();
  if (key.length === 0) return key;
  const lower = key.toLowerCase();
  const platform = isMac ? MAC_SYMBOLS[lower] : OTHER_SYMBOLS[lower];
  if (platform !== undefined) return platform;
  const universal = UNIVERSAL_SYMBOLS[lower];
  if (universal !== undefined) return universal;
  return key.length === 1 ? key.toUpperCase() : key;
}

export interface KbdProps extends React.HTMLAttributes<HTMLElement> {
  /** Shortcut in `Mod+Shift+K` form. `Mod` resolves to ⌘ on macOS, Ctrl elsewhere. */
  keys?: string;
  size?: 'sm' | 'md';
  /** Renders `+` between chips instead of butting them together. */
  separated?: boolean;
}

const CHIP_SIZES: Record<NonNullable<KbdProps['size']>, string> = {
  sm: 'h-4 min-w-4 px-1 text-[9.5px]',
  md: 'h-5 min-w-5 px-1.5 text-[10.5px]',
};

/**
 * Keyboard shortcut chips. Platform detection is SSR-safe: the server and the
 * first client render both assume a non-Mac layout, then settle.
 */
export const Kbd = React.forwardRef<HTMLElement, KbdProps>(function Kbd(
  { keys, size = 'md', separated = false, className, children, ...props },
  ref,
) {
  const isMac = useIsMac();
  const tokens = React.useMemo(
    () =>
      keys === undefined
        ? []
        : keys
            .split('+')
            .map((token) => renderKey(token, isMac))
            .filter((token) => token.length > 0),
    [isMac, keys],
  );

  const chip = cn(
    'nf-numeric inline-flex shrink-0 items-center justify-center rounded border border-hairline-strong',
    'bg-white/[0.06] font-medium text-ink-muted shadow-[0_1px_0_0_rgb(0_0_0/0.35)]',
    CHIP_SIZES[size],
  );

  return (
    <kbd
      ref={ref}
      className={cn('inline-flex shrink-0 items-center gap-0.5 leading-none', className)}
      {...props}
    >
      {tokens.map((token, index) => (
        <React.Fragment key={`${token}-${index}`}>
          {separated && index > 0 && (
            <span aria-hidden className="text-[9px] text-ink-faint">
              +
            </span>
          )}
          <span className={chip}>{token}</span>
        </React.Fragment>
      ))}
      {children !== undefined && <span className={chip}>{children}</span>}
    </kbd>
  );
});
