'use client';

import { useCallback, useEffect, useRef } from 'react';
import { Tooltip } from '@neuroforge/ui';

import { useDock } from '@/lib/dock-store';
import type { DockSide, DockTab } from '@/lib/dock-store';

export interface DockEntry {
  tab: DockTab;
  title: string;
  icon: React.ReactNode;
  shortcut?: string;
  render: () => React.ReactNode;
}

interface DockProps {
  side: DockSide;
  entries: readonly DockEntry[];
}

/** Thickness of the icon rail, matching the value the layout reserves. */
const RAIL = 34;

/**
 * A docked panel column with an icon rail.
 *
 * The rail is always visible so every tool is one click away and the set of
 * available tools is discoverable without opening menus; only one panel body per
 * side is mounted at a time, which also means a hidden panel costs nothing —
 * several of these run simulations or rebuild matrices when open.
 */
export function Dock({ side, entries }: DockProps) {
  const state = useDock((s) => s[side]);
  const toggle = useDock((s) => s.toggle);
  const resize = useDock((s) => s.resize);
  const dragging = useRef(false);

  const active = entries.find((entry) => entry.tab === state.active) ?? null;
  const vertical = side !== 'bottom';

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      dragging.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!dragging.current) return;
      // Measured from the window edge the dock is attached to, so the handle
      // tracks the pointer exactly regardless of where the drag started.
      if (side === 'left') resize(side, event.clientX - RAIL);
      else if (side === 'right') resize(side, window.innerWidth - event.clientX - RAIL);
      else resize(side, window.innerHeight - event.clientY);
    },
    [resize, side],
  );

  const onPointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    event.currentTarget.releasePointerCapture(event.pointerId);
  }, []);

  // A pointer released outside the window never fires pointerup on the handle,
  // which would leave the dock resizing on the next unrelated pointer move.
  useEffect(() => {
    const stop = () => {
      dragging.current = false;
    };
    window.addEventListener('pointercancel', stop);
    window.addEventListener('blur', stop);
    return () => {
      window.removeEventListener('pointercancel', stop);
      window.removeEventListener('blur', stop);
    };
  }, []);

  const rail = (
    <div
      className={`flex shrink-0 gap-0.5 border-hairline bg-panel/60 ${
        vertical
          ? `w-[${RAIL}px] flex-col items-center border-l border-r py-1.5`
          : 'h-[34px] flex-row items-center border-t px-1.5'
      }`}
      style={vertical ? { width: RAIL } : { height: RAIL }}
    >
      {entries.map((entry) => {
        const on = state.active === entry.tab;
        return (
          <Tooltip
            key={entry.tab}
            content={entry.title}
            shortcut={entry.shortcut}
            side={side === 'right' ? 'left' : side === 'left' ? 'right' : 'top'}
          >
            <button
              type="button"
              aria-pressed={on}
              aria-label={entry.title}
              onClick={() => toggle(side, entry.tab)}
              className={`flex size-[26px] items-center justify-center rounded-[5px] transition-colors [&>svg]:size-[15px] ${
                on
                  ? 'bg-accent/15 text-accent'
                  : 'text-ink-faint hover:bg-panel-raised hover:text-ink-muted'
              }`}
            >
              {entry.icon}
            </button>
          </Tooltip>
        );
      })}
    </div>
  );

  const handle = active ? (
    <div
      role="separator"
      aria-orientation={vertical ? 'vertical' : 'horizontal'}
      aria-label={`Resize ${side} panel`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      className={`shrink-0 bg-transparent transition-colors hover:bg-accent/30 ${
        vertical ? 'w-[3px] cursor-col-resize' : 'h-[3px] cursor-row-resize'
      }`}
    />
  ) : null;

  const body = active ? (
    <div
      className="nf-docked relative flex min-h-0 min-w-0 flex-col overflow-hidden border-hairline bg-panel/45"
      style={vertical ? { width: state.size } : { height: state.size }}
    >
      {active.render()}
    </div>
  ) : null;

  if (side === 'bottom') {
    return (
      <div className="flex shrink-0 flex-col">
        {handle}
        {body}
        {rail}
      </div>
    );
  }

  return (
    <div className="flex min-h-0 shrink-0">
      {side === 'right' ? handle : null}
      {side === 'right' ? body : null}
      {rail}
      {side === 'left' ? body : null}
      {side === 'left' ? handle : null}
    </div>
  );
}
