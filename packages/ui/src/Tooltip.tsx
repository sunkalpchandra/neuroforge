'use client';

import * as React from 'react';
import * as RadixTooltip from '@radix-ui/react-tooltip';
import { cn } from './cn';
import { useRuntimeStyles } from './styles';
import { Kbd } from './Kbd';

export interface TooltipProps {
  /** Tooltip body. `null` or `undefined` renders the trigger untouched. */
  content: React.ReactNode;
  /** Exactly one focusable element; the tooltip attaches to it via `asChild`. */
  children: React.ReactNode;
  /** Shortcut chip rendered after the label, e.g. `"Mod+K"`. */
  shortcut?: string;
  side?: RadixTooltip.TooltipContentProps['side'];
  align?: RadixTooltip.TooltipContentProps['align'];
  sideOffset?: number;
  /** Hover dwell before the tooltip opens, in ms. */
  delay?: number;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  className?: string;
}

/**
 * Self-contained tooltip: it mounts its own Radix provider so a consumer never
 * has to remember to wrap the tree. Nested providers are cheap and the delay is
 * carried on the provider rather than the root, which is what makes the
 * skip-delay grouping work for adjacent toolbar buttons.
 */
export const Tooltip = React.forwardRef<HTMLDivElement, TooltipProps>(function Tooltip(
  {
    content,
    children,
    shortcut,
    side = 'bottom',
    align = 'center',
    sideOffset = 6,
    delay = 260,
    open,
    defaultOpen,
    onOpenChange,
    className,
  },
  ref,
) {
  useRuntimeStyles();

  if (content === null || content === undefined || content === false) {
    return <>{children}</>;
  }

  return (
    <RadixTooltip.Provider delayDuration={delay} skipDelayDuration={320}>
      <RadixTooltip.Root open={open} defaultOpen={defaultOpen} onOpenChange={onOpenChange}>
        <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
        <RadixTooltip.Portal>
          <RadixTooltip.Content
            ref={ref}
            side={side}
            align={align}
            sideOffset={sideOffset}
            collisionPadding={8}
            className={cn(
              'nf-glass-raised nf-anim-pop z-[70] flex max-w-64 items-center gap-2',
              'rounded-[0.4rem] px-2 py-1 text-[11px] leading-snug text-ink',
              className,
            )}
          >
            <span className="min-w-0">{content}</span>
            {shortcut !== undefined && <Kbd keys={shortcut} size="sm" />}
          </RadixTooltip.Content>
        </RadixTooltip.Portal>
      </RadixTooltip.Root>
    </RadixTooltip.Provider>
  );
});
