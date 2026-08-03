'use client';

import * as React from 'react';
import * as RadixTabs from '@radix-ui/react-tabs';
import { cn } from './cn';
import { FOCUS_RING_INSET, MOTION_BASE } from './styles';

export type TabsProps = React.ComponentPropsWithoutRef<typeof RadixTabs.Root>;

/** Tab group root. Roving focus and arrow-key navigation come from Radix. */
export const Tabs = React.forwardRef<HTMLDivElement, TabsProps>(function Tabs(
  { className, ...props },
  ref,
) {
  return (
    <RadixTabs.Root
      ref={ref}
      className={cn('flex min-h-0 flex-col', className)}
      {...props}
    />
  );
});

export type TabsListProps = React.ComponentPropsWithoutRef<typeof RadixTabs.List>;

export const TabsList = React.forwardRef<HTMLDivElement, TabsListProps>(function TabsList(
  { className, ...props },
  ref,
) {
  return (
    <RadixTabs.List
      ref={ref}
      className={cn(
        'flex shrink-0 items-center gap-0.5 border-b border-hairline px-1.5',
        className,
      )}
      {...props}
    />
  );
});

export interface TabProps extends React.ComponentPropsWithoutRef<typeof RadixTabs.Trigger> {
  icon?: React.ReactNode;
  /** Trailing count chip, e.g. the number of selected neurons. */
  badge?: React.ReactNode;
}

/**
 * A single tab. The active underline is an `::after`-style element scaled with
 * `transform`, so switching tabs never triggers layout.
 */
export const Tab = React.forwardRef<HTMLButtonElement, TabProps>(function Tab(
  { className, children, icon, badge, ...props },
  ref,
) {
  return (
    <RadixTabs.Trigger
      ref={ref}
      className={cn(
        'group relative flex h-8 items-center gap-1.5 rounded-t px-2.5 text-[11.5px] font-medium',
        'text-ink-faint hover:text-ink-muted',
        'data-[state=active]:text-ink',
        'disabled:pointer-events-none disabled:opacity-40',
        MOTION_BASE,
        FOCUS_RING_INSET,
        className,
      )}
      {...props}
    >
      {icon !== undefined && <span className="flex shrink-0 items-center">{icon}</span>}
      <span className="truncate">{children}</span>
      {badge !== undefined && (
        <span className="nf-numeric shrink-0 rounded-full bg-white/[0.07] px-1 text-[9.5px] leading-[14px] text-ink-faint">
          {badge}
        </span>
      )}
      <span
        aria-hidden
        className={cn(
          'pointer-events-none absolute inset-x-1.5 -bottom-px h-px origin-center scale-x-0 bg-accent opacity-0',
          'shadow-[0_0_10px_1px_rgb(79_209_255/0.7)]',
          MOTION_BASE,
          'group-data-[state=active]:scale-x-100 group-data-[state=active]:opacity-100',
        )}
      />
    </RadixTabs.Trigger>
  );
});

export type TabPanelProps = React.ComponentPropsWithoutRef<typeof RadixTabs.Content>;

export const TabPanel = React.forwardRef<HTMLDivElement, TabPanelProps>(function TabPanel(
  { className, ...props },
  ref,
) {
  return (
    <RadixTabs.Content
      ref={ref}
      className={cn(
        'min-h-0 flex-1 focus-visible:outline-none',
        'data-[state=inactive]:hidden',
        className,
      )}
      {...props}
    />
  );
});
