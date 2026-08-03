'use client';

import * as React from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from './cn';
import { FOCUS_RING_INSET, MOTION_FAST, useRuntimeStyles } from './styles';
import { useControllableState } from './hooks';

export interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Use the brighter glass surface reserved for floating, non-docked panels. */
  raised?: boolean;
}

/** The glass surface every floating panel in the app is built on. */
export const Panel = React.forwardRef<HTMLDivElement, PanelProps>(function Panel(
  { raised = false, className, children, ...props },
  ref,
) {
  useRuntimeStyles();

  return (
    <div
      ref={ref}
      className={cn(
        'flex min-h-0 flex-col overflow-hidden rounded-panel text-ink',
        raised ? 'nf-glass-raised' : 'nf-glass',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
});

export interface PanelHeaderProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  title: React.ReactNode;
  /** Secondary line under the title. */
  subtitle?: React.ReactNode;
  icon?: React.ReactNode;
  /** Controls pinned to the trailing edge; typically IconButtons. */
  actions?: React.ReactNode;
  /** Marks the header as a drag handle for a window-manager layer above this one. */
  draggable?: boolean;
}

export const PanelHeader = React.forwardRef<HTMLDivElement, PanelHeaderProps>(
  function PanelHeader(
    { title, subtitle, icon, actions, draggable = false, className, children, ...props },
    ref,
  ) {
    return (
      <div
        ref={ref}
        data-drag-handle={draggable ? '' : undefined}
        className={cn(
          'flex shrink-0 items-center gap-2.5 border-b border-hairline px-3',
          subtitle === undefined ? 'h-10' : 'py-2',
          draggable && 'cursor-grab select-none active:cursor-grabbing',
          className,
        )}
        {...props}
      >
        {icon !== undefined && (
          <span className="flex size-4 shrink-0 items-center justify-center text-ink-muted">
            {icon}
          </span>
        )}
        <div className="flex min-w-0 flex-col">
          <span className="truncate text-[12px] font-medium tracking-[-0.01em] text-ink">
            {title}
          </span>
          {subtitle !== undefined && (
            <span className="truncate text-[10.5px] leading-tight text-ink-faint">{subtitle}</span>
          )}
        </div>
        {children}
        {actions !== undefined && (
          <div className="ml-auto flex shrink-0 items-center gap-0.5">{actions}</div>
        )}
      </div>
    );
  },
);

export interface PanelSectionProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'onToggle'> {
  /** Section eyebrow. Omit for an unlabelled group of controls. */
  label?: React.ReactNode;
  /** Trailing content on the label row, e.g. a reset button or a readout. */
  aside?: React.ReactNode;
  collapsible?: boolean;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Removes the top hairline; use on the first section of a panel. */
  flush?: boolean;
}

/** A labelled group of controls inside a Panel, optionally collapsible. */
export const PanelSection = React.forwardRef<HTMLDivElement, PanelSectionProps>(
  function PanelSection(
    {
      label,
      aside,
      collapsible = false,
      open,
      defaultOpen = true,
      onOpenChange,
      flush = false,
      className,
      children,
      ...props
    },
    ref,
  ) {
    const [isOpen, setOpen] = useControllableState(open, defaultOpen, onOpenChange);
    const contentId = React.useId();
    const expanded = collapsible ? isOpen : true;

    const heading =
      label === undefined && aside === undefined ? null : (
        <div className="flex h-7 items-center gap-2 px-3">
          {collapsible ? (
            <button
              type="button"
              aria-expanded={expanded}
              aria-controls={contentId}
              onClick={() => setOpen(!isOpen)}
              className={cn(
                'group -mx-1 flex min-w-0 flex-1 items-center gap-1.5 rounded px-1 py-0.5 text-left',
                MOTION_FAST,
                FOCUS_RING_INSET,
                'hover:text-ink',
              )}
            >
              <ChevronDown
                size={11}
                aria-hidden
                className={cn(
                  'shrink-0 text-ink-faint',
                  MOTION_FAST,
                  expanded ? 'rotate-0' : '-rotate-90',
                )}
              />
              <span className="truncate text-[10px] font-semibold uppercase tracking-[0.09em] text-ink-muted">
                {label}
              </span>
            </button>
          ) : (
            <span className="min-w-0 flex-1 truncate text-[10px] font-semibold uppercase tracking-[0.09em] text-ink-muted">
              {label}
            </span>
          )}
          {aside !== undefined && <div className="flex shrink-0 items-center gap-1">{aside}</div>}
        </div>
      );

    return (
      <div
        ref={ref}
        className={cn('flex flex-col py-2', !flush && 'border-t border-hairline', className)}
        {...props}
      >
        {heading}
        {expanded && (
          <div id={contentId} className="flex flex-col gap-1.5 px-3 pb-1 pt-1">
            {children}
          </div>
        )}
      </div>
    );
  },
);
