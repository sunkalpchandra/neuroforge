'use client';

import * as React from 'react';
import { cn } from './cn';
import { FOCUS_RING_INSET, MOTION_SLOW } from './styles';
import { useEventCallback } from './hooks';

export interface SegmentedOption<T extends string> {
  value: T;
  label: React.ReactNode;
  icon?: React.ReactNode;
  disabled?: boolean;
  /** Native tooltip; use the Tooltip component for anything richer. */
  title?: string;
}

export interface SegmentedControlProps<T extends string = string>
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'onChange' | 'defaultValue'> {
  value: T;
  onChange: (value: T) => void;
  options: readonly SegmentedOption<T>[];
  size?: 'sm' | 'md';
  /** Stretch to the width of the container instead of hugging the options. */
  fullWidth?: boolean;
}

const SEGMENT_SIZES: Record<'sm' | 'md', string> = {
  sm: 'h-6 text-[10.5px]',
  md: 'h-7 text-[11.5px]',
};

function SegmentedControlInner<T extends string>(
  {
    value,
    onChange,
    options,
    size = 'md',
    fullWidth = false,
    className,
    ...props
  }: SegmentedControlProps<T>,
  ref: React.ForwardedRef<HTMLDivElement>,
): React.ReactElement {
  const activeIndex = options.findIndex((option) => option.value === value);
  const buttonsRef = React.useRef<(HTMLButtonElement | null)[]>([]);

  const focusOption = useEventCallback((index: number) => {
    const count = options.length;
    if (count === 0) return;
    for (let offset = 0; offset < count; offset += 1) {
      const candidate = ((index + offset) % count + count) % count;
      if (options[candidate].disabled === true) continue;
      buttonsRef.current[candidate]?.focus();
      onChange(options[candidate].value);
      return;
    }
  });

  const handleKeyDown = useEventCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    const current = activeIndex < 0 ? 0 : activeIndex;
    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        event.preventDefault();
        focusOption(current + 1);
        return;
      case 'ArrowLeft':
      case 'ArrowUp':
        event.preventDefault();
        focusOption(current - 1);
        return;
      case 'Home':
        event.preventDefault();
        focusOption(0);
        return;
      case 'End':
        event.preventDefault();
        focusOption(options.length - 1);
        return;
      default:
        break;
    }
  });

  return (
    <div
      ref={ref}
      role="radiogroup"
      onKeyDown={handleKeyDown}
      className={cn(
        'relative isolate grid gap-0 rounded-control border border-hairline bg-black/25 p-0.5',
        'shadow-[0_1px_0_0_rgb(255_255_255/0.03)_inset]',
        fullWidth ? 'w-full' : 'w-max',
        className,
      )}
      style={{ gridTemplateColumns: `repeat(${Math.max(options.length, 1)}, minmax(0, 1fr))` }}
      {...props}
    >
      {/*
        The indicator occupies the first grid cell, so its width is always one
        segment; selection moves it with translateX alone and nothing reflows.
        This is why the segments are equal-width rather than content-sized.
      */}
      <span
        aria-hidden
        className={cn(
          'pointer-events-none col-start-1 row-start-1 rounded-[0.375rem] border border-accent/30',
          'bg-accent/16 shadow-[0_1px_0_0_rgb(255_255_255/0.06)_inset,0_4px_14px_-8px_rgb(79_209_255/0.8)]',
          MOTION_SLOW,
          activeIndex < 0 && 'opacity-0',
        )}
        style={{ transform: `translateX(${Math.max(activeIndex, 0) * 100}%)` }}
      />
      {options.map((option, index) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            ref={(node) => {
              buttonsRef.current[index] = node;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            title={option.title}
            disabled={option.disabled}
            tabIndex={selected || (activeIndex < 0 && index === 0) ? 0 : -1}
            onClick={() => onChange(option.value)}
            style={{ gridColumnStart: index + 1, gridRowStart: 1 }}
            className={cn(
              'z-10 flex min-w-0 items-center justify-center gap-1.5 rounded-[0.375rem] px-2.5 font-medium',
              'disabled:pointer-events-none disabled:opacity-35',
              MOTION_SLOW,
              FOCUS_RING_INSET,
              SEGMENT_SIZES[size],
              selected ? 'text-ink' : 'text-ink-muted hover:text-ink',
            )}
          >
            {option.icon !== undefined && (
              <span className="flex shrink-0 items-center">{option.icon}</span>
            )}
            <span className="truncate">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/**
 * Animated pill selector. Options share one column width so the indicator can
 * slide purely with `transform`.
 */
export const SegmentedControl = React.forwardRef(SegmentedControlInner) as <T extends string>(
  props: SegmentedControlProps<T> & React.RefAttributes<HTMLDivElement>,
) => React.ReactElement;
