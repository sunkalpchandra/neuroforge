'use client';

import * as React from 'react';
import * as RadixSlider from '@radix-ui/react-slider';
import { cn } from './cn';
import { MOTION_FAST } from './styles';
import { useFieldControl } from './Field';
import { useEventCallback } from './hooks';
import { clamp, decimalsForStep, formatFixed, snapToStep, toNormalized } from './numeric';

const FINE_MULTIPLIER = 0.1;
const COARSE_MULTIPLIER = 10;

const ARROW_DIRECTION: Record<string, number> = {
  ArrowRight: 1,
  ArrowUp: 1,
  ArrowLeft: -1,
  ArrowDown: -1,
};

export interface SliderProps
  extends Omit<
    React.ComponentPropsWithoutRef<typeof RadixSlider.Root>,
    'value' | 'defaultValue' | 'onChange' | 'onValueChange' | 'onValueCommit' | 'asChild'
  > {
  value: number;
  onChange: (value: number) => void;
  /** Fired when the drag ends or a keyboard adjustment settles. */
  onCommit?: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  /** Fill outward from `origin` instead of from the minimum. */
  bipolar?: boolean;
  /** Centre of a bipolar fill and the double-click reset target. Defaults to 0
   *  when bipolar, otherwise to `min`. */
  origin?: number;
  /** Value restored by a double-click. Defaults to `origin`. */
  defaultValue?: number;
  /** Inline readout on the trailing edge of the track. */
  showValue?: boolean;
  unit?: string;
  formatValue?: (value: number) => string;
  /** Tinting for the fill; use `secondary` for inhibitory parameters. */
  tone?: 'accent' | 'secondary' | 'success' | 'warning' | 'danger';
}

const TONE_FILL: Record<NonNullable<SliderProps['tone']>, string> = {
  accent: 'from-accent-dim to-accent',
  secondary: 'from-secondary-dim to-secondary',
  success: 'from-success/50 to-success',
  warning: 'from-warning/50 to-warning',
  danger: 'from-danger/50 to-danger',
};

const TONE_THUMB: Record<NonNullable<SliderProps['tone']>, string> = {
  accent:
    'border-accent bg-accent shadow-[0_0_0_3px_rgb(79_209_255/0.14)] hover:shadow-[0_0_0_7px_rgb(79_209_255/0.20)]',
  secondary:
    'border-secondary bg-secondary shadow-[0_0_0_3px_rgb(182_107_255/0.14)] hover:shadow-[0_0_0_7px_rgb(182_107_255/0.20)]',
  success:
    'border-success bg-success shadow-[0_0_0_3px_rgb(74_222_128/0.14)] hover:shadow-[0_0_0_7px_rgb(74_222_128/0.20)]',
  warning:
    'border-warning bg-warning shadow-[0_0_0_3px_rgb(251_191_36/0.14)] hover:shadow-[0_0_0_7px_rgb(251_191_36/0.20)]',
  danger:
    'border-danger bg-danger shadow-[0_0_0_3px_rgb(251_113_133/0.14)] hover:shadow-[0_0_0_7px_rgb(251_113_133/0.20)]',
};

/**
 * Single-value slider over the Radix primitive.
 *
 * Adds three things Radix leaves to the consumer: a bipolar fill that grows out
 * from a centre origin, Shift/Alt modifiers on arrow keys, and double-click to
 * restore the default.
 */
export const Slider = React.forwardRef<HTMLSpanElement, SliderProps>(function Slider(
  {
    value,
    onChange,
    onCommit,
    min = 0,
    max = 1,
    step = 0.01,
    bipolar = false,
    origin,
    defaultValue,
    showValue = false,
    unit,
    formatValue,
    tone = 'accent',
    disabled,
    id,
    className,
    onKeyDown,
    ...rest
  },
  ref,
) {
  const field = useFieldControl(id);
  const isDisabled = disabled === true || field.disabled;
  const centre = origin ?? (bipolar ? 0 : min);
  const decimals = decimalsForStep(step);

  const readout = React.useMemo(() => {
    const text = formatValue?.(value) ?? formatFixed(value, decimals);
    return unit === undefined ? text : `${text} ${unit}`;
  }, [decimals, formatValue, unit, value]);

  const handleValueChange = useEventCallback((next: readonly number[]) => {
    onChange(clamp(next[0], min, max));
  });

  const handleValueCommit = useEventCallback((next: readonly number[]) => {
    onCommit?.(clamp(next[0], min, max));
  });

  const handleKeyDown = useEventCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    onKeyDown?.(event);
    if (event.defaultPrevented || isDisabled) return;
    const direction = ARROW_DIRECTION[event.key];
    if (direction === undefined) return;
    if (!event.shiftKey && !event.altKey) return;

    // Radix steps by exactly `step`; intercept so modifiers can refine or
    // coarsen the grid, then stop it from stepping a second time.
    event.preventDefault();
    const multiplier = event.shiftKey ? FINE_MULTIPLIER : COARSE_MULTIPLIER;
    const grid = step * multiplier;
    const next = clamp(snapToStep(value + direction * grid, grid, min), min, max);
    if (next !== value) onChange(next);
    onCommit?.(next);
  });

  const handleDoubleClick = useEventCallback(() => {
    if (isDisabled) return;
    const target = clamp(defaultValue ?? centre, min, max);
    if (target !== value) onChange(target);
    onCommit?.(target);
  });

  const valuePercent = toNormalized(value, min, max) * 100;
  const originPercent = toNormalized(centre, min, max) * 100;
  const fillLeft = Math.min(valuePercent, originPercent);
  const fillWidth = Math.abs(valuePercent - originPercent);

  return (
    <div className={cn('group/slider flex min-w-0 items-center gap-2.5', className)}>
      <RadixSlider.Root
        {...rest}
        ref={ref}
        id={field.id}
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={isDisabled}
        onValueChange={handleValueChange}
        onValueCommit={handleValueCommit}
        onKeyDown={handleKeyDown}
        onDoubleClick={handleDoubleClick}
        aria-describedby={rest['aria-describedby'] ?? field.describedBy}
        aria-labelledby={
          rest['aria-labelledby'] ?? (rest['aria-label'] === undefined ? field.labelledBy : undefined)
        }
        className={cn(
          'relative flex h-5 min-w-0 flex-1 touch-none select-none items-center',
          isDisabled && 'pointer-events-none opacity-40',
        )}
      >
        <RadixSlider.Track className="relative h-1 w-full grow overflow-hidden rounded-full bg-white/[0.07] shadow-[0_1px_0_0_rgb(255_255_255/0.03)_inset]">
          {bipolar ? (
            <span
              aria-hidden
              className={cn('absolute inset-y-0 bg-gradient-to-r', TONE_FILL[tone])}
              style={{ left: `${fillLeft}%`, width: `${fillWidth}%` }}
            />
          ) : (
            <RadixSlider.Range
              className={cn('absolute h-full bg-gradient-to-r', TONE_FILL[tone])}
            />
          )}
        </RadixSlider.Track>
        {bipolar && (
          <span
            aria-hidden
            className="pointer-events-none absolute top-1/2 h-2.5 w-px -translate-y-1/2 bg-white/20"
            style={{ left: `${originPercent}%` }}
          />
        )}
        <RadixSlider.Thumb
          className={cn(
            'block size-3 rounded-full border',
            MOTION_FAST,
            TONE_THUMB[tone],
            'hover:scale-110 active:scale-95',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-bg',
          )}
        />
      </RadixSlider.Root>
      {showValue && (
        <span className="nf-numeric w-14 shrink-0 text-right text-[11px] tabular-nums text-ink-muted">
          {readout}
        </span>
      )}
    </div>
  );
});
