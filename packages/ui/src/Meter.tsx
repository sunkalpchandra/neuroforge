'use client';

import * as React from 'react';
import { cn } from './cn';
import { MOTION_BASE } from './styles';
import { clamp } from './numeric';

export type MeterTone = 'accent' | 'secondary' | 'success' | 'warning' | 'danger';

export interface MeterProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> {
  /** Normalised 0..1; values outside the range are clamped. */
  value: number;
  label?: React.ReactNode;
  /** Right-aligned readout; supply a formatted string with its unit. */
  valueLabel?: React.ReactNode;
  tone?: MeterTone;
  size?: 'sm' | 'md';
  /** Faint tick marks across the track; 0 disables them. */
  ticks?: number;
  /** Marker drawn at a normalised position, e.g. a budget or a threshold. */
  marker?: number;
}

const TONE_FILL: Record<MeterTone, string> = {
  accent: 'from-accent-dim via-accent to-accent',
  secondary: 'from-secondary-dim via-secondary to-secondary',
  success: 'from-success/40 via-success to-success',
  warning: 'from-warning/40 via-warning to-warning',
  danger: 'from-danger/40 via-danger to-danger',
};

const TONE_GLOW: Record<MeterTone, string> = {
  accent: 'shadow-[0_0_10px_-1px_rgb(79_209_255/0.7)]',
  secondary: 'shadow-[0_0_10px_-1px_rgb(182_107_255/0.7)]',
  success: 'shadow-[0_0_10px_-1px_rgb(74_222_128/0.7)]',
  warning: 'shadow-[0_0_10px_-1px_rgb(251_191_36/0.7)]',
  danger: 'shadow-[0_0_10px_-1px_rgb(251_113_133/0.7)]',
};

const TRACK_SIZES: Record<NonNullable<MeterProps['size']>, string> = {
  sm: 'h-1',
  md: 'h-1.5',
};

/**
 * Horizontal bar for a normalised value. The fill is scaled with `transform`
 * rather than resized, so a per-frame telemetry readout never triggers layout.
 * The fill has square ends and the track clips it, which keeps the leading edge
 * crisp at every scale.
 */
export const Meter = React.forwardRef<HTMLDivElement, MeterProps>(function Meter(
  { value, label, valueLabel, tone = 'accent', size = 'md', ticks = 0, marker, className, ...props },
  ref,
) {
  const fraction = Number.isFinite(value) ? clamp(value, 0, 1) : 0;

  return (
    <div
      ref={ref}
      role="meter"
      aria-valuenow={fraction}
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuetext={typeof valueLabel === 'string' ? valueLabel : undefined}
      className={cn('flex min-w-0 flex-col gap-1', className)}
      {...props}
    >
      {(label !== undefined || valueLabel !== undefined) && (
        <div className="flex min-w-0 items-baseline gap-2">
          {label !== undefined && (
            <span className="truncate text-[10.5px] text-ink-muted">{label}</span>
          )}
          {valueLabel !== undefined && (
            <span className="nf-numeric ml-auto shrink-0 text-[10.5px] text-ink">{valueLabel}</span>
          )}
        </div>
      )}
      <div
        className={cn(
          'relative w-full overflow-hidden rounded-full bg-white/[0.07]',
          'shadow-[0_1px_0_0_rgb(0_0_0/0.35)_inset]',
          TRACK_SIZES[size],
        )}
      >
        <span
          aria-hidden
          className={cn(
            'absolute inset-0 origin-left bg-gradient-to-r will-change-transform',
            MOTION_BASE,
            TONE_FILL[tone],
            fraction > 0 && TONE_GLOW[tone],
          )}
          style={{ transform: `scaleX(${fraction})` }}
        />
        {ticks > 0 &&
          Array.from({ length: ticks - 1 }, (_unused, index) => (
            <span
              key={index}
              aria-hidden
              className="absolute inset-y-0 w-px bg-bg/60"
              style={{ left: `${((index + 1) / ticks) * 100}%` }}
            />
          ))}
        {marker !== undefined && Number.isFinite(marker) && (
          <span
            aria-hidden
            className="absolute inset-y-0 w-px bg-ink/70"
            style={{ left: `${clamp(marker, 0, 1) * 100}%` }}
          />
        )}
      </div>
    </div>
  );
});
