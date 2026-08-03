'use client';

import * as React from 'react';
import { cn } from './cn';
import { clamp } from './numeric';

export interface SparklineProps
  extends Omit<React.SVGAttributes<SVGSVGElement>, 'values' | 'min' | 'max' | 'fill' | 'color'> {
  /** Sample buffer. May be a ring buffer that is only partially filled. */
  values: Float32Array;
  /** Valid sample count. Defaults to the whole array. */
  count?: number;
  /**
   * Ring-buffer write cursor, counted in total writes since reset. Supplying it
   * reorders the samples oldest-first; omit it for a plain left-to-right array.
   */
  head?: number;
  /** Aspect basis of the viewBox; the element itself stretches to its container. */
  width?: number;
  height?: number;
  /** Pin the value axis. Either bound may be given on its own. */
  min?: number;
  max?: number;
  /** Dashed reference line in data units, e.g. a spike threshold. */
  threshold?: number;
  /** Value the gradient fill is anchored to. Defaults to the bottom of the plot. */
  baseline?: number;
  /** Stroke and gradient colour. Defaults to the element's `currentColor`. */
  color?: string;
  fill?: boolean;
  strokeWidth?: number;
  /** Dot on the most recent sample. */
  endpoint?: boolean;
  /** Accessible name. Without it the chart is exposed as decorative. */
  label?: string;
  /** Inset in viewBox units so the stroke and endpoint are never clipped. */
  padding?: number;
}

interface Extent {
  lo: number;
  hi: number;
}

/** Value range with a little headroom, so a flat trace still has a sensible axis. */
function resolveExtent(
  values: Float32Array,
  indexOf: (i: number) => number,
  count: number,
  min: number | undefined,
  max: number | undefined,
  threshold: number | undefined,
): Extent {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i += 1) {
    const sample = values[indexOf(i)];
    if (!Number.isFinite(sample)) continue;
    if (sample < lo) lo = sample;
    if (sample > hi) hi = sample;
  }
  if (threshold !== undefined && Number.isFinite(threshold) && min === undefined && max === undefined) {
    if (threshold < lo) lo = threshold;
    if (threshold > hi) hi = threshold;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  if (hi === lo) {
    const pad = Math.max(Math.abs(hi) * 0.05, 0.5);
    lo -= pad;
    hi += pad;
  } else {
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
  }
  let resolvedLo = min !== undefined && Number.isFinite(min) ? min : lo;
  let resolvedHi = max !== undefined && Number.isFinite(max) ? max : hi;
  // A pinned bound can invert or collapse the axis — `min` above every sample,
  // or `min === max`. Normalise here so the caller downstream can divide by the
  // span unconditionally instead of plotting an upside-down or zero-height trace.
  if (resolvedHi < resolvedLo) {
    const swap = resolvedLo;
    resolvedLo = resolvedHi;
    resolvedHi = swap;
  }
  if (resolvedHi === resolvedLo) {
    const pad = Math.max(Math.abs(resolvedHi) * 0.05, 0.5);
    resolvedLo -= pad;
    resolvedHi += pad;
  }
  return { lo: resolvedLo, hi: resolvedHi };
}

/**
 * Inline SVG trace for a simulation probe.
 *
 * Deliberately not memoised: the caller re-renders with the same Float32Array
 * whose contents mutate in place, so any cache keyed on identity would go stale.
 * Building the path is a few hundred arithmetic operations, which is far cheaper
 * than the render that already decided to call it.
 */
export const Sparkline = React.forwardRef<SVGSVGElement, SparklineProps>(function Sparkline(
  {
    values,
    count,
    head,
    width = 120,
    height = 32,
    min,
    max,
    threshold,
    baseline,
    color,
    fill = true,
    strokeWidth = 1.25,
    endpoint = true,
    label,
    padding = 1.5,
    className,
    ...props
  },
  ref,
) {
  const rawId = React.useId();
  const gradientId = `nf-spark-${rawId.replace(/[^a-zA-Z0-9]/g, '')}`;

  const capacity = values.length;
  const requested = count ?? capacity;
  const total = capacity === 0 ? 0 : Math.max(0, Math.min(requested, capacity));
  // A ring buffer's oldest sample sits `total` writes behind the cursor.
  const start =
    head === undefined || capacity === 0 ? 0 : (((head - total) % capacity) + capacity) % capacity;
  const indexOf = (i: number): number => (capacity === 0 ? 0 : (start + i) % capacity);

  const extent = resolveExtent(values, indexOf, total, min, max, threshold);
  const span = extent.hi - extent.lo;
  const innerWidth = Math.max(width - padding * 2, 1);
  const innerHeight = Math.max(height - padding * 2, 1);
  const bottom = height - padding;

  const toX = (i: number): number =>
    total <= 1 ? padding + innerWidth * 0.5 : padding + (i / (total - 1)) * innerWidth;
  // Samples outside an explicitly pinned axis saturate at its edge; without the
  // clamp they would be drawn outside the box, which `overflow-visible` (needed
  // so the stroke is not shaved) would happily render on top of neighbouring UI.
  const toY = (value: number): number =>
    span === 0
      ? bottom
      : clamp(bottom - ((value - extent.lo) / span) * innerHeight, padding, bottom);

  const anchorY = baseline !== undefined && Number.isFinite(baseline) ? toY(baseline) : bottom;

  const strokeSegments: string[] = [];
  const fillSegments: string[] = [];
  let runStart = -1;
  let lastX = 0;
  let lastY = 0;
  let hasEndpoint = false;

  const flushRun = (endIndex: number): void => {
    if (runStart < 0) return;
    const points: string[] = [];
    for (let i = runStart; i <= endIndex; i += 1) {
      points.push(`${toX(i).toFixed(2)} ${toY(values[indexOf(i)]).toFixed(2)}`);
    }
    // A single-sample run is emitted as a zero-length line so the round cap
    // renders it as a dot rather than disappearing.
    const body = points.length === 1 ? `M${points[0]}L${points[0]}` : `M${points.join('L')}`;
    strokeSegments.push(body);
    if (fill) {
      const first = toX(runStart).toFixed(2);
      const last = toX(endIndex).toFixed(2);
      const base = anchorY.toFixed(2);
      fillSegments.push(`M${first} ${base}L${points.join('L')}L${last} ${base}Z`);
    }
    runStart = -1;
  };

  for (let i = 0; i < total; i += 1) {
    const sample = values[indexOf(i)];
    if (Number.isFinite(sample)) {
      if (runStart < 0) runStart = i;
      lastX = toX(i);
      lastY = toY(sample);
      hasEndpoint = true;
    } else {
      flushRun(i - 1);
    }
  }
  flushRun(total - 1);

  const stroke = color ?? 'currentColor';
  // Only drawn when it genuinely falls inside the axis. Saturating it at an edge
  // the way samples are saturated would assert that the threshold sits at the top
  // or bottom of the range, which is the opposite of what a reference line means.
  const thresholdY =
    threshold !== undefined &&
    Number.isFinite(threshold) &&
    threshold >= extent.lo &&
    threshold <= extent.hi
      ? toY(threshold)
      : undefined;

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      width="100%"
      height={height}
      role={label === undefined ? 'presentation' : 'img'}
      aria-label={label}
      aria-hidden={label === undefined ? true : undefined}
      className={cn('block overflow-visible text-accent', className)}
      {...props}
    >
      {fill && (
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={stroke} stopOpacity={0.32} />
            <stop offset="65%" stopColor={stroke} stopOpacity={0.08} />
            <stop offset="100%" stopColor={stroke} stopOpacity={0} />
          </linearGradient>
        </defs>
      )}

      {thresholdY !== undefined && (
        <line
          x1={padding}
          x2={width - padding}
          y1={thresholdY}
          y2={thresholdY}
          stroke="currentColor"
          strokeOpacity={0.28}
          strokeWidth={1}
          strokeDasharray="3 3"
          vectorEffect="non-scaling-stroke"
        />
      )}

      {fill &&
        fillSegments.map((path, index) => (
          <path key={`fill-${index}`} d={path} fill={`url(#${gradientId})`} stroke="none" />
        ))}

      {strokeSegments.map((path, index) => (
        <path
          key={`line-${index}`}
          d={path}
          fill="none"
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      ))}

      {/*
        The endpoint is a zero-length round-capped stroke rather than a circle:
        the viewBox is stretched horizontally to fill its container, which would
        smear a real circle into an ellipse, whereas a non-scaling stroke stays
        round at any aspect ratio.
      */}
      {endpoint && hasEndpoint && (
        <path
          d={`M${lastX.toFixed(2)} ${lastY.toFixed(2)}L${lastX.toFixed(2)} ${lastY.toFixed(2)}`}
          fill="none"
          stroke={stroke}
          strokeWidth={strokeWidth * 2.6}
          strokeLinecap="round"
          strokeOpacity={0.95}
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
});
