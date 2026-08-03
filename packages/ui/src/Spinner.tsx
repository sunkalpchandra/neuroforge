'use client';

import * as React from 'react';
import { cn } from './cn';

export interface SpinnerProps extends Omit<React.SVGAttributes<SVGSVGElement>, 'width' | 'height'> {
  /** Edge length in pixels. */
  size?: number;
  /** Announced to assistive tech; omit for a purely decorative spinner. */
  label?: string;
  strokeWidth?: number;
}

/**
 * Indeterminate progress. The arc is a dashed circle rotated with `animate-spin`,
 * so nothing but `transform` changes while it runs.
 */
export const Spinner = React.forwardRef<SVGSVGElement, SpinnerProps>(function Spinner(
  { size = 14, label, strokeWidth = 2, className, ...props },
  ref,
) {
  const radius = 8;
  const circumference = 2 * Math.PI * radius;

  return (
    <svg
      ref={ref}
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      role={label === undefined ? 'presentation' : 'status'}
      aria-label={label}
      aria-hidden={label === undefined ? true : undefined}
      className={cn('shrink-0 animate-spin', className)}
      {...props}
    >
      <circle
        cx="10"
        cy="10"
        r={radius}
        stroke="currentColor"
        strokeOpacity={0.18}
        strokeWidth={strokeWidth}
      />
      <circle
        cx="10"
        cy="10"
        r={radius}
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeDasharray={`${circumference * 0.28} ${circumference}`}
      />
    </svg>
  );
});
