'use client';

import * as React from 'react';
import { cva } from 'class-variance-authority';
import type { VariantProps } from 'class-variance-authority';
import { cn } from './cn';

const badgeVariants = cva(
  'inline-flex shrink-0 items-center gap-1 rounded-full border font-medium leading-none whitespace-nowrap',
  {
    variants: {
      variant: {
        neutral: 'border-hairline-strong bg-white/[0.06] text-ink-muted',
        accent: 'border-accent/30 bg-accent/12 text-accent',
        secondary: 'border-secondary/30 bg-secondary/12 text-secondary',
        success: 'border-success/30 bg-success/12 text-success',
        warning: 'border-warning/30 bg-warning/12 text-warning',
        danger: 'border-danger/30 bg-danger/12 text-danger',
        outline: 'border-hairline-strong bg-transparent text-ink-faint',
      },
      size: {
        sm: 'h-4 px-1.5 text-[9.5px]',
        md: 'h-5 px-2 text-[10.5px]',
      },
    },
    defaultVariants: { variant: 'neutral', size: 'md' },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  /** Leading status dot tinted with the variant colour. */
  dot?: boolean;
  /** Renders numbers with tabular figures so a live counter does not jitter. */
  numeric?: boolean;
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(function Badge(
  { className, variant, size, dot = false, numeric = false, children, ...props },
  ref,
) {
  return (
    <span
      ref={ref}
      className={cn(badgeVariants({ variant, size }), numeric && 'nf-numeric', className)}
      {...props}
    >
      {dot && (
        <span aria-hidden className="size-1 shrink-0 rounded-full bg-current shadow-[0_0_6px_currentColor]" />
      )}
      {children}
    </span>
  );
});
