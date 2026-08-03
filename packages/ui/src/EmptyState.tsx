'use client';

import * as React from 'react';
import { cn } from './cn';

export interface EmptyStateProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  title: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  /** Primary affordance, typically a Button. */
  action?: React.ReactNode;
  /** Tighter layout for use inside a narrow inspector column. */
  compact?: boolean;
}

/** Placeholder for a panel with nothing to show yet. */
export const EmptyState = React.forwardRef<HTMLDivElement, EmptyStateProps>(function EmptyState(
  { title, description, icon, action, compact = false, className, children, ...props },
  ref,
) {
  return (
    <div
      ref={ref}
      className={cn(
        'flex flex-col items-center justify-center text-center',
        compact ? 'gap-1.5 px-4 py-6' : 'gap-2.5 px-6 py-12',
        className,
      )}
      {...props}
    >
      {icon !== undefined && (
        <span
          aria-hidden
          className={cn(
            'flex items-center justify-center rounded-full border border-hairline',
            'bg-white/[0.03] text-ink-faint',
            compact ? 'mb-0.5 size-8' : 'mb-1 size-11',
          )}
        >
          {icon}
        </span>
      )}
      <span
        className={cn(
          'font-medium tracking-[-0.01em] text-ink',
          compact ? 'text-[12px]' : 'text-[13px]',
        )}
      >
        {title}
      </span>
      {description !== undefined && (
        <span
          className={cn(
            'max-w-[36ch] leading-relaxed text-ink-faint',
            compact ? 'text-[10.5px]' : 'text-[11.5px]',
          )}
        >
          {description}
        </span>
      )}
      {children}
      {action !== undefined && <div className="mt-1.5 flex items-center gap-2">{action}</div>}
    </div>
  );
});
