'use client';

import * as React from 'react';
import * as RadixSeparator from '@radix-ui/react-separator';
import { cn } from './cn';

export interface SeparatorProps
  extends React.ComponentPropsWithoutRef<typeof RadixSeparator.Root> {
  /** Caption centred on a horizontal rule; ignored when vertical. */
  label?: React.ReactNode;
}

export const Separator = React.forwardRef<HTMLDivElement, SeparatorProps>(function Separator(
  { className, orientation = 'horizontal', decorative = true, label, ...props },
  ref,
) {
  if (label !== undefined && orientation === 'horizontal') {
    return (
      <div className={cn('flex w-full items-center gap-2', className)}>
        <RadixSeparator.Root
          ref={ref}
          orientation="horizontal"
          decorative={decorative}
          className="h-px flex-1 bg-hairline"
          {...props}
        />
        <span className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.09em] text-ink-faint">
          {label}
        </span>
        <span aria-hidden className="h-px flex-1 bg-hairline" />
      </div>
    );
  }

  return (
    <RadixSeparator.Root
      ref={ref}
      orientation={orientation}
      decorative={decorative}
      className={cn(
        'shrink-0 bg-hairline',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className,
      )}
      {...props}
    />
  );
});
