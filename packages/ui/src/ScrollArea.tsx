'use client';

import * as React from 'react';
import * as RadixScrollArea from '@radix-ui/react-scroll-area';
import { cn } from './cn';
import { MOTION_FAST } from './styles';

export interface ScrollAreaProps
  extends React.ComponentPropsWithoutRef<typeof RadixScrollArea.Root> {
  orientation?: 'vertical' | 'horizontal' | 'both';
  /** Classes for the scrolling viewport, where padding usually belongs. */
  viewportClassName?: string;
  /** Ref to the scrolling element, for programmatic scrolling and measurement. */
  viewportRef?: React.Ref<HTMLDivElement>;
}

function Scrollbar({
  orientation,
}: {
  orientation: 'vertical' | 'horizontal';
}): React.ReactElement {
  return (
    <RadixScrollArea.Scrollbar
      orientation={orientation}
      className={cn(
        'flex touch-none select-none p-0.5',
        MOTION_FAST,
        'data-[state=hidden]:opacity-0',
        orientation === 'vertical' ? 'w-2' : 'h-2 flex-col',
      )}
    >
      <RadixScrollArea.Thumb
        className={cn(
          'relative flex-1 rounded-full bg-white/15',
          MOTION_FAST,
          'hover:bg-white/25 active:bg-white/30',
        )}
      />
    </RadixScrollArea.Scrollbar>
  );
}

/**
 * Overlay scrollbars matching the panel language. The viewport is a real
 * scrolling element, so `scrollIntoView` and wheel behaviour are native.
 */
export const ScrollArea = React.forwardRef<HTMLDivElement, ScrollAreaProps>(function ScrollArea(
  { className, viewportClassName, viewportRef, orientation = 'vertical', children, ...props },
  ref,
) {
  return (
    <RadixScrollArea.Root
      ref={ref}
      scrollHideDelay={600}
      className={cn('relative min-h-0 overflow-hidden', className)}
      {...props}
    >
      <RadixScrollArea.Viewport
        ref={viewportRef}
        // Radix sets `display: table` inline on the inner wrapper, which defeats
        // width constraints; block is the only thing that has to be forced here.
        className={cn('size-full [&>div]:block!', viewportClassName)}
      >
        {children}
      </RadixScrollArea.Viewport>
      {orientation !== 'horizontal' && <Scrollbar orientation="vertical" />}
      {orientation !== 'vertical' && <Scrollbar orientation="horizontal" />}
      <RadixScrollArea.Corner />
    </RadixScrollArea.Root>
  );
});
