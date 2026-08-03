'use client';

import * as React from 'react';
import * as RadixPopover from '@radix-ui/react-popover';
import { cn } from './cn';
import { useRuntimeStyles } from './styles';

export interface PopoverProps
  extends Omit<React.ComponentPropsWithoutRef<typeof RadixPopover.Content>, 'content'> {
  /** Element that opens the popover; wired with `asChild`. */
  trigger: React.ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Trap focus and block outside interaction, like a small dialog. */
  modal?: boolean;
  /** Positions relative to this element instead of the trigger. */
  anchor?: React.ReactNode;
  arrow?: boolean;
}

/** Floating surface anchored to a trigger, on the raised glass. */
export const Popover = React.forwardRef<HTMLDivElement, PopoverProps>(function Popover(
  {
    trigger,
    open,
    defaultOpen,
    onOpenChange,
    modal = false,
    anchor,
    arrow = false,
    side = 'bottom',
    align = 'start',
    sideOffset = 6,
    collisionPadding = 8,
    className,
    children,
    ...props
  },
  ref,
) {
  useRuntimeStyles();

  return (
    <RadixPopover.Root
      open={open}
      defaultOpen={defaultOpen}
      onOpenChange={onOpenChange}
      modal={modal}
    >
      <RadixPopover.Trigger asChild>{trigger}</RadixPopover.Trigger>
      {anchor !== undefined && <RadixPopover.Anchor asChild>{anchor}</RadixPopover.Anchor>}
      <RadixPopover.Portal>
        <RadixPopover.Content
          ref={ref}
          side={side}
          align={align}
          sideOffset={sideOffset}
          collisionPadding={collisionPadding}
          className={cn(
            'nf-glass-raised nf-anim-pop z-40 rounded-panel text-ink',
            'max-h-[var(--radix-popover-content-available-height)] focus:outline-none',
            className,
          )}
          {...props}
        >
          {children}
          {arrow && <RadixPopover.Arrow className="fill-panel-raised" width={10} height={5} />}
        </RadixPopover.Content>
      </RadixPopover.Portal>
    </RadixPopover.Root>
  );
});
