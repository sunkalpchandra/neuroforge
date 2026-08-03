'use client';

import * as React from 'react';
import * as RadixSwitch from '@radix-ui/react-switch';
import { cn } from './cn';
import { FOCUS_RING, MOTION_BASE } from './styles';
import { useFieldControl } from './Field';

export interface SwitchProps
  extends Omit<React.ComponentPropsWithoutRef<typeof RadixSwitch.Root>, 'asChild'> {
  size?: 'sm' | 'md';
}

const TRACK_SIZES: Record<NonNullable<SwitchProps['size']>, string> = {
  sm: 'h-3.5 w-6 p-[2px]',
  md: 'h-4.5 w-8 p-[2px]',
};

const THUMB_SIZES: Record<NonNullable<SwitchProps['size']>, string> = {
  sm: 'size-2.5 data-[state=checked]:translate-x-2.5',
  md: 'size-3.5 data-[state=checked]:translate-x-3.5',
};

/** Binary toggle. The thumb travels with `transform`, never with layout. */
export const Switch = React.forwardRef<HTMLButtonElement, SwitchProps>(function Switch(
  { size = 'md', className, disabled, id, ...props },
  ref,
) {
  const field = useFieldControl(id);
  const isDisabled = disabled === true || field.disabled;

  return (
    <RadixSwitch.Root
      {...props}
      ref={ref}
      id={field.id}
      disabled={isDisabled}
      aria-describedby={props['aria-describedby'] ?? field.describedBy}
      className={cn(
        'group inline-flex shrink-0 items-center rounded-full border border-hairline',
        'bg-white/[0.06] shadow-[0_1px_0_0_rgb(0_0_0/0.4)_inset]',
        'data-[state=checked]:border-accent/50 data-[state=checked]:bg-accent/25',
        'disabled:pointer-events-none disabled:opacity-40',
        TRACK_SIZES[size],
        MOTION_BASE,
        FOCUS_RING,
        className,
      )}
    >
      <RadixSwitch.Thumb
        className={cn(
          'block rounded-full bg-ink-muted shadow-[0_1px_2px_rgb(0_0_0/0.5)]',
          'translate-x-0 will-change-transform',
          'group-data-[state=checked]:bg-accent group-data-[state=checked]:shadow-[0_0_8px_rgb(79_209_255/0.6)]',
          THUMB_SIZES[size],
          MOTION_BASE,
        )}
      />
    </RadixSwitch.Root>
  );
});
