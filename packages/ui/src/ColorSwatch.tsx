'use client';

import * as React from 'react';
import { Check } from 'lucide-react';
import { cn } from './cn';
import { FOCUS_RING, MOTION_FAST, useRuntimeStyles } from './styles';

export interface ColorSwatchProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'onSelect' | 'color'> {
  /** Any CSS colour; alpha is revealed against a checkerboard. */
  color: string;
  size?: 'sm' | 'md' | 'lg';
  selected?: boolean;
  /** Makes the swatch a button. Without it the swatch renders as a static chip. */
  onSelect?: (color: string) => void;
  /** Accessible name; defaults to the colour value. */
  label?: string;
}

const SWATCH_SIZES: Record<NonNullable<ColorSwatchProps['size']>, string> = {
  sm: 'size-4',
  md: 'size-5',
  lg: 'size-7',
};

const CHECK_SIZES: Record<NonNullable<ColorSwatchProps['size']>, number> = {
  sm: 9,
  md: 11,
  lg: 14,
};

/** A colour chip, selectable when `onSelect` is supplied. */
export const ColorSwatch = React.forwardRef<HTMLButtonElement, ColorSwatchProps>(
  function ColorSwatch(
    { color, size = 'md', selected = false, onSelect, label, className, disabled, ...props },
    ref,
  ) {
    useRuntimeStyles();
    const interactive = onSelect !== undefined;

    return (
      <button
        ref={ref}
        type="button"
        role={interactive ? 'radio' : 'img'}
        aria-checked={interactive ? selected : undefined}
        aria-label={label ?? color}
        title={label ?? color}
        disabled={disabled === true || !interactive}
        onClick={interactive ? () => onSelect(color) : undefined}
        className={cn(
          'nf-checker relative shrink-0 overflow-hidden rounded-[0.3rem] border border-hairline-strong',
          SWATCH_SIZES[size],
          MOTION_FAST,
          interactive && 'hover:scale-110 active:scale-95',
          interactive ? FOCUS_RING : 'cursor-default',
          selected && 'ring-2 ring-accent ring-offset-2 ring-offset-bg',
          disabled === true && 'pointer-events-none opacity-40',
          className,
        )}
        {...props}
      >
        <span aria-hidden className="absolute inset-0" style={{ backgroundColor: color }} />
        {selected && (
          <span className="absolute inset-0 flex items-center justify-center">
            <Check
              size={CHECK_SIZES[size]}
              strokeWidth={3}
              aria-hidden
              className="text-bg drop-shadow-[0_0_2px_rgb(255_255_255/0.6)]"
            />
          </span>
        )}
      </button>
    );
  },
);
