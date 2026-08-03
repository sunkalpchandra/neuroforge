'use client';

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva } from 'class-variance-authority';
import type { VariantProps } from 'class-variance-authority';
import { cn } from './cn';
import { FOCUS_RING, MOTION_FAST } from './styles';
import { Spinner } from './Spinner';

const buttonVariants = cva(
  cn(
    'relative inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap',
    'rounded-control border font-medium tracking-[-0.005em] select-none',
    'active:scale-[0.97] disabled:pointer-events-none disabled:opacity-40',
    MOTION_FAST,
    FOCUS_RING,
  ),
  {
    variants: {
      variant: {
        primary: cn(
          'border-accent/35 bg-accent/14 text-accent',
          'shadow-[0_1px_0_0_rgb(255_255_255/0.06)_inset,0_10px_28px_-16px_rgb(79_209_255/0.75)]',
          'hover:border-accent/55 hover:bg-accent/22 hover:text-ink',
        ),
        secondary: cn(
          'border-hairline bg-panel-raised text-ink',
          'shadow-[0_1px_0_0_rgb(255_255_255/0.04)_inset]',
          'hover:border-hairline-strong hover:bg-white/[0.07]',
        ),
        ghost: cn(
          'border-transparent bg-transparent text-ink-muted',
          'hover:bg-white/[0.05] hover:text-ink',
        ),
        danger: cn(
          'border-danger/35 bg-danger/12 text-danger',
          'hover:border-danger/55 hover:bg-danger/20',
        ),
      },
      size: {
        sm: 'h-7 px-2.5 text-[11px]',
        md: 'h-8 px-3 text-[12px]',
        lg: 'h-10 px-4 text-[13px]',
      },
      iconOnly: {
        true: 'px-0',
        false: '',
      },
    },
    compoundVariants: [
      { iconOnly: true, size: 'sm', class: 'w-7' },
      { iconOnly: true, size: 'md', class: 'w-8' },
      { iconOnly: true, size: 'lg', class: 'w-10' },
    ],
    defaultVariants: {
      variant: 'secondary',
      size: 'md',
      iconOnly: false,
    },
  },
);

export type ButtonVariant = NonNullable<VariantProps<typeof buttonVariants>['variant']>;
export type ButtonSize = NonNullable<VariantProps<typeof buttonVariants>['size']>;

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  /** Render the child element instead of a `<button>`, forwarding all styling. */
  asChild?: boolean;
  /** Replaces the leading icon with a spinner and blocks interaction. */
  loading?: boolean;
  icon?: React.ReactNode;
  trailingIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    className,
    variant,
    size,
    iconOnly,
    asChild = false,
    loading = false,
    icon,
    trailingIcon,
    disabled,
    children,
    type,
    ...props
  },
  ref,
) {
  const Component = asChild ? Slot : 'button';
  const spinnerSize = size === 'lg' ? 16 : size === 'sm' ? 12 : 14;
  const leading = loading ? <Spinner size={spinnerSize} /> : icon;
  const isDisabled = disabled === true || loading;

  return (
    <Component
      ref={ref}
      type={asChild ? undefined : (type ?? 'button')}
      // `disabled` is not a valid attribute on the arbitrary element `asChild`
      // may render, so express the state with ARIA there instead.
      disabled={asChild ? undefined : isDisabled}
      aria-disabled={asChild && isDisabled ? true : undefined}
      data-disabled={isDisabled ? '' : undefined}
      aria-busy={loading || undefined}
      className={cn(buttonVariants({ variant, size, iconOnly }), className)}
      {...props}
    >
      {asChild ? (
        children
      ) : (
        <>
          {leading}
          {children}
          {trailingIcon}
        </>
      )}
    </Component>
  );
});

export interface IconButtonProps extends Omit<ButtonProps, 'iconOnly' | 'children' | 'icon'> {
  /** Accessible name; icon-only controls have no text to fall back on. */
  label: string;
  children: React.ReactNode;
}

/** A square, icon-only Button. Defaults to the ghost variant used across the chrome. */
export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton({ label, children, variant = 'ghost', className, ...props }, ref) {
    return (
      <Button
        ref={ref}
        variant={variant}
        iconOnly
        aria-label={label}
        className={className}
        {...props}
      >
        {children}
      </Button>
    );
  },
);
