'use client';

import * as React from 'react';
import * as RadixSelect from '@radix-ui/react-select';
import { Check, ChevronDown, ChevronUp } from 'lucide-react';
import { cn } from './cn';
import { FOCUS_RING, MOTION_FAST, useRuntimeStyles } from './styles';
import { useFieldControl } from './Field';

export interface SelectProps
  extends Omit<React.ComponentPropsWithoutRef<typeof RadixSelect.Trigger>, 'onChange' | 'value'> {
  value: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  /** Options: `SelectItem` elements, optionally grouped with `SelectGroup`. */
  children: React.ReactNode;
  size?: 'sm' | 'md';
  /** Marks the underlying form control as required. */
  required?: boolean;
  /** Forwarded to the popup so a caller can widen or cap it. */
  contentClassName?: string;
  /** Rendered before the value inside the trigger. */
  icon?: React.ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  dir?: 'ltr' | 'rtl';
}

const TRIGGER_SIZES: Record<NonNullable<SelectProps['size']>, string> = {
  sm: 'h-6 px-1.5 text-[11px]',
  md: 'h-7 px-2 text-[12px]',
};

/** Styled wrapper around the Radix select. Options are `SelectItem` children. */
export const Select = React.forwardRef<HTMLButtonElement, SelectProps>(function Select(
  {
    value,
    onValueChange,
    placeholder = 'Select…',
    children,
    size = 'md',
    contentClassName,
    icon,
    open,
    defaultOpen,
    onOpenChange,
    dir,
    disabled,
    name,
    required,
    id,
    className,
    ...rest
  },
  ref,
) {
  useRuntimeStyles();
  const field = useFieldControl(id);
  const isDisabled = disabled === true || field.disabled;

  return (
    <RadixSelect.Root
      value={value}
      onValueChange={onValueChange}
      open={open}
      defaultOpen={defaultOpen}
      onOpenChange={onOpenChange}
      disabled={isDisabled}
      name={name}
      required={required}
      dir={dir}
    >
      <RadixSelect.Trigger
        {...rest}
        ref={ref}
        id={field.id}
        aria-describedby={rest['aria-describedby'] ?? field.describedBy}
        aria-invalid={field.invalid || undefined}
        className={cn(
          'group flex min-w-0 items-center gap-1.5 rounded-control border border-hairline',
          'bg-panel-raised text-ink shadow-[0_1px_0_0_rgb(255_255_255/0.03)_inset]',
          'data-[placeholder]:text-ink-faint disabled:pointer-events-none disabled:opacity-40',
          'hover:border-hairline-strong hover:bg-white/[0.04]',
          'data-[state=open]:border-accent/50',
          TRIGGER_SIZES[size],
          MOTION_FAST,
          FOCUS_RING,
          className,
        )}
      >
        {icon !== undefined && (
          <span className="flex shrink-0 items-center text-ink-muted">{icon}</span>
        )}
        <span className="min-w-0 flex-1 truncate text-left">
          <RadixSelect.Value placeholder={placeholder} />
        </span>
        <RadixSelect.Icon asChild>
          <ChevronDown
            size={12}
            aria-hidden
            className={cn(
              'shrink-0 text-ink-faint group-hover:text-ink-muted',
              MOTION_FAST,
              'group-data-[state=open]:rotate-180',
            )}
          />
        </RadixSelect.Icon>
      </RadixSelect.Trigger>

      <RadixSelect.Portal>
        <RadixSelect.Content
          position="popper"
          sideOffset={6}
          collisionPadding={8}
          className={cn(
            'nf-glass-raised nf-anim-pop z-50 overflow-hidden rounded-control',
            'max-h-[min(24rem,var(--radix-select-content-available-height))]',
            'min-w-[max(9rem,var(--radix-select-trigger-width))]',
            contentClassName,
          )}
        >
          <RadixSelect.ScrollUpButton className="flex h-5 items-center justify-center text-ink-faint">
            <ChevronUp size={12} aria-hidden />
          </RadixSelect.ScrollUpButton>
          <RadixSelect.Viewport className="p-1">{children}</RadixSelect.Viewport>
          <RadixSelect.ScrollDownButton className="flex h-5 items-center justify-center text-ink-faint">
            <ChevronDown size={12} aria-hidden />
          </RadixSelect.ScrollDownButton>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  );
});

export interface SelectItemProps
  extends React.ComponentPropsWithoutRef<typeof RadixSelect.Item> {
  /** Secondary line under the label; useful for model or receptor descriptions. */
  description?: string;
  icon?: React.ReactNode;
}

export const SelectItem = React.forwardRef<HTMLDivElement, SelectItemProps>(function SelectItem(
  { className, children, description, icon, ...props },
  ref,
) {
  return (
    <RadixSelect.Item
      ref={ref}
      className={cn(
        'relative flex cursor-default select-none items-center gap-2 rounded-[0.375rem] py-1.5 pl-2 pr-7',
        'text-[12px] text-ink-muted outline-none',
        'data-[highlighted]:bg-white/[0.07] data-[highlighted]:text-ink',
        'data-[state=checked]:text-ink',
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-40',
        MOTION_FAST,
        className,
      )}
      {...props}
    >
      {icon !== undefined && (
        <span className="flex shrink-0 items-center text-ink-faint">{icon}</span>
      )}
      <span className="flex min-w-0 flex-col">
        <RadixSelect.ItemText>{children}</RadixSelect.ItemText>
        {description !== undefined && (
          <span className="truncate text-[10.5px] leading-tight text-ink-faint">{description}</span>
        )}
      </span>
      <RadixSelect.ItemIndicator className="absolute right-2 flex items-center text-accent">
        <Check size={12} aria-hidden strokeWidth={2.75} />
      </RadixSelect.ItemIndicator>
    </RadixSelect.Item>
  );
});

export interface SelectGroupProps
  extends React.ComponentPropsWithoutRef<typeof RadixSelect.Group> {
  label?: React.ReactNode;
}

/** Optional labelled grouping inside a Select popup. */
export const SelectGroup = React.forwardRef<HTMLDivElement, SelectGroupProps>(
  function SelectGroup({ label, className, children, ...props }, ref) {
    return (
      <RadixSelect.Group ref={ref} className={cn('py-0.5', className)} {...props}>
        {label !== undefined && (
          <RadixSelect.Label className="px-2 pb-1 pt-1.5 text-[10px] font-semibold uppercase tracking-[0.09em] text-ink-faint">
            {label}
          </RadixSelect.Label>
        )}
        {children}
      </RadixSelect.Group>
    );
  },
);
