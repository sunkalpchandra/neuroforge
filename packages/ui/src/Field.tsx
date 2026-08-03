'use client';

import * as React from 'react';
import * as RadixLabel from '@radix-ui/react-label';
import { cn } from './cn';

interface FieldContextValue {
  controlId: string;
  descriptionId: string | undefined;
  invalid: boolean;
  disabled: boolean;
}

const FieldContext = React.createContext<FieldContextValue | null>(null);

/**
 * Lets a control adopt the id, description and disabled state of the Field that
 * wraps it, so `<Field label="Threshold"><NumberField …/></Field>` is correctly
 * associated without the caller wiring ids by hand.
 */
export function useFieldControl(explicitId?: string): {
  id: string;
  describedBy: string | undefined;
  invalid: boolean;
  disabled: boolean;
} {
  const field = React.useContext(FieldContext);
  const fallbackId = React.useId();
  return {
    id: explicitId ?? field?.controlId ?? fallbackId,
    describedBy: field?.descriptionId,
    invalid: field?.invalid ?? false,
    disabled: field?.disabled ?? false,
  };
}

export interface LabelProps extends React.ComponentPropsWithoutRef<typeof RadixLabel.Root> {
  /** Renders the muted asterisk used across the inspector for required inputs. */
  required?: boolean;
}

/** The small-caps control label used throughout the chrome. */
export const Label = React.forwardRef<HTMLLabelElement, LabelProps>(function Label(
  { className, required = false, children, ...props },
  ref,
) {
  return (
    <RadixLabel.Root
      ref={ref}
      className={cn(
        'select-none text-[11px] font-medium leading-tight tracking-[-0.005em] text-ink-muted',
        'peer-disabled:opacity-40',
        className,
      )}
      {...props}
    >
      {children}
      {required && (
        <span aria-hidden className="ml-0.5 text-accent/70">
          *
        </span>
      )}
    </RadixLabel.Root>
  );
});

export interface FieldProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: React.ReactNode;
  /** Helper text rendered under the control and wired up via aria-describedby. */
  description?: React.ReactNode;
  /** Replaces `description` and turns the field red when set. */
  error?: React.ReactNode;
  /** Trailing content on the label row — a readout, a reset affordance. */
  aside?: React.ReactNode;
  required?: boolean;
  disabled?: boolean;
  /** `row` puts the label and the control on one line, which is the inspector default. */
  orientation?: 'row' | 'column';
  /** Explicit id for the control; generated when omitted. */
  htmlFor?: string;
}

/** Label + control + description, with the accessibility wiring done once. */
export const Field = React.forwardRef<HTMLDivElement, FieldProps>(function Field(
  {
    label,
    description,
    error,
    aside,
    required = false,
    disabled = false,
    orientation = 'row',
    htmlFor,
    className,
    children,
    ...props
  },
  ref,
) {
  const generatedId = React.useId();
  const controlId = htmlFor ?? generatedId;
  const descriptionId = `${controlId}-description`;
  const hasHint = error !== undefined || description !== undefined;

  const context = React.useMemo<FieldContextValue>(
    () => ({
      controlId,
      descriptionId: hasHint ? descriptionId : undefined,
      invalid: error !== undefined,
      disabled,
    }),
    [controlId, descriptionId, hasHint, error, disabled],
  );

  return (
    <FieldContext.Provider value={context}>
      <div
        ref={ref}
        data-disabled={disabled ? '' : undefined}
        className={cn(
          'flex min-w-0',
          orientation === 'row' ? 'flex-row items-center gap-3' : 'flex-col gap-1.5',
          disabled && 'pointer-events-none opacity-45',
          className,
        )}
        {...props}
      >
        {(label !== undefined || aside !== undefined) && (
          <div
            className={cn(
              'flex min-w-0 items-center gap-1.5',
              orientation === 'row' ? 'w-[38%] shrink-0' : 'w-full',
            )}
          >
            {label !== undefined && (
              <Label htmlFor={controlId} required={required} className="truncate">
                {label}
              </Label>
            )}
            {aside !== undefined && (
              <div className="ml-auto flex shrink-0 items-center gap-1">{aside}</div>
            )}
          </div>
        )}
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          {children}
          {hasHint && (
            <span
              id={descriptionId}
              className={cn(
                'text-[10.5px] leading-snug',
                error !== undefined ? 'text-danger' : 'text-ink-faint',
              )}
            >
              {error ?? description}
            </span>
          )}
        </div>
      </div>
    </FieldContext.Provider>
  );
});
