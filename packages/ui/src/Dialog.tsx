'use client';

import * as React from 'react';
import * as RadixDialog from '@radix-ui/react-dialog';
import { X } from 'lucide-react';
import { cn } from './cn';
import { useRuntimeStyles } from './styles';
import { IconButton } from './Button';

export interface DialogProps {
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Required: the accessible name of the dialog. */
  title: React.ReactNode;
  /** Visible and announced; supply one unless the body is self-explanatory. */
  description?: React.ReactNode;
  children?: React.ReactNode;
  /** Action row pinned to the bottom, above the surface edge. */
  footer?: React.ReactNode;
  /** Element that opens the dialog; wired with `asChild`. */
  trigger?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showClose?: boolean;
  /** Prevents dismissal by Escape or an outside click, for destructive flows. */
  modalLock?: boolean;
  className?: string;
  overlayClassName?: string;
}

const DIALOG_SIZES: Record<NonNullable<DialogProps['size']>, string> = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
};

/**
 * Modal dialog. The backdrop blurs the scene behind it and the card scales in
 * with an expo-out curve; the centring is done by a full-viewport grid so the
 * animation only ever touches `transform` and `opacity`.
 */
export const Dialog = React.forwardRef<HTMLDivElement, DialogProps>(function Dialog(
  {
    open,
    defaultOpen,
    onOpenChange,
    title,
    description,
    children,
    footer,
    trigger,
    size = 'md',
    showClose = true,
    modalLock = false,
    className,
    overlayClassName,
  },
  ref,
) {
  useRuntimeStyles();
  const blockDismiss = (event: Event): void => {
    if (modalLock) event.preventDefault();
  };

  return (
    <RadixDialog.Root open={open} defaultOpen={defaultOpen} onOpenChange={onOpenChange}>
      {trigger !== undefined && <RadixDialog.Trigger asChild>{trigger}</RadixDialog.Trigger>}
      <RadixDialog.Portal>
        <RadixDialog.Overlay
          className={cn(
            'nf-anim-overlay fixed inset-0 z-50 bg-bg/70 backdrop-blur-md backdrop-saturate-150',
            overlayClassName,
          )}
        />
        <RadixDialog.Content
          ref={ref}
          onEscapeKeyDown={blockDismiss}
          onPointerDownOutside={blockDismiss}
          onInteractOutside={blockDismiss}
          // Radix points `aria-describedby` at its own Description; clear it when
          // we render none so the reference cannot dangle.
          {...(description === undefined ? { 'aria-describedby': undefined } : null)}
          // The Content fills the viewport purely to centre the card. Radix sets
          // `pointer-events: auto` inline on the layer, so the pass-through has to
          // be an inline style too, otherwise backdrop clicks would land here
          // instead of on the Overlay and the dialog would never dismiss.
          style={{ pointerEvents: 'none' }}
          className="nf-anim-dialog fixed inset-0 z-50 grid place-items-center p-6 focus:outline-none"
        >
          <div
            className={cn(
              'nf-glass-raised pointer-events-auto flex max-h-[min(85vh,48rem)] w-full',
              'flex-col overflow-hidden rounded-panel text-ink',
              DIALOG_SIZES[size],
              className,
            )}
          >
            <div className="flex shrink-0 items-start gap-3 border-b border-hairline px-4 py-3">
              <div className="flex min-w-0 flex-col gap-0.5">
                <RadixDialog.Title className="truncate text-[13px] font-semibold tracking-[-0.014em] text-ink">
                  {title}
                </RadixDialog.Title>
                {description !== undefined && (
                  <RadixDialog.Description className="text-[11.5px] leading-snug text-ink-faint">
                    {description}
                  </RadixDialog.Description>
                )}
              </div>
              {showClose && (
                <RadixDialog.Close asChild>
                  <IconButton label="Close dialog" size="sm" className="ml-auto shrink-0">
                    <X size={13} aria-hidden />
                  </IconButton>
                </RadixDialog.Close>
              )}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3.5 text-[12px] leading-relaxed text-ink-muted">
              {children}
            </div>

            {footer !== undefined && (
              <div className="flex shrink-0 items-center justify-end gap-2 border-t border-hairline px-4 py-3">
                {footer}
              </div>
            )}
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
});
