'use client';

import * as React from 'react';
import { CircleAlert, CircleCheck, CircleX, Info, TriangleAlert, X } from 'lucide-react';
import type { LucideProps } from 'lucide-react';
import { cn } from './cn';
import { MOTION_BASE, useRuntimeStyles } from './styles';
import { useEventCallback } from './hooks';
import { IconButton } from './Button';
import {
  dismissToast,
  getToastServerSnapshot,
  getToastSnapshot,
  subscribeToToasts,
} from './toast-store';
import type { ToastAction, ToastTone } from './toast-store';

/** Pointer travel that commits a swipe dismissal. */
const SWIPE_DISMISS_PX = 72;
/** Travel before the gesture is treated as a swipe rather than a click. */
const SWIPE_THRESHOLD_PX = 4;
/** Must match the exit transition in MOTION_BASE. */
const EXIT_DURATION_MS = 170;

const TONE_ICONS: Record<ToastTone, React.ComponentType<LucideProps>> = {
  neutral: Info,
  accent: CircleAlert,
  success: CircleCheck,
  warning: TriangleAlert,
  danger: CircleX,
};

const TONE_ACCENTS: Record<ToastTone, string> = {
  neutral: 'text-ink-muted',
  accent: 'text-accent',
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
};

const TONE_RAILS: Record<ToastTone, string> = {
  neutral: 'bg-ink-faint',
  accent: 'bg-accent shadow-[0_0_10px_rgb(79_209_255/0.8)]',
  success: 'bg-success shadow-[0_0_10px_rgb(74_222_128/0.8)]',
  warning: 'bg-warning shadow-[0_0_10px_rgb(251_191_36/0.8)]',
  danger: 'bg-danger shadow-[0_0_10px_rgb(251_113_133/0.8)]',
};

export interface ToastProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title' | 'action'> {
  title: React.ReactNode;
  description?: React.ReactNode;
  tone?: ToastTone;
  action?: ToastAction;
  /** Milliseconds until auto-dismiss. `0` or `Infinity` pins the toast open. */
  duration?: number;
  onDismiss?: () => void;
  /** Disables the horizontal swipe gesture. */
  swipeable?: boolean;
}

/**
 * One notification card. Owns its dismissal timer — which pauses while the
 * pointer or keyboard focus is on it — and its swipe gesture, so it behaves the
 * same whether it comes from the store or is rendered directly.
 */
export const Toast = React.forwardRef<HTMLDivElement, ToastProps>(function Toast(
  {
    title,
    description,
    tone = 'neutral',
    action,
    duration = 5000,
    onDismiss,
    swipeable = true,
    className,
    children,
    ...props
  },
  ref,
) {
  useRuntimeStyles();

  const [paused, setPaused] = React.useState(false);
  const [offset, setOffset] = React.useState(0);
  const [swiping, setSwiping] = React.useState(false);
  const [exiting, setExiting] = React.useState(false);

  const exitingRef = React.useRef(false);
  const remainingRef = React.useRef(duration);
  const dragRef = React.useRef<{ pointerId: number; startX: number } | null>(null);
  // The gesture is decided from refs, not state: pointermove is a continuous
  // event that React may batch, so the value read on pointerup could otherwise
  // lag behind the last movement of a fast flick.
  const offsetRef = React.useRef(0);
  const swipingRef = React.useRef(false);
  const suppressClickRef = React.useRef(false);

  const Icon = TONE_ICONS[tone];

  const setSwipeOffset = useEventCallback((next: number) => {
    offsetRef.current = next;
    setOffset(next);
  });

  const beginExit = useEventCallback((viaSwipe: boolean) => {
    if (exitingRef.current) return;
    exitingRef.current = true;
    swipingRef.current = false;
    setExiting(true);
    setSwiping(false);
    if (viaSwipe) setSwipeOffset(SWIPE_DISMISS_PX * 4);
    setTimeout(() => onDismiss?.(), EXIT_DURATION_MS);
  });

  React.useEffect(() => {
    remainingRef.current = duration;
  }, [duration]);

  React.useEffect(() => {
    if (paused || exiting) return;
    if (!Number.isFinite(duration) || duration <= 0) return;
    const startedAt = Date.now();
    const timer = setTimeout(() => beginExit(false), remainingRef.current);
    return () => {
      clearTimeout(timer);
      remainingRef.current = Math.max(0, remainingRef.current - (Date.now() - startedAt));
    };
  }, [beginExit, duration, exiting, paused]);

  const isInteractiveTarget = (target: EventTarget | null): boolean =>
    target instanceof globalThis.Element && target.closest('button,a,input,textarea') !== null;

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!swipeable || exiting || event.button !== 0) return;
    if (isInteractiveTarget(event.target)) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, startX: event.clientX };
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag === null || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.startX;
    if (!swipingRef.current && Math.abs(dx) < SWIPE_THRESHOLD_PX) return;
    swipingRef.current = true;
    setSwiping(true);
    setSwipeOffset(Math.max(0, dx));
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag === null || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const swiped = swipingRef.current;
    swipingRef.current = false;
    // A released swipe is followed by a click event; it must not also dismiss.
    suppressClickRef.current = swiped;
    setSwiping(false);
    if (swiped && offsetRef.current >= SWIPE_DISMISS_PX) beginExit(true);
    else setSwipeOffset(0);
  };

  const handleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (exitingRef.current) return;
    if (isInteractiveTarget(event.target)) return;
    beginExit(false);
  };

  const transform = exiting
    ? offset > 0
      ? `translate3d(${offset}px,0,0)`
      : 'translate3d(0,8px,0) scale(0.97)'
    : `translate3d(${offset}px,0,0)`;

  return (
    <div
      ref={ref}
      role="status"
      data-tone={tone}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onLostPointerCapture={handlePointerUp}
      onPointerEnter={() => setPaused(true)}
      onPointerLeave={() => setPaused(false)}
      onFocusCapture={() => setPaused(true)}
      onBlurCapture={() => setPaused(false)}
      onClick={handleClick}
      style={{
        transform,
        opacity: exiting ? 0 : Math.max(0, 1 - offset / 200),
        // Tracking the finger must be immediate; the spring-back and the exit
        // are what the transition class is for.
        transition: swiping ? 'none' : undefined,
      }}
      className={cn(
        'nf-glass-raised nf-anim-toast-in pointer-events-auto relative flex w-80 max-w-[calc(100vw-2rem)]',
        'items-start gap-2.5 overflow-hidden rounded-panel py-2.5 pl-3.5 pr-2 text-ink',
        swipeable && !exiting && 'cursor-grab active:cursor-grabbing',
        MOTION_BASE,
        className,
      )}
      {...props}
    >
      <span
        aria-hidden
        className={cn('absolute inset-y-2 left-0 w-0.5 rounded-full', TONE_RAILS[tone])}
      />
      <Icon size={14} aria-hidden className={cn('mt-px shrink-0', TONE_ACCENTS[tone])} />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="text-[12px] font-medium leading-snug tracking-[-0.008em] text-ink">
          {title}
        </span>
        {description !== undefined && (
          <span className="text-[11px] leading-snug text-ink-muted">{description}</span>
        )}
        {children}
        {action !== undefined && (
          <button
            type="button"
            onClick={() => {
              action.onClick();
              beginExit(false);
            }}
            className={cn(
              'mt-1.5 self-start rounded px-1.5 py-0.5 text-[11px] font-medium text-accent',
              'outline-none hover:bg-accent/12 focus-visible:ring-2 focus-visible:ring-accent/60',
              MOTION_BASE,
            )}
          >
            {action.label}
          </button>
        )}
      </div>
      <IconButton
        label="Dismiss notification"
        size="sm"
        className="-mr-0.5 shrink-0"
        onClick={() => beginExit(false)}
      >
        <X size={12} aria-hidden />
      </IconButton>
    </div>
  );
});

export type ToastPosition =
  | 'top-left'
  | 'top-center'
  | 'top-right'
  | 'bottom-left'
  | 'bottom-center'
  | 'bottom-right';

export interface ToastViewportProps extends React.HTMLAttributes<HTMLDivElement> {
  position?: ToastPosition;
  /** Cap on simultaneously rendered toasts. */
  limit?: number;
}

const POSITION_CLASSES: Record<ToastPosition, string> = {
  'top-left': 'top-3 left-3 items-start flex-col',
  'top-center': 'top-3 left-1/2 -translate-x-1/2 items-center flex-col',
  'top-right': 'top-3 right-3 items-end flex-col',
  'bottom-left': 'bottom-3 left-3 items-start flex-col-reverse',
  'bottom-center': 'bottom-3 left-1/2 -translate-x-1/2 items-center flex-col-reverse',
  'bottom-right': 'bottom-3 right-3 items-end flex-col-reverse',
};

/**
 * Renders the notification stack. Mount exactly one of these near the root; the
 * store is a module singleton, so `pushToast` works from anywhere afterwards.
 */
export const ToastViewport = React.forwardRef<HTMLDivElement, ToastViewportProps>(
  function ToastViewport({ position = 'bottom-right', limit = 5, className, ...props }, ref) {
    const toasts = React.useSyncExternalStore(
      subscribeToToasts,
      getToastSnapshot,
      getToastServerSnapshot,
    );
    const visible = limit >= toasts.length ? toasts : toasts.slice(0, limit);

    return (
      <div
        ref={ref}
        role="region"
        aria-label="Notifications"
        className={cn(
          'pointer-events-none fixed z-[60] flex gap-2',
          POSITION_CLASSES[position],
          className,
        )}
        {...props}
      >
        {visible.map((toast) => (
          <Toast
            key={toast.id}
            title={toast.title}
            description={toast.description}
            tone={toast.tone}
            action={toast.action}
            duration={toast.duration}
            onDismiss={() => dismissToast(toast.id)}
          />
        ))}
      </div>
    );
  },
);
