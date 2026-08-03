'use client';

import * as React from 'react';
import { cn } from './cn';
import { MOTION_FAST } from './styles';
import { useFieldControl } from './Field';
import { useLatest, useMergedRefs } from './hooks';
import {
  clamp,
  decimalsForStep,
  formatAdaptive,
  formatFixed,
  fromLogNormalized,
  isLogRange,
  parseNumber,
  roundSignificant,
  roundTo,
  snapToStep,
  toLogNormalized,
} from './numeric';

/** Pointer travel before a press is treated as a scrub rather than a click. */
const DRAG_THRESHOLD_PX = 3;
/** Shift: one tenth of the nominal rate and grid. */
const FINE_MULTIPLIER = 0.1;
/** Alt: ten times the nominal rate and grid. */
const COARSE_MULTIPLIER = 10;
/** Pixels of travel that sweep a bounded logarithmic range end to end. */
const LOG_DRAG_PIXELS = 360;
/** Decades per pixel when a logarithmic field has no finite bounds. */
const LOG_DECADES_PER_PIXEL = 1 / 260;
/** Arrow presses that sweep a bounded logarithmic range end to end. */
const LOG_KEY_SUBDIVISIONS = 120;
/** Per-press ratio when a logarithmic field has no finite bounds (~4.9%). */
const LOG_KEY_RATIO = 10 ** (1 / 48);
/** Significant digits kept when quantising a logarithmic value. */
const LOG_SIGNIFICANT_DIGITS = 6;
/** Extra decimals revealed when a fine drag produces a value off the step grid. */
const EXTRA_DECIMALS = 2;

interface DragSession {
  pointerId: number;
  startX: number;
  startY: number;
  lastX: number;
  moved: boolean;
  /** Unquantised value accumulator, so quantisation never causes drift. */
  raw: number;
  /** Position in log space, for bounded logarithmic fields. */
  t: number;
  emitted: number;
}

export interface NumberFieldProps
  extends Omit<
    React.InputHTMLAttributes<HTMLInputElement>,
    'value' | 'defaultValue' | 'onChange' | 'min' | 'max' | 'step' | 'size' | 'type'
  > {
  value: number;
  onChange: (value: number) => void;
  /** Fired once when a gesture ends: drag release, keyboard step, typed commit. */
  onCommit?: (value: number) => void;
  min?: number;
  max?: number;
  /** Nominal increment: one pixel of drag, one arrow press, one grid cell. */
  step?: number;
  /** Fixed decimal places. Derived from `step` when omitted. */
  precision?: number;
  /** Suffix rendered inside the control but outside the editable text. */
  unit?: string;
  /** Scrub and step in log space, for parameters that span decades. */
  logarithmic?: boolean;
  /** Value restored by a double-click. Defaults to the value at mount. */
  defaultValue?: number;
  /** Value units per pixel of drag before modifiers. Defaults to one `step`. */
  sensitivity?: number;
  /** Right alignment is the inspector default; left reads better in prose rows. */
  align?: 'left' | 'right';
  className?: string;
  inputClassName?: string;
}

/**
 * Numeric input with pointer-lock drag scrubbing.
 *
 * Drag horizontally to scrub, hold Shift for a tenth of the rate or Alt for ten
 * times it, press the arrow keys to step, double-click to restore the default,
 * or type to enter a raw edit that commits on Enter or blur and reverts on
 * Escape. During a scrub the value accumulates in a private unquantised
 * accumulator; only the emitted value is snapped, so repeated small movements
 * never get swallowed by the grid.
 */
export const NumberField = React.forwardRef<HTMLInputElement, NumberFieldProps>(
  function NumberField(
    {
      value,
      onChange,
      onCommit,
      min = Number.NEGATIVE_INFINITY,
      max = Number.POSITIVE_INFINITY,
      step = 1,
      precision,
      unit,
      logarithmic = false,
      defaultValue,
      sensitivity,
      align = 'right',
      disabled,
      id,
      className,
      inputClassName,
      onKeyDown,
      onBlur,
      ...rest
    },
    forwardedRef,
  ) {
    const field = useFieldControl(id);
    const isDisabled = disabled === true || field.disabled;

    const inputRef = React.useRef<HTMLInputElement | null>(null);
    const mergedRef = useMergedRefs(forwardedRef, inputRef);
    const dragRef = React.useRef<DragSession | null>(null);
    const mountValueRef = React.useRef(value);
    const selectAllRef = React.useRef(true);

    const [editing, setEditing] = React.useState(false);
    const [draft, setDraft] = React.useState('');
    const [dragging, setDragging] = React.useState(false);

    const gridStep = step > 0 ? step : 0;
    const rateStep = sensitivity ?? (step > 0 ? step : 1);
    const logBounded = logarithmic && isLogRange(min, max);

    /** Clamp, then snap to the (possibly modifier-scaled) grid. */
    const quantize = React.useCallback(
      (raw: number, grid: number): number => {
        if (!Number.isFinite(raw)) return clamp(0, min, max);
        let next = clamp(raw, min, max);
        if (logarithmic) {
          next = roundSignificant(next, LOG_SIGNIFICANT_DIGITS);
        } else if (grid > 0) {
          next = snapToStep(next, grid, Number.isFinite(min) ? min : 0);
        }
        return clamp(next, min, max);
      },
      [logarithmic, max, min],
    );

    const display = React.useMemo(() => {
      if (precision !== undefined) return formatFixed(value, precision);
      if (logarithmic) return formatAdaptive(value, 4);
      const base = decimalsForStep(gridStep);
      for (let decimals = base; decimals < base + EXTRA_DECIMALS; decimals += 1) {
        if (roundTo(value, decimals) === value) return formatFixed(value, decimals);
      }
      return formatFixed(value, base + EXTRA_DECIMALS);
    }, [gridStep, logarithmic, precision, value]);

    const valueRef = useLatest(value);
    const displayRef = useLatest(display);
    const draftRef = useLatest(draft);
    const changeRef = useLatest(onChange);
    const commitRef = useLatest(onCommit);

    const emit = React.useCallback(
      (next: number, commit: boolean): number => {
        if (next !== valueRef.current) changeRef.current(next);
        if (commit) commitRef.current?.(next);
        return next;
      },
      [changeRef, commitRef, valueRef],
    );

    const beginEdit = React.useCallback(
      (seed?: string) => {
        if (isDisabled) return;
        selectAllRef.current = seed === undefined;
        setDraft(seed ?? displayRef.current);
        setEditing(true);
      },
      [displayRef, isDisabled],
    );

    React.useLayoutEffect(() => {
      if (!editing) return;
      const element = inputRef.current;
      if (element === null) return;
      element.focus();
      if (selectAllRef.current) element.select();
      else element.setSelectionRange(element.value.length, element.value.length);
    }, [editing]);

    const commitEdit = React.useCallback(() => {
      setEditing(false);
      const parsed = parseNumber(draftRef.current);
      if (parsed === null) return;
      emit(quantize(parsed, gridStep), true);
    }, [draftRef, emit, gridStep, quantize]);

    const stepBy = React.useCallback(
      (steps: number, fine: boolean, coarse: boolean) => {
        if (isDisabled) return;
        const multiplier = fine ? FINE_MULTIPLIER : coarse ? COARSE_MULTIPLIER : 1;
        const amount = steps * multiplier;
        const current = valueRef.current;
        let next: number;
        if (logBounded) {
          const anchor = current > 0 ? current : min;
          const t = toLogNormalized(clamp(anchor, min, max), min, max);
          next = fromLogNormalized(t + amount / LOG_KEY_SUBDIVISIONS, min, max);
        } else if (logarithmic) {
          const anchor = current > 0 ? current : Math.max(rateStep, Number.MIN_VALUE);
          next = anchor * LOG_KEY_RATIO ** amount;
        } else {
          next = current + amount * rateStep;
        }
        emit(quantize(next, gridStep * multiplier), true);
      },
      [
        emit,
        gridStep,
        isDisabled,
        logBounded,
        logarithmic,
        max,
        min,
        quantize,
        rateStep,
        valueRef,
      ],
    );

    const resetToDefault = React.useCallback(() => {
      if (isDisabled) return;
      setEditing(false);
      emit(quantize(defaultValue ?? mountValueRef.current, gridStep), true);
    }, [defaultValue, emit, gridStep, isDisabled, quantize]);

    const releaseDrag = React.useCallback((pointerId: number) => {
      const element = inputRef.current;
      dragRef.current = null;
      setDragging(false);
      if (element === null) return;
      if (element.hasPointerCapture(pointerId)) element.releasePointerCapture(pointerId);
      if (document.pointerLockElement === element) document.exitPointerLock();
    }, []);

    const handlePointerDown = (event: React.PointerEvent<HTMLInputElement>) => {
      if (isDisabled || editing || event.button !== 0 || event.ctrlKey) return;
      const element = event.currentTarget;
      element.setPointerCapture(event.pointerId);
      const start =
        logarithmic && value <= 0
          ? Number.isFinite(min) && min > 0
            ? min
            : Math.max(rateStep, Number.MIN_VALUE)
          : value;
      dragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX,
        moved: false,
        raw: start,
        t: logBounded ? toLogNormalized(clamp(start, min, max), min, max) : 0,
        emitted: value,
      };
    };

    const handlePointerMove = (event: React.PointerEvent<HTMLInputElement>) => {
      const drag = dragRef.current;
      if (drag === null || drag.pointerId !== event.pointerId) return;
      const element = event.currentTarget;

      if (!drag.moved) {
        const travelled =
          Math.abs(event.clientX - drag.startX) >= DRAG_THRESHOLD_PX ||
          Math.abs(event.clientY - drag.startY) >= DRAG_THRESHOLD_PX;
        if (!travelled) return;
        drag.moved = true;
        setDragging(true);
        requestPointerLock(element);
      }

      const locked = document.pointerLockElement === element;
      const dx = locked ? event.movementX : event.clientX - drag.lastX;
      drag.lastX = event.clientX;
      if (dx === 0) return;

      const multiplier = event.shiftKey
        ? FINE_MULTIPLIER
        : event.altKey
          ? COARSE_MULTIPLIER
          : 1;

      if (logBounded) {
        drag.t = clamp(drag.t + (dx / LOG_DRAG_PIXELS) * multiplier, 0, 1);
        drag.raw = fromLogNormalized(drag.t, min, max);
      } else if (logarithmic) {
        drag.raw = clamp(drag.raw * 10 ** (dx * LOG_DECADES_PER_PIXEL * multiplier), min, max);
      } else {
        drag.raw = clamp(drag.raw + dx * rateStep * multiplier, min, max);
      }

      drag.emitted = emit(quantize(drag.raw, gridStep * multiplier), false);
    };

    const handlePointerUp = (event: React.PointerEvent<HTMLInputElement>) => {
      const drag = dragRef.current;
      if (drag === null || drag.pointerId !== event.pointerId) return;
      const { moved, emitted } = drag;
      releaseDrag(event.pointerId);
      if (moved) commitRef.current?.(emitted);
      else beginEdit();
    };

    const handlePointerCancel = (event: React.PointerEvent<HTMLInputElement>) => {
      const drag = dragRef.current;
      if (drag === null || drag.pointerId !== event.pointerId) return;
      releaseDrag(event.pointerId);
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
      onKeyDown?.(event);
      if (event.defaultPrevented || isDisabled) return;

      if (editing) {
        if (event.key === 'Enter') {
          event.preventDefault();
          commitEdit();
        } else if (event.key === 'Escape') {
          event.preventDefault();
          event.stopPropagation();
          setEditing(false);
        }
        return;
      }

      const { shiftKey, altKey } = event;
      switch (event.key) {
        case 'ArrowUp':
        case 'ArrowRight':
          event.preventDefault();
          stepBy(1, shiftKey, altKey);
          return;
        case 'ArrowDown':
        case 'ArrowLeft':
          event.preventDefault();
          stepBy(-1, shiftKey, altKey);
          return;
        case 'PageUp':
          event.preventDefault();
          stepBy(10, shiftKey, altKey);
          return;
        case 'PageDown':
          event.preventDefault();
          stepBy(-10, shiftKey, altKey);
          return;
        case 'Home':
          if (!Number.isFinite(min)) return;
          event.preventDefault();
          emit(quantize(min, gridStep), true);
          return;
        case 'End':
          if (!Number.isFinite(max)) return;
          event.preventDefault();
          emit(quantize(max, gridStep), true);
          return;
        case 'Enter':
        case 'F2':
          event.preventDefault();
          beginEdit();
          return;
        case 'Backspace':
        case 'Delete':
          event.preventDefault();
          beginEdit('');
          return;
        default:
          break;
      }

      if (
        event.key.length === 1 &&
        !event.ctrlKey &&
        !event.metaKey &&
        /[-0-9.+]/.test(event.key)
      ) {
        event.preventDefault();
        beginEdit(event.key);
      }
    };

    const handleBlur = (event: React.FocusEvent<HTMLInputElement>) => {
      if (editing) commitEdit();
      onBlur?.(event);
    };

    return (
      <div
        data-dragging={dragging ? '' : undefined}
        data-editing={editing ? '' : undefined}
        className={cn(
          'group relative flex h-7 min-w-0 items-center overflow-hidden rounded-control border',
          'bg-panel-raised shadow-[0_1px_0_0_rgb(255_255_255/0.03)_inset]',
          'has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-accent/60 has-[:focus-visible]:ring-offset-2 has-[:focus-visible]:ring-offset-bg',
          MOTION_FAST,
          dragging ? 'border-accent/60' : 'border-hairline hover:border-hairline-strong',
          isDisabled && 'pointer-events-none opacity-40',
          className,
        )}
      >
        <span
          aria-hidden
          className={cn(
            'pointer-events-none absolute inset-y-0 left-0 w-px origin-center bg-accent',
            MOTION_FAST,
            dragging ? 'scale-y-100 opacity-90' : 'scale-y-0 opacity-0 group-hover:scale-y-50 group-hover:opacity-40',
          )}
        />
        <input
          {...rest}
          ref={mergedRef}
          id={field.id}
          type="text"
          role="spinbutton"
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          readOnly={!editing}
          disabled={isDisabled}
          value={editing ? draft : display}
          aria-valuenow={value}
          aria-valuemin={Number.isFinite(min) ? min : undefined}
          aria-valuemax={Number.isFinite(max) ? max : undefined}
          aria-valuetext={unit === undefined ? display : `${display} ${unit}`}
          aria-describedby={rest['aria-describedby'] ?? field.describedBy}
          aria-invalid={field.invalid || undefined}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleBlur}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          onLostPointerCapture={handlePointerCancel}
          onDoubleClick={resetToDefault}
          onDragStart={(event) => event.preventDefault()}
          className={cn(
            'nf-numeric min-w-0 flex-1 bg-transparent px-2 text-[12px] leading-none text-ink',
            'outline-none placeholder:text-ink-faint',
            align === 'right' ? 'text-right' : 'text-left',
            editing
              ? 'cursor-text caret-accent selection:bg-accent/25'
              : 'cursor-ew-resize select-none caret-transparent',
            unit !== undefined && 'pr-0.5',
            inputClassName,
          )}
        />
        {unit !== undefined && (
          <span
            aria-hidden
            className="nf-numeric pointer-events-none shrink-0 pr-2 text-[10.5px] text-ink-faint"
          >
            {unit}
          </span>
        )}
      </div>
    );
  },
);

/**
 * Pointer lock lets a scrub run past the edge of the screen. It can legitimately
 * fail — a sandboxed frame, a lock requested too soon after the last exit — in
 * which case the drag falls back to clientX deltas under pointer capture.
 */
function requestPointerLock(element: Element): void {
  try {
    const result: unknown = element.requestPointerLock();
    if (result instanceof Promise) result.catch(() => undefined);
  } catch {
    /* Pointer lock is unavailable; clientX deltas remain correct. */
  }
}
