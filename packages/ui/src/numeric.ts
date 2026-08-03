/**
 * Numeric helpers shared by NumberField, Slider, Meter and Sparkline.
 *
 * Everything here is pure and free of React so the drag-scrub maths can be
 * reasoned about — and, if it ever comes to it, tested — on its own.
 */

export function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

/** Linear interpolation without the usual `a + (b - a) * t` precision loss at t = 1. */
export function lerp(a: number, b: number, t: number): number {
  return a * (1 - t) + b * t;
}

/**
 * Decimal places implied by a step. `0.25 -> 2`, `1 -> 0`, `1e-4 -> 4`.
 * Derived from the literal's own representation so that a step the caller wrote
 * as `0.05` never produces a 17-digit display.
 */
export function decimalsForStep(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0;
  const text = String(step);
  const exponentIndex = text.indexOf('e-');
  if (exponentIndex >= 0) {
    const mantissa = text.slice(0, exponentIndex).split('.');
    const mantissaDecimals = mantissa.length > 1 ? mantissa[1].length : 0;
    return Number(text.slice(exponentIndex + 2)) + mantissaDecimals;
  }
  const dotIndex = text.indexOf('.');
  return dotIndex < 0 ? 0 : text.length - dotIndex - 1;
}

/** Round to a fixed number of decimals, killing binary-floating-point dust. */
export function roundTo(value: number, decimals: number): number {
  if (!Number.isFinite(value)) return value;
  return Number(value.toFixed(clamp(Math.trunc(decimals), 0, 100)));
}

/** Round to a number of significant digits; used where values span decades. */
export function roundSignificant(value: number, digits: number): number {
  if (value === 0 || !Number.isFinite(value)) return value;
  return Number(value.toPrecision(clamp(Math.trunc(digits), 1, 21)));
}

/** Snap to the step grid anchored at `origin`. */
export function snapToStep(value: number, step: number, origin: number): number {
  if (!Number.isFinite(step) || step <= 0) return value;
  const anchor = Number.isFinite(origin) ? origin : 0;
  const snapped = anchor + Math.round((value - anchor) / step) * step;
  return roundTo(snapped, decimalsForStep(step));
}

/**
 * Position of `value` within a logarithmic [min,max] range, as 0..1.
 * Both bounds must be finite and strictly positive; callers check first.
 */
export function toLogNormalized(value: number, min: number, max: number): number {
  const span = Math.log(max / min);
  if (span === 0) return 0;
  return clamp(Math.log(value / min) / span, 0, 1);
}

/** Inverse of {@link toLogNormalized}. */
export function fromLogNormalized(t: number, min: number, max: number): number {
  return min * (max / min) ** clamp(t, 0, 1);
}

/** True when a logarithmic mapping over [min,max] is well defined. */
export function isLogRange(min: number, max: number): boolean {
  return Number.isFinite(min) && Number.isFinite(max) && min > 0 && max > min;
}

/** Position of `value` within a linear [min,max] range, as 0..1. */
export function toNormalized(value: number, min: number, max: number): number {
  if (max === min) return 0;
  return clamp((value - min) / (max - min), 0, 1);
}

function trimTrailingZeros(text: string): string {
  if (!text.includes('.')) return text;
  return text.replace(/\.?0+$/, '');
}

/** Fixed-decimal formatting; `-0` is normalised so readouts never flicker sign. */
export function formatFixed(value: number, decimals: number): string {
  if (!Number.isFinite(value)) return value > 0 ? '∞' : value < 0 ? '-∞' : 'NaN';
  const normalised = value === 0 ? 0 : value;
  return normalised.toFixed(clamp(Math.trunc(decimals), 0, 100));
}

/**
 * Significant-digit formatting that switches to exponential notation only when
 * a fixed representation would be unreadable. Used for parameters whose
 * magnitude is not known in advance.
 */
export function formatAdaptive(value: number, significant = 4): string {
  if (!Number.isFinite(value)) return value > 0 ? '∞' : value < 0 ? '-∞' : 'NaN';
  if (value === 0) return '0';
  const digits = clamp(Math.trunc(significant), 1, 21);
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6 || magnitude < 1e-4) {
    return value.toExponential(digits - 1).replace(/\.?0+e/, 'e');
  }
  return trimTrailingZeros(value.toPrecision(digits));
}

/**
 * Parse user-typed text into a number. Accepts leading/trailing whitespace, a
 * leading `+`, exponent notation, and a trailing unit suffix (`"12 ms"`).
 * Returns `null` when nothing numeric was entered.
 */
export function parseNumber(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed.length === 0) return null;
  const match = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/.exec(trimmed);
  if (match === null) return null;
  const parsed = Number(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
}
