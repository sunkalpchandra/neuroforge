/**
 * Readout formatting.
 *
 * Every number shown in the chrome goes through here so that units, precision
 * and thousands separators are consistent, and so that values render at a fixed
 * width — a status bar whose numbers change width jitters the layout on every
 * frame.
 */

const COMPACT = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
});

const GROUPED = new Intl.NumberFormat('en-US');

/** 12 345 -> "12.3k". Used where horizontal space is tight. */
export function compact(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return COMPACT.format(value);
}

/** 12 345 -> "12,345". */
export function grouped(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return GROUPED.format(Math.round(value));
}

/** Fixed-precision without the exponent surprises of toPrecision. */
export function fixed(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

/**
 * Format a physical quantity with its unit, choosing a precision that keeps the
 * string a stable width as the value moves.
 */
export function quantity(value: number, unit: string, digits = 1): string {
  if (!Number.isFinite(value)) return `— ${unit}`;
  return `${value.toFixed(digits)} ${unit}`;
}

/** Simulated milliseconds as a clock, e.g. "1 240.5 ms" or "12.40 s". */
export function simTime(ms: number): string {
  if (!Number.isFinite(ms)) return '—';
  if (ms < 1000) return `${ms.toFixed(1)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = (ms % 60_000) / 1000;
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`;
}

/**
 * The ratio of simulated time to wall-clock time, which is the number that tells
 * a user whether they are watching real-time biology or a slideshow.
 */
export function realtime(factor: number): string {
  if (!Number.isFinite(factor) || factor <= 0) return '—';
  if (factor >= 1) return `${factor.toFixed(2)}×`;
  return `1/${(1 / factor).toFixed(1)}×`;
}

/** Membrane voltage, always signed so -70 and +30 align in a column. */
export function millivolts(value: number): string {
  if (!Number.isFinite(value)) return '— mV';
  const sign = value >= 0 ? '+' : '−';
  return `${sign}${Math.abs(value).toFixed(1)} mV`;
}

/** Relative time for "saved 3 s ago" style labels. */
export function since(timestamp: number, now: number): string {
  if (!timestamp) return 'never';
  const seconds = Math.max(0, (now - timestamp) / 1000);
  if (seconds < 2) return 'just now';
  if (seconds < 60) return `${Math.floor(seconds)} s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.floor(hours / 24)} d ago`;
}
