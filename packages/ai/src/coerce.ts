/**
 * Narrowing helpers for values that arrived from a language model, a proxy or
 * IndexedDB. Nothing in this module trusts its input.
 */

export function asRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

export function asArray(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null;
}

export function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/** A trimmed, length-bounded string, or null when there is nothing usable. */
export function asText(value: unknown, maxLength: number): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (trimmed.length === 0) return null;
  return trimmed.length > maxLength ? `${trimmed.slice(0, maxLength - 1)}…` : trimmed;
}

export function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function asBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

export function asEnum<T extends string>(value: unknown, allowed: readonly T[]): T | null {
  if (typeof value !== 'string') return null;
  return (allowed as readonly string[]).includes(value) ? (value as T) : null;
}

export function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

/** A finite number clamped into range, or null when the value is unusable. */
export function boundedNumber(value: unknown, min: number, max: number): number | null {
  const n = asFiniteNumber(value);
  return n === null ? null : clamp(n, min, max);
}

/** A finite integer clamped into range, or null when the value is unusable. */
export function boundedInteger(value: unknown, min: number, max: number): number | null {
  const n = asFiniteNumber(value);
  return n === null ? null : clamp(Math.round(n), min, max);
}

/**
 * FNV-1a over a string. Used wherever a layout or connectivity seed has to be
 * derived from text, so the same prompt always produces the same circuit.
 */
export function hashText(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}
