import { clsx } from 'clsx';
import type { ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Class-name composer used by every component in this package.
 *
 * clsx supplies the conditional semantics; tailwind-merge resolves conflicting
 * Tailwind utilities in favour of the later argument. Because every component
 * appends the caller's `className` last, a consumer can always override a
 * default without resorting to `!important`.
 */
export function cn(...classes: unknown[]): string {
  return twMerge(clsx(classes as ClassValue[]));
}
