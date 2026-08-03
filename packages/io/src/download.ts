/** Browser download helper for export results. */

import type { ExportResult } from './export/index';

/**
 * Some browsers cancel an in-flight download if the object URL is revoked in
 * the same task as the click, so the revoke is deferred.
 */
const REVOKE_DELAY_MS = 1000;

function isBrowser(): boolean {
  return (
    typeof document !== 'undefined' &&
    typeof URL !== 'undefined' &&
    typeof URL.createObjectURL === 'function' &&
    typeof Blob !== 'undefined'
  );
}

/**
 * Save an export result to the user's disk.
 *
 * Throws outside a browser rather than failing silently: there is no meaningful
 * fallback, and a silent no-op would hide a wiring mistake during SSR.
 */
export function downloadExport(result: ExportResult): void {
  if (!isBrowser()) {
    throw new Error('downloadExport requires a browser environment with Blob and URL support');
  }

  const part: BlobPart =
    typeof result.content === 'string'
      ? result.content
      : // A fresh copy detaches the blob from any buffer the caller may reuse.
        new Uint8Array(result.content).buffer;
  const blob = new Blob([part], { type: result.mimeType });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = result.filename;
  anchor.rel = 'noopener';
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, REVOKE_DELAY_MS);
  }
}
