'use client';

import { COLORS } from '@neuroforge/shared';

/**
 * Temporary shell used to validate the build pipeline. Replaced by the real
 * workspace composition once the panel packages land.
 */
export function Workspace() {
  return (
    <main
      className="flex h-dvh w-dvw items-center justify-center"
      style={{ backgroundColor: COLORS.background }}
    >
      <p className="nf-numeric text-sm text-ink-muted">NeuroForge</p>
    </main>
  );
}
