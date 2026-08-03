'use client';

import { useMemo } from 'react';
import { useEditor } from '@neuroforge/editor';
import { COLOR_MODE_LABELS, POLARITY_COLORS, identityColorHex, hsvToRgb } from '@neuroforge/shared';
import type { ColorMode } from '@neuroforge/shared';

/** Matches TRANSMITTER_TINT in the renderer; both describe the same two classes. */
const TRANSMITTER = [
  { label: 'Cholinergic (exc.)', color: '#F5A524' },
  { label: 'GABAergic (inh.)', color: '#5B8DEF' },
];

/** Endpoints of the continuous ramps, matching rampTint in the renderer. */
function rampStops(count: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < count; i += 1) {
    const t = count === 1 ? 0 : i / (count - 1);
    const [r, g, b] = hsvToRgb(0.62 - 0.62 * t, 0.85, 1);
    const to = (c: number): string =>
      Math.round(Math.min(1, Math.max(0, c)) * 255)
        .toString(16)
        .padStart(2, '0');
    out.push(`#${to(r)}${to(g)}${to(b)}`);
  }
  return out;
}

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      className="size-2 shrink-0 rounded-[2px]"
      style={{ backgroundColor: color }}
    />
  );
}

/**
 * Key for whatever the scene is currently coloured by.
 *
 * Colour only carries meaning if the mapping is visible. Identity mode has no
 * key by construction — the hue *is* the identity and there is nothing to look
 * up — so it says so rather than showing a meaningless strip of samples.
 */
export function ColorLegend() {
  const mode: ColorMode = useEditor((s) => s.circuit.render.colorMode);
  const populations = useEditor((s) => s.circuit.populations);

  const ramp = useMemo(() => rampStops(7), []);

  const body = (() => {
    switch (mode) {
      case 'identity':
        return (
          <p className="text-[10px] leading-snug text-ink-faint">
            Hue is the cell&apos;s own identity; there is nothing to look up.
          </p>
        );

      case 'population':
        if (populations.length === 0) {
          return <p className="text-[10px] text-ink-faint">No populations defined.</p>;
        }
        return (
          <ul className="flex flex-col gap-0.5">
            {populations.slice(0, 10).map((population, index) => (
              <li key={population.id} className="flex items-center gap-1.5">
                <Swatch color={identityColorHex(index * 2654435761 + 0x9e37)} />
                <span className="min-w-0 flex-1 truncate text-[10px] text-ink-muted">
                  {population.name}
                </span>
                <span className="nf-numeric text-[10px] text-ink-faint">{population.size}</span>
              </li>
            ))}
            {populations.length > 10 ? (
              <li className="text-[10px] text-ink-faint">+{populations.length - 10} more</li>
            ) : null}
          </ul>
        );

      case 'receptor':
        return (
          <ul className="flex flex-col gap-0.5">
            {TRANSMITTER.map((entry) => (
              <li key={entry.label} className="flex items-center gap-1.5">
                <Swatch color={entry.color} />
                <span className="text-[10px] text-ink-muted">{entry.label}</span>
              </li>
            ))}
          </ul>
        );

      case 'polarity':
        return (
          <ul className="flex flex-col gap-0.5">
            <li className="flex items-center gap-1.5">
              <Swatch color={POLARITY_COLORS.excitatory} />
              <span className="text-[10px] text-ink-muted">Excitatory</span>
            </li>
            <li className="flex items-center gap-1.5">
              <Swatch color={POLARITY_COLORS.inhibitory} />
              <span className="text-[10px] text-ink-muted">Inhibitory</span>
            </li>
          </ul>
        );

      case 'voltage':
      case 'rate': {
        const low = mode === 'voltage' ? '−80 mV' : '0 Hz';
        const high = mode === 'voltage' ? '+30 mV' : '80 Hz';
        return (
          <div>
            <div
              className="h-2 w-full rounded-[2px]"
              style={{ background: `linear-gradient(to right, ${ramp.join(', ')})` }}
            />
            <div className="mt-0.5 flex justify-between">
              <span className="nf-numeric text-[10px] text-ink-faint">{low}</span>
              <span className="nf-numeric text-[10px] text-ink-faint">{high}</span>
            </div>
          </div>
        );
      }

      default:
        return null;
    }
  })();

  return (
    <div className="border-t border-hairline px-2 py-1.5">
      <div className="mb-1 text-[10px] tracking-[0.08em] text-ink-faint uppercase">
        {COLOR_MODE_LABELS[mode]}
      </div>
      {body}
    </div>
  );
}
