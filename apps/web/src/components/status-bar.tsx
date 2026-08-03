'use client';

import { useSyncExternalStore } from 'react';
import { Cpu, Gauge, Radio, Waves, Zap } from 'lucide-react';
import { Tooltip } from '@neuroforge/ui';

import { getStatsServerSnapshot, getStatsSnapshot, subscribeStats } from '@/lib/runtime';
import { compact, fixed, grouped, realtime, simTime } from '@/lib/format';

/** Subscribe to the throttled engine statistics. */
function useFrameStats() {
  return useSyncExternalStore(subscribeStats, getStatsSnapshot, getStatsServerSnapshot);
}

interface ReadoutProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  hint: string;
  tone?: 'default' | 'good' | 'warn' | 'bad';
}

const TONE_CLASS: Record<NonNullable<ReadoutProps['tone']>, string> = {
  default: 'text-ink',
  good: 'text-success',
  warn: 'text-warning',
  bad: 'text-danger',
};

function Readout({ icon, label, value, hint, tone = 'default' }: ReadoutProps) {
  return (
    <Tooltip content={hint}>
      {/* Focusable so the tooltip is reachable by keyboard; the readout itself
          has no action, so it is a group rather than a control. */}
      <div
        role="group"
        tabIndex={0}
        aria-label={`${label}: ${value}`}
        className="flex items-center gap-1.5 rounded-sm px-2.5 py-1 select-none focus-visible:outline-1"
      >
        <span className="text-ink-faint [&>svg]:size-3.5">{icon}</span>
        <span className="text-[11px] tracking-wide text-ink-faint uppercase">{label}</span>
        {/* Tabular figures keep the bar from reflowing as values change. */}
        <span className={`nf-numeric text-[11px] ${TONE_CLASS[tone]}`}>{value}</span>
      </div>
    </Tooltip>
  );
}

/**
 * The bottom instrument strip.
 *
 * Everything here is read from the throttled statistics snapshot rather than
 * from the engine directly, so this component re-renders ten times a second
 * regardless of whether the scene is running at 60 or 144 frames per second.
 */
export function StatusBar() {
  const stats = useFrameStats();

  const fpsTone = stats.fps >= 100 ? 'good' : stats.fps >= 50 ? 'default' : stats.fps > 0 ? 'warn' : 'default';
  const realtimeTone =
    stats.realtimeFactor >= 0.95 ? 'good' : stats.realtimeFactor > 0 ? 'warn' : 'default';

  return (
    <footer
      className="nf-glass flex h-[var(--nf-statusbar-h)] items-center justify-between border-t border-hairline px-1"
      style={{ borderRadius: 0 }}
    >
      <div className="flex items-center divide-x divide-hairline">
        <Readout
          icon={<Gauge />}
          label="fps"
          value={stats.fps > 0 ? fixed(stats.fps, 0) : '—'}
          hint={`Frame time ${fixed(stats.frameMs, 2)} ms`}
          tone={fpsTone}
        />
        <Readout
          icon={<Cpu />}
          label="sim"
          value={`${fixed(stats.simMs, 2)} ms`}
          hint={`${stats.substeps} integration substeps in the last frame`}
        />
        <Readout
          icon={<Waves />}
          label="t"
          value={simTime(stats.simTime)}
          hint="Simulated time elapsed"
        />
        <Readout
          icon={<Radio />}
          label="rt"
          value={realtime(stats.realtimeFactor)}
          hint="Simulated time per unit of wall-clock time"
          tone={realtimeTone}
        />
      </div>

      <div className="flex items-center divide-x divide-hairline">
        <Readout
          icon={<Zap />}
          label="rate"
          value={`${fixed(stats.meanRate, 1)} Hz`}
          hint="Mean firing rate across the network"
        />
        <Readout
          icon={<Zap />}
          label="spk"
          value={grouped(stats.spikes)}
          hint="Spikes delivered in the last frame"
        />
        <Readout
          icon={<Cpu />}
          label="n"
          value={compact(stats.neurons)}
          hint={`${grouped(stats.neurons)} neurons, ${grouped(stats.synapses)} synapses`}
        />
        <Readout
          icon={<Cpu />}
          label="backend"
          value={stats.backend}
          hint="Active compute backend"
          tone={stats.backend === 'cpu' ? 'default' : 'good'}
        />
      </div>
    </footer>
  );
}
