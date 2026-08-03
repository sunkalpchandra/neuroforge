'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Copy, FlaskConical, Play, Square, TriangleAlert, X } from 'lucide-react';
import {
  Badge,
  Button,
  EmptyState,
  Field,
  IconButton,
  Meter,
  NumberField,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  Select,
  SelectItem,
  Separator,
  Tooltip,
  cn,
  pushToast,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { NEURON_MODEL_LABELS, RECEPTOR_LABELS, identityColorHex } from '@neuroforge/shared';
import type { Neuron, Synapse } from '@neuroforge/shared';

import { fixed, grouped } from '@/lib/format';
import {
  MAX_DT,
  MAX_INTERVALS,
  MAX_LEVELS,
  MIN_DT,
  ProtocolAbort,
  rheobaseProbeEstimate,
  runAdaptation,
  runFiCurve,
  runIvCurve,
  runMembraneTau,
  runPairedPulse,
  runRheobase,
  sweepCount,
} from '@/lib/experiments/protocols';
import type {
  AdaptationResult,
  FiResult,
  IvResult,
  PprResult,
  ProtocolKind,
  ProtocolResult,
  RheobaseResult,
  TauResult,
} from '@/lib/experiments/protocols';

/* ------------------------------------------------------------- protocols -- */

interface ProtocolSpec {
  value: ProtocolKind;
  label: string;
  description: string;
}

const PROTOCOLS: readonly ProtocolSpec[] = [
  { value: 'fi', label: 'F–I curve', description: 'Steady rate against injected current' },
  { value: 'iv', label: 'I–V curve', description: 'Subthreshold voltage and input resistance' },
  { value: 'tau', label: 'Membrane τ', description: 'Decay of a small hyperpolarising step' },
  { value: 'adaptation', label: 'Adaptation', description: 'Intervals through a sustained step' },
  { value: 'ppr', label: 'Paired pulse', description: 'Short-term plasticity at one synapse' },
  { value: 'rheobase', label: 'Rheobase', description: 'Bisection for the minimum spiking current' },
];

const PROTOCOL_LABEL = new Map<ProtocolKind, string>(
  PROTOCOLS.map((protocol) => [protocol.value, protocol.label]),
);

/* ------------------------------------------------------------------ plot -- */

const PLOT_W = 320;
const PAD_LEFT = 42;
const PAD_RIGHT = 10;
const PAD_TOP = 10;
const PAD_BOTTOM = 26;

/** Near-black field, matching the colour the 3D viewport clears to. */
const FIELD = 'rgb(5,7,10)';
const GRID = 'rgba(255,255,255,0.055)';

interface PlotPoint {
  x: number;
  y: number;
  title?: string;
  /** Hollow marker, used for conditions excluded from a fit. */
  open?: boolean;
}

interface PlotSeries {
  id: string;
  points: readonly PlotPoint[];
  color: string;
  line?: boolean;
  dots?: boolean;
  dashed?: boolean;
  width?: number;
  opacity?: number;
}

interface PlotMarker {
  axis: 'x' | 'y';
  value: number;
  color: string;
  label: string;
}

interface PlotProps {
  series: readonly PlotSeries[];
  markers?: readonly PlotMarker[];
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
  height?: number;
  /** Force these values into the vertical domain, e.g. the origin of a rate axis. */
  includeY?: readonly number[];
}

/**
 * Tick positions on a 1-2-5 ladder, so the axis reads in round numbers whatever
 * the data span turns out to be.
 */
function niceTicks(min: number, max: number, target: number): { values: number[]; step: number } {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(max > min)) {
    return { values: [min], step: 1 };
  }
  const raw = (max - min) / Math.max(1, target);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalised = raw / magnitude;
  const step =
    (normalised >= 5 ? 10 : normalised >= 2.5 ? 5 : normalised >= 1.5 ? 2 : 1) * magnitude;
  const values: number[] = [];
  const first = Math.ceil(min / step - 1e-9) * step;
  for (let value = first; value <= max + step * 1e-9; value += step) {
    // Accumulated float error puts a zero tick at -4.4e-16, which prints as "-0".
    values.push(Math.abs(value) < step * 1e-6 ? 0 : value);
  }
  return { values, step };
}

function tickText(value: number, step: number): string {
  const decimals = Math.min(3, Math.max(0, -Math.floor(Math.log10(Math.abs(step)))));
  return value.toFixed(decimals);
}

/**
 * Inline SVG scatter / line plot with labelled axes.
 *
 * Deliberately not a sparkline: an F-I curve without a current axis and a rate
 * axis is a decoration, not a measurement. Everything is drawn from one linear
 * mapping so a point, a fitted line and a marker rule can never disagree about
 * where a value sits.
 */
function Plot({
  series,
  markers = [],
  xLabel,
  yLabel,
  ariaLabel,
  height = 148,
  includeY = [],
}: PlotProps) {
  const domain = useMemo(() => {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    const visit = (x: number, y: number): void => {
      if (Number.isFinite(x)) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
      }
      if (Number.isFinite(y)) {
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    };
    for (const entry of series) for (const point of entry.points) visit(point.x, point.y);
    for (const marker of markers) {
      if (marker.axis === 'x') visit(marker.value, Number.NaN);
      else visit(Number.NaN, marker.value);
    }
    for (const value of includeY) visit(Number.NaN, value);

    if (!Number.isFinite(x0)) {
      x0 = 0;
      x1 = 1;
    }
    if (!Number.isFinite(y0)) {
      y0 = 0;
      y1 = 1;
    }
    if (x1 - x0 < 1e-9) {
      const pad = Math.max(Math.abs(x0) * 0.05, 0.5);
      x0 -= pad;
      x1 += pad;
    }
    if (y1 - y0 < 1e-9) {
      const pad = Math.max(Math.abs(y0) * 0.05, 0.5);
      y0 -= pad;
      y1 += pad;
    }
    const padY = (y1 - y0) * 0.08;
    return { x0, x1, y0: y0 - padY, y1: y1 + padY };
  }, [series, markers, includeY]);

  const plotW = PLOT_W - PAD_LEFT - PAD_RIGHT;
  const plotH = height - PAD_TOP - PAD_BOTTOM;
  const xTicks = niceTicks(domain.x0, domain.x1, 4);
  const yTicks = niceTicks(domain.y0, domain.y1, 4);

  const sx = (value: number): number =>
    PAD_LEFT + ((value - domain.x0) / (domain.x1 - domain.x0)) * plotW;
  const sy = (value: number): number =>
    PAD_TOP + plotH - ((value - domain.y0) / (domain.y1 - domain.y0)) * plotH;

  return (
    <svg
      viewBox={`0 0 ${PLOT_W} ${height}`}
      width="100%"
      height={height}
      role="img"
      aria-label={ariaLabel}
      className="nf-numeric block"
    >
      <rect x={PAD_LEFT} y={PAD_TOP} width={plotW} height={plotH} fill={FIELD} />

      {xTicks.values.map((tick) => (
        <line
          key={`gx${tick}`}
          x1={sx(tick)}
          y1={PAD_TOP}
          x2={sx(tick)}
          y2={PAD_TOP + plotH}
          stroke={GRID}
          strokeWidth={0.5}
        />
      ))}
      {yTicks.values.map((tick) => (
        <line
          key={`gy${tick}`}
          x1={PAD_LEFT}
          y1={sy(tick)}
          x2={PAD_LEFT + plotW}
          y2={sy(tick)}
          stroke={GRID}
          strokeWidth={0.5}
        />
      ))}

      {markers.map((marker) => {
        const horizontal = marker.axis === 'y';
        const x1 = horizontal ? PAD_LEFT : sx(marker.value);
        const x2 = horizontal ? PAD_LEFT + plotW : sx(marker.value);
        const y1 = horizontal ? sy(marker.value) : PAD_TOP;
        const y2 = horizontal ? sy(marker.value) : PAD_TOP + plotH;
        return (
          <g key={`${marker.axis}${marker.value}${marker.label}`}>
            <line
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={marker.color}
              strokeWidth={0.75}
              strokeDasharray="3 2"
              opacity={0.8}
            />
            <text
              x={horizontal ? PAD_LEFT + plotW - 2 : Math.min(x1 + 3, PAD_LEFT + plotW - 2)}
              y={horizontal ? y1 - 2.5 : PAD_TOP + 7}
              textAnchor={horizontal ? 'end' : 'start'}
              fontSize={7.5}
              fill={marker.color}
              opacity={0.9}
            >
              {marker.label}
            </text>
          </g>
        );
      })}

      {series.map((entry) => {
        const path =
          entry.line === true && entry.points.length > 1
            ? entry.points.map((point) => `${sx(point.x)},${sy(point.y)}`).join(' ')
            : null;
        return (
          <g key={entry.id} opacity={entry.opacity ?? 1}>
            {path !== null && (
              <polyline
                points={path}
                fill="none"
                stroke={entry.color}
                strokeWidth={entry.width ?? 1.1}
                strokeLinejoin="round"
                strokeLinecap="round"
                strokeDasharray={entry.dashed === true ? '3 2.5' : undefined}
              />
            )}
            {entry.dots !== false &&
              entry.points.map((point, index) => (
                <circle
                  key={`${entry.id}-${index}`}
                  cx={sx(point.x)}
                  cy={sy(point.y)}
                  r={2.1}
                  fill={point.open === true ? FIELD : entry.color}
                  stroke={entry.color}
                  strokeWidth={0.9}
                >
                  {point.title !== undefined && <title>{point.title}</title>}
                </circle>
              ))}
          </g>
        );
      })}

      <line
        x1={PAD_LEFT}
        y1={PAD_TOP + plotH}
        x2={PAD_LEFT + plotW}
        y2={PAD_TOP + plotH}
        stroke="var(--color-ink-faint)"
        strokeWidth={0.75}
        opacity={0.7}
      />
      <line
        x1={PAD_LEFT}
        y1={PAD_TOP}
        x2={PAD_LEFT}
        y2={PAD_TOP + plotH}
        stroke="var(--color-ink-faint)"
        strokeWidth={0.75}
        opacity={0.7}
      />

      {xTicks.values.map((tick) => (
        <text
          key={`tx${tick}`}
          x={sx(tick)}
          y={PAD_TOP + plotH + 9}
          textAnchor="middle"
          fontSize={8}
          fill="var(--color-ink-faint)"
        >
          {tickText(tick, xTicks.step)}
        </text>
      ))}
      {yTicks.values.map((tick) => (
        <text
          key={`ty${tick}`}
          x={PAD_LEFT - 4}
          y={sy(tick) + 2.8}
          textAnchor="end"
          fontSize={8}
          fill="var(--color-ink-faint)"
        >
          {tickText(tick, yTicks.step)}
        </text>
      ))}

      <text
        x={PAD_LEFT + plotW / 2}
        y={height - 1}
        textAnchor="middle"
        fontSize={8.5}
        fill="var(--color-ink-muted)"
      >
        {xLabel}
      </text>
      <text
        transform={`translate(8.5 ${PAD_TOP + plotH / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize={8.5}
        fill="var(--color-ink-muted)"
      >
        {yLabel}
      </text>
    </svg>
  );
}

/** Spike times as hairline ticks along a time axis. */
function SpikeStrip({ times, durationMs, color }: {
  times: readonly number[];
  durationMs: number;
  color: string;
}) {
  const width = PLOT_W;
  const height = 16;
  const span = durationMs > 0 ? durationMs : 1;
  // The strip is ~280 units wide, so past a few hundred ticks every extra
  // element lands on a column already drawn. A uniform stride keeps relative
  // density intact while bounding the DOM a fast cell over a long step produces.
  const stride = Math.max(1, Math.ceil(times.length / 600));
  const drawn = stride === 1 ? times : times.filter((_, index) => index % stride === 0);
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      role="img"
      aria-label={`${times.length} spikes over ${fixed(durationMs, 0)} milliseconds`}
      className="block"
    >
      <rect x={PAD_LEFT} y={2} width={width - PAD_LEFT - PAD_RIGHT} height={height - 4} fill={FIELD} />
      {drawn.map((time, index) => (
        <line
          key={index}
          x1={PAD_LEFT + (time / span) * (width - PAD_LEFT - PAD_RIGHT)}
          y1={3}
          x2={PAD_LEFT + (time / span) * (width - PAD_LEFT - PAD_RIGHT)}
          y2={height - 3}
          stroke={color}
          strokeWidth={0.9}
        />
      ))}
      <text x={PAD_LEFT - 4} y={height - 5} textAnchor="end" fontSize={8} fill="var(--color-ink-faint)">
        spk
      </text>
    </svg>
  );
}

/* --------------------------------------------------------------- readout -- */

function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint: string;
  tone?: string;
}) {
  return (
    <Tooltip content={hint} side="top">
      <div
        role="group"
        tabIndex={0}
        aria-label={`${label}: ${value}`}
        className="flex flex-col gap-0.5 rounded-sm focus-visible:outline-1"
      >
        <span className="truncate text-[9.5px] font-medium tracking-[0.07em] text-ink-faint uppercase">
          {label}
        </span>
        <span className={cn('nf-numeric text-[12px] leading-none text-ink', tone)}>{value}</span>
      </div>
    </Tooltip>
  );
}

function StatGrid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-x-3 gap-y-2">{children}</div>;
}

function Note({ children, tone = 'faint' }: { children: ReactNode; tone?: 'faint' | 'warning' }) {
  return (
    <p
      className={cn(
        'text-[10.5px] leading-snug',
        tone === 'warning' ? 'text-warning' : 'text-ink-faint',
      )}
    >
      {children}
    </p>
  );
}

/* --------------------------------------------------------------- helpers -- */

function optional(value: number | null, digits: number, unit: string): string {
  return value === null ? '—' : `${fixed(value, digits)} ${unit}`;
}

/** Reduce a dense trace to at most `max` points so the polyline stays cheap. */
function decimate(t: Float32Array, v: Float32Array, max: number): PlotPoint[] {
  const n = Math.min(t.length, v.length);
  if (n === 0) return [];
  const stride = Math.max(1, Math.ceil(n / max));
  const out: PlotPoint[] = [];
  for (let i = 0; i < n; i += stride) out.push({ x: t[i], y: v[i] });
  if ((n - 1) % stride !== 0) out.push({ x: t[n - 1], y: v[n - 1] });
  return out;
}

/* --------------------------------------------------------------- results -- */

function FiView({ result, color }: { result: FiResult; color: string }) {
  const data = result.points.map((point) => ({
    x: point.currentPa,
    y: point.rateHz,
    title: `${fixed(point.currentPa, 1)} pA → ${fixed(point.rateHz, 2)} Hz (${point.spikes} spikes)`,
  }));
  const series: PlotSeries[] = [{ id: 'fi', points: data, color, line: true, dots: true }];

  if (result.fit !== null && result.fit.n >= 2) {
    const firing = result.points.filter((point) => point.spikes > 0);
    const from = firing[0].currentPa;
    const to = firing[firing.length - 1].currentPa;
    series.push({
      id: 'fit',
      points: [
        { x: from, y: result.fit.intercept + result.fit.slope * from },
        { x: to, y: result.fit.intercept + result.fit.slope * to },
      ],
      color: 'var(--color-accent)',
      line: true,
      dots: false,
      dashed: true,
      width: 0.9,
      opacity: 0.85,
    });
  }

  return (
    <>
      <StatGrid>
        <Stat
          label="Rheobase"
          value={optional(result.rheobasePa, 1, 'pA')}
          tone="text-warning"
          hint="Lowest swept command current that produced any spike. Resolution equals the sweep step; use the rheobase search for a converged value."
        />
        <Stat
          label="Gain"
          value={optional(result.gainHzPerPa, 3, 'Hz/pA')}
          tone="text-accent"
          hint="Slope of a least-squares line through every level with a non-zero steady rate."
        />
        <Stat
          label="Fit r²"
          value={result.fit === null ? '—' : fixed(result.fit.r2, 4)}
          hint={
            result.fit === null
              ? 'No suprathreshold level to fit.'
              : `Fitted over ${result.fit.n} suprathreshold levels; well below 1 means the F-I relation is not linear here.`
          }
        />
        <Stat
          label="Peak rate"
          value={`${fixed(result.maxRateHz, 1)} Hz`}
          hint="Highest steady rate reached anywhere in the sweep."
        />
      </StatGrid>
      <Plot
        series={series}
        markers={
          result.rheobasePa === null
            ? []
            : [
                {
                  axis: 'x',
                  value: result.rheobasePa,
                  color: 'var(--color-warning)',
                  label: 'rheobase',
                },
              ]
        }
        xLabel="command current (pA)"
        yLabel="rate (Hz)"
        includeY={[0]}
        ariaLabel={`F-I curve over ${result.points.length} current levels, peak ${fixed(
          result.maxRateHz,
          1,
        )} hertz`}
      />
      <Note>
        {grouped(result.points.length)} levels · {fixed(result.params.settleMs, 0)} ms settling then{' '}
        {fixed(result.params.measureMs, 0)} ms counted.
      </Note>
    </>
  );
}

function IvView({ result, color }: { result: IvResult; color: string }) {
  const passive = result.points.filter((point) => !point.spiked);
  const spiking = result.points.filter((point) => point.spiked);
  const series: PlotSeries[] = [
    {
      id: 'iv',
      points: passive.map((point) => ({
        x: point.currentPa,
        y: point.steadyMv,
        title: `${fixed(point.currentPa, 1)} pA → ${fixed(point.steadyMv, 2)} mV`,
      })),
      color,
      line: true,
      dots: true,
    },
  ];
  if (spiking.length > 0) {
    series.push({
      id: 'iv-spiking',
      points: spiking.map((point) => ({
        x: point.currentPa,
        y: point.steadyMv,
        open: true,
        title: `${fixed(point.currentPa, 1)} pA — fired, excluded from the fit`,
      })),
      color: 'var(--color-warning)',
      dots: true,
    });
  }
  if (result.fit !== null && passive.length >= 2) {
    const from = passive[0].currentPa;
    const to = passive[passive.length - 1].currentPa;
    series.push({
      id: 'iv-fit',
      points: [
        { x: from, y: result.fit.intercept + result.fit.slope * from },
        { x: to, y: result.fit.intercept + result.fit.slope * to },
      ],
      color: 'var(--color-accent)',
      line: true,
      dots: false,
      dashed: true,
      width: 0.9,
      opacity: 0.85,
    });
  }

  return (
    <>
      <StatGrid>
        <Stat
          label="Input resistance"
          value={optional(result.inputResistanceMohm, 1, 'MΩ')}
          tone="text-accent"
          hint="Slope of the subthreshold line. dV/dI in mV per pA is a resistance in gigaohms; shown here in megaohms."
        />
        <Stat
          label="V at 0 pA"
          value={optional(result.interceptMv, 2, 'mV')}
          hint="Potential the fit extrapolates to with no command current — the cell's resting level at its holding bias."
        />
        <Stat
          label="Fit r²"
          value={result.fit === null ? '—' : fixed(result.fit.r2, 5)}
          hint={
            result.fit === null
              ? 'Fewer than two subthreshold levels survived.'
              : `Fitted over ${result.fit.n} subthreshold levels.`
          }
        />
        <Stat
          label="Excluded"
          value={grouped(result.excluded)}
          tone={result.excluded > 0 ? 'text-warning' : undefined}
          hint="Levels that fired and are therefore not subthreshold. They are plotted hollow and take no part in the fit."
        />
      </StatGrid>
      <Plot
        series={series}
        xLabel="command current (pA)"
        yLabel="steady V (mV)"
        ariaLabel={`I-V curve over ${result.points.length} levels, ${result.excluded} excluded for spiking`}
      />
      {result.excluded > 0 ? (
        <Note tone="warning">
          {grouped(result.excluded)} of {grouped(result.points.length)} levels fired and were
          excluded from the resistance fit. Narrow the sweep to stay subthreshold.
        </Note>
      ) : (
        <Note>
          All {grouped(result.points.length)} levels stayed subthreshold — the fit uses every point.
        </Note>
      )}
    </>
  );
}

function TauView({ result, color }: { result: TauResult; color: string }) {
  const trace = decimate(result.traceT, result.traceV, 400);
  const fitPoints: PlotPoint[] = [];
  const samples = 120;
  for (let i = 0; i <= samples; i += 1) {
    const t = (i / samples) * result.fitWindowMs;
    fitPoints.push({
      x: t,
      y: result.fitOffsetMv + result.fitAmplitudeMv * Math.exp(-t / result.tauMs),
    });
  }

  return (
    <>
      <StatGrid>
        <Stat
          label="Fitted τ"
          value={`${fixed(result.tauMs, 3)} ms`}
          tone="text-accent"
          hint={`Least-squares A·exp(-t/τ)+C over the first ${fixed(
            result.fitWindowMs,
            1,
          )} ms of the step, ${result.fitPoints} samples, r² ${fixed(result.fitR2, 5)}.`}
        />
        <Stat
          label="Analytic cm/gL"
          value={result.analyticTauMs === null ? 'n/a' : `${fixed(result.analyticTauMs, 3)} ms`}
          hint={
            result.analyticTauMs === null
              ? `The ${NEURON_MODEL_LABELS[result.meta.model]} model does not state a passive cm and gL, so there is nothing to compare against.`
              : 'Passive time constant implied by the cell parameters.'
          }
        />
        <Stat
          label="Error"
          value={result.errorPercent === null ? '—' : `${fixed(result.errorPercent, 2)} %`}
          tone={
            result.errorPercent !== null && Math.abs(result.errorPercent) > 5
              ? 'text-warning'
              : 'text-success'
          }
          hint="Signed difference between the measured and analytic time constants. A few percent is expected wherever a second process shares the membrane — AdEx's subthreshold adaptation conductance makes the true response a sum of two exponentials, not one."
        />
        <Stat
          label="Deflection"
          value={`${fixed(result.deflectionMv, 2)} mV`}
          hint={`From ${fixed(result.baselineMv, 2)} mV baseline to ${fixed(
            result.steadyMv,
            2,
          )} mV steady state.`}
        />
      </StatGrid>
      <Plot
        series={[
          { id: 'trace', points: trace, color, line: true, dots: false, width: 1 },
          {
            id: 'fit',
            points: fitPoints,
            color: 'var(--color-accent)',
            line: true,
            dots: false,
            dashed: true,
            width: 0.9,
            opacity: 0.9,
          },
        ]}
        markers={[
          { axis: 'y', value: result.fitOffsetMv, color: 'var(--color-ink-faint)', label: 'asymptote' },
          { axis: 'x', value: 0, color: 'var(--color-warning)', label: 'step' },
        ]}
        height={158}
        xLabel="time from step onset (ms)"
        yLabel="V (mV)"
        ariaLabel={`Membrane decay with a fitted time constant of ${fixed(result.tauMs, 2)} milliseconds`}
      />
      <Note>
        Dashed trace is the fit, taken over the first {fixed(result.fitWindowMs, 1)} ms — up to the
        peak of the deflection, so a slower process pulling the membrane back afterwards cannot
        contaminate τ. Fit r² {fixed(result.fitR2, 4)}.
      </Note>
    </>
  );
}

function AdaptationView({ result, color }: { result: AdaptationResult; color: string }) {
  const isiPoints = result.isisMs.map((isi, index) => ({
    x: index + 1,
    y: isi,
    title: `interval ${index + 1}: ${fixed(isi, 2)} ms (${fixed(1000 / isi, 1)} Hz)`,
  }));

  return (
    <>
      <StatGrid>
        <Stat
          label="Adaptation index"
          value={result.adaptationIndex === null ? '—' : fixed(result.adaptationIndex, 3)}
          tone={
            result.adaptationIndex !== null && result.adaptationIndex > 1.05
              ? 'text-warning'
              : 'text-accent'
          }
          hint="Last interval over first. Above 1 the cell slows through the step; exactly 1 means no adaptation at all."
        />
        <Stat
          label="Spikes"
          value={grouped(result.spikeCount)}
          hint={`Over ${fixed(result.params.durationMs, 0)} ms of sustained ${fixed(
            result.params.amplitudePa,
            0,
          )} pA.`}
        />
        <Stat
          label="Instantaneous"
          value={optional(result.instantaneousHz, 1, 'Hz')}
          hint="Reciprocal of the first inter-spike interval — the onset rate before adaptation engages."
        />
        <Stat
          label="Steady state"
          value={optional(result.steadyHz, 1, 'Hz')}
          hint="Reciprocal of the mean of the final fifth of the interval series."
        />
      </StatGrid>
      {result.spikeCount > 0 ? (
        <SpikeStrip
          times={result.spikeTimesMs}
          durationMs={result.params.durationMs}
          color={color}
        />
      ) : null}
      {isiPoints.length >= 1 ? (
        <Plot
          series={[{ id: 'isi', points: isiPoints, color, line: true, dots: true }]}
          markers={
            result.steadyHz === null
              ? []
              : [
                  {
                    axis: 'y',
                    value: 1000 / result.steadyHz,
                    color: 'var(--color-ink-faint)',
                    label: 'steady',
                  },
                ]
          }
          xLabel="interval number"
          yLabel="ISI (ms)"
          includeY={[0]}
          ariaLabel={`Inter-spike intervals across ${isiPoints.length} intervals`}
        />
      ) : (
        <Note tone="warning">
          {result.spikeCount === 0
            ? 'The step evoked no spikes at all — raise the amplitude above rheobase.'
            : 'Only one spike was evoked, so there is no interval to measure. Lengthen the step or raise the amplitude.'}
        </Note>
      )}
      {result.latencyMs !== null ? (
        <Note>First spike {fixed(result.latencyMs, 2)} ms after onset · mean {fixed(result.meanHz, 2)} Hz over the step.</Note>
      ) : null}
    </>
  );
}

function PprView({ result, color }: { result: PprResult; color: string }) {
  const measured = result.points.filter((point) => point.ratio !== null);
  const points = measured.map((point) => ({
    x: point.intervalMs,
    y: point.ratio ?? 0,
    title: `${fixed(point.intervalMs, 0)} ms: ${fixed(point.peak2Ns, 4)} / ${fixed(
      point.peak1Ns,
      4,
    )} nS = ${fixed(point.ratio ?? 0, 3)}`,
  }));
  const failed = result.points.length - measured.length;
  const firstRatio = measured.length > 0 ? measured[0].ratio : null;
  const meanPeak1 =
    measured.length > 0
      ? measured.reduce((sum, point) => sum + point.peak1Ns, 0) / measured.length
      : 0;

  return (
    <>
      {!result.synapse.stpEnabled ? (
        <Note tone="warning">
          Short-term plasticity is disabled on this synapse, so every release is identical and the
          ratio is 1 at every interval by construction. Enable STP in the inspector for this
          protocol to measure anything.
        </Note>
      ) : null}
      {result.synapse.releaseProbability < 1 && result.params.trials < 4 ? (
        <Note tone="warning">
          Release probability is {fixed(result.synapse.releaseProbability, 2)}, so each trial is a
          coin flip. Raise the trial count to average the stochastic failures out.
        </Note>
      ) : null}

      <StatGrid>
        <Stat
          label={`PPR @ ${measured.length > 0 ? fixed(measured[0].intervalMs, 0) : '—'} ms`}
          value={firstRatio === null ? '—' : fixed(firstRatio, 3)}
          tone={
            firstRatio !== null && firstRatio > 1.02
              ? 'text-success'
              : firstRatio !== null && firstRatio < 0.98
                ? 'text-warning'
                : undefined
          }
          hint="Second response over first at the shortest interval measured. Above 1 is facilitation, below 1 is depression."
        />
        <Stat
          label="First response"
          value={`${fixed(meanPeak1, 4)} nS`}
          hint="Mean peak conductance of the control response across the sweep — the denominator of every ratio."
        />
        <Stat
          label="Weight"
          value={`${fixed(result.synapse.weightNs, 3)} nS`}
          hint={`${RECEPTOR_LABELS[result.synapse.receptor]}, ${fixed(
            result.synapse.delayMs,
            2,
          )} ms conduction delay.`}
        />
        <Stat
          label="Stimulus"
          value={`${fixed(result.stimulusPa, 0)} pA`}
          hint={`Calibrated current pulse of ${fixed(
            result.stimulusMs,
            1,
          )} ms, held only until the presynaptic cell fires.`}
        />
      </StatGrid>

      {points.length > 0 ? (
        <Plot
          series={[{ id: 'ppr', points, color, line: true, dots: true }]}
          markers={[{ axis: 'y', value: 1, color: 'var(--color-ink-faint)', label: 'no change' }]}
          xLabel="inter-stimulus interval (ms)"
          yLabel="P2 / P1"
          includeY={[1]}
          ariaLabel={`Paired-pulse ratio across ${points.length} intervals`}
        />
      ) : (
        <Note tone="warning">
          No interval produced a measurable pair of responses. The first failure reported was:{' '}
          {result.points.find((point) => point.failure !== null)?.failure ?? 'unknown'}.
        </Note>
      )}

      <Note>
        Onto {result.synapse.targetLabel}
        {result.synapse.stpEnabled
          ? ` · u ${fixed(result.synapse.stpU, 2)}, τ_rec ${fixed(
              result.synapse.tauRecMs,
              0,
            )} ms, τ_facil ${fixed(result.synapse.tauFacilMs, 0)} ms`
          : ' · STP off'}
        {failed > 0 ? ` · ${grouped(failed)} interval(s) failed` : ''}
      </Note>
    </>
  );
}

function RheobaseView({ result, color }: { result: RheobaseResult; color: string }) {
  const spiked = result.probes.filter((probe) => probe.spiked);
  const silent = result.probes.filter((probe) => !probe.spiked);
  const toPoint = (probe: { iteration: number; currentPa: number; spiked: boolean }): PlotPoint => ({
    x: probe.iteration,
    y: probe.currentPa,
    open: !probe.spiked,
    title: `probe ${probe.iteration}: ${fixed(probe.currentPa, 3)} pA — ${
      probe.spiked ? 'fired' : 'silent'
    }`,
  });

  return (
    <>
      <StatGrid>
        <Stat
          label="Rheobase"
          value={`${fixed(result.rheobasePa, 3)} pA`}
          tone="text-accent"
          hint={`Smallest current proved suprathreshold within ${fixed(
            result.params.windowMs,
            0,
          )} ms. The true value lies inside the bracket below.`}
        />
        <Stat
          label="Iterations"
          value={grouped(result.iterations)}
          hint={`${result.probes.length} levels integrated in total, including both bracket checks.`}
        />
        <Stat
          label="Bracket"
          value={`${fixed(result.bracketHighPa - result.bracketLowPa, 4)} pA`}
          hint={`Converged to [${fixed(result.bracketLowPa, 3)}, ${fixed(
            result.bracketHighPa,
            3,
          )}] pA against a tolerance of ${fixed(result.params.tolerancePa, 4)} pA.`}
        />
        <Stat
          label="Latency"
          value={optional(result.latencyMs, 2, 'ms')}
          hint="Time from step onset to the first spike at the returned current. Near-rheobase latencies are long by nature."
        />
      </StatGrid>
      <Plot
        series={[
          { id: 'spiked', points: spiked.map(toPoint), color, dots: true },
          {
            id: 'silent',
            points: silent.map(toPoint),
            color: 'var(--color-ink-faint)',
            dots: true,
          },
        ]}
        markers={[
          {
            axis: 'y',
            value: result.rheobasePa,
            color: 'var(--color-accent)',
            label: 'rheobase',
          },
        ]}
        xLabel="probe"
        yLabel="current (pA)"
        ariaLabel={`Rheobase bisection over ${result.probes.length} probes converging on ${fixed(
          result.rheobasePa,
          2,
        )} picoamps`}
      />
      {result.boundedBelow ? (
        <Note tone="warning">
          The lower bound already fired, so the true rheobase is below {fixed(result.params.lowPa, 1)}{' '}
          pA. Lower it and run again.
        </Note>
      ) : (
        <Note>Filled markers fired, hollow ones stayed silent for the whole window.</Note>
      )}
    </>
  );
}

/* ----------------------------------------------------------------- panel -- */

interface Progress {
  done: number;
  total: number;
  label: string;
}

export interface EphysPanelProps {
  /** Rendered only when true, so a host can toggle it like the other panels. */
  open?: boolean;
  /** Supplying this adds a close control to the header. */
  onClose?: () => void;
  /** Overrides the default placement. */
  className?: string;
}

/**
 * Single-cell electrophysiology against the selected neuron.
 *
 * Every protocol runs in its own `SimulationEngine` over a reduced copy of the
 * document, so nothing here touches the network on screen: the user can keep the
 * simulation running while a sweep is measured. Long sweeps yield between
 * conditions and honour a cancel, because a hundred-condition F-I curve that
 * froze the tab would be worse than no F-I curve at all.
 */
export function EphysPanel({ open = true, onClose, className }: EphysPanelProps) {
  const circuit = useEditor((s) => s.circuit);
  const selection = useEditor((s) => s.selection);

  const neuron = useMemo<Neuron | null>(() => {
    if (selection.length === 0) return null;
    return circuit.neurons.find((candidate) => candidate.id === selection[0]) ?? null;
  }, [circuit.neurons, selection]);

  const [protocol, setProtocol] = useState<ProtocolKind>('fi');
  const [dt, setDt] = useState(circuit.simulation.dt);
  const [fi, setFi] = useState({ fromPa: 0, toPa: 500, stepPa: 25, settleMs: 200, measureMs: 500 });
  const [iv, setIv] = useState({ fromPa: -100, toPa: 60, stepPa: 10, settleMs: 200, measureMs: 100 });
  const [tau, setTau] = useState({ amplitudePa: -50, baselineMs: 100, stepMs: 300 });
  const [adapt, setAdapt] = useState({ amplitudePa: 400, settleMs: 100, durationMs: 1000 });
  const [ppr, setPpr] = useState({ fromMs: 10, toMs: 200, stepMs: 10, windowMs: 150, trials: 1 });
  const [rheo, setRheo] = useState({ lowPa: 0, highPa: 1000, tolerancePa: 0.5, windowMs: 500 });
  const [synapseId, setSynapseId] = useState('');

  const [result, setResult] = useState<ProtocolResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<Progress>({ done: 0, total: 1, label: '' });

  const abortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // A result describes one cell under one protocol. Changing either makes the
  // readout a lie, so it is dropped rather than left on screen next to controls
  // that no longer produced it.
  const neuronId = neuron === null ? null : neuron.id;
  useEffect(() => {
    abortRef.current?.abort();
    setResult(null);
    setError(null);
  }, [neuronId, protocol]);

  const outgoing = useMemo<Synapse[]>(() => {
    if (neuronId === null) return [];
    return circuit.synapses.filter((synapse) => synapse.source === neuronId);
  }, [circuit.synapses, neuronId]);

  const usable = useMemo(
    () => outgoing.filter((synapse) => synapse.enabled && synapse.receptor !== 'gap'),
    [outgoing],
  );

  // One pass over the document to name the handful of postsynaptic partners,
  // rather than a find() per synapse, which would be O(synapses · neurons).
  const targets = useMemo(() => {
    const wanted = new Set(outgoing.map((synapse) => synapse.target));
    const found = new Map<string, { label: string; seed: number }>();
    if (wanted.size === 0) return found;
    for (const candidate of circuit.neurons) {
      if (!wanted.has(candidate.id)) continue;
      found.set(candidate.id, {
        label: candidate.label.length > 0 ? candidate.label : candidate.id.slice(0, 8),
        seed: candidate.morphology.seed,
      });
      if (found.size === wanted.size) break;
    }
    return found;
  }, [circuit.neurons, outgoing]);

  useEffect(() => {
    if (usable.some((synapse) => synapse.id === synapseId)) return;
    setSynapseId(usable.length > 0 ? usable[0].id : '');
  }, [usable, synapseId]);

  const conditions = useMemo(() => {
    switch (protocol) {
      case 'fi':
        return sweepCount(fi.fromPa, fi.toPa, fi.stepPa, MAX_LEVELS);
      case 'iv':
        return sweepCount(iv.fromPa, iv.toPa, iv.stepPa, MAX_LEVELS);
      case 'ppr':
        // Two trials per repeat — a single-pulse control and the pair — plus the
        // one condition spent calibrating the stimulus.
        return (
          1 +
          sweepCount(ppr.fromMs, ppr.toMs, ppr.stepMs, MAX_INTERVALS) *
            Math.max(1, Math.round(ppr.trials)) *
            2
        );
      case 'rheobase':
        return rheobaseProbeEstimate(rheo.lowPa, rheo.highPa, rheo.tolerancePa);
      default:
        return 1;
    }
  }, [protocol, fi, iv, ppr, rheo]);

  /** Why Run is unavailable, or null when the protocol is ready to go. */
  const blocked = ((): string | null => {
    if (neuron === null) return 'Select a neuron to record from.';
    // A disabled cell is skipped by the integrator entirely, so every protocol
    // would report a flat line rather than a measurement.
    if (!neuron.enabled) {
      return 'This cell is disabled and excluded from integration. Enable it in the inspector to record from it.';
    }
    if (protocol !== 'ppr') return null;
    if (outgoing.length === 0) return 'This cell has no outgoing synapses to stimulate through.';
    if (usable.length === 0) return 'Every outgoing synapse is disabled or a gap junction.';
    if (!usable.some((synapse) => synapse.id === synapseId)) {
      return 'Choose a synapse to stimulate through.';
    }
    return null;
  })();

  const run = useCallback(async () => {
    if (neuron === null) return;
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;

    setRunning(true);
    setError(null);
    setResult(null);
    setProgress({ done: 0, total: 1, label: 'preparing' });

    const options = {
      signal: controller.signal,
      onProgress: (done: number, total: number, label: string) => {
        if (mountedRef.current) setProgress({ done, total, label });
      },
    };

    try {
      let next: ProtocolResult;
      switch (protocol) {
        case 'fi':
          next = await runFiCurve(circuit, neuron.id, { ...fi, dt }, options);
          break;
        case 'iv':
          next = await runIvCurve(circuit, neuron.id, { ...iv, dt }, options);
          break;
        case 'tau':
          next = await runMembraneTau(circuit, neuron.id, { ...tau, dt }, options);
          break;
        case 'adaptation':
          next = await runAdaptation(circuit, neuron.id, { ...adapt, dt }, options);
          break;
        case 'ppr':
          next = await runPairedPulse(circuit, { ...ppr, synapseId, dt }, options);
          break;
        case 'rheobase':
          next = await runRheobase(circuit, neuron.id, { ...rheo, dt }, options);
          break;
      }
      if (!mountedRef.current || controller.signal.aborted) return;
      setResult(next);
    } catch (cause) {
      // Cancelling is an outcome the user asked for, not a failure to report.
      if (cause instanceof ProtocolAbort) return;
      if (!mountedRef.current) return;
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (mountedRef.current) setRunning(false);
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [adapt, circuit, dt, fi, iv, neuron, ppr, protocol, rheo, synapseId, tau]);

  const copyCsv = useCallback(() => {
    if (result === null) return;
    // Absent outside a secure context, where the panel should say so rather than
    // throw an unhandled rejection into the console.
    const clipboard: Clipboard | undefined = navigator.clipboard;
    if (clipboard === undefined) {
      pushToast({
        title: 'Clipboard unavailable',
        description: 'This browser only exposes the clipboard over HTTPS.',
        tone: 'danger',
      });
      return;
    }
    clipboard.writeText(result.csv).then(
      () => {
        pushToast({
          title: 'Sweep copied',
          description: `${PROTOCOL_LABEL.get(result.kind) ?? 'Protocol'} data is on the clipboard as CSV.`,
          tone: 'success',
        });
      },
      (cause: unknown) => {
        pushToast({
          title: 'Could not copy',
          description: cause instanceof Error ? cause.message : String(cause),
          tone: 'danger',
        });
      },
    );
  }, [result]);

  if (!open) return null;

  const placement = className ?? 'absolute top-3 right-3 bottom-3 w-[352px]';
  const colour = neuron === null ? 'var(--color-accent)' : identityColorHex(neuron.morphology.seed);

  const header = (
    <PanelHeader
      title="Electrophysiology"
      subtitle={
        neuron === null
          ? 'No cell selected'
          : `${neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8)} · ${
              NEURON_MODEL_LABELS[neuron.params.kind]
            }`
      }
      icon={<FlaskConical />}
      actions={
        <>
          {result !== null ? (
            <Tooltip content="Wall-clock time the whole protocol took">
              <Badge variant="outline" size="sm" numeric tabIndex={0}>
                {fixed(result.meta.elapsedMs, 0)} ms
              </Badge>
            </Tooltip>
          ) : null}
          <Tooltip content="Copy the raw sweep as CSV">
            <IconButton
              label="Copy sweep as CSV"
              size="sm"
              onClick={copyCsv}
              disabled={result === null}
            >
              <Copy />
            </IconButton>
          </Tooltip>
          {onClose ? (
            <IconButton label="Close electrophysiology panel" size="sm" onClick={onClose}>
              <X />
            </IconButton>
          ) : null}
        </>
      }
    />
  );

  if (neuron === null) {
    return (
      <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
        {header}
        <EmptyState
          icon={<FlaskConical />}
          title="No cell to record from"
          description="Select a neuron in the viewport and its excitability, input resistance, membrane time constant and adaptation can all be measured here."
        />
      </Panel>
    );
  }

  return (
    <Panel className={cn('pointer-events-auto flex flex-col', placement)}>
      {header}

      <ScrollArea className="min-h-0 flex-1">
        <PanelSection label="Preparation" flush>
          <div className="flex items-center gap-2">
            <span
              aria-hidden
              className="size-3 shrink-0 rounded-[3px] ring-1 ring-white/15"
              style={{ backgroundColor: colour }}
            />
            <span className="min-w-0 flex-1 truncate text-[11px] text-ink">
              {neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8)}
            </span>
            {neuron.polarity === 'inhibitory' ? (
              <Badge variant="secondary" size="sm">
                inh
              </Badge>
            ) : null}
            <span className="nf-numeric shrink-0 text-[10px] text-ink-faint">
              {fixed(neuron.bias, 0)} pA hold
            </span>
          </div>
          {selection.length > 1 ? (
            <Note>
              {grouped(selection.length)} cells are selected; the first is the one recorded from.
            </Note>
          ) : null}
          <Note>
            The cell is characterised alone: synapses, stimuli and noise are removed, and its own{' '}
            {fixed(neuron.bias, 0)} pA bias is the holding level every command current adds to. The
            running network is untouched.
          </Note>
        </PanelSection>

        <Separator />

        <PanelSection label="Protocol">
          <Select
            value={protocol}
            onValueChange={(value) => {
              const found = PROTOCOLS.find((entry) => entry.value === value);
              if (found !== undefined) setProtocol(found.value);
            }}
            size="sm"
            aria-label="Protocol"
          >
            {PROTOCOLS.map((entry) => (
              <SelectItem key={entry.value} value={entry.value} description={entry.description}>
                {entry.label}
              </SelectItem>
            ))}
          </Select>

          <Field label="Timestep" description="Integration step used for this protocol only.">
            <NumberField
              value={dt}
              onChange={setDt}
              min={MIN_DT}
              max={MAX_DT}
              step={0.005}
              unit="ms"
              logarithmic
            />
          </Field>

          {protocol === 'fi' ? (
            <>
              <Field label="From">
                <NumberField
                  value={fi.fromPa}
                  onChange={(value) => setFi((v) => ({ ...v, fromPa: value }))}
                  step={5}
                  unit="pA"
                />
              </Field>
              <Field label="To">
                <NumberField
                  value={fi.toPa}
                  onChange={(value) => setFi((v) => ({ ...v, toPa: value }))}
                  step={5}
                  unit="pA"
                />
              </Field>
              <Field label="Step">
                <NumberField
                  value={fi.stepPa}
                  onChange={(value) => setFi((v) => ({ ...v, stepPa: value }))}
                  min={0.01}
                  step={1}
                  unit="pA"
                />
              </Field>
              <Field label="Settle" description="Discarded so adaptation has run its course.">
                <NumberField
                  value={fi.settleMs}
                  onChange={(value) => setFi((v) => ({ ...v, settleMs: value }))}
                  min={0}
                  max={20000}
                  step={10}
                  unit="ms"
                />
              </Field>
              <Field label="Measure" description="Window the steady rate is counted over.">
                <NumberField
                  value={fi.measureMs}
                  onChange={(value) => setFi((v) => ({ ...v, measureMs: value }))}
                  min={1}
                  max={20000}
                  step={10}
                  unit="ms"
                />
              </Field>
            </>
          ) : null}

          {protocol === 'iv' ? (
            <>
              <Field label="From">
                <NumberField
                  value={iv.fromPa}
                  onChange={(value) => setIv((v) => ({ ...v, fromPa: value }))}
                  step={5}
                  unit="pA"
                />
              </Field>
              <Field label="To">
                <NumberField
                  value={iv.toPa}
                  onChange={(value) => setIv((v) => ({ ...v, toPa: value }))}
                  step={5}
                  unit="pA"
                />
              </Field>
              <Field label="Step">
                <NumberField
                  value={iv.stepPa}
                  onChange={(value) => setIv((v) => ({ ...v, stepPa: value }))}
                  min={0.01}
                  step={1}
                  unit="pA"
                />
              </Field>
              <Field label="Settle">
                <NumberField
                  value={iv.settleMs}
                  onChange={(value) => setIv((v) => ({ ...v, settleMs: value }))}
                  min={0}
                  max={20000}
                  step={10}
                  unit="ms"
                />
              </Field>
              <Field label="Measure">
                <NumberField
                  value={iv.measureMs}
                  onChange={(value) => setIv((v) => ({ ...v, measureMs: value }))}
                  min={1}
                  max={20000}
                  step={10}
                  unit="ms"
                />
              </Field>
            </>
          ) : null}

          {protocol === 'tau' ? (
            <>
              <Field label="Amplitude" description="Must be hyperpolarising.">
                <NumberField
                  value={tau.amplitudePa}
                  onChange={(value) => setTau((v) => ({ ...v, amplitudePa: value }))}
                  max={-0.1}
                  step={5}
                  unit="pA"
                />
              </Field>
              <Field label="Baseline">
                <NumberField
                  value={tau.baselineMs}
                  onChange={(value) => setTau((v) => ({ ...v, baselineMs: value }))}
                  min={1}
                  max={20000}
                  step={10}
                  unit="ms"
                />
              </Field>
              <Field label="Step" description="Long enough to reach the new steady level.">
                <NumberField
                  value={tau.stepMs}
                  onChange={(value) => setTau((v) => ({ ...v, stepMs: value }))}
                  min={1}
                  max={20000}
                  step={10}
                  unit="ms"
                />
              </Field>
            </>
          ) : null}

          {protocol === 'adaptation' ? (
            <>
              <Field label="Amplitude" description="Hold well above rheobase.">
                <NumberField
                  value={adapt.amplitudePa}
                  onChange={(value) => setAdapt((v) => ({ ...v, amplitudePa: value }))}
                  step={10}
                  unit="pA"
                />
              </Field>
              <Field label="Settle">
                <NumberField
                  value={adapt.settleMs}
                  onChange={(value) => setAdapt((v) => ({ ...v, settleMs: value }))}
                  min={0}
                  max={20000}
                  step={10}
                  unit="ms"
                />
              </Field>
              <Field label="Duration">
                <NumberField
                  value={adapt.durationMs}
                  onChange={(value) => setAdapt((v) => ({ ...v, durationMs: value }))}
                  min={1}
                  max={60000}
                  step={50}
                  unit="ms"
                />
              </Field>
            </>
          ) : null}

          {protocol === 'ppr' ? (
            <>
              <Field label="Synapse" description="The connection the two stimuli are read through.">
                <Select
                  value={synapseId}
                  onValueChange={setSynapseId}
                  size="sm"
                  placeholder={outgoing.length === 0 ? 'No outgoing synapses' : 'Select…'}
                  disabled={outgoing.length === 0}
                  aria-label="Synapse under study"
                >
                  {outgoing.map((synapse) => {
                    const target = targets.get(synapse.target);
                    const unusable = !synapse.enabled || synapse.receptor === 'gap';
                    return (
                      <SelectItem
                        key={synapse.id}
                        value={synapse.id}
                        disabled={unusable}
                        icon={
                          <span
                            aria-hidden
                            className="size-2 rounded-[2px]"
                            style={{
                              backgroundColor:
                                target === undefined
                                  ? 'var(--color-ink-faint)'
                                  : identityColorHex(target.seed),
                            }}
                          />
                        }
                        description={`${RECEPTOR_LABELS[synapse.receptor]} · ${fixed(
                          synapse.weight,
                          2,
                        )} nS · ${fixed(synapse.delay, 1)} ms${
                          synapse.stp.enabled ? ' · STP' : ' · no STP'
                        }${!synapse.enabled ? ' · disabled' : ''}`}
                      >
                        {target?.label ?? synapse.target.slice(0, 8)}
                      </SelectItem>
                    );
                  })}
                </Select>
              </Field>
              <Field label="From">
                <NumberField
                  value={ppr.fromMs}
                  onChange={(value) => setPpr((v) => ({ ...v, fromMs: value }))}
                  min={1}
                  max={5000}
                  step={1}
                  unit="ms"
                />
              </Field>
              <Field label="To">
                <NumberField
                  value={ppr.toMs}
                  onChange={(value) => setPpr((v) => ({ ...v, toMs: value }))}
                  min={1}
                  max={5000}
                  step={5}
                  unit="ms"
                />
              </Field>
              <Field label="Step">
                <NumberField
                  value={ppr.stepMs}
                  onChange={(value) => setPpr((v) => ({ ...v, stepMs: value }))}
                  min={0.1}
                  step={1}
                  unit="ms"
                />
              </Field>
              <Field label="Window" description="Recorded after the second stimulus.">
                <NumberField
                  value={ppr.windowMs}
                  onChange={(value) => setPpr((v) => ({ ...v, windowMs: value }))}
                  min={1}
                  max={5000}
                  step={10}
                  unit="ms"
                />
              </Field>
              <Field
                label="Trials"
                description="Averaged per interval. Only matters when release is stochastic."
              >
                <NumberField
                  value={ppr.trials}
                  onChange={(value) => setPpr((v) => ({ ...v, trials: Math.round(value) }))}
                  min={1}
                  max={64}
                  step={1}
                  precision={0}
                />
              </Field>
            </>
          ) : null}

          {protocol === 'rheobase' ? (
            <>
              <Field label="Lower bound" description="Must be subthreshold.">
                <NumberField
                  value={rheo.lowPa}
                  onChange={(value) => setRheo((v) => ({ ...v, lowPa: value }))}
                  step={5}
                  unit="pA"
                />
              </Field>
              <Field label="Upper bound" description="Must fire within the window.">
                <NumberField
                  value={rheo.highPa}
                  onChange={(value) => setRheo((v) => ({ ...v, highPa: value }))}
                  step={10}
                  unit="pA"
                />
              </Field>
              <Field label="Tolerance">
                <NumberField
                  value={rheo.tolerancePa}
                  onChange={(value) => setRheo((v) => ({ ...v, tolerancePa: value }))}
                  min={0.0001}
                  step={0.1}
                  unit="pA"
                  logarithmic
                />
              </Field>
              <Field label="Window" description="How long a level is held before it is called silent.">
                <NumberField
                  value={rheo.windowMs}
                  onChange={(value) => setRheo((v) => ({ ...v, windowMs: value }))}
                  min={1}
                  max={20000}
                  step={50}
                  unit="ms"
                />
              </Field>
            </>
          ) : null}
        </PanelSection>

        {error !== null ? (
          <>
            <Separator />
            <PanelSection label="Result">
              <div className="flex items-start gap-2 rounded-control border border-danger/30 bg-danger/[0.08] px-2 py-1.5">
                <TriangleAlert size={12} aria-hidden className="mt-px shrink-0 text-danger" />
                <span className="text-[10.5px] leading-snug text-danger">{error}</span>
              </div>
            </PanelSection>
          </>
        ) : null}

        {result !== null ? (
          <>
            <Separator />
            <PanelSection
              label={PROTOCOL_LABEL.get(result.kind) ?? 'Result'}
              aside={
                <span className="nf-numeric text-[9.5px] text-ink-faint">
                  {fixed(result.meta.simulatedMs, 0)} ms simulated
                </span>
              }
            >
              {result.kind === 'fi' ? <FiView result={result} color={colour} /> : null}
              {result.kind === 'iv' ? <IvView result={result} color={colour} /> : null}
              {result.kind === 'tau' ? <TauView result={result} color={colour} /> : null}
              {result.kind === 'adaptation' ? (
                <AdaptationView result={result} color={colour} />
              ) : null}
              {result.kind === 'ppr' ? <PprView result={result} color={colour} /> : null}
              {result.kind === 'rheobase' ? <RheobaseView result={result} color={colour} /> : null}
            </PanelSection>
          </>
        ) : null}
      </ScrollArea>

      <div className="shrink-0 border-t border-hairline px-3 py-2">
        {running ? (
          <div className="flex flex-col gap-1.5">
            <Meter
              value={progress.total > 0 ? progress.done / progress.total : 0}
              label={progress.label.length > 0 ? progress.label : 'running'}
              valueLabel={`${grouped(progress.done)}/${grouped(progress.total)}`}
              size="sm"
            />
            <Button size="sm" variant="danger" icon={<Square />} onClick={cancel} className="w-full">
              Cancel
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="primary"
              icon={<Play />}
              onClick={() => {
                void run();
              }}
              disabled={blocked !== null}
              className="flex-1"
            >
              Run {PROTOCOL_LABEL.get(protocol)}
            </Button>
            <span className="nf-numeric shrink-0 text-[9.5px] text-ink-faint">
              {grouped(conditions)} cond
            </span>
          </div>
        )}
        {blocked !== null && !running ? (
          <p className="mt-1.5 text-[10px] leading-snug text-warning">{blocked}</p>
        ) : null}
      </div>
    </Panel>
  );
}
