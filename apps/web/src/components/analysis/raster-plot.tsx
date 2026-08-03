'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, ChevronRight } from 'lucide-react';
import { ColorSwatch, SegmentedControl, Tooltip, cn } from '@neuroforge/ui';
import type { SegmentedOption } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { identityColor, identityColorHex } from '@neuroforge/shared';
import type { NeuronId } from '@neuroforge/shared';

import { getEngine } from '@/lib/runtime';
import { compact, fixed, grouped } from '@/lib/format';

/* ------------------------------------------------------------- constants -- */

type WindowKey = '250' | '500' | '1000' | '2000' | '5000';

const WINDOWS: Record<WindowKey, { ms: number; tick: number; label: string; short: string }> = {
  '250': { ms: 250, tick: 50, label: '250 ms', short: '250' },
  '500': { ms: 500, tick: 100, label: '500 ms', short: '500' },
  '1000': { ms: 1000, tick: 200, label: '1 s', short: '1s' },
  '2000': { ms: 2000, tick: 500, label: '2 s', short: '2s' },
  '5000': { ms: 5000, tick: 1000, label: '5 s', short: '5s' },
};

const WINDOW_KEYS: readonly WindowKey[] = ['250', '500', '1000', '2000', '5000'];

const WINDOW_OPTIONS: readonly SegmentedOption<WindowKey>[] = WINDOW_KEYS.map((key) => ({
  value: key,
  label: WINDOWS[key].short,
  title: `Show the last ${WINDOWS[key].label} of simulated time`,
}));

const MIN_HEIGHT = 84;
const MAX_HEIGHT = 460;
const DEFAULT_HEIGHT = 156;
const RESIZE_STEP = 12;

/** Bin width the rate trace aims for; widened only when the window is long. */
const TARGET_BIN_MS = 2;
const MAX_RATE_BINS = 1024;

/** Selection bands are bounded so a select-all cannot dominate a frame. */
const MAX_SELECTION_BANDS = 24;

/** Raster field, in the same near-black the 3D viewport clears to. */
const FIELD: readonly [number, number, number] = [5, 7, 10];
/** Time that holds no data — before t=0, or older than the ring buffer retains. */
const FIELD_INERT: readonly [number, number, number] = [14, 16, 19];
const GRID: readonly [number, number, number] = [27, 31, 37];
const SELECTION: readonly [number, number, number] = [79, 209, 255];

const SURFACE_CSS = '#0a0d11';
const FIELD_CSS = 'rgb(5,7,10)';
const FIELD_INERT_CSS = 'rgb(14,16,19)';
const GRID_CSS = 'rgb(27,31,37)';
const INK_FAINT_CSS = '#5a626d';
const INK_CSS = '#f5f7fa';
const ACCENT_CSS = '#4fd1ff';
const HAIRLINE_CSS = 'rgba(255,255,255,0.10)';
const CROSSHAIR_CSS = 'rgba(245,247,250,0.20)';
const LABEL_BG_CSS = 'rgba(10,13,17,0.92)';

/* ----------------------------------------------------------------- types -- */

interface HotCell {
  slot: number;
  seed: number;
  spikes: number;
}

/** Everything the header prints, republished at a fixed cadence rather than per frame. */
interface Readout {
  cells: number;
  totalSpikes: number;
  windowSpikes: number;
  meanRate: number;
  peakRate: number;
  active: number;
  binMs: number;
  hot: readonly HotCell[];
}

const EMPTY_READOUT: Readout = {
  cells: 0,
  totalSpikes: 0,
  windowSpikes: 0,
  meanRate: 0,
  peakRate: 0,
  active: 0,
  binMs: TARGET_BIN_MS,
  hot: [],
};

/**
 * Whether two readouts would print identically.
 *
 * Publishing on a timer alone re-renders the header five times a second forever,
 * including while the simulation is paused and nothing is moving. Comparing
 * first lets React bail out, so the header re-renders only when a digit it shows
 * actually changes — the rates are compared at the precision they are printed
 * at, since a change too small to display is not a change.
 */
function sameReadout(a: Readout, b: Readout): boolean {
  if (
    a.cells !== b.cells ||
    a.totalSpikes !== b.totalSpikes ||
    a.windowSpikes !== b.windowSpikes ||
    a.active !== b.active ||
    a.binMs !== b.binMs ||
    a.hot.length !== b.hot.length
  ) {
    return false;
  }
  if (Math.abs(a.meanRate - b.meanRate) >= 0.05) return false;
  if (Math.abs(a.peakRate - b.peakRate) >= 0.05) return false;
  for (let i = 0; i < a.hot.length; i += 1) {
    const x = a.hot[i];
    const y = b.hot[i];
    if (x.slot !== y.slot || x.seed !== y.seed || x.spikes !== y.spikes) return false;
  }
  return true;
}

/** Plot geometry in CSS pixels, republished each frame so hit-testing matches the paint. */
interface Geometry {
  plotX: number;
  plotY: number;
  plotW: number;
  plotH: number;
  cells: number;
}

/** Per-slot identity colour kept as bytes, so the hot loop never allocates. */
interface ColorCache {
  seeds: Uint32Array;
  rgb: Uint8ClampedArray;
  valid: Uint8Array;
}

interface PointerState {
  x: number;
  y: number;
  inside: boolean;
}

export interface RasterPlotProps {
  className?: string;
}

/* --------------------------------------------------------------- helpers -- */

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/** Axis label for a point `offset` ms in the past. */
function tickLabel(offset: number, windowMs: number, withUnit: boolean): string {
  if (offset <= 0) return 'now';
  if (windowMs >= 2000) {
    const seconds = offset / 1000;
    const text = `−${seconds.toFixed(seconds % 1 === 0 ? 0 : 1)}`;
    return withUnit ? `${text} s` : text;
  }
  return withUnit ? `−${offset} ms` : `−${offset}`;
}

/**
 * The spike raster and population-rate readout.
 *
 * Everything inside the plot is painted into a single canvas: at tens of
 * thousands of spikes per window a DOM node per event is not an option, and the
 * raster body is rebuilt from an ImageData buffer so no per-spike drawing state
 * changes hands. Each dot carries its neuron's identity colour, the same hue the
 * cell is drawn in by the renderer, so a burst in the raster can be traced to a
 * cell in the scene by colour alone.
 */
export function RasterPlot({ className }: RasterPlotProps) {
  const [open, setOpen] = useState(true);
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [windowKey, setWindowKey] = useState<WindowKey>('1000');
  const [readout, setReadout] = useState<Readout>(EMPTY_READOUT);
  // Set when the browser refuses a 2D context, so the panel explains itself
  // instead of sitting there as an unexplained black rectangle.
  const [unavailable, setUnavailable] = useState(false);

  const select = useEditor((s) => s.select);
  const selection = useEditor((s) => s.selection);

  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<ImageData | null>(null);
  const colorsRef = useRef<ColorCache | null>(null);
  const countsRef = useRef<Uint32Array | null>(null);
  const binsRef = useRef<Float32Array | null>(null);
  const ceilingRef = useRef(0);
  const pointerRef = useRef<PointerState>({ x: 0, y: 0, inside: false });
  const geometryRef = useRef<Geometry>({ plotX: 0, plotY: 0, plotW: 0, plotH: 0, cells: 0 });

  // The draw loop reads the selection through a ref: tearing down and rebuilding
  // the animation frame every time the user clicks a neuron would be absurd.
  const selectionRef = useRef<readonly NeuronId[]>(selection);
  useEffect(() => {
    selectionRef.current = selection;
  }, [selection]);

  const selectSlot = useCallback(
    (slot: number) => {
      const id = getEngine().idOf(slot);
      if (id !== null) select([id as NeuronId]);
    },
    [select],
  );

  /* ------------------------------------------------------------- resize -- */

  const startResize = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      event.preventDefault();
      const handle = event.currentTarget;
      const startY = event.clientY;
      const startHeight = height;
      handle.setPointerCapture(event.pointerId);

      const move = (moveEvent: PointerEvent) => {
        setHeight(clamp(startHeight - (moveEvent.clientY - startY), MIN_HEIGHT, MAX_HEIGHT));
      };
      const finish = () => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', finish);
        window.removeEventListener('pointercancel', finish);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', finish);
      window.addEventListener('pointercancel', finish);
    },
    [height],
  );

  const resizeByKey = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? RESIZE_STEP * 3 : RESIZE_STEP;
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHeight((value) => clamp(value + step, MIN_HEIGHT, MAX_HEIGHT));
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      setHeight((value) => clamp(value - step, MIN_HEIGHT, MAX_HEIGHT));
    } else if (event.key === 'Home') {
      event.preventDefault();
      setHeight(MIN_HEIGHT);
    } else if (event.key === 'End') {
      event.preventDefault();
      setHeight(MAX_HEIGHT);
    }
  }, []);

  /* ------------------------------------------------------------ pointer -- */

  const trackPointer = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const state = pointerRef.current;
    state.x = event.clientX - rect.left;
    state.y = event.clientY - rect.top;
    state.inside = true;
  }, []);

  const releasePointer = useCallback(() => {
    pointerRef.current.inside = false;
  }, []);

  const pickCell = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const geometry = geometryRef.current;
      if (geometry.cells <= 0 || geometry.plotH <= 0) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if (x < geometry.plotX || x > geometry.plotX + geometry.plotW) return;
      if (y < geometry.plotY || y > geometry.plotY + geometry.plotH) return;
      const row = ((y - geometry.plotY) / geometry.plotH) * geometry.cells;
      selectSlot(clamp(Math.floor(row), 0, geometry.cells - 1));
    },
    [selectSlot],
  );

  /* --------------------------------------------------------------- draw -- */

  useEffect(() => {
    if (!open) return;
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (canvas === null || host === null) return;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (ctx === null) {
      setUnavailable(true);
      return;
    }
    setUnavailable(false);

    // The canvas carries `nf-numeric`, so its computed family is the same mono
    // face the rest of the chrome prints numbers in.
    const mono = getComputedStyle(canvas).fontFamily || 'ui-monospace, monospace';

    let cssWidth = host.clientWidth;
    let cssHeight = host.clientHeight;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry === undefined) return;
      cssWidth = entry.contentRect.width;
      cssHeight = entry.contentRect.height;
    });
    observer.observe(host);

    const windowMs = WINDOWS[windowKey].ms;
    const tickMs = WINDOWS[windowKey].tick;

    let frame = 0;
    let lastFrameAt = performance.now();
    let publishedAt = 0;

    // Objects tied to this context's lifetime. Both depend only on geometry that
    // changes when the panel is resized, so rebuilding them per frame would be
    // pure garbage; they are effect-local rather than refs because a canvas
    // gradient belongs to the context that created it and dies with it.
    let fill: CanvasGradient | null = null;
    let fillTop = -1;
    let fillBottom = -1;
    let fontSpec = '';
    let fontDpr = 0;

    const draw = (now: number) => {
      frame = requestAnimationFrame(draw);

      const dt = clamp((now - lastFrameAt) / 1000, 0, 0.25);
      lastFrameAt = now;

      // Retina is worth it for hairline ticks, but a wide panel dragged tall on a
      // 5K display would push tens of megabytes through putImageData every
      // frame, so past that area the plot drops to one device pixel per CSS pixel.
      let dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (cssWidth * cssHeight * dpr * dpr > 4_000_000) dpr = 1;

      const deviceW = Math.max(2, Math.round(cssWidth * dpr));
      const deviceH = Math.max(2, Math.round(cssHeight * dpr));
      if (canvas.width !== deviceW || canvas.height !== deviceH) {
        canvas.width = deviceW;
        canvas.height = deviceH;
      }

      ctx.fillStyle = SURFACE_CSS;
      ctx.fillRect(0, 0, deviceW, deviceH);

      const padL = Math.round(31 * dpr);
      const padR = Math.round(9 * dpr);
      const padT = Math.round(6 * dpr);
      const axisH = Math.round(13 * dpr);
      const gap = Math.round(7 * dpr);
      const rateH = Math.round(clamp(cssHeight * 0.26, 22, 48) * dpr);

      const plotX = padL;
      const plotW = deviceW - padL - padR;
      const plotY = padT;
      const plotH = deviceH - padT - gap - rateH - axisH;
      const rateY = plotY + plotH + gap;

      // Hit-testing reads this object every pointer event, so it is mutated in
      // place: replacing it would allocate a fresh object every frame for the
      // sake of five numbers.
      const geometry = geometryRef.current;
      if (plotW < 8 || plotH < 8) {
        geometry.plotX = 0;
        geometry.plotY = 0;
        geometry.plotW = 0;
        geometry.plotH = 0;
        geometry.cells = 0;
        return;
      }

      // Buffers are reallocated on every structural edit, so nothing about them
      // may be cached across frames.
      const buffers = getEngine().buffers;
      const neurons = buffers.neurons;
      const log = buffers.spikes;
      const cells = neurons.count;

      geometry.plotX = plotX / dpr;
      geometry.plotY = plotY / dpr;
      geometry.plotW = plotW / dpr;
      geometry.plotH = plotH / dpr;
      geometry.cells = cells;

      const tEnd = buffers.time;
      const tStart = tEnd - windowMs;
      const perMs = plotW / windowMs;

      // `head` counts writes since reset, so the live window is the last
      // min(head, capacity) entries and it wraps at `capacity`.
      const capacity = log.capacity;
      const head = log.head;
      const valid = Math.min(head, capacity);
      const oldestIndex = head - valid;
      const horizon = head > capacity ? log.time[oldestIndex % capacity] : 0;

      // Everything left of this column is time the ring buffer cannot speak for.
      const inertUntil = clamp(Math.ceil((Math.max(0, horizon) - tStart) * perMs), 0, plotW);
      const lineW = Math.max(1, Math.round(dpr));

      /* ---- raster field ------------------------------------------------- */

      let image = imageRef.current;
      if (image === null || image.width !== plotW || image.height !== plotH) {
        image = ctx.createImageData(plotW, plotH);
        imageRef.current = image;
      }
      const px = image.data;
      const rowBytes = plotW * 4;

      // Background and gridlines vary only by column, so one row is composed and
      // then memmoved down the field, which is far cheaper than touching every
      // pixel individually at this size.
      for (let x = 0; x < plotW; x += 1) {
        const i = x * 4;
        const tone = x < inertUntil ? FIELD_INERT : FIELD;
        px[i] = tone[0];
        px[i + 1] = tone[1];
        px[i + 2] = tone[2];
        px[i + 3] = 255;
      }
      for (let offset = 0; offset <= windowMs; offset += tickMs) {
        const start = Math.min(Math.round(plotW - offset * perMs), plotW - lineW);
        for (let x = start; x < start + lineW; x += 1) {
          if (x < 0 || x >= plotW) continue;
          const i = x * 4;
          px[i] = GRID[0];
          px[i + 1] = GRID[1];
          px[i + 2] = GRID[2];
        }
      }
      for (let y = 1; y < plotH; y += 1) px.copyWithin(y * rowBytes, 0, rowBytes);

      const rowH = cells > 0 ? plotH / cells : plotH;

      /* ---- selected cells get a tinted band so they can be found --------- */

      const selected = selectionRef.current;
      if (cells > 0 && selected.length > 0) {
        const engine = getEngine();
        const bands = Math.min(selected.length, MAX_SELECTION_BANDS);
        for (let s = 0; s < bands; s += 1) {
          const slot = engine.slotOf(selected[s]);
          if (slot < 0 || slot >= cells) continue;
          const y0 = clamp(Math.floor(slot * rowH), 0, plotH - 1);
          const y1 = clamp(Math.max(y0 + 1, Math.ceil((slot + 1) * rowH)), 0, plotH);
          for (let y = y0; y < y1; y += 1) {
            const row = y * rowBytes;
            for (let x = 0; x < plotW; x += 1) {
              const i = row + x * 4;
              px[i] += (SELECTION[0] - px[i]) * 0.2;
              px[i + 1] += (SELECTION[1] - px[i + 1]) * 0.2;
              px[i + 2] += (SELECTION[2] - px[i + 2]) * 0.2;
            }
          }
        }
      }

      /* ---- one pass over the ring: dots, rate bins and per-cell counts --- */

      const binMs = Math.max(TARGET_BIN_MS, windowMs / Math.min(MAX_RATE_BINS, plotW));
      const binCount = Math.max(1, Math.ceil(windowMs / binMs));
      let bins = binsRef.current;
      if (bins === null || bins.length < binCount) {
        bins = new Float32Array(binCount);
        binsRef.current = bins;
      }
      bins.fill(0, 0, binCount);

      let counts = countsRef.current;
      if (counts === null || counts.length < neurons.capacity) {
        counts = new Uint32Array(neurons.capacity);
        countsRef.current = counts;
      }
      counts.fill(0, 0, cells);

      let colors = colorsRef.current;
      if (colors === null || colors.valid.length < neurons.capacity) {
        colors = {
          seeds: new Uint32Array(neurons.capacity),
          rgb: new Uint8ClampedArray(neurons.capacity * 3),
          valid: new Uint8Array(neurons.capacity),
        };
        colorsRef.current = colors;
      }

      const dotW = Math.max(1, Math.round(1.4 * dpr));
      const dotH = Math.max(1, Math.min(Math.round(rowH * 0.6), Math.round(9 * dpr)));
      let windowSpikes = 0;

      // Spikes are appended in time order, so walking backwards and stopping at
      // the first entry older than the window visits only what is on screen.
      for (let k = valid - 1; k >= 0; k -= 1) {
        const index = (oldestIndex + k) % capacity;
        const t = log.time[index];
        if (t < tStart) break;
        // A rewound clock leaves entries ahead of `now`; they are not history.
        if (t > tEnd) continue;
        const slot = log.neuron[index];
        if (slot >= cells) continue;

        windowSpikes += 1;
        counts[slot] += 1;

        // A spike logged on the last substep carries exactly `tEnd`, which lands
        // one past the final bin; it belongs to that bin, not to nothing.
        const bin = Math.min(binCount - 1, Math.floor((t - tStart) / binMs));
        if (bin >= 0) bins[bin] += 1;

        const seed = neurons.seed[slot];
        if (colors.valid[slot] === 0 || colors.seeds[slot] !== seed) {
          const [cr, cg, cb] = identityColor(seed);
          colors.rgb[slot * 3] = cr * 255;
          colors.rgb[slot * 3 + 1] = cg * 255;
          colors.rgb[slot * 3 + 2] = cb * 255;
          colors.seeds[slot] = seed;
          colors.valid[slot] = 1;
        }
        const r = colors.rgb[slot * 3];
        const g = colors.rgb[slot * 3 + 1];
        const b = colors.rgb[slot * 3 + 2];

        const x0 = clamp(Math.round((t - tStart) * perMs) - (dotW >> 1), 0, plotW - dotW);
        const y0 = clamp(Math.round((slot + 0.5) * rowH - dotH / 2), 0, plotH - dotH);
        for (let y = y0; y < y0 + dotH; y += 1) {
          const row = y * rowBytes;
          for (let x = x0; x < x0 + dotW; x += 1) {
            const i = row + x * 4;
            px[i] = r;
            px[i + 1] = g;
            px[i + 2] = b;
          }
        }
      }

      ctx.putImageData(image, plotX, plotY);

      /* ---- population rate ---------------------------------------------- */

      const perBinHz = cells > 0 ? 1000 / binMs / cells : 0;
      let peak = 0;
      for (let b = 0; b < binCount; b += 1) {
        const hz = bins[b] * perBinHz;
        if (hz > peak) peak = hz;
      }
      // Ease the axis rather than snapping it, so one burst does not make the
      // whole trace jump.
      const target = Math.max(4, peak * 1.15);
      const previous =
        Number.isFinite(ceilingRef.current) && ceilingRef.current > 0 ? ceilingRef.current : target;
      ceilingRef.current = previous + (target - previous) * (1 - Math.exp(-dt / 0.25));
      const scale = Math.max(1e-3, ceilingRef.current);

      ctx.fillStyle = FIELD_CSS;
      ctx.fillRect(plotX, rateY, plotW, rateH);
      if (inertUntil > 0) {
        ctx.fillStyle = FIELD_INERT_CSS;
        ctx.fillRect(plotX, rateY, inertUntil, rateH);
      }
      ctx.fillStyle = GRID_CSS;
      for (let offset = 0; offset <= windowMs; offset += tickMs) {
        const x = plotX + Math.min(Math.round(plotW - offset * perMs), plotW - lineW);
        if (x < plotX) continue;
        ctx.fillRect(x, rateY, lineW, rateH);
      }

      const baseline = rateY + rateH;
      const binX = (b: number): number => plotX + ((b + 0.5) / binCount) * plotW;
      const binY = (b: number): number =>
        baseline - clamp((bins[b] * perBinHz) / scale, 0, 1) * rateH;

      if (fill === null || fillTop !== rateY || fillBottom !== baseline) {
        fill = ctx.createLinearGradient(0, rateY, 0, baseline);
        fill.addColorStop(0, 'rgba(79,209,255,0.42)');
        fill.addColorStop(1, 'rgba(79,209,255,0.03)');
        fillTop = rateY;
        fillBottom = baseline;
      }
      ctx.beginPath();
      ctx.moveTo(plotX, baseline);
      for (let b = 0; b < binCount; b += 1) ctx.lineTo(binX(b), binY(b));
      ctx.lineTo(plotX + plotW, baseline);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();

      ctx.beginPath();
      for (let b = 0; b < binCount; b += 1) {
        const x = binX(b);
        const y = binY(b);
        if (b === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = ACCENT_CSS;
      ctx.lineWidth = lineW;
      ctx.lineJoin = 'round';
      ctx.stroke();

      /* ---- gutter and axis labels --------------------------------------- */

      // Resizing a canvas resets every context property, so the font has to be
      // reapplied each frame — but the string itself only changes with the
      // device ratio, and building it per frame is a needless allocation.
      if (fontDpr !== dpr) {
        fontDpr = dpr;
        fontSpec = `${(9 * dpr).toFixed(1)}px ${mono}`;
      }
      ctx.font = fontSpec;
      ctx.fillStyle = INK_FAINT_CSS;
      ctx.textAlign = 'right';
      const gutter = plotX - Math.round(4 * dpr);

      ctx.textBaseline = 'top';
      ctx.fillText('0', gutter, plotY);
      ctx.textBaseline = 'bottom';
      ctx.fillText(compact(Math.max(cells - 1, 0)), gutter, plotY + plotH);
      ctx.textBaseline = 'top';
      ctx.fillText(fixed(scale, scale < 10 ? 1 : 0), gutter, rateY);
      ctx.textBaseline = 'bottom';
      ctx.fillText('0', gutter, baseline);

      ctx.textBaseline = 'middle';
      const axisY = deviceH - axisH / 2;
      for (let offset = 0; offset <= windowMs; offset += tickMs) {
        const x = plotX + plotW - offset * perMs;
        if (x < plotX - 1) continue;
        const text = tickLabel(offset, windowMs, offset + tickMs > windowMs);
        const half = ctx.measureText(text).width / 2;
        if (x + half > plotX + plotW) {
          ctx.textAlign = 'right';
          ctx.fillText(text, plotX + plotW, axisY);
        } else if (x - half < plotX) {
          ctx.textAlign = 'left';
          ctx.fillText(text, Math.max(x, plotX), axisY);
        } else {
          ctx.textAlign = 'center';
          ctx.fillText(text, x, axisY);
        }
      }

      /* ---- hover crosshair ----------------------------------------------- */

      const pointer = pointerRef.current;
      const hx = pointer.x * dpr;
      const hy = pointer.y * dpr;
      if (pointer.inside && hx >= plotX && hx <= plotX + plotW && hy >= plotY && hy <= baseline) {
        ctx.fillStyle = CROSSHAIR_CSS;
        ctx.fillRect(Math.round(hx), plotY, lineW, baseline - plotY);

        const cursorMs = Math.round((plotX + plotW - hx) / perMs);
        const bin = clamp(Math.floor(((hx - plotX) / plotW) * binCount), 0, binCount - 1);
        const parts = [tickLabel(cursorMs, windowMs, true), `${fixed(bins[bin] * perBinHz, 1)} Hz`];
        let swatch: string | null = null;

        if (hy <= plotY + plotH && cells > 0) {
          const slot = clamp(Math.floor((hy - plotY) / rowH), 0, cells - 1);
          ctx.fillRect(plotX, Math.round(hy), plotW, lineW);
          parts.splice(1, 0, `#${slot}`);
          swatch = identityColorHex(neurons.seed[slot]);
        }

        const text = parts.join('   ');
        const padX = 4 * dpr;
        const chip = swatch === null ? 0 : 5 * dpr + padX;
        const boxW = ctx.measureText(text).width + padX * 2 + chip;
        const boxH = 14 * dpr;
        const boxX = clamp(hx + 6 * dpr, plotX, Math.max(plotX, plotX + plotW - boxW));
        const boxY = plotY + 3 * dpr;

        ctx.fillStyle = LABEL_BG_CSS;
        ctx.fillRect(boxX, boxY, boxW, boxH);
        ctx.strokeStyle = HAIRLINE_CSS;
        ctx.lineWidth = 1;
        ctx.strokeRect(boxX + 0.5, boxY + 0.5, boxW - 1, boxH - 1);
        if (swatch !== null) {
          ctx.fillStyle = swatch;
          ctx.fillRect(boxX + padX, boxY + boxH / 2 - 2.5 * dpr, 5 * dpr, 5 * dpr);
        }
        ctx.fillStyle = INK_CSS;
        ctx.textAlign = 'left';
        ctx.fillText(text, boxX + padX + chip, boxY + boxH / 2);
      }

      /* ---- header statistics --------------------------------------------- */

      if (now - publishedAt >= 200) {
        publishedAt = now;

        let active = 0;
        let first = -1;
        let second = -1;
        let third = -1;
        for (let s = 0; s < cells; s += 1) {
          const c = counts[s];
          if (c === 0) continue;
          active += 1;
          if (first < 0 || c > counts[first]) {
            third = second;
            second = first;
            first = s;
          } else if (second < 0 || c > counts[second]) {
            third = second;
            second = s;
          } else if (third < 0 || c > counts[third]) {
            third = s;
          }
        }

        const hot: HotCell[] = [];
        if (first >= 0) hot.push({ slot: first, seed: neurons.seed[first], spikes: counts[first] });
        if (second >= 0) {
          hot.push({ slot: second, seed: neurons.seed[second], spikes: counts[second] });
        }
        if (third >= 0) hot.push({ slot: third, seed: neurons.seed[third], spikes: counts[third] });

        // Rate is quoted over the span actually covered by data, not over the
        // nominal window, so a half-filled window does not read as half as busy.
        const span = tEnd - Math.max(tStart, horizon, 0);
        const next: Readout = {
          cells,
          totalSpikes: head,
          windowSpikes,
          meanRate: span > 0 && cells > 0 ? windowSpikes / (span / 1000) / cells : 0,
          peakRate: peak,
          active,
          binMs,
          hot,
        };
        setReadout((previous) => (sameReadout(previous, next) ? previous : next));
      }
    };

    frame = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      // Four bytes per plot pixel is by far the largest thing this panel owns,
      // and a collapsed panel has no business holding megabytes of it. The
      // smaller caches are keyed by neuron capacity and survive, so expanding
      // again does not have to recompute every identity colour.
      imageRef.current = null;
    };
  }, [open, windowKey]);

  /* --------------------------------------------------------------- view -- */

  const windowLabel = WINDOWS[windowKey].label;
  const silent = unavailable
    ? 'This browser would not give the raster a 2D canvas — the plot cannot be drawn.'
    : readout.cells === 0
      ? 'No neurons yet — build a circuit and its spikes appear here.'
      : readout.totalSpikes === 0
        ? 'No spikes yet — run the simulation or inject current to drive the network.'
        : readout.windowSpikes === 0
          ? `Silent for the last ${windowLabel}.`
          : null;

  // Keyed to the selection rather than rebuilt per render: the readout republishes
  // several times a second, and the selection can run to the whole network.
  const selectedIds = useMemo(() => new Set<string>(selection), [selection]);

  return (
    <section
      aria-label="Spike raster and population rate"
      className={cn(
        'nf-glass pointer-events-auto relative flex w-full shrink-0 flex-col',
        // The glass surface draws a border on all four sides; docked full width
        // above the status bar, only the top edge should read as a rule.
        'border-x-0 border-t border-b-0 border-hairline',
        className,
      )}
      style={{ borderRadius: 0 }}
    >
      {open ? (
        <div
          role="separator"
          aria-orientation="horizontal"
          aria-label="Raster height"
          aria-valuenow={height}
          aria-valuemin={MIN_HEIGHT}
          aria-valuemax={MAX_HEIGHT}
          tabIndex={0}
          onPointerDown={startResize}
          onKeyDown={resizeByKey}
          className="group absolute inset-x-0 -top-1 z-10 h-2 cursor-ns-resize touch-none focus-visible:outline-1"
        >
          <span
            aria-hidden
            className="mx-auto mt-[3px] block h-px w-8 bg-hairline-strong transition-colors group-hover:bg-accent/60"
          />
        </div>
      ) : null}

      <header className="flex h-7 shrink-0 items-center gap-1.5 px-2">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
          className="-mx-1 flex shrink-0 items-center gap-1 rounded px-1 py-0.5 text-ink-muted transition-colors hover:text-ink focus-visible:outline-1"
        >
          <ChevronRight
            size={11}
            aria-hidden
            className={cn('shrink-0 text-ink-faint transition-transform', open && 'rotate-90')}
          />
          <Activity size={11} aria-hidden className="shrink-0" />
          <span className="text-[10px] font-semibold tracking-[0.09em] uppercase">Raster</span>
        </button>

        {open ? (
          <>
            <div className="flex min-w-0 items-center gap-0.5 overflow-hidden">
              <Stat
                label="spk"
                value={grouped(readout.windowSpikes)}
                hint={`Spikes in the visible ${windowLabel} window`}
              />
              <Stat
                label="mean"
                value={`${fixed(readout.meanRate, 1)} Hz`}
                hint="Mean firing rate per neuron across the window"
              />
              <Stat
                label="peak"
                value={`${fixed(readout.peakRate, 1)} Hz`}
                hint={`Peak population rate, binned at ${fixed(readout.binMs, 1)} ms`}
              />
              <Stat
                label="active"
                value={`${compact(readout.active)}/${compact(readout.cells)}`}
                hint="Cells that fired at least once in the window"
              />
            </div>

            {readout.hot.length > 0 ? (
              <div
                role="radiogroup"
                aria-label="Most active cells"
                className="ml-1 flex shrink-0 items-center gap-1.5"
              >
                <span className="text-[9.5px] tracking-[0.08em] text-ink-faint uppercase">hot</span>
                {readout.hot.map((cell) => {
                  const id = getEngine().idOf(cell.slot);
                  return (
                    <span key={cell.slot} className="flex items-center gap-1">
                      <ColorSwatch
                        color={identityColorHex(cell.seed)}
                        size="sm"
                        className="size-2.5 rounded-[3px]"
                        selected={id !== null && selectedIds.has(id)}
                        onSelect={() => selectSlot(cell.slot)}
                        label={`Select cell #${cell.slot} — ${cell.spikes} spikes in the window`}
                      />
                      <span aria-hidden className="nf-numeric text-[10px] text-ink-faint">
                        {compact(cell.spikes)}
                      </span>
                    </span>
                  );
                })}
              </div>
            ) : null}

            <div className="ml-auto flex shrink-0 items-center gap-1.5">
              <span className="text-[9.5px] tracking-[0.08em] text-ink-faint uppercase">window</span>
              <SegmentedControl
                value={windowKey}
                onChange={setWindowKey}
                options={WINDOW_OPTIONS}
                size="sm"
                aria-label="Raster time window"
              />
            </div>
          </>
        ) : (
          <span className="ml-auto text-[10px] text-ink-faint">
            Spike raster and population rate — expand to plot
          </span>
        )}
      </header>

      {open ? (
        <div
          ref={hostRef}
          style={{ height }}
          className="relative w-full cursor-crosshair touch-none"
          onPointerMove={trackPointer}
          onPointerLeave={releasePointer}
          onPointerDown={pickCell}
        >
          <canvas
            ref={canvasRef}
            role="img"
            aria-label={`Spike raster over the last ${windowLabel}: ${grouped(
              readout.windowSpikes,
            )} spikes from ${grouped(readout.active)} of ${grouped(
              readout.cells,
            )} cells, mean ${fixed(readout.meanRate, 1)} hertz`}
            className="nf-numeric absolute inset-0 block h-full w-full"
          />
          {silent !== null ? (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <span className="text-[10.5px] text-ink-faint">{silent}</span>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

/** One header readout. Focusable so its tooltip is reachable from the keyboard. */
function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <Tooltip content={hint} side="top">
      <div
        role="group"
        tabIndex={0}
        aria-label={`${label}: ${value}`}
        className="flex shrink-0 items-baseline gap-1 rounded-sm px-1 select-none focus-visible:outline-1"
      >
        <span className="nf-numeric text-[10.5px] text-ink">{value}</span>
        <span className="text-[9.5px] tracking-[0.08em] text-ink-faint uppercase">{label}</span>
      </div>
    </Tooltip>
  );
}
