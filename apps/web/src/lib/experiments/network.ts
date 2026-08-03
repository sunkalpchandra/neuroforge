/**
 * Population-level experiments.
 *
 * Every measurement here runs in its own `SimulationEngine` loaded from a
 * structured clone of the document. The live engine is never touched: an
 * experiment that stepped the running simulation would leave the user's network
 * somewhere they did not put it, and a lesion study would permanently delete
 * cells from the thing on screen.
 *
 * Two rules make the comparisons mean anything:
 *
 *  - Every run of a paired protocol is driven from the *same* seed. The engine
 *    seeds its own stochastic stream from `circuit.simulation.seed` on `load`,
 *    and the CPU integrator seeds its noise stream from the value handed to its
 *    constructor, so a fresh harness is built per run rather than reusing one —
 *    reloading a used engine would leave the integrator's noise stream wherever
 *    the previous run left it and the two runs would differ for a reason that
 *    has nothing to do with the manipulation.
 *  - Control and lesion measure the *same* set of cells. Ablating a cell and
 *    then averaging over "all cells" changes the denominator as well as the
 *    network, which is the classic way to report a rate change that is really an
 *    arithmetic artefact.
 *
 * Long sweeps yield to the host between slices and honour an AbortSignal, so the
 * UI keeps painting and Cancel takes effect within a frame.
 */

import { Rng, dominantFrequency, powerSpectrum } from '@neuroforge/math';
import { SimulationEngine } from '@neuroforge/simulation';
import type { Circuit, NeuronId, SimulationBuffers, SpikeLog } from '@neuroforge/shared';

import { computeGraphMetrics } from '@/lib/graph-metrics';

/* --------------------------------------------------------------- constants -- */

/** Lowest frequency the spectrum is searched over; below this a "rhythm" is drift. */
const SPECTRUM_MIN_HZ = 1;
/** Ripple is the top of the physiological range; nothing above it is reported. */
const SPECTRUM_MAX_HZ = 200;

/**
 * Below this many spikes a power spectrum is describing sampling noise. Fifty is
 * roughly ten spikes per cycle at the coarsest resolution a one-second record
 * supports, which is the point at which a peak stops moving when the seed moves.
 */
export const MIN_SPECTRUM_SPIKES = 50;

/** Below this many cells a population average is a description of individuals. */
export const MIN_POPULATION_CELLS = 8;

/** Neurons whose individual rate variance feeds the synchrony denominator. */
const DEFAULT_SAMPLE_CAP = 256;

/** Wall-clock milliseconds of stepping between yields to the host. */
const CHUNK_BUDGET_MS = 12;

/** Minimum interval between progress publications. */
const PROGRESS_INTERVAL_MS = 80;

/**
 * Injected events are specified as the amplitude of an equivalent 1 ms current
 * pulse but delivered as a single-step impulse of the same charge. Holding a
 * pulse open across steps would need per-cell expiry bookkeeping and would stop
 * coincident events summing; an impulse sums exactly and, because the amplitude
 * is divided by `dt`, delivers the same charge whatever the timestep.
 */
const EVENT_MS = 1;

/* ------------------------------------------------------------------ bands -- */

export type BandKey = 'delta' | 'theta' | 'alpha' | 'beta' | 'gamma' | 'ripple';

export interface FrequencyBand {
  key: BandKey;
  label: string;
  symbol: string;
  lowHz: number;
  /** Exclusive upper edge, so the bands tile without overlap. */
  highHz: number;
  color: string;
}

export const FREQUENCY_BANDS: readonly FrequencyBand[] = [
  { key: 'delta', label: 'delta', symbol: 'δ', lowHz: 1, highHz: 4, color: '#2B8FB5' },
  { key: 'theta', label: 'theta', symbol: 'θ', lowHz: 4, highHz: 8, color: '#4FD1FF' },
  { key: 'alpha', label: 'alpha', symbol: 'α', lowHz: 8, highHz: 13, color: '#4ADE80' },
  { key: 'beta', label: 'beta', symbol: 'β', lowHz: 13, highHz: 30, color: '#FBBF24' },
  { key: 'gamma', label: 'gamma', symbol: 'γ', lowHz: 30, highHz: 80, color: '#FB7185' },
  { key: 'ripple', label: 'ripple', symbol: 'ρ', lowHz: 80, highHz: 200, color: '#B66BFF' },
];

export function bandFor(hz: number): FrequencyBand | null {
  for (const band of FREQUENCY_BANDS) {
    if (hz >= band.lowHz && hz < band.highHz) return band;
  }
  return null;
}

function bandIndexFor(hz: number): number {
  for (let i = 0; i < FREQUENCY_BANDS.length; i += 1) {
    const band = FREQUENCY_BANDS[i];
    if (hz >= band.lowHz && hz < band.highHz) return i;
  }
  return -1;
}

/* ------------------------------------------------------------ cancellation -- */

export class ExperimentAborted extends Error {
  constructor() {
    super('Experiment cancelled');
    this.name = 'ExperimentAborted';
  }
}

export function isAborted(error: unknown): boolean {
  return error instanceof ExperimentAborted;
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw new ExperimentAborted();
}

/**
 * Hand control back to the browser.
 *
 * An animation frame is the right unit — it is exactly the gap in which the
 * viewport repaints — but a backgrounded tab is served no frames at all, and a
 * sweep left half-finished until the user comes back is worse than one that runs
 * a little coarser. The timer is the floor under that.
 */
function yieldToHost(): Promise<void> {
  return new Promise<void>((resolve) => {
    if (typeof requestAnimationFrame !== 'function') {
      setTimeout(resolve, 0);
      return;
    }
    let settled = false;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      resolve();
    };
    requestAnimationFrame(finish);
    setTimeout(finish, 32);
  });
}

/* --------------------------------------------------------------- progress -- */

export interface ExperimentProgress {
  /** 0..1 across the whole protocol. */
  fraction: number;
  /** What is running right now. */
  label: string;
}

export interface ExperimentContext {
  signal: AbortSignal;
  onProgress?: (progress: ExperimentProgress) => void;
}

class Tracker {
  private done = 0;
  private label = '';
  private lastAt = 0;

  constructor(
    private readonly total: number,
    private readonly ctx: ExperimentContext,
  ) {}

  stage(label: string): void {
    this.label = label;
    this.emit(true);
  }

  step(count = 1): void {
    this.done += count;
  }

  emit(force = false): void {
    const now = performance.now();
    if (!force && now - this.lastAt < PROGRESS_INTERVAL_MS) return;
    this.lastAt = now;
    const fraction = this.total > 0 ? Math.min(1, this.done / this.total) : 0;
    this.ctx.onProgress?.({ fraction, label: this.label });
  }
}

/**
 * Run `steps` simulation steps, yielding whenever a slice has held the thread
 * for longer than the budget. The clock is read every eighth step: reading it
 * every step is measurable on a small circuit, and eight steps of a large one is
 * still well inside a frame.
 */
async function drive(
  steps: number,
  ctx: ExperimentContext,
  tracker: Tracker,
  step: (index: number) => void,
): Promise<void> {
  throwIfAborted(ctx.signal);
  let sliceStart = performance.now();
  for (let i = 0; i < steps; i += 1) {
    step(i);
    tracker.step();
    if ((i & 7) === 7 && performance.now() - sliceStart >= CHUNK_BUDGET_MS) {
      tracker.emit();
      await yieldToHost();
      throwIfAborted(ctx.signal);
      sliceStart = performance.now();
    }
  }
  tracker.emit(true);
}

/* ---------------------------------------------------------------- harness -- */

interface Harness {
  engine: SimulationEngine;
  buffers: SimulationBuffers;
  dt: number;
  count: number;
}

/**
 * A private engine holding a copy of the document.
 *
 * The seed is forced into the cloned document *and* into the engine
 * constructor: `load` reseeds the engine's own stream from the document, while
 * the CPU integrator's noise stream is fixed when it is constructed, and both
 * have to agree for two runs to be comparable.
 */
function createHarness(circuit: Circuit, seed: number): Harness {
  const clone = structuredClone(circuit);
  clone.simulation = { ...clone.simulation, seed, backend: 'cpu' };
  const engine = new SimulationEngine({ seed, backend: 'cpu' });
  engine.load(clone);
  return {
    engine,
    buffers: engine.buffers,
    dt: engine.settings.dt,
    count: engine.buffers.neurons.count,
  };
}

function stepsFor(durationMs: number, dt: number): number {
  if (!(dt > 0)) return 0;
  return Math.max(0, Math.round(durationMs / dt));
}

/** Slots the document says take part in integration. */
function liveMask(buffers: SimulationBuffers): Uint8Array {
  return buffers.neurons.enabled.slice(0, buffers.neurons.count);
}

/* --------------------------------------------------------------- recorder -- */

/**
 * Accumulates a recording window off the engine's spike ring.
 *
 * Reading `neurons.spike` every step would cost O(cells) per step whether or not
 * anything fired; the ring costs O(spikes). It is drained after every step so
 * that a burst can never outrun its capacity.
 */
class Recorder {
  readonly bins: number;
  readonly binMs: number;
  /** Spikes per bin from the measured cells only. */
  readonly population: Float32Array;
  /** Spikes per slot over the window, for every slot, measured or not. */
  readonly perNeuron: Uint32Array;
  /** Slots whose individual variance feeds the synchrony denominator. */
  readonly sampleSlots: Int32Array;
  /** Per-sampled-cell spikes per bin, row major. */
  readonly sampleBins: Float32Array;
  readonly measuredCount: number;

  private readonly measured: Uint8Array;
  private readonly row: Int32Array;
  private log: SpikeLog | null = null;
  private origin = 0;
  private cursor = 0;
  private counted = 0;
  private lost = 0;

  constructor(measured: Uint8Array, durationMs: number, binMs: number, sampleCap: number) {
    const count = measured.length;
    this.binMs = binMs;
    this.bins = Math.max(1, Math.round(durationMs / binMs));
    this.population = new Float32Array(this.bins);
    this.perNeuron = new Uint32Array(count);
    this.measured = measured;
    this.row = new Int32Array(count).fill(-1);

    let measuredCount = 0;
    for (let i = 0; i < count; i += 1) if (measured[i] === 1) measuredCount += 1;
    this.measuredCount = measuredCount;

    const size = Math.min(sampleCap, measuredCount);
    this.sampleSlots = new Int32Array(size);
    // Spread through slot order rather than taking a prefix: slots are laid out
    // population by population, so the first N cells are usually one group.
    let taken = 0;
    let seen = 0;
    for (let i = 0; i < count && taken < size; i += 1) {
      if (measured[i] !== 1) continue;
      if (Math.floor((seen * size) / measuredCount) === taken) {
        this.row[i] = taken;
        this.sampleSlots[taken] = i;
        taken += 1;
      }
      seen += 1;
    }
    this.sampleBins = new Float32Array(size * this.bins);
  }

  /** Spikes counted inside the window from measured cells. */
  get spikes(): number {
    return this.counted;
  }

  /** Events the ring overwrote before they could be read; always 0 in practice. */
  get dropped(): number {
    return this.lost;
  }

  get sampleSize(): number {
    return this.sampleSlots.length;
  }

  begin(buffers: SimulationBuffers): void {
    this.log = buffers.spikes;
    this.origin = buffers.time;
    this.cursor = buffers.spikes.head;
  }

  absorb(): void {
    const log = this.log;
    if (log === null) return;
    const head = log.head;
    let k = this.cursor;
    if (head - k > log.capacity) {
      this.lost += head - k - log.capacity;
      k = head - log.capacity;
    }
    for (; k < head; k += 1) {
      const index = k % log.capacity;
      const slot = log.neuron[index];
      if (slot >= this.perNeuron.length) continue;
      this.perNeuron[slot] += 1;
      if (this.measured[slot] !== 1) continue;
      const bin = Math.floor((log.time[index] - this.origin) / this.binMs);
      if (bin < 0 || bin >= this.bins) continue;
      this.population[bin] += 1;
      this.counted += 1;
      const row = this.row[slot];
      if (row >= 0) this.sampleBins[row * this.bins + bin] += 1;
    }
    this.cursor = head;
  }

  /** Population rate in Hz per measured cell, one sample per bin. */
  rateTrace(): Float32Array {
    const out = new Float32Array(this.bins);
    if (this.measuredCount === 0) return out;
    const scale = 1000 / (this.binMs * this.measuredCount);
    for (let i = 0; i < this.bins; i += 1) out[i] = this.population[i] * scale;
    return out;
  }

  activeCells(): number {
    let active = 0;
    for (let i = 0; i < this.perNeuron.length; i += 1) {
      if (this.measured[i] === 1 && this.perNeuron[i] > 0) active += 1;
    }
    return active;
  }

  /** Total spikes fired by an arbitrary set of slots, measured or not. */
  spikesOf(slots: readonly number[]): number {
    let total = 0;
    for (const slot of slots) {
      if (slot >= 0 && slot < this.perNeuron.length) total += this.perNeuron[slot];
    }
    return total;
  }
}

/* --------------------------------------------------------------- analysis -- */

export interface BandPower {
  band: FrequencyBand;
  /** Summed power in the band, in (Hz per cell)^2. */
  power: number;
  /** Share of the total power between 1 Hz and the top of the ripple band. */
  share: number;
}

export interface SpectrumResult {
  /** Single-sided power; index k is the power at k * binHz. */
  power: Float32Array;
  binHz: number;
  nyquistHz: number;
  /**
   * Frequency resolution of the record itself, 1000 / durationMs. The Hann
   * window widens the effective main lobe beyond this, so two peaks closer
   * together than a couple of these are one peak in the plot.
   */
  resolutionHz: number;
  /** Highest frequency the search covered. */
  searchMaxHz: number;
  dominantHz: number;
  dominantPower: number;
  band: FrequencyBand | null;
  bands: readonly BandPower[];
  totalPower: number;
  /** Share of the in-band power held by the band the peak falls in. */
  dominantShare: number;
}

function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

const EMPTY_SPECTRUM: SpectrumResult = {
  power: new Float32Array(1),
  binHz: 0,
  nyquistHz: 0,
  resolutionHz: 0,
  searchMaxHz: 0,
  dominantHz: 0,
  dominantPower: 0,
  band: null,
  bands: FREQUENCY_BANDS.map((band) => ({ band, power: 0, share: 0 })),
  totalPower: 0,
  dominantShare: 0,
};

/**
 * Power spectrum of a population rate trace, with the peak resolved to a band.
 *
 * The mean is removed before the transform. A network firing at 20 Hz per cell
 * has a DC term orders of magnitude above any rhythm riding on it, and the
 * window's sidelobes would smear that straight across the delta band and make
 * every circuit look like it was oscillating at 1 Hz.
 */
export function measureSpectrum(signal: Float32Array, sampleRateHz: number): SpectrumResult {
  const m = signal.length;
  if (m < 4 || !(sampleRateHz > 0)) return EMPTY_SPECTRUM;

  let sum = 0;
  for (let i = 0; i < m; i += 1) sum += signal[i];
  const mean = sum / m;
  const centred = new Float32Array(m);
  for (let i = 0; i < m; i += 1) centred[i] = signal[i] - mean;

  const nfft = nextPowerOfTwo(m);
  const power = new Float32Array(nfft / 2 + 1);
  const binHz = powerSpectrum(centred, sampleRateHz, power);
  if (!(binHz > 0)) return EMPTY_SPECTRUM;

  const nyquistHz = sampleRateHz / 2;
  const searchMaxHz = Math.min(SPECTRUM_MAX_HZ, nyquistHz);
  const dominantHz = dominantFrequency(power, binHz, SPECTRUM_MIN_HZ, searchMaxHz);

  const first = Math.max(1, Math.ceil(SPECTRUM_MIN_HZ / binHz));
  const last = Math.min(power.length - 1, Math.floor(searchMaxHz / binHz));
  const accumulated = new Float64Array(FREQUENCY_BANDS.length);
  let totalPower = 0;
  for (let k = first; k <= last; k += 1) {
    const index = bandIndexFor(k * binHz);
    if (index < 0) continue;
    accumulated[index] += power[k];
    totalPower += power[k];
  }

  const bands: BandPower[] = FREQUENCY_BANDS.map((band, index) => ({
    band,
    power: accumulated[index],
    share: totalPower > 0 ? accumulated[index] / totalPower : 0,
  }));

  const peakBin = Math.min(power.length - 1, Math.max(0, Math.round(dominantHz / binHz)));
  const band = dominantHz > 0 ? bandFor(dominantHz) : null;
  const bandIndex = band === null ? -1 : bandIndexFor(dominantHz);

  return {
    power,
    binHz,
    nyquistHz,
    resolutionHz: sampleRateHz / m,
    searchMaxHz,
    dominantHz,
    dominantPower: power[peakBin],
    band,
    bands,
    totalPower,
    dominantShare: bandIndex >= 0 && totalPower > 0 ? accumulated[bandIndex] / totalPower : 0,
  };
}

export interface SynchronyResult {
  /**
   * Golomb–Rinzel χ²: the variance of the population activity over the mean of
   * the individual cells' variances. 1 is lockstep, 1/N is independent.
   */
  index: number | null;
  /** √χ², the form some of the literature quotes. */
  chi: number | null;
  sampleSize: number;
  binMs: number;
  populationVariance: number;
  meanCellVariance: number;
  /** The floor χ² would sit at for a perfectly asynchronous population. */
  asynchronousFloor: number;
}

function varianceOf(values: Float32Array, offset: number, count: number): number {
  if (count < 2) return 0;
  let sum = 0;
  for (let i = 0; i < count; i += 1) sum += values[offset + i];
  const mean = sum / count;
  let acc = 0;
  for (let i = 0; i < count; i += 1) {
    const d = values[offset + i] - mean;
    acc += d * d;
  }
  return acc / (count - 1);
}

/**
 * Synchrony of a population from its binned spike counts.
 *
 * The numerator is measured over every cell in the population; the denominator
 * is averaged over a sample of them, because it is a mean over cells and a few
 * hundred is plenty to estimate one, whereas holding a per-bin trace for a
 * hundred thousand cells is not.
 */
export function measureSynchrony(
  population: Float32Array,
  measuredCount: number,
  sampleBins: Float32Array,
  sampleSize: number,
  bins: number,
  binMs: number,
): SynchronyResult {
  const floor = measuredCount > 0 ? 1 / measuredCount : 0;
  if (measuredCount === 0 || sampleSize === 0 || bins < 2) {
    return {
      index: null,
      chi: null,
      sampleSize,
      binMs,
      populationVariance: 0,
      meanCellVariance: 0,
      asynchronousFloor: floor,
    };
  }

  const activity = new Float32Array(bins);
  for (let t = 0; t < bins; t += 1) activity[t] = population[t] / measuredCount;
  const populationVariance = varianceOf(activity, 0, bins);

  let cellVarianceSum = 0;
  for (let s = 0; s < sampleSize; s += 1) {
    cellVarianceSum += varianceOf(sampleBins, s * bins, bins);
  }
  const meanCellVariance = cellVarianceSum / sampleSize;

  if (!(meanCellVariance > 0)) {
    return {
      index: null,
      chi: null,
      sampleSize,
      binMs,
      populationVariance,
      meanCellVariance,
      asynchronousFloor: floor,
    };
  }

  const index = populationVariance / meanCellVariance;
  return {
    index,
    chi: Math.sqrt(Math.max(0, index)),
    sampleSize,
    binMs,
    populationVariance,
    meanCellVariance,
    asynchronousFloor: floor,
  };
}

/* ---------------------------------------------------------------- targets -- */

export interface CellRef {
  slot: number;
  id: NeuronId | null;
  /** Morphology seed — the key `identityColor` is derived from. */
  colorSeed: number;
  label: string;
}

export function cellRef(circuit: Circuit, slot: number): CellRef {
  const neuron = circuit.neurons[slot];
  if (neuron === undefined) {
    return { slot, id: null, colorSeed: 0, label: `#${slot}` };
  }
  return {
    slot,
    id: neuron.id,
    colorSeed: neuron.morphology.seed,
    label: neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8),
  };
}

export type TargetKind = 'all' | 'selection' | 'population';

export interface TargetSpec {
  kind: TargetKind;
  /** Only meaningful when `kind` is 'population'. */
  populationId?: string;
}

export function targetSpecValue(spec: TargetSpec): string {
  return spec.kind === 'population' ? `population:${spec.populationId ?? ''}` : spec.kind;
}

export function parseTargetSpec(value: string): TargetSpec {
  if (value.startsWith('population:')) {
    return { kind: 'population', populationId: value.slice('population:'.length) };
  }
  return { kind: value === 'selection' ? 'selection' : 'all' };
}

/**
 * Slots named by a spec.
 *
 * A slot is the index of the neuron in `circuit.neurons`, which is exactly how
 * `SimulationEngine.load` assigns them, so a set resolved here is valid in the
 * live engine and in every harness built from the same document.
 */
export function resolveTargetSlots(
  circuit: Circuit,
  spec: TargetSpec,
  selection: readonly NeuronId[],
  exclude?: readonly number[] | null,
): number[] {
  const excluded = exclude === null || exclude === undefined ? null : new Set(exclude);
  const keep = (slot: number): boolean =>
    slot >= 0 && slot < circuit.neurons.length && (excluded === null || !excluded.has(slot));

  if (spec.kind === 'all') {
    const out: number[] = [];
    for (let i = 0; i < circuit.neurons.length; i += 1) if (keep(i)) out.push(i);
    return out;
  }

  const index = new Map<string, number>();
  circuit.neurons.forEach((neuron, slot) => index.set(neuron.id, slot));

  const ids =
    spec.kind === 'selection'
      ? selection
      : (circuit.populations.find((p) => p.id === spec.populationId)?.members ?? []);

  const out: number[] = [];
  const seen = new Set<number>();
  for (const id of ids) {
    const slot = index.get(id);
    if (slot === undefined || !keep(slot) || seen.has(slot)) continue;
    seen.add(slot);
    out.push(slot);
  }
  return out;
}

/** The `count` most connected cells, by total degree, of a loaded network. */
export function topHubSlots(buffers: SimulationBuffers, count: number): number[] {
  if (count <= 0 || buffers.neurons.count === 0) return [];
  return computeGraphMetrics(buffers, count).hubs.map((hub) => hub.slot);
}

/** A reproducible random sample of the enabled cells; the same seed always hits the same set. */
export function randomSlotSample(circuit: Circuit, count: number, seed: number): number[] {
  const pool: number[] = [];
  for (let i = 0; i < circuit.neurons.length; i += 1) {
    if (circuit.neurons[i].enabled) pool.push(i);
  }
  new Rng(seed).shuffle(pool);
  return pool.slice(0, Math.max(0, Math.min(count, pool.length))).sort((a, b) => a - b);
}

/* ------------------------------------------------------------- run summary -- */

export interface RunSummary {
  /** Slots in the network. */
  neurons: number;
  /** Cells the population signal was measured over. */
  measured: number;
  ablated: number;
  /** Spikes from measured cells inside the recording window. */
  spikes: number;
  activeCells: number;
  meanRateHz: number;
  durationMs: number;
  warmupMs: number;
  binMs: number;
  sampleSize: number;
  seed: number;
  steps: number;
}

interface Measured {
  recorder: Recorder;
  summary: RunSummary;
  rate: Float32Array;
  spectrum: SpectrumResult;
  synchrony: SynchronyResult;
}

interface MeasureOptions {
  seed: number;
  warmupMs: number;
  durationMs: number;
  binMs: number;
  label: string;
  /** Slots excluded from integration for this run. */
  ablate?: readonly number[];
  /** Slots the population signal covers. Defaults to every live cell. */
  measure?: readonly number[] | null;
  /** Called before each step; the place external drive is injected. */
  onStep?: (engine: SimulationEngine, recording: boolean) => void;
}

/**
 * One recorded run in a private engine.
 *
 * The warm-up is stepped but not recorded: the first tens of milliseconds after
 * a reset are the network falling out of its initial condition, and a spectrum
 * computed over that transient reports the onset, not the rhythm.
 */
async function measureRun(
  circuit: Circuit,
  options: MeasureOptions,
  ctx: ExperimentContext,
  tracker: Tracker,
): Promise<Measured> {
  const harness = createHarness(circuit, options.seed);
  try {
    const { engine, buffers, dt } = harness;
    const neurons = buffers.neurons;

    const ablate = options.ablate ?? [];
    for (const slot of ablate) {
      if (slot >= 0 && slot < neurons.count) neurons.enabled[slot] = 0;
    }

    let measured: Uint8Array;
    if (options.measure === null || options.measure === undefined) {
      measured = liveMask(buffers);
    } else {
      measured = new Uint8Array(neurons.count);
      for (const slot of options.measure) {
        if (slot >= 0 && slot < neurons.count && neurons.enabled[slot] === 1) measured[slot] = 1;
      }
    }

    const recorder = new Recorder(
      measured,
      options.durationMs,
      options.binMs,
      DEFAULT_SAMPLE_CAP,
    );

    const warmupSteps = stepsFor(options.warmupMs, dt);
    const recordSteps = stepsFor(options.durationMs, dt);

    tracker.stage(options.label);
    await drive(warmupSteps, ctx, tracker, () => {
      options.onStep?.(engine, false);
      engine.stepOnce();
    });

    recorder.begin(buffers);
    await drive(recordSteps, ctx, tracker, () => {
      options.onStep?.(engine, true);
      engine.stepOnce();
      recorder.absorb();
    });

    const seconds = (recordSteps * dt) / 1000;
    const meanRateHz =
      recorder.measuredCount > 0 && seconds > 0
        ? recorder.spikes / (recorder.measuredCount * seconds)
        : 0;

    const rate = recorder.rateTrace();
    const spectrum = measureSpectrum(rate, 1000 / options.binMs);
    const synchrony = measureSynchrony(
      recorder.population,
      recorder.measuredCount,
      recorder.sampleBins,
      recorder.sampleSize,
      recorder.bins,
      options.binMs,
    );

    return {
      recorder,
      rate,
      spectrum,
      synchrony,
      summary: {
        neurons: neurons.count,
        measured: recorder.measuredCount,
        ablated: ablate.length,
        spikes: recorder.spikes,
        activeCells: recorder.activeCells(),
        meanRateHz,
        durationMs: recordSteps * dt,
        warmupMs: warmupSteps * dt,
        binMs: options.binMs,
        sampleSize: recorder.sampleSize,
        seed: options.seed,
        steps: warmupSteps + recordSteps,
      },
    };
  } finally {
    harness.engine.dispose();
  }
}

/* ----------------------------------------------------------- shared checks -- */

function circuitWarnings(circuit: Circuit): string[] {
  const warnings: string[] = [];
  if (circuit.simulation.plasticityEnabled) {
    warnings.push(
      'Plasticity is on, so the network rewires itself during the run — repeat measurements drift even at a fixed seed.',
    );
  }
  return warnings;
}

function stochasticRelease(circuit: Circuit): boolean {
  for (const synapse of circuit.synapses) {
    if (synapse.enabled && synapse.releaseProbability < 1) return true;
  }
  return false;
}

function noisy(circuit: Circuit): boolean {
  if (circuit.simulation.noise > 0) return true;
  for (const neuron of circuit.neurons) if (neuron.noise > 0) return true;
  return false;
}

function spectrumWarnings(summary: RunSummary, spectrum: SpectrumResult, what: string): string[] {
  const warnings: string[] = [];
  if (summary.spikes < MIN_SPECTRUM_SPIKES) {
    warnings.push(
      `Only ${summary.spikes} spikes in the ${what} window — under ${MIN_SPECTRUM_SPIKES} the spectrum is describing sampling noise, not a rhythm. Drive the network harder or record for longer.`,
    );
  }
  if (summary.measured < MIN_POPULATION_CELLS) {
    warnings.push(
      `The population signal covers ${summary.measured} cell${summary.measured === 1 ? '' : 's'} — a population average over fewer than ${MIN_POPULATION_CELLS} is a description of individuals.`,
    );
  }
  if (spectrum.dominantHz > 0 && spectrum.dominantHz < spectrum.resolutionHz * 2) {
    warnings.push(
      `The peak at ${spectrum.dominantHz.toFixed(1)} Hz is within two bins of this record's ${spectrum.resolutionHz.toFixed(1)} Hz resolution — record for longer before believing it.`,
    );
  }
  return warnings;
}

/* ----------------------------------------------------- protocol: rhythm ----- */

export interface RhythmParams {
  durationMs: number;
  warmupMs: number;
  binMs: number;
  seed: number;
}

export interface CellRate extends CellRef {
  spikes: number;
  rateHz: number;
}

export interface RhythmResult {
  kind: 'rhythm';
  spectrum: SpectrumResult;
  synchrony: SynchronyResult;
  summary: RunSummary;
  /** Population rate in Hz per cell, one sample per bin. */
  rate: Float32Array;
  busiest: readonly CellRate[];
  warnings: readonly string[];
  elapsedMs: number;
}

/**
 * Bin the population spike train, transform it, and say which band the rhythm
 * landed in. This is how a user checks that "make it oscillate at gamma"
 * actually produced gamma rather than a plausible-looking raster.
 */
export async function runRhythm(
  circuit: Circuit,
  params: RhythmParams,
  ctx: ExperimentContext,
): Promise<RhythmResult> {
  if (circuit.neurons.length === 0) {
    throw new Error('There are no neurons to measure.');
  }
  const started = performance.now();
  const totalSteps = stepsFor(params.warmupMs + params.durationMs, circuit.simulation.dt);
  const tracker = new Tracker(totalSteps, ctx);

  const run = await measureRun(
    circuit,
    {
      seed: params.seed,
      warmupMs: params.warmupMs,
      durationMs: params.durationMs,
      binMs: params.binMs,
      label: 'Recording the population',
    },
    ctx,
    tracker,
  );

  const busiest = rankCells(circuit, run.recorder, run.summary.durationMs, 6);
  const warnings = [
    ...circuitWarnings(circuit),
    ...spectrumWarnings(run.summary, run.spectrum, 'recording'),
  ];
  if (run.recorder.dropped > 0) {
    warnings.push(
      `${run.recorder.dropped} spikes overflowed the engine's event ring and are missing from this measurement.`,
    );
  }

  return {
    kind: 'rhythm',
    spectrum: run.spectrum,
    synchrony: run.synchrony,
    summary: run.summary,
    rate: run.rate,
    busiest,
    warnings,
    elapsedMs: performance.now() - started,
  };
}

function rankCells(
  circuit: Circuit,
  recorder: Recorder,
  durationMs: number,
  count: number,
): CellRate[] {
  const seconds = durationMs / 1000;
  const order: number[] = [];
  for (let slot = 0; slot < recorder.perNeuron.length; slot += 1) {
    if (recorder.perNeuron[slot] > 0) order.push(slot);
  }
  order.sort((a, b) => recorder.perNeuron[b] - recorder.perNeuron[a]);
  return order.slice(0, count).map((slot) => ({
    ...cellRef(circuit, slot),
    spikes: recorder.perNeuron[slot],
    rateHz: seconds > 0 ? recorder.perNeuron[slot] / seconds : 0,
  }));
}

/* ----------------------------------------------------- protocol: lesion ----- */

export type LesionTarget = 'selection' | 'hubs' | 'random';

export interface LesionParams {
  target: LesionTarget;
  /** Size of the ablated set for 'hubs' and 'random'. */
  size: number;
  selection: readonly NeuronId[];
  durationMs: number;
  warmupMs: number;
  binMs: number;
  seed: number;
}

export interface LesionMetrics {
  meanRateHz: number;
  synchrony: number | null;
  dominantHz: number;
  band: FrequencyBand | null;
  spikes: number;
  activeCells: number;
  spectrum: SpectrumResult;
  rate: Float32Array;
}

export interface LesionResult {
  kind: 'lesion';
  target: LesionTarget;
  ablated: readonly CellRef[];
  survivors: number;
  control: LesionMetrics;
  lesioned: LesionMetrics;
  summary: RunSummary;
  warnings: readonly string[];
  elapsedMs: number;
}

function metricsOf(run: Measured): LesionMetrics {
  return {
    meanRateHz: run.summary.meanRateHz,
    synchrony: run.synchrony.index,
    dominantHz: run.spectrum.dominantHz,
    band: run.spectrum.band,
    spikes: run.summary.spikes,
    activeCells: run.summary.activeCells,
    spectrum: run.spectrum,
    rate: run.rate,
  };
}

/**
 * Ablate a set of cells and compare against an unlesioned control.
 *
 * Both runs use the same seed and measure the same surviving cells. The seed is
 * what makes the two trajectories comparable at all; the shared measurement set
 * is what stops the removal of a cell from moving the mean by arithmetic rather
 * than by physiology.
 */
export async function runLesion(
  circuit: Circuit,
  params: LesionParams,
  ctx: ExperimentContext,
): Promise<LesionResult> {
  if (circuit.neurons.length === 0) {
    throw new Error('There are no neurons to ablate.');
  }
  const started = performance.now();

  let ablated: number[];
  if (params.target === 'selection') {
    ablated = resolveTargetSlots(circuit, { kind: 'selection' }, params.selection);
    if (ablated.length === 0) {
      throw new Error('Nothing is selected. Select the cells to ablate, or pick hubs or a random sample.');
    }
  } else if (params.target === 'hubs') {
    const probe = createHarness(circuit, params.seed);
    try {
      ablated = topHubSlots(probe.buffers, params.size);
    } finally {
      probe.engine.dispose();
    }
    if (ablated.length === 0) {
      throw new Error('This circuit has no connections, so it has no hubs. Ablate the selection or a random sample instead.');
    }
  } else {
    ablated = randomSlotSample(circuit, params.size, params.seed);
    if (ablated.length === 0) {
      throw new Error('There are no enabled cells to sample.');
    }
  }

  const live: number[] = [];
  for (let i = 0; i < circuit.neurons.length; i += 1) {
    if (circuit.neurons[i].enabled) live.push(i);
  }
  const ablatedSet = new Set(ablated);
  const survivors = live.filter((slot) => !ablatedSet.has(slot));
  if (survivors.length === 0) {
    throw new Error('The lesion removes every cell in the network; there would be nothing left to measure.');
  }

  const perRun = stepsFor(params.warmupMs + params.durationMs, circuit.simulation.dt);
  const tracker = new Tracker(perRun * 2, ctx);

  const shared = {
    seed: params.seed,
    warmupMs: params.warmupMs,
    durationMs: params.durationMs,
    binMs: params.binMs,
    measure: survivors,
  };

  const control = await measureRun(
    circuit,
    { ...shared, label: 'Control run' },
    ctx,
    tracker,
  );
  const lesioned = await measureRun(
    circuit,
    { ...shared, label: 'Lesioned run', ablate: ablated },
    ctx,
    tracker,
  );

  const warnings = [...circuitWarnings(circuit)];
  warnings.push(...spectrumWarnings(control.summary, control.spectrum, 'control'));
  if (noisy(circuit)) {
    warnings.push(
      'This network has per-cell noise. Both runs start from the same seed, but a disabled cell draws no noise, so the two noise streams separate after the first ablated cell — treat small differences as noise, not as effect.',
    );
  }
  if (ablated.length / live.length > 0.5) {
    warnings.push(
      `The lesion removes ${((ablated.length / live.length) * 100).toFixed(0)}% of the live cells; at that scale the comparison is between two different networks rather than one network with a hole in it.`,
    );
  }

  return {
    kind: 'lesion',
    target: params.target,
    ablated: ablated.map((slot) => cellRef(circuit, slot)),
    survivors: survivors.length,
    control: metricsOf(control),
    lesioned: metricsOf(lesioned),
    summary: control.summary,
    warnings,
    elapsedMs: performance.now() - started,
  };
}

/* --------------------------------------------------- protocol: transfer ----- */

export interface TransferParams {
  input: TargetSpec;
  output: TargetSpec;
  /** True when the readout is "everything the drive does not touch". */
  outputExcludesInput: boolean;
  minRateHz: number;
  maxRateHz: number;
  levels: number;
  /** Amplitude of the equivalent 1 ms pulse carried by one input event (pA). */
  amplitudePa: number;
  durationMs: number;
  warmupMs: number;
  seed: number;
}

export interface TransferPoint {
  requestedHz: number;
  /** Events actually delivered per driven cell per second. */
  deliveredHz: number;
  /** Firing rate of the driven cells themselves. */
  inputRateHz: number;
  /** Firing rate of the readout population — the transfer function's output. */
  outputRateHz: number;
  outputSpikes: number;
}

export interface TransferResult {
  kind: 'transfer';
  points: readonly TransferPoint[];
  inputCells: number;
  outputCells: number;
  inputPreview: readonly CellRef[];
  outputPreview: readonly CellRef[];
  overlap: number;
  /** Steepest segment of the curve, in output Hz per input Hz. */
  maxSlope: number;
  maxSlopeAtHz: number;
  /** Output rate at the highest drive level tested. */
  ceilingHz: number;
  amplitudePa: number;
  durationMs: number;
  warnings: readonly string[];
  elapsedMs: number;
}

/**
 * Sweep a Poisson drive into one population and read the rate out of another.
 *
 * Every level runs from the same network seed *and* the same input seed. Sharing
 * the input stream means the event times at 20 Hz are a subset of those at
 * 40 Hz, so a difference between two points on the curve is the rate change and
 * not a different draw — the common-random-numbers trick, and it is what makes
 * an eight-point curve legible instead of jagged.
 */
export async function runTransfer(
  circuit: Circuit,
  params: TransferParams,
  ctx: ExperimentContext,
  selection: readonly NeuronId[],
): Promise<TransferResult> {
  if (circuit.neurons.length === 0) {
    throw new Error('There are no neurons to drive.');
  }

  const started = performance.now();
  const inputSlots = resolveTargetSlots(circuit, params.input, selection);
  if (inputSlots.length === 0) {
    throw new Error('The input population is empty. Choose a population, or select the cells to drive.');
  }
  const outputSlots = resolveTargetSlots(
    circuit,
    params.output,
    selection,
    params.outputExcludesInput ? inputSlots : null,
  );
  if (outputSlots.length === 0) {
    throw new Error('The readout population is empty.');
  }

  const inputSet = new Set(inputSlots);
  const overlap = outputSlots.reduce((count, slot) => (inputSet.has(slot) ? count + 1 : count), 0);

  const levels = Math.max(2, Math.round(params.levels));
  const dt = circuit.simulation.dt;
  const perLevel = stepsFor(params.warmupMs + params.durationMs, dt);
  const tracker = new Tracker(perLevel * levels, ctx);

  const points: TransferPoint[] = [];
  for (let level = 0; level < levels; level += 1) {
    const requestedHz =
      params.minRateHz + ((params.maxRateHz - params.minRateHz) * level) / (levels - 1);
    // Per-step Bernoulli probability of one event on one driven cell.
    const probability = Math.min(1, Math.max(0, (requestedHz * dt) / 1000));
    // Charge is held constant across timesteps by scaling the impulse by 1/dt.
    const impulse = (params.amplitudePa * EVENT_MS) / dt;
    const rng = new Rng(params.seed);
    let delivered = 0;

    const run = await measureRun(
      circuit,
      {
        seed: params.seed,
        warmupMs: params.warmupMs,
        durationMs: params.durationMs,
        binMs: 1,
        measure: outputSlots,
        label: `Drive ${requestedHz.toFixed(0)} Hz · level ${level + 1} of ${levels}`,
        onStep: (engine, recording) => {
          if (probability <= 0) return;
          for (const slot of inputSlots) {
            if (rng.next() >= probability) continue;
            engine.poke(slot, impulse);
            if (recording) delivered += 1;
          }
        },
      },
      ctx,
      tracker,
    );

    const seconds = run.summary.durationMs / 1000;
    const inputSpikes = run.recorder.spikesOf(inputSlots);
    points.push({
      requestedHz,
      deliveredHz: seconds > 0 ? delivered / (inputSlots.length * seconds) : 0,
      inputRateHz: seconds > 0 ? inputSpikes / (inputSlots.length * seconds) : 0,
      outputRateHz: run.summary.meanRateHz,
      outputSpikes: run.summary.spikes,
    });
  }

  let maxSlope = 0;
  let maxSlopeAtHz = 0;
  for (let i = 1; i < points.length; i += 1) {
    const dx = points[i].deliveredHz - points[i - 1].deliveredHz;
    if (!(dx > 0)) continue;
    const slope = (points[i].outputRateHz - points[i - 1].outputRateHz) / dx;
    if (slope > maxSlope) {
      maxSlope = slope;
      maxSlopeAtHz = (points[i].deliveredHz + points[i - 1].deliveredHz) / 2;
    }
  }

  const warnings = [...circuitWarnings(circuit)];
  if (overlap > 0) {
    warnings.push(
      `${overlap} cell${overlap === 1 ? ' is' : 's are'} in both the driven set and the readout, so part of this curve is the drive measuring itself.`,
    );
  }
  const ceiling = points[points.length - 1].outputRateHz;
  if (ceiling <= 0) {
    warnings.push(
      'The readout population never fired at any drive level. Raise the event amplitude, or check that the driven cells project to the readout.',
    );
  }
  if ((params.maxRateHz * dt) / 1000 >= 1) {
    warnings.push(
      `At ${params.maxRateHz.toFixed(0)} Hz the per-step event probability saturates at one event per timestep, so the top of the sweep delivers less than it asks for.`,
    );
  }

  return {
    kind: 'transfer',
    points,
    inputCells: inputSlots.length,
    outputCells: outputSlots.length,
    inputPreview: inputSlots.slice(0, 8).map((slot) => cellRef(circuit, slot)),
    outputPreview: outputSlots.slice(0, 8).map((slot) => cellRef(circuit, slot)),
    overlap,
    maxSlope,
    maxSlopeAtHz,
    ceilingHz: ceiling,
    amplitudePa: params.amplitudePa,
    durationMs: params.durationMs,
    warnings,
    elapsedMs: performance.now() - started,
  };
}

/* ----------------------------------------------- protocol: perturbation ----- */

export interface PerturbationParams {
  /** Cell to perturb; null falls back to the highest-degree cell. */
  target: NeuronId | null;
  amplitudePa: number;
  warmupMs: number;
  durationMs: number;
  sampleMs: number;
  seed: number;
}

export interface PerturbationResult {
  kind: 'perturbation';
  cell: CellRef;
  /** Milliseconds since the perturbation, one entry per sample. */
  times: Float32Array;
  /** RMS membrane-potential difference across the network (mV). */
  distance: Float32Array;
  /** Share of cells whose spike output differed inside each sample. */
  spikeDivergence: Float32Array;
  samples: number;
  /** Fitted exponential growth rate of the distance, per second. */
  lambdaPerSecond: number | null;
  fitFromMs: number;
  fitToMs: number;
  peakDistance: number;
  finalDistance: number;
  /** Cells whose spike train differed from the control at any point. */
  divergedCells: number;
  /** Spikes in the perturbed run minus spikes in the control. */
  spikeDelta: number;
  /** True when the injected pulse actually made the target fire an extra spike. */
  evoked: boolean;
  neurons: number;
  amplitudePa: number;
  warnings: readonly string[];
  elapsedMs: number;
}

interface Fit {
  slopePerMs: number;
  fromMs: number;
  toMs: number;
}

/**
 * Least squares of log distance against time over the growing, unsaturated part
 * of the trace. Once the two trajectories are as far apart as two unrelated
 * states can be, the distance stops growing for reasons that have nothing to do
 * with the divergence rate, so the fit stops at half the peak.
 */
function fitGrowth(times: Float32Array, values: Float32Array, count: number): Fit | null {
  let peak = 0;
  for (let i = 0; i < count; i += 1) if (values[i] > peak) peak = values[i];
  if (!(peak > 0)) return null;
  const ceiling = peak * 0.5;

  let start = -1;
  let end = -1;
  for (let i = 0; i < count; i += 1) {
    if (values[i] <= 0) {
      if (start >= 0) break;
      continue;
    }
    if (start < 0) start = i;
    if (values[i] > ceiling) break;
    end = i;
  }
  if (start < 0 || end - start + 1 < 4) return null;

  const n = end - start + 1;
  let sx = 0;
  let sy = 0;
  for (let i = start; i <= end; i += 1) {
    sx += times[i];
    sy += Math.log(values[i]);
  }
  const mx = sx / n;
  const my = sy / n;
  let sxy = 0;
  let sxx = 0;
  for (let i = start; i <= end; i += 1) {
    const dx = times[i] - mx;
    sxy += dx * (Math.log(values[i]) - my);
    sxx += dx * dx;
  }
  if (!(sxx > 0)) return null;
  return { slopePerMs: sxy / sxx, fromMs: times[start], toMs: times[end] };
}

/**
 * Run the same network twice from the same seed, inject one extra spike into one
 * cell in the second run, and watch the two trajectories separate.
 *
 * On a recurrent network this is the criticality probe: a positive growth rate
 * means one spike reorganises the whole population, a negative one means the
 * network forgets the perturbation. The two engines are stepped in lockstep so
 * the comparison never needs a stored trajectory.
 */
export async function runPerturbation(
  circuit: Circuit,
  params: PerturbationParams,
  ctx: ExperimentContext,
): Promise<PerturbationResult> {
  if (circuit.neurons.length === 0) {
    throw new Error('There are no neurons to perturb.');
  }
  const started = performance.now();

  const control = createHarness(circuit, params.seed);
  const perturbed = createHarness(circuit, params.seed);
  try {
    const dt = control.dt;
    const count = control.count;
    const live = liveMask(control.buffers);

    let slot = -1;
    if (params.target !== null) {
      slot = circuit.neurons.findIndex((neuron) => neuron.id === params.target);
    }
    if (slot < 0) {
      const hubs = topHubSlots(control.buffers, 1);
      slot = hubs.length > 0 ? hubs[0] : 0;
    }
    if (live[slot] !== 1) {
      throw new Error('The cell to perturb is disabled, so an injected spike would go nowhere.');
    }

    const warmupSteps = stepsFor(params.warmupMs, dt);
    const observeSteps = stepsFor(params.durationMs, dt);
    const perSample = Math.max(1, stepsFor(params.sampleMs, dt));
    const samples = Math.floor(observeSteps / perSample);
    if (samples < 4) {
      throw new Error('The observation window is too short to trace a divergence. Lengthen it or shorten the sample interval.');
    }

    // One extra step carries the perturbation itself; the observation loop runs
    // whole samples, which is at most `perSample - 1` steps short of the window.
    const tracker = new Tracker(warmupSteps + 1 + samples * perSample, ctx);

    tracker.stage('Settling both runs');
    await drive(warmupSteps, ctx, tracker, () => {
      control.engine.stepOnce();
      perturbed.engine.stepOnce();
    });

    // Charge is held constant across timesteps by scaling the impulse by 1/dt,
    // so the same amplitude perturbs the same amount whatever `dt` is.
    perturbed.engine.poke(slot, (params.amplitudePa * EVENT_MS) / dt);
    control.engine.stepOnce();
    perturbed.engine.stepOnce();
    tracker.step();

    const controlNeurons = control.buffers.neurons;
    const perturbedNeurons = perturbed.buffers.neurons;
    const evoked = perturbedNeurons.spike[slot] === 1 && controlNeurons.spike[slot] === 0;

    const times = new Float32Array(samples);
    const distance = new Float32Array(samples);
    const spikeDivergence = new Float32Array(samples);
    const everDiverged = new Uint8Array(count);
    let liveCells = 0;
    for (let i = 0; i < count; i += 1) if (live[i] === 1) liveCells += 1;

    let sample = 0;
    let mismatches = 0;
    let controlSpikes = 0;
    let perturbedSpikes = 0;

    tracker.stage('Tracing the divergence');
    await drive(samples * perSample, ctx, tracker, (index) => {
      control.engine.stepOnce();
      perturbed.engine.stepOnce();

      for (let i = 0; i < count; i += 1) {
        if (live[i] !== 1) continue;
        const a = controlNeurons.spike[i];
        const b = perturbedNeurons.spike[i];
        controlSpikes += a;
        perturbedSpikes += b;
        if (a === b) continue;
        mismatches += 1;
        everDiverged[i] = 1;
      }

      if ((index + 1) % perSample !== 0 || sample >= samples) return;

      let squared = 0;
      for (let i = 0; i < count; i += 1) {
        if (live[i] !== 1) continue;
        const d = perturbedNeurons.v[i] - controlNeurons.v[i];
        squared += d * d;
      }
      times[sample] = (index + 1) * dt;
      distance[sample] = liveCells > 0 ? Math.sqrt(squared / liveCells) : 0;
      spikeDivergence[sample] = liveCells > 0 ? mismatches / (liveCells * perSample) : 0;
      mismatches = 0;
      sample += 1;
    });

    let peak = 0;
    for (let i = 0; i < samples; i += 1) if (distance[i] > peak) peak = distance[i];
    let divergedCells = 0;
    for (let i = 0; i < count; i += 1) if (everDiverged[i] === 1) divergedCells += 1;

    const fit = fitGrowth(times, distance, samples);

    const warnings = [...circuitWarnings(circuit)];
    if (!evoked) {
      warnings.push(
        'The injected pulse did not produce an extra spike in the target — it may have arrived while the cell was refractory, or the amplitude may be too small. Raise the amplitude or move the perturbation later.',
      );
    }
    if (stochasticRelease(circuit)) {
      warnings.push(
        'Some synapses release stochastically. A release draw is taken per spike, so the extra spike shifts the random stream itself and part of the measured divergence is that shift rather than the network dynamics.',
      );
    }
    if (peak <= 0) {
      warnings.push(
        'The two runs never separated at all: the perturbation had no downstream effect within this window.',
      );
    }

    return {
      kind: 'perturbation',
      cell: cellRef(circuit, slot),
      times,
      distance,
      spikeDivergence,
      samples,
      lambdaPerSecond: fit === null ? null : fit.slopePerMs * 1000,
      fitFromMs: fit === null ? 0 : fit.fromMs,
      fitToMs: fit === null ? 0 : fit.toMs,
      peakDistance: peak,
      finalDistance: distance[samples - 1],
      divergedCells,
      spikeDelta: perturbedSpikes - controlSpikes,
      evoked,
      neurons: liveCells,
      amplitudePa: params.amplitudePa,
      warnings,
      elapsedMs: performance.now() - started,
    };
  } finally {
    control.engine.dispose();
    perturbed.engine.dispose();
  }
}

/* ------------------------------------------------------------------ union -- */

export type ExperimentResult =
  | RhythmResult
  | LesionResult
  | TransferResult
  | PerturbationResult;
