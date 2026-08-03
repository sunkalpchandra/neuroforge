/**
 * Single-cell electrophysiology protocols.
 *
 * Each function here is one of the characterisations a physiologist runs at a
 * rig: an F-I curve, an I-V curve, a membrane time constant, spike-frequency
 * adaptation, a paired-pulse sweep and a rheobase search. Every one builds its
 * own `SimulationEngine` over a cloned, cut-down document and disposes it
 * afterwards, so nothing here can perturb the network the user is watching.
 *
 * The preparation is deliberately reductive: only the cell under study (and, for
 * paired-pulse, its partner across one synapse) is loaded, stimuli and probes
 * are removed, and stochastic drive is silenced. That is the in-vitro analogy —
 * synaptic blockers and a quiet bath — and it is what makes a number repeatable.
 * The cell keeps its own constant bias current, because that is part of the cell
 * as the modeller specified it, and every current reported below is a *command*
 * current on top of that holding level.
 *
 * Units follow the simulation core throughout: mV, ms, pA, pF, nS.
 */

import { SimulationEngine } from '@neuroforge/simulation';
import type {
  Circuit,
  Neuron,
  NeuronModelKind,
  NeuronParams,
  Synapse,
} from '@neuroforge/shared';

/* ------------------------------------------------------------------ bounds -- */

/** Widest sweep a single run is allowed to expand to. */
export const MAX_LEVELS = 128;
/** Widest paired-pulse interval sweep. */
export const MAX_INTERVALS = 64;
/** Hard ceiling on bisection depth, independent of the requested tolerance. */
const MAX_BISECTIONS = 40;
/** Doublings tried when searching for a stimulus that reliably evokes a spike. */
const MAX_STIM_DOUBLINGS = 18;
/** First stimulus amplitude tried by the paired-pulse calibration (pA). */
const STIM_SEED_PA = 50;
/** Quiescent period held before the first stimulus of a paired-pulse trial (ms). */
const PPR_SETTLE_MS = 50;

export const MIN_DT = 0.005;
export const MAX_DT = 1;

/* ------------------------------------------------------------------ errors -- */

/** Thrown when the caller's AbortSignal fires between conditions. */
export class ProtocolAbort extends Error {
  constructor() {
    super('Protocol cancelled');
    this.name = 'ProtocolAbort';
  }
}

/** A protocol that cannot produce a meaningful answer for this cell. */
export class ProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ProtocolError';
  }
}

/* ------------------------------------------------------------------- types -- */

export type ProtocolKind = 'fi' | 'iv' | 'tau' | 'adaptation' | 'ppr' | 'rheobase';

export interface RunOptions {
  signal?: AbortSignal;
  /** Called once before each condition, so a caller can paint progress. */
  onProgress?: (done: number, total: number, label: string) => void;
}

/** Provenance attached to every result. */
export interface ProtocolMeta {
  neuronId: string;
  label: string;
  seed: number;
  model: NeuronModelKind;
  /** The cell's own constant current, which every command current adds to (pA). */
  holdingPa: number;
  dt: number;
  /** Wall-clock cost of the whole protocol. */
  elapsedMs: number;
  /** Total simulated time integrated across every condition. */
  simulatedMs: number;
}

export interface LinearFit {
  slope: number;
  intercept: number;
  /** Coefficient of determination of the fit. */
  r2: number;
  n: number;
}

/* ------------------------------------------------------------------ F-I ---- */

export interface FiParams {
  fromPa: number;
  toPa: number;
  stepPa: number;
  /** Discarded transient before counting begins. */
  settleMs: number;
  /** Window the steady-state rate is counted over. */
  measureMs: number;
  dt: number;
}

export interface FiPoint {
  currentPa: number;
  spikes: number;
  rateHz: number;
  /** Time from step onset to the first spike, or null if the cell stayed silent. */
  latencyMs: number | null;
  /** Mean membrane potential over the measurement window. */
  meanMv: number;
}

export interface FiResult {
  kind: 'fi';
  meta: ProtocolMeta;
  params: FiParams;
  points: readonly FiPoint[];
  /** Lowest swept current that produced at least one spike (pA). */
  rheobasePa: number | null;
  /** Slope of the suprathreshold branch (Hz/pA). */
  gainHzPerPa: number | null;
  fit: LinearFit | null;
  maxRateHz: number;
  csv: string;
}

/* ------------------------------------------------------------------ I-V ---- */

export interface IvParams {
  fromPa: number;
  toPa: number;
  stepPa: number;
  settleMs: number;
  measureMs: number;
  dt: number;
}

export interface IvPoint {
  currentPa: number;
  steadyMv: number;
  /** A level that fired is not subthreshold and takes no part in the fit. */
  spiked: boolean;
}

export interface IvResult {
  kind: 'iv';
  meta: ProtocolMeta;
  params: IvParams;
  points: readonly IvPoint[];
  /** Input resistance from the subthreshold slope (MΩ). */
  inputResistanceMohm: number | null;
  /** Membrane potential the fit extrapolates to at zero command current (mV). */
  interceptMv: number | null;
  fit: LinearFit | null;
  excluded: number;
  csv: string;
}

/* ------------------------------------------------------------------ tau ---- */

export interface TauParams {
  /** Hyperpolarising command amplitude; must be negative (pA). */
  amplitudePa: number;
  baselineMs: number;
  stepMs: number;
  dt: number;
}

export interface TauResult {
  kind: 'tau';
  meta: ProtocolMeta;
  params: TauParams;
  /** Time relative to step onset (ms); negative during the baseline. */
  traceT: Float32Array;
  traceV: Float32Array;
  /** Index in the trace at which the step begins. */
  onsetIndex: number;
  tauMs: number;
  baselineMv: number;
  steadyMv: number;
  deflectionMv: number;
  fitR2: number;
  fitPoints: number;
  /** cm/gL where the model defines it, else null. */
  analyticTauMs: number | null;
  /** Signed percentage error of the fitted tau against the analytic one. */
  errorPercent: number | null;
  csv: string;
}

/* ----------------------------------------------------------- adaptation ---- */

export interface AdaptationParams {
  amplitudePa: number;
  settleMs: number;
  durationMs: number;
  dt: number;
}

export interface AdaptationResult {
  kind: 'adaptation';
  meta: ProtocolMeta;
  params: AdaptationParams;
  /** Spike times relative to step onset (ms). */
  spikeTimesMs: readonly number[];
  isisMs: readonly number[];
  /** Last ISI over first ISI. Above 1 is adaptation, below 1 is facilitation. */
  adaptationIndex: number | null;
  /** 1000 / first ISI. */
  instantaneousHz: number | null;
  /** Reciprocal of the mean of the final fifth of the ISI series. */
  steadyHz: number | null;
  meanHz: number;
  spikeCount: number;
  /** Time from step onset to the first spike (ms). */
  latencyMs: number | null;
  csv: string;
}

/* ---------------------------------------------------------- paired pulse --- */

export interface PprParams {
  synapseId: string;
  fromMs: number;
  toMs: number;
  stepMs: number;
  /** Recording window after the second stimulus. */
  windowMs: number;
  /** Repeats averaged per interval; only matters when release is stochastic. */
  trials: number;
  dt: number;
}

export interface PprPoint {
  intervalMs: number;
  /** Peak conductance of the first response (nS). */
  peak1Ns: number;
  /** Peak of the isolated second response, first response subtracted (nS). */
  peak2Ns: number;
  ratio: number | null;
  /** Trials in which both stimuli evoked exactly the spikes they should. */
  evoked: number;
  failure: string | null;
}

export interface PprSynapseInfo {
  synapseId: string;
  sourceId: string;
  targetId: string;
  targetLabel: string;
  targetSeed: number;
  receptor: Synapse['receptor'];
  weightNs: number;
  delayMs: number;
  releaseProbability: number;
  stpEnabled: boolean;
  stpU: number;
  tauRecMs: number;
  tauFacilMs: number;
}

export interface PprResult {
  kind: 'ppr';
  meta: ProtocolMeta;
  params: PprParams;
  synapse: PprSynapseInfo;
  points: readonly PprPoint[];
  /** Command amplitude the calibration settled on for the stimulus pulse (pA). */
  stimulusPa: number;
  stimulusMs: number;
  csv: string;
}

/* ------------------------------------------------------------- rheobase ---- */

export interface RheobaseParams {
  lowPa: number;
  highPa: number;
  /** Bracket width the search converges to (pA). */
  tolerancePa: number;
  /** How long a level is held before it is declared subthreshold. */
  windowMs: number;
  dt: number;
}

export interface RheobaseProbe {
  iteration: number;
  currentPa: number;
  spiked: boolean;
  lowPa: number;
  highPa: number;
}

export interface RheobaseResult {
  kind: 'rheobase';
  meta: ProtocolMeta;
  params: RheobaseParams;
  /** Smallest current the search proved suprathreshold (pA). */
  rheobasePa: number;
  bracketLowPa: number;
  bracketHighPa: number;
  iterations: number;
  /** Latency to the first spike at the returned rheobase (ms). */
  latencyMs: number | null;
  /** Set when the lower bound already fired, so the true value is below it. */
  boundedBelow: boolean;
  probes: readonly RheobaseProbe[];
  csv: string;
}

export type ProtocolResult =
  | FiResult
  | IvResult
  | TauResult
  | AdaptationResult
  | PprResult
  | RheobaseResult;

/* ----------------------------------------------------------------- shared -- */

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function stepsFor(ms: number, dt: number): number {
  if (!(dt > 0) || !(ms > 0)) return 0;
  return Math.max(1, Math.round(ms / dt));
}

/**
 * Expand a from/to/step sweep into concrete levels.
 *
 * The count is capped rather than the range truncated: a user who types a step
 * of 0.01 across 500 pA gets a coarser sweep of the range they asked for, not
 * the first 128 levels of it and silence about the rest.
 */
export function sweepLevels(from: number, to: number, step: number, cap: number): number[] {
  const lo = Math.min(from, to);
  const hi = Math.max(from, to);
  if (!(hi > lo)) return [lo];
  const width = Math.abs(step);
  const requested = width > 0 ? Math.floor((hi - lo) / width) + 1 : 2;
  const count = clamp(requested, 2, cap);
  const spacing = (hi - lo) / (count - 1);
  const out = new Array<number>(count);
  for (let i = 0; i < count; i += 1) out[i] = lo + i * spacing;
  // The endpoint is written exactly rather than accumulated, so a sweep to
  // 500 pA ends at 500 and not at 499.99999999999994.
  out[count - 1] = hi;
  return out;
}

/** Number of conditions `sweepLevels` will actually produce, for a live readout. */
export function sweepCount(from: number, to: number, step: number, cap: number): number {
  return sweepLevels(from, to, step, cap).length;
}

function linearFit(xs: readonly number[], ys: readonly number[]): LinearFit | null {
  const n = xs.length;
  if (n < 2 || ys.length !== n) return null;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i += 1) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  if (!(sxx > 0)) return null;
  const slope = sxy / sxx;
  return {
    slope,
    intercept: my - slope * mx,
    r2: syy > 0 ? (sxy * sxy) / (sxx * syy) : 1,
    n,
  };
}

/** Read through a call so control-flow analysis cannot cache the last answer. */
function isAborted(signal: AbortSignal | undefined): boolean {
  return signal !== undefined && signal.aborted;
}

async function condition(
  options: RunOptions,
  done: number,
  total: number,
  label: string,
): Promise<void> {
  if (isAborted(options.signal)) throw new ProtocolAbort();
  options.onProgress?.(done, total, label);
  // A macrotask, not a microtask: the microtask queue drains before the browser
  // paints, so awaiting one would advance the bar without ever showing it.
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });
  if (isAborted(options.signal)) throw new ProtocolAbort();
}

function findNeuron(circuit: Circuit, neuronId: string): Neuron {
  const neuron = circuit.neurons.find((candidate) => candidate.id === neuronId);
  if (neuron === undefined) {
    throw new ProtocolError('The selected cell is no longer in the document.');
  }
  return neuron;
}

function cellLabel(neuron: Neuron): string {
  return neuron.label.length > 0 ? neuron.label : neuron.id.slice(0, 8);
}

/**
 * Build the reduced document an isolated engine is loaded with.
 *
 * Every entity handed over is a structured clone, so the experiment's engine and
 * the live one share no mutable state at all — an engine rewrites weights and
 * parameters in place, and a shared object would let a protocol corrupt the
 * document it is measuring.
 */
function preparation(
  circuit: Circuit,
  neurons: readonly Neuron[],
  synapses: readonly Synapse[],
  dt: number,
): Circuit {
  return {
    ...circuit,
    // Per-neuron noise is the model's stand-in for uncontrolled synaptic
    // bombardment. A characterisation run wants it blocked, or every number
    // below becomes a sample from a distribution rather than a measurement.
    neurons: neurons.map((neuron) => ({ ...structuredClone(neuron), noise: 0 })),
    synapses: synapses.map((synapse) => structuredClone(synapse)),
    populations: [],
    projections: [],
    stimuli: [],
    probes: [],
    simulation: {
      ...circuit.simulation,
      dt: clamp(dt, MIN_DT, MAX_DT),
      noise: 0,
      speed: 1,
      plasticityEnabled: false,
      backend: 'cpu',
    },
  };
}

interface CellRig {
  engine: SimulationEngine;
  slot: number;
  /** The cell's document bias, which command currents are added to. */
  holdingPa: number;
}

function openCell(circuit: Circuit, neuron: Neuron, dt: number): CellRig {
  const engine = new SimulationEngine();
  try {
    engine.load(preparation(circuit, [neuron], [], dt));
    const slot = engine.slotOf(neuron.id);
    if (slot < 0) throw new ProtocolError('The cell could not be loaded into an engine.');
    return { engine, slot, holdingPa: engine.buffers.neurons.bias[slot] };
  } catch (cause) {
    engine.dispose();
    throw cause;
  }
}

function metaFor(
  neuron: Neuron,
  rig: CellRig,
  dt: number,
  startedAt: number,
  simulatedMs: number,
): ProtocolMeta {
  return {
    neuronId: neuron.id,
    label: cellLabel(neuron),
    seed: neuron.morphology.seed,
    model: neuron.params.kind,
    holdingPa: rig.holdingPa,
    dt,
    elapsedMs: performance.now() - startedAt,
    simulatedMs,
  };
}

/** Hold a command current and integrate, discarding everything measured. */
function settle(engine: SimulationEngine, ms: number): void {
  const steps = stepsFor(ms, engine.settings.dt);
  for (let i = 0; i < steps; i += 1) engine.stepOnce();
}

/** Integrate for `ms`, appending the simulation time of every spike on `slot`. */
function collectSpikes(
  engine: SimulationEngine,
  slot: number,
  ms: number,
  out: number[],
): void {
  const steps = stepsFor(ms, engine.settings.dt);
  const buffers = engine.buffers;
  for (let i = 0; i < steps; i += 1) {
    engine.stepOnce();
    if (buffers.neurons.spike[slot] === 1) out.push(buffers.time);
  }
}

/* -------------------------------------------------------------------- CSV -- */

function csvNumber(value: number): string {
  if (!Number.isFinite(value)) return '';
  return String(Number(value.toPrecision(8)));
}

function csvHeader(meta: ProtocolMeta, title: string, extra: readonly string[] = []): string[] {
  return [
    `# NeuroForge ${title}`,
    `# cell: ${meta.label} (${meta.neuronId})`,
    `# model: ${meta.model}`,
    `# holding current: ${csvNumber(meta.holdingPa)} pA`,
    `# dt: ${csvNumber(meta.dt)} ms`,
    ...extra.map((line) => `# ${line}`),
  ];
}

/* ------------------------------------------------------------- 1. F-I ------ */

/**
 * Sweep injected current and report the steady-state rate at each level.
 *
 * Rate is counted over the measurement window only; the settling period exists
 * so that adaptation has run its course before anything is counted, which is the
 * difference between a steady-state F-I curve and an onset one.
 */
export async function runFiCurve(
  circuit: Circuit,
  neuronId: string,
  params: FiParams,
  options: RunOptions = {},
): Promise<FiResult> {
  const neuron = findNeuron(circuit, neuronId);
  const startedAt = performance.now();
  const levels = sweepLevels(params.fromPa, params.toPa, params.stepPa, MAX_LEVELS);
  const rig = openCell(circuit, neuron, params.dt);
  const points: FiPoint[] = [];
  let simulatedMs = 0;

  try {
    const { engine, slot, holdingPa } = rig;
    const buffers = engine.buffers;

    for (let i = 0; i < levels.length; i += 1) {
      const current = levels[i];
      await condition(options, i, levels.length, `${current.toFixed(1)} pA`);

      engine.reset();
      buffers.neurons.bias[slot] = holdingPa + current;

      // Latency is watched from the moment the command steps on, which is before
      // the settling period rather than after it — the first spike of the
      // response is the one a physiologist times, and it has almost always
      // happened by the time the rate is steady enough to count.
      let latencyMs: number | null = null;
      const settleSteps = stepsFor(params.settleMs, engine.settings.dt);
      for (let s = 0; s < settleSteps; s += 1) {
        engine.stepOnce();
        if (latencyMs === null && buffers.neurons.spike[slot] === 1) latencyMs = buffers.time;
      }

      let count = 0;
      const measureSteps = stepsFor(params.measureMs, engine.settings.dt);
      let voltageSum = 0;
      for (let s = 0; s < measureSteps; s += 1) {
        engine.stepOnce();
        voltageSum += buffers.neurons.v[slot];
        if (buffers.neurons.spike[slot] === 1) {
          count += 1;
          if (latencyMs === null) latencyMs = buffers.time;
        }
      }
      simulatedMs += params.settleMs + params.measureMs;

      const seconds = params.measureMs / 1000;
      points.push({
        currentPa: current,
        spikes: count,
        rateHz: seconds > 0 ? count / seconds : 0,
        latencyMs,
        meanMv: measureSteps > 0 ? voltageSum / measureSteps : buffers.neurons.v[slot],
      });
    }
  } finally {
    rig.engine.dispose();
  }

  // Rheobase is the lowest level that fired *at all*, which can sit one level
  // below the lowest that still fires once adaptation has settled; the gain is
  // fitted only over levels with a steady rate, since a single onset spike says
  // nothing about a slope.
  const firstSpiking = points.find((point) => point.latencyMs !== null);
  const rheobasePa = firstSpiking === undefined ? null : firstSpiking.currentPa;
  const firing = points.filter((point) => point.spikes > 0);
  const fit = linearFit(
    firing.map((point) => point.currentPa),
    firing.map((point) => point.rateHz),
  );
  const maxRateHz = points.reduce((max, point) => Math.max(max, point.rateHz), 0);
  const meta = metaFor(neuron, rig, params.dt, startedAt, simulatedMs);

  const csv = [
    ...csvHeader(meta, 'F-I curve', [
      `settle: ${csvNumber(params.settleMs)} ms, measure: ${csvNumber(params.measureMs)} ms`,
      rheobasePa === null ? 'rheobase: not reached' : `rheobase: ${csvNumber(rheobasePa)} pA`,
      fit === null ? 'gain: undefined' : `gain: ${csvNumber(fit.slope)} Hz/pA`,
    ]),
    'current_pA,spikes,rate_Hz,first_spike_latency_ms,mean_voltage_mV',
    ...points.map((point) =>
      [
        csvNumber(point.currentPa),
        String(point.spikes),
        csvNumber(point.rateHz),
        point.latencyMs === null ? '' : csvNumber(point.latencyMs),
        csvNumber(point.meanMv),
      ].join(','),
    ),
  ].join('\n');

  return {
    kind: 'fi',
    meta,
    params,
    points,
    rheobasePa,
    gainHzPerPa: fit === null ? null : fit.slope,
    fit,
    maxRateHz,
    csv,
  };
}

/* ------------------------------------------------------------- 2. I-V ------ */

/**
 * Subthreshold I-V relation and the input resistance its slope implies.
 *
 * Any level that fired is recorded but excluded from the fit: once a spike
 * happens the mean potential is dominated by resets and the point no longer
 * lies on the passive line.
 */
export async function runIvCurve(
  circuit: Circuit,
  neuronId: string,
  params: IvParams,
  options: RunOptions = {},
): Promise<IvResult> {
  const neuron = findNeuron(circuit, neuronId);
  const startedAt = performance.now();
  const levels = sweepLevels(params.fromPa, params.toPa, params.stepPa, MAX_LEVELS);
  const rig = openCell(circuit, neuron, params.dt);
  const points: IvPoint[] = [];
  let simulatedMs = 0;

  try {
    const { engine, slot, holdingPa } = rig;
    const buffers = engine.buffers;

    for (let i = 0; i < levels.length; i += 1) {
      const current = levels[i];
      await condition(options, i, levels.length, `${current.toFixed(1)} pA`);

      engine.reset();
      buffers.neurons.bias[slot] = holdingPa + current;
      settle(engine, params.settleMs);
      let spiked = buffers.neurons.spikeCount[slot] > 0;

      const measureSteps = stepsFor(params.measureMs, engine.settings.dt);
      let sum = 0;
      for (let s = 0; s < measureSteps; s += 1) {
        engine.stepOnce();
        sum += buffers.neurons.v[slot];
        if (buffers.neurons.spike[slot] === 1) spiked = true;
      }
      simulatedMs += params.settleMs + params.measureMs;

      points.push({
        currentPa: current,
        steadyMv: measureSteps > 0 ? sum / measureSteps : buffers.neurons.v[slot],
        spiked,
      });
    }
  } finally {
    rig.engine.dispose();
  }

  const passive = points.filter((point) => !point.spiked);
  const fit = linearFit(
    passive.map((point) => point.currentPa),
    passive.map((point) => point.steadyMv),
  );
  const meta = metaFor(neuron, rig, params.dt, startedAt, simulatedMs);
  const excluded = points.length - passive.length;
  // dV/dI in mV/pA is a resistance in gigaohms; the readout is in megaohms.
  const inputResistanceMohm = fit === null ? null : fit.slope * 1000;

  const csv = [
    ...csvHeader(meta, 'I-V curve', [
      `settle: ${csvNumber(params.settleMs)} ms, measure: ${csvNumber(params.measureMs)} ms`,
      `levels excluded for spiking: ${excluded}`,
      inputResistanceMohm === null
        ? 'input resistance: undefined'
        : `input resistance: ${csvNumber(inputResistanceMohm)} MOhm`,
    ]),
    'current_pA,steady_voltage_mV,spiked',
    ...points.map((point) =>
      [csvNumber(point.currentPa), csvNumber(point.steadyMv), point.spiked ? '1' : '0'].join(','),
    ),
  ].join('\n');

  return {
    kind: 'iv',
    meta,
    params,
    points,
    inputResistanceMohm,
    interceptMv: fit === null ? null : fit.intercept,
    fit,
    excluded,
    csv,
  };
}

/* ------------------------------------------------------- 3. time constant -- */

/** cm/gL for the models that state both, in ms. */
function analyticTau(params: NeuronParams): number | null {
  if (params.kind === 'lif' || params.kind === 'adex') {
    return params.gL > 0 ? params.cm / params.gL : null;
  }
  return null;
}

/**
 * Membrane time constant from the decay of a small hyperpolarising step.
 *
 * The steady level is measured first and subtracted, then the remaining
 * deviation is fitted in the log domain, which turns the exponential into a
 * straight line and makes the fit a single least-squares pass rather than an
 * iterative solve. Samples below 2% of the deflection are dropped: they carry
 * almost no information about the slope but plenty of float noise.
 */
export async function runMembraneTau(
  circuit: Circuit,
  neuronId: string,
  params: TauParams,
  options: RunOptions = {},
): Promise<TauResult> {
  if (!(params.amplitudePa < 0)) {
    throw new ProtocolError(
      'The step must be hyperpolarising — use a negative amplitude, or the cell will spike instead of decaying.',
    );
  }

  const neuron = findNeuron(circuit, neuronId);
  const startedAt = performance.now();
  const rig = openCell(circuit, neuron, params.dt);

  let traceT: Float32Array;
  let traceV: Float32Array;
  let onsetIndex: number;
  let simulatedMs: number;

  try {
    const { engine, slot, holdingPa } = rig;
    const buffers = engine.buffers;
    const dt = engine.settings.dt;
    const baselineSteps = stepsFor(params.baselineMs, dt);
    const stepSteps = stepsFor(params.stepMs, dt);
    if (stepSteps < 8) {
      throw new ProtocolError('The step is too short to fit a decay; lengthen it or reduce dt.');
    }

    await condition(options, 0, 1, 'membrane step');

    engine.reset();
    buffers.neurons.bias[slot] = holdingPa;

    const total = baselineSteps + stepSteps;
    traceT = new Float32Array(total);
    traceV = new Float32Array(total);
    onsetIndex = baselineSteps;

    for (let s = 0; s < baselineSteps; s += 1) {
      engine.stepOnce();
      traceT[s] = (s + 1 - baselineSteps) * dt;
      traceV[s] = buffers.neurons.v[slot];
    }

    buffers.neurons.bias[slot] = holdingPa + params.amplitudePa;
    for (let s = 0; s < stepSteps; s += 1) {
      engine.stepOnce();
      traceT[baselineSteps + s] = (s + 1) * dt;
      traceV[baselineSteps + s] = buffers.neurons.v[slot];
    }
    simulatedMs = params.baselineMs + params.stepMs;

    if (buffers.neurons.spikeCount[slot] > 0) {
      throw new ProtocolError(
        'The cell fired during the step, so no passive decay exists to fit. Reduce the amplitude.',
      );
    }
  } finally {
    rig.engine.dispose();
  }

  const tailFrom = onsetIndex + Math.floor((traceT.length - onsetIndex) * 0.8);
  const baselineFrom = Math.max(0, Math.floor(onsetIndex * 0.8));

  let baselineMv = traceV[Math.max(0, onsetIndex - 1)];
  if (onsetIndex > baselineFrom) {
    let sum = 0;
    for (let i = baselineFrom; i < onsetIndex; i += 1) sum += traceV[i];
    baselineMv = sum / (onsetIndex - baselineFrom);
  }

  let steadySum = 0;
  for (let i = tailFrom; i < traceV.length; i += 1) steadySum += traceV[i];
  const steadyMv = steadySum / Math.max(1, traceV.length - tailFrom);
  const deflectionMv = steadyMv - baselineMv;

  if (Math.abs(deflectionMv) < 0.05) {
    throw new ProtocolError(
      'The step barely moved the membrane. Increase the amplitude so the decay rises above numerical noise.',
    );
  }

  const y0 = Math.abs(baselineMv - steadyMv);
  const floor = y0 * 0.02;
  const ts: number[] = [];
  const logs: number[] = [];
  for (let i = onsetIndex; i < traceV.length; i += 1) {
    const deviation = Math.abs(traceV[i] - steadyMv);
    if (deviation <= floor || deviation > y0) continue;
    ts.push(traceT[i]);
    logs.push(Math.log(deviation));
  }

  const fit = linearFit(ts, logs);
  if (fit === null || !(fit.slope < 0)) {
    throw new ProtocolError(
      'The response did not decay monotonically toward a steady level, so no single time constant describes it.',
    );
  }

  const tauMs = -1 / fit.slope;
  const analytic = analyticTau(neuron.params);
  const meta = metaFor(neuron, rig, params.dt, startedAt, simulatedMs);

  const csv = [
    ...csvHeader(meta, 'Membrane time constant', [
      `step: ${csvNumber(params.amplitudePa)} pA for ${csvNumber(params.stepMs)} ms`,
      `tau: ${csvNumber(tauMs)} ms (r2 ${csvNumber(fit.r2)}, ${fit.n} points)`,
      analytic === null
        ? 'analytic cm/gL: not defined for this model'
        : `analytic cm/gL: ${csvNumber(analytic)} ms`,
      `baseline: ${csvNumber(baselineMv)} mV, steady: ${csvNumber(steadyMv)} mV`,
    ]),
    'time_ms,voltage_mV',
    ...Array.from(traceT, (t, i) => `${csvNumber(t)},${csvNumber(traceV[i])}`),
  ].join('\n');

  return {
    kind: 'tau',
    meta,
    params,
    traceT,
    traceV,
    onsetIndex,
    tauMs,
    baselineMv,
    steadyMv,
    deflectionMv,
    fitR2: fit.r2,
    fitPoints: fit.n,
    analyticTauMs: analytic,
    errorPercent:
      analytic !== null && analytic > 0 ? ((tauMs - analytic) / analytic) * 100 : null,
    csv,
  };
}

/* -------------------------------------------------------- 4. adaptation ---- */

/**
 * Inter-spike intervals through a sustained suprathreshold step.
 *
 * The adaptation index is the last interval over the first, so a regular-spiking
 * cell reports well above 1 and a non-adapting integrator reports 1 exactly.
 */
export async function runAdaptation(
  circuit: Circuit,
  neuronId: string,
  params: AdaptationParams,
  options: RunOptions = {},
): Promise<AdaptationResult> {
  const neuron = findNeuron(circuit, neuronId);
  const startedAt = performance.now();
  const rig = openCell(circuit, neuron, params.dt);
  const spikeTimesMs: number[] = [];
  let simulatedMs: number;

  try {
    const { engine, slot, holdingPa } = rig;
    const buffers = engine.buffers;

    await condition(options, 0, 1, 'sustained step');

    engine.reset();
    buffers.neurons.bias[slot] = holdingPa;
    settle(engine, params.settleMs);
    const onset = buffers.time;

    buffers.neurons.bias[slot] = holdingPa + params.amplitudePa;
    const absolute: number[] = [];
    collectSpikes(engine, slot, params.durationMs, absolute);
    for (const time of absolute) spikeTimesMs.push(time - onset);
    simulatedMs = params.settleMs + params.durationMs;
  } finally {
    rig.engine.dispose();
  }

  const isisMs: number[] = [];
  for (let i = 1; i < spikeTimesMs.length; i += 1) {
    isisMs.push(spikeTimesMs[i] - spikeTimesMs[i - 1]);
  }

  const first = isisMs.length > 0 ? isisMs[0] : null;
  const last = isisMs.length > 0 ? isisMs[isisMs.length - 1] : null;
  const tailFrom = Math.max(0, isisMs.length - Math.max(1, Math.round(isisMs.length * 0.2)));
  let tailSum = 0;
  for (let i = tailFrom; i < isisMs.length; i += 1) tailSum += isisMs[i];
  const tailMean = isisMs.length > 0 ? tailSum / (isisMs.length - tailFrom) : 0;

  const meta = metaFor(neuron, rig, params.dt, startedAt, simulatedMs);
  const adaptationIndex = first !== null && last !== null && first > 0 ? last / first : null;

  const csv = [
    ...csvHeader(meta, 'Spike-frequency adaptation', [
      `step: ${csvNumber(params.amplitudePa)} pA for ${csvNumber(params.durationMs)} ms`,
      `spikes: ${spikeTimesMs.length}`,
      adaptationIndex === null
        ? 'adaptation index: undefined (fewer than two intervals)'
        : `adaptation index: ${csvNumber(adaptationIndex)}`,
    ]),
    'interval_index,spike_time_ms,isi_ms,instantaneous_rate_Hz',
    ...spikeTimesMs.map((time, index) => {
      const isi = index > 0 ? isisMs[index - 1] : null;
      return [
        String(index),
        csvNumber(time),
        isi === null ? '' : csvNumber(isi),
        isi === null || isi <= 0 ? '' : csvNumber(1000 / isi),
      ].join(',');
    }),
  ].join('\n');

  return {
    kind: 'adaptation',
    meta,
    params,
    spikeTimesMs,
    isisMs,
    adaptationIndex,
    instantaneousHz: first !== null && first > 0 ? 1000 / first : null,
    steadyHz: tailMean > 0 ? 1000 / tailMean : null,
    meanHz: params.durationMs > 0 ? spikeTimesMs.length / (params.durationMs / 1000) : 0,
    spikeCount: spikeTimesMs.length,
    latencyMs: spikeTimesMs.length > 0 ? spikeTimesMs[0] : null,
    csv,
  };
}

/* ------------------------------------------------------ 5. paired pulse ---- */

interface PairRig {
  engine: SimulationEngine;
  preSlot: number;
  /** Index of the synapse under study inside the loaded pair. */
  synIndex: number;
}

function openPair(circuit: Circuit, synapse: Synapse, dt: number): PairRig {
  const source = findNeuron(circuit, synapse.source);
  const target = findNeuron(circuit, synapse.target);
  // An autapse names the same cell twice; loading it twice would give the engine
  // two neurons with one id and a synapse pointing at whichever won the map.
  const neurons = source.id === target.id ? [source] : [source, target];

  const engine = new SimulationEngine();
  try {
    engine.load(preparation(circuit, neurons, [synapse], dt));
    const preSlot = engine.slotOf(synapse.source);
    if (preSlot < 0 || engine.buffers.synapses.count !== 1) {
      throw new ProtocolError('The synapse could not be loaded into an isolated pair.');
    }
    return { engine, preSlot, synIndex: 0 };
  } catch (cause) {
    engine.dispose();
    throw cause;
  }
}

interface PulseSchedule {
  pulseSteps: number;
  pulse1Step: number;
  pulse2Step: number;
  totalSteps: number;
}

/**
 * Run one paired-pulse trial and fill `out` with the synaptic conductance at
 * every step. Returns true when exactly the expected spikes were evoked.
 *
 * The stimulus is a brief current pulse that stops the moment the cell fires,
 * which is the direct analogue of stimulating the axon at a rig: the identity of
 * the presynaptic spike matters, its cause does not.
 */
function runPulseTrial(
  rig: PairRig,
  amplitudePa: number,
  schedule: PulseSchedule,
  paired: boolean,
  out: Float64Array,
): boolean {
  const { engine, preSlot, synIndex } = rig;
  const buffers = engine.buffers;
  const synapses = buffers.synapses;
  engine.reset();

  let fired1 = false;
  let fired2 = false;
  let spikes = 0;

  for (let s = 0; s < schedule.totalSteps; s += 1) {
    if (!fired1 && s >= schedule.pulse1Step && s < schedule.pulse1Step + schedule.pulseSteps) {
      engine.poke(preSlot, amplitudePa);
    }
    if (
      paired &&
      !fired2 &&
      s >= schedule.pulse2Step &&
      s < schedule.pulse2Step + schedule.pulseSteps
    ) {
      engine.poke(preSlot, amplitudePa);
    }

    engine.stepOnce();

    if (buffers.neurons.spike[preSlot] === 1) {
      spikes += 1;
      if (!fired1) fired1 = true;
      else if (s >= schedule.pulse2Step) fired2 = true;
    }

    out[s] = synapses.gDecay[synIndex] - synapses.gRise[synIndex];
  }

  return paired ? fired1 && fired2 && spikes === 2 : fired1 && spikes === 1;
}

/**
 * Paired-pulse ratio across a sweep of inter-stimulus intervals.
 *
 * The second response is isolated by subtracting a single-pulse control trial
 * run on the identical schedule rather than by extrapolating the decay of the
 * first. Conductances superpose linearly and the first response cannot know
 * about a later stimulus, so the subtraction is exact rather than a fit — and it
 * needs no assumption about the shape of the tail it removes.
 */
export async function runPairedPulse(
  circuit: Circuit,
  params: PprParams,
  options: RunOptions = {},
): Promise<PprResult> {
  const synapse = circuit.synapses.find((candidate) => candidate.id === params.synapseId);
  if (synapse === undefined) {
    throw new ProtocolError('That synapse is no longer in the document.');
  }
  if (synapse.receptor === 'gap') {
    throw new ProtocolError(
      'A gap junction carries current continuously rather than in response to spikes, so a paired-pulse ratio is not defined for it.',
    );
  }
  if (!synapse.enabled) {
    throw new ProtocolError('That synapse is disabled and transmits nothing.');
  }

  const source = findNeuron(circuit, synapse.source);
  const target = findNeuron(circuit, synapse.target);
  const startedAt = performance.now();
  const intervals = sweepLevels(params.fromMs, params.toMs, params.stepMs, MAX_INTERVALS);
  const trials = Math.max(1, Math.round(params.trials));

  const rig = openPair(circuit, synapse, params.dt);
  const points: PprPoint[] = [];
  let stimulusPa = 0;
  let simulatedMs = 0;
  let holdingPa: number;

  const stimulusMs = Math.max(params.dt, 2);
  const total = 1 + intervals.length * trials * 2;

  try {
    const { engine, preSlot } = rig;
    const buffers = engine.buffers;
    const dt = engine.settings.dt;
    holdingPa = buffers.neurons.bias[preSlot];

    await condition(options, 0, total, 'calibrating stimulus');

    const preSteps = stepsFor(PPR_SETTLE_MS, dt);
    const pulseSteps = Math.max(1, stepsFor(stimulusMs, dt));
    const windowSteps = Math.max(1, stepsFor(params.windowMs, dt));

    // A cell that fires on its own cannot be paired-pulse stimulated: there is no
    // way to say which spike belongs to which pulse. The check runs for as long
    // as the longest trial will, so a slow drift into firing is caught too.
    const quietMs =
      PPR_SETTLE_MS + intervals[intervals.length - 1] + stimulusMs + params.windowMs;
    engine.reset();
    settle(engine, quietMs);
    if (buffers.neurons.spikeCount[preSlot] > 0) {
      throw new ProtocolError(
        'The presynaptic cell fires on its own at its holding current, so evoked spikes cannot be separated from spontaneous ones.',
      );
    }
    simulatedMs += quietMs;

    const calibrationSchedule: PulseSchedule = {
      pulseSteps,
      pulse1Step: preSteps,
      pulse2Step: preSteps,
      totalSteps: preSteps + pulseSteps + windowSteps,
    };
    const calibrationTrace = new Float64Array(calibrationSchedule.totalSteps);
    for (let i = 0; i < MAX_STIM_DOUBLINGS; i += 1) {
      const amplitude = STIM_SEED_PA * 2 ** i;
      simulatedMs += calibrationSchedule.totalSteps * dt;
      if (runPulseTrial(rig, amplitude, calibrationSchedule, false, calibrationTrace)) {
        stimulusPa = amplitude;
        break;
      }
    }
    if (stimulusPa === 0) {
      throw new ProtocolError(
        'No current pulse within the search range evoked exactly one presynaptic spike; the cell may be inexcitable at this holding level.',
      );
    }

    let done = 1;
    for (let i = 0; i < intervals.length; i += 1) {
      const intervalMs = intervals[i];
      const intervalSteps = stepsFor(intervalMs, dt);

      if (intervalSteps <= pulseSteps) {
        points.push({
          intervalMs,
          peak1Ns: 0,
          peak2Ns: 0,
          ratio: null,
          evoked: 0,
          failure: 'shorter than the stimulus pulse',
        });
        done += trials * 2;
        continue;
      }

      const schedule: PulseSchedule = {
        pulseSteps,
        pulse1Step: preSteps,
        pulse2Step: preSteps + intervalSteps,
        totalSteps: preSteps + intervalSteps + pulseSteps + windowSteps,
      };
      const control = new Float64Array(schedule.totalSteps);
      const pairedTrace = new Float64Array(schedule.totalSteps);
      const controlSum = new Float64Array(schedule.totalSteps);
      const pairedSum = new Float64Array(schedule.totalSteps);
      let evoked = 0;
      let failure: string | null = null;

      for (let t = 0; t < trials; t += 1) {
        await condition(options, done, total, `${intervalMs.toFixed(0)} ms — control ${t + 1}`);
        done += 1;
        const controlOk = runPulseTrial(rig, stimulusPa, schedule, false, control);
        simulatedMs += schedule.totalSteps * dt;

        await condition(options, done, total, `${intervalMs.toFixed(0)} ms — paired ${t + 1}`);
        done += 1;
        const pairedOk = runPulseTrial(rig, stimulusPa, schedule, true, pairedTrace);
        simulatedMs += schedule.totalSteps * dt;

        if (!controlOk || !pairedOk) {
          failure =
            failure ??
            (controlOk
              ? 'the second stimulus fell inside the refractory period'
              : 'the control trial did not evoke exactly one presynaptic spike');
          continue;
        }
        evoked += 1;
        for (let s = 0; s < schedule.totalSteps; s += 1) {
          controlSum[s] += control[s];
          pairedSum[s] += pairedTrace[s];
        }
      }

      if (evoked === 0) {
        points.push({
          intervalMs,
          peak1Ns: 0,
          peak2Ns: 0,
          ratio: null,
          evoked: 0,
          failure: failure ?? 'no trial evoked both spikes',
        });
        continue;
      }

      let peak1 = 0;
      let peak2 = 0;
      for (let s = 0; s < schedule.totalSteps; s += 1) {
        const first = controlSum[s] / evoked;
        if (first > peak1) peak1 = first;
        if (s < schedule.pulse2Step) continue;
        const second = pairedSum[s] / evoked - first;
        if (second > peak2) peak2 = second;
      }

      points.push({
        intervalMs,
        peak1Ns: peak1,
        peak2Ns: peak2,
        ratio: peak1 > 0 ? peak2 / peak1 : null,
        evoked,
        failure: evoked < trials ? failure : null,
      });
    }
  } finally {
    rig.engine.dispose();
  }

  const meta: ProtocolMeta = {
    neuronId: source.id,
    label: cellLabel(source),
    seed: source.morphology.seed,
    model: source.params.kind,
    holdingPa,
    dt: params.dt,
    elapsedMs: performance.now() - startedAt,
    simulatedMs,
  };

  const info: PprSynapseInfo = {
    synapseId: synapse.id,
    sourceId: synapse.source,
    targetId: synapse.target,
    targetLabel: cellLabel(target),
    targetSeed: target.morphology.seed,
    receptor: synapse.receptor,
    weightNs: synapse.weight,
    delayMs: synapse.delay,
    releaseProbability: synapse.releaseProbability,
    stpEnabled: synapse.stp.enabled,
    stpU: synapse.stp.u,
    tauRecMs: synapse.stp.tauRec,
    tauFacilMs: synapse.stp.tauFacil,
  };

  const csv = [
    ...csvHeader(meta, 'Paired-pulse ratio', [
      `synapse: ${synapse.id} -> ${info.targetLabel} (${info.receptor})`,
      `weight: ${csvNumber(info.weightNs)} nS, delay: ${csvNumber(info.delayMs)} ms`,
      `release probability: ${csvNumber(info.releaseProbability)}`,
      info.stpEnabled
        ? `STP: u ${csvNumber(info.stpU)}, tauRec ${csvNumber(info.tauRecMs)} ms, tauFacil ${csvNumber(info.tauFacilMs)} ms`
        : 'STP: disabled — the ratio is 1 by construction',
      `stimulus: ${csvNumber(stimulusPa)} pA for ${csvNumber(stimulusMs)} ms, ${trials} trial(s) per interval`,
    ]),
    'interval_ms,peak1_nS,peak2_nS,ratio,trials_evoked',
    ...points.map((point) =>
      [
        csvNumber(point.intervalMs),
        csvNumber(point.peak1Ns),
        csvNumber(point.peak2Ns),
        point.ratio === null ? '' : csvNumber(point.ratio),
        String(point.evoked),
      ].join(','),
    ),
  ].join('\n');

  return {
    kind: 'ppr',
    meta,
    params,
    synapse: info,
    points,
    stimulusPa,
    stimulusMs,
    csv,
  };
}

/* ---------------------------------------------------------- 6. rheobase ---- */

/**
 * Bisect for the smallest sustained current that produces a spike.
 *
 * The bracket is verified at both ends before the search starts: bisecting an
 * interval that is entirely sub- or supra-threshold converges quickly and
 * confidently to the wrong number.
 */
export async function runRheobase(
  circuit: Circuit,
  neuronId: string,
  params: RheobaseParams,
  options: RunOptions = {},
): Promise<RheobaseResult> {
  const neuron = findNeuron(circuit, neuronId);
  const startedAt = performance.now();
  const lowStart = Math.min(params.lowPa, params.highPa);
  const highStart = Math.max(params.lowPa, params.highPa);
  const tolerance = Math.max(1e-4, Math.abs(params.tolerancePa));
  const rig = openCell(circuit, neuron, params.dt);

  const probes: RheobaseProbe[] = [];
  let simulatedMs = 0;
  let iterations = 0;
  let latencyMs: number | null;
  let boundedBelow = false;
  let low = lowStart;
  let high = highStart;

  const estimate =
    2 + Math.max(1, Math.ceil(Math.log2(Math.max(1, (highStart - lowStart) / tolerance))));

  try {
    const { engine, slot, holdingPa } = rig;
    const buffers = engine.buffers;

    // Each probe stops at the first spike: a level either fires within the window
    // or it does not, and integrating past the answer buys nothing.
    const test = (current: number): number | null => {
      engine.reset();
      buffers.neurons.bias[slot] = holdingPa + current;
      const dt = engine.settings.dt;
      const steps = stepsFor(params.windowMs, dt);
      let latency: number | null = null;
      let ran = 0;
      for (let s = 0; s < steps; s += 1) {
        engine.stepOnce();
        ran += 1;
        if (buffers.neurons.spike[slot] === 1) {
          latency = buffers.time;
          break;
        }
      }
      simulatedMs += ran * dt;
      return latency;
    };

    await condition(options, 0, estimate, `${highStart.toFixed(1)} pA (upper bound)`);
    const highLatency = test(highStart);
    probes.push({
      iteration: 0,
      currentPa: highStart,
      spiked: highLatency !== null,
      lowPa: low,
      highPa: high,
    });
    if (highLatency === null) {
      throw new ProtocolError(
        `No spike at ${highStart.toFixed(1)} pA within ${params.windowMs.toFixed(0)} ms. Raise the upper bound or lengthen the window.`,
      );
    }
    latencyMs = highLatency;

    await condition(options, 1, estimate, `${lowStart.toFixed(1)} pA (lower bound)`);
    const lowLatency = test(lowStart);
    probes.push({
      iteration: 1,
      currentPa: lowStart,
      spiked: lowLatency !== null,
      lowPa: low,
      highPa: high,
    });
    if (lowLatency !== null) {
      boundedBelow = true;
      latencyMs = lowLatency;
      high = lowStart;
    } else {
      while (high - low > tolerance && iterations < MAX_BISECTIONS) {
        const mid = (low + high) / 2;
        await condition(
          options,
          probes.length,
          Math.max(estimate, probes.length + 1),
          `${mid.toFixed(2)} pA`,
        );
        const midLatency = test(mid);
        if (midLatency === null) {
          low = mid;
        } else {
          high = mid;
          latencyMs = midLatency;
        }
        iterations += 1;
        probes.push({
          iteration: probes.length,
          currentPa: mid,
          spiked: midLatency !== null,
          lowPa: low,
          highPa: high,
        });
      }
    }
  } finally {
    rig.engine.dispose();
  }

  const meta = metaFor(neuron, rig, params.dt, startedAt, simulatedMs);
  const csv = [
    ...csvHeader(meta, 'Rheobase search', [
      `window: ${csvNumber(params.windowMs)} ms, tolerance: ${csvNumber(tolerance)} pA`,
      `rheobase: ${csvNumber(high)} pA after ${iterations} bisection(s)`,
      boundedBelow
        ? 'the lower bound already fired, so the true rheobase is below it'
        : `bracket: [${csvNumber(low)}, ${csvNumber(high)}] pA`,
    ]),
    'iteration,current_pA,spiked,bracket_low_pA,bracket_high_pA',
    ...probes.map((probe) =>
      [
        String(probe.iteration),
        csvNumber(probe.currentPa),
        probe.spiked ? '1' : '0',
        csvNumber(probe.lowPa),
        csvNumber(probe.highPa),
      ].join(','),
    ),
  ].join('\n');

  return {
    kind: 'rheobase',
    meta,
    params,
    rheobasePa: high,
    bracketLowPa: low,
    bracketHighPa: high,
    iterations,
    latencyMs,
    boundedBelow,
    probes,
    csv,
  };
}
