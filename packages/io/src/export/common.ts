/**
 * Shared analysis and code-generation helpers used by every exporter.
 *
 * The single most important thing defined here is the *canonical export
 * ordering*: neurons are re-ordered so that all neurons sharing a membrane
 * model form one contiguous block. Every exporter emits neurons, weights and
 * recordings in that order, which is what lets the Brian2 / NEST / PyTorch /
 * ONNX outputs be compared against each other slot for slot.
 */

import { MODEL_FROM_CODE } from '@neuroforge/shared';
import type {
  Circuit,
  Neuron,
  NeuronId,
  NeuronModelKind,
  NeuronParams,
  PlasticityConfig,
  Probe,
  ReceptorKind,
  ReceptorKinetics,
  ShortTermPlasticity,
  Stimulus,
  StimulusPattern,
  Synapse,
} from '@neuroforge/shared';

/** Simulated duration written into every generated script. */
const DEFAULT_RUN_MS = 1000;

/**
 * Upper bound on the number of distinct synaptic conductance channels an
 * exporter will materialise on a postsynaptic population. Kinetics are stored
 * per synapse, so a circuit that jitters them would otherwise produce one
 * channel per synapse.
 */
const MAX_CHANNELS = 12;

/** Reversal potentials at or above this are treated as excitatory (mV). */
const EXCITATORY_REVERSAL_SPLIT = -50;

const GROUP_NAMES: Record<NeuronModelKind, string> = {
  lif: 'lif',
  izhikevich: 'izh',
  'hodgkin-huxley': 'hh',
  adex: 'adex',
  'morris-lecar': 'ml',
};

const MODEL_DESCRIPTIONS: Record<NeuronModelKind, string> = {
  lif: 'leaky integrate-and-fire',
  izhikevich: 'Izhikevich',
  'hodgkin-huxley': 'Hodgkin-Huxley',
  adex: 'adaptive exponential integrate-and-fire',
  'morris-lecar': 'Morris-Lecar',
};

/** Replace a non-finite number with a fallback so generated code never contains NaN. */
export function finite(value: number, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function modelLabel(kind: NeuronModelKind): string {
  return MODEL_DESCRIPTIONS[kind];
}

/**
 * A synaptic conductance waveform.
 *
 * `weight` in the document is the *peak* conductance, so the difference of
 * exponentials has to be normalised: at its peak the raw difference
 * `exp(-t/tauDecay) - exp(-t/tauRise)` is less than one.
 */
export interface ConductanceKernel {
  /** `single` when rise and decay are indistinguishable and the kernel degenerates. */
  form: 'dual' | 'single';
  tauRise: number;
  tauDecay: number;
  /** Multiplier on the weight that makes the waveform peak at exactly `weight`. */
  norm: number;
  /** Integral of the unit-peak waveform in ms; charge = weight * area * drivingForce. */
  area: number;
}

function conductanceKernel(tauRiseMs: number, tauDecayMs: number): ConductanceKernel {
  const a = Math.max(finite(tauRiseMs, 0), 0);
  const b = Math.max(finite(tauDecayMs, 1), 1e-4);
  const tauDecay = Math.max(a, b);
  const tauRise = Math.min(a, b);
  // g = gDecay - gRise is identically zero when the two constants coincide, so
  // a rise phase shorter than 0.1% of the decay collapses to a single exponential.
  if (tauRise <= 0 || tauDecay - tauRise < 1e-3 * tauDecay) {
    return { form: 'single', tauRise: 0, tauDecay, norm: 1, area: tauDecay };
  }
  const tPeak = ((tauDecay * tauRise) / (tauDecay - tauRise)) * Math.log(tauDecay / tauRise);
  const peak = Math.exp(-tPeak / tauDecay) - Math.exp(-tPeak / tauRise);
  const norm = 1 / peak;
  return { form: 'dual', tauRise, tauDecay, norm, area: norm * (tauDecay - tauRise) };
}

/** A distinct conductance channel: one set of kinetics on a postsynaptic neuron. */
export interface ExportChannel {
  index: number;
  /** Python-safe suffix, e.g. `c0`. */
  name: string;
  receptor: ReceptorKind;
  kinetics: ReceptorKinetics;
  kernel: ConductanceKernel;
  excitatory: boolean;
}

export interface ExportSynapse {
  id: string;
  /** Canonical index of the presynaptic neuron. */
  pre: number;
  /** Canonical index of the postsynaptic neuron. */
  post: number;
  channel: number;
  receptor: ReceptorKind;
  /** Peak conductance in nS, already multiplied by the global gain. */
  weight: number;
  /** Conduction delay in ms, clamped to be non-negative. */
  delay: number;
  plasticity: PlasticityConfig;
  stp: ShortTermPlasticity;
  releaseProbability: number;
}

export interface ExportGroup {
  kind: NeuronModelKind;
  /** Python-safe identifier fragment, unique across groups. */
  name: string;
  /** First canonical index belonging to this group. */
  offset: number;
  size: number;
}

export interface ExportStimulus {
  stimulus: Stimulus;
  /** Canonical indices of the neurons the stimulus drives. */
  targets: readonly number[];
}

export interface ExportProbe {
  probe: Probe;
  /** Canonical index of the probed neuron. */
  target: number;
}

/** The circuit reshaped into everything the exporters need, in canonical order. */
export interface ExportCircuit {
  circuit: Circuit;
  neurons: readonly Neuron[];
  groups: readonly ExportGroup[];
  /** Canonical index -> index into `groups`. */
  groupOf: readonly number[];
  /** Canonical index -> index within its own group. */
  localOf: readonly number[];
  indexOf: ReadonlyMap<NeuronId, number>;
  synapses: readonly ExportSynapse[];
  channels: readonly ExportChannel[];
  stimuli: readonly ExportStimulus[];
  probes: readonly ExportProbe[];
  /** Human-readable notes about anything omitted or approximated. */
  notes: readonly string[];
  dt: number;
  gain: number;
  /** Global background noise amplitude in pA. */
  globalNoise: number;
  seed: number;
  plasticityEnabled: boolean;
  duration: number;
}

function kineticsSignature(receptor: ReceptorKind, k: ReceptorKinetics): string {
  return [
    receptor,
    finite(k.tauRise, 0).toFixed(6),
    finite(k.tauDecay, 1).toFixed(6),
    finite(k.eRev, 0).toFixed(6),
    finite(k.mgBlock, 0).toFixed(6),
  ].join('|');
}

function normaliseKinetics(k: ReceptorKinetics): ReceptorKinetics {
  return {
    tauRise: Math.max(finite(k.tauRise, 0), 0),
    tauDecay: Math.max(finite(k.tauDecay, 1), 1e-4),
    eRev: finite(k.eRev, 0),
    mgBlock: Math.max(finite(k.mgBlock, 0), 0),
  };
}

function meanKinetics(samples: readonly ReceptorKinetics[]): ReceptorKinetics {
  const n = Math.max(samples.length, 1);
  let tauRise = 0;
  let tauDecay = 0;
  let eRev = 0;
  let mgBlock = 0;
  for (const s of samples) {
    tauRise += s.tauRise;
    tauDecay += s.tauDecay;
    eRev += s.eRev;
    mgBlock += s.mgBlock;
  }
  return { tauRise: tauRise / n, tauDecay: tauDecay / n, eRev: eRev / n, mgBlock: mgBlock / n };
}

function buildChannels(
  synapses: readonly Synapse[],
  notes: string[],
): { channels: ExportChannel[]; channelOf: Map<string, number> } {
  const bySignature = new Map<string, { receptor: ReceptorKind; samples: ReceptorKinetics[] }>();
  for (const s of synapses) {
    const kin = normaliseKinetics(s.kinetics);
    const sig = kineticsSignature(s.receptor, kin);
    const entry = bySignature.get(sig);
    if (entry) entry.samples.push(kin);
    else bySignature.set(sig, { receptor: s.receptor, samples: [kin] });
  }

  const makeChannel = (index: number, receptor: ReceptorKind, kin: ReceptorKinetics): ExportChannel => ({
    index,
    name: `c${index}`,
    receptor,
    kinetics: kin,
    kernel: conductanceKernel(kin.tauRise, kin.tauDecay),
    excitatory: kin.eRev >= EXCITATORY_REVERSAL_SPLIT,
  });

  const channelOf = new Map<string, number>();
  if (bySignature.size <= MAX_CHANNELS) {
    const channels: ExportChannel[] = [];
    for (const [sig, entry] of bySignature) {
      channelOf.set(sig, channels.length);
      channels.push(makeChannel(channels.length, entry.receptor, entry.samples[0]));
    }
    return { channels, channelOf };
  }

  // Too many distinct kinetics to give each its own channel: collapse to one
  // channel per receptor kind using the mean of the kinetics seen on it.
  notes.push(
    `${bySignature.size} distinct synaptic kinetics were merged into one channel per receptor kind; ` +
      `per-synapse rise/decay/reversal jitter is averaged in this export.`,
  );
  const byReceptor = new Map<ReceptorKind, ReceptorKinetics[]>();
  for (const entry of bySignature.values()) {
    const list = byReceptor.get(entry.receptor);
    if (list) list.push(...entry.samples);
    else byReceptor.set(entry.receptor, [...entry.samples]);
  }
  const channels: ExportChannel[] = [];
  const receptorChannel = new Map<ReceptorKind, number>();
  for (const [receptor, samples] of byReceptor) {
    receptorChannel.set(receptor, channels.length);
    channels.push(makeChannel(channels.length, receptor, meanKinetics(samples)));
  }
  for (const [sig, entry] of bySignature) {
    channelOf.set(sig, receptorChannel.get(entry.receptor) ?? 0);
  }
  return { channels, channelOf };
}

/**
 * Reshape a document into the canonical export form. Disabled neurons and
 * synapses, and synapses whose endpoints are missing, are dropped and reported
 * in `notes` rather than being silently ignored.
 */
export function indexCircuit(circuit: Circuit): ExportCircuit {
  const notes: string[] = [];

  const live = circuit.neurons.filter((n) => n.enabled);
  const disabled = circuit.neurons.length - live.length;
  if (disabled > 0) notes.push(`${disabled} disabled neuron(s) omitted.`);

  const groups: ExportGroup[] = [];
  const neurons: Neuron[] = [];
  const groupOf: number[] = [];
  const localOf: number[] = [];
  for (const kind of MODEL_FROM_CODE) {
    const members = live.filter((n) => n.params.kind === kind);
    if (members.length === 0) continue;
    const groupIndex = groups.length;
    const offset = neurons.length;
    for (let i = 0; i < members.length; i += 1) {
      neurons.push(members[i]);
      groupOf.push(groupIndex);
      localOf.push(i);
    }
    groups.push({ kind, name: GROUP_NAMES[kind], offset, size: members.length });
  }

  const indexOf = new Map<NeuronId, number>();
  for (let i = 0; i < neurons.length; i += 1) indexOf.set(neurons[i].id, i);

  const usable: Synapse[] = [];
  let droppedDisabled = 0;
  let droppedDangling = 0;
  for (const s of circuit.synapses) {
    if (!s.enabled) {
      droppedDisabled += 1;
      continue;
    }
    if (!indexOf.has(s.source) || !indexOf.has(s.target)) {
      droppedDangling += 1;
      continue;
    }
    usable.push(s);
  }
  if (droppedDisabled > 0) notes.push(`${droppedDisabled} disabled synapse(s) omitted.`);
  if (droppedDangling > 0) {
    notes.push(`${droppedDangling} synapse(s) omitted because an endpoint is missing or disabled.`);
  }

  const { channels, channelOf } = buildChannels(usable, notes);
  const gain = finite(circuit.simulation.gain, 1);
  const synapses: ExportSynapse[] = usable.map((s) => {
    const kin = normaliseKinetics(s.kinetics);
    return {
      id: s.id,
      pre: indexOf.get(s.source) ?? 0,
      post: indexOf.get(s.target) ?? 0,
      channel: channelOf.get(kineticsSignature(s.receptor, kin)) ?? 0,
      receptor: s.receptor,
      weight: gain * finite(s.weight, 0),
      delay: Math.max(0, finite(s.delay, 0)),
      plasticity: s.plasticity,
      stp: s.stp,
      releaseProbability: Math.min(1, Math.max(0, finite(s.releaseProbability, 1))),
    };
  });

  const stimuli: ExportStimulus[] = [];
  for (const stimulus of circuit.stimuli) {
    if (!stimulus.enabled) continue;
    const targets: number[] = [];
    for (const id of stimulus.targets) {
      const index = indexOf.get(id);
      if (index !== undefined) targets.push(index);
    }
    if (targets.length === 0) continue;
    // Duplicated targets would double-count in exporters that scatter-add.
    const unique = [...new Set(targets)].sort((a, b) => a - b);
    stimuli.push({ stimulus, targets: unique });
  }

  const probes: ExportProbe[] = [];
  for (const probe of circuit.probes) {
    if (!probe.enabled) continue;
    const target = indexOf.get(probe.target);
    if (target === undefined) continue;
    probes.push({ probe, target });
  }

  return {
    circuit,
    neurons,
    groups,
    groupOf,
    localOf,
    indexOf,
    synapses,
    channels,
    stimuli,
    probes,
    notes,
    dt: Math.max(1e-4, finite(circuit.simulation.dt, 0.1)),
    gain,
    globalNoise: Math.max(0, finite(circuit.simulation.noise, 0)),
    seed: Math.floor(Math.abs(finite(circuit.simulation.seed, 1))) % 0x7fffffff,
    plasticityEnabled: circuit.simulation.plasticityEnabled,
    duration: DEFAULT_RUN_MS,
  };
}

/** Total noise amplitude (pA) seen by one neuron: two independent sources add in quadrature. */
export function effectiveNoise(neuron: Neuron, model: ExportCircuit): number {
  return Math.hypot(Math.max(0, finite(neuron.noise, 0)), model.globalNoise);
}

/** Resting potential used when linearising a model for a current-based target. */
export function restingOf(params: NeuronParams): number {
  switch (params.kind) {
    case 'lif':
      return finite(params.eL, -70);
    case 'izhikevich':
      return finite(params.c, -65);
    case 'hodgkin-huxley':
      return -65;
    case 'adex':
      return finite(params.eL, -70.6);
    case 'morris-lecar':
      return -60.9;
  }
}

/**
 * Equivalent single-compartment LIF used by targets that only implement one
 * membrane model (currently the ONNX graph).
 */
export interface LifEquivalent {
  /** Membrane capacitance in pF. */
  cm: number;
  /** Membrane time constant in ms. */
  tauM: number;
  vRest: number;
  vThresh: number;
  vReset: number;
  tRefract: number;
  /** Multiplier applied to input current before it reaches the membrane. */
  currentScale: number;
  exact: boolean;
}

export function lifEquivalent(params: NeuronParams): LifEquivalent {
  switch (params.kind) {
    case 'lif': {
      const cm = Math.max(1e-3, finite(params.cm, 200));
      const gL = Math.max(1e-6, finite(params.gL, 10));
      return {
        cm,
        tauM: cm / gL,
        vRest: finite(params.eL, -70),
        vThresh: finite(params.vThresh, -50),
        vReset: finite(params.vReset, -58),
        tRefract: Math.max(0, finite(params.tRefract, 0)),
        currentScale: 1,
        exact: true,
      };
    }
    case 'adex': {
      const cm = Math.max(1e-3, finite(params.cm, 281));
      const gL = Math.max(1e-6, finite(params.gL, 30));
      return {
        cm,
        tauM: cm / gL,
        vRest: finite(params.eL, -70.6),
        vThresh: finite(params.vT, -50.4),
        vReset: finite(params.vReset, -70.6),
        tRefract: Math.max(0, finite(params.tRefract, 0)),
        currentScale: 1,
        exact: false,
      };
    }
    case 'izhikevich': {
      const rest = finite(params.c, -65);
      // Linearise 0.04v^2 + 5v + 140 around the reset potential; the slope is
      // negative in the sub-threshold regime, and its reciprocal is the
      // effective membrane time constant.
      const slope = 0.08 * rest + 5;
      const tauM = slope < -1e-3 ? -1 / slope : 10;
      const iScale = Math.max(1e-9, finite(params.iScale, 0.04));
      return {
        cm: 1 / iScale,
        tauM,
        vRest: rest,
        vThresh: finite(params.vPeak, 30),
        vReset: rest,
        tRefract: 0,
        currentScale: 1,
        exact: false,
      };
    }
    case 'hodgkin-huxley': {
      const cm = Math.max(1e-3, finite(params.cm, 100));
      const gL = Math.max(1e-6, finite(params.gL, 30));
      return {
        cm,
        tauM: cm / gL,
        vRest: -65,
        vThresh: finite(params.vDetect, -20),
        vReset: -65,
        tRefract: 1,
        currentScale: 1,
        exact: false,
      };
    }
    case 'morris-lecar': {
      const cm = Math.max(1e-3, finite(params.cm, 20));
      const gL = Math.max(1e-6, finite(params.gL, 2));
      return {
        cm,
        tauM: cm / gL,
        vRest: -60.9,
        vThresh: finite(params.vDetect, 0),
        vReset: finite(params.eL, -60),
        tRefract: 0,
        currentScale: 1,
        exact: false,
      };
    }
  }
}

/** (exp(x) - 1) / x, evaluated stably at its removable singularity. */
function exprel(x: number): number {
  return Math.abs(x) < 1e-8 ? 1 + x / 2 : Math.expm1(x) / x;
}

/** Steady-state Hodgkin-Huxley gating variables at a given membrane potential. */
export function hhSteadyState(v: number): { m: number; h: number; n: number } {
  const alphaM = 1 / exprel(-(v + 40) / 10);
  const betaM = 4 * Math.exp(-(v + 65) / 18);
  const alphaH = 0.07 * Math.exp(-(v + 65) / 20);
  const betaH = 1 / (1 + Math.exp(-(v + 35) / 10));
  const alphaN = 0.1 / exprel(-(v + 55) / 10);
  const betaN = 0.125 * Math.exp(-(v + 65) / 80);
  return {
    m: alphaM / (alphaM + betaM),
    h: alphaH / (alphaH + betaH),
    n: alphaN / (alphaN + betaN),
  };
}

/** Every state variable of a neuron at rest, in the order the exporters emit them. */
export interface RestingState {
  v: number;
  /** Izhikevich u, AdEx w or Morris-Lecar w. */
  w: number;
  m: number;
  h: number;
  n: number;
}

export function restingState(params: NeuronParams): RestingState {
  switch (params.kind) {
    case 'lif':
      return { v: finite(params.eL, -70), w: 0, m: 0, h: 0, n: 0 };
    case 'adex':
      return { v: finite(params.eL, -70.6), w: 0, m: 0, h: 0, n: 0 };
    case 'izhikevich': {
      const v = finite(params.c, -65);
      return { v, w: finite(params.b, 0.2) * v, m: 0, h: 0, n: 0 };
    }
    case 'hodgkin-huxley': {
      const steady = hhSteadyState(-65);
      return { v: -65, w: 0, m: steady.m, h: steady.h, n: steady.n };
    }
    case 'morris-lecar': {
      const v = -60.9;
      const w = 0.5 * (1 + Math.tanh((v - finite(params.v3, 2)) / Math.max(finite(params.v4, 30), 1e-6)));
      return { v, w, m: 0, h: 0, n: 0 };
    }
  }
}

/** Conduction delay expressed in integration steps; a recurrence needs at least one. */
export function delayToSteps(delayMs: number, dt: number): number {
  return Math.max(1, Math.round(Math.max(0, finite(delayMs, 0)) / dt));
}

export interface DelayBinning {
  /** Representative delay of each bin, in integration steps. */
  bins: number[];
  /** Parallel to the input: which bin each delay was assigned to. */
  assignment: number[];
  /** True when delays had to be merged to respect `maxBins`. */
  quantised: boolean;
}

/**
 * Reduce a list of per-synapse delays to at most `maxBins` distinct values.
 * Exporters that materialise one weight matrix per delay need this bound.
 */
export function binDelays(steps: readonly number[], maxBins: number): DelayBinning {
  const limit = Math.max(1, maxBins);
  const unique = Array.from(new Set(steps)).sort((a, b) => a - b);
  if (unique.length === 0) return { bins: [1], assignment: [], quantised: false };
  if (unique.length <= limit) {
    const lookup = new Map<number, number>();
    unique.forEach((value, i) => lookup.set(value, i));
    return {
      bins: unique,
      assignment: steps.map((s) => lookup.get(s) ?? 0),
      quantised: false,
    };
  }

  const min = unique[0];
  const max = unique[unique.length - 1];
  const width = Math.max((max - min) / limit, Number.EPSILON);
  const rawBin = (s: number): number => Math.min(limit - 1, Math.floor((s - min) / width));
  const sums = new Array<number>(limit).fill(0);
  const counts = new Array<number>(limit).fill(0);
  for (const s of steps) {
    const b = rawBin(s);
    sums[b] += s;
    counts[b] += 1;
  }
  const bins: number[] = [];
  const rawToFinal = new Map<number, number>();
  const valueToFinal = new Map<number, number>();
  for (let b = 0; b < limit; b += 1) {
    if (counts[b] === 0) continue;
    const value = Math.max(1, Math.round(sums[b] / counts[b]));
    const existing = valueToFinal.get(value);
    if (existing !== undefined) {
      rawToFinal.set(b, existing);
      continue;
    }
    valueToFinal.set(value, bins.length);
    rawToFinal.set(b, bins.length);
    bins.push(value);
  }
  return {
    bins,
    assignment: steps.map((s) => rawToFinal.get(rawBin(s)) ?? 0),
    quantised: true,
  };
}

/* ------------------------------------------------------------------------ */
/* Python source formatting                                                  */
/* ------------------------------------------------------------------------ */

/** Format a number as a Python float literal that round-trips exactly. */
export function pyFloat(value: number): string {
  if (Number.isNaN(value)) return "float('nan')";
  if (value === Number.POSITIVE_INFINITY) return "float('inf')";
  if (value === Number.NEGATIVE_INFINITY) return "-float('inf')";
  if (Number.isInteger(value) && Math.abs(value) < 1e15) {
    return Object.is(value, -0) ? '-0.0' : `${value}.0`;
  }
  return String(value);
}

export function pyInt(value: number): string {
  return String(Math.trunc(finite(value, 0)));
}

export function pyStr(value: string): string {
  const escaped = value
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
  return `'${escaped}'`;
}

function wrapItems(items: readonly string[], perLine: number, indentWidth: number): string {
  if (items.length === 0) return '[]';
  if (items.length <= perLine) return `[${items.join(', ')}]`;
  const pad = ' '.repeat(indentWidth);
  const lines: string[] = [];
  for (let i = 0; i < items.length; i += perLine) {
    lines.push(pad + items.slice(i, i + perLine).join(', ') + ',');
  }
  return `[\n${lines.join('\n')}\n]`;
}

export function pyFloatList(values: readonly number[], perLine = 8, indentWidth = 4): string {
  return wrapItems(values.map(pyFloat), perLine, indentWidth);
}

export function pyIntList(values: readonly number[], perLine = 16, indentWidth = 4): string {
  return wrapItems(values.map(pyInt), perLine, indentWidth);
}

/**
 * Smallest stimulus frequency an exporter will emit, in Hz.
 *
 * Every generated `build_stimulus` turns a frequency into a period with
 * `1000.0 / frequency`, so a zero frequency would divide by zero at runtime.
 * A document that reaches an exporter without going through `migrateCircuit`
 * carries no such guarantee, hence the floor here.
 */
const MIN_STIMULUS_FREQUENCY = 1e-6;

/**
 * A stimulus pattern as a Python dict literal.
 *
 * Shared by the Brian2, PyTorch and NumPy targets so that their three
 * `build_stimulus` implementations are fed byte-for-byte identical parameters.
 */
export function pyStimulusPayload(pattern: StimulusPattern): string {
  switch (pattern.kind) {
    case 'constant':
      return `{'amplitude': ${pyFloat(finite(pattern.amplitude, 0))}}`;
    case 'step':
      return (
        `{'amplitude': ${pyFloat(finite(pattern.amplitude, 0))}, ` +
        `'start': ${pyFloat(Math.max(0, finite(pattern.start, 0)))}, ` +
        `'duration': ${pyFloat(Math.max(0, finite(pattern.duration, 0)))}}`
      );
    case 'pulse-train':
      return (
        `{'amplitude': ${pyFloat(finite(pattern.amplitude, 0))}, ` +
        `'frequency': ${pyFloat(Math.max(MIN_STIMULUS_FREQUENCY, finite(pattern.frequency, 10)))}, ` +
        `'width': ${pyFloat(Math.max(0, finite(pattern.width, 1)))}, ` +
        `'start': ${pyFloat(Math.max(0, finite(pattern.start, 0)))}}`
      );
    case 'sine':
      return (
        `{'amplitude': ${pyFloat(finite(pattern.amplitude, 0))}, ` +
        `'frequency': ${pyFloat(Math.max(0, finite(pattern.frequency, 10)))}, ` +
        `'offset': ${pyFloat(finite(pattern.offset, 0))}}`
      );
    case 'poisson':
      return (
        `{'rate': ${pyFloat(Math.max(0, finite(pattern.rate, 0)))}, ` +
        `'amplitude': ${pyFloat(finite(pattern.amplitude, 0))}, ` +
        `'seed': ${pyInt(pattern.seed)}}`
      );
    case 'ramp':
      return (
        `{'from': ${pyFloat(finite(pattern.from, 0))}, ` +
        `'to': ${pyFloat(finite(pattern.to, 0))}, ` +
        `'start': ${pyFloat(Math.max(0, finite(pattern.start, 0)))}, ` +
        `'duration': ${pyFloat(Math.max(0, finite(pattern.duration, 0)))}}`
      );
  }
}

/** Indent every line of a block by `spaces`, leaving blank lines empty. */
export function indentBlock(source: string, spaces: number): string {
  const pad = ' '.repeat(spaces);
  return source
    .split('\n')
    .map((line) => (line.length === 0 ? line : pad + line))
    .join('\n');
}

/** True when every element equals the first one. */
export function allEqual(values: readonly number[]): boolean {
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] !== values[0]) return false;
  }
  return true;
}

/** Filesystem-safe slug derived from a document name. */
export function slugify(name: string, fallback = 'circuit'): string {
  const slug = name
    .normalize('NFKD')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
  return slug.length > 0 ? slug.slice(0, 64) : fallback;
}

/**
 * The provenance banner shared by every generated artefact. Returned without a
 * comment prefix so each target can apply its own.
 */
export function bannerLines(model: ExportCircuit, target: string): string[] {
  const c = model.circuit;
  const lines = [
    `NeuroForge export - ${target}`,
    '',
    `Circuit: ${c.name || 'untitled'} (${c.id})`,
  ];
  if (c.description.trim().length > 0) {
    for (const line of c.description.split('\n')) lines.push(`  ${line}`);
  }
  lines.push(
    `Schema version ${c.version}, exported ${new Date().toISOString()}`,
    `${model.neurons.length} neurons, ${model.synapses.length} synapses, ` +
      `${model.groups.length} membrane model(s), ${model.channels.length} synaptic channel(s)`,
    `dt = ${model.dt} ms, integrator = ${c.simulation.integrator}, weight gain = ${model.gain}, seed = ${model.seed}`,
    '',
    'Units follow the NeuroForge convention: mV, ms, pA, pF, nS.',
    'Synaptic weight is the PEAK conductance in nS; the difference-of-exponentials',
    'waveform is scaled so that its maximum equals the weight.',
  );
  if (model.globalNoise > 0) {
    lines.push(
      `Per-neuron noise is combined in quadrature with the global background of ${model.globalNoise} pA.`,
    );
  }
  if (model.notes.length > 0) {
    lines.push('', 'Notes:');
    for (const note of model.notes) lines.push(`  - ${note}`);
  }
  return lines;
}

/** Prefix each line with `#` to form a Python comment block. */
export function pyComment(lines: readonly string[]): string {
  return lines.map((line) => (line.length === 0 ? '#' : `# ${line}`)).join('\n');
}

/** Wrap the banner in a Python module docstring. */
export function pyDocstring(lines: readonly string[]): string {
  const body = lines.join('\n').replace(/"""/g, '\\"\\"\\"');
  return `"""\n${body}\n"""`;
}

/** Synapses grouped by an arbitrary key, preserving first-seen order. */
export function groupBy<K, T>(items: readonly T[], key: (item: T) => K): Map<K, T[]> {
  const out = new Map<K, T[]>();
  for (const item of items) {
    const k = key(item);
    const list = out.get(k);
    if (list) list.push(item);
    else out.set(k, [item]);
  }
  return out;
}
