import type {
  CircuitId,
  NeuronId,
  PopulationId,
  ProbeId,
  SnapshotId,
  StimulusId,
} from './ids';
import type { Morphology, Neuron, NeuronParams, NeuronPolarity } from './neuron';
import type { ConnectivityRule, Synapse } from './synapse';
import type { Vec3 } from './geometry';

/** Spatial arrangement used when a population is instantiated. */
export type PopulationLayout =
  | { kind: 'grid'; columns: number; rows: number; layers: number; spacing: number }
  | { kind: 'sphere'; radius: number; jitter: number; seed: number }
  | { kind: 'disc'; radius: number; thickness: number; seed: number }
  | { kind: 'column'; radius: number; height: number; seed: number }
  | { kind: 'explicit'; positions: readonly Vec3[] };

/**
 * A named group of neurons sharing a model, polarity and morphology. Populations
 * are the unit the AI builder and the layout engine operate on; individual
 * neurons remain independently editable after instantiation.
 */
export interface Population {
  readonly id: PopulationId;
  name: string;
  size: number;
  polarity: NeuronPolarity;
  params: NeuronParams;
  morphology: Morphology;
  layout: PopulationLayout;
  /** World-space offset applied on top of the layout. */
  origin: Vec3;
  /** Accent colour override as a hex string, or null to use the polarity default. */
  color: string | null;
  members: readonly NeuronId[];
  collapsed: boolean;
}

/** A recorded projection between two populations, kept for provenance and re-wiring. */
export interface Projection {
  readonly id: string;
  name: string;
  source: PopulationId;
  target: PopulationId;
  rule: ConnectivityRule;
  weightMean: number;
  weightJitter: number;
  delayMean: number;
  delayJitter: number;
}

/** External input injected into the circuit. */
export type StimulusPattern =
  | { kind: 'constant'; amplitude: number }
  | { kind: 'step'; amplitude: number; start: number; duration: number }
  | { kind: 'pulse-train'; amplitude: number; frequency: number; width: number; start: number }
  | { kind: 'sine'; amplitude: number; frequency: number; offset: number }
  | { kind: 'poisson'; rate: number; amplitude: number; seed: number }
  | { kind: 'ramp'; from: number; to: number; start: number; duration: number };

export interface Stimulus {
  readonly id: StimulusId;
  name: string;
  targets: readonly NeuronId[];
  pattern: StimulusPattern;
  enabled: boolean;
}

/** A recording probe attached to a neuron. */
export interface Probe {
  readonly id: ProbeId;
  target: NeuronId;
  /** Which state variable to trace. */
  signal: 'voltage' | 'current' | 'conductance' | 'calcium' | 'adaptation' | 'spikes';
  /** Ring-buffer capacity in samples. */
  capacity: number;
  color: string;
  enabled: boolean;
}

/** Global integration settings. */
export interface SimulationSettings {
  /** Integration timestep (ms). */
  dt: number;
  /** Integrator used for continuous state. */
  integrator: 'euler' | 'rk2' | 'rk4' | 'exponential-euler';
  /** Simulated milliseconds per wall-clock second when running in real time. */
  speed: number;
  /** Global synaptic weight multiplier. */
  gain: number;
  /** Background noise applied to every neuron (pA). */
  noise: number;
  /** RNG seed for all stochastic processes. */
  seed: number;
  /** Whether plasticity rules update weights during integration. */
  plasticityEnabled: boolean;
  /** Maximum substeps executed per animation frame, to bound frame time. */
  maxSubstepsPerFrame: number;
  /** Compute backend preference. */
  backend: 'auto' | 'gpu' | 'wasm' | 'cpu';
}

export const DEFAULT_SIMULATION_SETTINGS: SimulationSettings = {
  dt: 0.1,
  integrator: 'exponential-euler',
  speed: 1,
  gain: 1,
  noise: 0,
  seed: 0x9e3779b9,
  plasticityEnabled: false,
  maxSubstepsPerFrame: 32,
  backend: 'auto',
};

/** Camera state persisted with the document so a reload restores the view. */
export interface CameraState {
  position: Vec3;
  target: Vec3;
  fov: number;
  mode: 'orbit' | 'fly' | 'first-person' | 'cinematic';
}

export const DEFAULT_CAMERA: CameraState = {
  position: { x: 0, y: 34, z: 96 },
  target: { x: 0, y: 0, z: 0 },
  fov: 42,
  mode: 'orbit',
};

/** Post-processing and scene appearance, persisted per document. */
export interface RenderSettings {
  bloomIntensity: number;
  bloomThreshold: number;
  bloomRadius: number;
  depthOfField: boolean;
  focusDistance: number;
  focalLength: number;
  bokehScale: number;
  fogDensity: number;
  ambientOcclusion: boolean;
  aoIntensity: number;
  vignette: number;
  chromaticAberration: number;
  exposure: number;
  gridVisible: boolean;
  gridFade: number;
  showDendrites: boolean;
  showAxons: boolean;
  showParticles: boolean;
  particleDensity: number;
  neuronScale: number;
  /** Renders voltage as a colour ramp rather than a flat accent. */
  voltageColoring: boolean;
}

export const DEFAULT_RENDER_SETTINGS: RenderSettings = {
  bloomIntensity: 1.15,
  bloomThreshold: 0.62,
  bloomRadius: 0.72,
  depthOfField: true,
  focusDistance: 0.035,
  focalLength: 0.09,
  bokehScale: 3.2,
  fogDensity: 0.0125,
  ambientOcclusion: true,
  aoIntensity: 0.62,
  vignette: 0.42,
  chromaticAberration: 0.0009,
  exposure: 1.05,
  gridVisible: true,
  gridFade: 0.55,
  showDendrites: true,
  showAxons: true,
  showParticles: true,
  particleDensity: 1,
  neuronScale: 1,
  voltageColoring: true,
};

/**
 * The complete serialisable document. This is what gets written to IndexedDB,
 * exported as JSON, diffed for undo/redo, and handed to the exporters.
 */
export interface Circuit {
  readonly id: CircuitId;
  name: string;
  description: string;
  /** Schema version; bumped whenever a migration is required. */
  version: number;
  createdAt: number;
  updatedAt: number;
  neurons: readonly Neuron[];
  synapses: readonly Synapse[];
  populations: readonly Population[];
  projections: readonly Projection[];
  stimuli: readonly Stimulus[];
  probes: readonly Probe[];
  simulation: SimulationSettings;
  camera: CameraState;
  render: RenderSettings;
  tags: readonly string[];
}

export const CIRCUIT_SCHEMA_VERSION = 1;

/** A point-in-time copy of a circuit, used by version history. */
export interface Snapshot {
  readonly id: SnapshotId;
  readonly circuitId: CircuitId;
  label: string;
  createdAt: number;
  /** Automatic snapshots are pruned first when history is trimmed. */
  automatic: boolean;
  neuronCount: number;
  synapseCount: number;
  circuit: Circuit;
}
