# NeuroForge package contracts

This file is the integration contract. Every package is written independently and
must export **exactly** the symbols listed under its heading, with the given
signatures. Nothing else is guaranteed to exist at integration time.

Units, everywhere: voltage **mV**, time **ms**, current **pA**, capacitance **pF**,
conductance **nS**. World-space distances are in abstract "units" where a typical
soma radius is `1.0`.

Read `packages/shared/src/*.ts` before writing anything. Do not modify `shared`.
If you need a type that lives in another package, define it locally rather than
reaching across a boundary that this document does not sanction.

Import workspace code by package name (`@neuroforge/math`), never by relative
path across packages. Inside a package, use relative paths.

---

## `@neuroforge/math`

```ts
// rng.ts — deterministic, seedable, no global state
export class Rng {
  constructor(seed: number);
  next(): number;              // uniform [0,1)
  int(maxExclusive: number): number;
  range(min: number, max: number): number;
  normal(mean?: number, stdDev?: number): number;
  exponential(rate: number): number;
  poisson(lambda: number): number;
  bool(probability: number): boolean;
  pick<T>(items: readonly T[]): T;
  shuffle<T>(items: T[]): T[];
  onSphere(): { x: number; y: number; z: number };
  inSphere(): { x: number; y: number; z: number };
  fork(salt: number): Rng;     // independent stream derived from this one
  get seed(): number;
  reset(seed?: number): void;
}
export function hashSeed(...values: number[]): number;

// noise.ts
export function simplex3(x: number, y: number, z: number): number;   // [-1,1]
export function fbm3(x: number, y: number, z: number, octaves?: number, lacunarity?: number, gain?: number): number;
export function curlNoise3(x: number, y: number, z: number, epsilon?: number): { x: number; y: number; z: number };

// spline.ts — all splines operate on flat Float32Array xyz triples
export function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number;
export function cubicBezier(p0: number, p1: number, p2: number, p3: number, t: number): number;
/** Sample a quadratic arc from a to b bulging by `sag` along `up`. Writes 3*samples floats. */
export function sampleArc(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  sag: number, samples: number, out: Float32Array, offset?: number,
): void;
/** Arc-length lookup table for constant-speed traversal of a sampled polyline. */
export function buildArcLengthTable(points: Float32Array, count: number): Float32Array;
export function sampleAtDistance(points: Float32Array, table: Float32Array, count: number, distance: number, out: { x: number; y: number; z: number }): void;

// easing.ts
export const easeOutExpo: (t: number) => number;
export const easeInOutQuart: (t: number) => number;
export const easeOutBack: (t: number) => number;
export const easeOutElastic: (t: number) => number;
export function damp(current: number, target: number, lambda: number, dt: number): number;
export function dampAngle(current: number, target: number, lambda: number, dt: number): number;
export class SpringScalar {
  constructor(value: number, stiffness?: number, damping?: number, mass?: number);
  set target(v: number);
  get target(): number;
  get value(): number;
  jump(v: number): void;
  step(dt: number): number;
  get settled(): boolean;
}
export class SpringVec3 { /* same shape, x/y/z */
  constructor(x: number, y: number, z: number, stiffness?: number, damping?: number, mass?: number);
  setTarget(x: number, y: number, z: number): void;
  jump(x: number, y: number, z: number): void;
  step(dt: number): void;
  readonly x: number; readonly y: number; readonly z: number;
  get settled(): boolean;
}

// spatial.ts — uniform grid for picking and neighbour queries over SoA positions
export class SpatialHash {
  constructor(cellSize: number);
  rebuild(positions: Float32Array, count: number): void;
  /** Returns the number of hits written into `out`. */
  queryRadius(x: number, y: number, z: number, radius: number, out: Uint32Array): number;
  nearest(x: number, y: number, z: number, maxRadius: number): number; // slot index or -1
  /** Nearest slot along a ray; used for click-picking. */
  raycast(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, threshold: number): number;
}

// stats.ts
export function mean(values: ArrayLike<number>, count?: number): number;
export function stdDev(values: ArrayLike<number>, count?: number): number;
export function interSpikeIntervals(times: ArrayLike<number>, count: number, out: Float32Array): number;
export function coefficientOfVariation(intervals: ArrayLike<number>, count: number): number;
export function fanoFactor(counts: ArrayLike<number>, count: number): number;
/** In-place radix-2 FFT; both arrays must be power-of-two length. */
export function fft(real: Float32Array, imag: Float32Array): void;
/** Single-sided power spectrum of a real signal. Returns bin width in Hz. */
export function powerSpectrum(signal: Float32Array, sampleRateHz: number, out: Float32Array): number;
/** Dominant frequency in Hz within [minHz,maxHz]. */
export function dominantFrequency(spectrum: Float32Array, binHz: number, minHz: number, maxHz: number): number;
export function crossCorrelation(a: Float32Array, b: Float32Array, maxLag: number, out: Float32Array): void;

// index.ts re-exports all of the above.
```

---

## `@neuroforge/shaders`

WGSL and GLSL are exported as **plain template-literal strings from `.ts` files**.
No bundler loader is configured; do not create `.wgsl` files and expect them to
import. Prefix each with a `/* wgsl */` or `/* glsl */` comment for editor
highlighting.

```ts
// wgsl/integrate.ts
/** Compute shader integrating all five membrane models. Workgroup size 64. */
export const NEURON_INTEGRATE_WGSL: string;
/** Bind group layout that NEURON_INTEGRATE_WGSL expects, as a plain description. */
export const NEURON_INTEGRATE_BINDINGS: readonly { binding: number; name: string; type: 'storage' | 'read-only-storage' | 'uniform' }[];

// wgsl/synapse.ts
export const SYNAPSE_PROPAGATE_WGSL: string;   // conductance update + current accumulation
export const SYNAPSE_STDP_WGSL: string;        // trace decay + weight update
export const SYNAPSE_PROPAGATE_BINDINGS: readonly { binding: number; name: string; type: string }[];

// wgsl/particles.ts
export const PARTICLE_UPDATE_WGSL: string;     // advects impulse particles along axon splines
export const PARTICLE_EMIT_WGSL: string;

// glsl/neuron.ts — used by the WebGL2 instanced glyph material
export const NEURON_VERTEX_GLSL: string;
export const NEURON_FRAGMENT_GLSL: string;

// glsl/axon.ts — instanced tube/ribbon along a spline with travelling impulse
export const AXON_VERTEX_GLSL: string;
export const AXON_FRAGMENT_GLSL: string;

// glsl/grid.ts — infinite analytic grid with distance fade, no geometry
export const GRID_VERTEX_GLSL: string;
export const GRID_FRAGMENT_GLSL: string;

// glsl/particle.ts — additive point sprites for spikes
export const PARTICLE_VERTEX_GLSL: string;
export const PARTICLE_FRAGMENT_GLSL: string;

// glsl/common.ts — shared chunks injected into the above
export const VOLTAGE_RAMP_GLSL: string;   // vec3 voltageRamp(float v)
export const FOG_GLSL: string;            // vec3 applyFog(vec3 color, float depth, ...)
export const NOISE_GLSL: string;          // hash/simplex helpers
```

All GLSL must be **GLSL ES 3.00** compatible with Three.js's `RawShaderMaterial`
conventions *or* written as `ShaderMaterial` bodies (no `#version`, no precision
header — Three injects those). Use the `ShaderMaterial` convention.

---

## `@neuroforge/simulation`

```ts
import type { SimulationBuffers, SimulationSettings, Circuit, FrameStats, ComputeBackend, BackendCapabilities } from '@neuroforge/shared';

/** Result of one integrator advance. */
export interface StepResult {
  steps: number;
  spikes: number;
  simMs: number;
}

export interface Integrator {
  readonly backend: ComputeBackend;
  /** Advance by `steps` substeps of `settings.dt`. */
  step(buffers: SimulationBuffers, settings: SimulationSettings, steps: number): StepResult;
  /** Reset all state variables to rest without changing topology. */
  reset(buffers: SimulationBuffers): void;
  dispose(): void;
}

export class CpuIntegrator implements Integrator { constructor(); }
export class WasmIntegrator implements Integrator {
  static isAvailable(): boolean;
  static create(): Promise<WasmIntegrator | null>;
}
export class GpuIntegrator implements Integrator {
  static create(device: GPUDevice): Promise<GpuIntegrator | null>;
}

export async function detectCapabilities(): Promise<BackendCapabilities>;
/** Picks the best integrator for `preference`, falling back down the chain. */
export async function createIntegrator(preference: SimulationSettings['backend'], device?: GPUDevice | null): Promise<Integrator>;

/** Owns the buffers, the integrator and the real-time clock. */
export class SimulationEngine {
  constructor(settings?: Partial<SimulationSettings>);
  readonly buffers: SimulationBuffers;
  get settings(): SimulationSettings;
  get running(): boolean;
  get stats(): FrameStats;
  setSettings(patch: Partial<SimulationSettings>): void;
  /** Rebuild buffers from a document. Invalidates all cached array references. */
  load(circuit: Circuit): void;
  /** Slot index for a neuron id, or -1. */
  slotOf(id: string): number;
  /** Neuron id for a slot, or null. */
  idOf(slot: number): string | null;
  play(): void;
  pause(): void;
  reset(): void;
  /** Advance in real time given a wall-clock delta in seconds. Call once per frame. */
  advance(dtSeconds: number): StepResult;
  /** Single deterministic substep, for the step button. */
  stepOnce(): StepResult;
  /** Inject a current pulse into one neuron, used by the poke tool. */
  poke(slot: number, amplitude: number): void;
  setBackend(preference: SimulationSettings['backend']): Promise<void>;
  attachDevice(device: GPUDevice | null): Promise<void>;
  dispose(): void;
}

/** Ring buffers holding traces for the inspector plots. */
export class ProbeRecorder {
  constructor(capacity?: number);
  track(slot: number, signal: string): void;
  untrack(slot: number, signal: string): void;
  sample(buffers: SimulationBuffers): void;
  /** Returns the trace and how many samples are valid. */
  read(slot: number, signal: string): { values: Float32Array; times: Float32Array; count: number } | null;
  clear(): void;
}

/** Applies Stimulus records into buffers.neurons.iExt each step. */
export function applyStimuli(buffers: SimulationBuffers, stimuli: readonly import('@neuroforge/shared').Stimulus[], slotOf: (id: string) => number, time: number, rng: import('@neuroforge/math').Rng): void;
```

The CPU integrator is the reference implementation and must be numerically
correct for all five models; WASM and GPU must agree with it to within float
tolerance on a smoke network.

---

## `@neuroforge/renderer`

Everything here is Three.js. No React. The app wraps these in R3F components.

```ts
import * as THREE from 'three';

/** Procedural neuron glyph geometry built from a Morphology descriptor. */
export function buildSomaGeometry(detail?: number): THREE.BufferGeometry;
export function buildDendriteGeometry(morphology: import('@neuroforge/shared').Morphology): THREE.BufferGeometry;
export function buildAxonGeometry(morphology: import('@neuroforge/shared').Morphology): THREE.BufferGeometry;
/** Cache keyed by archetype+seed so identical neurons share one geometry. */
export class GlyphLibrary {
  constructor(maxVariants?: number);
  get(archetype: number, seed: number): { soma: THREE.BufferGeometry; dendrites: THREE.BufferGeometry; axon: THREE.BufferGeometry };
  dispose(): void;
}

/** Instanced renderer for every neuron in the scene. */
export class NeuronField extends THREE.Group {
  constructor(library: GlyphLibrary);
  /** Rebuild instance attributes from buffers. Call after structural edits. */
  rebuild(buffers: import('@neuroforge/shared').SimulationBuffers): void;
  /** Per-frame attribute refresh from live state. Cheap; no allocation. */
  update(buffers: import('@neuroforge/shared').SimulationBuffers, dt: number, settings: import('@neuroforge/shared').RenderSettings): void;
  raycastSlot(raycaster: THREE.Raycaster): number;
  dispose(): void;
}

/** GPU spline axons connecting neurons, with travelling impulses. */
export class AxonField extends THREE.Group {
  constructor();
  rebuild(buffers: import('@neuroforge/shared').SimulationBuffers): void;
  update(buffers: import('@neuroforge/shared').SimulationBuffers, dt: number, settings: import('@neuroforge/shared').RenderSettings): void;
  dispose(): void;
}

/** Additive GPU particle system emitting on spikes. */
export class SpikeParticles extends THREE.Points {
  constructor(capacity?: number);
  emitFromSpikes(buffers: import('@neuroforge/shared').SimulationBuffers): void;
  update(dt: number, settings: import('@neuroforge/shared').RenderSettings): void;
  get liveCount(): number;
  dispose(): void;
}

/** Infinite analytic grid drawn on a single screen-space quad. */
export class InfiniteGrid extends THREE.Mesh {
  constructor();
  update(camera: THREE.Camera, settings: import('@neuroforge/shared').RenderSettings): void;
  dispose(): void;
}

/** Selection halo + hover glow drawn as instanced billboards. */
export class SelectionOverlay extends THREE.Group {
  constructor();
  update(buffers: import('@neuroforge/shared').SimulationBuffers, dt: number): void;
  dispose(): void;
}

/** Creates a WebGPU renderer when available, else a tuned WebGL2 renderer. */
export async function createRenderer(canvas: HTMLCanvasElement): Promise<{
  renderer: THREE.WebGLRenderer | object;
  backend: 'webgpu' | 'webgl2';
  device: GPUDevice | null;
}>;

export type CameraMode = 'orbit' | 'fly' | 'first-person' | 'cinematic';
/** Unified camera controller with inertia; never snaps. */
export class CameraRig {
  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement);
  mode: CameraMode;
  update(dt: number): void;
  /** Smoothly frame a bounding box. */
  frame(min: THREE.Vector3, max: THREE.Vector3, duration?: number): void;
  focusOn(x: number, y: number, z: number, distance?: number): void;
  getState(): import('@neuroforge/shared').CameraState;
  setState(state: import('@neuroforge/shared').CameraState, immediate?: boolean): void;
  dispose(): void;
}
```

---

## `@neuroforge/physics`

```ts
/** Deterministic Barnes-Hut force-directed layout over SoA positions. */
export interface LayoutSettings {
  repulsion: number;
  attraction: number;
  gravity: number;
  damping: number;
  theta: number;
  maxSpeed: number;
  dimensions: 2 | 3;
  seed: number;
}
export const DEFAULT_LAYOUT_SETTINGS: LayoutSettings;

export class ForceLayout {
  constructor(settings?: Partial<LayoutSettings>);
  /** Bind to the live buffers; positions are mutated in place. */
  attach(buffers: import('@neuroforge/shared').SimulationBuffers): void;
  setSettings(patch: Partial<LayoutSettings>): void;
  /** One relaxation step. Returns total kinetic energy, for convergence tests. */
  step(dt: number): number;
  /** Run until energy falls below `epsilon` or `maxIterations` elapse. */
  solve(maxIterations?: number, epsilon?: number): number;
  pin(slot: number, pinned: boolean): void;
  reset(): void;
}

/** Analytic layouts used when instantiating populations. */
export function layoutGrid(count: number, columns: number, rows: number, layers: number, spacing: number, out: Float32Array, offset?: number): void;
export function layoutSphere(count: number, radius: number, jitter: number, seed: number, out: Float32Array, offset?: number): void;
export function layoutDisc(count: number, radius: number, thickness: number, seed: number, out: Float32Array, offset?: number): void;
export function layoutColumn(count: number, radius: number, height: number, seed: number, out: Float32Array, offset?: number): void;
/** Fibonacci sphere, used by layoutSphere and available directly. */
export function fibonacciSphere(index: number, count: number, radius: number, out: { x: number; y: number; z: number }): void;
```

---

## `@neuroforge/editor`

The Zustand store and the command system. This is the only package that mutates
the document.

```ts
import type { Circuit, Neuron, Synapse, NeuronId, SynapseId, ... } from '@neuroforge/shared';

/** An undoable operation. `apply` and `revert` must be exact inverses. */
export interface Command {
  readonly label: string;
  apply(draft: Circuit): void;
  revert(draft: Circuit): void;
  /** Commands with the same mergeKey issued within the coalesce window merge. */
  readonly mergeKey?: string;
}

export type Tool = 'select' | 'place' | 'connect' | 'erase' | 'probe' | 'stimulate' | 'pan';

export interface EditorState {
  circuit: Circuit;
  selection: readonly NeuronId[];
  selectedSynapses: readonly SynapseId[];
  hovered: NeuronId | null;
  tool: Tool;
  undoDepth: number;
  redoDepth: number;
  dirty: boolean;
  lastSavedAt: number;
  // panels
  inspectorOpen: boolean;
  builderOpen: boolean;
  libraryOpen: boolean;
  commandPaletteOpen: boolean;
}

export interface EditorActions {
  execute(command: Command): void;
  undo(): void;
  redo(): void;
  transaction(label: string, fn: (draft: Circuit) => void): void;

  addNeuron(partial?: Partial<Neuron>): NeuronId;
  removeNeurons(ids: readonly NeuronId[]): void;
  updateNeuron(id: NeuronId, patch: Partial<Neuron>): void;
  updateNeurons(ids: readonly NeuronId[], patch: Partial<Neuron>): void;
  connect(source: NeuronId, target: NeuronId, partial?: Partial<Synapse>): SynapseId | null;
  removeSynapses(ids: readonly SynapseId[]): void;
  updateSynapse(id: SynapseId, patch: Partial<Synapse>): void;

  addPopulation(spec: PopulationSpec): PopulationId;
  connectPopulations(spec: ProjectionSpec): void;

  select(ids: readonly NeuronId[], additive?: boolean): void;
  selectAll(): void;
  clearSelection(): void;
  setHovered(id: NeuronId | null): void;
  setTool(tool: Tool): void;

  setSimulationSettings(patch: Partial<SimulationSettings>): void;
  setRenderSettings(patch: Partial<RenderSettings>): void;
  setCamera(state: CameraState): void;

  loadCircuit(circuit: Circuit): void;
  newCircuit(name?: string): void;
  togglePanel(panel: 'inspector' | 'builder' | 'library' | 'commandPalette', open?: boolean): void;
}

export const useEditor: import('zustand').UseBoundStore<import('zustand').StoreApi<EditorState & EditorActions>>;

export interface PopulationSpec { name: string; size: number; polarity: NeuronPolarity; model: NeuronModelKind; params?: Partial<NeuronParams>; layout: PopulationLayout; origin?: Vec3; archetype?: MorphologyArchetype; color?: string | null; }
export interface ProjectionSpec { name: string; source: PopulationId; target: PopulationId; rule: ConnectivityRule; receptor?: ReceptorKind; weightMean?: number; weightJitter?: number; delayMean?: number; delayJitter?: number; plasticity?: PlasticityKind; }

/** Build neurons for a population spec without touching the store. */
export function instantiatePopulation(spec: PopulationSpec): { population: Population; neurons: Neuron[] };
export function instantiateProjection(spec: ProjectionSpec, circuit: Circuit): Synapse[];

export function createEmptyCircuit(name?: string): Circuit;

/** Keyboard map; the app binds these to the window. */
export interface Shortcut { keys: string; label: string; group: string; run(): void; }
export function buildShortcuts(): Shortcut[];
```

---

## `@neuroforge/io`

```ts
import Dexie from 'dexie';
import type { Circuit, Snapshot, CircuitId, SnapshotId } from '@neuroforge/shared';

export class NeuroForgeDb extends Dexie {
  circuits: Dexie.Table<Circuit, string>;
  snapshots: Dexie.Table<Snapshot, string>;
  settings: Dexie.Table<{ key: string; value: unknown }, string>;
  constructor();
}
export const db: NeuroForgeDb;

export async function saveCircuit(circuit: Circuit): Promise<void>;
export async function loadCircuit(id: CircuitId): Promise<Circuit | null>;
export async function listCircuits(): Promise<Circuit[]>;
export async function deleteCircuit(id: CircuitId): Promise<void>;
export async function createSnapshot(circuit: Circuit, label: string, automatic?: boolean): Promise<Snapshot>;
export async function listSnapshots(id: CircuitId): Promise<Snapshot[]>;
export async function restoreSnapshot(id: SnapshotId): Promise<Circuit | null>;
export async function pruneSnapshots(id: CircuitId, keep?: number): Promise<void>;
export async function getSetting<T>(key: string, fallback: T): Promise<T>;
export async function setSetting(key: string, value: unknown): Promise<void>;

/** Debounced autosave driver. */
export class Autosaver {
  constructor(intervalMs?: number);
  start(getCircuit: () => Circuit, onSaved: (at: number) => void): void;
  touch(): void;
  flush(): Promise<void>;
  stop(): void;
}

export type ExportFormat = 'json' | 'brian2' | 'nest' | 'pytorch' | 'onnx' | 'python';
export interface ExportResult { filename: string; mimeType: string; content: string | Uint8Array; }
export function exportCircuit(circuit: Circuit, format: ExportFormat): ExportResult;
export function importCircuitJson(text: string): { circuit: Circuit | null; errors: string[] };
/** Validate and migrate a document read from disk or an older schema version. */
export function migrateCircuit(raw: unknown): { circuit: Circuit | null; errors: string[] };
export function downloadExport(result: ExportResult): void;
```

Exporters must emit **runnable** code, not sketches. Brian2 and NEST scripts must
reproduce the model equations, per-neuron parameters, connectivity, delays and
stimuli. PyTorch export emits a `torch.nn.Module` implementing the same dynamics
as a discrete-time recurrent spiking layer. ONNX export emits a valid ONNX
protobuf graph of the weight matrix and the surrogate-gradient LIF cell.

---

## `@neuroforge/ai`

```ts
export type AiProvider = 'anthropic' | 'openai';
export interface AiCredentials { provider: AiProvider; apiKey: string; model: string; proxyUrl?: string; }
export const DEFAULT_MODELS: Record<AiProvider, string>;

/** One structural edit the model asked for. Validated before it is applied. */
export type CircuitAction =
  | { type: 'create-population'; spec: import('@neuroforge/editor').PopulationSpec }
  | { type: 'connect-populations'; spec: Omit<import('@neuroforge/editor').ProjectionSpec, 'source' | 'target'> & { sourceName: string; targetName: string } }
  | { type: 'set-simulation'; patch: Partial<import('@neuroforge/shared').SimulationSettings> }
  | { type: 'set-render'; patch: Partial<import('@neuroforge/shared').RenderSettings> }
  | { type: 'add-stimulus'; targetPopulation: string; pattern: import('@neuroforge/shared').StimulusPattern; name: string }
  | { type: 'tune-population'; name: string; params: Partial<import('@neuroforge/shared').NeuronParams>; bias?: number; noise?: number }
  | { type: 'tune-projection'; name: string; weightMean?: number; delayMean?: number; plasticity?: import('@neuroforge/shared').PlasticityKind }
  | { type: 'clear' };

export interface AiPlan { summary: string; actions: CircuitAction[]; warnings: string[]; }

export interface AiRequest { prompt: string; circuit: import('@neuroforge/shared').Circuit; credentials: AiCredentials; signal?: AbortSignal; }
export interface AiStreamEvent { kind: 'text' | 'plan' | 'error' | 'done'; text?: string; plan?: AiPlan; error?: string; }

export async function* streamCircuitPlan(request: AiRequest): AsyncGenerator<AiStreamEvent>;
/** Runs without any network access; used when no key is configured. */
export function planLocally(prompt: string, circuit: import('@neuroforge/shared').Circuit): AiPlan;
export function validatePlan(plan: AiPlan, circuit: import('@neuroforge/shared').Circuit): { plan: AiPlan; errors: string[] };
export async function loadCredentials(): Promise<AiCredentials | null>;
export async function storeCredentials(credentials: AiCredentials | null): Promise<void>;
export const CIRCUIT_TOOL_SCHEMA: object;   // JSON Schema for the tool the model calls
export const SYSTEM_PROMPT: string;
```

`planLocally` is not a stub: it is a real deterministic parser covering the
documented example prompts (region + counts + polarity + rhythm targets) so the
builder works with no API key at all.

---

## `@neuroforge/ui`

Presentational primitives only. No domain types, no store access.

```ts
export function cn(...classes: unknown[]): string;
export const Button, IconButton, Panel, PanelHeader, PanelSection, Field, Label,
  NumberField, Slider, Select, SelectItem, Switch, Tabs, TabsList, Tab, TabPanel,
  Tooltip, Dialog, Popover, ScrollArea, Separator, Badge, Kbd, Spinner,
  SegmentedControl, ColorSwatch, Sparkline, Meter, EmptyState, Toast, ToastViewport;
```

Every component forwards refs, accepts `className`, and is keyboard accessible.
Sliders and number fields must support drag-scrub, arrow keys, shift for fine and
double-click to reset.

---

## `apps/web`

React composition layer. Imports from every package; owns no domain logic beyond
wiring. Path alias `@/*` maps to `apps/web/src/*`.
