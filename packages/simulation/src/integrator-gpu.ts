import { NEURON_PARAM_STRIDE, SYNAPSE_PARAM_STRIDE } from '@neuroforge/shared';
import type { SimulationBuffers, SimulationSettings, SpikeLog } from '@neuroforge/shared';
import {
  NEURON_INTEGRATE_BINDINGS,
  NEURON_INTEGRATE_WGSL,
  SYNAPSE_CURRENT_SCALE,
  SYNAPSE_PROPAGATE_BINDINGS,
  SYNAPSE_PROPAGATE_WGSL,
  SYNAPSE_STDP_BINDINGS,
  SYNAPSE_STDP_WGSL,
  WGSL_WORKGROUP_SIZE,
} from '@neuroforge/shaders';
import type { ShaderBinding } from '@neuroforge/shaders';
import {
  WGSL_META,
  WGSL_NEURON_STRUCTS,
  WGSL_RANDOM,
  WGSL_SYNAPSE_STRUCTS,
} from '@neuroforge/shaders/wgsl/common';

import { CpuIntegrator } from './integrator-cpu';
import type { Integrator, StepResult } from './types';

/**
 * WebGPU compute backend.
 *
 * The device comes from `navigator.gpu` independently of any canvas, as
 * `CONTRACTS.md` requires: compute is WebGPU, rendering stays WebGL2. Nothing
 * here touches a swap chain.
 *
 * ## Where the kernels come from
 *
 * `@neuroforge/shaders` owns the neuron, synapse-propagation and STDP kernels,
 * and its WGSL chunks own the GPU-side record layouts. Two stages that package
 * does not provide are declared below, because without them the pipeline has no
 * closed loop:
 *
 * - `AXON_DELIVER_WGSL`, the axonal delay stage. `SYNAPSE_PROPAGATE_WGSL`
 *   documents `arrival[i]` as "written by the delay stage", but no such kernel
 *   is exported, and the CPU-side `DelayQueue` cannot drive it: the spikes that
 *   feed it are only visible to the host a frame later, which would smear every
 *   conduction delay by the whole readback latency.
 * - `SPIKE_CAPTURE_WGSL`, which records `(slot, time)` pairs for
 *   `SimulationBuffers.spikes`. The ring inside `NEURON_INTEGRATE_WGSL` stores
 *   the neuron index only, and a spike log without times cannot drive the
 *   particle system or a raster plot.
 *
 * ## Latency
 *
 * State is read back with `mapAsync` into the same CPU columns the renderer
 * already reads, which costs one frame of latency and no pipeline stall. Two
 * staging buffers rotate so a submission never waits on a map still in flight;
 * when neither is free the readback is skipped for that step rather than
 * blocking. `step()` therefore reports the spike count of the most recently
 * *completed* readback, and accumulates the deltas so the running total stays
 * conserved even when an individual step reports zero.
 */

/** Reciprocal of the fixed-point scale the atomic current accumulator uses. */
const INV_CURRENT_SCALE = 1 / SYNAPSE_CURRENT_SCALE;

/* Display-only time constants of the CPU reference. They are module-private
 * there, so they are restated here as the uniforms that reproduce it. The rate
 * kernel adds 1000/rateTau per spike, which equals the reference's
 * (1000/dt) * (dt/RATE_TAU), so the two agree exactly at any timestep. */
const RATE_TAU = 120;
const FLASH_TAU = 45;
const CALCIUM_TAU = 220;
const CALCIUM_GAIN = 1;

/** Synaptic activity envelope decay (ms), matching the CPU reference. */
const ACTIVITY_TAU = 60;

/**
 * Magnesium concentration in mM. The propagate kernel forms the Jahr-Stevens
 * block as `mgBlock * (mg / 3.57) * exp(-0.062 v)`; the reference uses a fixed
 * coefficient of 0.28, and 1 / 3.57 = 0.2801, so 1 mM reproduces it.
 */
const MG_CONCENTRATION = 1;

/** Multiplier on every synapse's own learning rate; the reference has none. */
const GLOBAL_LEARNING_RATE = 1;

/** Word counts of the GPU records declared by `@neuroforge/shaders`. */
const NEURON_DYNAMIC_WORDS = 8;
const NEURON_STATIC_WORDS = 4;
const NEURON_OUTPUT_WORDS = 4;
const SYNAPSE_STATIC_WORDS = 10;
const SYNAPSE_DYNAMIC_WORDS = 5;
const SYNAPSE_TRACE_WORDS = 4;

/** One captured spike is a neuron slot followed by its time, bitcast to u32. */
const SPIKE_EVENT_WORDS = 2;

/** Words of the spike bookkeeping buffer; only word 0, the ticket, is read. */
const SPIKE_META_WORDS = 4;

/**
 * Capacity of the GPU spike ring, in events. A power of two, so ticket
 * arithmetic stays exact across the u32 wrap. Overflow within a single readback
 * drops the oldest events exactly as the CPU `SpikeLog` ring does, and the head
 * still advances by the true count so the loss stays visible.
 */
const SPIKE_RING_CAPACITY = 8192;

/** Substeps encoded into one command buffer before submitting. */
const MAX_BATCH_SUBSTEPS = 64;

/** Ceiling on the delay ring, in bytes, independent of what the device allows. */
const DELAY_RING_BYTE_BUDGET = 32 * 1024 * 1024;

/** Largest struct any of the five compute uniforms declares, in bytes. */
const MAX_UNIFORM_STRUCT_BYTES = 48;

/** Staging buffers in the readback ring. Two is enough to never block. */
const STAGING_SLOTS = 2;

type StagingState = 'free' | 'inflight' | 'ready' | 'failed';

/* --------------------------------------------------------------- kernels -- */

const DELIVER_BINDINGS: readonly ShaderBinding[] = [
  { binding: 0, name: 'uni', type: 'uniform' },
  { binding: 1, name: 'synapses', type: 'read-only-storage' },
  { binding: 2, name: 'outputs', type: 'read-only-storage' },
  { binding: 3, name: 'ring', type: 'storage' },
  { binding: 4, name: 'arrival', type: 'storage' },
];

/**
 * Axonal delay, one invocation per synapse.
 *
 * `ring` is a `ringSlots x synapseCount` grid of release amplitudes. On substep
 * `cursor` a synapse reads and clears its cell in column `cursor`, then writes
 * this substep's release into column `cursor + lead`. Because a synapse owns
 * one cell of every column, nothing is shared between invocations and no
 * atomics are needed; because the cell is cleared on the way past, a column
 * that receives no release reads as silence on the next revolution rather than
 * replaying a spike from one horizon ago.
 *
 * `lead` is `round(delay/dt) - 1` because delivery already costs one substep:
 * this kernel reads the spike flags the integrator published on the previous
 * substep, so a lead of zero still arrives one timestep after emission. Leads
 * past the ring's horizon are clamped, which is the same behaviour the CPU
 * reference has at its calendar queue's horizon.
 */
const AXON_DELIVER_WGSL = /* wgsl */ `
${WGSL_META}
${WGSL_RANDOM}
${WGSL_NEURON_STRUCTS}
${WGSL_SYNAPSE_STRUCTS}

struct DeliverUniforms {
  @align(16) count : u32,
  ringSlots : u32,
  cursor : u32,
  seed : u32,
  step : u32,
  invDt : f32,
}

@group(0) @binding(0) var<uniform> uni : DeliverUniforms;
@group(0) @binding(1) var<storage, read> synapses : array<SynapseStatic>;
@group(0) @binding(2) var<storage, read> outputs : array<NeuronOutput>;
@group(0) @binding(3) var<storage, read_write> ring : array<f32>;
@group(0) @binding(4) var<storage, read_write> arrival : array<f32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= uni.count) {
    return;
  }

  let syn = synapses[index];
  let cell = uni.cursor * uni.count + index;
  let due = ring[cell];
  ring[cell] = 0.0;

  // A vesicle releases whole or not at all, so what travels down the axon is a
  // fraction of a full release, never a weight: the propagate kernel applies
  // the weight, the global gain and short-term depression itself.
  var release = 0.0;
  if (metaEnabled(syn.meta) && outputs[syn.pre].spike != 0u) {
    release = 1.0;
    if (syn.releaseProb < 1.0 &&
        randomUnit(hashCombine(index ^ uni.seed, uni.step)) >= syn.releaseProb) {
      release = 0.0;
    }
  }

  let lead = clamp(round(syn.delay * uni.invDt) - 1.0, 0.0, f32(uni.ringSlots - 1u));
  if (lead < 0.5) {
    arrival[index] = due + release;
  } else {
    arrival[index] = due;
    ring[((uni.cursor + u32(lead)) % uni.ringSlots) * uni.count + index] = release;
  }
}
`;

const CAPTURE_BINDINGS: readonly ShaderBinding[] = [
  { binding: 0, name: 'uni', type: 'uniform' },
  { binding: 1, name: 'outputs', type: 'read-only-storage' },
  { binding: 2, name: 'meta', type: 'storage' },
  { binding: 3, name: 'events', type: 'storage' },
];

/**
 * Spike log capture, one invocation per neuron.
 *
 * `meta[0]` is a monotonic ticket: its delta between two readbacks is the exact
 * number of spikes emitted in between, which is what `StepResult.spikes`
 * reports. The ticket also places the event in the ring, so a reader that knows
 * it can order the events without a second cursor.
 */
const SPIKE_CAPTURE_WGSL = /* wgsl */ `
${WGSL_NEURON_STRUCTS}

struct CaptureUniforms {
  @align(16) count : u32,
  capacity : u32,
  time : f32,
}

@group(0) @binding(0) var<uniform> uni : CaptureUniforms;
@group(0) @binding(1) var<storage, read> outputs : array<NeuronOutput>;
@group(0) @binding(2) var<storage, read_write> meta : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> events : array<u32>;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= uni.count || outputs[index].spike == 0u) {
    return;
  }
  let ticket = atomicAdd(&meta[0], 1u);
  if (uni.capacity == 0u) {
    return;
  }
  let slot = (ticket % uni.capacity) * ${SPIKE_EVENT_WORDS}u;
  events[slot] = index;
  events[slot + 1u] = bitcast<u32>(uni.time);
}
`;

/* --------------------------------------------------------------- helpers -- */

function alignUp(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

/** Pack four `Uint8Array` columns into the `meta` word the WGSL chunks decode. */
function packMeta(kind: number, subKind: number, enabled: number, flags: number): number {
  return (
    (kind & 0xff) | ((subKind & 0xff) << 8) | ((enabled & 0xff) << 16) | ((flags & 0xff) << 24)
  );
}

/** Substitute a finite value for a parameter WGSL would turn into a NaN. */
function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function layoutFromBindings(
  device: GPUDevice,
  bindings: readonly ShaderBinding[],
  label: string,
): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
    entries: bindings.map((entry) => ({
      binding: entry.binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer:
        entry.type === 'uniform'
          ? { type: 'uniform' as const, hasDynamicOffset: true }
          : { type: entry.type },
    })),
  });
}

async function createPipeline(
  device: GPUDevice,
  code: string,
  layout: GPUBindGroupLayout,
  label: string,
): Promise<GPUComputePipeline> {
  return device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
    compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
  });
}

/**
 * Copy the tail of a foreign ring of spike events into a `SpikeLog`.
 *
 * Shared by the GPU and WASM backends, which both keep their spikes in a ring
 * of their own and expose a monotonic total rather than a cursor. `emitted` is
 * the true number of events since the last call; when it exceeds the source
 * ring's capacity only the most recent survive, and the head still advances by
 * the full amount so the gap stays visible instead of being silently closed.
 *
 * The two source rings are read through a stride rather than through accessor
 * callbacks so that this runs allocation-free on the step path: the GPU keeps
 * interleaved `(slot, time)` pairs in one buffer (`stride` 2, `timeOffset` 1),
 * the WASM core keeps two parallel columns (`stride` 1, `timeOffset` 0).
 */
export function replaySpikeLog(
  log: SpikeLog,
  emitted: number,
  ringCapacity: number,
  ringTotal: number,
  neuronSource: Uint32Array,
  timeSource: Float32Array,
  stride: number,
  timeOffset: number,
): void {
  if (emitted <= 0 || ringCapacity <= 0 || log.capacity <= 0) return;
  const available = Math.min(emitted, ringCapacity);
  // ringTotal is an unsigned 32-bit counter, so the first surviving ticket can
  // sit on the far side of a wrap; normalise it into the ring before indexing.
  const first = (((ringTotal - available) % ringCapacity) + ringCapacity) % ringCapacity;
  const base = log.head + emitted - available;
  for (let k = 0; k < available; k += 1) {
    const source = (first + k) % ringCapacity;
    const target = (base + k) % log.capacity;
    log.neuron[target] = neuronSource[source * stride];
    log.time[target] = timeSource[source * stride + timeOffset];
  }
  log.head += emitted;
}

/* --------------------------------------------------------------- records -- */

interface Section {
  readonly offset: number;
  readonly bytes: number;
}

interface ReadbackLayout {
  readonly neuronDynamic: Section;
  readonly neuronOutput: Section;
  readonly current: Section;
  readonly synapseDynamic: Section;
  readonly synapseWeight: Section;
  readonly synapseTrace: Section;
  readonly spikeMeta: Section;
  readonly spikeEvent: Section;
  readonly bytes: number;
}

function buildReadbackLayout(neurons: number, synapses: number): ReadbackLayout {
  let cursor = 0;
  const take = (bytes: number): Section => {
    const section: Section = { offset: cursor, bytes };
    cursor += bytes;
    return section;
  };
  return {
    neuronDynamic: take(neurons * NEURON_DYNAMIC_WORDS * 4),
    neuronOutput: take(neurons * NEURON_OUTPUT_WORDS * 4),
    current: take(neurons * 4),
    synapseDynamic: take(synapses * SYNAPSE_DYNAMIC_WORDS * 4),
    synapseWeight: take(synapses * 4),
    synapseTrace: take(synapses * SYNAPSE_TRACE_WORDS * 4),
    spikeMeta: take(SPIKE_META_WORDS * 4),
    spikeEvent: take(SPIKE_RING_CAPACITY * SPIKE_EVENT_WORDS * 4),
    bytes: cursor,
  };
}

interface Pipelines {
  readonly deliver: GPUComputePipeline;
  readonly propagate: GPUComputePipeline;
  readonly integrate: GPUComputePipeline;
  readonly capture: GPUComputePipeline;
  readonly stdp: GPUComputePipeline;
  readonly deliverLayout: GPUBindGroupLayout;
  readonly propagateLayout: GPUBindGroupLayout;
  readonly integrateLayout: GPUBindGroupLayout;
  readonly captureLayout: GPUBindGroupLayout;
  readonly stdpLayout: GPUBindGroupLayout;
}

interface Storage {
  readonly neuronParams: GPUBuffer;
  readonly neuronStatic: GPUBuffer;
  readonly neuronDynamic: GPUBuffer;
  readonly neuronOutput: GPUBuffer;
  readonly current: GPUBuffer;
  readonly spikeSink: GPUBuffer;
  readonly synapseStatic: GPUBuffer;
  readonly synapseParams: GPUBuffer;
  readonly synapseWeight: GPUBuffer;
  readonly synapseDynamic: GPUBuffer;
  readonly synapseTrace: GPUBuffer;
  readonly arrival: GPUBuffer;
  readonly ring: GPUBuffer;
  readonly spikeMeta: GPUBuffer;
  readonly spikeEvent: GPUBuffer;
}

interface Groups {
  readonly deliver: GPUBindGroup;
  readonly propagate: GPUBindGroup;
  readonly integrate: GPUBindGroup;
  readonly capture: GPUBindGroup;
  readonly stdp: GPUBindGroup;
}

/** Host mirror of one compute uniform, laid out as an array of aligned slots. */
interface UniformStream {
  readonly gpu: GPUBuffer;
  readonly f32: Float32Array;
  readonly u32: Uint32Array;
  /** 32-bit words between consecutive slots. */
  readonly stride: number;
}

interface Streams {
  readonly deliver: UniformStream;
  readonly propagate: UniformStream;
  readonly integrate: UniformStream;
  readonly capture: UniformStream;
  readonly stdp: UniformStream;
  /** The five above, hoisted so the per-batch upload loop allocates nothing. */
  readonly all: readonly UniformStream[];
}

/** Host staging for the interleaved GPU records, reused every step. */
interface Scratch {
  readonly neuronStaticF32: Float32Array;
  readonly neuronStaticU32: Uint32Array;
  readonly neuronDynamic: Float32Array;
  readonly neuronOutputF32: Float32Array;
  readonly neuronOutputU32: Uint32Array;
  readonly synapseStaticF32: Float32Array;
  readonly synapseStaticU32: Uint32Array;
  readonly synapseDynamic: Float32Array;
  readonly synapseTrace: Float32Array;
}

interface StagingSlot {
  readonly buffer: GPUBuffer;
  state: StagingState;
  /** Bumped whenever the slot is reused or destroyed, to fence stale callbacks. */
  stamp: number;
  epoch: number;
  neurons: number;
  synapses: number;
  plasticity: boolean;
}

/* ----------------------------------------------------------------- class -- */

export class GpuIntegrator implements Integrator {
  readonly backend = 'webgpu' as const;

  private readonly device: GPUDevice;
  private readonly uniformSlotBytes: number;

  /**
   * The reference implementation, used for `reset` only. Rest state — and in
   * particular the Hodgkin-Huxley gate equilibrium that stops a spurious onset
   * spike — is defined by `CpuIntegrator`, and deriving it a second time here
   * would let the two definitions drift.
   */
  private readonly reference = new CpuIntegrator();

  private pipelines: Pipelines | null;
  private storage: Storage | null = null;
  private groups: Groups | null = null;
  private streams: Streams | null = null;
  private scratch: Scratch | null = null;
  private readback: ReadbackLayout | null = null;
  private staging: StagingSlot[] = [];

  private uploadedNeurons = -1;
  private uploadedSynapses = -1;
  private uploadedVersion = -1;
  private topologyVersion = 0;

  private ringSlots = 0;
  private ringCursor = 0;

  /** Cached longest conduction delay, and the topology it was measured on. */
  private longestDelay = 0;
  private delayScanVersion = -1;
  private delayScanCount = -1;

  /** Bumped whenever GPU state is re-initialised; fences in-flight readbacks. */
  private epoch = 0;

  private substep = 0;
  private spikeTicket = 0;
  private pendingSpikes = 0;

  private writeIndex = 0;
  private readIndex = 0;

  private lost = false;
  private disposed = false;
  private overCapacity = false;

  private constructor(device: GPUDevice, pipelines: Pipelines) {
    this.device = device;
    this.pipelines = pipelines;
    this.uniformSlotBytes = alignUp(
      Math.max(MAX_UNIFORM_STRUCT_BYTES, device.limits.minUniformBufferOffsetAlignment),
      device.limits.minUniformBufferOffsetAlignment,
    );
    void device.lost.then(() => {
      this.lost = true;
    });
  }

  /**
   * Build the five pipelines. Returns null — never throws — when any of them
   * fails to compile or the device refuses a layout, so the caller can fall
   * through to the next backend.
   */
  static async create(device: GPUDevice): Promise<GpuIntegrator | null> {
    try {
      device.pushErrorScope('validation');
      let pipelines: Pipelines | null = null;
      try {
        const deliverLayout = layoutFromBindings(device, DELIVER_BINDINGS, 'nf-deliver');
        const propagateLayout = layoutFromBindings(
          device,
          SYNAPSE_PROPAGATE_BINDINGS,
          'nf-propagate',
        );
        const integrateLayout = layoutFromBindings(
          device,
          NEURON_INTEGRATE_BINDINGS,
          'nf-integrate',
        );
        const captureLayout = layoutFromBindings(device, CAPTURE_BINDINGS, 'nf-capture');
        const stdpLayout = layoutFromBindings(device, SYNAPSE_STDP_BINDINGS, 'nf-stdp');

        const [deliver, propagate, integrate, capture, stdp] = await Promise.all([
          createPipeline(device, AXON_DELIVER_WGSL, deliverLayout, 'nf-deliver'),
          createPipeline(device, SYNAPSE_PROPAGATE_WGSL, propagateLayout, 'nf-propagate'),
          createPipeline(device, NEURON_INTEGRATE_WGSL, integrateLayout, 'nf-integrate'),
          createPipeline(device, SPIKE_CAPTURE_WGSL, captureLayout, 'nf-capture'),
          createPipeline(device, SYNAPSE_STDP_WGSL, stdpLayout, 'nf-stdp'),
        ]);

        pipelines = {
          deliver,
          propagate,
          integrate,
          capture,
          stdp,
          deliverLayout,
          propagateLayout,
          integrateLayout,
          captureLayout,
          stdpLayout,
        };
      } finally {
        if ((await device.popErrorScope()) !== null) pipelines = null;
      }
      return pipelines === null ? null : new GpuIntegrator(device, pipelines);
    } catch {
      return null;
    }
  }

  /** A structural edit happened; every GPU mirror of the topology is stale. */
  invalidate(): void {
    this.topologyVersion += 1;
  }

  step(buffers: SimulationBuffers, settings: SimulationSettings, steps: number): StepResult {
    const started = performance.now();
    if (this.pipelines === null || this.lost || this.disposed) {
      return { steps: 0, spikes: 0, simMs: 0 };
    }

    this.drainStaging(buffers);

    if (buffers.neurons.count === 0 || steps <= 0 || !(settings.dt > 0)) {
      return { steps: 0, spikes: this.takeSpikes(), simMs: performance.now() - started };
    }

    if (!this.ensureUploaded(buffers, settings)) {
      return { steps: 0, spikes: this.takeSpikes(), simMs: performance.now() - started };
    }

    this.uploadPerStep(buffers);

    // Time advances per batch rather than once at the end, because the uniforms
    // of the next batch are written from it and would otherwise replay the
    // simulated interval the previous batch already covered.
    let done = 0;
    while (done < steps) {
      const batch = Math.min(steps - done, MAX_BATCH_SUBSTEPS);
      this.encodeBatch(buffers, settings, batch, done + batch >= steps);
      buffers.time += batch * settings.dt;
      done += batch;
    }

    return { steps: done, spikes: this.takeSpikes(), simMs: performance.now() - started };
  }

  /**
   * Return every neuron to rest without changing topology.
   *
   * The host buffers are reset by the reference implementation and the GPU
   * mirror is dropped, so the next step re-uploads the rested state instead of
   * carrying the old one forward. Bumping the epoch discards any readback still
   * in flight, which would otherwise overwrite the fresh state with stale
   * voltages a frame later.
   */
  reset(buffers: SimulationBuffers): void {
    this.reference.reset(buffers);
    this.uploadedVersion = -1;
    this.epoch += 1;
    this.substep = 0;
    this.ringCursor = 0;
    this.spikeTicket = 0;
    this.pendingSpikes = 0;
  }

  /**
   * Destroy every GPU buffer. `GPUComputePipeline` and `GPUBindGroup` have no
   * explicit destructor in WebGPU; dropping the last reference is the whole of
   * their teardown.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.releaseStorage();
    this.releaseStaging();
    this.pipelines = null;
    this.reference.dispose();
  }

  private takeSpikes(): number {
    const spikes = this.pendingSpikes;
    this.pendingSpikes = 0;
    return spikes;
  }

  /* -------------------------------------------------------------- upload -- */

  /**
   * Bring every GPU mirror in line with the host buffers, reallocating when the
   * network changed shape. Returns false when the network cannot be represented
   * within the device's limits, in which case the caller must not dispatch.
   */
  private ensureUploaded(buffers: SimulationBuffers, settings: SimulationSettings): boolean {
    const neurons = buffers.neurons.count;
    const synapses = buffers.synapses.count;
    const ringSlots = this.requiredRingSlots(buffers, settings.dt);

    const shapeChanged =
      neurons !== this.uploadedNeurons ||
      synapses !== this.uploadedSynapses ||
      ringSlots !== this.ringSlots;
    if (!shapeChanged && this.uploadedVersion === this.topologyVersion) return !this.overCapacity;

    if (shapeChanged) {
      this.releaseStorage();
      this.releaseStaging();
      this.overCapacity = !this.allocate(neurons, synapses, ringSlots);
      if (this.overCapacity) return false;
    }

    this.uploadTopology(buffers);
    this.uploadState(buffers);
    this.clearTransient();

    this.uploadedNeurons = neurons;
    this.uploadedSynapses = synapses;
    this.uploadedVersion = this.topologyVersion;
    this.epoch += 1;
    this.spikeTicket = 0;
    this.ringCursor = 0;
    return true;
  }

  /**
   * The longest conduction delay in the network, in ms.
   *
   * Scanning the whole delay column is O(synapses), so it runs only when the
   * topology could have changed rather than on every step: at a hundred thousand
   * synapses the unconditional scan cost more than the dispatch it sized.
   */
  private longestDelayOf(buffers: SimulationBuffers): number {
    const { synapses } = buffers;
    if (this.delayScanVersion === this.topologyVersion && this.delayScanCount === synapses.count) {
      return this.longestDelay;
    }
    let longest = 0;
    for (let s = 0; s < synapses.count; s += 1) {
      const delay = synapses.delay[s];
      if (Number.isFinite(delay) && delay > longest) longest = delay;
    }
    this.longestDelay = longest;
    this.delayScanVersion = this.topologyVersion;
    this.delayScanCount = synapses.count;
    return longest;
  }

  /**
   * Columns the delay ring needs at the current timestep.
   *
   * One column per substep of the longest conduction delay present, plus two so
   * that the largest lead can never alias onto the column being read. Longer
   * delays are clamped by the kernel, which is what the CPU reference does at
   * its calendar queue's horizon.
   */
  private requiredRingSlots(buffers: SimulationBuffers, dt: number): number {
    const { synapses } = buffers;
    const needed = Math.ceil(this.longestDelayOf(buffers) / dt) + 2;
    const columnBytes = Math.max(1, synapses.count) * 4;
    const budget = Math.min(DELAY_RING_BYTE_BUDGET, this.device.limits.maxStorageBufferBindingSize);
    const affordable = Math.max(2, Math.floor(budget / columnBytes));
    return Math.max(2, Math.min(needed, affordable));
  }

  private allocate(neurons: number, synapses: number, ringSlots: number): boolean {
    const device = this.device;
    const pipelines = this.pipelines;
    if (pipelines === null) return false;

    const n = Math.max(1, neurons);
    const m = Math.max(1, synapses);
    const limit = device.limits.maxStorageBufferBindingSize;
    const groupLimit = device.limits.maxComputeWorkgroupsPerDimension;

    const ringBytes = ringSlots * m * 4;
    const largest = Math.max(
      n * NEURON_PARAM_STRIDE * 4,
      m * SYNAPSE_PARAM_STRIDE * 4,
      m * SYNAPSE_STATIC_WORDS * 4,
      ringBytes,
    );
    if (largest > limit) return false;
    if (Math.ceil(n / WGSL_WORKGROUP_SIZE) > groupLimit) return false;
    if (Math.ceil(m / WGSL_WORKGROUP_SIZE) > groupLimit) return false;

    const storageOnly = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const readable = storageOnly | GPUBufferUsage.COPY_SRC;
    const make = (bytes: number, usage: number, label: string): GPUBuffer =>
      device.createBuffer({ label, size: bytes, usage });

    const storage: Storage = {
      neuronParams: make(n * NEURON_PARAM_STRIDE * 4, storageOnly, 'nf-neuron-params'),
      neuronStatic: make(n * NEURON_STATIC_WORDS * 4, storageOnly, 'nf-neuron-static'),
      neuronDynamic: make(n * NEURON_DYNAMIC_WORDS * 4, readable, 'nf-neuron-dynamic'),
      neuronOutput: make(n * NEURON_OUTPUT_WORDS * 4, readable, 'nf-neuron-output'),
      current: make(n * 4, readable, 'nf-current'),
      spikeSink: make(SPIKE_META_WORDS * 4, storageOnly, 'nf-spike-sink'),
      synapseStatic: make(m * SYNAPSE_STATIC_WORDS * 4, storageOnly, 'nf-synapse-static'),
      synapseParams: make(m * SYNAPSE_PARAM_STRIDE * 4, storageOnly, 'nf-synapse-params'),
      synapseWeight: make(m * 4, readable, 'nf-synapse-weight'),
      synapseDynamic: make(m * SYNAPSE_DYNAMIC_WORDS * 4, readable, 'nf-synapse-dynamic'),
      synapseTrace: make(m * SYNAPSE_TRACE_WORDS * 4, readable, 'nf-synapse-trace'),
      arrival: make(m * 4, storageOnly, 'nf-arrival'),
      ring: make(ringBytes, storageOnly, 'nf-delay-ring'),
      spikeMeta: make(SPIKE_META_WORDS * 4, readable, 'nf-spike-meta'),
      spikeEvent: make(SPIKE_RING_CAPACITY * SPIKE_EVENT_WORDS * 4, readable, 'nf-spike-events'),
    };

    const streams = this.createStreams();
    const layout = buildReadbackLayout(n, m);

    // The two views of an interleaved record must alias one allocation, which
    // is why the backing store is created before the views rather than by each.
    const neuronStatic = new ArrayBuffer(n * NEURON_STATIC_WORDS * 4);
    const neuronOutput = new ArrayBuffer(n * NEURON_OUTPUT_WORDS * 4);
    const synapseStatic = new ArrayBuffer(m * SYNAPSE_STATIC_WORDS * 4);

    this.storage = storage;
    this.streams = streams;
    this.groups = this.createGroups(storage, pipelines, streams);
    this.scratch = {
      neuronStaticF32: new Float32Array(neuronStatic),
      neuronStaticU32: new Uint32Array(neuronStatic),
      neuronDynamic: new Float32Array(n * NEURON_DYNAMIC_WORDS),
      neuronOutputF32: new Float32Array(neuronOutput),
      neuronOutputU32: new Uint32Array(neuronOutput),
      synapseStaticF32: new Float32Array(synapseStatic),
      synapseStaticU32: new Uint32Array(synapseStatic),
      synapseDynamic: new Float32Array(m * SYNAPSE_DYNAMIC_WORDS),
      synapseTrace: new Float32Array(m * SYNAPSE_TRACE_WORDS),
    };
    this.readback = layout;
    this.staging = [];
    for (let index = 0; index < STAGING_SLOTS; index += 1) {
      this.staging.push({
        buffer: device.createBuffer({
          label: `nf-readback-${index}`,
          size: layout.bytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        state: 'free',
        stamp: 0,
        epoch: -1,
        neurons: 0,
        synapses: 0,
        plasticity: false,
      });
    }
    this.writeIndex = 0;
    this.readIndex = 0;
    this.ringSlots = ringSlots;
    return true;
  }

  private createStreams(): Streams {
    const device = this.device;
    const stride = this.uniformSlotBytes / 4;
    const build = (label: string): UniformStream => {
      const words = new ArrayBuffer(MAX_BATCH_SUBSTEPS * this.uniformSlotBytes);
      return {
        gpu: device.createBuffer({
          label,
          size: words.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        f32: new Float32Array(words),
        u32: new Uint32Array(words),
        stride,
      };
    };
    const deliver = build('nf-uniform-deliver');
    const propagate = build('nf-uniform-propagate');
    const integrate = build('nf-uniform-integrate');
    const capture = build('nf-uniform-capture');
    const stdp = build('nf-uniform-stdp');
    return {
      deliver,
      propagate,
      integrate,
      capture,
      stdp,
      all: [deliver, propagate, integrate, capture, stdp],
    };
  }

  private createGroups(storage: Storage, pipelines: Pipelines, streams: Streams): Groups {
    const device = this.device;
    const size = this.uniformSlotBytes;
    const uniform = (stream: UniformStream): GPUBindGroupEntry => ({
      binding: 0,
      resource: { buffer: stream.gpu, offset: 0, size },
    });
    const bind = (buffer: GPUBuffer, binding: number): GPUBindGroupEntry => ({
      binding,
      resource: { buffer },
    });

    return {
      deliver: device.createBindGroup({
        label: 'nf-deliver',
        layout: pipelines.deliverLayout,
        entries: [
          uniform(streams.deliver),
          bind(storage.synapseStatic, 1),
          bind(storage.neuronOutput, 2),
          bind(storage.ring, 3),
          bind(storage.arrival, 4),
        ],
      }),
      propagate: device.createBindGroup({
        label: 'nf-propagate',
        layout: pipelines.propagateLayout,
        entries: [
          uniform(streams.propagate),
          bind(storage.synapseStatic, 1),
          bind(storage.synapseWeight, 2),
          bind(storage.synapseParams, 3),
          bind(storage.synapseDynamic, 4),
          bind(storage.arrival, 5),
          bind(storage.current, 6),
          bind(storage.neuronDynamic, 7),
        ],
      }),
      integrate: device.createBindGroup({
        label: 'nf-integrate',
        layout: pipelines.integrateLayout,
        entries: [
          uniform(streams.integrate),
          bind(storage.neuronParams, 1),
          bind(storage.neuronStatic, 2),
          bind(storage.neuronDynamic, 3),
          bind(storage.neuronOutput, 4),
          bind(storage.current, 5),
          bind(storage.spikeSink, 6),
        ],
      }),
      capture: device.createBindGroup({
        label: 'nf-capture',
        layout: pipelines.captureLayout,
        entries: [
          uniform(streams.capture),
          bind(storage.neuronOutput, 1),
          bind(storage.spikeMeta, 2),
          bind(storage.spikeEvent, 3),
        ],
      }),
      stdp: device.createBindGroup({
        label: 'nf-stdp',
        layout: pipelines.stdpLayout,
        entries: [
          uniform(streams.stdp),
          bind(storage.synapseStatic, 1),
          bind(storage.synapseParams, 2),
          bind(storage.synapseWeight, 3),
          bind(storage.synapseTrace, 4),
          bind(storage.neuronOutput, 5),
        ],
      }),
    };
  }

  private uploadTopology(buffers: SimulationBuffers): void {
    const storage = this.storage;
    const scratch = this.scratch;
    if (storage === null || scratch === null) return;
    const queue = this.device.queue;
    const { neurons, synapses } = buffers;
    const n = neurons.count;
    const m = synapses.count;

    if (n > 0) {
      queue.writeBuffer(storage.neuronParams, 0, neurons.params, 0, n * NEURON_PARAM_STRIDE);
    }
    if (m === 0) return;

    queue.writeBuffer(storage.synapseParams, 0, synapses.params, 0, m * SYNAPSE_PARAM_STRIDE);
    queue.writeBuffer(storage.synapseWeight, 0, synapses.weight, 0, m);

    const f32 = scratch.synapseStaticF32;
    const u32 = scratch.synapseStaticU32;
    for (let s = 0; s < m; s += 1) {
      const b = s * SYNAPSE_STATIC_WORDS;
      u32[b] = synapses.pre[s];
      u32[b + 1] = synapses.post[s];
      u32[b + 2] = packMeta(synapses.receptor[s], synapses.plasticity[s], synapses.enabled[s], 0);
      f32[b + 3] = Math.max(0, finiteOr(synapses.delay[s], 0));
      f32[b + 4] = finiteOr(synapses.releaseProb[s], 1);
      f32[b + 5] = finiteOr(synapses.tauRise[s], 0);
      f32[b + 6] = finiteOr(synapses.tauDecay[s], 0);
      f32[b + 7] = finiteOr(synapses.eRev[s], 0);
      f32[b + 8] = finiteOr(synapses.mgBlock[s], 0);
      f32[b + 9] = finiteOr(synapses.arc[s], 0);
    }
    queue.writeBuffer(storage.synapseStatic, 0, f32, 0, m * SYNAPSE_STATIC_WORDS);
  }

  /** Push the host's current state into the GPU mirror. */
  private uploadState(buffers: SimulationBuffers): void {
    const storage = this.storage;
    const scratch = this.scratch;
    if (storage === null || scratch === null) return;
    const queue = this.device.queue;
    const { neurons, synapses } = buffers;
    const n = neurons.count;
    const m = synapses.count;

    if (n > 0) {
      const dynamic = scratch.neuronDynamic;
      const outF = scratch.neuronOutputF32;
      const outU = scratch.neuronOutputU32;
      for (let i = 0; i < n; i += 1) {
        const d = i * NEURON_DYNAMIC_WORDS;
        dynamic[d] = neurons.v[i];
        dynamic[d + 1] = neurons.w[i];
        dynamic[d + 2] = neurons.gateM[i];
        dynamic[d + 3] = neurons.gateH[i];
        dynamic[d + 4] = neurons.gateN[i];
        dynamic[d + 5] = neurons.calcium[i];
        dynamic[d + 6] = neurons.lastSpike[i];
        dynamic[d + 7] = neurons.refractoryUntil[i];

        const o = i * NEURON_OUTPUT_WORDS;
        outU[o] = neurons.spike[i];
        outU[o + 1] = neurons.spikeCount[i];
        outF[o + 2] = neurons.flash[i];
        outF[o + 3] = neurons.rate[i];
      }
      queue.writeBuffer(storage.neuronDynamic, 0, dynamic, 0, n * NEURON_DYNAMIC_WORDS);
      queue.writeBuffer(storage.neuronOutput, 0, outF, 0, n * NEURON_OUTPUT_WORDS);
    }

    if (m > 0) {
      const dynamic = scratch.synapseDynamic;
      const traces = scratch.synapseTrace;
      for (let s = 0; s < m; s += 1) {
        const d = s * SYNAPSE_DYNAMIC_WORDS;
        dynamic[d] = synapses.gRise[s];
        dynamic[d + 1] = synapses.gDecay[s];
        dynamic[d + 2] = synapses.stpR[s];
        dynamic[d + 3] = synapses.stpU[s];
        dynamic[d + 4] = synapses.activity[s];

        const t = s * SYNAPSE_TRACE_WORDS;
        traces[t] = synapses.preTrace[s];
        traces[t + 1] = synapses.postTrace[s];
        traces[t + 2] = synapses.preTraceSlow[s];
        traces[t + 3] = synapses.postTraceSlow[s];
      }
      queue.writeBuffer(storage.synapseDynamic, 0, dynamic, 0, m * SYNAPSE_DYNAMIC_WORDS);
      queue.writeBuffer(storage.synapseTrace, 0, traces, 0, m * SYNAPSE_TRACE_WORDS);
    }
  }

  /** Zero everything that carries no meaning across a re-upload. */
  private clearTransient(): void {
    const storage = this.storage;
    if (storage === null) return;
    const encoder = this.device.createCommandEncoder({ label: 'nf-clear' });
    encoder.clearBuffer(storage.current);
    encoder.clearBuffer(storage.arrival);
    encoder.clearBuffer(storage.ring);
    encoder.clearBuffer(storage.spikeMeta);
    encoder.clearBuffer(storage.spikeEvent);
    encoder.clearBuffer(storage.spikeSink);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Re-upload the columns the host rewrites between steps.
   *
   * `iExt` is rewritten by `applyStimuli` and by the poke tool every frame, so
   * the whole static record goes up each step rather than being diffed.
   *
   * The noise column travels as the raw per-neuron amplitude. The integrate
   * kernel applies the 1/sqrt(dt) white-noise scaling itself — see
   * `whiteNoiseScale` in `NEURON_INTEGRATE_WGSL` — exactly as the reference
   * does, so pre-scaling it here would apply the factor twice and inflate every
   * noise current by 1/sqrt(dt): a further 3.2x at the default dt = 0.1 ms.
   */
  private uploadPerStep(buffers: SimulationBuffers): void {
    const storage = this.storage;
    const scratch = this.scratch;
    if (storage === null || scratch === null) return;
    const { neurons } = buffers;
    const n = neurons.count;
    if (n === 0) return;

    const f32 = scratch.neuronStaticF32;
    const u32 = scratch.neuronStaticU32;
    for (let i = 0; i < n; i += 1) {
      const b = i * NEURON_STATIC_WORDS;
      u32[b] = packMeta(neurons.model[i], neurons.polarity[i], neurons.enabled[i], neurons.flags[i]);
      f32[b + 1] = finiteOr(neurons.iExt[i], 0);
      f32[b + 2] = finiteOr(neurons.bias[i], 0);
      f32[b + 3] = Math.max(0, finiteOr(neurons.noise[i], 0));
    }
    this.device.queue.writeBuffer(storage.neuronStatic, 0, f32, 0, n * NEURON_STATIC_WORDS);
  }

  /* ------------------------------------------------------------ dispatch -- */

  private encodeBatch(
    buffers: SimulationBuffers,
    settings: SimulationSettings,
    batch: number,
    final: boolean,
  ): void {
    const pipelines = this.pipelines;
    const storage = this.storage;
    const groups = this.groups;
    const readback = this.readback;
    if (pipelines === null || storage === null || groups === null || readback === null) return;

    const neuronCount = buffers.neurons.count;
    const synapseCount = buffers.synapses.count;
    const neuronGroups = Math.ceil(neuronCount / WGSL_WORKGROUP_SIZE);
    const synapseGroups = Math.ceil(synapseCount / WGSL_WORKGROUP_SIZE);
    const plasticity = settings.plasticityEnabled && synapseCount > 0;

    this.writeUniforms(buffers, settings, batch, neuronCount, synapseCount);

    const target = final ? this.acquireStaging() : null;
    const device = this.device;
    const encoder = device.createCommandEncoder({ label: 'nf-step' });
    let pass = encoder.beginComputePass({ label: 'nf-substeps' });

    for (let s = 0; s < batch; s += 1) {
      const offset = s * this.uniformSlotBytes;

      if (synapseGroups > 0) {
        pass.setPipeline(pipelines.deliver);
        pass.setBindGroup(0, groups.deliver, [offset]);
        pass.dispatchWorkgroups(synapseGroups);

        pass.setPipeline(pipelines.propagate);
        pass.setBindGroup(0, groups.propagate, [offset]);
        pass.dispatchWorkgroups(synapseGroups);
      }

      // The integrate kernel consumes and clears the current accumulator, so
      // the only moment `iSyn` holds the drive the membranes actually saw is
      // between the final propagate and the final integrate. The pass is split
      // there because copies may not be encoded inside a compute pass.
      if (target !== null && s === batch - 1) {
        pass.end();
        encoder.copyBufferToBuffer(
          storage.current,
          0,
          target.buffer,
          readback.current.offset,
          readback.current.bytes,
        );
        pass = encoder.beginComputePass({ label: 'nf-final-substep' });
      }

      pass.setPipeline(pipelines.integrate);
      pass.setBindGroup(0, groups.integrate, [offset]);
      pass.dispatchWorkgroups(neuronGroups);

      pass.setPipeline(pipelines.capture);
      pass.setBindGroup(0, groups.capture, [offset]);
      pass.dispatchWorkgroups(neuronGroups);

      if (plasticity) {
        pass.setPipeline(pipelines.stdp);
        pass.setBindGroup(0, groups.stdp, [offset]);
        pass.dispatchWorkgroups(synapseGroups);
      }
    }
    pass.end();

    this.substep += batch;
    this.ringCursor = (this.ringCursor + batch) % this.ringSlots;

    if (target !== null) this.encodeReadback(encoder, target, buffers, plasticity);
    device.queue.submit([encoder.finish()]);
  }

  /**
   * Take the next staging buffer, or null when both are still busy.
   *
   * Marking it in flight here rather than after the copies is what keeps the
   * ring a strict FIFO: `writeIndex` and the slot's state always move together,
   * so a skipped readback can never leave a hole that stalls the reader.
   */
  private acquireStaging(): StagingSlot | null {
    if (this.staging.length === 0) return null;
    const slot = this.staging[this.writeIndex % this.staging.length];
    if (slot.state !== 'free') return null;
    slot.state = 'inflight';
    slot.stamp += 1;
    this.writeIndex += 1;
    return slot;
  }

  private encodeReadback(
    encoder: GPUCommandEncoder,
    slot: StagingSlot,
    buffers: SimulationBuffers,
    plasticity: boolean,
  ): void {
    const storage = this.storage;
    const readback = this.readback;
    if (storage === null || readback === null) return;

    const copy = (source: GPUBuffer, section: Section): void => {
      if (section.bytes > 0) {
        encoder.copyBufferToBuffer(source, 0, slot.buffer, section.offset, section.bytes);
      }
    };
    copy(storage.neuronDynamic, readback.neuronDynamic);
    copy(storage.neuronOutput, readback.neuronOutput);
    copy(storage.synapseDynamic, readback.synapseDynamic);
    copy(storage.spikeMeta, readback.spikeMeta);
    copy(storage.spikeEvent, readback.spikeEvent);
    if (plasticity) {
      copy(storage.synapseWeight, readback.synapseWeight);
      copy(storage.synapseTrace, readback.synapseTrace);
    }

    slot.epoch = this.epoch;
    slot.neurons = buffers.neurons.count;
    slot.synapses = buffers.synapses.count;
    slot.plasticity = plasticity;

    const stamp = slot.stamp;
    void slot.buffer.mapAsync(GPUMapMode.READ).then(
      () => {
        if (slot.stamp === stamp && slot.state === 'inflight') slot.state = 'ready';
      },
      () => {
        if (slot.stamp === stamp && slot.state === 'inflight') slot.state = 'failed';
      },
    );
  }

  private writeUniforms(
    buffers: SimulationBuffers,
    settings: SimulationSettings,
    batch: number,
    neuronCount: number,
    synapseCount: number,
  ): void {
    const streams = this.streams;
    if (streams === null) return;
    const dt = settings.dt;
    const seed = settings.seed >>> 0;
    const gain = settings.gain;
    const invDt = 1 / dt;
    // Raw amplitude, not pre-scaled: the kernel adds this to the per-neuron
    // column and applies 1/sqrt(dt) to the sum, as the reference does.
    const globalNoise = Math.max(0, settings.noise);
    // rk2 and rk4 have no GPU counterpart. Exponential Euler is stabler and
    // cheaper than either at the timesteps this app runs, and it is what the
    // reference uses for every linear relaxation.
    const mode = settings.integrator === 'euler' ? 0 : 1;
    const stride = streams.deliver.stride;

    for (let s = 0; s < batch; s += 1) {
      const base = s * stride;
      const time = buffers.time + (s + 1) * dt;
      const substep = (this.substep + s) >>> 0;

      const d = streams.deliver;
      d.u32[base] = synapseCount;
      d.u32[base + 1] = this.ringSlots;
      d.u32[base + 2] = (this.ringCursor + s) % this.ringSlots;
      d.u32[base + 3] = seed;
      d.u32[base + 4] = substep;
      d.f32[base + 5] = invDt;

      const p = streams.propagate;
      p.f32[base] = dt;
      p.f32[base + 1] = time;
      p.f32[base + 2] = gain;
      p.f32[base + 3] = MG_CONCENTRATION;
      p.f32[base + 4] = ACTIVITY_TAU;
      p.u32[base + 5] = synapseCount;
      p.u32[base + 6] = seed;
      p.u32[base + 7] = substep;

      const i = streams.integrate;
      i.f32[base] = dt;
      i.f32[base + 1] = time;
      i.f32[base + 2] = globalNoise;
      i.f32[base + 3] = FLASH_TAU;
      i.f32[base + 4] = RATE_TAU;
      i.f32[base + 5] = CALCIUM_TAU;
      i.f32[base + 6] = CALCIUM_GAIN;
      i.u32[base + 7] = neuronCount;
      i.u32[base + 8] = seed;
      i.u32[base + 9] = substep;
      // Spike identities are captured by SPIKE_CAPTURE_WGSL, which also records
      // the time each one happened. A capacity of zero leaves this kernel's own
      // ring unused — an explicitly supported configuration — while keeping its
      // required binding satisfied.
      i.u32[base + 10] = 0;
      i.u32[base + 11] = mode;

      const c = streams.capture;
      c.u32[base] = neuronCount;
      c.u32[base + 1] = SPIKE_RING_CAPACITY;
      c.f32[base + 2] = time;

      const t = streams.stdp;
      t.f32[base] = dt;
      t.f32[base + 1] = time;
      t.f32[base + 2] = GLOBAL_LEARNING_RATE;
      t.u32[base + 3] = synapseCount;
    }

    const queue = this.device.queue;
    const bytes = batch * this.uniformSlotBytes;
    for (const stream of streams.all) {
      queue.writeBuffer(stream.gpu, 0, stream.f32.buffer, 0, bytes);
    }
  }

  /* ------------------------------------------------------------ readback -- */

  private drainStaging(buffers: SimulationBuffers): void {
    while (this.staging.length > 0) {
      const slot = this.staging[this.readIndex % this.staging.length];
      if (slot.state === 'failed') {
        // A rejected map leaves the buffer unmapped, so the slot is reusable;
        // the ticket is absolute, so the next readback recovers the lost count.
        slot.state = 'free';
        this.readIndex += 1;
        continue;
      }
      if (slot.state !== 'ready') return;
      this.unpack(slot, buffers);
      slot.buffer.unmap();
      slot.state = 'free';
      this.readIndex += 1;
    }
  }

  private unpack(slot: StagingSlot, buffers: SimulationBuffers): void {
    const layout = this.readback;
    if (layout === null) return;
    // A reset or a re-upload since this readback was queued means the state it
    // carries describes a network that no longer exists.
    if (slot.epoch !== this.epoch) return;

    const range = slot.buffer.getMappedRange();
    const { neurons, synapses } = buffers;

    const meta = new Uint32Array(range, layout.spikeMeta.offset, SPIKE_META_WORDS);
    const ticket = meta[0] >>> 0;
    const emitted = (ticket - this.spikeTicket) >>> 0;
    this.spikeTicket = ticket;
    this.pendingSpikes += emitted;

    if (emitted > 0) {
      const words = SPIKE_RING_CAPACITY * SPIKE_EVENT_WORDS;
      const events = new Uint32Array(range, layout.spikeEvent.offset, words);
      const times = new Float32Array(range, layout.spikeEvent.offset, words);
      replaySpikeLog(buffers.spikes, emitted, SPIKE_RING_CAPACITY, ticket, events, times, SPIKE_EVENT_WORDS, 1);
    }

    const n = Math.min(slot.neurons, neurons.count);
    if (n > 0) {
      const dynamic = new Float32Array(
        range,
        layout.neuronDynamic.offset,
        slot.neurons * NEURON_DYNAMIC_WORDS,
      );
      const outF = new Float32Array(
        range,
        layout.neuronOutput.offset,
        slot.neurons * NEURON_OUTPUT_WORDS,
      );
      const outU = new Uint32Array(
        range,
        layout.neuronOutput.offset,
        slot.neurons * NEURON_OUTPUT_WORDS,
      );
      const current = new Int32Array(range, layout.current.offset, slot.neurons);

      for (let i = 0; i < n; i += 1) {
        const d = i * NEURON_DYNAMIC_WORDS;
        neurons.v[i] = dynamic[d];
        neurons.w[i] = dynamic[d + 1];
        neurons.gateM[i] = dynamic[d + 2];
        neurons.gateH[i] = dynamic[d + 3];
        neurons.gateN[i] = dynamic[d + 4];
        neurons.calcium[i] = dynamic[d + 5];
        neurons.lastSpike[i] = dynamic[d + 6];
        neurons.refractoryUntil[i] = dynamic[d + 7];

        const o = i * NEURON_OUTPUT_WORDS;
        neurons.spike[i] = outU[o] === 0 ? 0 : 1;
        neurons.spikeCount[i] = outU[o + 1];
        neurons.flash[i] = outF[o + 2];
        neurons.rate[i] = outF[o + 3];
        neurons.iSyn[i] = current[i] * INV_CURRENT_SCALE;
      }
    }

    const m = Math.min(slot.synapses, synapses.count);
    if (m === 0) return;

    const dynamic = new Float32Array(
      range,
      layout.synapseDynamic.offset,
      slot.synapses * SYNAPSE_DYNAMIC_WORDS,
    );
    for (let s = 0; s < m; s += 1) {
      const d = s * SYNAPSE_DYNAMIC_WORDS;
      synapses.gRise[s] = dynamic[d];
      synapses.gDecay[s] = dynamic[d + 1];
      synapses.stpR[s] = dynamic[d + 2];
      synapses.stpU[s] = dynamic[d + 3];
      synapses.activity[s] = dynamic[d + 4];
    }

    // Weights and traces only move while a plasticity rule is running. Copying
    // them back unconditionally would overwrite an edit made between frames
    // with a value the GPU never changed.
    if (!slot.plasticity) return;
    const weight = new Float32Array(range, layout.synapseWeight.offset, slot.synapses);
    const traces = new Float32Array(
      range,
      layout.synapseTrace.offset,
      slot.synapses * SYNAPSE_TRACE_WORDS,
    );
    for (let s = 0; s < m; s += 1) {
      synapses.weight[s] = weight[s];
      const t = s * SYNAPSE_TRACE_WORDS;
      synapses.preTrace[s] = traces[t];
      synapses.postTrace[s] = traces[t + 1];
      synapses.preTraceSlow[s] = traces[t + 2];
      synapses.postTraceSlow[s] = traces[t + 3];
    }
  }

  /* ------------------------------------------------------------ teardown -- */

  private releaseStorage(): void {
    const storage = this.storage;
    if (storage !== null) {
      for (const buffer of Object.values(storage)) buffer.destroy();
    }
    const streams = this.streams;
    if (streams !== null) {
      for (const stream of streams.all) stream.gpu.destroy();
    }
    this.storage = null;
    this.streams = null;
    this.groups = null;
    this.scratch = null;
    this.uploadedNeurons = -1;
    this.uploadedSynapses = -1;
    this.uploadedVersion = -1;
    this.ringSlots = 0;
  }

  private releaseStaging(): void {
    for (const slot of this.staging) {
      // Bumping the stamp first fences the pending map callback, which would
      // otherwise mark a destroyed buffer readable.
      slot.stamp += 1;
      slot.state = 'free';
      slot.buffer.destroy();
    }
    this.staging = [];
    this.readback = null;
    this.writeIndex = 0;
    this.readIndex = 0;
    this.epoch += 1;
  }
}
