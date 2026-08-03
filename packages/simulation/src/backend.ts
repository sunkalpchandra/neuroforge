import type { SimulationBuffers, SimulationSettings } from '@neuroforge/shared';

import { requestComputeDevice } from './capabilities';
import { CpuIntegrator } from './integrator-cpu';
import { GpuIntegrator, replaySpikeLog } from './integrator-gpu';
import { INTEGRATOR_CODE } from './models';
import type { Integrator, StepResult } from './types';

/**
 * Backend selection and the WebAssembly bridge.
 *
 * The three integrators are interchangeable at the `Integrator` interface and
 * differ only in where the arithmetic happens. `CpuIntegrator` is the floor:
 * `createIntegrator` always returns something, so the app never has to handle a
 * missing simulation.
 */

/**
 * Path to the wasm-bindgen glue for `crates/neuroforge-core`.
 *
 * Assembled at runtime rather than written as a literal: the artifact is
 * gitignored and usually absent, and a literal specifier would make the module
 * a compile-time dependency of a package that must build without it.
 */
const WASM_MODULE_PATH = ['..', 'wasm', 'neuroforge_core.js'].join('/');

/**
 * The subset of the Rust core's `Network` this bridge calls.
 *
 * Every `*_ptr` accessor returns a byte offset into wasm linear memory. The
 * crate's contract is that any resize may reallocate *and* grow linear memory,
 * which detaches every view JavaScript holds — so views are rebuilt whenever a
 * resize reports movement or the memory's backing `ArrayBuffer` identity
 * changes, never cached across those calls.
 */
interface WasmNetwork {
  resize_neurons(count: number): boolean;
  resize_synapses(count: number): boolean;
  configure_delays(buckets: number, resolution: number, stride: number): void;
  configure_spike_log(capacity: number): void;
  neuron_capacity(): number;
  synapse_capacity(): number;

  neuron_position_ptr(): number;
  neuron_v_ptr(): number;
  neuron_w_ptr(): number;
  neuron_gate_m_ptr(): number;
  neuron_gate_h_ptr(): number;
  neuron_gate_n_ptr(): number;
  neuron_calcium_ptr(): number;
  neuron_i_syn_ptr(): number;
  neuron_i_ext_ptr(): number;
  neuron_bias_ptr(): number;
  neuron_noise_ptr(): number;
  neuron_spike_ptr(): number;
  neuron_last_spike_ptr(): number;
  neuron_refractory_until_ptr(): number;
  neuron_flash_ptr(): number;
  neuron_rate_ptr(): number;
  neuron_spike_count_ptr(): number;
  neuron_model_ptr(): number;
  neuron_polarity_ptr(): number;
  neuron_enabled_ptr(): number;
  neuron_params_ptr(): number;
  neuron_scale_ptr(): number;
  neuron_seed_ptr(): number;
  neuron_archetype_ptr(): number;
  neuron_population_ptr(): number;
  neuron_flags_ptr(): number;

  synapse_pre_ptr(): number;
  synapse_post_ptr(): number;
  synapse_weight_ptr(): number;
  synapse_delay_ptr(): number;
  synapse_g_rise_ptr(): number;
  synapse_g_decay_ptr(): number;
  synapse_tau_rise_ptr(): number;
  synapse_tau_decay_ptr(): number;
  synapse_e_rev_ptr(): number;
  synapse_mg_block_ptr(): number;
  synapse_pre_trace_ptr(): number;
  synapse_post_trace_ptr(): number;
  synapse_pre_trace_slow_ptr(): number;
  synapse_post_trace_slow_ptr(): number;
  synapse_stp_r_ptr(): number;
  synapse_stp_u_ptr(): number;
  synapse_release_prob_ptr(): number;
  synapse_receptor_ptr(): number;
  synapse_plasticity_ptr(): number;
  synapse_enabled_ptr(): number;
  synapse_params_ptr(): number;
  synapse_activity_ptr(): number;
  synapse_arc_ptr(): number;

  spike_log_neuron_ptr(): number;
  spike_log_time_ptr(): number;
  spike_log_capacity(): number;
  spike_log_head(): number;

  set_seed(seed: number): void;
  time(): number;
  set_time(time: number): void;
  reset(): void;
  step(
    steps: number,
    dt: number,
    integrator: number,
    gain: number,
    noise: number,
    plasticity: boolean,
  ): number;
  free?(): void;
}

type WasmNetworkConstructor = new (
  neuronCapacity: number,
  synapseCapacity: number,
) => WasmNetwork;

interface WasmModule {
  readonly Network: WasmNetworkConstructor;
  readonly memory: WebAssembly.Memory;
}

/** Methods probed at load time to reject a module that is not the core crate. */
const REQUIRED_NETWORK_METHODS: readonly (keyof WasmNetwork)[] = [
  'step',
  'reset',
  'resize_neurons',
  'resize_synapses',
  'neuron_v_ptr',
  'synapse_weight_ptr',
  'spike_log_head',
];

/** Typed views onto the crate's columns, all pointing into wasm memory. */
interface Views {
  /** Identity of the `ArrayBuffer` these views were built on. */
  readonly backing: ArrayBufferLike;
  readonly neuronCapacity: number;
  readonly synapseCapacity: number;

  readonly position: Float32Array;
  readonly v: Float32Array;
  readonly w: Float32Array;
  readonly gateM: Float32Array;
  readonly gateH: Float32Array;
  readonly gateN: Float32Array;
  readonly calcium: Float32Array;
  readonly iSyn: Float32Array;
  readonly iExt: Float32Array;
  readonly bias: Float32Array;
  readonly noise: Float32Array;
  readonly spike: Uint8Array;
  readonly lastSpike: Float32Array;
  readonly refractoryUntil: Float32Array;
  readonly flash: Float32Array;
  readonly rate: Float32Array;
  readonly spikeCount: Uint32Array;
  readonly model: Uint8Array;
  readonly polarity: Uint8Array;
  readonly neuronEnabled: Uint8Array;
  readonly neuronParams: Float32Array;
  readonly scale: Float32Array;
  readonly seed: Uint32Array;
  readonly archetype: Uint8Array;
  readonly population: Uint16Array;
  readonly flags: Uint8Array;

  readonly pre: Uint32Array;
  readonly post: Uint32Array;
  readonly weight: Float32Array;
  readonly delay: Float32Array;
  readonly gRise: Float32Array;
  readonly gDecay: Float32Array;
  readonly tauRise: Float32Array;
  readonly tauDecay: Float32Array;
  readonly eRev: Float32Array;
  readonly mgBlock: Float32Array;
  readonly preTrace: Float32Array;
  readonly postTrace: Float32Array;
  readonly preTraceSlow: Float32Array;
  readonly postTraceSlow: Float32Array;
  readonly stpR: Float32Array;
  readonly stpU: Float32Array;
  readonly releaseProb: Float32Array;
  readonly receptor: Uint8Array;
  readonly plasticity: Uint8Array;
  readonly synapseEnabled: Uint8Array;
  readonly synapseParams: Float32Array;
  readonly activity: Float32Array;
  readonly arc: Float32Array;

  readonly logNeuron: Uint32Array;
  readonly logTime: Float32Array;
}

/** Floats per neuron and per synapse in the crate's packed parameter blocks. */
const NEURON_PARAM_STRIDE = 16;
const SYNAPSE_PARAM_STRIDE = 12;

function readMemory(value: unknown): WebAssembly.Memory | null {
  if (value instanceof WebAssembly.Memory) return value;
  if (typeof value === 'object' && value !== null) {
    const candidate = (value as { memory?: unknown }).memory;
    if (candidate instanceof WebAssembly.Memory) return candidate;
  }
  return null;
}

/**
 * Narrow a freshly imported module to the shape this bridge needs.
 *
 * wasm-bindgen's `--target web` output exports the classes directly and returns
 * the instance's exports — including `memory` — from its default init function;
 * other targets export `memory` from the module itself. Both are probed, and
 * anything else is rejected rather than half-adopted.
 */
async function adoptModule(imported: unknown): Promise<WasmModule | null> {
  if (typeof imported !== 'object' || imported === null) return null;
  const record = imported as Record<string, unknown>;

  let memory = readMemory(record);
  if (memory === null && typeof record.default === 'function') {
    const init = record.default as () => Promise<unknown>;
    memory = readMemory(await init());
  }
  if (memory === null && typeof record.initSync === 'function') {
    const initSync = record.initSync as () => unknown;
    memory = readMemory(initSync());
  }
  if (memory === null) return null;

  const constructor = record.Network;
  if (typeof constructor !== 'function') return null;
  const prototype = (constructor as { prototype?: unknown }).prototype;
  if (typeof prototype !== 'object' || prototype === null) return null;
  const methods = prototype as Record<string, unknown>;
  for (const name of REQUIRED_NETWORK_METHODS) {
    if (typeof methods[name] !== 'function') return null;
  }

  return { Network: constructor as WasmNetworkConstructor, memory };
}

/**
 * The loaded module, or null once a load attempt has failed.
 *
 * Cached as a promise so that concurrent callers share one import and a missing
 * artifact costs exactly one failed resolution for the lifetime of the page.
 */
let modulePromise: Promise<WasmModule | null> | null = null;

function loadModule(): Promise<WasmModule | null> {
  if (modulePromise === null) {
    modulePromise = (async () => {
      if (typeof WebAssembly === 'undefined') return null;
      try {
        const imported: unknown = await import(WASM_MODULE_PATH);
        return await adoptModule(imported);
      } catch {
        return null;
      }
    })();
  }
  return modulePromise;
}

/**
 * Native integrator backed by `crates/neuroforge-core` compiled to WebAssembly.
 *
 * The crate owns its arrays; this class copies the columns across the boundary
 * rather than handing over the app's own buffers, because the renderer reads
 * those same `SimulationBuffers` directly and cannot follow a wasm heap that
 * moves underneath it. Static columns cross once per structural edit, the
 * external-current column and the state columns cross once per step.
 */
export class WasmIntegrator implements Integrator {
  readonly backend = 'wasm' as const;

  private readonly network: WasmNetwork;
  private readonly memory: WebAssembly.Memory;

  /** Rest state is defined by the reference implementation, not re-derived. */
  private readonly reference = new CpuIntegrator();

  private views: Views | null = null;
  private uploadedNeurons = -1;
  private uploadedSynapses = -1;
  private uploadedVersion = -1;
  private topologyVersion = 0;
  private logHead = 0;
  private disposed = false;

  private constructor(network: WasmNetwork, memory: WebAssembly.Memory) {
    this.network = network;
    this.memory = memory;
  }

  /**
   * Whether a WASM backend could exist at all.
   *
   * True only says the runtime supports WebAssembly and no load has failed yet;
   * `create` is what actually proves the artifact is present.
   */
  static isAvailable(): boolean {
    return typeof WebAssembly !== 'undefined';
  }

  /** Returns null — never throws — when the compiled core is not present. */
  static async create(): Promise<WasmIntegrator | null> {
    const module = await loadModule();
    if (module === null) return null;
    try {
      return new WasmIntegrator(new module.Network(1024, 4096), module.memory);
    } catch {
      return null;
    }
  }

  invalidate(): void {
    this.topologyVersion += 1;
  }

  step(buffers: SimulationBuffers, settings: SimulationSettings, steps: number): StepResult {
    const started = performance.now();
    if (this.disposed || buffers.neurons.count === 0 || steps <= 0 || !(settings.dt > 0)) {
      return { steps: 0, spikes: 0, simMs: 0 };
    }

    const views = this.synchronise(buffers, settings);
    const n = buffers.neurons.count;
    views.iExt.set(buffers.neurons.iExt.subarray(0, n));

    if (Math.abs(this.network.time() - buffers.time) > 1e-9) {
      this.network.set_time(buffers.time);
    }

    const spikes = this.network.step(
      steps,
      settings.dt,
      INTEGRATOR_CODE[settings.integrator],
      settings.gain,
      settings.noise,
      settings.plasticityEnabled,
    );

    // `step` rebuilds the crate's adjacency, which allocates and can therefore
    // grow linear memory — detaching every view built on the old ArrayBuffer.
    this.download(buffers, this.currentViews(), settings.plasticityEnabled);
    buffers.time = this.network.time();
    return { steps, spikes, simMs: performance.now() - started };
  }

  reset(buffers: SimulationBuffers): void {
    if (this.disposed) return;
    // The crate has its own reset, but rest state must be identical across all
    // three backends, so the reference defines it and the result is uploaded.
    this.reference.reset(buffers);
    this.network.reset();
    this.logHead = 0;
    this.uploadedVersion = -1;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.views = null;
    this.reference.dispose();
    this.network.free?.();
  }

  /* --------------------------------------------------------- marshalling -- */

  /** Resize the crate's storage to match the document and refresh the views. */
  private synchronise(buffers: SimulationBuffers, settings: SimulationSettings): Views {
    const neurons = buffers.neurons.count;
    const synapses = buffers.synapses.count;
    const stale =
      neurons !== this.uploadedNeurons ||
      synapses !== this.uploadedSynapses ||
      this.uploadedVersion !== this.topologyVersion;

    if (!stale) return this.currentViews();

    const { delays, spikes } = buffers;
    this.network.resize_neurons(neurons);
    this.network.resize_synapses(synapses);
    this.network.configure_delays(delays.buckets, delays.resolution, delays.stride);
    this.network.configure_spike_log(spikes.capacity);
    this.network.set_seed(settings.seed >>> 0);
    this.logHead = 0;

    const views = this.buildViews();
    this.views = views;
    this.upload(buffers, views);

    this.uploadedNeurons = neurons;
    this.uploadedSynapses = synapses;
    this.uploadedVersion = this.topologyVersion;
    return views;
  }

  /** The live views, rebuilt whenever wasm memory grew out from under them. */
  private currentViews(): Views {
    const views = this.views;
    if (views !== null && views.backing === this.memory.buffer) return views;
    const rebuilt = this.buildViews();
    this.views = rebuilt;
    return rebuilt;
  }

  private buildViews(): Views {
    const network = this.network;
    const backing = this.memory.buffer;
    const n = network.neuron_capacity();
    const m = network.synapse_capacity();
    const logCapacity = network.spike_log_capacity();

    const f32 = (ptr: number, length: number): Float32Array =>
      new Float32Array(backing, ptr, length);
    const u32 = (ptr: number, length: number): Uint32Array => new Uint32Array(backing, ptr, length);
    const u16 = (ptr: number, length: number): Uint16Array => new Uint16Array(backing, ptr, length);
    const u8 = (ptr: number, length: number): Uint8Array => new Uint8Array(backing, ptr, length);

    return {
      backing,
      neuronCapacity: n,
      synapseCapacity: m,

      position: f32(network.neuron_position_ptr(), n * 3),
      v: f32(network.neuron_v_ptr(), n),
      w: f32(network.neuron_w_ptr(), n),
      gateM: f32(network.neuron_gate_m_ptr(), n),
      gateH: f32(network.neuron_gate_h_ptr(), n),
      gateN: f32(network.neuron_gate_n_ptr(), n),
      calcium: f32(network.neuron_calcium_ptr(), n),
      iSyn: f32(network.neuron_i_syn_ptr(), n),
      iExt: f32(network.neuron_i_ext_ptr(), n),
      bias: f32(network.neuron_bias_ptr(), n),
      noise: f32(network.neuron_noise_ptr(), n),
      spike: u8(network.neuron_spike_ptr(), n),
      lastSpike: f32(network.neuron_last_spike_ptr(), n),
      refractoryUntil: f32(network.neuron_refractory_until_ptr(), n),
      flash: f32(network.neuron_flash_ptr(), n),
      rate: f32(network.neuron_rate_ptr(), n),
      spikeCount: u32(network.neuron_spike_count_ptr(), n),
      model: u8(network.neuron_model_ptr(), n),
      polarity: u8(network.neuron_polarity_ptr(), n),
      neuronEnabled: u8(network.neuron_enabled_ptr(), n),
      neuronParams: f32(network.neuron_params_ptr(), n * NEURON_PARAM_STRIDE),
      scale: f32(network.neuron_scale_ptr(), n),
      seed: u32(network.neuron_seed_ptr(), n),
      archetype: u8(network.neuron_archetype_ptr(), n),
      population: u16(network.neuron_population_ptr(), n),
      flags: u8(network.neuron_flags_ptr(), n),

      pre: u32(network.synapse_pre_ptr(), m),
      post: u32(network.synapse_post_ptr(), m),
      weight: f32(network.synapse_weight_ptr(), m),
      delay: f32(network.synapse_delay_ptr(), m),
      gRise: f32(network.synapse_g_rise_ptr(), m),
      gDecay: f32(network.synapse_g_decay_ptr(), m),
      tauRise: f32(network.synapse_tau_rise_ptr(), m),
      tauDecay: f32(network.synapse_tau_decay_ptr(), m),
      eRev: f32(network.synapse_e_rev_ptr(), m),
      mgBlock: f32(network.synapse_mg_block_ptr(), m),
      preTrace: f32(network.synapse_pre_trace_ptr(), m),
      postTrace: f32(network.synapse_post_trace_ptr(), m),
      preTraceSlow: f32(network.synapse_pre_trace_slow_ptr(), m),
      postTraceSlow: f32(network.synapse_post_trace_slow_ptr(), m),
      stpR: f32(network.synapse_stp_r_ptr(), m),
      stpU: f32(network.synapse_stp_u_ptr(), m),
      releaseProb: f32(network.synapse_release_prob_ptr(), m),
      receptor: u8(network.synapse_receptor_ptr(), m),
      plasticity: u8(network.synapse_plasticity_ptr(), m),
      synapseEnabled: u8(network.synapse_enabled_ptr(), m),
      synapseParams: f32(network.synapse_params_ptr(), m * SYNAPSE_PARAM_STRIDE),
      activity: f32(network.synapse_activity_ptr(), m),
      arc: f32(network.synapse_arc_ptr(), m),

      logNeuron: u32(network.spike_log_neuron_ptr(), logCapacity),
      logTime: f32(network.spike_log_time_ptr(), logCapacity),
    };
  }

  /** Copy every column the crate reads, static and state alike. */
  private upload(buffers: SimulationBuffers, views: Views): void {
    const { neurons, synapses } = buffers;
    const n = neurons.count;
    const m = synapses.count;

    views.position.set(neurons.position.subarray(0, n * 3));
    views.v.set(neurons.v.subarray(0, n));
    views.w.set(neurons.w.subarray(0, n));
    views.gateM.set(neurons.gateM.subarray(0, n));
    views.gateH.set(neurons.gateH.subarray(0, n));
    views.gateN.set(neurons.gateN.subarray(0, n));
    views.calcium.set(neurons.calcium.subarray(0, n));
    views.iSyn.set(neurons.iSyn.subarray(0, n));
    views.iExt.set(neurons.iExt.subarray(0, n));
    views.bias.set(neurons.bias.subarray(0, n));
    views.noise.set(neurons.noise.subarray(0, n));
    views.spike.set(neurons.spike.subarray(0, n));
    views.lastSpike.set(neurons.lastSpike.subarray(0, n));
    views.refractoryUntil.set(neurons.refractoryUntil.subarray(0, n));
    views.flash.set(neurons.flash.subarray(0, n));
    views.rate.set(neurons.rate.subarray(0, n));
    views.spikeCount.set(neurons.spikeCount.subarray(0, n));
    views.model.set(neurons.model.subarray(0, n));
    views.polarity.set(neurons.polarity.subarray(0, n));
    views.neuronEnabled.set(neurons.enabled.subarray(0, n));
    views.neuronParams.set(neurons.params.subarray(0, n * NEURON_PARAM_STRIDE));
    views.scale.set(neurons.scale.subarray(0, n));
    views.seed.set(neurons.seed.subarray(0, n));
    views.archetype.set(neurons.archetype.subarray(0, n));
    views.population.set(neurons.population.subarray(0, n));
    views.flags.set(neurons.flags.subarray(0, n));

    views.pre.set(synapses.pre.subarray(0, m));
    views.post.set(synapses.post.subarray(0, m));
    views.weight.set(synapses.weight.subarray(0, m));
    views.delay.set(synapses.delay.subarray(0, m));
    views.gRise.set(synapses.gRise.subarray(0, m));
    views.gDecay.set(synapses.gDecay.subarray(0, m));
    views.tauRise.set(synapses.tauRise.subarray(0, m));
    views.tauDecay.set(synapses.tauDecay.subarray(0, m));
    views.eRev.set(synapses.eRev.subarray(0, m));
    views.mgBlock.set(synapses.mgBlock.subarray(0, m));
    views.preTrace.set(synapses.preTrace.subarray(0, m));
    views.postTrace.set(synapses.postTrace.subarray(0, m));
    views.preTraceSlow.set(synapses.preTraceSlow.subarray(0, m));
    views.postTraceSlow.set(synapses.postTraceSlow.subarray(0, m));
    views.stpR.set(synapses.stpR.subarray(0, m));
    views.stpU.set(synapses.stpU.subarray(0, m));
    views.releaseProb.set(synapses.releaseProb.subarray(0, m));
    views.receptor.set(synapses.receptor.subarray(0, m));
    views.plasticity.set(synapses.plasticity.subarray(0, m));
    views.synapseEnabled.set(synapses.enabled.subarray(0, m));
    views.synapseParams.set(synapses.params.subarray(0, m * SYNAPSE_PARAM_STRIDE));
    views.activity.set(synapses.activity.subarray(0, m));
    views.arc.set(synapses.arc.subarray(0, m));
  }

  /** Copy back only what the step actually changed. */
  private download(buffers: SimulationBuffers, views: Views, plasticity: boolean): void {
    const { neurons, synapses } = buffers;
    const n = neurons.count;
    const m = synapses.count;

    neurons.v.set(views.v.subarray(0, n));
    neurons.w.set(views.w.subarray(0, n));
    neurons.gateM.set(views.gateM.subarray(0, n));
    neurons.gateH.set(views.gateH.subarray(0, n));
    neurons.gateN.set(views.gateN.subarray(0, n));
    neurons.calcium.set(views.calcium.subarray(0, n));
    neurons.iSyn.set(views.iSyn.subarray(0, n));
    neurons.spike.set(views.spike.subarray(0, n));
    neurons.lastSpike.set(views.lastSpike.subarray(0, n));
    neurons.refractoryUntil.set(views.refractoryUntil.subarray(0, n));
    neurons.flash.set(views.flash.subarray(0, n));
    neurons.rate.set(views.rate.subarray(0, n));
    neurons.spikeCount.set(views.spikeCount.subarray(0, n));

    synapses.gRise.set(views.gRise.subarray(0, m));
    synapses.gDecay.set(views.gDecay.subarray(0, m));
    synapses.stpR.set(views.stpR.subarray(0, m));
    synapses.stpU.set(views.stpU.subarray(0, m));
    synapses.activity.set(views.activity.subarray(0, m));

    // Weights and traces only move while a rule is running; copying them back
    // unconditionally would overwrite an edit the crate never touched.
    if (plasticity) {
      synapses.weight.set(views.weight.subarray(0, m));
      synapses.preTrace.set(views.preTrace.subarray(0, m));
      synapses.postTrace.set(views.postTrace.subarray(0, m));
      synapses.preTraceSlow.set(views.preTraceSlow.subarray(0, m));
      synapses.postTraceSlow.set(views.postTraceSlow.subarray(0, m));
    }

    const head = this.network.spike_log_head() >>> 0;
    const emitted = (head - this.logHead) >>> 0;
    this.logHead = head;
    replaySpikeLog(
      buffers.spikes,
      emitted,
      views.logNeuron.length,
      head,
      (index) => views.logNeuron[index],
      (index) => views.logTime[index],
    );
  }
}

/* ------------------------------------------------------------- selection -- */

type BackendPreference = SimulationSettings['backend'];

/**
 * Backends to try, in order, for a preference.
 *
 * A preference is a ceiling, not a demand: asking for `gpu` on a machine
 * without one gets WASM and then the CPU, while asking for `cpu` stays there.
 */
function fallbackChain(preference: BackendPreference): readonly BackendPreference[] {
  switch (preference) {
    case 'cpu':
      return ['cpu'];
    case 'wasm':
      return ['wasm', 'cpu'];
    default:
      return ['gpu', 'wasm', 'cpu'];
  }
}

/**
 * Pick the best integrator for `preference`, falling back down the chain.
 *
 * `device` distinguishes three cases deliberately. A `GPUDevice` is used as
 * given, which is how the app shares the one device `createRenderer` acquired.
 * `null` means the caller already knows there is no device and GPU is skipped
 * without probing. Omitting it entirely lets this function acquire its own,
 * which is what makes `createIntegrator('gpu')` usable on its own — at the cost
 * of a second device if the app also has one, so callers that have a device
 * should pass it.
 */
export async function createIntegrator(
  preference: BackendPreference,
  device?: GPUDevice | null,
): Promise<Integrator> {
  for (const candidate of fallbackChain(preference)) {
    if (candidate === 'gpu') {
      if (device === null) continue;
      const target = device ?? (await requestComputeDevice());
      if (target === null) continue;
      const gpu = await GpuIntegrator.create(target);
      if (gpu !== null) return gpu;
      continue;
    }

    if (candidate === 'wasm') {
      if (!WasmIntegrator.isAvailable()) continue;
      const wasm = await WasmIntegrator.create();
      if (wasm !== null) return wasm;
      continue;
    }

    break;
  }

  return new CpuIntegrator();
}
