import {
  allocateSimulationBuffers,
  DEFAULT_SIMULATION_SETTINGS,
  MODEL_CODE,
  PLASTICITY_CODE,
  RECEPTOR_CODE,
  EMPTY_FRAME_STATS,
  ema,
  growNeuronBuffers,
  growSynapseBuffers,
  packNeuronParams,
  packSynapseParams,
} from '@neuroforge/shared';
import type {
  Circuit,
  FrameStats,
  SimulationBuffers,
  SimulationSettings,
} from '@neuroforge/shared';
import { Rng } from '@neuroforge/math';

import { createIntegrator } from './backend';
import { CpuIntegrator } from './integrator-cpu';
import { applyStimuli } from './stimuli';
import { EMPTY_STEP_RESULT } from './types';
import type { Integrator, StepResult } from './types';

/**
 * Owns the simulation buffers, the active integrator and the real-time clock.
 *
 * The buffers are the single copy of network state in the application. The
 * renderer reads them directly, the editor mutates the document and calls
 * `load` to rebuild them, and the integrator advances them in place. Nothing
 * else holds a duplicate, which is what keeps a hundred thousand neurons
 * affordable.
 */
export class SimulationEngine {
  private _buffers: SimulationBuffers;
  private _settings: SimulationSettings;
  private _running = false;
  private _stats: FrameStats = { ...EMPTY_FRAME_STATS };

  private integrator: Integrator;
  private rng: Rng;

  /** Document ids in slot order, and the inverse lookup. */
  private slotToId: string[] = [];
  private idToSlot = new Map<string, number>();

  /** Stimuli captured from the last loaded document. */
  private stimuli: Circuit['stimuli'] = [];

  /** Unconsumed simulated milliseconds carried between frames. */
  private accumulator = 0;

  /**
   * Device offered to GPU backends. `undefined` means none has been offered and
   * `createIntegrator` may acquire its own; `null` means the app has told us
   * there is no device, and GPU is skipped without probing for one.
   */
  private device: GPUDevice | null | undefined = undefined;

  /** Guards against an earlier backend swap resolving after a later one. */
  private backendGeneration = 0;

  private disposed = false;

  /**
   * Charge injected by `poke` since the last step, in pA.
   *
   * It cannot live in `iExt` alone: `applyStimuli` clears that column at the top
   * of every step, so a poke written straight into it is erased before any
   * integrator sees it. Held here, it is re-applied after the clear and consumed
   * by exactly one step, which is what makes it a pulse.
   */
  private pokeCharge = new Float32Array(0);
  private pokePending = false;

  /** Stimulus slot lookup, bound once so the step path allocates no closure. */
  private readonly lookupSlot = (id: string): number => this.slotOf(id);

  /** Wall-clock milliseconds of simulated time in the last second, for the FPS readout. */
  private lastFrameAt = 0;

  constructor(settings: Partial<SimulationSettings> = {}) {
    this._settings = { ...DEFAULT_SIMULATION_SETTINGS, ...settings };
    this._buffers = allocateSimulationBuffers(1024, 4096);
    this.rng = new Rng(this._settings.seed);
    this.integrator = new CpuIntegrator(this._settings.seed);
  }

  get buffers(): SimulationBuffers {
    return this._buffers;
  }

  get settings(): SimulationSettings {
    return this._settings;
  }

  get running(): boolean {
    return this._running;
  }

  get stats(): FrameStats {
    return this._stats;
  }

  setSettings(patch: Partial<SimulationSettings>): void {
    const seedChanged = patch.seed !== undefined && patch.seed !== this._settings.seed;
    this._settings = { ...this._settings, ...patch };
    if (seedChanged) this.rng = new Rng(this._settings.seed);
  }

  slotOf(id: string): number {
    const slot = this.idToSlot.get(id);
    return slot === undefined ? -1 : slot;
  }

  idOf(slot: number): string | null {
    return slot >= 0 && slot < this.slotToId.length ? this.slotToId[slot] : null;
  }

  /**
   * Rebuild the buffers from a document.
   *
   * Every typed array may be replaced, so any reference a caller cached across
   * this call is stale afterwards. The renderer's rebuild path exists precisely
   * to re-acquire them.
   */
  load(circuit: Circuit): void {
    const neuronCount = circuit.neurons.length;
    const synapseCount = circuit.synapses.length;

    growNeuronBuffers(this._buffers.neurons, Math.max(1, neuronCount));
    growSynapseBuffers(this._buffers.synapses, Math.max(1, synapseCount));

    const { neurons, synapses } = this._buffers;
    neurons.count = neuronCount;
    synapses.count = synapseCount;

    this.slotToId = new Array<string>(neuronCount);
    this.idToSlot = new Map<string, number>();

    for (let i = 0; i < neuronCount; i += 1) {
      const n = circuit.neurons[i];
      this.slotToId[i] = n.id;
      this.idToSlot.set(n.id, i);

      const p = i * 3;
      neurons.position[p] = n.position.x;
      neurons.position[p + 1] = n.position.y;
      neurons.position[p + 2] = n.position.z;

      neurons.model[i] = MODEL_CODE[n.params.kind];
      neurons.polarity[i] = n.polarity === 'inhibitory' ? 1 : 0;
      neurons.enabled[i] = n.enabled ? 1 : 0;
      neurons.bias[i] = n.bias;
      neurons.noise[i] = n.noise;
      neurons.seed[i] = n.morphology.seed >>> 0;
      neurons.scale[i] = n.morphology.scale;
      neurons.flags[i] = 0;
      packNeuronParams(n.params, neurons.params, i);
    }

    // Population index is stored as a compact slot rather than an id so the
    // renderer can colour by population without a string lookup per instance.
    const populationSlot = new Map<string, number>();
    circuit.populations.forEach((pop, index) => populationSlot.set(pop.id, index));
    for (let i = 0; i < neuronCount; i += 1) {
      const owner = circuit.neurons[i].population;
      neurons.population[i] = owner === null ? 0xffff : (populationSlot.get(owner) ?? 0xffff);
    }

    const archetypeIndex = new Map<string, number>();
    for (let i = 0; i < neuronCount; i += 1) {
      const archetype = circuit.neurons[i].morphology.archetype;
      let index = archetypeIndex.get(archetype);
      if (index === undefined) {
        index = archetypeIndex.size;
        archetypeIndex.set(archetype, index);
      }
      neurons.archetype[i] = index;
    }

    let kept = 0;
    for (let i = 0; i < synapseCount; i += 1) {
      const s = circuit.synapses[i];
      const pre = this.idToSlot.get(s.source);
      const post = this.idToSlot.get(s.target);
      // A synapse referencing a deleted neuron is dropped rather than pointed at
      // slot 0, which would silently rewire the network.
      if (pre === undefined || post === undefined) continue;

      synapses.pre[kept] = pre;
      synapses.post[kept] = post;
      synapses.weight[kept] = s.weight;
      synapses.delay[kept] = s.delay;
      synapses.tauRise[kept] = s.kinetics.tauRise;
      synapses.tauDecay[kept] = s.kinetics.tauDecay;
      synapses.eRev[kept] = s.kinetics.eRev;
      synapses.mgBlock[kept] = s.kinetics.mgBlock;
      synapses.receptor[kept] = RECEPTOR_CODE[s.receptor];
      synapses.plasticity[kept] = PLASTICITY_CODE[s.plasticity.kind];
      synapses.releaseProb[kept] = s.releaseProbability;
      synapses.arc[kept] = s.arc;
      synapses.enabled[kept] = s.enabled ? 1 : 0;
      synapses.gRise[kept] = 0;
      synapses.gDecay[kept] = 0;
      synapses.activity[kept] = 0;
      packSynapseParams(s.plasticity, s.stp, synapses.params, kept);
      kept += 1;
    }
    synapses.count = kept;

    this.stimuli = circuit.stimuli;
    this._settings = { ...this._settings, ...circuit.simulation };
    this.rng = new Rng(this._settings.seed);

    this.integrator.invalidate?.();
    this.integrator.reset(this._buffers);
    this.accumulator = 0;
    this.clearPokes();
    this.refreshCounts();
  }

  private refreshCounts(): void {
    this._stats = {
      ...this._stats,
      neurons: this._buffers.neurons.count,
      synapses: this._buffers.synapses.count,
      backend: this.integrator.backend,
    };
  }

  play(): void {
    this._running = true;
    this.lastFrameAt = 0;
  }

  pause(): void {
    this._running = false;
  }

  reset(): void {
    this.integrator.reset(this._buffers);
    this.accumulator = 0;
    this.clearPokes();
    this.rng = new Rng(this._settings.seed);
    this._stats = { ...this._stats, simTime: 0, spikes: 0, meanRate: 0 };
  }

  private clearPokes(): void {
    this.pokeCharge.fill(0);
    this.pokePending = false;
  }

  /** Structural edit happened; drop any cached derivation of the topology. */
  invalidate(): void {
    this.integrator.invalidate?.();
    this.refreshCounts();
  }

  /**
   * Advance in real time. `dtSeconds` is the wall-clock frame delta.
   *
   * The number of substeps is bounded by `maxSubstepsPerFrame`, which is what
   * stops a slow frame from triggering a longer frame and spiralling. When the
   * bound is hit the surplus is discarded rather than carried, so the simulation
   * runs slower than real time instead of freezing the tab trying to catch up.
   */
  advance(dtSeconds: number): StepResult {
    if (!this._running || this._buffers.neurons.count === 0) return EMPTY_STEP_RESULT;

    const { dt, speed, maxSubstepsPerFrame } = this._settings;
    this.accumulator += dtSeconds * 1000 * speed;

    let steps = Math.floor(this.accumulator / dt);
    if (steps <= 0) return EMPTY_STEP_RESULT;

    if (steps > maxSubstepsPerFrame) {
      steps = maxSubstepsPerFrame;
      this.accumulator = 0;
    } else {
      this.accumulator -= steps * dt;
    }

    return this.runSteps(steps, dtSeconds);
  }

  /** One deterministic substep, independent of the clock. */
  stepOnce(): StepResult {
    if (this._buffers.neurons.count === 0) return EMPTY_STEP_RESULT;
    return this.runSteps(1, this._settings.dt / 1000);
  }

  private runSteps(steps: number, dtSeconds: number): StepResult {
    applyStimuli(
      this._buffers,
      this.stimuli,
      this.lookupSlot,
      this._buffers.time,
      this.rng,
      this._settings.dt,
    );
    this.drainPokes();

    const result = this.integrator.step(this._buffers, this._settings, steps);
    this._buffers.step += result.steps;

    const simulatedMs = result.steps * this._settings.dt;
    const wallMs = dtSeconds * 1000;

    let rateSum = 0;
    const count = this._buffers.neurons.count;
    for (let i = 0; i < count; i += 1) rateSum += this._buffers.neurons.rate[i];

    this._stats = {
      ...this._stats,
      simMs: ema(this._stats.simMs, result.simMs, 0.15),
      substeps: result.steps,
      spikes: result.spikes,
      simTime: this._buffers.time,
      meanRate: count > 0 ? rateSum / count : 0,
      realtimeFactor: wallMs > 0 ? simulatedMs / wallMs : 0,
      neurons: count,
      synapses: this._buffers.synapses.count,
      backend: this.integrator.backend,
    };

    return result;
  }

  /** Record a rendered frame. Kept separate from stepping so the FPS readout
   * stays honest while the simulation is paused. */
  recordFrame(frameMs: number): void {
    this._stats = {
      ...this._stats,
      frameMs: ema(this._stats.frameMs, frameMs, 0.12),
      fps: frameMs > 0 ? ema(this._stats.fps, 1000 / frameMs, 0.12) : this._stats.fps,
    };
  }

  /** Inject a brief current into one neuron, used by the poke tool. */
  poke(slot: number, amplitude: number): void {
    const neurons = this._buffers.neurons;
    if (slot < 0 || slot >= neurons.count) return;
    if (this.pokeCharge.length < neurons.capacity) {
      const next = new Float32Array(neurons.capacity);
      next.set(this.pokeCharge);
      this.pokeCharge = next;
    }
    this.pokeCharge[slot] += amplitude;
    this.pokePending = true;
    // Also written straight through so a poke while paused is visible to the
    // inspector immediately, rather than only once the clock is running.
    neurons.iExt[slot] += amplitude;
  }

  /** Fold pending pokes into `iExt` after `applyStimuli` has cleared it. */
  private drainPokes(): void {
    if (!this.pokePending) return;
    const { iExt } = this._buffers.neurons;
    const count = Math.min(this._buffers.neurons.count, this.pokeCharge.length);
    for (let i = 0; i < count; i += 1) {
      const charge = this.pokeCharge[i];
      if (charge === 0) continue;
      iExt[i] += charge;
      this.pokeCharge[i] = 0;
    }
    this.pokeCharge.fill(0, count);
    this.pokePending = false;
  }

  /**
   * Swap the integrator. The existing buffers are handed to the new backend
   * untouched, so switching backends mid-run continues from the current state
   * rather than restarting.
   */
  async setIntegrator(next: Integrator): Promise<void> {
    if (next === this.integrator) return;
    this.integrator.dispose();
    this.integrator = next;
    this.integrator.invalidate?.();
    this.refreshCounts();
  }

  /**
   * Choose a compute backend, falling back down the chain when the preference is
   * unavailable. The preference is recorded in the settings either way, so the
   * document keeps asking for the backend the user wanted even on a machine that
   * cannot provide it.
   */
  async setBackend(preference: SimulationSettings['backend']): Promise<void> {
    this._settings = { ...this._settings, backend: preference };
    await this.adoptBackend(preference);
  }

  /**
   * Hand the renderer's `GPUDevice` to the engine, or `null` to say there is
   * none. Sharing the device the app already owns is what stops the GPU backend
   * from acquiring a second one; passing null keeps it from probing at all.
   */
  async attachDevice(device: GPUDevice | null): Promise<void> {
    if (device === this.device) return;
    this.device = device;
    await this.adoptBackend(this._settings.backend);
  }

  /**
   * Resolve a preference into an integrator and install it.
   *
   * Backend selection is asynchronous, so two overlapping requests can resolve
   * out of order; the generation stamp drops every result but the newest and
   * disposes the integrator it was about to install rather than leaking it.
   */
  private async adoptBackend(preference: SimulationSettings['backend']): Promise<void> {
    const generation = (this.backendGeneration += 1);
    const next = await createIntegrator(preference, this.device);
    if (generation !== this.backendGeneration || this.disposed) {
      next.dispose();
      return;
    }
    await this.setIntegrator(next);
  }

  dispose(): void {
    this.disposed = true;
    this.integrator.dispose();
    this._running = false;
  }
}
