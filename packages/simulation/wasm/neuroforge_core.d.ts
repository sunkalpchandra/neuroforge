/* tslint:disable */
/* eslint-disable */

/**
 * The simulation. Owns every buffer; JavaScript views them in place.
 */
export class Network {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Resize the delay calendar; `buckets * resolution` is the longest
     * representable conduction delay. Discards anything queued.
     */
    configure_delays(buckets: number, resolution: number, stride: number): void;
    /**
     * Resize the spike-log ring. Discards its contents.
     */
    configure_spike_log(capacity: number): void;
    delay_amplitude_ptr(): number;
    delay_buckets(): number;
    /**
     * Events whose delay exceeded the horizon and were pulled back to it.
     */
    delay_clamped(): number;
    delay_counts_ptr(): number;
    /**
     * Events lost because their arrival bucket was full.
     */
    delay_dropped(): number;
    delay_entries_ptr(): number;
    /**
     * Longest conduction delay the queue can represent (ms).
     */
    delay_horizon(): number;
    delay_pending(): number;
    delay_resolution(): number;
    delay_stride(): number;
    /**
     * Copy up to `max` pending spike events to `out_ptr` and remove them from
     * the pending queue. Returns the number of events written.
     *
     * Each event occupies two 32-bit words: word 0 is the neuron slot as a
     * `u32`, word 1 is the spike time as an `f32`. Overlay a `Uint32Array` and a
     * `Float32Array` on the same region and read index `2*k` from the first and
     * `2*k+1` from the second. `out_ptr` must be 4-byte aligned and address at
     * least `2 * max` words of memory the caller owns — `reserve_scratch(2 * max)`
     * followed by `scratch_ptr()` is the intended source.
     */
    drain_spikes(out_ptr: number, max: number): number;
    graph_average_clustering(): number;
    /**
     * Local clustering coefficient per neuron slot, on the undirected
     * projection of the circuit.
     */
    graph_clustering(): Float32Array;
    graph_component_count(): number;
    /**
     * Weakly connected component id per neuron slot.
     */
    graph_components(): Uint32Array;
    graph_has_cycle(): boolean;
    graph_in_degrees(): Uint32Array;
    graph_out_degrees(): Uint32Array;
    /**
     * Slots that participate in a directed cycle, ascending.
     */
    graph_recurrent_nodes(): Uint32Array;
    /**
     * Breadth-first shortest path from `from` to `to`, inclusive of both
     * endpoints. Empty when no directed path exists.
     */
    graph_shortest_path(from: number, to: number): Uint32Array;
    /**
     * Strongly connected component id per neuron slot. Any id shared by two or
     * more neurons is a recurrent motif.
     */
    graph_strongly_connected_components(): Uint32Array;
    graph_strongly_connected_count(): number;
    neuron_archetype_ptr(): number;
    neuron_bias_ptr(): number;
    neuron_calcium_ptr(): number;
    neuron_capacity(): number;
    neuron_enabled_ptr(): number;
    neuron_flags_ptr(): number;
    neuron_flash_ptr(): number;
    neuron_gate_h_ptr(): number;
    neuron_gate_m_ptr(): number;
    neuron_gate_n_ptr(): number;
    neuron_i_ext_ptr(): number;
    neuron_i_syn_ptr(): number;
    neuron_last_spike_ptr(): number;
    neuron_len(): number;
    neuron_model_ptr(): number;
    neuron_noise_ptr(): number;
    neuron_params_ptr(): number;
    neuron_polarity_ptr(): number;
    neuron_population_ptr(): number;
    neuron_position_ptr(): number;
    neuron_rate_ptr(): number;
    neuron_refractory_until_ptr(): number;
    neuron_scale_ptr(): number;
    neuron_seed_ptr(): number;
    neuron_spike_count_ptr(): number;
    neuron_spike_ptr(): number;
    neuron_v_ptr(): number;
    neuron_w_ptr(): number;
    /**
     * Allocate a network with room for `neuron_capacity` neurons and
     * `synapse_capacity` synapses. Both live counts start at zero; call
     * `resize_neurons` / `resize_synapses` to make slots active.
     */
    constructor(neuron_capacity: number, synapse_capacity: number);
    pending_spikes(): number;
    /**
     * Reserve a scratch region inside wasm memory that JavaScript can hand to
     * `drain_spikes` as a destination. Returns true when the region moved, which
     * invalidates any view built on the previous `scratch_ptr()`.
     */
    reserve_scratch(words: number): boolean;
    /**
     * Return every state variable to rest without changing topology.
     *
     * Voltages go to the model's resting potential and the Hodgkin-Huxley gates
     * to their steady state there; conductances, traces, currents, the delay
     * calendar and the spike log are cleared. Parameters, positions, weights,
     * delays and connectivity are untouched.
     */
    reset(): void;
    /**
     * Set the live neuron count, growing capacity by doubling when needed.
     *
     * Returns true when the backing storage moved, in which case **every**
     * neuron pointer previously handed to JavaScript is stale and all views must
     * be rebuilt from fresh `*_ptr()` calls.
     */
    resize_neurons(count: number): boolean;
    /**
     * Set the live synapse count, growing capacity by doubling when needed.
     *
     * Returns true when the backing storage moved; see `resize_neurons`.
     */
    resize_synapses(count: number): boolean;
    scratch_len(): number;
    scratch_ptr(): number;
    seed(): number;
    /**
     * Reseed the deterministic generator. The same seed replayed over the same
     * circuit reproduces a run exactly.
     */
    set_seed(seed: number): void;
    /**
     * Move simulation time without touching state. Pending delay events keep
     * their absolute arrival buckets and the calendar cursor re-anchors.
     */
    set_time(t: number): void;
    /**
     * Total spikes emitted since the last reset.
     */
    spike_count(): number;
    spike_log_capacity(): number;
    /**
     * Total events written since the last reset; the live write cursor is
     * `spike_log_head() % spike_log_capacity()`.
     */
    spike_log_head(): number;
    spike_log_neuron_ptr(): number;
    spike_log_time_ptr(): number;
    /**
     * Advance the simulation by `steps` substeps of `dt` milliseconds.
     *
     * `integrator` is 0 Euler, 1 RK2, 2 RK4, 3 exponential Euler; any other
     * value falls back to exponential Euler. `gain` scales every synaptic
     * conductance, `noise` adds a per-neuron current with that standard
     * deviation on top of each neuron's own noise column, and `plasticity`
     * gates the weight-update pass. Returns the number of spikes emitted across
     * all substeps.
     */
    step(steps: number, dt: number, integrator: number, gain: number, noise: number, plasticity: boolean): number;
    /**
     * Substeps executed since the last reset.
     */
    step_index(): number;
    synapse_activity_ptr(): number;
    synapse_arc_ptr(): number;
    synapse_capacity(): number;
    synapse_delay_ptr(): number;
    synapse_e_rev_ptr(): number;
    synapse_enabled_ptr(): number;
    synapse_g_decay_ptr(): number;
    synapse_g_rise_ptr(): number;
    synapse_len(): number;
    synapse_mg_block_ptr(): number;
    synapse_params_ptr(): number;
    synapse_plasticity_ptr(): number;
    synapse_post_ptr(): number;
    synapse_post_trace_ptr(): number;
    synapse_post_trace_slow_ptr(): number;
    synapse_pre_ptr(): number;
    synapse_pre_trace_ptr(): number;
    synapse_pre_trace_slow_ptr(): number;
    synapse_receptor_ptr(): number;
    synapse_release_prob_ptr(): number;
    synapse_stp_r_ptr(): number;
    synapse_stp_u_ptr(): number;
    synapse_tau_decay_ptr(): number;
    synapse_tau_rise_ptr(): number;
    synapse_weight_ptr(): number;
    time(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_network_free: (a: number, b: number) => void;
    readonly network_configure_delays: (a: number, b: number, c: number, d: number) => void;
    readonly network_configure_spike_log: (a: number, b: number) => void;
    readonly network_delay_amplitude_ptr: (a: number) => number;
    readonly network_delay_buckets: (a: number) => number;
    readonly network_delay_clamped: (a: number) => number;
    readonly network_delay_counts_ptr: (a: number) => number;
    readonly network_delay_dropped: (a: number) => number;
    readonly network_delay_entries_ptr: (a: number) => number;
    readonly network_delay_horizon: (a: number) => number;
    readonly network_delay_pending: (a: number) => number;
    readonly network_delay_resolution: (a: number) => number;
    readonly network_delay_stride: (a: number) => number;
    readonly network_drain_spikes: (a: number, b: number, c: number) => number;
    readonly network_graph_average_clustering: (a: number) => number;
    readonly network_graph_clustering: (a: number) => [number, number];
    readonly network_graph_component_count: (a: number) => number;
    readonly network_graph_components: (a: number) => [number, number];
    readonly network_graph_has_cycle: (a: number) => number;
    readonly network_graph_in_degrees: (a: number) => [number, number];
    readonly network_graph_out_degrees: (a: number) => [number, number];
    readonly network_graph_recurrent_nodes: (a: number) => [number, number];
    readonly network_graph_shortest_path: (a: number, b: number, c: number) => [number, number];
    readonly network_graph_strongly_connected_components: (a: number) => [number, number];
    readonly network_graph_strongly_connected_count: (a: number) => number;
    readonly network_neuron_archetype_ptr: (a: number) => number;
    readonly network_neuron_bias_ptr: (a: number) => number;
    readonly network_neuron_calcium_ptr: (a: number) => number;
    readonly network_neuron_capacity: (a: number) => number;
    readonly network_neuron_enabled_ptr: (a: number) => number;
    readonly network_neuron_flags_ptr: (a: number) => number;
    readonly network_neuron_flash_ptr: (a: number) => number;
    readonly network_neuron_gate_h_ptr: (a: number) => number;
    readonly network_neuron_gate_m_ptr: (a: number) => number;
    readonly network_neuron_gate_n_ptr: (a: number) => number;
    readonly network_neuron_i_ext_ptr: (a: number) => number;
    readonly network_neuron_i_syn_ptr: (a: number) => number;
    readonly network_neuron_last_spike_ptr: (a: number) => number;
    readonly network_neuron_len: (a: number) => number;
    readonly network_neuron_model_ptr: (a: number) => number;
    readonly network_neuron_noise_ptr: (a: number) => number;
    readonly network_neuron_params_ptr: (a: number) => number;
    readonly network_neuron_polarity_ptr: (a: number) => number;
    readonly network_neuron_population_ptr: (a: number) => number;
    readonly network_neuron_position_ptr: (a: number) => number;
    readonly network_neuron_rate_ptr: (a: number) => number;
    readonly network_neuron_refractory_until_ptr: (a: number) => number;
    readonly network_neuron_scale_ptr: (a: number) => number;
    readonly network_neuron_seed_ptr: (a: number) => number;
    readonly network_neuron_spike_count_ptr: (a: number) => number;
    readonly network_neuron_spike_ptr: (a: number) => number;
    readonly network_neuron_v_ptr: (a: number) => number;
    readonly network_neuron_w_ptr: (a: number) => number;
    readonly network_new: (a: number, b: number) => number;
    readonly network_pending_spikes: (a: number) => number;
    readonly network_reserve_scratch: (a: number, b: number) => number;
    readonly network_reset: (a: number) => void;
    readonly network_resize_neurons: (a: number, b: number) => number;
    readonly network_resize_synapses: (a: number, b: number) => number;
    readonly network_scratch_len: (a: number) => number;
    readonly network_scratch_ptr: (a: number) => number;
    readonly network_seed: (a: number) => number;
    readonly network_set_seed: (a: number, b: number) => void;
    readonly network_set_time: (a: number, b: number) => void;
    readonly network_spike_count: (a: number) => number;
    readonly network_spike_log_capacity: (a: number) => number;
    readonly network_spike_log_head: (a: number) => number;
    readonly network_spike_log_neuron_ptr: (a: number) => number;
    readonly network_spike_log_time_ptr: (a: number) => number;
    readonly network_step: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly network_step_index: (a: number) => number;
    readonly network_synapse_activity_ptr: (a: number) => number;
    readonly network_synapse_arc_ptr: (a: number) => number;
    readonly network_synapse_capacity: (a: number) => number;
    readonly network_synapse_delay_ptr: (a: number) => number;
    readonly network_synapse_e_rev_ptr: (a: number) => number;
    readonly network_synapse_enabled_ptr: (a: number) => number;
    readonly network_synapse_g_decay_ptr: (a: number) => number;
    readonly network_synapse_g_rise_ptr: (a: number) => number;
    readonly network_synapse_len: (a: number) => number;
    readonly network_synapse_mg_block_ptr: (a: number) => number;
    readonly network_synapse_params_ptr: (a: number) => number;
    readonly network_synapse_plasticity_ptr: (a: number) => number;
    readonly network_synapse_post_ptr: (a: number) => number;
    readonly network_synapse_post_trace_ptr: (a: number) => number;
    readonly network_synapse_post_trace_slow_ptr: (a: number) => number;
    readonly network_synapse_pre_ptr: (a: number) => number;
    readonly network_synapse_pre_trace_ptr: (a: number) => number;
    readonly network_synapse_pre_trace_slow_ptr: (a: number) => number;
    readonly network_synapse_receptor_ptr: (a: number) => number;
    readonly network_synapse_release_prob_ptr: (a: number) => number;
    readonly network_synapse_stp_r_ptr: (a: number) => number;
    readonly network_synapse_stp_u_ptr: (a: number) => number;
    readonly network_synapse_tau_decay_ptr: (a: number) => number;
    readonly network_synapse_tau_rise_ptr: (a: number) => number;
    readonly network_synapse_weight_ptr: (a: number) => number;
    readonly network_time: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
