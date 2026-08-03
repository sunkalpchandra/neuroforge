/* @ts-self-types="./neuroforge_core.d.ts" */

/**
 * The simulation. Owns every buffer; JavaScript views them in place.
 */
export class Network {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        NetworkFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_network_free(ptr, 0);
    }
    /**
     * Resize the delay calendar; `buckets * resolution` is the longest
     * representable conduction delay. Discards anything queued.
     * @param {number} buckets
     * @param {number} resolution
     * @param {number} stride
     */
    configure_delays(buckets, resolution, stride) {
        wasm.network_configure_delays(this.__wbg_ptr, buckets, resolution, stride);
    }
    /**
     * Resize the spike-log ring. Discards its contents.
     * @param {number} capacity
     */
    configure_spike_log(capacity) {
        wasm.network_configure_spike_log(this.__wbg_ptr, capacity);
    }
    /**
     * @returns {number}
     */
    delay_amplitude_ptr() {
        const ret = wasm.network_delay_amplitude_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    delay_buckets() {
        const ret = wasm.network_delay_buckets(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Events whose delay exceeded the horizon and were pulled back to it.
     * @returns {number}
     */
    delay_clamped() {
        const ret = wasm.network_delay_clamped(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    delay_counts_ptr() {
        const ret = wasm.network_delay_counts_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Events lost because their arrival bucket was full.
     * @returns {number}
     */
    delay_dropped() {
        const ret = wasm.network_delay_dropped(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    delay_entries_ptr() {
        const ret = wasm.network_delay_entries_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Longest conduction delay the queue can represent (ms).
     * @returns {number}
     */
    delay_horizon() {
        const ret = wasm.network_delay_horizon(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    delay_pending() {
        const ret = wasm.network_delay_pending(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    delay_resolution() {
        const ret = wasm.network_delay_resolution(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    delay_stride() {
        const ret = wasm.network_delay_stride(this.__wbg_ptr);
        return ret >>> 0;
    }
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
     * @param {number} out_ptr
     * @param {number} max
     * @returns {number}
     */
    drain_spikes(out_ptr, max) {
        const ret = wasm.network_drain_spikes(this.__wbg_ptr, out_ptr, max);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    graph_average_clustering() {
        const ret = wasm.network_graph_average_clustering(this.__wbg_ptr);
        return ret;
    }
    /**
     * Local clustering coefficient per neuron slot, on the undirected
     * projection of the circuit.
     * @returns {Float32Array}
     */
    graph_clustering() {
        const ret = wasm.network_graph_clustering(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    graph_component_count() {
        const ret = wasm.network_graph_component_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Weakly connected component id per neuron slot.
     * @returns {Uint32Array}
     */
    graph_components() {
        const ret = wasm.network_graph_components(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {boolean}
     */
    graph_has_cycle() {
        const ret = wasm.network_graph_has_cycle(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {Uint32Array}
     */
    graph_in_degrees() {
        const ret = wasm.network_graph_in_degrees(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Uint32Array}
     */
    graph_out_degrees() {
        const ret = wasm.network_graph_out_degrees(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Slots that participate in a directed cycle, ascending.
     * @returns {Uint32Array}
     */
    graph_recurrent_nodes() {
        const ret = wasm.network_graph_recurrent_nodes(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Breadth-first shortest path from `from` to `to`, inclusive of both
     * endpoints. Empty when no directed path exists.
     * @param {number} from
     * @param {number} to
     * @returns {Uint32Array}
     */
    graph_shortest_path(from, to) {
        const ret = wasm.network_graph_shortest_path(this.__wbg_ptr, from, to);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Strongly connected component id per neuron slot. Any id shared by two or
     * more neurons is a recurrent motif.
     * @returns {Uint32Array}
     */
    graph_strongly_connected_components() {
        const ret = wasm.network_graph_strongly_connected_components(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    graph_strongly_connected_count() {
        const ret = wasm.network_graph_strongly_connected_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_archetype_ptr() {
        const ret = wasm.network_neuron_archetype_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_bias_ptr() {
        const ret = wasm.network_neuron_bias_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_calcium_ptr() {
        const ret = wasm.network_neuron_calcium_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_capacity() {
        const ret = wasm.network_neuron_capacity(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_enabled_ptr() {
        const ret = wasm.network_neuron_enabled_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_flags_ptr() {
        const ret = wasm.network_neuron_flags_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_flash_ptr() {
        const ret = wasm.network_neuron_flash_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_gate_h_ptr() {
        const ret = wasm.network_neuron_gate_h_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_gate_m_ptr() {
        const ret = wasm.network_neuron_gate_m_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_gate_n_ptr() {
        const ret = wasm.network_neuron_gate_n_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_i_ext_ptr() {
        const ret = wasm.network_neuron_i_ext_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_i_syn_ptr() {
        const ret = wasm.network_neuron_i_syn_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_last_spike_ptr() {
        const ret = wasm.network_neuron_last_spike_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_len() {
        const ret = wasm.network_neuron_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_model_ptr() {
        const ret = wasm.network_neuron_model_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_noise_ptr() {
        const ret = wasm.network_neuron_noise_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_params_ptr() {
        const ret = wasm.network_neuron_params_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_polarity_ptr() {
        const ret = wasm.network_neuron_polarity_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_population_ptr() {
        const ret = wasm.network_neuron_population_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_position_ptr() {
        const ret = wasm.network_neuron_position_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_rate_ptr() {
        const ret = wasm.network_neuron_rate_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_refractory_until_ptr() {
        const ret = wasm.network_neuron_refractory_until_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_scale_ptr() {
        const ret = wasm.network_neuron_scale_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_seed_ptr() {
        const ret = wasm.network_neuron_seed_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_spike_count_ptr() {
        const ret = wasm.network_neuron_spike_count_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_spike_ptr() {
        const ret = wasm.network_neuron_spike_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_v_ptr() {
        const ret = wasm.network_neuron_v_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    neuron_w_ptr() {
        const ret = wasm.network_neuron_w_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Allocate a network with room for `neuron_capacity` neurons and
     * `synapse_capacity` synapses. Both live counts start at zero; call
     * `resize_neurons` / `resize_synapses` to make slots active.
     * @param {number} neuron_capacity
     * @param {number} synapse_capacity
     */
    constructor(neuron_capacity, synapse_capacity) {
        const ret = wasm.network_new(neuron_capacity, synapse_capacity);
        this.__wbg_ptr = ret;
        NetworkFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    pending_spikes() {
        const ret = wasm.network_pending_spikes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reserve a scratch region inside wasm memory that JavaScript can hand to
     * `drain_spikes` as a destination. Returns true when the region moved, which
     * invalidates any view built on the previous `scratch_ptr()`.
     * @param {number} words
     * @returns {boolean}
     */
    reserve_scratch(words) {
        const ret = wasm.network_reserve_scratch(this.__wbg_ptr, words);
        return ret !== 0;
    }
    /**
     * Return every state variable to rest without changing topology.
     *
     * Voltages go to the model's resting potential and the Hodgkin-Huxley gates
     * to their steady state there; conductances, traces, currents, the delay
     * calendar and the spike log are cleared. Parameters, positions, weights,
     * delays and connectivity are untouched.
     */
    reset() {
        wasm.network_reset(this.__wbg_ptr);
    }
    /**
     * Set the live neuron count, growing capacity by doubling when needed.
     *
     * Returns true when the backing storage moved, in which case **every**
     * neuron pointer previously handed to JavaScript is stale and all views must
     * be rebuilt from fresh `*_ptr()` calls.
     * @param {number} count
     * @returns {boolean}
     */
    resize_neurons(count) {
        const ret = wasm.network_resize_neurons(this.__wbg_ptr, count);
        return ret !== 0;
    }
    /**
     * Set the live synapse count, growing capacity by doubling when needed.
     *
     * Returns true when the backing storage moved; see `resize_neurons`.
     * @param {number} count
     * @returns {boolean}
     */
    resize_synapses(count) {
        const ret = wasm.network_resize_synapses(this.__wbg_ptr, count);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    scratch_len() {
        const ret = wasm.network_scratch_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    scratch_ptr() {
        const ret = wasm.network_scratch_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    seed() {
        const ret = wasm.network_seed(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reseed the deterministic generator. The same seed replayed over the same
     * circuit reproduces a run exactly.
     * @param {number} seed
     */
    set_seed(seed) {
        wasm.network_set_seed(this.__wbg_ptr, seed);
    }
    /**
     * Move simulation time without touching state. Pending delay events keep
     * their absolute arrival buckets and the calendar cursor re-anchors.
     * @param {number} t
     */
    set_time(t) {
        wasm.network_set_time(this.__wbg_ptr, t);
    }
    /**
     * Total spikes emitted since the last reset.
     * @returns {number}
     */
    spike_count() {
        const ret = wasm.network_spike_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    spike_log_capacity() {
        const ret = wasm.network_spike_log_capacity(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Total events written since the last reset; the live write cursor is
     * `spike_log_head() % spike_log_capacity()`.
     * @returns {number}
     */
    spike_log_head() {
        const ret = wasm.network_spike_log_head(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    spike_log_neuron_ptr() {
        const ret = wasm.network_spike_log_neuron_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    spike_log_time_ptr() {
        const ret = wasm.network_spike_log_time_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Advance the simulation by `steps` substeps of `dt` milliseconds.
     *
     * `integrator` is 0 Euler, 1 RK2, 2 RK4, 3 exponential Euler; any other
     * value falls back to exponential Euler. `gain` scales every synaptic
     * conductance, `noise` adds a per-neuron current with that standard
     * deviation on top of each neuron's own noise column, and `plasticity`
     * gates the weight-update pass. Returns the number of spikes emitted across
     * all substeps.
     * @param {number} steps
     * @param {number} dt
     * @param {number} integrator
     * @param {number} gain
     * @param {number} noise
     * @param {boolean} plasticity
     * @returns {number}
     */
    step(steps, dt, integrator, gain, noise, plasticity) {
        const ret = wasm.network_step(this.__wbg_ptr, steps, dt, integrator, gain, noise, plasticity);
        return ret >>> 0;
    }
    /**
     * Substeps executed since the last reset.
     * @returns {number}
     */
    step_index() {
        const ret = wasm.network_step_index(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_activity_ptr() {
        const ret = wasm.network_synapse_activity_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_arc_ptr() {
        const ret = wasm.network_synapse_arc_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_capacity() {
        const ret = wasm.network_synapse_capacity(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_delay_ptr() {
        const ret = wasm.network_synapse_delay_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_e_rev_ptr() {
        const ret = wasm.network_synapse_e_rev_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_enabled_ptr() {
        const ret = wasm.network_synapse_enabled_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_g_decay_ptr() {
        const ret = wasm.network_synapse_g_decay_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_g_rise_ptr() {
        const ret = wasm.network_synapse_g_rise_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_len() {
        const ret = wasm.network_synapse_len(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_mg_block_ptr() {
        const ret = wasm.network_synapse_mg_block_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_params_ptr() {
        const ret = wasm.network_synapse_params_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_plasticity_ptr() {
        const ret = wasm.network_synapse_plasticity_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_post_ptr() {
        const ret = wasm.network_synapse_post_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_post_trace_ptr() {
        const ret = wasm.network_synapse_post_trace_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_post_trace_slow_ptr() {
        const ret = wasm.network_synapse_post_trace_slow_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_pre_ptr() {
        const ret = wasm.network_synapse_pre_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_pre_trace_ptr() {
        const ret = wasm.network_synapse_pre_trace_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_pre_trace_slow_ptr() {
        const ret = wasm.network_synapse_pre_trace_slow_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_receptor_ptr() {
        const ret = wasm.network_synapse_receptor_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_release_prob_ptr() {
        const ret = wasm.network_synapse_release_prob_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_stp_r_ptr() {
        const ret = wasm.network_synapse_stp_r_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_stp_u_ptr() {
        const ret = wasm.network_synapse_stp_u_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_tau_decay_ptr() {
        const ret = wasm.network_synapse_tau_decay_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_tau_rise_ptr() {
        const ret = wasm.network_synapse_tau_rise_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    synapse_weight_ptr() {
        const ret = wasm.network_synapse_weight_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    time() {
        const ret = wasm.network_time(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Network.prototype[Symbol.dispose] = Network.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./neuroforge_core_bg.js": import0,
    };
}

const NetworkFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_network_free(ptr, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('neuroforge_core_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
