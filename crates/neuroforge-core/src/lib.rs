//! NeuroForge native simulation core.
//!
//! # Memory model
//!
//! Rust owns every simulation array. JavaScript does not allocate buffers and
//! hand them over; it asks this crate for a pointer per column and builds typed
//! array views straight onto wasm linear memory. Nothing is copied per frame in
//! either direction.
//!
//! The column set, the strides and the numeric codes mirror
//! `packages/shared/src/buffers.ts` exactly — `NeuronBuffers`, `SynapseBuffers`,
//! `DelayQueue` and `SpikeLog` each have a one-to-one counterpart here. See the
//! LAYOUT CONTRACT notes in `model.rs`, `synapse.rs` and `delay.rs` for the slot
//! offsets.
//!
//! # Views are invalidated by growth
//!
//! **Any call to `resize_neurons` or `resize_synapses` may reallocate, and wasm
//! linear memory may itself grow, which detaches every `ArrayBuffer` view
//! JavaScript is holding.** Both functions return `true` when a reallocation
//! happened. Treat `true` as "every previously acquired pointer is stale": drop
//! all views and re-acquire every `*_ptr()` before touching the buffers again.
//! Reading through a detached view throws; reading through a stale pointer is
//! worse, because it silently reads freed memory. The same caution applies after
//! `configure_delays`, `configure_spike_log` and `reserve_scratch`.
//!
//! # Units
//!
//! Voltage mV, time ms, current pA, capacitance pF, conductance nS, matching the
//! unit system declared in `CONTRACTS.md`.

mod delay;
mod graph;
pub mod model;
pub mod plasticity;
pub mod rng;
pub mod synapse;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use wasm_bindgen::prelude::*;

use delay::DelayQueue;
use model::{NEURON_PARAM_STRIDE, Params, State};
use plasticity::{PLASTICITY_STATIC, Traces};
use rng::Pcg32;
use synapse::{Kinetics, SYNAPSE_PARAM_STRIDE, StpState, SynParams};

/// Time constant of the exponentially-smoothed firing-rate column (ms).
const RATE_TAU_MS: f64 = 100.0;
/// Decay time constant of the display calcium column (ms).
const CALCIUM_TAU_MS: f64 = 50.0;
/// Calcium added per spike, in the arbitrary units the column documents.
const CALCIUM_PER_SPIKE: f32 = 1.0;
const DEFAULT_SPIKE_LOG_CAPACITY: usize = 65536;
const NO_POPULATION: u16 = 0xffff;

/// Borrow a fixed-size parameter block out of a packed column without any
/// possibility of panicking on a malformed index.
#[inline]
fn block<const N: usize>(data: &[f32], index: usize) -> Option<&[f32; N]> {
    let base = index.checked_mul(N)?;
    data.get(base..base.checked_add(N)?)?.try_into().ok()
}

struct Neurons {
    capacity: usize,
    count: usize,
    position: Vec<f32>,
    v: Vec<f32>,
    w: Vec<f32>,
    gate_m: Vec<f32>,
    gate_h: Vec<f32>,
    gate_n: Vec<f32>,
    calcium: Vec<f32>,
    i_syn: Vec<f32>,
    i_ext: Vec<f32>,
    bias: Vec<f32>,
    noise: Vec<f32>,
    spike: Vec<u8>,
    last_spike: Vec<f32>,
    refractory_until: Vec<f32>,
    flash: Vec<f32>,
    rate: Vec<f32>,
    spike_count: Vec<u32>,
    model: Vec<u8>,
    polarity: Vec<u8>,
    enabled: Vec<u8>,
    params: Vec<f32>,
    scale: Vec<f32>,
    seed: Vec<u32>,
    archetype: Vec<u8>,
    population: Vec<u16>,
    flags: Vec<u8>,
}

impl Neurons {
    fn new(capacity: usize) -> Self {
        let mut n = Neurons {
            capacity: 0,
            count: 0,
            position: Vec::new(),
            v: Vec::new(),
            w: Vec::new(),
            gate_m: Vec::new(),
            gate_h: Vec::new(),
            gate_n: Vec::new(),
            calcium: Vec::new(),
            i_syn: Vec::new(),
            i_ext: Vec::new(),
            bias: Vec::new(),
            noise: Vec::new(),
            spike: Vec::new(),
            last_spike: Vec::new(),
            refractory_until: Vec::new(),
            flash: Vec::new(),
            rate: Vec::new(),
            spike_count: Vec::new(),
            model: Vec::new(),
            polarity: Vec::new(),
            enabled: Vec::new(),
            params: Vec::new(),
            scale: Vec::new(),
            seed: Vec::new(),
            archetype: Vec::new(),
            population: Vec::new(),
            flags: Vec::new(),
        };
        n.grow_to(capacity.max(1));
        n
    }

    /// Column defaults reproduce `allocateNeuronBuffers` and the tail fills in
    /// `growNeuronBuffers`: lastSpike -Infinity, enabled 1, scale 1, population
    /// 0xffff, everything else zero.
    fn grow_to(&mut self, capacity: usize) {
        self.position.resize(capacity * 3, 0.0);
        self.v.resize(capacity, 0.0);
        self.w.resize(capacity, 0.0);
        self.gate_m.resize(capacity, 0.0);
        self.gate_h.resize(capacity, 0.0);
        self.gate_n.resize(capacity, 0.0);
        self.calcium.resize(capacity, 0.0);
        self.i_syn.resize(capacity, 0.0);
        self.i_ext.resize(capacity, 0.0);
        self.bias.resize(capacity, 0.0);
        self.noise.resize(capacity, 0.0);
        self.spike.resize(capacity, 0);
        self.last_spike.resize(capacity, f32::NEG_INFINITY);
        self.refractory_until.resize(capacity, 0.0);
        self.flash.resize(capacity, 0.0);
        self.rate.resize(capacity, 0.0);
        self.spike_count.resize(capacity, 0);
        self.model.resize(capacity, 0);
        self.polarity.resize(capacity, 0);
        self.enabled.resize(capacity, 1);
        self.params.resize(capacity * NEURON_PARAM_STRIDE, 0.0);
        self.scale.resize(capacity, 1.0);
        self.seed.resize(capacity, 0);
        self.archetype.resize(capacity, 0);
        self.population.resize(capacity, NO_POPULATION);
        self.flags.resize(capacity, 0);
        self.capacity = capacity;
    }
}

struct Synapses {
    capacity: usize,
    count: usize,
    pre: Vec<u32>,
    post: Vec<u32>,
    weight: Vec<f32>,
    delay: Vec<f32>,
    g_rise: Vec<f32>,
    g_decay: Vec<f32>,
    tau_rise: Vec<f32>,
    tau_decay: Vec<f32>,
    e_rev: Vec<f32>,
    mg_block: Vec<f32>,
    pre_trace: Vec<f32>,
    post_trace: Vec<f32>,
    pre_trace_slow: Vec<f32>,
    post_trace_slow: Vec<f32>,
    stp_r: Vec<f32>,
    stp_u: Vec<f32>,
    release_prob: Vec<f32>,
    receptor: Vec<u8>,
    plasticity: Vec<u8>,
    enabled: Vec<u8>,
    params: Vec<f32>,
    activity: Vec<f32>,
    arc: Vec<f32>,
}

impl Synapses {
    fn new(capacity: usize) -> Self {
        let mut s = Synapses {
            capacity: 0,
            count: 0,
            pre: Vec::new(),
            post: Vec::new(),
            weight: Vec::new(),
            delay: Vec::new(),
            g_rise: Vec::new(),
            g_decay: Vec::new(),
            tau_rise: Vec::new(),
            tau_decay: Vec::new(),
            e_rev: Vec::new(),
            mg_block: Vec::new(),
            pre_trace: Vec::new(),
            post_trace: Vec::new(),
            pre_trace_slow: Vec::new(),
            post_trace_slow: Vec::new(),
            stp_r: Vec::new(),
            stp_u: Vec::new(),
            release_prob: Vec::new(),
            receptor: Vec::new(),
            plasticity: Vec::new(),
            enabled: Vec::new(),
            params: Vec::new(),
            activity: Vec::new(),
            arc: Vec::new(),
        };
        s.grow_to(capacity.max(1));
        s
    }

    /// Column defaults reproduce `allocateSynapseBuffers`: stpR 1, releaseProb 1,
    /// enabled 1, everything else zero.
    fn grow_to(&mut self, capacity: usize) {
        self.pre.resize(capacity, 0);
        self.post.resize(capacity, 0);
        self.weight.resize(capacity, 0.0);
        self.delay.resize(capacity, 0.0);
        self.g_rise.resize(capacity, 0.0);
        self.g_decay.resize(capacity, 0.0);
        self.tau_rise.resize(capacity, 0.0);
        self.tau_decay.resize(capacity, 0.0);
        self.e_rev.resize(capacity, 0.0);
        self.mg_block.resize(capacity, 0.0);
        self.pre_trace.resize(capacity, 0.0);
        self.post_trace.resize(capacity, 0.0);
        self.pre_trace_slow.resize(capacity, 0.0);
        self.post_trace_slow.resize(capacity, 0.0);
        self.stp_r.resize(capacity, 1.0);
        self.stp_u.resize(capacity, 0.0);
        self.release_prob.resize(capacity, 1.0);
        self.receptor.resize(capacity, 0);
        self.plasticity.resize(capacity, 0);
        self.enabled.resize(capacity, 1);
        self.params.resize(capacity * SYNAPSE_PARAM_STRIDE, 0.0);
        self.activity.resize(capacity, 0.0);
        self.arc.resize(capacity, 0.0);
        self.capacity = capacity;
    }
}

/// Compressed adjacency from presynaptic slot to outgoing synapse indices.
#[derive(Default)]
struct OutEdges {
    offsets: Vec<u32>,
    cursor: Vec<u32>,
    targets: Vec<u32>,
}

impl OutEdges {
    fn rebuild(&mut self, neuron_count: usize, synapse_count: usize, pre: &[u32]) {
        self.offsets.clear();
        self.offsets.resize(neuron_count + 1, 0);
        for i in 0..synapse_count {
            let p = pre.get(i).copied().unwrap_or(u32::MAX) as usize;
            if p < neuron_count {
                self.offsets[p + 1] += 1;
            }
        }
        for i in 0..neuron_count {
            self.offsets[i + 1] += self.offsets[i];
        }
        let total = self.offsets[neuron_count] as usize;
        self.targets.clear();
        self.targets.resize(total, 0);
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.offsets);
        for i in 0..synapse_count {
            let p = pre.get(i).copied().unwrap_or(u32::MAX) as usize;
            if p < neuron_count {
                let slot = self.cursor[p] as usize;
                if slot < total {
                    self.targets[slot] = i as u32;
                    self.cursor[p] += 1;
                }
            }
        }
    }

    #[inline]
    fn outgoing(&self, pre: usize) -> &[u32] {
        match (self.offsets.get(pre), self.offsets.get(pre + 1)) {
            (Some(&a), Some(&b)) if a <= b && (b as usize) <= self.targets.len() => {
                &self.targets[a as usize..b as usize]
            }
            _ => &[],
        }
    }
}

/// The simulation. Owns every buffer; JavaScript views them in place.
#[wasm_bindgen]
pub struct Network {
    neurons: Neurons,
    synapses: Synapses,
    delays: DelayQueue,
    out_edges: OutEdges,
    rng: Pcg32,

    kinetics: Vec<Kinetics>,
    kin_tau_rise: Vec<f32>,
    kin_tau_decay: Vec<f32>,
    kin_dt: f64,

    log_neuron: Vec<u32>,
    log_time: Vec<f32>,
    log_head: u32,
    pending: VecDeque<(u32, f32)>,
    scratch: Vec<u32>,

    fired: Vec<u32>,
    time: f64,
    steps: u64,
    total_spikes: u32,
}

#[wasm_bindgen]
impl Network {
    /// Allocate a network with room for `neuron_capacity` neurons and
    /// `synapse_capacity` synapses. Both live counts start at zero; call
    /// `resize_neurons` / `resize_synapses` to make slots active.
    #[wasm_bindgen(constructor)]
    pub fn new(neuron_capacity: usize, synapse_capacity: usize) -> Network {
        let synapses = Synapses::new(synapse_capacity);
        let cap = synapses.capacity;
        Network {
            neurons: Neurons::new(neuron_capacity),
            synapses,
            delays: DelayQueue::new(
                delay::DEFAULT_BUCKETS,
                delay::DEFAULT_RESOLUTION,
                delay::DEFAULT_STRIDE,
            ),
            out_edges: OutEdges::default(),
            rng: Pcg32::default(),
            kinetics: vec![Kinetics::default(); cap],
            kin_tau_rise: vec![f32::NAN; cap],
            kin_tau_decay: vec![f32::NAN; cap],
            kin_dt: f64::NAN,
            log_neuron: vec![0; DEFAULT_SPIKE_LOG_CAPACITY],
            log_time: vec![0.0; DEFAULT_SPIKE_LOG_CAPACITY],
            log_head: 0,
            pending: VecDeque::new(),
            scratch: Vec::new(),
            fired: Vec::new(),
            time: 0.0,
            steps: 0,
            total_spikes: 0,
        }
    }

    // ----------------------------------------------------------------- sizing

    /// Set the live neuron count, growing capacity by doubling when needed.
    ///
    /// Returns true when the backing storage moved, in which case **every**
    /// neuron pointer previously handed to JavaScript is stale and all views must
    /// be rebuilt from fresh `*_ptr()` calls.
    pub fn resize_neurons(&mut self, count: usize) -> bool {
        let mut moved = false;
        if count > self.neurons.capacity {
            let mut cap = self.neurons.capacity.max(1);
            while cap < count {
                cap *= 2;
            }
            self.neurons.grow_to(cap);
            moved = true;
        }
        self.neurons.count = count;
        moved
    }

    /// Set the live synapse count, growing capacity by doubling when needed.
    ///
    /// Returns true when the backing storage moved; see `resize_neurons`.
    pub fn resize_synapses(&mut self, count: usize) -> bool {
        let mut moved = false;
        if count > self.synapses.capacity {
            let mut cap = self.synapses.capacity.max(1);
            while cap < count {
                cap *= 2;
            }
            self.synapses.grow_to(cap);
            self.kinetics.resize(cap, Kinetics::default());
            // NaN never compares equal, so every entry reads as dirty and is
            // recomputed on the next step.
            self.kin_tau_rise.clear();
            self.kin_tau_rise.resize(cap, f32::NAN);
            self.kin_tau_decay.clear();
            self.kin_tau_decay.resize(cap, f32::NAN);
            moved = true;
        }
        self.synapses.count = count;
        moved
    }

    /// Resize the delay calendar; `buckets * resolution` is the longest
    /// representable conduction delay. Discards anything queued.
    pub fn configure_delays(&mut self, buckets: usize, resolution: f64, stride: usize) {
        self.delays = DelayQueue::new(buckets, resolution, stride);
        self.delays.clear(self.time);
    }

    /// Resize the spike-log ring. Discards its contents.
    pub fn configure_spike_log(&mut self, capacity: usize) {
        let cap = capacity.max(1);
        self.log_neuron.clear();
        self.log_neuron.resize(cap, 0);
        self.log_time.clear();
        self.log_time.resize(cap, 0.0);
        self.log_head = 0;
    }

    pub fn neuron_len(&self) -> u32 {
        self.neurons.count as u32
    }

    pub fn neuron_capacity(&self) -> u32 {
        self.neurons.capacity as u32
    }

    pub fn synapse_len(&self) -> u32 {
        self.synapses.count as u32
    }

    pub fn synapse_capacity(&self) -> u32 {
        self.synapses.capacity as u32
    }

    // --------------------------------------------------------- neuron columns

    pub fn neuron_position_ptr(&self) -> u32 {
        ptr_of(&self.neurons.position)
    }
    pub fn neuron_v_ptr(&self) -> u32 {
        ptr_of(&self.neurons.v)
    }
    pub fn neuron_w_ptr(&self) -> u32 {
        ptr_of(&self.neurons.w)
    }
    pub fn neuron_gate_m_ptr(&self) -> u32 {
        ptr_of(&self.neurons.gate_m)
    }
    pub fn neuron_gate_h_ptr(&self) -> u32 {
        ptr_of(&self.neurons.gate_h)
    }
    pub fn neuron_gate_n_ptr(&self) -> u32 {
        ptr_of(&self.neurons.gate_n)
    }
    pub fn neuron_calcium_ptr(&self) -> u32 {
        ptr_of(&self.neurons.calcium)
    }
    pub fn neuron_i_syn_ptr(&self) -> u32 {
        ptr_of(&self.neurons.i_syn)
    }
    pub fn neuron_i_ext_ptr(&self) -> u32 {
        ptr_of(&self.neurons.i_ext)
    }
    pub fn neuron_bias_ptr(&self) -> u32 {
        ptr_of(&self.neurons.bias)
    }
    pub fn neuron_noise_ptr(&self) -> u32 {
        ptr_of(&self.neurons.noise)
    }
    pub fn neuron_spike_ptr(&self) -> u32 {
        ptr_of(&self.neurons.spike)
    }
    pub fn neuron_last_spike_ptr(&self) -> u32 {
        ptr_of(&self.neurons.last_spike)
    }
    pub fn neuron_refractory_until_ptr(&self) -> u32 {
        ptr_of(&self.neurons.refractory_until)
    }
    pub fn neuron_flash_ptr(&self) -> u32 {
        ptr_of(&self.neurons.flash)
    }
    pub fn neuron_rate_ptr(&self) -> u32 {
        ptr_of(&self.neurons.rate)
    }
    pub fn neuron_spike_count_ptr(&self) -> u32 {
        ptr_of(&self.neurons.spike_count)
    }
    pub fn neuron_model_ptr(&self) -> u32 {
        ptr_of(&self.neurons.model)
    }
    pub fn neuron_polarity_ptr(&self) -> u32 {
        ptr_of(&self.neurons.polarity)
    }
    pub fn neuron_enabled_ptr(&self) -> u32 {
        ptr_of(&self.neurons.enabled)
    }
    pub fn neuron_params_ptr(&self) -> u32 {
        ptr_of(&self.neurons.params)
    }
    pub fn neuron_scale_ptr(&self) -> u32 {
        ptr_of(&self.neurons.scale)
    }
    pub fn neuron_seed_ptr(&self) -> u32 {
        ptr_of(&self.neurons.seed)
    }
    pub fn neuron_archetype_ptr(&self) -> u32 {
        ptr_of(&self.neurons.archetype)
    }
    pub fn neuron_population_ptr(&self) -> u32 {
        ptr_of(&self.neurons.population)
    }
    pub fn neuron_flags_ptr(&self) -> u32 {
        ptr_of(&self.neurons.flags)
    }

    // -------------------------------------------------------- synapse columns

    pub fn synapse_pre_ptr(&self) -> u32 {
        ptr_of(&self.synapses.pre)
    }
    pub fn synapse_post_ptr(&self) -> u32 {
        ptr_of(&self.synapses.post)
    }
    pub fn synapse_weight_ptr(&self) -> u32 {
        ptr_of(&self.synapses.weight)
    }
    pub fn synapse_delay_ptr(&self) -> u32 {
        ptr_of(&self.synapses.delay)
    }
    pub fn synapse_g_rise_ptr(&self) -> u32 {
        ptr_of(&self.synapses.g_rise)
    }
    pub fn synapse_g_decay_ptr(&self) -> u32 {
        ptr_of(&self.synapses.g_decay)
    }
    pub fn synapse_tau_rise_ptr(&self) -> u32 {
        ptr_of(&self.synapses.tau_rise)
    }
    pub fn synapse_tau_decay_ptr(&self) -> u32 {
        ptr_of(&self.synapses.tau_decay)
    }
    pub fn synapse_e_rev_ptr(&self) -> u32 {
        ptr_of(&self.synapses.e_rev)
    }
    pub fn synapse_mg_block_ptr(&self) -> u32 {
        ptr_of(&self.synapses.mg_block)
    }
    pub fn synapse_pre_trace_ptr(&self) -> u32 {
        ptr_of(&self.synapses.pre_trace)
    }
    pub fn synapse_post_trace_ptr(&self) -> u32 {
        ptr_of(&self.synapses.post_trace)
    }
    pub fn synapse_pre_trace_slow_ptr(&self) -> u32 {
        ptr_of(&self.synapses.pre_trace_slow)
    }
    pub fn synapse_post_trace_slow_ptr(&self) -> u32 {
        ptr_of(&self.synapses.post_trace_slow)
    }
    pub fn synapse_stp_r_ptr(&self) -> u32 {
        ptr_of(&self.synapses.stp_r)
    }
    pub fn synapse_stp_u_ptr(&self) -> u32 {
        ptr_of(&self.synapses.stp_u)
    }
    pub fn synapse_release_prob_ptr(&self) -> u32 {
        ptr_of(&self.synapses.release_prob)
    }
    pub fn synapse_receptor_ptr(&self) -> u32 {
        ptr_of(&self.synapses.receptor)
    }
    pub fn synapse_plasticity_ptr(&self) -> u32 {
        ptr_of(&self.synapses.plasticity)
    }
    pub fn synapse_enabled_ptr(&self) -> u32 {
        ptr_of(&self.synapses.enabled)
    }
    pub fn synapse_params_ptr(&self) -> u32 {
        ptr_of(&self.synapses.params)
    }
    pub fn synapse_activity_ptr(&self) -> u32 {
        ptr_of(&self.synapses.activity)
    }
    pub fn synapse_arc_ptr(&self) -> u32 {
        ptr_of(&self.synapses.arc)
    }

    // ----------------------------------------------------- delay queue access

    pub fn delay_entries_ptr(&self) -> u32 {
        ptr_of(self.delays.entries())
    }
    pub fn delay_amplitude_ptr(&self) -> u32 {
        ptr_of(self.delays.amplitudes())
    }
    pub fn delay_counts_ptr(&self) -> u32 {
        ptr_of(self.delays.counts())
    }
    pub fn delay_buckets(&self) -> u32 {
        self.delays.buckets() as u32
    }
    pub fn delay_stride(&self) -> u32 {
        self.delays.stride() as u32
    }
    pub fn delay_resolution(&self) -> f64 {
        self.delays.resolution()
    }
    /// Longest conduction delay the queue can represent (ms).
    pub fn delay_horizon(&self) -> f64 {
        self.delays.horizon()
    }
    /// Events lost because their arrival bucket was full.
    pub fn delay_dropped(&self) -> u32 {
        self.delays.dropped()
    }
    /// Events whose delay exceeded the horizon and were pulled back to it.
    pub fn delay_clamped(&self) -> u32 {
        self.delays.clamped()
    }
    pub fn delay_pending(&self) -> u32 {
        self.delays.pending()
    }

    // -------------------------------------------------------------- spike log

    pub fn spike_log_neuron_ptr(&self) -> u32 {
        ptr_of(&self.log_neuron)
    }
    pub fn spike_log_time_ptr(&self) -> u32 {
        ptr_of(&self.log_time)
    }
    pub fn spike_log_capacity(&self) -> u32 {
        self.log_neuron.len() as u32
    }
    /// Total events written since the last reset; the live write cursor is
    /// `spike_log_head() % spike_log_capacity()`.
    pub fn spike_log_head(&self) -> u32 {
        self.log_head
    }

    /// Reserve a scratch region inside wasm memory that JavaScript can hand to
    /// `drain_spikes` as a destination. Returns true when the region moved, which
    /// invalidates any view built on the previous `scratch_ptr()`.
    pub fn reserve_scratch(&mut self, words: usize) -> bool {
        if words <= self.scratch.len() {
            return false;
        }
        self.scratch.clear();
        self.scratch.resize(words, 0);
        true
    }

    pub fn scratch_ptr(&self) -> u32 {
        ptr_of(&self.scratch)
    }

    pub fn scratch_len(&self) -> u32 {
        self.scratch.len() as u32
    }

    /// Copy up to `max` pending spike events to `out_ptr` and remove them from
    /// the pending queue. Returns the number of events written.
    ///
    /// Each event occupies two 32-bit words: word 0 is the neuron slot as a
    /// `u32`, word 1 is the spike time as an `f32`. Overlay a `Uint32Array` and a
    /// `Float32Array` on the same region and read index `2*k` from the first and
    /// `2*k+1` from the second. `out_ptr` must be 4-byte aligned and address at
    /// least `2 * max` words of memory the caller owns — `reserve_scratch(2 * max)`
    /// followed by `scratch_ptr()` is the intended source.
    pub fn drain_spikes(&mut self, out_ptr: u32, max: u32) -> u32 {
        if out_ptr == 0 || max == 0 {
            return 0;
        }
        let n = self.pending.len().min(max as usize);
        if n == 0 {
            return 0;
        }
        // SAFETY: the caller supplies an aligned pointer into this module's own
        // linear memory with room for 2 * max words, as documented above. The
        // slice is confined to the 2 * n words actually written.
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as usize as *mut u32, n * 2) };
        self.drain_into(out)
    }

    pub fn pending_spikes(&self) -> u32 {
        self.pending.len() as u32
    }

    // --------------------------------------------------------------- controls

    /// Reseed the deterministic generator. The same seed replayed over the same
    /// circuit reproduces a run exactly.
    pub fn set_seed(&mut self, seed: u32) {
        self.rng.reset(seed);
    }

    pub fn seed(&self) -> u32 {
        self.rng.seed()
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    /// Move simulation time without touching state. Pending delay events keep
    /// their absolute arrival buckets and the calendar cursor re-anchors.
    pub fn set_time(&mut self, t: f64) {
        self.time = if t.is_finite() { t } else { 0.0 };
        self.delays.rebase(self.time);
    }

    /// Substeps executed since the last reset.
    pub fn step_index(&self) -> u32 {
        self.steps as u32
    }

    /// Total spikes emitted since the last reset.
    pub fn spike_count(&self) -> u32 {
        self.total_spikes
    }

    /// Return every state variable to rest without changing topology.
    ///
    /// Voltages go to the model's resting potential and the Hodgkin-Huxley gates
    /// to their steady state there; conductances, traces, currents, the delay
    /// calendar and the spike log are cleared. Parameters, positions, weights,
    /// delays and connectivity are untouched.
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.steps = 0;
        self.total_spikes = 0;

        let n = self.neurons.count.min(self.neurons.capacity);
        for i in 0..n {
            let model = self.neurons.model[i];
            let rest = match block::<NEURON_PARAM_STRIDE>(&self.neurons.params, i) {
                Some(p) => model::resting_state(model, p),
                None => State::default(),
            };
            self.neurons.v[i] = rest.v as f32;
            self.neurons.w[i] = rest.w as f32;
            self.neurons.gate_m[i] = rest.m as f32;
            self.neurons.gate_h[i] = rest.h as f32;
            self.neurons.gate_n[i] = rest.n as f32;
            self.neurons.calcium[i] = 0.0;
            self.neurons.i_syn[i] = 0.0;
            self.neurons.spike[i] = 0;
            self.neurons.last_spike[i] = f32::NEG_INFINITY;
            self.neurons.refractory_until[i] = 0.0;
            self.neurons.rate[i] = 0.0;
            self.neurons.spike_count[i] = 0;
        }

        let m = self.synapses.count.min(self.synapses.capacity);
        for i in 0..m {
            self.synapses.g_rise[i] = 0.0;
            self.synapses.g_decay[i] = 0.0;
            self.synapses.pre_trace[i] = 0.0;
            self.synapses.post_trace[i] = 0.0;
            self.synapses.pre_trace_slow[i] = 0.0;
            self.synapses.post_trace_slow[i] = 0.0;
            self.synapses.stp_r[i] = 1.0;
            self.synapses.stp_u[i] = block::<SYNAPSE_PARAM_STRIDE>(&self.synapses.params, i)
                .map(|p| p[synapse::SYN_STP_U])
                .unwrap_or(0.0);
            self.synapses.activity[i] = 0.0;
        }

        self.delays.clear(0.0);
        self.log_head = 0;
        self.pending.clear();
        self.fired.clear();
        self.kin_dt = f64::NAN;
    }

    /// Advance the simulation by `steps` substeps of `dt` milliseconds.
    ///
    /// `integrator` is 0 Euler, 1 RK2, 2 RK4, 3 exponential Euler; any other
    /// value falls back to exponential Euler. `gain` scales every synaptic
    /// conductance, `noise` adds a per-neuron current with that standard
    /// deviation on top of each neuron's own noise column, and `plasticity`
    /// gates the weight-update pass. Returns the number of spikes emitted across
    /// all substeps.
    pub fn step(
        &mut self,
        steps: u32,
        dt: f32,
        integrator: u32,
        gain: f32,
        noise: f32,
        plasticity: bool,
    ) -> u32 {
        let dt = dt as f64;
        if !dt.is_finite() || dt <= 0.0 || steps == 0 {
            return 0;
        }
        // Derived fresh on every call rather than from a dirty flag: `pre` lives
        // in linear memory that JavaScript rewrites without notifying us, so a
        // cached adjacency could route spikes down edges that no longer exist.
        self.out_edges
            .rebuild(self.neurons.count, self.synapses.count, &self.synapses.pre);

        let gain = gain as f64;
        let noise = (noise as f64).max(0.0);
        let mut emitted = 0u32;
        for _ in 0..steps {
            emitted = emitted.saturating_add(self.substep(dt, integrator, gain, noise, plasticity));
        }
        emitted
    }

    // -------------------------------------------------------- graph readouts

    /// Weakly connected component id per neuron slot.
    pub fn graph_components(&self) -> Vec<u32> {
        graph::components(&self.topology())
    }

    pub fn graph_component_count(&self) -> u32 {
        graph::distinct_count(&graph::components(&self.topology()))
    }

    /// Strongly connected component id per neuron slot. Any id shared by two or
    /// more neurons is a recurrent motif.
    pub fn graph_strongly_connected_components(&self) -> Vec<u32> {
        graph::strongly_connected_components(&self.topology())
    }

    pub fn graph_strongly_connected_count(&self) -> u32 {
        graph::distinct_count(&graph::strongly_connected_components(&self.topology()))
    }

    /// Slots that participate in a directed cycle, ascending.
    pub fn graph_recurrent_nodes(&self) -> Vec<u32> {
        graph::recurrent_nodes(&self.topology())
    }

    pub fn graph_in_degrees(&self) -> Vec<u32> {
        graph::in_degrees(&self.topology())
    }

    pub fn graph_out_degrees(&self) -> Vec<u32> {
        graph::out_degrees(&self.topology())
    }

    /// Breadth-first shortest path from `from` to `to`, inclusive of both
    /// endpoints. Empty when no directed path exists.
    pub fn graph_shortest_path(&self, from: usize, to: usize) -> Vec<u32> {
        graph::shortest_path(&self.topology(), from, to)
    }

    pub fn graph_has_cycle(&self) -> bool {
        graph::has_cycle(&self.topology())
    }

    /// Local clustering coefficient per neuron slot, on the undirected
    /// projection of the circuit.
    pub fn graph_clustering(&self) -> Vec<f32> {
        graph::clustering(&self.topology())
    }

    pub fn graph_average_clustering(&self) -> f32 {
        graph::average_clustering(&graph::clustering(&self.topology()))
    }
}

impl Network {
    /// Shared body of `drain_spikes`, working on a checked slice so that the only
    /// unsafe code in the crate is the pointer-to-slice conversion itself.
    fn drain_into(&mut self, out: &mut [u32]) -> u32 {
        let n = self.pending.len().min(out.len() / 2);
        for k in 0..n {
            let Some(&(slot, time)) = self.pending.get(k) else {
                break;
            };
            out[k * 2] = slot;
            out[k * 2 + 1] = time.to_bits();
        }
        self.pending.drain(..n);
        n as u32
    }

    fn topology(&self) -> graph::Topology {
        graph::build(
            self.neurons.count,
            self.synapses.count,
            &self.synapses.pre,
            &self.synapses.post,
            &self.synapses.enabled,
        )
    }

    /// Recompute the cached per-synapse decay factors and peak normalisation.
    /// Entries refresh only when `dt` or one of the time constants changed, so
    /// the steady state costs two float compares per synapse instead of three
    /// transcendentals.
    fn refresh_kinetics(&mut self, dt: f64) {
        if self.kinetics.len() < self.synapses.capacity {
            self.kinetics
                .resize(self.synapses.capacity, Kinetics::default());
            self.kin_tau_rise.resize(self.synapses.capacity, f32::NAN);
            self.kin_tau_decay.resize(self.synapses.capacity, f32::NAN);
        }
        let m = self.synapses.count.min(self.synapses.capacity);
        let dt_changed = self.kin_dt != dt;
        if dt_changed {
            self.kin_dt = dt;
        }
        for i in 0..m {
            let tr = self.synapses.tau_rise[i];
            let td = self.synapses.tau_decay[i];
            if dt_changed || self.kin_tau_rise[i] != tr || self.kin_tau_decay[i] != td {
                self.kin_tau_rise[i] = tr;
                self.kin_tau_decay[i] = td;
                self.kinetics[i] = synapse::kinetics(tr, td, dt);
            }
        }
    }

    fn substep(
        &mut self,
        dt: f64,
        integrator: u32,
        gain: f64,
        global_noise: f64,
        plasticity: bool,
    ) -> u32 {
        self.refresh_kinetics(dt);
        let t0 = self.time;
        let t_after = t0 + dt;
        let n = self.neurons.count.min(self.neurons.capacity);
        let m = self.synapses.count.min(self.synapses.capacity);

        // 1. Deliver everything the calendar says has arrived by now.
        {
            let Network {
                delays,
                synapses,
                kinetics,
                ..
            } = self;
            let live = m;
            delays.drain_due(t0, |index, amplitude| {
                let i = index as usize;
                if i >= live {
                    return;
                }
                let Some(k) = kinetics.get(i) else { return };
                let scaled = amplitude * k.norm;
                synapses.g_decay[i] += scaled;
                if !k.single {
                    synapses.g_rise[i] += scaled;
                }
            });
        }

        // 2. Decay conductances, relax short-term plasticity, accumulate current.
        for i in 0..n {
            self.neurons.i_syn[i] = 0.0;
        }
        {
            let Network {
                synapses,
                neurons,
                kinetics,
                ..
            } = self;
            for i in 0..m {
                let Some(k) = kinetics.get(i).copied() else {
                    continue;
                };
                synapses.g_rise[i] *= k.decay_rise;
                synapses.g_decay[i] *= k.decay_decay;

                if let Some(p) = block::<SYNAPSE_PARAM_STRIDE>(&synapses.params, i)
                    && synapse::stp_enabled(p)
                {
                    let mut stp = StpState {
                        r: synapses.stp_r[i],
                        u: synapses.stp_u[i],
                    };
                    synapse::stp_relax(&mut stp, p, dt);
                    synapses.stp_r[i] = stp.r;
                    synapses.stp_u[i] = stp.u;
                }

                // Travel envelope: 1 at emission, linearly to 0 on arrival, so the
                // renderer can place the impulse at 1 - activity along the axon
                // without any extra per-synapse state.
                if synapses.activity[i] > 0.0 {
                    let flight = (synapses.delay[i] as f64).max(dt);
                    synapses.activity[i] =
                        (synapses.activity[i] as f64 - dt / flight).max(0.0) as f32;
                }

                if synapses.enabled[i] == 0 {
                    continue;
                }
                let post = synapses.post[i] as usize;
                if post >= n {
                    continue;
                }

                // Electrical coupling is continuous: current flows whenever the
                // two membranes differ, with no presynaptic spike involved. The
                // TypeScript reference and the WGSL kernel both model it this
                // way, and routing it through the event-driven conductance state
                // would instead make a gap junction silent between spikes.
                if synapses.receptor[i] == synapse::RECEPTOR_GAP {
                    let pre = synapses.pre[i] as usize;
                    if pre < n {
                        let delta = neurons.v[pre] as f64 - neurons.v[post] as f64;
                        let current = synapses.weight[i] as f64 * delta * gain;
                        if current.is_finite() {
                            neurons.i_syn[post] += current as f32;
                        }
                        synapses.activity[i] = (delta.abs() * 0.02).min(1.0) as f32;
                    }
                    continue;
                }

                let g = if k.single {
                    synapses.g_decay[i] as f64
                } else {
                    (synapses.g_decay[i] - synapses.g_rise[i]) as f64
                };
                if g == 0.0 {
                    continue;
                }
                let current = synapse::synaptic_current(
                    g * gain,
                    synapses.e_rev[i] as f64,
                    neurons.v[post] as f64,
                    synapses.mg_block[i] as f64,
                );
                if current.is_finite() {
                    neurons.i_syn[post] += current as f32;
                }
            }
        }

        // 3. Integrate membranes and detect spikes.
        let rate_alpha = (1.0 - (-dt / RATE_TAU_MS).exp()) as f32;
        let calcium_decay = (-dt / CALCIUM_TAU_MS).exp() as f32;
        let instant_rate = (1000.0 / dt) as f32;
        self.fired.clear();
        {
            let Network {
                neurons,
                rng,
                fired,
                ..
            } = self;
            for i in 0..n {
                neurons.calcium[i] *= calcium_decay;
                if neurons.enabled[i] == 0 {
                    neurons.spike[i] = 0;
                    continue;
                }
                let Some(params) = block::<NEURON_PARAM_STRIDE>(&neurons.params, i) else {
                    neurons.spike[i] = 0;
                    continue;
                };
                let params: &Params = params;
                let mut state = State {
                    v: neurons.v[i] as f64,
                    w: neurons.w[i] as f64,
                    m: neurons.gate_m[i] as f64,
                    h: neurons.gate_h[i] as f64,
                    n: neurons.gate_n[i] as f64,
                };
                let mut current = (neurons.i_syn[i] + neurons.i_ext[i] + neurons.bias[i]) as f64;
                let sigma = neurons.noise[i] as f64 + global_noise;
                if sigma > 0.0 {
                    current += rng.normal() * sigma;
                }
                let mut refractory = neurons.refractory_until[i];
                let spiked = model::step_neuron(
                    neurons.model[i],
                    &mut state,
                    params,
                    current,
                    dt,
                    integrator,
                    t0,
                    &mut refractory,
                );
                neurons.v[i] = state.v as f32;
                neurons.w[i] = state.w as f32;
                neurons.gate_m[i] = state.m as f32;
                neurons.gate_h[i] = state.h as f32;
                neurons.gate_n[i] = state.n as f32;
                neurons.refractory_until[i] = refractory;
                neurons.spike[i] = spiked as u8;

                let target = if spiked { instant_rate } else { 0.0 };
                neurons.rate[i] += (target - neurons.rate[i]) * rate_alpha;

                if spiked {
                    neurons.last_spike[i] = t_after as f32;
                    neurons.spike_count[i] = neurons.spike_count[i].saturating_add(1);
                    neurons.calcium[i] += CALCIUM_PER_SPIKE;
                    fired.push(i as u32);
                }
            }
        }

        // 4. Record and propagate.
        let emitted = self.fired.len() as u32;
        if emitted > 0 {
            let capacity = self.log_neuron.len();
            for k in 0..self.fired.len() {
                let slot = self.fired[k];
                if capacity > 0 {
                    let cursor = (self.log_head as usize) % capacity;
                    self.log_neuron[cursor] = slot;
                    self.log_time[cursor] = t_after as f32;
                }
                self.log_head = self.log_head.wrapping_add(1);
                if self.pending.len() >= capacity.max(1) {
                    self.pending.pop_front();
                }
                self.pending.push_back((slot, t_after as f32));
            }

            let Network {
                synapses,
                delays,
                rng,
                out_edges,
                fired,
                ..
            } = self;
            for &slot in fired.iter() {
                for &edge in out_edges.outgoing(slot as usize) {
                    let s = edge as usize;
                    if s >= m || synapses.enabled[s] == 0 {
                        continue;
                    }
                    // Electrical synapses carry current continuously and are
                    // handled in the conductance pass; queueing an event for one
                    // would accumulate conductance nothing ever consumes.
                    if synapses.receptor[s] == synapse::RECEPTOR_GAP {
                        continue;
                    }
                    if !rng.bernoulli(synapses.release_prob[s]) {
                        continue;
                    }
                    let mut amplitude = synapses.weight[s];
                    if let Some(p) = block::<SYNAPSE_PARAM_STRIDE>(&synapses.params, s)
                        && synapse::stp_enabled(p)
                    {
                        let mut stp = StpState {
                            r: synapses.stp_r[s],
                            u: synapses.stp_u[s],
                        };
                        amplitude *= synapse::stp_release(&mut stp, p);
                        synapses.stp_r[s] = stp.r;
                        synapses.stp_u[s] = stp.u;
                    }
                    synapses.activity[s] = 1.0;
                    if amplitude == 0.0 {
                        continue;
                    }
                    let arrival = t_after + (synapses.delay[s] as f64).max(0.0);
                    delays.schedule(arrival, edge, amplitude);
                }
            }
        }

        // 5. Plasticity, driven by the spikes this substep produced.
        if plasticity {
            let Network {
                synapses, neurons, ..
            } = self;
            for i in 0..m {
                let kind = synapses.plasticity[i];
                if kind == PLASTICITY_STATIC {
                    continue;
                }
                let pre = synapses.pre[i] as usize;
                let post = synapses.post[i] as usize;
                if pre >= n || post >= n {
                    continue;
                }
                let Some(params) = block::<SYNAPSE_PARAM_STRIDE>(&synapses.params, i) else {
                    continue;
                };
                let params: &SynParams = params;
                let mut traces = Traces {
                    pre: synapses.pre_trace[i] as f64,
                    post: synapses.post_trace[i] as f64,
                    pre_slow: synapses.pre_trace_slow[i] as f64,
                    post_slow: synapses.post_trace_slow[i] as f64,
                };
                let weight = plasticity::update(
                    kind,
                    synapses.weight[i],
                    &mut traces,
                    params,
                    neurons.spike[pre] != 0,
                    neurons.spike[post] != 0,
                    dt,
                );
                synapses.pre_trace[i] = traces.pre as f32;
                synapses.post_trace[i] = traces.post as f32;
                synapses.pre_trace_slow[i] = traces.pre_slow as f32;
                synapses.post_trace_slow[i] = traces.post_slow as f32;
                synapses.weight[i] = weight;
            }
        }

        self.time = t_after;
        self.steps = self.steps.wrapping_add(1);
        self.total_spikes = self.total_spikes.saturating_add(emitted);
        emitted
    }
}

/// Address of a slice inside wasm linear memory, as the `u32` JavaScript needs
/// for a typed-array `byteOffset`.
#[inline]
fn ptr_of<T>(data: &[T]) -> u32 {
    data.as_ptr() as usize as u32
}
