//! Synaptic conductances, NMDA magnesium block and short-term plasticity.
//!
//! LAYOUT CONTRACT: `RECEPTOR_*` mirrors `RECEPTOR_CODE` and the `SYN_*` slot
//! constants mirror `SYN_PARAM_SLOT` in `packages/shared/src/buffers.ts`.
//! `SYNAPSE_PARAM_STRIDE` mirrors the constant of the same name.

/// Floats reserved per synapse in the packed parameter block.
pub const SYNAPSE_PARAM_STRIDE: usize = 12;

/// Packed parameter block for one synapse.
pub type SynParams = [f32; SYNAPSE_PARAM_STRIDE];

pub const RECEPTOR_AMPA: u8 = 0;
pub const RECEPTOR_NMDA: u8 = 1;
pub const RECEPTOR_GABAA: u8 = 2;
pub const RECEPTOR_GABAB: u8 = 3;
pub const RECEPTOR_GAP: u8 = 4;

pub const SYN_A_PLUS: usize = 0;
pub const SYN_A_MINUS: usize = 1;
pub const SYN_TAU_PLUS: usize = 2;
pub const SYN_TAU_MINUS: usize = 3;
pub const SYN_TAU_X: usize = 4;
pub const SYN_TAU_Y: usize = 5;
pub const SYN_W_MIN: usize = 6;
pub const SYN_W_MAX: usize = 7;
pub const SYN_LEARNING_RATE: usize = 8;
pub const SYN_STP_U: usize = 9;
pub const SYN_STP_TAU_REC: usize = 10;
pub const SYN_STP_TAU_FACIL: usize = 11;

/// Magnesium concentration scale of the Jahr-Stevens NMDA block. The `mgBlock`
/// column carries [Mg2+] in mM; 1.0 mM is the physiological default written by
/// `RECEPTOR_DEFAULTS` in synapse.ts.
const MG_SCALE: f64 = 3.57;
const MG_SLOPE: f64 = 0.062;

/// Per-synapse kinetics derived from the tauRise/tauDecay pair.
#[derive(Clone, Copy, Debug)]
pub struct Kinetics {
    /// Multiplier applied to an arriving amplitude so that peak(gDecay - gRise)
    /// equals that amplitude.
    pub norm: f32,
    pub decay_rise: f32,
    pub decay_decay: f32,
    /// True when the two time constants are indistinguishable (or inverted), in
    /// which case the waveform degenerates to a single exponential carried
    /// entirely by gDecay. The `gap` receptor default (0.05 / 0.05) lands here.
    pub single: bool,
}

impl Default for Kinetics {
    fn default() -> Self {
        Kinetics {
            norm: 1.0,
            decay_rise: 0.0,
            decay_decay: 0.0,
            single: true,
        }
    }
}

/// Peak-normalised dual-exponential kinetics for one (tauRise, tauDecay, dt).
pub fn kinetics(tau_rise: f32, tau_decay: f32, dt: f64) -> Kinetics {
    let tr = tau_rise as f64;
    let td = tau_decay as f64;
    let decay_decay = if td > 0.0 { (-dt / td).exp() } else { 0.0 };
    let decay_rise = if tr > 0.0 { (-dt / tr).exp() } else { 0.0 };
    if !(tr > 0.0) || !(td > 0.0) || td - tr <= 1e-6 * td.max(1.0) {
        return Kinetics {
            norm: 1.0,
            decay_rise: decay_rise as f32,
            decay_decay: decay_decay as f32,
            single: true,
        };
    }
    let t_peak = (tr * td / (td - tr)) * (td / tr).ln();
    let peak = (-t_peak / td).exp() - (-t_peak / tr).exp();
    let norm = if peak > 1e-9 { 1.0 / peak } else { 1.0 };
    Kinetics {
        norm: norm as f32,
        decay_rise: decay_rise as f32,
        decay_decay: decay_decay as f32,
        single: false,
    }
}

/// Jahr-Stevens voltage-dependent magnesium block, a sigmoid in postsynaptic
/// voltage. Returns 1.0 (no block) when the synapse carries no magnesium.
#[inline]
pub fn mg_block(v_post: f64, mg: f64) -> f64 {
    if mg <= 0.0 {
        return 1.0;
    }
    1.0 / (1.0 + mg * (-MG_SLOPE * v_post).exp() / MG_SCALE)
}

/// Tsodyks-Markram state for one synapse: available resources and utilisation.
#[derive(Clone, Copy, Debug)]
pub struct StpState {
    pub r: f32,
    pub u: f32,
}

/// True when the parameter block asks for short-term plasticity. `packSynapseParams`
/// writes 0 into the STP_U slot when STP is disabled, so a non-zero utilisation is
/// the discriminator.
#[inline]
pub fn stp_enabled(params: &SynParams) -> bool {
    params[SYN_STP_U] > 0.0
}

/// Continuous relaxation of the Tsodyks-Markram variables between spikes.
///
/// Solving R and u continuously rather than from an inter-spike interval keeps
/// the state in the two columns that buffers.ts actually provides (`stpR`,
/// `stpU`); there is no per-synapse last-spike-time column to difference against.
pub fn stp_relax(state: &mut StpState, params: &SynParams, dt: f64) {
    let u_base = params[SYN_STP_U] as f64;
    let tau_rec = params[SYN_STP_TAU_REC] as f64;
    let tau_facil = params[SYN_STP_TAU_FACIL] as f64;

    let r = state.r as f64;
    state.r = if tau_rec > 0.0 {
        (1.0 - (1.0 - r) * (-dt / tau_rec).exp()) as f32
    } else {
        1.0
    };

    let u = state.u as f64;
    state.u = if tau_facil > 0.0 {
        (u_base + (u - u_base) * (-dt / tau_facil).exp()) as f32
    } else {
        u_base as f32
    };
}

/// Consume resources for one presynaptic spike and return the release amplitude
/// factor in 0..1 that scales the peak conductance.
pub fn stp_release(state: &mut StpState, params: &SynParams) -> f32 {
    let u_base = params[SYN_STP_U] as f64;
    let tau_facil = params[SYN_STP_TAU_FACIL] as f64;
    let mut u = state.u as f64;
    if tau_facil > 0.0 {
        u += u_base * (1.0 - u);
    } else {
        u = u_base;
    }
    let r = (state.r as f64).clamp(0.0, 1.0);
    let amplitude = u * r;
    state.u = u.clamp(0.0, 1.0) as f32;
    state.r = (r - amplitude).clamp(0.0, 1.0) as f32;
    amplitude as f32
}

/// Postsynaptic current in pA for a conductance `g` (nS) at voltage `v_post` (mV).
///
/// Sign is carried entirely by the reversal potential, per the note on
/// `Synapse.weight` in synapse.ts: an inhibitory neuron gets a negative eRev, not
/// a negative weight, so neuron polarity never enters this expression.
#[inline]
pub fn synaptic_current(g: f64, e_rev: f64, v_post: f64, mg: f64) -> f64 {
    let conductance = if mg > 0.0 { g * mg_block(v_post, mg) } else { g };
    conductance * (e_rev - v_post)
}
