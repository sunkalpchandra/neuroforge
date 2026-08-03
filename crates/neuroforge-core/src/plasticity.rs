//! Long-term plasticity rules over the four trace columns.
//!
//! LAYOUT CONTRACT: `PLASTICITY_*` mirrors `PLASTICITY_CODE` in
//! `packages/shared/src/buffers.ts`; the trace fields map onto the `preTrace`,
//! `postTrace`, `preTraceSlow` and `postTraceSlow` columns in that order.

use crate::synapse::{
    SYN_A_MINUS, SYN_A_PLUS, SYN_LEARNING_RATE, SYN_TAU_MINUS, SYN_TAU_PLUS, SYN_TAU_X, SYN_TAU_Y,
    SYN_W_MAX, SYN_W_MIN, SynParams,
};

pub const PLASTICITY_STATIC: u8 = 0;
pub const PLASTICITY_STDP: u8 = 1;
pub const PLASTICITY_TRIPLET_STDP: u8 = 2;
pub const PLASTICITY_HEBBIAN: u8 = 3;
pub const PLASTICITY_OJA: u8 = 4;

/// The four eligibility traces of one synapse.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Traces {
    pub pre: f64,
    pub post: f64,
    pub pre_slow: f64,
    pub post_slow: f64,
}

#[inline]
fn decay(x: f64, tau: f64, dt: f64) -> f64 {
    if tau > 0.0 { x * (-dt / tau).exp() } else { 0.0 }
}

/// Advance the traces and the weight of one synapse by `dt`.
///
/// Traces are decayed first, the weight update reads the decayed values, and only
/// then do this step's spikes increment the traces. That ordering is what makes
/// the triplet rule read the *previous* slow trace (the "epsilon before the
/// spike" convention of Pfister & Gerstner) instead of one contaminated by the
/// spike currently being processed.
pub fn update(
    kind: u8,
    weight: f32,
    traces: &mut Traces,
    params: &SynParams,
    pre_spike: bool,
    post_spike: bool,
    dt: f64,
) -> f32 {
    if kind == PLASTICITY_STATIC {
        return weight;
    }

    let a_plus = params[SYN_A_PLUS] as f64;
    let a_minus = params[SYN_A_MINUS] as f64;
    let tau_plus = params[SYN_TAU_PLUS] as f64;
    let tau_minus = params[SYN_TAU_MINUS] as f64;
    let tau_x = params[SYN_TAU_X] as f64;
    let tau_y = params[SYN_TAU_Y] as f64;
    let w_min = params[SYN_W_MIN] as f64;
    let w_max = params[SYN_W_MAX] as f64;
    let rate = params[SYN_LEARNING_RATE] as f64;

    traces.pre = decay(traces.pre, tau_plus, dt);
    traces.post = decay(traces.post, tau_minus, dt);
    traces.pre_slow = decay(traces.pre_slow, tau_x, dt);
    traces.post_slow = decay(traces.post_slow, tau_y, dt);

    let w = weight as f64;
    let mut delta = 0.0;
    match kind {
        PLASTICITY_STDP => {
            if post_spike {
                delta += a_plus * traces.pre;
            }
            if pre_spike {
                delta -= a_minus * traces.post;
            }
        }
        PLASTICITY_TRIPLET_STDP => {
            // The parameter block carries a single amplitude per direction, so the
            // pair term and the triplet term of each direction share it: A2+ = A3+
            // = aPlus and A2- = A3- = aMinus.
            if post_spike {
                delta += traces.pre * (a_plus + a_plus * traces.post_slow);
            }
            if pre_spike {
                delta -= traces.post * (a_minus + a_minus * traces.pre_slow);
            }
        }
        PLASTICITY_HEBBIAN => {
            delta += a_plus * traces.pre * traces.post * dt;
        }
        PLASTICITY_OJA => {
            delta += a_plus * traces.post * (traces.pre - traces.post * w) * dt;
        }
        _ => {}
    }

    if pre_spike {
        traces.pre += 1.0;
        traces.pre_slow += 1.0;
    }
    if post_spike {
        traces.post += 1.0;
        traces.post_slow += 1.0;
    }

    let next = w + rate * delta;
    if !next.is_finite() {
        return weight;
    }
    let (lo, hi) = if w_min <= w_max {
        (w_min, w_max)
    } else {
        (w_max, w_min)
    };
    next.clamp(lo, hi) as f32
}
