//! The five membrane models and the four integrators.
//!
//! LAYOUT CONTRACT: `MODEL_*` mirrors `MODEL_CODE` and the `*_SLOT` constants
//! mirror `PARAM_SLOT` in `packages/shared/src/buffers.ts`. `NEURON_PARAM_STRIDE`
//! mirrors the constant of the same name. The TypeScript integrator, the WGSL
//! shaders and this file all index the same packed block; changing an offset in
//! one place without the others silently corrupts every simulation.
//!
//! Integrator codes follow the declaration order of
//! `SimulationSettings['integrator']` in `packages/shared/src/circuit.ts`.
//!
//! Arithmetic is performed in f64 and stored as f32. The TypeScript reference
//! reads an f32 column, computes in f64 (every JS number is an f64) and rounds
//! back to f32 on store; doing the same here is what lets the two backends agree
//! to float tolerance rather than merely to within their own truncation error.

/// Floats reserved per neuron in the packed parameter block.
pub const NEURON_PARAM_STRIDE: usize = 16;

/// Packed parameter block for one neuron.
pub type Params = [f32; NEURON_PARAM_STRIDE];

pub const MODEL_LIF: u8 = 0;
pub const MODEL_IZHIKEVICH: u8 = 1;
pub const MODEL_HODGKIN_HUXLEY: u8 = 2;
pub const MODEL_ADEX: u8 = 3;
pub const MODEL_MORRIS_LECAR: u8 = 4;

pub const INTEGRATOR_EULER: u32 = 0;
pub const INTEGRATOR_RK2: u32 = 1;
pub const INTEGRATOR_RK4: u32 = 2;
pub const INTEGRATOR_EXP_EULER: u32 = 3;

pub const LIF_CM: usize = 0;
pub const LIF_GL: usize = 1;
pub const LIF_EL: usize = 2;
pub const LIF_VTHRESH: usize = 3;
pub const LIF_VRESET: usize = 4;
pub const LIF_TREFRACT: usize = 5;

pub const IZH_A: usize = 0;
pub const IZH_B: usize = 1;
pub const IZH_C: usize = 2;
pub const IZH_D: usize = 3;
pub const IZH_VPEAK: usize = 4;
pub const IZH_ISCALE: usize = 5;

pub const HH_CM: usize = 0;
pub const HH_GNA: usize = 1;
pub const HH_GK: usize = 2;
pub const HH_GL: usize = 3;
pub const HH_ENA: usize = 4;
pub const HH_EK: usize = 5;
pub const HH_EL: usize = 6;
pub const HH_VDETECT: usize = 7;
pub const HH_Q10: usize = 8;

pub const ADEX_CM: usize = 0;
pub const ADEX_GL: usize = 1;
pub const ADEX_EL: usize = 2;
pub const ADEX_DELTAT: usize = 3;
pub const ADEX_VT: usize = 4;
pub const ADEX_VPEAK: usize = 5;
pub const ADEX_VRESET: usize = 6;
pub const ADEX_A: usize = 7;
pub const ADEX_B: usize = 8;
pub const ADEX_TAUW: usize = 9;
pub const ADEX_TREFRACT: usize = 10;

pub const ML_CM: usize = 0;
pub const ML_GCA: usize = 1;
pub const ML_GK: usize = 2;
pub const ML_GL: usize = 3;
pub const ML_ECA: usize = 4;
pub const ML_EK: usize = 5;
pub const ML_EL: usize = 6;
pub const ML_V1: usize = 7;
pub const ML_V2: usize = 8;
pub const ML_V3: usize = 9;
pub const ML_V4: usize = 10;
pub const ML_PHI: usize = 11;
pub const ML_VDETECT: usize = 12;

/// `restingPotential()` in `packages/shared/src/defaults.ts` hard-codes -65 mV
/// for Hodgkin-Huxley and -60.9 mV for Morris-Lecar; both are reproduced here so
/// that a Rust-side reset lands on the same voltage as a TypeScript-side one.
const HH_REST: f64 = -65.0;
const ML_REST: f64 = -60.9;

/// Largest exponent handed to `exp()` in the AdEx upstroke. exp(30) ~ 1e13, which
/// keeps `gL * deltaT * exp(..)` finite in f64 for any sane parameter set while
/// still driving v past vPeak within a single step.
const ADEX_EXP_CLAMP: f64 = 30.0;

/// Continuous state of one neuron. Which fields are live depends on the model:
/// LIF uses v; Izhikevich/AdEx/Morris-Lecar use v and w; Hodgkin-Huxley uses
/// v, m, h and n.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct State {
    pub v: f64,
    pub w: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
}

impl State {
    #[inline]
    fn offset(&self, d: &State, h: f64) -> State {
        State {
            v: self.v + d.v * h,
            w: self.w + d.w * h,
            m: self.m + d.m * h,
            h: self.h + d.h * h,
            n: self.n + d.n * h,
        }
    }
}

#[inline]
fn p(params: &Params, slot: usize) -> f64 {
    params[slot] as f64
}

/// `x / (1 - exp(-x))`, the shape shared by the Hodgkin-Huxley `alpha_m` and
/// `alpha_n` rate functions. The singularity at x = 0 is removable with limit 1;
/// the guard uses the first-order expansion so the function stays smooth through
/// the neighbourhood rather than snapping to a constant.
#[inline]
fn exprel_inv(x: f64) -> f64 {
    let denom = 1.0 - (-x).exp();
    if denom.abs() < 1e-6 {
        1.0 + 0.5 * x
    } else {
        x / denom
    }
}

#[inline]
fn hh_alpha_m(v: f64) -> f64 {
    exprel_inv((v + 40.0) / 10.0)
}

#[inline]
fn hh_beta_m(v: f64) -> f64 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}

#[inline]
fn hh_alpha_h(v: f64) -> f64 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}

#[inline]
fn hh_beta_h(v: f64) -> f64 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}

#[inline]
fn hh_alpha_n(v: f64) -> f64 {
    0.1 * exprel_inv((v + 55.0) / 10.0)
}

#[inline]
fn hh_beta_n(v: f64) -> f64 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

#[inline]
fn q10_of(params: &Params) -> f64 {
    let q = p(params, HH_Q10);
    if q > 0.0 { q } else { 1.0 }
}

#[inline]
fn ml_m_inf(v: f64, params: &Params) -> f64 {
    let v2 = p(params, ML_V2);
    if v2 == 0.0 {
        return if v >= p(params, ML_V1) { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + ((v - p(params, ML_V1)) / v2).tanh())
}

#[inline]
fn ml_w_inf(v: f64, params: &Params) -> f64 {
    let v4 = p(params, ML_V4);
    if v4 == 0.0 {
        return if v >= p(params, ML_V3) { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + ((v - p(params, ML_V3)) / v4).tanh())
}

/// Rate constant of the Morris-Lecar K gate, i.e. `1 / tau_w`.
#[inline]
fn ml_w_rate(v: f64, params: &Params) -> f64 {
    let v4 = p(params, ML_V4);
    let phi = p(params, ML_PHI);
    if v4 == 0.0 {
        return phi.max(0.0);
    }
    phi * ((v - p(params, ML_V3)) / (2.0 * v4)).cosh()
}

/// Right-hand side of the ODE system for one model. Reset rules are not applied
/// here: they belong after the continuous advance, so that an RK stage never
/// observes a discontinuity.
pub fn derivatives(model: u8, s: &State, params: &Params, current: f64) -> State {
    let mut d = State::default();
    match model {
        MODEL_LIF => {
            let cm = p(params, LIF_CM);
            if cm > 0.0 {
                d.v = (-p(params, LIF_GL) * (s.v - p(params, LIF_EL)) + current) / cm;
            }
        }
        MODEL_IZHIKEVICH => {
            let i = current * p(params, IZH_ISCALE);
            d.v = 0.04 * s.v * s.v + 5.0 * s.v + 140.0 - s.w + i;
            d.w = p(params, IZH_A) * (p(params, IZH_B) * s.v - s.w);
        }
        MODEL_HODGKIN_HUXLEY => {
            let cm = p(params, HH_CM);
            let q = q10_of(params);
            d.m = q * (hh_alpha_m(s.v) * (1.0 - s.m) - hh_beta_m(s.v) * s.m);
            d.h = q * (hh_alpha_h(s.v) * (1.0 - s.h) - hh_beta_h(s.v) * s.h);
            d.n = q * (hh_alpha_n(s.v) * (1.0 - s.n) - hh_beta_n(s.v) * s.n);
            if cm > 0.0 {
                let i_na = p(params, HH_GNA) * s.m * s.m * s.m * s.h * (s.v - p(params, HH_ENA));
                let i_k = p(params, HH_GK) * s.n * s.n * s.n * s.n * (s.v - p(params, HH_EK));
                let i_l = p(params, HH_GL) * (s.v - p(params, HH_EL));
                d.v = (current - i_na - i_k - i_l) / cm;
            }
        }
        MODEL_ADEX => {
            let cm = p(params, ADEX_CM);
            let gl = p(params, ADEX_GL);
            let el = p(params, ADEX_EL);
            let delta_t = p(params, ADEX_DELTAT);
            let exp_term = if delta_t > 0.0 {
                gl * delta_t * (((s.v - p(params, ADEX_VT)) / delta_t).min(ADEX_EXP_CLAMP)).exp()
            } else {
                0.0
            };
            if cm > 0.0 {
                d.v = (-gl * (s.v - el) + exp_term - s.w + current) / cm;
            }
            let tau_w = p(params, ADEX_TAUW);
            if tau_w > 0.0 {
                d.w = (p(params, ADEX_A) * (s.v - el) - s.w) / tau_w;
            }
        }
        MODEL_MORRIS_LECAR => {
            let cm = p(params, ML_CM);
            if cm > 0.0 {
                let m_inf = ml_m_inf(s.v, params);
                let i_ca = p(params, ML_GCA) * m_inf * (s.v - p(params, ML_ECA));
                let i_k = p(params, ML_GK) * s.w * (s.v - p(params, ML_EK));
                let i_l = p(params, ML_GL) * (s.v - p(params, ML_EL));
                d.v = (current - i_ca - i_k - i_l) / cm;
            }
            d.w = ml_w_rate(s.v, params) * (ml_w_inf(s.v, params) - s.w);
        }
        _ => {}
    }
    d
}

fn advance_explicit(model: u8, s: &mut State, params: &Params, current: f64, dt: f64, order: u32) {
    match order {
        INTEGRATOR_RK2 => {
            let k1 = derivatives(model, s, params, current);
            let mid = s.offset(&k1, dt * 0.5);
            let k2 = derivatives(model, &mid, params, current);
            *s = s.offset(&k2, dt);
        }
        INTEGRATOR_RK4 => {
            let k1 = derivatives(model, s, params, current);
            let s2 = s.offset(&k1, dt * 0.5);
            let k2 = derivatives(model, &s2, params, current);
            let s3 = s.offset(&k2, dt * 0.5);
            let k3 = derivatives(model, &s3, params, current);
            let s4 = s.offset(&k3, dt);
            let k4 = derivatives(model, &s4, params, current);
            let sixth = dt / 6.0;
            s.v += sixth * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);
            s.w += sixth * (k1.w + 2.0 * k2.w + 2.0 * k3.w + k4.w);
            s.m += sixth * (k1.m + 2.0 * k2.m + 2.0 * k3.m + k4.m);
            s.h += sixth * (k1.h + 2.0 * k2.h + 2.0 * k3.h + k4.h);
            s.n += sixth * (k1.n + 2.0 * k2.n + 2.0 * k3.n + k4.n);
        }
        _ => {
            let k1 = derivatives(model, s, params, current);
            *s = s.offset(&k1, dt);
        }
    }
}

/// Analytic relaxation of `x` towards `target` with rate `lambda` over `dt`.
/// Falls back to a forward step when the rate is degenerate.
#[inline]
fn relax(x: f64, target: f64, lambda: f64, dt: f64) -> f64 {
    if lambda > 0.0 {
        let decay = (-lambda * dt).exp();
        target + (x - target) * decay
    } else {
        x
    }
}

fn advance_exponential(model: u8, s: &mut State, params: &Params, current: f64, dt: f64) {
    match model {
        MODEL_LIF => {
            let cm = p(params, LIF_CM);
            let gl = p(params, LIF_GL);
            if cm <= 0.0 {
                return;
            }
            if gl > 0.0 {
                let v_inf = p(params, LIF_EL) + current / gl;
                s.v = relax(s.v, v_inf, gl / cm, dt);
            } else {
                s.v += dt * current / cm;
            }
        }
        MODEL_IZHIKEVICH => {
            // The v equation is quadratic with no linear leak to solve exactly, so
            // it takes a forward step; u is linear in v and is solved analytically.
            let d = derivatives(model, s, params, current);
            let a = p(params, IZH_A);
            let target = p(params, IZH_B) * s.v;
            s.v += dt * d.v;
            s.w = relax(s.w, target, a, dt);
        }
        MODEL_HODGKIN_HUXLEY => {
            let cm = p(params, HH_CM);
            let q = q10_of(params);
            let v = s.v;
            let (am, bm) = (q * hh_alpha_m(v), q * hh_beta_m(v));
            let (ah, bh) = (q * hh_alpha_h(v), q * hh_beta_h(v));
            let (an, bn) = (q * hh_alpha_n(v), q * hh_beta_n(v));
            // Gates advance first; the conductances the voltage step sees are then
            // the end-of-step ones, which is the standard exponential-Euler HH
            // scheme and is what makes it stable at dt = 0.1 ms.
            s.m = relax_gate(s.m, am, bm, dt);
            s.h = relax_gate(s.h, ah, bh, dt);
            s.n = relax_gate(s.n, an, bn, dt);
            if cm <= 0.0 {
                return;
            }
            let g_na = p(params, HH_GNA) * s.m * s.m * s.m * s.h;
            let g_k = p(params, HH_GK) * s.n * s.n * s.n * s.n;
            let g_l = p(params, HH_GL);
            let g_total = g_na + g_k + g_l;
            if g_total > 0.0 {
                let drive = (g_na * p(params, HH_ENA)
                    + g_k * p(params, HH_EK)
                    + g_l * p(params, HH_EL)
                    + current)
                    / g_total;
                s.v = relax(s.v, drive, g_total / cm, dt);
            } else {
                s.v += dt * current / cm;
            }
        }
        MODEL_ADEX => {
            let cm = p(params, ADEX_CM);
            let gl = p(params, ADEX_GL);
            let el = p(params, ADEX_EL);
            let delta_t = p(params, ADEX_DELTAT);
            let tau_w = p(params, ADEX_TAUW);
            let exp_term = if delta_t > 0.0 {
                gl * delta_t * (((s.v - p(params, ADEX_VT)) / delta_t).min(ADEX_EXP_CLAMP)).exp()
            } else {
                0.0
            };
            let w0 = s.w;
            let v0 = s.v;
            if cm > 0.0 {
                if gl > 0.0 {
                    let v_inf = el + (exp_term - w0 + current) / gl;
                    s.v = relax(s.v, v_inf, gl / cm, dt);
                } else {
                    s.v += dt * (exp_term - w0 + current) / cm;
                }
            }
            if tau_w > 0.0 {
                // The adaptation target must be read from the voltage at the START
                // of the step. On a spike upstroke the clamped exponential drives v
                // to ~1e19 before the threshold comparison and reset run, so using
                // the post-update voltage here poisons w with an astronomical value
                // that the subsequent reset cannot undo — w then diverges and drags
                // v to physically impossible potentials within a few spikes.
                s.w = relax(w0, p(params, ADEX_A) * (v0 - el), 1.0 / tau_w, dt);
            }
        }
        MODEL_MORRIS_LECAR => {
            let cm = p(params, ML_CM);
            let v = s.v;
            s.w = relax(s.w, ml_w_inf(v, params), ml_w_rate(v, params), dt);
            if cm <= 0.0 {
                return;
            }
            let g_ca = p(params, ML_GCA) * ml_m_inf(v, params);
            let g_k = p(params, ML_GK) * s.w;
            let g_l = p(params, ML_GL);
            let g_total = g_ca + g_k + g_l;
            if g_total > 0.0 {
                let drive = (g_ca * p(params, ML_ECA)
                    + g_k * p(params, ML_EK)
                    + g_l * p(params, ML_EL)
                    + current)
                    / g_total;
                s.v = relax(s.v, drive, g_total / cm, dt);
            } else {
                s.v += dt * current / cm;
            }
        }
        _ => {}
    }
}

#[inline]
fn relax_gate(x: f64, alpha: f64, beta: f64, dt: f64) -> f64 {
    let sum = alpha + beta;
    if sum > 0.0 {
        let inf = alpha / sum;
        inf + (x - inf) * (-sum * dt).exp()
    } else {
        x
    }
}

fn advance(model: u8, s: &mut State, params: &Params, current: f64, dt: f64, integrator: u32) {
    match integrator {
        INTEGRATOR_EULER | INTEGRATOR_RK2 | INTEGRATOR_RK4 => {
            advance_explicit(model, s, params, current, dt, integrator)
        }
        // Exponential Euler is the default: any unrecognised code lands here
        // rather than producing an unintegrated neuron.
        _ => advance_exponential(model, s, params, current, dt),
    }
}

/// Advance one neuron by `dt` and apply its reset rule.
///
/// `t_now` is simulation time at the start of the step; the spike, if any, is
/// timestamped `t_now + dt`. `refractory_until` is read and written in place.
/// Returns true when the neuron emitted a spike on this step.
pub fn step_neuron(
    model: u8,
    s: &mut State,
    params: &Params,
    current: f64,
    dt: f64,
    integrator: u32,
    t_now: f64,
    refractory_until: &mut f32,
) -> bool {
    let t_after = t_now + dt;
    match model {
        MODEL_LIF => {
            let v_reset = p(params, LIF_VRESET);
            if t_now < *refractory_until as f64 {
                s.v = v_reset;
                return false;
            }
            advance(model, s, params, current, dt, integrator);
            if !s.v.is_finite() {
                s.v = v_reset;
                return false;
            }
            if s.v >= p(params, LIF_VTHRESH) {
                s.v = v_reset;
                *refractory_until = (t_after + p(params, LIF_TREFRACT).max(0.0)) as f32;
                return true;
            }
            false
        }
        MODEL_IZHIKEVICH => {
            let c = p(params, IZH_C);
            advance(model, s, params, current, dt, integrator);
            if !s.v.is_finite() || !s.w.is_finite() {
                s.v = c;
                s.w = p(params, IZH_B) * c;
                return false;
            }
            if s.v >= p(params, IZH_VPEAK) {
                s.v = c;
                s.w += p(params, IZH_D);
                return true;
            }
            false
        }
        MODEL_HODGKIN_HUXLEY => {
            let v_prev = s.v;
            advance(model, s, params, current, dt, integrator);
            if !s.v.is_finite() {
                *s = resting_state(model, params);
                return false;
            }
            s.m = s.m.clamp(0.0, 1.0);
            s.h = s.h.clamp(0.0, 1.0);
            s.n = s.n.clamp(0.0, 1.0);
            let v_detect = p(params, HH_VDETECT);
            v_prev < v_detect && s.v >= v_detect
        }
        MODEL_ADEX => {
            let v_reset = p(params, ADEX_VRESET);
            if t_now < *refractory_until as f64 {
                // v is clamped through the refractory period but the adaptation
                // current keeps relaxing, which is what produces AdEx bursting.
                s.v = v_reset;
                let tau_w = p(params, ADEX_TAUW);
                if tau_w > 0.0 {
                    s.w = relax(
                        s.w,
                        p(params, ADEX_A) * (v_reset - p(params, ADEX_EL)),
                        1.0 / tau_w,
                        dt,
                    );
                }
                return false;
            }
            advance(model, s, params, current, dt, integrator);
            if !s.v.is_finite() || !s.w.is_finite() {
                s.v = v_reset;
                s.w = 0.0;
                return false;
            }
            if s.v >= p(params, ADEX_VPEAK) {
                s.v = v_reset;
                s.w += p(params, ADEX_B);
                *refractory_until = (t_after + p(params, ADEX_TREFRACT).max(0.0)) as f32;
                return true;
            }
            false
        }
        MODEL_MORRIS_LECAR => {
            let v_prev = s.v;
            advance(model, s, params, current, dt, integrator);
            if !s.v.is_finite() || !s.w.is_finite() {
                *s = resting_state(model, params);
                return false;
            }
            s.w = s.w.clamp(0.0, 1.0);
            let v_detect = p(params, ML_VDETECT);
            v_prev < v_detect && s.v >= v_detect
        }
        _ => false,
    }
}

/// State a neuron is reset to: the resting potential from
/// `restingPotential()` in defaults.ts, with every gate at its steady-state
/// value there. Starting the HH gates anywhere else fires a spurious onset
/// spike in the first few milliseconds.
pub fn resting_state(model: u8, params: &Params) -> State {
    let mut s = State::default();
    match model {
        MODEL_LIF => s.v = p(params, LIF_EL),
        MODEL_IZHIKEVICH => {
            s.v = p(params, IZH_C);
            s.w = p(params, IZH_B) * s.v;
        }
        MODEL_HODGKIN_HUXLEY => {
            s.v = HH_REST;
            let (am, bm) = (hh_alpha_m(s.v), hh_beta_m(s.v));
            let (ah, bh) = (hh_alpha_h(s.v), hh_beta_h(s.v));
            let (an, bn) = (hh_alpha_n(s.v), hh_beta_n(s.v));
            s.m = if am + bm > 0.0 { am / (am + bm) } else { 0.0 };
            s.h = if ah + bh > 0.0 { ah / (ah + bh) } else { 0.0 };
            s.n = if an + bn > 0.0 { an / (an + bn) } else { 0.0 };
        }
        MODEL_ADEX => s.v = p(params, ADEX_EL),
        MODEL_MORRIS_LECAR => {
            s.v = ML_REST;
            s.w = ml_w_inf(s.v, params);
        }
        _ => {}
    }
    s
}
