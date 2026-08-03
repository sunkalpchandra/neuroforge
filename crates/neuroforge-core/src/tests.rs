use super::*;
use crate::delay::DelayQueue;
use crate::model::{
    INTEGRATOR_EULER, INTEGRATOR_EXP_EULER, INTEGRATOR_RK2, INTEGRATOR_RK4, MODEL_ADEX,
    MODEL_HODGKIN_HUXLEY, MODEL_IZHIKEVICH, MODEL_LIF, MODEL_MORRIS_LECAR,
};
use crate::plasticity::{
    PLASTICITY_HEBBIAN, PLASTICITY_OJA, PLASTICITY_STDP, PLASTICITY_TRIPLET_STDP, Traces,
};
use crate::rng::Pcg32;
use crate::synapse::{SYNAPSE_PARAM_STRIDE, StpState, SynParams};

// Parameter sets copied verbatim from packages/shared/src/defaults.ts. If a test
// starts failing after a defaults.ts edit, these are what went stale.
const LIF: [f32; 6] = [200.0, 10.0, -70.0, -50.0, -58.0, 2.0];
const IZH_RS: [f32; 6] = [0.02, 0.2, -65.0, 8.0, 30.0, 0.04];
const HH: [f32; 9] = [
    100.0, 12000.0, 3600.0, 30.0, 50.0, -77.0, -54.4, -20.0, 1.0,
];
const ADEX: [f32; 11] = [
    281.0, 30.0, -70.6, 2.0, -50.4, 20.0, -70.6, 4.0, 80.5, 144.0, 2.0,
];
const ML: [f32; 13] = [
    20.0, 4.4, 8.0, 2.0, 120.0, -84.0, -60.0, -1.2, 18.0, 2.0, 30.0, 0.04, 0.0,
];
// DEFAULT_PLASTICITY from packages/shared/src/synapse.ts, packed by SYN_PARAM_SLOT.
const STDP_PARAMS: SynParams = [
    0.008, 0.009, 16.8, 33.7, 101.0, 125.0, 0.0, 4.0, 1.0, 0.0, 800.0, 0.0,
];

fn single(model: u8, params: &[f32], current: f32) -> Network {
    let mut net = Network::new(4, 4);
    net.resize_neurons(1);
    net.neurons.model[0] = model;
    for (i, &value) in params.iter().enumerate() {
        net.neurons.params[i] = value;
    }
    net.reset();
    net.neurons.i_ext[0] = current;
    net
}

/// Run `duration` ms one substep at a time and return the spike times.
fn spike_times(net: &mut Network, dt: f32, integrator: u32, duration: f32) -> Vec<f64> {
    let steps = (duration / dt).round() as u32;
    let mut times = Vec::new();
    for _ in 0..steps {
        if net.step(1, dt, integrator, 1.0, 0.0, false) > 0 {
            times.push(net.time());
        }
    }
    times
}

fn intervals(times: &[f64]) -> Vec<f64> {
    times.windows(2).map(|w| w[1] - w[0]).collect()
}

// ------------------------------------------------------------------ LIF

#[test]
fn lif_fires_at_the_analytic_rate() {
    // tau = cm/gL = 20 ms, V_inf = eL + I/gL = -20 mV.
    // ISI = tRefract + tau * ln((V_inf - vReset)/(V_inf - vThresh))
    //     = 2 + 20 * ln(38/30) = 6.727776 ms  ->  148.64 Hz.
    const EXPECTED_ISI: f64 = 6.727776;
    let mut net = single(MODEL_LIF, &LIF, 500.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 200.0);
    assert!(times.len() > 25, "expected sustained firing, got {times:?}");
    let isis = intervals(&times);
    let mean = isis.iter().sum::<f64>() / isis.len() as f64;
    assert!(
        (mean - EXPECTED_ISI).abs() < 0.05,
        "mean ISI {mean} deviates from analytic {EXPECTED_ISI}"
    );
    let rate = 1000.0 / mean;
    assert!((rate - 148.6375).abs() < 1.5, "rate {rate} Hz");
}

#[test]
fn lif_is_silent_below_rheobase() {
    // Rheobase is gL * (vThresh - eL) = 10 * 20 = 200 pA.
    let mut net = single(MODEL_LIF, &LIF, 150.0);
    let times = spike_times(&mut net, 0.05, INTEGRATOR_EXP_EULER, 500.0);
    assert!(times.is_empty(), "subthreshold LIF fired: {times:?}");
    assert!(net.neurons.v[0] < -50.0);
}

#[test]
fn lif_respects_the_refractory_clamp() {
    // At 5000 pA, V_inf = 430 mV, so recharging from vReset to vThresh takes only
    // 20 * ln(488/480) = 0.3306 ms. The interval is therefore the 2 ms refractory
    // period plus that, and can never fall below the refractory period however
    // hard the neuron is driven.
    const EXPECTED_ISI: f64 = 2.330585;
    let mut net = single(MODEL_LIF, &LIF, 5000.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 100.0);
    let isis = intervals(&times);
    assert!(!isis.is_empty());
    for isi in isis {
        assert!(isi >= 2.0, "fired inside the refractory period: ISI {isi}");
        assert!(
            (isi - EXPECTED_ISI).abs() < 0.05,
            "ISI {isi} deviates from analytic {EXPECTED_ISI}"
        );
    }
}

#[test]
fn all_integrators_agree_on_lif() {
    // The LIF subthreshold equation is linear, so every scheme converges to the
    // same trajectory as dt shrinks; at dt = 0.005 ms they must already agree.
    let reference = {
        let mut net = single(MODEL_LIF, &LIF, 500.0);
        spike_times(&mut net, 0.005, INTEGRATOR_EXP_EULER, 100.0)
    };
    for integrator in [INTEGRATOR_EULER, INTEGRATOR_RK2, INTEGRATOR_RK4] {
        let mut net = single(MODEL_LIF, &LIF, 500.0);
        let times = spike_times(&mut net, 0.005, integrator, 100.0);
        assert_eq!(times.len(), reference.len(), "integrator {integrator}");
        for (a, b) in times.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 0.02, "integrator {integrator}: {a} vs {b}");
        }
    }
}

// ---------------------------------------------------------- Izhikevich

#[test]
fn izhikevich_regular_spiking_reaches_the_expected_interval() {
    // Reference from an independent forward-Euler integration of
    // dv = 0.04v^2 + 5v + 140 - u + I, du = a(bv - u) at dt = 0.01 ms with
    // I = 250 pA * iScale 0.04 = 10 model units: 23 spikes per second, the
    // interval adapting from 23.15 ms to a steady 44.84 ms.
    let mut net = single(MODEL_IZHIKEVICH, &IZH_RS, 250.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EULER, 1000.0);
    assert_eq!(times.len(), 23, "spike count changed: {}", times.len());
    let isis = intervals(&times);
    assert!(
        (isis[0] - 23.15).abs() < 0.2,
        "first ISI {} != 23.15",
        isis[0]
    );
    let steady = isis[isis.len() - 1];
    assert!((steady - 44.84).abs() < 0.5, "steady ISI {steady} != 44.84");
    // Spike-frequency adaptation: the interval lengthens monotonically.
    assert!(steady > isis[0]);
}

#[test]
fn izhikevich_resets_v_and_bumps_u() {
    let mut net = single(MODEL_IZHIKEVICH, &IZH_RS, 500.0);
    let u_before = net.neurons.w[0];
    let mut fired = false;
    for _ in 0..20000 {
        if net.step(1, 0.01, INTEGRATOR_EULER, 1.0, 0.0, false) > 0 {
            fired = true;
            break;
        }
    }
    assert!(fired);
    assert!((net.neurons.v[0] - IZH_RS[2]).abs() < 1e-4, "v reset to c");
    assert!(net.neurons.w[0] > u_before + 7.0, "u incremented by d");
}

#[test]
fn izhikevich_is_silent_without_drive() {
    let mut net = single(MODEL_IZHIKEVICH, &IZH_RS, 0.0);
    let times = spike_times(&mut net, 0.05, INTEGRATOR_EXP_EULER, 500.0);
    assert!(times.is_empty(), "unstimulated Izhikevich fired");
}

// ------------------------------------------------------ Hodgkin-Huxley

#[test]
fn hodgkin_huxley_fires_for_a_suprathreshold_step() {
    // 1200 pA over the 100 pF membrane is 12 uA/cm^2 in the classic squid
    // parameterisation, comfortably above the ~6 uA/cm^2 repetitive-firing
    // threshold. An independent f64 integration gives 8 spikes in 100 ms.
    let mut net = single(MODEL_HODGKIN_HUXLEY, &HH, 1200.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 100.0);
    assert!(
        (6..=10).contains(&times.len()),
        "expected ~8 action potentials, got {}",
        times.len()
    );
    assert!(times[0] < 5.0, "first spike late: {}", times[0]);
    // A real action potential overshoots well past 0 mV.
    let mut net = single(MODEL_HODGKIN_HUXLEY, &HH, 1200.0);
    let mut peak = f32::NEG_INFINITY;
    for _ in 0..1000 {
        net.step(1, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
        peak = peak.max(net.neurons.v[0]);
    }
    assert!(peak > 20.0, "action potential peaked at only {peak} mV");
}

#[test]
fn hodgkin_huxley_stays_silent_below_threshold() {
    let mut net = single(MODEL_HODGKIN_HUXLEY, &HH, 100.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 100.0);
    assert!(times.is_empty(), "subthreshold HH fired: {times:?}");
    assert!(
        net.neurons.v[0] < -55.0,
        "v drifted to {}",
        net.neurons.v[0]
    );
}

#[test]
fn hodgkin_huxley_rests_with_gates_at_steady_state() {
    let mut net = single(MODEL_HODGKIN_HUXLEY, &HH, 0.0);
    assert!((net.neurons.v[0] - -65.0).abs() < 1e-4);
    // Classic values at -65 mV.
    assert!((net.neurons.gate_m[0] - 0.0529).abs() < 1e-3);
    assert!((net.neurons.gate_h[0] - 0.5961).abs() < 1e-3);
    assert!((net.neurons.gate_n[0] - 0.3177).abs() < 1e-3);
    // No onset transient: an unstimulated neuron must not fire.
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 50.0);
    assert!(times.is_empty(), "spurious onset spike: {times:?}");
}

#[test]
fn hodgkin_huxley_rate_singularities_are_guarded() {
    // alpha_m is singular at -40 mV and alpha_n at -55 mV. Stepping a neuron
    // parked exactly there must stay finite.
    for v in [-40.0f32, -55.0] {
        let mut net = single(MODEL_HODGKIN_HUXLEY, &HH, 0.0);
        net.neurons.v[0] = v;
        for integrator in [
            INTEGRATOR_EULER,
            INTEGRATOR_RK2,
            INTEGRATOR_RK4,
            INTEGRATOR_EXP_EULER,
        ] {
            net.neurons.v[0] = v;
            net.step(1, 0.01, integrator, 1.0, 0.0, false);
            assert!(
                net.neurons.v[0].is_finite() && net.neurons.gate_m[0].is_finite(),
                "singularity at {v} mV leaked through integrator {integrator}"
            );
        }
    }
}

// ---------------------------------------------------------------- AdEx

#[test]
fn adex_spikes_and_adapts() {
    // 800 pA is above rheobase for the Brette-Gerstner defaults; b = 80.5 pA of
    // spike-triggered adaptation stretches the interval sharply.
    let mut net = single(MODEL_ADEX, &ADEX, 800.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 500.0);
    assert!(
        times.len() >= 3,
        "expected an adapting burst, got {}",
        times.len()
    );
    let isis = intervals(&times);
    assert!(
        isis[isis.len() - 1] > isis[0] * 2.0,
        "no spike-frequency adaptation: {isis:?}"
    );
    assert!(net.neurons.w[0] > 0.0, "adaptation current never built up");
}

#[test]
fn adex_is_silent_below_rheobase() {
    let mut net = single(MODEL_ADEX, &ADEX, 100.0);
    let times = spike_times(&mut net, 0.05, INTEGRATOR_EXP_EULER, 500.0);
    assert!(times.is_empty(), "subthreshold AdEx fired: {times:?}");
    assert!(net.neurons.v[0].is_finite());
}

#[test]
fn adex_exponent_clamp_keeps_the_upstroke_finite() {
    let mut net = single(MODEL_ADEX, &ADEX, 0.0);
    // Park v far above vT so exp((v - vT)/deltaT) would overflow unclamped.
    net.neurons.v[0] = 400.0;
    let fired = net.step(1, 0.1, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert_eq!(fired, 1);
    assert!(net.neurons.v[0].is_finite());
    assert!((net.neurons.v[0] - ADEX[6]).abs() < 1e-4, "v reset to vReset");
}

// -------------------------------------------------------- Morris-Lecar

#[test]
fn morris_lecar_oscillates_above_the_hopf_bifurcation() {
    // The class II parameter set bifurcates near 94 units of drive; an
    // independent f64 integration gives 6 spikes in 500 ms at 100 with an
    // interval of 85.4 ms.
    let mut net = single(MODEL_MORRIS_LECAR, &ML, 100.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 500.0);
    assert!(
        (5..=7).contains(&times.len()),
        "expected ~6 oscillations, got {}",
        times.len()
    );
    let isis = intervals(&times);
    let last = isis[isis.len() - 1];
    assert!((last - 85.36).abs() < 5.0, "interval {last} != 85.4");
}

#[test]
fn morris_lecar_is_quiescent_below_the_bifurcation() {
    let mut net = single(MODEL_MORRIS_LECAR, &ML, 50.0);
    let times = spike_times(&mut net, 0.01, INTEGRATOR_EXP_EULER, 500.0);
    assert!(times.is_empty(), "quiescent Morris-Lecar fired: {times:?}");
    assert!((0.0..=1.0).contains(&net.neurons.w[0]));
}

// ----------------------------------------------------------- plasticity

/// Drive one synapse through a spike-pair protocol and return the final weight.
/// Spike times are converted to step indices so an accumulating clock cannot
/// straddle a boundary and fire the same spike twice.
fn stdp_pair(kind: u8, pre_at: f64, post_at: f64, dt: f64, duration: f64) -> f32 {
    let mut weight = 1.0f32;
    let mut traces = Traces::default();
    let steps = (duration / dt).round() as u32;
    let pre_step = (pre_at / dt).round() as u32;
    let post_step = (post_at / dt).round() as u32;
    for step in 0..steps {
        weight = plasticity::update(
            kind,
            weight,
            &mut traces,
            &STDP_PARAMS,
            step == pre_step,
            step == post_step,
            dt,
        );
    }
    weight
}

#[test]
fn stdp_potentiates_for_pre_before_post() {
    let w = stdp_pair(PLASTICITY_STDP, 10.0, 15.0, 0.1, 60.0);
    assert!(w > 1.0, "pre-before-post must potentiate, got {w}");
    // aPlus * exp(-5 / tauPlus) = 0.008 * exp(-5/16.8) = 0.005938.
    assert!(
        (w - 1.005938).abs() < 2e-4,
        "potentiation magnitude {} off",
        w - 1.0
    );
}

#[test]
fn stdp_depresses_for_post_before_pre() {
    let w = stdp_pair(PLASTICITY_STDP, 15.0, 10.0, 0.1, 60.0);
    assert!(w < 1.0, "post-before-pre must depress, got {w}");
    // aMinus * exp(-5 / tauMinus) = 0.009 * exp(-5/33.7) = 0.007757.
    assert!(
        (w - 0.992243).abs() < 2e-4,
        "depression magnitude {} off",
        1.0 - w
    );
}

#[test]
fn stdp_window_decays_with_the_time_constant() {
    let close = stdp_pair(PLASTICITY_STDP, 10.0, 12.0, 0.1, 60.0);
    let far = stdp_pair(PLASTICITY_STDP, 10.0, 40.0, 0.1, 80.0);
    assert!(close > far, "a wider pairing must potentiate less");
    assert!(far > 1.0);
}

#[test]
fn triplet_stdp_exceeds_pair_stdp_on_a_burst() {
    // Two post spikes bracketing a pre spike leave a slow postsynaptic trace that
    // only the triplet rule reads, so it must potentiate strictly more.
    let mut pair_w = 1.0f32;
    let mut triplet_w = 1.0f32;
    let mut pair_traces = Traces::default();
    let mut triplet_traces = Traces::default();
    let dt = 0.1;
    for step in 0..1200 {
        let pre = step == 200;
        let post = step == 100 || step == 250;
        pair_w = plasticity::update(
            PLASTICITY_STDP,
            pair_w,
            &mut pair_traces,
            &STDP_PARAMS,
            pre,
            post,
            dt,
        );
        triplet_w = plasticity::update(
            PLASTICITY_TRIPLET_STDP,
            triplet_w,
            &mut triplet_traces,
            &STDP_PARAMS,
            pre,
            post,
            dt,
        );
    }
    assert!(
        triplet_w > pair_w,
        "triplet {triplet_w} should exceed pair {pair_w}"
    );
}

#[test]
fn weights_are_clamped_to_the_configured_bounds() {
    let mut params = STDP_PARAMS;
    params[synapse::SYN_W_MIN] = 0.5;
    params[synapse::SYN_W_MAX] = 1.5;
    params[synapse::SYN_A_PLUS] = 10.0;
    params[synapse::SYN_A_MINUS] = 10.0;
    let mut traces = Traces::default();
    let mut w = 1.0f32;
    for _ in 0..50 {
        w = plasticity::update(PLASTICITY_STDP, w, &mut traces, &params, true, true, 0.1);
    }
    assert!((0.5..=1.5).contains(&w), "weight escaped its bounds: {w}");
}

#[test]
fn hebbian_grows_with_correlated_activity_and_oja_normalises() {
    let mut params = STDP_PARAMS;
    params[synapse::SYN_A_PLUS] = 0.05;
    params[synapse::SYN_W_MAX] = 100.0;

    let mut hebb = 1.0f32;
    let mut hebb_traces = Traces::default();
    let mut oja = 1.0f32;
    let mut oja_traces = Traces::default();
    for step in 0..4000 {
        let spike = step % 50 == 0;
        hebb = plasticity::update(
            PLASTICITY_HEBBIAN,
            hebb,
            &mut hebb_traces,
            &params,
            spike,
            spike,
            0.1,
        );
        oja = plasticity::update(
            PLASTICITY_OJA,
            oja,
            &mut oja_traces,
            &params,
            spike,
            spike,
            0.1,
        );
    }
    assert!(hebb > 1.0, "Hebbian did not potentiate: {hebb}");
    assert!(
        oja < hebb,
        "Oja's decay term must hold the weight below plain Hebbian: {oja} vs {hebb}"
    );
    assert!(oja.is_finite() && oja > 0.0);
}

// --------------------------------------------------------------- synapses

#[test]
fn dual_exponential_peaks_at_the_requested_amplitude() {
    // AMPA defaults from RECEPTOR_DEFAULTS: 0.4 ms rise, 2.0 ms decay.
    let dt = 0.001;
    let k = synapse::kinetics(0.4, 2.0, dt);
    assert!(!k.single);
    let (mut rise, mut decay) = (k.norm as f64, k.norm as f64);
    let mut peak = 0.0f64;
    for _ in 0..20000 {
        rise *= k.decay_rise as f64;
        decay *= k.decay_decay as f64;
        peak = peak.max(decay - rise);
    }
    assert!(
        (peak - 1.0).abs() < 1e-3,
        "unit amplitude peaked at {peak} instead of 1"
    );
}

#[test]
fn equal_time_constants_degenerate_to_a_single_exponential() {
    // The gap-junction default has tauRise == tauDecay, where the dual-exponential
    // normalisation is singular.
    let k = synapse::kinetics(0.05, 0.05, 0.01);
    assert!(k.single);
    assert!(k.norm.is_finite() && (k.norm - 1.0).abs() < 1e-6);
    let inverted = synapse::kinetics(5.0, 1.0, 0.01);
    assert!(inverted.single, "inverted taus must not produce negative g");
}

#[test]
fn magnesium_block_is_a_sigmoid_in_postsynaptic_voltage() {
    let at_rest = synapse::mg_block(-65.0, 1.0);
    let depolarised = synapse::mg_block(0.0, 1.0);
    let high = synapse::mg_block(40.0, 1.0);
    assert!(at_rest < 0.1, "NMDA should be strongly blocked at rest");
    assert!(depolarised > 0.7, "block should relieve on depolarisation");
    assert!(high > depolarised && high < 1.0);
    assert_eq!(synapse::mg_block(-65.0, 0.0), 1.0, "mgBlock 0 disables it");
}

#[test]
fn reversal_potential_sets_the_sign_of_the_current() {
    // Excitatory: eRev 0 mV above a resting -65 mV post gives inward (positive) current.
    assert!(synapse::synaptic_current(1.0, 0.0, -65.0, 0.0) > 0.0);
    // Inhibitory: eRev -70 mV below a depolarised -60 mV post gives outward current.
    assert!(synapse::synaptic_current(1.0, -70.0, -60.0, 0.0) < 0.0);
    // At the reversal potential the synapse is silent regardless of conductance.
    assert_eq!(synapse::synaptic_current(5.0, -70.0, -70.0, 0.0), 0.0);
}

#[test]
fn short_term_plasticity_depresses_a_train() {
    // Tsodyks-Markram with tauFacil 0 is pure depression: each release consumes
    // resources that recover over tauRec.
    let mut params: SynParams = [0.0; SYNAPSE_PARAM_STRIDE];
    params[synapse::SYN_STP_U] = 0.5;
    params[synapse::SYN_STP_TAU_REC] = 800.0;
    params[synapse::SYN_STP_TAU_FACIL] = 0.0;
    let mut state = StpState { r: 1.0, u: 0.5 };
    let mut amplitudes = Vec::new();
    for _ in 0..5 {
        amplitudes.push(synapse::stp_release(&mut state, &params));
        for _ in 0..100 {
            synapse::stp_relax(&mut state, &params, 0.1);
        }
    }
    for pair in amplitudes.windows(2) {
        assert!(pair[1] < pair[0], "train did not depress: {amplitudes:?}");
    }
    assert!((amplitudes[0] - 0.5).abs() < 1e-6);
}

#[test]
fn short_term_plasticity_facilitates_when_tau_facil_is_set() {
    let mut params: SynParams = [0.0; SYNAPSE_PARAM_STRIDE];
    params[synapse::SYN_STP_U] = 0.15;
    params[synapse::SYN_STP_TAU_REC] = 100.0;
    params[synapse::SYN_STP_TAU_FACIL] = 500.0;
    let mut state = StpState { r: 1.0, u: 0.15 };
    let first = synapse::stp_release(&mut state, &params);
    for _ in 0..100 {
        synapse::stp_relax(&mut state, &params, 0.1);
    }
    let second = synapse::stp_release(&mut state, &params);
    assert!(
        second > first,
        "facilitation absent: {first} then {second}"
    );
}

// ------------------------------------------------------------ delay queue

#[test]
fn delay_queue_delivers_at_the_scheduled_time() {
    let mut queue = DelayQueue::new(256, 0.25, 64);
    queue.clear(0.0);
    assert!(queue.schedule(5.0, 7, 2.5));

    let mut arrival = None;
    let mut payload = 0.0f32;
    let mut t = 0.0f64;
    for _ in 0..200 {
        queue.drain_due(t, |index, amplitude| {
            assert_eq!(index, 7);
            payload = amplitude;
            arrival = Some(t);
        });
        t += 0.1;
    }
    let arrival = arrival.expect("event never delivered");
    assert!(
        (arrival - 5.0).abs() <= 0.25,
        "delivered at {arrival}, expected 5.0 within one bucket"
    );
    assert_eq!(payload, 2.5);
    assert_eq!(queue.pending(), 0);
}

#[test]
fn delay_queue_preserves_ordering_across_a_wrap() {
    let mut queue = DelayQueue::new(8, 1.0, 4);
    queue.clear(0.0);
    // Horizon is 8 ms; schedule inside it, drain past the wrap, schedule again.
    assert!(queue.schedule(3.0, 1, 1.0));
    assert!(queue.schedule(6.0, 2, 2.0));
    let mut order = Vec::new();
    let mut t = 0.0;
    for _ in 0..40 {
        queue.drain_due(t, |i, _| order.push(i));
        if (t - 4.0).abs() < 1e-9 {
            assert!(queue.schedule(11.0, 3, 3.0));
        }
        t += 0.5;
    }
    assert_eq!(order, vec![1, 2, 3]);
}

#[test]
fn delay_queue_drops_on_bucket_overflow_without_corruption() {
    let mut queue = DelayQueue::new(16, 0.5, 4);
    queue.clear(0.0);
    for i in 0..4 {
        assert!(queue.schedule(2.0, i, 1.0), "bucket rejected event {i} early");
    }
    for i in 4..10 {
        assert!(!queue.schedule(2.0, i, 1.0), "bucket accepted overflow {i}");
    }
    assert_eq!(queue.dropped(), 6);

    let mut delivered = Vec::new();
    let mut t = 0.0;
    for _ in 0..20 {
        queue.drain_due(t, |i, _| delivered.push(i));
        t += 0.5;
    }
    assert_eq!(delivered, vec![0, 1, 2, 3], "surviving events were corrupted");
}

#[test]
fn delay_queue_clamps_beyond_the_horizon() {
    let mut queue = DelayQueue::new(8, 0.25, 4);
    queue.clear(0.0);
    assert_eq!(queue.horizon(), 2.0);
    assert!(queue.schedule(500.0, 1, 1.0));
    assert_eq!(queue.clamped(), 1);
    let mut seen = false;
    let mut t = 0.0;
    for _ in 0..40 {
        queue.drain_due(t, |_, _| seen = true);
        t += 0.1;
    }
    assert!(seen, "clamped event must still be delivered, not lost");
}

// -------------------------------------------------------------------- rng

#[test]
fn rng_is_reproducible_from_a_seed() {
    let mut a = Pcg32::new(12345);
    let first: Vec<u32> = (0..64).map(|_| a.next_u32()).collect();
    let mut b = Pcg32::new(12345);
    let second: Vec<u32> = (0..64).map(|_| b.next_u32()).collect();
    assert_eq!(first, second);

    let mut c = Pcg32::new(12346);
    let other: Vec<u32> = (0..64).map(|_| c.next_u32()).collect();
    assert_ne!(first, other, "distinct seeds must give distinct streams");

    a.reset(12345);
    let replay: Vec<u32> = (0..64).map(|_| a.next_u32()).collect();
    assert_eq!(first, replay, "reset must rewind the stream exactly");
}

#[test]
fn rng_normal_has_unit_variance_and_resets_its_spare() {
    let mut rng = Pcg32::new(7);
    const N: usize = 200_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..N {
        let x = rng.normal();
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / N as f64;
    let var = sum_sq / N as f64 - mean * mean;
    assert!(mean.abs() < 0.02, "mean {mean}");
    assert!((var - 1.0).abs() < 0.02, "variance {var}");

    // The Box-Muller spare is part of the reproducible state: reset must clear it,
    // otherwise an odd number of draws would desynchronise a replay.
    let mut a = Pcg32::new(3);
    a.normal();
    a.reset(3);
    let mut b = Pcg32::new(3);
    assert_eq!(a.normal(), b.normal());
}

#[test]
fn rng_poisson_matches_its_rate() {
    let mut rng = Pcg32::new(99);
    for lambda in [0.05, 1.0, 12.0, 64.0] {
        const N: usize = 40_000;
        let total: u64 = (0..N).map(|_| rng.poisson(lambda) as u64).sum();
        let mean = total as f64 / N as f64;
        assert!(
            (mean - lambda).abs() < 0.05 * lambda.max(1.0),
            "lambda {lambda} produced mean {mean}"
        );
    }
    assert_eq!(rng.poisson(0.0), 0);
    assert_eq!(rng.poisson(f64::NAN), 0);
}

#[test]
fn rng_uniforms_stay_in_range() {
    let mut rng = Pcg32::new(2024);
    for _ in 0..100_000 {
        let f = rng.next_f32();
        assert!((0.0..1.0).contains(&f));
        let d = rng.next_f64();
        assert!((0.0..1.0).contains(&d));
        assert!(rng.next_below(7) < 7);
    }
    assert_eq!(rng.next_below(0), 0);
}

// ------------------------------------------------------------ integration

/// Two LIF neurons wired 0 -> 1 with an AMPA synapse.
fn pair_network(delay: f32, weight: f32) -> Network {
    let mut net = Network::new(2, 1);
    net.resize_neurons(2);
    net.resize_synapses(1);
    for slot in 0..2 {
        net.neurons.model[slot] = MODEL_LIF;
        for (i, &value) in LIF.iter().enumerate() {
            net.neurons.params[slot * NEURON_PARAM_STRIDE + i] = value;
        }
    }
    net.synapses.pre[0] = 0;
    net.synapses.post[0] = 1;
    net.synapses.weight[0] = weight;
    net.synapses.delay[0] = delay;
    net.synapses.tau_rise[0] = 0.4;
    net.synapses.tau_decay[0] = 2.0;
    net.synapses.e_rev[0] = 0.0;
    net.reset();
    net
}

#[test]
fn a_spike_crosses_the_synapse_after_its_conduction_delay() {
    let mut net = pair_network(3.0, 40.0);
    net.neurons.i_ext[0] = 500.0;

    let mut presynaptic = None;
    let mut postsynaptic_onset = None;
    for _ in 0..2000 {
        net.step(1, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
        if presynaptic.is_none() && net.neurons.spike[0] != 0 {
            presynaptic = Some(net.time());
        }
        if postsynaptic_onset.is_none() && net.neurons.i_syn[1].abs() > 1e-6 {
            postsynaptic_onset = Some(net.time());
        }
    }
    let fired = presynaptic.expect("presynaptic neuron never fired");
    let arrived = postsynaptic_onset.expect("current never reached the postsynaptic neuron");
    let lag = arrived - fired;
    assert!(
        (lag - 3.0).abs() <= 0.3,
        "conduction lag {lag} ms, expected 3.0"
    );
    assert!(
        net.neurons.i_syn[1] != 0.0 || net.neurons.v[1] > -70.0,
        "postsynaptic neuron was never depolarised"
    );
}

#[test]
fn a_strong_synapse_drives_the_postsynaptic_neuron_to_fire() {
    let mut net = pair_network(1.0, 400.0);
    net.neurons.i_ext[0] = 500.0;
    net.step(4000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert!(
        net.neurons.spike_count[1] > 0,
        "postsynaptic neuron never fired despite a 400 nS synapse"
    );
    // Gain scales the whole synaptic drive: at zero gain the target falls silent.
    let mut quiet = pair_network(1.0, 400.0);
    quiet.neurons.i_ext[0] = 500.0;
    quiet.step(4000, 0.01, INTEGRATOR_EXP_EULER, 0.0, 0.0, false);
    assert_eq!(quiet.neurons.spike_count[1], 0);
    assert!(quiet.neurons.spike_count[0] > 0);
}

#[test]
fn an_inhibitory_synapse_suppresses_the_target() {
    let mut excited = pair_network(1.0, 0.0);
    excited.neurons.i_ext[0] = 500.0;
    excited.neurons.i_ext[1] = 260.0;
    excited.step(20000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    let baseline = excited.neurons.spike_count[1];
    assert!(baseline > 0, "control neuron should fire on its own");

    let mut inhibited = pair_network(1.0, 200.0);
    inhibited.synapses.e_rev[0] = -70.0;
    inhibited.synapses.tau_decay[0] = 6.0;
    inhibited.synapses.tau_rise[0] = 0.5;
    inhibited.neurons.i_ext[0] = 500.0;
    inhibited.neurons.i_ext[1] = 260.0;
    inhibited.step(20000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert!(
        inhibited.neurons.spike_count[1] < baseline,
        "GABA-A synapse did not suppress: {} vs {baseline}",
        inhibited.neurons.spike_count[1]
    );
}

#[test]
fn disabled_rows_are_excluded_from_integration() {
    let mut net = pair_network(1.0, 400.0);
    net.neurons.i_ext[0] = 500.0;
    net.neurons.enabled[1] = 0;
    net.step(2000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert!(net.neurons.spike_count[0] > 0);
    assert_eq!(net.neurons.spike_count[1], 0);
    assert_eq!(net.neurons.v[1], -70.0, "disabled neuron must not integrate");

    let mut off = pair_network(1.0, 400.0);
    off.neurons.i_ext[0] = 500.0;
    off.synapses.enabled[0] = 0;
    off.step(2000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert_eq!(off.neurons.spike_count[1], 0);
}

#[test]
fn out_of_range_endpoints_are_ignored_rather_than_trusted() {
    // `pre` and `post` are written by JavaScript straight into linear memory and
    // carry no guarantee, so a stale index must not read out of bounds.
    let mut net = pair_network(1.0, 400.0);
    net.neurons.i_ext[0] = 500.0;
    net.synapses.pre[0] = 9999;
    net.synapses.post[0] = 12345;
    let spikes = net.step(2000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert!(spikes > 0);
    assert_eq!(net.neurons.spike_count[1], 0);
}

#[test]
fn stochastic_release_thins_the_train() {
    let mut always = pair_network(1.0, 40.0);
    always.neurons.i_ext[0] = 500.0;
    always.step(10000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    let full = always.delays.pending();

    let mut sometimes = pair_network(1.0, 40.0);
    sometimes.set_seed(4242);
    sometimes.synapses.release_prob[0] = 0.25;
    sometimes.neurons.i_ext[0] = 500.0;
    sometimes.step(10000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    // Both networks fire the same presynaptic train; only release differs.
    assert_eq!(sometimes.neurons.spike_count[0], always.neurons.spike_count[0]);
    assert!(
        sometimes.neurons.v[1] <= always.neurons.v[1] || full == 0,
        "stochastic release should not exceed deterministic release"
    );
    assert!(sometimes.neurons.spike_count[0] > 5);
}

#[test]
fn the_same_seed_reproduces_a_run_exactly() {
    let run = |seed: u32| {
        let mut net = pair_network(1.0, 40.0);
        net.set_seed(seed);
        net.synapses.release_prob[0] = 0.5;
        net.neurons.i_ext[0] = 400.0;
        net.neurons.noise[0] = 60.0;
        net.neurons.noise[1] = 60.0;
        net.step(5000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 15.0, false);
        (
            net.neurons.v[0],
            net.neurons.v[1],
            net.neurons.spike_count[0],
            net.neurons.spike_count[1],
            net.spike_count(),
        )
    };
    assert_eq!(run(0xfeed), run(0xfeed), "identical seeds diverged");
    assert_ne!(
        run(0xfeed).4,
        run(0xbeef).4,
        "distinct seeds produced identical spike totals"
    );
}

#[test]
fn reset_returns_every_model_to_rest_without_touching_topology() {
    let mut net = pair_network(2.0, 40.0);
    net.neurons.model[1] = MODEL_HODGKIN_HUXLEY;
    for (i, &value) in HH.iter().enumerate() {
        net.neurons.params[NEURON_PARAM_STRIDE + i] = value;
    }
    net.neurons.i_ext[0] = 500.0;
    net.step(3000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert!(net.spike_count() > 0);
    assert!(net.time() > 0.0);

    net.reset();
    assert_eq!(net.time(), 0.0);
    assert_eq!(net.spike_count(), 0);
    assert_eq!(net.neurons.v[0], -70.0);
    assert!((net.neurons.v[1] - -65.0).abs() < 1e-4);
    assert!(net.neurons.gate_h[1] > 0.5, "HH gates restored to steady state");
    assert_eq!(net.synapses.g_decay[0], 0.0);
    assert_eq!(net.delays.pending(), 0);
    // Topology and parameters survive.
    assert_eq!(net.synapses.pre[0], 0);
    assert_eq!(net.synapses.post[0], 1);
    assert_eq!(net.synapses.weight[0], 40.0);
}

#[test]
fn plasticity_only_runs_when_it_is_enabled() {
    let mut net = pair_network(1.0, 1.0);
    net.synapses.plasticity[0] = PLASTICITY_STDP;
    for (i, &value) in STDP_PARAMS.iter().enumerate() {
        net.synapses.params[i] = value;
    }
    net.neurons.i_ext[0] = 500.0;
    net.neurons.i_ext[1] = 500.0;
    net.step(5000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert_eq!(net.synapses.weight[0], 1.0, "weight moved with plasticity off");

    net.step(5000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, true);
    assert!(
        net.synapses.weight[0] != 1.0,
        "weight frozen with plasticity on"
    );
    assert!((0.0..=4.0).contains(&net.synapses.weight[0]));
}

#[test]
fn resize_preserves_existing_rows_and_reports_reallocation() {
    let mut net = Network::new(2, 2);
    assert!(!net.resize_neurons(2), "fitting in capacity must not realloc");
    net.neurons.model[0] = MODEL_IZHIKEVICH;
    net.neurons.params[3] = 8.0;
    net.neurons.v[1] = -55.5;

    let moved = net.resize_neurons(9);
    assert!(moved, "growing past capacity must report a reallocation");
    assert_eq!(net.neuron_len(), 9);
    assert!(net.neuron_capacity() >= 9);
    assert_eq!(net.neurons.model[0], MODEL_IZHIKEVICH);
    assert_eq!(net.neurons.params[3], 8.0);
    assert_eq!(net.neurons.v[1], -55.5);
    // Fresh rows carry the documented defaults from allocateNeuronBuffers.
    assert_eq!(net.neurons.enabled[8], 1);
    assert_eq!(net.neurons.scale[8], 1.0);
    assert_eq!(net.neurons.population[8], 0xffff);
    assert_eq!(net.neurons.last_spike[8], f32::NEG_INFINITY);

    assert!(net.resize_synapses(5));
    assert_eq!(net.synapse_len(), 5);
    assert_eq!(net.synapses.stp_r[4], 1.0);
    assert_eq!(net.synapses.release_prob[4], 1.0);
    assert_eq!(net.synapses.enabled[4], 1);
}

#[test]
fn spike_log_and_drain_expose_the_same_events() {
    let mut net = single(MODEL_LIF, &LIF, 500.0);
    // 100 ms at a 6.73 ms interval, after a 10.2 ms first charge: ~14 spikes.
    net.step(10000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    let emitted = net.spike_count();
    assert!(emitted > 10, "only {emitted} spikes logged");
    assert_eq!(net.spike_log_head(), emitted);
    assert_eq!(net.pending_spikes(), emitted);

    // `drain_spikes` takes a u32 address, which only addresses anything real on
    // wasm32; on a 64-bit test host the pointer would truncate. The checked body
    // underneath it is what carries the packing logic, so that is what is
    // exercised here.
    let mut out = vec![0u32; 2 * emitted as usize];
    let drained = net.drain_into(&mut out);
    assert_eq!(drained, emitted);
    assert_eq!(net.pending_spikes(), 0);

    for k in 0..drained as usize {
        assert_eq!(out[k * 2], 0, "neuron slot");
        let time = f32::from_bits(out[k * 2 + 1]);
        assert!(time > 0.0 && time <= net.time() as f32 + 1e-3);
        assert_eq!(net.log_time[k], time, "ring and drain disagree");
    }
    assert_eq!(net.drain_into(&mut out), 0, "drained queue must stay empty");
    // The guard clauses of the exported wrapper never dereference.
    assert_eq!(net.drain_spikes(0, 16), 0, "null destination must be refused");
    assert_eq!(net.drain_spikes(64, 0), 0, "zero maximum must be refused");

    // A destination smaller than the queue truncates rather than overruns.
    net.step(5000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    let queued = net.pending_spikes();
    assert!(queued >= 4);
    let mut small = vec![0u32; 4];
    assert_eq!(net.drain_into(&mut small), 2);
    assert_eq!(net.pending_spikes(), queued - 2);
}

#[test]
fn set_time_moves_the_clock_and_rebases_the_calendar() {
    let mut net = pair_network(2.0, 40.0);
    net.set_time(1000.0);
    assert_eq!(net.time(), 1000.0);
    net.neurons.i_ext[0] = 500.0;
    net.step(2000, 0.01, INTEGRATOR_EXP_EULER, 1.0, 0.0, false);
    assert!(net.neurons.spike_count[0] > 0);
    assert!(net.neurons.last_spike[0] > 1000.0);
    net.set_time(f64::NAN);
    assert_eq!(net.time(), 0.0, "a non-finite time must fall back to zero");
}

#[test]
fn a_zero_or_negative_timestep_is_a_no_op() {
    let mut net = single(MODEL_LIF, &LIF, 500.0);
    let v = net.neurons.v[0];
    assert_eq!(net.step(10, 0.0, INTEGRATOR_EXP_EULER, 1.0, 0.0, false), 0);
    assert_eq!(net.step(10, -1.0, INTEGRATOR_EXP_EULER, 1.0, 0.0, false), 0);
    assert_eq!(net.step(0, 0.1, INTEGRATOR_EXP_EULER, 1.0, 0.0, false), 0);
    assert_eq!(net.neurons.v[0], v);
    assert_eq!(net.time(), 0.0);
}

#[test]
fn an_unknown_integrator_falls_back_to_exponential_euler() {
    let mut reference = single(MODEL_LIF, &LIF, 500.0);
    let expected = spike_times(&mut reference, 0.05, INTEGRATOR_EXP_EULER, 100.0);
    let mut net = single(MODEL_LIF, &LIF, 500.0);
    let actual = spike_times(&mut net, 0.05, 99, 100.0);
    assert_eq!(expected, actual);
}

// ------------------------------------------------------------------ graph

/// 0 -> 1 -> 2 -> 0 forms a cycle; 3 is isolated.
fn ring_plus_isolate() -> Network {
    let mut net = Network::new(4, 3);
    net.resize_neurons(4);
    net.resize_synapses(3);
    let edges = [(0u32, 1u32), (1, 2), (2, 0)];
    for (i, (a, b)) in edges.iter().enumerate() {
        net.synapses.pre[i] = *a;
        net.synapses.post[i] = *b;
    }
    net
}

#[test]
fn graph_finds_components_and_recurrent_motifs() {
    let net = ring_plus_isolate();

    let components = net.graph_components();
    assert_eq!(components.len(), 4);
    assert_eq!(components[0], components[1]);
    assert_eq!(components[1], components[2]);
    assert_ne!(components[3], components[0]);
    assert_eq!(net.graph_component_count(), 2);

    let scc = net.graph_strongly_connected_components();
    assert_eq!(scc[0], scc[1]);
    assert_eq!(scc[1], scc[2]);
    assert_ne!(scc[3], scc[0]);
    assert_eq!(net.graph_strongly_connected_count(), 2);
    assert_eq!(net.graph_recurrent_nodes(), vec![0, 1, 2]);
    assert!(net.graph_has_cycle());
}

#[test]
fn graph_reports_degrees_and_shortest_paths() {
    let net = ring_plus_isolate();
    assert_eq!(net.graph_in_degrees(), vec![1, 1, 1, 0]);
    assert_eq!(net.graph_out_degrees(), vec![1, 1, 1, 0]);
    assert_eq!(net.graph_shortest_path(0, 2), vec![0, 1, 2]);
    assert_eq!(net.graph_shortest_path(2, 1), vec![2, 0, 1]);
    assert_eq!(net.graph_shortest_path(0, 0), vec![0]);
    assert!(net.graph_shortest_path(0, 3).is_empty(), "3 is unreachable");
    assert!(net.graph_shortest_path(0, 99).is_empty(), "out of range");
}

#[test]
fn graph_clustering_matches_the_hand_computed_value() {
    // Every ring member has two neighbours that are themselves connected, so its
    // local coefficient is 1; the isolate scores 0. Mean = 0.75.
    let net = ring_plus_isolate();
    let coefficients = net.graph_clustering();
    assert_eq!(coefficients.len(), 4);
    for i in 0..3 {
        assert!((coefficients[i] - 1.0).abs() < 1e-6, "slot {i}");
    }
    assert_eq!(coefficients[3], 0.0);
    assert!((net.graph_average_clustering() - 0.75).abs() < 1e-6);
}

#[test]
fn graph_ignores_disabled_synapses_and_bad_endpoints() {
    let mut net = ring_plus_isolate();
    net.synapses.enabled[2] = 0;
    assert!(!net.graph_has_cycle(), "disabled edge must break the cycle");
    assert_eq!(net.graph_strongly_connected_count(), 4);
    assert!(net.graph_recurrent_nodes().is_empty());

    net.synapses.enabled[2] = 1;
    net.synapses.post[1] = 77;
    assert_eq!(net.graph_in_degrees(), vec![1, 1, 0, 0]);
    assert!(!net.graph_has_cycle());
}

#[test]
fn graph_handles_an_empty_network() {
    let net = Network::new(4, 4);
    assert!(net.graph_components().is_empty());
    assert_eq!(net.graph_component_count(), 0);
    assert!(!net.graph_has_cycle());
    assert_eq!(net.graph_average_clustering(), 0.0);
    assert!(net.graph_shortest_path(0, 1).is_empty());
}

#[test]
fn graph_detects_a_self_loop_as_recurrent() {
    let mut net = Network::new(2, 1);
    net.resize_neurons(2);
    net.resize_synapses(1);
    net.synapses.pre[0] = 1;
    net.synapses.post[0] = 1;
    assert!(net.graph_has_cycle());
    assert_eq!(net.graph_recurrent_nodes(), vec![1]);
}
