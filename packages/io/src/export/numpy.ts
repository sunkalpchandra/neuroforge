/**
 * Dependency-free Python exporter: a NumPy reference implementation of the
 * circuit.
 *
 * This is the target for the plain `python` format. Unlike the Brian2 and NEST
 * exports it needs nothing but NumPy, and unlike the PyTorch export it is not
 * about training — it is the readable reference the other exports can be
 * checked against. All five membrane models, per-synapse conduction delays,
 * dual-exponential conductances, short-term plasticity, stochastic release and
 * all four plasticity rules are implemented.
 */

import type { Circuit, Neuron, PlasticityKind } from '@neuroforge/shared';

import {
  bannerLines,
  delayToSteps,
  effectiveNoise,
  finite,
  indentBlock,
  indexCircuit,
  modelLabel,
  pyDocstring,
  pyFloat,
  pyFloatList,
  pyInt,
  pyIntList,
  pyStr,
  restingState,
} from './common';
import type { ExportCircuit, ExportGroup, ExportSynapse } from './common';

/** Coefficients that turn the four plasticity rules into one vectorised update. */
interface RuleCoefficients {
  prePair: number;
  preTriplet: number;
  postPair: number;
  postTriplet: number;
  ojaDecay: number;
}

function ruleCoefficients(synapse: ExportSynapse, enabled: boolean): RuleCoefficients {
  const zero: RuleCoefficients = { prePair: 0, preTriplet: 0, postPair: 0, postTriplet: 0, ojaDecay: 0 };
  const kind: PlasticityKind = enabled ? synapse.plasticity.kind : 'static';
  const p = synapse.plasticity;
  const lr = finite(p.learningRate, 1);
  const aPlus = lr * finite(p.aPlus, 0);
  const aMinus = lr * finite(p.aMinus, 0);
  switch (kind) {
    case 'static':
      return zero;
    case 'stdp':
      return { ...zero, prePair: -aMinus, postPair: aPlus };
    case 'triplet-stdp':
      return { prePair: -aMinus, preTriplet: -aMinus, postPair: aPlus, postTriplet: aPlus, ojaDecay: 0 };
    case 'hebbian':
      return { ...zero, prePair: aPlus, postPair: aPlus };
    case 'oja':
      return { ...zero, postPair: aPlus, ojaDecay: aMinus };
  }
}

function groupNeurons(model: ExportCircuit, group: ExportGroup): Neuron[] {
  return model.neurons.slice(group.offset, group.offset + group.size);
}

function arrayConst(name: string, values: readonly number[]): string {
  return `${name} = np.array(${pyFloatList(values)})`;
}

function intArrayConst(name: string, values: readonly number[]): string {
  return `${name} = np.array(${pyIntList(values)}, dtype=np.int64)`;
}

function groupConstants(model: ExportCircuit, group: ExportGroup, index: number): string[] {
  const neurons = groupNeurons(model, group);
  const g = `G${index}`;
  const lines: string[] = [];
  switch (group.kind) {
    case 'lif': {
      const p = (n: Neuron) => (n.params.kind === 'lif' ? n.params : null);
      lines.push(arrayConst(`${g}_CM`, neurons.map((n) => p(n)?.cm ?? 200)));
      lines.push(arrayConst(`${g}_GL`, neurons.map((n) => p(n)?.gL ?? 10)));
      lines.push(arrayConst(`${g}_EL`, neurons.map((n) => p(n)?.eL ?? -70)));
      lines.push(arrayConst(`${g}_VTH`, neurons.map((n) => p(n)?.vThresh ?? -50)));
      lines.push(arrayConst(`${g}_VRESET`, neurons.map((n) => p(n)?.vReset ?? -58)));
      lines.push(arrayConst(`${g}_TREF`, neurons.map((n) => Math.max(p(n)?.tRefract ?? 0, 0))));
      break;
    }
    case 'adex': {
      const p = (n: Neuron) => (n.params.kind === 'adex' ? n.params : null);
      lines.push(arrayConst(`${g}_CM`, neurons.map((n) => p(n)?.cm ?? 281)));
      lines.push(arrayConst(`${g}_GL`, neurons.map((n) => p(n)?.gL ?? 30)));
      lines.push(arrayConst(`${g}_EL`, neurons.map((n) => p(n)?.eL ?? -70.6)));
      lines.push(arrayConst(`${g}_DELTA`, neurons.map((n) => Math.max(p(n)?.deltaT ?? 2, 1e-6))));
      lines.push(arrayConst(`${g}_VT`, neurons.map((n) => p(n)?.vT ?? -50.4)));
      lines.push(arrayConst(`${g}_VPEAK`, neurons.map((n) => p(n)?.vPeak ?? 20)));
      lines.push(arrayConst(`${g}_VRESET`, neurons.map((n) => p(n)?.vReset ?? -70.6)));
      lines.push(arrayConst(`${g}_A`, neurons.map((n) => p(n)?.a ?? 4)));
      lines.push(arrayConst(`${g}_B`, neurons.map((n) => p(n)?.b ?? 80.5)));
      lines.push(arrayConst(`${g}_TAUW`, neurons.map((n) => Math.max(p(n)?.tauW ?? 144, 1e-6))));
      lines.push(arrayConst(`${g}_TREF`, neurons.map((n) => Math.max(p(n)?.tRefract ?? 0, 0))));
      break;
    }
    case 'izhikevich': {
      const p = (n: Neuron) => (n.params.kind === 'izhikevich' ? n.params : null);
      lines.push(arrayConst(`${g}_A`, neurons.map((n) => p(n)?.a ?? 0.02)));
      lines.push(arrayConst(`${g}_B`, neurons.map((n) => p(n)?.b ?? 0.2)));
      lines.push(arrayConst(`${g}_C`, neurons.map((n) => p(n)?.c ?? -65)));
      lines.push(arrayConst(`${g}_D`, neurons.map((n) => p(n)?.d ?? 8)));
      lines.push(arrayConst(`${g}_VPEAK`, neurons.map((n) => p(n)?.vPeak ?? 30)));
      lines.push(arrayConst(`${g}_ISCALE`, neurons.map((n) => p(n)?.iScale ?? 0.04)));
      break;
    }
    case 'hodgkin-huxley': {
      const p = (n: Neuron) => (n.params.kind === 'hodgkin-huxley' ? n.params : null);
      lines.push(arrayConst(`${g}_CM`, neurons.map((n) => p(n)?.cm ?? 100)));
      lines.push(arrayConst(`${g}_GNA`, neurons.map((n) => p(n)?.gNa ?? 12000)));
      lines.push(arrayConst(`${g}_GK`, neurons.map((n) => p(n)?.gK ?? 3600)));
      lines.push(arrayConst(`${g}_GL`, neurons.map((n) => p(n)?.gL ?? 30)));
      lines.push(arrayConst(`${g}_ENA`, neurons.map((n) => p(n)?.eNa ?? 50)));
      lines.push(arrayConst(`${g}_EK`, neurons.map((n) => p(n)?.eK ?? -77)));
      lines.push(arrayConst(`${g}_EL`, neurons.map((n) => p(n)?.eL ?? -54.4)));
      lines.push(arrayConst(`${g}_VDETECT`, neurons.map((n) => p(n)?.vDetect ?? -20)));
      lines.push(arrayConst(`${g}_Q10`, neurons.map((n) => p(n)?.q10 ?? 1)));
      break;
    }
    case 'morris-lecar': {
      const p = (n: Neuron) => (n.params.kind === 'morris-lecar' ? n.params : null);
      lines.push(arrayConst(`${g}_CM`, neurons.map((n) => p(n)?.cm ?? 20)));
      lines.push(arrayConst(`${g}_GCA`, neurons.map((n) => p(n)?.gCa ?? 4.4)));
      lines.push(arrayConst(`${g}_GK`, neurons.map((n) => p(n)?.gK ?? 8)));
      lines.push(arrayConst(`${g}_GL`, neurons.map((n) => p(n)?.gL ?? 2)));
      lines.push(arrayConst(`${g}_ECA`, neurons.map((n) => p(n)?.eCa ?? 120)));
      lines.push(arrayConst(`${g}_EK`, neurons.map((n) => p(n)?.eK ?? -84)));
      lines.push(arrayConst(`${g}_EL`, neurons.map((n) => p(n)?.eL ?? -60)));
      lines.push(arrayConst(`${g}_V1`, neurons.map((n) => p(n)?.v1 ?? -1.2)));
      lines.push(arrayConst(`${g}_V2`, neurons.map((n) => Math.max(p(n)?.v2 ?? 18, 1e-6))));
      lines.push(arrayConst(`${g}_V3`, neurons.map((n) => p(n)?.v3 ?? 2)));
      lines.push(arrayConst(`${g}_V4`, neurons.map((n) => Math.max(p(n)?.v4 ?? 30, 1e-6))));
      lines.push(arrayConst(`${g}_PHI`, neurons.map((n) => p(n)?.phi ?? 0.04)));
      lines.push(arrayConst(`${g}_VDETECT`, neurons.map((n) => p(n)?.vDetect ?? 0)));
      break;
    }
  }
  return lines;
}

function groupUpdate(group: ExportGroup, index: number): string[] {
  const g = `G${index}`;
  const slice = `${pyInt(group.offset)}:${pyInt(group.offset + group.size)}`;
  const head = [
    `# ${modelLabel(group.kind)}`,
    `vg = v[${slice}]`,
    `wg = w[${slice}]`,
    `ig = i_total[${slice}]`,
  ];
  switch (group.kind) {
    case 'lif':
      return [
        ...head,
        `free = refractory[${slice}] <= 0.0`,
        `v_free = np.where(free, vg + (${g}_GL * (${g}_EL - vg) + ig) / ${g}_CM * DT, vg)`,
        `fired = v_free > ${g}_VTH`,
        `v[${slice}] = np.where(fired, ${g}_VRESET, v_free)`,
        `refractory[${slice}] = np.where(fired, ${g}_TREF, np.maximum(refractory[${slice}] - DT, 0.0))`,
        `spikes[${slice}] = fired`,
      ];
    case 'adex':
      return [
        ...head,
        `free = refractory[${slice}] <= 0.0`,
        // Clamped before exp() so a diverging trajectory spikes rather than overflowing.
        `expo = ${g}_GL * ${g}_DELTA * np.exp(np.minimum((vg - ${g}_VT) / ${g}_DELTA, 30.0))`,
        `dv = (${g}_GL * (${g}_EL - vg) + expo - wg + ig) / ${g}_CM`,
        `v_free = np.where(free, vg + dv * DT, vg)`,
        `w_free = wg + ((${g}_A * (vg - ${g}_EL) - wg) / ${g}_TAUW) * DT`,
        `fired = v_free > ${g}_VPEAK`,
        `v[${slice}] = np.where(fired, ${g}_VRESET, v_free)`,
        `w[${slice}] = w_free + np.where(fired, ${g}_B, 0.0)`,
        `refractory[${slice}] = np.where(fired, ${g}_TREF, np.maximum(refractory[${slice}] - DT, 0.0))`,
        `spikes[${slice}] = fired`,
      ];
    case 'izhikevich':
      return [
        ...head,
        `dv = 0.04 * vg * vg + 5.0 * vg + 140.0 - wg + ig * ${g}_ISCALE`,
        `v_free = vg + dv * DT`,
        `w_free = wg + ${g}_A * (${g}_B * vg - wg) * DT`,
        `fired = v_free > ${g}_VPEAK`,
        `v[${slice}] = np.where(fired, ${g}_C, v_free)`,
        `w[${slice}] = w_free + np.where(fired, ${g}_D, 0.0)`,
        `spikes[${slice}] = fired`,
      ];
    case 'hodgkin-huxley':
      return [
        ...head,
        `mg_ = gate_m[${slice}]`,
        `hg_ = gate_h[${slice}]`,
        `ng_ = gate_n[${slice}]`,
        `alpha_m = 1.0 / _exprel(-(vg + 40.0) / 10.0)`,
        `beta_m = 4.0 * np.exp(-(vg + 65.0) / 18.0)`,
        `alpha_h = 0.07 * np.exp(-(vg + 65.0) / 20.0)`,
        `beta_h = 1.0 / (1.0 + np.exp(-(vg + 35.0) / 10.0))`,
        `alpha_n = 0.1 / _exprel(-(vg + 55.0) / 10.0)`,
        `beta_n = 0.125 * np.exp(-(vg + 65.0) / 80.0)`,
        `gate_m[${slice}] = mg_ + ${g}_Q10 * (alpha_m * (1.0 - mg_) - beta_m * mg_) * DT`,
        `gate_h[${slice}] = hg_ + ${g}_Q10 * (alpha_h * (1.0 - hg_) - beta_h * hg_) * DT`,
        `gate_n[${slice}] = ng_ + ${g}_Q10 * (alpha_n * (1.0 - ng_) - beta_n * ng_) * DT`,
        `i_ion = (${g}_GNA * mg_ ** 3 * hg_ * (${g}_ENA - vg) + ${g}_GK * ng_ ** 4 * (${g}_EK - vg)` +
          ` + ${g}_GL * (${g}_EL - vg))`,
        `v_free = vg + (i_ion + ig) / ${g}_CM * DT`,
        `v[${slice}] = v_free`,
        // Only the rising edge counts, so one action potential is one spike.
        `crossing = v_free > ${g}_VDETECT`,
        `spikes[${slice}] = crossing & ~above[${slice}]`,
        `above[${slice}] = crossing`,
      ];
    case 'morris-lecar':
      return [
        ...head,
        `m_inf = 0.5 * (1.0 + np.tanh((vg - ${g}_V1) / ${g}_V2))`,
        `w_inf = 0.5 * (1.0 + np.tanh((vg - ${g}_V3) / ${g}_V4))`,
        `lam = ${g}_PHI * np.cosh((vg - ${g}_V3) / (2.0 * ${g}_V4))`,
        `i_ion = (${g}_GCA * m_inf * (${g}_ECA - vg) + ${g}_GK * wg * (${g}_EK - vg)` +
          ` + ${g}_GL * (${g}_EL - vg))`,
        `v_free = vg + (i_ion + ig) / ${g}_CM * DT`,
        `v[${slice}] = v_free`,
        `w[${slice}] = wg + lam * (w_inf - wg) * DT`,
        `crossing = v_free > ${g}_VDETECT`,
        `spikes[${slice}] = crossing & ~above[${slice}]`,
        `above[${slice}] = crossing`,
      ];
  }
}

function stimulusTable(model: ExportCircuit): string[] {
  const rows: string[] = [];
  for (const { stimulus, targets } of model.stimuli) {
    const p = stimulus.pattern;
    let payload: string;
    switch (p.kind) {
      case 'constant':
        payload = `{'amplitude': ${pyFloat(p.amplitude)}}`;
        break;
      case 'step':
        payload = `{'amplitude': ${pyFloat(p.amplitude)}, 'start': ${pyFloat(p.start)}, 'duration': ${pyFloat(p.duration)}}`;
        break;
      case 'pulse-train':
        payload =
          `{'amplitude': ${pyFloat(p.amplitude)}, 'frequency': ${pyFloat(p.frequency)}, ` +
          `'width': ${pyFloat(p.width)}, 'start': ${pyFloat(p.start)}}`;
        break;
      case 'sine':
        payload = `{'amplitude': ${pyFloat(p.amplitude)}, 'frequency': ${pyFloat(p.frequency)}, 'offset': ${pyFloat(p.offset)}}`;
        break;
      case 'poisson':
        payload = `{'rate': ${pyFloat(p.rate)}, 'amplitude': ${pyFloat(p.amplitude)}, 'seed': ${pyInt(p.seed)}}`;
        break;
      case 'ramp':
        payload =
          `{'from': ${pyFloat(p.from)}, 'to': ${pyFloat(p.to)}, ` +
          `'start': ${pyFloat(p.start)}, 'duration': ${pyFloat(p.duration)}}`;
        break;
    }
    rows.push(`    (${pyStr(p.kind)}, ${pyIntList(targets)}, ${payload}),  # ${stimulus.name}`);
  }
  return rows;
}

const PRELUDE = `def _exprel(x):
    """(exp(x) - 1) / x, finite at the removable singularity x = 0."""
    safe = np.where(np.abs(x) < 1e-9, 1.0, x)
    return np.where(np.abs(x) < 1e-9, 1.0 + x / 2.0, np.expm1(safe) / safe)


def build_stimulus(n_steps):
    """The document's stimuli as an [n_steps, N] injected current in pA."""
    values = np.zeros((n_steps, N))
    t = np.arange(n_steps) * DT
    for kind, targets, p in STIMULI:
        idx = np.asarray(targets, dtype=np.int64)
        if kind == 'constant':
            wave = np.full((n_steps, 1), p['amplitude'])
        elif kind == 'step':
            active = (t >= p['start']) & (t < p['start'] + p['duration'])
            wave = np.where(active, p['amplitude'], 0.0)[:, None]
        elif kind == 'pulse-train':
            period = 1000.0 / p['frequency']
            phase = np.where(t >= p['start'], np.mod(t - p['start'], period), period)
            active = (t >= p['start']) & (phase < p['width'])
            wave = np.where(active, p['amplitude'], 0.0)[:, None]
        elif kind == 'sine':
            wave = (p['amplitude'] * np.sin(2.0 * np.pi * p['frequency'] * t / 1000.0) + p['offset'])[:, None]
        elif kind == 'poisson':
            rng = np.random.default_rng(p['seed'])
            hit = rng.random((n_steps, idx.size)) < (p['rate'] * DT / 1000.0)
            wave = np.where(hit, p['amplitude'], 0.0)
        elif kind == 'ramp':
            span = max(p['duration'], 1e-9)
            frac = np.clip((t - p['start']) / span, 0.0, 1.0)
            active = (t >= p['start']) & (t < p['start'] + p['duration'])
            wave = np.where(active, p['from'] + (p['to'] - p['from']) * frac, 0.0)[:, None]
        else:
            raise ValueError('unknown stimulus pattern: ' + kind)
        values[:, idx] += wave
    return values`;

/** Generate a self-contained NumPy simulation of the circuit. */
export function exportNumpy(circuit: Circuit): string {
  const model = indexCircuit(circuit);
  const n = model.neurons.length;
  const channels = model.channels;
  const synapses = model.synapses;
  const delays = synapses.map((s) => delayToSteps(s.delay, model.dt));
  const historyLength = Math.max(2, delays.reduce((max, d) => Math.max(max, d), 1) + 1);
  const rest = model.neurons.map((neuron) => restingState(neuron.params));
  const noise = model.neurons.map((neuron) => effectiveNoise(neuron, model));
  const plastic =
    model.plasticityEnabled && synapses.some((s) => s.plasticity.kind !== 'static');
  const stochastic = synapses.some((s) => s.releaseProbability < 1);
  const coefficients = synapses.map((s) => ruleCoefficients(s, model.plasticityEnabled));

  const out: string[] = [];
  out.push(pyDocstring(bannerLines(model, 'NumPy reference implementation')));
  out.push('');
  out.push('import numpy as np');
  out.push('');
  out.push(`N = ${pyInt(n)}`);
  out.push(`S = ${pyInt(synapses.length)}`);
  out.push(`C = ${pyInt(channels.length)}`);
  out.push(`DT = ${pyFloat(model.dt)}`);
  out.push(`STEPS = ${pyInt(Math.max(1, Math.round(model.duration / model.dt)))}`);
  out.push(`SEED = ${pyInt(Math.max(1, model.seed))}`);
  out.push(`HISTORY = ${pyInt(historyLength)}`);
  out.push(`PLASTICITY = ${plastic ? 'True' : 'False'}`);
  out.push('');
  out.push('NEURON_IDS = [');
  for (const neuron of model.neurons) out.push(`    ${pyStr(neuron.id)},`);
  out.push(']');
  out.push('');
  out.push(arrayConst('BIAS', model.neurons.map((neuron) => finite(neuron.bias, 0))));
  out.push(arrayConst('SIGMA', noise));
  out.push(arrayConst('V_REST', rest.map((r) => r.v)));
  out.push(arrayConst('W_REST', rest.map((r) => r.w)));
  out.push(arrayConst('M_REST', rest.map((r) => r.m)));
  out.push(arrayConst('H_REST', rest.map((r) => r.h)));
  out.push(arrayConst('N_REST', rest.map((r) => r.n)));
  out.push('');
  out.push('# Synaptic channels: one entry per distinct set of receptor kinetics.');
  out.push(arrayConst('CHANNEL_EREV', channels.map((c) => c.kinetics.eRev)));
  out.push(arrayConst('CHANNEL_MG', channels.map((c) => c.kinetics.mgBlock)));
  out.push(arrayConst('CHANNEL_NORM', channels.map((c) => c.kernel.norm)));
  out.push(
    `CHANNEL_DUAL = np.array(${pyIntList(channels.map((c) => (c.kernel.form === 'dual' ? 1 : 0)))}, dtype=bool)`,
  );
  out.push(
    arrayConst(
      'RISE_DECAY',
      channels.map((c) => (c.kernel.form === 'dual' ? Math.exp(-model.dt / c.kernel.tauRise) : 0)),
    ),
  );
  out.push(arrayConst('DECAY_DECAY', channels.map((c) => Math.exp(-model.dt / c.kernel.tauDecay))));
  out.push('');
  out.push('# Connectivity.');
  out.push(intArrayConst('SYN_PRE', synapses.map((s) => s.pre)));
  out.push(intArrayConst('SYN_POST', synapses.map((s) => s.post)));
  out.push(intArrayConst('SYN_CHANNEL', synapses.map((s) => s.channel)));
  out.push(intArrayConst('SYN_DELAY', delays));
  out.push(arrayConst('SYN_WEIGHT0', synapses.map((s) => s.weight)));
  out.push(arrayConst('SYN_RELEASE', synapses.map((s) => s.releaseProbability)));
  out.push('');
  out.push('# Tsodyks-Markram short-term plasticity. A synapse without it uses U = 1,');
  out.push('# instant recovery and instant utilisation decay, which makes the update a no-op.');
  out.push(arrayConst('STP_U0', synapses.map((s) => (s.stp.enabled ? finite(s.stp.u, 0.5) : 1))));
  out.push(
    arrayConst(
      'STP_X_RECOVER',
      synapses.map((s) =>
        s.stp.enabled ? 1 - Math.exp(-model.dt / Math.max(finite(s.stp.tauRec, 800), 1e-6)) : 1,
      ),
    ),
  );
  out.push(
    arrayConst(
      'STP_U_DECAY',
      synapses.map((s) =>
        s.stp.enabled && finite(s.stp.tauFacil, 0) > 0 ? Math.exp(-model.dt / s.stp.tauFacil) : 0,
      ),
    ),
  );
  if (plastic) {
    out.push('');
    out.push('# Plasticity, expressed as per-synapse coefficients on four eligibility traces.');
    out.push('# The learning rate is already folded into each coefficient.');
    out.push(arrayConst('COEF_PRE_PAIR', coefficients.map((c) => c.prePair)));
    out.push(arrayConst('COEF_PRE_TRIPLET', coefficients.map((c) => c.preTriplet)));
    out.push(arrayConst('COEF_POST_PAIR', coefficients.map((c) => c.postPair)));
    out.push(arrayConst('COEF_POST_TRIPLET', coefficients.map((c) => c.postTriplet)));
    out.push(arrayConst('COEF_OJA', coefficients.map((c) => c.ojaDecay)));
    out.push(arrayConst('W_MIN', synapses.map((s) => finite(s.plasticity.wMin, 0))));
    out.push(arrayConst('W_MAX', synapses.map((s) => finite(s.plasticity.wMax, 4))));
    out.push(
      arrayConst(
        'DEC_PLUS',
        synapses.map((s) => Math.exp(-model.dt / Math.max(finite(s.plasticity.tauPlus, 16.8), 1e-6))),
      ),
    );
    out.push(
      arrayConst(
        'DEC_MINUS',
        synapses.map((s) => Math.exp(-model.dt / Math.max(finite(s.plasticity.tauMinus, 33.7), 1e-6))),
      ),
    );
    out.push(
      arrayConst(
        'DEC_X',
        synapses.map((s) => Math.exp(-model.dt / Math.max(finite(s.plasticity.tauX, 101), 1e-6))),
      ),
    );
    out.push(
      arrayConst(
        'DEC_Y',
        synapses.map((s) => Math.exp(-model.dt / Math.max(finite(s.plasticity.tauY, 125), 1e-6))),
      ),
    );
  }
  out.push('');
  model.groups.forEach((group, index) => {
    out.push(`# ${modelLabel(group.kind)}: neurons [${group.offset}:${group.offset + group.size})`);
    out.push(...groupConstants(model, group, index));
  });
  out.push('');
  out.push('STIMULI = [');
  out.push(...stimulusTable(model));
  out.push(']');
  out.push('');
  out.push('');
  out.push(PRELUDE);
  out.push('');
  out.push('');
  out.push('def simulate(steps=STEPS, record_voltage=True):');
  out.push('    """Integrate the circuit. Returns (spikes[steps, N], voltage[steps, N] or None)."""');
  out.push('    rng = np.random.default_rng(SEED)');
  out.push('    v = V_REST.copy()');
  out.push('    w = W_REST.copy()');
  out.push('    gate_m = M_REST.copy()');
  out.push('    gate_h = H_REST.copy()');
  out.push('    gate_n = N_REST.copy()');
  out.push('    refractory = np.zeros(N)');
  out.push('    above = np.zeros(N, dtype=bool)');
  out.push('    g_rise = np.zeros((C, N))');
  out.push('    g_decay = np.zeros((C, N))');
  out.push('    history = np.zeros((HISTORY, N))');
  out.push('    cursor = 0');
  out.push('    weights = SYN_WEIGHT0.copy()');
  out.push('    stp_x = np.ones(S)');
  out.push('    stp_u = np.zeros(S)');
  if (plastic) {
    out.push('    pre_fast = np.zeros(S)');
    out.push('    pre_slow = np.zeros(S)');
    out.push('    post_fast = np.zeros(S)');
    out.push('    post_slow = np.zeros(S)');
  }
  out.push('    stimulus = build_stimulus(steps)');
  out.push('    spike_log = np.zeros((steps, N), dtype=np.uint8)');
  out.push('    voltage_log = np.zeros((steps, N)) if record_voltage else None');
  out.push('    spikes = np.zeros(N, dtype=bool)');
  out.push('');
  out.push('    for step in range(steps):');
  if (channels.length > 0 && synapses.length > 0) {
    out.push('        # Deliver the spikes that left the presynaptic soma SYN_DELAY steps ago.');
    out.push('        arrived = history[(cursor - SYN_DELAY) % HISTORY, SYN_PRE]');
    out.push('        stp_x = stp_x + (1.0 - stp_x) * STP_X_RECOVER');
    out.push('        stp_u = stp_u * STP_U_DECAY');
    out.push('        utilisation = stp_u + STP_U0 * (1.0 - stp_u)');
    if (stochastic) {
      out.push('        released = (rng.random(S) < SYN_RELEASE).astype(np.float64)');
    } else {
      out.push('        released = 1.0');
    }
    out.push('        efficacy = arrived * utilisation * stp_x * released');
    out.push('        stp_u = np.where(arrived > 0.0, utilisation, stp_u)');
    out.push('        stp_x = stp_x - np.where(arrived > 0.0, utilisation * stp_x * released, 0.0)');
    out.push('        arrivals = np.zeros((C, N))');
    out.push('        np.add.at(arrivals, (SYN_CHANNEL, SYN_POST), efficacy * weights)');
    out.push('        arrivals *= CHANNEL_NORM[:, None]');
    out.push('        g_rise = g_rise * RISE_DECAY[:, None] + arrivals');
    out.push('        g_decay = g_decay * DECAY_DECAY[:, None] + arrivals');
    out.push('        conductance = np.where(CHANNEL_DUAL[:, None], g_decay - g_rise, g_decay)');
    out.push('        # The Mg block is inert wherever its strength is zero.');
    out.push('        block = 1.0 + CHANNEL_MG[:, None] * np.exp(-0.062 * v[None, :]) / 3.57');
    out.push('        i_syn = np.sum(conductance * (CHANNEL_EREV[:, None] - v[None, :]) / block, axis=0)');
  } else {
    out.push('        i_syn = np.zeros(N)');
  }
  out.push('        i_total = i_syn + stimulus[step] + BIAS');
  if (noise.some((s) => s > 0)) {
    out.push('        i_total = i_total + SIGMA * rng.standard_normal(N)');
  }
  out.push('');
  model.groups.forEach((group, index) => {
    for (const line of groupUpdate(group, index)) out.push(indentBlock(line, 8));
    out.push('');
  });
  out.push('        history[cursor] = spikes');
  out.push('        cursor = (cursor + 1) % HISTORY');
  out.push('        spike_log[step] = spikes');
  out.push('        if record_voltage:');
  out.push('            voltage_log[step] = v');
  if (plastic) {
    out.push('');
    out.push('        # Trace-based plasticity. Traces decay first, the weight update reads');
    out.push('        # them before this step\'s spikes are added, which is what makes the');
    out.push('        # pair rule causal and the triplet rule use the pre-spike trace values.');
    out.push('        pre_fast *= DEC_PLUS');
    out.push('        pre_slow *= DEC_X');
    out.push('        post_fast *= DEC_MINUS');
    out.push('        post_slow *= DEC_Y');
    out.push('        post_fired = spikes[SYN_POST].astype(np.float64)');
    out.push('        delta = arrived * (COEF_PRE_PAIR + COEF_PRE_TRIPLET * pre_slow) * post_fast');
    out.push('        delta += post_fired * (COEF_POST_PAIR + COEF_POST_TRIPLET * post_slow) * pre_fast');
    out.push('        delta -= post_fired * COEF_OJA * post_fast * weights');
    out.push('        weights = np.clip(weights + delta, W_MIN, W_MAX)');
    out.push('        pre_fast += arrived');
    out.push('        pre_slow += arrived');
    out.push('        post_fast += post_fired');
    out.push('        post_slow += post_fired');
  }
  out.push('');
  out.push('    return spike_log, voltage_log');
  out.push('');
  out.push('');
  out.push("if __name__ == '__main__':");
  out.push('    spike_log, voltage_log = simulate()');
  out.push('    seconds = STEPS * DT / 1000.0');
  out.push('    counts = spike_log.sum(axis=0)');
  out.push(
    "    print('%d neurons, %d synapses, %d steps' % (N, S, STEPS))",
  );
  out.push(
    "    print('%d spikes, %.2f Hz mean rate' % (counts.sum(), counts.sum() / max(N * seconds, 1e-9)))",
  );
  out.push('    if N > 0:');
  out.push("        print('voltage range: %.2f .. %.2f mV' % (voltage_log.min(), voltage_log.max()))");

  return `${out.join('\n')}\n`;
}
