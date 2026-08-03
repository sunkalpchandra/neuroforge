/**
 * Brian2 exporter.
 *
 * Emits a complete, runnable Python script: one `NeuronGroup` per membrane
 * model with the model's own differential equations in Brian2 syntax, one
 * `Synapses` object per (source group, target group, conductance channel,
 * plasticity rule, short-term-plasticity flag, stochastic-release flag)
 * partition, per-neuron parameter arrays, delays, stimuli rebuilt as
 * `TimedArray`s, and spike/state monitors.
 *
 * Synaptic conductances live on the postsynaptic group, one pair of state
 * variables per distinct set of receptor kinetics, which is what lets several
 * `Synapses` objects feed the same target without fighting over a summed
 * variable.
 */

import type { Circuit, Neuron, NeuronModelKind, PlasticityKind } from '@neuroforge/shared';

import {
  allEqual,
  bannerLines,
  effectiveNoise,
  finite,
  groupBy,
  hhSteadyState,
  indexCircuit,
  modelLabel,
  pyDocstring,
  pyFloat,
  pyFloatList,
  pyInt,
  pyIntList,
  pyStimulusPayload,
  pyStr,
} from './common';
import type { ExportChannel, ExportCircuit, ExportGroup, ExportSynapse } from './common';

const MAX_RECORDED_PER_GROUP = 8;

/* ------------------------------------------------------------------ */
/* Assignment helpers                                                   */
/* ------------------------------------------------------------------ */

function assign(lines: string[], target: string, field: string, values: number[], unit: string): void {
  if (values.length === 0) return;
  if (allEqual(values)) {
    lines.push(`${target}.${field} = ${pyFloat(values[0])}${unit}`);
    return;
  }
  lines.push(`${target}.${field} = np.array(${pyFloatList(values)})${unit}`);
}

function groupNeurons(model: ExportCircuit, group: ExportGroup): Neuron[] {
  return model.neurons.slice(group.offset, group.offset + group.size);
}

function column(neurons: readonly Neuron[], pick: (n: Neuron) => number): number[] {
  return neurons.map(pick);
}

/* ------------------------------------------------------------------ */
/* Equations                                                            */
/* ------------------------------------------------------------------ */

function channelEquations(channels: readonly ExportChannel[]): string[] {
  const lines: string[] = [];
  for (const channel of channels) {
    const c = channel.name;
    const eRev = `${pyFloat(channel.kinetics.eRev)}*mV`;
    const drive =
      channel.kinetics.mgBlock > 0
        ? `(${eRev} - v)/(1 + ${pyFloat(channel.kinetics.mgBlock)}*exp(-0.062*v/mV)/3.57)`
        : `(${eRev} - v)`;
    if (channel.kernel.form === 'dual') {
      lines.push(`I_${c} = (gd_${c} - gr_${c}) * ${drive} : amp`);
      lines.push(`dgr_${c}/dt = -gr_${c}/(${pyFloat(channel.kernel.tauRise)}*ms) : siemens`);
      lines.push(`dgd_${c}/dt = -gd_${c}/(${pyFloat(channel.kernel.tauDecay)}*ms) : siemens`);
    } else {
      lines.push(`I_${c} = gd_${c} * ${drive} : amp`);
      lines.push(`dgd_${c}/dt = -gd_${c}/(${pyFloat(channel.kernel.tauDecay)}*ms) : siemens`);
    }
  }
  return lines;
}

interface GroupPlan {
  group: ExportGroup;
  neurons: Neuron[];
  channels: ExportChannel[];
  noise: number[];
  hasNoise: boolean;
  stimulusEntries: string[];
  method: string;
}

function brianMethod(kind: NeuronModelKind, integrator: Circuit['simulation']['integrator'], hasNoise: boolean): string {
  if (hasNoise) return integrator === 'rk2' ? 'heun' : 'euler';
  switch (integrator) {
    case 'euler':
      return 'euler';
    case 'rk2':
      return 'heun';
    case 'rk4':
      return 'rk4';
    case 'exponential-euler':
      // The Izhikevich voltage equation is quadratic, so it is not
      // conditionally linear and exponential Euler cannot be applied.
      return kind === 'izhikevich' ? 'euler' : 'exponential_euler';
  }
}

function membraneEquations(plan: GroupPlan): { eqs: string[]; threshold: string; reset: string; refractory: string } {
  const { group, channels } = plan;
  const synapticTerms = channels.map((c) => `I_${c.name}`);
  const shared: string[] = [];
  shared.push(
    synapticTerms.length > 0 ? `I_syn = ${synapticTerms.join(' + ')} : amp` : 'I_syn : amp',
  );
  shared.push(...channelEquations(channels));
  shared.push(plan.stimulusEntries.length > 0 ? `I_ext = stim_${group.name}(t, i) : amp` : 'I_ext : amp');
  shared.push('I_bias : amp');
  if (plan.hasNoise) shared.push('sigma_i : amp');

  const drive = 'I_syn + I_ext + I_bias';
  const noiseTerm = plan.hasNoise ? ' + (sigma_i*noise_scale/C_m)*xi' : '';

  switch (group.kind) {
    case 'lif':
      return {
        eqs: [
          `dv/dt = (g_L*(E_L - v) + ${drive})/C_m${noiseTerm} : volt (unless refractory)`,
          ...shared,
          'C_m : farad',
          'g_L : siemens',
          'E_L : volt',
          'V_th : volt',
          'V_reset : volt',
          't_ref : second',
        ],
        threshold: 'v > V_th',
        reset: 'v = V_reset',
        refractory: 't_ref',
      };
    case 'adex':
      return {
        eqs: [
          `dv/dt = (g_L*(E_L - v) + g_L*Delta_T*exp((v - V_T)/Delta_T) - w_adapt + ${drive})/C_m${noiseTerm} : volt (unless refractory)`,
          'dw_adapt/dt = (a_adapt*(v - E_L) - w_adapt)/tau_w : amp',
          ...shared,
          'C_m : farad',
          'g_L : siemens',
          'E_L : volt',
          'Delta_T : volt',
          'V_T : volt',
          'V_peak : volt',
          'V_reset : volt',
          'a_adapt : siemens',
          'b_adapt : amp',
          'tau_w : second',
          't_ref : second',
        ],
        threshold: 'v > V_peak',
        reset: 'v = V_reset\nw_adapt += b_adapt',
        refractory: 't_ref',
      };
    case 'izhikevich': {
      const izhNoise = plan.hasNoise ? ' + (izh_iscale*(sigma_i/pA)*noise_scale)*xi*mV/ms' : '';
      return {
        eqs: [
          `dv/dt = (0.04*(v/mV)**2 + 5*(v/mV) + 140 - u + izh_iscale*(${drive})/pA)*mV/ms${izhNoise} : volt`,
          'du/dt = izh_a*(izh_b*(v/mV) - u)/ms : 1',
          ...shared,
          'izh_a : 1',
          'izh_b : 1',
          'izh_d : 1',
          'izh_iscale : 1',
          'V_c : volt',
          'V_peak : volt',
        ],
        threshold: 'v > V_peak',
        reset: 'v = V_c\nu += izh_d',
        refractory: '',
      };
    }
    case 'hodgkin-huxley':
      return {
        eqs: [
          `dv/dt = (g_Na*m**3*h*(E_Na - v) + g_K*n_gate**4*(E_K - v) + g_L*(E_L - v) + ${drive})/C_m${noiseTerm} : volt`,
          'dm/dt = q10*(alpha_m*(1 - m) - beta_m*m) : 1',
          'dh/dt = q10*(alpha_h*(1 - h) - beta_h*h) : 1',
          'dn_gate/dt = q10*(alpha_n*(1 - n_gate) - beta_n*n_gate) : 1',
          // exprel keeps the alpha rates finite at their removable singularities.
          'alpha_m = (1.0/exprel(-(v/mV + 40.0)/10.0))/ms : Hz',
          'beta_m = (4.0*exp(-(v/mV + 65.0)/18.0))/ms : Hz',
          'alpha_h = (0.07*exp(-(v/mV + 65.0)/20.0))/ms : Hz',
          'beta_h = (1.0/(1.0 + exp(-(v/mV + 35.0)/10.0)))/ms : Hz',
          'alpha_n = (0.1/exprel(-(v/mV + 55.0)/10.0))/ms : Hz',
          'beta_n = (0.125*exp(-(v/mV + 65.0)/80.0))/ms : Hz',
          ...shared,
          'C_m : farad',
          'g_Na : siemens',
          'g_K : siemens',
          'g_L : siemens',
          'E_Na : volt',
          'E_K : volt',
          'E_L : volt',
          'V_detect : volt',
          'q10 : 1',
        ],
        threshold: 'v > V_detect',
        reset: '',
        refractory: 'v > V_detect',
      };
    case 'morris-lecar':
      return {
        eqs: [
          `dv/dt = (g_Ca*m_inf*(E_Ca - v) + g_K*w_ml*(E_K - v) + g_L*(E_L - v) + ${drive})/C_m${noiseTerm} : volt`,
          'dw_ml/dt = phi_ml*(w_inf - w_ml)*cosh((v - V3)/(2*V4)) : 1',
          'm_inf = 0.5*(1 + tanh((v - V1)/V2)) : 1',
          'w_inf = 0.5*(1 + tanh((v - V3)/V4)) : 1',
          ...shared,
          'C_m : farad',
          'g_Ca : siemens',
          'g_K : siemens',
          'g_L : siemens',
          'E_Ca : volt',
          'E_K : volt',
          'E_L : volt',
          'V1 : volt',
          'V2 : volt',
          'V3 : volt',
          'V4 : volt',
          'phi_ml : Hz',
          'V_detect : volt',
        ],
        threshold: 'v > V_detect',
        reset: '',
        refractory: 'v > V_detect',
      };
  }
}

function parameterAssignments(plan: GroupPlan): string[] {
  const lines: string[] = [];
  const g = plan.group.name;
  const ns = plan.neurons;
  assign(lines, g, 'I_bias', column(ns, (n) => finite(n.bias, 0)), '*pA');
  if (plan.hasNoise) assign(lines, g, 'sigma_i', plan.noise, '*pA');

  switch (plan.group.kind) {
    case 'lif': {
      assign(lines, g, 'C_m', column(ns, (n) => (n.params.kind === 'lif' ? n.params.cm : 200)), '*pF');
      assign(lines, g, 'g_L', column(ns, (n) => (n.params.kind === 'lif' ? n.params.gL : 10)), '*nS');
      assign(lines, g, 'E_L', column(ns, (n) => (n.params.kind === 'lif' ? n.params.eL : -70)), '*mV');
      assign(lines, g, 'V_th', column(ns, (n) => (n.params.kind === 'lif' ? n.params.vThresh : -50)), '*mV');
      assign(lines, g, 'V_reset', column(ns, (n) => (n.params.kind === 'lif' ? n.params.vReset : -58)), '*mV');
      assign(lines, g, 't_ref', column(ns, (n) => (n.params.kind === 'lif' ? n.params.tRefract : 0)), '*ms');
      assign(lines, g, 'v', column(ns, (n) => (n.params.kind === 'lif' ? n.params.eL : -70)), '*mV');
      break;
    }
    case 'adex': {
      const p = (n: Neuron) => (n.params.kind === 'adex' ? n.params : null);
      assign(lines, g, 'C_m', column(ns, (n) => p(n)?.cm ?? 281), '*pF');
      assign(lines, g, 'g_L', column(ns, (n) => p(n)?.gL ?? 30), '*nS');
      assign(lines, g, 'E_L', column(ns, (n) => p(n)?.eL ?? -70.6), '*mV');
      assign(lines, g, 'Delta_T', column(ns, (n) => p(n)?.deltaT ?? 2), '*mV');
      assign(lines, g, 'V_T', column(ns, (n) => p(n)?.vT ?? -50.4), '*mV');
      assign(lines, g, 'V_peak', column(ns, (n) => p(n)?.vPeak ?? 20), '*mV');
      assign(lines, g, 'V_reset', column(ns, (n) => p(n)?.vReset ?? -70.6), '*mV');
      assign(lines, g, 'a_adapt', column(ns, (n) => p(n)?.a ?? 4), '*nS');
      assign(lines, g, 'b_adapt', column(ns, (n) => p(n)?.b ?? 80.5), '*pA');
      assign(lines, g, 'tau_w', column(ns, (n) => p(n)?.tauW ?? 144), '*ms');
      assign(lines, g, 't_ref', column(ns, (n) => p(n)?.tRefract ?? 0), '*ms');
      assign(lines, g, 'v', column(ns, (n) => p(n)?.eL ?? -70.6), '*mV');
      lines.push(`${g}.w_adapt = 0.0*pA`);
      break;
    }
    case 'izhikevich': {
      const p = (n: Neuron) => (n.params.kind === 'izhikevich' ? n.params : null);
      assign(lines, g, 'izh_a', column(ns, (n) => p(n)?.a ?? 0.02), '');
      assign(lines, g, 'izh_b', column(ns, (n) => p(n)?.b ?? 0.2), '');
      assign(lines, g, 'izh_d', column(ns, (n) => p(n)?.d ?? 8), '');
      assign(lines, g, 'izh_iscale', column(ns, (n) => p(n)?.iScale ?? 0.04), '');
      assign(lines, g, 'V_c', column(ns, (n) => p(n)?.c ?? -65), '*mV');
      assign(lines, g, 'V_peak', column(ns, (n) => p(n)?.vPeak ?? 30), '*mV');
      assign(lines, g, 'v', column(ns, (n) => p(n)?.c ?? -65), '*mV');
      assign(
        lines,
        g,
        'u',
        column(ns, (n) => {
          const params = p(n);
          return params ? params.b * params.c : -13;
        }),
        '',
      );
      break;
    }
    case 'hodgkin-huxley': {
      const p = (n: Neuron) => (n.params.kind === 'hodgkin-huxley' ? n.params : null);
      assign(lines, g, 'C_m', column(ns, (n) => p(n)?.cm ?? 100), '*pF');
      assign(lines, g, 'g_Na', column(ns, (n) => p(n)?.gNa ?? 12000), '*nS');
      assign(lines, g, 'g_K', column(ns, (n) => p(n)?.gK ?? 3600), '*nS');
      assign(lines, g, 'g_L', column(ns, (n) => p(n)?.gL ?? 30), '*nS');
      assign(lines, g, 'E_Na', column(ns, (n) => p(n)?.eNa ?? 50), '*mV');
      assign(lines, g, 'E_K', column(ns, (n) => p(n)?.eK ?? -77), '*mV');
      assign(lines, g, 'E_L', column(ns, (n) => p(n)?.eL ?? -54.4), '*mV');
      assign(lines, g, 'V_detect', column(ns, (n) => p(n)?.vDetect ?? -20), '*mV');
      assign(lines, g, 'q10', column(ns, (n) => p(n)?.q10 ?? 1), '');
      const steady = hhSteadyState(-65);
      lines.push(`${g}.v = -65.0*mV`);
      lines.push(`${g}.m = ${pyFloat(steady.m)}`);
      lines.push(`${g}.h = ${pyFloat(steady.h)}`);
      lines.push(`${g}.n_gate = ${pyFloat(steady.n)}`);
      break;
    }
    case 'morris-lecar': {
      const p = (n: Neuron) => (n.params.kind === 'morris-lecar' ? n.params : null);
      assign(lines, g, 'C_m', column(ns, (n) => p(n)?.cm ?? 20), '*pF');
      assign(lines, g, 'g_Ca', column(ns, (n) => p(n)?.gCa ?? 4.4), '*nS');
      assign(lines, g, 'g_K', column(ns, (n) => p(n)?.gK ?? 8), '*nS');
      assign(lines, g, 'g_L', column(ns, (n) => p(n)?.gL ?? 2), '*nS');
      assign(lines, g, 'E_Ca', column(ns, (n) => p(n)?.eCa ?? 120), '*mV');
      assign(lines, g, 'E_K', column(ns, (n) => p(n)?.eK ?? -84), '*mV');
      assign(lines, g, 'E_L', column(ns, (n) => p(n)?.eL ?? -60), '*mV');
      assign(lines, g, 'V1', column(ns, (n) => p(n)?.v1 ?? -1.2), '*mV');
      assign(lines, g, 'V2', column(ns, (n) => p(n)?.v2 ?? 18), '*mV');
      assign(lines, g, 'V3', column(ns, (n) => p(n)?.v3 ?? 2), '*mV');
      assign(lines, g, 'V4', column(ns, (n) => p(n)?.v4 ?? 30), '*mV');
      assign(lines, g, 'phi_ml', column(ns, (n) => (p(n)?.phi ?? 0.04) * 1000), '*Hz');
      assign(lines, g, 'V_detect', column(ns, (n) => p(n)?.vDetect ?? 0), '*mV');
      const rest = -60.9;
      lines.push(`${g}.v = ${pyFloat(rest)}*mV`);
      assign(
        lines,
        g,
        'w_ml',
        column(ns, (n) => {
          const params = p(n);
          if (!params) return 0;
          return 0.5 * (1 + Math.tanh((rest - params.v3) / Math.max(params.v4, 1e-6)));
        }),
        '',
      );
      break;
    }
  }
  return lines;
}

/* ------------------------------------------------------------------ */
/* Synapse partitions                                                   */
/* ------------------------------------------------------------------ */

interface Partition {
  key: string;
  name: string;
  sourceGroup: ExportGroup;
  targetGroup: ExportGroup;
  channel: ExportChannel;
  plasticity: PlasticityKind;
  useStp: boolean;
  useRelease: boolean;
  synapses: ExportSynapse[];
}

function partitionSynapses(model: ExportCircuit): Partition[] {
  const keyed = groupBy(model.synapses, (s) => {
    const plasticity = model.plasticityEnabled ? s.plasticity.kind : 'static';
    const stp = s.stp.enabled ? 'stp' : 'nostp';
    const rel = s.releaseProbability < 1 ? 'rel' : 'norel';
    return `${model.groupOf[s.pre]}|${model.groupOf[s.post]}|${s.channel}|${plasticity}|${stp}|${rel}`;
  });
  const partitions: Partition[] = [];
  for (const [key, synapses] of keyed) {
    const first = synapses[0];
    const plasticity: PlasticityKind = model.plasticityEnabled ? first.plasticity.kind : 'static';
    partitions.push({
      key,
      name: `syn_${partitions.length}`,
      sourceGroup: model.groups[model.groupOf[first.pre]],
      targetGroup: model.groups[model.groupOf[first.post]],
      channel: model.channels[first.channel],
      plasticity,
      useStp: first.stp.enabled,
      useRelease: first.releaseProbability < 1,
      synapses,
    });
  }
  return partitions;
}

interface SynapseCode {
  model: string[];
  onPre: string[];
  onPost: string[];
  /** Assignments applied after `connect`. */
  assignments: (name: string) => string[];
  /** Brian2 integration method for the clock-driven synaptic equations. */
  method: string | null;
}

function plasticityCode(partition: Partition): SynapseCode {
  const model: string[] = [];
  const onPre: string[] = [];
  const onPost: string[] = [];
  const values = partition.synapses;

  switch (partition.plasticity) {
    case 'static':
      break;
    case 'stdp':
      model.push(
        'dapre/dt = -apre/tau_plus : siemens (event-driven)',
        'dapost/dt = -apost/tau_minus : siemens (event-driven)',
        'A_plus : siemens',
        'A_minus : siemens',
        'lr : 1',
        'w_min : siemens',
        'w_max : siemens',
        'tau_plus : second',
        'tau_minus : second',
      );
      onPre.push('w_syn = clip(w_syn - lr*apost, w_min, w_max)', 'apre += A_plus');
      onPost.push('w_syn = clip(w_syn + lr*apre, w_min, w_max)', 'apost += A_minus');
      break;
    case 'triplet-stdp':
      model.push(
        'dr1/dt = -r1/tau_plus : 1 (event-driven)',
        'dr2/dt = -r2/tau_x : 1 (event-driven)',
        'do1/dt = -o1/tau_minus : 1 (event-driven)',
        'do2/dt = -o2/tau_y : 1 (event-driven)',
        'A_plus : siemens',
        'A_minus : siemens',
        'lr : 1',
        'w_min : siemens',
        'w_max : siemens',
        'tau_plus : second',
        'tau_minus : second',
        'tau_x : second',
        'tau_y : second',
      );
      // Pfister & Gerstner (2006). The document exposes a single potentiation
      // and a single depression amplitude, so the pair and triplet terms of the
      // rule share them.
      onPre.push('w_syn = clip(w_syn - lr*o1*(A_minus + A_minus*r2), w_min, w_max)', 'r1 += 1', 'r2 += 1');
      onPost.push('w_syn = clip(w_syn + lr*r1*(A_plus + A_plus*o2), w_min, w_max)', 'o1 += 1', 'o2 += 1');
      break;
    case 'hebbian':
      model.push(
        'dapre/dt = -apre/tau_plus : 1 (event-driven)',
        'dapost/dt = -apost/tau_minus : 1 (event-driven)',
        'A_plus : siemens',
        'lr : 1',
        'w_min : siemens',
        'w_max : siemens',
        'tau_plus : second',
        'tau_minus : second',
      );
      // Symmetric correlation rule: coincidence in either order potentiates.
      onPre.push('w_syn = clip(w_syn + lr*A_plus*apost, w_min, w_max)', 'apre += 1');
      onPost.push('w_syn = clip(w_syn + lr*A_plus*apre, w_min, w_max)', 'apost += 1');
      break;
    case 'oja':
      model.push(
        'dapre/dt = -apre/tau_plus : 1 (event-driven)',
        'dapost/dt = -apost/tau_minus : 1 (event-driven)',
        'A_plus : siemens',
        'A_decay : 1',
        'lr : 1',
        'w_min : siemens',
        'w_max : siemens',
        'tau_plus : second',
        'tau_minus : second',
      );
      // dw = lr*(x*y - y^2*w): Hebbian growth minus a forgetting term that is
      // proportional to postsynaptic activity and to the weight itself.
      onPre.push('apre += 1');
      onPost.push(
        'w_syn = clip(w_syn + lr*(A_plus*apre - A_decay*apost*w_syn), w_min, w_max)',
        'apost += 1',
      );
      break;
  }

  if (partition.useStp) {
    model.push(
      'dx_stp/dt = (1 - x_stp)*r_rec : 1 (clock-driven)',
      'du_stp/dt = -u_stp*r_facil : 1 (clock-driven)',
      'U_stp : 1',
      'r_rec : Hz',
      'r_facil : Hz',
    );
  }
  if (partition.useRelease) model.push('p_rel : 1');
  model.push('w_syn : siemens');

  const delivery: string[] = [];
  if (partition.useStp) delivery.push('u_stp += U_stp*(1 - u_stp)');
  if (partition.useRelease) delivery.push('rel_ok = int(rand() < p_rel)');

  const factors = ['w_syn'];
  if (partition.useStp) factors.push('u_stp', 'x_stp');
  if (partition.useRelease) factors.push('rel_ok');
  const amount = `${pyFloat(partition.channel.kernel.norm)}*${factors.join('*')}`;
  const c = partition.channel.name;
  if (partition.channel.kernel.form === 'dual') {
    delivery.push(`gr_${c}_post += ${amount}`, `gd_${c}_post += ${amount}`);
  } else {
    delivery.push(`gd_${c}_post += ${amount}`);
  }
  if (partition.useStp) {
    delivery.push(partition.useRelease ? 'x_stp = x_stp - u_stp*x_stp*rel_ok' : 'x_stp = x_stp - u_stp*x_stp');
  }
  onPre.push(...delivery);

  const assignments = (name: string): string[] => {
    const lines: string[] = [];
    assign(lines, name, 'w_syn', values.map((s) => s.weight), '*nS');
    assign(lines, name, 'delay', values.map((s) => s.delay), '*ms');
    if (partition.useStp) {
      assign(lines, name, 'U_stp', values.map((s) => Math.min(1, Math.max(0, finite(s.stp.u, 0.5)))), '');
      assign(
        lines,
        name,
        'r_rec',
        values.map((s) => 1000 / Math.max(finite(s.stp.tauRec, 800), 1e-3)),
        '*Hz',
      );
      // A zero facilitation constant means "no facilitation": the utilisation
      // variable must be back at rest before the next spike, which an
      // effectively infinite rate guarantees for any timestep.
      assign(
        lines,
        name,
        'r_facil',
        values.map((s) => (finite(s.stp.tauFacil, 0) > 0 ? 1000 / s.stp.tauFacil : 1e6)),
        '*Hz',
      );
      lines.push(`${name}.x_stp = 1.0`);
      lines.push(`${name}.u_stp = 0.0`);
    }
    if (partition.useRelease) {
      assign(lines, name, 'p_rel', values.map((s) => s.releaseProbability), '');
    }
    if (partition.plasticity !== 'static') {
      assign(lines, name, 'lr', values.map((s) => finite(s.plasticity.learningRate, 1)), '');
      assign(lines, name, 'w_min', values.map((s) => finite(s.plasticity.wMin, 0)), '*nS');
      assign(lines, name, 'w_max', values.map((s) => finite(s.plasticity.wMax, 4)), '*nS');
      assign(lines, name, 'tau_plus', values.map((s) => finite(s.plasticity.tauPlus, 16.8)), '*ms');
      assign(lines, name, 'tau_minus', values.map((s) => finite(s.plasticity.tauMinus, 33.7)), '*ms');
      assign(lines, name, 'A_plus', values.map((s) => finite(s.plasticity.aPlus, 0)), '*nS');
      if (partition.plasticity === 'stdp' || partition.plasticity === 'triplet-stdp') {
        assign(lines, name, 'A_minus', values.map((s) => finite(s.plasticity.aMinus, 0)), '*nS');
      }
      if (partition.plasticity === 'triplet-stdp') {
        assign(lines, name, 'tau_x', values.map((s) => finite(s.plasticity.tauX, 101)), '*ms');
        assign(lines, name, 'tau_y', values.map((s) => finite(s.plasticity.tauY, 125)), '*ms');
      }
      if (partition.plasticity === 'oja') {
        assign(lines, name, 'A_decay', values.map((s) => finite(s.plasticity.aMinus, 0)), '');
      }
    }
    return lines;
  };

  return { model, onPre, onPost, assignments, method: partition.useStp ? 'exact' : null };
}

/* ------------------------------------------------------------------ */
/* Stimuli                                                              */
/* ------------------------------------------------------------------ */

function stimulusEntries(model: ExportCircuit, group: ExportGroup): string[] {
  const entries: string[] = [];
  for (const { stimulus, targets } of model.stimuli) {
    const local = targets
      .filter((index) => index >= group.offset && index < group.offset + group.size)
      .map((index) => index - group.offset);
    if (local.length === 0) continue;
    const p = stimulus.pattern;
    entries.push(
      `    (${pyStr(p.kind)}, ${pyIntList(local)}, ${pyStimulusPayload(p)}),  # ${stimulus.name}`,
    );
  }
  return entries;
}

const STIMULUS_BUILDER = `def build_stimulus(n_steps, dt_ms, n_neurons, entries):
    """Rebuild the document's stimuli as an [n_steps, n_neurons] array of pA."""
    values = np.zeros((n_steps, n_neurons))
    t = np.arange(n_steps) * dt_ms
    for kind, targets, p in entries:
        idx = np.asarray(targets, dtype=int)
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
            hit = rng.random((n_steps, idx.size)) < (p['rate'] * dt_ms / 1000.0)
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

/* ------------------------------------------------------------------ */
/* Script assembly                                                      */
/* ------------------------------------------------------------------ */

function recordedIndices(model: ExportCircuit, group: ExportGroup): number[] {
  const probed = new Set<number>();
  for (const { target } of model.probes) {
    if (target >= group.offset && target < group.offset + group.size) probed.add(target - group.offset);
  }
  if (probed.size > 0) return [...probed].sort((a, b) => a - b);
  const count = Math.min(group.size, MAX_RECORDED_PER_GROUP);
  return Array.from({ length: count }, (_, i) => i);
}

function extraRecordings(kind: NeuronModelKind): string[] {
  switch (kind) {
    case 'izhikevich':
      return ['u'];
    case 'adex':
      return ['w_adapt'];
    case 'morris-lecar':
      return ['w_ml'];
    case 'hodgkin-huxley':
      return ['m', 'h', 'n_gate'];
    case 'lif':
      return [];
  }
}

/** Generate a complete Brian2 simulation script for a circuit. */
export function exportBrian2(circuit: Circuit): string {
  const model = indexCircuit(circuit);
  const out: string[] = [];

  out.push(pyDocstring(bannerLines(model, 'Brian2')));
  out.push('');
  out.push('from brian2 import *');
  out.push('import numpy as np');
  out.push('');
  out.push(`defaultclock.dt = ${pyFloat(model.dt)}*ms`);
  out.push(`DURATION = ${pyFloat(model.duration)}*ms`);
  out.push(`DT_MS = ${pyFloat(model.dt)}`);
  out.push('N_STEPS = int(round(float(DURATION/ms) / DT_MS))');
  out.push(`seed(${pyInt(model.seed)})`);
  out.push('');
  out.push('# White-noise currents are scaled so that one timestep injects a Gaussian');
  out.push('# current with exactly the standard deviation configured in the document.');
  out.push('noise_scale = sqrt(defaultclock.dt)');
  out.push('');

  if (model.neurons.length === 0) {
    out.push('# This circuit contains no enabled neurons.');
    out.push("print('nothing to simulate')");
    return `${out.join('\n')}\n`;
  }

  const anyStimuli = model.stimuli.length > 0;
  if (anyStimuli) {
    out.push('');
    out.push(STIMULUS_BUILDER);
    out.push('');
  }

  const partitions = partitionSynapses(model);
  const channelsByGroup = new Map<number, Set<number>>();
  for (const partition of partitions) {
    const groupIndex = model.groups.indexOf(partition.targetGroup);
    const set = channelsByGroup.get(groupIndex) ?? new Set<number>();
    set.add(partition.channel.index);
    channelsByGroup.set(groupIndex, set);
  }

  const plans: GroupPlan[] = model.groups.map((group, groupIndex) => {
    const neurons = groupNeurons(model, group);
    const noise = neurons.map((n) => effectiveNoise(n, model));
    const channelIndices = [...(channelsByGroup.get(groupIndex) ?? new Set<number>())].sort((a, b) => a - b);
    const hasNoise = noise.some((s) => s > 0);
    return {
      group,
      neurons,
      channels: channelIndices.map((i) => model.channels[i]),
      noise,
      hasNoise,
      stimulusEntries: stimulusEntries(model, group),
      method: brianMethod(group.kind, circuit.simulation.integrator, hasNoise),
    };
  });

  for (const plan of plans) {
    const g = plan.group.name;
    out.push(`# ${'='.repeat(70)}`);
    out.push(`# Population '${g}': ${plan.group.size} ${modelLabel(plan.group.kind)} neurons`);
    out.push(`# ${'='.repeat(70)}`);
    if (plan.stimulusEntries.length > 0) {
      out.push(`stim_entries_${g} = [`);
      out.push(...plan.stimulusEntries);
      out.push(']');
      out.push(
        `stim_${g} = TimedArray(build_stimulus(N_STEPS, DT_MS, ${pyInt(plan.group.size)}, stim_entries_${g})*pA, ` +
          'dt=defaultclock.dt)',
      );
    }
    const { eqs, threshold, reset, refractory } = membraneEquations(plan);
    out.push(`eqs_${g} = '''`);
    out.push(...eqs);
    out.push("'''");
    const args = [
      pyInt(plan.group.size),
      `eqs_${g}`,
      `threshold=${pyStr(threshold)}`,
    ];
    if (reset.length > 0) args.push(`reset=${pyStr(reset)}`);
    if (refractory.length > 0) args.push(`refractory=${pyStr(refractory)}`);
    args.push(`method=${pyStr(plan.method)}`, `name=${pyStr(g)}`);
    out.push(`${g} = NeuronGroup(${args.join(', ')})`);
    out.push(...parameterAssignments(plan));
    out.push('');
  }

  if (partitions.length > 0) {
    out.push(`# ${'='.repeat(70)}`);
    out.push('# Synapses');
    out.push(`# ${'='.repeat(70)}`);
    for (const partition of partitions) {
      const code = plasticityCode(partition);
      const label =
        `${partition.sourceGroup.name} -> ${partition.targetGroup.name} via ${partition.channel.receptor} ` +
        `(channel ${partition.channel.name}, ${partition.plasticity}` +
        `${partition.useStp ? ', short-term plasticity' : ''}` +
        `${partition.useRelease ? ', stochastic release' : ''})`;
      out.push(`# ${label}: ${partition.synapses.length} connection(s)`);
      const args = [partition.sourceGroup.name, partition.targetGroup.name];
      args.push(`model='''${['', ...code.model, ''].join('\n')}'''`);
      args.push(`on_pre='''${['', ...code.onPre, ''].join('\n')}'''`);
      if (code.onPost.length > 0) args.push(`on_post='''${['', ...code.onPost, ''].join('\n')}'''`);
      if (code.method) args.push(`method=${pyStr(code.method)}`);
      args.push(`name=${pyStr(partition.name)}`);
      out.push(`${partition.name} = Synapses(${args.join(', ')})`);
      const i = partition.synapses.map((s) => s.pre - partition.sourceGroup.offset);
      const j = partition.synapses.map((s) => s.post - partition.targetGroup.offset);
      out.push(`${partition.name}.connect(i=np.array(${pyIntList(i)}), j=np.array(${pyIntList(j)}))`);
      out.push(...code.assignments(partition.name));
      out.push('');
    }
  }

  out.push(`# ${'='.repeat(70)}`);
  out.push('# Monitors');
  out.push(`# ${'='.repeat(70)}`);
  const monitorNames: string[] = [];
  for (const plan of plans) {
    const g = plan.group.name;
    out.push(`spikes_${g} = SpikeMonitor(${g}, name='spikes_${g}')`);
    const recorded = recordedIndices(model, plan.group);
    const signals = ['v', ...extraRecordings(plan.group.kind)];
    out.push(
      `state_${g} = StateMonitor(${g}, [${signals.map(pyStr).join(', ')}], record=${pyIntList(recorded)}, name='state_${g}')`,
    );
    monitorNames.push(g);
  }
  out.push('');
  out.push("run(DURATION, report='text')");
  out.push('');
  out.push('# ' + '='.repeat(70));
  out.push('# Summary');
  out.push('# ' + '='.repeat(70));
  out.push('for _name, _group, _monitor in [');
  for (const name of monitorNames) out.push(`    (${pyStr(name)}, ${name}, spikes_${name}),`);
  out.push(']:');
  out.push('    _rate = _monitor.num_spikes / (len(_group) * float(DURATION/second))');
  out.push("    print('%-6s %5d neurons %8d spikes %8.2f Hz mean rate' % (_name, len(_group), _monitor.num_spikes, _rate))");

  return `${out.join('\n')}\n`;
}
