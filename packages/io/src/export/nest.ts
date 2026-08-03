/**
 * PyNEST exporter.
 *
 * Model mapping:
 *   LIF            -> iaf_cond_exp   (conductance-based, matching our synapses)
 *   AdEx           -> aeif_cond_exp
 *   Izhikevich     -> izhikevich     (delta synapses; weights become voltage jumps)
 *   Hodgkin-Huxley -> hh_psc_alpha   (current-based; weights become peak currents)
 *   Morris-Lecar   -> no built-in NEST model; emitted as a commented note
 *
 * NEST's unit conventions are not ours. Capacitance is pF and conductance nS as
 * here, but the built-in models differ in how a synaptic event is delivered:
 * `iaf_cond_exp`/`aeif_cond_exp` take a conductance in nS and pick the receptor
 * from the sign of the weight, `hh_psc_alpha` takes a peak current in pA, and
 * `izhikevich` takes an instantaneous voltage jump in mV and folds the input
 * current scale into its equation. Every one of those conversions is done here
 * rather than copying raw numbers across.
 */

import type { Circuit, NeuronModelKind, PlasticityKind } from '@neuroforge/shared';

import {
  bannerLines,
  effectiveNoise,
  finite,
  groupBy,
  indexCircuit,
  modelLabel,
  pyComment,
  pyDocstring,
  pyFloat,
  pyFloatList,
  pyInt,
  pyIntList,
  pyStr,
  restingOf,
} from './common';
import type { ExportChannel, ExportCircuit, ExportGroup, ExportSynapse } from './common';

const NEST_MODEL: Record<NeuronModelKind, string | null> = {
  lif: 'iaf_cond_exp',
  izhikevich: 'izhikevich',
  'hodgkin-huxley': 'hh_psc_alpha',
  adex: 'aeif_cond_exp',
  'morris-lecar': null,
};

const RECORDABLES: Record<NeuronModelKind, string[]> = {
  lif: ['V_m', 'g_ex', 'g_in'],
  izhikevich: ['V_m', 'U_m'],
  'hodgkin-huxley': ['V_m'],
  adex: ['V_m', 'g_ex', 'g_in', 'w'],
  'morris-lecar': ['V_m'],
};

/** NEST synapse models that reproduce our plasticity rules. */
const NEST_SYNAPSE: Record<PlasticityKind, string> = {
  static: 'static_synapse',
  stdp: 'stdp_synapse',
  'triplet-stdp': 'stdp_triplet_synapse',
  hebbian: 'static_synapse',
  oja: 'static_synapse',
};

const RULE = `# ${'='.repeat(70)}`;

interface ChannelSummary {
  tauDecay: number;
  eRev: number;
  count: number;
  distinct: number;
}

function summariseChannels(entries: readonly { channel: ExportChannel; count: number }[]): ChannelSummary {
  let total = 0;
  let tau = 0;
  let eRev = 0;
  for (const entry of entries) {
    total += entry.count;
    tau += entry.channel.kernel.tauDecay * entry.count;
    eRev += entry.channel.kinetics.eRev * entry.count;
  }
  if (total === 0) return { tauDecay: 2, eRev: 0, count: 0, distinct: 0 };
  return { tauDecay: tau / total, eRev: eRev / total, count: total, distinct: entries.length };
}

interface GroupPlan {
  group: ExportGroup;
  nestModel: string | null;
  excitatory: ChannelSummary;
  inhibitory: ChannelSummary;
}

function planGroups(model: ExportCircuit): GroupPlan[] {
  return model.groups.map((group, groupIndex) => {
    const excitatory = new Map<number, { channel: ExportChannel; count: number }>();
    const inhibitory = new Map<number, { channel: ExportChannel; count: number }>();
    for (const synapse of model.synapses) {
      if (model.groupOf[synapse.post] !== groupIndex) continue;
      const channel = model.channels[synapse.channel];
      const bucket = channel.excitatory ? excitatory : inhibitory;
      const entry = bucket.get(channel.index);
      if (entry) entry.count += 1;
      else bucket.set(channel.index, { channel, count: 1 });
    }
    return {
      group,
      nestModel: NEST_MODEL[group.kind],
      excitatory: summariseChannels([...excitatory.values()]),
      inhibitory: summariseChannels([...inhibitory.values()]),
    };
  });
}

function neuronParams(model: ExportCircuit, plan: GroupPlan): string[] {
  const entries: string[] = [];
  const neurons = model.neurons.slice(plan.group.offset, plan.group.offset + plan.group.size);
  for (const neuron of neurons) {
    const params = neuron.params;
    const bias = finite(neuron.bias, 0);
    const fields: string[] = [];
    switch (params.kind) {
      case 'lif':
        fields.push(
          `'C_m': ${pyFloat(params.cm)}`,
          `'g_L': ${pyFloat(params.gL)}`,
          `'E_L': ${pyFloat(params.eL)}`,
          `'V_m': ${pyFloat(params.eL)}`,
          `'V_th': ${pyFloat(params.vThresh)}`,
          `'V_reset': ${pyFloat(params.vReset)}`,
          `'t_ref': ${pyFloat(Math.max(params.tRefract, 0))}`,
          `'E_ex': ${pyFloat(plan.excitatory.eRev)}`,
          `'E_in': ${pyFloat(plan.inhibitory.eRev)}`,
          `'tau_syn_ex': ${pyFloat(plan.excitatory.tauDecay)}`,
          `'tau_syn_in': ${pyFloat(plan.inhibitory.tauDecay)}`,
          `'I_e': ${pyFloat(bias)}`,
        );
        break;
      case 'adex':
        fields.push(
          `'C_m': ${pyFloat(params.cm)}`,
          `'g_L': ${pyFloat(params.gL)}`,
          `'E_L': ${pyFloat(params.eL)}`,
          `'V_m': ${pyFloat(params.eL)}`,
          `'Delta_T': ${pyFloat(params.deltaT)}`,
          `'V_th': ${pyFloat(params.vT)}`,
          `'V_peak': ${pyFloat(params.vPeak)}`,
          `'V_reset': ${pyFloat(params.vReset)}`,
          `'a': ${pyFloat(params.a)}`,
          `'b': ${pyFloat(params.b)}`,
          `'tau_w': ${pyFloat(params.tauW)}`,
          `'t_ref': ${pyFloat(Math.max(params.tRefract, 0))}`,
          `'E_ex': ${pyFloat(plan.excitatory.eRev)}`,
          `'E_in': ${pyFloat(plan.inhibitory.eRev)}`,
          `'tau_syn_ex': ${pyFloat(plan.excitatory.tauDecay)}`,
          `'tau_syn_in': ${pyFloat(plan.inhibitory.tauDecay)}`,
          `'I_e': ${pyFloat(bias)}`,
        );
        break;
      case 'izhikevich':
        // NEST's izhikevich adds the input current straight into the mV/ms
        // equation, so our pA -> model-unit scale is folded into I_e here.
        fields.push(
          `'a': ${pyFloat(params.a)}`,
          `'b': ${pyFloat(params.b)}`,
          `'c': ${pyFloat(params.c)}`,
          `'d': ${pyFloat(params.d)}`,
          `'V_m': ${pyFloat(params.c)}`,
          `'U_m': ${pyFloat(params.b * params.c)}`,
          `'V_th': ${pyFloat(params.vPeak)}`,
          `'I_e': ${pyFloat(bias * params.iScale)}`,
          "'consistent_integration': True",
        );
        break;
      case 'hodgkin-huxley':
        fields.push(
          `'C_m': ${pyFloat(params.cm)}`,
          `'g_Na': ${pyFloat(params.gNa)}`,
          `'g_K': ${pyFloat(params.gK)}`,
          `'g_L': ${pyFloat(params.gL)}`,
          `'E_Na': ${pyFloat(params.eNa)}`,
          `'E_K': ${pyFloat(params.eK)}`,
          `'E_L': ${pyFloat(params.eL)}`,
          "'V_m': -65.0",
          `'tau_syn_ex': ${pyFloat(plan.excitatory.tauDecay)}`,
          `'tau_syn_in': ${pyFloat(plan.inhibitory.tauDecay)}`,
          `'I_e': ${pyFloat(bias)}`,
        );
        break;
      case 'morris-lecar':
        break;
    }
    entries.push(`    {${fields.join(', ')}},`);
  }
  return entries;
}

/** Convert a document weight into the unit the target NEST model expects. */
function nestWeight(model: ExportCircuit, synapse: ExportSynapse): number {
  const channel = model.channels[synapse.channel];
  const magnitude = Math.abs(synapse.weight);
  const target = model.neurons[synapse.post];
  const vRest = restingOf(target.params);
  switch (target.params.kind) {
    case 'lif':
    case 'adex':
      // Conductance in nS; NEST selects the receptor from the sign.
      return channel.excitatory ? magnitude : -magnitude;
    case 'hodgkin-huxley':
      // Peak postsynaptic current in pA, evaluated at the resting potential.
      return magnitude * (channel.kinetics.eRev - vRest);
    case 'izhikevich': {
      const iScale = finite(target.params.iScale, 0.04);
      // Charge delivered by the whole waveform, converted to a voltage jump.
      return iScale * magnitude * (channel.kinetics.eRev - vRest) * channel.kernel.area;
    }
    case 'morris-lecar':
      return 0;
  }
}

/** NEST delays must be a positive multiple of the kernel resolution. */
function quantiseTime(value: number, dt: number, minimumSteps: number): number {
  const steps = Math.max(minimumSteps, Math.round(value / dt));
  return Number((steps * dt).toFixed(6));
}

interface ConnectionPartition {
  synapseModel: string;
  plasticity: PlasticityKind;
  /** Scalar synapse parameters shared by every connection in the partition. */
  scalars: string[];
  synapses: ExportSynapse[];
  note: string | null;
}

function plasticitySignature(synapse: ExportSynapse, enabled: boolean): string {
  if (!enabled || synapse.plasticity.kind === 'static') return 'static';
  const p = synapse.plasticity;
  return [
    p.kind,
    p.aPlus,
    p.aMinus,
    p.tauPlus,
    p.tauMinus,
    p.tauX,
    p.tauY,
    p.wMin,
    p.wMax,
    p.learningRate,
  ].join(':');
}

function partitionConnections(model: ExportCircuit, usable: readonly ExportSynapse[]): ConnectionPartition[] {
  const keyed = groupBy(usable, (s) => plasticitySignature(s, model.plasticityEnabled));
  const partitions: ConnectionPartition[] = [];
  for (const [signature, synapses] of keyed) {
    const first = synapses[0];
    const kind: PlasticityKind = signature === 'static' ? 'static' : first.plasticity.kind;
    const scalars: string[] = [];
    let note: string | null = null;
    if (kind === 'stdp' || kind === 'triplet-stdp') {
      const p = first.plasticity;
      const wMax = Math.max(Math.abs(p.wMax), 1e-6);
      const aPlus = Math.max(Math.abs(p.aPlus), 1e-12);
      if (kind === 'stdp') {
        // With mu_plus = mu_minus = 0 NEST's rule is additive:
        //   potentiation dw = +lambda * Wmax * K_plus
        //   depression   dw = -alpha * lambda * Wmax * K_minus
        // which matches dw = +lr*aPlus*K_plus / -lr*aMinus*K_minus.
        scalars.push(
          `'lambda': ${pyFloat((p.learningRate * aPlus) / wMax)}`,
          `'alpha': ${pyFloat(Math.abs(p.aMinus) / aPlus)}`,
          "'mu_plus': 0.0",
          "'mu_minus': 0.0",
          `'tau_plus': ${pyFloat(p.tauPlus)}`,
          `'Wmax': ${pyFloat(wMax)}`,
        );
      } else {
        scalars.push(
          `'Aplus': ${pyFloat(p.learningRate * p.aPlus)}`,
          `'Aminus': ${pyFloat(p.learningRate * p.aMinus)}`,
          `'Aplus_triplet': ${pyFloat(p.learningRate * p.aPlus)}`,
          `'Aminus_triplet': ${pyFloat(p.learningRate * p.aMinus)}`,
          `'tau_plus': ${pyFloat(p.tauPlus)}`,
          `'tau_plus_triplet': ${pyFloat(p.tauX)}`,
          `'Wmax': ${pyFloat(wMax)}`,
        );
      }
    } else if (kind === 'hebbian' || kind === 'oja') {
      note =
        `NEST ships no built-in ${kind} synapse, so these ${synapses.length} connection(s) are created as ` +
        'static_synapse. Rebuild the rule with NESTML (or use the Brian2 export, which implements it) ' +
        'if the learning dynamics matter.';
    }
    partitions.push({ synapseModel: NEST_SYNAPSE[kind], plasticity: kind, scalars, synapses, note });
  }
  return partitions;
}

function targetCollection(indices: readonly number[]): string {
  return `nest.NodeCollection(NEURON_IDS[${pyIntList([...indices].sort((a, b) => a - b))}].tolist())`;
}

/** One point of a `step_current_generator` schedule. */
interface TimedValue {
  time: number;
  value: number;
}

/**
 * Quantise a schedule onto the kernel grid and force it strictly increasing.
 *
 * `step_current_generator` rejects an `amplitude_times` list that is not
 * strictly increasing, and quantisation alone can push two nominally distinct
 * points onto the same step. Points are supplied in non-decreasing time order,
 * so dropping every earlier point that is not strictly before the new one
 * amounts to "the last value written at a given time wins" — exactly what a
 * schedule whose pulse-off lands on the next pulse-on should mean.
 */
function stepSchedule(points: readonly TimedValue[], dt: number): { times: number[]; values: number[] } {
  const times: number[] = [];
  const values: number[] = [];
  for (const point of points) {
    const time = quantiseTime(finite(point.time, 0), dt, 1);
    while (times.length > 0 && times[times.length - 1] >= time) {
      times.pop();
      values.pop();
    }
    times.push(time);
    values.push(finite(point.value, 0));
  }
  return { times, values };
}

function stepCurrentGenerator(name: string, points: readonly TimedValue[], dt: number): string[] {
  const { times, values } = stepSchedule(points, dt);
  if (times.length === 0) return [];
  return [
    `${name} = nest.Create('step_current_generator', params={'amplitude_times': ${pyFloatList(times)}, ` +
      `'amplitude_values': ${pyFloatList(values)}})`,
  ];
}

function stimulusDevices(model: ExportCircuit, nestIndex: readonly number[]): string[] {
  const lines: string[] = [];
  let index = 0;
  for (const { stimulus, targets } of model.stimuli) {
    const mapped = targets.map((t) => nestIndex[t]).filter((t) => t >= 0);
    if (mapped.length === 0) continue;
    const name = `stim_${index}`;
    const p = stimulus.pattern;
    const device: string[] = [];
    let connectSpec = '';
    switch (p.kind) {
      case 'constant':
        device.push(`${name} = nest.Create('dc_generator', params={'amplitude': ${pyFloat(p.amplitude)}})`);
        break;
      case 'step':
        device.push(
          `${name} = nest.Create('dc_generator', params={'amplitude': ${pyFloat(p.amplitude)}, ` +
            `'start': ${pyFloat(Math.max(0, p.start))}, ` +
            `'stop': ${pyFloat(Math.max(0, p.start) + Math.max(0, p.duration))}})`,
        );
        break;
      case 'pulse-train': {
        const period = 1000 / Math.max(finite(p.frequency, 10), 1e-9);
        // A pulse at least as wide as the period leaves the current permanently
        // on — that is what `phase < width` means in the other exporters — so
        // the off-point is clamped to the next on-point, where `stepSchedule`
        // discards it.
        const width = Math.min(Math.max(finite(p.width, 1), model.dt), period);
        const points: TimedValue[] = [];
        for (let t = Math.max(0, p.start); t < model.duration; t += period) {
          points.push({ time: t, value: p.amplitude });
          const off = t + width;
          if (off >= model.duration) break;
          points.push({ time: off, value: 0 });
        }
        device.push(...stepCurrentGenerator(name, points, model.dt));
        break;
      }
      case 'sine':
        device.push(
          `${name} = nest.Create('ac_generator', params={'amplitude': ${pyFloat(p.amplitude)}, ` +
            `'frequency': ${pyFloat(p.frequency)}, 'offset': ${pyFloat(p.offset)}})`,
        );
        break;
      case 'poisson':
        device.push(
          '# A Poisson current stimulus becomes a Poisson spike train whose events carry the',
          '# stimulus amplitude as their synaptic weight.',
          `${name} = nest.Create('poisson_generator', params={'rate': ${pyFloat(Math.max(0, p.rate))}})`,
        );
        connectSpec = `, syn_spec={'weight': ${pyFloat(p.amplitude)}, 'delay': ${pyFloat(model.dt)}}`;
        break;
      case 'ramp': {
        // A ramp of no duration injects nothing, exactly as in the other
        // exporters, where the active window is empty.
        if (p.duration > 0) {
          // step_current_generator holds each value until the next time point,
          // so the ramp is sampled on a 1 ms grid: finer than that only bloats
          // the script without changing the current a neuron sees.
          const steps = Math.max(2, Math.min(512, Math.ceil(p.duration / Math.max(model.dt, 1)) + 1));
          const points: TimedValue[] = [];
          for (let i = 0; i < steps; i += 1) {
            const frac = i / (steps - 1);
            points.push({
              time: Math.max(0, p.start) + frac * p.duration,
              value: p.from + (p.to - p.from) * frac,
            });
          }
          points.push({ time: Math.max(0, p.start) + p.duration + model.dt, value: 0 });
          device.push(...stepCurrentGenerator(name, points, model.dt));
        }
        break;
      }
    }
    // A pattern that resolves to no schedule at all creates no device, rather
    // than a device NEST would reject.
    if (device.length === 0) continue;
    index += 1;
    lines.push(`# stimulus ${pyStr(stimulus.name)}`);
    lines.push(...device);
    lines.push(`nest.Connect(${name}, ${targetCollection(mapped)}${connectSpec})`);
    lines.push('');
  }
  return lines;
}

/** Generate a complete PyNEST simulation script for a circuit. */
export function exportNest(circuit: Circuit): string {
  const model = indexCircuit(circuit);
  const plans = planGroups(model);
  const notes: string[] = [];

  // NEST ids are only assigned to neurons whose model has a NEST equivalent.
  const nestIndex = new Array<number>(model.neurons.length).fill(-1);
  let cursor = 0;
  for (const plan of plans) {
    if (plan.nestModel === null) continue;
    for (let i = 0; i < plan.group.size; i += 1) {
      nestIndex[plan.group.offset + i] = cursor;
      cursor += 1;
    }
  }

  for (const plan of plans) {
    if (plan.nestModel === null) {
      notes.push(
        `${plan.group.size} ${modelLabel(plan.group.kind)} neuron(s) have no NEST equivalent and are not created.`,
      );
      continue;
    }
    if (plan.excitatory.distinct > 1) {
      notes.push(
        `Population '${plan.group.name}' receives ${plan.excitatory.distinct} distinct excitatory kinetics; ` +
          `${plan.nestModel} has a single excitatory port, so tau_syn_ex and E_ex are connection-count means.`,
      );
    }
    if (plan.inhibitory.distinct > 1) {
      notes.push(
        `Population '${plan.group.name}' receives ${plan.inhibitory.distinct} distinct inhibitory kinetics; ` +
          `${plan.nestModel} has a single inhibitory port, so tau_syn_in and E_in are connection-count means.`,
      );
    }
    if (plan.group.kind === 'hodgkin-huxley') {
      notes.push(
        'hh_psc_alpha uses fixed squid-axon rate constants and detects a spike at 0 mV, so the q10 factor ' +
          'and the custom detection threshold in the document are not representable.',
      );
    }
  }

  const usableSynapses = model.synapses.filter((s) => nestIndex[s.pre] >= 0 && nestIndex[s.post] >= 0);
  const skipped = model.synapses.length - usableSynapses.length;
  if (skipped > 0) {
    notes.push(`${skipped} synapse(s) touch a neuron with no NEST equivalent and are not created.`);
  }
  if (model.synapses.some((s) => s.weight < 0)) {
    notes.push(
      'Conductance-based NEST models select the receptor from the sign of the weight, so negative document ' +
        'weights are exported as their magnitude on the receptor implied by the reversal potential.',
    );
  }
  if (model.channels.some((c) => c.kernel.form === 'dual')) {
    notes.push(
      'iaf_cond_exp / aeif_cond_exp use a single-exponential conductance: the decay constant is kept, the ' +
        'rise phase is dropped, and the peak conductance is preserved.',
    );
  }

  const banner = bannerLines(model, 'PyNEST');
  if (notes.length > 0) {
    banner.push('', 'NEST-specific notes:');
    for (const note of notes) banner.push(`  - ${note}`);
  }

  const out: string[] = [];
  out.push(pyDocstring(banner));
  out.push('');
  out.push('import nest');
  out.push('import numpy as np');
  out.push('');
  out.push('nest.ResetKernel()');
  out.push(
    `nest.SetKernelStatus({'resolution': ${pyFloat(model.dt)}, 'rng_seed': ${pyInt(Math.max(1, model.seed))}, ` +
      "'print_time': False})",
  );
  out.push(`SIM_DURATION = ${pyFloat(model.duration)}`);
  out.push('');

  const created: string[] = [];
  for (const plan of plans) {
    const g = plan.group.name;
    out.push(RULE);
    out.push(`# Population '${g}': ${plan.group.size} ${modelLabel(plan.group.kind)} neurons`);
    out.push(RULE);
    if (plan.nestModel === null) {
      out.push(
        pyComment([
          `NEST has no built-in ${modelLabel(plan.group.kind)} model, so these ${plan.group.size} neurons are`,
          'deliberately NOT instantiated rather than silently replaced by a different model.',
          'To simulate them either',
          '  1. describe the model in NESTML, build it into a module and load it here:',
          "         nest.Install('mymodule')",
          `         ${plan.group.name} = nest.Create('morris_lecar', ${plan.group.size})`,
          '  2. or run this population from the Brian2 export, which implements it directly.',
          'Connections that touch these neurons are listed in the banner above and are not created below.',
        ]),
      );
      out.push('');
      continue;
    }
    out.push(`${g} = nest.Create(${pyStr(plan.nestModel)}, ${pyInt(plan.group.size)})`);
    out.push(`${g}_params = [`);
    out.push(...neuronParams(model, plan));
    out.push(']');
    out.push(`nest.SetStatus(${g}, ${g}_params)`);
    out.push('');
    created.push(g);
  }

  if (created.length === 0) {
    out.push('# No population could be mapped onto a NEST model.');
    out.push("print('nothing to simulate')");
    return `${out.join('\n')}\n`;
  }

  out.push(`ALL_NEURONS = ${created.join(' + ')}`);
  out.push('NEURON_IDS = np.array(ALL_NEURONS.tolist(), dtype=np.uint64)');
  out.push('');

  const noiseTargets = model.neurons
    .map((neuron, index) => ({ index, sigma: effectiveNoise(neuron, model) }))
    .filter((entry) => entry.sigma > 0 && nestIndex[entry.index] >= 0);
  if (noiseTargets.length > 0) {
    out.push(RULE);
    out.push('# Background noise');
    out.push(RULE);
    let noiseIndex = 0;
    for (const [sigma, entries] of groupBy(noiseTargets, (entry) => entry.sigma)) {
      const name = `noise_${noiseIndex}`;
      noiseIndex += 1;
      out.push(
        `${name} = nest.Create('noise_generator', params={'mean': 0.0, 'std': ${pyFloat(sigma)}, ` +
          `'dt': ${pyFloat(model.dt)}})`,
      );
      out.push(`nest.Connect(${name}, ${targetCollection(entries.map((e) => nestIndex[e.index]))})`);
    }
    out.push('');
  }

  const stimulusLines = stimulusDevices(model, nestIndex);
  if (stimulusLines.length > 0) {
    out.push(RULE);
    out.push('# Stimuli');
    out.push(RULE);
    out.push(...stimulusLines);
  }

  if (usableSynapses.length > 0) {
    out.push(RULE);
    out.push('# Connections');
    out.push(RULE);
    out.push('# nest.Connect accepts arrays of node ids with the one_to_one rule, which is the only way');
    out.push('# to express an explicit connection list in which a source appears more than once.');
    out.push('# The rule must be passed explicitly: NEST rejects arrays of node ids under any other.');
    const partitions = partitionConnections(model, usableSynapses);

    // stdp_synapse and stdp_triplet_synapse read their postsynaptic trace
    // constants from the target neuron rather than from the synapse.
    const traceConstants = new Map<number, { tauMinus: number; tauY: number }>();
    for (const partition of partitions) {
      if (partition.plasticity !== 'stdp' && partition.plasticity !== 'triplet-stdp') continue;
      for (const synapse of partition.synapses) {
        traceConstants.set(nestIndex[synapse.post], {
          tauMinus: finite(synapse.plasticity.tauMinus, 33.7),
          tauY: finite(synapse.plasticity.tauY, 125),
        });
      }
    }
    if (traceConstants.size > 0) {
      const byValue = groupBy([...traceConstants], ([, value]) => `${value.tauMinus}|${value.tauY}`);
      for (const [, entries] of byValue) {
        const value = entries[0][1];
        out.push(
          `nest.SetStatus(${targetCollection(entries.map(([id]) => id))}, ` +
            `{'tau_minus': ${pyFloat(value.tauMinus)}, 'tau_minus_triplet': ${pyFloat(value.tauY)}})`,
        );
      }
      out.push('');
    }

    let partitionIndex = 0;
    for (const partition of partitions) {
      const name = `conn_${partitionIndex}`;
      partitionIndex += 1;
      if (partition.note) out.push(pyComment([partition.note]));
      out.push(`# ${partition.synapses.length} connection(s) using ${partition.synapseModel}`);
      out.push(`${name}_pre = NEURON_IDS[${pyIntList(partition.synapses.map((s) => nestIndex[s.pre]))}]`);
      out.push(`${name}_post = NEURON_IDS[${pyIntList(partition.synapses.map((s) => nestIndex[s.post]))}]`);
      out.push(
        `${name}_weight = np.array(${pyFloatList(partition.synapses.map((s) => nestWeight(model, s)))})`,
      );
      out.push(
        `${name}_delay = np.array(${pyFloatList(
          partition.synapses.map((s) => quantiseTime(s.delay, model.dt, 1)),
        )})`,
      );
      const spec = [
        `'synapse_model': ${pyStr(partition.synapseModel)}`,
        `'weight': ${name}_weight`,
        `'delay': ${name}_delay`,
        ...partition.scalars,
      ];
      out.push(`nest.Connect(${name}_pre, ${name}_post, 'one_to_one', syn_spec={${spec.join(', ')}})`);
      out.push('');
    }
  }

  out.push(RULE);
  out.push('# Recording devices');
  out.push(RULE);
  out.push("spikes = nest.Create('spike_recorder')");
  out.push('nest.Connect(ALL_NEURONS, spikes)');
  for (const plan of plans) {
    if (plan.nestModel === null) continue;
    const g = plan.group.name;
    out.push(
      `mm_${g} = nest.Create('multimeter', params={'record_from': [` +
        `${RECORDABLES[plan.group.kind].map(pyStr).join(', ')}], 'interval': ${pyFloat(model.dt)}})`,
    );
    out.push(`nest.Connect(mm_${g}, ${g})`);
  }
  out.push('');
  out.push('nest.Simulate(SIM_DURATION)');
  out.push('');
  out.push(RULE);
  out.push('# Summary');
  out.push(RULE);
  out.push("senders = spikes.get('events')['senders']");
  out.push("print('%d spikes from %d neurons over %.1f ms' % (len(senders), len(ALL_NEURONS), SIM_DURATION))");
  out.push("print('mean rate: %.2f Hz' % (len(senders) / (len(ALL_NEURONS) * SIM_DURATION / 1000.0)))");

  return `${out.join('\n')}\n`;
}
