/**
 * PyTorch exporter.
 *
 * Emits a `torch.nn.Module` that runs the circuit as a discrete-time recurrent
 * spiking layer:
 *
 *   * spikes are produced by a custom `autograd.Function` with a Heaviside
 *     forward pass and an arctan surrogate derivative, so the recurrence is
 *     trainable by backpropagation through time;
 *   * the recurrent weight matrix is an `nn.Parameter` initialised from the
 *     document's synaptic weights;
 *   * conduction delays are honoured through a rolling spike-history buffer,
 *     with one boolean connectivity mask per (conductance channel, delay) pair;
 *   * membrane potential, adaptation, gating variables, refractory timers and
 *     synaptic conductances are registered buffers, so a long trial can be run
 *     in chunks.
 *
 * All five membrane models are implemented; neurons are emitted grouped by
 * model so each group is a contiguous slice of the state vectors.
 */

import type { Circuit, Neuron } from '@neuroforge/shared';

import {
  bannerLines,
  binDelays,
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
import type { ExportCircuit, ExportGroup } from './common';

/** Upper bound on the number of distinct delay values given a dense mask each. */
const MAX_DELAY_BINS = 32;

interface Block {
  channel: number;
  delaySteps: number;
}

function buildBlocks(model: ExportCircuit): { blocks: Block[]; assignment: number[]; quantised: boolean } {
  if (model.synapses.length === 0) return { blocks: [], assignment: [], quantised: false };
  const steps = model.synapses.map((s) => delayToSteps(s.delay, model.dt));
  const binning = binDelays(steps, MAX_DELAY_BINS);
  const blocks: Block[] = [];
  const lookup = new Map<string, number>();
  const assignment = model.synapses.map((synapse, i) => {
    const delaySteps = binning.bins[binning.assignment[i]] ?? 1;
    const key = `${synapse.channel}|${delaySteps}`;
    const existing = lookup.get(key);
    if (existing !== undefined) return existing;
    lookup.set(key, blocks.length);
    blocks.push({ channel: synapse.channel, delaySteps });
    return blocks.length - 1;
  });
  return { blocks, assignment, quantised: binning.quantised };
}

function groupNeurons(model: ExportCircuit, group: ExportGroup): Neuron[] {
  return model.neurons.slice(group.offset, group.offset + group.size);
}

function buffer(name: string, values: readonly number[]): string {
  return `self.register_buffer(${pyStr(name)}, torch.tensor([${pyFloatList(values, 8, 12)}], dtype=dtype))`;
}

function parameterBuffers(model: ExportCircuit, group: ExportGroup, index: number): string[] {
  const neurons = groupNeurons(model, group);
  const g = `g${index}`;
  const lines: string[] = [];
  switch (group.kind) {
    case 'lif': {
      const p = (n: Neuron) => (n.params.kind === 'lif' ? n.params : null);
      lines.push(buffer(`${g}_cm`, neurons.map((n) => p(n)?.cm ?? 200)));
      lines.push(buffer(`${g}_gl`, neurons.map((n) => p(n)?.gL ?? 10)));
      lines.push(buffer(`${g}_el`, neurons.map((n) => p(n)?.eL ?? -70)));
      lines.push(buffer(`${g}_vth`, neurons.map((n) => p(n)?.vThresh ?? -50)));
      lines.push(buffer(`${g}_vreset`, neurons.map((n) => p(n)?.vReset ?? -58)));
      lines.push(buffer(`${g}_tref`, neurons.map((n) => Math.max(p(n)?.tRefract ?? 0, 0))));
      break;
    }
    case 'adex': {
      const p = (n: Neuron) => (n.params.kind === 'adex' ? n.params : null);
      lines.push(buffer(`${g}_cm`, neurons.map((n) => p(n)?.cm ?? 281)));
      lines.push(buffer(`${g}_gl`, neurons.map((n) => p(n)?.gL ?? 30)));
      lines.push(buffer(`${g}_el`, neurons.map((n) => p(n)?.eL ?? -70.6)));
      lines.push(buffer(`${g}_delta`, neurons.map((n) => Math.max(p(n)?.deltaT ?? 2, 1e-6))));
      lines.push(buffer(`${g}_vt`, neurons.map((n) => p(n)?.vT ?? -50.4)));
      lines.push(buffer(`${g}_vpeak`, neurons.map((n) => p(n)?.vPeak ?? 20)));
      lines.push(buffer(`${g}_vreset`, neurons.map((n) => p(n)?.vReset ?? -70.6)));
      lines.push(buffer(`${g}_a`, neurons.map((n) => p(n)?.a ?? 4)));
      lines.push(buffer(`${g}_b`, neurons.map((n) => p(n)?.b ?? 80.5)));
      lines.push(buffer(`${g}_tauw`, neurons.map((n) => Math.max(p(n)?.tauW ?? 144, 1e-6))));
      lines.push(buffer(`${g}_tref`, neurons.map((n) => Math.max(p(n)?.tRefract ?? 0, 0))));
      break;
    }
    case 'izhikevich': {
      const p = (n: Neuron) => (n.params.kind === 'izhikevich' ? n.params : null);
      lines.push(buffer(`${g}_a`, neurons.map((n) => p(n)?.a ?? 0.02)));
      lines.push(buffer(`${g}_b`, neurons.map((n) => p(n)?.b ?? 0.2)));
      lines.push(buffer(`${g}_c`, neurons.map((n) => p(n)?.c ?? -65)));
      lines.push(buffer(`${g}_d`, neurons.map((n) => p(n)?.d ?? 8)));
      lines.push(buffer(`${g}_vpeak`, neurons.map((n) => p(n)?.vPeak ?? 30)));
      lines.push(buffer(`${g}_iscale`, neurons.map((n) => p(n)?.iScale ?? 0.04)));
      break;
    }
    case 'hodgkin-huxley': {
      const p = (n: Neuron) => (n.params.kind === 'hodgkin-huxley' ? n.params : null);
      lines.push(buffer(`${g}_cm`, neurons.map((n) => p(n)?.cm ?? 100)));
      lines.push(buffer(`${g}_gna`, neurons.map((n) => p(n)?.gNa ?? 12000)));
      lines.push(buffer(`${g}_gk`, neurons.map((n) => p(n)?.gK ?? 3600)));
      lines.push(buffer(`${g}_gl`, neurons.map((n) => p(n)?.gL ?? 30)));
      lines.push(buffer(`${g}_ena`, neurons.map((n) => p(n)?.eNa ?? 50)));
      lines.push(buffer(`${g}_ek`, neurons.map((n) => p(n)?.eK ?? -77)));
      lines.push(buffer(`${g}_el`, neurons.map((n) => p(n)?.eL ?? -54.4)));
      lines.push(buffer(`${g}_vdetect`, neurons.map((n) => p(n)?.vDetect ?? -20)));
      lines.push(buffer(`${g}_q10`, neurons.map((n) => p(n)?.q10 ?? 1)));
      break;
    }
    case 'morris-lecar': {
      const p = (n: Neuron) => (n.params.kind === 'morris-lecar' ? n.params : null);
      lines.push(buffer(`${g}_cm`, neurons.map((n) => p(n)?.cm ?? 20)));
      lines.push(buffer(`${g}_gca`, neurons.map((n) => p(n)?.gCa ?? 4.4)));
      lines.push(buffer(`${g}_gk`, neurons.map((n) => p(n)?.gK ?? 8)));
      lines.push(buffer(`${g}_gl`, neurons.map((n) => p(n)?.gL ?? 2)));
      lines.push(buffer(`${g}_eca`, neurons.map((n) => p(n)?.eCa ?? 120)));
      lines.push(buffer(`${g}_ek`, neurons.map((n) => p(n)?.eK ?? -84)));
      lines.push(buffer(`${g}_el`, neurons.map((n) => p(n)?.eL ?? -60)));
      lines.push(buffer(`${g}_v1`, neurons.map((n) => p(n)?.v1 ?? -1.2)));
      lines.push(buffer(`${g}_v2`, neurons.map((n) => Math.max(p(n)?.v2 ?? 18, 1e-6))));
      lines.push(buffer(`${g}_v3`, neurons.map((n) => p(n)?.v3 ?? 2)));
      lines.push(buffer(`${g}_v4`, neurons.map((n) => Math.max(p(n)?.v4 ?? 30, 1e-6))));
      lines.push(buffer(`${g}_phi`, neurons.map((n) => p(n)?.phi ?? 0.04)));
      lines.push(buffer(`${g}_vdetect`, neurons.map((n) => p(n)?.vDetect ?? 0)));
      break;
    }
  }
  return lines;
}

/** Per-model update, written against the slice variables prepared by the caller. */
function groupUpdate(group: ExportGroup, index: number): string[] {
  const g = `g${index}`;
  const a = pyInt(group.offset);
  const b = pyInt(group.offset + group.size);
  const slice = `${a}:${b}`;
  const head = [
    `# ${modelLabel(group.kind)}: neurons ${a}..${pyInt(group.offset + group.size - 1)}`,
    `v_${g} = v[:, ${slice}]`,
    `w_${g} = w_adapt[:, ${slice}]`,
    `r_${g} = refractory[:, ${slice}]`,
    `i_${g} = i_total[:, ${slice}]`,
  ];
  switch (group.kind) {
    case 'lif':
      return [
        ...head,
        `dv = (self.${g}_gl * (self.${g}_el - v_${g}) + i_${g}) / self.${g}_cm`,
        `free_${g} = (r_${g} <= 0.0).to(v.dtype)`,
        `v_free = v_${g} + dv * self.dt`,
        `v_free = free_${g} * v_free + (1.0 - free_${g}) * v_${g}`,
        `s_${g} = spike(v_free - self.${g}_vth)`,
        `nv_${g} = s_${g} * self.${g}_vreset + (1.0 - s_${g}) * v_free`,
        `nw_${g} = w_${g}`,
        `nr_${g} = s_${g} * self.${g}_tref + (1.0 - s_${g}) * torch.clamp(r_${g} - self.dt, min=0.0)`,
        `na_${g} = above[:, ${slice}]`,
        `nm_${g} = gate_m[:, ${slice}]`,
        `nh_${g} = gate_h[:, ${slice}]`,
        `nn_${g} = gate_n[:, ${slice}]`,
      ];
    case 'adex':
      return [
        ...head,
        // The exponential term is clamped before exp() so that a diverging
        // trajectory produces a spike instead of an inf.
        `expo = self.${g}_gl * self.${g}_delta * torch.exp(torch.clamp((v_${g} - self.${g}_vt) / self.${g}_delta, max=30.0))`,
        `dv = (self.${g}_gl * (self.${g}_el - v_${g}) + expo - w_${g} + i_${g}) / self.${g}_cm`,
        `free_${g} = (r_${g} <= 0.0).to(v.dtype)`,
        `v_free = v_${g} + dv * self.dt`,
        `v_free = free_${g} * v_free + (1.0 - free_${g}) * v_${g}`,
        `w_free = w_${g} + ((self.${g}_a * (v_${g} - self.${g}_el) - w_${g}) / self.${g}_tauw) * self.dt`,
        `s_${g} = spike(v_free - self.${g}_vpeak)`,
        `nv_${g} = s_${g} * self.${g}_vreset + (1.0 - s_${g}) * v_free`,
        `nw_${g} = w_free + s_${g} * self.${g}_b`,
        `nr_${g} = s_${g} * self.${g}_tref + (1.0 - s_${g}) * torch.clamp(r_${g} - self.dt, min=0.0)`,
        `na_${g} = above[:, ${slice}]`,
        `nm_${g} = gate_m[:, ${slice}]`,
        `nh_${g} = gate_h[:, ${slice}]`,
        `nn_${g} = gate_n[:, ${slice}]`,
      ];
    case 'izhikevich':
      return [
        ...head,
        `drive = i_${g} * self.${g}_iscale`,
        `dv = 0.04 * v_${g} * v_${g} + 5.0 * v_${g} + 140.0 - w_${g} + drive`,
        `v_free = v_${g} + dv * self.dt`,
        `w_free = w_${g} + self.${g}_a * (self.${g}_b * v_${g} - w_${g}) * self.dt`,
        `s_${g} = spike(v_free - self.${g}_vpeak)`,
        `nv_${g} = s_${g} * self.${g}_c + (1.0 - s_${g}) * v_free`,
        `nw_${g} = w_free + s_${g} * self.${g}_d`,
        `nr_${g} = r_${g}`,
        `na_${g} = above[:, ${slice}]`,
        `nm_${g} = gate_m[:, ${slice}]`,
        `nh_${g} = gate_h[:, ${slice}]`,
        `nn_${g} = gate_n[:, ${slice}]`,
      ];
    case 'hodgkin-huxley':
      return [
        ...head,
        `m_${g} = gate_m[:, ${slice}]`,
        `h_${g} = gate_h[:, ${slice}]`,
        `n_${g} = gate_n[:, ${slice}]`,
        `alpha_m = 1.0 / _exprel(-(v_${g} + 40.0) / 10.0)`,
        `beta_m = 4.0 * torch.exp(-(v_${g} + 65.0) / 18.0)`,
        `alpha_h = 0.07 * torch.exp(-(v_${g} + 65.0) / 20.0)`,
        `beta_h = 1.0 / (1.0 + torch.exp(-(v_${g} + 35.0) / 10.0))`,
        `alpha_n = 0.1 / _exprel(-(v_${g} + 55.0) / 10.0)`,
        `beta_n = 0.125 * torch.exp(-(v_${g} + 65.0) / 80.0)`,
        `nm_${g} = m_${g} + self.${g}_q10 * (alpha_m * (1.0 - m_${g}) - beta_m * m_${g}) * self.dt`,
        `nh_${g} = h_${g} + self.${g}_q10 * (alpha_h * (1.0 - h_${g}) - beta_h * h_${g}) * self.dt`,
        `nn_${g} = n_${g} + self.${g}_q10 * (alpha_n * (1.0 - n_${g}) - beta_n * n_${g}) * self.dt`,
        `i_ion = (self.${g}_gna * m_${g} ** 3 * h_${g} * (self.${g}_ena - v_${g})` +
          ` + self.${g}_gk * n_${g} ** 4 * (self.${g}_ek - v_${g})` +
          ` + self.${g}_gl * (self.${g}_el - v_${g}))`,
        `nv_${g} = v_${g} + (i_ion + i_${g}) / self.${g}_cm * self.dt`,
        // A spike is registered on the rising edge only, so the wide action
        // potential is not counted once per timestep.
        `cross_${g} = spike(nv_${g} - self.${g}_vdetect)`,
        `s_${g} = cross_${g} * (1.0 - above[:, ${slice}])`,
        `na_${g} = cross_${g}`,
        `nw_${g} = w_${g}`,
        `nr_${g} = r_${g}`,
      ];
    case 'morris-lecar':
      return [
        ...head,
        `m_inf = 0.5 * (1.0 + torch.tanh((v_${g} - self.${g}_v1) / self.${g}_v2))`,
        `w_inf = 0.5 * (1.0 + torch.tanh((v_${g} - self.${g}_v3) / self.${g}_v4))`,
        `lam = self.${g}_phi * torch.cosh((v_${g} - self.${g}_v3) / (2.0 * self.${g}_v4))`,
        `i_ion = (self.${g}_gca * m_inf * (self.${g}_eca - v_${g})` +
          ` + self.${g}_gk * w_${g} * (self.${g}_ek - v_${g})` +
          ` + self.${g}_gl * (self.${g}_el - v_${g}))`,
        `nv_${g} = v_${g} + (i_ion + i_${g}) / self.${g}_cm * self.dt`,
        `nw_${g} = w_${g} + lam * (w_inf - w_${g}) * self.dt`,
        `cross_${g} = spike(nv_${g} - self.${g}_vdetect)`,
        `s_${g} = cross_${g} * (1.0 - above[:, ${slice}])`,
        `na_${g} = cross_${g}`,
        `nr_${g} = r_${g}`,
        `nm_${g} = gate_m[:, ${slice}]`,
        `nh_${g} = gate_h[:, ${slice}]`,
        `nn_${g} = gate_n[:, ${slice}]`,
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
    """(exp(x) - 1) / x, finite at x = 0 and safe to differentiate."""
    safe = torch.where(x.abs() < 1e-6, torch.ones_like(x), x)
    return torch.where(x.abs() < 1e-6, 1.0 + x / 2.0, torch.expm1(safe) / safe)


class ArctanSpike(torch.autograd.Function):
    """Heaviside forward pass with an arctan surrogate derivative.

    The derivative is that of (1/pi) * arctan(pi/2 * alpha * x) + 1/2, which is
    the standard smooth relaxation of the step used for surrogate-gradient
    training of spiking networks.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        scaled = (math.pi / 2.0) * SURROGATE_ALPHA * x
        return grad_output * SURROGATE_ALPHA / (2.0 * (1.0 + scaled * scaled))


def spike(x):
    return ArctanSpike.apply(x)


def build_stimulus(n_steps, dtype=torch.float32, device=None):
    """Rebuild the document's stimuli as an [n_steps, N_NEURONS] current in pA."""
    values = torch.zeros(n_steps, N_NEURONS, dtype=torch.float64)
    t = torch.arange(n_steps, dtype=torch.float64) * DT
    for kind, targets, p in STIMULI:
        idx = torch.tensor(targets, dtype=torch.long)
        if kind == 'constant':
            wave = torch.full((n_steps, 1), p['amplitude'], dtype=torch.float64)
        elif kind == 'step':
            active = (t >= p['start']) & (t < p['start'] + p['duration'])
            wave = torch.where(active, torch.tensor(p['amplitude'], dtype=torch.float64),
                               torch.zeros((), dtype=torch.float64)).unsqueeze(1)
        elif kind == 'pulse-train':
            period = 1000.0 / p['frequency']
            phase = torch.where(t >= p['start'], torch.remainder(t - p['start'], period),
                                torch.tensor(period, dtype=torch.float64))
            active = (t >= p['start']) & (phase < p['width'])
            wave = torch.where(active, torch.tensor(p['amplitude'], dtype=torch.float64),
                               torch.zeros((), dtype=torch.float64)).unsqueeze(1)
        elif kind == 'sine':
            wave = (p['amplitude'] * torch.sin(2.0 * math.pi * p['frequency'] * t / 1000.0)
                    + p['offset']).unsqueeze(1)
        elif kind == 'poisson':
            generator = torch.Generator().manual_seed(int(p['seed']))
            draws = torch.rand(n_steps, idx.numel(), generator=generator, dtype=torch.float64)
            wave = torch.where(draws < p['rate'] * DT / 1000.0,
                               torch.tensor(p['amplitude'], dtype=torch.float64),
                               torch.zeros((), dtype=torch.float64))
        elif kind == 'ramp':
            span = max(p['duration'], 1e-9)
            frac = torch.clamp((t - p['start']) / span, 0.0, 1.0)
            active = (t >= p['start']) & (t < p['start'] + p['duration'])
            wave = torch.where(active, p['from'] + (p['to'] - p['from']) * frac,
                               torch.zeros((), dtype=torch.float64)).unsqueeze(1)
        else:
            raise ValueError('unknown stimulus pattern: ' + kind)
        values[:, idx] += wave
    return values.to(dtype=dtype, device=device)`;

/** Generate a runnable PyTorch module implementing the circuit. */
export function exportPyTorch(circuit: Circuit): string {
  const model = indexCircuit(circuit);
  const { blocks, assignment, quantised } = buildBlocks(model);
  const notes: string[] = [];
  if (quantised) {
    notes.push(
      `Conduction delays were quantised to ${MAX_DELAY_BINS} bins so that each (channel, delay) pair ` +
        'can carry one dense connectivity mask.',
    );
  }
  if (model.plasticityEnabled && model.synapses.some((s) => s.plasticity.kind !== 'static')) {
    notes.push(
      'Synaptic plasticity is not replayed here: the weight matrix is an nn.Parameter, so learning is ' +
        'expected to come from the surrogate gradient rather than from the document’s STDP rule.',
    );
  }
  const maxDelay = blocks.reduce((max, block) => Math.max(max, block.delaySteps), 1);
  const banner = bannerLines(model, 'PyTorch');
  if (notes.length > 0) {
    banner.push('', 'PyTorch-specific notes:');
    for (const note of notes) banner.push(`  - ${note}`);
  }

  const noise = model.neurons.map((n) => effectiveNoise(n, model));
  const hasNoise = noise.some((s) => s > 0);
  const rest = model.neurons.map((n) => restingState(n.params));
  const channels = model.channels;

  const out: string[] = [];
  out.push(pyDocstring(banner));
  out.push('');
  out.push('import math');
  out.push('');
  out.push('import torch');
  out.push('import torch.nn as nn');
  out.push('');
  out.push(`N_NEURONS = ${pyInt(model.neurons.length)}`);
  out.push(`DT = ${pyFloat(model.dt)}`);
  out.push(`HISTORY = ${pyInt(maxDelay + 1)}`);
  out.push(`N_CHANNELS = ${pyInt(channels.length)}`);
  out.push('SURROGATE_ALPHA = 2.0');
  out.push(`DEFAULT_STEPS = ${pyInt(Math.max(1, Math.round(model.duration / model.dt)))}`);
  out.push('');
  out.push('# Neuron ids in export order; groups of the same membrane model are contiguous.');
  out.push(`NEURON_IDS = [`);
  for (const neuron of model.neurons) out.push(`    ${pyStr(neuron.id)},`);
  out.push(']');
  out.push('');
  out.push('# Connectivity, as parallel per-synapse arrays.');
  out.push(`SYN_PRE = ${pyIntList(model.synapses.map((s) => s.pre))}`);
  out.push(`SYN_POST = ${pyIntList(model.synapses.map((s) => s.post))}`);
  out.push(`SYN_WEIGHT = ${pyFloatList(model.synapses.map((s) => s.weight))}`);
  out.push(`SYN_BLOCK = ${pyIntList(assignment)}`);
  out.push('');
  out.push('# One block per (conductance channel, delay in steps) pair.');
  out.push(`BLOCK_CHANNEL = ${pyIntList(blocks.map((b) => b.channel))}`);
  out.push(`BLOCK_DELAY = ${pyIntList(blocks.map((b) => b.delaySteps))}`);
  out.push('');
  out.push('# Synaptic channels: rise/decay in ms, reversal in mV, Mg block strength.');
  out.push(`CHANNEL_TAU_RISE = ${pyFloatList(channels.map((c) => c.kernel.tauRise))}`);
  out.push(`CHANNEL_TAU_DECAY = ${pyFloatList(channels.map((c) => c.kernel.tauDecay))}`);
  out.push(`CHANNEL_EREV = ${pyFloatList(channels.map((c) => c.kinetics.eRev))}`);
  out.push(`CHANNEL_MG = ${pyFloatList(channels.map((c) => c.kinetics.mgBlock))}`);
  out.push(`CHANNEL_NORM = ${pyFloatList(channels.map((c) => c.kernel.norm))}`);
  out.push(`CHANNEL_DUAL = ${pyIntList(channels.map((c) => (c.kernel.form === 'dual' ? 1 : 0)))}`);
  out.push('');
  out.push('STIMULI = [');
  out.push(...stimulusTable(model));
  out.push(']');
  out.push('');
  out.push('');
  out.push(PRELUDE);
  out.push('');
  out.push('');
  out.push('class NeuroForgeCircuit(nn.Module):');
  out.push('    """Discrete-time recurrent spiking layer for the exported circuit."""');
  out.push('');
  out.push('    def __init__(self, dtype=torch.float32):');
  out.push('        super().__init__()');
  out.push('        self.n = N_NEURONS');
  out.push('        self.dt = DT');
  out.push('        self.history = HISTORY');
  out.push('        self.n_channels = N_CHANNELS');
  out.push('        self.blocks = list(zip(BLOCK_CHANNEL, BLOCK_DELAY))');
  out.push('');
  out.push('        weight = torch.zeros(N_NEURONS, N_NEURONS, dtype=dtype)');
  out.push('        pre = torch.tensor(SYN_PRE, dtype=torch.long)');
  out.push('        post = torch.tensor(SYN_POST, dtype=torch.long)');
  out.push('        weight.index_put_((post, pre), torch.tensor(SYN_WEIGHT, dtype=dtype), accumulate=True)');
  out.push('        # weight[post, pre] is the peak conductance in nS. Trainable.');
  out.push('        self.weight = nn.Parameter(weight)');
  out.push('');
  out.push('        mask = torch.zeros(len(self.blocks), N_NEURONS, N_NEURONS, dtype=torch.bool)');
  out.push('        if len(SYN_BLOCK) > 0:');
  out.push('            mask[torch.tensor(SYN_BLOCK, dtype=torch.long), post, pre] = True');
  out.push("        self.register_buffer('block_mask', mask)");
  out.push('');
  out.push('        tau_rise = torch.tensor(CHANNEL_TAU_RISE, dtype=dtype).view(1, -1, 1)');
  out.push('        tau_decay = torch.tensor(CHANNEL_TAU_DECAY, dtype=dtype).view(1, -1, 1)');
  out.push("        self.register_buffer('rise_decay', torch.exp(-self.dt / torch.clamp(tau_rise, min=1e-6)))");
  out.push("        self.register_buffer('decay_decay', torch.exp(-self.dt / torch.clamp(tau_decay, min=1e-6)))");
  out.push(
    "        self.register_buffer('channel_erev', torch.tensor(CHANNEL_EREV, dtype=dtype).view(1, -1, 1))",
  );
  out.push("        self.register_buffer('channel_mg', torch.tensor(CHANNEL_MG, dtype=dtype).view(1, -1, 1))");
  out.push(
    "        self.register_buffer('channel_norm', torch.tensor(CHANNEL_NORM, dtype=dtype).view(1, -1, 1))",
  );
  out.push(
    "        self.register_buffer('channel_dual', torch.tensor(CHANNEL_DUAL, dtype=torch.bool).view(1, -1, 1))",
  );
  out.push('');
  out.push('        ' + buffer('bias', model.neurons.map((n) => finite(n.bias, 0))));
  out.push('        ' + buffer('sigma', noise));
  out.push('        ' + buffer('v_rest', rest.map((r) => r.v)));
  out.push('        ' + buffer('w_rest', rest.map((r) => r.w)));
  out.push('        ' + buffer('m_rest', rest.map((r) => r.m)));
  out.push('        ' + buffer('h_rest', rest.map((r) => r.h)));
  out.push('        ' + buffer('n_rest', rest.map((r) => r.n)));
  out.push('');
  model.groups.forEach((group, index) => {
    out.push(`        # ${modelLabel(group.kind)}: slice [${group.offset}:${group.offset + group.size})`);
    for (const line of parameterBuffers(model, group, index)) out.push(`        ${line}`);
  });
  out.push('');
  out.push('        self.cursor = 0');
  out.push("        self.register_buffer('v', self.v_rest.clone())");
  out.push("        self.register_buffer('w_adapt', self.w_rest.clone())");
  out.push("        self.register_buffer('gate_m', self.m_rest.clone())");
  out.push("        self.register_buffer('gate_h', self.h_rest.clone())");
  out.push("        self.register_buffer('gate_n', self.n_rest.clone())");
  out.push("        self.register_buffer('refractory', torch.zeros(1, N_NEURONS, dtype=dtype))");
  out.push("        self.register_buffer('above', torch.zeros(1, N_NEURONS, dtype=dtype))");
  out.push(
    "        self.register_buffer('g_rise', torch.zeros(1, N_CHANNELS, N_NEURONS, dtype=dtype))",
  );
  out.push(
    "        self.register_buffer('g_decay', torch.zeros(1, N_CHANNELS, N_NEURONS, dtype=dtype))",
  );
  out.push("        self.register_buffer('spike_history', torch.zeros(HISTORY, 1, N_NEURONS, dtype=dtype))");
  out.push('');
  out.push('    def reset_state(self, batch_size=1, device=None, dtype=None):');
  out.push('        """Return every state variable to rest for a given batch size."""');
  out.push('        device = self.weight.device if device is None else device');
  out.push('        dtype = self.weight.dtype if dtype is None else dtype');
  out.push('');
  out.push('        def spread(source):');
  out.push('            return source.to(device=device, dtype=dtype).expand(batch_size, -1).clone()');
  out.push('');
  out.push('        self.v = spread(self.v_rest)');
  out.push('        self.w_adapt = spread(self.w_rest)');
  out.push('        self.gate_m = spread(self.m_rest)');
  out.push('        self.gate_h = spread(self.h_rest)');
  out.push('        self.gate_n = spread(self.n_rest)');
  out.push('        self.refractory = torch.zeros(batch_size, self.n, device=device, dtype=dtype)');
  out.push('        self.above = torch.zeros(batch_size, self.n, device=device, dtype=dtype)');
  out.push(
    '        self.g_rise = torch.zeros(batch_size, self.n_channels, self.n, device=device, dtype=dtype)',
  );
  out.push(
    '        self.g_decay = torch.zeros(batch_size, self.n_channels, self.n, device=device, dtype=dtype)',
  );
  out.push(
    '        self.spike_history = torch.zeros(self.history, batch_size, self.n, device=device, dtype=dtype)',
  );
  out.push('        self.cursor = 0');
  out.push('');
  out.push('    def weight_matrix(self):');
  out.push('        """Dense [post, pre] conductance matrix in nS."""');
  out.push('        return self.weight.detach().clone()');
  out.push('');
  out.push('    def forward(self, current=None, steps=None):');
  out.push('        """Run the circuit over a time dimension.');
  out.push('');
  out.push('        current: [T, N] or [T, B, N] injected current in pA. When omitted the');
  out.push("                 document's own stimuli are used for `steps` timesteps.");
  out.push('        Returns (spikes, voltages), both [T, B, N].');
  out.push('        """');
  out.push('        if current is None:');
  out.push('            n_steps = DEFAULT_STEPS if steps is None else int(steps)');
  out.push(
    '            current = build_stimulus(n_steps, dtype=self.weight.dtype, device=self.weight.device)',
  );
  out.push('        if current.dim() == 2:');
  out.push('            current = current.unsqueeze(1)');
  out.push('        n_steps = current.shape[0]');
  out.push('        batch = current.shape[1]');
  out.push('        if self.v.shape[0] != batch or self.v.device != current.device:');
  out.push('            self.reset_state(batch, device=current.device, dtype=current.dtype)');
  out.push('');
  out.push('        v = self.v');
  out.push('        w_adapt = self.w_adapt');
  out.push('        gate_m = self.gate_m');
  out.push('        gate_h = self.gate_h');
  out.push('        gate_n = self.gate_n');
  out.push('        refractory = self.refractory');
  out.push('        above = self.above');
  out.push('        g_rise = self.g_rise');
  out.push('        g_decay = self.g_decay');
  out.push('        history = list(self.spike_history.unbind(0))');
  out.push('        cursor = self.cursor');
  out.push('        zero_channel = torch.zeros(batch, self.n, dtype=v.dtype, device=v.device)');
  out.push('');
  out.push('        out_spikes = []');
  out.push('        out_voltage = []');
  out.push('        for step in range(n_steps):');
  if (channels.length > 0) {
    out.push('            arrivals = [zero_channel] * self.n_channels');
    out.push('            for block, (channel, delay) in enumerate(self.blocks):');
    out.push('                delayed = history[(cursor - delay) % self.history]');
    out.push('                contribution = delayed @ (self.weight * self.block_mask[block]).t()');
    out.push('                arrivals[channel] = arrivals[channel] + contribution');
    out.push('            raw = torch.stack(arrivals, dim=1) * self.channel_norm');
    out.push('            g_rise = g_rise * self.rise_decay + raw');
    out.push('            g_decay = g_decay * self.decay_decay + raw');
    out.push('            conductance = torch.where(self.channel_dual, g_decay - g_rise, g_decay)');
    out.push('            driving = self.channel_erev - v.unsqueeze(1)');
    out.push('            # Mg block is inert when its strength is zero.');
    out.push('            block_factor = 1.0 + self.channel_mg * torch.exp(-0.062 * v.unsqueeze(1)) / 3.57');
    out.push('            i_syn = (conductance * driving / block_factor).sum(dim=1)');
  } else {
    out.push('            i_syn = torch.zeros_like(v)');
  }
  out.push('            i_total = i_syn + current[step] + self.bias');
  if (hasNoise) {
    out.push('            i_total = i_total + self.sigma * torch.randn_like(i_total)');
  }
  out.push('');
  model.groups.forEach((group, index) => {
    for (const line of groupUpdate(group, index)) out.push(indentBlock(line, 12));
  });
  const names = model.groups.map((_, index) => `g${index}`);
  const cat = (prefix: string): string =>
    names.length === 1
      ? `${prefix}_${names[0]}`
      : `torch.cat([${names.map((n) => `${prefix}_${n}`).join(', ')}], dim=1)`;
  out.push('');
  out.push(`            spikes = ${cat('s')}`);
  out.push(`            v = ${cat('nv')}`);
  out.push(`            w_adapt = ${cat('nw')}`);
  out.push(`            gate_m = ${cat('nm')}`);
  out.push(`            gate_h = ${cat('nh')}`);
  out.push(`            gate_n = ${cat('nn')}`);
  out.push(`            refractory = ${cat('nr')}`);
  out.push(`            above = ${cat('na')}`);
  out.push('            history[cursor] = spikes');
  out.push('            cursor = (cursor + 1) % self.history');
  out.push('            out_spikes.append(spikes)');
  out.push('            out_voltage.append(v)');
  out.push('');
  out.push('        # State is carried across calls but detached, so gradients are');
  out.push('        # truncated at chunk boundaries rather than growing without bound.');
  out.push('        self.v = v.detach()');
  out.push('        self.w_adapt = w_adapt.detach()');
  out.push('        self.gate_m = gate_m.detach()');
  out.push('        self.gate_h = gate_h.detach()');
  out.push('        self.gate_n = gate_n.detach()');
  out.push('        self.refractory = refractory.detach()');
  out.push('        self.above = above.detach()');
  out.push('        self.g_rise = g_rise.detach()');
  out.push('        self.g_decay = g_decay.detach()');
  out.push('        self.spike_history = torch.stack(history).detach()');
  out.push('        self.cursor = cursor');
  out.push('        return torch.stack(out_spikes), torch.stack(out_voltage)');
  out.push('');
  out.push('');
  out.push("if __name__ == '__main__':");
  out.push('    net = NeuroForgeCircuit()');
  out.push('    net.reset_state(1)');
  out.push('    with torch.no_grad():');
  out.push('        spikes, voltages = net(steps=DEFAULT_STEPS)');
  out.push('    seconds = DEFAULT_STEPS * DT / 1000.0');
  out.push('    rate = spikes.sum().item() / max(N_NEURONS * seconds, 1e-9)');
  out.push(
    "    print('%d neurons, %d steps, %d spikes, %.2f Hz mean rate'" +
      ' % (N_NEURONS, DEFAULT_STEPS, int(spikes.sum().item()), rate))',
  );
  out.push("    print('voltage range: %.2f .. %.2f mV' % (voltages.min().item(), voltages.max().item()))");

  return `${out.join('\n')}\n`;
}
