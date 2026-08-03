/**
 * ONNX exporter.
 *
 * This writes a real ONNX protobuf: `ModelProto` / `GraphProto` / `NodeProto` /
 * `TensorProto` / `ValueInfoProto` are serialised field by field with the
 * hand-rolled wire-format writer in `protobuf.ts`, so the result is a byte
 * sequence `onnx.load()` parses without any external dependency here.
 *
 * The graph is a surrogate-gradient-free forward LIF cell unrolled over a fixed
 * horizon: ONNX `Loop` would let the horizon be dynamic, but the body would
 * still need one `MatMul` per delay bin and a scan-output per state variable,
 * and the unrolled form stays readable in Netron. The horizon, the delay
 * binning and every approximation are stated in the model's `doc_string`.
 *
 *   spikes_in [T, N] --MatMul--> input drive
 *                     recurrent MatMul against the binned weight matrices
 *                     membrane decay, threshold via Greater + Cast
 *   -> spikes_out [T, N], voltage_out [T, N]
 */

import type { Circuit } from '@neuroforge/shared';

import {
  bannerLines,
  binDelays,
  delayToSteps,
  indexCircuit,
  lifEquivalent,
} from './common';
import type { ExportCircuit } from './common';
import { ProtoWriter, packFloat32, packInt64 } from './protobuf';

/** ONNX IR version 8 (ONNX 1.13) paired with opset 17. */
const IR_VERSION = 8;
const OPSET_VERSION = 17;

const ELEM_FLOAT = 1;
const ELEM_INT64 = 7;

const ATTR_INT = 2;

/** Total float elements the weight initialisers may occupy. */
const WEIGHT_ELEMENT_BUDGET = 4_000_000;
const MAX_DELAY_BINS = 8;
const MAX_NODES = 2600;
const MAX_HORIZON = 64;
const MIN_HORIZON = 8;

/* ------------------------------------------------------------------ */
/* Proto builders                                                      */
/* ------------------------------------------------------------------ */

function tensorProto(name: string, dims: readonly number[], elemType: number, raw: Uint8Array): ProtoWriter {
  const w = new ProtoWriter(raw.length + 64);
  for (const dim of dims) w.varint(1, dim);
  w.varint(2, elemType);
  w.string(8, name);
  w.bytes(9, raw);
  return w;
}

function floatTensor(name: string, dims: readonly number[], values: ArrayLike<number>): ProtoWriter {
  return tensorProto(name, dims, ELEM_FLOAT, packFloat32(values));
}

function int64Tensor(name: string, dims: readonly number[], values: readonly number[]): ProtoWriter {
  return tensorProto(name, dims, ELEM_INT64, packInt64(values));
}

function tensorTypeProto(elemType: number, dims: readonly number[]): ProtoWriter {
  const shape = new ProtoWriter(32);
  for (const dim of dims) {
    const dimension = new ProtoWriter(16);
    dimension.varint(1, dim);
    shape.message(1, dimension);
  }
  const tensor = new ProtoWriter(64);
  tensor.varint(1, elemType);
  tensor.message(2, shape);
  const type = new ProtoWriter(80);
  type.message(1, tensor);
  return type;
}

function valueInfo(name: string, elemType: number, dims: readonly number[]): ProtoWriter {
  const w = new ProtoWriter(128);
  w.string(1, name);
  w.message(2, tensorTypeProto(elemType, dims));
  return w;
}

function attributeInt(name: string, value: number): ProtoWriter {
  const w = new ProtoWriter(32);
  w.string(1, name);
  w.varint(3, value);
  w.varint(20, ATTR_INT);
  return w;
}

function nodeProto(
  opType: string,
  inputs: readonly string[],
  outputs: readonly string[],
  name: string,
  attributes: readonly ProtoWriter[] = [],
): ProtoWriter {
  const w = new ProtoWriter(128);
  for (const input of inputs) w.string(1, input);
  for (const output of outputs) w.string(2, output);
  w.string(3, name);
  w.string(4, opType);
  for (const attribute of attributes) w.message(5, attribute);
  return w;
}

/** Accumulates a graph so nodes stay in topological order. */
class GraphBuilder {
  readonly nodes: ProtoWriter[] = [];
  readonly initializers: ProtoWriter[] = [];
  #counter = 0;

  add(
    opType: string,
    inputs: readonly string[],
    output: string,
    attributes: readonly ProtoWriter[] = [],
  ): string {
    this.#counter += 1;
    this.nodes.push(nodeProto(opType, inputs, [output], `${opType}_${this.#counter}`, attributes));
    return output;
  }

  initializer(tensor: ProtoWriter): void {
    this.initializers.push(tensor);
  }

  build(name: string, inputs: readonly ProtoWriter[], outputs: readonly ProtoWriter[], docString: string): ProtoWriter {
    const graph = new ProtoWriter(4096);
    for (const node of this.nodes) graph.message(1, node);
    graph.string(2, name);
    for (const initializer of this.initializers) graph.message(5, initializer);
    graph.string(10, docString);
    for (const input of inputs) graph.message(11, input);
    for (const output of outputs) graph.message(12, output);
    return graph;
  }
}

function modelProto(graph: ProtoWriter, docString: string): Uint8Array {
  const opset = new ProtoWriter(32);
  opset.string(1, '');
  opset.varint(2, OPSET_VERSION);

  const model = new ProtoWriter(graph.length + 512);
  model.varint(1, IR_VERSION);
  model.string(2, 'neuroforge');
  model.string(3, '0.1.0');
  model.string(4, 'ai.neuroforge');
  model.varint(5, 1);
  model.string(6, docString);
  model.message(7, graph);
  model.message(8, opset);
  return model.toBytes();
}

/* ------------------------------------------------------------------ */
/* Graph construction                                                  */
/* ------------------------------------------------------------------ */

interface CellParameters {
  n: number;
  alpha: Float32Array;
  vRest: Float32Array;
  vThresh: Float32Array;
  vReset: Float32Array;
  biasTerm: Float32Array;
  /** One [N, N] pre-major matrix per delay bin, in mV per presynaptic spike. */
  weights: Float32Array[];
  /** Delay of each bin in integration steps. */
  delays: number[];
  /** Sum of every bin, used for the external input path. */
  external: Float32Array;
  quantised: boolean;
  approximated: boolean;
}

function buildCell(model: ExportCircuit, maxBins: number): CellParameters {
  const n = model.neurons.length;
  const alpha = new Float32Array(n);
  const vRest = new Float32Array(n);
  const vThresh = new Float32Array(n);
  const vReset = new Float32Array(n);
  const biasTerm = new Float32Array(n);
  const capacitance = new Float32Array(n);
  let approximated = false;

  for (let i = 0; i < n; i += 1) {
    const neuron = model.neurons[i];
    const cell = lifEquivalent(neuron.params);
    if (!cell.exact) approximated = true;
    const decay = Math.exp(-model.dt / Math.max(cell.tauM, 1e-6));
    alpha[i] = decay;
    vRest[i] = cell.vRest;
    vThresh[i] = cell.vThresh;
    vReset[i] = cell.vReset;
    capacitance[i] = cell.cm;
    const gL = cell.cm / Math.max(cell.tauM, 1e-6);
    biasTerm[i] = (neuron.bias * (1 - decay)) / Math.max(gL, 1e-9);
  }

  const steps = model.synapses.map((s) => delayToSteps(s.delay, model.dt));
  const binning = binDelays(steps, maxBins);
  const binCount = model.synapses.length === 0 ? 1 : binning.bins.length;
  const weights: Float32Array[] = [];
  for (let b = 0; b < binCount; b += 1) weights.push(new Float32Array(n * n));
  const external = new Float32Array(n * n);

  model.synapses.forEach((synapse, index) => {
    const channel = model.channels[synapse.channel];
    const post = synapse.post;
    // Charge delivered by the whole conductance waveform, divided by the
    // postsynaptic capacitance: the resulting jump is in mV per spike.
    const drive = channel.kinetics.eRev - vRest[post];
    const millivolts = (synapse.weight * channel.kernel.area * drive) / Math.max(capacitance[post], 1e-9);
    const bin = binning.assignment[index] ?? 0;
    weights[bin][synapse.pre * n + post] += millivolts;
    external[synapse.pre * n + post] += millivolts;
  });

  return {
    n,
    alpha,
    vRest,
    vThresh,
    vReset,
    biasTerm,
    weights,
    delays: model.synapses.length === 0 ? [1] : binning.bins,
    external,
    quantised: binning.quantised,
    approximated,
  };
}

function buildGraph(cell: CellParameters, horizon: number): GraphBuilder {
  const g = new GraphBuilder();
  const n = cell.n;

  g.initializer(floatTensor('alpha', [n], cell.alpha));
  g.initializer(floatTensor('v_rest', [n], cell.vRest));
  g.initializer(floatTensor('v_thresh', [n], cell.vThresh));
  g.initializer(floatTensor('v_reset', [n], cell.vReset));
  g.initializer(floatTensor('bias_term', [n], cell.biasTerm));
  g.initializer(floatTensor('one', [], [1]));
  g.initializer(int64Tensor('axis_zero', [1], [0]));
  g.initializer(floatTensor('W_ext', [n, n], cell.external));
  cell.weights.forEach((matrix, index) => {
    g.initializer(floatTensor(`W_delay_${index}`, [n, n], matrix));
  });
  for (let t = 0; t < horizon; t += 1) {
    g.initializer(int64Tensor(`step_${t}`, [], [t]));
  }

  const spikeNames: string[] = [];
  const voltageNames: string[] = [];
  let previousVoltage = 'v_rest';

  for (let t = 0; t < horizon; t += 1) {
    const x = g.add('Gather', ['spikes_in', `step_${t}`], `x_${t}`, [attributeInt('axis', 0)]);
    let accumulated = g.add('MatMul', [x, 'W_ext'], `drive_${t}`);
    cell.delays.forEach((delay, bin) => {
      const source = t - delay;
      if (source < 0) return;
      const recurrent = g.add('MatMul', [spikeNames[source], `W_delay_${bin}`], `rec_${t}_${bin}`);
      accumulated = g.add('Add', [accumulated, recurrent], `acc_${t}_${bin}`);
    });
    accumulated = g.add('Add', [accumulated, 'bias_term'], `input_${t}`);

    const offset = g.add('Sub', [previousVoltage, 'v_rest'], `offset_${t}`);
    const decayed = g.add('Mul', [offset, 'alpha'], `decayed_${t}`);
    const relaxed = g.add('Add', ['v_rest', decayed], `relaxed_${t}`);
    const charged = g.add('Add', [relaxed, accumulated], `charged_${t}`);

    const fired = g.add('Greater', [charged, 'v_thresh'], `fired_${t}`);
    const spikes = g.add('Cast', [fired], `spikes_${t}`, [attributeInt('to', ELEM_FLOAT)]);
    const keep = g.add('Sub', ['one', spikes], `keep_${t}`);
    const held = g.add('Mul', [charged, keep], `held_${t}`);
    const reset = g.add('Mul', ['v_reset', spikes], `reset_${t}`);
    const voltage = g.add('Add', [held, reset], `v_${t}`);

    spikeNames.push(spikes);
    voltageNames.push(voltage);
    previousVoltage = voltage;

    g.add('Unsqueeze', [spikes, 'axis_zero'], `spikes_row_${t}`);
    g.add('Unsqueeze', [voltage, 'axis_zero'], `v_row_${t}`);
  }

  g.add(
    'Concat',
    spikeNames.map((_, t) => `spikes_row_${t}`),
    'spikes_out',
    [attributeInt('axis', 0)],
  );
  g.add(
    'Concat',
    voltageNames.map((_, t) => `v_row_${t}`),
    'voltage_out',
    [attributeInt('axis', 0)],
  );
  return g;
}

/** Serialise the circuit as an ONNX model. */
export function exportOnnx(circuit: Circuit): Uint8Array {
  const model = indexCircuit(circuit);
  const n = model.neurons.length;

  if (n === 0) {
    const g = new GraphBuilder();
    g.add('Identity', ['spikes_in'], 'spikes_out');
    g.add('Identity', ['spikes_in'], 'voltage_out');
    const doc = [...bannerLines(model, 'ONNX'), '', 'The circuit has no enabled neurons.'].join('\n');
    const graph = g.build(
      'neuroforge_empty',
      [valueInfo('spikes_in', ELEM_FLOAT, [1, 0])],
      [valueInfo('spikes_out', ELEM_FLOAT, [1, 0]), valueInfo('voltage_out', ELEM_FLOAT, [1, 0])],
      doc,
    );
    return modelProto(graph, doc);
  }

  const maxBins = Math.max(1, Math.min(MAX_DELAY_BINS, Math.floor(WEIGHT_ELEMENT_BUDGET / (n * n))));
  const cell = buildCell(model, maxBins);
  const perStep = 12 + cell.delays.length * 2;
  const horizon = Math.max(MIN_HORIZON, Math.min(MAX_HORIZON, Math.floor(MAX_NODES / perStep)));

  const doc = [...bannerLines(model, 'ONNX')];
  doc.push(
    '',
    'Graph shape:',
    `  input   spikes_in    [${horizon}, ${n}]  presynaptic spikes driving the network`,
    `  output  spikes_out   [${horizon}, ${n}]  emitted spikes, 1.0 or 0.0`,
    `  output  voltage_out  [${horizon}, ${n}]  membrane potential in mV after reset`,
    '',
    'Construction:',
    `  The membrane recurrence is UNROLLED over a fixed horizon of ${horizon} timesteps of ` +
      `${model.dt} ms rather than expressed with a Loop node, because every delay bin needs its own ` +
      'MatMul inside the body and the unrolled form stays inspectable in a graph viewer. Feed the ' +
      'model in chunks of that many steps to run longer trials.',
    `  Conduction delays are grouped into ${cell.delays.length} bin(s) of ` +
      `${cell.delays.map((d) => `${d} step(s)`).join(', ')}; each bin owns one [N, N] weight ` +
      'initializer W_delay_i, and W_ext is their sum, used for the external input path.',
    '  A weight entry is the membrane depolarisation in mV produced by one presynaptic spike: ' +
      'peak conductance x waveform area x driving force at rest / postsynaptic capacitance. The ' +
      'synaptic waveform is therefore collapsed to a single-step impulse that preserves total charge.',
    '  Threshold crossing is Greater followed by Cast to float; the reset is the algebraic ' +
      'blend v*(1-s) + v_reset*s.',
  );
  if (cell.quantised) {
    doc.push(
      `  Delays were quantised: the circuit uses more than ${maxBins} distinct delays, so they were ` +
        'merged into the bins listed above.',
    );
  }
  if (cell.approximated) {
    doc.push(
      '  This circuit contains membrane models other than LIF. ONNX carries a single LIF cell, so ' +
        'those neurons are exported with their equivalent linear parameters (time constant from ' +
        'C/gL or from the linearisation of the model at rest, threshold and reset from the model). ' +
        'Use the Brian2, NEST or PyTorch export for their exact dynamics.',
    );
  }
  doc.push(
    '  Refractory periods, adaptation currents, short-term plasticity, stochastic release and ' +
      'background noise are not part of this graph.',
  );
  const docString = doc.join('\n');

  const graph = buildGraph(cell, horizon).build(
    'neuroforge_lif',
    [valueInfo('spikes_in', ELEM_FLOAT, [horizon, n])],
    [
      valueInfo('spikes_out', ELEM_FLOAT, [horizon, n]),
      valueInfo('voltage_out', ELEM_FLOAT, [horizon, n]),
    ],
    docString,
  );
  return modelProto(graph, docString);
}
