/**
 * The attribute vocabulary of the query language.
 *
 * Every queryable property of a cell is declared exactly once, here: its name,
 * its type, the prose the help panel shows, and the accessor that reads it out
 * of the running network. Autocomplete, the operator type check in the parser
 * and the mask evaluation in `evaluate.ts` all read this one table, so a field
 * cannot exist in the grammar and be missing from the evaluator, or be offered
 * by the UI and rejected by the parser.
 *
 * Accessors take a `FieldSource` rather than raw buffers because three of the
 * fields are graph properties — degree, weight and receptor — that no column of
 * the SoA holds. The evaluator derives those once per query and hands them back
 * in, which is what keeps a clause O(cells) instead of O(cells · synapses).
 *
 * Reading is slot-indexed and allocation-free. String fields write into a
 * caller-owned array instead of returning one, because a query over fifty
 * thousand cells that allocated a string array per cell would spend more time in
 * the collector than in the search.
 *
 * A string field may hold several values — `receptor` is the one that does in
 * practice — so the matching rules are set rules: a positive operator matches
 * when *any* value satisfies it, a negative one when *none* does. A cell with no
 * value for a field (an unlabelled cell, a cell in no population) satisfies
 * every negative operator and no positive one, which is what makes `{missing}`
 * and `!=` agree with each other.
 */

import type { Circuit, Neuron, SimulationBuffers } from '@neuroforge/shared';
import { MODEL_FROM_CODE, NEURON_FLAG, RECEPTOR_FROM_CODE } from '@neuroforge/shared';

export type FieldType = 'string' | 'numeric' | 'boolean';

/**
 * Everything a field accessor may read.
 *
 * The derived columns are parallel to the neuron slots and are built once per
 * query evaluation. `receptorMask` packs one bit per `RECEPTOR_CODE`, so the
 * receptor field costs a shift rather than a set.
 */
export interface FieldSource {
  buffers: SimulationBuffers;
  /** The document the buffers were loaded from; slot `i` is `circuit.neurons[i]`. */
  circuit: Circuit;
  /** Neuron slots covered. */
  count: number;
  /** Enabled synapses arriving at each cell. Parallel synapses count separately. */
  inDegree: Uint32Array;
  outDegree: Uint32Array;
  /** Summed peak conductance arriving at / leaving each cell (nS). */
  weightIn: Float32Array;
  weightOut: Float32Array;
  /** Bit `RECEPTOR_CODE` set for each receptor on the cell's efferent synapses. */
  receptorMask: Uint8Array;
}

interface FieldMeta {
  /** Lowercase name written in a query. */
  name: string;
  /** One line for the help panel and the autocomplete row. */
  description: string;
}

export interface StringField extends FieldMeta {
  type: 'string';
  /**
   * Writes this cell's values into `out` and returns how many were written.
   * Zero means the cell has no value for the field, which is what `{missing}`
   * tests for.
   */
  readInto(source: FieldSource, slot: number, out: string[]): number;
  /** Upper bound on the return of `readInto`, so callers can size scratch. */
  maxValues: number;
}

export interface NumericField extends FieldMeta {
  type: 'numeric';
  /** Display unit, or an empty string when the quantity is dimensionless. */
  unit: string;
  read(source: FieldSource, slot: number): number;
}

export interface BooleanField extends FieldMeta {
  type: 'boolean';
  read(source: FieldSource, slot: number): boolean;
}

export type QueryField = StringField | NumericField | BooleanField;

/* ------------------------------------------------------------- accessors -- */

/**
 * The document record behind a slot, or null when the engine is running ahead
 * of the document. Returning null rather than throwing means a query issued
 * during the frame between a structural edit and the engine reload reports
 * "no value" for the few stale slots instead of failing outright.
 */
function neuronAt(source: FieldSource, slot: number): Neuron | null {
  const neurons = source.circuit.neurons;
  return slot < neurons.length ? neurons[slot] : null;
}

const RECEPTOR_COUNT = RECEPTOR_FROM_CODE.length;

/* ---------------------------------------------------------------- table --- */

const LABEL: StringField = {
  name: 'label',
  type: 'string',
  maxValues: 1,
  description: 'Human-facing cell name. Absent on cells that were never named.',
  readInto(source, slot, out) {
    const neuron = neuronAt(source, slot);
    if (neuron === null || neuron.label === '') return 0;
    out[0] = neuron.label;
    return 1;
  },
};

const ID: StringField = {
  name: 'id',
  type: 'string',
  maxValues: 1,
  description: 'Stable document identifier. Unique, and never reused.',
  readInto(source, slot, out) {
    const neuron = neuronAt(source, slot);
    if (neuron === null) return 0;
    out[0] = neuron.id;
    return 1;
  },
};

const MODEL: StringField = {
  name: 'model',
  type: 'string',
  maxValues: 1,
  description: 'Membrane model: lif, izhikevich, hodgkin-huxley, adex, morris-lecar.',
  readInto(source, slot, out) {
    const code = source.buffers.neurons.model[slot];
    if (code >= MODEL_FROM_CODE.length) return 0;
    out[0] = MODEL_FROM_CODE[code];
    return 1;
  },
};

const POLARITY: StringField = {
  name: 'polarity',
  type: 'string',
  maxValues: 1,
  description: 'excitatory or inhibitory.',
  readInto(source, slot, out) {
    out[0] = source.buffers.neurons.polarity[slot] === 1 ? 'inhibitory' : 'excitatory';
    return 1;
  },
};

const ARCHETYPE: StringField = {
  name: 'archetype',
  type: 'string',
  maxValues: 1,
  description: 'Morphology class: pyramidal, basket, granule, purkinje, stellate, motor, bipolar.',
  readInto(source, slot, out) {
    const neuron = neuronAt(source, slot);
    if (neuron === null) return 0;
    out[0] = neuron.morphology.archetype;
    return 1;
  },
};

const POPULATION: StringField = {
  name: 'population',
  type: 'string',
  maxValues: 1,
  description: 'Name of the owning population. Absent on cells placed individually.',
  readInto(source, slot, out) {
    const index = source.buffers.neurons.population[slot];
    const populations = source.circuit.populations;
    // 0xffff is the buffers' "no population" sentinel and is out of range here.
    if (index >= populations.length) return 0;
    const name = populations[index].name;
    if (name === '') return 0;
    out[0] = name;
    return 1;
  },
};

const RECEPTOR: StringField = {
  name: 'receptor',
  type: 'string',
  maxValues: RECEPTOR_COUNT,
  description:
    'Receptors this cell drives in its targets, read off its outgoing synapses. Multi-valued.',
  readInto(source, slot, out) {
    const mask = source.receptorMask[slot];
    if (mask === 0) return 0;
    let written = 0;
    for (let code = 0; code < RECEPTOR_COUNT; code += 1) {
      if ((mask & (1 << code)) === 0) continue;
      out[written] = RECEPTOR_FROM_CODE[code];
      written += 1;
    }
    return written;
  },
};

const RATE: NumericField = {
  name: 'rate',
  type: 'numeric',
  unit: 'Hz',
  description: 'Exponentially-smoothed firing rate, live off the simulation.',
  read: (source, slot) => source.buffers.neurons.rate[slot],
};

const VOLTAGE: NumericField = {
  name: 'voltage',
  type: 'numeric',
  unit: 'mV',
  description: 'Membrane potential at this instant.',
  read: (source, slot) => source.buffers.neurons.v[slot],
};

const DEGREE: NumericField = {
  name: 'degree',
  type: 'numeric',
  unit: '',
  description: 'Enabled synapses touching this cell, incoming and outgoing together.',
  read: (source, slot) => source.inDegree[slot] + source.outDegree[slot],
};

const IN_DEGREE: NumericField = {
  name: 'in_degree',
  type: 'numeric',
  unit: '',
  description: 'Enabled synapses arriving at this cell.',
  read: (source, slot) => source.inDegree[slot],
};

const OUT_DEGREE: NumericField = {
  name: 'out_degree',
  type: 'numeric',
  unit: '',
  description: 'Enabled synapses leaving this cell.',
  read: (source, slot) => source.outDegree[slot],
};

const WEIGHT_IN: NumericField = {
  name: 'weight_in',
  type: 'numeric',
  unit: 'nS',
  description: 'Summed peak conductance arriving at this cell.',
  read: (source, slot) => source.weightIn[slot],
};

const WEIGHT_OUT: NumericField = {
  name: 'weight_out',
  type: 'numeric',
  unit: 'nS',
  description: 'Summed peak conductance leaving this cell.',
  read: (source, slot) => source.weightOut[slot],
};

const BIAS: NumericField = {
  name: 'bias',
  type: 'numeric',
  unit: 'pA',
  description: 'Constant injected current.',
  read: (source, slot) => source.buffers.neurons.bias[slot],
};

const NOISE: NumericField = {
  name: 'noise',
  type: 'numeric',
  unit: 'pA',
  description: 'Per-cell input noise amplitude.',
  read: (source, slot) => source.buffers.neurons.noise[slot],
};

const X: NumericField = {
  name: 'x',
  type: 'numeric',
  unit: '',
  description: 'World position along X.',
  read: (source, slot) => source.buffers.neurons.position[slot * 3],
};

const Y: NumericField = {
  name: 'y',
  type: 'numeric',
  unit: '',
  description: 'World position along Y.',
  read: (source, slot) => source.buffers.neurons.position[slot * 3 + 1],
};

const Z: NumericField = {
  name: 'z',
  type: 'numeric',
  unit: '',
  description: 'World position along Z.',
  read: (source, slot) => source.buffers.neurons.position[slot * 3 + 2],
};

const SPIKES: NumericField = {
  name: 'spikes',
  type: 'numeric',
  unit: '',
  description: 'Spikes emitted since the last simulation reset.',
  read: (source, slot) => source.buffers.neurons.spikeCount[slot],
};

const ENABLED: BooleanField = {
  name: 'enabled',
  type: 'boolean',
  description: 'False for cells excluded from integration.',
  read: (source, slot) => source.buffers.neurons.enabled[slot] === 1,
};

const SELECTED: BooleanField = {
  name: 'selected',
  type: 'boolean',
  description: 'True for cells in the current selection.',
  read: (source, slot) => (source.buffers.neurons.flags[slot] & NEURON_FLAG.SELECTED) !== 0,
};

const SPIKING: BooleanField = {
  name: 'spiking',
  type: 'boolean',
  description: 'True for cells that fired on the most recent integration step.',
  read: (source, slot) => source.buffers.neurons.spike[slot] === 1,
};

/**
 * Every queryable attribute, in the order the help panel lists them: identity
 * first, then class, then live signals, then graph position, then geometry.
 */
export const QUERY_FIELDS: readonly QueryField[] = [
  LABEL,
  ID,
  MODEL,
  POLARITY,
  ARCHETYPE,
  POPULATION,
  RECEPTOR,
  RATE,
  VOLTAGE,
  SPIKES,
  DEGREE,
  IN_DEGREE,
  OUT_DEGREE,
  WEIGHT_IN,
  WEIGHT_OUT,
  BIAS,
  NOISE,
  X,
  Y,
  Z,
  ENABLED,
  SELECTED,
  SPIKING,
];

const BY_NAME = new Map<string, QueryField>(QUERY_FIELDS.map((field) => [field.name, field]));

export const QUERY_FIELD_NAMES: readonly string[] = QUERY_FIELDS.map((field) => field.name);

/** Widest `maxValues` in the table, so one scratch array serves every field. */
export const MAX_FIELD_VALUES: number = QUERY_FIELDS.reduce(
  (widest, field) => (field.type === 'string' ? Math.max(widest, field.maxValues) : widest),
  1,
);

/** Field named by `name`, case-insensitively, or null when there is no such field. */
export function findField(name: string): QueryField | null {
  return BY_NAME.get(name.toLowerCase()) ?? null;
}

/**
 * Whether a cell carries a value for a field at all.
 *
 * String fields answer by value count, numeric fields by finiteness — a cell the
 * engine has not integrated yet has a NaN rate, which is genuinely missing
 * rather than zero — and boolean fields always answer yes, because false is a
 * value.
 */
export function fieldPresent(
  field: QueryField,
  source: FieldSource,
  slot: number,
  scratch: string[],
): boolean {
  if (field.type === 'string') return field.readInto(source, slot, scratch) > 0;
  if (field.type === 'numeric') return Number.isFinite(field.read(source, slot));
  return true;
}
