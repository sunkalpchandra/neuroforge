import { NEURON_MODEL_KINDS } from '@neuroforge/shared';
import type { NeuronModelKind, NeuronParams } from '@neuroforge/shared';
import { asEnum, asFiniteNumber, asRecord, clamp } from './coerce';

/**
 * The parameter names each membrane model owns, mirroring the field order of the
 * interfaces in `@neuroforge/shared/neuron`. Both the tool schema and the plan
 * validator are generated from this table so neither can drift from the other.
 */
export const PARAM_KEYS: Record<NeuronModelKind, readonly string[]> = {
  lif: ['cm', 'gL', 'eL', 'vThresh', 'vReset', 'tRefract'],
  izhikevich: ['a', 'b', 'c', 'd', 'vPeak', 'iScale'],
  'hodgkin-huxley': ['cm', 'gNa', 'gK', 'gL', 'eNa', 'eK', 'eL', 'vDetect', 'q10'],
  adex: ['cm', 'gL', 'eL', 'deltaT', 'vT', 'vPeak', 'vReset', 'a', 'b', 'tauW', 'tRefract'],
  'morris-lecar': [
    'cm',
    'gCa',
    'gK',
    'gL',
    'eCa',
    'eK',
    'eL',
    'v1',
    'v2',
    'v3',
    'v4',
    'phi',
    'vDetect',
  ],
};

/** Every parameter name any model accepts, sorted for a stable schema. */
export const ALL_PARAM_KEYS: readonly string[] = [
  ...new Set(NEURON_MODEL_KINDS.flatMap((kind) => PARAM_KEYS[kind])),
].sort();

/** No membrane parameter in this unit system legitimately exceeds this magnitude. */
const PARAM_LIMIT = 1e6;

export interface ParamSanitisation {
  params: Partial<NeuronParams>;
  /** Keys that were dropped because they are not part of `kind` or were not finite. */
  rejected: string[];
}

/**
 * Keep only the finite numeric fields that belong to `kind`, and stamp the kind
 * so the applier can tell a parameter edit from a model switch.
 */
export function sanitiseNeuronParams(value: unknown, kind: NeuronModelKind): ParamSanitisation {
  const source = asRecord(value);
  const rejected: string[] = [];
  const out: Record<string, unknown> = { kind };
  if (source === null) {
    if (value !== undefined && value !== null) rejected.push('params');
    return { params: out as Partial<NeuronParams>, rejected };
  }
  const allowed = PARAM_KEYS[kind];
  for (const key of Object.keys(source)) {
    if (key === 'kind') continue;
    if (!allowed.includes(key)) {
      rejected.push(key);
      continue;
    }
    const n = asFiniteNumber(source[key]);
    if (n === null) {
      rejected.push(key);
      continue;
    }
    out[key] = clamp(n, -PARAM_LIMIT, PARAM_LIMIT);
  }
  // Every retained key is drawn from `kind`'s own field list, which is what makes
  // the assertion to the discriminated union sound.
  return { params: out as Partial<NeuronParams>, rejected };
}

/** The model kind a partial parameter object declares, if it declares one. */
export function declaredKind(value: unknown): NeuronModelKind | null {
  const record = asRecord(value);
  return record === null ? null : asEnum(record.kind, NEURON_MODEL_KINDS);
}
