import {
  MORPHOLOGY_ARCHETYPES,
  NEURON_MODEL_KINDS,
  PLASTICITY_KINDS,
  RECEPTOR_KINDS,
} from '@neuroforge/shared';
import type {
  Circuit,
  ConnectivityRule,
  NeuronModelKind,
  NeuronParams,
  NeuronPolarity,
  PlasticityKind,
  PopulationLayout,
  RenderSettings,
  SimulationSettings,
  StimulusPattern,
  Vec3,
} from '@neuroforge/shared';
import {
  asArray,
  asBoolean,
  asEnum,
  asFiniteNumber,
  asRecord,
  asString,
  asText,
  boundedInteger,
  boundedNumber,
  clamp,
  hashText,
} from './coerce';
import { declaredKind, sanitiseNeuronParams } from './params';
import {
  MAX_ACTIONS,
  MAX_POPULATION_SIZE,
  MAX_SYNAPSES_PER_PROJECTION,
  MAX_TOTAL_NEURONS,
  estimateSynapses,
} from './schema';
import type { AiPlan, CircuitAction, NamedProjectionSpec, PopulationSpec } from './types';

const MAX_NAME_LENGTH = 64;
const MAX_PROJECTION_NAME_LENGTH = 96;
const MAX_SUMMARY_LENGTH = 1200;
const MAX_WARNING_LENGTH = 400;
const MAX_WARNINGS = 32;
const MAX_EXPLICIT_POSITIONS = 4096;

const POLARITIES: readonly NeuronPolarity[] = ['excitatory', 'inhibitory'];
const INTEGRATORS: readonly SimulationSettings['integrator'][] = [
  'euler',
  'rk2',
  'rk4',
  'exponential-euler',
];
const BACKENDS: readonly SimulationSettings['backend'][] = ['auto', 'gpu', 'wasm', 'cpu'];
const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;

type NumericKeys<T> = { [K in keyof T]-?: T[K] extends number ? K : never }[keyof T];
type BooleanKeys<T> = { [K in keyof T]-?: T[K] extends boolean ? K : never }[keyof T];

interface NumericField<T> {
  key: NumericKeys<T>;
  min: number;
  max: number;
  integer?: boolean;
}

const SIMULATION_NUMBERS: readonly NumericField<SimulationSettings>[] = [
  { key: 'dt', min: 1e-4, max: 10 },
  { key: 'speed', min: 0.01, max: 100 },
  { key: 'gain', min: 0, max: 100 },
  { key: 'noise', min: 0, max: 10_000 },
  { key: 'seed', min: -2_147_483_648, max: 4_294_967_295, integer: true },
  { key: 'maxSubstepsPerFrame', min: 1, max: 1024, integer: true },
];

const RENDER_NUMBERS: readonly NumericField<RenderSettings>[] = [
  { key: 'bloomIntensity', min: 0, max: 10 },
  { key: 'bloomThreshold', min: 0, max: 2 },
  { key: 'bloomRadius', min: 0, max: 2 },
  { key: 'focusDistance', min: 0, max: 1 },
  { key: 'focalLength', min: 0, max: 1 },
  { key: 'bokehScale', min: 0, max: 20 },
  { key: 'fogDensity', min: 0, max: 1 },
  { key: 'aoIntensity', min: 0, max: 4 },
  { key: 'vignette', min: 0, max: 1 },
  { key: 'chromaticAberration', min: 0, max: 0.1 },
  { key: 'exposure', min: 0, max: 8 },
  { key: 'gridFade', min: 0, max: 1 },
  { key: 'particleDensity', min: 0, max: 4 },
  { key: 'neuronScale', min: 0.05, max: 10 },
];

const RENDER_BOOLEANS: readonly BooleanKeys<RenderSettings>[] = [
  'depthOfField',
  'ambientOcclusion',
  'gridVisible',
  'showDendrites',
  'showAxons',
  'showParticles',
  'voltageColoring',
];

interface KnownPopulation {
  name: string;
  size: number;
  model: NeuronModelKind;
}

/**
 * Sanitise a plan produced by a language model.
 *
 * Every field is treated as hostile: types are re-checked, numbers are clamped to
 * ranges the simulator can actually integrate, and any action naming a population
 * or projection that will not exist when it runs is dropped. The returned plan is
 * always safe to apply; `errors` explains everything that was changed or removed.
 */
export function validatePlan(plan: AiPlan, circuit: Circuit): { plan: AiPlan; errors: string[] } {
  const errors: string[] = [];
  const source = asRecord(plan);
  if (source === null) {
    return {
      plan: { summary: '', actions: [], warnings: [] },
      errors: ['The plan was not an object and could not be applied.'],
    };
  }

  const populations = new Map<string, KnownPopulation>();
  for (const population of circuit.populations) {
    populations.set(population.name.toLowerCase(), {
      name: population.name,
      size: population.size,
      model: population.params.kind,
    });
  }
  // Keyed by lower-cased name so a plan can address a projection in any casing,
  // and carrying the canonical name so the action it produces still matches the
  // document exactly — the applier resolves projections by name.
  const projections = new Map<string, string>();
  for (const projection of circuit.projections) {
    projections.set(projection.name.toLowerCase(), projection.name);
  }
  let neuronBudget = circuit.neurons.length;

  const parsedActions = asArray(source.actions);
  if (parsedActions === null && source.actions !== undefined) {
    errors.push('The plan had no usable actions array.');
  }
  const rawActions = parsedActions ?? [];
  if (rawActions.length > MAX_ACTIONS) {
    errors.push(`The plan contained ${rawActions.length} actions; only the first ${MAX_ACTIONS} were kept.`);
  }

  const actions: CircuitAction[] = [];
  const limit = Math.min(rawActions.length, MAX_ACTIONS);
  for (let i = 0; i < limit; i += 1) {
    const record = asRecord(rawActions[i]);
    const label = `Action ${i + 1}`;
    if (record === null) {
      errors.push(`${label} was not an object and was dropped.`);
      continue;
    }
    const type = asString(record.type);
    switch (type) {
      case 'create-population': {
        const result = validateCreatePopulation(record, label, populations, neuronBudget, errors);
        if (result === null) break;
        neuronBudget += result.spec.size;
        populations.set(result.spec.name.toLowerCase(), {
          name: result.spec.name,
          size: result.spec.size,
          model: result.spec.model,
        });
        actions.push(result);
        break;
      }
      case 'connect-populations': {
        const result = validateConnect(record, label, populations, projections, errors);
        if (result === null) break;
        projections.set(result.spec.name.toLowerCase(), result.spec.name);
        actions.push(result);
        break;
      }
      case 'set-simulation': {
        const patch = validateSimulationPatch(record.patch, label, errors);
        if (patch === null) break;
        actions.push({ type: 'set-simulation', patch });
        break;
      }
      case 'set-render': {
        const patch = validateRenderPatch(record.patch, label, errors);
        if (patch === null) break;
        actions.push({ type: 'set-render', patch });
        break;
      }
      case 'add-stimulus': {
        const result = validateStimulus(record, label, populations, errors);
        if (result === null) break;
        actions.push(result);
        break;
      }
      case 'tune-population': {
        const result = validateTunePopulation(record, label, populations, errors);
        if (result === null) break;
        actions.push(result);
        break;
      }
      case 'tune-projection': {
        const result = validateTuneProjection(record, label, projections, errors);
        if (result === null) break;
        actions.push(result);
        break;
      }
      case 'clear':
        populations.clear();
        projections.clear();
        neuronBudget = 0;
        actions.push({ type: 'clear' });
        break;
      default:
        errors.push(`${label} had an unknown type ${JSON.stringify(record.type)} and was dropped.`);
        break;
    }
  }

  const warnings: string[] = [];
  for (const raw of asArray(source.warnings) ?? []) {
    const text = asText(raw, MAX_WARNING_LENGTH);
    if (text !== null && warnings.length < MAX_WARNINGS) warnings.push(text);
  }

  return {
    plan: { summary: asText(source.summary, MAX_SUMMARY_LENGTH) ?? '', actions, warnings },
    errors,
  };
}

/**
 * A unique, length-bounded display name, suffixed when it collides. `taken` holds
 * lower-cased names; both the population map and the projection map are keyed
 * that way, so either can be passed without copying it into a set.
 */
function uniqueName(
  base: string,
  taken: { has(key: string): boolean },
  label: string,
  errors: string[],
): string {
  if (!taken.has(base.toLowerCase())) return base;
  for (let n = 2; ; n += 1) {
    const candidate = `${base} ${n}`;
    if (!taken.has(candidate.toLowerCase())) {
      errors.push(`${label} reused the name "${base}"; it was renamed to "${candidate}".`);
      return candidate;
    }
  }
}

function validateVec3(value: unknown): Vec3 | null {
  const record = asRecord(value);
  if (record === null) return null;
  const x = asFiniteNumber(record.x);
  const y = asFiniteNumber(record.y);
  const z = asFiniteNumber(record.z);
  if (x === null || y === null || z === null) return null;
  return { x: clamp(x, -1e6, 1e6), y: clamp(y, -1e6, 1e6), z: clamp(z, -1e6, 1e6) };
}

function validateLayout(value: unknown, size: number, seed: number): PopulationLayout | null {
  const record = asRecord(value);
  if (record === null) return null;
  switch (asString(record.kind)) {
    case 'grid': {
      const columns = boundedInteger(record.columns, 1, 512);
      const rows = boundedInteger(record.rows, 1, 512);
      const layers = boundedInteger(record.layers, 1, 512);
      const spacing = boundedNumber(record.spacing, 0.01, 500);
      if (columns === null || rows === null || layers === null || spacing === null) return null;
      return { kind: 'grid', columns, rows, layers, spacing };
    }
    case 'sphere': {
      const radius = boundedNumber(record.radius, 0.01, 500);
      if (radius === null) return null;
      return {
        kind: 'sphere',
        radius,
        jitter: boundedNumber(record.jitter, 0, 1) ?? 0,
        seed: boundedInteger(record.seed, 0, 0x7fffffff) ?? seed,
      };
    }
    case 'disc': {
      const radius = boundedNumber(record.radius, 0.01, 500);
      if (radius === null) return null;
      return {
        kind: 'disc',
        radius,
        thickness: boundedNumber(record.thickness, 0, 200) ?? 0,
        seed: boundedInteger(record.seed, 0, 0x7fffffff) ?? seed,
      };
    }
    case 'column': {
      const radius = boundedNumber(record.radius, 0.01, 500);
      const height = boundedNumber(record.height, 0.01, 1000);
      if (radius === null || height === null) return null;
      return {
        kind: 'column',
        radius,
        height,
        seed: boundedInteger(record.seed, 0, 0x7fffffff) ?? seed,
      };
    }
    case 'explicit': {
      const raw = asArray(record.positions);
      if (raw === null) return null;
      const positions: Vec3[] = [];
      for (const entry of raw.slice(0, Math.min(MAX_EXPLICIT_POSITIONS, size))) {
        const point = validateVec3(entry);
        if (point !== null) positions.push(point);
      }
      return positions.length === 0 ? null : { kind: 'explicit', positions };
    }
    default:
      return null;
  }
}

function validateCreatePopulation(
  record: Record<string, unknown>,
  label: string,
  populations: ReadonlyMap<string, KnownPopulation>,
  neuronBudget: number,
  errors: string[],
): { type: 'create-population'; spec: PopulationSpec } | null {
  const spec = asRecord(record.spec);
  if (spec === null) {
    errors.push(`${label} (create-population) had no spec and was dropped.`);
    return null;
  }
  const rawName = asText(spec.name, MAX_NAME_LENGTH);
  if (rawName === null) {
    errors.push(`${label} (create-population) had no usable name and was dropped.`);
    return null;
  }
  const rawSize = asFiniteNumber(spec.size);
  if (rawSize === null || rawSize < 1) {
    errors.push(
      `${label} (create-population) asked for a size of ${JSON.stringify(spec.size)}, which is not a positive number; it was dropped.`,
    );
    return null;
  }
  let size = Math.round(rawSize);
  if (size > MAX_POPULATION_SIZE) {
    errors.push(
      `${label} (create-population) asked for ${size} neurons; capped at ${MAX_POPULATION_SIZE}.`,
    );
    size = MAX_POPULATION_SIZE;
  }
  if (neuronBudget + size > MAX_TOTAL_NEURONS) {
    errors.push(
      `${label} (create-population) would take the circuit past the ${MAX_TOTAL_NEURONS} neuron limit; it was dropped.`,
    );
    return null;
  }

  const polarity = asEnum(spec.polarity, POLARITIES);
  if (polarity === null && spec.polarity !== undefined) {
    errors.push(`${label} (create-population) had an unknown polarity; it was treated as excitatory.`);
  }
  const model = asEnum(spec.model, NEURON_MODEL_KINDS);
  if (model === null && spec.model !== undefined) {
    errors.push(`${label} (create-population) named an unknown model; it was built as lif.`);
  }
  const resolvedModel = model ?? 'lif';

  const seed = hashText(rawName) % 0x7fffffff;
  let layout = validateLayout(spec.layout, size, seed);
  if (layout === null) {
    errors.push(
      `${label} (create-population) had an unusable layout; a sphere sized for ${size} neurons was used instead.`,
    );
    layout = { kind: 'sphere', radius: clamp(2.6 * Math.cbrt(size), 4, 60), jitter: 0.35, seed };
  }

  const name = uniqueName(rawName, populations, `${label} (create-population)`, errors);
  const out: PopulationSpec = {
    name,
    size,
    polarity: polarity ?? 'excitatory',
    model: resolvedModel,
    layout,
  };

  if (spec.params !== undefined && spec.params !== null) {
    const { params, rejected } = sanitiseNeuronParams(spec.params, resolvedModel);
    if (rejected.length > 0) {
      errors.push(
        `${label} (create-population) had parameters that do not belong to ${resolvedModel} or were not finite: ${rejected.join(', ')}.`,
      );
    }
    out.params = params;
  }
  const origin = validateVec3(spec.origin);
  if (origin !== null) out.origin = origin;
  const archetype = asEnum(spec.archetype, MORPHOLOGY_ARCHETYPES);
  if (archetype !== null) out.archetype = archetype;
  const color = asString(spec.color);
  out.color = color !== null && HEX_COLOR.test(color) ? color : null;

  return { type: 'create-population', spec: out };
}

function validateRule(value: unknown, seed: number): ConnectivityRule | null {
  const record = asRecord(value);
  if (record === null) return null;
  const kind = asString(record.kind);
  switch (kind) {
    case 'all-to-all':
      return { kind: 'all-to-all', selfConnections: asBoolean(record.selfConnections) ?? false };
    case 'random': {
      const probability = boundedNumber(record.probability, 0, 1);
      if (probability === null) return null;
      return {
        kind: 'random',
        probability,
        seed: boundedInteger(record.seed, 0, 0x7fffffff) ?? seed,
        selfConnections: asBoolean(record.selfConnections) ?? false,
      };
    }
    case 'one-to-one':
      return { kind: 'one-to-one' };
    case 'gaussian': {
      const sigma = boundedNumber(record.sigma, 0.01, 1e4);
      const maxProbability = boundedNumber(record.maxProbability, 0, 1);
      if (sigma === null || maxProbability === null) return null;
      return {
        kind: 'gaussian',
        sigma,
        maxProbability,
        seed: boundedInteger(record.seed, 0, 0x7fffffff) ?? seed,
      };
    }
    case 'distance-threshold': {
      const radius = boundedNumber(record.radius, 0.01, 1e4);
      const probability = boundedNumber(record.probability, 0, 1);
      if (radius === null || probability === null) return null;
      return {
        kind: 'distance-threshold',
        radius,
        probability,
        seed: boundedInteger(record.seed, 0, 0x7fffffff) ?? seed,
      };
    }
    case 'fixed-in-degree':
    case 'fixed-out-degree': {
      const degree = boundedInteger(record.degree, 1, 10_000);
      if (degree === null) return null;
      const ruleSeed = boundedInteger(record.seed, 0, 0x7fffffff) ?? seed;
      return kind === 'fixed-in-degree'
        ? { kind: 'fixed-in-degree', degree, seed: ruleSeed }
        : { kind: 'fixed-out-degree', degree, seed: ruleSeed };
    }
    default:
      return null;
  }
}

function validateConnect(
  record: Record<string, unknown>,
  label: string,
  populations: ReadonlyMap<string, KnownPopulation>,
  projections: ReadonlyMap<string, string>,
  errors: string[],
): { type: 'connect-populations'; spec: NamedProjectionSpec } | null {
  const spec = asRecord(record.spec);
  if (spec === null) {
    errors.push(`${label} (connect-populations) had no spec and was dropped.`);
    return null;
  }
  const sourceKey = (asText(spec.sourceName, MAX_NAME_LENGTH) ?? '').toLowerCase();
  const targetKey = (asText(spec.targetName, MAX_NAME_LENGTH) ?? '').toLowerCase();
  const source = populations.get(sourceKey);
  const target = populations.get(targetKey);
  if (source === undefined || target === undefined) {
    const missing = source === undefined ? spec.sourceName : spec.targetName;
    errors.push(
      `${label} (connect-populations) referenced the unknown population ${JSON.stringify(missing)}; it was dropped.`,
    );
    return null;
  }

  const seed = hashText(`${source.name}->${target.name}`) % 0x7fffffff;
  const rule = validateRule(spec.rule, seed);
  if (rule === null) {
    errors.push(`${label} (connect-populations) had an unusable connectivity rule; it was dropped.`);
    return null;
  }
  const estimate = estimateSynapses(rule, source.size, target.size);
  if (estimate > MAX_SYNAPSES_PER_PROJECTION) {
    errors.push(
      `${label} (connect-populations) would create about ${Math.round(estimate)} synapses, past the ${MAX_SYNAPSES_PER_PROJECTION} limit for one projection; it was dropped.`,
    );
    return null;
  }

  const name = uniqueName(
    asText(spec.name, MAX_PROJECTION_NAME_LENGTH) ?? `${source.name} -> ${target.name}`,
    projections,
    `${label} (connect-populations)`,
    errors,
  );
  const out: NamedProjectionSpec = {
    name,
    sourceName: source.name,
    targetName: target.name,
    rule,
  };
  const receptor = asEnum(spec.receptor, RECEPTOR_KINDS);
  if (receptor !== null) out.receptor = receptor;
  const plasticity = asEnum(spec.plasticity, PLASTICITY_KINDS);
  if (plasticity !== null) out.plasticity = plasticity;
  const weightMean = boundedNumber(spec.weightMean, 0, 1000);
  if (weightMean !== null) out.weightMean = weightMean;
  const weightJitter = boundedNumber(spec.weightJitter, 0, 1000);
  if (weightJitter !== null) out.weightJitter = weightJitter;
  const delayMean = boundedNumber(spec.delayMean, 0, 1000);
  if (delayMean !== null) out.delayMean = delayMean;
  const delayJitter = boundedNumber(spec.delayJitter, 0, 1000);
  if (delayJitter !== null) out.delayJitter = delayJitter;
  return { type: 'connect-populations', spec: out };
}

function validateSimulationPatch(
  value: unknown,
  label: string,
  errors: string[],
): Partial<SimulationSettings> | null {
  const record = asRecord(value);
  if (record === null) {
    errors.push(`${label} (set-simulation) had no patch and was dropped.`);
    return null;
  }
  const patch: Partial<SimulationSettings> = {};
  let changed = false;
  for (const field of SIMULATION_NUMBERS) {
    if (record[field.key] === undefined) continue;
    const value2 = field.integer
      ? boundedInteger(record[field.key], field.min, field.max)
      : boundedNumber(record[field.key], field.min, field.max);
    if (value2 === null) {
      errors.push(`${label} (set-simulation) gave a non-finite ${field.key}; it was ignored.`);
      continue;
    }
    patch[field.key] = value2;
    changed = true;
  }
  const integrator = asEnum(record.integrator, INTEGRATORS);
  if (integrator !== null) {
    patch.integrator = integrator;
    changed = true;
  }
  const backend = asEnum(record.backend, BACKENDS);
  if (backend !== null) {
    patch.backend = backend;
    changed = true;
  }
  const plasticityEnabled = asBoolean(record.plasticityEnabled);
  if (plasticityEnabled !== null) {
    patch.plasticityEnabled = plasticityEnabled;
    changed = true;
  }
  if (!changed) {
    errors.push(`${label} (set-simulation) contained no settings this build understands; it was dropped.`);
    return null;
  }
  return patch;
}

function validateRenderPatch(
  value: unknown,
  label: string,
  errors: string[],
): Partial<RenderSettings> | null {
  const record = asRecord(value);
  if (record === null) {
    errors.push(`${label} (set-render) had no patch and was dropped.`);
    return null;
  }
  const patch: Partial<RenderSettings> = {};
  let changed = false;
  for (const field of RENDER_NUMBERS) {
    if (record[field.key] === undefined) continue;
    const numeric = boundedNumber(record[field.key], field.min, field.max);
    if (numeric === null) {
      errors.push(`${label} (set-render) gave a non-finite ${field.key}; it was ignored.`);
      continue;
    }
    patch[field.key] = numeric;
    changed = true;
  }
  for (const key of RENDER_BOOLEANS) {
    const flag = asBoolean(record[key]);
    if (flag === null) continue;
    patch[key] = flag;
    changed = true;
  }
  if (!changed) {
    errors.push(`${label} (set-render) contained no settings this build understands; it was dropped.`);
    return null;
  }
  return patch;
}

function validateStimulusPattern(value: unknown, seed: number): StimulusPattern | null {
  const record = asRecord(value);
  if (record === null) return null;
  const amplitude = boundedNumber(record.amplitude, -1e5, 1e5);
  switch (asString(record.kind)) {
    case 'constant':
      return amplitude === null ? null : { kind: 'constant', amplitude };
    case 'step': {
      const duration = boundedNumber(record.duration, 0.01, 1e6);
      if (amplitude === null || duration === null) return null;
      return {
        kind: 'step',
        amplitude,
        start: boundedNumber(record.start, 0, 1e6) ?? 0,
        duration,
      };
    }
    case 'pulse-train': {
      const frequency = boundedNumber(record.frequency, 0.01, 1000);
      const width = boundedNumber(record.width, 0.01, 1e4);
      if (amplitude === null || frequency === null || width === null) return null;
      return {
        kind: 'pulse-train',
        amplitude,
        frequency,
        width,
        start: boundedNumber(record.start, 0, 1e6) ?? 0,
      };
    }
    case 'sine': {
      const frequency = boundedNumber(record.frequency, 0.01, 1000);
      if (amplitude === null || frequency === null) return null;
      return {
        kind: 'sine',
        amplitude,
        frequency,
        offset: boundedNumber(record.offset, -1e5, 1e5) ?? 0,
      };
    }
    case 'poisson': {
      const rate = boundedNumber(record.rate, 0, 10_000);
      if (amplitude === null || rate === null) return null;
      return {
        kind: 'poisson',
        rate,
        amplitude,
        seed: boundedInteger(record.seed, 0, 0x7fffffff) ?? seed,
      };
    }
    case 'ramp': {
      const from = boundedNumber(record.from, -1e5, 1e5);
      const to = boundedNumber(record.to, -1e5, 1e5);
      const duration = boundedNumber(record.duration, 0.01, 1e6);
      if (from === null || to === null || duration === null) return null;
      return { kind: 'ramp', from, to, start: boundedNumber(record.start, 0, 1e6) ?? 0, duration };
    }
    default:
      return null;
  }
}

function validateStimulus(
  record: Record<string, unknown>,
  label: string,
  populations: ReadonlyMap<string, KnownPopulation>,
  errors: string[],
): { type: 'add-stimulus'; targetPopulation: string; pattern: StimulusPattern; name: string } | null {
  const targetKey = (asText(record.targetPopulation, MAX_NAME_LENGTH) ?? '').toLowerCase();
  const target = populations.get(targetKey);
  if (target === undefined) {
    errors.push(
      `${label} (add-stimulus) targeted the unknown population ${JSON.stringify(record.targetPopulation)}; it was dropped.`,
    );
    return null;
  }
  const pattern = validateStimulusPattern(record.pattern, hashText(target.name) % 0x7fffffff);
  if (pattern === null) {
    errors.push(`${label} (add-stimulus) had an unusable pattern; it was dropped.`);
    return null;
  }
  return {
    type: 'add-stimulus',
    targetPopulation: target.name,
    pattern,
    name: asText(record.name, MAX_NAME_LENGTH) ?? `${pattern.kind} drive`,
  };
}

function validateTunePopulation(
  record: Record<string, unknown>,
  label: string,
  populations: ReadonlyMap<string, KnownPopulation>,
  errors: string[],
): {
  type: 'tune-population';
  name: string;
  params: Partial<NeuronParams>;
  bias?: number;
  noise?: number;
} | null {
  const key = (asText(record.name, MAX_NAME_LENGTH) ?? '').toLowerCase();
  const population = populations.get(key);
  if (population === undefined) {
    errors.push(
      `${label} (tune-population) referenced the unknown population ${JSON.stringify(record.name)}; it was dropped.`,
    );
    return null;
  }
  const kind = declaredKind(record.params) ?? population.model;
  const { params, rejected } = sanitiseNeuronParams(record.params, kind);
  if (rejected.length > 0) {
    errors.push(
      `${label} (tune-population) had parameters that do not belong to ${kind} or were not finite: ${rejected.join(', ')}.`,
    );
  }
  const out: {
    type: 'tune-population';
    name: string;
    params: Partial<NeuronParams>;
    bias?: number;
    noise?: number;
  } = { type: 'tune-population', name: population.name, params };
  if (record.bias !== undefined) {
    const bias = boundedNumber(record.bias, -1e5, 1e5);
    if (bias === null) errors.push(`${label} (tune-population) gave a non-finite bias; it was ignored.`);
    else out.bias = bias;
  }
  if (record.noise !== undefined) {
    const noise = boundedNumber(record.noise, 0, 1e4);
    if (noise === null) errors.push(`${label} (tune-population) gave a non-finite noise; it was ignored.`);
    else out.noise = noise;
  }
  return out;
}

function validateTuneProjection(
  record: Record<string, unknown>,
  label: string,
  projections: ReadonlyMap<string, string>,
  errors: string[],
): {
  type: 'tune-projection';
  name: string;
  weightMean?: number;
  delayMean?: number;
  plasticity?: PlasticityKind;
} | null {
  const raw = asText(record.name, MAX_PROJECTION_NAME_LENGTH);
  // The action carries the document's own spelling, not the model's: the applier
  // looks projections up by name, and a case-folded name would not be found.
  const canonical = raw === null ? undefined : projections.get(raw.toLowerCase());
  if (canonical === undefined) {
    errors.push(
      `${label} (tune-projection) referenced the unknown projection ${JSON.stringify(record.name)}; it was dropped.`,
    );
    return null;
  }
  const out: {
    type: 'tune-projection';
    name: string;
    weightMean?: number;
    delayMean?: number;
    plasticity?: PlasticityKind;
  } = { type: 'tune-projection', name: canonical };
  let changed = false;
  if (record.weightMean !== undefined) {
    const weightMean = boundedNumber(record.weightMean, 0, 1000);
    if (weightMean === null) {
      errors.push(`${label} (tune-projection) gave a non-finite weightMean; it was ignored.`);
    } else {
      out.weightMean = weightMean;
      changed = true;
    }
  }
  if (record.delayMean !== undefined) {
    const delayMean = boundedNumber(record.delayMean, 0, 1000);
    if (delayMean === null) {
      errors.push(`${label} (tune-projection) gave a non-finite delayMean; it was ignored.`);
    } else {
      out.delayMean = delayMean;
      changed = true;
    }
  }
  const plasticity = asEnum(record.plasticity, PLASTICITY_KINDS);
  if (plasticity !== null) {
    out.plasticity = plasticity;
    changed = true;
  }
  if (!changed) {
    errors.push(`${label} (tune-projection) asked for no change and was dropped.`);
    return null;
  }
  return out;
}
