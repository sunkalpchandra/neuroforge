/**
 * Validation and migration of documents read from disk, an import dialog or an
 * older schema version.
 *
 * Nothing here trusts its input. Every field is checked; values that can be
 * repaired safely are coerced and reported, and anything that would leave a
 * structurally broken document (a missing identifier, a collection of the wrong
 * shape, a file from a future schema) is treated as fatal so the caller gets a
 * null circuit and a list of reasons instead of a half-built graph.
 */

import {
  CIRCUIT_SCHEMA_VERSION,
  DEFAULT_ADEX,
  DEFAULT_CAMERA,
  DEFAULT_HODGKIN_HUXLEY,
  DEFAULT_IZHIKEVICH,
  DEFAULT_LIF,
  DEFAULT_MORRIS_LECAR,
  DEFAULT_PLASTICITY,
  DEFAULT_RENDER_SETTINGS,
  DEFAULT_SIMULATION_SETTINGS,
  DEFAULT_STP,
  MORPHOLOGY_ARCHETYPES,
  NEURON_MODEL_KINDS,
  PLASTICITY_KINDS,
  RECEPTOR_DEFAULTS,
  RECEPTOR_KINDS,
  defaultMorphology,
  newCircuitId,
} from '@neuroforge/shared';
import type {
  AdExParams,
  CameraState,
  Circuit,
  CircuitId,
  ConnectivityRule,
  HodgkinHuxleyParams,
  IzhikevichParams,
  LifParams,
  Morphology,
  MorphologyArchetype,
  MorrisLecarParams,
  Neuron,
  NeuronId,
  NeuronModelKind,
  NeuronParams,
  NeuronPolarity,
  PlasticityConfig,
  PlasticityKind,
  Population,
  PopulationId,
  PopulationLayout,
  Probe,
  ProbeId,
  Projection,
  ReceptorKind,
  ReceptorKinetics,
  RenderSettings,
  ShortTermPlasticity,
  SimulationSettings,
  Stimulus,
  StimulusId,
  StimulusPattern,
  Synapse,
  SynapseId,
  Vec3,
} from '@neuroforge/shared';

const POLARITIES: readonly NeuronPolarity[] = ['excitatory', 'inhibitory'];
const INTEGRATORS: readonly SimulationSettings['integrator'][] = [
  'euler',
  'rk2',
  'rk4',
  'exponential-euler',
];
const BACKENDS: readonly SimulationSettings['backend'][] = ['auto', 'gpu', 'wasm', 'cpu'];
const CAMERA_MODES: readonly CameraState['mode'][] = ['orbit', 'fly', 'first-person', 'cinematic'];
const PROBE_SIGNALS: readonly Probe['signal'][] = [
  'voltage',
  'current',
  'conductance',
  'calcium',
  'adaptation',
  'spikes',
];
const LAYOUT_KINDS = ['grid', 'sphere', 'disc', 'column', 'explicit'] as const;
const STIMULUS_KINDS = ['constant', 'step', 'pulse-train', 'sine', 'poisson', 'ramp'] as const;
const RULE_KINDS = [
  'all-to-all',
  'random',
  'one-to-one',
  'gaussian',
  'distance-threshold',
  'fixed-in-degree',
  'fixed-out-degree',
] as const;

/** Collects human-readable problems and remembers whether any was unrecoverable. */
class Report {
  readonly messages: string[] = [];
  fatal = false;

  /** A value that was repaired; the document remains loadable. */
  repair(path: string, message: string): void {
    this.messages.push(`${path}: ${message}`);
  }

  /** A problem that makes the document unusable. */
  reject(path: string, message: string): void {
    this.messages.push(`${path}: ${message}`);
    this.fatal = true;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function describe(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'nothing';
  if (Array.isArray(value)) return 'an array';
  const type = typeof value;
  if (type === 'string') return `the string ${JSON.stringify(value)}`;
  if (type === 'number' || type === 'boolean') return String(value);
  return `a ${type}`;
}

interface NumberOptions {
  min?: number;
  max?: number;
  integer?: boolean;
  /** Report when the field is absent. Sections that are wholly optional pass false. */
  required?: boolean;
}

function num(
  report: Report,
  value: unknown,
  path: string,
  fallback: number,
  options: NumberOptions = {},
): number {
  const { min, max, integer, required = true } = options;
  let n: number;
  if (typeof value === 'number') {
    n = value;
  } else if (typeof value === 'string' && value.trim().length > 0 && Number.isFinite(Number(value))) {
    n = Number(value);
    report.repair(path, `expected a number, got a string; parsed as ${n}`);
  } else if (value === undefined || value === null) {
    if (required) report.repair(path, `missing; using ${fallback}`);
    return fallback;
  } else {
    report.repair(path, `expected a number, got ${describe(value)}; using ${fallback}`);
    return fallback;
  }
  if (!Number.isFinite(n)) {
    report.repair(path, `expected a finite number, got ${String(n)}; using ${fallback}`);
    return fallback;
  }
  if (integer && !Number.isInteger(n)) {
    const rounded = Math.round(n);
    report.repair(path, `expected an integer, got ${n}; rounded to ${rounded}`);
    n = rounded;
  }
  if (min !== undefined && n < min) {
    report.repair(path, `${n} is below the minimum ${min}; clamped`);
    n = min;
  }
  if (max !== undefined && n > max) {
    report.repair(path, `${n} is above the maximum ${max}; clamped`);
    n = max;
  }
  return n;
}

function str(
  report: Report,
  value: unknown,
  path: string,
  fallback: string,
  required = true,
): string {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) {
    if (required) report.repair(path, `missing; using ${JSON.stringify(fallback)}`);
    return fallback;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    const coerced = String(value);
    report.repair(path, `expected a string, got ${describe(value)}; using ${JSON.stringify(coerced)}`);
    return coerced;
  }
  report.repair(path, `expected a string, got ${describe(value)}; using ${JSON.stringify(fallback)}`);
  return fallback;
}

function bool(report: Report, value: unknown, path: string, fallback: boolean, required = true): boolean {
  if (typeof value === 'boolean') return value;
  if (value === undefined || value === null) {
    if (required) report.repair(path, `missing; using ${String(fallback)}`);
    return fallback;
  }
  if (value === 'true' || value === 'false') {
    report.repair(path, `expected a boolean, got a string; parsed as ${value}`);
    return value === 'true';
  }
  report.repair(path, `expected a boolean, got ${describe(value)}; using ${String(fallback)}`);
  return fallback;
}

function enumOf<T extends string>(
  report: Report,
  value: unknown,
  path: string,
  allowed: readonly T[],
  fallback: T,
): T {
  if (typeof value === 'string' && (allowed as readonly string[]).includes(value)) {
    return value as T;
  }
  if (value === undefined || value === null) {
    report.repair(path, `missing; using '${fallback}'`);
    return fallback;
  }
  report.repair(
    path,
    `expected one of ${allowed.map((a) => `'${a}'`).join(', ')}, got ${describe(value)}; using '${fallback}'`,
  );
  return fallback;
}

function vec3(report: Report, value: unknown, path: string, fallback: Vec3): Vec3 {
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected an {x,y,z} object, got ${describe(value)}; using the origin`);
    } else {
      report.repair(path, 'missing; using the origin');
    }
    return { ...fallback };
  }
  return {
    x: num(report, value.x, `${path}.x`, fallback.x),
    y: num(report, value.y, `${path}.y`, fallback.y),
    z: num(report, value.z, `${path}.z`, fallback.z),
  };
}

/**
 * Returns the array, or null after reporting. `required` distinguishes absent
 * from malformed; `fatal` marks the collections a document cannot be rebuilt
 * without, where the wrong type has to abort the load rather than be repaired.
 */
function arrayOf(
  report: Report,
  value: unknown,
  path: string,
  required: boolean,
  fatal: boolean,
): unknown[] | null {
  if (Array.isArray(value)) return value;
  if (value === undefined || value === null) {
    if (required) report.repair(path, 'missing; using an empty list');
    return null;
  }
  if (fatal) report.reject(path, `expected an array, got ${describe(value)}`);
  else report.repair(path, `expected an array, got ${describe(value)}; using an empty list`);
  return null;
}

function stringList(report: Report, value: unknown, path: string): string[] {
  const raw = arrayOf(report, value, path, false, false);
  if (!raw) return [];
  const out: string[] = [];
  raw.forEach((item, i) => {
    if (typeof item === 'string') out.push(item);
    else report.repair(`${path}[${i}]`, `expected a string, got ${describe(item)}; dropped`);
  });
  return out;
}

/* ------------------------------------------------------------------------ */
/* Neuron                                                                    */
/* ------------------------------------------------------------------------ */

function readParams(report: Report, value: unknown, path: string): NeuronParams {
  if (!isRecord(value)) {
    report.repair(path, `expected a parameter object, got ${describe(value)}; using LIF defaults`);
    return { ...DEFAULT_LIF };
  }
  const kind = enumOf<NeuronModelKind>(report, value.kind, `${path}.kind`, NEURON_MODEL_KINDS, 'lif');
  switch (kind) {
    case 'lif': {
      const params: LifParams = {
        kind: 'lif',
        cm: num(report, value.cm, `${path}.cm`, DEFAULT_LIF.cm, { min: 1e-6 }),
        gL: num(report, value.gL, `${path}.gL`, DEFAULT_LIF.gL, { min: 1e-9 }),
        eL: num(report, value.eL, `${path}.eL`, DEFAULT_LIF.eL),
        vThresh: num(report, value.vThresh, `${path}.vThresh`, DEFAULT_LIF.vThresh),
        vReset: num(report, value.vReset, `${path}.vReset`, DEFAULT_LIF.vReset),
        tRefract: num(report, value.tRefract, `${path}.tRefract`, DEFAULT_LIF.tRefract, { min: 0 }),
      };
      return params;
    }
    case 'izhikevich': {
      const params: IzhikevichParams = {
        kind: 'izhikevich',
        a: num(report, value.a, `${path}.a`, DEFAULT_IZHIKEVICH.a),
        b: num(report, value.b, `${path}.b`, DEFAULT_IZHIKEVICH.b),
        c: num(report, value.c, `${path}.c`, DEFAULT_IZHIKEVICH.c),
        d: num(report, value.d, `${path}.d`, DEFAULT_IZHIKEVICH.d),
        vPeak: num(report, value.vPeak, `${path}.vPeak`, DEFAULT_IZHIKEVICH.vPeak),
        iScale: num(report, value.iScale, `${path}.iScale`, DEFAULT_IZHIKEVICH.iScale),
      };
      return params;
    }
    case 'hodgkin-huxley': {
      const params: HodgkinHuxleyParams = {
        kind: 'hodgkin-huxley',
        cm: num(report, value.cm, `${path}.cm`, DEFAULT_HODGKIN_HUXLEY.cm, { min: 1e-6 }),
        gNa: num(report, value.gNa, `${path}.gNa`, DEFAULT_HODGKIN_HUXLEY.gNa, { min: 0 }),
        gK: num(report, value.gK, `${path}.gK`, DEFAULT_HODGKIN_HUXLEY.gK, { min: 0 }),
        gL: num(report, value.gL, `${path}.gL`, DEFAULT_HODGKIN_HUXLEY.gL, { min: 0 }),
        eNa: num(report, value.eNa, `${path}.eNa`, DEFAULT_HODGKIN_HUXLEY.eNa),
        eK: num(report, value.eK, `${path}.eK`, DEFAULT_HODGKIN_HUXLEY.eK),
        eL: num(report, value.eL, `${path}.eL`, DEFAULT_HODGKIN_HUXLEY.eL),
        vDetect: num(report, value.vDetect, `${path}.vDetect`, DEFAULT_HODGKIN_HUXLEY.vDetect),
        q10: num(report, value.q10, `${path}.q10`, DEFAULT_HODGKIN_HUXLEY.q10, { min: 1e-6 }),
      };
      return params;
    }
    case 'adex': {
      const params: AdExParams = {
        kind: 'adex',
        cm: num(report, value.cm, `${path}.cm`, DEFAULT_ADEX.cm, { min: 1e-6 }),
        gL: num(report, value.gL, `${path}.gL`, DEFAULT_ADEX.gL, { min: 1e-9 }),
        eL: num(report, value.eL, `${path}.eL`, DEFAULT_ADEX.eL),
        deltaT: num(report, value.deltaT, `${path}.deltaT`, DEFAULT_ADEX.deltaT, { min: 1e-6 }),
        vT: num(report, value.vT, `${path}.vT`, DEFAULT_ADEX.vT),
        vPeak: num(report, value.vPeak, `${path}.vPeak`, DEFAULT_ADEX.vPeak),
        vReset: num(report, value.vReset, `${path}.vReset`, DEFAULT_ADEX.vReset),
        a: num(report, value.a, `${path}.a`, DEFAULT_ADEX.a),
        b: num(report, value.b, `${path}.b`, DEFAULT_ADEX.b),
        tauW: num(report, value.tauW, `${path}.tauW`, DEFAULT_ADEX.tauW, { min: 1e-6 }),
        tRefract: num(report, value.tRefract, `${path}.tRefract`, DEFAULT_ADEX.tRefract, { min: 0 }),
      };
      return params;
    }
    case 'morris-lecar': {
      const params: MorrisLecarParams = {
        kind: 'morris-lecar',
        cm: num(report, value.cm, `${path}.cm`, DEFAULT_MORRIS_LECAR.cm, { min: 1e-6 }),
        gCa: num(report, value.gCa, `${path}.gCa`, DEFAULT_MORRIS_LECAR.gCa, { min: 0 }),
        gK: num(report, value.gK, `${path}.gK`, DEFAULT_MORRIS_LECAR.gK, { min: 0 }),
        gL: num(report, value.gL, `${path}.gL`, DEFAULT_MORRIS_LECAR.gL, { min: 0 }),
        eCa: num(report, value.eCa, `${path}.eCa`, DEFAULT_MORRIS_LECAR.eCa),
        eK: num(report, value.eK, `${path}.eK`, DEFAULT_MORRIS_LECAR.eK),
        eL: num(report, value.eL, `${path}.eL`, DEFAULT_MORRIS_LECAR.eL),
        v1: num(report, value.v1, `${path}.v1`, DEFAULT_MORRIS_LECAR.v1),
        v2: num(report, value.v2, `${path}.v2`, DEFAULT_MORRIS_LECAR.v2, { min: 1e-6 }),
        v3: num(report, value.v3, `${path}.v3`, DEFAULT_MORRIS_LECAR.v3),
        v4: num(report, value.v4, `${path}.v4`, DEFAULT_MORRIS_LECAR.v4, { min: 1e-6 }),
        phi: num(report, value.phi, `${path}.phi`, DEFAULT_MORRIS_LECAR.phi, { min: 0 }),
        vDetect: num(report, value.vDetect, `${path}.vDetect`, DEFAULT_MORRIS_LECAR.vDetect),
      };
      return params;
    }
  }
}

function readMorphology(report: Report, value: unknown, path: string): Morphology {
  const base = defaultMorphology();
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected a morphology object, got ${describe(value)}; using defaults`);
    } else {
      report.repair(path, 'missing; using the pyramidal default');
    }
    return base;
  }
  const archetype = enumOf<MorphologyArchetype>(
    report,
    value.archetype,
    `${path}.archetype`,
    MORPHOLOGY_ARCHETYPES,
    base.archetype,
  );
  const preset = defaultMorphology(archetype);
  return {
    seed: num(report, value.seed, `${path}.seed`, preset.seed, { integer: true, min: 0 }),
    somaRadius: num(report, value.somaRadius, `${path}.somaRadius`, preset.somaRadius, { min: 1e-3 }),
    dendriteCount: num(report, value.dendriteCount, `${path}.dendriteCount`, preset.dendriteCount, {
      integer: true,
      min: 0,
      max: 64,
    }),
    dendriteDepth: num(report, value.dendriteDepth, `${path}.dendriteDepth`, preset.dendriteDepth, {
      integer: true,
      min: 0,
      max: 12,
    }),
    dendriteLength: num(report, value.dendriteLength, `${path}.dendriteLength`, preset.dendriteLength, {
      min: 0,
    }),
    dendriteTaper: num(report, value.dendriteTaper, `${path}.dendriteTaper`, preset.dendriteTaper, {
      min: 0,
      max: 1,
    }),
    dendriteSpread: num(report, value.dendriteSpread, `${path}.dendriteSpread`, preset.dendriteSpread, {
      min: 0,
      max: Math.PI,
    }),
    axonLength: num(report, value.axonLength, `${path}.axonLength`, preset.axonLength, { min: 0 }),
    axonTerminals: num(report, value.axonTerminals, `${path}.axonTerminals`, preset.axonTerminals, {
      integer: true,
      min: 0,
      max: 64,
    }),
    scale: num(report, value.scale, `${path}.scale`, preset.scale, { min: 1e-3 }),
    archetype,
  };
}

function readNeuron(report: Report, value: unknown, path: string): Neuron | null {
  if (!isRecord(value)) {
    report.reject(path, `expected a neuron object, got ${describe(value)}`);
    return null;
  }
  if (typeof value.id !== 'string' || value.id.length === 0) {
    report.reject(`${path}.id`, 'a neuron must carry a non-empty string id');
    return null;
  }
  const population =
    typeof value.population === 'string' && value.population.length > 0
      ? (value.population as PopulationId)
      : null;
  if (value.population !== undefined && value.population !== null && population === null) {
    report.repair(`${path}.population`, `expected a population id, got ${describe(value.population)}; cleared`);
  }
  return {
    id: value.id as NeuronId,
    label: str(report, value.label, `${path}.label`, '', false),
    position: vec3(report, value.position, `${path}.position`, { x: 0, y: 0, z: 0 }),
    params: readParams(report, value.params, `${path}.params`),
    polarity: enumOf<NeuronPolarity>(report, value.polarity, `${path}.polarity`, POLARITIES, 'excitatory'),
    morphology: readMorphology(report, value.morphology, `${path}.morphology`),
    population,
    bias: num(report, value.bias, `${path}.bias`, 0),
    noise: num(report, value.noise, `${path}.noise`, 0, { min: 0 }),
    enabled: bool(report, value.enabled, `${path}.enabled`, true, false),
  };
}

/* ------------------------------------------------------------------------ */
/* Synapse                                                                   */
/* ------------------------------------------------------------------------ */

function readKinetics(
  report: Report,
  value: unknown,
  path: string,
  receptor: ReceptorKind,
): ReceptorKinetics {
  const preset = RECEPTOR_DEFAULTS[receptor];
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected a kinetics object, got ${describe(value)}; using the ${receptor} defaults`);
    } else {
      report.repair(path, `missing; using the ${receptor} defaults`);
    }
    return { ...preset };
  }
  return {
    tauRise: num(report, value.tauRise, `${path}.tauRise`, preset.tauRise, { min: 0 }),
    tauDecay: num(report, value.tauDecay, `${path}.tauDecay`, preset.tauDecay, { min: 1e-4 }),
    eRev: num(report, value.eRev, `${path}.eRev`, preset.eRev),
    mgBlock: num(report, value.mgBlock, `${path}.mgBlock`, preset.mgBlock, { min: 0 }),
  };
}

function readPlasticity(report: Report, value: unknown, path: string): PlasticityConfig {
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected a plasticity object, got ${describe(value)}; using static defaults`);
    } else {
      report.repair(path, 'missing; using static defaults');
    }
    return { ...DEFAULT_PLASTICITY };
  }
  const config: PlasticityConfig = {
    kind: enumOf<PlasticityKind>(report, value.kind, `${path}.kind`, PLASTICITY_KINDS, 'static'),
    aPlus: num(report, value.aPlus, `${path}.aPlus`, DEFAULT_PLASTICITY.aPlus),
    aMinus: num(report, value.aMinus, `${path}.aMinus`, DEFAULT_PLASTICITY.aMinus),
    tauPlus: num(report, value.tauPlus, `${path}.tauPlus`, DEFAULT_PLASTICITY.tauPlus, { min: 1e-4 }),
    tauMinus: num(report, value.tauMinus, `${path}.tauMinus`, DEFAULT_PLASTICITY.tauMinus, { min: 1e-4 }),
    tauX: num(report, value.tauX, `${path}.tauX`, DEFAULT_PLASTICITY.tauX, { min: 1e-4 }),
    tauY: num(report, value.tauY, `${path}.tauY`, DEFAULT_PLASTICITY.tauY, { min: 1e-4 }),
    wMin: num(report, value.wMin, `${path}.wMin`, DEFAULT_PLASTICITY.wMin),
    wMax: num(report, value.wMax, `${path}.wMax`, DEFAULT_PLASTICITY.wMax),
    learningRate: num(report, value.learningRate, `${path}.learningRate`, DEFAULT_PLASTICITY.learningRate, {
      min: 0,
    }),
  };
  if (config.wMax < config.wMin) {
    report.repair(`${path}.wMax`, `upper bound ${config.wMax} is below the lower bound ${config.wMin}; swapped`);
    const min = config.wMax;
    config.wMax = config.wMin;
    config.wMin = min;
  }
  return config;
}

function readStp(report: Report, value: unknown, path: string): ShortTermPlasticity {
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected a short-term plasticity object, got ${describe(value)}; disabled`);
    }
    return { ...DEFAULT_STP };
  }
  return {
    enabled: bool(report, value.enabled, `${path}.enabled`, DEFAULT_STP.enabled, false),
    u: num(report, value.u, `${path}.u`, DEFAULT_STP.u, { min: 0, max: 1 }),
    tauRec: num(report, value.tauRec, `${path}.tauRec`, DEFAULT_STP.tauRec, { min: 1e-4 }),
    tauFacil: num(report, value.tauFacil, `${path}.tauFacil`, DEFAULT_STP.tauFacil, { min: 0 }),
  };
}

function readSynapse(report: Report, value: unknown, path: string): Synapse | null {
  if (!isRecord(value)) {
    report.reject(path, `expected a synapse object, got ${describe(value)}`);
    return null;
  }
  if (typeof value.id !== 'string' || value.id.length === 0) {
    report.reject(`${path}.id`, 'a synapse must carry a non-empty string id');
    return null;
  }
  if (typeof value.source !== 'string' || value.source.length === 0) {
    report.reject(`${path}.source`, 'a synapse must reference a source neuron id');
    return null;
  }
  if (typeof value.target !== 'string' || value.target.length === 0) {
    report.reject(`${path}.target`, 'a synapse must reference a target neuron id');
    return null;
  }
  const receptor = enumOf<ReceptorKind>(report, value.receptor, `${path}.receptor`, RECEPTOR_KINDS, 'ampa');
  return {
    id: value.id as SynapseId,
    source: value.source as NeuronId,
    target: value.target as NeuronId,
    receptor,
    weight: num(report, value.weight, `${path}.weight`, 1),
    delay: num(report, value.delay, `${path}.delay`, 1, { min: 0 }),
    kinetics: readKinetics(report, value.kinetics, `${path}.kinetics`, receptor),
    plasticity: readPlasticity(report, value.plasticity, `${path}.plasticity`),
    stp: readStp(report, value.stp, `${path}.stp`),
    releaseProbability: num(report, value.releaseProbability, `${path}.releaseProbability`, 1, {
      min: 0,
      max: 1,
    }),
    arc: num(report, value.arc, `${path}.arc`, 0),
    enabled: bool(report, value.enabled, `${path}.enabled`, true, false),
  };
}

/* ------------------------------------------------------------------------ */
/* Populations, projections, stimuli, probes                                 */
/* ------------------------------------------------------------------------ */

function readLayout(report: Report, value: unknown, path: string): PopulationLayout {
  const fallback: PopulationLayout = { kind: 'sphere', radius: 20, jitter: 0, seed: 1 };
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected a layout object, got ${describe(value)}; using a sphere layout`);
    } else {
      report.repair(path, 'missing; using a sphere layout');
    }
    return fallback;
  }
  const kind = enumOf(report, value.kind, `${path}.kind`, LAYOUT_KINDS, 'sphere');
  switch (kind) {
    case 'grid':
      return {
        kind: 'grid',
        columns: num(report, value.columns, `${path}.columns`, 1, { integer: true, min: 1 }),
        rows: num(report, value.rows, `${path}.rows`, 1, { integer: true, min: 1 }),
        layers: num(report, value.layers, `${path}.layers`, 1, { integer: true, min: 1 }),
        spacing: num(report, value.spacing, `${path}.spacing`, 4, { min: 0 }),
      };
    case 'sphere':
      return {
        kind: 'sphere',
        radius: num(report, value.radius, `${path}.radius`, 20, { min: 0 }),
        jitter: num(report, value.jitter, `${path}.jitter`, 0, { min: 0 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
    case 'disc':
      return {
        kind: 'disc',
        radius: num(report, value.radius, `${path}.radius`, 20, { min: 0 }),
        thickness: num(report, value.thickness, `${path}.thickness`, 2, { min: 0 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
    case 'column':
      return {
        kind: 'column',
        radius: num(report, value.radius, `${path}.radius`, 10, { min: 0 }),
        height: num(report, value.height, `${path}.height`, 30, { min: 0 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
    case 'explicit': {
      const raw = arrayOf(report, value.positions, `${path}.positions`, true, false);
      const positions: Vec3[] = [];
      if (raw) {
        raw.forEach((item, i) => {
          positions.push(vec3(report, item, `${path}.positions[${i}]`, { x: 0, y: 0, z: 0 }));
        });
      }
      return { kind: 'explicit', positions };
    }
  }
}

function readPopulation(
  report: Report,
  value: unknown,
  path: string,
  knownNeurons: ReadonlySet<string>,
): Population | null {
  if (!isRecord(value)) {
    report.reject(path, `expected a population object, got ${describe(value)}`);
    return null;
  }
  if (typeof value.id !== 'string' || value.id.length === 0) {
    report.reject(`${path}.id`, 'a population must carry a non-empty string id');
    return null;
  }
  const rawMembers = stringList(report, value.members, `${path}.members`);
  const members: NeuronId[] = [];
  let missing = 0;
  for (const id of rawMembers) {
    if (knownNeurons.has(id)) members.push(id as NeuronId);
    else missing += 1;
  }
  if (missing > 0) {
    report.repair(`${path}.members`, `${missing} member id(s) do not match any neuron; dropped`);
  }
  const color =
    typeof value.color === 'string' && value.color.length > 0 ? value.color : null;
  if (value.color !== undefined && value.color !== null && color === null) {
    report.repair(`${path}.color`, `expected a hex colour string, got ${describe(value.color)}; cleared`);
  }
  return {
    id: value.id as PopulationId,
    name: str(report, value.name, `${path}.name`, 'population'),
    size: num(report, value.size, `${path}.size`, members.length, { integer: true, min: 0 }),
    polarity: enumOf<NeuronPolarity>(report, value.polarity, `${path}.polarity`, POLARITIES, 'excitatory'),
    params: readParams(report, value.params, `${path}.params`),
    morphology: readMorphology(report, value.morphology, `${path}.morphology`),
    layout: readLayout(report, value.layout, `${path}.layout`),
    origin: vec3(report, value.origin, `${path}.origin`, { x: 0, y: 0, z: 0 }),
    color,
    members,
    collapsed: bool(report, value.collapsed, `${path}.collapsed`, false, false),
  };
}

function readRule(report: Report, value: unknown, path: string): ConnectivityRule {
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected a connectivity rule, got ${describe(value)}; using all-to-all`);
    } else {
      report.repair(path, 'missing; using all-to-all');
    }
    return { kind: 'all-to-all', selfConnections: false };
  }
  const kind = enumOf(report, value.kind, `${path}.kind`, RULE_KINDS, 'all-to-all');
  switch (kind) {
    case 'all-to-all':
      return {
        kind: 'all-to-all',
        selfConnections: bool(report, value.selfConnections, `${path}.selfConnections`, false, false),
      };
    case 'random':
      return {
        kind: 'random',
        probability: num(report, value.probability, `${path}.probability`, 0.1, { min: 0, max: 1 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
        selfConnections: bool(report, value.selfConnections, `${path}.selfConnections`, false, false),
      };
    case 'one-to-one':
      return { kind: 'one-to-one' };
    case 'gaussian':
      return {
        kind: 'gaussian',
        sigma: num(report, value.sigma, `${path}.sigma`, 10, { min: 1e-6 }),
        maxProbability: num(report, value.maxProbability, `${path}.maxProbability`, 1, { min: 0, max: 1 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
    case 'distance-threshold':
      return {
        kind: 'distance-threshold',
        radius: num(report, value.radius, `${path}.radius`, 10, { min: 0 }),
        probability: num(report, value.probability, `${path}.probability`, 1, { min: 0, max: 1 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
    case 'fixed-in-degree':
      return {
        kind: 'fixed-in-degree',
        degree: num(report, value.degree, `${path}.degree`, 1, { integer: true, min: 0 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
    case 'fixed-out-degree':
      return {
        kind: 'fixed-out-degree',
        degree: num(report, value.degree, `${path}.degree`, 1, { integer: true, min: 0 }),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
  }
}

function readProjection(
  report: Report,
  value: unknown,
  path: string,
  knownPopulations: ReadonlySet<string>,
): Projection | null {
  if (!isRecord(value)) {
    report.reject(path, `expected a projection object, got ${describe(value)}`);
    return null;
  }
  if (typeof value.id !== 'string' || value.id.length === 0) {
    report.reject(`${path}.id`, 'a projection must carry a non-empty string id');
    return null;
  }
  const source = str(report, value.source, `${path}.source`, '');
  const target = str(report, value.target, `${path}.target`, '');
  if (source.length === 0 || target.length === 0) {
    report.repair(path, 'projection has no source or target population; dropped');
    return null;
  }
  if (!knownPopulations.has(source) || !knownPopulations.has(target)) {
    report.repair(path, 'projection references a population that does not exist; dropped');
    return null;
  }
  return {
    id: value.id,
    name: str(report, value.name, `${path}.name`, 'projection'),
    source: source as PopulationId,
    target: target as PopulationId,
    rule: readRule(report, value.rule, `${path}.rule`),
    weightMean: num(report, value.weightMean, `${path}.weightMean`, 1),
    weightJitter: num(report, value.weightJitter, `${path}.weightJitter`, 0, { min: 0 }),
    delayMean: num(report, value.delayMean, `${path}.delayMean`, 1, { min: 0 }),
    delayJitter: num(report, value.delayJitter, `${path}.delayJitter`, 0, { min: 0 }),
  };
}

function readPattern(report: Report, value: unknown, path: string): StimulusPattern {
  if (!isRecord(value)) {
    if (value !== undefined && value !== null) {
      report.repair(path, `expected a stimulus pattern, got ${describe(value)}; using a zero constant`);
    } else {
      report.repair(path, 'missing; using a zero constant');
    }
    return { kind: 'constant', amplitude: 0 };
  }
  const kind = enumOf(report, value.kind, `${path}.kind`, STIMULUS_KINDS, 'constant');
  switch (kind) {
    case 'constant':
      return { kind: 'constant', amplitude: num(report, value.amplitude, `${path}.amplitude`, 0) };
    case 'step':
      return {
        kind: 'step',
        amplitude: num(report, value.amplitude, `${path}.amplitude`, 0),
        start: num(report, value.start, `${path}.start`, 0, { min: 0 }),
        duration: num(report, value.duration, `${path}.duration`, 100, { min: 0 }),
      };
    case 'pulse-train':
      return {
        kind: 'pulse-train',
        amplitude: num(report, value.amplitude, `${path}.amplitude`, 0),
        frequency: num(report, value.frequency, `${path}.frequency`, 10, { min: 1e-6 }),
        width: num(report, value.width, `${path}.width`, 1, { min: 0 }),
        start: num(report, value.start, `${path}.start`, 0, { min: 0 }),
      };
    case 'sine':
      return {
        kind: 'sine',
        amplitude: num(report, value.amplitude, `${path}.amplitude`, 0),
        frequency: num(report, value.frequency, `${path}.frequency`, 10, { min: 0 }),
        offset: num(report, value.offset, `${path}.offset`, 0),
      };
    case 'poisson':
      return {
        kind: 'poisson',
        rate: num(report, value.rate, `${path}.rate`, 10, { min: 0 }),
        amplitude: num(report, value.amplitude, `${path}.amplitude`, 0),
        seed: num(report, value.seed, `${path}.seed`, 1, { integer: true }),
      };
    case 'ramp':
      return {
        kind: 'ramp',
        from: num(report, value.from, `${path}.from`, 0),
        to: num(report, value.to, `${path}.to`, 0),
        start: num(report, value.start, `${path}.start`, 0, { min: 0 }),
        duration: num(report, value.duration, `${path}.duration`, 100, { min: 0 }),
      };
  }
}

function readStimulus(
  report: Report,
  value: unknown,
  path: string,
  knownNeurons: ReadonlySet<string>,
): Stimulus | null {
  if (!isRecord(value)) {
    report.reject(path, `expected a stimulus object, got ${describe(value)}`);
    return null;
  }
  if (typeof value.id !== 'string' || value.id.length === 0) {
    report.reject(`${path}.id`, 'a stimulus must carry a non-empty string id');
    return null;
  }
  const rawTargets = stringList(report, value.targets, `${path}.targets`);
  const targets: NeuronId[] = [];
  let missing = 0;
  for (const id of rawTargets) {
    if (knownNeurons.has(id)) targets.push(id as NeuronId);
    else missing += 1;
  }
  if (missing > 0) {
    report.repair(`${path}.targets`, `${missing} target id(s) do not match any neuron; dropped`);
  }
  return {
    id: value.id as StimulusId,
    name: str(report, value.name, `${path}.name`, 'stimulus'),
    targets,
    pattern: readPattern(report, value.pattern, `${path}.pattern`),
    enabled: bool(report, value.enabled, `${path}.enabled`, true, false),
  };
}

function readProbe(
  report: Report,
  value: unknown,
  path: string,
  knownNeurons: ReadonlySet<string>,
): Probe | null {
  if (!isRecord(value)) {
    report.reject(path, `expected a probe object, got ${describe(value)}`);
    return null;
  }
  if (typeof value.id !== 'string' || value.id.length === 0) {
    report.reject(`${path}.id`, 'a probe must carry a non-empty string id');
    return null;
  }
  const target = str(report, value.target, `${path}.target`, '');
  if (!knownNeurons.has(target)) {
    report.repair(path, 'probe references a neuron that does not exist; dropped');
    return null;
  }
  return {
    id: value.id as ProbeId,
    target: target as NeuronId,
    signal: enumOf<Probe['signal']>(report, value.signal, `${path}.signal`, PROBE_SIGNALS, 'voltage'),
    capacity: num(report, value.capacity, `${path}.capacity`, 2048, { integer: true, min: 1 }),
    color: str(report, value.color, `${path}.color`, '#4FD1FF', false),
    enabled: bool(report, value.enabled, `${path}.enabled`, true, false),
  };
}

/* ------------------------------------------------------------------------ */
/* Settings blocks                                                           */
/* ------------------------------------------------------------------------ */

function readSimulation(report: Report, value: unknown, path: string): SimulationSettings {
  if (!isRecord(value)) {
    report.repair(path, 'missing or malformed; using the default simulation settings');
    return { ...DEFAULT_SIMULATION_SETTINGS };
  }
  return {
    dt: num(report, value.dt, `${path}.dt`, DEFAULT_SIMULATION_SETTINGS.dt, { min: 1e-4, max: 10 }),
    integrator: enumOf(report, value.integrator, `${path}.integrator`, INTEGRATORS, DEFAULT_SIMULATION_SETTINGS.integrator),
    speed: num(report, value.speed, `${path}.speed`, DEFAULT_SIMULATION_SETTINGS.speed, { min: 0 }),
    gain: num(report, value.gain, `${path}.gain`, DEFAULT_SIMULATION_SETTINGS.gain),
    noise: num(report, value.noise, `${path}.noise`, DEFAULT_SIMULATION_SETTINGS.noise, { min: 0 }),
    seed: num(report, value.seed, `${path}.seed`, DEFAULT_SIMULATION_SETTINGS.seed, { integer: true }),
    plasticityEnabled: bool(
      report,
      value.plasticityEnabled,
      `${path}.plasticityEnabled`,
      DEFAULT_SIMULATION_SETTINGS.plasticityEnabled,
    ),
    maxSubstepsPerFrame: num(
      report,
      value.maxSubstepsPerFrame,
      `${path}.maxSubstepsPerFrame`,
      DEFAULT_SIMULATION_SETTINGS.maxSubstepsPerFrame,
      { integer: true, min: 1, max: 4096 },
    ),
    backend: enumOf(report, value.backend, `${path}.backend`, BACKENDS, DEFAULT_SIMULATION_SETTINGS.backend),
  };
}

function readCamera(report: Report, value: unknown, path: string): CameraState {
  if (!isRecord(value)) {
    report.repair(path, 'missing or malformed; using the default camera');
    return { ...DEFAULT_CAMERA, position: { ...DEFAULT_CAMERA.position }, target: { ...DEFAULT_CAMERA.target } };
  }
  return {
    position: vec3(report, value.position, `${path}.position`, DEFAULT_CAMERA.position),
    target: vec3(report, value.target, `${path}.target`, DEFAULT_CAMERA.target),
    fov: num(report, value.fov, `${path}.fov`, DEFAULT_CAMERA.fov, { min: 1, max: 179 }),
    mode: enumOf(report, value.mode, `${path}.mode`, CAMERA_MODES, DEFAULT_CAMERA.mode),
  };
}

function readRender(report: Report, value: unknown, path: string): RenderSettings {
  if (!isRecord(value)) {
    report.repair(path, 'missing or malformed; using the default render settings');
    return { ...DEFAULT_RENDER_SETTINGS };
  }
  const d = DEFAULT_RENDER_SETTINGS;
  return {
    bloomIntensity: num(report, value.bloomIntensity, `${path}.bloomIntensity`, d.bloomIntensity, { min: 0 }),
    bloomThreshold: num(report, value.bloomThreshold, `${path}.bloomThreshold`, d.bloomThreshold, { min: 0 }),
    bloomRadius: num(report, value.bloomRadius, `${path}.bloomRadius`, d.bloomRadius, { min: 0 }),
    depthOfField: bool(report, value.depthOfField, `${path}.depthOfField`, d.depthOfField),
    focusDistance: num(report, value.focusDistance, `${path}.focusDistance`, d.focusDistance, { min: 0 }),
    focalLength: num(report, value.focalLength, `${path}.focalLength`, d.focalLength, { min: 0 }),
    bokehScale: num(report, value.bokehScale, `${path}.bokehScale`, d.bokehScale, { min: 0 }),
    fogDensity: num(report, value.fogDensity, `${path}.fogDensity`, d.fogDensity, { min: 0 }),
    ambientOcclusion: bool(report, value.ambientOcclusion, `${path}.ambientOcclusion`, d.ambientOcclusion),
    aoIntensity: num(report, value.aoIntensity, `${path}.aoIntensity`, d.aoIntensity, { min: 0 }),
    vignette: num(report, value.vignette, `${path}.vignette`, d.vignette, { min: 0 }),
    chromaticAberration: num(
      report,
      value.chromaticAberration,
      `${path}.chromaticAberration`,
      d.chromaticAberration,
      { min: 0 },
    ),
    exposure: num(report, value.exposure, `${path}.exposure`, d.exposure, { min: 0 }),
    gridVisible: bool(report, value.gridVisible, `${path}.gridVisible`, d.gridVisible),
    gridFade: num(report, value.gridFade, `${path}.gridFade`, d.gridFade, { min: 0 }),
    showDendrites: bool(report, value.showDendrites, `${path}.showDendrites`, d.showDendrites),
    showAxons: bool(report, value.showAxons, `${path}.showAxons`, d.showAxons),
    showParticles: bool(report, value.showParticles, `${path}.showParticles`, d.showParticles),
    particleDensity: num(report, value.particleDensity, `${path}.particleDensity`, d.particleDensity, { min: 0 }),
    neuronScale: num(report, value.neuronScale, `${path}.neuronScale`, d.neuronScale, { min: 1e-3 }),
    voltageColoring: bool(report, value.voltageColoring, `${path}.voltageColoring`, d.voltageColoring),
  };
}

/* ------------------------------------------------------------------------ */
/* Entry point                                                               */
/* ------------------------------------------------------------------------ */

/**
 * Validate and migrate a parsed document into a `Circuit`.
 *
 * Returns `circuit: null` whenever a problem was unrecoverable. `errors` is
 * always populated with every problem found, including the recoverable ones, so
 * the caller can show the user exactly what was changed on the way in.
 */
export function migrateCircuit(raw: unknown): { circuit: Circuit | null; errors: string[] } {
  const report = new Report();

  // A document may be wrapped in an export envelope; unwrap it transparently.
  let source: unknown = raw;
  if (isRecord(raw) && !('neurons' in raw) && isRecord(raw.circuit)) {
    source = raw.circuit;
  }

  if (!isRecord(source)) {
    report.reject('document', `expected an object, got ${describe(source)}`);
    return { circuit: null, errors: report.messages };
  }

  const version = num(report, source.version, 'version', CIRCUIT_SCHEMA_VERSION, {
    integer: true,
    min: 0,
    required: false,
  });
  if (version > CIRCUIT_SCHEMA_VERSION) {
    report.reject(
      'version',
      `document uses schema version ${version} but this build understands at most ${CIRCUIT_SCHEMA_VERSION}`,
    );
    return { circuit: null, errors: report.messages };
  }

  const rawNeurons = arrayOf(report, source.neurons, 'neurons', true, true);
  if (report.fatal) return { circuit: null, errors: report.messages };

  const neurons: Neuron[] = [];
  const seenNeurons = new Set<string>();
  if (rawNeurons) {
    rawNeurons.forEach((item, i) => {
      const neuron = readNeuron(report, item, `neurons[${i}]`);
      if (!neuron) return;
      if (seenNeurons.has(neuron.id)) {
        report.repair(`neurons[${i}].id`, `duplicate neuron id '${neuron.id}'; the later record is dropped`);
        return;
      }
      seenNeurons.add(neuron.id);
      neurons.push(neuron);
    });
  }

  const rawSynapses = arrayOf(report, source.synapses, 'synapses', true, true);
  const synapses: Synapse[] = [];
  const seenSynapses = new Set<string>();
  if (rawSynapses) {
    rawSynapses.forEach((item, i) => {
      const synapse = readSynapse(report, item, `synapses[${i}]`);
      if (!synapse) return;
      if (seenSynapses.has(synapse.id)) {
        report.repair(`synapses[${i}].id`, `duplicate synapse id '${synapse.id}'; the later record is dropped`);
        return;
      }
      if (!seenNeurons.has(synapse.source) || !seenNeurons.has(synapse.target)) {
        report.repair(
          `synapses[${i}]`,
          `references a neuron that is not in the document (${synapse.source} -> ${synapse.target}); dropped`,
        );
        return;
      }
      seenSynapses.add(synapse.id);
      synapses.push(synapse);
    });
  }

  const rawPopulations = arrayOf(report, source.populations, 'populations', false, false);
  const populations: Population[] = [];
  const seenPopulations = new Set<string>();
  if (rawPopulations) {
    rawPopulations.forEach((item, i) => {
      const population = readPopulation(report, item, `populations[${i}]`, seenNeurons);
      if (!population) return;
      if (seenPopulations.has(population.id)) {
        report.repair(
          `populations[${i}].id`,
          `duplicate population id '${population.id}'; the later record is dropped`,
        );
        return;
      }
      seenPopulations.add(population.id);
      populations.push(population);
    });
  }

  // A neuron may only claim membership of a population that survived validation.
  for (let i = 0; i < neurons.length; i += 1) {
    const population = neurons[i].population;
    if (population !== null && !seenPopulations.has(population)) {
      report.repair(`neurons[${i}].population`, `population '${population}' does not exist; cleared`);
      neurons[i] = { ...neurons[i], population: null };
    }
  }

  const rawProjections = arrayOf(report, source.projections, 'projections', false, false);
  const projections: Projection[] = [];
  if (rawProjections) {
    rawProjections.forEach((item, i) => {
      const projection = readProjection(report, item, `projections[${i}]`, seenPopulations);
      if (projection) projections.push(projection);
    });
  }

  const rawStimuli = arrayOf(report, source.stimuli, 'stimuli', false, false);
  const stimuli: Stimulus[] = [];
  if (rawStimuli) {
    rawStimuli.forEach((item, i) => {
      const stimulus = readStimulus(report, item, `stimuli[${i}]`, seenNeurons);
      if (stimulus) stimuli.push(stimulus);
    });
  }

  const rawProbes = arrayOf(report, source.probes, 'probes', false, false);
  const probes: Probe[] = [];
  if (rawProbes) {
    rawProbes.forEach((item, i) => {
      const probe = readProbe(report, item, `probes[${i}]`, seenNeurons);
      if (probe) probes.push(probe);
    });
  }

  if (report.fatal) return { circuit: null, errors: report.messages };

  const now = Date.now();
  const hasId = typeof source.id === 'string' && source.id.length > 0;
  const id = hasId ? (source.id as CircuitId) : newCircuitId();
  if (!hasId) report.repair('id', 'missing or malformed; a new id was minted');

  const createdAt = num(report, source.createdAt, 'createdAt', now, { min: 0, required: false });
  const circuit: Circuit = {
    id,
    name: str(report, source.name, 'name', 'Untitled circuit', false),
    description: str(report, source.description, 'description', '', false),
    version: CIRCUIT_SCHEMA_VERSION,
    createdAt,
    updatedAt: num(report, source.updatedAt, 'updatedAt', createdAt, { min: 0, required: false }),
    neurons,
    synapses,
    populations,
    projections,
    stimuli,
    probes,
    simulation: readSimulation(report, source.simulation, 'simulation'),
    camera: readCamera(report, source.camera, 'camera'),
    render: readRender(report, source.render, 'render'),
    tags: stringList(report, source.tags, 'tags'),
  };

  if (version < CIRCUIT_SCHEMA_VERSION) {
    report.repair('version', `migrated from schema version ${version} to ${CIRCUIT_SCHEMA_VERSION}`);
  }

  return { circuit: report.fatal ? null : circuit, errors: report.messages };
}
