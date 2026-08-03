/**
 * The canonical on-disk document format.
 *
 * Serialisation writes every key in a fixed order so that two exports of the
 * same document differ only in the `exportedAt` stamp, which makes the files
 * diffable in version control. Parsing runs the payload through the same
 * validator used for IndexedDB reads, so a hand-edited file cannot smuggle a
 * malformed circuit into the editor.
 */

import type {
  Circuit,
  Morphology,
  Neuron,
  NeuronParams,
  PlasticityConfig,
  Population,
  PopulationLayout,
  Probe,
  Projection,
  ReceptorKinetics,
  ShortTermPlasticity,
  Stimulus,
  StimulusPattern,
  Synapse,
  Vec3,
} from '@neuroforge/shared';
import { CIRCUIT_SCHEMA_VERSION } from '@neuroforge/shared';

import { migrateCircuit } from '../migrate';

/** Discriminator written at the top of every exported file. */
export const DOCUMENT_KIND = 'neuroforge.circuit';

/** The envelope written to disk. The circuit itself is nested under `circuit`. */
export interface CircuitDocument {
  kind: typeof DOCUMENT_KIND;
  schema: number;
  application: string;
  exportedAt: string;
  circuit: Circuit;
}

/**
 * Ordered plain-object views. TypeScript cannot enforce key order, so each
 * builder below writes the keys explicitly in document order.
 */
type Json = Record<string, unknown>;

function vec(v: Vec3): Json {
  return { x: v.x, y: v.y, z: v.z };
}

function params(p: NeuronParams): Json {
  switch (p.kind) {
    case 'lif':
      return {
        kind: p.kind,
        cm: p.cm,
        gL: p.gL,
        eL: p.eL,
        vThresh: p.vThresh,
        vReset: p.vReset,
        tRefract: p.tRefract,
      };
    case 'izhikevich':
      return { kind: p.kind, a: p.a, b: p.b, c: p.c, d: p.d, vPeak: p.vPeak, iScale: p.iScale };
    case 'hodgkin-huxley':
      return {
        kind: p.kind,
        cm: p.cm,
        gNa: p.gNa,
        gK: p.gK,
        gL: p.gL,
        eNa: p.eNa,
        eK: p.eK,
        eL: p.eL,
        vDetect: p.vDetect,
        q10: p.q10,
      };
    case 'adex':
      return {
        kind: p.kind,
        cm: p.cm,
        gL: p.gL,
        eL: p.eL,
        deltaT: p.deltaT,
        vT: p.vT,
        vPeak: p.vPeak,
        vReset: p.vReset,
        a: p.a,
        b: p.b,
        tauW: p.tauW,
        tRefract: p.tRefract,
      };
    case 'morris-lecar':
      return {
        kind: p.kind,
        cm: p.cm,
        gCa: p.gCa,
        gK: p.gK,
        gL: p.gL,
        eCa: p.eCa,
        eK: p.eK,
        eL: p.eL,
        v1: p.v1,
        v2: p.v2,
        v3: p.v3,
        v4: p.v4,
        phi: p.phi,
        vDetect: p.vDetect,
      };
  }
}

function morphology(m: Morphology): Json {
  return {
    seed: m.seed,
    somaRadius: m.somaRadius,
    dendriteCount: m.dendriteCount,
    dendriteDepth: m.dendriteDepth,
    dendriteLength: m.dendriteLength,
    dendriteTaper: m.dendriteTaper,
    dendriteSpread: m.dendriteSpread,
    axonLength: m.axonLength,
    axonTerminals: m.axonTerminals,
    scale: m.scale,
    archetype: m.archetype,
  };
}

function neuron(n: Neuron): Json {
  return {
    id: n.id,
    label: n.label,
    position: vec(n.position),
    params: params(n.params),
    polarity: n.polarity,
    morphology: morphology(n.morphology),
    population: n.population,
    bias: n.bias,
    noise: n.noise,
    enabled: n.enabled,
  };
}

function kinetics(k: ReceptorKinetics): Json {
  return { tauRise: k.tauRise, tauDecay: k.tauDecay, eRev: k.eRev, mgBlock: k.mgBlock };
}

function plasticity(p: PlasticityConfig): Json {
  return {
    kind: p.kind,
    aPlus: p.aPlus,
    aMinus: p.aMinus,
    tauPlus: p.tauPlus,
    tauMinus: p.tauMinus,
    tauX: p.tauX,
    tauY: p.tauY,
    wMin: p.wMin,
    wMax: p.wMax,
    learningRate: p.learningRate,
  };
}

function stp(s: ShortTermPlasticity): Json {
  return { enabled: s.enabled, u: s.u, tauRec: s.tauRec, tauFacil: s.tauFacil };
}

function synapse(s: Synapse): Json {
  return {
    id: s.id,
    source: s.source,
    target: s.target,
    receptor: s.receptor,
    weight: s.weight,
    delay: s.delay,
    kinetics: kinetics(s.kinetics),
    plasticity: plasticity(s.plasticity),
    stp: stp(s.stp),
    releaseProbability: s.releaseProbability,
    arc: s.arc,
    enabled: s.enabled,
  };
}

function layout(l: PopulationLayout): Json {
  switch (l.kind) {
    case 'grid':
      return { kind: l.kind, columns: l.columns, rows: l.rows, layers: l.layers, spacing: l.spacing };
    case 'sphere':
      return { kind: l.kind, radius: l.radius, jitter: l.jitter, seed: l.seed };
    case 'disc':
      return { kind: l.kind, radius: l.radius, thickness: l.thickness, seed: l.seed };
    case 'column':
      return { kind: l.kind, radius: l.radius, height: l.height, seed: l.seed };
    case 'explicit':
      return { kind: l.kind, positions: l.positions.map(vec) };
  }
}

function population(p: Population): Json {
  return {
    id: p.id,
    name: p.name,
    size: p.size,
    polarity: p.polarity,
    params: params(p.params),
    morphology: morphology(p.morphology),
    layout: layout(p.layout),
    origin: vec(p.origin),
    color: p.color,
    members: [...p.members],
    collapsed: p.collapsed,
  };
}

function projection(p: Projection): Json {
  const rule = p.rule;
  let ruleJson: Json;
  switch (rule.kind) {
    case 'all-to-all':
      ruleJson = { kind: rule.kind, selfConnections: rule.selfConnections };
      break;
    case 'random':
      ruleJson = {
        kind: rule.kind,
        probability: rule.probability,
        seed: rule.seed,
        selfConnections: rule.selfConnections,
      };
      break;
    case 'one-to-one':
      ruleJson = { kind: rule.kind };
      break;
    case 'gaussian':
      ruleJson = { kind: rule.kind, sigma: rule.sigma, maxProbability: rule.maxProbability, seed: rule.seed };
      break;
    case 'distance-threshold':
      ruleJson = { kind: rule.kind, radius: rule.radius, probability: rule.probability, seed: rule.seed };
      break;
    case 'fixed-in-degree':
    case 'fixed-out-degree':
      ruleJson = { kind: rule.kind, degree: rule.degree, seed: rule.seed };
      break;
  }
  return {
    id: p.id,
    name: p.name,
    source: p.source,
    target: p.target,
    rule: ruleJson,
    weightMean: p.weightMean,
    weightJitter: p.weightJitter,
    delayMean: p.delayMean,
    delayJitter: p.delayJitter,
  };
}

function pattern(p: StimulusPattern): Json {
  switch (p.kind) {
    case 'constant':
      return { kind: p.kind, amplitude: p.amplitude };
    case 'step':
      return { kind: p.kind, amplitude: p.amplitude, start: p.start, duration: p.duration };
    case 'pulse-train':
      return { kind: p.kind, amplitude: p.amplitude, frequency: p.frequency, width: p.width, start: p.start };
    case 'sine':
      return { kind: p.kind, amplitude: p.amplitude, frequency: p.frequency, offset: p.offset };
    case 'poisson':
      return { kind: p.kind, rate: p.rate, amplitude: p.amplitude, seed: p.seed };
    case 'ramp':
      return { kind: p.kind, from: p.from, to: p.to, start: p.start, duration: p.duration };
  }
}

function stimulus(s: Stimulus): Json {
  return {
    id: s.id,
    name: s.name,
    targets: [...s.targets],
    pattern: pattern(s.pattern),
    enabled: s.enabled,
  };
}

function probe(p: Probe): Json {
  return {
    id: p.id,
    target: p.target,
    signal: p.signal,
    capacity: p.capacity,
    color: p.color,
    enabled: p.enabled,
  };
}

/** The circuit as an ordered plain object, ready for `JSON.stringify`. */
export function circuitToJson(circuit: Circuit): Json {
  const s = circuit.simulation;
  const c = circuit.camera;
  const r = circuit.render;
  return {
    id: circuit.id,
    name: circuit.name,
    description: circuit.description,
    version: circuit.version,
    createdAt: circuit.createdAt,
    updatedAt: circuit.updatedAt,
    neurons: circuit.neurons.map(neuron),
    synapses: circuit.synapses.map(synapse),
    populations: circuit.populations.map(population),
    projections: circuit.projections.map(projection),
    stimuli: circuit.stimuli.map(stimulus),
    probes: circuit.probes.map(probe),
    simulation: {
      dt: s.dt,
      integrator: s.integrator,
      speed: s.speed,
      gain: s.gain,
      noise: s.noise,
      seed: s.seed,
      plasticityEnabled: s.plasticityEnabled,
      maxSubstepsPerFrame: s.maxSubstepsPerFrame,
      backend: s.backend,
    },
    camera: { position: vec(c.position), target: vec(c.target), fov: c.fov, mode: c.mode },
    render: {
      bloomIntensity: r.bloomIntensity,
      bloomThreshold: r.bloomThreshold,
      bloomRadius: r.bloomRadius,
      depthOfField: r.depthOfField,
      focusDistance: r.focusDistance,
      focalLength: r.focalLength,
      bokehScale: r.bokehScale,
      fogDensity: r.fogDensity,
      ambientOcclusion: r.ambientOcclusion,
      aoIntensity: r.aoIntensity,
      vignette: r.vignette,
      chromaticAberration: r.chromaticAberration,
      exposure: r.exposure,
      gridVisible: r.gridVisible,
      gridFade: r.gridFade,
      showDendrites: r.showDendrites,
      showAxons: r.showAxons,
      showParticles: r.showParticles,
      particleDensity: r.particleDensity,
      neuronScale: r.neuronScale,
      voltageColoring: r.voltageColoring,
    },
    tags: [...circuit.tags],
  };
}

/** Serialise a circuit into the exchange envelope. */
export function serializeCircuit(circuit: Circuit, indent = 2): string {
  const document: Json = {
    kind: DOCUMENT_KIND,
    schema: CIRCUIT_SCHEMA_VERSION,
    application: 'neuroforge',
    exportedAt: new Date().toISOString(),
    circuit: circuitToJson(circuit),
  };
  return `${JSON.stringify(document, null, indent)}\n`;
}

/**
 * Parse a document written by `serializeCircuit`, a bare circuit object, or an
 * older schema version. Syntax errors and validation errors are reported the
 * same way so callers only need one error path.
 */
export function parseCircuitDocument(text: string): { circuit: Circuit | null; errors: string[] } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { circuit: null, errors: [`document: not valid JSON (${message})`] };
  }

  const errors: string[] = [];
  if (
    typeof parsed === 'object' &&
    parsed !== null &&
    'kind' in parsed &&
    (parsed as { kind: unknown }).kind !== DOCUMENT_KIND
  ) {
    errors.push(
      `kind: expected '${DOCUMENT_KIND}', got ${JSON.stringify((parsed as { kind: unknown }).kind)}; ` +
        'attempting to read it as a circuit anyway',
    );
  }

  const result = migrateCircuit(parsed);
  return { circuit: result.circuit, errors: [...errors, ...result.errors] };
}
