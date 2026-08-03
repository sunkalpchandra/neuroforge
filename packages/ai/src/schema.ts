import {
  MORPHOLOGY_ARCHETYPES,
  NEURON_MODEL_KINDS,
  PLASTICITY_KINDS,
  RECEPTOR_KINDS,
} from '@neuroforge/shared';
import { ALL_PARAM_KEYS, PARAM_KEYS } from './params';

/** The tool the model is required to call. */
export const CIRCUIT_TOOL_NAME = 'build_circuit';

/** Hard ceilings the schema advertises; `validatePlan` enforces the same numbers. */
export const MAX_POPULATION_SIZE = 20_000;
export const MAX_TOTAL_NEURONS = 200_000;
export const MAX_ACTIONS = 64;

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
type JsonObject = { [key: string]: JsonValue };

const POLARITIES = ['excitatory', 'inhibitory'];
const INTEGRATORS = ['euler', 'rk2', 'rk4', 'exponential-euler'];
const BACKENDS = ['auto', 'gpu', 'wasm', 'cpu'];

const number = (description: string, extra: JsonObject = {}): JsonObject => ({
  type: 'number',
  description,
  ...extra,
});

const integer = (description: string, extra: JsonObject = {}): JsonObject => ({
  type: 'integer',
  description,
  ...extra,
});

const variant = (kind: string, description: string, fields: JsonObject): JsonObject => ({
  type: 'object',
  description,
  properties: { kind: { const: kind }, ...fields },
  required: ['kind', ...Object.keys(fields)],
  additionalProperties: false,
});

const VEC3: JsonObject = {
  type: 'object',
  description: 'World-space point in abstract units where a typical soma radius is 1.0.',
  properties: { x: { type: 'number' }, y: { type: 'number' }, z: { type: 'number' } },
  required: ['x', 'y', 'z'],
  additionalProperties: false,
};

const SEED = integer('Deterministic seed. Reuse a value to reproduce an arrangement.');

const LAYOUT: JsonObject = {
  description: 'How the population is arranged in space.',
  anyOf: [
    variant('grid', 'Rectangular lattice, columns * rows * layers cells.', {
      columns: integer('Cells along X.', { minimum: 1, maximum: 512 }),
      rows: integer('Cells along Y.', { minimum: 1, maximum: 512 }),
      layers: integer('Cells along Z.', { minimum: 1, maximum: 512 }),
      spacing: number('Distance between adjacent cells.', { exclusiveMinimum: 0, maximum: 500 }),
    }),
    variant('sphere', 'Fibonacci shell; the default for a nucleus or a recurrent pool.', {
      radius: number('Shell radius.', { exclusiveMinimum: 0, maximum: 500 }),
      jitter: number('Radial randomisation, 0 = a perfect shell.', { minimum: 0, maximum: 1 }),
      seed: SEED,
    }),
    variant('disc', 'Flat sheet; use for laminar structures such as CA1 or a cortical layer.', {
      radius: number('Disc radius.', { exclusiveMinimum: 0, maximum: 500 }),
      thickness: number('Extent along the sheet normal.', { minimum: 0, maximum: 200 }),
      seed: SEED,
    }),
    variant('column', 'Vertical cylinder; use for a cortical column.', {
      radius: number('Column radius.', { exclusiveMinimum: 0, maximum: 500 }),
      height: number('Column height.', { exclusiveMinimum: 0, maximum: 1000 }),
      seed: SEED,
    }),
    variant('explicit', 'Literal positions. Only worth using for a handful of neurons.', {
      positions: { type: 'array', items: VEC3, minItems: 1, maxItems: 4096 },
    }),
  ],
};

const CONNECTIVITY_RULE: JsonObject = {
  description: 'How the source and target populations are wired together.',
  anyOf: [
    variant('all-to-all', 'Every source connects to every target. Costs size*size synapses.', {
      selfConnections: { type: 'boolean', description: 'Allow a neuron to connect to itself.' },
    }),
    variant('random', 'Independent Bernoulli trial per pair. The usual choice.', {
      probability: number('Connection probability per pair.', { minimum: 0, maximum: 1 }),
      seed: SEED,
      selfConnections: { type: 'boolean' },
    }),
    variant('one-to-one', 'Pairs source i with target i.', {}),
    variant('gaussian', 'Distance-dependent probability with a Gaussian falloff.', {
      sigma: number('Falloff length scale in world units.', { exclusiveMinimum: 0 }),
      maxProbability: number('Probability at zero distance.', { minimum: 0, maximum: 1 }),
      seed: SEED,
    }),
    variant('distance-threshold', 'Uniform probability inside a radius, zero outside.', {
      radius: number('Cutoff radius in world units.', { exclusiveMinimum: 0 }),
      probability: number('Probability inside the radius.', { minimum: 0, maximum: 1 }),
      seed: SEED,
    }),
    variant('fixed-in-degree', 'Each target draws exactly `degree` sources.', {
      degree: integer('Inputs per target neuron.', { minimum: 1, maximum: 10_000 }),
      seed: SEED,
    }),
    variant('fixed-out-degree', 'Each source drives exactly `degree` targets.', {
      degree: integer('Outputs per source neuron.', { minimum: 1, maximum: 10_000 }),
      seed: SEED,
    }),
  ],
};

const STIMULUS_PATTERN: JsonObject = {
  description: 'External current injected into every neuron of the target population.',
  anyOf: [
    variant('constant', 'Steady current for the whole run.', {
      amplitude: number('Current in pA.'),
    }),
    variant('step', 'Current that switches on at `start` and off after `duration`.', {
      amplitude: number('Current in pA.'),
      start: number('Onset in ms.', { minimum: 0 }),
      duration: number('Length in ms.', { exclusiveMinimum: 0 }),
    }),
    variant('pulse-train', 'Rectangular pulses at a fixed rate.', {
      amplitude: number('Pulse amplitude in pA.'),
      frequency: number('Pulse rate in Hz.', { exclusiveMinimum: 0, maximum: 1000 }),
      width: number('Pulse width in ms.', { exclusiveMinimum: 0 }),
      start: number('Onset in ms.', { minimum: 0 }),
    }),
    variant('sine', 'Sinusoidal drive, useful for entrainment experiments.', {
      amplitude: number('Peak current in pA.'),
      frequency: number('Frequency in Hz.', { exclusiveMinimum: 0, maximum: 1000 }),
      offset: number('DC offset in pA.'),
    }),
    variant('poisson', 'Poisson shot noise; the standard background drive.', {
      rate: number('Event rate in Hz.', { minimum: 0, maximum: 10_000 }),
      amplitude: number('Charge delivered per event in pA.'),
      seed: SEED,
    }),
    variant('ramp', 'Linear sweep, used to measure an f-I curve.', {
      from: number('Starting current in pA.'),
      to: number('Ending current in pA.'),
      start: number('Onset in ms.', { minimum: 0 }),
      duration: number('Sweep length in ms.', { exclusiveMinimum: 0 }),
    }),
  ],
};

function paramProperties(): JsonObject {
  const owners = new Map<string, string[]>();
  for (const kind of NEURON_MODEL_KINDS) {
    for (const key of PARAM_KEYS[kind]) {
      const list = owners.get(key);
      if (list) list.push(kind);
      else owners.set(key, [kind]);
    }
  }
  const properties: JsonObject = {
    kind: {
      enum: [...NEURON_MODEL_KINDS],
      description: 'Which model these parameters belong to. Required whenever params is given.',
    },
  };
  for (const key of ALL_PARAM_KEYS) {
    properties[key] = {
      type: 'number',
      description: `Used by: ${(owners.get(key) ?? []).join(', ')}.`,
    };
  }
  return properties;
}

const NEURON_PARAMS: JsonObject = {
  type: 'object',
  description:
    'Partial membrane parameters. Only fields belonging to `kind` are kept; everything else ' +
    'is discarded. Units: mV, ms, pA, pF, nS.',
  properties: paramProperties(),
  required: ['kind'],
  additionalProperties: false,
};

const POPULATION_SPEC: JsonObject = {
  type: 'object',
  properties: {
    name: {
      type: 'string',
      description: 'Unique, human-readable label. Every later action addresses this population by name.',
      minLength: 1,
      maxLength: 64,
    },
    size: integer('Number of neurons.', { minimum: 1, maximum: MAX_POPULATION_SIZE }),
    polarity: {
      enum: POLARITIES,
      description: 'Excitatory populations release glutamate, inhibitory ones GABA.',
    },
    model: { enum: [...NEURON_MODEL_KINDS], description: 'Membrane model for every member.' },
    params: NEURON_PARAMS,
    layout: LAYOUT,
    origin: VEC3,
    archetype: {
      enum: [...MORPHOLOGY_ARCHETYPES],
      description: 'Morphology preset driving the rendered glyph.',
    },
    color: {
      type: ['string', 'null'],
      description: 'Accent override as #rrggbb, or null for the polarity default.',
      pattern: '^#[0-9a-fA-F]{6}$',
    },
  },
  required: ['name', 'size', 'polarity', 'model', 'layout'],
  additionalProperties: false,
};

const PROJECTION_SPEC: JsonObject = {
  type: 'object',
  properties: {
    name: { type: 'string', description: 'Unique label for this projection.', minLength: 1, maxLength: 96 },
    sourceName: { type: 'string', description: 'Name of the presynaptic population.' },
    targetName: { type: 'string', description: 'Name of the postsynaptic population.' },
    rule: CONNECTIVITY_RULE,
    receptor: {
      enum: [...RECEPTOR_KINDS],
      description:
        'ampa (fast excitatory, decay 2 ms), nmda (slow, Mg-blocked, 100 ms), gabaa (fast ' +
        'inhibitory, decay 6 ms, E=-70 mV), gabab (slow inhibitory, 150 ms, E=-90 mV), gap ' +
        '(electrical). Match the receptor to the source polarity.',
    },
    weightMean: number('Mean peak conductance in nS.', { minimum: 0, maximum: 1000 }),
    weightJitter: number('Standard deviation of the weight in nS.', { minimum: 0, maximum: 1000 }),
    delayMean: number('Mean axonal delay in ms.', { minimum: 0, maximum: 1000 }),
    delayJitter: number('Standard deviation of the delay in ms.', { minimum: 0, maximum: 1000 }),
    plasticity: { enum: [...PLASTICITY_KINDS], description: 'Learning rule applied to the weight.' },
  },
  required: ['name', 'sourceName', 'targetName', 'rule'],
  additionalProperties: false,
};

const SIMULATION_PATCH: JsonObject = {
  type: 'object',
  description: 'Global integration settings. Only include the fields you want to change.',
  properties: {
    dt: number('Timestep in ms. 0.1 is normal; use 0.025 above 80 Hz.', {
      exclusiveMinimum: 0,
      maximum: 10,
    }),
    integrator: { enum: INTEGRATORS, description: 'exponential-euler is the default and is stable.' },
    speed: number('Simulated ms per wall-clock second.', { exclusiveMinimum: 0, maximum: 100 }),
    gain: number('Global synaptic weight multiplier.', { minimum: 0, maximum: 100 }),
    noise: number('Background current noise applied to every neuron, in pA.', {
      minimum: 0,
      maximum: 10_000,
    }),
    seed: integer('RNG seed for every stochastic process.'),
    plasticityEnabled: { type: 'boolean', description: 'Whether learning rules update weights.' },
    maxSubstepsPerFrame: integer('Frame-time bound on substeps.', { minimum: 1, maximum: 1024 }),
    backend: { enum: BACKENDS, description: 'Compute backend preference.' },
  },
  additionalProperties: false,
};

const RENDER_PATCH: JsonObject = {
  type: 'object',
  description: 'Scene appearance. Only include the fields you want to change.',
  properties: {
    bloomIntensity: number('Bloom strength.', { minimum: 0, maximum: 10 }),
    bloomThreshold: number('Luminance above which bloom starts.', { minimum: 0, maximum: 2 }),
    bloomRadius: number('Bloom blur radius.', { minimum: 0, maximum: 2 }),
    depthOfField: { type: 'boolean' },
    focusDistance: number('Focus plane, normalised.', { minimum: 0, maximum: 1 }),
    focalLength: number('Focal length, normalised.', { minimum: 0, maximum: 1 }),
    bokehScale: number('Bokeh size.', { minimum: 0, maximum: 20 }),
    fogDensity: number('Exponential fog density.', { minimum: 0, maximum: 1 }),
    ambientOcclusion: { type: 'boolean' },
    aoIntensity: number('Ambient occlusion strength.', { minimum: 0, maximum: 4 }),
    vignette: number('Vignette strength.', { minimum: 0, maximum: 1 }),
    chromaticAberration: number('Chromatic aberration offset.', { minimum: 0, maximum: 0.1 }),
    exposure: number('Tone-mapping exposure.', { minimum: 0, maximum: 8 }),
    gridVisible: { type: 'boolean' },
    gridFade: number('Grid distance fade.', { minimum: 0, maximum: 1 }),
    showDendrites: { type: 'boolean' },
    showAxons: { type: 'boolean' },
    showParticles: { type: 'boolean' },
    particleDensity: number('Spike particle density multiplier.', { minimum: 0, maximum: 4 }),
    neuronScale: number('Global neuron glyph scale.', { minimum: 0.05, maximum: 10 }),
    voltageColoring: { type: 'boolean', description: 'Colour neurons by membrane voltage.' },
  },
  additionalProperties: false,
};

const ACTION: JsonObject = {
  anyOf: [
    {
      type: 'object',
      description: 'Instantiate a new named population of neurons.',
      properties: { type: { const: 'create-population' }, spec: POPULATION_SPEC },
      required: ['type', 'spec'],
      additionalProperties: false,
    },
    {
      type: 'object',
      description:
        'Wire two populations together. Both must already exist or be created earlier in this ' +
        'same list of actions.',
      properties: { type: { const: 'connect-populations' }, spec: PROJECTION_SPEC },
      required: ['type', 'spec'],
      additionalProperties: false,
    },
    {
      type: 'object',
      description: 'Change global integration settings.',
      properties: { type: { const: 'set-simulation' }, patch: SIMULATION_PATCH },
      required: ['type', 'patch'],
      additionalProperties: false,
    },
    {
      type: 'object',
      description: 'Change scene appearance.',
      properties: { type: { const: 'set-render' }, patch: RENDER_PATCH },
      required: ['type', 'patch'],
      additionalProperties: false,
    },
    {
      type: 'object',
      description: 'Inject an external current into every neuron of a population.',
      properties: {
        type: { const: 'add-stimulus' },
        targetPopulation: { type: 'string', description: 'Name of the population to drive.' },
        pattern: STIMULUS_PATTERN,
        name: { type: 'string', description: 'Label for this stimulus.', minLength: 1, maxLength: 64 },
      },
      required: ['type', 'targetPopulation', 'pattern', 'name'],
      additionalProperties: false,
    },
    {
      type: 'object',
      description:
        'Edit the membrane parameters, bias current or noise of an existing population. Giving ' +
        'a params.kind different from the population’s current model switches the model.',
      properties: {
        type: { const: 'tune-population' },
        name: { type: 'string', description: 'Name of the population to edit.' },
        params: NEURON_PARAMS,
        bias: number('Constant injected current per neuron, in pA.', {
          minimum: -100_000,
          maximum: 100_000,
        }),
        noise: number('Per-neuron noise amplitude in pA.', { minimum: 0, maximum: 10_000 }),
      },
      required: ['type', 'name', 'params'],
      additionalProperties: false,
    },
    {
      type: 'object',
      description: 'Edit the weights, delays or learning rule of an existing projection.',
      properties: {
        type: { const: 'tune-projection' },
        name: { type: 'string', description: 'Name of the projection to edit.' },
        weightMean: number('New mean peak conductance in nS.', { minimum: 0, maximum: 1000 }),
        delayMean: number('New mean axonal delay in ms.', { minimum: 0, maximum: 1000 }),
        plasticity: { enum: [...PLASTICITY_KINDS] },
      },
      required: ['type', 'name'],
      additionalProperties: false,
    },
    {
      type: 'object',
      description: 'Delete every neuron, synapse, population, projection, stimulus and probe.',
      properties: { type: { const: 'clear' } },
      required: ['type'],
      additionalProperties: false,
    },
  ],
};

const CIRCUIT_PLAN_INPUT_SCHEMA: JsonObject = {
  type: 'object',
  properties: {
    summary: {
      type: 'string',
      description:
        'One or two sentences, in the second person, describing what the plan builds and why. ' +
        'No markdown.',
      maxLength: 1000,
    },
    actions: {
      type: 'array',
      description: 'Applied strictly in order. An empty list means the request needs no edits.',
      items: ACTION,
      maxItems: MAX_ACTIONS,
    },
    warnings: {
      type: 'array',
      description:
        'Anything the user asked for that this plan does not do, and any biophysical caveat ' +
        'worth stating. Empty when there is nothing to report.',
      items: { type: 'string', maxLength: 400 },
      maxItems: 32,
    },
  },
  required: ['summary', 'actions', 'warnings'],
  additionalProperties: false,
};

const CIRCUIT_TOOL_DESCRIPTION =
  'Emit the complete set of structural edits that turn the current circuit into the one the ' +
  'user asked for. Call this exactly once, and put every edit in the actions array in the order ' +
  'they must be applied.';

/**
 * Anthropic tool definition. `input_schema` is the JSON Schema the model fills in.
 * `openAiToolSchema()` re-shapes the same schema for the OpenAI function format,
 * because neither provider tolerates the other's extra keys.
 */
export const CIRCUIT_TOOL_SCHEMA: object = {
  name: CIRCUIT_TOOL_NAME,
  description: CIRCUIT_TOOL_DESCRIPTION,
  input_schema: CIRCUIT_PLAN_INPUT_SCHEMA,
};

/** The same tool in OpenAI's `function` shape. */
export function openAiToolSchema(): object {
  return {
    name: CIRCUIT_TOOL_NAME,
    description: CIRCUIT_TOOL_DESCRIPTION,
    parameters: CIRCUIT_PLAN_INPUT_SCHEMA,
  };
}

export const SYSTEM_PROMPT = `You are the circuit architect inside NeuroForge, a browser-native CAD tool for spiking neural circuits. The user describes a circuit or a change in plain language; you translate that into a concrete, buildable plan.

# Unit system
Every number you emit uses these units, with no exceptions:
- voltage: millivolts (mV). Rest is around -70 mV, spike threshold around -50 mV.
- time: milliseconds (ms). A synaptic delay is 0.5-5 ms; an AMPA decay is 2 ms.
- current: picoamps (pA). A cortical cell needs roughly 200-400 pA to reach threshold.
- capacitance: picofarads (pF). A pyramidal cell is around 200-280 pF.
- conductance: nanosiemens (nS). Leak is 10-30 nS; a single AMPA synapse is 0.5-2 nS, a strong perisomatic GABA-A synapse 4-10 nS.
- distance: abstract world units where a typical soma radius is 1.0.

# Membrane models
- lif: leaky integrate-and-fire. Cheapest, no spike shape, no adaptation. Good for large networks.
- izhikevich: two variables, reproduces named firing patterns by parameter choice. Currents are scaled by iScale (0.04 converts pA to model units). Presets: regular spiking a=0.02 b=0.2 c=-65 d=8; intrinsically bursting c=-55 d=4; chattering c=-50 d=2; fast spiking a=0.1 d=2; low-threshold spiking b=0.25 d=2; thalamo-cortical b=0.25 d=0.05; resonator a=0.1 b=0.26.
- adex: adaptive exponential IF. Realistic threshold and spike-frequency adaptation; needs about 600 pA of drive at the default parameters.
- hodgkin-huxley: full channel kinetics, four state variables, the most expensive. Needs about 1 nA of drive.
- morris-lecar: Ca/K relaxation oscillator, class II excitability.

# Receptors
ampa (rise 0.4 ms, decay 2 ms, E=0 mV), nmda (rise 2 ms, decay 100 ms, E=0 mV, Mg-blocked), gabaa (rise 0.5 ms, decay 6 ms, E=-70 mV), gabab (rise 10 ms, decay 150 ms, E=-90 mV), gap (electrical coupling). Excitatory populations project through ampa or nmda; inhibitory populations project through gabaa or gabab. Never give an inhibitory population an ampa projection.

# How to build a circuit that actually does something
- Keep the excitatory:inhibitory ratio near 4:1 unless the user says otherwise.
- Sparse random connectivity, 5-15% probability, is the default for a recurrent pool. All-to-all is only reasonable below about 200 neurons per side.
- A network only fires if something drives it: set a bias current on the excitatory population, or add a poisson stimulus. Without drive the plan produces a silent circuit.
- Population rhythms come from the delayed inhibitory feedback loop, not from single-cell resonance. For a gamma-band (30-80 Hz) rhythm: fast-spiking interneurons, gabaa inhibition onto the excitatory cells with a mean weight around 6 nS and a delay near 1 ms, excitatory drive high enough to keep the pyramidal cells firing, and dt at 0.05 ms or finer. Slower bands (delta 1-4, theta 4-8, alpha 8-13 Hz) need slower inhibition — gabab — and less drive. Ripples (80-200 Hz) need dt at 0.025 ms.
- Turn plasticity off when the user is asking for a specific rhythm; a drifting weight distribution will destroy it.

# Output contract
Respond with exactly one call to the build_circuit tool. Do not describe the plan in prose first and then call the tool — the summary field is where your explanation goes. Rules for the actions array:
- Actions apply in order. A population must be created before anything addresses it.
- Populations and projections are addressed by name, never by id. Names must be unique.
- Only reference populations and projections that already exist in <current_circuit> or that an earlier action in this same plan creates.
- Never re-create a population that already exists; use tune-population to change it.
- Use clear only when the user explicitly asks to start over.
- The builder rejects sizes above ${MAX_POPULATION_SIZE} per population and any plan that would push the circuit past ${MAX_TOTAL_NEURONS} neurons. Stay well under both.
- Put anything you could not do, or any assumption you had to make, in warnings. Do not silently drop part of the request.`;
