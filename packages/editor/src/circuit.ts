/**
 * The document a user sees before they have built anything.
 *
 * An empty scene is a dead scene, so a new circuit is seeded with a small
 * cortical column: 60 excitatory pyramidal cells and 15 inhibitory basket cells,
 * sparsely and recurrently wired. It is tuned to fire continuously and
 * irregularly forever without ever running away. The reasoning is recorded
 * beside the constants below, because the numbers are only defensible together.
 */

import type { Circuit, Neuron, Probe, Synapse } from '@neuroforge/shared';
import {
  CIRCUIT_SCHEMA_VERSION,
  COLORS,
  DEFAULT_CAMERA,
  DEFAULT_RENDER_SETTINGS,
  DEFAULT_SIMULATION_SETTINGS,
  newCircuitId,
  newProbeId,
} from '@neuroforge/shared';

import { newProjectionId } from './entities';
import type { PopulationSpec, ProjectionSpec } from './populations';
import {
  DEFAULT_DELAY_JITTER,
  DEFAULT_DELAY_MEAN,
  DEFAULT_WEIGHT_JITTER,
  DEFAULT_WEIGHT_MEAN,
  instantiatePopulation,
  instantiateProjection,
} from './populations';

/**
 * Both populations use the Izhikevich model. It has a hard peak-and-reset, so
 * the voltage cannot diverge numerically whatever the input does, and its spike
 * reaches +30 mV, which is what lights up the renderer's voltage ramp.
 *
 * Rheobase for these parameters is 100 pA (I_model = 4 at iScale 0.04). The
 * resting drive table in `entities.ts` puts both populations below that on the
 * mean and lets noise carry them across, which is the fluctuation-driven regime:
 * every spike is a fluctuation, and the mean drive alone can never sustain one.
 *
 * Excitatory cells: bias 55 pA leaves the fast subsystem's threshold about
 * 8.4 mV above rest, and 78 pA of input noise gives the membrane about 3.8 mV of
 * standard deviation — roughly 2.2 sigma from threshold, a handful of hertz.
 * Interneurons sit at ~1.8 sigma and so run several times faster, which is what
 * basket cells do.
 *
 * Recurrent excitation is deliberately subcritical. Each pyramidal cell receives
 * about 4.7 excitatory inputs, each worth roughly a 4 mV depolarisation. The
 * chance that one of them lands close enough to threshold to cause a spike is
 * about 3.5%, so one excitatory spike begets about 0.16 further spikes. A loop
 * gain that far below 1 cannot run away: recurrence amplifies the ongoing rate
 * by about 20% and nothing more.
 *
 * Inhibition is the brake for the case the estimate above is wrong. GABA-A
 * reverses at -70 mV, a few millivolts below rest, so it is nearly silent while
 * the network is quiet and acquires real driving force exactly when the network
 * depolarises. Basket cells receive dense, strong excitation (about 9 inputs at
 * 1.05 nS) so they are recruited by any excitatory surge within a millisecond,
 * and they project back onto the pyramidal cells densely (about 3.8 inputs at
 * 1.8 nS). A synchronous excitatory volley therefore recruits its own inhibition
 * and is cut off; the worst case is a gamma-band rhythm, not divergence. The
 * Izhikevich recovery variable adds a further 8 units of adaptation per spike,
 * bounding sustained rates on top of that.
 */
const EXCITATORY_SIZE = 60;
const INHIBITORY_SIZE = 15;

const PYRAMIDAL_SPEC: PopulationSpec = {
  name: 'Pyramidal',
  size: EXCITATORY_SIZE,
  polarity: 'excitatory',
  model: 'izhikevich',
  // Izhikevich's regular-spiking parameters; these are also the shared defaults.
  params: { a: 0.02, b: 0.2, c: -65, d: 8 },
  layout: { kind: 'column', radius: 15, height: 44, seed: 0x5eed01 },
  origin: { x: 0, y: 0, z: 0 },
  archetype: 'pyramidal',
  color: COLORS.accent,
};

const BASKET_SPEC: PopulationSpec = {
  name: 'Basket',
  size: INHIBITORY_SIZE,
  polarity: 'inhibitory',
  model: 'izhikevich',
  // Fast-spiking: quick recovery, small spike-triggered adaptation.
  params: { a: 0.1, b: 0.2, c: -65, d: 2 },
  layout: { kind: 'column', radius: 9, height: 30, seed: 0x5eed02 },
  origin: { x: 0, y: 1.5, z: 0 },
  archetype: 'basket',
  color: COLORS.secondary,
};

/** Projection specs, minus the population ids, which only exist once built. */
const PROJECTIONS: readonly (Omit<ProjectionSpec, 'source' | 'target'> & {
  from: 'pyramidal' | 'basket';
  to: 'pyramidal' | 'basket';
})[] = [
  {
    name: 'Pyramidal → Pyramidal',
    from: 'pyramidal',
    to: 'pyramidal',
    // Distance-dependent, so the recurrent web reads as local clustering rather
    // than as a uniform hairball. Mean connection probability lands near 0.08,
    // giving each cell about 4.7 recurrent inputs.
    rule: { kind: 'gaussian', sigma: 16, maxProbability: 0.16, seed: 0x1a2b3c },
    weightMean: 0.85,
    weightJitter: 0.28,
    delayMean: 1.6,
    delayJitter: 0.7,
  },
  {
    name: 'Pyramidal → Basket',
    from: 'pyramidal',
    to: 'basket',
    rule: { kind: 'random', probability: 0.15, seed: 0x2b3c4d, selfConnections: false },
    weightMean: 1.05,
    weightJitter: 0.3,
    delayMean: 0.9,
    delayJitter: 0.35,
  },
  {
    name: 'Basket → Pyramidal',
    from: 'basket',
    to: 'pyramidal',
    rule: { kind: 'random', probability: 0.25, seed: 0x3c4d5e, selfConnections: false },
    weightMean: 1.8,
    weightJitter: 0.55,
    delayMean: 0.8,
    delayJitter: 0.3,
  },
  {
    name: 'Basket → Basket',
    from: 'basket',
    to: 'basket',
    rule: { kind: 'random', probability: 0.12, seed: 0x4d5e6f, selfConnections: false },
    weightMean: 1.2,
    weightJitter: 0.4,
    delayMean: 0.6,
    delayJitter: 0.25,
  },
];

function makeProbe(target: Neuron | undefined, color: string): Probe | null {
  if (target === undefined) return null;
  return {
    id: newProbeId(),
    target: target.id,
    signal: 'voltage',
    capacity: 2048,
    color,
    enabled: true,
  };
}

/**
 * A new document, seeded with the demo column.
 *
 * Ids are freshly minted on every call, so two circuits never collide; the
 * topology, positions and parameters are fully determined by the seeds above.
 */
export function createEmptyCircuit(name = 'Untitled circuit'): Circuit {
  const now = Date.now();

  const pyramidal = instantiatePopulation(PYRAMIDAL_SPEC);
  const basket = instantiatePopulation(BASKET_SPEC);

  const populationOf = {
    pyramidal: pyramidal.population,
    basket: basket.population,
  } as const;

  const neurons: Neuron[] = [...pyramidal.neurons, ...basket.neurons];

  // instantiateProjection resolves members through the document, so the wiring
  // pass needs a document with the neurons and populations already in it.
  const wiring: Circuit = {
    id: newCircuitId(),
    name,
    description: 'A small cortical column: pyramidal cells reined in by basket interneurons.',
    version: CIRCUIT_SCHEMA_VERSION,
    createdAt: now,
    updatedAt: now,
    neurons,
    synapses: [],
    populations: [pyramidal.population, basket.population],
    projections: [],
    stimuli: [],
    probes: [],
    simulation: { ...DEFAULT_SIMULATION_SETTINGS },
    camera: {
      position: { ...DEFAULT_CAMERA.position },
      target: { ...DEFAULT_CAMERA.target },
      fov: DEFAULT_CAMERA.fov,
      mode: DEFAULT_CAMERA.mode,
    },
    render: { ...DEFAULT_RENDER_SETTINGS },
    tags: ['demo', 'cortex'],
  };

  const synapses: Synapse[] = [];
  const projections = PROJECTIONS.map((entry) => {
    const spec: ProjectionSpec = {
      name: entry.name,
      source: populationOf[entry.from].id,
      target: populationOf[entry.to].id,
      rule: entry.rule,
      weightMean: entry.weightMean,
      weightJitter: entry.weightJitter,
      delayMean: entry.delayMean,
      delayJitter: entry.delayJitter,
    };
    for (const synapse of instantiateProjection(spec, wiring)) synapses.push(synapse);
    return {
      id: newProjectionId(),
      name: entry.name,
      source: spec.source,
      target: spec.target,
      rule: entry.rule,
      weightMean: entry.weightMean ?? DEFAULT_WEIGHT_MEAN,
      weightJitter: entry.weightJitter ?? DEFAULT_WEIGHT_JITTER,
      delayMean: entry.delayMean ?? DEFAULT_DELAY_MEAN,
      delayJitter: entry.delayJitter ?? DEFAULT_DELAY_JITTER,
    };
  });

  const probes: Probe[] = [];
  const excitatoryProbe = makeProbe(pyramidal.neurons[0], COLORS.accent);
  const inhibitoryProbe = makeProbe(basket.neurons[0], COLORS.secondary);
  if (excitatoryProbe !== null) probes.push(excitatoryProbe);
  if (inhibitoryProbe !== null) probes.push(inhibitoryProbe);

  return { ...wiring, synapses, projections, probes };
}
