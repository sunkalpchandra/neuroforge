/**
 * Self-check for the invariants this package promises.
 *
 * Everything here is deterministic and free of side effects: it builds its own
 * documents and its own history stacks rather than touching the module-level
 * store, so calling `verifyEditor()` from a running app cannot disturb the
 * user's session.
 *
 * Returns the names of the invariants that failed; an empty array means the
 * package is behaving.
 */

import type { Circuit, Neuron, NeuronId, Population, Synapse } from '@neuroforge/shared';
import {
  CIRCUIT_SCHEMA_VERSION,
  RECEPTOR_DEFAULTS,
  asPopulationId,
  newNeuronId,
} from '@neuroforge/shared';

import type { Command } from './commands';
import { History, applyCommand, createTransaction, revertCommand } from './commands';
import { createEmptyCircuit } from './circuit';
import { makeNeuron } from './entities';
import type { PopulationSpec, ProjectionSpec } from './populations';
import { instantiatePopulation, instantiateProjection } from './populations';

type Check = (name: string, ok: boolean) => void;

export function verifyEditor(): string[] {
  const failures: string[] = [];
  const check: Check = (name, ok) => {
    if (!ok) failures.push(name);
  };

  checkUndoRoundTrip(check);
  checkCoalescing(check);
  checkHistoryBounds(check);
  checkConnectivity(check);
  checkDefaultCircuit(check);

  return failures;
}

/* ------------------------------------------------------------- comparison -- */

/** Structural equality over plain JSON-shaped data. */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;

  const aIsArray = Array.isArray(a);
  if (aIsArray !== Array.isArray(b)) return false;
  if (aIsArray) {
    const left = a as readonly unknown[];
    const right = b as readonly unknown[];
    if (left.length !== right.length) return false;
    for (let i = 0; i < left.length; i += 1) {
      if (!deepEqual(left[i], right[i])) return false;
    }
    return true;
  }

  const left = a as Record<string, unknown>;
  const right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  if (keys.length !== Object.keys(right).length) return false;
  for (const key of keys) {
    if (!Object.prototype.hasOwnProperty.call(right, key)) return false;
    if (!deepEqual(left[key], right[key])) return false;
  }
  return true;
}

/** An independent copy, so a later mutation cannot make the comparison pass. */
function snapshot<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/* ------------------------------------------------------------ undo / redo -- */

interface Session {
  circuit: Circuit;
  history: History;
}

function newSession(now?: () => number, options?: { depth?: number; coalesceMs?: number }): Session {
  return {
    circuit: createEmptyCircuit('verify'),
    history: new History({ ...options, now }),
  };
}

function runEdit(
  session: Session,
  label: string,
  mutate: (draft: Circuit) => void,
  mergeKey?: string,
): void {
  const result = createTransaction(session.circuit, label, mutate, mergeKey);
  if (result.command === null) return;
  session.history.record(result.command);
  session.circuit = result.circuit;
}

function undo(session: Session): void {
  const command = session.history.takeUndo();
  if (command === null) return;
  session.circuit = revertCommand(session.circuit, command);
}

function redo(session: Session): void {
  const command = session.history.takeRedo();
  if (command === null) return;
  session.circuit = applyCommand(session.circuit, command);
}

function checkUndoRoundTrip(check: Check): void {
  // 1. A scalar field edit on one entity.
  {
    const session = newSession();
    const before = snapshot(session.circuit);
    const id = session.circuit.neurons[7].id;
    runEdit(session, 'bias', (draft) => {
      const neuron = draft.neurons.find((candidate) => candidate.id === id);
      if (neuron !== undefined) neuron.bias = 999;
    });
    const edited = snapshot(session.circuit);
    check('undo.field.applied', session.circuit.neurons[7].bias === 999);
    check('undo.field.isolated', before.neurons[7].bias !== 999);
    undo(session);
    check('undo.field.roundtrip', deepEqual(session.circuit, before));
    redo(session);
    check('redo.field.roundtrip', deepEqual(session.circuit, edited));
  }

  // 2. A nested parameter object, mutated in place through the draft.
  {
    const session = newSession();
    const before = snapshot(session.circuit);
    runEdit(session, 'params', (draft) => {
      const neuron = draft.neurons[3];
      if (neuron.params.kind === 'izhikevich') neuron.params.d = 42;
      neuron.morphology.somaRadius = 3.25;
      neuron.position.x = -17.5;
    });
    const params = session.circuit.neurons[3].params;
    check('undo.nested.applied', params.kind === 'izhikevich' && params.d === 42);
    check(
      'undo.nested.no-aliasing',
      before.neurons[3].morphology.somaRadius !== 3.25 && before.neurons[3].position.x !== -17.5,
    );
    undo(session);
    check('undo.nested.roundtrip', deepEqual(session.circuit, before));
  }

  // 3. Structural change: insert and delete across several collections.
  {
    const session = newSession();
    const before = snapshot(session.circuit);
    const doomed = new Set<string>([session.circuit.neurons[0].id, session.circuit.neurons[1].id]);
    const added = makeNeuron(newNeuronId(), { label: 'extra', position: { x: 1, y: 2, z: 3 } });
    runEdit(session, 'structure', (draft) => {
      draft.neurons = [added, ...draft.neurons.filter((neuron) => !doomed.has(neuron.id))];
      draft.synapses = draft.synapses.filter(
        (synapse) => !doomed.has(synapse.source) && !doomed.has(synapse.target),
      );
      draft.probes = draft.probes.filter((probe) => !doomed.has(probe.target));
      draft.name = 'restructured';
    });
    const edited = snapshot(session.circuit);
    check('undo.structure.applied', session.circuit.neurons.length === before.neurons.length - 1);
    check('undo.structure.order', session.circuit.neurons[0].id === added.id);
    check('undo.structure.name', session.circuit.name === 'restructured');
    undo(session);
    check('undo.structure.roundtrip', deepEqual(session.circuit, before));
    redo(session);
    check('redo.structure.roundtrip', deepEqual(session.circuit, edited));
    undo(session);
    check('undo.structure.roundtrip.twice', deepEqual(session.circuit, before));
  }

  // 4. A long chain unwinds to exactly the starting document.
  {
    const session = newSession();
    const before = snapshot(session.circuit);
    for (let i = 0; i < 24; i += 1) {
      runEdit(session, `step ${i}`, (draft) => {
        draft.neurons[i].noise = i;
        draft.simulation.dt = 0.05 + i * 0.001;
      });
    }
    check('undo.chain.depth', session.history.undoDepth === 24);
    for (let i = 0; i < 24; i += 1) undo(session);
    check('undo.chain.roundtrip', deepEqual(session.circuit, before));
    check('undo.chain.exhausted', session.history.undoDepth === 0);
  }

  // 5. An externally supplied Command with hand-written inverses.
  {
    const session = newSession();
    const before = snapshot(session.circuit);
    const command: Command = {
      label: 'external',
      apply(draft) {
        draft.description = 'changed';
        draft.tags = [...draft.tags, 'external'];
      },
      revert(draft) {
        draft.description = before.description;
        draft.tags = before.tags;
      },
    };
    session.circuit = applyCommand(session.circuit, command);
    session.history.record(command);
    check('undo.external.applied', session.circuit.description === 'changed');
    check('undo.external.isolated', before.description !== 'changed');
    undo(session);
    check('undo.external.roundtrip', deepEqual(session.circuit, before));
  }

  // 6. A transaction that changes nothing must not produce a history entry.
  {
    const session = newSession();
    runEdit(session, 'noop', () => undefined);
    check('undo.noop.ignored', session.history.undoDepth === 0);
  }
}

/* ------------------------------------------------------------- coalescing -- */

function checkCoalescing(check: Check): void {
  let clock = 1000;
  const session = newSession(() => clock, { coalesceMs: 400 });
  const before = snapshot(session.circuit);
  const id = session.circuit.neurons[2].id;

  const drag = (weight: number): void => {
    runEdit(
      session,
      'Edit synapse',
      (draft) => {
        const neuron = draft.neurons.find((candidate) => candidate.id === id);
        if (neuron !== undefined) neuron.bias = weight;
      },
      'drag:bias',
    );
  };

  for (let i = 0; i < 400; i += 1) {
    clock += 8;
    drag(100 + i);
  }
  check('coalesce.single-entry', session.history.undoDepth === 1);
  check('coalesce.final-value', session.circuit.neurons[2].bias === 499);

  undo(session);
  check('coalesce.roundtrip', deepEqual(session.circuit, before));
  check('coalesce.emptied', session.history.undoDepth === 0);

  redo(session);
  check('coalesce.redo', session.circuit.neurons[2].bias === 499);

  // Outside the window, the same merge key must start a new entry.
  const spaced = newSession(() => clock, { coalesceMs: 400 });
  clock = 0;
  for (let i = 0; i < 3; i += 1) {
    clock += 1000;
    runEdit(
      spaced,
      'spaced',
      (draft) => {
        draft.neurons[0].bias = 10 + i;
      },
      'drag:bias',
    );
  }
  check('coalesce.window-respected', spaced.history.undoDepth === 3);

  // Different merge keys never merge, however fast they arrive.
  const distinct = newSession(() => clock, { coalesceMs: 400 });
  for (let i = 0; i < 3; i += 1) {
    runEdit(
      distinct,
      'distinct',
      (draft) => {
        draft.neurons[i].bias = 5;
      },
      `drag:${i}`,
    );
  }
  check('coalesce.distinct-keys', distinct.history.undoDepth === 3);

  // An edit with no merge key is never absorbed into the previous entry.
  const unkeyed = newSession(() => clock, { coalesceMs: 400 });
  for (let i = 0; i < 3; i += 1) {
    runEdit(unkeyed, 'unkeyed', (draft) => {
      draft.neurons[i].noise = 1;
    });
  }
  check('coalesce.unkeyed', unkeyed.history.undoDepth === 3);

  // An undo breaks the run: the next edit may not merge across it.
  const broken = newSession(() => clock, { coalesceMs: 400 });
  runEdit(broken, 'a', (draft) => { draft.neurons[0].bias = 1; }, 'drag:bias');
  runEdit(broken, 'b', (draft) => { draft.neurons[0].bias = 2; }, 'drag:bias');
  check('coalesce.merged-before-undo', broken.history.undoDepth === 1);
  undo(broken);
  runEdit(broken, 'c', (draft) => { draft.neurons[0].bias = 3; }, 'drag:bias');
  check('coalesce.no-merge-across-undo', broken.history.undoDepth === 1);
  check('coalesce.redo-cleared', broken.history.redoDepth === 0);
}

function checkHistoryBounds(check: Check): void {
  let clock = 0;
  const session = newSession(() => (clock += 1000), { depth: 8 });
  for (let i = 0; i < 40; i += 1) {
    runEdit(session, `edit ${i}`, (draft) => {
      draft.neurons[i % 30].noise = i;
    });
  }
  check('history.bounded', session.history.undoDepth === 8);
  for (let i = 0; i < 20; i += 1) undo(session);
  check('history.drained', session.history.undoDepth === 0);
  check('history.redo-filled', session.history.redoDepth === 8);
}

/* ----------------------------------------------------------- connectivity -- */

interface Wired {
  circuit: Circuit;
  source: Population;
  target: Population;
}

function population(name: string, size: number, polarity: Neuron['polarity'], origin: number): PopulationSpec {
  return {
    name,
    size,
    polarity,
    model: 'izhikevich',
    layout: { kind: 'sphere', radius: 12, jitter: 0.5, seed: 0x1234 },
    origin: { x: origin, y: 0, z: 0 },
  };
}

function wire(sourceSize: number, targetSize: number): Wired {
  const source = instantiatePopulation(population('Src', sourceSize, 'excitatory', -40));
  const target = instantiatePopulation(population('Dst', targetSize, 'inhibitory', 40));
  const circuit: Circuit = {
    ...createEmptyCircuit('wire'),
    neurons: [...source.neurons, ...target.neurons],
    synapses: [],
    populations: [source.population, target.population],
    projections: [],
    probes: [],
  };
  return { circuit, source: source.population, target: target.population };
}

function projection(bed: Wired, rule: ProjectionSpec['rule'], name = 'P'): Synapse[] {
  return instantiateProjection(
    {
      name,
      source: bed.source.id,
      target: bed.target.id,
      rule,
      weightMean: 1,
      weightJitter: 0.2,
      delayMean: 1.5,
      delayJitter: 0.5,
    },
    bed.circuit,
  );
}

/** Same topology and same drawn values, ignoring the freshly minted ids. */
function sameWiring(a: readonly Synapse[], b: readonly Synapse[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (
      a[i].source !== b[i].source ||
      a[i].target !== b[i].target ||
      a[i].weight !== b[i].weight ||
      a[i].delay !== b[i].delay ||
      a[i].receptor !== b[i].receptor
    ) {
      return false;
    }
  }
  return true;
}

function hasSelfConnection(synapses: readonly Synapse[]): boolean {
  return synapses.some((synapse) => synapse.source === synapse.target);
}

function maxDuplicatePartners(synapses: readonly Synapse[], by: 'target' | 'source'): number {
  const seen = new Map<string, Set<string>>();
  let worst = 0;
  for (const synapse of synapses) {
    const key = by === 'target' ? synapse.target : synapse.source;
    const other = by === 'target' ? synapse.source : synapse.target;
    let partners = seen.get(key);
    if (partners === undefined) {
      partners = new Set<string>();
      seen.set(key, partners);
    }
    const sizeBefore = partners.size;
    partners.add(other);
    if (partners.size === sizeBefore) worst += 1;
  }
  return worst;
}

function checkConnectivity(check: Check): void {
  const bed = wire(20, 16);

  // all-to-all across disjoint populations is exactly the product.
  {
    const synapses = projection(bed, { kind: 'all-to-all', selfConnections: false });
    check('rule.all-to-all.count', synapses.length === 20 * 16);
    check(
      'rule.all-to-all.determinism',
      sameWiring(synapses, projection(bed, { kind: 'all-to-all', selfConnections: false })),
    );
    check('rule.all-to-all.receptor', synapses.every((synapse) => synapse.receptor === 'ampa'));
    check(
      'rule.all-to-all.kinetics',
      synapses.every((synapse) => deepEqual(synapse.kinetics, RECEPTOR_DEFAULTS.ampa)),
    );
  }

  // one-to-one pairs by index, bounded by the smaller population.
  {
    const synapses = projection(bed, { kind: 'one-to-one' });
    check('rule.one-to-one.count', synapses.length === 16);
    check(
      'rule.one-to-one.pairing',
      synapses.every((synapse, i) => synapse.source === bed.source.members[i]),
    );
    check('rule.one-to-one.determinism', sameWiring(synapses, projection(bed, { kind: 'one-to-one' })));
  }

  // random at the boundaries is exact; in between it is binomial.
  {
    const all = projection(bed, { kind: 'random', probability: 1, seed: 7, selfConnections: false });
    const none = projection(bed, { kind: 'random', probability: 0, seed: 7, selfConnections: false });
    const half = projection(bed, { kind: 'random', probability: 0.5, seed: 7, selfConnections: false });
    check('rule.random.saturated', all.length === 20 * 16);
    check('rule.random.empty', none.length === 0);
    check('rule.random.expected', Math.abs(half.length - 160) < 40);
    check(
      'rule.random.determinism',
      sameWiring(half, projection(bed, { kind: 'random', probability: 0.5, seed: 7, selfConnections: false })),
    );
    check(
      'rule.random.seed-sensitivity',
      !sameWiring(half, projection(bed, { kind: 'random', probability: 0.5, seed: 8, selfConnections: false })),
    );
  }

  // gaussian with an enormous sigma degenerates to all-to-all at peak probability.
  {
    const wide = projection(bed, { kind: 'gaussian', sigma: 1e6, maxProbability: 1, seed: 11 });
    const narrow = projection(bed, { kind: 'gaussian', sigma: 0.01, maxProbability: 1, seed: 11 });
    check('rule.gaussian.saturated', wide.length === 20 * 16);
    check('rule.gaussian.local', narrow.length === 0);
    check(
      'rule.gaussian.determinism',
      sameWiring(wide, projection(bed, { kind: 'gaussian', sigma: 1e6, maxProbability: 1, seed: 11 })),
    );
    const partial = projection(bed, { kind: 'gaussian', sigma: 60, maxProbability: 1, seed: 11 });
    check('rule.gaussian.decays', partial.length > 0 && partial.length < 20 * 16);
  }

  // distance-threshold: the two populations sit 80 units apart on x.
  {
    const near = projection(bed, { kind: 'distance-threshold', radius: 10, probability: 1, seed: 3 });
    const far = projection(bed, { kind: 'distance-threshold', radius: 1000, probability: 1, seed: 3 });
    check('rule.distance-threshold.excluded', near.length === 0);
    check('rule.distance-threshold.included', far.length === 20 * 16);
    check(
      'rule.distance-threshold.determinism',
      sameWiring(far, projection(bed, { kind: 'distance-threshold', radius: 1000, probability: 1, seed: 3 })),
    );
  }

  // fixed degrees: exact counts, and no partner drawn twice.
  {
    const inDegree = projection(bed, { kind: 'fixed-in-degree', degree: 5, seed: 21 });
    check('rule.fixed-in-degree.count', inDegree.length === 16 * 5);
    check('rule.fixed-in-degree.no-replacement', maxDuplicatePartners(inDegree, 'target') === 0);
    check(
      'rule.fixed-in-degree.determinism',
      sameWiring(inDegree, projection(bed, { kind: 'fixed-in-degree', degree: 5, seed: 21 })),
    );
    const saturatedIn = projection(bed, { kind: 'fixed-in-degree', degree: 999, seed: 21 });
    check('rule.fixed-in-degree.clamped', saturatedIn.length === 16 * 20);

    const outDegree = projection(bed, { kind: 'fixed-out-degree', degree: 4, seed: 22 });
    check('rule.fixed-out-degree.count', outDegree.length === 20 * 4);
    check('rule.fixed-out-degree.no-replacement', maxDuplicatePartners(outDegree, 'source') === 0);
    check(
      'rule.fixed-out-degree.determinism',
      sameWiring(outDegree, projection(bed, { kind: 'fixed-out-degree', degree: 4, seed: 22 })),
    );
    const saturatedOut = projection(bed, { kind: 'fixed-out-degree', degree: 999, seed: 22 });
    check('rule.fixed-out-degree.clamped', saturatedOut.length === 20 * 16);
  }

  // Self-connections: only all-to-all and random may opt in, and only they do.
  {
    const self = instantiatePopulation(population('Self', 12, 'excitatory', 0));
    const circuit: Circuit = {
      ...createEmptyCircuit('self'),
      neurons: self.neurons,
      synapses: [],
      populations: [self.population],
      projections: [],
      probes: [],
    };
    const recurrent = (rule: ProjectionSpec['rule']): Synapse[] =>
      instantiateProjection(
        { name: 'R', source: self.population.id, target: self.population.id, rule },
        circuit,
      );

    check(
      'rule.self.all-to-all.rejected',
      recurrent({ kind: 'all-to-all', selfConnections: false }).length === 12 * 11,
    );
    check(
      'rule.self.all-to-all.allowed',
      recurrent({ kind: 'all-to-all', selfConnections: true }).length === 12 * 12,
    );
    check(
      'rule.self.random.rejected',
      !hasSelfConnection(recurrent({ kind: 'random', probability: 1, seed: 5, selfConnections: false })),
    );
    check('rule.self.one-to-one.rejected', recurrent({ kind: 'one-to-one' }).length === 0);
    check(
      'rule.self.gaussian.rejected',
      !hasSelfConnection(recurrent({ kind: 'gaussian', sigma: 1e6, maxProbability: 1, seed: 5 })),
    );
    check(
      'rule.self.distance-threshold.rejected',
      !hasSelfConnection(recurrent({ kind: 'distance-threshold', radius: 1e6, probability: 1, seed: 5 })),
    );

    const inDegree = recurrent({ kind: 'fixed-in-degree', degree: 4, seed: 5 });
    check('rule.self.fixed-in-degree.rejected', !hasSelfConnection(inDegree));
    check('rule.self.fixed-in-degree.count', inDegree.length === 12 * 4);
    check('rule.self.fixed-in-degree.no-replacement', maxDuplicatePartners(inDegree, 'target') === 0);

    const outDegree = recurrent({ kind: 'fixed-out-degree', degree: 4, seed: 5 });
    check('rule.self.fixed-out-degree.rejected', !hasSelfConnection(outDegree));
    check('rule.self.fixed-out-degree.count', outDegree.length === 12 * 4);
    check('rule.self.fixed-out-degree.no-replacement', maxDuplicatePartners(outDegree, 'source') === 0);

    // An inhibitory source must release GABA-A with the shared kinetics.
    const inhibitory = instantiatePopulation(population('Inh', 6, 'inhibitory', 0));
    const inhibitoryCircuit: Circuit = {
      ...circuit,
      neurons: [...self.neurons, ...inhibitory.neurons],
      populations: [self.population, inhibitory.population],
    };
    const gaba = instantiateProjection(
      {
        name: 'G',
        source: inhibitory.population.id,
        target: self.population.id,
        rule: { kind: 'all-to-all', selfConnections: false },
      },
      inhibitoryCircuit,
    );
    check('rule.polarity.gabaa', gaba.every((synapse) => synapse.receptor === 'gabaa'));
    check(
      'rule.polarity.gabaa.kinetics',
      gaba.every((synapse) => deepEqual(synapse.kinetics, RECEPTOR_DEFAULTS.gabaa)),
    );
  }

  // Weight and delay draws respect their bounds.
  {
    const synapses = instantiateProjection(
      {
        name: 'Bounds',
        source: bed.source.id,
        target: bed.target.id,
        rule: { kind: 'all-to-all', selfConnections: false },
        weightMean: 2,
        weightJitter: 0.5,
        delayMean: 3,
        delayJitter: 1,
      },
      bed.circuit,
    );
    check(
      'rule.weights.bounded',
      synapses.every((synapse) => synapse.weight >= 1.5 - 1e-9 && synapse.weight <= 2.5 + 1e-9),
    );
    check(
      'rule.delays.bounded',
      synapses.every((synapse) => synapse.delay >= 2 - 1e-9 && synapse.delay <= 4 + 1e-9),
    );
    check('rule.delays.positive', synapses.every((synapse) => synapse.delay > 0));
  }

  // An unresolvable projection produces nothing rather than throwing.
  {
    const orphan = instantiateProjection(
      {
        name: 'Orphan',
        source: asPopulationId('missing'),
        target: bed.target.id,
        rule: { kind: 'all-to-all', selfConnections: false },
      },
      bed.circuit,
    );
    check('rule.missing-population', orphan.length === 0);
  }
}

/* -------------------------------------------------------- default circuit -- */

function checkDefaultCircuit(check: Check): void {
  const circuit = createEmptyCircuit();

  const ids = new Set<string>();
  for (const neuron of circuit.neurons) ids.add(neuron.id);
  check('circuit.neuron-ids-unique', ids.size === circuit.neurons.length);
  check('circuit.version', circuit.version === CIRCUIT_SCHEMA_VERSION);

  const excitatory = circuit.neurons.filter((neuron) => neuron.polarity === 'excitatory');
  const inhibitory = circuit.neurons.filter((neuron) => neuron.polarity === 'inhibitory');
  check('circuit.excitatory-count', excitatory.length === 60);
  check('circuit.inhibitory-count', inhibitory.length === 15);

  check(
    'circuit.synapse-endpoints',
    circuit.synapses.every((synapse) => ids.has(synapse.source) && ids.has(synapse.target)),
  );
  check('circuit.no-autapses', !hasSelfConnection(circuit.synapses));
  check(
    'circuit.synapse-count',
    circuit.synapses.length > 250 && circuit.synapses.length < 1500,
  );

  const polarityOf = new Map<string, Neuron['polarity']>();
  for (const neuron of circuit.neurons) polarityOf.set(neuron.id, neuron.polarity);
  check(
    'circuit.receptor-matches-polarity',
    circuit.synapses.every((synapse) => {
      const expected = polarityOf.get(synapse.source) === 'inhibitory' ? 'gabaa' : 'ampa';
      return synapse.receptor === expected && deepEqual(synapse.kinetics, RECEPTOR_DEFAULTS[expected]);
    }),
  );

  check(
    'circuit.population-members-resolve',
    circuit.populations.every((pop) => pop.members.every((member) => ids.has(member))),
  );
  check(
    'circuit.population-size-matches',
    circuit.populations.every((pop) => pop.members.length === pop.size),
  );
  const populationIds = new Set<string>(circuit.populations.map((pop) => pop.id));
  check(
    'circuit.projections-resolve',
    circuit.projections.every(
      (proj) => populationIds.has(proj.source) && populationIds.has(proj.target),
    ),
  );
  check('circuit.probes-resolve', circuit.probes.every((probe) => ids.has(probe.target)));

  // Dynamics. Rheobase for these Izhikevich parameters is 100 pA; every neuron
  // must sit below it on the mean, with enough noise to cross occasionally.
  const izhikevich = circuit.neurons.every((neuron) => neuron.params.kind === 'izhikevich');
  check('circuit.model', izhikevich);
  check(
    'circuit.subrheobase-drive',
    circuit.neurons.every((neuron) => neuron.bias > 0 && neuron.bias < 100),
  );
  check(
    'circuit.noise-present',
    circuit.neurons.every((neuron) => neuron.noise > 40 && neuron.noise < 200),
  );
  check('circuit.all-enabled', circuit.neurons.every((neuron) => neuron.enabled));

  // Recurrent excitation must stay subcritical: the mean excitatory in-degree
  // times the per-input spike probability has to be well under one.
  const excitatoryIds = new Set<string>(excitatory.map((neuron) => neuron.id));
  let excitatoryEdges = 0;
  let inhibitoryEdges = 0;
  let excitatoryWeight = 0;
  let inhibitoryWeight = 0;
  for (const synapse of circuit.synapses) {
    if (!excitatoryIds.has(synapse.target)) continue;
    if (excitatoryIds.has(synapse.source)) {
      excitatoryEdges += 1;
      excitatoryWeight += synapse.weight;
    } else {
      inhibitoryEdges += 1;
      inhibitoryWeight += synapse.weight;
    }
  }
  const excitatoryInDegree = excitatoryEdges / excitatory.length;
  check('circuit.recurrent-excitation', excitatoryInDegree > 2 && excitatoryInDegree < 9);
  check('circuit.inhibition-present', inhibitoryEdges / excitatory.length > 2);
  // Charge per spike scales with weight times the conductance integral, which is
  // 2.5x larger for GABA-A than for AMPA. Total inhibitory drive onto the
  // excitatory pool must exceed the recurrent excitatory drive it opposes.
  check('circuit.inhibition-dominates', inhibitoryWeight * 2.5 > excitatoryWeight);

  check('circuit.simulation-defaults', circuit.simulation.dt > 0 && circuit.simulation.dt <= 0.5);
  check('circuit.named', createEmptyCircuit('Named').name === 'Named');

  // Two circuits differ only in their identifiers.
  const other = createEmptyCircuit();
  check('circuit.reproducible-topology', other.synapses.length === circuit.synapses.length);
  check(
    'circuit.reproducible-positions',
    other.neurons.every((neuron, i) => deepEqual(neuron.position, circuit.neurons[i].position)),
  );
  check('circuit.distinct-ids', other.id !== circuit.id && other.neurons[0].id !== circuit.neurons[0].id);

  // The seeded document must survive an edit/undo cycle unchanged.
  const session: Session = { circuit, history: new History() };
  const before = snapshot(circuit);
  const victim: NeuronId = circuit.neurons[10].id;
  runEdit(session, 'probe', (draft) => {
    const neuron = draft.neurons.find((candidate) => candidate.id === victim);
    if (neuron !== undefined) neuron.enabled = false;
  });
  undo(session);
  check('circuit.survives-undo', deepEqual(session.circuit, before));
}
