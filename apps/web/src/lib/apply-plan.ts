import { useEditor } from '@neuroforge/editor';
import type { CircuitAction, AiPlan } from '@neuroforge/ai';
import { newStimulusId } from '@neuroforge/shared';
import type { NeuronId, PopulationId, Stimulus } from '@neuroforge/shared';

/**
 * Apply a validated plan to the document.
 *
 * The AI package emits plans and never touches the store; the editor owns
 * mutation and knows nothing about plans. This is the seam between them, and it
 * lives in the app because it is the only layer allowed to depend on both.
 *
 * Population references in a plan are by *name*, because the model has no way to
 * know the ids a previous action will mint. Names are therefore resolved against
 * the live document after every action rather than once up front.
 */
export interface ApplyResult {
  applied: number;
  skipped: number;
  errors: string[];
}

function findPopulationId(name: string): PopulationId | null {
  const target = name.trim().toLowerCase();
  const match = useEditor
    .getState()
    .circuit.populations.find((p) => p.name.trim().toLowerCase() === target);
  return match ? match.id : null;
}

function applyAction(action: CircuitAction, errors: string[]): boolean {
  const store = useEditor.getState();

  switch (action.type) {
    case 'clear':
      store.newCircuit('Untitled circuit');
      return true;

    case 'create-population':
      store.addPopulation(action.spec);
      return true;

    case 'connect-populations': {
      const source = findPopulationId(action.spec.sourceName);
      const target = findPopulationId(action.spec.targetName);
      if (source === null || target === null) {
        errors.push(
          `Cannot connect "${action.spec.sourceName}" to "${action.spec.targetName}": ` +
            `${source === null ? action.spec.sourceName : action.spec.targetName} does not exist.`,
        );
        return false;
      }
      const { sourceName: _sourceName, targetName: _targetName, ...rest } = action.spec;
      store.connectPopulations({ ...rest, source, target });
      return true;
    }

    case 'set-simulation':
      store.setSimulationSettings(action.patch);
      return true;

    case 'set-render':
      store.setRenderSettings(action.patch);
      return true;

    case 'tune-population': {
      const id = findPopulationId(action.name);
      if (id === null) {
        errors.push(`Cannot tune "${action.name}": no population with that name.`);
        return false;
      }
      const circuit = useEditor.getState().circuit;
      const population = circuit.populations.find((p) => p.id === id);
      if (!population) return false;

      const members = population.members as readonly NeuronId[];
      store.transaction(`Tune ${population.name}`, (draft) => {
        const owned = new Set<string>(members);
        for (const neuron of draft.neurons) {
          if (!owned.has(neuron.id)) continue;
          // Model parameters are merged rather than replaced so a plan that
          // adjusts one constant does not silently reset the rest.
          neuron.params = { ...neuron.params, ...action.params } as typeof neuron.params;
          if (action.bias !== undefined) neuron.bias = action.bias;
          if (action.noise !== undefined) neuron.noise = action.noise;
        }
        const target = draft.populations.find((p) => p.id === id);
        if (target) {
          target.params = { ...target.params, ...action.params } as typeof target.params;
        }
      });
      return true;
    }

    case 'tune-projection': {
      const circuit = useEditor.getState().circuit;
      const projection = circuit.projections.find(
        (p) => p.name.trim().toLowerCase() === action.name.trim().toLowerCase(),
      );
      if (!projection) {
        errors.push(`Cannot tune "${action.name}": no projection with that name.`);
        return false;
      }
      const source = circuit.populations.find((p) => p.id === projection.source);
      const target = circuit.populations.find((p) => p.id === projection.target);
      if (!source || !target) return false;
      const sourceMembers = new Set<string>(source.members);
      const targetMembers = new Set<string>(target.members);

      store.transaction(`Tune ${projection.name}`, (draft) => {
        for (const synapse of draft.synapses) {
          if (!sourceMembers.has(synapse.source) || !targetMembers.has(synapse.target)) continue;
          if (action.weightMean !== undefined) synapse.weight = action.weightMean;
          if (action.delayMean !== undefined) synapse.delay = action.delayMean;
          if (action.plasticity !== undefined) synapse.plasticity.kind = action.plasticity;
        }
        const record = draft.projections.find((p) => p.id === projection.id);
        if (record) {
          if (action.weightMean !== undefined) record.weightMean = action.weightMean;
          if (action.delayMean !== undefined) record.delayMean = action.delayMean;
        }
      });
      return true;
    }

    case 'add-stimulus': {
      const id = findPopulationId(action.targetPopulation);
      if (id === null) {
        errors.push(`Cannot add stimulus: no population named "${action.targetPopulation}".`);
        return false;
      }
      const population = useEditor.getState().circuit.populations.find((p) => p.id === id);
      if (!population) return false;
      const stimulus: Stimulus = {
        id: newStimulusId(),
        name: action.name,
        targets: population.members,
        pattern: action.pattern,
        enabled: true,
      };
      store.transaction(`Add stimulus ${action.name}`, (draft) => {
        draft.stimuli = [...draft.stimuli, stimulus];
      });
      return true;
    }

    default:
      return false;
  }
}

/**
 * Run every action in order.
 *
 * A failing action is recorded and skipped rather than aborting the plan: a
 * model that names one population wrongly should still get the other nine
 * populations it asked for, and the caller surfaces what was dropped.
 */
export function applyPlan(plan: AiPlan): ApplyResult {
  const errors: string[] = [];
  let applied = 0;
  let skipped = 0;

  for (const action of plan.actions) {
    let ok: boolean;
    try {
      ok = applyAction(action, errors);
    } catch (error) {
      errors.push(`${action.type} failed: ${(error as Error).message}`);
      ok = false;
    }
    if (ok) applied += 1;
    else skipped += 1;
  }

  return { applied, skipped, errors };
}
