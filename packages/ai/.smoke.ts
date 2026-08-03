import { planLocally } from '@neuroforge/ai';
import { validatePlan } from '@neuroforge/ai';
import { createEmptyCircuit } from '@neuroforge/editor';
import type { Circuit } from '@neuroforge/shared';

const prompts = [
  'Create a hippocampal CA3 recurrent circuit with 200 excitatory neurons and inhibitory basket cells',
  'Make this oscillate at gamma frequency',
  'Add 50 fast-spiking interneurons',
  'Connect the excitatory population to the inhibitory one with 10% probability',
  'Make it sparser',
  'Make it denser',
  'Make it faster',
  'Make it slower',
  'Switch everything to Izhikevich',
  'Clear the circuit',
  'Build a cortical column with two hundred pyramidal cells and 50 basket cells, then make it oscillate at 40 Hz',
  'Add a poisson input of 200 pA at 20 Hz to the excitatory population',
  'Make a cerebellar network with 1k granule cells and purkinje cells',
  'Add 30 bursting thalamic relay neurons and sprinkle some magic pixie dust',
];

let circuit: Circuit = createEmptyCircuit('Smoke');

// Build a base circuit from prompt 1 so the follow-ups have something to act on.
const base = planLocally(prompts[0], circuit);
console.log('=== BASE ===');
console.log(JSON.stringify(base, null, 1));

// Fabricate the resulting document by hand so later prompts see populations.
import { instantiatePopulation, instantiateProjection } from '@neuroforge/editor';
for (const action of base.actions) {
  if (action.type === 'create-population') {
    const { population, neurons } = instantiatePopulation(action.spec);
    circuit = {
      ...circuit,
      populations: [...circuit.populations, population],
      neurons: [...circuit.neurons, ...neurons],
    };
  }
}
const byName = new Map(circuit.populations.map((p) => [p.name, p.id]));
for (const action of base.actions) {
  if (action.type === 'connect-populations') {
    const source = byName.get(action.spec.sourceName)!;
    const target = byName.get(action.spec.targetName)!;
    const synapses = instantiateProjection({ ...action.spec, source, target }, circuit);
    circuit = {
      ...circuit,
      synapses: [...circuit.synapses, ...synapses],
      projections: [
        ...circuit.projections,
        {
          id: action.spec.name,
          name: action.spec.name,
          source,
          target,
          rule: action.spec.rule,
          weightMean: action.spec.weightMean ?? 1,
          weightJitter: action.spec.weightJitter ?? 0,
          delayMean: action.spec.delayMean ?? 1,
          delayJitter: action.spec.delayJitter ?? 0,
        },
      ],
    };
  }
}
console.log(`\nBase circuit: ${circuit.neurons.length} neurons, ${circuit.synapses.length} synapses, ${circuit.populations.length} pops, ${circuit.projections.length} projections\n`);

for (const prompt of prompts) {
  const plan = planLocally(prompt, circuit);
  const { plan: safe, errors } = validatePlan(plan, circuit);
  console.log('----------------------------------------------------------');
  console.log('PROMPT :', prompt);
  console.log('SUMMARY:', plan.summary);
  console.log('ACTIONS:', JSON.stringify(plan.actions));
  console.log('WARN   :', JSON.stringify(plan.warnings));
  if (errors.length) console.log('VALIDATION ERRORS:', JSON.stringify(errors));
  if (safe.actions.length !== plan.actions.length) console.log('!! DROPPED', plan.actions.length - safe.actions.length);
}
