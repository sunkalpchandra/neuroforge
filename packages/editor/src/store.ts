/**
 * The editor store.
 *
 * This is the only object in NeuroForge that writes to a circuit document. Every
 * mutation funnels through `commit`, which runs the change against a
 * copy-on-write draft, records an undoable command, and prunes any selection or
 * hover reference the change invalidated.
 */

import { create } from 'zustand';

import type {
  CameraState,
  Circuit,
  Neuron,
  NeuronId,
  PopulationId,
  Projection,
  RenderSettings,
  SimulationSettings,
  Stimulus,
  Synapse,
  SynapseId,
} from '@neuroforge/shared';
import { newNeuronId } from '@neuroforge/shared';

import type { Command } from './commands';
import { History, applyCommand, createTransaction, revertCommand } from './commands';
import { createEmptyCircuit } from './circuit';
import { runDraft } from './draft';
import { arcFor, distanceBetween, makeNeuron, makeSynapse, newProjectionId } from './entities';
import type { PopulationSpec, ProjectionSpec } from './populations';
import { instantiatePopulation, instantiateProjection } from './populations';

export type Tool = 'select' | 'place' | 'connect' | 'erase' | 'probe' | 'stimulate' | 'pan';

export interface EditorState {
  circuit: Circuit;
  selection: readonly NeuronId[];
  selectedSynapses: readonly SynapseId[];
  hovered: NeuronId | null;
  tool: Tool;
  undoDepth: number;
  redoDepth: number;
  dirty: boolean;
  lastSavedAt: number;
  // panels
  inspectorOpen: boolean;
  builderOpen: boolean;
  libraryOpen: boolean;
  commandPaletteOpen: boolean;
}

export interface EditorActions {
  execute(command: Command): void;
  undo(): void;
  redo(): void;
  transaction(label: string, fn: (draft: Circuit) => void): void;

  addNeuron(partial?: Partial<Neuron>): NeuronId;
  removeNeurons(ids: readonly NeuronId[]): void;
  updateNeuron(id: NeuronId, patch: Partial<Neuron>): void;
  updateNeurons(ids: readonly NeuronId[], patch: Partial<Neuron>): void;
  connect(source: NeuronId, target: NeuronId, partial?: Partial<Synapse>): SynapseId | null;
  removeSynapses(ids: readonly SynapseId[]): void;
  updateSynapse(id: SynapseId, patch: Partial<Synapse>): void;

  addPopulation(spec: PopulationSpec): PopulationId;
  connectPopulations(spec: ProjectionSpec): void;

  select(ids: readonly NeuronId[], additive?: boolean): void;
  selectAll(): void;
  clearSelection(): void;
  setHovered(id: NeuronId | null): void;
  setTool(tool: Tool): void;

  setSimulationSettings(patch: Partial<SimulationSettings>): void;
  setRenderSettings(patch: Partial<RenderSettings>): void;
  setCamera(state: CameraState): void;

  loadCircuit(circuit: Circuit): void;
  newCircuit(name?: string): void;
  togglePanel(panel: 'inspector' | 'builder' | 'library' | 'commandPalette', open?: boolean): void;
}

const history = new History();

/* ---------------------------------------------------------------- helpers -- */

/** Patch keys, sorted, so two edits to the same field share a merge key. */
function patchSignature(patch: object): string {
  return Object.keys(patch).sort().join(',');
}

/** Cheap stable signature for a set of ids, used only as a merge key. */
function idsSignature(ids: readonly string[]): string {
  if (ids.length === 0) return '0';
  return `${ids.length}:${ids[0]}:${ids[ids.length - 1]}`;
}

function sameVec(a: { x: number; y: number; z: number }, b: { x: number; y: number; z: number }): boolean {
  return a.x === b.x && a.y === b.y && a.z === b.z;
}

function sameCamera(a: CameraState, b: CameraState): boolean {
  return a.fov === b.fov && a.mode === b.mode && sameVec(a.position, b.position) && sameVec(a.target, b.target);
}

/**
 * Copy the writable fields of a patch onto an entity in a draft. Readonly
 * identity fields are never taken from a patch: an edit may not change what an
 * entity *is*, only what it holds.
 */
function assignNeuron(target: Neuron, patch: Partial<Neuron>): void {
  if (patch.label !== undefined) target.label = patch.label;
  if (patch.position !== undefined) {
    target.position = { x: patch.position.x, y: patch.position.y, z: patch.position.z };
  }
  if (patch.params !== undefined) target.params = { ...patch.params } as Neuron['params'];
  if (patch.polarity !== undefined) target.polarity = patch.polarity;
  if (patch.morphology !== undefined) target.morphology = { ...patch.morphology };
  if (patch.population !== undefined) target.population = patch.population;
  if (patch.bias !== undefined) target.bias = patch.bias;
  if (patch.noise !== undefined) target.noise = patch.noise;
  if (patch.enabled !== undefined) target.enabled = patch.enabled;
}

function assignSynapse(target: Synapse, patch: Partial<Synapse>): void {
  if (patch.receptor !== undefined) target.receptor = patch.receptor;
  if (patch.weight !== undefined) target.weight = patch.weight;
  if (patch.delay !== undefined) target.delay = patch.delay;
  if (patch.kinetics !== undefined) target.kinetics = { ...patch.kinetics };
  if (patch.plasticity !== undefined) target.plasticity = { ...patch.plasticity };
  if (patch.stp !== undefined) target.stp = { ...patch.stp };
  if (patch.releaseProbability !== undefined) target.releaseProbability = patch.releaseProbability;
  if (patch.arc !== undefined) target.arc = patch.arc;
  if (patch.enabled !== undefined) target.enabled = patch.enabled;
}

interface References {
  selection: readonly NeuronId[];
  selectedSynapses: readonly SynapseId[];
  hovered: NeuronId | null;
}

/**
 * Drop selection and hover entries whose entity no longer exists. Returns the
 * same array instances when nothing was dropped, so subscribers that compare by
 * reference do not re-render.
 *
 * An entity can only disappear by the collection array itself being replaced:
 * the draft copies an array before writing to it and never mutates one in place,
 * so an unchanged array reference is proof that every id still resolves. Both
 * scans are gated on that, which is what keeps the edits that republish the
 * document without touching its entities — a camera orbit, which runs on every
 * animation frame — off an O(neurons) path. Scanning unconditionally cost 1.8 ms
 * per orbit frame on a 20 000-neuron document whenever anything was selected.
 */
function pruneReferences(circuit: Circuit, previous: Circuit, current: References): References {
  let selection = current.selection;
  let selectedSynapses = current.selectedSynapses;
  let hovered = current.hovered;

  if (circuit.neurons !== previous.neurons && (selection.length > 0 || hovered !== null)) {
    const live = new Set<string>();
    for (const neuron of circuit.neurons) live.add(neuron.id);
    if (selection.length > 0) {
      const kept = selection.filter((id) => live.has(id));
      if (kept.length !== selection.length) selection = kept;
    }
    if (hovered !== null && !live.has(hovered)) hovered = null;
  }

  if (circuit.synapses !== previous.synapses && selectedSynapses.length > 0) {
    const live = new Set<string>();
    for (const synapse of circuit.synapses) live.add(synapse.id);
    const kept = selectedSynapses.filter((id) => live.has(id));
    if (kept.length !== selectedSynapses.length) selectedSynapses = kept;
  }

  return { selection, selectedSynapses, hovered };
}

/* ------------------------------------------------------------------ store -- */

export const useEditor = create<EditorState & EditorActions>((set, get) => {
  /**
   * Publish a new document plus the history depths it produced. Every caller is
   * an edit, including undo, so the result always differs from what was last
   * saved; `dirty` is set unconditionally here.
   */
  const publish = (circuit: Circuit): void => {
    const state = get();
    if (circuit === state.circuit) {
      set({ undoDepth: history.undoDepth, redoDepth: history.redoDepth });
      return;
    }
    const references = pruneReferences(circuit, state.circuit, state);
    set({
      circuit,
      selection: references.selection,
      selectedSynapses: references.selectedSynapses,
      hovered: references.hovered,
      undoDepth: history.undoDepth,
      redoDepth: history.redoDepth,
      dirty: true,
    });
  };

  /** Run a mutation, record it as one undoable entry, and publish the result. */
  const edit = (label: string, mutate: (draft: Circuit) => void, mergeKey?: string): void => {
    const result = createTransaction(get().circuit, label, mutate, mergeKey);
    if (result.command === null) return;
    history.record(result.command);
    publish(result.circuit);
  };

  /** Run a mutation that is deliberately outside the undo history. */
  const editWithoutHistory = (mutate: (draft: Circuit) => void): void => {
    publish(runDraft(get().circuit, mutate, false).circuit);
  };

  const adopt = (circuit: Circuit): void => {
    history.clear();
    set({
      circuit,
      selection: [],
      selectedSynapses: [],
      hovered: null,
      undoDepth: 0,
      redoDepth: 0,
      dirty: false,
      lastSavedAt: Date.now(),
    });
  };

  return {
    circuit: createEmptyCircuit(),
    selection: [],
    selectedSynapses: [],
    hovered: null,
    tool: 'select',
    undoDepth: 0,
    redoDepth: 0,
    dirty: false,
    lastSavedAt: 0,
    inspectorOpen: true,
    builderOpen: false,
    libraryOpen: false,
    commandPaletteOpen: false,

    execute(command) {
      const circuit = get().circuit;
      const next = applyCommand(circuit, command);
      // A command that wrote nothing has nothing to undo either, since apply and
      // revert are required to be inverses. Recording it would put an entry in
      // the stack that no keystroke can visibly remove.
      if (next === circuit) return;
      history.record(command);
      publish(next);
    },

    undo() {
      const command = history.takeUndo();
      if (command === null) return;
      publish(revertCommand(get().circuit, command));
    },

    redo() {
      const command = history.takeRedo();
      if (command === null) return;
      publish(applyCommand(get().circuit, command));
    },

    transaction(label, fn) {
      edit(label, fn);
    },

    addNeuron(partial) {
      const id = partial?.id ?? newNeuronId();
      const neuron = makeNeuron(id, partial);
      edit('Add neuron', (draft) => {
        draft.neurons = [...draft.neurons, neuron];
      });
      return id;
    },

    removeNeurons(ids) {
      if (ids.length === 0) return;
      const doomed = new Set<string>(ids);
      const circuit = get().circuit;
      const survivors = circuit.neurons.filter((neuron) => !doomed.has(neuron.id));
      if (survivors.length === circuit.neurons.length) return;

      // Anything pointing at a removed neuron has to go with it, or the document
      // is left holding references the simulation cannot resolve.
      const synapses = circuit.synapses.filter(
        (synapse) => !doomed.has(synapse.source) && !doomed.has(synapse.target),
      );
      const populations = circuit.populations.map((population) => {
        const members = population.members.filter((member) => !doomed.has(member));
        if (members.length === population.members.length) return population;
        return { ...population, members, size: members.length };
      });
      // A stimulus that loses some but not all of its targets survives with a
      // shorter target list, which leaves the array the same length as before.
      // Comparing lengths would miss exactly that case and strand a reference to
      // a deleted neuron, so the rewrite is tracked explicitly.
      let stimuliChanged = false;
      const stimuli: Stimulus[] = [];
      for (const stimulus of circuit.stimuli) {
        const targets = stimulus.targets.filter((target) => !doomed.has(target));
        if (targets.length === stimulus.targets.length) {
          stimuli.push(stimulus);
          continue;
        }
        stimuliChanged = true;
        // With every target gone the stimulus drives nothing; it goes too.
        if (targets.length > 0) stimuli.push({ ...stimulus, targets });
      }
      const probes = circuit.probes.filter((probe) => !doomed.has(probe.target));

      edit('Delete neurons', (draft) => {
        draft.neurons = survivors;
        if (synapses.length !== circuit.synapses.length) draft.synapses = synapses;
        if (populations.some((population, i) => population !== circuit.populations[i])) {
          draft.populations = populations;
        }
        if (stimuliChanged) draft.stimuli = stimuli;
        if (probes.length !== circuit.probes.length) draft.probes = probes;
      });
    },

    updateNeuron(id, patch) {
      // Resolving the index against the published document rather than the draft
      // keeps the edit O(1) in proxies: only the neuron being changed is ever
      // reached through one.
      const index = get().circuit.neurons.findIndex((neuron) => neuron.id === id);
      if (index < 0) return;
      edit(
        'Edit neuron',
        (draft) => {
          assignNeuron(draft.neurons[index], patch);
        },
        `neuron:${id}:${patchSignature(patch)}`,
      );
    },

    updateNeurons(ids, patch) {
      if (ids.length === 0) return;
      const targets = new Set<string>(ids);
      const indices: number[] = [];
      get().circuit.neurons.forEach((neuron, index) => {
        if (targets.has(neuron.id)) indices.push(index);
      });
      if (indices.length === 0) return;
      edit(
        indices.length === 1 ? 'Edit neuron' : 'Edit neurons',
        (draft) => {
          for (const index of indices) assignNeuron(draft.neurons[index], patch);
        },
        `neurons:${idsSignature(ids)}:${patchSignature(patch)}`,
      );
    },

    connect(source, target, partial) {
      const { circuit } = get();
      const from = circuit.neurons.find((neuron) => neuron.id === source);
      const to = circuit.neurons.find((neuron) => neuron.id === target);
      if (from === undefined || to === undefined) return null;

      const synapse = makeSynapse(source, target, from.polarity, {
        arc: arcFor(distanceBetween(from.position, to.position)),
        ...partial,
      });
      edit('Connect', (draft) => {
        draft.synapses = [...draft.synapses, synapse];
      });
      return synapse.id;
    },

    removeSynapses(ids) {
      if (ids.length === 0) return;
      const doomed = new Set<string>(ids);
      const circuit = get().circuit;
      const survivors = circuit.synapses.filter((synapse) => !doomed.has(synapse.id));
      if (survivors.length === circuit.synapses.length) return;
      edit('Delete synapses', (draft) => {
        draft.synapses = survivors;
      });
    },

    updateSynapse(id, patch) {
      const index = get().circuit.synapses.findIndex((synapse) => synapse.id === id);
      if (index < 0) return;
      edit(
        'Edit synapse',
        (draft) => {
          assignSynapse(draft.synapses[index], patch);
        },
        `synapse:${id}:${patchSignature(patch)}`,
      );
    },

    addPopulation(spec) {
      const built = instantiatePopulation(spec);
      edit(`Add ${spec.name}`, (draft) => {
        draft.neurons = [...draft.neurons, ...built.neurons];
        draft.populations = [...draft.populations, built.population];
      });
      return built.population.id;
    },

    connectPopulations(spec) {
      const synapses = instantiateProjection(spec, get().circuit);
      if (synapses.length === 0) return;
      const projection: Projection = {
        id: newProjectionId(),
        name: spec.name,
        source: spec.source,
        target: spec.target,
        rule: spec.rule,
        weightMean: spec.weightMean ?? 1,
        weightJitter: spec.weightJitter ?? 0,
        delayMean: spec.delayMean ?? 1.5,
        delayJitter: spec.delayJitter ?? 0,
      };
      edit(`Connect ${spec.name}`, (draft) => {
        draft.synapses = [...draft.synapses, ...synapses];
        draft.projections = [...draft.projections, projection];
      });
    },

    select(ids, additive = false) {
      const { selection } = get();
      if (!additive) {
        if (selection.length === ids.length && selection.every((id, i) => id === ids[i])) return;
        set({ selection: [...ids] });
        return;
      }
      const merged = new Set<NeuronId>(selection);
      let added = false;
      for (const id of ids) {
        if (!merged.has(id)) {
          merged.add(id);
          added = true;
        }
      }
      if (!added) return;
      set({ selection: [...merged] });
    },

    selectAll() {
      const { circuit } = get();
      set({
        selection: circuit.neurons.map((neuron) => neuron.id),
        selectedSynapses: circuit.synapses.map((synapse) => synapse.id),
      });
    },

    clearSelection() {
      const { selection, selectedSynapses } = get();
      if (selection.length === 0 && selectedSynapses.length === 0) return;
      set({ selection: [], selectedSynapses: [] });
    },

    setHovered(id) {
      if (get().hovered === id) return;
      set({ hovered: id });
    },

    setTool(tool) {
      if (get().tool === tool) return;
      set({ tool });
    },

    setSimulationSettings(patch) {
      edit(
        'Simulation settings',
        (draft) => {
          draft.simulation = { ...draft.simulation, ...patch };
        },
        `simulation:${patchSignature(patch)}`,
      );
    },

    setRenderSettings(patch) {
      edit(
        'Render settings',
        (draft) => {
          draft.render = { ...draft.render, ...patch };
        },
        `render:${patchSignature(patch)}`,
      );
    },

    setCamera(state) {
      const { circuit } = get();
      if (sameCamera(circuit.camera, state)) return;
      // The camera moves every frame while orbiting. Recording it would bury the
      // undo stack under view changes, so it is written straight to the document.
      editWithoutHistory((draft) => {
        draft.camera = {
          position: { ...state.position },
          target: { ...state.target },
          fov: state.fov,
          mode: state.mode,
        };
      });
    },

    loadCircuit(circuit) {
      adopt(circuit);
    },

    newCircuit(name) {
      adopt(createEmptyCircuit(name));
    },

    togglePanel(panel, open) {
      const state = get();
      switch (panel) {
        case 'inspector':
          set({ inspectorOpen: open ?? !state.inspectorOpen });
          return;
        case 'builder':
          set({ builderOpen: open ?? !state.builderOpen });
          return;
        case 'library':
          set({ libraryOpen: open ?? !state.libraryOpen });
          return;
        case 'commandPalette':
          set({ commandPaletteOpen: open ?? !state.commandPaletteOpen });
          return;
      }
    },
  };
});
