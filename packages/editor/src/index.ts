/**
 * @neuroforge/editor
 *
 * The document model's only writer: the Zustand store, the undo/redo command
 * system it records into, the population and projection builders, and the
 * keyboard map.
 */

export type { Command } from './commands';

export type { EditorActions, EditorState, Tool } from './store';
export { useEditor } from './store';

export type { PopulationSpec, ProjectionSpec } from './populations';
export { instantiatePopulation, instantiateProjection } from './populations';

export { createEmptyCircuit } from './circuit';

export type { Shortcut } from './shortcuts';
export { buildShortcuts } from './shortcuts';

export { verifyEditor } from './__verify';
