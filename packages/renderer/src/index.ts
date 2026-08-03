/**
 * @neuroforge/renderer
 *
 * Pure Three.js. No React: the app wraps these objects in R3F components.
 *
 * The scene is built out of four instanced fields (neurons, axons, spike
 * particles, selection halos) plus an analytic grid, all driven from the
 * simulation's structure-of-arrays buffers. Nothing here allocates on a frame
 * path and nothing rebuilds geometry when state changes; a frame is a linear
 * pass writing floats into buffers that already exist.
 */

export { buildAxonGeometry, buildDendriteGeometry, buildSomaGeometry } from './glyph';
export { GlyphLibrary } from './glyph-library';
export { NeuronField } from './neuron-field';
export { AxonField } from './axon-field';
export { SpikeParticles } from './particles';
export { InfiniteGrid } from './grid';
export { SelectionOverlay } from './selection';
export { createRenderer } from './renderer';
export { CameraRig } from './camera';
export type { CameraMode } from './camera';
