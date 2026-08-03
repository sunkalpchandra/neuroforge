/**
 * @neuroforge/shaders
 *
 * WGSL compute kernels and GLSL programs as plain strings. No bundler loader is
 * involved: every shader is a template literal in a `.ts` file, prefixed with a
 * `wgsl` or `glsl` comment so editors highlight it.
 *
 * The GLSL programs follow the `THREE.ShaderMaterial` convention — no `#version`
 * and no precision header, and the attributes, matrices and varying aliases
 * Three injects are used rather than redeclared. Each fragment program already
 * contains the shared chunks it needs; the chunks carry include guards, so
 * prepending them again is harmless.
 *
 * The `*_BINDINGS` tables describe group 0 of each compute pipeline, and the
 * `*_UNIFORMS` tables name and type every uniform of each GLSL program, so both
 * sides can be wired from data rather than from reading shader source.
 */

export type { BindingType, ShaderBinding, UniformTable, UniformType } from './types';

export { SYNAPSE_CURRENT_SCALE, WGSL_WORKGROUP_SIZE } from './wgsl/common';

export { NEURON_INTEGRATE_BINDINGS, NEURON_INTEGRATE_WGSL } from './wgsl/integrate';
export {
  SYNAPSE_PROPAGATE_BINDINGS,
  SYNAPSE_PROPAGATE_WGSL,
  SYNAPSE_STDP_BINDINGS,
  SYNAPSE_STDP_WGSL,
} from './wgsl/synapse';
export {
  PARTICLE_COMPUTE_BINDINGS,
  PARTICLE_EMIT_WGSL,
  PARTICLE_UPDATE_WGSL,
} from './wgsl/particles';

export { FOG_GLSL, NOISE_GLSL, VOLTAGE_RAMP_GLSL } from './glsl/common';
export { NEURON_FRAGMENT_GLSL, NEURON_UNIFORMS, NEURON_VERTEX_GLSL } from './glsl/neuron';
export { AXON_FRAGMENT_GLSL, AXON_UNIFORMS, AXON_VERTEX_GLSL } from './glsl/axon';
export { GRID_FRAGMENT_GLSL, GRID_UNIFORMS, GRID_VERTEX_GLSL } from './glsl/grid';
export { PARTICLE_FRAGMENT_GLSL, PARTICLE_UNIFORMS, PARTICLE_VERTEX_GLSL } from './glsl/particle';
