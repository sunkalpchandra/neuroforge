/**
 * Descriptor types for the companion tables that accompany every shader in this
 * package. They exist so the renderer and the GPU integrator can wire uniforms
 * and bind groups from data instead of transcribing names out of shader source.
 */

/** GLSL uniform types used by the `*_UNIFORMS` tables. */
export type UniformType =
  | 'float'
  | 'int'
  | 'bool'
  | 'vec2'
  | 'vec3'
  | 'vec4'
  | 'mat3'
  | 'mat4'
  | 'sampler2D';

/** Map of uniform name to its GLSL type for one shader program. */
export type UniformTable = Readonly<Record<string, UniformType>>;

/** WebGPU binding kinds used by the WGSL bind group descriptions. */
export type BindingType = 'storage' | 'read-only-storage' | 'uniform';

/** One entry of a compute pipeline's bind group layout, in group 0. */
export interface ShaderBinding {
  readonly binding: number;
  readonly name: string;
  readonly type: BindingType;
}
