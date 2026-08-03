import type { ShaderBinding } from '../types';
import {
  WGSL_NEURON_STRUCTS,
  WGSL_PARTICLE_STRUCTS,
  WGSL_PRELUDE,
  WGSL_SPLINE,
  WGSL_SYNAPSE_STRUCTS,
  WGSL_WORKGROUP_SIZE,
} from './common';

/**
 * Bindings shared by both particle kernels.
 *
 * The two entry points live in separate modules but declare an identical layout
 * so one bind group and one explicit `GPUPipelineLayout` serve both. Each kernel
 * leaves the bindings it does not touch unread; with an explicit pipeline layout
 * that is legal and avoids rebinding between the emit and update dispatches.
 */
const PARTICLE_BINDING_BLOCK = /* wgsl */ `
struct ParticleUniforms {
  @align(16) dt : f32,
  time : f32,
  speed : f32,
  life : f32,
  size : f32,
  density : f32,
  speedJitter : f32,
  sizeJitter : f32,
  capacity : u32,
  synapseCount : u32,
  seed : u32,
  step : u32,
}

@group(0) @binding(0) var<uniform> uni : ParticleUniforms;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> counter : ParticleCounter;
@group(0) @binding(3) var<storage, read> synapses : array<SynapseStatic>;
@group(0) @binding(4) var<storage, read> positions : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> outputs : array<NeuronOutput>;
`;

const PARTICLE_COMMON = [
  WGSL_PRELUDE,
  WGSL_NEURON_STRUCTS,
  WGSL_SYNAPSE_STRUCTS,
  WGSL_PARTICLE_STRUCTS,
  WGSL_SPLINE,
  PARTICLE_BINDING_BLOCK,
].join('\n');

/**
 * Advects impulse particles along axon splines.
 *
 * One invocation per particle slot. The axon is the same quadratic Bezier the
 * renderer draws, so a particle is stored as a curve parameter rather than a
 * free position and can never drift off the ribbon. Advancing that parameter by
 * `speed * dt / |dP/du|` moves the particle a fixed arc length per unit time
 * regardless of how the parameterisation bunches up near the sag, which is why
 * no arc-length lookup table is needed on the GPU.
 *
 * A particle retires when it reaches the terminal or outlives its lifetime; its
 * slot is then free for the emitter to claim. The host zeroes `counter.live`
 * before dispatching so the tally this kernel builds is the live count for the
 * frame.
 */
export const PARTICLE_UPDATE_WGSL = /* wgsl */ `
${PARTICLE_COMMON}

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= uni.capacity) {
    return;
  }

  var particle = particles[index];
  if (particle.life <= 0.0) {
    return;
  }

  let dt = uni.dt;
  let syn = synapses[particle.synapse];
  let from = positions[syn.pre].xyz;
  let to = positions[syn.post].xyz;

  let tangent = axonTangent(from, to, syn.arc, particle.u);
  let arcPerParameter = max(length(tangent), EPSILON);
  let travelled = particle.speed * dt;

  particle.u = particle.u + travelled / arcPerParameter;
  particle.distance = particle.distance + travelled;
  particle.life = particle.life - dt;

  if (particle.u >= 1.0 || particle.life <= 0.0) {
    particle.u = 1.0;
    particle.life = 0.0;
    particle.position = to;
    particle.velocity = vec3<f32>(0.0, 0.0, 0.0);
    particles[index] = particle;
    return;
  }

  let next = axonPoint(from, to, syn.arc, particle.u);
  particle.velocity = (next - particle.position) / max(dt, EPSILON);
  particle.position = next;

  particles[index] = particle;
  atomicAdd(&counter.live, 1u);
}
`;

/**
 * Emits impulse particles for the spikes published by the integrator.
 *
 * One invocation per synapse: a synapse emits when its presynaptic neuron fired
 * this step. Slots are claimed from a monotonically increasing cursor modulo the
 * capacity, so emission is a single atomic with no free list and no compaction;
 * under sustained pressure the oldest particles are overwritten, which is the
 * correct visual degradation for a spike shower.
 *
 * `density` below 1 thins emission stochastically; above 1 it emits several
 * particles per spike, each with its own speed and size jitter so they separate
 * into a train rather than travelling as one blob.
 */
export const PARTICLE_EMIT_WGSL = /* wgsl */ `
${PARTICLE_COMMON}

const MAX_EMIT_PER_SPIKE : u32 = 4u;

@compute @workgroup_size(${WGSL_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= uni.synapseCount || uni.capacity == 0u) {
    return;
  }

  let syn = synapses[index];
  if (!metaEnabled(syn.meta) || outputs[syn.pre].spike == 0u) {
    return;
  }

  let root = hashCombine(index ^ uni.seed, uni.step);
  let density = clamp(uni.density, 0.0, f32(MAX_EMIT_PER_SPIKE));
  var emitCount = u32(floor(density));
  if (randomUnit(root) < density - floor(density)) {
    emitCount = emitCount + 1u;
  }
  if (emitCount == 0u) {
    return;
  }

  let from = positions[syn.pre].xyz;
  let to = positions[syn.post].xyz;
  let origin = axonPoint(from, to, syn.arc, 0.0);

  // Jitter is held below 1 so neither the speed nor the size can be driven
  // negative by a badly chosen setting.
  let speedJitter = clamp(uni.speedJitter, 0.0, 0.9);
  let sizeJitter = clamp(uni.sizeJitter, 0.0, 0.9);

  for (var k : u32 = 0u; k < emitCount; k = k + 1u) {
    let noise = hashCombine(root, k);
    let slot = atomicAdd(&counter.cursor, 1u) % uni.capacity;

    var particle : Particle;
    particle.position = origin;
    particle.velocity = vec3<f32>(0.0, 0.0, 0.0);
    particle.synapse = index;
    particle.u = 0.0;
    particle.distance = 0.0;
    particle.life = uni.life;
    particle.speed = uni.speed * (1.0 + speedJitter * (randomUnit(noise) * 2.0 - 1.0));
    particle.size = uni.size * (1.0 + sizeJitter * (randomUnit(noise ^ 0x85ebca6bu) * 2.0 - 1.0));
    particles[slot] = particle;
  }
}
`;

/** Bind group layout shared by PARTICLE_UPDATE_WGSL and PARTICLE_EMIT_WGSL. */
export const PARTICLE_COMPUTE_BINDINGS: readonly ShaderBinding[] = [
  { binding: 0, name: 'uni', type: 'uniform' },
  { binding: 1, name: 'particles', type: 'storage' },
  { binding: 2, name: 'counter', type: 'storage' },
  { binding: 3, name: 'synapses', type: 'read-only-storage' },
  { binding: 4, name: 'positions', type: 'read-only-storage' },
  { binding: 5, name: 'outputs', type: 'read-only-storage' },
];
