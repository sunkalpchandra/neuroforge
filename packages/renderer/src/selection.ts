import * as THREE from 'three';
import { COLORS, NEURON_FLAG } from '@neuroforge/shared';
import type { SimulationBuffers } from '@neuroforge/shared';
import { linearColor } from './palette';
import { somaRadiusForCode } from './morphology';
import { growthCapacity, instancedAttribute } from './instancing';

/**
 * Selection and hover halos.
 *
 * One instanced billboard per marked neuron, expanded in view space so the ring
 * keeps its shape whatever angle the camera sits at. Drawn additively with the
 * depth test off, because a selection ring that a dendrite can hide is not doing
 * its job.
 */

const HALO_VERTEX = /* glsl */ `
attribute vec3 instanceOffset;
attribute float instanceRadius;
attribute float instanceState;
attribute float instancePhase;

uniform float uSpread;

varying vec2 vLocal;
varying float vState;
varying float vPhase;

void main() {
  vec4 viewPosition = modelViewMatrix * vec4(instanceOffset, 1.0);
  float radius = instanceRadius * uSpread;
  viewPosition.xy += position.xy * radius * 2.0;

  vLocal = position.xy * 2.0;
  vState = instanceState;
  vPhase = instancePhase;

  gl_Position = projectionMatrix * viewPosition;
}
`;

const HALO_FRAGMENT = /* glsl */ `
uniform vec3 uSelectedColor;
uniform vec3 uHoveredColor;
uniform float uTime;
uniform float uRingWidth;
uniform float uIntensity;

varying vec2 vLocal;
varying float vState;
varying float vPhase;

const float PI = 3.141592653589793;

void main() {
  float distance = length(vLocal);
  if (distance > 1.0) {
    discard;
  }

  // Selected rings sit a touch wider and pulse; hovered ones stay put, so the
  // two states never read as the same mark at different brightnesses.
  float selected = step(0.5, vState);
  float pulse = 0.5 + 0.5 * sin(uTime * 2.6 + vPhase * PI * 2.0);
  float radius = mix(0.62, 0.74 + 0.04 * pulse, selected);
  float width = uRingWidth * mix(1.0, 1.25, selected);

  float ring = exp(-pow((distance - radius) / width, 2.0));
  // Rotating ticks give the ring a direction of travel and stop a dense
  // selection from reading as a field of static blobs.
  float angle = atan(vLocal.y, vLocal.x);
  float ticks = 0.72 + 0.28 * sin(angle * 16.0 - uTime * 1.7 + vPhase * 12.0);
  float halo = exp(-distance * distance * 3.2) * 0.22;

  vec3 color = mix(uHoveredColor, uSelectedColor, selected);
  float alpha = ring * ticks + halo;
  gl_FragColor = vec4(color * uIntensity * alpha, alpha);
}
`;

interface HaloPool {
  offset: THREE.InstancedBufferAttribute;
  radius: THREE.InstancedBufferAttribute;
  state: THREE.InstancedBufferAttribute;
  phase: THREE.InstancedBufferAttribute;
}

function createPool(capacity: number): HaloPool {
  return {
    offset: instancedAttribute(capacity, 3),
    radius: instancedAttribute(capacity, 1),
    state: instancedAttribute(capacity, 1),
    phase: instancedAttribute(capacity, 1),
  };
}

export class SelectionOverlay extends THREE.Group {
  #geometry: THREE.InstancedBufferGeometry;
  #material: THREE.ShaderMaterial;
  #mesh: THREE.Mesh;
  #pool: HaloPool;
  #capacity: number;
  #time = 0;

  constructor() {
    super();
    this.name = 'SelectionOverlay';

    const quad = new THREE.PlaneGeometry(1, 1);
    this.#geometry = new THREE.InstancedBufferGeometry();
    const position = quad.getAttribute('position');
    const index = quad.getIndex();
    this.#geometry.setAttribute('position', position);
    if (index) this.#geometry.setIndex(index);
    quad.deleteAttribute('position');
    quad.deleteAttribute('normal');
    quad.deleteAttribute('uv');
    quad.index = null;
    quad.dispose();

    this.#capacity = growthCapacity(1, 0);
    this.#pool = createPool(this.#capacity);
    this.#bind();
    this.#geometry.instanceCount = 0;

    this.#material = new THREE.ShaderMaterial({
      vertexShader: HALO_VERTEX,
      fragmentShader: HALO_FRAGMENT,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uSpread: { value: 2.6 },
        uSelectedColor: { value: linearColor(COLORS.accent) },
        uHoveredColor: { value: linearColor(COLORS.secondary) },
        uTime: { value: 0 },
        uRingWidth: { value: 0.09 },
        uIntensity: { value: 2.2 },
      },
    });

    this.#mesh = new THREE.Mesh(this.#geometry, this.#material);
    this.#mesh.frustumCulled = false;
    this.#mesh.renderOrder = 10;
    this.add(this.#mesh);
  }

  update(buffers: SimulationBuffers, dt: number): void {
    this.#time += dt;
    this.#material.uniforms.uTime.value = this.#time;

    const neurons = buffers.neurons;
    const count = neurons.count;
    const marked = NEURON_FLAG.SELECTED | NEURON_FLAG.HOVERED;

    let needed = 0;
    for (let i = 0; i < count; i += 1) {
      if ((neurons.flags[i] & marked) !== 0) needed += 1;
    }
    if (needed > this.#capacity) this.#reserve(needed);

    const offset = this.#pool.offset.array as Float32Array;
    const radius = this.#pool.radius.array as Float32Array;
    const state = this.#pool.state.array as Float32Array;
    const phase = this.#pool.phase.array as Float32Array;

    let written = 0;
    for (let i = 0; i < count && written < this.#capacity; i += 1) {
      const flags = neurons.flags[i];
      if ((flags & marked) === 0) continue;
      const p = i * 3;
      offset[written * 3] = neurons.position[p];
      offset[written * 3 + 1] = neurons.position[p + 1];
      offset[written * 3 + 2] = neurons.position[p + 2];
      radius[written] = somaRadiusForCode(neurons.archetype[i]) * neurons.scale[i];
      state[written] = (flags & NEURON_FLAG.SELECTED) !== 0 ? 1 : 0;
      phase[written] = (i % 32) / 32;
      written += 1;
    }

    this.#pool.offset.needsUpdate = true;
    this.#pool.radius.needsUpdate = true;
    this.#pool.state.needsUpdate = true;
    this.#pool.phase.needsUpdate = true;
    this.#geometry.instanceCount = written;
    this.#mesh.visible = written > 0;
  }

  dispose(): void {
    this.remove(this.#mesh);
    this.#geometry.dispose();
    this.#material.dispose();
  }

  #bind(): void {
    this.#geometry.setAttribute('instanceOffset', this.#pool.offset);
    this.#geometry.setAttribute('instanceRadius', this.#pool.radius);
    this.#geometry.setAttribute('instanceState', this.#pool.state);
    this.#geometry.setAttribute('instancePhase', this.#pool.phase);
  }

  #reserve(needed: number): void {
    const position = this.#geometry.getAttribute('position');
    const index = this.#geometry.getIndex();
    this.#geometry.deleteAttribute('position');
    this.#geometry.index = null;
    this.#geometry.dispose();

    this.#geometry = new THREE.InstancedBufferGeometry();
    this.#geometry.setAttribute('position', position);
    if (index) this.#geometry.setIndex(index);
    this.#capacity = growthCapacity(needed, this.#capacity);
    this.#pool = createPool(this.#capacity);
    this.#bind();
    this.#geometry.instanceCount = 0;
    this.#mesh.geometry = this.#geometry;
  }
}
