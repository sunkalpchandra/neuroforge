import * as THREE from 'three';
import { RECEPTOR_FROM_CODE, RECEPTOR_COLORS, srgbToLinear } from '@neuroforge/shared';
import type { RenderSettings, SimulationBuffers } from '@neuroforge/shared';
import { sampleArc } from '@neuroforge/math';

/**
 * Point markers at synaptic contact sites.
 *
 * Connectome viewers draw synapses as discrete annotations rather than leaving
 * them implied by a connecting line, because where a contact sits along a
 * process is itself the finding — a cell targeted on its soma and one targeted
 * on a distal dendrite are wired differently even though the wiring diagram is
 * identical.
 *
 * Positions are taken from the same arc sampler the axon ribbon shader was
 * verified against, so a marker sits exactly on its ribbon rather than floating
 * near it.
 */

/**
 * Position along the arc, 0 at the presynaptic cell and 1 at the postsynaptic
 * one. Just short of the end so the marker reads as a contact onto the target
 * rather than disappearing inside its soma.
 */
const CONTACT_T = 0.88;

/** Arc samples used to locate the contact point. */
const ARC_SAMPLES = 16;

const CONTACT_INDEX = Math.min(ARC_SAMPLES - 1, Math.round(CONTACT_T * (ARC_SAMPLES - 1)));

function receptorTable(): Float32Array {
  const table = new Float32Array(RECEPTOR_FROM_CODE.length * 3);
  for (let i = 0; i < RECEPTOR_FROM_CODE.length; i += 1) {
    const [r, g, b] = srgbToLinear(RECEPTOR_COLORS[RECEPTOR_FROM_CODE[i]]);
    table[i * 3] = r;
    table[i * 3 + 1] = g;
    table[i * 3 + 2] = b;
  }
  return table;
}

const RECEPTOR_LINEAR = receptorTable();

const VERTEX = /* glsl */ `
attribute vec3 markerColor;
attribute float markerSize;
attribute float markerActivity;

uniform float uPixelScale;
uniform float uSizeScale;

varying vec3 vColor;
varying float vActivity;

void main() {
  vColor = markerColor;
  vActivity = markerActivity;
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * viewPosition;
  // Perspective attenuation, with a floor so a distant contact stays visible as
  // a point rather than vanishing — the count of contacts is the signal.
  float dist = max(-viewPosition.z, 0.001);
  gl_PointSize = clamp(uPixelScale * uSizeScale * markerSize / dist, 1.0, 14.0);
}
`;

const FRAGMENT = /* glsl */ `
precision highp float;

varying vec3 vColor;
varying float vActivity;

void main() {
  // Round sprite with a soft edge; the square would read as a voxel, which is
  // exactly the impression this project is trying not to give.
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) discard;
  float edge = smoothstep(1.0, 0.35, r2);
  float core = smoothstep(0.6, 0.0, r2);
  vec3 color = vColor * (0.35 + 0.3 * core) + vColor * vActivity * 2.2;
  // Resting contacts sit well back; a transmitting one rises out of the field.
  gl_FragColor = vec4(color, edge * (0.38 + 0.62 * vActivity));
}
`;

export class SynapseMarkers extends THREE.Points {
  #positions: Float32Array;
  #colors: Float32Array;
  #sizes: Float32Array;
  #activity: Float32Array;
  #capacity: number;
  #count = 0;
  /** Reused arc buffer; locating a contact must not allocate per synapse. */
  #arc = new Float32Array(ARC_SAMPLES * 3);

  constructor(capacity = 4096) {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: {
        uPixelScale: { value: 300 },
        uSizeScale: { value: 1 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });
    super(geometry, material);

    this.#capacity = Math.max(16, capacity);
    this.#positions = new Float32Array(this.#capacity * 3);
    this.#colors = new Float32Array(this.#capacity * 3);
    this.#sizes = new Float32Array(this.#capacity);
    this.#activity = new Float32Array(this.#capacity);

    geometry.setAttribute('position', new THREE.BufferAttribute(this.#positions, 3));
    geometry.setAttribute('markerColor', new THREE.BufferAttribute(this.#colors, 3));
    geometry.setAttribute('markerSize', new THREE.BufferAttribute(this.#sizes, 1));
    geometry.setAttribute('markerActivity', new THREE.BufferAttribute(this.#activity, 1));
    geometry.setDrawRange(0, 0);
    this.frustumCulled = false;
    this.renderOrder = 3;
  }

  #grow(required: number): void {
    if (required <= this.#capacity) return;
    let capacity = this.#capacity;
    while (capacity < required) capacity *= 2;
    this.#capacity = capacity;
    this.#positions = new Float32Array(capacity * 3);
    this.#colors = new Float32Array(capacity * 3);
    this.#sizes = new Float32Array(capacity);
    this.#activity = new Float32Array(capacity);

    const geometry = this.geometry;
    geometry.setAttribute('position', new THREE.BufferAttribute(this.#positions, 3));
    geometry.setAttribute('markerColor', new THREE.BufferAttribute(this.#colors, 3));
    geometry.setAttribute('markerSize', new THREE.BufferAttribute(this.#sizes, 1));
    geometry.setAttribute('markerActivity', new THREE.BufferAttribute(this.#activity, 1));
  }

  /** Recompute contact positions. Call after a structural edit or a layout move. */
  rebuild(buffers: SimulationBuffers): void {
    const { neurons, synapses } = buffers;
    const total = synapses.count;
    this.#grow(Math.max(1, total));

    const position = neurons.position;
    const neuronCount = neurons.count;
    let kept = 0;

    for (let s = 0; s < total; s += 1) {
      if (synapses.enabled[s] === 0) continue;
      const pre = synapses.pre[s];
      const post = synapses.post[s];
      if (pre >= neuronCount || post >= neuronCount) continue;

      const a = pre * 3;
      const b = post * 3;
      sampleArc(
        position[a],
        position[a + 1],
        position[a + 2],
        position[b],
        position[b + 1],
        position[b + 2],
        synapses.arc[s],
        ARC_SAMPLES,
        this.#arc,
        0,
      );

      const p = CONTACT_INDEX * 3;
      const o = kept * 3;
      this.#positions[o] = this.#arc[p];
      this.#positions[o + 1] = this.#arc[p + 1];
      this.#positions[o + 2] = this.#arc[p + 2];

      const code = synapses.receptor[s];
      const c = (code >= 0 && code < RECEPTOR_FROM_CODE.length ? code : 0) * 3;
      this.#colors[o] = RECEPTOR_LINEAR[c];
      this.#colors[o + 1] = RECEPTOR_LINEAR[c + 1];
      this.#colors[o + 2] = RECEPTOR_LINEAR[c + 2];

      // Sub-linear in weight: a synapse ten times stronger is not ten times
      // wider on screen, or one strong contact hides its neighbours entirely.
      this.#sizes[kept] = 0.9 + Math.sqrt(Math.abs(synapses.weight[s])) * 0.7;
      kept += 1;
    }

    this.#count = kept;
    this.geometry.setDrawRange(0, kept);
    this.geometry.getAttribute('position').needsUpdate = true;
    this.geometry.getAttribute('markerColor').needsUpdate = true;
    this.geometry.getAttribute('markerSize').needsUpdate = true;
    this.geometry.computeBoundingSphere();
  }

  /** Per-frame refresh of the activity glow only. Allocates nothing. */
  update(buffers: SimulationBuffers, settings: RenderSettings): void {
    const material = this.material as THREE.ShaderMaterial;
    material.uniforms.uSizeScale.value = settings.neuronScale;

    const { synapses, neurons } = buffers;
    const total = synapses.count;
    const neuronCount = neurons.count;
    let kept = 0;
    for (let s = 0; s < total && kept < this.#count; s += 1) {
      if (synapses.enabled[s] === 0) continue;
      if (synapses.pre[s] >= neuronCount || synapses.post[s] >= neuronCount) continue;
      this.#activity[kept] = synapses.activity[s];
      kept += 1;
    }
    this.geometry.getAttribute('markerActivity').needsUpdate = true;
  }

  /** Contacts currently drawn. Named to avoid shadowing Points.count. */
  get markerCount(): number {
    return this.#count;
  }

  dispose(): void {
    this.geometry.dispose();
    (this.material as THREE.ShaderMaterial).dispose();
  }
}
