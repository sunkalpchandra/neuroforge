import * as THREE from 'three';
import { AXON_FRAGMENT_GLSL, AXON_VERTEX_GLSL } from '@neuroforge/shaders';
import { COLORS, DEFAULT_RENDER_SETTINGS } from '@neuroforge/shared';
import type { RenderSettings, SimulationBuffers } from '@neuroforge/shared';
import { hashSeed } from '@neuroforge/math';
import { RECEPTOR_LINEAR, linearColor, receptorOffset, viewportHeight, viewportWidth } from './palette';
import { axonSag, chordLength } from './axon-spline';
import { growthCapacity, instancedAttribute } from './instancing';

/**
 * Every synapse in the scene as one instanced ribbon.
 *
 * The curve is evaluated in the vertex shader from the per-instance endpoints
 * and sag, so the mesh never changes shape: adding a synapse writes into buffers
 * that already exist and bumps `instanceCount`. Nothing is rebuilt, nothing is
 * re-uploaded except the instance rows themselves.
 */

/** Samples along the spline. The strip is one quad per sample interval. */
const RIBBON_SEGMENTS = 28;

/** Spare instance rows kept so that ordinary editing never reallocates. */
const HEADROOM = 1.35;

/** Impulse travel in spline parameter per second. */
const PULSE_SPEED = 0.85;

/** Weight above which an axon renders at full thickness (nS). */
const WEIGHT_REFERENCE = 4;

interface RibbonPool {
  start: THREE.InstancedBufferAttribute;
  end: THREE.InstancedBufferAttribute;
  color: THREE.InstancedBufferAttribute;
  sag: THREE.InstancedBufferAttribute;
  width: THREE.InstancedBufferAttribute;
  activity: THREE.InstancedBufferAttribute;
  pulse: THREE.InstancedBufferAttribute;
}

function buildStrip(): { position: Float32Array; index: Uint16Array } {
  const points = RIBBON_SEGMENTS + 1;
  const position = new Float32Array(points * 2 * 3);
  const index = new Uint16Array(RIBBON_SEGMENTS * 6);
  for (let i = 0; i < points; i += 1) {
    const t = i / RIBBON_SEGMENTS;
    position[i * 6] = t;
    position[i * 6 + 1] = -1;
    position[i * 6 + 2] = 0;
    position[i * 6 + 3] = t;
    position[i * 6 + 4] = 1;
    position[i * 6 + 5] = 0;
  }
  for (let i = 0; i < RIBBON_SEGMENTS; i += 1) {
    const a = i * 2;
    index[i * 6] = a;
    index[i * 6 + 1] = a + 2;
    index[i * 6 + 2] = a + 1;
    index[i * 6 + 3] = a + 1;
    index[i * 6 + 4] = a + 2;
    index[i * 6 + 5] = a + 3;
  }
  return { position, index };
}

function createPool(capacity: number): RibbonPool {
  return {
    start: instancedAttribute(capacity, 3),
    end: instancedAttribute(capacity, 3),
    color: instancedAttribute(capacity, 3),
    sag: instancedAttribute(capacity, 1),
    width: instancedAttribute(capacity, 1),
    activity: instancedAttribute(capacity, 1),
    pulse: instancedAttribute(capacity, 1),
  };
}

export class AxonField extends THREE.Group {
  #geometry: THREE.InstancedBufferGeometry;
  #material: THREE.ShaderMaterial;
  #mesh: THREE.Mesh;
  #pool: RibbonPool;
  #capacity = 0;
  #count = -1;
  #phase = 0;
  #resolutionWidth = 0;
  #resolutionHeight = 0;

  constructor() {
    super();
    this.name = 'AxonField';

    const strip = buildStrip();
    this.#geometry = new THREE.InstancedBufferGeometry();
    this.#geometry.setAttribute('position', new THREE.BufferAttribute(strip.position, 3));
    this.#geometry.setIndex(new THREE.BufferAttribute(strip.index, 1));
    this.#geometry.instanceCount = 0;

    this.#capacity = growthCapacity(1, 0);
    this.#pool = createPool(this.#capacity);
    this.#bind();

    this.#material = new THREE.ShaderMaterial({
      vertexShader: AXON_VERTEX_GLSL,
      fragmentShader: AXON_FRAGMENT_GLSL,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      uniforms: {
        uWidth: { value: 0.15 },
        uTaper: { value: 0.55 },
        uMinPixelWidth: { value: 1.4 },
        uResolution: { value: new THREE.Vector2(1920, 1080) },
        uPulseColor: { value: linearColor('#F2D4FF').multiplyScalar(1.6) },
        uFogColor: { value: linearColor(COLORS.background) },
        uPulse: { value: 0 },
        uPulseWidth: { value: 0.05 },
        uPulseTrail: { value: 0.11 },
        uPulseIntensity: { value: 3.4 },
        uBaseIntensity: { value: 0.28 },
        uActivityIntensity: { value: 1.6 },
        uCoreSoftness: { value: 3.1 },
        uOpacity: { value: 0.9 },
        uFogDensity: { value: DEFAULT_RENDER_SETTINGS.fogDensity },
      },
    });

    this.#mesh = new THREE.Mesh(this.#geometry, this.#material);
    // Endpoints live in instance attributes, so the strip's own bounds describe
    // nothing; the scene graph above decides visibility.
    this.#mesh.frustumCulled = false;
    this.#mesh.renderOrder = -1;
    this.add(this.#mesh);
  }

  rebuild(buffers: SimulationBuffers): void {
    const count = buffers.synapses.count;
    this.#reserve(count);
    this.#writeStatic(buffers, count);
    this.#writeDynamic(buffers, count);
    this.#count = count;
    this.#geometry.instanceCount = count;
  }

  update(buffers: SimulationBuffers, dt: number, settings: RenderSettings): void {
    const count = buffers.synapses.count;
    if (count > this.#capacity) {
      this.#reserve(count);
      this.#writeStatic(buffers, count);
    } else if (count !== this.#count) {
      this.#writeStatic(buffers, count);
    }
    this.#writeDynamic(buffers, count);
    this.#count = count;
    this.#geometry.instanceCount = count;

    this.#phase = (this.#phase + dt * PULSE_SPEED) % 1;
    const uniforms = this.#material.uniforms;
    uniforms.uPulse.value = this.#phase;
    uniforms.uFogDensity.value = settings.fogDensity;

    const width = viewportWidth();
    const height = viewportHeight();
    if (width !== this.#resolutionWidth || height !== this.#resolutionHeight) {
      this.#resolutionWidth = width;
      this.#resolutionHeight = height;
      const resolution = uniforms.uResolution.value as THREE.Vector2;
      resolution.set(width, height);
    }

    this.#mesh.visible = settings.showAxons;
  }

  dispose(): void {
    this.remove(this.#mesh);
    this.#geometry.dispose();
    this.#material.dispose();
  }

  #bind(): void {
    this.#geometry.setAttribute('instanceStart', this.#pool.start);
    this.#geometry.setAttribute('instanceEnd', this.#pool.end);
    this.#geometry.setAttribute('instanceColor', this.#pool.color);
    this.#geometry.setAttribute('instanceSag', this.#pool.sag);
    this.#geometry.setAttribute('instanceWidth', this.#pool.width);
    this.#geometry.setAttribute('instanceActivity', this.#pool.activity);
    this.#geometry.setAttribute('instancePulse', this.#pool.pulse);
  }

  #reserve(count: number): void {
    const required = Math.max(1, Math.ceil(count * HEADROOM));
    if (required <= this.#capacity) return;
    // The old instance buffers belong to the geometry, so they are handed back
    // by disposing it; the strip attributes are rebuilt from the same arrays.
    const position = this.#geometry.getAttribute('position');
    const index = this.#geometry.getIndex();
    this.#geometry.deleteAttribute('position');
    this.#geometry.index = null;
    this.#geometry.dispose();
    this.#geometry = new THREE.InstancedBufferGeometry();
    this.#geometry.setAttribute('position', position);
    if (index) this.#geometry.setIndex(index);
    this.#capacity = growthCapacity(required, 0);
    this.#pool = createPool(this.#capacity);
    this.#bind();
    this.#geometry.instanceCount = 0;
    this.#mesh.geometry = this.#geometry;
  }

  /** Receptor colour, thickness and impulse phase: constant per synapse. */
  #writeStatic(buffers: SimulationBuffers, count: number): void {
    const synapses = buffers.synapses;
    const color = this.#pool.color.array as Float32Array;
    const width = this.#pool.width.array as Float32Array;
    const pulse = this.#pool.pulse.array as Float32Array;

    for (let i = 0; i < count; i += 1) {
      const source = receptorOffset(synapses.receptor[i]);
      color[i * 3] = RECEPTOR_LINEAR[source];
      color[i * 3 + 1] = RECEPTOR_LINEAR[source + 1];
      color[i * 3 + 2] = RECEPTOR_LINEAR[source + 2];
      const weight = Math.min(Math.abs(synapses.weight[i]), WEIGHT_REFERENCE) / WEIGHT_REFERENCE;
      width[i] = 0.55 + weight * 0.9;
      // A per-axon phase offset; without it every impulse in a projection fires
      // in lockstep and the bundle strobes.
      pulse[i] = (hashSeed(i) >>> 8) / 0x1000000;
    }

    this.#pool.color.needsUpdate = true;
    this.#pool.width.needsUpdate = true;
    this.#pool.pulse.needsUpdate = true;
  }

  /** Endpoints, sag and travel envelope: these follow the live simulation. */
  #writeDynamic(buffers: SimulationBuffers, count: number): void {
    const synapses = buffers.synapses;
    const neurons = buffers.neurons;
    const position = neurons.position;
    const limit = neurons.count;
    const start = this.#pool.start.array as Float32Array;
    const end = this.#pool.end.array as Float32Array;
    const sag = this.#pool.sag.array as Float32Array;
    const activity = this.#pool.activity.array as Float32Array;

    for (let i = 0; i < count; i += 1) {
      const pre = synapses.pre[i];
      const post = synapses.post[i];
      if (pre >= limit || post >= limit) {
        start[i * 3] = 0;
        start[i * 3 + 1] = 0;
        start[i * 3 + 2] = 0;
        end[i * 3] = 0;
        end[i * 3 + 1] = 0;
        end[i * 3 + 2] = 0;
        sag[i] = 0;
        activity[i] = 0;
        continue;
      }
      const a = pre * 3;
      const b = post * 3;
      const ax = position[a];
      const ay = position[a + 1];
      const az = position[a + 2];
      const bx = position[b];
      const by = position[b + 1];
      const bz = position[b + 2];
      start[i * 3] = ax;
      start[i * 3 + 1] = ay;
      start[i * 3 + 2] = az;
      end[i * 3] = bx;
      end[i * 3 + 1] = by;
      end[i * 3 + 2] = bz;
      sag[i] = axonSag(synapses.arc[i], chordLength(ax, ay, az, bx, by, bz));
      activity[i] = synapses.enabled[i] === 0 ? 0 : synapses.activity[i];
    }

    this.#pool.start.needsUpdate = true;
    this.#pool.end.needsUpdate = true;
    this.#pool.sag.needsUpdate = true;
    this.#pool.activity.needsUpdate = true;
  }
}
