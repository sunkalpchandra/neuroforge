import * as THREE from 'three';
import { PARTICLE_FRAGMENT_GLSL, PARTICLE_VERTEX_GLSL } from '@neuroforge/shaders';
import { COLORS, DEFAULT_RENDER_SETTINGS, RECEPTOR_CODE } from '@neuroforge/shared';
import type { RenderSettings, SimulationBuffers } from '@neuroforge/shared';
import { hashSeed } from '@neuroforge/math';
import { RECEPTOR_LINEAR, linearColor, receptorOffset, viewportHeight } from './palette';
import { arcLength, axonSag, chordLength, controlPoint, pointAt, tangentAt } from './axon-spline';
import { growthCapacity } from './instancing';

/**
 * Additive impulse particles emitted on spikes and advected along axons.
 *
 * Storage is a fixed pool with a free list: emission pops an index, retirement
 * pushes it back, and nothing is ever allocated once the pool exists. The spike
 * ring buffer is consumed by cursor, so an event is never emitted twice and a
 * frame that lands after the ring has wrapped simply starts from the oldest
 * event still present instead of replaying history.
 */

const DEFAULT_CAPACITY = 4096;

/** Particles per axon per spike, before the density setting scales it. */
const PER_AXON = 2;

/** Outgoing axons lit by one spike; a hub neuron does not get to flood the pool. */
const MAX_AXONS_PER_SPIKE = 6;

/** World units per second an impulse travels. */
const TRAVEL_SPEED = 26;

/** Length of the burst emitted by a neuron that has no outgoing synapses. */
const SPARK_LENGTH = 3.5;

/** Fraction of the pool a single call may consume. */
const EMIT_BUDGET = 0.25;

function createMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: PARTICLE_VERTEX_GLSL,
    fragmentShader: PARTICLE_FRAGMENT_GLSL,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uSize: { value: 26 },
      uScale: { value: 540 },
      uStretch: { value: 0.06 },
      uFadeIn: { value: 0.12 },
      uCoreColor: { value: linearColor('#FFFFFF').multiplyScalar(1.4) },
      uFogColor: { value: linearColor(COLORS.background) },
      uSoftness: { value: 3.4 },
      uIntensity: { value: 2.6 },
      uFogDensity: { value: DEFAULT_RENDER_SETTINGS.fogDensity },
    },
  });
}

export class SpikeParticles extends THREE.Points {
  readonly #capacity: number;

  readonly #position: THREE.BufferAttribute;
  readonly #life: THREE.BufferAttribute;
  readonly #size: THREE.BufferAttribute;
  readonly #color: THREE.BufferAttribute;
  readonly #velocity: THREE.BufferAttribute;

  readonly #start: Float32Array;
  readonly #end: Float32Array;
  readonly #control: Float32Array;
  readonly #travel: Float32Array;
  readonly #rate: Float32Array;
  readonly #free: Int32Array;

  #freeCount: number;
  #high = 0;
  #consumed = 0;
  #density = DEFAULT_RENDER_SETTINGS.particleDensity;
  #viewportHeight = 0;

  /** Prefix offsets into `#outList`, one per neuron slot plus a terminator. */
  #outStart = new Uint32Array(1);
  #outList = new Uint32Array(0);
  /** Scatter cursors, parallel to `#outStart`; a field so a rebuild allocates nothing. */
  #outCursor = new Uint32Array(1);
  #adjacencyNeurons = -1;
  #adjacencySynapses = -1;
  #adjacencySource: Uint32Array | null = null;

  constructor(capacity = DEFAULT_CAPACITY) {
    const size = Math.max(64, Math.floor(capacity));
    const geometry = new THREE.BufferGeometry();
    const position = new THREE.BufferAttribute(new Float32Array(size * 3), 3);
    const life = new THREE.BufferAttribute(new Float32Array(size), 1);
    const sizeAttribute = new THREE.BufferAttribute(new Float32Array(size), 1);
    const color = new THREE.BufferAttribute(new Float32Array(size * 3), 3);
    const velocity = new THREE.BufferAttribute(new Float32Array(size * 3), 3);
    position.setUsage(THREE.DynamicDrawUsage);
    life.setUsage(THREE.DynamicDrawUsage);
    sizeAttribute.setUsage(THREE.DynamicDrawUsage);
    color.setUsage(THREE.DynamicDrawUsage);
    velocity.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', position);
    geometry.setAttribute('aLife', life);
    geometry.setAttribute('aSize', sizeAttribute);
    geometry.setAttribute('aColor', color);
    geometry.setAttribute('aVelocity', velocity);
    geometry.setDrawRange(0, 0);

    super(geometry, createMaterial());
    this.name = 'SpikeParticles';
    this.frustumCulled = false;
    this.renderOrder = 2;

    this.#capacity = size;
    this.#position = position;
    this.#life = life;
    this.#size = sizeAttribute;
    this.#color = color;
    this.#velocity = velocity;

    this.#start = new Float32Array(size * 3);
    this.#end = new Float32Array(size * 3);
    this.#control = new Float32Array(size * 3);
    this.#travel = new Float32Array(size);
    this.#rate = new Float32Array(size);
    this.#free = new Int32Array(size);
    for (let i = 0; i < size; i += 1) this.#free[i] = size - 1 - i;
    this.#freeCount = size;
  }

  get liveCount(): number {
    return this.#capacity - this.#freeCount;
  }

  /** Consume every spike written since the last call and light its axons. */
  emitFromSpikes(buffers: SimulationBuffers): void {
    const log = buffers.spikes;
    const head = log.head;
    if (head === this.#consumed) return;
    // A reset rewinds the cursor; replaying the old tail would fire every spike
    // in the ring a second time.
    if (head < this.#consumed) {
      this.#consumed = head;
      return;
    }

    this.#ensureAdjacency(buffers);

    const oldest = Math.max(this.#consumed, head - log.capacity);
    let budget = Math.floor(this.#capacity * EMIT_BUDGET);
    const perAxon = Math.max(1, Math.round(PER_AXON * this.#density));
    const neuronCount = buffers.neurons.count;

    for (let event = oldest; event < head && budget > 0; event += 1) {
      const slot = log.neuron[event % log.capacity];
      if (slot >= neuronCount) continue;
      const first = this.#outStart[slot];
      const degree = this.#outStart[slot + 1] - first;

      if (degree === 0) {
        budget -= this.#emitSpark(buffers, slot, perAxon, budget);
        continue;
      }
      const take = Math.min(degree, MAX_AXONS_PER_SPIKE);
      const stride = Math.max(1, Math.floor(degree / take));
      for (let k = 0; k < take && budget > 0; k += 1) {
        const synapse = this.#outList[first + ((k * stride) % degree)];
        budget -= this.#emitAlongAxon(buffers, synapse, perAxon, budget);
      }
    }

    this.#consumed = head;
    this.#size.needsUpdate = true;
    this.#color.needsUpdate = true;
  }

  update(dt: number, settings: RenderSettings): void {
    this.#density = settings.particleDensity;
    this.visible = settings.showParticles;

    const material = this.material as THREE.ShaderMaterial;
    material.uniforms.uFogDensity.value = settings.fogDensity;
    const height = viewportHeight();
    if (height !== this.#viewportHeight) {
      this.#viewportHeight = height;
      material.uniforms.uScale.value = height * 0.5;
    }

    const position = this.#position.array as Float32Array;
    const life = this.#life.array as Float32Array;
    const velocity = this.#velocity.array as Float32Array;
    const travel = this.#travel;
    const rate = this.#rate;
    const start = this.#start;
    const end = this.#end;
    const control = this.#control;
    let highest = 0;

    for (let i = 0; i < this.#high; i += 1) {
      if (life[i] <= 0) continue;
      const t = travel[i] + rate[i] * dt;
      if (t >= 1) {
        life[i] = 0;
        this.#free[this.#freeCount] = i;
        this.#freeCount += 1;
        continue;
      }
      travel[i] = t;
      life[i] = 1 - t;
      const p = i * 3;
      pointAt(
        start[p],
        start[p + 1],
        start[p + 2],
        end[p],
        end[p + 1],
        end[p + 2],
        control,
        p,
        t,
        position,
        p,
      );
      tangentAt(
        start[p],
        start[p + 1],
        start[p + 2],
        end[p],
        end[p + 1],
        end[p + 2],
        control,
        p,
        t,
        velocity,
        p,
      );
      // The tangent is per unit of parameter; scaling by the rate turns it into
      // world units per second, which is what the streak length wants.
      velocity[p] *= rate[i];
      velocity[p + 1] *= rate[i];
      velocity[p + 2] *= rate[i];
      highest = i + 1;
    }

    this.#high = highest;
    this.geometry.setDrawRange(0, highest);
    this.#position.needsUpdate = true;
    this.#life.needsUpdate = true;
    this.#velocity.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    const material = this.material as THREE.ShaderMaterial;
    material.dispose();
  }

  /** Returns how many particles were taken from the pool. */
  #emitAlongAxon(
    buffers: SimulationBuffers,
    synapse: number,
    wanted: number,
    budget: number,
  ): number {
    const synapses = buffers.synapses;
    const neurons = buffers.neurons;
    const pre = synapses.pre[synapse];
    const post = synapses.post[synapse];
    if (pre >= neurons.count || post >= neurons.count) return 0;

    const a = pre * 3;
    const b = post * 3;
    const ax = neurons.position[a];
    const ay = neurons.position[a + 1];
    const az = neurons.position[a + 2];
    const bx = neurons.position[b];
    const by = neurons.position[b + 1];
    const bz = neurons.position[b + 2];
    const sag = axonSag(synapses.arc[synapse], chordLength(ax, ay, az, bx, by, bz));
    const length = arcLength(ax, ay, az, bx, by, bz, sag);
    const source = receptorOffset(synapses.receptor[synapse]);

    return this.#spawn(
      ax,
      ay,
      az,
      bx,
      by,
      bz,
      sag,
      length,
      RECEPTOR_LINEAR[source],
      RECEPTOR_LINEAR[source + 1],
      RECEPTOR_LINEAR[source + 2],
      synapse,
      Math.min(wanted, budget),
    );
  }

  /** A neuron with no outgoing axon still shows that it fired. */
  #emitSpark(
    buffers: SimulationBuffers,
    slot: number,
    wanted: number,
    budget: number,
  ): number {
    const neurons = buffers.neurons;
    const p = slot * 3;
    const ax = neurons.position[p];
    const ay = neurons.position[p + 1];
    const az = neurons.position[p + 2];
    const inhibitory = neurons.polarity[slot] !== 0;
    const source = receptorOffset(inhibitory ? RECEPTOR_CODE.gabaa : RECEPTOR_CODE.ampa);
    let taken = 0;
    const count = Math.min(wanted, budget);

    for (let i = 0; i < count; i += 1) {
      const noise = hashSeed(slot, i, buffers.step);
      const theta = ((noise >>> 8) / 0x1000000) * Math.PI * 2;
      const height = ((noise & 0xff) / 0xff) * 2 - 1;
      const radius = Math.sqrt(Math.max(0, 1 - height * height));
      const dx = Math.cos(theta) * radius;
      const dz = Math.sin(theta) * radius;
      taken += this.#spawn(
        ax,
        ay,
        az,
        ax + dx * SPARK_LENGTH,
        ay + height * SPARK_LENGTH,
        az + dz * SPARK_LENGTH,
        0,
        SPARK_LENGTH,
        RECEPTOR_LINEAR[source],
        RECEPTOR_LINEAR[source + 1],
        RECEPTOR_LINEAR[source + 2],
        slot,
        1,
      );
    }
    return taken;
  }

  #spawn(
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    sag: number,
    length: number,
    red: number,
    green: number,
    blue: number,
    salt: number,
    count: number,
  ): number {
    if (count <= 0) return 0;
    const rate = TRAVEL_SPEED / Math.max(length, 0.25);
    const life = this.#life.array as Float32Array;
    const sizes = this.#size.array as Float32Array;
    const colors = this.#color.array as Float32Array;
    const position = this.#position.array as Float32Array;
    let taken = 0;

    for (let i = 0; i < count; i += 1) {
      if (this.#freeCount === 0) break;
      this.#freeCount -= 1;
      const index = this.#free[this.#freeCount];
      const p = index * 3;

      this.#start[p] = ax;
      this.#start[p + 1] = ay;
      this.#start[p + 2] = az;
      this.#end[p] = bx;
      this.#end[p + 1] = by;
      this.#end[p + 2] = bz;
      controlPoint(ax, ay, az, bx, by, bz, sag, this.#control, p);

      // Successive particles of one burst start a little way apart so the volley
      // reads as a packet travelling rather than as a single brighter dot.
      const offset = (i * 0.06) % 0.5;
      this.#travel[index] = offset;
      this.#rate[index] = rate;
      life[index] = 1 - offset;
      const noise = (hashSeed(salt, index) >>> 8) / 0x1000000;
      sizes[index] = 0.65 + noise * 0.7;
      colors[p] = red;
      colors[p + 1] = green;
      colors[p + 2] = blue;
      position[p] = ax;
      position[p + 1] = ay;
      position[p + 2] = az;

      if (index + 1 > this.#high) this.#high = index + 1;
      taken += 1;
    }
    return taken;
  }

  #ensureAdjacency(buffers: SimulationBuffers): void {
    const neurons = buffers.neurons;
    const synapses = buffers.synapses;
    if (
      this.#adjacencyNeurons === neurons.count &&
      this.#adjacencySynapses === synapses.count &&
      this.#adjacencySource === synapses.pre
    ) {
      return;
    }

    const n = neurons.count;
    const m = synapses.count;
    // Grown in blocks rather than to the exact size, so a network being built up
    // one neuron at a time does not reallocate on every frame.
    if (this.#outStart.length < n + 2) {
      const cells = growthCapacity(n + 2, this.#outStart.length);
      this.#outStart = new Uint32Array(cells);
      this.#outCursor = new Uint32Array(cells);
    }
    this.#outStart.fill(0);

    let total = 0;
    for (let i = 0; i < m; i += 1) {
      const pre = synapses.pre[i];
      if (pre >= n || synapses.enabled[i] === 0) continue;
      this.#outStart[pre + 1] += 1;
      total += 1;
    }
    for (let i = 0; i < n; i += 1) {
      this.#outStart[i + 1] += this.#outStart[i];
    }
    if (this.#outList.length < total) {
      this.#outList = new Uint32Array(growthCapacity(total, this.#outList.length));
    }

    const cursor = this.#outCursor;
    cursor.set(this.#outStart.subarray(0, n + 1));
    for (let i = 0; i < m; i += 1) {
      const pre = synapses.pre[i];
      if (pre >= n || synapses.enabled[i] === 0) continue;
      this.#outList[cursor[pre]] = i;
      cursor[pre] += 1;
    }

    this.#adjacencyNeurons = n;
    this.#adjacencySynapses = m;
    this.#adjacencySource = synapses.pre;
  }
}
