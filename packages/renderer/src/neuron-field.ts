import * as THREE from 'three';
import { NEURON_FRAGMENT_GLSL, NEURON_VERTEX_GLSL } from '@neuroforge/shaders';
import {
  COLORS,
  DEFAULT_RENDER_SETTINGS,
  NEURON_FLAG,
  POLARITY_COLORS,
  hslToRgb,
  identityColor,
  srgbToLinear,
} from '@neuroforge/shared';
import type { ColorMode, RenderSettings, SimulationBuffers } from '@neuroforge/shared';
import { SpatialHash } from '@neuroforge/math';
import { linearColor } from './palette';
import type { GlyphLibrary } from './glyph-library';
import {
  VARIANTS_PER_ARCHETYPE,
  somaRadiusForCode,
  variantKey,
  variantOf,
} from './morphology';
import { growthCapacity, instancedAttribute, releaseGeometry, shareGeometry } from './instancing';

/**
 * Instanced renderer for every neuron in the scene.
 *
 * Geometry is never rebuilt for a state change. Each (archetype, seed bucket)
 * pair owns one instance pool whose six per-instance buffers are shared by the
 * soma, dendrite and axon meshes, so a frame update is a single linear pass that
 * writes floats into arrays that already exist and flips six version counters.
 */

const PART_COUNT = 3;
const PART_SOMA = 0;
const PART_DENDRITES = 1;
const PART_AXON = 2;

/** Decay constant of the renderer-side spike envelope, in seconds. */
const FLASH_TAU = 0.085;

/** How often the pick acceleration structure is refreshed, in seconds. */
const HASH_INTERVAL = 0.25;

/** Cell size relative to the largest soma; wide enough that a walk is short. */
const HASH_CELL_FACTOR = 6;


/**
 * Neurotransmitter tints, keyed by polarity. Excitatory cells in an insect brain
 * are overwhelmingly cholinergic and inhibitory ones GABAergic, so polarity is
 * the honest proxy here — the document does not carry a transmitter field.
 */
const TRANSMITTER_TINT: readonly (readonly [number, number, number])[] = [
  srgbToLinear('#F5A524'),
  srgbToLinear('#5B8DEF'),
];

const POLARITY_TINT: readonly (readonly [number, number, number])[] = [
  srgbToLinear(POLARITY_COLORS.excitatory),
  srgbToLinear(POLARITY_COLORS.inhibitory),
];

/** Reusable scratch so the per-instance path allocates nothing. */
const TINT_SCRATCH: [number, number, number] = [0, 0, 0];

function rampTint(value: number, out: [number, number, number]): void {
  // Cool-to-hot through the accent hues, matching the voltage ramp's endpoints
  // without paying for a full gradient lookup per instance.
  const t = value < 0 ? 0 : value > 1 ? 1 : value;
  const rgb = hslToRgb(0.62 - 0.62 * t, 0.85, 0.42 + 0.22 * t);
  out[0] = rgb[0];
  out[1] = rgb[1];
  out[2] = rgb[2];
}

/**
 * Write one cell's tint in linear space.
 *
 * `identity` is the mode that makes a dense field readable: the hue comes from
 * the neuron's own procedural seed, so a cell keeps its colour across reloads,
 * across layout changes, and between the scene and the chrome's swatches.
 */
function writeTint(
  out: Float32Array,
  at: number,
  mode: ColorMode,
  neurons: SimulationBuffers['neurons'],
  slot: number,
): void {
  switch (mode) {
    case 'identity': {
      const rgb = identityColor(neurons.seed[slot]);
      out[at] = srgbChannelToLinear(rgb[0]);
      out[at + 1] = srgbChannelToLinear(rgb[1]);
      out[at + 2] = srgbChannelToLinear(rgb[2]);
      return;
    }
    case 'population': {
      const population = neurons.population[slot];
      // Unassigned cells stay neutral so grouped ones stand out against them.
      if (population === 0xffff) {
        out[at] = 0.20;
        out[at + 1] = 0.22;
        out[at + 2] = 0.25;
        return;
      }
      // Offset keeps population hues from colliding with the identity sequence.
      const rgb = identityColor(population * 2654435761 + 0x9e37);
      out[at] = srgbChannelToLinear(rgb[0]);
      out[at + 1] = srgbChannelToLinear(rgb[1]);
      out[at + 2] = srgbChannelToLinear(rgb[2]);
      return;
    }
    case 'receptor': {
      const tint = TRANSMITTER_TINT[neurons.polarity[slot] === 1 ? 1 : 0];
      out[at] = tint[0];
      out[at + 1] = tint[1];
      out[at + 2] = tint[2];
      return;
    }
    case 'polarity': {
      const tint = POLARITY_TINT[neurons.polarity[slot] === 1 ? 1 : 0];
      out[at] = tint[0];
      out[at + 1] = tint[1];
      out[at + 2] = tint[2];
      return;
    }
    case 'voltage': {
      rampTint((neurons.v[slot] + 80) / 110, TINT_SCRATCH);
      out[at] = srgbChannelToLinear(TINT_SCRATCH[0]);
      out[at + 1] = srgbChannelToLinear(TINT_SCRATCH[1]);
      out[at + 2] = srgbChannelToLinear(TINT_SCRATCH[2]);
      return;
    }
    case 'rate': {
      rampTint(neurons.rate[slot] / 80, TINT_SCRATCH);
      out[at] = srgbChannelToLinear(TINT_SCRATCH[0]);
      out[at + 1] = srgbChannelToLinear(TINT_SCRATCH[1]);
      out[at + 2] = srgbChannelToLinear(TINT_SCRATCH[2]);
      return;
    }
    default: {
      out[at] = 1;
      out[at + 1] = 1;
      out[at + 2] = 1;
    }
  }
}

/** Single-channel sRGB to linear; the triple form lives in shared/theme. */
function srgbChannelToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

interface InstancePool {
  offset: THREE.InstancedBufferAttribute;
  scale: THREE.InstancedBufferAttribute;
  voltage: THREE.InstancedBufferAttribute;
  flash: THREE.InstancedBufferAttribute;
  polarity: THREE.InstancedBufferAttribute;
  flags: THREE.InstancedBufferAttribute;
  tint: THREE.InstancedBufferAttribute;
}

interface FieldGroup {
  slots: Uint32Array;
  count: number;
  capacity: number;
  pool: InstancePool;
  sources: THREE.BufferGeometry[];
  geometries: THREE.InstancedBufferGeometry[];
  meshes: THREE.Mesh[];
}

function createPool(capacity: number): InstancePool {
  return {
    offset: instancedAttribute(capacity, 3),
    scale: instancedAttribute(capacity, 1),
    voltage: instancedAttribute(capacity, 1),
    flash: instancedAttribute(capacity, 1),
    polarity: instancedAttribute(capacity, 1),
    flags: instancedAttribute(capacity, 1),
    tint: instancedAttribute(capacity, 3),
  };
}

function bindPool(geometry: THREE.InstancedBufferGeometry, pool: InstancePool): void {
  geometry.setAttribute('instanceOffset', pool.offset);
  geometry.setAttribute('instanceScale', pool.scale);
  geometry.setAttribute('instanceVoltage', pool.voltage);
  geometry.setAttribute('instanceFlash', pool.flash);
  geometry.setAttribute('instancePolarity', pool.polarity);
  geometry.setAttribute('instanceFlags', pool.flags);
  geometry.setAttribute('instanceTint', pool.tint);
}

type PartStyle = {
  swell: number;
  rim: number;
  emissive: number;
  roughness: number;
  translucency: number;
  opacity: number;
};

/*
 * Surfaces are matte and opaque, the way a segmentation mesh reads: a strong rim
 * and a translucent membrane make a cell look like glass, and glass takes its
 * colour from whatever is behind it — which in a dense field is other cells.
 *
 * Emissive is small because the identity palette leaves no headroom above it.
 * Neuroglancer's colour model pins value to 1, so a cell is already at full
 * channel intensity before anything is added — the old strengths were tuned
 * against a dimmer palette and now drive every firing cell straight to white,
 * where the bloom pass smears the whole network into one glowing mass. These
 * values keep a spike clearly brighter than rest while staying inside the range
 * the tone map can still resolve as colour.
 */
const PART_STYLE: readonly PartStyle[] = [
  { swell: 0.26, rim: 0.5, emissive: 0.85, roughness: 0.62, translucency: 0.22, opacity: 1 },
  { swell: 0.1, rim: 0.36, emissive: 0.5, roughness: 0.72, translucency: 0.16, opacity: 1 },
  { swell: 0.08, rim: 0.3, emissive: 0.4, roughness: 0.78, translucency: 0.12, opacity: 1 },
];

function createMaterial(part: number): THREE.ShaderMaterial {
  const style = PART_STYLE[part];
  return new THREE.ShaderMaterial({
    vertexShader: NEURON_VERTEX_GLSL,
    fragmentShader: NEURON_FRAGMENT_GLSL,
    transparent: true,
    depthWrite: true,
    side: THREE.FrontSide,
    uniforms: {
      uNeuronScale: { value: DEFAULT_RENDER_SETTINGS.neuronScale },
      uSpikeSwell: { value: style.swell },
      uExcitatoryColor: { value: linearColor(POLARITY_COLORS.excitatory) },
      uInhibitoryColor: { value: linearColor(POLARITY_COLORS.inhibitory) },
      // View space: the key light rides with the camera so a neuron never turns
      // its unlit side to the viewer while they orbit.
      uLightDirection: { value: new THREE.Vector3(0.42, 0.68, 0.6).normalize() },
      uLightColor: { value: linearColor('#CFE4FF').multiplyScalar(1.25) },
      uFillColor: { value: linearColor('#2E4A7D') },
      uAmbientColor: { value: linearColor('#131A22') },
      uRimColor: { value: linearColor(COLORS.accent) },
      uSelectionColor: { value: linearColor(COLORS.accent) },
      uHoverColor: { value: linearColor(COLORS.text) },
      uFogColor: { value: linearColor(COLORS.background) },
      uRimPower: { value: 2.6 },
      uRimStrength: { value: style.rim },
      uEmissiveStrength: { value: style.emissive },
      uRoughness: { value: style.roughness },
      uOpacity: { value: style.opacity },
      uGhostOpacity: { value: 0.16 },
      uVoltageColoring: { value: DEFAULT_RENDER_SETTINGS.voltageColoring ? 1 : 0 },
      uSaturation: { value: DEFAULT_RENDER_SETTINGS.saturation },
      uDimUnselected: { value: DEFAULT_RENDER_SETTINGS.dimUnselected },
      uHasSelection: { value: 0 },
      uTranslucency: { value: style.translucency },
      uFogDensity: { value: DEFAULT_RENDER_SETTINGS.fogDensity },
    },
  });
}

export class NeuronField extends THREE.Group {
  readonly #library: GlyphLibrary;
  readonly #materials: THREE.ShaderMaterial[] = [];
  readonly #groups = new Map<number, FieldGroup>();
  /** Same groups as `#groups`, in a form a frame loop can walk without a Map
   * iterator; iterating a Map allocates, and this runs twice per frame. */
  readonly #order: FieldGroup[] = [];
  readonly #counts = new Map<number, number>();
  readonly #seeds = new Map<number, number>();

  /** Renderer-owned spike envelope, indexed by neuron slot. */
  #envelope = new Float32Array(0);

  #hash: SpatialHash | null = null;
  #hashCell = 0;
  #hashAge = 0;
  #positions: Float32Array | null = null;
  #neuronCount = 0;
  #pickRadius = 1;
  #neuronScale = DEFAULT_RENDER_SETTINGS.neuronScale;
  /** Last mode seen by update(); a rebuild can happen before the first frame. */
  #colorMode: ColorMode = DEFAULT_RENDER_SETTINGS.colorMode;
  #hasSelection = false;

  constructor(library: GlyphLibrary) {
    super();
    this.#library = library;
    this.name = 'NeuronField';
    for (let part = 0; part < PART_COUNT; part += 1) {
      this.#materials.push(createMaterial(part));
    }
  }

  /** Rebuild instance pools from the buffers. Call after a structural edit. */
  rebuild(buffers: SimulationBuffers): void {
    const neurons = buffers.neurons;
    const count = neurons.count;

    this.#counts.clear();
    this.#seeds.clear();
    for (let i = 0; i < count; i += 1) {
      const key = variantKey(neurons.archetype[i], variantOf(neurons.seed[i]));
      const running = this.#counts.get(key);
      if (running === undefined) {
        this.#counts.set(key, 1);
        this.#seeds.set(key, neurons.seed[i]);
      } else {
        this.#counts.set(key, running + 1);
      }
    }

    for (const [key, group] of this.#groups) {
      if (!this.#counts.has(key)) {
        this.#destroyGroup(group);
        this.#groups.delete(key);
      }
    }

    for (const [key, needed] of this.#counts) {
      const existing = this.#groups.get(key);
      if (existing && existing.capacity >= needed) {
        existing.count = 0;
        continue;
      }
      // Growing means new instance buffers, and the only way to hand the old
      // ones back to the driver is to dispose the geometry that owns them.
      if (existing) this.#destroyGroup(existing);
      this.#groups.set(key, this.#createGroup(key, this.#seeds.get(key) ?? 0, needed));
    }

    this.#order.length = 0;
    for (const group of this.#groups.values()) this.#order.push(group);

    for (let i = 0; i < count; i += 1) {
      const key = variantKey(neurons.archetype[i], variantOf(neurons.seed[i]));
      const group = this.#groups.get(key);
      if (!group) continue;
      group.slots[group.count] = i;
      group.count += 1;
    }

    if (this.#envelope.length < count) this.#envelope = new Float32Array(growthCapacity(count, 0));

    let pickRadius = 0;
    for (let i = 0; i < count; i += 1) {
      const radius = somaRadiusForCode(neurons.archetype[i]) * neurons.scale[i];
      if (radius > pickRadius) pickRadius = radius;
    }
    this.#pickRadius = pickRadius > 0 ? pickRadius : 1;
    this.#positions = neurons.position;
    this.#neuronCount = count;
    this.#refreshHash();
    this.#writeInstances(buffers, 0, this.#colorMode);
  }

  /** Per-frame attribute refresh. Touches no allocator and no geometry. */
  update(buffers: SimulationBuffers, dt: number, settings: RenderSettings): void {
    // Whether anything is selected drives the dimming of everything else, so it
    // is answered once per frame with an early exit rather than per instance.
    const flagColumn = buffers.neurons.flags;
    const neuronTotal = buffers.neurons.count;
    let anySelected = false;
    for (let i = 0; i < neuronTotal; i += 1) {
      if ((flagColumn[i] & NEURON_FLAG.SELECTED) !== 0) {
        anySelected = true;
        break;
      }
    }
    this.#hasSelection = anySelected;

    this.#neuronScale = settings.neuronScale;
    this.#colorMode = settings.colorMode;
    for (let part = 0; part < PART_COUNT; part += 1) {
      const uniforms = this.#materials[part].uniforms;
      uniforms.uNeuronScale.value = settings.neuronScale;
      uniforms.uVoltageColoring.value = settings.voltageColoring ? 1 : 0;
      uniforms.uFogDensity.value = settings.fogDensity;
      uniforms.uSaturation.value = settings.saturation;
      uniforms.uDimUnselected.value = settings.dimUnselected;
      uniforms.uHasSelection.value = this.#hasSelection ? 1 : 0;
    }

    for (let g = 0; g < this.#order.length; g += 1) {
      const meshes = this.#order[g].meshes;
      meshes[PART_DENDRITES].visible = settings.showDendrites;
      meshes[PART_AXON].visible = settings.showAxons;
    }

    this.#writeInstances(buffers, dt, settings.colorMode);

    this.#hashAge += dt;
    if (this.#hashAge >= HASH_INTERVAL) {
      this.#positions = buffers.neurons.position;
      this.#neuronCount = buffers.neurons.count;
      this.#refreshHash();
    }
  }

  /** Neuron slot under the ray, or -1. */
  raycastSlot(raycaster: THREE.Raycaster): number {
    const hash = this.#hash;
    if (!hash || this.#neuronCount === 0) return -1;
    const origin = raycaster.ray.origin;
    const direction = raycaster.ray.direction;
    return hash.raycast(
      origin.x,
      origin.y,
      origin.z,
      direction.x,
      direction.y,
      direction.z,
      this.#pickRadius * this.#neuronScale,
    );
  }

  dispose(): void {
    for (const group of this.#groups.values()) this.#destroyGroup(group);
    this.#groups.clear();
    this.#order.length = 0;
    for (const material of this.#materials) material.dispose();
    this.#materials.length = 0;
    this.#hash = null;
    this.#positions = null;
  }

  #writeInstances(buffers: SimulationBuffers, dt: number, mode: ColorMode): void {
    const neurons = buffers.neurons;
    const position = neurons.position;
    const scaleColumn = neurons.scale;
    const voltage = neurons.v;
    const spike = neurons.spike;
    const simFlash = neurons.flash;
    const polarity = neurons.polarity;
    const flags = neurons.flags;
    const enabled = neurons.enabled;
    const envelope = this.#envelope;
    const decay = dt > 0 ? Math.exp(-dt / FLASH_TAU) : 1;

    for (let g = 0; g < this.#order.length; g += 1) {
      const group = this.#order[g];
      const slots = group.slots;
      const total = group.count;
      const offsetArray = group.pool.offset.array as Float32Array;
      const scaleArray = group.pool.scale.array as Float32Array;
      const voltageArray = group.pool.voltage.array as Float32Array;
      const flashArray = group.pool.flash.array as Float32Array;
      const polarityArray = group.pool.polarity.array as Float32Array;
      const flagsArray = group.pool.flags.array as Float32Array;
      const tintArray = group.pool.tint.array as Float32Array;

      for (let i = 0; i < total; i += 1) {
        const slot = slots[i];
        const p = slot * 3;
        offsetArray[i * 3] = position[p];
        offsetArray[i * 3 + 1] = position[p + 1];
        offsetArray[i * 3 + 2] = position[p + 2];
        scaleArray[i] = scaleColumn[slot];
        voltageArray[i] = voltage[slot];

        let level = envelope[slot] * decay;
        if (spike[slot] !== 0) level = 1;
        const reported = simFlash[slot];
        if (reported > level) level = reported;
        envelope[slot] = level;
        flashArray[i] = level;

        polarityArray[i] = polarity[slot];
        // A disabled neuron is still drawn, dimmed, which is exactly what the
        // ghost bit means to the fragment program.
        flagsArray[i] = enabled[slot] === 0 ? flags[slot] | NEURON_FLAG.GHOSTED : flags[slot];

        const t = i * 3;
        writeTint(tintArray, t, mode, neurons, slot);
      }

      group.pool.offset.needsUpdate = true;
      group.pool.scale.needsUpdate = true;
      group.pool.voltage.needsUpdate = true;
      group.pool.flash.needsUpdate = true;
      group.pool.polarity.needsUpdate = true;
      group.pool.flags.needsUpdate = true;
      group.pool.tint.needsUpdate = true;
      for (let part = 0; part < PART_COUNT; part += 1) {
        group.geometries[part].instanceCount = total;
      }
    }
  }

  #refreshHash(): void {
    const positions = this.#positions;
    if (!positions) return;
    const cell = Math.max(0.5, this.#pickRadius * HASH_CELL_FACTOR);
    if (!this.#hash || Math.abs(cell - this.#hashCell) > this.#hashCell * 0.5) {
      this.#hash = new SpatialHash(cell);
      this.#hashCell = cell;
    }
    this.#hash.rebuild(positions, this.#neuronCount);
    this.#hashAge = 0;
  }

  #createGroup(key: number, seed: number, needed: number): FieldGroup {
    const archetype = Math.floor(key / VARIANTS_PER_ARCHETYPE);
    const glyph = this.#library.get(archetype, seed);
    const capacity = growthCapacity(needed, 0);
    const pool = createPool(capacity);
    const sources = [glyph.soma, glyph.dendrites, glyph.axon];
    const geometries: THREE.InstancedBufferGeometry[] = [];
    const meshes: THREE.Mesh[] = [];

    for (let part = 0; part < PART_COUNT; part += 1) {
      const geometry = shareGeometry(sources[part]);
      bindPool(geometry, pool);
      geometry.instanceCount = 0;
      const mesh = new THREE.Mesh(geometry, this.#materials[part]);
      // Instance offsets live in an attribute, so the glyph's own bounds say
      // nothing about where the field is; culling is handled by the scene graph
      // above this node instead.
      mesh.frustumCulled = false;
      mesh.renderOrder = part === PART_SOMA ? 1 : 0;
      geometries.push(geometry);
      meshes.push(mesh);
      this.add(mesh);
    }

    return {
      slots: new Uint32Array(capacity),
      count: 0,
      capacity,
      pool,
      sources,
      geometries,
      meshes,
    };
  }

  #destroyGroup(group: FieldGroup): void {
    for (let part = 0; part < PART_COUNT; part += 1) {
      this.remove(group.meshes[part]);
      releaseGeometry(group.geometries[part], group.sources[part]);
    }
  }
}
