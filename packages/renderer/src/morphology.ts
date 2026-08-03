import { MORPHOLOGY_ARCHETYPES, defaultMorphology } from '@neuroforge/shared';
import type { Morphology, MorphologyArchetype } from '@neuroforge/shared';
import { hashSeed } from '@neuroforge/math';

/**
 * Distinct procedural variants generated per archetype.
 *
 * The glyph cache is bounded by variant count, not by neuron count: a hundred
 * thousand neurons still resolve to `ARCHETYPES * VARIANTS_PER_ARCHETYPE`
 * geometries, and every neuron sharing a bucket shares one set of buffers.
 */
export const VARIANTS_PER_ARCHETYPE = 3;

export const ARCHETYPE_COUNT = MORPHOLOGY_ARCHETYPES.length;

/**
 * Number of distinct glyphs the renderer can ever ask for. Every cache key is
 * below this, which is what lets the glyph library hold all of them at once.
 */
export const VARIANT_COUNT = ARCHETYPE_COUNT * VARIANTS_PER_ARCHETYPE;

/**
 * `NeuronBuffers.archetype` is a `Uint8Array`, so it can hold codes no archetype
 * corresponds to. Folding those onto the fallback here rather than at each use
 * keeps the key space exactly `VARIANT_COUNT` wide: without it a stray code
 * would mint its own cache entry for a glyph identical to the fallback's.
 */
export function archetypeCode(code: number): number {
  const index = Math.floor(code);
  return index >= 0 && index < ARCHETYPE_COUNT ? index : 0;
}

/** `NeuronBuffers.archetype` stores an index into `MORPHOLOGY_ARCHETYPES`. */
export function archetypeName(code: number): MorphologyArchetype {
  return MORPHOLOGY_ARCHETYPES[archetypeCode(code)];
}

/** Seed bucket a neuron falls into. Identical buckets share one geometry. */
export function variantOf(seed: number): number {
  return (hashSeed(seed) >>> 0) % VARIANTS_PER_ARCHETYPE;
}

/** Stable cache key for an (archetype, bucket) pair, always below `VARIANT_COUNT`. */
export function variantKey(archetype: number, variant: number): number {
  return archetypeCode(archetype) * VARIANTS_PER_ARCHETYPE + variant;
}

/**
 * The descriptor a variant is built from. Every neuron in the bucket renders
 * this morphology, so the per-neuron seed only selects the bucket.
 */
export function morphologyForVariant(archetype: number, variant: number): Morphology {
  const name = archetypeName(archetype);
  return defaultMorphology(name, hashSeed(archetype, variant, 0x51ed270b));
}

/** Radius of the soma in glyph-local units, before the per-instance scale. */
export function somaRadiusOf(morphology: Morphology): number {
  return morphology.somaRadius * morphology.scale;
}

const RADIUS_BY_ARCHETYPE: Float32Array = buildRadiusTable();

function buildRadiusTable(): Float32Array {
  const table = new Float32Array(ARCHETYPE_COUNT);
  for (let i = 0; i < ARCHETYPE_COUNT; i += 1) {
    table[i] = somaRadiusOf(defaultMorphology(MORPHOLOGY_ARCHETYPES[i], 1));
  }
  return table;
}

/** Soma radius for an archetype code, used for picking and selection halos. */
export function somaRadiusForCode(code: number): number {
  return RADIUS_BY_ARCHETYPE[archetypeCode(code)];
}
