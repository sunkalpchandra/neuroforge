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

/** `NeuronBuffers.archetype` stores an index into `MORPHOLOGY_ARCHETYPES`. */
export function archetypeName(code: number): MorphologyArchetype {
  const index = Math.floor(code);
  if (index >= 0 && index < ARCHETYPE_COUNT) return MORPHOLOGY_ARCHETYPES[index];
  return MORPHOLOGY_ARCHETYPES[0];
}

/** Seed bucket a neuron falls into. Identical buckets share one geometry. */
export function variantOf(seed: number): number {
  return (hashSeed(seed) >>> 0) % VARIANTS_PER_ARCHETYPE;
}

/** Stable cache key for an (archetype, bucket) pair. */
export function variantKey(archetype: number, variant: number): number {
  return archetype * VARIANTS_PER_ARCHETYPE + variant;
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
  const index = Math.floor(code);
  if (index >= 0 && index < ARCHETYPE_COUNT) return RADIUS_BY_ARCHETYPE[index];
  return RADIUS_BY_ARCHETYPE[0];
}
