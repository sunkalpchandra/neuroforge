import type * as THREE from 'three';
import { buildAxonGeometry, buildDendriteGeometry, createSomaGeometry } from './glyph';
import { VARIANT_COUNT, morphologyForVariant, variantKey, variantOf } from './morphology';

export interface GlyphGeometries {
  soma: THREE.BufferGeometry;
  dendrites: THREE.BufferGeometry;
  axon: THREE.BufferGeometry;
}

const DEFAULT_MAX_VARIANTS = 24;
const SOMA_DETAIL = 2;

/**
 * Least-recently-used cache of built glyphs, keyed by archetype and seed bucket.
 *
 * A hundred thousand neurons collapse onto at most `VARIANT_COUNT` entries
 * because the seed is bucketed before it is used, so the build cost is a
 * function of the variant budget rather than of network size. Eviction is
 * genuine: the evicted geometries are disposed, and so is everything still
 * resident when the library itself is disposed.
 */
export class GlyphLibrary {
  readonly #maxVariants: number;
  /** Insertion order is recency order: Map preserves it and re-insert refreshes it. */
  readonly #entries = new Map<number, GlyphGeometries>();

  constructor(maxVariants = DEFAULT_MAX_VARIANTS) {
    const requested = Number.isFinite(maxVariants)
      ? Math.floor(maxVariants)
      : DEFAULT_MAX_VARIANTS;
    // The floor is not a tuning choice, it is a correctness one. A `NeuronField`
    // hands the cached `BufferAttribute` objects straight to its instance pools
    // rather than copying them, so disposing an entry that is still on screen
    // frees the very GPU buffers those pools draw from. Since there are only
    // `VARIANT_COUNT` distinct glyphs in existence, holding every one of them
    // costs a few megabytes and makes that impossible by construction; a larger
    // request is still honoured, in case the key space ever widens.
    this.#maxVariants = Math.max(VARIANT_COUNT, requested);
  }

  get(archetype: number, seed: number): GlyphGeometries {
    const key = variantKey(archetype, variantOf(seed));
    const hit = this.#entries.get(key);
    if (hit) {
      this.#entries.delete(key);
      this.#entries.set(key, hit);
      return hit;
    }

    const morphology = morphologyForVariant(archetype, variantOf(seed));
    const built: GlyphGeometries = {
      soma: createSomaGeometry(morphology, SOMA_DETAIL),
      dendrites: buildDendriteGeometry(morphology),
      axon: buildAxonGeometry(morphology),
    };
    this.#entries.set(key, built);

    while (this.#entries.size > this.#maxVariants) {
      const oldest = this.#entries.keys().next();
      if (oldest.done) break;
      const evicted = this.#entries.get(oldest.value);
      this.#entries.delete(oldest.value);
      if (evicted) disposeGlyph(evicted);
    }

    return built;
  }

  dispose(): void {
    for (const entry of this.#entries.values()) disposeGlyph(entry);
    this.#entries.clear();
  }
}

function disposeGlyph(glyph: GlyphGeometries): void {
  glyph.soma.dispose();
  glyph.dendrites.dispose();
  glyph.axon.dispose();
}
