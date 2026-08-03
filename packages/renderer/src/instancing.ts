import * as THREE from 'three';

/**
 * Wrap a glyph geometry for instanced drawing without copying it.
 *
 * The vertex attributes and the index are the *same* `BufferAttribute` objects
 * as the cached glyph, so all three parts of a neuron share one GPU upload no
 * matter how many instance groups reference them. Only the per-instance buffers
 * are owned by the wrapper.
 */
export function shareGeometry(source: THREE.BufferGeometry): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  for (const name of Object.keys(source.attributes)) {
    geometry.setAttribute(name, source.attributes[name]);
  }
  geometry.index = source.index;
  geometry.boundingBox = source.boundingBox;
  geometry.boundingSphere = source.boundingSphere;
  geometry.instanceCount = 0;
  return geometry;
}

/**
 * Free only the instance buffers the wrapper owns.
 *
 * Detaching the shared attributes first is what stops disposal from deleting
 * the glyph's vertex buffers out from under the cache that still owns them.
 */
export function releaseGeometry(
  geometry: THREE.InstancedBufferGeometry,
  source: THREE.BufferGeometry,
): void {
  for (const name of Object.keys(source.attributes)) {
    geometry.deleteAttribute(name);
  }
  geometry.index = null;
  geometry.dispose();
}

export function instancedAttribute(
  capacity: number,
  itemSize: number,
): THREE.InstancedBufferAttribute {
  const attribute = new THREE.InstancedBufferAttribute(
    new Float32Array(capacity * itemSize),
    itemSize,
  );
  attribute.setUsage(THREE.DynamicDrawUsage);
  return attribute;
}

/** Capacity policy shared by every instance pool: grow in coarse blocks. */
export function growthCapacity(required: number, current: number): number {
  if (required <= current) return current;
  let next = Math.max(64, current);
  while (next < required) next *= 2;
  return next;
}
