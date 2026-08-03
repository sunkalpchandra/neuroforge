/** Plain-data vector and bounds types shared by every package. */

export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quat {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Aabb {
  min: Vec3;
  max: Vec3;
}

export const vec3 = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });
export const vec2 = (x = 0, y = 0): Vec2 => ({ x, y });

export const ZERO3: Readonly<Vec3> = Object.freeze({ x: 0, y: 0, z: 0 });
export const ONE3: Readonly<Vec3> = Object.freeze({ x: 1, y: 1, z: 1 });

export const IDENTITY_QUAT: Readonly<Quat> = Object.freeze({ x: 0, y: 0, z: 0, w: 1 });

export const emptyAabb = (): Aabb => ({
  min: { x: Infinity, y: Infinity, z: Infinity },
  max: { x: -Infinity, y: -Infinity, z: -Infinity },
});

export function expandAabb(box: Aabb, p: Vec3): Aabb {
  if (p.x < box.min.x) box.min.x = p.x;
  if (p.y < box.min.y) box.min.y = p.y;
  if (p.z < box.min.z) box.min.z = p.z;
  if (p.x > box.max.x) box.max.x = p.x;
  if (p.y > box.max.y) box.max.y = p.y;
  if (p.z > box.max.z) box.max.z = p.z;
  return box;
}

export function aabbCenter(box: Aabb): Vec3 {
  return {
    x: (box.min.x + box.max.x) * 0.5,
    y: (box.min.y + box.max.y) * 0.5,
    z: (box.min.z + box.max.z) * 0.5,
  };
}

export function aabbIsEmpty(box: Aabb): boolean {
  return !Number.isFinite(box.min.x) || box.max.x < box.min.x;
}

/** Longest edge of the box; 0 for an empty box. */
export function aabbExtent(box: Aabb): number {
  if (aabbIsEmpty(box)) return 0;
  return Math.max(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z);
}
