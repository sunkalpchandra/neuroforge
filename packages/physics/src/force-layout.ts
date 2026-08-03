import type { SimulationBuffers } from '@neuroforge/shared';

import { Octree } from './octree';

export interface LayoutSettings {
  /** Strength of the inverse-square repulsion between every pair of neurons. */
  repulsion: number;
  /** Spring constant pulling connected neurons together. */
  attraction: number;
  /** Pull toward the origin, which stops disconnected components drifting away. */
  gravity: number;
  /** Velocity retained per step; 1 is frictionless, 0 halts instantly. */
  damping: number;
  /** Barnes-Hut opening angle. Larger is faster and coarser; 0 is exact. */
  theta: number;
  /** Upper bound on per-step displacement, in world units per second. */
  maxSpeed: number;
  /** Collapse to the XY plane when 2. */
  dimensions: 2 | 3;
  seed: number;
}

export const DEFAULT_LAYOUT_SETTINGS: LayoutSettings = {
  repulsion: 240,
  attraction: 0.06,
  gravity: 0.02,
  damping: 0.82,
  theta: 0.7,
  maxSpeed: 60,
  dimensions: 3,
  seed: 0x2545f491,
};

/**
 * Softening length for the repulsion kernel. Without it, two neurons at the same
 * position produce an infinite force and the layout explodes on the first step.
 */
const SOFTENING = 1.2;

/**
 * Deterministic force-directed layout over the live position column.
 *
 * Positions are mutated in place in `buffers.neurons.position`, so the renderer
 * and the picker see the result with no copy. The solver is deterministic given
 * the same seed and the same starting positions: there is no Math.random in the
 * step path, which is what lets a laid-out circuit round-trip through a saved
 * document unchanged.
 */
export class ForceLayout {
  private settings: LayoutSettings;
  private buffers: SimulationBuffers | null = null;
  private tree = new Octree();

  private velocity = new Float32Array(0);
  private pinned = new Uint8Array(0);
  private force = { x: 0, y: 0, z: 0 };

  constructor(settings: Partial<LayoutSettings> = {}) {
    this.settings = { ...DEFAULT_LAYOUT_SETTINGS, ...settings };
  }

  attach(buffers: SimulationBuffers): void {
    this.buffers = buffers;
    this.ensureCapacity(buffers.neurons.capacity);
  }

  private ensureCapacity(capacity: number): void {
    if (this.velocity.length >= capacity * 3) return;
    const velocity = new Float32Array(capacity * 3);
    velocity.set(this.velocity);
    this.velocity = velocity;
    const pinned = new Uint8Array(capacity);
    pinned.set(this.pinned);
    this.pinned = pinned;
  }

  setSettings(patch: Partial<LayoutSettings>): void {
    this.settings = { ...this.settings, ...patch };
  }

  getSettings(): Readonly<LayoutSettings> {
    return this.settings;
  }

  pin(slot: number, pinned: boolean): void {
    if (slot < 0 || slot >= this.pinned.length) return;
    this.pinned[slot] = pinned ? 1 : 0;
  }

  isPinned(slot: number): boolean {
    return slot >= 0 && slot < this.pinned.length && this.pinned[slot] === 1;
  }

  reset(): void {
    this.velocity.fill(0);
  }

  /**
   * One relaxation step. Returns the total kinetic energy of the system, which
   * falls monotonically as the layout settles and is what `solve` tests against.
   */
  step(dt: number): number {
    const buffers = this.buffers;
    if (buffers === null) return 0;

    const { neurons, synapses } = buffers;
    const count = neurons.count;
    if (count === 0) return 0;

    this.ensureCapacity(neurons.capacity);

    const positions = neurons.position;
    const velocity = this.velocity;
    const { repulsion, attraction, gravity, damping, theta, maxSpeed, dimensions } = this.settings;

    // Clamp the timestep. A long frame must not be allowed to inject enough
    // impulse to throw the layout apart before damping can absorb it.
    const h = Math.min(dt, 1 / 30);

    this.tree.build(positions, count);

    for (let i = 0; i < count; i += 1) {
      if (this.pinned[i] === 1) continue;
      const p = i * 3;
      const x = positions[p];
      const y = positions[p + 1];
      const z = positions[p + 2];

      this.tree.repulsion(i, x, y, z, repulsion, theta, SOFTENING, this.force);

      velocity[p] += this.force.x * h;
      velocity[p + 1] += this.force.y * h;
      velocity[p + 2] += this.force.z * h;

      velocity[p] -= x * gravity * h;
      velocity[p + 1] -= y * gravity * h;
      velocity[p + 2] -= z * gravity * h;
    }

    // Springs along synapses. Force is applied to both endpoints so momentum is
    // conserved and the whole network does not drift.
    const synCount = synapses.count;
    for (let s = 0; s < synCount; s += 1) {
      if (synapses.enabled[s] === 0) continue;
      const a = synapses.pre[s];
      const b = synapses.post[s];
      if (a === b || a >= count || b >= count) continue;

      const pa = a * 3;
      const pb = b * 3;
      const dx = positions[pb] - positions[pa];
      const dy = positions[pb + 1] - positions[pa + 1];
      const dz = positions[pb + 2] - positions[pa + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;

      // Hooke's law about a rest length derived from the connection weight:
      // stronger connections sit closer together, which makes the topology
      // legible at a glance.
      const rest = 8 + 6 / (1 + Math.abs(synapses.weight[s]));
      const f = (dist - rest) * attraction;
      const ux = (dx / dist) * f;
      const uy = (dy / dist) * f;
      const uz = (dz / dist) * f;

      if (this.pinned[a] === 0) {
        velocity[pa] += ux * h;
        velocity[pa + 1] += uy * h;
        velocity[pa + 2] += uz * h;
      }
      if (this.pinned[b] === 0) {
        velocity[pb] -= ux * h;
        velocity[pb + 1] -= uy * h;
        velocity[pb + 2] -= uz * h;
      }
    }

    let energy = 0;
    const maxStep = maxSpeed * h;

    for (let i = 0; i < count; i += 1) {
      const p = i * 3;
      if (this.pinned[i] === 1) {
        velocity[p] = 0;
        velocity[p + 1] = 0;
        velocity[p + 2] = 0;
        continue;
      }

      velocity[p] *= damping;
      velocity[p + 1] *= damping;
      velocity[p + 2] *= damping;

      if (dimensions === 2) velocity[p + 2] = 0;

      let dx = velocity[p] * h;
      let dy = velocity[p + 1] * h;
      let dz = velocity[p + 2] * h;

      const stepLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (stepLen > maxStep && stepLen > 0) {
        const scale = maxStep / stepLen;
        dx *= scale;
        dy *= scale;
        dz *= scale;
      }

      positions[p] += dx;
      positions[p + 1] += dy;
      positions[p + 2] += dz;
      if (dimensions === 2) positions[p + 2] = 0;

      energy += velocity[p] * velocity[p] + velocity[p + 1] * velocity[p + 1] + velocity[p + 2] * velocity[p + 2];
    }

    return energy;
  }

  /**
   * Relax until the system is quiet or the iteration budget runs out. Returns
   * the final kinetic energy so a caller can tell convergence from exhaustion.
   *
   * `epsilon` is compared against energy *per neuron*, so the same threshold
   * means the same thing for a twenty-neuron sketch and a fifty-thousand-neuron
   * cortex. A total-energy threshold would be unreachable for any large network.
   */
  solve(maxIterations = 400, epsilon = 0.05): number {
    const count = this.buffers?.neurons.count ?? 0;
    if (count === 0) return 0;
    let energy = 0;
    for (let i = 0; i < maxIterations; i += 1) {
      energy = this.step(1 / 60);
      if (energy / count < epsilon) break;
    }
    return energy;
  }
}
