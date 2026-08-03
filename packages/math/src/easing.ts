import { TAU, clamp01 } from './internal';

/**
 * Easing curves, frame-rate-independent damping and spring integrators.
 *
 * Nothing in the product animates on a fixed step, so every time-dependent
 * routine here takes a real delta and produces the same trajectory whether it is
 * called at 30, 60 or 144 Hz.
 */

export const easeOutExpo = (t: number): number => {
  const x = clamp01(t);
  return x >= 1 ? 1 : 1 - Math.pow(2, -10 * x);
};

export const easeInOutQuart = (t: number): number => {
  const x = clamp01(t);
  return x < 0.5 ? 8 * x * x * x * x : 1 - Math.pow(-2 * x + 2, 4) / 2;
};

const BACK_C1 = 1.70158;
const BACK_C3 = BACK_C1 + 1;

export const easeOutBack = (t: number): number => {
  const x = clamp01(t);
  const u = x - 1;
  return 1 + BACK_C3 * u * u * u + BACK_C1 * u * u;
};

const ELASTIC_C4 = TAU / 3;

export const easeOutElastic = (t: number): number => {
  const x = clamp01(t);
  if (x === 0 || x === 1) return x;
  return Math.pow(2, -10 * x) * Math.sin((x * 10 - 0.75) * ELASTIC_C4) + 1;
};

/**
 * Exponential approach to a target. `lambda` is the rate in reciprocal units of
 * `dt`: after 1/lambda the remaining error has fallen by 1/e. Because the decay
 * is exponential, composing two half-steps gives exactly the same result as one
 * whole step, which is the property lerp-by-a-constant-factor lacks.
 */
export function damp(current: number, target: number, lambda: number, dt: number): number {
  return target + (current - target) * Math.exp(-lambda * dt);
}

/** Wrap to (-pi, pi]. */
function wrapPi(angle: number): number {
  const a = (angle + Math.PI) % TAU;
  return (a < 0 ? a + TAU : a) - Math.PI;
}

/** As `damp`, but takes the shorter way round the circle. */
export function dampAngle(current: number, target: number, lambda: number, dt: number): number {
  const delta = wrapPi(target - current);
  return wrapPi(damp(current, current + delta, lambda, dt));
}

const DEFAULT_STIFFNESS = 170;
const DEFAULT_DAMPING = 26;
const DEFAULT_MASS = 1;

/** Substep ceiling in seconds; keeps stiff springs stable when a frame is long. */
const MAX_SUBSTEP = 1 / 240;
/** Hard cap so a tab that was backgrounded for a minute cannot stall the frame. */
const MAX_SUBSTEPS = 64;

const SETTLE_DISPLACEMENT = 1e-4;
const SETTLE_VELOCITY = 1e-3;

function substepCount(dt: number): number {
  const wanted = Math.ceil(dt / MAX_SUBSTEP);
  return wanted < 1 ? 1 : wanted > MAX_SUBSTEPS ? MAX_SUBSTEPS : wanted;
}

/**
 * One degree of freedom of a damped harmonic oscillator, integrated with
 * semi-implicit (symplectic) Euler: velocity is updated from the current
 * position, then position from the *new* velocity. Explicit Euler injects energy
 * and blows up at large steps; this form does not.
 */
class SpringAxis {
  value: number;
  target: number;
  velocity = 0;

  constructor(value: number) {
    this.value = value;
    this.target = value;
  }

  advance(stiffness: number, damping: number, invMass: number, h: number): void {
    const accel = (-stiffness * (this.value - this.target) - damping * this.velocity) * invMass;
    this.velocity += accel * h;
    this.value += this.velocity * h;
  }

  jump(value: number): void {
    this.value = value;
    this.target = value;
    this.velocity = 0;
  }

  get settled(): boolean {
    return (
      Math.abs(this.value - this.target) <= SETTLE_DISPLACEMENT &&
      Math.abs(this.velocity) <= SETTLE_VELOCITY
    );
  }
}

export class SpringScalar {
  #axis: SpringAxis;
  #stiffness: number;
  #damping: number;
  #invMass: number;

  constructor(
    value: number,
    stiffness = DEFAULT_STIFFNESS,
    damping = DEFAULT_DAMPING,
    mass = DEFAULT_MASS,
  ) {
    this.#axis = new SpringAxis(value);
    this.#stiffness = stiffness;
    this.#damping = damping;
    this.#invMass = mass > 0 ? 1 / mass : 1;
  }

  set target(v: number) {
    this.#axis.target = v;
  }

  get target(): number {
    return this.#axis.target;
  }

  get value(): number {
    return this.#axis.value;
  }

  /** Teleport: value and target move together and velocity is cleared. */
  jump(v: number): void {
    this.#axis.jump(v);
  }

  step(dt: number): number {
    if (!(dt > 0)) return this.#axis.value;
    const steps = substepCount(dt);
    const h = dt / steps;
    for (let i = 0; i < steps; i += 1) {
      this.#axis.advance(this.#stiffness, this.#damping, this.#invMass, h);
    }
    return this.#axis.value;
  }

  get settled(): boolean {
    return this.#axis.settled;
  }
}

export class SpringVec3 {
  #x: SpringAxis;
  #y: SpringAxis;
  #z: SpringAxis;
  #stiffness: number;
  #damping: number;
  #invMass: number;

  constructor(
    x: number,
    y: number,
    z: number,
    stiffness = DEFAULT_STIFFNESS,
    damping = DEFAULT_DAMPING,
    mass = DEFAULT_MASS,
  ) {
    this.#x = new SpringAxis(x);
    this.#y = new SpringAxis(y);
    this.#z = new SpringAxis(z);
    this.#stiffness = stiffness;
    this.#damping = damping;
    this.#invMass = mass > 0 ? 1 / mass : 1;
  }

  setTarget(x: number, y: number, z: number): void {
    this.#x.target = x;
    this.#y.target = y;
    this.#z.target = z;
  }

  jump(x: number, y: number, z: number): void {
    this.#x.jump(x);
    this.#y.jump(y);
    this.#z.jump(z);
  }

  step(dt: number): void {
    if (!(dt > 0)) return;
    const steps = substepCount(dt);
    const h = dt / steps;
    const k = this.#stiffness;
    const c = this.#damping;
    const m = this.#invMass;
    for (let i = 0; i < steps; i += 1) {
      this.#x.advance(k, c, m, h);
      this.#y.advance(k, c, m, h);
      this.#z.advance(k, c, m, h);
    }
  }

  get x(): number {
    return this.#x.value;
  }

  get y(): number {
    return this.#y.value;
  }

  get z(): number {
    return this.#z.value;
  }

  get settled(): boolean {
    return this.#x.settled && this.#y.settled && this.#z.settled;
  }
}
