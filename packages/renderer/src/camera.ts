import * as THREE from 'three';
import { SpringScalar, SpringVec3, damp, easeInOutQuart } from '@neuroforge/math';
import { DEFAULT_CAMERA } from '@neuroforge/shared';
import type { CameraState } from '@neuroforge/shared';

export type CameraMode = 'orbit' | 'fly' | 'first-person' | 'cinematic';

/**
 * Unified camera controller.
 *
 * Every quantity the camera is built from — pivot, yaw, pitch, distance, free
 * position, field of view — is a spring, and every input writes a spring
 * *target* rather than a value. Nothing in here ever assigns a pose directly
 * except `setState(state, true)`, which exists precisely so a document load can
 * teleport. That is what makes mode changes free: switching mode re-derives the
 * new mode's parameters from the pose the old mode is currently producing, so
 * the camera does not move at the instant of the switch and the springs carry it
 * from there.
 */

const WORLD_UP = new THREE.Vector3(0, 1, 0);

const ROTATE_SPEED = 0.0062;
const LOOK_SPEED = 0.0022;
const PAN_SPEED = 0.0017;

/**
 * Wheel response is exponential in the wheel delta, so one notch is always the
 * same *ratio*. That is what makes a dolly feel identical whether the camera is
 * two units from a soma or two hundred thousand from the whole network.
 */
const ZOOM_RATE = 0.0016;
const MIN_DISTANCE = 0.05;
const MAX_DISTANCE = 5e5;

const MAX_PITCH = Math.PI / 2 - 0.02;

/** Rate at which released-pointer spin bleeds away, in reciprocal seconds. */
const INERTIA_LAMBDA = 3.6;
const INERTIA_EPSILON = 1e-4;

const FLY_BASE_SPEED = 16;
const FLY_BOOST = 4;
const FLY_ACCEL_LAMBDA = 9;
const FLY_SPEED_MIN = 0.02;
const FLY_SPEED_MAX = 400;

const CINEMATIC_RATE = 0.075;

/** Distance to the synthetic look-at point used by the free-flight modes. */
const LOOK_AHEAD = 10;

const FRAME_PADDING = 1.18;
const DEFAULT_FRAME_DURATION = 0.9;

function clamp(value: number, low: number, high: number): number {
  return value < low ? low : value > high ? high : value;
}

interface Transition {
  active: boolean;
  elapsed: number;
  duration: number;
  fromX: number;
  fromY: number;
  fromZ: number;
  toX: number;
  toY: number;
  toZ: number;
  fromDistance: number;
  toDistance: number;
}

export class CameraRig {
  readonly #camera: THREE.PerspectiveCamera;
  readonly #dom: HTMLElement;

  #mode: CameraMode = DEFAULT_CAMERA.mode;

  readonly #focus = new SpringVec3(0, 0, 0, 110, 22, 1.1);
  readonly #flight = new SpringVec3(0, 0, 0, 240, 31, 1);
  readonly #yaw = new SpringScalar(0, 150, 24, 1);
  readonly #pitch = new SpringScalar(0, 150, 24, 1);
  readonly #distance = new SpringScalar(1, 130, 23, 1);
  readonly #fov = new SpringScalar(DEFAULT_CAMERA.fov, 170, 27, 1);

  #focusX = 0;
  #focusY = 0;
  #focusZ = 0;

  #flyX = 0;
  #flyY = 0;
  #flyZ = 0;
  #velocityX = 0;
  #velocityY = 0;
  #velocityZ = 0;
  #speedScale = 1;

  #yawVelocity = 0;
  #pitchVelocity = 0;

  #pointerId = -1;
  #dragButton = -1;
  #lastX = 0;
  #lastY = 0;
  #lastMoveAt = 0;
  #locked = false;
  #disposed = false;

  readonly #keys = new Set<string>();

  readonly #transition: Transition = {
    active: false,
    elapsed: 0,
    duration: 0,
    fromX: 0,
    fromY: 0,
    fromZ: 0,
    toX: 0,
    toY: 0,
    toZ: 0,
    fromDistance: 0,
    toDistance: 0,
  };

  readonly #forward = new THREE.Vector3();
  readonly #right = new THREE.Vector3();
  readonly #up = new THREE.Vector3();
  readonly #eye = new THREE.Vector3();
  readonly #look = new THREE.Vector3();
  readonly #desired = new THREE.Vector3();

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement) {
    this.#camera = camera;
    this.#dom = domElement;
    domElement.style.touchAction = 'none';

    this.setState(
      {
        position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
        target: DEFAULT_CAMERA.target,
        fov: camera.fov,
        mode: DEFAULT_CAMERA.mode,
      },
      true,
    );

    domElement.addEventListener('pointerdown', this.#onPointerDown);
    domElement.addEventListener('pointermove', this.#onPointerMove);
    domElement.addEventListener('pointerup', this.#onPointerUp);
    domElement.addEventListener('pointercancel', this.#onPointerUp);
    domElement.addEventListener('lostpointercapture', this.#onLostCapture);
    domElement.addEventListener('wheel', this.#onWheel, { passive: false });
    domElement.addEventListener('contextmenu', this.#onContextMenu);
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', this.#onKeyDown);
      window.addEventListener('keyup', this.#onKeyUp);
      window.addEventListener('blur', this.#onBlur);
    }
    if (typeof document !== 'undefined') {
      document.addEventListener('pointerlockchange', this.#onPointerLockChange);
    }
  }

  get mode(): CameraMode {
    return this.#mode;
  }

  set mode(next: CameraMode) {
    if (next === this.#mode) return;
    this.#adopt(next);
  }

  update(dt: number): void {
    const step = dt > 0 ? Math.min(dt, 0.25) : 0;

    this.#advanceTransition(step);

    if (this.#mode === 'cinematic') {
      this.#yaw.target += CINEMATIC_RATE * step;
    }
    this.#applyInertia(step);

    this.#pitch.target = clamp(this.#pitch.target, -MAX_PITCH, MAX_PITCH);
    this.#distance.target = clamp(this.#distance.target, MIN_DISTANCE, MAX_DISTANCE);

    this.#yaw.step(step);
    this.#pitch.step(step);
    this.#distance.step(step);
    this.#fov.step(step);
    this.#focus.setTarget(this.#focusX, this.#focusY, this.#focusZ);
    this.#focus.step(step);

    this.#basisFromAngles();

    if (this.#mode === 'fly' || this.#mode === 'first-person') {
      this.#integrateFlight(step);
      this.#flight.setTarget(this.#flyX, this.#flyY, this.#flyZ);
      this.#flight.step(step);
      this.#eye.set(this.#flight.x, this.#flight.y, this.#flight.z);
      this.#look.copy(this.#eye).addScaledVector(this.#forward, LOOK_AHEAD);
    } else {
      const distance = this.#distance.value;
      const cosPitch = Math.cos(this.#pitch.value);
      this.#eye.set(
        this.#focus.x + Math.sin(this.#yaw.value) * cosPitch * distance,
        this.#focus.y + Math.sin(this.#pitch.value) * distance,
        this.#focus.z + Math.cos(this.#yaw.value) * cosPitch * distance,
      );
      this.#look.set(this.#focus.x, this.#focus.y, this.#focus.z);
      // Free flight keeps tracking the eye so a switch into it starts where the
      // orbit left off rather than wherever it was last released.
      this.#flyX = this.#eye.x;
      this.#flyY = this.#eye.y;
      this.#flyZ = this.#eye.z;
      this.#flight.jump(this.#eye.x, this.#eye.y, this.#eye.z);
    }

    this.#applyPose();
  }

  /** Ease the view until the box fits, without ever cutting to it. */
  frame(min: THREE.Vector3, max: THREE.Vector3, duration = DEFAULT_FRAME_DURATION): void {
    const centreX = (min.x + max.x) * 0.5;
    const centreY = (min.y + max.y) * 0.5;
    const centreZ = (min.z + max.z) * 0.5;
    const dx = max.x - min.x;
    const dy = max.y - min.y;
    const dz = max.z - min.z;
    const radius = Math.max(0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz), 0.5);

    // A perspective camera is narrower on whichever axis the aspect ratio
    // squeezes, so the bounding sphere has to be fitted to that one.
    const vertical = THREE.MathUtils.degToRad(this.#camera.fov);
    const horizontal = 2 * Math.atan(Math.tan(vertical * 0.5) * this.#camera.aspect);
    const fitting = Math.max(0.05, Math.min(vertical, horizontal));
    const distance = clamp(
      (radius / Math.sin(fitting * 0.5)) * FRAME_PADDING,
      MIN_DISTANCE,
      MAX_DISTANCE,
    );

    if (duration <= 0) {
      this.#transition.active = false;
      this.#focusX = centreX;
      this.#focusY = centreY;
      this.#focusZ = centreZ;
      this.#distance.target = distance;
      return;
    }

    this.#transition.active = true;
    this.#transition.elapsed = 0;
    this.#transition.duration = duration;
    this.#transition.fromX = this.#focusX;
    this.#transition.fromY = this.#focusY;
    this.#transition.fromZ = this.#focusZ;
    this.#transition.toX = centreX;
    this.#transition.toY = centreY;
    this.#transition.toZ = centreZ;
    this.#transition.fromDistance = this.#distance.target;
    this.#transition.toDistance = distance;
  }

  focusOn(x: number, y: number, z: number, distance?: number): void {
    this.#transition.active = false;
    this.#focusX = x;
    this.#focusY = y;
    this.#focusZ = z;
    if (distance !== undefined) {
      this.#distance.target = clamp(distance, MIN_DISTANCE, MAX_DISTANCE);
    }
  }

  getState(): CameraState {
    return {
      position: {
        x: this.#camera.position.x,
        y: this.#camera.position.y,
        z: this.#camera.position.z,
      },
      target: { x: this.#focusX, y: this.#focusY, z: this.#focusZ },
      fov: this.#camera.fov,
      mode: this.#mode,
    };
  }

  setState(state: CameraState, immediate = false): void {
    this.#mode = state.mode;
    this.#transition.active = false;

    const dx = state.position.x - state.target.x;
    const dy = state.position.y - state.target.y;
    const dz = state.position.z - state.target.z;
    const distance = clamp(
      Math.sqrt(dx * dx + dy * dy + dz * dz),
      MIN_DISTANCE,
      MAX_DISTANCE,
    );
    const pitch = clamp(Math.asin(clamp(dy / distance, -1, 1)), -MAX_PITCH, MAX_PITCH);
    const yaw = Math.atan2(dx, dz);

    this.#focusX = state.target.x;
    this.#focusY = state.target.y;
    this.#focusZ = state.target.z;
    this.#flyX = state.position.x;
    this.#flyY = state.position.y;
    this.#flyZ = state.position.z;
    this.#velocityX = 0;
    this.#velocityY = 0;
    this.#velocityZ = 0;
    this.#yawVelocity = 0;
    this.#pitchVelocity = 0;

    if (immediate) {
      this.#focus.jump(state.target.x, state.target.y, state.target.z);
      this.#flight.jump(state.position.x, state.position.y, state.position.z);
      this.#yaw.jump(yaw);
      this.#pitch.jump(pitch);
      this.#distance.jump(distance);
      this.#fov.jump(state.fov);
      this.#basisFromAngles();
      this.#eye.set(state.position.x, state.position.y, state.position.z);
      this.#look.set(state.target.x, state.target.y, state.target.z);
      this.#applyPose();
      return;
    }

    this.#focus.setTarget(state.target.x, state.target.y, state.target.z);
    this.#yaw.target = yaw;
    this.#pitch.target = pitch;
    this.#distance.target = distance;
    this.#fov.target = state.fov;
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const dom = this.#dom;
    dom.removeEventListener('pointerdown', this.#onPointerDown);
    dom.removeEventListener('pointermove', this.#onPointerMove);
    dom.removeEventListener('pointerup', this.#onPointerUp);
    dom.removeEventListener('pointercancel', this.#onPointerUp);
    dom.removeEventListener('lostpointercapture', this.#onLostCapture);
    dom.removeEventListener('wheel', this.#onWheel);
    dom.removeEventListener('contextmenu', this.#onContextMenu);
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', this.#onKeyDown);
      window.removeEventListener('keyup', this.#onKeyUp);
      window.removeEventListener('blur', this.#onBlur);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('pointerlockchange', this.#onPointerLockChange);
      if (document.pointerLockElement === dom) document.exitPointerLock();
    }
    if (this.#pointerId !== -1 && dom.hasPointerCapture(this.#pointerId)) {
      dom.releasePointerCapture(this.#pointerId);
    }
    this.#pointerId = -1;
    this.#keys.clear();
  }

  #applyPose(): void {
    this.#camera.position.copy(this.#eye);
    this.#camera.up.copy(WORLD_UP);
    this.#camera.lookAt(this.#look);
    const fov = this.#fov.value;
    if (Math.abs(this.#camera.fov - fov) > 1e-4) {
      this.#camera.fov = fov;
      this.#camera.updateProjectionMatrix();
    }
    this.#camera.updateMatrixWorld();
  }

  #basisFromAngles(): void {
    const cosPitch = Math.cos(this.#pitch.value);
    this.#forward.set(
      -Math.sin(this.#yaw.value) * cosPitch,
      -Math.sin(this.#pitch.value),
      -Math.cos(this.#yaw.value) * cosPitch,
    );
    this.#right.copy(this.#forward).cross(WORLD_UP);
    if (this.#right.lengthSq() < 1e-8) this.#right.set(1, 0, 0);
    else this.#right.normalize();
    this.#up.copy(this.#right).cross(this.#forward).normalize();
  }

  #advanceTransition(dt: number): void {
    const transition = this.#transition;
    if (!transition.active) return;
    transition.elapsed += dt;
    const raw = transition.duration > 0 ? transition.elapsed / transition.duration : 1;
    const t = easeInOutQuart(clamp(raw, 0, 1));
    this.#focusX = transition.fromX + (transition.toX - transition.fromX) * t;
    this.#focusY = transition.fromY + (transition.toY - transition.fromY) * t;
    this.#focusZ = transition.fromZ + (transition.toZ - transition.fromZ) * t;
    this.#distance.target =
      transition.fromDistance + (transition.toDistance - transition.fromDistance) * t;
    if (raw >= 1) transition.active = false;
  }

  #applyInertia(dt: number): void {
    if (dt <= 0 || this.#dragButton !== -1 || this.#locked) return;
    if (Math.abs(this.#yawVelocity) > INERTIA_EPSILON) {
      this.#yaw.target += this.#yawVelocity * dt;
      this.#yawVelocity = damp(this.#yawVelocity, 0, INERTIA_LAMBDA, dt);
    } else {
      this.#yawVelocity = 0;
    }
    if (Math.abs(this.#pitchVelocity) > INERTIA_EPSILON) {
      this.#pitch.target += this.#pitchVelocity * dt;
      this.#pitchVelocity = damp(this.#pitchVelocity, 0, INERTIA_LAMBDA, dt);
    } else {
      this.#pitchVelocity = 0;
    }
  }

  #integrateFlight(dt: number): void {
    let ahead = 0;
    let sideways = 0;
    let vertical = 0;
    if (this.#keys.has('w')) ahead += 1;
    if (this.#keys.has('s')) ahead -= 1;
    if (this.#keys.has('d')) sideways += 1;
    if (this.#keys.has('a')) sideways -= 1;
    if (this.#keys.has('e')) vertical += 1;
    if (this.#keys.has('q')) vertical -= 1;

    this.#desired.set(0, 0, 0);
    if (ahead !== 0) {
      if (this.#mode === 'first-person') {
        // Walking: the heading is the forward vector flattened onto the ground,
        // so looking at the floor does not drive the camera into it.
        const fx = this.#forward.x;
        const fz = this.#forward.z;
        const length = Math.hypot(fx, fz);
        if (length > 1e-6) this.#desired.set((fx / length) * ahead, 0, (fz / length) * ahead);
      } else {
        this.#desired.addScaledVector(this.#forward, ahead);
      }
    }
    this.#desired.addScaledVector(this.#right, sideways);
    this.#desired.y += vertical;

    if (this.#desired.lengthSq() > 1e-8) {
      this.#desired.normalize();
      const boost = this.#keys.has('shift') ? FLY_BOOST : 1;
      this.#desired.multiplyScalar(FLY_BASE_SPEED * this.#speedScale * boost);
    }

    this.#velocityX = damp(this.#velocityX, this.#desired.x, FLY_ACCEL_LAMBDA, dt);
    this.#velocityY = damp(this.#velocityY, this.#desired.y, FLY_ACCEL_LAMBDA, dt);
    this.#velocityZ = damp(this.#velocityZ, this.#desired.z, FLY_ACCEL_LAMBDA, dt);
    this.#flyX += this.#velocityX * dt;
    this.#flyY += this.#velocityY * dt;
    this.#flyZ += this.#velocityZ * dt;
  }

  /**
   * Re-derive the incoming mode's parameters from the pose the outgoing mode is
   * currently producing. Nothing moves at the switch.
   */
  #adopt(next: CameraMode): void {
    this.#basisFromAngles();
    const eye = this.#camera.position;

    if (next === 'fly' || next === 'first-person') {
      this.#flyX = eye.x;
      this.#flyY = eye.y;
      this.#flyZ = eye.z;
      this.#flight.jump(eye.x, eye.y, eye.z);
      this.#velocityX = 0;
      this.#velocityY = 0;
      this.#velocityZ = 0;
      this.#speedScale = clamp(this.#distance.value / 40, FLY_SPEED_MIN, FLY_SPEED_MAX);
    } else {
      const distance = clamp(this.#distance.value, MIN_DISTANCE, MAX_DISTANCE);
      this.#focusX = eye.x + this.#forward.x * distance;
      this.#focusY = eye.y + this.#forward.y * distance;
      this.#focusZ = eye.z + this.#forward.z * distance;
      this.#focus.jump(this.#focusX, this.#focusY, this.#focusZ);
      this.#distance.jump(distance);
    }

    this.#transition.active = false;
    this.#mode = next;
  }

  #rotate(dx: number, dy: number, dt: number): void {
    const yawDelta = -dx * ROTATE_SPEED;
    const pitchDelta = dy * ROTATE_SPEED;
    this.#yaw.target += yawDelta;
    this.#pitch.target = clamp(this.#pitch.target + pitchDelta, -MAX_PITCH, MAX_PITCH);
    const interval = Math.max(dt, 1 / 240);
    this.#yawVelocity = this.#yawVelocity * 0.55 + (yawDelta / interval) * 0.45;
    this.#pitchVelocity = this.#pitchVelocity * 0.55 + (pitchDelta / interval) * 0.45;
  }

  #lookDelta(dx: number, dy: number): void {
    this.#yaw.target -= dx * LOOK_SPEED;
    this.#pitch.target = clamp(this.#pitch.target + dy * LOOK_SPEED, -MAX_PITCH, MAX_PITCH);
  }

  #pan(dx: number, dy: number): void {
    this.#basisFromAngles();
    const scale = this.#distance.value * PAN_SPEED;
    this.#focusX += (-this.#right.x * dx + this.#up.x * dy) * scale;
    this.#focusY += (-this.#right.y * dx + this.#up.y * dy) * scale;
    this.#focusZ += (-this.#right.z * dx + this.#up.z * dy) * scale;
    if (this.#mode === 'fly' || this.#mode === 'first-person') {
      this.#flyX += (-this.#right.x * dx + this.#up.x * dy) * scale;
      this.#flyY += (-this.#right.y * dx + this.#up.y * dy) * scale;
      this.#flyZ += (-this.#right.z * dx + this.#up.z * dy) * scale;
    }
    this.#transition.active = false;
  }

  #now(): number {
    return typeof performance !== 'undefined' ? performance.now() / 1000 : Date.now() / 1000;
  }

  #onPointerDown = (event: PointerEvent): void => {
    if (this.#pointerId !== -1) return;
    this.#pointerId = event.pointerId;
    this.#dragButton = event.button;
    this.#lastX = event.clientX;
    this.#lastY = event.clientY;
    this.#lastMoveAt = this.#now();
    this.#yawVelocity = 0;
    this.#pitchVelocity = 0;
    this.#dom.setPointerCapture(event.pointerId);
    if (event.button === 1) event.preventDefault();
    if (event.button === 0 && (this.#mode === 'fly' || this.#mode === 'first-person')) {
      this.#requestLock();
    }
  };

  #onPointerMove = (event: PointerEvent): void => {
    if (this.#locked) {
      this.#lookDelta(event.movementX, event.movementY);
      return;
    }
    if (event.pointerId !== this.#pointerId) return;
    const dx = event.clientX - this.#lastX;
    const dy = event.clientY - this.#lastY;
    this.#lastX = event.clientX;
    this.#lastY = event.clientY;
    const now = this.#now();
    const dt = now - this.#lastMoveAt;
    this.#lastMoveAt = now;
    if (dx === 0 && dy === 0) return;

    const panning = this.#dragButton === 1 || this.#dragButton === 2 || event.shiftKey;
    if (panning) this.#pan(dx, dy);
    else this.#rotate(dx, dy, dt);
  };

  #onPointerUp = (event: PointerEvent): void => {
    if (event.pointerId !== this.#pointerId) return;
    if (this.#dom.hasPointerCapture(event.pointerId)) {
      this.#dom.releasePointerCapture(event.pointerId);
    }
    this.#pointerId = -1;
    this.#dragButton = -1;
  };

  #onLostCapture = (): void => {
    this.#pointerId = -1;
    this.#dragButton = -1;
  };

  #onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 100 : 1;
    const delta = event.deltaY * unit;
    if (this.#mode === 'fly' || this.#mode === 'first-person') {
      this.#speedScale = clamp(
        this.#speedScale * Math.exp(-delta * ZOOM_RATE),
        FLY_SPEED_MIN,
        FLY_SPEED_MAX,
      );
      return;
    }
    this.#transition.active = false;
    this.#distance.target = clamp(
      this.#distance.target * Math.exp(delta * ZOOM_RATE),
      MIN_DISTANCE,
      MAX_DISTANCE,
    );
  };

  #onContextMenu = (event: Event): void => {
    event.preventDefault();
  };

  #onKeyDown = (event: KeyboardEvent): void => {
    if (isEditableTarget(event.target)) return;
    this.#keys.add(normaliseKey(event));
  };

  #onKeyUp = (event: KeyboardEvent): void => {
    this.#keys.delete(normaliseKey(event));
  };

  #onBlur = (): void => {
    this.#keys.clear();
  };

  #onPointerLockChange = (): void => {
    this.#locked = typeof document !== 'undefined' && document.pointerLockElement === this.#dom;
  };

  #requestLock(): void {
    const result: unknown = this.#dom.requestPointerLock();
    if (result instanceof Promise) result.catch(() => undefined);
  }
}

function normaliseKey(event: KeyboardEvent): string {
  if (event.key === 'Shift') return 'shift';
  return event.key.toLowerCase();
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
}
