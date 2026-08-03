import * as THREE from 'three';
import { GRID_FRAGMENT_GLSL, GRID_VERTEX_GLSL } from '@neuroforge/shaders';
import { COLORS, DEFAULT_RENDER_SETTINGS } from '@neuroforge/shared';
import type { RenderSettings } from '@neuroforge/shared';
import { linearColor } from './palette';

/**
 * Infinite analytic ground grid.
 *
 * There is no grid geometry: one clip-space quad carries a ray per fragment, the
 * ray is intersected with the ground plane and the lines are evaluated from the
 * hit point using `fwidth`, so a line is the same width in pixels at every zoom
 * and no tile ever repeats.
 *
 * Cell size follows the camera by decades. Only the coarse decade's opacity is
 * ramped across a switch: the fine lines are already below the shader's own
 * pixel-density cut-off when it happens, so the change lands on lines nobody can
 * see yet.
 */

const MINOR_OPACITY = 0.3;
const MAJOR_OPACITY = 0.62;
const AXIS_OPACITY = 0.9;

/** Cell size is chosen to sit this many world units per pixel-ish decade. */
const CELL_ANCHOR = 0.03;

const MIN_CELL = 1e-4;
const MAX_CELL = 1e7;

export class InfiniteGrid extends THREE.Mesh {
  #material: THREE.ShaderMaterial;

  constructor() {
    const geometry = new THREE.PlaneGeometry(2, 2);
    const material = new THREE.ShaderMaterial({
      vertexShader: GRID_VERTEX_GLSL,
      fragmentShader: GRID_FRAGMENT_GLSL,
      transparent: true,
      // The fragment stage writes gl_FragDepth so the plane occludes correctly;
      // it must not also write to the depth buffer, or it would swallow the
      // transparent passes drawn after it.
      depthWrite: false,
      depthTest: true,
      side: THREE.DoubleSide,
      uniforms: {
        uInverseProjectionMatrix: { value: new THREE.Matrix4() },
        uInverseViewMatrix: { value: new THREE.Matrix4() },
        uGridColor: { value: linearColor('#2A3644') },
        uMajorColor: { value: linearColor('#43586E') },
        uAxisXColor: { value: linearColor(COLORS.danger) },
        uAxisZColor: { value: linearColor(COLORS.accent) },
        uFogColor: { value: linearColor(COLORS.background) },
        uHeight: { value: 0 },
        uCellSize: { value: 1 },
        uMajorEvery: { value: 10 },
        uLineWidth: { value: 0.6 },
        uMajorWidth: { value: 0.9 },
        uAxisWidth: { value: 1.1 },
        uMinorOpacity: { value: MINOR_OPACITY },
        uMajorOpacity: { value: MAJOR_OPACITY },
        uAxisOpacity: { value: AXIS_OPACITY },
        uFadeStart: { value: 40 },
        uFadeEnd: { value: 400 },
        uOpacity: { value: 1 },
        uFogDensity: { value: DEFAULT_RENDER_SETTINGS.fogDensity },
      },
    });

    super(geometry, material);
    this.name = 'InfiniteGrid';
    this.#material = material;
    this.frustumCulled = false;
    this.renderOrder = -10;
  }

  update(camera: THREE.Camera, settings: RenderSettings): void {
    this.visible = settings.gridVisible;
    if (!settings.gridVisible) return;

    const uniforms = this.#material.uniforms;
    // `matrixWorld` is the inverse view matrix by definition, and three keeps
    // `projectionMatrixInverse` current, so neither needs inverting here.
    (uniforms.uInverseProjectionMatrix.value as THREE.Matrix4).copy(camera.projectionMatrixInverse);
    (uniforms.uInverseViewMatrix.value as THREE.Matrix4).copy(camera.matrixWorld);

    // Height above the plane is the only scale the grid has to track: it is what
    // decides how many pixels a cell covers.
    const height = Math.max(Math.abs(camera.position.y), 1e-3);
    const decade = Math.log10(Math.max(height * CELL_ANCHOR, MIN_CELL));
    const step = Math.floor(decade);
    const fraction = decade - step;
    const cell = Math.min(MAX_CELL, Math.max(MIN_CELL, Math.pow(10, step)));

    uniforms.uCellSize.value = cell;
    uniforms.uMinorOpacity.value = MINOR_OPACITY;
    // Approaching a decade switch, the coarse lines relax toward the weight they
    // will carry as fine lines on the other side of it.
    uniforms.uMajorOpacity.value =
      MAJOR_OPACITY + (MINOR_OPACITY - MAJOR_OPACITY) * smoothstep(0.65, 1, fraction);

    const fade = 0.35 + Math.max(0, Math.min(1, settings.gridFade)) * 1.6;
    const reach = height * 26 * fade + cell * 20;
    uniforms.uFadeStart.value = reach * 0.12;
    uniforms.uFadeEnd.value = reach;
    uniforms.uFogDensity.value = settings.fogDensity;
  }

  dispose(): void {
    this.geometry.dispose();
    this.#material.dispose();
  }
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
