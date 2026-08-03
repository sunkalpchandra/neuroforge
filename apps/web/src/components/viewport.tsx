'use client';

import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import {
  Bloom,
  ChromaticAberration,
  DepthOfField,
  EffectComposer,
  N8AO,
  Noise,
  SMAA,
  ToneMapping,
  Vignette,
} from '@react-three/postprocessing';
import { BlendFunction, ToneMappingMode } from 'postprocessing';
import * as THREE from 'three';
import {
  AxonField,
  CameraRig,
  GlyphLibrary,
  InfiniteGrid,
  NeuronField,
  SelectionOverlay,
  SpikeParticles,
} from '@neuroforge/renderer';
import { createIntegrator, requestComputeDevice } from '@neuroforge/simulation';
import { COLORS, DEFAULT_RENDER_SETTINGS } from '@neuroforge/shared';
import type { RenderSettings } from '@neuroforge/shared';

import { consumeCameraFrame, getEngine, getProbes, publishStats } from '@/lib/runtime';

export interface ViewportProps {
  render?: RenderSettings;
  onPick?: (slot: number) => void;
}

/**
 * The simulation and render loop.
 *
 * Runs at a negative priority so it executes before R3F's own render pass:
 * the fields must be updated from the freshly integrated buffers within the
 * same frame, or every visual lags the simulation by one frame.
 */
function Loop({
  fields,
  settings,
}: {
  fields: {
    neurons: NeuronField;
    axons: AxonField;
    particles: SpikeParticles;
    grid: InfiniteGrid;
    selection: SelectionOverlay;
    rig: CameraRig;
  };
  settings: RenderSettings;
}) {
  const { camera, gl, scene } = useThree();
  const probeClock = useRef(0);

  useFrame((_, delta) => {
    const frameStart = performance.now();
    const engine = getEngine();
    const buffers = engine.buffers;

    // Clamp the frame delta. A backgrounded tab resumes with a delta of many
    // seconds, which would otherwise ask the integrator for tens of thousands
    // of substeps in one frame.
    const dt = Math.min(delta, 1 / 20);

    engine.advance(dt);

    // Framing is requested from outside the scene graph; the rig is only
    // reachable here, so the request is picked up at the top of the frame.
    const framing = consumeCameraFrame();
    if (framing) {
      fields.rig.frame(
        new THREE.Vector3(framing.min[0], framing.min[1], framing.min[2]),
        new THREE.Vector3(framing.max[0], framing.max[1], framing.max[2]),
      );
    }

    fields.rig.update(dt);
    fields.grid.update(camera, settings);
    fields.neurons.update(buffers, dt, settings);
    fields.axons.update(buffers, dt, settings);
    fields.selection.update(buffers, dt);
    if (settings.showParticles) {
      fields.particles.emitFromSpikes(buffers);
      fields.particles.update(dt, settings);
    }

    // Probe traces are sampled at a fixed 200 Hz rather than per frame, so a
    // trace means the same thing regardless of the display's refresh rate.
    probeClock.current += dt;
    if (probeClock.current >= 0.005) {
      probeClock.current = 0;
      getProbes().sample(buffers);
    }

    gl.render(scene, camera);
    engine.recordFrame(performance.now() - frameStart);
    publishStats(frameStart);
  }, 1);

  return null;
}

function Scene({ settings, onPick }: { settings: RenderSettings; onPick?: (slot: number) => void }) {
  const { camera, gl } = useThree();

  // Every renderer object owns GPU resources and must outlive re-renders.
  const fields = useMemo(() => {
    const library = new GlyphLibrary(24);
    return {
      library,
      neurons: new NeuronField(library),
      axons: new AxonField(),
      particles: new SpikeParticles(65536),
      grid: new InfiniteGrid(),
      selection: new SelectionOverlay(),
      rig: new CameraRig(camera as THREE.PerspectiveCamera, gl.domElement),
    };
  }, [camera, gl]);

  useEffect(() => {
    const engine = getEngine();
    fields.neurons.rebuild(engine.buffers);
    fields.axons.rebuild(engine.buffers);
  }, [fields]);

  useEffect(() => {
    return () => {
      fields.neurons.dispose();
      fields.axons.dispose();
      fields.particles.dispose();
      fields.grid.dispose();
      fields.selection.dispose();
      fields.rig.dispose();
      fields.library.dispose();
    };
  }, [fields]);

  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const pointer = useMemo(() => new THREE.Vector2(), []);

  useEffect(() => {
    if (!onPick) return;
    const element = gl.domElement;
    const handle = (event: PointerEvent) => {
      const rect = element.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      onPick(fields.neurons.raycastSlot(raycaster));
    };
    element.addEventListener('pointerdown', handle);
    return () => element.removeEventListener('pointerdown', handle);
  }, [camera, fields, gl, onPick, pointer, raycaster]);

  return (
    <>
      {/* Key light warm from above, fill cool from the opposite side: the
          contrast is what gives the glyphs readable form against near-black. */}
      <ambientLight intensity={0.22} />
      <directionalLight position={[40, 80, 30]} intensity={1.4} color="#ffffff" castShadow />
      <directionalLight position={[-50, -20, -40]} intensity={0.5} color={COLORS.secondary} />
      <hemisphereLight args={[COLORS.accent, COLORS.background, 0.35]} />

      <fogExp2 attach="fog" args={[COLORS.scene, settings.fogDensity]} />

      {settings.gridVisible ? <primitive object={fields.grid} /> : null}
      <primitive object={fields.neurons} />
      {settings.showAxons ? <primitive object={fields.axons} /> : null}
      {settings.showParticles ? <primitive object={fields.particles} /> : null}
      <primitive object={fields.selection} />

      <Loop fields={fields} settings={settings} />
    </>
  );
}

function Post({ settings }: { settings: RenderSettings }) {
  const aberration = useMemo(
    () => new THREE.Vector2(settings.chromaticAberration, settings.chromaticAberration),
    [settings.chromaticAberration],
  );

  return (
    <EffectComposer multisampling={0} enableNormalPass>
      {settings.ambientOcclusion ? (
        <N8AO intensity={settings.aoIntensity} aoRadius={6} distanceFalloff={1.2} quality="medium" />
      ) : (
        <></>
      )}
      <Bloom
        intensity={settings.bloomIntensity}
        luminanceThreshold={settings.bloomThreshold}
        luminanceSmoothing={settings.bloomRadius}
        mipmapBlur
      />
      {settings.depthOfField ? (
        <DepthOfField
          focusDistance={settings.focusDistance}
          focalLength={settings.focalLength}
          bokehScale={settings.bokehScale}
        />
      ) : (
        <></>
      )}
      <ChromaticAberration offset={aberration} radialModulation modulationOffset={0.3} />
      <Vignette eskil={false} offset={0.22} darkness={settings.vignette} />
      <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
      <SMAA />
      {/* Faint grain keeps the near-black background from banding into visible
          steps on 8-bit displays. */}
      <Noise premultiply blendFunction={BlendFunction.SOFT_LIGHT} opacity={0.035} />
    </EffectComposer>
  );
}

/**
 * The 3D workspace.
 *
 * `frameloop="always"` is deliberate: the scene animates continuously even when
 * the document is idle, because neurons are always integrating.
 */
export function Viewport({ render = DEFAULT_RENDER_SETTINGS, onPick }: ViewportProps) {
  useEffect(() => {
    // Compute and rendering deliberately use different backends, so the compute
    // device is acquired independently of the WebGL context R3F creates. If
    // WebGPU is missing or its pipelines fail to build, createIntegrator falls
    // back down the chain and the app keeps running on the CPU reference.
    let cancelled = false;
    void (async () => {
      const engine = getEngine();
      const device = await requestComputeDevice();
      if (cancelled) return;
      const integrator = await createIntegrator(engine.settings.backend, device);
      if (cancelled) {
        integrator.dispose();
        return;
      }
      await engine.setIntegrator(integrator);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Canvas
      className="absolute inset-0"
      dpr={[1, 2]}
      frameloop="always"
      shadows
      gl={{
        antialias: false,
        alpha: false,
        powerPreference: 'high-performance',
        stencil: false,
        depth: true,
        toneMapping: THREE.ACESFilmicToneMapping,
        outputColorSpace: THREE.SRGBColorSpace,
      }}
      camera={{ position: [0, 34, 96], fov: 42, near: 0.1, far: 4000 }}
      onCreated={({ gl, scene }) => {
        gl.setClearColor(COLORS.scene, 1);
        gl.toneMappingExposure = render.exposure;
        scene.background = new THREE.Color(COLORS.scene);
      }}
    >
      <Scene settings={render} onPick={onPick} />
      <Post settings={render} />
    </Canvas>
  );
}
