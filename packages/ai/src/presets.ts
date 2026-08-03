import { IZHIKEVICH_PRESETS } from '@neuroforge/shared';
import type { NeuronModelKind, NeuronParams, PopulationLayout } from '@neuroforge/shared';
import { clamp } from './coerce';
import type { AnalyticLayout, FiringPattern } from './lexicon';

/** Firing patterns only the Izhikevich model can express by parameter choice. */
export const IZHIKEVICH_ONLY_PATTERNS: readonly FiringPattern[] = [
  'intrinsically-bursting',
  'chattering',
  'resonator',
  'thalamo-cortical',
];

export function izhikevichParams(pattern: FiringPattern): Partial<NeuronParams> {
  return { kind: 'izhikevich', ...IZHIKEVICH_PRESETS[pattern] };
}

/**
 * A parameter patch that changes nothing but names the model it applies to. The
 * discriminant has to be a literal for the union to accept it, which is why this
 * cannot be written as `{ kind }`.
 */
export function kindOnlyParams(kind: NeuronModelKind): Partial<NeuronParams> {
  switch (kind) {
    case 'lif':
      return { kind: 'lif' };
    case 'izhikevich':
      return { kind: 'izhikevich' };
    case 'hodgkin-huxley':
      return { kind: 'hodgkin-huxley' };
    case 'adex':
      return { kind: 'adex' };
    case 'morris-lecar':
      return { kind: 'morris-lecar' };
  }
}

/**
 * Parameters that turn a cell of any model into a fast perisomatic interneuron
 * (short membrane time constant, little adaptation) or a slower low-threshold
 * one. Izhikevich uses the published presets; the others are shifted from the
 * defaults in `@neuroforge/shared`.
 */
export function interneuronParams(kind: NeuronModelKind, fast: boolean): Partial<NeuronParams> {
  switch (kind) {
    case 'izhikevich':
      return izhikevichParams(fast ? 'fast-spiking' : 'low-threshold-spiking');
    case 'lif':
      return {
        kind: 'lif',
        cm: fast ? 100 : 180,
        gL: fast ? 12 : 10,
        eL: -70,
        vThresh: fast ? -52 : -50,
        vReset: fast ? -62 : -58,
        tRefract: fast ? 1 : 2,
      };
    case 'adex':
      return {
        kind: 'adex',
        cm: fast ? 150 : 240,
        gL: fast ? 20 : 26,
        a: fast ? 2 : 4,
        b: fast ? 10 : 60,
        tauW: fast ? 30 : 120,
        tRefract: fast ? 1 : 2,
      };
    case 'hodgkin-huxley':
      return { kind: 'hodgkin-huxley', q10: fast ? 1.4 : 1 };
    case 'morris-lecar':
      return { kind: 'morris-lecar', phi: fast ? 0.08 : 0.04 };
  }
}

/**
 * Bias current, in pA, that holds a cell of this model near `hz` when it is also
 * receiving recurrent inhibition. Derived from each model's rheobase at the
 * defaults in `@neuroforge/shared`: LIF needs gL*(vThresh-eL) = 200 pA, AdEx
 * needs gL*(vT-eL) ≈ 600 pA plus its adaptation current, Izhikevich scales input
 * by iScale = 0.04, and Morris-Lecar's capacitance is only 20 pF.
 */
const DRIVE_TABLE: Record<NeuronModelKind, { base: number; perHz: number; max: number }> = {
  lif: { base: 210, perHz: 2.6, max: 1400 },
  izhikevich: { base: 90, perHz: 3.4, max: 900 },
  adex: { base: 620, perHz: 12, max: 3200 },
  'hodgkin-huxley': { base: 600, perHz: 10, max: 3000 },
  'morris-lecar': { base: 45, perHz: 1.4, max: 400 },
};

export function sustainedDrive(kind: NeuronModelKind, hz: number): number {
  const row = DRIVE_TABLE[kind];
  return Math.round(clamp(row.base + row.perHz * hz, 0, row.max));
}

/** Radius of the sphere that encloses a layout, used to space populations apart. */
export function layoutRadius(layout: PopulationLayout): number {
  switch (layout.kind) {
    case 'grid': {
      const x = (layout.columns - 1) * layout.spacing;
      const y = (layout.rows - 1) * layout.spacing;
      const z = (layout.layers - 1) * layout.spacing;
      return 0.5 * Math.sqrt(x * x + y * y + z * z);
    }
    case 'sphere':
      return layout.radius * (1 + layout.jitter);
    case 'disc':
      return Math.max(layout.radius, layout.thickness * 0.5);
    case 'column':
      return Math.max(layout.radius, layout.height * 0.5);
    case 'explicit': {
      let far = 0;
      for (const p of layout.positions) {
        far = Math.max(far, Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z));
      }
      return far;
    }
  }
}

/**
 * A layout sized so that neuron density stays roughly constant as the population
 * grows: volumetric arrangements scale with the cube root of the count, laminar
 * ones with the square root.
 */
export function layoutFor(kind: AnalyticLayout, size: number, seed: number): PopulationLayout {
  const n = Math.max(1, size);
  switch (kind) {
    case 'sphere':
      return {
        kind: 'sphere',
        radius: Math.round(clamp(2.6 * Math.cbrt(n), 4, 60) * 10) / 10,
        jitter: 0.35,
        seed,
      };
    case 'disc': {
      const radius = Math.round(clamp(1.35 * Math.sqrt(n), 5, 90) * 10) / 10;
      return {
        kind: 'disc',
        radius,
        thickness: Math.round(Math.max(1.5, radius * 0.12) * 10) / 10,
        seed,
      };
    }
    case 'column': {
      const radius = Math.round(clamp(1.05 * Math.sqrt(n), 4, 40) * 10) / 10;
      return { kind: 'column', radius, height: Math.round(radius * 3.2 * 10) / 10, seed };
    }
    case 'grid': {
      const layers = Math.max(1, Math.round(Math.cbrt(n) / 2));
      const columns = Math.max(1, Math.ceil(Math.sqrt(n / layers)));
      const rows = Math.max(1, Math.ceil(n / (columns * layers)));
      return { kind: 'grid', columns, rows, layers, spacing: 3.2 };
    }
  }
}
