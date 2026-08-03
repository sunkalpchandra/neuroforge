import type {
  MorphologyArchetype,
  NeuronModelKind,
  NeuronPolarity,
  ReceptorKind,
} from '@neuroforge/shared';

/** Layout families the offline planner picks between. `explicit` is never generated. */
export type AnalyticLayout = 'grid' | 'sphere' | 'disc' | 'column';

/** Keys into `IZHIKEVICH_PRESETS` in `@neuroforge/shared`. */
export type FiringPattern =
  | 'regular-spiking'
  | 'intrinsically-bursting'
  | 'chattering'
  | 'fast-spiking'
  | 'low-threshold-spiking'
  | 'thalamo-cortical'
  | 'resonator';

export const POLARITY_WORDS: Readonly<Record<string, NeuronPolarity>> = {
  excitatory: 'excitatory',
  excitory: 'excitatory',
  exc: 'excitatory',
  glutamatergic: 'excitatory',
  glutamate: 'excitatory',
  principal: 'excitatory',
  pyramidal: 'excitatory',
  pyramid: 'excitatory',
  mossy: 'excitatory',
  inhibitory: 'inhibitory',
  inhibatory: 'inhibitory',
  inh: 'inhibitory',
  gabaergic: 'inhibitory',
  interneuron: 'inhibitory',
  interneurons: 'inhibitory',
  basket: 'inhibitory',
  chandelier: 'inhibitory',
  purkinje: 'inhibitory',
};

export const ARCHETYPE_WORDS: Readonly<Record<string, MorphologyArchetype>> = {
  pyramidal: 'pyramidal',
  pyramid: 'pyramidal',
  basket: 'basket',
  chandelier: 'basket',
  granule: 'granule',
  granular: 'granule',
  purkinje: 'purkinje',
  stellate: 'stellate',
  'medium-spiny': 'stellate',
  motor: 'motor',
  motoneuron: 'motor',
  motoneurons: 'motor',
  bipolar: 'bipolar',
};

export const MODEL_WORDS: Readonly<Record<string, NeuronModelKind>> = {
  lif: 'lif',
  iaf: 'lif',
  'integrate-and-fire': 'lif',
  leaky: 'lif',
  izhikevich: 'izhikevich',
  izh: 'izhikevich',
  'hodgkin-huxley': 'hodgkin-huxley',
  hodgkin: 'hodgkin-huxley',
  huxley: 'hodgkin-huxley',
  hh: 'hodgkin-huxley',
  adex: 'adex',
  aeif: 'adex',
  'adaptive-exponential': 'adex',
  'morris-lecar': 'morris-lecar',
  morris: 'morris-lecar',
  lecar: 'morris-lecar',
};

export const FIRING_WORDS: Readonly<Record<string, FiringPattern>> = {
  'fast-spiking': 'fast-spiking',
  fs: 'fast-spiking',
  'regular-spiking': 'regular-spiking',
  rs: 'regular-spiking',
  bursting: 'intrinsically-bursting',
  burster: 'intrinsically-bursting',
  bursty: 'intrinsically-bursting',
  'intrinsically-bursting': 'intrinsically-bursting',
  chattering: 'chattering',
  'low-threshold': 'low-threshold-spiking',
  lts: 'low-threshold-spiking',
  'thalamo-cortical': 'thalamo-cortical',
  relay: 'thalamo-cortical',
  resonator: 'resonator',
  resonant: 'resonator',
};

export const RECEPTOR_WORDS: Readonly<Record<string, ReceptorKind>> = {
  ampa: 'ampa',
  glutamatergic: 'ampa',
  nmda: 'nmda',
  gabaa: 'gabaa',
  'gaba-a': 'gabaa',
  gabab: 'gabab',
  'gaba-b': 'gabab',
  'gap-junction': 'gap',
  electrical: 'gap',
};

/**
 * Words that name a spatial arrangement, longest-lived first so `columnar` is
 * read the same way as `column`. A match overrides the region's default layout.
 */
export const LAYOUT_WORDS: readonly (readonly [string, AnalyticLayout])[] = [
  ['lattice', 'grid'],
  ['grid', 'grid'],
  ['spherical', 'sphere'],
  ['sphere', 'sphere'],
  ['ball', 'sphere'],
  ['laminar', 'disc'],
  ['lamina', 'disc'],
  ['sheet', 'disc'],
  ['disc', 'disc'],
  ['disk', 'disc'],
  ['columnar', 'column'],
  ['column', 'column'],
  ['cylinder', 'column'],
  ['barrel', 'column'],
];

/** Nouns that end a population phrase. */
export const HEAD_NOUNS: ReadonlySet<string> = new Set([
  'neuron',
  'neurons',
  'neurone',
  'neurones',
  'cell',
  'cells',
  'interneuron',
  'interneurons',
  'unit',
  'units',
  'population',
  'populations',
  'pyramids',
  'motoneuron',
  'motoneurons',
]);

/** Words that terminate the backwards walk collecting a population's modifiers. */
export const PHRASE_BOUNDARIES: ReadonlySet<string> = new Set([
  'a', 'add', 'adding', 'also', 'an', 'and', 'as', 'at', 'between', 'build', 'circuit', 'connect',
  'create', 'delete', 'drive', 'each', 'for', 'from', 'generate', 'give', 'in', 'include',
  'including', 'into', 'it', 'layer', 'link', 'make', 'microcircuit', 'model', 'network', 'of',
  'onto', 'place', 'plus', 'population', 'populations', 'put', 'region', 'remove', 'set',
  'simulation', 'stimulate', 'that', 'the', 'them', 'then', 'this', 'to', 'use', 'using', 'which',
  'wire', 'with', 'wiring',
]);

/** Verbs that mean "take neurons away", which this planner cannot express. */
export const REMOVAL_VERBS: ReadonlySet<string> = new Set([
  'remove',
  'delete',
  'drop',
  'kill',
  'prune',
  'detach',
]);

export interface RegionProfile {
  /** Prefix applied to generated population names. */
  readonly label: string;
  readonly excitatoryArchetype: MorphologyArchetype;
  readonly inhibitoryArchetype: MorphologyArchetype;
  readonly excitatoryPattern: FiringPattern;
  readonly inhibitoryPattern: FiringPattern;
  readonly layout: AnalyticLayout;
  /** Polarity assumed for a population the prompt does not qualify. */
  readonly defaultPolarity: NeuronPolarity;
  /** Whether this structure is recurrent enough to wire itself by default. */
  readonly recurrent: boolean;
}

const HIPPOCAMPAL_CA3: RegionProfile = {
  label: 'CA3',
  excitatoryArchetype: 'pyramidal',
  inhibitoryArchetype: 'basket',
  excitatoryPattern: 'regular-spiking',
  inhibitoryPattern: 'fast-spiking',
  layout: 'sphere',
  defaultPolarity: 'excitatory',
  recurrent: true,
};

const HIPPOCAMPAL_CA1: RegionProfile = {
  ...HIPPOCAMPAL_CA3,
  label: 'CA1',
  layout: 'disc',
  recurrent: false,
};

const DENTATE: RegionProfile = {
  label: 'Dentate',
  excitatoryArchetype: 'granule',
  inhibitoryArchetype: 'basket',
  excitatoryPattern: 'regular-spiking',
  inhibitoryPattern: 'fast-spiking',
  layout: 'disc',
  defaultPolarity: 'excitatory',
  recurrent: false,
};

const CORTEX: RegionProfile = {
  label: 'Cortex',
  excitatoryArchetype: 'pyramidal',
  inhibitoryArchetype: 'basket',
  excitatoryPattern: 'regular-spiking',
  inhibitoryPattern: 'fast-spiking',
  layout: 'column',
  defaultPolarity: 'excitatory',
  recurrent: true,
};

const CEREBELLUM: RegionProfile = {
  label: 'Cerebellum',
  excitatoryArchetype: 'granule',
  inhibitoryArchetype: 'purkinje',
  excitatoryPattern: 'regular-spiking',
  inhibitoryPattern: 'fast-spiking',
  layout: 'disc',
  defaultPolarity: 'excitatory',
  recurrent: false,
};

const THALAMUS: RegionProfile = {
  label: 'Thalamus',
  excitatoryArchetype: 'stellate',
  inhibitoryArchetype: 'basket',
  excitatoryPattern: 'thalamo-cortical',
  inhibitoryPattern: 'low-threshold-spiking',
  layout: 'sphere',
  defaultPolarity: 'excitatory',
  recurrent: true,
};

const STRIATUM: RegionProfile = {
  label: 'Striatum',
  excitatoryArchetype: 'pyramidal',
  inhibitoryArchetype: 'stellate',
  excitatoryPattern: 'regular-spiking',
  inhibitoryPattern: 'low-threshold-spiking',
  layout: 'sphere',
  defaultPolarity: 'inhibitory',
  recurrent: false,
};

export const GENERIC_REGION: RegionProfile = {
  label: '',
  excitatoryArchetype: 'pyramidal',
  inhibitoryArchetype: 'basket',
  excitatoryPattern: 'regular-spiking',
  inhibitoryPattern: 'fast-spiking',
  layout: 'sphere',
  defaultPolarity: 'excitatory',
  recurrent: false,
};

/** Region words, longest-lived first: `ca3` beats the bare `hippocampus`. */
export const REGION_WORDS: readonly (readonly [string, RegionProfile])[] = [
  ['ca3', HIPPOCAMPAL_CA3],
  ['ca1', HIPPOCAMPAL_CA1],
  ['ca2', HIPPOCAMPAL_CA1],
  ['dentate', DENTATE],
  ['dentate-gyrus', DENTATE],
  ['hippocampal', HIPPOCAMPAL_CA1],
  ['hippocampus', HIPPOCAMPAL_CA1],
  ['cortical', CORTEX],
  ['cortex', CORTEX],
  ['neocortex', CORTEX],
  ['neocortical', CORTEX],
  ['cerebellar', CEREBELLUM],
  ['cerebellum', CEREBELLUM],
  ['thalamic', THALAMUS],
  ['thalamus', THALAMUS],
  ['striatal', STRIATUM],
  ['striatum', STRIATUM],
];

/**
 * Region word to the prefix that generated population names carry, lower-cased,
 * so "the thalamic cells" can be matched against "Thalamus Stellate". Regions
 * with no label are left out: they prefix nothing.
 */
export const REGION_LABELS: Readonly<Record<string, string>> = Object.fromEntries(
  REGION_WORDS.filter(([, profile]) => profile.label !== '').map(([word, profile]) => [
    word,
    profile.label.toLowerCase(),
  ]),
);

export interface RhythmBand {
  readonly name: string;
  readonly minHz: number;
  readonly maxHz: number;
  /** Frequency used when the band is named without a number. */
  readonly targetHz: number;
  readonly words: readonly string[];
}

export const RHYTHM_BANDS: readonly RhythmBand[] = [
  { name: 'delta', minHz: 1, maxHz: 4, targetHz: 2.5, words: ['delta'] },
  { name: 'theta', minHz: 4, maxHz: 8, targetHz: 6, words: ['theta'] },
  { name: 'alpha', minHz: 8, maxHz: 13, targetHz: 10, words: ['alpha', 'mu'] },
  { name: 'beta', minHz: 13, maxHz: 30, targetHz: 20, words: ['beta'] },
  { name: 'gamma', minHz: 30, maxHz: 80, targetHz: 40, words: ['gamma'] },
  {
    name: 'ripple',
    minHz: 80,
    maxHz: 200,
    targetHz: 150,
    words: ['ripple', 'ripples', 'sharp-wave', 'fast-ripple'],
  },
];

/** The band containing `hz`, or the nearest one when it falls outside every band. */
export function bandForFrequency(hz: number): RhythmBand {
  for (const band of RHYTHM_BANDS) {
    if (hz >= band.minHz && hz < band.maxHz) return band;
  }
  return hz < RHYTHM_BANDS[0].minHz ? RHYTHM_BANDS[0] : RHYTHM_BANDS[RHYTHM_BANDS.length - 1];
}

export const RHYTHM_WORDS: readonly string[] = [
  'oscillate',
  'oscillates',
  'oscillating',
  'oscillation',
  'oscillations',
  'rhythm',
  'rhythmic',
  'resonate',
  'synchronise',
  'synchronize',
  'synchronous',
  'sync',
];
