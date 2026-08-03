import type { NeuronModelKind, NeuronParams } from '@neuroforge/shared';

/**
 * Editable parameter descriptors per membrane model.
 *
 * Ranges are the physiologically meaningful span, not the numerically legal
 * one: a slider that lets a user set a 4000 pF membrane capacitance is worse
 * than one that stops where the biology does. Values outside a range remain
 * representable by typing, which is what the `soft` flag records.
 */
export interface ParamField {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  precision: number;
  /** Parameters spanning decades are edited on a logarithmic scale. */
  log?: boolean;
  /** The range is advisory; typed values beyond it are accepted. */
  soft?: boolean;
  hint: string;
}

const LIF: ParamField[] = [
  { key: 'cm', label: 'Capacitance', unit: 'pF', min: 20, max: 800, step: 1, precision: 0, hint: 'Membrane capacitance; larger is slower to charge.' },
  { key: 'gL', label: 'Leak conductance', unit: 'nS', min: 1, max: 100, step: 0.5, precision: 1, hint: 'Sets the membrane time constant together with capacitance.' },
  { key: 'eL', label: 'Resting potential', unit: 'mV', min: -90, max: -50, step: 0.5, precision: 1, hint: 'Potential the membrane relaxes toward with no input.' },
  { key: 'vThresh', label: 'Threshold', unit: 'mV', min: -65, max: -20, step: 0.5, precision: 1, hint: 'Crossing this emits a spike and resets the membrane.' },
  { key: 'vReset', label: 'Reset', unit: 'mV', min: -90, max: -40, step: 0.5, precision: 1, hint: 'Potential the membrane is clamped to after a spike.' },
  { key: 'tRefract', label: 'Refractory', unit: 'ms', min: 0, max: 20, step: 0.1, precision: 1, hint: 'Dead time after a spike during which input is ignored.' },
];

const IZHIKEVICH: ParamField[] = [
  { key: 'a', label: 'Recovery rate', unit: '', min: 0.005, max: 0.2, step: 0.001, precision: 3, hint: 'Time scale of the recovery variable. Larger recovers faster.' },
  { key: 'b', label: 'Recovery coupling', unit: '', min: -0.1, max: 0.4, step: 0.01, precision: 2, hint: 'Sensitivity of recovery to subthreshold voltage.' },
  { key: 'c', label: 'Reset voltage', unit: 'mV', min: -80, max: -40, step: 1, precision: 0, hint: 'Post-spike voltage reset. Less negative gives bursting.' },
  { key: 'd', label: 'Recovery jump', unit: '', min: 0, max: 12, step: 0.05, precision: 2, hint: 'Added to recovery after each spike. Larger adapts more.' },
  { key: 'vPeak', label: 'Peak', unit: 'mV', min: 10, max: 50, step: 1, precision: 0, hint: 'Cutoff voltage at which a spike is registered.' },
  { key: 'iScale', label: 'Input scale', unit: '', min: 0.005, max: 0.2, step: 0.005, precision: 3, hint: 'Converts injected picoamps into model units.' },
];

const HODGKIN_HUXLEY: ParamField[] = [
  { key: 'cm', label: 'Capacitance', unit: 'pF', min: 20, max: 400, step: 1, precision: 0, hint: 'Membrane capacitance.' },
  { key: 'gNa', label: 'Sodium', unit: 'nS', min: 0, max: 40000, step: 100, precision: 0, log: true, hint: 'Maximal sodium conductance; drives the spike upstroke.' },
  { key: 'gK', label: 'Potassium', unit: 'nS', min: 0, max: 15000, step: 50, precision: 0, log: true, hint: 'Maximal potassium conductance; repolarises the membrane.' },
  { key: 'gL', label: 'Leak', unit: 'nS', min: 0, max: 200, step: 1, precision: 0, hint: 'Passive leak conductance.' },
  { key: 'eNa', label: 'Sodium reversal', unit: 'mV', min: 20, max: 80, step: 1, precision: 0, hint: 'Sodium equilibrium potential.' },
  { key: 'eK', label: 'Potassium reversal', unit: 'mV', min: -100, max: -60, step: 1, precision: 0, hint: 'Potassium equilibrium potential.' },
  { key: 'eL', label: 'Leak reversal', unit: 'mV', min: -80, max: -40, step: 0.5, precision: 1, hint: 'Leak equilibrium potential.' },
  { key: 'vDetect', label: 'Spike detect', unit: 'mV', min: -40, max: 20, step: 1, precision: 0, hint: 'Upward crossing of this voltage counts as a spike.' },
  { key: 'q10', label: 'Temperature', unit: '×', min: 0.5, max: 4, step: 0.05, precision: 2, hint: 'Scales all channel rate constants.' },
];

const ADEX: ParamField[] = [
  { key: 'cm', label: 'Capacitance', unit: 'pF', min: 20, max: 800, step: 1, precision: 0, hint: 'Membrane capacitance.' },
  { key: 'gL', label: 'Leak conductance', unit: 'nS', min: 1, max: 100, step: 0.5, precision: 1, hint: 'Passive leak conductance.' },
  { key: 'eL', label: 'Resting potential', unit: 'mV', min: -90, max: -50, step: 0.5, precision: 1, hint: 'Resting membrane potential.' },
  { key: 'deltaT', label: 'Sharpness', unit: 'mV', min: 0.1, max: 10, step: 0.1, precision: 1, hint: 'Sharpness of the exponential upstroke. Smaller is sharper.' },
  { key: 'vT', label: 'Soft threshold', unit: 'mV', min: -65, max: -30, step: 0.5, precision: 1, hint: 'Voltage where the exponential term takes over.' },
  { key: 'vPeak', label: 'Peak', unit: 'mV', min: -10, max: 50, step: 1, precision: 0, hint: 'Numerical cutoff at which the spike is registered.' },
  { key: 'vReset', label: 'Reset', unit: 'mV', min: -90, max: -40, step: 0.5, precision: 1, hint: 'Post-spike reset potential.' },
  { key: 'a', label: 'Subthreshold adapt', unit: 'nS', min: -10, max: 40, step: 0.5, precision: 1, hint: 'Couples adaptation to subthreshold voltage.' },
  { key: 'b', label: 'Spike adapt', unit: 'pA', min: 0, max: 400, step: 1, precision: 0, hint: 'Added to the adaptation current on every spike.' },
  { key: 'tauW', label: 'Adaptation τ', unit: 'ms', min: 1, max: 1000, step: 1, precision: 0, log: true, hint: 'Adaptation time constant.' },
  { key: 'tRefract', label: 'Refractory', unit: 'ms', min: 0, max: 20, step: 0.1, precision: 1, hint: 'Dead time after a spike.' },
];

const MORRIS_LECAR: ParamField[] = [
  { key: 'cm', label: 'Capacitance', unit: 'pF', min: 1, max: 100, step: 1, precision: 0, hint: 'Membrane capacitance.' },
  { key: 'gCa', label: 'Calcium', unit: 'nS', min: 0, max: 30, step: 0.1, precision: 1, hint: 'Maximal calcium conductance.' },
  { key: 'gK', label: 'Potassium', unit: 'nS', min: 0, max: 40, step: 0.1, precision: 1, hint: 'Maximal potassium conductance.' },
  { key: 'gL', label: 'Leak', unit: 'nS', min: 0, max: 20, step: 0.1, precision: 1, hint: 'Passive leak conductance.' },
  { key: 'eCa', label: 'Calcium reversal', unit: 'mV', min: 60, max: 160, step: 1, precision: 0, hint: 'Calcium equilibrium potential.' },
  { key: 'eK', label: 'Potassium reversal', unit: 'mV', min: -110, max: -60, step: 1, precision: 0, hint: 'Potassium equilibrium potential.' },
  { key: 'eL', label: 'Leak reversal', unit: 'mV', min: -90, max: -30, step: 1, precision: 0, hint: 'Leak equilibrium potential.' },
  { key: 'v1', label: 'Ca half-activation', unit: 'mV', min: -30, max: 20, step: 0.5, precision: 1, hint: 'Voltage of half-maximal calcium activation.' },
  { key: 'v2', label: 'Ca slope', unit: 'mV', min: 1, max: 40, step: 0.5, precision: 1, hint: 'Steepness of calcium activation.' },
  { key: 'v3', label: 'K half-activation', unit: 'mV', min: -30, max: 30, step: 0.5, precision: 1, hint: 'Voltage of half-maximal potassium activation.' },
  { key: 'v4', label: 'K slope', unit: 'mV', min: 1, max: 60, step: 0.5, precision: 1, hint: 'Steepness of potassium activation.' },
  { key: 'phi', label: 'Rate scale', unit: '1/ms', min: 0.001, max: 0.5, step: 0.001, precision: 3, log: true, hint: 'Reference rate for the recovery variable.' },
  { key: 'vDetect', label: 'Spike detect', unit: 'mV', min: -30, max: 30, step: 1, precision: 0, hint: 'Upward crossing of this voltage counts as a spike.' },
];

export const PARAM_FIELDS: Record<NeuronModelKind, readonly ParamField[]> = {
  lif: LIF,
  izhikevich: IZHIKEVICH,
  'hodgkin-huxley': HODGKIN_HUXLEY,
  adex: ADEX,
  'morris-lecar': MORRIS_LECAR,
};

/** Read a parameter by key without widening the union to an index signature. */
export function readParam(params: NeuronParams, key: string): number {
  const value = (params as unknown as Record<string, unknown>)[key];
  return typeof value === 'number' ? value : 0;
}

/** Produce a patched copy of a params object with one key replaced. */
export function writeParam(params: NeuronParams, key: string, value: number): NeuronParams {
  return { ...params, [key]: value } as NeuronParams;
}
