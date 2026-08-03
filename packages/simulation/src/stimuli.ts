import type { SimulationBuffers, Stimulus } from '@neuroforge/shared';
import type { Rng } from '@neuroforge/math';

/**
 * Evaluate a stimulus waveform at a point in time, in pA.
 *
 * Poisson stimuli are the one stochastic case: the amplitude is delivered as
 * discrete events at the configured rate rather than as a continuous level, so
 * the caller must pass the integration step to convert a rate into a per-step
 * probability.
 */
function amplitudeAt(stimulus: Stimulus, time: number, dt: number, rng: Rng): number {
  const p = stimulus.pattern;
  switch (p.kind) {
    case 'constant':
      return p.amplitude;

    case 'step':
      return time >= p.start && time < p.start + p.duration ? p.amplitude : 0;

    case 'ramp': {
      if (time < p.start) return p.from;
      if (time >= p.start + p.duration) return p.to;
      const t = (time - p.start) / p.duration;
      return p.from + (p.to - p.from) * t;
    }

    case 'sine':
      // Frequency is in Hz and time in ms, hence the factor of 1000.
      return p.offset + p.amplitude * Math.sin((2 * Math.PI * p.frequency * time) / 1000);

    case 'pulse-train': {
      if (time < p.start) return 0;
      if (p.frequency <= 0) return 0;
      const period = 1000 / p.frequency;
      const phase = (time - p.start) % period;
      return phase < p.width ? p.amplitude : 0;
    }

    case 'poisson': {
      // Rate is in Hz, dt in ms.
      const probability = (p.rate * dt) / 1000;
      return rng.next() < probability ? p.amplitude : 0;
    }

    default:
      return 0;
  }
}

/**
 * Write every enabled stimulus into the external-current column.
 *
 * The column is cleared first, so a stimulus that has been disabled or whose
 * window has passed stops contributing immediately rather than leaving its last
 * value latched into the neuron forever.
 */
export function applyStimuli(
  buffers: SimulationBuffers,
  stimuli: readonly Stimulus[],
  slotOf: (id: string) => number,
  time: number,
  rng: Rng,
  dt = 0.1,
): void {
  buffers.neurons.iExt.fill(0, 0, buffers.neurons.count);
  if (stimuli.length === 0) return;

  for (const stimulus of stimuli) {
    if (!stimulus.enabled || stimulus.targets.length === 0) continue;
    const value = amplitudeAt(stimulus, time, dt, rng);
    if (value === 0) continue;
    for (const target of stimulus.targets) {
      const slot = slotOf(target);
      if (slot >= 0 && slot < buffers.neurons.count) {
        buffers.neurons.iExt[slot] += value;
      }
    }
  }
}
