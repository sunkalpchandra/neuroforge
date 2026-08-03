import type { PopulationSpec, ProjectionSpec } from '@neuroforge/editor';
import type {
  Circuit,
  NeuronParams,
  PlasticityKind,
  RenderSettings,
  SimulationSettings,
  StimulusPattern,
} from '@neuroforge/shared';

/** Re-exported so the action union below can be consumed without a second import. */
export type { PopulationSpec, ProjectionSpec };

export type AiProvider = 'anthropic' | 'openai';

export interface AiCredentials {
  provider: AiProvider;
  apiKey: string;
  model: string;
  proxyUrl?: string;
}

export const DEFAULT_MODELS: Record<AiProvider, string> = {
  anthropic: 'claude-fable-5',
  openai: 'gpt-5',
};

/**
 * A projection between two populations addressed by name rather than by id.
 *
 * A plan is written before any population exists, so it cannot carry
 * `PopulationId`s; the applier resolves the names once the populations are in the
 * document. The editor's spec type is imported for its shape only — this package
 * emits plans and never applies them, so it pulls in no editor runtime code.
 */
export type NamedProjectionSpec = Omit<ProjectionSpec, 'source' | 'target'> & {
  sourceName: string;
  targetName: string;
};

/** One structural edit the model asked for. Validated before it is applied. */
export type CircuitAction =
  | { type: 'create-population'; spec: PopulationSpec }
  | { type: 'connect-populations'; spec: NamedProjectionSpec }
  | { type: 'set-simulation'; patch: Partial<SimulationSettings> }
  | { type: 'set-render'; patch: Partial<RenderSettings> }
  | { type: 'add-stimulus'; targetPopulation: string; pattern: StimulusPattern; name: string }
  | {
      type: 'tune-population';
      name: string;
      params: Partial<NeuronParams>;
      bias?: number;
      noise?: number;
    }
  | {
      type: 'tune-projection';
      name: string;
      weightMean?: number;
      delayMean?: number;
      plasticity?: PlasticityKind;
    }
  | { type: 'clear' };

export interface AiPlan {
  summary: string;
  actions: CircuitAction[];
  warnings: string[];
}

export interface AiRequest {
  prompt: string;
  circuit: Circuit;
  credentials: AiCredentials;
  signal?: AbortSignal;
}

export interface AiStreamEvent {
  kind: 'text' | 'plan' | 'error' | 'done';
  text?: string;
  plan?: AiPlan;
  error?: string;
}
