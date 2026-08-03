/**
 * @neuroforge/ai — prompt to circuit planning.
 *
 * The package produces `AiPlan`s and never applies them: the editor owns the
 * document, so everything here is a description of edits that something else
 * carries out after `validatePlan` has vetted it.
 *
 * There are two planners. `streamCircuitPlan` talks to Anthropic or OpenAI,
 * either directly from the browser with the user's own key or through the
 * FastAPI proxy in `services/api`. `planLocally` is a deterministic parser that
 * needs no network and no key at all, and is what the builder falls back to.
 */

export type {
  AiCredentials,
  AiPlan,
  AiProvider,
  AiRequest,
  AiStreamEvent,
  CircuitAction,
} from './types';
export { DEFAULT_MODELS } from './types';

export { CIRCUIT_TOOL_SCHEMA, SYSTEM_PROMPT } from './schema';
export { streamCircuitPlan } from './providers';
export { planLocally } from './local-planner';
export { validatePlan } from './validate';
export { loadCredentials, storeCredentials } from './credentials';
