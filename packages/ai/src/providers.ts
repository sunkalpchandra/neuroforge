import type { Circuit } from '@neuroforge/shared';
import { asArray, asFiniteNumber, asRecord, asString } from './coerce';
import { CIRCUIT_TOOL_NAME, CIRCUIT_TOOL_SCHEMA, SYSTEM_PROMPT, openAiToolSchema } from './schema';
import { SseDecoder } from './sse';
import type { SseFrame } from './sse';
import { DEFAULT_MODELS } from './types';
import type { AiPlan, AiProvider, AiRequest, AiStreamEvent } from './types';
import { validatePlan } from './validate';

const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';
const OPENAI_URL = 'https://api.openai.com/v1/chat/completions';
const ANTHROPIC_VERSION = '2023-06-01';
const MAX_OUTPUT_TOKENS = 8192;

/** Matches the caps the FastAPI proxy enforces, so a direct call behaves the same. */
const MAX_PROMPT_CHARS = 8000;
const MAX_CIRCUIT_BYTES = 512 * 1024;
const MAX_CONTEXT_ITEMS = 200;
const MAX_ERROR_BODY_CHARS = 600;

interface HttpRequest {
  url: string;
  headers: Record<string, string>;
  body: string;
}

interface StreamState {
  /** Accumulated tool-input JSON, keyed by the provider's block or call index. */
  tool: Map<number, string>;
}

interface FrameResult {
  text?: string;
  error?: string;
  finished?: boolean;
}

/**
 * Stream a circuit plan from Anthropic or OpenAI, either straight from the
 * browser with the user's own key or through the FastAPI proxy in
 * `services/api`. The proxy relays the upstream SSE bytes unmodified, so the same
 * parser handles both.
 *
 * This generator never throws: every failure — a bad key, an aborted request, a
 * malformed stream, a model that never called the tool — arrives as an `error`
 * event, and the stream always ends with `done`.
 */
export async function* streamCircuitPlan(request: AiRequest): AsyncGenerator<AiStreamEvent> {
  try {
    yield* runStream(request);
  } catch (error) {
    yield { kind: 'error', error: describeError(error) };
  }
  yield { kind: 'done' };
}

async function* runStream(request: AiRequest): AsyncGenerator<AiStreamEvent> {
  const { credentials, signal } = request;
  const viaProxy = typeof credentials.proxyUrl === 'string' && credentials.proxyUrl.trim() !== '';
  if (!viaProxy && credentials.apiKey.trim() === '') {
    yield {
      kind: 'error',
      error: 'No API key is configured. Add one in settings, or use the offline planner.',
    };
    return;
  }
  if (signal?.aborted === true) {
    yield { kind: 'error', error: 'The request was cancelled.' };
    return;
  }

  const provider: AiProvider = credentials.provider === 'openai' ? 'openai' : 'anthropic';
  const http = viaProxy ? proxyRequest(request, provider) : directRequest(request, provider);

  const response = await fetch(http.url, {
    method: 'POST',
    headers: http.headers,
    body: http.body,
    signal,
  });

  if (!response.ok) {
    yield { kind: 'error', error: await describeHttpError(response, provider, viaProxy) };
    return;
  }
  if (response.body === null) {
    yield { kind: 'error', error: 'The provider returned an empty response body.' };
    return;
  }

  const state: StreamState = { tool: new Map() };
  const decoder = new TextDecoder();
  const sse = new SseDecoder();
  const reader = response.body.getReader();
  let failed = false;

  try {
    let finished = false;
    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      const frames = sse.push(decoder.decode(value, { stream: true }));
      for (const frame of frames) {
        const result = handleFrame(frame, provider, state);
        if (result.text !== undefined && result.text !== '') {
          yield { kind: 'text', text: result.text };
        }
        if (result.error !== undefined) {
          yield { kind: 'error', error: result.error };
          failed = true;
          finished = true;
          break;
        }
        if (result.finished === true) finished = true;
      }
    }
    if (!failed) {
      for (const frame of sse.flush()) {
        const result = handleFrame(frame, provider, state);
        if (result.text !== undefined && result.text !== '') {
          yield { kind: 'text', text: result.text };
        }
        if (result.error !== undefined) {
          yield { kind: 'error', error: result.error };
          failed = true;
        }
      }
    }
  } finally {
    void reader.cancel().catch(() => undefined);
  }

  if (failed) return;
  yield finalise(state, request.circuit);
}

// ------------------------------------------------------------------- requests

function resolveModel(model: string, provider: AiProvider): string {
  const trimmed = model.trim();
  return trimmed === '' ? DEFAULT_MODELS[provider] : trimmed;
}

function userContent(prompt: string, circuit: Circuit): string {
  const context = JSON.stringify(circuitContext(circuit));
  return `${clampPrompt(prompt)}\n\n<current_circuit>\n${context}\n</current_circuit>`;
}

function clampPrompt(prompt: string): string {
  const trimmed = prompt.trim();
  return trimmed.length > MAX_PROMPT_CHARS ? trimmed.slice(0, MAX_PROMPT_CHARS) : trimmed;
}

function directRequest(request: AiRequest, provider: AiProvider): HttpRequest {
  const model = resolveModel(request.credentials.model, provider);
  const content = userContent(request.prompt, request.circuit);
  if (provider === 'anthropic') {
    return {
      url: ANTHROPIC_URL,
      headers: {
        'content-type': 'application/json',
        accept: 'text/event-stream',
        'x-api-key': request.credentials.apiKey.trim(),
        'anthropic-version': ANTHROPIC_VERSION,
        // Without this header the API refuses any request carrying an Origin.
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model,
        max_tokens: MAX_OUTPUT_TOKENS,
        stream: true,
        system: SYSTEM_PROMPT,
        messages: [{ role: 'user', content }],
        tools: [CIRCUIT_TOOL_SCHEMA],
        tool_choice: { type: 'tool', name: CIRCUIT_TOOL_NAME },
      }),
    };
  }
  return {
    url: OPENAI_URL,
    headers: {
      'content-type': 'application/json',
      accept: 'text/event-stream',
      authorization: `Bearer ${request.credentials.apiKey.trim()}`,
    },
    body: JSON.stringify({
      model,
      stream: true,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content },
      ],
      tools: [{ type: 'function', function: openAiToolSchema() }],
      tool_choice: { type: 'function', function: { name: CIRCUIT_TOOL_NAME } },
    }),
  };
}

function proxyRequest(request: AiRequest, provider: AiProvider): HttpRequest {
  const base = (request.credentials.proxyUrl ?? '').trim().replace(/\/+$/, '');
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: 'text/event-stream',
  };
  const key = request.credentials.apiKey.trim();
  // The proxy falls back to its own environment key when this header is absent.
  if (key !== '') headers['X-Provider-Key'] = key;
  return {
    url: `${base}/v1/plan`,
    headers,
    body: JSON.stringify({
      prompt: clampPrompt(request.prompt),
      provider,
      model: resolveModel(request.credentials.model, provider),
      circuit: circuitContext(request.circuit),
      system: SYSTEM_PROMPT,
      tool_schema: provider === 'anthropic' ? CIRCUIT_TOOL_SCHEMA : openAiToolSchema(),
    }),
  };
}

/**
 * A compact description of the document, small enough to stay under the proxy's
 * 512 KB context cap. Individual neurons and synapses are never sent: the model
 * addresses the circuit by population and projection name, so the counts and the
 * per-population summary are everything it can act on.
 */
function circuitContext(circuit: Circuit): Record<string, unknown> {
  const populationNames = new Map<string, string>();
  for (const population of circuit.populations) populationNames.set(population.id, population.name);

  const context: Record<string, unknown> = {
    name: circuit.name,
    neuronCount: circuit.neurons.length,
    synapseCount: circuit.synapses.length,
    simulation: circuit.simulation,
    populations: circuit.populations.slice(0, MAX_CONTEXT_ITEMS).map((population) => ({
      name: population.name,
      size: population.size,
      polarity: population.polarity,
      model: population.params.kind,
      archetype: population.morphology.archetype,
      layout: population.layout.kind,
      origin: population.origin,
    })),
    projections: circuit.projections.slice(0, MAX_CONTEXT_ITEMS).map((projection) => ({
      name: projection.name,
      source: populationNames.get(projection.source) ?? projection.source,
      target: populationNames.get(projection.target) ?? projection.target,
      rule: projection.rule,
      weightMean: projection.weightMean,
      delayMean: projection.delayMean,
    })),
    stimuli: circuit.stimuli.slice(0, MAX_CONTEXT_ITEMS).map((stimulus) => ({
      name: stimulus.name,
      pattern: stimulus.pattern,
      targetCount: stimulus.targets.length,
      enabled: stimulus.enabled,
    })),
  };

  for (const droppable of ['stimuli', 'projections', 'populations']) {
    if (JSON.stringify(context).length <= MAX_CIRCUIT_BYTES) return context;
    context[droppable] = 'omitted: too large to send as context';
  }
  return JSON.stringify(context).length <= MAX_CIRCUIT_BYTES ? context : { name: circuit.name };
}

// -------------------------------------------------------------------- parsing

function handleFrame(frame: SseFrame, provider: AiProvider, state: StreamState): FrameResult {
  if (frame.data === '' || frame.data === '[DONE]') {
    return frame.data === '[DONE]' ? { finished: true } : {};
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(frame.data);
  } catch {
    // The proxy emits `event: error` frames whose payload is JSON; anything else
    // that fails to parse is a transport artefact and is safe to skip.
    return frame.event === 'error'
      ? { error: `The planning service reported: ${frame.data.slice(0, MAX_ERROR_BODY_CHARS)}` }
      : {};
  }
  const record = asRecord(parsed);
  if (record === null) return {};

  const relayed = relayedError(record);
  if (relayed !== null) return { error: relayed };

  return provider === 'anthropic'
    ? handleAnthropicFrame(frame, record, state)
    : handleOpenAiFrame(record, state);
}

/** Both providers, and the proxy, report failures as a top-level `error` field. */
function relayedError(record: Record<string, unknown>): string | null {
  const direct = asString(record.error);
  if (direct !== null) return direct;
  const nested = asRecord(record.error);
  if (nested === null) return null;
  return asString(nested.message) ?? asString(nested.type) ?? 'The provider reported an error.';
}

function handleAnthropicFrame(
  frame: SseFrame,
  record: Record<string, unknown>,
  state: StreamState,
): FrameResult {
  const type = asString(record.type) ?? frame.event;
  const index = asFiniteNumber(record.index) ?? 0;
  switch (type) {
    case 'content_block_start': {
      const block = asRecord(record.content_block);
      if (block !== null && asString(block.type) === 'tool_use') {
        state.tool.set(index, '');
      }
      return {};
    }
    case 'content_block_delta': {
      const delta = asRecord(record.delta);
      if (delta === null) return {};
      const deltaType = asString(delta.type);
      if (deltaType === 'text_delta') return { text: asString(delta.text) ?? '' };
      if (deltaType === 'input_json_delta') {
        const partial = asString(delta.partial_json) ?? '';
        state.tool.set(index, (state.tool.get(index) ?? '') + partial);
      }
      return {};
    }
    case 'message_stop':
      return { finished: true };
    default:
      return {};
  }
}

function handleOpenAiFrame(record: Record<string, unknown>, state: StreamState): FrameResult {
  const choices = asArray(record.choices);
  if (choices === null || choices.length === 0) return {};
  const choice = asRecord(choices[0]);
  if (choice === null) return {};
  const delta = asRecord(choice.delta);
  let text = '';
  if (delta !== null) {
    text = asString(delta.content) ?? '';
    for (const rawCall of asArray(delta.tool_calls) ?? []) {
      const call = asRecord(rawCall);
      if (call === null) continue;
      const index = asFiniteNumber(call.index) ?? 0;
      const fn = asRecord(call.function);
      const args = fn === null ? null : asString(fn.arguments);
      state.tool.set(index, (state.tool.get(index) ?? '') + (args ?? ''));
    }
  }
  // Some gateways close the stream without the terminal `[DONE]`; a populated
  // finish_reason is the other reliable signal that the choice is complete.
  return asString(choice.finish_reason) === null ? { text } : { text, finished: true };
}

/** Turn the accumulated tool input into a validated plan, or an error event. */
function finalise(state: StreamState, circuit: Circuit): AiStreamEvent {
  const candidates = [...state.tool.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, json]) => json)
    .filter((json) => json.trim().length > 0);

  if (candidates.length === 0) {
    return {
      kind: 'error',
      error: `The model finished without calling ${CIRCUIT_TOOL_NAME}, so there is no plan to apply. Try rephrasing the request as a concrete circuit change.`,
    };
  }

  for (const json of candidates) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch {
      continue;
    }
    // validatePlan re-checks every field, so handing it an unverified value is safe.
    const { plan, errors } = validatePlan(parsed as AiPlan, circuit);
    if (errors.length > 0) plan.warnings = [...plan.warnings, ...errors];
    return { kind: 'plan', plan };
  }

  return {
    kind: 'error',
    error: `The tool call was cut off or malformed and could not be read as JSON: ${candidates[0].slice(0, MAX_ERROR_BODY_CHARS)}`,
  };
}

// --------------------------------------------------------------------- errors

async function describeHttpError(
  response: Response,
  provider: AiProvider,
  viaProxy: boolean,
): Promise<string> {
  let detail = '';
  try {
    detail = (await response.text()).slice(0, MAX_ERROR_BODY_CHARS).trim();
  } catch {
    detail = '';
  }
  const parsedDetail = extractDetail(detail);
  const origin = viaProxy ? 'The planning proxy' : provider === 'openai' ? 'OpenAI' : 'Anthropic';
  const suffix = parsedDetail === '' ? '' : ` — ${parsedDetail}`;

  if (response.status === 401 || response.status === 403) {
    return `${origin} rejected the credentials (HTTP ${response.status})${suffix}`;
  }
  if (response.status === 404) {
    return `${origin} has no endpoint at that address (HTTP 404)${suffix}`;
  }
  if (response.status === 429) {
    const retry = response.headers.get('retry-after');
    const wait = retry === null ? '' : ` Retry in ${retry}s.`;
    return `${origin} is rate limiting this key (HTTP 429)${suffix}.${wait}`;
  }
  if (response.status >= 500) {
    return `${origin} is unavailable right now (HTTP ${response.status})${suffix}`;
  }
  return `${origin} rejected the request (HTTP ${response.status})${suffix}`;
}

function extractDetail(body: string): string {
  if (body === '') return '';
  try {
    const record = asRecord(JSON.parse(body));
    if (record !== null) {
      const message = relayedError(record) ?? asString(record.detail);
      if (message !== null && message !== '') return message;
    }
  } catch {
    // Not JSON; fall through to the raw body.
  }
  return body;
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    if (error.name === 'AbortError') return 'The request was cancelled.';
    if (error.name === 'TypeError') {
      return `Could not reach the model provider: ${error.message}. Check the network connection, and the proxy URL if one is configured.`;
    }
    return error.message === '' ? error.name : error.message;
  }
  return typeof error === 'string' ? error : 'The planning request failed for an unknown reason.';
}
