import { getSetting, setSetting } from '@neuroforge/io';
import { asRecord, asString } from './coerce';
import { DEFAULT_MODELS } from './types';
import type { AiCredentials, AiProvider } from './types';

/** Key in the `settings` table of the document database. */
const CREDENTIALS_KEY = 'ai.credentials';

const PROVIDERS: readonly AiProvider[] = ['anthropic', 'openai'];

/**
 * Normalise whatever came back from storage, or was handed in by a caller, into
 * credentials the transports can use. Returns null when there is no key to work
 * with, which is the signal to fall back to the offline planner.
 *
 * The API key is never logged, never included in an error message and never put
 * anywhere but the credentials record itself.
 */
function normalise(value: unknown): AiCredentials | null {
  const record = asRecord(value);
  if (record === null) return null;
  const apiKey = (asString(record.apiKey) ?? '').trim();
  const proxyUrl = (asString(record.proxyUrl) ?? '').trim().replace(/\/+$/, '');
  // A proxy can hold the key server-side, so an empty key is only fatal without one.
  if (apiKey === '' && proxyUrl === '') return null;

  const rawProvider = asString(record.provider);
  const provider: AiProvider =
    rawProvider !== null && (PROVIDERS as readonly string[]).includes(rawProvider)
      ? (rawProvider as AiProvider)
      : 'anthropic';
  const model = (asString(record.model) ?? '').trim();

  const credentials: AiCredentials = {
    provider,
    apiKey,
    model: model === '' ? DEFAULT_MODELS[provider] : model,
  };
  if (proxyUrl !== '') credentials.proxyUrl = proxyUrl;
  return credentials;
}

/**
 * Read the stored credentials from IndexedDB. Resolves to null when nothing is
 * stored, when the record is unusable, or when persistence is unavailable — in
 * every one of those cases the caller should use `planLocally` instead.
 */
export async function loadCredentials(): Promise<AiCredentials | null> {
  try {
    return normalise(await getSetting<unknown>(CREDENTIALS_KEY, null));
  } catch {
    return null;
  }
}

/**
 * Persist credentials, or clear them by passing null. Rejects when IndexedDB is
 * unavailable, because silently dropping a key the user just entered would leave
 * the builder looking broken.
 */
export async function storeCredentials(credentials: AiCredentials | null): Promise<void> {
  if (credentials === null) {
    await setSetting(CREDENTIALS_KEY, null);
    return;
  }
  const normalised = normalise(credentials);
  if (normalised === null) {
    throw new Error('Credentials need either an API key or a proxy URL.');
  }
  await setSetting(CREDENTIALS_KEY, normalised);
}
