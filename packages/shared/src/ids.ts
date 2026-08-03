/**
 * Branded identifier types. These are structurally strings at runtime but
 * mutually incompatible at compile time, which prevents the single most common
 * class of bug in a graph editor: passing a NeuronId where a SynapseId belongs.
 */

declare const brand: unique symbol;

type Brand<T, B extends string> = T & { readonly [brand]: B };

export type NeuronId = Brand<string, 'NeuronId'>;
export type SynapseId = Brand<string, 'SynapseId'>;
export type PopulationId = Brand<string, 'PopulationId'>;
export type CircuitId = Brand<string, 'CircuitId'>;
export type SnapshotId = Brand<string, 'SnapshotId'>;
export type ProbeId = Brand<string, 'ProbeId'>;
export type StimulusId = Brand<string, 'StimulusId'>;

/**
 * Monotonic, collision-resistant id generator. Uses a 48-bit time prefix so
 * ids sort chronologically, plus 32 bits of entropy. crypto.getRandomValues is
 * available in every browser target and in Node >= 19.
 */
const ID_ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz';

function randomSuffix(bytes: number): string {
  const buf = new Uint8Array(bytes);
  if (typeof globalThis.crypto !== 'undefined' && globalThis.crypto.getRandomValues) {
    globalThis.crypto.getRandomValues(buf);
  } else {
    for (let i = 0; i < bytes; i += 1) buf[i] = Math.floor(Math.random() * 256);
  }
  let out = '';
  for (let i = 0; i < bytes; i += 1) {
    out += ID_ALPHABET[buf[i] % ID_ALPHABET.length];
  }
  return out;
}

function mintId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}${randomSuffix(6)}`;
}

export const newNeuronId = (): NeuronId => mintId('n') as NeuronId;
export const newSynapseId = (): SynapseId => mintId('s') as SynapseId;
export const newPopulationId = (): PopulationId => mintId('p') as PopulationId;
export const newCircuitId = (): CircuitId => mintId('c') as CircuitId;
export const newSnapshotId = (): SnapshotId => mintId('v') as SnapshotId;
export const newProbeId = (): ProbeId => mintId('pr') as ProbeId;
export const newStimulusId = (): StimulusId => mintId('st') as StimulusId;

/** Re-brand a raw string that came from disk or the network. */
export const asNeuronId = (raw: string): NeuronId => raw as NeuronId;
export const asSynapseId = (raw: string): SynapseId => raw as SynapseId;
export const asPopulationId = (raw: string): PopulationId => raw as PopulationId;
export const asCircuitId = (raw: string): CircuitId => raw as CircuitId;
