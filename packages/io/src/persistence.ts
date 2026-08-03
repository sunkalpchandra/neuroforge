/**
 * Document persistence.
 *
 * Everything read back from IndexedDB goes through `migrateCircuit`: the store
 * may hold documents written by an older build, and a record that has been
 * corrupted must not reach the editor as a half-built graph. Problems are
 * reported on the console and the record is skipped rather than loaded.
 */

import { newSnapshotId } from '@neuroforge/shared';
import type { Circuit, CircuitId, Snapshot, SnapshotId } from '@neuroforge/shared';

import { db, isPersistenceAvailable } from './db';
import { migrateCircuit } from './migrate';

/** How many snapshots a circuit keeps before the oldest are pruned. */
export const DEFAULT_SNAPSHOT_LIMIT = 50;

function requirePersistence(operation: string): void {
  if (!isPersistenceAvailable()) {
    throw new Error(`${operation} requires IndexedDB, which is not available in this environment`);
  }
}

/**
 * A detached copy of a document. IndexedDB structured-clones on write anyway,
 * but taking the copy here means a later in-place edit by the editor cannot
 * race the write.
 */
function clone<T>(value: T): T {
  if (typeof structuredClone === 'function') return structuredClone(value);
  return JSON.parse(JSON.stringify(value)) as T;
}

function validate(raw: unknown, source: string): Circuit | null {
  const { circuit, errors } = migrateCircuit(raw);
  if (errors.length > 0) {
    const label = circuit ? 'repaired' : 'rejected';
    console.warn(`[neuroforge/io] ${label} ${source}:\n  ${errors.join('\n  ')}`);
  }
  return circuit;
}

export async function saveCircuit(circuit: Circuit): Promise<void> {
  requirePersistence('saveCircuit');
  // The stored copy always carries the write time so `listCircuits` can order
  // by recency even if the caller never touched `updatedAt`.
  const record: Circuit = { ...clone(circuit), updatedAt: Date.now() };
  await db.circuits.put(record);
}

export async function loadCircuit(id: CircuitId): Promise<Circuit | null> {
  if (!isPersistenceAvailable()) return null;
  const raw: unknown = await db.circuits.get(id);
  if (raw === undefined || raw === null) return null;
  return validate(raw, `circuit '${id}'`);
}

export async function listCircuits(): Promise<Circuit[]> {
  if (!isPersistenceAvailable()) return [];
  const rows: unknown[] = await db.circuits.toArray();
  const circuits: Circuit[] = [];
  for (const raw of rows) {
    const circuit = validate(raw, 'a stored circuit');
    if (circuit) circuits.push(circuit);
  }
  circuits.sort((a, b) => b.updatedAt - a.updatedAt);
  return circuits;
}

export async function deleteCircuit(id: CircuitId): Promise<void> {
  requirePersistence('deleteCircuit');
  await db.transaction('rw', db.circuits, db.snapshots, async () => {
    await db.circuits.delete(id);
    await db.snapshots.where('circuitId').equals(id).delete();
  });
}

export async function createSnapshot(
  circuit: Circuit,
  label: string,
  automatic = false,
): Promise<Snapshot> {
  requirePersistence('createSnapshot');
  const snapshot: Snapshot = {
    id: newSnapshotId(),
    circuitId: circuit.id,
    label,
    createdAt: Date.now(),
    automatic,
    neuronCount: circuit.neurons.length,
    synapseCount: circuit.synapses.length,
    circuit: clone(circuit),
  };
  await db.snapshots.put(snapshot);
  await pruneSnapshots(circuit.id);
  return snapshot;
}

/** Newest first. */
export async function listSnapshots(id: CircuitId): Promise<Snapshot[]> {
  if (!isPersistenceAvailable()) return [];
  const rows = await db.snapshots.where('circuitId').equals(id).toArray();
  rows.sort((a, b) => b.createdAt - a.createdAt);
  return rows;
}

export async function restoreSnapshot(id: SnapshotId): Promise<Circuit | null> {
  if (!isPersistenceAvailable()) return null;
  const snapshot = await db.snapshots.get(id);
  if (!snapshot) return null;
  return validate(snapshot.circuit, `snapshot '${id}'`);
}

/**
 * Trim a circuit's version history to `keep` entries. Automatic snapshots are
 * dropped before manual ones, and within each class the oldest go first.
 */
export async function pruneSnapshots(id: CircuitId, keep = DEFAULT_SNAPSHOT_LIMIT): Promise<void> {
  if (!isPersistenceAvailable()) return;
  const limit = Math.max(0, Math.floor(keep));
  const rows = await db.snapshots.where('circuitId').equals(id).toArray();
  if (rows.length <= limit) return;
  const ordered = [...rows].sort((a, b) => {
    if (a.automatic !== b.automatic) return a.automatic ? -1 : 1;
    return a.createdAt - b.createdAt;
  });
  const doomed = ordered.slice(0, rows.length - limit).map((snapshot) => snapshot.id);
  await db.snapshots.bulkDelete(doomed);
}

export async function getSetting<T>(key: string, fallback: T): Promise<T> {
  if (!isPersistenceAvailable()) return fallback;
  const record = await db.settings.get(key);
  if (!record || record.value === undefined || record.value === null) return fallback;
  return record.value as T;
}

export async function setSetting(key: string, value: unknown): Promise<void> {
  requirePersistence('setSetting');
  await db.settings.put({ key, value: clone(value) });
}
