/**
 * IndexedDB schema.
 *
 * The table properties are declared with `declare` rather than as real class
 * fields: with `useDefineForClassFields` a plain field declaration would emit
 * `this.circuits = undefined` in the constructor and wipe the tables Dexie
 * installs from `stores()`.
 */

import Dexie from 'dexie';
import type { Circuit, Snapshot } from '@neuroforge/shared';

export const DATABASE_NAME = 'neuroforge';

/** A single persisted preference. */
export interface SettingRecord {
  key: string;
  value: unknown;
}

export class NeuroForgeDb extends Dexie {
  declare circuits: Dexie.Table<Circuit, string>;
  declare snapshots: Dexie.Table<Snapshot, string>;
  declare settings: Dexie.Table<SettingRecord, string>;

  constructor() {
    super(DATABASE_NAME);
    this.version(1).stores({
      // `automatic` is deliberately not indexed: IndexedDB has no boolean key
      // type, so such an index would silently drop every record.
      circuits: 'id, name, updatedAt, createdAt',
      snapshots: 'id, circuitId, createdAt, [circuitId+createdAt]',
      settings: 'key',
    });
  }
}

export const db = new NeuroForgeDb();

/** True when IndexedDB is reachable; false under SSR, in workers without it, and in tests. */
export function isPersistenceAvailable(): boolean {
  return typeof indexedDB !== 'undefined' && indexedDB !== null;
}
