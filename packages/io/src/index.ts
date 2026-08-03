/**
 * @neuroforge/io — persistence, document validation and the exporters.
 *
 * The public surface is exactly the one declared in CONTRACTS.md.
 */

export { NeuroForgeDb, db } from './db';

export {
  saveCircuit,
  loadCircuit,
  listCircuits,
  deleteCircuit,
  createSnapshot,
  listSnapshots,
  restoreSnapshot,
  pruneSnapshots,
  getSetting,
  setSetting,
} from './persistence';

export { Autosaver } from './autosave';

export { exportCircuit, importCircuitJson } from './export/index';
export type { ExportFormat, ExportResult } from './export/index';

export { migrateCircuit } from './migrate';
export { downloadExport } from './download';
