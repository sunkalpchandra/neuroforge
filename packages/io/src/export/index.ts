/** Export dispatch: one entry point for every target format. */

import type { Circuit } from '@neuroforge/shared';

import { slugify } from './common';
import { exportBrian2 } from './brian2';
import { exportNest } from './nest';
import { exportNumpy } from './numpy';
import { exportOnnx } from './onnx';
import { exportPyTorch } from './pytorch';
import { parseCircuitDocument, serializeCircuit } from './json';

export type ExportFormat = 'json' | 'brian2' | 'nest' | 'pytorch' | 'onnx' | 'python';

export interface ExportResult {
  filename: string;
  mimeType: string;
  content: string | Uint8Array;
}

const PYTHON_MIME = 'text/x-python';

/** Human-facing description of each format, for menus and tooltips. */
export const EXPORT_FORMAT_LABELS: Record<ExportFormat, string> = {
  json: 'NeuroForge document (.json)',
  brian2: 'Brian2 simulation script (.py)',
  nest: 'PyNEST simulation script (.py)',
  pytorch: 'PyTorch spiking module (.py)',
  onnx: 'ONNX graph (.onnx)',
  python: 'NumPy reference implementation (.py)',
};

/** Render a circuit into the requested format. */
export function exportCircuit(circuit: Circuit, format: ExportFormat): ExportResult {
  const slug = slugify(circuit.name);
  switch (format) {
    case 'json':
      return {
        filename: `${slug}.neuroforge.json`,
        mimeType: 'application/json',
        content: serializeCircuit(circuit),
      };
    case 'brian2':
      return {
        filename: `${slug}.brian2.py`,
        mimeType: PYTHON_MIME,
        content: exportBrian2(circuit),
      };
    case 'nest':
      return {
        filename: `${slug}.nest.py`,
        mimeType: PYTHON_MIME,
        content: exportNest(circuit),
      };
    case 'pytorch':
      return {
        filename: `${slug}.torch.py`,
        mimeType: PYTHON_MIME,
        content: exportPyTorch(circuit),
      };
    case 'python':
      return {
        filename: `${slug}.numpy.py`,
        mimeType: PYTHON_MIME,
        content: exportNumpy(circuit),
      };
    case 'onnx':
      return {
        filename: `${slug}.onnx`,
        mimeType: 'application/octet-stream',
        content: exportOnnx(circuit),
      };
  }
}

/** Read a `.neuroforge.json` document (or any older circuit JSON). */
export function importCircuitJson(text: string): { circuit: Circuit | null; errors: string[] } {
  return parseCircuitDocument(text);
}
