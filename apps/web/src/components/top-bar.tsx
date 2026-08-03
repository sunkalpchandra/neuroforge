'use client';

import { useCallback } from 'react';
import {
  Boxes,
  Command,
  Eraser,
  Hand,
  MousePointer2,
  Redo2,
  Share2,
  Sparkles,
  SlidersHorizontal,
  Undo2,
  Waypoints,
  Zap,
} from 'lucide-react';
import { IconButton, Separator, Tooltip } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import { useDock } from '@/lib/dock-store';
import type { Tool } from '@neuroforge/editor';

import { Transport } from './transport';

const TOOLS: { tool: Tool; icon: React.ReactNode; label: string; key: string }[] = [
  { tool: 'select', icon: <MousePointer2 />, label: 'Select', key: '1' },
  { tool: 'place', icon: <Boxes />, label: 'Place neuron', key: '2' },
  { tool: 'connect', icon: <Waypoints />, label: 'Connect', key: '3' },
  { tool: 'stimulate', icon: <Zap />, label: 'Stimulate', key: '4' },
  { tool: 'erase', icon: <Eraser />, label: 'Erase', key: '5' },
  { tool: 'pan', icon: <Hand />, label: 'Pan', key: '6' },
];

/**
 * The top chrome: identity, tools, transport and panel toggles.
 *
 * Each control reads exactly the slice of the store it needs so that changing
 * the tool does not re-render the transport, and vice versa.
 */
export function TopBar() {
  const tool = useEditor((s) => s.tool);
  const setTool = useEditor((s) => s.setTool);
  const undo = useEditor((s) => s.undo);
  const redo = useEditor((s) => s.redo);
  const undoDepth = useEditor((s) => s.undoDepth);
  const redoDepth = useEditor((s) => s.redoDepth);
  const togglePanel = useEditor((s) => s.togglePanel);
  // Panel visibility is the dock's to decide now, so these read and drive it
  // rather than the editor's flags, which the dock publishes into.
  const toggleDock = useDock((s) => s.toggle);
  const leftActive = useDock((s) => s.left.active);
  const rightActive = useDock((s) => s.right.active);
  const builderOpen = leftActive === 'builder';
  const inspectorOpen = rightActive === 'inspector';

  const openPalette = useCallback(() => togglePanel('commandPalette', true), [togglePanel]);

  return (
    <header className="nf-glass relative z-30 flex h-[var(--nf-topbar-h)] items-center gap-2 border-b border-hairline px-3">
      <div className="flex items-center gap-2 pr-1">
        <div
          aria-hidden
          className="size-4 rounded-[5px]"
          style={{
            background: 'linear-gradient(135deg, #4FD1FF 0%, #B66BFF 100%)',
            boxShadow: '0 0 14px rgba(79,209,255,.45)',
          }}
        />
        <span className="text-[13px] font-medium tracking-tight text-ink">NeuroForge</span>
      </div>

      <Separator orientation="vertical" className="h-5" />

      <div className="flex items-center gap-0.5">
        {TOOLS.map((entry) => (
          <Tooltip key={entry.tool} content={entry.label} shortcut={entry.key}>
            <IconButton
              label={entry.label}
              size="sm"
              variant={tool === entry.tool ? 'secondary' : 'ghost'}
              aria-pressed={tool === entry.tool}
              onClick={() => setTool(entry.tool)}
            >
              {entry.icon}
            </IconButton>
          </Tooltip>
        ))}
      </div>

      <Separator orientation="vertical" className="h-5" />

      <div className="flex items-center gap-0.5">
        <Tooltip content="Undo" shortcut="Mod+Z">
          <IconButton label="Undo" size="sm" disabled={undoDepth === 0} onClick={undo}>
            <Undo2 />
          </IconButton>
        </Tooltip>
        <Tooltip content="Redo" shortcut="Mod+Shift+Z">
          <IconButton label="Redo" size="sm" disabled={redoDepth === 0} onClick={redo}>
            <Redo2 />
          </IconButton>
        </Tooltip>
      </div>

      <Separator orientation="vertical" className="h-5" />

      <Transport />

      <div className="flex-1" />

      <Tooltip content="Command palette" shortcut="Mod+K">
        <IconButton label="Open command palette" size="sm" onClick={openPalette}>
          <Command />
        </IconButton>
      </Tooltip>

      <Tooltip content="AI builder" shortcut="Mod+J">
        <IconButton
          label="Toggle AI builder"
          size="sm"
          variant={builderOpen ? 'secondary' : 'ghost'}
          aria-pressed={builderOpen}
          onClick={() => toggleDock('left', 'builder')}
        >
          <Sparkles />
        </IconButton>
      </Tooltip>

      <Tooltip content="Inspector" shortcut="I">
        <IconButton
          label="Toggle inspector"
          size="sm"
          variant={inspectorOpen ? 'secondary' : 'ghost'}
          aria-pressed={inspectorOpen}
          onClick={() => toggleDock('right', 'inspector')}
        >
          <SlidersHorizontal />
        </IconButton>
      </Tooltip>

      <Tooltip content="Export circuit">
        <IconButton
          label="Export circuit"
          size="sm"
          onClick={() => toggleDock('right', 'library')}
        >
          <Share2 />
        </IconButton>
      </Tooltip>
    </header>
  );
}
