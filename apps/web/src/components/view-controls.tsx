'use client';

import { useCallback, useState } from 'react';
import { ChevronDown, Eye, Layers } from 'lucide-react';
import { IconButton, Slider, Tooltip } from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import {
  COLOR_MODES,
  COLOR_MODE_LABELS,
  RENDER_PRESETS,
  RENDER_PRESET_HINTS,
  RENDER_PRESET_LABELS,
  RENDER_PRESET_PATCHES,
  identityColorHex,
} from '@neuroforge/shared';
import type { ColorMode, RenderPreset, RenderSettings } from '@neuroforge/shared';

import { ColorLegend } from './color-legend';

/** Layer toggles, in the order they stack visually in the scene. */
const LAYERS: { key: keyof RenderSettings; label: string }[] = [
  { key: 'showDendrites', label: 'Dendrites' },
  { key: 'showAxons', label: 'Axons' },
  { key: 'showParticles', label: 'Impulses' },
  { key: 'gridVisible', label: 'Grid' },
];

/** Six representative seeds, purely to show the palette the scene is using. */
const SWATCH_SEEDS = [3, 11, 29, 47, 73, 101];

/**
 * Display controls.
 *
 * Deliberately dense and low-chrome: this sits over the canvas permanently, so
 * every pixel it takes is a pixel of connectome it hides. Collapsed by default
 * to a single row of layer toggles.
 */
export function ViewControls() {
  const render = useEditor((s) => s.circuit.render);
  const setRenderSettings = useEditor((s) => s.setRenderSettings);
  const [expanded, setExpanded] = useState(false);

  const setMode = useCallback(
    (mode: ColorMode) => setRenderSettings({ colorMode: mode }),
    [setRenderSettings],
  );

  const applyPreset = useCallback(
    (preset: RenderPreset) => setRenderSettings(RENDER_PRESET_PATCHES[preset]),
    [setRenderSettings],
  );

  const toggleLayer = useCallback(
    (key: keyof RenderSettings) => {
      setRenderSettings({ [key]: !render[key] } as Partial<RenderSettings>);
    },
    [render, setRenderSettings],
  );

  return (
    <div className="nf-glass pointer-events-auto absolute bottom-3 left-3 w-[228px] rounded-panel">
      <div className="flex items-center gap-1.5 border-b border-hairline px-2 py-1.5">
        <Layers className="size-3 text-ink-faint" />
        <span className="text-[10px] tracking-[0.08em] text-ink-faint uppercase">Display</span>
        <div className="flex-1" />
        <IconButton
          label={expanded ? 'Collapse display controls' : 'Expand display controls'}
          size="sm"
          onClick={() => setExpanded((v) => !v)}
        >
          <ChevronDown
            className={`transition-transform duration-200 ${expanded ? '' : '-rotate-90'}`}
          />
        </IconButton>
      </div>

      <div className="flex flex-wrap gap-1 px-2 py-1.5">
        {LAYERS.map((layer) => {
          const on = Boolean(render[layer.key]);
          return (
            <button
              key={layer.key}
              type="button"
              aria-pressed={on}
              onClick={() => toggleLayer(layer.key)}
              className={`rounded-[4px] border px-1.5 py-0.5 text-[10px] transition-colors ${
                on
                  ? 'border-accent/40 bg-accent/12 text-accent'
                  : 'border-hairline text-ink-faint hover:text-ink-muted'
              }`}
            >
              {layer.label}
            </button>
          );
        })}
      </div>

      <ColorLegend />

      {expanded ? (
        <div className="border-t border-hairline px-2 py-2">
          <div className="mb-1 text-[10px] tracking-[0.08em] text-ink-faint uppercase">Look</div>
          <div className="mb-2 flex gap-1">
            {RENDER_PRESETS.map((preset) => (
              <Tooltip key={preset} content={RENDER_PRESET_HINTS[preset]}>
                <button
                  type="button"
                  onClick={() => applyPreset(preset)}
                  className="flex-1 rounded-[4px] border border-hairline px-1.5 py-1 text-[10px] text-ink-muted transition-colors hover:border-hairline-strong hover:bg-panel-raised hover:text-ink"
                >
                  {RENDER_PRESET_LABELS[preset]}
                </button>
              </Tooltip>
            ))}
          </div>

          <div className="mb-1 flex items-center gap-1.5">
            <Eye className="size-3 text-ink-faint" />
            <span className="text-[10px] tracking-[0.08em] text-ink-faint uppercase">
              Colour by
            </span>
          </div>

          <div className="mb-2 flex flex-col gap-px">
            {COLOR_MODES.map((mode) => (
              <button
                key={mode}
                type="button"
                aria-pressed={render.colorMode === mode}
                onClick={() => setMode(mode)}
                className={`flex items-center justify-between rounded-[4px] px-1.5 py-1 text-left text-[11px] transition-colors ${
                  render.colorMode === mode
                    ? 'bg-accent/12 text-accent'
                    : 'text-ink-muted hover:bg-panel-raised hover:text-ink'
                }`}
              >
                <span>{COLOR_MODE_LABELS[mode]}</span>
                {mode === 'identity' ? (
                  <span className="flex gap-px">
                    {SWATCH_SEEDS.map((seed) => (
                      <span
                        key={seed}
                        className="size-1.5 rounded-[1px]"
                        style={{ backgroundColor: identityColorHex(seed) }}
                      />
                    ))}
                  </span>
                ) : null}
              </button>
            ))}
          </div>

          <label className="mb-1 flex items-center justify-between text-[10px] text-ink-faint">
            <span>Dim unselected</span>
            <span className="nf-numeric text-ink-muted">
              {Math.round(render.dimUnselected * 100)}%
            </span>
          </label>
          <Slider
            value={render.dimUnselected}
            onChange={(v) => setRenderSettings({ dimUnselected: v })}
            min={0}
            max={1}
            step={0.01}
            aria-label="Dim unselected cells"
          />

          <label className="mt-2 mb-1 flex items-center justify-between text-[10px] text-ink-faint">
            <span>Saturation</span>
            <span className="nf-numeric text-ink-muted">{render.saturation.toFixed(2)}</span>
          </label>
          <Slider
            value={render.saturation}
            onChange={(v) => setRenderSettings({ saturation: v })}
            min={0}
            max={2}
            step={0.01}
            aria-label="Cell colour saturation"
          />

          <label className="mt-2 mb-1 flex items-center justify-between text-[10px] text-ink-faint">
            <span>Bloom</span>
            <span className="nf-numeric text-ink-muted">{render.bloomIntensity.toFixed(2)}</span>
          </label>
          <Slider
            value={render.bloomIntensity}
            onChange={(v) => setRenderSettings({ bloomIntensity: v })}
            min={0}
            max={3}
            step={0.01}
            aria-label="Bloom intensity"
          />

          <div className="mt-2 flex flex-wrap gap-1">
            <Tooltip content="Screen-space ambient occlusion">
              <button
                type="button"
                aria-pressed={render.ambientOcclusion}
                onClick={() => setRenderSettings({ ambientOcclusion: !render.ambientOcclusion })}
                className={`rounded-[4px] border px-1.5 py-0.5 text-[10px] transition-colors ${
                  render.ambientOcclusion
                    ? 'border-accent/40 bg-accent/12 text-accent'
                    : 'border-hairline text-ink-faint hover:text-ink-muted'
                }`}
              >
                AO
              </button>
            </Tooltip>
            <Tooltip content="Depth of field">
              <button
                type="button"
                aria-pressed={render.depthOfField}
                onClick={() => setRenderSettings({ depthOfField: !render.depthOfField })}
                className={`rounded-[4px] border px-1.5 py-0.5 text-[10px] transition-colors ${
                  render.depthOfField
                    ? 'border-accent/40 bg-accent/12 text-accent'
                    : 'border-hairline text-ink-faint hover:text-ink-muted'
                }`}
              >
                DOF
              </button>
            </Tooltip>
            <Tooltip content="Overlay the membrane-voltage ramp on the cell colour">
              <button
                type="button"
                aria-pressed={render.voltageColoring}
                onClick={() => setRenderSettings({ voltageColoring: !render.voltageColoring })}
                className={`rounded-[4px] border px-1.5 py-0.5 text-[10px] transition-colors ${
                  render.voltageColoring
                    ? 'border-accent/40 bg-accent/12 text-accent'
                    : 'border-hairline text-ink-faint hover:text-ink-muted'
                }`}
              >
                Vm
              </button>
            </Tooltip>
          </div>
        </div>
      ) : null}
    </div>
  );
}
