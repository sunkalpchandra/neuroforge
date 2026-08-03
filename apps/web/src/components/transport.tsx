'use client';

import { useCallback, useEffect, useState } from 'react';
import { Pause, Play, RotateCcw, SkipForward } from 'lucide-react';
import { IconButton, SegmentedControl, Tooltip } from '@neuroforge/ui';
import type { SimulationSettings } from '@neuroforge/shared';

import { getEngine, getProbes } from '@/lib/runtime';

const SPEEDS: { label: string; value: string }[] = [
  { label: '0.1×', value: '0.1' },
  { label: '0.5×', value: '0.5' },
  { label: '1×', value: '1' },
  { label: '2×', value: '2' },
  { label: '5×', value: '5' },
];

export interface TransportProps {
  className?: string;
}

/**
 * Play, pause, single-step and reset.
 *
 * The engine's running flag is the source of truth; this component mirrors it
 * into React state on mount and after every action rather than keeping an
 * independent copy, so a keyboard shortcut that toggles the engine directly
 * cannot leave the button showing the wrong icon.
 */
export function Transport({ className }: TransportProps) {
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState('1');

  useEffect(() => {
    setRunning(getEngine().running);
  }, []);

  const toggle = useCallback(() => {
    const engine = getEngine();
    if (engine.running) engine.pause();
    else engine.play();
    setRunning(engine.running);
  }, []);

  const stepOnce = useCallback(() => {
    const engine = getEngine();
    engine.pause();
    engine.stepOnce();
    getProbes().sample(engine.buffers);
    setRunning(false);
  }, []);

  const reset = useCallback(() => {
    const engine = getEngine();
    engine.reset();
    getProbes().clear();
    setRunning(engine.running);
  }, []);

  const changeSpeed = useCallback((next: string) => {
    setSpeed(next);
    const patch: Partial<SimulationSettings> = { speed: Number.parseFloat(next) };
    getEngine().setSettings(patch);
  }, []);

  // The engine may be toggled from the keyboard map or the command palette, so
  // poll its flag at a low rate to stay in sync without owning the state.
  useEffect(() => {
    const id = setInterval(() => {
      const live = getEngine().running;
      setRunning((previous) => (previous === live ? previous : live));
    }, 250);
    return () => clearInterval(id);
  }, []);

  return (
    <div className={`flex items-center gap-1 ${className ?? ''}`}>
      <Tooltip content={running ? 'Pause' : 'Play'} shortcut="Space">
        <IconButton
          variant={running ? 'secondary' : 'primary'}
          size="sm"
          label={running ? 'Pause simulation' : 'Play simulation'}
          onClick={toggle}
        >
          {running ? <Pause /> : <Play />}
        </IconButton>
      </Tooltip>

      <Tooltip content="Step one timestep" shortcut="→">
        <IconButton variant="ghost" size="sm" label="Step one timestep" onClick={stepOnce}>
          <SkipForward />
        </IconButton>
      </Tooltip>

      <Tooltip content="Reset to rest" shortcut="R">
        <IconButton variant="ghost" size="sm" label="Reset simulation" onClick={reset}>
          <RotateCcw />
        </IconButton>
      </Tooltip>

      <SegmentedControl
        options={SPEEDS}
        value={speed}
        onChange={changeSpeed}
        size="sm"
        aria-label="Simulation speed"
      />
    </div>
  );
}
