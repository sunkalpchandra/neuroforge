'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from 'react';
import {
  BoxSelect,
  Check,
  FilePlus2,
  GitBranch,
  LayoutGrid,
  Library,
  Palette,
  Play,
  Redo2,
  RotateCcw,
  Search,
  SkipForward,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Undo2,
  Wand2,
  Waypoints,
  X,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Kbd, ScrollArea, cn, pushToast } from '@neuroforge/ui';
import { buildShortcuts, useEditor } from '@neuroforge/editor';
import type { Shortcut } from '@neuroforge/editor';
import { COLOR_MODES, COLOR_MODE_LABELS } from '@neuroforge/shared';
import type { RenderSettings } from '@neuroforge/shared';

import { getEngine, getProbes } from '@/lib/runtime';

/* ------------------------------------------------------------ fuzzy match -- */

const MATCH_SCORE = 16;
const CONTIGUOUS_BONUS = 14;
const BOUNDARY_BONUS = 10;
const HEAD_BONUS = 6;
const GAP_PENALTY = 3;
/** Sentinel for "no alignment reaches this cell". Never within a real score's range. */
const UNREACHABLE = -1e9;
/** Above this, a cell holds a real alignment rather than the sentinel. */
const REACHABLE = UNREACHABLE / 2;

/**
 * How much less a matched character is worth in a keyword than in the title.
 *
 * Subtracted rather than scaled: a score can be negative — a lone character
 * found late in a long title pays more in gap penalty than one match earns — and
 * scaling a negative number by a factor below one *raises* it, which would have
 * ranked a keyword hit above the identical hit in a title.
 */
const KEYWORD_PENALTY = 11;
/** Breaks ties towards the tighter label when two commands match equally well. */
const LENGTH_PENALTY = 0.15;

const NO_POSITIONS: readonly number[] = [];

interface FuzzyMatch {
  score: number;
  positions: readonly number[];
}

/** True where `index` starts a word, which is where a match reads as intentional. */
function isBoundary(text: string, index: number): boolean {
  if (index === 0) return true;
  const previous = text[index - 1];
  if (previous === ' ' || previous === '-' || previous === '/' || previous === '.') return true;
  const current = text[index];
  return previous !== previous.toUpperCase() && current !== current.toLowerCase();
}

/**
 * Subsequence match with an optimal alignment.
 *
 * A greedy left-to-right scan is not enough: "gr" against "Toggle grid" would
 * take the g of "Toggle" and never find the contiguous "gr", scoring and
 * highlighting the worse of the two alignments. So this is a dynamic program
 * over `row[j]` — the best score whose i-th query character lands on j — with an
 * explicit parent column per cell, and the alignment that is reported is the
 * one the score was actually achieved by.
 *
 * Only the leading gap and the gaps between matches are charged. Charging the
 * trailing gap as well would make the penalty `GAP_PENALTY * (n - m)` for every
 * alignment of a given text — a constant that cancels out and leaves position
 * with no influence at all, which would have highlighted the *last* of several
 * equally good occurrences rather than the first.
 */
function fuzzyMatch(query: string, text: string): FuzzyMatch | null {
  const m = query.length;
  const n = text.length;
  if (m === 0) return { score: 0, positions: NO_POSITIONS };
  if (m > n) return null;

  const lower = text.toLowerCase();
  const from = new Int32Array(m * n);
  let previous = new Float64Array(n);
  let row = new Float64Array(n);

  for (let i = 0; i < m; i += 1) {
    const wanted = query.charCodeAt(i);
    // Running maximum over the previous row with the gap penalty folded in as
    // `+ GAP_PENALTY * j`, which turns "best predecessor, less one penalty per
    // character skipped since" into a plain comparison and keeps the whole
    // matcher linear in m*n. `carriedAt` holds the leftmost column attaining it,
    // so equally good alignments highlight the earlier one.
    let carried = UNREACHABLE;
    let carriedAt = 0;

    for (let j = 0; j < n; j += 1) {
      if (i > 0 && j > 0 && previous[j - 1] > REACHABLE) {
        const lifted = previous[j - 1] + GAP_PENALTY * (j - 1);
        if (lifted > carried) {
          carried = lifted;
          carriedAt = j - 1;
        }
      }

      let score = UNREACHABLE;
      let parent = -1;

      if (lower.charCodeAt(j) === wanted) {
        let base = UNREACHABLE;
        if (i === 0) {
          // Every character skipped before the first match costs, which is what
          // makes an earlier match outrank an otherwise identical later one.
          base = -GAP_PENALTY * j;
        } else if (j > 0) {
          const chained =
            previous[j - 1] > REACHABLE ? previous[j - 1] + CONTIGUOUS_BONUS : UNREACHABLE;
          const skipped = carried > REACHABLE ? carried - GAP_PENALTY * (j - 1) : UNREACHABLE;
          // Ties go to the contiguous run: same score, better highlight.
          if (chained >= skipped) {
            base = chained;
            parent = j - 1;
          } else {
            base = skipped;
            parent = carriedAt;
          }
        }
        if (base > REACHABLE) {
          score =
            base +
            MATCH_SCORE +
            (isBoundary(text, j) ? BOUNDARY_BONUS : 0) +
            (j === 0 ? HEAD_BONUS : 0);
        } else {
          parent = -1;
        }
      }

      row[j] = score;
      from[i * n + j] = parent;
    }

    const spare = previous;
    previous = row;
    row = spare;
  }

  // `previous` now holds the last query character's row; a strict comparison
  // keeps the leftmost of several columns that score the same.
  let total = UNREACHABLE;
  let end = -1;
  for (let j = 0; j < n; j += 1) {
    if (previous[j] > total) {
      total = previous[j];
      end = j;
    }
  }
  if (end < 0 || total <= REACHABLE) return null;

  const positions = new Array<number>(m);
  let column = end;
  for (let i = m - 1; i >= 0; i -= 1) {
    positions[i] = column;
    column = from[i * n + column];
  }

  return { score: total, positions };
}

/* ---------------------------------------------------------------- commands -- */

/** Section order with no query is the order `buildCommands` pushes them in. */
type Section = 'Simulation' | 'Edit' | 'View' | 'Panels' | 'Circuit';

interface PaletteCommand {
  id: string;
  section: Section;
  title: string;
  keywords: readonly string[];
  icon: LucideIcon;
  /** Canonical `Mod+Shift+K` form, rendered with Kbd. */
  keys?: string;
  /** Current value of a command that flips a setting. */
  toggled?: boolean;
  /** Marks the active member of a mutually exclusive group. */
  selected?: boolean;
  /** Right-aligned numeric readout, e.g. how many cells a command would affect. */
  hint?: string;
  run(): void;
}

interface CommandContext {
  shortcuts: Map<string, Shortcut>;
  render: RenderSettings;
  inspectorOpen: boolean;
  builderOpen: boolean;
  libraryOpen: boolean;
  selectionCount: number;
  synapseCount: number;
  undoDepth: number;
  redoDepth: number;
  running: boolean;
}

function countHint(value: number): string | undefined {
  return value > 0 ? String(value) : undefined;
}

function buildCommands(context: CommandContext): PaletteCommand[] {
  const { render, shortcuts } = context;
  const commands: PaletteCommand[] = [];
  const selectionHint = countHint(context.selectionCount);

  /**
   * Entries whose behaviour already exists in the editor's keyboard map take
   * both their action and their binding from it, so a palette row and the
   * keystroke that does the same thing can never drift apart. A descriptor that
   * cannot be found is dropped rather than rendered as a row that does nothing.
   */
  const bound = (label: string, spec: Omit<PaletteCommand, 'run' | 'keys'>): void => {
    const shortcut = shortcuts.get(label);
    if (shortcut === undefined) return;
    commands.push({ ...spec, keys: shortcut.keys, run: shortcut.run });
  };

  /* -- Simulation ----------------------------------------------------------- */

  // The transport descriptors hand their work to whoever owns the engine by
  // publishing a window event. This component holds that same runtime singleton,
  // so it drives it directly.
  //
  // No `keys` chip on these two: their descriptors' bindings — Space and R —
  // only emit `neuroforge:shortcut`, and nothing in the app listens for it, so
  // the chip would advertise a keystroke that does nothing. They belong here as
  // soon as `transport.tsx`, which owns these buttons, handles those actions.
  commands.push({
    id: 'sim.play',
    section: 'Simulation',
    title: 'Play / pause simulation',
    keywords: ['run', 'start', 'stop', 'resume', 'transport', 'integrate'],
    icon: Play,
    hint: context.running ? 'running' : 'paused',
    run: () => {
      const engine = getEngine();
      if (engine.running) engine.pause();
      else engine.play();
    },
  });

  commands.push({
    id: 'sim.step',
    section: 'Simulation',
    title: 'Step one timestep',
    keywords: ['advance', 'frame', 'single', 'substep', 'tick'],
    icon: SkipForward,
    run: () => {
      const engine = getEngine();
      // Stepping only means anything from a stopped clock.
      engine.pause();
      engine.stepOnce();
      getProbes().sample(engine.buffers);
    },
  });

  commands.push({
    id: 'sim.reset',
    section: 'Simulation',
    title: 'Reset simulation',
    keywords: ['rest', 'clear state', 'restart', 'zero', 'voltage'],
    icon: RotateCcw,
    run: () => {
      const engine = getEngine();
      engine.reset();
      // Traces recorded before the reset describe a state that no longer exists.
      getProbes().clear();
    },
  });

  /* -- Edit ----------------------------------------------------------------- */

  bound('Undo', {
    id: 'edit.undo',
    section: 'Edit',
    title: 'Undo',
    keywords: ['revert', 'back', 'history'],
    icon: Undo2,
    hint: countHint(context.undoDepth),
  });

  bound('Redo', {
    id: 'edit.redo',
    section: 'Edit',
    title: 'Redo',
    keywords: ['forward', 'again', 'history'],
    icon: Redo2,
    hint: countHint(context.redoDepth),
  });

  bound('Select all', {
    id: 'edit.select-all',
    section: 'Edit',
    title: 'Select all',
    keywords: ['everything', 'neurons', 'synapses', 'whole circuit'],
    icon: BoxSelect,
  });

  bound('Clear selection', {
    id: 'edit.clear-selection',
    section: 'Edit',
    title: 'Clear selection',
    keywords: ['deselect', 'none', 'nothing'],
    icon: X,
    hint: selectionHint,
  });

  bound('Delete selection', {
    id: 'edit.delete-selection',
    section: 'Edit',
    title: 'Delete selection',
    keywords: ['remove', 'erase', 'destroy', 'cut'],
    icon: Trash2,
    hint: countHint(context.selectionCount + context.synapseCount),
  });

  /* -- View ----------------------------------------------------------------- */

  bound('Toggle grid', {
    id: 'view.grid',
    section: 'View',
    title: 'Toggle grid',
    keywords: ['floor', 'ground', 'reference', 'lines'],
    icon: LayoutGrid,
    toggled: render.gridVisible,
  });

  commands.push({
    id: 'view.axons',
    section: 'View',
    title: 'Toggle axons',
    keywords: ['connections', 'edges', 'wires', 'projections', 'synapses'],
    icon: Waypoints,
    toggled: render.showAxons,
    run: () => {
      const state = useEditor.getState();
      state.setRenderSettings({ showAxons: !state.circuit.render.showAxons });
    },
  });

  commands.push({
    id: 'view.dendrites',
    section: 'View',
    title: 'Toggle dendrites',
    keywords: ['branches', 'arbor', 'morphology', 'processes'],
    icon: GitBranch,
    toggled: render.showDendrites,
    run: () => {
      const state = useEditor.getState();
      state.setRenderSettings({ showDendrites: !state.circuit.render.showDendrites });
    },
  });

  commands.push({
    id: 'view.particles',
    section: 'View',
    title: 'Toggle spike particles',
    keywords: ['impulses', 'sparks', 'effects', 'traffic'],
    icon: Sparkles,
    toggled: render.showParticles,
    run: () => {
      const state = useEditor.getState();
      state.setRenderSettings({ showParticles: !state.circuit.render.showParticles });
    },
  });

  for (const mode of COLOR_MODES) {
    commands.push({
      id: `view.colour.${mode}`,
      section: 'View',
      title: `Colour by ${COLOR_MODE_LABELS[mode].toLowerCase()}`,
      keywords: ['color', 'colour', 'mode', 'tint', 'hue', 'palette', mode],
      icon: Palette,
      selected: render.colorMode === mode,
      run: () => useEditor.getState().setRenderSettings({ colorMode: mode }),
    });
  }

  // `Frame selection` is deliberately absent. Its descriptor only publishes a
  // `neuroforge:shortcut` event for whoever owns the camera, and the camera
  // lives inside `viewport.tsx`, which registers no listener for it — so the
  // row would open the palette, close it and do nothing. It belongs here again
  // as soon as the viewport handles the `frame-selection` action.

  /* -- Panels --------------------------------------------------------------- */

  commands.push({
    id: 'panels.inspector',
    section: 'Panels',
    title: 'Toggle inspector',
    keywords: ['neuron', 'parameters', 'properties', 'membrane', 'sidebar'],
    icon: SlidersHorizontal,
    toggled: context.inspectorOpen,
    run: () => useEditor.getState().togglePanel('inspector'),
  });

  commands.push({
    id: 'panels.builder',
    section: 'Panels',
    title: 'Toggle builder',
    keywords: ['ai', 'prompt', 'generate', 'populations', 'assistant'],
    icon: Wand2,
    toggled: context.builderOpen,
    run: () => useEditor.getState().togglePanel('builder'),
  });

  commands.push({
    id: 'panels.library',
    section: 'Panels',
    title: 'Toggle library',
    keywords: ['saved', 'circuits', 'export', 'import', 'snapshots'],
    icon: Library,
    toggled: context.libraryOpen,
    run: () => useEditor.getState().togglePanel('library'),
  });

  /* -- Circuit -------------------------------------------------------------- */

  commands.push({
    id: 'circuit.new',
    section: 'Circuit',
    title: 'New circuit',
    keywords: ['empty', 'blank', 'reset document', 'start over'],
    icon: FilePlus2,
    run: () => {
      useEditor.getState().newCircuit();
      pushToast({
        tone: 'success',
        title: 'New circuit',
        description: 'The previous circuit keeps its own id and stays in the library.',
      });
    },
  });

  return commands;
}

/* ----------------------------------------------------------------- ranking -- */

interface Ranked {
  command: PaletteCommand;
  positions: readonly number[];
  score: number;
}

/** Best alignment of one term against a command's title, keywords or section. */
function rankTerm(term: string, command: PaletteCommand): FuzzyMatch | null {
  const title = fuzzyMatch(term, command.title);
  let score = title === null ? UNREACHABLE : title.score - command.title.length * LENGTH_PENALTY;

  const demotion = term.length * KEYWORD_PENALTY;
  const consider = (text: string): void => {
    const hit = fuzzyMatch(term, text);
    if (hit === null) return;
    const demoted = hit.score - text.length * LENGTH_PENALTY - demotion;
    if (demoted > score) score = demoted;
  };

  for (const keyword of command.keywords) consider(keyword);
  // The section name is searchable too, so typing "view" surfaces the whole
  // group. Matched here rather than copied into every command's keyword list,
  // which would rebuild all of those lists on every store change.
  consider(command.section);

  if (score <= REACHABLE) return null;
  // Only the title is rendered, so only its alignment can be highlighted.
  return { score, positions: title?.positions ?? NO_POSITIONS };
}

/**
 * Every whitespace-separated term must match, and the scores add. Terms are
 * matched independently so that words the user knows belong together but cannot
 * remember the order of — "mode colour" — still find the command.
 */
function rank(terms: readonly string[], command: PaletteCommand): Ranked | null {
  let score = 0;
  const positions = new Set<number>();
  for (const term of terms) {
    const hit = rankTerm(term, command);
    if (hit === null) return null;
    score += hit.score;
    for (const position of hit.positions) positions.add(position);
  }
  return { command, positions: [...positions].sort((a, b) => a - b), score };
}

interface Group {
  section: Section;
  results: Ranked[];
  /** Index of this group's first row in the flattened keyboard order. */
  start: number;
}

function group(
  terms: readonly string[],
  commands: readonly PaletteCommand[],
): { groups: Group[]; flat: Ranked[] } {
  const matched: Ranked[] = [];
  if (terms.length === 0) {
    for (const command of commands) {
      matched.push({ command, positions: NO_POSITIONS, score: 0 });
    }
  } else {
    for (const command of commands) {
      const hit = rank(terms, command);
      if (hit !== null) matched.push(hit);
    }
    matched.sort((a, b) => b.score - a.score);
  }

  // Insertion order is the ranked order, so with a query the best-matching
  // section leads and without one the declaration order stands.
  const bySection = new Map<Section, Ranked[]>();
  for (const entry of matched) {
    const existing = bySection.get(entry.command.section);
    if (existing === undefined) bySection.set(entry.command.section, [entry]);
    else existing.push(entry);
  }

  const groups: Group[] = [];
  const flat: Ranked[] = [];
  for (const [section, results] of bySection) {
    groups.push({ section, results, start: flat.length });
    for (const result of results) flat.push(result);
  }
  return { groups, flat };
}

/* --------------------------------------------------------------- rendering -- */

const LIST_ID = 'nf-command-palette-list';
const PAGE_JUMP = 8;
const NO_COMMANDS: readonly PaletteCommand[] = [];

function Highlighted({ text, positions }: { text: string; positions: readonly number[] }) {
  if (positions.length === 0) return <>{text}</>;

  const nodes: ReactNode[] = [];
  let cursor = 0;
  let index = 0;
  while (index < positions.length) {
    const start = positions[index];
    let end = start + 1;
    while (index + 1 < positions.length && positions[index + 1] === end) {
      end += 1;
      index += 1;
    }
    index += 1;
    if (start > cursor) nodes.push(<span key={`plain-${cursor}`}>{text.slice(cursor, start)}</span>);
    nodes.push(
      <mark key={`hit-${start}`} className="bg-transparent font-medium text-accent">
        {text.slice(start, end)}
      </mark>,
    );
    cursor = end;
  }
  if (cursor < text.length) nodes.push(<span key={`plain-${cursor}`}>{text.slice(cursor)}</span>);
  return <>{nodes}</>;
}

/**
 * The command palette.
 *
 * Opens on Mod+K from anywhere, including from inside a text field, which the
 * app's global key map deliberately skips. That binding is handled here in the
 * capture phase and stopped there: the same chord is in `buildShortcuts`, and
 * letting both handlers see one keypress would toggle the palette twice.
 */
export function CommandPalette() {
  const open = useEditor((s) => s.commandPaletteOpen);
  const render = useEditor((s) => s.circuit.render);
  const inspectorOpen = useEditor((s) => s.inspectorOpen);
  const builderOpen = useEditor((s) => s.builderOpen);
  const libraryOpen = useEditor((s) => s.libraryOpen);
  const selection = useEditor((s) => s.selection);
  const selectedSynapses = useEditor((s) => s.selectedSynapses);
  const undoDepth = useEditor((s) => s.undoDepth);
  const redoDepth = useEditor((s) => s.redoDepth);

  const [mounted, setMounted] = useState(false);
  const [entered, setEntered] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const [running, setRunning] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);

  const shortcuts = useMemo(() => {
    const map = new Map<string, Shortcut>();
    // Delete and Backspace are two bindings of one command; the first wins.
    for (const shortcut of buildShortcuts()) {
      if (!map.has(shortcut.label)) map.set(shortcut.label, shortcut);
    }
    return map;
  }, []);

  // Nothing is built while the palette is shut. These subscriptions fire on
  // every selection change — including each pointer move of a drag-select — and
  // rebuilding two dozen command objects to render `null` with them is work no
  // one can see.
  const commands = useMemo(
    () =>
      open
        ? buildCommands({
            shortcuts,
            render,
            inspectorOpen,
            builderOpen,
            libraryOpen,
            selectionCount: selection.length,
            synapseCount: selectedSynapses.length,
            undoDepth,
            redoDepth,
            running,
          })
        : NO_COMMANDS,
    [
      open,
      shortcuts,
      render,
      inspectorOpen,
      builderOpen,
      libraryOpen,
      selection.length,
      selectedSynapses.length,
      undoDepth,
      redoDepth,
      running,
    ],
  );

  const { groups, flat } = useMemo(() => {
    const terms = query.toLowerCase().split(/\s+/).filter((term) => term.length > 0);
    return group(terms, commands);
  }, [query, commands]);

  // Derived rather than corrected in an effect, so a list that shrinks under the
  // cursor never renders a frame pointing at a row that is no longer there.
  const activeIndex = flat.length === 0 ? 0 : Math.min(active, flat.length - 1);

  const close = useCallback(() => {
    useEditor.getState().togglePanel('commandPalette', false);
  }, []);

  const invoke = useCallback((command: PaletteCommand) => {
    // The palette closes before the command runs. "Clear selection" reuses the
    // Escape descriptor, which dismisses an open palette instead of clearing
    // anything, so it has to see a palette that is already closed.
    useEditor.getState().togglePanel('commandPalette', false);
    try {
      command.run();
    } catch (error) {
      // The palette reaches the simulation engine, whose integrator may be a
      // WebGPU pipeline that failed to build. A throw here would otherwise
      // escape a React event handler and take the whole tree down, from a
      // surface the user opened expecting to be able to back out of it.
      pushToast({
        tone: 'danger',
        title: `${command.title} failed`,
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }, []);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!open) {
      setEntered(false);
      return;
    }
    setQuery('');
    setActive(0);
    // The engine's flag is the only truth about the transport; read it fresh
    // rather than mirroring it, so the row cannot describe a stale state.
    setRunning(getEngine().running);
    const frame = requestAnimationFrame(() => setEntered(true));
    return () => cancelAnimationFrame(frame);
  }, [open]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      const key = event.key.toLowerCase();
      // Either modifier opens it, so a muscle-memory Ctrl+K on a Mac still works.
      if (key === 'k' && (event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey) {
        event.preventDefault();
        event.stopImmediatePropagation();
        useEditor.getState().togglePanel('commandPalette');
        return;
      }
      if (key === 'escape' && useEditor.getState().commandPaletteOpen) {
        event.preventDefault();
        event.stopImmediatePropagation();
        useEditor.getState().togglePanel('commandPalette', false);
      }
    };
    // Capture on the window runs before every other listener in the document,
    // which is what makes this binding independent of mount order.
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, []);

  useEffect(() => {
    if (!open || !mounted) return;
    const restoreTo = document.activeElement as HTMLElement | null;
    inputRef.current?.focus();

    const onFocusIn = (event: Event): void => {
      const container = containerRef.current;
      const target = event.target as HTMLElement | null;
      if (container === null || target === null || container.contains(target)) return;
      inputRef.current?.focus();
    };
    document.addEventListener('focusin', onFocusIn);

    return () => {
      document.removeEventListener('focusin', onFocusIn);
      if (restoreTo !== null && restoreTo.isConnected) restoreTo.focus();
    };
  }, [open, mounted]);

  useEffect(() => {
    if (!open) {
      // Rows unmount with a null ref, so this only drops the empty slots — but
      // it also keeps the array from staying as long as the longest list ever
      // shown for the rest of the session.
      rowRefs.current.length = 0;
      return;
    }
    rowRefs.current.length = flat.length;
    rowRefs.current[activeIndex]?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex, open, flat]);

  const move = useCallback(
    (delta: number, wrap: boolean) => {
      setActive((current) => {
        const count = flat.length;
        if (count === 0) return 0;
        const from = Math.min(current, count - 1);
        if (!wrap) return Math.min(count - 1, Math.max(0, from + delta));
        return (((from + delta) % count) + count) % count;
      });
    },
    [flat.length],
  );

  const onInputKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLInputElement>) => {
      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault();
          move(1, true);
          return;
        case 'ArrowUp':
          event.preventDefault();
          move(-1, true);
          return;
        case 'PageDown':
          event.preventDefault();
          move(PAGE_JUMP, false);
          return;
        case 'PageUp':
          event.preventDefault();
          move(-PAGE_JUMP, false);
          return;
        case 'Home':
          event.preventDefault();
          setActive(0);
          return;
        case 'End':
          event.preventDefault();
          setActive(Math.max(0, flat.length - 1));
          return;
        case 'Enter': {
          event.preventDefault();
          const chosen = flat[activeIndex];
          if (chosen !== undefined) invoke(chosen.command);
          return;
        }
        case 'Tab':
          // The palette is the only thing focusable while it is open.
          event.preventDefault();
          return;
        default:
      }
    },
    [activeIndex, flat, invoke, move],
  );

  if (!mounted || !open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[12vh]">
      <div
        aria-hidden
        onMouseDown={close}
        className={cn(
          'absolute inset-0 bg-bg/70 backdrop-blur-md backdrop-saturate-150',
          'transition-opacity duration-200 ease-out-expo',
          entered ? 'opacity-100' : 'opacity-0',
        )}
      />

      <div
        ref={containerRef}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className={cn(
          'nf-glass-raised relative flex w-[min(38rem,92vw)] flex-col overflow-hidden rounded-panel',
          // A capped height on the flex column is what gives the scroll region a
          // resolved height to scroll within; without it the list would grow.
          'max-h-[min(64vh,32rem)]',
          'transition-[opacity,transform] duration-200 ease-out-expo',
          entered ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-1 scale-[0.985] opacity-0',
        )}
      >
        <div className="flex h-11 shrink-0 items-center gap-2.5 border-b border-hairline px-3">
          <Search size={14} strokeWidth={1.75} className="shrink-0 text-ink-faint" aria-hidden />
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded
            aria-controls={flat.length > 0 ? LIST_ID : undefined}
            aria-autocomplete="list"
            aria-label="Search commands"
            aria-activedescendant={
              flat.length > 0 ? `nf-command-palette-option-${activeIndex}` : undefined
            }
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActive(0);
            }}
            onKeyDown={onInputKeyDown}
            placeholder="Search commands…"
            spellCheck={false}
            autoComplete="off"
            className="h-full min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-faint"
          />
          <span className="nf-numeric shrink-0 text-[10px] text-ink-faint">{flat.length}</span>
          <Kbd keys="Escape" size="sm" />
        </div>

        {flat.length === 0 ? (
          <p className="px-3 py-6 text-center text-[11px] text-ink-faint">
            No command matches “{query.trim()}”.
          </p>
        ) : (
          <ScrollArea className="min-h-0 flex-1">
            <div id={LIST_ID} role="listbox" aria-label="Commands" className="py-1">
              {groups.map((entry) => (
                <div
                  key={entry.section}
                  role="group"
                  aria-labelledby={`nf-command-palette-section-${entry.section}`}
                >
                  <div
                    id={`nf-command-palette-section-${entry.section}`}
                    className="px-3 pt-2 pb-1 text-[9.5px] font-medium tracking-[0.14em] text-ink-faint uppercase"
                  >
                    {entry.section}
                  </div>
                  {entry.results.map((result, offset) => {
                    const index = entry.start + offset;
                    const isActive = index === activeIndex;
                    const Icon = result.command.icon;
                    return (
                      <div
                        key={result.command.id}
                        id={`nf-command-palette-option-${index}`}
                        role="option"
                        aria-selected={isActive}
                        ref={(node) => {
                          rowRefs.current[index] = node;
                        }}
                        onMouseMove={() => setActive(index)}
                        // Keeps the click from pulling focus out of the input.
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => invoke(result.command)}
                        className={cn(
                          'relative flex h-8 cursor-default items-center gap-2.5 px-3',
                          isActive ? 'bg-white/[0.06] text-ink' : 'text-ink-muted',
                        )}
                      >
                        {isActive ? (
                          <span aria-hidden className="absolute inset-y-0 left-0 w-px bg-accent" />
                        ) : null}
                        <Icon
                          size={13}
                          strokeWidth={1.75}
                          aria-hidden
                          className={cn('shrink-0', isActive ? 'text-accent' : 'text-ink-faint')}
                        />
                        <span className="min-w-0 flex-1 truncate text-[12px]">
                          <Highlighted text={result.command.title} positions={result.positions} />
                        </span>
                        {result.command.selected ? (
                          <Check size={12} strokeWidth={2} aria-hidden className="text-accent" />
                        ) : null}
                        {result.command.toggled !== undefined ? (
                          <span
                            className={cn(
                              'nf-numeric text-[9.5px] tracking-[0.08em] uppercase',
                              result.command.toggled ? 'text-success' : 'text-ink-faint',
                            )}
                          >
                            {result.command.toggled ? 'on' : 'off'}
                          </span>
                        ) : null}
                        {result.command.hint !== undefined ? (
                          <span className="nf-numeric text-[10px] text-ink-faint">
                            {result.command.hint}
                          </span>
                        ) : null}
                        {result.command.keys !== undefined ? (
                          <Kbd keys={result.command.keys} size="sm" />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </ScrollArea>
        )}

        <div className="flex h-8 shrink-0 items-center gap-3 border-t border-hairline px-3 text-[10px] text-ink-faint">
          <span className="flex items-center gap-1">
            <Kbd keys="ArrowUp" size="sm" />
            <Kbd keys="ArrowDown" size="sm" />
            navigate
          </span>
          <span className="flex items-center gap-1">
            <Kbd keys="Enter" size="sm" />
            run
          </span>
          <span className="ml-auto flex items-center gap-1">
            <Kbd keys="Mod+K" size="sm" />
            palette
          </span>
        </div>
      </div>
    </div>,
    document.body,
  );
}
