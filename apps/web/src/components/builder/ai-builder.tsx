'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, KeyRound, Send, Sparkles, X } from 'lucide-react';
import {
  Badge,
  Button,
  Field,
  IconButton,
  Panel,
  PanelHeader,
  PanelSection,
  ScrollArea,
  Select,
  SelectItem,
  Separator,
  Spinner,
  pushToast,
} from '@neuroforge/ui';
import { useEditor } from '@neuroforge/editor';
import {
  DEFAULT_MODELS,
  loadCredentials,
  planLocally,
  storeCredentials,
  streamCircuitPlan,
  validatePlan,
} from '@neuroforge/ai';
import type { AiCredentials, AiPlan, AiProvider } from '@neuroforge/ai';

import { applyPlan } from '@/lib/apply-plan';

const EXAMPLES = [
  'Create a hippocampal CA3 recurrent circuit with 200 excitatory neurons and inhibitory basket cells',
  'Make this oscillate at gamma frequency',
  'Add 50 fast-spiking interneurons',
  'Switch everything to Hodgkin-Huxley',
];

/** Human-readable one-line description of an action, for the plan preview. */
function describe(action: AiPlan['actions'][number]): string {
  switch (action.type) {
    case 'create-population':
      return `Create ${action.spec.size} ${action.spec.polarity} "${action.spec.name}"`;
    case 'connect-populations':
      return `Connect ${action.spec.sourceName} → ${action.spec.targetName}`;
    case 'tune-population':
      return `Tune ${action.name}`;
    case 'tune-projection':
      return `Tune projection ${action.name}`;
    case 'add-stimulus':
      return `Stimulate ${action.targetPopulation}`;
    case 'set-simulation':
      return 'Adjust simulation settings';
    case 'set-render':
      return 'Adjust scene settings';
    case 'clear':
      return 'Clear the circuit';
    default:
      return 'Unknown action';
  }
}

/**
 * Prompt-to-circuit builder.
 *
 * Works with no API key at all: the offline planner is a real deterministic
 * parser, not a fallback stub, so the feature is usable before a user has
 * decided whether to trust a key to the browser.
 */
export function AiBuilder() {
  const open = useEditor((s) => s.builderOpen);
  const togglePanel = useEditor((s) => s.togglePanel);
  const circuit = useEditor((s) => s.circuit);

  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [stream, setStream] = useState('');
  const [plan, setPlan] = useState<AiPlan | null>(null);
  const [credentials, setCredentials] = useState<AiCredentials | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [keyDraft, setKeyDraft] = useState('');
  const [provider, setProvider] = useState<AiProvider>('anthropic');
  const abort = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    void loadCredentials().then((loaded) => {
      if (!loaded) return;
      setCredentials(loaded);
      setProvider(loaded.provider);
    });
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => () => abort.current?.abort(), []);

  const run = useCallback(async () => {
    const text = prompt.trim();
    if (!text || busy) return;

    setBusy(true);
    setStream('');
    setPlan(null);

    try {
      let draft: AiPlan;
      if (credentials && credentials.apiKey) {
        abort.current?.abort();
        const controller = new AbortController();
        abort.current = controller;
        let last: AiPlan | null = null;
        for await (const event of streamCircuitPlan({
          prompt: text,
          circuit,
          credentials,
          signal: controller.signal,
        })) {
          if (event.kind === 'text' && event.text) setStream((s) => s + event.text);
          if (event.kind === 'plan' && event.plan) last = event.plan;
          if (event.kind === 'error') {
            pushToast({ tone: 'danger', title: 'Planning failed', description: event.error });
          }
        }
        // A hosted model that returned prose but no tool call still leaves the
        // user with something usable rather than nothing.
        draft = last ?? planLocally(text, circuit);
      } else {
        draft = planLocally(text, circuit);
      }

      const checked = validatePlan(draft, circuit);
      setPlan(checked.plan);
      for (const error of checked.errors) {
        pushToast({ tone: 'warning', title: 'Adjusted', description: error });
      }
    } catch (error) {
      pushToast({
        tone: 'danger',
        title: 'Planning failed',
        description: (error as Error).message,
      });
    } finally {
      setBusy(false);
    }
  }, [prompt, busy, credentials, circuit]);

  const commit = useCallback(() => {
    if (!plan) return;
    const result = applyPlan(plan);
    for (const error of result.errors) {
      pushToast({ tone: 'warning', title: 'Skipped', description: error });
    }
    pushToast({
      tone: result.skipped > 0 ? 'warning' : 'success',
      title: `Applied ${result.applied} change${result.applied === 1 ? '' : 's'}`,
      description: result.skipped > 0 ? `${result.skipped} skipped` : plan.summary,
    });
    setPlan(null);
    setStream('');
    setPrompt('');
  }, [plan]);

  const saveKey = useCallback(async () => {
    const key = keyDraft.trim();
    const next: AiCredentials | null = key
      ? { provider, apiKey: key, model: DEFAULT_MODELS[provider] }
      : null;
    await storeCredentials(next);
    setCredentials(next);
    setKeyDraft('');
    setShowKey(false);
    pushToast({
      tone: 'success',
      title: next ? 'Key saved' : 'Key cleared',
      description: next
        ? 'Stored in this browser only; it never reaches a NeuroForge server.'
        : 'The offline planner will be used.',
    });
  }, [keyDraft, provider]);

  if (!open) return null;

  return (
    <Panel className="pointer-events-auto absolute top-3 bottom-3 left-3 flex w-[380px] flex-col">
      <PanelHeader
        title="Builder"
        subtitle={credentials?.apiKey ? `via ${credentials.provider}` : 'offline planner'}
        icon={<Sparkles />}
        actions={
          <>
            <IconButton
              label="API key settings"
              size="sm"
              variant={showKey ? 'secondary' : 'ghost'}
              onClick={() => setShowKey((v) => !v)}
            >
              <KeyRound />
            </IconButton>
            <IconButton label="Close builder" size="sm" onClick={() => togglePanel('builder', false)}>
              <X />
            </IconButton>
          </>
        }
      />

      {showKey ? (
        <PanelSection label="Model access" flush>
          <Field
            label="Provider"
            description="Your key is stored in this browser's IndexedDB and sent only to the provider you choose."
          >
            <Select value={provider} onValueChange={(v) => setProvider(v as AiProvider)}>
              <SelectItem value="anthropic">Anthropic</SelectItem>
              <SelectItem value="openai">OpenAI</SelectItem>
            </Select>
          </Field>
          <Field label="API key" className="mt-2">
            <input
              type="password"
              value={keyDraft}
              onChange={(e) => setKeyDraft(e.target.value)}
              placeholder={credentials?.apiKey ? '•••••••• stored' : 'sk-…'}
              className="w-full rounded-control border border-hairline bg-bg px-2 py-1.5 font-mono text-[12px] text-ink outline-none focus-visible:border-accent"
            />
          </Field>
          <div className="mt-2 flex gap-2">
            <Button size="sm" onClick={saveKey}>
              {keyDraft.trim() ? 'Save key' : 'Clear key'}
            </Button>
          </div>
        </PanelSection>
      ) : null}

      <Separator />

      <ScrollArea className="min-h-0 flex-1">
        <PanelSection label="Describe a circuit" flush>
          <textarea
            ref={inputRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                void run();
              }
            }}
            rows={3}
            placeholder="Create a hippocampal CA3 recurrent circuit with 200 excitatory neurons and inhibitory basket cells"
            className="w-full resize-y rounded-control border border-hairline bg-bg px-2.5 py-2 text-[12px] leading-relaxed text-ink outline-none placeholder:text-ink-faint focus-visible:border-accent"
          />
          <div className="mt-2 flex items-center gap-2">
            <Button size="sm" onClick={() => void run()} disabled={busy || !prompt.trim()}>
              {busy ? <Spinner /> : <Send />}
              {busy ? 'Planning…' : 'Plan'}
            </Button>
            <span className="nf-numeric text-[10px] text-ink-faint">⌘↵</span>
          </div>
        </PanelSection>

        {!prompt && !plan ? (
          <PanelSection label="Try">
            <ul className="flex flex-col gap-1">
              {EXAMPLES.map((example) => (
                <li key={example}>
                  <button
                    type="button"
                    onClick={() => setPrompt(example)}
                    className="w-full rounded-control px-2 py-1.5 text-left text-[11px] leading-snug text-ink-muted transition-colors hover:bg-panel-raised hover:text-ink focus-visible:bg-panel-raised"
                  >
                    {example}
                  </button>
                </li>
              ))}
            </ul>
          </PanelSection>
        ) : null}

        {stream ? (
          <PanelSection label="Reasoning">
            <p className="text-[11px] leading-relaxed whitespace-pre-wrap text-ink-muted">{stream}</p>
          </PanelSection>
        ) : null}

        {plan ? (
          <PanelSection label="Proposed changes">
            <p className="mb-2 text-[11px] leading-relaxed text-ink">{plan.summary}</p>

            {plan.warnings.length > 0 ? (
              <div className="mb-2 flex flex-col gap-1">
                {plan.warnings.map((warning) => (
                  <p
                    key={warning}
                    className="flex items-start gap-1.5 text-[11px] leading-snug text-warning"
                  >
                    <AlertTriangle className="mt-px size-3 shrink-0" />
                    {warning}
                  </p>
                ))}
              </div>
            ) : null}

            <ul className="mb-3 flex flex-col gap-1">
              {plan.actions.map((action, index) => (
                <li
                  key={`${action.type}-${index}`}
                  className="flex items-center justify-between rounded-control bg-panel-raised px-2 py-1 text-[11px]"
                >
                  <span className="truncate text-ink">{describe(action)}</span>
                  <Badge variant="secondary" size="sm">
                    {action.type.replace('-', ' ')}
                  </Badge>
                </li>
              ))}
            </ul>

            <div className="flex gap-2">
              <Button size="sm" onClick={commit} disabled={plan.actions.length === 0}>
                Apply {plan.actions.length} change{plan.actions.length === 1 ? '' : 's'}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setPlan(null)}>
                Discard
              </Button>
            </div>
          </PanelSection>
        ) : null}
      </ScrollArea>
    </Panel>
  );
}
