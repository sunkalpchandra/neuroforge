/**
 * Copy-on-write drafting of a Circuit.
 *
 * Every mutation of the document goes through `runDraft`. The draft handed to
 * the mutator is a Proxy: reads are transparent, and the first write to any path
 * shallow-copies each node along that path before touching it. Nothing that was
 * already in the document is ever mutated in place, so the previous Circuit
 * stays a valid, complete value and can be handed back by undo without having
 * been cloned.
 *
 * That single invariant — *document nodes are immutable once published; only a
 * draft's private copies are written* — is what lets the diff below store bare
 * references as its before/after values instead of deep clones. It holds because
 * this package is the only writer of the document, and this file is the only
 * writer inside this package.
 */

import type { Circuit } from '@neuroforge/shared';

/** Marker read off a proxy to recover the object it currently stands for. */
const RAW = Symbol('neuroforge.editor.raw');

type Node = Record<string, unknown>;

interface Identified extends Node {
  id: string;
}

export const COLLECTION_KEYS = [
  'neurons',
  'synapses',
  'populations',
  'projections',
  'stimuli',
  'probes',
] as const;

export type CollectionKey = (typeof COLLECTION_KEYS)[number];

/**
 * Top-level scalar and settings fields. `id` is deliberately absent: a
 * document's identity is not editable, so no command may change it.
 */
export const FIELD_KEYS = [
  'name',
  'description',
  'version',
  'createdAt',
  'updatedAt',
  'simulation',
  'camera',
  'render',
  'tags',
] as const;

export type FieldKey = (typeof FIELD_KEYS)[number];

/** One entity that appeared, disappeared or was replaced. */
export interface EntityChange {
  readonly id: string;
  /** Absent when the entity was created by this change. */
  readonly before: Identified | undefined;
  /** Absent when the entity was removed by this change. */
  readonly after: Identified | undefined;
}

export interface CollectionChange {
  readonly key: CollectionKey;
  /** Present only when membership or ordering changed. */
  readonly order: { readonly before: readonly string[]; readonly after: readonly string[] } | null;
  readonly entities: readonly EntityChange[];
}

export interface FieldChange {
  readonly key: FieldKey;
  readonly before: unknown;
  readonly after: unknown;
}

/** The minimal description of what a mutation did, in both directions. */
export interface CircuitDiff {
  readonly collections: readonly CollectionChange[];
  readonly fields: readonly FieldChange[];
}

export const EMPTY_DIFF: CircuitDiff = { collections: [], fields: [] };

export function isEmptyDiff(diff: CircuitDiff): boolean {
  return diff.collections.length === 0 && diff.fields.length === 0;
}

/* ------------------------------------------------------------------ nodes -- */

function isContainer(value: unknown): value is Node {
  return typeof value === 'object' && value !== null;
}

function isPlainObject(value: unknown): value is Node {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function shallowCopy(value: Node): Node {
  return Array.isArray(value) ? (value.slice() as unknown as Node) : { ...value };
}

function readKey(node: Node, key: string | symbol): unknown {
  return (node as unknown as Record<string | symbol, unknown>)[key];
}

function writeKey(node: Node, key: string, value: unknown): void {
  node[key] = value;
}

function resolve(root: Node, path: readonly string[]): Node {
  let node = root;
  for (let i = 0; i < path.length; i += 1) {
    node = node[path[i]] as Node;
  }
  return node;
}

function pathKey(path: readonly string[], key?: string): string {
  const joined = path.join('.');
  if (key === undefined) return joined;
  return joined.length === 0 ? key : `${joined}.${key}`;
}

/* ----------------------------------------------------------------- proxies -- */

interface Handle {
  /** The published document, never written. */
  readonly pristine: Node;
  /** The working value; identical to `pristine` until the first write. */
  current: Node;
  /** Dot-joined paths already copied for writing; `''` is the root. */
  readonly copied: Set<string>;
  readonly proxies: Map<string, object>;
}

/**
 * Copy every node from the root down to `path` that is still shared with the
 * published document, and return the writable node at the end of the path.
 */
function ensurePath(handle: Handle, path: readonly string[]): Node {
  if (!handle.copied.has('')) {
    handle.current = shallowCopy(handle.pristine);
    handle.copied.add('');
  }
  let node = handle.current;
  let key = '';
  for (let i = 0; i < path.length; i += 1) {
    const segment = path[i];
    key = key.length === 0 ? segment : `${key}.${segment}`;
    const child = node[segment];
    if (!isContainer(child)) return node;
    if (!handle.copied.has(key)) {
      const copy = shallowCopy(child);
      node[segment] = copy;
      handle.copied.add(key);
      node = copy;
    } else {
      node = child;
    }
  }
  return node;
}

/**
 * Strip draft proxies out of a value on its way into the document, so a
 * reference captured from the draft can never end up stored inside it.
 */
function detach(value: unknown): unknown {
  if (!isContainer(value)) return value;
  const raw = readKey(value, RAW);
  if (isContainer(raw)) return raw;

  if (Array.isArray(value)) {
    let changed = false;
    const out = new Array<unknown>(value.length);
    for (let i = 0; i < value.length; i += 1) {
      const item: unknown = value[i];
      const clean = detach(item);
      if (clean !== item) changed = true;
      out[i] = clean;
    }
    return changed ? out : value;
  }

  if (!isPlainObject(value)) return value;
  let changed = false;
  const out: Node = {};
  for (const key of Object.keys(value)) {
    const item = value[key];
    const clean = detach(item);
    if (clean !== item) changed = true;
    out[key] = clean;
  }
  return changed ? out : value;
}

function proxyFor(handle: Handle, path: readonly string[]): object {
  const cacheKey = pathKey(path);
  const cached = handle.proxies.get(cacheKey);
  if (cached !== undefined) return cached;

  const target = resolve(handle.current, path);

  const handler: ProxyHandler<Node> = {
    get(_target, key, receiver: object) {
      if (key === RAW) return resolve(handle.current, path);
      const node = resolve(handle.current, path);
      const value = readKey(node, key);
      if (typeof value === 'function') {
        // Bound to the proxy, not the node, so array methods read and write
        // through these traps and therefore copy on write like everything else.
        return (value as (...args: unknown[]) => unknown).bind(receiver);
      }
      if (isContainer(value) && typeof key === 'string') return proxyFor(handle, [...path, key]);
      return value;
    },

    set(_target, key, value) {
      if (typeof key !== 'string') return false;
      const node = ensurePath(handle, path);
      writeKey(node, key, detach(value));
      handle.copied.add(pathKey(path, key));
      return true;
    },

    defineProperty(_target, key, descriptor) {
      if (typeof key !== 'string') return false;
      const node = ensurePath(handle, path);
      const next =
        'value' in descriptor ? { ...descriptor, value: detach(descriptor.value) } : descriptor;
      Reflect.defineProperty(node, key, next);
      handle.copied.add(pathKey(path, key));
      return true;
    },

    deleteProperty(_target, key) {
      if (typeof key !== 'string') return false;
      const node = ensurePath(handle, path);
      delete node[key];
      handle.copied.add(pathKey(path, key));
      return true;
    },

    has(_target, key) {
      return key in resolve(handle.current, path);
    },

    ownKeys() {
      return Reflect.ownKeys(resolve(handle.current, path));
    },

    getOwnPropertyDescriptor(fixed, key) {
      const descriptor = Reflect.getOwnPropertyDescriptor(resolve(handle.current, path), key);
      if (descriptor === undefined) return undefined;
      // A non-configurable key on the target (an array's `length`) must be
      // reported exactly as the target has it or the proxy invariant trips.
      const own = Reflect.getOwnPropertyDescriptor(fixed, key);
      if (own !== undefined && own.configurable === false) return own;
      return { ...descriptor, configurable: true };
    },
  };

  const proxy = new Proxy(target, handler);
  handle.proxies.set(cacheKey, proxy);
  return proxy;
}

/* -------------------------------------------------------------------- diff -- */

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function diffCollection(
  key: CollectionKey,
  before: readonly Identified[],
  after: readonly Identified[],
): CollectionChange | null {
  if (before === after) return null;

  const previous = new Map<string, Identified>();
  for (let i = 0; i < before.length; i += 1) previous.set(before[i].id, before[i]);

  const entities: EntityChange[] = [];
  const surviving = new Set<string>();
  for (let i = 0; i < after.length; i += 1) {
    const entity = after[i];
    surviving.add(entity.id);
    const prior = previous.get(entity.id);
    if (prior === undefined) entities.push({ id: entity.id, before: undefined, after: entity });
    else if (prior !== entity) entities.push({ id: entity.id, before: prior, after: entity });
  }
  for (let i = 0; i < before.length; i += 1) {
    const entity = before[i];
    if (!surviving.has(entity.id)) {
      entities.push({ id: entity.id, before: entity, after: undefined });
    }
  }

  const beforeOrder = before.map((entity) => entity.id);
  const afterOrder = after.map((entity) => entity.id);
  const order = sameOrder(beforeOrder, afterOrder)
    ? null
    : { before: beforeOrder, after: afterOrder };

  if (order === null && entities.length === 0) return null;
  return { key, order, entities };
}

function extractDiff(before: Node, after: Node, touched: ReadonlySet<string>): CircuitDiff {
  const fields: FieldChange[] = [];
  for (const key of FIELD_KEYS) {
    if (!touched.has(key)) continue;
    if (before[key] !== after[key]) {
      fields.push({ key, before: before[key], after: after[key] });
    }
  }

  const collections: CollectionChange[] = [];
  for (const key of COLLECTION_KEYS) {
    if (!touched.has(key)) continue;
    const change = diffCollection(
      key,
      before[key] as readonly Identified[],
      after[key] as readonly Identified[],
    );
    if (change !== null) collections.push(change);
  }

  return { collections, fields };
}

/** First path segment of every write performed during a session. */
function touchedRoots(handle: Handle): Set<string> {
  const roots = new Set<string>();
  for (const key of handle.copied) {
    if (key.length === 0) continue;
    const dot = key.indexOf('.');
    roots.add(dot === -1 ? key : key.slice(0, dot));
  }
  return roots;
}

export interface DraftResult {
  readonly circuit: Circuit;
  /** Null when the session was not asked to track, empty when nothing changed. */
  readonly diff: CircuitDiff | null;
}

/**
 * Run `mutate` against a copy-on-write draft of `base`.
 *
 * If `mutate` throws, `base` is left untouched — every write landed in a private
 * copy that is simply discarded — so a failed edit can never corrupt the
 * document.
 */
export function runDraft(
  base: Circuit,
  mutate: (draft: Circuit) => void,
  track: boolean,
): DraftResult {
  const root = base as unknown as Node;
  const handle: Handle = {
    pristine: root,
    current: root,
    copied: new Set<string>(),
    proxies: new Map<string, object>(),
  };

  mutate(proxyFor(handle, []) as unknown as Circuit);

  if (handle.copied.size === 0) {
    return { circuit: base, diff: track ? EMPTY_DIFF : null };
  }

  const next = handle.current as unknown as Circuit;
  const diff = track ? extractDiff(root, handle.current, touchedRoots(handle)) : null;
  if (track && diff !== null && isEmptyDiff(diff)) {
    return { circuit: base, diff: EMPTY_DIFF };
  }
  return { circuit: next, diff };
}

/* ------------------------------------------------------------- diff replay -- */

/** Write one side of a diff into a draft. */
export function applyDiff(draft: Circuit, diff: CircuitDiff, side: 'before' | 'after'): void {
  const target = draft as unknown as Node;

  for (const field of diff.fields) {
    target[field.key] = side === 'before' ? field.before : field.after;
  }

  for (const change of diff.collections) {
    const current = target[change.key] as readonly Identified[];
    const members = new Map<string, Identified>();
    for (let i = 0; i < current.length; i += 1) members.set(current[i].id, current[i]);

    for (const entity of change.entities) {
      const value = side === 'before' ? entity.before : entity.after;
      if (value === undefined) members.delete(entity.id);
      else members.set(entity.id, value);
    }

    const order =
      change.order === null
        ? current.map((entity) => entity.id)
        : side === 'before'
          ? change.order.before
          : change.order.after;

    const next: Identified[] = [];
    for (let i = 0; i < order.length; i += 1) {
      const value = members.get(order[i]);
      // A missing id means the diff is being replayed against a document it did
      // not come from; dropping is the only non-destructive answer.
      if (value !== undefined) next.push(value);
    }
    target[change.key] = next;
  }
}

/**
 * Compose two consecutive diffs into one with the same net effect. Used when
 * coalescing merges two history entries.
 */
export function mergeDiff(first: CircuitDiff, second: CircuitDiff): CircuitDiff {
  const fields = new Map<FieldKey, FieldChange>();
  for (const field of first.fields) fields.set(field.key, field);
  for (const field of second.fields) {
    const prior = fields.get(field.key);
    fields.set(
      field.key,
      prior === undefined ? field : { key: field.key, before: prior.before, after: field.after },
    );
  }

  const collections = new Map<CollectionKey, CollectionChange>();
  for (const change of first.collections) collections.set(change.key, change);
  for (const change of second.collections) {
    const prior = collections.get(change.key);
    if (prior === undefined) {
      collections.set(change.key, change);
      continue;
    }

    const entities = new Map<string, EntityChange>();
    for (const entity of prior.entities) entities.set(entity.id, entity);
    for (const entity of change.entities) {
      const earlier = entities.get(entity.id);
      entities.set(
        entity.id,
        earlier === undefined
          ? entity
          : { id: entity.id, before: earlier.before, after: entity.after },
      );
    }

    // Whichever diff did not move the collection leaves the other's ordering as
    // the net ordering, because it observed and preserved it.
    let order: CollectionChange['order'] = null;
    if (prior.order !== null || change.order !== null) {
      order = {
        before: prior.order !== null ? prior.order.before : (change.order as NonNullable<CollectionChange['order']>).before,
        after: change.order !== null ? change.order.after : (prior.order as NonNullable<CollectionChange['order']>).after,
      };
      if (sameOrder(order.before, order.after)) order = null;
    }

    const merged: EntityChange[] = [];
    for (const entity of entities.values()) {
      if (entity.before === entity.after) continue;
      merged.push(entity);
    }
    if (order === null && merged.length === 0) collections.delete(change.key);
    else collections.set(change.key, { key: change.key, order, entities: merged });
  }

  const netFields: FieldChange[] = [];
  for (const field of fields.values()) {
    if (field.before !== field.after) netFields.push(field);
  }

  return { collections: [...collections.values()], fields: netFields };
}
