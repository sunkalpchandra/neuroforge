/**
 * Copy-on-write drafting of a Circuit.
 *
 * Every mutation of the document goes through `runDraft`. The draft handed to
 * the mutator is a Proxy: reads are transparent, and the first write to any node
 * shallow-copies that node and each of its ancestors before touching it. Nothing
 * already in the document is ever mutated in place, so the previous Circuit
 * stays a valid, complete value and can be handed back by undo without having
 * been cloned.
 *
 * That single invariant — *document nodes are immutable once published; only a
 * draft's private copies are written* — is what lets the diff below store bare
 * references as its before/after values instead of deep clones. It holds because
 * this package is the only writer of the document, and this file is the only
 * writer inside this package.
 *
 * Proxies are bound to the object they were created from, not to the position
 * that object occupied. That distinction matters: `synapses.sort(...)` moves
 * every element, and a proxy that meant "whatever is at index 4" would silently
 * start standing for a different synapse halfway through the operation.
 */

import type { Circuit } from '@neuroforge/shared';

/** Marker read off a proxy to recover the object it stands for. */
const RAW = Symbol('neuroforge.editor.raw');

type Node = Record<string, unknown>;

interface Identified extends Node {
  id: string;
}

const COLLECTION_KEYS = [
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
const FIELD_KEYS = [
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

const EMPTY_DIFF: CircuitDiff = { collections: [], fields: [] };

export function isEmptyDiff(diff: CircuitDiff): boolean {
  return diff.collections.length === 0 && diff.fields.length === 0;
}

/* ------------------------------------------------------------------ nodes -- */

function isContainer(value: unknown): value is Node {
  return typeof value === 'object' && value !== null;
}

function shallowCopy(value: Node): Node {
  return Array.isArray(value) ? (value.slice() as unknown as Node) : { ...value };
}

function readKey(node: Node, key: string | symbol): unknown {
  return (node as unknown as Record<string | symbol, unknown>)[key];
}

/* --------------------------------------------------------------- drafting -- */

/**
 * One node of the document as the draft sees it: the object it was found as,
 * the writable copy once one exists, and where it lives so that copy can be put
 * back.
 */
interface Slot {
  readonly origin: Node;
  copy: Node | null;
  /** Null for the document root. */
  owner: Slot | null;
  /** Property name or array index under `owner`. */
  key: string;
  /** Top-level document field this slot sits under; null for the root. */
  rootKey: string | null;
  /** 0 for the document, 1 for a collection or settings block, 2 for an entity. */
  depth: number;
  proxy: object | null;
}

const COLLECTION_LOOKUP: ReadonlySet<string> = new Set<string>(COLLECTION_KEYS);

class Draft {
  readonly root: Slot;
  /** Filed under both the original object and its copy. */
  private readonly slots = new Map<Node, Slot>();
  /** Top-level document keys that were written. */
  readonly touched = new Set<string>();
  /**
   * Collections whose array itself was written — assigned wholesale, spliced,
   * sorted, pushed to. Only these need the O(collection) reconciliation in
   * `extractDiff`; everything else is a set of in-place entity edits the draft
   * already knows by name.
   */
  readonly restructured = new Set<string>();
  /** Entities copied on write, grouped by the collection holding them. */
  readonly copiedEntities = new Map<string, Slot[]>();

  constructor(base: Node) {
    this.root = {
      origin: base,
      copy: null,
      owner: null,
      key: '',
      rootKey: null,
      depth: 0,
      proxy: null,
    };
    this.slots.set(base, this.root);
  }

  get result(): Node {
    return this.root.copy ?? this.root.origin;
  }

  get changed(): boolean {
    return this.root.copy !== null;
  }

  current(slot: Slot): Node {
    return slot.copy ?? slot.origin;
  }

  /** The slot for a child node, creating or relocating it as needed. */
  child(owner: Slot, key: string, value: Node): Slot {
    const existing = this.slots.get(value);
    if (existing !== undefined) {
      // Once copied, a slot is already attached where it belongs. Before that,
      // the newest sighting wins, which is what keeps an object that was moved
      // or shared between two parents attached to the right one.
      if (existing.copy === null && existing.origin === value) {
        existing.owner = owner;
        existing.key = key;
        existing.rootKey = owner.depth === 0 ? key : owner.rootKey;
        existing.depth = owner.depth + 1;
      }
      return existing;
    }
    const slot: Slot = {
      origin: value,
      copy: null,
      owner,
      key,
      rootKey: owner.depth === 0 ? key : owner.rootKey,
      depth: owner.depth + 1,
      proxy: null,
    };
    this.slots.set(value, slot);
    return slot;
  }

  /** Copy this slot and every ancestor that is still shared, and return it. */
  writable(slot: Slot): Node {
    if (slot.copy !== null) return slot.copy;
    const copy = shallowCopy(slot.origin);
    slot.copy = copy;
    this.slots.set(copy, slot);
    if (slot.depth === 2 && slot.rootKey !== null && COLLECTION_LOOKUP.has(slot.rootKey)) {
      const siblings = this.copiedEntities.get(slot.rootKey);
      if (siblings === undefined) this.copiedEntities.set(slot.rootKey, [slot]);
      else siblings.push(slot);
    }
    if (slot.owner !== null) {
      attach(this.writable(slot.owner), slot.key, slot.origin, copy);
    }
    return copy;
  }

  markTouched(slot: Slot, key: string): void {
    if (slot.depth === 0) {
      this.touched.add(key);
      this.restructured.add(key);
      return;
    }
    const rootKey = slot.rootKey ?? key;
    this.touched.add(rootKey);
    // A write landing directly on a top-level array is a membership or ordering
    // change; a write two levels down is an edit to one entity's contents.
    if (slot.depth === 1) this.restructured.add(rootKey);
  }

  proxy(slot: Slot): object {
    if (slot.proxy !== null) return slot.proxy;

    const handler: ProxyHandler<Node> = {
      get: (_target, key, receiver: object) => {
        if (key === RAW) return this.current(slot);
        const node = this.current(slot);
        const value = readKey(node, key);
        if (typeof value === 'function') {
          // Bound to the proxy rather than the node, so array methods read and
          // write through these traps and copy on write like everything else.
          return (value as (...args: unknown[]) => unknown).bind(receiver);
        }
        if (isContainer(value) && typeof key === 'string') {
          return this.proxy(this.child(slot, key, value));
        }
        return value;
      },

      set: (_target, key, value) => {
        if (typeof key !== 'string') return false;
        this.writable(slot)[key] = detach(value);
        this.markTouched(slot, key);
        return true;
      },

      defineProperty: (_target, key, descriptor) => {
        if (typeof key !== 'string') return false;
        const next =
          'value' in descriptor ? { ...descriptor, value: detach(descriptor.value) } : descriptor;
        Reflect.defineProperty(this.writable(slot), key, next);
        this.markTouched(slot, key);
        return true;
      },

      deleteProperty: (_target, key) => {
        if (typeof key !== 'string') return false;
        delete this.writable(slot)[key];
        this.markTouched(slot, key);
        return true;
      },

      has: (_target, key) => key in this.current(slot),

      ownKeys: () => Reflect.ownKeys(this.current(slot)),

      getOwnPropertyDescriptor: (target, key) => {
        const descriptor = Reflect.getOwnPropertyDescriptor(this.current(slot), key);
        if (descriptor === undefined) return undefined;
        // A non-configurable key on the fixed target (an array's `length`) has
        // to be reported exactly as the target holds it, or the proxy
        // invariant check throws.
        const own = Reflect.getOwnPropertyDescriptor(target, key);
        if (own !== undefined && own.configurable === false) return own;
        return { ...descriptor, configurable: true };
      },
    };

    const proxy = new Proxy(slot.origin, handler);
    slot.proxy = proxy;
    return proxy;
  }
}

/**
 * Put a freshly made copy back where its original sat. The recorded key is
 * right except after a reordering, in which case the original is looked up by
 * identity; an original that is simply gone needs no reattachment.
 */
function attach(owner: Node, key: string, origin: Node, copy: Node): void {
  const held = owner[key];
  if (held === origin || held === copy) {
    owner[key] = copy;
    return;
  }
  if (Array.isArray(owner)) {
    const index = (owner as unknown as unknown[]).indexOf(origin);
    if (index >= 0) (owner as unknown as unknown[])[index] = copy;
    return;
  }
  owner[key] = copy;
}

/**
 * Strip draft proxies out of a value on its way into the document, so a
 * reference read out of the draft can never end up stored inside it.
 *
 * The overwhelmingly common case is a value with no proxy anywhere in it — an
 * array of document entities the caller assembled from the published circuit —
 * and that case must cost nothing but the scan. The copy is therefore made
 * lazily, on the first substitution rather than up front: rebuilding every node
 * eagerly and throwing the rebuild away allocated one object per entity and per
 * nested block, which on an 80 000-synapse assignment is 300 000 objects
 * discarded to discover that nothing needed replacing.
 */
function detach(value: unknown): unknown {
  if (!isContainer(value)) return value;
  const raw = readKey(value, RAW);
  if (isContainer(raw)) return raw;

  if (Array.isArray(value)) {
    let out: unknown[] | null = null;
    for (let i = 0; i < value.length; i += 1) {
      const item: unknown = value[i];
      const clean = detach(item);
      if (clean === item) continue;
      if (out === null) out = value.slice();
      out[i] = clean;
    }
    return out ?? value;
  }

  let out: Node | null = null;
  for (const key of Object.keys(value)) {
    const item = value[key];
    const clean = detach(item);
    if (clean === item) continue;
    if (out === null) out = { ...value };
    out[key] = clean;
  }
  return out ?? value;
}

/* -------------------------------------------------------------------- diff -- */

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Reconcile a collection whose array was rewritten. This is the only path that
 * costs O(collection); it compares references, never contents, and deep-copies
 * nothing.
 */
function diffCollection(
  key: CollectionKey,
  before: readonly Identified[],
  after: readonly Identified[],
): CollectionChange | null {
  if (before === after) return null;

  // Same members in the same places: only replacements, no ordering record.
  if (before.length === after.length) {
    let aligned = true;
    for (let i = 0; i < before.length; i += 1) {
      if (before[i].id !== after[i].id) {
        aligned = false;
        break;
      }
    }
    if (aligned) {
      const replaced: EntityChange[] = [];
      for (let i = 0; i < before.length; i += 1) {
        if (before[i] !== after[i]) {
          replaced.push({ id: after[i].id, before: before[i], after: after[i] });
        }
      }
      return replaced.length === 0 ? null : { key, order: null, entities: replaced };
    }
  }

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

function extractDiff(draft: Draft, before: Node, after: Node): CircuitDiff {
  const fields: FieldChange[] = [];
  for (const key of FIELD_KEYS) {
    if (!draft.touched.has(key)) continue;
    if (before[key] !== after[key]) {
      fields.push({ key, before: before[key], after: after[key] });
    }
  }

  const collections: CollectionChange[] = [];
  for (const key of COLLECTION_KEYS) {
    if (!draft.touched.has(key)) continue;

    if (!draft.restructured.has(key)) {
      // Nothing moved, so the draft's own record of which entities it copied is
      // the complete answer and the collection never has to be walked.
      const copied = draft.copiedEntities.get(key);
      if (copied === undefined || copied.length === 0) continue;
      const entities: EntityChange[] = copied.map((slot) => ({
        id: (slot.origin as Identified).id,
        before: slot.origin as Identified,
        after: slot.copy as Identified,
      }));
      collections.push({ key, order: null, entities });
      continue;
    }

    const change = diffCollection(
      key,
      before[key] as readonly Identified[],
      after[key] as readonly Identified[],
    );
    if (change !== null) collections.push(change);
  }

  return { collections, fields };
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
 * copy that is simply discarded — so a failed edit cannot corrupt the document.
 */
export function runDraft(
  base: Circuit,
  mutate: (draft: Circuit) => void,
  track: boolean,
): DraftResult {
  const root = base as unknown as Node;
  const draft = new Draft(root);

  mutate(draft.proxy(draft.root) as unknown as Circuit);

  if (!draft.changed) {
    return { circuit: base, diff: track ? EMPTY_DIFF : null };
  }

  const next = draft.result;
  if (!track) return { circuit: next as unknown as Circuit, diff: null };

  const diff = extractDiff(draft, root, next);
  if (isEmptyDiff(diff)) return { circuit: base, diff: EMPTY_DIFF };
  return { circuit: next as unknown as Circuit, diff };
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

    // Whichever diff left the ordering alone observed and preserved the other's,
    // so the surviving pair of orders is the net one.
    let order: CollectionChange['order'] = null;
    const priorOrder = prior.order;
    const changeOrder = change.order;
    if (priorOrder !== null || changeOrder !== null) {
      const beforeOrder = priorOrder !== null ? priorOrder.before : (changeOrder as NonNullable<typeof changeOrder>).before;
      const afterOrder = changeOrder !== null ? changeOrder.after : (priorOrder as NonNullable<typeof priorOrder>).after;
      order = sameOrder(beforeOrder, afterOrder) ? null : { before: beforeOrder, after: afterOrder };
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
