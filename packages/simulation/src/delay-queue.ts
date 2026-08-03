import type { DelayQueue } from '@neuroforge/shared';

/**
 * Axonal delay as a bucketed calendar queue.
 *
 * A presynaptic spike is not delivered immediately; it is filed into the bucket
 * covering its arrival time and collected when simulation time reaches it. The
 * alternative — a per-synapse ring buffer — costs memory proportional to
 * synapses times the maximum delay, which is unaffordable at the scale this has
 * to run at.
 *
 * Buckets wrap modulo `buckets`, so the representable horizon is
 * `buckets * resolution` milliseconds. Delays beyond it are clamped rather than
 * silently aliasing onto an earlier bucket, which would deliver a spike *before*
 * it was sent.
 */

/** Absolute bucket number for a time, before wrapping. */
function absoluteBucket(time: number, resolution: number): number {
  return Math.floor(time / resolution);
}

export interface DelayCursor {
  /** Absolute bucket number that has been fully drained. */
  processed: number;
}

export function createCursor(time: number, queue: DelayQueue): DelayCursor {
  return { processed: absoluteBucket(time, queue.resolution) - 1 };
}

/** Longest delay this queue can represent without aliasing. */
export function maxDelay(queue: DelayQueue): number {
  return (queue.buckets - 1) * queue.resolution;
}

/**
 * File a synaptic event for delivery at `arrivalTime`.
 *
 * Returns false when the target bucket is full. A dropped event is a visible
 * loss of signal, so callers count them rather than ignoring the result: a queue
 * that is quietly discarding half the spikes looks exactly like a network that
 * is merely quiet.
 */
export function schedule(
  queue: DelayQueue,
  synapse: number,
  amplitude: number,
  arrivalTime: number,
): boolean {
  const bucket = absoluteBucket(arrivalTime, queue.resolution) % queue.buckets;
  const index = bucket < 0 ? bucket + queue.buckets : bucket;
  const count = queue.counts[index];
  if (count >= queue.stride) return false;
  const slot = index * queue.stride + count;
  queue.entries[slot] = synapse;
  queue.amplitude[slot] = amplitude;
  queue.counts[index] = count + 1;
  return true;
}

/**
 * Drain every bucket up to and including the one containing `time`, invoking
 * `deliver` for each queued event.
 *
 * If more than `buckets` worth of simulated time passed since the last call the
 * cursor is fast-forwarded, because draining a full wrap would replay stale
 * events that have already been overwritten.
 */
export function drain(
  queue: DelayQueue,
  cursor: DelayCursor,
  time: number,
  deliver: (synapse: number, amplitude: number) => void,
): void {
  const target = absoluteBucket(time, queue.resolution);
  if (target - cursor.processed > queue.buckets) {
    cursor.processed = target - queue.buckets;
  }
  for (let absolute = cursor.processed + 1; absolute <= target; absolute += 1) {
    const index = ((absolute % queue.buckets) + queue.buckets) % queue.buckets;
    const count = queue.counts[index];
    if (count > 0) {
      const base = index * queue.stride;
      for (let i = 0; i < count; i += 1) {
        deliver(queue.entries[base + i], queue.amplitude[base + i]);
      }
      queue.counts[index] = 0;
    }
  }
  cursor.processed = target;
}

export function clearQueue(queue: DelayQueue): void {
  queue.counts.fill(0);
}
