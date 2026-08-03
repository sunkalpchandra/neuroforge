//! Bucketed calendar queue for axonal conduction delays.
//!
//! LAYOUT CONTRACT: mirrors the `DelayQueue` interface in
//! `packages/shared/src/buffers.ts` field for field — `buckets` time buckets of
//! `stride` entries each, a flat `entries` array of synapse indices, a parallel
//! `amplitude` array, and one occupancy `counts` entry per bucket. The defaults
//! match `allocateDelayQueue()`: 256 buckets of 0.25 ms, 64 entries each, giving
//! a 64 ms horizon.
//!
//! Arrival times are quantised down to a bucket, so an event is delivered up to
//! one `resolution` early. That is the price of O(1) scheduling and is why the
//! resolution defaults to well below a typical 1 ms delay.

pub const DEFAULT_BUCKETS: usize = 256;
pub const DEFAULT_RESOLUTION: f64 = 0.25;
pub const DEFAULT_STRIDE: usize = 64;

pub struct DelayQueue {
    buckets: usize,
    stride: usize,
    resolution: f64,
    entries: Vec<u32>,
    amplitude: Vec<f32>,
    counts: Vec<u32>,
    /// Highest absolute bucket index already drained. Absolute rather than
    /// modular so the queue survives wrapping without ambiguity.
    cursor: i64,
    dropped: u32,
    clamped: u32,
}

impl DelayQueue {
    pub fn new(buckets: usize, resolution: f64, stride: usize) -> Self {
        let buckets = buckets.max(1);
        let stride = stride.max(1);
        let resolution = if resolution.is_finite() && resolution > 0.0 {
            resolution
        } else {
            DEFAULT_RESOLUTION
        };
        DelayQueue {
            buckets,
            stride,
            resolution,
            entries: vec![0; buckets * stride],
            amplitude: vec![0.0; buckets * stride],
            counts: vec![0; buckets],
            cursor: -1,
            dropped: 0,
            clamped: 0,
        }
    }

    pub fn buckets(&self) -> usize {
        self.buckets
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Longest delay the queue can represent, in ms.
    pub fn horizon(&self) -> f64 {
        self.buckets as f64 * self.resolution
    }

    pub fn dropped(&self) -> u32 {
        self.dropped
    }

    /// Number of events whose requested arrival exceeded the horizon and were
    /// pulled back to the last representable bucket.
    pub fn clamped(&self) -> u32 {
        self.clamped
    }

    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    pub fn amplitudes(&self) -> &[f32] {
        &self.amplitude
    }

    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Drop every pending event and re-anchor the cursor to `time`.
    pub fn clear(&mut self, time: f64) {
        self.entries.fill(0);
        self.amplitude.fill(0.0);
        self.counts.fill(0);
        self.cursor = Self::absolute(time, self.resolution) - 1;
        self.dropped = 0;
        self.clamped = 0;
    }

    /// Re-anchor the cursor without discarding events. Used when simulation time
    /// is set explicitly; pending events keep their buckets.
    pub fn rebase(&mut self, time: f64) {
        self.cursor = Self::absolute(time, self.resolution) - 1;
    }

    #[inline]
    fn absolute(time: f64, resolution: f64) -> i64 {
        if !time.is_finite() {
            return 0;
        }
        (time / resolution).floor() as i64
    }

    /// Schedule `synapse` to deliver `amplitude` at absolute time `arrival`.
    ///
    /// Returns false when the target bucket is full; the event is dropped and the
    /// `dropped` counter advances. Dropping is deliberate: a full bucket means the
    /// network is firing harder than the queue was sized for, and losing an event
    /// is preferable to reallocating on the hot path or writing out of bounds.
    pub fn schedule(&mut self, arrival: f64, synapse: u32, amplitude: f32) -> bool {
        let mut abs = Self::absolute(arrival, self.resolution);
        let earliest = self.cursor + 1;
        let latest = self.cursor + self.buckets as i64;
        if abs > latest {
            abs = latest;
            self.clamped = self.clamped.saturating_add(1);
        } else if abs < earliest {
            // Sub-resolution delay, or an arrival inside the bucket just drained:
            // deliver at the next opportunity rather than losing the event.
            abs = earliest;
        }
        let bucket = abs.rem_euclid(self.buckets as i64) as usize;
        let count = self.counts[bucket] as usize;
        if count >= self.stride {
            self.dropped = self.dropped.saturating_add(1);
            return false;
        }
        let slot = bucket * self.stride + count;
        self.entries[slot] = synapse;
        self.amplitude[slot] = amplitude;
        self.counts[bucket] = count as u32 + 1;
        true
    }

    /// Deliver every event whose bucket has been reached by `now`.
    ///
    /// Buckets are walked one at a time, so a `dt` larger than the resolution
    /// still drains every intervening bucket exactly once and in order.
    pub fn drain_due<F: FnMut(u32, f32)>(&mut self, now: f64, mut deliver: F) {
        let target = Self::absolute(now, self.resolution);
        if target < self.cursor {
            // Time moved backwards (a reset or a scrub). Re-anchor rather than
            // spinning through a full wrap of buckets.
            self.cursor = target;
            return;
        }
        // A single pass can never usefully cover more than one wrap.
        if target - self.cursor > self.buckets as i64 {
            self.cursor = target - self.buckets as i64;
        }
        while self.cursor < target {
            self.cursor += 1;
            let bucket = self.cursor.rem_euclid(self.buckets as i64) as usize;
            let count = (self.counts[bucket] as usize).min(self.stride);
            let base = bucket * self.stride;
            for k in 0..count {
                deliver(self.entries[base + k], self.amplitude[base + k]);
            }
            self.counts[bucket] = 0;
        }
    }

    /// Total events currently queued. Diagnostic only; walks every bucket.
    pub fn pending(&self) -> u32 {
        self.counts.iter().copied().fold(0u32, u32::saturating_add)
    }
}
