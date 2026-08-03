//! PCG32 — a small, statistically sound, fully deterministic generator.
//!
//! Determinism is a hard requirement: the same seed replayed against the same
//! circuit must reproduce a run exactly, which rules out any host entropy and
//! makes the Box-Muller spare value part of the reproducible state.

const PCG_MULTIPLIER: u64 = 6364136223846793005;
const PCG_DEFAULT_STREAM: u64 = 1442695040888963407;

/// Above this rate the Knuth product method needs too many iterations, so the
/// draw is split into independent Poissons and summed (Poisson is additive in
/// its rate, so the split is exact rather than an approximation).
const POISSON_CHUNK: f64 = 30.0;

pub struct Pcg32 {
    state: u64,
    inc: u64,
    seed: u32,
    spare_normal: f64,
    has_spare: bool,
}

impl Pcg32 {
    pub fn new(seed: u32) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: 0,
            seed,
            spare_normal: 0.0,
            has_spare: false,
        };
        rng.reset(seed);
        rng
    }

    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Re-seed and discard all derived state, including the Box-Muller spare.
    pub fn reset(&mut self, seed: u32) {
        self.seed = seed;
        self.inc = (PCG_DEFAULT_STREAM << 1) | 1;
        self.state = 0;
        self.spare_normal = 0.0;
        self.has_spare = false;
        self.next_u32();
        self.state = self.state.wrapping_add(seed as u64);
        self.next_u32();
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1). 24 significant bits, matching f32 precision.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16777216.0)
    }

    /// Uniform in [0, 1) with 53 significant bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let hi = (self.next_u32() >> 5) as u64;
        let lo = (self.next_u32() >> 6) as u64;
        ((hi << 26) | lo) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Uniform integer in [0, max_exclusive), debiased by rejection.
    pub fn next_below(&mut self, max_exclusive: u32) -> u32 {
        if max_exclusive == 0 {
            return 0;
        }
        let threshold = max_exclusive.wrapping_neg() % max_exclusive;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % max_exclusive;
            }
        }
    }

    /// Standard normal via the polar form of Box-Muller. Draws come in pairs; the
    /// unused one is cached so the stream stays cheap and reproducible.
    pub fn normal(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare_normal;
        }
        loop {
            let u = self.next_f64() * 2.0 - 1.0;
            let v = self.next_f64() * 2.0 - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                if factor.is_finite() {
                    self.spare_normal = v * factor;
                    self.has_spare = true;
                    return u * factor;
                }
            }
        }
    }

    /// Poisson deviate by Knuth's product method, split into chunks so that the
    /// iteration count stays bounded for large rates.
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        if !(lambda > 0.0) || !lambda.is_finite() {
            return 0;
        }
        let mut remaining = lambda;
        let mut total: u32 = 0;
        while remaining > 0.0 {
            let chunk = remaining.min(POISSON_CHUNK);
            remaining -= chunk;
            let limit = (-chunk).exp();
            let mut k: u32 = 0;
            let mut product = self.next_f64();
            // The loop terminates with probability 1; the cap only guards against
            // a denormal `limit` pinning it open.
            while product > limit && k < 1_000_000 {
                product *= self.next_f64();
                k += 1;
            }
            total = total.saturating_add(k);
        }
        total
    }

    /// True with probability `p`.
    #[inline]
    pub fn bernoulli(&mut self, p: f32) -> bool {
        if p >= 1.0 {
            return true;
        }
        if !(p > 0.0) {
            return false;
        }
        self.next_f32() < p
    }
}

impl Default for Pcg32 {
    fn default() -> Self {
        Pcg32::new(0x9e37_79b9)
    }
}
