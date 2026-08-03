/**
 * WGSL chunks shared by the three compute pipelines in this package.
 *
 * The structs below are the GPU mirror of the structure-of-arrays layout in
 * `@neuroforge/shared/buffers`. The GPU side is array-of-structs because a
 * WebGPU device only guarantees eight storage buffers per shader stage, which is
 * fewer than the number of SoA columns the simulation owns; grouping the columns
 * that a single invocation touches into one 32-byte record also means each
 * thread reads one contiguous cache line instead of striding across six buffers.
 *
 * Byte layouts are documented per struct because the host has to interleave the
 * SoA columns into them on upload and split them back out on readback.
 */

/** Workgroup size every compute entry point in this package declares. */
export const WGSL_WORKGROUP_SIZE = 64;

/**
 * Fixed-point scale used when synaptic current is accumulated with atomics.
 *
 * WGSL has no atomic float, so `SYNAPSE_PROPAGATE_WGSL` accumulates into an
 * `array<atomic<i32>>` of current in units of 1/1024 pA. That gives a
 * quantisation of ~0.001 pA and a headroom of +-2.09e6 pA before the i32
 * saturates, which is several orders of magnitude above the largest total
 * synaptic drive a single neuron can plausibly receive.
 */
export const SYNAPSE_CURRENT_SCALE = 1024;

/** Numeric guards and the fixed-point conversion shared by every kernel. */
export const WGSL_CONSTANTS = /* wgsl */ `
const WORKGROUP_SIZE : u32 = ${WGSL_WORKGROUP_SIZE}u;

// Smallest conductance and time constant treated as non-zero. Both exist so a
// user-authored parameter of exactly zero degrades to a sane limit instead of
// producing an infinity that then poisons every downstream frame.
const MIN_CONDUCTANCE : f32 = 1.0e-6;
const MIN_TAU : f32 = 1.0e-6;
const EPSILON : f32 = 1.0e-9;

// Membrane potential is held inside this range as a divergence guard. It is far
// outside every model's physiological span, so it never alters a healthy
// trajectory; it only stops a runaway parameter set from producing an infinity.
const V_LIMIT : f32 = 1000.0;

const CURRENT_SCALE : f32 = ${SYNAPSE_CURRENT_SCALE.toFixed(1)};
const INV_CURRENT_SCALE : f32 = ${(1 / SYNAPSE_CURRENT_SCALE).toPrecision(12)};
// i32 saturation point, kept a little inside 2^31 so rounding cannot overflow.
const CURRENT_FIXED_LIMIT : f32 = 2.147e9;

fn toFixedCurrent(currentPa : f32) -> i32 {
  let scaled = clamp(currentPa * CURRENT_SCALE, -CURRENT_FIXED_LIMIT, CURRENT_FIXED_LIMIT);
  return i32(round(scaled));
}
`;

/** Numerics: guarded exponentials and the two integration schemes. */
export const WGSL_MATH = /* wgsl */ `
const INTEGRATOR_FORWARD_EULER : u32 = 0u;
const INTEGRATOR_EXPONENTIAL : u32 = 1u;

/** exp() with the argument clamped to the range where f32 stays finite. */
fn safeExp(x : f32) -> f32 {
  return exp(clamp(x, -60.0, 60.0));
}

/** Fraction of the way to the target an exponential relaxation travels in dt. */
fn relaxFactor(dt : f32, tau : f32) -> f32 {
  if (tau <= MIN_TAU) {
    return 1.0;
  }
  return 1.0 - exp(-dt / tau);
}

/** Exponential Euler for dx/dt = (target - x) / tau. Exact for constant target. */
fn relax(x : f32, target : f32, dt : f32, tau : f32) -> f32 {
  return x + (target - x) * relaxFactor(dt, tau);
}

/**
 * Integrate one linear relaxation with the scheme the document asks for.
 *
 * Exponential Euler is the default because it is unconditionally stable for the
 * leak term: the membrane can never overshoot its own steady state no matter how
 * large dt is, which is what lets the app run at dt = 0.1 ms with stiff
 * conductances present.
 */
fn integrateLinear(x : f32, target : f32, dt : f32, tau : f32, mode : u32) -> f32 {
  if (mode == INTEGRATOR_EXPONENTIAL) {
    return relax(x, target, dt, tau);
  }
  return x + (target - x) * (dt / max(tau, EPSILON));
}

/** Exponential Euler for a gate obeying dx/dt = alpha*(1-x) - beta*x. */
fn gateStep(x : f32, alpha : f32, beta : f32, dt : f32) -> f32 {
  let sum = alpha + beta;
  if (sum <= EPSILON) {
    return x;
  }
  let steady = alpha / sum;
  return steady + (x - steady) * exp(-sum * dt);
}
`;

/** PCG hash and the derived uniform / normal deviates. Stateless and stable. */
export const WGSL_RANDOM = /* wgsl */ `
const TWO_PI : f32 = 6.283185307179586;
// 1 / 2^24. Twenty-four bits is the widest integer range f32 represents exactly,
// so a quotient built from that many bits is always strictly below 1. Using all
// 32 bits instead rounds the top of the range up to exactly 1.0, which breaks
// the half-open interval randomUnit promises and can hand log() a zero.
const INV_U24 : f32 = 5.9604644775390625e-8;

fn pcgHash(input : u32) -> u32 {
  let state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hashCombine(a : u32, b : u32) -> u32 {
  return pcgHash(a ^ pcgHash(b));
}

/** Uniform deviate in [0,1). */
fn randomUnit(seed : u32) -> f32 {
  return f32(pcgHash(seed) >> 8u) * INV_U24;
}

/** Standard normal deviate via Box-Muller; one of the two outputs is kept. */
fn randomNormal(seed : u32) -> f32 {
  let u1 = max(randomUnit(seed), 1.0e-7);
  let u2 = randomUnit(seed ^ 0x9e3779b9u);
  return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}
`;

/**
 * Packed metadata byte lanes.
 *
 * `NeuronStatic.meta` and `SynapseStatic.meta` each fold four `Uint8Array`
 * columns from the shared buffers into one u32 so they can travel in a struct:
 *   byte 0 - neuron model code / synapse receptor code
 *   byte 1 - neuron polarity / synapse plasticity code
 *   byte 2 - enabled
 *   byte 3 - render flags (NEURON_FLAG bits), unused by synapses
 */
export const WGSL_META = /* wgsl */ `
const META_KIND_SHIFT : u32 = 0u;
const META_SUBKIND_SHIFT : u32 = 8u;
const META_ENABLED_SHIFT : u32 = 16u;
const META_FLAGS_SHIFT : u32 = 24u;

fn metaByte(meta : u32, shift : u32) -> u32 {
  return (meta >> shift) & 0xffu;
}

fn metaEnabled(meta : u32) -> bool {
  return metaByte(meta, META_ENABLED_SHIFT) != 0u;
}
`;

/**
 * Per-neuron records.
 *
 * NeuronDynamic (32 B, everything the integrator mutates):
 *   0  v                 mV
 *   4  w                 Izhikevich u / AdEx w / Morris-Lecar w
 *   8  m                 Hodgkin-Huxley sodium activation
 *   12 h                 Hodgkin-Huxley sodium inactivation
 *   16 n                 Hodgkin-Huxley potassium activation
 *   20 calcium           display-only, arbitrary units
 *   24 lastSpike         ms, -inf when the neuron has never fired
 *   28 refractoryUntil   ms
 *
 * NeuronStatic (16 B, constant across a step):
 *   0  meta   packed model | polarity | enabled | flags
 *   4  iExt   pA
 *   8  bias   pA
 *   12 noise  pA (per-neuron standard deviation)
 *
 * NeuronOutput (16 B, produced by the integrator, read by every other stage):
 *   0  spike      1 on the step the neuron fired
 *   4  spikeCount total since reset
 *   8  flash      0..1 render envelope
 *   12 rate       Hz, exponentially smoothed
 */
export const WGSL_NEURON_STRUCTS = /* wgsl */ `
struct NeuronDynamic {
  v : f32,
  w : f32,
  m : f32,
  h : f32,
  n : f32,
  calcium : f32,
  lastSpike : f32,
  refractoryUntil : f32,
}

struct NeuronStatic {
  meta : u32,
  iExt : f32,
  bias : f32,
  noise : f32,
}

struct NeuronOutput {
  spike : u32,
  spikeCount : u32,
  flash : f32,
  rate : f32,
}
`;

/**
 * Per-synapse records.
 *
 * SynapseStatic (40 B, topology and kinetics; only the editor changes these):
 *   0  pre          presynaptic neuron slot
 *   4  post         postsynaptic neuron slot
 *   8  meta         packed receptor | plasticity | enabled
 *   12 delay        ms, consumed by the delay stage rather than by these kernels
 *   16 releaseProb  0..1, consumed by the delay stage
 *   20 tauRise      ms
 *   24 tauDecay     ms
 *   28 eRev         mV
 *   32 mgBlock      NMDA magnesium block strength; 0 disables the nonlinearity
 *   36 arc          spline sag in world units, used by the particle kernels
 *
 * SynapseDynamic (20 B):
 *   0  gRise     rising component of the dual exponential
 *   4  gDecay    decaying component; the conductance is gDecay - gRise
 *   8  stpR      Tsodyks-Markram available resources
 *   12 stpU      Tsodyks-Markram utilisation
 *   16 activity  0..1 render envelope for the travelling impulse
 *
 * SynapseTraces (16 B): the four STDP eligibility traces.
 */
export const WGSL_SYNAPSE_STRUCTS = /* wgsl */ `
struct SynapseStatic {
  pre : u32,
  post : u32,
  meta : u32,
  delay : f32,
  releaseProb : f32,
  tauRise : f32,
  tauDecay : f32,
  eRev : f32,
  mgBlock : f32,
  arc : f32,
}

struct SynapseDynamic {
  gRise : f32,
  gDecay : f32,
  stpR : f32,
  stpU : f32,
  activity : f32,
}

struct SynapseTraces {
  preTrace : f32,
  postTrace : f32,
  preTraceSlow : f32,
  postTraceSlow : f32,
}
`;

/**
 * Axon geometry, evaluated identically on the GPU and by `sampleArc` in
 * `@neuroforge/math`: a quadratic Bezier from the presynaptic soma to the
 * postsynaptic soma whose control point is the midpoint displaced by `sag` along
 * world up, projected perpendicular to the chord and renormalised. The
 * projection is what keeps the bulge the same size no matter how the two somata
 * are oriented; a chord that is itself nearly vertical falls back to whichever
 * horizontal axis it is least aligned with so the projection stays conditioned.
 *
 * `axonTangent` returns dP/dt, whose length is the local arc length per unit of
 * the curve parameter. Dividing a desired arc-length step by it advances a
 * particle at a constant speed in world units without any arc-length table.
 */
export const WGSL_SPLINE = /* wgsl */ `
const SPLINE_LENGTH_EPSILON : f32 = 1.0e-9;
const SPLINE_PARALLEL_EPSILON : f32 = 1.0e-6;

fn axonControl(a : vec3<f32>, b : vec3<f32>, sag : f32) -> vec3<f32> {
  let chord = b - a;
  let chordLength = length(chord);
  var normal = vec3<f32>(0.0, 1.0, 0.0);
  if (chordLength > SPLINE_LENGTH_EPSILON) {
    let u = chord / chordLength;
    var reference = vec3<f32>(0.0, 1.0, 0.0);
    if (u.y * u.y > 1.0 - SPLINE_PARALLEL_EPSILON) {
      reference = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(u.x) <= abs(u.z));
    }
    let perpendicular = reference - u * dot(u, reference);
    let perpendicularLength = length(perpendicular);
    if (perpendicularLength > SPLINE_LENGTH_EPSILON) {
      normal = perpendicular / perpendicularLength;
    }
  }
  return mix(a, b, 0.5) + normal * sag;
}

fn axonPoint(a : vec3<f32>, b : vec3<f32>, sag : f32, t : f32) -> vec3<f32> {
  let c = axonControl(a, b, sag);
  let s = 1.0 - t;
  return s * s * a + 2.0 * s * t * c + t * t * b;
}

fn axonTangent(a : vec3<f32>, b : vec3<f32>, sag : f32, t : f32) -> vec3<f32> {
  let c = axonControl(a, b, sag);
  return 2.0 * (1.0 - t) * (c - a) + 2.0 * t * (b - c);
}
`;

/**
 * Particle record (48 B, vec3-aligned).
 *   0  position  world space
 *   12 life      remaining lifetime in ms; <= 0 means the slot is free
 *   16 velocity  world units per ms, for the fragment-side streak
 *   28 speed     arc-length speed in world units per ms
 *   32 synapse   index of the axon this particle rides
 *   36 u         curve parameter in [0,1]
 *   40 distance  arc length travelled in world units
 *   44 size      per-particle size multiplier
 *
 * ParticleCounter holds the ring cursor the emitter advances and the live count
 * the update pass tallies. The host zeroes `live` before each update dispatch.
 */
export const WGSL_PARTICLE_STRUCTS = /* wgsl */ `
struct Particle {
  position : vec3<f32>,
  life : f32,
  velocity : vec3<f32>,
  speed : f32,
  synapse : u32,
  u : f32,
  distance : f32,
  size : f32,
}

struct ParticleCounter {
  cursor : atomic<u32>,
  live : atomic<u32>,
}
`;

/** Every chunk a compute kernel in this package may reference, in order. */
export const WGSL_PRELUDE = [WGSL_CONSTANTS, WGSL_MATH, WGSL_RANDOM, WGSL_META].join('\n');
