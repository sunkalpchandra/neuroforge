import { PARAM_SLOT } from '@neuroforge/shared';

/**
 * Membrane model integration.
 *
 * Unit system, enforced everywhere: v in mV, t in ms, I in pA, cm in pF,
 * g in nS. The combination is self-consistent because 1 nS x 1 mV = 1 pA and
 * 1 pA / 1 pF = 1 mV/ms, so no conversion factors appear in the equations.
 *
 * Parameters are read from the packed SoA block at the offsets declared in
 * @neuroforge/shared's PARAM_SLOT table. The WGSL kernels and the Rust core
 * index the same table; changing an offset means changing all three.
 */

export const INTEGRATOR_CODE = {
  euler: 0,
  rk2: 1,
  rk4: 2,
  'exponential-euler': 3,
} as const;

export type IntegratorCode = (typeof INTEGRATOR_CODE)[keyof typeof INTEGRATOR_CODE];

/**
 * Below this magnitude a denominator of the form (1 - exp(-x/k)) is replaced by
 * its analytic limit. Both alpha_m and alpha_n in the Hodgkin-Huxley rate
 * functions have a removable singularity that evaluates to 0/0 in floating point
 * and would otherwise emit NaN exactly at the voltages a neuron passes through
 * on every single spike.
 */
const SINGULARITY_EPS = 1e-6;

/** Largest exponent passed to Math.exp, chosen to stay well inside float range. */
const EXP_CLAMP = 50;

function safeExp(x: number): number {
  return Math.exp(x < -EXP_CLAMP ? -EXP_CLAMP : x > EXP_CLAMP ? EXP_CLAMP : x);
}

/* ---------------------------------------------------------------- LIF ---- */

/**
 * Leaky integrate-and-fire. The subthreshold equation is linear, so
 * exponential Euler is exact for constant input over the step rather than
 * merely more stable, and it is used regardless of the requested integrator.
 */
export function stepLif(
  v: number,
  current: number,
  dt: number,
  params: Float32Array,
  base: number,
): number {
  const cm = params[base + PARAM_SLOT.LIF_CM];
  const gL = params[base + PARAM_SLOT.LIF_GL];
  const eL = params[base + PARAM_SLOT.LIF_EL];
  if (gL <= 0 || cm <= 0) return v;
  const tau = cm / gL;
  const vInf = eL + current / gL;
  return vInf + (v - vInf) * safeExp(-dt / tau);
}

/* -------------------------------------------------------- Izhikevich ---- */

/**
 * Izhikevich 2003. The voltage equation is integrated as two half-steps, which
 * is the author's own recommendation: the quadratic term makes a single full
 * step numerically unstable near the spike upstroke.
 */
export function stepIzhikevich(
  state: { v: number; u: number },
  current: number,
  dt: number,
  params: Float32Array,
  base: number,
): void {
  const a = params[base + PARAM_SLOT.IZH_A];
  const b = params[base + PARAM_SLOT.IZH_B];
  const iScale = params[base + PARAM_SLOT.IZH_ISCALE];
  const i = current * iScale;
  const half = dt * 0.5;

  let v = state.v;
  const u = state.u;
  v += half * (0.04 * v * v + 5 * v + 140 - u + i);
  v += half * (0.04 * v * v + 5 * v + 140 - u + i);
  state.v = v;
  state.u = u + dt * a * (b * v - u);
}

/* ---------------------------------------------------- Hodgkin-Huxley ---- */

function alphaN(v: number): number {
  const x = v + 55;
  if (Math.abs(x) < SINGULARITY_EPS) return 0.1;
  return (0.01 * x) / (1 - safeExp(-x / 10));
}

function betaN(v: number): number {
  return 0.125 * safeExp(-(v + 65) / 80);
}

function alphaM(v: number): number {
  const x = v + 40;
  if (Math.abs(x) < SINGULARITY_EPS) return 1.0;
  return (0.1 * x) / (1 - safeExp(-x / 10));
}

function betaM(v: number): number {
  return 4 * safeExp(-(v + 65) / 18);
}

function alphaH(v: number): number {
  return 0.07 * safeExp(-(v + 65) / 20);
}

function betaH(v: number): number {
  return 1 / (1 + safeExp(-(v + 35) / 10));
}

/**
 * Advance one gating variable using the Rush-Larsen scheme: treat alpha and beta
 * as frozen over the step and solve the resulting linear ODE exactly. Gates stay
 * inside [0,1] for any dt, which forward Euler does not guarantee.
 */
function stepGate(x: number, alpha: number, beta: number, dt: number): number {
  const sum = alpha + beta;
  if (sum <= 0) return x;
  const inf = alpha / sum;
  return inf + (x - inf) * safeExp(-dt * sum);
}

export interface HhState {
  v: number;
  m: number;
  h: number;
  n: number;
}

export function stepHodgkinHuxley(
  state: HhState,
  current: number,
  dt: number,
  params: Float32Array,
  base: number,
): void {
  const cm = params[base + PARAM_SLOT.HH_CM];
  const gNa = params[base + PARAM_SLOT.HH_GNA];
  const gK = params[base + PARAM_SLOT.HH_GK];
  const gL = params[base + PARAM_SLOT.HH_GL];
  const eNa = params[base + PARAM_SLOT.HH_ENA];
  const eK = params[base + PARAM_SLOT.HH_EK];
  const eL = params[base + PARAM_SLOT.HH_EL];
  const q10 = params[base + PARAM_SLOT.HH_Q10] || 1;
  if (cm <= 0) return;

  const v = state.v;

  state.m = stepGate(state.m, alphaM(v) * q10, betaM(v) * q10, dt);
  state.h = stepGate(state.h, alphaH(v) * q10, betaH(v) * q10, dt);
  state.n = stepGate(state.n, alphaN(v) * q10, betaN(v) * q10, dt);

  const m3h = state.m * state.m * state.m * state.h;
  const n4 = state.n * state.n * state.n * state.n;

  // Conductances are frozen over the step, which makes the voltage equation
  // linear and lets exponential Euler solve it exactly. gTotal is strictly
  // positive because gL is, so the division is safe.
  const gTotal = gNa * m3h + gK * n4 + gL;
  const iRev = gNa * m3h * eNa + gK * n4 * eK + gL * eL;
  const vInf = (iRev + current) / gTotal;
  const tau = cm / gTotal;
  state.v = vInf + (v - vInf) * safeExp(-dt / tau);
}

/** Steady-state gate values at a given potential, used to initialise state. */
export function hhSteadyState(v: number): { m: number; h: number; n: number } {
  const m = alphaM(v) / (alphaM(v) + betaM(v));
  const h = alphaH(v) / (alphaH(v) + betaH(v));
  const n = alphaN(v) / (alphaN(v) + betaN(v));
  return { m, h, n };
}

/* ---------------------------------------------------------------- AdEx --- */

export function stepAdEx(
  state: { v: number; w: number },
  current: number,
  dt: number,
  params: Float32Array,
  base: number,
): void {
  const cm = params[base + PARAM_SLOT.ADEX_CM];
  const gL = params[base + PARAM_SLOT.ADEX_GL];
  const eL = params[base + PARAM_SLOT.ADEX_EL];
  const deltaT = params[base + PARAM_SLOT.ADEX_DELTAT];
  const vT = params[base + PARAM_SLOT.ADEX_VT];
  const a = params[base + PARAM_SLOT.ADEX_A];
  const tauW = params[base + PARAM_SLOT.ADEX_TAUW];
  if (cm <= 0 || tauW <= 0) return;

  const v = state.v;
  const w = state.w;

  // The exponential term is the whole point of the model but also the thing that
  // overflows: clamping the argument caps the upstroke slope instead of
  // producing Infinity, and the spike is detected on the very next comparison
  // against vPeak anyway.
  const expTerm = deltaT > 0 ? gL * deltaT * safeExp((v - vT) / deltaT) : 0;

  const dv = (-gL * (v - eL) + expTerm - w + current) / cm;
  state.v = v + dt * dv;
  state.w = w + (dt * (a * (v - eL) - w)) / tauW;
}

/* -------------------------------------------------------- Morris-Lecar --- */

export function stepMorrisLecar(
  state: { v: number; w: number },
  current: number,
  dt: number,
  params: Float32Array,
  base: number,
): void {
  const cm = params[base + PARAM_SLOT.ML_CM];
  const gCa = params[base + PARAM_SLOT.ML_GCA];
  const gK = params[base + PARAM_SLOT.ML_GK];
  const gL = params[base + PARAM_SLOT.ML_GL];
  const eCa = params[base + PARAM_SLOT.ML_ECA];
  const eK = params[base + PARAM_SLOT.ML_EK];
  const eL = params[base + PARAM_SLOT.ML_EL];
  const v1 = params[base + PARAM_SLOT.ML_V1];
  const v2 = params[base + PARAM_SLOT.ML_V2];
  const v3 = params[base + PARAM_SLOT.ML_V3];
  const v4 = params[base + PARAM_SLOT.ML_V4];
  const phi = params[base + PARAM_SLOT.ML_PHI];
  if (cm <= 0 || v2 === 0 || v4 === 0) return;

  const v = state.v;
  const w = state.w;

  const mInf = 0.5 * (1 + Math.tanh((v - v1) / v2));
  const wInf = 0.5 * (1 + Math.tanh((v - v3) / v4));
  const tauW = 1 / (phi * Math.cosh((v - v3) / (2 * v4)));

  const gTotal = gL + gCa * mInf + gK * w;
  const iRev = gL * eL + gCa * mInf * eCa + gK * w * eK;
  const vInf = (iRev + current) / gTotal;
  const tau = cm / gTotal;

  state.v = vInf + (v - vInf) * safeExp(-dt / tau);
  state.w = wInf + (w - wInf) * safeExp(-dt / tauW);
}

/** Steady-state Morris-Lecar recovery variable, used to initialise state. */
export function mlSteadyState(v: number, v3: number, v4: number): number {
  if (v4 === 0) return 0;
  return 0.5 * (1 + Math.tanh((v - v3) / v4));
}
