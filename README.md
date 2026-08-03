# NeuroForge

**[sunkalpchandra.github.io/neuroforge](https://sunkalpchandra.github.io/neuroforge/)**

Browser-native neural circuit CAD. Design spiking networks on an infinite canvas,
simulate them with real biophysics, and watch the result render as glowing
procedural neuron glyphs wired by spline axons carrying travelling impulses.

Not an educational toy — a design tool that happens to be beautiful.

## What it does

**Five membrane models, switchable live.** Leaky integrate-and-fire, Izhikevich,
Hodgkin–Huxley with full m/h/n channel kinetics, adaptive exponential IF, and
Morris–Lecar. Every neuron carries its own parameters; a circuit can mix models
freely and the exporters handle the mix rather than flattening it.

**Real synapses.** Dual-exponential conductances with per-receptor kinetics
(AMPA, NMDA with voltage-dependent magnesium block, GABA-A, GABA-B, gap
junctions), axonal conduction delays through a calendar queue, stochastic
release, Tsodyks–Markram short-term facilitation and depression, and pair or
triplet STDP.

**Three compute backends.** WebGPU compute shaders when the browser exposes an
adapter, a Rust core compiled to WASM when it does not, and a TypeScript
reference integrator that always works. The reference implementation is the
numerical ground truth; the other two are validated against it.

**Structure-of-arrays throughout.** Neurons and synapses live in flat typed
arrays, not objects. The same buffers are handed to WASM as linear memory,
uploaded to GPU storage buffers, and read by instanced-mesh attribute updaters —
without a copy anywhere in the chain.

**Export that runs.** Brian2, PyNEST, PyTorch (as a surrogate-gradient spiking
module), ONNX, and round-trippable JSON. The generated scripts reproduce the
equations, per-neuron parameters, connectivity, delays and stimuli — they are
meant to be executed, not read.

## Stack

Next.js 16 · React 19 · TypeScript 6 · Tailwind v4 · Three.js · React Three Fiber
· WebGPU · Zustand · Motion · Dexie · Rust → WASM · petgraph · WGSL

## Layout

```
apps/web              React composition layer; owns wiring, not domain logic
packages/shared       Types, SoA buffer layout, design tokens — the contract
packages/math         RNG, noise, splines, springs, spatial hash, FFT
packages/shaders      WGSL compute kernels and GLSL materials
packages/simulation   Integrators (CPU / WASM / GPU), engine, probes
packages/renderer     Procedural glyphs, instanced fields, particles, camera
packages/physics      Barnes-Hut force layout and analytic population layouts
packages/editor       Zustand store, undoable command system, circuit CRUD
packages/io           Dexie persistence, snapshots, exporters
packages/ai           Prompt → circuit planning with a local fallback planner
packages/ui           Presentational primitives
crates/neuroforge-core  Rust simulation core and petgraph topology analysis
```

`CONTRACTS.md` is the integration contract between packages and is authoritative
when this README and the code disagree.

## Running it

```bash
npm install
npm run dev
```

Packages ship raw TypeScript and are compiled by Next through
`transpilePackages`, so there is no per-package build step and types stay live
across boundaries.

To rebuild the Rust core (optional — the app falls back to the TypeScript
integrator when the artifact is absent):

```bash
npm run wasm
```

## Verification

```bash
npm run lint
npm run typecheck
npm run build
cargo test --manifest-path crates/neuroforge-core/Cargo.toml
```

CI runs all of these on every branch. `main` additionally builds the WebAssembly
core and deploys the static export to GitHub Pages.

Beyond type and lint checks, the numerics are held to closed-form results rather
than to snapshots:

- LIF firing rate matches the analytic inter-spike interval to within 1%.
- Hodgkin–Huxley rests at −65 mV and fires at a physiological rate for a step.
- AdEx shows spike-frequency adaptation; conduction delays never deliver early.
- STDP potentiates pre-before-post and depresses post-before-pre.
- The WebAssembly core agrees with the TypeScript reference on spike counts for
  all five models, gap junctions included.
- Barnes-Hut forces stay within 0.3% of an exact N-body sum.
- Frequency detection is exact from 7 to 120 Hz, which is what lets the builder
  confirm it actually hit the gamma band.
- Generated Brian2, NEST and PyTorch scripts parse as Python; the ONNX export is
  a structurally valid protobuf.

## Deployment

The frontend is a fully static export and needs no server. AI features use a
bring-your-own-key model: the key is held in IndexedDB in the browser and never
transits a NeuroForge server, because on a static host there is no server to
transit. Point the client at your own proxy instead if you would rather not put
a key in a browser.

## Units

Voltage mV · time ms · current pA · capacitance pF · conductance nS. World-space
distance is abstract, with a typical soma radius of 1.0. This convention holds
across the integrators, the shaders and every exporter.
