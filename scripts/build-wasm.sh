#!/usr/bin/env bash
#
# Build the neuroforge-core crate to wasm and drop the artifact where the
# simulation package expects it.
#
# packages/simulation/wasm/ is gitignored, so this output is never committed and
# the JavaScript side must degrade to its CPU integrator when the directory is
# absent. The cargo target directory is removed afterwards: it is two orders of
# magnitude larger than the artifact and nothing downstream reads it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$ROOT/crates/neuroforge-core"
OUT="$ROOT/packages/simulation/wasm"

export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "build-wasm: wasm-pack not found on PATH." >&2
  echo "  install with: cargo install wasm-pack" >&2
  exit 1
fi

if ! rustup target list --installed 2>/dev/null | grep -qx 'wasm32-unknown-unknown'; then
  echo "build-wasm: the wasm32-unknown-unknown target is not installed." >&2
  echo "  install with: rustup target add wasm32-unknown-unknown" >&2
  exit 1
fi

echo "build-wasm: building $CRATE -> $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

wasm-pack build "$CRATE" \
  --release \
  --target web \
  --out-dir "$OUT" \
  --out-name neuroforge_core

# Reclaim the target directory; the artifact in $OUT is the whole deliverable.
cargo clean --manifest-path "$CRATE/Cargo.toml"

echo "build-wasm: done"
ls -lh "$OUT"
