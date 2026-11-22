#!/usr/bin/env bash
# Build the web app's iroh module into <out>/remote/ and its service worker
# into <out>/sw.js, beside the page that loads them on demand (default <out>:
# the app's web package, .nsh/out/AIdaemon).
# Needs the wasm32 target, Homebrew's LLVM (ring's C) and wasm-bindgen-cli
# at the version Cargo.toml pins.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/../.nsh/out/AIdaemon}"
llvm="$(brew --prefix llvm 2>/dev/null || echo /usr)"
export CC_wasm32_unknown_unknown="${CC_wasm32_unknown_unknown:-$llvm/bin/clang}"
export AR_wasm32_unknown_unknown="${AR_wasm32_unknown_unknown:-$llvm/bin/llvm-ar}"
export RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
cd "$here"
cargo build --release --target wasm32-unknown-unknown --lib
mkdir -p "$out/remote"
wasm-bindgen --target web --no-typescript --out-dir "$out/remote" \
  target/wasm32-unknown-unknown/release/aidaemon_app_remote.wasm
# The service worker sits at the page's level: its scope is the app.
cp sw.js "$out/sw.js"
echo "iroh module: $out/remote, service worker: $out/sw.js"
