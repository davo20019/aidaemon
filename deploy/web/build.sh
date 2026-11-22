#!/usr/bin/env bash
# Build the web app for app.aidaemon.ai into deploy/web/dist: the app's web
# package (nsh), its iroh module and service worker, and the header rules.
# Needs what clients/app's web build needs (nsh, wasm-bindgen-cli, Homebrew
# LLVM). Then deploy: cd deploy/web && npx wrangler deploy
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
app="$here/../../clients/app"
llvm="$(brew --prefix llvm 2>/dev/null || echo /usr)"
export CC_wasm32_unknown_unknown="${CC_wasm32_unknown_unknown:-$llvm/bin/clang}"
export AR_wasm32_unknown_unknown="${AR_wasm32_unknown_unknown:-$llvm/bin/llvm-ar}"
(cd "$app" && nsh package --target web)
rm -rf "$here/dist"
cp -R "$app/.nsh/out/AIdaemon" "$here/dist"
# Workers compress on their own; the precompressed copies are not needed.
find "$here/dist" \( -name '*.br' -o -name '*.gz' \) -delete
"$app/remote/build.sh" "$here/dist"
cp "$here/_headers" "$here/dist/_headers"
echo "built $here/dist"
