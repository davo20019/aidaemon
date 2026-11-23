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
# The iroh module again under a name its content picks (remote/v/<hash>/),
# kept by browsers without asking again (_headers); the page names it
# (js/browser.js reads the meta). remote/ keeps its own copy for a page
# loaded before this release.
version="$(cat "$here/dist/remote/aidaemon_app_remote.js" "$here/dist/remote/aidaemon_app_remote_bg.wasm" | shasum -a 256 | cut -c1-16)"
mkdir -p "$here/dist/remote/v/$version"
cp "$here/dist/remote/aidaemon_app_remote.js" "$here/dist/remote/aidaemon_app_remote_bg.wasm" "$here/dist/remote/v/$version/"
module="remote/v/$version/aidaemon_app_remote.js"
tag="<meta name=\"aidaemon-remote\" content=\"$module\"><link rel=\"modulepreload\" href=\"$module\">"
python3 - "$here/dist/index.html" "$tag" <<'PY'
import sys
path, tag = sys.argv[1], sys.argv[2]
page = open(path).read()
assert page.count("</head>") == 1, "index.html has no single </head>"
open(path, "w").write(page.replace("</head>", tag + "</head>"))
PY
cp "$here/_headers" "$here/dist/_headers"
echo "built $here/dist"
