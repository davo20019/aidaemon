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
# The module's wasm downloads with the app's own, not after its script has
# run (wasm-bindgen's init fetches it same-origin, which `crossorigin` matches).
wasm="remote/v/$version/aidaemon_app_remote_bg.wasm"
tag="<meta name=\"aidaemon-remote\" content=\"$module\"><link rel=\"modulepreload\" href=\"$module\"><link rel=\"preload\" href=\"$wasm\" as=\"fetch\" type=\"application/wasm\" crossorigin>"
python3 - "$here/dist/index.html" "$tag" <<'PY'
import sys
path, tag = sys.argv[1], sys.argv[2]
page = open(path).read()
assert page.count("</head>") == 1, "index.html has no single </head>"
open(path, "w").write(page.replace("</head>", tag + "</head>"))
PY
cp "$here/_headers" "$here/dist/_headers"
# nsh names every file but the page and the manifest by a digest of its
# content (nsh.<digest>.wasm, browser.<digest>/, …), so browsers keep them
# without asking again: otherwise each launch revalidates the 12 MB engine
# and every module with the host before the app starts. Listed by name, as a
# catch-all rule would add its Cache-Control to the no-cache ones above.
for entry in "$here/dist"/*; do
  name="$(basename "$entry")"
  case "$name" in
    index.html | manifest.webmanifest | sw.js | _headers | remote) continue ;;
  esac
  if [ -d "$entry" ]; then rule="/$name/*"; else rule="/$name"; fi
  printf '%s\n  Cache-Control: public, max-age=31536000, immutable\n' "$rule" >>"$here/dist/_headers"
done
# nsh builds the app's describe program from a workspace of its own in the
# app's target dir, so Cargo keeps a second copy of most dependencies, and an
# older copy stays whenever that workspace changes. Units no build has read
# for a day go (scripts/prune-target.py; Cargo never removes them itself).
python3 "$here/../../scripts/prune-target.py" --target "${CARGO_TARGET_DIR:-$app/target}" --days 1 \
    || echo "build.sh: could not prune the app's target dir" >&2
echo "built $here/dist"
