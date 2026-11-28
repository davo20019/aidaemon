#!/usr/bin/env bash
# Assemble deploy/eink/dist: the e-ink page plus the bridge protocol's page
# side (one source: crates/aidaemon-bridge-proto/js/bridge-proto.js).
# Serve dist/ from any static HTTPS host (WebCrypto needs a secure context;
# http://localhost also counts for development). Set the bridge in
# dist/config.js.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
rm -rf "$here/dist"
mkdir -p "$here/dist"
cp "$here/public/"* "$here/dist/"
cp "$root/crates/aidaemon-bridge-proto/js/bridge-proto.js" "$here/dist/"
if [ -n "${AIDAEMON_EINK_BRIDGE:-}" ]; then
  printf "window.AIDAEMON_EINK_CONFIG = { bridge: '%s' };\n" "$AIDAEMON_EINK_BRIDGE" > "$here/dist/config.js"
fi
echo "built $here/dist"
