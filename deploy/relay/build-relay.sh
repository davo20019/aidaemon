#!/bin/bash
# Builds iroh-relay with the AIDaemon landing page (index.html next to this
# script) and deploys it to the relay server. Run again after editing the page
# or when aidaemon moves to a new iroh version (set VERSION to match the
# workspace's iroh, and IROH_RELAY_VERSION in cloud-init.sh).
#
# Needs: zig + cargo-zigbuild (brew install zig; cargo install cargo-zigbuild),
# rustup target x86_64-unknown-linux-musl, ssh key ~/.ssh/aidaemon_relay.
set -euo pipefail

VERSION="${VERSION:-1.5.0}"
HOST="${HOST:-ubuntu@relay.aidaemon.ai}"
KEY="${KEY:-$HOME/.ssh/aidaemon_relay}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/.cache/aidaemon-relay-build}"
TARGET="${TARGET:-x86_64-unknown-linux-musl}"   # aarch64-unknown-linux-musl for ARM servers
SRC="$WORK/iroh-relay-$VERSION"

mkdir -p "$WORK"
if [ ! -d "$SRC" ]; then
  curl -fsSL "https://static.crates.io/crates/iroh-relay/iroh-relay-$VERSION.crate" | tar -xz -C "$WORK"
  cp "$SRC/src/server.rs" "$SRC/src/server.rs.orig"
fi

# Patch from the pristine source each time, so the page can change freely.
python3 - "$SRC/src/server.rs.orig" "$SRC/src/server.rs" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
# The landing page: served from index.html, baked into the binary.
s, n = re.subn(r'const INDEX: &\[u8\] = br#".*?"#;',
               'const INDEX: &[u8] = include_bytes!("aidaemon-index.html");', s, count=1, flags=re.S)
assert n == 1, "INDEX constant not found; check iroh-relay's server.rs"
# Allow the page's own inline styles and its data: favicon; nothing else.
old = "default-src 'none'; frame-ancestors"
new = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors"
assert old in s, "Content-Security-Policy not found; check iroh-relay's server.rs"
s = s.replace(old, new, 1)
open(dst, "w").write(s)
PY
cp "$HERE/index.html" "$SRC/src/aidaemon-index.html"

( cd "$SRC" && cargo zigbuild --release --locked --features server --bin iroh-relay --target "$TARGET" )
BIN="$SRC/target/$TARGET/release/iroh-relay"
case "$TARGET" in x86_64*) WANT="x86-64" ;; aarch64*) WANT="aarch64" ;; esac
file "$BIN" | grep -q "$WANT" || { echo "unexpected binary: $(file "$BIN")"; exit 1; }

scp -q -i "$KEY" "$BIN" "$HOST:/tmp/iroh-relay.new"
ssh -i "$KEY" "$HOST" 'set -e
  sudo install -m 0755 /tmp/iroh-relay.new /usr/local/bin/iroh-relay
  rm -f /tmp/iroh-relay.new
  sudo systemctl restart iroh-relay
  for i in $(seq 1 20); do curl -fsS -m 3 -o /dev/null --resolve relay.aidaemon.ai:443:127.0.0.1 https://relay.aidaemon.ai/healthz && break; sleep 1; done
  systemctl is-active iroh-relay'
ssh -i "$KEY" "$HOST" 'curl -fsS --resolve relay.aidaemon.ai:443:127.0.0.1 https://relay.aidaemon.ai/' | grep -q "AIDaemon Relay" && echo "deployed to $HOST"
