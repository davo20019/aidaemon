#!/usr/bin/env bash
# Build aidaemon-bridge for the server and install it there.
#   HOST=ubuntu@bridge.aidaemon.ai TARGET=aarch64-unknown-linux-musl ./deploy.sh
# TARGET: aarch64-unknown-linux-musl (Oracle A1) or x86_64-unknown-linux-musl.
# Needs cargo-zigbuild. SSH_KEY defaults to ~/.ssh/aidaemon_relay.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
HOST="${HOST:?set HOST=user@server}"
TARGET="${TARGET:-aarch64-unknown-linux-musl}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/aidaemon_relay}"
(cd "$root" && cargo zigbuild --release -p aidaemon-bridge --target "$TARGET")
bin="$root/target/$TARGET/release/aidaemon-bridge"
sum="$(shasum -a 256 "$bin" | cut -d' ' -f1)"
# Staged in a folder only the deploy user can write (not a shared /tmp
# path another account could swap), and checked before root installs it.
stage="$(ssh -i "$SSH_KEY" "$HOST" 'mktemp -d')"
scp -i "$SSH_KEY" "$bin" "$HOST:$stage/aidaemon-bridge"
ssh -i "$SSH_KEY" "$HOST" "echo '$sum  $stage/aidaemon-bridge' | sha256sum -c - && sudo install -m 0755 '$stage/aidaemon-bridge' /usr/local/bin/aidaemon-bridge && rm -rf '$stage' && sudo systemctl restart aidaemon-bridge && systemctl is-active aidaemon-bridge"
