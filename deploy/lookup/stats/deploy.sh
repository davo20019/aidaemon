#!/bin/bash
# Builds lookup-stats and installs it on the lookup server with a daily timer.
# Run again after changing it, or when iroh-dns-server moves to a redb with a
# new file format (bump redb in Cargo.toml to the server's Cargo.lock).
#
# The maintainers' own daemons are left out of the count: the public key of
# each remote-key file in OWN_KEYS (default: ~/.aidaemon/remote-key) is added
# to /etc/aidaemon-stats/exclude on the server. Keys already there stay.
#
# Needs: zig + cargo-zigbuild, rustup target x86_64-unknown-linux-musl,
# ssh key ~/.ssh/aidaemon_relay. Counts land in
# /var/lib/aidaemon-stats/daily.csv on the server.
set -euo pipefail

HOST="${HOST:-ubuntu@dns.aidaemon.ai}"
KEY="${KEY:-$HOME/.ssh/aidaemon_relay}"
TARGET="${TARGET:-x86_64-unknown-linux-musl}"   # aarch64-unknown-linux-musl for ARM servers
OWN_KEYS="${OWN_KEYS:-$HOME/.aidaemon/remote-key}"
HERE="$(cd "$(dirname "$0")" && pwd)"

cd "$HERE"
cargo test -q --locked
cargo zigbuild -q --release --locked --target "$TARGET"
BIN="$HERE/target/$TARGET/release/lookup-stats"
case "$TARGET" in x86_64*) WANT="x86-64" ;; aarch64*) WANT="aarch64" ;; esac
file "$BIN" | grep -q "$WANT" || { echo "unexpected binary: $(file "$BIN")"; exit 1; }

EXCLUDE="$(mktemp)"
trap 'rm -f "$EXCLUDE"' EXIT
for file in $OWN_KEYS; do
  [ -f "$file" ] && cargo run -q --locked -- key "$file" >> "$EXCLUDE"
done

scp -q -i "$KEY" "$BIN" lookup-stats.service lookup-stats.timer "$EXCLUDE" "$HOST:/tmp/"
ssh -i "$KEY" "$HOST" "set -e
  sudo install -m 0755 /tmp/lookup-stats /usr/local/bin/lookup-stats
  sudo install -m 0644 /tmp/lookup-stats.service /tmp/lookup-stats.timer /etc/systemd/system/
  sudo install -d -m 0755 /etc/aidaemon-stats
  sudo touch /etc/aidaemon-stats/exclude
  cat /tmp/$(basename "$EXCLUDE") | sudo tee -a /etc/aidaemon-stats/exclude >/dev/null
  sudo sort -u -o /etc/aidaemon-stats/exclude /etc/aidaemon-stats/exclude
  rm -f /tmp/lookup-stats /tmp/lookup-stats.service /tmp/lookup-stats.timer /tmp/$(basename "$EXCLUDE")
  sudo systemctl daemon-reload
  sudo systemctl enable --now lookup-stats.timer
  sudo systemctl start lookup-stats.service
  systemctl list-timers lookup-stats.timer --no-pager | head -2
  sudo tail -n 3 /var/lib/aidaemon-stats/daily.csv"
echo "deployed to $HOST"
