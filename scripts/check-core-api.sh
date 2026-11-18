#!/usr/bin/env bash
# Diff aidaemon-core's public API against the committed snapshot.
#
# The kernel's contract is meant to stay small and stable, so every change to
# it should be a reviewed diff, not a side effect of a refactor. CI runs this
# script; after an intended change, regenerate the snapshot with
#
#   scripts/check-core-api.sh --bless
#
# and commit crates/aidaemon-core/public-api.txt with the change.
#
# Requires the pinned nightly (rustdoc JSON) and cargo-public-api:
#   rustup toolchain install nightly-2026-07-01
#   cargo install cargo-public-api --locked --version 0.52.0
set -euo pipefail

TOOLCHAIN="${CORE_API_TOOLCHAIN:-nightly-2026-07-01}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SNAPSHOT="$ROOT/crates/aidaemon-core/public-api.txt"

current="$(mktemp)"
trap 'rm -f "$current"' EXIT
cargo "+$TOOLCHAIN" public-api --manifest-path "$ROOT/Cargo.toml" \
    -p aidaemon-core --all-features -sss >"$current"

if [[ "${1:-}" == "--bless" ]]; then
    cp "$current" "$SNAPSHOT"
    echo "Updated $SNAPSHOT"
    exit 0
fi

if ! diff -u "$SNAPSHOT" "$current"; then
    echo
    echo "aidaemon-core's public API changed. If that is intended, run"
    echo "scripts/check-core-api.sh --bless and commit the snapshot."
    exit 1
fi
echo "aidaemon-core public API matches the snapshot."
