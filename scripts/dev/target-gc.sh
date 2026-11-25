#!/usr/bin/env bash
# Prune every cargo target dir of this repository's checkouts: the main
# checkout and each git worktree, the daemon's target/ and the app's
# clients/app/target/. Units unused for AIDAEMON_GC_DAYS days go, then the
# least recently used ones until each dir fits AIDAEMON_GC_MAX_GB.
#
#   scripts/dev/target-gc.sh             # prune
#   scripts/dev/target-gc.sh --dry-run   # report only
#
# scripts/dev/install-target-gc.sh runs it daily. Deleting a unit that is
# still wanted only costs rebuilding it.
set -euo pipefail

days="${AIDAEMON_GC_DAYS:-3}"
max_gb="${AIDAEMON_GC_MAX_GB:-25}"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
prune="$repo/scripts/prune-target.py"

git -C "$repo" worktree list --porcelain | sed -n 's/^worktree //p' | while read -r tree; do
  for target in "$tree/target" "$tree/clients/app/target"; do
    [ -f "$target/CACHEDIR.TAG" ] || continue
    printf '%s: ' "$target"
    python3 "$prune" --target "$target" --days "$days" --max-gb "$max_gb" "$@" | tail -1
  done
done
