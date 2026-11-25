#!/usr/bin/env bash
# Prune every cargo target dir of this repository's checkouts: the main
# checkout and each git worktree, the daemon's target/ and the app's
# clients/app/target/. Incremental caches unused for
# AIDAEMON_GC_INCREMENTAL_DAYS days go, other units after AIDAEMON_GC_DAYS,
# then the least recently used ones until each dir fits AIDAEMON_GC_MAX_GB.
#
# When the disk has less than AIDAEMON_GC_MIN_FREE_GB free it prunes harder:
# half the budget, a day's age for units, six hours for incremental caches,
# and only the last 15 minutes' units are safe.
#
#   scripts/dev/target-gc.sh             # prune
#   scripts/dev/target-gc.sh --dry-run   # report only
#
# scripts/dev/install-target-gc.sh runs it hourly: with incremental builds on,
# one busy session can add 20 GB in an afternoon. Deleting a unit that is
# still wanted only costs rebuilding it.
set -euo pipefail

days="${AIDAEMON_GC_DAYS:-3}"
inc_days="${AIDAEMON_GC_INCREMENTAL_DAYS:-1}"
max_gb="${AIDAEMON_GC_MAX_GB:-25}"
min_free_gb="${AIDAEMON_GC_MIN_FREE_GB:-40}"
recent_min=60
repo="${AIDAEMON_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
prune="$(cd "$(dirname "$0")/.." && pwd)/prune-target.py"

# One run at a time (a reinstall's run-at-load can meet an hourly run).
lock="${TMPDIR:-/tmp}/aidaemon-target-gc.lock"
# A run killed outright leaves its lock behind: one over two hours old is stale.
find "$lock" -maxdepth 0 -mmin +120 -exec rmdir {} \; 2>/dev/null || true
if ! mkdir "$lock" 2>/dev/null; then
  echo "$(date '+%F %T') another run holds $lock; skipping"
  exit 0
fi
trap 'rmdir "$lock"' EXIT

free_gb() { df -Pk "$repo" | awk 'NR == 2 { print int($4 / 1048576) }'; }

free="$(free_gb)"
if [ "$free" -lt "$min_free_gb" ]; then
  echo "$(date '+%F %T') low disk (${free} GB free): pruning harder"
  days=1
  inc_days=0.25
  max_gb=$((max_gb / 2))
  recent_min=15
fi

git -C "$repo" worktree list --porcelain | sed -n 's/^worktree //p' | while read -r tree; do
  for target in "$tree/target" "$tree/clients/app/target"; do
    [ -f "$target/CACHEDIR.TAG" ] || continue
    printf '%s %s: ' "$(date '+%F %T')" "$target"
    python3 "$prune" --target "$target" --days "$days" --incremental-days "$inc_days" \
      --max-gb "$max_gb" --keep-recent-min "$recent_min" "$@" | tail -1
  done
done
