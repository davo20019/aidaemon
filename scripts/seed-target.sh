#!/usr/bin/env bash
# Give this worktree its own target dir, seeded from another checkout's.
#
#   scripts/seed-target.sh ../aidaemon/target
#
# Worktrees must not share one target dir: cargo names a workspace crate's
# artifacts by a hash that is the same in every checkout, and judges them
# fresh by comparing modification times, so worktree A's build of a crate is
# taken as up to date in worktree B whenever B's files are older. That serves
# B code it does not have ("impossible" errors that a `touch` fixes).
#
# A separate target need not cost a fresh build or the disk for one: on APFS
# (macOS) `cp -c` clones files copy-on-write, so the seed shares every block
# with the source until one side rebuilds it. On Linux `cp --reflink=auto`
# does the same on btrfs/xfs and copies elsewhere.
#
# The workspace's own crates are dropped from the copy, so their first build
# here compiles this worktree's sources; dependencies are reused as they are.
set -euo pipefail

src="${1:?usage: scripts/seed-target.sh <source target dir>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dst="${CARGO_TARGET_DIR:-$root/target}"

[ -f "$src/CACHEDIR.TAG" ] || { echo "$src is not a cargo target dir" >&2; exit 1; }
src="$(cd "$src" && pwd)"
[ "$src" != "$(mkdir -p "$dst" && cd "$dst" && pwd)" ] || { echo "source and destination are the same" >&2; exit 1; }
if [ -n "$(ls -A "$dst" 2>/dev/null)" ]; then
    echo "$dst is not empty; remove it first" >&2
    exit 1
fi
rmdir "$dst"

if [ "$(uname)" = Darwin ]; then
    cp -c -R -p "$src" "$dst"
else
    cp -R -p --reflink=auto "$src" "$dst"
fi

# Workspace members, by package name: their unit dirs are build/<name>/<hash>.
members="$(cd "$root" && cargo metadata --no-deps --format-version 1 --offline |
    python3 -c 'import json,sys; print("\n".join(p["name"] for p in json.load(sys.stdin)["packages"]))')"
for prof in "$dst"/*/ "$dst"/*/*/; do
    [ -d "$prof/build" ] || continue
    while IFS= read -r name; do
        rm -rf "${prof}build/$name"
    done <<<"$members"
    rm -rf "${prof}incremental"
done
rm -f "$dst/.prune-mark"
echo "seeded $dst from $src (workspace crates dropped)"
