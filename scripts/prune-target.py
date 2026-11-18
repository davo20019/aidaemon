#!/usr/bin/env python3
"""Delete build artifacts no recent cargo invocation used (cargo-sweep style).

Cargo never removes a unit's artifacts when its hash stops being used (a
feature set, profile or toolchain that changed), so a target dir grows
without bound. Every cargo invocation reads the fingerprint of every unit in
its build graph, fresh or not, which updates the fingerprint files' access
time. A unit whose fingerprint nobody read lately is dead weight; this script
removes those units, whole.

Layout (cargo 1.100+): one directory per unit, `<profile>/build/<pkg>/<hash>/`
holding `fingerprint/` and `out/`, plus `<profile>/incremental/<crate>-<hash>/`.

macOS (APFS) and Linux (relatime) update a file's access time on read only
when it is older than the modification time or a day old, so plain
`--days N` is exact to within a day. For a precise sweep, `--mark` first
resets the fingerprints' access times (contents and mtimes untouched); then
after the builds you want to keep (yours and anyone else's on the same
target), `--since-mark` deletes every unit none of them read.

Safe next to running builds: a unit being built or checked has just been
written or read, so it is kept. Deleting a unit that is still wanted only
costs a rebuild of it.

    scripts/prune-target.py --days 2 --dry-run   # what would go
    scripts/prune-target.py --days 2             # delete
    scripts/prune-target.py --mark               # start a precise window
    cargo test ...; cargo clippy ...             # the builds to keep
    scripts/prune-target.py --since-mark         # delete the rest

The target dir is $CARGO_TARGET_DIR, else ./target, else --target.
"""

import argparse
import os
import shutil
import sys
import time

MARK_FILE = ".prune-mark"
HEX = set("0123456789abcdef")


def is_hash(name):
    return len(name) == 16 and set(name) <= HEX


def files_under(path):
    for root, _dirs, files in os.walk(path):
        for name in files:
            yield os.path.join(root, name)


def size_of(path):
    total = 0
    for f in files_under(path):
        try:
            total += os.lstat(f).st_blocks * 512
        except OSError:
            pass
    return total


def last_use(paths):
    """Newest access or modification time among `paths` (0 if none)."""
    newest = 0.0
    for f in paths:
        try:
            st = os.lstat(f)
        except OSError:
            continue
        newest = max(newest, st.st_atime, st.st_mtime)
    return newest


def profile_dirs(target):
    """`<target>/<profile>` and `<target>/<triple>/<profile>` holding a build/ dir."""
    for entry in sorted(os.listdir(target)):
        p = os.path.join(target, entry)
        if not os.path.isdir(p):
            continue
        if os.path.isdir(os.path.join(p, "build")):
            yield p
        else:
            for sub in sorted(os.listdir(p)):
                q = os.path.join(p, sub)
                if os.path.isdir(os.path.join(q, "build")):
                    yield q


def units(target):
    """(path, fingerprint files) for every unit and incremental dir."""
    for prof in profile_dirs(target):
        build = os.path.join(prof, "build")
        for pkg in sorted(os.listdir(build)):
            pkg_dir = os.path.join(build, pkg)
            if not os.path.isdir(pkg_dir):
                continue
            for h in sorted(os.listdir(pkg_dir)):
                unit = os.path.join(pkg_dir, h)
                if not (is_hash(h) and os.path.isdir(unit)):
                    continue
                fp = os.path.join(unit, "fingerprint")
                probe = list(files_under(fp)) if os.path.isdir(fp) else list(files_under(unit))
                yield unit, probe
        inc = os.path.join(prof, "incremental")
        if os.path.isdir(inc):
            for name in sorted(os.listdir(inc)):
                d = os.path.join(inc, name)
                if os.path.isdir(d):
                    yield d, list(files_under(d))


def mark(target):
    count = 0
    for _unit, probe in units(target):
        for f in probe:
            try:
                st = os.lstat(f)
                # atime older than mtime: the next read updates it.
                os.utime(f, (0, st.st_mtime), follow_symlinks=False)
                count += 1
            except OSError:
                pass
    with open(os.path.join(target, MARK_FILE), "w") as fh:
        fh.write(str(time.time()))
    print(f"marked {count} fingerprint files in {target}")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", help="target dir (default $CARGO_TARGET_DIR or ./target)")
    when = ap.add_mutually_exclusive_group(required=True)
    when.add_argument("--days", type=float, help="delete units unused for this many days")
    when.add_argument("--mark", action="store_true", help="start a --since-mark window")
    when.add_argument("--since-mark", action="store_true", help="delete units unused since --mark")
    ap.add_argument("--dry-run", action="store_true", help="list what would be deleted")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each deleted unit")
    args = ap.parse_args()

    target = args.target or os.environ.get("CARGO_TARGET_DIR") or "target"
    target = os.path.abspath(target)
    if not os.path.isfile(os.path.join(target, "CACHEDIR.TAG")):
        sys.exit(f"{target} does not look like a cargo target dir (no CACHEDIR.TAG)")

    if args.mark:
        mark(target)
        return

    if args.since_mark:
        try:
            with open(os.path.join(target, MARK_FILE)) as fh:
                cutoff = float(fh.read().strip())
        except (OSError, ValueError):
            sys.exit("no mark: run --mark first, then your builds")
    else:
        cutoff = time.time() - args.days * 86400

    freed = kept = removed = 0
    for unit, probe in units(target):
        if last_use(probe) >= cutoff:
            kept += 1
            continue
        size = size_of(unit)
        freed += size
        removed += 1
        if args.verbose or args.dry_run:
            print(f"{human(size):>10}  {os.path.relpath(unit, target)}")
        if not args.dry_run:
            shutil.rmtree(unit, ignore_errors=True)
    # Empty package dirs left behind.
    if not args.dry_run:
        for prof in profile_dirs(target):
            build = os.path.join(prof, "build")
            for pkg in os.listdir(build):
                d = os.path.join(build, pkg)
                try:
                    os.rmdir(d)
                except OSError:
                    pass
    verb = "would free" if args.dry_run else "freed"
    print(f"{verb} {human(freed)} from {removed} units; kept {kept}")


if __name__ == "__main__":
    main()
