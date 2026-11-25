#!/usr/bin/env python3
"""Make `cargo test -p <crate>` reuse the workspace build's units.

Cargo resolves features per command: `cargo test -p aidaemon-runtime` turns
on only the features that crate and its dev-dependencies ask for, so every
workspace crate below it whose feature set differs from the root build's is
compiled again as a new unit (and so is everything above it). On nightly,
`-Zfeature-unification` with `resolver.feature-unification = "workspace"`
fixes that; this script does the same on stable.

It takes each workspace crate's features as a `cargo tree --workspace` build
resolves them (the union of every member's normal, build and dev edges), and
gives every member a dev-dependency on each workspace crate in its own build
with exactly that set. Features are additive and the union is a superset of
what any one command asks for, so nothing a crate's tests relied on goes
away; dev-dependencies are not published, so the crates on crates.io are
unchanged. Third-party crates are unified by workspace-hack/ (cargo-hakari).

    scripts/dev/unify-dev-features.py           # rewrite Cargo.toml files
    scripts/dev/unify-dev-features.py --check   # exit 1 if any is stale

Rerun it after a dependency or feature change between workspace crates.
"""

import json
import os
import re
import subprocess
import sys
import tomllib

# tomlkit (pip install tomlkit) only to rewrite manifests; --check needs nothing.
try:
    import tomlkit
except ImportError:
    tomlkit = None

def cargo(*args):
    return subprocess.check_output(["cargo", *args], text=True, stderr=subprocess.DEVNULL)


def members():
    meta = json.loads(cargo("metadata", "--format-version", "1", "--no-deps"))
    ids = set(meta["workspace_members"])
    out = {}
    for p in meta["packages"]:
        if p["id"] in ids:
            out[p["name"]] = p
    return out, meta["workspace_root"]


def resolved(selector):
    """{workspace crate: set of enabled features} for one cargo build selection."""
    lines = cargo(
        "tree", *selector, "-e", "normal,build,dev", "--prefix", "none",
        "-f", "{p}|{f}", "--color", "never",
    )
    feats = {}
    for line in lines.splitlines():
        line = line.replace(" (*)", "")
        if "|" not in line:
            continue
        pkg, f = line.rsplit("|", 1)
        if "(/" not in pkg and "(C:" not in pkg:
            continue  # registry crate: workspace-hack's job
        name = pkg.split(" ", 1)[0]
        feats.setdefault(name, set()).update(x for x in f.split(",") if x)
    return feats


def spaced(text):
    """`{path = ..}` as tomlkit writes a new inline table -> `{ path = .. }`."""
    return re.sub(r"(?m)^([\w-]+ = )\{(path = .*)\}$", r"\1{ \2 }", text)


def dev_table(doc):
    if "dev-dependencies" not in doc:
        doc["dev-dependencies"] = tomlkit.table()
    return doc["dev-dependencies"]


def main():
    check = "--check" in sys.argv
    stale = set()
    for _round in range(5):
        changed = sync(check)
        stale |= set(changed)
        if check or not changed:
            break
    if check:
        if stale:
            print("dev-dependency features differ from the workspace's in:", *sorted(stale), sep="\n  ")
            print("run scripts/dev/unify-dev-features.py")
            return 1
        print("dev-dependency features match the workspace build")
        return 0
    print(f"updated {len(stale)} manifests" if stale else "nothing to update")
    return 0


def sync(check):
    """One pass; returns the manifests that were (or, with check, would be) changed."""
    pkgs, root = members()
    unified = resolved(["--workspace"])
    stale = []

    for name, p in sorted(pkgs.items()):
        manifest = p["manifest_path"]
        here = os.path.dirname(manifest)
        mine = resolved(["-p", name])
        with open(manifest) as fh:
            text = fh.read()
        if check:
            doc = tomllib.loads(text)
            dev = doc.get("dev-dependencies", {})
        else:
            if tomlkit is None:
                sys.exit("pip install tomlkit (needed to rewrite manifests)")
            doc = tomlkit.parse(text)
            dev = dev_table(doc)
        changed = False
        for dep, have in sorted(mine.items()):
            want = unified.get(dep, set())
            if have >= want or dep not in pkgs:
                continue
            path = os.path.relpath(os.path.dirname(pkgs[dep]["manifest_path"]), here)
            feats = sorted(f for f in want if f != "default")
            # The key a manifest already uses for this crate (renames aside).
            key = next(
                (k for k, v in dev.items()
                 if k == dep or (isinstance(v, dict) and v.get("package") == dep)),
                dep,
            )
            entry = dev.get(key)
            if entry is None or not isinstance(entry, dict):
                entry = tomlkit.inline_table() if tomlkit else {}
                entry["path"] = path
                if "default" not in want:
                    entry["default-features"] = False
            old = set(entry.get("features", []))
            if old >= set(feats) and key in dev:
                continue  # already asks for it; cargo tree just disagrees on default
            entry["features"] = sorted(old | set(feats))
            dev[key] = entry
            changed = True
        if changed:
            stale.append(os.path.relpath(manifest, root))
            if not check:
                with open(manifest, "w") as fh:
                    fh.write(spaced(tomlkit.dumps(doc)))

    return stale


if __name__ == "__main__":
    sys.exit(main())
