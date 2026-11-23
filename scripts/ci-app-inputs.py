#!/usr/bin/env python3
"""The paths the app (clients/app) is built from, one per line.

CI runs the app's jobs on a pull request only when the pull request changes
one of them: the app itself, every local crate it reaches through path
dependencies (directly, or through the root workspace's dependency table),
the root manifest those crates inherit from, the toolchain pin, and CI's own
definition.

    scripts/ci-app-inputs.py                 # list the paths
    scripts/ci-app-inputs.py --changed A B…  # exit 0 if any of A B… is in them
"""

import os
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALWAYS = [
    "clients/app",
    "Cargo.toml",
    "rust-toolchain.toml",
    ".github/workflows/ci.yml",
    ".github/actions",
    "scripts/ci-app-inputs.py",
]


def manifest(dir):
    with open(os.path.join(dir, "Cargo.toml"), "rb") as f:
        return tomllib.load(f)


def dependency_tables(toml):
    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield toml.get(key, {})
    for target in toml.get("target", {}).values():
        for key in ("dependencies", "dev-dependencies", "build-dependencies"):
            yield target.get(key, {})


def path_deps(dir, workspace_deps):
    """Directories of the local crates `dir`'s manifest depends on."""
    for table in dependency_tables(manifest(dir)):
        for name, spec in table.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("workspace"):
                spec = workspace_deps.get(spec.get("package", name), {})
                if not isinstance(spec, dict) or "path" not in spec:
                    continue
                yield os.path.normpath(os.path.join(ROOT, spec["path"]))
            elif "path" in spec:
                yield os.path.normpath(os.path.join(dir, spec["path"]))


def inputs():
    workspace_deps = manifest(ROOT).get("workspace", {}).get("dependencies", {})
    seen = set()
    todo = [os.path.join(ROOT, "clients/app"), os.path.join(ROOT, "clients/app/remote")]
    while todo:
        dir = todo.pop()
        if dir in seen or not os.path.exists(os.path.join(dir, "Cargo.toml")):
            continue
        seen.add(dir)
        todo.extend(path_deps(dir, workspace_deps))
    found = {os.path.relpath(d, ROOT) for d in seen}
    # Crates inside the app's folder are covered by it.
    found = {p for p in found if not (p + "/").startswith("clients/app/")}
    return sorted(set(ALWAYS) | found)


def covers(path, changed):
    return changed == path or changed.startswith(path + "/")


def main():
    paths = inputs()
    if sys.argv[1:2] == ["--changed"]:
        changed = sys.argv[2:]
        hits = [c for c in changed if any(covers(p, c) for p in paths)]
        for c in hits[:20]:
            print(c)
        sys.exit(0 if hits else 1)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
