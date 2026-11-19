#!/usr/bin/env python3
"""Print the workspace crates crates.io gets, in publish order.

Each crate comes after every workspace crate it depends on (normal and build
dependencies; dev-dependencies do not count), so the release can publish them
one by one. The daemon (`aidaemon`, the root package) comes last.

  scripts/publish-order.py           # one package name per line
  scripts/publish-order.py --check   # also fail if a crate under crates/ has
                                     # `publish = false` (examples/ may)

Reads `cargo metadata --no-deps` for the workspace in the current directory.
"""

import json
import os
import subprocess
import sys


def main() -> int:
    meta = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"]
        )
    )
    root = meta["workspace_root"]
    packages = {p["name"]: p for p in meta["packages"]}

    def rel(p):
        return os.path.relpath(p["manifest_path"], root)

    # `publish = false` comes out as an empty registry list.
    publishable = {n for n, p in packages.items() if p.get("publish") != []}

    if "--check" in sys.argv:
        held = sorted(
            rel(p) for n, p in packages.items()
            if n not in publishable and rel(p).startswith("crates/")
        )
        if held:
            print("::error::These crates have publish = false, so a release "
                  "would skip them: " + ", ".join(held), file=sys.stderr)
            return 1

    deps = {
        n: sorted(
            {
                d["name"]
                for d in packages[n]["dependencies"]
                if d.get("kind") != "dev" and d["name"] in packages
            }
        )
        for n in publishable
    }
    for n, ds in deps.items():
        held = [d for d in ds if d not in publishable]
        if held:
            print(f"::error::{n} depends on unpublishable {held}", file=sys.stderr)
            return 1

    order, done, visiting = [], set(), set()

    def visit(n):
        if n in done:
            return
        if n in visiting:
            raise SystemExit(f"dependency cycle through {n}")
        visiting.add(n)
        for d in deps[n]:
            visit(d)
        visiting.discard(n)
        done.add(n)
        order.append(n)

    daemon = "aidaemon"
    for n in sorted(publishable):
        if n != daemon:
            visit(n)
    if daemon in publishable:
        visit(daemon)
    print("\n".join(order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
