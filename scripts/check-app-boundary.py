#!/usr/bin/env python3
"""Fail when agent behavior lands in the app instead of a crate.

The daemon's `src/` is meant to hold only wiring and the product shell; every
agent behavior lives in a crate (Phase 9 of the SDK plan). This check holds
the parts already moved to that line:

- Every top-level module in `src/` is on one of three lists: `SHELL`
  (wiring and the product shell, which stay), `REEXPORTS` (old paths that
  only re-export crate items, and must hold nothing else) or `PENDING`
  (modules whose behavior still waits to move, each with the plan step that
  moves it). A new module fails until it is put on a list. A listed module
  that is gone fails too, so the lists only shrink.
- Directories on `MOVED` are done (`src/agent/`, `src/nodes/`, `src/channels/`): each of
  their files is either an adapter on that directory's list (the app as the
  crates' host) or holds nothing but tests and module plumbing once its
  `#[cfg(test)]` items are removed. A function, impl, type or constant
  outside a test fails.

Add a directory to `MOVED`, with its adapter list, when its step lands, so
it is held the same way.

Usage: scripts/check-app-boundary.py [repo root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Wiring and the product shell: what the app keeps (design, Goal 1).
SHELL = {
    "main.rs": "binary entry point",
    "lib.rs": "crate root: module wiring and the CLI",
    "bin": "binaries such as db_probe",
    "core.rs": "assembly: runs the daemon from its parts",
    "startup": "assembly: builds each crate's parts from the config",
    "providers": "provider construction from the config",
    "config.rs": "config parsing and its mapping onto crate settings",
    "wizard.rs": "setup wizard",
    "updater.rs": "self-update",
    "single_instance.rs": "process handling",
    "daemon.rs": "service install and process handling",
    "logging.rs": "log setup",
    "dashboard.rs": "the dashboard",
    "dashboard.html": "the dashboard",
    "otel_export.rs": "OTel export setup",
    "runtime_ports.rs": "the app's port traits onto crate parts",
    "agent": "the agent as the crates' host (held by MOVED and AGENT_ADAPTERS)",
    "nodes": "the Node service's host adapters (held by MOVED and NODE_ADAPTERS)",
    "channels": "the daemon's channel hosts (held by MOVED and CHANNEL_ADAPTERS)",
    "testing.rs": "the daemon's test harness",
    "integration_tests.rs": "tests of the assembled product",
    "integration_tests": "tests of the assembled product",
    "live_e2e_tests.rs": "live end-to-end tests of the assembled product",
    "contract_shape_tests.rs": "tests of the assembled product",
    "harness_eval": "the eval subject over the assembled daemon",
}

# Old paths kept as re-exports of crate items: they may hold nothing else.
REEXPORTS = {
    "context_budget.rs",
    "conversation.rs",
    "execution",
    "events",
    "execution_policy.rs",
    "health",
    "heartbeat",
    "llm_markers.rs",
    "llm_runtime.rs",
    "llm_runtime",
    "mcp",
    "oauth",
    "router.rs",
    "skills",
    "state",
    "tools",
    "traits",
    "traits.rs",
    "types.rs",
    "utils.rs",
}

# Behavior still in the app, and the plan step that moves it.
PENDING = {
    "memory": "9.1/9.6: memory host adapters, procedures, task learning, retention",
}


# The agent as the crates' host: the files that may hold non-test code.
AGENT_ADAPTERS = {
    "mod.rs": "the Agent struct, its setters and its entry points (delegations)",
    "construct.rs": "builds the Agent from the daemon's parts",
    "turn_host.rs": "Agent as TurnHost and ChildAgent",
    "runtime_ports.rs": "Agent as the channels' and tools' ports",
    "runtime/spawn.rs": "create_child_agent: builds a child Agent",
    "runtime/models.rs": "provider reload from the daemon config, session clears",
    "memory_plugin.rs": "the daemon memory as a MemoryPlugin",
    "memory_plugin/system_memory.rs": "SystemMemory over the daemon stores",
    "memory_plugin/recall.rs": "recall over the daemon stores",
    "computer_use.rs": "computer_use pins as ComputerUseState",
}

# The Node service as the crates' host: the files that may hold non-test code.
NODE_ADAPTERS = {
    "mod.rs": "re-exports, and the instance key from the daemon's secret store",
    "service.rs": "the agent behind Node turns (NodeConversationIngress)",
    "cli.rs": "the `aidaemon node` CLI subcommand over the crate's services",
}

# The daemon's channel hosts: the bots in the adapter crates on the daemon's
# stores, config file and process. The files that may hold non-test code.
CHANNEL_ADAPTERS = {
    "mod.rs": "ChannelRuntimeDeps: the daemon's services as the bots' BotDeps",
    "commands.rs": "the shared slash commands on the daemon's stores, /reload on its config",
    "connect.rs": "/connect token checks by platform, as this build has them",
    "discord.rs": "builds the Discord bot on the daemon's services",
    "slack.rs": "the Slack bot's host: workspace grants from the config, /restart",
    "set_key.rs": "/setkey: which config entries take a key, saving one to the config",
    "telegram.rs": "the Telegram bot's host: config-file commands (/setkey, /connect, "
    "/bots, /restart, owner claim), /setup over the setup wizard, the terminal surface",
}

# Directories whose step is done, with the files that may hold non-test code.
MOVED = {"agent": AGENT_ADAPTERS, "nodes": NODE_ADAPTERS, "channels": CHANNEL_ADAPTERS}

TEST_FILE = re.compile(r"(^|/)(tests?|[a-z0-9_]+_tests)\.rs$")


def strip_comments_and_strings(text: str) -> str:
    """Blank out comments and string literals so braces in them don't count."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == "r" and re.match(r'r#*"', text[i:]) and (i == 0 or not text[i - 1].isalnum()):
            hashes = re.match(r"r(#*)\"", text[i:]).group(1)
            end = '"' + hashes
            j = text.find(end, i + 2 + len(hashes))
            j = n if j < 0 else j + len(end)
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        elif c == "'" and re.match(r"'(\\.|[^\\'])'", text[i:]):
            m = re.match(r"'(\\.|[^\\'])'", text[i:])
            out.append(" " * len(m.group(0)))
            i += len(m.group(0))
        else:
            out.append(c)
            i += 1
    return "".join(out)


def item_end(text: str, start: int) -> int:
    """The end of the item starting at `start`: its `;` or matching `}`."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
            if depth == 0 and c == "}":
                # A block item ends here; a `use a::{b, c};` at its `;`.
                rest = text[i + 1 :]
                k = i + 1 + len(rest) - len(rest.lstrip())
                return k + 1 if text.startswith(";", k) else i + 1
        elif c == ";" and depth == 0:
            return i + 1
        i += 1
    return len(text)


CFG_TEST = re.compile(r"#\[cfg\((test|all\(test[^\]]*|any\(test[^\]]*)\)\]")


def non_test_code(path: Path) -> list[str]:
    """The lines of `path` outside `#[cfg(test)]` items, comments and blanks."""
    text = strip_comments_and_strings(path.read_text())
    while True:
        m = CFG_TEST.search(text)
        if not m:
            break
        # The item the attribute applies to (skipping further attributes).
        j = m.end()
        while True:
            k = len(text) - len(text[j:].lstrip())
            if text.startswith("#[", k):
                j = item_end(text, k)
                continue
            break
        end = item_end(text, j)
        text = text[: m.start()] + re.sub(r"[^\n]", " ", text[m.start() : end]) + text[end:]
    return [line.strip() for line in text.splitlines() if line.strip()]


# Module plumbing: declarations that hold no behavior.
PLUMBING = re.compile(
    r"^(#!?\[(path|allow|cfg|doc)[^\]]*\]"
    r"|(pub(\([^)]*\))? )?(mod [a-z_0-9]+ *;|use [^;]*;?|mod [a-z_0-9]+ *\{|\}|\};?)"
    r"|[A-Za-z_:0-9{}, *]+[,;]?)$"
)


def plumbing_only(lines: list[str]) -> list[str]:
    """The lines that are not module plumbing (a `use` may span lines)."""
    offending = []
    in_use = False
    for line in lines:
        if in_use:
            in_use = not line.endswith(";")
            continue
        if re.match(r"^(pub(\([^)]*\))? )?use ", line) and not line.endswith(";"):
            in_use = True
            continue
        if not PLUMBING.match(line):
            offending.append(line)
    return offending


MOD_DECL = re.compile(
    r"((?:#\[[^\]]*\]\s*)*)(?:pub(?:\([^)]*\))?\s+)?mod\s+([a-z_0-9]+)\s*;"
)


def test_only_files(base: Path) -> set[Path]:
    """Files under `base` that only a `#[cfg(test)] mod x;` brings in."""
    test_only: set[Path] = set()
    for path in base.rglob("*.rs"):
        text = strip_comments_and_strings_keep_paths(path.read_text())
        for m in MOD_DECL.finditer(text):
            attrs, name = m.group(1), m.group(2)
            if not CFG_TEST.search(attrs):
                continue
            explicit = re.search(r'#\[path\s*=\s*"([^"]+)"\]', attrs)
            if explicit:
                target = path.parent / explicit.group(1)
            elif path.name in ("mod.rs", "lib.rs", "main.rs"):
                target = path.parent / f"{name}.rs"
                if not target.exists():
                    target = path.parent / name / "mod.rs"
            else:
                target = path.parent / path.stem / f"{name}.rs"
            test_only.add(target.resolve())
    return test_only


def strip_comments_and_strings_keep_paths(text: str) -> str:
    """Comments removed; strings kept (for `#[path = "..."]`)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)//.*$", "", text)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
    src = root / "src"
    errors: list[str] = []

    # Only entries that hold Rust code count; empty or data-only folders
    # (e.g. one a local run left behind) are not modules.
    present = {
        p.name
        for p in src.iterdir()
        if not p.name.startswith(".")
        and (p.suffix == ".rs" or (p.is_dir() and any(p.rglob("*.rs"))))
    }
    for name in sorted(present):
        if name not in SHELL and name not in PENDING and name not in REEXPORTS:
            errors.append(
                f"src/{name}: not on the app's module lists. Agent behavior belongs in a "
                "crate; add it to SHELL in scripts/check-app-boundary.py only if it is "
                "wiring or the product shell."
            )
    for name in sorted(set(PENDING) - present):
        errors.append(f"src/{name}: moved out; drop it from PENDING in scripts/check-app-boundary.py.")

    for name in sorted(REEXPORTS & present):
        path = src / name
        files = sorted(path.rglob("*.rs")) if path.is_dir() else [path]
        test_only = test_only_files(path) if path.is_dir() else set()
        for file in files:
            rel = file.relative_to(src).as_posix()
            if TEST_FILE.search(rel) or file.resolve() in test_only:
                continue
            offending = plumbing_only(non_test_code(file))
            if offending:
                errors.append(
                    f"src/{rel}: an old-path re-export holds code "
                    f"(first: `{offending[0][:80]}`). Put it in a crate and re-export it."
                )
    for name in sorted(REEXPORTS - present):
        errors.append(f"src/{name}: gone; drop it from REEXPORTS in scripts/check-app-boundary.py.")

    for moved, adapters in MOVED.items():
        base = src / moved
        test_only = test_only_files(base)
        for path in sorted(base.rglob("*.rs")):
            rel = path.relative_to(base).as_posix()
            if rel in adapters or TEST_FILE.search(rel) or path.resolve() in test_only:
                continue
            offending = plumbing_only(non_test_code(path))
            if offending:
                errors.append(
                    f"src/{moved}/{rel}: non-test code outside the adapter list "
                    f"(first: `{offending[0][:80]}`). Move it into a crate behind the "
                    "host traits, or gate it with #[cfg(test)] if only tests use it."
                )
        for rel in sorted(adapters):
            if not (base / rel).exists():
                errors.append(f"src/{moved}/{rel}: gone; drop it from its adapter list.")

    for error in errors:
        print(f"error: {error}")
    if errors:
        print(f"\n{len(errors)} app boundary error(s).")
        return 1
    print("App boundary holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
