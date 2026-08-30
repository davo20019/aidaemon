# AGENTS.md / CLAUDE.md

This is the shared coding-agent guide for this repository. `AGENTS.md` is a
symlink to this file, so keep the content agent-neutral unless a rule truly only
applies to one tool.

## Build & Run

```bash
cargo build
cargo build --release
cargo build --features browser
cargo build --features discord
cargo build --features slack
cargo build --features encryption
cargo build --features "browser,slack,discord"
```

```bash
cargo test
cargo test --all-features
cargo test router
cargo test --lib memory
cargo test integration_tests
cargo test <test_name>
```

```bash
cargo fmt
cargo fmt --check
cargo clippy --all-features -- -D warnings
```

No `rustfmt.toml` is present; use default Rust formatting.

The agent loop is a deep async state machine and debug-profile poll frames are
large. The daemon runtime already uses an 8 MiB worker stack (`src/lib.rs`);
CI runs tests with `RUST_MIN_STACK=8388608` so test threads get the same budget. If a local `cargo test` aborts with
`has overflowed its stack`, run it with that variable set rather than
shrinking a frame by hand; the suite sits within a few percent of libtest's
2 MiB default.

## Pre-Commit Checklist

Before committing release-quality code:

1. `cargo fmt`
2. `cargo clippy --all-features -- -D warnings`
3. `cargo test`

For CI parity on broad changes, prefer:

```bash
cargo test --all-features
cargo test --lib harness_eval --all-features
cargo build --release --features "browser"
```

## Git Safety

This repository often has active, unrelated work in the tree. Do not clean up,
discard, or revert files unless the user explicitly asks for that exact action.

Never run broad destructive commands such as:

```bash
git checkout *
git checkout .
git restore .
git reset --hard
git clean -fd
```

If a change needs to be reverted, inspect the diff first and revert only the
specific lines or files you changed. Prefer `git diff`, `git status --short`,
and targeted patches. When unrelated files are modified, leave them alone.

## Release Notes

Release flow:

1. Bump `Cargo.toml`.
2. Add a `CHANGELOG.md` entry using Keep a Changelog sections.
3. Run the pre-commit checklist.
4. Commit all release changes, including `Cargo.lock`.
5. Push to `master`.
6. Tag with `git tag vX.Y.Z` after checks pass, then push the tag.
7. Create a GitHub release.

`cargo publish` packages only git-tracked files. Do not publish with
`--allow-dirty`. If package upload fails for size, inspect contents with:

```bash
cargo package --list
```

## Project Shape

`aidaemon` is a single Rust daemon for personal AI-agent workflows across chat
channels, local tools, MCP integrations, persistent memory, goals, scheduling,
and delegated agent work.

Current high-level flow:

```text
main.rs -> config.rs -> core.rs -> startup/* -> channels + agent + background tasks
```

Where to look first:

- `src/core.rs` wires the daemon together and coordinates startup modules.
- `src/startup/` owns most subsystem setup: stores, provider/router, tools,
  skills, MCP, memory pipeline, channel startup, DB security, and final wiring.
- `src/agent/` contains the agent runtime. Important subareas are `loop/`,
  `runtime/`, `intent/`, `policy/`, `tools/`, `specialists/`, and `eval/`.
- `src/tools/` contains LLM-callable tools plus deterministic local tools such
  as file read/write/edit/search, command execution, git helpers, browser,
  computer-use, MCP, OAuth/API management, memory, goals, and diagnostics.
- `src/traits.rs` is a thin re-export layer. The actual trait definitions live
  under `src/traits/`.
- `src/state/sqlite/` is the SQLite persistence implementation, split by domain:
  facts, messages, goals, people, skills, dynamic bots/agents/MCP, OAuth,
  settings, prompt snapshots, notifications, token usage, and migrations.
- `src/events/` stores immutable activity events and event-derived context.
- `src/memory/` handles embeddings, retrieval/scoring, retention, procedures,
  people intelligence, task learning, proactive memory, and background
  consolidation.
- `src/plans/` handles persistent multi-step plans and recovery.
- `src/channels/` contains Telegram, Slack, Discord, channel connection, command,
  attachment, approval, and hub routing code.
- `specialists/` contains bundled specialist prompts used by `spawn_agent`.
- `tests/` and `src/integration_tests/` contain CLI/package smoke tests and
  agent-loop integration tests.

Prefer reading the current module files over trusting stale architecture notes.
This codebase changes quickly; keep this guide focused on durable invariants and
navigation pointers.

## Project Instruction Loading

For a repo-scoped task, the direct AIDaemon agent loads project guidance through
`src/project_instructions.rs` into the volatile task-context tail. Discovery is
broad-to-specific from the nearest project root to the selected working
directory. At each directory, prefer a non-empty `AGENTS.override.md`, then
`AGENTS.md`, and use `CLAUDE.md` or `GEMINI.md` only as compatibility fallbacks;
`README.md` is never authoritative instruction content.

Bootstrap loads only the hierarchy applicable to the selected working
directory. Before a high-intent file read/write, project inspection, Git action,
or shell command enters a deeper subtree, the central tool prelude resolves any
newly applicable nested instructions, appends them to the system-level task
context, and deliberately defers the whole tool batch without executing it. The
model must review the new guidance and retry. Already-delivered source paths are
task-local deduplicated, broad file searches do not trigger eager instruction
loads, and the execution phase repeats the check after working-directory
injection as a defense-in-depth fallback.

Project instructions apply only inside their directory scope. They cannot
expand channel/workspace authority, grant secret access, authorize destructive
or external actions, or override system/security rules or the user's explicit
current request. Symlinks are accepted only when their canonical target remains
inside the project root. Instruction files and each delivered hierarchy remain
size-bounded. Keep direct-agent and CLI-agent instruction discovery on this
shared resolver so their behavior does not drift.

## Core Traits

The core interfaces are re-exported from `src/traits.rs`:

- `Tool`: LLM-callable capability.
- `Channel`: inbound/outbound chat channel abstraction.
- `StateStore` and domain store traits: persistence facade and SQLite-backed
  implementations.
- `ModelProvider`: LLM backend abstraction.
- Goal, memory, people, dynamic config, dialogue, and trigger-event traits/types
  are split into focused files under `src/traits/`.

When calling store-trait methods on concrete types, import:

```rust
use crate::traits::store_prelude::*;
```

## Tool Schema Rule

`Tool::schema()` must return the full OpenAI function object with `name`,
`description`, and `parameters`. Do not return only the parameters object.

```rust
fn schema(&self) -> Value {
    json!({
        "name": "my_tool",
        "description": "What this tool does and when to use it",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }
    })
}
```

Tool schemas are linted in `src/tools/schema_lint_tests.rs`.

## Dynamic Bot Rule

Channel-connected bots can come from both static config and dynamic database
state. When registering tools or features that depend on channel tokens, check
both:

- `config.all_slack_bots()` for config-based Slack bots.
- `state.get_dynamic_bots().await` for dynamically connected bots.

Do not assume the config file is the only source of connected channels.

## Keyword Matching Rule

For natural-language keyword or phrase matching, use word-boundary matching via
`contains_keyword_as_words()` from `src/agent/intent/intent_routing.rs` (also
re-exported through `agent` helpers). Avoid substring matching for user/LLM text.

Examples:

```rust
contains_keyword_as_words("deploy the app", "deploy");      // true
contains_keyword_as_words("check deployed sites", "deploy"); // false
contains_keyword_as_words("set up monitoring", "set up");    // true
```

Substring matching with `.contains()` is appropriate for structural markers and
protocol fragments such as `[tool_use:`, `[tool_call:`, `[INTENT_GATE]`, or for
long fixed phrases where substring overlap is not a concern.

### Prefer Structural Fixes Over Phrase Patches

Do not fix autonomy, continuation, completion, delegation, retry, recovery, or
routing incidents by adding the observed wording (or a list of its likely
paraphrases) to a classifier. A report such as "the agent stopped when asked a
short status question" is evidence of a missing lifecycle or state invariant,
not a request to recognize that question's words.

Trace the failure to the authoritative state boundary and repair the general
contract there. Prefer persisted and typed signals such as request/task status,
task IDs, tool receipts, outcome metadata, mutation effects, completion
obligations, event relationships, and explicit protocol markers. For example,
an unresolved request should retain its outstanding obligations across a
related turn regardless of how that turn is phrased; a delegated background
step should advance or re-enter its parent objective based on lifecycle state,
not on notification prose.

Natural-language matching is appropriate only when language is itself the
feature (for example, an explicit command syntax) or as a broad advisory hint
that cannot independently decide success, authority, completion, or whether an
objective is abandoned. Making a one-off phrase list word-boundary-safe does
not make it a structural fix.

Regression tests for behavioral fixes should prove the invariant below the
wording layer when possible: construct the typed state/contract directly,
exercise at least one differently phrased continuation, and retain negative
coverage for a genuinely separate request or an explicit user constraint.

### Task-Kernel Freeze: Evidence Beats Contract

The completion contract (`RunAggregate` obligations, cardinalities, evidence
scopes, dispatch-stop rules, response contracts) is an LLM-proposed description
of the work, produced by a small model several times per turn. It is a planning
hint and a fail-closed safety boundary, not the arbiter of whether work
happened. Observed receipts are the ground truth: a run whose terminal receipts
all succeeded or were credited closes as `succeeded` with
`proof_basis=evidence` even when the contract could not credit it
(`RunAggregate::evidence_closed`, `RunTerminalDecision::SucceededByEvidence`).

The ledger-first arbiter (`RunAggregate::closeout`, admissibility in
`src/agent/loop/closeout.rs`) is the intended home for any new "may the loop
demand more work?" decision: authority gates admission, expectations are
verified by receipts, and only a *reachable* expectation may ask for another
pass. It is authoritative: the completion phase asks for more work only on a
`reachable` verdict (`ledger_expectations_required`), and records every
verdict as a `ledger_closeout` decision point. `CompletionContract::authority()`
is the only contract view a gate may read; `expectations()` is verified by
receipts, never enforced directly. Do not add a gate anywhere else.

Do not add new obligation classes, cardinality features, stop-trigger kinds,
or response-projection rules to make a single live test pass. If a live
failure shows the contract disagreeing with receipts, fix it by making the
receipts more typed (adapter/verifier assertions) or by widening what evidence
closes, never by making the kernel stricter. Measure autonomy changes as
N-of-M pass rates over repeated runs, not single-shot rounds; a stochastic
producer makes one pass or one fail meaningless.

## Test & Fixture Data Hygiene

This project is open source and published. Never put real personal data into the
repository: real names, addresses, phone numbers, email addresses, employers,
birthdays, private channel IDs, or other PII.

Use clearly synthetic placeholders in tests, fixtures, docs, specs, comments,
and subagent briefs. Good examples:

- `Alice Rivera`
- `Jordan Lee`
- `Acme Corp`
- `telegram:synthetic-user-1`
- daughters `Mia` and `Zoe`

When debugging with live daemon memory or real user examples, translate the case
to synthetic equivalents before writing tests or docs.

## Testing Guidance

Use focused tests for narrow changes and broader suites for shared behavior.

Useful targets:

```bash
cargo test integration_tests
cargo test --lib memory
cargo test --lib harness_eval --all-features
cargo test proptest
```

Integration tests exercise the real `Agent::handle_message` path with
`MockProvider`, `TestChannel`, temp SQLite state, event/plan stores, embeddings,
and a small default tool set. See `src/testing.rs` for harness helpers.

The first embedding-backed test run may download the fastembed model into
`.fastembed_cache/`.

## CI/CD

GitHub Actions are the source of truth:

- `.github/workflows/ci.yml` runs formatting, clippy, all-feature tests on Linux
  and macOS, harness eval, release build checks, coverage visibility, and release
  tag guarding.
- `.github/workflows/release.yml` gates releases on `cargo test --all-features`,
  builds release artifacts, creates GitHub releases, publishes crates.io, and
  triggers Homebrew tap updates.
- `.github/workflows/update-homebrew.yml` updates the tap after releases.

If this document and workflow files disagree, trust the workflow files and update
this guide.

## Feature Flags

Current important feature flags:

- `browser`: headless browser tool via `chromiumoxide`.
- `discord`: Discord channel via `serenity`.
- `slack`: Slack channel via `tokio-tungstenite`.
- `terminal-bridge`: terminal bridge websocket/crypto/PTY support.
- `encryption`: SQLCipher via `libsqlite3-sys/bundled-sqlcipher`.
- `computer_use`: shared computer-use code.
- `computer_use-macos`: macOS computer-use dependencies and probe support.

Check `Cargo.toml` for the exact current defaults and dependency mapping.

## Debugging with db_probe

The encrypted database can be inspected with `src/bin/db_probe.rs`.

Prerequisites:

- `AIDAEMON_ENCRYPTION_KEY` in the environment or `.env`.
- Optional `AIDAEMON_DB_PATH` (defaults to `aidaemon.db`).

Examples:

```bash
cargo run --bin db_probe --features encryption
cargo run --bin db_probe --features encryption -- --search "error"
cargo run --bin db_probe --features encryption -- --session "telegram:12345"
cargo run --bin db_probe --features encryption -- --task "task-uuid-here"
cargo run --bin db_probe --features encryption -- --invocation 42
cargo run --bin db_probe --features encryption -- --repair-stale-cli 24
cargo run --bin db_probe --features encryption -- --token-hours 24
cargo run --bin db_probe --features encryption -- --fabrication-audit --eval-hours 72
```

`--fabrication-audit` is a post-hoc trace check: it flags tasks whose final
assistant reply claims a side-effecting action (posted/ran/deployed/wrote a
file, etc.) while the task made zero tool calls — a candidate fabricated
completion. It verifies outcomes rather than predicting tool-need up front.

## Specialist System

Specialists are bundled prompts in `specialists/<kind>.md` with optional user
overrides in `~/.aidaemon/specialists/` (`config.specialists_override_dir`).

Implementation lives in `src/agent/specialists/`:

- `parse.rs`: frontmatter and template parsing.
- `registry.rs`: bundled/user override registry.
- `render.rs`: prompt rendering.
- `validation.rs`: validation and clamping.
- `equivalence_tests.rs` and `override_tests.rs`: migration and override tests.

`spawn_agent` exposes specialist selection through `src/tools/spawn.rs` and
runtime support in `src/agent/runtime/spawn.rs`. LLM-visible specialist
descriptions come from `SpecialistRegistry::llm_visible_kinds()` and appear in
both the tool schema and root system prompt.

## MCP / Browser Notes

MCP startup and dynamic MCP state live in `src/startup/mcp.rs`,
`src/tools/manage_mcp.rs`, and `src/state/sqlite/dynamic_mcp.rs`.

When using browser inspection tools, prefer screenshots over full accessibility
snapshots unless element IDs are needed for interaction.
