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
```

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
