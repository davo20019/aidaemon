# AGENTS.md / CLAUDE.md

This is the shared coding-agent guide for this repository. `AGENTS.md` is a
symlink to this file, so keep the content agent-neutral unless a rule truly only
applies to one tool. Keep it to durable invariants and navigation pointers;
prefer reading current module files over trusting architecture notes.

## Build & Test

Build with `cargo build` (add `--release` and/or `--features` from
`Cargo.toml`: `browser`, `discord`, `slack`, `encryption`, `terminal-bridge`,
`computer_use`, `computer_use-macos`). Test with `cargo test [filter]`, e.g.
`integration_tests`, `--lib memory`, `--lib harness_eval --all-features`,
`proptest`. Default `rustfmt`; no `rustfmt.toml`.

Debug-profile agent futures are large. If `cargo test` aborts with
`has overflowed its stack`, run it with `RUST_MIN_STACK=8388608` (CI does; the
daemon uses an 8 MiB worker stack in `src/lib.rs`) rather than shrinking a
frame by hand.

Integration tests drive the real `Agent::handle_message` path with
`MockProvider`, `TestChannel` and temp stores; helpers are in `src/testing.rs`.
The first embedding-backed run downloads the fastembed model into
`.fastembed_cache/`.

## Pre-Commit Checklist

1. `cargo fmt`
2. `cargo clippy --all-features -- -D warnings`
3. `cargo test`

For CI parity on broad changes also run `cargo test --all-features`,
`cargo test --lib harness_eval --all-features` and
`cargo build --release --features "browser"`. CI lives in
`.github/workflows/` (`ci.yml`, `release.yml`, `update-homebrew.yml`); if this
guide and the workflows disagree, trust the workflows and fix this guide.

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

## Releases

Bump `Cargo.toml`, add a Keep a Changelog entry to `CHANGELOG.md`, run the
pre-commit checklist, commit everything including `Cargo.lock`, push to
`master`, then tag `vX.Y.Z`, push the tag and create the GitHub release.
`cargo publish` packages only git-tracked files: never use `--allow-dirty`;
inspect size problems with `cargo package --list`.

## Project Map

Flow: `main.rs -> config.rs -> core.rs -> startup/* -> channels + agent +
background tasks`.

- `src/startup/`: subsystem setup (stores, provider/router, tools, skills, MCP,
  memory, channels, DB security, wiring).
- `src/agent/`: runtime. `turn/` (bootstrap, tool offering, closeout) hands
  each turn to the SDK engine in `runtime/sdk_engine/`.
- `src/tools/`: LLM-callable and deterministic local tools.
- `src/traits.rs` and `src/types.rs` re-export the contract types and store
  traits, which live in the crates that own them; `src/traits/state_store.rs`
  keeps only what the app assembles itself (`StateStore`, `MandateToolStore`,
  `SkillsStore`, `HealthCheckStore`).
- `src/state/sqlite/`: SQLite persistence by domain. Goals, mandates, tasks,
  work coordination and notifications: `crates/aidaemon-work`. Memory:
  `aidaemon_memory::sqlite`. Events and run aggregate:
  `crates/aidaemon-events` (host adapter in `src/events/`).
- `src/memory/`, `src/plans/`, `src/channels/` (Telegram, Slack, Discord, hub).
- `src/setup/`: first run. No config runs setup mode (the gateway crate's
  `setup` router, no agent) until a model passes its test; the running
  daemon's client host then serves the chat step and the greeting.
- Clients: `clients/app/` is the one nsh (Native Shell) Rust app for desktop, web and terminal (its own Cargo workspace; read `clients/app/AGENTS.md`). Shared client logic is `crates/aidaemon-client-core` (`machine::LinkMachine`, reducer, sign-in). The gateway serves its web build (installed to `~/.aidaemon/app-web`) at `/client/v1/next/`; releases ship `aidaemon-app` and `aidaemon-app-web.tar.gz`.
- Specialists: prompts in `specialists/<kind>.md`, user overrides in
  `~/.aidaemon/specialists/`, code in `src/agent/specialists/`, exposed via
  `spawn_agent`.
- MCP: `src/startup/mcp.rs`, `src/tools/manage_mcp.rs`,
  `src/state/sqlite/dynamic_mcp.rs`.

## Project Instruction Loading

The daemon's agents load project guidance through `src/project_instructions.rs`
(one shared resolver for direct and CLI agents; keep it that way). Per
directory it prefers `AGENTS.override.md`, then `AGENTS.md`, then `CLAUDE.md` /
`GEMINI.md`; never `README.md`. Nested instructions entering scope defer the
tool batch so the model reviews them first. Project instructions apply only in
their directory scope and can never expand authority, grant secrets, authorize
destructive or external actions, or override system/security rules or the
user's explicit request. Symlinks must resolve inside the project root.

## Core Traits

The core interfaces are re-exported from `src/traits.rs`:

- `Tool`: LLM-callable capability.
- `HubChannel`: a bot as the channel hub sees it (where a session's
  messages, media and approvals go). The transport underneath is an SDK
  adapter (`aidaemon_channels::Channel`); session turns run on
  `channels::turns::TurnRunner` and prompts on `channels::prompts::Prompts`,
  so a bot supplies only access, session ids, commands and a `TurnSurface`.
- `StateStore` and domain store traits: persistence facade and SQLite-backed
  implementations.
- `ModelProvider`: LLM backend abstraction.
- Goal, memory, people, dynamic config, dialogue, and trigger-event traits/types
  are defined in their crates and re-exported by focused files under `src/traits/`.

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

## Runtime Invariants

One line each; the full rules live next to the code. Read
`src/agent/CLAUDE.md` before changing completion or closeout, approvals,
trust or taint, or any tool's effects, prompts or refusals, and
`src/llm_runtime/CLAUDE.md` before adding a model call outside the agent loop.

- Structural fixes over phrase patches: never fix an autonomy, continuation,
  completion, delegation, retry, recovery or routing incident by matching the
  observed wording; repair the typed state invariant.
- Evidence beats contract: receipts are the ground truth; the completion
  contract is a planning hint and never a source of tool authority. A new
  "may the loop demand more work?" decision is a `closeout::StopDemand`, not
  a new gate. Never make the kernel stricter to pass one live test; measure
  N-of-M.
- Untrusted content: trust, egress and taint are typed per adapter
  (`output_trust`, `call_annotations`, `call_outbound`); never key them on
  tool names or on request/content wording.
- Owner approval: every safeguard that can refuse, approve or scan a tool
  call is a kernel hook judging typed effects. A tool declares its
  `CallEffects` and never prompts the owner or refuses on policy inside
  `call()`.
- Side calls: a model call outside the agent loop is a declared `SideCall`
  made through `SideCallGateway`; never hold a raw provider elsewhere.

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

## Debugging with db_probe

`src/bin/db_probe.rs` inspects the encrypted DB (needs
`AIDAEMON_ENCRYPTION_KEY`; optional `AIDAEMON_DB_PATH`, default `aidaemon.db`):
`cargo run --bin db_probe --features encryption -- <flags>`. Useful flags:
`--search`, `--session`, `--task <uuid>`, `--invocation`, `--token-hours N`,
`--repair-stale-cli N`, and with `--eval-hours N`:

- `--fabrication-audit`: final replies claiming a side effect from tasks that
  made zero tool calls.
- `--gate-outcomes`: each gate's interventions labelled with the run's terminal
  status. Check it before calling a gate "too strict" or "working".
- `--cache-reuse`: prompt-cache reuse by model, first vs later calls. Check it
  before and after changing message assembly or provider request building.
- `--decisions`: decision-layer calls per decision and backend (latency,
  outcomes, answers over threshold and acted on), and the turn digest's
  relationship against the committed one by confidence band.

When using browser inspection tools, prefer screenshots over full accessibility
snapshots unless element IDs are needed for interaction.
