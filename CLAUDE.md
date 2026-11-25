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

`workspace-hack/` (cargo-hakari) unifies third-party features so `-p` and
feature-flag builds reuse one build of each dependency. After adding,
removing or re-featuring a dependency run `cargo hakari generate` and
`cargo hakari manage-deps`; after changing which features one workspace
crate asks of another run `scripts/dev/unify-dev-features.py` (it gives each
crate's dev-dependencies the workspace build's feature sets, so
`cargo test -p` reuses that build). CI's `hakari` job checks both. Target dirs only
grow; `scripts/dev/target-gc.sh` prunes every checkout's to a budget
(`scripts/dev/install-target-gc.sh` runs it hourly, and harder when the disk is low).

Integration tests drive the real `Agent::handle_message` path with
`MockProvider`, `TestChannel` and temp stores; helpers are in `src/testing.rs`.
The first embedding-backed run downloads the fastembed model into
`.fastembed_cache/`.

## Pre-Commit Checklist

1. `cargo fmt`
2. `cargo clippy --all-features -- -D warnings`
3. `cargo test`

A pre-push hook (`scripts/git-hooks/pre-push`) runs the checks CI most often
fails on, scoped to what the push changes: app boundary, fmt, clippy (daemon
and SDK crates, all targets), rustdoc `-D warnings` per changed SDK crate, the
`aidaemon-core` public API snapshot, the tool schema goldens and the app's
clippy. Install it once per clone with `scripts/git-hooks/install.sh` (each worktree
then runs its own branch's copy).
A failure only in files with uncommitted edits (another session's) warns
instead of blocking. When a tool schema or the core API changes on purpose,
update its snapshot in the same commit:
`UPDATE_TOOL_GOLDENS=1 cargo test --all-features --lib catalog_` and
`scripts/check-core-api.sh --bless`. Tests must not assume the host's time
zone (CI runs on UTC), a fast shell (Windows) or a Unix-only command.

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

The next release has an integration branch, `release/X.Y.Z` (now
`release/0.24.1`): feature branches start from it and their pull requests
target it (CI runs on `release/**`), and it merges to `master` when the
release is ready. Do not push feature work straight to `master`.

Bump `Cargo.toml`, add a Keep a Changelog entry to `CHANGELOG.md`, run the
pre-commit checklist, commit everything including `Cargo.lock` and push to
`master`. `tag-release.yml` tags `vX.Y.Z` once CI passes on that commit and
starts `release.yml`, which reuses CI's result, builds, creates the GitHub
release and publishes the crates in `scripts/publish-order.py` order (CI's
"Package crates" job packages every crate on each pull request and builds
each from its package on master, so a release never meets an unbuilt one).
`cargo publish` packages only git-tracked files: never use `--allow-dirty`;
inspect size problems with `cargo package --list`.

## Project Map

Flow: `main.rs -> config.rs -> core.rs -> startup/* -> channels + agent +
background tasks`.

- `src/startup/`: subsystem setup (stores, provider/router, tools, skills, MCP,
  memory, channels, DB security, wiring).
- `src/agent/`: the daemon's `Agent` (construction, host ports) and the
  runtime's tests on it. The runtime is `crates/aidaemon-runtime-core/src/`
  (everything but the built-in tools and the agent, which are
  `crates/aidaemon-runtime/src/{tools,agent}`; that crate re-exports core's
  modules, so `aidaemon_runtime::turn` paths are unchanged):
  `turn/` (bootstrap, closeout, completion, memory), `engine/` (tool roster,
  gate, dispatch) and `engine/turn_engine/`, one turn on the SDK kernel.
- Tools: `crates/aidaemon-tools` and `crates/aidaemon-runtime/src/tools/`
  (terminal, cli_agent, ...); `src/tools/` holds the app-assembled ones and
  re-exports.
- `src/traits.rs` and `src/types.rs` re-export the contract types and store
  traits, which live in the crates that own them; `src/traits/state_store.rs`
  keeps only what the app assembles itself (`StateStore`, `MandateToolStore`,
  `SkillsStore`, `HealthCheckStore`).
- State: `crates/aidaemon-runtime-core/src/state/` (the SQLite `StateStore` and
  its schema; `src/state/` re-exports). Goals, mandates, tasks, work
  coordination and notifications: `crates/aidaemon-work/src/sqlite/`; task
  plans: `crates/aidaemon-work/src/plans/`. Memory: `crates/aidaemon-memory`
  (`aidaemon_memory::sqlite`; host adapters in `aidaemon_runtime::memory`).
  Events and run aggregate: `crates/aidaemon-events` (host adapter in
  `src/events/`).
- Channels: bots in `crates/aidaemon-channel-{telegram,slack,discord}`, hub,
  turns and prompts in `crates/aidaemon-channels`; `src/channels/` is the
  daemon's side (config, commands).
- `src/setup/`: first run. No config runs setup mode (the gateway crate's
  `setup` router, no agent) until a model passes its test; the running
  daemon's client host then serves the chat step and the greeting.
- Clients: `clients/app/` is the one nsh (Native Shell) Rust app for desktop, web and terminal (its own Cargo workspace; read `clients/app/AGENTS.md`). Shared client logic is `crates/aidaemon-client-core` (`machine::LinkMachine`, reducer, sign-in). The gateway serves its web build (installed to `~/.aidaemon/app-web`) at `/client/v1/next/`; releases ship `aidaemon-app` and `aidaemon-app-web.tar.gz`.
- `deploy/`: setup for the project's public servers, `relay.aidaemon.ai`
  (iroh-relay, the remote-access default) and `dns.aidaemon.ai` (address
  lookup). They must run the workspace's iroh version; see `deploy/README.md`.
- Specialists: prompts in `crates/aidaemon-runtime-core/specialists/<kind>.md`, user overrides in
  `~/.aidaemon/specialists/`, code in `crates/aidaemon-runtime-core/src/specialists/`, exposed via
  `spawn_agent`.
- MCP: `src/startup/mcp.rs`, `crates/aidaemon-mcp` (`manage_mcp` is
  `registry/manage.rs`), `crates/aidaemon-runtime-core/src/state/dynamic_mcp.rs`.
- Project memory: notes per project id in `aidaemon_memory::project`
  (files, never the database); a turn's project is resolved from typed state
  in `aidaemon-runtime-core/src/turn/turn_project.rs` and gated by
  `turn::memory::memory_project` (owner, private/internal, no mandate).
  Personal-only categories never become notes (`project::route`).

## Project Instruction Loading

The daemon's agents load project guidance through
`crates/aidaemon-runtime-core/src/turn/project_instructions.rs`
(one shared resolver for direct and CLI agents; keep it that way). Per
directory it prefers `AGENTS.override.md`, then `AGENTS.md`, then `CLAUDE.md` /
`GEMINI.md`; never `README.md`. The owner's `~/.aidaemon/AGENTS.md` comes
first, as the broadest file. Nested instructions entering scope defer the
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

Decisions come from typed state and structured signals (receipts, outcome
metadata, annotations, typed errors, schema fields), never from matching words
in user, model or tool prose. Keyword or substring matching is kept only for:

- exact command words (`/clear`, a CLI flag);
- markers the daemon itself writes (`[SYSTEM]`, `[UNTRUSTED EXTERNAL DATA`,
  tool-call protocol tokens), preferably through a shared constant;
- machine formats: shell command lines (tokenized where cheap), URLs, paths,
  JSON, DOM class/id markers;
- external text with no structured signal, narrowed and commented with why.

Each kept match says which of these it is in a comment next to it.

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
