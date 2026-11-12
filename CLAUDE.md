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
- `src/traits.rs` re-exports traits defined under `src/traits/`.
- `src/state/sqlite/`: SQLite persistence by domain. Goals, mandates, tasks,
  work coordination and notifications: `crates/aidaemon-work`. Memory:
  `aidaemon_memory::sqlite`. Events and run aggregate:
  `crates/aidaemon-events` (host adapter in `src/events/`).
- `src/memory/`, `src/plans/`, `src/channels/` (Telegram, Slack, Discord, hub).
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
scopes, dispatch-stop rules, response contracts) is a description
of the work: the exact resources the request names, plus whatever the executing
model declares through `track_requirements`. It is a planning hint, not the
arbiter of whether work happened and not a source of tool authority: that comes
only from the explicit `Authority` (role, workspace grant, goal workspace,
session mode, config) enforced by the kernel `AuthorityHook`, and a no-tools
contract may only reduce what the loop demands. Observed receipts are the
ground truth: a run whose terminal receipts all succeeded or were credited
closes as `succeeded` with `proof_basis=evidence` even when the contract could
not credit it (`RunAggregate::evidence_closed`,
`RunTerminalDecision::SucceededByEvidence`).

The ledger-first arbiter (`RunAggregate::closeout`, admissibility in
`src/agent/turn/closeout.rs`) is the intended home for any new "may the loop
demand more work?" decision: the explicit Authority gates admission,
expectations are verified by receipts, and only a *reachable* expectation may
ask for another pass. It is authoritative: the completion phase asks for more
work only on a `reachable` verdict (`ledger_expectations_required`), and
records every verdict as a `ledger_closeout` decision point. The SDK engine's
other stop-time asks (mandate decision commit, GUI click verification, reply
repair) are `StopDemand`s judged by `closeout::stop_demand_verdict` against the
same authority and visible tools, each with a budget, and recorded as
`stop_demand` decision points; a new stop-time ask is a new `StopDemand`, not a
new check. Tool authority is never read from the contract. `expectations()` is
verified by receipts, never enforced directly. Do not add a gate anywhere else.

Do not add new obligation classes, cardinality features, stop-trigger kinds,
or response-projection rules to make a single live test pass. If a live
failure shows the contract disagreeing with receipts, fix it by making the
receipts more typed (adapter/verifier assertions) or by widening what evidence
closes, never by making the kernel stricter. Measure autonomy changes as
N-of-M pass rates over repeated runs, not single-shot rounds; a stochastic
producer makes one pass or one fail meaningless.

Expectations come from the executing model. The structural contract only names
exact resources; the model's `track_requirements`
checklist (`ExecutorExpectationsDeclared` → `obligation:checklist:<n>`) is the
model's own typed statement of what it will do, so open reachable checklist
items are demanded even after other work completed and a run that ends with
one open is `partial`, not evidence-closed. Only receipts close them.
An item the model marks `deferred`/`skipped` or drops from its list becomes
`RunObligationState::Deferred`: the closeout never demands it (stopping is
permitted, `UnreachableReason::ExecutorDeferred`), but it never counts toward
`is_fulfilled`, `work_is_fulfilled` or `evidence_closed`, so the run ends
`partial`. Editing a checklist never turns unperformed work into
accomplished work. In the SDK kernel the same split is
`StopDecision::AcceptUnmet` and `RunOutcome::is_succeeded()`: `Completed`
means the run stopped, not that its objective was met.
The model's own verdict is `track_requirements`' `outcome`
(`done | partial | blocked`, `RunAggregate::executor_claimed_outcome`):
`partial`/`blocked` stop the run with `AcceptUnmet` and cap the task at
`partial`; receipts can downgrade a claimed `done`, nothing upgrades the
other two. A run closed by a policy denial is `partial` too: the work did
not happen, and a denial is final, never retried as a failure.

## Untrusted Content Rule

After a run reads content someone other than the owner wrote (web pages,
trigger input, other people's messages, MCP results, files outside the trusted
roots), its egress calls are interrupted by the kernel's `UntrustedHook` and
run only if a clean, tool-less verifier
(`src/agent/runtime/untrusted.rs`) that sees just the original request and the
proposed call finds them consistent. Owner level `unrestricted` records these
calls instead of checking them.

- Source trust is typed per adapter: each tool declares `output_trust` from
  its capabilities and arguments. The result boundary
  (`tool_result_fit.rs`) sanitizes and wraps from that trust, once, for both
  the model's view and the stored history.
- Egress is the `ToolAnnotations.egress` bit, derived per call from typed
  mutation effects (`call_annotations`).
- A read can still carry data out through what it sends. A tool declares that
  with `Outbound` (`call_outbound`): a string already present word for word in
  a user message or tool result holds nothing the run composed and passes; any
  other URL or a search query goes to the verifier after an untrusted read.
- A call the verifier does not pass goes to the owner as a one-time card when
  the owner wrote the request in a private chat
  (`refused_check_may_ask_owner`); a decline binds the run through
  `Resolution::OwnerDeclined`. Every other run keeps the verifier's refusal.
- Tool output of a tainted turn is left out of conversation summaries, and a
  summary that folded in a tainted turn taints later turns until `/clear` or
  three clean rewrites.
- A turn starts tainted from explicit facts only (`starts_tainted`): role,
  channel visibility, trigger session, a tainted turn still in the projected
  prefix, or a tainted parent.
- Never key trust, egress or taint on tool names or on request/content
  wording.

## Owner Approval Rule

The rule: every safeguard that can refuse, approve or scan a tool call is a
kernel hook judging typed effects. A tool must not prompt the owner or refuse
on policy inside `call()`; a decline binds the run for that effect and target
across tools. Do not add a new in-tool prompt or policy refusal.

Every tool now asks through the kernel: `config_manager`, `share_memory`,
`cli_agent`, `browser`, `computer_use` and the `manage_*` tools that need the
owner build their card in `owner_approval_request`, and startup passes tools
no approval broker. The self-updater (`src/updater.rs`) is not a tool and
still asks through its own `ApprovalBroker`. `cli_agent`, `browser` and `computer_use`
judge state they read before the call: they return the card from
`owner_approval_request`, pick its buttons with `Tool::approval_card_kind`,
keep the kernel's answer (`keeps_own_grants`, `record_owner_answer`) for the
exact call or session it covers, and refuse a call at execution that needs
the owner and has no matching approval.

A call whose only effect is the agent's own internal state (its memory,
people store, checklist or goal bookkeeping) declares `CallEffects::none()`,
so a decline never makes it ask. Anything that sends, schedules, spawns work,
changes config or credentials, or widens who can see data keeps declaring its
effects (or `unknown()`).

- Tools declare a call's effects (`EffectKey`: egress `host:port`, absolute
  paths for delete/write/configuration, named credential/process). The
  kernel refuses any call overlapping `RunState::declined` with
  `[OWNER DECLINED]` before any hook runs, and sends no new card.
- `aidaemon_policies::ApprovalPolicy` asks; the app supplies only its rules
  (`AppApprovalRules` in `src/agent/runtime/approvals.rs`: the http_request
  and terminal ladders, with turn facts, never `_` arguments), its grant
  stores, and the resolver that sends the card. Timeout, send failure and
  Deny all decline. `EgressGuard` refuses secrets in outbound calls with
  `[EGRESS BLOCKED]` before anyone is asked.
- After a decline, a call whose effects are incomplete is asked again even
  when a rule or grant would allow it: an egress or destructive one after any
  decline, any non-read-only one after a declined delete, write,
  configuration or call.
- A child agent starts with its parent run's declines
  (`InheritedAuthority::declined`, seeded into the child's `RunState` after
  its user message is set). Session grants are shared with children.
- The dispatcher runs a terminal call exactly as the kernel judged it,
  keyed by run and call id, with the judged scope; a terminal call with no
  judgment is refused ("requires approval through the agent kernel").
- A run in which the owner declined something never ends `Succeeded`.

## Side Call Rule

A model call outside the agent loop is a side call. Declare it as a
`SideCall` in `src/llm_runtime/side_calls.rs` (purpose, class, model, total
deadline, failure policy) and make it through `SideCallGateway`
(`src/llm_runtime/gateway.rs`): `ask` for a prompt you own, `provider` for
library code that takes a provider. Do not hold a raw provider or record a
call by hand anywhere else; `llm_runtime::door_tests` fails if you do.

A closed question for the optional decision backend (`[decisions]`,
`aidaemon_decisions::Decider`) is a `Decision` in
`src/llm_runtime/decisions.rs`, asked through `SideCallGateway::decide`; it
may skip optional work, pick between equal-authority paths or make a check
stricter, never grant authority or decide completion.

A safety judge refuses on any failure. A memory job skips, and never writes
from a tainted run. A side call runs no tools. A request that forces one
named tool keeps that tool, because that is how it asks for structured output:
the call comes back to the caller and nothing executes it. Work that needs
tools to run or a second pass is a kernel agent, not a side call.

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
