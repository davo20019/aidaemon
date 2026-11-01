# aidaemon SDK: plan

This is the working plan for turning aidaemon into a framework people can
build their own agents with. It is written so that any later session, human
or agent, can pick up where the last one stopped. `DESIGN.md` explains *why*
the kernel looks the way it does; this file says *what is done, what is next,
and how to resume*.

Keep it current: when a step finishes, tick it, add the commit, and move the
"Next up" marker.

## Where things stand

| Item | State |
|---|---|
| Branch | `ccr-986102ee-12nt48` (branched from `master` at `4a36e32`) |
| Crates | `aidaemon-core` plus 13 SDK crates (see `DESIGN.md`, "The SDK crates"), all `publish = false` |
| Tests | all passing across the 14 crates (incl. conformance kits); clippy and fmt clean |
| CI | Lint and tests for every SDK crate run in `ci.yml`. macOS job never gets a runner (see "Known issues") |
| PR | #13 was closed without merging; the branch was kept. Open a new PR only when asked |
| Research | Harness Field Guide: https://claude.ai/artifact/Fm5Ax7oX2vzmACjtb7iPxF |

### What exists

- **Kernel** (`aidaemon-core`): agent builder, streaming `Provider` + `Retry`,
  `Tool` with annotations, access manifests and timeout hints, five hook
  points, receipts for every call, typed outcomes, serializable state with
  interrupt/resume, crash-safe in-flight batches, typed message origins,
  events, `RunStore`, limits, stuck detection, `testing` helpers.
- **SDK crates**: providers (OpenAI-compatible, Anthropic), tools (file,
  search, sandboxed shell, web, checklist), SQLite store, MCP client,
  agents-as-tools, policies (approvals, scope, completion, redaction,
  untrusted output, project instructions, context window, memory, tracing),
  eval kit, swappable long-term memory ported from the daemon, chat
  channels (runner + Telegram, Slack, Discord adapters), and the
  `aidaemon-sdk` bundle with a `coding_agent` example.

## Decisions already made

1. **Same repo, new crates.** The SDK grows as workspace crates in this repo,
   so it can reuse `harness_eval`, providers and tools, and so the daemon
   becomes its first user. No separate repository.
2. **Small kernel, everything else is a hook.** If a feature cannot be built
   on the public hooks, add a hook point; do not add a kernel primitive.
3. **New loop, not an extraction.** `src/agent/loop/` (~62k lines) is not
   moved into the SDK. Daemon policies are re-implemented as hooks one at a
   time.
4. **Port only what measures well.** A daemon policy is ported only if
   repeated live runs (N-of-M pass rates, not single runs) show it helps.
5. **Receipts are the source of truth** for "did the work happen"; completion
   checks read receipts, not the model's prose.

## Phase 0 decisions (settled 2026-10-31)

Kept the sketch's defaults, now part of the API:

- [x] Tool errors go back to the model as error results; they never end a run.
- [x] Provider outages and limits are `Ok(RunOutcome { status })`; `Err` is
      only misuse and store failures.
- [x] Pausing for a human is a hook decision (`Interrupt`), resumed with
      `Agent::resume`.
- [x] Instructions are re-sent on every request and never stored.
- [x] Naming: `aidaemon-core` is the kernel; `aidaemon-sdk` is the
      user-facing bundle.

## Roadmap

Each phase ends in something runnable and tested. Phases 1-3 can overlap.

### Phase 0: lock the primitives (done)
### Phase 1: a real model (done)
OpenAI-compatible and Anthropic providers with streaming and prompt caching,
tested against local mock servers. **Not yet run against a live model**:
needs an API key (see Phase 4).

### Phase 2: tools worth having (done)
Workspace-confined tools, sandbox backends (local, bubblewrap, seatbelt,
docker), access manifests. Bubblewrap and seatbelt were only tested on the
argument level here (not installed in the build container).

### Phase 3: durability (done)
SQLite store with versioned migrations; crash recovery without repeating
non-idempotent calls, tested by killing a run mid-tool and reopening the
database.

### Phase 4: measurement  ← Next up
- [x] `aidaemon-eval`: scenarios, trajectory assertions, N-of-M reports,
      baseline comparison, 5 scripted scenarios.
- [ ] Live runs: run `crates/aidaemon-eval/scenarios` and the
      `coding_agent` example against real models (needs API keys).
- [ ] Old-vs-new comparison: translate the daemon's 16 fixtures in
      `tests/harness_eval/fixtures/` into `aidaemon-eval` scenarios (or add a
      `core` engine to `src/harness_eval`) and report both loops side by side.
- [ ] Baseline the plain kernel (no policies) so each policy has a number
      to beat.

### Phase 5: daemon policies as hooks (implemented, not yet measured)
All listed policies exist in `aidaemon-policies`. What remains is the
measurement gate: keep each one only if N-of-M runs show a gain.
- [ ] Measure each policy on/off with `aidaemon_eval::compare`.
- [ ] Just-in-time discovery of deeper `AGENTS.md` files (only start-up
      discovery is ported).
- [x] Port the daemon's memory (`aidaemon-memory`): facts with supersession,
      episodes, hybrid scoring, decay/retention, fastembed behind a feature,
      model-driven extraction; store/embedder/scorer are swappable traits.
- [ ] Memory gaps: graph signal (weight kept, always 0), near-duplicate
      merging, privacy/channel scoping, re-embedding after a model change.

### Phase 6: the rest of the framework (done, with gaps)
Agents-as-tools, MCP client, tracing sink, `aidaemon-sdk` bundle and example.
- [ ] Async approval inside sub-agents (a child that interrupts currently
      returns an error to its parent).
- [x] Images in messages and tool results (`Image`; Anthropic, OpenAI-compatible, MCP).
- [ ] Reasoning content in messages.
- [ ] Publish plan: versions, `publish = true`, crates.io names.

### Phase 6b: channels (done, with gaps)
`aidaemon-channels` (contract + runner) and Telegram, Slack and Discord
adapters, tested against local mock servers.
- [ ] Persist pending approvals (they live in memory; a restart loses them).
- [ ] Attachment download helpers (Slack `url_private` needs the bot token;
      Telegram needs `getFile`); attachments reach the agent as names only.
- [ ] Contract follow-ups from the adapters: `Attachment::kind`, a message
      ref on `ApprovalAnswer` so buttons can be removed, parent/thread link on
      `InboundMessage`, clearer `edit` semantics for over-long text.
- [ ] Not ported: LaTeX stripping, display-name lookup, slash commands,
      Discord sharding/compression.

### Phase 6c: open to contributors
- [x] Conformance kits for `Tool`, `Provider`, `Channel` and `MemoryStore`;
      every built-in implementation passes them in CI.
- [x] `CONTRIBUTING.md` and `crates/EXTENDING.md` (naming conventions
      `aidaemon-channel-*`, `aidaemon-provider-*`, ...).
- [x] Draft publishing plan in `crates/PUBLISHING.md`.
- [ ] Owner decision on publishing (names, 0.1.0, release automation).

### Phase 7: the daemon runs on the SDK (in progress, behind a flag)
- [x] Engine switch: build with `--features sdk-engine` and set
      `[agent.engine] kind = "sdk"`. Every turn that reaches
      `handle_message_impl` (channels, continuations, goal task leads,
      spawned children) runs on the kernel (`src/agent/runtime/sdk_engine.rs`).
      Daemon tools are bridged and dispatched through `execute_tool_outcome`;
      the turn emits TaskStart/UserMessage/AssistantResponse/TaskEnd so
      channel envelopes and history work. Anthropic provider only.
      Live check: `live_sdk_engine_runs_daemon_turns` (ignored; 5/5 on
      claude-haiku-5-5).
- [x] Live on the owner's daemon via Telegram (2026-10-31): prompt from the
      legacy builder (memory facts, skills, project instructions, mandate
      isolation), tool calls/results recorded and replayed as paired tool
      use, `/clear` respected, mandate authority gate per call, sub-agents
      and status updates working. Fixed on the way: top-level schema
      combinators rejected by Anthropic; `truncate_with_anchor` dropping the
      newest message (shared with the legacy loop).
- [ ] Not yet on the SDK path: attachments (paths only), `/model` and the
      daemon's provider router/fallbacks (Anthropic only), ToolCall events
      and receipts (ToolResult is recorded), intent gating and completion
      contracts (deliberately off). Scheduled goals not yet observed live.
- [ ] Daemon builds its agent through `AgentBuilder` with its hooks.
- [ ] Old loop runs side by side behind a flag until the new one matches or
      beats it on Phase 4 numbers.
- [ ] Delete the old loop.

## Parallel track: fixes found in the daemon

Independent of the SDK; each is a small PR against `master`.

- [ ] `src/tools/cli_agent.rs:662-711` launches external agents with their
      safety off (`--dangerously-skip-permissions`,
      `--dangerously-bypass-approvals-and-sandbox`, `--yolo`). Default to
      sandboxed / permission modes, make full bypass opt-in.
- [ ] No default sandbox on Linux (Docker backend is behind an empty,
      off-by-default `execution-docker` feature).
- [ ] `PlanRecovery::recover_interrupted_plans` (`src/plans/recovery.rs:39`)
      is only called from tests.
- [ ] `is_non_delegable_tool` (`src/mandates/mod.rs:11`) checks an `mcp__`
      prefix, but MCP tools are named `{server}__{tool}`. Add a test.
- [ ] State migrations have no version table.

## Known issues

- **macOS CI job never runs.** `Test (macos-14)` is cancelled within a second
  with no runner assigned (`runner_id: 0`) on every push since 2026-10-31.
  Master passed on macOS on 2026-10-23. Likely Actions billing or a spending
  limit for macOS minutes; check GitHub Settings → Billing and plans →
  Actions. Not caused by code.

## Working agreements

- Commits are authored as **David Loor M. <davo20019@gmail.com>**. No
  `Co-Authored-By: Claude` or session trailers in commits, and no
  "Generated with Claude Code" footer in PR descriptions.
- A new cloud container defaults git to Claude's identity. Either add
  `git config --global user.name/user.email` to the environment's setup
  script, or run them at the start of a session.
- Follow the repo's `CLAUDE.md`: `cargo fmt`, `cargo clippy --all-features
  -- -D warnings`, and the relevant tests before committing. For the SDK
  crates, run clippy and tests with the `-p` list from "How to resume".
- Synthetic data only in tests and docs (no real names, IDs or PII).

## How to resume

```bash
git fetch origin ccr-986102ee-12nt48
git checkout ccr-986102ee-12nt48
git config user.name "David Loor M."
git config user.email "davo20019@gmail.com"
cargo test -p aidaemon-core -p aidaemon-providers -p aidaemon-tools \
  -p aidaemon-store-sqlite -p aidaemon-mcp -p aidaemon-agents \
  -p aidaemon-policies -p aidaemon-eval -p aidaemon-sdk --all-features
cargo run -p aidaemon-core --example quickstart --features testing
ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example coding_agent -- <dir> "<task>"
```

Prompt to start a new session:

> Continue the aidaemon SDK work. Check out branch `ccr-986102ee-12nt48`,
> read `crates/aidaemon-core/PLAN.md` and `DESIGN.md`, then work on the
> step marked "Next up". Commit as David Loor M. with no Claude attribution.
