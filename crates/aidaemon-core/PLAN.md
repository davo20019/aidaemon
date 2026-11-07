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
| Branch | `ccr-695cf12a-023uky` (carries the earlier `ccr-986102ee-12nt48` work) |
| Crates | `aidaemon-core` plus 13 SDK crates (see `DESIGN.md`, "The SDK crates"), all `publish = false` |
| Tests | all passing across the 14 crates (incl. conformance kits); clippy and fmt clean |
| CI | Lint and tests for every SDK crate run in `ci.yml`. macOS job never gets a runner (see "Known issues") |
| PR | #14 (draft). #13 was closed without merging |
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

### Phase 4: measurement
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

### Phase 7: the daemon runs on the SDK (done except packaging)
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
- [x] Ported to the SDK path: media, `/model` and the daemon's provider
      router/fallbacks, ToolCall events and receipts, the assessor contract
      and authority lane (parity plan Tasks 1-7 and packages P1-P8).
- [x] Close the remaining gaps and replace the old loop: the SDK kernel is
      the only agent loop (`2c138cd`); the legacy loop and its dead code are
      deleted (`2ae738d`). The `[agent.engine] kind` setting is ignored.
      Owner decision (2026-10-31): no side-by-side comparison; once the test suite passes on
      the SDK engine, the old loop is deleted. Task-by-task breakdown with
      tests: `plans/2026-10-31-sdk-engine-parity.md` (this directory). All
      tasks there are ticked except Task 8 Step 4.
- [ ] `cargo package` for the daemon: blocked on the owner's decision on
      how `aidaemon-core` ships (publish it, or fold it into the main crate).
- [ ] Task wall-clock timeout: the legacy loop stopped a task after
      `subagents.task_timeout_secs` (default 30 min, scheduled goals
      exempt); the SDK engine leaves `sdk::Limits::timeout` unset, so
      nothing enforces it. Set it in `sdk_engine.rs` where the limits are
      built.
- [x] Global daily token budget: dropped. `[state] daily_token_budget` is
      accepted and ignored (logged once); a daily cap has to count every
      model call, so it belongs at the provider, not in the turn loop.
- [x] Scheduled goals: a scheduled run with a `budget_per_check` is bounded
      by that run budget alone; the task budget no longer stacks on top
      (`sdk_engine/budgets.rs`, `task_token_budget`).
- [ ] Whole-branch review, then live N-of-M runs on goals and mandates.

### Phase 8: the daemon is built on the SDK  ← Next up

Owner ruling (2026-11-04): the app is built on top of the SDK, so effort is
never duplicated. Mechanisms live in `aidaemon-core`. Reusable policy lives
in `aidaemon-policies`, extending what is there (`ApprovalPolicy`,
`Redactor`, `shell`) rather than adding parallel types. The daemon supplies
only facts specific to it (its tools' effects, the owner's rules, the
channel that asks the owner) and deletes the logic it duplicates.

Where duplication comes from: the daemon uses `aidaemon-core` and
`aidaemon-providers` (every model call, since 8.8's providers step), and
uses `aidaemon-policies` and `aidaemon-memory` only in part. It does not use
`aidaemon-tools`, the channel crates, `aidaemon-mcp`,
`aidaemon-store-sqlite` or `aidaemon-agents`, because it has larger copies
of each in `src/`.

Direction (owner): promote the daemon's implementations into the SDK
crates, replacing the thinner SDK versions, one domain per plan.

Order:
- [ ] 8.1 Authority Phase 3a: typed call effects, owner approvals as typed
      kernel interrupts, a decline binds the run across tools (kernel
      run-loop invariant), and `aidaemon_policies::EgressGuard` sharing
      `Redactor`'s patterns. Plan:
      `docs/superpowers/plans/2026-11-04-authority-phase-3a-effects-and-approvals.md`
      (local; `docs/` is gitignored).
- [x] 8.1b Provider failover in the kernel (done by a parallel session).
      - `c8dc38c`: a kernel `Failover` provider next to `Retry`. Each
        attempt gets its own deadline (time to the first stream event), a
        `FailoverPolicy` decides retry, next step or stop, and each switch
        is a `StreamEvent::Switched` reported as `Event::ProviderSwitched`.
      - `c13ab02`: a policy can rewrite the request for a retry
        (`RetryWith`) and skip an attempt without sending it
        (`before_attempt`).
      - `c65cbc6`: the daemon's model-call recovery runs as a `Failover`
        policy (`DaemonRecovery`). The chain is the requested model, the
        router's other models, then the failover providers' models (app
        policy). The first attempt's deadline sits under the call's idle
        watchdog, so a hung primary hands over to the fallbacks instead of
        the watchdog cancelling the call.
      - `bbd2603`, `9e4bfd0`: `ProviderError::QuotaExhausted` and a
        `QuotaGuard` provider wrapper (one per configured account) that
        remembers a used-up quota until its reset, an account that cannot
        pay, or a rejected key, and fails later calls without sending them.
        A chain moves past the account's other models at once; backends
        classify plan caps, daily quotas and out-of-credit errors.
- [x] 8.2 Tool contract. The kernel `Tool` gained typed `Extensions`
      and `ToolOutput::finish` (`RunStatus::Finished`, downgraded to
      `Partial` after a decline). `aidaemon_tools::ToolSemantics` holds
      what describes a call (including `dispatch_facts` and
      `owner_approval_request`), `ToolCallContext` carries the
      dispatcher's typed per-call facts (no `_` argument injection), the
      store traits live in `aidaemon_tools::services`, and tools receive
      their `ExecutionBackend` through constructors. The daemon `Tool` is
      `Tool: ToolSemantics` plus execution. Run-ending goes only through
      a tool's typed finish; MCP tools are bridged through the kernel,
      execute the instance judged at turn start, and their annotation
      hints never loosen a safeguard. The approval rules and the
      dispatcher no longer match tools by name or downcast.
- [ ] 8.3 The dispatcher bridge becomes SDK hooks. Today every call goes
      back over a channel to `src/agent/runtime/sdk_engine/tool_calls.rs`,
      which does about ten steps per call: idempotency and replay, ledger
      events, result fitting, instruction deferral, the mandate gate,
      `send_file` dedupe and more. Until these are hooks, the daemon is not
      really built on the SDK. Working-directory injection must become a
      kernel `Rewrite` hook that runs before call effects are computed
      (today a terminal call's relative paths are declared incomplete
      because the directory is injected after the kernel judged it).
      The daemon `Tool` trait and `BridgedTool` are deleted here. Parked
      from 8.2:
      - the global `active_execution_backend()` readers outside
        `src/tools` (runtime, slack, checkpoints, workspaces, and
        `tool_exec`'s guest-scoped file tools);
      - `cli_agent` and `send_file` detecting a background handoff from
        their output text;
      - `stop.rs` closeout lookups that see daemon tools only;
      - a background handoff that bypasses a typed `FailedPermanent`;
      - name checks left in `tool_exec.rs`: checkpoints for
        `write_file`/`edit_file`, `cli_agent`'s watchdog bypass, the
        terminal's trust roots, `spawn_agent`'s turn authority, the
        goal-context tool list (`spawn_agent`, `cli_agent`, `terminal`,
        `manage_mandates`, `manage_goal_tasks`) and `computer_use`'s live
        model; and the `manage_memories`/`manage_people` status summary
        in `tool_calls.rs`;
      - `semantics_for_exact_tool_call` resolves an MCP tool with a fresh
        `registry.find_tool`, which can disagree with the instance judged
        at turn start after a server reload;
      - the goals/mandates store slice and the concrete `EventStore`
        (8.8).
- [ ] 8.4 Tools converge into `aidaemon-tools`.
- [ ] 8.5 Authority Phases 3b and 3c, as SDK policies over effects:
      scope, sensitive paths, SSRF, destructive deletes, the remaining
      in-tool approvals, one output-wrapping layer.
- [ ] 8.6 Authority Phase 3d: children assembled on request through
      `AgentBuilder`. Includes child→parent decline propagation (a child's
      Deny does not reach its parent today) and kernel-level decline
      inheritance (the kernel passes a run's declines to its children
      directly, including children already running when the parent
      declines; today the daemon seeds them only at spawn).
- [ ] 8.7 Authority Phase 4 (end check, typed final outcome, budgets) and
      Phase 5 (memory policy; `aidaemon-memory` becomes the daemon's
      default memory instead of the opt-in `sdk-memory` feature; the
      assessor is retired).
- [ ] 8.7b Services: an SDK extension point for background work. It
      covers start and stop, scheduled ticks, and every agent run built
      through `AgentBuilder`, so background runs carry the same safeguards.
      The daemon's heartbeat, schedules and mandate reviews become
      services. It comes before 8.8 because channel runners are services
      too.
- [ ] 8.8 Channels, MCP and store, each promoted into its SDK crate.
      Providers are done: every daemon model call runs on an
      `aidaemon-providers` backend and the daemon `ModelProvider` trait is
      deleted (`e0e8899`..`29c3824`).

## Parallel track: fixes found in the daemon

Independent of the SDK; each is a small PR against `master`.

- [x] `src/tools/cli_agent.rs` launched external agents with their
      safety off (`--dangerously-skip-permissions`,
      `--dangerously-bypass-approvals-and-sandbox`, `--yolo`). Phase 3a
      Task 0 (`772a293`, `4b9b2ed`): every launch path strips the bypass
      flags (`strip_permission_bypass_flags`). The defaults still list
      them; an allowlist argument policy is 3c.
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
git fetch origin ccr-695cf12a-023uky
git checkout ccr-695cf12a-023uky
git config user.name "David Loor M."
git config user.email "davo20019@gmail.com"
cargo test -p aidaemon-core -p aidaemon-providers -p aidaemon-tools \
  -p aidaemon-store-sqlite -p aidaemon-mcp -p aidaemon-agents \
  -p aidaemon-policies -p aidaemon-eval -p aidaemon-sdk --all-features
cargo run -p aidaemon-core --example quickstart --features testing
ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example coding_agent -- <dir> "<task>"
```

Prompt to start a new session:

> Continue the aidaemon SDK work. Check out branch `ccr-695cf12a-023uky`,
> read `crates/aidaemon-core/PLAN.md` and `DESIGN.md`, then work on the
> step marked "Next up". Commit as David Loor M. with no Claude attribution.
