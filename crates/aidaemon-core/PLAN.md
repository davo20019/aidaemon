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
| Crates | `aidaemon-core` plus 16 SDK crates under `crates/aidaemon-*` (see `DESIGN.md`, "The SDK crates"), all `publish = false` |
| Tests | passing across the crates CI covers (incl. conformance kits); clippy and fmt clean |
| CI | Lint and tests for 16 of the 17 crates run in `ci.yml`, on Linux and macOS. `aidaemon-decisions` (experimental) is not in either `-p` list |
| PR | #14 (draft). #13 was closed without merging |
| Research | Harness Field Guide: https://claude.ai/artifact/Fm5Ax7oX2vzmACjtb7iPxF |

### Next up

First: 8.7 items 1 and 3 together (honest final status and the typed final
outcome are one status mapping). Write the failing test first: the engine
already ends a run `Partial` when `RunOutcome::is_succeeded()` is false
(`sdk_engine/outcome.rs`) and the ledger reconciles at task end, so confirm
what is actually recorded before changing the stop logic.

## Definition of done

The goal is met when all four hold (from the thin-app design, "How we know
it's done", plus an outside-user check):

1. **An outsider can build an agent.** From the published crates and the
   getting-started guide alone, a new project assembles a non-coding agent
   (`support_triage` is the model) and it completes its task against a live
   model, N-of-M. The first working agent, with one custom tool written
   with `#[tool]`, fits in about 30 lines.
2. **Parity example.** An example that depends only on SDK crates assembles
   the preset in a short `main.rs` and passes the shared behavior suite
   (scripted provider, in-process channel): memory recall and writing, tool
   use, approvals, sub-agents, a channel round trip. Run again with another
   `MemorySystem`, everything but the daemon memory's own tests passes.
3. **Published.** `cargo package` passes for every crate and `0.1.0` is on
   crates.io.
4. **The daemon runs on the same crates** with no behavior change, checked
   live N-of-M.

The aim (owner, 2026-11-11) is not to beat other frameworks but to be the
comparable product in the Rust ecosystem: someone who has used the Claude
Agent SDK, the Vercel AI SDK or LangGraph should find the same things
here.

Milestone 2 meets all four. Milestone 3 is the owner's further goal (the
app is only wiring) and is not needed for outside users.

## Milestones

Every open item belongs to exactly one. Details stay under each phase.

### Milestone 1: merge PR #14 (the daemon is built on the SDK)

Gate: decision 7.

- [ ] 8.7 items 1+3: honest final status and typed final outcome (test
      first).
- [ ] 8.7 item 7: learning opt-out.
- [ ] Delete the task wall-clock timeout setting (decision 1).
- [ ] Add `aidaemon-decisions` to CI and `DESIGN.md` (decision 6).
- [ ] Commit the background-handoff work in the tree.
- [ ] Whole-branch review.
- [ ] Live N-of-M on goals and mandates.

After the merge the daemon releases as GitHub binaries and Homebrew only;
its crates.io publish is paused until Milestone 2, because it cannot be
published while it depends on unpublished path crates.

### Milestone 2: SDK 0.1.0 on crates.io (the framework goal)

Proof that the SDK works without the daemon:
- [ ] Run `coding_agent`, `support_triage` and
      `crates/aidaemon-eval/scenarios` against live models, N-of-M. Fix
      what breaks; this path has never run live.
- [ ] Plain-kernel baseline, then each policy on/off (Phases 4 and 5,
      decision 4). Remove a policy that shows no gain before it becomes
      public API.
- [ ] The parity example and its behavior suite (definition 2).
- [ ] Sandbox: run the bubblewrap backend's tests for real on Linux CI
      (only argument-level today), and make a sandbox the default for the
      shell tool on Linux.
- [ ] SQLite `ApprovalStore`, so a `ChannelRunner` user keeps pending
      approvals across a restart.

Contracts settle before they are published:
- [ ] Phase 9 step 0: contracts out of `src/types.rs` and the facade store
      traits, into crates.
- [ ] Phase 9 step 1: the daemon's memory system becomes
      `aidaemon-memory`'s full `MemorySystem`, the preset's default. This
      closes 8.7's `src/memory/` port ruling, item 2 (the crate's SQL
      reading the host's `events` table) and item 8 (memory policy from
      role and channel).
- [ ] 8.7 leftovers: tool names matched outside the dispatcher,
      `context_window.rs` to policies, `skill_promotion.rs` to skills.
- [ ] 8.7 item 4 (end check), only if decision 5's condition holds.
- [ ] Review `public-api.txt` for `aidaemon-core`: anything not meant to be
      a promise becomes private or `#[doc(hidden)]` now.

Developer experience, so it feels like the SDKs people already know
(Claude Agent SDK, Vercel AI SDK, LangGraph), in Rust:
- [ ] A short start: a builder preset in `aidaemon-sdk` (provider, default
      tools, store) so a working agent is about 15 lines. `quickstart.rs`
      is 121 today. This is the assembly half of 9.7, pulled forward; the
      daemon moving onto the preset stays in Milestone 3.
- [ ] Tools without hand-written JSON: a `#[tool]` attribute macro (new
      `aidaemon-macros` crate, re-exported by `aidaemon-sdk`) over an async
      fn or a typed args struct, with the schema derived through
      `schemars`. It emits the full function object the schema lint
      expects, and annotations/effects stay explicit arguments of the
      macro: a derived tool never gets a looser default than a
      hand-written one. Add it to the `Tool` conformance kit.
- [ ] Typed structured output: a call that returns `T: DeserializeOwned +
      JsonSchema` on top of the existing `ResponseFormat::JsonSchema`, with
      one repair retry on a parse failure, for a single model call and as
      an agent's final answer.
- [ ] Examples, each short and run in CI against the scripted provider:
      custom tool, approval and decline, resume after a crash, sub-agent,
      memory swap, channel bot, a run streamed over SSE from axum, and
      traces exported with the existing `otel` feature of
      `aidaemon-policies`.
- Not doing: graph or workflow authoring (LangGraph's core). Interrupt and
  resume, agents-as-tools and services cover the uses; revisit only if
  outside users ask.

Publish prep (`crates/PUBLISHING.md`, as amended by decision 3):
- [ ] `[workspace.package]` version `0.1.0`, `rust-version`, and per-crate
      `readme`, `keywords`, `categories`, `documentation`. Internal crates
      say so in their description.
- [ ] A `README.md` per crate (none exist), and a `crates/CHANGELOG.md`.
- [ ] Getting-started guide on docs.aidaemon.ai: install, quickstart,
      a custom tool, a hook, swapping memory, a channel. Present it as a
      general agent SDK for Rust; approvals, receipts and untrusted-content
      handling are features, not the pitch. One page mapping concepts from
      the Claude Agent SDK, Vercel AI SDK and LangGraph to ours.
- [ ] CI: `cargo doc --no-deps --all-features` per crate, MSRV job,
      `cargo-semver-checks` on the public-contract tier.
- [ ] `cargo package --list` reviewed for every crate.
- [ ] `release.yml` publishes the crates in dependency order, then the
      daemon with exact pins.
- [ ] Re-check the crate names are free, then the owner approves the
      publish. `publish = false` is removed only then.
- [ ] The outsider check (definition 1), from the published crates.

### Milestone 3: the app is only wiring (Phase 9 steps 2-7)

Not needed for outside users; see Phase 9. It ends with the app boundary
check in CI and the daemon running on the preset.

## Decisions (2026-11-11)

Delegated by the owner to the reviewing session; the owner can overrule any
of them.

1. **Task wall-clock timeout: delete the setting.** A run is bounded by
   progress and cost (inactivity watchdog, per-call timeouts, limits, token
   budgets), not by elapsed time; a wall clock kills legitimate long work
   and `cfd4fce` already removed the channel's for that reason.
   `subagents.task_timeout_secs` becomes accepted-and-ignored with one log
   line, like `daily_token_budget`; remove `Limits::task_timeout` and the
   dead `_effective_task_timeout` in `turn/main_loop.rs`.
2. **How `aidaemon-core` ships: publish the crates, do not fold them in.**
   Folding the kernel into the main crate would undo Phase 8 (the app is
   built on the SDK). `cargo publish` of the daemon needs every path
   dependency on crates.io, so all crates the daemon depends on are
   published, including the daemon-shaped ones.
3. **Publishing shape.**
   - One shared version for all SDK crates (`[workspace.package]`),
     starting at `0.1.0`, separate from the daemon's version. Independent
     versions (the draft in `PUBLISHING.md`) cost more than they return
     with one maintainer and one consumer; split later if outside users
     need it.
   - The daemon pins them exactly (`=0.1.x`), so a daemon release always
     builds against the crates it was tested with.
   - Two tiers. Public contract: `aidaemon-core`, `aidaemon-sdk`, the
     `channel` module, the memory traits; semver-checked in CI. Everything
     else, and `aidaemon-events`, `aidaemon-work` and `aidaemon-decisions`
     in particular, is published as internal: description says "internal
     to aidaemon, no stability promise".
   - Automation: `release.yml` publishes the SDK crates in dependency
     order, skipping versions already on crates.io, then the daemon. No
     separate tag scheme.
   - Timing: first publish is Milestone 2, after Phase 9 steps 0 and 1,
     so the contracts are settled and proven by the daemon before they
     become permanent. No placeholder 0.0.x versions to reserve names;
     until then the daemon ships as GitHub binaries and Homebrew only.
     These settle the thin-app design's two open decisions. Publishing is
     permanent, so the owner runs or approves that release.
4. **Decision 4 is amended**, since the old loop no longer exists to
   compare against: what is already ported stays; the plain-kernel baseline
   and per-policy on/off numbers are still collected, and a policy that
   shows no gain is removed. From now on no new policy, gate or stop-time
   demand lands without an N-of-M number. The Phase 4 old-vs-new comparison
   is dropped.
5. **8.7 items 4-6.**
   - Item 5 (checklist before the first change): dropped. A gate on the
     first mutating call is the kind that composes into dead ends, and the
     checklist is already demanded at stop time, where receipts can judge
     it.
   - Item 6 (three-routes stuck rule): dropped as a separate rule. Budgets,
     the streak guard and kernel stuck detection bound the same failure;
     reopen only if `db_probe --gate-outcomes` shows runs burning budget on
     repeated failed routes.
   - Item 4 (independent end check): deferred, not dropped. Build it after
     items 1+3, and only if the fabrication audit and N-of-M runs still
     show runs recorded as succeeded with unmet work. Then it is one
     `StopDemand` with a budget of one reopen, calling a declared
     `SideCall`.
6. **`aidaemon-decisions` joins CI's clippy and test `-p` lists** and the
   "SDK crates" section of `DESIGN.md`. Every workspace crate is linted
   and tested, experimental or not.
7. **Merging PR #14.** Gate: CI green, the whole-branch review done, and
   live N-of-M on goals and mandates. Merge with a merge commit, not a
   squash: this plan and the memory notes cite commit hashes on the
   branch.

## What exists

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
  `aidaemon-sdk` bundle with a `coding_agent` example and a non-coding
  `support_triage` example (custom tools, approvals on writes, no file or
  shell tools; not yet run against a live model).

## Decisions already made

1. **Same repo, new crates.** The SDK grows as workspace crates in this repo,
   so it can reuse `harness_eval`, providers and tools, and so the daemon
   becomes its first user. No separate repository.
2. **Small kernel, everything else is a hook.** If a feature cannot be built
   on the public hooks, add a hook point; do not add a kernel primitive.
3. **New loop, not an extraction.** The daemon's legacy loop
   (`src/agent/loop/`, ~62k lines, deleted in `2ae738d`) was not moved into
   the SDK. Daemon policies are re-implemented as hooks one at a time.
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
Five backends, each with streaming and tests against local mock servers:
Anthropic, OpenAI-compatible, Google Gemini, xAI, and the ChatGPT
subscription backend (`aidaemon-providers`). The daemon runs every model
call through them, so they have run live under the daemon. The SDK's own
examples have not yet been run live: see Phase 4.

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
- [ ] Live runs: run `crates/aidaemon-eval/scenarios`, the `coding_agent`
      example and the `support_triage` example against real models (needs
      API keys; `support_triage` is the non-coding check for "any kind of
      agent"). Live tests that exist: `src/integration_tests/live_sdk_turns.rs`
      (six `#[ignore]` tests against the ChatGPT backend).
- [ ] Old-vs-new comparison: not doable as written, so left unticked. The
      fixture translation happened (8.9), but "report both loops side by
      side" cannot: the legacy loop was deleted in 2ae738d. The 8.9
      differential compared the old and new *checks* on the one remaining
      loop, which says nothing about whether the SDK loop beats the old one.
      A real comparison would need the legacy loop built from before 2ae738d,
      or recorded numbers from it.
- [ ] Baseline the plain kernel (no policies) so each policy has a number
      to beat.

### Phase 5: daemon policies as hooks (implemented, not yet measured)
All listed policies exist in `aidaemon-policies`. What remains is the
measurement gate: keep each one only if N-of-M runs show a gain.
- [ ] Measure each policy on/off with `aidaemon_eval::compare`.
- [x] Just-in-time discovery of deeper `AGENTS.md` files:
      `NestedInstructions` reads a call's access manifest. A shell command
      declares no paths, so it finds nothing.
- [x] Port the daemon's memory (`aidaemon-memory`): facts with supersession,
      episodes, hybrid scoring, decay/retention, fastembed behind a feature,
      model-driven extraction; store/embedder/scorer are swappable traits.
- [x] Memory gaps in the scope-keyed store (`MemoryStore`, schema v2): graph
      signal from key-namespace and co-mention links (orders relevant facts,
      never admits one; facts only), near-duplicate merging through the
      existing key (`MergeConfig`), privacy/channel scoping
      (`Memory::recall_facts_for`, `RecallAudience`; a host with shared
      channels must set `audience_resolver`, because `MemoryTurn` carries no
      visibility), re-embedding in resumable batches
      (`Memory::reembed_batch`; nothing schedules it yet).

### Phase 6: the rest of the framework (done, with gaps)
Agents-as-tools, MCP client, tracing sink, `aidaemon-sdk` bundle and example.
- [x] Approval inside sub-agents: `AgentTool::interrupt_resolver` settles a
      child's interruption inside the parent's tool call, and the child keeps
      going. Without a resolver a child that interrupts is still an error.
- [x] Durable parent pause for a child's approval in the kernel and
      `AgentTool`: a tool that declares `Tool::suspends` answers with
      `ToolOutput::suspended`, the child's `RunState` is nested in
      `Pending::child` (`RUN_STATE_SCHEMA` 2; older snapshots still load), and
      `Agent::resume` passes the answer down at any depth.
      `AgentTool::suspend_on_interrupt` opts in; the live resolver is asked
      first. A crash after the answer is the crash of any running call: the
      delegation re-runs if idempotent and is indeterminate otherwise, so the
      child is not continued.
- [x] Owner ruling (2026-11-09): the daemon does not use this and keeps its
      policy. Its children are `spawn_agent` runs that ask through an
      in-memory card, and a restart declines pending approvals
      (`sdk_engine/recovery.rs`), because a stale card arriving hours later
      is riskier than asking again.
- [x] Images in messages and tool results (`Image`; Anthropic, OpenAI-compatible, MCP).
- [x] Reasoning content in messages: `Message::reasoning` keeps each model
      turn's visible reasoning in the transcript; the OpenAI-compatible
      provider now reads it too. It is not sent back to providers.
- [ ] Publish plan: versions, `publish = true`, crates.io names.

### Phase 6b: channels (done, with gaps)
`aidaemon-channels` (contract + runner) and Telegram, Slack and Discord
adapters, tested against local mock servers.
- [x] Persist pending approvals: `ApprovalStore` (with `MemoryApprovalStore`,
      the same shape as `SessionIndex`) and `ChannelRunnerBuilder::approval_store`.
      No SQLite implementation yet, and the daemon does not use `ChannelRunner`.
- [x] Attachment download helpers: `Channel::download_limited`, a shared
      bounded downloader (feature `http`) that keeps a token on the platform's
      own hosts, used by Telegram, Slack and Discord. `ChannelRunner` passes
      images as images and the owner's other files as a saved path
      (`AttachmentConfig`); a run that received them counts as having read
      untrusted content. The daemon still uses its own turn path.
- [x] Contract follow-ups from the adapters: typed `Attachment::kind`,
      `ApprovalAnswer::message` with `Channel::remove_buttons`,
      `InboundMessage::parent_message_id` and `thread_id`, and one `edit` rule
      for over-long text (one message, cut at the end, marked).
- [x] LaTeX stripping, display-name lookup, slash commands, Discord
      sharding/compression. `ChannelRunner` rewrites the model's LaTeX, code
      excepted, unless `Channel::strips_latex` says no; a direct `send` on
      Slack or Discord stays verbatim, so the daemon's own paths are
      unchanged. Names are cleaned and cached per adapter (`DisplayNameCache`)
      and never decide identity. Hosts add commands with
      `ChannelRunnerBuilder::command`; the runner sets each platform's menu.
      Discord takes `with_shard_count` or `with_recommended_shards`, one
      connection per shard with identify pacing and per-shard resume, and
      `zlib-stream` behind the optional `zlib` feature (`flate2`). The daemon
      still uses its own command handling and does not use `ChannelRunner`.

### Phase 6c: open to contributors
- [x] Conformance kits for `Tool`, `Provider`, `Channel` and `MemoryStore`;
      every built-in implementation passes them in CI.
- [x] `CONTRIBUTING.md` and `crates/EXTENDING.md` (naming conventions
      `aidaemon-channel-*`, `aidaemon-provider-*`, ...).
- [x] Draft publishing plan in `crates/PUBLISHING.md`.
- [ ] Owner decision on publishing (names, 0.1.0, release automation).

### Phase 7: the daemon runs on the SDK (done except packaging)
- [x] Engine switch (historical: the `sdk-engine` feature and the engine
      setting are gone, and the engine is no longer Anthropic-only; see
      "Close the remaining gaps" below). As shipped: build with
      `--features sdk-engine` and set
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
      Checked 2026-11-11: still unenforced; `turn/main_loop.rs` computes
      `_effective_task_timeout` and drops it. `cfd4fce` removed the
      channel's 20-minute clock on purpose (long work is bounded by
      inactivity, timeouts, limits and budgets), so the choice is to
      enforce `subagents.task_timeout_secs` or delete the setting.
- [x] Global daily token budget: dropped. `[state] daily_token_budget` is
      accepted and ignored (logged once); a daily cap has to count every
      model call, so it belongs at the provider, not in the turn loop.
- [x] Scheduled goals: a scheduled run with a `budget_per_check` is bounded
      by that run budget alone; the task budget no longer stacks on top
      (`sdk_engine/budgets.rs`, `task_token_budget`).
- [ ] Whole-branch review, then live N-of-M runs on goals and mandates.

### Phase 8: the daemon is built on the SDK  ← In progress (only 8.7 open)

Status (2026-11-11): 8.1, 8.1b, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7b, 8.8 and 8.9
done. 8.7: the memory port is done (compared live), the assessor is deleted
(`e857faf`) and budgets and stuck recording run through the kernel
(`2c443dc`, `b510f61`); Authority Phase 4's end check, typed final outcome,
checklist-before-change gate, stuck rule and learning opt-out are open.

Owner ruling (2026-11-04): the app is built on top of the SDK, so effort is
never duplicated. Mechanisms live in `aidaemon-core`. Reusable policy lives
in `aidaemon-policies`, extending what is there (`ApprovalPolicy`,
`Redactor`, `shell`) rather than adding parallel types. The daemon supplies
only facts specific to it (its tools' effects, the owner's rules, the
channel that asks the owner) and deletes the logic it duplicates.

Where duplication comes from (checked 2026-11-07 by import count): the daemon
imports `aidaemon-core` (62 files), `aidaemon-memory` (29),
`aidaemon-policies` (13), `aidaemon-tools` (8), the channel crates (8),
`aidaemon-store-sqlite` (6), `aidaemon-providers` (5) and `aidaemon-mcp`
(1). It does not import `aidaemon-agents`. Most of these are partial uses:
the daemon keeps larger copies of tools and policies in `src/`, and only
some of each crate is used. Providers are the exception: every model call
goes through them.

Direction (owner): promote the daemon's implementations into the SDK
crates, replacing the thinner SDK versions, one domain per plan.

Order:
- [x] 8.1 Authority Phase 3a: typed call effects, owner approvals as typed
      kernel interrupts, a decline binds the run across tools (kernel
      run-loop invariant), and `aidaemon_policies::EgressGuard` sharing
      `Redactor`'s patterns. Shipped 2026-11-04..06 (`f489c1c` kernel
      resolver, `1df0944` typed scope, `c3d9fa2`/`9c32fa4` terminal,
      `606066a`/`848455a` a declined run never succeeds, `319af63` children
      start with the parent's declines, `6cf68f3` one `[EGRESS BLOCKED]`,
      `46ed3e9` manage_mcp). Plan:
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
- [x] 8.3 The dispatcher bridge becomes SDK hooks. Done 2026-11-07:
      - `2134409`: the kernel's `ToolDecision::Answer` lets a hook answer in
        the tool's place (`ReceiptStatus::Answered`: not a success, never
        ends the run).
      - `a68d1e1`: `CanonicalCallHook`, first in the chain, rewrites each
        call into the form the dispatcher executes (project directory
        injected, model keys and unauthorized read roots dropped) before
        effects, scope and approvals are judged. The approval rules and the
        dispatcher read its per-call record instead of canonicalizing again.
      - `1885349`: `DispatchGate`, last in the chain (after approval, as
        before), runs the dispatcher's gates once per call: unjudged
        terminal call, live session mode, untrusted-reference block, scope
        check and mandate authority `Deny`; nested-instruction deferral,
        duplicate `send_file` and durable replay `Answer`. Kernel receipts
        now say what happened; the ledger records each refusal once.
      - `613718c`: call semantics come from the tool judged at turn start,
        not a fresh MCP registry lookup.
      - `3a22100`: the dispatcher's name checks are typed
        `ToolDispatchFacts` (`manages_own_timeout`, `records_launch_trust`,
        `goal_context`, `checkpoints_writes`, `needs_live_model`,
        `status_from_arguments`); the guest workspace file tools keep their
        deliberate name match.
      - `0b321da`: `cli_agent` and `send_file` type a background handoff
        where it happens instead of reading their own output text.
      - `cb89795`: a call typed `FailedPermanent` never hands the run off.
      - `c4df424`: the closeout counts the turn's MCP tools as reachable.
      Remaining:
      - Done 2026-11-09: after-call bookkeeping (result fitting, receipts,
        status line, learning, checklist) runs from the kernel's `after_tool`
        hook (`AfterCallRecord`, first in the chain), on the turn's task. The
        call step still classifies the outcome and marks taint: the kernel's
        receipt is built before `after_tool`, and a later call in the same
        batch must already see the turn as tainted;
      - Done 2026-11-09: the daemon `Tool` trait, `KernelTool` and
        `BridgedTool` are gone. Every daemon tool is an
        `aidaemon_tools::OutcomeTool` with one `execute(arguments, facts)`
        (`crate::traits::Tool` is that trait re-exported); progress travels
        in `facts.extensions`. `aidaemon_tools::KernelAdapter` makes any
        `OutcomeTool` a kernel `Tool`, and the daemon supplies only its
        `OutcomeRunner` (the turn's dispatcher);
      - Done 2026-11-09: production code no longer reads a process-wide
        execution backend. `core::run` builds it once and passes it on, and
        `Agent` holds it; text and path helpers take `&dyn ExecutionBackend`.
        The run ledger holds only `LexicalPaths` (flavor, workspace root,
        home), given to `EventStore` at construction.
        `active_execution_backend()` is `#[cfg(test)]`: the default local
        backend for tests that construct a tool directly;
      - Done 2026-11-09: the goals/mandates store slice and the concrete
        `EventStore` are in `aidaemon-work` and `aidaemon-events` (8.8).
- [x] 8.4 Tools converge into `aidaemon-tools`. Map (2026-11-07): no
      daemon tool can wrap today's thin crate tools without behavior
      changes. The crate tools implement only the kernel `Tool` and do I/O
      through `Workspace`, not `ExecutionBackend`. So the daemon versions
      move into the crate and replace the thin ones. Steps, one commit each:
      1. Add a generic adapter: crate `Tool + ToolSemantics` as daemon
         `Tool`.
      2. Move `file_output_trust` and path-key extraction into
         `semantics.rs`.
      3. Move the sensitive and secret path patterns in as a
         `PathClassifier` set by constructors, replacing the global in
         `fs_utils.rs`.
      4. Move `read_file` over `ExecutionBackend`, keeping its schema,
         metadata and tests.
      5. Move `edit_file`, with backups and diagnostics as a post-write
         hook.
      6. Move `write_file`, with append and replay, and fix the crate's
         `idempotent: true`.
      7. Move `search_files`.
      8. Move the SSRF classifier and capped body reader.
      9. Move `web_fetch`, with readability and the mandate no-redirect
         mode.
      10. Share one quote-aware `contains_shell_operator`.
      The terminal, `http_request` and `web_search` stay in the daemon.
      Done 2026-11-07: step 1 (`KernelTool`, `db34f0c`), step 2
      (`file_output_trust` in `semantics.rs`, `18ab5e8`) and step 3
      (`PathClassifier`, `7c623cf`; the daemon still holds one instance in
      a static, and passing it to each tool is the follow-up).
      Done 2026-11-08: steps 4-9 (`e612f19`, `11011ab`). The daemon's
      implementations run on `ExecutionBackend` in the crate, behind
      `OutcomeTool` and `FilePolicy`; `KernelTool` wraps them. `Glob` and
      `Grep` stay beside `SearchFiles` (they honour gitignore and offer
      content modes). The SDK `web_fetch` gained the private-network policy
      and kept its same-host redirect limit; the daemon follows public
      redirects. `track_requirements` stays in the daemon: it feeds the
      receipts ledger, and the crate's `Checklist` is an in-memory list.
      Done 2026-11-09: step 10 and the `PathClassifier` static. The
      terminal's quote-aware `contains_shell_operator` moved unchanged to
      `aidaemon_policies::shell`. `run_command` deliberately stays off it,
      on `fs_utils::mentions_shell_operator`: it runs what passes through a
      shell after only a prefix check, and the quote-aware scan skips
      `$(...)` and backticks in double quotes, so switching it would let
      `ls "$(id)"` run. One scan for both needs a scan that treats double
      quotes the way a shell does, which changes what both tools accept.
      The daemon's classifier is now a `DaemonPaths` value built once in
      `run()` and passed through `Agent` to the file tools, the scope
      checks, the approval rules and Slack.
      Left: nothing from the ten steps.
- [x] 8.5 Authority Phases 3b and 3c, as SDK policies over effects:
      scope, sensitive paths, SSRF, destructive deletes, the remaining
      in-tool approvals, one output-wrapping layer.
      Done 2026-11-08: destructive deletes. `irreversible_delete_reason`
      in `aidaemon-policies` replaces the terminal's in-tool
      `hard_block_reason` and is applied by `ApprovalPolicy` when given a
      `workspace_root`. It sees through launchers and subshells and also
      covers `rm -r`, `xargs rm` and `find -exec rm`, which all escaped
      before and ran unasked in scheduled sessions.
      Done 2026-11-08: one output-wrapping layer. The result boundary
      (`tool_result_fit`) sanitizes and wraps from the call's typed trust,
      after capping, for the model's view and the stored history alike. It
      replaces the tool-name list and `http_request`'s own wrapper; the
      kernel's `UntrustedHook` no longer re-marks daemon output. Text that
      never passes through a tool result (the `cli_agent` continuity message)
      still wraps where it is written.
      Done 2026-11-08: network targets and protected paths are hooks over
      what a call declares. `NetworkGuard` refuses cloud metadata endpoints
      for every tool and other private addresses by the tool's typed
      `NetworkReach`, before any card. `ProtectedPathGuard` resolves file
      tools' paths through their backend, follows symlinks, and replaces the
      checks inside `read_file`, `write_file` and `edit_file`. Shell effects
      now see through launchers for egress.
      Stay in the tools on purpose: each redirect hop (it exists only once
      the request runs), the first-request check in `web_fetch` and
      `http_request` (their effects are unknown on an unparsable URL), and
      the per-entry checks of `search_files`'s collaboration walk.
      Done 2026-11-08: the browser's `render_pdf` source goes through the
      guard (`FileGuard::SensitiveReads`), and stored CLI agents launch only
      with arguments listed for their program (`known_launch_arguments`);
      the defaults no longer list bypass flags. An agent in the owner's
      `config.toml`, or a program with no list, is held to the bypass flags
      alone. The in-tool approvals are done (8.1). Scope is judged by
      `DispatchGate` already.
- [x] 8.6 Authority Phase 3d: children assembled on request through
      `AgentBuilder`. Done 2026-11-07: declines move both ways between
      parent and child runs. `c98314c`: kernel `SharedDeclines`, attached
      with `RunState::share_declines` and exchanged with the run's own
      `declined` at turn start, before each call is judged, on a decline
      and at run end (a child's decline refuses the parent's overlapping
      calls with `[OWNER DECLINED]`; a parent's later decline reaches a
      running child; a parent whose child was declined never ends
      `Succeeded`). `fd163cd`: each daemon turn attaches one, a child turn
      takes its parent's through `InheritedAuthority::shared_declines`, a
      root turn or root goal dispatch starts its own; a run recovered after
      a restart shares none. Done 2026-11-08: `spawn_agent` takes `tools`
      and `access`, applied with `Authority::narrow`, so a parent can describe
      a child narrower than itself and never wider. Children already run
      on the kernel through `AgentBuilder`. Done 2026-11-09
      (`e1457cf`): a parent can pick a child's model tier, and the child's
      resolved authority is recorded as an event.
- [ ] 8.7 Authority Phase 4 (end check, typed final outcome, budgets) and
      Phase 5 (memory policy; `aidaemon-memory` becomes the daemon's
      default memory instead of the opt-in `sdk-memory` feature; the
      assessor is retired).
      Done 2026-11-08: the pre-turn assessor is deleted (`e857faf`); the
      model's `track_requirements` checklist is the only author of stop-time
      expectations. Budgets, forced text and stuck recording go through the
      kernel (`2c443dc`), and a request's sub-agent tree shares one token
      budget (`b510f61`).
      In the working tree, uncommitted (2026-11-11): a stop while a job the
      run launched is still running is a background handoff
      (`sdk_background_handoff` in `stop.rs`); the run ships the handoff
      reply and ends `Partial` (`outcome.rs`, `finish_tests.rs`).
      Open, checked against the code (2026-11-11). Order: 1 and 3 together,
      then 7; 4-6 only with evidence (see "Next up"); 2 and 8 with Phase 9:
      1. Honest final status. The daemon's stop logic
         (`sdk_engine/stop.rs`, `LedgerStop`) only returns `Accept` or
         `Continue`, never `AcceptUnmet` or `Halt`, so a run that stops with
         checklist items open is recorded as succeeded. A run closed by a
         policy refusal maps to `WorkSucceeded`
         (`aidaemon-events/src/run_aggregate.rs`, `ClosedByPolicyDenial`).
         Unverified: whether the task is actually recorded as succeeded.
         `outcome.rs` already maps a `Completed` run that is not
         `is_succeeded()` to `Partial`, and the ledger reconciles at task
         end; prove the gap with a test before fixing it.
      2. The memory crate's own SQL still reads the host's `events` table
         in three places (`sqlite/claims.rs`, `sqlite/mod.rs`,
         `sqlite/retention.rs`); the 11-09 ruling lists this as a gap.
      3. Typed final outcome: the working agent states
         `done | partial | blocked`, and the run's status follows it.
      4. Independent end check: an `on_stop` hook in which a cheap fresh
         agent compares the request with the checklist and receipts and
         may reopen the run once.
      5. Checklist before the first change: `checklist_declared` is tracked
         but only words a prompt header; nothing gates on it.
      6. Stuck rule: stop and report after about three different routes
         fail with nothing new learned (today: stuck recording and the
         same-tool streak guard only).
      7. "Don't remember this": a typed per-turn learning opt-out that
         extraction and consolidation honor.
      8. Memory policy from role and channel (authority design section 5):
         `compile_memory_pipeline_policy` still keys on the compiled
         contract, and `RecallOnlyNonUserTurn` remains.
      Smaller leftovers from earlier items:
      - Tool names still matched outside the dispatcher:
        `runtime/post_task.rs`, `runtime/turn_context.rs`,
        `turn/tool_execution/project_dir.rs`.
      - `memory/context_window.rs` belongs with the policies crate's
        `ContextWindow`; `memory/skill_promotion.rs` with skills.
      - The daemon's own memory system (`DaemonMemory`, recall, writer,
        its tools) is still in the app: the thin-app design's Phase 9
        step 1 (`docs/superpowers/specs/2026-11-11-sdk-thin-app-design.md`,
        local).
      - [x] Memory is replaceable as one unit: `aidaemon_policies::MemorySystem`
        (`MemoryTurn` in, `MemoryParts` out). `aidaemon_memory::ScopedMemory`
        implements it, and the daemon's `SystemMemory` plugs any
        implementation into turns, keeping the daemon's scope and permission
        rules.
      - [ ] Owner ruling (2026-11-07): port all of `src/memory/` into
        `aidaemon-memory`, including people intelligence, graph store,
        procedures, skill promotion, task learning, expertise, core profile
        and retention. Delete the old module only after parity tests and
        live N-of-M runs match the current daemon. Existing data must be
        migrated from the daemon database to the crate's store.
        Done 2026-11-07: embedding encoding and cosine (`c8bad7e`), scoring
        and fusion (`966582c`), the learning contract and expertise
        (`d0f011c`), procedure and error-solution helpers (`a7db714`).
        Decided 2026-11-07: the daemon-shaped contract wins and lives in
        `aidaemon-memory`. Done:
        - `ChannelVisibility`/`UserRole` are kernel vocabulary (`34c7ad7`);
        - `aidaemon_memory::{records, stores}` hold FactStore, EpisodeStore,
          HistorySearchStore, PeopleStore and the graph store (`eadc260`,
          `f62e870`); `aidaemon-tools` keeps only settings and OAuth;
        - ported onto them: neighborhoods, the core profile, the embedding
          index, people upkeep, retrieval metrics (`941bb96`), task
          learning (`04bf06d`), progressive extraction (`b959de3`);
        - phrase lists replaced by structure: `turn_importance`,
          `worth_extracting`, `procedure_base_name` (`f6b9d8d`).
        Done since: `ConversationSource` + `ConsolidationCursors`
        (`aidaemon_memory::history`, implemented by the daemon's
        `EventHistory`), fact consolidation (`d26bb8e`), episodes with typed
        error/decision signals (`2571c57`), profile verbosity, tool-sequence
        habits and decay as policy (`4835cd2`); the dead people-routing path
        is deleted. `memory/manager.rs` is now daemon orchestration only
        (heartbeat jobs, idle/long session selection, embedding backfill,
        goal review).
        Remaining, in order:
        1. Done 2026-11-09: a task-trajectory contract
           (`aidaemon_memory::trajectory`: `TraceEvent`, typed
           `DecisionPoint`, `CompletedPlan`) and `TaskLearner`, which takes
           a trace and writes procedures, error solutions, expertise,
           failure patterns and fact provenance through the stores.
           `Consolidator` only filters events (isolated sessions, suppressed
           tasks), converts them, looks up plans, calls the learner and marks
           the events; `retire_session` and `Pruner` stay there, because
           marking events is the host's. There are no event episodes to port:
           `MemoryManager` makes all of them. The `remember_fact` check moved
           into the learner, so a replacing system never runs it. All SQL on
           memory tables left `events/consolidation.rs` and
           `memory/manager.rs` for `aidaemon_memory::sqlite`, behind store
           methods with defaults; the re-export shims in `src/memory/` are
           gone. Parity: `2a590d0` holds both paths and a test that runs them
           over one synthetic history on separate in-memory databases and
           compares the rows; `078c4cc` deletes the old path and keeps the
           expected rows in `src/events/testdata/consolidation.golden`.
        2. Done 2026-11-09 (`159bee6`, `643a42d`): the SQLite stores are
           `aidaemon_memory::sqlite::SqliteMemory`, and the crate creates
           and upgrades the memory tables (`sqlite::migrate`). The state
           store forwards the memory traits to it. Retention's memory half
           is `sqlite::MemoryRetention` and stops under another system. The
           store runs without any host tables.
        3. Owner ruling (2026-11-09): people must be able to update and keep
           their data, and memory must stay replaceable by other open source
           systems. So:
           - No data migration for the default backend. The crate owns the
             same tables in the same database. Schema and row counts matched
             before and after on a new database, an August install and the
             live one.
           - The scope-keyed `MemoryStore` is NOT retired for now. It is the
             small contract an outside backend can implement.
           - Replaceable means more than `MemorySystem`, which only covers
             turn recall and extraction. Today these still go straight to
             the SQLite stores whichever backend is configured:
             the tools `manage_memories`, `share_memory`, `manage_people`,
             `search_history`, `cli_agent` and the memory tool;
             `agent/memory_plugin/recall.rs`; background work in
             `memory/{manager,retention,history_source}.rs` and
             `events/consolidation.rs`. `StateStore` also inherits every
             memory trait. The crate's own SQL reads the host's `events`
             table in three places.
           - Owner ruling (2026-11-09): another memory system may work in
             a completely different way. The seam says only what a system
             adds: per-turn hooks and tools, `guidance()` for the prompt, and
             background `services()`. The daemon assumes no facts, people,
             episodes or particular operations. Conversation history
             (`events`, `search_history`) stays the host's.
           - Done: `services()`, and old events still reach retention without
             learning (`Pruner::retire_old_events`); `guidance()`; the prompt's
             memory section, tool rows and fact rules apply only to the
             daemon's own memory (its prompt is byte-identical); `share_memory`
             is owned by it; post-run learning is its own. A test backend with
             one `journal_note` tool runs a whole turn and writes none of the
             SQLite memory tables (`memory_plugin/swap_tests.rs`).
           - Done 2026-11-09: `StateStore` no longer has the memory traits;
             memory's users hold `MemoryStores`, and `search_history` and
             `cli_agent` read the daemon's memory only while it is active.
             The turn treats memory tools by `MemoryToolFacts`, not by name;
             a replacing system's tools get facts from what they declare.
             Switching: `MemorySystem::absorb_history` learns from the
             retained conversations (`[memory] learn_from_history`, off by
             default, resumable, same privacy rules as a live turn), and
             `aidaemon memory-export DIR` writes the daemon's memory as
             Markdown.
        4. Done 2026-11-09: the old modules' memory logic is in the crate
           (`TaskLearner`, the stores). Consolidation parity was proved on one
           commit with both paths (`2a590d0`) and is guarded by a golden file.
           Live on Telegram, save/recall/forget with synthetic data: 3 of 3 on
           the build before the move, 3 of 3 after. The owner's facts and
           people were unchanged. The background consolidation pass ran
           on the new build and completed.
        Not memory, noted on the way: `memory/context_window.rs` now holds
        only token budgeting, which belongs with the policies crate's
        `ContextWindow`; `memory/skill_promotion.rs` belongs to skills.
- [x] 8.7b Services: an SDK extension point for background work.
      Done 2026-11-09: `aidaemon_core::service` (`Service`, `ServiceHost`,
      `Schedule`). The 20 periodic jobs are `Every` services, the goal cycle
      (schedules, mandate reviews, stuck tasks, notifications) is one
      `EveryTick` service because its phases depend on their order, and the
      channel bots are `Continuous` services. DB health is the host's gate,
      SIGTERM reaches the graceful path, and `heartbeat.rs` is split by
      phase. A long agent run is still interrupted by a restart and resumed
      by startup recovery.
- [x] 8.8 Channels, MCP and store, each promoted into its SDK crate.
      Providers are done: every daemon model call runs on an
      `aidaemon-providers` backend and the daemon `ModelProvider` trait is
      deleted (`e0e8899`..`29c3824`).
      - MCP map (2026-11-07). The daemon client (`src/mcp/client.rs`) is
        weaker than the crate's:
        - it reads the next stdout line as the reply, so a notification or
          a late reply puts later calls out of step;
        - it has no cancellation and no pagination;
        - it caps lines at 512 KiB.
        Its retry-after-restart path is dead code. Steps, one commit each:
        1. Crate: an annotation override (hints ignored, worst case) and
           schema normalization.
        2. Crate: `McpServerSpec`, `respawn`, and a stderr callback.
        3. Daemon: wrap `aidaemon_mcp::McpTool` behind the old interface,
           and port the hints-never-loosen test.
        4. `McpRegistry` holds crate servers.
        5. Delete the daemon client and the dead paths.
        Done (2026-11-09), all five. The crate gained
        `McpOptions::annotations` (the daemon sets the worst case, so server
        hints are never read), `normalize_schema`, `McpServerSpec`,
        `respawn`/`respawn_with_env`, `McpOptions::on_stderr`, `call_raw` and
        `detached`. The daemon's `McpTool` wraps `aidaemon_mcp::McpTool` and
        `McpRegistry` holds `McpServer`s; `src/mcp/client.rs` is deleted.
        What the daemon still owns on purpose: the `{server}__{tool}` name
        (unsanitized), the text a result renders to (`src/mcp/render.rs`,
        `isError` is a failed call), the suspicious-pattern logging, the owner
        approval on `restart`/`enable`, and `opaque_execution`.
        Differences: a tool whose server sends no object `inputSchema` (the
        MCP spec requires one) is now offered with an empty object schema, not
        `parameters: null`. `on_stderr` is not wired: the daemon has not logged
        server stderr since the transport moved (`c1ba172`).
        Separate fixes, done: `restart` and `enable` ask the owner before
        respawning a stored command; `is_non_delegable_tool` no longer
        relies on an `mcp__` prefix no real name has.
      - Store move, done (2026-11-09). The owner decided to move the goals and
        mandates slice and the concrete `EventStore` into crates; the case
        against it was that no second consumer exists yet. Layout, and neither
        crate depends on the root crate:
        - `aidaemon-events`: `EventStore`, payloads, the run aggregate,
          `conversation_turn`, `terminal_state`, `context`,
          `model_call_telemetry`, the `events` table DDL (`schema`), and the
          contract types they use (`Message`, the dialogue `Request*` types,
          `MandateAuthorityGrant`, `ProviderErrorKind`, `TokenUsage` and its
          store trait, session-id parsing, `execution_graph`).
        - `aidaemon-work`, above it: goals, mandates, tasks, work coordination
          and notifications: domain types, store traits, `sqlite::*`, the
          authority and admissibility rules the store enforces, and `schema`.
        - `LexicalPaths` moved to `aidaemon-tools`, next to `BackendPath`.
        `crate::traits`, `crate::events`, `crate::state`, `crate::mandates` and
        `crate::session` keep their names as re-exports.
        Seams, so nothing reaches back at the daemon:
        - `aidaemon_work::sqlite::WorkDatabase` (pool and `redact_secrets`).
          The store traits are implemented for anything that is one, so
          `SqliteStateStore` keeps implementing them with no forwarding.
          `SqliteWorkStore` is the ready-made one.
        - `aidaemon_events::EventHost`, given to `EventStore::on_paths`:
          exact-history projection, memory-span projection, error grouping.
          `DaemonEventHost` in `src/events/mod.rs` is the daemon's.
        - `is_one_shot_schedule` is a parameter of `migrate_work_tables`.
        Test-only behaviour that crossed a crate boundary is the `testing`
        feature (root dev-dependencies only): the `mandate_mutation_spy` name
        and `RunAggregate::new`. The daemon's tests get `EventStore::new` from
        `EventStoreForTests`, so they keep the daemon's seams.
        Migrations: `migrate_state` stays the orchestrator and still runs on
        every start, because a database gives no way to tell which steps it
        has seen. It calls `migrate_events`, then the daemon's own tables,
        `migrate_work_tables`, the daemon's token/summary/snapshot tables,
        `migrate_work_tail`, then history search: the same statements in the
        same order, and nothing changes table names, columns, rows or
        `state_schema_versions`. `src/state/sqlite/schema_golden/` pins a new
        database, a current one and an old goal database; the files were
        written from the code before the move.
        What stayed in the daemon, and why:
        - `src/events/consolidation.rs`, the host adapter into memory;
        - `state/sqlite/history_search.rs` and `project_event_span`'s call:
          the projection tables are the daemon's and memory's, reached through
          `EventHost`;
        - `extract_error_pattern`, `redact_secrets`, `cron_utils`: the daemon's
          own wrappers, patterns and grammar, passed in;
        - the facade traits in `src/traits/state_store.rs`, and every store
          outside this slice (messages, dialogue, skills, OAuth, settings);
        - three tests that exercise the daemon's seams
          (`src/events/run_aggregate_checklist_tests.rs`,
          `src/events/store_host_tests.rs`).
        `conversation_summaries` is unchanged: the SDK's `SummaryStore` compares
        `ConversationSummary::fingerprint`, which the table has no column for,
        and adding one is a schema change. Do it when an SDK user needs durable
        summaries, as a table of its own.
      - Channels map (2026-11-07). Update: the daemon's Telegram, Slack and
        Discord bots now run on the SDK channel adapters, and on the daemon's
        own turn runner (`src/channels/turns.rs`, `57788a2`, `fda3475`,
        `0fa59fe`). Discord has dropped serenity. The rest of this item
        records the plan as written before those commits.
        Use the crate adapters for transport only, not `ChannelRunner`: the
        daemon owns access, sessions, queueing, approvals and dynamic bots.
        Generic features the crates need first:
        - delete, file send and attachment download;
        - `Action` and `Command` events, with command registration;
        - Telegram webhook mode and `setMyCommands`;
        - name lookups, a Slack typing substitute, LaTeX stripping and a
          configurable max message length.
        Order: formatter golden tests, then a daemon bridge with
        session-id mapping, then Discord, then Slack, then Telegram.
        Done 2026-11-07: Telegram and Slack formatting (`df687af`, with
        `aidaemon_channels::latex`) and message splitting (`e6b154f`) come
        from the crates. The generic features are in the crates, each
        tested against a mock of its platform:
        - the contract (`edba5d0`): `delete`, `send_file`, `download`,
          `register_commands`, `answer_command` and
          `ChannelEvent::Command`; the runner answers a menu command and
          handles it as typed. App buttons beyond approvals (goal
          confirmations) go through `ask`, so there is no action event;
        - Discord (`df2c102`): uploads, CDN downloads, deletes, the global
          command menu (registered on READY), slash commands acked at once
          with a deferred reply that `answer_command` fills in;
        - Slack (`3d18e6a`): the external upload flow, `url_private`
          downloads, `chat.delete`, slash commands answered through their
          `response_url`; tokens and answers only go to Slack hosts;
        - Telegram (`739b9db`, `2ddcce6`): `sendDocument`, `getFile`
          downloads (20 MiB cap), `deleteMessage`, `setMyCommands`, and a
          webhook mode (feature `webhook`) with a secret token that falls
          back to long polling like the daemon's.
        Since (`1083c06`): name lookups, a configurable message length and
        Slack's typing substitute.
        The bots, each checked live on its adapter (2026-11-07), then:
        - one turn runner (`57788a2`, `channels::turns`): session
          registration, cancel words, queueing and dedup, task tracking,
          the inactivity watchdog and wall-clock limit, the working
          indicator, throttled status, delivery receipts and the queue
          drain are written once; a bot supplies a `TurnSurface` (how it
          says, shows work and status, delivers, and rechecks a queued
          message);
        - one prompt book (`channels::prompts`): approval and goal cards
          are sent with the adapter's `ask` and resolved from its answer,
          with the same buttons and outcomes on every platform;
        - Discord (`57788a2`): on both, serenity deleted;
        - Slack (`fda3475`): on both, the hand-written Socket Mode client
          deleted; the message path is tested against a mock Web API;
        - Telegram (`0fa59fe`): on both; the teloxide dispatcher, its
          poll-health watchdog and the HTML approval builders deleted;
        - the `[channels]` switch is gone (`3a9a7ff`) and the daemon's
          `Channel` trait is `HubChannel` (`95f552c`), the hub's outbound
          surface over an SDK transport.
        A new platform now needs its adapter crate, an access rule, a
        session-id scheme and a `TurnSurface`; turns, approvals and goal
        confirmations come for free. What stays per platform is genuinely
        platform behavior: Slack workspace grants, mention resolution and
        `file_shared` hydration (it still reads raw envelopes), Telegram's
        web app, terminal and setup commands.
        Telegram off teloxide (2026-11-08): the dependency is gone. The
        bot reads updates and messages as `channels::telegram_api` types
        (serde over the Bot API's own JSON, so key order no longer matters
        and `parse_update` is gone), and every call it makes goes through
        the adapter: replies, typing, the live status message, callback
        answers, reply keyboards and web-app buttons with `call_bot_api`;
        files with the adapter's `send_file` (pictures now go as
        `sendPhoto`) and `download` (which uses the configured API base
        instead of a hardcoded host); `/connect` and `/bots` token checks
        with `getMe`.
- [x] 8.9 Eval harness on `aidaemon-eval`. The crate gained a `Subject`
      seam returning a neutral `Trajectory`, typed `setup:` and `expect.ext:`
      blocks, a proposed-call lane next to the receipts, and script arguments
      as a string or an object. The daemon keeps `DaemonSubject` and its
      snapshot assertions (`src/harness_eval/subject.rs`, test-only) and the
      recorder (`recorder.rs`, used by `db_probe`, so it writes the scenario
      YAML from its own small structs; a test loads and replays that YAML).
      `report.rs` formats live telemetry and was never a duplicate. The 17
      fixtures are now `tests/harness_eval/scenarios/`. The old tool
      expectations map to `expect.proposed.*`, because the daemon counted
      proposals, and `response_contains` to `final_text_contains_exact`.
      Before the old checks were deleted (48bbcb9 has them side by side), a
      differential judged one run of each fixture with both, on the real
      expectations and on 880 mutated ones, and agreed on all. Deliberately
      wrong mappings made it fail. Proposals and receipts match in every
      fixture, so the lane split rests on a synthetic control run.

### Phase 9: the app is only wiring

Design: `docs/superpowers/specs/2026-11-11-sdk-thin-app-design.md` (local;
its rules for every move and its rulings apply). Steps 0 and 1 are in
Milestone 2, the rest in Milestone 3.

- [ ] 9.0 Contracts: the types in `src/types.rs` and the facade store
      traits move into crates, split by owner; each crate takes settings
      structs in place of `crate::config`.
- [ ] 9.1 The daemon's memory system (`DaemonMemory`, `DaemonRecall` and
      its guardrails, the run-end writer, its four tools, the orchestration
      in `src/memory/manager.rs`) becomes `aidaemon-memory`'s full
      `MemorySystem` over `SqliteMemory`, plugged in through `SystemMemory`.
- [ ] 9.2 The turn runtime (`src/agent/runtime/`, `src/agent/turn/`, intent
      and policy modules, the system prompt builder) into its crate; model
      routing next to the providers.
- [ ] 9.3 Work: background task lead, tasks, plans, mandates, triggers and
      reminders into `aidaemon-work`.
- [ ] 9.4 The remaining general tools into `aidaemon-tools`, or one crate
      per heavy tool behind features.
- [ ] 9.5 Channels: the daemon's turn runner, access, sessions, queueing,
      approvals, dynamic bots and slash commands into `aidaemon-channels`.
- [ ] 9.6 Platform services: execution backend, workspaces, checkpoints,
      OAuth and secrets, nodes, terminal bridge, skills, health.
- [ ] 9.7 Assembly and proof: the daemon runs on the `aidaemon-sdk` preset;
      a CI check fails on any `src/` module outside the product-shell list.

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
- [x] `PlanRecovery::recover_interrupted_plans` was only called from tests.
      Deleted: nothing reads `PlanStatus::Paused`, a resume is built from the
      event log (`agent/runtime/resume.rs`), and pausing at startup would
      have frozen a reused checklist, since `sync_active_step` promotes the
      next step only while the plan is `InProgress`.
- [x] `is_non_delegable_tool` checked an `mcp__` prefix, but MCP tools are
      named `{server}__{tool}`. Fixed with a typed
      `ToolDispatchFacts::opaque_execution` that MCP tools declare;
      `mandates::is_non_delegable(tool)` reads it.
- [x] State migrations have no version table. `state_schema_versions` now
      records the unversioned baseline as version 1 after every existing step
      has run; those steps are unchanged and still run on each start. New
      steps that cannot be idempotent take the next version.

## Known issues

- `aidaemon-decisions` is in the workspace and used by the daemon but is
  not in CI's SDK clippy/test lists and is not described in this plan.
- No other issue open. (The macOS CI job ran again once it moved from the rejected
  `macos-14` label to `macos-latest`, `43a5cd8`; it passed on `2d7d2e9`.)

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
  -p aidaemon-policies -p aidaemon-eval -p aidaemon-sdk -p aidaemon-memory \
  -p aidaemon-events -p aidaemon-work \
  -p aidaemon-channels -p aidaemon-channel-telegram -p aidaemon-channel-slack \
  -p aidaemon-channel-discord --all-features   # the list in ci.yml
cargo run -p aidaemon-core --example quickstart --features testing
ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example coding_agent -- <dir> "<task>"
ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example support_triage
```

Prompt to start a new session:

> Continue the aidaemon SDK work. Check out branch `ccr-695cf12a-023uky`,
> read `crates/aidaemon-core/PLAN.md` and `DESIGN.md`, then work on the
> item under "Next up". Commit as David Loor M. with no Claude
> attribution.
