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
  `aidaemon-sdk` bundle with a `coding_agent` example and a non-coding
  `support_triage` example (custom tools, approvals on writes, no file or
  shell tools; not yet run against a live model).

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

### Phase 8: the daemon is built on the SDK  ← In progress (2026-11-07)

Status: 8.1, 8.1b, 8.2 done; 8.3 mostly done; 8.4 steps 1-3 of 10; 8.8
providers done and MCP transport moved, channels on the SDK adapters; 8.5-8.7,
8.7b and 8.9 open. Next: 8.5 (in-tool approvals) and 8.6 (declines between
parent and child runs), which close safety gaps. 8.4, 8.8 (MCP and store) and
8.9 move working code between crates and wait for a second user of the SDK.

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
- [ ] 8.3 The dispatcher bridge becomes SDK hooks. Done 2026-11-07:
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
      - after-call bookkeeping (result fitting, receipts, status line,
        learning, taint) still runs inside the bridged tool's dispatch; it
        needs daemon metadata the kernel does not carry;
      - the daemon `Tool` trait and `BridgedTool` go with 8.4 (tools on the
        kernel `Tool`);
      - about 45 readers of the global `active_execution_backend()`
        outside `src/tools` (one backend per process, so plumbing it is
        cleanup, not a behavior change);
      - the goals/mandates store slice and the concrete `EventStore` (8.8).
- [ ] 8.4 Tools converge into `aidaemon-tools`. Map (2026-11-07): no
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
      Steps 4-7 depend on 8.3's name checks staying valid; the tool names
      do not change.
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
        1. A task-trajectory contract (a session's unconsolidated tasks:
           request, tool calls with results, outcome, errors, decision
           points, completed plans), close to kernel receipts, and the port
           of `events/consolidation.rs` (procedures, error solutions,
           expertise, failure patterns, event episodes, the pruner) onto it.
        2. (was 3, below)
        3. Move the SQLite implementation of the stores
           (`state/sqlite/{facts,memory,structured_memory,episodes,people,
           learning}.rs`) into the crate, add the memory half of retention
           as a store method, and retire the crate's scope-keyed
           `MemoryStore`, `Fact` and `Episode` (`ScopedMemory` runs on the
           unified stores).
        4. Migrate the owner's data, run live N-of-M against the current
           daemon, then delete the old daemon modules.
        Not memory, noted on the way: `memory/context_window.rs` now holds
        only token budgeting, which belongs with the policies crate's
        `ContextWindow`; `memory/skill_promotion.rs` belongs to skills.
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
        Steps 1-2 were skipped: the daemon keeps its own tool wrapper and
        rendering, and only the transport moved (done: `McpClient` wraps
        `aidaemon_mcp::McpServer`; the dead retry path is deleted).
        Separate fixes, done: `restart` and `enable` ask the owner before
        respawning a stored command; `is_non_delegable_tool` no longer
        relies on an `mcp__` prefix no real name has.
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
- [ ] 8.9 Eval harness on `aidaemon-eval`. Map (2026-11-07): only the
      scenario format and the trajectory assertions are duplicated.
      `src/harness_eval/report.rs` formats live telemetry and is not a
      duplicate. The crate needs:
      - a `Subject` seam returning a neutral `Trajectory`;
      - typed `setup:` and `expect.ext:` blocks;
      - a proposed-call lane next to the dispatched one;
      - script arguments as a string or an object.
      The daemon then keeps a `DaemonSubject`, its snapshot assertions and
      the recorder. A differential test runs old and new checks over all
      fixtures before the old checks are deleted. Watch two differences:
      `response_contains` is case-sensitive in the daemon, and the daemon
      counts proposals where the crate counts receipts.

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
- [x] `is_non_delegable_tool` checked an `mcp__` prefix, but MCP tools are
      named `{server}__{tool}`. Fixed with a typed
      `ToolDispatchFacts::opaque_execution` that MCP tools declare;
      `mandates::is_non_delegable(tool)` reads it.
- [ ] State migrations have no version table.

## Known issues

- None open. (The macOS CI job ran again once it moved from the rejected
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
  -p aidaemon-policies -p aidaemon-eval -p aidaemon-sdk --all-features
cargo run -p aidaemon-core --example quickstart --features testing
ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example coding_agent -- <dir> "<task>"
ANTHROPIC_API_KEY=... cargo run -p aidaemon-sdk --example support_triage
```

Prompt to start a new session:

> Continue the aidaemon SDK work. Check out branch `ccr-695cf12a-023uky`,
> read `crates/aidaemon-core/PLAN.md` and `DESIGN.md`, then work on the
> first unchecked step in Phase 8. Commit as David Loor M. with no Claude
> attribution.
