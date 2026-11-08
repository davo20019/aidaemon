# aidaemon-core extension surface (baseline, read from source at commit 612973c)

Sources are repository files, cited as `path:line` instead of URLs. `core/` means `crates/aidaemon-core/src/`, `policies/` means `crates/aidaemon-policies/src/`, and `engine/` means `src/agent/runtime/sdk_engine/`. Nothing was built or run: "verified" means read in the code path, not executed. All 11 hook points are default no-ops on one trait, `Hook` (`core/hook.rs:159-268`).

## For each hook: when does it fire, what can it do, how do several compose, and what decision wins?

### Takeaway
There is one ordered hook list per agent, and every point walks it in registration order. Decision hooks short-circuit on the first decisive answer (`Deny`/`Interrupt`/`Answer`, `Reject`, `Retry`, `Extend`, `Continue`, or a `Some` stuck decision). A negative answer such as `Stop`, `Fail` or `Accept` is not a veto, so a later hook can still override it. Only `on_run_start`, `before_model`, `after_tool` and `on_run_end` always reach every hook, and they return nothing.

### Cited Findings

**Common contract**
- Hooks receive a read-only `HookContext`: run, agent and session ids, turn, messages, receipts, session and run usage, `untrusted_seen`, `declined`. It has no cancel token, no `Extensions`, and no per-hook state slot — `core/hook.rs:18-37`.
- No hook method returns `Result`, so a hook cannot report its own failure except through a decision value — `core/hook.rs:159-268`.
- Order is registration order for every point. `Agent::with_hook` appends last. There are no priorities, per-point ordering or removal — `core/agent.rs:98-105`, `core/agent.rs:289-300`.
- `Hook::safeguards` is a build-time declaration: an agent with tools fails `build()` unless all five `Safeguard::REQUIRED` are covered by a hook or `attest`, or `allow_unguarded` is set — `core/agent.rs:373-384`, `core/safeguard.rs:34-43`.
- Turn order: cancel check, limits, StopConditions, `turn += 1`, `TurnStarted`, `edit_transcript`, build request, `before_model`, model call, usage counted, `after_model`, then `on_stop` or the tool batch — `core/run_loop.rs:188-363`.

**on_run_start**
- Fires once, only on the fresh-run branch and only when `state.turn == 0`, after `RunStarted` — `core/run_loop.rs:175-185`.
- `with_user_message` resets `turn` to 0, so each continued session message is a new run and fires it — `core/state.rs:207-223`.
- Observational: no return value and no `&mut`, so it cannot veto the run or inject messages. All hooks run — `core/hook.rs:168-171`, `core/run_loop.rs:182-184`.

**edit_transcript**
- Fires at the start of every turn, after `turn += 1` and before the request is built — `core/run_loop.rs:215-219`.
- Returning `Some(messages)` replaces the stored transcript durably; `None` abstains — `core/hook.rs:173-186`.
- Composition: all hooks run in sequence and each sees the previous edit, because the context is rebuilt per hook — `core/run_loop.rs:1206-1232`.
- An identical replacement is ignored. A malformed one (no user message, or a broken call/result pairing) is refused with `TranscriptEditRefused`, and later hooks still run — `core/run_loop.rs:1214-1221`, `core/run_loop.rs:1348-1373`.
- A successful edit triggers a checkpoint — `core/run_loop.rs:1233-1235`. Receipts and trust state are untouched — `core/hook.rs:181-183`, test `crates/aidaemon-core/tests/hook_point_tests.rs:615-649`.

**before_model**
- Fires every turn, including retries, because a retry goes back to the top of the loop — `core/run_loop.rs:219-225`, `crates/aidaemon-core/tests/hook_point_tests.rs:327-329`.
- Receives `&mut ModelRequest` (model, messages, tools, tool_choice, max_output_tokens, options) and affects that request only — `core/provider.rs:59-69`, `core/hook.rs:188-190`.
- All hooks run, each seeing earlier edits. Returns `()`. No `transcript_defect` validation is applied to these edits — `core/run_loop.rs:219-225`.

**after_model**
- Fires after a successful response, and after usage has been added, so a rejection is still paid for — `core/run_loop.rs:257-263`.
- Gets `&mut ModelResponse` and returns `Accept` or `Reject { message }`. Added calls face the same validation and `before_tool` — `core/hook.rs:192-203`, `crates/aidaemon-core/tests/hook_point_tests.rs:188-208`.
- The first `Reject` wins and stops the walk; mutations by earlier hooks persist — `core/run_loop.rs:1169-1177`.
- On reject, the text is kept without its calls, the message is appended with origin `Harness`, the state is checkpointed and another turn runs — `core/run_loop.rs:282-300`.
- Text deltas have already been emitted. `response.reasoning` and `response.note` are not written to the transcript: only `text` and calls are — `core/hook.rs:195-196`, `core/run_loop.rs:301-303`.

**on_model_error**
- Fires when the stream yields an error or ends with no `Done`, after whatever retries the Provider did. Timeouts and cancels do not reach it — `core/run_loop.rs:239-256`, `core/run_loop.rs:1022-1027`, `core/hook.rs:205-213`.
- Returns `Fail` or `Retry { after }`. The first `Retry` wins; a `Fail` from an earlier hook does not stop the walk — `core/run_loop.rs:1181-1189`.
- A retry emits `ModelRetried`, waits (stops on cancel, clamped to the wall clock), and consumes a turn. The transcript is untouched — `core/run_loop.rs:247-254`, `core/run_loop.rs:1193-1200`.

**on_limit**
- Fires at three sites: top of turn, in a `while` that asks once per limit kind per pass; model call cut off by wall clock; before a call inside a batch, after queued calls flush — `core/run_loop.rs:193-201`, `core/run_loop.rs:229-236`, `core/run_loop.rs:430-446`.
- Returns `Stop` or `Extend { amount }`. The first `Extend` with `amount > 0` wins and is added to `RunState::limit_grants`. A zero `Extend` or a `Stop` just moves to the next hook — `core/run_loop.rs:1143-1165`.
- `HARD_TURN_CAP` (1000) is never put to a hook — `core/run_loop.rs:1144-1146`, `core/limits.rs:7-9`, test `crates/aidaemon-core/tests/hook_point_tests.rs:554-570`.
- A limit left at `None` is unaffected by grants — `core/limits.rs:101-113`.

**before_tool**
- Fires per call, after the deterministic checks: tool lookup, `validate`, and the owner-decline check. A malformed call never reaches a hook — `core/run_loop.rs:454-497`, `core/run_loop.rs:688-692`.
- Decisions: `Allow`, `Deny`, `Rewrite`, `Interrupt`, `Answer` — `core/hook.rs:48-80`.
- The first `Deny`, `Interrupt` or `Answer` returns immediately — `core/run_loop.rs:696-703`.
- A changed `Rewrite` re-validates, recomputes annotations, access and effects, re-checks declines, and restarts at hook 0. More than 16 changes is a deny. An identical rewrite is a no-op — `core/run_loop.rs:704-733`.
- Hooks are judged serially, including for calls that later run concurrently. Queued reads are flushed first before a non-concurrent or egress call is judged, so taint is visible — `core/run_loop.rs:509-520`, `core/run_loop.rs:539-549`.
- `Answer`: the tool does not run, the receipt is `Answered` at stage `RejectedBeforeDispatch`, and its `finish` is dropped — `core/run_loop.rs:585-610`.
- `Interrupt`: the run ends `Interrupted` with `Pending`, including the rewritten call and the interrupting `hook_index`. `Approve` resumes judgment at `hook_index + 1`, and a later rewrite restarts from 0 — `core/run_loop.rs:612-636`, `core/run_loop.rs:529-535`.

**after_tool**
- Fires for every receipt, in call order, after truncation — `core/run_loop.rs:911-922`. Both `record` and `skip_rest` go through `append`, so that includes denied, unknown, invalid, skipped, cancelled, timed-out, answered and indeterminate results — `core/run_loop.rs:880-907`.
- Concurrent groups finish first (`join_all`), then `after_tool` runs serially per result — `core/run_loop.rs:764-768`.
- All hooks run. Each can mutate `&mut ToolOutput`; the receipt is already final — `core/hook.rs:236-244`.
- Fields read after the hooks: `content`, `images`, `finish`, `usage`. So a hook can set or clear `finish` on a dispatched call, or change the tool's reported usage. The receipt's `status`, `details` and `source` were fixed earlier — `core/run_loop.rs:923-949`. This is not documented on the hook; it follows from the read order.

**on_stop**
- Fires only when the post-`after_model` response has no tool calls — `core/run_loop.rs:305-306`. It does not fire for a tool `finish` (`core/tool.rs:195-197`), limits, cancels or stuck exits.
- Decisions: `Accept`, `Continue { message }`, `AcceptUnmet { unmet }` — `core/hook.rs:90-100`.
- The first `Continue` wins and breaks. Unmet lines are merged and deduplicated across hooks, but discarded if a later hook continues — `core/run_loop.rs:310-349`.
- `Continue` appends a `StopHook` message, checkpoints and loops; only run limits bound it. `final_text` is read-only here — `core/run_loop.rs:333-344`, `core/hook.rs:248`.

**on_stuck**
- Fires after `record` when the detector signals (repeat, repeated error, ping-pong). Skipped calls do not feed the detector — `core/run_loop.rs:878-907`, `core/run_loop.rs:960-971`.
- `Nudge`, `Stop` or `Ignore`. The first `Some` wins. If all abstain, the default nudges until `max_nudges`, then stops — `core/run_loop.rs:973-992`. A hook's `Nudge` also increments the count the default consults — `core/run_loop.rs:996-999`.
- The detector resets after any decision — `core/run_loop.rs:994`. `Stop` skips the rest of the batch — `core/run_loop.rs:424-428`.
- Any status other than `Succeeded` counts as failed, including `Denied` and `Answered` — `core/stuck.rs:86`.

**on_run_end**
- Fires after the final checkpoint and `Finished` event; observational. All hooks run — `core/run_loop.rs:1298-1323`, `core/run_loop.rs:1327-1343`.
- Under `run_resolving`, segments suppress it and it fires once at the end, even if the resolver leaves the run interrupted — `core/agent.rs:160-175`.
- Under plain `run` then `resume`, each segment's `Driver` keeps `report_run_end: true`, so it fires with `Interrupted` and again at the real end — `core/run_loop.rs:86-96`, `core/agent.rs:118-126`, `core/agent.rs:189-202`. This contradicts "Fires exactly once per run" at `core/hook.rs:252`.

### Inferences
- Because `Stop` and `Fail` are not vetoes, a policy that must prevent an extension or retry cannot express that as a hook; any permissive hook anywhere in the list wins.
- Because `before_tool` hooks are not re-asked on `Approve` before `hook_index`, registration order is security-relevant. The docs pin this order for that reason — `policies/lib.rs:27-31`.
- `after_tool` changes `output.content` before stuck fingerprinting (`core/run_loop.rs:905-906`), so a hook adding varying text could mask `RepeatedCall`.
- `before_model` can remove a tool from `request.tools`, but dispatch resolves against the agent's registered tools (`core/run_loop.rs:456`), so hiding a spec is not a capability boundary.

### Gaps
- `crates/aidaemon-core/tests/hook_point_tests.rs` covers `on_run_start`, `after_model`, `on_model_error`, `on_limit` and `edit_transcript`. I did not find a test asserting that `on_run_end` fires twice for plain `run` + `resume`; that is read from the code only.

## What happens when a hook panics or hangs? Are hooks under a timeout or cancel? Are tool panics treated differently?

### Takeaway
Tools are isolated and hooks are not. A tool panic becomes a failed receipt, and a tool is bounded by a timeout and cancel. A hook panic unwinds out of `Agent::run`, and a hung hook blocks the run indefinitely, ignoring both the cancel token and the wall-clock budget.

### Cited Findings
- The only `catch_unwind` in kernel runtime code wraps `tool.call` — `core/run_loop.rs:794-803` (the other occurrence, `core/conformance/report.rs:140`, is test tooling). Repo-wide search of `core/`, `crates/aidaemon-agents/src` and `policies/`.
- All hook calls are bare `.await`s with no `catch_unwind`, `timeout` or `select!` — `core/run_loop.rs:183`, `:223`, `:315`, `:697`, `:920`, `:977`, `:1151`, `:1172`, `:1184`, `:1209`, `:1341`.
- The wall-clock bound wraps only `call_model`; hooks sit outside it — `core/run_loop.rs:229`. Cancel is polled between steps and raced against the model stream and retry sleep, never against a hook — `core/run_loop.rs:190`, `:412`, `:1009-1013`, `:1195-1199`.
- The kernel spawns no task of its own (no `tokio::spawn` in `core/` or `crates/aidaemon-agents/src`), so the driver runs on the caller's task.
- Tool treatment: panic gives `Failed` with `the tool panicked: ...` and `Trusted` source — `core/run_loop.rs:831-839`, test `crates/aidaemon-core/tests/loop_tests.rs:1756`.
- Tool timeout is `min(remaining wall clock, tool.timeout() or Limits::tool_timeout)` and yields `TimedOut` — `core/run_loop.rs:786-792`, `:846-857`. Cancel uses a child token and yields `Cancelled` — `core/run_loop.rs:784`, `:795-797`, `:840-845`.
- Not inside the tool guard, and called on the driver: `tool.validate`, `annotations_for`, `annotations`, `access`, `effects` — `core/run_loop.rs:480`, `:498-506`.
- Other uncaught extension code: `StopCondition::check` (`core/run_loop.rs:203-207`), `EventSink::emit` (`core/run_loop.rs:1254-1257`), `Provider::stream` (`core/run_loop.rs:1007`).
- `RunStore` errors are not contained: `?` returns `CoreError::Store` and no `RunStatus` is produced, so `on_run_end` does not fire — `core/run_loop.rs:1254-1270`, `core/run_loop.rs:1307`, `core/error.rs` (`CoreError::Store`).
- The daemon polls the run future directly in a `select!` on the turn's task — `src/agent/runtime/sdk_engine.rs:1453-1476`.
- The daemon's `BridgedTool::call` only sends over a channel and awaits a oneshot; the real tool runs in `dispatch_sdk_tool_call`, pushed into a `FuturesUnordered` in that same `select!` — `engine/hooks.rs:278-295`, `src/agent/runtime/sdk_engine.rs:1478-1480`. In `src/`, `catch_unwind` appears only at `src/heartbeat.rs:761` and `src/providers/mod.rs:33`.

### Inferences
- A hook panic leaves the last checkpoint as the durable state and fires no `on_run_end` — unwinding leaves `drive` before `finish`.
- In the daemon, the kernel's tool panic isolation covers only the channel send, not the daemon tool's body. A panic in a real daemon tool would unwind the turn's task. The kernel's tool timeout drops the oneshot wait, while the dispatcher future appears to keep running.
- `tokio::time::timeout` is cooperative, so a tool blocking its thread without yielding is not interrupted.

### Gaps
- No test exercises a panicking or hanging hook. The only hook panic in the core tests is an assertion that a hook is never reached (`crates/aidaemon-core/tests/loop_tests.rs:177-194`).
- I did not trace what outer layer, if any, catches a panic or stalls a hung turn for interactive channel turns. `src/heartbeat.rs:761` is the one `catch_unwind` on a future.

## What other extension points exist, and what can only be changed by editing the kernel?

### Takeaway
Besides `Hook`, the public seams are `Provider` (with composable wrappers), `Tool`, `RunStore`, `EventSink`, `StopCondition`, `InterruptResolver`, `AuthorityCeiling`, `FailoverPolicy`, plus config values and tool-only `Extensions`. The turn sequence, scheduling, detection algorithm, recovery rules, status set and event set are fixed.

### Cited Findings
- **Provider**: `name`, `complete`, optional `stream`. The loop always calls `stream`, and the default wraps `complete` as a single `Done` — `core/provider.rs:176-198`. It receives only `ModelRequest`, with no run context.
- **Provider wrappers**: `Retry` retries retryable errors, and on streams only before any output — `core/provider.rs:231-316`. `Failover` runs a chain with per-attempt deadlines, committing once a step starts streaming — `core/failover.rs:184-202`.
- `FailoverPolicy` has `before_attempt` (can refuse an attempt) and `after_error` (`Retry`, `RetryWith` a rewritten request, `Next`, `Stop`) — `core/failover.rs:100-142`. `QuotaGuard` is a further wrapper — `core/quota.rs:130`.
- **Tool**: `spec` and `call` are required. Optional: `annotations`, `annotations_for`, `source_trust`, `validate`, `timeout`, `access`, `effects` — `core/tool.rs:347-415`.
- Conservative defaults: `DESTRUCTIVE` annotations and `Untrusted` trust — `core/tool.rs:73-77`, `core/tool.rs:375-377`.
- `ToolOutput` can carry `is_error`, `details`, `usage`, `trust` and `finish` — `core/tool.rs:229-254`.
- **RunStore**: `append_event`, `save_state`, `load_state` — `core/store.rs:20-24`. There is no list or query method on the trait; recovery queries are specific to `SqliteStore` — `crates/aidaemon-store-sqlite/src/lib.rs:240-280`.
- **EventSink**: synchronous, read-only `emit(run_id, &Event)` — `core/event.rs:95-99`. `Event` is `#[non_exhaustive]` with a fixed variant set, so there are no custom event types — `core/event.rs:12-93`.
- **StopCondition**: synchronous, sees only `RunSnapshot` (turn, tool_calls, usage, elapsed), checked at the top of each turn; the first `Some` ends the run as `Stopped` — `core/limits.rs:74-97`, `core/run_loop.rs:202-210`. No hook consulted.
- **InterruptResolver**: settles interruptions in `run_resolving`; `None` leaves the run interrupted — `core/agent.rs:21-30`.
- **AuthorityCeiling**: async live narrowing read before each call by `AuthorityHook` — `core/authority.rs:105-111`, `core/authority.rs:123-128`, `core/authority.rs:136-146`.
- **Extensions**: typed values set once at build, visible only on `ToolContext`, not `HookContext` — `core/tool.rs:116-137`, `core/tool.rs:160-161`, `core/agent.rs:347-352`.
- **Config values**: `Limits`, `StuckConfig`, `attest`, `allow_unguarded`, `max_output_tokens`, `instructions` — `core/agent.rs:263-352`.
- **Fixed by design statement**: "the order of a turn, tool concurrency, validation and the decline check before any hook, and output truncation before `after_tool`" — `crates/aidaemon-core/DESIGN.md:153-154`. Project rule: a feature that cannot be a hook gets a new hook point, not a kernel primitive — `crates/aidaemon-core/DESIGN.md:24-25`.
- **Fixed in code, no seam**:
  - Concurrency rule: consecutive `read_only` calls via `join_all`, using worst-case annotations — `core/run_loop.rs:497-502`, `:764`.
  - `HARD_TURN_CAP` — `core/limits.rs:9`; 16-rewrite cap — `core/run_loop.rs:706`.
  - Stuck detection algorithm and its 32-entry window; only thresholds are configurable — `core/stuck.rs:7-17`, `:74`, `:103-145`.
  - The truncation shape — `core/run_loop.rs:1445-1458`.
  - The `RunStatus` set — `core/outcome.rs:11-44`.
  - Checkpoint cadence and recovery rules — see the durability section.
  - The tool set is fixed at build — `core/agent.rs:361-372`.
  - `TextDelta` goes straight to sink and store, with no filter point — `core/run_loop.rs:1015-1017`.

### Inferences
- There is no kernel seam for injecting a user message mid-run except editing the request or transcript through a hook. The daemon does its steering inside its provider path — `engine/model_call.rs:585`.

### Gaps
- `crates/EXTENDING.md:101-103` lists only five hook points, and `crates/aidaemon-core/DESIGN.md:29-31` gives `run_loop.rs` as about 800 lines where it is 1497. Those docs are stale against the code. Trust the code.

## Durability: what is checkpointed and when, what happens on crash recovery and resume, and which hooks re-fire?

### Takeaway
`RunState` is a plain serializable snapshot, saved before and after each tool group, on transcript edits, on rejected responses or stops, and at every exit. Before the first such save nothing is durable. Recovery is explicit: the caller must find and re-drive a run. It finishes the batch without repeating non-idempotent calls, but hook-held state is not persisted.

### Cited Findings
- Checkpoint sites: reject — `core/run_loop.rs:298`; stop-continue — `:342`; batch recorded as in flight before execution — `:354-358`; before and after each flush — `:755`, `:769`; after a batch — `:165`, `:172`, `:360`; transcript edit — `:1233-1235`; every `finish` — `:1307`.
- Persisted fields: messages, receipts, usage, turn counters, elapsed, `pending`, `in_flight`, nudges, taint, stuck window, declines, unmet, finishing, limit grants — `core/state.rs:99-161`. Not serialized: `shared_declines` — `core/state.rs:141-146`. There is no field for hook state.
- Every event is also journaled per emit, including each `TextDelta` — `core/run_loop.rs:1254-1262`.
- `drive` picks one of three entries: pending, in flight, or fresh — `core/run_loop.rs:149-186`.
- Crash recovery: not-started calls run normally; started idempotent calls re-run; started non-idempotent calls get `Indeterminate` and are not repeated — `core/run_loop.rs:1032-1077`, tests `crates/aidaemon-core/tests/kernel_ext_tests.rs:225`, `:245`. Idempotence is read per call; a tool no longer registered counts as non-idempotent — `core/run_loop.rs:1045-1050`.
- Resume checks reject a newer schema or a different `agent_id` only; the hook list and tool list are not compared — `core/agent.rs:204-218`. `hook_index` is a raw position — `core/state.rs:65-68`.
- `Agent::run_from` refuses a pending state, `resume` refuses a non-pending one — `core/agent.rs:136-138`, `:196-198`.
- `DESIGN.md:168-172` says a non-idempotent tool in the in-flight turn "may run twice", citing a section that does not exist. `DESIGN.md:184-187` and the code say otherwise.

**Re-fire behaviour (traced from `drive`)**

| Situation | Entry | Event | Hooks |
|---|---|---|---|
| Death before first checkpoint (first model call) | No saved row exists; caller must start over | `RunStarted` | Everything, including `on_run_start` |
| Saved between turns (turn > 0) via `run_from` | Fresh branch | `RunStarted` again | No `on_run_start`; normal turn hooks |
| Saved mid-batch | `recover` | `Recovered` | `after_tool` and `on_stuck` for indeterminate calls (the stuck verdict is discarded, `:1068-1069`); `before_tool`/`after_tool` for re-run and unstarted calls; none for calls already done |
| Interrupted, `Approve` | Pending | `RunResumed` | `before_tool` from `hook_index + 1`; all hooks for later calls; no `on_run_start` |
| Interrupted, `Deny` | Pending | `RunResumed` | `after_tool` for the denial; no `before_tool` for that call |
| Any terminal end | n/a | `Finished` | `on_run_end`, per segment unless `run_resolving` |

**Daemon recovery**
- At the start of a turn, the session's most recent run is settled if it has `pending` or `in_flight` — `engine/recovery.rs:20-33`, `src/agent/runtime/sdk_engine.rs:1452-1465`. A crash between turns (neither set) is not re-driven.
- The recovered transcript is not reused for the next turn, and the shared decline handle is not reattached — `engine/recovery.rs:10-12`, `:38-42`. Test `engine/tests.rs:4401-4439` shows settling runs a model call before the new turn.

### Inferences
- `on_run_start` is documented for loading what other hooks need, but it does not re-fire after a crash at turn > 0, so a hook relying on it must also tolerate running without that load, or reload lazily.
- **Possible defect, not tested:** `in_flight` is cleared only on normal batch completion or an interrupt (`core/run_loop.rs:625`, `:641`). Exits for cancel, limit or stuck mid-batch return before that, so the final checkpoint keeps `in_flight` set. The daemon's `stranded_run` (`engine/recovery.rs:26`) would treat such a finished run as stranded, and `drive` would recover and go on to another model call.
- **Possible defect, not tested:** the daemon settles a stranded run with the new turn's hook chain. `memory_hooks` varies in length (`src/agent/runtime/sdk_engine.rs:1060-1081`) and sits before `Egress`, `Untrusted`, `Approval` and `Dispatch` (`engine/hooks.rs:118-130`). A saved `hook_index` could then shift, and on `Approve` with one fewer memory hook, the hook after the interrupter would be skipped (for example `DispatchGate`).
- `on_run_end` fires on the current turn's hook instances for a settled run, so `DaemonMemoryWriter`, which holds the new turn's context (`src/agent/memory_plugin/writer.rs:57`), would run with the stranded outcome.

### Gaps
- Crash timing was traced statically. I ran no crash test.

## Subagents: how do children inherit declines, limits and budgets? Is there a subagent lifecycle hook?

### Takeaway
The kernel has no subagent concept and no lifecycle hook. Children are tools. It offers parts: `ToolOutput::usage` roll-up, `Agent::with_limits`, `Authority::narrow`, `SharedDeclines`. The SDK's `AgentTool` uses only usage roll-up and cancel; the daemon does not use `AgentTool` and wires inheritance itself.

### Cited Findings
- No `Hook` method concerns children — `core/hook.rs:159-268`. A parent sees a child only as `before_tool`/`after_tool` on the delegating tool. A child runs only the hooks of its own `Agent`.
- `ToolOutput::usage` is added to the parent's session and run usage — `core/tool.rs:243-246`, `core/run_loop.rs:948-949`. Tool-call counts do not roll up: the parent counts one — `core/run_loop.rs:947`.
- `SharedDeclines`: a run family exchanges declines before each call, at turn start, on a decline and at end. It only grows, and a parent is refused what was declined in its child — `core/approval.rs:231-246`, `core/run_loop.rs:214`, `:689`, `:130`, `:1306`. Attach with `share_declines` — `core/state.rs:225-234`. It is cleared by `with_user_message` — `core/state.rs:217-218`.
- `Authority::narrow` can only shrink — `core/authority.rs:58-70`.
- **`AgentTool` (`crates/aidaemon-agents`)**:
  - Runs a fresh `RunState` via `run_with_cancel` in session `{parent}/sub/{call_id}`, with no shared declines and no authority narrowing — `crates/aidaemon-agents/src/lib.rs:394-400`, `:66-69`.
  - Cancel is inherited — `:398`. Depth is derived from `/sub/` segments and capped at 3 by default — `:64`, `:73-75`, `:377-389`.
  - Budget is per-child `ChildBudget` (default `max_turns` 20), replacing limits — `:121-155`, `:214-217`. The parent's tool timeout bounds the whole child run — `:159-160`.
  - An interrupted child is an error to the parent — `:450-455`, `crates/aidaemon-core/DESIGN.md:295-296`.
- The daemon does not depend on `aidaemon-agents` (no `aidaemon_agents` reference in `src/`).
- Daemon children are full daemon agents given `InheritedAuthority`: authority, owner level, task access, mode session, taint, origin request and trust, trusted roots, `declined`, and a live `shared_declines` handle — `src/agent/runtime/authority.rs:167-198`.
- At run start the daemon unions parent declines, then shares the set — `src/agent/runtime/sdk_engine.rs:1268-1282`. A child inherits `run_store` — `src/agent/runtime/spawn.rs:1530`. Iteration limits and token budget come from spawn arguments — `src/agent/runtime/spawn.rs:1464-1465`, `:1498`.
- `core/run_loop.rs:102-137`: only denied `OwnerApproval` interruptions add effects to `declined`, so a denied `Untrusted` interrupt or a hook `Deny` does not propagate.

### Inferences
- A child under `AgentTool` that hits the parent's tool timeout has its future dropped, so its `on_run_end` would not fire and it would not report usage.

### Gaps
- I did not confirm whether the daemon rolls a child's token usage into the parent's kernel usage. No `with_usage` or usage roll-up appears in `engine/tool_calls.rs`.

## Is there any way to skip a model call and supply a response? Is there a hook for tool failure as distinct from after_tool?

### Takeaway
No to both as first-class hooks. Tools can be short-circuited by `ToolDecision::Answer`, but `before_model` returns nothing. Skipping a model call requires a `Provider` wrapper. Tool failure is seen only through `after_tool`'s receipt status, which cannot retry.

### Cited Findings
- `before_model` returns `()` and the call follows unconditionally — `core/hook.rs:188-190`, `core/run_loop.rs:219-229`.
- `after_model` can replace the whole response, but only after the call has been made and its usage counted — `core/run_loop.rs:257-263`.
- `on_model_error` offers `Fail` or `Retry` only; it cannot supply a response — `core/hook.rs:133-142`.
- A `Provider` can return any `ModelResponse` without contacting a backend, and that is the kernel's only skip path; it has no run context — `core/provider.rs:184-198`.
- The tool-side equivalent exists: `ToolDecision::Answer` — `core/hook.rs:72-79`.
- `StopCondition` ends a run before the model call, but as `Stopped` with no response — `core/run_loop.rs:202-210`. A tool's `Finish` ends the run after its batch with no further model call — `core/tool.rs:195-197`.
- No method for tool failure exists on `Hook` — `core/hook.rs:159-268`. `after_tool` fires for all statuses and carries `receipt.status` and `receipt.stage` — `core/receipt.rs:18-39`, `core/run_loop.rs:919-921`.
- `after_tool` cannot retry or change the receipt — `core/hook.rs:236-237`.
- `on_stuck` with `RepeatedError` is the only failure-triggered escalation — `core/stuck.rs:115-126`.
- `Event::ToolFinished { receipt }` exposes the same data to sinks — `core/event.rs:49-51`.

### Inferences
- `Answer` output with `Untrusted` trust does not set `untrusted_seen`, because taint requires stage `Dispatched` (`core/run_loop.rs:942-945`). A hook replaying untrusted content cannot taint the run, since `HookContext` is read-only.

### Gaps
- None for this question.

## Which hook points does the daemon currently use and which are unused?

### Takeaway
The daemon overrides five of the eleven points. `on_run_start`, `edit_transcript`, `after_model`, `on_model_error`, `on_limit` and `on_stuck` have no non-test implementation anywhere in the repository; HEAD's commit added most of them. The daemon gets that behaviour from its `Provider` bridge instead.

### Cited Findings
The chain is built per turn: `src/agent/runtime/sdk_engine.rs:952`, `:1161-1192`, with order pinned in `engine/hooks.rs:118-130`.

| Hook point | Installed in the daemon |
|---|---|
| `on_run_start` | None |
| `edit_transcript` | None |
| `before_model` | `SummaryNote` when a summary exists (`src/agent/runtime/sdk_engine.rs:1076-1080`, `policies/conversation/hooks.rs:105`); `MemoryInjection` from the daemon memory plugin (`src/agent/memory_plugin.rs:375-376`) |
| `after_model` | None |
| `on_model_error` | None |
| `on_limit` | None |
| `before_tool` | `CanonicalCallHook` (rewrite), `AuthorityHook` with live ceiling, `SameToolStreakGuard`, `EgressGuard`, `UntrustedHook`, `ApprovalPolicy`, `DispatchGate` (deny or answer) — `engine/hooks.rs:118-130`, `engine/tool_calls.rs:211`, `engine/tool_calls.rs:285-288` |
| `after_tool` | `UntrustedHook` only — `core/untrusted.rs:83-88` |
| `on_stop` | `LedgerStop` — `engine/hooks.rs:14-41` |
| `on_stuck` | None; `Event::Stuck` is forwarded through the sink — `src/agent/runtime/sdk_engine.rs:1217-1219` |
| `on_run_end` | `DaemonMemoryWriter` (`src/agent/memory_plugin/writer.rs:57`), `SummaryNote` (`policies/conversation/hooks.rs:114`); with the SDK memory backend, `aidaemon_memory::MemoryWriter` (`crates/aidaemon-memory/src/writer.rs:98`) |

- Non-test overrides for the six unused points: none, by search for `async fn <name>` across `crates/` and `src/` outside test files. Non-test overrides elsewhere exist only for `before_tool`, `after_tool`, `before_model`, `on_stop`, `on_run_end`.
- Other seams the daemon does and does not use: it installs `store` and `event_sink`, `attest(Scope)`, and `run_resolving` — `src/agent/runtime/sdk_engine.rs:953`, `:1200-1204`, `:1466`.
- The daemon's `Limits` sets only `max_turns` and `tool_timeout`, so no kernel token, tool-call or wall-clock limit is active — `src/agent/runtime/sdk_engine.rs:958-962`.
- No use found of `stop_when`, `stuck_detection`, `tool_extensions`, `with_limits` or `with_hook` in engine code.
- `DaemonProvider` implements only `complete` and forwards to the daemon, which owns retries and fallback — `src/agent/runtime/sdk_provider.rs:1-7`, `:25-52`. That means no kernel streaming: the default `stream` yields one `Done`.
- The provider path does work that hook points now exist for:
  - Forcing `tool_choice` — `engine/model_call.rs:273`.
  - Steering messages — `engine/model_call.rs:585`.
  - Budgets, ended by returning a provider error — `engine/model_call.rs:190`, `:561`, `:650-659`.
- Each daemon turn builds a new kernel `Agent` and a new `RunState` seeded from its own projection — `src/agent/runtime/sdk_engine.rs:952`, `:1240-1244`.
- `DESIGN.md`'s mapping table (`crates/aidaemon-core/DESIGN.md:305-318`) lists budget auto-extension under `on_limit` and pivots under `on_stuck`; the daemon implements neither that way.

### Inferences
- A comparison against other frameworks should separate the kernel's offered surface (11 points) from the surface proven in production (5 points). The six newest points have only test coverage, with no production consumer.

### Gaps
- The memory backend is config-dependent, so the `on_run_end` and `before_model` rows vary with `[memory]` configuration and are absent for mandate runs — `src/agent/memory_plugin.rs:177-180`.
