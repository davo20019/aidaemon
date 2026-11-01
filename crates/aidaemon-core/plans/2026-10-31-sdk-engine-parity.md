# SDK Engine Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every daemon turn can run on the SDK kernel (`[agent.engine] kind = "sdk"`) with the same providers, records, budgets, cancellation and completion checks as the legacy loop, so the legacy loop can later be retired.

**Architecture:** The SDK engine (`src/agent/runtime/sdk_engine.rs`) builds one `aidaemon_core::Agent` per turn. Everything daemon-owned (tools, model calls, completion checks) is reached through one port: SDK-side adapters (`BridgedTool`, `DaemonProvider`, `LedgerStop` hook) send a `DaemonCall` over an mpsc channel, and the turn's `tokio::select!` loop serves it with `&Agent`. Daemon policies become SDK hooks or port handlers; no new kernel primitives.

**Tech Stack:** Rust, tokio, `aidaemon-core` (kernel), daemon `ModelProvider`/`Router`, `EventStore`/`RunAggregate`.

**Spec:** `crates/aidaemon-core/PLAN.md`, "Phase 7", plus the gap list below (from reading the legacy loop on 2026-10-31).

## Gap list this plan closes

| Gap | Task |
|---|---|
| Engine is a process-wide `OnceLock`; no hermetic tests | 1 |
| Anthropic only; no router, fallbacks, `/model`, failover providers | 1, 2 |
| No per-call `llm_call`/`token_usage` telemetry, no goal token budgets | 2 |
| No context-budget fitting (fixed 60 messages) | 2 |
| Tool exchanges persisted only after the run; no `ToolCall` events; receipts without `Dispatched` stage; `track_requirements` never declares expectations | 3 |
| Parent/goal cancellation ignored (`CancellationToken::new()`) | 4 |
| No ledger closeout; TaskEnd outcome ignores receipts | 5 |
| No assessor contract / authority lane | 6 |
| No same-tool-name stall limit | 7 |
| Two engines to maintain | 8 |

Not a gap: non-image attachments. Channels already prepend `build_inbound_text` stubs (`src/channels/attachments.rs:88`), same as the legacy loop.

## Global Constraints

- Tasks 1-7 keep the legacy loop as the default so `master` stays releasable mid-way; Task 8 removes it.
- Follow `CLAUDE.md`: `cargo fmt`, `cargo clippy --all-features -- -D warnings`, tests with `RUST_MIN_STACK=8388608`.
- Evidence beats contract: receipts close runs; the contract is a hint plus authority lane. No new obligation classes or stricter gates (`CLAUDE.md`, "Task-Kernel Freeze").
- Reuse daemon functions; do not copy legacy logic. Visibility bumps (`pub(super)` → `pub(in crate::agent)`) are fine.
- Synthetic data only in tests (`telegram:synthetic-user-1`, `Alice Rivera`).
- Commits authored by David Loor M., no Claude trailers.
- Every new behavior is tested hermetically with `MockProvider` through `Agent::handle_message` on the SDK engine (enabled by Task 1).

## Review Focus

1. A provider error mid-run (after tools ran): the run must end `ProviderFailed` with tool results already persisted, not lose them. Test in Task 3.
2. `/model` set by the owner must win over router selection on the SDK engine. Test in Task 2.
3. A cancelled goal must stop a running SDK child before its next model call and emit TaskEnd `cancelled`. Test in Task 4.
4. The closeout must not loop forever: at most 2 `Continue` demands per run, and none after a terminal policy denial. Test in Task 5.
5. History with an orphaned tool call (crash between call and result) must not produce an unpaired tool-use request after context fitting. Test in Task 2.

---

### Task 1: Engine settings on the Agent and the daemon provider bridge

**Files:**
- Modify: `src/agent/runtime/sdk_engine.rs` (replace `ENGINE` `OnceLock`, `install`, `active`; add `DaemonCall`, `DaemonProvider`)
- Modify: `src/agent/mod.rs` (field `sdk_engine: Option<Arc<SdkEngineSettings>>` on `Agent`), `src/agent/construct.rs` (default `None`; `with_depth` copies parent's), `src/core.rs:164-170`, `src/agent/loop/main_loop.rs:531`, `src/config.rs:803-815`
- Modify: `src/testing.rs` (helper `setup_test_agent_sdk(provider: MockProvider) -> anyhow::Result<TestHarness>`)

**Interfaces:**
- Produces: `pub(crate) struct SdkEngineSettings { model_override: Option<String>, max_turns: u32 }`; `pub(crate) fn settings_from_config(config: &AppConfig) -> Option<SdkEngineSettings>`; `enum DaemonCall { Tool(ToolRequest), Model(ModelCall) }`; `struct ModelCall { request: sdk::ModelRequest, reply: oneshot::Sender<Result<sdk::ModelResponse, sdk::ProviderError>> }`; `struct DaemonProvider { calls: mpsc::Sender<DaemonCall> }` implementing `sdk::Provider` (name `"daemon"`, `complete` only); `Agent::serve_sdk_model_call(&self, scope: &TurnScope<'_>, call: ModelCall)`.
- Produces (pure, unit-tested): `fn to_daemon_messages(request: &sdk::ModelRequest) -> Vec<Value>` (OpenAI style: system, user/assistant with `tool_calls`, `{"role":"tool","tool_call_id","name","content"}`; images as the daemon's multimodal content parts), `fn to_daemon_tools(specs: &[sdk::ToolSpec]) -> Vec<Value>` (`{"type":"function","function":{name,description,parameters}}`), `fn from_provider_response(resp: ProviderResponse) -> sdk::ModelResponse` (arguments string parsed to JSON, invalid → `{}`; usage mapped incl. cached/creation), `fn to_chat_options(choice: &sdk::ToolChoice, max_output_tokens: Option<u32>) -> ChatOptions`.

- [ ] **Step 1: Write failing tests**
  - `daemon_messages_round_trip_tool_pairs`: request with system + user + assistant(tool_calls `call_1`) + tool result → 4 values, assistant has `tool_calls[0].function.arguments == "{\"path\":\"a.txt\"}"`, tool message `tool_call_id == "call_1"`.
  - `provider_response_maps_usage_and_bad_arguments`: `ProviderResponse` with `arguments: "not json"` and usage `{input 100, output 7, cached Some(40)}` → `tool_calls[0].arguments == json!({})`, `usage.cached_input_tokens == 40`.
  - Integration `sdk_engine_runs_on_mock_provider` (in `src/integration_tests/`, `#[cfg(feature = "sdk-engine")]`): `setup_test_agent_sdk` with `MockProvider` scripted to one tool call to `remember_fact` then text `"Saved."`; `handle_message` returns `"Saved."`, `provider.call_count() == 2`, one `ToolResult` event for the task.
- [ ] **Step 2: Run** `RUST_MIN_STACK=8388608 cargo test --features sdk-engine --lib sdk_engine` — expect compile failure / FAIL.
- [ ] **Step 3: Implement.** Engine config: `model` becomes an optional override of daemon model selection; `api_key` is deprecated (log a `warn!` once if non-empty, ignore). Rename the select-loop channel to `mpsc::Sender<DaemonCall>`; `BridgedTool` sends `DaemonCall::Tool`. `serve_sdk_model_call` calls `self.call_llm_with_recovery(provider, router, model, &messages, &tools, &options, &mut telemetry, None)` with `provider/router` from `self.llm_runtime.snapshot()`; map `Err` to `sdk::ProviderError::Other(error.to_string())` (the daemon already retried/cascaded, so the kernel must not retry again: no `sdk::Retry` wrapper). Model for this task: `engine.model_override` else `self.model.read()`; Task 2 replaces it. Delete the `aidaemon-providers` dependency from the root `Cargo.toml` `sdk-engine` feature and the Anthropic builder.
- [ ] **Step 4: Run** the Step 2 command plus `RUST_MIN_STACK=8388608 cargo test --features sdk-engine integration_tests` — PASS. Update the ignored live test to use `setup_test_agent_sdk` with a real daemon provider or delete it if it only tested the removed Anthropic path.
- [ ] **Step 5: Commit** `feat(sdk-engine): route model calls through the daemon provider`.

### Task 2: Model selection, telemetry, goal budgets, context fitting

**Files:**
- Modify: `src/agent/runtime/sdk_engine.rs` (`serve_sdk_model_call`, new `sdk_turn_model`)
- Possibly modify visibility in `src/agent/loop/bootstrap/run.rs:1146-1215` if a selection helper is extracted there instead of copied.

**Interfaces:**
- Consumes: Task 1 `serve_sdk_model_call`, `TurnScope`.
- Produces: `async fn sdk_turn_model(&self, engine: &SdkEngineSettings) -> String`, stored on `TurnScope.model: String` (owned); `TurnScope.tokens_used: AtomicU64`.

- [ ] **Step 1: Write failing tests** (integration, MockProvider records the model string per call):
  - `sdk_engine_respects_model_override`: harness with router models; set `agent.model_override = true`, `agent.model = "override-model"` → every recorded call uses `"override-model"`.
  - `sdk_engine_uses_router_primary_by_default`: no override → recorded model equals `router.select(Tier::Primary)` (root agent) — matches legacy `select_for_profile(Balanced)`.
  - `sdk_engine_records_llm_call_telemetry`: after a 2-call run, `token_usage` rows for the session == 2 and `llm_call` events for the task == 2.
  - `sdk_engine_fits_history_to_budget`: `context_window.default_budget` small (e.g. 4000) with 60 long prior messages → the last recorded request's estimated tokens ≤ budget, first message role is `system`, last is the current user text, and every `tool` message is directly preceded by its assistant `tool_calls` (Review Focus 5: seed one orphaned assistant tool call in history).
- [ ] **Step 2: Run** `RUST_MIN_STACK=8388608 cargo test --features sdk-engine sdk_engine_` — FAIL.
- [ ] **Step 3: Implement.**
  - Selection mirrors `bootstrap/run.rs:1146-1215` without an assessor: `engine.model_override` → `self.model` if `model_override` → depth > 0: `self.model` → router `select_for_profile(ModelProfile::Balanced)` → `snapshot.primary_model()`. Prefer extracting the legacy selection into a shared `pub(in crate::agent) fn` over copying.
  - After every successful call: `crate::events::record_model_call_telemetry(emitter, state, ModelCallTelemetryInput { session_id, task_id, call_purpose: "sdk_engine", iteration, llm_call, token_usage })` (`src/events/model_call_telemetry.rs:38`), and the goal budget accounting from `llm_phase.rs:1596-1620` (`add_goal_tokens_and_get_budget_status`, `goal_token_registry.add_run_tokens`). When the goal budget is exhausted, return `sdk::ProviderError::Other("goal token budget exhausted")` so the run ends `ProviderFailed` and the reply says so.
  - Context: before the call, `budget = compute_available_budget_precomputed(model, system_tokens, &tools, &self.context_window_config)` and `fit_messages_with_source_quotas(messages, budget)` (`src/memory/context_window.rs:344,392`); then drop any `tool` message whose call is no longer directly before it and any assistant `tool_calls` entry without a following result. Raise `HISTORY_MESSAGES` to 200 now that fitting bounds the size.
- [ ] **Step 4: Run** Step 2 command — PASS.
- [ ] **Step 5: Commit** `feat(sdk-engine): daemon model selection, telemetry, budgets and context fitting`.

### Task 3: Live, typed tool records

**Files:**
- Modify: `src/agent/runtime/sdk_engine.rs` (`serve_sdk_model_call` persists assistant tool-call messages; `dispatch_sdk_tool_call` emits `ToolCall`, `ToolResult` with receipt, `ExecutorExpectationsDeclared`; delete `persist_sdk_tool_exchanges` and the post-run call)

**Interfaces:**
- Consumes: `ToolCallData::from_tool_call(id, name, args_json, Some(task_id))` (`src/events/payloads.rs:1609`), `append_tool_message_with_receipt_event_policy(..., receipt: Option<ToolReceiptV1>)` (`src/agent/runtime/turn_context.rs:865`), `executor_expectations_from_checklist_arguments(task_id, arguments)` (`src/tools/track_requirements.rs:297`), the legacy receipt builder used at `tool_execution/run.rs:3296` (find where its `ToolReceiptV1` is built and reuse it with `invocation_stage: ToolInvocationStage::Dispatched`).
- Produces: every dispatched call has, in event order, assistant message (with `tool_calls`), `ToolCall`, `ToolResult` (receipt stage `Dispatched`), all with the turn's `task_id`; mandate-refused calls get `ToolResult` with a non-dispatched/denied receipt.

- [ ] **Step 1: Write failing tests:**
  - `sdk_engine_emits_tool_call_before_result`: one tool call → events for the task in order `ToolCall`, `ToolResult`; `RunAggregate` via `event_store.task_run_aggregate(session, task)` has 1 operation with a terminal receipt and no invariant violations.
  - `sdk_engine_counts_dispatched_tools`: `harness_eval` `dispatched_tool_names` for the task == `["remember_fact"]`.
  - `sdk_engine_keeps_tool_results_when_provider_fails` (Review Focus 1): MockProvider returns a tool call, then an error → task has 1 `ToolResult` and TaskEnd `failed`.
  - `sdk_engine_track_requirements_declares_expectations`: a `track_requirements` call with 2 items → one `ExecutorExpectationsDeclared` event; aggregate `open_executor_expectations().len() == 2`.
- [ ] **Step 2: Run** `RUST_MIN_STACK=8388608 cargo test --features sdk-engine sdk_engine_` — FAIL.
- [ ] **Step 3: Implement.** Assistant messages are persisted in `serve_sdk_model_call` when the response has tool calls (text + `tool_calls_json`, ids through `wire_call_id`). Tool results persist in `dispatch_sdk_tool_call` with the exact text sent back to the model. The final answer is still persisted by the existing end-of-run code.
- [ ] **Step 4: Run** — PASS. Also `cargo test --features sdk-engine --lib sdk_engine::tests` for unchanged unit tests.
- [ ] **Step 5: Commit** `feat(sdk-engine): record tool calls and receipts as they happen`.

### Task 4: Cancellation

**Files:** Modify `src/agent/runtime/sdk_engine.rs` (`handle_message_sdk`).

- [ ] **Step 1: Write failing test** `sdk_engine_stops_when_cancelled` (Review Focus 3): agent with `cancel_token = Some(token)`; MockProvider first returns a tool call whose tool (test tool) cancels the token while running; the run makes no further provider call (`call_count() == 1`), TaskEnd status `cancelled`, reply `"Cancelled."`.
- [ ] **Step 2: Run** — FAIL (second call happens).
- [ ] **Step 3: Implement:** `run_from(state, self.cancel_token.as_ref().map(|t| t.child_token()).unwrap_or_default())`. The kernel already returns `RunStatus::Cancelled`, which `reply_for` maps.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `fix(sdk-engine): honour parent and goal cancellation`.

### Task 5: Ledger closeout as a stop hook; receipt-backed TaskEnd

**Files:**
- Modify: `src/agent/runtime/sdk_engine.rs` (`DaemonCall::Stop`, `LedgerStop` hook, `serve_sdk_stop`, TaskEnd outcome)
- Modify: `src/agent/loop/closeout.rs` (bump `ClosingAuthority`, `from_authority`, `obligation_admissible` to `pub(in crate::agent)`)

**Interfaces:**
- Produces: `struct StopCheck { final_text: String, reply: oneshot::Sender<sdk::StopDecision> }`; `struct LedgerStop { calls: mpsc::Sender<DaemonCall> }` implementing `sdk::Hook::on_stop`; `Agent::serve_sdk_stop(&self, scope, check)`; `TurnScope.demands: AtomicU32`.

- [ ] **Step 1: Write failing tests:**
  - `sdk_engine_demands_open_checklist_item`: model calls `track_requirements` with 2 items, does 1, answers → one extra turn happens with a message containing the open item; on the second answer with still one open item and no new completed results, the run ends (≤ 2 demands, Review Focus 4) with TaskEnd outcome `partial`.
  - `sdk_engine_evidence_closed_run_succeeds`: tool call succeeds, checklist fully done → outcome `succeeded`, `TaskEndData.completion_proof` / outcome matches `aggregate.terminal_decision() == SucceededByEvidence` path used by `graceful.rs:1086`.
  - `sdk_engine_no_demand_after_policy_denial`: aggregate with terminal policy denial → `Accept`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.** `serve_sdk_stop` replays `self.event_store.task_run_aggregate(session, task)`, runs `closeout(|o| obligation_admissible(o, &authority, &visible, read_only))` as `completion_phase.rs:703-760` does, and applies the same gate as `completion_phase.rs:1134` (demand only when reachable and either no completed results yet or a checklist obligation is open with progress since the last demand; never after `terminal_policy_denial()`; cap 2). The `Continue` message is the rendered `SystemDirective::LedgerExpectationsRequired { expectations }` text. Without contract/authority (until Task 6) use `ClosingAuthority` from the request's default authority. TaskEnd: reuse `emit_task_end` from `src/agent/runtime/graceful.rs:1086` (or its reconciliation helper) instead of the hand-built `TaskEndData`, so outcome folds `terminal_decision()`.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(sdk-engine): ledger closeout and receipt-backed task outcomes`.

### Task 6: Assessor contract and authority lane

**Files:**
- Modify: `src/agent/loop/bootstrap/run.rs` (extract the assessment + `TaskContractCompiled` emission in `finalize_turn_assessment`, run.rs:87 and :405-445, into `pub(in crate::agent) async fn assess_and_record_contract(agent: &Agent, emitter: &EventEmitter, session_id: &str, task_id: &str, user_text: &str, tool_names: &[String]) -> Option<CompletionContract>`; legacy calls it unchanged)
- Modify: `src/agent/runtime/sdk_engine.rs` (call it before the run; authority-gate tools in `dispatch_sdk_tool_call`; pass authority to `serve_sdk_stop`)

- [ ] **Step 1: Write failing tests:**
  - `sdk_engine_records_task_contract`: MockProvider scripted to answer the assessor (same fixture the legacy `finalize_turn_assessment` tests use) → one `TaskContractCompiled` event for the task.
  - `sdk_engine_refuses_tool_outside_authority`: contract authority forbids a mutating tool → call gets an error result, no `Dispatched` receipt, the run continues.
  - Legacy regression: `cargo test integration_tests` unchanged.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** the extraction (move, don't copy). The authority check uses `CompletionContract::authority()` only (CLAUDE.md: the only contract view a gate may read); expectations stay verified by receipts in Task 5.
- [ ] **Step 4: Run** sdk tests + `RUST_MIN_STACK=8388608 cargo test integration_tests` — PASS.
- [ ] **Step 5: Commit** `feat(sdk-engine): assessor contract and authority lane`.

### Task 7: Same-tool stall limit hook

**Files:** Modify `src/agent/runtime/sdk_engine.rs` (new `SameToolLimit` hook registered on the builder).

- [ ] **Step 1: Write failing test** `same_tool_limit_redirects_after_threshold` (unit, drives the hook directly with a `HookContext` whose `run_receipts` hold 8 consecutive `read_file` calls with distinct arguments): 9th `read_file` → `ToolDecision::Deny` with a redirect message; a different tool → `Allow`; research tools (`web_search`, `web_fetch`) use 16.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** with `MAX_CONSECUTIVE_SAME_TOOL` (`src/agent/mod.rs:104`), the +8/+4 terminal/read_file allowances and research ×16 from `tool_execution/guards.rs:283-330`; exact-repeat detection stays with the kernel's `StuckConfig` (already on).
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(sdk-engine): same-tool stall limit`.

### Task 8: SDK is the only engine; delete the legacy loop

Owner decision (2026-10-31): no side-by-side comparison and no second engine. When the suite passes on the SDK engine, the legacy loop goes.

**Files:**
- Modify: `Cargo.toml` (drop the `sdk-engine` feature; `aidaemon-core` becomes a required dependency, packaged per the owner's publishing decision), `src/config.rs` (`AgentEngineKind` removed; `[agent.engine] kind` accepted and ignored with a one-time `warn!` so old configs still load), `src/core.rs`, `src/agent/loop/main_loop.rs` (`handle_message_impl` calls the SDK turn directly)
- Delete: legacy-only modules under `src/agent/loop/` once nothing references them (compiler-driven: remove the dispatch, then delete what becomes dead; keep shared pieces still used by the SDK path such as `closeout.rs`, prompt building, model selection, assessment)
- Modify: tests that asserted legacy-only behavior (intent gate wording, recovery ladder, stall ladder) — port to the SDK path when the behavior was kept by Tasks 1-7, delete only when the behavior was deliberately dropped, and list each deletion in the commit message
- Modify: `crates/aidaemon-core/PLAN.md`, `config.toml.example`, `CHANGELOG.md` (Unreleased: engine replaced, `[agent.engine] api_key`/`kind` ignored)

- [ ] **Step 1:** Remove the engine switch so every turn runs on the SDK path; run `RUST_MIN_STACK=8388608 cargo test --all-features` and `cargo test --lib harness_eval --all-features`.
- [ ] **Step 2:** Fix failures by porting the missing behavior into the SDK path (hooks or port handlers), never by tightening gates; triage each failing legacy-only test as port/delete.
- [ ] **Step 3:** Delete dead legacy code until `cargo clippy --all-features -- -D warnings` has no dead-code warnings.
- [ ] **Step 4:** Full checklist: `cargo fmt`, clippy, `RUST_MIN_STACK=8388608 cargo test --all-features`, `cargo build --release --features browser`, `cargo package` succeeds.
- [ ] **Step 5: Commit** `refactor(agent): the SDK kernel is the only agent loop`.

## Task 8 measurement (2026-11-01, after Tasks 1-7)

Every agent forced onto the SDK engine, `cargo test --features sdk-engine --lib`:
**4314 passed, 106 failed** (72 integration, 22 characterization, 11 harness eval, 1 other).
Per-test list: `.superpowers/sdd/2026-10-31-sdk-engine-parity/t8-triage.txt` (local).
Causes, largest first:

| # | Legacy behavior not yet on the SDK path | Failing tests (approx.) |
|---|---|---|
| D | Recovery and verification ladder: required evidence passes, verification before completion, failed-mutation recovery pass, guided `tool_choice=required` forcing, force-text closeout grace, unknown-tool stall, fabricated-delegation block | 22 |
| C | Reply quality: empty-response retry, gutted / JSON-dump / pasted-page reply retry, `[INTENT_GATE]` and untrusted-envelope stripping, deterministic completion acks, honest fallbacks | 16 |
| E | Dispatcher safety: duplicate `send_file` suppression, durable-receipt replay of duplicate mutations, semantic read cache, postcondition re-execution, evidence gate on blind edits, pre-execution plan / critique gate, canonical read-only batch order, nested project-instruction deferral, truncation notices | 14 |
| B | Context projection: fresh-request isolation, `[Current Task]` boundary, old-turn truncation, compaction, byte-identical archived prefix, exact antecedent projection | 14 |
| A | `harness_eval` snapshot on TaskEnd | 11 |
| F | Budgets: task budget auto-extend / unproductive stop, goal budget auto-extend, scheduled-run budget stops | 6 |
| G | Misc: audio attachments, image payload shape, status-update summaries, mandate prose closeout and owner-input context, `report_blocker` termination, specialist plan pivot, reflection learning | 15 |

## Out of scope (owner decisions)

- How `aidaemon-core` ships with the published `aidaemon` crate (publish it to crates.io, or fold it into the main crate). Blocks Task 8 Step 4 (`cargo package`), not Tasks 1-7.
