# Mandate Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bounded mandates reliably reach a decision, stop the model from doing bookkeeping the runtime already has, and fail fast on mandates that can never work — without loosening any authority boundary.

**Architecture:** The safety core stays as is (owner-only creation with confirmation, task-lead/executor split, fences, quotas, receipt-verified evidence). Five changes remove the layers that turn into dead ends: the metric control loop becomes an autopilot-only requirement, the runtime attaches current-run evidence receipts, the draft rejects scopes whose tool can never reach the target, the owner's mandate control tool survives a request assessment's tool narrowing, and the task-lead prompt drops the rules those changes make obsolete.

**Tech Stack:** Rust, tokio, sqlx (SQLite), the daemon's SDK engine turn path.

**Spec:** this conversation's review (2026-11-02), summarized as the Problem section below.

## Problem (spec)

Evidence from 2026-11-02: creating a one-hour, one-note mandate took four attempts, each failing in a different layer; lifetime mandate history is 146 WAIT / 16 ACT / 12 successful mutations.

1. Bounded mandates get an LLM-invented `objective_control`, and any mandate with one rejects ACT and WAIT until a receipt-backed measurement exists (`src/state/sqlite/mandates.rs:3195`). The task-lead prompt also tells the model to ASK for a control loop whenever a success criterion names a quantity.
2. Measurements and decisions need the model to copy receipt identifiers it may not see; the runtime already knows the run's successful receipts.
3. A draft can authorize `web_fetch` for `localhost`, which `web_fetch` always refuses (`validate_url_for_ssrf`); the failure appears only at the first review.
4. A request assessment's `allowed_tool_names` (authority lane) can hide `manage_mandates` on the owner's own mandate-management turn (`src/agent/loop/main_loop.rs:760`, `ClosingAuthority::admits_tool`).
5. The task-lead protocol prompt (`src/agent/runtime/system_prompt.rs:705`) is ~5,000 characters, much of it rules for 1 and 2.

## Global Constraints

- Authority is never widened for a non-owner, a non-private channel, an internal/mandate worker, or a request that forbids tool use (`forbids_tool_use`).
- Owner-confirmation for mandate activation is unchanged (`confirmation_is_durable_atomic_and_required_for_activation` keeps passing).
- Receipts stay the only evidence: an attached receipt must be a current-run `succeeded` structured receipt, exactly as `validate_current_run_receipt_refs` already checks.
- Autopilot mandates keep their objective-control requirement and measurement gate unchanged.
- No phrase/keyword matching on user or model text for any decision (repo rule, CLAUDE.md).
- Test data is synthetic only (`Acme`, `telegram:synthetic-user-1`, `localhost` fakes).
- Every task ends with `cargo fmt`, `cargo clippy --all-features -- -D warnings`, and its focused tests; the last task runs `RUST_MIN_STACK=8388608 cargo test --all-features`.

## Review Focus

- A bounded mandate whose owner *did* specify an objective control: the control is kept and reported, but ACT/WAIT are not blocked on a measurement — pinned in Task 1.
- A run with no successful observation and empty `evidence_receipt_ids`: ACT must still be rejected with an actionable message (not silently attached to nothing) — pinned in Task 2.
- Receipts from the mandate's own control tools (`manage_mandates`, `manage_goal_tasks`, `track_requirements`, `report_blocker`) must never be attached as evidence — pinned in Task 2.
- `http_request` to `localhost` must stay allowed in scopes (the local test path) while `web_fetch` to `localhost`/`*.local`/metadata hosts is rejected — pinned in Task 3.
- An assessment that narrows tools on a *non-owner* or *shared-channel* turn must not gain `manage_mandates` — pinned in Task 4.

---

### Task 1: The measurement gate applies to autopilot mandates only

**Files:**
- Modify: `src/state/sqlite/mandates.rs` (decision gate near line 3195; same function's stagnation check that follows)
- Test: `src/state/sqlite/mandates.rs` test module

**Interfaces:**
- Consumes: `Mandate::autonomy_mode: MandateAutonomyMode`, `MandateAutonomyMode::is_autopilot()`.
- Produces: no new API. Behavior: for `Bounded` mandates an `objective_control` is advisory — `record_mandate_decision*` no longer requires a current-run measurement and skips the stagnation rejection; measurements, when recorded, are still validated and stored.

- [ ] **Step 1: Write the failing test** `bounded_mandate_with_objective_control_decides_without_a_measurement`: create a bounded mandate (`autonomy_mode = Bounded`) with a valid `objective_control` (`metric_source:synthetic-analytics`), claim it, start its review run, record a WAIT decision with no measurement → `Ok`, and the stored decision outcome is `Wait`. Second case in the same test: the same setup with `Autopilot` → `Err` containing `"receipt-backed metric measurement"`.
- [ ] **Step 2: Run** `RUST_MIN_STACK=8388608 cargo test --lib bounded_mandate_with_objective_control_decides_without_a_measurement` — FAIL (bounded case returns the measurement error).
- [ ] **Step 3: Implement** — guard the measurement requirement and the stagnation rejection with `mandate.autonomy_mode.is_autopilot()`; keep the runtime-fallback exemption.
- [ ] **Step 4: Run** the test plus `cargo test --lib mandates` — PASS.
- [ ] **Step 5: Commit** `fix(mandates): bounded mandates decide without a measurement gate`.

### Task 2: The runtime attaches current-run evidence receipts

**Files:**
- Modify: `src/traits/state_store.rs` (mandate store section, next to `get_mandate_decision_for_run`)
- Modify: `src/state/sqlite/mandates.rs` (implementation; reuse the predicate of `validate_current_run_receipt_refs`)
- Modify: `src/tools/manage_mandates.rs` (`record_measurement` ~line 1817, `record_decision` ~line 1943; `EVIDENCE_RECEIPT_IDS_DESCRIPTION`)
- Modify: `src/mandates/mod.rs` (new const)
- Test: `src/tools/manage_mandates.rs` test module

**Interfaces:**
- Produces: `pub const MANDATE_PROTOCOL_TOOLS: &[&str] = &["manage_mandates", "manage_goal_tasks", "track_requirements", "report_blocker"];` in `src/mandates/mod.rs`.
- Produces: `async fn current_run_evidence_receipt_ids(&self, goal_run_id: &str, limit: usize) -> anyhow::Result<Vec<String>>` on the store trait (default `Ok(Vec::new())`), returning the `tool_call_id`s of the run's `tool_result` events that pass the same checks as `validate_current_run_receipt_refs` (task in the goal run, `receipt.schema_version` current, `outcome_status = 'succeeded'`, `outcome_evidence IN ('tool_reported','structured_metadata')`), excluding `MANDATE_PROTOCOL_TOOLS` by `$.name`, newest first, at most `limit`.
- Behavior: when the model passes empty `evidence_receipt_ids` to `record_measurement` or `record_decision`, the tool fills them with `current_run_evidence_receipt_ids(run, MAX_EVIDENCE_RECEIPTS)`. Explicit ids are kept and validated as today. `validate_act_evidence` still requires non-empty evidence after filling.

- [ ] **Step 1: Write the failing tests** `record_decision_attaches_current_run_receipts_when_none_cited` (a run with one succeeded `http_request` result and one succeeded `manage_goal_tasks` result; ACT with `evidence_receipt_ids: []` and one observation succeeds; the stored decision's `evidence_receipt_ids` equals exactly the `http_request` call id) and `act_without_any_current_run_observation_is_still_rejected` (no observation results; ACT with empty ids → `Err` containing `"ACT requires at least one current-run sourced observation"`). Add the same attach assertion for `record_measurement` in `record_measurement_attaches_current_run_receipts_when_none_cited`.
- [ ] **Step 2: Run** the three tests — FAIL.
- [ ] **Step 3: Implement** the const, the trait method + SQLite query, and the fill in both tool actions. Update `EVIDENCE_RECEIPT_IDS_DESCRIPTION` to: optional; omitted means the runtime cites this run's successful observations.
- [ ] **Step 4: Run** the tests plus `cargo test --lib manage_mandates` and `test_sdk_task_lead_commits_act_and_delegates_the_mutation` — PASS.
- [ ] **Step 5: Commit** `feat(mandates): the runtime cites current-run evidence receipts`.

### Task 3: Drafts reject scopes their tool can never reach

**Files:**
- Modify: `src/tools/manage_mandates.rs` (draft readiness ~line 480–520 and the `create` validation path)
- Test: `src/tools/manage_mandates.rs` test module

**Interfaces:**
- Consumes: `crate::tools::web_fetch::validate_url_for_ssrf(url: &str) -> Result<(), String>`.
- Produces: `fn unreachable_scope_targets(scopes: &[MandateOperationScope]) -> Option<String>` — for every scope with `tool == "web_fetch"`, each target prefix must pass `validate_url_for_ssrf`; the first failure returns `"web_fetch cannot reach {prefix} ({reason}); use http_request for local or private targets"`. Other tools return `None`. Draft reports it as `validation_status` with `ready_to_confirm = false`; `create` bails with the same text.

- [ ] **Step 1: Write the failing tests** `draft_rejects_web_fetch_scope_for_localhost` (web_fetch GET `http://localhost:8977/feed` → `ready_to_confirm == false`, status contains `"use http_request"`) and `draft_accepts_http_request_scope_for_localhost` (same target with `http_request` → that status absent).
- [ ] **Step 2: Run** — first FAILS, second passes.
- [ ] **Step 3: Implement** `unreachable_scope_targets` and call it in draft and create.
- [ ] **Step 4: Run** both plus `cargo test --lib manage_mandates` — PASS.
- [ ] **Step 5: Commit** `fix(mandates): drafts reject scopes their tool cannot reach`.

### Task 4: The owner's mandate control survives tool narrowing

**Files:**
- Modify: `src/traits/tools.rs` (`Tool` trait)
- Modify: `src/tools/manage_mandates.rs` (`impl Tool`)
- Modify: `src/agent/loop/bootstrap/run.rs` (after the semantic contract is applied, before `TaskContractCompiled`; and the "Task assessment succeeded" metadata ~line 555)
- Test: `src/integration_tests/part_16_tool_result_boundary.rs`

**Interfaces:**
- Produces: `fn owner_control_plane(&self) -> bool { false }` on `Tool`; `ManageMandatesTool` returns `true`. Meaning: the tool only manages the owner's own durable policy and every committing action is already owner-gated (private channel + confirmation).
- Produces: `fn admit_owner_control_plane(contract: &mut CompletionContract, tools: &[Arc<dyn Tool>], user_role: UserRole, channel: &ChannelContext)` in `bootstrap/run.rs` — when `allowed_tool_names` is non-empty, `forbids_tool_use` is false, `user_role == Owner`, `channel.visibility == Private`, and the agent has no mandate fence, appends each available `owner_control_plane()` tool name not already present. Because `main_loop` and `ClosingAuthority` both read the contract, one edit covers visibility and the per-call gate.
- Produces: `"allowed_tool_names"` in the "Task assessment succeeded" decision-point metadata (it was missing, which hid this failure).

- [ ] **Step 1: Write the failing test** `test_owner_mandate_control_survives_assessment_tool_narrowing`: an assessment response (schema_version 10) whose contract sets `allowed_tool_names: ["http_request"]`; owner, `ChannelContext::private("test")`, harness from `setup_test_agent_with_mandates`; assert the first non-assessment `call_log` entry offers both `http_request` and `manage_mandates`. Second case: same assessment with `UserRole::Guest` (or a public channel) → `manage_mandates` absent.
- [ ] **Step 2: Run** — FAIL (owner case offers only `http_request`).
- [ ] **Step 3: Implement** the trait method, the override, `admit_owner_control_plane`, and the metadata field.
- [ ] **Step 4: Run** the test, `sdk_engine_refuses_tools_outside_the_contract_authority`, and `read_only_contract_admits_*` — PASS.
- [ ] **Step 5: Commit** `fix(agent): owner mandate control survives assessment tool narrowing`.

### Task 5: Slim the task-lead protocol prompt

**Files:**
- Modify: `src/agent/runtime/system_prompt.rs` (task-lead `role_rules` ~line 705; prompt tests ~line 1750)

**Interfaces:**
- Consumes: Tasks 1–2 behavior.
- Produces: task-lead rules where (a) `evidence_receipt_ids` are described as optional (the runtime attaches this run's successful observations), (b) the "objective_control is null but a success criterion names a measurable quantity → ASK" rule is removed, (c) the control-loop paragraph (record_measurement, stagnation, STOP on target) is rendered only when the mandate is autopilot with an objective control.

- [ ] **Step 1: Write the failing test** `bounded_task_lead_prompt_omits_control_loop_rules`: render the task-lead prompt for a bounded mandate without objective control → does not contain `"record_measurement"` nor `"proposes a complete objective_control"`; for an autopilot mandate with control → contains `"record_measurement"`. Assert the bounded role rules are at least 30% shorter than today's (capture today's length as a literal in the test).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** by splitting the rule string into a base block and a control-loop block chosen by `mandate.autonomy_mode.is_autopilot() && mandate.objective_control.is_some()`; drop the receipt-copy instructions.
- [ ] **Step 4: Run** the test and the existing system-prompt tests — PASS.
- [ ] **Step 5: Commit** `refactor(mandates): task-lead rules only carry what the mandate uses`.

### Task 6: Full suite, deploy, live check

- [ ] **Step 1: Run** `RUST_MIN_STACK=8388608 cargo test --all-features` — all pass.
- [ ] **Step 2: Deploy** when no mandate run is in flight: `cargo build --release --features computer_use-macos`, then `scripts/package-macos-app.sh --release`; confirm the `Agent engine: SDK kernel` startup line.
- [ ] **Step 3: Live check** with the local listener (`localhost:8977`): the owner sends a bounded one-hour, one-note mandate naming the tools; expect the draft to succeed on the first message, one POST in `notes.jsonl`, an ACT then `Succeeded` attempt, and WAITs for the rest of the hour.
