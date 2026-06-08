# Telemetry Outcome Semantics Design

## Status

Draft, revised after multi-agent design review on 2026-06-07.

## Purpose

Make task telemetry distinguish successful, partially successful, and failed
user outcomes, preserve terminal-cause information such as cancellation,
normalize error reporting, and give every token-accounted model call equivalent
detailed observability.

## Scope

This change covers:

- additive task-end outcome semantics alongside the existing terminal status;
- task outcome derivation from the root agent's existing completion evidence;
- error aggregation using the structured `error_type` field;
- one shared model-call telemetry recording path for root and background calls;
- correlation-based reconciliation for legacy or missed detailed events;
- migration of task-outcome consumers in diagnostics, consolidation, and
  reporting.

It does not change goal/task workflow statuses, provider retry behavior, tool
execution policy, or historical rows already stored in SQLite.

## Task-End Data Model

Introduce a serialized `TaskOutcome` enum:

- `succeeded`: the requested outcome was produced. Recovered or incidental
  errors did not prevent it.
- `partial`: the response provides user-visible progress, but at least one
  required outcome remains unresolved.
- `failed`: no requested outcome or useful fallback result was produced.

`TaskEndData` keeps the existing `status: TaskStatus` field and adds
`outcome: Option<TaskOutcome>`. These fields represent different dimensions:

- `status` records how execution terminated: `completed`, `cancelled`, or
  `failed`;
- `outcome` records the semantic result delivered by that execution:
  `succeeded`, `partial`, or `failed`.

New task-end events always serialize `Some(outcome)` alongside `status`. The
field is stored as an `Option` only so that legacy events, which predate the
field, deserialize via `#[serde(default)]` to `None` instead of failing. A
`None` is never written by new code. `partial` is not added to `TaskStatus`,
because workflow and lifecycle code must continue to treat a naturally
terminated partial response as completed execution.

| Terminal condition | `status` | `outcome` |
| --- | --- | --- |
| Natural completion, all required outcomes satisfied | `completed` | `succeeded` |
| Natural completion, useful progress with unresolved required outcomes | `completed` | `partial` |
| Natural completion, no requested outcome or useful fallback produced | `completed` | `failed` |
| User or parent cancellation | `cancelled` | `failed` |
| Timeout, watchdog termination, or unrecovered hard failure | `failed` | `failed` |

Cancellation remains separately observable through `status`; it is not counted
as an agent hard failure when reports group terminal causes.

### Compatibility And Precedence

Deserialization uses explicit compatibility rules:

1. If `outcome` is present, it is authoritative for semantic-outcome queries.
2. If `outcome` is absent and `status == "completed"`, the effective outcome is
   `succeeded`.
3. If `outcome` is absent and `status` is `cancelled` or `failed`, the effective
   outcome is `failed`.
4. `status` remains required. A payload with neither usable `status` nor a
   valid task-end shape is malformed and is excluded from typed outcome counts,
   while reconciliation reports it as unknown.

`TaskEndData::effective_outcome() -> TaskOutcome` is the single resolver and the
only place the fallback lives. It returns the stored `outcome` when present and
otherwise derives one from `status` per the rules above. Consumers always call
`effective_outcome()`; they never read the raw `Option`, compare JSON strings,
or independently reconstruct the mapping.

This deliberately avoids a custom `Deserialize` impl: the field deserializes
with a plain `#[serde(default)]`, and the `status`-based fallback exists in
exactly one read path (`effective_outcome()`) rather than being split between a
deserializer and a resolver where the two could disagree. Legacy task-end events
therefore remain readable without rewriting stored rows.

## Outcome Derivation

Outcome derivation occurs once for each agent task at the common
`emit_task_end` boundary. Every exit path supplies a typed
`TaskOutcomeDerivation` input or an explicit outcome override; callers no longer
select semantic success by passing a hard-coded `TaskStatus::Completed`.

The derivation input contains:

```rust
struct TaskOutcomeDerivation {
    response_produced: bool,
    response_has_user_value: bool,
    required_actions: RequestedActionSummary,
    has_unrecovered_model_error: bool,
    terminal_cause: Option<TaskTerminalCause>,
}

struct RequestedActionSummary {
    required: u32,
    satisfied: u32,
    unresolved: u32,
}
```

`RequestedActionSummary` maintains
`required == satisfied + unresolved`. `TaskTerminalCause` distinguishes
`cancelled`, `timeout`, `watchdog`, `hard_failure`, and
`unrecovered_model_failure`.

`response_produced` means the agent reached the final accepted-response branch
with sanitized, non-empty assistant content ready to return. It does not claim
that a channel successfully transmitted the message. Channel delivery failure
remains separate telemetry and is outside task outcome derivation.

`response_has_user_value` is determined by the existing completion and response
quality gates after repetition collapse, user-facing sanitization, visibility
redaction, and the final response-quality check. Empty output, structural-only
text, unrecovered timeout boilerplate, and generic acknowledgements produced
without supporting evidence do not qualify. This verdict is produced at the
final accepted-response branch; it is not currently a stored
`CompletionProgress` field.

`RequestedActionSummary` is built from existing completion evidence:

- active success criteria and matched criteria;
- linear-intent or plan-step completion state when present;
- verification targets and pending or failed observations;
- attempt reconciliation for failed actions linked to those required criteria,
  including observations as well as mutations;
- verified successful external mutations.

It is not derived from raw tool failure counts. An incidental failed lookup or
a failed attempt later corrected by an equivalent verified action does not
create an unresolved requested action. Existing reconciliation and validation
logic is reused where it applies. The implementation extends the requested
action summary beyond `uncorrected_failed_mutations()` so failed navigation,
inspection, search, and read actions can remain unresolved even though they are
not external mutations. Outcome derivation does not invent a separate recovery
algorithm.

The summary merges its sources by a defined rule so the
`required == satisfied + unresolved` invariant always holds:

- The **required set** is the union of explicit success criteria, incomplete
  linear-intent/plan steps, and pending verification targets — the actions the
  task was actually asked to accomplish. An action is counted once even if it
  appears in more than one source (criteria text matched to a verification
  target is the same required action, not two).
- `satisfied` is the subset of the required set with matching evidence (matched
  criteria, completed steps, verified mutations/observations).
- `unresolved` is the remainder of the required set: `required - satisfied`.
- Failed actions that are **not** in the required set — incidental lookups,
  exploratory reads, mutations with no linked criterion — never enter the
  summary and never create an `unresolved` count. This is what keeps an
  informational request with `required = 0` at `unresolved = 0` even when an
  incidental tool call failed, consistent with derivation rule 4.

When a required action is satisfied by an alternate path after an earlier
failure, it counts as satisfied, not unresolved — the existing reconciliation
view supplies that correction; the merge does not re-derive it.

Rules, in precedence order:

1. Cancellation, timeout, watchdog termination, or unrecovered hard/model
   failure produces `failed`.
2. No produced response or no user-value response produces `failed`.
3. One or more unresolved required actions produces `partial`.
4. Otherwise the outcome is `succeeded`.

Recovered errors do not affect the outcome after the corresponding required
action is satisfied. For informational requests with no required tool action,
a useful grounded response can therefore be `succeeded` even if an incidental
tool attempt failed.

For the observed browser example, failed navigation leaves the homepage
inspection criterion unresolved. A non-empty progress response therefore
produces `partial`; an empty or generic fallback with no user value produces
`failed`.

### Evidence Carriers And Emission Sites

The derivation inputs normalize evidence that already exists in the loop at the
point a task ends. The only new computation is assembling that evidence into one
requested-action summary:

- `required` and `satisfied` come from `ValidationState`:
  `active_success_criteria` and `matched_success_criteria`.
- unresolved mutation attempts reuse
  `ExecutionState::uncorrected_failed_mutations()` /
  `has_uncorrected_failed_external_mutations()`;
- unresolved non-mutation actions come from incomplete linear-intent steps,
  pending verification targets, and failed required observations already
  represented by `ExecutionState`, `ValidationState`, and
  `CompletionProgress`;
- `response_has_user_value` is computed from the accepted, sanitized response
  at the final response-quality branch. `CompletionProgress` supplies quality
  and verification context but does not currently store this verdict.

`emit_task_end` (`agent/runtime/graceful.rs`) is the one emission function, but
it is invoked from roughly three dozen call sites across the loop phases
(`completion_phase`, `stopping_phase`, `llm_phase`, `orchestration/routes`,
`tool_execution/guards`, `main_loop`, and others). These split into two kinds,
and the signature change accommodates both:

1. **Override paths** — stall kills, cancellation, watchdog/timeout, orchestration
   shortcuts, and other hard exits. These already know the terminal cause and
   pass an explicit `TaskOutcome` override (`failed`, occasionally `succeeded`
   for trivial shortcut completions). They do not assemble a
   `RequestedActionSummary`.
2. **Natural-completion paths** — any exit that delivers an accepted response and
   already holds the completion evidence, where the `succeeded`/`partial`
   distinction actually matters. This is not only `completion_phase.rs`:
   `stopping_phase.rs` also has multiple `emit_task_end(..., Completed, ...)`
   sites, and `StoppingPhaseCtx` carries the same `validation_state`,
   `execution_state`, and `completion_progress`. Every such site must use the
   helper, not just the primary one. The current code emits `TaskEnd` before
   repetition collapse, sanitization, and the response-quality guard; that guard
   can still reject the response and continue the loop. Natural-completion
   emission moves to the final accepted-response branch, immediately before
   returning the response. Wherever `ValidationState`, `ExecutionState`, and
   `CompletionProgress` are in scope, a single helper
   (`TaskOutcomeDerivation::from_completion_state(&ValidationState,
   &ExecutionState, &CompletionProgress, response_verdict)`) assembles the input
   at the call site with no new plumbing through intermediate phases. A
   natural-completion site that cannot reach the evidence structs is treated as a
   bug to fix, not an override path.

The implementation step is therefore: change `emit_task_end` to take a
`TaskOutcome` (resolved by caller) instead of inferring it, add the helper for
the natural-completion path, move that emission after final response acceptance,
and pass explicit overrides everywhere else. The requested-action helper extends
the existing reconciliation view to include required non-mutation actions; no
completion evidence needs to be threaded into phases that do not already hold
it.

### Root And Sub-Agent Boundaries

Each agent task, including spawned specialists, emits its own status and
outcome. A child failure does not automatically make the parent partial or
failed. It affects the parent only when the parent still has an unresolved
required action after considering alternate recovery paths and child results.

## Error Semantics

`ErrorData.error_type` is the canonical persisted error category. New
aggregation uses its snake-case serialized value and never groups by message
text.

`unknown` is an aggregation bucket, not initially a persisted `ErrorType`
variant. Aggregation reads raw event JSON so records that cannot deserialize as
current `ErrorData` are still counted. The fallback order is:

1. recognized `error_type`;
2. `unknown` when `error_type` is missing, malformed, or unrecognized.

Human-readable `message` remains available in diagnostic detail but is not used
to infer categories. The current persisted model has no separate legacy
category field, so no message or alternate-field inference is introduced.

## Unified Model-Call Telemetry

Create one shared `ModelCallTelemetryRecorder` operation that accepts:

- a unique `call_id`;
- session and optional task identifiers;
- iteration and `call_purpose`;
- requested/final model and fallback attempts;
- latency;
- optional token usage;
- optional prompt-build and prefix fingerprints.

The operation always attempts to write the detailed `llm_call` event. When
provider token usage is present, it also writes the aggregate `token_usage`
record. Absence of provider usage is recorded in the detailed event and is not
reported as a telemetry gap.

Both records carry the same unique `call_id`. This requires an additive nullable
`call_id` column and index on `token_usage`, plus `call_id` and `call_purpose`
fields on `LlmCallData`. Existing table names and existing columns remain
unchanged.

Root agent calls use the real task ID, iteration, build timing, and prefix
metadata. Background calls use:

- their existing session ID, such as `background:summarization`;
- a stable synthetic task ID equal to the background session ID;
- a stable `call_purpose`, such as `summarization`, `intent_classifier`,
  `consolidation`, or `skill_promotion`;
- a unique `call_id` per provider response;
- absent iteration, build, and prefix fields when they do not apply.

Fields that do not apply are optional and omitted. Event volume from migrated
background token-accounting calls is intentional and remains subject to normal
event retention.

### Dual-Write Behavior

The event and token stores currently have separate abstractions, so this design
does not require a cross-store transaction. The recorder attempts both writes
independently:

- both succeed: normal completion;
- detailed event succeeds and token write fails: log a structured one-sided
  write warning;
- token write succeeds and detailed event fails: log the corresponding warning;
- both fail: log both failures.

Warnings contain `call_id`, session, task or call purpose, and the failed side.
No compensating delete is attempted because deleting successful telemetry would
remove evidence needed for reconciliation. Telemetry failure does not fail the
user request.

Provider calls that intentionally do not record token usage remain out of scope
until they opt into the recorder. Every current call site that invokes
`record_token_usage` must migrate. The root path also uses the recorder for its
existing detailed event so there is only one construction path for
`LlmCallData`.

## Reconciliation

`db_probe` reports, for the selected time window:

- token usage rows with a non-null `call_id`;
- detailed LLM-call events that declare token usage present;
- call IDs present in both stores;
- token-only and event-only call IDs;
- uncorrelated legacy rows without a `call_id`;
- correlated and uncorrelated counts grouped by session;
- task outcomes grouped by semantic outcome and terminal status;
- errors grouped by canonical `error_type` or `unknown`.

Correlation by `call_id`, rather than count difference alone, detects
duplicates, offsetting one-sided failures, and mismatched records. A count
difference remains a summary, while orphaned call IDs are the authoritative
telemetry-gap evidence.

Historical rows are not backfilled. They remain visible as uncorrelated legacy
telemetry and are not mixed into the post-migration one-to-one success rate.

## Consumer Migration

### Writers

`emit_task_end` is the agent-loop writer, but two writers construct
`TaskEndData` directly and must also set `outcome` explicitly:

- the stale-task reconciler in `events/store.rs`, which emits synthetic
  `TaskEnd` events for `TaskStart`s that never terminated — these set
  `status: failed` and `outcome: Some(failed)`;
- the resume re-emission path in `agent/runtime/resume.rs`, which sets the
  outcome matching the status it records.

No `TaskEndData` is constructed with `outcome: None`; `None` exists solely for
reading legacy rows.

### Readers

The following consumers use `effective_outcome()` for semantic success:

- task outcome statistics in `events/store.rs`;
- diagnostics and failure filtering in `tools/diagnose.rs`;
- learning and procedure evidence in `events/consolidation.rs`;
- post-task learning in `agent/runtime/post_task.rs`;
- task outcome reporting in `db_probe`;
- dashboard task outcome aggregates.

`agent/runtime/post_task.rs` is the most important of these. `process_learning`
runs immediately after `emit_task_end` (`completion_phase.rs`, `llm_phase.rs`,
`tool_execution/run.rs`) and currently computes its own `task_success` from
`completed_naturally`, `explicit_positive_signals`, `explicit_negative_signals`,
and unrecovered-error counts, then feeds `increment_expertise` and reasoning-note
confidence. This must derive from the same `TaskOutcome` the task just emitted —
either by passing the resolved `TaskOutcome` into `process_learning` or by having
it call the same derivation helper with the same inputs. `succeeded` maps to
`task_success = true`; `partial` and `failed` map to `task_success = false`.
Without this, expertise and reasoning learning can record success for a task
telemetry reports as `partial` or `failed` — the dual-success-semantics gap this
design exists to remove.

Lifecycle behavior continues to use `status`:

- dialogue-state task closure;
- active-task and stale-task reconciliation;
- cancellation reporting;
- task-end hooks and cleanup.

This split prevents `partial` from reopening a completed task while ensuring
diagnostics and learning no longer treat every naturally completed response as
semantic success.

### Procedure Promotion Eligibility

Procedure extraction in `events/consolidation.rs` currently gates on
`status == TaskStatus::Completed` (`extract_procedures`, and the success mapping
at the per-task and label sites). Because `partial` tasks keep
`status: completed`, this gate must switch to `effective_outcome()` and treat
**only `succeeded`** as eligible. `partial` and `failed` tasks are excluded from
procedure promotion and from "successful" success-rate labels, even though their
terminal `status` is `completed`. Plan-step extraction, which already gates on
`StepStatus::Completed`, is unaffected.

## Testing

Tests are added before implementation for:

- serialization of every supported `status` and `outcome` combination;
- legacy `status: "completed"`, `"failed"`, and `"cancelled"` deserialization;
- authoritative `outcome` precedence when both fields are present;
- malformed task-end records being counted as unknown rather than silently
  successful;
- synthetic writers (stale-task reconciliation, resume re-emission) populating
  `outcome` rather than relying on legacy fallback;
- outcome derivation for success, partial success, failure, recovered errors,
  cancellation, timeout, and empty or valueless responses;
- corrected failures not creating unresolved requested actions;
- incidental tool failures not downgrading a satisfied informational request;
- failed required navigation, inspection, search, or read actions producing an
  unresolved action even though they are not mutations;
- an action appearing in multiple sources counted once so the
  `required == satisfied + unresolved` invariant holds;
- `process_learning` recording `task_success = false` for `partial` and `failed`
  outcomes, matching the emitted `TaskOutcome`;
- procedure promotion excluding `partial` tasks despite `status: completed`;
- the browser-failure scenario producing `partial`;
- a response rejected by the quality guard not emitting `TaskEnd` before the
  loop continues;
- natural completion emitting exactly one `TaskEnd` after sanitization and
  response acceptance;
- parent outcome derivation after child failure and alternate recovery;
- canonical `error_type` aggregation from raw JSON;
- unknown error aggregation without message-substring inference;
- shared recording producing a detailed event and, when usage exists, a token
  row with the same `call_id`;
- usage-absent calls producing a detailed event without a false gap;
- background recording preserving session, synthetic task ID, and call purpose;
- reconciliation detecting token-only, event-only, duplicate, legacy, and
  offsetting mismatch cases;
- migrated diagnostics, consolidation, `db_probe`, and dashboard consumers;
- all existing test fixtures that construct `TaskEndData` explicitly supplying
  `Some(outcome)`, except dedicated legacy-deserialization fixtures.

Focused tests run during development. Final verification follows the repository
checklist: `cargo fmt`, `cargo clippy --all-features -- -D warnings`, and
`cargo test`.

## Implementation Exit Criteria

- Every new task-end event contains both `status` and `outcome`.
- All task-end semantic consumers use `effective_outcome()`.
- `process_learning` and procedure promotion derive success from the task's
  `TaskOutcome`; no semantic-success path computes its own heuristic in parallel.
- All current `record_token_usage` callers use the shared recorder.
- Root and background token-accounted calls produce correlatable telemetry.
- `db_probe` reports no correlated gaps for a clean post-migration test run.
- Cancellation remains separately visible from hard failure.
- Existing legacy task-end and token-usage rows remain readable.

## Non-Goals

- Inferring whether response prose is truthful using another model.
- Retrofitting historical task outcomes or telemetry correlation IDs.
- Treating channel transmission as part of agent task outcome.
- Changing token prices or cost reporting.
- Changing provider APIs or retry policy.
- Renaming persisted goal/task workflow statuses.
