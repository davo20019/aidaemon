# Telemetry Outcome Semantics Design

## Purpose

Make task telemetry distinguish successful, partially successful, and failed
user outcomes, normalize error reporting, and ensure every recorded model call
has equivalent detailed observability.

## Scope

This change covers:

- task-end outcome semantics;
- task outcome derivation at the root agent boundary;
- error aggregation using the structured `error_type` field;
- one shared model-call telemetry recording path for root and background calls;
- reconciliation reporting for legacy or missed detailed events.

It does not change goal/task workflow statuses, provider retry behavior, tool
execution policy, or historical rows already stored in SQLite.

## Task Outcomes

Introduce a serialized `TaskOutcome` enum:

- `succeeded`: the requested outcome was delivered. Recovered or incidental
  errors do not prevent success.
- `partial`: the response delivered useful progress, but at least one requested
  outcome was not delivered.
- `failed`: the requested outcome was not delivered.

`TaskEndData` stores the enum as `outcome`. New events do not use the ambiguous
`completed` value as their outcome.

Deserialization remains compatible with legacy task-end events:

- legacy `status: "completed"` maps to `succeeded`;
- legacy failure statuses map to `failed`;
- missing outcome/status is interpreted conservatively by existing consumers
  without rewriting stored events.

The existing workflow status strings used by goals, scheduled tasks, and
delegated task persistence remain unchanged.

## Outcome Derivation

Outcome derivation occurs once at the root task completion boundary. Inputs are:

- whether a non-empty user-facing response was produced;
- whether requested tool actions succeeded;
- whether tool or model errors occurred;
- explicit terminal causes such as cancellation, timeout, or hard failure.

Rules:

1. Cancellation, timeout, unrecovered model failure, or no delivered response
   produces `failed`.
2. A useful response with failed requested actions produces `partial`.
3. A useful response with all requested actions delivered produces `succeeded`.
4. Recovered errors that do not affect the requested outcome remain
   `succeeded`.

The caller may provide an explicit terminal override for paths where the cause
is already known. Otherwise, a small pure derivation helper applies these
rules. The helper is unit tested independently.

For the observed browser example, two failed navigation calls followed by
`"I completed 2 actions."` produces `partial`, because a response exists but
the requested homepage inspection was not delivered. If no useful fallback
answer existed, it would produce `failed`.

## Error Semantics

`ErrorData.error_type` is the canonical error category. Diagnostic queries and
aggregation use its serialized value first, then fall back to legacy fields
only when reading older payloads.

Error summaries retain the human-readable `message`, but grouping does not use
message text. Unknown is reserved for malformed or legacy records that expose
no recognized category.

## Unified Model-Call Telemetry

Create one shared recording operation that accepts:

- session and optional task identifiers;
- iteration or call-purpose metadata;
- requested/final model and fallback attempts;
- latency;
- input, cached, cache-creation, fresh, and output tokens;
- optional prompt-build and prefix fingerprints.

The operation writes both:

1. the aggregate `token_usage` record;
2. the detailed `llm_call` event.

Root agent calls use full task and prefix metadata. Background calls use a
stable synthetic task identifier and call-purpose metadata, while fields that
do not apply remain absent. Existing background session IDs such as
`background:summarization` remain unchanged.

The shared operation reports either write failure. If only one write succeeds,
it logs a structured reconciliation warning containing the session, task/call
identifier, and which side failed. Telemetry failure does not fail the user
request.

Provider calls that intentionally do not record token usage remain out of scope
until they opt into this operation. Every current call site that invokes
`record_token_usage` must migrate to the shared operation.

## Reconciliation

`db_probe` reports, for the selected time window:

- token usage row count;
- detailed LLM-call event count;
- the difference;
- counts grouped by session;
- explicit task outcomes;
- errors grouped by canonical `error_type`.

A non-zero difference is labeled as a telemetry gap, not silently presented as
equivalent datasets. Historical gaps remain visible because existing database
records are not backfilled.

## Compatibility

- Existing task-end JSON remains readable.
- Existing event type names and token usage tables remain unchanged.
- Existing goal and scheduled-task status handling is unaffected.
- Consumers that previously checked task completion must switch to semantic
  outcome helpers rather than compare raw strings.
- New optional detailed-call fields use Serde defaults so older events continue
  to deserialize.

## Testing

Tests are added before implementation for:

- outcome derivation for success, partial success, failure, recovered errors,
  cancellation, timeout, and empty responses;
- legacy `status: "completed"` deserialization;
- new outcome serialization;
- canonical `error_type` aggregation with legacy fallback;
- shared recording producing both token and detailed records;
- shared recording preserving background session and synthetic task metadata;
- reconciliation detecting a one-sided write or count mismatch;
- the browser-failure scenario producing `partial`.

Focused tests run during development. Final verification follows the repository
checklist: `cargo fmt`, `cargo clippy --all-features -- -D warnings`, and
`cargo test`.

## Non-Goals

- Inferring whether response prose is truthful using another model.
- Retrofitting historical task outcomes.
- Changing token prices or cost reporting.
- Changing provider APIs or retry policy.
- Renaming persisted goal/task workflow statuses.
