# Agent runtime architecture: semantic model, deterministic kernel

Status: active migration target (2026-07-31)

## Decision

AIdaemon must not use English phrase lists as an authority for coreference,
target selection, tool success, mutation completion, or replay safety.

The model owns interpretation:

- what the user means in the context of the conversation;
- which prior object a natural-language reference denotes;
- task decomposition and tool selection;
- whether a final response answers the request.

The deterministic kernel owns protocol facts:

- JSON/schema validation and exact enum dispatch;
- authorization, approvals, sandboxing, and security policy;
- stable resource identity and content integrity;
- declared tool effects and typed outcome receipts;
- idempotency, replay reconciliation, and state transitions;
- verification of exact paths, URLs, resource IDs, and effect obligations.

The boundary is simple: a model may propose an interpretation or operation,
but only structured evidence can authorize it or prove it happened.

This follows the current agent-loop pattern described by OpenAI: a model run
selects tools, the application executes and validates them, tool results resume
the same run, sessions preserve state, and approval interruptions preserve a
resumable run state rather than becoming a fabricated final answer.

- <https://developers.openai.com/api/docs/guides/agents>
- <https://developers.openai.com/api/docs/guides/agents/running-agents>
- <https://developers.openai.com/api/docs/guides/agents/guardrails-approvals>
- <https://developers.openai.com/api/docs/guides/function-calling>
- <https://github.com/modelcontextprotocol/specification/blob/main/schema/2025-06-18/schema.ts>

### Layered control topology

AIdaemon deliberately uses three control shapes instead of forcing every task
through one graph runtime:

```text
natural-language request
          |
          v
 semantic task contract
    +-----+----------------+
    |                      |
    v                      v
one turn             durable complex work
typed phase cycle    dependency DAG
    |                      |
    +---- model/tool loop -+

ongoing mandate -> leased mandate state machine
                    ACT | WAIT | ASK | STOP
                     |
                     +-- ACT may create DAG work whose nodes use the same loop
```

The per-turn runtime is an explicit typed cyclic state machine: phases advance,
finish, or restart with a typed reason. The model remains free to choose and
revise its approach inside that cycle. Complex durable goals use the existing
versioned, cycle-checked task dependency DAG for ordering and bounded
parallelism. Long-lived mandates use their separately persisted lifecycle and
decision state machine. A general graph framework is intentionally unnecessary:
the Rust type system, event log, SQLite transactions, leases, and receipts are
the authoritative control and recovery surfaces.

Simple requests do not acquire graph overhead. A richer model-proposed plan DAG
is warranted only when a finite task truly needs branching or parallelism; it
must reuse the durable task graph rather than create a second orchestration
system inside the turn loop.

## Target runtime

```text
conversation + registered resources
                 |
                 v
       semantic task assessment
       (typed TaskContractV1)
                 |
                 v
         model proposes tool call
                 |
                 v
   deterministic operation validator
   schema | auth | scope | approval | idempotency
                 |
                 v
             tool adapter
                 |
                 v
       typed ToolReceiptV1 + resources
                 |
                 v
       durable event log / run state
                 |
                 +---- replay or resume ----+
                 |                          |
                 v                          |
        model sees exact result ------------+
                 |
                 v
       completion from effect evidence
```

### 1. Semantic task contract

The existing task assessment becomes the only semantic classifier that can
refine task kind, required observations, required mutation effects, negative
constraints, and task shape. Its output must be versioned, schema-validated,
confidence-gated, and grounded to the current request for negative constraints.

Lexical classifiers may temporarily supply fallback hints when assessment is
unavailable. A fallback may make execution more conservative, but it must not:

- bind a pronoun to an object;
- invent a target;
- authorize a side effect;
- claim an effect or delivery occurred;
- override a typed receipt;
- turn an interruption into success or failure.

### 2. Resource registry instead of guessed antecedents

Every attachment and tool-produced artifact receives an opaque `resource_id`,
provenance, MIME type, exact path when applicable, and SHA-256 digest. The model
sees these handles in conversation context and can pass `resource_id` to a tool.
Execution resolves the handle within the current session and verifies the
digest immediately before use.

Natural language such as “edit it” is therefore handled in two stages:

1. the model resolves “it” using conversation history and proposes
   `resource_id=res_...`;
2. the kernel verifies that exact handle, session, resource kind, and digest.

No pronoun list participates in either proof.

### 3. Producer-owned tool semantics and receipts

Each mixed-action tool classifies exact action enum values in its own adapter.
Generic runtime code does not infer read/write behavior from the tool name,
description, HTTP-looking words, or result prose. Unknown non-read-only actions
default conservatively to mutation.

Every completed call emits `ToolReceiptV1`. The receipt carries:

- typed outcome status;
- evidence source;
- declared observation/mutation effects;
- idempotency key;
- process, HTTP, timeout, detach, notification, and transport facts when known.

Specific completion obligations require matching specific effects. A legacy
`unspecified` mutation can no longer prove source modification, deployment, or
delivery.

### 4. Operation identity and replay

Mutation idempotency is derived from stable operation identity:

```text
execution_id + tool_name + canonical_arguments_hash
```

It does not depend on an iteration number or provider-generated tool-call ID.
Before I/O, the runtime checks the event log:

- completed receipt: replay the receipt without repeating I/O;
- unresolved prior claim: fail closed and request reconciliation;
- no prior claim: append the call claim and execute.

### 5. Durable interactions and interruptions

All approval-producing tools pass through `ApprovalBroker`. The broker persists
a versioned `InteractionRequested` event before channel delivery and a matching
`InteractionResolved` event before forwarding the response. Pending approvals
are a projection of the event log, not just an in-memory oneshot receiver.

Process interruption is `TaskStatus::Interrupted` with a partial outcome. It is
not recorded as failure. Resume reuses durable execution identity and reconciles
the latest typed receipt before another operation can run.

### 6. Evidence-based completion

Completion is computed from the finalized task contract and the effect ledger:

- observation obligations require successful observation receipts;
- mutation obligations require matching typed mutation effects;
- delivery requires an external-delivery receipt;
- post-mutation verification requires a later observation of the exact target;
- backgrounded is not succeeded;
- model prose is never stronger evidence than a receipt.

Unstructured response analysis remains a compatibility layer for providers that
cannot yet emit structured final-output metadata. It must not override the
effect ledger.

### 7. Bounded autonomous mandates

Long-lived autonomy is represented as a durable control loop, not a frequent
prompt and not a standing trusted session:

```text
owner Mandate vN
  objective | constraints | success/stop conditions | authority envelope
  owner-pinned strategy snapshot
             |
             v
leased review -> observations/beliefs -> ACT | WAIT | ASK | STOP
                                      ACT |
                                          v
                              one revocable Intention
                                          |
                                          v
                               ordinary tasks + receipts
```

This combines established patterns without giving the model protocol authority:

- the MAPE-K autonomic-computing loop supplies the recurring
  monitor/analyze/plan/execute shape and durable knowledge;
- BDI-style separation keeps an owner's desired objective (`Mandate`) distinct
  from current evidence (`belief_snapshot`) and the agent's one-cycle commitment
  (`Intention`);
- lease/fencing and optimistic-version patterns provide single-writer review
  admission and immediate revocation of stale decisions;
- capability-style least authority becomes an exact observation/action-tool,
  mutation-effect, target, and action-count envelope, checked by the
  deterministic dispatcher rather than the prompt.

A mandate may also pin one installed skill as an immutable, content-addressed
strategy snapshot. The snapshot informs how the worker reasons, but never adds
tools, targets, effects, quota, or any other authority. Later edits to the
installed skill do not silently change an already-confirmed delegation.

The model may decide that an action is worthwhile, but the kernel grants no
mutation until that run has committed ACT against the current mandate version.
The grant is bound to the mandate, decision cycle, canonical tool arguments, and
typed semantics by a SHA-256 digest. After all local guards pass, the dispatcher
durably reserves the operation, then atomically claims that exact reservation
at the final I/O boundary while revalidating the grant. A claimed reservation
consumes its cycle, rolling-24-hour, and cooldown slots even if the adapter
fails or the result is ambiguous; a reservation that never reaches the claim is
recorded as `never_dispatched` and does not consume the rolling quota. Only a
strict typed success receipt can satisfy ACT. Unknown tools and mutation effects
fail closed.
Observation tools are scoped by the same explicit allowlist, preventing an
output mandate from reading unrelated private data before it acts.

Changing, pausing, or cancelling a mandate invalidates work that has not crossed
the final I/O boundary; ordinary scheduled-goal trust cannot satisfy this
boundary. Revocation cannot undo an external request that was already issued,
so unresolved reserved actions enter reconciliation instead of being replayed.

WAIT is a first-class successful outcome: it advances the adaptive review clock
without manufacturing work or notifying the owner. ASK creates a typed
`awaiting_answer` suspension and emits one concrete question; answering that
question cannot accidentally resolve an ambiguous side effect. Lease loss,
review failure, and unresolved mutations use distinct typed suspensions and
require their matching recovery action. STOP completes the mandate only when it
names the exact owner-authored success or stop condition (or a typed safety
termination); success/stop completion must cite successful, same-run typed
receipts. A failed cycle is retried within its owner-approved timing bounds,
while ambiguous external effects require an explicit, audited reconciliation.

V1 deliberately exposes a narrow autonomous execution surface:

- a personal goal is not authority and is never auto-converted into a mandate;
  creation or expansion requires explicit owner confirmation in a verified
  private control channel;
- only directly governed, deterministic adapters can be delegated. MCP,
  terminal/shell, filesystem/project, browser/computer-use, scheduler/health,
  and other opaque nested-action tools are denied;
- scoped HTTP authority binds the exact canonical URL scope, exact
  authentication-profile resource identifier, and (for every authenticated
  request) exact stable remote account ID. The adapter verifies the
  model-supplied `account_id` against the profile's configured `user_id` before
  network I/O. Autonomous requests do not follow redirects, refresh or replay
  OAuth after a 401, accept argument/query/authority drift, or treat a 202/3xx
  response as completion evidence. An expired or rejected OAuth credential
  therefore becomes ASK/manual reauthorization, never an autonomous refresh.
  Rotating credentials or changing a
  profile's `user_id` can rebind its remote identity: pause or revoke every
  mandate that references that profile, make the change, and explicitly
  reconfirm its profile and account scopes before resuming. V1 persists the
  exact profile name and stable account ID, but not an immutable credential
  generation, so this operational pause/reconfirmation is a required boundary;
- mandate workers use a fixed isolated prompt. They do not inherit a private
  session persona, user memory, unpinned skills, project instructions,
  checkpoints, result spill, or global learning/reflection channels;
- canonical mandate turns carry durable execution provenance. Completion-time
  fact extraction and summaries, periodic memory/episode/activity jobs, and
  delayed event consolidation all reject that provenance before global writes,
  embeddings, or auxiliary model calls;
- continuity is limited to a byte-bounded, typed view of the same mandate's
  recent outcomes, receipts, quota state, and evidence-linked learning notes.
  Every learning note must cite successful structured receipts from that same
  mandate and remains advisory: it cannot broaden the authority envelope.
  Generated rationale, questions, task prose, tool bodies, errors, and
  cross-mandate history are excluded and cannot become authority;
- success means satisfying the owner-authored criteria with typed evidence. It
  does not promise engagement, follower counts, availability of a third-party
  API, or any other external outcome.

These restrictions are capability boundaries, not prompt suggestions. Broader
adapters can be added only after they provide owner-pinned semantics, exact
target identity, per-action metering, strict receipts, and replay-safe behavior.

Design lineage:

- J. O. Kephart and D. M. Chess, *The Vision of Autonomic Computing*,
  <https://doi.org/10.1109/MC.2003.1160055>.
- A. S. Rao and M. P. Georgeff, *BDI Agents: From Theory to Practice*,
  ICMAS 1995, <https://www.math.pku.edu.cn/teachers/linzq/teaching/stm/references/BDI%20Agents%20From%20Theory%20to%20Practice.pdf>.
- C. G. Gray and D. R. Cheriton, *Leases: An Efficient Fault-Tolerant
  Mechanism for Distributed File Cache Consistency*,
  <https://doi.org/10.1145/74850.74870>.
- J. H. Saltzer and M. D. Schroeder, *The Protection of Information in
  Computer Systems*, <https://doi.org/10.1109/PROC.1975.9939>.

## Natural-language word-list inventory

This inventory covers production lists that interpret user/model language. It
intentionally separates them from security and protocol lists, which should
remain deterministic.

### Language interpretation: migration debt

| Area | Locations and list families | Current role | Target disposition |
|---|---|---|---|
| Follow-up/coreference | `agent/runtime/followup.rs`: clarifying, style, strong-follow-up, source/status/shared-context, explanation, deictic, complaint, response and imperative lists; `agent/loop/compaction.rs`: `REFERENTIAL_PHRASES` | Context shaping and follow-up mode | Advisory context shaping only. Resource binding and blocking by pronoun were removed. |
| Intent and routing | `agent/intent/intent_routing.rs`: scheduling verbs, file verbs, local execution phrases, connected-service targets/actions/resources, artifact actions/types, complexity markers; `agent/intent/keywords.rs`: memory and scheduling lists; `agent/intent/relational_prefilter.rs`: relational nouns/verbs | Schedule, memory, tool exposure, connected API and fallback route classification | Replace authority with versioned semantic task contract. Keep only exact synthetic/protocol commands as deterministic. |
| Initial planning | `agent/loop/bootstrap/task_planning.rs`: compound create/execute/fix/verify groups, control commands, planning-worthy markers | Decides whether to run semantic assessment and pads weak-model plans | Use model trust tier and typed task shape; retain exact control commands only. |
| Policy scoring | `agent/policy/policy_signals.rs`: target, mutation, deploy/write, schedule, environment, expected-output, rollback, risk, feedback, correction and request-shape lists | Pre-assessment risk/uncertainty and feedback telemetry | Advisory routing/telemetry only. The side-effect target gate is now based on structured arguments, not pronouns. |
| Completion contract | `agent/runtime/completion_contract.rs`: question prefixes, change/run/mutation/artifact cues, report-only phrases, verification, delivery, diagnosis, live-state and negative-mutation phrases | English fallback contract and negative constraints | Semantic task contract is primary. Retain grounded negative constraints as conservative defense-in-depth until structured coverage is measured. |
| Model-response compatibility | `agent/response_analysis.rs`: defer phrases, past-action verbs, filler, knowledge-only verbs, acknowledgments; `agent/loop/answer_grounding.rs`: denial phrases; `agent/loop/completion_checks.rs`: success/failure/partial-result claims; `agent/loop/tool_prelude_phase.rs`: approval/proposal/action phrases | Detects unstructured promises, claims, denials, and malformed final answers | Replace with typed final-output/result fields. Never override receipts. |
| Goal/plan inference | `agent/goal_dispatch.rs`: auto-send blocks and challenge phrases; `plans/detection.rs`: sequential, verify, action and imperative lists; `plans/store.rs`: delivery verbs/nouns | Goal shape, checklist detection and delivery attribution | Structured task/plan steps and declared output resources. |
| Memory retrieval | `agent/loop/orchestration/memory_scope.rs`: generic phrases and stopwords; `agent/policy/recall_guardrails.rs`: recall/store and relationship cues; `memory/context_window.rs`, `memory/neighborhood.rs`; `state/sqlite/facts.rs`: `EXHAUSTIVE_QUERY_MARKERS`; `tools/memory.rs`: `PERSONA_PATTERNS` | Retrieval scope and exhaustive-query hints | Embedding/graph retrieval plus structured recall intent; keep lists only as recall hints. |
| Miscellaneous language fallbacks | `agent/agent_helpers.rs`: local-action and non-path phrases; `agent/runtime/dialogue_state.rs`; `agent/runtime/project_scope.rs`; `agent/runtime/spawn.rs`; `agent/loop/loop_utils.rs`; `agent/loop/stopping_phase.rs`; `agent/loop/validation_state.rs`; `agent/loop/tool_execution/reflection.rs`; `agent/loop/tool_execution/result_learning.rs` | Formatting, correction, scope and recovery hints | Audit individually; hints may affect prompts or telemetry, never execution facts. |

### Snapshot-wide source catalog

The table above groups the behavior. For completeness, the 2026-07-31 audit
also checked every production Rust source containing a named string slice,
local string array, phrase-matching helper call, or literal membership test.
Tests and display-only copy are excluded. The user/model-language catalog is:

- `agent/agent_helpers.rs`, `agent/goal_dispatch.rs`;
- `agent/intent/intent_routing.rs`, `agent/intent/keywords.rs`,
  `agent/intent/relational_prefilter.rs`;
- `agent/loop/answer_grounding.rs`, `agent/loop/compaction.rs`,
  `agent/loop/completion_checks.rs`, `agent/loop/execution_state.rs`,
  `agent/loop/loop_utils.rs`, `agent/loop/main_loop.rs`,
  `agent/loop/stopping_phase.rs`, `agent/loop/tool_prelude_phase.rs`, and
  `agent/loop/validation_state.rs`;
- `agent/loop/bootstrap/shortcuts.rs`,
  `agent/loop/bootstrap/task_planning.rs`,
  `agent/loop/orchestration/memory_scope.rs`,
  `agent/loop/orchestration/routes.rs`,
  `agent/loop/tool_execution/reflection.rs`, and
  `agent/loop/tool_execution/result_learning.rs`;
- `agent/policy/policy_signals.rs`, `agent/policy/recall_guardrails.rs`,
  `agent/response_analysis.rs`;
- `agent/runtime/completion_contract.rs`,
  `agent/runtime/dialogue_state.rs`, `agent/runtime/followup.rs`,
  `agent/runtime/post_task.rs`, `agent/runtime/project_scope.rs`, and
  `agent/runtime/spawn.rs`;
- `agent/tools/tool_defs.rs`, `memory/context_window.rs`,
  `memory/neighborhood.rs`, `memory/skill_promotion.rs`;
- `plans/detection.rs`, `plans/store.rs`, `state/sqlite/facts.rs`,
  `tools/memory.rs`, and `tools/spawn.rs`.

Operational-text and compatibility lists—not conversational intent—also occur
in `agent/correction_sandbox.rs`, `agent/policy/trust_tier.rs`,
`agent/loop/stopping_helpers.rs`, `agent/loop/tool_execution/post_loop.rs`,
`memory/manager.rs`, `providers/error.rs`, `providers/openai_compatible.rs`,
`providers/openai_chatgpt.rs`, `tools/background_deliverable.rs`,
`tools/browser/backend.rs`, `tools/cli_agent.rs`, `tools/computer_use/capability.rs`,
`tools/project_inspect.rs`, `tools/service_status.rs`, and `tools/terminal.rs`.
These recognize provider protocols, exact tool names, subprocess output, model
families, or file/command syntax; they do not bind a natural-language target.

The audit can be repeated with searches for static `&[&str]` declarations,
local string arrays, and calls to `contains_keyword_as_words`,
`text_contains_any_phrase`, and related membership helpers. Any newly added
match site must be classified into one of the two tables in this document.

### Protocol, syntax, and security lists: intentionally deterministic

| Area | Locations | Why they stay deterministic |
|---|---|---|
| Command safety and shell semantics | `tools/command_risk.rs`, `tools/command_semantics.rs`, `tools/daemon_guard.rs`, `tools/verification.rs`, `tools/run_command.rs` | These parse a command DSL, enforce allow/deny policy, or classify exact executable behavior—not conversational meaning. Unknowns fail conservatively. |
| Filesystem and delivery safety | `tools/fs_utils.rs`, `tools/file_delivery.rs` | Sensitive path segments, project markers and blocked delivery paths are security boundaries. |
| Browser/computer consequential actions | `tools/browser/policy.rs`, `tools/computer_use/policy.rs` | Defense-in-depth around consequential UI operations. Final authorization still uses exact action schemas and approval state. |
| Network/MCP safety | `tools/web_fetch.rs`, `mcp/mod.rs`, `tools/manage_mcp.rs` | SSRF host blocks, suspicious input/output patterns and allowed executable protocols. |
| Secret and control-text sanitization | `tools/sanitize.rs` | Prevents credential leakage and control-marker injection. |
| Provider compatibility | `providers/error.rs`, `providers/openai_compatible.rs`, `tools/browser/backend.rs`, `tools/computer_use/capability.rs` | Matches provider error/capability protocol text; it does not interpret user intent. |
| Exact action enums | mixed-action tools under `tools/manage_*`, `config_manager.rs`, `health_probe.rs`, `scheduled_goal_runs.rs` | Exact schema enum matching is protocol dispatch. Unknown actions remain mutations. |

Configured MCP servers are registered through an owner-controlled config or an
approval-gated install. Their structured MCP ToolAnnotations map to typed
capabilities using the specification defaults; names and descriptions are not
parsed. An absent `readOnlyHint` therefore remains a mutation, while an explicit
read-only annotation becomes observation semantics.

## Migration gates

The migration is complete when all of the following hold:

1. no user-language list binds or selects a resource;
2. no generic runtime code infers tool effects from names or prose;
3. every mutating adapter emits a typed receipt;
4. every artifact crosses tool boundaries by resource ID plus digest;
5. every approval is reconstructable from durable events;
6. retrying the same operation cannot repeat its side effect;
7. completion tests assert effects and receipts, not success-sounding strings;
8. language-list fallbacks are measured in shadow telemetry and can be deleted
   without changing execution authorization or outcome truth.
