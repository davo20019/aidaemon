# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.28] - 2026-03-19

### Added

- **Sliding window context management**: New `sliding_window` module with deterministic 1-line tool result summaries (e.g., "terminal: cargo test -> exit 0") for compact context preservation.
- **Adaptive sliding window**: Budget-aware `calculate_window_size()` replaces brittle `is_new_task` branching — returns min(5, pairs fitting in 30% of available token budget).
- **Age-based tool result clearing**: 3-tier system replaces binary keep/drop — current interaction keeps all, Prior 1 gets 1-line summaries, Prior 2+ drops entirely.
- **Compaction module**: Trigger detection (idle gap, file upload, window overflow), pending-pair tracking with cap of 3, LLM-based summarization prompts (initial and incremental modes).
- **Skeleton extraction**: `extract_skeleton()` utility for future compaction module use.

### Changed

- **Orphaned assistant messages preserved**: Messages with all tool_calls stripped now get "[Action completed]" content instead of being dropped, preventing dangling user messages that trigger completion compulsion.
- **Identity-critical tool results exempt from summarization**: Tool results critical to identity/context are preserved verbatim through the sliding window.

### Fixed

- **Integration test updates**: 6 tests updated for sliding window context changes — task boundary markers, critical facts format, mutation-contract nudge iterations, reflection test assertions.

## [0.9.27] - 2026-03-14

### Added

- **Task-start planning**: LLM-driven planning call before the main loop generates a structured `TaskPlan` with goal, steps, tool hints, and success criteria. Installed as a `LinearIntentPlan` and injected into the model context each iteration.
- **Budget promotion for multi-step plans**: When a captured task plan has 3+ steps and the initial budget tier is None/Small, the budget is promoted to Standard to avoid premature exhaustion.
- **Dialogue state persistence**: New `DialogueState` system tracks open questions, user responses, and resolution status across turns. Stored in SQLite via `dialogue_state` table.
- **Scheduled goals tool**: New `ScheduledGoalsTool` for managing scheduled goal operations directly.
- **`display_tool_call()` helper**: Converts raw tool call strings (e.g. `manage_memories(search)`) to human-friendly display names that survive sanitization.
- **Task plan context injection**: Each loop iteration receives the current plan as a `TaskPlanContext` system directive.
- **Terminal background command re-engagement**: After sending completion notifications, the terminal tool re-engages the agent via `handle_message()` with a synthetic follow-up containing the command output for analysis.

### Changed

- **Blocked/partial result messages include plan progress**: `build_reduce_scope_request` and `build_partial_done_blocked_request` now include plan step completion status and remaining work from the linear intent plan.
- **Activity summaries use `display_tool_call()`**: Background detach summaries in `stopping_phase.rs` now use human-friendly display names instead of raw tool call format, preventing the sanitizer from stripping them to "that".
- **Skills matching simplified**: Removed redundant matching logic in `skills/mod.rs`.
- **Telegram channel cleanup**: Simplified message handling logic.
- **Guard thresholds adjusted**: Tool execution guards updated for better multi-step task support.

### Fixed

- **Sanitizer stripping activity summaries**: Raw `tool_name(args)` format in activity summaries matched `strip_tool_name_references()` regex, replacing all entries with "that". Fixed by using `display_tool_call()`.
- **Background command notifications without agent analysis**: Completion notifications were sent to the user as raw output with no agent processing. Agent now re-engages to analyze the output.

## [0.9.26] - 2026-03-13

### Added

- **Outcome ledger**: Structured `OutcomeEntry` records for every tool call attempt, tracking success/failure, HTTP status, external mutation flag, and planned step association. Powers deterministic reconciliation at completion time.
- **Linear intent plan**: `LinearIntentPlan` with ordered `LinearIntentStep` entries allows the execution state to track multi-step workflows (e.g. "post 5 tweets"), advancing a cursor after each external success.
- **Outcome reconciliation at completion**: When external mutations fail, a two-pass reconciliation injects verified facts into the LLM context and validates the reply against the ledger. If the reply misrepresents the outcome (wrong ratios, claims unqualified success despite failures), a system-generated fallback reply is used instead.
- **`ReconciliationOverview`**: Structured summary with attempt-level or planned-step-level mode, success/failure counts, and failed step indices for precise outcome reporting.
- **Heartbeat liveness check**: Heartbeat loop now detects and logs when the event loop appears stalled.
- **`ToolSemantics.external_mutation` field**: Explicit flag for tools that mutate external state, used by the outcome ledger to distinguish reads from writes.
- **System directive `OutcomeReconciliation`**: Injects verified reconciliation facts into the LLM context during the completion phase.

### Changed

- **Completion phase handles failed external mutations before observation contract**: Failed external mutations now block completion independently and trigger reconciliation before the standard verification flow runs.
- **`http_request` response handling improved**: Better structured error extraction, HTTP status propagation to outcome ledger, and response truncation for large payloads.
- **System directive detection in completion**: Internal directives (`[SYSTEM]`, `[CONTENT FILTERED]`) are now classified as trivial tool output to prevent them from leaking as user-facing replies.
- **Terminal tool propagates external mutation semantics**: Terminal commands with external side effects are tracked in the outcome ledger.
- **Sanitization expanded**: New patterns for filtering leaked internal reconciliation markers from user-facing output.
- **Tool prelude phase supports linear intent plan installation**: Detected multi-step external workflows get a structured plan installed into execution state before the first tool call.
- **Turn context tracks explicit scheduled-run and multi-target intent patterns**: History-based scope extraction improved for batch/multi-target requests.

### Fixed

- **LLM claiming "all succeeded" when some external mutations failed**: Outcome reconciliation detects misrepresented success claims and substitutes a system-verified reply.
- **Completion loop deadlocked when verification required but tools unavailable**: Verification pending flag is now cleared when tools cannot run, with the partial result surfaced to the user.

## [0.9.25] - 2026-03-12

### Changed

- **Project scope extraction uses explicit paths only for current message**: Contextual nickname matching (e.g. "modern" resolving to `modern-plants-site`) is now skipped for the current user message to prevent false positives from common English words. Only explicit filesystem paths (`~/foo`, `/foo`, `./foo`) are extracted. History messages still use nickname matching.
- **Primary project scope prefers text order over disk existence**: For current-turn scopes, the first extracted scope (text order) is used instead of preferring scopes that exist on disk. This fixes new-project creation where a false-positive match to an existing directory would override the user's explicit path.
- **Scope-lock uses only user-explicit project scope**: `allowed_project_scope` no longer falls back to `known_project_dir` (inferred from tool results). This prevents history context pollution from locking the agent to the wrong project.
- **Project dir injection no longer updates `known_project_dir`**: `resolve_injected_working_dir` fallback to a parent directory could downgrade from the correct target to a parent, causing subsequent tool calls to latch onto unrelated projects.
- **`http_request` budget cap raised to 15**: Multi-step API workflows (posting threads, paginated APIs, auth+retry) legitimately need more sequential calls than the generic 8-call cap.
- **CI workflows bumped to `actions/checkout@v5`**: Silences the Node.js 20 deprecation warning.

### Fixed

- **New project creation scoped to wrong existing project**: When creating a new project (e.g. `ai-news-hub-2026`), common words in the prompt could match existing project directories, causing the agent to scope-lock to the wrong project and edit files there instead.
- **Budget exhaustion logging missing context**: Added session_id, task_id, iteration, and outcome to budget exhaustion and tool blocking warnings for easier debugging.

## [0.9.24] - 2026-03-12

### Added

- **`trigger_now` action for `manage_memories`**: Manually dispatch scheduled goals without waiting for the next cron tick.

### Fixed

- **Sub-sessions blocked by plain-text enforcement policy**: Spawned TaskLead/Executor agents were subject to the plain-text policy, which blocked terminal calls when the task description lacked known action verbs. Sub-sessions now always bypass this check.
- **Scheduled goals restricted to management tools only**: Cron-triggered tasks were missing terminal/write_file/browser access. TaskLead and Executor agents spawned for scheduled goals now receive full tools.
- **Validation rounds not extended by progress**: `max_validation_rounds` was the only budget dimension not extended by `extend_budget_on_progress()`, causing productive multi-step runs to hit budget exhaustion after just 3 completion-verification cycles.
- **Missing action verbs in policy signals**: Added pull, push, run, execute, fetch, merge, start, stop, compile, download, clone, migrate to the `asks_change` signal list.

## [0.9.23] - 2026-03-12

### Added

- **Non-resolving confirmation shortcut**: Bare "yes"/"ok"/"go ahead" replies to assistant questions that require a specific answer (which/what/or-choices) now get a direct follow-up requesting the actual choice instead of entering the full agent loop.
- **Progress-based budget extension**: Each successful tool execution extends the budget (LLM calls, tool calls, steps, wall-clock time), so productive multi-step runs are never artificially stopped by the initial budget ceiling.
- **Successful tool calls count as concrete work**: The stopping phase now considers any successfully completed tool call as concrete execution, preventing premature abandon when tools execute but their effects don't classify as observation/mutation (common for MCP tools).

### Changed

- **Pre-execution planning no longer charges budget**: Planning and critique passes are system-initiated quality checks — they no longer count against the agent's LLM call and validation round budget.
- **Budget tier limits raised**: None tier LLM calls 8→14, validation rounds 1→3; Small tier steps 12→16, LLM calls 10→14, tool calls 10→14, validation rounds 2→3.
- **Clarification question detection expanded**: "want me to" and "shall i" now recognized as clarification patterns for followup classification.

### Fixed

- **Productive runs stopped by tight budget**: Runs with successful tool calls were budget-exhausted even though every call succeeded. Progress-based extension ensures productive work continues.

## [0.9.22] - 2026-03-11

### Added

- **Reflection feedback loop**: New `reflection.rs` module watches for repeated tool errors with the same semantic signature, invokes the LLM as a diagnostician to analyze root cause and suggest a fix, persists the learning as an `ErrorSolution`, and verifies on the next successful call that the solution was actually applied.
- **Evidence state tracking**: New `EvidenceState` records concrete observations (file reads, command output, API responses, verification results) per turn, supports pre-execution evidence gates blocking premature writes, and provides contradiction detection.
- **Execution state machine**: New `ExecutionState` / `StepExecutionPlan` / `StepExecutionOutcome` hierarchy that selects a budget tier at turn start, tracks plan version, idempotency keys, and background-handoff flags, and can suspend its own budget for a force-text closeout.
- **Validation state**: New `ValidationState` records matched success criteria, failed checks (`BudgetExhausted`, `ScopeViolation`, `PlanRejected`, etc.), and exposes helpers for building structured `ExecutorStepResult` / `PartialResult` / blocker payloads.
- **Pre-execution planning and critique gate**: LLM-driven planning pass before the first tool call generates a `PlanState` (goal, first action, success criteria), a critique pass checks it, and both are logged as decision events.
- **Execution replay notes and diagnose surfacing**: `LearningContext` accumulates `ReplayNote` entries during a turn; `DiagnoseTool` builds an `ExecutionReplaySummary` from decision-point events.
- **Execution checkpoint in context window**: When history is trimmed, an `[SYSTEM] EXECUTION CHECKPOINT` message is injected summarising the active request, completed work, and latest evidence.
- **Scheduled run health tracking**: `ScheduledRunHealth` struct (evidence gains, tool-call count, stall/error counts) persisted in `scheduled_run_state` and used for budget auto-extension decisions.
- **New system directives**: `GlobalDailyBudgetAutoExtended`, `EvidenceGroundingRequired`, `StructuredToolResultSynthesis`, `ReflectionDiagnosis`.
- **Execution failure taxonomy**: Errors classified as `ToolContractFailure`, `ToolInvocationFailure`, `EnvironmentFailure`, or `LogicFailure` for targeted recovery coaching.
- **Executor handoff for CLI agents**: CLI agent invocations acting as task executors now persist `ExecutorHandoff` context and write structured `ExecutorStepResult` back to the task.
- **Resume checkpoint reconstructs execution snapshot**: `build_resume_checkpoint()` reads `ExecutionStateSnapshot` decision events to reconstruct last known execution state.
- **`report_blocker` structured outcome fields**: Accepts `outcome`, `exact_need`, `next_step`, `target`, `consequence_if_not_provided`, and `artifacts` for machine-readable blocker records.
- **Outbound media delivery notes**: After delivering attachments, the hub records an assistant note so the agent context reflects what was sent.
- **Owner-only fact consolidation guard**: Consolidation checks event-level `user_role` before running fact extraction; non-owner traffic excluded from owner memory.
- **Auth/integration management intent bypass**: Requests involving OAuth/token setup always force the tool loop.
- **Extended intent gate complexity enum**: Six granular complexity values for finer intent classification.

### Changed

- **Stopping phase integrates execution and validation state**: Consults `ExecutionState`, `ValidationState`, and `CompletionProgress` when budget limits are hit to decide between force-text closeout, one-time grace, or hard stop.
- **Scheduled run auto-extension uses health metrics**: Requires meaningful budget progress and blocks extension for clearly unproductive runs (stalls, non-diverse repetition, errors outpacing successes).
- **Repetitive call guard context-aware for API tools**: Redirect message for repeated identical `http_request` calls now includes the previous result hint instead of a generic "blocked" message.
- **Result learning returns semantic failure info**: `apply_result_learning` returns `ResultLearningOutcome` with `semantic_failure` and receives `ExecutionFailureKind` for targeted coaching notices.
- **Web tool budget caps raised**: `web_search` per-tool cap from 3→5, combined web cap 6→10, `web_fetch` cap 4→6.
- **Read-saturation thresholds raised**: Nudge threshold 2→4, escalation 4→7, reducing false-positive saturation nudges.
- **Scope violation check refactored to `TargetScope`**: Uses structured `StepExecutionPlan.target_scope` with typed `ToolTargetHint` matching.
- **Completion phase smarter about tool output synthesis**: Compacts HTTP/JSON outputs and filters low-signal headers before presenting structured excerpts.
- **Force-text closeout prohibits future-tense promises**: `SummarizeAndComplete` directive forces concrete results and blockers instead of "let me try…" phrasing.
- **Connected-API scope matching expanded**: Matches possessive phrases for all pronouns, adds content-delivery detection helpers.
- **Policy complexity signals expanded**: New helpers for scoped targets, mutation requests, and deployment/external writes.

### Fixed

- **Repeated API call guard falsely claimed requests were "blocked"**: Now includes the actual previous result hint in its coaching message.
- **`manage_memories` schedule actions called without `goal_id`**: Added contract-violation check returning immediate guidance instead of silent failures.
- **Session memory contaminated by non-owner traffic**: Consolidation loop checks `user_role` before fact extraction.
- **Scheduled run budget extended despite unproductive runs**: Auto-extension blocked when health metrics indicate a stuck run.
- **Force-text closeout still promised future work**: Directive now explicitly forbids future-tense phrases.
- **CLI agent delegation lost task context on failure**: Executors now persist handoff context and structured step results.
- **Execution state not recoverable after crash/restart**: Resume checkpoint reconstructs last execution state from decision events.

## [0.9.21] - 2026-03-09

### Added

- **Durable pending OAuth flows**: Pending browser auth flows are now persisted in SQLite, and recent callback outcomes are cached so reconnects/timeouts survive restarts and duplicate callbacks return a useful result.
- **Direct tool replies for interactive auth**: Tools can now attach a user-facing `direct_response`, allowing `manage_oauth connect` to finish the turn directly after browser completion instead of forcing another LLM pass.
- **Legacy message migration**: Startup now migrates the legacy `messages` table into canonical `events`, removes the old table, and clears obsolete projection settings.
- **JSON response summaries in `http_request`**: JSON API responses now include a compact key/array summary before the pretty-printed body.

### Changed

- **Agent loop terminology and structure**: Consultant-specific routing modules were consolidated into `orchestration`, `response`, `completion`, `direct_return`, and `fallthrough` phases, with policy metrics and dashboard fields renamed to match the new model.
- **OAuth/API request plumbing**: `http_request` now shares the OAuth gateway, can refresh OAuth profiles and retry once, and keeps auth scoped to same-origin redirects.
- **Verification scope extraction**: Plain-word project nicknames are only resolved as verification scopes when the user text includes explicit local/project cues, reducing accidental project-scope inference.
- **Live-work recovery guidance**: New response-analysis heuristics and system directives detect incomplete "What I tried" summaries and false "can't browse/search" denials after successful tool results, forcing the loop to use the evidence it already gathered.

### Fixed

- **OAuth timeout cleanup**: Timed-out browser auth attempts now expire their pending `state` records instead of leaving stale flows behind.
- **Final-answer regressions after successful tools**: The loop now carries successful live tool evidence forward instead of allowing misleading capability denials or pseudo tool-call text to leak into the answer.
- **Policy metrics naming drift**: Dashboard API responses and `policy_metrics` output now line up with the renamed response/orchestration counters.

## [0.9.20] - 2026-03-08

### Added

- **Completion contracts and verification targets**: `history.rs` now infers task kind, observation requirements, and verification targets (URLs, paths, project scopes) for each turn, allowing the loop to track when work still needs a final read-only confirmation step.
- **Structured tool-call semantics**: New `ToolCallSemantics`, `ToolCallOutcome`, target hints, and verification modes let tools report whether a call observed state, mutated state, or both. Added shell-command semantics classification for `run_command` and `terminal`.
- **Project-scope inheritance and enforcement**: Turn context now extracts and normalizes project scopes from user text/history, injects `_project_scope` into delegated agents, and blocks out-of-scope tool calls unless multi-project work was explicitly requested.
- **Goal failure summaries**: `fail_goal` now persists a `failure_summary`, and the agent can build failure notifications from stored summaries or failed task details.
- **Deferred-no-tool recovery metrics**: New policy counters track forced tool-call retries, deferred-action detections, fallback model switches, and terminal deferred-no-tool error markers.

### Changed

- **Completion gating tightened**: Consultant completion now blocks success responses until required verification happens, or returns an explicit "verification pending" reply when tools are unavailable in the current phase.
- **Goal completion guardrails**: `manage_goal_tasks(action="complete_goal")` now refuses summaries that admit partial progress or pending verification, while still requiring every concrete task to be resolved first.
- **Tool execution plumbing upgraded**: Tool execution now propagates structured semantics/metadata, merges fallback semantics for legacy plain-text tools, and uses those semantics to update completion progress.
- **Project path normalization**: Project-scope resolution now promotes nested directories to project roots and resolves configured project aliases consistently across follow-ups and delegated runs.
- **OAuth reauthorization flow safety**: `manage_oauth connect` now keeps the existing connection active until the new OAuth flow completes, and agent guidance prefers reconnecting over removing a working connection first.
- **Release automation hardening**: Release workflows now verify that the tag matches `Cargo.toml`, fail loudly on real crates.io publish errors, poll crates.io for the expected version, and flag release commits that reach `master` without a matching tag.

### Fixed

- **False-success replies after writes**: Requests that mutate files, URLs, or project state no longer claim completion before a matching verification step.
- **Delegation scope spoofing**: Executor-spawned `_project_scope` arguments are now overwritten with the trusted parent scope instead of honoring model-supplied values.
- **Pre-tool deferral loops**: Repeated "I'll do it" text-only replies before the first tool call now trigger a hard tool-call directive, a fallback-model retry window, and clearer blocker text when no tools are available.
- **OAuth disconnect footgun**: Removing an OAuth connection now requires explicit `confirm_disconnect=true` plus approval, reducing accidental token deletion during scope refreshes.
- **Machine-specific CI test path**: The CLI-agent prompt-recovery test now uses a temporary working directory instead of a hardcoded local filesystem path, so Linux and macOS CI runners pass consistently.

## [0.9.19] - 2026-03-08

### Added

- **Connected API intent classification**: New `intent_routing.rs` module with `classify_connected_api_intent()` that detects runtime capability validation ("Are you connected to GitHub?"), read actions ("List my open GitHub issues"), and write actions ("Create a GitHub issue") across 30+ external service targets. Forces tool-loop routing for connected API work.
- **Connected API tool pinning**: `ensure_connected_api_tools_exposed()` pins `manage_oauth` and `http_request` tools to the top of the tool list when connected API intent is detected, ensuring they survive policy filtering.
- **`http_request` hardening**: Auth-header detection (`header_name_looks_like_auth`, `header_value_looks_like_auth`), embedded JSON query param recovery, session-scoped approval cache, and credential-in-URL stripping. Major rewrite of request validation and sanitization.
- **External action timeout ack**: `finalize_external_action_timeout_ack()` in `llm_phase.rs` handles graceful completion when external actions (OAuth flows, HTTP requests) exceed the LLM call timeout.
- **Daemon runtime context in `check_environment`**: Shows working dir, config path, env file path, and other daemon runtime info.
- **OAuth callback URL normalization**: `normalize_callback_url()` accepts both base URLs and full callback URLs, preventing double `/oauth/callback` suffixes.
- **Generic API auth onboarding**: New `manage_http_auth` tool creates, inspects, verifies, and removes manual HTTP auth profiles for bearer/header/basic/OAuth1a APIs, binds secrets from keychain or `.env`, and refreshes runtime auth state without a restart.
- **Custom OAuth provider onboarding**: `manage_oauth` can now register, describe, and remove custom OAuth 2.0 PKCE, authorization-code, and client-credentials providers at runtime, making OAuth setup possible from a clean install without manual config edits.
- **Automatic API guide generation**: `manage_skills` gains `learn_api`, which fetches an OpenAPI/Swagger URL or documentation page and turns it into a reusable API guide skill tied to the user's connected/authenticated API workflow.
- **Deterministic API onboarding**: New `manage_api` tool orchestrates connect + learn + verify in one flow by composing `manage_oauth`, `manage_http_auth`, `manage_skills`, and `http_request`.
- **Stronger API learning ingestion**: `manage_skills learn_api` now crawls multiple docs pages on the same host, discovers linked OpenAPI/Swagger specs from docs, bundles remote OpenAPI `$ref` documents, and captures GraphQL patterns from docs-only sources.
- **Auto-derived verification probes**: `manage_api` can now derive a safe verification probe from learned OpenAPI specs, and can reuse GraphQL introspection as the probe for GraphQL APIs when no explicit `verify_url` was provided.
- **GraphQL schema introspection learning**: `manage_skills learn_api` can now introspect GraphQL endpoints with the live auth profile instead of relying only on docs text heuristics.

### Changed

- **`http_request`, `manage_http_auth`, and `manage_oauth` added to `ESSENTIAL_TOOLS`**: Always available regardless of risk/policy filtering.
- **Connected API tool exposure expanded again**: Connected API turns now keep `manage_api` exposed alongside auth/request/learning tools so the agent can choose a deterministic end-to-end onboarding path.
- **Connected API intent overrides intent gate**: `infer_intent_gate()` now forces `needs_tools=true` when connected API intent is detected, bypassing consultant text-only routing.
- **Intent routing verb refinements**: Removed "open" from connected API write verbs (false positive on "open issues" meaning "not-closed"), removed "read" from read verbs (false positive on "read the docs").
- **Runtime capability validation expanded**: Added "are you connected to" and "are you hooked up to" as direct phrase triggers.
- **Generic API path always available**: `http_request` is now registered even before any profiles exist so the agent can onboard and use new APIs from a clean install.
- **OAuth path always available**: `manage_oauth` is now registered even when OAuth was not pre-enabled in config, so users can add and connect built-in or custom OAuth providers from zero.
- **Connected API tool exposure expanded**: Connected API turns now keep `manage_skills` exposed alongside auth/request tools so the agent can learn an API before using it.

### Fixed

- **Clippy too-many-arguments**: Added `#[allow]` on `finalize_external_action_timeout_ack`.
- **Connected API false positives**: "Write a GitHub Actions workflow" no longer classified as a write action; "Read the GitHub Actions docs" no longer classified as a read action.

## [0.9.17] - 2026-03-07

### Added

- **Cross-provider failover chain**: `[[provider.fallbacks]]` in config.toml defines ordered alternate providers with independent API keys, models, and base URLs. On primary provider failure, the agent cascades through local model fallbacks first, then alternate providers and their model chains.
- **`add_failover_provider` config action**: `ConfigManagerTool` gains a new action to append failover providers at runtime, with keychain storage for API keys and full preset support.
- **Scheduled run per-check budget**: New `scheduled_run_state` SQLite table and `GoalRunBudgetState` in `GoalTokenRegistry` track token usage per scheduled run independently from the daily budget, with persistence across task-lead/executor restarts.
- **`SystemDirective` enum**: Replaces raw `String` system messages with typed, structured directives (`RouteFailsafeActive`, `TaskTokenBudgetWarning`, `ForceTextToolLimitReached`, `EditStallWriteFileHint`, etc.) for cleaner agent loop control.
- **`ToolResultNotice` enum**: Structured post-tool-result notices replacing ad-hoc string messages.
- **`MessageAnnotation` system**: Structured annotations for conversation messages (`EntireSystemNotice`, `AppendedDiagnostic`, `WrappedUntrustedExternalData`, etc.) with inference from legacy marker text and primary content extraction.
- **Project root detection**: `fs_utils.rs` gains `PROJECT_ROOT_MARKERS`, `find_nearest_project_root()`, and `normalize_project_scope_path()` for promoting subdirectory paths to their project root.
- **Path alias support**: `extract_project_dir_hint_with_aliases()` resolves user-defined path aliases (e.g., "projects" → "~/projects") when detecting project directories from user text.
- **`ProviderError::recovery_failed_message()`**: Terminal error messages that don't promise retries when all recovery attempts have been exhausted.
- **Terminal bridge dynamic bot merging**: `merge_daemon_bot_tokens()` combines configured and dynamic Telegram bots for daemon bootstrap auth.
- **Skipped replay status messages**: `build_skipped_stdout_replay_status_message()` and review-stream equivalents inform users when buffered output is skipped on reconnection.

### Changed

- **Scheduled goal budget defaults raised**: Continuous scheduled goals now default to 100K per-check / 500K daily (up from 50K/200K), with a migration to bump existing goals at the old defaults.
- **Scheduled goal iteration limits removed**: Scheduled goals no longer enforce hard/soft iteration caps or warn-at thresholds; budget control is entirely token-based.
- **Scheduled goal budget extensions**: Relaxed productivity check for scheduled goals (1 tool call or 1 evidence gain + 0 stalls), with higher extension limits (12 vs 3) and hard token cap (20M vs 2M).
- **Cascade fallback improvements**: `cascade_fallback()` now iterates all provider-local fallback models, then all failover provider targets and their model chains, with `ProviderError` propagation at each stage.
- **`ProviderError` is now `Clone`**: Enables error propagation through fallback chains.
- **Sanitization preserves code blocks**: `strip_internal_control_markers()` and `strip_diagnostic_blocks()` now split content at fenced code block boundaries and only strip markers from prose segments.
- **Config secret resolution recursive**: `resolve_secrets()` recurses into `provider.fallbacks[]` with indexed keychain key prefixes (`provider_fallback_0_api_key`, etc.).
- **History window further refined**: Message build phase checks that the current user message is the LAST user message (not just any match) to prevent false boundary detection on retried prompts.

### Fixed

- **Clippy warnings**: Elided unnecessary lifetimes in `get_failover_array()` and removed needless `Ok()?` wrap in `normalize_failover_array_mut()`.

## [0.9.16] - 2026-03-06

### Added

- **LLM hard wall-clock timeout**: All providers (OpenAI-compatible, Anthropic, Google GenAI) now enforce a 360-second `tokio::time::timeout` safety net around API calls, preventing indefinite hangs when servers trickle data past reqwest's built-in timeout.
- **`ProviderError::timeout_msg()`**: New constructor for custom timeout error messages.
- **Background process lifecycle modes**: Three distinct modes for background processes — task-owned (killed on task-end), background with notifier (survives task-end, notifier delivers result), and detached (survives everything). Controlled via `notifier_active` field on `RunningProcess`.
- **Duplicate prompt deduplication**: Message build phase now removes old user messages with identical content to the current prompt (and their assistant responses), preventing the model from thinking a retried task was already completed.
- **Force-text fast-path in consultant completion**: When the model is in force-text mode after 3+ successful tool calls, all tool-requiring guards (file-recheck, deferred-action, tool-required) are bypassed to prevent deadlocks.

### Changed

- **Background handoff responses**: Stopping phase now includes an activity summary of tool calls performed before the background task started, not just the technical "moved to background" message.
- **Background notification promise**: `build_background_detach_ack()` and system messages now only promise completion notifications when the notifier is actually active, based on tool result content rather than assuming all background processes have notifications.
- **Telegram consecutive Thinking suppression**: Status updates now track `last_was_thinking` and suppress consecutive "Thinking..." messages, sending a typing indicator instead.
- **History window scaling**: Message build phase scales the history event limit based on iteration count (up to 120) so long-running tasks don't lose early tool calls from the current task.
- **`latest_non_system_tool_result` boundary fix**: Now stops at user message boundaries to avoid leaking tool results from previous interactions.
- **File re-check guard in force-text**: Clears the guard instead of blocking when tools are unavailable or force-text is active.

### Fixed

- **Background process notification broken**: `on_task_end` was killing all task-owned background processes AND suppressing their notifications. Processes with active notifiers are now disowned instead of killed, preserving the completion notification promise.
- **Read_file output recovery in multi-tool sessions**: When the latest tool is `read_file` but multiple tools ran, skips raw file dump recovery in favor of the activity summary.
- **Integration test time sensitivity**: Scheduler test changed from "today at 11:09pm" to "tomorrow at 11:09pm" to avoid flaky failures near midnight.

## [0.9.15] - 2026-03-05

### Added

- **CLI agent prompt alias support**: `cli_agent` tool now accepts `mission`, `task`, `command`, and `description` as alternative parameter names for `prompt`, recovering gracefully when the LLM emits non-standard argument names.
- **Duplicate send_file loop breaker**: When a duplicate `send_file` is suppressed, the agent now forces text-only mode and injects a system nudge to prevent re-emission loops.
- **Send file completion reply**: Dedicated `send_file_completion_reply()` provides a consistent closeout message after file delivery, used across force-text, low-signal recovery, and stall-exit paths.

### Changed

- **Heartbeat schedule error logging**: Silent `let _ =` discards on goal/schedule updates in `heartbeat.rs` replaced with `warn!()` logging to surface schedule persistence failures.
- **Path validation hardening**: `validate_path()` now normalizes `..` components before the traversal check, and `is_sensitive_path()` uses component-level matching to avoid false positives on substrings (e.g., `my_environment.txt` no longer matches `.env`).
- **Browser SSRF protection**: `navigate` action in the browser tool now validates URLs against internal/private IP ranges before navigation.
- **Telegram file download timeout**: Added 60-second timeout to Telegram file download requests.
- **Telegram callback validation**: Callback query handler now rejects empty callback IDs.
- **Tool argument summary improvements**: `summarize_tool_args` for `cli_agent` checks `task`/`mission`/`description`/`command` fields; `send_file` summary reads `file_path` instead of `path`.

### Fixed

- **`is_trivial_tool_output` gap**: "Duplicate send_file suppressed:" output is now correctly classified as trivial, preventing it from being surfaced as a task result.
- **`latest_non_system_tool_result` returns tool name**: Callers can now distinguish `send_file` results from other tool outputs for targeted completion handling.

## [0.9.14] - 2026-03-05

### Added

- **Unified loop control evaluator**: `LoopControlInputs::evaluate()` centralizes hard iteration cap, task timeout, pre-tool deferral, and post-tool stall checks with context-aware stall limits (transient errors, empty responses, deferred-no-tool recovery get extra room).
- **Security injection detection**: Expanded prompt injection defense to detect social engineering attacks (fake "system override", "authorized security audit", `/etc/passwd` reads, API key extraction attempts) with dedicated system reminder and assistant prefill.
- **StallMode classification**: Stall events now carry a mode (`Default`, `DeferredNoTool`, `Transient`, `EmptyResponse`) for better diagnostics and adaptive limits.

### Changed

- **UTF-8 safe string truncation**: Added `floor_char_boundary()` helper in `utils.rs` and applied safe byte-position slicing across 15+ files (events/payloads, channels/formatting, tools/read_file, tools/edit_file, tools/write_file, tools/search_files, tools/service_status, tools/browser, tools/cli_agent, tools/http_request, tools/diagnose, mcp/client, mcp/mod, memory/context_window, plans/generation, memory/procedures) to prevent panics on multi-byte UTF-8 characters.
- **LLM token-limit truncation recovery**: OpenAI-compatible provider now detects `finish_reason=length` and signals the agent loop, which injects a retry nudge asking the model to continue from where it left off.
- **Recall guardrails improvements**: Tightened policy recall filtering for cleaner memory context injection.
- **Execution policy tool budgets**: Adjusted tool budget limits across policy profiles (Cheap/Balanced/Strong) for better alignment with force-text escalation thresholds.
- **Read-saturation detection**: Consecutive `read_file` nudge at 2 calls, escalation at 4 in post-loop processing to prevent unproductive read loops.
- **Context window management**: Improved token accounting and message truncation in context window module.
- **Agent loop message building**: Refined message construction phase for better tool call chain handling.
- **System prompt updates**: Enhanced system prompt construction in agent runtime.

### Fixed

- **UTF-8 panic on multi-byte characters**: Truncation operations across the codebase could panic when slicing through multi-byte UTF-8 code points (e.g., emoji, CJK characters). Comprehensive fix ensures all string truncation respects character boundaries.
- **Stale budget extension logic**: Simplified and consolidated goal daily budget extension handling in LLM phase to reduce code complexity and prevent edge cases.

## [0.9.13] - 2026-03-03

### Added

- **Telegram command menu auto-registration**: Bot commands are now registered with Telegram's `setMyCommands` API at startup, populating the `/` command menu automatically.
- **Command definition registry**: Single source of truth for all command definitions (`CommandDef` struct with name, description, usage, and platform category) in `channels::commands`. Adding a new command to the registry automatically surfaces it in Telegram's menu and `/help` output.
- **Platform-scoped command lists**: `telegram_commands()`, `slack_commands()`, and `discord_commands()` filter the registry by `CommandCategory` (Core, Restart, Connect, Terminal) so each platform only shows relevant commands.

### Changed

- **`build_help_text()` refactored**: Now accepts `&[CommandDef]` instead of boolean flags, generating the command list dynamically from the registry.

## [0.9.12] - 2026-03-03

### Added

- **Telegram webhook low-latency mode**: Opt-in webhook support for Telegram bots, replacing polling with direct HTTPS push for significantly lower message latency. Per-bot or global defaults configuration with auto-port assignment.
- **`aidaemon setup low-latency` command**: Interactive setup wizard for configuring webhook mode with Cloudflare tunnel integration, multi-bot hostname assignment, port conflict resolution, and config backup.
- **Shared command dispatcher**: Unified `/model`, `/models`, `/auto`, `/reload`, `/tasks`, `/cancel`, `/clear`, `/cost` handling across Telegram, Discord, and Slack channels via new `channels::commands` module.
- **Channel token validation helpers**: `channels::connect` module with `validate_telegram_token()`, `validate_discord_token()`, and `validate_slack_tokens()` for dynamic bot setup flows.
- **Terminal agent permission aliases**: `normalize_terminal_agent_permission_aliases()` rewrites `--allow-dangerously-skip-permissions` to `--dangerously-skip-permissions` for Claude agent CLI compatibility.

### Changed

- **Terminal-lite extracted to reusable module**: `TerminalLiteManager` moved from Telegram-only internals to `src/terminal_lite.rs` for potential reuse by other channels.
- **CLI agent flag discovery extracted**: Agent flag discovery, caching, pagination, and defaults management moved to `src/cli_agent_flags.rs`.
- **Terminal bridge exponential backoff**: Replaced fixed 3-second reconnect delay with exponential backoff (1s → 30s) with jitter and 60-second stability reset.
- **Webhook configuration in `config.toml.example`**: New `[telegram.webhook]` and `[telegram_webhook_defaults]` sections with documentation.

### Fixed

- **Terminal bridge stale session on fresh agent start**: When the Mini App's old session expires and requests a new agent, the daemon now tears down the stale session instead of reattaching to it with a dead agent process. Previously, `bootstrapped_agent` remained `true` from the old session, preventing the new agent command from being sent to the shell, resulting in a blank terminal.

## [0.9.11] - 2026-03-02

### Added

- **Native terminal handoff ("Continue on Computer")**: Seamlessly transfer an `/agent` session between Telegram Mini App and your native terminal. One-time, time-limited (5 min TTL) handoff codes allow secure session resumption across devices. New `aidaemon attach <code>` CLI command connects to the running daemon's local attach endpoint.
- **Local attach endpoint**: Terminal bridge opens a loopback TCP listener on startup, writing connection details to `~/.aidaemon-terminal/attach-endpoint.json`. Enables CLI commands (`attach`, `start`, `share`) to communicate with the running daemon without going through the broker.
- **`aidaemon agent share` command**: Generate a Telegram resume code from the command line for sharing active terminal sessions.
- **`aidaemon agent start` command**: Launch terminal agents (`codex`, `claude`, `gemini`, `opencode`) through the bridge with working directory and flag pass-through support.
- **CLI agent shortcuts**: `aidaemon codex`, `aidaemon claude`, `aidaemon gemini`, `aidaemon opencode` as thin aliases for `agent start` with optional `[cwd] [-- flags...]`.
- **Telegram Mini App handoff commands**: `/agent share`, `/agent resume <code>`, and "Continue on Computer" inline button on `/agent open`.
- **Web App data action parsing**: Structured action types from the Mini App (`agent_message.v1`, `open_on_computer.v1`, `continue_on_computer.v1`) with flexible field-name support.
- **Terminal bridge hot-start**: Auto-starts terminal bridge on owner auto-claim without requiring `/restart`.

### Changed

- **Outbound message queue**: Replaced direct write-per-event with a two-priority queue (High for control, Low for bulk stdout). Flushes up to 24 frames per tick, preventing large PTY bursts from starving control messages.
- **Biased select loop**: WS read arm is now prioritized first, ensuring incoming broker messages are never starved by shell output.
- **Smarter re-attach replay**: Decides whether to replay buffered stdout based on frame count, byte size, and interactive content detection. Skips replay for large/interactive sessions to avoid terminal rendering artifacts.
- **PTY UTF-8 streaming**: Replaced `from_utf8_lossy` with a carry-buffer approach that flushes only complete UTF-8 sequences. C1 control bytes normalized to ESC-prefixed equivalents.
- **Duplicate bridge startup guard**: `AtomicBool` prevents racing hot-start and normal startup from spawning two bridge tasks.

### Fixed

- **Session map lock scope** (Discord, Slack, Telegram): Write lock guard is now dropped before `save_session_channel` async call, preventing potential lock ordering issues across await points.
- **Clippy warnings**: Removed needless `return` statements in CLI argument handling.

## [0.9.10] - 2026-03-01

### Added

- **Terminal bridge daemon**: New WebSocket-based daemon (`terminal_bridge.rs`) connects to the terminal.aidaemon.ai broker, enabling Telegram Mini App terminal sessions from anywhere. Secure P-256 ECDH key exchange with AES-256-GCM encrypted message tunnel. Supports PTY-based interactive shell sessions with real-time output streaming, sequence-numbered frames for replay, and session isolation.
- **Terminal Lite (chat-based shell)**: `/terminal lite [agent] [working_dir]` starts an interactive shell session directly in Telegram chat without the Mini App. Supports `cd`, command prefix validation, TUI app detection, and 90-second execution timeout.
- **Agent launcher commands**: `/agent [agent] [working_dir]` launches the full Mini App terminal. `/agent flags [agent]` discovers available CLI flags with 24-hour caching. `/agent defaults [action] [agent] [flags...]` persists default flags per agent per chat.
- **Code review workflow**: Structured review payloads with git diff capture, file change context collection, multiple review profiles, and streaming review output. Max 220K chars context with smart filtering of binary/generated files.
- **File upload support**: Chunked uploads (32 KiB per chunk) with TTL-based cleanup, concurrent upload tracking (max 4 pending per session), and mime type validation.
- **New Telegram commands**: `/models` lists available models with active marker, `/auto` re-enables automatic model routing, `/reload` hot-reloads config with `.toml.bak` auto-restore, `/restart` performs graceful daemon restart, `/tasks` lists active tasks with elapsed time, `/cancel <task-id>` cancels a running task.
- **Expandable message formatting**: Long Telegram replies (>1800 chars) use `<blockquote expandable>` HTML tags for collapsible display, with fallback to chunked delivery.
- **`[terminal]` configuration section**: 8 new config fields (`web_app_url`, `bridge_enabled`, `daemon_ws_url`, `daemon_connect_token`, `allow_static_token_fallback`, `daemon_user_id`, `daemon_device_id`, `daemon_shell`) with environment variable overrides and keychain support.
- **`terminal-bridge` feature flag**: Enabled by default. Activates the full WebSocket bridge and secure daemon pairing system. New dependencies: `p256`, `aes-gcm`, `hkdf`, `portable-pty`.

### Changed

- **Help text includes terminal commands**: Telegram help now shows `/agent` and `/terminal lite` documentation. Discord and Slack omit terminal-specific commands.
- **CLI agent output format**: Updated example configs from `--output-format stream-json` to `--output-format json`.

### Fixed

- **Clippy warnings resolved**: Fixed `div_ceil` reimplementation, manual char comparison, manual `RangeInclusive::contains`, and suppressed `too_many_arguments` for internal async helpers.

## [0.9.9] - 2026-02-28

### Added

- **Multi-segment schedule parsing**: Users can create multiple scheduled goals in a single message (e.g., "1) every day at 9am check server health. 2) in 2 hours send status report"). The scheduler splits, parses, and confirms all segments as a batch with a single confirmation prompt.
- **Named-month date scheduling**: Schedule expressions now support calendar dates like "on March 5th at 3pm" or "March 15" with automatic year rollover for past dates.
- **Specific day-of-week scheduling**: Support for "every Monday and Friday at 3pm" and similar multi-day expressions, parsed into correct cron day-of-week fields.
- **Task-scoped terminal process lifecycle**: Background terminal processes are now task-owned by default — auto-killed when the owning agent task ends. New `detach=true` parameter opts into long-lived execution that survives task boundaries.
- **`on_task_end` tool lifecycle hook**: New `Tool` trait method called after `TaskEnd` events, enabling tools to clean up task-scoped resources. Terminal tool uses this for automatic background process cleanup.
- **Duplicate background command suppression**: Within the same goal/task scope, re-running an equivalent command that is already tracked in the background returns a reference to the existing process instead of spawning a duplicate.
- **Internal maintenance intent guard**: Scheduling requests for built-in maintenance operations (memory consolidation, embeddings, decay) are intercepted with a message explaining these run automatically.
- **Schedule-only description detection**: When a user's message is purely a schedule expression with no task description, the system detects this and prompts for a task description rather than creating an empty goal.

### Changed

- **Schedule detection refactored to `cron_utils`**: All schedule extraction regex patterns moved from `intent_routing.rs` to `cron_utils.rs` as `LazyLock` statics, improving startup performance and enabling reuse across the codebase.
- **Task descriptions auto-cleaned**: Schedule phrases and filler prefixes ("remind me to", "schedule a task to") are stripped from goal descriptions in both the fast-path and tool-path, producing cleaner goal text (e.g., "Send release notes" instead of "in 2 hours remind me to send release notes").
- **Goal confirmation timeout extended**: Telegram approval timeout increased from 5 minutes to 30 minutes, preventing race conditions when users confirm near the boundary.
- **Daemon commands require explicit `detach=true`**: Daemonization primitives (`nohup`, `&`, `disown`) are now blocked unless `detach=true` is set, preventing accidental long-lived orphaned processes.
- **Detached execution blocked in trusted sessions**: Scheduled/trusted sessions cannot use `detach=true`, preventing unattended creation of long-lived background processes.

### Fixed

- **Batch schedule confirmation/cancellation**: Multi-segment schedule requests now confirm or cancel all goals atomically instead of handling only the first one.
- **Empty command validation**: Terminal tool now rejects empty/whitespace-only commands with a clear error instead of passing them to the shell.

## [0.9.8] - 2026-02-23

### Added

- **ToolCapabilities metadata for all tools**: Every tool now declares structured capabilities (`read_only`, `external_side_effect`, `needs_approval`, `idempotent`, `high_impact_write`) giving the agent loop metadata for smarter execution decisions.
- **Schema lint test suite**: Four compile-time tests enforce schema hygiene — `additionalProperties: false` in all schemas, explicit `capabilities()` on all tools, no silent argument parse error swallowing, and schema size limits (6,500 chars per tool, 90,000 total).
- **Action-verb guard for intent gate**: The consultant intent gate previously let "simple/knowledge" classifications override `needs_tools`, causing tool-requiring queries like "Find all TODO comments" to short-circuit into fabricated answers. A new guard scans for 18 action verbs and blocks the override when detected.
- **G2 stall pattern diagnostic**: Warns when the agent completes with zero tool calls but produces deferred-action text (e.g., "I'll search for TODOs..."), catching promise-without-execution patterns.
- **Fresh-context isolation marker**: When message history has no prior assistant/tool messages (e.g., after `/clear`), a system message tells the LLM this is a fresh conversation, preventing stale tool-call pattern drift from pinned memories.
- **Daemon command early return in terminal**: Commands detected as daemon/background launches (`nohup`, `&`, `disown`) now return immediately after timeout with a success message and pid, instead of entering an infinite background tracking loop.
- **Large heredoc soft-block in terminal**: Terminal commands containing `<<` (heredoc) exceeding 500 characters are soft-blocked with a message redirecting the LLM to `write_file`.
- **`limit` parameter for ManageMemoriesTool**: `list`, `search`, `list_goals`, `list_scheduled`, and `list_scheduled_matching` actions accept an optional `limit` parameter (max 200) with "showing X of Y" counts and truncation notices.

### Changed

- **`additionalProperties: false` added to all tool schemas**: Prevents the LLM from inventing nonexistent parameters across all tools.
- **Stricter conditional `required` fields via `anyOf`**: `TerminalTool` and `CliAgentTool` schemas now enforce action-specific required fields (e.g., `command` for `run`, `pid` for `check`/`kill`).
- **Strong model profile tool surface capped at 28**: Previously unlimited, now capped to prevent unbounded tool surfaces that confuse the LLM.
- **Argument parsing errors propagated, not swallowed**: Multiple tools previously used `.unwrap_or(json!({}))` which silently accepted malformed JSON. Now properly returns errors. Affected: `BrowserTool`, `ReadChannelHistoryTool`, `HttpRequestTool`, `ManageOAuthTool`, `UseSkillTool`, `WebFetchTool`, `WebSearchTool`.
- **`web_fetch` max_chars clamped**: Now clamped to `[1, 50_000]` (previously unbounded, default 20,000).
- **`web_search` max_results clamped**: Now clamped to `[1, 10]` (previously unbounded, default 5).
- **Terminal tool description warns against heredoc/echo patterns**: Directs the LLM to use `write_file` instead.

### Fixed

- **Knowledge-complexity override bypassing tool execution**: Queries like "Find all TODO comments" were classified as simple/knowledge, causing fabricated answers without running any tools. The action-verb guard now prevents this.
- **Daemon commands causing infinite background tracking**: Commands spawning background processes would timeout and enter a tracking loop that never finished. Now detected and returned immediately.
- **BrowserTool missing required parameter validation**: The `action` parameter was silently accepting missing values via `.unwrap_or("")`; now returns a proper error.

## [0.9.7] - 2026-02-21

### Added

- **Signature-based semantic failure tracking**: Tool failure lockout now tracks error *signatures* (normalized fingerprints) rather than raw failure counts. Different errors from the same tool no longer pile up toward the lockout limit — only repeated identical failure patterns trigger lockout. Error signatures normalize file paths, line numbers, PIDs, and exit codes so the same root cause is detected even when surface details change.
- **Error coaching in tool failure feedback**: First-time tool failures now extract the key error line (via pattern-priority heuristics) and quote it back to the LLM with explicit coaching to try a different approach. Subsequent failures include the specific error context.
- **User-facing error explanations in graceful stop messages**: When the agent stalls or times out, the graceful response now includes an "Issues encountered" section listing actual errors (deduplicated, recent-error priority, resolved ones marked) plus a "Blocked capabilities" section showing which tool categories hit the lockout limit.
- **"Tool Locked Out" stall classification**: `classify_stall()` returns a new category with actionable user guidance when any tool has hit its semantic failure limit.
- **File lookup miss as transient failure**: File-not-found errors from file-oriented tools are classified as transient rather than semantic, so missing-file errors during project exploration do not consume the semantic lockout budget or trigger cooldown periods.
- **Provider `extra_headers` configuration**: All three provider backends (OpenAI-compatible, Anthropic, Google GenAI) support an `extra_headers` map in `config.toml` for injecting custom HTTP headers on every API request, with `"keychain"` resolution for secrets.
- **Anthropic `max_tokens` configuration**: The Anthropic provider accepts an optional `max_tokens` setting (default: 16384, previously hardcoded to 4096).
- **Anthropic dynamic model listing**: `list_models()` now queries the `/models` API endpoint instead of returning a hardcoded list, with fallback to known models if the API call fails.
- **MCP protocol version negotiation**: MCP client initialization tries multiple protocol versions (`2025-06-18`, `2025-03-26`, `2024-11-05`) with automatic fallback — if a version fails, the server is restarted with the next version.
- **MCP rich content block rendering**: MCP tool results now render non-text content blocks (images, resources, structured content) as descriptive placeholders instead of raw JSON. The `isError` flag is respected and converted to an error.
- **MCP automatic server restart on transport failure**: When an MCP tool call fails with a transport-level error (broken pipe, closed stdout, connection reset, timeout), the system automatically restarts the server and retries the call once. Application-level errors are not retried.
- **MCP server enable/disable**: Dynamic MCP servers can be enabled and disabled without deletion. `manage_mcp` tool gains `enable` and `disable` actions; `list` shows enabled/disabled status and source type (static vs. dynamic).
- **Skill enable/disable via filesystem markers**: Skills can be disabled/enabled using `.disabled` marker files. Disabled skills appear in listings with status metadata but are excluded from matching and activation.
- **Skill management approval flow**: `manage_skills` now requires user approval for `add` (from URL), `install` (from registry), and `update` actions.
- **Skill body sanitization on activation**: `UseSkillTool` runs skill body content through `sanitize_external_content()` before returning it to the LLM, filtering prompt injection attempts.
- **Skill resource auto-registration**: `SkillResourcesTool` automatically registers a skill's directory path with the `FileSystemResolver` on first access, so directory-based skills loaded at runtime work without prior manual registration.
- **Raw tool-call token sanitization**: The sanitization pipeline now strips leaked LLM tool-calling protocol tokens (`<|tool_calls_section_begin|>`, `functions.terminal:0 {...}`, etc.) from user-facing output.

### Changed

- **ExecutionPolicy tool budgets increased**: Cheap 6→15, Balanced 12→35, Strong 20→60 — previous values caused hard stops before the force-text escalation system could engage.
- **Empty-response recovery preserves parent context**: When the LLM returns an empty response and the agent retries, the immediate parent user+assistant exchange is now preserved (truncated to 800 chars each), giving the LLM enough conversational context to produce a meaningful reply.
- **Background task lead deduplication**: When executor task results are already sent inline during goal execution, the completion notification sends only a brief signal instead of repeating the full results summary.
- **Telegram session ID stability**: Session IDs are derived from a stable namespace locked for the process lifetime, preventing mid-run session ID drift when the bot username is resolved asynchronously.
- **Telegram typing indicator improvements**: Interval reduced from 4s to 3s; typing action is re-sent immediately after status/thinking messages (Telegram clears typing indicators when a message is sent).
- **Tool name trimming across all providers**: All three providers now trim whitespace from tool names returned by LLMs, preventing spurious "unknown tool" errors from models that occasionally add leading/trailing spaces.
- **Anthropic and Google provider constructors accept `base_url`**: Both providers now accept optional custom `base_url` parameters, enabling use with proxy endpoints and alternative API-compatible services.
- **Skill frontmatter parser rewritten**: Line-based parsing replaces `find("---")` which could split on `---` sequences embedded in description values. Body content can now contain horizontal rules without corruption.
- **Skill cache uses tree fingerprinting**: `SkillCache` computes a recursive filesystem fingerprint instead of relying on top-level directory mtime, correctly detecting changes to nested `SKILL.md` files inside directory-based skills.
- **Skill and MCP listings enriched with status**: `manage_skills list` and `manage_mcp list` now display enabled/disabled counts and per-item status tags.
- **Dependencies trimmed**: Removed `thiserror` and `insta` crate dependencies.

### Fixed

- **Goals table schema mismatch**: `extract_goal()` used `let id: i64` on a `TEXT PRIMARY KEY` column (UUIDs), causing sqlx panics. Fixed to `let id: String`. INSERT was also missing the `id` and `session_id` columns.
- **Stale goal detection type mismatch**: `detect_stale_goals()` also used `let id: i64` for the TEXT id column; fixed to `let id: String`.

## [0.9.6] - 2026-02-20

### Added

- **Inline goal confirmation buttons**: Scheduled goals now show Confirm ✅ / Cancel ❌ buttons (Telegram) instead of reusing the command approval flow. New `ApprovalKind` enum distinguishes goal confirmations from command approvals. `Channel` trait gains `request_goal_confirmation()` with auto-confirm default for channels without button support.
- **Deterministic tool contract violations**: Pre-execution guard blocks `scheduled_goal_runs` calls missing `goal_id` before they reach the tool, with coaching messages redirecting the LLM to the correct tool (`remember_fact` for fact storage, `manage_memories(list_scheduled)` for ID lookup).
- **Result learning for goal_id errors**: Post-execution coaching when `scheduled_goal_runs`, `manage_memories`, `goal_trace`, or `tool_trace` fail with missing `goal_id`, with special detection for fact-storage requests mistakenly routed to goal tools.
- **`ManageMemoriesTool` button-based confirmation**: `create_scheduled_goal` action uses inline approval buttons (via approval channel) when available, with automatic activation on confirm and cleanup (cancel goal + delete schedules) on deny.

### Changed

- **`scheduled_goal_runs` schema requires `goal_id`**: Previously optional, now required in the tool schema to prevent underspecified calls.
- **Tool descriptions clarify scope**: `scheduled_goal_runs` and `remember_fact` descriptions explicitly state their purpose boundaries — `scheduled_goal_runs` is not for fact storage, `remember_fact` is the tool for "learn/remember/save" requests.
- **System prompt tool routing**: Added explicit routing row for "User says learn/remember/save these" → `remember_fact` (not `manage_memories` or `scheduled_goal_runs`). Added fact-storage guidance to memory rules.
- **First-interaction message**: Changed from "I learn from our conversations" to "I adapt my communication style over time based on our conversations" for clarity.

## [0.9.5] - 2026-02-19

### Added

- **Skill promotion quality gates**: Pre-LLM substance check (minimum 2 steps, 8 words) rejects trivial procedures before spending an LLM call. Post-generation `skill_is_valuable()` filter rejects skills with generic triggers (yes/no/ok/hello), insufficient body content, or missing description.
- **`MalformedResponse` provider error kind**: Reason-aware recovery distinguishes parse errors (transient — exponential backoff + cascade fallback) from shape errors (likely deterministic — single retry, fail fast). Per-provider/model/reason breakdown metrics via `LlmPayloadInvalidMetric`.
- **Input token estimation metrics**: Tracks tool schema overhead per LLM call with high-share (>=35%) and high-absolute (>=1500 tokens) threshold counters exposed via policy metrics tool and dashboard.
- **Session-scoped cancel-all**: `cancel_scheduled` with `goal_id="all"` cancels all non-protected scheduled goals scoped to the calling session only.
- **Scheduled goal dedup**: `create_scheduled_goal` detects duplicate schedules by canonicalizing descriptions (stripping execution wrappers, normalizing whitespace) and comparing cron expressions.
- **Internal execution context guard**: Prevents schedule creation from within internal scheduled-task execution (`sub-*` sessions or `internal` channel visibility).
- **Schedule auto-confirm**: `AllowSession`/`AllowAlways` approval responses are remembered per session, auto-confirming subsequent schedule creations without re-prompting.
- **Canonical filename collision detection**: Skill draft approval and filesystem persist check for collisions using `sanitize_skill_filename()` normalization, catching variants like "send-resume" vs "send resume".
- **Explicit approve flag for draft review**: `manage_skills review` now requires `approve: true` or `approve: false` — omitting the flag returns guidance instead of silently skipping.

### Changed

- **Skill promotion LLM prompt**: Updated to instruct the model to skip trivially simple/generic procedures and conversational filler behaviors.
- **`skill_draft_exists_for_procedure`**: Now checks all draft statuses (pending, approved, dismissed) to prevent re-promotion of previously dismissed procedures.
- **Scheduled goal descriptions**: `build_scheduled_goal_description()` normalizes composed goal text by extracting original request and follow-up parts, preventing description corruption from multi-turn wrapping.
- **System prompt scheduling guidance**: Replaced proactive scheduling suggestions with explicit-request-only guidance — only create exactly what was requested.
- **`ProviderKind` made `Copy`**: Simple enum no longer requires `.clone()` calls throughout the codebase.
- **Provider response body errors**: `resp.text().await` failures in all three providers (OpenAI, Anthropic, Google) now classified as `Network` errors instead of propagating as unclassified `anyhow` errors.
- **OpenAI `choices[0]` and `message` extraction**: Now returns `MalformedResponse(Shape)` instead of generic `anyhow` errors, enabling structured recovery.

## [0.9.4] - 2026-02-18

### Added

- **Semantic fact dedup**: `upsert_fact()` detects synonym keys (e.g., `editor` vs `preferred_editor`) via embedding similarity (0.85 threshold) with token-overlap guards to prevent false merges.
- **Mid-session episode creation**: Long-running sessions (20+ events since last episode) get episodes captured before context rotates out, preventing permanent context loss.
- **Multiple episodes per session**: Removed unique constraint on `session_id` in episodes table to support incremental episode capture.
- **Batch fact storage**: `remember_fact` tool accepts a `facts` array for storing multiple facts in one call.
- **Fuzzy forget matching**: Forget action uses canonical, case-insensitive, and substring matching with cross-category fallback.
- **Reply sanitization pipeline**: Strips model identity leaks, internal tool name references, and diagnostic/system blocks from user-facing replies.
- **Default + fallback model routing**: New `default_model` + `fallback_models` config replaces the old fast/primary/smart tier system (legacy keys still work).
- **Deterministic pre-routing**: Schedule/cancel/goal fast-paths handled before first LLM call, removing the consultant classification pass.
- **Cloudflare AI Gateway support**: Optional `gateway_token` for OpenAI-compatible providers with automatic gateway detection.
- **Moonshot and MiniMax provider presets**: New OpenAI-compatible provider presets.
- **Comprehensive memory tests**: New test suite covering fact dedup, episode lifecycle, and memory retrieval edge cases.
- **Scheduler flaw tests**: New integration tests for scheduler edge cases.

### Changed

- **Episode retrieval threshold**: Lowered from 0.5 to 0.3, matching the fact retrieval threshold for better recall.
- **Agent proactivity default**: `asks_before_acting` defaults to `false` for new user profiles — agent only confirms destructive actions.
- **Telegram message splitting**: Long replies split at 4096-char boundary instead of truncating.
- **Tools available from iteration 1**: All LLM calls have tools available, removing the tool-free consultant pass.

### Fixed

- **Stale fact duplicates (BUG-8)**: Semantically identical facts with different keys no longer accumulate as duplicates.
- **Episode recall on context rotation (BUG-9)**: Long sessions no longer lose conversational context when the 20-message window rotates.
- **Path normalization**: `validate_path()` now normalizes `.` components correctly.
- **Provider JSON error messages**: Anthropic and Google providers wrap parse errors with proper `ProviderError` context.

## [0.9.3] - 2026-02-17

### Added

- **Transient failure classification**: Tool failure detection distinguishes transient errors (rate limits, timeouts, network) from semantic errors, triggering cooldowns instead of outright blocking.
- **Tool result head+tail compression**: Large tool results preserve both beginning and end of output while dropping the middle, improving visibility into critical information.
- **Query-aware fact selection**: System prompt uses intelligent fact scoring with freshness boosting to surface more relevant facts during owner DM conversations.
- **Tool loop prompt optimization**: Compact `ToolLoopPromptStyle` on subsequent iterations reduces prompt size while maintaining model context.
- **Internal control marker stripping**: Sanitization removes agent-internal markers (`[SYSTEM]`, `[DIAGNOSTIC]`, etc.) from final replies to prevent leaking control flow to users.
- **Adaptive tool limits for goals**: Tool call limits increase from 30 to 55 when actively working on a goal for more exploration budget.

### Changed

- **Stall detection**: Thresholds dynamically adjust based on failure patterns — transient failures and empty responses get +2 iterations before stalling.
- **Result learning refactored**: Error handling pipeline restructured to use `classify_tool_result_failure()` with structured pattern matching (HTTP status codes, JSON error fields, exit codes).
- **Policy signal word-boundary matching**: Risk estimation uses word-boundary matching instead of substring matching for action keywords, reducing false positives.
- **Tool failure tracking**: Semantic and transient failures tracked separately with distinct counters and blocking behavior.
- **Error solution injection**: Diagnostic hints from error memory moved to first-failure only, reducing noise.
- **README simplified**: Reduced from 633 to 99 lines; detailed docs moved to external documentation site.

### Fixed

- **Tool failure categorization**: Comprehensive pattern detection for HTTP statuses, JSON error payloads, exit codes, and transient error strings.
- **Final reply sanitization**: Internal control markers never leak into user-facing responses.
- **Cascade fallback**: Returns response for current call only without persistent model downgrade.

## [0.9.2] - 2026-02-16

### Added

- **Provider ChatOptions**: New `ChatOptions` struct with `ResponseMode` (Text/JsonObject/JsonSchema) and `ToolChoiceMode` (Auto/None/Required/Specific) for per-call LLM behavior control.
- **Intent gate JSON schema enforcement**: Consultant phase requests structured intent analysis via `ResponseMode::JsonSchema` with strict validation and `ToolChoiceMode::None`.
- **Deferred no-tool recovery**: When model defers work without attempting tools, subsequent calls force `tool_choice=required` to break the deferral loop.
- **Terminal background process completion cache**: Background processes retain final output in 128-entry LRU cache (10-min TTL), allowing result retrieval after automatic cleanup.
- **Terminal hub integration**: Terminal tool holds weak reference to ChannelHub for direct delivery of background process progress/completion events.
- **Spawn tool fallback notifications**: SpawnAgentTool gains queued notification path when hub delivery fails for background sub-agent completion.
- **Goal daily token budget extension approval**: When goal budget is exhausted mid-execution, owner is prompted to approve extension (up to hard cap) instead of immediate termination.
- **Goal task result excerpts**: Summarizes up to 3 recent completed task results for goal completion notifications instead of only the final task.
- **Latest tool output excerpt fallback**: Agent extracts latest non-system tool output for completion replies when LLM produces empty response after successful tool execution.
- **Path aliases configuration**: New optional `[path_aliases]` config section for user-friendly path shortcuts (e.g., `projects = ["~/projects"]`).
- **Local execution deterministic intent override**: Intent gate forces `needs_tools=true` for explicit local execution keywords and local version queries.

### Changed

- **Consultant completion recovery**: When reply is empty after tool execution, agent recovers with latest tool output excerpt instead of generic "Done" message.
- **Needs-tools enforcement**: When consultant marks turn as needing tools and model returns text-only, text is suppressed and model retries with forced tool calls.
- **Deterministic background acknowledgment**: Background detach messages are deterministically enforced rather than relying on model compliance.
- **Spawn tool background mode**: Now requires at least one of hub OR state store (notification queue) instead of hub exclusively.

### Fixed

- **Format string compilation error**: Fixed unescaped path alias example in system prompt format macro.
- **Clippy warnings**: Fixed derivable Default impls, collapsible if, `is_multiple_of()` usage.

## [0.9.1] - 2026-02-16

### Added

- **Turn context resolution**: Followup mode classification (new/followup/clarification) with scope carryover detection and multi-project scope awareness.
- **Graceful partial-stall responses**: When the agent stalls after meaningful progress (3+ successful tool calls), it now acknowledges progress before stopping instead of a generic stall message.
- **Project directory scope constraints**: Tool calls are validated against the resolved project scope to prevent cross-project file operations.
- **Hard-block destructive commands**: `find -delete` and `rm -rf` on broad/sensitive paths are now blocked before the approval flow, even in yolo mode.
- **Context integrity metrics**: New policy counters for context bleed prevention, mismatch preflight drops, followup mode overrides, and cross-scope blocking.
- **Stall classification tests**: Tests for tool policy block, edit target drift, generic value filtering, and provider server error detection.
- **npm command rejection guidance**: `run_command` now shows allowed npm prefixes and suggests `terminal` for installs.

### Changed

- **Stopping phase**: Meaningful progress detection (total_successful_tool_calls >= 3 or evidence_gain_count >= 2) triggers graceful exit instead of hard stall.
- **Project directory hints**: Seeded from turn context's primary project scope rather than just user text extraction.
- **System prompt guidance**: Prefer `search_files`/`project_inspect` over raw terminal for discovery; added recursive grep guidance.

### Fixed

- **Compilation error**: Fixed `borrow of moved value` in tool_defs.rs capabilities entry.
- **Format string error**: Fixed unescaped parentheses in system prompt format macro.

## [0.9.0] - 2026-02-16

### Added

- **Consultant system**: Fast-path decision making for intent classification via separate classifier + executor LLM calls. Includes intent gate parsing, policy signal detection, and orchestration phases.
- **Agent module decomposition**: Refactored monolithic `agent.rs` (~3,300 lines) into 65 files across 9 subdirectories (`loop/`, `consultant/`, `intent/`, `policy/`, `runtime/`, `tools/`), with explicit phase-based control flow (bootstrap, message build, LLM, tool execution, stopping).
- **SharedLlmRuntime**: Centralized LLM provider + router abstraction (`Arc<RwLock>`) enabling runtime provider reloads without recreating dependent components.
- **Library crate**: Moved module declarations to `src/lib.rs` for programmatic usage of aidaemon as a library.
- **Policy metrics and autotuning**: Lock-free `AtomicU64` counters for policy decisions with dynamic uncertainty threshold adjustment based on failure ratios.
- **Recall guardrails**: Personal memory privacy rules with tool filtering — blocks browser/external tool escalation during personal recall turns.
- **Route health diagnostics**: Intent gate route health monitoring in the diagnose tool — detects empty direct replies, sustained clarification rate spikes, and routing anomalies.
- **Critical fact signal detection**: Memory summarization now detects identity/relationship statements ("my name is", "wife", "daughter") and preserves them through context window compression.
- **Batch project inspection**: `project_inspect` tool accepts `paths` array (max 12) for multi-directory inspection in a single call.
- **Lightweight interjection filtering**: Grace-period-aware spam filter for rapid-fire greetings within 120 seconds of daemon restart.
- **Duplicate message suppression**: Per-session deduplication cache (10-second window) in ChannelHub for identical heartbeat/status messages.
- **Goal cancellation support**: Unified `/cancel`, `/stop`, `/abort` commands across Slack, Telegram, and Discord for both tasks and active goals.
- **Post-install migration**: `install.sh` now runs `aidaemon migrate` automatically after installation to handle database schema upgrades.
- **CLI smoke tests**: `assert_cmd`-based tests for `--help` and `--version` flags.
- **CHANGELOG.md**: Added changelog for release tracking.

### Changed

- **Legacy messages to events migration**: Canonical event stream is now the single source of truth. Messages table is migrated to events and dropped. Migration is idempotent with completion tracking via settings key.
- **CLI agent concurrency**: Replaced `HashSet` locks with `WorkingDirClaim` structs and Jaccard similarity checking (>50% blocks duplicate prompts). Added `Semaphore` for true concurrent limit enforcement.
- **Router simplification**: Removed `classify_query()` and `ClassificationResult` — router now only provides model selection; classification logic moved to agent layer.
- **Memory manager initialization**: Takes `SharedLlmRuntime` instead of separate provider + fast_model. Removed 30-second event-to-messages projection loop.
- **Retention policy**: Migrated from legacy messages table to canonical events table for cleanup.
- **CI formatting gate**: `cargo fmt --check` is now a hard gate (removed `continue-on-error`).
- **Deprecated policy config**: Removed `classify_retirement_enabled`, `classify_retirement_window_days`, `classify_retirement_max_divergence` from config.

### Fixed

- **UTF-8 panic**: Fixed multi-byte character boundary panic in `compress_tool_result()` — now uses char boundaries instead of raw byte slicing.
- **RwLock poison handling**: Improved recovery with `.unwrap_or_else(|poisoned| poisoned.into_inner())` across channel implementations.
- **Integration test stability**: Fixed `test_personal_recall_challenge_inherits_previous_turn_context` to use the new consultant classifier + executor flow.

### Security

- **Test data sanitization**: Removed all personal information (names, file paths, project references) from test files. Replaced with generic test fixtures.

## [0.8.0] - 2025-12-15

### Added

- Goal system unification with personal goals and multi-schedule support
- Browser tool enhancements

## [0.7.7] - 2025-12-01

### Added

- Inline tool failure diagnostics
- In-session error learning
- Error solution deduplication

## [0.7.6] - 2025-11-15

### Fixed

- Rust 1.93 clippy lints
- Event store enhancements
- Post-task improvements
- Telegram fixes

## [0.7.5] - 2025-11-01

### Fixed

- CLI agent heartbeat fix
- Empty response recovery
- Module decomposition
- Token budgets
