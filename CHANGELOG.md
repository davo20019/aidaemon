# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
