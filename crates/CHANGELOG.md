# Changelog: aidaemon SDK crates

All SDK crates (`crates/aidaemon-*`) share one version and are released together.
`aidaemon-node-protocol` and `aidaemon-node-sdk` are versioned separately.
The daemon (`aidaemon`) has its own version and changelog (`../CHANGELOG.md`) and pins these crates exactly.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). While 0.x, breaking
changes land only in a new minor version; the public-contract tier
(`aidaemon-core`, `aidaemon-sdk`, the `channel` module of `aidaemon-channels`, the
`aidaemon-memory` traits) is checked with `cargo-semver-checks`; the rest has no
stability promise.

## [Unreleased]

### Security
- `aidaemon-sdk`: `preset` without a workspace no longer runs a tool that may change things but declares no access without anyone seeing it: the approval policy asks the owner first (a `#[tool(write)] fn save(path, ..)` with no `access` could write `~/.bashrc` unchecked). With a workspace, a read-only tool that declares no paths is now refused like a writer (it could read `~/.ssh`). New `Preset::path_free_tool` marks tools that touch no files (business APIs); the tools a `Preset::memory` system gives are marked automatically; new `Preset::undeclared_reads`. An explicit `Preset::undeclared_writes(Allow)` still runs undeclared writers without asking.
- `aidaemon-policies`: `ScopeGuard::path_free_tool`, `ApprovalPolicy::ask_undeclared_writes` and `ApprovalPolicy::path_free_tool`.
- `aidaemon-tools`: the shell refuses a command when its backend cannot enforce a restriction the shell was given (`Shell::deny_read`, or an explicit `Shell::allow_network(false)`), instead of running it unrestricted on `LocalSandbox`. New `Sandbox::enforces_deny_read` and `Sandbox::enforces_network_off` (default `false`). `DockerSandbox` now covers denied paths inside the workspace. Bubblewrap and seatbelt deny reads of `HOME_SECRET_PATHS` under `$HOME` by default (`default_deny_read(..)` replaces the list; `SeatbeltSandbox` is no longer a unit struct).

### Fixed
- `aidaemon-policies`: `ContextWindow` pins the user's latest message as well as the first, so a second question is never dropped behind big tool rounds.

### Removed
- `aidaemon-policies`: `ReceiptCompletionCheck` and `Requirement`. Measured live (Decision 4: a policy that shows no gain is removed before it becomes public API), it gained nothing: 34/40 with and without on the first eight scenarios, and 5/5 with and without on `completion_retry`, the scenario built for it (N=5, `claude-haiku-5-5`), at slightly more turns. It is not a safety boundary. A receipt-based check remains a few lines of `on_stop` (see `aidaemon-core/DESIGN.md`); `completion_retry` stays as a plain-kernel regression.

### Changed
- `aidaemon-policies`: `ContextWindow` elides in coarse steps that keep the request prefix stable between turns (so the prompt cache keeps hitting), leaves a stub naming each elided result's tool, arguments, size and first and last lines, counts tool definitions against the budget (`compact_request`) and never sends a request over it while anything can still be elided or dropped. Default `keep_recent_turns` is now 1. On `context_trail` (Haiku, N=5): 3/5 at 141k tokens/run before, 5/5 at 32k after.
- Phase 9.4: the daemon's general tools move into `aidaemon-tools` (or a crate per heavy tool). First the support they share: `command_risk`, `command_effects`, `daemon_guard`, `sanitize`, `fs_utils`, `result_spill`, `background_deliverable`, `file_delivery`, `owner_answers`, `consequential`, `process_control`, and `text` (`truncate_str`, `truncate_with_note`, `contains_keyword_as_words`; `aidaemon-events::text` re-exports them). `backend` gains `SharedExecutionBackend` and `normalize_path_lexically`. `fs_utils::local_backup_path` and `backup_before_overwrite` take the backup root from the host. `aidaemon-tools` now depends on `aidaemon-policies`.
- Phase 9.4: `aidaemon-tools` gains the daemon's `web_search` (`web_search::WebSearch`, its backends and `SearchConfig`; `with_description` for a host's wording) and `http_request` (`http_request::HttpRequestTool` with auth profiles `HttpAuthProfile`/`HttpAuthType`; OAuth refresh through the new `OAuthTokens` trait), plus `backoff` and, behind the new `testing` feature, `testing::ToolCalls` and `call_facts`. Names, descriptions and schemas are unchanged (daemon goldens).
- Phase 9.4: `aidaemon-tools::local_backend::LocalBackend`, the daemon's host backend (workspace-confined unless told otherwise), and `testing::active_execution_backend`, `call_with_facts`, `call_through_kernel` and `CallWithFacts`; `text::floor_char_boundary`. `aidaemon-events::attachments`: resource handles and saved tool artifacts and observations (`tool_artifact_attachment`, `save_tool_observation_image`, `sanitize_filename`).
- New crate `aidaemon-tool-browser` (Phase 9.4): the daemon's `browser` tool behind the `chromium` feature, with `BrowserSettings` and `SessionIsolation` always available. Name, description and schema unchanged (daemon golden).
- Phase 9.4: `aidaemon-tools` gains the daemon's `system_info`, `git_info`, `git_commit`, `project_inspect` and `service_status` tools (modules of the same names). Schemas unchanged (daemon goldens).
- Phase 9.0: the daemon's shared contract types and store traits moved into the crates that own them: approval vocabulary, `status::StatusUpdate`, `TerminalConfinement`, `NothingSentError` and the CLI-agent store into `aidaemon-tools`; `specialist::SpecialistKind` into `aidaemon-agents`; the message, dialogue, summary and prompt-snapshot stores into `aidaemon-events::stores`; `triggers::TriggerEvent` and the work-only composite stores into `aidaemon-work`; `registry` into `aidaemon-mcp`; internal `context` and `hub` modules into `aidaemon-channels`; `skills` into `aidaemon-memory`. Serde shapes unchanged.
- Phase 9.3: `aidaemon-work` holds the work domain's logic: `plans` (task plans, their store and the `task_plans` migration), `mandates::policy` and `mandates::history`, `reminders`, `schedule` (cron parsing and next-run times) and `goal_tokens` (goal cancellation and daily budget overrides). Tables unchanged.
- Phase 9.3: `aidaemon-work::tools::manage_mandates` (the `manage_mandates` tool, unchanged name, schema and confirmation cards) and `tools::objective_status`; `stores::MandateToolStore`. A host gives strategy skills through `StrategySkills`.
- Phase 9.3: `aidaemon-work::scheduler`, the goal cycle (due mandate reviews, scheduled goals and reminders, stuck and orphaned tasks, escalated recoveries, goal notifications) as `GoalCycle` and `GoalCycleService`, over `SchedulerStore`. The host gives its agent runtime (`TaskLeadHost`), channels (`NoticeDelivery`) and presentation (`WorkPresenter`). `goals::{tasks_satisfy_goal_completion, goal_completion_summary_indicates_not_finished, is_terminal_task_status}`; `goal_tokens::SCHEDULED_AUTONOMOUS_*`.
- Phase 9.3: `aidaemon-work::triggers` holds the trigger event bus (`event_bus`, `EventSender`, `EventReceiver`), `TriggerSettings` and, behind the new `email-trigger` feature, the IMAP `TriggerManager`.

## [0.2.0]: first release (not yet published)

### aidaemon-core (public contract)
- Hook-driven agent loop: `Agent` builder, streaming `Provider` with `Retry`, `Tool` with annotations, access manifests and timeout hints, five hook points.
- Receipts for every tool call, typed outcomes, limits, stuck detection, events and `RunStore`.
- Serializable `RunState` with interrupt/resume and crash-safe in-flight tool batches.
- `testing` feature: `ScriptedProvider` and conformance kits for tools and providers.

### aidaemon-sdk (public contract)
- One-crate bundle re-exporting the kernel, with a feature per subsystem.
- `preset(id)`: a working agent in about 15 lines (provider from env, default safeguard hooks, store, tools, any `MemorySystem`).
- `#[tool]` (feature `macros`, from `aidaemon-macros`): tools from async fns with a derived JSON schema and explicit effects.
- Typed structured output: `structured::complete_typed::<T>` and `AgentStructuredExt::run_typed`, with one repair retry.
- Examples: quickstart, custom tool, approvals, resume after a crash, sub-agent, memory swap, channel bot, SSE server, OpenTelemetry traces, coding agent, support triage.

### aidaemon-channels (public contract: `channel` module)
- `Channel` trait and `ChannelRunner` (one session per conversation, approvals as buttons); conformance kit for adapter authors.

### aidaemon-memory (public contract: traits)
- `MemoryStore` and `Embedder` traits; facts with supersession, episodes, hybrid recall, SQLite store, offline `HashEmbedder`, optional `fastembed`.

### Implementation tier (no stability promise)
- `aidaemon-providers`: Anthropic and OpenAI-compatible providers (and Google, xAI, ChatGPT), streaming.
- `aidaemon-tools`: workspace file tools, search, sandboxed shell, web fetch, checklist.
- `aidaemon-store-sqlite`: SQLite `RunStore` with journal, snapshots, upkeep helpers.
- `aidaemon-mcp`: MCP stdio client exposing server tools.
- `aidaemon-agents`: agents as tools with child budgets.
- `aidaemon-policies`: approvals, scope and egress guards, redaction, untrusted output, context window, memory injection, tracing.
- `aidaemon-eval`: YAML scenario eval kit with N-of-M pass rates.
- `aidaemon-channel-telegram`, `-slack`, `-discord`: channel adapters.

### Internal tier (published for the daemon, no stability promise)
- `aidaemon-events`, `aidaemon-work`, `aidaemon-decisions`.
