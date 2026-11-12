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

### Removed
- `aidaemon-policies`: `ReceiptCompletionCheck` and `Requirement`. Measured live (Decision 4: a policy that shows no gain is removed before it becomes public API), it gained nothing: 34/40 with and without on the first eight scenarios, and 5/5 with and without on `completion_retry`, the scenario built for it (N=5, `claude-haiku-5-5`), at slightly more turns. It is not a safety boundary. A receipt-based check remains a few lines of `on_stop` (see `aidaemon-core/DESIGN.md`); `completion_retry` stays as a plain-kernel regression.

### Changed
- Phase 9.0: the daemon's shared contract types and store traits moved into the crates that own them: approval vocabulary, `status::StatusUpdate`, `TerminalConfinement`, `NothingSentError` and the CLI-agent store into `aidaemon-tools`; `specialist::SpecialistKind` into `aidaemon-agents`; the message, dialogue, summary and prompt-snapshot stores into `aidaemon-events::stores`; `triggers::TriggerEvent` and the work-only composite stores into `aidaemon-work`; `registry` into `aidaemon-mcp`; internal `context` and `hub` modules into `aidaemon-channels`; `skills` into `aidaemon-memory`. Serde shapes unchanged.

## [0.1.0]: first release (not yet published)

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
