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
- `aidaemon-policies`: approvals, scope and egress guards, completion checks, redaction, untrusted output, context window, memory injection, tracing.
- `aidaemon-eval`: YAML scenario eval kit with N-of-M pass rates.
- `aidaemon-channel-telegram`, `-slack`, `-discord`: channel adapters.

### Internal tier (published for the daemon, no stability promise)
- `aidaemon-events`, `aidaemon-work`, `aidaemon-decisions`.
