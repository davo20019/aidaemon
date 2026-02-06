# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
cargo build                          # debug build
cargo build --release                # release build
cargo build --features discord       # with Discord channel
cargo build --features slack         # with Slack channel
cargo build --features browser       # with headless Chrome tool
cargo build --features encryption    # with SQLCipher encryption
cargo build --features "discord,slack,browser"  # multiple features
```

```bash
cargo test                           # run all tests
cargo test router                    # run router tests only
cargo test --lib memory              # run memory tests only
cargo test <test_name>               # run a single test by name
```

```bash
cargo clippy                         # lint
cargo fmt --check                    # check formatting
cargo fmt                            # auto-format
```

No `rustfmt.toml` — uses default Rust formatting conventions.

## Architecture

**aidaemon** is a personal AI agent daemon (single Rust binary) accessible via Telegram/Slack/Discord with agentic tool use, MCP integration, and persistent memory.

### Core Flow

```
main.rs → config loading → core.rs (subsystem init) → spawn channels + agent + background tasks
```

The **agent loop** (`agent.rs`) is the heart: user message → build history → smart router selects model tier → LLM call → if tool calls, execute and loop → return response. It has stall detection (same tool 3+ times), repetition detection, and hard iteration limits.

### Key Abstractions (traits.rs)

Four core traits drive the architecture:
- **`Tool`** — anything the LLM can call (`name()`, `schema()`, `call()`)
- **`Channel`** — input sources (`send_text()`, `send_media()`, `request_approval()`)
- **`StateStore`** — persistence layer (SQLite impl in `state/sqlite.rs`)
- **`ModelProvider`** — LLM backends (`chat()`, `list_models()`)

### Module Map

- **`core.rs`** — orchestrates startup: creates state store, event store, provider, router, tools, agent, channels, dashboard. Handles the deferred wiring for `SpawnAgentTool` (circular dep: Agent ↔ SpawnAgentTool resolved via weak reference + `set_agent()`).
- **`agent.rs`** (~100KB) — agent loop, message handling, system prompt construction, tool execution with status updates. Largest file; most feature changes touch this.
- **`channels/hub.rs`** — `ChannelHub` routes messages between session IDs and channels via `SessionMap: Arc<RwLock<HashMap<String, String>>>`.
- **`state/sqlite.rs`** (~83KB) — multi-layer memory: messages, facts (with embeddings), episodes, goals, behavior patterns, procedures, error solutions, expertise, user profiles, token usage. Schema migrations are inline.
- **`router.rs`** — classifies queries into Fast/Primary/Smart model tiers using keyword heuristics and message length.
- **`events/`** — event sourcing: all agent activity is immutable events. `consolidation.rs` processes events into facts/procedures daily. `context.rs` compiles session context from events.
- **`memory/`** — background consolidation (embeddings every 5s, fact extraction every 6h, decay daily). Uses `fastembed` AllMiniLML6V2 for vector embeddings.
- **`plans/`** — persistent multi-step task plans with detection, generation, tracking, and crash recovery.
- **`tools/terminal.rs`** — shell execution with risk assessment (`command_risk.rs`) and inline approval flow (Allow Once / Allow Always / Deny).
- **`providers/`** — `openai_compatible.rs`, `google_genai.rs`, `anthropic_native.rs` — pluggable LLM backends.
- **`skills.rs`** — trigger-based markdown files loaded from a directory; keywords in user messages activate matching skills which inject into the system prompt.
- **`config.rs`** — loads `config.toml` with secret resolution: `"keychain"` → OS credential store, `"${ENV_VAR}"` → env var, or plain value.

### Concurrency Model

- Tokio async runtime throughout
- `Arc<RwLock<...>>` for shared state
- Background tasks via `tokio::spawn` (memory consolidation, event pruning, health probes, scheduler ticks)
- Channels run their own event loops (Telegram polling, Discord gateway, Slack Socket Mode)

### Feature Flags

- `browser` — `chromiumoxide` for headless Chrome
- `discord` — `serenity` for Discord bot
- `slack` — `tokio-tungstenite` for Slack Socket Mode
- `encryption` — `libsqlite3-sys/bundled-sqlcipher` for SQLCipher

### Platform-Specific

Keyring crate uses platform-native backends: `apple-native` (macOS), `sync-secret-service` (Linux), `windows-native` (Windows). These are selected via `[target.'cfg(...)'.dependencies]` in Cargo.toml.

### Testing

Tests are spread across ~25 files as `#[cfg(test)]` modules. Key test areas: router tier classification, memory/embedding math, plan detection, event context, command risk patterns, skill matching, scheduler parsing. No integration test suite — validation is primarily via runtime.
