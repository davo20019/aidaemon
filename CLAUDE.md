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

#### Tool Schema Format (IMPORTANT)

`schema()` must return the **full OpenAI function object** with `name`, `description`, and `parameters`. Do NOT return just the parameters object — the LLM won't know what the tool is called or what it does.

```rust
// CORRECT — includes name, description, and parameters wrapper
fn schema(&self) -> Value {
    json!({
        "name": "my_tool",
        "description": "What this tool does and when to use it",
        "parameters": {
            "type": "object",
            "properties": { ... },
            "additionalProperties": false
        }
    })
}

// WRONG — missing name/description, LLM can't identify or select this tool
fn schema(&self) -> Value {
    json!({
        "type": "object",
        "properties": { ... }
    })
}
```

#### Dynamic Bots (IMPORTANT)

Bots can be added two ways: **config-based** (in `config.toml`) or **dynamic** (added via `/connect` command, stored in `dynamic_bots` SQLite table). When registering tools or features that depend on channel tokens (e.g., `ReadChannelHistoryTool` needs Slack bot_tokens), you MUST check BOTH sources:
- `config.all_slack_bots()` — config-based bots only
- `state.get_dynamic_bots().await` — dynamic bots from DB

Failing to check dynamic bots will cause features to silently not register even though the channel is connected and working.

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
- **`skills/mod.rs`** — advanced skill system with trigger-based matching, bundled resources, and dynamic management. Skills can come from filesystem (`.md` files), URLs, inline content, remote registries, or auto-promotion from successful procedures. `SharedSkillRegistry` (`Arc<RwLock<Vec<Skill>>>`) allows runtime add/remove. Each skill has metadata (name, description, triggers, source, source_url, enabled flag) and can bundle resource files (scripts, references, configs) in a directory structure. Matching is whole-word + case-insensitive with optional LLM confirmation via fast model.
- **`skills/resources.rs`** — `ResourceEntry` and `ResourceResolver` trait (`FileSystemResolver` impl) for loading bundled skill files on demand with path traversal protection and 32KB size cap.
- **`tools/use_skill.rs`** — `UseSkillTool` lets the agent activate skills on demand by name.
- **`tools/manage_skills.rs`** — `ManageSkillsTool` with 10 actions: add (from URL), add_inline, list, remove, enable, disable, browse (search registries), install (from registry), update (re-fetch from source). Includes SSRF protection.
- **`tools/skill_resources.rs`** — `SkillResourcesTool` with list/read actions for loading bundled resource files from skills.
- **`tools/skill_registry.rs`** — registry client for browsing/searching/installing skills from remote JSON manifests configured in `[skills.registries]`.
- **`memory/skill_promotion.rs`** — `SkillPromoter` background task (12h cycle) that auto-converts successful procedures (≥5 uses, ≥80% success rate) into skills via LLM generation.
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

Tests are spread across 40+ files as `#[cfg(test)]` modules, totaling 400+ tests. Key test areas:

- **Unit tests:** router tier classification, memory/embedding math, plan detection, event context, command risk patterns, skill matching, scheduler parsing, SQLite state store CRUD, provider message conversion, terminal output formatting, channel hub routing, content sanitization, markdown formatting
- **Integration tests:** 60+ tests exercising the full agent loop with mock LLM
- **Property-based tests:** `proptest` for fuzz-testing command risk classification, string truncation, content sanitization, and markdown formatting
- **Dev-dependencies:** `tempfile`, `proptest`, `insta` (with `yaml` feature)

```bash
cargo test                           # run all tests
cargo test integration_tests         # run integration tests only
cargo test test_tool_execution       # run a single test by name
cargo test --lib memory              # run memory-related tests only
cargo test proptest                  # run property-based tests
```

#### CI/CD

The project uses GitHub Actions for continuous integration and release gating.

**CI pipeline** (`.github/workflows/ci.yml`) — runs on push to `master` and all PRs:
- `check` job: `cargo fmt --check` (continue-on-error) + `cargo clippy --all-features -- -D warnings`
- `test` job: `cargo test --all-features` on ubuntu-latest and macos-14
- `build-check` job: `cargo build --release --features "browser,slack,discord"`
- `coverage` job: `cargo-llvm-cov` → Codecov (continue-on-error, visibility only)

**Release gating** (`.github/workflows/release.yml`):
- `quality-gate` job runs `cargo test --all-features` before any build/release job
- All downstream jobs (build, GitHub Release, crates.io, Homebrew) are blocked if tests fail

To generate local coverage: `cargo llvm-cov --all-features --lcov --output-path lcov.info`

#### Integration Tests

Integration tests exercise the real agent loop (`Agent::handle_message`) with a mock LLM provider and temp-file SQLite DB. They verify the same code path all channels use.

```bash
cargo test integration_tests          # run integration tests only
cargo test test_tool_execution        # run a single integration test
```

**What they test:** Agent loop, tool execution, memory/state persistence, multi-turn history, session isolation, channel auth simulation, memory privacy (channel-scoped, private, global), security (sanitization, prompt injection defense), stall detection, multi-step workflows, system prompt structure.

**Test infrastructure** (`src/testing.rs`):
- **`MockProvider`** — mock `ModelProvider` with scripted responses and call logging. Use `MockProvider::new()` for default "Mock response", or `MockProvider::with_responses(vec![...])` for scripted sequences. Helpers: `text_response()`, `tool_call_response()`.
- **`TestChannel`** — mock `Channel` that captures outgoing messages. Not wired to ChannelHub — tests call `agent.handle_message()` directly.
- **`setup_test_agent(provider)`** — creates a fully wired `Agent` with real `SqliteStateStore` (temp file), real `EventStore`/`PlanStore`, real `EmbeddingService`, and `SystemInfoTool` only. Returns `TestHarness { agent, state, provider, channel }`. Each call creates an isolated DB for safe parallel execution.

First run downloads the fastembed model (~25MB, cached in `~/.cache/`).
