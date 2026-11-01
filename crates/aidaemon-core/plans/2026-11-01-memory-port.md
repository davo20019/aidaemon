# Memory Through SDK Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The daemon agent reaches long-term memory only through SDK seams, so a different memory backend (for example `aidaemon-memory`) can replace the daemon's SQLite memory for agent turns without editing the turn code.

**Why:** The SDK crates are already separated (`aidaemon-core` knows nothing about memory; `aidaemon-policies::Recall` is the read contract; `aidaemon-memory` is an optional plugin). The daemon's SDK engine does not use them: memory is rendered into the system prompt by `build_system_prompt_for_message` (`src/agent/runtime/system_prompt.rs`), written by `spawn_post_turn_memory` / `spawn_post_task_learning` called directly from `handle_message_sdk`, and the memory tools are hard-wired daemon tools. Owner request 2026-11-01: make the daemon a fully SDK agent.

**Sequencing:** Run after `2026-10-31-sdk-engine-parity.md` Task 8 (SDK is the only engine, legacy loop deleted). Doing it before would mean keeping legacy prompt memory and SDK-injected memory in sync.

**Architecture:**

```text
handle_message_sdk
  └─ agent.memory: Arc<dyn MemoryPlugin>
       .turn(&MemoryTurn) -> MemoryParts { hooks, tools, owned_daemon_tools }
         DaemonMemory (default):  MemoryInjection<DaemonRecall>  (before_model)
                                  DaemonMemoryWriter             (on_run_end)
                                  owns remember_fact / manage_memories / manage_people
         SdkMemory (aidaemon-memory): MemoryInjection<MemoryRecall>, MemoryWriter,
                                  RememberTool, RecallTool
```

`MemoryTurn` carries everything privacy scoping needs (session, task, role, `ChannelContext`, `memory_pipeline_allowed`, mandate / depth / memory-isolated-session flags), so the thin `Recall { session_id, query, limit }` contract stays enough: the plugin builds a per-turn recall that already knows the channel.

## Global Constraints

- Same as the parity plan: `cargo fmt`, `cargo clippy --all-features -- -D warnings`, `RUST_MIN_STACK=8388608` tests, synthetic data only, commits by David Loor M. with no Claude trailers.
- Privacy behavior must not change: owner-DM vs group vs public-external scoping, mandate isolation (no memory at all), memory-isolated worker sessions (summaries yes, owner facts no), people intelligence opt-in. Every existing privacy test must pass unchanged.
- Move code, do not copy it. The memory half of `build_system_prompt_for_message` moves into `DaemonRecall`; the prompt builder keeps only non-memory context.
- No new kernel primitives except `Hook::on_run_end` (Task 1). It is a lifecycle event, not a gate: it cannot change the outcome.

## Review Focus

1. A mandate worker turn gets zero memory hooks and zero memory tools (Task 3 test).
2. A public-external turn gets no personal facts, episodes, people or procedures in any model request (Task 4 test).
3. The injected memory note sits after every leading system message and before history, and the core system message (message zero) bytes do not change between two turns with different queries (prompt-cache stability, Task 4 test).
4. With the `aidaemon-memory` plugin configured, a remembered fact is recalled on the next turn and the daemon `facts` table stays empty (Task 7).
5. Post-turn extraction runs once per completed turn, not once per `on_stop` attempt, and never on a cancelled or failed run (Task 5 test).

---

### Task 1: `Hook::on_run_end` in the kernel

**Files:** `crates/aidaemon-core/src/hook.rs`, `crates/aidaemon-core/src/run_loop.rs`, kernel tests.

- [ ] **Step 1: Failing test** `on_run_end_fires_once_with_final_status`: a hook counting calls; runs ending `Completed`, `Cancelled`, `ProviderFailed`, `MaxTurns` each fire it exactly once with that status and `final_text` (Some only when completed). A run whose `on_stop` returned `Continue` once still fires it once.
- [ ] **Step 2: Run** `cargo test -p aidaemon-core on_run_end` — FAIL.
- [ ] **Step 3: Implement** `async fn on_run_end(&self, _ctx: &HookContext<'_>, _outcome: &RunOutcome) {}` (no-op default), called in registration order after the outcome is final, on every exit path of the run loop (including interrupt). Document it in the trait docs next to `on_stop`.
- [ ] **Step 4: Run** — PASS, plus `cargo test -p aidaemon-core` and the core conformance kit.
- [ ] **Step 5: Commit** `feat(core): on_run_end lifecycle hook`.

### Task 2: Custom rendering for `Recall`

**Files:** `crates/aidaemon-policies/src/memory.rs`.

- [ ] **Step 1: Failing tests:** `recall_render_default_matches_snippet_note` (default `render` produces the current banner + bullets, byte-identical to today's note); `custom_render_replaces_note` (a `Recall` overriding `render` gets its text injected verbatim as one system message); `render_none_injects_nothing`.
- [ ] **Step 2: Run** `cargo test -p aidaemon-policies memory` — FAIL.
- [ ] **Step 3: Implement** a provided method `async fn render(&self, session_id: &str, query: &str, limit: usize) -> Option<String>` whose default calls `recall` and builds today's note. `MemoryInjection` calls `render` and caches its output per session and query (the cache already exists; it now stores the rendered note). `aidaemon-memory`'s `MemoryRecall` is unchanged.
- [ ] **Step 4: Run** — PASS, plus `cargo test -p aidaemon-memory`.
- [ ] **Step 5: Commit** `feat(policies): let a Recall render its own memory note`.

### Task 3: `MemoryPlugin` on the daemon agent

**Files:** new `src/agent/memory_plugin.rs`; `src/agent/mod.rs` (field `memory: Arc<dyn MemoryPlugin>`), `src/agent/construct.rs` (default `DaemonMemory`, `with_depth` shares the parent's), `src/agent/runtime/sdk_engine.rs` (register the parts), root `Cargo.toml` (depend on `aidaemon-policies`).

**Interfaces:**

```rust
pub struct MemoryTurn {
    pub session_id: String,
    pub task_id: String,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub memory_pipeline_allowed: bool,
    pub memory_persistence_allowed: bool,
    pub mandate: bool,
    pub depth: usize,
    pub owner_dm_fact_cache: Option<Vec<Fact>>,
    pub model: String,
}

pub struct MemoryParts {
    pub hooks: Vec<Arc<dyn sdk::Hook>>,
    pub tools: Vec<Arc<dyn sdk::Tool>>,
    /// Daemon tools this plugin owns; offered only while it is the active plugin.
    pub owned_daemon_tools: &'static [&'static str],
}

#[async_trait]
pub trait MemoryPlugin: Send + Sync {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts;
}
```

- [ ] **Step 1: Failing tests:** `mandate_turn_gets_no_memory_parts`; `memory_tools_follow_the_active_plugin` (with a stub plugin owning nothing, `remember_fact` is not offered to the model; with `DaemonMemory` it is); `plugin_hooks_run_on_sdk_turns` (stub plugin hook records a `before_model` call during `handle_message`).
- [ ] **Step 2: Run** `RUST_MIN_STACK=8388608 cargo test memory_plugin` — FAIL.
- [ ] **Step 3: Implement.** `handle_message_sdk` builds `MemoryTurn` from `SdkTurnSetup` + channel context, calls `self.memory.turn(..)`, adds `hooks` after `SameToolLimit` and `LedgerStop` (so a rejected final answer never reaches a writer), adds `tools` with `tool_arc`, and filters daemon memory tools out of `offered_tools` unless listed in `owned_daemon_tools`. `DaemonMemory` returns empty parts in this task; Tasks 4-5 fill it.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(agent): memory plugin seam on SDK turns`.

### Task 4: `DaemonRecall` — memory leaves the system prompt

**Files:** `src/agent/runtime/system_prompt.rs` (remove the memory half), new `src/agent/memory_plugin/recall.rs`, `src/skills/mod.rs` (split `build_system_prompt_with_memory_report` into a skills-content renderer and a memory renderer), `src/agent/runtime/core_prompt.rs` (core profile input removed).

What moves into `DaemonRecall::render`: channel-scoped facts, critical (identity) facts, episodes, personal goals, behavior patterns, procedures, error solutions, communication profile, people / current-speaker context and privacy rules, core profile, `record_memory_recall`, and the `MemoryRetrieval` decision point. What stays in the tail: date/time, session context, session summary, matched skill content, project instructions, resume checkpoint, new-request dialogue note.

- [ ] **Step 1: Failing tests** (integration, `MockProvider` records requests):
  - `owner_dm_request_carries_recalled_memory_note`: seeded fact `favorite color = teal` → the request has a system message after the tail containing `teal`; the tail itself does not.
  - `public_external_turn_has_no_personal_memory` (Review Focus 2).
  - `core_prompt_stable_across_queries` (Review Focus 3): two turns, different queries and recalled facts → identical message-zero bytes.
  - `memory_note_cached_within_a_turn`: a 3-call run queries the store once (count `get_relevant_facts_for_channel` via a counting state wrapper or the decision point count).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.** `DaemonRecall` holds `Arc<dyn StateStore>`, the event emitter and the `MemoryTurn`; the fetch-and-scope code moves verbatim from `build_system_prompt_for_message`. `recall()` returns the rendered fact lines (for generic callers); `render()` returns the full daemon memory block with its section headings, so the model sees the same sections as before. Core profile moves out of the core prompt; update `core_cache_decision` components and its tests. Move (do not duplicate) privacy tests that called the prompt builder so they exercise `DaemonRecall`.
- [ ] **Step 4: Run** the new tests, `RUST_MIN_STACK=8388608 cargo test --lib memory`, `cargo test integration_tests`, `cargo test --lib harness_eval --all-features`.
- [ ] **Step 5: Commit** `refactor(agent): recall daemon memory through MemoryInjection`.

### Task 5: `DaemonMemoryWriter` — post-turn memory as an `on_run_end` hook

**Files:** `src/agent/memory_plugin/writer.rs`, `src/agent/loop/completion_phase.rs` (or wherever the functions live after the parity Task 8; make `post_completion_memory_plan` reachable), `src/agent/runtime/sdk_engine.rs` (remove the direct `spawn_post_turn_memory` / `spawn_post_task_learning` calls; `TurnScope.learning` becomes `Arc<Mutex<LearningContext>>`).

- [ ] **Step 1: Failing tests:**
  - `writer_extracts_once_per_completed_turn` (Review Focus 5): a run with one `on_stop` `Continue` → one extraction spawn.
  - `writer_skips_failed_and_cancelled_runs`.
  - `writer_sanitizes_reply_before_extraction`: final text containing an internal marker reaches the extractor without it (`sanitize_user_facing_reply`).
  - `memory_isolated_session_summarizes_but_does_not_extract` (existing behavior via `post_completion_memory_plan`).
  - `learning_records_run_tool_calls`: `process_learning` receives the tool calls the run made.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.** The hook holds the provider/router snapshot, `state`, `event_store`, `context_window_config`, the `MemoryTurn` and the shared learning context. On `Completed` it spawns progressive extraction + incremental summarization with the same plan as today, then learning; on other statuses it records learning with `completed_naturally = false` (matches current behavior) and skips extraction.
- [ ] **Step 4: Run** — PASS, plus the full SDK integration tests.
- [ ] **Step 5: Commit** `refactor(agent): post-turn memory and learning as an SDK run-end hook`.

### Task 6: Memory tools owned by the plugin

**Files:** `src/agent/memory_plugin.rs`, `src/startup/` tool registration (only if the owned-tool filter needs the names there).

- [ ] **Step 1: Failing test** `daemon_memory_owns_its_tools`: `DaemonMemory.turn(..).owned_daemon_tools` == `["remember_fact", "manage_memories", "manage_people"]`, and the tool roster in the core prompt lists them only when they are offered.
- [ ] **Step 2-4:** Implement and run. `ESSENTIAL_TOOLS` keeps the names; the filter in Task 3 removes them for other plugins.
- [ ] **Step 5: Commit** `refactor(agent): memory tools follow the memory plugin`.

### Task 7: Swap proof with `aidaemon-memory`

**Files:** new `src/agent/memory_plugin/sdk_memory.rs` behind a `sdk-memory` cargo feature (root depends on `aidaemon-memory` only with it), config `[memory] backend = "daemon" | "sdk"` (default `daemon`), `config.toml.example`.

- [ ] **Step 1: Failing integration test** `sdk_memory_plugin_replaces_daemon_memory` (Review Focus 4): harness with `SdkMemory` over `InMemoryMemoryStore` + `HashEmbedder`; turn 1 the model calls `remember` (aidaemon-memory `RememberTool`) for `favorite color = teal`; turn 2's model request contains `teal` in the injected note; `harness.state.get_facts(None)` is empty; `remember_fact` was never offered.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** `SdkMemory { memory: aidaemon_memory::Memory }` returning `MemoryInjection<MemoryRecall>`, `MemoryWriter` (only when the turn allows persistence and is not mandate / memory-isolated), `RememberTool`, `RecallTool`, with a scope resolver keyed by channel id + visibility so group and DM memories stay apart. Startup builds it from config (`SqliteMemoryStore` at `[memory] path`).
- [ ] **Step 4: Run** — PASS. Full checklist: `cargo fmt`, clippy all features, `RUST_MIN_STACK=8388608 cargo test --all-features`, harness eval.
- [ ] **Step 5: Commit** `feat(memory): aidaemon-memory as a drop-in memory plugin`. CHANGELOG Unreleased entry.

## Non-goals

- Background memory jobs outside an agent turn (consolidation, retention, embedding backfill, people intelligence, `/memory` commands, dashboards) keep using the daemon store. With `backend = "sdk"` they simply find nothing new. Moving them is a separate decision.
- Context fitting (`crate::memory::context_window` budget/fit in `serve_sdk_model_call`) is transcript management, not long-term memory, and stays where the parity plan put it.
- Migrating existing daemon memory into an `aidaemon-memory` store.
