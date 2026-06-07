# Pillar B: Turn-Anchored Fetch & Monotonic Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the message-count sliding window with a turn-anchored history mechanism — whole archived turns rendered once into a byte-stable permanent form, fetched by turn-start sequence, evicted whole-turn at an anchor — so the `core → archived[..N-1]` prefix is byte-identical across turns and every remaining prefix break is an eviction, a logged mutation, or a logged late-write re-render.

**Architecture:** Per `docs/superpowers/specs/2026-06-06-cross-turn-prefix-stability-design.md` §Pillar B. Three layers land in order: (1) **turn identity** — `turn_id` on the canonical conversation events + `TaskEnd`, a `terminal_state` derivation, and a monotonic `(turn_seq, msg_seq)` ordering built from the existing `events.id` (turn_seq = MIN(id) per turn, joined at fetch — no denormalization, no new counter); (2) **turn-anchored fetch + pure `render_turn`** with a per-session/per-turn render cache keyed by a content fingerprint; (3) **whole-turn eviction** at an in-memory anchor against an archived-region budget. The age ladder (Prior-1/Prior-2 collapse, message-count trim, `current_user_injected` synthetic-user path, index-based identity-preserve bypass) is deleted and absorbed by the new mechanism. Pillar A's stable core, `[Task Context]` tail, core cache, and the `tail_hash`/`prefix_hash_archived` fingerprint regions already shipped and are the measurement harness this plan builds against.

**Tech Stack:** Rust; `sqlx`/SQLite (event store); existing modules `src/events/` (`mod.rs`, `payloads.rs`, `store.rs`, `conversation_turn.rs`), `src/db/migrations.rs`, `src/state/sqlite/mod.rs`, `src/agent/loop/message_build_phase.rs`, `src/agent/loop/sliding_window.rs`, `src/agent/loop/prefix_fingerprint.rs`, `src/agent/mod.rs`, `src/agent/construct.rs`; new module `src/agent/loop/turn_render.rs`; `scripts/cache-attribution.py`.

**Baselines (comparators):** post-C run 2026-06-07 — median payload 16,238 tokens; median turn-start evaluated 15,565 tokens. Pillar A made the core cross-turn stable (verified: `prefix_hash_system` constant across turns, one `Core prompt invalidated component=initial`, then cache hits) and left `prefix_hash_archived` churning by design — that churn is exactly what Pillar B removes. Spec exit target: **≥80% reduction of median turn-start evaluated tokens vs the post-C baseline** (the absolute bound is archived[N] + tail + new user ≈ 4–5k); this is a measured target whose first valid re-run sets the gating threshold, not a hard pass/fail.

---

## Execution caveats (read first)

- **Concurrent workstreams in the tree.** The working tree hosts other uncommitted work. Reference code by SYMBOL and re-locate before editing (`rg -n 'fn run_message_build_phase' src/`); ALL line numbers in this plan are advisory and WILL have drifted. Never `git add -A`/`-u`; stage only the files each task lists. Whole-file staging is the adjudicated approach for this branch (foreign hunks in a listed file may ride along), but do NOT edit unrelated code.
- **Per-commit gate (CLAUDE.md):** `cargo fmt` → `cargo clippy --all-features -- -D warnings` → tests must pass before EVERY commit. **Known-exempt, do NOT fix and do NOT count as new failures:** (1) test `startup::tools::tests::base_tool_registry_names_match_built_schema_names` (fails on the baseline, unrelated); (2) 3 pre-existing clippy errors in `src/bin/db_probe.rs` — use `cargo clippy --all-features --lib -- -D warnings` to scope around the bin. The gate is **no NEW failures/warnings** beyond these. `cargo test --lib` runs the integration suite too (it is `mod integration_tests;` inside the lib crate; the test paths are flattened as `integration_tests::<fn>` — there is no `part_NN` path segment, so filter by the test function name, not `part_10`).
- **Migration safety.** The events table is live and SQLCipher-encrypted. The `turn_id` column migration (Task 1) must be `ADD COLUMN` idempotent (`IF NOT EXISTS`-style guard via the existing migration framework) and back-compatible: pre-migration rows have `turn_id = NULL` and fall under the legacy rule (excluded from reconstruction, covered by the session summary). Inspect with `db_probe` (`cargo run --bin db_probe --features encryption -- --session "<id>"`).
- **Ordering key discipline (spec §prerequisite).** Timestamps are FORBIDDEN as an ordering key anywhere in the turn-anchored fetch/render path. `events.id` (INTEGER PRIMARY KEY AUTOINCREMENT, strictly monotonic, never reused) is the only ordering key. `msg_seq` = the event's own `id`; `turn_seq` = MIN(`id`) over the turn's rows (a cross-row value — see Task 3). Other event queries elsewhere keep their `created_at` ordering; only the turn-anchored conversation fetch migrates.
- **`turn_seq` materialization decision (locked):** computed **at fetch time** via a join on a `turn_id → MIN(id)` subquery — NOT denormalized onto every row and NOT a write-path lookup. Rationale: restart-safe, correct by construction, zero write-path change, and the fetch is anchored (bounded range, once per build). The supporting index is therefore `(session_id, turn_id, id)` (supports the per-turn MIN and the outer range/scan), which differs from the spec's denormalized `(session_id, turn_seq, msg_seq)` — a sanctioned deviation given the fetch-time-join choice. A pure SQLite expression index is impossible here (turn_seq lives on a different row) and is explicitly ruled out, consistent with the spec.
- **Readiness & sequencing.** Start at Task 1 (turn identity write path) — everything downstream needs `turn_id` persisted and ordered first. **Task 7 (the message-build integration) is the highest-risk step** — it deletes the age ladder and rewires the payload assembly atomically. Sequence Tasks 1–6 first so identity, ordering, rendering, caching, and eviction are all unit-proven before the integration consumes them. Tasks 1–2 and 3 are the prerequisite; if anything forces a redesign it surfaces in Task 3 (ordering) or Task 7 (integration).
- **Pillar A invariants to preserve.** Do not disturb: the stable core / message-zero (`render_core_prompt` + `core_prompts` cache), the `[Task Context]` tail at boundary−1, the final `sort_tool_definitions_by_name` before `MessageBuildData`, and force-text retaining tool defs (`llm_phase.rs`, `tool_choice=none` keeps the array for prefix stability). Pillar B inserts the archived turns BETWEEN the core (index 0) and the tail; it must not move the tail or the core.

---

### Task 1: Turn identity on the write path — `turn_id` column, payloads, emission

**Files:**
- Modify: `src/db/migrations.rs` (add `turn_id TEXT` column to the `events` table; add index `(session_id, turn_id, id)`)
- Modify: `src/events/mod.rs` (`Event` struct gains `turn_id: Option<String>`; `Event::new` extracts `data["turn_id"]` exactly as it extracts `task_id`)
- Modify: `src/events/store.rs` (`append` INSERT binds the new column; the SELECT lists it; `Event` row mapping reads it)
- Modify: `src/events/payloads.rs` (`UserMessageData`, `AssistantResponseData`, `ToolResultData`, `ToolCallData`, `TaskEndData` each gain `turn_id: Option<String>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`; update the round-trip serde tests)
- Modify: the emission sites that write these events (re-locate via `rg -n 'UserMessageData\s*{|AssistantResponseData\s*{|ToolResultData\s*{|ToolCallData\s*{|TaskEndData\s*{' src/`) so each sets `turn_id` from the in-process `current_turn_ids` map / the turn's user-message UUID. The agent already stamps `Message.turn_id` via `append_message_canonical` (`src/agent/runtime/turn_context.rs`); mirror that source.

**Context.** `turn_id` exists on `Message` (`src/traits/conversation.rs:235`, `Option<String>` = the turn's opening user-message UUID) and is stamped in-process today, but is NEVER persisted to events — so hydrated messages always get `turn_id: None` (`conversation_turn.rs:194`). `task_id` is the existing template: it is a first-class indexed column on `events`, extracted from `data["task_id"]` in `Event::new` (`events/mod.rs:60-63`) and bound in `append`. Replicate that exact pattern for `turn_id`. `TaskEnd` must carry `turn_id` too (spec §prerequisite): the archived render's terminal-state placeholder is derived from it (Task 2/4), so the completion record must be turn-attributable.

- [ ] **Step 1: Write the failing tests**

In `src/events/store.rs` tests (re-locate the `#[cfg(test)] mod tests`), add:

```rust
#[tokio::test]
async fn append_persists_and_reads_back_turn_id() {
    let store = test_event_store().await; // reuse the existing test harness ctor
    let mut data = serde_json::json!({"content": "hi", "turn_id": "turn-abc"});
    let ev = Event::new("sess-1", EventType::UserMessage, data.clone());
    assert_eq!(ev.turn_id.as_deref(), Some("turn-abc"), "Event::new extracts turn_id from data");
    let id = store.append(ev).await.unwrap();
    // Read it back via the existing per-session event query and confirm the column round-trips.
    let rows = store.query_events("sess-1").await.unwrap();
    let got = rows.iter().find(|e| e.id == id).unwrap();
    assert_eq!(got.turn_id.as_deref(), Some("turn-abc"));
}

#[tokio::test]
async fn append_turn_id_null_when_absent() {
    let store = test_event_store().await;
    let ev = Event::new("sess-1", EventType::UserMessage, serde_json::json!({"content": "hi"}));
    assert!(ev.turn_id.is_none());
    let id = store.append(ev).await.unwrap();
    let rows = store.query_events("sess-1").await.unwrap();
    assert!(rows.iter().find(|e| e.id == id).unwrap().turn_id.is_none());
}
```

In `src/events/payloads.rs` tests, extend the existing round-trip tests for `UserMessageData`/`AssistantResponseData`/`ToolResultData`/`TaskEndData`: construct with `turn_id: Some("t1".into())`, serialize→deserialize, assert the field survives; and a minimal-JSON (no `turn_id` key) deserialize → `turn_id == None` (back-compat).

- [ ] **Step 2: RED** — `cargo test --lib events 2>&1 | tail -5`. Expected: compile errors (`turn_id` field/extraction missing), then assertion failures.

- [ ] **Step 3: Implement**

Migration (`src/db/migrations.rs`, follow the existing migration-registration pattern — find how columns were added before, e.g. a prior `ALTER TABLE events ADD COLUMN`): add `turn_id TEXT` (nullable, no default needed; NULL = legacy) and `CREATE INDEX IF NOT EXISTS idx_events_turn ON events(session_id, turn_id, id)`. Make it idempotent (guard with the framework's "already applied" check or `ADD COLUMN` wrapped so a re-run is a no-op).

`events/mod.rs`: add `pub turn_id: Option<String>` to `Event` (next to `task_id`); in `Event::new`, after the `task_id` extraction, add `let turn_id = data.get("turn_id").and_then(|v| v.as_str()).map(String::from);` and set the field.

`events/store.rs`: in `append`, add `turn_id` to the column list and the bind (`.bind(&event.turn_id)`); in every `SELECT` that maps rows to `Event` for the conversation path, add `turn_id` to the projection and the row read (`row.get("turn_id")`). The general `query_events` used by the test must read it.

`events/payloads.rs`: add `#[serde(default, skip_serializing_if = "Option::is_none")] pub turn_id: Option<String>` to the five structs.

Emission sites: set `turn_id` from the same source `Message.turn_id` uses (the per-turn UUID in `agent.current_turn_ids` / passed through `append_message_canonical`). For `TaskEndData`, thread the current turn's id from the loop context where `TaskEnd` is emitted (re-locate via `rg -n 'TaskEnd' src/agent/`).

- [ ] **Step 4: GREEN + sweep** — `cargo test --lib events 2>&1 | tail -3`; then a focused run of any agent-loop test that asserts on persisted events. Confirm migration applies cleanly against a fresh temp DB (the test harness creates one).

- [ ] **Step 5: Commit**

```bash
git add src/db/migrations.rs src/events/mod.rs src/events/store.rs src/events/payloads.rs <emission-site files>
git commit -m "feat(pillar-b): persist turn_id on conversation events + TaskEnd; events(session_id,turn_id,id) index"
```

---

### Task 2: Turn identity on the read path + `terminal_state` derivation

**Files:**
- Modify: `src/events/conversation_turn.rs` (`ConversationTurn` gains `turn_id: Option<String>`; `turn_from_event` reads `data["turn_id"]`; `into_message` passes it through — delete the `turn_id: None` hardcode at ~:194)
- Create: `src/events/terminal_state.rs` (the `TerminalState` enum + derivation from `TaskEnd`/`TaskStatus`)
- Modify: `src/events/mod.rs` (register `pub mod terminal_state;` and re-export `TerminalState`; it is consumed by the renderer in Task 4)
- Modify: `src/state/sqlite/mod.rs` (the `hydrate_from_events` path already calls `turn_from_event(...).into_message()` — no change needed beyond confirming `turn_id` now flows; add a hydration test)

**Context.** Both hydration paths (`state/sqlite/mod.rs::hydrate_from_events` and `events/store.rs::get_conversation_history`) funnel through `ConversationTurn::turn_from_event → into_message`, so fixing the one `turn_id: None` site heals both. `TaskStatus` (`events/mod.rs`, `{Completed, Cancelled, Failed}`) is the lifecycle status on `TaskEndData`; the spec's `terminal_state ∈ {completed, failed, cancelled, interrupted}` mirrors it, with `interrupted` = the ABSENCE of any `TaskEnd` for the turn (crash). `content_fp` (Task 5) and the archived placeholder (Task 4) both consume `terminal_state`, so a tool-only turn that later gets a completion record re-renders instead of serving stale bytes.

- [ ] **Step 1: Write the failing tests**

In `src/events/conversation_turn.rs` tests:

```rust
#[test]
fn into_message_propagates_turn_id() {
    let data = serde_json::json!({"message_id": "m1", "content": "hello", "turn_id": "turn-9"});
    let turn = turn_from_event(1, "sess", "user_message", &data, Utc::now()).unwrap();
    assert_eq!(turn.turn_id.as_deref(), Some("turn-9"));
    let msg = turn.into_message();
    assert_eq!(msg.turn_id.as_deref(), Some("turn-9"));
}

#[test]
fn into_message_turn_id_none_for_legacy_event() {
    let data = serde_json::json!({"message_id": "m1", "content": "hello"});
    let turn = turn_from_event(1, "sess", "user_message", &data, Utc::now()).unwrap();
    assert!(turn.into_message().turn_id.is_none());
}
```

In the new `src/events/terminal_state.rs`:

```rust
#[test]
fn terminal_state_from_task_status() {
    assert_eq!(TerminalState::from_task_status(Some(TaskStatus::Completed)), TerminalState::Completed);
    assert_eq!(TerminalState::from_task_status(Some(TaskStatus::Failed)), TerminalState::Failed);
    assert_eq!(TerminalState::from_task_status(Some(TaskStatus::Cancelled)), TerminalState::Cancelled);
    // No TaskEnd record for the turn → crash/interrupted.
    assert_eq!(TerminalState::from_task_status(None), TerminalState::Interrupted);
}

#[test]
fn terminal_state_placeholder_strings_are_fixed() {
    assert_eq!(TerminalState::Completed.placeholder(3), "[completed: 3 tool steps, no text reply]");
    assert_eq!(TerminalState::Failed.placeholder(2), "[failed: 2 tool steps, no text reply]");
    assert_eq!(TerminalState::Cancelled.placeholder(1), "[cancelled: 1 tool steps, no text reply]");
    assert_eq!(TerminalState::Interrupted.placeholder(0), "[task interrupted]");
}
```

- [ ] **Step 2: RED** — `cargo test --lib conversation_turn 2>&1 | tail -5` and `cargo test --lib terminal_state 2>&1 | tail -5`.

- [ ] **Step 3: Implement**

`conversation_turn.rs`: add `pub turn_id: Option<String>` to `ConversationTurn`; in `turn_from_event`, read `let turn_id = data.get("turn_id").and_then(|v| v.as_str()).map(String::from);` and set it on the constructed turn; in `into_message`, replace `turn_id: None` with `turn_id: self.turn_id`.

`terminal_state.rs`:

```rust
//! Pillar B: per-turn terminal state for archived rendering.
//! Spec: 2026-06-06-cross-turn-prefix-stability-design.md §Rendering/§prerequisite.
use crate::events::TaskStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalState { Completed, Failed, Cancelled, Interrupted }

impl TerminalState {
    /// `None` = no TaskEnd record exists for the turn (crash) → Interrupted.
    pub fn from_task_status(status: Option<TaskStatus>) -> Self {
        match status {
            Some(TaskStatus::Completed) => Self::Completed,
            Some(TaskStatus::Failed) => Self::Failed,
            Some(TaskStatus::Cancelled) => Self::Cancelled,
            None => Self::Interrupted,
        }
    }
    /// Fixed deterministic placeholder for a no-text-reply turn (spec §Rendering).
    pub fn placeholder(self, tool_steps: usize) -> String {
        match self {
            Self::Completed => format!("[completed: {tool_steps} tool steps, no text reply]"),
            Self::Failed => format!("[failed: {tool_steps} tool steps, no text reply]"),
            Self::Cancelled => format!("[cancelled: {tool_steps} tool steps, no text reply]"),
            Self::Interrupted => "[task interrupted]".to_string(),
        }
    }
}
```

`events/mod.rs`: `pub mod terminal_state;` and `pub use terminal_state::TerminalState;`.

Add a hydration test in `state/sqlite/mod.rs` tests: append a user-message event with `turn_id`, hydrate, assert the resulting `Message.turn_id` is set.

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib conversation_turn 2>&1 | tail -2
cargo test --lib terminal_state 2>&1 | tail -2
git add src/events/conversation_turn.rs src/events/terminal_state.rs src/events/mod.rs src/state/sqlite/mod.rs
git commit -m "feat(pillar-b): hydrate turn_id into Message; TerminalState derivation + fixed placeholders"
```

---

### Task 3: Monotonic turn-anchored fetch query

**Files:**
- Modify: `src/events/store.rs` (new `get_turns_from_anchor` query + a `turn_seq`-aware row type)
- Modify: `src/events/conversation_turn.rs` (or a small new helper) to group hydrated messages into whole turns

**Context.** Today the fetch is a message-count window (`message_build_phase.rs`: `history_limit = max(40, iter*3).min(120)` → `load_recent_history`), ordered by `created_at`. Pillar B replaces it with: fetch all messages whose turn-start sequence is `>= anchor_turn_seq`, ordered by `(turn_seq, msg_seq)`. `turn_seq = MIN(events.id)` per `turn_id` (cross-row value, joined at fetch — see Execution caveats), `msg_seq = events.id`. This task builds ONLY the query + grouping primitive; wiring into the build path is Task 7.

- [ ] **Step 1: Write the failing tests** (`src/events/store.rs` tests)

```rust
#[tokio::test]
async fn turn_anchored_fetch_orders_by_turn_then_msg_seq() {
    let store = test_event_store().await;
    // Turn A: user(id1) assistant(id2). Turn B: user(id3) tool(id4).
    append_user(&store, "sess", "turn-A", "a-user").await;
    append_assistant(&store, "sess", "turn-A", "a-asst").await;
    append_user(&store, "sess", "turn-B", "b-user").await;
    append_tool(&store, "sess", "turn-B", "b-tool").await;
    let turns = store.get_turns_from_anchor("sess", 0).await.unwrap();
    // Two whole turns, in turn_seq order; within each, msg_seq (id) order.
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].turn_id.as_deref(), Some("turn-A"));
    assert_eq!(turns[0].messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(), vec!["user","assistant"]);
    assert_eq!(turns[1].turn_id.as_deref(), Some("turn-B"));
}

#[tokio::test]
async fn turn_anchored_fetch_late_write_sorts_last_within_its_turn() {
    let store = test_event_store().await;
    append_user(&store, "sess", "turn-A", "a-user").await;       // id1
    append_assistant(&store, "sess", "turn-A", "a-asst").await;  // id2
    append_user(&store, "sess", "turn-B", "b-user").await;       // id3
    // Late write under already-finished turn-A (id4 > id3) — a background notifier.
    append_tool(&store, "sess", "turn-A", "late-tool").await;    // id4
    let turns = store.get_turns_from_anchor("sess", 0).await.unwrap();
    // turn-A's turn_seq is MIN(id)=1 < turn-B's 3, so turn-A still sorts first;
    // the late tool (id4) sorts LAST inside turn-A by msg_seq.
    assert_eq!(turns[0].turn_id.as_deref(), Some("turn-A"));
    assert_eq!(turns[0].messages.last().unwrap().content.as_deref(), Some("late-tool"));
    assert_eq!(turns[1].turn_id.as_deref(), Some("turn-B"));
}

#[tokio::test]
async fn turn_anchored_fetch_respects_anchor_floor_and_excludes_legacy_null() {
    let store = test_event_store().await;
    append_legacy_user(&store, "sess", "legacy").await;          // turn_id NULL
    append_user(&store, "sess", "turn-A", "a-user").await;       // turn_seq = its id
    append_user(&store, "sess", "turn-B", "b-user").await;
    let all = store.get_turns_from_anchor("sess", 0).await.unwrap();
    // Legacy NULL-turn rows are excluded from reconstruction (covered by summary).
    assert!(all.iter().all(|t| t.turn_id.is_some()));
    // Anchor at turn-B's turn_seq drops turn-A.
    let b_seq = all.iter().find(|t| t.turn_id.as_deref()==Some("turn-B")).unwrap().turn_seq;
    let from_b = store.get_turns_from_anchor("sess", b_seq).await.unwrap();
    assert_eq!(from_b.len(), 1);
    assert_eq!(from_b[0].turn_id.as_deref(), Some("turn-B"));
}
```

Add the small test helpers (`append_user`/`append_assistant`/`append_tool`/`append_legacy_user`) near the tests if not already present; each appends the corresponding event with `turn_id` in `data` (legacy variant omits it).

- [ ] **Step 2: RED** — `cargo test --lib store 2>&1 | tail -5` (compile error: `get_turns_from_anchor`, `FetchedTurn` missing).

- [ ] **Step 3: Implement**

Define a grouped-turn type (in `store.rs` or `conversation_turn.rs`):

```rust
pub struct FetchedTurn {
    pub turn_id: Option<String>,
    pub turn_seq: i64,            // MIN(events.id) over the turn
    pub messages: Vec<Message>,   // ordered by msg_seq (events.id) within the turn
}
```

`get_turns_from_anchor(&self, session_id: &str, anchor_turn_seq: i64) -> anyhow::Result<Vec<FetchedTurn>>`:

```sql
SELECT e.id, e.event_type, e.data, e.created_at, e.turn_id, t.turn_seq
FROM events e
JOIN (
    SELECT turn_id, MIN(id) AS turn_seq
    FROM events
    WHERE session_id = ?1 AND turn_id IS NOT NULL
    GROUP BY turn_id
) t ON e.turn_id = t.turn_id
WHERE e.session_id = ?1
  AND e.turn_id IS NOT NULL
  AND t.turn_seq >= ?2
  AND e.event_type IN ('user_message','assistant_response','tool_result')
ORDER BY t.turn_seq ASC, e.id ASC
```

Map each row through `turn_from_event(id, session_id, event_type, &data, created_at).map(|t| t.into_message())` (same hydration as the legacy path — `turn_id` now flows), then GROUP consecutive rows by `turn_id` into `FetchedTurn { turn_id, turn_seq, messages }` (rows are already ordered, so a single linear pass groups them). No `created_at` ordering anywhere. The `(session_id, turn_id, id)` index from Task 1 supports both the subquery MIN and the outer scan.

Edge cases (assert by test where listed): legacy `turn_id IS NULL` excluded (WHERE clause); late writes group with their turn via `turn_id` and sort last by `id` (covered); anchor floor via `turn_seq >= ?2`.

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib store 2>&1 | tail -2
git add src/events/store.rs src/events/conversation_turn.rs
git commit -m "feat(pillar-b): get_turns_from_anchor — whole-turn fetch ordered by (turn_seq, msg_seq), id-only"
```

---

### Task 4: Pure `render_turn` + `RenderMode` + `renderer_version`

**Files:**
- Create: `src/agent/loop/turn_render.rs`
- Modify: `src/agent/loop/mod.rs` (or wherever loop submodules are registered — re-locate via `rg -n 'mod sliding_window' src/agent/loop`) to register `pub(crate) mod turn_render;`

**Context.** `render_turn(turn_messages, mode, renderer_version) -> Vec<Value>` is the single pure renderer. **Current** mode = full append-only conversion (the existing `&Message → Value` logic from `message_build_phase.rs:887-936`, including the orphan-`tool_calls` filter and the `tool_call_id`/`name` mapping). **Archived** mode = the single permanent survivorship form:
- user message text survives **in full**;
- the **last assistant-role record with non-empty content** wins (later empty assistant records lose; tool-call-bearing assistant records contribute only their content), truncated by `MAX_OLD_ASSISTANT_CONTENT_CHARS` (`agent/mod.rs:69`, currently 200);
- if **no** assistant record has non-empty content, emit the `terminal_state` placeholder (Task 2);
- tool results survive as deterministic summaries via `summarize_tool_result` (`sliding_window.rs:32`);
- messages matching `text_relates_to_critical_identity` (`policy/recall_guardrails.rs:323`) survive **verbatim**.
Purity: no timestamps, no map-iteration order, no env formatting. This replaces the Prior-1/Prior-2 ladder, the message-count trim, the `current_user_injected` synthetic path, and the index-based identity bypass — all deleted in Task 7.

- [ ] **Step 1: Write the failing golden tests** (`turn_render.rs`)

```rust
fn user(c: &str) -> Message { /* Message::runtime_defaults with role=user, content=Some(c) */ }
fn assistant(c: &str) -> Message { /* role=assistant */ }
fn assistant_empty_with_tool_call() -> Message { /* role=assistant, content=None, tool_calls_json=Some(...) */ }
fn tool(name: &str, call_id: &str, result: &str) -> Message { /* role=tool */ }

#[test]
fn archived_keeps_user_full_and_last_nonempty_assistant_truncated() {
    let turn = vec![
        user("please do the long thing with lots of detail ...full text..."),
        assistant_empty_with_tool_call(),
        tool("terminal", "c1", "exit 0"),
        assistant(&"X".repeat(500)),       // last non-empty assistant — wins, truncated
        assistant(""),                      // later empty — loses
    ];
    let out = render_turn(&turn, RenderMode::Archived { terminal_state: TerminalState::Completed }, RENDERER_VERSION);
    let joined = serde_json::to_string(&out).unwrap();
    assert!(joined.contains("...full text..."), "user text survives in full");
    assert!(joined.contains(&"X".repeat(MAX_OLD_ASSISTANT_CONTENT_CHARS)), "assistant truncated to cap");
    assert!(!joined.contains(&"X".repeat(MAX_OLD_ASSISTANT_CONTENT_CHARS + 1)));
    assert!(joined.contains("terminal: -> exit 0"), "tool result summarized deterministically");
}

#[test]
fn archived_no_text_reply_uses_terminal_state_placeholder() {
    let turn = vec![ user("run it"), assistant_empty_with_tool_call(), tool("terminal","c1","exit 1") ];
    let out = render_turn(&turn, RenderMode::Archived { terminal_state: TerminalState::Failed }, RENDERER_VERSION);
    let joined = serde_json::to_string(&out).unwrap();
    assert!(joined.contains("[failed: 1 tool steps, no text reply]"));
}

#[test]
fn archived_interrupted_turn_renders_interrupted_placeholder() {
    let turn = vec![ user("hello"), tool("terminal","c1","exit 0") ];
    let out = render_turn(&turn, RenderMode::Archived { terminal_state: TerminalState::Interrupted }, RENDERER_VERSION);
    assert!(serde_json::to_string(&out).unwrap().contains("[task interrupted]"));
}

#[test]
fn archived_preserves_identity_critical_verbatim() {
    let turn = vec![ user("my name is David Loor"), assistant("noted") ];
    let out = render_turn(&turn, RenderMode::Archived { terminal_state: TerminalState::Completed }, RENDERER_VERSION);
    assert!(serde_json::to_string(&out).unwrap().contains("my name is David Loor"));
}

#[test]
fn render_is_deterministic() {
    let turn = vec![ user("hi"), assistant("there"), tool("read_file","c1","12 lines") ];
    let a = render_turn(&turn, RenderMode::Archived { terminal_state: TerminalState::Completed }, RENDERER_VERSION);
    let b = render_turn(&turn, RenderMode::Archived { terminal_state: TerminalState::Completed }, RENDERER_VERSION);
    assert_eq!(a, b);
}

#[test]
fn current_mode_is_append_only_full() {
    let turn = vec![ user("hi"), assistant("there") ];
    let out = render_turn(&turn, RenderMode::Current, RENDERER_VERSION);
    // Current keeps full content, both messages, in order.
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["content"], "hi");
    assert_eq!(out[1]["content"], "there");
}
```

- [ ] **Step 2: RED** — `cargo test --lib turn_render 2>&1 | tail -5`.

- [ ] **Step 3: Implement**

```rust
//! Pillar B: pure per-turn rendering. Spec §Rendering.
//! No timestamps, no map-iteration order, no env-dependent formatting —
//! enforced by golden tests + the debug re-render assertion (Task 5).
use crate::events::TerminalState;

/// Bump when the rendering ALGORITHM changes; invalidates all cached renders.
pub(crate) const RENDERER_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RenderMode {
    Current,
    Archived { terminal_state: TerminalState },
}

pub(crate) fn render_turn(turn_messages: &[Message], mode: RenderMode, _version: u32) -> Vec<Value> {
    match mode {
        RenderMode::Current => render_current(turn_messages),
        RenderMode::Archived { terminal_state } => render_archived(turn_messages, terminal_state),
    }
}
```

`render_current`: lift the existing `&Message → Value` conversion (role/content/`tool_calls` with orphan filter against the turn's own tool-result ids/`tool_call_id`/`name`) out of `message_build_phase.rs` into a shared helper so both Current-mode and the legacy path can call it (avoid divergence; Task 7 deletes the inline copy). Append-only, full content.

`render_archived`: single pass over `turn_messages`:
1. emit the user message verbatim (full content);
2. select the last assistant record with non-empty trimmed content; if found, emit it truncated to `MAX_OLD_ASSISTANT_CONTENT_CHARS` (char-safe truncation — use the existing `floor_char_boundary`/`truncate_str` helper in `utils.rs`, not byte slicing); if none, emit one assistant message whose content is `terminal_state.placeholder(tool_step_count)`;
3. for each tool result, emit a `tool`-role message whose content is `summarize_tool_result(tool_name, args_json, result)` (preserve `tool_call_id`/`name` so ordering fixup stays valid);
4. any message matching `text_relates_to_critical_identity(content)` is emitted verbatim regardless of the above (identity safety cannot rest on summaries).
Reuse `MAX_OLD_ASSISTANT_CONTENT_CHARS`, `summarize_tool_result`, `text_relates_to_critical_identity` (make them `pub(crate)` if not already; that is the only change to their home files — note it in staging).

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib turn_render 2>&1 | tail -2
git add src/agent/loop/turn_render.rs src/agent/loop/mod.rs src/agent/loop/sliding_window.rs src/agent/policy/recall_guardrails.rs src/agent/mod.rs
git commit -m "feat(pillar-b): pure render_turn (Current/Archived) with terminal-state placeholders + identity-verbatim"
```

(Stage `sliding_window.rs`/`recall_guardrails.rs`/`mod.rs` only if you widened visibility of the reused helpers/the const.)

---

### Task 5: Per-session/per-turn render cache + `content_fp`

**Files:**
- Create: `src/agent/loop/turn_render_cache.rs` (the `CachedRender`, the `content_fp` computation, and the pure cache-decision helper)
- Modify: `src/agent/mod.rs` (`TurnRenderCache` type alias + Agent field, mirroring `core_prompts`/`window_keep_from_tracker`)
- Modify: `src/agent/construct.rs` (initialize the field in BOTH construction paths)
- Modify: `src/agent/loop/mod.rs` (register `mod turn_render_cache;`)

**Context.** Mirror Pillar A's proven `core_prompts` cache shape and its pure `core_cache_decision` helper (`runtime/core_prompt.rs`). The render cache is `HashMap<session_id, HashMap<turn_id, CachedRender>>`. `content_fp` = `hash_canonical` (reuse `prefix_fingerprint::hash_canonical`) over the **complete ordered render input** — every message in sequence with all fields (`role`, `content`, `tool_name`, `tool_call_id`, `tool_calls_json`, `annotations`, AND the sequence position) **plus the turn's `terminal_state`**. Denylist (never in the fp): `embedding`, `importance`, `created_at`. `Message.id` IS retained in the fp (DB-stable, strengthens input identity; never rendered). Omissions fail CLOSED (spurious re-render), never open (stale bytes). Lookup requires `content_fp + renderer_version + mode` to match. Debug/test builds always re-render and assert byte equality (nondeterminism = test failure). Logging: hit `debug!`, miss/`fp_mismatch` `info!`.

- [ ] **Step 1: Write the failing tests** (`turn_render_cache.rs`)

```rust
fn sample_turn() -> Vec<Message> { vec![/* user + assistant + tool */] }

#[test]
fn fp_stable_for_identical_input() {
    let a = content_fp(&sample_turn(), TerminalState::Completed);
    let b = content_fp(&sample_turn(), TerminalState::Completed);
    assert_eq!(a, b);
}

#[test]
fn fp_changes_when_terminal_state_changes() {
    let t = sample_turn();
    assert_ne!(content_fp(&t, TerminalState::Completed), content_fp(&t, TerminalState::Failed));
}

#[test]
fn fp_changes_on_late_write() {
    let mut t = sample_turn();
    let base = content_fp(&t, TerminalState::Completed);
    t.push(/* a late tool message */);
    assert_ne!(content_fp(&t, TerminalState::Completed), base, "late write must invalidate the turn");
}

#[test]
fn fp_ignores_denylisted_fields() {
    let mut a = sample_turn();
    let fp_a = content_fp(&a, TerminalState::Completed);
    for m in &mut a { m.embedding = Some(vec![1.0, 2.0]); m.importance = 0.99; /* created_at differs */ }
    assert_eq!(content_fp(&a, TerminalState::Completed), fp_a, "embedding/importance/created_at excluded");
}

#[test]
fn cache_decision_hit_and_miss() {
    // pure helper: (prev: Option<&CachedRender>, fp, version, mode, render_fn) -> (bytes, hit, reason)
    let prev = CachedRender { content_fp: "fp1".into(), renderer_version: RENDERER_VERSION,
                              mode_tag: "archived".into(), bytes: vec![/* rendered */] };
    let (_, hit, _) = render_cache_decision(Some(&prev), "fp1", RENDERER_VERSION, "archived", || unreachable!());
    assert!(hit, "matching fp+version+mode is a hit, render_fn NOT called");
    let (_, hit2, reason) = render_cache_decision(Some(&prev), "fp2", RENDERER_VERSION, "archived", || vec![]);
    assert!(!hit2); assert_eq!(reason, "fp_mismatch");
    let (_, hit3, reason3) = render_cache_decision(Some(&prev), "fp1", RENDERER_VERSION + 1, "archived", || vec![]);
    assert!(!hit3); assert_eq!(reason3, "version_mismatch");
}
```

- [ ] **Step 2: RED** — `cargo test --lib turn_render_cache 2>&1 | tail -5`.

- [ ] **Step 3: Implement**

```rust
//! Pillar B: per-session/per-turn render cache. Spec §Render cache.
use crate::agent::prefix_fingerprint::hash_canonical;
use crate::events::TerminalState;

#[derive(Clone)]
pub(crate) struct CachedRender {
    pub content_fp: String,
    pub renderer_version: u32,
    pub mode_tag: String,   // "archived" | "current"
    pub bytes: Vec<Value>,  // the rendered messages
}

/// Canonical fingerprint of the COMPLETE ordered render input + terminal_state.
/// Denylist: embedding, importance, created_at. Message.id retained. Fail-closed.
pub(crate) fn content_fp(turn_messages: &[Message], terminal_state: TerminalState) -> String {
    let items: Vec<Value> = turn_messages.iter().enumerate().map(|(seq, m)| json!({
        "seq": seq,
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "tool_name": m.tool_name,
        "tool_call_id": m.tool_call_id,
        "tool_calls_json": m.tool_calls_json,
        "annotations": m.annotations,
        // EXCLUDED (denylist): embedding, importance, created_at.
    })).collect();
    hash_canonical(&json!({ "messages": items, "terminal_state": format!("{terminal_state:?}") }))
}

pub(crate) fn render_cache_decision(
    prev: Option<&CachedRender>, fp: &str, version: u32, mode_tag: &str,
    render: impl FnOnce() -> Vec<Value>,
) -> (Vec<Value>, bool, &'static str) {
    if let Some(p) = prev {
        if p.renderer_version != version { return (render(), false, "version_mismatch"); }
        if p.mode_tag != mode_tag { return (render(), false, "mode_mismatch"); }
        if p.content_fp != fp { return (render(), false, "fp_mismatch"); }
        return (p.bytes.clone(), true, "hit");
    }
    (render(), false, "miss")
}
```

`agent/mod.rs`: `type TurnRenderCache = Arc<tokio::sync::RwLock<HashMap<String, HashMap<String, CachedRender>>>>;` + field `turn_renders: TurnRenderCache`. `construct.rs`: init `turn_renders: Arc::new(tokio::sync::RwLock::new(HashMap::new()))` in both ctor paths (struct literal enforces both). The debug re-render assertion and the hit/miss/`fp_mismatch` `info!`/`debug!` logging live at the call site (Task 7), not in the pure helper — keep the helper pure so it stays unit-testable, exactly as `core_cache_decision` does.

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib turn_render_cache 2>&1 | tail -2
git add src/agent/loop/turn_render_cache.rs src/agent/loop/mod.rs src/agent/mod.rs src/agent/construct.rs
git commit -m "feat(pillar-b): per-turn render cache + content_fp (denylist embedding/importance/created_at, id retained)"
```

---

### Task 6: Eviction, archived-region budget, and the in-memory anchor

**Files:**
- Create: `src/agent/loop/turn_eviction.rs` (the budget formula + the pure "which turns to keep" decision)
- Modify: `src/agent/mod.rs` (`TurnAnchorMemory` type alias + field, mirroring `window_keep_from_tracker`)
- Modify: `src/agent/construct.rs` (init in both ctor paths)
- Modify: `src/agent/loop/mod.rs` (register `mod turn_eviction;`)

**Context (spec §Eviction).** Budget is computed over the **evictable region only**:

```
archived_budget = (context_budget − core − tool_defs − task_tail_estimate
                   − current_turn_reserve − output_reserve) × (1 − safety_margin)
low_water = 60% of archived_budget
```

When the archived estimate exceeds `archived_budget`, evict oldest WHOLE turns until at or below `low_water`, advance the anchor, log via `Window decision`. Estimation uses provider-equivalent serialized renderings (token estimates over the FINAL rendered Archived content, not raw message lengths) with a 10% safety margin. **Degenerate case:** if the non-evictable region alone ≥ context budget, carry ZERO archived turns (`warn!` naming the overflowing components); never truncate inside a turn (that's Pillar C). The anchor is in-memory (accept one re-prefill per restart). Reuse `estimate_tokens`, `model_context_budget`, `CONTEXT_RESPONSE_RESERVE_TOKENS` from `memory/context_window.rs`.

- [ ] **Step 1: Write the failing tests** (`turn_eviction.rs`)

```rust
struct RenderedTurn { turn_seq: i64, est_tokens: usize } // test stand-in for an archived rendering

#[test]
fn archived_budget_and_low_water() {
    let b = archived_budget(/*context*/ 32000, /*core*/ 4000, /*tools*/ 3000,
                            /*tail*/ 2000, /*current_reserve*/ 4000, /*output_reserve*/ 1536,
                            /*safety_margin*/ 0.10);
    // evictable = 32000 - (4000+3000+2000+4000+1536) = 17464; * 0.90 = 15717.6 -> 15717
    assert_eq!(b, 15717);
    assert_eq!(low_water(b), (b as f64 * 0.60) as usize); // 9430

}

#[test]
fn evict_oldest_until_low_water() {
    let budget = 1000usize;
    let turns = vec![ // oldest first
        RenderedTurn{turn_seq:1, est_tokens:400},
        RenderedTurn{turn_seq:2, est_tokens:400},
        RenderedTurn{turn_seq:3, est_tokens:400},
        RenderedTurn{turn_seq:4, est_tokens:400}, // total 1600 > budget 1000
    ];
    // low_water = 600. Evict oldest whole turns until <= 600.
    // low_water(1000) = 600. kept starts 1600; evict oldest until kept <= 600:
    // 1600 → 1200 (evict t1) → 800 (evict t2) → 400 (evict t3, now <= 600, stop).
    let plan = plan_eviction(&turns, budget);
    assert_eq!(plan.evicted_count, 3);
    assert_eq!(plan.kept_est_tokens, 400);
    assert!(plan.kept_est_tokens <= low_water(budget));
    assert_eq!(plan.new_anchor_turn_seq, 4); // oldest kept turn (turns[3])
    assert!(!plan.degenerate);
}

#[test]
fn degenerate_zero_archived_when_non_evictable_exceeds_budget() {
    // context fully consumed by core+tools+tail+reserves -> archived_budget == 0
    let b = archived_budget(8000, 5000, 2000, 1500, 1000, 1536, 0.10);
    assert_eq!(b, 0);
    let plan = plan_eviction(&[RenderedTurn{turn_seq:1, est_tokens:100}], b);
    assert_eq!(plan.kept_est_tokens, 0, "zero archived turns carried");
    assert!(plan.degenerate, "degenerate flag set for warn! at the call site");
}
```

- [ ] **Step 2: RED** — `cargo test --lib turn_eviction 2>&1 | tail -5`.

- [ ] **Step 3: Implement**

```rust
//! Pillar B: whole-turn eviction against the archived-region budget. Spec §Eviction.
pub(crate) fn archived_budget(context_budget: usize, core: usize, tools: usize,
    tail: usize, current_reserve: usize, output_reserve: usize, safety_margin: f64) -> usize {
    let evictable = context_budget
        .saturating_sub(core + tools + tail + current_reserve + output_reserve);
    ((evictable as f64) * (1.0 - safety_margin)) as usize
}
pub(crate) fn low_water(archived_budget: usize) -> usize { (archived_budget as f64 * 0.60) as usize }

pub(crate) struct EvictionPlan {
    pub new_anchor_turn_seq: i64,  // turn_seq of the OLDEST kept turn
    pub kept_est_tokens: usize,
    pub evicted_count: usize,
    pub degenerate: bool,          // non-evictable >= budget -> zero archived, warn at call site
}

/// `turns` ordered oldest→newest, each with its final-Archived est_tokens.
pub(crate) fn plan_eviction(turns: &[RenderedTurn], archived_budget: usize) -> EvictionPlan {
    let total: usize = turns.iter().map(|t| t.est_tokens).sum();
    if archived_budget == 0 {
        let anchor = turns.last().map(|t| t.turn_seq + 1).unwrap_or(0); // keep nothing
        return EvictionPlan { new_anchor_turn_seq: anchor, kept_est_tokens: 0,
                              evicted_count: turns.len(), degenerate: true };
    }
    if total <= archived_budget {
        let anchor = turns.first().map(|t| t.turn_seq).unwrap_or(0);
        return EvictionPlan { new_anchor_turn_seq: anchor, kept_est_tokens: total,
                              evicted_count: 0, degenerate: false };
    }
    // Over budget: evict oldest whole turns until kept estimate <= low_water.
    let lw = low_water(archived_budget);
    let mut kept: usize = total; let mut evicted = 0; let mut anchor = turns.first().map(|t| t.turn_seq).unwrap_or(0);
    for t in turns {
        if kept <= lw { break; }
        kept -= t.est_tokens; evicted += 1;
        anchor = t.turn_seq + 1; // tentative; corrected below to the next kept turn's seq
    }
    if let Some(first_kept) = turns.get(evicted) { anchor = first_kept.turn_seq; }
    EvictionPlan { new_anchor_turn_seq: anchor, kept_est_tokens: kept, evicted_count: evicted, degenerate: false }
}
```

`agent/mod.rs`: `type TurnAnchorMemory = Arc<tokio::sync::RwLock<HashMap<String, i64>>>;` + field `turn_anchors: TurnAnchorMemory` (session_id → anchor turn_seq). `construct.rs`: init both paths. Anchor **initialization** (cold start / restart) is a Task-7 concern that consumes `plan_eviction`: walk turns newest→oldest accumulating est_tokens, stop at the last whole turn fitting the archived budget — implement that walk in Task 7 against `get_turns_from_anchor("…", 0)` (bounded by the budget). State the accepted "one re-prefill per restart" in the Task-7 log.

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib turn_eviction 2>&1 | tail -2
git add src/agent/loop/turn_eviction.rs src/agent/loop/mod.rs src/agent/mod.rs src/agent/construct.rs
git commit -m "feat(pillar-b): whole-turn eviction plan + archived-region budget/low-water + anchor tracker"
```

---

### Task 7: Integrate into `message_build_phase` — replace the age ladder

> ⚠️ **HIGHEST-RISK TASK.** This deletes the age ladder and rewires payload assembly atomically. It consumes Tasks 1–6. The Pillar A regions (core at index 0, `[Task Context]` tail at boundary−1, final tool sort) MUST be preserved exactly.

**Files:**
- Modify: `src/agent/loop/message_build_phase.rs`
- Modify: `src/agent/loop/sliding_window.rs` (delete `calculate_window_size`; keep `summarize_tool_result` — now called by `turn_render`)
- Modify: `src/memory/context_window.rs` (scope `fit_messages_with_source_quotas` to the current-turn region; delete dead `fit_messages_to_budget`)
- Modify: `src/agent/loop/turn_context.rs` if `load_recent_history` is replaced/removed there

**The new build flow (replaces the fetch → age-collapse → sliding-window → JSON-conversion stages, `message_build_phase.rs` ~:262–648 and ~:817–938):**

1. **Anchor resolve.** Read `agent.turn_anchors[session_id]`; if absent (cold start/restart), initialize by walking `get_turns_from_anchor(session_id, 0)` newest→oldest, accumulating each turn's Archived est_tokens, stopping at the last turn that fits `archived_budget(...)`; set the anchor to that turn's `turn_seq` and `info!` "anchor initialized (cold start)" noting the accepted one-re-prefill.
2. **Fetch** `let turns = agent.event_store.get_turns_from_anchor(session_id, anchor).await?;` (Task 3). The current turn is the last `FetchedTurn` whose `turn_id == current_turn_id`.
3. **Evict.** Render each archived turn (step 4), estimate tokens, `plan_eviction(...)` (Task 6). If it advances the anchor, update `agent.turn_anchors`, drop the evicted turns, and `info!(… , "Window decision")` with the new fields (below). Degenerate → `warn!` naming overflowing components, zero archived turns.
4. **Render archived turns** (all but the current): for each, derive `terminal_state` (look up the turn's `TaskEnd` status by `turn_id` — add an event-store helper `terminal_state_for_turn(session_id, turn_id)` or fold it into `get_turns_from_anchor`'s return; `None` → `Interrupted`), compute `content_fp`, consult `agent.turn_renders[session_id][turn_id]` via `render_cache_decision` (Task 5), render on miss with `render_turn(.., Archived{terminal_state}, RENDERER_VERSION)`, store, and emit hit `debug!` / miss·`fp_mismatch` `info!`. In `cfg(debug_assertions)`/tests, always re-render and `assert_eq!` against the cached bytes (nondeterminism = panic).
5. **Render the current turn** with `render_turn(.., Current, RENDERER_VERSION)` (append-only, full).
6. **Assemble** `messages = [archived_turn_0 .., archived_turn_k .., current_turn ..]` then keep the existing Pillar A tail insertion (tail at boundary−1, before the current user message) and core insertion (index 0). The current turn's tool chain + transient suffix (execution checkpoints `:1248`, one-shot directives `:1293`) stay as today.
7. **Current-region fitting only.** `fit_messages_with_source_quotas` now receives ONLY the current-turn slice (everything after the last archived turn) and a current-turn budget; it must never touch archived turns (they are whole-turn-evicted, never trimmed — spec invariant 3). Emit `Prefix mutation reason=history_fitting` when it actually drops anything (Task 8 finalizes the line).
8. **Keep** `collapse_repeated_tool_errors` and the empty-response retry rebuild (current-turn only; Task 8 adds their `Prefix mutation` lines). **Keep** the final `sort_tool_definitions_by_name` before `MessageBuildData`.

**Deletions (spec §Rendering, "absorbed and deleted"):**
- Prior-1/Prior-2 age collapse block (`message_build_phase.rs` ~:390–463) incl. `prior_1_start`, `prior_1_tool_ids`, the `stage="age_collapse"` fingerprint, and the inline Prior-1 summarization at ~:828–848 (now in `render_turn`).
- The adaptive sliding-window block (~:501–648) incl. `skeleton_pairs`, `calculate_window_size`, the idle-gap reset, the `keep_from` filter.
- `current_user_injected` synthetic-user path (~:278–315) and its flag — the whole-turn fetch always contains the current user message.
- Index-based identity-preserve bypass (`identity_preserve_indices`, ~:336–352 + its uses at ~:424/:628) — identity is now handled at turn granularity inside `render_turn` (verbatim survival).
- The message-count trim safe-collapse fallback (~:429–463) — the `last_user_pos == None` case cannot occur with turn-anchored fetch.
- `history_limit` (~:262) and the `load_recent_history` call (replace with the anchored fetch).

**`window_keep_from_tracker`** can be retired or repurposed; the new boundary signal is the anchor (`turn_anchors`). If removing the tracker is invasive, leave the field unused with a deprecation comment and remove its writes — note the choice.

- [ ] **Step 1: Failing build-phase tests** (beside existing `message_build_phase` tests, using the existing test-agent harness). Assert on the built `messages` Vec / `MessageBuildData`:
  1. **Archived turns are whole and in Archived form:** after two completed turns, building a third turn yields archived renderings of turns 1 and 2 (user full, assistant truncated/placeholder, tool results summarized) positioned between message zero (core) and the `[Task Context]` tail.
  2. **Cross-turn archived stability:** the rendered bytes of archived turn 1 are byte-identical when built in turn 2 vs turn 3 (render-cache hit; no `fp_mismatch`).
  3. **Current turn is full/append-only:** the current turn's messages carry full content (not summarized).
  4. **Tail + core position preserved:** exactly one system message starts with `TASK_CONTEXT_TAIL_MARKER` at boundary−1; message zero equals the core bytes.
  5. **Eviction advances the anchor:** with a tiny archived budget (inject a small `context_window_config` budget), building after several turns evicts the oldest whole turns and emits `Window decision` with `turns_evicted > 0`; no archived turn is partially trimmed.
  6. **Late write re-renders one turn:** appending a tool message under an already-archived `turn_id` between builds flips that turn's `content_fp` → exactly one `fp_mismatch`/re-render, localized to that turn; the other archived turn stays byte-identical.
  7. **No synthetic user / no age_collapse:** assert the build no longer produces a `synthetic-user-*` id and no `stage="age_collapse"` fingerprint is emitted.

- [ ] **Step 2: Implement** the new flow and perform the deletions. Re-locate every span by symbol before editing. Keep the diff reviewable: do the deletions and the new assembly in one coherent pass (an intermediate half-deleted state won't compile).

- [ ] **Step 3: Existing-test fallout.** Run `cargo test --lib message_build_phase 2>&1 | tail -5`, `cargo test --lib context_window 2>&1 | tail -5`, and the full `cargo test --lib 2>&1 | tail -5`. Update tests that asserted the OLD layout: anything checking Prior-1/Prior-2 behavior, `current_user_injected`, `keep_from`/`window_size`, the message-count trim, or `load_recent_history`. `integration_tests` that assert on history shape (e.g. the part_10 summary-position test, and any asserting on age-collapsed tool results) must be updated to the turn-anchored layout — find them via `rg -n 'age_collapse|keep_from|window_size|current_user_injected|Prior 1|Prior 2' src/`. List every updated test in the task report with a one-line reason; do not silently weaken an assertion.

- [ ] **Step 4: Commit**

```bash
git add src/agent/loop/message_build_phase.rs src/agent/loop/sliding_window.rs src/memory/context_window.rs src/agent/loop/turn_context.rs <updated test files>
git commit -m "feat(pillar-b): turn-anchored fetch+render+evict replaces the age ladder in message build"
```

---

### Task 8: `Prefix mutation reason=<mechanism>` logging

**Files:**
- Modify: `src/agent/loop/message_build_phase.rs` (history-fitting overflow + empty-response retry rebuild call sites)
- Modify: `src/agent/loop/loop_utils.rs` (`collapse_repeated_tool_errors` — emit when it collapses ≥1)

**Context (spec §Observability + invariant 3).** Three retained mechanisms rewrite stable-region bytes within a task and are excluded from the prefix invariant ONLY if each emits an attribution line: `collapse_repeated_tool_errors`, current-region history-fitting overflow, and the empty-response retry rebuild. Without these lines a stable-region re-evaluation is an unattributed prefix break — which the live gate (Task 11) reads as a bug.

- [ ] **Step 1: Write the failing tests.** Where a log-capture harness exists, assert the line fires; otherwise (matching Pillar A's approach) assert via the return value the line is keyed off — e.g. `collapse_repeated_tool_errors` returns the collapse count; a test asserts count>0 on a repeated-error fixture, and you wire the `info!` to fire on `count > 0`. For history-fitting, assert `fit_messages_with_source_quotas` reports whether it dropped anything (add a small returned `dropped: usize` if not already exposed) and the call site logs on `dropped > 0`.

- [ ] **Step 2: Implement.** At each site emit `info!(session_id, reason = "<mechanism>", "Prefix mutation")` with `reason ∈ {repeated_tool_error_collapse, history_fitting, empty_response_retry}`. Fire only when the mechanism ACTUALLY mutates (non-zero effect), so quiet turns stay silent.

- [ ] **Step 3: GREEN + commit**

```bash
cargo test --lib message_build_phase 2>&1 | tail -2
git add src/agent/loop/message_build_phase.rs src/agent/loop/loop_utils.rs
git commit -m "feat(pillar-b): Prefix mutation reason logging for the three retained stable-region mutators"
```

---

### Task 9: Attribution script + provider/integration prefix assertions

**Files:**
- Modify: `scripts/cache-attribution.py` (commit with `git add -f`)
- Modify: `src/providers/openai_compatible.rs` (test module — the cross-turn invariant assertions)
- Add tests: `src/integration_tests/` (a new `part_*` or extend an existing one — the 4 cross-turn invariants + identity regression)

**Context (spec §Testing + exit criteria).** Cross-turn cache-prefix assertions apply ONLY to the OpenAI-compatible adapter (anthropic/google hoist system content — determinism only, already covered by Pillar A Task 8). The four invariants, asserted on the converted message SEQUENCE (element-wise), not serialized JSON bodies.

- [ ] **Step 1: cache-attribution.py.** Add an `eviction (expected)` cause: a `prefix_hash_archived` flip that pairs with a `Window decision` line (anchor advanced) is expected, not `pre_boundary_changed_unattributed`. Add a `prefix_mutation (expected)` cause keyed off `Prefix mutation reason=…`. Add a `late_write_rerender (expected)` cause keyed off render-cache `fp_mismatch`. Extend the self-test with fixture lines for each; assert old behavior unchanged when the lines are absent (back-compat). Run `python3 scripts/cache-attribution.py --self-test 2>&1 | tail -2` → `PASS`.

- [ ] **Step 2: Integration invariants** (OpenAI adapter; reuse the body-builder unit seam from Pillar A Task 8). Drive the real agent loop / build path across turns with the mock provider:
  1. within a task, iteration k+1's stable region extends iteration k element-wise (transient suffix — checkpoints/one-shot directives — identified and excluded by the harness); each mutator path instead emits its `Prefix mutation` line;
  2. across two turns, `core + archived[..N-1]` elements are byte-identical, and `archived[N]` is byte-stable in a third turn;
  3. storing a fact between turns changes the tail element only (core + archived identical);
  4. a skills-catalog change between turns produces exactly one `Core prompt invalidated component=skills_catalog` and new core bytes (this is Pillar A behavior re-verified end-to-end under the new history mechanism).

- [ ] **Step 3: Identity regression.** Run the existing identity/security integration suites against archived-form history (build a multi-turn fixture where an identity-critical statement falls into an archived turn; assert it survives verbatim in the built payload). Find the suites via `rg -n 'identity|critical_identity' src/integration_tests/`.

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib 2>&1 | tail -3
python3 scripts/cache-attribution.py --self-test 2>&1 | tail -2
git add -f scripts/cache-attribution.py
git add src/providers/openai_compatible.rs src/integration_tests/<files>
git commit -m "test(pillar-b): cross-turn prefix invariants + identity regression; attribution understands eviction/mutation/late-write"
```

---

### Task 10: Full verification + gate

**Files:** none — verification.

- [ ] **Step 1:** `cargo test --lib 2>&1 | tail -3` — green except the known-exempt `base_tool_registry_names_match_built_schema_names`. List any other failure explicitly; do not wave it through.
- [ ] **Step 2:** `cargo fmt --check && cargo clippy --all-features -- -D warnings 2>&1 | tail -3` — clean (the 3 db_probe bin clippy errors are the known exception; if the gate must be all-green, scope with `--lib`).
- [ ] **Step 3:** Spec §Testing sweep — confirm each maps to a passing test: render golden bytes per mode incl. crashed-turn + identity (Task 4); cache fp/version/mode mismatch (Task 5); the 4 cross-turn invariants (Task 9); identity regression (Task 9); per-adapter determinism (Pillar A Task 8, still green).
- [ ] **Step 4:** Deletion sweep — `rg -n 'calculate_window_size|skeleton_pairs|current_user_injected|identity_preserve_indices|age_collapse|Prior 1|Prior 2|history_limit|fit_messages_to_budget' src/` returns only historical comments / the renamed call sites, no live age-ladder code.
- [ ] **Step 5:** Migration check — on a fresh temp DB the `turn_id` migration applies cleanly; on a copy of a pre-migration DB (or a test simulating legacy rows) the NULL-turn rows hydrate and are excluded from reconstruction without panicking.

---

### Task 11: Live measurement run (A/B-gate)

**Files:** none — operational; requires the user's environment.

- [ ] **Step 1: Preconditions** — `tool_filter_enforce = false` (already set); daemon started manually under `caffeinate -i` with `RUST_LOG="info,aidaemon::agent::loop::message_build_phase=debug"`, stdout → `/tmp/aidaemon-attribution-run.log` (launchd `ai.aidaemon` booted out for the run, restored after via `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.aidaemon.plist`). Rebuild first (`cargo build`). Control MCP-trigger variance (disable query-triggered MCP injection or use turns verified to keep emitted tool membership identical); record the mode.
- [ ] **Step 2: Offsets** — record `llama-from-line` and `daemon-from-line` at idle (the established procedure).
- [ ] **Step 3: Protocol** — 10 fresh distinct turns (file/script lifecycle in a NEW scratch-dir name) in a single session; extend if observed breaks < 20.
- [ ] **Step 4: Analysis** — `python3 scripts/cache-attribution.py --daemon-log <segment> --session "<session>" --llama-from-line <N>`.

  **PASS requires (spec exit criteria 1–2):**
  - criterion 1 (within-task system stability) stays PASS;
  - every cross-turn `prefix_hash_system` flip pairs with a `Core prompt invalidated` line (expected: zero in a quiet run);
  - every `prefix_hash_archived` flip pairs with a logged eviction (`Window decision`), a `Prefix mutation` line, or a render-cache `fp_mismatch` (late write / late completion record) — an archived flip with NONE of these is an archived-region bug;
  - tail-only flips (`tail_hash` changed, `prefix_hash_archived` stable) reported expected;
  - `tool_defs_hash` cross-turn stable within the run (no in-run flip; the §Pillar A anti-pattern forbids per-turn roster gating). Force-text turns are NOT special on the OpenAI adapter (tool defs retained, `tool_choice=none`).
  - **Target (not a hard gate):** median turn-start evaluated tokens drop ≥80% vs the 15,565 post-C baseline (absolute bound ≈ archived[N] + tail + new user ≈ 4–5k). The first valid post-A/B re-run establishes the measured bound; set the gating regression threshold from it.

- [ ] **Step 5: Record + changelog + commit docs** — results table appended to this plan; spec §Pillar B gains a "Measured post-B" note; CHANGELOG [Unreleased] updated with the measured numbers (the Pillar A entry's "Live A-gate measurement … pending" is superseded by the combined A/B measurement). Commit docs with `git add -f` for the spec/plan files.

---

## Out of scope (per spec §Out of scope)

- Durable render/anchor persistence across daemon restarts (the plan accepts **one re-prefill per restart per session**; persisting the anchor is the named upgrade path if warm llama.cpp KV across restarts becomes a requirement).
- Prompt-cache optimization for the Anthropic/Google adapters (deterministic serialization only — they hoist the system tail).
- Queued-message durability across restarts.
- Multi-session contention on the shared llama-server slot (`--parallel 1` stands; interleaved sessions still evict each other's KV).
- Any further per-call payload reduction beyond turn-anchoring (tail compaction, schema slimming) — that is Pillar C, already shipped; the degenerate zero-archived case defers to it.
