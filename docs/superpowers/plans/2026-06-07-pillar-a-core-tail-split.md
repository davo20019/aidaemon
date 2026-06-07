# Pillar A: Stable Core / Task Context Tail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze message zero into a byte-stable core (recompiled only on logged, component-attributed input changes) and move all volatile per-task context into a task context tail message placed before the current turn — making `prefix_hash_system` cross-turn stable and every remaining prefix flip attributable.

**Architecture:** Per `docs/superpowers/specs/2026-06-06-cross-turn-prefix-stability-design.md` §Pillar A, §Payload layout, §Observability. `build_system_prompt_for_message` splits into `build_core_prompt(core_inputs)` (component-hashed, session-cached, pure) and `build_context_tail(per_task_inputs)` (timestamp, session context, session summary, query-ranked memories, matched skill content, speaker context, resume checkpoint — compiled once per task). The tail is a system message inserted **before the current user message**; the session summary leaves message index 1. The Phase 0 fingerprint gains `tail_hash` and `prefix_hash_archived`; `session_summary_hash` retires by reporting empty.

**Tech Stack:** Rust; existing modules `src/agent/runtime/system_prompt.rs`, `src/agent/loop/message_build_phase.rs`, `src/agent/loop/prefix_fingerprint.rs`; `scripts/cache-attribution.py`.

**Baselines (comparators from the post-C run, 2026-06-07):** median payload 16,238 tokens; median turn-start evaluated 15,565 tokens; system prompt 35,104 bytes (of which the session-context block alone measured ~10.6k pre-C); `tool_defs_hash` stable. Pillar A's success is *attribution* (exit criterion 2 of the spec) plus partial turn-start reduction from core reuse; the ≥80% target lands with Pillar B.

**Plan revisions (post-review, 2026-06-07):** reconciled against two code-verified reviews (this agent + @cursor). Changes: Task 6 deletes the **fit-stage** `[Conversation summary]` insertion (spec §Tail — blocking gap); Task 1's fixture inserts the tail before the current user message via `boundary_pos`; bootstrap integration path corrected to `src/agent/loop/bootstrap/run.rs`; Task 3 module registration corrected to the `#[path]` site; marker-visibility note in Task 6. **From @cursor (pass 1):** Task 1 extends `LlmCallData` event fields (not just the log line); Task 3/7 originally proposed `base_tool_defs` plus the production `tool_filter_enforce` decision; Task 6/7 specify `BootstrapData`/`MessageBuildInputs` plumbing and within-task tail reuse; Task 6 clarifies the `session_summary` stage-hash repurpose and `part_10` fallout; Task 2 covers the attribution stage ladder; spawn/depth->0 scope note added. **From @cursor (pass 2, implementer nits):** `test_core_inputs()` made `pub(crate)` for cross-module reuse; Task 1 names the `LlmCallData` construction sites (`store.rs`, `llm_phase.rs`, payload test); `assemble_core_inputs()` helper added in Task 4 to dedupe Task 6/7 assembly; bootstrap return shape spelled out (`(core_bytes, tail, active_skill_names)`); the vacuous tool-order test repurposed to skills/specialists determinism and the real tool-array canonical-order assertion moved to Task 8; weak `--lib bootstrap` test filter corrected. **From @codex (pass 3, data-flow blockers — these reopened the plan):** ⚠️ `base_tool_defs` is NOT session-static (it is `user_text`/MCP-trigger built + per-turn restricted) — Task 3 now creates a new session-static `core_tool_roster` accessor (+ Task 7 regression test 4: distinct queries → identical core); summary load reordered before tail construction (was `run.rs:593`, after the `:550` prompt build); budget accounting extended to reserve core + tail (`message_build_phase.rs:1156`); tool-array name-sort moved UPSTREAM (providers emit verbatim, `openai_compatible.rs:235`); `assemble_core_inputs` takes explicit stable inputs, NOT `&BootstrapData` (which is built at `:604`, after the build site); `store.rs` added to Task 1 staging. **From @codex (pass 4, execution ownership):** the roster accessor and shared sort helper now land in Task 3 before Task 4 consumes them; Task 6 canonicalizes the final emitted subset immediately before `MessageBuildData` reaches fingerprint/provider code (not only at initial bootstrap); Task 7 includes both `Agent` constructor sites; the bootstrap-level ordering regression test lives in Task 6, leaving Task 8 provider-only. **Final review:** Tasks 5 and 6 are one atomic implementation/commit unit, and Task 10 explicitly controls or reports MCP-triggered emitted-tool membership.

**Execution caveats (read first):**
- The working tree hosts concurrent workstreams. Reference code by SYMBOL, re-locate before editing (`rg -n 'fn build_system_prompt_for_message' src/`); never `git add -A`; stage only your hunks.
- **Per-commit gate:** CLAUDE.md requires `cargo fmt && cargo clippy --all-features -- -D warnings && cargo test` to pass before EVERY commit, not only at Task 9. Do not create intermediate commits with production functions that have no call site; Tasks 5 and 6 are therefore one atomic commit.
- **Core roster needs a NEW session-static source — NOT `base_tool_defs`.** ⚠️ Correction (found in code review): `base_tool_defs` is **not** session-static. It is built at `bootstrap/run.rs:344` from `agent.tool_definitions_with_capabilities(user_text)`, which does per-turn MCP-trigger matching (`registry.match_tools(user_message)`, `tool_defs.rs:103`), and then has per-turn restrictions applied (channel-visibility retain, `restrict_to_personal_memory_tools`, `restrict_untrusted_external_reference_tools`, `run.rs:356-365`). Hashing it into the core would invalidate the core on ordinary query changes — exactly the spec's per-turn-gating anti-pattern. **Resolution (Task 3):** define a genuinely session-static `core_tool_roster` = the registered tool set for the session's `(user_role, channel_visibility_class)`, computed **without** `user_text` (no MCP-trigger matching) and **without** the per-turn personal-memory/untrusted restrictions. Add `agent.session_static_tool_roster(role, visibility) -> Vec<(name, schema)>` rather than reusing the bootstrap `base_tool_defs`. **The static source already exists structurally:** the `for tool in &self.tools` loop in `tool_definitions_with_capabilities` (`tool_defs.rs:76-99`) is independent of `user_message`; the MCP stage (`:101-125`) and the per-turn restriction calls (`run.rs:356-365`) are exactly what the accessor must exclude. Channel-visibility filtering (`run.rs:347-353`) IS session-static and may stay; refactor the static loop into a shared helper so both `session_static_tool_roster` and `tool_definitions_with_capabilities` use it (no logic duplication). MCP-triggered tools and per-turn restrictions then affect ONLY the emitted provider `tool_defs` array, never the core hash. **Consequence to state in the gate (Task 10):** while per-turn MCP injection or `tool_filter_enforce = true` (default; `config.rs:2174` → `run.rs:417` sets `tool_defs = shadow_filtered`) is active, the *emitted* tool array still varies per turn, so `tool_defs_hash` flips and the end-to-end rendered prefix still breaks at the tool array. The **core**/`prefix_hash_system` stays stable (that is what Pillar A delivers); the end-to-end cache win additionally requires membership-stable emission (measurement forces `tool_filter_enforce = false`; per-turn MCP injection is the spec's "future passes must not reintroduce per-turn roster variance" item).
- Attribution/measurement runs require `tool_filter_enforce = false` (shadow) in config.toml — per-request roster gating is the spec's anti-pattern and breaks `tool_defs_hash` stability. Verify before any measurement.
- The launchd plist (`~/Library/LaunchAgents/ai.aidaemon.plist`) lacks `RUST_LOG`; measurement runs use a manually-started daemon with `RUST_LOG="info,aidaemon::agent::message_build_phase=debug"` (the Pillar C Task 8 procedure) or the plist gains the env var first.
- **Readiness & entry point:** **Start at Task 1** — it is pure observability (`tail_hash`/`prefix_hash_archived` + `LlmCallData`), has no dependency on the roster/tail work, and everything downstream depends on those regions being visible first. **Task 3 (including the `core_tool_roster` extraction consumed by Task 4) is the highest-risk step** — see the ⚠️ flag on Task 3. If anything forces a redesign mid-plan, it surfaces there; sequence it after Task 1/2 so the instrumentation is already live to measure it.

---

### Task 1: Fingerprint regions — `tail_hash`, `prefix_hash_archived`, retire `session_summary_hash`

**Files:**
- Modify: `src/agent/loop/prefix_fingerprint.rs`
- Modify: `src/agent/loop/llm_phase.rs` (the `Provider-call prefix fingerprint` info! line gains the two fields, AND the `LlmCallData { … }` construction at `llm_phase.rs:575,593` sets them)
- Modify: `src/events/payloads.rs` (`LlmCallData` struct at `:179` carries `prefix_hash_system`, `tool_defs_hash`, `session_summary_hash` today — add `tail_hash` and `prefix_hash_archived` as `Option<String>`; update the round-trip serialization test at `:860-890`)
- Modify: `src/events/store.rs` (constructs `LlmCallData` with explicit fields — set the two new fields there too)

The tail does not exist yet; these fields must land FIRST so every later task is observable. Until the tail ships (Task 6), `tail_hash` reports empty and `prefix_hash_archived == prefix_hash_pre_boundary`.

The fingerprint is logged AND persisted on the `LlmCallData` event — both surfaces must carry the new fields, or post-hoc DB attribution (`db_probe`) sees stale regions. `session_summary_hash` stays on `LlmCallData` for parser/back-compat but is written empty (retired), mirroring the struct field.

- [ ] **Step 1: Write the failing tests** (append to `prefix_fingerprint.rs` tests)

```rust
#[test]
fn tail_hash_separates_tail_from_archived_region() {
    // Payload: [system, history-a, history-b, TAIL, user]. The tail is
    // located by TASK_CONTEXT_TAIL_MARKER; prefix_hash_archived covers
    // [1..boundary) EXCLUDING the tail; tail_hash covers the tail alone.
    let mut messages = sample_messages();
    // The tail sits immediately BEFORE the current user message, INSIDE the
    // [1..boundary) region the fingerprint searches. In sample_messages() the
    // current user message ("current question") is at index 3 and its tool
    // chain follows it (indices 4-5) — the last message is a `tool`, NOT the
    // user message. Locate the insertion via boundary_pos so the fixture stays
    // correct (and inside the searched region) regardless of the sample shape.
    let tail_pos = boundary_pos(&messages, "current question");
    messages.insert(
        tail_pos,
        serde_json::json!({
            "role": "system",
            "content": format!("{TASK_CONTEXT_TAIL_MARKER}\n[Current Date & Time]\nstub"),
        }),
    );
    let fp = provider_call_fingerprint(&messages, "current question", &[], false);
    assert!(!fp.tail_hash.is_empty(), "tail must be located and hashed");

    // Changing ONLY the tail flips tail_hash and pre_boundary, but NOT archived.
    let mut tail_changed = messages.clone();
    tail_changed[tail_pos]["content"] =
        format!("{TASK_CONTEXT_TAIL_MARKER}\n[Current Date & Time]\nother").into();
    let fp2 = provider_call_fingerprint(&tail_changed, "current question", &[], false);
    assert_ne!(fp.tail_hash, fp2.tail_hash);
    assert_eq!(fp.prefix_hash_archived, fp2.prefix_hash_archived);
    assert_ne!(fp.hash_pre_boundary, fp2.hash_pre_boundary);

    // Changing an archived message flips archived but not tail.
    let mut hist_changed = messages.clone();
    hist_changed[1]["content"] = "mutated history".into();
    let fp3 = provider_call_fingerprint(&hist_changed, "current question", &[], false);
    assert_ne!(fp.prefix_hash_archived, fp3.prefix_hash_archived);
    assert_eq!(fp.tail_hash, fp3.tail_hash);
}

#[test]
fn no_tail_marker_means_empty_tail_hash_and_archived_equals_pre_boundary() {
    let messages = sample_messages();
    let fp = provider_call_fingerprint(&messages, "current question", &[], false);
    assert!(fp.tail_hash.is_empty());
    assert_eq!(fp.prefix_hash_archived, fp.hash_pre_boundary);
}

#[test]
fn session_summary_hash_is_retired() {
    // Field stays for parser compatibility but always reports empty.
    let messages = sample_messages();
    let fp = provider_call_fingerprint(&messages, "current question", &[], false);
    assert!(fp.session_summary_hash.is_empty());
}
```

- [ ] **Step 2: Run to observe RED**

Run: `cargo test --lib prefix_fingerprint 2>&1 | tail -3`
Expected: compile error (`TASK_CONTEXT_TAIL_MARKER`, `prefix_hash_archived`, `tail_hash` missing), then assertion failures after stubbing.

- [ ] **Step 3: Implement**

In `prefix_fingerprint.rs`:
```rust
/// Marker prefix for the task context tail message. SHARED between the tail
/// builder (system_prompt.rs) and this module — the same arrangement that
/// keeps SESSION_SUMMARY_MARKER from drifting. The tail builder re-exports
/// this constant; do not duplicate the literal.
pub(crate) const TASK_CONTEXT_TAIL_MARKER: &str = "[Task Context]";
```
Add struct fields `pub tail_hash: String` and `pub prefix_hash_archived: String` with doc comments quoting the spec's diagnosis rule (archived flip without Window decision / Prefix mutation / fp_mismatch = bug; tail-only flip = expected). In `provider_call_fingerprint`: locate the tail in `[1..boundary)` by content starting with the marker; `tail_hash` = hash of that message or `String::new()`; `prefix_hash_archived` = hash of `[1..boundary)` minus the tail message (equals `hash_pre_boundary` when no tail). Set `session_summary_hash: String::new()` unconditionally and update its doc comment ("retired by Pillar A; summary participates in tail_hash").

In `llm_phase.rs`, add to the fingerprint info! line: `tail_hash = %fp.tail_hash, prefix_hash_archived = %fp.prefix_hash_archived,`.

Wire the new fields into the persisted event too: add `tail_hash: Option<String>` and `prefix_hash_archived: Option<String>` to `LlmCallData` (`events/payloads.rs:179`) and set them at EVERY construction site — `llm_phase.rs` (the `LlmCallData { … }` at `:575/:593`, from `prefix_fp`), plus the explicit-field literals in `src/events/store.rs` and the round-trip test fixture in `events/payloads.rs:~860` (which constructs the struct with explicit values). Missing a site is a compile error (struct literal), so the compiler enforces completeness — but set `session_summary_hash` to empty/`None` at each, consistent with retirement.

- [ ] **Step 4: GREEN + sweep**

Run: `cargo test --lib prefix_fingerprint 2>&1 | tail -2` then `cargo test --lib llm_phase 2>&1 | tail -2`
Expected: all pass (existing `session_summary_hash` assertions in old tests updated to expect empty).

- [ ] **Step 5: Commit**

```bash
git add src/agent/loop/prefix_fingerprint.rs src/agent/loop/llm_phase.rs src/events/payloads.rs src/events/store.rs
git commit -m "feat(pillar-a): tail_hash and prefix_hash_archived fingerprint regions (log + LlmCallData); session_summary_hash retired"
```

---

### Task 2: cache-attribution.py parses the new regions

**Files:**
- Modify: `scripts/cache-attribution.py` (untracked-by-default: commit with `git add -f`)

- [ ] **Step 1: Extend the parser and attribution**

In `parse_daemon_log`, capture `tail_hash` and `prefix_hash_archived` from the fingerprint line. In `attribute()`: a pair where `prefix_hash_archived` is stable but `tail_hash` changed classifies as `tail_replacement (expected)` and never counts toward `pre_boundary_changed_unattributed`; an archived flip keeps the existing cause ladder. In the report, add a `tail-only flips (expected): N` summary line.

**Stage-attribution ladder (build-stage `Build stage pre-boundary fingerprint` lines).** The parser also consumes the per-stage `stage=` fingerprints. The `session_summary` stage is repurposed in Task 6 (the summary moves into the tail; see Task 6 Step 3): if the script keys a cause off `stage=session_summary` as "index-1 summary churn," update it so that signal now attributes to `tail_replacement (expected)` rather than an archived/pre-boundary cause. If the script only reads the provider-call fingerprint (not stage lines), note that explicitly so a later reader does not assume stage coverage exists. Either way, do NOT leave a `session_summary`-stage cause that fires as a false archived-region flip after the move.

- [ ] **Step 2: Extend the self-test**

Add fixture lines carrying the two new fields; assert: tail-only change → `tail_replacement (expected)`; archived change → existing causes; absent fields (old logs) → behavior unchanged (backwards compatible).

- [ ] **Step 3: Run**

Run: `python3 scripts/cache-attribution.py --self-test 2>&1 | tail -2`
Expected: `self-test: PASS`

- [ ] **Step 4: Commit**

```bash
git add -f scripts/cache-attribution.py
git commit -m "feat(pillar-a): attribution understands tail_hash/prefix_hash_archived; tail-only flips expected"
```

---

### Task 3: Core-input canonicalization and component hashing

> ⚠️ **HIGHEST-RISK TASK.** This task creates the `core_tool_roster` accessor before Task 4 consumes it. The `CoreInputs` fields must each map to a genuinely session-static source — the roster especially (see the Core-roster caveat: build from the `tool_defs.rs:76-99` static loop, NOT `base_tool_defs`). Before writing the hashing, **confirm each input is query-independent**: the channel-visibility filter stays (session-static) while the MCP-trigger stage and per-turn personal-memory/untrusted restrictions are excluded. Task 7's regression test 4 (distinct queries → identical core) is the end-to-end gate that proves this; if it cannot pass, the roster source is wrong and the fix belongs here, not downstream.

**Files:**
- Create: `src/agent/runtime/core_prompt.rs`
- Modify: `src/agent/mod.rs` — register via `#[path = "runtime/core_prompt.rs"] mod core_prompt;` next to the existing runtime module declarations (e.g. the `#[path = "runtime/system_prompt.rs"]` block around `src/agent/mod.rs:248`). There is NO `src/agent/runtime/mod.rs` — every runtime submodule is registered through a `#[path]` attribute in `src/agent/mod.rs`.
- Modify: `src/agent/tools/tool_defs.rs` — extract the query-independent registered-tool collection, add `session_static_tool_roster`, and add the shared name-sort helper used by Task 6.

- [ ] **Step 1: Write the failing tests** (same file, `#[cfg(test)]`)

Define ONE `test_core_inputs()` helper and reuse it across these tests and Task 4's golden tests — the `/* as above */` shorthand below must become real calls to it, not copy-pasted literals (drift between copies would silently weaken the attribution tests). Task 4's golden tests live in `system_prompt.rs`, a DIFFERENT module, so a private helper won't cross the boundary — declare it `#[cfg(test)] pub(crate) fn test_core_inputs() -> CoreInputs` in `core_prompt.rs` and `use crate::agent::core_prompt::test_core_inputs;` from the `system_prompt.rs` test module.

```rust
#[cfg(test)]
pub(crate) fn test_core_inputs() -> CoreInputs {
    CoreInputs {
        base_template: "T".into(),
        tool_roster: vec![("b".into(), "{}".into()), ("a".into(), "{}".into())],
        skills_catalog: vec![("s2".into(), "d2".into(), true), ("s1".into(), "d1".into(), true)],
        specialists: vec![("x".into(), "dx".into())],
        channel_rules: "R".into(),
        persona: "P".into(),
    }
}

#[test]
fn component_hash_is_order_insensitive_for_unordered_inputs() {
    let a = test_core_inputs();
    let mut b = a.clone();
    b.tool_roster.reverse();
    b.skills_catalog.reverse();
    assert_eq!(a.component_hashes(), b.component_hashes());
    assert_eq!(a.aggregate_hash(), b.aggregate_hash());
}

#[test]
fn changed_component_is_named() {
    let a = test_core_inputs();
    let mut b = a.clone();
    b.skills_catalog.push(("s3".into(), "d3".into(), true));
    let diff = a.component_hashes().diff(&b.component_hashes());
    assert_eq!(diff, vec!["skills_catalog"]);
}

#[test]
fn aggregate_hash_is_hash_of_component_hashes() {
    // Pin the construction so a future field addition cannot silently
    // bypass component attribution.
    let a = test_core_inputs();
    let ch = a.component_hashes();
    assert_eq!(a.aggregate_hash(), ch.aggregate());
}
```

- [ ] **Step 2: RED**

Run: `cargo test --lib core_prompt 2>&1 | tail -3` — compile failure, then stub failures.

- [ ] **Step 3: Implement**

```rust
//! Pillar A core-prompt inputs: canonicalization + component hashing.
//! Spec: 2026-06-06-cross-turn-prefix-stability-design.md §Pillar A.
//! Hash actual content inputs, never proxies; canonicalize unordered
//! collections (sort by name) BEFORE hashing. Provider tool-array ordering is
//! enforced upstream and asserted in Tasks 6 and 8. No timestamps, map
//! iteration, or env-dependent formatting.

#[derive(Clone, Debug)]
pub(crate) struct CoreInputs {
    pub base_template: String,
    /// (tool name, serialized schema) — sorted by name in canonical form.
    /// SOURCED FROM the session-static `core_tool_roster` (registered tools for
    /// the (role, channel-visibility) class, NO user_text/MCP-trigger gating,
    /// NO per-turn restrictions). NOT `base_tool_defs` (which is per-turn) and
    /// NOT the filtered `tool_defs` — see Execution caveats and Task 7.
    pub tool_roster: Vec<(String, String)>,
    /// (skill name, one-line description, enabled) — availability catalog
    /// only; matched skill CONTENT is tail-side.
    pub skills_catalog: Vec<(String, String, bool)>,
    /// (specialist kind, description).
    pub specialists: Vec<(String, String)>,
    pub channel_rules: String,
    pub persona: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ComponentHashes { /* one String per component, in a fixed array with names */ }
```
`component_hashes()` sorts the unordered collections, hashes each component with the existing `hash_canonical`-style SHA-256 (reuse `prefix_fingerprint::hash_canonical` if visible, else a local equivalent over canonical JSON); `aggregate()` hashes the concatenated component hashes; `diff()` returns the names of differing components in fixed order.

In `src/agent/tools/tool_defs.rs`, extract the existing registered-tool loop into a query-independent helper that returns validated OpenAI definitions plus capability metadata. `tool_definitions_with_capabilities(user_message)` calls that helper first, then performs its existing MCP-trigger composition unchanged. Add:

```rust
pub(super) fn session_static_tool_roster(
    &self,
    user_role: UserRole,
    visibility: ChannelVisibility,
) -> Vec<(String, String)>
```

For non-owners it returns an empty roster. For owners it uses only registered, currently available tools; applies the existing `PublicExternal` allowlist because visibility is session-static; excludes MCP trigger matching and all query-dependent policy/personal-memory/untrusted-reference restrictions; serializes each validated schema deterministically; and sorts by tool name.

Also add one shared helper:

```rust
pub(super) fn sort_tool_definitions_by_name(defs: &mut [Value])
```

It compares `Agent::tool_name_from_definition` and uses serialized definition bytes as a deterministic tie-breaker. Task 6 calls it both after bootstrap assembly and on the final effective provider subset immediately before `MessageBuildData` is returned.

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib core_prompt 2>&1 | tail -2
cargo test --lib tool_defs 2>&1 | tail -2
git add src/agent/runtime/core_prompt.rs src/agent/mod.rs src/agent/tools/tool_defs.rs
git commit -m "feat(pillar-a): CoreInputs canonicalization with component-attributed hashing"
```

---

### Task 4: Split `build_core_prompt` out of `build_system_prompt_for_message`

**Files:**
- Modify: `src/agent/runtime/system_prompt.rs`
- Modify: `src/agent/runtime/core_prompt.rs` (render fn lives with its inputs)

**Section disposition (from the spec's payload layout — this is the contract):**
- CORE: base prompt (identity, persona, behavioral/core rules, orchestrator- or direct-mode guidance, Tool Selection Guide table, coding/planning/behavior sections, `## Tools` pointer, CLI Agent Delegation/Availability, Built-in Channels, Scheduling, Response Completeness, Tool Result Reporting, Conversation Context, Self-Maintenance, Expertise rubric TEXT), Available Specialists block, channel/privacy rule set, skills availability catalog.
- TAIL (Task 5): `[Current Date & Time]`, session context block ("Current Session Activity"), session summary content, query-ranked facts/procedures ("Your Memory" recall), matched skill content, people/current-speaker context, resume checkpoint, API runtime context (profile names are runtime values), and the expertise *level selection* if it is rendered per-user.

- [ ] **Step 1: Write the failing purity/golden tests**

```rust
// render_core_prompt is a pure, synchronous fn over CoreInputs — use a plain
// #[test] (no tokio runtime needed; the same test_core_inputs() helper from Task 3).
#[test]
fn core_prompt_renders_identically_for_identical_inputs() {
    let inputs = test_core_inputs();
    let a = render_core_prompt(&inputs);
    let b = render_core_prompt(&inputs);
    assert_eq!(a, b, "core render must be deterministic");
    assert!(!a.contains("[Current Date & Time]"), "timestamp belongs to the tail");
    assert!(!a.contains(TASK_CONTEXT_TAIL_MARKER));
}

#[test]
fn core_prompt_is_order_insensitive_for_unordered_inputs() {
    // Determinism of the core BYTES under input reordering. Reorder the
    // unordered collections the core actually emits (skills catalog,
    // specialists); rendered bytes must not change.
    // NOTE: per spec §Pillar A the name-sorted canonical order binds the
    // PROVIDER TOOL ARRAY, not the `## Tools` prose (which is a selection
    // guide needing only determinism). So tool-array emission order is
    // asserted at the provider boundary in Task 8, NOT here — asserting it
    // against core bytes would be vacuous if the roster isn't emitted verbatim
    // into the core.
    let mut inputs = test_core_inputs();
    let a = render_core_prompt(&inputs);
    inputs.skills_catalog.reverse();
    inputs.specialists.reverse();
    assert_eq!(a, render_core_prompt(&inputs));
}
```

- [ ] **Step 2: RED, then implement `render_core_prompt(&CoreInputs) -> String`**

Mechanically: move the static format! sections from `build_base_system_prompt`/`build_system_prompt_for_message` into the new renderer, parameterized only by `CoreInputs`. Canonicalize the unordered collections (sort by name) inside `render_core_prompt` before emitting, so the rendered bytes are emission-order-stable. `build_system_prompt_for_message` now calls `render_core_prompt` for the core and keeps (for one task) appending the volatile sections after it — behavior-identical output until Task 5 splits the tail message. Existing prompt tests must stay green (the rendered concatenation is unchanged at this task's end).

Also add `assemble_core_inputs(...) -> CoreInputs` here (co-located with `render_core_prompt`), mapping each field per the Task 7 sources table. **Signature: explicit stable inputs, NOT `&BootstrapData`** — that struct is constructed at `run.rs:604`, after the prompt is built at `:550`, so it does not exist yet at the call site (code-review finding). Take `(user_role, &ChannelContext, &AppConfig/persona source, core_tool_roster, skills_catalog snapshot, specialists)` — i.e. the session-static values available where `build_system_prompt_for_message` runs. `tool_roster` comes from the new session-static `core_tool_roster` accessor (NOT `base_tool_defs`; see caveat). Tasks 6 and 7 both call this single assembler — do NOT duplicate assembly logic across the build phase and the cache hook.

- [ ] **Step 3: GREEN + full prompt-test sweep + commit**

```bash
cargo test --lib system_prompt 2>&1 | tail -2
cargo test --lib core 2>&1 | tail -2
git add src/agent/runtime/system_prompt.rs src/agent/runtime/core_prompt.rs
git commit -m "feat(pillar-a): render_core_prompt extracted; deterministic golden tests"
```

---

### Task 5: `build_context_tail` and the per-task tail message

**Files:**
- Modify: `src/agent/runtime/system_prompt.rs`

- [ ] **Step 1: Failing test**

```rust
#[tokio::test]
async fn context_tail_carries_all_volatile_sections_and_marker() {
    let tail = /* build via the new fn with fixture context */;
    assert!(tail.starts_with(TASK_CONTEXT_TAIL_MARKER));
    for needle in ["[Current Date & Time]"] {
        assert!(tail.contains(needle), "missing {needle}");
    }
    // Session summary content participates here, not at message index 1
    // (asserted end-to-end in Task 6's build-phase tests).
}

#[tokio::test]
async fn resume_checkpoint_renders_into_tail_not_core() {
    // Spec §Tail: the resume checkpoint MOVES from the core
    // (system_prompt.rs:772) to the tail. Assert it lands in the tail and is
    // ABSENT from render_core_prompt, so a later pass cannot "optimize" it
    // back into the core (which would make the core per-resume-state).
    let checkpoint = test_resume_checkpoint();
    let tail = /* build_context_tail with Some(&checkpoint) */;
    assert!(tail.contains(&checkpoint.render_prompt_section()));
    let core = render_core_prompt(&test_core_inputs());
    assert!(!core.contains("Resume checkpoint")); // adjust to the section's stable header
}
```

- [ ] **Step 2: Implement `build_context_tail(...) -> String`**

Move the volatile sections (Task 4's disposition list) into it; first line is `TASK_CONTEXT_TAIL_MARKER` (import the shared constant from `prefix_fingerprint`). **Relocate the resume-checkpoint injection out of the core** (`system_prompt.rs:772`, the `resume_checkpoint.render_prompt_section()` block) and into this tail builder — it is a move, not a copy (spec §Tail). Compiled once per task in the same bootstrap path that compiles the prompt today — no per-iteration recompute (within-task byte stability is invariant 3's stable region).

`build_context_tail` is exercised by tests here but not wired into the payload until Task 6, so the non-test build sees it as dead code. **Execute Tasks 5 and 6 as one atomic implementation unit and one commit.** Do not create an intermediate Task 5 commit with temporary `#[allow(dead_code)]`; write the Task 5 tests/builder, continue directly into Task 6 wiring, then run the combined gate and commit at Task 6 Step 4.

- [ ] **Step 3: Focused GREEN, then continue directly to Task 6**

```bash
cargo test --lib system_prompt 2>&1 | tail -2
```

Expected: the focused tests pass. Do not commit yet; Task 6 supplies the production call site and the atomic commit boundary.

---

### Task 6: Payload assembly — tail before current turn, summary leaves index 1

**Files:**
- Modify: `src/agent/loop/message_build_phase.rs`
- Modify: `src/agent/loop/main_loop.rs` (`BootstrapData` destructuring and `MessageBuildCtx` construction receive core + tail separately)
- Modify: `src/memory/context_window.rs` (delete the fit-stage summary insertion — see Step 2)
- Modify: `src/agent/loop/bootstrap/run.rs` (the compiled prompt is attached to the task at the `build_system_prompt_for_message` call, currently `src/agent/loop/bootstrap/run.rs:550`; re-locate via `rg -n 'build_system_prompt_for_message' src/`). NOTE: this is the `bootstrap/` directory, NOT the sibling `bootstrap_phase.rs` file — they coexist and are easy to confuse.
- Modify: `src/agent/loop/bootstrap/types.rs` (`BootstrapData` plumbing — see Step 2)

**State plumbing (do this first — the tests depend on it).** `BootstrapData` today carries a single `system_prompt: String` (`bootstrap/types.rs`, around field `:39`); message-build takes `system_prompt: &'a str` (`message_build_phase.rs:12`). The split needs two task-scoped values reaching message-build:
- `core_prompt_bytes: String` — message zero (rendered/cached in Task 7; here render directly);
- `task_context_tail: String` — compiled once per task, inserted at boundary − 1.

Replace `system_prompt` with these two fields on `BootstrapData` (and thread both through the message-build inputs struct that currently takes `system_prompt: &'a str`), or add the tail alongside a renamed `core_prompt`. The within-task loop must reuse the SAME `task_context_tail` string every iteration — assert byte identity across iterations (Step 1, test 4). `bootstrap/run.rs` populates both when it calls the core renderer + `build_context_tail`.

**Refactor boundary.** `build_system_prompt_for_message` today is `async fn … -> anyhow::Result<(String, Vec<String>)>` (system prompt + `active_skill_names`), bound at `bootstrap/run.rs:550` as `let (system_prompt, active_skill_names) = …`. After the split it returns `(core_bytes, task_context_tail, active_skill_names)` (the core render is sync but the tail assembly stays async — memory/skill fetches). Update the call site's destructuring and the `BootstrapData` population in the same hunk so no dead single-string concatenation path is left behind.

**Data-flow reorder (REQUIRED — the current order makes the tail impossible).** Four ordering/accounting facts in `bootstrap/run.rs` must change, or the split silently breaks:

1. **Summary loads AFTER the prompt today.** `build_system_prompt_for_message` runs at `run.rs:550`, but `get_conversation_summary(session_id)` runs at `run.rs:593` and `BootstrapData` is constructed at `run.rs:604`. The tail must CONTAIN the summary, so move the `session_summary` load (and any other tail inputs currently fetched after `:550`) to BEFORE tail construction, and pass the snapshot into `build_context_tail`. Verify nothing between `:550` and `:593` depends on ordering that the move would break.

2. **`assemble_core_inputs` cannot take `&BootstrapData`** — that struct does not exist yet at `:550`. It takes the explicit stable inputs available at the build site: `(role, &channel_ctx, &config/persona, core_tool_roster, skills_catalog snapshot, specialists)`. (Correcting the Task 4 signature note: NOT `&BootstrapData`.) Likewise the tail builder takes the explicit per-task snapshots (summary, facts, matched-skill content, people/speaker ctx, resume checkpoint), not `&BootstrapData`.

3. **Budget accounting must include the tail.** `compute_available_budget(model, system_prompt, tool_defs, …)` (`message_build_phase.rs:1156`) currently reserves space for the full compiled `system_prompt`. After the split, passing only `core_bytes` as `system_prompt` drops the tail's tokens from the reservation, so core + a large tail could exceed the model window. Update the budget call to account for BOTH core and tail (sum their `estimate_tokens`, or pass core+tail). The current-turn fitter (`fit_messages_*`, now summary-arg-free) trims the current region against this corrected budget.

4. **Name-sort the emitted roster upstream at BOTH required boundaries.** Call Task 3's `sort_tool_definitions_by_name` immediately after `base_tool_defs`/`tool_defs` are built (`run.rs:344-366`) so later retain/filter/widen operations begin from canonical order. More importantly, call it on `effective_tool_defs` in `message_build_phase.rs` immediately before constructing `MessageBuildData`. Later policy exposure, route-failsafe, connected-API exposure, and tool-widening paths can append definitions after the initial bootstrap sort; therefore the final pre-return sort is the authoritative guarantee that the provider-call fingerprint and every provider adapter receive the same canonical order. Providers remain order-preserving and do not sort.

- [ ] **Step 1: Failing build-phase tests** (place beside existing message-build tests)

Assert on the built `messages` Vec for a fixture task:
1. exactly one system message starts with `TASK_CONTEXT_TAIL_MARKER`, positioned immediately BEFORE the current user message (boundary − 1);
2. no message at index 1 starts with `[Session Summary]` and no message contains `[Conversation summary:` (the summary string now appears only inside the tail). `SESSION_SUMMARY_MARKER` is a private `const` in `prefix_fingerprint.rs:85` — the build-phase test cannot import it, so assert against the literal (as the existing test at `message_build_phase.rs:2021` does), or bump the const to `pub(crate)` if you prefer the symbol;
3. message zero equals `render_core_prompt(inputs)` bytes exactly (no volatile suffix);
4. within-task tail reuse: two consecutive build iterations of the same task produce a byte-identical tail message (the task-compiled `task_context_tail` is reused, not recompiled per iteration);
5. final emitted tool order: start with an unsorted roster, exercise a late append/widening case, and assert `MessageBuildData.tool_defs` is name-sorted. This is the bootstrap/message-build regression proving the sort occurs after mutations, not only at initial collection.

- [ ] **Step 2: Implement**

Message zero = cached core bytes (Task 7 wires the cache; here call render directly). Insert the tail message at the boundary before the current user message. Within-task iterations must NOT rebuild the tail — reuse the task-compiled string.

Canonicalize tools at both upstream boundaries described above. The final operation on `effective_tool_defs` before returning `MessageBuildData` must be:

```rust
Agent::sort_tool_definitions_by_name(&mut effective_tool_defs);
```

Do not sort inside provider adapters. `provider_call_fingerprint` and the selected adapter must observe the same ordered slice.

Delete BOTH summary insertion paths so the summary lives only in the tail (spec §Tail, "Single summary insertion point" — leaving either in place re-injects a summary near index 1 in the archived region and silently defeats Pillar A's tail/`prefix_hash_archived` stability):
- **Build-stage** `[Session Summary]` at `message_build_phase.rs:1230` — remove the injection site (keep the `[Session Summary]` literal/marker handling in the fingerprint, and the content now flows through the tail).
- **Fit-stage** `[Conversation summary: …]` — `fit_messages_to_budget` (`context_window.rs:197`) and `fit_messages_with_source_quotas` (`:303`) each insert a summary system message. Per spec, `fit_messages_*` keeps its current-turn message-fitting role but **no longer takes or emits a summary argument**: drop the `session_summary: Option<&str>` parameter and the `if let Some(summary)` insertion blocks from both functions, fix the `dropped` accounting that referenced `session_summary.is_some()`, and update the callers in `message_build_phase.rs` (and the fit-fn unit tests in `context_window.rs`) to stop passing it. Verify nothing else relies on fit injecting the summary (`rg -n 'fit_messages_to_budget|fit_messages_with_source_quotas' src/`).

- [ ] **Step 3: Stage-hash and existing-test fallout**

Run: `cargo test --lib message_build_phase 2>&1 | tail -3`, `cargo test --lib context_window 2>&1 | tail -3`, and `cargo test integration_tests 2>&1 | tail -3` (the build tests live in the `message_build_phase` module; `--lib message_build` may not match the filter, and the integration suite is its own test target).

Existing-test fallout to update to the new layout:
- the `context_window.rs` fit-stage tests asserting `[Conversation summary]` is injected (e.g. `context_window.rs:899`);
- `src/integration_tests/part_10.rs` asserts the compaction summary appears in the LLM context at its old position — update it to find the summary inside the task tail;
- any test asserting message-zero contains a volatile suffix section.

**Stage-hash semantics (correction).** The Phase 0 stage hashes are *extended, never redefined* (requirement 6), but the `session_summary` build stage (`message_build_phase.rs:1235`) currently exists specifically to fingerprint the index-1 summary insertion — which this task DELETES. So that stage does change meaning: either repurpose it to fingerprint the assembled tail (rename `stage = "tail"` and hash the tail message) or drop the `session_summary` stage and rely on the provider-call `tail_hash` for tail attribution. Update its comment accordingly and keep Task 2's attribution script in sync (Task 2 Step 1 stage-ladder note). No pre-boundary/system stage is removed.

- [ ] **Step 4: Commit**

```bash
git add src/agent/loop/message_build_phase.rs src/agent/loop/main_loop.rs src/agent/runtime/system_prompt.rs src/memory/context_window.rs src/agent/loop/bootstrap/run.rs src/agent/loop/bootstrap/types.rs
git commit -m "feat(pillar-a): task context tail precedes current turn; both summary insertion paths retired"
```

---

### Task 7: Core cache + `Core prompt invalidated component=` logging

**Files:**
- Modify: `src/agent/mod.rs` (Agent field)
- Modify: `src/agent/construct.rs` (initialize the cache in both `Agent` construction paths)
- Modify: `src/agent/loop/bootstrap/run.rs` (per-task hook — the `build_system_prompt_for_message` call site, ~`:550`; the `bootstrap/` directory, not `bootstrap_phase.rs`)

- [ ] **Step 1: Failing integration-style tests**

1. Two consecutive tasks, unchanged inputs → message zero bytes identical AND no `Core prompt invalidated` line (assert via a log-capture helper if available, else via cache state exposed `#[cfg(test)]`);
2. toggle one skill between tasks → exactly one invalidation naming `component=skills_catalog`, new core bytes;
3. store a fact between tasks → core bytes identical (facts are tail-side; this is spec §Testing item 3);
4. **different `user_text` between tasks (distinct queries, e.g. one that would MCP-trigger a tool and one that would not) → core bytes identical** — the direct regression test for the `core_tool_roster` fix (finding #1): the core roster must be query-independent. If this flips with `component=tool_roster`, the roster source is still per-turn.

- [ ] **Step 2: Implement**

`core_prompts: Arc<RwLock<HashMap<String, CachedCore>>>` on Agent where `CachedCore { aggregate: String, components: ComponentHashes, bytes: String }`. Per task bootstrap: assemble `CoreInputs` (cheap — names/strings already in memory), compute hashes, compare; on hit reuse bytes verbatim; on miss render, log `info!(session_id, component = %changed.join(","), "Core prompt invalidated")`, replace entry.

**CoreInputs sources (map each field to its origin so the wrong roster/value is never hashed):**

| Field | Source in `bootstrap/run.rs` context | Notes |
|---|---|---|
| `base_template` | the role/mode base prompt (`build_base_system_prompt` / `build_system_prompt_for_message` static sections) | per role + orchestrator/direct mode |
| `tool_roster` | **new `core_tool_roster` accessor** (registered tools for `(role, visibility)`, no user_text/MCP gating, no per-turn restriction) | NOT `base_tool_defs` (per-turn) and NOT filtered `tool_defs` — see caveat. Sort the static superset by name for hashing/core rendering; independently sort whatever per-turn subset is emitted to the provider (Task 6/8). |
| `skills_catalog` | active skill registry snapshot (names + one-line descriptions + enabled) | matched skill *content* is tail-side, not here |
| `specialists` | `SpecialistRegistry::llm_visible_kinds()` | same surface as the spawn schema |
| `channel_rules` | channel/privacy rule set for the session's visibility class (`channel_ctx`) | visibility-class scoped |
| `persona` | persona/identity config | static config |

If any of these is genuinely per-turn (it should not be after Pillar C), that is a design bug to surface, not to hash into the core — the `component=` log will expose it.

- [ ] **Step 3: GREEN + commit**

```bash
cargo test --lib core_prompt 2>&1 | tail -2 && cargo test core_cache 2>&1 | tail -2
# NOTE: `--lib bootstrap` is a weak filter — bootstrap tests live across
# bootstrap/{run,shortcuts,phase_impl,task_planning}.rs. Filter by the test
# names you add (e.g. `core_cache`) or run the whole `--lib` sweep in Task 9.
git add src/agent/mod.rs src/agent/construct.rs src/agent/loop/bootstrap/run.rs
git commit -m "feat(pillar-a): per-session core cache with component-attributed invalidation logging"
```

---

### Task 8: Provider-conversion assertions

**Files:**
- Test: `src/providers/openai_compatible.rs`, `src/providers/anthropic_native.rs`, `src/providers/google_genai.rs` (test modules; locate each adapter's body-construction fn)

- [ ] **Step 1: openai_compatible — order-preservation test**

Feed a message sequence `[system(core), assistant, user, system(tail), user]` to the body builder; assert the converted message array preserves count, order, and roles (system messages inline, not merged/hoisted). Prefix property: converting `seq` and `seq + [new tool exchange]` yields arrays where the first `len(seq)` elements are byte-identical.

- [ ] **Step 2: anthropic_native / google_genai — determinism only**

Same logical input twice → identical serialized body bytes (their hoist/merge mapping may reshape, but must be a pure function). Per spec §Testing, cross-turn cache-prefix assertions DO NOT apply to these adapters.

- [ ] **Step 3: tool-array canonical emission order** (the assertion Task 4 deferred from core to the provider boundary)

Spec §Pillar A: the name-sorted canonical order must be the **emission order of the provider payload's tool array**, not just the hash order — otherwise a source-order change leaves the core hash stable while flipping `tool_defs_hash` and the rendered prefix (a break with no `Core prompt invalidated` line, which exit criterion 2 reads as a bug).

**Sort UPSTREAM, not in the provider** (code-review finding): the OpenAI adapter emits `body["tools"] = json!(tools)` verbatim in incoming order (`openai_compatible.rs:235`) — it does NOT sort, and should not start (the other adapters would each need their own copy). Task 6 performs an initial bootstrap sort and, critically, a final sort of `effective_tool_defs` immediately before `MessageBuildData` reaches the fingerprint/provider boundary. Then:
- the provider test asserts the adapter **preserves** the (already-sorted) incoming order — pass a sorted slice in, assert the emitted `tools` array order matches element-for-element;
- the bootstrap/message-build regression asserting late-appended definitions are still emitted name-sorted is already owned and committed by Task 6.

**The invariant is "whatever subset is emitted is name-sorted"** — under `tool_filter_enforce = true` or per-turn MCP injection the emitted set is a per-turn subset, but it must still be name-sorted so its order is deterministic for a given membership (so `tool_defs_hash` flips only on membership change, never on order). The full session-static superset is what feeds `CoreInputs.tool_roster`.

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib providers 2>&1 | tail -2
git add src/providers/openai_compatible.rs src/providers/anthropic_native.rs src/providers/google_genai.rs
git commit -m "test(pillar-a): adapter conversion order-preservation (openai) and determinism (anthropic/google)"
```

---

### Task 9: Full verification + gate

**Files:** none — verification.

- [ ] **Step 1:** `cargo test --lib 2>&1 | tail -3` — green (coordinate with concurrent workstreams; known-ignorable failures must be explicitly listed in the report, not waved through).
- [ ] **Step 2:** `cargo fmt --check && cargo clippy --all-features -- -D warnings 2>&1 | tail -2` — clean.
- [ ] **Step 3:** Spec §Testing checklist sweep: items 1 (stable-region extension — covered by Task 6/8 tests), 3 (fact-store → tail-only), 4 (skills change → one named invalidation) each map to a passing test; item 2 (archived[N] stability) is Pillar B — explicitly out of scope here.
- [ ] **Step 4:** Single-summary-insertion check (spec §Tail): `rg -n 'Conversation summary' src/` returns no insertion site (only the deleted/historical references), and the build-stage `[Session Summary]` index-1 injection is gone — the summary now appears only inside the tail. This is the precondition that keeps `prefix_hash_archived` from flipping on overflow during the Task 10 gate run.

---

### Task 10: Live measurement run (A-gate)

**Files:** none — operational; requires the user's environment.

- [ ] **Step 1: Preconditions** — `tool_filter_enforce = false`; daemon started manually with `RUST_LOG="info,aidaemon::agent::message_build_phase=debug"` and stdout to `/tmp/aidaemon-attribution-run.log` (launchd agent booted out for the run, restored after — Pillar C Task 8 procedure). Control MCP-trigger variance as well: either disable query-triggered MCP tool injection for this gate, or use a protocol whose turns are verified to produce identical emitted tool membership. Record which mode was used and report every observed `tool_defs_hash` flip with the triggering membership change; `tool_filter_enforce = false` alone does not make MCP membership static.
- [ ] **Step 2: Offsets** — record `llama-from-line` and `daemon-from-line` at idle (the established procedure).
- [ ] **Step 3: Protocol** — 10 fresh distinct turns (file/script lifecycle in a NEW scratch dir name never used before), single session, plus extension turns if observed breaks < 20 (the post-C baseline saw 16; the floor applies to gate runs).
- [ ] **Step 4: Analysis + A-gate** —

```bash
python3 scripts/cache-attribution.py --daemon-log <segment> --session "<session>" --llama-from-line <N>
```
**Semantics note (state in the results).** Post-A, message zero is core-only, so `prefix_hash_system` is now the **core-stability metric** — its cross-turn stability is the direct measurement of Pillar A's central claim (it previously fingerprinted the full compiled prompt incl. timestamp/memories, which flipped every turn by construction). The volatile content it used to absorb now lives in `tail_hash`.

PASS requires:
- criterion 1 still PASS (0 within-task system flips);
- **every cross-turn `prefix_hash_system` flip pairs with a `Core prompt invalidated` line** (expected count in a quiet run: ZERO flips);
- every `prefix_hash_archived` flip pairs with `Window decision` / `Prefix mutation` / render-cache `fp_mismatch` — at this phase archived churn from age_collapse is still expected and attributed, NOT a failure (Pillar B removes it);
- tail-only flips reported as expected;
- `tool_defs_hash` stable within the controlled run; if MCP injection was intentionally left active, any membership-driven flips are reported separately and the core-stability result is not misclassified;
- record median turn-start evaluated tokens vs the 15,565 baseline (expected improvement ≈ core-region size; document the measured number as Pillar B's starting point). This is an attribution/partial-improvement gate, not an ≥80% efficiency gate.

- [ ] **Step 5: Record + changelog + commit docs** — results table appended to this plan; spec §Pillar A gains a "Measured post-A" note; CHANGELOG [Unreleased] entry with the measured numbers. Commit docs with `git add -f` for the spec/plan files.

---

## Implementation status (2026-06-07)

Tasks 1–9 implemented via subagent-driven development (fresh implementer + two-stage spec/quality review per task), each committed after the per-commit gate (`cargo fmt`, `cargo clippy --all-features -- -D warnings`, tests). Commit chain:

| Task | Commit(s) | Notes |
|---|---|---|
| 1 | `469f0b8`, `0096766` | `tail_hash`/`prefix_hash_archived` fingerprint regions + `LlmCallData`; `session_summary_hash` retired; tail location requires `role == "system"` |
| 2 | `4b33559`, `2313d1e` | `cache-attribution.py` parses new regions; tail-only flips classified expected; `session_summary` dropped from `STAGE_ORDER` |
| 3 | `279597b`, `1ccce3c` | `CoreInputs`/`ComponentHashes` canonicalization + component hashing; session-static `core_tool_roster` accessor + `sort_tool_definitions_by_name` (query-independent roster — NOT `base_tool_defs`) |
| 4 | `047bb9f`, `253a86b` | `render_core_prompt` extracted (pure/sync) + `assemble_core_inputs` (explicit stable inputs); disabled skills filtered |
| 5+6 | `092ee7c`, `a9f0522` | `build_context_tail` at boundary−1; `BootstrapData` core/tail split; both summary insertion paths retired (summary lives only in tail); budget reserves core+tail; emitted roster name-sorted at both boundaries; channel-rules + skills-catalog routed through the renderer (no double emission) |
| 7 | `3083f0c`, `4f5de17` | Per-session core cache (`core_prompts`) with `Core prompt invalidated component=…` logging; HIT reuses bytes without re-render |
| 8 | `04aeea7` | Provider conversion assertions: openai order-preservation + prefix property; anthropic/google determinism; tool-array passthrough (adapters do not sort) |

**Gate (Task 9):** `cargo fmt --check` clean; `cargo clippy --all-features -- -D warnings` clean; `cargo test --lib` 2468 passed / 1 failed — the single failure `startup::tools::tests::base_tool_registry_names_match_built_schema_names` is **pre-existing** (fails identically on the baseline before any Pillar A work) and unrelated to this plan. Integration tests (flattened into the `integration_tests::` module, run within the `--lib` sweep) pass, including the updated `part_10` summary-position assertion. Single-summary-insertion check: the only summary in the LLM message payload is inside the tail (the `main_loop.rs` `[Session Summary]` reference is the separate task-planner's ephemeral context, not the payload).

**Task 10 (live A-gate) — daemon prepped, awaiting operator turns.** The launchd agent (`ai.aidaemon`) was booted out and the daemon restarted manually under `caffeinate -i` with `RUST_LOG="info,aidaemon::agent::loop::message_build_phase=debug"`, stdout → `/tmp/aidaemon-attribution-run.log`. `tool_filter_enforce = false` is already set in `config.toml`. Remaining operator steps: record llama/daemon line offsets at idle; drive 10 fresh distinct turns (new scratch-dir name) in a single session; then `python3 scripts/cache-attribution.py --daemon-log <segment> --session "<session>" --llama-from-line <N>`. PASS = 0 within-task system flips; every cross-turn `prefix_hash_system` flip pairs with a `Core prompt invalidated` line (expected: zero in a quiet run); archived flips attributed to Window decision / Prefix mutation / fp_mismatch; tail-only flips reported expected; `tool_defs_hash` stable (or MCP-membership flips reported separately). Record median turn-start evaluated tokens vs the 15,565 baseline. After the run: results table here, spec §Pillar A "Measured post-A" note, CHANGELOG numbers (the [Unreleased] entry currently marks measurement pending).

## Out of scope (Pillar B's plan)
Turn-anchored fetch, `turn_id` in event payloads/hydration, archived turn renders + cache, whole-turn eviction/anchor, the age_collapse ladder removal. After this plan lands, archived-region churn remains — visible and attributed via `prefix_hash_archived`, which is exactly the instrumentation Pillar B builds against.

**Spawned agents (depth > 0).** Specialists/executors spawned via `spawn_agent` build their prompt through the same `build_system_prompt_for_message` path, so they inherit the core/tail split automatically — no separate task. The core CACHE is keyed by `session_id`, so each spawned session gets its own core entry (correct: a sub-agent's role/tool roster differs from the root's). The live A-gate (Task 10) measures the **root single-session** path only; sub-agent cache behavior is not separately gated here. If a future audit wants per-spawn-session core-stability numbers, that is an added measurement, not a code change.
