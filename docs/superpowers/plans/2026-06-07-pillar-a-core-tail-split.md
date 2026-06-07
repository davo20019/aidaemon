# Pillar A: Stable Core / Task Context Tail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze message zero into a byte-stable core (recompiled only on logged, component-attributed input changes) and move all volatile per-task context into a task context tail message placed before the current turn — making `prefix_hash_system` cross-turn stable and every remaining prefix flip attributable.

**Architecture:** Per `docs/superpowers/specs/2026-06-06-cross-turn-prefix-stability-design.md` §Pillar A, §Payload layout, §Observability. `build_system_prompt_for_message` splits into `build_core_prompt(core_inputs)` (component-hashed, session-cached, pure) and `build_context_tail(per_task_inputs)` (timestamp, session context, session summary, query-ranked memories, matched skill content, speaker context, resume checkpoint — compiled once per task). The tail is a system message inserted **before the current user message**; the session summary leaves message index 1. The Phase 0 fingerprint gains `tail_hash` and `prefix_hash_archived`; `session_summary_hash` retires by reporting empty.

**Tech Stack:** Rust; existing modules `src/agent/runtime/system_prompt.rs`, `src/agent/loop/message_build_phase.rs`, `src/agent/loop/prefix_fingerprint.rs`; `scripts/cache-attribution.py`.

**Baselines (comparators from the post-C run, 2026-06-07):** median payload 16,238 tokens; median turn-start evaluated 15,565 tokens; system prompt 35,104 bytes (of which the session-context block alone measured ~10.6k pre-C); `tool_defs_hash` stable. Pillar A's success is *attribution* (exit criterion 2 of the spec) plus partial turn-start reduction from core reuse; the ≥80% target lands with Pillar B.

**Execution caveats (read first):**
- The working tree hosts concurrent workstreams. Reference code by SYMBOL, re-locate before editing (`rg -n 'fn build_system_prompt_for_message' src/`); never `git add -A`; stage only your hunks.
- Attribution/measurement runs require `tool_filter_enforce = false` (shadow) in config.toml — per-request roster gating is the spec's anti-pattern and breaks `tool_defs_hash` stability. Verify before any measurement.
- The launchd plist (`~/Library/LaunchAgents/ai.aidaemon.plist`) lacks `RUST_LOG`; measurement runs use a manually-started daemon with `RUST_LOG="info,aidaemon::agent::message_build_phase=debug"` (the Pillar C Task 8 procedure) or the plist gains the env var first.

---

### Task 1: Fingerprint regions — `tail_hash`, `prefix_hash_archived`, retire `session_summary_hash`

**Files:**
- Modify: `src/agent/loop/prefix_fingerprint.rs`
- Modify: `src/agent/loop/llm_phase.rs` (the `Provider-call prefix fingerprint` info! line gains the two fields)

The tail does not exist yet; these fields must land FIRST so every later task is observable. Until the tail ships (Task 6), `tail_hash` reports empty and `prefix_hash_archived == prefix_hash_pre_boundary`.

- [ ] **Step 1: Write the failing tests** (append to `prefix_fingerprint.rs` tests)

```rust
#[test]
fn tail_hash_separates_tail_from_archived_region() {
    // Payload: [system, history-a, history-b, TAIL, user]. The tail is
    // located by TASK_CONTEXT_TAIL_MARKER; prefix_hash_archived covers
    // [1..boundary) EXCLUDING the tail; tail_hash covers the tail alone.
    let mut messages = sample_messages();
    let boundary = messages.len() - 1; // sample's last message is the user msg
    messages.insert(
        boundary,
        serde_json::json!({
            "role": "system",
            "content": format!("{TASK_CONTEXT_TAIL_MARKER}\n[Current Date & Time]\nstub"),
        }),
    );
    let fp = provider_call_fingerprint(&messages, "current question", &[], false);
    assert!(!fp.tail_hash.is_empty(), "tail must be located and hashed");

    // Changing ONLY the tail flips tail_hash and pre_boundary, but NOT archived.
    let mut tail_changed = messages.clone();
    tail_changed[boundary]["content"] =
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

- [ ] **Step 4: GREEN + sweep**

Run: `cargo test --lib prefix_fingerprint 2>&1 | tail -2` then `cargo test --lib llm_phase 2>&1 | tail -2`
Expected: all pass (existing `session_summary_hash` assertions in old tests updated to expect empty).

- [ ] **Step 5: Commit**

```bash
git add src/agent/loop/prefix_fingerprint.rs src/agent/loop/llm_phase.rs
git commit -m "feat(pillar-a): tail_hash and prefix_hash_archived fingerprint regions; session_summary_hash retired"
```

---

### Task 2: cache-attribution.py parses the new regions

**Files:**
- Modify: `scripts/cache-attribution.py` (untracked-by-default: commit with `git add -f`)

- [ ] **Step 1: Extend the parser and attribution**

In `parse_daemon_log`, capture `tail_hash` and `prefix_hash_archived` from the fingerprint line. In `attribute()`: a pair where `prefix_hash_archived` is stable but `tail_hash` changed classifies as `tail_replacement (expected)` and never counts toward `pre_boundary_changed_unattributed`; an archived flip keeps the existing cause ladder. In the report, add a `tail-only flips (expected): N` summary line.

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

**Files:**
- Create: `src/agent/runtime/core_prompt.rs`
- Modify: `src/agent/runtime/mod.rs` or the `#[path]` registration site in `src/agent/mod.rs` (match existing module-registration style)

- [ ] **Step 1: Write the failing tests** (same file, `#[cfg(test)]`)

```rust
#[test]
fn component_hash_is_order_insensitive_for_unordered_inputs() {
    let a = CoreInputs {
        base_template: "T".into(),
        tool_roster: vec![("b".into(), "{}".into()), ("a".into(), "{}".into())],
        skills_catalog: vec![("s2".into(), "d2".into(), true), ("s1".into(), "d1".into(), true)],
        specialists: vec![("x".into(), "dx".into())],
        channel_rules: "R".into(),
        persona: "P".into(),
    };
    let mut b = a.clone();
    b.tool_roster.reverse();
    b.skills_catalog.reverse();
    assert_eq!(a.component_hashes(), b.component_hashes());
    assert_eq!(a.aggregate_hash(), b.aggregate_hash());
}

#[test]
fn changed_component_is_named() {
    let a = CoreInputs { /* as above */ };
    let mut b = a.clone();
    b.skills_catalog.push(("s3".into(), "d3".into(), true));
    let diff = a.component_hashes().diff(&b.component_hashes());
    assert_eq!(diff, vec!["skills_catalog"]);
}

#[test]
fn aggregate_hash_is_hash_of_component_hashes() {
    // Pin the construction so a future field addition cannot silently
    // bypass component attribution.
    let a = CoreInputs { /* as above */ };
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
//! collections (sort by name) BEFORE hashing; the canonical order is also
//! the EMISSION order for the tool array (asserted by golden tests in
//! Task 4). No timestamps, map iteration, or env-dependent formatting.

#[derive(Clone, Debug)]
pub(crate) struct CoreInputs {
    pub base_template: String,
    /// (tool name, serialized schema) — sorted by name in canonical form.
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

- [ ] **Step 4: GREEN + commit**

```bash
cargo test --lib core_prompt 2>&1 | tail -2
git add src/agent/runtime/core_prompt.rs src/agent/mod.rs
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
#[tokio::test]
async fn core_prompt_renders_identically_for_identical_inputs() {
    let inputs = test_core_inputs(); // helper constructing a fixed CoreInputs
    let a = render_core_prompt(&inputs);
    let b = render_core_prompt(&inputs);
    assert_eq!(a, b, "core render must be deterministic");
    assert!(!a.contains("[Current Date & Time]"), "timestamp belongs to the tail");
    assert!(!a.contains(TASK_CONTEXT_TAIL_MARKER));
}

#[test]
fn core_prompt_emits_tools_in_hash_order() {
    // Canonical order is the emission order (spec §Pillar A): reorder the
    // roster input; rendered bytes must not change.
    let mut inputs = test_core_inputs();
    let a = render_core_prompt(&inputs);
    inputs.tool_roster.reverse();
    assert_eq!(a, render_core_prompt(&inputs));
}
```

- [ ] **Step 2: RED, then implement `render_core_prompt(&CoreInputs) -> String`**

Mechanically: move the static format! sections from `build_base_system_prompt`/`build_system_prompt_for_message` into the new renderer, parameterized only by `CoreInputs`. `build_system_prompt_for_message` now calls `render_core_prompt` for the core and keeps (for one task) appending the volatile sections after it — behavior-identical output until Task 5 splits the tail message. Existing prompt tests must stay green (the rendered concatenation is unchanged at this task's end).

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
```

- [ ] **Step 2: Implement `build_context_tail(...) -> String`**

Move the volatile sections (Task 4's disposition list) into it; first line is `TASK_CONTEXT_TAIL_MARKER` (import the shared constant from `prefix_fingerprint`). Compiled once per task in the same bootstrap path that compiles the prompt today — no per-iteration recompute (within-task byte stability is invariant 3's stable region).

- [ ] **Step 3: GREEN + commit**

```bash
cargo test --lib system_prompt 2>&1 | tail -2
git add src/agent/runtime/system_prompt.rs
git commit -m "feat(pillar-a): build_context_tail with shared marker; volatile sections leave message zero"
```

---

### Task 6: Payload assembly — tail before current turn, summary leaves index 1

**Files:**
- Modify: `src/agent/loop/message_build_phase.rs`
- Modify: `src/agent/loop/bootstrap_phase/…` (wherever the compiled prompt is attached to the task; locate via `rg -n 'build_system_prompt_for_message' src/`)

- [ ] **Step 1: Failing build-phase tests** (place beside existing message-build tests)

Assert on the built `messages` Vec for a fixture task:
1. exactly one system message starts with `TASK_CONTEXT_TAIL_MARKER`, positioned immediately BEFORE the current user message (boundary − 1);
2. no message at index 1 starts with `SESSION_SUMMARY_MARKER` (the summary string now appears only inside the tail);
3. message zero equals `render_core_prompt(inputs)` bytes exactly (no volatile suffix).

- [ ] **Step 2: Implement**

Message zero = cached core bytes (Task 7 wires the cache; here call render directly). Insert the tail message at the boundary before the current user message; delete the index-1 summary insertion (`rg -n 'SESSION_SUMMARY_MARKER' src/agent/loop/` and remove the build-phase injection site, keeping the constant for the fingerprint and tail). Within-task iterations must NOT rebuild the tail — reuse the task-compiled string.

- [ ] **Step 3: Stage-hash and existing-test fallout**

Run: `cargo test --lib message_build 2>&1 | tail -3` and `cargo test --lib integration_tests 2>&1 | tail -3`
Existing tests asserting the old summary position or message-zero suffix sections update to the new layout. The Phase 0 stage hashes keep their semantics (requirement 6: extended, never redefined) — the tail simply becomes part of what `session_summary`-stage and pre-boundary stage hashes see; no stage is removed.

- [ ] **Step 4: Commit**

```bash
git add src/agent/loop/message_build_phase.rs src/agent/runtime/system_prompt.rs
git commit -m "feat(pillar-a): task context tail precedes current turn; session summary leaves index 1"
```

---

### Task 7: Core cache + `Core prompt invalidated component=` logging

**Files:**
- Modify: `src/agent/mod.rs` (Agent field), `src/agent/loop/bootstrap_phase/…` (per-task hook)

- [ ] **Step 1: Failing integration-style tests**

1. Two consecutive tasks, unchanged inputs → message zero bytes identical AND no `Core prompt invalidated` line (assert via a log-capture helper if available, else via cache state exposed `#[cfg(test)]`);
2. toggle one skill between tasks → exactly one invalidation naming `component=skills_catalog`, new core bytes;
3. store a fact between tasks → core bytes identical (facts are tail-side; this is spec §Testing item 3).

- [ ] **Step 2: Implement**

`core_prompts: Arc<RwLock<HashMap<String, CachedCore>>>` on Agent where `CachedCore { aggregate: String, components: ComponentHashes, bytes: String }`. Per task bootstrap: assemble `CoreInputs` (cheap — names/strings already in memory), compute hashes, compare; on hit reuse bytes verbatim; on miss render, log `info!(session_id, component = %changed.join(","), "Core prompt invalidated")`, replace entry.

- [ ] **Step 3: GREEN + commit**

```bash
cargo test --lib core_prompt 2>&1 | tail -2 && cargo test --lib bootstrap 2>&1 | tail -2
git add src/agent/mod.rs src/agent/loop/bootstrap_phase
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

- [ ] **Step 3: GREEN + commit**

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

---

### Task 10: Live measurement run (A-gate)

**Files:** none — operational; requires the user's environment.

- [ ] **Step 1: Preconditions** — `tool_filter_enforce = false`; daemon started manually with `RUST_LOG="info,aidaemon::agent::message_build_phase=debug"` and stdout to `/tmp/aidaemon-attribution-run.log` (launchd agent booted out for the run, restored after — Pillar C Task 8 procedure).
- [ ] **Step 2: Offsets** — record `llama-from-line` and `daemon-from-line` at idle (the established procedure).
- [ ] **Step 3: Protocol** — 10 fresh distinct turns (file/script lifecycle in a NEW scratch dir name never used before), single session, plus extension turns if observed breaks < 20 (the post-C baseline saw 16; the floor applies to gate runs).
- [ ] **Step 4: Analysis + A-gate** —

```bash
python3 scripts/cache-attribution.py --daemon-log <segment> --session "<session>" --llama-from-line <N>
```
PASS requires:
- criterion 1 still PASS (0 within-task system flips);
- **every cross-turn `prefix_hash_system` flip pairs with a `Core prompt invalidated` line** (expected count in a quiet run: ZERO flips);
- every `prefix_hash_archived` flip pairs with `Window decision` / `Prefix mutation` / render-cache `fp_mismatch` — at this phase archived churn from age_collapse is still expected and attributed, NOT a failure (Pillar B removes it);
- tail-only flips reported as expected;
- `tool_defs_hash` stable within the run;
- record median turn-start evaluated tokens vs the 15,565 baseline (expected improvement ≈ core-region size; document the measured number as Pillar B's starting point).

- [ ] **Step 5: Record + changelog + commit docs** — results table appended to this plan; spec §Pillar A gains a "Measured post-A" note; CHANGELOG [Unreleased] entry with the measured numbers. Commit docs with `git add -f` for the spec/plan files.

---

## Out of scope (Pillar B's plan)
Turn-anchored fetch, `turn_id` in event payloads/hydration, archived turn renders + cache, whole-turn eviction/anchor, the age_collapse ladder removal. After this plan lands, archived-region churn remains — visible and attributed via `prefix_hash_archived`, which is exactly the instrumentation Pillar B builds against.
