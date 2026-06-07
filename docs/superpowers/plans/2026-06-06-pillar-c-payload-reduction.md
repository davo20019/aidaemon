# Pillar C: Payload Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink the per-call LLM payload from ~23k to ~16–17k tokens by deduplicating the `## Tools` prose against the schemas and slimming verbose admin-tool schemas — without changing tool-selection behavior.

**Architecture:** Per `docs/superpowers/specs/2026-06-06-cross-turn-prefix-stability-design.md` §Pillar C. Two reductions: (1) the `## Tools` bullet list in `build_base_system_prompt` (core.rs) duplicates the schemas and is deleted — the existing Tool Selection Guide *table* stays as the brief selection guide, and the handful of prose-only behavioral rules migrate into their tools' schema descriptions (schemas are the single source of truth); (2) the ten fattest admin-tool schemas are compressed under per-tool byte budgets enforced by tests. Tool roster membership does NOT change (per the spec's per-turn-gating anti-pattern: static-but-smaller roster).

**Tech Stack:** Rust, serde_json, existing `#[cfg(test)]` modules per tool file, integration suite in `src/integration_tests/`.

**Constraints from the spec (read them before starting):**
- This is the only Phase 1 pillar that can change model behavior. The tool-selection integration suite must stay green, and the live smoke (Task 9) must pass before declaring the pillar done.
- Membership of the tool roster must not change; only description text shrinks.
- Lands as one deployment → exactly one sanctioned `tool_defs_hash` change, **between** attribution runs.

---

### Task 1: Schema byte-budget tests (the pillar's RED state)

Each fat tool gets a budget test in its own file's existing `#[cfg(test)]` module, asserting the serialized schema fits its ceiling. All ten tests are written FIRST and must FAIL (current sizes exceed the ceilings) — they go green one task at a time as the slimming lands. Budgets (current → ceiling, ~35–45% cuts):

| tool | file | current bytes | ceiling |
|---|---|---|---|
| manage_api | src/tools/manage_api.rs | 3317 | 2000 |
| manage_memories | src/tools/manage_memories.rs | 3291 | 2000 |
| manage_people | src/tools/manage_people.rs | 3102 | 1900 |
| manage_oauth | src/tools/manage_oauth.rs | 2731 | 1700 |
| manage_goal_tasks | src/tools/manage_goal_tasks.rs | 2711 | 1700 |
| manage_cli_agents | src/tools/manage_cli_agents.rs | 2560 | 1500 |
| manage_config | src/tools/config_manager.rs | 2552 | 1600 |
| manage_skills | src/tools/manage_skills.rs | 2311 | 1500 |
| manage_http_auth | src/tools/manage_http_auth.rs | 2225 | 1400 |
| goal_trace | src/tools/goal_trace.rs | 1852 | 1200 |

(Sum of cuts ≈ 8.6k bytes ≈ 2.1–2.9k tokens from schemas; the `## Tools` prose deletion in Task 2 contributes the rest toward the spec's ~6k-token target.)

**Files:**
- Modify: each tool file listed above (test module only, this task)

- [ ] **Step 1: For each of the ten files, locate the existing `#[cfg(test)] mod tests` and how it constructs the tool**

Run for each file (example shown for manage_api; repeat per file):
```bash
grep -n "#\[cfg(test)\]" src/tools/manage_api.rs
grep -n "fn schema" src/tools/manage_api.rs
```
Read the first existing test in that module to see the tool construction it uses (each file's tests already construct the tool; reuse that construction verbatim — do not invent a new one).

- [ ] **Step 2: Append one budget test per file**

Template — identical in every file except the construction line (copied from that file's existing tests), the tool name, and the ceiling from the table above:

```rust
#[test]
fn schema_fits_payload_budget() {
    // Pillar C of 2026-06-06-cross-turn-prefix-stability-design.md:
    // admin-tool schemas ride in EVERY provider call; this ceiling is the
    // per-tool payload budget. If you trip this assert by adding features,
    // compress the description text — do not raise the ceiling without
    // updating the spec's Pillar C table.
    let tool = /* construct exactly as the existing tests in this module do */;
    let bytes = serde_json::to_string(&tool.schema()).unwrap().len();
    assert!(
        bytes <= 2000,
        "manage_api schema is {bytes} bytes, budget is 2000"
    );
}
```

- [ ] **Step 3: Run all ten and verify they FAIL with the current sizes**

Run: `cargo test --lib schema_fits_payload_budget 2>&1 | tail -3`
Expected: `10 failed` (each panic message reports the current byte count — record these, they're the baseline).

- [ ] **Step 4: Commit the red tests**

```bash
git add src/tools/manage_api.rs src/tools/manage_memories.rs src/tools/manage_people.rs src/tools/manage_oauth.rs src/tools/manage_goal_tasks.rs src/tools/manage_cli_agents.rs src/tools/config_manager.rs src/tools/manage_skills.rs src/tools/manage_http_auth.rs src/tools/goal_trace.rs
git commit -m "test: Pillar C schema byte budgets (red) for ten admin tools"
```
(Committing red tests is deliberate here: they are the pillar's tracked TODO list. CI runs on push, not per-commit — do not push until Task 8.)

---

### Task 2: Replace the `## Tools` bullet list with a pointer; migrate prose-only rules into schemas

**Files:**
- Modify: `src/core.rs` (the `build_base_system_prompt` literal, `## Tools` section at ~lines 1754–1805, and the conditional `{*_tool_doc}` block definitions earlier in the same function)
- Modify: `src/tools/terminal.rs`, `src/tools/web_search.rs` (or the file holding web_search's schema), `src/tools/file_tools.rs` (or wherever edit_file's schema lives) — schema descriptions gain the migrated rules

**Background:** the bullet list duplicates schema content, EXCEPT four behavioral rules that exist only in prose. Those move into the owning tool's schema description (source of truth), then the bullets are deleted. The Tool Selection Guide table above the bullets is the "brief selection guide" the spec requires — it stays, unchanged.

- [ ] **Step 1: Find the four prose-only rules and their owning schemas**

```bash
grep -n "dangerous segment" src/core.rs        # terminal chain-refusal rule
grep -n "do NOT re-search" src/core.rs         # web_search focus rule
grep -n "retry once before asking" src/core.rs # edit_file recovery rule
grep -n "heredoc commands trigger" src/core.rs # write_file heredoc rule
grep -rn "fn schema" src/tools/terminal.rs
grep -rn "\"web_search\"" src/tools/*.rs | grep -l schema || grep -rln "web_search" src/tools/ | head -3
```

- [ ] **Step 2: Append each rule to its owning schema description (only if not already present there)**

terminal schema description gains (check first — part may already exist):
```text
 If a command chain (&&, ||, ;, |) contains ANY dangerous segment, refuse the ENTIRE chain and ask which specific operation the user wants — never split a chain to run only the "safe" parts.
```

web_search schema description gains:
```text
 One focused search is almost always enough; for factual lookups do NOT re-search with rephrased queries — synthesize promptly.
```

edit_file schema description gains:
```text
 On not-found/ambiguous text, read_file the same path and retry once before asking the user.
```

write_file schema description gains (if its description does not already say it):
```text
 ALWAYS prefer write_file over terminal heredocs (cat > file << EOF) — heredocs trigger the approval flow.
```

- [ ] **Step 3: Delete the `## Tools` bullet section in core.rs and replace with the pointer**

The section from the line `## Tools` through the final `{...}{direct_mode_doc}` placeholder line (currently `- \`read_file\`: ...` through `{manage_oauth_tool_doc}{direct_mode_doc}`) becomes:

```text
## Tools
Your tool schemas are the authoritative reference for what each tool does and
how to call it. Use the Tool Selection Guide table above to pick the right
tool for a task; consult the schema for parameters and semantics.
{direct_mode_doc}
```

Note `{direct_mode_doc}` is retained — it is mode guidance, not a tool description.

- [ ] **Step 4: Empty the now-unreferenced conditional `*_tool_doc` bindings**

Each conditional binding earlier in `build_base_system_prompt` (e.g. `browser_tool_doc`, `send_file_tool_doc`, `spawn_tool_doc`, `cli_agent_tool_doc`, `manage_cli_agents_tool_doc`, `health_probe_tool_doc`, `manage_skills_tool_doc`, `use_skill_tool_doc`, `skill_resources_tool_doc`, `manage_people_tool_doc`, `http_request_tool_doc`, `manage_api_tool_doc`, `manage_http_auth_tool_doc`, `manage_oauth_tool_doc`) is now unused by the literal: delete the bindings AND their placeholders' usage. Their `{*_table_row}` counterparts in the Tool Selection Guide REMAIN — the table is the selection guide. Before deleting each binding, check whether its text contains a behavioral rule absent from the tool's schema (same test as Step 2); migrate any such rule first, then delete.

Run: `cargo build 2>&1 | grep -E "unused variable|^error"` and remove anything the compiler flags as newly unused.

- [ ] **Step 5: Fix base-prompt tests that assert on the deleted bullets**

Run: `cargo test --lib system_prompt 2>&1 | tail -5` and `cargo test --lib core 2>&1 | tail -5`
Any test asserting the presence of a deleted bullet updates to assert the new pointer text instead. Tests asserting the Tool Selection Guide table stay untouched.

- [ ] **Step 6: Verify the size win**

```bash
cargo test --lib 2>&1 | tail -3   # all green except the 10 red budget tests
```
Then measure: add (temporarily, do not commit) `eprintln!("base prompt bytes: {}", base_system_prompt.len());` in core.rs after `build_base_system_prompt`, run any integration test, confirm a reduction of ≥4000 bytes vs. before, remove the eprintln. Alternatively diff `AIDAEMON_DUMP_LLM_REQUESTS` output before/after on the live daemon.

- [ ] **Step 7: Commit**

```bash
git add src/core.rs src/tools/terminal.rs src/tools/web_search.rs
git commit -m "feat(pillar-c): replace ## Tools prose with schema pointer; migrate prose-only rules into owning schemas"
```
(Adjust the file list to whichever files actually held the web_search/edit_file/write_file schemas.)

---

### Task 3: Slim manage_cli_agents and manage_skills (the two verbose top-level descriptions)

**Files:**
- Modify: `src/tools/manage_cli_agents.rs:190` (schema), `src/tools/manage_skills.rs:3074` (schema/description)

- [ ] **Step 1: Replace manage_cli_agents' 597-byte description with:**

```text
Manage CLI AI agents (claude/gemini/codex/etc.). Actions: add (requires approval), remove, list, enable, disable, history. Discovery is automatic; use this only to add custom agents or inspect invocation history.
```

- [ ] **Step 2: Replace manage_skills' 413-byte description with:**

```text
Manage skills at runtime. Actions: add (from URL), add_inline, list, remove, enable, disable, browse (search registries), install (from registry), update (re-fetch). Skills can bundle resource files — see skill_resources.
```

- [ ] **Step 3: Compress per-parameter descriptions in both schemas to ≤80 chars each**

Rules (apply to every `"description"` inside `"parameters"`): drop examples that restate the enum, drop "This parameter is used to..." framing, keep units/constraints/foreign-key references. Enum values themselves are never removed.

- [ ] **Step 4: Run the two budget tests — expect PASS**

Run: `cargo test --lib manage_cli_agents::tests::schema_fits_payload_budget manage_skills 2>&1 | tail -3` (or filter by `schema_fits_payload_budget` and confirm these two flipped green)

- [ ] **Step 5: Run both files' full test modules**

Run: `cargo test --lib manage_cli_agents 2>&1 | tail -2 && cargo test --lib manage_skills 2>&1 | tail -2`
Expected: all green (description-content asserts, if any, updated to the new text).

- [ ] **Step 6: Commit**

```bash
git add src/tools/manage_cli_agents.rs src/tools/manage_skills.rs
git commit -m "feat(pillar-c): slim manage_cli_agents and manage_skills schemas under budget"
```

---

### Task 4: Slim manage_api, manage_memories, manage_people

**Files:**
- Modify: `src/tools/manage_api.rs:756`, `src/tools/manage_memories.rs:254`, `src/tools/manage_people.rs:66`

These three are parameter-description-heavy (top-level descriptions are already short). Apply the Task 3 Step 3 compression rules to every parameter description; additionally:

- [ ] **Step 1: manage_api — compress each action's parameter docs to ≤80 chars; keep the action enum complete; keep SSRF/safety wording intact wherever it appears** (safety text is behavioral, not descriptive — it stays)

- [ ] **Step 2: manage_memories — keep the full action enum (search/forget/list plus the goal actions); compress the per-action parameter text. The goal-id prefix-matching note and the "not for storing facts" redirect STAY** (they prevent misuse; they may be shortened but not removed):

```text
Goal ids accept a unique prefix. Not for storing facts — use remember_fact.
```

- [ ] **Step 3: manage_people — keep all 12 action names and the privacy-sensitive wording; compress everything else**

- [ ] **Step 4: Run the three budget tests — expect PASS; run each file's full test module — expect green**

Run: `cargo test --lib schema_fits_payload_budget 2>&1 | tail -3`
Expected: 5 of 10 now pass (Tasks 3–4 done), 5 still fail.

- [ ] **Step 5: Commit**

```bash
git add src/tools/manage_api.rs src/tools/manage_memories.rs src/tools/manage_people.rs
git commit -m "feat(pillar-c): slim manage_api, manage_memories, manage_people schemas under budget"
```

---

### Task 5: Slim manage_oauth, manage_goal_tasks, manage_config

**Files:**
- Modify: `src/tools/manage_oauth.rs:860`, `src/tools/manage_goal_tasks.rs:218`, `src/tools/config_manager.rs:1381`

- [ ] **Step 1: Apply the compression rules to all three; manage_config keeps the guided-action redirects** (`switch_provider`, `list_failover_providers`, `add_failover_provider`, `remove_failover_provider` must remain named in the description — the base prompt's Built-in Channels section references them)

- [ ] **Step 2: Run budget tests — 8 of 10 green; run the three files' test modules — green**

- [ ] **Step 3: Commit**

```bash
git add src/tools/manage_oauth.rs src/tools/manage_goal_tasks.rs src/tools/config_manager.rs
git commit -m "feat(pillar-c): slim manage_oauth, manage_goal_tasks, manage_config schemas under budget"
```

---

### Task 6: Slim manage_http_auth and goal_trace

**Files:**
- Modify: `src/tools/manage_http_auth.rs:1048`, `src/tools/goal_trace.rs:396`

- [ ] **Step 1: Apply the compression rules; goal_trace keeps both action names (goal_trace, tool_trace) and the tool_trace alias cross-reference**

- [ ] **Step 2: Run budget tests — ALL 10 green**

Run: `cargo test --lib schema_fits_payload_budget 2>&1 | tail -3`
Expected: `10 passed; 0 failed`

- [ ] **Step 3: Commit**

```bash
git add src/tools/manage_http_auth.rs src/tools/goal_trace.rs
git commit -m "feat(pillar-c): slim manage_http_auth and goal_trace schemas; all ten budgets green"
```

---

### Task 7: Tool-selection behavioral guard (integration suite)

**Files:** none modified — verification task.

- [ ] **Step 1: Run the full integration suite**

Run: `cargo test --lib integration_tests 2>&1 | tail -3`
Expected: all green. Failures here mean a migrated/deleted prose rule was load-bearing for a scripted behavior — restore that rule into the owning tool's schema (NOT into core.rs prose) and re-run.

- [ ] **Step 2: Run the orchestration/tool-selection subset explicitly**

Run: `cargo test --lib integration_tests::test_orchestration 2>&1 | tail -2`
Expected: all green (27+ tests).

- [ ] **Step 3: Full pre-commit gate**

```bash
cargo fmt && cargo clippy --all-features -- -D warnings && cargo test --all-features 2>&1 | tail -5
```
Expected: zero warnings, zero failures.

---

### Task 8: CHANGELOG + push

**Files:**
- Modify: `CHANGELOG.md` ([Unreleased] → Changed)

- [ ] **Step 1: Add the changelog entry**

```markdown
- **Per-call LLM payload reduced ~25–30%** (Pillar C of the cross-turn prefix
  stability design): the `## Tools` prose section now defers to tool schemas
  as the single source of truth (prose-only behavioral rules were migrated
  into the owning schemas), and ten admin-tool schemas were compressed under
  test-enforced byte budgets. Tool roster membership is unchanged.
```

- [ ] **Step 2: Commit and push the branch**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for Pillar C payload reduction"
git push origin feature/sliding-window-phase0-observability
```

---

### Task 9: Live smoke + post-C baseline attribution run (gate for Pillars A/B)

**Files:** none — operational task. Requires the user's environment (local llama-server, Telegram bot).

- [ ] **Step 1: Rebuild and restart the daemon under caffeinate**

```bash
cargo build
# coordinate with the user: confirm the daemon is idle, then
kill <aidaemon pid> <caffeinate pid>
RUST_LOG="info,aidaemon::agent::message_build_phase=debug" /usr/bin/caffeinate -i ./target/debug/aidaemon > /tmp/aidaemon-attribution-run.log 2>&1 &
```

- [ ] **Step 2: Live smoke — representative tool-selection tasks via the bot (single session, distinct texts)**

Send, one at a time, waiting for each completion: a file-create task (expects write_file), a file-read question (read_file), a shell task (terminal/run_command), a memory store ("note that I prefer X" → remember_fact), a memory recall (manage_memories search), a web lookup (web_search). PASS = each task picks the expected tool family on the first attempt (check the daemon log's tool events). Any wrong-tool regression → revisit the slimmed description of the affected tool before proceeding.

- [ ] **Step 3: Record clean-window offsets and run the 10-turn attribution protocol**

```bash
echo "llama-from-line: $(($(wc -l < ~/.aidaemon/llama-server.log | tr -d ' ')+1))"
echo "daemon-lines: $(wc -l < /tmp/aidaemon-attribution-run.log | tr -d ' ')"
```
Then run a fresh 10-turn protocol (distinct turn texts, single session, no other bots — same run rules as Phase 0) and analyze:

```bash
tail -n +<daemon-lines> /tmp/aidaemon-attribution-run.log > /tmp/post-c-baseline-segment.log
python3 scripts/cache-attribution.py --daemon-log /tmp/post-c-baseline-segment.log \
  --session "<session-id>" --llama-from-line <llama-from-line>
```

- [ ] **Step 4: Verify the Pillar C exit gate (spec §Exit criteria item 4)**

- Median per-call payload ≤ ~17k tokens (read the `prompt` column of the report)
- `tool_defs_hash` cross-turn stable within the run (the deployment break happened between runs)
- Record the post-C baseline numbers in the spec's Pillar C section — they are the comparator for Pillars A/B.

---

## Post-plan note

Pillars A and B get their own implementation plans, written **after** this pillar's Task 9 establishes the post-C baseline (their exit targets are defined against it, and B's eviction budgets use the post-C payload sizes). Do not start A/B from this document.
