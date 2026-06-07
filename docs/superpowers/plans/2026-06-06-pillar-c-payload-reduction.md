# Pillar C: Payload Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink the per-call LLM payload from ~23k to ~16–17k tokens by deduplicating the `## Tools` prose against the schemas and slimming verbose admin-tool schemas, while holding tool-selection behavior constant as verified by tests and live smoke.

**Architecture:** Per `docs/superpowers/specs/2026-06-06-cross-turn-prefix-stability-design.md` §Pillar C. Two reductions: (1) replace the duplicative `## Tools` catalog in `build_base_system_prompt` with a compact guide. Single-tool semantics move to schemas, cross-tool routing stays in the Tool Selection Guide, compact `cli_agent` coordination rules stay in the prompt, and dynamic HTTP profile context stays runtime-generated; (2) compress the ten fattest admin-tool schemas under per-tool byte budgets enforced by tests. Tool roster membership does NOT change (per the spec's per-turn-gating anti-pattern: static-but-smaller roster).

**Tech Stack:** Rust, serde_json, per-tool `#[cfg(test)]` modules, integration suite in `src/integration_tests/`.

**Constraints from the spec (read them before starting):**
- This is the only Phase 1 pillar that can change model behavior. The integration suite must stay green, and the live smoke (Task 8) must pass before declaring the pillar done.
- Membership of the tool roster must not change; only description text shrinks.
- Lands as one deployment → exactly one sanctioned `tool_defs_hash` change, **between** attribution runs.
- Byte budgets are regression proxies, not the exit gate. The full provider payload's measured token count in the post-C attribution run is authoritative.
- Do not move whole onboarding essays into schemas. Preserve only behavioral invariants, safety rules, action discoverability, and critical routing.

---

### Task 1: Capture baselines and prepare the schema-budget strategy

Each fat tool will get a budget test asserting the serialized schema fits its ceiling. Add each test in the same task as its implementation so every commit remains green per the repository pre-commit policy. Budgets below are planning baselines; re-measure before editing because the branch may have moved.

**RECALIBRATED 2026-06-07 (Task 1 wire measurement):** the original
planning baselines measured json! source bytes; the live dump measures the
serialized wire format, which is authoritative. Membership also shifted:
`manage_goal_tasks` does not appear in the normal payload (goal-context-only
registration) and is dropped; `manage_mcp` and `scheduled_goal_runs` do
appear and are added. (`spawn_agent` 2436, `http_request` 1751, and
`remember_fact` 1717 are the true top-3 but are high-traffic,
behavior-critical schemas — out of scope per the spec's "rarely-used admin
tools" framing; candidates for a future pass.)

| tool | file | wire bytes (measured) | ceiling |
|---|---|---|---|
| manage_memories | src/tools/manage_memories.rs | 1579 | 1100 |
| manage_cli_agents | src/tools/manage_cli_agents.rs | 1504 | 1000 |
| manage_skills | src/tools/manage_skills.rs | 1441 | 1000 |
| manage_oauth | src/tools/manage_oauth.rs | 1438 | 1000 |
| manage_people | src/tools/manage_people.rs | 1411 | 1000 |
| manage_mcp | src/mcp/ (schema site: rg 'fn schema' src/mcp) | 1390 | 1000 |
| manage_api | src/tools/manage_api.rs | 1359 | 950 |
| manage_config | src/tools/config_manager.rs | 1250 | 900 |
| scheduled_goal_runs | src/tools/scheduled_goal_runs.rs | 1098 | 800 |
| manage_http_auth | src/tools/manage_http_auth.rs | 1032 | 800 |
| goal_trace | src/tools/goal_trace.rs | 1032 | 800 |

(Eleven tools; planned wire cuts ≈ 4.2k bytes ≈ ~1.0k tokens. The `## Tools`
catalog measured **17,999 bytes (~4.5k tokens)** of the 52,880-byte system
prompt — Task 2 is the dominant lever, schemas are the long tail.)

(Sum of planned cuts ≈ 8.6k bytes ≈ 2.1–2.9k tokens from schemas; this conversion is approximate. The compact `## Tools` replacement in Task 2 contributes the rest toward the spec's ~6k-token target.)

**Files:**
- None modified — measurement and construction audit only

- [ ] **Step 1: Measure the current serialized sizes**

Start the current daemon with request dumping enabled, send one representative owner request that exposes the normal static roster, and select its first non-force-text dump:

```bash
rm -rf /tmp/pillar-c-before
AIDAEMON_DUMP_LLM_REQUESTS=/tmp/pillar-c-before ./target/debug/aidaemon
```

In another terminal, send the request through the configured channel, stop the daemon, then set:

```bash
DUMP="$(find /tmp/pillar-c-before -type f -name '*.json' | sort | head -1)"
jq -c '.tools[] | {name, bytes: (tojson | length)}' "$DUMP"
echo "base-system-bytes: $(jq -r '.messages[0].content' "$DUMP" | wc -c | tr -d ' ')"
echo "all-tool-def-bytes: $(jq -c '.tools' "$DUMP" | wc -c | tr -d ' ')"
echo "full-dump-bytes: $(jq -c . "$DUMP" | wc -c | tr -d ' ')"
```

Record the ten actual schema sizes in the table above and record the three aggregate values in an implementation-results note at the bottom of this plan. Request dumps contain raw conversation content; delete `/tmp/pillar-c-before` after recording the measurements.

- [ ] **Step 2: Audit test construction for all ten tools**

Locate the existing `#[cfg(test)] mod tests` and determine whether it already constructs the tool:
```bash
rg -n '#\[cfg\(test\)\]|mod tests|fn schema' \
  src/tools/manage_api.rs src/tools/manage_memories.rs \
  src/tools/manage_people.rs src/tools/manage_oauth.rs \
  src/tools/manage_goal_tasks.rs src/tools/config_manager.rs \
  src/tools/manage_skills.rs src/tools/manage_http_auth.rs \
  src/tools/goal_trace.rs
```

`src/tools/manage_cli_agents.rs` has no test module and its constructor requires runtime dependencies. `src/tools/manage_api.rs` has tests but no simple tool fixture. For either file, and for any other schema whose existing fixture is disproportionately expensive, extract its static schema literal into a private pure function:

```rust
fn manage_cli_agents_schema() -> Value {
    json!({ /* existing schema, unchanged initially */ })
}
```

Then make `Tool::schema()` return the private function and test that function without constructing runtime dependencies. Do not add a public test-only API.

- [ ] **Step 3: Use this budget-test template in Tasks 3–6**

```rust
#[test]
fn schema_fits_payload_budget() {
    // Pillar C of 2026-06-06-cross-turn-prefix-stability-design.md:
    // admin-tool schemas ride in EVERY provider call; this ceiling is the
    // per-tool payload budget. If you trip this assert by adding features,
    // compress the description text — do not raise the ceiling without
    // updating the Pillar C implementation plan.
    let tool = /* construct exactly as the existing tests in this module do */;
    let bytes = serde_json::to_string(&tool.schema()).unwrap().len();
    assert!(
        bytes <= 2000,
        "manage_api schema is {bytes} bytes, budget is 2000"
    );
}
```

For a schema extracted to a private helper, call that helper instead of constructing a tool. In each slimming task: add the relevant tests, run them to observe RED, apply compression, then run them to GREEN before committing.

---

### Task 2: Replace the `## Tools` catalog with a compact, audited guide

**Files:**
- Modify: `src/core.rs` (the `build_base_system_prompt` literal, `## Tools` section at ~lines 1754–1805, and the conditional `{*_tool_doc}` block definitions earlier in the same function)
- Modify: `src/tools/terminal.rs`, `src/tools/web_search.rs`, `src/tools/edit_file.rs`, `src/tools/write_file.rs`, `src/tools/cli_agent.rs`, and any owning tool schemas identified by the audit

> **This is the heaviest, highest-risk task in the plan** — 11 steps across `core.rs` plus 6+ tool-schema files, and the only one that can silently drop load-bearing behavioral guidance. Budget the bulk of the implementation effort here, migrate before deleting (never the reverse), and treat the Task 7 Step 3 diff audit (grep every deleted `NEVER`/`ALWAYS`/`Do NOT`/`prefer`/`required` imperative) as the mandatory safety net for this task.

**Background:** much of the catalog duplicates schema content, but the prose also contains single-tool invariants, cross-tool routing, orchestration policy, and runtime-generated API context. Do not treat the four rules below as exhaustive. Apply this migration policy to every deleted block:

| Content type | Destination |
|---|---|
| Parameter meaning, action behavior, single-tool safety invariant | Owning tool schema |
| Cross-tool choice or fallback (`web_fetch` → `http_request`, `run_command` → `terminal`) | Tool Selection Guide row |
| `cli_agent` coordination/no-double-dipping rules | Compact prompt subsection retained below the table |
| Runtime auth profile names and profiles missing API guides | Compact runtime-generated prompt subsection |
| Examples, repeated parameter lists, narrative onboarding scripts | Delete |

- [ ] **Step 1: Inventory every catalog block before deleting anything**

```bash
rg -n 'let .*_tool_doc|## Tools|dangerous segment|do NOT re-search|retry once before asking|heredoc commands trigger' src/core.rs
```

Audit all of: base file/search/command tools, `browser`, `send_file`, `spawn_agent`, `cli_agent`, `manage_cli_agents`, `health_probe`, `manage_skills`, `use_skill`, `skill_resources`, `manage_people`, `http_request`, `manage_api`, `manage_http_auth`, and `manage_oauth`. For every normative sentence, mark its destination using the table above.

- [ ] **Step 2: Migrate the known single-tool rules**

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

Also migrate any other single-tool safety or misuse-prevention rules found in Step 1. The `edit_file` retry is already present in the kept Decision Framework table, but retain it in the schema as the authoritative tool behavior.

- [ ] **Step 3: Preserve cross-tool routing in the Tool Selection Guide**

Update existing rows rather than adding prose paragraphs:

- `run_command`: fallback says `terminal for arbitrary commands or commands requiring approval`
- `web_fetch`: fallback/usage says `http_request for REST/JSON APIs; browser for login/JS pages`
- `terminal`: preferred only when no dedicated tool fits
- `scheduled_goal_runs`: row remains explicitly scoped to scheduled goals, not fact storage
- `manage_api`, `manage_http_auth`, `manage_oauth`: rows retain the end-to-end/manual-auth/browser-OAuth split

- [ ] **Step 4: Retain compact `cli_agent` coordination guidance**

Replace `cli_agent_tool_doc` with a short non-catalog subsection. Keep only the behavior that is not expressible as parameter semantics:

```text
## CLI Agent Delegation
Use cli_agent for complex multi-step work when available. Always set working_dir.
Do not send the same task to multiple agents or run agents concurrently in the
same working_dir. After delegating, do not duplicate the same work with direct
tools; review the agent's result and use direct tools only for validation or
clearly separate follow-up work.
```

Keep the existing dynamic availability guidance from `direct_mode_doc`. Move parameter descriptions and available-agent details to `src/tools/cli_agent.rs`; do not copy the old orchestration essay into the schema.

- [ ] **Step 5: Retain compact dynamic HTTP context**

`http_request_tool_doc` currently includes runtime auth profile names and `skill_warning`; static schemas cannot represent those values. Replace the long block with a compact runtime-generated subsection:

```text
## API Runtime Context
Available manual HTTP auth profiles: {profile_names_or_none}.
Profiles missing API guides: {profiles_missing_skills_or_none}.
For a missing guide, use manage_api for end-to-end onboarding or
manage_skills(action='learn_api') with official docs/OpenAPI.
Never ask the user to paste credentials into chat.
```

Delete the multi-step user-facing script and repeated parameter list. Preserve domain binding, HTTPS, and approval semantics in the `http_request` schema if absent.

- [ ] **Step 6: Compress API/auth guidance without violating schema budgets**

For `manage_api`, `manage_http_auth`, and `manage_oauth`, migrate only:

- secret-handling invariant: never request pasted credentials
- manual-auth vs browser-OAuth routing
- verify-before-first-use/runtime-refresh behavior
- reconnect/remove safety for OAuth

Delete examples and onboarding narration. These invariants count against the Task 4–6 schema budgets; if a rule is inherently cross-tool, put it in the selection table instead.

- [ ] **Step 7: Replace the remaining `## Tools` catalog with a pointer**

After the audit and migrations, the static catalog from `- \`read_file\`` through the conditional placeholders becomes:

```text
## Tools
Your tool schemas are the authoritative reference for what each tool does and
how to call it. Use the Tool Selection Guide table above to pick the right
tool for a task; consult the schema for parameters and semantics.
{cli_agent_guidance}
{api_runtime_context}
{direct_mode_doc}
```

The exact binding names may differ, but all three retained sections must remain conditional on their current feature/config availability.

- [ ] **Step 8: Delete only the audited, superseded bindings**

Delete the old `browser_tool_doc`, `send_file_tool_doc`, `spawn_tool_doc`, `manage_cli_agents_tool_doc`, `health_probe_tool_doc`, `manage_skills_tool_doc`, `use_skill_tool_doc`, `skill_resources_tool_doc`, `manage_people_tool_doc`, `manage_api_tool_doc`, `manage_http_auth_tool_doc`, and `manage_oauth_tool_doc` only after completing the inventory. Replace, rather than blindly delete, `cli_agent_tool_doc` and `http_request_tool_doc`.

Run: `cargo build`
Expected: success with no newly unused bindings.

- [ ] **Step 9: Add prompt regression tests and fix obsolete assertions**

Add assertions covering:

- pointer text is present
- old static `- \`read_file\`:` catalog entry is absent
- `cli_agent` guidance appears only when enabled
- dynamic API context contains configured profile names and missing-guide names
- critical routing rows remain present

Run: `cargo test --lib core`
Expected: all core prompt tests pass.

- [ ] **Step 10: Verify the intermediate size win**

```bash
cargo test --lib
```
All tests must remain green because no future budget tests have been added yet. Measure the same representative configuration used in Task 1 and confirm the base prompt shrank by at least 4000 bytes. Record the before/after byte counts in the plan's implementation notes; remove temporary instrumentation.

- [ ] **Step 11: Commit**

```bash
git add src/core.rs src/tools/terminal.rs src/tools/web_search.rs \
  src/tools/edit_file.rs src/tools/write_file.rs src/tools/cli_agent.rs
git commit -m "feat(pillar-c): replace tool catalog with compact routing guidance"
```

Include any additional owning schema files changed during the audit.

---

### Task 3: Slim manage_cli_agents and manage_skills (the two verbose top-level descriptions)

**Files:**
- Modify: `src/tools/manage_cli_agents.rs:190` (schema), `src/tools/manage_skills.rs:3074` (schema/description)

- [ ] **Step 1: Extract and test `manage_cli_agents`' static schema**

Move the existing `json!` literal into `fn manage_cli_agents_schema() -> Value`, make `Tool::schema()` call it, and add a new `#[cfg(test)] mod tests` with the 1000-byte budget test from Task 1.

Run: `cargo test --lib manage_cli_agents::tests::schema_fits_payload_budget`
Expected: FAIL with the measured size above 1000 bytes.

- [ ] **Step 2: Add the `manage_skills` budget and semantic tests**

Add the 1000-byte budget test to its existing test module. Also add a test that the schema description contains every non-obvious action name:

```rust
#[tokio::test]
async fn schema_description_keeps_action_discoverability() {
    let (tool, _state, _skills_dir) = setup_tool().await;
    let schema = tool.schema();
    let description = schema["description"].as_str().unwrap();
    for action in ["learn_api", "remove_all", "review"] {
        assert!(description.contains(action), "missing action {action}");
    }
}
```

Run: `cargo test --lib manage_skills::tests::schema_fits_payload_budget`
Expected: FAIL with the measured size above 1000 bytes.

- [ ] **Step 3: Replace manage_cli_agents' verbose description with:**

```text
Manage CLI AI agents (claude/gemini/codex/etc.). Actions: add (requires approval), remove, list, enable, disable, history. Discovery is automatic; use this only to add custom agents or inspect invocation history.
```

- [ ] **Step 4: Replace manage_skills' verbose description with:**

```text
Manage skills at runtime. Actions: add, add_inline, learn_api (docs/OpenAPI), list, remove, remove_all, enable, disable, browse, install, update, review. Bundled files are available through skill_resources.
```

- [ ] **Step 5: Compress per-parameter descriptions in both schemas to ≤80 chars each**

Rules (apply to every `"description"` inside `"parameters"`): drop examples that restate the enum, drop "This parameter is used to..." framing, keep units/constraints/foreign-key references. Enum values themselves are never removed.

- [ ] **Step 6: Run the two budget and semantic tests — expect PASS**

```bash
cargo test --lib manage_cli_agents::tests
cargo test --lib manage_skills::tests
```

Expected: all green, including action discoverability.

- [ ] **Step 7: Commit**

```bash
git add src/tools/manage_cli_agents.rs src/tools/manage_skills.rs
git commit -m "feat(pillar-c): slim manage_cli_agents and manage_skills schemas under budget"
```

---

### Task 4: Slim manage_api, manage_memories, manage_people

**Files:**
- Modify: `src/tools/manage_api.rs:756`, `src/tools/manage_memories.rs:254`, `src/tools/manage_people.rs:66`

These three are parameter-description-heavy (top-level descriptions are already short). Apply the Task 3 compression rules to every parameter description; additionally:

- [ ] **Step 1: Add the three budget tests and verify RED**

Add ceilings 950 (manage_api), 1100 (manage_memories), and 1000 (manage_people) using the Task 1 template. For `manage_api`, first extract the static schema to a private helper as described in Task 1, then test the helper.

Run each module separately:
```bash
cargo test --lib manage_api::tests::schema_fits_payload_budget
cargo test --lib manage_memories::tests::schema_fits_payload_budget
cargo test --lib manage_people::tests::schema_fits_payload_budget
```
Expected: each new budget test fails before compression.

- [ ] **Step 2: manage_api — compress each action's parameter docs to ≤80 chars; keep the action enum complete; keep SSRF/safety wording intact wherever it appears** (safety text is behavioral, not descriptive — it stays)

- [ ] **Step 3: manage_memories — keep the full action enum (search/forget/list plus the goal actions); compress the per-action parameter text. The goal-id prefix-matching note and the "not for storing facts" redirect STAY** (they prevent misuse; they may be shortened but not removed):

```text
Goal ids accept a unique prefix. Not for storing facts — use remember_fact.
```

- [ ] **Step 4: manage_people — keep all 12 action names and the privacy-sensitive wording; compress everything else**

- [ ] **Step 5: Run the three full test modules — expect green**

```bash
cargo test --lib manage_api::tests
cargo test --lib manage_memories::tests
cargo test --lib manage_people::tests
```

- [ ] **Step 6: Commit**

```bash
git add src/tools/manage_api.rs src/tools/manage_memories.rs src/tools/manage_people.rs
git commit -m "feat(pillar-c): slim manage_api, manage_memories, manage_people schemas under budget"
```

---

### Task 5: Slim manage_oauth, manage_mcp, manage_config

**Files:**
- Modify: `src/tools/manage_oauth.rs:860`, the `manage_mcp` schema site (rg 'fn schema' src/mcp src/tools | rg -i mcp), `src/tools/config_manager.rs:1381`

- [ ] **Step 1: Add the three budget tests and verify RED**

Add ceilings 1000 (manage_oauth), 1000 (manage_mcp), and 900 (manage_config). Run each specific test and confirm it fails before compression.

```bash
cargo test --lib manage_oauth::tests::schema_fits_payload_budget
cargo test --lib schema_fits_payload_budget 2>&1 | grep mcp   # manage_mcp's module path depends on its file
cargo test --lib config_manager::tests::schema_fits_payload_budget
```

- [ ] **Step 2: Apply the compression rules to all three**

`manage_config` keeps the guided-action redirects: `switch_provider`, `list_failover_providers`, `add_failover_provider`, and `remove_failover_provider` must remain named.

`manage_oauth` keeps these behavioral invariants in compressed form:

- inspect providers/connection state before requesting credentials
- never ask for credentials in chat
- use `connect` to reauthorize; do not `remove` first unless disconnect was requested
- API-key/bearer/header/basic/OAuth1a routes to `manage_http_auth`

- [ ] **Step 3: Run the three full test modules — expect green**

```bash
cargo test --lib manage_oauth::tests
cargo test --lib mcp 2>&1 | tail -2
cargo test --lib config_manager::tests
```

- [ ] **Step 4: Commit**

```bash
git add src/tools/manage_oauth.rs src/tools/config_manager.rs   # plus the manage_mcp schema file
git commit -m "feat(pillar-c): slim manage_oauth, manage_mcp, manage_config schemas under budget"
```

---

### Task 6: Slim manage_http_auth, scheduled_goal_runs, goal_trace

**Files:**
- Modify: `src/tools/manage_http_auth.rs:1048`, `src/tools/scheduled_goal_runs.rs:654`, `src/tools/goal_trace.rs:396`

- [ ] **Step 1: Add both budget tests and verify RED**

Add ceilings 800, 800, and 800. Run each specific test and confirm it fails before compression.

```bash
cargo test --lib manage_http_auth::tests::schema_fits_payload_budget
cargo test --lib scheduled_goal_runs::tests::schema_fits_payload_budget
cargo test --lib goal_trace::tests::schema_fits_payload_budget
```

- [ ] **Step 2: Apply the compression rules**

`goal_trace` keeps both action names (`goal_trace`, `tool_trace`) and the alias cross-reference.

`manage_http_auth` keeps the secret-handling invariant, verify/runtime-refresh behavior, allowed-domain constraint, and safe GET/HEAD verification constraint.

- [ ] **Step 3: Run both modules and all ten budget tests**

```bash
cargo test --lib manage_http_auth::tests
cargo test --lib scheduled_goal_runs::tests
cargo test --lib goal_trace::tests
cargo test --lib schema_fits_payload_budget
```
Expected: `11 passed; 0 failed`

- [ ] **Step 4: Commit**

```bash
git add src/tools/manage_http_auth.rs src/tools/scheduled_goal_runs.rs src/tools/goal_trace.rs
git commit -m "feat(pillar-c): slim manage_http_auth, scheduled_goal_runs, goal_trace; all eleven budgets green"
```

---

### Task 7: Structural and behavioral verification

**Files:** none modified — verification task.

- [ ] **Step 1: Run the full integration suite**

Run: `cargo test --lib integration_tests`
Expected: all green.

These tests primarily verify agent-loop structure with scripted mock responses; they do **not** prove that a real model still chooses the same tools from changed descriptions.

- [ ] **Step 2: Run the orchestration subset explicitly**

Run: `cargo test --lib test_orchestration`
Expected: all green (27+ tests).

- [ ] **Step 3: Inspect the complete diff for semantic losses**

```bash
git diff -- src/core.rs src/tools
```

For every deleted uppercase imperative (`NEVER`, `ALWAYS`, `Do NOT`, `prefer`, `required`), verify that it was either duplicated, migrated according to Task 2's table, or intentionally removed as narrative. Do not use byte-budget success as justification for deleting a safety or routing invariant.

- [ ] **Step 4: Measure complete tool-definition and prompt bytes**

Reuse Task 1's measurement procedure verbatim so the dumps are comparable — same representative request, same dump location:

```bash
rm -rf /tmp/pillar-c-after
AIDAEMON_DUMP_LLM_REQUESTS=/tmp/pillar-c-after ./target/debug/aidaemon
# send the same representative request, stop the daemon, then:
DUMP="$(find /tmp/pillar-c-after -type f -name '*.json' | sort | head -1)"
echo "base-system-bytes: $(jq -r '.messages[0].content' "$DUMP" | wc -c | tr -d ' ')"
echo "all-tool-def-bytes: $(jq -c '.tools' "$DUMP" | wc -c | tr -d ' ')"
echo "full-dump-bytes: $(jq -c . "$DUMP" | wc -c | tr -d ' ')"
```

Record:

- serialized base-system-prompt bytes
- serialized complete tool-definition bytes
- total request bytes/tokens

Compare against Task 1's baseline (the env var accepts a path → that directory, or `1`/`true` → the default `llm_request_dumps/` dir; use a path both times so the two dumps land in known, distinct locations). This is the aggregate guard against growth outside the ten targeted schemas. Delete `/tmp/pillar-c-after` after recording — dumps contain raw conversation content.

- [ ] **Step 5: Full pre-commit gate**

```bash
cargo fmt
cargo clippy --all-features -- -D warnings
cargo test --all-features
```
Expected: zero warnings, zero failures.

---

### Task 8: Live smoke + post-C baseline attribution run (gate for completion and Pillars A/B)

**Files:** none — operational task. Requires the user's environment (local llama-server, Telegram bot).

- [ ] **Step 1: Rebuild and restart the daemon under caffeinate**

```bash
cargo build
# coordinate with the user: confirm the daemon is idle, then
kill <aidaemon pid> <caffeinate pid>
RUST_LOG="info,aidaemon::agent::message_build_phase=debug" /usr/bin/caffeinate -i ./target/debug/aidaemon > /tmp/aidaemon-attribution-run.log 2>&1 &
```

- [ ] **Step 2: Live smoke — common tool-selection tasks**

Send, one at a time, waiting for each completion: a file-create task (expects write_file), a file-read question (read_file), a shell task (terminal/run_command), a memory store ("note that I prefer X" → remember_fact), a memory recall (manage_memories search), a web lookup (web_search). PASS = each task picks the expected tool family on the first attempt (check the daemon log's tool events). Any wrong-tool regression → revisit the slimmed description of the affected tool before proceeding.

- [ ] **Step 3: Live smoke — guidance most affected by Task 2**

Use read-only/list requests where possible:

- "List the installed CLI agents" → `manage_cli_agents`, not `cli_agent`
- a complex coding/research task → `cli_agent` with `working_dir`; no duplicate direct execution
- "List my installed skills" → `manage_skills(action='list')`
- "Learn this API from its OpenAPI URL" → `manage_skills(action='learn_api')` or `manage_api` when end-to-end onboarding is requested
- "Show available OAuth providers" → `manage_oauth(action='providers')`
- "List manual HTTP auth profiles" → `manage_http_auth(action='list')`
- a REST/JSON fetch request → `http_request`, not `web_fetch`

PASS = correct tool family on the first attempt and compliance with the retained coordination/safety rule. Do not approve mutations merely for smoke testing.

- [ ] **Step 4: Record clean-window offsets and run the 10-turn attribution protocol**

```bash
echo "llama-from-line: $(($(wc -l < ~/.aidaemon/llama-server.log | tr -d ' ')+1))"
echo "daemon-from-line: $(($(wc -l < /tmp/aidaemon-attribution-run.log | tr -d ' ')+1))"
```
Then run a fresh 10-turn protocol (distinct turn texts, single session, no other bots — same run rules as Phase 0) and analyze:

```bash
tail -n +<daemon-from-line> /tmp/aidaemon-attribution-run.log > /tmp/post-c-baseline-segment.log
python3 scripts/cache-attribution.py --daemon-log /tmp/post-c-baseline-segment.log \
  --session "<session-id>" --llama-from-line <llama-from-line>
```

- [ ] **Step 5: Verify the Pillar C exit gate (spec §Exit criteria item 4)**

- Median per-call payload ≤ ~17k tokens (read the `prompt` column of the report)
- `tool_defs_hash` cross-turn stable within the run (the deployment break happened between runs)
- Common and affected-guidance live smokes pass

If the payload misses the target, inspect the dumped aggregate composition before lowering any semantic budget. If tool choice regresses, restore the smallest necessary routing/invariant text in its proper destination and repeat Tasks 7–8.

---

### Task 9: Record measured results, changelog, commit, and push

**Files:**
- Modify: `docs/superpowers/specs/2026-06-06-cross-turn-prefix-stability-design.md` (Pillar C measured baseline)
- Modify: `docs/superpowers/plans/2026-06-06-pillar-c-payload-reduction.md` (checklist/results notes)
- Modify: `CHANGELOG.md` ([Unreleased] → Changed)

- [ ] **Step 1: Record the post-C baseline**

Add the measured median prompt tokens, base-prompt bytes, complete tool-definition bytes, observed reduction percentage, and stable `tool_defs_hash` result to the spec's Pillar C section. These values are the comparator for Pillars A/B.

- [ ] **Step 2: Add a changelog entry using the measured percentage**

Write the entry in this form, substituting the actual Task 8 percentage for `N`:

```text
- **Per-call LLM payload reduced by N%** (Pillar C of the cross-turn prefix
  stability design): the duplicative tool catalog was replaced by compact
  routing, delegation, and runtime API guidance, and ten admin-tool schemas
  were compressed under test-enforced byte budgets. Tool roster membership is
  unchanged.
```

Do not claim 25–30% unless the attribution run measured it.

- [ ] **Step 3: Run the final repository gate**

```bash
cargo fmt
cargo clippy --all-features -- -D warnings
cargo test --all-features
```

- [ ] **Step 4: Commit and push the current branch**

```bash
git add CHANGELOG.md \
  docs/superpowers/specs/2026-06-06-cross-turn-prefix-stability-design.md \
  docs/superpowers/plans/2026-06-06-pillar-c-payload-reduction.md
git commit -m "docs: record Pillar C payload reduction baseline"
git push origin HEAD
```

---

## Post-plan note

Pillars A and B get their own implementation plans, written **after** this pillar's Task 9 establishes the post-C baseline (their exit targets are defined against it, and B's eviction budgets use the post-C payload sizes). Do not start A/B from this document.

## Implementation results

Fill this table during Tasks 1, 7, and 8, then commit it in Task 9:

| Metric | Pre-C | Post-C |
|---|---:|---:|
| Base system prompt bytes | 52,880 | — |
| Complete tool-definition bytes | 36,710 (38 tools) | — |
| Compact full-request bytes | 91,732 | — |
| `## Tools` section bytes | 17,999 | — |
| Median prompt tokens | ~22.3k (Phase 0 run) | — |
| Payload reduction | — | — |
| `tool_defs_hash` stable within run | — | — |
| Common live smoke | — | — |
| Affected-guidance live smoke | — | — |

Pre-C measured 2026-06-07 02:26Z, dump of iter002 of "What files are in my
home directory right now?" (owner DM, 38-tool roster). Top prompt sections:
Tools 17,999 / Current Session Activity 10,624 (volatile — Pillar A's
concern) / Core Rules 3,564 / Available Skills 3,201 / Tool Selection Guide
2,972 (keeper).
