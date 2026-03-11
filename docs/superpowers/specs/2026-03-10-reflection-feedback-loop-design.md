# Reflection Feedback Loop Design

## Problem

When the agent encounters repeated tool failures (wrong API URL, incorrect parameters, misunderstood error responses), it retries the same failing approach until budget is exhausted, then promises future actions it can't deliver because tools are stripped. It lacks the ability to:

1. Diagnose *why* it's failing by cross-referencing loaded skills/context
2. Self-correct mid-task based on that diagnosis
3. Store learnings persistently so the same mistake never happens twice

## Solution

A general-purpose reflection system inspired by the Reflexion framework. On the 2nd repeated failure with the same error signature, an LLM call analyzes the failure, cross-references the task's already-active skills, and produces a diagnosis that gets injected as a high-priority system directive before the next main-model call. Learnings are stored as *unverified* in the existing `error_solutions` table and only promoted to verified (retrievable by future tasks) when the agent actually recovers in the same task.

## Architecture: Approach B — Dedicated Reflection Module

### New File

`src/agent/loop/tool_execution/reflection.rs`

### Trigger Mechanism

Reflection fires after `apply_result_learning()` records the **2nd semantic failure with the same signature** for any tool. Criteria:

- Same tool name AND same normalized error signature (via `derive_failure_signature()`)
- NOT already reflected for this (tool, signature) pair (tracked in `reflection_completed: HashSet<(String, String)>`)
- One reflection per (tool, signature) pair per task — no repeated reflections

Why 2nd failure: 1st is normal exploration, 3rd+ is near the block limit. 2nd is the earliest moment we have a pattern without being too late to act.

### Error History Accumulator

New field on `ResultLearningState`:

```rust
pub tool_error_history: &'a mut HashMap<(String, String), Vec<ToolErrorEntry>>,  // keyed by (tool_name, signature)
pub reflection_completed: &'a mut HashSet<(String, String)>,
```

```rust
pub(super) struct ToolErrorEntry {
    pub iteration: usize,
    pub arguments_summary: String,  // truncated to 500 chars, secrets redacted
    pub error_text: String,         // truncated to 1000 chars, secrets redacted
}
```

Each semantic failure appends to `tool_error_history` keyed by `(tool_name, signature)` (capped at 5 entries per key). This ensures only same-signature errors are fed into the reflection prompt — tools like `http_request` with varied failure modes won't mix unrelated errors. **Arguments and error text are redacted via `redact_secrets()` before storage** to prevent auth tokens, connection strings, or secret-bearing data from leaking into the reflection LLM prompt. When the 2nd same-signature failure occurs and `reflection_completed` doesn't contain the pair, reflection triggers.

### Reflection Context & Output

```rust
pub(super) struct ReflectionCtx<'a> {
    pub provider: &'a Arc<dyn ModelProvider>,
    pub model: String,                          // fast model from router
    pub tool_name: String,
    pub failure_signature: String,
    pub error_history: Vec<ToolErrorEntry>,
    pub user_task: String,                       // original user_text
    pub relevant_skill_excerpt: Option<String>,  // matched skill body, truncated to 2000 chars
    pub task_id: String,
}

pub(super) struct ReflectionDiagnosis {
    pub root_cause: String,
    pub recommended_action: String,
    pub learning: Option<ErrorSolutionDraft>,
}

pub(super) struct ErrorSolutionDraft {
    pub error_pattern: String,
    pub domain: Option<String>,
    pub solution_summary: String,
    pub solution_steps: Vec<String>,
}
```

### Reflection Prompt

Sent to the fast model with no tools (text-only):

```
You are a failure analysis system. An AI agent is stuck repeating the same error.

TASK: {user_task}
FAILING TOOL: {tool_name}
ERROR PATTERN: {failure_signature}

ERROR HISTORY (most recent first):
{for each error: iteration N — args: {args_summary} — error: {error_text}}

{if relevant_skill_excerpt:}
RELEVANT SKILL/API GUIDE (loaded for this task):
{skill excerpt, truncated to 2000 chars}
{end if}

Analyze why the agent keeps failing. Respond in exactly this format:

ROOT_CAUSE: <one sentence explaining the fundamental reason for repeated failure>
RECOMMENDED_ACTION: <one concrete, specific action the agent should take next>
LEARNING: <one sentence that would prevent this mistake in future tasks, or NONE if too situation-specific>
```

### Skill Cross-Referencing

Reflection is scoped to **already-active skills** — the skills that were matched and confirmed during the task's bootstrap phase (the same set injected into the system prompt). This is safer and more relevant than scanning all enabled skills via `skill_cache.get()`.

The active skill names are extracted from `build_system_prompt_for_message()` (which already computes the confirmed skill set), stored in `BootstrapData`, and threaded through `ToolExecutionCtx` into the reflection context. For each active skill, we check:
- `source_url` domain matches the failing URL domain (extracted from tool arguments)
- OR triggers match the **tool arguments** using **word-boundary matching** via `contains_keyword_as_words()` (repo convention — no substring `.contains()`)
- OR triggers match the **tool name** using the same word-boundary matching (catches skills that trigger on the tool itself, e.g., a skill with trigger `"http_request"` matches when `http_request` is the failing tool)

If a matching skill is found, its `body` is included (truncated to 2000 chars, **secrets redacted** via `redact_secrets()`) in the reflection prompt so the model can spot discrepancies between what the skill says and what the agent did.

### Model Selection

Uses `router.default_model()` directly. The current router maps all tiers to the default model and reserves fallbacks for error-recovery cascades only — there is no separate "fast model" routing. The reflection call is ~200 input + ~50 output tokens — negligible cost. 10-second timeout; on failure/timeout, skip gracefully.

### Diagnosis Injection

New `SystemDirective` variant:

```rust
ReflectionDiagnosis {
    tool_name: String,
    root_cause: String,
    recommended_action: String,
}
```

Rendered as:

```
[SYSTEM] SELF-DIAGNOSIS for {tool_name}: {root_cause}.
ACTION REQUIRED: {recommended_action}.
Do NOT repeat the same failing approach. If you cannot fix the issue, report the actual error honestly to the user.
```

Pushed to `pending_system_messages` immediately after reflection completes, injected before the very next main-model LLM call.

### Verified Persistent Learning

When `ReflectionDiagnosis.learning` is `Some(draft)`, stored immediately via `state.insert_error_solution()` with `success_count=0, failure_count=0` (unverified). The existing retrieval filter (`success_count > failure_count`) naturally gates unverified learnings from future tasks.

**Verification flow:** Reflection solution IDs are tracked in `pending_reflection_solutions: HashMap<(String, String), Vec<i64>>` (keyed by `(tool_name, signature)`). Verification is **exact-key**: on tool success, the most recent failure signature for that tool is looked up from `tool_error_history` (by highest iteration number), and only the `(tool, that_signature)` entry is promoted — not all entries for the tool. This prevents a successful `http_request` to one endpoint from incorrectly verifying a reflection learning about a different failure signature. This is separate from the existing `pending_error_solution_ids` mechanism (which is tool-agnostic and handles diagnostic-hint verification).

**Deduplication:** Uses the existing `insert_error_solution()` path which upserts on `(error_pattern, domain, solution_summary)`. Multiple solutions for the same pattern are intentionally preserved — the existing schema allows alternative fixes.

Future task retrieval flow (only for verified learnings):
1. 1st failure with similar pattern
2. `apply_result_learning()` queries `get_relevant_error_solutions()` (existing code, runs on 1st failure)
3. Finds verified learning (`success_count > failure_count`), appends coaching notice to tool result
4. Agent self-corrects without needing reflection

### Force-Text Promise Prevention

When `ForceTextToolLimitReached` activates in `stopping_phase.rs`, its render appends:

```
Do NOT promise future actions like "let me try..." or "I'll search for..." — your tools have been disabled.
Report what you found, what failed, and what the user can try instead.
```

### Integration Point

In `run.rs`, after `apply_result_learning()` returns:

```rust
if let Some(diagnosis) = self.maybe_trigger_reflection(
    &tc, &reflection_ctx, &mut state
).await {
    state.pending_system_messages.push(
        SystemDirective::ReflectionDiagnosis {
            tool_name: tc.name.clone(),
            root_cause: diagnosis.root_cause,
            recommended_action: diagnosis.recommended_action,
        }
    );
    if let Some(draft) = diagnosis.learning {
        store_reflection_learning(&self.state, draft).await;
    }
}
```

### Error Handling

Every step is wrapped in fallible logic. If the LLM call fails, times out, or returns unparseable output, log a warning and return `None`. Reflection never panics, never blocks indefinitely, never affects tool execution outcomes.

### Testing

Unit tests with `MockProvider`:
- Reflection triggers on 2nd same-signature failure, not 1st or 3rd
- Reflection doesn't re-trigger for same (tool, signature) pair
- Diagnosis injected as SystemDirective
- Learning stored via state
- Timeout/error gracefully returns None
- Skill matching finds relevant skills by URL domain
- Force-text promise prevention text appears in ForceTextToolLimitReached render

## Files Modified

| File | Change |
|------|--------|
| `src/agent/loop/tool_execution/reflection.rs` | **NEW** — ReflectionDiagnosis, maybe_trigger_reflection(), response parsing, skill matching (word-boundary, sanitized, scoped to active skills) |
| `src/agent/loop/tool_execution/result_learning.rs` | Add ToolErrorEntry, tool_error_history accumulation, return just-incremented signature, tool-scoped reflection verification on recovery |
| `src/agent/loop/tool_execution/run.rs` | Call maybe_trigger_reflection() using signature from apply_result_learning(), track in pending_reflection_solutions |
| `src/agent/loop/tool_execution/phase_impl.rs` | Declare `mod reflection;` |
| `src/agent/loop/tool_execution/types.rs` | Add tool_error_history, reflection_completed, pending_reflection_solutions, active_skill_names to ToolExecutionCtx |
| `src/agent/loop/system_directives.rs` | Add ReflectionDiagnosis variant + render, append promise prevention to ForceTextToolLimitReached |
| `src/agent/loop/bootstrap/types.rs` | Add active_skill_names to BootstrapData |
| `src/agent/runtime/system_prompt.rs` | Return active skill names alongside system prompt string |
| `src/agent/loop/bootstrap/run.rs` | Capture active skill names, store in BootstrapData |
| `src/agent/loop/main_loop.rs` | Initialize all new loop state fields, destructure active_skill_names from bootstrap |

## End-to-End Example

```
Iteration 1: http_request("api.clinicaltrials.gov/...") → 404
  → 1st semantic failure, signature="http 404 not found"
  → queries error_solutions — finds nothing
  → appends coaching notice

Iteration 2: http_request("api.clinicaltrials.gov/...") → 404 (same signature)
  → 2nd semantic failure, TRIGGERS REFLECTION
  → LLM sees error history + active skill excerpt (says "use clinicaltrials.gov/api/v2")
  → returns: ROOT_CAUSE: "Wrong hostname — skill says use clinicaltrials.gov/api/v2"
  → injects SystemDirective::ReflectionDiagnosis
  → stores ErrorSolution with success_count=0 (UNVERIFIED)

Iteration 3: LLM sees [SYSTEM] SELF-DIAGNOSIS → uses correct URL → SUCCESS
  → recovery detected: same tool succeeded after reflection
  → ErrorSolution.success_count bumped to 1 (now VERIFIED)

Future task (weeks later):
  → 1st failure with similar pattern
  → error_solutions match (success_count=1 > failure_count=0)
  → coaching notice from verified learning
  → self-corrects without needing reflection
```
