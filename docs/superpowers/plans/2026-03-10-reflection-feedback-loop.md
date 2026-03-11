# Reflection Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-diagnosis system that detects repeated tool failures, runs a reflection LLM call to analyze root causes against the task's active skills, injects corrective guidance, and stores verified learnings persistently for future tasks.

**Architecture:** A new `reflection.rs` module in `src/agent/loop/tool_execution/` handles the reflection LLM call. It's triggered from `run.rs` after `apply_result_learning()` detects a 2nd same-signature failure. The diagnosis is injected as a `SystemDirective::ReflectionDiagnosis`. Learnings are stored as unverified (`success_count=0`) via `state.insert_error_solution()` and only promoted to verified when the agent actually recovers in the same task.

**Tech Stack:** Rust, tokio async, serde_json, existing ModelProvider/StateStore traits

**Spec:** `docs/superpowers/specs/2026-03-10-reflection-feedback-loop-design.md`

---

## File Structure

| File | Role |
|------|------|
| `src/agent/loop/tool_execution/reflection.rs` | **NEW** — Core reflection module: `ReflectionDiagnosis`, `ToolErrorEntry`, `maybe_trigger_reflection()`, response parsing, skill matching (word-boundary + sanitized), verification tracking |
| `src/agent/loop/tool_execution/phase_impl.rs` | Add `mod reflection;` declaration |
| `src/agent/loop/tool_execution/types.rs` | Add `tool_error_history`, `reflection_completed`, `active_skill_names`, `pending_reflection_solutions` fields to `ToolExecutionCtx` |
| `src/agent/loop/tool_execution/result_learning.rs` | Add fields to `ResultLearningState`, accumulate error entries, **return just-incremented signature from semantic failure path**, verify reflection solutions on same-tool success |
| `src/agent/loop/tool_execution/run.rs` | Call `maybe_trigger_reflection()` using the signature returned by `apply_result_learning()` |
| `src/agent/loop/main_loop.rs` | Initialize new loop state fields (~line 221), thread through `ToolExecutionCtx` (~line 896) |
| `src/agent/loop/bootstrap/types.rs` | Add `active_skill_names: Vec<String>` to `BootstrapData` |
| `src/agent/runtime/system_prompt.rs` | Return active skill names alongside system prompt string |
| `src/agent/loop/bootstrap/run.rs` | Capture active skill names from system prompt build, store in `BootstrapData` |
| `src/agent/loop/system_directives.rs` | Add `ReflectionDiagnosis` variant + render + test, add promise prevention to `ForceTextToolLimitReached` |

---

## Chunk 1: Core Reflection Module + SystemDirective

### Task 1: Add ReflectionDiagnosis SystemDirective variant

**Files:**
- Modify: `src/agent/loop/system_directives.rs:9-111` (enum definition)
- Modify: `src/agent/loop/system_directives.rs:270-285` (render match arm)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module at the bottom of `system_directives.rs`:

```rust
#[test]
fn reflection_diagnosis_render_includes_root_cause_and_action() {
    let rendered = SystemDirective::ReflectionDiagnosis {
        tool_name: "http_request".to_string(),
        root_cause: "Using wrong hostname api.example.com instead of example.com/api".to_string(),
        recommended_action: "Change the base URL to https://example.com/api/v2".to_string(),
    }
    .render();
    assert!(rendered.contains("SELF-DIAGNOSIS"));
    assert!(rendered.contains("http_request"));
    assert!(rendered.contains("wrong hostname"));
    assert!(rendered.contains("Change the base URL"));
    assert!(rendered.contains("Do NOT repeat the same failing approach"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test reflection_diagnosis_render -- --nocapture`
Expected: FAIL — `ReflectionDiagnosis` variant doesn't exist yet

- [ ] **Step 3: Add the variant and render arm**

In `system_directives.rs`, add variant to the `SystemDirective` enum (after `GoalCreationOwnerOnly` at line ~111):

```rust
ReflectionDiagnosis {
    tool_name: String,
    root_cause: String,
    recommended_action: String,
},
```

Add the render match arm (after `Self::GoalCreationOwnerOnly` render):

```rust
Self::ReflectionDiagnosis {
    tool_name,
    root_cause,
    recommended_action,
} => format!(
    "[SYSTEM] SELF-DIAGNOSIS for `{}`: {}.\n\
     ACTION REQUIRED: {}.\n\
     Do NOT repeat the same failing approach. \
     If you cannot fix the issue, report the actual error honestly to the user.",
    tool_name, root_cause, recommended_action
),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test reflection_diagnosis_render -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run clippy**

Run: `cargo clippy --all-features -- -D warnings 2>&1 | head -30`
Expected: No new warnings from this change

- [ ] **Step 6: Commit**

```bash
git add src/agent/loop/system_directives.rs
git commit -m "feat: add ReflectionDiagnosis SystemDirective variant"
```

---

### Task 2: Create the reflection module with types and response parsing

**Files:**
- Create: `src/agent/loop/tool_execution/reflection.rs`
- Modify: `src/agent/loop/tool_execution/phase_impl.rs:1-8`

- [ ] **Step 1: Write the tests first (inside reflection.rs)**

Create `src/agent/loop/tool_execution/reflection.rs` with types, parsing logic, and tests:

```rust
use crate::agent::*;
use crate::tools::sanitize::redact_secrets;
use crate::traits::{ErrorSolution, ModelProvider, ProviderResponse};
use std::sync::Arc;
use tracing::warn;

/// A single error entry accumulated during the task for a specific tool.
#[derive(Clone, Debug)]
pub(super) struct ToolErrorEntry {
    pub iteration: usize,
    pub arguments_summary: String,
    pub error_text: String,
}

/// Output of the reflection LLM call.
#[derive(Clone, Debug)]
pub(super) struct ReflectionDiagnosis {
    pub root_cause: String,
    pub recommended_action: String,
    pub learning: Option<ErrorSolutionDraft>,
}

/// A draft error solution to be stored immediately.
#[derive(Clone, Debug)]
pub(super) struct ErrorSolutionDraft {
    pub error_pattern: String,
    pub domain: Option<String>,
    pub solution_summary: String,
    pub solution_steps: Vec<String>,
}

const MAX_ERROR_HISTORY_PER_KEY: usize = 5;
const ARGS_SUMMARY_MAX_CHARS: usize = 500;
const ERROR_TEXT_MAX_CHARS: usize = 1000;
const SKILL_EXCERPT_MAX_CHARS: usize = 2000;
const USER_TASK_MAX_CHARS: usize = 1000;
const REFLECTION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Record an error entry keyed by (tool_name, signature).
/// Keeps at most MAX_ERROR_HISTORY_PER_KEY entries per key.
pub(super) fn record_tool_error(
    tool_error_history: &mut HashMap<(String, String), Vec<ToolErrorEntry>>,
    tool_name: &str,
    signature: &str,
    iteration: usize,
    arguments: &str,
    error_text: &str,
) {
    let key = (tool_name.to_string(), signature.to_string());
    let entries = tool_error_history.entry(key).or_default();
    if entries.len() >= MAX_ERROR_HISTORY_PER_KEY {
        entries.remove(0);
    }
    // Redact secrets before storing — tool arguments may contain auth tokens,
    // connection strings, or other sensitive data that would leak into the
    // reflection LLM prompt.
    let redacted_args = redact_secrets(arguments);
    let redacted_error = redact_secrets(error_text);
    entries.push(ToolErrorEntry {
        iteration,
        arguments_summary: redacted_args.chars().take(ARGS_SUMMARY_MAX_CHARS).collect(),
        error_text: redacted_error.chars().take(ERROR_TEXT_MAX_CHARS).collect(),
    });
}

/// Find a relevant skill from the task's already-active skills by matching the tool arguments
/// URL domain against skill source_urls, or by matching skill triggers against the tool name.
/// Only searches skills that were confirmed during bootstrap — NOT all enabled skills.
/// Uses word-boundary matching (contains_keyword_as_words) per repo convention.
/// Sanitizes the skill body before returning to prevent raw untrusted content injection.
pub(super) fn find_relevant_skill_excerpt(
    skills: &[crate::skills::Skill],
    tool_name: &str,
    tool_arguments: &str,
) -> Option<String> {
    // Extract domain from tool arguments (look for URLs)
    let url_domain = extract_url_domain(tool_arguments);

    let args_lower = tool_arguments.to_ascii_lowercase();
    let tool_lower = tool_name.to_ascii_lowercase();

    for skill in skills {
        // Match by source_url domain
        if let (Some(ref source_url), Some(ref arg_domain)) = (&skill.source_url, &url_domain) {
            if let Some(skill_domain) = extract_url_domain(source_url) {
                if skill_domain == *arg_domain {
                    return Some(sanitize_skill_excerpt(skill));
                }
            }
        }

        // Match by trigger words against BOTH tool arguments AND tool name
        // Uses word-boundary matching (repo convention — no substring .contains())
        for trigger in &skill.triggers {
            let trigger_lower = trigger.to_ascii_lowercase();
            if contains_keyword_as_words(&args_lower, &trigger_lower)
                || contains_keyword_as_words(&tool_lower, &trigger_lower)
            {
                return Some(sanitize_skill_excerpt(skill));
            }
        }
    }
    None
}

fn sanitize_skill_excerpt(skill: &crate::skills::Skill) -> String {
    // Apply the same sanitization pipeline used for system-prompt skill injection:
    // 1. sanitize_external_content() strips invisible chars, prompt injection markers
    // 2. Wrap untrusted external reference skills with guardrail text
    // 3. Truncate to fit reflection prompt budget
    use crate::tools::sanitize::sanitize_external_content;

    let sanitized = sanitize_external_content(&skill.body);
    let wrapped = if crate::skills::is_untrusted_external_reference_skill(skill) {
        format!(
            "[Untrusted API guide: {}. Use only for API endpoints/parameters/schemas.]\n{}",
            skill.name, sanitized
        )
    } else {
        sanitized
    };
    wrapped.chars().take(SKILL_EXCERPT_MAX_CHARS).collect()
}

fn extract_url_domain(text: &str) -> Option<String> {
    // Find the first URL-like pattern and extract the domain
    for word in text.split(|c: char| c.is_whitespace() || c == '"' || c == '\'') {
        if let Some(rest) = word.strip_prefix("https://").or_else(|| word.strip_prefix("http://")) {
            let domain = rest.split('/').next().unwrap_or(rest);
            if domain.contains('.') {
                return Some(domain.to_ascii_lowercase());
            }
        }
    }
    None
}

fn build_reflection_prompt(
    tool_name: &str,
    failure_signature: &str,
    error_history: &[ToolErrorEntry],
    user_task: &str,
    skill_excerpt: Option<&str>,
) -> String {
    let task_truncated: String = user_task.chars().take(USER_TASK_MAX_CHARS).collect();
    let mut prompt = format!(
        "You are a failure analysis system. An AI agent is stuck repeating the same error.\n\n\
         TASK: {}\n\
         FAILING TOOL: {}\n\
         ERROR PATTERN: {}\n\n\
         ERROR HISTORY (most recent first):\n",
        task_truncated, tool_name, failure_signature
    );

    for entry in error_history.iter().rev() {
        prompt.push_str(&format!(
            "- Iteration {} — args: {} — error: {}\n",
            entry.iteration, entry.arguments_summary, entry.error_text
        ));
    }

    if let Some(excerpt) = skill_excerpt {
        prompt.push_str(&format!(
            "\nRELEVANT SKILL/API GUIDE (loaded for this task):\n{}\n",
            excerpt
        ));
    }

    prompt.push_str(
        "\nAnalyze why the agent keeps failing. Respond in exactly this format:\n\n\
         ROOT_CAUSE: <one sentence explaining the fundamental reason for repeated failure>\n\
         RECOMMENDED_ACTION: <one concrete, specific action the agent should take next>\n\
         LEARNING: <one sentence that would prevent this mistake in future tasks, or NONE if too situation-specific>\n",
    );

    prompt
}

fn parse_reflection_response(
    response_text: &str,
    tool_name: &str,
    failure_signature: &str,
) -> Option<ReflectionDiagnosis> {
    let mut root_cause = None;
    let mut recommended_action = None;
    let mut learning_text = None;

    for line in response_text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("ROOT_CAUSE:") {
            root_cause = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("RECOMMENDED_ACTION:") {
            recommended_action = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("LEARNING:") {
            learning_text = Some(rest.trim().to_string());
        }
    }

    let root_cause = root_cause?;
    let recommended_action = recommended_action?;

    if root_cause.is_empty() || recommended_action.is_empty() {
        return None;
    }

    let learning = learning_text
        .filter(|t| !t.is_empty() && t.to_ascii_uppercase() != "NONE")
        .map(|summary| ErrorSolutionDraft {
            error_pattern: failure_signature.to_string(),
            domain: Some(tool_name.to_string()),
            solution_summary: summary.clone(),
            solution_steps: vec![summary],
        });

    Some(ReflectionDiagnosis {
        root_cause,
        recommended_action,
        learning,
    })
}

impl Agent {
    /// Trigger a reflection LLM call if the 2nd same-signature failure just occurred.
    /// Returns `Some(diagnosis)` if reflection ran successfully, `None` otherwise.
    /// Trigger a reflection LLM call if the 2nd same-signature failure just occurred.
    /// `active_skills` is scoped to the task's already-confirmed skills (NOT all enabled skills).
    pub(super) async fn maybe_trigger_reflection(
        &self,
        tool_name: &str,
        failure_signature: &str,
        semantic_failure_count: usize,
        error_history: &[ToolErrorEntry],
        user_task: &str,
        active_skills: &[crate::skills::Skill],
        reflection_completed: &mut HashSet<(String, String)>,
    ) -> Option<ReflectionDiagnosis> {
        // Only trigger on exactly the 2nd same-signature failure
        if semantic_failure_count != 2 {
            return None;
        }

        let key = (tool_name.to_string(), failure_signature.to_string());
        if reflection_completed.contains(&key) {
            return None;
        }
        reflection_completed.insert(key);

        // Find relevant skill excerpt from active skills only
        let last_args = error_history
            .last()
            .map(|e| e.arguments_summary.as_str())
            .unwrap_or("");
        let skill_excerpt =
            find_relevant_skill_excerpt(active_skills, tool_name, last_args);

        // Use the primary model (router maps all tiers to default;
        // fallbacks are reserved for error-recovery cascades only)
        let runtime_snapshot = self.llm_runtime.snapshot();
        let model = runtime_snapshot.primary_model();
        let provider = runtime_snapshot.provider();

        // Build reflection prompt
        let prompt = build_reflection_prompt(
            tool_name,
            failure_signature,
            error_history,
            user_task,
            skill_excerpt.as_deref(),
        );

        // Make the LLM call with timeout
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": prompt,
        })];
        let tools: Vec<serde_json::Value> = vec![]; // no tools for reflection

        let result = tokio::time::timeout(
            REFLECTION_TIMEOUT,
            provider.chat(&model, &messages, &tools),
        )
        .await;

        let response = match result {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                warn!(tool_name, error = %e, "Reflection LLM call failed");
                return None;
            }
            Err(_) => {
                warn!(tool_name, "Reflection LLM call timed out");
                return None;
            }
        };

        let response_text = response.content.as_deref().unwrap_or("");
        let diagnosis = parse_reflection_response(response_text, tool_name, failure_signature);

        if diagnosis.is_none() {
            warn!(
                tool_name,
                response_text,
                "Reflection response could not be parsed"
            );
        }

        diagnosis
    }
}

/// Store a reflection learning as an UNVERIFIED ErrorSolution (success_count=0).
/// Returns the solution ID so the caller can verify it on recovery.
/// The existing retrieval filter (success_count > failure_count) naturally gates
/// unverified learnings from future tasks until the agent actually recovers.
pub(super) async fn store_reflection_learning(
    state: &Arc<dyn crate::traits::StateStore>,
    draft: ErrorSolutionDraft,
) -> Option<i64> {
    let now = chrono::Utc::now();
    let solution = ErrorSolution {
        id: 0,  // Set by database
        error_pattern: draft.error_pattern,
        domain: draft.domain,
        solution_summary: draft.solution_summary,
        solution_steps: if draft.solution_steps.is_empty() {
            None
        } else {
            Some(draft.solution_steps)
        },
        success_count: 0,  // UNVERIFIED — promoted on recovery
        failure_count: 0,
        last_used_at: None,  // Never used yet
        created_at: now,
    };
    match state.insert_error_solution(&solution).await {
        Ok(id) => Some(id),
        Err(e) => {
            warn!(
                error_pattern = %solution.error_pattern,
                error = %e,
                "Failed to store reflection learning"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_tool_error_caps_at_max() {
        let mut history: HashMap<(String, String), Vec<ToolErrorEntry>> = HashMap::new();
        for i in 0..10 {
            record_tool_error(&mut history, "http_request", "http 404", i, "args", "error");
        }
        let key = ("http_request".to_string(), "http 404".to_string());
        assert_eq!(
            history.get(&key).unwrap().len(),
            MAX_ERROR_HISTORY_PER_KEY
        );
        // Oldest entries should have been evicted
        assert_eq!(history.get(&key).unwrap()[0].iteration, 5);
    }

    #[test]
    fn test_record_tool_error_separates_by_signature() {
        let mut history: HashMap<(String, String), Vec<ToolErrorEntry>> = HashMap::new();
        record_tool_error(&mut history, "http_request", "http 404", 1, "args1", "not found");
        record_tool_error(&mut history, "http_request", "http 500", 2, "args2", "server error");
        let key_404 = ("http_request".to_string(), "http 404".to_string());
        let key_500 = ("http_request".to_string(), "http 500".to_string());
        assert_eq!(history.get(&key_404).unwrap().len(), 1);
        assert_eq!(history.get(&key_500).unwrap().len(), 1);
    }

    #[test]
    fn test_record_tool_error_truncates_long_text() {
        let mut history: HashMap<(String, String), Vec<ToolErrorEntry>> = HashMap::new();
        let long_args = "x".repeat(1000);
        let long_error = "e".repeat(2000);
        record_tool_error(&mut history, "test_tool", "sig", 1, &long_args, &long_error);
        let key = ("test_tool".to_string(), "sig".to_string());
        let entry = &history.get(&key).unwrap()[0];
        assert_eq!(entry.arguments_summary.len(), ARGS_SUMMARY_MAX_CHARS);
        assert_eq!(entry.error_text.len(), ERROR_TEXT_MAX_CHARS);
    }

    #[test]
    fn test_extract_url_domain() {
        assert_eq!(
            extract_url_domain("https://api.example.com/v2/studies"),
            Some("api.example.com".to_string())
        );
        assert_eq!(
            extract_url_domain(r#"{"url": "https://clinicaltrials.gov/api/v2/studies"}"#),
            Some("clinicaltrials.gov".to_string())
        );
        assert_eq!(extract_url_domain("no urls here"), None);
        assert_eq!(extract_url_domain("http://localhost"), None); // no dot
    }

    #[test]
    fn test_find_relevant_skill_by_source_url() {
        let skills = vec![crate::skills::Skill {
            name: "test-api".to_string(),
            description: "Test API guide".to_string(),
            triggers: vec!["test-api".to_string()],
            body: "Use https://api.example.com/v2 as base URL".to_string(),
            origin: None,
            source: None,
            source_url: Some("https://api.example.com/oas/v2".to_string()),
            dir_path: None,
            resources: vec![],
        }];
        let result = find_relevant_skill_excerpt(
            &skills,
            "http_request",
            r#"{"url": "https://api.example.com/v2/data"}"#,
        );
        assert!(result.is_some());
        assert!(result.unwrap().contains("base URL"));
    }

    #[test]
    fn test_find_relevant_skill_by_trigger() {
        let skills = vec![crate::skills::Skill {
            name: "clinicaltrials-gov".to_string(),
            description: "ClinicalTrials.gov API".to_string(),
            triggers: vec!["clinicaltrials".to_string()],
            body: "Base URL: https://clinicaltrials.gov/api/v2".to_string(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: None,
            resources: vec![],
        }];
        let result = find_relevant_skill_excerpt(
            &skills,
            "http_request",
            r#"{"url": "https://api.clinicaltrials.gov/v2"}"#,
        );
        assert!(result.is_some());
        assert!(result.unwrap().contains("clinicaltrials.gov/api/v2"));
    }

    #[test]
    fn test_find_relevant_skill_by_tool_name_trigger() {
        let skills = vec![crate::skills::Skill {
            name: "http-tools".to_string(),
            description: "HTTP request guide".to_string(),
            triggers: vec!["http_request".to_string()],
            body: "Always check the base URL in the skill guide".to_string(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: None,
            resources: vec![],
        }];
        // Tool name "http_request" matches trigger "http_request" even when args don't match
        let result = find_relevant_skill_excerpt(
            &skills,
            "http_request",
            r#"{"url": "https://unrelated.example.com"}"#,
        );
        assert!(result.is_some());
        assert!(result.unwrap().contains("base URL"));
    }

    #[test]
    fn test_find_relevant_skill_no_match() {
        let skills = vec![crate::skills::Skill {
            name: "unrelated".to_string(),
            description: "Unrelated".to_string(),
            triggers: vec!["weather".to_string()],
            body: "Weather API guide".to_string(),
            origin: None,
            source: None,
            source_url: Some("https://weather.example.com".to_string()),
            dir_path: None,
            resources: vec![],
        }];
        let result = find_relevant_skill_excerpt(
            &skills,
            "http_request",
            r#"{"url": "https://clinicaltrials.gov/api/v2"}"#,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_reflection_response_valid() {
        let response = "\
ROOT_CAUSE: The agent is using the wrong hostname api.clinicaltrials.gov which does not exist
RECOMMENDED_ACTION: Change the base URL to https://clinicaltrials.gov/api/v2
LEARNING: Always use clinicaltrials.gov/api/v2 as the base URL, not api.clinicaltrials.gov";

        let diagnosis = parse_reflection_response(response, "http_request", "http 404 not found");
        assert!(diagnosis.is_some());
        let d = diagnosis.unwrap();
        assert!(d.root_cause.contains("wrong hostname"));
        assert!(d.recommended_action.contains("clinicaltrials.gov/api/v2"));
        assert!(d.learning.is_some());
        assert!(d.learning.unwrap().solution_summary.contains("clinicaltrials.gov/api/v2"));
    }

    #[test]
    fn test_parse_reflection_response_no_learning() {
        let response = "\
ROOT_CAUSE: Temporary server error
RECOMMENDED_ACTION: Retry with the same parameters
LEARNING: NONE";

        let diagnosis =
            parse_reflection_response(response, "http_request", "http 500 internal server error");
        assert!(diagnosis.is_some());
        let d = diagnosis.unwrap();
        assert!(d.learning.is_none());
    }

    #[test]
    fn test_parse_reflection_response_missing_fields() {
        let response = "ROOT_CAUSE: something went wrong";
        let diagnosis = parse_reflection_response(response, "test", "error");
        assert!(diagnosis.is_none()); // missing RECOMMENDED_ACTION
    }

    #[test]
    fn test_parse_reflection_response_empty() {
        let diagnosis = parse_reflection_response("", "test", "error");
        assert!(diagnosis.is_none());
    }

    #[test]
    fn test_build_reflection_prompt_includes_all_sections() {
        let errors = vec![
            ToolErrorEntry {
                iteration: 1,
                arguments_summary: r#"{"url":"https://api.example.com"}"#.to_string(),
                error_text: "HTTP 404 Not Found".to_string(),
            },
            ToolErrorEntry {
                iteration: 3,
                arguments_summary: r#"{"url":"https://api.example.com"}"#.to_string(),
                error_text: "HTTP 404 Not Found".to_string(),
            },
        ];
        let prompt = build_reflection_prompt(
            "http_request",
            "http 404 not found",
            &errors,
            "Find clinical trials",
            Some("Use https://example.com/api/v2"),
        );
        assert!(prompt.contains("TASK: Find clinical trials"));
        assert!(prompt.contains("FAILING TOOL: http_request"));
        assert!(prompt.contains("ERROR PATTERN: http 404 not found"));
        assert!(prompt.contains("Iteration 1"));
        assert!(prompt.contains("Iteration 3"));
        assert!(prompt.contains("RELEVANT SKILL/API GUIDE"));
        assert!(prompt.contains("https://example.com/api/v2"));
        assert!(prompt.contains("ROOT_CAUSE:"));
        assert!(prompt.contains("RECOMMENDED_ACTION:"));
        assert!(prompt.contains("LEARNING:"));
    }

    #[test]
    fn test_build_reflection_prompt_without_skill() {
        let errors = vec![ToolErrorEntry {
            iteration: 1,
            arguments_summary: "args".to_string(),
            error_text: "error".to_string(),
        }];
        let prompt =
            build_reflection_prompt("terminal", "exit code 1", &errors, "Run tests", None);
        assert!(!prompt.contains("RELEVANT SKILL"));
        assert!(prompt.contains("TASK: Run tests"));
    }
}
```

- [ ] **Step 2: Declare the module in phase_impl.rs**

In `src/agent/loop/tool_execution/phase_impl.rs`, add `mod reflection;` after `mod result_learning;` (line 6):

```rust
mod reflection;
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cargo test reflection --lib -- --nocapture 2>&1 | tail -20`
Expected: All tests in `reflection::tests` pass

- [ ] **Step 4: Run clippy**

Run: `cargo clippy --all-features -- -D warnings 2>&1 | head -30`
Expected: No new warnings

- [ ] **Step 5: Commit**

```bash
git add src/agent/loop/tool_execution/reflection.rs src/agent/loop/tool_execution/phase_impl.rs
git commit -m "feat: add reflection module with diagnosis types, parsing, and skill matching"
```

---

## Chunk 2: Wire Reflection into the Agent Loop

### Task 3: Add error history and reflection state to ToolExecutionCtx and ResultLearningState

**Files:**
- Modify: `src/agent/loop/tool_execution/types.rs:21-88` (ToolExecutionCtx struct)
- Modify: `src/agent/loop/tool_execution/result_learning.rs:30-59` (ResultLearningState struct)
- Modify: `src/agent/loop/main_loop.rs:221-224` (state initialization)
- Modify: `src/agent/loop/main_loop.rs:893-929` (ToolExecutionCtx construction)

- [ ] **Step 1: Add fields to ToolExecutionCtx**

In `src/agent/loop/tool_execution/types.rs`, add after line 87 (`pub tool_result_cache: ...`), before the closing `}`:

```rust
    pub tool_error_history: &'a mut HashMap<(String, String), Vec<super::reflection::ToolErrorEntry>>,
    pub reflection_completed: &'a mut HashSet<(String, String)>,
    /// Reflection learnings awaiting verification, keyed by (tool_name, signature).
    /// Only promoted to verified when the SAME tool+signature succeeds (not any tool).
    pub pending_reflection_solutions: &'a mut HashMap<(String, String), Vec<i64>>,
    /// Names of skills that were activated for this task during bootstrap.
    /// Used by reflection to scope skill cross-referencing to already-active skills only.
    pub active_skill_names: &'a [String],
```

- [ ] **Step 2: Add fields to ResultLearningState**

In `src/agent/loop/tool_execution/result_learning.rs`, add after `pub dirs_with_search_no_matches` (line 58), before the closing `}`:

```rust
    pub tool_error_history: &'a mut HashMap<(String, String), Vec<super::reflection::ToolErrorEntry>>,
    pub reflection_completed: &'a mut HashSet<(String, String)>,
    pub pending_reflection_solutions: &'a mut HashMap<(String, String), Vec<i64>>,
    /// Set by apply_result_learning when a semantic failure is recorded.
    /// Contains (tool_name, signature, count) for the just-incremented failure.
    /// Consumed by run.rs to trigger reflection without scanning the map.
    pub last_semantic_failure: &'a mut Option<(String, String, usize)>,
```

- [ ] **Step 3: Add active_skill_names to BootstrapData**

In `src/agent/loop/bootstrap/types.rs`, add after `pub session_summary` (line 40):

```rust
    pub active_skill_names: Vec<String>,
```

In `src/agent/runtime/system_prompt.rs`, modify `build_system_prompt_for_message()` to return `(String, Vec<String>)` — the system prompt string and the final active skill names. Extract the `final_skill_names` that are already computed at line ~233:

```rust
    // After the LLM confirmation block, at the end:
    let final_skill_names: Vec<String> = active_skills.iter().map(|s| s.name.clone()).collect();
    Ok((system_prompt_string, final_skill_names))
```

Update the call site in `src/agent/loop/bootstrap/run.rs` (line ~518) to capture both:

```rust
        let (system_prompt, active_skill_names) = self
            .build_system_prompt_for_message(...)
            .await?;
```

And pass `active_skill_names` into `BootstrapData`.

- [ ] **Step 4: Initialize state in main_loop.rs**

In `src/agent/loop/main_loop.rs`, add after `let mut tool_cooldown_until_iteration` (around line 224):

```rust
        let mut tool_error_history: HashMap<(String, String), Vec<super::tool_execution::reflection::ToolErrorEntry>> = HashMap::new();
        let mut reflection_completed: HashSet<(String, String)> = HashSet::new();
        let mut pending_reflection_solutions: HashMap<(String, String), Vec<i64>> = HashMap::new();
```

Destructure `active_skill_names` from `BootstrapData` alongside existing fields (line ~120):

```rust
            active_skill_names,
```

Note: `reflection` module types need to be pub(super) accessible. If the import path doesn't resolve, the module re-export in `phase_impl.rs` may need a `pub(in crate::agent)` on the module declaration, or use a `use` re-export.

- [ ] **Step 4: Thread fields through ToolExecutionCtx construction**

In `src/agent/loop/main_loop.rs`, in the `ToolExecutionCtx` construction block (around line 893-929), add after `tool_result_cache`:

```rust
                    tool_error_history: &mut tool_error_history,
                    reflection_completed: &mut reflection_completed,
                    pending_reflection_solutions: &mut pending_reflection_solutions,
                    active_skill_names: &active_skill_names,
```

- [ ] **Step 5: Thread fields through ResultLearningState construction in run.rs**

In `src/agent/loop/tool_execution/run.rs`, find the `ResultLearningState` construction (around line 1470-1494), add after `dirs_with_search_no_matches`:

```rust
                tool_error_history: ctx.tool_error_history,
                reflection_completed: ctx.reflection_completed,
                pending_reflection_solutions: ctx.pending_reflection_solutions,
                last_semantic_failure: &mut last_semantic_failure_slot,
```

Note: `last_semantic_failure_slot` is a local `let mut last_semantic_failure_slot: Option<(String, String, usize)> = None;` declared just before the `ResultLearningState` construction in `run.rs`. It resets each iteration — consumed by the reflection trigger block below.

- [ ] **Step 6: Verify compilation**

Run: `cargo build 2>&1 | tail -20`
Expected: Compiles successfully (no logic changes yet, just plumbing)

- [ ] **Step 7: Commit**

```bash
git add src/agent/loop/tool_execution/types.rs src/agent/loop/tool_execution/result_learning.rs src/agent/loop/main_loop.rs src/agent/loop/tool_execution/run.rs src/agent/loop/bootstrap/types.rs src/agent/runtime/system_prompt.rs src/agent/loop/bootstrap/run.rs
git commit -m "feat: thread reflection state fields through agent loop (error history, reflection completed, pending solutions, active skills)"
```

---

### Task 4: Accumulate error entries and surface the just-incremented signature

**Files:**
- Modify: `src/agent/loop/tool_execution/result_learning.rs:160-189` (record_semantic_failure_signature)
- Modify: `src/agent/loop/tool_execution/result_learning.rs:266-271` (after record_semantic_failure_signature call)
- Modify: `src/agent/loop/tool_execution/result_learning.rs:193-202` (apply_result_learning return type)

- [ ] **Step 1: Make record_semantic_failure_signature return the signature string**

Change `record_semantic_failure_signature` (line ~170) to return `(usize, String)` instead of just `usize`:

```rust
fn record_semantic_failure_signature(
    tool_failure_count: &mut HashMap<String, usize>,
    tool_failure_signatures: &mut HashMap<(String, String), usize>,
    tool_name: &str,
    error_text: &str,
) -> (usize, String) {  // returns (count, signature)
    let signature = derive_failure_signature(error_text);
    let count = tool_failure_signatures
        .entry((tool_name.to_string(), signature.clone()))
        .or_insert(0);
    *count += 1;
    let repeated_count = *count;
    let per_tool = tool_failure_count
        .entry(tool_name.to_string())
        .or_insert(repeated_count);
    if repeated_count > *per_tool {
        *per_tool = repeated_count;
    }
    (repeated_count, signature)
}
```

Update all call sites to destructure the tuple (there's only one call site in `apply_result_learning`).

- [ ] **Step 1b: Update existing unit tests for the new return type**

The existing tests in `result_learning.rs` (`test_repeated_signature_persists_without_reset` at line ~783, `test_signature_counter_only_rises_on_repeated_same_error` at line ~806) assert `usize` return values. Update them to destructure the `(usize, String)` tuple:

```rust
    #[test]
    fn test_repeated_signature_persists_without_reset() {
        let mut counts = HashMap::new();
        let mut signatures = HashMap::new();
        let (first, _sig1) = record_semantic_failure_signature(
            &mut counts, &mut signatures, "read_file",
            "Error: missing required field `path`",
        );
        let (second, sig2) = record_semantic_failure_signature(
            &mut counts, &mut signatures, "read_file",
            "Error: missing required field `path`",
        );
        assert_eq!(first, 1);
        assert_eq!(second, 2);
        assert_eq!(counts.get("read_file").copied(), Some(2));
        // Verify the signature is returned correctly
        assert!(!sig2.is_empty());
    }

    #[test]
    fn test_signature_counter_only_rises_on_repeated_same_error() {
        let mut counts = HashMap::new();
        let mut signatures = HashMap::new();

        let (first, sig_a) = record_semantic_failure_signature(
            &mut counts, &mut signatures, "read_file",
            "Error: missing required field `path`",
        );
        let (second_unique, sig_b) = record_semantic_failure_signature(
            &mut counts, &mut signatures, "read_file",
            "Error: permission denied",
        );
        let (third_repeat, sig_c) = record_semantic_failure_signature(
            &mut counts, &mut signatures, "read_file",
            "Error: missing required field `path`",
        );

        assert_eq!(first, 1);
        assert_eq!(second_unique, 1);
        assert_eq!(third_repeat, 2);
        // Different error texts produce different signatures
        assert_ne!(sig_a, sig_b);
        // Same error text produces the same signature
        assert_eq!(sig_a, sig_c);
    }
```

- [ ] **Step 2: Add error recording and surface the signature**

In `apply_result_learning()`, update the semantic failure block:

```rust
                let (semantic_count, failure_signature) = record_semantic_failure_signature(
                    state.tool_failure_count,
                    state.tool_failure_signatures,
                    &tc.name,
                    &base_error,
                );

                // Accumulate error entry for reflection, keyed by (tool, signature)
                super::reflection::record_tool_error(
                    state.tool_error_history,
                    &tc.name,
                    &failure_signature,
                    env.iteration,
                    &tc.arguments,
                    &base_error,
                );
```

- [ ] **Step 3: Add a `last_semantic_failure` output field**

Add a new field to `ResultLearningState` (or add a return value from `apply_result_learning`) to surface the just-incremented `(tool_name, signature, count)` triple. The simplest approach is a new field:

```rust
    pub last_semantic_failure: &'a mut Option<(String, String, usize)>,  // (tool, sig, count)
```

Set it inside `apply_result_learning` after `record_semantic_failure_signature`:

```rust
                *state.last_semantic_failure = Some((tc.name.clone(), failure_signature.clone(), semantic_count));
```

This goes right after `let semantic_count = record_semantic_failure_signature(...)` and before the `if semantic_count == 1 && looks_like_missing_goal_id_error(...)` block.

- [ ] **Step 2: Verify compilation and tests**

Run: `cargo test --lib 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add src/agent/loop/tool_execution/result_learning.rs
git commit -m "feat: accumulate tool error entries on semantic failures for reflection"
```

---

### Task 5: Call maybe_trigger_reflection from run.rs

**Files:**
- Modify: `src/agent/loop/tool_execution/run.rs:1495-1510` (after apply_result_learning)

- [ ] **Step 1: Add the reflection trigger call**

In `run.rs`, after the `apply_result_learning` call block (after line ~1509, after the `commit_state!(); return Ok(outcome);` block), add before the `if !is_error {` line:

```rust
            // Trigger reflection using the exact signature that just incremented
            // (avoids non-deterministic map scanning and wrong-signature diagnosis)
            if let Some((ref fail_tool, ref fail_sig, fail_count)) =
                *learning_state.last_semantic_failure
            {
                if fail_count == 2 {
                    // Error history is keyed by (tool, signature) — only same-signature errors
                    let history_key = (fail_tool.clone(), fail_sig.clone());
                    let error_history = learning_state
                        .tool_error_history
                        .get(&history_key)
                        .cloned()
                        .unwrap_or_default();

                    // Get active skills for scoped cross-referencing
                    let skills_snapshot = self.skill_cache.get();
                    let active_skills: Vec<_> = skills_snapshot
                        .iter()
                        .filter(|s| ctx.active_skill_names.contains(&s.name))
                        .cloned()
                        .collect();

                    if let Some(diagnosis) = self
                        .maybe_trigger_reflection(
                            fail_tool,
                            fail_sig,
                            2,
                            &error_history,
                            ctx.user_text,
                            &active_skills,
                            learning_state.reflection_completed,
                        )
                        .await
                    {
                        info!(
                            tool = %fail_tool,
                            root_cause = %diagnosis.root_cause,
                            "Reflection diagnosis produced"
                        );
                        learning_state.pending_system_messages.push(
                            SystemDirective::ReflectionDiagnosis {
                                tool_name: fail_tool.clone(),
                                root_cause: diagnosis.root_cause,
                                recommended_action: diagnosis.recommended_action,
                            },
                        );
                        // Store as UNVERIFIED (success_count=0).
                        // Track in tool-scoped pending_reflection_solutions so
                        // verification only fires when THIS tool succeeds,
                        // not any unrelated tool.
                        if let Some(draft) = diagnosis.learning {
                            if let Some(solution_id) =
                                super::reflection::store_reflection_learning(
                                    &self.state, draft,
                                )
                                .await
                            {
                                learning_state
                                    .pending_reflection_solutions
                                    .entry((fail_tool.clone(), fail_sig.clone()))
                                    .or_default()
                                    .push(solution_id);
                            }
                        }
                    }
                }
                // Clear after processing
                *learning_state.last_semantic_failure = None;
            }
```

**Verification on same-tool success:** In `apply_result_learning`, in the success path (the `if !state.learning_ctx.errors.is_empty()` block around line ~658), add tool-specific verification:

```rust
        // Verify reflection learnings: only promote the EXACT (tool, signature) that
        // was reflected on, not all pending entries for this tool. This prevents a
        // successful http_request to one endpoint from incorrectly verifying a
        // reflection learning about a completely different failure signature.
        //
        // Find the most recent failure signature for this tool from tool_error_history.
        // That's the signature we just recovered from.
        let recovered_signature: Option<String> = state
            .tool_error_history
            .keys()
            .filter(|(t, _)| t == &tc.name)
            .max_by_key(|key| {
                state.tool_error_history.get(key)
                    .and_then(|entries| entries.last())
                    .map(|e| e.iteration)
                    .unwrap_or(0)
            })
            .map(|(_, sig)| sig.clone());

        if let Some(sig) = recovered_signature {
            let key = (tc.name.clone(), sig);
            if let Some(solution_ids) = state.pending_reflection_solutions.remove(&key) {
                for solution_id in solution_ids {
                    let state_store = self.state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = state_store
                            .update_error_solution_outcome(solution_id, true)
                            .await
                        {
                            warn!(solution_id, error = %e, "Failed to verify reflection learning");
                        }
                    });
                }
            }
        }
```

This is separate from the existing `pending_error_solution_ids` mechanism (which is tool-agnostic and handles diagnostic-hint verification). Reflection solutions use their own `(tool, signature)`-keyed map, and verification only promotes the exact signature whose most recent failure entry has the highest iteration number — i.e., the failure we most recently recovered from.

- [ ] **Step 2: Verify compilation**

Run: `cargo build 2>&1 | tail -20`
Expected: Compiles successfully

- [ ] **Step 3: Run all tests**

Run: `cargo test 2>&1 | tail -10`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/agent/loop/tool_execution/run.rs
git commit -m "feat: trigger reflection LLM call on 2nd same-signature tool failure"
```

---

## Chunk 3: Force-Text Promise Prevention + Integration Test

### Task 6: Add promise prevention to ForceTextToolLimitReached

**Files:**
- Modify: `src/agent/loop/system_directives.rs:270-285` (ForceTextToolLimitReached render)

- [ ] **Step 1: Update the existing test**

In `system_directives.rs`, find the `force_text_tool_limit_render_preserves_sections` test (line ~378). Add an assertion for the new text:

```rust
assert!(rendered.contains("Do NOT promise future actions"));
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test force_text_tool_limit_render -- --nocapture`
Expected: FAIL — new assertion not yet satisfied

- [ ] **Step 3: Add promise prevention text to the render**

In the `ForceTextToolLimitReached` render arm (line ~274), append to the format string, after the `Focus only on concrete results and outcomes for the CURRENT task.` line:

```rust
                 \n\nDo NOT promise future actions like \"let me try...\" or \
                 \"I'll search for...\" — your tools have been disabled. \
                 Report what you found, what failed, and what the user can try instead.",
```

Replace the existing closing `"` of the format string so it ends with this new text.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test force_text_tool_limit_render -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/loop/system_directives.rs
git commit -m "feat: add promise prevention to ForceTextToolLimitReached directive"
```

---

### Task 7: Integration test for the full reflection flow

**Files:**
- Modify: `src/agent/loop/tool_execution/reflection.rs` (add integration-style test)

- [ ] **Step 1: Write the integration test**

Add to `reflection.rs` `#[cfg(test)]` module:

```rust
    #[tokio::test]
    async fn test_maybe_trigger_reflection_fires_on_second_failure() {
        use crate::testing::{setup_test_agent, MockProvider};

        // Script the mock with the reflection response
        let provider = MockProvider::with_responses(vec![
            MockProvider::text_response(
                "ROOT_CAUSE: Wrong API endpoint used\n\
                 RECOMMENDED_ACTION: Use the correct base URL from the skill guide\n\
                 LEARNING: Always check the skill guide for the correct base URL before making API calls",
            ),
        ]);
        let harness = setup_test_agent(provider).await;

        let mut reflection_completed: HashSet<(String, String)> = HashSet::new();
        let error_history = vec![
            ToolErrorEntry {
                iteration: 1,
                arguments_summary: r#"{"url":"https://wrong.example.com/api"}"#.to_string(),
                error_text: "HTTP 404 Not Found".to_string(),
            },
            ToolErrorEntry {
                iteration: 3,
                arguments_summary: r#"{"url":"https://wrong.example.com/api"}"#.to_string(),
                error_text: "HTTP 404 Not Found".to_string(),
            },
        ];
        let active_skills: Vec<crate::skills::Skill> = vec![];

        // Should trigger on count == 2
        let result = harness
            .agent
            .maybe_trigger_reflection(
                "http_request",
                "http 404 not found",
                2,
                &error_history,
                "Find clinical trials near me",
                &active_skills,
                &mut reflection_completed,
            )
            .await;

        assert!(result.is_some(), "Reflection should fire on 2nd failure");
        let diagnosis = result.unwrap();
        assert!(diagnosis.root_cause.contains("Wrong API endpoint"));
        assert!(diagnosis.recommended_action.contains("correct base URL"));
        assert!(diagnosis.learning.is_some());

        // Should NOT trigger again for same pair
        let result2 = harness
            .agent
            .maybe_trigger_reflection(
                "http_request",
                "http 404 not found",
                2,
                &error_history,
                "Find clinical trials near me",
                &active_skills,
                &mut reflection_completed,
            )
            .await;
        assert!(
            result2.is_none(),
            "Reflection should not re-trigger for same (tool, signature)"
        );
    }

    #[tokio::test]
    async fn test_maybe_trigger_reflection_skips_on_first_failure() {
        use crate::testing::{setup_test_agent, MockProvider};

        let provider = MockProvider::new();
        let harness = setup_test_agent(provider).await;
        let mut reflection_completed: HashSet<(String, String)> = HashSet::new();
        let active_skills: Vec<crate::skills::Skill> = vec![];

        let result = harness
            .agent
            .maybe_trigger_reflection(
                "http_request",
                "http 404",
                1, // count == 1, should not trigger
                &[],
                "test task",
                &active_skills,
                &mut reflection_completed,
            )
            .await;

        assert!(result.is_none(), "Reflection should not fire on 1st failure");
    }

    #[tokio::test]
    async fn test_maybe_trigger_reflection_skips_on_third_failure() {
        use crate::testing::{setup_test_agent, MockProvider};

        let provider = MockProvider::new();
        let harness = setup_test_agent(provider).await;
        let mut reflection_completed: HashSet<(String, String)> = HashSet::new();
        let active_skills: Vec<crate::skills::Skill> = vec![];

        let result = harness
            .agent
            .maybe_trigger_reflection(
                "http_request",
                "http 404",
                3, // count == 3, should not trigger
                &[],
                "test task",
                &active_skills,
                &mut reflection_completed,
            )
            .await;

        assert!(
            result.is_none(),
            "Reflection should not fire on 3rd failure (only 2nd)"
        );
    }
```

- [ ] **Step 2: Run the integration tests**

Run: `cargo test maybe_trigger_reflection -- --nocapture 2>&1 | tail -20`
Expected: All 3 tests pass

- [ ] **Step 3: Run the full test suite**

Run: `cargo test 2>&1 | tail -10`
Expected: All tests pass

- [ ] **Step 4: Run clippy and fmt**

Run: `cargo fmt && cargo clippy --all-features -- -D warnings 2>&1 | head -30`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add src/agent/loop/tool_execution/reflection.rs
git commit -m "test: add integration tests for reflection trigger logic"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cargo test --all-features 2>&1 | tail -20`
Expected: All tests pass

- [ ] **Step 2: Run clippy with all features**

Run: `cargo clippy --all-features -- -D warnings 2>&1 | tail -20`
Expected: No warnings

- [ ] **Step 3: Check formatting**

Run: `cargo fmt --check 2>&1 | head -20`
Expected: Clean (or only pre-existing issues)

- [ ] **Step 4: Final commit if any formatting changes**

```bash
cargo fmt
git add -A
git commit -m "style: format reflection feedback loop code"
```
