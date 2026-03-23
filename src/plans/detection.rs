//! Plan detection heuristics.
//!
//! Determines when a task should have a plan created for it.
//! Detects multi-step tasks, high-stakes operations, and tasks requiring
//! post-execution verification.

/// Reasons why a plan should be created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanTrigger {
    /// Auto-create immediately (high-stakes operation)
    AutoCreate(String),
    /// Suggest to LLM via prompt hint
    Suggest(String),
    /// No plan needed
    None,
}

impl PlanTrigger {
    /// Check if this trigger indicates a plan should be created or suggested.
    pub fn should_plan(&self) -> bool {
        !matches!(self, PlanTrigger::None)
    }

    /// Check if this trigger requires automatic plan creation.
    pub fn is_auto_create(&self) -> bool {
        matches!(self, PlanTrigger::AutoCreate(_))
    }

    /// Get the reason string, if any.
    pub fn reason(&self) -> Option<&str> {
        match self {
            PlanTrigger::AutoCreate(r) | PlanTrigger::Suggest(r) => Some(r),
            PlanTrigger::None => None,
        }
    }
}

/// Word-boundary keyword matching (same semantics as `contains_keyword_as_words`
/// in `agent/intent/intent_routing.rs`). Avoids substring false positives.
fn has_keyword(text: &str, keyword: &str) -> bool {
    let normalize = |w: &str| -> String {
        w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
            .to_lowercase()
    };
    let text_words: Vec<String> = text
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty())
        .collect();
    let kw_words: Vec<String> = keyword
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty())
        .collect();
    if kw_words.is_empty() {
        return false;
    }
    text_words
        .windows(kw_words.len())
        .any(|window| window == kw_words.as_slice())
}

/// Analyze a user message to determine if a plan should be created.
pub fn should_create_plan(user_message: &str) -> PlanTrigger {
    let trimmed = user_message.trim();
    if trimmed.is_empty() {
        return PlanTrigger::None;
    }

    // Explicit command mode (no lexical guessing).
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("/plan auto") {
        let reason = rest.trim();
        return PlanTrigger::AutoCreate(if reason.is_empty() {
            "explicit /plan auto".to_string()
        } else {
            reason.to_string()
        });
    }
    if let Some(rest) = lower.strip_prefix("/plan suggest") {
        let reason = rest.trim();
        return PlanTrigger::Suggest(if reason.is_empty() {
            "explicit /plan suggest".to_string()
        } else {
            reason.to_string()
        });
    }

    // Explicit inline markers.
    if let Some(reason) = parse_plan_marker(trimmed, "PLAN_AUTO") {
        return PlanTrigger::AutoCreate(reason);
    }
    if let Some(reason) = parse_plan_marker(trimmed, "PLAN_SUGGEST") {
        return PlanTrigger::Suggest(reason);
    }

    // --- Heuristic detection ---

    // Short messages (< 8 words) are unlikely multi-step tasks.
    let word_count = trimmed.split_whitespace().count();
    if word_count < 8 {
        return PlanTrigger::None;
    }

    // High-stakes external operations: deploy, publish, release, migrate, etc.
    // These always benefit from structured execution with verification.
    let external_ops: &[&str] = &[
        "deploy",
        "publish",
        "release",
        "migrate",
        "push to production",
        "push to prod",
        "ship to production",
        "go live",
    ];
    for op in external_ops {
        if has_keyword(&lower, op) {
            return PlanTrigger::Suggest(format!("task involving {}", op));
        }
    }

    // Sequential markers: explicit multi-step intent.
    let sequential_markers: &[&str] = &[
        "first",
        "then",
        "after that",
        "once that",
        "and then",
        "next",
        "finally",
        "step 1",
        "step 2",
        "1.",
    ];
    let seq_count = sequential_markers
        .iter()
        .filter(|m| has_keyword(&lower, m))
        .count();
    if seq_count >= 2 {
        return PlanTrigger::Suggest("multi-step task with sequential dependencies".to_string());
    }

    // Verification-required markers: user explicitly wants confirmation.
    let verify_markers: &[&str] = &[
        "and verify",
        "and validate",
        "and confirm",
        "make sure",
        "and check",
        "and test",
    ];
    let has_verify = verify_markers.iter().any(|m| has_keyword(&lower, m));
    if has_verify {
        // Only suggest if there's also an action verb (not just "make sure X is true").
        let action_verbs: &[&str] = &[
            "deploy",
            "build",
            "create",
            "install",
            "configure",
            "set up",
            "update",
            "push",
            "run",
            "execute",
            "write",
            "generate",
            "migrate",
        ];
        if action_verbs.iter().any(|v| has_keyword(&lower, v)) {
            return PlanTrigger::Suggest(
                "task with action and verification requirement".to_string(),
            );
        }
    }

    // Multi-sentence imperative detection: 3+ sentences starting with
    // imperative verbs = likely multi-step task.
    let sentences: Vec<&str> = trimmed
        .split(['.', '!', '\n'])
        .map(|s| s.trim())
        .filter(|s| s.len() > 3)
        .collect();
    if sentences.len() >= 3 {
        let imperative_verbs: &[&str] = &[
            "deploy",
            "build",
            "create",
            "install",
            "configure",
            "set up",
            "update",
            "push",
            "run",
            "execute",
            "write",
            "generate",
            "migrate",
            "publish",
            "release",
            "commit",
            "add",
            "remove",
            "delete",
            "fix",
            "check",
            "verify",
            "test",
            "start",
            "stop",
            "kill",
            "restart",
            "open",
            "close",
            "send",
            "fetch",
            "download",
            "upload",
        ];
        let imperative_count = sentences
            .iter()
            .filter(|s| {
                let s_lower = s.to_ascii_lowercase();
                imperative_verbs.iter().any(|v| s_lower.starts_with(v))
            })
            .count();
        if imperative_count >= 3 {
            return PlanTrigger::Suggest("multi-step task with multiple actions".to_string());
        }
    }

    PlanTrigger::None
}

/// Parse explicit marker forms:
/// - [PLAN_AUTO]
/// - [PLAN_AUTO: reason text]
fn parse_plan_marker(text: &str, marker: &str) -> Option<String> {
    let upper = text.to_ascii_uppercase();
    let open = format!("[{}", marker);
    let start = upper.find(&open)?;
    let rest = &text[start + open.len()..];
    let end = rest.find(']')?;
    let inside = rest[..end].trim();
    if inside.is_empty() {
        return Some(format!("explicit {}", marker.to_ascii_lowercase()));
    }
    if let Some(reason) = inside.strip_prefix(':') {
        let reason = reason.trim();
        if reason.is_empty() {
            return Some(format!("explicit {}", marker.to_ascii_lowercase()));
        }
        return Some(reason.to_string());
    }
    None
}

/// Get a prompt hint for suggesting structured execution to the LLM.
pub fn get_plan_suggestion_prompt(trigger: &PlanTrigger) -> Option<String> {
    match trigger {
        PlanTrigger::Suggest(reason) => Some(format!(
            "[SYSTEM] This looks like a {} that requires structured execution. \
             Break it into discrete steps BEFORE executing. For each step that modifies external state \
             (deploys, publishes, sends, pushes), include a verification step that confirms the change \
             was applied correctly. Check prerequisites (committed changes, installed dependencies, \
             correct configuration) before executing mutations. Never claim success without verification.",
            reason
        )),
        PlanTrigger::AutoCreate(_) => None,
        PlanTrigger::None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_auto_command() {
        let trigger = should_create_plan("/plan auto production deployment");
        assert_eq!(
            trigger,
            PlanTrigger::AutoCreate("production deployment".to_string())
        );
    }

    #[test]
    fn test_explicit_suggest_command() {
        let trigger = should_create_plan("/plan suggest split into phases");
        assert_eq!(
            trigger,
            PlanTrigger::Suggest("split into phases".to_string())
        );
    }

    #[test]
    fn test_explicit_auto_marker_with_reason() {
        let trigger = should_create_plan("Please do this. [PLAN_AUTO: high-risk change]");
        assert_eq!(
            trigger,
            PlanTrigger::AutoCreate("high-risk change".to_string())
        );
    }

    #[test]
    fn test_explicit_suggest_marker_without_reason() {
        let trigger = should_create_plan("Please walk through this [PLAN_SUGGEST]");
        assert_eq!(
            trigger,
            PlanTrigger::Suggest("explicit plan_suggest".to_string())
        );
    }

    // --- Heuristic detection tests ---

    #[test]
    fn test_short_messages_no_plan() {
        // Short messages never trigger heuristic detection.
        assert_eq!(should_create_plan("Deploy the app"), PlanTrigger::None);
        assert_eq!(should_create_plan("fix the bug"), PlanTrigger::None);
        assert_eq!(should_create_plan("what time is it"), PlanTrigger::None);
    }

    #[test]
    fn test_deploy_heuristic() {
        let trigger =
            should_create_plan("Deploy the latest changes to Cloudflare Pages and check the site");
        assert!(trigger.should_plan());
        assert!(matches!(trigger, PlanTrigger::Suggest(_)));
    }

    #[test]
    fn test_publish_heuristic() {
        let trigger = should_create_plan(
            "Publish the new version of the package to npm and update the changelog",
        );
        assert!(trigger.should_plan());
    }

    #[test]
    fn test_release_heuristic() {
        let trigger =
            should_create_plan("Release version 2.0 with the new features and tag it in git");
        assert!(trigger.should_plan());
    }

    #[test]
    fn test_migrate_heuristic() {
        let trigger = should_create_plan(
            "Migrate the database schema to add the new user_preferences table and verify",
        );
        assert!(trigger.should_plan());
    }

    #[test]
    fn test_sequential_markers() {
        let trigger = should_create_plan(
            "First commit the changes, then build the project, and finally push to the remote",
        );
        assert!(trigger.should_plan());
        assert!(matches!(trigger, PlanTrigger::Suggest(ref r) if r.contains("sequential")));
    }

    #[test]
    fn test_verification_with_action() {
        // "deploy" fires the external-ops heuristic first.
        let trigger = should_create_plan(
            "Build the project and deploy it, and verify the site loads correctly",
        );
        assert!(trigger.should_plan());

        // Without a high-stakes keyword, verification + action triggers separately.
        let trigger = should_create_plan(
            "Run the build pipeline and generate the artifacts, and verify the output is correct",
        );
        assert!(trigger.should_plan());
        assert!(matches!(trigger, PlanTrigger::Suggest(ref r) if r.contains("verification")));
    }

    #[test]
    fn test_multi_sentence_imperative() {
        let trigger = should_create_plan(
            "Start the dev server. Run the test suite. Check the coverage report. Kill the server.",
        );
        assert!(trigger.should_plan());
        assert!(matches!(trigger, PlanTrigger::Suggest(ref r) if r.contains("multiple actions")));
    }

    #[test]
    fn test_simple_questions_no_plan() {
        assert_eq!(
            should_create_plan("What is the current status of the deployment pipeline?"),
            PlanTrigger::None
        );
        assert_eq!(
            should_create_plan("Can you explain how the router module works?"),
            PlanTrigger::None
        );
    }

    #[test]
    fn test_single_action_no_plan() {
        // Single actions without sequential/verification markers don't trigger.
        assert_eq!(
            should_create_plan("Create a new file called config.toml with the database settings"),
            PlanTrigger::None
        );
    }

    #[test]
    fn test_plan_suggestion_prompt_content() {
        let trigger = PlanTrigger::Suggest("multi-step task".to_string());
        let prompt = get_plan_suggestion_prompt(&trigger);
        assert!(prompt.is_some());
        let text = prompt.unwrap();
        assert!(text.contains("structured execution"));
        assert!(text.contains("verification"));
        assert!(text.contains("prerequisites"));

        let trigger = PlanTrigger::None;
        assert!(get_plan_suggestion_prompt(&trigger).is_none());
    }
}
