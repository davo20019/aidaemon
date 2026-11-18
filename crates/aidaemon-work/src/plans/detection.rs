//! Explicit plan routing.
//!
//! Semantic task planning belongs to the model-facing assessment contract.
//! This module recognizes only typed/explicit control markers; user prose is
//! never converted into orchestration state by a vocabulary list.

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

/// Resolve explicit plan-control syntax. Ordinary natural language returns
/// `None` and is assessed semantically by the model in the active runtime.
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
             FIRST call the track_requirements tool with the full checklist of concrete requirements \
             for this request (include deliverables like sending a file). As you finish each one, call \
             track_requirements again with that item marked 'completed'. For each step that modifies \
             external state (deploys, publishes, sends, pushes), include a verification step that confirms \
             the change was applied correctly. Check prerequisites (committed changes, installed \
             dependencies, correct configuration) before executing mutations. Do not claim you are done \
             while any item is still pending, and never claim success without verification.",
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
    fn test_suggest_hint_instructs_track_requirements() {
        let hint = get_plan_suggestion_prompt(&PlanTrigger::Suggest("deployment".to_string()))
            .expect("suggest trigger should yield a hint");
        assert!(
            hint.contains("track_requirements"),
            "hint must steer the model to the checklist tool: {hint}"
        );
        // None/AutoCreate triggers do not produce a suggestion hint.
        assert!(get_plan_suggestion_prompt(&PlanTrigger::None).is_none());
    }

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

    #[test]
    fn ordinary_language_never_routes_through_phrase_lists() {
        assert_eq!(should_create_plan("Deploy the app"), PlanTrigger::None);
        assert_eq!(should_create_plan("fix the bug"), PlanTrigger::None);
        assert_eq!(should_create_plan("what time is it"), PlanTrigger::None);
        assert_eq!(
            should_create_plan(
                "Publish the new version of the package to npm and update the changelog",
            ),
            PlanTrigger::None
        );
        assert_eq!(
            should_create_plan(
                "First commit the changes, then build, verify, and deploy the release"
            ),
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
