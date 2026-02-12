//! Plan detection heuristics.
//!
//! Determines when a task should have a plan created for it.

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

/// Analyze a user message to determine if a plan should be created.
pub fn should_create_plan(user_message: &str) -> PlanTrigger {
    let trimmed = user_message.trim();
    if trimmed.is_empty() {
        return PlanTrigger::None;
    }

    // Explicit command mode (no lexical guessing).
    // Examples:
    // - /plan auto production deployment
    // - /plan suggest break into phases
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

/// Get a prompt hint for suggesting plan creation to the LLM.
pub fn get_plan_suggestion_prompt(trigger: &PlanTrigger) -> Option<String> {
    match trigger {
        PlanTrigger::Suggest(reason) => Some(format!(
            "\n\n**Note:** This looks like a {} that might benefit from a step-by-step plan. \
             Consider using the `plan_manager` tool with action=\"create\" to break it down into discrete steps. \
             This helps with tracking progress and enables recovery if something goes wrong.",
            reason
        )),
        PlanTrigger::AutoCreate(_) => None, // Auto-created, no need to suggest
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

    #[test]
    fn test_no_heuristic_plan_detection() {
        let trigger = should_create_plan("Deploy the app to production");
        assert_eq!(trigger, PlanTrigger::None);

        let trigger = should_create_plan("Create a plan for the feature");
        assert_eq!(trigger, PlanTrigger::None);

        let trigger = should_create_plan("First do A, then do B");
        assert_eq!(trigger, PlanTrigger::None);
    }

    #[test]
    fn test_plan_suggestion_prompt() {
        let trigger = PlanTrigger::Suggest("refactoring task".to_string());
        let prompt = get_plan_suggestion_prompt(&trigger);
        assert!(prompt.is_some());
        assert!(prompt.unwrap().contains("refactoring task"));

        let trigger = PlanTrigger::None;
        assert!(get_plan_suggestion_prompt(&trigger).is_none());
    }
}
