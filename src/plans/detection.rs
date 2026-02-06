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
    let lower = user_message.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // === AUTO-CREATE: High-stakes operations ===
    // These are too risky to leave to LLM discretion

    // Production deployments
    if (lower.contains("deploy") || lower.contains("release") || lower.contains("ship"))
        && (lower.contains("prod") || lower.contains("production") || lower.contains("live"))
    {
        return PlanTrigger::AutoCreate("production deployment".to_string());
    }

    // Database migrations
    if lower.contains("migrat") && (lower.contains("database") || lower.contains("db") || lower.contains("schema")) {
        return PlanTrigger::AutoCreate("database migration".to_string());
    }

    // Bulk deletions
    if lower.contains("delete") && (lower.contains("all") || lower.contains("every") || lower.contains("bulk")) {
        return PlanTrigger::AutoCreate("bulk deletion".to_string());
    }

    // System upgrades
    if lower.contains("upgrade") && (lower.contains("system") || lower.contains("server") || lower.contains("infrastructure")) {
        return PlanTrigger::AutoCreate("system upgrade".to_string());
    }

    // === SUGGEST: Complex but not critical ===
    // LLM decides based on context

    // User explicitly wants planning
    if lower.contains("step by step")
        || lower.contains("step-by-step")
        || lower.contains("create a plan")
        || lower.contains("make a plan")
        || lower.contains("plan out")
        || lower.contains("plan for")
    {
        return PlanTrigger::Suggest("user requested planning".to_string());
    }

    // Refactoring tasks (often multi-step)
    if lower.contains("refactor") {
        return PlanTrigger::Suggest("refactoring task".to_string());
    }

    // Implementation tasks
    if lower.contains("implement") && words.len() > 10 {
        return PlanTrigger::Suggest("implementation task".to_string());
    }

    // Multi-phase indicators
    let multi_phase_keywords = [
        "then",
        "after that",
        "next",
        "finally",
        "first",
        "second",
        "third",
        "and also",
        "as well as",
        "followed by",
    ];
    let multi_count = multi_phase_keywords
        .iter()
        .filter(|&&kw| lower.contains(kw))
        .count();
    if multi_count >= 2 {
        return PlanTrigger::Suggest("multi-phase task".to_string());
    }

    // Long requests (>50 words often indicate complex tasks)
    if words.len() > 50 {
        return PlanTrigger::Suggest("complex request".to_string());
    }

    // Build/setup tasks
    if (lower.contains("set up") || lower.contains("setup") || lower.contains("build"))
        && (lower.contains("project") || lower.contains("environment") || lower.contains("pipeline"))
    {
        return PlanTrigger::Suggest("setup task".to_string());
    }

    PlanTrigger::None
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
    fn test_auto_create_production_deploy() {
        let trigger = should_create_plan("Deploy the app to production");
        assert!(trigger.is_auto_create());

        let trigger = should_create_plan("release to prod");
        assert!(trigger.is_auto_create());

        let trigger = should_create_plan("ship it to production servers");
        assert!(trigger.is_auto_create());
    }

    #[test]
    fn test_auto_create_database_migration() {
        let trigger = should_create_plan("Run the database migration");
        assert!(trigger.is_auto_create());

        let trigger = should_create_plan("migrate the schema");
        assert!(trigger.is_auto_create());
    }

    #[test]
    fn test_auto_create_bulk_delete() {
        let trigger = should_create_plan("Delete all the old logs");
        assert!(trigger.is_auto_create());

        let trigger = should_create_plan("bulk delete inactive users");
        assert!(trigger.is_auto_create());
    }

    #[test]
    fn test_suggest_user_request() {
        let trigger = should_create_plan("Help me step by step with this");
        assert_eq!(trigger, PlanTrigger::Suggest("user requested planning".to_string()));

        let trigger = should_create_plan("Create a plan for the feature");
        assert_eq!(trigger, PlanTrigger::Suggest("user requested planning".to_string()));
    }

    #[test]
    fn test_suggest_refactoring() {
        let trigger = should_create_plan("Refactor the authentication module");
        assert_eq!(trigger, PlanTrigger::Suggest("refactoring task".to_string()));
    }

    #[test]
    fn test_suggest_multi_phase() {
        let trigger = should_create_plan(
            "First update the config, then restart the service, and finally verify it works",
        );
        assert_eq!(trigger, PlanTrigger::Suggest("multi-phase task".to_string()));
    }

    #[test]
    fn test_no_plan_simple_request() {
        let trigger = should_create_plan("What's the weather?");
        assert_eq!(trigger, PlanTrigger::None);

        let trigger = should_create_plan("Fix the typo in README");
        assert_eq!(trigger, PlanTrigger::None);

        let trigger = should_create_plan("Show me the logs");
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
