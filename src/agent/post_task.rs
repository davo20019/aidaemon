use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tracing::warn;

use crate::traits::StateStore;

/// Context accumulated during handle_message for post-task learning.
pub(super) struct LearningContext {
    pub(super) user_text: String,
    pub(super) intent_domains: Vec<String>,
    pub(super) tool_calls: Vec<String>,     // "tool_name(summary)"
    pub(super) errors: Vec<(String, bool)>, // (error_text, was_recovered)
    pub(super) first_error: Option<String>,
    pub(super) recovery_actions: Vec<String>,
    #[allow(dead_code)] // Reserved for duration-based learning
    pub(super) start_time: chrono::DateTime<Utc>,
    pub(super) completed_naturally: bool,
    pub(super) explicit_positive_signals: u32,
    pub(super) explicit_negative_signals: u32,
}

/// Process learning from a completed task - runs in background.
pub(super) async fn process_learning(
    state: &Arc<dyn StateStore>,
    ctx: LearningContext,
) -> anyhow::Result<()> {
    use crate::memory::{expertise, procedures};

    // Determine if task was successful
    let unrecovered_errors = ctx
        .errors
        .iter()
        .filter(|(_, recovered)| !recovered)
        .count();
    let task_success = if ctx.explicit_negative_signals > 0 {
        false
    } else if ctx.explicit_positive_signals > 0 {
        true
    } else {
        ctx.completed_naturally && unrecovered_errors == 0
    };

    // 1. Update expertise for detected domains
    let domains = expertise::detect_domains(&ctx.intent_domains);
    for domain in &domains {
        let error = if !task_success {
            ctx.errors.first().map(|(e, _)| e.as_str())
        } else {
            None
        };
        if let Err(e) = state.increment_expertise(domain, task_success, error).await {
            warn!(domain = %domain, error = %e, "Failed to update expertise");
        }
    }

    // 2. Save procedure if successful with 2+ actions
    if task_success && ctx.tool_calls.len() >= 2 {
        let generalized = procedures::generalize_procedure(&ctx.tool_calls);
        let procedure = procedures::create_procedure(
            procedures::generate_procedure_name(&ctx.user_text),
            procedures::extract_trigger_pattern(&ctx.user_text),
            generalized,
        );
        if let Err(e) = state.upsert_procedure(&procedure).await {
            warn!(procedure = %procedure.name, error = %e, "Failed to save procedure");
        }
    }

    // 3. Learn error-solution if error was recovered
    if let Some(error) = ctx.first_error {
        if !ctx.recovery_actions.is_empty() {
            let solution = procedures::create_error_solution(
                procedures::extract_error_pattern(&error),
                domains.into_iter().next(),
                procedures::summarize_solution(&ctx.recovery_actions),
                Some(ctx.recovery_actions),
            );
            if let Err(e) = state.insert_error_solution(&solution).await {
                warn!(error_pattern = %solution.error_pattern, error = %e, "Failed to save error solution");
            }
        }
    }

    Ok(())
}

/// Classify the stall cause from recent errors for actionable guidance.
pub(super) fn classify_stall(
    learning_ctx: &LearningContext,
    deferred_no_tool_error_marker: &str,
) -> (&'static str, &'static str) {
    let recent_errors: String = learning_ctx
        .errors
        .iter()
        .rev()
        .take(5)
        .map(|(e, _)| e.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    if recent_errors.contains("rate limit") || recent_errors.contains("429") {
        (
            "Rate Limited",
            "The AI provider is throttling requests. Try again in a few minutes, or consider switching to a different model tier.",
        )
    } else if recent_errors.contains("timed out") || recent_errors.contains("timeout") {
        (
            "Timeout",
            "The AI provider is responding slowly. This usually resolves on its own — try again shortly, or try a simpler request.",
        )
    } else if recent_errors.contains("network") || recent_errors.contains("connection") {
        (
            "Network Error",
            "There's a connectivity issue reaching the AI provider. Check your network connection and try again.",
        )
    } else if recent_errors.contains("server error")
        || recent_errors.contains("500")
        || recent_errors.contains("502")
        || recent_errors.contains("503")
    {
        (
            "Server Error",
            "The AI provider is experiencing issues. This is usually temporary — try again in a few minutes.",
        )
    } else if recent_errors.contains("unknown tool") {
        (
            "Unknown Tool",
            "The model tried to call a tool that doesn't exist. This usually resolves on retry — if it recurs, try a different model or rephrase your request.",
        )
    } else if recent_errors.contains("unauthorized")
        || recent_errors.contains("api key")
        || recent_errors.contains("authentication failed")
        || recent_errors.contains("invalid auth")
    {
        (
            "Authentication",
            "There may be an issue with API credentials. Check your provider configuration.",
        )
    } else if recent_errors.contains(deferred_no_tool_error_marker) {
        (
            "Deferred No-Tool Loop",
            "The model repeatedly promised actions but never called tools. Retry the request; if it recurs, switch model/profile or ask for a direct text answer.",
        )
    } else {
        (
            "Stuck",
            "Try rephrasing your request or providing more specific guidance.",
        )
    }
}

/// Graceful response when task timeout is reached.
pub(super) fn graceful_timeout_response(
    learning_ctx: &LearningContext,
    elapsed: Duration,
) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut summary = format!(
        "I've been working on this task for {} minutes and reached the time limit. \
            Here's what I accomplished:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n{}\
            The task may be incomplete. You can continue where I left off or try breaking it into smaller parts.",
        elapsed.as_secs() / 60,
        learning_ctx.tool_calls.len(),
        learning_ctx.errors.len(),
        activity,
    );
    if summary.len() > 1500 {
        summary.truncate(1500);
        summary.push('…');
    }
    summary
}

/// Graceful response when task token budget is exhausted.
pub(super) fn graceful_budget_response(learning_ctx: &LearningContext, tokens_used: u64) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut summary = format!(
        "I've used {} tokens on this task and reached the budget limit. \
            Here's what I accomplished:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n{}\
            The task may be incomplete. You can continue where I left off.",
        tokens_used,
        learning_ctx.tool_calls.len(),
        learning_ctx.errors.len(),
        activity,
    );
    // Cap to avoid bloating conversation history while preserving key context.
    if summary.len() > 1500 {
        summary.truncate(1500);
        summary.push('…');
    }
    summary
}

/// Graceful response when a goal hits its daily token budget.
pub(super) fn graceful_goal_daily_budget_response(
    learning_ctx: &LearningContext,
    tokens_used_today: i64,
    budget_daily: i64,
) -> String {
    format!(
        "This goal has reached its daily token budget (used {} / limit {}). \
            I'm stopping now to prevent runaway spend.\n\n\
            Here's what I accomplished:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n\
            If you want me to continue, you can retry later after the daily budget resets, \
            or increase the goal's budget.",
        tokens_used_today,
        budget_daily,
        learning_ctx.tool_calls.len(),
        learning_ctx.errors.len()
    )
}

/// Graceful response when agent is stalled (no progress).
pub(super) fn graceful_stall_response(
    learning_ctx: &LearningContext,
    sent_file_successfully: bool,
    deferred_no_tool_error_marker: &str,
) -> String {
    let (label, suggestion) = classify_stall(learning_ctx, deferred_no_tool_error_marker);
    let recent_errors = learning_ctx
        .errors
        .iter()
        .rev()
        .take(3)
        .map(|(e, _)| format!("- {}", e.chars().take(100).collect::<String>()))
        .collect::<Vec<_>>()
        .join("\n");
    let summary = if sent_file_successfully {
        format!(
            "I already sent at least one requested file, then got stuck in follow-up steps.\n\n\
                Stopping reason: **{}**\n\
                - {} tool calls executed\n\
                - {} errors encountered\n\n\
                {}\n\n\
                Recent errors:\n{}",
            label,
            learning_ctx.tool_calls.len(),
            learning_ctx.errors.len(),
            suggestion,
            recent_errors
        )
    } else {
        format!(
            "I'm unable to make progress — **{}**.\n\n\
                Here's what I tried:\n\
                - {} tool calls executed\n\
                - {} errors encountered\n\n\
                {}\n\n\
                Recent errors:\n{}",
            label,
            learning_ctx.tool_calls.len(),
            learning_ctx.errors.len(),
            suggestion,
            recent_errors
        )
    };
    summary
}

/// Graceful response when repetitive tool calls are detected.
pub(super) fn graceful_repetitive_response(
    learning_ctx: &LearningContext,
    tool_name: &str,
) -> String {
    // Plan pausing removed (plans deprecated in favor of goals/tasks).
    let summary = format!(
        "I noticed I'm calling `{}` repeatedly with similar parameters, which suggests I'm stuck in a loop. \
            Here's what I've done so far:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n\
            Please try a different approach or provide more specific instructions.",
        tool_name,
        learning_ctx.tool_calls.len(),
        learning_ctx.errors.len()
    );
    summary
}

/// Graceful response when hard iteration cap is reached (legacy mode).
pub(super) fn graceful_cap_response(learning_ctx: &LearningContext, iterations: usize) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut summary = format!(
        "I've reached the maximum iteration limit ({} iterations). \
            Here's what I accomplished:\n\n\
            - {} tool calls executed\n\
            - {} errors encountered\n\n{}\
            The task may be incomplete. Consider increasing the iteration limit in config or using unlimited mode.",
        iterations,
        learning_ctx.tool_calls.len(),
        learning_ctx.errors.len(),
        activity,
    );
    if summary.len() > 1500 {
        summary.truncate(1500);
        summary.push('…');
    }
    summary
}

/// Categorize tool calls into a human-readable activity summary.
///
/// Parses entries like `"read_file(Hero.jsx)"` and `"terminal(\`pip install fpdf\`)"` into
/// grouped categories so the next interaction can understand what was already done.
fn categorize_tool_calls(tool_calls: &[String]) -> String {
    let mut files_read: Vec<&str> = Vec::new();
    let mut files_written: Vec<&str> = Vec::new();
    let mut commands_run: Vec<&str> = Vec::new();
    let mut files_sent: Vec<&str> = Vec::new();
    let mut searches: Vec<&str> = Vec::new();
    let mut other: Vec<&str> = Vec::new();

    for entry in tool_calls {
        // Parse "tool_name(summary)" format
        let (name, args) = match entry.find('(') {
            Some(idx) => {
                let name = &entry[..idx];
                let args = entry[idx + 1..].trim_end_matches(')');
                (name, args)
            }
            None => (entry.as_str(), ""),
        };
        match name {
            "read_file" => files_read.push(args),
            "write_file" | "edit_file" => files_written.push(args),
            "terminal" | "run_command" => commands_run.push(args),
            "send_file" | "send_media" => files_sent.push(args),
            "web_search" | "search_files" => searches.push(args),
            "project_inspect" => files_read.push(args),
            _ => {
                if !args.is_empty() {
                    other.push(args);
                }
            }
        }
    }

    let mut sections = Vec::new();

    if !files_read.is_empty() {
        let items: Vec<&str> = files_read.iter().copied().take(15).collect();
        sections.push(format!("Files read: {}", items.join(", ")));
    }
    if !files_written.is_empty() {
        let items: Vec<&str> = files_written.iter().copied().take(10).collect();
        sections.push(format!("Files written: {}", items.join(", ")));
    }
    if !commands_run.is_empty() {
        let items: Vec<&str> = commands_run.iter().copied().take(10).collect();
        sections.push(format!("Commands run: {}", items.join(", ")));
    }
    if !files_sent.is_empty() {
        let items: Vec<&str> = files_sent.iter().copied().take(5).collect();
        sections.push(format!("Files sent: {}", items.join(", ")));
    }
    if !searches.is_empty() {
        let items: Vec<&str> = searches.iter().copied().take(5).collect();
        sections.push(format!("Searches: {}", items.join(", ")));
    }

    if sections.is_empty() {
        return String::new();
    }

    let mut result = String::from("Activity summary:\n");
    for section in &sections {
        result.push_str("- ");
        result.push_str(section);
        result.push('\n');
    }
    result.push('\n');
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorize_tool_calls_groups_correctly() {
        let calls = vec![
            "read_file(Hero.jsx)".to_string(),
            "read_file(App.jsx)".to_string(),
            "terminal(`pip install fpdf`)".to_string(),
            "write_file(generate_pdf.py)".to_string(),
            "terminal(`python3 generate_pdf.py`)".to_string(),
            "send_file(Guide.pdf)".to_string(),
            "web_search(top things Chantilly VA)".to_string(),
            "project_inspect(chantilly-va-site)".to_string(),
        ];
        let result = categorize_tool_calls(&calls);
        assert!(result.contains("Files read:"));
        assert!(result.contains("Hero.jsx"));
        assert!(result.contains("chantilly-va-site"));
        assert!(result.contains("Files written:"));
        assert!(result.contains("generate_pdf.py"));
        assert!(result.contains("Commands run:"));
        assert!(result.contains("Files sent:"));
        assert!(result.contains("Guide.pdf"));
        assert!(result.contains("Searches:"));
    }

    #[test]
    fn test_categorize_tool_calls_empty() {
        let result = categorize_tool_calls(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_categorize_tool_calls_limits_items() {
        let calls: Vec<String> = (0..20)
            .map(|i| format!("read_file(file_{}.rs)", i))
            .collect();
        let result = categorize_tool_calls(&calls);
        // Should include max 15 files
        assert!(result.contains("file_14.rs"));
        assert!(!result.contains("file_15.rs"));
    }

    #[test]
    fn test_graceful_budget_response_includes_activity() {
        let ctx = LearningContext {
            user_text: "Create a PDF".to_string(),
            intent_domains: vec![],
            tool_calls: vec![
                "read_file(App.jsx)".to_string(),
                "terminal(`pip install fpdf`)".to_string(),
                "write_file(gen.py)".to_string(),
                "send_file(out.pdf)".to_string(),
            ],
            errors: vec![],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
        };
        let result = graceful_budget_response(&ctx, 500_000);
        assert!(result.contains("500000 tokens"));
        assert!(result.contains("4 tool calls"));
        assert!(result.contains("Activity summary:"));
        assert!(result.contains("Files read: App.jsx"));
        assert!(result.contains("Files sent: out.pdf"));
    }

    #[test]
    fn test_graceful_budget_response_caps_length() {
        let calls: Vec<String> = (0..100)
            .map(|i| {
                format!(
                    "terminal(`very long command number {} that does things`)",
                    i
                )
            })
            .collect();
        let ctx = LearningContext {
            user_text: "big task".to_string(),
            intent_domains: vec![],
            tool_calls: calls,
            errors: vec![],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
        };
        let result = graceful_budget_response(&ctx, 500_000);
        assert!(result.len() <= 1502); // 1500 + "…"
    }
}
