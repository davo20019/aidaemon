use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tracing::warn;

use crate::traits::StateStore;

use super::MAX_CONSECUTIVE_SAME_TOOL;

/// Context accumulated during handle_message for post-task learning.
#[derive(Clone)]
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

    // 2. Save procedure evidence for both successful and failed workflows.
    // This lets us accumulate failure_count for repeated bad paths.
    if ctx.tool_calls.len() >= 2 {
        let generalized = procedures::generalize_procedure(&ctx.tool_calls);
        let base_name = procedures::generate_procedure_name(&ctx.user_text);
        let keyed_name = procedures::generate_procedure_keyed_name(&base_name, &generalized);
        let procedure = procedures::create_procedure_with_outcome(
            keyed_name,
            procedures::extract_trigger_pattern(&ctx.user_text),
            generalized,
            task_success,
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
            "I'm being rate-limited right now. Try again in a few minutes.",
        )
    } else if recent_errors.contains("timed out") || recent_errors.contains("timeout") {
        (
            "Timeout",
            "Responses are taking too long right now. This usually resolves on its own -- try again shortly, or try a simpler request.",
        )
    } else if recent_errors.contains("network") || recent_errors.contains("connection") {
        (
            "Network Error",
            "There seems to be a connectivity issue. Check your network connection and try again.",
        )
    } else if is_tool_policy_block(&recent_errors) {
        (
            "Tool Policy Block",
            "A step was blocked by safety policy. Try adjusting the request or running the blocked command yourself.",
        )
    } else if is_edit_target_drift(&recent_errors) {
        (
            "Edit Target Drift",
            "I had trouble editing files because the content changed while I was working. Try again so I can re-read the files first.",
        )
    } else if looks_like_provider_server_error(&recent_errors) {
        (
            "Server Error",
            "I'm experiencing temporary server issues. Try again in a few minutes.",
        )
    } else if recent_errors.contains("unknown tool") {
        (
            "Unknown Tool",
            "Something went wrong on my end. This usually resolves on retry -- try again, or rephrase your request.",
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
            "I'm having trouble processing this request. Could you try rephrasing it or breaking it into smaller steps?",
        )
    } else {
        (
            "Stuck",
            "Try rephrasing your request or providing more specific guidance.",
        )
    }
}

fn is_tool_policy_block(recent_errors: &str) -> bool {
    recent_errors.contains("not in the safe command list")
        || recent_errors.contains("safe command list")
        || recent_errors.contains("use 'terminal' for this command")
        || recent_errors.contains("requires approval")
        || recent_errors.contains("not allowed")
}

fn is_edit_target_drift(recent_errors: &str) -> bool {
    recent_errors.contains("text not found in")
        || recent_errors.contains("text not found")
        || recent_errors.contains("search string not found")
        || recent_errors.contains("needle not found")
}

fn looks_like_provider_server_error(recent_errors: &str) -> bool {
    const PROVIDER_SERVER_PATTERNS: &[&str] = &[
        "internal server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "status code 500",
        "status code: 500",
        "status 500",
        "http 500",
        "status code 502",
        "status code: 502",
        "status 502",
        "http 502",
        "status code 503",
        "status code: 503",
        "status 503",
        "http 503",
    ];
    if PROVIDER_SERVER_PATTERNS
        .iter()
        .any(|needle| recent_errors.contains(needle))
    {
        return true;
    }

    let mentions_provider = ["provider", "openai", "anthropic", "api", "llm"]
        .iter()
        .any(|needle| recent_errors.contains(needle));
    mentions_provider && recent_errors.contains("server error")
}

/// Graceful response when task timeout is reached.
pub(super) fn graceful_timeout_response(
    learning_ctx: &LearningContext,
    elapsed: Duration,
) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut summary = format!(
        "I've been working on this for {} minutes and reached the time limit. \
            Here's what I accomplished so far:\n\n{}\
            The task may be incomplete. You can ask me to continue where I left off or try breaking it into smaller parts.",
        elapsed.as_secs() / 60,
        activity,
    );
    if summary.len() > 1500 {
        let mut t = 1500;
        while t > 0 && !summary.is_char_boundary(t) {
            t -= 1;
        }
        summary.truncate(t);
        summary.push('…');
    }
    summary
}

/// Graceful response when task token budget is exhausted.
pub(super) fn graceful_budget_response(learning_ctx: &LearningContext, _tokens_used: u64) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut summary = format!(
        "I've reached my processing limit for this task. \
            Here's what I accomplished so far:\n\n{}\
            The task may be incomplete. You can ask me to continue where I left off.",
        activity,
    );
    // Cap to avoid bloating conversation history while preserving key context.
    if summary.len() > 1500 {
        let mut t = 1500;
        while t > 0 && !summary.is_char_boundary(t) {
            t -= 1;
        }
        summary.truncate(t);
        summary.push('…');
    }
    summary
}

/// Graceful response when a goal hits its daily token budget.
pub(super) fn graceful_goal_daily_budget_response(
    learning_ctx: &LearningContext,
    _tokens_used_today: i64,
    _budget_daily: i64,
) -> String {
    let tool_count = learning_ctx.tool_calls.len();
    let error_count = learning_ctx.errors.len();
    let mut msg = String::from(
        "I've reached my processing limit for this task today. ",
    );
    if tool_count > 0 || error_count > 0 {
        msg.push_str(&format!(
            "Here's what I accomplished: {} steps completed",
            tool_count
        ));
        if error_count > 0 {
            msg.push_str(&format!(", {} issues encountered", error_count));
        }
        msg.push_str(".\n\n");
    }
    msg.push_str(
        "You can ask me to continue this later, or I'll pick it up when my limit resets.",
    );
    msg
}

/// Graceful response when agent is stalled (no progress).
pub(super) fn graceful_stall_response(
    learning_ctx: &LearningContext,
    sent_file_successfully: bool,
    deferred_no_tool_error_marker: &str,
) -> String {
    let (_label, suggestion) = classify_stall(learning_ctx, deferred_no_tool_error_marker);
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    if sent_file_successfully {
        let mut msg = String::from(
            "I sent the requested file(s), but ran into issues with the remaining steps.\n\n",
        );
        if !activity.is_empty() {
            msg.push_str(&activity);
        }
        msg.push_str(suggestion);
        msg
    } else {
        let mut msg = String::from("I wasn't able to complete this task.\n\n");
        if !activity.is_empty() {
            msg.push_str(&activity);
            msg.push('\n');
        }
        msg.push_str(suggestion);
        msg
    }
}

/// Graceful response when agent stalled after making meaningful progress.
pub(super) fn graceful_partial_stall_response(
    learning_ctx: &LearningContext,
    sent_file_successfully: bool,
    deferred_no_tool_error_marker: &str,
) -> String {
    let (_label, suggestion) = classify_stall(learning_ctx, deferred_no_tool_error_marker);
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    if sent_file_successfully {
        let mut msg = String::from(
            "I completed the main deliverable but wasn't able to finish everything.\n\n",
        );
        if !activity.is_empty() {
            msg.push_str(&activity);
        }
        msg.push_str(suggestion);
        msg
    } else {
        let mut msg = String::from(
            "I made some progress but wasn't able to fully complete the task.\n\n",
        );
        if !activity.is_empty() {
            msg.push_str(&activity);
            msg.push('\n');
        }
        msg.push_str(suggestion);
        msg
    }
}

/// Graceful response when repetitive tool calls are detected.
pub(super) fn graceful_repetitive_response(
    learning_ctx: &LearningContext,
    _tool_name: &str,
) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut msg = String::from(
        "I seem to be stuck on this task.\n\n",
    );
    if !activity.is_empty() {
        msg.push_str("Here's what I've done so far:\n");
        msg.push_str(&activity);
        msg.push('\n');
    }
    msg.push_str("Could you try a different approach or provide more specific instructions?");
    msg
}

/// Graceful response when hard iteration cap is reached (legacy mode).
pub(super) fn graceful_cap_response(learning_ctx: &LearningContext, _iterations: usize) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut summary = format!(
        "I've reached my processing limit for this task. \
            Here's what I accomplished so far:\n\n{}\
            The task may be incomplete. You can ask me to continue where I left off or try breaking it into smaller parts.",
        activity,
    );
    if summary.len() > 1500 {
        let mut t = 1500;
        while t > 0 && !summary.is_char_boundary(t) {
            t -= 1;
        }
        summary.truncate(t);
        summary.push('…');
    }
    summary
}

/// Determine whether the current task execution is making productive progress,
/// warranting a token budget auto-extension.
///
/// Criteria:
/// 1. At most one stall detected (`stall_count <= 1`)
/// 2. Same-tool repetition: if count ≥ `MAX_CONSECUTIVE_SAME_TOOL`, check diversity
///    (`unique * 2 > count`); if not diverse OR count ≥ `MAX_CONSECUTIVE_SAME_TOOL + 4`
///    → not productive
/// 3. Error rate: unrecovered errors * 2 < max(1, total_successful)
/// 4. Minimum 8 successful tool calls
pub(super) fn is_productive(
    learning_ctx: &LearningContext,
    stall_count: usize,
    consecutive_same_tool_count: usize,
    consecutive_same_tool_unique_args: usize,
    total_successful_tool_calls: usize,
) -> bool {
    // 1. Allow one transient stall (e.g., provider timeout) before disqualifying.
    if stall_count > 1 {
        return false;
    }

    // 2. Same-tool repetition check
    let diverse_limit = MAX_CONSECUTIVE_SAME_TOOL + 4;
    if consecutive_same_tool_count >= diverse_limit {
        return false;
    }
    if consecutive_same_tool_count >= MAX_CONSECUTIVE_SAME_TOOL {
        let is_diverse = consecutive_same_tool_unique_args * 2 > consecutive_same_tool_count;
        if !is_diverse {
            return false;
        }
    }

    // 3. Error rate: unrecovered errors must be < half of successful calls
    let unrecovered = learning_ctx
        .errors
        .iter()
        .filter(|(_, recovered)| !recovered)
        .count();
    let denominator = total_successful_tool_calls.max(1);
    if unrecovered * 2 >= denominator {
        return false;
    }

    // 4. Minimum activity threshold — require meaningful progress, not just
    // a few tool calls.  The old threshold of 3 was too low: an agent that
    // made 3 calls and hit the budget would still qualify for auto-extension.
    if total_successful_tool_calls < 8 {
        return false;
    }

    // 5. Error ratio sanity: at least 75% of activity should be successful
    let total_activity = total_successful_tool_calls + unrecovered;
    if total_activity > 0 && total_successful_tool_calls * 4 < total_activity * 3 {
        return false;
    }

    true
}

/// Categorize tool calls into a human-readable activity summary.
///
/// Parses entries like `"read_file(Hero.jsx)"` and `"terminal(\`pip install fpdf\`)"` into
/// grouped categories so the next interaction can understand what was already done.
pub(in crate::agent) fn categorize_tool_calls(tool_calls: &[String]) -> String {
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
        assert!(result.contains("processing limit"));
        assert!(result.contains("Activity summary:"));
        assert!(result.contains("Files read: App.jsx"));
        assert!(result.contains("Files sent: out.pdf"));
        // Should NOT contain internal details
        assert!(!result.contains("500000 tokens"));
        assert!(!result.contains("tool calls executed"));
        assert!(!result.contains("errors encountered"));
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

    #[test]
    fn test_graceful_partial_stall_response_mentions_progress() {
        let ctx = LearningContext {
            user_text: "fix build".to_string(),
            intent_domains: vec![],
            tool_calls: vec![
                "read_file(Cargo.toml)".to_string(),
                "run_command(cargo build)".to_string(),
                "edit_file(src/lib.rs)".to_string(),
                "run_command(cargo build)".to_string(),
            ],
            errors: vec![("Text not found in src/lib.rs".to_string(), false)],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
        };
        let result = graceful_partial_stall_response(&ctx, false, "deferred");
        assert!(result.contains("some progress"));
        assert!(result.contains("Activity summary:"));
        // Should NOT contain internal details
        assert!(!result.contains("tool calls executed"));
        assert!(!result.contains("errors encountered"));
        assert!(!result.contains("Stopping reason"));
    }

    fn make_learning_ctx() -> LearningContext {
        LearningContext {
            user_text: "deploy the app".to_string(),
            intent_domains: vec![],
            tool_calls: vec![
                "read_file(main.rs)".to_string(),
                "terminal(`cargo build`)".to_string(),
                "write_file(deploy.sh)".to_string(),
                "terminal(`./deploy.sh`)".to_string(),
            ],
            errors: vec![],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
        }
    }

    #[test]
    fn test_is_productive_happy_path() {
        let ctx = make_learning_ctx();
        assert!(is_productive(&ctx, 0, 0, 0, 10));
    }

    #[test]
    fn test_is_productive_stalling() {
        let ctx = make_learning_ctx();
        assert!(is_productive(&ctx, 1, 0, 0, 10));
        assert!(!is_productive(&ctx, 2, 0, 0, 10));
    }

    #[test]
    fn test_is_productive_too_many_errors() {
        let mut ctx = make_learning_ctx();
        ctx.errors = vec![
            ("error 1".to_string(), false),
            ("error 2".to_string(), false),
            ("error 3".to_string(), false),
        ];
        // 3 unrecovered errors, 5 successful → 3*2=6 >= 5 → not productive
        assert!(!is_productive(&ctx, 0, 0, 0, 5));
    }

    #[test]
    fn test_is_productive_low_activity() {
        let ctx = make_learning_ctx();
        // Only 2 successful tool calls → below minimum
        assert!(!is_productive(&ctx, 0, 0, 0, 2));
    }

    #[test]
    fn test_is_productive_diverse_args_ok() {
        let ctx = make_learning_ctx();
        // At MAX_CONSECUTIVE_SAME_TOOL consecutive same tool calls, but diverse args → productive
        assert!(is_productive(
            &ctx,
            0,
            MAX_CONSECUTIVE_SAME_TOOL,
            MAX_CONSECUTIVE_SAME_TOOL / 2 + 2,
            20
        ));
    }

    #[test]
    fn test_is_productive_same_args_not_ok() {
        let ctx = make_learning_ctx();
        // At MAX_CONSECUTIVE_SAME_TOOL consecutive same tool calls, too few unique args → not diverse
        assert!(!is_productive(&ctx, 0, MAX_CONSECUTIVE_SAME_TOOL, 3, 20));
    }

    #[test]
    fn test_is_productive_diverse_but_over_20() {
        let ctx = make_learning_ctx();
        // Over the diverse limit (MAX + 4) → not productive regardless of diversity
        assert!(!is_productive(
            &ctx,
            0,
            MAX_CONSECUTIVE_SAME_TOOL + 4,
            MAX_CONSECUTIVE_SAME_TOOL,
            25
        ));
    }

    fn ctx_with_single_error(error: &str) -> LearningContext {
        LearningContext {
            user_text: "do something".to_string(),
            intent_domains: vec![],
            tool_calls: vec![],
            errors: vec![(error.to_string(), false)],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
        }
    }

    #[test]
    fn test_classify_stall_prefers_tool_policy_block() {
        let ctx = ctx_with_single_error(
            "Command 'npm install tailwindcss' is not in the safe command list. Use 'terminal' for this command.",
        );
        let (label, suggestion) = classify_stall(&ctx, "deferred-no-tool");
        assert_eq!(label, "Tool Policy Block");
        assert!(suggestion.contains("safety policy"));
    }

    #[test]
    fn test_classify_stall_detects_edit_target_drift() {
        let ctx = ctx_with_single_error(
            "Text not found in ~/projects/oaxaca-mezcal-tours/src/components/ContactForm.jsx. The old_text did not match.",
        );
        let (label, suggestion) = classify_stall(&ctx, "deferred-no-tool");
        assert_eq!(label, "Edit Target Drift");
        assert!(suggestion.contains("re-read"));
    }

    #[test]
    fn test_classify_stall_ignores_generic_5000_values() {
        let ctx = ctx_with_single_error("Exceeded 5000 characters while building summary.");
        let (label, _) = classify_stall(&ctx, "deferred-no-tool");
        assert_eq!(label, "Stuck");
    }

    #[test]
    fn test_classify_stall_detects_provider_server_status_codes() {
        let ctx = ctx_with_single_error("OpenAI API returned status code 503 Service Unavailable.");
        let (label, _) = classify_stall(&ctx, "deferred-no-tool");
        assert_eq!(label, "Server Error");
    }
}
