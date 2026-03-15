use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use once_cell::sync::Lazy;
use regex::Regex;
use tracing::warn;

use crate::traits::StateStore;

use super::{extract_key_error_line, semantic_failure_limit, MAX_CONSECUTIVE_SAME_TOOL};

/// Context accumulated during handle_message for post-task learning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ReplayNoteCategory {
    PlanRevision,
    EvidenceGate,
    ValidationFailure,
    RetryReason,
}

impl ReplayNoteCategory {
    fn as_str(&self) -> &'static str {
        match self {
            Self::PlanRevision => "plan_revision",
            Self::EvidenceGate => "evidence_gate",
            Self::ValidationFailure => "validation_failure",
            Self::RetryReason => "retry_reason",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ReplayNote {
    pub(super) category: ReplayNoteCategory,
    pub(super) code: String,
    pub(super) summary: String,
    pub(super) blocking: bool,
}

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
    pub(super) replay_notes: Vec<ReplayNote>,
}

impl LearningContext {
    pub(super) fn record_replay_note(
        &mut self,
        category: ReplayNoteCategory,
        code: impl Into<String>,
        summary: impl Into<String>,
        blocking: bool,
    ) {
        const MAX_REPLAY_NOTES: usize = 24;

        let code = code.into().trim().to_string();
        let summary = summary.into().trim().to_string();
        if code.is_empty() || summary.is_empty() {
            return;
        }

        let note = ReplayNote {
            category,
            code,
            summary,
            blocking,
        };
        if self.replay_notes.contains(&note) {
            return;
        }
        if self.replay_notes.len() >= MAX_REPLAY_NOTES {
            self.replay_notes.remove(0);
        }
        self.replay_notes.push(note);
    }
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
    if let Some(error) = ctx.first_error.clone() {
        if !ctx.recovery_actions.is_empty() {
            let solution = procedures::create_error_solution(
                procedures::extract_error_pattern(&error),
                domains.into_iter().next(),
                procedures::summarize_solution(&ctx.recovery_actions),
                Some(ctx.recovery_actions.clone()),
            );
            if let Err(e) = state.insert_error_solution(&solution).await {
                warn!(error_pattern = %solution.error_pattern, error = %e, "Failed to save error solution");
            }
        }
    }

    if !ctx.replay_notes.is_empty() {
        record_reasoning_failure_patterns(state, &ctx, task_success).await;
    }

    Ok(())
}

async fn record_reasoning_failure_patterns(
    state: &Arc<dyn StateStore>,
    ctx: &LearningContext,
    task_success: bool,
) {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for note in ctx.replay_notes.iter().filter(|note| note.blocking) {
        let trigger_context = format!("{}:{}", note.category.as_str(), note.code);
        let action = reasoning_action_for_note(note);
        if !seen.insert((trigger_context.clone(), action.to_string())) {
            continue;
        }

        let confidence = reasoning_confidence_for_note(note, task_success);
        if let Err(e) = state
            .record_behavior_pattern(
                "reasoning_failure",
                &note.summary,
                Some(&trigger_context),
                Some(action),
                confidence,
                1,
            )
            .await
        {
            warn!(
                trigger_context = %trigger_context,
                action = %action,
                error = %e,
                "Failed to save reasoning failure pattern"
            );
        }
    }
}

fn reasoning_action_for_note(note: &ReplayNote) -> &'static str {
    match note.code.as_str() {
        "missing_pre_execution_evidence" => "gather_direct_evidence_before_mutation",
        "plan_rejected" => "replan_first_risky_step_before_execution",
        "critique_rejected" => "address_critique_and_replan",
        "target_scope_violation" => "confirm_target_scope_before_mutation",
        "tool_contract_violation" => "fix_tool_arguments_before_retry",
        "contradictory_file_evidence" => "recheck_conflicting_state_before_completion",
        "verification_pending" => "run_verification_before_claiming_success",
        "verification_unavailable_in_phase" => "surface_partial_result_until_verification_can_run",
        "validation_budget_exhausted" => "reduce_scope_when_validation_budget_exhausts",
        "execution_budget_exhausted" => "reduce_scope_or_abandon_when_execution_budget_exhausts",
        "retry_step" => "retry_only_when_failure_is_local_and_correctable",
        "replan_required" => "replan_after_logic_or_environment_failure",
        _ => "review_reasoning_trace_before_retry",
    }
}

fn reasoning_confidence_for_note(note: &ReplayNote, task_success: bool) -> f32 {
    let base = match note.code.as_str() {
        "missing_pre_execution_evidence" | "verification_pending" => 0.78,
        "target_scope_violation" => 0.84,
        "plan_rejected" | "critique_rejected" => 0.72,
        "contradictory_file_evidence" => 0.76,
        "validation_budget_exhausted" | "execution_budget_exhausted" => 0.68,
        "retry_step" | "replan_required" => 0.58,
        _ => 0.52,
    };
    if task_success {
        (base - 0.08_f32).clamp(0.25_f32, 0.96_f32)
    } else {
        (base + 0.06_f32).clamp(0.25_f32, 0.96_f32)
    }
}

#[allow(dead_code)] // Used by targeted replay/learning tests and future replay surfacing.
pub(in crate::agent) fn summarize_replay_notes(notes: &[ReplayNote]) -> String {
    if notes.is_empty() {
        return String::new();
    }

    let mut grouped: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for note in notes {
        grouped
            .entry(note.category.as_str())
            .or_default()
            .push(note.summary.clone());
    }

    let mut sections = Vec::new();
    for (category, items) in grouped {
        let unique: Vec<String> = items.into_iter().fold(Vec::new(), |mut acc, item| {
            if !acc.contains(&item) {
                acc.push(item);
            }
            acc
        });
        if unique.is_empty() {
            continue;
        }
        sections.push(format!("{}: {}", category, unique.join(" | ")));
    }
    sections.join("\n")
}

/// Classify the stall cause from recent errors for actionable guidance.
pub(super) fn classify_stall(
    learning_ctx: &LearningContext,
    deferred_no_tool_error_marker: &str,
    tool_failure_count: &HashMap<String, usize>,
) -> (&'static str, &'static str) {
    let any_locked = tool_failure_count
        .iter()
        .any(|(name, count)| *count >= semantic_failure_limit(name));
    if any_locked {
        return (
            "Tool Locked Out",
            "I ran into repeated errors with a command and got locked out. Try rephrasing or specifying the exact command to use.",
        );
    }

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

fn user_friendly_tool_description(tool_name: &str) -> &'static str {
    match tool_name {
        "terminal" => "command execution",
        "web_search" | "web_fetch" => "web access",
        "http_request" => "API access",
        "cli_agent" => "external agent",
        "edit_file" | "write_file" | "read_file" => "file operations",
        "browser" => "browser interaction",
        _ => "other capability",
    }
}

pub(super) fn format_tool_failure_summary(tool_failure_count: &HashMap<String, usize>) -> String {
    let blocked_labels: BTreeSet<&'static str> = tool_failure_count
        .iter()
        .filter(|(name, count)| **count >= semantic_failure_limit(name))
        .map(|(name, _)| user_friendly_tool_description(name))
        .collect();

    if blocked_labels.is_empty() {
        return String::new();
    }

    let mut summary = String::from("Blocked capabilities due to repeated errors:\n");
    for label in blocked_labels {
        summary.push_str("- ");
        summary.push_str(label);
        summary.push('\n');
    }
    summary.push('\n');
    summary
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
pub(super) fn graceful_budget_response(
    learning_ctx: &LearningContext,
    _tokens_used: u64,
) -> String {
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

/// Graceful response when a scheduled run hits its per-run budget.
pub(super) fn graceful_scheduled_run_budget_response(
    learning_ctx: &LearningContext,
    tokens_used: i64,
    budget_per_check: i64,
) -> String {
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let mut summary = format!(
        "This scheduled run hit its per-run processing budget (used {} / {} tokens). \
            Here's what I accomplished so far:\n\n{}\
            The run stopped because it no longer looked productive enough to keep extending automatically. \
            If this task legitimately needs more room, raise `budget_per_check`.",
        tokens_used, budget_per_check, activity,
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

/// Graceful response when a goal hits its daily token budget.
pub(super) fn graceful_goal_daily_budget_response(
    learning_ctx: &LearningContext,
    tokens_used_today: i64,
    budget_daily: i64,
    is_scheduled_goal: bool,
) -> String {
    let tool_count = learning_ctx.tool_calls.len();
    let error_count = learning_ctx.errors.len();
    let next_reset = Utc::now()
        .date_naive()
        .succ_opt()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|dt| dt.format("%B %-d, %Y 00:00 UTC").to_string())
        .unwrap_or_else(|| "the next UTC day boundary".to_string());
    let scope = if is_scheduled_goal {
        "This scheduled goal hit its daily processing budget"
    } else {
        "This goal hit its daily processing budget"
    };
    let mut msg = format!(
        "{} (used {} / {} tokens). The budget resets at {}. ",
        scope, tokens_used_today, budget_daily, next_reset
    );
    if tool_count > 0 || error_count > 0 {
        msg.push_str(&format!(
            "Here's what I accomplished so far: {} steps completed",
            tool_count
        ));
        if error_count > 0 {
            msg.push_str(&format!(", {} issues encountered", error_count));
        }
        msg.push_str(".\n\n");
    }
    if is_scheduled_goal {
        msg.push_str(
            "To prevent this, raise the scheduled goal's `budget_daily` or reduce how often it runs.",
        );
    } else {
        msg.push_str("You can ask me to continue this later, or increase the goal's daily budget.");
    }
    msg
}

/// Graceful response when agent is stalled (no progress).
pub(super) fn graceful_stall_response(
    learning_ctx: &LearningContext,
    sent_file_successfully: bool,
    deferred_no_tool_error_marker: &str,
    tool_failure_count: &HashMap<String, usize>,
) -> String {
    let (_label, suggestion) = classify_stall(
        learning_ctx,
        deferred_no_tool_error_marker,
        tool_failure_count,
    );
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let error_explanation = format_error_explanation(&learning_ctx.errors);
    let failure_summary = format_tool_failure_summary(tool_failure_count);
    if sent_file_successfully {
        let mut msg = String::from(
            "I sent the requested file(s), but ran into issues with the remaining steps.\n\n",
        );
        if !activity.is_empty() {
            msg.push_str(&activity);
        }
        if !error_explanation.is_empty() {
            msg.push_str(&error_explanation);
        }
        if !failure_summary.is_empty() {
            msg.push_str(&failure_summary);
        }
        msg.push_str(suggestion);
        msg
    } else {
        let mut msg = String::from("I wasn't able to complete this task.\n\n");
        if !activity.is_empty() {
            msg.push_str(&activity);
            msg.push('\n');
        }
        if !error_explanation.is_empty() {
            msg.push_str(&error_explanation);
        }
        if !failure_summary.is_empty() {
            msg.push_str(&failure_summary);
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
    tool_failure_count: &HashMap<String, usize>,
) -> String {
    let (_label, suggestion) = classify_stall(
        learning_ctx,
        deferred_no_tool_error_marker,
        tool_failure_count,
    );
    let activity = categorize_tool_calls(&learning_ctx.tool_calls);
    let error_explanation = format_error_explanation(&learning_ctx.errors);
    let failure_summary = format_tool_failure_summary(tool_failure_count);
    if sent_file_successfully {
        let mut msg = String::from(
            "I completed the main deliverable but wasn't able to finish everything.\n\n",
        );
        if !activity.is_empty() {
            msg.push_str(&activity);
        }
        if !error_explanation.is_empty() {
            msg.push_str(&error_explanation);
        }
        if !failure_summary.is_empty() {
            msg.push_str(&failure_summary);
        }
        msg.push_str(suggestion);
        msg
    } else {
        let mut msg =
            String::from("I made some progress but wasn't able to fully complete the task.\n\n");
        if !activity.is_empty() {
            msg.push_str(&activity);
            msg.push('\n');
        }
        if !error_explanation.is_empty() {
            msg.push_str(&error_explanation);
        }
        if !failure_summary.is_empty() {
            msg.push_str(&failure_summary);
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
    let error_explanation = format_error_explanation(&learning_ctx.errors);
    let mut msg = String::from("I seem to be stuck on this task.\n\n");
    if !activity.is_empty() {
        msg.push_str("Here's what I've done so far:\n");
        msg.push_str(&activity);
        msg.push('\n');
    }
    if !error_explanation.is_empty() {
        msg.push_str(&error_explanation);
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
/// 4. Minimum 3 successful tool calls
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

    // 4. Minimum activity threshold — require more than one or two lucky calls,
    // but do not force long runs to hit a high fixed bar before they can
    // continue. Three successful tool calls is enough to prove the run is
    // underway without blocking short productive tasks.
    if total_successful_tool_calls < 3 {
        return false;
    }

    // 5. Error ratio sanity: at least 75% of activity should be successful
    let total_activity = total_successful_tool_calls + unrecovered;
    if total_activity > 0 && total_successful_tool_calls * 4 < total_activity * 3 {
        return false;
    }

    true
}

/// Format a user-facing explanation of what went wrong, from the error list.
fn format_error_explanation(errors: &[(String, bool)]) -> String {
    if errors.is_empty() {
        return String::new();
    }

    // Build deduped list from the most recent errors (the final blocker is most
    // relevant). We take the last 5 to avoid processing the entire history, but
    // prioritize recency over first-seen order.
    // If same line appears both recovered and unrecovered, prefer unrecovered (worse case).
    let recent_start = errors.len().saturating_sub(5);
    let mut seen: Vec<(String, bool)> = Vec::new();
    for (error_text, recovered) in errors[recent_start..].iter() {
        let key_line = extract_key_error_line(error_text);
        if key_line.is_empty() {
            continue;
        }
        let redacted = redact_error_line_for_summary(&key_line);
        if let Some(existing) = seen.iter_mut().find(|(line, _)| *line == redacted) {
            // If new occurrence is unrecovered, upgrade to unrecovered
            if !recovered {
                existing.1 = false;
            }
        } else {
            seen.push((redacted, *recovered));
        }
    }

    if seen.is_empty() {
        return String::new();
    }

    let mut result = String::from("Issues encountered:\n");
    for (line, recovered) in seen.iter().take(3) {
        result.push_str("- ");
        result.push_str(line);
        if *recovered {
            result.push_str(" (resolved)");
        }
        result.push('\n');
    }
    result.push('\n');
    result
}

static ABS_UNIX_PATH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"/(?:Users|home|etc)/[^\s]+").expect("unix path regex must compile"));
static ABS_WINDOWS_PATH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Z]:[\\/][^\s]+").expect("windows path regex must compile"));

fn split_trailing_path_punctuation(raw: &str) -> (&str, &str) {
    let mut idx = raw.len();
    for (i, ch) in raw.char_indices().rev() {
        if matches!(ch, ')' | ']' | '}' | ',' | ';' | ':' | '.' | '"' | '\'') {
            idx = i;
            continue;
        }
        break;
    }
    (&raw[..idx], &raw[idx..])
}

fn abbreviate_absolute_path(path: &str) -> String {
    let (core, suffix) = split_trailing_path_punctuation(path);
    let tail = core
        .rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or(core);
    format!("[path:.../{}]{}", tail, suffix)
}

fn redact_error_line_for_summary(key_line: &str) -> String {
    let unix_summarized = ABS_UNIX_PATH_RE
        .replace_all(key_line, |caps: &regex::Captures<'_>| {
            abbreviate_absolute_path(caps.get(0).map(|m| m.as_str()).unwrap_or_default())
        })
        .to_string();
    let windows_summarized = ABS_WINDOWS_PATH_RE
        .replace_all(&unix_summarized, |caps: &regex::Captures<'_>| {
            abbreviate_absolute_path(caps.get(0).map(|m| m.as_str()).unwrap_or_default())
        })
        .to_string();
    crate::tools::sanitize::redact_secrets(&windows_summarized)
}

/// Categorize tool calls into a human-readable activity summary.
/// Convert a raw `"tool_name(args)"` entry into a user-friendly display string.
///
/// This avoids the `tool_name(...)` format that `strip_tool_name_references`
/// would replace with "that".  Output uses "Display Name — args" instead.
pub(in crate::agent) fn display_tool_call(call: &str) -> String {
    let (name, args) = match call.find('(') {
        Some(idx) => {
            let n = &call[..idx];
            let a = call[idx + 1..].trim_end_matches(')');
            (n, a)
        }
        None => (call, ""),
    };
    let display_name = match name {
        "manage_memories" => "Searched memories",
        "remember_fact" => "Saved to memory",
        "search_files" => "Searched files",
        "read_file" => "Read file",
        "write_file" => "Wrote file",
        "edit_file" => "Edited file",
        "terminal" | "run_command" => "Ran command",
        "web_search" => "Web search",
        "web_fetch" => "Fetched URL",
        "http_request" => "HTTP request",
        "goal_trace" => "Checked goal history",
        "tool_trace" => "Checked tool history",
        "manage_goals" | "scheduled_goal_runs" => "Checked goals",
        "send_file" => "Sent file",
        "list_files" | "project_inspect" => "Listed files",
        "spawn_agent" => "Spawned agent",
        "cli_agent" => "Delegated to agent",
        _ => name,
    };
    if args.is_empty() {
        display_name.to_string()
    } else {
        format!("{} — {}", display_name, args)
    }
}

///
/// Parses entries like `"read_file(Hero.jsx)"` and `"terminal(\`pip install fpdf\`)"` into
/// grouped categories so the next interaction can understand what was already done.
pub(in crate::agent) fn categorize_tool_calls(tool_calls: &[String]) -> String {
    let mut files_read: Vec<&str> = Vec::new();
    let mut files_written: Vec<&str> = Vec::new();
    let mut commands_run: Vec<&str> = Vec::new();
    let mut files_sent: Vec<&str> = Vec::new();
    let mut searches: Vec<&str> = Vec::new();
    let mut external_reads: Vec<&str> = Vec::new();
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
            "web_fetch" | "http_request" => external_reads.push(args),
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
    if !external_reads.is_empty() {
        let items: Vec<&str> = external_reads.iter().copied().take(8).collect();
        sections.push(format!("External sources checked: {}", items.join(", ")));
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
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::StateStore;
    use std::sync::Arc;

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
    fn test_display_tool_call_survives_sanitization() {
        use crate::tools::sanitize::sanitize_user_facing_reply;

        let tool_calls = vec![
            "manage_memories(search)",
            "manage_memories(list_scheduled)",
            "search_files(twitter|schedule|cancelled|removed)",
            "goal_trace()",
            "terminal(`cd /tmp && find ~ -type d -name \"aidaemon\"`)",
            "read_file(src/main.rs)",
            "write_file(output.txt)",
            "web_search(rust async patterns)",
        ];

        let mut summary = String::from("Here's what I did before the background task started:\n");
        for (i, call) in tool_calls.iter().enumerate() {
            summary.push_str(&format!("{}. {}\n", i + 1, display_tool_call(call)));
        }

        let sanitized = sanitize_user_facing_reply(&summary);

        // The sanitized output should NOT contain "that" replacements
        assert!(
            !sanitized.contains("\n1. that\n"),
            "display_tool_call output was stripped by sanitizer: {}",
            sanitized
        );
        // Should preserve meaningful content
        assert!(
            sanitized.contains("Searched memories"),
            "sanitized: {}",
            sanitized
        );
        assert!(
            sanitized.contains("Searched files"),
            "sanitized: {}",
            sanitized
        );
        assert!(
            sanitized.contains("Checked goal history"),
            "sanitized: {}",
            sanitized
        );
        assert!(
            sanitized.contains("Ran command"),
            "sanitized: {}",
            sanitized
        );
        assert!(sanitized.contains("Read file"), "sanitized: {}", sanitized);
        assert!(sanitized.contains("Wrote file"), "sanitized: {}", sanitized);
        assert!(sanitized.contains("Web search"), "sanitized: {}", sanitized);
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
            replay_notes: Vec::new(),
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
            replay_notes: Vec::new(),
        };
        let result = graceful_budget_response(&ctx, 500_000);
        assert!(result.len() <= 1502); // 1500 + "…"
    }

    #[test]
    fn test_graceful_goal_daily_budget_response_mentions_budget_and_reset() {
        let ctx = LearningContext {
            user_text: "run the scheduled build".to_string(),
            intent_domains: vec![],
            tool_calls: vec![
                "system_info({})".to_string(),
                "write_file(index.html)".to_string(),
            ],
            errors: vec![("temporary issue".to_string(), false)],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            replay_notes: Vec::new(),
        };

        let result = graceful_goal_daily_budget_response(&ctx, 60, 60, true);
        assert!(result.contains("scheduled goal hit its daily processing budget"));
        assert!(result.contains("used 60 / 60 tokens"));
        assert!(result.contains("00:00 UTC"));
        assert!(result.contains("budget_daily"));
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
            replay_notes: Vec::new(),
        };
        let result = graceful_partial_stall_response(&ctx, false, "deferred", &HashMap::new());
        assert!(result.contains("some progress"));
        assert!(result.contains("Activity summary:"));
        // Error explanation should appear since ctx has an error
        assert!(result.contains("Issues encountered:"));
        assert!(result.contains("not found"));
        // Should NOT contain internal details
        assert!(!result.contains("tool calls executed"));
        assert!(!result.contains("errors encountered"));
        assert!(!result.contains("Stopping reason"));
    }

    #[test]
    fn test_graceful_stall_response_includes_failure_summary() {
        let ctx = make_learning_ctx();
        let mut tool_failure_count = HashMap::new();
        tool_failure_count.insert("terminal".to_string(), semantic_failure_limit("terminal"));
        let result = graceful_stall_response(&ctx, false, "deferred-no-tool", &tool_failure_count);
        assert!(result.contains("Blocked capabilities due to repeated errors:"));
        assert!(result.contains("command execution"));
        assert!(!result.contains("terminal"));
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
            replay_notes: Vec::new(),
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
    fn test_is_productive_short_productive_run() {
        let ctx = make_learning_ctx();
        assert!(is_productive(&ctx, 0, 0, 0, 3));
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
            replay_notes: Vec::new(),
        }
    }

    #[test]
    fn test_classify_stall_prefers_tool_policy_block() {
        let ctx = ctx_with_single_error(
            "Command 'npm install tailwindcss' is not in the safe command list. Use 'terminal' for this command.",
        );
        let (label, suggestion) = classify_stall(&ctx, "deferred-no-tool", &HashMap::new());
        assert_eq!(label, "Tool Policy Block");
        assert!(suggestion.contains("safety policy"));
    }

    #[test]
    fn test_classify_stall_detects_edit_target_drift() {
        let ctx = ctx_with_single_error(
            "Text not found in ~/projects/oaxaca-mezcal-tours/src/components/ContactForm.jsx. The old_text did not match.",
        );
        let (label, suggestion) = classify_stall(&ctx, "deferred-no-tool", &HashMap::new());
        assert_eq!(label, "Edit Target Drift");
        assert!(suggestion.contains("re-read"));
    }

    #[test]
    fn test_classify_stall_ignores_generic_5000_values() {
        let ctx = ctx_with_single_error("Exceeded 5000 characters while building summary.");
        let (label, _) = classify_stall(&ctx, "deferred-no-tool", &HashMap::new());
        assert_eq!(label, "Stuck");
    }

    #[test]
    fn test_classify_stall_detects_provider_server_status_codes() {
        let ctx = ctx_with_single_error("OpenAI API returned status code 503 Service Unavailable.");
        let (label, _) = classify_stall(&ctx, "deferred-no-tool", &HashMap::new());
        assert_eq!(label, "Server Error");
    }

    #[test]
    fn test_classify_stall_detects_tool_lockout_from_counts() {
        let ctx = make_learning_ctx();
        let mut tool_failure_count = HashMap::new();
        tool_failure_count.insert("terminal".to_string(), semantic_failure_limit("terminal"));
        let (label, suggestion) = classify_stall(&ctx, "deferred-no-tool", &tool_failure_count);
        assert_eq!(label, "Tool Locked Out");
        assert!(suggestion.contains("locked out"));
    }

    #[test]
    fn test_tool_failure_summary_format() {
        let mut tool_failure_count = HashMap::new();
        tool_failure_count.insert("terminal".to_string(), semantic_failure_limit("terminal"));
        tool_failure_count.insert(
            "web_search".to_string(),
            semantic_failure_limit("web_search"),
        );
        tool_failure_count.insert("web_fetch".to_string(), semantic_failure_limit("web_fetch"));
        let summary = format_tool_failure_summary(&tool_failure_count);
        assert!(summary.contains("Blocked capabilities due to repeated errors:"));
        assert!(summary.contains("- command execution"));
        assert!(summary.contains("- web access"));
        let web_access_mentions = summary.matches("web access").count();
        assert_eq!(web_access_mentions, 1, "web access should be deduplicated");
    }

    #[test]
    fn test_tool_failure_summary_labels_http_request_as_api_access() {
        let mut tool_failure_count = HashMap::new();
        tool_failure_count.insert(
            "http_request".to_string(),
            semantic_failure_limit("http_request"),
        );
        let summary = format_tool_failure_summary(&tool_failure_count);
        assert!(summary.contains("- API access"));
        assert!(!summary.contains("other capability"));
    }

    #[test]
    fn test_summarize_replay_notes_groups_categories() {
        let summary = summarize_replay_notes(&[
            ReplayNote {
                category: ReplayNoteCategory::PlanRevision,
                code: "plan_rejected".to_string(),
                summary: "Rejected the first deploy step.".to_string(),
                blocking: true,
            },
            ReplayNote {
                category: ReplayNoteCategory::ValidationFailure,
                code: "verification_pending".to_string(),
                summary: "Verification was still pending.".to_string(),
                blocking: true,
            },
        ]);
        assert!(summary.contains("plan_revision: Rejected the first deploy step."));
        assert!(summary.contains("validation_failure: Verification was still pending."));
    }

    #[tokio::test]
    async fn test_process_learning_records_reasoning_failure_patterns() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 32, None, embedding_service)
                .await
                .unwrap(),
        );

        let ctx = LearningContext {
            user_text: "deploy the app".to_string(),
            intent_domains: vec!["deploy".to_string()],
            tool_calls: vec!["read_file(src/main.rs)".to_string()],
            errors: Vec::new(),
            first_error: None,
            recovery_actions: Vec::new(),
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 1,
            replay_notes: vec![ReplayNote {
                category: ReplayNoteCategory::ValidationFailure,
                code: "verification_pending".to_string(),
                summary: "Blocked completion until final verification could run.".to_string(),
                blocking: true,
            }],
        };

        process_learning(&(state.clone() as Arc<dyn StateStore>), ctx)
            .await
            .unwrap();

        let patterns = state.get_behavior_patterns(0.0).await.unwrap();
        let pattern = patterns
            .iter()
            .find(|pattern| pattern.pattern_type == "reasoning_failure")
            .expect("reasoning failure pattern");
        assert_eq!(
            pattern.trigger_context.as_deref(),
            Some("validation_failure:verification_pending")
        );
        assert_eq!(
            pattern.action.as_deref(),
            Some("run_verification_before_claiming_success")
        );
    }

    #[test]
    fn test_format_error_explanation_empty_on_no_errors() {
        let result = format_error_explanation(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_error_explanation_marks_recovered() {
        let errors = vec![
            ("command not found: drush".to_string(), true),
            (
                "Unable to install modules entity_reference".to_string(),
                false,
            ),
        ];
        let result = format_error_explanation(&errors);
        assert!(result.contains("Issues encountered:"));
        assert!(result.contains("command not found: drush (resolved)"));
        assert!(result.contains("Unable to install modules entity_reference"));
        // The unrecovered error should NOT have "(resolved)"
        assert!(!result.contains("entity_reference (resolved)"));
    }

    #[test]
    fn test_format_error_explanation_dedup_prefers_unrecovered() {
        let errors = vec![
            ("Error: command not found: drush".to_string(), true),
            ("Error: command not found: drush".to_string(), false),
        ];
        let result = format_error_explanation(&errors);
        // Same error appears twice — once recovered, once not. Should show as unrecovered.
        assert!(result.contains("command not found: drush"));
        assert!(!result.contains("(resolved)"));
        // Should only appear once (deduped)
        assert_eq!(result.matches("command not found: drush").count(), 1);
    }

    #[test]
    fn test_format_error_explanation_preserves_order() {
        let errors = vec![
            ("Error: first problem".to_string(), false),
            ("Error: second problem".to_string(), false),
            ("Error: third problem".to_string(), false),
        ];
        let result = format_error_explanation(&errors);
        let first_pos = result.find("first problem").unwrap();
        let second_pos = result.find("second problem").unwrap();
        let third_pos = result.find("third problem").unwrap();
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    #[test]
    fn test_format_error_explanation_redacts_secrets() {
        // sk- followed by 20+ alphanumeric chars matches the API key pattern
        let errors = vec![(
            "Error: Invalid API key sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDEF".to_string(),
            false,
        )];
        let result = format_error_explanation(&errors);
        assert!(result.contains("Issues encountered:"));
        assert!(!result.contains("sk-abcdef"));
        assert!(result.contains("[REDACTED:"));
    }

    #[test]
    fn test_format_error_explanation_keeps_safe_path_tail() {
        let errors = vec![(
            "Error: Text not found in /Users/alice/projects/plants-site/src/main.rs.".to_string(),
            false,
        )];
        let result = format_error_explanation(&errors);
        assert!(result.contains("[path:.../main.rs]."));
        assert!(!result.contains("/Users/alice/projects/plants-site/src/main.rs"));
        assert!(!result.contains("[REDACTED:File path]"));
    }

    #[test]
    fn test_format_error_explanation_prefers_recent_errors() {
        // When more than 5 errors exist, only the last 5 should be considered.
        // This ensures the user sees the final blocker, not stale resolved issues.
        let errors: Vec<(String, bool)> = (1..=8)
            .map(|i| (format!("Error: problem number {}", i), i <= 3))
            .collect();
        let result = format_error_explanation(&errors);
        // Errors 1-3 are old (indices 0-2) and should be skipped (recent_start=3).
        assert!(
            !result.contains("problem number 1"),
            "Stale error 1 should not appear"
        );
        assert!(
            !result.contains("problem number 2"),
            "Stale error 2 should not appear"
        );
        assert!(
            !result.contains("problem number 3"),
            "Stale error 3 should not appear"
        );
        // Recent errors (4-8) should be represented (display capped at 3 deduped).
        // At minimum, the most recent errors should appear.
        assert!(
            result.contains("problem number"),
            "Recent errors should appear"
        );
    }

    #[test]
    fn test_graceful_stall_includes_error_explanation() {
        let mut ctx = make_learning_ctx();
        ctx.errors = vec![(
            "Unable to install modules entity_reference due to missing modules".to_string(),
            false,
        )];
        let result = graceful_stall_response(&ctx, false, "deferred-no-tool", &HashMap::new());
        assert!(result.contains("Issues encountered:"));
        assert!(result.contains("Unable to install modules entity_reference"));
    }

    #[test]
    fn test_graceful_stall_no_error_section_when_clean() {
        let ctx = make_learning_ctx(); // no errors
        let result = graceful_stall_response(&ctx, false, "deferred-no-tool", &HashMap::new());
        assert!(!result.contains("Issues encountered:"));
    }
}
