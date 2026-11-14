//! After a run: what it teaches memory, whether it stalled and why, and the
//! user-facing words for a run that ran out of budget.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use regex::Regex;
use std::sync::LazyLock;

use aidaemon_events::TaskOutcome;

use crate::turn::loop_utils::{MAX_CONSECUTIVE_SAME_TOOL, semantic_failure_limit};

#[derive(Clone)]
pub struct LearningContext {
    pub user_text: String,
    /// False when the task's compiled capability policy suppresses all
    /// automatic memory persistence. Canonical task/event telemetry is still
    /// recorded; only learned procedures, expertise, and failure patterns are
    /// skipped.
    pub memory_persistence_allowed: bool,
    pub intent_domains: Vec<String>,
    pub tool_calls: Vec<String>,     // "tool_name(summary)"
    pub errors: Vec<(String, bool)>, // (error_text, was_recovered)
    pub first_error: Option<String>,
    pub recovery_actions: Vec<String>,
    #[allow(dead_code)] // Reserved for duration-based learning
    pub start_time: chrono::DateTime<Utc>,
    pub completed_naturally: bool,
    pub explicit_positive_signals: u32,
    pub explicit_negative_signals: u32,
    /// Semantic outcome emitted at task end; authoritative for learning when set.
    pub task_outcome: Option<TaskOutcome>,
}

impl LearningContext {
    /// What the full memory system learns from: this run, with the first
    /// error stripped of the daemon's scaffolding lines.
    pub fn into_run_learning(self) -> aidaemon_memory::full::writer::RunLearning {
        aidaemon_memory::full::writer::RunLearning {
            user_text: self.user_text,
            persistence_allowed: self.memory_persistence_allowed,
            intent_domains: self.intent_domains,
            tool_calls: self.tool_calls,
            errors: self.errors,
            first_error: self.first_error.map(|error| {
                error
                    .lines()
                    .filter(|line| !crate::message_content::is_internal_scaffolding_line(line))
                    .collect::<Vec<_>>()
                    .join("\n")
            }),
            recovery_actions: self.recovery_actions,
            completed_naturally: self.completed_naturally,
            explicit_positive_signals: self.explicit_positive_signals,
            explicit_negative_signals: self.explicit_negative_signals,
            succeeded: self.task_outcome.map(|outcome| outcome.task_success()),
        }
    }
}

/// Process learning from a completed task - runs in background.
pub async fn process_learning(
    state: &Arc<dyn aidaemon_memory::stores::MemoryStores>,
    ctx: LearningContext,
) -> anyhow::Result<()> {
    aidaemon_memory::full::writer::learn_from_run(state.as_ref(), ctx.into_run_learning()).await
}

/// Learn from a finished task in the background.
pub fn spawn_learning(
    store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    learning_ctx: LearningContext,
) {
    tokio::spawn(async move {
        if let Err(e) = process_learning(&store, learning_ctx).await {
            tracing::warn!("Learning failed: {}", e);
        }
    });
}

/// Classify the stall cause from recent errors for actionable guidance.
pub fn classify_stall(
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
            "Automatic provider recovery remained rate-limited.",
        )
    } else if recent_errors.contains("timed out") || recent_errors.contains("timeout") {
        (
            "Timeout",
            "Automatic recovery still exceeded the current response-time limits.",
        )
    } else if recent_errors.contains("network") || recent_errors.contains("connection") {
        (
            "Network Error",
            "Automatic recovery could not get a working network connection.",
        )
    } else if is_tool_policy_block(&recent_errors) {
        (
            "Tool Policy Block",
            "A step was blocked by safety policy. Try adjusting the request or running the blocked command yourself.",
        )
    } else if is_edit_target_drift(&recent_errors) {
        (
            "Edit Target Drift",
            "The edit target kept changing after automatic re-read and retry recovery.",
        )
    } else if looks_like_provider_server_error(&recent_errors) {
        (
            "Server Error",
            "Automatic provider recovery remained blocked by server errors.",
        )
    } else if recent_errors.contains("unknown tool") {
        (
            "Unknown Tool",
            "Automatic tool-resolution recovery could not find a working capability.",
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
            "Automatic tool and model recovery did not produce an executable step.",
        )
    } else {
        (
            "Stuck",
            "The available in-run recovery strategies were exhausted.",
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

/// Graceful response when task token budget is exhausted.
pub fn graceful_budget_response(learning_ctx: &LearningContext, _tokens_used: u64) -> String {
    let activity = categorize_tool_calls_user_facing(&learning_ctx.tool_calls);
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
pub fn graceful_scheduled_run_budget_response(
    learning_ctx: &LearningContext,
    tokens_used: i64,
    budget_per_check: i64,
) -> String {
    let activity = categorize_tool_calls_user_facing(&learning_ctx.tool_calls);
    let mut summary = format!(
        "This scheduled run hit its per-run processing budget (used {} / {} tokens). \
            Here's what I accomplished so far:\n\n{}\
            The run used its bounded autonomous compute envelope, so it stopped instead of \
            spending indefinitely. Recorded work remains available to the recurring workflow; \
            no owner action is required.",
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
pub fn graceful_goal_daily_budget_response(
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
        "This scheduled goal hit its cumulative daily processing budget across today's runs"
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
            "This total is not the cost of the current run alone. The recurring schedule remains active and no owner action is required; the next eligible run can resume after the UTC reset.",
        );
    } else {
        msg.push_str("You can ask me to continue this later, or increase the goal's daily budget.");
    }
    msg
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
pub fn is_productive(
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
    total_activity == 0 || total_successful_tool_calls * 4 >= total_activity * 3
}

static ABS_UNIX_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"/(?:Users|home|etc)/[^\s]+").expect("unix path regex must compile")
});
static ABS_WINDOWS_PATH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z]:[\\/][^\s]+").expect("windows path regex must compile"));

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

pub fn redact_error_line_for_summary(key_line: &str) -> String {
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
    aidaemon_tools::sanitize::redact_secrets(&windows_summarized)
}

///
/// Parses entries like `"read_file(Hero.jsx)"` and `"terminal(\`pip install fpdf\`)"` into
/// grouped categories so the next interaction can understand what was already done.
///
/// Recap for internal/LLM context — keeps raw command text so the model can see
/// exactly what already ran (helps avoid re-running the same command).
pub fn categorize_tool_calls(tool_calls: &[String]) -> String {
    categorize_tool_calls_inner(tool_calls, false)
}

/// Recap for user-facing replies — redacts raw shell commands to a count so they
/// never reach chat, while keeping file / search / source detail.
pub fn categorize_tool_calls_user_facing(tool_calls: &[String]) -> String {
    categorize_tool_calls_inner(tool_calls, true)
}

fn categorize_tool_calls_inner(tool_calls: &[String], redact_commands: bool) -> String {
    let mut files_read: Vec<&str> = Vec::new();
    let mut files_written: Vec<&str> = Vec::new();
    let mut commands_run: Vec<&str> = Vec::new();
    let mut files_sent: Vec<&str> = Vec::new();
    let mut searches: Vec<&str> = Vec::new();
    let mut external_reads: Vec<&str> = Vec::new();
    let mut other: Vec<&str> = Vec::new();

    let mut failed_mutations: Vec<&str> = Vec::new();

    for entry in tool_calls {
        // Detect and strip [FAILED] suffix — failed tool calls are tracked
        // separately so activity summaries don't claim success on failures.
        let (clean_entry, is_failed, was_rejected) =
            if let Some(stripped) = entry.strip_suffix(" [FAILED]") {
                (stripped, true, false)
            } else if let Some(stripped) = entry.strip_suffix(" [REJECTED]") {
                (stripped, false, true)
            } else {
                (entry.as_str(), false, false)
            };

        // Parse "tool_name(summary)" format
        let (name, args) = match clean_entry.find('(') {
            Some(idx) => {
                let name = &clean_entry[..idx];
                let args = clean_entry[idx + 1..].trim_end_matches(')');
                (name, args)
            }
            None => (clean_entry, ""),
        };

        // Failed write/edit operations go to a separate bucket.
        if is_failed && matches!(name, "write_file" | "edit_file") {
            if !args.is_empty() {
                failed_mutations.push(args);
            }
            continue;
        }
        // A failed or pre-I/O-rejected invocation is an attempted operation,
        // not completed activity. Its concrete outcome remains in the tool
        // receipt/model context and must not become “Files read” or “Commands
        // run” in a fallback completion card.
        if is_failed || was_rejected {
            continue;
        }

        match name {
            "read_file" => files_read.push(args),
            "write_file" | "edit_file" => files_written.push(args),
            "terminal" | "run_command" => commands_run.push(args),
            "send_file" | "send_media" => files_sent.push(args),
            "web_search" | "search_files" => searches.push(args),
            "web_fetch" | "http_request" => external_reads.push(args),
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
        if redact_commands {
            // Never surface raw shell commands in user-facing recaps — count only.
            let n = commands_run.len();
            sections.push(format!(
                "Commands run: {} command{}",
                n,
                if n == 1 { "" } else { "s" }
            ));
        } else {
            let items: Vec<&str> = commands_run.iter().copied().take(10).collect();
            sections.push(format!("Commands run: {}", items.join(", ")));
        }
    }
    if !files_sent.is_empty() {
        let items: Vec<&str> = files_sent.iter().copied().take(5).collect();
        sections.push(format!("Files sent: {}", items.join(", ")));
    }
    if !searches.is_empty() {
        // Dedupe and drop empty-arg entries — repeated globs otherwise render
        // as "Searches: acme, , acme".
        let mut seen = std::collections::BTreeSet::new();
        let items: Vec<&str> = searches
            .iter()
            .copied()
            .filter(|s| !s.trim().is_empty() && seen.insert(s.trim()))
            .take(5)
            .collect();
        if !items.is_empty() {
            sections.push(format!("Searches: {}", items.join(", ")));
        }
    }
    if !external_reads.is_empty() {
        let items: Vec<&str> = external_reads.iter().copied().take(8).collect();
        sections.push(format!("External sources checked: {}", items.join(", ")));
    }
    if !failed_mutations.is_empty() {
        let items: Vec<&str> = failed_mutations.iter().copied().take(5).collect();
        sections.push(format!(
            "Failed writes (permission/error): {}",
            items.join(", ")
        ));
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
            "read_file(chantilly-va-site)".to_string(),
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
    fn user_facing_recap_redacts_raw_commands_to_a_count() {
        let calls = vec![
            "read_file(netprobe6.py)".to_string(),
            "terminal(`cd '/tmp' && timeout 45s python3 /tmp/netprobe6.py`)".to_string(),
            "terminal(`cd '/tmp' && python3 /tmp/netprobe6.py`)".to_string(),
        ];
        let internal = categorize_tool_calls(&calls);
        let user = categorize_tool_calls_user_facing(&calls);
        // Internal recap keeps the raw command so the model can see what ran.
        assert!(internal.contains("cd '/tmp'"));
        // User-facing recap NEVER surfaces a raw shell command.
        assert!(
            !user.contains("cd '/tmp'") && !user.contains("python3"),
            "user-facing recap leaked a command: {user}"
        );
        assert!(user.contains("Commands run: 2 commands"));
        // File detail is still shown to the user.
        assert!(user.contains("Files read: netprobe6.py"));
    }

    #[test]
    fn test_categorize_tool_calls_empty() {
        let result = categorize_tool_calls(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn failed_and_rejected_attempts_are_not_completed_activity() {
        let calls = vec![
            "read_file(synthetic.toml) [REJECTED]".to_string(),
            "terminal(`/usr/bin/false`) [FAILED]".to_string(),
        ];
        assert!(categorize_tool_calls(&calls).is_empty());
        assert!(categorize_tool_calls_user_facing(&calls).is_empty());
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
            memory_persistence_allowed: true,
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
            task_outcome: None,
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
            memory_persistence_allowed: true,
            intent_domains: vec![],
            tool_calls: calls,
            errors: vec![],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            task_outcome: None,
        };
        let result = graceful_budget_response(&ctx, 500_000);
        assert!(result.len() <= 1502); // 1500 + "…"
    }

    #[test]
    fn test_graceful_goal_daily_budget_response_mentions_budget_and_reset() {
        let ctx = LearningContext {
            user_text: "run the scheduled build".to_string(),
            memory_persistence_allowed: true,
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
            task_outcome: None,
        };

        let result = graceful_goal_daily_budget_response(&ctx, 60, 60, true);
        assert!(result.contains("cumulative daily processing budget across today's runs"));
        assert!(result.contains("used 60 / 60 tokens"));
        assert!(result.contains("not the cost of the current run alone"));
        assert!(result.contains("00:00 UTC"));
        assert!(result.contains("no owner action is required"));
    }

    fn make_learning_ctx() -> LearningContext {
        LearningContext {
            user_text: "deploy the app".to_string(),
            memory_persistence_allowed: true,
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
            task_outcome: None,
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
            memory_persistence_allowed: true,
            intent_domains: vec![],
            tool_calls: vec![],
            errors: vec![(error.to_string(), false)],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            task_outcome: None,
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
    fn process_learning_uses_task_outcome_for_partial_and_failed() {
        let partial_ctx = LearningContext {
            user_text: "inspect homepage".to_string(),
            memory_persistence_allowed: true,
            intent_domains: vec!["web".to_string()],
            tool_calls: vec!["browser(navigate)".to_string()],
            errors: vec![],
            first_error: None,
            recovery_actions: vec![],
            start_time: Utc::now(),
            completed_naturally: true,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            task_outcome: Some(aidaemon_events::TaskOutcome::Partial),
        };
        assert!(!aidaemon_events::TaskOutcome::Partial.task_success());

        let failed_ctx = LearningContext {
            task_outcome: Some(aidaemon_events::TaskOutcome::Failed),
            ..partial_ctx.clone()
        };
        assert!(!aidaemon_events::TaskOutcome::Failed.task_success());
        assert!(!failed_ctx.task_outcome.unwrap().task_success());

        let succeeded_ctx = LearningContext {
            task_outcome: Some(aidaemon_events::TaskOutcome::Succeeded),
            ..partial_ctx
        };
        assert!(aidaemon_events::TaskOutcome::Succeeded.task_success());
        assert!(succeeded_ctx.task_outcome.unwrap().task_success());
    }
}
