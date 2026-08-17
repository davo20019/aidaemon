//! Goal/task dispatch helpers extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Groups wait-prefix parsing, scheduled-run provenance/state persistence, file
//! delivery helpers, and goal/task result summary builders.

use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::goal_tokens::{GoalRunBudgetStatus, GoalTokenRegistry};
use crate::traits::{Goal, ScheduledRunState, StateStore, Task};

use super::Agent;

/// Recurring runs may spend one additional envelope when their durable health
/// snapshot shows concrete progress. This is intentionally small: autonomy
/// means adapting inside a bounded policy, not silently turning a runaway run
/// into unbounded spend.
pub(crate) const SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS: usize = 1;
pub(crate) const SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP: i64 = 2_000_000;

/// Blocked path patterns for auto-sent files (mirrors SendFileTool).
const AUTO_SEND_BLOCKED_PATTERNS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".env",
    "credentials",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "config.toml",
];

/// Extract absolute file paths from text (e.g. goal completion messages).
/// Only returns paths that exist on disk and aren't security-sensitive.
pub(crate) fn extract_file_paths_from_text(text: &str) -> Vec<String> {
    let re = regex::Regex::new(r"(/[\w./-]+\.\w{1,10})").unwrap();
    let mut paths = Vec::new();
    for cap in re.captures_iter(text) {
        let path_str = &cap[1];
        let path = std::path::Path::new(path_str);

        // Must exist and be a regular file
        if !path.exists() || !path.is_file() {
            continue;
        }

        // Check against blocked patterns
        let path_display = path.to_string_lossy();
        let blocked = AUTO_SEND_BLOCKED_PATTERNS.iter().any(|pattern| {
            if pattern.starts_with('.') || pattern.starts_with('/') {
                path_display.contains(&format!("/{}", pattern))
                    || path_display.contains(&format!("/{}/", pattern))
            } else {
                path.file_name()
                    .map(|n| n.to_string_lossy() == *pattern)
                    .unwrap_or(false)
                    || path_display.contains(&format!("/{}", pattern))
                    || path_display.contains(&format!("/{}/", pattern))
            }
        });
        if blocked {
            continue;
        }

        // Also block .key and .pem extensions
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy();
            if ext == "key" || ext == "pem" {
                continue;
            }
        }

        paths.push(path_str.to_string());
    }
    paths
}

/// Parse simple wait instructions like "Wait for 5 minutes." into seconds.
/// Returns None when the task is not a plain wait command.
pub(in crate::agent) fn parse_wait_task_seconds(task_description: &str) -> Option<u64> {
    static WAIT_TASK_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*wait\s+for\s+(\d+)\s*(seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\b",
        )
        .expect("wait task regex should compile")
    });

    let caps = WAIT_TASK_RE.captures(task_description.trim())?;
    let value: u64 = caps.get(1)?.as_str().parse().ok()?;
    let unit = caps.get(2)?.as_str().to_ascii_lowercase();

    match unit.as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(value),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(value.saturating_mul(60)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(value.saturating_mul(3600)),
        _ => None,
    }
}

/// Parse a leading wait/delay prefix from a goal description.
/// Handles patterns like:
///   - "wait for 2 minutes then check disk space"
///   - "in 5 minutes check disk space"
///   - "after 30 seconds run df -h"
///   - "wait 2 minutes" (pure wait, no remainder)
///
/// Returns the wait duration in seconds, or None if no wait prefix is found.
pub(in crate::agent) fn parse_goal_leading_wait(description: &str) -> Option<u64> {
    static LEADING_WAIT_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*(?:wait\s+(?:for\s+)?|in\s+|after\s+)(\d+)\s*(seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\b",
        )
        .expect("leading wait regex should compile")
    });

    let caps = LEADING_WAIT_RE.captures(description.trim())?;
    let value: u64 = caps.get(1)?.as_str().parse().ok()?;
    let unit = caps.get(2)?.as_str().to_ascii_lowercase();

    match unit.as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(value),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(value.saturating_mul(60)),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(value.saturating_mul(3600)),
        _ => None,
    }
}

/// Strip the leading wait/delay prefix from a goal description, returning
/// the remainder (the actual work to do after the wait).
/// Returns empty string if there's nothing meaningful after the wait.
pub(in crate::agent) fn strip_leading_wait(description: &str) -> String {
    static STRIP_WAIT_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)^\s*(?:wait\s+(?:for\s+)?|in\s+|after\s+)\d+\s*(?:seconds?|secs?|s|minutes?|mins?|min|m|hours?|hrs?|h)\s*[,;]?\s*(?:then\s+|and\s+|,\s*)?",
        )
        .expect("strip wait regex should compile")
    });

    let remainder = STRIP_WAIT_RE.replace(description.trim(), "").to_string();
    let trimmed = remainder.trim().to_string();
    // If what's left is too short to be a real instruction, treat as pure wait
    if trimmed.len() < 3 {
        String::new()
    } else {
        trimmed
    }
}

/// Check whether a session ID corresponds to a group/public channel (not a DM).
/// Used to suppress noisy progress updates in shared channels.
pub fn is_group_session(session_id: &str) -> bool {
    crate::session::is_group_session(session_id)
}

pub(in crate::agent) fn user_facing_task_description(description: &str) -> String {
    let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(description.trim());
    let collapsed = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        "current task".to_string()
    } else {
        collapsed
    }
}

pub(in crate::agent) async fn task_has_scheduled_provenance(
    state: &Arc<dyn StateStore>,
    task_id: Option<&str>,
) -> bool {
    if let Some(tid) = task_id {
        if let Ok(Some(run)) = state.get_goal_run_for_task(tid).await {
            return run.trigger_type == "scheduled";
        }
    }

    false
}

pub(in crate::agent) async fn active_scheduled_root_task_id(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
) -> Option<String> {
    state
        .get_goal_runs(goal_id)
        .await
        .ok()?
        .into_iter()
        .filter(|run| run.trigger_type == "scheduled")
        .filter(|run| !matches!(run.status.as_str(), "completed" | "failed" | "cancelled"))
        .find_map(|run| run.root_task_id)
}

pub(in crate::agent) async fn goal_has_scheduled_provenance(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
    task_id: Option<&str>,
) -> bool {
    if task_has_scheduled_provenance(state, task_id).await {
        return true;
    }

    if let Ok(schedules) = state.get_schedules_for_goal(goal_id).await {
        if !schedules.is_empty() {
            return true;
        }
    }

    if let Ok(runs) = state.get_goal_runs(goal_id).await {
        if runs.iter().any(|run| run.trigger_type == "scheduled") {
            return true;
        }
    }

    false
}

pub(in crate::agent) async fn persist_scheduled_run_state(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
    root_task_id_hint: Option<&str>,
    status: &GoalRunBudgetStatus,
) {
    let existing = state.get_scheduled_run_state(goal_id).await.ok().flatten();
    // A caller-provided root always identifies the current run. Reusing the
    // prior persisted root here mixed token accounting across separate fires.
    let root_task_id = if let Some(root_task_id) = root_task_id_hint {
        Some(root_task_id.to_string())
    } else if let Some(record) = existing.as_ref() {
        Some(record.root_task_id.clone())
    } else {
        active_scheduled_root_task_id(state, goal_id).await
    };

    let Some(root_task_id) = root_task_id else {
        return;
    };

    let now = chrono::Utc::now().to_rfc3339();
    let created_at = existing
        .as_ref()
        .filter(|record| record.root_task_id == root_task_id)
        .map(|record| record.created_at.clone())
        .unwrap_or_else(|| now.clone());
    let record = ScheduledRunState {
        goal_id: goal_id.to_string(),
        root_task_id,
        effective_budget_per_check: status.effective_budget_per_check,
        tokens_used: status.tokens_used,
        budget_extensions_count: status.budget_extensions_count,
        health: status.health.clone(),
        created_at,
        updated_at: now,
    };
    let _ = state.upsert_scheduled_run_state(&record).await;
}

pub(in crate::agent) async fn clear_scheduled_run_state(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
) {
    let _ = state.delete_scheduled_run_state(goal_id).await;
}

pub(in crate::agent) fn auto_dispatch_scheduled_run_extension_budget(
    status: &GoalRunBudgetStatus,
    max_budget_extensions: usize,
    hard_token_cap: i64,
) -> Option<i64> {
    let old_budget = status.effective_budget_per_check;
    let new_budget = old_budget
        .saturating_mul(2)
        .max(status.tokens_used.saturating_add(old_budget / 2))
        .min(hard_token_cap);

    let has_meaningful_progress = Agent::scheduled_run_has_structural_progress(&status.health);
    let clearly_unproductive =
        Agent::scheduled_run_metrics_are_clearly_unproductive(&status.health);

    if status.budget_extensions_count < max_budget_extensions
        && old_budget < hard_token_cap
        && new_budget > status.tokens_used
        && has_meaningful_progress
        && !clearly_unproductive
    {
        Some(new_budget)
    } else {
        None
    }
}

pub(in crate::agent) async fn effective_goal_daily_budget(
    goal: &Goal,
    registry: Option<&GoalTokenRegistry>,
) -> Option<i64> {
    let shared = if let Some(registry) = registry {
        registry.get_effective_daily_budget(&goal.id).await
    } else {
        None
    };
    shared.or(goal.budget_daily)
}

pub(in crate::agent) fn truncate_goal_result_text(text: &str, max_chars: usize) -> String {
    let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(text);
    let trimmed = sanitized.trim();
    let truncated: String = trimmed.chars().take(max_chars).collect();
    if trimmed.chars().count() > max_chars {
        format!("{truncated}...")
    } else {
        truncated
    }
}

pub(in crate::agent) fn goal_failure_summary_from_context(goal: &Goal) -> Option<String> {
    goal.context
        .as_deref()
        .and_then(|ctx| serde_json::from_str::<serde_json::Value>(ctx).ok())
        .and_then(|ctx| {
            ctx.get("failure_summary")
                .and_then(|v| v.as_str())
                .map(ToOwned::to_owned)
        })
        .map(|summary| summary.trim().to_string())
        .filter(|summary| !summary.is_empty())
}

pub(in crate::agent) fn latest_problem_task_summary(tasks: &[Task]) -> Option<String> {
    tasks
        .iter()
        .filter(|task| matches!(task.status.as_str(), "failed" | "blocked"))
        .max_by(|a, b| {
            let a_key = a
                .completed_at
                .as_deref()
                .or(a.started_at.as_deref())
                .unwrap_or(a.created_at.as_str());
            let b_key = b
                .completed_at
                .as_deref()
                .or(b.started_at.as_deref())
                .unwrap_or(b.created_at.as_str());
            a_key
                .cmp(b_key)
                .then_with(|| a.task_order.cmp(&b.task_order))
                .then_with(|| a.id.cmp(&b.id))
        })
        .and_then(|task| {
            let detail = task
                .result
                .as_deref()
                .or(task.error.as_deref())
                .or(task.blocker.as_deref())
                .map(str::trim)
                .filter(|detail| !detail.is_empty())?;
            Some(format!(
                "{}: {}",
                task.description,
                truncate_goal_result_text(detail, 1000)
            ))
        })
}

pub(crate) fn build_goal_failure_summary(
    goal: Option<&Goal>,
    tasks: &[Task],
    task_lead_response: Option<&str>,
    task_lead_error: Option<&str>,
) -> String {
    let mut summary = goal
        .and_then(goal_failure_summary_from_context)
        .or_else(|| {
            task_lead_response
                .map(str::trim)
                .filter(|reply| !reply.is_empty())
                .map(ToOwned::to_owned)
        })
        .or_else(|| latest_problem_task_summary(tasks))
        .or_else(|| {
            task_lead_error
                .map(str::trim)
                .filter(|err| !err.is_empty())
                .map(ToOwned::to_owned)
        })
        .or_else(|| goal.map(|g| g.description.clone()))
        .unwrap_or_else(|| "task lead exited without completing all tasks".to_string());

    if summary.to_ascii_lowercase().starts_with("goal failed:") {
        summary = summary["Goal failed:".len()..].trim().to_string();
    }

    truncate_goal_result_text(&summary, 3500)
}

/// Build a user-facing summary from successful task results.
/// Includes recent completed tasks (bounded) instead of only the last one.
/// Generated activity recaps ("Activity summary: - Commands run: ...") are
/// bookkeeping, not deliverables — order them after substantive results.
fn is_activity_summary_result(task: &Task) -> bool {
    task.result
        .as_deref()
        .map(str::trim)
        .is_some_and(|r| r.starts_with("Activity summary:"))
}

pub(crate) fn build_goal_task_results_summary(
    tasks: &[Task],
    root_task_id: Option<&str>,
    fallback: &str,
) -> String {
    const MAX_INCLUDED_TASK_RESULTS: usize = 3;
    const MAX_CHARS_PER_TASK_RESULT: usize = 800;
    const MAX_CHARS_PRIMARY_RESULT: usize = 2500;

    let mut successful: Vec<&Task> = tasks
        .iter()
        .filter(|t| t.completed_successfully())
        .filter(|t| t.result.as_deref().is_some_and(|r| !r.trim().is_empty()))
        // Root identity is persisted on the run. Description text is payload,
        // never a lifecycle discriminator.
        .filter(|t| Some(t.id.as_str()) != root_task_id)
        .collect();

    if successful.is_empty() {
        return truncate_goal_result_text(fallback, 4000);
    }

    successful.sort_by(|a, b| {
        let a_key = a.completed_at.as_deref().unwrap_or(a.created_at.as_str());
        let b_key = b.completed_at.as_deref().unwrap_or(b.created_at.as_str());
        a_key
            .cmp(b_key)
            .then_with(|| a.task_order.cmp(&b.task_order))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut selected: Vec<&Task> = successful
        .iter()
        .rev()
        .take(MAX_INCLUDED_TASK_RESULTS)
        .copied()
        .collect();
    selected.reverse();

    // Deliverable-first ordering: substantive results lead, generated
    // activity recaps follow. The first section is usually what the user
    // actually asked for, so it gets a larger truncation budget.
    let (mut ordered, bookkeeping): (Vec<&Task>, Vec<&Task>) = selected
        .into_iter()
        .partition(|t| !is_activity_summary_result(t));
    ordered.extend(bookkeeping);
    let selected = ordered;

    let sections: Vec<String> = selected
        .iter()
        .enumerate()
        .map(|(idx, t)| {
            let result = t.result.as_deref().unwrap_or("");
            let cap = if idx == 0 {
                MAX_CHARS_PRIMARY_RESULT
            } else {
                MAX_CHARS_PER_TASK_RESULT
            };
            format!(
                "**{}**\n{}",
                user_facing_task_description(&t.description),
                truncate_goal_result_text(result, cap)
            )
        })
        .collect();

    if sections.is_empty() {
        return truncate_goal_result_text(fallback, 4000);
    }

    let omitted = successful.len().saturating_sub(selected.len());
    if sections.len() == 1 && omitted == 0 {
        // A lone task description is often the entire scheduled-goal prompt.
        // Repeating it as a large bold heading makes the useful result harder to
        // find, especially on mobile. The surrounding notification already gives
        // the user enough context, so lead directly with the outcome.
        return truncate_goal_result_text(
            selected[0].result.as_deref().unwrap_or(""),
            MAX_CHARS_PRIMARY_RESULT,
        );
    }

    let mut summary = format!(
        "{}/{} tasks completed.\n\n{}",
        successful.len(),
        tasks.len(),
        sections.join("\n\n")
    );
    if omitted > 0 {
        let suffix = if omitted == 1 { "" } else { "s" };
        summary.push_str(&format!(
            "\n\n(+{} earlier completed task result{} omitted)",
            omitted, suffix
        ));
    }
    summary
}

#[cfg(test)]
mod summary_tests {
    use super::*;

    fn completed_task(id: &str, order: i32, completed_at: &str, result: &str) -> Task {
        Task {
            id: id.to_string(),
            goal_id: "goal-1".to_string(),
            description: format!("Task {id}"),
            status: "completed".to_string(),
            priority: "medium".to_string(),
            task_order: order,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some(result.to_string()),
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: "2026-06-12T13:00:00Z".to_string(),
            started_at: None,
            completed_at: Some(completed_at.to_string()),
        }
    }

    #[test]
    fn goal_summary_leads_with_deliverable_not_activity_summary() {
        let bookkeeping = completed_task(
            "locate",
            1,
            "2026-06-12T13:05:00Z",
            "Activity summary:\n- Commands run: `cd '~/projects' && ls`",
        );
        // A substantive deliverable larger than the old 800-char per-task cap.
        let module_rows: String = (0..60)
            .map(|i| format!("| drupal/module_{i} | 2.{i}.0 |\n"))
            .collect();
        let deliverable_text = format!(
            "### Current Drupal Modules\n| Module | Version |\n|---|---|\n{module_rows}END_OF_MODULE_LIST"
        );
        let deliverable = completed_task("modules", 2, "2026-06-12T13:10:00Z", &deliverable_text);

        let summary =
            build_goal_task_results_summary(&[bookkeeping, deliverable], None, "fallback text");

        let deliverable_pos = summary
            .find("Current Drupal Modules")
            .expect("deliverable must be included");
        let bookkeeping_pos = summary
            .find("Activity summary:")
            .expect("bookkeeping may follow the deliverable");
        assert!(
            deliverable_pos < bookkeeping_pos,
            "deliverable must come before activity-summary bookkeeping:\n{summary}"
        );
        assert!(
            summary.contains("END_OF_MODULE_LIST"),
            "the primary deliverable must not be cut at the old 800-char cap:\n{summary}"
        );
    }

    #[test]
    fn single_task_summary_does_not_repeat_the_full_task_instruction() {
        let mut task = completed_task(
            "publish",
            1,
            "2026-06-12T13:10:00Z",
            "Published the post and verified the live URL returned HTTP 200.",
        );
        task.description =
            "Create, deploy, and verify a post using these detailed instructions".to_string();

        let summary = build_goal_task_results_summary(&[task], None, "fallback text");

        assert_eq!(
            summary,
            "Published the post and verified the live URL returned HTTP 200."
        );
        assert!(!summary.contains("Create, deploy"));
    }
}
