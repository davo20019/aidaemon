//! Background task-lead spawner extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Pure relocation — no logic changes. Houses the `spawn_background_task_lead`
//! free function (kept as a free fn to satisfy `Send` bounds for the spawned
//! background future).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use tracing::{error, info, warn};

use crate::events::TaskOutcome;
use crate::traits::{
    AgentRole, Mandate, MandateDecisionOutcome, MandateFinalizationRejectReason,
    MandateRunFinalizationRequest, MandateRunFinalizationResult, MandateRunNotification,
    MandateRunNotificationKind, MandateStatus, StateStore, SAFE_FALLBACK_WAIT_RATIONALE,
};
use crate::types::{ChannelContext, UserRole};

use super::parent_delivery;
use super::{
    auto_dispatch_scheduled_run_extension_budget, build_goal_failure_summary,
    build_goal_task_results_summary, clear_scheduled_run_state, effective_goal_daily_budget,
    extract_file_paths_from_text, goal_has_scheduled_provenance, is_goal_run_root_task_description,
    is_group_session, parse_goal_leading_wait, parse_wait_task_seconds,
    persist_scheduled_run_state, strip_leading_wait, truncate_goal_result_text,
    user_facing_task_description, Agent, SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS,
    SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
};

/// Progress-heartbeat wait schedule: quick early updates, then exponential
/// backoff settling at 15 minutes. Replaces the old hard cap of 4 messages,
/// which left long-running goals completely silent after ~2 minutes.
fn heartbeat_wait_secs(interval_count: u32) -> u64 {
    const SCHEDULE: [u64; 6] = [15, 30, 60, 120, 300, 600];
    SCHEDULE
        .get(interval_count as usize)
        .copied()
        .unwrap_or(900)
}

fn count_successful_completed_work(
    tasks: &[crate::traits::Task],
    dispatch_trigger_task_id: Option<&str>,
) -> usize {
    tasks
        .iter()
        .filter(|task| Some(task.id.as_str()) != dispatch_trigger_task_id)
        .filter(|task| task.completed_successfully())
        .count()
}

fn task_failed_only_due_to_provider_infrastructure(task: &crate::traits::Task) -> bool {
    task.status == "failed"
        && task
            .error
            .as_deref()
            .or(task.result.as_deref())
            .is_some_and(crate::providers::is_provider_infra_error_text)
}

fn resolve_dispatch_project_scope(mission: &str, alias_roots: &[String]) -> Option<String> {
    let mut scopes = Vec::new();
    super::project_scope::extract_explicit_path_scopes_from_text(
        mission,
        &mut scopes,
        8,
        alias_roots,
    );
    scopes.sort();
    scopes.dedup();
    // Multiple distinct exact paths are an ambiguous authority boundary. Do
    // not parse nearby negation words to guess which one is writable, and do
    // not widen them to a common ancestor. A persisted typed workspace is
    // required for cross-project scheduled work.
    if scopes.len() > 1 {
        return None;
    }
    if let Some(scope) = scopes.pop() {
        return Some(scope);
    }
    // A public site URL is also durable target identity when its hostname is
    // the exact name of a local project root. Scheduled missions commonly
    // carry the URL but no filesystem path. Resolve that exact identity;
    // otherwise recovery can inherit an empty synthetic task directory even
    // though the authorized repository is locally available under the same
    // hostname.
    if scopes.is_empty() {
        for raw in mission.split_whitespace() {
            let candidate = raw.trim_matches(|ch: char| {
                ch.is_ascii_whitespace()
                    || matches!(
                        ch,
                        '`' | '\''
                            | '"'
                            | ','
                            | ';'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                    )
            });
            let Ok(url) = reqwest::Url::parse(candidate) else {
                continue;
            };
            if !matches!(url.scheme(), "http" | "https") {
                continue;
            }
            let Some(host) = url.host_str() else {
                continue;
            };
            let host = host.strip_prefix("www.").unwrap_or(host);
            if let Some(scope) =
                crate::tools::fs_utils::resolve_named_project_root(host, alias_roots)
            {
                super::project_scope::push_project_scope(
                    &mut scopes,
                    scope.to_string_lossy().to_string(),
                    8,
                );
            }
        }
    }
    scopes.into_iter().next()
}

fn sqlite_busy_code(value: &str) -> bool {
    value
        .parse::<i32>()
        .ok()
        .is_some_and(|code| code & 0xff == 5)
}

fn is_sqlite_busy_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(sqlx::Error::Database(database)) = cause.downcast_ref::<sqlx::Error>() {
            if database.code().is_some_and(|code| sqlite_busy_code(&code)) {
                return true;
            }
        }
        let message = cause.to_string().to_ascii_lowercase();
        message.contains("database is locked")
            || message.contains("database table is locked")
            || message.contains("sqlite_busy")
            || message.split("code:").skip(1).any(|suffix| {
                let code = suffix
                    .trim_start()
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>();
                sqlite_busy_code(&code)
            })
    })
}

async fn promote_dispatch_trigger_attempt(
    state: &Arc<dyn crate::traits::StateStore>,
    attempt: &crate::traits::TaskAttempt,
    worker_id: &str,
) -> anyhow::Result<bool> {
    const MAX_BUSY_RETRIES: usize = 5;
    let patch = crate::traits::TaskAttemptPatch {
        status: "running".to_string(),
        ..Default::default()
    };

    for busy_retry in 0..=MAX_BUSY_RETRIES {
        match state
            .bind_task_attempt_worker(
                &attempt.id,
                &attempt.lease_token,
                worker_id,
                Some("profile-task-lead"),
            )
            .await
        {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(error) if busy_retry < MAX_BUSY_RETRIES && is_sqlite_busy_error(&error) => {
                let delay_ms = 25_u64 << busy_retry;
                warn!(
                    attempt_id = %attempt.id,
                    retry = busy_retry + 1,
                    delay_ms,
                    "SQLite busy while binding task-lead trigger; retrying"
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                continue;
            }
            Err(error) => return Err(error),
        }

        match state
            .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
            .await
        {
            Ok(result) => return Ok(result),
            Err(error) if busy_retry < MAX_BUSY_RETRIES && is_sqlite_busy_error(&error) => {
                let delay_ms = 25_u64 << busy_retry;
                warn!(
                    attempt_id = %attempt.id,
                    retry = busy_retry + 1,
                    delay_ms,
                    "SQLite busy while starting task-lead trigger; retrying"
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            }
            Err(error) => return Err(error),
        }
    }

    unreachable!("bounded task-lead trigger retry loop must return")
}

fn render_recovery_task_ledger(tasks: &[crate::traits::Task]) -> String {
    tasks
        .iter()
        .take(16)
        .map(|task| {
            let mut entry = format!(
                "- [{}] {}",
                task.status,
                task.description.chars().take(240).collect::<String>()
            );
            if let Some(result) = task
                .result
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                entry.push_str(&format!(
                    "\n  Result: {}",
                    result.chars().take(1200).collect::<String>()
                ));
            }
            if let Some(error) = task
                .error
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                entry.push_str(&format!(
                    "\n  Error: {}",
                    error.chars().take(500).collect::<String>()
                ));
            }
            if let Some(blocker) = task
                .blocker
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                entry.push_str(&format!(
                    "\n  Blocker: {}",
                    blocker.chars().take(500).collect::<String>()
                ));
            }
            entry
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Return a heartbeat-owned trigger to the pending queue when another task
/// lead won the goal run guard. The claim was only an admission mechanism; it
/// must not strand the task as running after the duplicate spawn yields.
async fn release_duplicate_dispatch_trigger(
    state: &Arc<dyn crate::traits::StateStore>,
    task_id: &str,
    attempt: Option<&crate::traits::TaskAttempt>,
) {
    if let Some(attempt) = attempt {
        let patch = crate::traits::TaskAttemptPatch {
            status: "cancelled".to_string(),
            error: Some(
                "Duplicate heartbeat dispatch yielded to the active task lead.".to_string(),
            ),
            ..Default::default()
        };
        let mut busy_retry = 0_usize;
        loop {
            match state
                .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
                .await
            {
                Ok(true) => break,
                Ok(false) => return,
                Err(error) if busy_retry < 5 && is_sqlite_busy_error(&error) => {
                    let delay_ms = 25_u64 << busy_retry;
                    busy_retry += 1;
                    warn!(
                        attempt_id = %attempt.id,
                        retry = busy_retry,
                        delay_ms,
                        "SQLite busy while releasing task-lead trigger; retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                Err(error) => {
                    warn!(
                        attempt_id = %attempt.id,
                        %error,
                        "Failed to release task-lead trigger attempt"
                    );
                    return;
                }
            }
        }
    }

    if let Ok(Some(mut task)) = state.get_task(task_id).await {
        let heartbeat_owned = task
            .agent_id
            .as_deref()
            .is_some_and(|agent_id| agent_id.starts_with("heartbeat-dispatch-"));
        if task.status == "cancelled" || heartbeat_owned {
            task.status = "pending".to_string();
            task.agent_id = None;
            task.error = None;
            task.blocker = None;
            task.started_at = None;
            task.completed_at = None;
            let mut busy_retry = 0_usize;
            loop {
                match state.update_task(&task).await {
                    Ok(()) => break,
                    Err(error) if busy_retry < 5 && is_sqlite_busy_error(&error) => {
                        let delay_ms = 25_u64 << busy_retry;
                        busy_retry += 1;
                        warn!(
                            task_id = %task_id,
                            retry = busy_retry,
                            delay_ms,
                            "SQLite busy while re-queueing task-lead trigger; retrying"
                        );
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    Err(error) => {
                        warn!(
                            task_id = %task_id,
                            %error,
                            "Failed to re-queue task-lead trigger"
                        );
                        break;
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
struct AutoDispatchContext {
    agent: Arc<Agent>,
    state: Arc<dyn crate::traits::StateStore>,
    mission: String,
    goal_id: String,
    approval_session_id: String,
    channel_ctx: ChannelContext,
    user_role: UserRole,
    project_scope: Option<String>,
}

async fn execute_auto_dispatch_task(
    context: AutoDispatchContext,
    task: crate::traits::Task,
) -> Option<(crate::traits::Task, anyhow::Result<String>)> {
    let AutoDispatchContext {
        agent,
        state,
        mission,
        goal_id,
        approval_session_id,
        channel_ctx,
        user_role,
        project_scope,
    } = context;
    let specialist =
        Agent::select_specialist_kind(AgentRole::Executor, &mission, &task.description);
    let profile_id = format!("profile-{}", specialist.as_str().replace('_', "-"));
    let attempt = match state
        .claim_task_with_lease(
            &task.id,
            &format!("auto-dispatch-{goal_id}"),
            Some(&profile_id),
            180,
        )
        .await
    {
        Ok(Some(attempt)) => attempt,
        Ok(None) => return None,
        Err(error) => return Some((task, Err(error))),
    };

    if let Some(wait_secs) = parse_wait_task_seconds(&task.description) {
        info!(
            goal_id = %goal_id,
            task_id = %task.id,
            wait_secs,
            "Executing wait task locally"
        );
        let mut remaining = wait_secs;
        while remaining > 0 {
            let step = remaining.min(45);
            tokio::time::sleep(Duration::from_secs(step)).await;
            remaining = remaining.saturating_sub(step);
            match state
                .heartbeat_task_attempt(&attempt.id, &attempt.lease_token, 180)
                .await
            {
                Ok(true) => {}
                Ok(false) => {
                    return Some((
                        task,
                        Err(anyhow::anyhow!("Wait task lost its execution lease")),
                    ));
                }
                Err(error) => return Some((task, Err(error))),
            }
        }
        let summary = format!("Waited for {} second(s).", wait_secs);
        let handoff = crate::traits::TaskHandoff {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task.id.clone(),
            attempt_id: attempt.id.clone(),
            summary: summary.clone(),
            artifacts: Vec::new(),
            verification: vec!["Timer elapsed without cancellation.".to_string()],
            remaining_risk: None,
            next_step: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        let patch = crate::traits::TaskAttemptPatch {
            status: "completed".to_string(),
            result: Some(summary.clone()),
            handoff: Some(handoff),
            ..Default::default()
        };
        return match state
            .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
            .await
        {
            Ok(true) => Some((task, Ok(summary))),
            Ok(false) => Some((
                task,
                Err(anyhow::anyhow!("Wait task result was rejected as stale")),
            )),
            Err(error) => Some((task, Err(error))),
        };
    }

    let task_text = task.description.clone();
    let task_id = task.id.clone();
    let result = agent
        .spawn_child_with_outcome(
            &mission,
            &task_text,
            None,
            channel_ctx,
            user_role,
            Some(AgentRole::Executor),
            Some(&goal_id),
            Some(&task_id),
            project_scope.as_deref(),
            None,
            Some(&approval_session_id),
        )
        .await
        .map(|run| run.response);
    Some((task, result))
}

async fn deliver_auto_dispatch_result(
    agent: &Arc<Agent>,
    state: &Arc<dyn crate::traits::StateStore>,
    hub: Option<&Weak<dyn crate::runtime_ports::OutboundRouter>>,
    session_id: &str,
    task: &crate::traits::Task,
    result: &anyhow::Result<String>,
    owner_visible: bool,
) -> bool {
    if !owner_visible {
        return false;
    }
    let Ok(response) = result else {
        return false;
    };
    let persisted = state.get_task(&task.id).await.ok().flatten();
    let delivery_text = if !response.trim().is_empty() {
        response.clone()
    } else {
        persisted
            .and_then(|task| {
                task.result
                    .filter(|result| !result.trim().is_empty())
                    .or_else(|| task.blocker.filter(|blocker| !blocker.trim().is_empty()))
            })
            .unwrap_or_default()
    };
    if delivery_text.trim().is_empty() {
        return false;
    }
    match agent
        .deliver_parent_text_result(
            hub,
            session_id,
            &delivery_text,
            parent_delivery::ParentDeliveryKind::ExecutorResult,
        )
        .await
    {
        Ok(outcome) => outcome.sent,
        Err(error) => {
            warn!(
                session_id,
                task_id = %task.id,
                %error,
                "Failed to record parent-mediated executor result"
            );
            false
        }
    }
}

async fn tasks_for_current_run(
    state: &Arc<dyn crate::traits::StateStore>,
    goal_id: &str,
    goal_run_id: Option<&str>,
) -> Vec<crate::traits::Task> {
    match goal_run_id {
        Some(run_id) => state
            .get_tasks_for_goal_run(run_id)
            .await
            .unwrap_or_default(),
        None => state.get_tasks_for_goal(goal_id).await.unwrap_or_default(),
    }
}

async fn work_tasks_for_current_run(
    state: &Arc<dyn crate::traits::StateStore>,
    goal_id: &str,
    goal_run_id: Option<&str>,
    dispatch_trigger_task_id: Option<&str>,
) -> Vec<crate::traits::Task> {
    tasks_for_current_run(state, goal_id, goal_run_id)
        .await
        .into_iter()
        .filter(|task| Some(task.id.as_str()) != dispatch_trigger_task_id)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecurringRunTerminalOutcome {
    Completed,
    Failed,
}

fn recurring_run_terminal_outcome(run_status: Option<&str>) -> Option<RecurringRunTerminalOutcome> {
    match run_status {
        Some("completed") => Some(RecurringRunTerminalOutcome::Completed),
        Some("failed" | "blocked") => Some(RecurringRunTerminalOutcome::Failed),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchTriggerDisposition {
    Completed,
    Continuable,
    Failed,
}

fn dispatch_trigger_disposition(
    task_lead_succeeded: bool,
    work_tasks: &[crate::traits::Task],
) -> DispatchTriggerDisposition {
    if work_tasks.is_empty() {
        return if task_lead_succeeded {
            DispatchTriggerDisposition::Completed
        } else {
            DispatchTriggerDisposition::Failed
        };
    }
    if work_tasks.iter().any(|task| {
        matches!(
            task.status.as_str(),
            "failed" | "blocked" | "interrupted" | "cancelled"
        ) || task
            .error
            .as_deref()
            .is_some_and(|error| !error.trim().is_empty())
            || task
                .blocker
                .as_deref()
                .is_some_and(|blocker| !blocker.trim().is_empty())
    }) {
        return DispatchTriggerDisposition::Failed;
    }
    if work_tasks
        .iter()
        .all(crate::traits::Task::satisfies_run_completion)
    {
        DispatchTriggerDisposition::Completed
    } else {
        // The task lead has durably handed off unfinished work. Its own turn is
        // complete, while the enclosing goal run stays open for continuation.
        DispatchTriggerDisposition::Continuable
    }
}

fn goal_run_terminal_status(
    goal_status: &str,
    trigger_status: Option<&str>,
    work_tasks: &[crate::traits::Task],
) -> Option<&'static str> {
    if work_tasks.iter().any(|task| task.status == "blocked") {
        Some("blocked")
    } else if matches!(goal_status, "failed" | "stalled" | "cancelled")
        || trigger_status == Some("failed")
        || work_tasks
            .iter()
            .any(|task| matches!(task.status.as_str(), "failed" | "interrupted" | "cancelled"))
    {
        Some("failed")
    } else if (goal_status == "completed" || trigger_status == Some("completed"))
        && work_tasks
            .iter()
            .all(crate::traits::Task::satisfies_run_completion)
    {
        Some("completed")
    } else {
        None
    }
}

fn task_lead_semantically_succeeded(
    task_lead_outcome: Option<TaskOutcome>,
    is_mandate_run: bool,
    mandate_decision_ready: bool,
) -> bool {
    task_lead_outcome == Some(TaskOutcome::Succeeded) || (is_mandate_run && mandate_decision_ready)
}

async fn patch_task_attempt_with_transient_retry(
    state: &Arc<dyn crate::traits::StateStore>,
    attempt: &crate::traits::TaskAttempt,
    patch: &crate::traits::TaskAttemptPatch,
) -> anyhow::Result<bool> {
    const MAX_ATTEMPTS: usize = 3;
    let mut delay_ms = 25u64;
    for try_number in 1..=MAX_ATTEMPTS {
        match state
            .patch_task_from_attempt(&attempt.id, &attempt.lease_token, patch)
            .await
        {
            Ok(applied) => return Ok(applied),
            Err(error) if try_number < MAX_ATTEMPTS => {
                warn!(
                    task_id = %attempt.task_id,
                    attempt_id = %attempt.id,
                    try_number,
                    %error,
                    "Transient task-attempt finalization error; retrying"
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                delay_ms *= 2;
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("bounded retry loop always returns")
}

fn clear_partial_success_context(goal: &mut crate::traits::Goal) {
    let Some(raw) = goal.context.as_deref() else {
        return;
    };
    let Ok(mut context) = serde_json::from_str::<serde_json::Value>(raw) else {
        return;
    };
    let Some(object) = context.as_object_mut() else {
        return;
    };
    for key in ["partial_success", "completed", "failed", "blocked", "total"] {
        object.remove(key);
    }
    goal.context = (!object.is_empty()).then(|| context.to_string());
}

fn recovery_task_succeeded(outcome: TaskOutcome, persisted: Option<&crate::traits::Task>) -> bool {
    outcome == TaskOutcome::Succeeded
        && persisted.is_some_and(crate::traits::Task::completed_successfully)
}

fn is_terminal_recovery_task(task: &crate::traits::Task) -> bool {
    task.context
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|context| {
            context
                .get("terminal_recovery")
                .and_then(|value| value.as_bool())
        })
        .unwrap_or(false)
}

async fn rebind_scheduled_recovery_state(
    state: &Arc<dyn StateStore>,
    registry: Option<&crate::goal_tokens::GoalTokenRegistry>,
    goal_id: &str,
    recovery_task_id: &str,
) {
    if let Some(registry) = registry {
        if let Some(status) = registry.get_run_budget(goal_id).await {
            persist_scheduled_run_state(state, goal_id, Some(recovery_task_id), &status).await;
            return;
        }
    }

    // A restart or terminal transition can clear the in-memory registry before
    // recovery is created. Preserve the persisted counters and health while
    // rebinding their root (and, via the store upsert, goal_run_id) to the
    // recovery task's actual run.
    if let Ok(Some(mut record)) = state.get_scheduled_run_state(goal_id).await {
        record.root_task_id = recovery_task_id.to_string();
        record.updated_at = chrono::Utc::now().to_rfc3339();
        let _ = state.upsert_scheduled_run_state(&record).await;
    }
}

fn terminal_recovery_eligible(
    goal_type: &str,
    goal_status: &str,
    run_status: Option<&str>,
    already_notified: bool,
    tasks: &[crate::traits::Task],
) -> bool {
    if tasks.iter().any(is_terminal_recovery_task) {
        return false;
    }
    match goal_type {
        "finite" => !already_notified && matches!(goal_status, "failed" | "stalled"),
        "continuous" => matches!(run_status, Some("failed" | "blocked")),
        _ => false,
    }
}

fn failed_run_summary(tasks: &[crate::traits::Task], fallback: Option<&str>) -> String {
    let failures = tasks
        .iter()
        .filter(|task| !task.satisfies_run_completion())
        .take(3)
        .map(|task| {
            let detail = task
                .error
                .as_deref()
                .filter(|error| !error.trim().is_empty())
                .or_else(|| {
                    task.blocker
                        .as_deref()
                        .filter(|blocker| !blocker.trim().is_empty())
                })
                .unwrap_or(task.status.as_str());
            format!("{}: {}", task.description, detail)
        })
        .collect::<Vec<_>>();
    if !failures.is_empty() {
        return failures.join("\n");
    }
    fallback
        .filter(|error| !error.trim().is_empty())
        .unwrap_or("The run ended before every required task completed.")
        .to_string()
}

async fn keep_mandate_controller_open(
    state: &Arc<dyn StateStore>,
    mandate: &Mandate,
    goal: &crate::traits::Goal,
) {
    // Do not overwrite an owner pause/cancel that raced this review. Runtime
    // repair is allowed only while the mandate itself still says `active`.
    if mandate.status != MandateStatus::Active {
        return;
    }
    if matches!(goal.status.as_str(), "active" | "pending") && goal.dispatch_failures == 0 {
        return;
    }
    // The state store checks the current mandate status and exact authority
    // epoch in the same SQL statement that repairs the goal. A pause/cancel or
    // policy update racing this stale finalizer snapshot therefore wins.
    if let Err(error) = state
        .keep_mandate_controller_active(&mandate.id, mandate.version)
        .await
    {
        warn!(
            mandate_id = %mandate.id,
            goal_id = %goal.id,
            %error,
            "Failed to keep mandate controller goal active"
        );
    }
}

/// A completed deliberator turn without `record_decision` has granted no
/// authority and created no intention. Persist the least-privilege semantic
/// result (WAIT). Finalization recognizes the exact runtime-authored rationale
/// and records a retriable review failure, so the fallback can never authorize
/// a mutation or masquerade as a successful deliberation.
async fn persist_safe_wait_if_decision_missing(
    state: &Arc<dyn StateStore>,
    goal_id: &str,
    goal_run_id: &str,
) -> bool {
    let existing = match state.get_mandate_decision_for_run(goal_run_id).await {
        Ok(existing) => existing,
        Err(error) => {
            warn!(goal_id, run_id = goal_run_id, %error, "Could not inspect mandate decision");
            return false;
        }
    };
    let mandate = match state.get_mandate_for_goal(goal_id).await {
        Ok(Some(mandate)) if mandate.is_active() => mandate,
        Ok(_) => return false,
        Err(error) => {
            warn!(goal_id, run_id = goal_run_id, %error, "Could not load mandate for safe WAIT");
            return false;
        }
    };
    if let Some(decision) = existing {
        return decision.mandate_id == mandate.id && decision.mandate_version == mandate.version;
    }
    let mut decision = crate::traits::MandateDecisionCycle::new(
        &mandate.id,
        goal_run_id,
        MandateDecisionOutcome::Wait,
        SAFE_FALLBACK_WAIT_RATIONALE,
        mandate.version,
    );
    decision.reconsider_at = Some(mandate.bounded_next_review_at(None, chrono::Utc::now()));
    if let Err(error) = state.record_mandate_decision(&decision, None, None).await {
        // A concurrent exact decision wins. Reload once before reporting that
        // the safe fallback failed.
        let committed = state
            .get_mandate_decision_for_run(goal_run_id)
            .await
            .ok()
            .flatten()
            .is_some_and(|decision| {
                decision.mandate_id == mandate.id && decision.mandate_version == mandate.version
            });
        if !committed {
            warn!(goal_id, run_id = goal_run_id, %error, "Could not persist safe mandate WAIT");
        }
        return committed;
    }
    warn!(
        goal_id,
        run_id = goal_run_id,
        "Mandate deliberator omitted its decision; persisted retriable safe-WAIT failure marker"
    );
    true
}

/// Finalize one mandate deliberation independently from scheduled-run state.
///
/// Any returned notice was already committed by the state store in the same
/// transaction as the terminal proof state. The caller may attempt prompt
/// delivery, but must never enqueue a second, post-commit copy.
async fn finalize_mandate_review(
    state: &Arc<dyn StateStore>,
    goal: &crate::traits::Goal,
    goal_run_id: &str,
    _trigger_status: Option<&str>,
    _run_tasks: &[crate::traits::Task],
    _fallback_error: Option<&str>,
    _executor_results_already_sent: bool,
) -> Option<MandateRunNotification> {
    let mandate = match state.get_mandate_for_goal(&goal.id).await {
        Ok(Some(mandate)) => mandate,
        Ok(None) => {
            warn!(goal_id = %goal.id, run_id = %goal_run_id, "Mandate run has no mandate record");
            return None;
        }
        Err(error) => {
            warn!(goal_id = %goal.id, run_id = %goal_run_id, %error, "Failed to load mandate during finalization");
            return None;
        }
    };

    let finalized_at = chrono::Utc::now().to_rfc3339();
    let proof = match state
        .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
            mandate_id: mandate.id.clone(),
            expected_mandate_version: mandate.version,
            goal_run_id: goal_run_id.to_string(),
            finalized_at: finalized_at.clone(),
        })
        .await
    {
        Ok(proof) => proof,
        Err(error) => {
            warn!(
                mandate_id = %mandate.id,
                run_id = %goal_run_id,
                %error,
                "Atomic mandate proof finalization failed"
            );
            return None;
        }
    };

    let notice = |kind, counts| {
        MandateRunNotification::new(
            &mandate.id,
            mandate.version,
            &goal.id,
            goal_run_id,
            &goal.session_id,
            kind,
            counts,
            &finalized_at,
        )
    };
    match proof {
        MandateRunFinalizationResult::ActSatisfied { counts } => {
            keep_mandate_controller_open(state, &mandate, goal).await;
            Some(notice(MandateRunNotificationKind::ActSatisfied, counts))
        }
        MandateRunFinalizationResult::NonActionSatisfied { outcome, counts } => match outcome {
            MandateDecisionOutcome::Wait => {
                keep_mandate_controller_open(state, &mandate, goal).await;
                None
            }
            MandateDecisionOutcome::Ask => Some(notice(MandateRunNotificationKind::Ask, counts)),
            MandateDecisionOutcome::Stop => {
                Some(notice(MandateRunNotificationKind::Stopped, counts))
            }
            MandateDecisionOutcome::Act => {
                warn!(
                    mandate_id = %mandate.id,
                    run_id = %goal_run_id,
                    "State store returned ACT as a non-action proof"
                );
                None
            }
        },
        MandateRunFinalizationResult::ReconciliationRequired { reason, counts } => Some(notice(
            MandateRunNotificationKind::ReconciliationRequired { reason },
            counts,
        )),
        MandateRunFinalizationResult::Stale { reason } => {
            info!(
                mandate_id = %mandate.id,
                run_id = %goal_run_id,
                ?reason,
                "Mandate review finalization was already stale"
            );
            None
        }
        MandateRunFinalizationResult::Rejected { reason } => match reason {
            MandateFinalizationRejectReason::InvalidRequest => {
                warn!(
                    mandate_id = %mandate.id,
                    run_id = %goal_run_id,
                    "State store rejected an internally constructed mandate finalization request"
                );
                None
            }
            MandateFinalizationRejectReason::DecisionMissing
            | MandateFinalizationRejectReason::InvalidDecisionState
            | MandateFinalizationRejectReason::DeliberatorFailed => Some(notice(
                MandateRunNotificationKind::ReviewFailed { reason },
                Default::default(),
            )),
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinuousDispatchOutcome {
    Progress,
    TransientInfrastructureFailure,
    ErrorWithoutProgress,
    Blocked,
    NoProgress,
}

fn classify_continuous_dispatch_outcome(
    dispatch_made_progress: bool,
    task_lead_errored: bool,
    transient_infrastructure_failure: bool,
    all_remaining_blocked: bool,
) -> ContinuousDispatchOutcome {
    if dispatch_made_progress {
        ContinuousDispatchOutcome::Progress
    } else if transient_infrastructure_failure {
        ContinuousDispatchOutcome::TransientInfrastructureFailure
    } else if task_lead_errored {
        ContinuousDispatchOutcome::ErrorWithoutProgress
    } else if all_remaining_blocked {
        ContinuousDispatchOutcome::Blocked
    } else {
        ContinuousDispatchOutcome::NoProgress
    }
}

fn run_completion_should_close_goal(goal_type: &str, all_done: bool) -> bool {
    all_done && goal_type == "finite"
}

/// Emit a progress update onto a SINGLE self-editing surface: the first update
/// creates a tracked message; subsequent updates edit it in place. Collapses the
/// heartbeat into one updating message instead of a stream of pings. Falls back
/// to a fresh tracked send if the channel can't edit (e.g. the message aged out).
async fn emit_progress_surface(
    hub: &Option<Weak<dyn crate::runtime_ports::OutboundRouter>>,
    session: &str,
    surface_id: &mut Option<String>,
    text: &str,
) {
    let local_hour = chrono::Timelike::hour(&chrono::Local::now());
    if !crate::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(local_hour) {
        return;
    }
    let Some(hub_arc) = hub.as_ref().and_then(|w| w.upgrade()) else {
        return;
    };
    if let Some(id) = surface_id.clone() {
        match hub_arc.edit_text(session, &id, text).await {
            Ok(true) => return,
            // Couldn't edit (no editable surface) — drop the id and post fresh.
            _ => *surface_id = None,
        }
    }
    if let Ok(Some(new_id)) = hub_arc.send_text_tracked(session, text).await {
        *surface_id = Some(new_id);
    }
}

/// Spawn a task lead in the background (free function to satisfy Send requirements).
/// This runs `spawn_child` on the given agent with TaskLead role, then updates
/// the goal and notifies the user when complete.
#[allow(clippy::too_many_arguments)]
pub fn spawn_background_task_lead(
    agent: Arc<Agent>,
    goal: crate::traits::Goal,
    user_text: String,
    session_id: String,
    channel_ctx: ChannelContext,
    user_role: UserRole,
    state: Arc<dyn crate::traits::StateStore>,
    hub: Option<Weak<dyn crate::runtime_ports::OutboundRouter>>,
    goal_token_registry: Option<crate::goal_tokens::GoalTokenRegistry>,
    dispatch_trigger_task_id: Option<String>,
    // When the caller already posted a tracked "starting" message (e.g. the
    // scheduled-run "🔄 Running scheduled task" announcement), pass its id here so
    // the progress heartbeat edits THAT message in place instead of posting a new
    // one — folding the announcement and progress into one self-updating surface.
    initial_surface_id: Option<String>,
) {
    tokio::spawn(async move {
        // Teardown handles for the self-correction bridge (3c P3b.3). The body
        // below has many early `return`s, so the existing logic is wrapped in an
        // inner future; whichever way it exits, we clear any correction-execution
        // context registered for this goal id afterward. For non-remediation
        // goals this is a cheap no-op (the key was never registered); for
        // dispatched remediations it tears the context down on completion
        // (success OR error) instead of leaking until bounded FIFO eviction.
        let teardown_agent = agent.clone();
        let teardown_goal_id = goal.id.clone();
        let teardown_state = state.clone();
        let teardown_goal_token_registry = goal_token_registry.clone();
        let finalize_scheduled_run = Arc::new(AtomicBool::new(false));
        let body_finalize_scheduled_run = finalize_scheduled_run.clone();

        let body = async move {
            let goal_id = goal.id.clone();
            let mission = goal.description.clone();
            let initial_goal = goal.clone();
            let goal_run = state.get_current_goal_run(&goal_id).await.ok().flatten();
            let goal_run_id = goal_run.as_ref().map(|run| run.id.clone());
            let is_mandate_run = goal_run
                .as_ref()
                .is_some_and(|run| run.trigger_type == "mandate");
            let dispatch_attempt =
                if let Some(trigger_task_id) = dispatch_trigger_task_id.as_deref() {
                    state
                        .get_current_task_attempt(trigger_task_id)
                        .await
                        .ok()
                        .flatten()
                } else {
                    None
                };

            // Take the per-goal run guard before binding the heartbeat trigger.
            // If another lead is already active, release the admission claim
            // immediately instead of leaving a phantom running task behind.
            let _run_guard = if let Some(ref registry) = goal_token_registry {
                match registry.try_acquire_run(&goal_id) {
                    Some(guard) => Some(guard),
                    None => {
                        info!(
                            goal_id = %goal_id,
                            session_id = %session_id,
                            "Goal already has an active task lead; skipping duplicate background spawn"
                        );
                        if let Some(trigger_task_id) = dispatch_trigger_task_id.as_deref() {
                            release_duplicate_dispatch_trigger(
                                &state,
                                trigger_task_id,
                                dispatch_attempt.as_ref(),
                            )
                            .await;
                        }
                        return;
                    }
                }
            } else {
                None
            };
            // Clone channel_ctx and user_role for potential direct fallback and auto-dispatch
            let fallback_channel_ctx = channel_ctx.clone();
            let dispatch_channel_ctx = channel_ctx.clone();
            let fallback_user_role = user_role;
            let dispatch_project_scope =
                resolve_dispatch_project_scope(&mission, &agent.path_aliases.projects);

            // Heartbeat dispatch claims a "trigger" task before spawning this background
            // lead. Keep it in "running" state (not "pending") so dispatch_pending_tasks
            // won't re-dispatch it on the next tick. The task lead will process it through
            // its normal flow (manage_goal_tasks / auto-dispatch).
            //
            // Previously this released the claim back to "pending", which created a race:
            // the next heartbeat tick would see the task as orphaned-pending and re-dispatch
            // it, causing duplicate execution (e.g., double tweet posts).
            if let Some(ref trigger_task_id) = dispatch_trigger_task_id {
                if let Some(attempt) = dispatch_attempt.as_ref() {
                    let worker_id = format!("task-lead-{}", goal_id);
                    match promote_dispatch_trigger_attempt(&state, attempt, &worker_id).await {
                        Ok(true) => {}
                        Ok(false) => {
                            warn!(
                                task_id = %trigger_task_id,
                                attempt_id = %attempt.id,
                                "Task-lead trigger lease was lost before execution"
                            );
                            return;
                        }
                        Err(error) => {
                            warn!(
                                task_id = %trigger_task_id,
                                attempt_id = %attempt.id,
                                %error,
                                "Task-lead trigger transition failed; releasing it for retry"
                            );
                            release_duplicate_dispatch_trigger(
                                &state,
                                trigger_task_id,
                                Some(attempt),
                            )
                            .await;
                            return;
                        }
                    }
                } else {
                    match state.get_task(trigger_task_id).await {
                        Ok(Some(task))
                            if (task.status == "claimed" || task.status == "running")
                                && task
                                    .agent_id
                                    .as_deref()
                                    .is_some_and(|aid| aid.starts_with("heartbeat-dispatch-")) =>
                        {
                            let mut updated = task.clone();
                            updated.status = "running".to_string();
                            updated.agent_id = Some(format!("task-lead-{}", goal_id));
                            // Keep started_at from the claim so dispatch sees it as active
                            if let Err(e) = state.update_task(&updated).await {
                                warn!(
                                    task_id = %trigger_task_id,
                                    goal_id = %goal_id,
                                    error = %e,
                                    "Failed to update dispatch trigger task to running"
                                );
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            warn!(
                                task_id = %trigger_task_id,
                                goal_id = %goal_id,
                                error = %e,
                                "Failed to load dispatch trigger task"
                            );
                        }
                    }
                }
            }

            // Snapshot genuine completed work before this dispatch. The scheduled
            // trigger task is bookkeeping and is finalized below, so it must not
            // count as progress by itself.
            let initial_run_tasks =
                tasks_for_current_run(&state, &goal_id, goal_run_id.as_deref()).await;
            let completed_work_before = count_successful_completed_work(
                &initial_run_tasks,
                dispatch_trigger_task_id.as_deref(),
            );

            // This background lead owns the complete scheduled cycle: planning,
            // child execution, trigger finalization, and terminal notification.
            // Nested Agent turns must not clear the shared budget when they
            // return, because later siblings still belong to this same cycle.
            let is_scheduled_run = dispatch_trigger_task_id.is_some()
                && goal_has_scheduled_provenance(
                    &state,
                    &goal_id,
                    dispatch_trigger_task_id.as_deref(),
                )
                .await;
            if is_scheduled_run {
                body_finalize_scheduled_run.store(true, Ordering::Release);
            }

            // The watchdog considers recent task activity, not merely a live
            // Tokio future. Record a lightweight heartbeat for the scheduled
            // root while its task lead is running so a valid long LLM/tool call
            // is not interrupted and then mistaken for retryable work.
            let (keepalive_cancel_tx, mut keepalive_cancel_rx) =
                tokio::sync::oneshot::channel::<()>();
            let keepalive_state = state.clone();
            let keepalive_task_id = dispatch_trigger_task_id.clone();
            let keepalive_attempt = dispatch_attempt.clone();
            let task_lead_lease_lost = Arc::new(AtomicBool::new(false));
            let keepalive_lease_lost = task_lead_lease_lost.clone();
            let keepalive_handle = tokio::spawn(async move {
                let Some(task_id) = keepalive_task_id else {
                    let _ = keepalive_cancel_rx.await;
                    return;
                };
                let mut keepalive_interval = tokio::time::interval_at(
                    tokio::time::Instant::now() + Duration::from_secs(45),
                    Duration::from_secs(45),
                );
                keepalive_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    tokio::select! {
                        _ = keepalive_interval.tick() => {
                            if let Some(attempt) = keepalive_attempt.as_ref() {
                                let renewed = match keepalive_state
                                    .heartbeat_task_attempt(
                                        &attempt.id,
                                        &attempt.lease_token,
                                        180,
                                    )
                                    .await
                                {
                                    Ok(renewed) => renewed,
                                    Err(error) => {
                                        // A transient pool/SQLite failure is not
                                        // evidence that ownership was revoked.
                                        // Keep trying; an actual expired or
                                        // superseded lease is reported as Ok(false)
                                        // by heartbeat_task_attempt and remains a
                                        // hard stop.
                                        warn!(
                                            task_id = %task_id,
                                            attempt_id = %attempt.id,
                                            %error,
                                            "Task-lead keepalive could not renew; will retry"
                                        );
                                        continue;
                                    }
                                };
                                if !renewed {
                                    keepalive_lease_lost.store(true, Ordering::Release);
                                    warn!(
                                        task_id = %task_id,
                                        attempt_id = %attempt.id,
                                        "Task-lead keepalive lost its execution lease"
                                    );
                                    break;
                                }
                            }
                            let activity = crate::traits::TaskActivity {
                                id: 0,
                                task_id: task_id.clone(),
                                activity_type: "status_change".to_string(),
                                tool_name: None,
                                tool_args: None,
                                result: Some("task lead keepalive".to_string()),
                                success: None,
                                tokens_used: None,
                                created_at: chrono::Utc::now().to_rfc3339(),
                            };
                            if let Err(error) = keepalive_state.log_task_activity(&activity).await {
                                warn!(task_id = %task_id, %error, "Failed to record task-lead keepalive");
                            }
                        }
                        _ = &mut keepalive_cancel_rx => break,
                    }
                }
            });

            // Progress heartbeat: send periodic status updates while the task lead works.
            // This prevents the "goal appears abandoned" UX problem where the user sees
            // nothing between "On it." and the final notification.
            // Only send progress updates to DM sessions — group channels already have the
            // "Running scheduled task" notification and the final result. Progress updates
            // every 30s are too noisy for shared channels.
            let is_group_channel = is_group_session(&session_id);
            let heartbeat_hub = hub.clone();
            let heartbeat_session = session_id.clone();
            let heartbeat_state = state.clone();
            let heartbeat_goal_id = goal_id.clone();
            let heartbeat_goal_run_id = goal_run_id.clone();
            let heartbeat_trigger_task_id = dispatch_trigger_task_id.clone();
            let heartbeat_initial_surface = initial_surface_id;
            let suppress_mandate_progress = is_mandate_run;
            let (heartbeat_cancel_tx, mut heartbeat_cancel_rx) =
                tokio::sync::oneshot::channel::<()>();
            let heartbeat_handle = tokio::spawn(async move {
                if is_group_channel || suppress_mandate_progress {
                    // Shared channels avoid progress spam. Mandate runs also
                    // suppress it because generated task descriptions and
                    // external-derived state are not control-plane-safe owner
                    // notifications; only the static finalizer may surface a
                    // mandate cycle outcome.
                    let _ = heartbeat_cancel_rx.await;
                    return heartbeat_initial_surface;
                }
                let mut interval_count = 0u32;
                let mut last_progress_key: Option<String> = None;
                let mut planning_msg_count = 0u32;
                // One self-editing surface for the whole run's progress stream —
                // seeded with the caller's "starting" message when present, so that
                // message morphs into the progress updates instead of a new one.
                let mut surface_id: Option<String> = heartbeat_initial_surface;
                loop {
                    // Backoff schedule: 15s, 30s, 1m, 2m, 5m, 10m, then every 15m.
                    // Long-running goals keep emitting (no message cap) — the
                    // growing interval is what prevents spam.
                    let wait_secs = heartbeat_wait_secs(interval_count);
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs(wait_secs)) => {},
                        _ = &mut heartbeat_cancel_rx => break,
                    }
                    interval_count += 1;

                    // Build progress message from task statuses
                    let tasks = work_tasks_for_current_run(
                        &heartbeat_state,
                        &heartbeat_goal_id,
                        heartbeat_goal_run_id.as_deref(),
                        heartbeat_trigger_task_id.as_deref(),
                    )
                    .await;
                    if tasks.is_empty() {
                        // Tasks not yet created — send one planning message on the
                        // first empty-tasks heartbeat. If planning is still running
                        // by the 1-minute tick, resume pinging on the backoff
                        // schedule so a hung planning phase is never silent.
                        planning_msg_count += 1;
                        if planning_msg_count == 1 || interval_count >= 3 {
                            emit_progress_surface(
                                &heartbeat_hub,
                                &heartbeat_session,
                                &mut surface_id,
                                "⏳ **Scheduled run in progress**\n\nPlanning the steps…",
                            )
                            .await;
                        }
                    } else {
                        // Count genuinely completed tasks (exclude cancelled ones with errors)
                        let completed = tasks.iter().filter(|t| t.completed_successfully()).count();
                        let started = tasks.iter().filter(|t| t.status != "pending").count();
                        let total = tasks.len();
                        let in_progress: Vec<String> = tasks
                            .iter()
                            .filter(|t| t.status == "claimed" || t.status == "running")
                            // Skip the parent "Scheduled check: <goal>" task — its
                            // description is the full (often huge) goal text, which
                            // is internal noise, not a user-facing step.
                            .filter(|t| !is_goal_run_root_task_description(&t.description))
                            .take(2)
                            // Keep each step label short so the progress line stays a
                            // glanceable one-liner, never a wall of text.
                            .map(|t| {
                                truncate_goal_result_text(
                                    &user_facing_task_description(&t.description),
                                    80,
                                )
                            })
                            .collect();
                        let progress_msg = if total == 1 {
                            // Single-step goals: step-count jargon ("0/1 steps
                            // completed, 1 in progress") reads as internal state.
                            // Give the first interval a chance to finish silently,
                            // then send a plain humane update.
                            if interval_count < 2 {
                                continue;
                            }
                            "⏳ **Scheduled run in progress**\n\nWorking…".to_string()
                        } else if !in_progress.is_empty() {
                            format!(
                                "⏳ **Scheduled run in progress**\n\n{} of {} steps complete\n\n_Currently: {}_",
                                completed,
                                total,
                                in_progress.join(", ")
                            )
                        } else if completed == total || started > completed {
                            format!(
                                "⏳ **Scheduled run in progress**\n\n{} of {} steps complete",
                                completed, total
                            )
                        } else {
                            "⏳ **Scheduled run in progress**\n\nWorking…".to_string()
                        };

                        // Dedup key uses only completed|total so we don't spam when
                        // sub-tasks change status without any step actually completing.
                        // The early fast ticks (15s/30s apart) only report actual
                        // progress; from the 1-minute tick onward every tick emits,
                        // so long-running goals are never silent.
                        let progress_key = format!("{}|{}", completed, total);
                        let state_changed =
                            last_progress_key.as_deref() != Some(progress_key.as_str());
                        if state_changed {
                            last_progress_key = Some(progress_key);
                        }
                        if !state_changed && interval_count < 3 {
                            continue;
                        }
                        emit_progress_surface(
                            &heartbeat_hub,
                            &heartbeat_session,
                            &mut surface_id,
                            &progress_msg,
                        )
                        .await;
                    }
                }
                surface_id
            });

            // Intercept pure wait/sleep goals to avoid spawning a full LLM task lead
            // just to orchestrate a timer.  For compound goals ("wait 2 minutes then
            // check disk space") we sleep first and then let the task lead handle the
            // remainder — but only if there actually IS a remainder after the wait.
            let effective_mission;
            let effective_user_text;
            if !is_mandate_run {
                if let Some(wait_secs) = parse_goal_leading_wait(&mission) {
                    let remainder = strip_leading_wait(&mission);
                    info!(
                        goal_id = %goal_id,
                        wait_secs,
                        has_remainder = !remainder.is_empty(),
                        "Intercepted wait prefix in goal — sleeping locally"
                    );
                    tokio::time::sleep(Duration::from_secs(wait_secs)).await;

                    if remainder.is_empty() {
                        // Pure wait goal with nothing after — mark complete, skip LLM entirely.
                        let _ = heartbeat_cancel_tx.send(());
                        let terminal_surface_id = heartbeat_handle.await.ok().flatten();
                        let _ = keepalive_cancel_tx.send(());
                        let _ = keepalive_handle.await;
                        let now = chrono::Utc::now().to_rfc3339();
                        let msg = format!("Waited for {} second(s).", wait_secs);
                        if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                            if g.status == "active" || g.status == "pending" {
                                g.status = "completed".to_string();
                                g.completed_at = Some(now.clone());
                                g.updated_at = now.clone();
                                let _ = state.update_goal(&g).await;
                            }
                        }

                        // Finalize any non-terminal tasks so we don't leave pending rows behind
                        // after a local pure-wait short-circuit.
                        for task in
                            tasks_for_current_run(&state, &goal_id, goal_run_id.as_deref()).await
                        {
                            if task.status != "completed"
                                && task.status != "failed"
                                && task.status != "cancelled"
                            {
                                if Some(task.id.as_str()) == dispatch_trigger_task_id.as_deref() {
                                    if let Some(attempt) = dispatch_attempt.as_ref() {
                                        let handoff = crate::traits::TaskHandoff {
                                            id: uuid::Uuid::new_v4().to_string(),
                                            task_id: task.id.clone(),
                                            attempt_id: attempt.id.clone(),
                                            summary: msg.clone(),
                                            artifacts: Vec::new(),
                                            verification: vec![
                                                "Timer elapsed without cancellation.".to_string(),
                                            ],
                                            remaining_risk: None,
                                            next_step: None,
                                            created_at: now.clone(),
                                        };
                                        let patch = crate::traits::TaskAttemptPatch {
                                            status: "completed".to_string(),
                                            result: Some(msg.clone()),
                                            handoff: Some(handoff),
                                            ..Default::default()
                                        };
                                        let _ = state
                                            .patch_task_from_attempt(
                                                &attempt.id,
                                                &attempt.lease_token,
                                                &patch,
                                            )
                                            .await;
                                        continue;
                                    }
                                }
                                let mut updated = task.clone();
                                updated.status = "completed".to_string();
                                updated.error = None;
                                updated.result = Some(msg.clone());
                                updated.completed_at = Some(now.clone());
                                let _ = state.update_task(&updated).await;
                            }
                        }
                        if let Some(run_id) = goal_run_id.as_deref() {
                            let _ = state.finish_goal_run(run_id, "completed", Some(&msg)).await;
                        }

                        if let Err(err) = agent
                            .deliver_parent_text_result_to_surface(
                                hub.as_ref(),
                                &session_id,
                                terminal_surface_id.as_deref(),
                                &msg,
                                parent_delivery::ParentDeliveryKind::WaitResult,
                            )
                            .await
                        {
                            warn!(
                                session_id = %session_id,
                                error = %err,
                                "Failed to record parent-mediated wait result"
                            );
                        }
                        return;
                    }
                    effective_mission = remainder.clone();
                    effective_user_text = remainder;
                } else {
                    effective_mission = mission.clone();
                    effective_user_text = user_text.clone();
                }
            } else {
                effective_mission = mission.clone();
                effective_user_text = user_text.clone();
            }

            let task_lead_run = agent
                .spawn_child_with_outcome_and_attempt(
                    &effective_mission,
                    &effective_user_text,
                    None,
                    channel_ctx,
                    fallback_user_role,
                    Some(AgentRole::TaskLead),
                    Some(goal_id.as_str()),
                    dispatch_trigger_task_id.as_deref(),
                    None,
                    None, // arg_specialist (task lead spawn — not LLM-tool-selectable)
                    Some(&session_id),
                    dispatch_attempt.clone(),
                )
                .await;

            // A child can finish after its keepalive observed a revoked or
            // expired lease. Revalidate the exact attempt before consuming its
            // response or dispatching any tasks it planned. The conditional
            // heartbeat is both a renewal and an authoritative ownership read.
            if let Some(attempt) = dispatch_attempt.as_ref() {
                let mut lease_live = !task_lead_lease_lost.load(Ordering::Acquire);
                if lease_live {
                    // A final read error is not enough to discard a completed
                    // task-lead result. Retry briefly so transient SQLite/pool
                    // contention cannot strand the run with an expiring root
                    // lease; Ok(false) remains the authoritative loss signal.
                    lease_live = false;
                    for retry in 0..3 {
                        match state
                            .heartbeat_task_attempt(&attempt.id, &attempt.lease_token, 180)
                            .await
                        {
                            Ok(live) => {
                                lease_live = live;
                                break;
                            }
                            Err(error) => {
                                warn!(
                                    goal_id = %goal_id,
                                    attempt_id = %attempt.id,
                                    retry,
                                    %error,
                                    "Could not revalidate task-lead lease; retrying"
                                );
                                tokio::time::sleep(Duration::from_millis(250)).await;
                            }
                        }
                    }
                }
                if !lease_live {
                    warn!(
                        goal_id = %goal_id,
                        attempt_id = %attempt.id,
                        "Discarding task-lead result after execution lease loss"
                    );
                    let _ = heartbeat_cancel_tx.send(());
                    let _ = heartbeat_handle.await;
                    let _ = keepalive_cancel_tx.send(());
                    let _ = keepalive_handle.await;
                    return;
                }
            }
            if is_mandate_run {
                if let Err(error) = &task_lead_run {
                    warn!(
                        goal_id = %goal_id,
                        run_id = goal_run_id.as_deref().unwrap_or("missing"),
                        %error,
                        "Mandate deliberator exited before returning a semantic outcome"
                    );
                }
            }
            let task_lead_outcome = task_lead_run.as_ref().ok().map(|run| run.outcome);
            let result = task_lead_run.map(|run| run.response);

            // Keep the task-lead textual response, but defer relay until we know
            // whether the goal is terminal. For terminal goals, we prefer the
            // canonical completion summary built from task results.
            let task_lead_response = result
                .as_ref()
                .ok()
                .map(|response| response.trim().to_string())
                .filter(|response| !response.is_empty());

            // Track whether any executor results were already sent inline to the user.
            // Used to avoid duplicate content in the completion notification.
            let mut any_executor_results_sent = false;
            let auto_dispatch_context = AutoDispatchContext {
                agent: agent.clone(),
                state: state.clone(),
                mission: mission.clone(),
                goal_id: goal_id.clone(),
                approval_session_id: session_id.clone(),
                channel_ctx: dispatch_channel_ctx.clone(),
                user_role: fallback_user_role,
                project_scope: dispatch_project_scope.clone(),
            };

            // Generic runs may auto-dispatch pending work. Mandate runs must use
            // the explicit control-plane claim path, which atomically binds the
            // child attempt to the root lease, current ACT, committed intention,
            // and immutable mandate version. The generic profile-based claimant
            // is therefore unavailable for mandate-origin work.
            if !is_mandate_run {
                let max_dispatch_rounds = 4; // safety limit — keep low to bound token usage
                let mut budget_exhausted = false;
                for _round in 0..max_dispatch_rounds {
                    if task_lead_lease_lost.load(Ordering::Acquire) {
                        warn!(
                            goal_id = %goal_id,
                            "Stopping task-lead auto-dispatch after execution lease loss"
                        );
                        break;
                    }
                    let all_tasks = work_tasks_for_current_run(
                        &state,
                        &goal_id,
                        goal_run_id.as_deref(),
                        dispatch_trigger_task_id.as_deref(),
                    )
                    .await;

                    // Build set of completed task IDs for dependency checking
                    let completed_ids: std::collections::HashSet<String> = all_tasks
                        .iter()
                        .filter(|t| {
                            matches!(t.status.as_str(), "completed" | "skipped" | "superseded")
                        })
                        .map(|t| t.id.clone())
                        .collect();

                    // Filter to pending tasks whose dependencies are all met
                    let dispatchable: Vec<crate::traits::Task> = all_tasks
                        .iter()
                        .filter(|t| t.status == "pending")
                        .filter(|t| match &t.depends_on {
                            None => true,
                            Some(deps_json) => serde_json::from_str::<Vec<String>>(deps_json)
                                .unwrap_or_default()
                                .iter()
                                .all(|dep_id| completed_ids.contains(dep_id)),
                        })
                        .cloned()
                        .collect();

                    if dispatchable.is_empty() {
                        break; // No more tasks to dispatch
                    }

                    // Conservative fallback behavior: only dispatch the earliest
                    // task_order in each round. This preserves intended sequencing
                    // when a task lead created ordered tasks but omitted depends_on.
                    let min_task_order =
                        dispatchable.iter().map(|t| t.task_order).min().unwrap_or(0);
                    let dispatch_batch: Vec<crate::traits::Task> = dispatchable
                        .into_iter()
                        .filter(|t| t.task_order == min_task_order)
                        .collect();

                    info!(
                        goal_id = %goal_id,
                        count = dispatch_batch.len(),
                        task_order = min_task_order,
                        round = _round,
                        "Auto-dispatching pending tasks after task lead"
                    );

                    let mut dispatched_parallel_groups = std::collections::HashSet::new();
                    for task in &dispatch_batch {
                        if task
                            .parallel_group
                            .as_ref()
                            .is_some_and(|group| dispatched_parallel_groups.contains(group))
                        {
                            continue;
                        }
                        // Stop dispatching when the active run has exhausted its
                        // shared per-run budget, or when a non-scheduled goal hits
                        // its daily budget.
                        if let Ok(Some(g)) = state.get_goal(&goal_id).await {
                            let is_scheduled =
                                goal_has_scheduled_provenance(&state, &goal_id, Some(&task.id))
                                    .await;
                            if is_scheduled {
                                let run_budget =
                                    if let Some(registry) = goal_token_registry.as_ref() {
                                        registry.get_run_budget(&goal_id).await
                                    } else {
                                        None
                                    };
                                if let Some(run_budget) = run_budget {
                                    if run_budget.tokens_used
                                        >= run_budget.effective_budget_per_check
                                    {
                                        let old_budget = run_budget.effective_budget_per_check;
                                        if let Some(new_budget) =
                                            auto_dispatch_scheduled_run_extension_budget(
                                                &run_budget,
                                                SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS,
                                                SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
                                            )
                                        {
                                            if let Some(registry) = goal_token_registry.as_ref() {
                                                if let Some(updated) = registry
                                                    .auto_extend_run_budget(&goal_id, new_budget)
                                                    .await
                                                {
                                                    persist_scheduled_run_state(
                                                        &state, &goal_id, None, &updated,
                                                    )
                                                    .await;
                                                    info!(
                                                        goal_id = %goal_id,
                                                        tokens_used = updated.tokens_used,
                                                        old_budget,
                                                        new_budget,
                                                        extension = updated.budget_extensions_count,
                                                        "Auto-extended scheduled run budget during auto-dispatch"
                                                    );
                                                } else {
                                                    budget_exhausted = true;
                                                    info!(
                                                        goal_id = %goal_id,
                                                        tokens_used = run_budget.tokens_used,
                                                        budget = run_budget.effective_budget_per_check,
                                                        "Stopping auto-dispatch — scheduled run budget exhausted"
                                                    );
                                                    break;
                                                }
                                            }
                                        } else {
                                            budget_exhausted = true;
                                            info!(
                                                goal_id = %goal_id,
                                                tokens_used = run_budget.tokens_used,
                                                budget = run_budget.effective_budget_per_check,
                                                "Stopping auto-dispatch — scheduled run budget exhausted"
                                            );
                                            break;
                                        }
                                    }
                                }
                            } else if let Some(budget_daily) =
                                effective_goal_daily_budget(&g, goal_token_registry.as_ref()).await
                            {
                                if g.tokens_used_today >= budget_daily {
                                    budget_exhausted = true;
                                    info!(
                                        goal_id = %goal_id,
                                        tokens_used = g.tokens_used_today,
                                        budget = budget_daily,
                                        "Stopping auto-dispatch — goal daily budget exhausted"
                                    );
                                    break;
                                }
                            }
                        }

                        if let Some(group) = task.parallel_group.as_ref() {
                            let grouped = dispatch_batch
                                .iter()
                                .filter(|candidate| {
                                    candidate.parallel_group.as_deref() == Some(group.as_str())
                                })
                                .cloned()
                                .collect::<Vec<_>>();
                            if grouped.len() > 1 {
                                dispatched_parallel_groups.insert(group.clone());
                                // Profile caps apply at claim time; this local
                                // cap also bounds aggregate load across profiles.
                                for chunk in grouped.chunks(4) {
                                    let executions = futures::future::join_all(
                                        chunk.iter().cloned().map(|parallel_task| {
                                            execute_auto_dispatch_task(
                                                auto_dispatch_context.clone(),
                                                parallel_task,
                                            )
                                        }),
                                    )
                                    .await;
                                    for (executed_task, execution_result) in
                                        executions.into_iter().flatten()
                                    {
                                        any_executor_results_sent |= deliver_auto_dispatch_result(
                                            &agent,
                                            &state,
                                            hub.as_ref(),
                                            &session_id,
                                            &executed_task,
                                            &execution_result,
                                            !is_scheduled_run,
                                        )
                                        .await;
                                    }
                                }
                                continue;
                            }
                        }

                        if let Some((executed_task, execution_result)) =
                            execute_auto_dispatch_task(auto_dispatch_context.clone(), task.clone())
                                .await
                        {
                            any_executor_results_sent |= deliver_auto_dispatch_result(
                                &agent,
                                &state,
                                hub.as_ref(),
                                &session_id,
                                &executed_task,
                                &execution_result,
                                !is_scheduled_run,
                            )
                            .await;
                        }
                    }

                    if budget_exhausted {
                        break;
                    }
                }
            }

            // Mark the trigger task as completed now that the task lead and auto-dispatch
            // have finished. The trigger task was kept in "running" to prevent duplicate
            // dispatch; now finalize it so it doesn't appear stuck.
            let tasks_before_trigger_finalization =
                tasks_for_current_run(&state, &goal_id, goal_run_id.as_deref()).await;
            let completed_work_after = count_successful_completed_work(
                &tasks_before_trigger_finalization,
                dispatch_trigger_task_id.as_deref(),
            );
            let dispatch_made_progress = completed_work_after > completed_work_before;
            let mut trigger_work_tasks = tasks_before_trigger_finalization
                .iter()
                .filter(|task| Some(task.id.as_str()) != dispatch_trigger_task_id.as_deref())
                .cloned()
                .collect::<Vec<_>>();
            // A task lead may recover a delegated provider outage by completing
            // the mission directly. Reconcile only that narrow, evidence-backed
            // case so the abandoned child attempt cannot veto the successful
            // root outcome. Semantic/tool failures remain failed and continue to
            // block the run.
            if !is_mandate_run && task_lead_outcome == Some(TaskOutcome::Succeeded) {
                for task in &mut trigger_work_tasks {
                    if !task_failed_only_due_to_provider_infrastructure(task) {
                        continue;
                    }
                    let mut superseded = task.clone();
                    superseded.status = "superseded".to_string();
                    superseded.result = Some(
                        "The task lead completed the mission directly after this delegated provider failure."
                            .to_string(),
                    );
                    superseded.error = None;
                    superseded.blocker = None;
                    superseded.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    match state.update_task(&superseded).await {
                        Ok(()) => *task = superseded,
                        Err(error) => warn!(
                            goal_id = %goal_id,
                            task_id = %task.id,
                            %error,
                            "Failed to supersede a provider-failed child after direct task-lead recovery"
                        ),
                    }
                }
            }
            // No durable decision means no ACT authority exists, regardless of
            // how the deliberator process itself exited. Close that semantic
            // state as WAIT even after a provider/setup/orchestration error;
            // this can never manufacture mutation permission.
            let safe_wait_committed = if is_mandate_run {
                match goal_run_id.as_deref() {
                    Some(run_id) => {
                        persist_safe_wait_if_decision_missing(&state, &goal_id, run_id).await
                    }
                    None => false,
                }
            } else {
                false
            };
            let mandate_decision_ready = if !is_mandate_run || safe_wait_committed {
                true
            } else {
                match goal_run_id.as_deref() {
                    Some(run_id) => match state.get_mandate_decision_for_run(run_id).await {
                        Ok(Some(decision)) => state
                            .get_mandate_for_goal(&goal_id)
                            .await
                            .ok()
                            .flatten()
                            .is_some_and(|mandate| mandate.version == decision.mandate_version),
                        _ => false,
                    },
                    None => false,
                }
            };
            // When a durable attempt exists, no goal/run/mandate finalization
            // below is authoritative until this exact lease atomically commits
            // the trigger result. A stale finalizer must not close the run that
            // a replacement task lead now owns.
            let mut trigger_finalization_authoritative = dispatch_attempt.is_none();
            if let Some(ref trigger_task_id) = dispatch_trigger_task_id {
                if let Ok(Some(trigger_task)) = state.get_task(trigger_task_id).await {
                    if trigger_task.status == "running" || trigger_task.status == "claimed" {
                        let semantic_task_lead_success = task_lead_semantically_succeeded(
                            task_lead_outcome,
                            is_mandate_run,
                            mandate_decision_ready,
                        );
                        let disposition = dispatch_trigger_disposition(
                            semantic_task_lead_success,
                            &trigger_work_tasks,
                        );
                        let disposition = if mandate_decision_ready {
                            disposition
                        } else {
                            DispatchTriggerDisposition::Failed
                        };
                        let status = match disposition {
                            DispatchTriggerDisposition::Completed
                            | DispatchTriggerDisposition::Continuable => "completed".to_string(),
                            DispatchTriggerDisposition::Failed => "failed".to_string(),
                        };
                        let error = if is_mandate_run && !mandate_decision_ready {
                            Some(
                                "Mandate review ended without exactly one current durable decision."
                                    .to_string(),
                            )
                        } else if disposition == DispatchTriggerDisposition::Failed {
                            Some(failed_run_summary(
                                &trigger_work_tasks,
                                result
                                    .as_ref()
                                    .err()
                                    .map(|error| error.to_string())
                                    .as_deref(),
                            ))
                        } else {
                            None
                        };
                        if let Some(attempt) = dispatch_attempt.as_ref() {
                            let handoff = crate::traits::TaskHandoff {
                                id: uuid::Uuid::new_v4().to_string(),
                                task_id: trigger_task_id.clone(),
                                attempt_id: attempt.id.clone(),
                                summary: match disposition {
                                    DispatchTriggerDisposition::Completed => {
                                        "Task-lead run completed.".to_string()
                                    }
                                    DispatchTriggerDisposition::Continuable => {
                                        "Task lead committed durable unfinished work for autonomous continuation."
                                            .to_string()
                                    }
                                    DispatchTriggerDisposition::Failed => error
                                        .clone()
                                        .unwrap_or_else(|| "Task-lead run failed.".to_string()),
                                },
                                artifacts: Vec::new(),
                                verification: Vec::new(),
                                remaining_risk: error.clone(),
                                next_step: match disposition {
                                    DispatchTriggerDisposition::Completed => None,
                                    DispatchTriggerDisposition::Continuable => Some(
                                        "Resume the next pending child obligation from this goal run."
                                            .to_string(),
                                    ),
                                    DispatchTriggerDisposition::Failed => {
                                        Some("Inspect failed or blocked child tasks.".to_string())
                                    }
                                },
                                created_at: chrono::Utc::now().to_rfc3339(),
                            };
                            let patch = crate::traits::TaskAttemptPatch {
                                status,
                                error,
                                handoff: Some(handoff),
                                ..Default::default()
                            };
                            match patch_task_attempt_with_transient_retry(&state, attempt, &patch)
                                .await
                            {
                                Ok(true) => trigger_finalization_authoritative = true,
                                Ok(false) => {
                                    let persisted_status = state
                                        .get_task(trigger_task_id)
                                        .await
                                        .ok()
                                        .flatten()
                                        .map(|task| task.status)
                                        .unwrap_or_else(|| "missing".to_string());
                                    warn!(
                                        task_id = %trigger_task_id,
                                        goal_id = %goal_id,
                                        attempt_id = %attempt.id,
                                        persisted_status,
                                        "Ignored stale task-lead finalization"
                                    );
                                    trigger_finalization_authoritative = false;
                                }
                                Err(error) => {
                                    error!(
                                        task_id = %trigger_task_id,
                                        goal_id = %goal_id,
                                        attempt_id = %attempt.id,
                                        %error,
                                        "Failed to finalize task-lead attempt after retries"
                                    );
                                    trigger_finalization_authoritative = false;
                                }
                            }
                        } else {
                            let mut updated = trigger_task;
                            updated.status = status;
                            updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            updated.error = error;
                            if let Err(e) = state.update_task(&updated).await {
                                warn!(
                                    task_id = %trigger_task_id,
                                    goal_id = %goal_id,
                                    error = %e,
                                    "Failed to finalize dispatch trigger task"
                                );
                            }
                        }
                    }
                }
            }

            // Stop the heartbeat
            let _ = heartbeat_cancel_tx.send(());
            let terminal_surface_id = heartbeat_handle.await.ok().flatten();
            let _ = keepalive_cancel_tx.send(());
            let _ = keepalive_handle.await;

            if !trigger_finalization_authoritative {
                warn!(
                    goal_id = %goal_id,
                    "Aborting goal-run finalization because the task-lead lease was lost"
                );
                return;
            }

            // Check the actual goal status from DB — the task lead may have already
            // set it via complete_goal/fail_goal. Only update if still "active".
            let current_goal = state.get_goal(&goal.id).await;
            let needs_status_update = match &current_goal {
                Ok(Some(g)) => g.status == "active" || g.status == "pending",
                _ => true, // fallback: update if we can't read
            };

            if needs_status_update && !is_mandate_run {
                // Task lead returned without explicitly completing/failing the goal.
                // Use progress-based circuit breaker: compare completed task count
                // before vs after to detect whether the dispatch made progress.
                let tasks = work_tasks_for_current_run(
                    &state,
                    &goal_id,
                    goal_run_id.as_deref(),
                    dispatch_trigger_task_id.as_deref(),
                )
                .await;
                let completed_after = tasks
                    .iter()
                    .filter(|task| task.completed_successfully())
                    .count();
                let all_done =
                    !tasks.is_empty() && tasks.iter().all(|t| t.satisfies_run_completion());

                let mut updated_goal = match state.get_goal(&goal_id).await {
                    Ok(Some(g)) => g,
                    _ => goal,
                };

                let scheduled_goal_active =
                    goal_has_scheduled_provenance(&state, &goal_id, None).await;
                let scheduled_run_budget_exhausted = if scheduled_goal_active {
                    if let Some(registry) = goal_token_registry.as_ref() {
                        registry
                            .get_run_budget(&goal_id)
                            .await
                            .is_some_and(|status| {
                                status.tokens_used >= status.effective_budget_per_check
                            })
                    } else {
                        false
                    }
                } else {
                    false
                };
                let effective_goal_budget =
                    effective_goal_daily_budget(&updated_goal, goal_token_registry.as_ref()).await;
                let goal_budget_exhausted = !scheduled_goal_active
                    && effective_goal_budget.is_some_and(|b| updated_goal.tokens_used_today >= b);

                // For finite goals: detect when no tasks were completed after
                // the task lead finished — fail immediately since there's no
                // re-dispatch mechanism for finite goals.
                let is_finite = updated_goal.goal_type == "finite";
                let any_completed = tasks.iter().any(|t| t.status == "completed");
                let no_tasks_completed_finite = is_finite && !tasks.is_empty() && !any_completed;

                if is_mandate_run {
                    // ACT/WAIT/ASK/STOP are legitimate cycle outcomes. In
                    // particular, WAIT has no child work by design and must not
                    // increment the generic continuous-goal no-progress breaker.
                    // The mandate finalizer below owns controller/mandate state.
                    updated_goal.dispatch_failures = 0;
                } else if all_done {
                    // A recurring goal completes a run, not the goal itself.
                    // Keep it active for the next schedule fire.
                    if run_completion_should_close_goal(&updated_goal.goal_type, all_done) {
                        updated_goal.status = "completed".to_string();
                        updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    }
                    updated_goal.dispatch_failures = 0;
                    clear_partial_success_context(&mut updated_goal);
                } else if scheduled_run_budget_exhausted {
                    updated_goal.dispatch_failures = 0;
                    info!(
                        goal_id = %goal_id,
                        "Goal dispatch paused: scheduled run budget exhausted"
                    );
                } else if goal_budget_exhausted {
                    // Budget exhausted is a safety stop, not "no progress". Keep the goal active
                    // and avoid stalling it; it can resume after budgets reset.
                    updated_goal.dispatch_failures = 0;
                    info!(
                        goal_id = %goal_id,
                        tokens_used = updated_goal.tokens_used_today,
                        budget = effective_goal_budget.unwrap_or(0),
                        "Goal dispatch paused: daily token budget exhausted"
                    );
                } else if no_tasks_completed_finite {
                    // Finite goal with zero completed tasks — fail fast.
                    // This covers tasks stuck in any non-completed status:
                    // pending, claimed, blocked, or failed. Since finite goals
                    // have no re-dispatch loop, waiting is pointless.
                    updated_goal.status = "failed".to_string();
                    updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    let pending = tasks
                        .iter()
                        .filter(|t| t.status == "pending" || t.status == "claimed")
                        .count();
                    let blocked = tasks.iter().filter(|t| t.status == "blocked").count();
                    let failed = tasks.iter().filter(|t| t.status == "failed").count();
                    info!(
                        goal_id = %goal_id,
                        pending,
                        blocked,
                        failed,
                        "Finite goal failed: no tasks completed after dispatch"
                    );
                } else if is_finite {
                    // Finite goals are all-or-nothing at the goal boundary.
                    // Preserve useful partial work in context, but never turn it
                    // into a successful terminal state while required tasks remain.
                    let completed_count =
                        tasks.iter().filter(|t| t.completed_successfully()).count();
                    let failed_count = tasks.iter().filter(|t| t.status == "failed").count();
                    let blocked_count = tasks.iter().filter(|t| t.status == "blocked").count();
                    let remaining = tasks
                        .iter()
                        .filter(|t| !t.satisfies_run_completion())
                        .count();
                    updated_goal.status = "failed".to_string();
                    updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());

                    let summary = serde_json::json!({
                        "partial_success": completed_count > 0,
                        "completed": completed_count,
                        "failed": failed_count,
                        "blocked": blocked_count,
                        "total": tasks.len(),
                    });
                    updated_goal.context = Some(summary.to_string());
                    info!(
                        goal_id = %goal_id,
                        completed_count,
                        failed_count,
                        blocked_count,
                        remaining,
                        "Finite goal failed with partial work preserved"
                    );
                } else {
                    // Continuous goal: evaluate progress made by this dispatch,
                    // not merely the task lead's final Result. A child can return
                    // a partial/error outcome after completing useful tasks; that
                    // must reset the consecutive no-progress breaker.
                    // Check if all remaining non-completed tasks are blocked
                    // (waiting on external input/dependencies). Blocked tasks are
                    // waiting, not failing — don't count as "no progress".
                    let all_remaining_blocked = tasks
                        .iter()
                        .filter(|t| !t.satisfies_run_completion())
                        .all(|t| t.status == "blocked");

                    match classify_continuous_dispatch_outcome(
                        dispatch_made_progress,
                        result.is_err(),
                        result.as_ref().err().is_some_and(|error| {
                            crate::providers::is_provider_infra_error_text(&error.to_string())
                        }),
                        all_remaining_blocked && !tasks.is_empty(),
                    ) {
                        ContinuousDispatchOutcome::Progress => {
                            updated_goal.dispatch_failures = 0;
                            info!(
                                goal_id = %goal_id,
                                completed_work_before,
                                completed_work_after,
                                task_lead_errored = result.is_err(),
                                "Dispatch made progress; reset dispatch_failures"
                            );
                        }
                        ContinuousDispatchOutcome::TransientInfrastructureFailure => {
                            info!(
                                goal_id = %goal_id,
                                dispatch_failures = updated_goal.dispatch_failures,
                                "Transient provider failure made no progress; preserving the goal failure counter for a later retry"
                            );
                        }
                        ContinuousDispatchOutcome::ErrorWithoutProgress => {
                            updated_goal.dispatch_failures += 1;
                            info!(
                                goal_id = %goal_id,
                                dispatch_failures = updated_goal.dispatch_failures,
                                "Task lead errored without completing work"
                            );
                        }
                        ContinuousDispatchOutcome::Blocked => {
                            info!(
                                goal_id = %goal_id,
                                blocked_tasks = tasks.iter().filter(|t| t.status == "blocked").count(),
                                "All remaining tasks are blocked — not incrementing dispatch_failures"
                            );
                        }
                        ContinuousDispatchOutcome::NoProgress => {
                            updated_goal.dispatch_failures += 1;
                            info!(
                                goal_id = %goal_id,
                                dispatch_failures = updated_goal.dispatch_failures,
                                completed_tasks = completed_after,
                                remaining_tasks = tasks.iter().filter(|t| t.status == "pending" || t.status == "claimed").count(),
                                "No progress this dispatch cycle"
                            );
                        }
                    }
                }

                // Circuit breaker: stall after 3 consecutive failures
                const MAX_DISPATCH_FAILURES: i32 = 3;
                if updated_goal.dispatch_failures >= MAX_DISPATCH_FAILURES
                    && updated_goal.status != "completed"
                    && updated_goal.status != "failed"
                {
                    updated_goal.status = "stalled".to_string();
                    info!(
                        goal_id = %goal_id,
                        dispatch_failures = updated_goal.dispatch_failures,
                        "Goal stalled: {} consecutive dispatch cycles with no progress",
                        updated_goal.dispatch_failures
                    );
                }

                updated_goal.updated_at = chrono::Utc::now().to_rfc3339();
                let _ = state.update_goal(&updated_goal).await;

                // If goal is stalled or failed, cancel remaining pending tasks
                if updated_goal.status == "stalled" || updated_goal.status == "failed" {
                    let mut cancelled = 0;
                    for task in &tasks {
                        if (task.status == "pending" || task.status == "claimed")
                            && state
                                .cancel_work_task(&task.id, "task-lead", None)
                                .await
                                .unwrap_or(false)
                        {
                            cancelled += 1;
                        }
                    }
                    if cancelled > 0 {
                        info!(goal_id = %goal_id, cancelled, "Cancelled orphaned tasks for stalled goal");
                    }
                }
            }

            // Enqueue notification for delivery (persisted in SQLite).
            // Then attempt immediate delivery via hub if available.
            let final_goal = state.get_goal(&goal_id).await;
            let status = final_goal
                .as_ref()
                .ok()
                .and_then(|g| g.as_ref())
                .map(|g| g.status.as_str())
                .unwrap_or("unknown");
            let final_run_tasks = work_tasks_for_current_run(
                &state,
                &goal_id,
                goal_run_id.as_deref(),
                dispatch_trigger_task_id.as_deref(),
            )
            .await;
            let trigger_status = if let Some(trigger_task_id) = dispatch_trigger_task_id.as_deref()
            {
                state
                    .get_task(trigger_task_id)
                    .await
                    .ok()
                    .flatten()
                    .map(|task| task.status)
            } else {
                None
            };

            // Mandate cycles have their own lifecycle and wake clock. They are
            // never recurring scheduled runs, even though their backing goal is
            // continuous, so finalize the decision/intention/lease and return
            // before the generic scheduled-run notification branch below.
            if is_mandate_run {
                if let Some(run_id) = goal_run_id.as_deref() {
                    let controller_goal = final_goal
                        .as_ref()
                        .ok()
                        .and_then(|goal| goal.as_ref())
                        .unwrap_or(&initial_goal);
                    let fallback_error = result.as_ref().err().map(|error| error.to_string());
                    if let Some(notice) = finalize_mandate_review(
                        &state,
                        controller_goal,
                        run_id,
                        trigger_status.as_deref(),
                        &final_run_tasks,
                        fallback_error.as_deref(),
                        any_executor_results_sent,
                    )
                    .await
                    {
                        // The state finalizer already committed this exact
                        // deterministic outbox row with terminal proof state.
                        // Immediate delivery is only a latency optimization;
                        // failures leave the critical row pending for retry.
                        let entry = notice.to_notification_entry();
                        let notification_id = entry.id.clone();
                        match agent
                            .deliver_parent_text_result(
                                hub.as_ref(),
                                &session_id,
                                &entry.message,
                                parent_delivery::ParentDeliveryKind::GoalNotification,
                            )
                            .await
                        {
                            Ok(outcome) => {
                                if outcome.recorded
                                    && notice.kind == MandateRunNotificationKind::Ask
                                {
                                    if let Err(error) = agent
                                        .record_mandate_owner_input_context(
                                            &session_id,
                                            &notice.mandate_id,
                                            notice.mandate_version,
                                            &notification_id,
                                        )
                                        .await
                                    {
                                        warn!(
                                            mandate_id = %notice.mandate_id,
                                            run_id,
                                            %error,
                                            "Failed to bind delivered mandate ASK to owner dialogue state"
                                        );
                                    }
                                }
                                if outcome.sent {
                                    let _ =
                                        state.mark_notification_delivered(&notification_id).await;
                                }
                            }
                            Err(error) => warn!(
                                goal_id = %goal_id,
                                run_id,
                                %error,
                                "Failed to deliver mandate-review notification"
                            ),
                        }
                    }
                } else {
                    warn!(goal_id = %goal_id, "Mandate task lead had no durable goal run");
                }

                if let Some(registry) = goal_token_registry.as_ref() {
                    registry.remove(&goal_id).await;
                }
                return;
            }

            let run_status =
                goal_run_terminal_status(status, trigger_status.as_deref(), &final_run_tasks);
            let continuous_controller = initial_goal.goal_type == "continuous";
            let defer_run_finalization_for_recovery = continuous_controller
                && matches!(run_status, Some("failed" | "blocked"))
                && !final_run_tasks.iter().any(is_terminal_recovery_task);
            if let Some(run_id) = goal_run_id.as_deref() {
                if let Some(run_status) =
                    run_status.filter(|_| !defer_run_finalization_for_recovery)
                {
                    let summary = build_goal_task_results_summary(
                        &final_run_tasks,
                        if run_status == "completed" {
                            "Run completed."
                        } else {
                            "Run needs attention."
                        },
                    );
                    if let Err(error) = state
                        .finish_goal_run(run_id, run_status, Some(&summary))
                        .await
                    {
                        warn!(
                            goal_id = %goal_id,
                            run_id,
                            %error,
                            "Failed to finalize durable goal run"
                        );
                    }
                }
            }

            // A recurring goal stays active after each cycle, but an individual
            // cycle can still complete or fail. Surface that terminal run outcome instead
            // of silently treating the durable goal's active status as
            // "nothing to notify."
            if status == "active" || status == "pending" {
                let terminal_notification = match recurring_run_terminal_outcome(run_status) {
                    Some(RecurringRunTerminalOutcome::Failed) => {
                        let task_lead_error = result.as_ref().err().map(|error| error.to_string());
                        Some((
                            "failed",
                            failed_run_summary(&final_run_tasks, task_lead_error.as_deref()),
                        ))
                    }
                    Some(RecurringRunTerminalOutcome::Completed) => {
                        let summary = build_goal_task_results_summary(
                            &final_run_tasks,
                            "All required tasks completed.",
                        );
                        Some(("completed", truncate_goal_result_text(&summary, 3500)))
                    }
                    None => None,
                };

                // A recoverable terminal cycle is not yet an owner-facing
                // event. Route it through the same bounded direct-recovery path
                // used by finite goals. The durable recovery-task marker keeps
                // this to one attempt for the exact run.
                let defer_to_autonomous_recovery = continuous_controller
                    && matches!(
                        recurring_run_terminal_outcome(run_status),
                        Some(RecurringRunTerminalOutcome::Failed)
                    )
                    && !final_run_tasks.iter().any(is_terminal_recovery_task);

                if !defer_to_autonomous_recovery {
                    if let Some((notification_type, msg)) = terminal_notification {
                        let msg = crate::channels::present_scheduled_run_notification(
                            notification_type,
                            &msg,
                            true,
                        );
                        let entry = crate::traits::NotificationEntry::new(
                            &goal_id,
                            &session_id,
                            notification_type,
                            &msg,
                        );
                        let notification_id = entry.id.clone();
                        let queued = match state.enqueue_notification(&entry).await {
                            Ok(()) => true,
                            Err(error) => {
                                warn!(
                                    goal_id = %goal_id,
                                    notification_id = %notification_id,
                                    %error,
                                    "Failed to queue recurring-run terminal notification"
                                );
                                false
                            }
                        };
                        let local_hour = chrono::Timelike::hour(&chrono::Local::now());
                        if !entry.should_deliver_at_local_hour(local_hour) {
                            // The durable outbox remains pending for the heartbeat
                            // to deliver after quiet hours.
                            if let Some(ref registry) = goal_token_registry {
                                registry.remove(&goal_id).await;
                            }
                            return;
                        }
                        match agent
                            .deliver_parent_text_result_to_surface(
                                hub.as_ref(),
                                &session_id,
                                terminal_surface_id.as_deref(),
                                &msg,
                                parent_delivery::ParentDeliveryKind::GoalNotification,
                            )
                            .await
                        {
                            Ok(outcome) if outcome.sent => {
                                if queued {
                                    let _ =
                                        state.mark_notification_delivered(&notification_id).await;
                                }
                            }
                            Ok(_) => {}
                            Err(error) => {
                                warn!(
                                    session_id = %session_id,
                                    notification_id = %notification_id,
                                    %error,
                                    "Failed to deliver recurring-run terminal notification"
                                );
                            }
                        }
                    } else {
                        // Durable child obligations remain open. Routine
                        // handoffs are internal coordination, not owner-facing
                        // attention events; heartbeat will resume them.
                    }

                    // Clean up cancellation token and return.
                    if run_status.is_some() {
                        if let Some(ref registry) = goal_token_registry {
                            registry.remove(&goal_id).await;
                        }
                    }
                    return;
                }
            }

            // For failed/stalled finite goals: attempt direct fallback before giving up.
            // The goal system decomposed the request into subtasks but they weren't
            // completed. Instead of sending a cryptic failure message, try handling
            // the request directly through the agent's main capabilities.
            //
            // Skip fallback if the goal was already notified — this means another
            // task lead (e.g., spawned by the heartbeat) already handled the failure.
            let goal_already_notified = final_goal
                .as_ref()
                .ok()
                .and_then(|g| g.as_ref())
                .map(|g| g.notified_at.is_some())
                .unwrap_or(false);
            let goal_type = final_goal
                .as_ref()
                .ok()
                .and_then(|goal| goal.as_ref())
                .map(|goal| goal.goal_type.as_str())
                .unwrap_or(initial_goal.goal_type.as_str());
            let (notification_type, msg) = if terminal_recovery_eligible(
                goal_type,
                status,
                run_status,
                goal_already_notified,
                &final_run_tasks,
            ) {
                info!(goal_id = %goal_id, is_scheduled_run, "Goal run failed — attempting bounded durable direct recovery");

                // Prevent the heartbeat from racing a duplicate terminal notice
                // while a separately tracked recovery run is active.
                if !continuous_controller {
                    let _ = state.mark_goal_notified(&goal_id).await;
                }
                if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                    g.status = "active".to_string();
                    g.completed_at = None;
                    if !continuous_controller {
                        g.notified_at = None;
                    }
                    g.updated_at = chrono::Utc::now().to_rfc3339();
                    let _ = state.update_goal(&g).await;
                }

                let local_hour = chrono::Timelike::hour(&chrono::Local::now());
                if crate::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(
                    local_hour,
                ) {
                    if let Some(hub_weak) = &hub {
                        if let Some(hub_arc) = hub_weak.upgrade() {
                            let retry_text = if is_scheduled_run {
                                "⏳ **Scheduled run in progress**\n\nThe first attempt did not finish, so I’m retrying safely…"
                            } else {
                                "⏳ **Still working**\n\nThe first attempt did not finish, so I’m retrying safely…"
                            };
                            let edited = if let Some(surface_id) = terminal_surface_id.as_deref() {
                                hub_arc
                                    .edit_text(&session_id, surface_id, retry_text)
                                    .await
                                    .unwrap_or(false)
                            } else {
                                false
                            };
                            // A scheduled retry is routine unattended recovery. If
                            // no existing progress surface can be updated, stay
                            // silent until the run reaches a genuine terminal state.
                            if !edited && !is_scheduled_run {
                                let _ = hub_arc.send_text(&session_id, retry_text).await;
                            }
                        }
                    }
                }

                let now = chrono::Utc::now().to_rfc3339();
                let recovery_task_id = uuid::Uuid::new_v4().to_string();
                let recovery_task = crate::traits::Task {
                    id: recovery_task_id.clone(),
                    goal_id: goal_id.clone(),
                    description: format!("Directly recover and finish: {user_text}"),
                    status: "pending".to_string(),
                    priority: "high".to_string(),
                    task_order: final_run_tasks
                        .iter()
                        .map(|task| task.task_order)
                        .max()
                        .unwrap_or(0)
                        + 1,
                    parallel_group: None,
                    depends_on: None,
                    agent_id: None,
                    context: Some(
                        serde_json::json!({
                            "recovery_for_run": goal_run_id.as_deref(),
                            "terminal_recovery": true,
                            "project_scope": dispatch_project_scope.as_deref(),
                            "prior_report": task_lead_response.as_deref().map(|report| {
                                report.chars().take(2000).collect::<String>()
                            }),
                        })
                        .to_string(),
                    ),
                    result: None,
                    error: None,
                    blocker: None,
                    idempotent: false,
                    retry_count: 0,
                    max_retries: 1,
                    created_at: now,
                    started_at: None,
                    completed_at: None,
                };
                let recovery_created = state.create_task(&recovery_task).await;
                let recovery_run_id = if recovery_created.is_ok() {
                    if is_scheduled_run {
                        rebind_scheduled_recovery_state(
                            &state,
                            goal_token_registry.as_ref(),
                            &goal_id,
                            &recovery_task_id,
                        )
                        .await;
                    }
                    state
                        .get_current_goal_run(&goal_id)
                        .await
                        .ok()
                        .flatten()
                        .map(|run| run.id)
                        .or_else(|| goal_run_id.clone())
                } else {
                    goal_run_id.clone()
                };
                let recovery_ledger = render_recovery_task_ledger(&final_run_tasks);
                let prior_report = task_lead_response
                    .as_deref()
                    .map(|report| report.chars().take(3000).collect::<String>())
                    .unwrap_or_else(|| "No usable prior summary was returned.".to_string());
                let recovery_prompt = format!(
                    "{user_text}\n\nRecover the whole mission autonomously, using the durable task ledger below as authoritative evidence. Preserve successful work and never repeat a completed external side effect. Inspect the exact failure, make only the smallest in-scope correction needed, then rerun the documented validation and continue from the first unmet obligation. Scope is causal, not chronological: when the required build, test, validation, or deployment workflow identifies a pre-existing file inside the authorized project as its concrete blocker, that file is a task dependency. Inspect it and make the smallest reversible repair that restores the existing mechanical invariant while preserving body content and unrelated behavior. Dirty or untracked status alone does not make that dependency unrelated. Do not change files with no observed causal connection, broaden external authority, make destructive changes, or ask the owner about routine tactics. If the requested artifact, deployment, and verification are already proven by the ledger, report the mission complete without demanding a second verification surface. Only execute work that is genuinely unmet.\n\nExisting task ledger:\n{recovery_ledger}\n\nPrevious task-lead report:\n{prior_report}"
                );
                let fallback_result = match recovery_created {
                    Ok(()) => {
                        agent
                            .spawn_child_with_outcome(
                                &user_text,
                                &recovery_prompt,
                                None,
                                fallback_channel_ctx,
                                fallback_user_role,
                                Some(AgentRole::Executor),
                                Some(&goal_id),
                                Some(&recovery_task_id),
                                dispatch_project_scope.as_deref(),
                                Some("executor"),
                                Some(&session_id),
                            )
                            .await
                    }
                    Err(error) => Err(error),
                };

                match fallback_result {
                    Ok(run) => {
                        let persisted = state.get_task(&recovery_task_id).await.ok().flatten();
                        let structurally_succeeded =
                            recovery_task_succeeded(run.outcome, persisted.as_ref());
                        if structurally_succeeded {
                            for prior in &final_run_tasks {
                                if prior.satisfies_run_completion() {
                                    continue;
                                }
                                let mut superseded = prior.clone();
                                superseded.status = "superseded".to_string();
                                superseded.error = None;
                                superseded.blocker = None;
                                superseded.result = Some(format!(
                                    "Replaced by successful recovery task {recovery_task_id}."
                                ));
                                superseded.completed_at = Some(chrono::Utc::now().to_rfc3339());
                                let _ = state.update_task(&superseded).await;
                            }
                            if let Some(trigger_task_id) = dispatch_trigger_task_id.as_deref() {
                                if let Ok(Some(mut trigger)) = state.get_task(trigger_task_id).await
                                {
                                    if !trigger.satisfies_run_completion() {
                                        trigger.status = "superseded".to_string();
                                        trigger.error = None;
                                        trigger.blocker = None;
                                        trigger.result = Some(format!(
                                            "Replaced by successful recovery task {recovery_task_id}."
                                        ));
                                        trigger.completed_at =
                                            Some(chrono::Utc::now().to_rfc3339());
                                        let _ = state.update_task(&trigger).await;
                                    }
                                }
                            }
                            if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                                if continuous_controller {
                                    g.status = "active".to_string();
                                    g.completed_at = None;
                                } else {
                                    g.status = "completed".to_string();
                                    g.completed_at = Some(chrono::Utc::now().to_rfc3339());
                                }
                                g.updated_at = chrono::Utc::now().to_rfc3339();
                                clear_partial_success_context(&mut g);
                                let _ = state.update_goal(&g).await;
                            }
                            if let Some(run_id) = recovery_run_id.as_deref() {
                                let _ = state
                                    .finish_goal_run(run_id, "completed", Some(&run.response))
                                    .await;
                            }
                            info!(goal_id = %goal_id, task_id = %recovery_task_id, "Direct recovery succeeded");
                            (
                                "completed",
                                format!(
                                    "Goal completed: {}",
                                    truncate_goal_result_text(&run.response, 4000)
                                ),
                            )
                        } else {
                            if let Some(run_id) = recovery_run_id.as_deref() {
                                let _ = state
                                    .finish_goal_run(
                                        run_id,
                                        "failed",
                                        Some("Recovery task did not produce a successful structured outcome."),
                                    )
                                    .await;
                            }
                            if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                                if continuous_controller {
                                    g.status = "active".to_string();
                                    g.completed_at = None;
                                } else {
                                    g.status = "failed".to_string();
                                    g.completed_at = Some(chrono::Utc::now().to_rfc3339());
                                }
                                g.updated_at = chrono::Utc::now().to_rfc3339();
                                let _ = state.update_goal(&g).await;
                            }
                            info!(goal_id = %goal_id, task_id = %recovery_task_id, outcome = ?run.outcome, "Direct recovery remained incomplete");
                            (
                                "failed",
                                format!(
                                    "The recovery task did not complete every required outcome:\n\n{}",
                                    truncate_goal_result_text(&run.response, 3500)
                                ),
                            )
                        }
                    }
                    Err(error) => {
                        if let Some(run_id) = recovery_run_id.as_deref() {
                            let _ = state
                                .finish_goal_run(run_id, "failed", Some(&error.to_string()))
                                .await;
                        }
                        if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                            if continuous_controller {
                                g.status = "active".to_string();
                                g.completed_at = None;
                            } else {
                                g.status = "failed".to_string();
                                g.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            }
                            g.updated_at = chrono::Utc::now().to_rfc3339();
                            let _ = state.update_goal(&g).await;
                        }
                        let task_summary =
                            failed_run_summary(&final_run_tasks, Some(&error.to_string()));
                        info!(goal_id = %goal_id, task_id = %recovery_task_id, %error, "Direct recovery failed");
                        (
                            "failed",
                            format!("I was not able to complete the request:\n\n{task_summary}"),
                        )
                    }
                }
            } else {
                let completed_tasks = work_tasks_for_current_run(
                    &state,
                    &goal_id,
                    goal_run_id.as_deref(),
                    dispatch_trigger_task_id.as_deref(),
                )
                .await;
                let task_lead_error = result.as_ref().err().map(|e| e.to_string());
                match status {
                    "completed" => {
                        if !crate::tools::manage_goal_tasks::tasks_satisfy_goal_completion(
                            &completed_tasks,
                        ) {
                            (
                                "failed",
                                format!(
                                    "Goal state was inconsistent with its required tasks:\n\n{}",
                                    failed_run_summary(
                                        &completed_tasks,
                                        task_lead_error.as_deref()
                                    )
                                ),
                            )
                        } else if any_executor_results_sent {
                            // Executor results were already sent inline — don't repeat them.
                            // Send a brief completion signal instead.
                            let desc_preview: String = final_goal
                                .as_ref()
                                .ok()
                                .and_then(|g| g.as_ref())
                                .map(|g| g.description.chars().take(100).collect::<String>())
                                .unwrap_or_default();
                            ("completed", format!("Goal completed: {}", desc_preview))
                        } else {
                            // No inline results sent — include full task results in notification.
                            let fallback_summary = match &result {
                                Ok(r) => r.as_str(),
                                Err(_) => "All tasks completed.",
                            };
                            let task_results_summary =
                                build_goal_task_results_summary(&completed_tasks, fallback_summary);

                            // Check for partial success metadata in the goal context
                            let partial_info = final_goal
                                .as_ref()
                                .ok()
                                .and_then(|g| g.as_ref())
                                .and_then(|g| g.context.as_deref())
                                .and_then(|ctx| serde_json::from_str::<serde_json::Value>(ctx).ok())
                                .filter(|v| {
                                    v.get("partial_success")
                                        .and_then(|p| p.as_bool())
                                        .unwrap_or(false)
                                });

                            if let Some(summary) = partial_info {
                                let completed = summary
                                    .get("completed")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let failed =
                                    summary.get("failed").and_then(|v| v.as_u64()).unwrap_or(0);
                                let blocked =
                                    summary.get("blocked").and_then(|v| v.as_u64()).unwrap_or(0);
                                let total =
                                    summary.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                                (
                                    "failed",
                                    format!(
                                        "Goal partially completed ({}/{} tasks succeeded, {} failed, {} blocked):\n\n{}",
                                        completed,
                                        total,
                                        failed,
                                        blocked,
                                        task_results_summary.chars().take(4000).collect::<String>()
                                    ),
                                )
                            } else {
                                (
                                    "completed",
                                    format!(
                                        "Goal completed:\n\n{}",
                                        task_results_summary.chars().take(4000).collect::<String>()
                                    ),
                                )
                            }
                        }
                    }
                    "failed" => (
                        "failed",
                        format!(
                            "Goal failed: {}",
                            build_goal_failure_summary(
                                final_goal.as_ref().ok().and_then(|g| g.as_ref()),
                                &completed_tasks,
                                task_lead_response.as_deref(),
                                task_lead_error.as_deref(),
                            )
                        ),
                    ),
                    "cancelled" => ("completed", "Goal was cancelled.".to_string()),
                    "stalled" => (
                        "failed",
                        format!(
                            "Goal stalled (no progress after 3 dispatch cycles): {}",
                            goal_id
                        ),
                    ),
                    _ => (
                        "failed",
                        format!(
                            "Goal failed: {}",
                            build_goal_failure_summary(
                                final_goal.as_ref().ok().and_then(|g| g.as_ref()),
                                &completed_tasks,
                                task_lead_response.as_deref(),
                                task_lead_error.as_deref(),
                            )
                        ),
                    ),
                }
            };

            let msg = if is_scheduled_run {
                crate::channels::present_scheduled_run_notification(notification_type, &msg, false)
            } else {
                crate::channels::present_notification(notification_type, &msg)
            };
            let entry = crate::traits::NotificationEntry::new(
                &goal_id,
                &session_id,
                notification_type,
                &msg,
            );
            let notification_id = entry.id.clone();
            let enqueue_result = if continuous_controller {
                state.enqueue_notification(&entry).await.map(|()| true)
            } else {
                state.enqueue_goal_notification(&entry).await
            };
            match enqueue_result {
                Ok(true) => {
                    let local_hour = chrono::Timelike::hour(&chrono::Local::now());
                    if !entry.should_deliver_at_local_hour(local_hour) {
                        // Persisted successfully; normal notification delivery
                        // will resume outside quiet hours.
                        if let Some(ref registry) = goal_token_registry {
                            registry.remove(&goal_id).await;
                        }
                        return;
                    }
                    // Attempt immediate delivery — if it fails, heartbeat will retry from queue.
                    match agent
                        .deliver_parent_text_result_to_surface(
                            hub.as_ref(),
                            &session_id,
                            terminal_surface_id.as_deref(),
                            &msg,
                            parent_delivery::ParentDeliveryKind::GoalNotification,
                        )
                        .await
                    {
                        Ok(outcome) if outcome.sent => {
                            let _ = state.mark_notification_delivered(&notification_id).await;

                            // Auto-send any files referenced in the completion message
                            let file_paths = extract_file_paths_from_text(&msg);
                            if let Some(hub_weak) = &hub {
                                if let Some(hub_arc) = hub_weak.upgrade() {
                                    for path in file_paths {
                                        let filename = std::path::Path::new(&path)
                                            .file_name()
                                            .map(|n| n.to_string_lossy().to_string())
                                            .unwrap_or_else(|| "file".to_string());
                                        let media = crate::types::MediaMessage {
                                            session_id: session_id.clone(),
                                            caption: filename.clone(),
                                            kind: crate::types::MediaKind::Document {
                                                file_path: path.clone(),
                                                filename,
                                            },
                                            // Fire-and-forget: no delivery receipt awaited.
                                            result_tx: None,
                                        };
                                        if let Err(e) =
                                            hub_arc.send_media(&session_id, &media).await
                                        {
                                            warn!("Failed to auto-send goal file {}: {}", path, e);
                                        }
                                    }
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(err) => {
                            warn!(
                                session_id = %session_id,
                                notification_id = %notification_id,
                                error = %err,
                                "Failed to record parent-mediated goal notification"
                            );
                        }
                    }
                }
                Ok(false) => {
                    info!(
                        goal_id = %goal_id,
                        "Terminal notification was already claimed by another worker"
                    );
                }
                Err(err) => {
                    warn!(
                        goal_id = %goal_id,
                        error = %err,
                        "Failed to atomically enqueue terminal notification; attempting direct delivery"
                    );
                    match agent
                        .deliver_parent_text_result_to_surface(
                            hub.as_ref(),
                            &session_id,
                            terminal_surface_id.as_deref(),
                            &msg,
                            parent_delivery::ParentDeliveryKind::GoalNotification,
                        )
                        .await
                    {
                        Ok(outcome) if outcome.sent => {}
                        Ok(_) => warn!(
                            goal_id = %goal_id,
                            "Direct terminal-notification fallback had no delivery route"
                        ),
                        Err(delivery_error) => warn!(
                            goal_id = %goal_id,
                            %delivery_error,
                            "Direct terminal-notification fallback also failed"
                        ),
                    }
                }
            }

            // Clean up cancellation token
            if let Some(ref registry) = goal_token_registry {
                registry.remove(&goal_id).await;
            }
        }; // end of `body`

        body.await;

        // Run-budget state belongs to the whole scheduled cycle, not to any
        // individual task-lead/executor turn. Clear it exactly once after the
        // body has finalized all required work and the trigger task. The marker
        // stays false for duplicate spawns that returned before acquiring the
        // per-goal run guard, so they cannot disrupt the active owner.
        if finalize_scheduled_run.load(Ordering::Acquire) {
            let run_still_open = teardown_state
                .get_current_goal_run(&teardown_goal_id)
                .await
                .ok()
                .flatten()
                .is_some_and(|run| run.status == "running");
            if !run_still_open {
                if let Some(registry) = teardown_goal_token_registry.as_ref() {
                    registry.clear_run_budget(&teardown_goal_id).await;
                }
                clear_scheduled_run_state(&teardown_state, &teardown_goal_id).await;
            } else {
                tracing::debug!(
                    goal_id = %teardown_goal_id,
                    "Preserving scheduled-run budget and health for durable continuation"
                );
            }
        }

        // Self-correction bridge teardown (3c P3b.3): clear any correction-
        // execution context registered under this goal id. Idempotent — for the
        // overwhelming majority of (non-remediation) goals the key was never
        // registered, so this is a no-op; for a dispatched remediation it removes
        // the context now that the task lead (and its executors) have finished,
        // so contexts don't linger until bounded FIFO eviction.
        teardown_agent
            .clear_correction_context(&teardown_goal_id)
            .await;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{Goal, Intention, MandateAuthority, MandateDecisionCycle, Task};

    async fn mandate_test_state() -> (Arc<dyn StateStore>, tempfile::NamedTempFile) {
        let database = tempfile::NamedTempFile::new().unwrap();
        let store: Arc<dyn StateStore> = Arc::new(
            SqliteStateStore::new(
                database.path().to_str().unwrap(),
                100,
                None,
                Arc::new(EmbeddingService::new().unwrap()),
            )
            .await
            .unwrap(),
        );
        (store, database)
    }

    fn due_mandate_controller(session_id: &str) -> (Goal, Mandate) {
        let goal = Goal::new_continuous(
            "Steward an account",
            session_id,
            Some(10_000),
            Some(100_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Maintain a useful account presence",
            session_id,
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        (goal, mandate)
    }

    async fn claimed_mandate_run(
        state: &Arc<dyn StateStore>,
        goal: &Goal,
        mandate: &Mandate,
    ) -> crate::traits::GoalRun {
        state
            .create_mandate_controller(goal, mandate)
            .await
            .unwrap();
        assert_eq!(
            state
                .claim_due_mandates(1, "background-finalizer-test", 300)
                .await
                .unwrap()
                .len(),
            1
        );
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let run = state
            .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
            .await
            .unwrap();
        state
            .create_task(&Task {
                id: root_task_id.clone(),
                goal_id: goal.id.clone(),
                description: "Run one bounded mandate review".to_string(),
                status: "pending".to_string(),
                priority: goal.priority.clone(),
                task_order: 0,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: None,
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at: chrono::Utc::now().to_rfc3339(),
                started_at: None,
                completed_at: None,
            })
            .await
            .unwrap();
        let attempt = state
            .claim_task_with_lease(
                &root_task_id,
                "background-finalizer-test-root",
                Some("profile-task-lead"),
                300,
            )
            .await
            .unwrap()
            .unwrap();
        assert!(state
            .patch_task_from_attempt(
                &attempt.id,
                &attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("bounded review completed".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        run
    }

    #[test]
    fn task_lead_sqlite_busy_classifier_is_narrow() {
        assert!(is_sqlite_busy_error(&anyhow::anyhow!(
            "error returned from database: (code: 517) database is locked"
        )));
        assert!(is_sqlite_busy_error(&anyhow::anyhow!("SQLITE_BUSY")));
        assert!(!is_sqlite_busy_error(&anyhow::anyhow!(
            "task execution lease was lost"
        )));
    }

    #[tokio::test]
    async fn dispatch_trigger_promotion_moves_the_claimed_attempt_to_running() {
        let (state, _database) = mandate_test_state().await;
        let goal = Goal::new_finite("Coordinate synthetic work", "owner-session");
        state.create_goal(&goal).await.unwrap();
        state
            .start_goal_run(&goal.id, "manual", None, None)
            .await
            .unwrap();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Run a synthetic task lead".to_string(),
            status: "pending".to_string(),
            priority: goal.priority.clone(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&task).await.unwrap();
        let attempt = state
            .claim_task_with_lease(
                &task.id,
                "heartbeat-dispatch-synthetic",
                Some("profile-task-lead"),
                180,
            )
            .await
            .unwrap()
            .expect("trigger claim");

        assert!(
            promote_dispatch_trigger_attempt(&state, &attempt, "task-lead-synthetic")
                .await
                .unwrap()
        );

        let promoted_task = state.get_task(&task.id).await.unwrap().unwrap();
        assert_eq!(promoted_task.status, "running");
        assert_eq!(
            promoted_task.agent_id.as_deref(),
            Some("task-lead-synthetic")
        );
        let promoted_attempt = state
            .get_current_task_attempt(&task.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(promoted_attempt.status, "running");
        assert_eq!(promoted_attempt.worker_instance_id, "task-lead-synthetic");
    }

    #[tokio::test]
    async fn wait_finalization_keeps_controller_active_and_does_not_require_a_live_lease() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mandate) = due_mandate_controller("owner-session");
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let mut decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Wait,
            "No useful action is available yet",
            mandate.version,
        );
        decision.reconsider_at =
            Some((chrono::Utc::now() + chrono::Duration::minutes(10)).to_rfc3339());
        state
            .record_mandate_decision(&decision, None, None)
            .await
            .unwrap();
        let after_decision = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert!(after_decision.review_lease_token.is_none());

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await;

        assert!(notification.is_none());
        assert_eq!(
            state.get_goal(&goal.id).await.unwrap().unwrap().status,
            "active"
        );
        assert_eq!(
            state
                .get_mandate(&mandate.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            MandateStatus::Active
        );
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "completed"
        );
    }

    #[tokio::test]
    async fn completed_review_without_explicit_decision_is_a_retriable_failure() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mut mandate) = due_mandate_controller("owner-session");
        let expiry = chrono::Utc::now() + chrono::Duration::minutes(2);
        mandate.expires_at = Some(expiry.to_rfc3339());
        let run = claimed_mandate_run(&state, &goal, &mandate).await;

        assert!(persist_safe_wait_if_decision_missing(&state, &goal.id, &run.id).await);
        let decision = state
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(decision.outcome, MandateDecisionOutcome::Wait);
        let reconsider_at = chrono::DateTime::parse_from_rfc3339(
            decision
                .reconsider_at
                .as_deref()
                .expect("safe WAIT reconsideration"),
        )
        .unwrap()
        .with_timezone(&chrono::Utc);
        assert_eq!(reconsider_at, expiry);

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await
                .expect("safe fallback emits a durable review-failure notice");
        assert!(matches!(
            notification.kind,
            MandateRunNotificationKind::ReviewFailed {
                reason: MandateFinalizationRejectReason::DeliberatorFailed
            }
        ));
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "failed"
        );
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0]
                .outcome_summary
                .as_deref(),
            Some("mandate_review_failed:deliberator_failed")
        );
        let pending = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].notification_type, "mandate_review_failed");
    }

    #[tokio::test]
    async fn mandate_notifications_never_promote_generated_question_or_task_prose() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mandate) = due_mandate_controller("owner-session");
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let sentinel = "UNTRUSTED_QUESTION_DO_NOT_PROMOTE";
        let mut decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Ask,
            &format!("generated rationale {sentinel}"),
            mandate.version,
        );
        decision.question = Some(format!("generated question {sentinel}"));
        state
            .record_mandate_decision(&decision, None, None)
            .await
            .unwrap();

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await
                .expect("ASK emits a static control notification");
        let entry = notification.to_notification_entry();

        assert_eq!(entry.notification_type, "mandate_ask");
        assert_eq!(entry.priority, "critical");
        assert!(entry.expires_at.is_none());
        assert!(!entry.message.contains(sentinel));
        assert!(entry
            .message
            .contains("stored as untrusted mandate-local data"));
        assert!(entry.message.contains("manage_mandates(action=\"get\""));
        let pending = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, entry.id);
        assert_eq!(pending[0].message, entry.message);
        assert!(finalize_mandate_review(
            &state,
            &goal,
            &run.id,
            Some("completed"),
            &[],
            None,
            false,
        )
        .await
        .is_none());
        assert_eq!(state.get_pending_notifications(10).await.unwrap().len(), 1);
        let finalized = state.get_goal_runs(&goal.id).await.unwrap();
        let summary = finalized[0].outcome_summary.as_deref().unwrap_or_default();
        assert!(!summary.contains(sentinel));
        assert_eq!(summary, "mandate_non_action_satisfied");
    }

    #[tokio::test]
    async fn failed_review_without_a_decision_releases_lease_for_bounded_retry() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mandate) = due_mandate_controller("owner-session");
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let before = chrono::Utc::now();

        let notification = finalize_mandate_review(
            &state,
            &goal,
            &run.id,
            Some("failed"),
            &[],
            Some("deliberator exited"),
            false,
        )
        .await;

        let entry = notification.unwrap().to_notification_entry();
        assert_eq!(entry.notification_type, "mandate_review_failed");
        let pending = state.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, entry.id);
        assert_eq!(pending[0].message, entry.message);
        let retriable = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert!(retriable.review_lease_token.is_none());
        let retry_at = chrono::DateTime::parse_from_rfc3339(&retriable.next_review_at)
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(retry_at >= before + chrono::Duration::seconds(mandate.min_review_secs));
        assert!(retry_at <= chrono::Utc::now() + chrono::Duration::seconds(65));
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "failed"
        );
    }

    #[tokio::test]
    async fn policy_revision_suspends_an_act_intention_instead_of_satisfying_it() {
        let (state, _database) = mandate_test_state().await;
        let (goal, mut mandate) = due_mandate_controller("owner-session");
        mandate.authority = MandateAuthority {
            allowed_tools: vec!["http_request".to_string()],
            allowed_mutation_effects: vec!["external_delivery".to_string()],
            allowed_target_prefixes: vec!["https://api.example.test/v1/".to_string()],
            max_mutating_actions_per_cycle: 1,
            max_mutating_actions_per_rolling_24h: 1,
            min_seconds_between_mutations: 900,
            ..MandateAuthority::default()
        };
        let run = claimed_mandate_run(&state, &goal, &mandate).await;
        let decision = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Act,
            "A useful action is available",
            mandate.version,
        );
        let intention = Intention::new(
            &mandate.id,
            &decision.id,
            &run.id,
            "Publish one relevant update",
            "It advances the mandate",
        );
        state
            .record_mandate_decision(&decision, Some(&intention), None)
            .await
            .unwrap();
        let mut revised = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        revised.version += 1;
        revised.objective = "Revised owner policy".to_string();
        state.update_mandate(&revised).await.unwrap();

        let notification =
            finalize_mandate_review(&state, &goal, &run.id, Some("completed"), &[], None, false)
                .await;

        assert!(notification.is_none());
        let intentions = state.list_intentions(&mandate.id, 10).await.unwrap();
        assert_eq!(intentions.len(), 1);
        assert_eq!(
            intentions[0].status,
            crate::traits::IntentionStatus::Suspended
        );
        assert_eq!(
            state.get_goal_runs(&goal.id).await.unwrap()[0].status,
            "cancelled"
        );
    }

    #[test]
    fn heartbeat_backoff_starts_fast_then_grows() {
        // Quick early updates...
        assert_eq!(heartbeat_wait_secs(0), 15);
        assert_eq!(heartbeat_wait_secs(1), 30);
        // ...then exponential backoff...
        assert_eq!(heartbeat_wait_secs(2), 60);
        assert_eq!(heartbeat_wait_secs(3), 120);
        assert_eq!(heartbeat_wait_secs(4), 300);
        assert_eq!(heartbeat_wait_secs(5), 600);
        // ...settling at 15 minutes forever: long goals never go silent.
        assert_eq!(heartbeat_wait_secs(6), 900);
        assert_eq!(heartbeat_wait_secs(100), 900);
    }

    #[test]
    fn progress_count_ignores_trigger_and_cancelled_completion_rows() {
        let completed_task = |id: &str, description: &str| crate::traits::Task {
            id: id.to_string(),
            goal_id: "goal-1".to_string(),
            description: description.to_string(),
            status: "completed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
        };
        let trigger = completed_task("trigger-1", "Scheduled check");
        let mut completed = completed_task("work-1", "Write artifact");
        // Existing SQLite rows commonly deserialize a missing error as Some("").
        completed.error = Some(String::new());
        let mut cancelled = completed_task("work-2", "Publish artifact");
        cancelled.error = Some("Cancelled after stall".to_string());

        assert_eq!(
            count_successful_completed_work(&[trigger, completed, cancelled], Some("trigger-1")),
            1
        );
    }

    #[test]
    fn completed_work_wins_over_task_lead_error_for_stall_accounting() {
        assert_eq!(
            classify_continuous_dispatch_outcome(true, true, true, false),
            ContinuousDispatchOutcome::Progress
        );
        assert_eq!(
            classify_continuous_dispatch_outcome(false, true, false, false),
            ContinuousDispatchOutcome::ErrorWithoutProgress
        );
        assert_eq!(
            classify_continuous_dispatch_outcome(false, true, true, false),
            ContinuousDispatchOutcome::TransientInfrastructureFailure
        );
    }

    #[test]
    fn only_provider_infrastructure_failures_can_be_superseded_by_direct_recovery() {
        let failed_task = |error: &str| crate::traits::Task {
            id: "work-1".to_string(),
            goal_id: "goal-1".to_string(),
            description: "Publish synthetic artifact".to_string(),
            status: "failed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: Some(error.to_string()),
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
        };

        assert!(task_failed_only_due_to_provider_infrastructure(
            &failed_task(
                "LLM error: Codex stream failed: Our servers are currently overloaded. Please try again later."
            )
        ));
        assert!(!task_failed_only_due_to_provider_infrastructure(
            &failed_task("Verification failed: the public URL returned HTTP 404")
        ));
    }

    #[test]
    fn scheduled_dispatch_resolves_exact_repository_path() {
        let root = tempfile::tempdir().expect("project root");
        let blog = root.path().join("blog.aidaemon.ai");
        std::fs::create_dir_all(&blog).expect("blog directory");
        std::fs::write(blog.join("package.json"), r#"{"name":"synthetic-blog"}"#)
            .expect("project marker");

        let resolved = resolve_dispatch_project_scope(
            &format!(
                "Inspect {}, write one post, deploy it, and verify the public URL.",
                blog.display()
            ),
            &[root.path().to_string_lossy().to_string()],
        );
        assert_eq!(resolved.as_deref(), Some(blog.to_string_lossy().as_ref()));
    }

    #[test]
    fn scheduled_dispatch_resolves_exact_public_host_identity() {
        let root = tempfile::tempdir().expect("project root");
        let blog = root.path().join("blog.aidaemon.ai");
        let daemon = root.path().join("aidaemon");
        std::fs::create_dir_all(&blog).expect("blog directory");
        std::fs::create_dir_all(&daemon).expect("daemon directory");
        std::fs::write(blog.join("package.json"), r#"{"name":"synthetic-blog"}"#)
            .expect("blog marker");
        std::fs::write(
            daemon.join("Cargo.toml"),
            "[package]\nname='synthetic-daemon'\n",
        )
        .expect("daemon marker");

        let resolved = resolve_dispatch_project_scope(
            "Independently manage the aidaemon blog at https://blog.aidaemon.ai/: publish and verify it.",
            &[root.path().to_string_lossy().to_string()],
        );
        assert_eq!(resolved.as_deref(), Some(blog.to_string_lossy().as_ref()));
    }

    #[test]
    fn scheduled_dispatch_refuses_to_choose_between_distinct_exact_paths() {
        let root = tempfile::tempdir().expect("project root");
        let first = root.path().join("first-project");
        let second = root.path().join("second-project");
        std::fs::create_dir_all(&first).expect("first directory");
        std::fs::create_dir_all(&second).expect("second directory");

        let resolved = resolve_dispatch_project_scope(
            &format!(
                "Work in {} but never write in {}",
                first.display(),
                second.display()
            ),
            &[root.path().to_string_lossy().to_string()],
        );
        assert_eq!(resolved, None);
    }

    #[tokio::test]
    async fn scheduled_recovery_rebinds_persisted_run_state_to_its_real_root() {
        let (state, _database) = mandate_test_state().await;
        let goal = Goal::new_continuous(
            "Publish a synthetic site update",
            "synthetic-session",
            Some(400_000),
            Some(1_000_000),
        );
        state.create_goal(&goal).await.unwrap();
        let make_task = |id: &str, description: &str| Task {
            id: id.to_string(),
            goal_id: goal.id.clone(),
            description: description.to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };

        let old_task = make_task("old-root", "Initial scheduled attempt");
        state.create_task(&old_task).await.unwrap();
        let old_run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        state
            .finish_goal_run(&old_run.id, "failed", Some("synthetic failure"))
            .await
            .unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        state
            .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
                goal_id: goal.id.clone(),
                root_task_id: old_task.id,
                effective_budget_per_check: 800_000,
                tokens_used: 525_063,
                budget_extensions_count: 1,
                health: crate::traits::ScheduledRunHealth::default(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await
            .unwrap();

        let recovery_task = make_task("recovery-root", "Recover the failed scheduled attempt");
        state.create_task(&recovery_task).await.unwrap();
        let recovery_run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
        assert_eq!(
            recovery_run.root_task_id.as_deref(),
            Some(recovery_task.id.as_str())
        );

        rebind_scheduled_recovery_state(&state, None, &goal.id, &recovery_task.id).await;

        let rebound = state
            .get_scheduled_run_state(&goal.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(rebound.root_task_id, recovery_task.id);
        assert_eq!(rebound.tokens_used, 525_063);
        assert_eq!(rebound.effective_budget_per_check, 800_000);
        assert_eq!(rebound.budget_extensions_count, 1);
    }

    #[test]
    fn trigger_distinguishes_continuable_work_from_terminal_failure() {
        let task = |id: &str, status: &str, error: Option<&str>| crate::traits::Task {
            id: id.to_string(),
            goal_id: "goal-1".to_string(),
            description: format!("Required work {id}"),
            status: status.to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: error.map(str::to_string),
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
        };
        let completed = task("research", "completed", None);
        let failed = task(
            "publish",
            "failed",
            Some("per-run processing budget used 109976 / 100000 tokens"),
        );
        let mut superseded = failed.clone();
        superseded.status = "superseded".to_string();
        superseded.error = None;
        superseded.result =
            Some("Replaced by task 01234567, which completed the publish directly.".to_string());

        assert_eq!(
            dispatch_trigger_disposition(false, &[completed.clone(), failed.clone()]),
            DispatchTriggerDisposition::Failed
        );
        assert_eq!(
            dispatch_trigger_disposition(true, &[completed.clone(), failed.clone()]),
            DispatchTriggerDisposition::Failed
        );
        assert_eq!(
            dispatch_trigger_disposition(true, &[completed]),
            DispatchTriggerDisposition::Completed
        );
        assert_eq!(
            dispatch_trigger_disposition(true, &[task("review", "completed", None), superseded]),
            DispatchTriggerDisposition::Completed
        );
        assert_eq!(
            dispatch_trigger_disposition(true, &[task("publish", "pending", None)]),
            DispatchTriggerDisposition::Continuable
        );
        assert_eq!(
            dispatch_trigger_disposition(false, &[task("publish", "pending", None)]),
            DispatchTriggerDisposition::Continuable
        );
        assert_eq!(
            dispatch_trigger_disposition(true, &[]),
            DispatchTriggerDisposition::Completed
        );
        assert_eq!(
            dispatch_trigger_disposition(false, &[]),
            DispatchTriggerDisposition::Failed
        );
        assert!(failed_run_summary(&[failed], None).contains("109976 / 100000"));

        assert!(terminal_recovery_eligible(
            "finite",
            "failed",
            Some("failed"),
            false,
            &[]
        ));
        assert!(terminal_recovery_eligible(
            "continuous",
            "active",
            Some("failed"),
            true,
            &[]
        ));
        let mut prior_recovery = task("recovery", "failed", Some("still incomplete"));
        prior_recovery.context = Some(serde_json::json!({"terminal_recovery": true}).to_string());
        assert!(!terminal_recovery_eligible(
            "continuous",
            "active",
            Some("failed"),
            false,
            &[prior_recovery]
        ));
    }

    #[test]
    fn open_child_obligation_keeps_run_nonterminal_regardless_of_trigger_wording() {
        let mut pending = Goal::new_continuous("synthetic", "session", None, None);
        let task = crate::traits::Task {
            id: "pending-1".to_string(),
            goal_id: pending.id.clone(),
            description: "Finish the remaining verified work".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };

        for goal_status in ["active", "pending"] {
            pending.status = goal_status.to_string();
            assert_eq!(
                goal_run_terminal_status(&pending.status, Some("completed"), &[task.clone()]),
                None
            );
        }
    }

    #[test]
    fn durable_mandate_decision_is_semantic_success_after_child_error() {
        assert!(task_lead_semantically_succeeded(
            Some(TaskOutcome::Failed),
            true,
            true,
        ));
        assert!(task_lead_semantically_succeeded(None, true, true));
        assert!(!task_lead_semantically_succeeded(
            Some(TaskOutcome::Failed),
            true,
            false,
        ));
        assert!(!task_lead_semantically_succeeded(
            Some(TaskOutcome::Failed),
            false,
            true,
        ));
    }

    #[test]
    fn direct_recovery_requires_structured_and_persisted_success() {
        let mut task = crate::traits::Task {
            id: "recovery-1".to_string(),
            goal_id: "goal-1".to_string(),
            description: "Finish every unmet requirement".to_string(),
            status: "completed".to_string(),
            priority: "high".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("Verified final result".to_string()),
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 1,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
        };

        assert!(recovery_task_succeeded(TaskOutcome::Succeeded, Some(&task)));
        assert!(!recovery_task_succeeded(TaskOutcome::Partial, Some(&task)));
        assert!(!recovery_task_succeeded(TaskOutcome::Succeeded, None));

        task.status = "failed".to_string();
        task.error = Some("Build failed".to_string());
        assert!(!recovery_task_succeeded(
            TaskOutcome::Succeeded,
            Some(&task)
        ));
    }

    #[test]
    fn recovery_ledger_preserves_success_and_failure_evidence() {
        let task = |id: &str, status: &str, result: Option<&str>, error: Option<&str>| {
            crate::traits::Task {
                id: id.to_string(),
                goal_id: "goal-1".to_string(),
                description: format!("Step {id}"),
                status: status.to_string(),
                priority: "medium".to_string(),
                task_order: 0,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: None,
                result: result.map(str::to_string),
                error: error.map(str::to_string),
                blocker: None,
                idempotent: true,
                retry_count: 0,
                max_retries: 1,
                created_at: chrono::Utc::now().to_rfc3339(),
                started_at: None,
                completed_at: Some(chrono::Utc::now().to_rfc3339()),
            }
        };
        let ledger = render_recovery_task_ledger(&[
            task(
                "deploy",
                "completed",
                Some("Published https://example.workers.dev"),
                None,
            ),
            task(
                "research",
                "failed",
                Some("Direction was incorporated by implementation."),
                Some("Late database lock"),
            ),
        ]);

        assert!(ledger.contains("[completed] Step deploy"));
        assert!(ledger.contains("https://example.workers.dev"));
        assert!(ledger.contains("[failed] Step research"));
        assert!(ledger.contains("Late database lock"));
    }

    #[test]
    fn completed_recurring_run_keeps_goal_open_for_next_fire() {
        assert!(!run_completion_should_close_goal("continuous", true));
        assert!(run_completion_should_close_goal("finite", true));
        assert!(!run_completion_should_close_goal("finite", false));
    }

    #[test]
    fn recurring_run_always_has_an_explicit_terminal_outcome() {
        assert_eq!(
            recurring_run_terminal_outcome(Some("completed")),
            Some(RecurringRunTerminalOutcome::Completed)
        );
        assert_eq!(
            recurring_run_terminal_outcome(Some("failed")),
            Some(RecurringRunTerminalOutcome::Failed)
        );
        assert_eq!(recurring_run_terminal_outcome(Some("running")), None);
        assert_eq!(recurring_run_terminal_outcome(None), None);
    }
}
