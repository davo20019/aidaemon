//! A goal's task lead run in the background: [`spawn_background_task_lead`]
//! claims the goal's run, spawns its task lead (and, when the lead leaves
//! work behind, the executors and a recovery), keeps the owner's progress
//! surface current, finalizes mandate reviews and scheduled runs, and
//! delivers the result. It runs on any [`TurnHost`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::engine::turn_engine::TurnEngine;
use crate::host::TurnHost;
use crate::turn::notes::TurnNotes;
use aidaemon_work::scheduler::SchedulerStore;

use tracing::{error, info, warn};

use aidaemon_channels::context::ChannelContext;
use aidaemon_events::TaskOutcome;
use aidaemon_tools::semantics::UserRole;
use aidaemon_work::traits::{
    AgentRole, Mandate, MandateDecisionOutcome, MandateFinalizationRejectReason,
    MandateRunFinalizationRequest, MandateRunFinalizationResult, MandateRunNotification,
    MandateRunNotificationKind, MandateStatus,
};

use crate::goal_dispatch::{
    auto_dispatch_scheduled_run_extension_budget, build_goal_failure_summary,
    build_goal_task_results_summary, clear_scheduled_run_state, effective_goal_daily_budget,
    extract_file_paths_from_text, goal_has_scheduled_provenance, is_group_session,
    parse_goal_leading_wait, parse_wait_task_seconds, persist_scheduled_run_state,
    strip_leading_wait, truncate_goal_result_text, user_facing_task_description,
    SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS, SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
};
use crate::turn::notes as parent_delivery;

/// Progress-heartbeat wait schedule: quick early updates, then exponential
/// backoff settling at 15 minutes. Replaces the old hard cap of 4 messages,
/// which left long-running goals completely silent after ~2 minutes.
pub fn heartbeat_wait_secs(interval_count: u32) -> u64 {
    const SCHEDULE: [u64; 6] = [15, 30, 60, 120, 300, 600];
    SCHEDULE
        .get(interval_count as usize)
        .copied()
        .unwrap_or(900)
}

pub fn count_successful_completed_work(
    tasks: &[aidaemon_work::traits::Task],
    dispatch_trigger_task_id: Option<&str>,
) -> usize {
    tasks
        .iter()
        .filter(|task| Some(task.id.as_str()) != dispatch_trigger_task_id)
        .filter(|task| task.completed_successfully())
        .count()
}

pub async fn task_failed_only_due_to_provider_infrastructure(
    task: &aidaemon_work::traits::Task,
    event_store: &aidaemon_events::EventStore,
) -> bool {
    if task.status != "failed" {
        return false;
    }
    event_store
        .get_task_llm_stats(&task.id)
        .await
        .ok()
        .and_then(|summary| summary.last_provider_error_kind)
        .is_some_and(aidaemon_events::provider::ProviderErrorKind::is_infrastructure_failure)
}

/// Project roots to search when a mission names a project by name or site
/// hostname. When the operator configured none, the parent directory of the
/// execution workspace is the structural default: sibling projects of the
/// daemon's own checkout live there, whatever that directory is called.
pub fn effective_alias_roots(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    configured: &[String],
) -> Vec<String> {
    if !configured.is_empty() {
        return configured.to_vec();
    }
    let workspace = backend.workspace_root().to_string();
    std::path::Path::new(&workspace)
        .parent()
        .filter(|parent| parent.is_dir())
        .map(|parent| vec![parent.to_string_lossy().to_string()])
        .unwrap_or_default()
}

/// A workspace the goal/task was explicitly bound to (`context.project_scope`)
/// takes precedence over anything derived from mission text. The binding is
/// typed durable state written by `scheduled_goal_runs bind_workspace` or an
/// earlier recovery; it must be an existing absolute directory.
pub fn bound_project_scope_from_context(context: Option<&str>) -> Option<String> {
    crate::turn::project_scope::bound_workspace_from_goal_context(context)
}

/// A scope an earlier recovery carried without a binding behind it
/// (`context.inferred_project_scope`): it steers where the run works, and is
/// never read as a binding.
pub fn inferred_project_scope_from_context(context: Option<&str>) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(context?).ok()?;
    crate::turn::project_scope::typed_workspace_binding(
        value.get("inferred_project_scope")?.as_str(),
    )
}

/// The context of a terminal-recovery task. The run's binding stays under
/// `project_scope`, which the next dispatch reads as a binding; a dispatch
/// scope with no binding behind it goes under `inferred_project_scope`.
pub fn recovery_task_context(
    goal_run_id: Option<&str>,
    bound_scope: Option<&str>,
    dispatch_scope: Option<&str>,
    prior_report: Option<&str>,
) -> String {
    serde_json::json!({
        "recovery_for_run": goal_run_id,
        "terminal_recovery": true,
        "project_scope": bound_scope,
        "inferred_project_scope": dispatch_scope.filter(|scope| Some(*scope) != bound_scope),
        "prior_report": prior_report.map(|report| report.chars().take(2000).collect::<String>()),
    })
    .to_string()
}

pub fn resolve_dispatch_project_scope(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    mission: &str,
    alias_roots: &[String],
) -> Option<String> {
    let mut scopes = Vec::new();
    crate::turn::project_scope::extract_explicit_path_scopes_from_text(
        backend,
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
                aidaemon_tools::fs_utils::resolve_named_project_root(backend, host, alias_roots)
            {
                crate::turn::project_scope::push_project_scope(
                    &mut scopes,
                    scope.to_string_lossy().to_string(),
                    8,
                );
            }
        }
    }
    scopes.into_iter().next()
}

pub fn sqlite_busy_code(value: &str) -> bool {
    value
        .parse::<i32>()
        .ok()
        .is_some_and(|code| code & 0xff == 5)
}

pub fn is_sqlite_busy_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(sqlx::Error::Database(database)) = cause.downcast_ref::<sqlx::Error>()
            && database.code().is_some_and(|code| sqlite_busy_code(&code)) {
                return true;
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

pub async fn promote_dispatch_trigger_attempt(
    state: &Arc<dyn SchedulerStore>,
    attempt: &aidaemon_work::traits::TaskAttempt,
    worker_id: &str,
) -> anyhow::Result<bool> {
    const MAX_BUSY_RETRIES: usize = 5;
    let patch = aidaemon_work::traits::TaskAttemptPatch {
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

pub fn render_recovery_task_ledger(tasks: &[aidaemon_work::traits::Task]) -> String {
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
pub async fn release_duplicate_dispatch_trigger(
    state: &Arc<dyn SchedulerStore>,
    task_id: &str,
    attempt: Option<&aidaemon_work::traits::TaskAttempt>,
) {
    if let Some(attempt) = attempt {
        let patch = aidaemon_work::traits::TaskAttemptPatch {
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
pub struct AutoDispatchContext {
    agent: Arc<dyn TurnHost>,
    state: Arc<dyn SchedulerStore>,
    mission: String,
    goal_id: String,
    approval_session_id: String,
    channel_ctx: ChannelContext,
    user_role: UserRole,
    project_scope: Option<String>,
}

pub async fn execute_auto_dispatch_task(
    context: AutoDispatchContext,
    task: aidaemon_work::traits::Task,
) -> Option<(aidaemon_work::traits::Task, anyhow::Result<String>)> {
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
    let specialist = crate::specialists::compose::select_specialist_kind(
        AgentRole::Executor,
        &mission,
        &task.description,
    );
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
        let handoff = aidaemon_work::traits::TaskHandoff {
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
        let patch = aidaemon_work::traits::TaskAttemptPatch {
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
    let mut request =
        crate::spawn::ChildAgentRequest::new(mission, task_text, channel_ctx, user_role);
    request.child_role = Some(AgentRole::Executor);
    request.goal_id = Some(goal_id);
    request.task_id = Some(task_id);
    request.project_scope = project_scope;
    request.approval_session_id = Some(approval_session_id);
    let result = TurnEngine::new(&*agent)
        .spawn_child(request)
        .await
        .map(|run| run.response);
    Some((task, result))
}

pub async fn deliver_auto_dispatch_result(
    agent: &Arc<dyn TurnHost>,
    state: &Arc<dyn SchedulerStore>,
    hub: Option<&Weak<dyn aidaemon_channels::ports::OutboundRouter>>,
    session_id: &str,
    task: &aidaemon_work::traits::Task,
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

pub async fn tasks_for_current_run(
    state: &Arc<dyn SchedulerStore>,
    goal_id: &str,
    goal_run_id: Option<&str>,
) -> Vec<aidaemon_work::traits::Task> {
    match goal_run_id {
        Some(run_id) => state
            .get_tasks_for_goal_run(run_id)
            .await
            .unwrap_or_default(),
        None => state.get_tasks_for_goal(goal_id).await.unwrap_or_default(),
    }
}

pub async fn work_tasks_for_current_run(
    state: &Arc<dyn SchedulerStore>,
    goal_id: &str,
    goal_run_id: Option<&str>,
    dispatch_trigger_task_id: Option<&str>,
) -> Vec<aidaemon_work::traits::Task> {
    tasks_for_current_run(state, goal_id, goal_run_id)
        .await
        .into_iter()
        .filter(|task| Some(task.id.as_str()) != dispatch_trigger_task_id)
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurringRunTerminalOutcome {
    Completed,
    Failed,
}

pub fn recurring_run_terminal_outcome(
    run_status: Option<&str>,
) -> Option<RecurringRunTerminalOutcome> {
    match run_status {
        Some("completed") => Some(RecurringRunTerminalOutcome::Completed),
        Some("failed" | "blocked") => Some(RecurringRunTerminalOutcome::Failed),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchTriggerDisposition {
    Completed,
    Continuable,
    Failed,
}

pub fn dispatch_trigger_disposition(
    task_lead_succeeded: bool,
    work_tasks: &[aidaemon_work::traits::Task],
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
        .all(aidaemon_work::traits::Task::satisfies_run_completion)
    {
        DispatchTriggerDisposition::Completed
    } else {
        // The task lead has durably handed off unfinished work. Its own turn is
        // complete, while the enclosing goal run stays open for continuation.
        DispatchTriggerDisposition::Continuable
    }
}

pub fn goal_run_terminal_status(
    goal_status: &str,
    trigger_status: Option<&str>,
    work_tasks: &[aidaemon_work::traits::Task],
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
            .all(aidaemon_work::traits::Task::satisfies_run_completion)
    {
        Some("completed")
    } else {
        None
    }
}

/// Whether this dispatch closed its run's work. Child rows that all completed
/// close it; so does a task lead that finished the mission itself, which
/// leaves no child rows and only its committed `completed` trigger as the
/// receipt.
pub fn dispatch_closed_run(
    trigger_completed: bool,
    work_tasks: &[aidaemon_work::traits::Task],
) -> bool {
    work_tasks
        .iter()
        .all(aidaemon_work::traits::Task::satisfies_run_completion)
        && (trigger_completed || !work_tasks.is_empty())
}

pub fn task_lead_semantically_succeeded(
    task_lead_outcome: Option<TaskOutcome>,
    is_mandate_run: bool,
    mandate_decision_ready: bool,
) -> bool {
    task_lead_outcome == Some(TaskOutcome::Succeeded) || (is_mandate_run && mandate_decision_ready)
}

pub async fn patch_task_attempt_with_transient_retry(
    state: &Arc<dyn SchedulerStore>,
    attempt: &aidaemon_work::traits::TaskAttempt,
    patch: &aidaemon_work::traits::TaskAttemptPatch,
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

pub fn clear_partial_success_context(goal: &mut aidaemon_work::traits::Goal) {
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

/// A task lead that handed its work to the background finishes `Partial`:
/// its run is still going, so its card must not call it complete.
pub fn scheduled_run_progress(
    task_lead_outcome: Option<TaskOutcome>,
) -> aidaemon_channels::presentation::ScheduledRunProgress {
    if task_lead_outcome == Some(TaskOutcome::Partial) {
        aidaemon_channels::presentation::ScheduledRunProgress::StillRunning
    } else {
        aidaemon_channels::presentation::ScheduledRunProgress::Finished
    }
}

pub fn recovery_task_succeeded(
    outcome: TaskOutcome,
    persisted: Option<&aidaemon_work::traits::Task>,
) -> bool {
    outcome == TaskOutcome::Succeeded
        && persisted.is_some_and(aidaemon_work::traits::Task::completed_successfully)
}

pub fn is_terminal_recovery_task(task: &aidaemon_work::traits::Task) -> bool {
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

pub async fn rebind_scheduled_recovery_state(
    state: &Arc<dyn SchedulerStore>,
    registry: Option<&aidaemon_work::goal_tokens::GoalTokenRegistry>,
    goal_id: &str,
    recovery_task_id: &str,
) {
    if let Some(registry) = registry
        && let Some(status) = registry.get_run_budget(goal_id).await {
            persist_scheduled_run_state(state, goal_id, Some(recovery_task_id), &status).await;
            return;
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

pub fn terminal_recovery_eligible(
    goal_type: &str,
    goal_status: &str,
    run_status: Option<&str>,
    already_notified: bool,
    tasks: &[aidaemon_work::traits::Task],
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

pub fn failed_run_summary(tasks: &[aidaemon_work::traits::Task], fallback: Option<&str>) -> String {
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

pub async fn keep_mandate_controller_open(
    state: &Arc<dyn SchedulerStore>,
    mandate: &Mandate,
    goal: &aidaemon_work::traits::Goal,
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
pub async fn persist_safe_wait_if_decision_missing(
    state: &Arc<dyn SchedulerStore>,
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
    let decision = aidaemon_work::traits::MandateDecisionCycle::runtime_fallback_wait(
        &mandate,
        goal_run_id,
        chrono::Utc::now(),
    );
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
pub async fn finalize_mandate_review(
    state: &Arc<dyn SchedulerStore>,
    goal: &aidaemon_work::traits::Goal,
    goal_run_id: &str,
    _trigger_status: Option<&str>,
    _run_tasks: &[aidaemon_work::traits::Task],
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
        .with_objective(Some(&mandate.objective))
    };
    match proof {
        MandateRunFinalizationResult::ActSatisfied { counts } => {
            keep_mandate_controller_open(state, &mandate, goal).await;
            Some(notice(MandateRunNotificationKind::ActSatisfied, counts))
        }
        MandateRunFinalizationResult::NonActionSatisfied { outcome, counts } => match outcome {
            MandateDecisionOutcome::Wait => {
                keep_mandate_controller_open(state, &mandate, goal).await;
                // A WAIT carrying its granted owner alert was committed with
                // an alert notice; hand it on for prompt delivery.
                let alert = state
                    .get_mandate_decision_for_run(goal_run_id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|decision| decision.owner_alert);
                alert.map(|alert| {
                    notice(MandateRunNotificationKind::Alert, counts).with_owner_alert(Some(&alert))
                })
            }
            MandateDecisionOutcome::Ask => {
                let question = state
                    .get_mandate_decision_for_run(goal_run_id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|decision| decision.question);
                Some(
                    notice(MandateRunNotificationKind::Ask, counts)
                        .with_owner_question(question.as_deref()),
                )
            }
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
            MandateFinalizationRejectReason::InvalidDecisionState => Some(notice(
                MandateRunNotificationKind::ReviewFailed { reason },
                Default::default(),
            )),
            // Retried reviews: the finalizer queued a notice only for the
            // failures the owner hears about; deliver no other.
            MandateFinalizationRejectReason::DecisionMissing
            | MandateFinalizationRejectReason::DeliberatorFailed
            | MandateFinalizationRejectReason::ActChangedNothing => {
                let failures = state
                    .get_goal(&goal.id)
                    .await
                    .ok()
                    .flatten()
                    .map_or(1, |goal| i64::from(goal.dispatch_failures));
                aidaemon_work::traits::review_failure_is_notified(failures).then(|| {
                    notice(
                        MandateRunNotificationKind::ReviewFailed { reason },
                        Default::default(),
                    )
                })
            }
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuousDispatchOutcome {
    Progress,
    TransientInfrastructureFailure,
    ErrorWithoutProgress,
    Blocked,
    NoProgress,
}

pub fn classify_continuous_dispatch_outcome(
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

pub fn run_completion_should_close_goal(goal_type: &str, all_done: bool) -> bool {
    all_done && goal_type == "finite"
}

/// Emit a progress update onto a SINGLE self-editing surface: the first update
/// creates a tracked message; subsequent updates edit it in place. Collapses the
/// heartbeat into one updating message instead of a stream of pings. Falls back
/// to a fresh tracked send if the channel can't edit (e.g. the message aged out).
pub async fn emit_progress_surface(
    hub: &Option<Weak<dyn aidaemon_channels::ports::OutboundRouter>>,
    session: &str,
    surface_id: &mut Option<String>,
    text: &str,
) {
    let local_hour = chrono::Timelike::hour(&chrono::Local::now());
    if !aidaemon_work::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(local_hour)
    {
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
    agent: Arc<dyn TurnHost>,
    goal: aidaemon_work::traits::Goal,
    user_text: String,
    session_id: String,
    channel_ctx: ChannelContext,
    user_role: UserRole,
    state: Arc<dyn SchedulerStore>,
    hub: Option<Weak<dyn aidaemon_channels::ports::OutboundRouter>>,
    goal_token_registry: Option<aidaemon_work::goal_tokens::GoalTokenRegistry>,
    dispatch_trigger_task_id: Option<String>,
    // When the caller already posted a tracked "starting" message (e.g. the
    // scheduled-run "🔄 Running scheduled task" announcement), pass its id here so
    // the progress heartbeat edits THAT message in place instead of posting a new
    // one — folding the announcement and progress into one self-updating surface.
    initial_surface_id: Option<String>,
) {
    tokio::spawn(async move {
        // Teardown handles. The body below has many early `return`s, so the
        // existing logic is wrapped in an inner future; whichever way it exits,
        // the scheduled-run teardown below runs afterward.
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
            let trigger_task_context =
                if let Some(trigger_task_id) = dispatch_trigger_task_id.as_deref() {
                    state
                        .get_task(trigger_task_id)
                        .await
                        .ok()
                        .flatten()
                        .and_then(|task| task.context)
                } else {
                    None
                };
            // Only a binding is saved back as one: a scope inferred from the
            // mission (or carried by an earlier recovery) steers where the
            // run works but is never promoted to `project_scope`.
            let bound_dispatch_scope =
                bound_project_scope_from_context(trigger_task_context.as_deref());
            let dispatch_project_scope = bound_dispatch_scope
                .clone()
                .or_else(|| inferred_project_scope_from_context(trigger_task_context.as_deref()))
                .or_else(|| {
                    resolve_dispatch_project_scope(
                        agent.execution_backend(),
                        &mission,
                        &effective_alias_roots(agent.execution_backend(), agent.project_roots()),
                    )
                });

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
                            let activity = aidaemon_work::traits::TaskActivity {
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
                            .filter(|t| Some(t.id.as_str()) != heartbeat_trigger_task_id.as_deref())
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
                        if let Ok(Some(mut g)) = state.get_goal(&goal_id).await
                            && (g.status == "active" || g.status == "pending") {
                                g.status = "completed".to_string();
                                g.completed_at = Some(now.clone());
                                g.updated_at = now.clone();
                                let _ = state.update_goal(&g).await;
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
                                if Some(task.id.as_str()) == dispatch_trigger_task_id.as_deref()
                                    && let Some(attempt) = dispatch_attempt.as_ref() {
                                        let handoff = aidaemon_work::traits::TaskHandoff {
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
                                        let patch = aidaemon_work::traits::TaskAttemptPatch {
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

            let mut request = crate::spawn::ChildAgentRequest::new(
                effective_mission.clone(),
                effective_user_text.clone(),
                channel_ctx,
                fallback_user_role,
            );
            request.child_role = Some(AgentRole::TaskLead);
            request.goal_id = Some(goal_id.clone());
            request.task_id = dispatch_trigger_task_id.clone();
            // The resolved/bound objective workspace is the lead's project
            // scope; its tools and every executor it spawns inherit it.
            // Without this the lead and its children ran in the daemon's own
            // checkout.
            request.project_scope = dispatch_project_scope.clone();
            request.approval_session_id = Some(session_id.clone());
            request.bound_task_attempt = dispatch_attempt.clone();
            let task_lead_run = TurnEngine::new(&*agent).spawn_child(request).await;

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
            if is_mandate_run
                && let Err(error) = &task_lead_run {
                    warn!(
                        goal_id = %goal_id,
                        run_id = goal_run_id.as_deref().unwrap_or("missing"),
                        %error,
                        "Mandate deliberator exited before returning a semantic outcome"
                    );
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
                    let dispatchable: Vec<aidaemon_work::traits::Task> = all_tasks
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
                    let dispatch_batch: Vec<aidaemon_work::traits::Task> = dispatchable
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
                                if let Some(run_budget) = run_budget
                                    && run_budget.tokens_used
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
                            } else if let Some(budget_daily) =
                                effective_goal_daily_budget(&g, goal_token_registry.as_ref()).await
                                && g.tokens_used_today >= budget_daily {
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
                    if !task_failed_only_due_to_provider_infrastructure(task, agent.events()).await
                    {
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
            if let Some(ref trigger_task_id) = dispatch_trigger_task_id
                && let Ok(Some(trigger_task)) = state.get_task(trigger_task_id).await
                    && (trigger_task.status == "running" || trigger_task.status == "claimed") {
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
                            let handoff = aidaemon_work::traits::TaskHandoff {
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
                            let patch = aidaemon_work::traits::TaskAttemptPatch {
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
                let trigger_completed = match dispatch_trigger_task_id.as_deref() {
                    Some(trigger_task_id) => state
                        .get_task(trigger_task_id)
                        .await
                        .ok()
                        .flatten()
                        .is_some_and(|task| task.status == "completed"),
                    None => false,
                };
                let all_done = dispatch_closed_run(trigger_completed, &tasks);

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
                    // Kept beside the goal's own context keys (a reminder's
                    // message, the owner's name for it), not in their place.
                    let mut context = updated_goal
                        .context
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
                        .and_then(|value| value.as_object().cloned())
                        .unwrap_or_default();
                    if let Some(summary) = summary.as_object() {
                        context.extend(summary.clone());
                    }
                    updated_goal.context = Some(serde_json::Value::Object(context).to_string());
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
                        result
                            .as_ref()
                            .err()
                            .and_then(|error| {
                                error.downcast_ref::<crate::chat_bridge::ProviderError>()
                            })
                            .is_some_and(|error| error.kind.is_infrastructure_failure()),
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
                                    && let Err(error) = agent
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
            if let Some(run_id) = goal_run_id.as_deref()
                && let Some(run_status) = run_status {
                    let summary = build_goal_task_results_summary(
                        &final_run_tasks,
                        dispatch_trigger_task_id.as_deref(),
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
                            dispatch_trigger_task_id.as_deref(),
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
                        let msg =
                            aidaemon_channels::presentation::present_scheduled_run_notification(
                                notification_type,
                                &msg,
                                true,
                                scheduled_run_progress(task_lead_outcome),
                            );
                        let entry = aidaemon_work::traits::NotificationEntry::new(
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
                    if run_status.is_some()
                        && let Some(ref registry) = goal_token_registry {
                            registry.remove(&goal_id).await;
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
                if aidaemon_work::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(
                    local_hour,
                )
                    && let Some(hub_weak) = &hub
                        && let Some(hub_arc) = hub_weak.upgrade() {
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

                let now = chrono::Utc::now().to_rfc3339();
                let recovery_task_id = uuid::Uuid::new_v4().to_string();
                let recovery_task = aidaemon_work::traits::Task {
                    id: recovery_task_id.clone(),
                    goal_id: goal_id.clone(),
                    description: if user_text.starts_with("Directly recover and finish: ") {
                        user_text.to_string()
                    } else {
                        format!("Directly recover and finish: {user_text}")
                    },
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
                    context: Some(recovery_task_context(
                        goal_run_id.as_deref(),
                        bound_dispatch_scope.as_deref(),
                        dispatch_project_scope.as_deref(),
                        task_lead_response.as_deref(),
                    )),
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
                // Recovery is an orchestration root, so persist that role
                // explicitly before inserting the task. Task creation order
                // is not allowed to infer lifecycle identity.
                let _ = state
                    .start_goal_run(&goal_id, "recovery", None, Some(&recovery_task_id))
                    .await;
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
                        let mut request = crate::spawn::ChildAgentRequest::new(
                            user_text.clone(),
                            recovery_prompt,
                            fallback_channel_ctx,
                            fallback_user_role,
                        );
                        request.child_role = Some(AgentRole::Executor);
                        request.goal_id = Some(goal_id.clone());
                        request.task_id = Some(recovery_task_id.clone());
                        request.project_scope = dispatch_project_scope.clone();
                        request.specialist = Some("executor".to_string());
                        request.approval_session_id = Some(session_id.clone());
                        TurnEngine::new(&*agent).spawn_child(request).await
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
                            if let Some(trigger_task_id) = dispatch_trigger_task_id.as_deref()
                                && let Ok(Some(mut trigger)) = state.get_task(trigger_task_id).await
                                    && !trigger.satisfies_run_completion() {
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
                        if !aidaemon_work::goals::tasks_satisfy_goal_completion(&completed_tasks) {
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
                            let task_results_summary = build_goal_task_results_summary(
                                &completed_tasks,
                                dispatch_trigger_task_id.as_deref(),
                                fallback_summary,
                            );

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
                aidaemon_channels::presentation::present_scheduled_run_notification(
                    notification_type,
                    &msg,
                    false,
                    scheduled_run_progress(task_lead_outcome),
                )
            } else {
                aidaemon_channels::presentation::present_notification(notification_type, &msg)
            };
            let entry = aidaemon_work::traits::NotificationEntry::new(
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
                            if let Some(hub_weak) = &hub
                                && let Some(hub_arc) = hub_weak.upgrade() {
                                    for path in file_paths {
                                        let filename = std::path::Path::new(&path)
                                            .file_name()
                                            .map(|n| n.to_string_lossy().to_string())
                                            .unwrap_or_else(|| "file".to_string());
                                        let media = aidaemon_channels::hub::MediaMessage {
                                            session_id: session_id.clone(),
                                            caption: filename.clone(),
                                            kind: aidaemon_channels::hub::MediaKind::Document {
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
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_work::traits::Goal;

    fn resolve_dispatch_project_scope(mission: &str, alias_roots: &[String]) -> Option<String> {
        super::resolve_dispatch_project_scope(
            &*aidaemon_tools::testing::active_execution_backend(),
            mission,
            alias_roots,
        )
    }

    fn effective_alias_roots(configured: &[String]) -> Vec<String> {
        super::effective_alias_roots(
            &*aidaemon_tools::testing::active_execution_backend(),
            configured,
        )
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
        let completed_task = |id: &str, description: &str| aidaemon_work::traits::Task {
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
    fn direct_task_lead_completion_closes_the_run() {
        let task = |status: &str| aidaemon_work::traits::Task {
            id: format!("work-{status}"),
            goal_id: "goal-1".to_string(),
            description: "Publish post".to_string(),
            status: status.to_string(),
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
            completed_at: None,
        };
        // The task lead did the work itself: no child rows, committed trigger.
        assert!(dispatch_closed_run(true, &[]));
        // No child rows and no committed trigger is not a receipt.
        assert!(!dispatch_closed_run(false, &[]));
        assert!(dispatch_closed_run(false, &[task("completed")]));
        assert!(!dispatch_closed_run(true, &[task("pending")]));
        assert!(!dispatch_closed_run(true, &[task("failed")]));
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

    #[test]
    fn trigger_distinguishes_continuable_work_from_terminal_failure() {
        let task = |id: &str, status: &str, error: Option<&str>| aidaemon_work::traits::Task {
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
        let task = aidaemon_work::traits::Task {
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
                goal_run_terminal_status(
                    &pending.status,
                    Some("completed"),
                    std::slice::from_ref(&task)
                ),
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
        let mut task = aidaemon_work::traits::Task {
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
            aidaemon_work::traits::Task {
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

    #[test]
    fn bound_project_scope_requires_an_existing_absolute_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_string_lossy().to_string();
        let bound = serde_json::json!({ "project_scope": path }).to_string();
        assert_eq!(
            bound_project_scope_from_context(Some(&bound)),
            Some(path.clone())
        );
        let missing = serde_json::json!({ "project_scope": format!("{path}/absent") }).to_string();
        assert_eq!(bound_project_scope_from_context(Some(&missing)), None);
        let relative = serde_json::json!({ "project_scope": "projects/blog" }).to_string();
        assert_eq!(bound_project_scope_from_context(Some(&relative)), None);
        assert_eq!(bound_project_scope_from_context(Some("not json")), None);
        assert_eq!(bound_project_scope_from_context(None), None);
    }

    #[test]
    fn recovery_does_not_promote_an_inferred_scope_to_a_binding() {
        let dir = tempfile::tempdir().unwrap();
        let inferred = dir.path().to_string_lossy().into_owned();
        let context = recovery_task_context(Some("run-1"), None, Some(&inferred), Some("report"));
        assert_eq!(bound_project_scope_from_context(Some(&context)), None);
        assert_eq!(
            inferred_project_scope_from_context(Some(&context)),
            Some(inferred.clone())
        );
        // A real binding is carried as one.
        let context = recovery_task_context(None, Some(&inferred), Some(&inferred), None);
        assert_eq!(
            bound_project_scope_from_context(Some(&context)),
            Some(inferred.clone())
        );
        assert_eq!(inferred_project_scope_from_context(Some(&context)), None);
    }

    #[test]
    fn effective_alias_roots_prefer_configuration_then_workspace_parent() {
        let configured = vec!["/tmp/synthetic-projects".to_string()];
        assert_eq!(effective_alias_roots(&configured), configured);
        let derived = effective_alias_roots(&[]);
        let workspace = aidaemon_tools::testing::active_execution_backend()
            .workspace_root()
            .to_string();
        let parent = std::path::Path::new(&workspace)
            .parent()
            .map(|parent| parent.to_string_lossy().to_string());
        match parent.filter(|parent| std::path::Path::new(parent).is_dir()) {
            Some(parent) => assert_eq!(derived, vec![parent]),
            None => assert!(derived.is_empty()),
        }
    }
}
