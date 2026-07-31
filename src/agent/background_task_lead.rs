//! Background task-lead spawner extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Pure relocation — no logic changes. Houses the `spawn_background_task_lead`
//! free function (kept as a free fn to satisfy `Send` bounds for the spawned
//! background future).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use tracing::{info, warn};

use crate::events::TaskOutcome;
use crate::traits::AgentRole;
use crate::types::{ChannelContext, UserRole};

use super::parent_delivery;
use super::{
    auto_dispatch_scheduled_run_extension_budget, build_goal_failure_summary,
    build_goal_task_results_summary, clear_scheduled_run_state, effective_goal_daily_budget,
    extract_file_paths_from_text, goal_has_scheduled_provenance, is_group_session,
    is_low_signal_task_lead_reply, is_scheduled_task_description, parse_goal_leading_wait,
    parse_wait_task_seconds, persist_scheduled_run_state, strip_leading_wait,
    truncate_goal_result_text, user_facing_task_description, Agent,
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
        if !state
            .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
            .await
            .unwrap_or(false)
        {
            return;
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
            let _ = state.update_task(&task).await;
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
    hub: Option<&Weak<crate::channels::ChannelHub>>,
    session_id: &str,
    task: &crate::traits::Task,
    result: &anyhow::Result<String>,
) -> bool {
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

fn recurring_run_terminal_outcome(
    trigger_status: Option<&str>,
) -> Option<RecurringRunTerminalOutcome> {
    match trigger_status {
        Some("completed") => Some(RecurringRunTerminalOutcome::Completed),
        Some("failed") => Some(RecurringRunTerminalOutcome::Failed),
        _ => None,
    }
}

fn dispatch_trigger_succeeded(
    task_lead_succeeded: bool,
    work_tasks: &[crate::traits::Task],
) -> bool {
    if work_tasks.is_empty() {
        return task_lead_succeeded;
    }
    work_tasks
        .iter()
        .all(|task| task.satisfies_run_completion())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinuousDispatchOutcome {
    Progress,
    ErrorWithoutProgress,
    Blocked,
    NoProgress,
}

fn classify_continuous_dispatch_outcome(
    dispatch_made_progress: bool,
    task_lead_errored: bool,
    all_remaining_blocked: bool,
) -> ContinuousDispatchOutcome {
    if dispatch_made_progress {
        ContinuousDispatchOutcome::Progress
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
    hub: &Option<Weak<crate::channels::ChannelHub>>,
    session: &str,
    surface_id: &mut Option<String>,
    text: &str,
) {
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
    hub: Option<Weak<crate::channels::ChannelHub>>,
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
            let goal_run_id = state
                .get_current_goal_run(&goal_id)
                .await
                .ok()
                .flatten()
                .map(|run| run.id);
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
            let dispatch_project_scope = {
                let mut scopes = Vec::new();
                super::project_scope::extract_positive_explicit_path_scopes_from_text(
                    &mission,
                    &mut scopes,
                    8,
                    &agent.path_aliases.projects,
                );
                super::project_scope::unify_current_turn_scopes(&scopes)
            };

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
                    let bound = state
                        .bind_task_attempt_worker(
                            &attempt.id,
                            &attempt.lease_token,
                            &worker_id,
                            Some("profile-task-lead"),
                        )
                        .await
                        .unwrap_or(false);
                    let patch = crate::traits::TaskAttemptPatch {
                        status: "running".to_string(),
                        ..Default::default()
                    };
                    let running = bound
                        && state
                            .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
                            .await
                            .unwrap_or(false);
                    if !running {
                        warn!(
                            task_id = %trigger_task_id,
                            attempt_id = %attempt.id,
                            "Task-lead trigger lease was lost before execution"
                        );
                        return;
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
            if dispatch_trigger_task_id.is_some()
                && goal_has_scheduled_provenance(
                    &state,
                    &goal_id,
                    dispatch_trigger_task_id.as_deref(),
                )
                .await
            {
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
            let keepalive_handle = tokio::spawn(async move {
                let Some(task_id) = keepalive_task_id else {
                    let _ = keepalive_cancel_rx.await;
                    return;
                };
                loop {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(60)) => {
                            if let Some(attempt) = keepalive_attempt.as_ref() {
                                let renewed = keepalive_state
                                    .heartbeat_task_attempt(
                                        &attempt.id,
                                        &attempt.lease_token,
                                        180,
                                    )
                                    .await
                                    .unwrap_or(false);
                                if !renewed {
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
            let (heartbeat_cancel_tx, mut heartbeat_cancel_rx) =
                tokio::sync::oneshot::channel::<()>();
            let heartbeat_handle = tokio::spawn(async move {
                if is_group_channel {
                    // In group channels, just wait for cancellation — no progress spam
                    let _ = heartbeat_cancel_rx.await;
                    return;
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
                                "⏳ Still working on your request — planning the steps...",
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
                            .filter(|t| !is_scheduled_task_description(&t.description))
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
                            "⏳ Still working on it...".to_string()
                        } else if !in_progress.is_empty() {
                            format!(
                                "⏳ Progress: {}/{} steps done — working on: {}",
                                completed,
                                total,
                                in_progress.join(", ")
                            )
                        } else if completed == total || started > completed {
                            format!("⏳ Progress: {}/{} steps done", completed, total)
                        } else {
                            "⏳ Still working on your request...".to_string()
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
            });

            // Intercept pure wait/sleep goals to avoid spawning a full LLM task lead
            // just to orchestrate a timer.  For compound goals ("wait 2 minutes then
            // check disk space") we sleep first and then let the task lead handle the
            // remainder — but only if there actually IS a remainder after the wait.
            let effective_mission;
            let effective_user_text;
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
                    let _ = heartbeat_handle.await;
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
                                            "Timer elapsed without cancellation.".to_string()
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
                        .deliver_parent_text_result(
                            hub.as_ref(),
                            &session_id,
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

            let task_lead_run = agent
                .spawn_child_with_outcome(
                    &effective_mission,
                    &effective_user_text,
                    None,
                    channel_ctx,
                    fallback_user_role,
                    Some(AgentRole::TaskLead),
                    Some(goal_id.as_str()),
                    None,
                    None,
                    None, // arg_specialist (task lead spawn — not LLM-tool-selectable)
                    Some(&session_id),
                )
                .await;
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

            // Auto-dispatch: dispatch remaining pending tasks after task lead returns.
            // This handles both cases: LLMs that create tasks but don't spawn executors,
            // AND task leads that completed some tasks but left others pending.
            // Uses a loop to re-evaluate after each batch — completing a task may
            // unblock dependent tasks that weren't dispatchable in the previous pass.
            {
                let max_dispatch_rounds = 4; // safety limit — keep low to bound token usage
                const AUTO_DISPATCH_MAX_BUDGET_EXTENSIONS: usize = 0;
                const AUTO_DISPATCH_HARD_TOKEN_CAP: i64 = 2_000_000;
                let mut budget_exhausted = false;
                for _round in 0..max_dispatch_rounds {
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
                                                AUTO_DISPATCH_MAX_BUDGET_EXTENSIONS,
                                                AUTO_DISPATCH_HARD_TOKEN_CAP,
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
            let trigger_work_tasks = tasks_before_trigger_finalization
                .iter()
                .filter(|task| Some(task.id.as_str()) != dispatch_trigger_task_id.as_deref())
                .cloned()
                .collect::<Vec<_>>();
            if let Some(ref trigger_task_id) = dispatch_trigger_task_id {
                if let Ok(Some(trigger_task)) = state.get_task(trigger_task_id).await {
                    if trigger_task.status == "running" || trigger_task.status == "claimed" {
                        let success = dispatch_trigger_succeeded(
                            task_lead_outcome == Some(TaskOutcome::Succeeded),
                            &trigger_work_tasks,
                        );
                        let status = if success {
                            "completed".to_string()
                        } else {
                            "failed".to_string()
                        };
                        let error = if success {
                            None
                        } else {
                            Some(failed_run_summary(
                                &trigger_work_tasks,
                                result
                                    .as_ref()
                                    .err()
                                    .map(|error| error.to_string())
                                    .as_deref(),
                            ))
                        };
                        if let Some(attempt) = dispatch_attempt.as_ref() {
                            let handoff = crate::traits::TaskHandoff {
                                id: uuid::Uuid::new_v4().to_string(),
                                task_id: trigger_task_id.clone(),
                                attempt_id: attempt.id.clone(),
                                summary: if success {
                                    "Task-lead run completed.".to_string()
                                } else {
                                    error
                                        .clone()
                                        .unwrap_or_else(|| "Task-lead run failed.".to_string())
                                },
                                artifacts: Vec::new(),
                                verification: Vec::new(),
                                remaining_risk: error.clone(),
                                next_step: (!success)
                                    .then(|| "Inspect failed or blocked child tasks.".to_string()),
                                created_at: chrono::Utc::now().to_rfc3339(),
                            };
                            let patch = crate::traits::TaskAttemptPatch {
                                status,
                                error,
                                handoff: Some(handoff),
                                ..Default::default()
                            };
                            if !state
                                .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
                                .await
                                .unwrap_or(false)
                            {
                                warn!(
                                    task_id = %trigger_task_id,
                                    goal_id = %goal_id,
                                    "Ignored stale task-lead finalization"
                                );
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
            let _ = heartbeat_handle.await;
            let _ = keepalive_cancel_tx.send(());
            let _ = keepalive_handle.await;

            // Check the actual goal status from DB — the task lead may have already
            // set it via complete_goal/fail_goal. Only update if still "active".
            let current_goal = state.get_goal(&goal.id).await;
            let needs_status_update = match &current_goal {
                Ok(Some(g)) => g.status == "active" || g.status == "pending",
                _ => true, // fallback: update if we can't read
            };

            if needs_status_update {
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

                if all_done {
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
            if let Some(run_id) = goal_run_id.as_deref() {
                let run_status = if final_run_tasks.iter().any(|task| task.status == "blocked") {
                    Some("blocked")
                } else if matches!(status, "failed" | "stalled" | "cancelled")
                    || trigger_status.as_deref() == Some("failed")
                    || final_run_tasks.iter().any(|task| {
                        matches!(task.status.as_str(), "failed" | "interrupted" | "cancelled")
                    })
                {
                    Some("failed")
                } else if status == "completed" || trigger_status.as_deref() == Some("completed") {
                    Some("completed")
                } else {
                    None
                };
                if let Some(run_status) = run_status {
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
                let terminal_notification = match recurring_run_terminal_outcome(
                    trigger_status.as_deref(),
                ) {
                    Some(RecurringRunTerminalOutcome::Failed) => {
                        let task_lead_error = result.as_ref().err().map(|error| error.to_string());
                        Some((
                                "failed",
                                format!(
                                    "Scheduled run failed before completion; the recurring schedule remains active.\n\n{}",
                                    failed_run_summary(
                                        &final_run_tasks,
                                        task_lead_error.as_deref()
                                    )
                                ),
                            ))
                    }
                    Some(RecurringRunTerminalOutcome::Completed) => {
                        let summary = build_goal_task_results_summary(
                            &final_run_tasks,
                            "All required tasks completed.",
                        );
                        Some((
                                "completed",
                                format!(
                                    "Scheduled run completed successfully; the recurring schedule remains active.\n\n{}",
                                    truncate_goal_result_text(&summary, 3500)
                                ),
                            ))
                    }
                    None => None,
                };

                if let Some((notification_type, msg)) = terminal_notification {
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
                    match agent
                        .deliver_parent_text_result(
                            hub.as_ref(),
                            &session_id,
                            &msg,
                            parent_delivery::ParentDeliveryKind::GoalNotification,
                        )
                        .await
                    {
                        Ok(outcome) if outcome.sent => {
                            if queued {
                                let _ = state.mark_notification_delivered(&notification_id).await;
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
                    // Goal still in progress: optionally relay substantive task-lead text,
                    // but only if executor results haven't already been sent inline
                    // (which would cover the same content).
                    if !any_executor_results_sent {
                        if let Some(response) = task_lead_response.as_ref() {
                            if !is_low_signal_task_lead_reply(response) {
                                if let Err(err) = agent
                                    .deliver_parent_text_result(
                                        hub.as_ref(),
                                        &session_id,
                                        response,
                                        parent_delivery::ParentDeliveryKind::TaskLeadResult,
                                    )
                                    .await
                                {
                                    warn!(
                                        session_id = %session_id,
                                        error = %err,
                                        "Failed to record parent-mediated task-lead result"
                                    );
                                }
                            }
                        }
                    }
                }

                // Clean up cancellation token and return.
                if let Some(ref registry) = goal_token_registry {
                    registry.remove(&goal_id).await;
                }
                return;
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
            let (notification_type, msg) = if (status == "failed" || status == "stalled")
                && !goal_already_notified
                && final_goal
                    .as_ref()
                    .ok()
                    .and_then(|g| g.as_ref())
                    .map(|g| g.goal_type == "finite")
                    .unwrap_or(false)
            {
                info!(goal_id = %goal_id, "Finite goal failed — attempting durable direct recovery");

                // Prevent the heartbeat from racing a duplicate terminal notice
                // while a separately tracked recovery run is active.
                let _ = state.mark_goal_notified(&goal_id).await;
                if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                    g.status = "active".to_string();
                    g.completed_at = None;
                    g.notified_at = None;
                    g.updated_at = chrono::Utc::now().to_rfc3339();
                    let _ = state.update_goal(&g).await;
                }

                if let Some(hub_weak) = &hub {
                    if let Some(hub_arc) = hub_weak.upgrade() {
                        let _ = hub_arc
                            .send_text(
                                &session_id,
                                "The planned run did not complete. I am retrying it as a directly tracked recovery task...",
                            )
                            .await;
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
                    state
                        .get_current_goal_run(&goal_id)
                        .await
                        .ok()
                        .flatten()
                        .map(|run| run.id)
                } else {
                    None
                };
                let recovery_ledger = render_recovery_task_ledger(&final_run_tasks);
                let prior_report = task_lead_response
                    .as_deref()
                    .map(|report| report.chars().take(3000).collect::<String>())
                    .unwrap_or_else(|| "No usable prior summary was returned.".to_string());
                let recovery_prompt = format!(
                    "{user_text}\n\nRecover the whole mission, using the durable task ledger below as authoritative evidence. Preserve successful work and never repeat a completed external side effect. If the requested artifact, deployment, and verification are already proven by the ledger, report the mission complete without demanding a second verification surface. Only execute work that is genuinely unmet.\n\nExisting task ledger:\n{recovery_ledger}\n\nPrevious task-lead report:\n{prior_report}"
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
                            if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                                g.status = "completed".to_string();
                                g.completed_at = Some(chrono::Utc::now().to_rfc3339());
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
                                g.status = "failed".to_string();
                                g.completed_at = Some(chrono::Utc::now().to_rfc3339());
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
                            g.status = "failed".to_string();
                            g.completed_at = Some(chrono::Utc::now().to_rfc3339());
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

            let entry = crate::traits::NotificationEntry::new(
                &goal_id,
                &session_id,
                notification_type,
                &msg,
            );
            let notification_id = entry.id.clone();
            match state.enqueue_goal_notification(&entry).await {
                Ok(true) => {
                    // Attempt immediate delivery — if it fails, heartbeat will retry from queue.
                    match agent
                        .deliver_parent_text_result(
                            hub.as_ref(),
                            &session_id,
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
                        .deliver_parent_text_result(
                            hub.as_ref(),
                            &session_id,
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
            if let Some(registry) = teardown_goal_token_registry.as_ref() {
                registry.clear_run_budget(&teardown_goal_id).await;
            }
            clear_scheduled_run_state(&teardown_state, &teardown_goal_id).await;
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
            classify_continuous_dispatch_outcome(true, true, false),
            ContinuousDispatchOutcome::Progress
        );
        assert_eq!(
            classify_continuous_dispatch_outcome(false, true, false),
            ContinuousDispatchOutcome::ErrorWithoutProgress
        );
    }

    #[test]
    fn trigger_requires_every_created_work_task_to_succeed() {
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

        assert!(!dispatch_trigger_succeeded(
            false,
            &[completed.clone(), failed.clone()]
        ));
        assert!(!dispatch_trigger_succeeded(
            true,
            &[completed.clone(), failed.clone()]
        ));
        assert!(dispatch_trigger_succeeded(true, &[completed]));
        assert!(dispatch_trigger_succeeded(
            true,
            &[task("review", "completed", None), superseded]
        ));
        assert!(dispatch_trigger_succeeded(true, &[]));
        assert!(!dispatch_trigger_succeeded(false, &[]));
        assert!(failed_run_summary(&[failed], None).contains("109976 / 100000"));
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
