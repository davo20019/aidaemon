//! Background task-lead spawner extracted from `agent/mod.rs` (Phase 5 decoupling).
//!
//! Pure relocation — no logic changes. Houses the `spawn_background_task_lead`
//! free function (kept as a free fn to satisfy `Send` bounds for the spawned
//! background future).

use std::sync::{Arc, Weak};
use std::time::Duration;

use tracing::{info, warn};

use crate::traits::AgentRole;
use crate::types::{ChannelContext, UserRole};

use super::parent_delivery;
use super::{
    auto_dispatch_scheduled_run_extension_budget, build_goal_failure_summary,
    build_goal_task_results_summary, effective_goal_daily_budget, extract_file_paths_from_text,
    goal_completion_response_indicates_incomplete_work, goal_has_scheduled_provenance,
    is_group_session, is_low_signal_task_lead_reply, is_scheduled_task_description,
    parse_goal_leading_wait, parse_wait_task_seconds, persist_scheduled_run_state,
    salvageable_task_lead_result, strip_leading_wait, truncate_goal_result_text,
    user_facing_task_description, Agent,
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

        let body = async move {
            let goal_id = goal.id.clone();
            let mission = goal.description.clone();
            // Clone channel_ctx and user_role for potential direct fallback and auto-dispatch
            let fallback_channel_ctx = channel_ctx.clone();
            let dispatch_channel_ctx = channel_ctx.clone();
            let fallback_user_role = user_role;

            // Heartbeat dispatch claims a "trigger" task before spawning this background
            // lead. Keep it in "running" state (not "pending") so dispatch_pending_tasks
            // won't re-dispatch it on the next tick. The task lead will process it through
            // its normal flow (manage_goal_tasks / auto-dispatch).
            //
            // Previously this released the claim back to "pending", which created a race:
            // the next heartbeat tick would see the task as orphaned-pending and re-dispatch
            // it, causing duplicate execution (e.g., double tweet posts).
            if let Some(ref trigger_task_id) = dispatch_trigger_task_id {
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

            // Prevent duplicate concurrent task leads (and duplicate heartbeats) for the same goal.
            // Multiple codepaths can attempt to dispatch work for a goal (initial spawn, heartbeat
            // orphan recovery, auto-dispatch). This in-memory guard keeps progress messages sane
            // and avoids overlapping TaskLead runs.
            let _run_guard = if let Some(ref registry) = goal_token_registry {
                match registry.try_acquire_run(&goal_id) {
                    Some(g) => Some(g),
                    None => {
                        info!(
                            goal_id = %goal_id,
                            session_id = %session_id,
                            "Goal already has an active task lead; skipping duplicate background spawn"
                        );
                        return;
                    }
                }
            } else {
                None
            };

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
                    let tasks = heartbeat_state
                        .get_tasks_for_goal(&heartbeat_goal_id)
                        .await
                        .unwrap_or_default();
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
                        let completed = tasks
                            .iter()
                            .filter(|t| t.status == "completed" && t.error.is_none())
                            .count();
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
                    if let Ok(tasks) = state.get_tasks_for_goal(&goal_id).await {
                        for task in tasks {
                            if task.status != "completed"
                                && task.status != "failed"
                                && task.status != "cancelled"
                            {
                                let mut updated = task.clone();
                                updated.status = "completed".to_string();
                                updated.error = None;
                                updated.result = Some(msg.clone());
                                updated.completed_at = Some(now.clone());
                                let _ = state.update_task(&updated).await;
                            }
                        }
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

            let result = agent
                .spawn_child(
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
                )
                .await;

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

            // Auto-dispatch: dispatch remaining pending tasks after task lead returns.
            // This handles both cases: LLMs that create tasks but don't spawn executors,
            // AND task leads that completed some tasks but left others pending.
            // Uses a loop to re-evaluate after each batch — completing a task may
            // unblock dependent tasks that weren't dispatchable in the previous pass.
            {
                let max_dispatch_rounds = 4; // safety limit — keep low to bound token usage
                const AUTO_DISPATCH_MAX_BUDGET_EXTENSIONS: usize = 12;
                const AUTO_DISPATCH_HARD_TOKEN_CAP: i64 = 20_000_000;
                let mut budget_exhausted = false;
                for _round in 0..max_dispatch_rounds {
                    let all_tasks: Vec<crate::traits::Task> =
                        state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();

                    // Build set of completed task IDs for dependency checking
                    let completed_ids: std::collections::HashSet<String> = all_tasks
                        .iter()
                        .filter(|t| t.status == "completed" || t.status == "skipped")
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

                    for task in &dispatch_batch {
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

                        // Claim the task
                        let claimed = match state
                            .claim_task(&task.id, &format!("auto-dispatch-{}", goal_id))
                            .await
                        {
                            Ok(c) => c,
                            Err(_) => continue,
                        };
                        if !claimed {
                            continue;
                        }

                        // Execute pure wait tasks locally to avoid unnecessary LLM
                        // calls and provider rate-limit churn.
                        if let Some(wait_secs) = parse_wait_task_seconds(&task.description) {
                            info!(
                                goal_id = %goal_id,
                                task_id = %task.id,
                                wait_secs,
                                "Executing wait task locally"
                            );

                            // Keep the claimed task fresh so heartbeat stuck-task
                            // detection does not interrupt legitimate waits.
                            let mut remaining = wait_secs;
                            while remaining > 0 {
                                let step = remaining.min(60);
                                tokio::time::sleep(Duration::from_secs(step)).await;
                                remaining = remaining.saturating_sub(step);
                                if remaining > 0 {
                                    if let Ok(Some(mut claimed_task)) =
                                        state.get_task(&task.id).await
                                    {
                                        claimed_task.started_at =
                                            Some(chrono::Utc::now().to_rfc3339());
                                        claimed_task.status = "claimed".to_string();
                                        let _ = state.update_task(&claimed_task).await;
                                    }
                                }
                            }

                            if let Ok(Some(mut completed_task)) = state.get_task(&task.id).await {
                                completed_task.status = "completed".to_string();
                                completed_task.result =
                                    Some(format!("Waited for {} second(s).", wait_secs));
                                completed_task.error = None;
                                completed_task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                                let _ = state.update_task(&completed_task).await;
                            }
                            continue;
                        }

                        // Spawn executor
                        let exec_result = agent
                            .spawn_child(
                                &task.description,
                                &task.description,
                                None,
                                dispatch_channel_ctx.clone(),
                                fallback_user_role,
                                Some(AgentRole::Executor),
                                Some(goal_id.as_str()),
                                Some(task.id.as_str()),
                                None,
                                None, // arg_specialist (task-lead dispatch — not from LLM tool call)
                            )
                            .await;

                        let mut latest_task = state.get_task(&task.id).await.ok().flatten();
                        match exec_result {
                            Ok(response) => {
                                let delivery_text = if !response.trim().is_empty() {
                                    response.clone()
                                } else {
                                    latest_task
                                        .as_ref()
                                        .and_then(|task| {
                                            task.result
                                                .clone()
                                                .filter(|result| !result.trim().is_empty())
                                                .or_else(|| {
                                                    task.blocker.clone().filter(|blocker| {
                                                        !blocker.trim().is_empty()
                                                    })
                                                })
                                        })
                                        .unwrap_or_default()
                                };

                                if !delivery_text.trim().is_empty() {
                                    match agent
                                        .deliver_parent_text_result(
                                            hub.as_ref(),
                                            &session_id,
                                            &delivery_text,
                                            parent_delivery::ParentDeliveryKind::ExecutorResult,
                                        )
                                        .await
                                    {
                                        Ok(outcome) => {
                                            if outcome.sent {
                                                any_executor_results_sent = true;
                                            }
                                        }
                                        Err(err) => {
                                            warn!(
                                                session_id = %session_id,
                                                error = %err,
                                                "Failed to record parent-mediated executor result"
                                            );
                                        }
                                    }
                                }

                                if let Some(ref mut current_task) = latest_task {
                                    if current_task
                                        .result
                                        .as_deref()
                                        .is_none_or(|result| result.trim().is_empty())
                                        && !response.trim().is_empty()
                                    {
                                        current_task.result = Some(response);
                                        current_task.completed_at =
                                            Some(chrono::Utc::now().to_rfc3339());
                                        if !matches!(
                                            current_task.status.as_str(),
                                            "completed" | "blocked" | "failed"
                                        ) {
                                            current_task.status = "completed".to_string();
                                            current_task.blocker = None;
                                        }
                                        let _ = state.update_task(current_task).await;
                                    }
                                }
                            }
                            Err(e) => {
                                if let Some(ref mut current_task) = latest_task {
                                    if !matches!(
                                        current_task.status.as_str(),
                                        "completed" | "blocked" | "failed"
                                    ) {
                                        current_task.status = "failed".to_string();
                                        current_task.error = Some(e.to_string());
                                        current_task.completed_at =
                                            Some(chrono::Utc::now().to_rfc3339());
                                        let _ = state.update_task(current_task).await;
                                    }
                                } else {
                                    let mut updated = task.clone();
                                    updated.status = "failed".to_string();
                                    updated.error = Some(e.to_string());
                                    let _ = state.update_task(&updated).await;
                                }
                            }
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
            if let Some(ref trigger_task_id) = dispatch_trigger_task_id {
                if let Ok(Some(trigger_task)) = state.get_task(trigger_task_id).await {
                    if trigger_task.status == "running" || trigger_task.status == "claimed" {
                        let mut updated = trigger_task;
                        let success = result.is_ok();
                        updated.status = if success {
                            "completed".to_string()
                        } else {
                            "failed".to_string()
                        };
                        updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        if !success {
                            updated.error = result.as_ref().err().map(|e| e.to_string());
                        }
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

            // Stop the heartbeat
            let _ = heartbeat_cancel_tx.send(());
            let _ = heartbeat_handle.await;

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
                let completed_after = state
                    .count_completed_tasks_for_goal(&goal_id)
                    .await
                    .unwrap_or(0);

                let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();
                let all_done = !tasks.is_empty()
                    && tasks
                        .iter()
                        .all(|t| t.status == "completed" || t.status == "skipped");

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
                    // All tasks finished — goal is complete
                    updated_goal.status = "completed".to_string();
                    updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    updated_goal.dispatch_failures = 0;
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
                } else if result.is_err() {
                    // Task lead crashed — count as no progress
                    updated_goal.dispatch_failures += 1;
                    info!(
                        goal_id = %goal_id,
                        dispatch_failures = updated_goal.dispatch_failures,
                        "Task lead errored, incrementing dispatch_failures"
                    );
                } else if is_finite {
                    // Finite goal with some tasks completed but others remain.
                    // Since finite goals have no re-dispatch, mark as completed
                    // (partial success) rather than leaving it stuck.
                    let completed_count = tasks
                        .iter()
                        .filter(|t| t.status == "completed" && t.error.is_none())
                        .count();
                    let failed_count = tasks.iter().filter(|t| t.status == "failed").count();
                    let blocked_count = tasks.iter().filter(|t| t.status == "blocked").count();
                    let remaining = tasks
                        .iter()
                        .filter(|t| t.status != "completed" && t.status != "skipped")
                        .count();
                    updated_goal.status = "completed".to_string();
                    updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());

                    // Store completion summary in context for notification enrichment
                    if failed_count > 0 || blocked_count > 0 {
                        let summary = serde_json::json!({
                            "partial_success": true,
                            "completed": completed_count,
                            "failed": failed_count,
                            "blocked": blocked_count,
                            "total": tasks.len(),
                        });
                        updated_goal.context = Some(summary.to_string());
                    }
                    info!(
                        goal_id = %goal_id,
                        completed_count,
                        failed_count,
                        blocked_count,
                        remaining,
                        "Finite goal partially completed after dispatch"
                    );
                } else {
                    // Continuous goal: task lead returned Ok but tasks remain.
                    // Check if any tasks were completed recently during this dispatch.
                    let recently_completed = tasks.iter().any(|t| {
                        t.status == "completed"
                            && t.completed_at.as_ref().is_some_and(|ca| {
                                chrono::DateTime::parse_from_rfc3339(ca)
                                    .map(|dt| {
                                        let age =
                                            chrono::Utc::now() - dt.with_timezone(&chrono::Utc);
                                        age.num_minutes() < 30
                                    })
                                    .unwrap_or(false)
                            })
                    });

                    // Check if all remaining non-completed tasks are blocked
                    // (waiting on external input/dependencies). Blocked tasks are
                    // waiting, not failing — don't count as "no progress".
                    let all_remaining_blocked = tasks
                        .iter()
                        .filter(|t| t.status != "completed" && t.status != "skipped")
                        .all(|t| t.status == "blocked");

                    if recently_completed {
                        // Progress was made — reset failures
                        updated_goal.dispatch_failures = 0;
                    } else if all_remaining_blocked && !tasks.is_empty() {
                        // All remaining tasks are blocked — don't increment failures
                        info!(
                            goal_id = %goal_id,
                            blocked_tasks = tasks.iter().filter(|t| t.status == "blocked").count(),
                            "All remaining tasks are blocked — not incrementing dispatch_failures"
                        );
                    } else {
                        // No progress this cycle
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
                        if task.status == "pending" || task.status == "claimed" {
                            let mut t = task.clone();
                            t.status = "completed".to_string();
                            t.error = Some(
                                "Cancelled: goal stalled (no progress after 3 dispatch cycles)"
                                    .to_string(),
                            );
                            t.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            let _ = state.update_task(&t).await;
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
            // Only notify for terminal states — "active" means it's still in progress
            if status == "active" || status == "pending" {
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

                // Goal is still active, no notification needed.
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
                // Salvage path: if the task lead already produced a substantive,
                // finished answer, surface it instead of discarding the work and
                // re-running a fresh, context-free direct fallback (which wastes
                // tokens and — as observed — can lose a perfectly good result).
                if let Some(salvaged) = salvageable_task_lead_result(task_lead_response.as_deref())
                {
                    let _ = state.mark_goal_notified(&goal_id).await;
                    if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                        g.status = "completed".to_string();
                        g.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        g.updated_at = chrono::Utc::now().to_rfc3339();
                        let _ = state.update_goal(&g).await;
                    }
                    info!(
                        goal_id = %goal_id,
                        "Salvaged substantive task-lead result for failed/stalled finite goal"
                    );
                    if let Some(ref registry) = goal_token_registry {
                        registry.remove(&goal_id).await;
                    }
                    (
                        "completed",
                        format!(
                            "Goal completed: {}",
                            truncate_goal_result_text(&salvaged, 4000)
                        ),
                    )
                } else {
                    info!(goal_id = %goal_id, "Finite goal failed — attempting direct fallback");

                    // Mark as notified immediately to prevent the heartbeat from
                    // sending a duplicate "Goal failed" notification while the
                    // fallback is in progress.
                    let _ = state.mark_goal_notified(&goal_id).await;

                    // Notify user we're retrying with a different approach
                    if let Some(hub_weak) = &hub {
                        if let Some(hub_arc) = hub_weak.upgrade() {
                            let _ = hub_arc
                        .send_text(
                            &session_id,
                            "The task planner couldn't complete this. Let me try handling it directly...",
                        )
                        .await;
                        }
                    }

                    // Spawn a direct executor to handle the original request
                    // without goal/task decomposition
                    let fallback_result = agent
                        .spawn_child(
                            &user_text,
                            &user_text,
                            None,
                            fallback_channel_ctx,
                            fallback_user_role,
                            None, // no specific role — gets full tool access
                            None, // no goal_id — prevents goal re-entry
                            None,
                            None,
                            None, // arg_specialist (direct executor fallback — no LLM tool selection)
                        )
                        .await;

                    match fallback_result {
                        Ok(response)
                            if !response.trim().is_empty()
                                && !goal_completion_response_indicates_incomplete_work(
                                    &response,
                                ) =>
                        {
                            // Direct handling succeeded — update goal to completed
                            if let Ok(Some(mut g)) = state.get_goal(&goal_id).await {
                                g.status = "completed".to_string();
                                g.completed_at = Some(chrono::Utc::now().to_rfc3339());
                                g.updated_at = chrono::Utc::now().to_rfc3339();
                                let _ = state.update_goal(&g).await;
                            }
                            info!(goal_id = %goal_id, "Direct fallback succeeded");
                            (
                                "completed",
                                format!(
                                    "Goal completed: {}",
                                    truncate_goal_result_text(&response, 4000)
                                ),
                            )
                        }
                        Ok(response) if !response.trim().is_empty() => {
                            info!(
                                goal_id = %goal_id,
                                "Direct fallback returned an incomplete/unverified response"
                            );
                            (
                                "failed",
                                format!(
                            "I made some progress, but I couldn't verify the final outcome:\n\n{}",
                            truncate_goal_result_text(&response, 3500)
                        ),
                            )
                        }
                        _ => {
                            // Direct handling also failed — give detailed info
                            let tasks =
                                state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();
                            let task_summary: String = tasks
                                .iter()
                                .take(5)
                                .map(|t| {
                                    let err = t.error.as_deref().unwrap_or("no details");
                                    format!("• {} ({})", t.description, err)
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            info!(goal_id = %goal_id, "Direct fallback also failed");
                            (
                        "failed",
                        format!(
                            "I wasn't able to complete your request. Here's what I tried:\n{}\n\nYou could try rephrasing or breaking it into smaller steps.",
                            if task_summary.is_empty() {
                                "(no task details available)".to_string()
                            } else {
                                task_summary
                            }
                        ),
                    )
                        }
                    }
                }
            } else {
                let completed_tasks = state.get_tasks_for_goal(&goal_id).await.unwrap_or_default();
                let task_lead_error = result.as_ref().err().map(|e| e.to_string());
                match status {
                    "completed" => {
                        if any_executor_results_sent {
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
                                "completed",
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
            let _ = state.enqueue_notification(&entry).await;

            // Mark goal as notified so heartbeat doesn't double-enqueue
            let _ = state.mark_goal_notified(&goal_id).await;

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
                                if let Err(e) = hub_arc.send_media(&session_id, &media).await {
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

            // Clean up cancellation token
            if let Some(ref registry) = goal_token_registry {
                registry.remove(&goal_id).await;
            }
        }; // end of `body`

        body.await;

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
}
