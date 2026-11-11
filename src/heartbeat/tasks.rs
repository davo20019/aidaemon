//! Stuck, blocked, retried and orphaned tasks.

use super::*;

impl HeartbeatCoordinator {
    /// Detect tasks that have been running/claimed longer than the timeout and mark them interrupted.
    pub(super) async fn detect_stuck_tasks(&self) {
        match self.state.recover_expired_task_attempts().await {
            Ok(tasks) if !tasks.is_empty() => {
                warn!(
                    count = tasks.len(),
                    "Recovered tasks whose execution leases expired"
                );
            }
            Ok(_) => {}
            Err(error) => {
                error!(%error, "Failed to recover expired execution leases");
                return;
            }
        }
        let stuck = match self
            .state
            .get_stuck_tasks(self.task_inactivity_timeout_secs)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                error!(error = %e, "Failed to get stuck tasks");
                return;
            }
        };
        for task in &stuck {
            let last_activity = self
                .state
                .get_task_activities(&task.id)
                .await
                .ok()
                .and_then(|acts| acts.last().map(|a| a.created_at.clone()));
            let inactivity_secs = task_inactivity_secs(
                last_activity.as_deref(),
                task.started_at.as_deref().unwrap_or(&task.created_at),
                chrono::Utc::now(),
            );
            warn!(
                task_id = %task.id,
                goal_id = %task.goal_id,
                inactivity_secs,
                "Marking stuck task as interrupted"
            );
            if let Err(e) = self.state.mark_task_interrupted(&task.id).await {
                error!(task_id = %task.id, error = %e, "Failed to mark stuck task");
            }
        }
    }

    pub(super) async fn block_pending_tasks_with_terminal_failed_dependencies(&self) {
        let goals = match self.state.get_active_goals().await {
            Ok(g) => g,
            Err(e) => {
                error!(error = %e, "Failed to get active goals for pending dependency cleanup");
                return;
            }
        };

        let now = chrono::Utc::now().to_rfc3339();
        let terminal_failed = |status: &str| {
            matches!(
                status,
                "failed" | "blocked" | "interrupted" | "cancelled" | "abandoned"
            )
        };

        for goal in &goals {
            let tasks = match self.state.get_tasks_for_goal(&goal.id).await {
                Ok(t) => t,
                Err(e) => {
                    error!(goal_id = %goal.id, error = %e, "Failed to get tasks for pending dependency cleanup");
                    continue;
                }
            };
            let graph = match crate::traits::task_execution_graph(&tasks) {
                Ok(graph) => graph,
                Err(error) => {
                    error!(goal_id = %goal.id, %error, "Invalid task dependency graph; leaving tasks unclaimed");
                    continue;
                }
            };
            let by_id: std::collections::HashMap<&str, &crate::traits::Task> =
                tasks.iter().map(|t| (t.id.as_str(), t)).collect();

            for task in tasks.iter().filter(|t| t.status == "pending") {
                let Some(failed_dep) = graph
                    .unresolved_dependencies(&task.id)
                    .iter()
                    .filter_map(|dependency| by_id.get(dependency.id.as_str()))
                    .find(|dep| terminal_failed(&dep.status))
                else {
                    continue;
                };

                let mut updated = task.clone();
                updated.status = "blocked".to_string();
                updated.blocker = Some(format!(
                    "Dependency {} ended with status {} before this task could run.",
                    failed_dep.id, failed_dep.status
                ));
                updated.completed_at = Some(now.clone());
                if let Err(e) = self.state.update_task(&updated).await {
                    error!(
                        task_id = %task.id,
                        goal_id = %goal.id,
                        dependency_id = %failed_dep.id,
                        error = %e,
                        "Failed to block pending task with terminal failed dependency"
                    );
                } else {
                    warn!(
                        task_id = %task.id,
                        goal_id = %goal.id,
                        dependency_id = %failed_dep.id,
                        dependency_status = %failed_dep.status,
                        "Blocked pending task with terminal failed dependency"
                    );
                }
            }
        }
    }

    /// Auto-retry failed idempotent tasks that haven't exceeded their max retries.
    /// Resets them to "pending" so they get picked up by dispatch_pending_tasks.
    pub(super) async fn auto_retry_failed_tasks(&self) {
        let goals = match self.state.get_active_goals().await {
            Ok(g) => g,
            Err(e) => {
                error!(error = %e, "Failed to get active goals for auto-retry");
                return;
            }
        };

        let mut retried = 0;
        for goal in &goals {
            // Only retry within active orchestration goals.
            if goal.status != "active" {
                continue;
            }

            // The active task lead owns retries for its run. Promoting one of
            // its freshly-failed tasks back to pending here can race the lead's
            // own completion accounting and manufacture an orphaned run.
            if self
                .goal_token_registry
                .as_ref()
                .is_some_and(|registry| registry.is_run_active(&goal.id))
            {
                continue;
            }

            let tasks = match self.state.get_tasks_for_goal(&goal.id).await {
                Ok(t) => t,
                Err(_) => continue,
            };
            let schedules = self
                .state
                .get_schedules_for_goal(&goal.id)
                .await
                .unwrap_or_default();
            let scheduled_run = if schedules.is_empty() {
                None
            } else {
                self.state
                    .get_scheduled_run_state(&goal.id)
                    .await
                    .ok()
                    .flatten()
            };

            // Scheduled goals intentionally stay active between runs. Without
            // an active run record there is no retryable cycle, regardless of
            // whether old child tasks were individually marked idempotent.
            if !schedules.is_empty() && scheduled_run.is_none() {
                continue;
            }

            for task in &tasks {
                if scheduled_run
                    .as_ref()
                    .is_some_and(|run| !timestamp_is_at_or_after(&task.created_at, &run.created_at))
                {
                    continue;
                }
                if task.status == "failed" && task.idempotent && task.retry_count < task.max_retries
                {
                    let mut retry_task = task.clone();
                    retry_task.status = "pending".to_string();
                    retry_task.retry_count += 1;
                    retry_task.result = None;
                    retry_task.error = None;
                    retry_task.blocker = None;
                    retry_task.agent_id = None;
                    retry_task.started_at = None;
                    retry_task.completed_at = None;

                    if let Err(e) = self.state.update_task(&retry_task).await {
                        error!(task_id = %task.id, error = %e, "Failed to auto-retry failed task");
                    } else {
                        retried += 1;
                        info!(
                            task_id = %task.id,
                            goal_id = %goal.id,
                            retry = retry_task.retry_count,
                            max_retries = retry_task.max_retries,
                            "Auto-retried failed idempotent task"
                        );

                        // Notify user that a retry is in progress
                        let task_desc: String = task.description.chars().take(160).collect();
                        let goal_desc: String = goal.description.chars().take(120).collect();
                        let msg = format!(
                            "Retrying task (attempt {}/{}): {} (goal: {})",
                            retry_task.retry_count, retry_task.max_retries, task_desc, goal_desc
                        );
                        let entry = crate::traits::NotificationEntry::new(
                            &goal.id,
                            &goal.session_id,
                            "status_update",
                            &msg,
                        );
                        if let Err(error) = self.state.enqueue_notification(&entry).await {
                            warn!(
                                task_id = %task.id,
                                %error,
                                "Failed to enqueue task retry notification"
                            );
                        }
                    }
                }
            }
        }

        if retried > 0 {
            info!(count = retried, "Auto-retried failed idempotent tasks");
        }
    }

    /// Dispatch orphaned pending tasks by atomically claiming them and spawning task leads.
    ///
    /// For each active goal with pending tasks but no running agent:
    /// 1. Atomically claim a task via `claim_task` (prevents duplicate dispatch)
    /// 2. Spawn a background task lead gated by the semaphore
    /// 3. The task lead picks up all pending tasks for the goal
    ///
    /// Falls back to notification if no agent reference is available.
    pub(super) async fn dispatch_pending_tasks(&self) {
        let pending = match self.state.get_pending_tasks_by_priority(20).await {
            Ok(t) => t,
            Err(e) => {
                error!(error = %e, "Failed to get pending tasks");
                return;
            }
        };
        if pending.is_empty() {
            return;
        }

        // Group pending tasks by goal_id
        let mut goals_with_pending: std::collections::HashMap<String, Vec<&crate::traits::Task>> =
            std::collections::HashMap::new();
        for task in &pending {
            goals_with_pending
                .entry(task.goal_id.clone())
                .or_default()
                .push(task);
        }

        for (goal_id, tasks) in &goals_with_pending {
            let mut goal = match self.state.get_goal(goal_id).await {
                Ok(Some(g)) => g,
                _ => continue,
            };

            // Self-heal the impossible state produced by older run_now code:
            // a manual run with a pending root task attached to a stalled goal.
            // The task dispatcher normally filters non-active goals, so without
            // this invariant repair the run can say "running" forever despite
            // never having acquired a worker.
            if goal.status == "stalled" {
                let stranded_run = match self.state.get_current_goal_run(goal_id).await {
                    Ok(Some(run)) => {
                        let run_tasks = self
                            .state
                            .get_tasks_for_goal_run(&run.id)
                            .await
                            .unwrap_or_default();
                        stranded_manual_run_matches_pending_tasks(&run, &run_tasks, tasks)
                            .then_some(run)
                    }
                    _ => None,
                };
                if let Some(run) = stranded_run {
                    goal.status = "active".to_string();
                    goal.completed_at = None;
                    goal.updated_at = chrono::Utc::now().to_rfc3339();
                    if let Err(error) = self.state.update_goal(&goal).await {
                        error!(goal_id, %error, "Failed to reactivate stranded manual run");
                        continue;
                    }
                    warn!(
                        goal_id,
                        run_id = %run.id,
                        pending_count = tasks.len(),
                        "Reactivated stranded manual run before dispatch"
                    );
                    let message = format!(
                        "Recovered the existing manual run `{}`; it had been queued behind a stalled goal and never started. No duplicate was created. Starting it now, with progress updates enabled.",
                        run.id
                    );
                    let entry = crate::traits::NotificationEntry::new(
                        goal_id,
                        &goal.session_id,
                        "status_update",
                        &message,
                    );
                    if let Err(error) = self.state.enqueue_notification(&entry).await {
                        warn!(goal_id, %error, "Failed to enqueue manual-run recovery update");
                    }
                }
            }

            // Only active goals are dispatchable. Stalled non-manual work is
            // intentionally left alone for explicit recovery.
            if goal.status != "active" {
                continue;
            }

            // A live task lead is authoritative for this goal even during a
            // brief interval where none of its task rows are marked running.
            // Do not let orphan recovery claim a pending task during that gap.
            if self
                .goal_token_registry
                .as_ref()
                .is_some_and(|registry| registry.is_run_active(goal_id))
            {
                continue;
            }

            // Mandate controllers are not ordinary unscheduled goals. Generic
            // orphan recovery may provide a fallback for the exact pending root
            // of the current mandate run, but it must never promote a model-
            // authored non-root action task into a task lead. Such tasks remain
            // owned by mandate finalization/orphan reconciliation.
            let mandate_controller = match self.state.get_mandate_for_goal(goal_id).await {
                Ok(mandate) => mandate.is_some(),
                Err(error) => {
                    warn!(
                        goal_id,
                        %error,
                        "Skipping orphan dispatch because mandate ownership could not be resolved"
                    );
                    continue;
                }
            };
            let current_mandate_root_task_id = if mandate_controller {
                self.state
                    .get_current_goal_run(goal_id)
                    .await
                    .ok()
                    .flatten()
                    .filter(mandate_run_may_dispatch_root)
                    .and_then(|run| run.root_task_id)
            } else {
                None
            };

            let schedules_for_goal = self.state.get_schedules_for_goal(goal_id).await.ok();
            let goal_runs = self.state.get_goal_runs(goal_id).await.unwrap_or_default();
            let is_scheduled_goal = schedules_for_goal
                .as_ref()
                .is_some_and(|schedules| !schedules.is_empty())
                || goal_runs.iter().any(|run| run.trigger_type == "scheduled");
            let active_scheduled_run = if is_scheduled_goal {
                self.state
                    .get_scheduled_run_state(goal_id)
                    .await
                    .ok()
                    .flatten()
            } else {
                None
            };
            // A goal run is the durable isolation boundary for one scheduled or
            // explicit manual firing.  In particular, `trigger_now` starts a new
            // goal run without advancing the cron schedule's `last_run_at`, so
            // using only the schedule timestamp here would let a mutation from
            // the previous run suppress the newly requested run. Prefer the
            // scheduled-run projection's root-task relationship: legacy runs
            // may have been created before trigger provenance was typed, but
            // the root task still identifies their exact owning run.
            let projected_goal_run = if let Some(scheduled_run) = active_scheduled_run.as_ref() {
                self.state
                    .get_goal_run_for_task(&scheduled_run.root_task_id)
                    .await
                    .ok()
                    .flatten()
                    .filter(|run| {
                        run.goal_id == *goal_id
                            && matches!(run.status.as_str(), "pending" | "running" | "blocked")
                    })
            } else {
                None
            };
            let current_goal_run = if is_scheduled_goal {
                match projected_goal_run {
                    Some(run) => Some(run),
                    None => self
                        .state
                        .get_current_goal_run(goal_id)
                        .await
                        .ok()
                        .flatten()
                        // An automatic recovery occurrence is a valid current
                        // cycle of a scheduled objective: its root task must be
                        // dispatched, never retired as a stale child.
                        .filter(|run| matches!(run.trigger_type.as_str(), "scheduled" | "recovery"))
                        // Legacy/repair-created open runs can contain old child
                        // rows without an execution root. They are not a valid
                        // current scheduled cycle and must keep using the stale
                        // child retirement fallback below.
                        .filter(|run| run.root_task_id.is_some()),
                }
            } else {
                None
            };
            let current_goal_run_task_ids = if let Some(run) = current_goal_run.as_ref() {
                self.state
                    .get_tasks_for_goal_run(&run.id)
                    .await
                    .ok()
                    .map(|tasks| {
                        tasks
                            .into_iter()
                            .map(|task| task.id)
                            .collect::<HashSet<_>>()
                    })
            } else {
                None
            };
            let mut eligible_tasks: Vec<&crate::traits::Task> = Vec::with_capacity(tasks.len());
            for task in tasks.iter().copied() {
                let eligible = if mandate_controller {
                    current_mandate_root_task_id.as_deref() == Some(task.id.as_str())
                } else if !is_scheduled_goal {
                    true
                } else if let Some(run_task_ids) = current_goal_run_task_ids.as_ref() {
                    // Durable run membership is authoritative and avoids
                    // timestamp races between task construction and the run
                    // row created milliseconds later.
                    run_task_ids.contains(&task.id)
                } else if let Some(run) = active_scheduled_run.as_ref() {
                    timestamp_is_at_or_after(&task.created_at, &run.created_at)
                } else {
                    // No typed current-run identity means ownership is
                    // indeterminate. Do not infer it from task prose.
                    false
                };
                if eligible {
                    eligible_tasks.push(task);
                    continue;
                }

                if mandate_controller {
                    // Non-root mandate tasks are intentionally invisible to
                    // generic orphan recovery. Leave their state untouched for
                    // the mandate finalizer/reconciler that owns this run.
                    continue;
                }

                let mut retired = task.clone();
                retired.status = "cancelled".to_string();
                retired.idempotent = false;
                retired.max_retries = 0;
                retired.error = Some(
                    "Cancelled stale task from a closed scheduled run; a future schedule fire \
                     will create a fresh root task."
                        .to_string(),
                );
                retired.completed_at = Some(chrono::Utc::now().to_rfc3339());
                if let Err(error) = self.state.update_task(&retired).await {
                    warn!(
                        task_id = %task.id,
                        goal_id = %goal_id,
                        %error,
                        "Failed to retire stale scheduled-run task"
                    );
                } else {
                    info!(
                        task_id = %task.id,
                        goal_id = %goal_id,
                        "Retired stale task from closed scheduled run"
                    );
                }
            }
            if eligible_tasks.is_empty() {
                continue;
            }

            // Daily budget is admission control for scheduled runs, not a reason
            // to abandon already-pending scheduled work.
            if !is_scheduled_goal {
                let budget_today = chrono::Utc::now().date_naive().to_string();
                if daily_budget_exhausted(
                    goal.budget_daily,
                    goal.tokens_used_today,
                    &goal.tokens_used_day,
                    &budget_today,
                ) {
                    tracing::info!(
                        goal_id = %goal.id,
                        tokens_used = goal.tokens_used_today,
                        budget = ?goal.budget_daily,
                        "Skipping pending-task dispatch — today's goal daily budget exhausted"
                    );
                    continue;
                }
            }

            // Check if any tasks are still running (task lead is alive)
            let all_tasks = match self.state.get_tasks_for_goal(goal_id).await {
                Ok(t) => t,
                Err(_) => continue,
            };
            // Only consider a task as actively running if it was started/claimed
            // within the last 10 minutes. Stale claimed tasks (executor crashed)
            // should not block dispatch forever.
            let stale_threshold_secs: i64 = 600; // 10 minutes
            let has_active_nonstale = all_tasks.iter().any(|t| {
                let belongs_to_current_run = if mandate_controller {
                    current_mandate_root_task_id.as_deref() == Some(t.id.as_str())
                } else if !is_scheduled_goal {
                    true
                } else if let Some(run_task_ids) = current_goal_run_task_ids.as_ref() {
                    run_task_ids.contains(&t.id)
                } else if let Some(run) = active_scheduled_run.as_ref() {
                    timestamp_is_at_or_after(&t.created_at, &run.created_at)
                } else {
                    false
                };
                if !belongs_to_current_run {
                    return false;
                }
                if t.status != "running" && t.status != "claimed" {
                    return false;
                }
                let timestamp = t.started_at.as_deref().unwrap_or(&t.created_at);
                chrono::DateTime::parse_from_rfc3339(timestamp)
                    .map(|dt| {
                        let age = chrono::Utc::now() - dt.with_timezone(&chrono::Utc);
                        age.num_seconds() < stale_threshold_secs
                    })
                    .unwrap_or(false)
            });

            if has_active_nonstale {
                // Task lead is still working (recently active) — it'll pick up the pending tasks
                continue;
            }

            // Orphaned: active goal with pending tasks but nothing running.
            // Only consider tasks orphaned if they've been pending for > 60 seconds.
            // This prevents racing with a task lead that just created the tasks
            // but hasn't started dispatching them yet.
            let min_age_secs = 60;
            let all_too_new = eligible_tasks.iter().all(|t| {
                chrono::DateTime::parse_from_rfc3339(&t.created_at)
                    .map(|dt| {
                        let age = chrono::Utc::now() - dt.with_timezone(&chrono::Utc);
                        age.num_seconds() < min_age_secs
                    })
                    .unwrap_or(false)
            });
            if all_too_new {
                continue;
            }

            // Idempotency backstop: these pending tasks are orphaned (nothing's
            // actively running, and they've sat >60s), which is exactly the
            // situation that let a scheduled goal's side effect get repeated —
            // e.g. a task lead posts a tweet, a downstream step (verification,
            // task tracking) hiccups, the run ends up interrupted, and a fresh
            // task lead gets dispatched here for what looks like unfinished
            // work but whose real-world action already succeeded. Before
            // reclaiming, check whether this exact goal run already has a
            // successful mutating (non-GET) http_request logged; if so, the
            // run's job is done — close out the leftover pending tasks instead
            // of spawning another attempt. Never use a prior run's receipt:
            // an explicit manual run is a new authorized execution even when
            // the recurring schedule's `last_run_at` has not advanced.
            // See docs/2026-06-30-telegram-edge-case-findings.md (2026-07-04
            // escalation: 5 duplicate tweets from repeated goal dispatch).
            if is_scheduled_goal {
                if let Some(current_run) = current_goal_run.as_ref() {
                    let mutating_success_task_ids =
                        self.goal_run_mutating_success_tasks(&current_run.id).await;
                    if !mutating_success_task_ids.is_empty() {
                        info!(
                            goal_id = %goal_id,
                            run_id = %current_run.id,
                            pending_count = eligible_tasks.len(),
                            "Goal run already has a successful mutating action; \
                             closing out orphaned pending tasks instead of re-dispatching"
                        );
                        // The receipt is authoritative evidence that the task's
                        // external mutation succeeded, even if a downstream
                        // verification/tracking hiccup left its task status as
                        // interrupted or failed.
                        for task_id in &mutating_success_task_ids {
                            let Ok(Some(mut done)) = self.state.get_task(task_id).await else {
                                continue;
                            };
                            if !done.satisfies_run_completion() {
                                done.status = "completed".to_string();
                                done.error = None;
                                done.blocker = None;
                                done.result = Some(
                                    "Auto-reconciled: this task has a successful mutating \
                                     tool receipt in the current goal run."
                                        .to_string(),
                                );
                                done.completed_at = Some(chrono::Utc::now().to_rfc3339());
                                let _ = self.state.update_task(&done).await;
                            }
                        }
                        for task in &eligible_tasks {
                            let mut done = (*task).clone();
                            done.status = "completed".to_string();
                            done.result = Some(
                                "Auto-closed: a mutating action for this goal's current cycle \
                                 already succeeded (see task_activity log); skipped to avoid a \
                                 duplicate side effect."
                                    .to_string(),
                            );
                            done.completed_at = Some(chrono::Utc::now().to_rfc3339());
                            let _ = self.state.update_task(&done).await;
                        }
                        if self
                            .state
                            .get_tasks_for_goal_run(&current_run.id)
                            .await
                            .is_ok_and(|run_tasks| {
                                !run_tasks.is_empty()
                                    && run_tasks.iter().all(|task| task.satisfies_run_completion())
                            })
                        {
                            let _ = self
                                .state
                                .finish_goal_run(
                                    &current_run.id,
                                    "completed",
                                    Some(
                                        "Auto-reconciled from a successful mutating receipt in \
                                         this goal run; duplicate dispatch was suppressed.",
                                    ),
                                )
                                .await;
                        }
                        continue;
                    }
                }
            }

            // Try to atomically claim the first pending task and spawn a task lead.
            let first_task = eligible_tasks[0];
            let agent_id = format!("heartbeat-dispatch-{}", uuid::Uuid::new_v4());

            let attempt = match self
                .state
                .claim_task_with_lease(&first_task.id, &agent_id, Some("profile-task-lead"), 180)
                .await
            {
                Ok(Some(attempt)) => attempt,
                Ok(None) => continue,
                Err(e) => {
                    error!(task_id = %first_task.id, error = %e, "Failed to claim task for dispatch");
                    continue;
                }
            };

            info!(
                goal_id = %goal_id,
                task_id = %first_task.id,
                pending_count = eligible_tasks.len(),
                "Claimed orphaned task, dispatching task lead"
            );

            // Try to spawn a task lead via agent reference
            if let Some(agent_weak) = &self.agent {
                if let Some(agent_arc) = agent_weak.upgrade() {
                    let state = self.state.clone();
                    let hub = self.hub.clone();
                    let goal_token_registry = self.goal_token_registry.clone();
                    let goal_clone = goal.clone();
                    let session_id = goal.session_id.clone();

                    // Register cancellation token for this goal if not already present
                    if let Some(ref registry) = goal_token_registry {
                        registry.register(&goal.id).await;
                    }

                    // spawn_background_task_lead internally calls tokio::spawn.
                    // Semaphore gating happens at the Agent level during LLM calls.
                    // Use the actual pending task description as TaskLead input.
                    // Passing a generic "resume orphaned tasks" string causes the
                    // TaskLead to re-scope work away from the user's goal.
                    let dispatch_task_text = if first_task.description.trim().is_empty() {
                        goal.description.clone()
                    } else {
                        first_task.description.clone()
                    };

                    crate::agent::spawn_background_task_lead(
                        agent_arc,
                        goal_clone,
                        dispatch_task_text,
                        session_id,
                        ChannelContext::internal(),
                        UserRole::Owner,
                        state,
                        hub,
                        goal_token_registry,
                        Some(first_task.id.clone()),
                        None,
                    );
                    continue;
                }
            }

            // No agent available — revert claimed task back to pending so it's
            // not stranded, then enqueue a stalled notification.
            warn!(
                goal_id = %goal_id,
                task_id = %first_task.id,
                pending_count = eligible_tasks.len(),
                "No agent available for dispatch — reverting claim and notifying user"
            );
            let patch = crate::traits::TaskAttemptPatch {
                status: "cancelled".to_string(),
                error: Some("No agent was available for dispatch.".to_string()),
                ..Default::default()
            };
            let released = self
                .state
                .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
                .await
                .unwrap_or(false);
            if released {
                let _ = self
                    .state
                    .retry_work_task(&first_task.id, "heartbeat", None)
                    .await;
            }

            let msg = format!(
                "Goal stalled: \"{}\" has {} pending task(s) but no active agent. \
                 You can re-trigger this by asking me about it again.",
                goal.description.chars().take(200).collect::<String>(),
                eligible_tasks.len(),
            );
            let entry =
                crate::traits::NotificationEntry::new(goal_id, &goal.session_id, "stalled", &msg);
            if let Err(error) = self.state.enqueue_notification(&entry).await {
                error!(
                    goal_id,
                    %error,
                    "Failed to enqueue stalled-goal notification"
                );
            }
        }
    }

    /// Idempotency backstop for scheduled goals: returns the tasks in one
    /// durable goal run that have a logged, successful, mutating (non-GET)
    /// `http_request` activity.
    ///
    /// This is a defense-in-depth check, not the primary fix — it exists so
    /// that *whatever* caused a scheduled goal's task to look unfinished
    /// (timeout, tool cooldown, orchestrator misjudgment, a bug nobody's hit
    /// yet) can never turn into a repeated real-world side effect. It reads
    /// task_activity rows that are already logged for every tool call
    /// (`src/agent/turn/tool_execution/run.rs`), so it needs no schema change.
    /// Fails closed toward "don't block" on any parse/lookup miss — a false
    /// negative here just falls back to prior behavior, never a false positive
    /// that would wrongly skip real unfinished work. Scoping by `goal_run_id`
    /// is what prevents a successful prior run from suppressing a new manual
    /// trigger.
    pub(super) async fn goal_run_mutating_success_tasks(&self, run_id: &str) -> Vec<String> {
        let tasks = match self.state.get_tasks_for_goal_run(run_id).await {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let mut successful_task_ids = Vec::new();
        for task in &tasks {
            let activities = match self.state.get_task_activities(&task.id).await {
                Ok(a) => a,
                Err(_) => continue,
            };
            // The dispatcher's typed record of an external change, made
            // from the call's own effects whatever the tool.
            if activities.iter().any(|activity| {
                activity.activity_type == crate::traits::REMOTE_CHANGE_ACTIVITY
                    && activity.success == Some(true)
            }) {
                successful_task_ids.push(task.id.clone());
            }
        }
        successful_task_ids
    }
}

/// A mandate run whose root may be dispatched. `pending` counts: when the
/// immediate claim finds no free slot, the run stays pending and this is the
/// only path that can still start it.
fn mandate_run_may_dispatch_root(run: &crate::traits::GoalRun) -> bool {
    run.trigger_type == "mandate" && matches!(run.status.as_str(), "pending" | "running")
}

#[cfg(test)]
mod mandate_dispatch_tests {
    use super::*;

    fn run(trigger: &str, status: &str) -> crate::traits::GoalRun {
        let mut run = crate::traits::GoalRun::new("goal", "project", trigger);
        run.status = status.to_string();
        run
    }

    #[test]
    fn a_review_whose_first_claim_failed_can_still_start() {
        assert!(mandate_run_may_dispatch_root(&run("mandate", "pending")));
        assert!(mandate_run_may_dispatch_root(&run("mandate", "running")));
        assert!(!mandate_run_may_dispatch_root(&run("mandate", "blocked")));
        assert!(!mandate_run_may_dispatch_root(&run("scheduled", "pending")));
    }
}
