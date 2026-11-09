//! Firing and reconciling scheduled goals.

use super::*;

impl HeartbeatCoordinator {
    /// Check for due schedules across active orchestration goals and enqueue tasks.
    ///
    /// Scheduling is per-schedule (`goal_schedules`) rather than a goal column.
    pub(super) async fn check_due_goal_schedules(&self) {
        let due = match self.state.get_due_goal_schedules(50).await {
            Ok(s) => s,
            Err(e) => {
                error!(error = %e, "Failed to get due goal schedules");
                return;
            }
        };

        if due.is_empty() {
            return;
        }

        info!(count = due.len(), "Found due goal schedules");
        for schedule in due {
            let schedule_id = schedule.id.clone();
            let goal_id = schedule.goal_id.clone();
            if let Err(e) = self.fire_due_schedule(schedule).await {
                error!(
                    schedule_id = %schedule_id,
                    goal_id = %goal_id,
                    error = %e,
                    "Failed to fire due schedule"
                );
            }
        }
    }

    pub(super) async fn reconcile_open_scheduled_runs(&self) {
        let goals = match self.state.get_scheduled_goals().await {
            Ok(goals) => goals,
            Err(error) => {
                error!(%error, "Failed to enumerate scheduled goals for lifecycle reconciliation");
                return;
            }
        };
        for goal in goals {
            if let Err(error) = self.reconcile_open_scheduled_runs_for_goal(&goal.id).await {
                error!(goal_id = %goal.id, %error, "Failed to reconcile scheduled goal lifecycle");
            }
        }
    }

    pub(super) async fn reconcile_open_scheduled_runs_for_goal(
        &self,
        goal_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::GoalRun>> {
        let candidate_open_runs = self
            .state
            .get_goal_runs(goal_id)
            .await?
            .into_iter()
            .filter(|run| matches!(run.status.as_str(), "pending" | "running" | "blocked"))
            .collect::<Vec<_>>();
        let mut open_runs = Vec::new();
        for run in candidate_open_runs {
            // While any worker holds a live execution lease on this run, that
            // worker owns the outcome: a supervising task lead may still
            // salvage a transiently failed child (observed live: the lead
            // re-completed a research task seconds after its child attempt
            // failed). Reconciliation is for orphaned runs; it must not race
            // a live lead, so it defers until every lease is gone.
            if self.state.goal_run_has_live_attempt(&run.id).await? {
                open_runs.push(run);
                continue;
            }
            let run_tasks = self.state.get_tasks_for_goal_run(&run.id).await?;
            if let Some(status) = scheduled_run_reconciliation_status(&run, &run_tasks) {
                self.state
                    .finish_goal_run(
                        &run.id,
                        status,
                        Some("Reconciled from the scheduled run's authoritative task graph."),
                    )
                    .await?;
                info!(
                    goal_id,
                    run_id = %run.id,
                    prior_status = %run.status,
                    reconciled_status = status,
                    "Reconciled split-brain scheduled run from task lifecycle"
                );
                continue;
            }
            open_runs.push(run);
        }
        Ok(open_runs)
    }

    /// Launch one automatic recovery run per escalated objective once its
    /// cool-down has elapsed, up to a fixed attempt cap. The recovery run is
    /// the same typed recovery lineage the task lead creates inline
    /// (`terminal_recovery` + `recovery_for_run`), so a verified success
    /// resets the failure budget and resumes the paused schedules through
    /// `finish_goal_run`, and a failure lands in the same budget accounting.
    /// The paused schedules stay paused during recovery so the cron cannot
    /// double-fire.
    pub(super) async fn launch_escalated_recoveries(&self) {
        let candidates = match self
            .state
            .list_escalated_recovery_candidates(
                Self::ESCALATED_RECOVERY_COOLDOWN_SECS,
                Self::ESCALATED_RECOVERY_MAX_ATTEMPTS,
            )
            .await
        {
            Ok(candidates) => candidates,
            Err(error) => {
                error!(%error, "Failed to list escalated recovery candidates");
                return;
            }
        };
        for recovery in candidates {
            let goal_id = recovery.goal_id.clone();
            let Ok(Some(goal)) = self.state.get_goal(&goal_id).await else {
                continue;
            };
            if goal.status != "active" {
                continue;
            }
            // Automatic recovery is admission-controlled by the owner's daily
            // budget exactly like a scheduled fire; it never spends past it.
            let budget_today = chrono::Utc::now().date_naive().to_string();
            if daily_budget_exhausted(
                goal.budget_daily,
                goal.tokens_used_today,
                &goal.tokens_used_day,
                &budget_today,
            ) || !daily_budget_has_run_capacity(
                goal.budget_daily,
                goal.budget_per_check,
                goal.tokens_used_today,
                &goal.tokens_used_day,
                &budget_today,
            ) {
                info!(
                    goal_id = %goal_id,
                    tokens_used = goal.tokens_used_today,
                    budget = ?goal.budget_daily,
                    "Deferring automatic recovery — today's goal daily budget cannot admit a run"
                );
                continue;
            }
            // An open occurrence (of any trigger type) means work is still in
            // flight or wedged; reconcile first and try again next tick.
            match self.reconcile_open_scheduled_runs_for_goal(&goal_id).await {
                Ok(open_runs) if !open_runs.is_empty() => {
                    let stale = open_runs.iter().all(|run| run.status == "blocked");
                    if !stale {
                        continue;
                    }
                    for run in &open_runs {
                        let _ = self
                            .state
                            .finish_goal_run(
                                &run.id,
                                "failed",
                                Some("Closed before automatic recovery."),
                            )
                            .await;
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    warn!(goal_id = %goal_id, %error, "Could not reconcile before recovery");
                    continue;
                }
            }
            // Bump the attempt counter before creating any work so a crash
            // between the two cannot hot-loop recovery launches.
            if let Err(error) = self.state.record_scheduled_recovery_attempt(&goal_id).await {
                warn!(goal_id = %goal_id, %error, "Failed to record recovery attempt");
                continue;
            }
            let attempt = recovery.recovery_attempts.saturating_add(1);
            let now = chrono::Utc::now().to_rfc3339();
            let recovery_task_id = uuid::Uuid::new_v4().to_string();
            let mut context = goal
                .context
                .as_deref()
                .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
                .filter(serde_json::Value::is_object)
                .unwrap_or_else(|| serde_json::json!({}));
            context["recovery_for_run"] = serde_json::json!(recovery.last_failed_run_id);
            context["terminal_recovery"] = serde_json::json!(true);
            context["recovery_attempt"] = serde_json::json!(attempt);
            context["recovery_cause"] = serde_json::json!(recovery
                .latest_failure_kind
                .map(crate::traits::ScheduledFailureKind::as_str));
            let task = crate::traits::Task {
                id: recovery_task_id.clone(),
                goal_id: goal_id.clone(),
                description: format!(
                    "Directly recover and finish: {} [SYSTEM: automatic recovery attempt {attempt} of {} after {} consecutive failed runs (typed cause {}). The schedule is paused until this run completes with verified receipts. Earlier failures recorded in the ledger are historical evidence from a previous environment, not proof that their cause persists: re-run the previously failing step once in this attempt and judge from its fresh receipt before concluding that recovery is exhausted. If the objective's repository or workspace is not present in the attempt workspace, locate it with project_inspect and bind it with scheduled_goal_runs bind_workspace before reporting a blocker.]",
                    goal.description,
                    Self::ESCALATED_RECOVERY_MAX_ATTEMPTS,
                    recovery.consecutive_failures,
                    recovery
                        .latest_failure_kind
                        .map_or("unknown", crate::traits::ScheduledFailureKind::as_str),
                ),
                status: "pending".to_string(),
                priority: "high".to_string(),
                task_order: 0,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: Some(context.to_string()),
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at: now,
                started_at: None,
                completed_at: None,
            };
            if let Err(error) = self
                .state
                .start_goal_run(&goal_id, "recovery", None, Some(&recovery_task_id))
                .await
            {
                warn!(goal_id = %goal_id, %error, "Failed to start automatic recovery run");
                continue;
            }
            match self.state.create_task(&task).await {
                Ok(()) => info!(
                    goal_id = %goal_id,
                    task_id = %recovery_task_id,
                    attempt,
                    max_attempts = Self::ESCALATED_RECOVERY_MAX_ATTEMPTS,
                    "Launched automatic recovery run for escalated scheduled objective"
                ),
                Err(error) => {
                    warn!(goal_id = %goal_id, %error, "Failed to create automatic recovery task")
                }
            }
        }
    }

    pub(super) async fn fire_due_schedule(&self, mut schedule: GoalSchedule) -> anyhow::Result<()> {
        // Guardrails (unknown policy/tz -> treat as coalesce/local-only).
        if schedule.tz != "local" {
            tracing::warn!(
                schedule_id = %schedule.id,
                goal_id = %schedule.goal_id,
                tz = %schedule.tz,
                "Skipping schedule with unsupported tz"
            );
            return Ok(());
        }

        let now = chrono::Utc::now();
        let now_ts = now.to_rfc3339();

        let Some(mut goal) = self.state.get_goal(&schedule.goal_id).await? else {
            return Ok(());
        };

        // Safety: only active orchestration goals should fire.
        if goal.domain != "orchestration" || goal.status != "active" {
            return Ok(());
        }

        // A continuous goal idle >30 days has likely stopped doing useful
        // work. We KEEP firing it (so it can recover on its own once it starts
        // succeeding again), but surface a one-time alert so a silently-dead
        // goal becomes visible instead of being skipped forever. The alert is
        // gated on `notified_at` so we don't re-alert on every tick while the
        // goal stays idle (e.g. blocked by an open task via coalescing).
        if goal.goal_type == "continuous" && goal.notified_at.is_none() {
            if let Some(ref last_action) = goal.last_useful_action {
                if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(last_action) {
                    let days_idle =
                        (chrono::Utc::now() - ts.with_timezone(&chrono::Utc)).num_days();
                    if days_idle > 30 {
                        warn!(
                            goal_id = %goal.id,
                            description = %goal.description,
                            days_idle,
                            "Continuous goal idle >30 days; alerting user and continuing to fire"
                        );
                        let msg = format!(
                            "Heads up: your recurring goal \"{}\" hasn't made progress in {} days. \
                             It's still scheduled and will keep trying — reply if you'd like to pause or cancel it.",
                            crate::tools::sanitize::short_goal_label(&goal.description),
                            days_idle
                        );
                        let entry = crate::traits::NotificationEntry::new(
                            &goal.id,
                            &goal.session_id,
                            "evergreen_alert",
                            &msg,
                        );
                        match self.state.enqueue_goal_notification(&entry).await {
                            Ok(_) => {
                                // Keep the in-memory copy consistent so the later
                                // update_goal (after task creation) doesn't clobber
                                // notified_at back to NULL and re-alert next tick.
                                goal.notified_at = Some(now_ts.clone());
                            }
                            Err(error) => {
                                warn!(
                                    goal_id = %goal.id,
                                    %error,
                                    "Failed to atomically enqueue idle-goal alert"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Only work belonging to an open goal run can apply backpressure to a
        // new scheduled firing. Historical blocked tasks are intentionally
        // retained as audit records after their run is closed; counting all
        // tasks for the goal would make those records suppress every future
        // coalesced firing forever.
        let open_runs = self
            .reconcile_open_scheduled_runs_for_goal(&goal.id)
            .await?;
        let mut open_count = 0;
        for run in &open_runs {
            open_count += self
                .state
                .get_tasks_for_goal_run(&run.id)
                .await?
                .iter()
                .filter(|task| task_blocks_later_schedule_fire(run, task))
                .count();
        }

        let fire_policy = schedule.fire_policy.as_str();
        let coalesce = fire_policy != "always_fire";
        const ALWAYS_FIRE_OPEN_TASK_CAP: usize = 3;

        // Backpressure: coalesce by default; always_fire only up to a cap.
        if (coalesce && open_count > 0) || (!coalesce && open_count >= ALWAYS_FIRE_OPEN_TASK_CAP) {
            if schedule.is_one_shot {
                // Keep the one-shot due, but avoid hot-looping while open work exists.
                schedule.next_run_at = (now + chrono::Duration::minutes(5)).to_rfc3339();
            } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
                schedule.next_run_at = next.to_rfc3339();
            }
            schedule.updated_at = now_ts.clone();
            let _ = self.state.update_goal_schedule(&schedule).await;
            info!(
                goal_id = %goal.id,
                schedule_id = %schedule.id,
                fire_policy,
                open_run_count = open_runs.len(),
                open_task_count = open_count,
                "Deferred schedule fire because open goal-run work is still active"
            );
            return Ok(());
        }

        // Budget check: skip if *today's* daily budget is exhausted, but back off
        // schedule to avoid hot-loop. Stale prior-day usage must not count, or the
        // goal deadlocks across the day boundary (it would defer forever and never
        // fire, so its counter would never reset).
        let budget_today = now.date_naive().to_string();
        let effective_daily_budget = if let Some(configured_budget) = goal.budget_daily {
            crate::goal_tokens::load_goal_daily_budget_override(
                self.state.as_ref(),
                &goal.id,
                configured_budget,
                crate::agent::SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
            )
            .await
            .map(|value| value.budget_daily)
            .or(Some(configured_budget))
        } else {
            None
        };
        if daily_budget_exhausted(
            effective_daily_budget,
            goal.tokens_used_today,
            &goal.tokens_used_day,
            &budget_today,
        ) {
            schedule.last_run_at = Some(now_ts.clone());
            schedule.next_run_at = if schedule.is_one_shot {
                (now + chrono::Duration::hours(24)).to_rfc3339()
            } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
                next.to_rfc3339()
            } else {
                (now + chrono::Duration::hours(24)).to_rfc3339()
            };
            schedule.updated_at = now_ts.clone();
            let _ = self.state.update_goal_schedule(&schedule).await;
            let msg = format!(
                "Skipped the scheduled run for \"{}\" because today's daily token budget \
                 ({}) is exhausted by cumulative usage across this goal's runs today, not by \
                 this unstarted run alone. No work was started. The schedule remains active and \
                 will try again at its next normal fire after the counter resets.",
                crate::tools::sanitize::short_goal_label(&goal.description),
                effective_daily_budget.unwrap_or_default(),
            );
            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                "token_alert",
                &msg,
            );
            if let Err(error) = self.state.enqueue_notification(&entry).await {
                error!(
                    goal_id = %goal.id,
                    %error,
                    "Failed to enqueue daily-budget notification"
                );
            }
            return Ok(());
        }
        if !daily_budget_has_run_capacity(
            effective_daily_budget,
            goal.budget_per_check,
            goal.tokens_used_today,
            &goal.tokens_used_day,
            &budget_today,
        ) {
            let remaining = effective_daily_budget
                .unwrap_or_default()
                .saturating_sub(goal.tokens_used_today)
                .max(0);
            let required = goal.budget_per_check.unwrap_or_default().max(0);
            schedule.last_run_at = Some(now_ts.clone());
            schedule.next_run_at = if schedule.is_one_shot {
                (now + chrono::Duration::minutes(15)).to_rfc3339()
            } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
                next.to_rfc3339()
            } else {
                (now + chrono::Duration::hours(24)).to_rfc3339()
            };
            schedule.updated_at = now_ts.clone();
            let _ = self.state.update_goal_schedule(&schedule).await;

            let msg = format!(
                "Skipped the scheduled run for \"{}\" because only {} daily-budget tokens \
                 remain and a full run is allowed up to {}. No work was started. The schedule \
                 remains active and will try again at its next normal fire after the daily \
                 counter resets.",
                crate::tools::sanitize::short_goal_label(&goal.description),
                remaining,
                required,
            );
            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                "token_alert",
                &msg,
            );
            if let Err(error) = self.state.enqueue_notification(&entry).await {
                error!(
                    goal_id = %goal.id,
                    %error,
                    "Failed to enqueue remaining-budget notification"
                );
            }
            info!(
                goal_id = %goal.id,
                remaining,
                required,
                "Skipped scheduled run before admission because remaining daily budget cannot fund one full run"
            );
            return Ok(());
        }

        // ── Advance schedule BEFORE creating task to prevent race-condition
        //    double-fires. If task creation fails afterwards, we skip one firing
        //    (harmless) rather than risk duplicate fires from concurrent ticks.
        let advanced_next_run = if schedule.is_one_shot {
            // Park one-shot far in the future; we delete it after successful task creation.
            (now + chrono::Duration::hours(24)).to_rfc3339()
        } else if let Ok(next) = crate::cron_utils::compute_next_run(&schedule.cron_expr) {
            next.to_rfc3339()
        } else {
            warn!(schedule_id = %schedule.id, cron = %schedule.cron_expr, "Failed to compute next run for recurring schedule");
            return Ok(());
        };

        schedule.last_run_at = Some(now_ts.clone());
        schedule.next_run_at = advanced_next_run;
        schedule.updated_at = now_ts.clone();
        if let Err(e) = self.state.update_goal_schedule(&schedule).await {
            warn!(schedule_id = %schedule.id, error = %e, "Failed to advance schedule before task creation — skipping to avoid potential duplicate");
            return Ok(());
        }

        let run_root_task_id = uuid::Uuid::new_v4().to_string();
        if coalesce {
            if let Some(open_run) = self.state.get_current_goal_run(&goal.id).await? {
                let run_tasks = self.state.get_tasks_for_goal_run(&open_run.id).await?;
                if !run_tasks.is_empty() {
                    let active_or_human_blocked = run_tasks
                        .iter()
                        .any(|task| task_blocks_later_schedule_fire(&open_run, task));
                    if active_or_human_blocked {
                        warn!(
                            goal_id = %goal.id,
                            run_id = %open_run.id,
                            "Skipped schedule fire because the previous goal run is still open"
                        );
                        return Ok(());
                    }
                    let status = if run_tasks.iter().any(task_is_terminal_schedule_failure) {
                        "failed"
                    } else {
                        "completed"
                    };
                    // A blocked child of the superseded occurrence is an
                    // audit record now; mark it so it never reads as open
                    // work waiting on an owner.
                    for task in run_tasks.iter().filter(|task| task.status == "blocked") {
                        let mut superseded = task.clone();
                        superseded.status = "superseded".to_string();
                        superseded.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        superseded.result = Some(format!(
                            "Superseded by the next scheduled occurrence. Prior blocker: {}",
                            task.blocker.as_deref().unwrap_or("none")
                        ));
                        if let Err(error) = self.state.update_task(&superseded).await {
                            warn!(task_id = %task.id, %error, "Failed to supersede blocked task");
                        }
                    }
                    let _ = self
                        .state
                        .finish_goal_run(
                            &open_run.id,
                            status,
                            Some("Closed before the next scheduled firing."),
                        )
                        .await?;
                }
            }
        }
        let goal_run = self
            .state
            .start_goal_run(
                &goal.id,
                "scheduled",
                Some(&schedule.id),
                Some(&run_root_task_id),
            )
            .await?;

        // ── Plain-reminder fast path: the only deliverable is a message back
        // to the user, so send it now instead of going through task creation
        // and the task-lead pipeline (which adds a minute-plus of dispatch and
        // LLM latency). Falls through to the normal pipeline if the message
        // can't be delivered.
        if let Some(fire_text) =
            crate::reminders::fire_text_for_goal(goal.context.as_deref(), &goal.description)
        {
            let delivered = if let Some(hub_arc) = self.hub.as_ref().and_then(|w| w.upgrade()) {
                hub_arc
                    .send_text(&goal.session_id, &fire_text)
                    .await
                    .is_ok()
            } else {
                false
            };
            if delivered {
                // Record a completed task so history/diagnostics show the run.
                let task = crate::traits::Task {
                    id: run_root_task_id.clone(),
                    goal_id: goal.id.clone(),
                    description: format!("Deliver reminder: {}", goal.description),
                    status: "completed".to_string(),
                    priority: "medium".to_string(),
                    task_order: 0,
                    parallel_group: None,
                    depends_on: None,
                    agent_id: Some("reminder-fast-path".to_string()),
                    context: None,
                    result: Some(fire_text),
                    error: None,
                    blocker: None,
                    idempotent: false,
                    retry_count: 0,
                    max_retries: 1,
                    created_at: now_ts.clone(),
                    started_at: Some(now_ts.clone()),
                    completed_at: Some(now_ts.clone()),
                };
                self.state.create_task(&task).await?;
                let _ = self
                    .state
                    .finish_goal_run(
                        &goal_run.id,
                        "completed",
                        Some("Reminder delivered successfully."),
                    )
                    .await?;

                let mut updated = goal.clone();
                if schedule.is_one_shot {
                    updated.status = "completed".to_string();
                    updated.completed_at = Some(now_ts.clone());
                    // The reminder itself is the notification — suppress the
                    // generic "Goal completed" follow-up.
                    updated.notified_at = Some(now_ts.clone());
                }
                updated.last_useful_action = Some(now_ts.clone());
                updated.updated_at = now_ts.clone();
                let _ = self.state.update_goal(&updated).await;

                if schedule.is_one_shot {
                    let _ = self.state.delete_goal_schedule(&schedule.id).await;
                }
                info!(
                    goal_id = %goal.id,
                    one_shot = schedule.is_one_shot,
                    "Delivered plain reminder via fast path"
                );
                return Ok(());
            }
            warn!(
                goal_id = %goal.id,
                "Reminder fast path could not deliver; falling back to task pipeline"
            );
        }

        // Create a pending task for this scheduled run.
        let task = crate::traits::Task {
            id: run_root_task_id,
            goal_id: goal.id.clone(),
            description: if schedule.is_one_shot || goal.goal_type == "finite" {
                format!(
                    "Execute scheduled goal: {} [SYSTEM: already scheduled and firing now; do not reschedule.]",
                    goal.description
                )
            } else {
                format!(
                    "Scheduled check: {} [SYSTEM: already scheduled and firing now; do not reschedule.]",
                    goal.description
                )
            },
            status: "pending".to_string(),
            priority: if goal.goal_type == "continuous" {
                "low".to_string()
            } else {
                "medium".to_string()
            },
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: goal.context.clone(),
            result: None,
            error: None,
            blocker: None,
            // A scheduled root can perform externally visible writes. Retrying
            // the whole orchestration after an ambiguous failure can duplicate
            // those writes, so only individual, explicitly idempotent child
            // tasks may be retried.
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now_ts.clone(),
            started_at: None,
            completed_at: None,
        };

        self.state.create_task(&task).await?;

        // Update goal timestamp.
        let mut updated_goal = goal.clone();
        updated_goal.last_useful_action = Some(now_ts.clone());
        updated_goal.updated_at = now_ts.clone();
        if let Err(e) = self.state.update_goal(&updated_goal).await {
            warn!(goal_id = %goal.id, error = %e, "Failed to update goal timestamp after schedule fire");
        }

        // Clean up one-shot schedules after successful task creation.
        // (Recurring schedules were already advanced before task creation.)
        if schedule.is_one_shot {
            if let Err(e) = self.state.delete_goal_schedule(&schedule.id).await {
                warn!(schedule_id = %schedule.id, error = %e, "Failed to delete one-shot schedule after fire");
            }
        }

        info!(
            goal_id = %goal.id,
            schedule_id = %schedule.id,
            task_id = %task.id,
            "Enqueued scheduled task"
        );

        // Notify user that the scheduled goal is executing (DMs only —
        // group channels just get the results without progress noise).
        // Post the "Running scheduled task" announcement as a TRACKED surface so
        // the task lead's progress heartbeat can edit it in place — one self-
        // updating message instead of a separate announcement + progress stream.
        let mut running_surface_id: Option<String> = None;
        let local_hour = chrono::Timelike::hour(&chrono::Local::now());
        if !is_group_session(&goal.session_id)
            && crate::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(local_hour)
        {
            if let Some(hub_weak) = &self.hub {
                if let Some(hub_arc) = hub_weak.upgrade() {
                    let short_desc = crate::tools::sanitize::short_goal_label(&goal.description);
                    running_surface_id = hub_arc
                        .send_text_tracked(
                            &goal.session_id,
                            &format!("⏳ **Scheduled run in progress**\n\n{}", short_desc),
                        )
                        .await
                        .ok()
                        .flatten();
                }
            }
        }

        // Dispatch immediately instead of waiting for the orphan-recovery pass
        // (which only picks up tasks older than 60s, so every scheduled run
        // used to start at least a minute late). If no agent is available the
        // orphan dispatcher still picks the task up on a later tick.
        if let Some(agent_arc) = self.agent.as_ref().and_then(|w| w.upgrade()) {
            let agent_id = format!("heartbeat-schedule-fire-{}", uuid::Uuid::new_v4());
            match self
                .state
                .claim_task_with_lease(&task.id, &agent_id, Some("profile-task-lead"), 180)
                .await
            {
                Ok(Some(_)) => {
                    if let Some(ref registry) = self.goal_token_registry {
                        registry.register(&goal.id).await;
                    }
                    crate::agent::spawn_background_task_lead(
                        agent_arc,
                        goal.clone(),
                        task.description.clone(),
                        goal.session_id.clone(),
                        ChannelContext::internal(),
                        UserRole::Owner,
                        self.state.clone(),
                        self.hub.clone(),
                        self.goal_token_registry.clone(),
                        Some(task.id.clone()),
                        running_surface_id,
                    );
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(
                        task_id = %task.id,
                        error = %e,
                        "Failed to claim scheduled task for immediate dispatch"
                    );
                }
            }
        }

        Ok(())
    }
}
