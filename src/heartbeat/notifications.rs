//! Delivering goal results to their chats.

use super::*;

impl HeartbeatCoordinator {
    /// Phase 5a: Scan goals that completed/failed and enqueue notifications.
    /// Phase 5b: Process notification queue — attempt delivery.
    /// Phase 5c: Cleanup expired status_update notifications.
    pub(super) async fn deliver_notifications(&self) {
        // Phase 5a: Enqueue notifications for goals that need them
        self.enqueue_goal_notifications().await;

        // Phase 5b: Process notification queue
        self.process_notification_queue().await;

        // Phase 5c: Cleanup expired status_update notifications (24h TTL)
        match self.state.cleanup_expired_notifications().await {
            Ok(count) if count > 0 => {
                info!(count, "Cleaned up expired status_update notifications");
            }
            Err(e) => {
                error!(error = %e, "Failed to cleanup expired notifications");
            }
            _ => {}
        }
    }

    /// Scan for goals needing notification and enqueue them.
    pub(super) async fn enqueue_goal_notifications(&self) {
        let goals = match self.state.get_goals_needing_notification().await {
            Ok(g) => g,
            Err(e) => {
                error!(error = %e, "Failed to get goals needing notification");
                return;
            }
        };
        for goal in &goals {
            let (notification_type, msg) = match goal.status.as_str() {
                "completed" => {
                    // Prefer the latest run's task state. Historical failed runs
                    // remain valuable audit data but cannot invalidate a later,
                    // successfully tracked recovery run.
                    let latest_run = self
                        .state
                        .get_goal_runs(&goal.id)
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .next();
                    let latest_root_task_id = latest_run
                        .as_ref()
                        .and_then(|run| run.root_task_id.as_deref())
                        .map(str::to_string);
                    let completed_tasks = match latest_run {
                        Some(run) => self
                            .state
                            .get_tasks_for_goal_run(&run.id)
                            .await
                            .unwrap_or_default(),
                        None => self
                            .state
                            .get_tasks_for_goal(&goal.id)
                            .await
                            .unwrap_or_default(),
                    };
                    let fallback_summary: String = goal.description.chars().take(300).collect();
                    let task_results_summary = build_goal_task_results_summary(
                        &completed_tasks,
                        latest_root_task_id.as_deref(),
                        &fallback_summary,
                    );

                    if !completed_tasks.is_empty()
                        && !crate::tools::manage_goal_tasks::tasks_satisfy_goal_completion(
                            &completed_tasks,
                        )
                    {
                        let mut corrected = goal.clone();
                        corrected.status = "failed".to_string();
                        corrected.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        corrected.updated_at = chrono::Utc::now().to_rfc3339();
                        if let Err(error) = self.state.update_goal(&corrected).await {
                            error!(goal_id = %goal.id, %error, "Failed to repair inconsistent completed goal state");
                        }
                        (
                            "failed",
                            format!(
                                "Goal incomplete: required tasks did not finish successfully.\n\n{}",
                                task_results_summary.chars().take(4000).collect::<String>()
                            ),
                        )
                    } else {
                        // Check for partial success metadata in context
                        let partial_info = goal
                            .context
                            .as_deref()
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
                            let blocked = summary
                                .get("blocked")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
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
                        } else if crate::tools::manage_goal_tasks::goal_completion_summary_indicates_not_finished(&task_results_summary) {
                            // The summary contains verification-blocked or incomplete-work
                            // language. Suppress a duplicate terminal notice; structural task
                            // state above remains the authoritative completion gate.
                            continue;
                        } else {
                            (
                                "completed",
                                format!(
                                    "{}\n\n{}",
                                    goal_completion_header(&completed_tasks),
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
                        goal.description.chars().take(300).collect::<String>()
                    ),
                ),
                _ => continue,
            };

            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                notification_type,
                &msg,
            );

            if let Err(e) = self.state.enqueue_goal_notification(&entry).await {
                error!(goal_id = %goal.id, error = %e, "Failed to atomically enqueue notification");
            }
        }
    }

    /// Queue delivery bypasses `deliver_parent_text_result`, so persist every
    /// delivered notice in owner history: a reply to it must reach an agent
    /// that can see what it answers.
    pub(crate) async fn record_delivered_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
        delivered_text: &str,
    ) {
        let Some(agent) = self.agent.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        // An untrusted notice joins a turn recorded tainted before the note
        // exists; if the record cannot be written the note is not written
        // either (it would read as clean).
        let turn_id = if entry.untrusted {
            match agent.tainted_note_turn(&entry.session_id).await {
                Ok(turn_id) => Some(turn_id),
                Err(error) => {
                    warn!(
                        notification_id = %entry.id,
                        %error,
                        "Untrusted notification not recorded in owner history: its taint could not be recorded"
                    );
                    return;
                }
            }
        } else {
            None
        };
        if let Err(error) = agent
            .record_parent_visible_result_note_in_turn(
                &entry.session_id,
                QUEUED_NOTIFICATION_NOTE_PREFIX,
                delivered_text,
                turn_id.as_deref(),
            )
            .await
        {
            warn!(
                notification_id = %entry.id,
                %error,
                "Failed to record delivered notification in owner history"
            );
        }
        if notice_accepts_mandate_answer(&entry.notification_type) {
            self.bind_mandate_answer_context(&agent, entry).await;
        }
    }

    /// Bind the owner's next reply to a delivered mandate question so a bare
    /// "yes" reaches `answer_question` instead of an agent with no referent.
    /// A timed-out ASK has already resumed, so any non-terminal mandate that
    /// still carries the question qualifies; the answer lands as late guidance.
    pub(super) async fn bind_mandate_answer_context(
        &self,
        agent: &Agent,
        entry: &crate::traits::NotificationEntry,
    ) {
        let mandate = match self.state.get_mandate_for_goal(&entry.goal_id).await {
            Ok(Some(mandate)) => mandate,
            Ok(None) => return,
            Err(error) => {
                warn!(
                    goal_id = %entry.goal_id,
                    notification_id = %entry.id,
                    %error,
                    "Failed to load mandate after delivering queued question notice"
                );
                return;
            }
        };
        let answerable = match entry.notification_type.as_str() {
            "mandate_ask" => mandate.status == MandateStatus::AwaitingInput,
            _ => !mandate.status.is_terminal(),
        };
        if !answerable {
            return;
        }
        if let Err(error) = agent
            .record_mandate_owner_input_context(
                &entry.session_id,
                &mandate.id,
                mandate.version,
                entry.action_token.as_deref().unwrap_or(&entry.id),
            )
            .await
        {
            warn!(
                mandate_id = %mandate.id,
                notification_id = %entry.id,
                %error,
                "Failed to bind queued mandate question to owner dialogue state"
            );
        }
    }

    /// Process the notification queue: attempt delivery, track attempts.
    pub(super) async fn process_notification_queue(&self) {
        // Read past the normal delivery batch so quiet-hour routine entries at
        // the front of the queue cannot hide a later owner-action request.
        let pending = match self.state.get_pending_notifications(100).await {
            Ok(n) => n,
            Err(e) => {
                error!(error = %e, "Failed to get pending notifications");
                return;
            }
        };

        let mut delivery_attempts = 0usize;
        for entry in &pending {
            let local_hour = chrono::Timelike::hour(&chrono::Local::now());
            if !entry.should_deliver_at_local_hour(local_hour) {
                tracing::debug!(
                    notification_id = %entry.id,
                    notification_type = %entry.notification_type,
                    local_hour,
                    "Deferring recoverable notification during quiet hours"
                );
                continue;
            }
            if delivery_attempts >= 20 {
                break;
            }
            delivery_attempts = delivery_attempts.saturating_add(1);
            if entry.notification_type == "escalation" {
                if let Some(task_id) = entry.task_id.as_deref() {
                    // Give the coordinator a brief chance to reconcile the
                    // executor handoff, then verify the linked task is still
                    // blocked. This prevents a stale critical notification
                    // from racing a same-turn unblock or completion.
                    if !task_escalation_has_settled(&entry.created_at, Utc::now()) {
                        continue;
                    }
                    match self.state.get_task(task_id).await {
                        Ok(Some(task)) if task.status == "blocked" => {}
                        Ok(_) => {
                            if let Err(error) =
                                self.state.mark_notification_delivered(&entry.id).await
                            {
                                warn!(
                                    notification_id = %entry.id,
                                    task_id,
                                    error = %error,
                                    "Failed to close stale task escalation"
                                );
                            }
                            continue;
                        }
                        Err(error) => {
                            warn!(
                                notification_id = %entry.id,
                                task_id,
                                error = %error,
                                "Deferred task escalation because task state could not be verified"
                            );
                            continue;
                        }
                    }
                }
            }
            let sanitized = crate::tools::sanitize::sanitize_user_facing_reply(&entry.message);
            let delivered = if let Some(hub) = self.hub.as_ref().and_then(|w| w.upgrade()) {
                let message =
                    crate::channels::present_notification(&entry.notification_type, &sanitized);
                // A channel adapter is external I/O. It must never be allowed to
                // pin the heartbeat forever because this same loop admits due
                // schedules and dispatches pending tasks on the next tick.
                match tokio::time::timeout(
                    Duration::from_secs(15),
                    hub.send_text(&entry.session_id, &message),
                )
                .await
                {
                    Ok(result) => result.is_ok(),
                    Err(_) => {
                        warn!(
                            notification_id = %entry.id,
                            session_id = %entry.session_id,
                            "Notification delivery timed out"
                        );
                        false
                    }
                }
            } else {
                false
            };

            if delivered {
                self.record_delivered_notification(entry, &sanitized).await;
                if let Err(e) = self.state.mark_notification_delivered(&entry.id).await {
                    error!(notification_id = %entry.id, error = %e, "Failed to mark notification delivered");
                }
            } else {
                // Increment attempt counter — critical notifications will keep retrying
                // (no expiry), status_update notifications will eventually expire via TTL
                if let Err(e) = self.state.increment_notification_attempt(&entry.id).await {
                    error!(notification_id = %entry.id, error = %e, "Failed to increment notification attempt");
                }
            }
        }
    }
}
