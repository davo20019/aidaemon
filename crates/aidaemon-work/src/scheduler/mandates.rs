//! Reviewing due owner mandates.

use super::*;

impl GoalCycle {
    /// Claim and dispatch due mandate reviews from the mandate-owned wake clock.
    ///
    /// This path is deliberately independent from `goal_schedules`: a mandate
    /// review gets `trigger_type = "mandate"`, no schedule id, and a distinct
    /// root-task description. The mandate lease is the admission fence; the
    /// task-attempt lease and per-goal run guard remain execution fences.
    pub async fn check_due_mandates(&self) {
        let lease_owner = format!("heartbeat-mandate-{}", uuid::Uuid::new_v4());
        let due = match self
            .state
            .claim_due_mandates(
                MANDATE_REVIEW_BATCH_SIZE,
                &lease_owner,
                MANDATE_REVIEW_LEASE_SECS,
            )
            .await
        {
            Ok(mandates) => mandates,
            Err(error) => {
                error!(%error, "Failed to claim due mandate reviews");
                return;
            }
        };

        if due.is_empty() {
            return;
        }

        info!(count = due.len(), "Found due mandate reviews");
        // A review no model can answer would only fail. When every
        // configured model is in a known cooldown (a used-up quota, a
        // rejected credential), move the reviews to when one ends.
        let unavailable_until = self
            .live_agent()
            .and_then(|agent| agent.next_model_answer_at());
        if let Some(until) = unavailable_until {
            let retry_at = chrono::DateTime::<chrono::Utc>::from(until).to_rfc3339();
            for mandate in due {
                info!(
                    mandate_id = %mandate.id,
                    retry_at,
                    "No configured model can answer; mandate review moved to when one can"
                );
                if let Some(token) = mandate.review_lease_token.as_deref() {
                    if let Err(error) = self
                        .state
                        .release_mandate_review_lease(&mandate.id, token, &retry_at)
                        .await
                    {
                        warn!(mandate_id = %mandate.id, %error, "Failed to defer mandate review");
                    }
                }
            }
            return;
        }
        for mandate in due {
            let mandate_id = mandate.id.clone();
            let lease_token = mandate.review_lease_token.clone();
            if let Err(error) = self
                .dispatch_due_mandate_with_busy_retry(mandate.clone())
                .await
            {
                error!(
                    mandate_id,
                    goal_id = %mandate.goal_id,
                    %error,
                    "Failed to dispatch due mandate review"
                );
                if let Some(token) = lease_token.as_deref() {
                    let retry_at = if is_sqlite_busy_error(&error) {
                        mandate_transient_retry_at(&mandate, chrono::Utc::now())
                    } else {
                        mandate_retry_at(&mandate, Some(mandate.min_review_secs))
                    };
                    if let Err(release_error) = self
                        .state
                        .release_mandate_review_lease(&mandate.id, token, &retry_at)
                        .await
                    {
                        warn!(
                            mandate_id = %mandate.id,
                            %release_error,
                            "Failed to release mandate review lease after dispatch error"
                        );
                    }
                }
            }
        }
    }

    pub async fn dispatch_due_mandate_with_busy_retry(
        &self,
        mandate: Mandate,
    ) -> anyhow::Result<()> {
        let mut busy_retries = 0_u32;
        loop {
            match self.dispatch_due_mandate(mandate.clone()).await {
                Ok(()) => return Ok(()),
                Err(error)
                    if busy_retries < MANDATE_DISPATCH_BUSY_RETRIES
                        && is_sqlite_busy_error(&error) =>
                {
                    busy_retries += 1;
                    let delay_ms = 100_u64 << (busy_retries - 1);
                    warn!(
                        mandate_id = %mandate.id,
                        retry = busy_retries,
                        delay_ms,
                        "Retrying mandate dispatch after transient SQLite contention"
                    );
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub async fn dispatch_due_mandate(&self, mandate: Mandate) -> anyhow::Result<()> {
        let lease_token = mandate
            .review_lease_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("claimed mandate did not carry a review lease token"))?;

        let Some(goal) = self.state.get_goal(&mandate.goal_id).await? else {
            let _ = self
                .state
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Active,
                    MandateStatus::Paused,
                )
                .await;
            let retry_at = mandate_retry_at(&mandate, None);
            let _ = self
                .state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await;
            anyhow::bail!("mandate backing goal is missing");
        };

        if goal.domain != "orchestration"
            || goal.goal_type != "continuous"
            || goal.status != "active"
        {
            let _ = self
                .state
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Active,
                    MandateStatus::Paused,
                )
                .await;
            let retry_at = mandate_retry_at(&mandate, None);
            let _ = self
                .state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await;
            let message = format!(
                "Mandate paused because its backing controller goal is not active: {}",
                self.presenter.goal_label(&mandate.objective)
            );
            let entry = crate::traits::NotificationEntry::new(
                &goal.id,
                &goal.session_id,
                "mandate_paused",
                &message,
            );
            let _ = self.state.enqueue_notification(&entry).await;
            return Ok(());
        }

        // The mandate lease prevents concurrent claims. The durable open-run
        // check is the crash/lease-expiry backstop: never create a second review
        // while an earlier mandate run still owns work for this controller.
        if let Some(open_run) = self.state.get_current_goal_run(&goal.id).await? {
            let retry_at = mandate_retry_at(&mandate, Some(mandate.min_review_secs));
            let released = self
                .state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await?;
            info!(
                mandate_id = %mandate.id,
                goal_id = %goal.id,
                run_id = %open_run.id,
                trigger_type = %open_run.trigger_type,
                lease_released = released,
                "Deferred mandate review because the controller already has an open run"
            );
            return Ok(());
        }

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
            let retry_at = mandate_budget_retry_at(&mandate, chrono::Utc::now());
            self.state
                .release_mandate_review_lease(&mandate.id, lease_token, &retry_at)
                .await?;
            info!(
                mandate_id = %mandate.id,
                goal_id = %goal.id,
                retry_at = %retry_at,
                "Deferred mandate review because its daily goal budget cannot fund a full cycle"
            );
            return Ok(());
        }

        let goal_run_id = uuid::Uuid::new_v4().to_string();
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let task = crate::traits::Task {
            id: root_task_id,
            goal_id: goal.id.clone(),
            description: mandate_review_task_description(&mandate, &goal_run_id),
            status: "pending".to_string(),
            priority: goal.priority.clone(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: Some(mandate_review_task_context(&mandate)),
            result: None,
            error: None,
            blocker: None,
            // A mandate can authorize external side effects. Never retry the
            // entire deliberation root after an ambiguous outcome.
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        let goal_run = self
            .state
            .create_mandate_review_run(&mandate.id, lease_token, &goal_run_id, &task)
            .await?;

        info!(
            mandate_id = %mandate.id,
            goal_id = %goal.id,
            run_id = %goal_run.id,
            task_id = %task.id,
            "Enqueued mandate review root task"
        );

        // Immediate dispatch avoids the orphan-recovery grace period. If the
        // agent or worker slot is unavailable, the pending task remains durable
        // and the ordinary orphan dispatcher may claim this same root later.
        if let Some(agent) = self.live_agent() {
            let worker = format!("heartbeat-dispatch-mandate-{}", uuid::Uuid::new_v4());
            match self
                .state
                .claim_task_with_lease(&task.id, &worker, Some("profile-task-lead"), 180)
                .await
            {
                Ok(Some(_)) => {
                    if let Some(registry) = self.goal_token_registry.as_ref() {
                        registry.register(&goal.id).await;
                    }
                    agent.spawn_task_lead(
                        goal.clone(),
                        task.description.clone(),
                        task.id.clone(),
                        None,
                    );
                }
                Ok(None) => {
                    tracing::debug!(
                        mandate_id = %mandate.id,
                        task_id = %task.id,
                        "Mandate root remains pending because no task-lead slot was available"
                    );
                }
                Err(error) => {
                    warn!(
                        mandate_id = %mandate.id,
                        task_id = %task.id,
                        %error,
                        "Failed to claim mandate root for immediate dispatch"
                    );
                }
            }
        }

        Ok(())
    }
}
