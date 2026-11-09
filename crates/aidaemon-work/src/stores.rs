//! The store traits for goals, mandates, tasks, work and notifications.

use async_trait::async_trait;

/// Goal lifecycle and confirmation-flow persistence. Task, schedule, budget,
/// scheduled-run, dispatch, and notification concerns live in the sibling
/// traits: [`TaskStore`], [`GoalScheduleStore`], [`GoalBudgetStore`],
/// [`ScheduledRunStore`], [`TaskDispatchStore`], and [`GoalNotificationStore`].
#[async_trait]
pub trait GoalStore: Send + Sync {
    /// Create a new goal.
    async fn create_goal(&self, _goal: &crate::traits::Goal) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a goal by ID.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_goal(&self, _id: &str) -> anyhow::Result<Option<crate::traits::Goal>> {
        Ok(None)
    }

    /// Update a goal (full replacement).
    #[allow(dead_code)] // Used in Phase 2
    async fn update_goal(&self, _goal: &crate::traits::Goal) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get all active orchestration goals (status = "active" or "pending").
    #[allow(dead_code)] // Used in Phase 2
    async fn get_active_goals(&self) -> anyhow::Result<Vec<crate::traits::Goal>> {
        Ok(vec![])
    }

    /// Get active personal goals (tracked, never dispatched).
    async fn get_active_personal_goals(&self, _limit: i64) -> anyhow::Result<Vec<crate::traits::Goal>> {
        Ok(vec![])
    }

    /// Update a personal goal's status and/or append a progress note.
    async fn update_personal_goal(
        &self,
        _goal_id: &str,
        _status: Option<&str>,
        _progress_note: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get orchestration goals for a specific session.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_goals_for_session(&self, _session_id: &str) -> anyhow::Result<Vec<crate::traits::Goal>> {
        Ok(vec![])
    }

    /// Get scheduled goals awaiting confirmation in a session.
    #[allow(dead_code)]
    async fn get_pending_confirmation_goals(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::Goal>> {
        Ok(vec![])
    }

    /// Activate a pending-confirmation goal.
    /// Returns true when the status transition was applied.
    async fn activate_goal(&self, _goal_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }
}

/// Owner mandates, agent deliberation cycles, and intentions.
#[async_trait]
pub trait MandateStore: Send + Sync {
    /// Atomically create the continuous controller goal and its mandate.
    /// Adaptive review timing lives on the mandate, not in `goal_schedules`.
    async fn create_mandate_controller(
        &self,
        _goal: &crate::traits::Goal,
        _mandate: &crate::traits::Mandate,
    ) -> anyhow::Result<()> {
        anyhow::bail!("mandate controllers are not supported by this store")
    }

    async fn get_mandate(&self, _id: &str) -> anyhow::Result<Option<crate::traits::Mandate>> {
        Ok(None)
    }

    async fn get_mandate_for_goal(&self, _goal_id: &str) -> anyhow::Result<Option<crate::traits::Mandate>> {
        Ok(None)
    }

    async fn list_mandates(
        &self,
        _session_id: Option<&str>,
        _include_terminal: bool,
    ) -> anyhow::Result<Vec<crate::traits::Mandate>> {
        Ok(Vec::new())
    }

    /// Authorize one exact presentation session through the mandate's stable
    /// principal mapping. Session-string similarity is never sufficient.
    async fn is_mandate_session_authorized(
        &self,
        _mandate_id: &str,
        _session_id: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Atomically bind a verified target session to the stable principal and
    /// move the mandate's delivery/controller route. The authority epoch is
    /// advanced and all open runs are invalidated before the move commits.
    async fn transfer_mandate_ownership(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
        _from_session_id: &str,
        _to_session_id: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Append one receipt-backed metric observation for the exact current
    /// mandate run and policy version.
    async fn record_mandate_objective_measurement(
        &self,
        _measurement: &crate::traits::MandateObjectiveMeasurement,
    ) -> anyhow::Result<()> {
        anyhow::bail!("mandate objective measurements are not supported by this store")
    }

    async fn list_mandate_objective_measurements(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::MandateObjectiveMeasurement>> {
        Ok(Vec::new())
    }

    /// Optimistic owner-policy update. `mandate.version` must be exactly one
    /// greater than the currently stored version.
    async fn update_mandate(&self, _mandate: &crate::traits::Mandate) -> anyhow::Result<()> {
        anyhow::bail!("mandates are not supported by this store")
    }

    /// Atomically consume an explicit owner confirmation and activate both the
    /// mandate and its continuous controller. This is the only path that may
    /// turn an unconfirmed mandate into active authority.
    async fn confirm_mandate(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
        _activation_duration_secs: Option<i64>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Atomic lifecycle/authority-epoch transition. Implementations must
    /// invalidate open decision runs and stale ACT authority before returning.
    async fn transition_mandate_status(
        &self,
        _mandate_id: &str,
        _from_status: crate::traits::MandateStatus,
        _to_status: crate::traits::MandateStatus,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Resume a paused/awaiting-input mandate and update its controller
    /// context in the same expected-version transaction. This prevents owner
    /// guidance from overwriting a concurrent cancel/pause lifecycle state.
    async fn resume_mandate_with_context(
        &self,
        _mandate_id: &str,
        _from_status: crate::traits::MandateStatus,
        _expected_version: i64,
        _controller_context: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Record a late owner answer on an active mandate by replacing its
    /// controller context, fenced on the mandate still being active at the
    /// expected version for this owner. Authority is never touched.
    async fn record_active_mandate_owner_guidance(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
        _owner_session: &str,
        _controller_context: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Atomically create one mandate review run and its root task under the
    /// caller's current review lease. A failed root insert must
    /// leave no open run behind.
    async fn create_mandate_review_run(
        &self,
        _mandate_id: &str,
        _review_lease_token: &str,
        _goal_run_id: &str,
        _root_task: &crate::traits::Task,
    ) -> anyhow::Result<crate::traits::GoalRun> {
        anyhow::bail!("mandate review runs are not supported by this store")
    }

    /// Repair controller runtime state only while the exact mandate authority
    /// epoch is still active. The state check and goal update must be one
    /// atomic database statement so a racing owner pause/cancel wins.
    async fn keep_mandate_controller_active(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Claim due reviews with a durable lease. Concurrent callers must never
    /// receive the same mandate before the lease expires.
    async fn claim_due_mandates(
        &self,
        _limit: i64,
        _lease_owner: &str,
        _lease_secs: i64,
    ) -> anyhow::Result<Vec<crate::traits::Mandate>> {
        Ok(Vec::new())
    }

    async fn release_mandate_review_lease(
        &self,
        _mandate_id: &str,
        _lease_token: &str,
        _retry_at: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Insert one non-root action task into an exact mandate run while the
    /// dispatcher-owned root attempt remains current. Implementations must
    /// validate the mandate/run/attempt fence and task cap atomically and must
    /// never create or rebind a goal run by goal id.
    async fn create_mandate_task_from_attempt(
        &self,
        _task: &crate::traits::Task,
        _mandate_id: &str,
        _mandate_version: i64,
        _goal_run_id: &str,
        _root_task_attempt_id: &str,
        _max_non_root_tasks: i64,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Claim one pending non-root task inside an exact running mandate cycle.
    /// The root task-attempt remains the control-plane lease; the returned
    /// child attempt deliberately carries no worker-profile/workspace binding.
    /// The independently fenced identifiers remain explicit because wrapping
    /// them would churn the stable store API without reducing validation work.
    #[allow(clippy::too_many_arguments)]
    async fn claim_mandate_task_from_attempt(
        &self,
        _task_id: &str,
        _worker_instance_id: &str,
        _mandate_id: &str,
        _mandate_version: i64,
        _goal_run_id: &str,
        _root_task_attempt_id: &str,
        _lease_secs: i64,
    ) -> anyhow::Result<Option<crate::traits::TaskAttempt>> {
        Ok(None)
    }

    /// Insert exactly one deliberation result for a goal run, plus its ACT
    /// intention when present. Implementations must commit both atomically.
    async fn record_mandate_decision(
        &self,
        _decision: &crate::traits::MandateDecisionCycle,
        _intention: Option<&crate::traits::Intention>,
        _task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("mandate decisions are not supported by this store")
    }

    async fn record_mandate_decision_with_updates(
        &self,
        decision: &crate::traits::MandateDecisionCycle,
        intention: Option<&crate::traits::Intention>,
        operating_updates: Option<&crate::traits::MandateOperatingUpdates>,
        task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            operating_updates.is_none_or(|updates| {
                updates.learning_note.is_none() && updates.strategy_revisions.is_empty()
            }),
            "adaptive mandate operating updates are not supported by this store"
        );
        self.record_mandate_decision(decision, intention, task_attempt_id)
            .await
    }

    async fn get_mandate_decision_for_run(
        &self,
        _goal_run_id: &str,
    ) -> anyhow::Result<Option<crate::traits::MandateDecisionCycle>> {
        Ok(None)
    }

    /// Tool-call ids of the run's successful structured receipts that may
    /// back a decision or measurement (newest first, at most `limit`): the
    /// same receipts `evidence_receipt_ids` accepts, minus the mandate
    /// protocol's own control tools.
    async fn current_run_evidence_receipt_ids(
        &self,
        _goal_run_id: &str,
        _limit: usize,
    ) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }

    async fn list_mandate_decisions(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::MandateDecisionCycle>> {
        Ok(Vec::new())
    }

    async fn list_intentions(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::Intention>> {
        Ok(Vec::new())
    }

    /// Persist one bounded, advisory learning note only after all cited
    /// receipts are proven to belong to this mandate.
    #[allow(dead_code)]
    async fn record_mandate_learning_note(
        &self,
        _note: &crate::traits::MandateLearningNote,
    ) -> anyhow::Result<()> {
        anyhow::bail!("mandate learning is not supported by this store")
    }

    async fn list_mandate_learning_notes(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::MandateLearningNote>> {
        Ok(Vec::new())
    }

    /// Return the latest adaptive strategy node for each key. Retired nodes
    /// remain visible for audit but must not be treated as active guidance.
    async fn list_current_mandate_strategy(
        &self,
        _mandate_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::MandateStrategyRevision>> {
        Ok(Vec::new())
    }

    /// Deduplicate and route one content-free external signal to matching
    /// Autopilot mandates. Matching is structural and cannot widen authority.
    async fn wake_mandates_for_signal(
        &self,
        _signal: &crate::traits::MandateWakeSignal,
    ) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Resolve an awaiting-input state with a typed owner action. Question
    /// answers and external-effect reconciliation are intentionally distinct.
    #[allow(clippy::too_many_arguments)]
    async fn resolve_mandate_suspension(
        &self,
        _mandate_id: &str,
        _expected_version: i64,
        _expected_kind: crate::traits::MandateSuspensionKind,
        _controller_context: Option<&str>,
        _reconciliation_resolution: Option<crate::traits::MandateReconciliationResolution>,
        _owner_guidance: &str,
        _owner_session: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Atomically reserve the exact next mutation attempt under the current
    /// cycle, rolling-24h, cooldown, mandate, run, intention, task, and lease
    /// fences. Implementations must insert the durable ledger row and advance
    /// the cycle counter in one transaction.
    async fn reserve_mandate_action_attempt(
        &self,
        _reservation: &crate::traits::MandateMutationReservation,
    ) -> anyhow::Result<Option<crate::traits::MandateMutationAttempt>> {
        Ok(None)
    }

    /// Atomically claim one exact reserved mutation at the last common
    /// dispatcher. A claim is one-use and must fail if any authority, run,
    /// task-attempt, tool-call, digest, or reservation field is stale.
    async fn claim_mandate_mutation_dispatch(
        &self,
        _claim: &crate::traits::MandateMutationDispatchClaim,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Read the typed, cross-cycle mutation quota projection at one instant.
    /// This is advisory for deliberation; reservation remains authoritative.
    async fn get_mandate_mutation_quota_state(
        &self,
        _mandate_id: &str,
        _as_of: &str,
    ) -> anyhow::Result<Option<crate::traits::MandateMutationQuotaState>> {
        Ok(None)
    }

    async fn list_mandate_mutation_attempts_for_run(
        &self,
        _goal_run_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::MandateMutationAttempt>> {
        Ok(Vec::new())
    }

    /// Atomically prove and close a mandate run. Satisfied outcomes close the
    /// goal run (and ACT intention) inside the proof transaction; reconciliation
    /// results leave lifecycle state untouched for the caller's explicit path.
    async fn finalize_mandate_run_from_proof(
        &self,
        _request: &crate::traits::MandateRunFinalizationRequest,
    ) -> anyhow::Result<crate::traits::MandateRunFinalizationResult> {
        Ok(crate::traits::MandateRunFinalizationResult::Rejected {
            reason: crate::traits::MandateFinalizationRejectReason::InvalidRequest,
        })
    }
}

/// Task persistence and task activity logs.
#[async_trait]
pub trait TaskStore: Send + Sync {
    /// Create a new task within a goal.
    #[allow(dead_code)] // Used in Phase 2
    async fn create_task(&self, _task: &crate::traits::Task) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a task by ID.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_task(&self, _id: &str) -> anyhow::Result<Option<crate::traits::Task>> {
        Ok(None)
    }

    /// Update a task (full replacement).
    #[allow(dead_code)] // Used in Phase 2
    async fn update_task(&self, _task: &crate::traits::Task) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get all tasks for a goal.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_tasks_for_goal(&self, _goal_id: &str) -> anyhow::Result<Vec<crate::traits::Task>> {
        Ok(vec![])
    }

    /// Count completed/skipped tasks for a goal (used by progress-based circuit breaker).
    #[allow(dead_code)] // Retained for stores and diagnostics; run-scoped code filters task rows.
    async fn count_completed_tasks_for_goal(&self, _goal_id: &str) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Atomically claim a pending task for an executor.
    #[allow(dead_code)] // Used in Phase 2
    async fn claim_task(&self, _task_id: &str, _agent_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Log an activity entry for a task.
    #[allow(dead_code)] // Used in Phase 2
    async fn log_task_activity(&self, _activity: &crate::traits::TaskActivity) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get activity log for a task.
    #[allow(dead_code)] // Used in Phase 2
    async fn get_task_activities(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::TaskActivity>> {
        Ok(vec![])
    }
}

/// Goal schedule and scheduled-run persistence.
#[async_trait]
pub trait GoalScheduleStore: Send + Sync {
    /// Create a new schedule for a goal.
    async fn create_goal_schedule(&self, _schedule: &crate::traits::GoalSchedule) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get a schedule by ID.
    async fn get_goal_schedule(
        &self,
        _schedule_id: &str,
    ) -> anyhow::Result<Option<crate::traits::GoalSchedule>> {
        Ok(None)
    }

    /// List schedules for a goal.
    async fn get_schedules_for_goal(
        &self,
        _goal_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::GoalSchedule>> {
        Ok(vec![])
    }

    /// Get due schedules for active orchestration goals.
    async fn get_due_goal_schedules(
        &self,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::GoalSchedule>> {
        Ok(vec![])
    }

    /// Update a schedule (full replacement).
    async fn update_goal_schedule(&self, _schedule: &crate::traits::GoalSchedule) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete a schedule by ID. Returns true if a row was deleted.
    async fn delete_goal_schedule(&self, _schedule_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Cancel pending-confirmation goals older than max_age_secs.
    async fn cancel_stale_pending_confirmation_goals(
        &self,
        _max_age_secs: i64,
    ) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Get all orchestration goals that have schedules or are awaiting confirmation.
    async fn get_scheduled_goals(&self) -> anyhow::Result<Vec<crate::traits::Goal>> {
        Ok(vec![])
    }

    /// Active continuous orchestration goals with no schedule and no mandate
    /// behind them: nothing will ever run them. Listed so they can be seen
    /// and cancelled.
    async fn get_unscheduled_active_goals(&self) -> anyhow::Result<Vec<crate::traits::Goal>> {
        Ok(vec![])
    }
}

/// Goal token budget persistence and accounting.
#[async_trait]
pub trait GoalBudgetStore: Send + Sync {
    /// Reset tokens_used_today only for active goals whose usage belongs to a
    /// prior UTC day. Safe to call repeatedly or at process startup.
    async fn reset_daily_token_budgets(&self) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Update budget columns only. `None` = keep current value (COALESCE), NOT "clear to NULL".
    /// This is a targeted UPDATE that avoids the race with `add_goal_tokens_and_get_budget_status()`
    /// which atomically increments `tokens_used_today` — a full `update_goal()` would clobber that.
    async fn set_goal_budgets(
        &self,
        _goal_id: &str,
        _budget_per_check: Option<i64>,
        _budget_daily: Option<i64>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Atomically add tokens to a goal's daily usage counter and return budget status.
    ///
    /// Use `delta_tokens = 0` to read the latest counters without modifying them.
    async fn add_goal_tokens_and_get_budget_status(
        &self,
        _goal_id: &str,
        _delta_tokens: i64,
    ) -> anyhow::Result<Option<crate::traits::GoalTokenBudgetStatus>> {
        Ok(None)
    }
}

/// Scheduled run runtime-state persistence.
#[async_trait]
pub trait ScheduledRunStore: Send + Sync {
    /// Persist runtime state for an active scheduled run.
    async fn upsert_scheduled_run_state(
        &self,
        _state: &crate::traits::ScheduledRunState,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Load persisted runtime state for an active scheduled run.
    async fn get_scheduled_run_state(
        &self,
        _goal_id: &str,
    ) -> anyhow::Result<Option<crate::traits::ScheduledRunState>> {
        Ok(None)
    }

    async fn get_scheduled_recovery_state(
        &self,
        _goal_id: &str,
    ) -> anyhow::Result<Option<crate::traits::ScheduledRecoveryState>> {
        Ok(None)
    }

    /// Escalated scheduled objectives that the recovery machine itself paused,
    /// whose cool-down has elapsed and whose automatic recovery attempts are
    /// still under the cap.
    async fn list_escalated_recovery_candidates(
        &self,
        _cooldown_secs: i64,
        _max_attempts: u16,
    ) -> anyhow::Result<Vec<crate::traits::ScheduledRecoveryState>> {
        Ok(Vec::new())
    }

    /// Record that an automatic recovery run was launched for the goal.
    async fn record_scheduled_recovery_attempt(&self, _goal_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete persisted runtime state for an active scheduled run.
    async fn delete_scheduled_run_state(&self, _goal_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }
}

/// Task dispatch bookkeeping for executor scheduling and recovery.
#[async_trait]
pub trait TaskDispatchStore: Send + Sync {
    /// Get pending tasks ordered by priority, filtering out those with unmet dependencies.
    async fn get_pending_tasks_by_priority(&self, _limit: i64) -> anyhow::Result<Vec<crate::traits::Task>> {
        Ok(vec![])
    }

    /// Get tasks stuck in running/claimed state longer than timeout_secs.
    async fn get_stuck_tasks(&self, _timeout_secs: i64) -> anyhow::Result<Vec<crate::traits::Task>> {
        Ok(vec![])
    }

    /// Get tasks completed after a given timestamp.
    #[allow(dead_code)]
    async fn get_recently_completed_tasks(&self, _since: &str) -> anyhow::Result<Vec<crate::traits::Task>> {
        Ok(vec![])
    }

    /// Mark a running/claimed task as interrupted (e.g., after crash or timeout).
    async fn mark_task_interrupted(&self, _task_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }
}

/// Durable coordination for goal runs, fenced task attempts, collaboration
/// records, named worker policies, project scope, and attempt workspaces.
#[async_trait]
pub trait WorkCoordinationStore: Send + Sync {
    async fn create_work_project(
        &self,
        _name: &str,
        _description: Option<&str>,
    ) -> anyhow::Result<crate::traits::WorkProject> {
        anyhow::bail!("work projects are not supported by this store")
    }

    async fn list_work_projects(&self) -> anyhow::Result<Vec<crate::traits::WorkProject>> {
        Ok(vec![])
    }

    async fn get_session_work_project(&self, _session_id: &str) -> anyhow::Result<String> {
        Ok(crate::traits::DEFAULT_PROJECT_ID.to_string())
    }

    async fn set_session_work_project(
        &self,
        _session_id: &str,
        _project_id: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn start_goal_run(
        &self,
        _goal_id: &str,
        _trigger_type: &str,
        _schedule_id: Option<&str>,
        _root_task_id: Option<&str>,
    ) -> anyhow::Result<crate::traits::GoalRun> {
        anyhow::bail!("goal runs are not supported by this store")
    }

    async fn get_current_goal_run(&self, _goal_id: &str) -> anyhow::Result<Option<crate::traits::GoalRun>> {
        Ok(None)
    }

    async fn get_goal_runs(&self, _goal_id: &str) -> anyhow::Result<Vec<crate::traits::GoalRun>> {
        Ok(vec![])
    }

    /// Typed lifecycle owner for a task. Callers must use this relationship,
    /// never infer run provenance from the task description.
    async fn get_goal_run_for_task(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<crate::traits::GoalRun>> {
        Ok(None)
    }

    async fn get_tasks_for_goal_run(&self, _run_id: &str) -> anyhow::Result<Vec<crate::traits::Task>> {
        Ok(vec![])
    }

    async fn finish_goal_run(
        &self,
        _run_id: &str,
        _status: &str,
        _summary: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn claim_task_with_lease(
        &self,
        _task_id: &str,
        _worker_instance_id: &str,
        _worker_profile_id: Option<&str>,
        _lease_secs: i64,
    ) -> anyhow::Result<Option<crate::traits::TaskAttempt>> {
        Ok(None)
    }

    async fn get_current_task_attempt(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<crate::traits::TaskAttempt>> {
        Ok(None)
    }

    async fn bind_task_attempt_worker(
        &self,
        _attempt_id: &str,
        _lease_token: &str,
        _worker_instance_id: &str,
        _worker_profile_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn heartbeat_task_attempt(
        &self,
        _attempt_id: &str,
        _lease_token: &str,
        _lease_secs: i64,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn patch_task_from_attempt(
        &self,
        _attempt_id: &str,
        _lease_token: &str,
        _patch: &crate::traits::TaskAttemptPatch,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    /// Expire elapsed leases and return the affected task IDs. Idempotent tasks
    /// with retry budget are re-queued; ambiguous writes require verification.
    async fn recover_expired_task_attempts(&self) -> anyhow::Result<Vec<String>> {
        Ok(vec![])
    }

    /// Whether any worker still holds an unexpired execution lease on an
    /// attempt belonging to this goal run. While one does, that worker owns
    /// the run's outcome; lifecycle reconciliation must not race it.
    async fn goal_run_has_live_attempt(&self, _run_id: &str) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn append_task_journal(&self, _entry: &crate::traits::TaskJournalEntry) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_task_journal(
        &self,
        _task_id: &str,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::TaskJournalEntry>> {
        Ok(vec![])
    }

    async fn unblock_task(
        &self,
        _task_id: &str,
        _resolution: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn retry_work_task(
        &self,
        _task_id: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn cancel_work_task(
        &self,
        _task_id: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn upsert_worker_profile(&self, _profile: &crate::traits::WorkerProfile) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_worker_profile(
        &self,
        _profile_id: &str,
    ) -> anyhow::Result<Option<crate::traits::WorkerProfile>> {
        Ok(None)
    }

    async fn list_worker_profiles(
        &self,
        _project_id: Option<&str>,
    ) -> anyhow::Result<Vec<crate::traits::WorkerProfile>> {
        Ok(vec![])
    }

    async fn assign_task_worker_profile(
        &self,
        _task_id: &str,
        _profile_id: &str,
        _actor_id: &str,
        _source_channel: Option<&str>,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn set_task_workspace_policy(
        &self,
        _task_id: &str,
        _policy: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }

    async fn get_task_workspace_policy(&self, _task_id: &str) -> anyhow::Result<String> {
        Ok("shared".to_string())
    }

    async fn create_task_workspace(&self, _workspace: &crate::traits::TaskWorkspace) -> anyhow::Result<()> {
        Ok(())
    }

    async fn update_task_workspace(&self, _workspace: &crate::traits::TaskWorkspace) -> anyhow::Result<()> {
        Ok(())
    }

    async fn get_task_workspace(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<crate::traits::TaskWorkspace>> {
        Ok(None)
    }

    async fn get_latest_task_handoff(
        &self,
        _task_id: &str,
    ) -> anyhow::Result<Option<crate::traits::TaskHandoff>> {
        Ok(None)
    }

    async fn list_work_goals(
        &self,
        _project_id: &str,
        _include_terminal: bool,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::WorkGoalSummary>> {
        Ok(vec![])
    }

    async fn list_work_tasks(
        &self,
        _project_id: &str,
        _lane: Option<&str>,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::WorkTaskSummary>> {
        Ok(vec![])
    }
}

/// Goal lifecycle cleanup and user notification bookkeeping.
#[async_trait]
pub trait GoalNotificationStore: Send + Sync {
    /// Count active evergreen (continuous) goals.
    async fn count_active_evergreen_goals(&self) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get goals that completed/failed but haven't been notified to the user yet.
    async fn get_goals_needing_notification(&self) -> anyhow::Result<Vec<crate::traits::Goal>> {
        Ok(vec![])
    }

    /// Mark a goal as notified (set notified_at timestamp).
    async fn mark_goal_notified(&self, _goal_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Mark stale active goals as abandoned/failed.
    ///
    /// - Finite orchestration goals: active goals with no update in `stale_hours` → failed
    /// - Continuous orchestration goals: skipped (they have their own idle detection)
    /// - Personal goals: skipped
    ///
    /// Returns the number of goals cleaned up.
    async fn cleanup_stale_goals(&self, _stale_hours: i64) -> anyhow::Result<u64> {
        Ok(0)
    }
}

/// Notification delivery queue persistence.
#[async_trait]
pub trait NotificationStore: Send + Sync {
    /// Record a background sub-agent that has started.
    async fn record_background_spawn(&self, _spawn: &crate::traits::BackgroundSpawn) -> anyhow::Result<()> {
        Ok(())
    }

    /// Forget a background sub-agent that has ended.
    async fn finish_background_spawn(&self, _id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// The background sub-agents running for a session.
    async fn background_spawns_for_session(
        &self,
        _session_id: &str,
    ) -> anyhow::Result<Vec<crate::traits::BackgroundSpawn>> {
        Ok(Vec::new())
    }

    /// Remove and return every recorded background sub-agent.
    async fn take_background_spawns(&self) -> anyhow::Result<Vec<crate::traits::BackgroundSpawn>> {
        Ok(Vec::new())
    }

    /// Enqueue a notification for delivery.
    async fn enqueue_notification(&self, _entry: &crate::traits::NotificationEntry) -> anyhow::Result<()> {
        Ok(())
    }

    /// Atomically claim a goal's notification slot and enqueue its notification.
    ///
    /// Returns `false` when another writer has already claimed the goal. Stores
    /// that do not support transactions may fall back to enqueueing only.
    async fn enqueue_goal_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
    ) -> anyhow::Result<bool> {
        self.enqueue_notification(entry).await?;
        Ok(true)
    }

    /// Get pending notifications ordered by priority (critical first), then creation time.
    async fn get_pending_notifications(
        &self,
        _limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::NotificationEntry>> {
        Ok(vec![])
    }

    /// Mark a notification as delivered.
    async fn mark_notification_delivered(&self, _notification_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Increment the attempt counter for a notification.
    async fn increment_notification_attempt(&self, _notification_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Delete expired status_update notifications (past their expires_at).
    async fn cleanup_expired_notifications(&self) -> anyhow::Result<i64> {
        Ok(0)
    }
}
