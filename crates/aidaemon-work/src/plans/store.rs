//! PlanStore - SQLite persistence for task plans.

use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};

use super::{PlanStatus, TaskPlan};

/// Persistent storage for task plans.
pub struct PlanStore {
    pool: SqlitePool,
}

/// One writer at a time for every read-change-store of a plan. A plan is
/// stored whole, so two writers that read it at once each store their own
/// copy and the later one drops the earlier one's change (two helpers one
/// `spawn_agent` batch starts link their steps concurrently). Process-wide,
/// so every `PlanStore` over the database shares it.
static PLAN_WRITES: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

impl PlanStore {
    /// Create a new PlanStore with the given database pool.
    /// Runs migrations to create the task_plans table.
    pub async fn new(pool: SqlitePool) -> anyhow::Result<Self> {
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    /// Get the underlying database pool.
    pub fn pool(&self) -> SqlitePool {
        self.pool.clone()
    }

    /// Run database migrations for the task_plans table.
    async fn migrate(&self) -> anyhow::Result<()> {
        migrate_task_plans(&self.pool).await
    }

    // =========================================================================
    // Write Operations
    // =========================================================================

    /// Create a new plan.
    pub async fn create(&self, plan: &TaskPlan) -> anyhow::Result<()> {
        let steps_json = serde_json::to_string(&plan.steps)?;
        let checkpoint_json = serde_json::to_string(&plan.checkpoint)?;

        sqlx::query(
            r#"
            INSERT INTO task_plans (
                id, session_id, description, trigger_message, steps,
                current_step, status, checkpoint, creation_reason,
                task_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&plan.id)
        .bind(&plan.session_id)
        .bind(&plan.description)
        .bind(&plan.trigger_message)
        .bind(&steps_json)
        .bind(plan.current_step as i64)
        .bind(plan.status.as_str())
        .bind(&checkpoint_json)
        .bind(&plan.creation_reason)
        .bind(&plan.task_id)
        .bind(plan.created_at.to_rfc3339())
        .bind(plan.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update an existing plan (full replacement).
    pub async fn update(&self, plan: &TaskPlan) -> anyhow::Result<()> {
        let steps_json = serde_json::to_string(&plan.steps)?;
        let checkpoint_json = serde_json::to_string(&plan.checkpoint)?;
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            UPDATE task_plans SET
                description = ?,
                trigger_message = ?,
                steps = ?,
                current_step = ?,
                status = ?,
                checkpoint = ?,
                creation_reason = ?,
                task_id = ?,
                updated_at = ?
            WHERE id = ?
            "#,
        )
        .bind(&plan.description)
        .bind(&plan.trigger_message)
        .bind(&steps_json)
        .bind(plan.current_step as i64)
        .bind(plan.status.as_str())
        .bind(&checkpoint_json)
        .bind(&plan.creation_reason)
        .bind(&plan.task_id)
        .bind(&now)
        .bind(&plan.id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// The session's latest plan, when it belongs to `task_id` (any task
    /// when `None`): the checklist a launch's `checklist_items` index into.
    pub async fn latest(
        &self,
        session_id: &str,
        task_id: Option<&str>,
    ) -> anyhow::Result<Option<TaskPlan>> {
        Ok(self
            .get_recent_for_session(session_id, 1)
            .await?
            .into_iter()
            .next()
            .filter(|plan| task_id.is_none() || plan.task_id.as_deref() == task_id))
    }

    /// The session's latest plan, when it belongs to `task_id` (any task
    /// when `None`), changed by `change` under the plan write lock and
    /// stored when `change` says it changed. Returns the stored plan, or
    /// `None` when there was no such plan or nothing changed.
    async fn change_latest(
        &self,
        session_id: &str,
        task_id: Option<&str>,
        change: impl FnOnce(&mut TaskPlan) -> bool,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let _write = PLAN_WRITES.lock().await;
        let Some(mut plan) = self
            .get_recent_for_session(session_id, 1)
            .await?
            .into_iter()
            .next()
            .filter(|plan| task_id.is_none() || plan.task_id.as_deref() == task_id)
        else {
            return Ok(None);
        };
        if !change(&mut plan) {
            return Ok(None);
        }
        self.update(&plan).await?;
        Ok(Some(plan))
    }

    /// Plan `plan_id` changed by `change` under the plan write lock, stored
    /// when `change` says it changed. Returns the plan as it now stands.
    pub async fn modify(
        &self,
        plan_id: &str,
        change: impl FnOnce(&mut TaskPlan) -> bool,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let _write = PLAN_WRITES.lock().await;
        let Some(mut plan) = self.get(plan_id).await? else {
            return Ok(None);
        };
        if change(&mut plan) {
            self.update(&plan).await?;
        }
        Ok(Some(plan))
    }

    /// Create or replace the active checklist for a session. Full-set replace:
    /// the provided items become the plan's steps. Reuses an existing incomplete
    /// plan (same id) when present so updates are idempotent.
    pub async fn upsert_checklist(
        &self,
        session_id: &str,
        task_id: Option<&str>,
        trigger: &str,
        items: &[(String, crate::plans::StepStatus)],
    ) -> anyhow::Result<TaskPlan> {
        let graph_items: Vec<crate::plans::ChecklistItem> = items
            .iter()
            .map(|(description, status)| crate::plans::ChecklistItem {
                description: description.clone(),
                status: *status,
                depends_on: Vec::new(),
                required_mutation_effects: crate::traits::ToolMutationEffects::NONE,
                requires_observation: false,
                expected_targets: Vec::new(),
            })
            .collect();
        self.upsert_checklist_graph(session_id, task_id, trigger, &graph_items)
            .await
    }

    /// Create or replace a checklist with explicit prerequisite edges.
    pub async fn upsert_checklist_graph(
        &self,
        session_id: &str,
        task_id: Option<&str>,
        trigger: &str,
        items: &[crate::plans::ChecklistItem],
    ) -> anyhow::Result<TaskPlan> {
        use crate::plans::{PlanStep, StepStatus};
        let now = Utc::now();
        let steps: Vec<PlanStep> = items
            .iter()
            .enumerate()
            .map(|(index, item)| PlanStep {
                index,
                description: item.description.clone(),
                depends_on: item.depends_on.clone(),
                required_mutation_effects: item.required_mutation_effects,
                requires_observation: item.requires_observation,
                expected_targets: item.expected_targets.clone(),
                evidence_receipt_ids: Vec::new(),
                background_jobs: Vec::new(),
                waited_jobs: Vec::new(),
                not_done: None,
                // The model's own tick: done on its word until evidence
                // confirms it.
                evidence: if item.status == StepStatus::Completed {
                    crate::plans::StepEvidence::ModelSaid
                } else {
                    crate::plans::StepEvidence::Settled
                },
                status: item.status,
                tool_call_ids: Vec::new(),
                result_summary: None,
                error: None,
                retry_count: 0,
                started_at: if matches!(item.status, StepStatus::InProgress | StepStatus::Completed)
                {
                    Some(now)
                } else {
                    None
                },
                completed_at: if matches!(item.status, StepStatus::Completed) {
                    Some(now)
                } else {
                    None
                },
            })
            .collect();
        let validation_plan = TaskPlan {
            id: "checklist-validation".to_string(),
            session_id: session_id.to_string(),
            description: trigger.to_string(),
            trigger_message: trigger.to_string(),
            steps: steps.clone(),
            current_step: 0,
            status: PlanStatus::InProgress,
            checkpoint: serde_json::Value::Object(serde_json::Map::new()),
            creation_reason: "graph_validation".to_string(),
            task_id: task_id.map(str::to_string),
            created_at: now,
            updated_at: now,
        };
        validation_plan
            .validate_graph()
            .map_err(anyhow::Error::msg)?;
        let current_step = steps
            .iter()
            .position(|s| matches!(s.status, StepStatus::Pending | StepStatus::InProgress))
            .unwrap_or(0);

        let _write = PLAN_WRITES.lock().await;
        // Only this request's own open plan is reused: a plan belongs to
        // the request that declared it, and a turn that continues that
        // request has adopted it ([`Self::carry_into_turn`]). An earlier
        // request's open plan is never the new request's (live 2026-11-14:
        // a new request's card showed "✅ Install WordPress … verify the
        // site" from a run hours before).
        let reusable = self
            .get_incomplete_for_session(session_id)
            .await?
            .filter(|plan| plan.task_id.as_deref() == task_id);
        if let Some(mut existing) = reusable {
            // A step a background job is still doing stays with that job:
            // a checklist update cannot finish or drop work it did not do.
            // Only the job's end settles it.
            let mut steps = steps;
            for step in &mut steps {
                let Some(previous) = existing.steps.get(step.index) else {
                    continue;
                };
                // The same step keeps the evidence that proved it: an
                // update ticking it again neither erases its proof nor
                // turns proven work into the model's word.
                if previous.description == step.description {
                    for receipt in &previous.evidence_receipt_ids {
                        if !step.evidence_receipt_ids.contains(receipt) {
                            step.evidence_receipt_ids.push(receipt.clone());
                        }
                    }
                    if step.status == StepStatus::Completed
                        && previous.status == StepStatus::Completed
                        && previous.evidence != crate::plans::StepEvidence::ModelSaid
                    {
                        step.evidence = crate::plans::StepEvidence::Settled;
                        step.completed_at = previous.completed_at;
                    }
                    // Still the step whose helper didn't start, until a
                    // launch takes it or the model resolves it.
                    if previous.helper_not_started()
                        && matches!(step.status, StepStatus::Pending | StepStatus::InProgress)
                    {
                        step.status = StepStatus::Failed;
                        step.evidence = crate::plans::StepEvidence::HelperNotStarted;
                        step.error = previous.error.clone();
                        step.completed_at = previous.completed_at;
                    }
                }
                if previous.status == StepStatus::Background
                    && previous.has_background_jobs()
                    && previous.description == step.description
                {
                    step.status = StepStatus::Background;
                    step.background_jobs = previous.background_jobs.clone();
                    step.waited_jobs = previous.waited_jobs.clone();
                    step.started_at = previous.started_at;
                    step.completed_at = None;
                }
            }
            existing.steps = steps;
            existing.current_step = current_step;
            existing.updated_at = now;
            existing.sync_active_step();
            self.update(&existing).await?;
            Ok(existing)
        } else {
            let mut plan = TaskPlan::new(
                session_id,
                trigger,
                "Requirement checklist",
                Vec::new(),
                "track_requirements",
            );
            plan.steps = steps;
            plan.current_step = current_step;
            plan.task_id = task_id.map(|t| t.to_string());
            plan.sync_active_step();
            self.create(&plan).await?;
            Ok(plan)
        }
    }

    /// Hand steps of the session's checklist for `task_id` to the background
    /// job `job_id` ([`TaskPlan::link_background_job`]). Returns the updated
    /// plan, or `None` when there is no such checklist or nothing changed.
    pub async fn link_background_job(
        &self,
        session_id: &str,
        task_id: Option<&str>,
        indices: &[usize],
        job_id: &str,
    ) -> anyhow::Result<Option<TaskPlan>> {
        self.change_latest(session_id, task_id, |plan| {
            plan.link_background_job(indices, job_id)
        })
        .await
    }

    /// The helper for the steps at `indices` of the session's checklist for
    /// `task_id` never started ([`TaskPlan::helper_did_not_start`]).
    /// Returns the updated plan, or `None` when nothing changed.
    pub async fn helper_did_not_start(
        &self,
        session_id: &str,
        task_id: Option<&str>,
        indices: &[usize],
        reason: &str,
    ) -> anyhow::Result<Option<TaskPlan>> {
        self.change_latest(session_id, task_id, |plan| {
            plan.helper_did_not_start(indices, reason)
        })
        .await
    }

    /// The session's checklist for `task_id`: its latest plan, when that
    /// plan belongs to the task.
    pub async fn for_task(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<Option<TaskPlan>> {
        Ok(self
            .get_recent_for_session(session_id, 1)
            .await?
            .into_iter()
            .next()
            .filter(|plan| plan.task_id.as_deref() == Some(task_id)))
    }

    /// The run's receipts proved the checklist items at `indices` of the
    /// session's checklist for `task_id` ([`TaskPlan::confirm_steps`]).
    /// Returns the updated plan, or `None` when nothing changed.
    pub async fn confirm_steps(
        &self,
        session_id: &str,
        task_id: &str,
        indices: &std::collections::BTreeSet<usize>,
    ) -> anyhow::Result<Option<TaskPlan>> {
        self.change_latest(session_id, Some(task_id), |plan| {
            plan.confirm_steps(indices)
        })
        .await
    }

    /// Settle the steps at `indices` of the session's checklist for
    /// `task_id` as not attempted ([`TaskPlan::mark_not_attempted`]).
    pub async fn mark_not_attempted(
        &self,
        session_id: &str,
        task_id: &str,
        indices: &std::collections::BTreeSet<usize>,
    ) -> anyhow::Result<Option<TaskPlan>> {
        self.change_latest(session_id, Some(task_id), |plan| {
            plan.mark_not_attempted(indices)
        })
        .await
    }

    /// The session's checklist for `task_id` waits on background jobs
    /// `job_ids` ([`TaskPlan::wait_on_background_jobs`]). Returns the updated
    /// plan, or `None` when there is no such checklist or nothing changed.
    pub async fn wait_on_background_jobs(
        &self,
        session_id: &str,
        task_id: Option<&str>,
        job_ids: &[String],
    ) -> anyhow::Result<Option<TaskPlan>> {
        self.change_latest(session_id, task_id, |plan| {
            plan.wait_on_background_jobs(job_ids)
        })
        .await
    }

    /// Record the end of background job `job_id` on the session's checklist
    /// that handed it work ([`TaskPlan::settle_background_job`]). Returns the
    /// updated plan, or `None` when no checklist step waited on that job.
    pub async fn settle_background_job(
        &self,
        session_id: &str,
        job_id: &str,
        end: super::JobEnd,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let _write = PLAN_WRITES.lock().await;
        for mut plan in self.get_recent_for_session(session_id, 5).await? {
            if plan.settle_background_job(job_id, end) {
                self.update(&plan).await?;
                return Ok(Some(plan));
            }
        }
        Ok(None)
    }

    /// Settle plan `plan_id`'s open steps as not done
    /// ([`TaskPlan::settle_open_steps`]). Returns the plan as it now stands.
    pub async fn settle_open_steps(&self, plan_id: &str) -> anyhow::Result<Option<TaskPlan>> {
        self.modify(plan_id, TaskPlan::settle_open_steps).await
    }

    /// Update just the status of a plan.
    pub async fn set_status(&self, plan_id: &str, status: PlanStatus) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            UPDATE task_plans SET status = ?, updated_at = ? WHERE id = ?
            "#,
        )
        .bind(status.as_str())
        .bind(&now)
        .bind(plan_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Set a checkpoint value (merges into existing checkpoint).
    pub async fn set_checkpoint(
        &self,
        plan_id: &str,
        key: &str,
        value: serde_json::Value,
    ) -> anyhow::Result<()> {
        // Fetch current checkpoint
        let row = sqlx::query("SELECT checkpoint FROM task_plans WHERE id = ?")
            .bind(plan_id)
            .fetch_optional(&self.pool)
            .await?;

        let mut checkpoint: serde_json::Map<String, serde_json::Value> = match row {
            Some(r) => {
                let json_str: String = r.get("checkpoint");
                serde_json::from_str(&json_str).unwrap_or_default()
            }
            None => return Err(anyhow::anyhow!("Plan not found: {}", plan_id)),
        };

        // Merge new value
        checkpoint.insert(key.to_string(), value);

        let checkpoint_json = serde_json::to_string(&checkpoint)?;
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            UPDATE task_plans SET checkpoint = ?, updated_at = ? WHERE id = ?
            "#,
        )
        .bind(&checkpoint_json)
        .bind(&now)
        .bind(plan_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Reconcile a successful tool-call receipt into explicit checklist evidence
    /// contracts. Step descriptions are display-only and never decide completion.
    pub async fn reconcile_checklist_after_tool_success(
        &self,
        session_id: &str,
        semantics: &crate::traits::ToolCallSemantics,
        receipt_id: &str,
        result: &str,
    ) -> anyhow::Result<Option<TaskPlan>> {
        use crate::plans::StepStatus;

        let Some(mut plan) = self.get_incomplete_for_session(session_id).await? else {
            return Ok(None);
        };

        let mut changed = false;
        for step in &mut plan.steps {
            if !matches!(step.status, StepStatus::Pending | StepStatus::InProgress) {
                continue;
            }
            let has_evidence_contract =
                !step.required_mutation_effects.is_empty() || step.requires_observation;
            if !has_evidence_contract {
                continue;
            }
            let effects_match = step.required_mutation_effects.is_empty()
                || semantics
                    .mutation_effects
                    .satisfies(step.required_mutation_effects);
            let observation_matches = !step.requires_observation || semantics.observes_state();
            let targets_match = step.expected_targets.is_empty()
                || step.expected_targets.iter().all(|expected| {
                    semantics
                        .target_hints
                        .iter()
                        .any(|actual| actual.value == *expected)
                });

            if effects_match && observation_matches && targets_match {
                step.status = StepStatus::Completed;
                step.evidence = crate::plans::StepEvidence::Settled;
                step.completed_at = Some(Utc::now());
                step.result_summary = Some(aidaemon_events::text::truncate_str(result, 200));
                if !step
                    .evidence_receipt_ids
                    .iter()
                    .any(|existing| existing == receipt_id)
                {
                    step.evidence_receipt_ids.push(receipt_id.to_string());
                }
                changed = true;
            }
        }

        if !changed {
            return Ok(None);
        }

        plan.sync_active_step();
        self.update(&plan).await?;
        Ok(Some(plan))
    }

    /// A turn that continues the same request (a runtime continuation, a
    /// resume, or a turn that adopted the request's open obligations)
    /// carries the request's open checklist: the session's open plan moves
    /// to this turn's task, so its card, its helpers' links and its outcome
    /// all read it. A new request calls nothing here and starts its own
    /// plan. Returns the carried plan.
    pub async fn carry_into_turn(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let _write = PLAN_WRITES.lock().await;
        let Some(mut plan) = self.get_incomplete_for_session(session_id).await? else {
            return Ok(None);
        };
        if plan.task_id.as_deref() != Some(task_id) {
            plan.task_id = Some(task_id.to_string());
            plan.updated_at = Utc::now();
            self.update(&plan).await?;
        }
        Ok(Some(plan))
    }

    /// Link a plan to a task ID from the event store.
    pub async fn set_task_id(&self, plan_id: &str, task_id: &str) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            r#"
            UPDATE task_plans SET task_id = ?, updated_at = ? WHERE id = ?
            "#,
        )
        .bind(task_id)
        .bind(&now)
        .bind(plan_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // =========================================================================
    // Read Operations
    // =========================================================================

    /// Get a plan by ID.
    pub async fn get(&self, plan_id: &str) -> anyhow::Result<Option<TaskPlan>> {
        let row = sqlx::query(
            r#"
            SELECT id, session_id, description, trigger_message, steps,
                   current_step, status, checkpoint, creation_reason,
                   task_id, created_at, updated_at
            FROM task_plans WHERE id = ?
            "#,
        )
        .bind(plan_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(r) => Ok(Some(self.row_to_plan(&r)?)),
            None => Ok(None),
        }
    }

    /// Get the incomplete plan for a session (if any).
    /// Returns the most recently updated incomplete plan.
    pub async fn get_incomplete_for_session(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<TaskPlan>> {
        let row = sqlx::query(
            r#"
            SELECT id, session_id, description, trigger_message, steps,
                   current_step, status, checkpoint, creation_reason,
                   task_id, created_at, updated_at
            FROM task_plans
            WHERE session_id = ?
              AND status IN ('planning', 'in_progress', 'paused')
            ORDER BY updated_at DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(r) => Ok(Some(self.row_to_plan(&r)?)),
            None => Ok(None),
        }
    }

    /// Get recent plans for a session.
    pub async fn get_recent_for_session(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<TaskPlan>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, description, trigger_message, steps,
                   current_step, status, checkpoint, creation_reason,
                   task_id, created_at, updated_at
            FROM task_plans
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut plans = Vec::new();
        for row in rows {
            plans.push(self.row_to_plan(&row)?);
        }
        Ok(plans)
    }

    /// Get all plans that are in progress.
    pub async fn get_all_in_progress(&self) -> anyhow::Result<Vec<TaskPlan>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, description, trigger_message, steps,
                   current_step, status, checkpoint, creation_reason,
                   task_id, created_at, updated_at
            FROM task_plans
            WHERE status = 'in_progress'
            ORDER BY updated_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut plans = Vec::new();
        for row in rows {
            plans.push(self.row_to_plan(&row)?);
        }
        Ok(plans)
    }

    /// Get completed plans since a given time (for consolidation).
    pub async fn get_completed_since(
        &self,
        session_id: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<TaskPlan>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, description, trigger_message, steps,
                   current_step, status, checkpoint, creation_reason,
                   task_id, created_at, updated_at
            FROM task_plans
            WHERE session_id = ?
              AND status = 'completed'
              AND updated_at >= ?
            ORDER BY updated_at DESC
            "#,
        )
        .bind(session_id)
        .bind(since.to_rfc3339())
        .fetch_all(&self.pool)
        .await?;

        let mut plans = Vec::new();
        for row in rows {
            plans.push(self.row_to_plan(&row)?);
        }
        Ok(plans)
    }

    // =========================================================================
    // Cleanup Operations
    // =========================================================================

    /// Delete old completed/failed/abandoned plans.
    pub async fn delete_old_completed(&self, older_than: DateTime<Utc>) -> anyhow::Result<u64> {
        let result = sqlx::query(
            r#"
            DELETE FROM task_plans
            WHERE status IN ('completed', 'failed', 'abandoned')
              AND updated_at < ?
            "#,
        )
        .bind(older_than.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected())
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    fn row_to_plan(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<TaskPlan> {
        let id: String = row.get("id");
        let session_id: String = row.get("session_id");
        let description: String = row.get("description");
        let trigger_message: String = row.get("trigger_message");
        let steps_json: String = row.get("steps");
        let current_step: i64 = row.get("current_step");
        let status_str: String = row.get("status");
        let checkpoint_json: String = row.get("checkpoint");
        let creation_reason: String = row.get("creation_reason");
        let task_id: Option<String> = row.get("task_id");
        let created_at_str: String = row.get("created_at");
        let updated_at_str: String = row.get("updated_at");

        let steps = steps_from_json(&steps_json)?;
        let checkpoint = serde_json::from_str(&checkpoint_json)?;
        let status = PlanStatus::from_str(&status_str).unwrap_or(PlanStatus::InProgress);
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)?.with_timezone(&Utc);
        let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)?.with_timezone(&Utc);

        Ok(TaskPlan {
            id,
            session_id,
            description,
            trigger_message,
            steps,
            current_step: current_step as usize,
            status,
            checkpoint,
            creation_reason,
            task_id,
            created_at,
            updated_at,
        })
    }
}

/// Creates the `task_plans` table and its indexes; idempotent.
pub async fn migrate_task_plans(pool: &SqlitePool) -> anyhow::Result<()> {
    // Create task_plans table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS task_plans (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            description TEXT NOT NULL,
            trigger_message TEXT NOT NULL,
            steps TEXT NOT NULL,
            current_step INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'in_progress',
            checkpoint TEXT NOT NULL DEFAULT '{}',
            creation_reason TEXT NOT NULL,
            task_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await?;

    // Index for finding incomplete plans for a session
    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_plans_session_status
        ON task_plans(session_id, status)
        "#,
    )
    .execute(pool)
    .await?;

    // Index for cleanup of old completed plans
    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_plans_updated
        ON task_plans(updated_at)
        "#,
    )
    .execute(pool)
    .await?;

    tracing::info!("Task plans table migration complete");
    Ok(())
}

/// A plan's steps as stored. Rows written before steps carried a typed
/// [`crate::plans::StepEvidence`] kept it as two flags (`said_done`,
/// `helper_not_started`); they read as the evidence they meant.
fn steps_from_json(json: &str) -> anyhow::Result<Vec<crate::plans::PlanStep>> {
    use crate::plans::StepEvidence;
    let mut raw: Vec<serde_json::Value> = serde_json::from_str(json)?;
    for step in &mut raw {
        let Some(fields) = step.as_object_mut() else {
            continue;
        };
        let flag = |fields: &mut serde_json::Map<String, serde_json::Value>, name: &str| {
            fields
                .remove(name)
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        };
        let said_done = flag(fields, "said_done");
        let not_started = flag(fields, "helper_not_started");
        // Rows written before a step kept the jobs it waits on apart listed
        // them as its own jobs under a `waits_on_jobs` flag.
        if flag(fields, "waits_on_jobs") && !fields.contains_key("waited_jobs") {
            let jobs = fields
                .remove("background_jobs")
                .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
            fields.insert("waited_jobs".to_string(), jobs);
        }
        if fields.contains_key("evidence") {
            continue;
        }
        let evidence = if not_started {
            StepEvidence::HelperNotStarted
        } else if said_done {
            StepEvidence::ModelSaid
        } else {
            continue;
        };
        fields.insert("evidence".to_string(), serde_json::to_value(evidence)?);
    }
    raw.into_iter()
        .map(|step| serde_json::from_value(step).map_err(anyhow::Error::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn create_test_store() -> PlanStore {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        PlanStore::new(pool).await.unwrap()
    }

    /// A step the model ticks is done on its word. A check or a read needs
    /// no receipt: the reply reports what it found, so the card shows it
    /// done with no note. Only an action step that declared a side effect
    /// and that nothing verified is flagged, in plain words; evidence (a
    /// receipt, a helper's success) clears the flag, and an update ticking
    /// the step again keeps the proof (live 2026-11-13: "Finished · 3 of 3
    /// done, 1 not confirmed by a tool" on a read-only check).
    #[tokio::test]
    async fn only_an_action_step_nothing_verified_is_flagged() {
        use crate::plans::{ChecklistItem, StepEvidence, StepStatus};
        use crate::traits::ToolMutationEffects;
        let store = create_test_store().await;
        let item = |text: &str, effects: ToolMutationEffects, status: StepStatus| ChecklistItem {
            description: text.to_string(),
            status,
            depends_on: Vec::new(),
            required_mutation_effects: effects,
            requires_observation: false,
            expected_targets: Vec::new(),
        };
        let ticked = |done: bool| {
            let status = if done {
                StepStatus::Completed
            } else {
                StepStatus::Pending
            };
            vec![
                item(
                    "Check whether ~/projects/claude-test-a.txt is already occupied",
                    ToolMutationEffects::NONE,
                    status,
                ),
                item(
                    "Write the letters file",
                    ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    status,
                ),
                item(
                    "Report both",
                    ToolMutationEffects::NONE,
                    StepStatus::InProgress,
                ),
            ]
        };
        store
            .upsert_checklist_graph("s1", Some("t1"), "test", &ticked(false))
            .await
            .unwrap();
        let plan = store
            .upsert_checklist_graph("s1", Some("t1"), "test", &ticked(true))
            .await
            .unwrap();
        let card = plan.render_compact_checklist();
        assert!(card.starts_with("⚙️ Working · step 3 of 3"), "{card}");
        assert!(
            card.contains("✅ Check whether ~/projects/claude-test-a.txt is already occupied\n"),
            "{card}"
        );
        assert!(
            card.contains("☑️ Write the letters file (I couldn't verify this one)"),
            "{card}"
        );
        assert_eq!(plan.unverified_steps(), 1);

        let proven = std::collections::BTreeSet::from([0]);
        store.confirm_steps("s1", "t1", &proven).await.unwrap();
        let plan = store
            .upsert_checklist_graph("s1", Some("t1"), "test", &ticked(true))
            .await
            .unwrap();
        assert_eq!(
            plan.steps[0].evidence,
            StepEvidence::Settled,
            "a later tick keeps the proof"
        );
        assert_eq!(plan.steps[1].evidence, StepEvidence::ModelSaid);

        store
            .link_background_job("s1", Some("t1"), &[2], "job-1")
            .await
            .unwrap()
            .unwrap();
        let plan = store
            .settle_background_job("s1", "job-1", crate::plans::JobEnd::Succeeded)
            .await
            .unwrap()
            .unwrap();
        let card = plan.render_compact_checklist();
        assert!(
            card.starts_with("☑️ Finished · 3 of 3 done; I couldn't verify 1"),
            "{card}"
        );
        assert!(card.contains("✅ Report both"), "{card}");
        assert!(!crate::plans::is_completed_checklist(&card), "{card}");
        assert!(!card.to_lowercase().contains("tool"), "{card}");

        // Every step a check: done, with nothing to flag.
        let checks = vec![item(
            "Check the first site",
            ToolMutationEffects::NONE,
            StepStatus::Completed,
        )];
        let plan = store
            .upsert_checklist_graph("s2", Some("t2"), "test", &checks)
            .await
            .unwrap();
        assert_eq!(plan.render_compact_checklist(), "✅ Check the first site");
    }

    /// Rows stored before steps carried typed evidence read as the
    /// evidence their flags meant.
    #[test]
    fn legacy_step_flags_read_as_typed_evidence() {
        use crate::plans::StepEvidence;
        let mut plan = TaskPlan::new("s", "t", "d", vec!["a".into(), "b".into(), "c".into()], "r");
        plan.steps[0].evidence = StepEvidence::ModelSaid;
        let mut raw = serde_json::to_value(&plan.steps).unwrap();
        let steps = raw.as_array_mut().unwrap();
        steps[0].as_object_mut().unwrap().remove("evidence");
        steps[0]["said_done"] = true.into();
        steps[1]["helper_not_started"] = true.into();
        let read = steps_from_json(&raw.to_string()).unwrap();
        assert_eq!(read[0].evidence, StepEvidence::ModelSaid);
        assert_eq!(read[1].evidence, StepEvidence::HelperNotStarted);
        assert_eq!(read[2].evidence, StepEvidence::Settled);
    }

    /// Rows stored before a step kept the jobs it only waits on apart read
    /// those jobs as waited on, never as its own.
    #[test]
    fn legacy_waiting_flag_reads_as_waited_jobs() {
        let plan = TaskPlan::new("s", "t", "d", vec!["a".into(), "b".into()], "r");
        let mut raw = serde_json::to_value(&plan.steps).unwrap();
        let steps = raw.as_array_mut().unwrap();
        steps[0]["background_jobs"] = serde_json::json!(["job-a"]);
        steps[0]["waits_on_jobs"] = true.into();
        steps[1]["background_jobs"] = serde_json::json!(["job-b"]);
        let read = steps_from_json(&raw.to_string()).unwrap();
        assert!(read[0].background_jobs.is_empty());
        assert_eq!(read[0].waited_jobs, vec!["job-a".to_string()]);
        assert_eq!(read[1].background_jobs, vec!["job-b".to_string()]);
        assert!(read[1].waited_jobs.is_empty());
    }

    /// A step whose helper never started is not done, the finished card says
    /// plainly that it didn't run, and a later launch for it takes it again.
    #[tokio::test]
    async fn a_step_whose_helper_did_not_start_reads_as_not_run_until_one_does() {
        use crate::plans::StepStatus;
        let store = create_test_store().await;
        let items = vec![
            ("Install the first site".to_string(), StepStatus::InProgress),
            ("Install the second site".to_string(), StepStatus::Pending),
        ];
        store
            .upsert_checklist("s1", Some("t1"), "test", &items)
            .await
            .unwrap();
        store
            .link_background_job("s1", Some("t1"), &[0], "job-1")
            .await
            .unwrap()
            .unwrap();
        store
            .helper_did_not_start("s1", Some("t1"), &[1], "its helper didn't start")
            .await
            .unwrap()
            .unwrap();
        let plan = store
            .settle_background_job("s1", "job-1", crate::plans::JobEnd::Succeeded)
            .await
            .unwrap()
            .unwrap();
        let card = plan.render_compact_checklist();
        assert!(card.starts_with("❌ Stopped · 1 of 2 done"), "{card}");
        assert!(
            card.contains("⚠️ Install the second site (didn't run)"),
            "{card}"
        );
        // Why is said once, under the rows.
        assert!(
            card.ends_with("Didn't run: its helper didn't start."),
            "{card}"
        );
        assert_eq!(card.matches("helper didn't start").count(), 1, "{card}");

        let plan = store
            .link_background_job("s1", Some("t1"), &[1], "job-2")
            .await
            .unwrap()
            .expect("a retried launch takes the step");
        assert_eq!(plan.steps[1].status, StepStatus::Background);
        assert!(!plan.steps[1].helper_not_started());
        assert_eq!(plan.status, PlanStatus::InProgress);
    }

    #[tokio::test]
    async fn typed_receipt_completes_only_its_exact_checklist_contract() {
        use crate::plans::{ChecklistItem, StepStatus};
        use crate::traits::{ToolCallSemantics, ToolMutationEffects, ToolTargetHintKind};
        let store = create_test_store().await;
        store
            .upsert_checklist_graph(
                "s1",
                None,
                "test",
                &[ChecklistItem {
                    description: "Opaque synthetic requirement".into(),
                    status: StepStatus::Pending,
                    depends_on: Vec::new(),
                    required_mutation_effects: ToolMutationEffects::EXTERNAL_DELIVERY,
                    requires_observation: false,
                    expected_targets: vec!["/tmp/report.pdf".into()],
                }],
            )
            .await
            .unwrap();

        let wrong_target = ToolCallSemantics::mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY)
            .with_target_hint(ToolTargetHintKind::Path, "/tmp/other.pdf");
        assert!(
            store
                .reconcile_checklist_after_tool_success(
                    "s1",
                    &wrong_target,
                    "receipt-wrong",
                    "delivered",
                )
                .await
                .unwrap()
                .is_none()
        );

        let exact = ToolCallSemantics::mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY)
            .with_target_hint(ToolTargetHintKind::Path, "/tmp/report.pdf");
        store
            .reconcile_checklist_after_tool_success("s1", &exact, "receipt-exact", "delivered")
            .await
            .unwrap();

        let plan = store
            .get_recent_for_session("s1", 1)
            .await
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(plan.steps[0].status, StepStatus::Completed);
        assert_eq!(plan.steps[0].evidence_receipt_ids, vec!["receipt-exact"]);
        assert_eq!(plan.status, PlanStatus::Completed);
    }

    #[tokio::test]
    async fn checklist_prose_cannot_complete_without_a_typed_evidence_contract() {
        use crate::plans::StepStatus;
        use crate::traits::{ToolCallSemantics, ToolMutationEffects};
        let store = create_test_store().await;
        store
            .upsert_checklist(
                "s1",
                None,
                "test",
                &[("Send the report file now".into(), StepStatus::Pending)],
            )
            .await
            .unwrap();
        let delivery = ToolCallSemantics::mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY);
        assert!(
            store
                .reconcile_checklist_after_tool_success("s1", &delivery, "receipt-1", "delivered",)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_upsert_checklist_creates_then_replaces() {
        use crate::plans::StepStatus;
        let store = create_test_store().await;
        let items = vec![
            ("create script".to_string(), StepStatus::InProgress),
            ("send the file".to_string(), StepStatus::Pending),
        ];
        let plan = store
            .upsert_checklist("sess-1", Some("task-1"), "make+send", &items)
            .await
            .unwrap();
        assert_eq!(plan.steps.len(), 2);
        let id = plan.id.clone();

        let items2 = vec![
            ("create script".to_string(), StepStatus::Completed),
            ("send the file".to_string(), StepStatus::Completed),
        ];
        let plan2 = store
            .upsert_checklist("sess-1", Some("task-1"), "make+send", &items2)
            .await
            .unwrap();
        assert_eq!(
            plan2.id, id,
            "should update existing plan, not create a new one"
        );
        assert_eq!(plan2.completed_steps(), 2);

        // Persisted state reflects the replacement.
        let fetched = store.get(&id).await.unwrap().unwrap();
        assert_eq!(fetched.completed_steps(), 2);
    }

    /// Each owner request gets its own plan: a new request's checklist
    /// never reuses (or shows) an earlier request's open plan, while a turn
    /// that continues the request carries it (live 2026-11-14: a new
    /// request's card showed "✅ Install WordPress … verify the site").
    #[tokio::test]
    async fn a_new_request_gets_its_own_plan_and_a_continuation_carries_it() {
        use crate::plans::StepStatus;
        let store = create_test_store().await;
        let earlier = store
            .upsert_checklist(
                "chat",
                Some("request-1"),
                "wp",
                &[
                    (
                        "Install WordPress and verify the site".to_string(),
                        StepStatus::Completed,
                    ),
                    ("Send the login link".to_string(), StepStatus::Pending),
                ],
            )
            .await
            .unwrap();

        let fresh = store
            .upsert_checklist(
                "chat",
                Some("request-2"),
                "drupal",
                &[
                    ("Install Drupal".to_string(), StepStatus::InProgress),
                    ("Verify Drupal".to_string(), StepStatus::Pending),
                ],
            )
            .await
            .unwrap();
        assert_ne!(fresh.id, earlier.id, "a new request starts its own plan");
        let card = fresh.card().text;
        assert!(!card.contains("WordPress"), "{card}");
        let kept = store.get(&earlier.id).await.unwrap().unwrap();
        assert_eq!(kept.task_id.as_deref(), Some("request-1"));
        assert_eq!(kept.steps.len(), 2);

        // A later turn of request 2 carries its plan.
        let carried = store
            .carry_into_turn("chat", "request-2-turn-2")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(carried.id, fresh.id);
        let again = store
            .upsert_checklist(
                "chat",
                Some("request-2-turn-2"),
                "drupal",
                &[
                    ("Install Drupal".to_string(), StepStatus::Completed),
                    ("Verify Drupal".to_string(), StepStatus::InProgress),
                ],
            )
            .await
            .unwrap();
        assert_eq!(again.id, fresh.id);
        assert!(
            store
                .for_task("chat", "request-2-turn-2")
                .await
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn test_create_and_get() {
        let store = create_test_store().await;

        let plan = TaskPlan::new(
            "session_123",
            "Deploy the app",
            "Production deployment",
            vec![
                "Test".to_string(),
                "Build".to_string(),
                "Deploy".to_string(),
            ],
            "high_stakes",
        );

        store.create(&plan).await.unwrap();

        let retrieved = store.get(&plan.id).await.unwrap().unwrap();
        assert_eq!(retrieved.id, plan.id);
        assert_eq!(retrieved.description, "Production deployment");
        assert_eq!(retrieved.steps.len(), 3);
    }

    #[tokio::test]
    async fn test_get_incomplete_for_session() {
        let store = create_test_store().await;

        // Create a completed plan
        let mut plan1 = TaskPlan::new(
            "session_123",
            "Task 1",
            "First task",
            vec!["Step".to_string()],
            "test",
        );
        plan1.status = PlanStatus::Completed;
        store.create(&plan1).await.unwrap();

        // Create an in-progress plan
        let plan2 = TaskPlan::new(
            "session_123",
            "Task 2",
            "Second task",
            vec!["Step".to_string()],
            "test",
        );
        store.create(&plan2).await.unwrap();

        // Should find the in-progress plan
        let incomplete = store
            .get_incomplete_for_session("session_123")
            .await
            .unwrap();
        assert!(incomplete.is_some());
        assert_eq!(incomplete.unwrap().description, "Second task");

        // Different session should find nothing
        let other = store
            .get_incomplete_for_session("session_456")
            .await
            .unwrap();
        assert!(other.is_none());
    }

    #[tokio::test]
    async fn test_update_status() {
        let store = create_test_store().await;

        let plan = TaskPlan::new(
            "session_123",
            "Test",
            "Test task",
            vec!["Step".to_string()],
            "test",
        );
        store.create(&plan).await.unwrap();

        store
            .set_status(&plan.id, PlanStatus::Paused)
            .await
            .unwrap();

        let retrieved = store.get(&plan.id).await.unwrap().unwrap();
        assert_eq!(retrieved.status, PlanStatus::Paused);
    }

    #[tokio::test]
    async fn test_checkpoint() {
        let store = create_test_store().await;

        let plan = TaskPlan::new(
            "session_123",
            "Test",
            "Test task",
            vec!["Step".to_string()],
            "test",
        );
        store.create(&plan).await.unwrap();

        store
            .set_checkpoint(&plan.id, "image_tag", serde_json::json!("v1.2.3"))
            .await
            .unwrap();
        store
            .set_checkpoint(&plan.id, "commit_sha", serde_json::json!("abc123"))
            .await
            .unwrap();

        let retrieved = store.get(&plan.id).await.unwrap().unwrap();
        assert_eq!(retrieved.checkpoint["image_tag"], "v1.2.3");
        assert_eq!(retrieved.checkpoint["commit_sha"], "abc123");
    }

    /// A checklist update cannot settle a step a background job is doing:
    /// it stays with the job until the job ends.
    #[tokio::test]
    async fn checklist_update_keeps_steps_a_background_job_is_doing() {
        use crate::plans::StepStatus;
        let store = create_test_store().await;
        let items = |status: StepStatus| {
            vec![
                ("inspect".to_string(), StepStatus::Completed),
                ("set up".to_string(), status),
            ]
        };
        store
            .upsert_checklist(
                "s",
                Some("t"),
                "track_requirements",
                &items(StepStatus::Pending),
            )
            .await
            .unwrap();
        store
            .link_background_job("s", Some("t"), &[1], "job-a")
            .await
            .unwrap()
            .expect("linked");

        let plan = store
            .upsert_checklist(
                "s",
                Some("t"),
                "track_requirements",
                &items(StepStatus::Deferred),
            )
            .await
            .unwrap();
        assert_eq!(plan.steps[1].status, StepStatus::Background);
        assert_eq!(plan.steps[1].background_jobs, vec!["job-a".to_string()]);

        let plan = store
            .settle_background_job("s", "job-a", crate::plans::JobEnd::Succeeded)
            .await
            .unwrap()
            .expect("settled");
        assert_eq!(plan.steps[1].status, StepStatus::Completed);
        assert_eq!(plan.status, PlanStatus::Completed);
    }

    /// Live 2026-11-13: the model marked the two helper steps "background",
    /// then one `spawn_agent` batch started both helpers without naming
    /// steps. The marked steps take both launches (the launches link
    /// concurrently and neither link is lost), and both runs' success
    /// settles them done: the card's header says done, as its total does.
    #[tokio::test]
    async fn a_batch_of_two_delegated_steps_settles_done_from_both_runs() {
        use crate::plans::{ChecklistItem, JobEnd, StepStatus};
        use std::sync::Arc;
        let store = Arc::new(create_test_store().await);
        let item = |text: &str, status: StepStatus| ChecklistItem {
            description: text.to_string(),
            status,
            depends_on: Vec::new(),
            // The helpers' steps change their sites; the check and the
            // report change nothing.
            required_mutation_effects: if status == StepStatus::Background {
                crate::traits::ToolMutationEffects::LOCAL_WORKSPACE_WRITE
            } else {
                crate::traits::ToolMutationEffects::NONE
            },
            requires_observation: false,
            expected_targets: Vec::new(),
        };
        store
            .upsert_checklist_graph(
                "chat",
                Some("t"),
                "track_requirements",
                &[
                    item("Check both sites", StepStatus::Completed),
                    item(
                        "Have a helper finish the Drupal site",
                        StepStatus::Background,
                    ),
                    item(
                        "Have a helper finish the WordPress site",
                        StepStatus::Background,
                    ),
                    item("Report each helper's result", StepStatus::Pending),
                ],
            )
            .await
            .unwrap();
        let jobs = ["job-drupal", "job-wordpress"].map(|job| vec![job.to_string()]);
        let (first, second) = tokio::join!(
            store.wait_on_background_jobs("chat", Some("t"), &jobs[0]),
            store.wait_on_background_jobs("chat", Some("t"), &jobs[1]),
        );
        assert!(first.unwrap().is_some() && second.unwrap().is_some());
        let plan = store.for_task("chat", "t").await.unwrap().unwrap();
        for (index, step) in plan.steps.iter().enumerate().skip(1) {
            assert_eq!(step.status, StepStatus::Background, "step {index}");
            assert_eq!(
                step.waited_jobs,
                vec!["job-drupal".to_string(), "job-wordpress".to_string()],
                "step {index} waits on both launches"
            );
        }
        assert!(
            plan.card()
                .text
                .starts_with("⏳ Continuing in the background · 1 of 4 done")
        );

        let plan = store
            .settle_background_job("chat", "job-drupal", JobEnd::Succeeded)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            plan.steps[1].status,
            StepStatus::Background,
            "one run still runs"
        );
        let plan = store
            .settle_background_job("chat", "job-wordpress", JobEnd::Succeeded)
            .await
            .unwrap()
            .unwrap();
        assert!(
            plan.steps
                .iter()
                .all(|step| step.status == StepStatus::Completed)
        );
        assert_eq!(plan.status, PlanStatus::Completed);
        let card = plan.card().text;
        assert!(card.starts_with("✅ Done · 4 of 4 steps"), "{card}");
        assert!(!card.contains("not done"), "{card}");
    }
}
