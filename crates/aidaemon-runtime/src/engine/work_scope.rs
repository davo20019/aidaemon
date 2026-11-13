//! Which work an agent's turn belongs to: the goal its tokens charge, the
//! mandate its tool calls answer to, whether it runs as trusted scheduled
//! automation, and the chat whose session mode narrows it ([`WorkScope`],
//! on any [`TurnHost`]).

use crate::host::TurnHost;

/// Where a turn's work belongs, read from the agent and the work stores.
#[async_trait::async_trait]
pub trait WorkScope: TurnHost {
    /// The goal whose token budget this agent's turns charge: its own goal,
    /// or the goal of the task it executes.
    async fn budget_goal_id(&self, session_id: &str) -> Option<String> {
        if let Some(gid) = self.goal_id() {
            return Some(gid.to_string());
        }
        let tid = self.task_id()?;
        match self.tasks().get_task(tid).await {
            Ok(Some(task)) => Some(task.goal_id),
            Ok(None) => {
                tracing::warn!(
                    session_id,
                    task_id = %tid,
                    "Task not found while resolving goal_id; goal budget enforcement disabled for this run"
                );
                None
            }
            Err(e) => {
                tracing::warn!(
                    session_id,
                    task_id = %tid,
                    error = %e,
                    "Failed to resolve goal_id from task; goal budget enforcement disabled for this run"
                );
                None
            }
        }
    }

    /// The mandate goal a tool call of this agent answers to: a mandate
    /// worker's own, else the goal (this agent's, or its task's or the
    /// call's task's) that a mandate governs.
    async fn mandate_goal_id_for_tool_exec(
        &self,
        call_task_id: Option<&str>,
    ) -> anyhow::Result<Option<String>> {
        // A run-bound child remains a mandate execution even if its controller
        // record is concurrently removed or corrupted. Never downgrade that
        // child into the ordinary owner/tool path.
        if let Some(fence) = self.mandate_execution() {
            anyhow::ensure!(
                self.goal_id() == Some(fence.goal_id.as_str()),
                "Mandate child goal identity is inconsistent."
            );
            return Ok(Some(fence.goal_id.clone()));
        }
        if let Some(goal_id) = self.goal_id() {
            if self
                .mandates()
                .get_mandate_for_goal(goal_id)
                .await?
                .is_some()
            {
                return Ok(Some(goal_id.to_string()));
            }
        }
        for task_id in [self.task_id(), call_task_id].into_iter().flatten() {
            if let Some(task) = self.tasks().get_task(task_id).await? {
                if self
                    .mandates()
                    .get_mandate_for_goal(&task.goal_id)
                    .await?
                    .is_some()
                {
                    return Ok(Some(task.goal_id));
                }
            }
        }
        Ok(None)
    }

    /// Whether calls of this agent's turn on `task_id` run as a trusted
    /// scheduled session: the channel says so, or the goal or task has
    /// scheduled provenance. Callers exclude mandate runs, whose exact
    /// action grants are their only authority.
    async fn session_trusted(&self, channel_trusted: bool, task_id: Option<&str>) -> bool {
        use aidaemon_work::scheduled_runs::{
            goal_has_scheduled_provenance, task_has_scheduled_provenance,
        };
        if channel_trusted {
            return true;
        }
        let work = self.work();
        if let Some(goal_id) = self.goal_id() {
            // Scheduled goal runs are user-confirmed automation, so treat
            // their tool calls as trusted even when the execution context
            // was recreated later by heartbeat/orphan dispatch.
            goal_has_scheduled_provenance(work, goal_id, self.task_id()).await
        } else if let Some(executor_task_id) = self.task_id() {
            if let Ok(Some(task)) = self.tasks().get_task(executor_task_id).await {
                goal_has_scheduled_provenance(work, &task.goal_id, Some(executor_task_id)).await
            } else {
                task_has_scheduled_provenance(work, Some(executor_task_id)).await
            }
        } else if let Some(task_id) = task_id {
            task_has_scheduled_provenance(work, Some(task_id)).await
        } else {
            false
        }
    }

    /// Revalidate the immutable mandate run and exact task-attempt lease. This
    /// runs before grant issuance and again at the final adapter boundary.
    async fn require_live_mandate_execution(&self, expected_goal_id: &str) -> anyhow::Result<()> {
        const FENCE_RENEWAL_SECS: i64 = 180;

        let fence = self.mandate_execution().ok_or_else(|| {
            anyhow::anyhow!("Mandate execution is missing its immutable run/task fence.")
        })?;
        anyhow::ensure!(
            fence.goal_id == expected_goal_id
                && self.goal_id() == Some(expected_goal_id)
                && self.task_id() == Some(fence.worker_task_id.as_str()),
            "Mandate execution identity does not match this child agent."
        );
        let mandate = self
            .mandates()
            .get_mandate_for_goal(expected_goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate authority record disappeared."))?;
        anyhow::ensure!(
            mandate.id == fence.mandate_id
                && mandate.version == fence.mandate_version
                && mandate.authority == fence.authority
                && mandate.is_active(),
            "Mandate authority changed or is no longer active."
        );

        let task = self
            .tasks()
            .get_task(&fence.worker_task_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate worker task no longer exists."))?;
        anyhow::ensure!(
            task.goal_id == expected_goal_id
                && matches!(task.status.as_str(), "claimed" | "running")
                && !task.has_error()
                && !task.has_blocker(),
            "Mandate worker task is no longer executable."
        );

        let attempt = self
            .work()
            .get_current_task_attempt(&fence.worker_task_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate task-attempt lease is no longer active."))?;
        anyhow::ensure!(
            attempt.id == fence.attempt_id
                && attempt.lease_token == fence.lease_token
                && attempt.task_id == fence.worker_task_id
                && attempt.goal_run_id == fence.goal_run_id
                && matches!(attempt.status.as_str(), "claimed" | "running"),
            "Mandate task-attempt lease was replaced or lost."
        );
        anyhow::ensure!(
            self.work()
                .heartbeat_task_attempt(&fence.attempt_id, &fence.lease_token, FENCE_RENEWAL_SECS,)
                .await?,
            "Mandate task-attempt lease expired or was revoked."
        );

        // Reload after the atomic lease check so a closed cycle cannot be
        // rebound by goal ID to a later run.
        let run = self
            .work()
            .get_current_goal_run(expected_goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Mandate decision cycle is no longer active."))?;
        anyhow::ensure!(
            run.id == fence.goal_run_id
                && run.goal_id == expected_goal_id
                && run.trigger_type == "mandate"
                && run.status == "running"
                && run.root_task_id.as_deref() == Some(fence.root_task_id.as_str()),
            "Mandate execution is stale, blocked, or bound to a different decision cycle."
        );
        Ok(())
    }

    /// Refuse an executor spawn for `task_id` unless the task exists, belongs
    /// to `expected_goal_id` (when given) and is claimed: no executor for an
    /// unclaimed, stale, running or finished task.
    async fn validate_executor_task_for_spawn(
        &self,
        task_id: &str,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let Some(task) = self.tasks().get_task(task_id).await? else {
            anyhow::bail!(
                "Task '{}' was not found. Use manage_goal_tasks(list_tasks) and pass a valid task_id.",
                task_id
            );
        };

        if let Some(goal_id) = expected_goal_id {
            if task.goal_id != goal_id {
                anyhow::bail!(
                    "Task '{}' belongs to goal '{}', not '{}'.",
                    task_id,
                    task.goal_id,
                    goal_id
                );
            }
        }

        match task.status.as_str() {
            "claimed" => Ok(()),
            "pending" => anyhow::bail!(
                "Task '{}' is still pending. Claim it first with manage_goal_tasks(action='claim_task').",
                task_id
            ),
            "running" => anyhow::bail!(
                "Task '{}' is already running. Do not spawn another executor for the same task.",
                task_id
            ),
            "completed" | "failed" | "blocked" | "cancelled" => anyhow::bail!(
                "Task '{}' is '{}' and should not be executed again without an explicit retry/reset.",
                task_id,
                task.status
            ),
            other => anyhow::bail!(
                "Task '{}' has unsupported status '{}' for executor spawn (expected 'claimed').",
                task_id,
                other
            ),
        }
    }

    /// The chat whose `/readonly` narrows this turn while it runs: a child
    /// answers to the chat that started the work, a root turn to its own.
    fn mode_session_for(&self, session_id: &str) -> String {
        self.inherited_authority()
            .and_then(|inherited| inherited.mode_session.clone())
            .unwrap_or_else(|| session_id.to_string())
    }
}

impl<H: TurnHost + ?Sized> WorkScope for H {}
