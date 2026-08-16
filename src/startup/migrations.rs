//! Startup-only data migrations that operate through domain store ports.
//!
//! Keeping compatibility migrations out of the composition root prevents
//! `core` from accumulating persistence-specific cleanup rules. The SQLite
//! pool is passed explicitly only for the one projection table that has no
//! domain-store operation yet.

use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::info;

use crate::traits::Goal;

pub(crate) const LEGACY_KNOWLEDGE_MAINTENANCE_GOAL_DESC: &str =
    "Maintain knowledge base: process embeddings, consolidate memories, decay old facts";
pub(crate) const LEGACY_MEMORY_HEALTH_GOAL_DESC: &str =
    "Maintain memory health: prune old events, clean up retention, remove stale data";
pub(crate) const LEGACY_SYSTEM_SESSION_ID: &str = "system";
const LEGACY_MAINTENANCE_MIGRATION_DONE_KEY: &str =
    "migration_legacy_system_maintenance_goals_retired_v1";

fn is_truthy_setting(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enabled"
    )
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LegacyMaintenanceMigrationStats {
    pub(crate) goals_matched: usize,
    pub(crate) goals_retired: usize,
    pub(crate) tasks_closed: usize,
    pub(crate) notifications_deleted: usize,
}

fn is_legacy_system_maintenance_goal(goal: &Goal) -> bool {
    if goal.session_id != LEGACY_SYSTEM_SESSION_ID {
        return false;
    }

    if let Some(ctx) = goal.context.as_deref() {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(ctx) {
            if let Some(system_goal) = value.get("system_goal").and_then(|v| v.as_str()) {
                return matches!(system_goal, "knowledge_maintenance" | "memory_health");
            }
        }
    }

    goal.description == LEGACY_KNOWLEDGE_MAINTENANCE_GOAL_DESC
        || goal.description == LEGACY_MEMORY_HEALTH_GOAL_DESC
}

fn is_open_goal_task_status(status: &str) -> bool {
    matches!(status, "pending" | "claimed" | "running")
}

pub(crate) async fn maybe_run_legacy_system_maintenance_goal_migration(
    state: Arc<dyn crate::traits::StateStore>,
    pool: SqlitePool,
) {
    let migration_done = match state
        .get_setting(LEGACY_MAINTENANCE_MIGRATION_DONE_KEY)
        .await
    {
        Ok(Some(v)) => is_truthy_setting(&v),
        Ok(None) => false,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "Failed to read legacy maintenance-goal migration marker; running migration"
            );
            false
        }
    };
    if !migration_done {
        match retire_legacy_system_maintenance_goals(state.clone(), pool).await {
            Ok(stats) => {
                if stats.goals_matched > 0
                    || stats.goals_retired > 0
                    || stats.tasks_closed > 0
                    || stats.notifications_deleted > 0
                {
                    info!(
                        matched = stats.goals_matched,
                        retired = stats.goals_retired,
                        tasks_closed = stats.tasks_closed,
                        notifications_deleted = stats.notifications_deleted,
                        "Applied legacy maintenance-goal migration"
                    );
                }
                if let Err(e) = state
                    .set_setting(LEGACY_MAINTENANCE_MIGRATION_DONE_KEY, "1")
                    .await
                {
                    tracing::warn!(
                        error = %e,
                        "Failed to persist legacy maintenance-goal migration marker"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Legacy maintenance-goal migration failed");
            }
        }
    }
}

pub(crate) async fn retire_legacy_system_maintenance_goals(
    state: Arc<dyn crate::traits::StateStore>,
    pool: SqlitePool,
) -> anyhow::Result<LegacyMaintenanceMigrationStats> {
    let mut stats = LegacyMaintenanceMigrationStats::default();
    let scheduled_goals = state.get_scheduled_goals().await?;
    let legacy_goals: Vec<Goal> = scheduled_goals
        .into_iter()
        .filter(is_legacy_system_maintenance_goal)
        .collect();
    stats.goals_matched = legacy_goals.len();

    if legacy_goals.is_empty() {
        return Ok(stats);
    }

    let now = chrono::Utc::now().to_rfc3339();
    let retirement_note = "Retired by startup migration: legacy system maintenance goal removed";

    for goal in legacy_goals {
        if goal.status != "cancelled" && goal.status != "completed" {
            let mut updated_goal = goal.clone();
            updated_goal.status = "cancelled".to_string();
            updated_goal.completed_at = Some(now.clone());
            updated_goal.updated_at = now.clone();
            state.update_goal(&updated_goal).await?;
            stats.goals_retired += 1;
        }

        let tasks = state.get_tasks_for_goal(&goal.id).await?;
        for mut task in tasks {
            if !is_open_goal_task_status(&task.status) {
                continue;
            }
            task.status = "completed".to_string();
            task.completed_at = Some(now.clone());
            task.error = None;
            let has_result = task
                .result
                .as_ref()
                .is_some_and(|result| !result.trim().is_empty());
            if !has_result {
                task.result = Some(retirement_note.to_string());
            }
            state.update_task(&task).await?;
            stats.tasks_closed += 1;
        }

        let deleted = sqlx::query("DELETE FROM notification_queue WHERE goal_id = ?")
            .bind(&goal.id)
            .execute(&pool)
            .await?;
        stats.notifications_deleted += deleted.rows_affected() as usize;
    }

    Ok(stats)
}
