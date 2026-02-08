//! PlanStore - SQLite persistence for task plans.

use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use tracing::info;

use super::{PlanStatus, TaskPlan};

/// Persistent storage for task plans.
pub struct PlanStore {
    pool: SqlitePool,
}

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
        .execute(&self.pool)
        .await?;

        // Index for finding incomplete plans for a session
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_plans_session_status
            ON task_plans(session_id, status)
            "#,
        )
        .execute(&self.pool)
        .await?;

        // Index for cleanup of old completed plans
        sqlx::query(
            r#"
            CREATE INDEX IF NOT EXISTS idx_plans_updated
            ON task_plans(updated_at)
            "#,
        )
        .execute(&self.pool)
        .await?;

        info!("Task plans table migration complete");
        Ok(())
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

    /// Get all plans that were in progress (for recovery after restart).
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

        let steps = serde_json::from_str(&steps_json)?;
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
}
