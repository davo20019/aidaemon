//! The checkpoint tables: lifecycle and audit metadata only (content lives
//! in the external Git object store). The host runs [`migrate`] on its pool.

use sqlx::SqlitePool;

/// Create the checkpoint tables and indexes if missing.
pub async fn migrate(pool: &SqlitePool) -> anyhow::Result<()> {
    // Local workspace source checkpoints. Content lives in an external bare
    // Git object store; this table contains only lifecycle and audit metadata.
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS filesystem_checkpoints (
            id TEXT PRIMARY KEY,
            scope_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            task_id TEXT,
            turn_id TEXT,
            backend_id TEXT NOT NULL,
            root_path TEXT NOT NULL,
            store_path TEXT NOT NULL,
            pre_tree TEXT NOT NULL,
            post_tree TEXT,
            state TEXT NOT NULL,
            origin_tool TEXT NOT NULL,
            included_paths INTEGER NOT NULL DEFAULT 0,
            included_bytes INTEGER NOT NULL DEFAULT 0,
            excluded_paths INTEGER NOT NULL DEFAULT 0,
            unsafe_reason TEXT,
            rollback_of TEXT,
            created_at TEXT NOT NULL,
            finalized_at TEXT,
            expires_at TEXT NOT NULL,
            UNIQUE(scope_id, backend_id, root_path)
        )
        "#,
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_filesystem_checkpoints_root_created
         ON filesystem_checkpoints(root_path, created_at DESC)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_filesystem_checkpoints_task_state
         ON filesystem_checkpoints(task_id, state)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS checkpoint_restore_runs (
            id TEXT PRIMARY KEY,
            checkpoint_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            state TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            next_index INTEGER NOT NULL DEFAULT 0,
            safety_checkpoint_id TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
        )
        "#,
    )
    .execute(pool)
    .await?;
    Ok(())
}
