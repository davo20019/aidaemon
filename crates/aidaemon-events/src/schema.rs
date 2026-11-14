//! The `events` table contract.

use sqlx::SqlitePool;
use tracing::info;

/// Centralized database migrations for all SQLite-backed stores.
///
/// Each migration is designed to be safe to call multiple times (idempotent) by
/// using `IF NOT EXISTS` where possible and best-effort `ALTER TABLE`s where not.
pub async fn migrate_events(pool: &SqlitePool) -> anyhow::Result<()> {
    // Create events table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            consolidated_at TEXT,
            task_id TEXT,
            tool_name TEXT,
            turn_id TEXT
        )
        "#,
    )
    .execute(pool)
    .await?;

    // Pillar B: turn-anchored history. `turn_id` is a globally-unique UUID
    // (the opening user-message id of a conversation turn). Existing databases
    // get the column via a best-effort idempotent ALTER (fresh DBs already have
    // it from CREATE TABLE above, where this ALTER harmlessly reports a
    // duplicate column and is discarded). On a large existing events table the
    // one-time index build below is a startup stall on first run after upgrade.
    let _ = sqlx::query("ALTER TABLE events ADD COLUMN turn_id TEXT")
        .execute(pool)
        .await;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_events_turn
         ON events(session_id, turn_id, id)",
    )
    .execute(pool)
    .await?;

    // Non-destructive `/clear` boundary. Created here (alongside events) so the
    // turn-anchored queries that reference it always find the table, including
    // in EventStore-only pools; the state-store migration also creates it
    // idempotently.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS session_context_boundaries (
            session_id TEXT PRIMARY KEY,
            cleared_after_id INTEGER NOT NULL,
            cleared_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // Create indexes for efficient queries
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_events_session_time
         ON events(session_id, created_at DESC)",
    )
    .execute(pool)
    .await?;

    // Canonical recent-event queries order by append identity, not wall time.
    // Keep LIMIT queries bounded even for sessions with long event histories.
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_events_session_id
         ON events(session_id, id)",
    )
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        .execute(pool)
        .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_events_task
         ON events(task_id) WHERE task_id IS NOT NULL",
    )
    .execute(pool)
    .await?;

    // Authoritative task lifecycle projection. Immutable events remain the
    // audit log, while this versioned row is the single transition authority
    // for start/terminal state and prevents duplicate terminal events under
    // watchdog, cancellation, recovery, and normal-close races.
    let lifecycle_existed = sqlx::query_scalar::<_, i64>(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'task_lifecycle'",
    )
    .fetch_one(pool)
    .await?
        > 0;
    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS task_lifecycle (
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('running', 'terminal')),
            status TEXT,
            outcome TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            start_event_id INTEGER,
            terminal_event_id INTEGER,
            started_at TEXT,
            ended_at TEXT,
            PRIMARY KEY(session_id, task_id),
            FOREIGN KEY(start_event_id) REFERENCES events(id) ON DELETE SET NULL,
            FOREIGN KEY(terminal_event_id) REFERENCES events(id) ON DELETE SET NULL
        )
        "#,
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_lifecycle_session_state
         ON task_lifecycle(session_id, state)",
    )
    .execute(&mut *tx)
    .await?;
    // Backfill the first start and first terminal event for existing logs,
    // once, when the table is created. New appends are projected atomically
    // by EventStore::append, so rerunning it on every start rescanned every
    // task event and inserted nothing. The table and its backfill commit
    // together, so an existing table always has its backfill.
    if !lifecycle_existed {
        sqlx::query(
            r#"
            INSERT OR IGNORE INTO task_lifecycle (
                session_id, task_id, state, status, outcome, version,
                start_event_id, terminal_event_id, started_at, ended_at
            )
            SELECT
                s.session_id,
                s.task_id,
                CASE WHEN terminal.id IS NULL THEN 'running' ELSE 'terminal' END,
                json_extract(terminal.data, '$.status'),
                json_extract(terminal.data, '$.outcome'),
                CASE WHEN terminal.id IS NULL THEN 1 ELSE 2 END,
                s.id,
                terminal.id,
                s.created_at,
                terminal.created_at
            FROM events AS s
            LEFT JOIN events AS terminal ON terminal.id = (
                SELECT MIN(e.id)
                FROM events AS e
                WHERE e.session_id = s.session_id
                  AND e.task_id = s.task_id
                  AND e.event_type = 'task_end'
            )
            WHERE s.id = (
                SELECT MIN(first_start.id)
                FROM events AS first_start
                WHERE first_start.session_id = s.session_id
                  AND first_start.task_id = s.task_id
                  AND first_start.event_type = 'task_start'
            )
              AND s.event_type = 'task_start'
              AND s.task_id IS NOT NULL
            "#,
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            r#"
            INSERT OR IGNORE INTO task_lifecycle (
                session_id, task_id, state, status, outcome, version,
                terminal_event_id, ended_at
            )
            SELECT e.session_id, e.task_id, 'terminal',
                   json_extract(e.data, '$.status'),
                   json_extract(e.data, '$.outcome'),
                   1, e.id, e.created_at
            FROM events AS e
            WHERE e.event_type = 'task_end'
              AND e.task_id IS NOT NULL
              AND e.id = (
                  SELECT MIN(first_end.id)
                  FROM events AS first_end
                  WHERE first_end.session_id = e.session_id
                    AND first_end.task_id = e.task_id
                    AND first_end.event_type = 'task_end'
              )
            "#,
        )
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_events_consolidation
         ON events(consolidated_at) WHERE consolidated_at IS NULL",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_events_prune
         ON events(created_at) WHERE consolidated_at IS NOT NULL",
    )
    .execute(pool)
    .await?;

    aidaemon_tools::checkpoints::migrate(pool).await?;

    // Tool-result stats: efficient per-tool lookups in time windows.
    // Partial index keeps it small (most events have tool_name = NULL and/or aren't tool_results).
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_events_tool_result_name_time
         ON events(tool_name, created_at DESC)
         WHERE event_type = 'tool_result' AND tool_name IS NOT NULL",
    )
    .execute(pool)
    .await?;

    info!("Events table migration complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn lifecycle_rows(pool: &SqlitePool) -> Vec<(String, String, Option<String>)> {
        sqlx::query_as("SELECT task_id, state, status FROM task_lifecycle ORDER BY task_id")
            .fetch_all(pool)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn task_lifecycle_is_backfilled_once_when_the_table_is_created() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        migrate_events(&pool).await.unwrap();
        // A log written before the projection existed.
        sqlx::query("DROP TABLE task_lifecycle")
            .execute(&pool)
            .await
            .unwrap();
        for (task, kind, data) in [
            ("task-a", "task_start", "{}"),
            ("task-a", "task_end", r#"{"status":"completed"}"#),
            ("task-b", "task_start", "{}"),
        ] {
            sqlx::query(
                "INSERT INTO events(session_id, event_type, data, created_at, task_id)
                 VALUES ('telegram:synthetic-user-1', ?, ?, '2026-01-01T00:00:00Z', ?)",
            )
            .bind(kind)
            .bind(data)
            .bind(task)
            .execute(&pool)
            .await
            .unwrap();
        }

        migrate_events(&pool).await.unwrap();
        let backfilled = vec![
            (
                "task-a".to_string(),
                "terminal".to_string(),
                Some("completed".to_string()),
            ),
            ("task-b".to_string(), "running".to_string(), None),
        ];
        assert_eq!(lifecycle_rows(&pool).await, backfilled);

        // Later starts leave the projection to appends and do not rescan.
        sqlx::query("DELETE FROM task_lifecycle WHERE task_id = 'task-b'")
            .execute(&pool)
            .await
            .unwrap();
        migrate_events(&pool).await.unwrap();
        assert_eq!(lifecycle_rows(&pool).await, backfilled[..1].to_vec());
    }
}
