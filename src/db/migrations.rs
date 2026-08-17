use sqlx::SqlitePool;
use tracing::info;

/// Centralized database migrations for all SQLite-backed stores.
///
/// Each migration is designed to be safe to call multiple times (idempotent) by
/// using `IF NOT EXISTS` where possible and best-effort `ALTER TABLE`s where not.
pub(crate) async fn migrate_events(pool: &SqlitePool) -> anyhow::Result<()> {
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
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_lifecycle_session_state
         ON task_lifecycle(session_id, state)",
    )
    .execute(pool)
    .await?;
    // Backfill the first start and first terminal event for existing logs.
    // New appends are projected atomically by EventStore::append.
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
    .execute(pool)
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
    .execute(pool)
    .await?;

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

pub(crate) async fn migrate_task_plans(pool: &SqlitePool) -> anyhow::Result<()> {
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

    info!("Task plans table migration complete");
    Ok(())
}

pub(crate) async fn migrate_health_probes(pool: &SqlitePool) -> anyhow::Result<()> {
    // Probe definitions table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS health_probes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            probe_type TEXT NOT NULL,
            target TEXT NOT NULL,
            schedule TEXT NOT NULL,
            source TEXT DEFAULT 'tool',
            config TEXT DEFAULT '{}',
            consecutive_failures_alert INTEGER DEFAULT 3,
            latency_threshold_ms INTEGER,
            alert_session_ids TEXT,
            is_paused INTEGER DEFAULT 0,
            last_run_at TEXT,
            next_run_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // Time-series results table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS probe_results (
            id INTEGER PRIMARY KEY,
            probe_id TEXT NOT NULL,
            status TEXT NOT NULL,
            latency_ms INTEGER,
            error_message TEXT,
            response_body TEXT,
            checked_at TEXT NOT NULL,
            FOREIGN KEY (probe_id) REFERENCES health_probes(id) ON DELETE CASCADE
        )",
    )
    .execute(pool)
    .await?;

    // Alert history table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS probe_alerts (
            id INTEGER PRIMARY KEY,
            probe_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            sent_at TEXT NOT NULL,
            first_failure_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // Indexes for efficient queries
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_probe_results_probe_time
         ON probe_results(probe_id, checked_at DESC)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_health_probes_next_run
         ON health_probes(next_run_at) WHERE is_paused = 0",
    )
    .execute(pool)
    .await?;

    Ok(())
}
