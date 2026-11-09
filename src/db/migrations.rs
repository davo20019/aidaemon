use sqlx::SqlitePool;
use tracing::info;

pub(crate) use aidaemon_events::schema::migrate_events;

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
