//! The `token_alert_state` table. Its spike scan was never run by a host and
//! is gone; the table stays so existing databases and the schema match.

use sqlx::SqlitePool;

/// Create the `token_alert_state` table if missing.
pub async fn migrate(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS token_alert_state (
            scope_type TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            last_alert_at TEXT NOT NULL,
            last_metric_tokens INTEGER NOT NULL DEFAULT 0,
            last_metric_calls INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (scope_type, scope_id)
        )",
    )
    .execute(pool)
    .await?;
    Ok(())
}
