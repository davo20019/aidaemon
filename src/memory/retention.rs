//! Retention policies for automatic cleanup of old data.
//!
//! Each table has configurable retention with safe guards:
//! - Conversation events: only deletes consolidated conversation events
//! - Facts: only deletes superseded versions
//! - Token usage: aggregates before deleting
//! - Episodes: only deletes unreferenced (recall_count=0)
//! - Behavior patterns: only deletes decayed (confidence <= 0.1)
//! - Goals: only deletes completed/abandoned
//! - Procedures: only deletes zero-success
//! - Error solutions: only deletes net-negative

use chrono::{Duration, Utc};
use sqlx::SqlitePool;
use tracing::warn;

use crate::config::RetentionConfig;

/// Per-table cleanup counts
#[derive(Debug, Default)]
pub struct RetentionStats {
    pub messages_deleted: u64,
    pub facts_deleted: u64,
    pub token_usage_aggregated: u64,
    pub token_usage_deleted: u64,
    pub episodes_deleted: u64,
    pub behavior_patterns_deleted: u64,
    pub goals_deleted: u64,
    pub procedures_deleted: u64,
    pub error_solutions_deleted: u64,
}

impl RetentionStats {
    pub fn total_deleted(&self) -> u64 {
        self.messages_deleted
            + self.facts_deleted
            + self.token_usage_deleted
            + self.episodes_deleted
            + self.behavior_patterns_deleted
            + self.goals_deleted
            + self.procedures_deleted
            + self.error_solutions_deleted
    }
}

pub struct RetentionManager {
    pool: SqlitePool,
    config: RetentionConfig,
}

impl RetentionManager {
    pub fn new(pool: SqlitePool, config: RetentionConfig) -> Self {
        Self { pool, config }
    }

    /// Run all retention cleanups. Each is independent; one failure doesn't block others.
    pub async fn run_all(&self) -> anyhow::Result<RetentionStats> {
        let mut stats = RetentionStats::default();

        match self.cleanup_messages().await {
            Ok(n) => stats.messages_deleted = n,
            Err(e) => warn!("Retention: messages cleanup failed: {}", e),
        }

        match self.cleanup_superseded_facts().await {
            Ok(n) => stats.facts_deleted = n,
            Err(e) => warn!("Retention: facts cleanup failed: {}", e),
        }

        match self.cleanup_token_usage().await {
            Ok((agg, del)) => {
                stats.token_usage_aggregated = agg;
                stats.token_usage_deleted = del;
            }
            Err(e) => warn!("Retention: token_usage cleanup failed: {}", e),
        }

        match self.cleanup_episodes().await {
            Ok(n) => stats.episodes_deleted = n,
            Err(e) => warn!("Retention: episodes cleanup failed: {}", e),
        }

        match self.cleanup_behavior_patterns().await {
            Ok(n) => stats.behavior_patterns_deleted = n,
            Err(e) => warn!("Retention: behavior_patterns cleanup failed: {}", e),
        }

        match self.cleanup_goals().await {
            Ok(n) => stats.goals_deleted = n,
            Err(e) => warn!("Retention: goals cleanup failed: {}", e),
        }

        match self.cleanup_procedures().await {
            Ok(n) => stats.procedures_deleted = n,
            Err(e) => warn!("Retention: procedures cleanup failed: {}", e),
        }

        match self.cleanup_error_solutions().await {
            Ok(n) => stats.error_solutions_deleted = n,
            Err(e) => warn!("Retention: error_solutions cleanup failed: {}", e),
        }

        Ok(stats)
    }

    /// Delete consolidated conversation events older than cutoff.
    /// Safety: never deletes unconsolidated events.
    async fn cleanup_messages(&self) -> anyhow::Result<u64> {
        if self.config.messages_days == 0 {
            return Ok(0);
        }
        let cutoff = (Utc::now() - Duration::days(self.config.messages_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM events WHERE id IN (
                SELECT id FROM events
                WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
                  AND consolidated_at IS NOT NULL
                  AND created_at < ?
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Delete old superseded fact versions.
    /// Safety: only deletes facts that have been superseded (replaced by newer version).
    async fn cleanup_superseded_facts(&self) -> anyhow::Result<u64> {
        if self.config.superseded_facts_days == 0 {
            return Ok(0);
        }
        let cutoff =
            (Utc::now() - Duration::days(self.config.superseded_facts_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM facts WHERE id IN (
                SELECT id FROM facts
                WHERE superseded_at IS NOT NULL AND superseded_at < ?
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Aggregate raw token_usage into daily summaries, then delete raw records.
    /// Safety: aggregation happens before deletion.
    async fn cleanup_token_usage(&self) -> anyhow::Result<(u64, u64)> {
        if self.config.token_usage_aggregate_days == 0 {
            return Ok((0, 0));
        }
        let cutoff = (Utc::now() - Duration::days(self.config.token_usage_aggregate_days as i64))
            .to_rfc3339();

        // Step 1: Aggregate into token_usage_daily
        let agg_result = sqlx::query(
            "INSERT OR REPLACE INTO token_usage_daily (date, model, total_input_tokens, total_output_tokens, request_count)
             SELECT DATE(created_at), model, SUM(input_tokens), SUM(output_tokens), COUNT(*)
             FROM token_usage
             WHERE created_at < ?
             GROUP BY DATE(created_at), model"
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        let aggregated = agg_result.rows_affected();

        // Step 2: Delete raw records that have been aggregated
        let del_result = sqlx::query(
            "DELETE FROM token_usage WHERE id IN (
                SELECT id FROM token_usage
                WHERE created_at < ?
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        let deleted = del_result.rows_affected();

        Ok((aggregated, deleted))
    }

    /// Delete episodes with recall_count=0 older than cutoff.
    /// Safety: preserves episodes that have been recalled (referenced).
    async fn cleanup_episodes(&self) -> anyhow::Result<u64> {
        if self.config.episodes_days == 0 {
            return Ok(0);
        }
        let cutoff = (Utc::now() - Duration::days(self.config.episodes_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM episodes WHERE id IN (
                SELECT id FROM episodes
                WHERE recall_count = 0 AND created_at < ?
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Delete behavior patterns at confidence floor that haven't been seen recently.
    /// Safety: only deletes already-decayed patterns (confidence <= 0.1).
    async fn cleanup_behavior_patterns(&self) -> anyhow::Result<u64> {
        if self.config.behavior_patterns_days == 0 {
            return Ok(0);
        }
        let cutoff =
            (Utc::now() - Duration::days(self.config.behavior_patterns_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM behavior_patterns WHERE id IN (
                SELECT id FROM behavior_patterns
                WHERE confidence <= 0.1 AND (last_seen_at IS NULL OR last_seen_at < ?)
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Delete completed/abandoned goals older than cutoff.
    /// Safety: never deletes active goals.
    async fn cleanup_goals(&self) -> anyhow::Result<u64> {
        if self.config.goals_days == 0 {
            return Ok(0);
        }
        let cutoff = (Utc::now() - Duration::days(self.config.goals_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM goals WHERE id IN (
                SELECT id FROM goals
                WHERE status IN ('completed', 'abandoned') AND updated_at < ?
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Delete zero-success procedures older than cutoff.
    /// Safety: preserves any procedure that has succeeded at least once.
    async fn cleanup_procedures(&self) -> anyhow::Result<u64> {
        if self.config.procedures_days == 0 {
            return Ok(0);
        }
        let cutoff = (Utc::now() - Duration::days(self.config.procedures_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM procedures WHERE id IN (
                SELECT id FROM procedures
                WHERE success_count = 0 AND (last_used_at IS NULL OR last_used_at < ?)
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Delete net-negative error solutions older than cutoff.
    /// Safety: preserves solutions where success_count >= failure_count.
    async fn cleanup_error_solutions(&self) -> anyhow::Result<u64> {
        if self.config.error_solutions_days == 0 {
            return Ok(0);
        }
        let cutoff =
            (Utc::now() - Duration::days(self.config.error_solutions_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM error_solutions WHERE id IN (
                SELECT id FROM error_solutions
                WHERE failure_count > success_count AND (last_used_at IS NULL OR last_used_at < ?)
                LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_test_db() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

        // Create minimal tables for testing
        sqlx::query(
            "CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT,
                superseded_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                recall_count INTEGER DEFAULT 0
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE token_usage_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                model TEXT NOT NULL,
                total_input_tokens INTEGER NOT NULL,
                total_output_tokens INTEGER NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(date, model)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                recall_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE behavior_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                last_seen_at TEXT,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE goals (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                session_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                trigger_pattern TEXT NOT NULL,
                steps TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE error_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_pattern TEXT NOT NULL,
                domain TEXT,
                solution_summary TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        pool
    }

    #[tokio::test]
    async fn test_cleanup_messages_only_consolidated() {
        let pool = setup_test_db().await;
        let old_date = "2020-01-01T00:00:00+00:00";

        // Insert unconsolidated old conversation event (should survive)
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at)
             VALUES ('s1', 'user_message', '{\"content\":\"hello\"}', ?)",
        )
        .bind(old_date)
        .execute(&pool)
        .await
        .unwrap();

        // Insert consolidated old conversation event (should be deleted)
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at, consolidated_at)
             VALUES ('s1', 'assistant_response', '{\"content\":\"world\"}', ?, ?)",
        )
        .bind(old_date)
        .bind(old_date)
        .execute(&pool)
        .await
        .unwrap();

        let mgr = RetentionManager::new(pool.clone(), RetentionConfig::default());
        let deleted = mgr.cleanup_messages().await.unwrap();
        assert_eq!(deleted, 1);

        // Verify unconsolidated event survived
        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM events
             WHERE session_id = 's1'
               AND event_type = 'user_message'
               AND consolidated_at IS NULL",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count.0, 1);
    }

    #[tokio::test]
    async fn test_cleanup_superseded_facts() {
        let pool = setup_test_db().await;
        let old_date = "2020-01-01T00:00:00+00:00";
        let now = Utc::now().to_rfc3339();

        // Superseded old fact (should be deleted)
        sqlx::query("INSERT INTO facts (category, key, value, superseded_at, created_at, updated_at) VALUES ('user', 'name', 'old', ?, ?, ?)")
            .bind(old_date).bind(old_date).bind(old_date).execute(&pool).await.unwrap();

        // Current fact (should survive)
        sqlx::query("INSERT INTO facts (category, key, value, created_at, updated_at) VALUES ('user', 'name', 'new', ?, ?)")
            .bind(&now).bind(&now).execute(&pool).await.unwrap();

        let mgr = RetentionManager::new(pool.clone(), RetentionConfig::default());
        let deleted = mgr.cleanup_superseded_facts().await.unwrap();
        assert_eq!(deleted, 1);

        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM facts")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count.0, 1);
    }

    #[tokio::test]
    async fn test_cleanup_token_usage_aggregation() {
        let pool = setup_test_db().await;
        let old_date = "2020-06-15T10:00:00+00:00";

        // Insert raw records
        sqlx::query("INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, created_at) VALUES ('s1', 'gpt-4', 100, 50, ?)")
            .bind(old_date).execute(&pool).await.unwrap();
        sqlx::query("INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, created_at) VALUES ('s2', 'gpt-4', 200, 100, ?)")
            .bind(old_date).execute(&pool).await.unwrap();

        let mgr = RetentionManager::new(pool.clone(), RetentionConfig::default());
        let (aggregated, deleted) = mgr.cleanup_token_usage().await.unwrap();

        assert!(aggregated > 0);
        assert_eq!(deleted, 2);

        // Verify daily aggregate was created
        let row: (i64, i64, i64) = sqlx::query_as(
            "SELECT total_input_tokens, total_output_tokens, request_count FROM token_usage_daily WHERE model = 'gpt-4'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(row.0, 300); // 100 + 200
        assert_eq!(row.1, 150); // 50 + 100
        assert_eq!(row.2, 2);
    }

    #[tokio::test]
    async fn test_cleanup_episodes_preserves_recalled() {
        let pool = setup_test_db().await;
        let old_date = "2020-01-01T00:00:00+00:00";

        // Recalled old episode (should survive)
        sqlx::query("INSERT INTO episodes (session_id, summary, recall_count, created_at) VALUES ('s1', 'important', 5, ?)")
            .bind(old_date).execute(&pool).await.unwrap();

        // Unreferenced old episode (should be deleted)
        sqlx::query("INSERT INTO episodes (session_id, summary, recall_count, created_at) VALUES ('s2', 'forgotten', 0, ?)")
            .bind(old_date).execute(&pool).await.unwrap();

        let mgr = RetentionManager::new(pool.clone(), RetentionConfig::default());
        let deleted = mgr.cleanup_episodes().await.unwrap();
        assert_eq!(deleted, 1);

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM episodes WHERE summary = 'important'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 1);
    }

    #[tokio::test]
    async fn test_disabled_cleanup_returns_zero() {
        let pool = setup_test_db().await;
        let config = RetentionConfig {
            messages_days: 0,
            superseded_facts_days: 0,
            token_usage_aggregate_days: 0,
            episodes_days: 0,
            behavior_patterns_days: 0,
            goals_days: 0,
            procedures_days: 0,
            error_solutions_days: 0,
        };
        let mgr = RetentionManager::new(pool, config);
        let stats = mgr.run_all().await.unwrap();
        assert_eq!(stats.total_deleted(), 0);
    }

    #[tokio::test]
    async fn test_cleanup_goals_preserves_active() {
        let pool = setup_test_db().await;
        let old_date = "2020-01-01T00:00:00+00:00";

        // Active goal (should survive)
        sqlx::query("INSERT INTO goals (id, description, status, session_id, updated_at) VALUES ('g1', 'learn rust', 'active', 'test-session', ?)")
            .bind(old_date).execute(&pool).await.unwrap();

        // Completed old goal (should be deleted)
        sqlx::query("INSERT INTO goals (id, description, status, session_id, updated_at) VALUES ('g2', 'done task', 'completed', 'test-session', ?)")
            .bind(old_date).execute(&pool).await.unwrap();

        let mgr = RetentionManager::new(pool.clone(), RetentionConfig::default());
        let deleted = mgr.cleanup_goals().await.unwrap();
        assert_eq!(deleted, 1);

        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM goals WHERE status = 'active'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count.0, 1);
    }

    #[tokio::test]
    async fn test_cleanup_procedures_preserves_successful() {
        let pool = setup_test_db().await;
        let old_date = "2020-01-01T00:00:00+00:00";

        // Successful procedure (should survive even when old)
        sqlx::query("INSERT INTO procedures (name, trigger_pattern, steps, success_count, last_used_at, created_at) VALUES ('good_proc', 'do thing', '[]', 5, ?, ?)")
            .bind(old_date).bind(old_date).execute(&pool).await.unwrap();

        // Zero-success old procedure (should be deleted)
        sqlx::query("INSERT INTO procedures (name, trigger_pattern, steps, success_count, last_used_at, created_at) VALUES ('bad_proc', 'fail thing', '[]', 0, ?, ?)")
            .bind(old_date).bind(old_date).execute(&pool).await.unwrap();

        let mgr = RetentionManager::new(pool.clone(), RetentionConfig::default());
        let deleted = mgr.cleanup_procedures().await.unwrap();
        assert_eq!(deleted, 1);

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM procedures WHERE name = 'good_proc'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 1);
    }
}
