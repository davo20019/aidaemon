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
    pub diagnostic_events_deleted: u64,
    pub facts_deleted: u64,
    pub token_usage_aggregated: u64,
    pub token_usage_deleted: u64,
    pub episodes_deleted: u64,
    pub behavior_patterns_deleted: u64,
    pub goals_deleted: u64,
    pub procedures_deleted: u64,
    pub error_solutions_deleted: u64,
    pub self_correction_attempts_deleted: u64,
    pub derived_memory_deleted: u64,
}

impl RetentionStats {
    pub fn total_deleted(&self) -> u64 {
        self.messages_deleted
            + self.diagnostic_events_deleted
            + self.facts_deleted
            + self.token_usage_deleted
            + self.episodes_deleted
            + self.behavior_patterns_deleted
            + self.goals_deleted
            + self.procedures_deleted
            + self.error_solutions_deleted
            + self.self_correction_attempts_deleted
            + self.derived_memory_deleted
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
        match self.cleanup_diagnostic_events().await {
            Ok(n) => stats.diagnostic_events_deleted = n,
            Err(e) => warn!("Retention: diagnostic events cleanup failed: {}", e),
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

        match self.cleanup_self_correction_attempts().await {
            Ok(n) => stats.self_correction_attempts_deleted = n,
            Err(e) => warn!("Retention: self_correction_attempts cleanup failed: {}", e),
        }

        match self.cleanup_derived_memory().await {
            Ok(n) => stats.derived_memory_deleted = n,
            Err(e) => warn!("Retention: derived memory cleanup failed: {}", e),
        }

        Ok(stats)
    }

    /// Delete the canonical history-bearing task envelope and exact messages.
    /// `messages_days = 0` is the sole permanent-history switch.
    async fn cleanup_messages(&self) -> anyhow::Result<u64> {
        if self.config.messages_days == 0 {
            return Ok(0);
        }
        let cutoff = (Utc::now() - Duration::days(self.config.messages_days as i64)).to_rfc3339();
        let mut deleted = 0_u64;
        for _ in 0..100 {
            let result = sqlx::query(
                "DELETE FROM events WHERE id IN (
                SELECT id FROM events
                WHERE event_type IN ('task_start', 'user_message', 'assistant_response', 'task_end')
                  AND consolidated_at IS NOT NULL
                  AND created_at < ?
                LIMIT 500
            )",
            )
            .bind(&cutoff)
            .execute(&self.pool)
            .await?;
            let count = result.rows_affected();
            deleted += count;
            if count < 500 {
                break;
            }
            tokio::task::yield_now().await;
        }
        // Derived rows are harmless if temporarily orphaned because all reads
        // join back to events, but clean them promptly to reclaim FTS space.
        for _ in 0..100 {
            match crate::state::sqlite::history_search::remove_orphans(&self.pool, 500).await {
                Ok(500) => tokio::task::yield_now().await,
                Ok(_) => break,
                Err(error) => {
                    warn!(%error, "Retention: exact-history orphan cleanup deferred");
                    break;
                }
            }
        }
        Ok(deleted)
    }

    /// Diagnostics and tool lifecycle data have a short, independent window so
    /// exact conversation retention is not dictated by telemetry volume.
    async fn cleanup_diagnostic_events(&self) -> anyhow::Result<u64> {
        if self.config.diagnostic_events_days == 0 {
            return Ok(0);
        }
        let cutoff =
            (Utc::now() - Duration::days(self.config.diagnostic_events_days as i64)).to_rfc3339();
        let mut deleted = 0_u64;
        for _ in 0..100 {
            let result = sqlx::query(
                "DELETE FROM events WHERE id IN (
                    SELECT id FROM events
                    WHERE event_type NOT IN
                        ('task_start', 'user_message', 'assistant_response', 'task_end')
                      AND consolidated_at IS NOT NULL
                      AND created_at < ?
                    LIMIT 500
                )",
            )
            .bind(&cutoff)
            .execute(&self.pool)
            .await?;
            let count = result.rows_affected();
            deleted += count;
            if count < 500 {
                break;
            }
            tokio::task::yield_now().await;
        }
        Ok(deleted)
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

    /// Delete self-correction attempt rows older than the configured cutoff.
    async fn cleanup_self_correction_attempts(&self) -> anyhow::Result<u64> {
        if self.config.self_correction_attempts_days == 0 {
            return Ok(0);
        }
        let cutoff = (Utc::now()
            - Duration::days(self.config.self_correction_attempts_days as i64))
        .to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM self_correction_attempts WHERE id IN (
                SELECT id FROM self_correction_attempts WHERE created_at < ? LIMIT 500
            )",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Remove projections whose authoritative source no longer exists. Derived
    /// rows are disposable and must not outlive an explicit source deletion.
    async fn cleanup_derived_memory(&self) -> anyhow::Result<u64> {
        let has_tables: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'memory_claims'",
        )
        .fetch_one(&self.pool)
        .await?;
        if has_tables == 0 {
            return Ok(0);
        }

        let mut deleted = 0;
        deleted += sqlx::query(
            "DELETE FROM memory_edges
             WHERE source_claim_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM memory_claims c WHERE c.id = source_claim_id)",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        deleted += sqlx::query(
            "DELETE FROM memory_edges
             WHERE source_claim_id IN (
                 SELECT c.id FROM memory_claims c
                 WHERE c.source_fact_id IS NOT NULL
                   AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.id = c.source_fact_id)
             )",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        deleted += sqlx::query(
            "DELETE FROM memory_embeddings
             WHERE (owner_type = 'claim' AND NOT EXISTS (
                       SELECT 1 FROM memory_claims c WHERE CAST(c.id AS TEXT) = owner_id))
                OR (owner_type = 'span' AND NOT EXISTS (
                       SELECT 1 FROM memory_spans s WHERE CAST(s.id AS TEXT) = owner_id))
                OR (owner_type = 'procedure' AND NOT EXISTS (
                       SELECT 1 FROM procedures p WHERE CAST(p.id AS TEXT) = owner_id))
                OR (owner_type = 'error_solution' AND NOT EXISTS (
                       SELECT 1 FROM error_solutions e WHERE CAST(e.id AS TEXT) = owner_id))
                OR (stale_at IS NOT NULL AND stale_at < datetime('now', '-30 days'))",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        deleted += sqlx::query(
            "DELETE FROM memory_claims
             WHERE source_fact_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.id = source_fact_id)",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        deleted += sqlx::query(
            "DELETE FROM memory_spans
             WHERE ((source_event_id IS NOT NULL AND NOT EXISTS (
                       SELECT 1 FROM events e WHERE e.id = source_event_id))
                OR (source_episode_id IS NOT NULL AND NOT EXISTS (
                       SELECT 1 FROM episodes ep WHERE ep.id = source_episode_id)))
               AND NOT EXISTS (
                   SELECT 1 FROM memory_claims c WHERE c.source_span_id = memory_spans.id
               )",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        deleted += sqlx::query(
            "DELETE FROM memory_embeddings
             WHERE (owner_type = 'claim' AND NOT EXISTS (
                       SELECT 1 FROM memory_claims c WHERE CAST(c.id AS TEXT) = owner_id))
                OR (owner_type = 'span' AND NOT EXISTS (
                       SELECT 1 FROM memory_spans s WHERE CAST(s.id AS TEXT) = owner_id))",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        deleted += sqlx::query(
            "DELETE FROM memory_entities
             WHERE canonical_name != 'owner'
               AND NOT EXISTS (
                   SELECT 1 FROM memory_edges edge
                   WHERE edge.source_entity_id = memory_entities.id
                      OR edge.target_entity_id = memory_entities.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM memory_aliases alias
                   WHERE alias.entity_id = memory_entities.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM memory_entity_facts fact
                   WHERE fact.subject_entity_id = memory_entities.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM memory_relationships relationship
                   WHERE relationship.source_entity_id = memory_entities.id
                      OR relationship.target_entity_id = memory_entities.id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM memory_write_audit audit
                   WHERE audit.entity_id = memory_entities.id
               )",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        Ok(deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::sync::Arc;

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

        sqlx::query(
            "CREATE TABLE self_correction_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL,
                subject_kind TEXT NOT NULL,
                approach_signature TEXT NOT NULL,
                attempt_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                blocked_reason TEXT,
                evidence_ref TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
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
    async fn messages_days_zero_is_permanent_across_retention_paths() {
        let pool = setup_test_db().await;
        let old_date = "2020-01-01T00:00:00+00:00";
        for event_type in [
            "task_start",
            "user_message",
            "assistant_response",
            "task_end",
        ] {
            sqlx::query(
                "INSERT INTO events(session_id,event_type,data,created_at,consolidated_at)
                 VALUES('permanent', ?, '{}', ?, ?)",
            )
            .bind(event_type)
            .bind(old_date)
            .bind(old_date)
            .execute(&pool)
            .await
            .unwrap();
        }
        sqlx::query(
            "INSERT INTO events(session_id,event_type,data,created_at,consolidated_at)
             VALUES('permanent','tool_result','{}',?,?)",
        )
        .bind(old_date)
        .bind(old_date)
        .execute(&pool)
        .await
        .unwrap();

        let manager = RetentionManager::new(
            pool.clone(),
            RetentionConfig {
                messages_days: 0,
                diagnostic_events_days: 7,
                ..RetentionConfig::default()
            },
        );
        assert_eq!(manager.cleanup_messages().await.unwrap(), 0);
        assert_eq!(manager.cleanup_diagnostic_events().await.unwrap(), 1);
        let history_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM events
             WHERE event_type IN ('task_start','user_message','assistant_response','task_end')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(history_count, 4);
    }

    #[tokio::test]
    async fn message_cleanup_drains_more_than_one_batch() {
        let pool = setup_test_db().await;
        let old_date = "2020-01-01T00:00:00+00:00";
        for _ in 0..1_205 {
            sqlx::query(
                "INSERT INTO events(session_id,event_type,data,created_at,consolidated_at)
                 VALUES('bulk','assistant_response','{}',?,?)",
            )
            .bind(old_date)
            .bind(old_date)
            .execute(&pool)
            .await
            .unwrap();
        }
        let manager = RetentionManager::new(pool.clone(), RetentionConfig::default());
        assert_eq!(manager.cleanup_messages().await.unwrap(), 1_205);
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM events")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
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
            diagnostic_events_days: 0,
            superseded_facts_days: 0,
            token_usage_aggregate_days: 0,
            episodes_days: 0,
            behavior_patterns_days: 0,
            goals_days: 0,
            procedures_days: 0,
            error_solutions_days: 0,
            self_correction_attempts_days: 0,
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

    #[tokio::test]
    async fn cleanup_self_correction_attempts_deletes_old_keeps_recent() {
        let pool = setup_test_db().await;
        let config = RetentionConfig {
            self_correction_attempts_days: 30,
            ..RetentionConfig::default()
        };
        let manager = RetentionManager::new(pool.clone(), config);

        // Insert one old row (40 days) and one recent row (1 day).
        sqlx::query(
            "INSERT INTO self_correction_attempts (subject_id, subject_kind, approach_signature, attempt_index, status, created_at) \
             VALUES ('old', 'task', 'sig-old', 1, 'failed', datetime('now','-40 days'))",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO self_correction_attempts (subject_id, subject_kind, approach_signature, attempt_index, status, created_at) \
             VALUES ('new', 'task', 'sig-new', 1, 'failed', datetime('now','-1 days'))",
        )
        .execute(&pool)
        .await
        .unwrap();

        let deleted = manager.cleanup_self_correction_attempts().await.unwrap();
        assert_eq!(deleted, 1, "only the 40-day-old row should be deleted");

        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM self_correction_attempts")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(remaining, 1);
        let survivor: String =
            sqlx::query_scalar("SELECT subject_id FROM self_correction_attempts")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(survivor, "new");
    }

    #[tokio::test]
    async fn derived_memory_does_not_outlive_deleted_source_fact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("retention-memory.db");
        let embeddings = Arc::new(EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(path.to_str().unwrap(), 20, None, embeddings)
            .await
            .unwrap();
        let pool = store.pool();
        let fact_id = sqlx::query(
            "INSERT INTO facts
             (category, key, value, source, created_at, updated_at, privacy, recall_count)
             VALUES ('technical', 'runtime', 'Rust', 'test', datetime('now'), datetime('now'), 'global', 0)",
        )
        .execute(&pool)
        .await
        .unwrap()
        .last_insert_rowid();
        store.project_fact_memory(fact_id).await.unwrap();
        sqlx::query("DELETE FROM facts WHERE id = ?")
            .bind(fact_id)
            .execute(&pool)
            .await
            .unwrap();

        let manager = RetentionManager::new(pool.clone(), RetentionConfig::default());
        assert!(manager.cleanup_derived_memory().await.unwrap() > 0);
        let claims: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_claims")
            .fetch_one(&pool)
            .await
            .unwrap();
        let edges: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_edges")
            .fetch_one(&pool)
            .await
            .unwrap();
        let embeddings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_embeddings")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!((claims, edges, embeddings), (0, 0, 0));
    }

    #[tokio::test]
    async fn derived_cleanup_preserves_structured_entities_with_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("retention-structured-memory.db");
        let embeddings = Arc::new(EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(path.to_str().unwrap(), 20, None, embeddings)
            .await
            .unwrap();
        let pool = store.pool();

        let entity_id = sqlx::query(
            "INSERT INTO memory_entities
             (entity_type, canonical_name, display_name, aliases_json, privacy, status, is_owner)
             VALUES ('person', 'isabella', 'Isabella', '[]', 'private', 'active', 0)",
        )
        .execute(&pool)
        .await
        .unwrap()
        .last_insert_rowid();
        sqlx::query(
            "INSERT INTO memory_aliases
             (entity_id, alias_type, value, normalized_value, source, privacy,
              asserted_at, created_at, updated_at)
             VALUES (?, 'nickname', 'Bella', 'bella', 'test', 'private',
                     datetime('now'), datetime('now'), datetime('now'))",
        )
        .bind(entity_id)
        .execute(&pool)
        .await
        .unwrap();

        let manager = RetentionManager::new(pool.clone(), RetentionConfig::default());
        manager.cleanup_derived_memory().await.unwrap();

        let remaining: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM memory_entities WHERE id = ?")
                .bind(entity_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(remaining, 1);
    }
}
