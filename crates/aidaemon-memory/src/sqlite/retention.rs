//! Age-based cleanup of the memory tables. Each step has a safe guard: only
//! superseded facts, unrecalled episodes, decayed patterns, zero-success
//! procedures and net-negative error solutions go, in small batches.

use std::sync::Arc;

use chrono::{Duration, Utc};
use sqlx::SqlitePool;

use crate::history::MessageLog;
use tracing::warn;

/// How long each kind of memory is kept, in days. Zero keeps it forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRetentionPolicy {
    pub superseded_facts_days: u32,
    pub episodes_days: u32,
    pub behavior_patterns_days: u32,
    pub procedures_days: u32,
    pub error_solutions_days: u32,
}

/// Rows removed by one pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRetentionStats {
    pub facts_deleted: u64,
    pub episodes_deleted: u64,
    pub behavior_patterns_deleted: u64,
    pub procedures_deleted: u64,
    pub error_solutions_deleted: u64,
    pub derived_memory_deleted: u64,
}

/// One pass of retention over a database holding the memory tables.
pub struct MemoryRetention {
    pool: SqlitePool,
    config: MemoryRetentionPolicy,
    messages: Option<Arc<dyn MessageLog>>,
}

impl MemoryRetention {
    pub fn new(pool: SqlitePool, config: MemoryRetentionPolicy) -> Self {
        Self {
            pool,
            config,
            messages: None,
        }
    }

    /// Drops indexed messages the host no longer keeps. Without a log,
    /// indexed messages stay until their claims and episodes go.
    pub fn with_message_log(mut self, messages: Arc<dyn MessageLog>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Delete message spans, not backing any claim, whose message the host
    /// no longer keeps.
    async fn cleanup_spans_of_dropped_messages(&self) -> anyhow::Result<u64> {
        let Some(messages) = &self.messages else {
            return Ok(0);
        };
        let spans: Vec<(i64, i64)> = sqlx::query_as(
            "SELECT id, source_event_id FROM memory_spans
             WHERE source_event_id IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM memory_claims c WHERE c.source_span_id = memory_spans.id
               )",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut dropped = Vec::new();
        for chunk in spans.chunks(500) {
            let sources: Vec<i64> = chunk.iter().map(|(_, source)| *source).collect();
            let kept = messages.retained(&sources).await?;
            dropped.extend(
                chunk
                    .iter()
                    .filter(|(_, source)| !kept.contains(source))
                    .map(|(span_id, _)| *span_id),
            );
        }
        if dropped.is_empty() {
            return Ok(0);
        }
        // One transaction, a set-based delete per batch of ids. A claim
        // filed since the scan keeps its span.
        let mut tx = self.pool.begin().await?;
        let mut deleted = 0;
        for ids in dropped.chunks(500) {
            let placeholders = vec!["?"; ids.len()].join(", ");
            let sql = format!(
                "DELETE FROM memory_spans WHERE id IN ({placeholders})
                   AND NOT EXISTS (
                       SELECT 1 FROM memory_claims c WHERE c.source_span_id = memory_spans.id
                   )"
            );
            let mut query = sqlx::query(&sql);
            for id in ids {
                query = query.bind(id);
            }
            deleted += query.execute(&mut *tx).await?.rows_affected();
        }
        tx.commit().await?;
        Ok(deleted)
    }

    /// Run every step. One failing does not stop the others.
    pub async fn run_all(&self) -> MemoryRetentionStats {
        fn counted(step: &str, result: anyhow::Result<u64>) -> u64 {
            result.unwrap_or_else(|error| {
                warn!("Retention: {step} cleanup failed: {error}");
                0
            })
        }
        MemoryRetentionStats {
            facts_deleted: counted("facts", self.cleanup_superseded_facts().await),
            episodes_deleted: counted("episodes", self.cleanup_episodes().await),
            behavior_patterns_deleted: counted(
                "behavior_patterns",
                self.cleanup_behavior_patterns().await,
            ),
            procedures_deleted: counted("procedures", self.cleanup_procedures().await),
            error_solutions_deleted: counted(
                "error_solutions",
                self.cleanup_error_solutions().await,
            ),
            derived_memory_deleted: counted("derived memory", self.cleanup_derived_memory().await),
        }
    }

    /// Delete old superseded fact versions.
    /// Safety: only deletes facts that have been superseded (replaced by newer version).
    pub async fn cleanup_superseded_facts(&self) -> anyhow::Result<u64> {
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

    /// Delete episodes with recall_count=0 older than cutoff.
    /// Safety: preserves episodes that have been recalled (referenced).
    pub async fn cleanup_episodes(&self) -> anyhow::Result<u64> {
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
    pub async fn cleanup_behavior_patterns(&self) -> anyhow::Result<u64> {
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

    /// Delete zero-success procedures older than cutoff.
    /// Safety: preserves any procedure that has succeeded at least once.
    pub async fn cleanup_procedures(&self) -> anyhow::Result<u64> {
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
    pub async fn cleanup_error_solutions(&self) -> anyhow::Result<u64> {
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

    /// Remove projections whose authoritative source no longer exists. Derived
    /// rows are disposable and must not outlive an explicit source deletion.
    pub async fn cleanup_derived_memory(&self) -> anyhow::Result<u64> {
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
                       SELECT 1 FROM memory_claims c
                       WHERE c.id = CAST(owner_id AS INTEGER)
                         AND CAST(c.id AS TEXT) = owner_id))
                OR (owner_type = 'span' AND NOT EXISTS (
                       SELECT 1 FROM memory_spans s
                       WHERE s.id = CAST(owner_id AS INTEGER)
                         AND CAST(s.id AS TEXT) = owner_id))
                OR (owner_type = 'procedure' AND NOT EXISTS (
                       SELECT 1 FROM procedures p
                       WHERE p.id = CAST(owner_id AS INTEGER)
                         AND CAST(p.id AS TEXT) = owner_id))
                OR (owner_type = 'error_solution' AND NOT EXISTS (
                       SELECT 1 FROM error_solutions e
                       WHERE e.id = CAST(owner_id AS INTEGER)
                         AND CAST(e.id AS TEXT) = owner_id))
                OR (owner_type = 'fact_key' AND NOT EXISTS (
                       SELECT 1 FROM facts f
                       WHERE f.id = CAST(owner_id AS INTEGER)
                         AND CAST(f.id AS TEXT) = owner_id
                         AND f.superseded_at IS NULL))
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
             WHERE source_episode_id IS NOT NULL AND NOT EXISTS (
                       SELECT 1 FROM episodes ep WHERE ep.id = source_episode_id)
               AND NOT EXISTS (
                   SELECT 1 FROM memory_claims c WHERE c.source_span_id = memory_spans.id
               )",
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        deleted += self.cleanup_spans_of_dropped_messages().await?;
        deleted += sqlx::query(
            "DELETE FROM memory_embeddings
             WHERE (owner_type = 'claim' AND NOT EXISTS (
                       SELECT 1 FROM memory_claims c
                       WHERE c.id = CAST(owner_id AS INTEGER)
                         AND CAST(c.id AS TEXT) = owner_id))
                OR (owner_type = 'span' AND NOT EXISTS (
                       SELECT 1 FROM memory_spans s
                       WHERE s.id = CAST(owner_id AS INTEGER)
                         AND CAST(s.id AS TEXT) = owner_id))",
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
    use std::collections::HashSet;

    /// A host log that keeps only the listed message ids.
    struct Keeps(HashSet<i64>);

    #[async_trait::async_trait]
    impl MessageLog for Keeps {
        async fn message_bounds(
            &self,
            _session_id: &str,
            _start: chrono::DateTime<Utc>,
            _end: chrono::DateTime<Utc>,
        ) -> anyhow::Result<(Option<i64>, Option<i64>)> {
            Ok((None, None))
        }

        async fn retained(&self, ids: &[i64]) -> anyhow::Result<HashSet<i64>> {
            Ok(ids
                .iter()
                .copied()
                .filter(|id| self.0.contains(id))
                .collect())
        }
    }

    async fn span(pool: &SqlitePool, source_event_id: i64) -> i64 {
        sqlx::query(
            "INSERT INTO memory_spans
                (span_kind, source_event_id, content, content_hash, valid_from)
             VALUES ('message', ?, 'synthetic', ?, '2026-01-01T00:00:00Z')",
        )
        .bind(source_event_id)
        .bind(format!("hash-{source_event_id}"))
        .execute(pool)
        .await
        .unwrap()
        .last_insert_rowid()
    }

    /// Spans of dropped messages go, in batches over the 500-id limit;
    /// spans of kept messages, and spans backing a claim, stay.
    #[tokio::test]
    async fn spans_of_dropped_messages_go_unless_kept_or_claimed() {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let mut kept = HashSet::new();
        let mut expected_left = Vec::new();
        for event_id in 1..=1200 {
            let span_id = span(&pool, event_id).await;
            if event_id % 3 == 0 {
                kept.insert(event_id);
                expected_left.push(span_id);
            }
        }
        let claimed = span(&pool, 5000).await;
        sqlx::query(
            "INSERT INTO memory_claims
                (subject, predicate, object, claim_text, source_span_id, provenance, valid_from)
             VALUES ('owner', 'likes', 'tea', 'owner likes tea', ?, 'test',
                     '2026-01-01T00:00:00Z')",
        )
        .bind(claimed)
        .execute(&pool)
        .await
        .unwrap();
        expected_left.push(claimed);

        let retention = MemoryRetention::new(
            pool.clone(),
            MemoryRetentionPolicy {
                superseded_facts_days: 0,
                episodes_days: 0,
                behavior_patterns_days: 0,
                procedures_days: 0,
                error_solutions_days: 0,
            },
        )
        .with_message_log(Arc::new(Keeps(kept)));
        assert_eq!(
            retention.cleanup_spans_of_dropped_messages().await.unwrap(),
            800
        );
        let mut left: Vec<i64> = sqlx::query_scalar("SELECT id FROM memory_spans")
            .fetch_all(&pool)
            .await
            .unwrap();
        left.sort();
        expected_left.sort();
        assert_eq!(left, expected_left);
        assert_eq!(
            retention.cleanup_spans_of_dropped_messages().await.unwrap(),
            0
        );
    }
}
