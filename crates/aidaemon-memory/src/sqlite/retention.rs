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
        let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
        let mut deleted = 0;
        for ids in dropped.chunks(500) {
            let placeholders = vec!["?"; ids.len()].join(", ");
            let sql = format!(
                "DELETE FROM memory_spans WHERE id IN ({placeholders})
                   AND NOT EXISTS (
                       SELECT 1 FROM memory_claims c WHERE c.source_span_id = memory_spans.id
                   )"
            );
            let mut query = sqlx::query(sqlx::AssertSqlSafe(&*sql));
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
    /// Preserves compact tombstones for deleted/folded rows before purging their
    /// payloads so background extractors still cannot resurrect them later.
    pub async fn cleanup_superseded_facts(&self) -> anyhow::Result<u64> {
        if self.config.superseded_facts_days == 0 {
            return Ok(0);
        }
        let cutoff =
            (Utc::now() - Duration::days(self.config.superseded_facts_days as i64)).to_rfc3339();
        let has_tombstones: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'memory_fact_tombstones'",
        )
        .fetch_one(&self.pool)
        .await?;
        let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
        let victims: Vec<(i64, String, String, String, String, Option<String>)> = sqlx::query_as(
            "SELECT id, category, key, value, superseded_at, superseded_reason
             FROM facts
             WHERE superseded_at IS NOT NULL AND superseded_at < ?
             LIMIT 500",
        )
        .bind(&cutoff)
        .fetch_all(&mut *tx)
        .await?;
        if victims.is_empty() {
            return Ok(0);
        }
        if has_tombstones > 0 {
            for (_, category, key, value, superseded_at, reason) in &victims {
                if !matches!(reason.as_deref(), Some("deleted" | "folded") | None) {
                    continue;
                }
                let canonical_key = crate::text::canonicalize_fact_key(key);
                let normalized_value = crate::text::normalize_fact_value(value);
                let reason_str = reason.as_deref().unwrap_or("unknown");
                sqlx::query(
                    "INSERT INTO memory_fact_tombstones
                        (category, canonical_key, normalized_value, reason, retired_at)
                     VALUES (?, ?, ?, ?, ?)
                     ON CONFLICT(category, canonical_key, normalized_value, reason)
                     DO UPDATE SET retired_at = MAX(retired_at, excluded.retired_at)",
                )
                .bind(category.trim().to_ascii_lowercase())
                .bind(&canonical_key)
                .bind(&normalized_value)
                .bind(reason_str)
                .bind(superseded_at)
                .execute(&mut *tx)
                .await?;
            }
        }
        let placeholders = vec!["?"; victims.len()].join(", ");
        let sql = format!("DELETE FROM facts WHERE id IN ({placeholders})");
        let mut query = sqlx::query(sqlx::AssertSqlSafe(&*sql));
        for (id, _, _, _, _, _) in &victims {
            query = query.bind(id);
        }
        let deleted = query.execute(&mut *tx).await?.rows_affected();
        tx.commit().await?;
        Ok(deleted)
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

    #[tokio::test]
    async fn deleted_fact_tombstone_survives_superseded_retention_cleanup() {
        use crate::records::FactPrivacy;
        use crate::stores::FactStore as _;

        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let mem = crate::sqlite::SqliteMemory::new(
            pool.clone(),
            Arc::new(crate::embeddings::EmbeddingService::deterministic()),
        );
        let old_seen = Utc::now() - Duration::days(120);
        mem.upsert_fact_with_provenance(
            "preference",
            "coffee_style",
            "iced latte",
            "user_stated",
            None,
            FactPrivacy::Global,
            Some(old_seen),
            None,
        )
        .await
        .unwrap();
        let id = mem.get_facts(Some("preference")).await.unwrap()[0].id;
        mem.delete_fact(id).await.unwrap();

        // Age the superseded row past the retention window and purge it.
        let ancient = (Utc::now() - Duration::days(100)).to_rfc3339();
        sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
            .bind(&ancient)
            .bind(id)
            .execute(&pool)
            .await
            .unwrap();
        let retention = MemoryRetention::new(
            pool.clone(),
            MemoryRetentionPolicy {
                superseded_facts_days: 30,
                episodes_days: 0,
                behavior_patterns_days: 0,
                procedures_days: 0,
                error_solutions_days: 0,
            },
        );
        assert_eq!(retention.cleanup_superseded_facts().await.unwrap(), 1);
        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM facts")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(remaining, 0, "old superseded payload was purged");

        // Automated consolidation with older evidence must STILL be blocked by the durable tombstone.
        mem.upsert_fact_with_provenance(
            "preference",
            "coffee_style",
            "iced latte",
            "consolidation",
            None,
            FactPrivacy::Global,
            Some(old_seen),
            None,
        )
        .await
        .unwrap();
        assert!(
            mem.get_facts(Some("preference")).await.unwrap().is_empty(),
            "durable tombstone must block resurrection after retention cleanup"
        );

        // Evidence newer than the deletion is allowed through.
        let fresh_seen = Utc::now() + Duration::seconds(5);
        mem.upsert_fact_with_provenance(
            "preference",
            "coffee_style",
            "iced latte",
            "consolidation",
            None,
            FactPrivacy::Global,
            Some(fresh_seen),
            None,
        )
        .await
        .unwrap();
        assert_eq!(mem.get_facts(Some("preference")).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn archive_restore_edit_and_expiry_lifecycle() {
        use crate::records::FactPrivacy;
        use crate::stores::FactStore as _;

        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let mem = crate::sqlite::SqliteMemory::new(
            pool.clone(),
            Arc::new(crate::embeddings::EmbeddingService::deterministic()),
        );
        mem.upsert_fact(
            "preference",
            "favorite_dish",
            "ramen",
            "owner_quoted",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
        let id = mem.get_facts(Some("preference")).await.unwrap()[0].id;

        // Inline edit upgrades provenance to user_stated and updates value.
        let edited = mem
            .update_fact_value(id, "tonkotsu ramen")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(edited.value, "tonkotsu ramen");
        assert_eq!(edited.source, "user_stated");

        // Archive removes from active list, invalidates memory_claims.valid_to,
        // puts in archived list, and blocks automated resurrection.
        assert!(mem.archive_fact(id).await.unwrap());
        let active_claims: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_claims WHERE source_fact_id = ? AND valid_to IS NULL",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            active_claims, 0,
            "archiving a fact must invalidate its active memory_claims row"
        );
        assert!(mem.get_facts(Some("preference")).await.unwrap().is_empty());
        assert_eq!(mem.get_archived_facts(20).await.unwrap().len(), 1);
        mem.upsert_fact(
            "preference",
            "favorite_dish",
            "ramen",
            "consolidation",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
        assert!(mem.get_facts(Some("preference")).await.unwrap().is_empty());

        // Restore returns it cleanly to active recall and re-activates its claim.
        let restored = mem.restore_fact(id).await.unwrap().unwrap();
        assert_eq!(restored.id, id);
        let active_claims_after: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_claims WHERE source_fact_id = ? AND valid_to IS NULL",
        )
        .bind(id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(active_claims_after, 1);
        assert_eq!(mem.get_facts(Some("preference")).await.unwrap().len(), 1);
        assert!(mem.get_archived_facts(20).await.unwrap().is_empty());

        // Typed expiry archives ONLY expired facts, never durable idle ones.
        mem.upsert_fact(
            "technical",
            "temp_release_branch",
            "release/synthetic",
            "derived",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
        let temp_id = mem.get_facts(Some("technical")).await.unwrap()[0].id;
        let past = (Utc::now() - Duration::hours(2)).to_rfc3339();
        sqlx::query("UPDATE facts SET expires_at = ? WHERE id = ?")
            .bind(&past)
            .bind(temp_id)
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(mem.archive_expired_facts(Utc::now()).await.unwrap(), 1);
        assert!(mem.get_facts(Some("technical")).await.unwrap().is_empty());
        assert_eq!(
            mem.get_facts(Some("preference")).await.unwrap().len(),
            1,
            "durable fact with recall_count=0 must stay active"
        );
    }

    #[tokio::test]
    async fn forgetting_canonical_owner_fact_also_retires_shadowed_flat_row() {
        use crate::records::FactPrivacy;
        use crate::stores::FactStore as _;

        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        let mem = crate::sqlite::SqliteMemory::new(
            pool,
            Arc::new(crate::embeddings::EmbeddingService::deterministic()),
        );
        mem.upsert_fact(
            "user",
            "residence",
            "Portland",
            "user_stated",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
        mem.backfill_structured_personal_memory().await.unwrap();
        let canonical_res = mem
            .get_canonical_memory_facts()
            .await
            .unwrap()
            .into_iter()
            .find(|f| f.category == "user" && f.key == "residence")
            .expect("canonical residence backfilled");
        assert!(canonical_res.id < 0);

        // Forget the canonical negative ID: BOTH canonical AND shadowed flat row must retire.
        mem.delete_fact(canonical_res.id).await.unwrap();
        assert!(
            mem.get_canonical_memory_facts()
                .await
                .unwrap()
                .iter()
                .all(|f| f.key != "residence"),
            "canonical residence must be retracted"
        );
        assert!(
            mem.get_facts(Some("user")).await.unwrap().is_empty(),
            "shadowed flat residence row must also be retired instead of resurrecting"
        );

        // Same for projected alias families (`username` -> `online_handle`):
        // editing canonical hides stale flat values in explicit search/recall,
        // forgetting canonical retires the flat row, and derived writes cannot
        // reactivate a forgotten alias.
        mem.upsert_fact(
            "user",
            "username",
            "@alice_dev",
            "user_stated",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
        mem.backfill_structured_personal_memory().await.unwrap();
        let canonical_handle = mem
            .get_canonical_memory_facts()
            .await
            .unwrap()
            .into_iter()
            .find(|f| f.category == "user" && f.key == "online_handle")
            .expect("canonical online_handle backfilled");
        mem.update_fact_value(canonical_handle.id, "@alice_new")
            .await
            .unwrap();
        let search_hits = mem
            .search_facts("alice_dev", 10, &crate::search::FactSearchConfig::default())
            .await
            .unwrap();
        assert!(
            search_hits
                .hits
                .iter()
                .chain(search_hits.related.iter())
                .all(|f| f.fact.value != "@alice_dev"),
            "edited canonical handle must shadow stale flat username in old-value-only search: {search_hits:?}"
        );
        let recall_hits = mem.get_relevant_facts("alice_dev", 10).await.unwrap();
        assert!(
            recall_hits.iter().all(|f| f.value != "@alice_dev"),
            "edited canonical handle must shadow stale flat username in old-value-only recall: {recall_hits:?}"
        );
        mem.delete_fact(canonical_handle.id).await.unwrap();
        assert!(
            mem.get_facts(Some("user")).await.unwrap().is_empty(),
            "shadowed flat username row must also retire when canonical online_handle is forgotten"
        );
        // Neither the retracted alias (@alice_new) nor the tombstoned former
        // value (@alice_dev) may be resurrected by a derived write.
        for candidate_val in ["@alice_new", "@alice_dev"] {
            let derived_write = crate::records::PersonalMemoryWrite {
                direct_user_statement: false,
                entities: vec![crate::records::PersonalEntityCandidate {
                    local_id: "owner".to_string(),
                    entity_type: "person".to_string(),
                    canonical_name: "Owner".to_string(),
                    is_reference: false,
                    canonical_name_confirmed: false,
                }],
                aliases: vec![crate::records::PersonalAliasCandidate {
                    entity_local_id: "owner".to_string(),
                    value: candidate_val.to_string(),
                    alias_type: "online_handle".to_string(),
                }],
                ..Default::default()
            };
            mem.reconcile_personal_memory(
                &derived_write,
                "derived",
                None,
                None,
                FactPrivacy::Private,
            )
            .await
            .unwrap();
        }
        assert!(
            mem.get_canonical_memory_facts()
                .await
                .unwrap()
                .iter()
                .all(|f| f.key != "online_handle"),
            "derived write must not resurrect a retracted or tombstoned canonical alias"
        );
    }
}
