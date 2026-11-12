use std::collections::HashSet;

use sqlx::Row;

use super::*;
use crate::embedding_index::{
    content_hash, EmbeddingHit, EmbeddingIndex, EmbeddingItem, EmbeddingSearchFilter, MemoryOwner,
};
use crate::stores::{
    GraphEdgeUpsert, GraphEntityUpsert, GraphNeighbor, GraphSearchScope, MemoryGraphStore,
};

/// Ids per lookup query, well under SQLite's bound-parameter limit.
const LOOKUP_CHUNK: usize = 500;

/// Hit scores by numeric owner id; ids that are not numbers are skipped.
fn owner_scores(hits: &[EmbeddingHit]) -> std::collections::HashMap<i64, f32> {
    let mut scores = std::collections::HashMap::new();
    for hit in hits {
        if let Ok(id) = hit.owner.owner_id.parse::<i64>() {
            scores
                .entry(id)
                .and_modify(|existing: &mut f32| *existing = existing.max(hit.score))
                .or_insert(hit.score);
        }
    }
    scores
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryClaimRecord {
    pub id: i64,
    pub claim_text: String,
    pub source_fact_id: Option<i64>,
    pub channel_id: Option<String>,
    pub privacy: FactPrivacy,
}

impl SqliteMemory {
    pub async fn canonical_memory_health(
        &self,
    ) -> anyhow::Result<crate::records::MemoryHealthReport> {
        async fn count(pool: &SqlitePool, sql: &str) -> anyhow::Result<i64> {
            Ok(sqlx::query_scalar(sql).fetch_one(pool).await?)
        }
        Ok(crate::records::MemoryHealthReport {
            spans: count(&self.pool, "SELECT COUNT(*) FROM memory_spans WHERE deleted_at IS NULL")
                .await?,
            active_claims: count(
                &self.pool,
                "SELECT COUNT(*) FROM memory_claims WHERE deleted_at IS NULL AND valid_to IS NULL",
            )
            .await?,
            entities: count(
                &self.pool,
                "SELECT COUNT(*) FROM memory_entities WHERE deleted_at IS NULL",
            )
            .await?,
            active_edges: count(
                &self.pool,
                "SELECT COUNT(*) FROM memory_edges WHERE deleted_at IS NULL AND valid_to IS NULL",
            )
            .await?,
            active_embeddings: count(
                &self.pool,
                "SELECT COUNT(*) FROM memory_embeddings WHERE stale_at IS NULL",
            )
            .await?,
            stale_embeddings: count(
                &self.pool,
                "SELECT COUNT(*) FROM memory_embeddings WHERE stale_at IS NOT NULL",
            )
            .await?,
            facts_missing_claims: count(
                &self.pool,
                "SELECT COUNT(*) FROM facts f LEFT JOIN memory_claims c ON c.source_fact_id = f.id
                 WHERE f.superseded_at IS NULL AND c.id IS NULL",
            )
            .await?,
            episodes_missing_spans: count(
                &self.pool,
                "SELECT COUNT(*) FROM episodes e LEFT JOIN memory_spans s ON s.source_episode_id = e.id
                 WHERE s.id IS NULL",
            )
            .await?,
            orphan_edges: count(
                &self.pool,
                "SELECT COUNT(*) FROM memory_edges edge
                 WHERE NOT EXISTS (SELECT 1 FROM memory_entities e WHERE e.id = edge.source_entity_id)
                    OR NOT EXISTS (SELECT 1 FROM memory_entities e WHERE e.id = edge.target_entity_id)
                    OR (edge.source_claim_id IS NOT NULL AND NOT EXISTS (
                        SELECT 1 FROM memory_claims c WHERE c.id = edge.source_claim_id))",
            )
            .await?,
            embedding_dimension_mismatches: count(
                &self.pool,
                "SELECT COUNT(*) FROM memory_embeddings
                 WHERE embedding_dim <= 0 OR length(embedding) != embedding_dim * 4",
            )
            .await?,
        })
    }

    pub async fn graph_fact_ids_for_query(&self, query: &str) -> anyhow::Result<HashSet<i64>> {
        let tokens: Vec<String> = query
            .split(|ch: char| !ch.is_alphanumeric())
            .map(str::trim)
            .filter(|token| token.len() >= 3)
            .take(12)
            .map(str::to_lowercase)
            .collect();
        let mut ids = HashSet::new();
        for token in tokens {
            let pattern = format!("%{token}%");
            let rows: Vec<i64> = sqlx::query_scalar(
                "SELECT DISTINCT c.source_fact_id
                 FROM memory_edges edge
                 JOIN memory_entities source ON source.id = edge.source_entity_id
                 JOIN memory_entities target ON target.id = edge.target_entity_id
                 JOIN memory_claims c ON c.id = edge.source_claim_id
                 WHERE edge.deleted_at IS NULL AND edge.valid_to IS NULL
                   AND c.deleted_at IS NULL AND c.valid_to IS NULL
                   AND c.source_fact_id IS NOT NULL
                   AND (lower(source.display_name) LIKE ? OR lower(source.canonical_name) LIKE ?
                        OR lower(target.display_name) LIKE ? OR lower(target.canonical_name) LIKE ?
                        OR lower(edge.relation) LIKE ?)
                 LIMIT 500",
            )
            .bind(&pattern)
            .bind(&pattern)
            .bind(&pattern)
            .bind(&pattern)
            .bind(&pattern)
            .fetch_all(&self.pool)
            .await?;
            ids.extend(rows);
        }
        Ok(ids)
    }

    pub async fn search_embedding_owners(
        &self,
        query: &[f32],
        owner_type: &str,
        purpose: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<EmbeddingHit>> {
        self.embedding_index
            .search(
                query,
                &EmbeddingSearchFilter {
                    owner_types: vec![owner_type.to_string()],
                    purpose: Some(purpose.to_string()),
                    model: Some(self.embedding_service.model_id()),
                },
                limit,
            )
            .await
    }

    pub async fn fact_embedding_scores(
        &self,
        query: &[f32],
        limit: usize,
    ) -> anyhow::Result<std::collections::HashMap<i64, f32>> {
        let hits = self
            .search_embedding_owners(query, "claim", "recall", limit)
            .await?;
        let claim_scores = owner_scores(&hits);
        let mut scores = std::collections::HashMap::new();
        // One query per chunk of hits, not one per hit: recall asks for as
        // many hits as there are facts.
        let ids: Vec<i64> = claim_scores.keys().copied().collect();
        for chunk in ids.chunks(LOOKUP_CHUNK) {
            let mut builder = sqlx::QueryBuilder::<sqlx::Sqlite>::new(
                "SELECT c.id AS claim_id, c.source_fact_id AS fact_id
                 FROM memory_claims c
                 JOIN facts f ON f.id = c.source_fact_id
                 JOIN memory_embeddings me
                   ON me.owner_type = 'claim' AND me.owner_id = CAST(c.id AS TEXT)
                  AND me.embedding_purpose = 'recall' AND me.embedding_model = ",
            );
            builder.push_bind(self.embedding_service.model_id());
            builder.push(
                " AND me.stale_at IS NULL
                 WHERE c.deleted_at IS NULL AND c.valid_to IS NULL
                   AND ((f.embedding IS NOT NULL AND me.embedding = f.embedding)
                     OR (f.embedding IS NULL
                         AND julianday(me.updated_at) >= julianday(f.updated_at)))
                   AND c.id IN (",
            );
            let mut separated = builder.separated(", ");
            for id in chunk {
                separated.push_bind(*id);
            }
            separated.push_unseparated(")");
            for row in builder.build().fetch_all(&self.pool).await? {
                let claim_id: i64 = row.get("claim_id");
                let Some(fact_id) = row.get::<Option<i64>, _>("fact_id") else {
                    continue;
                };
                let Some(score) = claim_scores.get(&claim_id).copied() else {
                    continue;
                };
                scores
                    .entry(fact_id)
                    .and_modify(|existing: &mut f32| *existing = existing.max(score))
                    .or_insert(score);
            }
        }
        Ok(scores)
    }

    pub async fn episode_embedding_scores(
        &self,
        query: &[f32],
        limit: usize,
    ) -> anyhow::Result<std::collections::HashMap<i64, f32>> {
        let hits = self
            .search_embedding_owners(query, "span", "recall", limit)
            .await?;
        let span_scores = owner_scores(&hits);
        let mut scores = std::collections::HashMap::new();
        let ids: Vec<i64> = span_scores.keys().copied().collect();
        for chunk in ids.chunks(LOOKUP_CHUNK) {
            let mut builder = sqlx::QueryBuilder::<sqlx::Sqlite>::new(
                "SELECT id, source_episode_id FROM memory_spans
                 WHERE deleted_at IS NULL AND valid_to IS NULL AND id IN (",
            );
            let mut separated = builder.separated(", ");
            for id in chunk {
                separated.push_bind(*id);
            }
            separated.push_unseparated(")");
            for row in builder.build().fetch_all(&self.pool).await? {
                let span_id: i64 = row.get("id");
                let Some(episode_id) = row.get::<Option<i64>, _>("source_episode_id") else {
                    continue;
                };
                let Some(score) = span_scores.get(&span_id).copied() else {
                    continue;
                };
                // Several spans can come from one episode: keep the best.
                scores
                    .entry(episode_id)
                    .and_modify(|existing: &mut f32| *existing = existing.max(score))
                    .or_insert(score);
            }
        }
        Ok(scores)
    }

    pub async fn upsert_owner_embedding(
        &self,
        owner_type: &str,
        owner_id: i64,
        purpose: &str,
        content: &str,
        embedding: Vec<f32>,
    ) -> anyhow::Result<()> {
        self.embedding_index
            .upsert(EmbeddingItem {
                owner: MemoryOwner::new(owner_type, owner_id),
                purpose: purpose.to_string(),
                model: self.embedding_service.model_id(),
                content_hash: content_hash(content),
                embedding,
            })
            .await
    }

    pub async fn project_fact_memory(&self, fact_id: i64) -> anyhow::Result<()> {
        let Some(row) = sqlx::query(
            "SELECT id, category, key, value, source, channel_id, privacy, embedding,
                    created_at, updated_at, superseded_at, first_seen_at, source_excerpt
             FROM facts WHERE id = ?",
        )
        .bind(fact_id)
        .fetch_optional(&self.pool)
        .await?
        else {
            return Ok(());
        };

        let superseded_at: Option<String> = row.get("superseded_at");
        if let Some(valid_to) = superseded_at {
            sqlx::query(
                "UPDATE memory_claims SET valid_to = COALESCE(valid_to, ?), updated_at = datetime('now')
                 WHERE source_fact_id = ?",
            )
            .bind(&valid_to)
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
            if let Some(claim_id) = sqlx::query_scalar::<_, i64>(
                "SELECT id FROM memory_claims WHERE source_fact_id = ?",
            )
            .bind(fact_id)
            .fetch_optional(&self.pool)
            .await?
            {
                self.embedding_index
                    .mark_stale(&MemoryOwner::new("claim", claim_id))
                    .await?;
                MemoryGraphStore::invalidate_claim(self, claim_id, &valid_to).await?;
            }
            return Ok(());
        }

        let category: String = row.get("category");
        let key: String = row.get("key");
        let value: String = row.get("value");
        let source: String = row.get("source");
        let channel_id: Option<String> = row.get("channel_id");
        let privacy: String = row.get("privacy");
        let valid_from: String = row.get("created_at");
        let first_seen_at: Option<String> = row.get("first_seen_at");
        let source_excerpt: Option<String> = row.get("source_excerpt");
        let evidence = find_evidence_span(
            &self.pool,
            source_excerpt.as_deref(),
            channel_id.as_deref(),
            first_seen_at.as_deref(),
        )
        .await?;
        let claim_text = super::facts::build_fact_embedding_text(&category, &key, &value);

        sqlx::query(
            "INSERT INTO memory_claims
                (subject, predicate, object, claim_text, source_fact_id, source_span_id,
                 source_event_id, provenance,
                 confidence, channel_id, privacy, valid_from, updated_at)
             VALUES ('owner', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(source_fact_id) DO UPDATE SET
                predicate = excluded.predicate,
                object = excluded.object,
                claim_text = excluded.claim_text,
                source_span_id = COALESCE(excluded.source_span_id, memory_claims.source_span_id),
                source_event_id = COALESCE(excluded.source_event_id, memory_claims.source_event_id),
                provenance = excluded.provenance,
                confidence = excluded.confidence,
                channel_id = excluded.channel_id,
                privacy = excluded.privacy,
                valid_to = NULL,
                deleted_at = NULL,
                updated_at = datetime('now')",
        )
        .bind(&key)
        .bind(&value)
        .bind(&claim_text)
        .bind(fact_id)
        .bind(evidence.map(|(span_id, _)| span_id))
        .bind(evidence.and_then(|(_, event_id)| event_id))
        .bind(&source)
        .bind(if source == "consolidation" { 0.7 } else { 1.0 })
        .bind(channel_id.as_deref())
        .bind(&privacy)
        .bind(&valid_from)
        .execute(&self.pool)
        .await?;

        let claim_id: i64 =
            sqlx::query_scalar("SELECT id FROM memory_claims WHERE source_fact_id = ?")
                .bind(fact_id)
                .fetch_one(&self.pool)
                .await?;

        if let Some(blob) = row.get::<Option<Vec<u8>>, _>("embedding") {
            if let Ok(embedding) = decode_embedding(&blob) {
                self.embedding_index
                    .upsert(EmbeddingItem {
                        owner: MemoryOwner::new("claim", claim_id),
                        purpose: "recall".to_string(),
                        model: self.embedding_service.model_id(),
                        content_hash: content_hash(&claim_text),
                        embedding,
                    })
                    .await?;
            }
        }

        self.project_claim_graph(claim_id, &category, &key, &privacy, channel_id.as_deref())
            .await?;
        Ok(())
    }

    pub async fn project_episode_memory(&self, episode_id: i64) -> anyhow::Result<()> {
        let Some(row) = sqlx::query(
            "SELECT id, session_id, summary, channel_id, embedding, start_time, end_time
             FROM episodes WHERE id = ?",
        )
        .bind(episode_id)
        .fetch_optional(&self.pool)
        .await?
        else {
            return Ok(());
        };
        let summary: String = row.get("summary");
        let session_id: String = row.get("session_id");
        let channel_id: Option<String> = row.get("channel_id");
        let start_time: String = row.get("start_time");
        let end_time: String = row.get("end_time");
        sqlx::query(
            "INSERT INTO memory_spans
                (span_kind, source_episode_id, session_id, channel_id, role, content,
                content_hash, privacy, observed_from, observed_to, valid_from)
             VALUES ('episode', ?, ?, ?, 'summary', ?, ?, 'channel', ?, ?, ?)
             ON CONFLICT(source_episode_id) DO UPDATE SET
                session_id = excluded.session_id,
                channel_id = excluded.channel_id,
                content = excluded.content,
                content_hash = excluded.content_hash,
                observed_from = excluded.observed_from,
                observed_to = excluded.observed_to,
                valid_from = excluded.valid_from,
                valid_to = NULL,
                deleted_at = NULL",
        )
        .bind(episode_id)
        .bind(&session_id)
        .bind(channel_id.as_deref())
        .bind(&summary)
        .bind(content_hash(&summary))
        .bind(&start_time)
        .bind(&end_time)
        .bind(&start_time)
        .execute(&self.pool)
        .await?;
        let span_id: i64 =
            sqlx::query_scalar("SELECT id FROM memory_spans WHERE source_episode_id = ?")
                .bind(episode_id)
                .fetch_one(&self.pool)
                .await?;
        if let Some(blob) = row.get::<Option<Vec<u8>>, _>("embedding") {
            if let Ok(embedding) = decode_embedding(&blob) {
                self.embedding_index
                    .upsert(EmbeddingItem {
                        owner: MemoryOwner::new("span", span_id),
                        purpose: "recall".to_string(),
                        model: self.embedding_service.model_id(),
                        content_hash: content_hash(&summary),
                        embedding,
                    })
                    .await?;
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn rebuild_memory_projections(&self) -> anyhow::Result<(usize, usize)> {
        let fact_ids: Vec<i64> = sqlx::query_scalar("SELECT id FROM facts ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        for id in &fact_ids {
            self.project_fact_memory(*id).await?;
        }
        let episode_ids: Vec<i64> = sqlx::query_scalar("SELECT id FROM episodes ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        for id in &episode_ids {
            self.project_episode_memory(*id).await?;
        }
        Ok((fact_ids.len(), episode_ids.len()))
    }

    pub async fn backfill_missing_memory_projections(
        &self,
    ) -> anyhow::Result<(usize, usize, usize, usize, usize)> {
        const BATCH_SIZE: i64 = 500;
        let selected_event_spans_before: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_spans WHERE source_event_id IS NOT NULL",
        )
        .fetch_one(&self.pool)
        .await?;
        let mut fact_count = 0usize;
        loop {
            let ids: Vec<i64> = sqlx::query_scalar(
                "SELECT f.id FROM facts f
                 LEFT JOIN memory_claims c ON c.source_fact_id = f.id
                 WHERE (f.superseded_at IS NULL AND c.id IS NULL)
                    OR (f.superseded_at IS NOT NULL
                        AND c.id IS NOT NULL
                        AND c.valid_to IS NULL)
                 ORDER BY f.id LIMIT ?",
            )
            .bind(BATCH_SIZE)
            .fetch_all(&self.pool)
            .await?;
            if ids.is_empty() {
                break;
            }
            fact_count += ids.len();
            for id in ids {
                self.project_fact_memory(id).await?;
            }
        }

        let mut episode_count = 0usize;
        loop {
            let ids: Vec<i64> = sqlx::query_scalar(
                "SELECT e.id FROM episodes e
                 LEFT JOIN memory_spans s ON s.source_episode_id = e.id
                 WHERE s.id IS NULL ORDER BY e.id LIMIT ?",
            )
            .bind(BATCH_SIZE)
            .fetch_all(&self.pool)
            .await?;
            if ids.is_empty() {
                break;
            }
            episode_count += ids.len();
            for id in ids {
                self.project_episode_memory(id).await?;
            }
        }

        // Raw user messages remain canonical events and exact-history rows.
        // They enter the durable memory graph only when a selected fact/claim
        // needs that exact event as provenance; an allowed memory policy is a
        // capability boundary, not a decision to remember every control turn.
        let evidence_fact_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT f.id
             FROM facts f
             JOIN memory_claims c ON c.source_fact_id = f.id
             WHERE c.source_span_id IS NULL
               AND f.source_excerpt IS NOT NULL
               AND trim(f.source_excerpt) != ''
             ORDER BY f.id",
        )
        .fetch_all(&self.pool)
        .await?;
        for id in evidence_fact_ids {
            self.project_fact_memory(id).await?;
        }
        let selected_event_spans_after: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_spans WHERE source_event_id IS NOT NULL",
        )
        .fetch_one(&self.pool)
        .await?;
        let event_count =
            selected_event_spans_after.saturating_sub(selected_event_spans_before) as usize;

        let mut procedure_count = 0usize;
        let mut after_id = 0i64;
        loop {
            let rows = sqlx::query(
                "SELECT p.id, p.trigger_pattern, p.trigger_embedding
                 FROM procedures p
                 WHERE p.id > ? AND p.trigger_embedding IS NOT NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM memory_embeddings me
                       WHERE me.owner_type = 'procedure' AND me.owner_id = CAST(p.id AS TEXT)
                         AND me.embedding_purpose = 'trigger' AND me.embedding_model = ?
                         AND me.stale_at IS NULL
                   )
                 ORDER BY p.id LIMIT ?",
            )
            .bind(after_id)
            .bind(self.embedding_service.model_id())
            .bind(BATCH_SIZE)
            .fetch_all(&self.pool)
            .await?;
            if rows.is_empty() {
                break;
            }
            for row in rows {
                let id: i64 = row.get("id");
                after_id = id;
                let content: String = row.get("trigger_pattern");
                let blob: Vec<u8> = row.get("trigger_embedding");
                if let Ok(embedding) = decode_embedding(&blob) {
                    self.upsert_owner_embedding("procedure", id, "trigger", &content, embedding)
                        .await?;
                    procedure_count += 1;
                }
            }
        }

        let mut error_count = 0usize;
        after_id = 0;
        loop {
            let rows = sqlx::query(
                "SELECT e.id, e.error_pattern, e.error_embedding
                 FROM error_solutions e
                 WHERE e.id > ? AND e.error_embedding IS NOT NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM memory_embeddings me
                       WHERE me.owner_type = 'error_solution' AND me.owner_id = CAST(e.id AS TEXT)
                         AND me.embedding_purpose = 'error' AND me.embedding_model = ?
                         AND me.stale_at IS NULL
                   )
                 ORDER BY e.id LIMIT ?",
            )
            .bind(after_id)
            .bind(self.embedding_service.model_id())
            .bind(BATCH_SIZE)
            .fetch_all(&self.pool)
            .await?;
            if rows.is_empty() {
                break;
            }
            for row in rows {
                let id: i64 = row.get("id");
                after_id = id;
                let content: String = row.get("error_pattern");
                let blob: Vec<u8> = row.get("error_embedding");
                if let Ok(embedding) = decode_embedding(&blob) {
                    self.upsert_owner_embedding("error_solution", id, "error", &content, embedding)
                        .await?;
                    error_count += 1;
                }
            }
        }

        Ok((
            fact_count,
            episode_count,
            event_count,
            procedure_count,
            error_count,
        ))
    }

    pub async fn sync_fact_memory_category(&self, category: &str) -> anyhow::Result<()> {
        let ids: Vec<i64> = sqlx::query_scalar("SELECT id FROM facts WHERE category = ?")
            .bind(category)
            .fetch_all(&self.pool)
            .await?;
        self.sync_fact_memory_rows(category, &ids).await
    }

    /// Project the facts one write changed: the rows of `category` whose
    /// `updated_at` or `superseded_at` is that write's timestamp. Every row
    /// a fact upsert touches carries its timestamp, and re-projecting the
    /// whole category on each write took seconds in a large one.
    pub(crate) async fn sync_fact_memory_written_at(
        &self,
        category: &str,
        written_at: &str,
    ) -> anyhow::Result<()> {
        let ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM facts
             WHERE category = ? AND (updated_at = ? OR superseded_at = ?)",
        )
        .bind(category)
        .bind(written_at)
        .bind(written_at)
        .fetch_all(&self.pool)
        .await?;
        self.sync_fact_memory_rows(category, &ids).await
    }

    async fn sync_fact_memory_rows(&self, category: &str, ids: &[i64]) -> anyhow::Result<()> {
        for id in ids {
            self.project_fact_memory(*id).await?;
        }
        sqlx::query(
            "UPDATE memory_claims AS old
             SET superseded_by_claim_id = (
                SELECT replacement.id
                FROM memory_claims replacement
                JOIN facts new_fact ON new_fact.id = replacement.source_fact_id
                JOIN facts old_fact ON old_fact.id = old.source_fact_id
                WHERE new_fact.category = old_fact.category
                  AND new_fact.key = old_fact.key
                  AND new_fact.superseded_at IS NULL
                LIMIT 1
             )
             WHERE old.source_fact_id IN (SELECT id FROM facts WHERE category = ?)
               AND old.valid_to IS NOT NULL",
        )
        .bind(category)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// FTS5 claim search with visibility and temporal filtering at the owner row.
    pub async fn search_memory_claims(
        &self,
        query: &str,
        channel_id: Option<&str>,
        visibility: ChannelVisibility,
        requester_is_owner: bool,
        limit: usize,
    ) -> anyhow::Result<Vec<MemoryClaimRecord>> {
        let Some(fts_query) = sanitize_fts_query(query) else {
            return Ok(Vec::new());
        };
        if matches!(visibility, ChannelVisibility::PublicExternal) || limit == 0 {
            return Ok(Vec::new());
        }

        let rows = match sqlx::query(
            "SELECT c.id, c.claim_text, c.source_fact_id, c.channel_id, c.privacy
             FROM memory_claims_fts f
             JOIN memory_claims c ON c.id = f.rowid
             WHERE memory_claims_fts MATCH ?
               AND c.deleted_at IS NULL AND c.valid_to IS NULL
             ORDER BY bm25(memory_claims_fts) LIMIT ?",
        )
        .bind(&fts_query)
        .bind((limit.saturating_mul(8).max(limit)) as i64)
        .fetch_all(&self.pool)
        .await
        {
            Ok(rows) => rows,
            Err(error) if error.to_string().contains("no such table") => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };

        let mut claims = Vec::new();
        for row in rows {
            let privacy = FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy"));
            let claim_channel: Option<String> = row.get("channel_id");
            if memory_visible(
                self.channels.as_ref(),
                privacy,
                claim_channel.as_deref(),
                channel_id,
                visibility,
                requester_is_owner,
            ) {
                claims.push(MemoryClaimRecord {
                    id: row.get("id"),
                    claim_text: row.get("claim_text"),
                    source_fact_id: row.get("source_fact_id"),
                    channel_id: claim_channel,
                    privacy,
                });
                if claims.len() == limit {
                    break;
                }
            }
        }
        Ok(claims)
    }

    /// Traverse active graph edges to at most `max_depth` (clamped to two).
    #[allow(dead_code)]
    pub async fn memory_graph_neighbors(
        &self,
        start_entity_id: i64,
        max_depth: usize,
        channel_id: Option<&str>,
        visibility: ChannelVisibility,
        requester_is_owner: bool,
    ) -> anyhow::Result<Vec<GraphNeighbor>> {
        let max_depth = max_depth.min(2);
        let mut queue = VecDeque::from([(start_entity_id, 0usize)]);
        let mut visited = HashSet::from([start_entity_id]);
        let mut output = Vec::new();
        while let Some((entity_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let rows = sqlx::query(
                "SELECT e.source_entity_id, e.target_entity_id, e.relation, e.channel_id,
                        e.privacy, target.display_name
                 FROM memory_edges e
                 JOIN memory_entities target ON target.id = CASE
                    WHEN e.source_entity_id = ? THEN e.target_entity_id ELSE e.source_entity_id END
                 WHERE (e.source_entity_id = ? OR e.target_entity_id = ?)
                   AND e.deleted_at IS NULL AND e.valid_to IS NULL
                   AND target.deleted_at IS NULL
                 LIMIT 100",
            )
            .bind(entity_id)
            .bind(entity_id)
            .bind(entity_id)
            .fetch_all(&self.pool)
            .await?;
            for row in rows {
                let privacy = FactPrivacy::from_str_lossy(&row.get::<String, _>("privacy"));
                let edge_channel: Option<String> = row.get("channel_id");
                if !memory_visible(
                    self.channels.as_ref(),
                    privacy,
                    edge_channel.as_deref(),
                    channel_id,
                    visibility,
                    requester_is_owner,
                ) {
                    continue;
                }
                let source: i64 = row.get("source_entity_id");
                let target: i64 = row.get("target_entity_id");
                let neighbor_id = if source == entity_id { target } else { source };
                if visited.insert(neighbor_id) {
                    output.push(GraphNeighbor {
                        entity_id: neighbor_id,
                        display_name: row.get("display_name"),
                        relation: row.get("relation"),
                        depth: depth + 1,
                    });
                    queue.push_back((neighbor_id, depth + 1));
                }
            }
        }
        Ok(output)
    }

    async fn project_claim_graph(
        &self,
        claim_id: i64,
        category: &str,
        key: &str,
        privacy: &str,
        channel_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let privacy_level = FactPrivacy::from_str_lossy(privacy);
        let owner_id = MemoryGraphStore::upsert_entity(
            self,
            GraphEntityUpsert {
                entity_type: "person".to_string(),
                canonical_name: "owner".to_string(),
                display_name: "Owner".to_string(),
                aliases: Vec::new(),
                channel_id: channel_id.map(str::to_owned),
                privacy: privacy_level,
            },
        )
        .await?;
        let canonical = format!("{}:{}", category.to_lowercase(), key.to_lowercase());
        let concept_id = MemoryGraphStore::upsert_entity(
            self,
            GraphEntityUpsert {
                entity_type: "concept".to_string(),
                canonical_name: canonical,
                display_name: key.to_string(),
                aliases: Vec::new(),
                channel_id: channel_id.map(str::to_owned),
                privacy: privacy_level,
            },
        )
        .await?;
        MemoryGraphStore::upsert_edge(
            self,
            GraphEdgeUpsert {
                source_entity_id: owner_id,
                target_entity_id: concept_id,
                relation: "has_attribute".to_string(),
                source_claim_id: claim_id,
                confidence: 1.0,
                channel_id: channel_id.map(str::to_owned),
                privacy: privacy_level,
            },
        )
        .await?;

        Ok(())
    }

    pub async fn persist_extracted_fact_graph(
        &self,
        category: &str,
        key: &str,
        source_excerpt: &str,
        graph: &crate::records::ExtractedMemoryGraph,
    ) -> anyhow::Result<()> {
        use std::collections::HashMap;

        const MIN_CONFIDENCE: f32 = 0.65;
        let claim = sqlx::query(
            "SELECT c.id, c.channel_id, c.privacy
             FROM memory_claims c
             JOIN facts f ON f.id = c.source_fact_id
             WHERE lower(f.category) = lower(?) AND lower(f.key) = lower(?)
               AND f.superseded_at IS NULL AND c.deleted_at IS NULL AND c.valid_to IS NULL
             ORDER BY f.updated_at DESC LIMIT 1",
        )
        .bind(category)
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;
        let Some(claim) = claim else {
            return Ok(());
        };
        let claim_id: i64 = claim.get("id");
        let channel_id: Option<String> = claim.get("channel_id");
        let privacy = FactPrivacy::from_str_lossy(&claim.get::<String, _>("privacy"));
        let evidence = source_excerpt.to_lowercase();
        let mut resolved = HashMap::new();

        for entity in graph.entities.iter().take(32) {
            let name = entity.name.trim();
            let entity_type = normalize_graph_token(&entity.entity_type, 48);
            let local_id = entity.local_id.trim();
            if entity.confidence < MIN_CONFIDENCE
                || name.is_empty()
                || name.len() > 200
                || entity_type.is_empty()
                || local_id.is_empty()
                || resolved.contains_key(local_id)
                || !entity_is_grounded(name, &entity.aliases, &evidence)
            {
                continue;
            }
            let canonical_name = normalize_graph_name(name);
            if canonical_name.is_empty() {
                continue;
            }
            let (canonical_name, aliases) = resolve_entity_identity(
                &self.pool,
                &entity_type,
                &canonical_name,
                name,
                &entity.aliases,
            )
            .await?;
            let entity_id = MemoryGraphStore::upsert_entity(
                self,
                GraphEntityUpsert {
                    entity_type,
                    canonical_name,
                    display_name: name.to_string(),
                    aliases,
                    channel_id: channel_id.clone(),
                    privacy,
                },
            )
            .await?;
            resolved.insert(local_id.to_string(), entity_id);
        }

        for relationship in graph.relationships.iter().take(64) {
            let relation = normalize_graph_token(&relationship.relation, 80);
            let (Some(&source_entity_id), Some(&target_entity_id)) = (
                resolved.get(relationship.source_id.trim()),
                resolved.get(relationship.target_id.trim()),
            ) else {
                continue;
            };
            if relationship.confidence < MIN_CONFIDENCE
                || relation.is_empty()
                || source_entity_id == target_entity_id
            {
                continue;
            }
            MemoryGraphStore::upsert_edge(
                self,
                GraphEdgeUpsert {
                    source_entity_id,
                    target_entity_id,
                    relation,
                    source_claim_id: claim_id,
                    confidence: relationship.confidence.clamp(0.0, 1.0),
                    channel_id: channel_id.clone(),
                    privacy,
                },
            )
            .await?;
        }
        Ok(())
    }
}

fn normalize_graph_token(value: &str, max_len: usize) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .chars()
        .take(max_len)
        .collect()
}

fn normalize_graph_name(value: &str) -> String {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn entity_is_grounded(name: &str, aliases: &[String], evidence: &str) -> bool {
    name.eq_ignore_ascii_case("owner")
        || evidence.contains(&name.to_lowercase())
        || aliases
            .iter()
            .filter(|alias| alias.trim().len() >= 2)
            .any(|alias| evidence.contains(&alias.trim().to_lowercase()))
}

async fn resolve_entity_identity(
    pool: &SqlitePool,
    entity_type: &str,
    proposed_canonical: &str,
    display_name: &str,
    proposed_aliases: &[String],
) -> anyhow::Result<(String, Vec<String>)> {
    let mut proposed_names = HashSet::from([proposed_canonical.to_string()]);
    proposed_names.extend(
        proposed_aliases
            .iter()
            .map(|alias| normalize_graph_name(alias))
            .filter(|alias| !alias.is_empty()),
    );
    let rows = sqlx::query(
        "SELECT canonical_name, display_name, aliases_json
         FROM memory_entities
         WHERE entity_type = ? AND deleted_at IS NULL
         ORDER BY updated_at DESC LIMIT 500",
    )
    .bind(entity_type)
    .fetch_all(pool)
    .await?;
    for row in rows {
        let canonical: String = row.get("canonical_name");
        let existing_display: String = row.get("display_name");
        let mut aliases: Vec<String> =
            serde_json::from_str(&row.get::<String, _>("aliases_json")).unwrap_or_default();
        let existing_names: HashSet<String> = std::iter::once(canonical.clone())
            .chain(std::iter::once(normalize_graph_name(&existing_display)))
            .chain(aliases.iter().map(|alias| normalize_graph_name(alias)))
            .collect();
        if proposed_names.is_disjoint(&existing_names) {
            continue;
        }
        aliases.push(display_name.to_string());
        aliases.extend(proposed_aliases.iter().cloned());
        aliases.sort_by_key(|alias| alias.to_lowercase());
        aliases.dedup_by(|left, right| left.eq_ignore_ascii_case(right));
        aliases.retain(|alias| !alias.eq_ignore_ascii_case(&existing_display));
        aliases.truncate(12);
        return Ok((canonical, aliases));
    }
    let mut aliases: Vec<String> = proposed_aliases.iter().take(12).cloned().collect();
    aliases.sort_by_key(|alias| alias.to_lowercase());
    aliases.dedup_by(|left, right| left.eq_ignore_ascii_case(right));
    Ok((proposed_canonical.to_string(), aliases))
}

async fn find_evidence_span(
    pool: &SqlitePool,
    source_excerpt: Option<&str>,
    channel_id: Option<&str>,
    first_seen_at: Option<&str>,
) -> anyhow::Result<Option<(i64, Option<i64>)>> {
    let Some(excerpt) = source_excerpt
        .map(str::trim)
        .filter(|text| !text.is_empty())
    else {
        return Ok(None);
    };
    let row = sqlx::query(
        "SELECT id, source_event_id FROM memory_spans
         WHERE span_kind = 'message' AND deleted_at IS NULL
           AND (instr(content, ?) > 0 OR instr(?, content) > 0)
           AND (? IS NULL OR channel_id = ? OR channel_id IS NULL)
         ORDER BY CASE WHEN ? IS NULL THEN 0
                       ELSE ABS(julianday(observed_from) - julianday(?)) END ASC,
                  id DESC
         LIMIT 1",
    )
    .bind(excerpt)
    .bind(excerpt)
    .bind(channel_id)
    .bind(channel_id)
    .bind(first_seen_at)
    .bind(first_seen_at)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|row| (row.get("id"), row.get("source_event_id"))))
}

/// Index a user message from the host's log as a searchable span. The host
/// passes only messages memory may learn from. `channels` derives the span's
/// channel from its session.
pub async fn project_message_span(
    pool: &SqlitePool,
    channels: &dyn ChannelIds,
    message: &crate::history::LoggedMessage,
) -> anyhow::Result<()> {
    let event_id = message.id;
    let content = &message.content;
    if content.trim().is_empty() {
        return Ok(());
    }
    let session_id = &message.session_id;
    let channel_id = channels.channel_of_session(session_id);
    let created_at = &message.created_at;
    sqlx::query(
        "INSERT INTO memory_spans
            (span_kind, source_event_id, session_id, channel_id, role, content,
             content_hash, privacy, observed_from, valid_from)
         VALUES ('message', ?, ?, ?, 'user', ?, ?, 'channel', ?, ?)
         ON CONFLICT(source_event_id) WHERE source_event_id IS NOT NULL DO UPDATE SET
            content = excluded.content,
            content_hash = excluded.content_hash,
            channel_id = excluded.channel_id,
            deleted_at = NULL",
    )
    .bind(event_id)
    .bind(session_id)
    .bind(channel_id.as_deref())
    .bind(content)
    .bind(content_hash(content))
    .bind(created_at)
    .bind(created_at)
    .execute(pool)
    .await?;
    Ok(())
}

async fn upsert_entity(
    pool: &SqlitePool,
    entity_type: &str,
    canonical_name: &str,
    display_name: &str,
    aliases: &[String],
    privacy: &str,
    channel_id: Option<&str>,
) -> anyhow::Result<i64> {
    sqlx::query(
        "INSERT INTO memory_entities
            (entity_type, canonical_name, display_name, aliases_json, channel_id, privacy)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(entity_type, canonical_name) DO UPDATE SET
            display_name = excluded.display_name,
            aliases_json = excluded.aliases_json,
            channel_id = COALESCE(memory_entities.channel_id, excluded.channel_id),
            privacy = CASE WHEN memory_entities.privacy = 'private' OR excluded.privacy = 'private'
                           THEN 'private' ELSE excluded.privacy END,
            deleted_at = NULL,
            updated_at = datetime('now')",
    )
    .bind(entity_type)
    .bind(canonical_name)
    .bind(display_name)
    .bind(serde_json::to_string(aliases)?)
    .bind(channel_id)
    .bind(privacy)
    .execute(pool)
    .await?;
    Ok(sqlx::query_scalar(
        "SELECT id FROM memory_entities WHERE entity_type = ? AND canonical_name = ?",
    )
    .bind(entity_type)
    .bind(canonical_name)
    .fetch_one(pool)
    .await?)
}

#[async_trait]
impl MemoryGraphStore for SqliteMemory {
    async fn upsert_entity(&self, entity: GraphEntityUpsert) -> anyhow::Result<i64> {
        upsert_entity(
            &self.pool,
            &entity.entity_type,
            &entity.canonical_name,
            &entity.display_name,
            &entity.aliases,
            &entity.privacy.to_string(),
            entity.channel_id.as_deref(),
        )
        .await
    }

    async fn upsert_edge(&self, edge: GraphEdgeUpsert) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO memory_edges
                (source_entity_id, target_entity_id, relation, source_claim_id,
                 confidence, channel_id, privacy, valid_from)
             VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(source_entity_id, target_entity_id, relation, source_claim_id)
             DO UPDATE SET confidence = excluded.confidence,
                           channel_id = excluded.channel_id,
                           privacy = excluded.privacy,
                           valid_to = NULL, deleted_at = NULL",
        )
        .bind(edge.source_entity_id)
        .bind(edge.target_entity_id)
        .bind(&edge.relation)
        .bind(edge.source_claim_id)
        .bind(edge.confidence)
        .bind(edge.channel_id.as_deref())
        .bind(edge.privacy.to_string())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn invalidate_claim(&self, claim_id: i64, valid_to: &str) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE memory_edges SET valid_to = COALESCE(valid_to, ?)
             WHERE source_claim_id = ? AND deleted_at IS NULL",
        )
        .bind(valid_to)
        .bind(claim_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn neighbors(
        &self,
        start_entity_id: i64,
        max_depth: usize,
        scope: &GraphSearchScope,
    ) -> anyhow::Result<Vec<GraphNeighbor>> {
        self.memory_graph_neighbors(
            start_entity_id,
            max_depth,
            scope.channel_id.as_deref(),
            scope.visibility,
            scope.requester_is_owner,
        )
        .await
    }
}

fn sanitize_fts_query(query: &str) -> Option<String> {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(str::to_lowercase)
        .filter(|token| token.len() >= 3 && !super::facts::is_stopword(token))
        .take(16)
        .map(|token| format!("\"{}\"", token.replace('"', "")))
        .collect();
    (!tokens.is_empty()).then(|| tokens.join(" OR "))
}

pub(crate) fn memory_visible(
    channels: &dyn ChannelIds,
    privacy: FactPrivacy,
    stored_channel: Option<&str>,
    current_channel: Option<&str>,
    visibility: ChannelVisibility,
    requester_is_owner: bool,
) -> bool {
    if matches!(visibility, ChannelVisibility::PublicExternal) {
        return false;
    }
    if requester_is_owner
        && matches!(
            visibility,
            ChannelVisibility::Private | ChannelVisibility::Internal
        )
    {
        return true;
    }
    match privacy {
        FactPrivacy::Private => false,
        FactPrivacy::Global => !matches!(visibility, ChannelVisibility::Public),
        FactPrivacy::Channel => stored_channel
            .zip(current_channel)
            .is_some_and(|(stored, current)| channels.same_channel(stored, current)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_does_not_expose_operators() {
        assert_eq!(
            sanitize_fts_query("editor OR secret:*"),
            Some("\"editor\" OR \"secret\"".to_string())
        );
    }

    #[test]
    fn visibility_is_fail_closed() {
        assert!(!memory_visible(
            &ExactChannelIds,
            FactPrivacy::Private,
            None,
            None,
            ChannelVisibility::Public,
            false
        ));
        assert!(!memory_visible(
            &ExactChannelIds,
            FactPrivacy::Global,
            None,
            None,
            ChannelVisibility::PublicExternal,
            true
        ));
    }
}
