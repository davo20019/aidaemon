use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Datelike, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use tokio::sync::RwLock;

use crate::traits::{
    BehaviorPattern, ConversationSummary, Episode, ErrorSolution, Expertise, Fact, Goal,
    GoalSchedule, GoalTokenBudgetStatus, Message, Procedure, Task, TaskActivity, TokenUsage,
    TokenUsageRecord, UserProfile,
};
use crate::types::{ChannelVisibility, FactPrivacy};

use crate::memory::binary::{decode_embedding, encode_embedding};
use crate::memory::embeddings::EmbeddingService;
use crate::utils::truncate_str;

/// Set restrictive file permissions (0600) on the database and WAL files.
fn set_db_file_permissions(db_path: &str) {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::Permissions::from_mode(0o600);
    // Main database file
    if let Err(e) = std::fs::set_permissions(db_path, mode.clone()) {
        tracing::warn!("Failed to set permissions on {}: {}", db_path, e);
    }
    // WAL and shared-memory files created by SQLite in WAL journal mode
    for suffix in &["-wal", "-shm"] {
        let path = format!("{}{}", db_path, suffix);
        if std::path::Path::new(&path).exists() {
            if let Err(e) = std::fs::set_permissions(&path, mode.clone()) {
                tracing::warn!("Failed to set permissions on {}: {}", path, e);
            }
        }
    }
}

/// Migrate facts schema to allow supersession history.
///
/// Legacy databases used a table-level `UNIQUE(category, key)` constraint which prevents
/// inserting a new version after marking the old one superseded. We rebuild the table once
/// to remove that constraint, then enforce at most one *active* fact per key via a partial
/// unique index: `UNIQUE(category, key) WHERE superseded_at IS NULL`.
async fn migrate_facts_history_schema(pool: &SqlitePool) -> anyhow::Result<()> {
    let table_row = sqlx::query(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'facts' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?;

    let Some(row) = table_row else {
        // No facts table yet.
        return Ok(());
    };

    let create_sql: Option<String> = row.try_get("sql").unwrap_or(None);
    let create_sql = create_sql.unwrap_or_default();
    let create_l = create_sql.to_lowercase();
    let has_legacy_unique = create_l.contains("unique(category, key)")
        || create_l.contains("unique(category,key)")
        || create_l.contains("unique (category, key)")
        || create_l.contains("unique (category,key)");

    if has_legacy_unique {
        let mut tx = pool.begin().await?;

        // Defensive: clean up a prior failed attempt.
        let _ = sqlx::query("DROP TABLE IF EXISTS facts_new")
            .execute(&mut *tx)
            .await;

        sqlx::query(
            "CREATE TABLE facts_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                superseded_at TEXT,
                recall_count INTEGER DEFAULT 0,
                last_recalled_at TEXT,
                channel_id TEXT,
                privacy TEXT DEFAULT 'global',
                embedding BLOB
            )",
        )
        .execute(&mut *tx)
        .await?;

        // Copy all existing columns. These columns should exist after migrations, but
        // keeping this in a single statement makes the rebuild fast.
        sqlx::query(
            "INSERT INTO facts_new
                (id, category, key, value, source, created_at, updated_at, superseded_at,
                 recall_count, last_recalled_at, channel_id, privacy, embedding)
             SELECT
                id, category, key, value, source, created_at, updated_at, superseded_at,
                recall_count, last_recalled_at, channel_id, privacy, embedding
             FROM facts",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query("DROP TABLE facts").execute(&mut *tx).await?;
        sqlx::query("ALTER TABLE facts_new RENAME TO facts")
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
    }

    // (Re)create indexes. If the table was rebuilt, previous indexes were dropped.
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_facts_channel ON facts(channel_id)")
        .execute(pool)
        .await;
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_facts_privacy ON facts(privacy)")
        .execute(pool)
        .await;
    if let Err(e) = sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_active_unique
         ON facts(category, key) WHERE superseded_at IS NULL",
    )
    .execute(pool)
    .await
    {
        tracing::warn!(
            "Failed to create idx_facts_active_unique (active fact dedupe not enforced): {}",
            e
        );
    }

    Ok(())
}

pub struct SqliteStateStore {
    pool: SqlitePool,
    working_memory: Arc<RwLock<HashMap<String, VecDeque<Message>>>>,
    cap: usize,
    embedding_service: Arc<EmbeddingService>,
}

impl SqliteStateStore {
    pub async fn new(
        db_path: &str,
        cap: usize,
        encryption_key: Option<&str>,
        embedding_service: Arc<EmbeddingService>,
    ) -> anyhow::Result<Self> {
        let mut opts = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

        // SQLCipher: set encryption key via connect options so it applies to
        // every connection in the pool (PRAGMA key must be the first statement).
        #[cfg(feature = "encryption")]
        if let Some(key) = encryption_key {
            if !key.is_empty() {
                let escaped_key = key.replace('\'', "''");
                opts = opts.pragma("key", format!("'{}'", escaped_key));
                tracing::info!("SQLCipher encryption enabled for database");
            }
        }

        #[cfg(not(feature = "encryption"))]
        if let Some(key) = encryption_key {
            if !key.is_empty() {
                anyhow::bail!(
                    "Database encryption_key is set in config but aidaemon was compiled without the 'encryption' feature. \
                     Rebuild with: cargo build --features encryption"
                );
            }
        }

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await?;

        // Set restrictive file permissions (owner-only read/write)
        set_db_file_permissions(db_path);

        migrations::migrate_state(&pool).await?;

        Ok(Self {
            pool,
            working_memory: Arc::new(RwLock::new(HashMap::new())),
            cap,
            embedding_service,
        })
    }

    async fn hydrate_from_events(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<VecDeque<Message>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at
            FROM events
            WHERE session_id = ?
              AND event_type IN ('user_message', 'assistant_response', 'tool_result')
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(limit.max(1) as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut deque = VecDeque::with_capacity(rows.len());
        for row in rows.into_iter().rev() {
            let event_id: i64 = row.get("id");
            let event_type: String = row.get("event_type");
            let row_session_id: String = row.get("session_id");
            let created_str: String = row.get("created_at");
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let data_raw: String = row.get("data");
            let data: serde_json::Value = match serde_json::from_str(&data_raw) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        session_id,
                        event_id,
                        event_type = %event_type,
                        error = %e,
                        "hydrate_from_events: skipping malformed event payload"
                    );
                    continue;
                }
            };

            if let Some(message) = crate::events::turn_from_event(
                event_id,
                &row_session_id,
                &event_type,
                &data,
                created_at,
            )
            .map(|turn| turn.into_message())
            {
                deque.push_back(message);
            }
        }

        Ok(deque)
    }

    /// Hydrate working memory for a session from the database.
    async fn hydrate(&self, session_id: &str) -> anyhow::Result<VecDeque<Message>> {
        let deque = self.hydrate_from_events(session_id, self.cap).await?;
        tracing::debug!(
            session_id,
            hydrated_count = deque.len(),
            "hydrate: loaded conversation from canonical event stream"
        );
        Ok(deque)
    }
    pub fn pool(&self) -> SqlitePool {
        self.pool.clone()
    }

    // ==================== Episode Methods ====================
    // Note: These advanced memory methods are reserved for future integration.

    /// Insert a new episode and return its ID.
    #[allow(dead_code)]
    pub async fn insert_episode(&self, episode: &Episode) -> anyhow::Result<i64> {
        let topics_json = episode
            .topics
            .as_ref()
            .map(|t| serde_json::to_string(t).unwrap_or_default());
        let result = sqlx::query(
            "INSERT INTO episodes (session_id, summary, topics, emotional_tone, outcome, embedding, importance, recall_count, last_recalled_at, message_count, start_time, end_time, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&episode.session_id)
        .bind(&episode.summary)
        .bind(&topics_json)
        .bind(&episode.emotional_tone)
        .bind(&episode.outcome)
        .bind::<Option<Vec<u8>>>(None) // embedding - set separately
        .bind(episode.importance)
        .bind(episode.recall_count)
        .bind(episode.last_recalled_at.map(|t| t.to_rfc3339()))
        .bind(episode.message_count)
        .bind(episode.start_time.to_rfc3339())
        .bind(episode.end_time.to_rfc3339())
        .bind(episode.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    /// Get episodes relevant to a query using embedding similarity.
    pub async fn get_relevant_episodes(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Episode>> {
        // First get all episodes with embeddings
        let rows = sqlx::query(
            "SELECT id, session_id, summary, topics, emotional_tone, outcome, importance, recall_count, last_recalled_at, message_count, start_time, end_time, created_at, embedding
             FROM episodes WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT 500"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || query.trim().is_empty() {
            // Return most recent if no query
            return self.get_recent_episodes(limit).await;
        }

        // Embed the query
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => return self.get_recent_episodes(limit).await,
        };

        // Score by similarity with memory decay
        // Threshold increased to 0.5 to avoid cross-session contamination from marginally related episodes
        const EPISODE_SIMILARITY_THRESHOLD: f32 = 0.5;

        let mut scored: Vec<(Episode, f32)> = Vec::new();
        for row in rows {
            let embedding: Option<Vec<u8>> = row.get("embedding");
            if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    let similarity = crate::memory::math::cosine_similarity(&query_vec, &vec);
                    let episode = self.row_to_episode(&row)?;
                    let score = crate::memory::scoring::memory_score(
                        similarity,
                        episode.created_at,
                        episode.recall_count,
                        episode.last_recalled_at,
                    );
                    if score > EPISODE_SIMILARITY_THRESHOLD {
                        scored.push((episode, score));
                    }
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Log retrieved episodes to help debug cross-session contamination
        if !scored.is_empty() {
            tracing::debug!(
                count = scored.len(),
                top_score = scored.first().map(|(_, s)| *s).unwrap_or(0.0),
                top_summary = scored
                    .first()
                    .map(|(e, _)| truncate_str(&e.summary, 50))
                    .unwrap_or_default(),
                "Retrieved relevant episodes"
            );
        }

        let episodes: Vec<Episode> = scored.into_iter().take(limit).map(|(e, _)| e).collect();
        Ok(episodes)
    }

    /// Get most recent episodes.
    pub async fn get_recent_episodes(&self, limit: usize) -> anyhow::Result<Vec<Episode>> {
        let rows = sqlx::query(
            "SELECT id, session_id, summary, topics, emotional_tone, outcome, importance, recall_count, last_recalled_at, message_count, start_time, end_time, created_at
             FROM episodes ORDER BY created_at DESC LIMIT ?"
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut episodes = Vec::with_capacity(rows.len());
        for row in rows {
            episodes.push(self.row_to_episode(&row)?);
        }
        Ok(episodes)
    }

    /// Increment recall count for an episode.
    #[allow(dead_code)]
    pub async fn increment_episode_recall(&self, episode_id: i64) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE episodes SET recall_count = recall_count + 1, last_recalled_at = ? WHERE id = ?")
            .bind(&now)
            .bind(episode_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Update episode embedding.
    #[allow(dead_code)]
    pub async fn update_episode_embedding(
        &self,
        episode_id: i64,
        embedding: &[f32],
    ) -> anyhow::Result<()> {
        let blob = encode_embedding(embedding);
        sqlx::query("UPDATE episodes SET embedding = ? WHERE id = ?")
            .bind(blob)
            .bind(episode_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Backfill missing episode embeddings.
    /// Called on startup to ensure all episodes have embeddings for semantic search.
    pub async fn backfill_episode_embeddings(&self) -> anyhow::Result<usize> {
        let rows: Vec<sqlx::sqlite::SqliteRow> =
            sqlx::query("SELECT id, summary FROM episodes WHERE embedding IS NULL")
                .fetch_all(&self.pool)
                .await?;

        if rows.is_empty() {
            return Ok(0);
        }

        tracing::info!(count = rows.len(), "Backfilling missing episode embeddings");

        let mut backfilled = 0;
        for row in rows {
            let id: i64 = row.get("id");
            let summary: String = row.get("summary");

            match self.embedding_service.embed(summary).await {
                Ok(embedding) => {
                    let blob = encode_embedding(&embedding);
                    sqlx::query("UPDATE episodes SET embedding = ? WHERE id = ?")
                        .bind(blob)
                        .bind(id)
                        .execute(&self.pool)
                        .await?;
                    backfilled += 1;
                }
                Err(e) => {
                    tracing::warn!(episode_id = id, error = %e, "Failed to generate embedding for episode");
                }
            }
        }

        tracing::info!(backfilled, "Episode embedding backfill complete");
        Ok(backfilled)
    }

    /// Backfill missing fact embeddings.
    /// Called on startup to ensure all active facts have pre-computed embeddings.
    pub async fn backfill_fact_embeddings(&self) -> anyhow::Result<usize> {
        let rows: Vec<sqlx::sqlite::SqliteRow> = sqlx::query(
            "SELECT id, category, key, value FROM facts WHERE embedding IS NULL AND superseded_at IS NULL",
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(0);
        }

        tracing::info!(count = rows.len(), "Backfilling missing fact embeddings");

        let mut backfilled = 0;
        for row in rows {
            let id: i64 = row.get("id");
            let category: String = row.get("category");
            let key: String = row.get("key");
            let value: String = row.get("value");
            let fact_text = format!("[{}] {}: {}", category, key, value);

            match self.embedding_service.embed(fact_text).await {
                Ok(embedding) => {
                    let blob = encode_embedding(&embedding);
                    sqlx::query("UPDATE facts SET embedding = ? WHERE id = ?")
                        .bind(blob)
                        .bind(id)
                        .execute(&self.pool)
                        .await?;
                    backfilled += 1;
                }
                Err(e) => {
                    tracing::warn!(fact_id = id, error = %e, "Failed to generate embedding for fact");
                }
            }
        }

        tracing::info!(backfilled, "Fact embedding backfill complete");
        Ok(backfilled)
    }

    fn row_to_fact(row: &sqlx::sqlite::SqliteRow) -> Fact {
        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");
        let superseded_str: Option<String> = row.get("superseded_at");
        let last_recalled_str: Option<String> = row.get("last_recalled_at");
        let privacy_str: Option<String> = row.try_get("privacy").unwrap_or(None);
        let channel_id: Option<String> = row.try_get("channel_id").unwrap_or(None);

        let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Fact {
            id: row.get("id"),
            category: row.get("category"),
            key: row.get("key"),
            value: row.get("value"),
            source: row.get("source"),
            created_at,
            updated_at,
            superseded_at: superseded_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            recall_count: row.try_get("recall_count").unwrap_or(0),
            last_recalled_at: last_recalled_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            channel_id,
            privacy: privacy_str
                .map(|s| FactPrivacy::from_str_lossy(&s))
                .unwrap_or(FactPrivacy::Global),
        }
    }

    fn row_to_episode(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Episode> {
        let topics_json: Option<String> = row.get("topics");
        let topics = topics_json.and_then(|j| serde_json::from_str(&j).ok());

        let start_str: String = row.get("start_time");
        let end_str: String = row.get("end_time");
        let created_str: String = row.get("created_at");
        let last_recalled_str: Option<String> = row.get("last_recalled_at");
        let channel_id: Option<String> = row.try_get("channel_id").unwrap_or(None);

        Ok(Episode {
            id: row.get("id"),
            session_id: row.get("session_id"),
            summary: row.get("summary"),
            topics,
            emotional_tone: row.get("emotional_tone"),
            outcome: row.get("outcome"),
            importance: row.get("importance"),
            recall_count: row.get("recall_count"),
            last_recalled_at: last_recalled_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            message_count: row.get("message_count"),
            start_time: chrono::DateTime::parse_from_rfc3339(&start_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            end_time: chrono::DateTime::parse_from_rfc3339(&end_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            channel_id,
        })
    }

    // ==================== UserProfile Methods ====================

    /// Get or create the user profile.
    pub async fn get_user_profile(&self) -> anyhow::Result<UserProfile> {
        let row = sqlx::query(
            "SELECT id, verbosity_preference, explanation_depth, tone_preference, emoji_preference, typical_session_length, active_hours, common_workflows, asks_before_acting, prefers_explanations, likes_suggestions, updated_at
             FROM user_profile LIMIT 1"
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let active_hours_json: Option<String> = row.get("active_hours");
            let workflows_json: Option<String> = row.get("common_workflows");
            let updated_str: String = row.get("updated_at");

            Ok(UserProfile {
                id: row.get("id"),
                verbosity_preference: row.get("verbosity_preference"),
                explanation_depth: row.get("explanation_depth"),
                tone_preference: row.get("tone_preference"),
                emoji_preference: row.get("emoji_preference"),
                typical_session_length: row.get("typical_session_length"),
                active_hours: active_hours_json.and_then(|j| serde_json::from_str(&j).ok()),
                common_workflows: workflows_json.and_then(|j| serde_json::from_str(&j).ok()),
                asks_before_acting: row.get::<i32, _>("asks_before_acting") == 1,
                prefers_explanations: row.get::<i32, _>("prefers_explanations") == 1,
                likes_suggestions: row.get::<i32, _>("likes_suggestions") == 1,
                updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        } else {
            // Create default profile
            let now = Utc::now();
            sqlx::query("INSERT INTO user_profile (updated_at) VALUES (?)")
                .bind(now.to_rfc3339())
                .execute(&self.pool)
                .await?;

            Ok(UserProfile {
                id: 1,
                verbosity_preference: "medium".to_string(),
                explanation_depth: "moderate".to_string(),
                tone_preference: "neutral".to_string(),
                emoji_preference: "none".to_string(),
                typical_session_length: None,
                active_hours: None,
                common_workflows: None,
                asks_before_acting: true,
                prefers_explanations: true,
                likes_suggestions: false,
                updated_at: now,
            })
        }
    }

    /// Update user profile fields.
    #[allow(dead_code)]
    pub async fn update_user_profile(&self, profile: &UserProfile) -> anyhow::Result<()> {
        let active_hours_json = profile
            .active_hours
            .as_ref()
            .map(|h| serde_json::to_string(h).unwrap_or_default());
        let workflows_json = profile
            .common_workflows
            .as_ref()
            .map(|w| serde_json::to_string(w).unwrap_or_default());
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            "UPDATE user_profile SET verbosity_preference = ?, explanation_depth = ?, tone_preference = ?, emoji_preference = ?, typical_session_length = ?, active_hours = ?, common_workflows = ?, asks_before_acting = ?, prefers_explanations = ?, likes_suggestions = ?, updated_at = ? WHERE id = ?"
        )
        .bind(&profile.verbosity_preference)
        .bind(&profile.explanation_depth)
        .bind(&profile.tone_preference)
        .bind(&profile.emoji_preference)
        .bind(profile.typical_session_length)
        .bind(&active_hours_json)
        .bind(&workflows_json)
        .bind(profile.asks_before_acting as i32)
        .bind(profile.prefers_explanations as i32)
        .bind(profile.likes_suggestions as i32)
        .bind(&now)
        .bind(profile.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ==================== BehaviorPattern Methods ====================

    /// Insert a new behavior pattern.
    #[allow(dead_code)]
    pub async fn insert_behavior_pattern(&self, pattern: &BehaviorPattern) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO behavior_patterns (pattern_type, description, trigger_context, action, confidence, occurrence_count, last_seen_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&pattern.pattern_type)
        .bind(&pattern.description)
        .bind(&pattern.trigger_context)
        .bind(&pattern.action)
        .bind(pattern.confidence)
        .bind(pattern.occurrence_count)
        .bind(pattern.last_seen_at.map(|t| t.to_rfc3339()))
        .bind(pattern.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    /// Get behavior patterns above a confidence threshold.
    pub async fn get_behavior_patterns(
        &self,
        min_confidence: f32,
    ) -> anyhow::Result<Vec<BehaviorPattern>> {
        let rows = sqlx::query(
            "SELECT id, pattern_type, description, trigger_context, action, confidence, occurrence_count, last_seen_at, created_at
             FROM behavior_patterns WHERE confidence >= ? ORDER BY confidence DESC, occurrence_count DESC"
        )
        .bind(min_confidence)
        .fetch_all(&self.pool)
        .await?;

        let mut patterns = Vec::with_capacity(rows.len());
        for row in rows {
            let created_str: String = row.get("created_at");
            let last_seen_str: Option<String> = row.get("last_seen_at");

            patterns.push(BehaviorPattern {
                id: row.get("id"),
                pattern_type: row.get("pattern_type"),
                description: row.get("description"),
                trigger_context: row.get("trigger_context"),
                action: row.get("action"),
                confidence: row.get("confidence"),
                occurrence_count: row.get("occurrence_count"),
                last_seen_at: last_seen_str.and_then(|s| {
                    chrono::DateTime::parse_from_rfc3339(&s)
                        .ok()
                        .map(|dt| dt.with_timezone(&Utc))
                }),
                created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            });
        }
        Ok(patterns)
    }

    /// Insert or update a behavior pattern by logical key
    /// (pattern_type + trigger_context + action).
    pub async fn record_behavior_pattern(
        &self,
        pattern_type: &str,
        description: &str,
        trigger_context: Option<&str>,
        action: Option<&str>,
        confidence_hint: f32,
        occurrence_delta: i32,
    ) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();

        let existing = sqlx::query(
            "SELECT id FROM behavior_patterns
             WHERE pattern_type = ?
               AND COALESCE(trigger_context, '') = COALESCE(?, '')
               AND COALESCE(action, '') = COALESCE(?, '')
             LIMIT 1",
        )
        .bind(pattern_type)
        .bind(trigger_context)
        .bind(action)
        .fetch_optional(&self.pool)
        .await?;

        let delta = occurrence_delta.max(1);
        let confidence_hint = confidence_hint.clamp(0.1, 0.98);

        if let Some(row) = existing {
            let id: i64 = row.get("id");
            sqlx::query(
                "UPDATE behavior_patterns
                 SET description = ?,
                     occurrence_count = occurrence_count + ?,
                     confidence = MIN(0.98, MAX(confidence, ?) + 0.02),
                     last_seen_at = ?
                 WHERE id = ?",
            )
            .bind(description)
            .bind(delta)
            .bind(confidence_hint)
            .bind(&now)
            .bind(id)
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query(
                "INSERT INTO behavior_patterns (pattern_type, description, trigger_context, action, confidence, occurrence_count, last_seen_at, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(pattern_type)
            .bind(description)
            .bind(trigger_context)
            .bind(action)
            .bind(confidence_hint)
            .bind(delta)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    /// Update pattern occurrence and confidence.
    #[allow(dead_code)]
    pub async fn update_behavior_pattern(
        &self,
        pattern_id: i64,
        confidence_delta: f32,
    ) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE behavior_patterns SET occurrence_count = occurrence_count + 1, confidence = MIN(1.0, confidence + ?), last_seen_at = ? WHERE id = ?"
        )
        .bind(confidence_delta)
        .bind(&now)
        .bind(pattern_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ==================== Procedure Methods ====================

    /// Insert a new procedure.
    pub async fn insert_procedure(&self, procedure: &Procedure) -> anyhow::Result<i64> {
        let steps_json = serde_json::to_string(&procedure.steps)?;
        // Best-effort embedding for semantic retrieval. This runs off the hot-path
        // (learning is background) and will be backfilled by MemoryManager if missing.
        let trigger_embedding = self
            .embedding_service
            .embed(procedure.trigger_pattern.clone())
            .await
            .ok()
            .map(|v| encode_embedding(&v));
        let result = sqlx::query(
            "INSERT INTO procedures (name, trigger_pattern, trigger_embedding, steps, success_count, failure_count, avg_duration_secs, last_used_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(name) DO UPDATE SET
                trigger_pattern = excluded.trigger_pattern,
                steps = excluded.steps,
                success_count = procedures.success_count + excluded.success_count,
                failure_count = procedures.failure_count + excluded.failure_count,
                avg_duration_secs = COALESCE(excluded.avg_duration_secs, procedures.avg_duration_secs),
                last_used_at = COALESCE(excluded.last_used_at, procedures.last_used_at),
                updated_at = excluded.updated_at,
                trigger_embedding = COALESCE(trigger_embedding, excluded.trigger_embedding)"
        )
        .bind(&procedure.name)
        .bind(&procedure.trigger_pattern)
        .bind(&trigger_embedding)
        .bind(&steps_json)
        .bind(procedure.success_count)
        .bind(procedure.failure_count)
        .bind(procedure.avg_duration_secs)
        .bind(procedure.last_used_at.map(|t| t.to_rfc3339()))
        .bind(procedure.created_at.to_rfc3339())
        .bind(procedure.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    /// Get procedures relevant to a query.
    pub async fn get_relevant_procedures(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Procedure>> {
        let rows = sqlx::query(
            "SELECT id, name, trigger_pattern, steps, success_count, failure_count, avg_duration_secs, last_used_at, created_at, updated_at, trigger_embedding
             FROM procedures WHERE success_count > failure_count ORDER BY success_count DESC LIMIT 100"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || query.trim().is_empty() {
            let mut procedures = Vec::new();
            for row in rows.into_iter().take(limit) {
                procedures.push(self.row_to_procedure(&row)?);
            }
            return Ok(procedures);
        }

        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => {
                let mut procedures = Vec::new();
                for row in rows.into_iter().take(limit) {
                    procedures.push(self.row_to_procedure(&row)?);
                }
                return Ok(procedures);
            }
        };

        let mut scored: Vec<(Procedure, f32)> = Vec::new();
        for row in rows {
            let embedding: Option<Vec<u8>> = row.get("trigger_embedding");
            let procedure = self.row_to_procedure(&row)?;

            let score = if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    crate::memory::math::cosine_similarity(&query_vec, &vec)
                } else {
                    0.0
                }
            } else {
                // Fall back to text matching
                if procedure
                    .trigger_pattern
                    .to_lowercase()
                    .contains(&query.to_lowercase())
                {
                    0.5
                } else {
                    0.0
                }
            };

            if score > 0.4 {
                scored.push((procedure, score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(p, _)| p).collect())
    }

    /// Update procedure success/failure and optionally update steps.
    #[allow(dead_code)] // Wired to trait but reserved for feedback loop
    pub async fn update_procedure(
        &self,
        procedure_id: i64,
        success: bool,
        new_steps: Option<&[String]>,
        duration_secs: Option<f32>,
    ) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();

        if success {
            sqlx::query("UPDATE procedures SET success_count = success_count + 1, last_used_at = ?, updated_at = ? WHERE id = ?")
                .bind(&now)
                .bind(&now)
                .bind(procedure_id)
                .execute(&self.pool)
                .await?;
        } else {
            sqlx::query("UPDATE procedures SET failure_count = failure_count + 1, last_used_at = ?, updated_at = ? WHERE id = ?")
                .bind(&now)
                .bind(&now)
                .bind(procedure_id)
                .execute(&self.pool)
                .await?;
        }

        if let Some(steps) = new_steps {
            let steps_json = serde_json::to_string(steps)?;
            sqlx::query("UPDATE procedures SET steps = ?, updated_at = ? WHERE id = ?")
                .bind(&steps_json)
                .bind(&now)
                .bind(procedure_id)
                .execute(&self.pool)
                .await?;
        }

        if let Some(duration) = duration_secs {
            // Update running average
            sqlx::query("UPDATE procedures SET avg_duration_secs = COALESCE((avg_duration_secs * (success_count + failure_count - 1) + ?) / (success_count + failure_count), ?), updated_at = ? WHERE id = ?")
                .bind(duration)
                .bind(duration)
                .bind(&now)
                .bind(procedure_id)
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    fn row_to_procedure(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Procedure> {
        let steps_json: String = row.get("steps");
        let steps: Vec<String> = serde_json::from_str(&steps_json).unwrap_or_default();
        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");
        let last_used_str: Option<String> = row.get("last_used_at");

        Ok(Procedure {
            id: row.get("id"),
            name: row.get("name"),
            trigger_pattern: row.get("trigger_pattern"),
            steps,
            success_count: row.get("success_count"),
            failure_count: row.get("failure_count"),
            avg_duration_secs: row.get("avg_duration_secs"),
            last_used_at: last_used_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    // ==================== Expertise Methods ====================

    /// Get or create expertise for a domain.
    pub async fn get_or_create_expertise(&self, domain: &str) -> anyhow::Result<Expertise> {
        let row = sqlx::query(
            "SELECT id, domain, tasks_attempted, tasks_succeeded, tasks_failed, current_level, confidence_score, common_errors, last_task_at, created_at, updated_at
             FROM expertise WHERE domain = ?"
        )
        .bind(domain)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            self.row_to_expertise(&row)
        } else {
            let now = Utc::now().to_rfc3339();
            let result = sqlx::query(
                "INSERT INTO expertise (domain, tasks_attempted, tasks_succeeded, tasks_failed, current_level, confidence_score, created_at, updated_at)
                 VALUES (?, 0, 0, 0, 'novice', 0.0, ?, ?)"
            )
            .bind(domain)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;

            Ok(Expertise {
                id: result.last_insert_rowid(),
                domain: domain.to_string(),
                tasks_attempted: 0,
                tasks_succeeded: 0,
                tasks_failed: 0,
                current_level: "novice".to_string(),
                confidence_score: 0.0,
                common_errors: None,
                last_task_at: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            })
        }
    }

    /// Get all expertise records.
    pub async fn get_all_expertise(&self) -> anyhow::Result<Vec<Expertise>> {
        let rows = sqlx::query(
            "SELECT id, domain, tasks_attempted, tasks_succeeded, tasks_failed, current_level, confidence_score, common_errors, last_task_at, created_at, updated_at
             FROM expertise ORDER BY confidence_score DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut expertise = Vec::with_capacity(rows.len());
        for row in rows {
            expertise.push(self.row_to_expertise(&row)?);
        }
        Ok(expertise)
    }

    /// Get trusted command patterns (3+ approvals) for AI context.
    pub async fn get_trusted_command_patterns(&self) -> anyhow::Result<Vec<(String, i32)>> {
        let rows: Vec<(String, i32)> = sqlx::query_as(
            "SELECT pattern, approval_count FROM command_patterns
             WHERE approval_count >= 3
             ORDER BY approval_count DESC
             LIMIT 10",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Increment expertise counters and update level.
    pub async fn increment_expertise(
        &self,
        domain: &str,
        success: bool,
        error: Option<&str>,
    ) -> anyhow::Result<()> {
        let _ = self.get_or_create_expertise(domain).await?; // Ensure exists
        let now = Utc::now().to_rfc3339();

        if success {
            sqlx::query("UPDATE expertise SET tasks_attempted = tasks_attempted + 1, tasks_succeeded = tasks_succeeded + 1, last_task_at = ?, updated_at = ? WHERE domain = ?")
                .bind(&now)
                .bind(&now)
                .bind(domain)
                .execute(&self.pool)
                .await?;
        } else {
            sqlx::query("UPDATE expertise SET tasks_attempted = tasks_attempted + 1, tasks_failed = tasks_failed + 1, last_task_at = ?, updated_at = ? WHERE domain = ?")
                .bind(&now)
                .bind(&now)
                .bind(domain)
                .execute(&self.pool)
                .await?;
        }

        // Record error if provided
        if let Some(err) = error {
            let row = sqlx::query("SELECT common_errors FROM expertise WHERE domain = ?")
                .bind(domain)
                .fetch_one(&self.pool)
                .await?;
            let errors_json: Option<String> = row.get("common_errors");
            let mut errors: Vec<String> = errors_json
                .and_then(|j| serde_json::from_str(&j).ok())
                .unwrap_or_default();
            if !errors.contains(&err.to_string()) {
                errors.push(err.to_string());
                if errors.len() > 10 {
                    errors.remove(0);
                }
                let errors_json = serde_json::to_string(&errors)?;
                sqlx::query("UPDATE expertise SET common_errors = ? WHERE domain = ?")
                    .bind(&errors_json)
                    .bind(domain)
                    .execute(&self.pool)
                    .await?;
            }
        }

        // Update level based on new counts
        let expertise = self.get_or_create_expertise(domain).await?;
        let (level, confidence) = crate::memory::expertise::calculate_expertise_level(
            expertise.tasks_succeeded,
            expertise.tasks_failed,
        );

        sqlx::query("UPDATE expertise SET current_level = ?, confidence_score = ?, updated_at = ? WHERE domain = ?")
            .bind(level)
            .bind(confidence)
            .bind(&now)
            .bind(domain)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    fn row_to_expertise(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Expertise> {
        let errors_json: Option<String> = row.get("common_errors");
        let common_errors = errors_json.and_then(|j| serde_json::from_str(&j).ok());
        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");
        let last_task_str: Option<String> = row.get("last_task_at");

        Ok(Expertise {
            id: row.get("id"),
            domain: row.get("domain"),
            tasks_attempted: row.get("tasks_attempted"),
            tasks_succeeded: row.get("tasks_succeeded"),
            tasks_failed: row.get("tasks_failed"),
            current_level: row.get("current_level"),
            confidence_score: row.get("confidence_score"),
            common_errors,
            last_task_at: last_task_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    // ==================== ErrorSolution Methods ====================

    /// Insert a new error solution.
    pub async fn insert_error_solution(&self, solution: &ErrorSolution) -> anyhow::Result<i64> {
        // Normalize so duplicates conflict and retrieval similarity is stable.
        let error_pattern =
            crate::memory::procedures::extract_error_pattern(&solution.error_pattern);
        let domain = solution.domain.clone().unwrap_or_default();
        let steps_json = solution
            .solution_steps
            .as_ref()
            .map(|s| serde_json::to_string(s).unwrap_or_default());
        // Best-effort embedding for semantic retrieval. Backfilled by MemoryManager if missing.
        let error_embedding = self
            .embedding_service
            .embed(error_pattern.clone())
            .await
            .ok()
            .map(|v| encode_embedding(&v));
        let now = Utc::now();
        let row: (i64,) = sqlx::query_as(
            r#"
            INSERT INTO error_solutions (
                error_pattern, error_embedding, domain, solution_summary, solution_steps,
                success_count, failure_count, last_used_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(error_pattern, domain, solution_summary) DO UPDATE SET
                error_embedding = COALESCE(error_solutions.error_embedding, excluded.error_embedding),
                solution_steps = COALESCE(excluded.solution_steps, error_solutions.solution_steps),
                success_count = error_solutions.success_count + excluded.success_count,
                failure_count = error_solutions.failure_count + excluded.failure_count,
                last_used_at = excluded.last_used_at
            RETURNING id
            "#,
        )
        .bind(&error_pattern)
        .bind(&error_embedding)
        .bind(&domain)
        .bind(&solution.solution_summary)
        .bind(&steps_json)
        .bind(solution.success_count)
        .bind(solution.failure_count)
        .bind(solution.last_used_at.unwrap_or(now).to_rfc3339())
        .bind(solution.created_at.to_rfc3339())
        .fetch_one(&self.pool)
        .await?;
        Ok(row.0)
    }

    /// Get error solutions relevant to an error message.
    pub async fn get_relevant_error_solutions(
        &self,
        error: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ErrorSolution>> {
        let rows = sqlx::query(
            "SELECT id, error_pattern, domain, solution_summary, solution_steps, success_count, failure_count, last_used_at, created_at, error_embedding
             FROM error_solutions
             WHERE success_count > failure_count
             ORDER BY (success_count - failure_count) DESC, COALESCE(last_used_at, created_at) DESC
             LIMIT 1000",
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || error.trim().is_empty() {
            let mut solutions = Vec::new();
            for row in rows.into_iter().take(limit) {
                solutions.push(self.row_to_error_solution(&row)?);
            }
            return Ok(solutions);
        }

        let query_vec = match self.embedding_service.embed(error.to_string()).await {
            Ok(v) => v,
            Err(_) => {
                let mut solutions = Vec::new();
                for row in rows.into_iter().take(limit) {
                    solutions.push(self.row_to_error_solution(&row)?);
                }
                return Ok(solutions);
            }
        };

        let mut scored: Vec<(ErrorSolution, f32)> = Vec::new();
        for row in rows {
            let embedding: Option<Vec<u8>> = row.get("error_embedding");
            let solution = self.row_to_error_solution(&row)?;

            let score = if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    crate::memory::math::cosine_similarity(&query_vec, &vec)
                } else {
                    0.0
                }
            } else {
                // Fall back to text matching
                if solution
                    .error_pattern
                    .to_lowercase()
                    .contains(&error.to_lowercase())
                {
                    0.5
                } else {
                    0.0
                }
            };

            if score > 0.4 {
                scored.push((solution, score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(s, _)| s).collect())
    }

    /// Update error solution success/failure.
    #[allow(dead_code)] // Wired to trait but reserved for feedback loop
    pub async fn update_error_solution(
        &self,
        solution_id: i64,
        success: bool,
    ) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        if success {
            sqlx::query("UPDATE error_solutions SET success_count = success_count + 1, last_used_at = ? WHERE id = ?")
                .bind(&now)
                .bind(solution_id)
                .execute(&self.pool)
                .await?;
        } else {
            sqlx::query("UPDATE error_solutions SET failure_count = failure_count + 1, last_used_at = ? WHERE id = ?")
                .bind(&now)
                .bind(solution_id)
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    fn row_to_error_solution(
        &self,
        row: &sqlx::sqlite::SqliteRow,
    ) -> anyhow::Result<ErrorSolution> {
        let steps_json: Option<String> = row.get("solution_steps");
        let solution_steps = steps_json.and_then(|j| serde_json::from_str(&j).ok());
        let created_str: String = row.get("created_at");
        let last_used_str: Option<String> = row.get("last_used_at");
        let domain_raw: Option<String> = row.get("domain");
        let domain = domain_raw.and_then(|d| {
            let t = d.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        });

        Ok(ErrorSolution {
            id: row.get("id"),
            error_pattern: row.get("error_pattern"),
            domain,
            solution_summary: row.get("solution_summary"),
            solution_steps,
            success_count: row.get("success_count"),
            failure_count: row.get("failure_count"),
            last_used_at: last_used_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }

    // ==================== Fact History Methods ====================

    /// Get the history of a fact (all versions including superseded).
    #[allow(dead_code)]
    pub async fn get_fact_history(&self, category: &str, key: &str) -> anyhow::Result<Vec<Fact>> {
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at
             FROM facts WHERE category = ? AND key = ? ORDER BY created_at DESC"
        )
        .bind(category)
        .bind(key)
        .fetch_all(&self.pool)
        .await?;

        let mut facts = Vec::with_capacity(rows.len());
        for row in rows {
            facts.push(self.row_to_fact_with_history(&row)?);
        }
        Ok(facts)
    }

    /// Increment recall count for a fact.
    #[allow(dead_code)]
    pub async fn increment_fact_recall(&self, fact_id: i64) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE facts SET recall_count = recall_count + 1, last_recalled_at = ? WHERE id = ?",
        )
        .bind(&now)
        .bind(fact_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    #[allow(dead_code)]
    fn row_to_fact_with_history(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Fact> {
        Ok(Self::row_to_fact(row))
    }

    fn row_to_person(row: &sqlx::sqlite::SqliteRow) -> crate::traits::Person {
        let aliases_str: String = row
            .try_get("aliases_json")
            .unwrap_or_else(|_| "[]".to_string());
        let platform_ids_str: String = row
            .try_get("platform_ids_json")
            .unwrap_or_else(|_| "{}".to_string());
        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");
        let last_interaction_str: Option<String> =
            row.try_get("last_interaction_at").unwrap_or(None);

        crate::traits::Person {
            id: row.get("id"),
            name: row.get("name"),
            aliases: serde_json::from_str(&aliases_str).unwrap_or_default(),
            relationship: row.try_get("relationship").unwrap_or(None),
            platform_ids: serde_json::from_str(&platform_ids_str).unwrap_or_default(),
            notes: row.try_get("notes").unwrap_or(None),
            communication_style: row.try_get("communication_style").unwrap_or(None),
            language_preference: row.try_get("language_preference").unwrap_or(None),
            last_interaction_at: last_interaction_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
            interaction_count: row.try_get("interaction_count").unwrap_or(0),
            created_at: parse_dt(created_str),
            updated_at: parse_dt(updated_str),
        }
    }

    fn row_to_person_fact(row: &sqlx::sqlite::SqliteRow) -> crate::traits::PersonFact {
        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");

        crate::traits::PersonFact {
            id: row.get("id"),
            person_id: row.get("person_id"),
            category: row.get("category"),
            key: row.get("key"),
            value: row.get("value"),
            source: row.get("source"),
            confidence: row.get("confidence"),
            created_at: parse_dt(created_str),
            updated_at: parse_dt(updated_str),
        }
    }
}

/// Resolve a "keychain:key_name" reference to the actual value from OS keychain.
/// Falls back to the raw string if keychain lookup fails or it's not a reference.
fn resolve_keychain_ref(value: &str) -> String {
    if let Some(key_name) = value.strip_prefix("keychain:") {
        match crate::config::resolve_from_keychain(key_name) {
            Ok(password) => return password,
            Err(_) => {
                tracing::warn!(
                    key = key_name,
                    "Failed to resolve keychain reference for dynamic bot"
                );
            }
        }
    }
    // Fall back to raw value (backward compat with pre-existing plaintext entries)
    value.to_string()
}

fn parse_dt(s: String) -> chrono::DateTime<Utc> {
    chrono::DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

/// Parse a date string and calculate days until it from today (wrapping year).
fn days_until_date(value: &str, today: chrono::NaiveDate) -> Option<i64> {
    use chrono::NaiveDate;

    let trimmed = value.trim();

    // Try "YYYY-MM-DD"
    if let Ok(d) = NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        let this_year = today.with_month(d.month())?.with_day(d.day())?;
        let diff = (this_year - today).num_days();
        return Some(if diff < 0 { diff + 365 } else { diff });
    }

    // Try "MM-DD" or "MM/DD"
    for fmt in &["%m-%d", "%m/%d"] {
        if let Ok(d) =
            NaiveDate::parse_from_str(&format!("2000-{}", trimmed), &format!("2000-{}", fmt))
        {
            let this_year = today.with_month(d.month())?.with_day(d.day())?;
            let diff = (this_year - today).num_days();
            return Some(if diff < 0 { diff + 365 } else { diff });
        }
    }

    // Try "Month DD" (e.g., "March 15")
    let months = [
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
    ];
    let lower = trimmed.to_lowercase();
    for (name, num) in &months {
        if let Some(rest) = lower.strip_prefix(name) {
            let rest = rest.trim().trim_start_matches([',', ' ']);
            if let Ok(day) = rest.parse::<u32>() {
                let this_year = today.with_month(*num)?.with_day(day)?;
                let diff = (this_year - today).num_days();
                return Some(if diff < 0 { diff + 365 } else { diff });
            }
        }
    }

    None
}

mod conversation_summary;
mod dynamic_bots;
mod dynamic_cli_agents;
mod dynamic_mcp;
mod episodes;
mod facts;
mod goals;
mod health_checks;
mod learning;
mod messages;
pub(crate) mod migrations;
mod notifications;
mod oauth;
mod people;
mod session_channels;
mod settings;
mod skills;
mod token_usage;

#[cfg(test)]
mod tests;
