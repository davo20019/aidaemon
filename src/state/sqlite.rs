use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use tokio::sync::RwLock;

use crate::traits::{
    BehaviorPattern, Episode, ErrorSolution, Expertise, Fact, Goal, Message, Procedure,
    StateStore, TokenUsage, TokenUsageRecord, UserProfile,
};

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

pub struct SqliteStateStore {
    pool: SqlitePool,
    working_memory: Arc<RwLock<HashMap<String, VecDeque<Message>>>>,
    cap: usize,
    embedding_service: Arc<EmbeddingService>,
}

impl SqliteStateStore {
    pub async fn new(db_path: &str, cap: usize, encryption_key: Option<&str>, embedding_service: Arc<EmbeddingService>) -> anyhow::Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await?;

        // Set restrictive file permissions (owner-only read/write)
        set_db_file_permissions(db_path);

        // SQLCipher: set encryption key if provided and the feature is enabled
        #[cfg(feature = "encryption")]
        if let Some(key) = encryption_key {
            if !key.is_empty() {
                // PRAGMA key must be the first statement on a new connection.
                // With a connection pool we set it here; sqlx runs it on the first acquired connection.
                sqlx::query(&format!("PRAGMA key = '{}'", key.replace('\'', "''")))
                    .execute(&pool)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to set SQLCipher encryption key: {}", e))?;
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

        // Create tables
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_name TEXT,
                tool_calls_json TEXT,
                created_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(category, key)
            )"
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)")
            .execute(&pool)
            .await?;

        // --- Migrations for Advanced Memory ---
        // 1. Add importance column
        let _ = sqlx::query("ALTER TABLE messages ADD COLUMN importance REAL DEFAULT 0.5")
            .execute(&pool)
            .await; // Ignore error if exists

        // 2. Add embedding column
        let _ = sqlx::query("ALTER TABLE messages ADD COLUMN embedding BLOB")
            .execute(&pool)
            .await; // Ignore error if exists

        // Add embedding_error column if it doesn't exist
        let _ = sqlx::query("ALTER TABLE messages ADD COLUMN embedding_error TEXT").execute(&pool).await;

        // 4. Add consolidated_at column for memory consolidation (Layer 6)
        let _ = sqlx::query("ALTER TABLE messages ADD COLUMN consolidated_at TEXT").execute(&pool).await;

        // --- Human-Like Memory System Migrations ---
        // 5. Add new columns to facts table for supersession and recall tracking
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN superseded_at TEXT").execute(&pool).await;
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN recall_count INTEGER DEFAULT 0").execute(&pool).await;
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN last_recalled_at TEXT").execute(&pool).await;

        // 6. Create episodes table (episodic memory)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                topics TEXT,
                emotional_tone TEXT,
                outcome TEXT,
                embedding BLOB,
                importance REAL DEFAULT 0.5,
                recall_count INTEGER DEFAULT 0,
                last_recalled_at TEXT,
                message_count INTEGER,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)")
            .execute(&pool)
            .await?;

        // 7. Create goals table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                priority TEXT DEFAULT 'medium',
                progress_notes TEXT,
                source_episode_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (source_episode_id) REFERENCES episodes(id)
            )"
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
            .execute(&pool)
            .await?;

        // 8. Create user_profile table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verbosity_preference TEXT DEFAULT 'medium',
                explanation_depth TEXT DEFAULT 'moderate',
                tone_preference TEXT DEFAULT 'neutral',
                emoji_preference TEXT DEFAULT 'none',
                typical_session_length INTEGER,
                active_hours TEXT,
                common_workflows TEXT,
                asks_before_acting INTEGER DEFAULT 1,
                prefers_explanations INTEGER DEFAULT 1,
                likes_suggestions INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        // 9. Create behavior_patterns table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS behavior_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                trigger_context TEXT,
                action TEXT,
                confidence REAL DEFAULT 0.5,
                occurrence_count INTEGER DEFAULT 1,
                last_seen_at TEXT,
                created_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        // 10. Create procedures table (procedural memory)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                trigger_pattern TEXT NOT NULL,
                trigger_embedding BLOB,
                steps TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                failure_count INTEGER DEFAULT 0,
                avg_duration_secs REAL,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        // 11. Create expertise table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS expertise (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL UNIQUE,
                tasks_attempted INTEGER DEFAULT 0,
                tasks_succeeded INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0,
                current_level TEXT DEFAULT 'novice',
                confidence_score REAL DEFAULT 0.0,
                common_errors TEXT,
                last_task_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        // 12. Create error_solutions table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS error_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_pattern TEXT NOT NULL,
                error_embedding BLOB,
                domain TEXT,
                solution_summary TEXT NOT NULL,
                solution_steps TEXT,
                success_count INTEGER DEFAULT 1,
                failure_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                created_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        // Terminal allowed prefixes (persisted "Allow Always" approvals)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS terminal_allowed_prefixes (
                prefix TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"
        )
        .execute(&pool)
        .await?;

        // Command patterns for learning command safety over time
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS command_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL UNIQUE,
                original_example TEXT NOT NULL,
                approval_count INTEGER DEFAULT 1,
                denial_count INTEGER DEFAULT 0,
                last_approved_at TEXT,
                last_denied_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"
        )
        .execute(&pool)
        .await?;

        // Scheduled tasks table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                cron_expr TEXT NOT NULL,
                original_schedule TEXT NOT NULL,
                prompt TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'tool',
                is_oneshot INTEGER NOT NULL DEFAULT 0,
                is_paused INTEGER NOT NULL DEFAULT 0,
                is_trusted INTEGER NOT NULL DEFAULT 0,
                last_run_at TEXT,
                next_run_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run
                ON scheduled_tasks(next_run_at) WHERE is_paused = 0"
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_tasks_name_source
                ON scheduled_tasks(name) WHERE source = 'config'"
        )
        .execute(&pool)
        .await?;

        // 3. Create macros table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS macros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_tool TEXT NOT NULL,
                trigger_args_pattern TEXT, 
                next_tool TEXT NOT NULL,
                next_args TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                used_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"
        )
        .execute(&pool)
        .await?;

        // Token usage tracking
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"
        )
        .execute(&pool)
        .await?;

        // Dynamic bots table - stores bot tokens added via /connect command
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS dynamic_bots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_type TEXT NOT NULL,
                bot_token TEXT NOT NULL,
                app_token TEXT,
                allowed_user_ids TEXT NOT NULL DEFAULT '[]',
                extra_config TEXT DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"
        )
        .execute(&pool)
        .await?;

        // Dynamic skills table - stores skills added via manage_skills tool
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS dynamic_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                triggers_json TEXT NOT NULL DEFAULT '[]',
                body TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'inline',
                source_url TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                version TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )"
        )
        .execute(&pool)
        .await?;

        Ok(Self {
            pool,
            working_memory: Arc::new(RwLock::new(HashMap::new())),
            cap,
            embedding_service,
        })
    }

    /// Hydrate working memory for a session from the database.
    async fn hydrate(&self, session_id: &str) -> anyhow::Result<VecDeque<Message>> {
        let rows = sqlx::query(
            "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at, importance
             FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?"
        )
        .bind(session_id)
        .bind(self.cap as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut deque = VecDeque::with_capacity(rows.len());
        for row in rows.into_iter().rev() {
            let created_str: String = row.get("created_at");
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let importance: f32 = row.try_get("importance").unwrap_or(0.5);

            deque.push_back(Message {
                id: row.get("id"),
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                tool_call_id: row.get("tool_call_id"),
                tool_name: row.get("tool_name"),
                tool_calls_json: row.get("tool_calls_json"),
                created_at,
                importance,
                embedding: None, // Don't load embeddings into working memory by default
            });
        }
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
        let topics_json = episode.topics.as_ref().map(|t| serde_json::to_string(t).unwrap_or_default());
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
    pub async fn get_relevant_episodes(&self, query: &str, limit: usize) -> anyhow::Result<Vec<Episode>> {
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
                if let Ok(vec) = serde_json::from_slice::<Vec<f32>>(&blob) {
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
                top_summary = scored.first().map(|(e, _)| truncate_str(&e.summary, 50)).unwrap_or_default(),
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
    pub async fn update_episode_embedding(&self, episode_id: i64, embedding: &[f32]) -> anyhow::Result<()> {
        let blob = serde_json::to_vec(embedding)?;
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
        let rows: Vec<sqlx::sqlite::SqliteRow> = sqlx::query(
            "SELECT id, summary FROM episodes WHERE embedding IS NULL"
        )
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
                    let blob = serde_json::to_vec(&embedding)?;
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

    fn row_to_episode(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Episode> {
        let topics_json: Option<String> = row.get("topics");
        let topics = topics_json.and_then(|j| serde_json::from_str(&j).ok());

        let start_str: String = row.get("start_time");
        let end_str: String = row.get("end_time");
        let created_str: String = row.get("created_at");
        let last_recalled_str: Option<String> = row.get("last_recalled_at");

        Ok(Episode {
            id: row.get("id"),
            session_id: row.get("session_id"),
            summary: row.get("summary"),
            topics,
            emotional_tone: row.get("emotional_tone"),
            outcome: row.get("outcome"),
            importance: row.get("importance"),
            recall_count: row.get("recall_count"),
            last_recalled_at: last_recalled_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
            message_count: row.get("message_count"),
            start_time: chrono::DateTime::parse_from_rfc3339(&start_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            end_time: chrono::DateTime::parse_from_rfc3339(&end_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
        })
    }

    // ==================== Goal Methods ====================

    /// Insert a new goal.
    #[allow(dead_code)]
    pub async fn insert_goal(&self, goal: &Goal) -> anyhow::Result<i64> {
        let progress_notes_json = goal.progress_notes.as_ref().map(|p| serde_json::to_string(p).unwrap_or_default());
        let result = sqlx::query(
            "INSERT INTO goals (description, status, priority, progress_notes, source_episode_id, created_at, updated_at, completed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&goal.description)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&progress_notes_json)
        .bind(goal.source_episode_id)
        .bind(goal.created_at.to_rfc3339())
        .bind(goal.updated_at.to_rfc3339())
        .bind(goal.completed_at.map(|t| t.to_rfc3339()))
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    /// Get active goals.
    pub async fn get_active_goals(&self) -> anyhow::Result<Vec<Goal>> {
        let rows = sqlx::query(
            "SELECT id, description, status, priority, progress_notes, source_episode_id, created_at, updated_at, completed_at
             FROM goals WHERE status = 'active' ORDER BY
             CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, created_at DESC
             LIMIT 10"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut goals = Vec::with_capacity(rows.len());
        for row in rows {
            goals.push(self.row_to_goal(&row)?);
        }
        Ok(goals)
    }

    /// Update goal status and optionally add a progress note.
    #[allow(dead_code)]
    pub async fn update_goal(&self, goal_id: i64, status: Option<&str>, progress_note: Option<&str>) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();

        if let Some(note) = progress_note {
            // Append progress note to existing notes
            let row = sqlx::query("SELECT progress_notes FROM goals WHERE id = ?")
                .bind(goal_id)
                .fetch_optional(&self.pool)
                .await?;

            let mut notes: Vec<String> = row
                .and_then(|r| r.get::<Option<String>, _>("progress_notes"))
                .and_then(|j| serde_json::from_str(&j).ok())
                .unwrap_or_default();
            notes.push(note.to_string());
            let notes_json = serde_json::to_string(&notes)?;

            sqlx::query("UPDATE goals SET progress_notes = ?, updated_at = ? WHERE id = ?")
                .bind(&notes_json)
                .bind(&now)
                .bind(goal_id)
                .execute(&self.pool)
                .await?;
        }

        if let Some(s) = status {
            let completed_at = if s == "completed" { Some(now.clone()) } else { None };
            sqlx::query("UPDATE goals SET status = ?, updated_at = ?, completed_at = COALESCE(?, completed_at) WHERE id = ?")
                .bind(s)
                .bind(&now)
                .bind(&completed_at)
                .bind(goal_id)
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    /// Find a goal similar to the given description using embeddings.
    #[allow(dead_code)]
    pub async fn find_similar_goal(&self, description: &str) -> anyhow::Result<Option<Goal>> {
        let goals = self.get_active_goals().await?;
        if goals.is_empty() {
            return Ok(None);
        }

        let query_vec = self.embedding_service.embed(description.to_string()).await?;
        let goal_texts: Vec<String> = goals.iter().map(|g| g.description.clone()).collect();
        let goal_embeddings = self.embedding_service.embed_batch(goal_texts).await?;

        let mut best_match: Option<(usize, f32)> = None;
        for (i, emb) in goal_embeddings.iter().enumerate() {
            let score = crate::memory::math::cosine_similarity(&query_vec, emb);
            if score > 0.75 {
                if best_match.is_none() || score > best_match.unwrap().1 {
                    best_match = Some((i, score));
                }
            }
        }

        Ok(best_match.map(|(i, _)| goals[i].clone()))
    }

    fn row_to_goal(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Goal> {
        let progress_notes_json: Option<String> = row.get("progress_notes");
        let progress_notes = progress_notes_json.and_then(|j| serde_json::from_str(&j).ok());

        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");
        let completed_str: Option<String> = row.get("completed_at");

        Ok(Goal {
            id: row.get("id"),
            description: row.get("description"),
            status: row.get("status"),
            priority: row.get("priority"),
            progress_notes,
            source_episode_id: row.get("source_episode_id"),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            completed_at: completed_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
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
                updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
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
        let active_hours_json = profile.active_hours.as_ref().map(|h| serde_json::to_string(h).unwrap_or_default());
        let workflows_json = profile.common_workflows.as_ref().map(|w| serde_json::to_string(w).unwrap_or_default());
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
    pub async fn get_behavior_patterns(&self, min_confidence: f32) -> anyhow::Result<Vec<BehaviorPattern>> {
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
                last_seen_at: last_seen_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
                created_at: chrono::DateTime::parse_from_rfc3339(&created_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            });
        }
        Ok(patterns)
    }

    /// Update pattern occurrence and confidence.
    #[allow(dead_code)]
    pub async fn update_behavior_pattern(&self, pattern_id: i64, confidence_delta: f32) -> anyhow::Result<()> {
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
        let result = sqlx::query(
            "INSERT INTO procedures (name, trigger_pattern, trigger_embedding, steps, success_count, failure_count, avg_duration_secs, last_used_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(name) DO UPDATE SET steps = excluded.steps, success_count = success_count + 1, updated_at = excluded.updated_at"
        )
        .bind(&procedure.name)
        .bind(&procedure.trigger_pattern)
        .bind::<Option<Vec<u8>>>(None)
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
    pub async fn get_relevant_procedures(&self, query: &str, limit: usize) -> anyhow::Result<Vec<Procedure>> {
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
                if let Ok(vec) = serde_json::from_slice::<Vec<f32>>(&blob) {
                    crate::memory::math::cosine_similarity(&query_vec, &vec)
                } else {
                    0.0
                }
            } else {
                // Fall back to text matching
                if procedure.trigger_pattern.to_lowercase().contains(&query.to_lowercase()) {
                    0.5
                } else {
                    0.0
                }
            };

            if score > 0.3 {
                scored.push((procedure, score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(p, _)| p).collect())
    }

    /// Update procedure success/failure and optionally update steps.
    #[allow(dead_code)] // Wired to trait but reserved for feedback loop
    pub async fn update_procedure(&self, procedure_id: i64, success: bool, new_steps: Option<&[String]>, duration_secs: Option<f32>) -> anyhow::Result<()> {
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
            last_used_at: last_used_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
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
             LIMIT 10"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Increment expertise counters and update level.
    pub async fn increment_expertise(&self, domain: &str, success: bool, error: Option<&str>) -> anyhow::Result<()> {
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
            let mut errors: Vec<String> = errors_json.and_then(|j| serde_json::from_str(&j).ok()).unwrap_or_default();
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
            last_task_at: last_task_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
        })
    }

    // ==================== ErrorSolution Methods ====================

    /// Insert a new error solution.
    pub async fn insert_error_solution(&self, solution: &ErrorSolution) -> anyhow::Result<i64> {
        let steps_json = solution.solution_steps.as_ref().map(|s| serde_json::to_string(s).unwrap_or_default());
        let result = sqlx::query(
            "INSERT INTO error_solutions (error_pattern, error_embedding, domain, solution_summary, solution_steps, success_count, failure_count, last_used_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&solution.error_pattern)
        .bind::<Option<Vec<u8>>>(None)
        .bind(&solution.domain)
        .bind(&solution.solution_summary)
        .bind(&steps_json)
        .bind(solution.success_count)
        .bind(solution.failure_count)
        .bind(solution.last_used_at.map(|t| t.to_rfc3339()))
        .bind(solution.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    /// Get error solutions relevant to an error message.
    pub async fn get_relevant_error_solutions(&self, error: &str, limit: usize) -> anyhow::Result<Vec<ErrorSolution>> {
        let rows = sqlx::query(
            "SELECT id, error_pattern, domain, solution_summary, solution_steps, success_count, failure_count, last_used_at, created_at, error_embedding
             FROM error_solutions WHERE success_count > failure_count ORDER BY success_count DESC LIMIT 100"
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
                if let Ok(vec) = serde_json::from_slice::<Vec<f32>>(&blob) {
                    crate::memory::math::cosine_similarity(&query_vec, &vec)
                } else {
                    0.0
                }
            } else {
                // Fall back to text matching
                if solution.error_pattern.to_lowercase().contains(&error.to_lowercase()) {
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
    pub async fn update_error_solution(&self, solution_id: i64, success: bool) -> anyhow::Result<()> {
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

    fn row_to_error_solution(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<ErrorSolution> {
        let steps_json: Option<String> = row.get("solution_steps");
        let solution_steps = steps_json.and_then(|j| serde_json::from_str(&j).ok());
        let created_str: String = row.get("created_at");
        let last_used_str: Option<String> = row.get("last_used_at");

        Ok(ErrorSolution {
            id: row.get("id"),
            error_pattern: row.get("error_pattern"),
            domain: row.get("domain"),
            solution_summary: row.get("solution_summary"),
            solution_steps,
            success_count: row.get("success_count"),
            failure_count: row.get("failure_count"),
            last_used_at: last_used_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
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
        sqlx::query("UPDATE facts SET recall_count = recall_count + 1, last_recalled_at = ? WHERE id = ?")
            .bind(&now)
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    #[allow(dead_code)]
    fn row_to_fact_with_history(&self, row: &sqlx::sqlite::SqliteRow) -> anyhow::Result<Fact> {
        let created_str: String = row.get("created_at");
        let updated_str: String = row.get("updated_at");
        let superseded_str: Option<String> = row.get("superseded_at");
        let last_recalled_str: Option<String> = row.get("last_recalled_at");

        Ok(Fact {
            id: row.get("id"),
            category: row.get("category"),
            key: row.get("key"),
            value: row.get("value"),
            source: row.get("source"),
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str).map(|dt| dt.with_timezone(&Utc)).unwrap_or_else(|_| Utc::now()),
            superseded_at: superseded_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
            recall_count: row.try_get("recall_count").unwrap_or(0),
            last_recalled_at: last_recalled_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
        })
    }
}

#[async_trait]
impl StateStore for SqliteStateStore {
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()> {
        // Persist to DB
        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at, importance, embedding)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&msg.id)
        .bind(&msg.session_id)
        .bind(&msg.role)
        .bind(&msg.content)
        .bind(&msg.tool_call_id)
        .bind(&msg.tool_name)
        .bind(&msg.tool_calls_json)
        .bind(msg.created_at.to_rfc3339())
        .bind(msg.importance)
        .bind(msg.embedding.as_ref().map(|v| {
            // Convert Vec<f32> to Vec<u8> (naive)
            // For now, let's just use JSON for safety/portability if we aren't using sqlite-vec yet
            serde_json::to_vec(v).unwrap_or_default()
        }))
        .execute(&self.pool)
        .await?;

        // Update working memory
        let mut wm = self.working_memory.write().await;
        let deque = wm.entry(msg.session_id.clone()).or_insert_with(VecDeque::new);
        deque.push_back(msg.clone());

        // Evict old messages but ALWAYS preserve the first user message (anchor)
        // This is critical for Gemini which requires tool_calls to follow user/tool messages
        let mut evicted = 0;
        while deque.len() > self.cap {
            // Find the first user message index
            let anchor_idx = deque.iter().position(|m| m.role == "user");

            if anchor_idx == Some(0) && deque.len() > 1 {
                // Anchor is at front - evict the second message instead
                deque.remove(1);
            } else {
                // Safe to evict from front
                deque.pop_front();
            }
            evicted += 1;
        }

        tracing::debug!(
            session_id = %msg.session_id,
            role = %msg.role,
            msg_id = %msg.id,
            deque_len = deque.len(),
            cap = self.cap,
            evicted,
            "append_message: added to working memory"
        );

        Ok(())
    }

    async fn get_history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>> {
        // Helper to truncate while preserving anchor user message
        fn truncate_with_anchor(messages: &[Message], limit: usize) -> Vec<Message> {
            if messages.len() <= limit {
                return messages.to_vec();
            }

            // Find the first user message (anchor)
            let anchor_idx = messages.iter().position(|m| m.role == "user");

            let skip = messages.len().saturating_sub(limit);

            if let Some(anchor) = anchor_idx {
                if skip > anchor {
                    // We would skip past the anchor - preserve it
                    let anchor_msg = messages[anchor].clone();
                    let remaining: Vec<_> = messages.iter().skip(skip).cloned().collect();

                    // Only prepend anchor if not already in remaining
                    if remaining.first().map(|m| m.role.as_str()) != Some("user") {
                        let mut result = vec![anchor_msg];
                        result.extend(remaining.into_iter().take(limit - 1));
                        return result;
                    }
                    return remaining;
                }
            }

            // Normal case - just take last N
            messages.iter().skip(skip).cloned().collect()
        }

        // Check working memory first
        {
            let wm = self.working_memory.read().await;
            tracing::debug!(
                session_id,
                wm_sessions = wm.len(),
                has_session = wm.contains_key(session_id),
                "get_history: checking working memory"
            );
            if let Some(deque) = wm.get(session_id) {
                let roles: Vec<&str> = deque.iter().map(|m| m.role.as_str()).collect();
                tracing::debug!(
                    session_id,
                    deque_len = deque.len(),
                    roles = ?roles,
                    "get_history: found session in working memory"
                );
                if !deque.is_empty() {
                    let msgs: Vec<_> = deque.iter().cloned().collect();
                    let result = truncate_with_anchor(&msgs, limit);
                    tracing::debug!(
                        session_id,
                        before_truncate = msgs.len(),
                        after_truncate = result.len(),
                        "get_history: returning from working memory"
                    );
                    return Ok(result);
                }
            }
        }

        // Cold start: hydrate from DB
        tracing::debug!(session_id, "get_history: cold start, hydrating from DB");
        let deque = self.hydrate(session_id).await?;
        let msgs: Vec<_> = deque.iter().cloned().collect();
        let result = truncate_with_anchor(&msgs, limit);
        tracing::debug!(
            session_id,
            hydrated_count = deque.len(),
            result_count = result.len(),
            "get_history: hydrated from DB"
        );

        // Cache in working memory
        let mut wm = self.working_memory.write().await;
        wm.insert(session_id.to_string(), deque);

        Ok(result)
    }

    async fn upsert_fact(&self, category: &str, key: &str, value: &str, source: &str) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();

        // Find existing current fact (not superseded)
        let existing = sqlx::query(
            "SELECT id, value FROM facts WHERE category = ? AND key = ? AND superseded_at IS NULL"
        )
        .bind(category)
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = existing {
            let old_value: String = row.get("value");
            let old_id: i64 = row.get("id");

            // If the value is different, mark old as superseded and insert new
            if old_value != value {
                sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
                    .bind(&now)
                    .bind(old_id)
                    .execute(&self.pool)
                    .await?;

                // Insert new fact
                sqlx::query(
                    "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count)
                     VALUES (?, ?, ?, ?, ?, ?, 0)"
                )
                .bind(category)
                .bind(key)
                .bind(value)
                .bind(source)
                .bind(&now)
                .bind(&now)
                .execute(&self.pool)
                .await?;
            } else {
                // Same value - just update the timestamp and source
                sqlx::query("UPDATE facts SET source = ?, updated_at = ? WHERE id = ?")
                    .bind(source)
                    .bind(&now)
                    .bind(old_id)
                    .execute(&self.pool)
                    .await?;
            }
        } else {
            // No existing fact - insert new
            sqlx::query(
                "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count)
                 VALUES (?, ?, ?, ?, ?, ?, 0)"
            )
            .bind(category)
            .bind(key)
            .bind(value)
            .bind(source)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;
        }
        Ok(())
    }

    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
        // Only return current (non-superseded) facts
        let rows = if let Some(cat) = category {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at FROM facts WHERE category = ? AND superseded_at IS NULL ORDER BY updated_at DESC")
                .bind(cat)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC")
                .fetch_all(&self.pool)
                .await?
        };

        let mut facts = Vec::with_capacity(rows.len());
        for row in rows {
            let created_str: String = row.get("created_at");
            let updated_str: String = row.get("updated_at");
            let superseded_str: Option<String> = row.get("superseded_at");
            let last_recalled_str: Option<String> = row.get("last_recalled_at");

            let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            facts.push(Fact {
                id: row.get("id"),
                category: row.get("category"),
                key: row.get("key"),
                value: row.get("value"),
                source: row.get("source"),
                created_at,
                updated_at,
                superseded_at: superseded_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
                recall_count: row.try_get("recall_count").unwrap_or(0),
                last_recalled_at: last_recalled_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
            });
        }
        Ok(facts)
    }

    async fn get_relevant_facts(&self, query: &str, max: usize) -> anyhow::Result<Vec<Fact>> {
        let all_facts = self.get_facts(None).await?;
        if all_facts.is_empty() || query.trim().is_empty() {
            let mut facts = all_facts;
            facts.truncate(max);
            return Ok(facts);
        }

        // Embed the query
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Failed to embed query for fact filtering, returning all facts: {}", e);
                let mut facts = all_facts;
                facts.truncate(max);
                return Ok(facts);
            }
        };

        // Embed each fact's combined text and score by similarity
        let fact_texts: Vec<String> = all_facts
            .iter()
            .map(|f| format!("[{}] {}: {}", f.category, f.key, f.value))
            .collect();

        let fact_embeddings = match self.embedding_service.embed_batch(fact_texts).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Failed to embed facts for filtering, returning all facts: {}", e);
                let mut facts = all_facts;
                facts.truncate(max);
                return Ok(facts);
            }
        };

        // Score and sort by relevance
        let mut scored: Vec<(usize, f32)> = fact_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, crate::memory::math::cosine_similarity(&query_vec, emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top `max` facts that are above a minimum relevance threshold (0.3)
        // Low threshold because we still want loosely related facts, just not completely irrelevant ones
        let relevant: Vec<Fact> = scored
            .into_iter()
            .filter(|(_, score)| *score > 0.3)
            .take(max)
            .map(|(i, _)| all_facts[i].clone())
            .collect();

        // If filtering left us with very few facts, pad with most recent ones
        if relevant.len() < max / 2 && all_facts.len() > relevant.len() {
            let mut result = relevant;
            let existing_ids: std::collections::HashSet<i64> = result.iter().map(|f| f.id).collect();
            for fact in &all_facts {
                if result.len() >= max {
                    break;
                }
                if !existing_ids.contains(&fact.id) {
                    result.push(fact.clone());
                }
            }
            return Ok(result);
        }

        Ok(relevant)
    }

    async fn get_context(&self, session_id: &str, query: &str, _limit: usize) -> anyhow::Result<Vec<Message>> {
        // 1. Recency (Last 10 conversation messages) - Critical for conversational flow
        // Only get user messages and final assistant responses (not tool calls/results)
        // This prevents intermediate tool messages from pushing out important context
        let recency_count = 10;
        let recency_rows = sqlx::query(
            "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at, importance
             FROM messages
             WHERE session_id = ?
               AND (role = 'user' OR (role = 'assistant' AND tool_calls_json IS NULL))
             ORDER BY created_at DESC
             LIMIT ?"
        )
        .bind(session_id)
        .bind(recency_count)
        .fetch_all(&self.pool)
        .await?;

        let mut messages: Vec<Message> = recency_rows.into_iter().map(|row| {
            let created_str: String = row.get("created_at");
            let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            Message {
                id: row.get("id"),
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                tool_call_id: row.get("tool_call_id"),
                tool_name: row.get("tool_name"),
                tool_calls_json: row.get("tool_calls_json"),
                created_at,
                importance: row.try_get("importance").unwrap_or(0.5),
                embedding: None,
            }
        }).collect();

        let mut seen_ids: std::collections::HashSet<String> = messages.iter().map(|m| m.id.clone()).collect();

        // 2. Salience (Importance >= 0.8) - Critical memory
        // Added session_id filter to prevent bleeding context from other sessions
        let limit_salience = 5;
        let salience = sqlx::query(
            "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at, importance
             FROM messages 
             WHERE session_id = ? AND importance >= 0.8 
             ORDER BY created_at DESC 
             LIMIT ?"
        )
        .bind(session_id)
        .bind(limit_salience)
        .fetch_all(&self.pool)
        .await?;
        
        for row in salience {
             let id: String = row.get("id");
             if seen_ids.contains(&id) { continue; }
             
             let created_str: String = row.get("created_at");
             let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
             
             messages.push(Message {
                id: id.clone(),
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                tool_call_id: row.get("tool_call_id"),
                tool_name: row.get("tool_name"),
                tool_calls_json: row.get("tool_calls_json"),
                created_at,
                importance: row.try_get("importance").unwrap_or(0.5),
                embedding: None,
             });
             seen_ids.insert(id);
        }

        // 3. Relevance (Vector Search)
        // Only run if we have a query
        if !query.trim().is_empty() {
             if let Ok(query_vec) = self.embedding_service.embed(query.to_string()).await {
                 // Fetch candidate embeddings (limit to recent 2000 to save compute)
                 // Added session_id filter
                 let candidates = sqlx::query(
                    "SELECT id, embedding FROM messages WHERE session_id = ? AND embedding IS NOT NULL ORDER BY created_at DESC LIMIT 2000"
                 )
                 .bind(session_id)
                 .fetch_all(&self.pool)
                 .await?;

                 let mut scored = Vec::new();
                 for row in candidates {
                    let id: String = row.get("id");
                    if seen_ids.contains(&id) { continue; }
                    let embedding: Option<Vec<u8>> = row.get("embedding");
                    
                    if let Some(blob) = embedding {
                        if let Ok(vec) = serde_json::from_slice::<Vec<f32>>(&blob) {
                            let score = crate::memory::math::cosine_similarity(&query_vec, &vec);
                            if score > 0.65 { // Relevance threshold
                                scored.push((id, score));
                            }
                        }
                    }
                 }
                 // Sort by score DESC
                 scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                 
                 // Take top 5
                 for (id, _score) in scored.into_iter().take(5) {
                     // Fetch full message
                     let row = sqlx::query(
                        "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at, importance
                         FROM messages WHERE id = ?"
                     )
                     .bind(&id)
                     .fetch_optional(&self.pool)
                     .await?;

                     if let Some(row) = row {
                         let created_str: String = row.get("created_at");
                         let created_at = chrono::DateTime::parse_from_rfc3339(&created_str)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now());

                         let msg = Message {
                            id: row.get("id"),
                            session_id: row.get("session_id"),
                            role: row.get("role"),
                            content: row.get("content"),
                            tool_call_id: row.get("tool_call_id"),
                            tool_name: row.get("tool_name"),
                            tool_calls_json: row.get("tool_calls_json"),
                            created_at,
                            importance: row.try_get("importance").unwrap_or(0.5),
                            embedding: None,
                         };
                         // (Optional) Append score to content for debugging?
                         // if let Some(c) = &mut msg.content {
                         //    *c = format!("(Similarity: {:.2}) {}", score, c);
                         // }
                         messages.push(msg);
                         seen_ids.insert(id);
                     }
                 }
             }
        }

        // Sort final list by created_at (Chronological)
        messages.sort_by_key(|m| m.created_at);

        // Enforce limit if we somehow exceeded it (we might have up to 20 now)
        // But we want to keep *all* relevant context we found, so maybe flexible limit.
        // We just return what we gathered.
        Ok(messages)
    }

    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        // Clear working memory
        {
            let mut wm = self.working_memory.write().await;
            wm.remove(session_id);
        }
        // Delete messages from DB
        sqlx::query("DELETE FROM messages WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn record_token_usage(&self, session_id: &str, usage: &TokenUsage) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, created_at)
             VALUES (?, ?, ?, ?, datetime('now'))"
        )
        .bind(session_id)
        .bind(&usage.model)
        .bind(usage.input_tokens as i64)
        .bind(usage.output_tokens as i64)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_token_usage_since(&self, since: &str) -> anyhow::Result<Vec<TokenUsageRecord>> {
        let rows = sqlx::query(
            "SELECT model, input_tokens, output_tokens, created_at
             FROM token_usage WHERE created_at >= ? ORDER BY created_at DESC"
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            records.push(TokenUsageRecord {
                model: row.get("model"),
                input_tokens: row.get("input_tokens"),
                output_tokens: row.get("output_tokens"),
                created_at: row.get("created_at"),
            });
        }
        Ok(records)
    }

    // ==================== Extended Memory Trait Methods ====================

    async fn get_relevant_episodes(&self, query: &str, limit: usize) -> anyhow::Result<Vec<Episode>> {
        // Delegate to inherent method
        SqliteStateStore::get_relevant_episodes(self, query, limit).await
    }

    async fn get_active_goals(&self) -> anyhow::Result<Vec<Goal>> {
        SqliteStateStore::get_active_goals(self).await
    }

    async fn get_behavior_patterns(&self, min_confidence: f32) -> anyhow::Result<Vec<BehaviorPattern>> {
        SqliteStateStore::get_behavior_patterns(self, min_confidence).await
    }

    async fn get_relevant_procedures(&self, query: &str, limit: usize) -> anyhow::Result<Vec<Procedure>> {
        SqliteStateStore::get_relevant_procedures(self, query, limit).await
    }

    async fn get_relevant_error_solutions(&self, error: &str, limit: usize) -> anyhow::Result<Vec<ErrorSolution>> {
        SqliteStateStore::get_relevant_error_solutions(self, error, limit).await
    }

    async fn get_all_expertise(&self) -> anyhow::Result<Vec<Expertise>> {
        SqliteStateStore::get_all_expertise(self).await
    }

    async fn get_user_profile(&self) -> anyhow::Result<Option<UserProfile>> {
        Ok(Some(SqliteStateStore::get_user_profile(self).await?))
    }

    async fn get_trusted_command_patterns(&self) -> anyhow::Result<Vec<(String, i32)>> {
        SqliteStateStore::get_trusted_command_patterns(self).await
    }

    // ==================== Write Methods for Learning System ====================

    async fn increment_expertise(&self, domain: &str, success: bool, error: Option<&str>) -> anyhow::Result<()> {
        SqliteStateStore::increment_expertise(self, domain, success, error).await
    }

    async fn upsert_procedure(&self, procedure: &Procedure) -> anyhow::Result<i64> {
        SqliteStateStore::insert_procedure(self, procedure).await
    }

    async fn update_procedure_outcome(&self, procedure_id: i64, success: bool, duration: Option<f32>) -> anyhow::Result<()> {
        SqliteStateStore::update_procedure(self, procedure_id, success, None, duration).await
    }

    async fn insert_error_solution(&self, solution: &ErrorSolution) -> anyhow::Result<i64> {
        SqliteStateStore::insert_error_solution(self, solution).await
    }

    async fn update_error_solution_outcome(&self, solution_id: i64, success: bool) -> anyhow::Result<()> {
        SqliteStateStore::update_error_solution(self, solution_id, success).await
    }

    // ==================== Dynamic Bots Methods ====================

    async fn add_dynamic_bot(&self, bot: &crate::traits::DynamicBot) -> anyhow::Result<i64> {
        let allowed_user_ids_json = serde_json::to_string(&bot.allowed_user_ids)?;
        let result = sqlx::query(
            "INSERT INTO dynamic_bots (channel_type, bot_token, app_token, allowed_user_ids, extra_config, created_at)
             VALUES (?, ?, ?, ?, ?, datetime('now'))"
        )
        .bind(&bot.channel_type)
        .bind(&bot.bot_token)
        .bind(&bot.app_token)
        .bind(&allowed_user_ids_json)
        .bind(&bot.extra_config)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn get_dynamic_bots(&self) -> anyhow::Result<Vec<crate::traits::DynamicBot>> {
        let rows = sqlx::query(
            "SELECT id, channel_type, bot_token, app_token, allowed_user_ids, extra_config, created_at
             FROM dynamic_bots ORDER BY created_at ASC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut bots = Vec::with_capacity(rows.len());
        for row in rows {
            let allowed_user_ids_json: String = row.get("allowed_user_ids");
            let allowed_user_ids: Vec<String> = serde_json::from_str(&allowed_user_ids_json).unwrap_or_default();

            bots.push(crate::traits::DynamicBot {
                id: row.get("id"),
                channel_type: row.get("channel_type"),
                bot_token: row.get("bot_token"),
                app_token: row.get("app_token"),
                allowed_user_ids,
                extra_config: row.get("extra_config"),
                created_at: row.get("created_at"),
            });
        }
        Ok(bots)
    }

    async fn delete_dynamic_bot(&self, id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dynamic_bots WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ==================== Dynamic Skills ====================

    async fn add_dynamic_skill(&self, skill: &crate::traits::DynamicSkill) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO dynamic_skills (name, description, triggers_json, body, source, source_url, enabled, version, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))"
        )
        .bind(&skill.name)
        .bind(&skill.description)
        .bind(&skill.triggers_json)
        .bind(&skill.body)
        .bind(&skill.source)
        .bind(&skill.source_url)
        .bind(skill.enabled)
        .bind(&skill.version)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn get_dynamic_skills(&self) -> anyhow::Result<Vec<crate::traits::DynamicSkill>> {
        let rows = sqlx::query(
            "SELECT id, name, description, triggers_json, body, source, source_url, enabled, version, created_at
             FROM dynamic_skills ORDER BY created_at ASC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut skills = Vec::new();
        for row in rows {
            skills.push(crate::traits::DynamicSkill {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                description: row.get::<String, _>("description"),
                triggers_json: row.get::<String, _>("triggers_json"),
                body: row.get::<String, _>("body"),
                source: row.get::<String, _>("source"),
                source_url: row.get::<Option<String>, _>("source_url"),
                enabled: row.get::<bool, _>("enabled"),
                version: row.get::<Option<String>, _>("version"),
                created_at: row.get::<String, _>("created_at"),
            });
        }
        Ok(skills)
    }

    async fn delete_dynamic_skill(&self, id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dynamic_skills WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_dynamic_skill_enabled(&self, id: i64, enabled: bool) -> anyhow::Result<()> {
        sqlx::query("UPDATE dynamic_skills SET enabled = ? WHERE id = ?")
            .bind(enabled)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_promotable_procedures(&self, min_success: i32, min_rate: f32) -> anyhow::Result<Vec<crate::traits::Procedure>> {
        let rows = sqlx::query(
            "SELECT id, name, trigger_pattern, steps, success_count, failure_count,
                    avg_duration_secs, last_used_at, created_at, updated_at
             FROM procedures
             WHERE success_count >= ?
               AND CAST(success_count AS REAL) / CAST(success_count + failure_count AS REAL) >= ?
             ORDER BY success_count DESC"
        )
        .bind(min_success)
        .bind(min_rate)
        .fetch_all(&self.pool)
        .await?;

        let mut procedures = Vec::new();
        for row in rows {
            let steps_json: String = row.get("steps");
            let steps: Vec<String> = serde_json::from_str(&steps_json).unwrap_or_default();
            procedures.push(crate::traits::Procedure {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                trigger_pattern: row.get::<String, _>("trigger_pattern"),
                steps,
                success_count: row.get::<i32, _>("success_count"),
                failure_count: row.get::<i32, _>("failure_count"),
                avg_duration_secs: row.get::<Option<f32>, _>("avg_duration_secs"),
                last_used_at: row.get::<Option<String>, _>("last_used_at")
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                        chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S").ok()
                            .map(|n| n.and_utc().into())
                    }))
                    .map(|d| d.with_timezone(&chrono::Utc)),
                created_at: row.get::<Option<String>, _>("created_at")
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                        chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S").ok()
                            .map(|n| n.and_utc().into())
                    }))
                    .map(|d| d.with_timezone(&chrono::Utc))
                    .unwrap_or_else(chrono::Utc::now),
                updated_at: row.get::<Option<String>, _>("updated_at")
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                        chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S").ok()
                            .map(|n| n.and_utc().into())
                    }))
                    .map(|d| d.with_timezone(&chrono::Utc))
                    .unwrap_or_else(chrono::Utc::now),
            });
        }
        Ok(procedures)
    }
}
