use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use tokio::sync::RwLock;

use crate::traits::{Fact, Message, StateStore};

use crate::memory::embeddings::EmbeddingService;

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

        // Terminal allowed prefixes (persisted "Allow Always" approvals)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS terminal_allowed_prefixes (
                prefix TEXT PRIMARY KEY,
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
        while deque.len() > self.cap {
            deque.pop_front();
        }

        Ok(())
    }

    async fn get_history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>> {
        // Check working memory first
        {
            let wm = self.working_memory.read().await;
            if let Some(deque) = wm.get(session_id) {
                if !deque.is_empty() {
                    let skip = deque.len().saturating_sub(limit);
                    return Ok(deque.iter().skip(skip).cloned().collect());
                }
            }
        }

        // Cold start: hydrate from DB
        let deque = self.hydrate(session_id).await?;
        let result: Vec<Message> = {
            let skip = deque.len().saturating_sub(limit);
            deque.iter().skip(skip).cloned().collect()
        };

        // Cache in working memory
        let mut wm = self.working_memory.write().await;
        wm.insert(session_id.to_string(), deque);

        Ok(result)
    }

    async fn upsert_fact(&self, category: &str, key: &str, value: &str, source: &str) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO facts (category, key, value, source, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?)
             ON CONFLICT(category, key) DO UPDATE SET value = excluded.value, source = excluded.source, updated_at = excluded.updated_at"
        )
        .bind(category)
        .bind(key)
        .bind(value)
        .bind(source)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
        let rows = if let Some(cat) = category {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at FROM facts WHERE category = ? ORDER BY updated_at DESC")
                .bind(cat)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at FROM facts ORDER BY updated_at DESC")
                .fetch_all(&self.pool)
                .await?
        };

        let mut facts = Vec::with_capacity(rows.len());
        for row in rows {
            let created_str: String = row.get("created_at");
            let updated_str: String = row.get("updated_at");
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
            });
        }
        Ok(facts)
    }

    async fn get_context(&self, session_id: &str, query: &str, _limit: usize) -> anyhow::Result<Vec<Message>> {
        // 1. Recency (Last 10) - Critical for conversational flow
        let recency_count = 10;
        let mut messages = self.get_history(session_id, recency_count).await?;
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
}
