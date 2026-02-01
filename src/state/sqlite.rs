use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use tokio::sync::RwLock;

use crate::traits::{Fact, Message, StateStore};

pub struct SqliteStateStore {
    pool: SqlitePool,
    /// In-memory working memory: session_id -> recent messages.
    working_memory: Arc<RwLock<HashMap<String, VecDeque<Message>>>>,
    cap: usize,
}

impl SqliteStateStore {
    pub async fn new(db_path: &str, cap: usize) -> anyhow::Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await?;

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

        Ok(Self {
            pool,
            working_memory: Arc::new(RwLock::new(HashMap::new())),
            cap,
        })
    }

    /// Hydrate working memory for a session from the database.
    async fn hydrate(&self, session_id: &str) -> anyhow::Result<VecDeque<Message>> {
        let rows = sqlx::query(
            "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at
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

            deque.push_back(Message {
                id: row.get("id"),
                session_id: row.get("session_id"),
                role: row.get("role"),
                content: row.get("content"),
                tool_call_id: row.get("tool_call_id"),
                tool_name: row.get("tool_name"),
                tool_calls_json: row.get("tool_calls_json"),
                created_at,
            });
        }
        Ok(deque)
    }
}

#[async_trait]
impl StateStore for SqliteStateStore {
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()> {
        // Persist to DB
        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&msg.id)
        .bind(&msg.session_id)
        .bind(&msg.role)
        .bind(&msg.content)
        .bind(&msg.tool_call_id)
        .bind(&msg.tool_name)
        .bind(&msg.tool_calls_json)
        .bind(msg.created_at.to_rfc3339())
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
}
