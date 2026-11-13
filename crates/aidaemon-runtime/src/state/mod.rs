//! The daemon's state database: one SQLite file behind every store slice of
//! the `StateStore` facade (`crate::stores`). The record types and traits
//! come from the crates that own them; this is their SQLite end, plus the
//! state schema (`migrations`).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{Row, SqlitePool};
use tokio::sync::RwLock;

#[cfg(test)]
use types::Task;
use types::{ConversationSummary, Message, TokenUsage, TokenUsageRecord};

/// The store's record types and traits, from the crates that own them.
mod types {
    #[cfg(test)]
    pub(crate) use crate::stores::prelude as store_prelude;
    pub(crate) use crate::stores::*;
    pub(crate) use aidaemon_channels::hub::DynamicBot;
    pub(crate) use aidaemon_events::conversation::{
        infer_message_annotations, ConversationSummary, Message, MessageAnnotation,
    };
    pub(crate) use aidaemon_events::dialogue::DialogueState;
    pub(crate) use aidaemon_events::provider::{TokenUsage, TokenUsageRecord};
    pub(crate) use aidaemon_mcp::registry::DynamicMcpServer;
    #[cfg(test)]
    pub(crate) use aidaemon_memory::records::{
        Episode, ExtractedMemoryEntity, ExtractedMemoryGraph, ExtractedMemoryRelationship,
        FactPrivacy,
    };
    pub(crate) use aidaemon_memory::records::{
        HistoryCoverage, HistoryMessage, HistoryScope, HistoryToolActivity, ProjectionStats,
        RecentToolTurn, TaskBookends,
    };
    pub(crate) use aidaemon_memory::skills::{DynamicSkill, SkillDraft};
    pub(crate) use aidaemon_memory::Procedure;
    #[cfg(test)]
    pub(crate) use aidaemon_memory::{BehaviorPattern, ErrorSolution};
    pub(crate) use aidaemon_tools::services::{
        CliAgentInvocation, DynamicCliAgent, OAuthConnection, PendingOAuthFlow,
    };
    #[cfg(test)]
    pub(crate) use aidaemon_work::goals::{
        Goal, GoalRun, GoalSchedule, NotificationEntry, Task, TaskActivity, TaskAttemptPatch,
    };
    #[cfg(test)]
    pub(crate) use aidaemon_work::mandates::{Mandate, MandateAuthority};
}

impl aidaemon_work::sqlite::WorkDatabase for SqliteStateStore {
    fn work_pool(&self) -> &SqlitePool {
        &self.pool
    }

    fn redact_secrets(&self, text: &str) -> String {
        aidaemon_tools::sanitize::redact_secrets(text)
    }
}

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
    memory: aidaemon_memory::sqlite::SqliteMemory,
    cap: usize,
    /// Where dynamic bots' tokens go instead of the database. Without one
    /// they stay in their rows (as in tests, which never open the OS store).
    secrets: Option<Arc<dyn aidaemon_secrets::SecretStore>>,
    #[cfg(test)]
    embedding_service: Arc<dyn aidaemon_memory::Embedder>,
}

impl SqliteStateStore {
    /// The daemon's own memory, over this database.
    pub fn memory(&self) -> &aidaemon_memory::sqlite::SqliteMemory {
        &self.memory
    }

    /// The same memory as the contract its users depend on.
    pub fn memory_stores(&self) -> Arc<dyn aidaemon_memory::stores::MemoryStores> {
        Arc::new(self.memory.clone())
    }
}

/// Lets tests seed and read memory on the store they already hold.
#[cfg(any(test, feature = "testing"))]
impl std::ops::Deref for SqliteStateStore {
    type Target = aidaemon_memory::sqlite::SqliteMemory;

    fn deref(&self) -> &Self::Target {
        &self.memory
    }
}

impl SqliteStateStore {
    /// Opens (or creates) the database at `db_path` and brings its schema
    /// up to date. `embeddings` embeds and reranks long-term memory.
    pub async fn new<E>(
        db_path: &str,
        cap: usize,
        encryption_key: Option<&str>,
        embeddings: Arc<E>,
    ) -> anyhow::Result<Self>
    where
        E: aidaemon_memory::Embedder + aidaemon_memory::sqlite::Reranker + 'static,
    {
        // Unit tests create thousands of NamedTempFile-backed databases. WAL
        // sidecars outlive the already-unlinked main file until every pooled
        // connection closes and can consume tens of gigabytes in one suite.
        // Delete journaling keeps test databases self-contained; production
        // retains WAL concurrency and crash behavior.
        let (journal_mode_name, journal_mode_pragma) = if cfg!(any(test, feature = "testing")) {
            ("delete", "PRAGMA journal_mode = DELETE")
        } else {
            ("wal", "PRAGMA journal_mode = WAL")
        };
        // Only the `encryption` feature adds to these.
        #[cfg_attr(not(feature = "encryption"), allow(unused_mut))]
        let mut opts = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            // Scrub deleted text from reusable SQLite pages. Explicit `/wipe`
            // also truncates the WAL after deleting active history.
            .pragma("secure_delete", "ON")
            // Startup projection/backfill and retention tasks can briefly
            // overlap. Wait for the writer instead of failing immediately
            // with SQLITE_BUSY and silently deferring an idempotent repair.
            .busy_timeout(std::time::Duration::from_secs(30))
            // synchronous=NORMAL is the recommended pairing with WAL: durable across
            // app crashes, only loses the very last transaction on an OS/power crash,
            // and avoids an fsync on every commit (the dominant write cost under FULL).
            .synchronous(SqliteSynchronous::Normal)
            // Larger page cache (negative = KiB, so 64 MiB) keeps hot tables resident.
            .pragma("cache_size", "-65536")
            // Memory-mapped reads (256 MiB) cut syscall overhead on read-heavy paths.
            .pragma("mmap_size", "268435456")
            // Keep temp B-trees (ORDER BY / GROUP BY spills) in RAM instead of on disk.
            .pragma("temp_store", "MEMORY");

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
            .max_connections(10)
            .acquire_timeout(std::time::Duration::from_secs(30))
            .connect_with(opts)
            .await?;

        // Journal mode belongs to the database, not to an individual pooled
        // operation. Set it once before migrations or concurrent work begins.
        // Putting it in SqliteConnectOptions makes every lazily opened pool
        // connection reissue the PRAGMA; SQLite can then return SQLITE_BUSY
        // if an established connection is already writing.
        let active_journal_mode: String = sqlx::query_scalar(journal_mode_pragma)
            .fetch_one(&pool)
            .await?;
        anyhow::ensure!(
            active_journal_mode.eq_ignore_ascii_case(journal_mode_name),
            "SQLite refused journal mode '{journal_mode_name}' (active mode: '{active_journal_mode}')"
        );

        // Set restrictive file permissions (owner-only read/write)
        set_db_file_permissions(db_path);

        migrations::migrate_state(&pool).await?;
        aidaemon_nodes::store::migrate(&pool).await?;

        let pool_for_memory = pool.clone();
        Ok(Self {
            pool,
            working_memory: Arc::new(RwLock::new(HashMap::new())),
            memory: sqlite_memory(pool_for_memory, embeddings.clone()),
            cap,
            secrets: None,
            #[cfg(test)]
            embedding_service: embeddings,
        })
    }

    /// Keeps dynamic bots' tokens in `secrets`, with a reference in the row.
    pub fn with_secret_store(mut self, secrets: Arc<dyn aidaemon_secrets::SecretStore>) -> Self {
        self.secrets = Some(secrets);
        self
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
            WHERE session_id = ?1
              AND event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND id > COALESCE(
                (SELECT cleared_after_id FROM session_context_boundaries WHERE session_id = ?1),
                0)
              AND (
                turn_id IS NULL
                OR (SELECT MIN(origin.id)
                    FROM events origin
                    WHERE origin.session_id = ?1
                      AND origin.turn_id = events.turn_id) > COALESCE(
                    (SELECT cleared_after_id FROM session_context_boundaries WHERE session_id = ?1),
                    0)
              )
            ORDER BY created_at DESC
            LIMIT ?2
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

            if let Some(message) = aidaemon_events::conversation_turn::turn_from_event(
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
}

/// The daemon's memory store over its state database. Errors keep only what
/// the failing tool said, without the daemon's own wrapper lines.
pub fn sqlite_memory<E>(
    pool: SqlitePool,
    embeddings: Arc<E>,
) -> aidaemon_memory::sqlite::SqliteMemory
where
    E: aidaemon_memory::Embedder + aidaemon_memory::sqlite::Reranker + 'static,
{
    fn without_scaffolding(error: &str) -> String {
        error
            .lines()
            .filter(|line| !crate::message_content::is_internal_scaffolding_line(line))
            .collect::<Vec<_>>()
            .join("\n")
    }
    aidaemon_memory::sqlite::SqliteMemory::new(pool.clone(), embeddings.clone())
        .with_message_log(Arc::new(crate::history::EventHistory::new(pool, None)))
        .with_reranker(embeddings)
        .with_channel_ids(Arc::new(crate::history::SessionChannelIds))
        .with_error_cleaner(without_scaffolding)
}

impl SqliteStateStore {
    /// Resolve a "keychain:key_name" reference to the actual value from the
    /// secret store. Falls back to the raw string if the lookup fails or it's
    /// not a reference.
    fn resolve_keychain_ref(&self, value: &str) -> String {
        if let Some(key_name) = value.strip_prefix("keychain:") {
            match self.secrets.as_deref().map(|secrets| secrets.get(key_name)) {
                Some(Ok(secret)) => return secret.into_exposed(),
                _ => {
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
}

mod conversation_summary;
mod dialogue_state;
mod dynamic_bots;
mod dynamic_cli_agents;
mod dynamic_mcp;
mod health_checks;
mod history_search;
mod messages;
pub mod migrations;
mod oauth;
mod prompt_snapshots;
pub mod retention;
mod session_channels;
mod settings;
mod skills;
mod token_usage;

#[cfg(test)]
mod memory_store_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod turn_id_hydration_tests {
    use super::*;
    use aidaemon_events::{Event, EventStore, EventType};
    use types::MessageStore;

    /// The event store on the daemon's seams, over the same pool.
    async fn event_store(pool: SqlitePool) -> EventStore {
        EventStore::on_paths(
            pool,
            aidaemon_tools::LexicalPaths::daemon_host(),
            Arc::new(crate::history::HistoryEventHost),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn hydrate_propagates_turn_id_from_event() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("turn_id_hydration.db");
        let db_path = db_path.to_str().unwrap();
        let embedding_service =
            Arc::new(aidaemon_memory::embeddings::EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(db_path, 50, None, embedding_service)
            .await
            .unwrap();

        // Append a user-message event carrying a turn_id through the event store,
        // sharing the same pool the state store hydrates from.
        let events = event_store(store.pool()).await;
        let ev = Event::new(
            "sess-hydrate",
            EventType::UserMessage,
            serde_json::json!({"content": "hello", "turn_id": "turn-hydrate"}),
        );
        events.append(ev).await.unwrap();

        let hydrated = store.hydrate("sess-hydrate").await.unwrap();
        let msg = hydrated
            .iter()
            .find(|m| m.role == "user")
            .expect("hydrated user message");
        assert_eq!(msg.turn_id.as_deref(), Some("turn-hydrate"));
    }

    async fn append_user(events: &EventStore, session: &str, text: &str) {
        events
            .append(Event::new(
                session,
                EventType::UserMessage,
                serde_json::json!({"content": text, "turn_id": format!("turn-{text}")}),
            ))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn clear_context_hides_history_but_keeps_events() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("clear_ctx.db");
        let db_path = db_path.to_str().unwrap();
        let embedding_service =
            Arc::new(aidaemon_memory::embeddings::EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(db_path, 50, None, embedding_service)
            .await
            .unwrap();
        let events = event_store(store.pool()).await;
        let sess = "sess-clear";

        append_user(&events, sess, "before-one").await;
        append_user(&events, sess, "before-two").await;
        assert_eq!(store.hydrate(sess).await.unwrap().len(), 2);

        // Non-destructive clear: context is hidden, events remain in the DB.
        store.clear_session_context(sess).await.unwrap();
        assert!(
            store.hydrate(sess).await.unwrap().is_empty(),
            "context after /clear must start fresh"
        );
        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM events WHERE session_id = ?")
            .bind(sess)
            .fetch_one(&store.pool())
            .await
            .unwrap();
        assert_eq!(remaining, 2, "/clear must NOT delete events");

        // New turns after the boundary ARE visible.
        append_user(&events, sess, "after-clear").await;
        let after = store.hydrate(sess).await.unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].content.as_deref(), Some("after-clear"));
    }

    #[tokio::test]
    async fn advanced_context_boundary_survives_hydration_and_excludes_late_old_turn_events() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("advance_ctx.db");
        let db_path = db_path.to_str().unwrap();
        let embedding_service =
            Arc::new(aidaemon_memory::embeddings::EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(db_path, 50, None, embedding_service)
            .await
            .unwrap();
        let events = event_store(store.pool()).await;
        let sess = "sess-advance";

        events
            .append(Event::new(
                sess,
                EventType::UserMessage,
                serde_json::json!({
                    "message_id": "old-user",
                    "content": "old request",
                    "turn_id": "turn-old"
                }),
            ))
            .await
            .unwrap();
        let current_event_id = events
            .append(Event::new(
                sess,
                EventType::UserMessage,
                serde_json::json!({
                    "message_id": "current-user",
                    "content": "current request",
                    "turn_id": "turn-current"
                }),
            ))
            .await
            .unwrap();

        // Prime the hot window with both turns, then advance the durable floor
        // to immediately before the current typed turn.
        assert_eq!(store.get_history(sess, 10).await.unwrap().len(), 2);
        store
            .advance_session_context_boundary(
                sess,
                current_event_id.saturating_sub(1),
                "turn-current",
            )
            .await
            .unwrap();
        let hot = store.get_history(sess, 10).await.unwrap();
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].id, "current-user");

        // A delayed callback from the old task arrives after the boundary. Its
        // event id is new, but its turn origin is not, so hydration and the
        // turn-anchored provider transcript must both keep it out.
        events
            .append(Event::new(
                sess,
                EventType::AssistantResponse,
                serde_json::json!({
                    "message_id": "late-old-answer",
                    "content": "phase=OLD",
                    "turn_id": "turn-old",
                    "task_id": "task-old",
                    "model": "test",
                    "referenced_receipts": []
                }),
            ))
            .await
            .unwrap();

        let hydrated = store.hydrate(sess).await.unwrap();
        assert_eq!(hydrated.len(), 1);
        assert_eq!(hydrated[0].id, "current-user");
        let turns = events.get_turns_from_anchor(sess, 0).await.unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].turn_id.as_deref(), Some("turn-current"));
        let contextual = events
            .query_context_events(sess, chrono::Utc::now() - chrono::Duration::hours(1))
            .await
            .unwrap();
        assert!(contextual.iter().any(|event| {
            event
                .data
                .get("message_id")
                .and_then(serde_json::Value::as_str)
                == Some("current-user")
        }));
        assert!(!contextual.iter().any(|event| {
            event
                .data
                .get("message_id")
                .and_then(serde_json::Value::as_str)
                == Some("late-old-answer")
        }));
    }

    #[tokio::test]
    async fn wipe_deletes_events_and_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("wipe.db");
        let db_path = db_path.to_str().unwrap();
        let embedding_service =
            Arc::new(aidaemon_memory::embeddings::EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(db_path, 50, None, embedding_service)
            .await
            .unwrap();
        let events = event_store(store.pool()).await;
        let sess = "sess-wipe";

        append_user(&events, sess, "gone-one").await;
        store.clear_session_context(sess).await.unwrap();
        append_user(&events, sess, "gone-two").await;

        // Destructive wipe: everything for the session is deleted.
        store.clear_session(sess).await.unwrap();
        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM events WHERE session_id = ?")
            .bind(sess)
            .fetch_one(&store.pool())
            .await
            .unwrap();
        assert_eq!(remaining, 0, "/wipe must delete events");
        let boundaries: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM session_context_boundaries WHERE session_id = ?",
        )
        .bind(sess)
        .fetch_one(&store.pool())
        .await
        .unwrap();
        assert_eq!(boundaries, 0, "/wipe must clear the boundary row too");
    }
}
