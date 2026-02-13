use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use tokio::sync::RwLock;

use crate::traits::{
    BehaviorPattern, ConversationSummary, Episode, ErrorSolution, Expertise, Fact, Goal,
    GoalTokenBudgetStatus, GoalV3, Message, Procedure, StateStore, TaskActivityV3, TaskV3,
    TokenUsage, TokenUsageRecord, UserProfile,
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

pub struct SqliteStateStore {
    pool: SqlitePool,
    working_memory: Arc<RwLock<HashMap<String, VecDeque<Message>>>>,
    cap: usize,
    embedding_service: Arc<EmbeddingService>,
}

fn parse_legacy_datetime_to_local(raw: &str) -> Option<chrono::DateTime<chrono::Local>> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|dt| dt.with_timezone(&chrono::Local))
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S")
                .ok()
                .and_then(|naive| match chrono::Local.from_local_datetime(&naive) {
                    chrono::LocalResult::Single(dt) => Some(dt),
                    chrono::LocalResult::Ambiguous(early, _) => Some(early),
                    chrono::LocalResult::None => None,
                })
        })
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
            )",
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
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)",
        )
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
        let _ = sqlx::query("ALTER TABLE messages ADD COLUMN embedding_error TEXT")
            .execute(&pool)
            .await;

        // 4. Add consolidated_at column for memory consolidation (Layer 6)
        let _ = sqlx::query("ALTER TABLE messages ADD COLUMN consolidated_at TEXT")
            .execute(&pool)
            .await;

        // --- Human-Like Memory System Migrations ---
        // 5. Add new columns to facts table for supersession and recall tracking
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN superseded_at TEXT")
            .execute(&pool)
            .await;
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN recall_count INTEGER DEFAULT 0")
            .execute(&pool)
            .await;
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN last_recalled_at TEXT")
            .execute(&pool)
            .await;

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
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)")
            .execute(&pool)
            .await?;

        // Prevent concurrent episode creation for the same session
        let _ = sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_session_unique ON episodes(session_id)",
        )
        .execute(&pool)
        .await;

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
            )",
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
            )",
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
            )",
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
            )",
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
            )",
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
            )",
        )
        .execute(&pool)
        .await?;

        // Terminal allowed prefixes (persisted "Allow Always" approvals)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS terminal_allowed_prefixes (
                prefix TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
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
            )",
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
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run
                ON scheduled_tasks(next_run_at) WHERE is_paused = 0",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_tasks_name_source
                ON scheduled_tasks(name) WHERE source = 'config'",
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
            )",
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
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_token_usage_created_at
             ON token_usage(created_at)",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_token_usage_session_created_at
             ON token_usage(session_id, created_at)",
        )
        .execute(&pool)
        .await?;

        // Token usage daily aggregates (for retention cleanup)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS token_usage_daily (
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
            )",
        )
        .execute(&pool)
        .await?;

        // Session-channel mapping — persists session_id → channel_name so the
        // hub can route notifications after a restart (session_map is in-memory).
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS session_channels (
                session_id TEXT PRIMARY KEY,
                channel_name TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
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
            )",
        )
        .execute(&pool)
        .await?;

        // Migration: add resources_json column if missing
        sqlx::query(
            "ALTER TABLE dynamic_skills ADD COLUMN resources_json TEXT NOT NULL DEFAULT '[]'",
        )
        .execute(&pool)
        .await
        .ok();

        // Skill drafts table - stores auto-promoted skill drafts pending user review
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS skill_drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                triggers_json TEXT NOT NULL DEFAULT '[]',
                body TEXT NOT NULL,
                source_procedure TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // Dynamic MCP servers table - stores MCP servers added via manage_mcp tool
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS dynamic_mcp_servers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                command TEXT NOT NULL,
                args_json TEXT NOT NULL DEFAULT '[]',
                env_keys_json TEXT NOT NULL DEFAULT '[]',
                triggers_json TEXT NOT NULL DEFAULT '[]',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // Dynamic CLI agents table - stores CLI agents added via manage_cli_agents tool
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS dynamic_cli_agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                command TEXT NOT NULL,
                args_json TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                timeout_secs INTEGER,
                max_output_chars INTEGER,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // CLI agent invocations table - logs each CLI agent run for auditing
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS cli_agent_invocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                prompt_summary TEXT NOT NULL,
                working_dir TEXT,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT,
                exit_code INTEGER,
                output_summary TEXT,
                success INTEGER,
                duration_secs REAL
            )",
        )
        .execute(&pool)
        .await?;

        // People tables - for tracking the owner's social circle
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                aliases_json TEXT NOT NULL DEFAULT '[]',
                relationship TEXT,
                platform_ids_json TEXT NOT NULL DEFAULT '{}',
                notes TEXT,
                communication_style TEXT,
                language_preference TEXT,
                last_interaction_at TEXT,
                interaction_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS person_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'agent',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(person_id, category, key)
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_people_name ON people(name)")
            .execute(&pool)
            .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_person_facts_person ON person_facts(person_id)",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_person_facts_category ON person_facts(category)",
        )
        .execute(&pool)
        .await?;

        // --- OAuth connections table ---
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS oauth_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL UNIQUE,
                auth_type TEXT NOT NULL,
                username TEXT,
                scopes TEXT NOT NULL DEFAULT '[]',
                token_expires_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // --- Settings table (generic key-value runtime toggles) ---
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        // --- Channel-Scoped Memory Migrations ---
        // Add channel_id and privacy columns to facts table
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN channel_id TEXT")
            .execute(&pool)
            .await;
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN privacy TEXT DEFAULT 'global'")
            .execute(&pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_facts_channel ON facts(channel_id)")
            .execute(&pool)
            .await;
        let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_facts_privacy ON facts(privacy)")
            .execute(&pool)
            .await;
        // Add channel_id column to episodes table
        let _ = sqlx::query("ALTER TABLE episodes ADD COLUMN channel_id TEXT")
            .execute(&pool)
            .await;

        // --- Binary Embedding Storage Migration ---
        // Add embedding column to facts table for pre-computed embeddings
        let _ = sqlx::query("ALTER TABLE facts ADD COLUMN embedding BLOB")
            .execute(&pool)
            .await;

        // --- V3 Orchestration Tables ---
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS goals_v3 (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                goal_type TEXT NOT NULL DEFAULT 'finite',
                status TEXT NOT NULL DEFAULT 'active',
                priority TEXT NOT NULL DEFAULT 'medium',
                conditions TEXT,
                schedule TEXT,
                context TEXT,
                resources TEXT,
                budget_per_check INTEGER,
                budget_daily INTEGER,
                tokens_used_today INTEGER NOT NULL DEFAULT 0,
                last_useful_action TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT,
                parent_goal_id TEXT,
                session_id TEXT NOT NULL,
                notified_at TEXT
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_v3_status ON goals_v3(status)")
            .execute(&pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_v3_session ON goals_v3(session_id)")
            .execute(&pool)
            .await?;

        // Migration: add notified_at column to goals_v3 (nullable)
        let _ = sqlx::query("ALTER TABLE goals_v3 ADD COLUMN notified_at TEXT")
            .execute(&pool)
            .await;

        // Migration: add notification_attempts column to goals_v3 for retry tracking
        let _ = sqlx::query(
            "ALTER TABLE goals_v3 ADD COLUMN notification_attempts INTEGER NOT NULL DEFAULT 0",
        )
        .execute(&pool)
        .await;

        // Migration: add dispatch_failures column for progress-based circuit breaker
        let _ = sqlx::query(
            "ALTER TABLE goals_v3 ADD COLUMN dispatch_failures INTEGER NOT NULL DEFAULT 0",
        )
        .execute(&pool)
        .await;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS tasks_v3 (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL REFERENCES goals_v3(id) ON DELETE CASCADE,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                priority TEXT NOT NULL DEFAULT 'medium',
                task_order INTEGER NOT NULL DEFAULT 0,
                parallel_group TEXT,
                depends_on TEXT,
                agent_id TEXT,
                context TEXT,
                result TEXT,
                error TEXT,
                blocker TEXT,
                idempotent INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                started_at TEXT,
                completed_at TEXT
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_v3_goal ON tasks_v3(goal_id)")
            .execute(&pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_v3_status ON tasks_v3(status)")
            .execute(&pool)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS task_activity_v3 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES tasks_v3(id) ON DELETE CASCADE,
                activity_type TEXT NOT NULL,
                tool_name TEXT,
                tool_args TEXT,
                result TEXT,
                success INTEGER,
                tokens_used INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_task_activity_v3_task ON task_activity_v3(task_id)",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_task_activity_v3_created_at
             ON task_activity_v3(created_at)",
        )
        .execute(&pool)
        .await?;

        // Notification queue — queued when channel unavailable, delivered on reconnect.
        // Retention: status_update expires after 24h, critical persists indefinitely.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS notification_queue (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                priority TEXT NOT NULL DEFAULT 'status_update',
                message TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                delivered_at TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                expires_at TEXT
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_notification_queue_pending
             ON notification_queue(delivered_at, priority, created_at)
             WHERE delivered_at IS NULL",
        )
        .execute(&pool)
        .await?;

        // Token alert detector dedupe/cooldown state.
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS token_alert_state (
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                last_alert_at TEXT NOT NULL,
                last_metric_tokens INTEGER NOT NULL DEFAULT 0,
                last_metric_calls INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (scope_type, scope_id)
            )",
        )
        .execute(&pool)
        .await?;

        // Conversation summaries for context window management
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS conversation_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                last_message_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await?;

        // Migration: deduplicate people entries and add unique index on LOWER(name).
        // Keeps the row with the lowest id for each name, merging interaction counts.
        let _ = sqlx::query(
            "DELETE FROM people WHERE id NOT IN (
                SELECT MIN(id) FROM people GROUP BY LOWER(name)
            )",
        )
        .execute(&pool)
        .await;
        let _ = sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_people_name_unique ON people(LOWER(name))",
        )
        .execute(&pool)
        .await;

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

            let message_id = data
                .get("message_id")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| event_id.to_string());

            let message = match event_type.as_str() {
                "user_message" => Some(Message {
                    id: message_id,
                    session_id: row.get("session_id"),
                    role: "user".to_string(),
                    content: Some(
                        data.get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                    ),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls_json: None,
                    created_at,
                    importance: 0.5,
                    embedding: None,
                }),
                "assistant_response" => {
                    let tool_calls_json = data
                        .get("tool_calls")
                        .and_then(|v| v.as_array())
                        .and_then(|calls| {
                            let mapped: Vec<crate::traits::ToolCall> = calls
                                .iter()
                                .filter_map(|tc| {
                                    let id = tc.get("id").and_then(|v| v.as_str())?;
                                    let name = tc.get("name").and_then(|v| v.as_str())?;
                                    Some(crate::traits::ToolCall {
                                        id: id.to_string(),
                                        name: name.to_string(),
                                        arguments: tc
                                            .get("arguments")
                                            .cloned()
                                            .unwrap_or_else(|| serde_json::json!({}))
                                            .to_string(),
                                        extra_content: tc.get("extra_content").cloned(),
                                    })
                                })
                                .collect();
                            if mapped.is_empty() {
                                None
                            } else {
                                serde_json::to_string(&mapped).ok()
                            }
                        });

                    Some(Message {
                        id: message_id,
                        session_id: row.get("session_id"),
                        role: "assistant".to_string(),
                        content: data
                            .get("content")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json,
                        created_at,
                        importance: 0.5,
                        embedding: None,
                    })
                }
                "tool_result" => Some(Message {
                    id: message_id,
                    session_id: row.get("session_id"),
                    role: "tool".to_string(),
                    content: Some(
                        data.get("result")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                    ),
                    tool_call_id: Some(
                        data.get("tool_call_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("event-tool-{}", event_id)),
                    ),
                    tool_name: Some(
                        data.get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("system")
                            .to_string(),
                    ),
                    tool_calls_json: None,
                    created_at,
                    importance: 0.5,
                    embedding: None,
                }),
                _ => None,
            };

            if let Some(message) = message {
                deque.push_back(message);
            }
        }

        Ok(deque)
    }

    async fn hydrate_from_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<VecDeque<Message>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at
            FROM messages
            WHERE session_id = ?
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
                importance: row.try_get("importance").unwrap_or(0.5),
                embedding: None,
            });
        }

        Ok(deque)
    }

    /// Hydrate working memory for a session from the database.
    async fn hydrate(&self, session_id: &str) -> anyhow::Result<VecDeque<Message>> {
        match self.hydrate_from_events(session_id, self.cap).await {
            Ok(deque) => {
                tracing::debug!(
                    session_id,
                    hydrated_count = deque.len(),
                    "hydrate: loaded conversation from canonical event stream"
                );
                Ok(deque)
            }
            Err(e) => {
                if e.to_string().contains("no such table: events") {
                    tracing::warn!(
                        session_id,
                        "hydrate: events table missing, falling back to legacy messages table"
                    );
                    let deque = self.hydrate_from_messages(session_id, self.cap).await?;
                    tracing::debug!(
                        session_id,
                        hydrated_count = deque.len(),
                        "hydrate: loaded conversation from legacy messages table"
                    );
                    Ok(deque)
                } else {
                    Err(e)
                }
            }
        }
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

    // ==================== Goal Methods ====================

    /// Insert a new goal.
    #[allow(dead_code)]
    pub async fn insert_goal(&self, goal: &Goal) -> anyhow::Result<i64> {
        let progress_notes_json = goal
            .progress_notes
            .as_ref()
            .map(|p| serde_json::to_string(p).unwrap_or_default());
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
    pub async fn update_goal(
        &self,
        goal_id: i64,
        status: Option<&str>,
        progress_note: Option<&str>,
    ) -> anyhow::Result<()> {
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
            let completed_at = if s == "completed" {
                Some(now.clone())
            } else {
                None
            };
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

        let query_vec = self
            .embedding_service
            .embed(description.to_string())
            .await?;
        let goal_texts: Vec<String> = goals.iter().map(|g| g.description.clone()).collect();
        let goal_embeddings = self.embedding_service.embed_batch(goal_texts).await?;

        let mut best_match: Option<(usize, f32)> = None;
        for (i, emb) in goal_embeddings.iter().enumerate() {
            let score = crate::memory::math::cosine_similarity(&query_vec, emb);
            if score > 0.75 && (best_match.is_none() || score > best_match.unwrap().1) {
                best_match = Some((i, score));
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
            created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            completed_at: completed_str.and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            }),
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

            if score > 0.3 {
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
        let steps_json = solution
            .solution_steps
            .as_ref()
            .map(|s| serde_json::to_string(s).unwrap_or_default());
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
    pub async fn get_relevant_error_solutions(
        &self,
        error: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ErrorSolution>> {
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

        Ok(ErrorSolution {
            id: row.get("id"),
            error_pattern: row.get("error_pattern"),
            domain: row.get("domain"),
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

#[async_trait]
impl StateStore for SqliteStateStore {
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()> {
        // Canonical persistence is event-sourced (EventStore). We only keep
        // an in-memory session window here for fast hot-path reads.
        let mut wm = self.working_memory.write().await;
        let deque = wm
            .entry(msg.session_id.clone())
            .or_insert_with(VecDeque::new);
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

    async fn upsert_fact(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        let privacy_str = privacy.to_string();

        // Pre-compute embedding for the fact text
        let fact_text = format!("[{}] {}: {}", category, key, value);
        let embedding_blob = self
            .embedding_service
            .embed(fact_text)
            .await
            .ok()
            .map(|v| encode_embedding(&v));

        // Find existing current fact (not superseded)
        let existing = sqlx::query(
            "SELECT id, value FROM facts WHERE category = ? AND key = ? AND superseded_at IS NULL",
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

                // Insert new fact with embedding — ignore duplicate entry errors (code 2067)
                // that can occur due to UNIQUE(category, key) constraint race conditions
                let insert_result = sqlx::query(
                    "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy, embedding)
                     VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
                )
                .bind(category)
                .bind(key)
                .bind(value)
                .bind(source)
                .bind(&now)
                .bind(&now)
                .bind(channel_id)
                .bind(&privacy_str)
                .bind(&embedding_blob)
                .execute(&self.pool)
                .await;

                match insert_result {
                    Ok(_) => {}
                    Err(sqlx::Error::Database(ref db_err))
                        if db_err.code().as_deref() == Some("2067") =>
                    {
                        // Duplicate entry — fact already exists, ignore
                    }
                    Err(e) => return Err(e.into()),
                }
            } else {
                // Same value - update timestamp/source and backfill embedding if missing
                sqlx::query(
                    "UPDATE facts SET source = ?, updated_at = ?, embedding = COALESCE(embedding, ?) WHERE id = ?",
                )
                .bind(source)
                .bind(&now)
                .bind(&embedding_blob)
                .bind(old_id)
                .execute(&self.pool)
                .await?;
            }
        } else {
            // No existing fact - insert new with embedding
            // Ignore duplicate entry errors (code 2067) from concurrent inserts
            let insert_result = sqlx::query(
                "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy, embedding)
                 VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
            )
            .bind(category)
            .bind(key)
            .bind(value)
            .bind(source)
            .bind(&now)
            .bind(&now)
            .bind(channel_id)
            .bind(&privacy_str)
            .bind(&embedding_blob)
            .execute(&self.pool)
            .await;

            match insert_result {
                Ok(_) => {}
                Err(sqlx::Error::Database(ref db_err))
                    if db_err.code().as_deref() == Some("2067") =>
                {
                    // Duplicate entry — fact already exists, ignore
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
        // Only return current (non-superseded) facts
        let rows = if let Some(cat) = category {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy FROM facts WHERE category = ? AND superseded_at IS NULL ORDER BY updated_at DESC")
                .bind(cat)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC")
                .fetch_all(&self.pool)
                .await?
        };

        let mut facts = Vec::with_capacity(rows.len());
        for row in rows {
            facts.push(Self::row_to_fact(&row));
        }
        Ok(facts)
    }

    async fn get_relevant_facts(&self, query: &str, max: usize) -> anyhow::Result<Vec<Fact>> {
        // Load facts with stored embeddings
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding
             FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let all_facts: Vec<Fact> = rows.iter().map(Self::row_to_fact).collect();

        if all_facts.is_empty() || query.trim().is_empty() {
            let mut facts = all_facts;
            facts.truncate(max);
            return Ok(facts);
        }

        // Embed the query
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "Failed to embed query for fact filtering, returning all facts: {}",
                    e
                );
                let mut facts = all_facts;
                facts.truncate(max);
                return Ok(facts);
            }
        };

        // Score facts using stored embeddings
        let mut scored: Vec<(usize, f32)> = Vec::new();
        let mut unscored: Vec<usize> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let embedding: Option<Vec<u8>> = row.get("embedding");
            if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    let score = crate::memory::math::cosine_similarity(&query_vec, &vec);
                    scored.push((i, score));
                    continue;
                }
            }
            // Facts without embeddings (during backfill) are included without scoring
            unscored.push(i);
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top `max` facts that are above a minimum relevance threshold
        let mut relevant: Vec<Fact> = scored
            .into_iter()
            .filter(|(_, score)| *score > 0.5)
            .take(max)
            .map(|(i, _)| all_facts[i].clone())
            .collect();

        // Include unscored facts (missing embeddings) to avoid dropping them
        for i in unscored {
            if relevant.len() >= max {
                break;
            }
            relevant.push(all_facts[i].clone());
        }

        // If filtering left us with very few facts, pad with most recent ones
        if relevant.len() < max / 3 && all_facts.len() > relevant.len() {
            let existing_ids: std::collections::HashSet<i64> =
                relevant.iter().map(|f| f.id).collect();
            for fact in &all_facts {
                if relevant.len() >= max {
                    break;
                }
                if !existing_ids.contains(&fact.id) {
                    relevant.push(fact.clone());
                }
            }
        }

        Ok(relevant)
    }

    async fn get_context(
        &self,
        session_id: &str,
        _query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        // Canonical context retrieval is event-backed. The in-memory working
        // window is hydrated from events on cold start by get_history().
        self.get_history(session_id, limit).await
    }

    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        // Clear working memory
        {
            let mut wm = self.working_memory.write().await;
            wm.remove(session_id);
        }

        // Delete session rows across canonical + legacy tables.
        // Some test/legacy DBs may not have all tables yet; treat missing tables as best-effort.
        for table in ["events", "messages", "conversation_summaries"] {
            let query = format!("DELETE FROM {table} WHERE session_id = ?");
            if let Err(e) = sqlx::query(&query)
                .bind(session_id)
                .execute(&self.pool)
                .await
            {
                let missing_table = format!("no such table: {table}");
                if !e.to_string().contains(&missing_table) {
                    return Err(e.into());
                }
            }
        }
        Ok(())
    }

    async fn record_token_usage(&self, session_id: &str, usage: &TokenUsage) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO token_usage (session_id, model, input_tokens, output_tokens, created_at)
             VALUES (?, ?, ?, ?, datetime('now'))",
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
             FROM token_usage WHERE created_at >= ? ORDER BY created_at DESC",
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

    async fn get_token_usage_by_session(
        &self,
        since: &str,
    ) -> anyhow::Result<Vec<(String, i64, i64, i64)>> {
        let rows = sqlx::query(
            "SELECT session_id, SUM(input_tokens) as total_input, \
             SUM(output_tokens) as total_output, COUNT(*) as request_count \
             FROM token_usage WHERE created_at >= ? \
             GROUP BY session_id ORDER BY (total_input + total_output) DESC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            results.push((
                row.get::<String, _>("session_id"),
                row.get::<i64, _>("total_input"),
                row.get::<i64, _>("total_output"),
                row.get::<i64, _>("request_count"),
            ));
        }
        Ok(results)
    }

    // ==================== Extended Memory Trait Methods ====================

    async fn get_relevant_episodes(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Episode>> {
        // Delegate to inherent method
        SqliteStateStore::get_relevant_episodes(self, query, limit).await
    }

    async fn get_active_goals(&self) -> anyhow::Result<Vec<Goal>> {
        SqliteStateStore::get_active_goals(self).await
    }

    async fn update_goal(
        &self,
        goal_id: i64,
        status: Option<&str>,
        progress_note: Option<&str>,
    ) -> anyhow::Result<()> {
        SqliteStateStore::update_goal(self, goal_id, status, progress_note).await
    }

    async fn get_behavior_patterns(
        &self,
        min_confidence: f32,
    ) -> anyhow::Result<Vec<BehaviorPattern>> {
        SqliteStateStore::get_behavior_patterns(self, min_confidence).await
    }

    async fn get_relevant_procedures(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Procedure>> {
        SqliteStateStore::get_relevant_procedures(self, query, limit).await
    }

    async fn get_relevant_error_solutions(
        &self,
        error: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ErrorSolution>> {
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

    // ==================== Channel-Scoped Memory Methods ====================

    async fn get_relevant_facts_for_channel(
        &self,
        query: &str,
        max: usize,
        channel_id: Option<&str>,
        visibility: ChannelVisibility,
    ) -> anyhow::Result<Vec<Fact>> {
        // In DM/Internal contexts, return all facts (existing behavior)
        if matches!(
            visibility,
            ChannelVisibility::Private | ChannelVisibility::Internal
        ) {
            return self.get_relevant_facts(query, max).await;
        }

        // PublicExternal: only same-channel facts + sanitized global facts
        // Public/PrivateGroup: global + same-channel facts (no private, no other-channel)
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding
             FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        // Build facts and track which indices pass the privacy filter
        let all_facts: Vec<Fact> = rows.iter().map(Self::row_to_fact).collect();
        let filtered_indices: Vec<usize> = all_facts
            .iter()
            .enumerate()
            .filter(|(_, f)| match f.privacy {
                FactPrivacy::Private => false,
                FactPrivacy::Global => {
                    if matches!(visibility, ChannelVisibility::PublicExternal) {
                        !matches!(f.category.as_str(), "personal" | "health" | "finance")
                    } else {
                        true
                    }
                }
                FactPrivacy::Channel => match (channel_id, &f.channel_id) {
                    (Some(current), Some(fact_ch)) => current == fact_ch,
                    (None, None) => true,
                    _ => false,
                },
            })
            .map(|(i, _)| i)
            .collect();

        let filtered: Vec<Fact> = filtered_indices
            .iter()
            .map(|&i| all_facts[i].clone())
            .collect();

        if filtered.is_empty() || query.trim().is_empty() {
            let mut facts = filtered;
            facts.truncate(max);
            return Ok(facts);
        }

        // Apply semantic filtering using stored embeddings
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => {
                let mut facts = filtered;
                facts.truncate(max);
                return Ok(facts);
            }
        };

        let mut scored: Vec<(usize, f32)> = Vec::new();
        let mut unscored: Vec<usize> = Vec::new();
        for (fi, &ri) in filtered_indices.iter().enumerate() {
            let embedding: Option<Vec<u8>> = rows[ri].get("embedding");
            if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    let score = crate::memory::math::cosine_similarity(&query_vec, &vec);
                    scored.push((fi, score));
                    continue;
                }
            }
            unscored.push(fi);
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut relevant: Vec<Fact> = scored
            .into_iter()
            .filter(|(_, score)| *score > 0.5)
            .take(max)
            .map(|(i, _)| filtered[i].clone())
            .collect();

        for i in unscored {
            if relevant.len() >= max {
                break;
            }
            relevant.push(filtered[i].clone());
        }

        if relevant.len() < max / 3 && filtered.len() > relevant.len() {
            let existing_ids: std::collections::HashSet<i64> =
                relevant.iter().map(|f| f.id).collect();
            for fact in &filtered {
                if relevant.len() >= max {
                    break;
                }
                if !existing_ids.contains(&fact.id) {
                    relevant.push(fact.clone());
                }
            }
        }

        Ok(relevant)
    }

    async fn get_cross_channel_hints(
        &self,
        query: &str,
        current_channel_id: &str,
        max: usize,
    ) -> anyhow::Result<Vec<Fact>> {
        // Get channel-scoped facts from OTHER channels that are relevant to the query
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding
             FROM facts
             WHERE superseded_at IS NULL
               AND privacy = 'channel'
               AND channel_id IS NOT NULL
               AND channel_id != ?
             ORDER BY updated_at DESC",
        )
        .bind(current_channel_id)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || query.trim().is_empty() {
            return Ok(vec![]);
        }

        let facts: Vec<Fact> = rows.iter().map(Self::row_to_fact).collect();

        // Apply semantic filtering using stored embeddings
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => return Ok(vec![]),
        };

        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let embedding: Option<Vec<u8>> = row.get("embedding");
            if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    let score = crate::memory::math::cosine_similarity(&query_vec, &vec);
                    scored.push((i, score));
                }
            }
            // Facts without embeddings are skipped for cross-channel hints (conservative)
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let hints: Vec<Fact> = scored
            .into_iter()
            .filter(|(_, score)| *score > 0.6) // Higher threshold for cross-channel hints
            .take(max)
            .map(|(i, _)| facts[i].clone())
            .collect();

        Ok(hints)
    }

    async fn update_fact_privacy(&self, fact_id: i64, privacy: FactPrivacy) -> anyhow::Result<()> {
        sqlx::query("UPDATE facts SET privacy = ? WHERE id = ?")
            .bind(privacy.to_string())
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn delete_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
            .bind(&now)
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<Fact>> {
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy
             FROM facts WHERE superseded_at IS NULL ORDER BY category, key"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(Self::row_to_fact).collect())
    }

    async fn get_relevant_episodes_for_channel(
        &self,
        query: &str,
        limit: usize,
        channel_id: Option<&str>,
    ) -> anyhow::Result<Vec<Episode>> {
        // For channel-scoped episode retrieval, filter episodes by channel_id
        // Episodes without channel_id (legacy) are accessible everywhere
        let rows = sqlx::query(
            "SELECT id, session_id, summary, topics, emotional_tone, outcome, importance, recall_count, last_recalled_at, message_count, start_time, end_time, created_at, channel_id, embedding
             FROM episodes WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT 500"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || query.trim().is_empty() {
            return Ok(vec![]);
        }

        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => return Ok(vec![]),
        };

        let mut scored: Vec<(Episode, f32)> = Vec::new();
        for row in rows {
            // Filter by channel: include episodes from same channel or legacy (no channel_id)
            let ep_channel_id: Option<String> = row.try_get("channel_id").unwrap_or(None);
            let include = match (&ep_channel_id, channel_id) {
                (None, _) => true,                                      // Legacy episodes: include
                (Some(ep_ch), Some(current_ch)) => ep_ch == current_ch, // Same channel
                (Some(_), None) => false, // Has channel but no current: skip
            };
            if !include {
                continue;
            }

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
                    if score > 0.5 {
                        scored.push((episode, score));
                    }
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let episodes: Vec<Episode> = scored.into_iter().take(limit).map(|(e, _)| e).collect();
        Ok(episodes)
    }

    // ==================== Write Methods for Learning System ====================

    async fn increment_expertise(
        &self,
        domain: &str,
        success: bool,
        error: Option<&str>,
    ) -> anyhow::Result<()> {
        SqliteStateStore::increment_expertise(self, domain, success, error).await
    }

    async fn upsert_procedure(&self, procedure: &Procedure) -> anyhow::Result<i64> {
        SqliteStateStore::insert_procedure(self, procedure).await
    }

    async fn update_procedure_outcome(
        &self,
        procedure_id: i64,
        success: bool,
        duration: Option<f32>,
    ) -> anyhow::Result<()> {
        SqliteStateStore::update_procedure(self, procedure_id, success, None, duration).await
    }

    async fn insert_error_solution(&self, solution: &ErrorSolution) -> anyhow::Result<i64> {
        SqliteStateStore::insert_error_solution(self, solution).await
    }

    async fn update_error_solution_outcome(
        &self,
        solution_id: i64,
        success: bool,
    ) -> anyhow::Result<()> {
        SqliteStateStore::update_error_solution(self, solution_id, success).await
    }

    // ==================== Dynamic Bots Methods ====================

    async fn add_dynamic_bot(&self, bot: &crate::traits::DynamicBot) -> anyhow::Result<i64> {
        let allowed_user_ids_json = serde_json::to_string(&bot.allowed_user_ids)?;

        // Store tokens in OS keychain to avoid plaintext storage in SQLite.
        // We insert first to get the row ID, then store in keychain and update
        // the row to hold a "keychain:key" reference instead of the raw token.
        let result = sqlx::query(
            "INSERT INTO dynamic_bots (channel_type, bot_token, app_token, allowed_user_ids, extra_config, created_at)
             VALUES (?, ?, ?, ?, ?, datetime('now'))"
        )
        .bind(&bot.channel_type)
        .bind(&bot.bot_token) // Temporarily store plaintext; will be replaced below
        .bind(&bot.app_token)
        .bind(&allowed_user_ids_json)
        .bind(&bot.extra_config)
        .execute(&self.pool)
        .await?;
        let row_id = result.last_insert_rowid();

        // Try to move the bot_token to keychain
        let bot_token_key = format!("dynamic_bot_{}_bot_token", row_id);
        if crate::config::store_in_keychain(&bot_token_key, &bot.bot_token).is_ok() {
            // Replace plaintext with keychain reference
            let _ = sqlx::query("UPDATE dynamic_bots SET bot_token = ? WHERE id = ?")
                .bind(format!("keychain:{}", bot_token_key))
                .bind(row_id)
                .execute(&self.pool)
                .await;
        }

        // Try to move the app_token to keychain (Slack bots)
        if let Some(ref app_tok) = bot.app_token {
            let app_token_key = format!("dynamic_bot_{}_app_token", row_id);
            if crate::config::store_in_keychain(&app_token_key, app_tok).is_ok() {
                let _ = sqlx::query("UPDATE dynamic_bots SET app_token = ? WHERE id = ?")
                    .bind(format!("keychain:{}", app_token_key))
                    .bind(row_id)
                    .execute(&self.pool)
                    .await;
            }
        }

        Ok(row_id)
    }

    async fn update_dynamic_bot_allowed_users(
        &self,
        bot_token: &str,
        allowed_user_ids: &[String],
    ) -> anyhow::Result<()> {
        // Find the bot by resolved token (tokens may be stored as keychain refs)
        let bots = self.get_dynamic_bots().await?;
        let bot = bots
            .iter()
            .find(|b| b.bot_token == bot_token)
            .ok_or_else(|| anyhow::anyhow!("Dynamic bot not found for token"))?;

        let allowed_json = serde_json::to_string(allowed_user_ids)?;
        sqlx::query("UPDATE dynamic_bots SET allowed_user_ids = ? WHERE id = ?")
            .bind(&allowed_json)
            .bind(bot.id)
            .execute(&self.pool)
            .await?;
        Ok(())
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
            let allowed_user_ids: Vec<String> =
                serde_json::from_str(&allowed_user_ids_json).unwrap_or_default();

            // Resolve keychain references: "keychain:key_name" -> actual value
            let raw_bot_token: String = row.get("bot_token");
            let bot_token = resolve_keychain_ref(&raw_bot_token);

            let raw_app_token: Option<String> = row.get("app_token");
            let app_token = raw_app_token.map(|t| resolve_keychain_ref(&t));

            bots.push(crate::traits::DynamicBot {
                id: row.get("id"),
                channel_type: row.get("channel_type"),
                bot_token,
                app_token,
                allowed_user_ids,
                extra_config: row.get("extra_config"),
                created_at: row.get("created_at"),
            });
        }
        Ok(bots)
    }

    async fn delete_dynamic_bot(&self, id: i64) -> anyhow::Result<()> {
        // Clean up keychain entries for this bot
        let _ = crate::config::delete_from_keychain(&format!("dynamic_bot_{}_bot_token", id));
        let _ = crate::config::delete_from_keychain(&format!("dynamic_bot_{}_app_token", id));

        sqlx::query("DELETE FROM dynamic_bots WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ==================== Session Channel Mapping ====================

    async fn save_session_channel(
        &self,
        session_id: &str,
        channel_name: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO session_channels (session_id, channel_name, updated_at)
             VALUES (?, ?, datetime('now'))
             ON CONFLICT(session_id) DO UPDATE SET
                channel_name = excluded.channel_name,
                updated_at = excluded.updated_at",
        )
        .bind(session_id)
        .bind(channel_name)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn load_session_channels(&self) -> anyhow::Result<Vec<(String, String)>> {
        let rows = sqlx::query("SELECT session_id, channel_name FROM session_channels")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .iter()
            .map(|r| {
                (
                    r.get::<String, _>("session_id"),
                    r.get::<String, _>("channel_name"),
                )
            })
            .collect())
    }

    // ==================== Dynamic Skills ====================

    async fn add_dynamic_skill(&self, skill: &crate::traits::DynamicSkill) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO dynamic_skills (name, description, triggers_json, body, source, source_url, enabled, version, resources_json, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))"
        )
        .bind(&skill.name)
        .bind(&skill.description)
        .bind(&skill.triggers_json)
        .bind(&skill.body)
        .bind(&skill.source)
        .bind(&skill.source_url)
        .bind(skill.enabled)
        .bind(&skill.version)
        .bind(&skill.resources_json)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn get_dynamic_skills(&self) -> anyhow::Result<Vec<crate::traits::DynamicSkill>> {
        let rows = sqlx::query(
            "SELECT id, name, description, triggers_json, body, source, source_url, enabled, version, resources_json, created_at
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
                resources_json: row
                    .try_get::<String, _>("resources_json")
                    .unwrap_or_else(|_| "[]".to_string()),
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

    async fn get_promotable_procedures(
        &self,
        min_success: i32,
        min_rate: f32,
    ) -> anyhow::Result<Vec<crate::traits::Procedure>> {
        let rows = sqlx::query(
            "SELECT id, name, trigger_pattern, steps, success_count, failure_count,
                    avg_duration_secs, last_used_at, created_at, updated_at
             FROM procedures
             WHERE success_count >= ?
               AND CAST(success_count AS REAL) / CAST(success_count + failure_count AS REAL) >= ?
             ORDER BY success_count DESC",
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
                last_used_at: row
                    .get::<Option<String>, _>("last_used_at")
                    .and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                                .ok()
                                .map(|n| n.and_utc().into())
                        })
                    })
                    .map(|d| d.with_timezone(&chrono::Utc)),
                created_at: row
                    .get::<Option<String>, _>("created_at")
                    .and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                                .ok()
                                .map(|n| n.and_utc().into())
                        })
                    })
                    .map(|d| d.with_timezone(&chrono::Utc))
                    .unwrap_or_else(chrono::Utc::now),
                updated_at: row
                    .get::<Option<String>, _>("updated_at")
                    .and_then(|s| {
                        chrono::DateTime::parse_from_rfc3339(&s).ok().or_else(|| {
                            chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                                .ok()
                                .map(|n| n.and_utc().into())
                        })
                    })
                    .map(|d| d.with_timezone(&chrono::Utc))
                    .unwrap_or_else(chrono::Utc::now),
            });
        }
        Ok(procedures)
    }

    // ==================== Skill Drafts ====================

    async fn add_skill_draft(&self, draft: &crate::traits::SkillDraft) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO skill_drafts (name, description, triggers_json, body, source_procedure, status, created_at)
             VALUES (?, ?, ?, ?, ?, 'pending', datetime('now'))",
        )
        .bind(&draft.name)
        .bind(&draft.description)
        .bind(&draft.triggers_json)
        .bind(&draft.body)
        .bind(&draft.source_procedure)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn get_pending_skill_drafts(&self) -> anyhow::Result<Vec<crate::traits::SkillDraft>> {
        let rows = sqlx::query(
            "SELECT id, name, description, triggers_json, body, source_procedure, status, created_at
             FROM skill_drafts WHERE status = 'pending' ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut drafts = Vec::new();
        for row in rows {
            drafts.push(crate::traits::SkillDraft {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                description: row.get::<String, _>("description"),
                triggers_json: row.get::<String, _>("triggers_json"),
                body: row.get::<String, _>("body"),
                source_procedure: row.get::<String, _>("source_procedure"),
                status: row.get::<String, _>("status"),
                created_at: row.get::<String, _>("created_at"),
            });
        }
        Ok(drafts)
    }

    async fn get_skill_draft(&self, id: i64) -> anyhow::Result<Option<crate::traits::SkillDraft>> {
        let row = sqlx::query(
            "SELECT id, name, description, triggers_json, body, source_procedure, status, created_at
             FROM skill_drafts WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|row| crate::traits::SkillDraft {
            id: row.get::<i64, _>("id"),
            name: row.get::<String, _>("name"),
            description: row.get::<String, _>("description"),
            triggers_json: row.get::<String, _>("triggers_json"),
            body: row.get::<String, _>("body"),
            source_procedure: row.get::<String, _>("source_procedure"),
            status: row.get::<String, _>("status"),
            created_at: row.get::<String, _>("created_at"),
        }))
    }

    async fn update_skill_draft_status(&self, id: i64, status: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE skill_drafts SET status = ? WHERE id = ?")
            .bind(status)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn skill_draft_exists_for_procedure(&self, procedure_name: &str) -> anyhow::Result<bool> {
        let row = sqlx::query(
            "SELECT COUNT(*) as cnt FROM skill_drafts WHERE source_procedure = ? AND status = 'pending'",
        )
        .bind(procedure_name)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("cnt") > 0)
    }

    // ==================== Dynamic MCP Servers ====================

    async fn save_dynamic_mcp_server(
        &self,
        server: &crate::traits::DynamicMcpServer,
    ) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO dynamic_mcp_servers (name, command, args_json, env_keys_json, triggers_json, enabled, created_at)
             VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(name) DO UPDATE SET command=excluded.command, args_json=excluded.args_json,
             env_keys_json=excluded.env_keys_json, triggers_json=excluded.triggers_json, enabled=excluded.enabled"
        )
        .bind(&server.name)
        .bind(&server.command)
        .bind(&server.args_json)
        .bind(&server.env_keys_json)
        .bind(&server.triggers_json)
        .bind(server.enabled)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn list_dynamic_mcp_servers(
        &self,
    ) -> anyhow::Result<Vec<crate::traits::DynamicMcpServer>> {
        let rows = sqlx::query(
            "SELECT id, name, command, args_json, env_keys_json, triggers_json, enabled, created_at
             FROM dynamic_mcp_servers ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut servers = Vec::new();
        for row in rows {
            servers.push(crate::traits::DynamicMcpServer {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                command: row.get::<String, _>("command"),
                args_json: row.get::<String, _>("args_json"),
                env_keys_json: row.get::<String, _>("env_keys_json"),
                triggers_json: row.get::<String, _>("triggers_json"),
                enabled: row.get::<bool, _>("enabled"),
                created_at: row.get::<String, _>("created_at"),
            });
        }
        Ok(servers)
    }

    async fn delete_dynamic_mcp_server(&self, id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dynamic_mcp_servers WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_dynamic_mcp_server(
        &self,
        server: &crate::traits::DynamicMcpServer,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE dynamic_mcp_servers SET command = ?, args_json = ?, env_keys_json = ?, triggers_json = ?, enabled = ? WHERE id = ?"
        )
        .bind(&server.command)
        .bind(&server.args_json)
        .bind(&server.env_keys_json)
        .bind(&server.triggers_json)
        .bind(server.enabled)
        .bind(server.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ==================== Dynamic CLI Agents ====================

    async fn save_dynamic_cli_agent(
        &self,
        agent: &crate::traits::DynamicCliAgent,
    ) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO dynamic_cli_agents (name, command, args_json, description, timeout_secs, max_output_chars, enabled, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
             ON CONFLICT(name) DO UPDATE SET command=excluded.command, args_json=excluded.args_json,
             description=excluded.description, timeout_secs=excluded.timeout_secs,
             max_output_chars=excluded.max_output_chars, enabled=excluded.enabled",
        )
        .bind(&agent.name)
        .bind(&agent.command)
        .bind(&agent.args_json)
        .bind(&agent.description)
        .bind(agent.timeout_secs.map(|v| v as i64))
        .bind(agent.max_output_chars.map(|v| v as i64))
        .bind(agent.enabled)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn list_dynamic_cli_agents(&self) -> anyhow::Result<Vec<crate::traits::DynamicCliAgent>> {
        let rows = sqlx::query(
            "SELECT id, name, command, args_json, description, timeout_secs, max_output_chars, enabled, created_at
             FROM dynamic_cli_agents ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut agents = Vec::new();
        for row in rows {
            agents.push(crate::traits::DynamicCliAgent {
                id: row.get::<i64, _>("id"),
                name: row.get::<String, _>("name"),
                command: row.get::<String, _>("command"),
                args_json: row.get::<String, _>("args_json"),
                description: row.get::<String, _>("description"),
                timeout_secs: row.get::<Option<i64>, _>("timeout_secs").map(|v| v as u64),
                max_output_chars: row
                    .get::<Option<i64>, _>("max_output_chars")
                    .map(|v| v as usize),
                enabled: row.get::<bool, _>("enabled"),
                created_at: row.get::<String, _>("created_at"),
            });
        }
        Ok(agents)
    }

    async fn delete_dynamic_cli_agent(&self, id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dynamic_cli_agents WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_dynamic_cli_agent(
        &self,
        agent: &crate::traits::DynamicCliAgent,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE dynamic_cli_agents SET command = ?, args_json = ?, description = ?, timeout_secs = ?, max_output_chars = ?, enabled = ? WHERE id = ?",
        )
        .bind(&agent.command)
        .bind(&agent.args_json)
        .bind(&agent.description)
        .bind(agent.timeout_secs.map(|v| v as i64))
        .bind(agent.max_output_chars.map(|v| v as i64))
        .bind(agent.enabled)
        .bind(agent.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn log_cli_agent_start(
        &self,
        session_id: &str,
        agent_name: &str,
        prompt_summary: &str,
        working_dir: Option<&str>,
    ) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "INSERT INTO cli_agent_invocations (session_id, agent_name, prompt_summary, working_dir, started_at)
             VALUES (?, ?, ?, ?, datetime('now'))",
        )
        .bind(session_id)
        .bind(agent_name)
        .bind(prompt_summary)
        .bind(working_dir)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn log_cli_agent_complete(
        &self,
        id: i64,
        exit_code: Option<i32>,
        output_summary: &str,
        success: bool,
        duration_secs: f64,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE cli_agent_invocations SET completed_at = datetime('now'), exit_code = ?, output_summary = ?, success = ?, duration_secs = ? WHERE id = ?",
        )
        .bind(exit_code)
        .bind(output_summary)
        .bind(success)
        .bind(duration_secs)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_cli_agent_invocations(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::traits::CliAgentInvocation>> {
        let rows = sqlx::query(
            "SELECT id, session_id, agent_name, prompt_summary, working_dir, started_at, completed_at, exit_code, output_summary, success, duration_secs
             FROM cli_agent_invocations ORDER BY started_at DESC LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut invocations = Vec::new();
        for row in rows {
            invocations.push(crate::traits::CliAgentInvocation {
                id: row.get::<i64, _>("id"),
                session_id: row.get::<String, _>("session_id"),
                agent_name: row.get::<String, _>("agent_name"),
                prompt_summary: row.get::<String, _>("prompt_summary"),
                working_dir: row.get::<Option<String>, _>("working_dir"),
                started_at: row.get::<String, _>("started_at"),
                completed_at: row.get::<Option<String>, _>("completed_at"),
                exit_code: row.get::<Option<i32>, _>("exit_code"),
                output_summary: row.get::<Option<String>, _>("output_summary"),
                success: row.get::<Option<bool>, _>("success"),
                duration_secs: row.get::<Option<f64>, _>("duration_secs"),
            });
        }
        Ok(invocations)
    }

    // ==================== Settings ====================

    async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
        let row = sqlx::query("SELECT value FROM settings WHERE key = ?")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get::<String, _>("value")))
    }

    async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        )
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ==================== People ====================

    async fn upsert_person(&self, person: &crate::traits::Person) -> anyhow::Result<i64> {
        let aliases_json = serde_json::to_string(&person.aliases)?;
        let platform_ids_json = serde_json::to_string(&person.platform_ids)?;
        let now = chrono::Utc::now().to_rfc3339();

        if person.id > 0 {
            sqlx::query(
                "UPDATE people SET name = ?, aliases_json = ?, relationship = ?, platform_ids_json = ?, \
                 notes = ?, communication_style = ?, language_preference = ?, updated_at = ? WHERE id = ?"
            )
            .bind(&person.name)
            .bind(&aliases_json)
            .bind(&person.relationship)
            .bind(&platform_ids_json)
            .bind(&person.notes)
            .bind(&person.communication_style)
            .bind(&person.language_preference)
            .bind(&now)
            .bind(person.id)
            .execute(&self.pool)
            .await?;
            Ok(person.id)
        } else {
            // Use INSERT OR IGNORE to handle the unique index on LOWER(name).
            // If a duplicate exists, the INSERT silently does nothing and we
            // return the existing row's id.
            sqlx::query(
                "INSERT OR IGNORE INTO people (name, aliases_json, relationship, platform_ids_json, notes, \
                 communication_style, language_preference, created_at, updated_at) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&person.name)
            .bind(&aliases_json)
            .bind(&person.relationship)
            .bind(&platform_ids_json)
            .bind(&person.notes)
            .bind(&person.communication_style)
            .bind(&person.language_preference)
            .bind(&now)
            .bind(&now)
            .execute(&self.pool)
            .await?;

            // Fetch the id — works whether we just inserted or the row already existed
            let row = sqlx::query("SELECT id FROM people WHERE LOWER(name) = ?")
                .bind(person.name.to_lowercase())
                .fetch_one(&self.pool)
                .await?;
            Ok(row.get::<i64, _>("id"))
        }
    }

    async fn get_person(&self, id: i64) -> anyhow::Result<Option<crate::traits::Person>> {
        let row = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Self::row_to_person(&r)))
    }

    async fn get_person_by_platform_id(
        &self,
        platform_id: &str,
    ) -> anyhow::Result<Option<crate::traits::Person>> {
        // Search platform_ids_json for a key matching the platform_id
        // SQLite json_each lets us iterate JSON object keys
        let row = sqlx::query(
            "SELECT p.id, p.name, p.aliases_json, p.relationship, p.platform_ids_json, p.notes, \
             p.communication_style, p.language_preference, p.last_interaction_at, p.interaction_count, \
             p.created_at, p.updated_at \
             FROM people p, json_each(p.platform_ids_json) j \
             WHERE j.key = ?"
        )
        .bind(platform_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| Self::row_to_person(&r)))
    }

    async fn find_person_by_name(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<crate::traits::Person>> {
        let name_lower = name.to_lowercase();
        // Check name first (case-insensitive)
        let row = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people WHERE LOWER(name) = ?",
        )
        .bind(&name_lower)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(r) = row {
            return Ok(Some(Self::row_to_person(&r)));
        }

        // Check aliases (JSON array search)
        let rows = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people",
        )
        .fetch_all(&self.pool)
        .await?;

        for r in &rows {
            let aliases_str: String = r.get("aliases_json");
            if let Ok(aliases) = serde_json::from_str::<Vec<String>>(&aliases_str) {
                if aliases.iter().any(|a| a.to_lowercase() == name_lower) {
                    return Ok(Some(Self::row_to_person(r)));
                }
            }
        }

        Ok(None)
    }

    async fn get_all_people(&self) -> anyhow::Result<Vec<crate::traits::Person>> {
        let rows = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people ORDER BY name ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(Self::row_to_person).collect())
    }

    async fn delete_person(&self, id: i64) -> anyhow::Result<()> {
        // person_facts has ON DELETE CASCADE, but be explicit
        sqlx::query("DELETE FROM person_facts WHERE person_id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        sqlx::query("DELETE FROM people WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn link_platform_id(
        &self,
        person_id: i64,
        platform_id: &str,
        display_name: &str,
    ) -> anyhow::Result<()> {
        // Read current platform_ids, add new one, write back
        let row = sqlx::query("SELECT platform_ids_json FROM people WHERE id = ?")
            .bind(person_id)
            .fetch_optional(&self.pool)
            .await?;

        let mut ids: std::collections::HashMap<String, String> = match row {
            Some(r) => {
                let json_str: String = r.get("platform_ids_json");
                serde_json::from_str(&json_str).unwrap_or_default()
            }
            None => return Err(anyhow::anyhow!("Person {} not found", person_id)),
        };

        ids.insert(platform_id.to_string(), display_name.to_string());
        let updated_json = serde_json::to_string(&ids)?;
        let now = chrono::Utc::now().to_rfc3339();

        sqlx::query("UPDATE people SET platform_ids_json = ?, updated_at = ? WHERE id = ?")
            .bind(&updated_json)
            .bind(&now)
            .bind(person_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn touch_person_interaction(&self, person_id: i64) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE people SET last_interaction_at = ?, interaction_count = interaction_count + 1, updated_at = ? WHERE id = ?"
        )
        .bind(&now)
        .bind(&now)
        .bind(person_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn upsert_person_fact(
        &self,
        person_id: i64,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        confidence: f32,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO person_facts (person_id, category, key, value, source, confidence, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(person_id, category, key) DO UPDATE SET \
             value = excluded.value, source = excluded.source, confidence = excluded.confidence, updated_at = excluded.updated_at"
        )
        .bind(person_id)
        .bind(category)
        .bind(key)
        .bind(value)
        .bind(source)
        .bind(confidence)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_person_facts(
        &self,
        person_id: i64,
        category: Option<&str>,
    ) -> anyhow::Result<Vec<crate::traits::PersonFact>> {
        let rows = if let Some(cat) = category {
            sqlx::query(
                "SELECT id, person_id, category, key, value, source, confidence, created_at, updated_at \
                 FROM person_facts WHERE person_id = ? AND category = ? ORDER BY category, key"
            )
            .bind(person_id)
            .bind(cat)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT id, person_id, category, key, value, source, confidence, created_at, updated_at \
                 FROM person_facts WHERE person_id = ? ORDER BY category, key"
            )
            .bind(person_id)
            .fetch_all(&self.pool)
            .await?
        };

        Ok(rows.iter().map(Self::row_to_person_fact).collect())
    }

    async fn delete_person_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM person_facts WHERE id = ?")
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn confirm_person_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query("UPDATE person_facts SET confidence = 1.0, source = 'owner', updated_at = ? WHERE id = ?")
            .bind(&now)
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_people_with_upcoming_dates(
        &self,
        within_days: i32,
    ) -> anyhow::Result<Vec<(crate::traits::Person, crate::traits::PersonFact)>> {
        // Get all birthday/important_date facts, then filter by upcoming date in Rust
        let rows = sqlx::query(
            "SELECT pf.id as fact_id, pf.person_id, pf.category, pf.key, pf.value, pf.source, pf.confidence, \
             pf.created_at as fact_created, pf.updated_at as fact_updated, \
             p.id, p.name, p.aliases_json, p.relationship, p.platform_ids_json, p.notes, \
             p.communication_style, p.language_preference, p.last_interaction_at, p.interaction_count, \
             p.created_at, p.updated_at \
             FROM person_facts pf JOIN people p ON pf.person_id = p.id \
             WHERE pf.category IN ('birthday', 'important_date')"
        )
        .fetch_all(&self.pool)
        .await?;

        let today = chrono::Utc::now().date_naive();
        let mut results = Vec::new();

        for r in &rows {
            let value: String = r.get("value");
            // Try to parse month-day from various formats (e.g., "March 15", "03-15", "2000-03-15")
            if let Some(upcoming_in) = days_until_date(&value, today) {
                if upcoming_in <= within_days as i64 && upcoming_in >= 0 {
                    let person = Self::row_to_person(r);
                    let fact = crate::traits::PersonFact {
                        id: r.get("fact_id"),
                        person_id: r.get("person_id"),
                        category: r.get("category"),
                        key: r.get("key"),
                        value: r.get("value"),
                        source: r.get("source"),
                        confidence: r.get("confidence"),
                        created_at: parse_dt(r.get::<String, _>("fact_created")),
                        updated_at: parse_dt(r.get::<String, _>("fact_updated")),
                    };
                    results.push((person, fact));
                }
            }
        }

        Ok(results)
    }

    async fn prune_stale_person_facts(&self, retention_days: u32) -> anyhow::Result<u64> {
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::days(retention_days as i64)).to_rfc3339();
        let result = sqlx::query(
            "DELETE FROM person_facts WHERE source = 'consolidation' AND confidence < 1.0 AND updated_at < ?"
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn get_people_needing_reconnect(
        &self,
        inactive_days: u32,
    ) -> anyhow::Result<Vec<crate::traits::Person>> {
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::days(inactive_days as i64)).to_rfc3339();
        let rows = sqlx::query(
            "SELECT id, name, aliases_json, relationship, platform_ids_json, notes, \
             communication_style, language_preference, last_interaction_at, interaction_count, \
             created_at, updated_at FROM people \
             WHERE last_interaction_at IS NOT NULL AND last_interaction_at < ? \
             AND relationship IN ('friend', 'family', 'coworker') \
             ORDER BY last_interaction_at ASC",
        )
        .bind(&cutoff)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(Self::row_to_person).collect())
    }

    // ==================== OAuth Connection Methods ====================

    async fn save_oauth_connection(
        &self,
        conn: &crate::traits::OAuthConnection,
    ) -> anyhow::Result<i64> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "INSERT INTO oauth_connections (service, auth_type, username, scopes, token_expires_at, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(service) DO UPDATE SET \
             auth_type = excluded.auth_type, username = excluded.username, scopes = excluded.scopes, \
             token_expires_at = excluded.token_expires_at, updated_at = excluded.updated_at",
        )
        .bind(&conn.service)
        .bind(&conn.auth_type)
        .bind(&conn.username)
        .bind(&conn.scopes)
        .bind(&conn.token_expires_at)
        .bind(&now)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(result.last_insert_rowid())
    }

    async fn get_oauth_connection(
        &self,
        service: &str,
    ) -> anyhow::Result<Option<crate::traits::OAuthConnection>> {
        let row = sqlx::query(
            "SELECT id, service, auth_type, username, scopes, token_expires_at, created_at, updated_at \
             FROM oauth_connections WHERE service = ?",
        )
        .bind(service)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| crate::traits::OAuthConnection {
            id: r.get("id"),
            service: r.get("service"),
            auth_type: r.get("auth_type"),
            username: r.try_get("username").unwrap_or(None),
            scopes: r.get("scopes"),
            token_expires_at: r.try_get("token_expires_at").unwrap_or(None),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
        }))
    }

    async fn list_oauth_connections(&self) -> anyhow::Result<Vec<crate::traits::OAuthConnection>> {
        let rows = sqlx::query(
            "SELECT id, service, auth_type, username, scopes, token_expires_at, created_at, updated_at \
             FROM oauth_connections ORDER BY service ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| crate::traits::OAuthConnection {
                id: r.get("id"),
                service: r.get("service"),
                auth_type: r.get("auth_type"),
                username: r.try_get("username").unwrap_or(None),
                scopes: r.get("scopes"),
                token_expires_at: r.try_get("token_expires_at").unwrap_or(None),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }

    async fn delete_oauth_connection(&self, service: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM oauth_connections WHERE service = ?")
            .bind(service)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn update_oauth_token_expiry(
        &self,
        service: &str,
        expires_at: Option<&str>,
    ) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE oauth_connections SET token_expires_at = ?, updated_at = ? WHERE service = ?",
        )
        .bind(expires_at)
        .bind(&now)
        .bind(service)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // ==================== V3 Orchestration Methods ====================

    async fn create_goal_v3(&self, goal: &GoalV3) -> anyhow::Result<()> {
        // Enforce hard cap of 10 active evergreen goals
        if goal.goal_type == "continuous" {
            let count = self.count_active_evergreen_goals().await?;
            if count >= 10 {
                anyhow::bail!(
                    "Cannot create evergreen goal: hard cap of 10 active evergreen goals reached (current: {})",
                    count
                );
            }
        }

        sqlx::query(
            "INSERT INTO goals_v3 (id, description, goal_type, status, priority, conditions,
             schedule, context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&goal.id)
        .bind(&goal.description)
        .bind(&goal.goal_type)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&goal.conditions)
        .bind(&goal.schedule)
        .bind(&goal.context)
        .bind(&goal.resources)
        .bind(goal.budget_per_check)
        .bind(goal.budget_daily)
        .bind(goal.tokens_used_today)
        .bind(&goal.last_useful_action)
        .bind(&goal.created_at)
        .bind(&goal.updated_at)
        .bind(&goal.completed_at)
        .bind(&goal.parent_goal_id)
        .bind(&goal.session_id)
        .bind(&goal.notified_at)
        .bind(goal.notification_attempts)
        .bind(goal.dispatch_failures)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_goal_v3(&self, id: &str) -> anyhow::Result<Option<GoalV3>> {
        let row = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3 WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| GoalV3 {
            id: r.get("id"),
            description: r.get("description"),
            goal_type: r.get("goal_type"),
            status: r.get("status"),
            priority: r.get("priority"),
            conditions: r.get("conditions"),
            schedule: r.get("schedule"),
            context: r.get("context"),
            resources: r.get("resources"),
            budget_per_check: r.get("budget_per_check"),
            budget_daily: r.get("budget_daily"),
            tokens_used_today: r.get("tokens_used_today"),
            last_useful_action: r.get("last_useful_action"),
            created_at: r.get("created_at"),
            updated_at: r.get("updated_at"),
            completed_at: r.get("completed_at"),
            parent_goal_id: r.get("parent_goal_id"),
            session_id: r.get("session_id"),
            notified_at: r.get("notified_at"),
            notification_attempts: r.get::<i32, _>("notification_attempts"),
            dispatch_failures: r.get::<i32, _>("dispatch_failures"),
        }))
    }

    async fn update_goal_v3(&self, goal: &GoalV3) -> anyhow::Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "UPDATE goals_v3 SET description = ?, goal_type = ?, status = ?, priority = ?,
             conditions = ?, schedule = ?, context = ?, resources = ?,
             budget_per_check = ?, budget_daily = ?, tokens_used_today = ?,
             last_useful_action = ?, updated_at = ?, completed_at = ?,
             parent_goal_id = ?, session_id = ?, notified_at = ?, notification_attempts = ?, dispatch_failures = ?
             WHERE id = ?",
        )
        .bind(&goal.description)
        .bind(&goal.goal_type)
        .bind(&goal.status)
        .bind(&goal.priority)
        .bind(&goal.conditions)
        .bind(&goal.schedule)
        .bind(&goal.context)
        .bind(&goal.resources)
        .bind(goal.budget_per_check)
        .bind(goal.budget_daily)
        .bind(goal.tokens_used_today)
        .bind(&goal.last_useful_action)
        .bind(&now)
        .bind(&goal.completed_at)
        .bind(&goal.parent_goal_id)
        .bind(&goal.session_id)
        .bind(&goal.notified_at)
        .bind(goal.notification_attempts)
        .bind(goal.dispatch_failures)
        .bind(&goal.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_active_goals_v3(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3 WHERE status IN ('active', 'pending')
             ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn get_goals_for_session_v3(&self, session_id: &str) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3 WHERE session_id = ?
             ORDER BY created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn migrate_legacy_scheduled_tasks_to_v3(&self) -> anyhow::Result<u64> {
        let rows = sqlx::query(
            "SELECT id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused,
                    last_run_at, next_run_at, created_at, updated_at
             FROM scheduled_tasks
             ORDER BY created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut migrated = 0u64;
        let now_local = chrono::Local::now();
        let now_rfc3339 = chrono::Utc::now().to_rfc3339();

        for r in &rows {
            let legacy_id: String = r.get("id");
            let legacy_name: String = r.get("name");
            let legacy_cron: String = r.get("cron_expr");
            let legacy_original_schedule: String = r.get("original_schedule");
            let legacy_prompt: String = r.get("prompt");
            let legacy_source: String = r.get("source");
            let legacy_is_oneshot: bool = r.get::<i64, _>("is_oneshot") != 0;
            let legacy_is_paused: bool = r.get::<i64, _>("is_paused") != 0;
            let legacy_last_run: Option<String> = r.get("last_run_at");
            let legacy_next_run: String = r.get("next_run_at");

            // Deterministic migrated goal ID keeps migration idempotent.
            let migrated_goal_id = format!("legacy-sched-{}", legacy_id);
            if self.get_goal_v3(&migrated_goal_id).await?.is_some() {
                continue;
            }

            let description = if !legacy_prompt.trim().is_empty() {
                legacy_prompt.trim().to_string()
            } else {
                legacy_name.clone()
            };

            let schedule_for_goal = if legacy_is_oneshot {
                // Legacy one-shot rows carry exact next_run_at. Convert to an
                // absolute one-shot cron expression in system-local timezone.
                let target_local = parse_legacy_datetime_to_local(&legacy_next_run)
                    .unwrap_or_else(|| now_local + chrono::Duration::minutes(1));
                let effective_target = if target_local <= now_local {
                    // Catch up overdue legacy one-shots quickly.
                    now_local + chrono::Duration::minutes(1)
                } else {
                    target_local
                };
                format!(
                    "{} {} {} {} *",
                    effective_target.minute(),
                    effective_target.hour(),
                    effective_target.day(),
                    effective_target.month()
                )
            } else {
                legacy_cron.clone()
            };

            let mut goal = if legacy_is_oneshot {
                GoalV3::new_finite(&description, "system")
            } else {
                GoalV3::new_continuous(
                    &description,
                    "system",
                    &schedule_for_goal,
                    Some(5000),
                    Some(20000),
                )
            };

            goal.id = migrated_goal_id;
            goal.schedule = Some(schedule_for_goal.clone());
            goal.status = if legacy_is_paused {
                "paused".to_string()
            } else {
                "active".to_string()
            };

            // Preserve historical timing where possible. If legacy timestamps
            // are non-RFC3339, keep goal timestamps in canonical RFC3339 now.
            goal.created_at = now_rfc3339.clone();
            goal.updated_at = now_rfc3339.clone();
            goal.last_useful_action = legacy_last_run
                .as_deref()
                .and_then(parse_legacy_datetime_to_local)
                .map(|dt| dt.with_timezone(&chrono::Utc).to_rfc3339());
            goal.context = Some(
                serde_json::json!({
                    "migrated_from": "scheduled_tasks",
                    "legacy_task_id": legacy_id,
                    "legacy_name": legacy_name,
                    "legacy_source": legacy_source,
                    "legacy_original_schedule": legacy_original_schedule,
                    "legacy_next_run_at": legacy_next_run,
                })
                .to_string(),
            );

            match self.create_goal_v3(&goal).await {
                Ok(()) => migrated += 1,
                Err(e) => {
                    tracing::warn!(
                        legacy_task_id = %legacy_id,
                        error = %e,
                        "Failed to migrate legacy scheduled task to V3 goal"
                    );
                }
            }
        }

        Ok(migrated)
    }

    async fn get_pending_confirmation_goals(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE session_id = ? AND status = 'pending_confirmation'
             ORDER BY created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn activate_goal_v3(&self, goal_id: &str) -> anyhow::Result<bool> {
        let goal_row = sqlx::query(
            "SELECT goal_type
             FROM goals_v3
             WHERE id = ? AND status = 'pending_confirmation'",
        )
        .bind(goal_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = goal_row else {
            return Ok(false);
        };

        let goal_type: String = row.get("goal_type");
        if goal_type == "continuous" {
            let active_evergreen = self.count_active_evergreen_goals().await?;
            if active_evergreen >= 10 {
                anyhow::bail!(
                    "Cannot activate recurring goal: hard cap of 10 active evergreen goals reached (current: {})",
                    active_evergreen
                );
            }
        }

        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE goals_v3
             SET status = 'active', updated_at = ?
             WHERE id = ? AND status = 'pending_confirmation'",
        )
        .bind(&now)
        .bind(goal_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn create_task_v3(&self, task: &TaskV3) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO tasks_v3 (id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&task.id)
        .bind(&task.goal_id)
        .bind(&task.description)
        .bind(&task.status)
        .bind(&task.priority)
        .bind(task.task_order)
        .bind(&task.parallel_group)
        .bind(&task.depends_on)
        .bind(&task.agent_id)
        .bind(&task.context)
        .bind(&task.result)
        .bind(&task.error)
        .bind(&task.blocker)
        .bind(task.idempotent as i32)
        .bind(task.retry_count)
        .bind(task.max_retries)
        .bind(&task.created_at)
        .bind(&task.started_at)
        .bind(&task.completed_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_task_v3(&self, id: &str) -> anyhow::Result<Option<TaskV3>> {
        let row = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at
             FROM tasks_v3 WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| TaskV3 {
            id: r.get("id"),
            goal_id: r.get("goal_id"),
            description: r.get("description"),
            status: r.get("status"),
            priority: r.get("priority"),
            task_order: r.get("task_order"),
            parallel_group: r.get("parallel_group"),
            depends_on: r.get("depends_on"),
            agent_id: r.get("agent_id"),
            context: r.get("context"),
            result: r.get("result"),
            error: r.get("error"),
            blocker: r.get("blocker"),
            idempotent: r.get::<i32, _>("idempotent") != 0,
            retry_count: r.get("retry_count"),
            max_retries: r.get("max_retries"),
            created_at: r.get("created_at"),
            started_at: r.get("started_at"),
            completed_at: r.get("completed_at"),
        }))
    }

    async fn update_task_v3(&self, task: &TaskV3) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE tasks_v3 SET description = ?, status = ?, priority = ?, task_order = ?,
             parallel_group = ?, depends_on = ?, agent_id = ?, context = ?,
             result = ?, error = ?, blocker = ?, idempotent = ?,
             retry_count = ?, max_retries = ?, started_at = ?, completed_at = ?
             WHERE id = ?",
        )
        .bind(&task.description)
        .bind(&task.status)
        .bind(&task.priority)
        .bind(task.task_order)
        .bind(&task.parallel_group)
        .bind(&task.depends_on)
        .bind(&task.agent_id)
        .bind(&task.context)
        .bind(&task.result)
        .bind(&task.error)
        .bind(&task.blocker)
        .bind(task.idempotent as i32)
        .bind(task.retry_count)
        .bind(task.max_retries)
        .bind(&task.started_at)
        .bind(&task.completed_at)
        .bind(&task.id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_tasks_for_goal_v3(&self, goal_id: &str) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error, blocker,
             idempotent, retry_count, max_retries, created_at, started_at, completed_at
             FROM tasks_v3 WHERE goal_id = ?
             ORDER BY task_order ASC",
        )
        .bind(goal_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskV3 {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn count_completed_tasks_for_goal(&self, goal_id: &str) -> anyhow::Result<i64> {
        let row = sqlx::query(
            "SELECT COUNT(*) as cnt FROM tasks_v3
             WHERE goal_id = ? AND status IN ('completed', 'skipped')",
        )
        .bind(goal_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("cnt"))
    }

    async fn claim_task_v3(&self, task_id: &str, agent_id: &str) -> anyhow::Result<bool> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE tasks_v3 SET status = 'claimed', agent_id = ?, started_at = ?
             WHERE id = ? AND status = 'pending'",
        )
        .bind(agent_id)
        .bind(&now)
        .bind(task_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn log_task_activity_v3(&self, activity: &TaskActivityV3) -> anyhow::Result<()> {
        // Redact secrets from tool_args and result before persisting
        let redacted_args = activity
            .tool_args
            .as_deref()
            .map(crate::tools::sanitize::redact_secrets);
        let redacted_result = activity
            .result
            .as_deref()
            .map(crate::tools::sanitize::redact_secrets);

        sqlx::query(
            "INSERT INTO task_activity_v3 (task_id, activity_type, tool_name, tool_args,
             result, success, tokens_used, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&activity.task_id)
        .bind(&activity.activity_type)
        .bind(&activity.tool_name)
        .bind(&redacted_args)
        .bind(&redacted_result)
        .bind(activity.success.map(|b| b as i32))
        .bind(activity.tokens_used)
        .bind(&activity.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_task_activities_v3(&self, task_id: &str) -> anyhow::Result<Vec<TaskActivityV3>> {
        let rows = sqlx::query(
            "SELECT id, task_id, activity_type, tool_name, tool_args, result, success,
             tokens_used, created_at
             FROM task_activity_v3 WHERE task_id = ?
             ORDER BY created_at ASC",
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskActivityV3 {
                id: r.get("id"),
                task_id: r.get("task_id"),
                activity_type: r.get("activity_type"),
                tool_name: r.get("tool_name"),
                tool_args: r.get("tool_args"),
                result: r.get("result"),
                success: r.get::<Option<i32>, _>("success").map(|v| v != 0),
                tokens_used: r.get("tokens_used"),
                created_at: r.get("created_at"),
            })
            .collect())
    }

    async fn get_due_evergreen_goals(&self) -> anyhow::Result<Vec<GoalV3>> {
        // Get all active continuous goals
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE goal_type = 'continuous' AND status = 'active' AND schedule IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await?;

        let now = chrono::Local::now();
        let mut due_goals = Vec::new();

        for r in &rows {
            let goal = GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            };

            // Check if the schedule is due using croner
            let schedule_str = match &goal.schedule {
                Some(s) => s,
                None => continue,
            };

            // Parse cron expression
            let cron: croner::Cron = match schedule_str.parse() {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Determine last run time — use last_useful_action or created_at
            let last_run = goal
                .last_useful_action
                .as_deref()
                .or(Some(goal.created_at.as_str()));
            let last_run_dt = match last_run {
                Some(ts) => match chrono::DateTime::parse_from_rfc3339(ts) {
                    Ok(dt) => dt.with_timezone(&chrono::Local),
                    Err(_) => continue,
                },
                None => continue,
            };

            // Check if there's a scheduled time between last_run and now
            if let Ok(next) = cron.find_next_occurrence(&last_run_dt, false) {
                if next <= now {
                    // Check daily budget
                    if let Some(daily_budget) = goal.budget_daily {
                        if goal.tokens_used_today >= daily_budget {
                            continue; // Over budget
                        }
                    }
                    due_goals.push(goal);
                }
            }
        }

        Ok(due_goals)
    }

    async fn get_due_scheduled_finite_goals(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE goal_type = 'finite' AND status = 'active' AND schedule IS NOT NULL",
        )
        .fetch_all(&self.pool)
        .await?;

        let now = chrono::Local::now();
        let mut due_goals = Vec::new();

        for r in &rows {
            let goal = GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            };

            let schedule_str = match &goal.schedule {
                Some(s) => s,
                None => continue,
            };

            let cron: croner::Cron = match schedule_str.parse() {
                Ok(c) => c,
                Err(_) => continue,
            };

            let anchor = goal
                .last_useful_action
                .as_deref()
                .or(Some(goal.created_at.as_str()));
            let anchor_dt = match anchor {
                Some(ts) => match chrono::DateTime::parse_from_rfc3339(ts) {
                    Ok(dt) => dt.with_timezone(&chrono::Local),
                    Err(_) => continue,
                },
                None => continue,
            };

            if let Ok(next) = cron.find_next_occurrence(&anchor_dt, false) {
                if next <= now {
                    due_goals.push(goal);
                }
            }
        }

        Ok(due_goals)
    }

    async fn cancel_stale_pending_confirmation_goals(
        &self,
        max_age_secs: i64,
    ) -> anyhow::Result<u64> {
        let cutoff = (chrono::Utc::now() - chrono::Duration::seconds(max_age_secs)).to_rfc3339();
        let result = sqlx::query(
            "UPDATE goals_v3
             SET status = 'cancelled', updated_at = datetime('now'), completed_at = datetime('now')
             WHERE status = 'pending_confirmation' AND created_at < ?",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn get_scheduled_goals_v3(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE schedule IS NOT NULL OR status = 'pending_confirmation'
             ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn reset_daily_token_budgets(&self) -> anyhow::Result<u64> {
        let result = sqlx::query(
            "UPDATE goals_v3 SET tokens_used_today = 0
             WHERE goal_type = 'continuous' AND status = 'active'",
        )
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    async fn add_goal_tokens_and_get_budget_status(
        &self,
        goal_id: &str,
        delta_tokens: i64,
    ) -> anyhow::Result<Option<GoalTokenBudgetStatus>> {
        let mut tx = self.pool.begin().await?;

        if delta_tokens != 0 {
            sqlx::query(
                "UPDATE goals_v3
                 SET tokens_used_today = MAX(0, tokens_used_today + ?)
                 WHERE id = ?",
            )
            .bind(delta_tokens)
            .bind(goal_id)
            .execute(&mut *tx)
            .await?;
        }

        let row = sqlx::query(
            "SELECT budget_per_check, budget_daily, tokens_used_today
             FROM goals_v3
             WHERE id = ?",
        )
        .bind(goal_id)
        .fetch_optional(&mut *tx)
        .await?;

        tx.commit().await?;

        Ok(row.map(|r| GoalTokenBudgetStatus {
            budget_per_check: r.get("budget_per_check"),
            budget_daily: r.get("budget_daily"),
            tokens_used_today: r.get("tokens_used_today"),
        }))
    }

    async fn get_conversation_summary(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ConversationSummary>> {
        let row = sqlx::query(
            "SELECT session_id, summary, message_count, last_message_id, updated_at
             FROM conversation_summaries WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|r| ConversationSummary {
            session_id: r.get("session_id"),
            summary: r.get("summary"),
            message_count: r.get::<i64, _>("message_count") as usize,
            last_message_id: r.get("last_message_id"),
            updated_at: r
                .get::<String, _>("updated_at")
                .parse::<DateTime<Utc>>()
                .unwrap_or_else(|_| Utc::now()),
        }))
    }

    async fn upsert_conversation_summary(
        &self,
        summary: &ConversationSummary,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO conversation_summaries (session_id, summary, message_count, last_message_id, updated_at)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&summary.session_id)
        .bind(&summary.summary)
        .bind(summary.message_count as i64)
        .bind(&summary.last_message_id)
        .bind(summary.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    async fn get_pending_tasks_by_priority(&self, limit: i64) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT t.id, t.goal_id, t.description, t.status, t.priority, t.task_order,
             t.parallel_group, t.depends_on, t.agent_id, t.context, t.result, t.error,
             t.blocker, t.idempotent, t.retry_count, t.max_retries, t.created_at,
             t.started_at, t.completed_at
             FROM tasks_v3 t
             JOIN goals_v3 g ON t.goal_id = g.id AND g.status = 'active'
             WHERE t.status = 'pending'
             AND NOT EXISTS (
                 SELECT 1 FROM json_each(COALESCE(t.depends_on, '[]')) AS dep
                 WHERE NOT EXISTS (
                     SELECT 1 FROM tasks_v3 d WHERE d.id = dep.value AND d.status = 'completed'
                 )
             )
             ORDER BY
                 CASE t.priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
                 t.task_order ASC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskV3 {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn get_stuck_tasks(&self, timeout_secs: i64) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error,
             blocker, idempotent, retry_count, max_retries, created_at,
             started_at, completed_at
             FROM tasks_v3
             WHERE status IN ('running', 'claimed')
             AND datetime(started_at) < datetime('now', '-' || ? || ' seconds')
             ORDER BY started_at ASC",
        )
        .bind(timeout_secs)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskV3 {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn get_recently_completed_tasks(&self, since: &str) -> anyhow::Result<Vec<TaskV3>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, description, status, priority, task_order,
             parallel_group, depends_on, agent_id, context, result, error,
             blocker, idempotent, retry_count, max_retries, created_at,
             started_at, completed_at
             FROM tasks_v3
             WHERE status = 'completed' AND completed_at > ?
             ORDER BY completed_at DESC",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| TaskV3 {
                id: r.get("id"),
                goal_id: r.get("goal_id"),
                description: r.get("description"),
                status: r.get("status"),
                priority: r.get("priority"),
                task_order: r.get("task_order"),
                parallel_group: r.get("parallel_group"),
                depends_on: r.get("depends_on"),
                agent_id: r.get("agent_id"),
                context: r.get("context"),
                result: r.get("result"),
                error: r.get("error"),
                blocker: r.get("blocker"),
                idempotent: r.get::<i32, _>("idempotent") != 0,
                retry_count: r.get("retry_count"),
                max_retries: r.get("max_retries"),
                created_at: r.get("created_at"),
                started_at: r.get("started_at"),
                completed_at: r.get("completed_at"),
            })
            .collect())
    }

    async fn mark_task_interrupted(&self, task_id: &str) -> anyhow::Result<bool> {
        let result = sqlx::query(
            "UPDATE tasks_v3 SET status = 'interrupted',
             completed_at = datetime('now')
             WHERE id = ? AND status IN ('running', 'claimed')",
        )
        .bind(task_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn count_active_evergreen_goals(&self) -> anyhow::Result<i64> {
        let row = sqlx::query(
            "SELECT COUNT(*) as cnt FROM goals_v3
             WHERE goal_type = 'continuous' AND status = 'active'",
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("cnt"))
    }

    async fn get_goals_needing_notification(&self) -> anyhow::Result<Vec<GoalV3>> {
        let rows = sqlx::query(
            "SELECT id, description, goal_type, status, priority, conditions, schedule,
             context, resources, budget_per_check, budget_daily, tokens_used_today,
             last_useful_action, created_at, updated_at, completed_at, parent_goal_id,
             session_id, notified_at, notification_attempts, dispatch_failures
             FROM goals_v3
             WHERE status IN ('completed', 'failed', 'stalled') AND notified_at IS NULL
             AND goal_type = 'finite' AND notification_attempts < 3",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .map(|r| GoalV3 {
                id: r.get("id"),
                description: r.get("description"),
                goal_type: r.get("goal_type"),
                status: r.get("status"),
                priority: r.get("priority"),
                conditions: r.get("conditions"),
                schedule: r.get("schedule"),
                context: r.get("context"),
                resources: r.get("resources"),
                budget_per_check: r.get("budget_per_check"),
                budget_daily: r.get("budget_daily"),
                tokens_used_today: r.get("tokens_used_today"),
                last_useful_action: r.get("last_useful_action"),
                created_at: r.get("created_at"),
                updated_at: r.get("updated_at"),
                completed_at: r.get("completed_at"),
                parent_goal_id: r.get("parent_goal_id"),
                session_id: r.get("session_id"),
                notified_at: r.get("notified_at"),
                notification_attempts: r.get::<i32, _>("notification_attempts"),
                dispatch_failures: r.get::<i32, _>("dispatch_failures"),
            })
            .collect())
    }

    async fn mark_goal_notified(&self, goal_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE goals_v3 SET notified_at = datetime('now') WHERE id = ?")
            .bind(goal_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn cleanup_stale_goals(&self, stale_hours: i64) -> anyhow::Result<(u64, u64)> {
        let cutoff = (chrono::Utc::now() - chrono::Duration::hours(stale_hours)).to_rfc3339();

        // Legacy goals table: active → abandoned
        let legacy = sqlx::query(
            "UPDATE goals SET status = 'abandoned', updated_at = datetime('now')
             WHERE status = 'active' AND updated_at < ?",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;

        // V3 finite goals: active/pending without schedule → failed.
        // Scheduled finite goals are handled by dedicated scheduler phases.
        let v3 = sqlx::query(
            "UPDATE goals_v3 SET status = 'failed',
                    updated_at = datetime('now'),
                    completed_at = datetime('now')
             WHERE status IN ('active', 'pending')
               AND goal_type = 'finite'
               AND schedule IS NULL
               AND updated_at < ?",
        )
        .bind(&cutoff)
        .execute(&self.pool)
        .await?;

        Ok((legacy.rows_affected(), v3.rows_affected()))
    }

    // --- Notification Queue ---

    async fn enqueue_notification(
        &self,
        entry: &crate::traits::NotificationEntry,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT INTO notification_queue (id, goal_id, session_id, notification_type, priority, message, created_at, delivered_at, attempts, expires_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&entry.id)
        .bind(&entry.goal_id)
        .bind(&entry.session_id)
        .bind(&entry.notification_type)
        .bind(&entry.priority)
        .bind(&entry.message)
        .bind(&entry.created_at)
        .bind(&entry.delivered_at)
        .bind(entry.attempts)
        .bind(&entry.expires_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_pending_notifications(
        &self,
        limit: i64,
    ) -> anyhow::Result<Vec<crate::traits::NotificationEntry>> {
        let rows = sqlx::query(
            "SELECT id, goal_id, session_id, notification_type, priority, message, created_at, delivered_at, attempts, expires_at
             FROM notification_queue
             WHERE delivered_at IS NULL
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
             ORDER BY
               CASE priority WHEN 'critical' THEN 0 ELSE 1 END ASC,
               created_at ASC
             LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let mut entries = Vec::with_capacity(rows.len());
        for row in &rows {
            entries.push(crate::traits::NotificationEntry {
                id: row.get("id"),
                goal_id: row.get("goal_id"),
                session_id: row.get("session_id"),
                notification_type: row.get("notification_type"),
                priority: row.get("priority"),
                message: row.get("message"),
                created_at: row.get("created_at"),
                delivered_at: row.get("delivered_at"),
                attempts: row.get("attempts"),
                expires_at: row.get("expires_at"),
            });
        }
        Ok(entries)
    }

    async fn mark_notification_delivered(&self, notification_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE notification_queue SET delivered_at = datetime('now') WHERE id = ?")
            .bind(notification_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn increment_notification_attempt(&self, notification_id: &str) -> anyhow::Result<()> {
        sqlx::query("UPDATE notification_queue SET attempts = attempts + 1 WHERE id = ?")
            .bind(notification_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn cleanup_expired_notifications(&self) -> anyhow::Result<i64> {
        let result = sqlx::query(
            "DELETE FROM notification_queue
             WHERE delivered_at IS NULL
               AND expires_at IS NOT NULL
               AND datetime(expires_at) <= datetime('now')",
        )
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::traits::{
        BehaviorPattern, DynamicBot, DynamicMcpServer, DynamicSkill, Episode, ErrorSolution, Goal,
        Message, Procedure, StateStore, TokenUsage,
    };
    use crate::types::FactPrivacy;
    use std::sync::Arc;

    async fn setup_test_store() -> (SqliteStateStore, tempfile::NamedTempFile) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap();
        (store, db_file)
    }

    async fn setup_test_store_with_cap(cap: usize) -> (SqliteStateStore, tempfile::NamedTempFile) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            cap,
            None,
            embedding_service,
        )
        .await
        .unwrap();
        (store, db_file)
    }

    fn make_message(session_id: &str, role: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        }
    }

    async fn create_events_table_for_tests(store: &SqliteStateStore) {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT,
                task_id TEXT,
                tool_name TEXT
            )
            "#,
        )
        .execute(&store.pool())
        .await
        .unwrap();
    }

    async fn insert_conversation_event_for_tests(
        store: &SqliteStateStore,
        session_id: &str,
        event_type: &str,
        data: serde_json::Value,
        created_at: chrono::DateTime<Utc>,
    ) {
        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind(event_type)
        .bind(data.to_string())
        .bind(created_at.to_rfc3339())
        .execute(&store.pool())
        .await
        .unwrap();
    }

    // ==================== Message Tests ====================

    #[tokio::test]
    async fn test_append_and_get_history() {
        let (store, _db) = setup_test_store().await;
        let session = "sess-1";

        let m1 = make_message(session, "user", "Hello");
        let m2 = make_message(session, "assistant", "Hi there");
        let m3 = make_message(session, "user", "How are you?");

        store.append_message(&m1).await.unwrap();
        store.append_message(&m2).await.unwrap();
        store.append_message(&m3).await.unwrap();

        let history = store.get_history(session, 100).await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].content.as_deref(), Some("Hello"));
        assert_eq!(history[1].content.as_deref(), Some("Hi there"));
        assert_eq!(history[2].content.as_deref(), Some("How are you?"));
    }

    #[tokio::test]
    async fn test_get_history_limit() {
        let (store, _db) = setup_test_store().await;
        let session = "sess-limit";

        for i in 0..10 {
            let msg = make_message(session, "user", &format!("Message {}", i));
            store.append_message(&msg).await.unwrap();
        }

        let history = store.get_history(session, 5).await.unwrap();
        assert_eq!(history.len(), 5);
        // The truncate_with_anchor logic preserves the first user message,
        // so the last message should be the most recent one
        assert_eq!(
            history.last().unwrap().content.as_deref(),
            Some("Message 9")
        );
    }

    #[tokio::test]
    async fn test_session_isolation() {
        let (store, _db) = setup_test_store().await;

        let m_a = make_message("session_a", "user", "From A");
        let m_b = make_message("session_b", "user", "From B");

        store.append_message(&m_a).await.unwrap();
        store.append_message(&m_b).await.unwrap();

        let history_a = store.get_history("session_a", 100).await.unwrap();
        let history_b = store.get_history("session_b", 100).await.unwrap();

        assert_eq!(history_a.len(), 1);
        assert_eq!(history_b.len(), 1);
        assert_eq!(history_a[0].content.as_deref(), Some("From A"));
        assert_eq!(history_b[0].content.as_deref(), Some("From B"));
    }

    #[tokio::test]
    async fn test_clear_session() {
        let (store, _db) = setup_test_store().await;
        let session = "sess-clear";

        store
            .append_message(&make_message(session, "user", "Hi"))
            .await
            .unwrap();
        store
            .append_message(&make_message(session, "assistant", "Hello"))
            .await
            .unwrap();

        let before = store.get_history(session, 100).await.unwrap();
        assert_eq!(before.len(), 2);

        store.clear_session(session).await.unwrap();

        let after = store.get_history(session, 100).await.unwrap();
        assert_eq!(after.len(), 0);
    }

    #[tokio::test]
    async fn test_working_memory_cap() {
        let (store, _db) = setup_test_store_with_cap(5).await;
        let session = "sess-cap";

        for i in 0..10 {
            let msg = make_message(session, "user", &format!("Msg {}", i));
            store.append_message(&msg).await.unwrap();
        }

        let history = store.get_history(session, 100).await.unwrap();
        assert!(
            history.len() <= 5,
            "Expected <= 5 messages in working memory, got {}",
            history.len()
        );
    }

    #[tokio::test]
    async fn test_get_history_hydrates_from_events_when_available() {
        let (store, _db) = setup_test_store().await;
        let session = "sess-events";
        let now = Utc::now();

        create_events_table_for_tests(&store).await;
        insert_conversation_event_for_tests(
            &store,
            session,
            "user_message",
            serde_json::json!({
                "content": "Hello from events",
                "message_id": "msg-user-1",
                "has_attachments": false
            }),
            now,
        )
        .await;
        insert_conversation_event_for_tests(
            &store,
            session,
            "assistant_response",
            serde_json::json!({
                "content": "Hi from assistant event",
                "message_id": "msg-assistant-1",
                "model": "test",
                "tool_calls": []
            }),
            now + chrono::Duration::seconds(1),
        )
        .await;
        insert_conversation_event_for_tests(
            &store,
            session,
            "tool_result",
            serde_json::json!({
                "message_id": "msg-tool-1",
                "tool_call_id": "tc-1",
                "name": "terminal",
                "result": "ok",
                "success": true,
                "duration_ms": 7,
                "error": null,
                "task_id": null
            }),
            now + chrono::Duration::seconds(2),
        )
        .await;

        let history = store.get_history(session, 100).await.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].id, "msg-user-1");
        assert_eq!(history[0].role, "user");
        assert_eq!(history[1].id, "msg-assistant-1");
        assert_eq!(history[1].role, "assistant");
        assert_eq!(history[2].id, "msg-tool-1");
        assert_eq!(history[2].role, "tool");
        assert_eq!(history[2].tool_call_id.as_deref(), Some("tc-1"));
    }

    #[tokio::test]
    async fn test_get_context_falls_back_to_history_when_messages_table_missing() {
        let (store, _db) = setup_test_store().await;
        let session = "sess-context-fallback";
        let now = Utc::now();

        create_events_table_for_tests(&store).await;
        insert_conversation_event_for_tests(
            &store,
            session,
            "user_message",
            serde_json::json!({
                "content": "Context from events",
                "message_id": "msg-context-1",
                "has_attachments": false
            }),
            now,
        )
        .await;

        sqlx::query("DROP TABLE messages")
            .execute(&store.pool())
            .await
            .unwrap();

        let context = store.get_context(session, "context", 10).await.unwrap();
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].id, "msg-context-1");
        assert_eq!(context[0].role, "user");
        assert_eq!(context[0].content.as_deref(), Some("Context from events"));
    }

    // ==================== Fact Tests ====================

    #[tokio::test]
    async fn test_upsert_fact_insert() {
        let (store, _db) = setup_test_store().await;

        store
            .upsert_fact(
                "preference",
                "language",
                "Rust",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let facts = store.get_facts(Some("preference")).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, "preference");
        assert_eq!(facts[0].key, "language");
        assert_eq!(facts[0].value, "Rust");
    }

    #[tokio::test]
    async fn test_upsert_fact_supersede() {
        let (store, _db) = setup_test_store().await;

        store
            .upsert_fact(
                "preference",
                "editor",
                "vim",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // Upserting the same category/key with the same value should succeed
        // (it updates timestamp/source rather than inserting a new row).
        store
            .upsert_fact(
                "preference",
                "editor",
                "vim",
                "observation",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let facts = store.get_facts(Some("preference")).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, "vim");
        // Source should be updated
        assert_eq!(facts[0].source, "observation");
    }

    #[tokio::test]
    async fn test_get_facts_by_category() {
        let (store, _db) = setup_test_store().await;

        store
            .upsert_fact("pref", "color", "blue", "user", None, FactPrivacy::Global)
            .await
            .unwrap();
        store
            .upsert_fact("info", "name", "Alice", "user", None, FactPrivacy::Global)
            .await
            .unwrap();
        store
            .upsert_fact("pref", "food", "pizza", "user", None, FactPrivacy::Global)
            .await
            .unwrap();

        let pref_facts = store.get_facts(Some("pref")).await.unwrap();
        assert_eq!(pref_facts.len(), 2);
        for f in &pref_facts {
            assert_eq!(f.category, "pref");
        }

        let info_facts = store.get_facts(Some("info")).await.unwrap();
        assert_eq!(info_facts.len(), 1);
        assert_eq!(info_facts[0].key, "name");

        let all_facts = store.get_facts(None).await.unwrap();
        assert_eq!(all_facts.len(), 3);
    }

    #[tokio::test]
    async fn test_delete_fact_soft_delete() {
        let (store, _db) = setup_test_store().await;

        store
            .upsert_fact(
                "temp",
                "item",
                "delete-me",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let facts = store.get_facts(Some("temp")).await.unwrap();
        assert_eq!(facts.len(), 1);
        let fact_id = facts[0].id;

        store.delete_fact(fact_id).await.unwrap();

        let after = store.get_facts(Some("temp")).await.unwrap();
        assert_eq!(after.len(), 0);
    }

    #[tokio::test]
    async fn test_increment_fact_recall() {
        let (store, _db) = setup_test_store().await;

        store
            .upsert_fact(
                "test",
                "recall_key",
                "recall_val",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let facts = store.get_facts(Some("test")).await.unwrap();
        let fact_id = facts[0].id;
        assert_eq!(facts[0].recall_count, 0);
        assert!(facts[0].last_recalled_at.is_none());

        store.increment_fact_recall(fact_id).await.unwrap();
        store.increment_fact_recall(fact_id).await.unwrap();

        let updated = store.get_facts(Some("test")).await.unwrap();
        assert_eq!(updated[0].recall_count, 2);
        assert!(updated[0].last_recalled_at.is_some());
    }

    #[tokio::test]
    async fn test_fact_privacy_channel_scoped() {
        let (store, _db) = setup_test_store().await;

        store
            .upsert_fact(
                "context",
                "project",
                "aidaemon",
                "user",
                Some("slack:C12345"),
                FactPrivacy::Channel,
            )
            .await
            .unwrap();

        let facts = store.get_facts(Some("context")).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].channel_id.as_deref(), Some("slack:C12345"));
        assert_eq!(facts[0].privacy, FactPrivacy::Channel);
    }

    #[tokio::test]
    async fn test_update_fact_privacy() {
        let (store, _db) = setup_test_store().await;

        store
            .upsert_fact(
                "secret",
                "api_key_hint",
                "starts with sk-",
                "user",
                Some("slack:C999"),
                FactPrivacy::Channel,
            )
            .await
            .unwrap();

        let facts = store.get_facts(Some("secret")).await.unwrap();
        assert_eq!(facts[0].privacy, FactPrivacy::Channel);
        let fact_id = facts[0].id;

        store
            .update_fact_privacy(fact_id, FactPrivacy::Global)
            .await
            .unwrap();

        let updated = store.get_facts(Some("secret")).await.unwrap();
        assert_eq!(updated[0].privacy, FactPrivacy::Global);
    }

    // ==================== Episode Tests ====================

    #[tokio::test]
    async fn test_insert_and_get_episodes() {
        let (store, _db) = setup_test_store().await;

        let episode = Episode {
            id: 0,
            session_id: "ep-sess".to_string(),
            summary: "We discussed Rust async patterns".to_string(),
            topics: Some(vec!["rust".to_string(), "async".to_string()]),
            emotional_tone: Some("curious".to_string()),
            outcome: Some("learned tokio basics".to_string()),
            importance: 0.8,
            recall_count: 0,
            last_recalled_at: None,
            message_count: 12,
            start_time: Utc::now() - chrono::Duration::hours(1),
            end_time: Utc::now(),
            created_at: Utc::now(),
            channel_id: None,
        };

        let ep_id = store.insert_episode(&episode).await.unwrap();
        assert!(ep_id > 0);

        let episodes = store.get_recent_episodes(10).await.unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].summary, "We discussed Rust async patterns");
        assert_eq!(episodes[0].message_count, 12);
        assert_eq!(
            episodes[0].topics,
            Some(vec!["rust".to_string(), "async".to_string()])
        );
    }

    #[tokio::test]
    async fn test_increment_episode_recall() {
        let (store, _db) = setup_test_store().await;

        let episode = Episode {
            id: 0,
            session_id: "ep-recall".to_string(),
            summary: "Recall test episode".to_string(),
            topics: None,
            emotional_tone: None,
            outcome: None,
            importance: 0.5,
            recall_count: 0,
            last_recalled_at: None,
            message_count: 5,
            start_time: Utc::now(),
            end_time: Utc::now(),
            created_at: Utc::now(),
            channel_id: None,
        };

        let ep_id = store.insert_episode(&episode).await.unwrap();

        store.increment_episode_recall(ep_id).await.unwrap();
        store.increment_episode_recall(ep_id).await.unwrap();

        let episodes = store.get_recent_episodes(10).await.unwrap();
        assert_eq!(episodes[0].recall_count, 2);
        assert!(episodes[0].last_recalled_at.is_some());
    }

    #[tokio::test]
    async fn test_backfill_episode_embeddings() {
        let (store, _db) = setup_test_store().await;

        let episode = Episode {
            id: 0,
            session_id: "ep-embed".to_string(),
            summary: "An episode about machine learning".to_string(),
            topics: None,
            emotional_tone: None,
            outcome: None,
            importance: 0.5,
            recall_count: 0,
            last_recalled_at: None,
            message_count: 3,
            start_time: Utc::now(),
            end_time: Utc::now(),
            created_at: Utc::now(),
            channel_id: None,
        };

        store.insert_episode(&episode).await.unwrap();

        // Episodes are inserted without embeddings
        let backfilled = store.backfill_episode_embeddings().await.unwrap();
        assert_eq!(backfilled, 1);

        // Running again should backfill 0 since all have embeddings now
        let backfilled_again = store.backfill_episode_embeddings().await.unwrap();
        assert_eq!(backfilled_again, 0);
    }

    // ==================== Goal Tests ====================

    #[tokio::test]
    async fn test_insert_and_get_active_goals() {
        let (store, _db) = setup_test_store().await;

        let goal = Goal {
            id: 0,
            description: "Learn Rust generics".to_string(),
            status: "active".to_string(),
            priority: "high".to_string(),
            progress_notes: None,
            source_episode_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            completed_at: None,
        };

        let goal_id = store.insert_goal(&goal).await.unwrap();
        assert!(goal_id > 0);

        let active = store.get_active_goals().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].description, "Learn Rust generics");
        assert_eq!(active[0].status, "active");
        assert_eq!(active[0].priority, "high");
    }

    #[tokio::test]
    async fn test_update_goal_status() {
        let (store, _db) = setup_test_store().await;

        let goal = Goal {
            id: 0,
            description: "Finish project".to_string(),
            status: "active".to_string(),
            priority: "medium".to_string(),
            progress_notes: None,
            source_episode_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            completed_at: None,
        };

        let goal_id = store.insert_goal(&goal).await.unwrap();

        store
            .update_goal(goal_id, Some("completed"), None)
            .await
            .unwrap();

        let active = store.get_active_goals().await.unwrap();
        assert_eq!(
            active.len(),
            0,
            "Completed goal should not appear in active goals"
        );
    }

    #[tokio::test]
    async fn test_update_goal_progress_note() {
        let (store, _db) = setup_test_store().await;

        let goal = Goal {
            id: 0,
            description: "Write tests".to_string(),
            status: "active".to_string(),
            priority: "low".to_string(),
            progress_notes: None,
            source_episode_id: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            completed_at: None,
        };

        let goal_id = store.insert_goal(&goal).await.unwrap();

        store
            .update_goal(goal_id, None, Some("Added 5 unit tests"))
            .await
            .unwrap();
        store
            .update_goal(goal_id, None, Some("Added 10 more tests"))
            .await
            .unwrap();

        let goals = store.get_active_goals().await.unwrap();
        assert_eq!(goals.len(), 1);
        let notes = goals[0].progress_notes.as_ref().unwrap();
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0], "Added 5 unit tests");
        assert_eq!(notes[1], "Added 10 more tests");
    }

    // ==================== User Profile Tests ====================

    #[tokio::test]
    async fn test_default_user_profile() {
        let (store, _db) = setup_test_store().await;

        let profile = store.get_user_profile().await.unwrap();
        assert_eq!(profile.verbosity_preference, "medium");
        assert_eq!(profile.explanation_depth, "moderate");
        assert_eq!(profile.tone_preference, "neutral");
        assert_eq!(profile.emoji_preference, "none");
        assert!(profile.asks_before_acting);
        assert!(profile.prefers_explanations);
        // likes_suggestions defaults to false in the code
        assert!(!profile.likes_suggestions);
    }

    #[tokio::test]
    async fn test_update_user_profile() {
        let (store, _db) = setup_test_store().await;

        // First call creates the default profile
        let mut profile = store.get_user_profile().await.unwrap();

        profile.verbosity_preference = "brief".to_string();
        profile.tone_preference = "casual".to_string();
        profile.emoji_preference = "frequent".to_string();
        profile.asks_before_acting = false;

        store.update_user_profile(&profile).await.unwrap();

        let updated = store.get_user_profile().await.unwrap();
        assert_eq!(updated.verbosity_preference, "brief");
        assert_eq!(updated.tone_preference, "casual");
        assert_eq!(updated.emoji_preference, "frequent");
        assert!(!updated.asks_before_acting);
        // Unchanged fields should remain
        assert_eq!(updated.explanation_depth, "moderate");
        assert!(updated.prefers_explanations);
    }

    // ==================== Behavior Pattern Tests ====================

    #[tokio::test]
    async fn test_insert_and_get_behavior_patterns() {
        let (store, _db) = setup_test_store().await;

        let pattern = BehaviorPattern {
            id: 0,
            pattern_type: "habit".to_string(),
            description: "Always runs tests after code changes".to_string(),
            trigger_context: Some("code modification".to_string()),
            action: Some("cargo test".to_string()),
            confidence: 0.7,
            occurrence_count: 3,
            last_seen_at: Some(Utc::now()),
            created_at: Utc::now(),
        };

        let pat_id = store.insert_behavior_pattern(&pattern).await.unwrap();
        assert!(pat_id > 0);

        let patterns = store.get_behavior_patterns(0.5).await.unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(
            patterns[0].description,
            "Always runs tests after code changes"
        );
        assert_eq!(patterns[0].confidence, 0.7);
        assert_eq!(patterns[0].occurrence_count, 3);

        // With a higher min_confidence threshold, it should not be returned
        let filtered = store.get_behavior_patterns(0.9).await.unwrap();
        assert_eq!(filtered.len(), 0);
    }

    #[tokio::test]
    async fn test_update_behavior_pattern_confidence() {
        let (store, _db) = setup_test_store().await;

        let pattern = BehaviorPattern {
            id: 0,
            pattern_type: "trigger".to_string(),
            description: "Checks git status before committing".to_string(),
            trigger_context: None,
            action: None,
            confidence: 0.5,
            occurrence_count: 1,
            last_seen_at: None,
            created_at: Utc::now(),
        };

        let pat_id = store.insert_behavior_pattern(&pattern).await.unwrap();

        store.update_behavior_pattern(pat_id, 0.1).await.unwrap();

        let patterns = store.get_behavior_patterns(0.0).await.unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].occurrence_count, 2);
        assert!((patterns[0].confidence - 0.6).abs() < 0.01);
        assert!(patterns[0].last_seen_at.is_some());
    }

    // ==================== Procedure Tests ====================

    #[tokio::test]
    async fn test_insert_and_get_procedures() {
        let (store, _db) = setup_test_store().await;

        let procedure = Procedure {
            id: 0,
            name: "deploy-app".to_string(),
            trigger_pattern: "deploy the application".to_string(),
            steps: vec![
                "cargo build --release".to_string(),
                "scp target/release/app server:".to_string(),
                "ssh server systemctl restart app".to_string(),
            ],
            success_count: 1,
            failure_count: 0,
            avg_duration_secs: Some(30.0),
            last_used_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let proc_id = store.upsert_procedure(&procedure).await.unwrap();
        assert!(proc_id > 0);

        let procs = store.get_relevant_procedures("deploy", 10).await.unwrap();
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].name, "deploy-app");
        assert_eq!(procs[0].steps.len(), 3);
    }

    #[tokio::test]
    async fn test_procedure_success_count_increments() {
        let (store, _db) = setup_test_store().await;

        let procedure = Procedure {
            id: 0,
            name: "run-tests".to_string(),
            trigger_pattern: "run the test suite".to_string(),
            steps: vec!["cargo test".to_string()],
            success_count: 1,
            failure_count: 0,
            avg_duration_secs: None,
            last_used_at: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        store.upsert_procedure(&procedure).await.unwrap();

        // Upsert again with the same name triggers ON CONFLICT DO UPDATE
        // which increments success_count
        store.upsert_procedure(&procedure).await.unwrap();

        let procs = store.get_relevant_procedures("test", 10).await.unwrap();
        assert_eq!(procs.len(), 1);
        assert!(
            procs[0].success_count >= 2,
            "Expected success_count >= 2 after upsert conflict, got {}",
            procs[0].success_count
        );
    }

    // ==================== Error Solution Tests ====================

    #[tokio::test]
    async fn test_insert_and_get_error_solutions() {
        let (store, _db) = setup_test_store().await;

        let solution = ErrorSolution {
            id: 0,
            error_pattern: "connection refused on port 5432".to_string(),
            domain: Some("database".to_string()),
            solution_summary: "Start the PostgreSQL service".to_string(),
            solution_steps: Some(vec![
                "sudo systemctl start postgresql".to_string(),
                "verify with pg_isready".to_string(),
            ]),
            success_count: 1,
            failure_count: 0,
            last_used_at: None,
            created_at: Utc::now(),
        };

        let sol_id = store.insert_error_solution(&solution).await.unwrap();
        assert!(sol_id > 0);

        let solutions = store
            .get_relevant_error_solutions("connection refused", 10)
            .await
            .unwrap();
        assert_eq!(solutions.len(), 1);
        assert_eq!(
            solutions[0].solution_summary,
            "Start the PostgreSQL service"
        );
        assert_eq!(solutions[0].domain.as_deref(), Some("database"));
    }

    #[tokio::test]
    async fn test_update_error_solution_outcome() {
        let (store, _db) = setup_test_store().await;

        let solution = ErrorSolution {
            id: 0,
            error_pattern: "file not found".to_string(),
            domain: None,
            solution_summary: "Check the file path".to_string(),
            solution_steps: None,
            success_count: 0,
            failure_count: 0,
            last_used_at: None,
            created_at: Utc::now(),
        };

        let sol_id = store.insert_error_solution(&solution).await.unwrap();

        // Record a success
        store.update_error_solution(sol_id, true).await.unwrap();
        // Record a failure
        store.update_error_solution(sol_id, false).await.unwrap();
        // Record another success
        store.update_error_solution(sol_id, true).await.unwrap();

        let solutions = store
            .get_relevant_error_solutions("file not found", 10)
            .await
            .unwrap();
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].success_count, 2);
        assert_eq!(solutions[0].failure_count, 1);
    }

    // ==================== Token Usage Tests ====================

    #[tokio::test]
    async fn test_record_and_get_token_usage() {
        let (store, _db) = setup_test_store().await;

        let usage = TokenUsage {
            model: "gpt-4".to_string(),
            input_tokens: 100,
            output_tokens: 50,
        };

        store
            .record_token_usage("token-sess", &usage)
            .await
            .unwrap();

        // Use a date in the past to capture all records
        let records = store
            .get_token_usage_since("2000-01-01T00:00:00Z")
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].model, "gpt-4");
        assert_eq!(records[0].input_tokens, 100);
        assert_eq!(records[0].output_tokens, 50);
    }

    #[tokio::test]
    async fn test_token_usage_since_filter() {
        let (store, _db) = setup_test_store().await;

        let usage1 = TokenUsage {
            model: "gpt-4".to_string(),
            input_tokens: 100,
            output_tokens: 50,
        };
        let usage2 = TokenUsage {
            model: "gpt-3.5".to_string(),
            input_tokens: 200,
            output_tokens: 80,
        };

        store.record_token_usage("sess-1", &usage1).await.unwrap();
        store.record_token_usage("sess-2", &usage2).await.unwrap();

        // A far-past date should return all records
        let all = store
            .get_token_usage_since("2000-01-01T00:00:00Z")
            .await
            .unwrap();
        assert_eq!(all.len(), 2);

        // A far-future date should return no records
        let none = store
            .get_token_usage_since("2099-01-01T00:00:00Z")
            .await
            .unwrap();
        assert_eq!(none.len(), 0);
    }

    // ==================== Dynamic Bot Tests ====================

    #[tokio::test]
    async fn test_dynamic_bots_crud() {
        let (store, _db) = setup_test_store().await;

        let bot = DynamicBot {
            id: 0,
            channel_type: "telegram".to_string(),
            bot_token: "123456:ABC".to_string(),
            app_token: None,
            allowed_user_ids: vec!["user1".to_string(), "user2".to_string()],
            extra_config: "{}".to_string(),
            created_at: String::new(),
        };

        // Add
        let bot_id = store.add_dynamic_bot(&bot).await.unwrap();
        assert!(bot_id > 0);

        // List
        let bots = store.get_dynamic_bots().await.unwrap();
        assert_eq!(bots.len(), 1);
        assert_eq!(bots[0].channel_type, "telegram");
        assert_eq!(bots[0].bot_token, "123456:ABC");
        assert_eq!(bots[0].allowed_user_ids.len(), 2);

        // Delete
        store.delete_dynamic_bot(bot_id).await.unwrap();

        let after = store.get_dynamic_bots().await.unwrap();
        assert_eq!(after.len(), 0);
    }

    // ==================== Dynamic Skill Tests ====================

    #[tokio::test]
    async fn test_dynamic_skills_crud() {
        let (store, _db) = setup_test_store().await;

        let skill = DynamicSkill {
            id: 0,
            name: "code-review".to_string(),
            description: "Review code for best practices".to_string(),
            triggers_json: r#"["review","code review"]"#.to_string(),
            body: "# Code Review\nCheck for...\n".to_string(),
            source: "inline".to_string(),
            source_url: None,
            enabled: true,
            version: Some("1.0".to_string()),
            created_at: String::new(),
            resources_json: "[]".to_string(),
        };

        // Add
        let skill_id = store.add_dynamic_skill(&skill).await.unwrap();
        assert!(skill_id > 0);

        // List
        let skills = store.get_dynamic_skills().await.unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "code-review");
        assert!(skills[0].enabled);

        // Disable
        store
            .update_dynamic_skill_enabled(skill_id, false)
            .await
            .unwrap();
        let skills = store.get_dynamic_skills().await.unwrap();
        assert!(!skills[0].enabled);

        // Re-enable
        store
            .update_dynamic_skill_enabled(skill_id, true)
            .await
            .unwrap();
        let skills = store.get_dynamic_skills().await.unwrap();
        assert!(skills[0].enabled);

        // Delete
        store.delete_dynamic_skill(skill_id).await.unwrap();
        let skills = store.get_dynamic_skills().await.unwrap();
        assert_eq!(skills.len(), 0);
    }

    // ==================== Dynamic MCP Server Tests ====================

    #[tokio::test]
    async fn test_dynamic_mcp_servers_crud() {
        let (store, _db) = setup_test_store().await;

        let server = DynamicMcpServer {
            id: 0,
            name: "test_server".to_string(),
            command: "npx".to_string(),
            args_json: r#"["@test/mcp-server"]"#.to_string(),
            env_keys_json: r#"["API_KEY"]"#.to_string(),
            triggers_json: r#"["test","testing"]"#.to_string(),
            enabled: true,
            created_at: String::new(),
        };

        // Save
        let server_id = store.save_dynamic_mcp_server(&server).await.unwrap();
        assert!(server_id > 0);

        // List
        let servers = store.list_dynamic_mcp_servers().await.unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "test_server");
        assert_eq!(servers[0].command, "npx");
        assert!(servers[0].enabled);

        // Update
        let mut updated_server = servers[0].clone();
        updated_server.command = "uvx".to_string();
        updated_server.enabled = false;
        store
            .update_dynamic_mcp_server(&updated_server)
            .await
            .unwrap();

        let servers = store.list_dynamic_mcp_servers().await.unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].command, "uvx");
        assert!(!servers[0].enabled);

        // Delete
        store
            .delete_dynamic_mcp_server(updated_server.id)
            .await
            .unwrap();
        let servers = store.list_dynamic_mcp_servers().await.unwrap();
        assert_eq!(servers.len(), 0);
    }

    #[tokio::test]
    async fn test_oauth_connection_crud() {
        let (store, _tmp) = setup_test_store().await;

        // Insert
        let conn = crate::traits::OAuthConnection {
            id: 0,
            service: "twitter".to_string(),
            auth_type: "oauth2_pkce".to_string(),
            username: Some("@testuser".to_string()),
            scopes: r#"["tweet.read","tweet.write"]"#.to_string(),
            token_expires_at: Some("2025-12-31T00:00:00Z".to_string()),
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        let id = store.save_oauth_connection(&conn).await.unwrap();
        assert!(id > 0);

        // Get by service
        let fetched = store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.service, "twitter");
        assert_eq!(fetched.auth_type, "oauth2_pkce");
        assert_eq!(fetched.username, Some("@testuser".to_string()));

        // List all
        let all = store.list_oauth_connections().await.unwrap();
        assert_eq!(all.len(), 1);

        // Update expiry
        store
            .update_oauth_token_expiry("twitter", Some("2026-06-30T00:00:00Z"))
            .await
            .unwrap();
        let updated = store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            updated.token_expires_at,
            Some("2026-06-30T00:00:00Z".to_string())
        );

        // Upsert (same service, different data)
        let conn2 = crate::traits::OAuthConnection {
            id: 0,
            service: "twitter".to_string(),
            auth_type: "oauth2_pkce".to_string(),
            username: Some("@newuser".to_string()),
            scopes: r#"["tweet.read"]"#.to_string(),
            token_expires_at: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        store.save_oauth_connection(&conn2).await.unwrap();
        let upserted = store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(upserted.username, Some("@newuser".to_string()));
        // Still just 1 connection (upserted, not duplicated)
        assert_eq!(store.list_oauth_connections().await.unwrap().len(), 1);

        // Delete
        store.delete_oauth_connection("twitter").await.unwrap();
        assert!(store
            .get_oauth_connection("twitter")
            .await
            .unwrap()
            .is_none());
        assert_eq!(store.list_oauth_connections().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_oauth_connection_not_found() {
        let (store, _tmp) = setup_test_store().await;
        let result = store.get_oauth_connection("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Dynamic CLI Agent CRUD tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_dynamic_cli_agent_crud() {
        let (store, _db) = setup_test_store().await;

        // Initially empty
        let agents = store.list_dynamic_cli_agents().await.unwrap();
        assert!(agents.is_empty());

        // Save a new agent
        let agent = crate::traits::DynamicCliAgent {
            id: 0,
            name: "test-agent".to_string(),
            command: "echo".to_string(),
            args_json: r#"["hello"]"#.to_string(),
            description: "Test echo agent".to_string(),
            timeout_secs: Some(30),
            max_output_chars: Some(5000),
            enabled: true,
            created_at: String::new(),
        };
        let id = store.save_dynamic_cli_agent(&agent).await.unwrap();
        assert!(id > 0);

        // List should return it
        let agents = store.list_dynamic_cli_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "test-agent");
        assert_eq!(agents[0].command, "echo");
        assert_eq!(agents[0].args_json, r#"["hello"]"#);
        assert_eq!(agents[0].description, "Test echo agent");
        assert_eq!(agents[0].timeout_secs, Some(30));
        assert_eq!(agents[0].max_output_chars, Some(5000));
        assert!(agents[0].enabled);

        // Update the agent
        let mut updated = agents[0].clone();
        updated.command = "bash".to_string();
        updated.enabled = false;
        store.update_dynamic_cli_agent(&updated).await.unwrap();

        let agents = store.list_dynamic_cli_agents().await.unwrap();
        assert_eq!(agents[0].command, "bash");
        assert!(!agents[0].enabled);

        // Delete the agent
        store.delete_dynamic_cli_agent(updated.id).await.unwrap();
        let agents = store.list_dynamic_cli_agents().await.unwrap();
        assert!(agents.is_empty());
    }

    #[tokio::test]
    async fn test_dynamic_cli_agent_upsert() {
        let (store, _db) = setup_test_store().await;

        let agent = crate::traits::DynamicCliAgent {
            id: 0,
            name: "upsert-agent".to_string(),
            command: "echo".to_string(),
            args_json: "[]".to_string(),
            description: "v1".to_string(),
            timeout_secs: None,
            max_output_chars: None,
            enabled: true,
            created_at: String::new(),
        };
        store.save_dynamic_cli_agent(&agent).await.unwrap();

        // Save again with same name — should upsert
        let agent2 = crate::traits::DynamicCliAgent {
            id: 0,
            name: "upsert-agent".to_string(),
            command: "bash".to_string(),
            args_json: r#"["-c"]"#.to_string(),
            description: "v2".to_string(),
            timeout_secs: Some(60),
            max_output_chars: Some(10000),
            enabled: true,
            created_at: String::new(),
        };
        store.save_dynamic_cli_agent(&agent2).await.unwrap();

        // Should still be 1 agent (upserted)
        let agents = store.list_dynamic_cli_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].command, "bash");
        assert_eq!(agents[0].description, "v2");
    }

    // -----------------------------------------------------------------------
    // CLI Agent Invocation logging tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_cli_agent_invocation_logging() {
        let (store, _db) = setup_test_store().await;

        // Initially empty
        let invocations = store.get_cli_agent_invocations(10).await.unwrap();
        assert!(invocations.is_empty());

        // Log a start
        let inv_id = store
            .log_cli_agent_start(
                "session1",
                "claude",
                "Create a website",
                Some("/tmp/project"),
            )
            .await
            .unwrap();
        assert!(inv_id > 0);

        // Should appear in list
        let invocations = store.get_cli_agent_invocations(10).await.unwrap();
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].session_id, "session1");
        assert_eq!(invocations[0].agent_name, "claude");
        assert_eq!(invocations[0].prompt_summary, "Create a website");
        assert_eq!(invocations[0].working_dir, Some("/tmp/project".to_string()));
        assert!(invocations[0].success.is_none()); // Not completed yet

        // Log completion
        store
            .log_cli_agent_complete(inv_id, Some(0), "Website created successfully", true, 45.5)
            .await
            .unwrap();

        // Should be updated
        let invocations = store.get_cli_agent_invocations(10).await.unwrap();
        assert_eq!(invocations.len(), 1);
        assert_eq!(invocations[0].exit_code, Some(0));
        assert_eq!(invocations[0].success, Some(true));
        assert_eq!(invocations[0].duration_secs, Some(45.5));
        assert_eq!(
            invocations[0].output_summary,
            Some("Website created successfully".to_string())
        );
    }

    #[tokio::test]
    async fn test_cli_agent_invocation_limit() {
        let (store, _db) = setup_test_store().await;

        // Log 5 invocations
        for i in 0..5 {
            store
                .log_cli_agent_start("session1", "claude", &format!("Task {}", i), None)
                .await
                .unwrap();
        }

        // Limit=3 should return exactly 3
        let invocations = store.get_cli_agent_invocations(3).await.unwrap();
        assert_eq!(invocations.len(), 3);

        // All 5 should be retrievable
        let all = store.get_cli_agent_invocations(100).await.unwrap();
        assert_eq!(all.len(), 5);
    }

    #[tokio::test]
    async fn test_cli_agent_invocation_failure() {
        let (store, _db) = setup_test_store().await;

        let inv_id = store
            .log_cli_agent_start("session1", "gemini", "Debug crash", None)
            .await
            .unwrap();

        // Log failure
        store
            .log_cli_agent_complete(inv_id, Some(1), "Error: command not found", false, 2.1)
            .await
            .unwrap();

        let invocations = store.get_cli_agent_invocations(10).await.unwrap();
        assert_eq!(invocations[0].exit_code, Some(1));
        assert_eq!(invocations[0].success, Some(false));
    }

    // ==================== V3 Orchestration Tests ====================

    #[tokio::test]
    async fn test_goals_v3_crud() {
        let (store, _file) = setup_test_store().await;

        // Create
        let goal = crate::traits::GoalV3::new_finite("Build a website", "session_1");
        store.create_goal_v3(&goal).await.unwrap();

        // Get
        let fetched = store.get_goal_v3(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.description, "Build a website");
        assert_eq!(fetched.status, "active");
        assert_eq!(fetched.goal_type, "finite");
        assert_eq!(fetched.session_id, "session_1");

        // Update
        let mut updated = fetched;
        updated.status = "completed".to_string();
        updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
        store.update_goal_v3(&updated).await.unwrap();

        let fetched2 = store.get_goal_v3(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched2.status, "completed");
        assert!(fetched2.completed_at.is_some());

        // Active goals should not include completed
        let active = store.get_active_goals_v3().await.unwrap();
        assert!(active.is_empty());
    }

    #[tokio::test]
    async fn test_goals_v3_session_filter() {
        let (store, _file) = setup_test_store().await;

        let goal1 = crate::traits::GoalV3::new_finite("Task A", "session_1");
        let goal2 = crate::traits::GoalV3::new_finite("Task B", "session_2");
        let goal3 = crate::traits::GoalV3::new_finite("Task C", "session_1");

        store.create_goal_v3(&goal1).await.unwrap();
        store.create_goal_v3(&goal2).await.unwrap();
        store.create_goal_v3(&goal3).await.unwrap();

        let session1_goals = store.get_goals_for_session_v3("session_1").await.unwrap();
        assert_eq!(session1_goals.len(), 2);
        assert!(session1_goals.iter().all(|g| g.session_id == "session_1"));

        let session2_goals = store.get_goals_for_session_v3("session_2").await.unwrap();
        assert_eq!(session2_goals.len(), 1);
    }

    #[tokio::test]
    async fn test_tasks_v3_crud() {
        let (store, _file) = setup_test_store().await;

        // Create parent goal first (FK)
        let goal = crate::traits::GoalV3::new_finite("Parent goal", "session_1");
        store.create_goal_v3(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = crate::traits::TaskV3 {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Step 1: create files".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 3,
            created_at: now.clone(),
            started_at: None,
            completed_at: None,
        };
        store.create_task_v3(&task).await.unwrap();

        // Get
        let fetched = store.get_task_v3(&task.id).await.unwrap().unwrap();
        assert_eq!(fetched.description, "Step 1: create files");
        assert_eq!(fetched.status, "pending");
        assert!(fetched.idempotent);

        // Update
        let mut updated = fetched;
        updated.status = "completed".to_string();
        updated.result = Some("Files created successfully".to_string());
        updated.completed_at = Some(now.clone());
        store.update_task_v3(&updated).await.unwrap();

        let fetched2 = store.get_task_v3(&task.id).await.unwrap().unwrap();
        assert_eq!(fetched2.status, "completed");
        assert!(fetched2.result.is_some());

        // List for goal
        let tasks = store.get_tasks_for_goal_v3(&goal.id).await.unwrap();
        assert_eq!(tasks.len(), 1);
    }

    #[tokio::test]
    async fn test_claim_task_v3() {
        let (store, _file) = setup_test_store().await;

        let goal = crate::traits::GoalV3::new_finite("Goal", "session_1");
        store.create_goal_v3(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = crate::traits::TaskV3 {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Claimable task".to_string(),
            status: "pending".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        store.create_task_v3(&task).await.unwrap();

        // First claim succeeds
        let claimed = store.claim_task_v3(&task.id, "agent-1").await.unwrap();
        assert!(claimed);

        // Second claim fails (already claimed)
        let claimed2 = store.claim_task_v3(&task.id, "agent-2").await.unwrap();
        assert!(!claimed2);

        // Verify agent_id was set
        let fetched = store.get_task_v3(&task.id).await.unwrap().unwrap();
        assert_eq!(fetched.agent_id, Some("agent-1".to_string()));
        assert_eq!(fetched.status, "claimed");
    }

    #[tokio::test]
    async fn test_task_activity_v3_log() {
        let (store, _file) = setup_test_store().await;

        let goal = crate::traits::GoalV3::new_finite("Goal", "session_1");
        store.create_goal_v3(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = crate::traits::TaskV3 {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Task".to_string(),
            status: "running".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: Some("agent-1".to_string()),
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: None,
        };
        store.create_task_v3(&task).await.unwrap();

        // Log activity
        let activity = crate::traits::TaskActivityV3 {
            id: 0,
            task_id: task.id.clone(),
            activity_type: "tool_call".to_string(),
            tool_name: Some("terminal".to_string()),
            tool_args: Some("{\"command\":\"ls\"}".to_string()),
            result: Some("file1.txt\nfile2.txt".to_string()),
            success: Some(true),
            tokens_used: Some(42),
            created_at: now,
        };
        store.log_task_activity_v3(&activity).await.unwrap();

        let activities = store.get_task_activities_v3(&task.id).await.unwrap();
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].activity_type, "tool_call");
        assert_eq!(activities[0].tool_name, Some("terminal".to_string()));
        assert_eq!(activities[0].success, Some(true));
        assert_eq!(activities[0].tokens_used, Some(42));
    }

    // --- Notification Queue Tests ---

    #[tokio::test]
    async fn test_notification_queue_enqueue_and_fetch() {
        let (store, _file) = setup_test_store().await;

        let entry = crate::traits::NotificationEntry::new(
            "goal-1",
            "session-1",
            "completed",
            "Goal completed: build website",
        );
        store.enqueue_notification(&entry).await.unwrap();

        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].goal_id, "goal-1");
        assert_eq!(pending[0].notification_type, "completed");
        assert_eq!(pending[0].priority, "critical");
        assert!(pending[0].expires_at.is_none()); // critical = no expiry
        assert!(pending[0].delivered_at.is_none());
    }

    #[tokio::test]
    async fn test_notification_queue_status_update_has_expiry() {
        let (store, _file) = setup_test_store().await;

        let entry = crate::traits::NotificationEntry::new(
            "goal-1",
            "session-1",
            "progress",
            "Goal 50% complete",
        );
        assert_eq!(entry.priority, "status_update");
        assert!(entry.expires_at.is_some());

        store.enqueue_notification(&entry).await.unwrap();
        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].expires_at.is_some());
    }

    #[tokio::test]
    async fn test_notification_queue_mark_delivered() {
        let (store, _file) = setup_test_store().await;

        let entry = crate::traits::NotificationEntry::new(
            "goal-1",
            "session-1",
            "failed",
            "Goal failed: deployment error",
        );
        store.enqueue_notification(&entry).await.unwrap();

        // Mark as delivered
        store.mark_notification_delivered(&entry.id).await.unwrap();

        // Should no longer appear in pending
        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 0);
    }

    #[tokio::test]
    async fn test_notification_queue_priority_ordering() {
        let (store, _file) = setup_test_store().await;

        // Enqueue status_update first
        let status = crate::traits::NotificationEntry::new(
            "goal-1",
            "session-1",
            "progress",
            "Progress update",
        );
        store.enqueue_notification(&status).await.unwrap();

        // Enqueue critical second
        let critical =
            crate::traits::NotificationEntry::new("goal-2", "session-1", "failed", "Goal failed");
        store.enqueue_notification(&critical).await.unwrap();

        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 2);
        // Critical should come first despite being enqueued second
        assert_eq!(pending[0].priority, "critical");
        assert_eq!(pending[1].priority, "status_update");
    }

    #[tokio::test]
    async fn test_notification_queue_cleanup_expired() {
        let (store, _file) = setup_test_store().await;

        // Insert a notification with an already-expired expires_at
        let mut entry =
            crate::traits::NotificationEntry::new("goal-1", "session-1", "stalled", "Goal stalled");
        // Set expires_at to the past
        entry.expires_at = Some((chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339());
        store.enqueue_notification(&entry).await.unwrap();

        // Also insert a non-expired critical notification
        let critical =
            crate::traits::NotificationEntry::new("goal-2", "session-1", "failed", "Goal failed");
        store.enqueue_notification(&critical).await.unwrap();

        // Cleanup should remove the expired one
        let cleaned = store.cleanup_expired_notifications().await.unwrap();
        assert_eq!(cleaned, 1);

        // Only the critical one remains
        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].notification_type, "failed");
    }

    #[tokio::test]
    async fn test_notification_queue_increment_attempt() {
        let (store, _file) = setup_test_store().await;

        let entry =
            crate::traits::NotificationEntry::new("goal-1", "session-1", "completed", "Goal done");
        store.enqueue_notification(&entry).await.unwrap();

        store
            .increment_notification_attempt(&entry.id)
            .await
            .unwrap();
        store
            .increment_notification_attempt(&entry.id)
            .await
            .unwrap();

        let pending = store.get_pending_notifications(10).await.unwrap();
        assert_eq!(pending[0].attempts, 2);
    }

    #[tokio::test]
    async fn test_cleanup_stale_goals() {
        let (store, _db) = setup_test_store().await;
        let now = chrono::Utc::now().to_rfc3339();
        let three_hours_ago = (chrono::Utc::now() - chrono::Duration::hours(3)).to_rfc3339();

        // Legacy goals: one stale active, one recent active, one completed
        sqlx::query::<sqlx::Sqlite>(
            "INSERT INTO goals (description, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
        )
        .bind("stale legacy goal")
        .bind("active")
        .bind(&three_hours_ago)
        .bind(&three_hours_ago)
        .execute(&store.pool)
        .await
        .unwrap();

        sqlx::query::<sqlx::Sqlite>(
            "INSERT INTO goals (description, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
        )
        .bind("recent legacy goal")
        .bind("active")
        .bind(&now)
        .bind(&now)
        .execute(&store.pool)
        .await
        .unwrap();

        sqlx::query::<sqlx::Sqlite>(
            "INSERT INTO goals (description, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
        )
        .bind("completed goal")
        .bind("completed")
        .bind(&three_hours_ago)
        .bind(&three_hours_ago)
        .execute(&store.pool)
        .await
        .unwrap();

        // V3 goals: one stale finite, one recent finite, one stale evergreen
        fn make_goal_v3(id: &str, desc: &str, goal_type: &str, ts: &str) -> GoalV3 {
            GoalV3 {
                id: id.to_string(),
                description: desc.to_string(),
                goal_type: goal_type.to_string(),
                status: "active".to_string(),
                priority: "medium".to_string(),
                conditions: None,
                schedule: None,
                context: None,
                resources: None,
                budget_per_check: None,
                budget_daily: None,
                tokens_used_today: 0,
                last_useful_action: None,
                created_at: ts.to_string(),
                updated_at: ts.to_string(),
                completed_at: None,
                parent_goal_id: None,
                session_id: "test".to_string(),
                notified_at: None,
                notification_attempts: 0,
                dispatch_failures: 0,
            }
        }

        store
            .create_goal_v3(&make_goal_v3(
                "stale-finite",
                "stale finite goal",
                "finite",
                &three_hours_ago,
            ))
            .await
            .unwrap();

        let mut stale_scheduled_finite = make_goal_v3(
            "stale-scheduled-finite",
            "stale scheduled finite goal",
            "finite",
            &three_hours_ago,
        );
        stale_scheduled_finite.schedule = Some("0 9 12 2 *".to_string());
        store.create_goal_v3(&stale_scheduled_finite).await.unwrap();

        store
            .create_goal_v3(&make_goal_v3(
                "recent-finite",
                "recent finite goal",
                "finite",
                &now,
            ))
            .await
            .unwrap();

        store
            .create_goal_v3(&make_goal_v3(
                "stale-evergreen",
                "stale evergreen goal",
                "evergreen",
                &three_hours_ago,
            ))
            .await
            .unwrap();

        // Run cleanup with 2-hour threshold
        let (legacy, v3) = store.cleanup_stale_goals(2).await.unwrap();
        assert_eq!(legacy, 1, "should abandon 1 stale legacy goal");
        assert_eq!(v3, 1, "should fail 1 stale finite v3 goal");

        // Verify legacy: stale → abandoned, recent stays active
        let row: (String,) = sqlx::query_as::<sqlx::Sqlite, _>(
            "SELECT status FROM goals WHERE description = 'stale legacy goal'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(row.0, "abandoned");

        let row: (String,) = sqlx::query_as::<sqlx::Sqlite, _>(
            "SELECT status FROM goals WHERE description = 'recent legacy goal'",
        )
        .fetch_one(&store.pool)
        .await
        .unwrap();
        assert_eq!(row.0, "active");

        // Verify V3: stale finite → failed, recent finite stays active, evergreen untouched
        let g = store.get_goal_v3("stale-finite").await.unwrap().unwrap();
        assert_eq!(g.status, "failed");
        assert!(g.completed_at.is_some());

        let g = store.get_goal_v3("recent-finite").await.unwrap().unwrap();
        assert_eq!(g.status, "active");

        let g = store
            .get_goal_v3("stale-scheduled-finite")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            g.status, "active",
            "scheduled finite goals should not be failed by stale cleanup"
        );

        let g = store.get_goal_v3("stale-evergreen").await.unwrap().unwrap();
        assert_eq!(
            g.status, "active",
            "evergreen goals should not be cleaned up"
        );
    }

    #[tokio::test]
    async fn test_migrate_legacy_scheduled_tasks_to_v3() {
        let (store, _db) = setup_test_store().await;
        let now = chrono::Utc::now();
        let next_hour = (now + chrono::Duration::hours(1)).to_rfc3339();

        // Recurring active legacy task
        sqlx::query::<sqlx::Sqlite>(
            "INSERT INTO scheduled_tasks
             (id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused, is_trusted, last_run_at, next_run_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind("legacy-recurring-1")
        .bind("legacy recurring")
        .bind("0 */6 * * *")
        .bind("every 6h")
        .bind("monitor API health")
        .bind("tool")
        .bind(0i64)
        .bind(0i64)
        .bind(0i64)
        .bind::<Option<String>>(None)
        .bind(&next_hour)
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&store.pool)
        .await
        .unwrap();

        // One-shot paused legacy task
        sqlx::query::<sqlx::Sqlite>(
            "INSERT INTO scheduled_tasks
             (id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused, is_trusted, last_run_at, next_run_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind("legacy-oneshot-1")
        .bind("legacy oneshot")
        .bind("0 9 * * *")
        .bind("tomorrow at 9am")
        .bind("check deployment")
        .bind("tool")
        .bind(1i64)
        .bind(1i64)
        .bind(0i64)
        .bind::<Option<String>>(None)
        .bind(&next_hour)
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&store.pool)
        .await
        .unwrap();

        let migrated = store.migrate_legacy_scheduled_tasks_to_v3().await.unwrap();
        assert_eq!(migrated, 2);

        // Idempotent rerun should not duplicate.
        let migrated_again = store.migrate_legacy_scheduled_tasks_to_v3().await.unwrap();
        assert_eq!(migrated_again, 0);

        let g1 = store
            .get_goal_v3("legacy-sched-legacy-recurring-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(g1.goal_type, "continuous");
        assert_eq!(g1.status, "active");
        assert_eq!(g1.session_id, "system");
        assert_eq!(g1.schedule.as_deref(), Some("0 */6 * * *"));

        let g2 = store
            .get_goal_v3("legacy-sched-legacy-oneshot-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(g2.goal_type, "finite");
        assert_eq!(g2.status, "paused");
        assert_eq!(g2.session_id, "system");
        assert!(g2.schedule.is_some());
    }
}
