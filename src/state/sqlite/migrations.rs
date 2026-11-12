use sqlx::Row;
use sqlx::SqlitePool;
use std::collections::HashSet;

use aidaemon_work::schema::{table_exists_in_pool, LEGACY_GOALS_IMPORTED_VERSION};

async fn migrate_legacy_messages_to_events(pool: &SqlitePool) -> anyhow::Result<()> {
    let has_messages = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();

    if !has_messages {
        return Ok(());
    }

    let rows = sqlx::query(
        "SELECT id, session_id, role, content, tool_call_id, tool_name, tool_calls_json, created_at
         FROM messages
         ORDER BY created_at ASC, id ASC",
    )
    .fetch_all(pool)
    .await?;

    let existing_rows = sqlx::query(
        "SELECT e.session_id AS session_id,
                e.event_type AS event_type,
                CAST(json_extract(e.data, '$.message_id') AS TEXT) AS message_id
         FROM events e
         INNER JOIN (SELECT DISTINCT session_id FROM messages) m
           ON m.session_id = e.session_id
         WHERE e.event_type IN ('user_message', 'assistant_response', 'tool_result')
           AND json_extract(e.data, '$.message_id') IS NOT NULL",
    )
    .fetch_all(pool)
    .await?;
    let mut existing_keys: HashSet<(String, String, String)> = existing_rows
        .into_iter()
        .map(|row| {
            (
                row.get("session_id"),
                row.get("event_type"),
                row.get("message_id"),
            )
        })
        .collect();

    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
    let mut scanned: u64 = 0;
    let mut migrated: u64 = 0;

    for row in rows {
        scanned += 1;

        let message_id: String = row.get("id");
        let message_id_key = message_id.clone();
        let session_id: String = row.get("session_id");
        let role: String = row.get("role");
        let content: Option<String> = row.get("content");
        let tool_call_id: Option<String> = row.get("tool_call_id");
        let tool_name: Option<String> = row.get("tool_name");
        let tool_calls_json: Option<String> = row.get("tool_calls_json");
        let created_at: String = row.get("created_at");

        let (event_type, payload, event_tool_name): (&str, serde_json::Value, Option<String>) =
            match role.as_str() {
                "user" => (
                    "user_message",
                    serde_json::json!({
                        "content": content.unwrap_or_default(),
                        "message_id": message_id,
                        "has_attachments": false
                    }),
                    None,
                ),
                "tool" => (
                    "tool_result",
                    {
                        let fallback_tool_call_id = format!("legacy-tool-{}", message_id);
                        serde_json::json!({
                            "message_id": message_id,
                            "tool_call_id": tool_call_id.unwrap_or(fallback_tool_call_id),
                            "name": tool_name.clone().unwrap_or_else(|| "system".to_string()),
                            "result": content.unwrap_or_default(),
                            "success": true,
                            "duration_ms": 0,
                            "error": serde_json::Value::Null,
                            "task_id": serde_json::Value::Null
                        })
                    },
                    tool_name,
                ),
                _ => {
                    let parsed_tool_calls = tool_calls_json
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<Vec<serde_json::Value>>(raw).ok())
                        .map(|calls| {
                            calls
                                .into_iter()
                                .filter_map(|tc| {
                                    let id = tc.get("id")?.as_str()?;
                                    let name = tc.get("name")?.as_str()?;
                                    let arguments = tc
                                        .get("arguments")
                                        .cloned()
                                        .and_then(|args| match args {
                                            serde_json::Value::String(raw) => {
                                                serde_json::from_str::<serde_json::Value>(&raw).ok()
                                            }
                                            other => Some(other),
                                        })
                                        .unwrap_or_else(|| serde_json::json!({}));
                                    let extra_content = tc.get("extra_content").cloned();

                                    let mut obj = serde_json::Map::new();
                                    obj.insert("id".to_string(), serde_json::json!(id));
                                    obj.insert("name".to_string(), serde_json::json!(name));
                                    obj.insert("arguments".to_string(), arguments);
                                    if let Some(extra) = extra_content {
                                        obj.insert("extra_content".to_string(), extra);
                                    }
                                    Some(serde_json::Value::Object(obj))
                                })
                                .collect::<Vec<_>>()
                        })
                        .filter(|v| !v.is_empty());

                    let mut payload = serde_json::Map::new();
                    payload.insert("message_id".to_string(), serde_json::json!(message_id));
                    payload.insert("content".to_string(), serde_json::json!(content));
                    payload.insert(
                        "model".to_string(),
                        serde_json::json!("legacy-messages-migration"),
                    );
                    if let Some(tool_calls) = parsed_tool_calls {
                        payload.insert("tool_calls".to_string(), serde_json::json!(tool_calls));
                    }

                    (
                        "assistant_response",
                        serde_json::Value::Object(payload),
                        None,
                    )
                }
            };

        let dedupe_key = (
            session_id.clone(),
            event_type.to_string(),
            message_id_key.clone(),
        );
        if existing_keys.contains(&dedupe_key) {
            continue;
        }

        sqlx::query(
            "INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name)
             VALUES (?, ?, ?, ?, NULL, ?)",
        )
        .bind(&session_id)
        .bind(event_type)
        .bind(payload.to_string())
        .bind(&created_at)
        .bind(event_tool_name.as_deref())
        .execute(&mut *tx)
        .await?;

        existing_keys.insert(dedupe_key);
        migrated += 1;
        if scanned.is_multiple_of(5_000) {
            tracing::info!(
                scanned_rows = scanned,
                migrated_rows = migrated,
                "Migrating legacy messages table into events"
            );
        }
    }

    // Legacy conversation rows are now represented in the canonical event log.
    sqlx::query("DROP TABLE IF EXISTS messages")
        .execute(&mut *tx)
        .await?;

    // Clean obsolete projection toggles from runtime settings.
    let _ = sqlx::query(
        "DELETE FROM settings WHERE key IN ('enable_event_to_messages_projection', 'event_projection_last_id')",
    )
    .execute(&mut *tx)
    .await;

    tx.commit().await?;

    tracing::info!(
        scanned_rows = scanned,
        migrated_rows = migrated,
        "Migrated legacy messages table into events and removed legacy table"
    );

    Ok(())
}

pub(crate) async fn migrate_state(pool: &SqlitePool) -> anyhow::Result<()> {
    // Ensure canonical events schema exists even when only SqliteStateStore is
    // initialized (without EventStore bootstrap).
    crate::db::migrations::migrate_events(pool).await?;

    // Long-term memory's tables belong to its store.
    aidaemon_memory::sqlite::migrate(pool).await?;

    // Terminal allowed prefixes (persisted "Allow Always" approvals)
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS terminal_allowed_prefixes (
            prefix TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    // Backend-scoped terminal approvals. Historical approvals apply only to
    // local execution; Docker and SSH targets must establish their own trust.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS terminal_backend_allowed_prefixes (
            backend_scope TEXT NOT NULL,
            prefix TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (backend_scope, prefix)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT OR IGNORE INTO terminal_backend_allowed_prefixes
            (backend_scope, prefix)
         SELECT 'local', prefix FROM terminal_allowed_prefixes",
    )
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
    .await?;

    // Token usage tracking
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cached_input_tokens INTEGER,
            cache_creation_input_tokens INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    let _ = sqlx::query("ALTER TABLE token_usage ADD COLUMN cached_input_tokens INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE token_usage ADD COLUMN cache_creation_input_tokens INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE token_usage ADD COLUMN call_id TEXT")
        .execute(pool)
        .await;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_token_usage_call_id
         ON token_usage(call_id)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_token_usage_created_at
         ON token_usage(created_at)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_token_usage_session_created_at
         ON token_usage(session_id, created_at)",
    )
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
    .await?;

    // Non-destructive `/clear` boundary: hides events at or before
    // `cleared_after_id` from CONTEXT retrieval without deleting them, so the
    // conversation starts fresh while the event history stays intact for memory
    // and audit. `/wipe` remains the explicit destructive path.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS session_context_boundaries (
            session_id TEXT PRIMARY KEY,
            cleared_after_id INTEGER NOT NULL,
            cleared_at TEXT NOT NULL
        )",
    )
    .execute(pool)
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
    .execute(pool)
    .await?;

    // Migration: add resources_json column if missing
    sqlx::query("ALTER TABLE dynamic_skills ADD COLUMN resources_json TEXT NOT NULL DEFAULT '[]'")
        .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
    .await?;

    // CLI agent invocations table - logs each CLI agent run for auditing
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS cli_agent_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
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
    .execute(pool)
    .await?;
    let _ = sqlx::query("ALTER TABLE cli_agent_invocations ADD COLUMN task_id TEXT")
        .execute(pool)
        .await;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_cli_agent_invocations_task_id
         ON cli_agent_invocations(task_id)",
    )
    .execute(pool)
    .await?;

    // --- OAuth connections table ---
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS oauth_connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL UNIQUE,
            auth_type TEXT NOT NULL,
            username TEXT,
            account_id TEXT,
            scopes TEXT NOT NULL DEFAULT '[]',
            token_expires_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    // Stable remote identity bound after an authenticated read-only proof.
    // Existing OAuth connections remain intentionally unbound until the owner
    // verifies them; never infer identity from the service or username.
    let _ = sqlx::query("ALTER TABLE oauth_connections ADD COLUMN account_id TEXT")
        .execute(pool)
        .await;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS pending_oauth_flows (
            state TEXT PRIMARY KEY,
            service TEXT NOT NULL,
            code_verifier TEXT,
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_pending_oauth_flows_created_at \
         ON pending_oauth_flows(created_at)",
    )
    .execute(pool)
    .await?;

    // --- Settings table (generic key-value runtime toggles) ---
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    // --- Dialogue state projection ---
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS dialogue_states (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            revision INTEGER NOT NULL,
            active_task_id TEXT,
            open_request_status TEXT,
            awaiting_user_reply INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_dialogue_states_active_task ON dialogue_states(active_task_id)")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_dialogue_states_open_request_status ON dialogue_states(open_request_status)")
        .execute(pool)
        .await?;

    // Migrate legacy message rows into canonical events and remove the table.
    migrate_legacy_messages_to_events(pool).await?;

    aidaemon_work::schema::migrate_work_tables(pool, crate::cron_utils::is_one_shot_schedule)
        .await?;

    // Token alert detector dedupe/cooldown state.
    aidaemon_work::token_alerts::migrate(pool).await?;

    // Conversation summaries for context window management
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS conversation_summaries (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            last_message_id TEXT NOT NULL,
            last_turn_seq INTEGER,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;
    // Legacy summaries were cursorless and could end in the middle of a tool
    // exchange. A NULL cursor makes that state explicit so the summarizer can
    // rebuild it once from canonical whole turns.
    let _ = sqlx::query("ALTER TABLE conversation_summaries ADD COLUMN last_turn_seq INTEGER")
        .execute(pool)
        .await;

    // Rendered system-prompt snapshots, deduplicated by content hash.
    // Written insert-or-ignore from the instructions-snapshot path so any past
    // llm_call can be replayed exactly (snapshot + message events).
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS prompt_snapshots (
            hash TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // Apply the retention bound during startup as well as on new writes. An
    // idle deployment may not save another prompt snapshot for a long time,
    // so write-time pruning alone leaves legacy databases permanently above
    // the cap.
    sqlx::query(
        "DELETE FROM prompt_snapshots
         WHERE hash NOT IN (
             SELECT hash FROM prompt_snapshots
             ORDER BY created_at DESC, hash DESC
             LIMIT 500
         )",
    )
    .execute(pool)
    .await?;

    aidaemon_work::schema::migrate_work_tail(pool).await?;

    if let Err(error) = super::history_search::migrate_history_search(pool).await {
        // Conversation events are authoritative. Exact-history search is a
        // replaceable projection and must never prevent the daemon starting.
        tracing::warn!(%error, "SQLite FTS5 unavailable; exact history search disabled");
    }

    // Written last, so a start that failed above records nothing.
    record_state_version(pool, STATE_BASELINE_VERSION, "unversioned_baseline").await?;
    // The import above has run: anything retention removes later stays removed.
    if table_exists_in_pool(pool, "_goals_legacy_deprecated").await? {
        record_state_version(pool, LEGACY_GOALS_IMPORTED_VERSION, "legacy_goals_imported").await?;
    }

    Ok(())
}

/// Everything `migrate_state` does above this record: steps written before
/// versions existed. They stay idempotent and keep running on every start,
/// because a database gives no way to tell which of them it has seen.
const STATE_BASELINE_VERSION: i64 = 1;

/// Record that a state schema version has been applied.
///
/// A new step that cannot be made idempotent takes the next version: it runs
/// once when that version is absent here, in a transaction that also records
/// it. Existing steps are never moved behind a version. The table is not the
/// SDK store's `schema_migrations`, which lives in the same database file and
/// counts its own migrations.
async fn record_state_version(pool: &SqlitePool, version: i64, name: &str) -> anyhow::Result<()> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS state_schema_versions (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query("INSERT OR IGNORE INTO state_schema_versions (version, name) VALUES (?, ?)")
        .bind(version)
        .bind(name)
        .execute(pool)
        .await?;
    Ok(())
}

#[cfg(test)]
mod legacy_goal_import_tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    #[tokio::test]
    async fn goals_retention_removed_are_not_imported_again() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let pool = SqlitePoolOptions::new()
            .max_connections(2)
            .connect_with(SqliteConnectOptions::new().filename(database.path()))
            .await
            .unwrap();
        migrate_state(&pool).await.unwrap();
        sqlx::query(
            "CREATE TABLE _goals_legacy_deprecated (
                id INTEGER PRIMARY KEY, description TEXT NOT NULL, status TEXT,
                priority TEXT, progress_notes TEXT, source_episode_id INTEGER,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO _goals_legacy_deprecated (id, description, status, created_at, updated_at)
             VALUES (7, 'Plan the Acme launch', 'completed', '2025-01-01', '2025-01-01')",
        )
        .execute(&pool)
        .await
        .unwrap();
        let imported = "SELECT COUNT(*) FROM goals WHERE legacy_int_id = 7";

        migrate_state(&pool).await.unwrap();
        let count: i64 = sqlx::query_scalar(imported).fetch_one(&pool).await.unwrap();
        assert_eq!(count, 1, "imported once");

        sqlx::query("DELETE FROM goals WHERE legacy_int_id = 7")
            .execute(&pool)
            .await
            .unwrap();
        migrate_state(&pool).await.unwrap();
        let count: i64 = sqlx::query_scalar(imported).fetch_one(&pool).await.unwrap();
        assert_eq!(count, 0, "what retention removed stays removed");
        let kept: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM _goals_legacy_deprecated")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(kept, 1, "the recovery copy is untouched");
    }
}

#[cfg(test)]
mod oauth_identity_upgrade_tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    #[tokio::test]
    async fn adds_account_binding_without_inventing_identity_for_existing_connections() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let options = SqliteConnectOptions::new()
            .filename(database.path())
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE oauth_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL UNIQUE,
                auth_type TEXT NOT NULL,
                username TEXT,
                scopes TEXT NOT NULL DEFAULT '[]',
                token_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO oauth_connections
                (service, auth_type, username, scopes, created_at, updated_at)
             VALUES ('synthetic', 'oauth2_pkce', 'mutable-alias', '[]',
                     '2026-08-02T00:00:00Z', '2026-08-02T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        migrate_state(&pool).await.unwrap();

        let columns = sqlx::query("PRAGMA table_info(oauth_connections)")
            .fetch_all(&pool)
            .await
            .unwrap();
        assert!(columns
            .iter()
            .any(|row| row.get::<String, _>("name") == "account_id"));
        let row = sqlx::query(
            "SELECT username, account_id FROM oauth_connections WHERE service = 'synthetic'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row.get::<String, _>("username"), "mutable-alias");
        assert!(row.get::<Option<String>, _>("account_id").is_none());
    }
}

#[cfg(test)]
mod mandate_ledger_upgrade_tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    #[tokio::test]
    async fn upgrades_legacy_mutation_status_check_to_never_dispatched() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let options = SqliteConnectOptions::new()
            .filename(database.path())
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap();

        // Start from a fully migrated database so the fixture exercises the
        // production foreign-key graph rather than inserting orphaned dummy
        // identifiers. Then replace only the ledger with its pre-upgrade
        // shape and seed one valid row that the rebuild must preserve.
        migrate_state(&pool).await.unwrap();
        assert_eq!(
            sqlx::query_scalar::<_, i64>("PRAGMA foreign_keys")
                .fetch_one(&pool)
                .await
                .unwrap(),
            1
        );
        sqlx::query(
            "INSERT INTO goals
                (id, description, domain, goal_type, status, session_id)
             VALUES ('goal', 'upgrade fixture', 'orchestration', 'continuous',
                     'active', 'owner-session')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO goal_runs
                (id, project_id, goal_id, trigger_type, status)
             VALUES ('run', 'default', 'goal', 'mandate', 'running')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO mandates
                (id, goal_id, objective, status, authority_json,
                 constraints_json, success_criteria_json, stop_conditions_json,
                 min_review_secs, max_review_secs, default_review_secs,
                 next_review_at, confirmed_at, version, owner_principal_id,
                 created_by_session)
             VALUES ('mandate', 'goal', 'upgrade fixture', 'active', '{}',
                     '[]', '[]', '[]', 60, 3600, 300,
                     '2026-08-02T12:00:00Z', '2026-08-02T11:00:00Z', 1,
                     'principal:upgrade-fixture', 'owner-session')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO mandate_decision_cycles
                (id, mandate_id, goal_run_id, mandate_version, outcome,
                 rationale, action_attempts)
             VALUES ('decision', 'mandate', 'run', 1, 'act',
                     'upgrade fixture', 1)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO intentions
                (id, mandate_id, decision_cycle_id, goal_run_id, description,
                 rationale, status)
             VALUES ('intention', 'mandate', 'decision', 'run',
                     'upgrade fixture', 'upgrade fixture', 'committed')",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query("DROP TABLE mandate_mutation_attempts")
            .execute(&pool)
            .await
            .unwrap();
        // This is the pre-upgrade ledger shape: it has neither the final
        // dispatch marker nor the terminal never_dispatched status, but it
        // retains the real production foreign keys and uniqueness rules.
        sqlx::query(
            "CREATE TABLE mandate_mutation_attempts (
                id TEXT PRIMARY KEY,
                mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
                mandate_version INTEGER NOT NULL CHECK (mandate_version > 0),
                decision_cycle_id TEXT NOT NULL REFERENCES mandate_decision_cycles(id) ON DELETE CASCADE,
                goal_run_id TEXT NOT NULL REFERENCES goal_runs(id) ON DELETE CASCADE,
                intention_id TEXT NOT NULL REFERENCES intentions(id) ON DELETE CASCADE,
                root_task_id TEXT NOT NULL,
                root_task_attempt_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_attempt_id TEXT NOT NULL,
                reserved_action_attempt INTEGER NOT NULL CHECK (reserved_action_attempt > 0),
                action_digest TEXT NOT NULL UNIQUE CHECK (length(action_digest) = 64),
                tool_call_id TEXT NOT NULL UNIQUE,
                tool_name TEXT NOT NULL,
                mutation_effects_json TEXT NOT NULL CHECK (json_valid(mutation_effects_json)),
                targets_json TEXT NOT NULL CHECK (json_valid(targets_json)),
                account_identifiers_json TEXT NOT NULL CHECK (json_valid(account_identifiers_json)),
                status TEXT NOT NULL DEFAULT 'reserved'
                    CHECK (status IN ('reserved', 'succeeded', 'failed', 'ambiguous')),
                outcome_evidence TEXT
                    CHECK (outcome_evidence IS NULL OR outcome_evidence IN ('tool_reported', 'structured_metadata')),
                http_status INTEGER CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
                exit_code INTEGER,
                reserved_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE (decision_cycle_id, reserved_action_attempt)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        for statement in [
            "CREATE INDEX idx_mandate_mutations_quota
             ON mandate_mutation_attempts(mandate_id, reserved_at)",
            "CREATE INDEX idx_mandate_mutations_run_status
             ON mandate_mutation_attempts(goal_run_id, status)",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        sqlx::query(
            "INSERT INTO mandate_mutation_attempts
                (id, mandate_id, mandate_version, decision_cycle_id, goal_run_id,
                 intention_id, root_task_id, root_task_attempt_id, task_id,
                 task_attempt_id, reserved_action_attempt, action_digest,
                 tool_call_id, tool_name, mutation_effects_json, targets_json,
                 account_identifiers_json, status, reserved_at)
             VALUES ('legacy-row', 'mandate', 1, 'decision', 'run', 'intention',
                     'root', 'root-attempt', 'task', 'task-attempt', 1,
                     'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                     'tool-legacy', 'http_request', '[\"external_delivery\"]',
                     '[]', '[]', 'reserved', '2026-08-02T12:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        migrate_state(&pool).await.unwrap();

        let schema = sqlx::query_scalar::<_, String>(
            "SELECT sql FROM sqlite_master
             WHERE type = 'table' AND name = 'mandate_mutation_attempts'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(schema.contains("never_dispatched"));
        let columns = sqlx::query("PRAGMA table_info(mandate_mutation_attempts)")
            .fetch_all(&pool)
            .await
            .unwrap();
        assert!(columns
            .iter()
            .any(|row| row.get::<String, _>("name") == "dispatch_claimed_at"));

        let preserved = sqlx::query(
            "SELECT status, action_digest, tool_call_id, dispatch_claimed_at
             FROM mandate_mutation_attempts WHERE id = 'legacy-row'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(preserved.get::<String, _>("status"), "reserved");
        assert_eq!(
            preserved.get::<String, _>("action_digest"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert_eq!(preserved.get::<String, _>("tool_call_id"), "tool-legacy");
        assert!(preserved
            .get::<Option<String>, _>("dispatch_claimed_at")
            .is_none());

        sqlx::query(
            "UPDATE mandate_mutation_attempts
             SET status = 'never_dispatched' WHERE id = 'legacy-row'",
        )
        .execute(&pool)
        .await
        .unwrap();
        assert!(sqlx::query(
            "UPDATE mandate_mutation_attempts
             SET status = 'invented_status' WHERE id = 'legacy-row'",
        )
        .execute(&pool)
        .await
        .is_err());

        let foreign_tables = sqlx::query("PRAGMA foreign_key_list(mandate_mutation_attempts)")
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String, _>("table"))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            foreign_tables,
            [
                "mandates".to_string(),
                "mandate_decision_cycles".to_string(),
                "goal_runs".to_string(),
                "intentions".to_string(),
            ]
            .into_iter()
            .collect()
        );
        assert!(
            sqlx::query("PRAGMA foreign_key_check(mandate_mutation_attempts)")
                .fetch_all(&pool)
                .await
                .unwrap()
                .is_empty()
        );

        let index_names = sqlx::query("PRAGMA index_list(mandate_mutation_attempts)")
            .fetch_all(&pool)
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String, _>("name"))
            .collect::<std::collections::HashSet<_>>();
        assert!(index_names.contains("idx_mandate_mutations_quota"));
        assert!(index_names.contains("idx_mandate_mutations_run_status"));

        // The rebuilt references remain enforced, including their cascade.
        assert!(sqlx::query(
            "INSERT INTO mandate_mutation_attempts
                (id, mandate_id, mandate_version, decision_cycle_id, goal_run_id,
                 intention_id, root_task_id, root_task_attempt_id, task_id,
                 task_attempt_id, reserved_action_attempt, action_digest,
                 tool_call_id, tool_name, mutation_effects_json, targets_json,
                 account_identifiers_json, status, reserved_at)
             VALUES ('orphan', 'missing-mandate', 1, 'decision', 'run',
                     'intention', 'root', 'root-attempt', 'task', 'task-attempt',
                     2,
                     'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                     'tool-orphan', 'http_request', '[\"external_delivery\"]',
                     '[]', '[]', 'reserved', '2026-08-02T12:00:01Z')",
        )
        .execute(&pool)
        .await
        .is_err());
        sqlx::query("DELETE FROM mandates WHERE id = 'mandate'")
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(
            sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM mandate_mutation_attempts WHERE id = 'legacy-row'",
            )
            .fetch_one(&pool)
            .await
            .unwrap(),
            0
        );
    }
}

#[cfg(test)]
mod state_version_tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::collections::BTreeMap;

    const VERSION_TABLE: &str = "state_schema_versions";

    async fn open(path: &std::path::Path) -> SqlitePool {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap()
    }

    /// Every stored object's DDL and every table's rows, without the version
    /// table this change adds.
    async fn snapshot(pool: &SqlitePool) -> BTreeMap<String, Vec<String>> {
        let objects = sqlx::query(
            "SELECT type, name, sql FROM sqlite_master
             WHERE name NOT LIKE 'sqlite_%' AND tbl_name <> ?",
        )
        .bind(VERSION_TABLE)
        .fetch_all(pool)
        .await
        .unwrap();
        // A full-text index's shadow tables hold segment counters that the
        // existing steps' rebuilds advance on every start; they are a
        // projection of rows compared elsewhere, not stored data.
        let virtual_tables: Vec<String> = objects
            .iter()
            .filter(|o| {
                o.get::<Option<String>, _>("sql")
                    .is_some_and(|sql| sql.to_ascii_uppercase().starts_with("CREATE VIRTUAL"))
            })
            .map(|o| format!("{}_", o.get::<String, _>("name")))
            .collect();
        let mut seen = BTreeMap::new();
        for object in objects {
            let kind: String = object.get("type");
            let name: String = object.get("name");
            let sql = object.get::<Option<String>, _>("sql").unwrap_or_default();
            let has_rows = kind == "table"
                && !sql.to_ascii_uppercase().starts_with("CREATE VIRTUAL")
                && !virtual_tables.iter().any(|prefix| name.starts_with(prefix));
            seen.insert(format!("ddl:{kind}:{name}"), vec![sql]);
            if !has_rows {
                continue;
            }
            let columns = sqlx::query(sqlx::AssertSqlSafe(format!(
                "PRAGMA table_info(\"{name}\")"
            )))
            .fetch_all(pool)
            .await
            .unwrap();
            let select = columns
                .iter()
                .map(|c| format!("quote(\"{}\")", c.get::<String, _>("name")))
                .collect::<Vec<_>>()
                .join(" || '|' || ");
            let mut rows: Vec<String> = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
                "SELECT {select} FROM \"{name}\""
            )))
            .fetch_all(pool)
            .await
            .unwrap();
            rows.sort();
            seen.insert(format!("rows:{name}"), rows);
        }
        seen
    }

    async fn recorded(pool: &SqlitePool) -> Vec<(i64, String, String)> {
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "SELECT version, name, applied_at FROM {VERSION_TABLE} ORDER BY version"
        )))
        .fetch_all(pool)
        .await
        .unwrap()
        .into_iter()
        .map(|row| (row.get("version"), row.get("name"), row.get("applied_at")))
        .collect()
    }

    #[tokio::test]
    async fn a_database_from_before_versions_upgrades_with_no_data_change() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let pool = open(database.path()).await;

        // A current database as the previous release left it: rows from real
        // use, then that release's own start, and no version record.
        migrate_state(&pool).await.unwrap();
        for statement in [
            "INSERT INTO events (session_id, event_type, data, created_at)
             VALUES ('telegram:synthetic-user-1', 'user_message',
                     '{\"content\":\"Book the Acme Corp review, Alice Rivera\"}',
                     '2026-10-01T09:00:00Z')",
            "INSERT INTO terminal_allowed_prefixes (prefix, created_at)
             VALUES ('git status', '2026-10-01 09:00:01')",
            "INSERT INTO settings (key, value) VALUES ('synthetic_setting', 'kept')",
            "INSERT INTO session_context_boundaries (session_id, cleared_after_id, cleared_at)
             VALUES ('telegram:synthetic-user-1', 0, '2026-10-01T09:00:02Z')",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        migrate_state(&pool).await.unwrap();
        sqlx::query(sqlx::AssertSqlSafe(format!("DROP TABLE {VERSION_TABLE}")))
            .execute(&pool)
            .await
            .unwrap();
        let before = snapshot(&pool).await;
        assert_eq!(before["rows:settings"].len(), 1);
        assert_eq!(before["rows:terminal_allowed_prefixes"].len(), 1);
        assert!(!before["rows:events"].is_empty());

        migrate_state(&pool).await.unwrap();

        assert_eq!(snapshot(&pool).await, before);
        let first = recorded(&pool).await;
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].0, STATE_BASELINE_VERSION);

        // Later starts add nothing and keep the first record.
        migrate_state(&pool).await.unwrap();
        assert_eq!(snapshot(&pool).await, before);
        assert_eq!(recorded(&pool).await, first);
    }

    #[tokio::test]
    async fn rows_in_an_old_shaped_table_survive_and_the_version_is_recorded() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let pool = open(database.path()).await;

        // Tables from early releases, before the columns later steps add.
        for statement in [
            "CREATE TABLE oauth_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL UNIQUE,
                auth_type TEXT NOT NULL,
                username TEXT,
                scopes TEXT NOT NULL DEFAULT '[]',
                token_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
            "INSERT INTO oauth_connections
                (service, auth_type, username, scopes, created_at, updated_at)
             VALUES ('synthetic', 'oauth2_pkce', 'jordan-lee', '[\"read\"]',
                     '2026-01-02T00:00:00Z', '2026-01-03T00:00:00Z')",
            "CREATE TABLE terminal_allowed_prefixes (
                prefix TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
            "INSERT INTO terminal_allowed_prefixes (prefix, created_at)
             VALUES ('cargo test', '2026-01-02 00:00:00')",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }

        migrate_state(&pool).await.unwrap();

        let connection = sqlx::query(
            "SELECT id, auth_type, username, scopes, created_at, updated_at, account_id
             FROM oauth_connections WHERE service = 'synthetic'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(connection.get::<i64, _>("id"), 1);
        assert_eq!(connection.get::<String, _>("auth_type"), "oauth2_pkce");
        assert_eq!(connection.get::<String, _>("username"), "jordan-lee");
        assert_eq!(connection.get::<String, _>("scopes"), "[\"read\"]");
        assert_eq!(
            connection.get::<String, _>("created_at"),
            "2026-01-02T00:00:00Z"
        );
        assert_eq!(
            connection.get::<String, _>("updated_at"),
            "2026-01-03T00:00:00Z"
        );
        assert!(connection.get::<Option<String>, _>("account_id").is_none());
        let prefixes: Vec<(String, String)> =
            sqlx::query_as("SELECT prefix, created_at FROM terminal_allowed_prefixes")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(
            prefixes,
            vec![("cargo test".to_string(), "2026-01-02 00:00:00".to_string())]
        );
        assert_eq!(recorded(&pool).await.len(), 1);
    }
}

/// Guards the relocation of these tables' migrations into their crates: every
/// object, in creation order, and every recorded version must stay what a
/// database upgrades to today. The goldens were written before the move.
#[cfg(test)]
mod schema_golden_tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    async fn open(path: &std::path::Path) -> SqlitePool {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap()
    }

    /// `sqlite_master` in creation order, the versions, and each table's row
    /// count. Full-text tables are left out: their presence depends on how
    /// SQLite was built, and they are a rebuildable projection.
    async fn dump(pool: &SqlitePool) -> String {
        let objects = sqlx::query(
            "SELECT type, name, tbl_name, sql FROM sqlite_master
             WHERE name NOT LIKE 'sqlite_%' ORDER BY rowid",
        )
        .fetch_all(pool)
        .await
        .unwrap();
        let virtual_tables: Vec<String> = objects
            .iter()
            .filter(|o| {
                o.get::<Option<String>, _>("sql")
                    .is_some_and(|sql| sql.to_ascii_uppercase().starts_with("CREATE VIRTUAL"))
            })
            .map(|o| o.get::<String, _>("name"))
            .collect();
        let projected = |name: &str| {
            virtual_tables
                .iter()
                .any(|table| name == table || name.starts_with(&format!("{table}_")))
        };
        let mut out = String::new();
        for object in &objects {
            let tbl_name: String = object.get("tbl_name");
            if projected(&tbl_name) {
                continue;
            }
            out.push_str(&format!(
                "{} {} on {}\n{}\n\n",
                object.get::<String, _>("type"),
                object.get::<String, _>("name"),
                tbl_name,
                object.get::<Option<String>, _>("sql").unwrap_or_default(),
            ));
        }
        for object in &objects {
            let name: String = object.get("name");
            if object.get::<String, _>("type") != "table" || projected(&name) {
                continue;
            }
            let rows: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
                "SELECT COUNT(*) FROM \"{name}\""
            )))
            .fetch_one(pool)
            .await
            .unwrap();
            out.push_str(&format!("rows {name} = {rows}\n"));
        }
        for row in sqlx::query("SELECT version, name FROM state_schema_versions ORDER BY version")
            .fetch_all(pool)
            .await
            .unwrap()
        {
            out.push_str(&format!(
                "version {} {}\n",
                row.get::<i64, _>("version"),
                row.get::<String, _>("name")
            ));
        }
        out
    }

    fn assert_matches_golden(name: &str, actual: &str) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/state/sqlite/schema_golden")
            .join(name);
        if std::env::var_os("UPDATE_SCHEMA_GOLDEN").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, actual).unwrap();
            return;
        }
        let expected = std::fs::read_to_string(&path).unwrap();
        if expected != actual {
            let difference = expected
                .lines()
                .zip(actual.lines())
                .enumerate()
                .find(|(_, (left, right))| left != right);
            panic!(
                "schema differs from {name}: first difference {difference:?}, \
                 {} expected lines, {} actual",
                expected.lines().count(),
                actual.lines().count()
            );
        }
    }

    #[tokio::test]
    async fn a_new_database_keeps_its_exact_schema() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let pool = open(database.path()).await;
        migrate_state(&pool).await.unwrap();
        assert_matches_golden("new_database.golden", &dump(&pool).await);
    }

    #[tokio::test]
    async fn a_current_database_keeps_its_exact_schema() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let pool = open(database.path()).await;
        migrate_state(&pool).await.unwrap();
        migrate_state(&pool).await.unwrap();
        // Memory's triggers are rebuilt on every start, so this differs from a
        // new database only by their position.
        assert_matches_golden("current_database.golden", &dump(&pool).await);
    }

    #[tokio::test]
    async fn an_old_goal_database_upgrades_to_its_exact_schema() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let pool = open(database.path()).await;
        for statement in [
            "CREATE TABLE goals (
                id INTEGER PRIMARY KEY, description TEXT NOT NULL, status TEXT,
                priority TEXT, progress_notes TEXT, source_episode_id INTEGER,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT
            )",
            "INSERT INTO goals (id, description, status, created_at, updated_at)
             VALUES (7, 'Plan the Acme launch', 'active', '2025-01-01', '2025-01-01')",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        migrate_state(&pool).await.unwrap();
        assert_matches_golden("old_goal_database.golden", &dump(&pool).await);
        let imported: String =
            sqlx::query_scalar("SELECT description FROM goals WHERE legacy_int_id = 7")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(imported, "Plan the Acme launch");
    }
}
