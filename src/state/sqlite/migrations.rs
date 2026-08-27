use chrono::{Datelike, TimeZone, Timelike};
use sqlx::Row;
use sqlx::SqlitePool;
use std::collections::HashSet;

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

    let mut tx = pool.begin().await?;
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

    // Create tables
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // --- Human-Like Memory System Migrations ---
    // 5. Add new columns to facts table for supersession and recall tracking
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN superseded_at TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN recall_count INTEGER DEFAULT 0")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN last_recalled_at TEXT")
        .execute(pool)
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
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)")
        .execute(pool)
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
            asks_before_acting INTEGER DEFAULT 0,
            prefers_explanations INTEGER DEFAULT 1,
            likes_suggestions INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
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
    .execute(pool)
    .await?;

    // Normalize historical NULL domains to empty string so dedupe/unique keys are stable.
    sqlx::query("UPDATE error_solutions SET domain = '' WHERE domain IS NULL")
        .execute(pool)
        .await?;

    // Dedupe: allow multiple solutions per error pattern, but avoid identical repeats.
    // Only do the (potentially expensive) cleanup once, before we install the unique index.
    let has_unique: Option<i64> = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_error_solutions_unique' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?;
    if has_unique.is_none() {
        // Remove exact duplicates before adding the unique index.
        // Keep the smallest id (oldest row) for each (error_pattern, domain, solution_summary) triple.
        sqlx::query(
            r#"
            DELETE FROM error_solutions
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM error_solutions
                GROUP BY error_pattern, domain, solution_summary
            )
            "#,
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_error_solutions_unique
             ON error_solutions(error_pattern, domain, solution_summary)",
        )
        .execute(pool)
        .await?;
    }

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
    .execute(pool)
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
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_people_name ON people(name)")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_person_facts_person ON person_facts(person_id)")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_person_facts_category ON person_facts(category)")
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

    // --- Channel-Scoped Memory Migrations ---
    // Add channel_id and privacy columns to facts table
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN channel_id TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN privacy TEXT DEFAULT 'global'")
        .execute(pool)
        .await;
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_facts_channel ON facts(channel_id)")
        .execute(pool)
        .await;
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_facts_privacy ON facts(privacy)")
        .execute(pool)
        .await;
    // Add channel_id column to episodes table
    let _ = sqlx::query("ALTER TABLE episodes ADD COLUMN channel_id TEXT")
        .execute(pool)
        .await;
    // Stable canonical bounds avoid timestamp-tie duplication when creating
    // multiple episodes for a long-running session.
    let _ = sqlx::query("ALTER TABLE episodes ADD COLUMN start_event_id INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE episodes ADD COLUMN end_event_id INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_episodes_session_end_event
         ON episodes(session_id, end_event_id)",
    )
    .execute(pool)
    .await;

    // --- Binary Embedding Storage Migration ---
    // Add embedding column to facts table for pre-computed embeddings
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN embedding BLOB")
        .execute(pool)
        .await;

    // --- Provenance Capture Migrations ---
    // Add first_seen_at and source_excerpt to facts table
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN first_seen_at TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN source_excerpt TEXT")
        .execute(pool)
        .await;

    // Backfill legacy source values: facts with an unknown source are marked
    // "inferred". The allowlist MUST contain every source value live writers
    // stamp ('progressive' from progressive extraction, 'agent' from
    // remember_fact, 'task_learning' from task learning, 'consolidation' from
    // memory consolidation, plus the stamped 'user_stated'/'derived'
    // provenance values) — this runs on every startup, so anything missing
    // here gets its provenance permanently scrubbed.
    sqlx::query("UPDATE facts SET source = 'inferred' WHERE source NOT IN ('consolidation', 'user_stated', 'derived', 'inferred', 'progressive', 'agent', 'task_learning')")
        .execute(pool)
        .await?;

    // --- Facts History Migration ---
    // Ensure facts can keep superseded history while enforcing a single active
    // row per (category, key).
    if let Err(e) = super::migrate_facts_history_schema(pool).await {
        tracing::warn!("Failed to migrate facts schema for history: {}", e);
    }

    // --- Goals/Tasks/Schedules (cleanup/unification) ---
    //
    // Historical schemas:
    // - `goals` (INTEGER PRIMARY KEY): personal memory goals (legacy)
    // - `scheduled_tasks`: legacy scheduler rows
    // - prior orchestration schema: `goals_v3`, `tasks_v3`, `task_activity_v3`
    //
    // Target schema:
    // - `goals` (TEXT PRIMARY KEY) with `domain` gating ("orchestration" vs "personal")
    // - `tasks`, `task_activity`
    // - `goal_schedules` (multiple schedules per goal with per-schedule state)
    //
    // Safety goals:
    // - Transactional table renames (all succeed or none)
    // - Legacy tables preserved as *_deprecated for recovery (not dropped)
    // - Idempotent (safe to run multiple times)

    let has_goals_v3 = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='goals_v3' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();
    let has_tasks_v3 = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks_v3' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();
    let has_task_activity_v3 = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_activity_v3' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();
    let has_scheduled_tasks = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='scheduled_tasks' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();

    let has_goals = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='goals' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();

    let (goals_has_goal_type, goals_has_legacy_int_id) = if has_goals {
        let cols = sqlx::query("PRAGMA table_info(goals)")
            .fetch_all(pool)
            .await?;
        let mut has_goal_type = false;
        let mut has_legacy_int_id = false;
        for name in cols
            .iter()
            .filter_map(|r| r.try_get::<String, _>("name").ok())
        {
            if name == "goal_type" {
                has_goal_type = true;
            } else if name == "legacy_int_id" {
                has_legacy_int_id = true;
            }
        }
        (has_goal_type, has_legacy_int_id)
    } else {
        (false, false)
    };
    let has_legacy_goals = has_goals && !goals_has_goal_type;

    let has_legacy_goals_deprecated = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_goals_legacy_deprecated' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();

    // Keep the deprecated table for recovery, but only re-run heavy goal schema
    // unification when it still contains rows that are not represented in unified goals.
    let legacy_goals_deprecated_needs_migration = if has_legacy_goals_deprecated {
        if !has_goals || !goals_has_goal_type || !goals_has_legacy_int_id {
            true
        } else {
            sqlx::query_scalar::<_, i64>(
                "SELECT 1
                 FROM _goals_legacy_deprecated lg
                 WHERE NOT EXISTS (
                     SELECT 1
                     FROM goals g
                     WHERE g.domain = 'personal'
                       AND g.legacy_int_id = lg.id
                 )
                 LIMIT 1",
            )
            .fetch_optional(pool)
            .await?
            .is_some()
        }
    } else {
        false
    };

    let should_unify_goal_schema = has_goals_v3
        || has_tasks_v3
        || has_task_activity_v3
        || has_scheduled_tasks
        || has_legacy_goals
        || legacy_goals_deprecated_needs_migration;

    if should_unify_goal_schema {
        tracing::info!(
            "Migrating database: unifying goals/tasks schema (legacy + prior schema -> clean names)"
        );

        // Best-effort datetime parser for legacy rows.
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

        let mut tx = pool.begin().await?;

        // Helper: column existence check (works even if the table doesn't exist).
        async fn column_exists(
            tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
            table: &str,
            column: &str,
        ) -> anyhow::Result<bool> {
            let rows = sqlx::query(&format!("PRAGMA table_info({})", table))
                .fetch_all(&mut **tx)
                .await?;
            Ok(rows
                .iter()
                .filter_map(|r| r.try_get::<String, _>("name").ok())
                .any(|n| n == column))
        }

        async fn table_exists(
            tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
            name: &str,
        ) -> anyhow::Result<bool> {
            Ok(sqlx::query_scalar::<_, i64>(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            )
            .bind(name)
            .fetch_optional(&mut **tx)
            .await?
            .is_some())
        }

        // 1) If a legacy `goals` table exists (INTEGER PK), rename it out of the way.
        // Drop the legacy index first to avoid name collisions when we create new indexes.
        let goals_is_legacy = table_exists(&mut tx, "goals").await?
            && !column_exists(&mut tx, "goals", "goal_type").await?;
        if goals_is_legacy && !table_exists(&mut tx, "_goals_legacy_deprecated").await? {
            let _ = sqlx::query("DROP INDEX IF EXISTS idx_goals_status")
                .execute(&mut *tx)
                .await;
            sqlx::query("ALTER TABLE goals RENAME TO _goals_legacy_deprecated")
                .execute(&mut *tx)
                .await?;
        }

        // 2) Rename prior orchestration tables to clean names.
        if table_exists(&mut tx, "goals_v3").await? && !table_exists(&mut tx, "goals").await? {
            sqlx::query("ALTER TABLE goals_v3 RENAME TO goals")
                .execute(&mut *tx)
                .await?;
        }
        if table_exists(&mut tx, "tasks_v3").await? && !table_exists(&mut tx, "tasks").await? {
            sqlx::query("ALTER TABLE tasks_v3 RENAME TO tasks")
                .execute(&mut *tx)
                .await?;
        }
        if table_exists(&mut tx, "task_activity_v3").await?
            && !table_exists(&mut tx, "task_activity").await?
        {
            sqlx::query("ALTER TABLE task_activity_v3 RENAME TO task_activity")
                .execute(&mut *tx)
                .await?;
        }

        // 3) Drop old index names (SQLite keeps index names on table rename).
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_goals_v3_status")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_goals_v3_session")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_tasks_v3_goal")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_tasks_v3_status")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_task_activity_v3_task")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_task_activity_v3_created_at")
            .execute(&mut *tx)
            .await;

        // 4) Create clean tables if missing (fresh installs or legacy DBs).
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT 'orchestration',
                goal_type TEXT NOT NULL DEFAULT 'finite',
                status TEXT NOT NULL DEFAULT 'active',
                priority TEXT NOT NULL DEFAULT 'medium',
                conditions TEXT,
                context TEXT,
                resources TEXT,
                budget_per_check INTEGER,
                budget_daily INTEGER,
                tokens_used_today INTEGER NOT NULL DEFAULT 0,
                tokens_used_day TEXT NOT NULL DEFAULT '1970-01-01',
                last_useful_action TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT,
                parent_goal_id TEXT,
                session_id TEXT NOT NULL,
                notified_at TEXT,
                notification_attempts INTEGER NOT NULL DEFAULT 0,
                dispatch_failures INTEGER NOT NULL DEFAULT 0,
                progress_notes TEXT,
                source_episode_id INTEGER REFERENCES episodes(id),
                legacy_int_id INTEGER
            )",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
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
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS task_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                activity_type TEXT NOT NULL,
                tool_name TEXT,
                tool_args TEXT,
                result TEXT,
                success INTEGER,
                tokens_used INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS goal_schedules (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                cron_expr TEXT NOT NULL,
                tz TEXT NOT NULL DEFAULT 'local',
                original_schedule TEXT,
                fire_policy TEXT NOT NULL DEFAULT 'coalesce',
                is_one_shot INTEGER NOT NULL DEFAULT 0,
                is_paused INTEGER NOT NULL DEFAULT 0,
                last_run_at TEXT,
                next_run_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&mut *tx)
        .await?;

        // 5) Ensure new columns exist on renamed goals table.
        let _ = sqlx::query(
            "ALTER TABLE goals ADD COLUMN domain TEXT NOT NULL DEFAULT 'orchestration'",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "ALTER TABLE goals ADD COLUMN tokens_used_day TEXT NOT NULL DEFAULT '1970-01-01'",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "ALTER TABLE goals ADD COLUMN notification_attempts INTEGER NOT NULL DEFAULT 0",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query(
            "ALTER TABLE goals ADD COLUMN dispatch_failures INTEGER NOT NULL DEFAULT 0",
        )
        .execute(&mut *tx)
        .await;
        let _ = sqlx::query("ALTER TABLE goals ADD COLUMN progress_notes TEXT")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("ALTER TABLE goals ADD COLUMN source_episode_id INTEGER")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("ALTER TABLE goals ADD COLUMN legacy_int_id INTEGER")
            .execute(&mut *tx)
            .await;

        // 6) Create clean indexes (drop potential collisions first).
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_goals_status")
            .execute(&mut *tx)
            .await;
        let _ = sqlx::query("DROP INDEX IF EXISTS idx_goals_session")
            .execute(&mut *tx)
            .await;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_session ON goals(session_id)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_domain_status ON goals(domain, status)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
            .execute(&mut *tx)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_task_activity_task ON task_activity(task_id)")
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_task_activity_created_at ON task_activity(created_at)",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_goal_schedules_goal ON goal_schedules(goal_id)",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_goal_schedules_next_run
             ON goal_schedules(next_run_at) WHERE is_paused = 0",
        )
        .execute(&mut *tx)
        .await?;

        // 7) Migrate legacy personal goals into unified `goals` (domain='personal').
        if table_exists(&mut tx, "_goals_legacy_deprecated").await? {
            sqlx::query(
                "INSERT OR IGNORE INTO goals (
                    id, description, domain, goal_type, status, priority,
                    conditions, context, resources,
                    budget_per_check, budget_daily,
                    tokens_used_today, tokens_used_day,
                    last_useful_action,
                    created_at, updated_at, completed_at,
                    parent_goal_id, session_id, notified_at,
                    notification_attempts, dispatch_failures,
                    progress_notes, source_episode_id, legacy_int_id
                )
                SELECT
                    'personal-legacy-' || id,
                    description,
                    'personal',
                    'finite',
                    COALESCE(status, 'active'),
                    COALESCE(priority, 'medium'),
                    NULL, NULL, NULL,
                    NULL, NULL,
                    0,
                    '1970-01-01',
                    NULL,
                    created_at,
                    updated_at,
                    completed_at,
                    NULL,
                    '_global',
                    NULL,
                    0,
                    0,
                    progress_notes,
                    source_episode_id,
                    id
                FROM _goals_legacy_deprecated",
            )
            .execute(&mut *tx)
            .await?;
        }

        // 8) Migrate schedules stored as `goals.schedule` into `goal_schedules`.
        if column_exists(&mut tx, "goals", "schedule").await? {
            let rows = sqlx::query(
                "SELECT id, goal_type, status, schedule, created_at, last_useful_action
                 FROM goals
                 WHERE schedule IS NOT NULL AND TRIM(schedule) != ''",
            )
            .fetch_all(&mut *tx)
            .await?;

            for r in &rows {
                let goal_id: String = r.get("id");
                let goal_type: String = r.get("goal_type");
                let status: String = r.get("status");
                let cron_expr: Option<String> = r.get("schedule");
                let created_at: String = r.get("created_at");
                let last_useful_action: Option<String> = r.get("last_useful_action");

                let Some(cron_expr) = cron_expr
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                else {
                    continue;
                };

                // Deterministic schedule ID keeps migration idempotent.
                let schedule_id = format!("sched-migrated-{}", goal_id);

                let cron: croner::Cron = match cron_expr.parse() {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Anchor next-run computation to last_useful_action or created_at,
                // matching prior behavior (so one-shots overdue on restart fire ASAP).
                let anchor_local = last_useful_action
                    .as_deref()
                    .and_then(parse_legacy_datetime_to_local)
                    .or_else(|| parse_legacy_datetime_to_local(&created_at))
                    .unwrap_or_else(chrono::Local::now);

                let next_local = match cron.find_next_occurrence(&anchor_local, false) {
                    Ok(dt) => dt,
                    Err(_) => continue,
                };

                let is_one_shot =
                    goal_type == "finite" && crate::cron_utils::is_one_shot_schedule(&cron_expr);
                let fire_policy = "coalesce";
                let tz = "local";
                let now = chrono::Utc::now().to_rfc3339();
                let next_run_at = next_local.with_timezone(&chrono::Utc).to_rfc3339();

                let schedule_paused = status == "paused";

                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO goal_schedules
                        (id, goal_id, cron_expr, tz, original_schedule, fire_policy, is_one_shot, is_paused, last_run_at, next_run_at, created_at, updated_at)
                     VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(&schedule_id)
                .bind(&goal_id)
                .bind(&cron_expr)
                .bind(tz)
                .bind(fire_policy)
                .bind(if is_one_shot { 1 } else { 0 })
                .bind(if schedule_paused { 1 } else { 0 })
                .bind(&last_useful_action)
                .bind(&next_run_at)
                .bind(&now)
                .bind(&now)
                .execute(&mut *tx)
                .await;
            }
        }

        // 9) Migrate legacy scheduled_tasks rows into goals + goal_schedules, then drop the table.
        if table_exists(&mut tx, "scheduled_tasks").await? {
            let rows = sqlx::query(
                "SELECT id, name, cron_expr, original_schedule, prompt, source, is_oneshot, is_paused,
                        last_run_at, next_run_at
                 FROM scheduled_tasks
                 ORDER BY created_at ASC",
            )
            .fetch_all(&mut *tx)
            .await?;

            let now_rfc3339 = chrono::Utc::now().to_rfc3339();
            let now_local = chrono::Local::now();

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

                let migrated_goal_id = format!("legacy-sched-{}", legacy_id);
                let description = if !legacy_prompt.trim().is_empty() {
                    legacy_prompt.trim().to_string()
                } else {
                    legacy_name.clone()
                };

                // If this goal already exists (e.g., migrated earlier by runtime code), skip creating it.
                let goal_exists =
                    sqlx::query_scalar::<_, i64>("SELECT 1 FROM goals WHERE id = ? LIMIT 1")
                        .bind(&migrated_goal_id)
                        .fetch_optional(&mut *tx)
                        .await?
                        .is_some();

                if !goal_exists {
                    let (goal_type, priority, budget_per_check, budget_daily) = if legacy_is_oneshot
                    {
                        ("finite", "medium", Some(100_000i64), Some(500_000i64))
                    } else {
                        ("continuous", "low", Some(100_000i64), Some(500_000i64))
                    };

                    let status = if legacy_is_paused { "paused" } else { "active" };

                    let ctx = serde_json::json!({
                        "migrated_from": "scheduled_tasks",
                        "legacy_task_id": legacy_id,
                        "legacy_name": legacy_name,
                        "legacy_source": legacy_source,
                        "legacy_original_schedule": legacy_original_schedule,
                        "legacy_next_run_at": legacy_next_run,
                    })
                    .to_string();

                    let _ = sqlx::query(
                        "INSERT OR IGNORE INTO goals
                            (id, description, domain, goal_type, status, priority, conditions, context, resources,
                             budget_per_check, budget_daily, tokens_used_today, tokens_used_day, last_useful_action,
                             created_at, updated_at, completed_at, parent_goal_id, session_id, notified_at,
                             notification_attempts, dispatch_failures, progress_notes, source_episode_id, legacy_int_id)
                         VALUES (?, ?, 'orchestration', ?, ?, ?, NULL, ?, NULL, ?, ?, 0, ?, ?, ?, ?, NULL, NULL, 'system', NULL, 0, 0, NULL, NULL, NULL)",
                    )
                    .bind(&migrated_goal_id)
                    .bind(&description)
                    .bind(goal_type)
                    .bind(status)
                    .bind(priority)
                    .bind(&ctx)
                    .bind(budget_per_check)
                    .bind(budget_daily)
                    .bind(chrono::Utc::now().date_naive().to_string())
                    .bind(legacy_last_run.as_deref().unwrap_or(""))
                    .bind(&now_rfc3339)
                    .bind(&now_rfc3339)
                    .execute(&mut *tx)
                    .await;
                }

                // Schedule: preserve legacy next_run_at when possible.
                let cron_expr = if legacy_is_oneshot {
                    let target_local = parse_legacy_datetime_to_local(&legacy_next_run)
                        .unwrap_or_else(|| now_local + chrono::Duration::minutes(1));
                    let effective_target = if target_local <= now_local {
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

                let next_run_at = parse_legacy_datetime_to_local(&legacy_next_run)
                    .map(|dt| dt.with_timezone(&chrono::Utc).to_rfc3339())
                    .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

                let schedule_id = format!("sched-legacy-{}", legacy_id);
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO goal_schedules
                        (id, goal_id, cron_expr, tz, original_schedule, fire_policy, is_one_shot, is_paused, last_run_at, next_run_at, created_at, updated_at)
                     VALUES (?, ?, ?, 'local', ?, 'coalesce', ?, ?, ?, ?, ?, ?)",
                )
                .bind(&schedule_id)
                .bind(&migrated_goal_id)
                .bind(&cron_expr)
                .bind(&legacy_original_schedule)
                .bind(if legacy_is_oneshot { 1 } else { 0 })
                .bind(if legacy_is_paused { 1 } else { 0 })
                .bind(&legacy_last_run)
                .bind(&next_run_at)
                .bind(&now_rfc3339)
                .bind(&now_rfc3339)
                .execute(&mut *tx)
                .await;
            }

            let _ = sqlx::query("DROP TABLE IF EXISTS scheduled_tasks")
                .execute(&mut *tx)
                .await;
        }

        tx.commit().await?;
    }

    // Ensure clean schema exists for fresh installs or already-migrated DBs.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT 'orchestration',
            goal_type TEXT NOT NULL DEFAULT 'finite',
            status TEXT NOT NULL DEFAULT 'active',
            priority TEXT NOT NULL DEFAULT 'medium',
            conditions TEXT,
            context TEXT,
            resources TEXT,
            budget_per_check INTEGER,
            budget_daily INTEGER,
            tokens_used_today INTEGER NOT NULL DEFAULT 0,
            tokens_used_day TEXT NOT NULL DEFAULT '1970-01-01',
            last_useful_action TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at TEXT,
            parent_goal_id TEXT,
            session_id TEXT NOT NULL,
            notified_at TEXT,
            notification_attempts INTEGER NOT NULL DEFAULT 0,
            dispatch_failures INTEGER NOT NULL DEFAULT 0,
            progress_notes TEXT,
            source_episode_id INTEGER REFERENCES episodes(id),
            legacy_int_id INTEGER
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
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
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS task_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            activity_type TEXT NOT NULL,
            tool_name TEXT,
            tool_args TEXT,
            result TEXT,
            success INTEGER,
            tokens_used INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS goal_schedules (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            cron_expr TEXT NOT NULL,
            tz TEXT NOT NULL DEFAULT 'local',
            original_schedule TEXT,
            fire_policy TEXT NOT NULL DEFAULT 'coalesce',
            is_one_shot INTEGER NOT NULL DEFAULT 0,
            is_paused INTEGER NOT NULL DEFAULT 0,
            last_run_at TEXT,
            next_run_at TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS scheduled_run_state (
            goal_id TEXT PRIMARY KEY REFERENCES goals(id) ON DELETE CASCADE,
            root_task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            effective_budget_per_check INTEGER NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            budget_extensions_count INTEGER NOT NULL DEFAULT 0,
            health_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    let _ = sqlx::query(
        "ALTER TABLE scheduled_run_state ADD COLUMN health_json TEXT NOT NULL DEFAULT '{}'",
    )
    .execute(pool)
    .await;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS scheduled_recovery_state (
            goal_id TEXT PRIMARY KEY REFERENCES goals(id) ON DELETE CASCADE,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            failure_budget INTEGER NOT NULL DEFAULT 3 CHECK (failure_budget BETWEEN 1 AND 10),
            disposition TEXT NOT NULL DEFAULT 'healthy'
                CHECK (disposition IN ('healthy', 'recovering', 'escalated')),
            latest_failure_kind TEXT,
            last_failed_run_id TEXT REFERENCES goal_runs(id) ON DELETE SET NULL,
            last_recovery_run_id TEXT REFERENCES goal_runs(id) ON DELETE SET NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;
    // Additive columns for automatic post-escalation recovery attempts.
    let _ = sqlx::query(
        "ALTER TABLE scheduled_recovery_state
            ADD COLUMN recovery_attempts INTEGER NOT NULL DEFAULT 0",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "ALTER TABLE scheduled_recovery_state ADD COLUMN last_recovery_attempt_at TEXT",
    )
    .execute(pool)
    .await;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS goal_run_recovery_links (
            failed_run_id TEXT NOT NULL REFERENCES goal_runs(id) ON DELETE CASCADE,
            recovery_run_id TEXT NOT NULL REFERENCES goal_runs(id) ON DELETE CASCADE,
            outcome_status TEXT NOT NULL
                CHECK (outcome_status IN ('recovering', 'verified', 'failed')),
            proof_receipt_ids_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (failed_run_id, recovery_run_id)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS scheduled_recovery_paused_schedules (
            schedule_id TEXT PRIMARY KEY REFERENCES goal_schedules(id) ON DELETE CASCADE,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            paused_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // Columns on goals added via ALTER for older migrated databases.
    let _ =
        sqlx::query("ALTER TABLE goals ADD COLUMN domain TEXT NOT NULL DEFAULT 'orchestration'")
            .execute(pool)
            .await;
    let _ = sqlx::query(
        "ALTER TABLE goals ADD COLUMN tokens_used_day TEXT NOT NULL DEFAULT '1970-01-01'",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "ALTER TABLE goals ADD COLUMN notification_attempts INTEGER NOT NULL DEFAULT 0",
    )
    .execute(pool)
    .await;
    let _ =
        sqlx::query("ALTER TABLE goals ADD COLUMN dispatch_failures INTEGER NOT NULL DEFAULT 0")
            .execute(pool)
            .await;
    let _ = sqlx::query("ALTER TABLE goals ADD COLUMN progress_notes TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE goals ADD COLUMN source_episode_id INTEGER")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE goals ADD COLUMN legacy_int_id INTEGER")
        .execute(pool)
        .await;

    // Durable collaborative-work model. The default project keeps existing
    // installations behaviorally unchanged while making project scope explicit.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS work_projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            description TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT OR IGNORE INTO work_projects (id, name, description)
         VALUES ('default', 'Default', 'Default Aidaemon work project')",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS session_work_projects (
            session_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES work_projects(id) ON DELETE CASCADE,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    let _ = sqlx::query("ALTER TABLE goals ADD COLUMN project_id TEXT NOT NULL DEFAULT 'default'")
        .execute(pool)
        .await;
    sqlx::query(
        "UPDATE goals SET project_id = 'default' WHERE project_id IS NULL OR project_id = ''",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS goal_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES work_projects(id) ON DELETE CASCADE,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            trigger_type TEXT NOT NULL,
            schedule_id TEXT REFERENCES goal_schedules(id) ON DELETE SET NULL,
            root_task_id TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            outcome_summary TEXT,
            started_at TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    // Owner-authorized autonomy. A mandate is intentionally separate from a
    // personal goal (desire) and from an intention (one agent commitment).
    // Every decision cycle is bound to exactly one durable goal run.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandates (
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL UNIQUE REFERENCES goals(id) ON DELETE CASCADE,
            source_goal_id TEXT REFERENCES goals(id) ON DELETE SET NULL,
            objective TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'paused', 'awaiting_input', 'completed', 'cancelled')),
            autonomy_mode TEXT NOT NULL DEFAULT 'bounded'
                CHECK (autonomy_mode IN ('bounded', 'autopilot')),
            authority_json TEXT NOT NULL,
            strategy_json TEXT,
            objective_control_json TEXT,
            suspension_json TEXT,
            constraints_json TEXT NOT NULL DEFAULT '[]',
            success_criteria_json TEXT NOT NULL DEFAULT '[]',
            stop_conditions_json TEXT NOT NULL DEFAULT '[]',
            min_review_secs INTEGER NOT NULL CHECK (min_review_secs > 0),
            max_review_secs INTEGER NOT NULL CHECK (max_review_secs >= min_review_secs),
            default_review_secs INTEGER NOT NULL
                CHECK (default_review_secs BETWEEN min_review_secs AND max_review_secs),
            review_effort TEXT NOT NULL DEFAULT 'balanced'
                CHECK (review_effort IN ('efficient', 'balanced', 'thorough', 'legacy_custom')),
            next_review_at TEXT NOT NULL,
            review_lease_token TEXT,
            review_lease_expires_at TEXT,
            expires_at TEXT,
            confirmed_at TEXT,
            version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
            owner_principal_id TEXT NOT NULL,
            created_by_session TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN owner_principal_id TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN objective_control_json TEXT")
        .execute(pool)
        .await;
    sqlx::query(
        "UPDATE mandates SET owner_principal_id = 'principal:' || id
         WHERE owner_principal_id IS NULL OR trim(owner_principal_id) = ''",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_principal_sessions (
            principal_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            linked_at TEXT NOT NULL,
            linked_by_session TEXT NOT NULL,
            PRIMARY KEY (principal_id, session_id)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT OR IGNORE INTO mandate_principal_sessions
            (principal_id, session_id, linked_at, linked_by_session)
         SELECT owner_principal_id, created_by_session, created_at, created_by_session
         FROM mandates",
    )
    .execute(pool)
    .await?;
    // Older mandates were assigned one random principal per mandate. Migrate
    // only channel identities whose platform semantics prove they are the
    // same private owner across bot routes; never infer group or arbitrary
    // session-name equivalence.
    let mandate_owner_rows = sqlx::query("SELECT id, created_by_session FROM mandates")
        .fetch_all(pool)
        .await?;
    for row in mandate_owner_rows {
        let mandate_id: String = row.get("id");
        let created_by_session: String = row.get("created_by_session");
        let Some(principal_id) =
            crate::session::stable_private_owner_principal_id(&created_by_session)
        else {
            continue;
        };
        sqlx::query("UPDATE mandates SET owner_principal_id = ? WHERE id = ?")
            .bind(&principal_id)
            .bind(&mandate_id)
            .execute(pool)
            .await?;
        sqlx::query(
            "INSERT OR IGNORE INTO mandate_principal_sessions
                (principal_id, session_id, linked_at, linked_by_session)
             VALUES (?, ?, datetime('now'), ?)",
        )
        .bind(&principal_id)
        .bind(&created_by_session)
        .bind(&created_by_session)
        .execute(pool)
        .await?;
    }
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_principal_sessions_session
         ON mandate_principal_sessions(session_id, principal_id)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_objective_measurements (
            id TEXT PRIMARY KEY,
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            mandate_version INTEGER NOT NULL CHECK (mandate_version > 0),
            goal_run_id TEXT NOT NULL REFERENCES goal_runs(id) ON DELETE CASCADE,
            value_micros INTEGER NOT NULL,
            confidence_bps INTEGER NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
            evidence_receipt_ids_json TEXT NOT NULL,
            attributed_intention_ids_json TEXT NOT NULL DEFAULT '[]',
            observed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (mandate_id, goal_run_id, observed_at)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_measurements_recent
         ON mandate_objective_measurements(mandate_id, observed_at DESC)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_ownership_transfers (
            id TEXT PRIMARY KEY,
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            principal_id TEXT NOT NULL,
            from_session_id TEXT NOT NULL,
            to_session_id TEXT NOT NULL,
            from_version INTEGER NOT NULL,
            to_version INTEGER NOT NULL,
            transferred_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;
    let _ = sqlx::query(
        "ALTER TABLE mandates ADD COLUMN autonomy_mode TEXT NOT NULL DEFAULT 'bounded'
         CHECK (autonomy_mode IN ('bounded', 'autopilot'))",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "ALTER TABLE mandates ADD COLUMN review_effort TEXT NOT NULL DEFAULT 'balanced'
         CHECK (review_effort IN ('efficient', 'balanced', 'thorough', 'legacy_custom'))",
    )
    .execute(pool)
    .await;

    // One immutable aggregate budget per autonomous decision cycle. The
    // short-lived call lease serializes lead/executor model calls so they
    // cannot each observe the same remaining balance and overspend in
    // parallel. Usage survives daemon restarts and is never auto-extended.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_run_token_budgets (
            goal_run_id TEXT PRIMARY KEY REFERENCES goal_runs(id) ON DELETE CASCADE,
            budget_per_cycle INTEGER NOT NULL CHECK (budget_per_cycle > 0),
            tokens_used INTEGER NOT NULL DEFAULT 0 CHECK (tokens_used >= 0),
            call_lease_token TEXT,
            call_lease_expires_at TEXT,
            call_dispatched INTEGER NOT NULL DEFAULT 0 CHECK (call_dispatched IN (0, 1)),
            call_tokens_used_before INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            CHECK (
                (call_lease_token IS NULL AND call_lease_expires_at IS NULL
                    AND call_dispatched = 0 AND call_tokens_used_before IS NULL)
                OR (call_lease_token IS NOT NULL AND call_lease_expires_at IS NOT NULL
                    AND ((call_dispatched = 0 AND call_tokens_used_before IS NULL)
                         OR (call_dispatched = 1 AND call_tokens_used_before IS NOT NULL)))
            )
        )",
    )
    .execute(pool)
    .await?;
    let _ = sqlx::query(
        "ALTER TABLE mandate_run_token_budgets
         ADD COLUMN call_dispatched INTEGER NOT NULL DEFAULT 0 CHECK (call_dispatched IN (0, 1))",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "ALTER TABLE mandate_run_token_budgets ADD COLUMN call_tokens_used_before INTEGER",
    )
    .execute(pool)
    .await;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_decision_cycles (
            id TEXT PRIMARY KEY,
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            goal_run_id TEXT NOT NULL UNIQUE REFERENCES goal_runs(id) ON DELETE CASCADE,
            mandate_version INTEGER NOT NULL,
            outcome TEXT NOT NULL CHECK (outcome IN ('act', 'wait', 'ask', 'stop')),
            activity_level TEXT NOT NULL DEFAULT 'quiet'
                CHECK (activity_level IN ('quiet', 'active', 'urgent')),
            rationale TEXT NOT NULL,
            belief_snapshot TEXT,
            evidence_receipt_ids_json TEXT NOT NULL DEFAULT '[]'
                CHECK (json_valid(evidence_receipt_ids_json)),
            question TEXT,
            termination_kind TEXT
                CHECK (termination_kind IS NULL OR termination_kind IN (
                    'success_criteria_satisfied', 'stop_condition_met', 'safety_termination'
                )),
            termination_match TEXT,
            reconsider_at TEXT,
            action_attempts INTEGER NOT NULL DEFAULT 0 CHECK (action_attempts >= 0),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    // Pre-release development databases may already contain the first mandate
    // draft. Keep startup migration idempotent while backfilling the durable
    // wake/lease and policy-version columns added before the first release.
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN next_review_at TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN review_lease_token TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN review_lease_expires_at TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN confirmed_at TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN strategy_json TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE mandates ADD COLUMN suspension_json TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query(
        "ALTER TABLE mandate_decision_cycles
         ADD COLUMN evidence_receipt_ids_json TEXT NOT NULL DEFAULT '[]'",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "ALTER TABLE mandate_decision_cycles
         ADD COLUMN activity_level TEXT NOT NULL DEFAULT 'quiet'
         CHECK (activity_level IN ('quiet', 'active', 'urgent'))",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query("ALTER TABLE mandate_decision_cycles ADD COLUMN termination_kind TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE mandate_decision_cycles ADD COLUMN termination_match TEXT")
        .execute(pool)
        .await;
    sqlx::query(
        "UPDATE mandates
         SET next_review_at = COALESCE(next_review_at, updated_at, created_at, datetime('now'))",
    )
    .execute(pool)
    .await?;
    // Backfill only states that could have been reached after confirmation in
    // the pre-release schema. A paused mandate whose controller is still
    // pending confirmation deliberately remains unconfirmed, as do cancelled
    // records where provenance can no longer be inferred safely.
    sqlx::query(
        "UPDATE mandates
         SET confirmed_at = COALESCE(confirmed_at, created_at, updated_at, datetime('now'))
         WHERE confirmed_at IS NULL
           AND (
               status IN ('active', 'awaiting_input', 'completed')
               OR (
                   status = 'paused'
                   AND EXISTS (
                       SELECT 1 FROM goals g
                       WHERE g.id = mandates.goal_id AND g.status = 'paused'
                   )
               )
           )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "UPDATE mandates
         SET suspension_json = json_object(
             'kind', CASE WHEN status = 'paused' THEN 'owner_paused' ELSE 'review_failed' END,
             'reason_code', 'legacy_state_backfill',
             'created_at', COALESCE(updated_at, created_at, datetime('now'))
         )
         WHERE suspension_json IS NULL
           AND confirmed_at IS NOT NULL
           AND status IN ('paused', 'awaiting_input')",
    )
    .execute(pool)
    .await?;
    let added_mandate_version = sqlx::query(
        "ALTER TABLE mandate_decision_cycles
         ADD COLUMN mandate_version INTEGER NOT NULL DEFAULT 1",
    )
    .execute(pool)
    .await
    .is_ok();
    if added_mandate_version {
        sqlx::query(
            "UPDATE mandate_decision_cycles
             SET mandate_version = COALESCE(
                 (SELECT m.version FROM mandates m
                  WHERE m.id = mandate_decision_cycles.mandate_id),
                 mandate_version
             )",
        )
        .execute(pool)
        .await?;
    }

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS intentions (
            id TEXT PRIMARY KEY,
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            decision_cycle_id TEXT NOT NULL UNIQUE REFERENCES mandate_decision_cycles(id) ON DELETE CASCADE,
            goal_run_id TEXT NOT NULL REFERENCES goal_runs(id) ON DELETE CASCADE,
            description TEXT NOT NULL,
            rationale TEXT NOT NULL,
            value_criterion TEXT,
            expected_benefit TEXT,
            risk TEXT,
            invalidation_criteria TEXT,
            status TEXT NOT NULL DEFAULT 'committed',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            completed_at TEXT
        )",
    )
    .execute(pool)
    .await?;
    // Existing installations may have the pre-value-contract intentions
    // table. Historical rows remain nullable and readable; every new
    // value-contract ACT is validated before insertion.
    let _ = sqlx::query("ALTER TABLE intentions ADD COLUMN value_criterion TEXT")
        .execute(pool)
        .await;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_learning_notes (
            id TEXT PRIMARY KEY,
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            mandate_version INTEGER NOT NULL CHECK (mandate_version > 0),
            learned_in_decision_cycle_id TEXT NOT NULL
                REFERENCES mandate_decision_cycles(id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            evidence_receipt_ids_json TEXT NOT NULL CHECK (json_valid(evidence_receipt_ids_json)),
            created_at TEXT NOT NULL,
            UNIQUE(mandate_id, learned_in_decision_cycle_id, summary)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_learning_notes_recent
         ON mandate_learning_notes(mandate_id, created_at DESC)",
    )
    .execute(pool)
    .await?;

    // Append-only adaptive operating-strategy revisions. They are deliberately
    // separate from the owner-confirmed mandate policy and therefore cannot
    // grant authority. The latest revision for one key is its current node.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_strategy_revisions (
            id TEXT PRIMARY KEY,
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            mandate_version INTEGER NOT NULL CHECK (mandate_version > 0),
            decision_cycle_id TEXT NOT NULL
                REFERENCES mandate_decision_cycles(id) ON DELETE CASCADE,
            strategy_key TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('reinforce', 'explore', 'avoid', 'retire')),
            guidance TEXT NOT NULL,
            confidence_bps INTEGER NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
            evidence_receipt_ids_json TEXT NOT NULL CHECK (json_valid(evidence_receipt_ids_json)),
            created_at TEXT NOT NULL,
            UNIQUE(decision_cycle_id, strategy_key)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_strategy_revisions_latest
         ON mandate_strategy_revisions(mandate_id, strategy_key, created_at DESC, id DESC)",
    )
    .execute(pool)
    .await?;

    // Content-free, deduplicated wake receipts for structured external signals.
    // Raw webhook bodies never enter this table or the mandate prompt.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_wake_signals (
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            mandate_version INTEGER NOT NULL CHECK (mandate_version > 0),
            signal_digest TEXT NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN (
                'mention', 'reply', 'reaction', 'metric_change',
                'delivery_failure', 'external_change'
            )),
            source TEXT NOT NULL,
            target_url TEXT NOT NULL,
            account_id TEXT,
            occurred_at TEXT NOT NULL,
            received_at TEXT NOT NULL,
            PRIMARY KEY(mandate_id, signal_digest)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_wake_signals_recent
         ON mandate_wake_signals(mandate_id, received_at DESC)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_reconciliations (
            id TEXT PRIMARY KEY,
            mandate_id TEXT NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
            suspended_version INTEGER NOT NULL CHECK (suspended_version > 0),
            suspension_kind TEXT NOT NULL,
            resolution TEXT NOT NULL CHECK (resolution IN (
                'confirmed_effect_occurred', 'confirmed_no_effect', 'abandon_attempt'
            )),
            owner_guidance TEXT NOT NULL,
            resolved_by_session TEXT NOT NULL,
            resolved_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;

    // Cross-cycle mutation safety ledger. This table intentionally contains
    // only typed identifiers and compact execution facts; raw tool arguments,
    // bodies, outputs, errors, and credentials belong nowhere in this ledger.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS mandate_mutation_attempts (
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
                CHECK (status IN ('reserved', 'never_dispatched', 'succeeded', 'failed', 'ambiguous')),
            outcome_evidence TEXT
                CHECK (outcome_evidence IS NULL OR outcome_evidence IN ('tool_reported', 'structured_metadata')),
            http_status INTEGER CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
            exit_code INTEGER,
            reserved_at TEXT NOT NULL,
            dispatch_claimed_at TEXT,
            completed_at TEXT,
            UNIQUE (decision_cycle_id, reserved_action_attempt)
        )",
    )
    .execute(pool)
    .await?;
    // Development and upgrade databases may already have the v1 ledger from
    // an earlier build. Duplicate-column is the expected no-op on fresh/newer
    // schemas; a successful ALTER upgrades the old table in place.
    let _added_dispatch_claim = sqlx::query(
        "ALTER TABLE mandate_mutation_attempts
         ADD COLUMN dispatch_claimed_at TEXT",
    )
    .execute(pool)
    .await
    .is_ok();

    // SQLite cannot widen an inline CHECK constraint in place. Rebuild the
    // pre-release ledger once so invalidated reservations that never crossed
    // the final dispatcher boundary can be closed without misclassifying them
    // as an ambiguous external effect.
    let mutation_attempts_schema = sqlx::query(
        "SELECT sql FROM sqlite_master
         WHERE type = 'table' AND name = 'mandate_mutation_attempts'",
    )
    .fetch_optional(pool)
    .await?
    .and_then(|row| row.try_get::<Option<String>, _>("sql").ok().flatten())
    .unwrap_or_default();
    if !mutation_attempts_schema.contains("never_dispatched") {
        let mut tx = pool.begin().await?;
        sqlx::query("DROP TABLE IF EXISTS mandate_mutation_attempts_v2")
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "CREATE TABLE mandate_mutation_attempts_v2 (
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
                    CHECK (status IN ('reserved', 'never_dispatched', 'succeeded', 'failed', 'ambiguous')),
                outcome_evidence TEXT
                    CHECK (outcome_evidence IS NULL OR outcome_evidence IN ('tool_reported', 'structured_metadata')),
                http_status INTEGER CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
                exit_code INTEGER,
                reserved_at TEXT NOT NULL,
                dispatch_claimed_at TEXT,
                completed_at TEXT,
                UNIQUE (decision_cycle_id, reserved_action_attempt)
            )",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "INSERT INTO mandate_mutation_attempts_v2
                (id, mandate_id, mandate_version, decision_cycle_id, goal_run_id,
                 intention_id, root_task_id, root_task_attempt_id, task_id,
                 task_attempt_id, reserved_action_attempt, action_digest,
                 tool_call_id, tool_name, mutation_effects_json, targets_json,
                 account_identifiers_json, status, outcome_evidence, http_status,
                 exit_code, reserved_at, dispatch_claimed_at, completed_at)
             SELECT id, mandate_id, mandate_version, decision_cycle_id, goal_run_id,
                    intention_id, root_task_id, root_task_attempt_id, task_id,
                    task_attempt_id, reserved_action_attempt, action_digest,
                    tool_call_id, tool_name, mutation_effects_json, targets_json,
                    account_identifiers_json, status, outcome_evidence, http_status,
                    exit_code, reserved_at, dispatch_claimed_at, completed_at
             FROM mandate_mutation_attempts",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query("DROP TABLE mandate_mutation_attempts")
            .execute(&mut *tx)
            .await?;
        sqlx::query("ALTER TABLE mandate_mutation_attempts_v2 RENAME TO mandate_mutation_attempts")
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
    }

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandates_due
         ON mandates(status, next_review_at, review_lease_expires_at)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_cycles_mandate
         ON mandate_decision_cycles(mandate_id, created_at DESC)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_intentions_mandate
         ON intentions(mandate_id, created_at DESC)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_mutations_quota
         ON mandate_mutation_attempts(mandate_id, reserved_at)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_mandate_mutations_run_status
         ON mandate_mutation_attempts(goal_run_id, status)",
    )
    .execute(pool)
    .await?;

    let _ = sqlx::query("ALTER TABLE tasks ADD COLUMN goal_run_id TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE tasks ADD COLUMN current_attempt_id TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE tasks ADD COLUMN worker_profile_id TEXT")
        .execute(pool)
        .await;
    let _ =
        sqlx::query("ALTER TABLE tasks ADD COLUMN workspace_policy TEXT NOT NULL DEFAULT 'shared'")
            .execute(pool)
            .await;
    let _ = sqlx::query(
        "ALTER TABLE tasks ADD COLUMN workspace_policy_explicit INTEGER NOT NULL DEFAULT 0",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query("ALTER TABLE tasks ADD COLUMN task_kind TEXT NOT NULL DEFAULT 'work'")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE tasks ADD COLUMN visibility TEXT NOT NULL DEFAULT 'internal'")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE tasks ADD COLUMN version INTEGER NOT NULL DEFAULT 0")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE tasks ADD COLUMN updated_at TEXT")
        .execute(pool)
        .await;
    sqlx::query(
        "UPDATE tasks
         SET updated_at = COALESCE(updated_at, completed_at, started_at, created_at)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS worker_profiles (
            id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES work_projects(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            specialist TEXT NOT NULL,
            model TEXT,
            tools_json TEXT,
            max_iterations INTEGER,
            tool_budget INTEGER,
            timeout_secs INTEGER,
            max_concurrency INTEGER NOT NULL DEFAULT 1 CHECK (max_concurrency > 0),
            workspace_policy TEXT NOT NULL DEFAULT 'shared',
            memory_scope TEXT NOT NULL DEFAULT 'project',
            version INTEGER NOT NULL DEFAULT 1,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(project_id, name)
        )",
    )
    .execute(pool)
    .await?;

    for (id, name, max_concurrency, workspace_policy) in [
        ("profile-task-lead", "task_lead", 2_i64, "shared"),
        ("profile-executor", "executor", 4_i64, "shared"),
        ("profile-research", "research", 4_i64, "shared"),
        (
            "profile-artifact-writer",
            "artifact_writer",
            2_i64,
            "shared",
        ),
        ("profile-code", "code", 2_i64, "shared"),
        (
            "profile-browser-verifier",
            "browser_verifier",
            2_i64,
            "shared",
        ),
        ("profile-review", "review", 2_i64, "shared"),
        ("profile-comms-draft", "comms_draft", 2_i64, "shared"),
        ("profile-generic", "generic", 2_i64, "shared"),
    ] {
        sqlx::query(
            "INSERT OR IGNORE INTO worker_profiles
                (id, project_id, name, specialist, max_concurrency, workspace_policy)
             VALUES (?, NULL, ?, ?, ?, ?)",
        )
        .bind(id)
        .bind(name)
        .bind(name)
        .bind(max_concurrency)
        .bind(workspace_policy)
        .execute(pool)
        .await?;
    }

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS task_attempts (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            goal_run_id TEXT NOT NULL REFERENCES goal_runs(id) ON DELETE CASCADE,
            worker_profile_id TEXT REFERENCES worker_profiles(id) ON DELETE SET NULL,
            worker_instance_id TEXT NOT NULL,
            lease_token TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL,
            last_heartbeat_at TEXT NOT NULL,
            workspace_id TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS task_journal (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES work_projects(id) ON DELETE CASCADE,
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            goal_run_id TEXT NOT NULL REFERENCES goal_runs(id) ON DELETE CASCADE,
            task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
            attempt_id TEXT REFERENCES task_attempts(id) ON DELETE SET NULL,
            entry_type TEXT NOT NULL,
            actor_type TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            source_channel TEXT,
            body TEXT NOT NULL,
            payload TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS task_handoffs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            attempt_id TEXT NOT NULL REFERENCES task_attempts(id) ON DELETE CASCADE,
            summary TEXT NOT NULL,
            artifacts_json TEXT NOT NULL DEFAULT '[]',
            verification_json TEXT NOT NULL DEFAULT '[]',
            remaining_risk TEXT,
            next_step TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS task_workspaces (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            attempt_id TEXT NOT NULL REFERENCES task_attempts(id) ON DELETE CASCADE,
            backend_id TEXT NOT NULL,
            policy TEXT NOT NULL,
            root_path TEXT NOT NULL,
            branch_name TEXT,
            base_ref TEXT,
            head_ref TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            released_at TEXT
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS work_channel_links (
            goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
            channel_session_id TEXT NOT NULL,
            thread_ref TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (goal_id, channel_session_id)
        )",
    )
    .execute(pool)
    .await?;

    // Episode analysis historically inserted inferred goal mentions without a
    // domain, so the orchestration default made them executable work. Reclassify
    // only rows that still have no task or schedule evidence; preserve them as
    // inert personal observations for audit and possible later confirmation.
    sqlx::query(
        "UPDATE goals
         SET domain = 'personal',
             status = CASE
                 WHEN status IN ('active', 'pending', 'pending_confirmation')
                 THEN 'observed'
                 ELSE status
             END,
             updated_at = datetime('now')
         WHERE domain = 'orchestration'
           AND source_episode_id IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.goal_id = goals.id)
           AND NOT EXISTS (SELECT 1 FROM goal_schedules s WHERE s.goal_id = goals.id)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "DELETE FROM work_channel_links
         WHERE goal_id IN (
             SELECT id FROM goals
             WHERE domain = 'personal' AND source_episode_id IS NOT NULL
         )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "UPDATE goal_runs
         SET status = 'cancelled',
             outcome_summary = COALESCE(
                 outcome_summary,
                 'Reclassified as a non-executable personal observation'
             ),
             completed_at = COALESCE(completed_at, datetime('now')),
             updated_at = datetime('now')
         WHERE status IN ('pending', 'running', 'blocked')
           AND goal_id IN (
               SELECT g.id FROM goals g
               WHERE g.domain = 'personal'
                 AND g.source_episode_id IS NOT NULL
                 AND NOT EXISTS (SELECT 1 FROM tasks t WHERE t.goal_id = g.id)
           )",
    )
    .execute(pool)
    .await?;

    // Existing goal/task data gets one explicit legacy run. New scheduled
    // firings create a fresh run before any task is inserted.
    sqlx::query(
        "INSERT OR IGNORE INTO goal_runs
            (id, project_id, goal_id, trigger_type, status, outcome_summary,
             started_at, completed_at, created_at, updated_at)
         SELECT
            'run-legacy-' || g.id,
            COALESCE(NULLIF(g.project_id, ''), 'default'),
            g.id,
            'legacy',
            CASE
                WHEN g.status IN ('completed', 'failed', 'cancelled') THEN g.status
                WHEN EXISTS (
                    SELECT 1 FROM tasks t
                    WHERE t.goal_id = g.id AND t.status = 'blocked'
                ) THEN 'blocked'
                ELSE 'running'
            END,
            NULL,
            g.created_at,
            CASE WHEN g.status IN ('completed', 'failed', 'cancelled')
                 THEN COALESCE(g.completed_at, g.updated_at) ELSE NULL END,
            g.created_at,
            g.updated_at
         FROM goals g
         WHERE g.domain = 'orchestration'",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "UPDATE tasks
         SET goal_run_id = COALESCE(
             goal_run_id,
             (SELECT gr.id FROM goal_runs gr
              WHERE gr.goal_id = tasks.goal_id
              ORDER BY julianday(gr.started_at) DESC, gr.id DESC LIMIT 1)
         )
         WHERE goal_run_id IS NULL OR goal_run_id = ''",
    )
    .execute(pool)
    .await?;

    let _ = sqlx::query("ALTER TABLE scheduled_run_state ADD COLUMN goal_run_id TEXT")
        .execute(pool)
        .await;
    sqlx::query(
        "UPDATE scheduled_run_state
         SET goal_run_id = COALESCE(
             goal_run_id,
             (SELECT t.goal_run_id FROM tasks t
              WHERE t.id = scheduled_run_state.root_task_id)
         )",
    )
    .execute(pool)
    .await?;

    // The first run-isolation migration grouped all pre-existing recurring
    // history into one legacy run. Once no live task or scheduled-run state
    // references that run, close it so current board views do not present old
    // outcomes as an active recurring cycle. Tasks and journals remain intact.
    sqlx::query(
        "UPDATE goal_runs
         SET status = CASE
                 WHEN EXISTS (
                     SELECT 1 FROM tasks t
                     WHERE t.goal_run_id = goal_runs.id
                       AND t.status IN ('failed', 'blocked', 'interrupted')
                 ) THEN 'failed'
                 ELSE 'completed'
             END,
             outcome_summary = COALESCE(
                 outcome_summary,
                 'Archived legacy history after recurring-run isolation'
             ),
             completed_at = COALESCE(completed_at, datetime('now')),
             updated_at = datetime('now')
         WHERE trigger_type = 'legacy'
           AND status IN ('pending', 'running', 'blocked')
           AND EXISTS (
               SELECT 1 FROM goals g
               WHERE g.id = goal_runs.goal_id
                 AND g.goal_type = 'continuous'
           )
           AND NOT EXISTS (
               SELECT 1 FROM tasks t
               WHERE t.goal_run_id = goal_runs.id
                 AND t.status IN ('pending', 'claimed', 'running')
           )
           AND NOT EXISTS (
               SELECT 1 FROM scheduled_run_state s
               WHERE s.goal_run_id = goal_runs.id
           )",
    )
    .execute(pool)
    .await?;

    let _ = sqlx::query("ALTER TABLE notification_queue ADD COLUMN task_id TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE notification_queue ADD COLUMN action_token TEXT")
        .execute(pool)
        .await;

    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_goals_project_status
         ON goals(project_id, status)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_goal_runs_goal_started
         ON goal_runs(goal_id, started_at DESC)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query("DROP INDEX IF EXISTS idx_goal_runs_one_open")
        .execute(pool)
        .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_goal_runs_open
         ON goal_runs(goal_id, status)
         WHERE status IN ('pending', 'running', 'blocked')",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_tasks_goal_run_status
         ON tasks(goal_run_id, status)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_task_attempts_one_active
         ON task_attempts(task_id)
         WHERE status IN ('claimed', 'running')",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_attempts_lease
         ON task_attempts(status, lease_expires_at)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_journal_task_created
         ON task_journal(task_id, created_at)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_handoffs_task_created
         ON task_handoffs(task_id, created_at)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_workspaces_task
         ON task_workspaces(task_id, created_at)",
    )
    .execute(pool)
    .await;

    // Indexes (idempotent).
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
        .execute(pool)
        .await;
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_session ON goals(session_id)")
        .execute(pool)
        .await;
    let _ =
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_domain_status ON goals(domain, status)")
            .execute(pool)
            .await;
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id)")
        .execute(pool)
        .await;
    let _ = sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        .execute(pool)
        .await;
    let _ =
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_task_activity_task ON task_activity(task_id)")
            .execute(pool)
            .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_activity_created_at ON task_activity(created_at)",
    )
    .execute(pool)
    .await;
    // Self-correcting-resilience Component A: the inactivity watchdog does a
    // lexical MAX(created_at) per task, so all rows must share one sortable UTC
    // format. Normalize any legacy RFC3339 rows ('...T...Z') to SQLite datetime,
    // and add a composite index to serve the correlated per-task MAX.
    let _ = sqlx::query(
        "UPDATE task_activity SET created_at = datetime(created_at) WHERE created_at LIKE '%T%'",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_activity_task_created_at \
         ON task_activity(task_id, created_at)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_goal_schedules_goal ON goal_schedules(goal_id)",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_goal_schedules_next_run
         ON goal_schedules(next_run_at) WHERE is_paused = 0",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_scheduled_run_state_root_task
         ON scheduled_run_state(root_task_id)",
    )
    .execute(pool)
    .await;

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
    .execute(pool)
    .await?;

    let _ = sqlx::query("ALTER TABLE notification_queue ADD COLUMN task_id TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE notification_queue ADD COLUMN action_token TEXT")
        .execute(pool)
        .await;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_notification_queue_pending
         ON notification_queue(delivered_at, priority, created_at)
         WHERE delivered_at IS NULL",
    )
    .execute(pool)
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
    .execute(pool)
    .await?;

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

    // Normalize legacy optional task fields. Empty strings caused completed
    // tasks to be misclassified as failures by callers expecting NULL.
    sqlx::query(
        "UPDATE tasks
         SET result = NULLIF(TRIM(result), ''),
             error = NULLIF(TRIM(error), ''),
             blocker = NULLIF(TRIM(blocker), '')
         WHERE (result IS NOT NULL AND TRIM(result) = '')
            OR (error IS NOT NULL AND TRIM(error) = '')
            OR (blocker IS NOT NULL AND TRIM(blocker) = '')",
    )
    .execute(pool)
    .await?;

    // Repair the historical race where an executor durably blocked an attempt,
    // then a stale coordinator overwrote the task row as completed without an
    // unblock or a newer attempt. The latest fenced attempt is authoritative.
    sqlx::query(
        "UPDATE tasks
         SET status = 'blocked',
             blocker = COALESCE(blocker, 'Latest execution attempt is blocked.'),
             completed_at = COALESCE(
                 (SELECT a.completed_at FROM task_attempts a
                  WHERE a.task_id = tasks.id
                  ORDER BY julianday(a.started_at) DESC, a.id DESC LIMIT 1),
                 completed_at
             ),
             updated_at = datetime('now'), version = version + 1
         WHERE status = 'completed'
           AND (
               SELECT a.status FROM task_attempts a
               WHERE a.task_id = tasks.id
               ORDER BY julianday(a.started_at) DESC, a.id DESC LIMIT 1
           ) = 'blocked'",
    )
    .execute(pool)
    .await?;

    // Task-linked escalation notifications are actionable only while the task
    // remains blocked. Close obsolete critical rows so they cannot surface
    // hours later after an unblock, retry, cancellation, or completion.
    sqlx::query(
        "UPDATE notification_queue
         SET delivered_at = datetime('now')
         WHERE delivered_at IS NULL
           AND notification_type = 'escalation'
           AND task_id IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM tasks t
               WHERE t.id = notification_queue.task_id AND t.status = 'blocked'
           )",
    )
    .execute(pool)
    .await?;

    // Migration: deduplicate people entries and add unique index on LOWER(name).
    // Keeps the row with the lowest id for each name, merging interaction counts.
    let _ = sqlx::query(
        "DELETE FROM people WHERE id NOT IN (
            SELECT MIN(id) FROM people GROUP BY LOWER(name)
        )",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_people_name_unique ON people(LOWER(name))",
    )
    .execute(pool)
    .await;

    // Migration: scheduled continuous goals were historically created with incorrect
    // 5K/20K budgets. Bump them to the current continuous defaults (100K/500K).
    // Safe + idempotent.
    let _ = sqlx::query(
        "UPDATE goals
         SET budget_per_check = 100000,
             budget_daily = 500000
         WHERE domain = 'orchestration'
           AND goal_type = 'continuous'
           AND budget_per_check = 5000
           AND budget_daily = 20000
           AND EXISTS (SELECT 1 FROM goal_schedules s WHERE s.goal_id = goals.id)",
    )
    .execute(pool)
    .await;

    // Migration: raise previously standard scheduled continuous defaults
    // (50K/200K) to the newer defaults (100K/500K). This only touches goals
    // still at the exact historical defaults, so explicit user-set budgets are
    // preserved.
    let _ = sqlx::query(
        "UPDATE goals
         SET budget_per_check = 100000,
             budget_daily = 500000
         WHERE domain = 'orchestration'
           AND goal_type = 'continuous'
           AND budget_per_check = 50000
           AND budget_daily = 200000
           AND EXISTS (SELECT 1 FROM goal_schedules s WHERE s.goal_id = goals.id)",
    )
    .execute(pool)
    .await;

    // Cleanup: schedules attached to terminal goals are dead rows. They can exist
    // after migrations from legacy schemas or older bulk-cancel implementations.
    // Safe + idempotent.
    let _ = sqlx::query(
        "DELETE FROM goal_schedules
         WHERE goal_id IN (
            SELECT id FROM goals WHERE status IN ('cancelled', 'completed')
         )",
    )
    .execute(pool)
    .await;

    let _ = sqlx::query(
        "DELETE FROM scheduled_run_state
         WHERE goal_id IN (
            SELECT id FROM goals WHERE status IN ('cancelled', 'completed', 'failed')
         )",
    )
    .execute(pool)
    .await;

    // Token arithmetic is an internal runaway guard for mandates. Migrate
    // historical low defaults to the automatic balanced policy and derive a
    // daily envelope that can fund every cadence slot. This only raises
    // capacity; explicit efficient/thorough choices are persisted separately.
    let _ = sqlx::query(
        "UPDATE goals
         SET budget_per_check = MAX(COALESCE(budget_per_check, 0), 250000),
             budget_daily = MAX(
                 COALESCE(budget_daily, 0),
                 2000000,
                 ((86400 + m.default_review_secs - 1) / m.default_review_secs)
                    * MAX(COALESCE(budget_per_check, 0), 250000)
             )
         FROM mandates AS m
         WHERE m.goal_id = goals.id
           AND m.review_effort = 'balanced'
           AND (
               COALESCE(goals.budget_per_check, 0) < 250000
               OR COALESCE(goals.budget_daily, 0) < MAX(
                   2000000,
                   ((86400 + m.default_review_secs - 1) / m.default_review_secs)
                       * MAX(COALESCE(goals.budget_per_check, 0), 250000)
               )
           )
           AND goals.status IN ('active', 'pending', 'pending_confirmation')",
    )
    .execute(pool)
    .await;

    // Bound only obviously runaway historical values. Managed thorough mode
    // legitimately exceeds the old two-million-token ceiling at fast cadences.
    let _ = sqlx::query(
        "UPDATE goals
         SET budget_daily = 50000000
         WHERE budget_daily > 50000000
           AND status IN ('active', 'pending', 'pending_confirmation')",
    )
    .execute(pool)
    .await;

    // Migration: allow multiple episodes per session (for mid-session episode
    // creation in long-running conversations). The non-unique index on session_id
    // (idx_episodes_session) already exists for lookups.
    let _ = sqlx::query("DROP INDEX IF EXISTS idx_episodes_session_unique")
        .execute(pool)
        .await;

    let _ = sqlx::query(
        "CREATE TABLE IF NOT EXISTS self_correction_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id TEXT NOT NULL,
            subject_kind TEXT NOT NULL,
            approach_signature TEXT NOT NULL,
            attempt_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            blocked_reason TEXT,
            evidence_ref TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await;
    let _ = sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_sc_attempts_subject \
         ON self_correction_attempts(subject_id, created_at)",
    )
    .execute(pool)
    .await;

    // Canonical long-term memory. Domain tables remain authoritative during
    // the rolling migration; these tables are durable projections with explicit
    // provenance, validity, and replaceable search indexes.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            span_kind TEXT NOT NULL,
            source_event_id INTEGER,
            source_episode_id INTEGER UNIQUE,
            session_id TEXT,
            channel_id TEXT,
            role TEXT,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            privacy TEXT NOT NULL DEFAULT 'global',
            observed_from TEXT,
            observed_to TEXT,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            deleted_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
    )
    .execute(pool)
    .await?;
    let _ = sqlx::query("ALTER TABLE memory_spans ADD COLUMN observed_from TEXT")
        .execute(pool)
        .await;
    let _ = sqlx::query("ALTER TABLE memory_spans ADD COLUMN observed_to TEXT")
        .execute(pool)
        .await;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            source_fact_id INTEGER UNIQUE,
            source_span_id INTEGER,
            source_event_id INTEGER,
            provenance TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
            channel_id TEXT,
            privacy TEXT NOT NULL DEFAULT 'global',
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            superseded_by_claim_id INTEGER,
            deleted_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(source_span_id) REFERENCES memory_spans(id),
            FOREIGN KEY(superseded_by_claim_id) REFERENCES memory_claims(id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            channel_id TEXT,
            privacy TEXT NOT NULL DEFAULT 'global',
            deleted_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(entity_type, canonical_name)
        )",
    )
    .execute(pool)
    .await?;
    for statement in [
        "ALTER TABLE memory_entities ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
        "ALTER TABLE memory_entities ADD COLUMN merged_into_entity_id INTEGER",
        "ALTER TABLE memory_entities ADD COLUMN is_owner INTEGER NOT NULL DEFAULT 0",
    ] {
        let _ = sqlx::query(statement).execute(pool).await;
    }

    // Entity-aware personal memory. These tables extend the rolling canonical
    // projection without changing or deleting the legacy facts/people tables.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            alias_type TEXT NOT NULL,
            value TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            source TEXT NOT NULL,
            provenance TEXT,
            confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
            channel_id TEXT,
            privacy TEXT NOT NULL DEFAULT 'private',
            asserted_at TEXT NOT NULL,
            confirmed_at TEXT,
            last_confirmed_at TEXT,
            valid_from TEXT,
            valid_to TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES memory_entities(id),
            UNIQUE(entity_id, alias_type, normalized_value)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_entity_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_entity_id INTEGER NOT NULL,
            predicate TEXT NOT NULL,
            value_type TEXT NOT NULL DEFAULT 'text',
            value TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            display_value TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            source TEXT NOT NULL,
            provenance TEXT,
            confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
            source_fact_id INTEGER,
            channel_id TEXT,
            privacy TEXT NOT NULL DEFAULT 'private',
            asserted_at TEXT NOT NULL,
            confirmed_at TEXT,
            last_confirmed_at TEXT,
            valid_from TEXT,
            valid_to TEXT,
            supersedes_fact_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(subject_entity_id) REFERENCES memory_entities(id),
            FOREIGN KEY(source_fact_id) REFERENCES facts(id),
            FOREIGN KEY(supersedes_fact_id) REFERENCES memory_entity_facts(id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            target_entity_id INTEGER NOT NULL,
            inverse_relationship_id INTEGER,
            status TEXT NOT NULL DEFAULT 'active',
            source TEXT NOT NULL,
            provenance TEXT,
            confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
            source_fact_id INTEGER,
            channel_id TEXT,
            privacy TEXT NOT NULL DEFAULT 'private',
            asserted_at TEXT NOT NULL,
            confirmed_at TEXT,
            last_confirmed_at TEXT,
            valid_from TEXT,
            valid_to TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_entity_id) REFERENCES memory_entities(id),
            FOREIGN KEY(target_entity_id) REFERENCES memory_entities(id),
            FOREIGN KEY(inverse_relationship_id) REFERENCES memory_relationships(id),
            FOREIGN KEY(source_fact_id) REFERENCES facts(id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_resolution_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_kind TEXT NOT NULL,
            normalized_reference TEXT,
            candidate_entity_ids_json TEXT NOT NULL DEFAULT '[]',
            payload_json TEXT NOT NULL,
            source TEXT NOT NULL,
            source_fact_id INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT,
            UNIQUE(review_kind, normalized_reference, payload_json, status)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_write_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL,
            entity_id INTEGER,
            record_type TEXT NOT NULL,
            record_id INTEGER,
            prior_state_json TEXT,
            new_state_json TEXT,
            source TEXT NOT NULL,
            source_fact_id INTEGER,
            provenance TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES memory_entities(id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            source_claim_id INTEGER,
            confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
            channel_id TEXT,
            privacy TEXT NOT NULL DEFAULT 'global',
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            deleted_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(source_entity_id) REFERENCES memory_entities(id),
            FOREIGN KEY(target_entity_id) REFERENCES memory_entities(id),
            FOREIGN KEY(source_claim_id) REFERENCES memory_claims(id),
            UNIQUE(source_entity_id, target_entity_id, relation, source_claim_id)
        )",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_type TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            embedding_purpose TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL CHECK(embedding_dim > 0),
            content_hash TEXT NOT NULL,
            embedding BLOB NOT NULL,
            stale_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(owner_type, owner_id, embedding_purpose, embedding_model, content_hash)
        )",
    )
    .execute(pool)
    .await?;

    for statement in [
        "CREATE INDEX IF NOT EXISTS idx_memory_claims_active ON memory_claims(privacy, channel_id, valid_to, deleted_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_spans_active ON memory_spans(privacy, channel_id, valid_to, deleted_at)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_spans_source_event ON memory_spans(source_event_id) WHERE source_event_id IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_memory_edges_source ON memory_edges(source_entity_id, valid_to, deleted_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_edges_target ON memory_edges(target_entity_id, valid_to, deleted_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_lookup ON memory_embeddings(embedding_model, embedding_purpose, owner_type, stale_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_owner ON memory_embeddings(owner_type, owner_id, stale_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_entities_status ON memory_entities(entity_type, status, canonical_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_entities_one_owner ON memory_entities(is_owner) WHERE is_owner = 1 AND status = 'active'",
        "CREATE INDEX IF NOT EXISTS idx_memory_aliases_lookup ON memory_aliases(normalized_value, alias_type, status)",
        "CREATE INDEX IF NOT EXISTS idx_memory_aliases_entity ON memory_aliases(entity_id, status)",
        "CREATE INDEX IF NOT EXISTS idx_memory_entity_facts_subject ON memory_entity_facts(subject_entity_id, predicate, status)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_entity_facts_active_exact ON memory_entity_facts(subject_entity_id, predicate, normalized_value) WHERE status = 'active' AND valid_to IS NULL",
        "CREATE INDEX IF NOT EXISTS idx_memory_relationships_source ON memory_relationships(source_entity_id, relationship_type, status)",
        "CREATE INDEX IF NOT EXISTS idx_memory_relationships_target ON memory_relationships(target_entity_id, relationship_type, status)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_relationships_active_exact ON memory_relationships(source_entity_id, relationship_type, target_entity_id) WHERE status = 'active' AND valid_to IS NULL",
        "CREATE INDEX IF NOT EXISTS idx_memory_reviews_status ON memory_resolution_reviews(status, review_kind)",
        "CREATE INDEX IF NOT EXISTS idx_memory_audit_entity ON memory_write_audit(entity_id, created_at)",
    ] {
        sqlx::query(statement).execute(pool).await?;
    }

    // Normalize task dependency edges. `tasks.depends_on` remains as a JSON
    // compatibility projection, while schedulers and claims use this table as
    // their authoritative graph. The join skips invalid legacy references so
    // migration cannot manufacture an edge to a nonexistent task.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS task_dependencies (
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            depends_on_task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY(task_id, depends_on_task_id),
            CHECK(task_id <> depends_on_task_id)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_dependencies_prerequisite
         ON task_dependencies(depends_on_task_id, task_id)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "INSERT OR IGNORE INTO task_dependencies(task_id, depends_on_task_id)
         SELECT t.id, CAST(dep.value AS TEXT)
         FROM tasks t
         JOIN json_each(
            CASE
                WHEN json_valid(COALESCE(t.depends_on, '[]'))
                 AND json_type(COALESCE(t.depends_on, '[]')) = 'array'
                THEN COALESCE(t.depends_on, '[]')
                ELSE '[]'
            END
         ) dep
         JOIN tasks prerequisite ON prerequisite.id = CAST(dep.value AS TEXT)
         WHERE CAST(dep.value AS TEXT) <> t.id",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TRIGGER IF NOT EXISTS trg_task_dependencies_no_cycle
         BEFORE INSERT ON task_dependencies
         WHEN EXISTS (
            WITH RECURSIVE prerequisites(id) AS (
                SELECT NEW.depends_on_task_id
                UNION
                SELECT edge.depends_on_task_id
                FROM task_dependencies edge
                JOIN prerequisites prior ON edge.task_id = prior.id
            )
            SELECT 1 FROM prerequisites WHERE id = NEW.task_id
         )
         BEGIN
            SELECT RAISE(ABORT, 'task dependency cycle');
         END",
    )
    .execute(pool)
    .await?;

    // FTS5 is present in standard SQLite and bundled SQLCipher builds. Keep the
    // daemon usable on custom SQLite builds without it; semantic and lexical
    // fallback retrieval remain available.
    if let Err(error) = create_memory_fts(pool).await {
        tracing::warn!(%error, "SQLite FTS5 unavailable; memory full-text index disabled");
    }
    if let Err(error) = super::history_search::migrate_history_search(pool).await {
        // Conversation events are authoritative. Exact-history search is a
        // replaceable projection and must never prevent the daemon starting.
        tracing::warn!(%error, "SQLite FTS5 unavailable; exact history search disabled");
    }

    Ok(())
}

async fn create_memory_fts(pool: &SqlitePool) -> anyhow::Result<()> {
    let claims_fts_existed = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_claims_fts'",
    )
    .fetch_optional(pool)
    .await?
    .is_some();
    let spans_fts_existed = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_spans_fts'",
    )
    .fetch_optional(pool)
    .await?
    .is_some();
    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_claims_fts USING fts5(
            claim_text, content='memory_claims', content_rowid='id', tokenize='unicode61'
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_spans_fts USING fts5(
            content, content='memory_spans', content_rowid='id', tokenize='unicode61'
        )",
    )
    .execute(pool)
    .await?;
    // FTS5 secure-delete corrupts external-content indexes under the
    // multi-connection SQLCipher 3.51.2 workload used by the daemon (the same
    // update sequence is healthy through a single connection). Keep the FTS
    // projection in compatibility mode. The database remains encrypted and
    // `PRAGMA secure_delete=ON` still scrubs ordinary SQLite pages; deleted FTS
    // terms are made unqueryable by the standard tombstone/merge mechanism.
    for table in ["memory_claims_fts", "memory_spans_fts"] {
        sqlx::query(&format!(
            "INSERT INTO {table}({table}, rank) VALUES('secure-delete', 0)"
        ))
        .execute(pool)
        .await?;
    }

    // `UPDATE OF column` fires even when an UPSERT assigns the existing value.
    // Repeated delete/reinsert churn of an unchanged row can corrupt FTS5
    // secure-delete segments on SQLCipher's SQLite build. Replace legacy
    // update triggers and suppress true no-op text updates.
    for trigger in ["memory_claims_au", "memory_spans_au"] {
        sqlx::query(&format!("DROP TRIGGER IF EXISTS {trigger}"))
            .execute(pool)
            .await?;
    }

    for statement in [
        "CREATE TRIGGER IF NOT EXISTS memory_claims_ai AFTER INSERT ON memory_claims BEGIN INSERT INTO memory_claims_fts(rowid, claim_text) VALUES (new.id, new.claim_text); END",
        "CREATE TRIGGER IF NOT EXISTS memory_claims_ad AFTER DELETE ON memory_claims BEGIN INSERT INTO memory_claims_fts(memory_claims_fts, rowid, claim_text) VALUES ('delete', old.id, old.claim_text); END",
        "CREATE TRIGGER memory_claims_au AFTER UPDATE OF claim_text ON memory_claims WHEN old.claim_text IS NOT new.claim_text BEGIN INSERT INTO memory_claims_fts(memory_claims_fts, rowid, claim_text) VALUES ('delete', old.id, old.claim_text); INSERT INTO memory_claims_fts(rowid, claim_text) VALUES (new.id, new.claim_text); END",
        "CREATE TRIGGER IF NOT EXISTS memory_spans_ai AFTER INSERT ON memory_spans BEGIN INSERT INTO memory_spans_fts(rowid, content) VALUES (new.id, new.content); END",
        "CREATE TRIGGER IF NOT EXISTS memory_spans_ad AFTER DELETE ON memory_spans BEGIN INSERT INTO memory_spans_fts(memory_spans_fts, rowid, content) VALUES ('delete', old.id, old.content); END",
        "CREATE TRIGGER memory_spans_au AFTER UPDATE OF content ON memory_spans WHEN old.content IS NOT new.content BEGIN INSERT INTO memory_spans_fts(memory_spans_fts, rowid, content) VALUES ('delete', old.id, old.content); INSERT INTO memory_spans_fts(rowid, content) VALUES (new.id, new.content); END",
    ] {
        sqlx::query(statement).execute(pool).await?;
    }

    if !claims_fts_existed {
        sqlx::query("INSERT INTO memory_claims_fts(memory_claims_fts) VALUES ('rebuild')")
            .execute(pool)
            .await?;
    }
    if !spans_fts_existed {
        sqlx::query("INSERT INTO memory_spans_fts(memory_spans_fts) VALUES ('rebuild')")
            .execute(pool)
            .await?;
    }
    Ok(())
}

/// Recreate the memory search indexes from their authoritative source tables.
///
/// These FTS tables are replaceable projections. Rebuilding both together keeps
/// their trigger and secure-delete configuration consistent after either index
/// is reported corrupt during startup integrity verification.
pub(crate) async fn rebuild_memory_fts_projections(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query("DROP TABLE IF EXISTS memory_claims_fts")
        .execute(pool)
        .await?;
    sqlx::query("DROP TABLE IF EXISTS memory_spans_fts")
        .execute(pool)
        .await?;
    create_memory_fts(pool).await
}

#[cfg(test)]
mod memory_fts_trigger_tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    #[tokio::test]
    async fn no_op_content_updates_do_not_churn_or_corrupt_secure_delete_indexes() {
        let database = tempfile::NamedTempFile::new().unwrap();
        let options = SqliteConnectOptions::new()
            .filename(database.path())
            .create_if_missing(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(2)
            .connect_with(options)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE memory_claims (
                id INTEGER PRIMARY KEY,
                claim_text TEXT NOT NULL
             );
             CREATE TABLE memory_spans (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL
             )",
        )
        .execute(&pool)
        .await
        .unwrap();

        create_memory_fts(&pool).await.unwrap();
        for table in ["memory_claims_fts_config", "memory_spans_fts_config"] {
            let secure_delete: i64 =
                sqlx::query_scalar(&format!("SELECT v FROM {table} WHERE k = 'secure-delete'"))
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            assert_eq!(secure_delete, 0);
        }
        sqlx::query("INSERT INTO memory_claims(id, claim_text) VALUES (1, 'alpha claim')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO memory_spans(id, content) VALUES (1, 'alpha span')")
            .execute(&pool)
            .await
            .unwrap();
        let claim_shadow_before: (i64, i64) = sqlx::query_as(
            "SELECT COUNT(*), COALESCE(SUM(length(block)), 0) FROM memory_claims_fts_data",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let span_shadow_before: (i64, i64) = sqlx::query_as(
            "SELECT COUNT(*), COALESCE(SUM(length(block)), 0) FROM memory_spans_fts_data",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        for _ in 0..64 {
            sqlx::query("UPDATE memory_claims SET claim_text = claim_text WHERE id = 1")
                .execute(&pool)
                .await
                .unwrap();
            sqlx::query("UPDATE memory_spans SET content = content WHERE id = 1")
                .execute(&pool)
                .await
                .unwrap();
        }

        let claim_shadow_after: (i64, i64) = sqlx::query_as(
            "SELECT COUNT(*), COALESCE(SUM(length(block)), 0) FROM memory_claims_fts_data",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let span_shadow_after: (i64, i64) = sqlx::query_as(
            "SELECT COUNT(*), COALESCE(SUM(length(block)), 0) FROM memory_spans_fts_data",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(claim_shadow_after, claim_shadow_before);
        assert_eq!(span_shadow_after, span_shadow_before);

        sqlx::query("UPDATE memory_claims SET claim_text = 'beta claim' WHERE id = 1")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("UPDATE memory_spans SET content = 'beta span' WHERE id = 1")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO memory_claims_fts(memory_claims_fts) VALUES('integrity-check')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO memory_spans_fts(memory_spans_fts) VALUES('integrity-check')")
            .execute(&pool)
            .await
            .unwrap();
        let beta_claims: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_claims_fts WHERE memory_claims_fts MATCH 'beta'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let beta_spans: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_spans_fts WHERE memory_spans_fts MATCH 'beta'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!((beta_claims, beta_spans), (1, 1));

        for trigger in ["memory_claims_au", "memory_spans_au"] {
            let sql: String =
                sqlx::query_scalar("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?")
                    .bind(trigger)
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            assert!(sql.contains("WHEN old."), "trigger SQL: {sql}");
            assert!(sql.contains(" IS NOT new."), "trigger SQL: {sql}");
        }
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
