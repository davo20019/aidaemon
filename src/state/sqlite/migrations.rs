use sqlx::SqlitePool;

pub(crate) async fn migrate_state(pool: &SqlitePool) -> anyhow::Result<()> {
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
    .execute(pool)
    .await?;

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

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)",
    )
    .execute(pool)
    .await?;

    // --- Migrations for Advanced Memory ---
    // 1. Add importance column
    let _ = sqlx::query("ALTER TABLE messages ADD COLUMN importance REAL DEFAULT 0.5")
        .execute(pool)
        .await; // Ignore error if exists

    // 2. Add embedding column
    let _ = sqlx::query("ALTER TABLE messages ADD COLUMN embedding BLOB")
        .execute(pool)
        .await; // Ignore error if exists

    // Add embedding_error column if it doesn't exist
    let _ = sqlx::query("ALTER TABLE messages ADD COLUMN embedding_error TEXT")
        .execute(pool)
        .await;

    // 4. Add consolidated_at column for memory consolidation (Layer 6)
    let _ = sqlx::query("ALTER TABLE messages ADD COLUMN consolidated_at TEXT")
        .execute(pool)
        .await;

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

    // Prevent concurrent episode creation for the same session
    let _ = sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_session_unique ON episodes(session_id)",
    )
    .execute(pool)
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
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
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
            asks_before_acting INTEGER DEFAULT 1,
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
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_next_run
            ON scheduled_tasks(next_run_at) WHERE is_paused = 0",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_tasks_name_source
            ON scheduled_tasks(name) WHERE source = 'config'",
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
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
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
            scopes TEXT NOT NULL DEFAULT '[]',
            token_expires_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )",
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

    // --- Binary Embedding Storage Migration ---
    // Add embedding column to facts table for pre-computed embeddings
    let _ = sqlx::query("ALTER TABLE facts ADD COLUMN embedding BLOB")
        .execute(pool)
        .await;

    // --- Facts History Migration ---
    // Ensure facts can keep superseded history while enforcing a single active
    // row per (category, key).
    if let Err(e) = super::migrate_facts_history_schema(pool).await {
        tracing::warn!("Failed to migrate facts schema for history: {}", e);
    }

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
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_v3_status ON goals_v3(status)")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_goals_v3_session ON goals_v3(session_id)")
        .execute(pool)
        .await?;

    // Migration: add notified_at column to goals_v3 (nullable)
    let _ = sqlx::query("ALTER TABLE goals_v3 ADD COLUMN notified_at TEXT")
        .execute(pool)
        .await;

    // Migration: add notification_attempts column to goals_v3 for retry tracking
    let _ = sqlx::query(
        "ALTER TABLE goals_v3 ADD COLUMN notification_attempts INTEGER NOT NULL DEFAULT 0",
    )
    .execute(pool)
    .await;

    // Migration: add dispatch_failures column for progress-based circuit breaker
    let _ =
        sqlx::query("ALTER TABLE goals_v3 ADD COLUMN dispatch_failures INTEGER NOT NULL DEFAULT 0")
            .execute(pool)
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
    .execute(pool)
    .await?;

    sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_v3_goal ON tasks_v3(goal_id)")
        .execute(pool)
        .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_v3_status ON tasks_v3(status)")
        .execute(pool)
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
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_activity_v3_task ON task_activity_v3(task_id)",
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_task_activity_v3_created_at
         ON task_activity_v3(created_at)",
    )
    .execute(pool)
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
    .execute(pool)
    .await?;

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
            updated_at TEXT NOT NULL
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

    Ok(())
}
