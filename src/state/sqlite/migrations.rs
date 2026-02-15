use chrono::{Datelike, TimeZone, Timelike};
use sqlx::Row;
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

    let goals_has_goal_type = if has_goals {
        let cols = sqlx::query("PRAGMA table_info(goals)")
            .fetch_all(pool)
            .await?;
        cols.iter()
            .filter_map(|r| r.try_get::<String, _>("name").ok())
            .any(|n| n == "goal_type")
    } else {
        false
    };
    let has_legacy_goals = has_goals && !goals_has_goal_type;

    let has_legacy_goals_deprecated = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_goals_legacy_deprecated' LIMIT 1",
    )
    .fetch_optional(pool)
    .await?
    .is_some();

    let should_unify_goal_schema = has_goals_v3
        || has_tasks_v3
        || has_task_activity_v3
        || has_scheduled_tasks
        || has_legacy_goals
        || has_legacy_goals_deprecated;

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
	                        ("continuous", "low", Some(50_000i64), Some(200_000i64))
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

    // Migration: scheduled continuous goals were historically created with incorrect
    // 5K/20K budgets. Bump them to the standard continuous defaults (50K/200K).
    // Safe + idempotent.
    let _ = sqlx::query(
        "UPDATE goals
         SET budget_per_check = 50000,
             budget_daily = 200000
         WHERE domain = 'orchestration'
           AND goal_type = 'continuous'
           AND budget_per_check = 5000
           AND budget_daily = 20000
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

    Ok(())
}
