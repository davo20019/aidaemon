//! The memory tables: created on a new database, brought up to date on an
//! existing one. Every step is idempotent and runs on each start, and none
//! removes a user's data, so upgrading the host keeps what it remembers.

use sqlx::{Row, SqlitePool};

/// Create or upgrade the memory tables in `pool`.
pub async fn migrate(pool: &SqlitePool) -> anyhow::Result<()> {
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
    if let Err(e) = migrate_facts_history_schema(pool).await {
        tracing::warn!("Failed to migrate facts schema for history: {}", e);
    }

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

    // Migration: allow multiple episodes per session (for mid-session episode
    // creation in long-running conversations). The non-unique index on session_id
    // (idx_episodes_session) already exists for lookups.
    let _ = sqlx::query("DROP INDEX IF EXISTS idx_episodes_session_unique")
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

    // A change counter for `memory_embeddings`, so the in-memory vector
    // cache in `SqliteEmbeddingIndex` knows when to reload. Triggers keep it
    // exact whichever code (or process) writes the table.
    for statement in [
        "CREATE TABLE IF NOT EXISTS memory_embedding_generation (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            generation INTEGER NOT NULL
        )",
        "INSERT OR IGNORE INTO memory_embedding_generation (id, generation) VALUES (1, 0)",
        "CREATE TRIGGER IF NOT EXISTS memory_embeddings_generation_insert
         AFTER INSERT ON memory_embeddings BEGIN
            UPDATE memory_embedding_generation SET generation = generation + 1 WHERE id = 1;
         END",
        "CREATE TRIGGER IF NOT EXISTS memory_embeddings_generation_update
         AFTER UPDATE ON memory_embeddings BEGIN
            UPDATE memory_embedding_generation SET generation = generation + 1 WHERE id = 1;
         END",
        "CREATE TRIGGER IF NOT EXISTS memory_embeddings_generation_delete
         AFTER DELETE ON memory_embeddings BEGIN
            UPDATE memory_embedding_generation SET generation = generation + 1 WHERE id = 1;
         END",
    ] {
        sqlx::query(statement).execute(pool).await?;
    }

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

    repair_blank_validity_bounds(pool).await?;

    // FTS5 is present in standard SQLite and bundled SQLCipher builds. Keep the
    // daemon usable on custom SQLite builds without it; semantic and lexical
    // fallback retrieval remain available.
    if let Err(error) = create_memory_fts(pool).await {
        tracing::warn!(%error, "SQLite FTS5 unavailable; memory full-text index disabled");
    }

    Ok(())
}

/// Canonical rows written with a blank validity bound (`""` from extraction,
/// meaning "no bound") were invisible to every `valid_to IS NULL` read: the
/// prompt never showed them and the next identical write inserted a
/// duplicate. Ends the later duplicate of each such row, then stores the
/// remaining blank bounds as NULL. Idempotent: a clean database matches
/// nothing.
async fn repair_blank_validity_bounds(pool: &SqlitePool) -> anyhow::Result<()> {
    // A twin that already has a NULL bound wins whatever its id; between two
    // blank twins the older one does.
    let twin = |alias: &str, row: &str| {
        format!(
            "({alias}.valid_to IS NULL OR (TRIM({alias}.valid_to) = '' AND {alias}.id < {row}.id))"
        )
    };
    let statements = [
        format!(
            "UPDATE memory_entity_facts AS f SET status = 'superseded', valid_to = f.updated_at
             WHERE f.status = 'active' AND TRIM(f.valid_to) = ''
               AND EXISTS (SELECT 1 FROM memory_entity_facts o
                           WHERE o.subject_entity_id = f.subject_entity_id
                             AND o.predicate = f.predicate
                             AND o.normalized_value = f.normalized_value
                             AND o.status = 'active' AND o.id <> f.id AND {})",
            twin("o", "f")
        ),
        format!(
            "UPDATE memory_relationships AS r SET status = 'superseded', valid_to = r.updated_at
             WHERE r.status = 'active' AND TRIM(r.valid_to) = ''
               AND EXISTS (SELECT 1 FROM memory_relationships o
                           WHERE o.source_entity_id = r.source_entity_id
                             AND o.relationship_type = r.relationship_type
                             AND o.target_entity_id = r.target_entity_id
                             AND o.status = 'active' AND o.id <> r.id AND {})",
            twin("o", "r")
        ),
    ];
    for statement in &statements {
        sqlx::query(sqlx::AssertSqlSafe(statement.as_str()))
            .execute(pool)
            .await?;
    }
    for table in [
        "memory_entity_facts",
        "memory_relationships",
        "memory_aliases",
    ] {
        for column in ["valid_from", "valid_to"] {
            sqlx::query(sqlx::AssertSqlSafe(format!(
                "UPDATE {table} SET {column} = NULL WHERE TRIM({column}) = ''"
            )))
            .execute(pool)
            .await?;
        }
    }
    Ok(())
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
        let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;

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
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "INSERT INTO {table}({table}, rank) VALUES('secure-delete', 0)"
        )))
        .execute(pool)
        .await?;
    }

    // `UPDATE OF column` fires even when an UPSERT assigns the existing value.
    // Repeated delete/reinsert churn of an unchanged row can corrupt FTS5
    // secure-delete segments on SQLCipher's SQLite build. Replace legacy
    // update triggers and suppress true no-op text updates.
    for trigger in ["memory_claims_au", "memory_spans_au"] {
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "DROP TRIGGER IF EXISTS {trigger}"
        )))
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
pub async fn rebuild_memory_fts_projections(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query("DROP TABLE IF EXISTS memory_claims_fts")
        .execute(pool)
        .await?;
    sqlx::query("DROP TABLE IF EXISTS memory_spans_fts")
        .execute(pool)
        .await?;
    create_memory_fts(pool).await
}

#[cfg(test)]
mod tests {
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
            let secure_delete: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
                "SELECT v FROM {table} WHERE k = 'secure-delete'"
            )))
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

    async fn database() -> (tempfile::NamedTempFile, SqlitePool) {
        let file = tempfile::NamedTempFile::new().unwrap();
        let pool = SqlitePoolOptions::new()
            .max_connections(2)
            .connect_with(SqliteConnectOptions::new().filename(file.path()))
            .await
            .unwrap();
        (file, pool)
    }

    async fn count(pool: &SqlitePool, sql: &str) -> i64 {
        sqlx::query_scalar(sqlx::AssertSqlSafe(sql))
            .fetch_one(pool)
            .await
            .unwrap()
    }

    /// A database from before channels, privacy, supersession history and
    /// the claim graph existed, as a long-running install still has.
    async fn old_install(pool: &SqlitePool) {
        for statement in [
            "CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(category, key)
            )",
            "INSERT INTO facts (category, key, value, source, created_at, updated_at) VALUES
                ('personal', 'favorite_color', 'teal', 'user', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z'),
                ('project', 'acme_stack', 'Rust', 'agent', '2025-01-02T00:00:00Z', '2025-01-02T00:00:00Z')",
            "CREATE TABLE people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
            "INSERT INTO people (name) VALUES ('Alice Rivera'), ('Jordan Lee')",
        ] {
            sqlx::query(statement).execute(pool).await.unwrap();
        }
    }

    #[tokio::test]
    async fn upgrading_an_old_install_keeps_what_it_remembers() {
        let (_file, pool) = database().await;
        old_install(&pool).await;

        migrate(&pool).await.unwrap();

        assert_eq!(count(&pool, "SELECT COUNT(*) FROM facts").await, 2);
        assert_eq!(count(&pool, "SELECT COUNT(*) FROM people").await, 2);
        let value: String =
            sqlx::query_scalar("SELECT value FROM facts WHERE key = 'favorite_color'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(value, "teal");
        assert_eq!(
            count(&pool, "SELECT COUNT(*) FROM facts WHERE privacy = 'global'").await,
            2,
            "existing facts stay visible as they were"
        );
        // The old one-row-per-key constraint is gone: a key can keep history.
        sqlx::query(
            "UPDATE facts SET superseded_at = '2026-01-01T00:00:00Z' WHERE key = 'favorite_color'",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO facts (category, key, value, source, created_at, updated_at)
             VALUES ('personal', 'favorite_color', 'navy', 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn every_start_leaves_existing_memory_alone() {
        let (_file, pool) = database().await;
        migrate(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO facts (category, key, value, source, created_at, updated_at)
             VALUES ('personal', 'favorite_color', 'teal', 'agent', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        migrate(&pool).await.unwrap();
        migrate(&pool).await.unwrap();

        let (value, source): (String, String) =
            sqlx::query_as("SELECT value, source FROM facts WHERE key = 'favorite_color'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!((value.as_str(), source.as_str()), ("teal", "agent"));
    }
}
