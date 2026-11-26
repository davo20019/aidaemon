//! The terminal's standing approvals, per execution backend: the command
//! prefixes the owner allowed for good and the capabilities allowed for
//! them. This module is their only owner: the schema, its one-time
//! conversions and every read and write live here, so the state migrations
//! and the terminal tool cannot disagree about them.
//!
//! Trust-all (every command allowed) is not an approval: it is AIdaemon's
//! permission level, Full access, kept in `config.toml`
//! ([`crate::permissions`]). Older databases kept it here, as a `*` prefix
//! and then as a `terminal_trust_all` row; [`ensure_schema`] gathers both
//! into that table and [`legacy_trust_all`] hands it to the daemon's
//! start, which drops it once it is in the level, so nothing copies an old shape again
//! (2026-11-21: the old table was copied back on every start, so
//! `/trustall off` lasted until the next restart).

use std::collections::HashSet;

use sqlx::SqlitePool;

/// The prefix older releases stored to mean "every command".
const LEGACY_TRUST_ALL_PREFIX: &str = "*";

/// Where releases before the level kept trust-all, one row per backend.
const LEGACY_TRUST_ALL_TABLE: &str = "CREATE TABLE IF NOT EXISTS terminal_trust_all (
    backend_scope TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)";

/// One backend's standing approvals.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TerminalGrants {
    /// Command prefixes allowed for good.
    pub prefixes: Vec<String>,
    /// `(prefix, capability)` pairs allowed for good.
    pub capabilities: HashSet<(String, String)>,
}

/// Whether an older release left trust-all on for any backend. The
/// daemon's start turns it into the Full access level
/// ([`crate::permissions`]), then [`drop_legacy_trust_all`].
pub async fn legacy_trust_all(pool: &SqlitePool) -> anyhow::Result<bool> {
    let table: Option<String> = sqlx::query_scalar(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'terminal_trust_all'",
    )
    .fetch_optional(pool)
    .await?;
    if table.is_none() {
        return Ok(false);
    }
    let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM terminal_trust_all")
        .fetch_one(pool)
        .await?;
    Ok(rows > 0)
}

/// Remove where older releases kept trust-all, once it is in the level.
pub async fn drop_legacy_trust_all(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query("DROP TABLE IF EXISTS terminal_trust_all")
        .execute(pool)
        .await?;
    Ok(())
}

/// Create the tables and convert older shapes, once. Safe to call on every
/// start and from every owner of a pool: a conversion runs only while its old
/// shape is there, and removes it.
pub async fn ensure_schema(pool: &SqlitePool) -> anyhow::Result<()> {
    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
    for statement in [
        "CREATE TABLE IF NOT EXISTS terminal_backend_allowed_prefixes (
            backend_scope TEXT NOT NULL,
            prefix TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (backend_scope, prefix)
        )",
        "CREATE TABLE IF NOT EXISTS terminal_capability_grants (
            backend_scope TEXT NOT NULL,
            prefix TEXT NOT NULL,
            capability TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (backend_scope, prefix, capability)
        )",
    ] {
        sqlx::query(statement).execute(&mut *tx).await?;
    }
    let legacy: Option<String> = sqlx::query_scalar(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'terminal_allowed_prefixes'",
    )
    .fetch_optional(&mut *tx)
    .await?;
    if legacy.is_some() {
        // Approvals from before backends were scoped held for local
        // execution only; Docker and SSH targets never inherit them.
        sqlx::query(
            "INSERT OR IGNORE INTO terminal_backend_allowed_prefixes
                (backend_scope, prefix, created_at)
             SELECT 'local', prefix, created_at FROM terminal_allowed_prefixes",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query("DROP TABLE terminal_allowed_prefixes")
            .execute(&mut *tx)
            .await?;
    }
    let star: Option<String> = sqlx::query_scalar(
        "SELECT backend_scope FROM terminal_backend_allowed_prefixes WHERE prefix = ? LIMIT 1",
    )
    .bind(LEGACY_TRUST_ALL_PREFIX)
    .fetch_optional(&mut *tx)
    .await?;
    if star.is_some() {
        sqlx::query(LEGACY_TRUST_ALL_TABLE)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "INSERT OR IGNORE INTO terminal_trust_all (backend_scope, created_at)
             SELECT backend_scope, created_at FROM terminal_backend_allowed_prefixes
             WHERE prefix = ?",
        )
        .bind(LEGACY_TRUST_ALL_PREFIX)
        .execute(&mut *tx)
        .await?;
    }
    sqlx::query("DELETE FROM terminal_backend_allowed_prefixes WHERE prefix = ?")
        .bind(LEGACY_TRUST_ALL_PREFIX)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}

/// `scope`'s standing approvals.
pub async fn load(pool: &SqlitePool, scope: &str) -> anyhow::Result<TerminalGrants> {
    let prefixes = sqlx::query_scalar(
        "SELECT prefix FROM terminal_backend_allowed_prefixes
         WHERE backend_scope = ? ORDER BY created_at, prefix",
    )
    .bind(scope)
    .fetch_all(pool)
    .await?;
    let capabilities = sqlx::query_as::<_, (String, String)>(
        "SELECT prefix, capability FROM terminal_capability_grants WHERE backend_scope = ?",
    )
    .bind(scope)
    .fetch_all(pool)
    .await?
    .into_iter()
    .collect();
    Ok(TerminalGrants {
        prefixes,
        capabilities,
    })
}

/// Allow `prefix` for good. `*` is never a prefix: trust-all is the Full
/// access level.
pub async fn add_prefix(pool: &SqlitePool, scope: &str, prefix: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        prefix != LEGACY_TRUST_ALL_PREFIX,
        "trust-all is not a prefix"
    );
    sqlx::query(
        "INSERT OR IGNORE INTO terminal_backend_allowed_prefixes (backend_scope, prefix)
         VALUES (?, ?)",
    )
    .bind(scope)
    .bind(prefix)
    .execute(pool)
    .await?;
    Ok(())
}

/// Take back a prefix allowed for good, with the capabilities allowed for
/// it. `false` when it was not there.
pub async fn remove_prefix(pool: &SqlitePool, scope: &str, prefix: &str) -> anyhow::Result<bool> {
    let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
    let removed = sqlx::query(
        "DELETE FROM terminal_backend_allowed_prefixes WHERE backend_scope = ? AND prefix = ?",
    )
    .bind(scope)
    .bind(prefix)
    .execute(&mut *tx)
    .await?
    .rows_affected()
        > 0;
    sqlx::query("DELETE FROM terminal_capability_grants WHERE backend_scope = ? AND prefix = ?")
        .bind(scope)
        .bind(prefix)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(removed)
}

/// Allow `capability` for good for commands under `prefix`.
pub async fn add_capability(
    pool: &SqlitePool,
    scope: &str,
    prefix: &str,
    capability: &str,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT OR IGNORE INTO terminal_capability_grants (backend_scope, prefix, capability)
         VALUES (?, ?, ?)",
    )
    .bind(scope)
    .bind(prefix)
    .bind(capability)
    .execute(pool)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One connection: each `:memory:` connection is its own database.
    async fn pool() -> SqlitePool {
        sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap()
    }

    /// A database from before backends were scoped, with trust-all stored as
    /// a `*` prefix in the old table: converted once, nothing lost, trust-all
    /// handed to the start once, and what the owner then removes stays
    /// removed across starts.
    #[tokio::test]
    async fn old_approvals_convert_once_and_removals_stay_removed() {
        let pool = pool().await;
        for statement in [
            "CREATE TABLE terminal_allowed_prefixes (
                prefix TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
            "INSERT INTO terminal_allowed_prefixes (prefix, created_at) VALUES
                ('*', '2026-07-30 17:09:42'), ('git', '2026-07-30 17:09:43')",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        ensure_schema(&pool).await.unwrap();
        assert_eq!(load(&pool, "local").await.unwrap().prefixes, ["git"]);
        let legacy: Option<String> = sqlx::query_scalar(
            "SELECT name FROM sqlite_master WHERE name = 'terminal_allowed_prefixes'",
        )
        .fetch_optional(&pool)
        .await
        .unwrap();
        assert!(legacy.is_none(), "the old table is gone");
        assert!(legacy_trust_all(&pool).await.unwrap());
        drop_legacy_trust_all(&pool).await.unwrap();
        assert!(!legacy_trust_all(&pool).await.unwrap(), "handed over once");

        assert!(remove_prefix(&pool, "local", "git").await.unwrap());
        // The next start.
        ensure_schema(&pool).await.unwrap();
        assert_eq!(
            load(&pool, "local").await.unwrap(),
            TerminalGrants::default()
        );
        assert!(!legacy_trust_all(&pool).await.unwrap());
    }

    /// Trust-all from a release that stored it as a `*` prefix in the scoped
    /// table, or as its own row, is handed to the start; `*` is never saved
    /// as a prefix again.
    #[tokio::test]
    async fn old_trust_all_is_handed_over_and_star_is_never_a_prefix() {
        let pool = pool().await;
        ensure_schema(&pool).await.unwrap();
        assert!(
            !legacy_trust_all(&pool).await.unwrap(),
            "a new database has none"
        );
        sqlx::query(
            "INSERT INTO terminal_backend_allowed_prefixes (backend_scope, prefix)
             VALUES ('ssh:synthetic', '*')",
        )
        .execute(&pool)
        .await
        .unwrap();
        ensure_schema(&pool).await.unwrap();
        assert!(
            load(&pool, "ssh:synthetic")
                .await
                .unwrap()
                .prefixes
                .is_empty()
        );
        assert!(legacy_trust_all(&pool).await.unwrap());
        drop_legacy_trust_all(&pool).await.unwrap();

        sqlx::query(LEGACY_TRUST_ALL_TABLE)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO terminal_trust_all (backend_scope) VALUES ('local')")
            .execute(&pool)
            .await
            .unwrap();
        assert!(legacy_trust_all(&pool).await.unwrap());
        assert!(add_prefix(&pool, "local", "*").await.is_err());
    }

    /// A prefix taken back takes its capabilities with it.
    #[tokio::test]
    async fn removing_a_prefix_removes_its_capabilities() {
        let pool = pool().await;
        ensure_schema(&pool).await.unwrap();
        add_prefix(&pool, "local", "curl").await.unwrap();
        add_capability(&pool, "local", "curl", "network")
            .await
            .unwrap();
        assert_eq!(load(&pool, "local").await.unwrap().capabilities.len(), 1);
        assert!(remove_prefix(&pool, "local", "curl").await.unwrap());
        assert!(!remove_prefix(&pool, "local", "curl").await.unwrap());
        assert_eq!(
            load(&pool, "local").await.unwrap(),
            TerminalGrants::default()
        );
    }
}
