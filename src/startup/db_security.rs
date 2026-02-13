use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use chrono::Utc;
use rand::RngCore;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use tracing::{info, warn};

use crate::config::AppConfig;

const DB_ENCRYPTION_ENV_KEY: &str = "AIDAEMON_ENCRYPTION_KEY";
const DB_ALLOW_PLAINTEXT_ENV_KEY: &str = "AIDAEMON_ALLOW_PLAINTEXT_DB";
const DB_ENV_FILE_ENV_KEY: &str = "AIDAEMON_ENV_FILE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DbMode {
    Missing,
    Plaintext,
    EncryptedOrUnknown,
}

/// Enforce encrypted-at-rest database operation by default.
///
/// Behavior:
/// - Generates an encryption key into `.env` when missing (new/plaintext DBs).
/// - Migrates existing plaintext SQLite DBs to SQLCipher automatically.
/// - Verifies encrypted DB accessibility with the resolved key.
/// - Supports emergency bypass via `AIDAEMON_ALLOW_PLAINTEXT_DB=1|true`.
pub async fn enforce_database_encryption(
    config: &mut AppConfig,
    config_path: &Path,
) -> anyhow::Result<()> {
    if truthy_env(DB_ALLOW_PLAINTEXT_ENV_KEY) {
        warn!(
            "{} is set; running without mandatory database encryption",
            DB_ALLOW_PLAINTEXT_ENV_KEY
        );
        return Ok(());
    }

    #[cfg(not(feature = "encryption"))]
    {
        anyhow::bail!(
            "Database encryption is required by default, but aidaemon was built without the \
             'encryption' feature. Rebuild without --no-default-features or set \
             {}=1 temporarily.",
            DB_ALLOW_PLAINTEXT_ENV_KEY
        );
    }

    #[cfg(feature = "encryption")]
    {
        enforce_database_encryption_impl(config, config_path).await
    }
}

#[cfg(feature = "encryption")]
async fn enforce_database_encryption_impl(
    config: &mut AppConfig,
    config_path: &Path,
) -> anyhow::Result<()> {
    let db_path = resolve_db_path(config_path, &config.state.db_path);
    let mode = detect_db_mode(&db_path)?;
    let encryption_key = resolve_or_create_encryption_key(config, config_path, mode)?;
    config.state.encryption_key = Some(encryption_key.clone());

    match mode {
        DbMode::Missing => {
            info!(
                db_path = %db_path.display(),
                "No existing database found; encrypted DB will be created on first write"
            );
        }
        DbMode::Plaintext => {
            migrate_plaintext_db_to_encrypted(&db_path, &encryption_key).await?;
        }
        DbMode::EncryptedOrUnknown => {
            verify_encrypted_database(&db_path, &encryption_key, None).await?;
            info!(db_path = %db_path.display(), "Encrypted database verified");
        }
    }

    Ok(())
}

#[cfg(feature = "encryption")]
fn resolve_or_create_encryption_key(
    config: &AppConfig,
    config_path: &Path,
    db_mode: DbMode,
) -> anyhow::Result<String> {
    if let Some(ref key) = config.state.encryption_key {
        if !key.trim().is_empty() {
            return Ok(key.trim().to_string());
        }
    }

    if let Ok(key) = std::env::var(DB_ENCRYPTION_ENV_KEY) {
        if !key.trim().is_empty() {
            return Ok(key.trim().to_string());
        }
    }

    if db_mode == DbMode::EncryptedOrUnknown {
        anyhow::bail!(
            "Database appears encrypted but no {} value was found in config, keychain, or environment. \
             Restore your encryption key before restarting.",
            DB_ENCRYPTION_ENV_KEY
        );
    }

    let key = generate_encryption_key_hex();
    let env_path = resolve_env_file_path(config_path);
    persist_key_to_env_file(&env_path, &key)?;
    std::env::set_var(DB_ENCRYPTION_ENV_KEY, &key);
    info!(
        env_path = %env_path.display(),
        "Generated a new database encryption key and stored it in .env"
    );
    Ok(key)
}

#[cfg(feature = "encryption")]
async fn migrate_plaintext_db_to_encrypted(db_path: &Path, key: &str) -> anyhow::Result<()> {
    let db_path_str = db_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Database path contains invalid UTF-8"))?;

    info!(db_path = %db_path.display(), "Migrating plaintext SQLite database to SQLCipher");

    let plain_pool = connect_sqlite_with_create(db_path_str, true).await?;
    let _ = sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
        .execute(&plain_pool)
        .await;
    let source_counts = collect_table_counts(&plain_pool).await?;

    let temp_path = encrypted_temp_path(db_path);
    if temp_path.exists() {
        std::fs::remove_file(&temp_path)?;
    }
    plain_pool.close().await;

    let attach_sql = format!(
        "ATTACH DATABASE '{}' AS encrypted KEY '{}'",
        temp_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Temp path contains invalid UTF-8"))?
            .replace('\'', "''"),
        key.replace('\'', "''")
    );
    let plain_pool = connect_sqlite_with_create(db_path_str, true).await?;
    sqlx::query(&attach_sql).execute(&plain_pool).await?;
    sqlx::query("SELECT sqlcipher_export('encrypted')")
        .execute(&plain_pool)
        .await?;
    sqlx::query("DETACH DATABASE encrypted")
        .execute(&plain_pool)
        .await?;
    plain_pool.close().await;
    set_file_mode_0600(&temp_path)?;

    if detect_db_mode(&temp_path)? != DbMode::EncryptedOrUnknown {
        anyhow::bail!(
            "Database encryption step completed, but output file still looks like plaintext SQLite"
        );
    }

    verify_encrypted_database(&temp_path, key, Some(&source_counts)).await?;

    let backup_path = backup_path(db_path);
    std::fs::copy(db_path, &backup_path)?;
    set_file_mode_0600(&backup_path)?;

    replace_database_file(&temp_path, db_path)?;
    remove_if_exists(wal_path(db_path));
    remove_if_exists(shm_path(db_path));

    info!(
        db_path = %db_path.display(),
        backup_path = %backup_path.display(),
        "Database migration completed successfully"
    );
    Ok(())
}

#[cfg(feature = "encryption")]
fn replace_database_file(temp_path: &Path, db_path: &Path) -> anyhow::Result<()> {
    match std::fs::rename(temp_path, db_path) {
        Ok(()) => Ok(()),
        Err(rename_err) => {
            // Windows does not always allow replacing an existing file via rename.
            warn!(
                error = %rename_err,
                "Atomic replace failed; falling back to rollback-safe swap"
            );
            let rollback_path = PathBuf::from(format!("{}.swap.old", db_path.display()));
            if rollback_path.exists() {
                let _ = std::fs::remove_file(&rollback_path);
            }

            if db_path.exists() {
                std::fs::rename(db_path, &rollback_path)?;
            }

            match std::fs::rename(temp_path, db_path) {
                Ok(()) => {
                    remove_if_exists(rollback_path);
                    Ok(())
                }
                Err(swap_err) => {
                    if rollback_path.exists() {
                        let _ = std::fs::rename(&rollback_path, db_path);
                    }
                    Err(swap_err.into())
                }
            }
        }
    }
}

#[cfg(feature = "encryption")]
async fn verify_encrypted_database(
    db_path: &Path,
    key: &str,
    expected_counts: Option<&HashMap<String, i64>>,
) -> anyhow::Result<()> {
    let db_path_str = db_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Database path contains invalid UTF-8"))?;
    let pool = connect_sqlite(db_path_str).await?;
    apply_encryption_key(&pool, key).await?;

    let (check,): (String,) = sqlx::query_as("PRAGMA integrity_check")
        .fetch_one(&pool)
        .await
        .map_err(|e| anyhow::anyhow!("Encrypted database integrity check failed: {}", e))?;
    if !check.eq_ignore_ascii_case("ok") {
        anyhow::bail!("Encrypted database integrity check returned '{}'", check);
    }

    if let Some(expected) = expected_counts {
        let actual = collect_table_counts(&pool).await?;
        compare_table_counts(expected, &actual)?;
    }

    pool.close().await;
    Ok(())
}

#[cfg(feature = "encryption")]
fn compare_table_counts(
    expected: &HashMap<String, i64>,
    actual: &HashMap<String, i64>,
) -> anyhow::Result<()> {
    for (table, expected_count) in expected {
        match actual.get(table) {
            Some(actual_count) if actual_count == expected_count => {}
            Some(actual_count) => {
                anyhow::bail!(
                    "Row count mismatch after migration for table '{}': expected {}, got {}",
                    table,
                    expected_count,
                    actual_count
                );
            }
            None => {
                anyhow::bail!(
                    "Table '{}' missing after migration; refusing to replace original database",
                    table
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "encryption")]
async fn connect_sqlite(db_path: &str) -> anyhow::Result<SqlitePool> {
    connect_sqlite_with_create(db_path, false).await
}

#[cfg(feature = "encryption")]
async fn connect_sqlite_with_create(
    db_path: &str,
    create_if_missing: bool,
) -> anyhow::Result<SqlitePool> {
    let opts = SqliteConnectOptions::new()
        .filename(db_path)
        .create_if_missing(create_if_missing);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await?;
    Ok(pool)
}

#[cfg(feature = "encryption")]
async fn apply_encryption_key(pool: &SqlitePool, key: &str) -> anyhow::Result<()> {
    let escaped_key = key.replace('\'', "''");
    sqlx::query(&format!("PRAGMA key = '{}'", escaped_key))
        .execute(pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to apply SQLCipher key: {}", e))?;
    Ok(())
}

#[cfg(feature = "encryption")]
async fn collect_table_counts(pool: &SqlitePool) -> anyhow::Result<HashMap<String, i64>> {
    let rows = sqlx::query(
        "SELECT name FROM sqlite_master \
         WHERE type = 'table' AND name NOT LIKE 'sqlite_%' \
         ORDER BY name",
    )
    .fetch_all(pool)
    .await?;

    let mut counts = HashMap::new();
    for row in rows {
        let table: String = row.get("name");
        let sql = format!(
            "SELECT COUNT(*) AS c FROM \"{}\"",
            table.replace('"', "\"\"")
        );
        let (count,): (i64,) = sqlx::query_as(&sql).fetch_one(pool).await?;
        counts.insert(table, count);
    }
    Ok(counts)
}

fn resolve_db_path(config_path: &Path, db_path: &str) -> PathBuf {
    let p = PathBuf::from(db_path);
    if p.is_absolute() {
        p
    } else {
        config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(p)
    }
}

fn detect_db_mode(db_path: &Path) -> anyhow::Result<DbMode> {
    if !db_path.exists() {
        return Ok(DbMode::Missing);
    }
    let metadata = std::fs::metadata(db_path)?;
    if metadata.len() == 0 {
        return Ok(DbMode::Missing);
    }

    let mut file = File::open(db_path)?;
    let mut header = [0_u8; 16];
    let bytes_read = file.read(&mut header)?;
    if bytes_read < 16 {
        return Ok(DbMode::EncryptedOrUnknown);
    }

    if &header == b"SQLite format 3\0" {
        Ok(DbMode::Plaintext)
    } else {
        Ok(DbMode::EncryptedOrUnknown)
    }
}

#[cfg(feature = "encryption")]
fn resolve_env_file_path(config_path: &Path) -> PathBuf {
    if let Ok(path) = std::env::var(DB_ENV_FILE_ENV_KEY) {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".env")
}

#[cfg(feature = "encryption")]
fn persist_key_to_env_file(env_path: &Path, key: &str) -> anyhow::Result<()> {
    let mut lines: Vec<String> = if env_path.exists() {
        std::fs::read_to_string(env_path)?
            .lines()
            .map(|l| l.to_string())
            .collect()
    } else {
        vec![]
    };

    let key_prefix = format!("{}=", DB_ENCRYPTION_ENV_KEY);
    let mut replaced = false;
    for line in &mut lines {
        if line.trim_start().starts_with(&key_prefix) {
            *line = format!("{}{}", key_prefix, key);
            replaced = true;
        }
    }
    if !replaced {
        lines.push(format!("{}{}", key_prefix, key));
    }

    if let Some(parent) = env_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut body = lines.join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    std::fs::write(env_path, body)?;
    set_file_mode_0600(env_path)?;
    Ok(())
}

#[cfg(feature = "encryption")]
fn generate_encryption_key_hex() -> String {
    let mut key_bytes = [0_u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut key_bytes);
    bytes_to_hex(&key_bytes)
}

#[cfg(feature = "encryption")]
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

fn truthy_env(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[cfg(feature = "encryption")]
fn backup_path(db_path: &Path) -> PathBuf {
    PathBuf::from(format!(
        "{}.bak.{}",
        db_path.display(),
        Utc::now().format("%Y%m%d%H%M%S")
    ))
}

#[cfg(feature = "encryption")]
fn encrypted_temp_path(db_path: &Path) -> PathBuf {
    PathBuf::from(format!(
        "{}.encrypted.tmp.{}",
        db_path.display(),
        Utc::now().timestamp_millis()
    ))
}

#[cfg(feature = "encryption")]
fn wal_path(db_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}-wal", db_path.display()))
}

#[cfg(feature = "encryption")]
fn shm_path(db_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}-shm", db_path.display()))
}

#[cfg(feature = "encryption")]
fn remove_if_exists(path: PathBuf) {
    if path.exists() {
        if let Err(e) = std::fs::remove_file(&path) {
            warn!(path = %path.display(), error = %e, "Failed to remove stale SQLite sidecar file");
        }
    }
}

fn set_file_mode_0600(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqliteConnectOptions;

    #[test]
    fn detect_db_mode_plaintext_header() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("plain.db");
        std::fs::write(&db_path, b"SQLite format 3\0test").unwrap();
        let mode = detect_db_mode(&db_path).unwrap();
        assert_eq!(mode, DbMode::Plaintext);
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn persist_key_to_env_file_updates_existing_key() {
        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join(".env");
        std::fs::write(
            &env_path,
            "AIDAEMON_API_KEY=test\nAIDAEMON_ENCRYPTION_KEY=old\n",
        )
        .unwrap();

        persist_key_to_env_file(&env_path, "newkey").unwrap();
        let content = std::fs::read_to_string(&env_path).unwrap();
        assert!(content.contains("AIDAEMON_API_KEY=test"));
        assert!(content.contains("AIDAEMON_ENCRYPTION_KEY=newkey"));
        assert!(!content.contains("AIDAEMON_ENCRYPTION_KEY=old"));
    }

    #[cfg(feature = "encryption")]
    #[tokio::test]
    async fn migrate_plaintext_db_to_encrypted_preserves_rows() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("state.db");
        let db_path_str = db_path.to_str().unwrap().to_string();

        let plain_opts = SqliteConnectOptions::new()
            .filename(&db_path_str)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);
        let plain_pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(plain_opts)
            .await
            .unwrap();

        sqlx::query("CREATE TABLE IF NOT EXISTS migration_test (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            .execute(&plain_pool)
            .await
            .unwrap();
        sqlx::query("INSERT INTO migration_test(value) VALUES ('alpha'), ('beta')")
            .execute(&plain_pool)
            .await
            .unwrap();
        plain_pool.close().await;

        let key = "unit-test-migration-key";
        migrate_plaintext_db_to_encrypted(&db_path, key)
            .await
            .unwrap();

        let mode = detect_db_mode(&db_path).unwrap();
        assert_eq!(mode, DbMode::EncryptedOrUnknown);

        verify_encrypted_database(&db_path, key, None)
            .await
            .unwrap();

        let encrypted_pool = connect_sqlite(db_path.to_str().unwrap()).await.unwrap();
        apply_encryption_key(&encrypted_pool, key).await.unwrap();
        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM migration_test")
            .fetch_one(&encrypted_pool)
            .await
            .unwrap();
        encrypted_pool.close().await;

        assert_eq!(count, 2);
    }
}
