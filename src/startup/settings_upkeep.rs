//! Settings changes that take effect across a restart, scheduled from the
//! owner's apps: moving the database file, rotating its encryption key, and
//! trying a new Client Gateway address that rolls back unless an app
//! reconnects. Each is a small JSON marker beside the config file, acted on
//! at the next start; what happened is left in a result file the Settings
//! page shows.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

/// How long a new gateway address waits for an app before rolling back.
pub(crate) const GATEWAY_TRIAL: Duration = Duration::from_secs(120);

/// Database upkeep scheduled for the next start.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub(crate) enum DbUpkeep {
    MoveTo { path: String },
    RotateKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpkeepResult {
    at_ms: u64,
    ok: bool,
    message: String,
}

fn dir_of(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn upkeep_marker(config_path: &Path) -> PathBuf {
    dir_of(config_path).join("db-upkeep.json")
}

fn upkeep_result(config_path: &Path) -> PathBuf {
    dir_of(config_path).join("db-upkeep-result.json")
}

fn trial_marker(config_path: &Path) -> PathBuf {
    dir_of(config_path).join("gateway-trial.json")
}

/// The config as it was before a gateway address change, put back when the
/// trial fails.
fn trial_previous(config_path: &Path) -> PathBuf {
    config_path.with_extension("toml.gateway-previous")
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn write_owner_only(path: &Path, text: &str) -> std::io::Result<()> {
    std::fs::write(path, text)?;
    aidaemon_tools::config_file::set_owner_only_permissions(path);
    Ok(())
}

/// Schedule `upkeep` for the next start (replacing what was scheduled).
pub(crate) fn schedule(config_path: &Path, upkeep: &DbUpkeep) -> anyhow::Result<()> {
    write_owner_only(&upkeep_marker(config_path), &serde_json::to_string(upkeep)?)?;
    Ok(())
}

/// Drop what is scheduled; `false` when nothing was.
pub(crate) fn cancel(config_path: &Path) -> bool {
    std::fs::remove_file(upkeep_marker(config_path)).is_ok()
}

pub(crate) fn scheduled(config_path: &Path) -> Option<DbUpkeep> {
    std::fs::read_to_string(upkeep_marker(config_path))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

/// What the Settings page says about upkeep and the gateway's address.
pub(crate) fn notices(config_path: &Path) -> Vec<String> {
    let mut notices = Vec::new();
    match scheduled(config_path) {
        Some(DbUpkeep::MoveTo { path }) => {
            notices.push(format!("The database moves to {path} at the next restart."))
        }
        Some(DbUpkeep::RotateKey) => {
            notices.push("The database key is rotated at the next restart.".to_string())
        }
        None => {}
    }
    if let Some(result) = std::fs::read_to_string(upkeep_result(config_path))
        .ok()
        .and_then(|text| serde_json::from_str::<UpkeepResult>(&text).ok())
        && now_ms().saturating_sub(result.at_ms) < 7 * 24 * 3600 * 1000
    {
        notices.push(if result.ok {
            result.message
        } else {
            format!("Database upkeep failed: {}", result.message)
        });
    }
    if trial_marker(config_path).exists() {
        notices.push(format!(
            "The app address changed. If no app reconnects within {} seconds of the next \
             start, the previous address comes back.",
            GATEWAY_TRIAL.as_secs()
        ));
    }
    notices
}

fn record(config_path: &Path, ok: bool, message: String) {
    if ok {
        info!("{message}");
    } else {
        error!("Database upkeep failed: {message}");
    }
    let result = UpkeepResult {
        at_ms: now_ms(),
        ok,
        message,
    };
    if let Ok(text) = serde_json::to_string(&result)
        && let Err(e) = write_owner_only(&upkeep_result(config_path), &text)
    {
        warn!(error = %e, "Could not record the database upkeep result");
    }
}

/// Run the upkeep scheduled for this start, before the database opens.
/// A failure leaves the database where and as it was and is recorded; it
/// never stops the daemon from starting.
pub(crate) async fn run_scheduled(config: &mut crate::config::AppConfig, config_path: &Path) {
    let Some(upkeep) = scheduled(config_path) else {
        return;
    };
    let _ = std::fs::remove_file(upkeep_marker(config_path));
    let outcome = match &upkeep {
        DbUpkeep::MoveTo { path } => move_database(config, config_path, path).await,
        DbUpkeep::RotateKey => rotate_key(config, config_path).await,
    };
    match outcome {
        Ok(message) => record(config_path, true, message),
        Err(e) => record(config_path, false, format!("{e:#}")),
    }
}

fn sidecars(db: &Path) -> [PathBuf; 2] {
    [
        PathBuf::from(format!("{}-wal", db.display())),
        PathBuf::from(format!("{}-shm", db.display())),
    ]
}

/// Copy the database (with its WAL) to `to`, check the copy opens with the
/// key, point `state.db_path` at it and keep the old file as a backup.
async fn move_database(
    config: &mut crate::config::AppConfig,
    config_path: &Path,
    to: &str,
) -> anyhow::Result<String> {
    let from = crate::startup::db_security::resolve_db_path(config_path, &config.state.db_path);
    let target = crate::startup::db_security::resolve_db_path(config_path, to);
    anyhow::ensure!(from.exists(), "{} does not exist", from.display());
    anyhow::ensure!(
        !target.exists(),
        "{} already exists; nothing was moved",
        target.display()
    );
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::single_instance::acquire_moved(Path::new(&format!("{}.lock", target.display())))?;
    // Fold the WAL into the file so one copy holds everything.
    crate::startup::db_security::checkpoint(&from, config.state.encryption_key.as_deref()).await?;
    std::fs::copy(&from, &target)?;
    aidaemon_tools::config_file::set_owner_only_permissions(&target);
    if let Err(e) =
        crate::startup::db_security::check_opens(&target, config.state.encryption_key.as_deref())
            .await
    {
        let _ = std::fs::remove_file(&target);
        return Err(e.context("the copy does not open; the database stays where it was"));
    }
    let target_text = target.to_string_lossy().into_owned();
    if let Err(e) = set_config_value(config_path, "state.db_path", &target_text).await {
        let _ = std::fs::remove_file(&target);
        return Err(e.context("could not save the new path; the database stays where it was"));
    }
    config.state.db_path = target_text;
    let kept = PathBuf::from(format!("{}.moved-{}", from.display(), now_ms()));
    std::fs::rename(&from, &kept)?;
    for sidecar in sidecars(&from) {
        let _ = std::fs::remove_file(sidecar);
    }
    Ok(format!(
        "Moved the database to {}. The old file is kept at {}; delete it once all is well.",
        target.display(),
        kept.display()
    ))
}

/// Re-encrypt the database under a new key, then store the key where the
/// old one came from. A backup copy under the old key is kept until the new
/// key is stored; if storing fails, the database goes back to the old key.
#[cfg(feature = "encryption")]
async fn rotate_key(
    config: &mut crate::config::AppConfig,
    config_path: &Path,
) -> anyhow::Result<String> {
    use crate::startup::db_security as db;
    let path = db::resolve_db_path(config_path, &config.state.db_path);
    let old = config
        .state
        .encryption_key
        .clone()
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("the database is not encrypted"))?;
    let source = key_source(config_path)?;
    db::checkpoint(&path, Some(&old)).await?;
    let backup = PathBuf::from(format!("{}.pre-rekey-{}", path.display(), now_ms()));
    std::fs::copy(&path, &backup)?;
    aidaemon_tools::config_file::set_owner_only_permissions(&backup);
    let new = db::new_encryption_key();
    db::rekey(&path, &old, &new).await?;
    if let Err(e) = db::check_opens(&path, Some(&new)).await {
        std::fs::copy(&backup, &path)?;
        return Err(e.context("the re-encrypted database did not open; the old one is back"));
    }
    if let Err(e) = store_key(&source, config_path, &new) {
        db::rekey(&path, &new, &old).await?;
        return Err(e.context("could not store the new key; the database is back on the old one"));
    }
    config.state.encryption_key = Some(new);
    let _ = std::fs::remove_file(&backup);
    Ok("Rotated the database encryption key.".to_string())
}

#[cfg(not(feature = "encryption"))]
async fn rotate_key(
    _config: &mut crate::config::AppConfig,
    _config_path: &Path,
) -> anyhow::Result<String> {
    anyhow::bail!("this build has no database encryption")
}

/// Where the database key is kept.
#[cfg(feature = "encryption")]
#[derive(Debug, PartialEq, Eq)]
enum KeySource {
    Keychain,
    EnvFile(PathBuf),
}

/// Where the key came from, when the daemon can store a new one there.
#[cfg(feature = "encryption")]
fn key_source(config_path: &Path) -> anyhow::Result<KeySource> {
    let file_value = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok())
        .and_then(|table| {
            table
                .get("state")?
                .get("encryption_key")?
                .as_str()
                .map(str::to_string)
        });
    match file_value.as_deref().map(str::trim) {
        Some("keychain") => return Ok(KeySource::Keychain),
        Some(value) if !value.is_empty() => anyhow::bail!(
            "the key is written in config.toml (or names an environment variable); move it to \
             the keychain first"
        ),
        _ => {}
    }
    let env_file = crate::startup::db_security::env_file_path(config_path);
    let in_file = std::fs::read_to_string(&env_file)
        .map(|text| {
            text.lines()
                .any(|line| line.trim_start().starts_with("AIDAEMON_ENCRYPTION_KEY="))
        })
        .unwrap_or(false);
    anyhow::ensure!(
        in_file,
        "the key comes from the environment; rotate it where the environment is set"
    );
    Ok(KeySource::EnvFile(env_file))
}

#[cfg(feature = "encryption")]
fn store_key(source: &KeySource, _config_path: &Path, key: &str) -> anyhow::Result<()> {
    match source {
        KeySource::Keychain => crate::config::store_in_keychain("encryption_key", key),
        KeySource::EnvFile(path) => crate::startup::db_security::write_env_key(path, key),
    }
}

/// Set one plain setting in the config file, kept as TOML text.
async fn set_config_value(config_path: &Path, key: &str, value: &str) -> anyhow::Result<()> {
    let file = crate::config::config_file(config_path);
    let mut doc: toml_edit::DocumentMut = file.read().await?.parse()?;
    let mut parts = key.split('.').peekable();
    let mut table = doc.as_table_mut();
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            table[part] = toml_edit::value(value);
        } else {
            if !table.contains_key(part) {
                table[part] = toml_edit::table();
            }
            table = table[part]
                .as_table_mut()
                .ok_or_else(|| anyhow::anyhow!("`{part}` is not a table"))?;
        }
    }
    file.save(&doc.to_string()).await
}

// ---------------------------------------------------------------------------
// The gateway address trial
// ---------------------------------------------------------------------------

/// Keep the config as it is now, so a gateway address change can roll back.
pub(crate) fn begin_trial(config_path: &Path) -> anyhow::Result<()> {
    // A second change during a trial keeps the address that last worked.
    if !trial_marker(config_path).exists() {
        std::fs::copy(config_path, trial_previous(config_path))?;
        aidaemon_tools::config_file::set_owner_only_permissions(&trial_previous(config_path));
    }
    write_owner_only(
        &trial_marker(config_path),
        &format!("{{\"at_ms\":{}}}", now_ms()),
    )?;
    Ok(())
}

/// Whether a gateway address change waits for its trial.
pub(crate) fn trial_active(config_path: &Path) -> bool {
    trial_marker(config_path).exists()
}

/// Drop a trial begun for a change that was then not saved.
pub(crate) fn cancel_trial(config_path: &Path) {
    let _ = std::fs::remove_file(trial_marker(config_path));
    let _ = std::fs::remove_file(trial_previous(config_path));
}

/// A gateway address trial running this start.
pub(crate) struct Trial {
    config_path: PathBuf,
    confirmed: AtomicBool,
}

impl Trial {
    /// An app reached the gateway: the new address works.
    pub(crate) fn confirm(&self) {
        if self.confirmed.swap(true, Ordering::SeqCst) {
            return;
        }
        let _ = std::fs::remove_file(trial_marker(&self.config_path));
        let _ = std::fs::remove_file(trial_previous(&self.config_path));
        info!("An app reconnected on the new gateway address; keeping it");
    }
}

/// At start: when a gateway address change is on trial, wait for an app to
/// reach the gateway; if none does in time, put the previous config back
/// and restart.
pub(crate) fn arm_trial(config_path: &Path) -> Option<Arc<Trial>> {
    if !trial_marker(config_path).exists() {
        return None;
    }
    let trial = Arc::new(Trial {
        config_path: config_path.to_path_buf(),
        confirmed: AtomicBool::new(false),
    });
    let waiting = trial.clone();
    tokio::spawn(async move {
        tokio::time::sleep(GATEWAY_TRIAL).await;
        if waiting.confirmed.load(Ordering::SeqCst) {
            return;
        }
        let previous = trial_previous(&waiting.config_path);
        warn!("No app reconnected on the new gateway address; restoring the previous config");
        if let Err(e) = std::fs::copy(&previous, &waiting.config_path) {
            error!(error = %e, "Could not restore the config from before the address change");
            return;
        }
        let _ = std::fs::remove_file(trial_marker(&waiting.config_path));
        let _ = std::fs::remove_file(previous);
        restart_in_place();
    });
    Some(trial)
}

/// Give the new process the env file's current values for the variables
/// this one took from it (a rotated database key, a secret stored there),
/// and drop those the file no longer has. Variables set outside the file
/// keep their values.
fn refresh_env_file_values(command: &mut std::process::Command) {
    let Some(keys) = crate::ENV_FILE_KEYS.get() else {
        return;
    };
    let Ok(path) = std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY) else {
        return;
    };
    let current: std::collections::HashMap<String, String> = dotenvy::from_path_iter(&path)
        .map(|items| items.filter_map(Result::ok).collect())
        .unwrap_or_default();
    for key in keys {
        match current.get(key) {
            Some(value) => command.env(key, value),
            None => command.env_remove(key),
        };
    }
    // Ones added to the file since this process started.
    let mut taken = keys.clone();
    for (key, value) in &current {
        if !keys.contains(key) && std::env::var_os(key).is_none() {
            command.env(key, value);
            taken.push(key.clone());
        }
    }
    command.env(crate::ENV_FILE_KEYS_VAR, taken.join(","));
}

/// Replace this process with a fresh one of itself (same arguments). Returns
/// only if that failed.
pub(crate) fn restart_in_place() {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let exe = match std::env::current_exe() {
            Ok(exe) => exe,
            Err(e) => {
                error!("Failed to get current exe path: {e}");
                return;
            }
        };
        let args: Vec<String> = std::env::args().skip(1).collect();
        info!(exe = %exe.display(), "Exec-ing new process");
        let mut command = std::process::Command::new(&exe);
        command.args(&args);
        refresh_env_file_values(&mut command);
        let err = command.exec();
        error!("exec failed: {err}");
    }
    // No exec: start a fresh copy, which waits for this one's instance
    // lock, then leave.
    #[cfg(not(unix))]
    {
        let exe = match std::env::current_exe() {
            Ok(exe) => exe,
            Err(e) => {
                error!("Failed to get current exe path: {e}");
                return;
            }
        };
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut command = std::process::Command::new(&exe);
        command
            .args(&args)
            .env(crate::single_instance::RESTART_WAIT_VAR, "1");
        refresh_env_file_values(&mut command);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: it outlives us.
            command.creation_flags(0x0000_0008 | 0x0000_0200);
        }
        match command.spawn() {
            Ok(_) => {
                info!(exe = %exe.display(), "Started a fresh process; this one exits");
                std::process::exit(0);
            }
            Err(e) => error!("Could not start a fresh process: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scheduled_upkeep_is_shown_and_can_be_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, "").unwrap();
        assert!(notices(&config).is_empty());
        schedule(
            &config,
            &DbUpkeep::MoveTo {
                path: "/data/synthetic.db".into(),
            },
        )
        .unwrap();
        assert_eq!(
            scheduled(&config),
            Some(DbUpkeep::MoveTo {
                path: "/data/synthetic.db".into()
            })
        );
        assert!(notices(&config)[0].contains("/data/synthetic.db"));
        assert!(cancel(&config));
        assert!(scheduled(&config).is_none());
    }

    #[tokio::test]
    async fn a_confirmed_trial_keeps_the_new_address() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, "[client_gateway]\nport = 1\n").unwrap();
        begin_trial(&config).unwrap();
        std::fs::write(&config, "[client_gateway]\nport = 2\n").unwrap();
        // A second change keeps the copy from before the first.
        begin_trial(&config).unwrap();
        assert!(
            std::fs::read_to_string(trial_previous(&config))
                .unwrap()
                .contains("port = 1")
        );
        let trial = arm_trial(&config).expect("on trial");
        trial.confirm();
        assert!(!trial_marker(&config).exists());
        assert!(!trial_previous(&config).exists());
        assert!(arm_trial(&config).is_none());
    }

    #[tokio::test]
    async fn set_config_value_adds_the_table_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, "[provider]\napi_key = \"keychain\"\n").unwrap();
        set_config_value(&config, "state.db_path", "/data/synthetic.db")
            .await
            .unwrap();
        let text = std::fs::read_to_string(&config).unwrap();
        let table: toml::Table = text.parse().unwrap();
        assert_eq!(
            table["state"]["db_path"].as_str(),
            Some("/data/synthetic.db")
        );
        assert_eq!(table["provider"]["api_key"].as_str(), Some("keychain"));
    }
}
