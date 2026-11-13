//! Secret storage for aidaemon agents.
//!
//! [`SecretStore`] gets, sets and deletes a secret by key. The default,
//! [`KeychainSecretStore`], keeps secrets in the OS credential store under one
//! service name and, when the environment variable named in its settings is
//! `1` or `true`, in an env file instead ([`EnvSecretStore`]).
//! [`MemorySecretStore`] keeps them in memory for tests. [`names`] builds the
//! account names aidaemon's configuration uses.
//!
//! Secret values never appear in errors or logs; keys and file paths may.
//! A store hands a value out as a [`Secret`]: its `Debug` is redacted, its
//! memory is wiped on drop, and creating it registers the value with
//! [`aidaemon_core::redaction`], so every agent run in the process redacts
//! it from model requests, events, receipts, approval questions and saved
//! state without being told about it.

// Tests need one `unsafe` helper to set environment variables (edition 2024);
// the library itself stays free of `unsafe`.
#![cfg_attr(not(test), forbid(unsafe_code))]
#![cfg_attr(test, deny(unsafe_code))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub mod names;

/// A secret value. `Debug` prints `Secret([REDACTED])`, there is no
/// `Display`, and the memory is zeroed on drop. Creating one registers the
/// value with the kernel's redaction ([`aidaemon_core::redaction`]), so a
/// run that uses it never sends it out (see that module for what is
/// covered). Read it with [`Secret::expose`] only where it is used.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(zeroize::Zeroizing<String>);

impl Secret {
    /// Wrap `value` and register it for redaction. A value too short to
    /// redact (under [`aidaemon_core::redaction::MIN_SECRET_CHARS`]) is
    /// still returned, with a warning logged (without the value): it will
    /// appear as is wherever it is used.
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        if let Err(error) = aidaemon_core::redaction::register_secret(&value) {
            tracing::warn!(%error, "secret not redacted");
        }
        Self(zeroize::Zeroizing::new(value))
    }

    /// The value, for the one place that needs it.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// The value as a plain `String`, which is no longer wiped on drop.
    pub fn into_exposed(self) -> String {
        self.0.to_string()
    }

    /// Whether the value is long enough for the kernel to redact it.
    pub fn is_redacted(&self) -> bool {
        self.0.chars().count() >= aidaemon_core::redaction::MIN_SECRET_CHARS
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret([REDACTED])")
    }
}

impl PartialEq<str> for Secret {
    fn eq(&self, other: &str) -> bool {
        self.expose() == other
    }
}

impl PartialEq<&str> for Secret {
    fn eq(&self, other: &&str) -> bool {
        self.expose() == *other
    }
}

/// Where secrets are kept, by key.
///
/// Implementations must never log or put a secret's value in an error.
pub trait SecretStore: Send + Sync {
    /// Read the secret stored under `key`. Errors when it is missing.
    /// Implementations build the value with [`Secret::new`], which
    /// registers it for redaction.
    fn get(&self, key: &str) -> anyhow::Result<Secret>;
    /// Store `value` under `key`, replacing any earlier value.
    fn set(&self, key: &str, value: &str) -> anyhow::Result<()>;
    /// Remove the secret under `key`.
    fn delete(&self, key: &str) -> anyhow::Result<()>;
    /// True when a secret is stored under `key`.
    fn contains(&self, key: &str) -> bool {
        self.get(key).is_ok()
    }
}

/// Secrets kept in memory, for tests and short-lived tools.
#[derive(Default)]
pub struct MemorySecretStore {
    values: Mutex<HashMap<String, String>>,
}

impl MemorySecretStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn values(&self) -> std::sync::MutexGuard<'_, HashMap<String, String>> {
        self.values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl SecretStore for MemorySecretStore {
    fn get(&self, key: &str) -> anyhow::Result<Secret> {
        self.values()
            .get(key)
            .cloned()
            .map(Secret::new)
            .ok_or_else(|| anyhow::anyhow!("secret '{}' not set", key))
    }

    fn set(&self, key: &str, value: &str) -> anyhow::Result<()> {
        let _ = aidaemon_core::redaction::register_secret(value);
        self.values().insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn delete(&self, key: &str) -> anyhow::Result<()> {
        self.values().remove(key);
        Ok(())
    }
}

/// Where an [`EnvSecretStore`] finds its env file.
#[derive(Debug, Clone)]
enum EnvFileLocation {
    Fixed(PathBuf),
    /// The first of these environment variables that is set and not blank,
    /// read at each call, else the default path.
    FromVars {
        vars: Vec<String>,
        default: PathBuf,
    },
}

/// Secrets kept in an env file (`KEY=value`, keys upper-cased), the
/// keychain-off mode of [`KeychainSecretStore`].
///
/// A read uses the file when it exists, else the process environment. Writes
/// go to the file, created with mode 0600.
#[derive(Debug, Clone)]
pub struct EnvSecretStore {
    location: EnvFileLocation,
}

impl EnvSecretStore {
    /// An env file at a fixed path.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            location: EnvFileLocation::Fixed(path.into()),
        }
    }

    /// An env file named by the first of `vars` that is set and not blank
    /// when a secret is read or written, else `default`.
    pub fn from_env_vars<I, S>(vars: I, default: impl Into<PathBuf>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            location: EnvFileLocation::FromVars {
                vars: vars.into_iter().map(Into::into).collect(),
                default: default.into(),
            },
        }
    }

    /// The env file this store reads and writes now.
    pub fn env_file_path(&self) -> PathBuf {
        match &self.location {
            EnvFileLocation::Fixed(path) => path.clone(),
            EnvFileLocation::FromVars { vars, default } => {
                for var in vars {
                    if let Ok(path) = std::env::var(var)
                        && !path.trim().is_empty()
                    {
                        return PathBuf::from(path);
                    }
                }
                default.clone()
            }
        }
    }
}

impl SecretStore for EnvSecretStore {
    fn get(&self, key: &str) -> anyhow::Result<Secret> {
        let env_key = key.to_uppercase();
        let env_path = self.env_file_path();
        if env_path.exists() {
            if let Some(val) = read_env_secret_from_file(&env_path, &env_key)?
                && !val.is_empty()
            {
                return Ok(Secret::new(val));
            }
            anyhow::bail!(
                "Keychain disabled and env var '{}' not found in '{}'",
                env_key,
                env_path.display()
            );
        }

        if let Ok(val) = std::env::var(&env_key)
            && !val.is_empty()
        {
            return Ok(Secret::new(val));
        }

        anyhow::bail!(
            "Keychain disabled and env var '{}' not set for '{}'",
            env_key,
            key
        );
    }

    fn set(&self, key: &str, value: &str) -> anyhow::Result<()> {
        let _ = aidaemon_core::redaction::register_secret(value);
        upsert_env_secret(&self.env_file_path(), key, value)
    }

    fn delete(&self, key: &str) -> anyhow::Result<()> {
        remove_env_secret(&self.env_file_path(), key)
    }
}

/// How a [`KeychainSecretStore`] reaches the OS credential store and its
/// env-file fallback.
#[derive(Debug, Clone)]
pub struct KeychainSettings {
    /// The keychain service every secret is filed under.
    pub service: String,
    /// An environment variable that, set to `1` or `true`, turns the keychain
    /// off in favour of the env file. Read at each call.
    pub disable_var: Option<String>,
    /// Environment variables naming the env file, first non-blank wins.
    pub env_file_vars: Vec<String>,
    /// The env file when none of `env_file_vars` is set.
    pub default_env_file: PathBuf,
}

impl KeychainSettings {
    /// The keychain under `service`, always on, with `.env` as the env file.
    pub fn new(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
            disable_var: None,
            env_file_vars: Vec::new(),
            default_env_file: PathBuf::from(".env"),
        }
    }
}

/// Secrets in the OS keychain, or in an env file while the keychain is
/// turned off (see [`KeychainSettings::disable_var`]).
pub struct KeychainSecretStore {
    settings: KeychainSettings,
    env: EnvSecretStore,
}

impl KeychainSecretStore {
    pub fn new(settings: KeychainSettings) -> Self {
        let env = EnvSecretStore::from_env_vars(
            settings.env_file_vars.clone(),
            settings.default_env_file.clone(),
        );
        Self { settings, env }
    }

    pub fn settings(&self) -> &KeychainSettings {
        &self.settings
    }

    /// True while the disable variable turns the keychain off.
    pub fn keychain_disabled(&self) -> bool {
        self.settings
            .disable_var
            .as_deref()
            .is_some_and(|var| std::env::var(var).is_ok_and(|v| v == "1" || v == "true"))
    }

    /// The env-file store used while the keychain is off.
    pub fn env_store(&self) -> &EnvSecretStore {
        &self.env
    }

    #[cfg(feature = "keychain")]
    fn entry(&self, key: &str) -> anyhow::Result<keyring::Entry> {
        Ok(keyring::Entry::new(&self.settings.service, key)?)
    }
}

#[cfg(feature = "keychain")]
impl SecretStore for KeychainSecretStore {
    fn get(&self, key: &str) -> anyhow::Result<Secret> {
        if self.keychain_disabled() {
            return self.env.get(key);
        }
        self.entry(key)?
            .get_password()
            .map(Secret::new)
            .map_err(|e| anyhow::anyhow!("Failed to read '{}' from OS keychain: {}", key, e))
    }

    fn set(&self, key: &str, value: &str) -> anyhow::Result<()> {
        let _ = aidaemon_core::redaction::register_secret(value);
        if self.keychain_disabled() {
            return self.env.set(key, value);
        }
        self.entry(key)?.set_password(value)?;
        Ok(())
    }

    fn delete(&self, key: &str) -> anyhow::Result<()> {
        if self.keychain_disabled() {
            return self.env.delete(key);
        }
        self.entry(key)?
            .delete_credential()
            .map_err(|e| anyhow::anyhow!("Failed to delete '{}' from OS keychain: {}", key, e))
    }
}

/// Without the `keychain` feature the store always uses its env file.
#[cfg(not(feature = "keychain"))]
impl SecretStore for KeychainSecretStore {
    fn get(&self, key: &str) -> anyhow::Result<Secret> {
        self.env.get(key)
    }

    fn set(&self, key: &str, value: &str) -> anyhow::Result<()> {
        self.env.set(key, value)
    }

    fn delete(&self, key: &str) -> anyhow::Result<()> {
        self.env.delete(key)
    }
}

/// One writer at a time for every env file in the process.
fn env_file_lock() -> &'static Mutex<()> {
    static LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn is_env_key_assignment(line: &str, env_key: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with('#') {
        return false;
    }
    let Some((k, _)) = trimmed.split_once('=') else {
        return false;
    };
    k.trim() == env_key
}

fn encode_env_value(value: &str) -> String {
    let is_simple = value.bytes().all(|b| {
        matches!(
            b,
            b'A'..=b'Z'
                | b'a'..=b'z'
                | b'0'..=b'9'
                | b'_'
                | b'-'
                | b'.'
                | b'/'
                | b':'
                | b'@'
                | b'+'
                | b'='
        )
    });
    if is_simple {
        return value.to_string();
    }
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '$' => escaped.push_str("\\$"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            _ => escaped.push(ch),
        }
    }
    format!("\"{}\"", escaped)
}

#[cfg(unix)]
fn set_file_mode_0600(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_file_mode_0600(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

fn upsert_env_secret(env_path: &Path, field_name: &str, value: &str) -> anyhow::Result<()> {
    let _guard = env_file_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("Failed to lock env file writer"))?;
    let env_key = field_name.to_uppercase();
    let assignment = format!("{}={}", env_key, encode_env_value(value));

    let mut lines: Vec<String> = if env_path.exists() {
        std::fs::read_to_string(env_path)?
            .lines()
            .map(|l| l.to_string())
            .collect()
    } else {
        Vec::new()
    };

    let mut replaced = false;
    for line in &mut lines {
        if is_env_key_assignment(line, &env_key) {
            *line = assignment.clone();
            replaced = true;
        }
    }
    if !replaced {
        lines.push(assignment);
    }

    if let Some(parent) = env_path.parent()
        && !parent.as_os_str().is_empty()
    {
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

fn remove_env_secret(env_path: &Path, field_name: &str) -> anyhow::Result<()> {
    let _guard = env_file_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("Failed to lock env file writer"))?;
    if !env_path.exists() {
        return Ok(());
    }

    let env_key = field_name.to_uppercase();
    let mut lines: Vec<String> = std::fs::read_to_string(env_path)?
        .lines()
        .map(|l| l.to_string())
        .collect();
    let original_len = lines.len();
    lines.retain(|line| !is_env_key_assignment(line, &env_key));

    if lines.len() == original_len {
        return Ok(());
    }

    let mut body = lines.join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    std::fs::write(env_path, body)?;
    set_file_mode_0600(env_path)?;
    Ok(())
}

fn decode_env_value_loose(raw: &str) -> String {
    let trimmed = raw.trim();
    let inner = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len().saturating_sub(1)]
    } else {
        trimmed
    };

    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('$') => out.push('$'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn read_env_secret_from_file(env_path: &Path, env_key: &str) -> anyhow::Result<Option<String>> {
    if !env_path.exists() {
        return Ok(None);
    }

    let mut found_via_dotenv: Option<String> = None;
    if let Ok(iter) = dotenvy::from_path_iter(env_path) {
        for (key, value) in iter.flatten() {
            // Malformed lines are skipped; the loose scan below still sees them.
            if key == env_key {
                found_via_dotenv = Some(value);
            }
        }
    }
    if found_via_dotenv.is_some() {
        return Ok(found_via_dotenv);
    }

    let content = std::fs::read_to_string(env_path)?;
    let mut found_via_loose_scan: Option<String> = None;
    for line in content.lines() {
        if is_env_key_assignment(line, env_key)
            && let Some((_, raw_value)) = line.split_once('=')
        {
            found_via_loose_scan = Some(decode_env_value_loose(raw_value));
        }
    }
    Ok(found_via_loose_scan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sets an environment variable that no other test uses, and removes it
    /// again when dropped.
    struct ScopedVar(&'static str);

    #[allow(unsafe_code)]
    impl ScopedVar {
        fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            // SAFETY: test-only, and the name is private to one test. std
            // serializes its own environment reads and writes; foreign code
            // calling getenv concurrently is a risk the test binary accepts.
            unsafe { std::env::set_var(name, value) };
            Self(name)
        }
    }

    #[allow(unsafe_code)]
    impl Drop for ScopedVar {
        fn drop(&mut self) {
            // SAFETY: as in `ScopedVar::set`.
            unsafe { std::env::remove_var(self.0) };
        }
    }

    #[test]
    fn env_store_set_quotes_and_keeps_other_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(&env_path, "EXISTING=1\n").unwrap();
        let store = EnvSecretStore::at(&env_path);

        store
            .set("oauth_twitter_access_token", "new token$1")
            .expect("write oauth token to env");

        let content = std::fs::read_to_string(&env_path).expect("read env file");
        assert!(content.contains("EXISTING=1"));
        assert!(content.contains("OAUTH_TWITTER_ACCESS_TOKEN=\"new token\\$1\""));
        assert_eq!(
            store.get("oauth_twitter_access_token").unwrap(),
            "new token$1"
        );
    }

    #[cfg(unix)]
    #[test]
    fn env_store_writes_the_file_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join("nested").join(".env");
        EnvSecretStore::at(&env_path).set("api_key", "x").unwrap();
        let mode = std::fs::metadata(&env_path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn env_store_delete_removes_only_that_key() {
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(
            &env_path,
            "KEEP=1\nOAUTH_TWITTER_ACCESS_TOKEN=oldtoken\nANOTHER=2\n",
        )
        .unwrap();

        EnvSecretStore::at(&env_path)
            .delete("oauth_twitter_access_token")
            .expect("delete oauth token from env");

        let content = std::fs::read_to_string(&env_path).expect("read env file");
        assert!(content.contains("KEEP=1"));
        assert!(content.contains("ANOTHER=2"));
        assert!(!content.contains("OAUTH_TWITTER_ACCESS_TOKEN"));
    }

    #[test]
    fn env_store_reads_upper_cased_key() {
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(&env_path, "OAUTH_TWITTER_CLIENT_ID=client123\n").unwrap();

        let value = EnvSecretStore::at(&env_path)
            .get("oauth_twitter_client_id")
            .expect("resolve client id");
        assert_eq!(value, "client123");
    }

    #[test]
    fn env_store_tolerates_malformed_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(
            &env_path,
            "BROKEN=\"unterminated\nOAUTH_TWITTER_CLIENT_ID=client123\n",
        )
        .unwrap();

        let value = EnvSecretStore::at(&env_path)
            .get("oauth_twitter_client_id")
            .expect("resolve client id despite malformed env line");
        assert_eq!(value, "client123");
    }

    #[test]
    fn env_store_missing_key_names_key_and_file_not_values() {
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(&env_path, "OTHER=hunter2\n").unwrap();
        let error = EnvSecretStore::at(&env_path)
            .get("api_key")
            .unwrap_err()
            .to_string();
        assert!(error.contains("'API_KEY' not found in"));
        assert!(!error.contains("hunter2"));
    }

    #[test]
    fn env_file_path_prefers_the_first_set_variable() {
        // Private variable names, so no other test races on them.
        let first = "AIDAEMON_SECRETS_TEST_FIRST_ENV_FILE";
        let second = "AIDAEMON_SECRETS_TEST_SECOND_ENV_FILE";
        let second_set = ScopedVar::set(second, "/some/other/.env");
        let store = EnvSecretStore::from_env_vars([first, second], ".env");
        {
            let _first = ScopedVar::set(first, "/daemon/root/.env");
            assert_eq!(store.env_file_path(), PathBuf::from("/daemon/root/.env"));
        }
        {
            let _first = ScopedVar::set(first, "  ");
            assert_eq!(store.env_file_path(), PathBuf::from("/some/other/.env"));
        }
        drop(second_set);
        assert_eq!(store.env_file_path(), PathBuf::from(".env"));
    }

    #[test]
    fn keychain_store_uses_the_env_file_while_disabled() {
        let disable = "AIDAEMON_SECRETS_TEST_NO_KEYCHAIN";
        let file_var = "AIDAEMON_SECRETS_TEST_KEYCHAIN_ENV_FILE";
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env");
        let _disable = ScopedVar::set(disable, "true");
        let _file_var = ScopedVar::set(file_var, &env_path);
        let store = KeychainSecretStore::new(KeychainSettings {
            disable_var: Some(disable.into()),
            env_file_vars: vec![file_var.into()],
            ..KeychainSettings::new("aidaemon-secrets-test")
        });
        assert!(store.keychain_disabled());
        store.set("bot_token", "abc").unwrap();
        assert_eq!(store.get("bot_token").unwrap(), "abc");
        assert!(
            std::fs::read_to_string(&env_path)
                .unwrap()
                .contains("BOT_TOKEN=abc")
        );
        store.delete("bot_token").unwrap();
        assert!(!store.contains("bot_token"));
    }

    #[test]
    fn memory_store_round_trip() {
        let store = MemorySecretStore::new();
        assert!(!store.contains("k"));
        store.set("k", "v").unwrap();
        assert_eq!(store.get("k").unwrap(), "v");
        store.delete("k").unwrap();
        assert!(store.get("k").is_err());
    }
}
