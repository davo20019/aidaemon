//! Write provenance for the untrusted-content rule.
//!
//! A file's trust is otherwise judged by where it lies (inside the owner's
//! trusted roots or not). A run that has read someone else's text can write
//! that text into a trusted root, and a later clean run reading it back
//! would take it as the owner's. So each file a tainted run writes, through
//! a tool that declares the paths it writes (`guarded_paths`), is recorded
//! with the digest of what it wrote; a later read of a recorded path whose
//! content still has that digest is untrusted output and taints the reader.
//! Once the file changes (the owner edits it) the record no longer applies.
//!
//! Not covered: writes a tool does not declare, such as a shell command's
//! (the terminal reports no typed list of the files it changed), and files
//! on a remote execution backend (digests are read from this machine).

use sha2::{Digest, Sha256};
use tracing::warn;

use aidaemon_tools::services::SettingsStore;

/// How many written files the record keeps (oldest dropped first).
const KEPT: usize = 1024;

/// Files larger than this are not digested (recorded writes of them are
/// skipped; a read of one is judged by location as before).
const MAX_DIGEST_BYTES: u64 = 64 * 1024 * 1024;

const KEY: &str = "untrusted_writes";

/// Serializes the read-modify-write in [`record`].
static WRITE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn digest(path: &str) -> Option<String> {
    let metadata = tokio::fs::metadata(path).await.ok()?;
    if !metadata.is_file() || metadata.len() > MAX_DIGEST_BYTES {
        return None;
    }
    let bytes = tokio::fs::read(path).await.ok()?;
    Some(hex::encode(Sha256::digest(&bytes)))
}

async fn load(state: &(impl SettingsStore + ?Sized)) -> Vec<(String, String)> {
    match state.get_setting(KEY).await {
        Ok(Some(raw)) if !raw.trim().is_empty() => {
            serde_json::from_str(&raw).unwrap_or_else(|error| {
                warn!(%error, "Replacing an unreadable untrusted-write record");
                Vec::new()
            })
        }
        Ok(_) => Vec::new(),
        Err(error) => {
            warn!(%error, "Could not read the untrusted-write record");
            Vec::new()
        }
    }
}

/// Record `paths`, written by a run that read untrusted content, with the
/// digest of what they now hold. Relative paths and files that cannot be
/// read are skipped.
pub async fn record(state: &(impl SettingsStore + ?Sized), paths: &[String]) -> anyhow::Result<()> {
    let mut written = Vec::new();
    for path in paths {
        if !aidaemon_tools::BackendPath::is_absolute(path) {
            continue;
        }
        if let Some(digest) = digest(path).await {
            written.push((path.clone(), digest));
        }
    }
    if written.is_empty() {
        return Ok(());
    }
    let _write = WRITE.lock().await;
    let mut record = load(state).await;
    for (path, digest) in written {
        record.retain(|(recorded, _)| *recorded != path);
        record.push((path, digest));
    }
    let excess = record.len().saturating_sub(KEPT);
    record.drain(..excess);
    state
        .set_setting(KEY, &serde_json::to_string(&record)?)
        .await
}

/// Whether any of `paths` still holds what a tainted run wrote there.
pub async fn holds_untrusted_write(
    state: &(impl SettingsStore + ?Sized),
    paths: &[String],
) -> bool {
    if paths.is_empty() {
        return false;
    }
    let record = load(state).await;
    for path in paths {
        let Some((_, recorded)) = record.iter().find(|(written, _)| written == path) else {
            continue;
        };
        if digest(path).await.as_deref() == Some(recorded.as_str()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MemorySettings(std::sync::Mutex<std::collections::HashMap<String, String>>);

    #[async_trait::async_trait]
    impl SettingsStore for MemorySettings {
        async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }

        async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_string(), value.to_string());
            Ok(())
        }
    }

    /// A file a tainted run wrote reads as untrusted while it holds what
    /// was written, and as before once someone changes it; a file never
    /// recorded is unaffected.
    #[tokio::test]
    async fn a_tainted_runs_write_stays_untrusted_until_it_changes() {
        let state = MemorySettings::default();
        let dir = tempfile::tempdir().unwrap();
        let notes = dir.path().join("notes.md");
        let other = dir.path().join("other.md");
        std::fs::write(&notes, "text from a web page").unwrap();
        std::fs::write(&other, "the owner's").unwrap();
        let notes = notes.to_string_lossy().to_string();
        let other = other.to_string_lossy().to_string();

        record(&state, &[notes.clone(), "relative.md".to_string()])
            .await
            .unwrap();
        assert!(holds_untrusted_write(&state, std::slice::from_ref(&notes)).await);
        assert!(!holds_untrusted_write(&state, std::slice::from_ref(&other)).await);
        assert!(holds_untrusted_write(&state, &[other.clone(), notes.clone()]).await);

        std::fs::write(&notes, "rewritten by the owner").unwrap();
        assert!(!holds_untrusted_write(&state, std::slice::from_ref(&notes)).await);
    }
}
