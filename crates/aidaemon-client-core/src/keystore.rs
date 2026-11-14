//! Where a paired device keeps its identity: the gateway it paired with, the
//! device id it was given and its private key. [`FileKeyStore`] keeps it in
//! one file readable by the owner only; a client may add an OS keychain
//! implementation.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A paired device's identity.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredDevice {
    /// The gateway's HTTP base (`http://127.0.0.1:8788`).
    pub gateway: String,
    pub device_id: String,
    /// The P-256 private scalar, 32 bytes, base64url.
    pub secret_key: String,
}

impl std::fmt::Debug for StoredDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredDevice")
            .field("gateway", &self.gateway)
            .field("device_id", &self.device_id)
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

/// Load, save and forget a device identity.
pub trait KeyStore: Send + Sync {
    /// The saved identity; `None` when the device is not paired.
    fn load(&self) -> anyhow::Result<Option<StoredDevice>>;
    fn save(&self, device: &StoredDevice) -> anyhow::Result<()>;
    /// Forget it (unpair). Forgetting what is not there succeeds.
    fn clear(&self) -> anyhow::Result<()>;
}

/// A JSON file, written atomically, mode 0600 on Unix.
#[derive(Debug, Clone)]
pub struct FileKeyStore {
    path: PathBuf,
}

impl FileKeyStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl KeyStore for FileKeyStore {
    fn load(&self) -> anyhow::Result<Option<StoredDevice>> {
        match std::fs::read_to_string(&self.path) {
            Ok(text) => Ok(Some(serde_json::from_str(&text)?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn save(&self, device: &StoredDevice) -> anyhow::Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(device)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    fn clear(&self) -> anyhow::Result<()> {
        match std::fs::remove_file(&self.path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error.into()),
            _ => Ok(()),
        }
    }
}

/// Kept in memory only: tests, and a client that pairs for one run.
#[derive(Debug, Default)]
pub struct MemoryKeyStore(std::sync::Mutex<Option<StoredDevice>>);

impl KeyStore for MemoryKeyStore {
    fn load(&self) -> anyhow::Result<Option<StoredDevice>> {
        Ok(self.0.lock().unwrap_or_else(|p| p.into_inner()).clone())
    }

    fn save(&self, device: &StoredDevice) -> anyhow::Result<()> {
        *self.0.lock().unwrap_or_else(|p| p.into_inner()) = Some(device.clone());
        Ok(())
    }

    fn clear(&self) -> anyhow::Result<()> {
        *self.0.lock().unwrap_or_else(|p| p.into_inner()) = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_store_round_trips_and_clears() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileKeyStore::new(dir.path().join("nested/device.json"));
        assert_eq!(store.load().unwrap(), None);
        let device = StoredDevice {
            gateway: "http://127.0.0.1:8788".into(),
            device_id: "dev-synthetic-1".into(),
            secret_key: "c2VjcmV0".into(),
        };
        store.save(&device).unwrap();
        assert_eq!(store.load().unwrap(), Some(device.clone()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(store.path())
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert!(!format!("{device:?}").contains("c2VjcmV0"));
        store.clear().unwrap();
        store.clear().unwrap();
        assert_eq!(store.load().unwrap(), None);
    }
}
