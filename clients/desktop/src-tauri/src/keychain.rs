//! The device identity in the OS keychain (macOS Keychain, Windows
//! Credential Manager, the Secret Service on Linux), so the private key
//! never sits in a file or enters the web view.

use aidaemon_client_core::keystore::{KeyStore, StoredDevice};

const SERVICE: &str = "ai.aidaemon.desktop";
const ACCOUNT: &str = "device";

pub struct KeychainStore;

impl KeychainStore {
    fn entry() -> anyhow::Result<keyring::Entry> {
        Ok(keyring::Entry::new(SERVICE, ACCOUNT)?)
    }
}

impl KeyStore for KeychainStore {
    fn load(&self) -> anyhow::Result<Option<StoredDevice>> {
        match Self::entry()?.get_password() {
            Ok(text) => Ok(Some(serde_json::from_str(&text)?)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(anyhow::anyhow!("the OS keychain could not be read: {error}")),
        }
    }

    fn save(&self, device: &StoredDevice) -> anyhow::Result<()> {
        Self::entry()?.set_password(&serde_json::to_string(device)?)?;
        Ok(())
    }

    fn clear(&self) -> anyhow::Result<()> {
        match Self::entry()?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(anyhow::anyhow!("the OS keychain could not be changed: {error}")),
        }
    }
}
