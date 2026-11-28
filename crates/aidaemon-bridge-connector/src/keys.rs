//! The daemon's two bridge keys: a login key (signs only the bridge's login
//! challenge) and an identity key (signs only handshakes), kept separate so
//! a signature made for one can never serve the other.

use std::path::Path;

use p256::ecdsa::SigningKey;
use p256::elliptic_curve::Generate as _;

/// The daemon's bridge keys.
pub struct BridgeKeys {
    pub login: SigningKey,
    pub identity: SigningKey,
}

impl BridgeKeys {
    /// Load the keys from `path`, or make and save new ones (owner-only
    /// file permissions on Unix).
    pub fn load_or_create(path: &Path) -> anyhow::Result<Self> {
        if let Ok(text) = std::fs::read_to_string(path) {
            let mut lines = text.lines();
            let login = key_of(lines.next())?;
            let identity = key_of(lines.next())?;
            return Ok(Self { login, identity });
        }
        let keys = Self {
            login: SigningKey::generate_from_rng(&mut rand::rng()),
            identity: SigningKey::generate_from_rng(&mut rand::rng()),
        };
        let text = format!(
            "{}\n{}\n",
            hex::encode(keys.login.to_bytes()),
            hex::encode(keys.identity.to_bytes())
        );
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        write_private(path, text.as_bytes())?;
        Ok(keys)
    }

    /// The login key's public point (65-byte SEC1).
    pub fn login_pub(&self) -> Vec<u8> {
        aidaemon_bridge_proto::ids::verifying_key_bytes(self.login.verifying_key()).to_vec()
    }

    /// The identity key's public point (65-byte SEC1).
    pub fn identity_pub(&self) -> Vec<u8> {
        aidaemon_bridge_proto::ids::verifying_key_bytes(self.identity.verifying_key()).to_vec()
    }
}

fn key_of(line: Option<&str>) -> anyhow::Result<SigningKey> {
    let bytes = hex::decode(line.unwrap_or_default().trim())?;
    SigningKey::from_slice(&bytes).map_err(|_| anyhow::anyhow!("unreadable bridge key"))
}

#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)
}

#[cfg(not(unix))]
fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}
