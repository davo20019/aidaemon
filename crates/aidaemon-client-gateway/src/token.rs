//! The local access token: minted by the owner on the machine
//! (`aidaemon client-token`), kept only as a SHA-256 hash in a file the
//! gateway reads on each check, so minting a new one revokes the old at
//! once without a restart.

use std::path::Path;

use sha2::{Digest, Sha256};

/// Every token starts with this, so a leaked one is recognisable.
pub const TOKEN_PREFIX: &str = "acg_";

fn hash(token: &str) -> String {
    hex::encode(Sha256::digest(token.trim().as_bytes()))
}

/// Mint a new token, replace the hash at `path` (readable by the owner
/// only) and return the token, which is not stored anywhere.
pub fn mint(path: &Path) -> std::io::Result<String> {
    let token = format!(
        "{TOKEN_PREFIX}{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, format!("{}\n", hash(&token)))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(token)
}

/// Whether `token` matches the hash at `path`. No file: nothing matches.
pub fn verify(path: &Path, token: &str) -> bool {
    let Ok(stored) = std::fs::read_to_string(path) else {
        return false;
    };
    let expected = stored.trim().as_bytes();
    let given = hash(token);
    let given = given.as_bytes();
    // Compare every byte, so the time taken says nothing about the hash.
    expected.len() == given.len()
        && expected
            .iter()
            .zip(given)
            .fold(0u8, |diff, (a, b)| diff | (a ^ b))
            == 0
}

// The owner's files for the local token: where its hash and the owner's
// own copy are kept, and minting or reading that copy.

use std::path::PathBuf;

use crate::ClientGatewayConfig;

/// Where the local client token's hash is kept.
pub fn token_file(config: &ClientGatewayConfig) -> PathBuf {
    match config.token_file.as_deref().map(str::trim) {
        Some(path) if !path.is_empty() => PathBuf::from(shellexpand::tilde(path).into_owned()),
        _ => dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".aidaemon")
            .join("client-gateway-token.sha256"),
    }
}

/// Where the owner's own copy of the local token is kept, next to its hash,
/// readable by the owner only, so `aidaemon tui` and `aidaemon web` need no
/// flags. Same trust as the config and the database beside it.
pub fn client_token_file(config: &ClientGatewayConfig) -> PathBuf {
    token_file(config).with_file_name("client-token")
}

/// Mint a new local token (revoking the old one at once), keep its hash for
/// the gateway and the owner's copy in [`client_token_file`]. Returns the
/// copy's path; the token itself is never printed or returned.
pub fn mint_local_token(config: &ClientGatewayConfig) -> std::io::Result<PathBuf> {
    let token = mint(&token_file(config))?;
    let path = client_token_file(config);
    write_owner_only_file(&path, &format!("{token}\n"))?;
    Ok(path)
}

/// The local token: `AIDAEMON_CLIENT_TOKEN`, else the owner's copy. A copy
/// that no longer matches the gateway's hash (minted elsewhere) is `None`.
pub fn local_token(config: &ClientGatewayConfig) -> Option<String> {
    if let Ok(token) = std::env::var("AIDAEMON_CLIENT_TOKEN")
        && !token.trim().is_empty()
    {
        return Some(token.trim().to_string());
    }
    let token = std::fs::read_to_string(client_token_file(config)).ok()?;
    let token = token.trim().to_string();
    verify(&token_file(config), &token).then_some(token)
}

fn write_owner_only_file(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, contents)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minted_token_verifies_until_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("client-token.sha256");
        assert!(!verify(&path, "acg_anything"));
        let first = mint(&path).unwrap();
        assert!(first.starts_with(TOKEN_PREFIX));
        assert!(verify(&path, &first));
        assert!(!verify(&path, "acg_wrong"));
        let second = mint(&path).unwrap();
        assert!(verify(&path, &second));
        assert!(
            !verify(&path, &first),
            "minting again revokes the old token"
        );
        assert!(!std::fs::read_to_string(&path).unwrap().contains(&second));
    }
}
