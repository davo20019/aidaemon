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
