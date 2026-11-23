//! The bridge's key material: the stored daemon key, base64 and
//! signing helpers, and the device id derived from the key.

use super::*;

#[derive(Debug, Deserialize)]
pub(crate) struct StoredDaemonKey {
    #[serde(default)]
    pub(crate) private_key_b64: Option<String>,
    #[serde(default)]
    pub(crate) private_jwk: Option<StoredPrivateJwk>,
    #[allow(dead_code)]
    pub(crate) created_at: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StoredPrivateJwk {
    pub(crate) d: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct StoredDaemonKeyOnDisk {
    pub(crate) private_key_b64: String,
    pub(crate) created_at: String,
}

#[derive(Clone)]
pub(crate) struct KeyMaterial {
    pub(crate) private_key: SecretKey,
    pub(crate) public_raw: Vec<u8>,
    pub(crate) fingerprint: String,
}

#[cfg(unix)]
pub(crate) fn set_owner_only_permissions(path: &Path, mode: u32) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn set_owner_only_permissions(_path: &Path, _mode: u32) -> anyhow::Result<()> {
    Ok(())
}

/// The device id used when none is configured: the host name plus a prefix
/// of this install's bridge key fingerprint.
///
/// It must stay the same across restarts. The broker lets a reconnect replace
/// a still-registered daemon socket only when the device id matches, and
/// refuses a different one with 409 Conflict. An id that changed on every
/// restart (it used to include the process id) left the bridge refused for as
/// long as the broker held the previous process's socket.
pub(crate) fn default_device_id(key_fingerprint: &str) -> String {
    let host = std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "local".to_string());
    let key: String = key_fingerprint.chars().take(12).collect();
    sanitize_device_id(&format!("{host}-{key}"))
}

pub(crate) fn sanitize_device_id(value: &str) -> String {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .collect::<String>();
    if normalized.is_empty() {
        "local".to_string()
    } else {
        normalized.chars().take(64).collect()
    }
}

pub(crate) fn b64_encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub(crate) fn b64url_encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

pub(crate) fn b64_decode(input: &str) -> anyhow::Result<Vec<u8>> {
    Ok(base64::engine::general_purpose::STANDARD.decode(input)?)
}

pub(crate) fn b64_decode_flexible(input: &str) -> anyhow::Result<Vec<u8>> {
    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(input) {
        return Ok(bytes);
    }
    if let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(input) {
        return Ok(bytes);
    }
    Ok(base64::engine::general_purpose::URL_SAFE.decode(input)?)
}

pub(crate) fn random_nonce_hex(num_bytes: usize) -> String {
    let mut bytes = vec![0u8; num_bytes.max(1)];
    UnwrapErr(SysRng).fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

pub(crate) fn daemon_bootstrap_signing_input(
    user_id: &str,
    device_id: &str,
    ts: i64,
    nonce: &str,
) -> String {
    format!(
        "v1\nuser_id={}\ndevice_id={}\nts={}\nnonce={}",
        user_id, device_id, ts, nonce
    )
}

pub(crate) fn derive_daemon_bootstrap_signing_key(bot_token: &str) -> anyhow::Result<[u8; 32]> {
    let hk = Hkdf::<Sha256>::new(Some(DAEMON_BOOTSTRAP_SIGNING_SALT), bot_token.as_bytes());
    let mut key = [0u8; 32];
    hk.expand(DAEMON_BOOTSTRAP_SIGNING_INFO, &mut key)
        .map_err(|_| anyhow::anyhow!("failed to derive daemon bootstrap signing key"))?;
    Ok(key)
}

pub(crate) fn sign_daemon_bootstrap_proof_hkdf(
    bot_token: &str,
    user_id: &str,
    device_id: &str,
    ts: i64,
    nonce: &str,
) -> anyhow::Result<String> {
    let signing_key = derive_daemon_bootstrap_signing_key(bot_token)?;
    let input = daemon_bootstrap_signing_input(user_id, device_id, ts, nonce);
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(&signing_key)
        .map_err(|_| anyhow::anyhow!("invalid HMAC key"))?;
    mac.update(input.as_bytes());
    let signature = mac.finalize().into_bytes();
    Ok(b64url_encode(signature.as_ref()))
}

pub(crate) fn sign_daemon_bootstrap_proof_legacy(
    bot_token: &str,
    user_id: &str,
    device_id: &str,
    ts: i64,
    nonce: &str,
) -> anyhow::Result<String> {
    let input = daemon_bootstrap_signing_input(user_id, device_id, ts, nonce);
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(bot_token.as_bytes())
        .map_err(|_| anyhow::anyhow!("invalid HMAC key"))?;
    mac.update(input.as_bytes());
    let signature = mac.finalize().into_bytes();
    Ok(b64url_encode(signature.as_ref()))
}

pub(crate) fn daemon_bootstrap_signature_candidates(
    bot_token: &str,
    user_id: &str,
    device_id: &str,
    ts: i64,
    nonce: &str,
) -> anyhow::Result<Vec<String>> {
    let primary = sign_daemon_bootstrap_proof_hkdf(bot_token, user_id, device_id, ts, nonce)?;
    let legacy = sign_daemon_bootstrap_proof_legacy(bot_token, user_id, device_id, ts, nonce)?;
    if primary == legacy {
        Ok(vec![primary])
    } else {
        Ok(vec![primary, legacy])
    }
}

pub(crate) fn parse_daemon_secret_key_from_encoded(encoded: &str) -> anyhow::Result<SecretKey> {
    let bytes = b64_decode_flexible(encoded)?;
    SecretKey::from_slice(&bytes).map_err(Into::into)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

pub(crate) async fn load_or_create_key_material(
    secrets: &dyn SecretStore,
) -> anyhow::Result<KeyMaterial> {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum KeySource {
        Keychain,
        LegacyFile,
        Generated,
    }

    let dir = terminal_bridge_state_dir();
    tokio::fs::create_dir_all(&dir).await?;
    set_owner_only_permissions(&dir, 0o700)?;
    let key_path = dir.join("daemon-key.json");
    // Security model: this daemon key is local machine state (not synced over network).
    // We protect it with owner-only filesystem permissions (0700 dir, 0600 file).
    // If backups or host compromise are in scope, treat this file as sensitive.

    let mut source = KeySource::Generated;
    let secret = match secrets.get(TERMINAL_DAEMON_KEYCHAIN_FIELD) {
        Ok(encoded) => match parse_daemon_secret_key_from_encoded(encoded.expose()) {
            Ok(secret) => {
                source = KeySource::Keychain;
                secret
            }
            Err(err) => {
                warn!(
                    error = %err,
                    "Terminal bridge keychain entry is invalid; falling back to legacy file/new key"
                );
                if tokio::fs::try_exists(&key_path).await.unwrap_or(false) {
                    let raw = tokio::fs::read_to_string(&key_path).await?;
                    set_owner_only_permissions(&key_path, 0o600)?;
                    let parsed: StoredDaemonKey = serde_json::from_str(&raw)?;
                    let encoded = parsed
                        .private_key_b64
                        .or_else(|| parsed.private_jwk.map(|jwk| jwk.d))
                        .ok_or_else(|| {
                            anyhow::anyhow!("daemon key file missing private key material")
                        })?;
                    source = KeySource::LegacyFile;
                    parse_daemon_secret_key_from_encoded(&encoded)?
                } else {
                    SecretKey::generate_from_rng(&mut UnwrapErr(SysRng))
                }
            }
        },
        Err(err) => {
            debug!(
                error = %err,
                "Terminal bridge keychain unavailable or missing; using legacy file fallback"
            );
            if tokio::fs::try_exists(&key_path).await.unwrap_or(false) {
                let raw = tokio::fs::read_to_string(&key_path).await?;
                set_owner_only_permissions(&key_path, 0o600)?;
                let parsed: StoredDaemonKey = serde_json::from_str(&raw)?;
                let encoded = parsed
                    .private_key_b64
                    .or_else(|| parsed.private_jwk.map(|jwk| jwk.d))
                    .ok_or_else(|| {
                        anyhow::anyhow!("daemon key file missing private key material")
                    })?;
                source = KeySource::LegacyFile;
                parse_daemon_secret_key_from_encoded(&encoded)?
            } else {
                SecretKey::generate_from_rng(&mut UnwrapErr(SysRng))
            }
        }
    };

    let encoded_secret = b64_encode(secret.to_bytes().as_ref());
    let keychain_ok = match secrets.set(TERMINAL_DAEMON_KEYCHAIN_FIELD, &encoded_secret) {
        Ok(()) => true,
        Err(err) => {
            warn!(
                error = %err,
                "Failed to persist terminal bridge key to OS keychain; continuing with filesystem fallback"
            );
            false
        }
    };

    if keychain_ok {
        if source == KeySource::LegacyFile
            && tokio::fs::try_exists(&key_path).await.unwrap_or(false)
        {
            if let Err(err) = tokio::fs::remove_file(&key_path).await {
                warn!(
                    error = %err,
                    path = %key_path.display(),
                    "Failed to remove legacy terminal bridge key file after keychain migration"
                );
            } else {
                info!("Migrated terminal bridge key from legacy file to OS keychain");
            }
        }
    } else if source == KeySource::Generated {
        let stored = StoredDaemonKeyOnDisk {
            private_key_b64: encoded_secret,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        let serialized = serde_json::to_string_pretty(&stored)?;
        tokio::fs::write(&key_path, serialized).await?;
        set_owner_only_permissions(&key_path, 0o600)?;
    }

    let public = secret.public_key();
    let public_raw = public.to_sec1_point(false).as_bytes().to_vec();
    let fingerprint = sha256_hex(&public_raw);
    Ok(KeyMaterial {
        private_key: secret,
        public_raw,
        fingerprint,
    })
}
