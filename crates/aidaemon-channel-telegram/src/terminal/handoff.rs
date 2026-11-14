//! One-time handoff codes and relay-session bindings between the Telegram
//! `/agent` command and the terminal bridge's local `attach`/`share`
//! commands, kept in the host's settings store.

use aidaemon_tools::services::SettingsStore;
use chrono::Utc;
use rand::{RngExt, rng as thread_rng};
use serde::{Deserialize, Serialize};

const HANDOFF_KEY_PREFIX: &str = "terminal_attach_handoff:";
const LAST_ACTIVE_RELAY_SESSION_KEY_PREFIX: &str = "terminal_last_active_relay_session:";
const HANDOFF_CODE_LEN: usize = 12;
const HANDOFF_TTL_SECS: i64 = 5 * 60;
const HANDOFF_ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const TELEGRAM_RELAY_BINDING_KEY_PREFIX: &str = "terminal_attach_telegram_session:";
const TELEGRAM_RELAY_BINDING_MAX_AGE_SECS: i64 = 6 * 60 * 60;

/// A one-time code that attaches a computer to a relay session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalAttachHandoff {
    pub code: String,
    pub relay_session_id: String,
    pub owner_user_id: u64,
    pub created_at_unix: i64,
    pub expires_at_unix: i64,
    pub consumed: bool,
    #[serde(default)]
    pub consumed_at_unix: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TelegramRelayBinding {
    relay_session_id: String,
    updated_at_unix: i64,
}

fn handoff_key(code: &str) -> String {
    format!("{}{}", HANDOFF_KEY_PREFIX, code)
}

fn last_active_relay_session_key(owner_user_id: u64) -> String {
    format!("{}{}", LAST_ACTIVE_RELAY_SESSION_KEY_PREFIX, owner_user_id)
}

fn telegram_relay_binding_key(raw_session_id: &str) -> Option<String> {
    let session_id = raw_session_id.trim();
    if session_id.is_empty() || session_id.len() > 128 {
        return None;
    }
    if !session_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_' | '.'))
    {
        return None;
    }
    Some(format!(
        "{}{}",
        TELEGRAM_RELAY_BINDING_KEY_PREFIX, session_id
    ))
}

fn normalize_code(raw: &str) -> Option<String> {
    let code = raw.trim().to_ascii_uppercase();
    if code.len() < 6 || code.len() > 64 {
        return None;
    }
    if !code.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(code)
}

fn generate_code() -> String {
    let mut rng = thread_rng();
    (0..HANDOFF_CODE_LEN)
        .map(|_| {
            let idx = rng.random_range(0..HANDOFF_ALPHABET.len());
            HANDOFF_ALPHABET[idx] as char
        })
        .collect()
}

fn parse_entry(raw: &str) -> anyhow::Result<TerminalAttachHandoff> {
    serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("invalid handoff payload: {}", e))
}

pub async fn create_handoff_code(
    state: &dyn SettingsStore,
    relay_session_id: &str,
    owner_user_id: u64,
) -> anyhow::Result<TerminalAttachHandoff> {
    let relay_session_id = relay_session_id.trim();
    if relay_session_id.is_empty() {
        anyhow::bail!("relay session id is required");
    }

    for _ in 0..8 {
        let code = generate_code();
        let key = handoff_key(&code);
        if state.get_setting(&key).await?.is_some() {
            continue;
        }
        let now = Utc::now().timestamp();
        let entry = TerminalAttachHandoff {
            code: code.clone(),
            relay_session_id: relay_session_id.to_string(),
            owner_user_id,
            created_at_unix: now,
            expires_at_unix: now + HANDOFF_TTL_SECS,
            consumed: false,
            consumed_at_unix: None,
        };
        state
            .set_setting(&key, &serde_json::to_string(&entry)?)
            .await?;
        return Ok(entry);
    }

    anyhow::bail!("failed to allocate unique handoff code");
}

pub async fn resolve_handoff_code(
    state: &dyn SettingsStore,
    raw_code: &str,
) -> anyhow::Result<TerminalAttachHandoff> {
    let code = normalize_code(raw_code).ok_or_else(|| anyhow::anyhow!("invalid handoff code"))?;
    let key = handoff_key(&code);
    let Some(raw) = state.get_setting(&key).await? else {
        anyhow::bail!("handoff code not found");
    };

    let entry = parse_entry(&raw)?;
    let now = Utc::now().timestamp();
    if entry.consumed {
        anyhow::bail!("handoff code already used");
    }
    if now > entry.expires_at_unix {
        anyhow::bail!("handoff code expired");
    }
    Ok(entry)
}

pub async fn consume_handoff_code(
    state: &dyn SettingsStore,
    raw_code: &str,
) -> anyhow::Result<TerminalAttachHandoff> {
    let code = normalize_code(raw_code).ok_or_else(|| anyhow::anyhow!("invalid handoff code"))?;
    let key = handoff_key(&code);
    let Some(raw) = state.get_setting(&key).await? else {
        anyhow::bail!("handoff code not found");
    };

    let mut entry = parse_entry(&raw)?;
    let now = Utc::now().timestamp();
    if entry.consumed {
        anyhow::bail!("handoff code already used");
    }
    if now > entry.expires_at_unix {
        anyhow::bail!("handoff code expired");
    }

    entry.consumed = true;
    entry.consumed_at_unix = Some(now);
    state
        .set_setting(&key, &serde_json::to_string(&entry)?)
        .await?;
    Ok(entry)
}

pub async fn set_last_active_relay_session_id(
    state: &dyn SettingsStore,
    owner_user_id: u64,
    relay_session_id: &str,
) -> anyhow::Result<()> {
    let relay_session_id = relay_session_id.trim();
    if relay_session_id.is_empty() {
        anyhow::bail!("relay session id is required");
    }
    let key = last_active_relay_session_key(owner_user_id);
    state.set_setting(&key, relay_session_id).await
}

pub async fn get_last_active_relay_session_id(
    state: &dyn SettingsStore,
    owner_user_id: u64,
) -> anyhow::Result<Option<String>> {
    let key = last_active_relay_session_key(owner_user_id);
    let value = state.get_setting(&key).await?;
    Ok(value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty()))
}

pub async fn bind_telegram_session_to_relay(
    state: &dyn SettingsStore,
    raw_telegram_session_id: &str,
    relay_session_id: &str,
) -> anyhow::Result<()> {
    let Some(key) = telegram_relay_binding_key(raw_telegram_session_id) else {
        anyhow::bail!("invalid telegram session id");
    };
    let relay_session_id = relay_session_id.trim();
    if relay_session_id.is_empty() {
        anyhow::bail!("relay session id is required");
    }
    let entry = TelegramRelayBinding {
        relay_session_id: relay_session_id.to_string(),
        updated_at_unix: Utc::now().timestamp(),
    };
    state
        .set_setting(&key, &serde_json::to_string(&entry)?)
        .await?;
    Ok(())
}

pub async fn resolve_relay_for_telegram_session(
    state: &dyn SettingsStore,
    raw_telegram_session_id: &str,
) -> anyhow::Result<Option<String>> {
    let Some(key) = telegram_relay_binding_key(raw_telegram_session_id) else {
        return Ok(None);
    };
    let Some(raw) = state.get_setting(&key).await? else {
        return Ok(None);
    };
    if raw.trim().is_empty() {
        return Ok(None);
    }

    if let Ok(entry) = serde_json::from_str::<TelegramRelayBinding>(&raw) {
        let now = Utc::now().timestamp();
        if now.saturating_sub(entry.updated_at_unix) > TELEGRAM_RELAY_BINDING_MAX_AGE_SECS {
            return Ok(None);
        }
        let relay = entry.relay_session_id.trim();
        if relay.is_empty() {
            return Ok(None);
        }
        return Ok(Some(relay.to_string()));
    }

    // Backward compatibility: treat a raw stored relay session id as valid.
    let relay = raw.trim();
    if relay.is_empty() {
        return Ok(None);
    }
    Ok(Some(relay.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemSettings(Mutex<HashMap<String, String>>);

    #[async_trait::async_trait]
    impl SettingsStore for MemSettings {
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

    #[tokio::test]
    async fn handoff_create_and_consume_once() {
        let state = MemSettings::default();

        let created = create_handoff_code(&state, "relay_abc", 1234)
            .await
            .expect("create");
        assert_eq!(created.relay_session_id, "relay_abc");
        assert!(!created.code.is_empty());

        let resolved = resolve_handoff_code(&state, &created.code)
            .await
            .expect("resolve");
        assert_eq!(resolved.code, created.code);
        assert!(!resolved.consumed);

        let consumed = consume_handoff_code(&state, &created.code)
            .await
            .expect("consume");
        assert!(consumed.consumed);
        assert!(consume_handoff_code(&state, &created.code).await.is_err());
    }

    #[tokio::test]
    async fn handoff_rejects_invalid_code_format() {
        let state = MemSettings::default();
        assert!(resolve_handoff_code(&state, "bad-code!!").await.is_err());
    }

    #[tokio::test]
    async fn telegram_relay_binding_round_trip() {
        let state = MemSettings::default();
        bind_telegram_session_to_relay(&state, "telegrambot:12345", "relay_abc")
            .await
            .expect("bind");
        let resolved = resolve_relay_for_telegram_session(&state, "telegrambot:12345")
            .await
            .expect("resolve");
        assert_eq!(resolved.as_deref(), Some("relay_abc"));
    }

    #[tokio::test]
    async fn telegram_relay_binding_ignores_stale_entries() {
        let state = MemSettings::default();
        let key = format!(
            "{}{}",
            TELEGRAM_RELAY_BINDING_KEY_PREFIX, "telegrambot:12345"
        );
        let stale = TelegramRelayBinding {
            relay_session_id: "relay_old".to_string(),
            updated_at_unix: Utc::now().timestamp() - TELEGRAM_RELAY_BINDING_MAX_AGE_SECS - 60,
        };
        state
            .set_setting(&key, &serde_json::to_string(&stale).expect("json"))
            .await
            .expect("set");

        let resolved = resolve_relay_for_telegram_session(&state, "telegrambot:12345")
            .await
            .expect("resolve");
        assert!(resolved.is_none());
    }
}
