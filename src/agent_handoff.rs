use chrono::Utc;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

use crate::traits::StateStore;

const HANDOFF_KEY_PREFIX: &str = "terminal_attach_handoff:";
const LAST_ACTIVE_RELAY_SESSION_KEY_PREFIX: &str = "terminal_last_active_relay_session:";
const HANDOFF_CODE_LEN: usize = 12;
const HANDOFF_TTL_SECS: i64 = 5 * 60;
const HANDOFF_ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const TELEGRAM_RELAY_BINDING_KEY_PREFIX: &str = "terminal_attach_telegram_session:";
const TELEGRAM_RELAY_BINDING_MAX_AGE_SECS: i64 = 6 * 60 * 60;

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
            let idx = rng.gen_range(0..HANDOFF_ALPHABET.len());
            HANDOFF_ALPHABET[idx] as char
        })
        .collect()
}

fn parse_entry(raw: &str) -> anyhow::Result<TerminalAttachHandoff> {
    serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("invalid handoff payload: {}", e))
}

pub async fn create_handoff_code(
    state: &dyn StateStore,
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
    state: &dyn StateStore,
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
    state: &dyn StateStore,
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
    state: &dyn StateStore,
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
    state: &dyn StateStore,
    owner_user_id: u64,
) -> anyhow::Result<Option<String>> {
    let key = last_active_relay_session_key(owner_user_id);
    let value = state.get_setting(&key).await?;
    Ok(value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty()))
}

pub async fn bind_telegram_session_to_relay(
    state: &dyn StateStore,
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
    state: &dyn StateStore,
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
    use std::sync::Arc;

    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::SettingsStore;
    use tempfile::tempdir;

    async fn make_test_state(name: &str) -> (tempfile::TempDir, SqliteStateStore) {
        let dir = tempdir().expect("tempdir");
        let db = dir.path().join(name);
        let embeddings = Arc::new(EmbeddingService::new().expect("embeddings"));
        let state = SqliteStateStore::new(db.to_str().expect("utf8 path"), 100, None, embeddings)
            .await
            .expect("state");
        (dir, state)
    }

    #[tokio::test]
    async fn handoff_create_and_consume_once() {
        let (_dir, state) = make_test_state("test.db").await;

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
        let (_dir, state) = make_test_state("test2.db").await;
        assert!(resolve_handoff_code(&state, "bad-code!!").await.is_err());
    }

    #[tokio::test]
    async fn telegram_relay_binding_round_trip() {
        let (_dir, state) = make_test_state("test3.db").await;
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
        let (_dir, state) = make_test_state("test4.db").await;
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
