//! Replacing a provider's API key from a chat, with no model involved: which
//! config entries take a key, a read-only check of a candidate key against
//! its provider, and saving it.

use std::path::Path;

use aidaemon_providers::OPENAI_DEFAULT_BASE_URL;
pub(crate) use aidaemon_providers::key_check::KeyCheck;
use aidaemon_providers::key_check::KeyEndpoint;

use crate::config::{ProviderConfig, ProviderKind};

/// One provider entry's place in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeySlot {
    Primary,
    Fallback(usize),
}

impl KeySlot {
    pub(crate) fn id(self) -> String {
        match self {
            Self::Primary => "p".to_string(),
            Self::Fallback(index) => format!("f{index}"),
        }
    }

    pub(crate) fn from_id(id: &str) -> Option<Self> {
        match id {
            "p" => Some(Self::Primary),
            _ => id.strip_prefix('f')?.parse().ok().map(Self::Fallback),
        }
    }

    /// The name `AppConfig` resolves a `"keychain"` value from.
    fn keychain_field(self) -> String {
        match self {
            Self::Primary => "api_key".to_string(),
            Self::Fallback(index) => format!("provider_fallback_{index}_api_key"),
        }
    }
}

/// A config entry that authenticates with an API key.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KeyTarget {
    pub slot: KeySlot,
    endpoint: KeyEndpoint,
}

impl KeyTarget {
    /// `None` for entries that log in some other way.
    fn new(slot: KeySlot, kind: ProviderKind, base_url: &str) -> Option<Self> {
        KeyEndpoint::new(kind, base_url).map(|endpoint| Self { slot, endpoint })
    }

    pub(crate) fn from_provider_config(slot: KeySlot, config: &ProviderConfig) -> Option<Self> {
        Self::new(slot, config.kind, &config.base_url)
    }

    pub(crate) fn provider_name(&self) -> String {
        self.endpoint.provider_name()
    }

    pub(crate) fn label(&self) -> String {
        match self.slot {
            KeySlot::Primary => format!("Primary: {}", self.provider_name()),
            KeySlot::Fallback(index) => format!("Fallback {}: {}", index + 1, self.provider_name()),
        }
    }
}

/// Entries of the config text that take an API key. Reads the raw file: a
/// missing secret, which `AppConfig::load` refuses, is a reason to be here.
pub(crate) fn key_targets(config_text: &str) -> anyhow::Result<Vec<KeyTarget>> {
    let doc: toml::Table = config_text.parse()?;
    let Some(provider) = doc.get("provider").and_then(toml::Value::as_table) else {
        return Ok(Vec::new());
    };
    let entry = |slot: KeySlot, table: &toml::Table| {
        let kind = table
            .get("kind")
            .cloned()
            .map_or(Ok(ProviderKind::default()), toml::Value::try_into)
            .ok()?;
        let base_url = table
            .get("base_url")
            .and_then(toml::Value::as_str)
            .unwrap_or(OPENAI_DEFAULT_BASE_URL);
        KeyTarget::new(slot, kind, base_url)
    };
    let mut targets: Vec<KeyTarget> = entry(KeySlot::Primary, provider).into_iter().collect();
    let fallbacks = provider
        .get("fallbacks")
        .or_else(|| provider.get("failover"))
        .and_then(toml::Value::as_array);
    for (index, fallback) in fallbacks.into_iter().flatten().enumerate() {
        if let Some(table) = fallback.as_table() {
            targets.extend(entry(KeySlot::Fallback(index), table));
        }
    }
    Ok(targets)
}

/// Ask the provider whether it accepts `key`, with one read-only request.
pub(crate) async fn check_key(target: &KeyTarget, key: &str) -> KeyCheck {
    aidaemon_providers::key_check::check_key(&target.endpoint, key).await
}

/// One line per configured key its provider rejects. A provider that gives
/// no verdict is left out: this only reports keys known to be bad.
pub(crate) async fn rejected_keys(provider: &ProviderConfig) -> Vec<String> {
    let entries = std::iter::once((KeySlot::Primary, provider)).chain(
        provider
            .fallbacks
            .iter()
            .enumerate()
            .map(|(index, fallback)| (KeySlot::Fallback(index), fallback)),
    );
    let mut rejected = Vec::new();
    for (slot, config) in entries {
        let key = config.api_key.trim();
        if key.is_empty() || key.starts_with("${") {
            continue;
        }
        let Some(target) = KeyTarget::from_provider_config(slot, config) else {
            continue;
        };
        if let KeyCheck::Rejected(detail) = check_key(&target, key).await {
            rejected.push(format!("{}: key rejected ({detail})", target.label()));
        }
    }
    rejected
}

/// The config text with `slot`'s `api_key` set to `value`; only that value
/// changes, so comments and layout survive.
fn with_api_key_value(content: &str, slot: KeySlot, value: &str) -> anyhow::Result<String> {
    let mut doc: toml_edit::DocumentMut = content.parse()?;
    let provider = doc
        .get_mut("provider")
        .and_then(toml_edit::Item::as_table_like_mut)
        .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
    let table = match slot {
        KeySlot::Primary => provider,
        KeySlot::Fallback(index) => {
            let name = if provider.contains_key("fallbacks") {
                "fallbacks"
            } else {
                "failover"
            };
            provider
                .get_mut(name)
                .and_then(toml_edit::Item::as_array_of_tables_mut)
                .and_then(|entries| entries.get_mut(index))
                .map(|entry| entry as &mut dyn toml_edit::TableLike)
                .ok_or_else(|| anyhow::anyhow!("fallback {index} is not in the config"))?
        }
    };
    table.insert("api_key", toml_edit::value(value));
    Ok(doc.to_string())
}

/// Where a saved key went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyStored {
    /// The daemon's secret store (the OS keychain, or its stand-in).
    SecretStore,
    /// Inline in the config file, readable only by the owner's account.
    ConfigFile,
}

/// The reply once a key was tested and saved: the provider, only what the
/// reload changed, and where the key is only when that is not the usual
/// secret store.
pub(crate) fn key_saved_reply(provider: &str, stored: KeyStored, reload: &str) -> String {
    let mut reply = format!("{provider} key saved and tested. {reload}");
    if stored == KeyStored::ConfigFile {
        reply.push_str(
            " Your keychain wasn't available, so the key is in the config file, which only \
             your account can read.",
        );
    }
    reply
}

/// Store `key` for `slot`: in the OS keychain, or inline with the config's
/// owner-only permissions when the keychain is unavailable. Returns where it
/// went.
pub(crate) async fn save_key(
    config_path: &Path,
    slot: KeySlot,
    key: &str,
) -> anyhow::Result<KeyStored> {
    let current = tokio::fs::read_to_string(config_path).await?;
    let (stored, note) = match crate::config::store_in_keychain(&slot.keychain_field(), key) {
        Ok(()) => ("keychain", KeyStored::SecretStore),
        Err(error) => {
            tracing::warn!(%error, "Keychain unavailable; saving the API key in the config");
            (key, KeyStored::ConfigFile)
        }
    };
    let updated = with_api_key_value(&current, slot, stored)?;
    if updated != current {
        crate::config::config_file(config_path)
            .save(&updated)
            .await?;
    }
    Ok(note)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    const CONFIG: &str = r#"
# Primary comment
[provider]
kind = "openai_chatgpt"
api_key = "subscription"

# Anthropic first
[[provider.fallbacks]]
kind = "anthropic"
api_key = "old-value" # trailing

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "https://openrouter.ai/api/v1"
api_key = "other-value"
"#;

    /// Answers each request by the key it carried.
    fn server(answer: fn(&str) -> (u16, String)) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut buffer = [0u8; 4096];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                let (status, body) = answer(&request);
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        base
    }

    #[test]
    fn subscription_logins_are_not_offered() {
        let targets = key_targets(CONFIG).unwrap();
        assert_eq!(
            targets.iter().map(KeyTarget::label).collect::<Vec<_>>(),
            ["Fallback 1: Anthropic", "Fallback 2: OpenRouter"]
        );
        assert_eq!(targets[1].slot, KeySlot::Fallback(1));
    }

    #[test]
    fn slot_ids_round_trip() {
        for slot in [KeySlot::Primary, KeySlot::Fallback(3)] {
            assert_eq!(KeySlot::from_id(&slot.id()), Some(slot));
        }
        assert_eq!(KeySlot::from_id("x"), None);
    }

    #[test]
    fn only_the_chosen_key_changes() {
        let updated = with_api_key_value(CONFIG, KeySlot::Fallback(0), "keychain").unwrap();
        assert!(updated.contains("# Anthropic first"));
        assert!(updated.contains("# Primary comment"));
        assert!(updated.contains("api_key = \"other-value\""));
        assert!(updated.contains("api_key = \"subscription\""));
        assert!(!updated.contains("old-value"));
        assert!(with_api_key_value(CONFIG, KeySlot::Fallback(9), "x").is_err());
    }

    #[tokio::test]
    async fn only_keys_the_provider_refuses_are_reported() {
        let refusing = server(|_| (401, "{}".to_string()));
        let down = server(|_| (503, "{}".to_string()));
        let entry = |kind, base_url: &str, key: &str| ProviderConfig {
            kind,
            base_url: base_url.to_string(),
            api_key: key.to_string(),
            ..toml::from_str("api_key = ''").unwrap()
        };
        let mut provider = entry(ProviderKind::OpenaiChatgpt, "", "subscription");
        provider.fallbacks = vec![
            entry(ProviderKind::Anthropic, &refusing, "synthetic-key"),
            entry(ProviderKind::Anthropic, &down, "synthetic-key"),
            entry(ProviderKind::Anthropic, &refusing, ""),
        ];
        assert_eq!(
            rejected_keys(&provider).await,
            ["Fallback 1: Anthropic: key rejected (HTTP 401)"]
        );
    }

    #[test]
    fn a_saved_key_reply_says_only_what_matters() {
        assert_eq!(
            key_saved_reply("Google", KeyStored::SecretStore, "Nothing else changed."),
            "Google key saved and tested. Nothing else changed."
        );
        let inline = key_saved_reply("Anthropic", KeyStored::ConfigFile, "Nothing else changed.");
        assert!(inline.contains("only your account can read"), "{inline}");
        assert!(
            !inline.contains(".env") && !inline.contains("Provider:"),
            "{inline}"
        );
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn saving_stores_the_secret_and_references_it() {
        let _keychain = crate::testing::KeychainIsolation::acquire();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let config = format!("{CONFIG}\n");
        std::fs::write(&path, config).unwrap();
        save_key(&path, KeySlot::Fallback(0), "synthetic-new-key")
            .await
            .unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.contains("api_key = \"keychain\""), "{saved}");
        assert!(!saved.contains("synthetic-new-key"));
        assert!(saved.contains("# Anthropic first"));
        assert_eq!(
            crate::config::resolve_from_keychain("provider_fallback_0_api_key").unwrap(),
            "synthetic-new-key"
        );
    }
}
