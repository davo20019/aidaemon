//! The model and effort a conversation is set to answer with, chosen in a
//! client's picker and kept per session in the settings table.
//!
//! Absent (or unreadable): the daemon picks as it does for every chat. A
//! pin names a provider of the live runtime (`main`, `failover-N`) and one
//! of its configured or catalog-offered models; a pin the runtime no longer has reads as
//! automatic, so a config change never strands a conversation.

use aidaemon_providers::routing::ProviderKind;
use aidaemon_tools::services::SettingsStore;
use serde::{Deserialize, Serialize};

use crate::llm::LlmRuntimeSnapshot;

/// The provider id of the runtime's main provider.
pub const MAIN_PROVIDER: &str = "main";

/// The efforts the daemon sends, lowest first.
pub const EFFORTS: [&str; 3] = ["low", "medium", "high"];

fn key(session_id: &str) -> String {
    format!("session_model:{session_id}")
}

/// A model pinned for a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinnedModel {
    pub provider: String,
    pub model: String,
}

/// What a conversation is set to. `None` fields: automatic.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionModelSetting {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned: Option<PinnedModel>,
    /// One of [`EFFORTS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

impl SessionModelSetting {
    pub fn is_auto(&self) -> bool {
        self.pinned.is_none() && self.effort.is_none()
    }
}

/// The setting of `session_id`; automatic when none is stored or it cannot
/// be read (the conversation then answers as every chat does).
pub async fn get(
    settings: &(impl SettingsStore + ?Sized),
    session_id: &str,
) -> SessionModelSetting {
    match settings.get_setting(&key(session_id)).await {
        Ok(Some(raw)) if !raw.is_empty() => serde_json::from_str(&raw).unwrap_or_else(|error| {
            tracing::warn!(session_id, %error, "Unreadable session model setting; using automatic");
            SessionModelSetting::default()
        }),
        Ok(_) => SessionModelSetting::default(),
        Err(error) => {
            tracing::warn!(session_id, %error, "Could not read the session model setting; using automatic");
            SessionModelSetting::default()
        }
    }
}

/// Store `setting` for `session_id` (an automatic one clears the entry).
pub async fn set(
    settings: &(impl SettingsStore + ?Sized),
    session_id: &str,
    setting: &SessionModelSetting,
) -> anyhow::Result<()> {
    let value = if setting.is_auto() {
        String::new()
    } else {
        serde_json::to_string(setting)?
    };
    settings.set_setting(&key(session_id), &value).await
}

/// The efforts a provider kind's adapter sends; empty when it sends none.
pub fn efforts_for(kind: ProviderKind) -> &'static [&'static str] {
    match kind {
        ProviderKind::OpenaiCompatible | ProviderKind::Anthropic | ProviderKind::OpenaiChatgpt => {
            &EFFORTS
        }
        ProviderKind::GoogleGenai | ProviderKind::XaiNative => &[],
    }
}

/// A provider kind in plain words, as a picker shows it.
pub fn provider_name(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::OpenaiCompatible => "OpenAI-compatible",
        ProviderKind::XaiNative => "xAI",
        ProviderKind::GoogleGenai => "Google Gemini",
        ProviderKind::Anthropic => "Anthropic",
        ProviderKind::OpenaiChatgpt => "ChatGPT",
    }
}

/// A provider a conversation can pin, as the runtime has it now.
#[derive(Clone)]
pub struct PickableProvider {
    pub id: String,
    pub kind: ProviderKind,
    /// Its configured models followed by catalog offers, each listed under
    /// the first provider that has it (the one a request for it goes to).
    pub models: Vec<String>,
}

/// The providers of `snapshot`, the main one first, then each failover.
pub fn pickable(snapshot: &LlmRuntimeSnapshot) -> Vec<PickableProvider> {
    let main_models = snapshot
        .router()
        .map(|router| router.all_models_ordered())
        .filter(|models| !models.is_empty())
        .unwrap_or_else(|| vec![snapshot.primary_model()]);
    let mut main_models = main_models;
    main_models.extend(snapshot.offered_models().iter().cloned());
    let mut out = vec![PickableProvider {
        id: MAIN_PROVIDER.to_string(),
        kind: snapshot.provider_kind(),
        models: main_models,
    }];
    for (index, target) in snapshot.failover_targets().into_iter().enumerate() {
        out.push(PickableProvider {
            id: format!("failover-{index}"),
            kind: target.provider_kind(),
            models: target
                .all_models_ordered()
                .into_iter()
                .chain(target.offered_models().iter().cloned())
                .collect(),
        });
    }
    let mut seen: Vec<String> = Vec::new();
    for provider in &mut out {
        provider.models.retain(|model| {
            let model = model.trim();
            if model.is_empty() || seen.iter().any(|known| known == model) {
                return false;
            }
            seen.push(model.to_string());
            true
        });
    }
    out
}

/// The provider `pin` resolves to in `snapshot`, if it still has the model.
pub fn resolve(snapshot: &LlmRuntimeSnapshot, pin: &PinnedModel) -> Option<PickableProvider> {
    pickable(snapshot)
        .into_iter()
        .find(|provider| provider.id == pin.provider && provider.models.contains(&pin.model))
}

/// Whether `effort` is one the pinned (or main) provider takes.
pub fn effort_valid(kind: ProviderKind, effort: &str) -> bool {
    efforts_for(kind).contains(&effort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemorySettings(Mutex<HashMap<String, String>>);

    #[async_trait::async_trait]
    impl SettingsStore for MemorySettings {
        async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
            self.0.lock().unwrap().insert(key.into(), value.into());
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_setting_round_trips_and_automatic_clears_it() {
        let settings = MemorySettings::default();
        assert!(get(&settings, "app:s1").await.is_auto());
        let pinned = SessionModelSetting {
            pinned: Some(PinnedModel {
                provider: MAIN_PROVIDER.into(),
                model: "synthetic-model-large".into(),
            }),
            effort: Some("high".into()),
        };
        set(&settings, "app:s1", &pinned).await.unwrap();
        assert_eq!(get(&settings, "app:s1").await, pinned);
        assert!(get(&settings, "app:s2").await.is_auto());
        set(&settings, "app:s1", &SessionModelSetting::default())
            .await
            .unwrap();
        assert!(get(&settings, "app:s1").await.is_auto());
    }

    #[tokio::test]
    async fn an_unreadable_setting_reads_as_automatic() {
        let settings = MemorySettings::default();
        settings
            .set_setting(&key("app:s1"), "{not json")
            .await
            .unwrap();
        assert!(get(&settings, "app:s1").await.is_auto());
    }

    #[test]
    fn only_adapters_that_send_effort_offer_it() {
        assert_eq!(efforts_for(ProviderKind::Anthropic), &EFFORTS);
        assert_eq!(efforts_for(ProviderKind::OpenaiChatgpt), &EFFORTS);
        assert!(efforts_for(ProviderKind::GoogleGenai).is_empty());
        assert!(efforts_for(ProviderKind::XaiNative).is_empty());
    }
}
