//! Keeping the owner on the newest model of the line they use.
//!
//! Providers release new versions every few weeks, and an owner who does
//! not follow launches keeps running an old one. Once a day this job reads
//! the main provider's model listing and, for each configured model with a
//! newer version of its line ([`aidaemon_providers::model_upgrade`]):
//!
//! - skips it when the harness would treat the new model worse (another
//!   trust tier, no vision where the old one had it, a smaller known
//!   window), or when the owner went back from it before;
//! - on a subscription (the ChatGPT sign-in, no per-call price) proves it
//!   with one real tool call, writes it to the config with the old model
//!   first among the fallbacks (a failing new model falls back to the old
//!   one on every call), reloads the provider and tells the owner;
//! - on a pay-per-token provider only tells the owner once: a newer model
//!   can cost more, and the daemon does not know prices.
//!
//! `provider.auto_upgrade = false` turns it off; a `/model` pin postpones
//! it. An owner who puts the old model back (in Settings or by asking)
//! keeps it: the switch is recorded, and a default that went back to what
//! it replaced declines that model for good.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::{Message, ModelRequest, RequestOptions, ToolChoice, ToolSpec};
use aidaemon_providers::model_upgrade::successor;
use aidaemon_providers::routing::ProviderKind;
use aidaemon_runtime::policy::trust_tier::{TrustTierSetting, classify_model_trust};
use aidaemon_runtime::settings::{VisionConfig, known_context_budget};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn};

use crate::channels::ChannelHub;
use crate::config::AppConfig;
use crate::runtime_ports::DaemonAgentRuntime;

/// The first check waits for channels to connect.
const FIRST_CHECK_AFTER: Duration = Duration::from_secs(120);
const CHECK_EVERY: Duration = Duration::from_secs(24 * 3600);
const TRIAL_WITHIN: Duration = Duration::from_secs(90);

/// What the job remembers between checks, beside the config file.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct UpgradeState {
    /// Models the owner went back from: never offered again.
    #[serde(default)]
    declined: Vec<String>,
    /// The last switch, to notice the owner undoing it.
    #[serde(default)]
    last: Option<Switch>,
    /// Models already suggested on a pay-per-token provider.
    #[serde(default)]
    suggested: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Switch {
    from: String,
    to: String,
    at_ms: u64,
}

fn state_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("model-upgrades.json")
}

fn load_state(config_path: &Path) -> UpgradeState {
    std::fs::read_to_string(state_path(config_path))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save_state(config_path: &Path, state: &UpgradeState) {
    let path = state_path(config_path);
    match serde_json::to_string_pretty(state) {
        Ok(text) => {
            if let Err(error) = std::fs::write(&path, text) {
                warn!(%error, "Could not save the model upgrade state");
            }
        }
        Err(error) => warn!(%error, "Could not encode the model upgrade state"),
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A model the provider offers, as the job reads it.
#[derive(Debug, Clone, PartialEq)]
struct Offered {
    id: String,
    display_name: Option<String>,
    description: Option<String>,
}

/// The `[provider.models]` entries as written in the file: only these are
/// moved (a model the file leaves to the catalog follows the release).
#[derive(Debug, Default, Clone, PartialEq)]
struct WrittenModels {
    default: Option<String>,
    fast: Option<String>,
    smart: Option<String>,
    fallback: Vec<String>,
}

fn written_models(doc: &toml_edit::DocumentMut) -> WrittenModels {
    let table = doc
        .get("provider")
        .and_then(|p| p.get("models"))
        .and_then(toml_edit::Item::as_table_like);
    let text = |key: &str| {
        table
            .and_then(|t| t.get(key))
            .and_then(toml_edit::Item::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    WrittenModels {
        default: text("default").or_else(|| text("primary")),
        fast: text("fast"),
        smart: text("smart"),
        fallback: table
            .and_then(|t| t.get("fallback"))
            .and_then(toml_edit::Item::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Whether the harness treats `new` at least as well as `old`: the same
/// trust tier, vision kept, and a known window no smaller. A model of a
/// family the harness has not met fails this until it is taught.
fn harness_keeps_up(old: &str, new: &str, vision: &VisionConfig) -> bool {
    let tier = |m| classify_model_trust(m, TrustTierSetting::Auto);
    if tier(old) != tier(new) {
        return false;
    }
    if vision.model_supports_vision(old) && !vision.model_supports_vision(new) {
        return false;
    }
    match (known_context_budget(old), known_context_budget(new)) {
        (Some(o), Some(n)) => n >= o,
        (Some(_), None) => false,
        (None, _) => true,
    }
}

/// For each written model with an allowed newer version, the model it moves to.
fn plan(
    written: &WrittenModels,
    offered: &[Offered],
    declined: &[String],
    vision: &VisionConfig,
) -> BTreeMap<String, String> {
    let candidates: Vec<&str> = offered
        .iter()
        .map(|m| m.id.as_str())
        .filter(|id| !declined.iter().any(|d| d == id))
        .collect();
    let mut moves = BTreeMap::new();
    let current = written
        .default
        .iter()
        .chain(&written.fast)
        .chain(&written.smart)
        .chain(&written.fallback);
    for model in current {
        if moves.contains_key(model) {
            continue;
        }
        if let Some(next) = successor(model, candidates.iter().copied())
            && harness_keeps_up(model, next, vision)
        {
            moves.insert(model.clone(), next.to_string());
        }
    }
    moves
}

/// The config text with `moves` applied, and the old default first among
/// the fallbacks.
fn apply(text: &str, moves: &BTreeMap<String, String>) -> anyhow::Result<String> {
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    let written = written_models(&doc);
    let models = doc["provider"]
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("no [provider] table"))?
        .entry("models")
        .or_insert_with(toml_edit::table)
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("provider.models is not a table"))?;
    let moved = |model: &str| {
        moves
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string())
    };
    for key in ["default", "primary", "fast", "smart"] {
        if let Some(value) = models.get(key).and_then(toml_edit::Item::as_str) {
            let next = moved(value);
            if next != value {
                models[key] = toml_edit::value(next);
            }
        }
    }
    if let Some(old_default) = &written.default
        && moves.contains_key(old_default)
    {
        let new_default = moved(old_default);
        let mut fallback = vec![old_default.clone()];
        for model in &written.fallback {
            let model = moved(model);
            if model != new_default && !fallback.contains(&model) {
                fallback.push(model);
            }
        }
        let mut array = toml_edit::Array::new();
        for model in fallback {
            array.push(model);
        }
        models["fallback"] = toml_edit::value(array);
    } else if !written.fallback.is_empty() {
        let mut array = toml_edit::Array::new();
        for model in &written.fallback {
            array.push(moved(model));
        }
        models["fallback"] = toml_edit::value(array);
    }
    Ok(doc.to_string())
}

/// When the owner put back what the last switch replaced, that model is
/// declined and the switch forgotten. `true` when the state changed.
fn note_undo(state: &mut UpgradeState, written: &WrittenModels) -> bool {
    let Some(last) = &state.last else {
        return false;
    };
    match written.default.as_deref() {
        Some(default) if default == last.from => {
            info!(model = %last.to, "The owner went back from an upgraded model; it is not offered again");
            if !state.declined.contains(&last.to) {
                state.declined.push(last.to.clone());
            }
            state.last = None;
            true
        }
        Some(default) if default != last.to => {
            // The owner chose some other model: the switch is history.
            state.last = None;
            true
        }
        _ => false,
    }
}

fn name_of(id: &str, offered: &[Offered]) -> String {
    offered
        .iter()
        .find(|m| m.id == id)
        .and_then(|m| m.display_name.clone())
        .unwrap_or_else(|| id.to_string())
}

fn switched_message(
    moves: &BTreeMap<String, String>,
    default: Option<&str>,
    offered: &[Offered],
    kind: ProviderKind,
) -> String {
    let maker = match kind {
        ProviderKind::OpenaiChatgpt | ProviderKind::OpenaiCompatible => "OpenAI",
        ProviderKind::Anthropic => "Anthropic",
        ProviderKind::GoogleGenai => "Google",
        ProviderKind::XaiNative => "xAI",
    };
    let mut lines = Vec::new();
    if let Some((from, to)) = default.and_then(|d| moves.get_key_value(d)) {
        lines.push(format!(
            "{maker} released a newer model, so I've switched to it: {} (it replaces {}).",
            name_of(to, offered),
            name_of(from, offered)
        ));
        if let Some(about) = offered
            .iter()
            .find(|m| &m.id == to)
            .and_then(|m| m.description.as_deref())
        {
            lines.push(about.to_string());
        }
    }
    for (from, to) in moves {
        if Some(from.as_str()) != default {
            lines.push(format!(
                "{} also moves to {}.",
                name_of(from, offered),
                name_of(to, offered)
            ));
        }
    }
    lines.push(
        "Nothing for you to do. If something seems off, the previous model takes over \
         automatically. To keep the previous one, tell me to switch back or change the \
         model in Settings, and I won't move you to this one again. Settings > \
         provider.auto_upgrade turns these switches off."
            .to_string(),
    );
    lines.join("\n\n")
}

fn suggested_message(from: &str, to: &str, offered: &[Offered]) -> String {
    let mut text = format!(
        "A newer version of the model I use is out: {} (I use {}). I haven't switched, \
         because it may cost more per message on your account.",
        name_of(to, offered),
        name_of(from, offered)
    );
    if let Some(about) = offered
        .iter()
        .find(|m| m.id == to)
        .and_then(|m| m.description.as_deref())
    {
        text.push_str("\n\n");
        text.push_str(about);
    }
    text.push_str("\n\nTell me \"switch to the newer model\" if you want to try it.");
    text
}

pub(crate) struct ModelUpgrader {
    config_path: PathBuf,
    agent: Arc<dyn DaemonAgentRuntime>,
    hub: Arc<ChannelHub>,
    notify_session_ids: Vec<String>,
}

impl ModelUpgrader {
    pub(crate) fn new(
        config_path: PathBuf,
        agent: Arc<dyn DaemonAgentRuntime>,
        hub: Arc<ChannelHub>,
        notify_session_ids: Vec<String>,
    ) -> Self {
        Self {
            config_path,
            agent,
            hub,
            notify_session_ids,
        }
    }

    pub(crate) fn spawn(self) {
        tokio::spawn(async move {
            tokio::time::sleep(FIRST_CHECK_AFTER).await;
            loop {
                if let Err(error) = self.check().await {
                    warn!(%error, "Model upgrade check failed");
                }
                tokio::time::sleep(CHECK_EVERY).await;
            }
        });
    }

    /// The models the main provider offers; `Err` when it could not say.
    async fn offered(config: &AppConfig) -> anyhow::Result<Vec<Offered>> {
        let provider = &config.provider;
        if provider.kind == ProviderKind::OpenaiChatgpt {
            let backend = crate::providers::factory::chatgpt_backend(None, None)?;
            let listed = backend.list_models().await?;
            return Ok(listed
                .into_iter()
                .map(|m| Offered {
                    id: m.id,
                    display_name: m.display_name,
                    description: m.description,
                })
                .collect());
        }
        let ids = aidaemon_providers::catalog::list_models(
            provider.kind,
            &provider.base_url,
            &provider.api_key,
        )
        .await
        .map_err(anyhow::Error::msg)?;
        Ok(ids
            .into_iter()
            .map(|id| Offered {
                id,
                display_name: None,
                description: None,
            })
            .collect())
    }

    /// One real call with a tool, as the agent makes them.
    async fn trial(config: &AppConfig, model: &str) -> anyhow::Result<()> {
        let (backend, _) = crate::providers::factory::build_provider(&config.provider)?;
        let request = ModelRequest {
            model: model.to_string(),
            messages: vec![Message::user(
                "This checks that a model works with this app. Call the `ready` tool \
                 with ok set to true.",
            )],
            tools: vec![ToolSpec {
                name: "ready".to_string(),
                description: "Report that you are ready.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": false
                }),
            }],
            tool_choice: ToolChoice::Auto,
            options: RequestOptions {
                single_attempt: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let response = tokio::time::timeout(TRIAL_WITHIN, backend.complete(&request))
            .await
            .map_err(|_| anyhow::anyhow!("{model} did not answer within {TRIAL_WITHIN:?}"))??;
        anyhow::ensure!(
            response.tool_calls.iter().any(|call| call.name == "ready"),
            "{model} answered without calling the tool"
        );
        Ok(())
    }

    async fn check(&self) -> anyhow::Result<()> {
        let config = AppConfig::load(&self.config_path)?;
        if !config.provider.auto_upgrade {
            info!("Model upgrades are off (provider.auto_upgrade = false)");
            return Ok(());
        }
        let file = crate::config::config_file(&self.config_path);
        let text = file.read().await?;
        let doc: toml_edit::DocumentMut = text.parse()?;
        let written = written_models(&doc);

        let mut state = load_state(&self.config_path);
        if note_undo(&mut state, &written) {
            save_state(&self.config_path, &state);
        }

        let local = config.provider.kind == ProviderKind::OpenaiCompatible
            && ["localhost", "127.0.0.1", "[::1]"]
                .iter()
                .any(|host| config.provider.base_url.contains(host));
        if local {
            return Ok(());
        }
        let offered = Self::offered(&config).await?;
        info!(
            provider = ?config.provider.kind,
            offered = offered.len(),
            models = %offered.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", "),
            "Model upgrade check: the provider's models"
        );
        let vision = VisionConfig::from_files(&config.files);
        let moves = plan(&written, &offered, &state.declined, &vision);
        if moves.is_empty() {
            info!("Model upgrade check: every configured model is the newest of its line");
            return Ok(());
        }

        // A subscription has no per-call price; elsewhere a newer model can
        // cost more, so the owner is told and decides.
        if config.provider.kind != ProviderKind::OpenaiChatgpt {
            for (from, to) in &moves {
                if state.suggested.contains(to) {
                    continue;
                }
                info!(%from, %to, "Suggesting a newer model (pay-per-token provider)");
                self.hub
                    .broadcast_text(
                        &self.notify_session_ids,
                        &suggested_message(from, to, &offered),
                    )
                    .await;
                state.suggested.push(to.clone());
            }
            save_state(&self.config_path, &state);
            return Ok(());
        }

        if self.agent.model_pinned().await {
            info!("Model upgrade postponed: the owner pinned a model with /model");
            return Ok(());
        }

        let mut proven = BTreeMap::new();
        for (from, to) in &moves {
            match Self::trial(&config, to).await {
                Ok(()) => {
                    proven.insert(from.clone(), to.clone());
                }
                Err(error) => {
                    warn!(%from, %to, %error, "A newer model failed its trial; not switching")
                }
            }
        }
        if proven.is_empty() {
            return Ok(());
        }

        let updated = apply(&text, &proven)?;
        file.save(&updated).await?;
        let reloaded = match AppConfig::load(&self.config_path) {
            Ok(new_config) => self.agent.reload_provider(&new_config).await,
            Err(error) => Err(error),
        };
        if let Err(error) = reloaded {
            warn!(%error, "The upgraded config did not load; putting the previous one back");
            file.save(&text).await?;
            if let Ok(previous) = AppConfig::load(&self.config_path) {
                let _ = self.agent.reload_provider(&previous).await;
            }
            return Err(error);
        }

        if let Some(from) = &written.default
            && let Some(to) = proven.get(from)
        {
            state.last = Some(Switch {
                from: from.clone(),
                to: to.clone(),
                at_ms: now_ms(),
            });
        }
        save_state(&self.config_path, &state);
        info!(?proven, "Switched to newer models");
        let message = switched_message(
            &proven,
            written.default.as_deref(),
            &offered,
            config.provider.kind,
        );
        self.hub
            .broadcast_text(&self.notify_session_ids, &message)
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offered(ids: &[&str]) -> Vec<Offered> {
        ids.iter()
            .map(|id| Offered {
                id: id.to_string(),
                display_name: None,
                description: None,
            })
            .collect()
    }

    fn vision() -> VisionConfig {
        VisionConfig {
            enabled: true,
            max_image_bytes: 1,
            mime_types: Vec::new(),
            model_patterns: vec!["gpt-6".into(), "grok-4".into()],
        }
    }

    const CONFIG: &str = r#"[provider]
kind = "openai_chatgpt"

[[provider.fallbacks]]
kind = "anthropic"

[provider.fallbacks.models]
default = "claude-haiku-5-5"

[provider.models]
default = "gpt-6-sol"
fallback = [
    "gpt-6-luna",
    "gpt-5.6-terra",
]
fast = "gpt-6-luna"
"#;

    fn written(text: &str) -> WrittenModels {
        written_models(&text.parse().unwrap())
    }

    #[test]
    fn the_default_moves_and_the_old_one_leads_the_fallbacks() {
        let listed = offered(&["gpt-6-sol", "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6-luna"]);
        let moves = plan(&written(CONFIG), &listed, &[], &vision());
        assert_eq!(
            moves,
            BTreeMap::from([("gpt-6-sol".to_string(), "gpt-6.2-sol".to_string())])
        );
        let updated = apply(CONFIG, &moves).unwrap();
        let after = written(&updated);
        assert_eq!(after.default.as_deref(), Some("gpt-6.2-sol"));
        assert_eq!(after.fast.as_deref(), Some("gpt-6-luna"));
        assert_eq!(after.fallback, ["gpt-6-sol", "gpt-6-luna", "gpt-5.6-terra"]);
        // The fallback provider's own models are untouched.
        assert!(updated.contains("default = \"claude-haiku-5-5\""));
    }

    #[test]
    fn a_declined_model_is_not_offered_again() {
        let listed = offered(&["gpt-6.2-sol"]);
        let declined = vec!["gpt-6.2-sol".to_string()];
        assert!(plan(&written(CONFIG), &listed, &declined, &vision()).is_empty());
    }

    #[test]
    fn a_model_the_harness_would_treat_worse_is_not_offered() {
        // An unknown family is Guided where the old one is Autonomous.
        assert!(!harness_keeps_up("grok-4", "grok-5", &vision()));
        assert!(harness_keeps_up("gpt-6-sol", "gpt-6.2-sol", &vision()));
    }

    #[test]
    fn going_back_declines_the_upgrade() {
        let mut state = UpgradeState {
            last: Some(Switch {
                from: "gpt-6-sol".into(),
                to: "gpt-6.2-sol".into(),
                at_ms: 1,
            }),
            ..Default::default()
        };
        // Still on the new model: nothing changes.
        let on_new = WrittenModels {
            default: Some("gpt-6.2-sol".into()),
            ..Default::default()
        };
        assert!(!note_undo(&mut state, &on_new));
        let back = WrittenModels {
            default: Some("gpt-6-sol".into()),
            ..Default::default()
        };
        assert!(note_undo(&mut state, &back));
        assert_eq!(state.declined, ["gpt-6.2-sol"]);
        assert!(state.last.is_none());
    }

    #[test]
    fn the_message_names_both_models_in_plain_words() {
        let listed = vec![Offered {
            id: "gpt-6.2-sol".into(),
            display_name: Some("GPT-6.2 Sol".into()),
            description: Some("Faster on everyday work.".into()),
        }];
        let moves = BTreeMap::from([("gpt-6-sol".to_string(), "gpt-6.2-sol".to_string())]);
        let text = switched_message(
            &moves,
            Some("gpt-6-sol"),
            &listed,
            ProviderKind::OpenaiChatgpt,
        );
        assert!(text.starts_with("OpenAI released a newer model, so I've switched to it: GPT-6.2 Sol (it replaces gpt-6-sol)."));
        assert!(text.contains("Faster on everyday work."));
    }
}
