//! Keeping the owner on the newest model of the line they use.
//!
//! Providers release new versions every few weeks, and an owner who does
//! not follow launches keeps running an old one. Once a day this job reads
//! the model listing of every configured provider (the main one and each
//! fallback) and, for each model written in its `models` table with a newer
//! version of its line ([`aidaemon_providers::model_upgrade`]):
//!
//! - skips it when the harness would treat the new model worse (another
//!   trust tier, no vision where the old one had it, a smaller known
//!   window), or when the owner went back from it before;
//! - on a subscription (the ChatGPT sign-in, no per-call price) proves it
//!   with one real tool call, writes it to the config with the old model
//!   first among that provider's fallbacks (a failing new model falls back
//!   to the old one on every call), reloads the provider and tells the owner;
//! - on a pay-per-token provider only tells the owner once: a newer model
//!   can cost more, and the daemon does not know prices.
//!
//! The owner hears it twice: a message through the notification queue in
//! each owner chat, or, for an owner with no chat bot, in the app
//! conversation they used last (kept in that conversation's history, so a
//! reply such as "switch back" is understood), and a line on the Settings
//! page for a week ([`notices`]).
//!
//! `provider.auto_upgrade = false` turns it off; a `/model` pin postpones
//! it. An owner who puts the old model back (in Settings or by asking)
//! keeps it: each switch is recorded, and a default that went back to what
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

use crate::config::{AppConfig, ProviderConfig};
use crate::runtime_ports::DaemonAgentRuntime;
use crate::traits::StateStore;

/// The first check waits for channels to connect.
const FIRST_CHECK_AFTER: Duration = Duration::from_secs(120);
const CHECK_EVERY: Duration = Duration::from_secs(24 * 3600);
const TRIAL_WITHIN: Duration = Duration::from_secs(90);
/// How long the Settings page mentions a switch or a suggestion.
const NOTICE_FOR_MS: u64 = 7 * 24 * 3600 * 1000;
/// The notification type of the messages this job queues.
const NOTIFICATION_TYPE: &str = "model_upgrade";

/// A provider entry of the config: the main one or a fallback by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Slot {
    Main,
    Fallback(usize),
}

impl Slot {
    fn label(self, kind: ProviderKind) -> String {
        match self {
            Slot::Main => "main model".to_string(),
            Slot::Fallback(_) => format!("backup on {}", maker(kind)),
        }
    }
}

/// What the job remembers between checks, beside the config file.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct UpgradeState {
    /// Models the owner went back from: never offered again.
    #[serde(default)]
    declined: Vec<String>,
    /// Switches made, newest last, to notice an undo and for Settings.
    #[serde(default)]
    switches: Vec<Switch>,
    /// Newer models told about on a pay-per-token provider.
    #[serde(default)]
    suggestions: Vec<Suggestion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Switch {
    slot: Slot,
    kind_name: String,
    from: String,
    to: String,
    at_ms: u64,
    /// Still the model in use (an undo or a later choice ends it).
    #[serde(default = "yes")]
    current: bool,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Suggestion {
    slot: Slot,
    kind_name: String,
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

fn day_of(at_ms: u64) -> String {
    chrono::DateTime::from_timestamp_millis(at_ms as i64)
        .map(|at| {
            at.with_timezone(&chrono::Local)
                .format("%b %-d")
                .to_string()
        })
        .unwrap_or_default()
}

/// What the Settings page says about model upgrades of the last week.
pub(crate) fn notices(config_path: &Path) -> Vec<String> {
    let state = load_state(config_path);
    let since = now_ms().saturating_sub(NOTICE_FOR_MS);
    let mut notices = Vec::new();
    for switch in state
        .switches
        .iter()
        .filter(|s| s.at_ms >= since && s.current)
    {
        notices.push(format!(
            "On {}, the {} moved to {}, a newer version of {}. Choose {} again to go back; \
             provider.auto_upgrade = false stops these switches.",
            day_of(switch.at_ms),
            match switch.slot {
                Slot::Main => "main model".to_string(),
                Slot::Fallback(_) => format!("{} backup", switch.kind_name),
            },
            switch.to,
            switch.from,
            switch.from
        ));
    }
    for suggestion in state.suggestions.iter().filter(|s| s.at_ms >= since) {
        notices.push(format!(
            "A newer {} model is out: {} (in use: {}). It may cost more per message; \
             choose it to switch.",
            suggestion.kind_name, suggestion.to, suggestion.from
        ));
    }
    notices
}

/// A model the provider offers, as the job reads it.
#[derive(Debug, Clone, PartialEq)]
struct Offered {
    id: String,
    display_name: Option<String>,
    description: Option<String>,
}

/// A slot's `models` entries as written in the file: only these are moved
/// (a model the file leaves to the catalog follows the release).
#[derive(Debug, Default, Clone, PartialEq)]
struct WrittenModels {
    default: Option<String>,
    fast: Option<String>,
    smart: Option<String>,
    fallback: Vec<String>,
}

fn fallback_models_table(
    doc: &toml_edit::DocumentMut,
    index: usize,
) -> Option<&dyn toml_edit::TableLike> {
    doc.get("provider")?
        .get("fallbacks")?
        .as_array_of_tables()?
        .get(index)?
        .get("models")?
        .as_table_like()
}

fn slot_table(doc: &toml_edit::DocumentMut, slot: Slot) -> Option<&dyn toml_edit::TableLike> {
    match slot {
        Slot::Main => doc.get("provider")?.get("models")?.as_table_like(),
        Slot::Fallback(i) => fallback_models_table(doc, i),
    }
}

fn written_models(doc: &toml_edit::DocumentMut, slot: Slot) -> WrittenModels {
    let table = slot_table(doc, slot);
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

fn slot_table_mut(
    doc: &mut toml_edit::DocumentMut,
    slot: Slot,
) -> anyhow::Result<&mut toml_edit::Table> {
    let provider = doc["provider"]
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("no [provider] table"))?;
    let entry = match slot {
        Slot::Main => provider,
        Slot::Fallback(i) => provider
            .get_mut("fallbacks")
            .and_then(toml_edit::Item::as_array_of_tables_mut)
            .and_then(|tables| tables.get_mut(i))
            .ok_or_else(|| anyhow::anyhow!("no provider.fallbacks[{i}]"))?,
    };
    entry
        .entry("models")
        .or_insert_with(toml_edit::table)
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("the models entry is not a table"))
}

/// The config text with `moves` applied to `slot`, and the old default
/// first among that slot's fallbacks.
fn apply(text: &str, slot: Slot, moves: &BTreeMap<String, String>) -> anyhow::Result<String> {
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    let written = written_models(&doc, slot);
    let models = slot_table_mut(&mut doc, slot)?;
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
    let mut fallback: Vec<String> = Vec::new();
    let new_default = written.default.as_deref().map(moved);
    if let Some(old_default) = &written.default
        && moves.contains_key(old_default)
    {
        fallback.push(old_default.clone());
    }
    for model in &written.fallback {
        let model = moved(model);
        if Some(&model) != new_default.as_ref() && !fallback.contains(&model) {
            fallback.push(model);
        }
    }
    if !fallback.is_empty() || !written.fallback.is_empty() {
        let mut array = toml_edit::Array::new();
        for model in fallback {
            array.push(model);
        }
        models["fallback"] = toml_edit::value(array);
    }
    Ok(doc.to_string())
}

/// Each slot's default as the file now has it.
fn defaults_by_slot(doc: &toml_edit::DocumentMut, slots: &[Slot]) -> Vec<(Slot, Option<String>)> {
    slots
        .iter()
        .map(|&slot| (slot, written_models(doc, slot).default))
        .collect()
}

/// When the owner put back what a switch replaced, that model is declined;
/// a switch whose model is no longer in use is no longer current. `true`
/// when the state changed.
fn note_undo(state: &mut UpgradeState, defaults: &[(Slot, Option<String>)]) -> bool {
    let mut changed = false;
    for switch in state.switches.iter_mut().filter(|s| s.current) {
        let Some((_, default)) = defaults.iter().find(|(slot, _)| *slot == switch.slot) else {
            continue;
        };
        match default.as_deref() {
            Some(model) if model == switch.to => {}
            Some(model) if model == switch.from => {
                info!(model = %switch.to, "The owner went back from an upgraded model; it is not offered again");
                if !state.declined.contains(&switch.to) {
                    state.declined.push(switch.to.clone());
                }
                switch.current = false;
                changed = true;
            }
            _ => {
                switch.current = false;
                changed = true;
            }
        }
    }
    changed
}

fn maker(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::OpenaiChatgpt => "OpenAI",
        ProviderKind::OpenaiCompatible => "your OpenAI-compatible provider",
        ProviderKind::Anthropic => "Anthropic",
        ProviderKind::GoogleGenai => "Google",
        ProviderKind::XaiNative => "xAI",
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
    slot: Slot,
    kind: ProviderKind,
    moves: &BTreeMap<String, String>,
    default: Option<&str>,
    offered: &[Offered],
) -> String {
    let mut lines = Vec::new();
    match default.and_then(|d| moves.get_key_value(d)) {
        Some((from, to)) => {
            lines.push(format!(
                "Model upgrade: {} released a newer model, so I switched my {} from {} to {}.",
                maker(kind),
                slot.label(kind),
                name_of(from, offered),
                name_of(to, offered)
            ));
            if let Some(about) = offered
                .iter()
                .find(|m| &m.id == to)
                .and_then(|m| m.description.as_deref())
            {
                lines.push(about.to_string());
            }
        }
        None => lines.push(format!(
            "Model upgrade: {} released newer models, so I switched to them.",
            maker(kind)
        )),
    }
    for (from, to) in moves {
        if Some(from.as_str()) != default {
            lines.push(format!(
                "{} also moved to {}.",
                name_of(from, offered),
                name_of(to, offered)
            ));
        }
    }
    lines.push(
        "Nothing for you to do: if the new model has trouble, the previous one takes over \
         on its own. To keep the previous model, tell me to switch back (or pick it in \
         Settings) and I won't move you to this one again."
            .to_string(),
    );
    lines.join("\n\n")
}

fn suggested_message(
    slot: Slot,
    kind: ProviderKind,
    from: &str,
    to: &str,
    offered: &[Offered],
) -> String {
    let mut text = format!(
        "Model update: {} released {}, a newer version of {} (my {}). I haven't switched, \
         because a newer model can cost more per message on your account.",
        maker(kind),
        name_of(to, offered),
        name_of(from, offered),
        slot.label(kind)
    );
    if let Some(about) = offered
        .iter()
        .find(|m| m.id == to)
        .and_then(|m| m.description.as_deref())
    {
        text.push_str("\n\n");
        text.push_str(about);
    }
    text.push_str(&format!(
        "\n\nTell me \"switch to {to}\" if you want to try it."
    ));
    text
}

pub(crate) struct ModelUpgrader {
    config_path: PathBuf,
    agent: Arc<dyn DaemonAgentRuntime>,
    state: Arc<dyn StateStore>,
    notify_session_ids: Vec<String>,
    /// The database, when the Client Gateway serves apps: an owner with no
    /// chat bot is told in their latest app conversation.
    app_pool: Option<sqlx::SqlitePool>,
}

impl ModelUpgrader {
    pub(crate) fn new(
        config_path: PathBuf,
        agent: Arc<dyn DaemonAgentRuntime>,
        state: Arc<dyn StateStore>,
        notify_session_ids: Vec<String>,
        app_pool: Option<sqlx::SqlitePool>,
    ) -> Self {
        Self {
            config_path,
            agent,
            state,
            notify_session_ids,
            app_pool,
        }
    }

    /// Where the owner is told: their chats, or else (only the app) the
    /// app conversation they used last.
    async fn owner_sessions(&self) -> Vec<String> {
        if !self.notify_session_ids.is_empty() {
            return self.notify_session_ids.clone();
        }
        match &self.app_pool {
            Some(pool) => crate::startup::client_sessions::latest_app_session(pool)
                .await
                .into_iter()
                .collect(),
            None => Vec::new(),
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

    /// The models a provider offers; `Err` when it could not say.
    async fn offered(provider: &ProviderConfig) -> anyhow::Result<Vec<Offered>> {
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
        // A backup left at the default base URL has none of its own.
        let base_url =
            crate::providers::factory::provider_specific_base_url_override(provider).unwrap_or("");
        let ids =
            aidaemon_providers::catalog::list_models(provider.kind, base_url, &provider.api_key)
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
    async fn trial(provider: &ProviderConfig, model: &str) -> anyhow::Result<()> {
        let (backend, _) = crate::providers::factory::build_provider(provider)?;
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

    /// Queue `message` for the owner (their chats, or their latest app
    /// conversation): delivered there and kept in that conversation's
    /// history, so a reply to it is understood.
    async fn tell_owner(&self, message: &str) {
        let sessions = self.owner_sessions().await;
        if sessions.is_empty() {
            info!("No owner conversation to tell about a model upgrade; Settings shows it");
        }
        for session_id in &sessions {
            let entry = crate::traits::NotificationEntry::new(
                NOTIFICATION_TYPE,
                session_id,
                NOTIFICATION_TYPE,
                message,
            );
            if let Err(error) = self.state.enqueue_notification(&entry).await {
                warn!(%error, %session_id, "Could not queue the model upgrade message");
            }
        }
    }

    async fn check(&self) -> anyhow::Result<()> {
        let config = AppConfig::load(&self.config_path)?;
        if !config.provider.auto_upgrade {
            info!("Model upgrades are off (provider.auto_upgrade = false)");
            return Ok(());
        }
        let file = crate::config::config_file(&self.config_path);
        let mut text = file.read().await?;
        let doc: toml_edit::DocumentMut = text.parse()?;

        let mut slots = vec![(Slot::Main, config.provider.clone())];
        for (i, fallback) in config.provider.fallbacks.iter().enumerate() {
            slots.push((Slot::Fallback(i), fallback.clone()));
        }
        let mut state = load_state(&self.config_path);
        let slot_ids: Vec<Slot> = slots.iter().map(|(slot, _)| *slot).collect();
        if note_undo(&mut state, &defaults_by_slot(&doc, &slot_ids)) {
            save_state(&self.config_path, &state);
        }

        let vision = VisionConfig::from_files(&config.files);
        let pinned = self.agent.model_pinned().await;
        let original = text.clone();
        let mut switched = Vec::new();
        for (slot, provider) in &slots {
            let local = provider.kind == ProviderKind::OpenaiCompatible
                && ["localhost", "127.0.0.1", "[::1]"]
                    .iter()
                    .any(|host| provider.base_url.contains(host));
            if local {
                continue;
            }
            let offered = match Self::offered(provider).await {
                Ok(offered) => offered,
                Err(error) => {
                    warn!(?slot, kind = ?provider.kind, %error, "Model upgrade check: could not read the provider's models");
                    continue;
                }
            };
            let current_doc: toml_edit::DocumentMut = text.parse()?;
            let written = written_models(&current_doc, *slot);
            info!(
                ?slot,
                kind = ?provider.kind,
                offered = offered.len(),
                models = %offered.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", "),
                configured = ?written,
                "Model upgrade check: the provider's models"
            );
            let moves = plan(&written, &offered, &state.declined, &vision);
            if moves.is_empty() {
                info!(
                    ?slot,
                    "Model upgrade check: these models are the newest of their line"
                );
                continue;
            }

            // A subscription has no per-call price; elsewhere a newer model
            // can cost more, so the owner is told and decides.
            if provider.kind != ProviderKind::OpenaiChatgpt {
                for (from, to) in &moves {
                    if state.suggestions.iter().any(|s| &s.to == to) {
                        continue;
                    }
                    info!(?slot, %from, %to, "Suggesting a newer model (pay-per-token provider)");
                    self.tell_owner(&suggested_message(*slot, provider.kind, from, to, &offered))
                        .await;
                    state.suggestions.push(Suggestion {
                        slot: *slot,
                        kind_name: maker(provider.kind).to_string(),
                        from: from.clone(),
                        to: to.clone(),
                        at_ms: now_ms(),
                    });
                }
                continue;
            }

            if pinned {
                info!("Model upgrade postponed: the owner pinned a model with /model");
                continue;
            }
            let mut proven = BTreeMap::new();
            for (from, to) in &moves {
                match Self::trial(provider, to).await {
                    Ok(()) => {
                        proven.insert(from.clone(), to.clone());
                    }
                    Err(error) => {
                        warn!(%from, %to, %error, "A newer model failed its trial; not switching")
                    }
                }
            }
            if proven.is_empty() {
                continue;
            }
            text = apply(&text, *slot, &proven)?;
            switched.push((
                *slot,
                provider.kind,
                proven,
                written.default.clone(),
                offered,
            ));
        }
        save_state(&self.config_path, &state);
        if switched.is_empty() {
            return Ok(());
        }

        file.save(&text).await?;
        let reloaded = match AppConfig::load(&self.config_path) {
            Ok(new_config) => self.agent.reload_provider(&new_config).await,
            Err(error) => Err(error),
        };
        if let Err(error) = reloaded {
            warn!(%error, "The upgraded config did not load; putting the previous one back");
            file.save(&original).await?;
            if let Ok(previous) = AppConfig::load(&self.config_path) {
                let _ = self.agent.reload_provider(&previous).await;
            }
            return Err(error);
        }

        for (slot, kind, proven, default, offered) in &switched {
            info!(?slot, ?proven, "Switched to newer models");
            if let Some(from) = default
                && let Some(to) = proven.get(from)
            {
                for earlier in state.switches.iter_mut().filter(|s| s.slot == *slot) {
                    earlier.current = false;
                }
                state.switches.push(Switch {
                    slot: *slot,
                    kind_name: maker(*kind).to_string(),
                    from: from.clone(),
                    to: to.clone(),
                    at_ms: now_ms(),
                    current: true,
                });
            }
            self.tell_owner(&switched_message(
                *slot,
                *kind,
                proven,
                default.as_deref(),
                offered,
            ))
            .await;
        }
        save_state(&self.config_path, &state);
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
            model_patterns: vec!["gpt-6".into(), "grok-4".into(), "claude-".into()],
        }
    }

    const CONFIG: &str = r#"[provider]
kind = "openai_chatgpt"

[[provider.fallbacks]]
kind = "anthropic"

[provider.fallbacks.models]
default = "claude-haiku-5-5"
fallback = ["claude-sonnet-5-5"]

[provider.models]
default = "gpt-6-sol"
fallback = [
    "gpt-6-luna",
    "gpt-5.6-terra",
]
fast = "gpt-6-luna"
"#;

    fn written(text: &str, slot: Slot) -> WrittenModels {
        written_models(&text.parse().unwrap(), slot)
    }

    #[test]
    fn the_default_moves_and_the_old_one_leads_the_fallbacks() {
        let listed = offered(&["gpt-6-sol", "gpt-6.2-sol", "gpt-6.2-astra", "gpt-6-luna"]);
        let moves = plan(&written(CONFIG, Slot::Main), &listed, &[], &vision());
        assert_eq!(
            moves,
            BTreeMap::from([("gpt-6-sol".to_string(), "gpt-6.2-sol".to_string())])
        );
        let updated = apply(CONFIG, Slot::Main, &moves).unwrap();
        let after = written(&updated, Slot::Main);
        assert_eq!(after.default.as_deref(), Some("gpt-6.2-sol"));
        assert_eq!(after.fast.as_deref(), Some("gpt-6-luna"));
        assert_eq!(after.fallback, ["gpt-6-sol", "gpt-6-luna", "gpt-5.6-terra"]);
        // The fallback provider's own models are untouched.
        assert_eq!(
            written(&updated, Slot::Fallback(0)),
            written(CONFIG, Slot::Fallback(0))
        );
    }

    #[test]
    fn a_fallback_provider_is_read_and_moved_on_its_own() {
        let fallback = written(CONFIG, Slot::Fallback(0));
        assert_eq!(fallback.default.as_deref(), Some("claude-haiku-5-5"));
        let listed = offered(&["claude-haiku-6", "claude-sonnet-6", "claude-opus-6"]);
        let moves = plan(&fallback, &listed, &[], &vision());
        assert_eq!(moves.len(), 2);
        let updated = apply(CONFIG, Slot::Fallback(0), &moves).unwrap();
        let after = written(&updated, Slot::Fallback(0));
        assert_eq!(after.default.as_deref(), Some("claude-haiku-6"));
        assert_eq!(after.fallback, ["claude-haiku-5-5", "claude-sonnet-6"]);
        assert_eq!(written(&updated, Slot::Main), written(CONFIG, Slot::Main));
    }

    #[test]
    fn a_declined_model_is_not_offered_again() {
        let listed = offered(&["gpt-6.2-sol"]);
        let declined = vec!["gpt-6.2-sol".to_string()];
        assert!(plan(&written(CONFIG, Slot::Main), &listed, &declined, &vision()).is_empty());
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
            switches: vec![Switch {
                slot: Slot::Main,
                kind_name: "OpenAI".into(),
                from: "gpt-6-sol".into(),
                to: "gpt-6.2-sol".into(),
                at_ms: 1,
                current: true,
            }],
            ..Default::default()
        };
        // Still on the new model: nothing changes.
        assert!(!note_undo(
            &mut state,
            &[(Slot::Main, Some("gpt-6.2-sol".into()))]
        ));
        assert!(note_undo(
            &mut state,
            &[(Slot::Main, Some("gpt-6-sol".into()))]
        ));
        assert_eq!(state.declined, ["gpt-6.2-sol"]);
        assert!(!state.switches[0].current);
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
            Slot::Main,
            ProviderKind::OpenaiChatgpt,
            &moves,
            Some("gpt-6-sol"),
            &listed,
        );
        assert!(text.starts_with(
            "Model upgrade: OpenAI released a newer model, so I switched my main model from \
             gpt-6-sol to GPT-6.2 Sol."
        ));
        assert!(text.contains("Faster on everyday work."));
        let backup = suggested_message(
            Slot::Fallback(0),
            ProviderKind::Anthropic,
            "claude-haiku-5-5",
            "claude-haiku-6",
            &[],
        );
        assert!(backup.contains("my backup on Anthropic"));
    }
}
