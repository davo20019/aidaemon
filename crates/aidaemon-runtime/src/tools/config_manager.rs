//! `manage_config`: read and change the host's config file, switch and
//! fail over providers, and turn on speech-to-text, the Client Gateway and
//! its exposure.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::warn;

use std::sync::Arc;

use aidaemon_secrets::SecretStore;
use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::config_file::ConfigFile;
use aidaemon_tools::semantics::{
    OwnerApprovalNeed, ToolCallContext, ToolCallSemantics, ToolCapabilities, ToolMutationEffects,
    semantics_for_exact_read_actions,
};

mod config_path;
use config_path::ConfigPath;

/// Key names that contain secrets and must be redacted before showing to the LLM.
const SENSITIVE_KEYS: &[&str] = &[
    "api_key",
    "gateway_token",
    "bot_token",
    "app_token",
    "password",
    "encryption_key",
];

/// Key names that are security-sensitive and require user approval to modify.
/// This includes secrets plus keys that control security behavior.
const APPROVAL_REQUIRED_KEYS: &[&str] = &[
    // Secrets
    "api_key",
    "gateway_token",
    "bot_token",
    "app_token",
    "password",
    "encryption_key",
    // Security-sensitive settings
    "allowed_prefixes",
    "allowed_user_ids",
    "allowed_command_prefixes",
    "base_url", // Could redirect API traffic
    "trusted",  // Scheduler trusted flag
    // Where the persona (part of every prompt) is read from, and headers
    // sent with every model request.
    "persona_file",
    "extra_headers",
    // Where the agent's own stores are read from and written to.
    "path",
    "projects_dir",
    "db_path",
];

/// The top-level tables a change may touch without the owner's approval:
/// how the agent talks and thinks, nothing about who may reach it, what it
/// may do, what runs on this machine or where data goes. Every other table
/// (channels and their allowlists, users, authority, terminal, MCP servers,
/// OAuth endpoints, telemetry, updates, the database, ...) is the owner's
/// call, including tables added later.
const APPROVAL_FREE_TABLES: &[&str] = &[
    "agent",
    "provider",
    "checkpoints",
    "heartbeat",
    "people",
    "memory",
    "diagnostics",
];

/// Placeholder shown instead of actual secret values.
const REDACTED: &str = "***REDACTED***";

#[derive(Clone, Copy)]
struct ProviderPreset {
    id: &'static str,
    aliases: &'static [&'static str],
    display_name: &'static str,
    kind: &'static str,
    base_url: &'static str,
    primary: &'static str,
    fast: &'static str,
    smart: &'static str,
    needs_api_key: bool,
    supports_gateway_token: bool,
    requires_custom_base_url: bool,
}

/// `switch_provider`'s presets: the SDK's provider catalog, the one
/// table setup, `manage_config` and the SDK preset all read.
static PROVIDER_PRESETS: std::sync::LazyLock<Vec<ProviderPreset>> =
    std::sync::LazyLock::new(|| {
        aidaemon_providers::catalog::CATALOG
            .iter()
            .map(|entry| ProviderPreset {
                id: entry.id,
                aliases: entry.aliases,
                display_name: entry.display_name,
                kind: entry.kind_name(),
                base_url: entry.base_url,
                primary: entry.models.primary,
                fast: entry.models.fast,
                smart: entry.models.smart,
                needs_api_key: entry.needs_api_key(),
                supports_gateway_token: entry.gateway_token,
                requires_custom_base_url: entry.custom_base_url,
            })
            .collect()
    });

#[derive(Clone)]
struct ResolvedProviderSelection {
    preset: &'static ProviderPreset,
    /// The secret store holds this provider slot's API key.
    api_key_stored: bool,
    base_url: String,
    primary_model: String,
    fast_model: String,
    smart_model: String,
    /// The secret store holds this provider slot's gateway token (only
    /// looked up for presets that take one).
    gateway_token_stored: bool,
}

impl ResolvedProviderSelection {
    fn fallback_models(&self) -> Vec<String> {
        let mut models = Vec::new();
        for candidate in [&self.smart_model, &self.fast_model] {
            let trimmed = candidate.trim();
            if trimmed.is_empty()
                || trimmed == self.primary_model
                || models.iter().any(|existing: &String| existing == trimmed)
            {
                continue;
            }
            models.push(trimmed.to_string());
        }
        models
    }
}

pub struct ConfigManagerTool {
    config: ConfigFile,
    secrets: Arc<dyn SecretStore>,
    /// The daemon's live permission choices, taken again from the file
    /// after each change (`[authority]`, `[cli_agents.permissions]`).
    permissions: Option<crate::permissions::Permissions>,
}

impl ConfigManagerTool {
    /// `secrets` keeps the provider keys a change stores as `"keychain"`.
    pub fn new(config: ConfigFile, secrets: Arc<dyn SecretStore>) -> Self {
        Self {
            config,
            secrets,
            permissions: None,
        }
    }

    /// Keep `permissions` the file's after every change this tool makes.
    pub fn with_permissions(mut self, permissions: crate::permissions::Permissions) -> Self {
        self.permissions = Some(permissions);
        self
    }

    /// Take the permission choices the file now holds: a change to
    /// `[authority]` or `[cli_agents.permissions]` (or a restored backup)
    /// takes effect at once, with no second copy.
    async fn refresh_permissions(&self) {
        if let Some(permissions) = &self.permissions {
            match self.config.load_table().await {
                Ok(table) => permissions.reload(&table),
                Err(error) => {
                    warn!(%error, "Could not read config.toml back for the permission level")
                }
            }
        }
    }

    /// Check if a key path requires user approval to modify.
    fn requires_approval(key_path: &str) -> bool {
        let last = key_path.rsplit('.').next().unwrap_or(key_path);
        APPROVAL_REQUIRED_KEYS.contains(&last)
    }

    /// Refuse a change that needs the owner's approval when the call did not
    /// come through the agent kernel (which asked the owner).
    fn require_kernel_approval(args: &ConfigArgs) -> anyhow::Result<()> {
        if args.kernel_approved {
            return Ok(());
        }
        anyhow::bail!(
            "manage_config {} requires approval through the agent kernel; it was called outside \
             an approved agent call",
            args.action
        )
    }

    /// The refusal for a `set` that would write a secret's value: a
    /// sensitive key only takes the reference `keychain`, and the owner puts
    /// the value in the secret store themselves. Never echoes the value.
    fn literal_secret_refusal(args: &ConfigArgs) -> Option<String> {
        if args.action != "set"
            || args.value.is_empty()
            || args.value.trim() == "keychain"
            || !is_sensitive_key(&args.key)
        {
            return None;
        }
        let path = ConfigPath::parse(&args.key).ok();
        let instruction = match path.as_ref().and_then(provider_secret_name) {
            Some((name, primary)) => Self::owner_secret_instruction(&[&name], primary),
            None => "Ask the owner to store it with `aidaemon keychain set <key>` in their own \
                     terminal; never ask them to paste a secret in chat."
                .to_string(),
        };
        Some(format!(
            "Nothing was changed: secret values are not accepted as tool arguments. {} \
             Then set `{}` to `keychain`.",
            instruction, args.key
        ))
    }

    /// The card summary for a `set` that needs approval: secrets are shown
    /// redacted, other security-sensitive values as given.
    fn set_summary(args: &ConfigArgs) -> String {
        let display_value = if is_sensitive_key(&args.key) {
            "[REDACTED]".to_string()
        } else {
            args.value.clone()
        };
        if args.action == "unset" {
            return format!("Config change: remove {}", args.key);
        }
        format!("Config change: {} = {}", args.key, display_value)
    }

    fn base_url_display(selection: &ResolvedProviderSelection) -> &str {
        if selection.base_url.is_empty() {
            "(not used)"
        } else {
            selection.base_url.as_str()
        }
    }

    fn switch_provider_summary(selection: &ResolvedProviderSelection) -> String {
        format!(
            "Switch provider to {} (kind={}, base_url={}, models={}/{}/{})",
            selection.preset.display_name,
            selection.preset.kind,
            Self::base_url_display(selection),
            selection.primary_model,
            selection.fast_model,
            selection.smart_model
        )
    }

    fn add_failover_summary(next_index: usize, selection: &ResolvedProviderSelection) -> String {
        format!(
            "Add failover provider #{}: {} (kind={}, base_url={}, models={}/{}/{})",
            next_index,
            selection.preset.display_name,
            selection.preset.kind,
            Self::base_url_display(selection),
            selection.primary_model,
            selection.fast_model,
            selection.smart_model
        )
    }

    /// The owner's card for removing failover `index`: its place counted
    /// from 1, its kind and model, as Settings shows them.
    fn remove_failover_card(entries: &[toml::Value], index: usize) -> String {
        format!(
            "Remove failover {}: {}. It is no longer tried when the providers before it fail.",
            index + 1,
            Self::failover_label(entries, index)
        )
    }

    /// Failover `index` in plain words: where it is (its address's host,
    /// else its kind) and its model, as "openrouter.ai (model-x)".
    fn failover_label(entries: &[toml::Value], index: usize) -> String {
        let table = entries.get(index).and_then(toml::Value::as_table);
        let field = |key: &str| {
            table
                .and_then(|t| t.get(key))
                .and_then(toml::Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let model = table
            .and_then(|t| t.get("models"))
            .and_then(toml::Value::as_table)
            .and_then(|m| m.get("default").or_else(|| m.get("primary")))
            .and_then(toml::Value::as_str)
            .unwrap_or_default();
        let base_url = field("base_url");
        let host = base_url
            .split("://")
            .nth(1)
            .and_then(|rest| rest.split(['/', ':']).next())
            .filter(|host| !host.is_empty());
        let mut label = host.map(str::to_string).unwrap_or_else(|| field("kind"));
        if !model.is_empty() {
            label.push_str(&format!(" ({model})"));
        }
        label
    }

    /// The card summary for a call that asks the owner, read from the
    /// config as the call will find it; `None` for a call that needs no card
    /// (a change that is not security-sensitive, or a call its action
    /// refuses before changing anything).
    async fn approval_summary(&self, args: &ConfigArgs) -> anyhow::Result<Option<String>> {
        match args.action.as_str() {
            "set" | "unset" => {
                if args.key.is_empty()
                    || (args.action == "set" && args.value.is_empty())
                    || Self::literal_secret_refusal(args).is_some()
                {
                    return Ok(None);
                }
                let needs = Self::change_needs_approval(args)
                    || (args.action == "unset"
                        && Self::unset_removes_approval_key(&self.config.read().await?, &args.key));
                Ok(needs.then(|| Self::set_summary(args)))
            }
            "enable_stt" => {
                let probe = Self::stt_probe(args);
                if !probe.is_complete() {
                    return Ok(None);
                }
                Ok(Some(Self::stt_summary(&probe, Self::stt_language(args))))
            }
            "restore" => Ok(Self::restore_candidate(&self.config.backups())
                .map(|(path, _)| Self::restore_summary(&path))),
            "enable_client_gateway" => {
                let content = self.config.read().await?;
                let settings = Self::client_gateway_settings(&content);
                Ok(Some(format!(
                    "Turn on the Client Gateway: local apps (terminal UI, web client) can chat, \
                     watch every chat's turns and answer approvals over {}:{} (loopback only), \
                     and the local token is saved to {} (minted if none works). Applies after a restart.",
                    settings.bind,
                    settings.port,
                    aidaemon_client_gateway::token::client_token_file(&settings).display()
                )))
            }
            "enable_exposure" => Ok(Self::exposure_settings(args)
                .ok()
                .map(|(service, settings)| Self::exposure_summary(&service, &settings))),
            "switch_provider" => Ok(self
                .resolve_provider_selection(args, "switch_provider", None)
                .ok()
                .map(|selection| Self::switch_provider_summary(&selection))),
            "add_failover_provider" => {
                let content = self.config.read().await?;
                let doc: toml::Table = content.parse()?;
                let next_index = match doc.get("provider") {
                    None => 0,
                    Some(value) => {
                        let provider_table = value
                            .as_table()
                            .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
                        Self::get_failover_array(provider_table)?
                            .map(|entries| entries.len())
                            .unwrap_or(0)
                    }
                };
                let key_prefix = Self::fallback_key_prefix(next_index);
                let Ok(selection) = self.resolve_provider_selection(
                    args,
                    "add_failover_provider",
                    Some(&key_prefix),
                ) else {
                    return Ok(None);
                };
                Ok(Some(Self::add_failover_summary(next_index, &selection)))
            }
            "remove_failover_provider" => {
                let Some(index) = args.failover_index else {
                    return Ok(None);
                };
                let content = self.config.read().await?;
                let doc: toml::Table = content.parse()?;
                let provider_table = doc
                    .get("provider")
                    .and_then(toml::Value::as_table)
                    .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
                let Some(entries) = Self::get_failover_array(provider_table)? else {
                    return Ok(None);
                };
                if index >= entries.len() {
                    return Ok(None);
                }
                Ok(Some(Self::remove_failover_card(entries, index)))
            }
            _ => Ok(None),
        }
    }

    /// The STT stack `enable_stt` would write: the overrides resolved the
    /// same way the call resolves them.
    fn stt_probe(args: &ConfigArgs) -> aidaemon_tools::stt_setup::SttProbe {
        aidaemon_tools::stt_setup::SttProbe::resolve(
            args.stt_cli_path.as_deref(),
            args.stt_model_path.as_deref(),
            args.stt_ffmpeg_path.as_deref(),
        )
    }

    fn stt_language(args: &ConfigArgs) -> &str {
        args.stt_language
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("en")
    }

    /// The card for `enable_stt`: the programs that will run on every voice
    /// message, by full path.
    fn stt_summary(probe: &aidaemon_tools::stt_setup::SttProbe, language: &str) -> String {
        let show = |path: &Option<std::path::PathBuf>| {
            path.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        };
        format!(
            "Turn on voice transcription: every voice message runs {} (with ffmpeg {}) on the \
             model {}, language `{language}`. Applies after a restart.",
            show(&probe.whisper_cli),
            show(&probe.ffmpeg),
            show(&probe.model_path),
        )
    }

    /// The backup `restore` copies over the config: the first in the chain
    /// that exists, with its content.
    fn restore_candidate(backups: &[std::path::PathBuf]) -> Option<(std::path::PathBuf, String)> {
        backups.iter().find_map(|candidate| {
            std::fs::read_to_string(candidate)
                .ok()
                .map(|content| (candidate.clone(), content))
        })
    }

    fn restore_summary(path: &std::path::Path) -> String {
        format!(
            "Replace config.toml with the backup {}: every setting, including who may reach \
             the agent and what it may run, returns to what that backup holds. Applies after \
             /reload.",
            path.display()
        )
    }

    /// Whether an `unset` would remove an approval-guarded key: the value at
    /// `key` in the current config is a table or array holding one (say,
    /// unsetting `memory` drops `memory.db_path`).
    fn unset_removes_approval_key(content: &str, key: &str) -> bool {
        let Ok(path) = ConfigPath::parse(key) else {
            return true;
        };
        let Ok(root) = content.parse::<toml::Table>() else {
            return true;
        };
        let root = toml::Value::Table(root);
        plain_holds_approval_key(path.lookup(&root))
    }

    /// The exact arguments the owner approves (a changed argument is a new
    /// card), with secret values replaced by their SHA-256 digest so no
    /// secret is kept in a grant or a decline.
    fn approval_subject(&self, args: &ConfigArgs) -> String {
        use sha2::{Digest, Sha256};
        let secret = |value: &str| -> String {
            if value.is_empty() {
                String::new()
            } else {
                format!("sha256:{}", hex::encode(Sha256::digest(value.as_bytes())))
            }
        };
        let subject = match args.action.as_str() {
            "set" | "unset" => {
                let value = if is_sensitive_key(&args.key) {
                    secret(&args.value)
                } else {
                    args.value.clone()
                };
                json!({"action": args.action, "key": args.key, "value": value})
            }
            "remove_failover_provider" => json!({
                "action": args.action,
                "failover_index": args.failover_index,
            }),
            "move_failover_provider" => json!({
                "action": args.action,
                "failover_index": args.failover_index,
                "failover_to_index": args.failover_to_index,
            }),
            "enable_exposure" => json!({
                "action": args.action,
                "key": args.key,
                "value": args.value,
            }),
            // What runs for every voice message: the resolved programs and
            // model, so a different binary is a new card.
            "enable_stt" => {
                let probe = Self::stt_probe(args);
                json!({
                    "action": args.action,
                    "whisper_cli": probe.whisper_cli,
                    "ffmpeg": probe.ffmpeg,
                    "model_path": probe.model_path,
                    "language": Self::stt_language(args),
                })
            }
            // The backup's exact content: a backup that changed after the
            // card is a new card.
            "restore" => {
                let digest = Self::restore_candidate(&self.config.backups())
                    .map(|(_, content)| hex::encode(Sha256::digest(content.as_bytes())));
                json!({"action": args.action, "backup_sha256": digest})
            }
            _ => json!({
                "action": args.action,
                "provider": args.provider,
                "base_url": args.base_url,
                "primary_model": args.primary_model,
                "fast_model": args.fast_model,
                "smart_model": args.smart_model,
            }),
        };
        subject.to_string()
    }

    /// The owner approval for one call: the same changes the tool asked
    /// about before (security-sensitive `set`, provider switch, failover
    /// add/remove), with the same wording. If the config cannot be read to
    /// word the card, it still asks.
    pub async fn owner_approval(&self, arguments: &Value) -> OwnerApprovalNeed {
        let Ok(args) = serde_json::from_value::<ConfigArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let summary = match self.approval_summary(&args).await {
            Ok(Some(summary)) => summary,
            Ok(None) => return OwnerApprovalNeed::NotNeeded,
            Err(e) => format!(
                "Config change: manage_config {} (the config could not be read: {e})",
                args.action
            ),
        };
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary,
            subject: self.approval_subject(&args),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec!["Modifying security-sensitive configuration".to_string()],
        })
    }

    /// The typed effects of one call: every change writes the config file,
    /// a configuration effect on that path. A change may also store secrets
    /// in the keychain or redirect provider traffic, so the effects are
    /// declared incomplete. Reads have none.
    pub fn effects_for(&self, arguments: &str) -> aidaemon_core::CallEffects {
        use aidaemon_core::{CallEffects, EffectClass, EffectKey};
        let Ok(args) = serde_json::from_str::<ConfigArgs>(arguments) else {
            return CallEffects::unknown();
        };
        if matches!(
            args.action.as_str(),
            "read" | "get" | "list_provider_presets" | "list_failover_providers"
        ) {
            return CallEffects::none();
        }
        let mut effects = CallEffects::unknown();
        let config_key = std::path::absolute(self.config.path())
            .ok()
            .and_then(|path| {
                path.to_str()
                    .and_then(|path| EffectKey::path(EffectClass::Configuration, path))
            });
        if let Some(key) = config_key {
            effects.keys.push(key);
        }
        effects
    }

    async fn create_backup(&self) -> anyhow::Result<()> {
        self.config.backup().await
    }

    /// Restore from the first available backup in the chain.
    async fn restore_backup(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let Some((candidate, backup)) = Self::restore_candidate(&self.config.backups()) else {
            anyhow::bail!("No backups found (checked .bak, .bak.1, .bak.2)")
        };
        if let Err(e) = self.config.validate(&backup) {
            anyhow::bail!(
                "the backup {} is not a valid config: {e}",
                candidate.display()
            );
        }
        let current = self.config.read().await.unwrap_or_default();
        if let Some(refusal) = Self::writes_exposure_command(&current, &backup) {
            anyhow::bail!("{refusal}");
        }
        // The whole config, allowlists included, comes back: the kernel
        // asked the owner (with this backup's digest) before this runs.
        Self::require_kernel_approval(args)?;
        // Write the exact content judged above, not whatever the file holds
        // by the time a copy would run.
        self.config.write_checked(&backup).await?;
        warn!(source = %candidate.display(), "Config restored from backup");
        Ok(format!(
            "Config restored from {}. Run /reload to apply.",
            candidate.display()
        ))
    }

    /// Why a change may not be written: it would add or change an
    /// `[exposure.*]` table that runs a `command` adapter. That adapter runs
    /// any program, so only the owner configures it, by editing the file.
    /// Judged on the parsed config before and after, whatever path the
    /// change named.
    fn writes_exposure_command(old: &str, new: &str) -> Option<String> {
        let exposure = |text: &str| -> aidaemon_exposure::ExposureConfig {
            text.parse::<toml::Table>()
                .ok()
                .and_then(|table| table.get("exposure").cloned())
                .and_then(|value| value.try_into().ok())
                .unwrap_or_default()
        };
        let (before, after) = (exposure(old), exposure(new));
        let pairs = [
            (
                "client_gateway",
                &before.client_gateway,
                &after.client_gateway,
            ),
            ("node_gateway", &before.node_gateway, &after.node_gateway),
        ];
        for (service, before, after) in pairs {
            if matches!(
                after,
                Some(aidaemon_exposure::ServiceExposureConfig::Command { .. })
            ) && before != after
            {
                return Some(format!(
                    "Refused: `exposure.{service}` with adapter = \"command\" runs an arbitrary \
                     program, so only the owner sets it, by editing config.toml by hand. \
                     The config was NOT modified."
                ));
            }
        }
        None
    }

    fn normalize_provider_key(raw: &str) -> String {
        raw.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect()
    }

    fn find_provider_preset(provider: &str) -> Option<&'static ProviderPreset> {
        let needle = Self::normalize_provider_key(provider);
        PROVIDER_PRESETS.iter().find(|preset| {
            Self::normalize_provider_key(preset.id) == needle
                || Self::normalize_provider_key(preset.display_name) == needle
                || preset
                    .aliases
                    .iter()
                    .any(|alias| Self::normalize_provider_key(alias) == needle)
        })
    }

    fn list_provider_presets_message() -> String {
        let mut lines = vec![
            "Provider presets:".to_string(),
            "".to_string(),
            "Use action='switch_provider' to change the primary provider, or action='add_failover_provider' to append a fallback provider."
                .to_string(),
            "".to_string(),
        ];

        for preset in PROVIDER_PRESETS.iter() {
            let key_req = if preset.needs_api_key {
                "api_key required"
            } else {
                "api_key optional"
            };
            let base_req = if preset.requires_custom_base_url {
                "base_url required"
            } else if preset.base_url.is_empty() {
                "base_url not used"
            } else {
                "base_url preset"
            };
            lines.push(format!(
                "- {} (`{}`): kind=`{}`, {}, {}, default models=`{}` / `{}` / `{}`",
                preset.display_name,
                preset.id,
                preset.kind,
                key_req,
                base_req,
                preset.primary,
                preset.fast,
                preset.smart
            ));
        }

        lines.push("".to_string());
        lines.push(
            "Example: {\"action\":\"switch_provider\",\"provider\":\"moonshot\"}".to_string(),
        );
        lines.push(
            "Failover example: {\"action\":\"add_failover_provider\",\"provider\":\"anthropic\"}"
                .to_string(),
        );
        lines.push(
            "API keys are never tool arguments: the owner stores them with `aidaemon keychain set <key>` in their own terminal (the reply names the key)."
                .to_string(),
        );

        lines.join("\n")
    }

    fn resolve_secret_config_value(
        &self,
        keychain_key: &str,
        secret: &str,
        save_to_keychain: bool,
    ) -> (toml::Value, bool, Option<String>) {
        if !save_to_keychain {
            return (toml::Value::String(secret.to_string()), false, None);
        }

        match self.secrets.set(keychain_key, secret) {
            Ok(()) => (toml::Value::String("keychain".to_string()), true, None),
            Err(e) => (
                toml::Value::String(secret.to_string()),
                false,
                Some(format!(
                    "Could not store `{}` in keychain ({}). Saved inline in config instead.",
                    keychain_key, e
                )),
            ),
        }
    }

    /// The provider a `switch_provider` / `add_failover_provider` call
    /// selects, for the slot whose secret-store keys start with `key_prefix`
    /// (`None` for the primary provider). Secrets are never arguments: the
    /// API key must already be in the secret store under that slot's key.
    fn resolve_provider_selection(
        &self,
        args: &ConfigArgs,
        action_name: &str,
        key_prefix: Option<&str>,
    ) -> Result<ResolvedProviderSelection, String> {
        let provider_name = args.provider.trim();
        if provider_name.is_empty() {
            return Err(format!(
                "Error: 'provider' is required for action='{}'. Use action='list_provider_presets' to see valid options.",
                action_name
            ));
        }

        let Some(preset) = Self::find_provider_preset(provider_name) else {
            return Err(format!(
                "Unknown provider '{}'. Use action='list_provider_presets' to see valid options.",
                provider_name
            ));
        };

        let api_key_name = Self::keychain_field_name(key_prefix, "api_key");
        let gateway_token_name = Self::keychain_field_name(key_prefix, "gateway_token");
        if args.api_key.is_some() || args.gateway_token.is_some() {
            let mut names = vec![api_key_name.as_str()];
            if preset.supports_gateway_token {
                names.push(gateway_token_name.as_str());
            }
            return Err(format!(
                "Nothing was changed: secrets are not accepted as tool arguments. {} \
                 Then call {} again without the secret.",
                Self::owner_secret_instruction(&names, key_prefix.is_none()),
                action_name
            ));
        }
        let api_key_stored = self.secrets.contains(&api_key_name);
        if preset.needs_api_key && !api_key_stored {
            return Err(format!(
                "No API key for {} in the secret store. {} Then call {} again.",
                preset.display_name,
                Self::owner_secret_instruction(&[&api_key_name], key_prefix.is_none()),
                action_name
            ));
        }

        let mut base_url = args.base_url.trim().to_string();
        if base_url.is_empty() {
            base_url = preset.base_url.to_string();
        }
        if preset.requires_custom_base_url && base_url.trim().is_empty() {
            return Err(format!(
                "Provider {} requires `base_url`. Example:\n{{\"action\":\"{}\",\"provider\":\"{}\",\"base_url\":\"https://api.example.com/v1\",\"api_key\":\"YOUR_KEY\"}}",
                preset.display_name, action_name, preset.id
            ));
        }
        if preset.id == "cloudflare_gateway"
            && (base_url.contains("<ACCOUNT_ID>") || base_url.contains("<GATEWAY_ID>"))
        {
            return Err(
                "Cloudflare AI Gateway requires your real gateway URL. Replace `<ACCOUNT_ID>` and `<GATEWAY_ID>` in `base_url` and retry."
                    .to_string(),
            );
        }

        let primary_model = if args.primary_model.trim().is_empty() {
            preset.primary.to_string()
        } else {
            args.primary_model.trim().to_string()
        };
        let fast_model = if args.fast_model.trim().is_empty() {
            preset.fast.to_string()
        } else {
            args.fast_model.trim().to_string()
        };
        let smart_model = if args.smart_model.trim().is_empty() {
            preset.smart.to_string()
        } else {
            args.smart_model.trim().to_string()
        };

        if primary_model.is_empty() {
            return Err(format!(
                "{} has no model to start with: name one with `primary_model`.",
                preset.display_name
            ));
        }
        // With no model of their own, the other jobs use the one named.
        let fast_model = if fast_model.is_empty() {
            primary_model.clone()
        } else {
            fast_model
        };
        let smart_model = if smart_model.is_empty() {
            primary_model.clone()
        } else {
            smart_model
        };

        let gateway_token_stored =
            preset.supports_gateway_token && self.secrets.contains(&gateway_token_name);

        Ok(ResolvedProviderSelection {
            preset,
            api_key_stored,
            base_url,
            primary_model,
            fast_model,
            smart_model,
            gateway_token_stored,
        })
    }

    /// What the owner does to put secrets `names` in the secret store, in
    /// their own terminal; `/setkey` also covers the primary provider's key.
    fn owner_secret_instruction(names: &[&str], primary_provider: bool) -> String {
        let commands = names
            .iter()
            .map(|name| format!("`aidaemon keychain set {name}`"))
            .collect::<Vec<_>>()
            .join(" and ");
        let setkey = if primary_provider {
            " (or /setkey on Telegram)"
        } else {
            ""
        };
        format!(
            "Ask the owner to run {commands} in their own terminal{setkey}; never ask them to \
             paste a secret in chat."
        )
    }

    fn fallback_key_prefix(index: usize) -> String {
        aidaemon_secrets::names::provider_fallback_prefix(None, index)
    }

    fn keychain_field_name(prefix: Option<&str>, field: &str) -> String {
        aidaemon_secrets::names::provider_field(prefix, field)
    }

    fn header_keychain_field_name(prefix: Option<&str>, header_name: &str) -> String {
        aidaemon_secrets::names::provider_header(prefix, header_name)
    }

    fn build_models_table(selection: &ResolvedProviderSelection) -> toml::Table {
        let mut models = toml::Table::new();
        models.insert(
            "default".to_string(),
            toml::Value::String(selection.primary_model.clone()),
        );

        let fallback_models = selection.fallback_models();
        if !fallback_models.is_empty() {
            models.insert(
                "fallback".to_string(),
                toml::Value::Array(
                    fallback_models
                        .into_iter()
                        .map(toml::Value::String)
                        .collect(),
                ),
            );
        }

        models.insert(
            "primary".to_string(),
            toml::Value::String(selection.primary_model.clone()),
        );
        models.insert(
            "fast".to_string(),
            toml::Value::String(selection.fast_model.clone()),
        );
        models.insert(
            "smart".to_string(),
            toml::Value::String(selection.smart_model.clone()),
        );
        models
    }

    fn apply_provider_selection(
        &self,
        provider_table: &mut toml::Table,
        selection: &ResolvedProviderSelection,
        key_prefix: Option<&str>,
        notes: &mut Vec<String>,
        note_prefix: &str,
    ) {
        provider_table.insert(
            "kind".to_string(),
            toml::Value::String(selection.preset.kind.to_string()),
        );

        let keychain = || toml::Value::String("keychain".to_string());
        let api_key_value = if selection.api_key_stored {
            notes.push(format!(
                "{}API key read from the secret store (`{}`).",
                if note_prefix.is_empty() {
                    String::new()
                } else {
                    format!("{} ", note_prefix)
                },
                Self::keychain_field_name(key_prefix, "api_key")
            ));
            keychain()
        } else {
            toml::Value::String(if selection.preset.kind == "openai_chatgpt" {
                // Subscription auth: an "ollama" placeholder here would be
                // actively misleading about where the credentials live.
                String::new()
            } else {
                "ollama".to_string()
            })
        };
        provider_table.insert("api_key".to_string(), api_key_value);

        if selection.base_url.trim().is_empty() {
            provider_table.remove("base_url");
        } else {
            provider_table.insert(
                "base_url".to_string(),
                toml::Value::String(selection.base_url.clone()),
            );
        }

        provider_table.insert(
            "models".to_string(),
            toml::Value::Table(Self::build_models_table(selection)),
        );

        if selection.gateway_token_stored {
            provider_table.insert("gateway_token".to_string(), keychain());
        } else {
            provider_table.remove("gateway_token");
            if selection.preset.supports_gateway_token {
                notes.push(format!(
                    "No gateway token in the secret store. To add one: {} Then repeat this call.",
                    Self::owner_secret_instruction(
                        &[&Self::keychain_field_name(key_prefix, "gateway_token")],
                        false,
                    )
                ));
            }
        }
    }

    fn summarize_provider_table(index: usize, provider_table: &toml::Table) -> String {
        let kind = provider_table
            .get("kind")
            .and_then(toml::Value::as_str)
            .unwrap_or("(missing)");
        let base_url = provider_table
            .get("base_url")
            .and_then(toml::Value::as_str)
            .filter(|v| !v.trim().is_empty())
            .unwrap_or("(not set)");
        let api_key_state = match provider_table.get("api_key").and_then(toml::Value::as_str) {
            Some("keychain") => "keychain",
            Some(value) if !value.trim().is_empty() => "inline",
            _ => "missing",
        };
        let gateway_token_state = match provider_table
            .get("gateway_token")
            .and_then(toml::Value::as_str)
        {
            Some("keychain") => "keychain",
            Some(value) if !value.trim().is_empty() => "inline",
            _ => "not set",
        };

        let (default_model, fallback_models) = provider_table
            .get("models")
            .and_then(toml::Value::as_table)
            .map(Self::model_summary_from_table)
            .unwrap_or_else(|| ("(missing)".to_string(), Vec::new()));
        let fallback_summary = if fallback_models.is_empty() {
            "(none)".to_string()
        } else {
            fallback_models.join(", ")
        };

        format!(
            "[{}] kind=`{}`, base_url=`{}`, default=`{}`, fallback=`{}`, api_key=`{}`, gateway_token=`{}`",
            index,
            kind,
            base_url,
            default_model,
            fallback_summary,
            api_key_state,
            gateway_token_state
        )
    }

    fn model_summary_from_table(models: &toml::Table) -> (String, Vec<String>) {
        let default_model = models
            .get("default")
            .and_then(toml::Value::as_str)
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                models
                    .get("primary")
                    .and_then(toml::Value::as_str)
                    .filter(|v| !v.trim().is_empty())
            })
            .unwrap_or("(missing)")
            .to_string();

        let mut fallback_models = Vec::new();
        if let Some(array) = models.get("fallback").and_then(toml::Value::as_array) {
            for value in array {
                let Some(model) = value.as_str() else {
                    continue;
                };
                let trimmed = model.trim();
                if trimmed.is_empty()
                    || trimmed == default_model
                    || fallback_models
                        .iter()
                        .any(|existing: &String| existing == trimmed)
                {
                    continue;
                }
                fallback_models.push(trimmed.to_string());
            }
        } else {
            for key in ["smart", "fast"] {
                let Some(model) = models.get(key).and_then(toml::Value::as_str) else {
                    continue;
                };
                let trimmed = model.trim();
                if trimmed.is_empty()
                    || trimmed == default_model
                    || fallback_models
                        .iter()
                        .any(|existing: &String| existing == trimmed)
                {
                    continue;
                }
                fallback_models.push(trimmed.to_string());
            }
        }

        (default_model, fallback_models)
    }

    fn get_failover_array(
        provider_table: &toml::Table,
    ) -> anyhow::Result<Option<&Vec<toml::Value>>> {
        match (
            provider_table.get("fallbacks"),
            provider_table.get("failover"),
        ) {
            (Some(_), Some(_)) => {
                anyhow::bail!("Config contains both `provider.fallbacks` and `provider.failover`")
            }
            (Some(value), None) => value
                .as_array()
                .map(Some)
                .ok_or_else(|| anyhow::anyhow!("`provider.fallbacks` is not an array")),
            (None, Some(value)) => value
                .as_array()
                .map(Some)
                .ok_or_else(|| anyhow::anyhow!("`provider.failover` is not an array")),
            (None, None) => Ok(None),
        }
    }

    fn normalize_failover_array_mut(
        provider_table: &mut toml::Table,
    ) -> anyhow::Result<&mut Vec<toml::Value>> {
        if provider_table.contains_key("fallbacks") && provider_table.contains_key("failover") {
            anyhow::bail!("Config contains both `provider.fallbacks` and `provider.failover`");
        }
        if !provider_table.contains_key("fallbacks") {
            if let Some(value) = provider_table.remove("failover") {
                provider_table.insert("fallbacks".to_string(), value);
            }
        } else {
            provider_table.remove("failover");
        }
        provider_table
            .entry("fallbacks")
            .or_insert_with(|| toml::Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| anyhow::anyhow!("`provider.fallbacks` is not an array"))
    }

    fn normalize_failover_array_mut_if_present(
        provider_table: &mut toml::Table,
    ) -> anyhow::Result<Option<&mut Vec<toml::Value>>> {
        if provider_table.contains_key("fallbacks") && provider_table.contains_key("failover") {
            anyhow::bail!("Config contains both `provider.fallbacks` and `provider.failover`");
        }
        if !provider_table.contains_key("fallbacks") {
            if let Some(value) = provider_table.remove("failover") {
                provider_table.insert("fallbacks".to_string(), value);
            }
        } else {
            provider_table.remove("failover");
        }
        Ok(provider_table
            .get_mut("fallbacks")
            .and_then(toml::Value::as_array_mut))
    }

    fn migrate_secret_string(
        &self,
        provider_table: &mut toml::Table,
        field_name: &str,
        old_key: &str,
        new_key: &str,
        cleanup_keys: &mut Vec<String>,
        notes: &mut Vec<String>,
    ) -> anyhow::Result<()> {
        if old_key == new_key {
            return Ok(());
        }

        let Some(current_value) = provider_table.get(field_name).and_then(toml::Value::as_str)
        else {
            return Ok(());
        };
        if current_value != "keychain" {
            return Ok(());
        }

        let secret = self.secrets.get(old_key)?;
        let (new_value, _stored, note) =
            self.resolve_secret_config_value(new_key, secret.expose(), true);
        provider_table.insert(field_name.to_string(), new_value);
        cleanup_keys.push(old_key.to_string());
        if let Some(note) = note {
            notes.push(note);
        }
        Ok(())
    }

    fn migrate_secret_headers(
        &self,
        provider_table: &mut toml::Table,
        old_prefix: &str,
        new_prefix: &str,
        cleanup_keys: &mut Vec<String>,
        notes: &mut Vec<String>,
    ) -> anyhow::Result<()> {
        if old_prefix == new_prefix {
            return Ok(());
        }

        let Some(headers) = provider_table
            .get_mut("extra_headers")
            .and_then(toml::Value::as_table_mut)
        else {
            return Ok(());
        };

        for (header_name, header_value) in headers.iter_mut() {
            let Some(current_value) = header_value.as_str() else {
                continue;
            };
            if current_value != "keychain" {
                continue;
            }

            let old_key = Self::header_keychain_field_name(Some(old_prefix), header_name);
            let new_key = Self::header_keychain_field_name(Some(new_prefix), header_name);
            if old_key == new_key {
                continue;
            }

            let secret = self.secrets.get(&old_key)?;
            let (new_value, _stored, note) =
                self.resolve_secret_config_value(&new_key, secret.expose(), true);
            *header_value = new_value;
            cleanup_keys.push(old_key);
            if let Some(note) = note {
                notes.push(note);
            }
        }

        Ok(())
    }

    fn rekey_provider_table(
        &self,
        provider_table: &toml::Table,
        old_prefix: &str,
        new_prefix: &str,
        cleanup_keys: &mut Vec<String>,
        notes: &mut Vec<String>,
    ) -> anyhow::Result<toml::Table> {
        let mut migrated = provider_table.clone();

        self.migrate_secret_string(
            &mut migrated,
            "api_key",
            &Self::keychain_field_name(Some(old_prefix), "api_key"),
            &Self::keychain_field_name(Some(new_prefix), "api_key"),
            cleanup_keys,
            notes,
        )?;
        self.migrate_secret_string(
            &mut migrated,
            "gateway_token",
            &Self::keychain_field_name(Some(old_prefix), "gateway_token"),
            &Self::keychain_field_name(Some(new_prefix), "gateway_token"),
            cleanup_keys,
            notes,
        )?;
        self.migrate_secret_headers(&mut migrated, old_prefix, new_prefix, cleanup_keys, notes)?;

        if let Some(fallbacks) = Self::normalize_failover_array_mut_if_present(&mut migrated)? {
            for (idx, value) in fallbacks.iter_mut().enumerate() {
                let nested = value
                    .as_table()
                    .ok_or_else(|| anyhow::anyhow!("fallback provider entry is not a table"))?
                    .clone();
                let old_nested_prefix =
                    aidaemon_secrets::names::provider_fallback_prefix(Some(old_prefix), idx);
                let new_nested_prefix =
                    aidaemon_secrets::names::provider_fallback_prefix(Some(new_prefix), idx);
                *value = toml::Value::Table(self.rekey_provider_table(
                    &nested,
                    &old_nested_prefix,
                    &new_nested_prefix,
                    cleanup_keys,
                    notes,
                )?);
            }
        }

        Ok(migrated)
    }

    async fn list_failover_providers(&self) -> anyhow::Result<String> {
        let content = self.config.read().await?;
        let doc: toml::Table = content.parse()?;
        let Some(provider_table) = doc.get("provider").and_then(toml::Value::as_table) else {
            return Ok("Config has no [provider] section.".to_string());
        };

        let Some(fallbacks) = Self::get_failover_array(provider_table)? else {
            return Ok("No failover providers configured.".to_string());
        };
        if fallbacks.is_empty() {
            return Ok("No failover providers configured.".to_string());
        }

        let mut lines = vec![
            format!("{} failover provider(s) configured:", fallbacks.len()),
            String::new(),
        ];
        for (idx, value) in fallbacks.iter().enumerate() {
            let provider_table = value
                .as_table()
                .ok_or_else(|| anyhow::anyhow!("fallback provider entry is not a table"))?;
            lines.push(Self::summarize_provider_table(idx, provider_table));
        }
        lines.push(String::new());
        lines.push(
            "Use action='add_failover_provider' to append another, or action='remove_failover_provider' with `failover_index` to remove one."
                .to_string(),
        );
        Ok(lines.join("\n"))
    }

    async fn add_failover_provider(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let content = self.config.read().await?;
        let mut doc: toml::Table = content.parse()?;
        let provider_table = doc
            .entry("provider")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
        let next_index = Self::get_failover_array(provider_table)?
            .map(|entries| entries.len())
            .unwrap_or(0);
        let key_prefix = Self::fallback_key_prefix(next_index);
        let selection =
            match self.resolve_provider_selection(args, "add_failover_provider", Some(&key_prefix))
            {
                Ok(selection) => selection,
                Err(message) => return Ok(message),
            };

        Self::require_kernel_approval(args)?;

        let mut notes = Vec::new();
        let mut new_provider = toml::Table::new();
        self.apply_provider_selection(
            &mut new_provider,
            &selection,
            Some(&key_prefix),
            &mut notes,
            "Failover",
        );

        Self::normalize_failover_array_mut(provider_table)?.push(toml::Value::Table(new_provider));

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = self.config.validate(&new_content) {
            return Ok(format!(
                "Refused to save failover provider: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        self.config.write_checked(&new_content).await?;

        let mut response = vec![
            format!(
                "Added failover provider #{}: {}.",
                next_index, selection.preset.display_name
            ),
            format!("- kind: `{}`", selection.preset.kind),
            format!(
                "- base_url: `{}`",
                if selection.base_url.is_empty() {
                    "(not set)"
                } else {
                    &selection.base_url
                }
            ),
            format!(
                "- models: `{}` / `{}` / `{}`",
                selection.primary_model, selection.fast_model, selection.smart_model
            ),
        ];
        if !notes.is_empty() {
            response.push(String::new());
            response.push("Notes:".to_string());
            for note in notes {
                response.push(format!("- {}", note));
            }
        }
        response.push(String::new());
        response.push("Config validated and saved. Run `/reload` to apply.".to_string());

        Ok(response.join("\n"))
    }

    /// The failover entries `tables` put in `order` (`order[new] = old`;
    /// an old index left out is dropped), each entry's keychain secrets
    /// moved to its new slot. Every moved secret is read before any slot is
    /// written (through a slot of its own), so a swap or a shift never reads
    /// a secret another entry has just written. Returns the entries, the
    /// keychain entries no longer used, and notes.
    #[allow(clippy::type_complexity)]
    fn reindex_failovers(
        &self,
        tables: Vec<toml::Table>,
        order: &[usize],
    ) -> anyhow::Result<(Vec<toml::Value>, Vec<String>, Vec<String>)> {
        let mut notes = Vec::new();
        let mut cleanup = Vec::new();
        let moving = |old: usize| format!("provider_fallback_moving_{old}");
        let mut staged = Vec::with_capacity(order.len());
        for (new, &old) in order.iter().enumerate() {
            let table = tables
                .get(old)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("failover index {old} out of range"))?;
            staged.push(if old == new {
                table
            } else {
                self.rekey_provider_table(
                    &table,
                    &Self::fallback_key_prefix(old),
                    &moving(old),
                    &mut cleanup,
                    &mut notes,
                )?
            });
        }
        let mut entries = Vec::with_capacity(order.len());
        for ((new, &old), table) in order.iter().enumerate().zip(staged) {
            entries.push(toml::Value::Table(if old == new {
                table
            } else {
                self.rekey_provider_table(
                    &table,
                    &moving(old),
                    &Self::fallback_key_prefix(new),
                    &mut cleanup,
                    &mut notes,
                )?
            }));
        }
        // A slot an entry holds now keeps its secrets, whoever held it before.
        let held: Vec<String> = (0..entries.len())
            .map(|new| format!("{}_", Self::fallback_key_prefix(new)))
            .collect();
        cleanup.retain(|key| !held.iter().any(|slot| key.starts_with(slot.as_str())));
        Ok((entries, cleanup, notes))
    }

    fn delete_stale_keys(&self, mut keys: Vec<String>) {
        keys.sort();
        keys.dedup();
        for key in keys {
            if let Err(e) = self.secrets.delete(&key) {
                warn!(key = %key, error = %e, "Failed to delete stale failover keychain entry");
            }
        }
    }

    async fn move_failover_provider(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let (Some(index), Some(to)) = (args.failover_index, args.failover_to_index) else {
            return Ok(
                "Error: 'failover_index' and 'failover_to_index' are required for \
                       action='move_failover_provider'."
                    .to_string(),
            );
        };
        let content = self.config.read().await?;
        let mut doc: toml::Table = content.parse()?;
        let provider_table = doc
            .get_mut("provider")
            .and_then(toml::Value::as_table_mut)
            .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
        let (moved_summary, existing_tables) = {
            let Some(current) = Self::get_failover_array(provider_table)? else {
                return Ok("No failover providers configured.".to_string());
            };
            if index >= current.len() || to >= current.len() {
                return Ok(format!(
                    "Failover provider index out of range. Current count: {}.",
                    current.len()
                ));
            }
            let summary = Self::failover_label(current, index);
            let tables: Vec<toml::Table> = current
                .iter()
                .map(|value| {
                    value
                        .as_table()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("fallback provider entry is not a table"))
                })
                .collect::<anyhow::Result<_>>()?;
            (summary, tables)
        };
        if index == to {
            return Ok(format!("Failover provider #{index} is already there."));
        }
        // No approval: the order only chooses among providers the owner
        // already connected, and each keeps its own key.

        let mut order: Vec<usize> = (0..existing_tables.len()).filter(|&i| i != index).collect();
        order.insert(to, index);
        let (entries, cleanup_keys, notes) = self.reindex_failovers(existing_tables, &order)?;
        *Self::normalize_failover_array_mut(provider_table)? = entries;

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = self.config.validate(&new_content) {
            return Ok(format!(
                "Refused to move failover provider: {}.\n\nThe config was NOT modified.",
                e
            ));
        }
        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }
        self.config.write_checked(&new_content).await?;
        self.delete_stale_keys(cleanup_keys);

        let mut response = vec![format!(
            "Moved {moved_summary} to failover {}: it is tried {} now.",
            to + 1,
            if to < index { "sooner" } else { "later" }
        )];
        if !notes.is_empty() {
            response.push(String::new());
            response.push("Notes:".to_string());
            for note in notes {
                response.push(format!("- {}", note));
            }
        }
        response.push(String::new());
        response.push("Config validated and saved. Run `/reload` to apply.".to_string());
        Ok(response.join("\n"))
    }

    async fn remove_failover_provider(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let Some(index) = args.failover_index else {
            return Ok(
                "Error: 'failover_index' is required for action='remove_failover_provider'."
                    .to_string(),
            );
        };

        let content = self.config.read().await?;
        let mut doc: toml::Table = content.parse()?;
        let provider_table = doc
            .get_mut("provider")
            .and_then(toml::Value::as_table_mut)
            .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
        let (current_entries_len, removed_summary, existing_tables) = {
            let current_entries = match Self::get_failover_array(provider_table)? {
                Some(entries) => entries,
                None => return Ok("No failover providers configured.".to_string()),
            };
            if current_entries.is_empty() {
                return Ok("No failover providers configured.".to_string());
            }
            if index >= current_entries.len() {
                return Ok(format!(
                    "Failover provider index {} is out of range. Current count: {}.",
                    index,
                    current_entries.len()
                ));
            }

            let removed_summary = Self::failover_label(current_entries, index);
            let existing_tables: Vec<toml::Table> = current_entries
                .iter()
                .map(|value| {
                    value
                        .as_table()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("fallback provider entry is not a table"))
                })
                .collect::<anyhow::Result<_>>()?;
            (current_entries.len(), removed_summary, existing_tables)
        };
        Self::require_kernel_approval(args)?;

        let order: Vec<usize> = (0..existing_tables.len()).filter(|&i| i != index).collect();
        let (remaining_entries, cleanup_keys, notes) =
            self.reindex_failovers(existing_tables, &order)?;

        if remaining_entries.is_empty() {
            provider_table.remove("fallbacks");
            provider_table.remove("failover");
        } else {
            *Self::normalize_failover_array_mut(provider_table)? = remaining_entries;
        }

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = self.config.validate(&new_content) {
            return Ok(format!(
                "Refused to remove failover provider: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        self.config.write_checked(&new_content).await?;
        self.delete_stale_keys(cleanup_keys);

        let mut response = vec![format!(
            "Removed failover {}: {removed_summary}.",
            index + 1
        )];
        if index < current_entries_len - 1 {
            response.push("- remaining failover secret references were reindexed.".to_string());
        }
        if !notes.is_empty() {
            response.push(String::new());
            response.push("Notes:".to_string());
            for note in notes {
                response.push(format!("- {}", note));
            }
        }
        response.push(String::new());
        response.push("Config validated and saved. Run `/reload` to apply.".to_string());

        Ok(response.join("\n"))
    }

    /// Whether provider `table` is the one `selection` names: the same kind
    /// at the same address.
    fn same_provider(table: &toml::Table, selection: &ResolvedProviderSelection) -> bool {
        let field = |key: &str| {
            table
                .get(key)
                .and_then(toml::Value::as_str)
                .unwrap_or_default()
                .trim()
                .trim_end_matches('/')
                .to_string()
        };
        if field("kind") != selection.preset.kind {
            return false;
        }
        match field("base_url") {
            // A native kind's address is its own; an OpenAI-compatible one
            // with none is no particular service.
            url if url.is_empty() => selection.preset.kind != "openai_compatible",
            url => url == selection.base_url.trim().trim_end_matches('/'),
        }
    }

    /// Before `selection` replaces the main provider, put the main provider
    /// in the failover list: in the place of a failover that is the same
    /// provider as `selection` (the two swap), else first. Its API key goes
    /// with it, read from [`PROVIDER_OUTGOING_API_KEY`] (where the app puts
    /// it before a new key takes the main slot), else from the main slot
    /// when the new provider stores no key there. Nothing moves when the
    /// main provider is the one selected (only its models change), or there
    /// is none. Returns where it went with its label, and the secret-store
    /// names left unused.
    ///
    /// [`PROVIDER_OUTGOING_API_KEY`]: aidaemon_secrets::names::PROVIDER_OUTGOING_API_KEY
    #[allow(clippy::type_complexity)]
    fn demote_main_provider(
        &self,
        provider_table: &mut toml::Table,
        selection: &ResolvedProviderSelection,
        notes: &mut Vec<String>,
    ) -> anyhow::Result<(Option<(usize, String)>, Vec<String>)> {
        use aidaemon_secrets::names::PROVIDER_OUTGOING_API_KEY;
        let main: toml::Table = provider_table
            .iter()
            .filter(|(key, _)| key.as_str() != "fallbacks" && key.as_str() != "failover")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let has_kind = main
            .get("kind")
            .and_then(toml::Value::as_str)
            .is_some_and(|kind| !kind.trim().is_empty());
        if !has_kind || Self::same_provider(&main, selection) {
            return Ok((None, Vec::new()));
        }
        let existing: Vec<toml::Table> = match Self::get_failover_array(provider_table)? {
            Some(entries) => entries
                .iter()
                .map(|value| {
                    value
                        .as_table()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("fallback provider entry is not a table"))
                })
                .collect::<anyhow::Result<_>>()?,
            None => Vec::new(),
        };
        let count = existing.len();
        // Its secrets wait in the slot just past the list, then move with
        // the others to where it lands.
        let parked = Self::fallback_key_prefix(count);
        let mut cleanup = Vec::new();
        let mut demoted = main.clone();
        let keyed = demoted.get("api_key").and_then(toml::Value::as_str) == Some("keychain");
        let source = if self.secrets.contains(PROVIDER_OUTGOING_API_KEY) {
            Some(PROVIDER_OUTGOING_API_KEY)
        } else if !selection.api_key_stored && self.secrets.contains("api_key") {
            Some("api_key")
        } else {
            None
        };
        match (keyed, source) {
            (true, Some(source)) => self.migrate_secret_string(
                &mut demoted,
                "api_key",
                source,
                &Self::keychain_field_name(Some(&parked), "api_key"),
                &mut cleanup,
                notes,
            )?,
            (true, None) => notes.push(
                "The provider it replaced has no API key to keep: add its key again in \
                 Settings."
                    .to_string(),
            ),
            (false, _) => {}
        }
        if !selection.gateway_token_stored {
            self.migrate_secret_string(
                &mut demoted,
                "gateway_token",
                "gateway_token",
                &Self::keychain_field_name(Some(&parked), "gateway_token"),
                &mut cleanup,
                notes,
            )?;
        }
        self.migrate_secret_headers(&mut demoted, "provider", &parked, &mut cleanup, notes)?;
        // Its headers are its own: the new main provider does not inherit them.
        provider_table.remove("extra_headers");

        let promoted = existing
            .iter()
            .position(|table| Self::same_provider(table, selection));
        let order: Vec<usize> = match promoted {
            Some(at) => (0..count)
                .map(|i| if i == at { count } else { i })
                .collect(),
            None => std::iter::once(count).chain(0..count).collect(),
        };
        let landed = promoted.unwrap_or(0);
        let mut tables = existing;
        tables.push(demoted);
        let (entries, mut stale, mut moved_notes) = self.reindex_failovers(tables, &order)?;
        notes.append(&mut moved_notes);
        let label = Self::failover_label(&entries, landed);
        *Self::normalize_failover_array_mut(provider_table)? = entries;
        cleanup.append(&mut stale);
        // A slot a failover holds now, or the main slot the new provider
        // reads, keeps its secrets.
        let held: Vec<String> = (0..order.len())
            .map(|index| format!("{}_", Self::fallback_key_prefix(index)))
            .collect();
        cleanup.retain(|key| {
            !held.iter().any(|slot| key.starts_with(slot.as_str()))
                && !(key == "api_key" && selection.api_key_stored)
        });
        Ok((Some((landed, label)), cleanup))
    }

    async fn switch_provider(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let selection = match self.resolve_provider_selection(args, "switch_provider", None) {
            Ok(selection) => selection,
            Err(message) => return Ok(message),
        };

        Self::require_kernel_approval(args)?;

        let content = self.config.read().await?;
        let mut doc: toml::Table = content.parse()?;

        let mut notes: Vec<String> = Vec::new();
        let provider_table = doc
            .entry("provider")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
        // The main provider replaced is kept: it becomes a failover (in the
        // place of the failover promoted, else first), never dropped.
        let (demoted_to, cleanup_keys) =
            self.demote_main_provider(provider_table, &selection, &mut notes)?;
        self.apply_provider_selection(provider_table, &selection, None, &mut notes, "");

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;

        if let Err(e) = self.config.validate(&new_content) {
            return Ok(format!(
                "Refused to save provider switch: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        self.config.write_checked(&new_content).await?;
        self.delete_stale_keys(cleanup_keys);

        let mut response = vec![
            format!("Switched provider to {}.", selection.preset.display_name),
            format!("- kind: `{}`", selection.preset.kind),
            format!(
                "- base_url: `{}`",
                if selection.base_url.is_empty() {
                    "(not set)"
                } else {
                    &selection.base_url
                }
            ),
            format!(
                "- models: `{}` / `{}` / `{}`",
                selection.primary_model, selection.fast_model, selection.smart_model
            ),
        ];
        if let Some((index, label)) = demoted_to {
            response.push(format!(
                "- the provider it replaced, {label}, is now failover {}",
                index + 1
            ));
        }
        if !notes.is_empty() {
            response.push(String::new());
            response.push("Notes:".to_string());
            for note in notes {
                response.push(format!("- {}", note));
            }
        }
        response.push(String::new());
        response.push("Config validated and saved. Run `/reload` to apply.".to_string());

        Ok(response.join("\n"))
    }

    async fn enable_stt(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let probe = Self::stt_probe(args);

        let mut issues = Vec::new();
        if probe.whisper_cli.is_none() {
            issues.push("whisper-cli not found (install whisper.cpp or set stt_cli_path)");
        }
        if probe.ffmpeg.is_none() {
            issues.push("ffmpeg not found (install ffmpeg)");
        }
        if probe.model_path.is_none() {
            issues.push("Whisper model not found (download a GGML .bin or set stt_model_path)");
        }
        if !issues.is_empty() {
            let bullet_list = issues
                .iter()
                .map(|issue| format!("- {issue}"))
                .collect::<Vec<_>>()
                .join("\n");
            return Ok(format!(
                "Cannot enable STT:\n{bullet_list}\n\n{}",
                aidaemon_tools::stt_setup::format_stt_probe_report(&probe)
            ));
        }

        let language = Self::stt_language(args);
        // The programs written here run on every voice message: the kernel
        // showed the owner their paths before this runs.
        Self::require_kernel_approval(args)?;

        let content = self.config.read().await?;
        let mut doc: toml::Table = content.parse()?;
        aidaemon_tools::stt_setup::apply_stt_to_config_table(&mut doc, &probe, true, language)?;

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = self.config.validate(&new_content) {
            return Ok(format!(
                "Refused to enable STT: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        self.config.write_checked(&new_content).await?;

        Ok(format!(
            "Enabled Whisper STT fallback ([files.stt]).\n\n{}\n- language: `{language}`\n\nConfig validated and saved. Restart aidaemon (`/restart`) to apply — STT is loaded at startup.",
            aidaemon_tools::stt_setup::format_stt_probe_report(&probe)
        ))
    }

    /// The `[client_gateway]` table of `content`, defaults where unset.
    fn client_gateway_settings(content: &str) -> aidaemon_client_gateway::ClientGatewayConfig {
        content
            .parse::<toml::Table>()
            .ok()
            .and_then(|table| table.get("client_gateway").cloned())
            .and_then(|value| value.try_into().ok())
            .unwrap_or_default()
    }

    /// Turn on the Client Gateway (keeping its other settings) and mint the
    /// local token into the owner's copy, so `aidaemon tui` and
    /// `aidaemon web` work with no flags. The token never enters the reply.
    async fn enable_client_gateway(&self, args: &ConfigArgs) -> anyhow::Result<ConfigReply> {
        Self::require_kernel_approval(args)?;
        let content = self.config.read().await?;
        let mut doc: toml::Table = content.parse()?;
        let already_on = Self::client_gateway_settings(&content).enabled;
        let table = doc
            .entry("client_gateway")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        let Some(table) = table.as_table_mut() else {
            return Ok(ConfigReply::refused(
                "Refused: `client_gateway` in config.toml is not a table. The config was NOT \
                 modified."
                    .to_string(),
            ));
        };
        table.insert("enabled".to_string(), toml::Value::Boolean(true));
        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = self.config.validate(&new_content) {
            return Ok(ConfigReply::refused(format!(
                "Refused to enable the Client Gateway: {e}.\n\nThe config was NOT modified."
            )));
        }
        if !already_on {
            if let Err(e) = self.create_backup().await {
                warn!("Failed to create backup: {}", e);
            }
            self.config.write_checked(&new_content).await?;
        }
        let settings = Self::client_gateway_settings(&new_content);
        let token_path = match aidaemon_client_gateway::token::local_token(&settings) {
            Some(_) => aidaemon_client_gateway::token::client_token_file(&settings),
            None => aidaemon_client_gateway::token::mint_local_token(&settings)?,
        };
        let apply = if already_on {
            "It was already on in config.toml; if `aidaemon tui` cannot connect, restart \
             aidaemon (`/restart`)."
        } else {
            "Restart aidaemon (`/restart`) to apply: the gateway starts with the daemon."
        };
        Ok(ConfigReply::done(format!(
            "Client Gateway enabled ([client_gateway] enabled = true, loopback {}:{}). The local \
             token is saved to {} (owner-only; not shown here).\n\n{apply}\n\nThen, on this \
             machine:\n- `aidaemon tui` for the terminal client\n- `aidaemon web` to open and \
             pair the web client\n- `/pair_device` here to pair a phone or another browser",
            settings.bind,
            settings.port,
            token_path.display()
        )))
    }

    /// The service and adapter settings an `enable_exposure` call names:
    /// `key` is `client_gateway` or `node_gateway`, `value` an inline table
    /// such as `{ adapter = "tailscale", mode = "funnel" }` or
    /// `{ adapter = "cloudflare", hostname = "app.example.com", tunnel = "t" }`.
    /// `command` is never one of them: only the owner writes that, by hand.
    fn exposure_settings(
        args: &ConfigArgs,
    ) -> Result<
        (
            aidaemon_exposure::ExposedService,
            aidaemon_exposure::ServiceExposureConfig,
        ),
        String,
    > {
        use aidaemon_exposure::{ExposedService, ServiceExposureConfig};
        let service = match args.key.trim().trim_start_matches("exposure.") {
            "client_gateway" => ExposedService::ClientGateway,
            "node_gateway" => ExposedService::NodeGateway,
            other => {
                return Err(format!(
                    "`key` must be client_gateway or node_gateway, not `{other}`"
                ));
            }
        };
        let table: toml::Table = format!("v = {}", args.value.trim())
            .parse::<toml::Table>()
            .ok()
            .and_then(|t| t.get("v").and_then(|v| v.as_table()).cloned())
            .ok_or(
                "`value` must be an inline table, e.g. { adapter = \"tailscale\" } or \
                 { adapter = \"cloudflare\", hostname = \"app.example.com\", tunnel = \"name\" } \
                 or { adapter = \"manual\", url = \"https://...\" }",
            )?;
        if table.get("adapter").and_then(toml::Value::as_str) == Some("command") {
            return Err(
                "the command adapter runs an arbitrary program, so only the owner \
                        configures it, by editing `[exposure]` in config.toml by hand"
                    .to_string(),
            );
        }
        let settings: ServiceExposureConfig = toml::Value::Table(table)
            .try_into()
            .map_err(|e| format!("invalid exposure settings: {e}"))?;
        settings.validate(&service)?;
        Ok((service, settings))
    }

    /// The approval card: what is published, through what, and who can
    /// reach it.
    fn exposure_summary(
        service: &aidaemon_exposure::ExposedService,
        settings: &aidaemon_exposure::ServiceExposureConfig,
    ) -> String {
        let what = match service {
            aidaemon_exposure::ExposedService::NodeGateway => "the Node Gateway (/node/v1)",
            _ => "the Client Gateway (/client/v1: chat, watch turns, answer approvals)",
        };
        format!(
            "Publish {what} through {} ({}): the daemon runs the tunnel while it runs. \
             Paired devices still need their keys. Applies after a restart.",
            settings.adapter_name(),
            settings.audience().describe()
        )
    }

    /// Write `[exposure.<service>]` for a stable, owner-approved adapter.
    async fn enable_exposure(&self, args: &ConfigArgs) -> anyhow::Result<ConfigReply> {
        Self::require_kernel_approval(args)?;
        let (service, settings) = match Self::exposure_settings(args) {
            Ok(found) => found,
            Err(e) => {
                return Ok(ConfigReply::refused(format!(
                    "Refused: {e}. The config was NOT modified."
                )));
            }
        };
        let content = self.config.read().await?;
        let mut doc: toml::Table = content.parse()?;
        let value = toml::Value::try_from(&settings)?;
        let exposure = doc
            .entry("exposure")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        let Some(exposure) = exposure.as_table_mut() else {
            return Ok(ConfigReply::refused(
                "Refused: `exposure` in config.toml is not a table. The config was NOT modified."
                    .to_string(),
            ));
        };
        exposure.insert(service.key(), value);
        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = self.config.validate(&new_content) {
            let hint = match service {
                aidaemon_exposure::ExposedService::ClientGateway
                    if e.contains("client_gateway.enabled") =>
                {
                    " Turn the gateway on first (action 'enable_client_gateway')."
                }
                _ => "",
            };
            return Ok(ConfigReply::refused(format!(
                "Refused to enable exposure: {e}.{hint}\n\nThe config was NOT modified."
            )));
        }
        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }
        self.config.write_checked(&new_content).await?;
        Ok(ConfigReply::done(format!(
            "Exposure saved: [exposure.{service}] adapter = \"{}\" ({}). Backup saved to \
             config.toml.bak.\n\nRestart aidaemon (`/restart`) to apply: the tunnel starts with \
             the daemon and stops with it. Then `/exposure` shows its state and address.",
            settings.adapter_name(),
            settings.audience().describe()
        )))
    }
}

#[derive(Deserialize)]
struct ConfigArgs {
    action: String,
    #[serde(default)]
    key: String,
    #[serde(default)]
    value: String,
    #[serde(default)]
    provider: String,
    /// No longer a parameter: secrets never travel as tool arguments. Kept
    /// only to notice a model that still sends one; the value is dropped
    /// unread.
    #[serde(default)]
    api_key: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    base_url: String,
    /// Same as `api_key`.
    #[serde(default)]
    gateway_token: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    primary_model: String,
    #[serde(default)]
    fast_model: String,
    #[serde(default)]
    smart_model: String,
    #[serde(default)]
    failover_index: Option<usize>,
    /// Where `move_failover_provider` puts failover `failover_index`.
    #[serde(default)]
    failover_to_index: Option<usize>,
    #[serde(default)]
    stt_cli_path: Option<String>,
    #[serde(default)]
    stt_model_path: Option<String>,
    #[serde(default)]
    stt_language: Option<String>,
    #[serde(default)]
    stt_ffmpeg_path: Option<String>,
    /// The kernel judged this call's owner approval (from the dispatcher's
    /// facts, never an argument).
    #[serde(skip)]
    kernel_approved: bool,
}

impl ConfigArgs {
    fn change(action: &str, key: &str, value: &str) -> Self {
        Self {
            action: action.to_string(),
            key: key.to_string(),
            value: value.to_string(),
            provider: String::new(),
            api_key: None,
            base_url: String::new(),
            gateway_token: None,
            primary_model: String::new(),
            fast_model: String::new(),
            smart_model: String::new(),
            failover_index: None,
            failover_to_index: None,
            stt_cli_path: None,
            stt_model_path: None,
            stt_language: None,
            stt_ffmpeg_path: None,
            kernel_approved: false,
        }
    }
}

fn apply_instruction_for_key(key: &str) -> &'static str {
    if key == "agent" || key.starts_with("agent.") {
        "Restart aidaemon to apply agent identity or persona changes."
    } else {
        "Run /reload to apply changes."
    }
}

/// The tool's one description: the schema the model reads and
/// `description()` both use it.
const TOOL_DESCRIPTION: &str = "Read (secrets hidden)/update config.toml (backup+validate). key: dotted path, list entry as a[2]. Secrets are never args: owner runs `aidaemon keychain set <key>`; a secret setting takes value `keychain`.";

fn manage_config_schema() -> Value {
    json!({
        "name": "manage_config",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "get", "set", "unset", "restore", "list_provider_presets", "switch_provider", "list_failover_providers", "add_failover_provider", "remove_failover_provider", "move_failover_provider", "enable_stt", "enable_client_gateway", "enable_exposure"]
                },
                "key": { "type": "string" },
                "value": { "type": "string" },
                "provider": { "type": "string" },
                "base_url": { "type": "string" },
                "primary_model": { "type": "string" },
                "fast_model": { "type": "string" },
                "smart_model": { "type": "string" },
                "failover_index": { "type": "integer" },
                "failover_to_index": { "type": "integer", "description": "move_failover_provider: new position" },
                "stt_cli_path": { "type": "string" },
                "stt_model_path": { "type": "string" },
                "stt_language": { "type": "string" }
            },
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

#[async_trait]
impl Tool for ConfigManagerTool {
    fn name(&self) -> &str {
        "manage_config"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        manage_config_schema()
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        let facts = facts.clone();

        let reply = self.run_reply(arguments, &facts).await?;
        let mut outcome = aidaemon_tools::semantics::ToolCallOutcome::from_output(reply.text);
        outcome.metadata.outcome_status = reply.status;
        Ok(outcome)
    }
}

/// One call's reply. `status` is the tool's own verdict when it knows it
/// (reads, `get`, `set`, `unset`), so a config dump that happens to mention
/// timeouts is never read back as a failure.
struct ConfigReply {
    text: String,
    status: Option<aidaemon_tools::semantics::ToolOutcomeStatus>,
}

impl ConfigReply {
    fn done(text: String) -> Self {
        Self {
            text,
            status: Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded),
        }
    }

    fn refused(text: String) -> Self {
        Self {
            text,
            status: Some(aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent),
        }
    }
}

impl From<String> for ConfigReply {
    fn from(text: String) -> Self {
        Self { text, status: None }
    }
}

impl ConfigManagerTool {
    async fn run_reply(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ConfigReply> {
        let mut args: ConfigArgs = serde_json::from_str(arguments)?;
        args.kernel_approved = facts.kernel_approved.is_some();
        let reply = self.run_args(args).await;
        self.refresh_permissions().await;
        reply
    }

    /// Run one call for the owner from outside the agent loop (the owner's
    /// own client), through the same checks, backups and refusals as the
    /// tool. `approved`: the owner approved this exact call (what
    /// [`Self::owner_approval`] asked), as the kernel's approval would mean.
    /// `Ok` says what changed; `Err` why nothing was.
    pub async fn run_owner_call(
        &self,
        arguments: &Value,
        approved: bool,
    ) -> Result<String, String> {
        let mut args: ConfigArgs =
            serde_json::from_value(arguments.clone()).map_err(|e| e.to_string())?;
        args.kernel_approved = approved;
        let reply = self.run_args(args).await;
        self.refresh_permissions().await;
        match reply {
            Ok(ConfigReply {
                status: Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded) | None,
                text,
            }) => Ok(text),
            Ok(ConfigReply { text, .. }) => Err(text),
            Err(e) => Err(e.to_string()),
        }
    }

    /// Whether a `set` or `unset` of `key` names a security-sensitive
    /// setting (a secret, or one that redirects traffic or widens access).
    pub fn is_guarded_key(key: &str) -> bool {
        Self::change_needs_approval(&ConfigArgs::change("unset", key, ""))
    }

    /// Whether `key` holds a secret, kept in the secret store.
    pub fn is_secret_key(key: &str) -> bool {
        is_sensitive_key(key)
    }

    async fn run_args(&self, args: ConfigArgs) -> anyhow::Result<ConfigReply> {
        match args.action.as_str() {
            "list_provider_presets" => Ok(ConfigReply::done(Self::list_provider_presets_message())),
            "switch_provider" => self.switch_provider(&args).await.map(Into::into),
            "list_failover_providers" => {
                self.list_failover_providers().await.map(ConfigReply::done)
            }
            "add_failover_provider" => self.add_failover_provider(&args).await.map(Into::into),
            "remove_failover_provider" => {
                self.remove_failover_provider(&args).await.map(Into::into)
            }
            "move_failover_provider" => self.move_failover_provider(&args).await.map(Into::into),
            "enable_stt" => self.enable_stt(&args).await.map(Into::into),
            "enable_client_gateway" => self.enable_client_gateway(&args).await,
            "enable_exposure" => self.enable_exposure(&args).await,
            "read" => {
                let content = self.config.read().await?;
                let mut doc: toml::Value = toml::from_str(&content)?;
                redact_secrets(&mut doc);
                let redacted = toml::to_string_pretty(&doc)?;
                Ok(ConfigReply::done(format!(
                    "Current config.toml at {} (secrets redacted):\n\n{}",
                    self.config.path().display(),
                    redacted
                )))
            }
            "get" => self.get_value(&args).await,
            "set" | "unset" => self.change_value(&args).await,
            "restore" => match self.restore_backup(&args).await {
                Ok(msg) => Ok(ConfigReply::done(msg)),
                Err(e) => Ok(ConfigReply::refused(format!("Restore failed: {}", e))),
            },
            _ => Ok(ConfigReply::refused(format!(
                "Unknown action: {}. Use 'read', 'get', 'set', 'unset', 'restore', 'list_provider_presets', 'switch_provider', 'list_failover_providers', 'add_failover_provider', 'remove_failover_provider', 'enable_stt', 'enable_client_gateway', or 'enable_exposure'.",
                args.action
            ))),
        }
    }

    async fn get_value(&self, args: &ConfigArgs) -> anyhow::Result<ConfigReply> {
        if args.key.is_empty() {
            return Ok(ConfigReply::refused(
                "Error: 'key' is required for 'get' action.".to_string(),
            ));
        }
        let path = match ConfigPath::parse(&args.key) {
            Ok(path) => path,
            Err(e) => return Ok(ConfigReply::refused(format!("Error: {e}."))),
        };
        if is_sensitive_key(&args.key) {
            return Ok(ConfigReply::done(format!(
                "{path} = \"{REDACTED}\" (redacted for security)"
            )));
        }
        let content = self.config.read().await?;
        let doc: toml::Value = toml::from_str(&content)?;
        match path.lookup(&doc) {
            Some(value) => {
                let mut value = value.clone();
                redact_secrets(&mut value);
                Ok(ConfigReply::done(format!("{path} = {value}")))
            }
            None => Ok(ConfigReply::refused(format!(
                "config.toml has no setting `{path}`."
            ))),
        }
    }

    /// `set` or `unset` one setting. The change is written only when the
    /// daemon's config would read it (a key it ignores is refused), and it is
    /// reported saved only after the file read back from disk holds it.
    async fn change_value(&self, args: &ConfigArgs) -> anyhow::Result<ConfigReply> {
        if let Some(refusal) = Self::literal_secret_refusal(args) {
            return Ok(ConfigReply::refused(refusal));
        }
        // The config expands `${NAME}` from the daemon's environment, which
        // holds its secrets: a value written here must not pull one into a
        // URL or any other setting.
        if args.action == "set" && args.value.contains("${") {
            return Ok(ConfigReply::refused(
                "Nothing was changed: a value may not reference an environment variable \
                 (`${...}`). Secrets take the value `keychain`; the owner can add an \
                 environment reference by editing the config file themselves."
                    .to_string(),
            ));
        }
        let unset = args.action == "unset";
        if args.key.is_empty() || (!unset && args.value.is_empty()) {
            return Ok(ConfigReply::refused(if unset {
                "Error: 'key' is required for 'unset' action.".to_string()
            } else {
                "Error: 'key' and 'value' are required for 'set' action.".to_string()
            }));
        }
        let path = match ConfigPath::parse(&args.key) {
            Ok(path) => path,
            Err(e) => {
                return Ok(ConfigReply::refused(format!(
                    "Error: {e}. The config was NOT modified."
                )));
            }
        };

        let content = self.config.read().await?;
        let mut doc: toml_edit::DocumentMut = content.parse()?;
        let new_value = (!unset).then(|| config_path::parse_value(&args.value));
        let edited = match &new_value {
            Some(value) => path.set(&mut doc, value.clone()),
            None => path.unset(&mut doc),
        };
        if let Err(e) = edited {
            return Ok(ConfigReply::refused(format!(
                "Refused: {e}. The config was NOT modified."
            )));
        }
        let new_content = doc.to_string();

        if let Err(e) = self.config.validate(&new_content) {
            return Ok(ConfigReply::refused(format!(
                "Refused to save: {}.\n\nThe config was NOT modified. Fix the value and try again.",
                e
            )));
        }
        if let Some(refusal) = Self::writes_exposure_command(&content, &new_content) {
            return Ok(ConfigReply::refused(refusal));
        }
        if let Some(value) = &new_value
            && !path.ends_in_index()
            && !self.config_reads(&doc, &path, value)
        {
            return Ok(ConfigReply::refused(format!(
                "Refused: aidaemon does not read a setting at `{path}`, so it would have no effect. \
                     The config was NOT modified. Check the key name with action 'read'."
            )));
        }

        // A change that would be written needs the owner's approval unless
        // it is in a tuning table; the kernel asked before this runs. (A
        // change refused above writes nothing and needs none.)
        if Self::change_needs_approval(args)
            || (unset && Self::unset_removes_approval_key(&content, &args.key))
        {
            Self::require_kernel_approval(args)?;
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
            // Continue anyway — better to fix config than refuse
        }
        self.config.write_checked(&new_content).await?;

        // What a reload will load: the file as it is on disk now.
        let written = self.config.read().await?;
        if let Err(e) = self.config.validate(&written) {
            return Ok(ConfigReply::refused(format!(
                "config.toml was written but no longer loads ({e}). The change is not in effect; \
                 use action 'restore' to bring back the backup."
            )));
        }
        let on_disk: toml::Value = toml::from_str(&written)?;
        let found = path.lookup(&on_disk);
        let apply_instruction = apply_instruction_for_key(&args.key);
        match &new_value {
            Some(value) => {
                let expected = config_path::plain_value(value);
                if found.is_none() || found != expected.as_ref() {
                    return Ok(ConfigReply::refused(format!(
                        "config.toml was written, but reading it back does not show `{path}` = {value}. \
                         The change is not in effect."
                    )));
                }
                let shown = if is_sensitive_key(&args.key) {
                    format!("\"{REDACTED}\"")
                } else {
                    value.to_string()
                };
                Ok(ConfigReply::done(format!(
                    "Updated {path} = {shown}\n\nBackup saved to config.toml.bak. Checked: the saved \
                     config loads and holds the new value. {apply_instruction}"
                )))
            }
            None => {
                if found.is_some() {
                    return Ok(ConfigReply::refused(format!(
                        "config.toml was written, but `{path}` is still there. The change is not in effect."
                    )));
                }
                Ok(ConfigReply::done(format!(
                    "Removed {path}.\n\nBackup saved to config.toml.bak. Checked: the saved config \
                     loads without it. {apply_instruction}"
                )))
            }
        }
    }

    /// Whether the daemon's config reads the setting at `path`: the same
    /// document with a value of the wrong shape there must fail to load. A
    /// key the config ignores loads either way.
    fn config_reads(
        &self,
        doc: &toml_edit::DocumentMut,
        path: &ConfigPath,
        value: &toml_edit::Value,
    ) -> bool {
        let mut probe = doc.clone();
        if path
            .set(&mut probe, config_path::probe_value(value))
            .is_err()
        {
            return true;
        }
        self.config.validate(&probe.to_string()).is_err()
    }

    /// A change needs the owner when it names a security-sensitive key, or
    /// replaces a table or list entry that holds one.
    fn change_needs_approval(args: &ConfigArgs) -> bool {
        // The Client Gateway opens a way to chat and answer approvals from
        // outside a channel: every key of its table is the owner's call.
        // Exposure publishes a gateway outside this machine: the same.
        // So is everything outside the few tables that only tune how the
        // agent talks and thinks: an allowlist of free tables, so a table
        // that grants access or sends data somewhere is never free because
        // nobody listed it.
        // Which model a provider runs on is the owner's call: the agent
        // asks before moving it (say, back from an automatic upgrade), so a
        // misread request never switches it.
        if is_provider_model_key(&args.key) {
            return true;
        }
        let table = args.key.split(['.', '[']).next().map(str::trim);
        if !table.is_some_and(|table| APPROVAL_FREE_TABLES.contains(&table)) {
            return true;
        }
        let Ok(path) = ConfigPath::parse(&args.key) else {
            return Self::requires_approval(&args.key);
        };
        if let Some(key) = path.last_key()
            && APPROVAL_REQUIRED_KEYS.contains(&key)
        {
            return true;
        }
        if args.action == "unset" {
            return path.last_key().is_none();
        }
        plain_holds_approval_key(
            config_path::plain_value(&config_path::parse_value(&args.value)).as_ref(),
        )
    }
}

/// `provider.models.default` (or legacy `primary`), on the main provider
/// or a fallback (`provider.fallbacks[1].models.default`).
fn is_provider_model_key(key: &str) -> bool {
    let key: String = key
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    key.starts_with("provider")
        && (key.ends_with("models.default") || key.ends_with("models.primary"))
}

fn plain_holds_approval_key(value: Option<&toml::Value>) -> bool {
    match value {
        Some(toml::Value::Table(table)) => table.iter().any(|(key, value)| {
            APPROVAL_REQUIRED_KEYS.contains(&key.as_str()) || plain_holds_approval_key(Some(value))
        }),
        Some(toml::Value::Array(items)) => items
            .iter()
            .any(|item| plain_holds_approval_key(Some(item))),
        _ => false,
    }
}

#[async_trait]
impl aidaemon_tools::semantics::ToolSemantics for ConfigManagerTool {
    /// Security-sensitive changes ask the owner, whatever the session. The
    /// dispatcher runs the call only with that answer; a decline is recorded
    /// by the kernel.
    async fn owner_approval_request(
        &self,
        call: &aidaemon_tools::semantics::ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        self.owner_approval(call.arguments).await
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        self.effects_for(arguments)
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(
            arguments,
            &[
                "read",
                "get",
                "list_provider_presets",
                "list_failover_providers",
            ],
            ToolMutationEffects::CONFIGURATION,
        )
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }
}

/// Remove a value from a TOML table at a dotted key path. Missing paths are ignored.
#[cfg(test)]
fn remove_toml_value(table: &mut toml::Table, path: &str) -> anyhow::Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        anyhow::bail!("Empty key path");
    }

    let mut current = table;
    for &key in &parts[..parts.len() - 1] {
        let Some(next) = current.get_mut(key).and_then(|v| v.as_table_mut()) else {
            return Ok(());
        };
        current = next;
    }

    if let Some(last_key) = parts.last() {
        current.remove(*last_key);
    }
    Ok(())
}

/// The secret-store key a provider secret at `path` is read from
/// (`provider.api_key` → `api_key`, `provider.fallbacks[0].api_key` →
/// `provider_fallback_0_api_key`), and whether it is the primary provider's.
/// `None` for any other path.
fn provider_secret_name(path: &ConfigPath) -> Option<(String, bool)> {
    use config_path::PathSegment;
    let segments = path.segments();
    let (last, chain) = segments.split_last()?;
    let PathSegment::Key(field) = last else {
        return None;
    };
    if !matches!(field.as_str(), "api_key" | "gateway_token") {
        return None;
    }
    let (first, mut rest) = chain.split_first()?;
    if !matches!(first, PathSegment::Key(key) if key == "provider") {
        return None;
    }
    let mut prefix: Option<String> = None;
    while let [PathSegment::Key(list), PathSegment::Index(index), tail @ ..] = rest {
        if !matches!(list.as_str(), "fallbacks" | "failover") {
            return None;
        }
        prefix = Some(aidaemon_secrets::names::provider_fallback_prefix(
            prefix.as_deref(),
            *index,
        ));
        rest = tail;
    }
    if !rest.is_empty() {
        return None;
    }
    let primary = prefix.is_none();
    Some((
        aidaemon_secrets::names::provider_field(prefix.as_deref(), field),
        primary,
    ))
}

/// Check if the last segment of a dotted key path is a sensitive key name.
fn is_sensitive_key(path: &str) -> bool {
    match ConfigPath::parse(path) {
        Ok(path) => {
            let keys: Vec<&str> = path
                .segments()
                .iter()
                .filter_map(|segment| match segment {
                    config_path::PathSegment::Key(key) => Some(key.as_str()),
                    config_path::PathSegment::Index(_) => None,
                })
                .collect();
            path.last_key().is_some_and(is_sensitive_name)
                || (keys.len() >= 2
                    && path.last_key().is_some()
                    && SECRET_VALUE_MAPS.contains(&keys[keys.len() - 2]))
        }
        Err(_) => {
            let last = path.rsplit('.').next().unwrap_or(path);
            is_sensitive_name(last)
        }
    }
}

fn is_sensitive_name(key: &str) -> bool {
    SENSITIVE_KEYS.contains(&key)
}

/// Tables whose every value can be a credential (request headers, a
/// server's environment).
const SECRET_VALUE_MAPS: &[&str] = &["extra_headers", "otlp_headers", "env"];

/// Recursively walk a TOML value and replace sensitive keys with a redacted placeholder.
fn redact_secrets(value: &mut toml::Value) {
    match value {
        toml::Value::Table(table) => {
            for (key, val) in table.iter_mut() {
                if is_sensitive_name(key) {
                    *val = toml::Value::String(REDACTED.to_string());
                } else if SECRET_VALUE_MAPS.contains(&key.as_str()) {
                    if let toml::Value::Table(entries) = val {
                        for (_, entry) in entries.iter_mut() {
                            if entry.as_str() != Some("keychain") {
                                *entry = toml::Value::String(REDACTED.to_string());
                            }
                        }
                    }
                } else {
                    redact_secrets(val);
                }
            }
        }
        toml::Value::Array(arr) => {
            for item in arr.iter_mut() {
                redact_secrets(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn moving_a_provider_s_model_asks_the_owner() {
        for key in [
            "provider.models.default",
            "provider.models.primary",
            "provider.fallbacks[1].models.default",
        ] {
            assert!(ConfigManagerTool::is_guarded_key(key), "{key}");
            assert!(ConfigManagerTool::change_needs_approval(
                &ConfigArgs::change("set", key, "\"gpt-6-sol\"")
            ));
        }
        // Other model settings stay free.
        assert!(!ConfigManagerTool::change_needs_approval(
            &ConfigArgs::change("set", "provider.models.fast", "\"gpt-6-luna\"")
        ));
    }

    /// The tool over `path`, checked like a host whose config has the
    /// exposure rules, with secrets in memory.
    fn tool_at(path: PathBuf) -> ConfigManagerTool {
        ConfigManagerTool::new(
            ConfigFile::new(path, Arc::new(test_check)),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        )
    }

    /// TOML syntax plus the exposure rules a host's config check holds.
    fn test_check(content: &str) -> Result<(), String> {
        let table: toml::Table = content
            .parse()
            .map_err(|e| format!("Invalid TOML syntax: {}", e))?;
        let enabled = |section: &str| {
            section
                .split('.')
                .try_fold(&toml::Value::Table(table.clone()), |value, key| {
                    value.get(key)
                })
                .and_then(|value| value.get("enabled"))
                .and_then(toml::Value::as_bool)
                .unwrap_or(false)
        };
        let exposure: aidaemon_exposure::ExposureConfig = table
            .get("exposure")
            .cloned()
            .map(|value| value.try_into())
            .transpose()
            .map_err(|e| format!("Invalid config structure: {}", e))?
            .unwrap_or_default();
        exposure.validate(enabled("client_gateway"), enabled("nodes.gateway"))
    }

    fn saved_table(path: &std::path::Path) -> toml::Table {
        fs::read_to_string(path).unwrap().parse().unwrap()
    }

    fn fallbacks(saved: &str) -> Vec<toml::Value> {
        let table: toml::Table = saved.parse().expect("parse saved config");
        table["provider"]["fallbacks"].as_array().unwrap().clone()
    }

    fn test_args() -> ConfigArgs {
        ConfigArgs {
            action: "switch_provider".to_string(),
            key: String::new(),
            value: String::new(),
            provider: String::new(),
            api_key: None,
            base_url: String::new(),
            gateway_token: None,
            primary_model: String::new(),
            fast_model: String::new(),
            smart_model: String::new(),
            failover_index: None,
            failover_to_index: None,
            stt_cli_path: None,
            stt_model_path: None,
            stt_language: None,
            stt_ffmpeg_path: None,
            kernel_approved: true,
        }
    }

    /// The tool over `path` with `secrets` already in its (in-memory)
    /// secret store, as if the owner had run `aidaemon keychain set`.
    fn tool_with_secrets(
        path: PathBuf,
        secrets: &[(&str, &str)],
    ) -> (ConfigManagerTool, Arc<aidaemon_secrets::MemorySecretStore>) {
        let store = Arc::new(aidaemon_secrets::MemorySecretStore::new());
        for (key, value) in secrets {
            store.set(key, value).unwrap();
        }
        let tool = ConfigManagerTool::new(
            ConfigFile::new(path, Arc::new(test_check)),
            store.clone() as Arc<dyn SecretStore>,
        );
        (tool, store)
    }

    fn test_tool() -> ConfigManagerTool {
        tool_at(PathBuf::from("/tmp/nonexistent-config.toml"))
    }

    fn write_temp_config(contents: &str) -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, contents).expect("write temp config");
        (dir, path)
    }

    /// The tool as the kernel runs it: test calls carry the kernel's
    /// approval (`test_args().kernel_approved`).
    fn approving_tool(config_path: PathBuf) -> ConfigManagerTool {
        tool_at(config_path)
    }

    #[test]
    fn normalize_provider_key_strips_non_alnum() {
        assert_eq!(
            ConfigManagerTool::normalize_provider_key("Cloudflare AI Gateway"),
            "cloudflareaigateway"
        );
    }

    #[test]
    fn find_provider_preset_supports_aliases() {
        let cloudflare = ConfigManagerTool::find_provider_preset("cloudflare").unwrap();
        assert_eq!(cloudflare.id, "cloudflare_gateway");

        let kimi = ConfigManagerTool::find_provider_preset("kimi").unwrap();
        assert_eq!(kimi.id, "moonshot");

        let xai = ConfigManagerTool::find_provider_preset("grok").unwrap();
        assert_eq!(xai.id, "xai");
    }

    #[test]
    fn list_provider_presets_message_includes_expected_entries() {
        let msg = ConfigManagerTool::list_provider_presets_message();
        assert!(msg.contains("xAI (Grok)"));
        assert!(msg.contains("cloudflare_gateway"));
        assert!(msg.contains("custom_openai_compatible"));
        assert!(msg.contains("\"action\":\"switch_provider\""));
        assert!(msg.contains("\"action\":\"add_failover_provider\""));
    }

    #[test]
    fn remove_toml_value_removes_nested_key() {
        let mut table: toml::Table = r#"
[provider]
kind = "openai_compatible"
[provider.models]
primary = "gpt-4o"
fast = "gpt-4o-mini"
"#
        .parse()
        .unwrap();

        remove_toml_value(&mut table, "provider.models.fast").unwrap();

        let provider = table
            .get("provider")
            .and_then(toml::Value::as_table)
            .unwrap();
        let models = provider
            .get("models")
            .and_then(toml::Value::as_table)
            .unwrap();
        assert_eq!(models.get("fast"), None);
        assert_eq!(
            models.get("primary").and_then(toml::Value::as_str),
            Some("gpt-4o")
        );

        remove_toml_value(&mut table, "provider.models.nonexistent").unwrap();
    }

    #[tokio::test]
    async fn switch_provider_requires_provider_name() {
        let tool = test_tool();
        let args = test_args();
        let reply = tool.switch_provider(&args).await.unwrap();
        assert!(reply.contains("'provider' is required"));
    }

    #[tokio::test]
    async fn switch_provider_rejects_unknown_provider() {
        let tool = test_tool();
        let mut args = test_args();
        args.provider = "unknown-provider".to_string();
        let reply = tool.switch_provider(&args).await.unwrap();
        assert!(reply.contains("Unknown provider"));
    }

    #[tokio::test]
    async fn switch_provider_rejects_cloudflare_placeholder_base_url() {
        let (tool, _store) = tool_with_secrets(
            PathBuf::from("/tmp/nonexistent-config.toml"),
            &[("api_key", "sk-synthetic-test-key")],
        );
        let mut args = test_args();
        args.provider = "cloudflare_gateway".to_string();
        args.base_url =
            "https://gateway.ai.cloudflare.com/v1/<ACCOUNT_ID>/<GATEWAY_ID>/compat".to_string();

        let reply = tool.switch_provider(&args).await.unwrap();
        assert!(reply.contains("requires your real gateway URL"));
    }

    #[tokio::test]
    async fn remove_failover_provider_requires_index() {
        let tool = test_tool();
        let mut args = test_args();
        args.action = "remove_failover_provider".to_string();
        let reply = tool.remove_failover_provider(&args).await.unwrap();
        assert!(reply.contains("'failover_index' is required"));
    }

    #[tokio::test]
    async fn add_failover_provider_appends_config() {
        let (_dir, path) = write_temp_config(
            r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"
"#,
        );
        let (tool, _store) = tool_with_secrets(
            path.clone(),
            &[("provider_fallback_0_api_key", "sk-synthetic-test-key")],
        );
        let mut args = test_args();
        args.action = "add_failover_provider".to_string();
        args.provider = "anthropic".to_string();

        let reply = tool.add_failover_provider(&args).await.unwrap();
        assert!(reply.contains("Added failover provider #0"));

        let saved = fs::read_to_string(&path).expect("read saved config");
        let fallbacks = fallbacks(&saved);
        assert_eq!(fallbacks.len(), 1);
        assert_eq!(fallbacks[0]["kind"].as_str(), Some("anthropic"));
        assert_eq!(fallbacks[0]["api_key"].as_str(), Some("keychain"));
        assert_eq!(
            fallbacks[0]["models"]["default"].as_str(),
            Some("claude-sonnet-5-5")
        );
    }

    #[tokio::test]
    async fn list_failover_providers_reads_alias_entries() {
        let (_dir, path) = write_temp_config(
            r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"

[[provider.failover]]
kind = "anthropic"
api_key = "secondary-key"
base_url = "https://api.anthropic.com/v1"

[provider.failover.models]
default = "claude-sonnet-4-20250514"
fallback = ["claude-haiku-4-20250414"]
"#,
        );
        let tool = approving_tool(path);

        let reply = tool.list_failover_providers().await.unwrap();
        assert!(reply.contains("1 failover provider(s) configured"));
        assert!(reply.contains("[0] kind=`anthropic`"));
        assert!(reply.contains("default=`claude-sonnet-4-20250514`"));
        assert!(reply.contains("api_key=`inline`"));
    }

    #[tokio::test]
    async fn remove_failover_provider_removes_selected_entry() {
        let (_dir, path) = write_temp_config(
            r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "first-key"

[provider.fallbacks.models]
default = "claude-sonnet-4-20250514"

[[provider.fallbacks]]
kind = "xai_native"
api_key = "second-key"

[provider.fallbacks.models]
default = "grok-4"
"#,
        );
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "remove_failover_provider".to_string();
        args.failover_index = Some(0);

        let reply = tool.remove_failover_provider(&args).await.unwrap();
        assert!(
            reply.starts_with("Removed failover 1: anthropic"),
            "{reply}"
        );

        let saved = fs::read_to_string(&path).expect("read saved config");
        let fallbacks = fallbacks(&saved);
        assert_eq!(fallbacks.len(), 1);
        assert_eq!(fallbacks[0]["kind"].as_str(), Some("xai_native"));
        assert_eq!(fallbacks[0]["api_key"].as_str(), Some("second-key"));
        assert_eq!(fallbacks[0]["models"]["default"].as_str(), Some("grok-4"));
    }

    const THREE_KEYCHAIN_FAILOVERS: &str = r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "keychain"

[provider.fallbacks.models]
default = "synthetic-a"

[[provider.fallbacks]]
kind = "xai_native"
api_key = "keychain"

[provider.fallbacks.models]
default = "synthetic-b"

[[provider.fallbacks]]
kind = "google_genai"
api_key = "keychain"

[provider.fallbacks.models]
default = "synthetic-c"
"#;

    const THREE_KEYS: [(&str, &str); 3] = [
        ("provider_fallback_0_api_key", "key-a"),
        ("provider_fallback_1_api_key", "key-b"),
        ("provider_fallback_2_api_key", "key-c"),
    ];

    /// Each failover's model and the key in its keychain slot, in order.
    fn failover_keys(
        path: &std::path::Path,
        store: &aidaemon_secrets::MemorySecretStore,
    ) -> Vec<(String, String)> {
        let saved = fs::read_to_string(path).expect("read saved config");
        fallbacks(&saved)
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let key = store
                    .get(&format!("provider_fallback_{i}_api_key"))
                    .map(|k| k.expose().to_string())
                    .unwrap_or_default();
                (f["models"]["default"].as_str().unwrap().to_string(), key)
            })
            .collect()
    }

    #[tokio::test]
    async fn move_failover_provider_moves_each_key_with_its_provider() {
        let (_dir, path) = write_temp_config(THREE_KEYCHAIN_FAILOVERS);
        let (tool, store) = tool_with_secrets(path.clone(), &THREE_KEYS);
        let mut args = test_args();
        args.action = "move_failover_provider".to_string();
        args.failover_index = Some(2);
        args.failover_to_index = Some(0);
        let reply = tool.move_failover_provider(&args).await.unwrap();
        assert!(
            reply.starts_with(
                "Moved google_genai (synthetic-c) to failover 1: it is tried sooner now."
            ),
            "{reply}"
        );
        let pairs = |list: &[(&str, &str)]| -> Vec<(String, String)> {
            list.iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect()
        };
        assert_eq!(
            failover_keys(&path, &store),
            pairs(&[
                ("synthetic-c", "key-c"),
                ("synthetic-a", "key-a"),
                ("synthetic-b", "key-b")
            ])
        );
        // A swap of neighbours.
        args.failover_index = Some(0);
        args.failover_to_index = Some(1);
        tool.move_failover_provider(&args).await.unwrap();
        assert_eq!(
            failover_keys(&path, &store),
            pairs(&[
                ("synthetic-a", "key-a"),
                ("synthetic-c", "key-c"),
                ("synthetic-b", "key-b")
            ])
        );
        assert!(store.get("provider_fallback_moving_0_api_key").is_err());
    }

    const SUBSCRIPTION_MAIN: &str = r#"
[provider]
kind = "openai_chatgpt"
api_key = ""

[provider.models]
default = "synthetic-main"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "keychain"

[provider.fallbacks.models]
default = "synthetic-a"

[[provider.fallbacks]]
kind = "google_genai"
api_key = "keychain"

[provider.fallbacks.models]
default = "synthetic-c"
"#;

    fn switch_args(provider: &str) -> ConfigArgs {
        let mut args = test_args();
        args.action = "switch_provider".to_string();
        args.provider = provider.to_string();
        args.primary_model = "synthetic-new".to_string();
        args
    }

    #[tokio::test]
    async fn making_a_failover_main_swaps_it_with_the_main_provider() {
        let (_dir, path) = write_temp_config(SUBSCRIPTION_MAIN);
        // The app put the failover's key in the main slot before the switch.
        let (tool, store) = tool_with_secrets(
            path.clone(),
            &[
                ("api_key", "key-a"),
                ("provider_fallback_0_api_key", "key-a"),
                ("provider_fallback_1_api_key", "key-c"),
            ],
        );
        let reply = tool
            .switch_provider(&switch_args("anthropic"))
            .await
            .unwrap();
        assert!(reply.contains("is now failover 1"), "{reply}");
        let saved = fs::read_to_string(&path).unwrap();
        let doc: toml::Table = saved.parse().unwrap();
        assert_eq!(doc["provider"]["kind"].as_str(), Some("anthropic"));
        let kinds: Vec<String> = fallbacks(&saved)
            .iter()
            .map(|f| f["kind"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(kinds, ["openai_chatgpt", "google_genai"]);
        assert_eq!(failover_keys(&path, &store)[1].1, "key-c");
        assert_eq!(store.get("api_key").unwrap().expose(), "key-a");
    }

    #[tokio::test]
    async fn a_new_main_provider_puts_the_old_one_first_with_its_key() {
        let (_dir, path) = write_temp_config(
            THREE_KEYCHAIN_FAILOVERS
                .replace("api_key = \"primary-key\"", "api_key = \"keychain\"")
                .as_str(),
        );
        let mut secrets = THREE_KEYS.to_vec();
        secrets.push(("api_key", "new-key"));
        secrets.push((
            aidaemon_secrets::names::PROVIDER_OUTGOING_API_KEY,
            "old-main-key",
        ));
        let (tool, store) = tool_with_secrets(path.clone(), &secrets);
        tool.switch_provider(&switch_args("moonshot"))
            .await
            .unwrap();
        assert_eq!(
            failover_keys(&path, &store),
            [
                ("primary-model", "old-main-key"),
                ("synthetic-a", "key-a"),
                ("synthetic-b", "key-b"),
                ("synthetic-c", "key-c"),
            ]
            .map(|(m, k)| (m.to_string(), k.to_string()))
        );
        assert_eq!(store.get("api_key").unwrap().expose(), "new-key");
        assert!(!store.contains(aidaemon_secrets::names::PROVIDER_OUTGOING_API_KEY));
    }

    #[tokio::test]
    async fn changing_only_the_main_providers_models_moves_nothing() {
        let (_dir, path) = write_temp_config(SUBSCRIPTION_MAIN);
        let (tool, _store) = tool_with_secrets(
            path.clone(),
            &[
                ("provider_fallback_0_api_key", "key-a"),
                ("provider_fallback_1_api_key", "key-c"),
            ],
        );
        tool.switch_provider(&switch_args("openai_chatgpt"))
            .await
            .unwrap();
        assert_eq!(fallbacks(&fs::read_to_string(&path).unwrap()).len(), 2);
    }

    #[tokio::test]
    async fn moving_a_failover_asks_no_approval() {
        let (_dir, path) = write_temp_config(THREE_KEYCHAIN_FAILOVERS);
        let (tool, store) = tool_with_secrets(path.clone(), &THREE_KEYS);
        let call = json!({
            "action": "move_failover_provider",
            "failover_index": 1,
            "failover_to_index": 0,
        });
        assert!(matches!(
            tool.owner_approval(&call).await,
            OwnerApprovalNeed::NotNeeded
        ));
        let mut args = test_args();
        args.action = "move_failover_provider".to_string();
        args.failover_index = Some(1);
        args.failover_to_index = Some(0);
        args.kernel_approved = false;
        tool.move_failover_provider(&args).await.unwrap();
        assert_eq!(failover_keys(&path, &store)[0].0, "synthetic-b");
    }

    #[tokio::test]
    async fn removing_the_first_of_three_keychain_failovers_keeps_the_others_keys() {
        let (_dir, path) = write_temp_config(THREE_KEYCHAIN_FAILOVERS);
        let (tool, store) = tool_with_secrets(path.clone(), &THREE_KEYS);
        let mut args = test_args();
        args.action = "remove_failover_provider".to_string();
        args.failover_index = Some(0);
        tool.remove_failover_provider(&args).await.unwrap();
        assert_eq!(
            failover_keys(&path, &store),
            vec![
                ("synthetic-b".to_string(), "key-b".to_string()),
                ("synthetic-c".to_string(), "key-c".to_string()),
            ]
        );
        assert!(store.get("provider_fallback_2_api_key").is_err());
    }

    #[tokio::test]
    async fn enable_stt_writes_files_stt_section() {
        let dir = tempfile::tempdir().unwrap();
        let cli = dir.path().join("whisper-cli");
        std::fs::write(&cli, b"").unwrap();
        let model = dir.path().join("model.bin");
        std::fs::write(&model, b"fake").unwrap();
        let ffmpeg = dir.path().join("ffmpeg");
        std::fs::write(&ffmpeg, b"").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
            std::fs::set_permissions(&ffmpeg, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let (_cfg_dir, path) = write_temp_config(
            r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gpt-4o"
fast = "gpt-4o-mini"
smart = "gpt-4o"
"#,
        );
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "enable_stt".to_string();
        args.stt_cli_path = Some(cli.to_string_lossy().to_string());
        args.stt_model_path = Some(model.to_string_lossy().to_string());
        args.stt_ffmpeg_path = Some(ffmpeg.to_string_lossy().to_string());
        args.stt_language = Some("en".to_string());

        let reply = tool.enable_stt(&args).await.unwrap();
        assert!(reply.contains("Enabled Whisper STT fallback"));
        assert!(reply.contains("/restart"));

        let saved = fs::read_to_string(&path).expect("read saved config");
        let cfg: toml::Table = saved.parse().expect("parse saved config");
        assert_eq!(cfg["files"]["stt"]["enabled"].as_bool(), Some(true));
        assert_eq!(cfg["files"]["stt"]["language"].as_str(), Some("en"));
    }

    #[tokio::test]
    async fn enable_stt_reports_missing_stack() {
        let (_dir, path) = write_temp_config(
            r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gpt-4o"
"#,
        );
        let tool = approving_tool(path);
        let mut args = test_args();
        args.action = "enable_stt".to_string();
        args.stt_cli_path = Some("/nonexistent/whisper-cli".to_string());
        args.stt_model_path = Some("/nonexistent/model.bin".to_string());
        args.stt_ffmpeg_path = Some("/nonexistent/ffmpeg".to_string());

        let reply = tool.enable_stt(&args).await.unwrap();
        assert!(reply.contains("Cannot enable STT"));
    }

    fn gateway_config(dir: &std::path::Path, extra: &str) -> String {
        format!(
            r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gpt-4o"

[client_gateway]
token_file = "{}"
{extra}
"#,
            dir.join("client-gateway-token.sha256").display()
        )
    }

    #[tokio::test]
    async fn enable_client_gateway_turns_it_on_and_saves_a_working_token() {
        let tokens = tempfile::tempdir().unwrap();
        let (_dir, path) = write_temp_config(&gateway_config(tokens.path(), "port = 9123"));
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "enable_client_gateway".to_string();

        let reply = tool.enable_client_gateway(&args).await.unwrap();
        assert!(reply.text.contains("aidaemon tui"), "{}", reply.text);
        assert!(reply.text.contains("/restart"));

        let saved = saved_table(&path);
        assert_eq!(saved["client_gateway"]["enabled"].as_bool(), Some(true));
        assert_eq!(
            saved["client_gateway"]["port"].as_integer(),
            Some(9123),
            "other settings are kept"
        );

        let copy = tokens.path().join("client-token");
        let token = fs::read_to_string(&copy).unwrap();
        let token = token.trim();
        assert!(aidaemon_client_gateway::token::verify(
            &tokens.path().join("client-gateway-token.sha256"),
            token
        ));
        assert!(
            !reply.text.contains(token),
            "the token never enters the reply"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&copy).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }

        // Again: nothing changes and the working token is kept.
        let again = tool.enable_client_gateway(&args).await.unwrap();
        assert!(again.text.contains("already on"));
        assert_eq!(fs::read_to_string(&copy).unwrap().trim(), token);
    }

    #[tokio::test]
    async fn enable_exposure_writes_a_stable_adapter_and_asks_the_owner() {
        let tokens = tempfile::tempdir().unwrap();
        let (_dir, path) = write_temp_config(&gateway_config(tokens.path(), "enabled = true"));
        let tool = approving_tool(path.clone());
        let call = json!({"action": "enable_exposure", "key": "client_gateway", "value": r#"{ adapter = "tailscale" }"#});
        match tool.owner_approval(&call).await {
            OwnerApprovalNeed::Ask(request) => {
                assert!(
                    request.summary.contains("private to your tailnet"),
                    "{}",
                    request.summary
                )
            }
            _ => panic!("enable_exposure must ask"),
        }
        let mut args = test_args();
        args.action = "enable_exposure".into();
        args.key = "client_gateway".into();
        args.value = r#"{ adapter = "tailscale" }"#.into();
        let reply = tool.enable_exposure(&args).await.unwrap();
        assert!(reply.text.contains("/restart"), "{}", reply.text);
        let saved = saved_table(&path);
        let exposure: aidaemon_exposure::ExposureConfig =
            saved["exposure"].clone().try_into().unwrap();
        assert_eq!(exposure.client_gateway.unwrap().adapter_name(), "tailscale");
        assert_eq!(
            saved["client_gateway"]["enabled"].as_bool(),
            Some(true),
            "other settings are kept"
        );

        args.kernel_approved = false;
        assert!(tool.enable_exposure(&args).await.is_err());
    }

    #[tokio::test]
    async fn enable_exposure_refuses_command_ephemeral_nodes_and_an_off_gateway() {
        let tokens = tempfile::tempdir().unwrap();
        let (_dir, path) = write_temp_config(&gateway_config(tokens.path(), ""));
        let before = fs::read_to_string(&path).unwrap();
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "enable_exposure".into();
        args.key = "client_gateway".into();
        args.value = r#"{ adapter = "command", command = ["x"] }"#.into();
        let reply = tool.enable_exposure(&args).await.unwrap();
        assert!(reply.text.contains("by hand"), "{}", reply.text);
        args.value = r#"{ adapter = "tailscale" }"#.into();
        let reply = tool.enable_exposure(&args).await.unwrap();
        assert!(
            reply.text.contains("enable_client_gateway"),
            "{}",
            reply.text
        );
        args.key = "node_gateway".into();
        args.value = r#"{ adapter = "cloudflare", mode = "quick" }"#.into();
        let reply = tool.enable_exposure(&args).await.unwrap();
        assert!(reply.text.contains("survives restarts"), "{}", reply.text);
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[tokio::test]
    async fn set_never_writes_a_command_exposure() {
        let tokens = tempfile::tempdir().unwrap();
        let owner_written = "enabled = true\n\n[exposure.client_gateway]\nadapter = \"command\"\ncommand = [\"my-tunnel\", \"{port}\"]\n";
        let (_dir, path) = write_temp_config(&gateway_config(tokens.path(), owner_written));
        let tool = approving_tool(path.clone());
        let before = fs::read_to_string(&path).unwrap();
        for (key, value) in [
            ("exposure.client_gateway.command", r#"["other-program"]"#),
            ("exposure.client_gateway.stable", "true"),
        ] {
            let mut args = test_args();
            args.action = "set".into();
            args.key = key.into();
            args.value = value.into();
            let reply = tool.change_value(&args).await.unwrap();
            assert!(reply.text.contains("by hand"), "{key}: {}", reply.text);
        }
        assert_eq!(fs::read_to_string(&path).unwrap(), before);

        // A whole table, or switching an adapter to `command`: the same.
        let (_dir2, path2) = write_temp_config(&gateway_config(
            tokens.path(),
            "enabled = true\n\n[exposure.client_gateway]\nadapter = \"manual\"\nurl = \"https://app.example.com\"\n",
        ));
        let tool = approving_tool(path2.clone());
        let mut args = test_args();
        args.action = "set".into();
        args.key = "exposure.client_gateway".into();
        args.value = r#"{ adapter = "command", command = ["x"] }"#.into();
        let reply = tool.change_value(&args).await.unwrap();
        assert!(reply.text.contains("by hand"), "{}", reply.text);
        // Removing the owner's command table is fine.
        let mut unset = test_args();
        unset.action = "unset".into();
        unset.key = "exposure.client_gateway".into();
        let tool = approving_tool(path.clone());
        let reply = tool.change_value(&unset).await.unwrap();
        assert!(reply.text.starts_with("Removed"), "{}", reply.text);
        assert!(matches!(
            tool.owner_approval(&json!({"action": "set", "key": "exposure.client_gateway.url", "value": "\"https://a.example.com\""})).await,
            OwnerApprovalNeed::Ask(_)
        ));
    }

    /// Only the tables that tune how the agent talks are free: where data
    /// goes, who may reach the daemon and what runs on the machine ask.
    #[tokio::test]
    async fn a_change_outside_the_tuning_tables_asks_the_owner() {
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let tool = approving_tool(path);
        for (key, value) in [
            (
                "telemetry.otlp_endpoint",
                "\"https://collector.example.invalid\"",
            ),
            ("mcp.files.command", "\"node\""),
            ("users.owner_ids.telegram", "[42]"),
            (
                "oauth.providers.acme.token_url",
                "\"https://acme.example.invalid/token\"",
            ),
            ("terminal.permission_mode", "\"yolo\""),
            ("search.searxng_url", "\"https://search.example.invalid\""),
            ("agent.persona_file", "\"/tmp/persona.md\""),
        ] {
            assert!(
                matches!(
                    tool.owner_approval(&json!({"action": "set", "key": key, "value": value}))
                        .await,
                    OwnerApprovalNeed::Ask(_)
                ),
                "{key} must ask"
            );
        }
        assert!(matches!(
            tool.owner_approval(&json!({"action": "set", "key": "agent.name", "value": "\"Ada\""}))
                .await,
            OwnerApprovalNeed::NotNeeded
        ));
    }

    /// The config expands `${NAME}` from the daemon's environment: the tool
    /// never writes such a reference.
    #[tokio::test]
    async fn a_value_naming_an_environment_variable_is_refused() {
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let before = fs::read_to_string(&path).unwrap();
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "set".into();
        args.key = "agent.name".into();
        args.value = "\"${AIDAEMON_ENCRYPTION_KEY}\"".into();
        let reply = tool.change_value(&args).await.unwrap();
        assert!(reply.text.contains("Nothing was changed"), "{}", reply.text);
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[tokio::test]
    async fn enable_client_gateway_needs_the_kernel_approval() {
        let tokens = tempfile::tempdir().unwrap();
        let (_dir, path) = write_temp_config(&gateway_config(tokens.path(), ""));
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "enable_client_gateway".to_string();
        args.kernel_approved = false;
        assert!(tool.enable_client_gateway(&args).await.is_err());
        let saved = saved_table(&path);
        assert_ne!(
            saved["client_gateway"]
                .get("enabled")
                .and_then(toml::Value::as_bool),
            Some(true)
        );
        assert!(!tokens.path().join("client-token").exists());
    }

    /// A complete synthetic STT stack in a temp dir: (dir, cli, model, ffmpeg).
    fn fake_stt_stack() -> (TempDir, String, String, String) {
        let dir = tempfile::tempdir().unwrap();
        let cli = dir.path().join("whisper-cli");
        let model = dir.path().join("model.bin");
        let ffmpeg = dir.path().join("ffmpeg");
        fs::write(&cli, b"").unwrap();
        fs::write(&model, b"fake").unwrap();
        fs::write(&ffmpeg, b"").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for bin in [&cli, &ffmpeg] {
                fs::set_permissions(bin, fs::Permissions::from_mode(0o755)).unwrap();
            }
        }
        let s = |p: &std::path::Path| p.to_string_lossy().to_string();
        let (cli, model, ffmpeg) = (s(&cli), s(&model), s(&ffmpeg));
        (dir, cli, model, ffmpeg)
    }

    #[tokio::test]
    async fn enable_stt_asks_the_owner_with_the_resolved_programs() {
        let (_stack, cli, model, ffmpeg) = fake_stt_stack();
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let tool = approving_tool(path.clone());
        let arguments = json!({
            "action": "enable_stt",
            "stt_cli_path": cli,
            "stt_model_path": model,
            "stt_ffmpeg_path": ffmpeg,
        });
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&arguments).await else {
            panic!("enable_stt must ask the owner");
        };
        assert!(request.summary.contains(&cli), "{}", request.summary);
        assert!(request.summary.contains(&ffmpeg), "{}", request.summary);
        assert!(request.subject.contains(&cli), "{}", request.subject);

        let before = fs::read_to_string(&path).unwrap();
        let mut args: ConfigArgs = serde_json::from_value(arguments).unwrap();
        args.kernel_approved = false;
        assert!(tool.enable_stt(&args).await.is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[tokio::test]
    async fn enable_stt_never_runs_a_shell_for_the_ffmpeg_override() {
        let (_stack, cli, model, _ffmpeg) = fake_stt_stack();
        let marker =
            std::env::temp_dir().join(format!("aidaemon-stt-injection-{}", uuid::Uuid::new_v4()));
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "enable_stt".to_string();
        args.stt_cli_path = Some(cli);
        args.stt_model_path = Some(model);
        args.stt_ffmpeg_path = Some(format!("x; touch {}", marker.display()));
        let _ = tool
            .owner_approval(&json!({
                "action": "enable_stt",
                "stt_ffmpeg_path": args.stt_ffmpeg_path,
            }))
            .await;
        let reply = tool.enable_stt(&args).await.unwrap();
        assert!(reply.contains("ffmpeg not found"), "{reply}");
        assert!(!marker.exists(), "the override ran a shell command");
    }

    #[tokio::test]
    async fn restore_asks_the_owner_and_writes_the_judged_backup() {
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let tool = approving_tool(path.clone());
        let backup = "[provider]\nkind = \"openai_compatible\"\napi_key = \"keychain\"\n\n[provider.models]\ndefault = \"older-model\"\n";
        fs::write(path.with_extension("toml.bak"), backup).unwrap();

        let OwnerApprovalNeed::Ask(request) =
            tool.owner_approval(&json!({"action": "restore"})).await
        else {
            panic!("restore must ask the owner");
        };
        assert!(
            request.subject.contains("backup_sha256"),
            "{}",
            request.subject
        );

        let mut args: ConfigArgs = serde_json::from_value(json!({"action": "restore"})).unwrap();
        args.kernel_approved = false;
        assert!(tool.restore_backup(&args).await.is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), PRIMARY_CONFIG);

        args.kernel_approved = true;
        tool.restore_backup(&args).await.unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), backup);
    }

    #[tokio::test]
    async fn unsetting_a_free_table_that_holds_a_guarded_key_asks() {
        let config = "[provider]\nkind = \"openai_compatible\"\napi_key = \"keychain\"\n\n[provider.models]\ndefault = \"primary-model\"\n\n[memory]\ndb_path = \"/tmp/synthetic-memory.db\"\n";
        let (_dir, path) = write_temp_config(config);
        let tool = approving_tool(path.clone());
        let arguments = json!({"action": "unset", "key": "memory"});
        assert!(matches!(
            tool.owner_approval(&arguments).await,
            OwnerApprovalNeed::Ask(_)
        ));
        let mut args: ConfigArgs = serde_json::from_value(arguments).unwrap();
        args.kernel_approved = false;
        assert!(tool.change_value(&args).await.is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), config);
    }

    const PRIMARY_CONFIG: &str = "[provider]\nkind = \"openai_compatible\"\napi_key = \"keychain\"\n\n[provider.models]\ndefault = \"primary-model\"\n";

    /// The calls a model makes, parsed as the tool parses them.
    fn approved_args(arguments: &str) -> ConfigArgs {
        let mut args: ConfigArgs = serde_json::from_str(arguments).unwrap();
        args.kernel_approved = true;
        args
    }

    #[tokio::test]
    async fn provider_secret_arguments_are_refused_unread() {
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let (tool, store) = tool_with_secrets(path.clone(), &[]);
        let before = fs::read_to_string(&path).unwrap();

        let switch = approved_args(
            r#"{"action":"switch_provider","provider":"anthropic","api_key":"sk-synthetic-test-key"}"#,
        );
        let reply = tool.switch_provider(&switch).await.unwrap();
        assert!(!reply.contains("sk-synthetic-test-key"), "{reply}");
        assert!(reply.contains("aidaemon keychain set api_key"), "{reply}");
        assert!(reply.contains("/setkey"), "{reply}");

        let failover = approved_args(
            r#"{"action":"add_failover_provider","provider":"cloudflare_gateway","gateway_token":"sk-synthetic-test-key"}"#,
        );
        let reply = tool.add_failover_provider(&failover).await.unwrap();
        assert!(!reply.contains("sk-synthetic-test-key"), "{reply}");
        assert!(
            reply.contains("aidaemon keychain set provider_fallback_0_gateway_token"),
            "{reply}"
        );

        assert_eq!(fs::read_to_string(&path).unwrap(), before);
        assert!(!store.contains("api_key"));
        assert!(!store.contains("provider_fallback_0_gateway_token"));
        assert!(matches!(
            tool.owner_approval(&json!({"action": "switch_provider", "provider": "anthropic", "api_key": "sk-synthetic-test-key"})).await,
            OwnerApprovalNeed::NotNeeded
        ));
    }

    #[tokio::test]
    async fn switch_provider_uses_the_owner_stored_key() {
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let (tool, _store) =
            tool_with_secrets(path.clone(), &[("api_key", "sk-synthetic-test-key")]);

        let reply = tool
            .switch_provider(&approved_args(
                r#"{"action":"switch_provider","provider":"anthropic"}"#,
            ))
            .await
            .unwrap();
        assert!(reply.contains("Switched provider to"), "{reply}");
        assert!(!reply.contains("sk-synthetic-test-key"), "{reply}");
        let saved = saved_table(&path);
        assert_eq!(saved["provider"]["api_key"].as_str(), Some("keychain"));
        assert_eq!(saved["provider"]["kind"].as_str(), Some("anthropic"));
    }

    #[tokio::test]
    async fn switch_provider_without_a_stored_key_names_the_owner_command() {
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let (tool, _store) = tool_with_secrets(path.clone(), &[]);
        let before = fs::read_to_string(&path).unwrap();

        let reply = tool
            .switch_provider(&approved_args(
                r#"{"action":"switch_provider","provider":"anthropic"}"#,
            ))
            .await
            .unwrap();
        assert!(reply.contains("aidaemon keychain set api_key"), "{reply}");
        assert_eq!(fs::read_to_string(&path).unwrap(), before);

        let reply = tool
            .add_failover_provider(&approved_args(
                r#"{"action":"add_failover_provider","provider":"anthropic"}"#,
            ))
            .await
            .unwrap();
        assert!(
            reply.contains("aidaemon keychain set provider_fallback_0_api_key"),
            "{reply}"
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[tokio::test]
    async fn set_refuses_a_literal_secret_but_takes_the_keychain_reference() {
        let (_dir, path) = write_temp_config(PRIMARY_CONFIG);
        let tool = approving_tool(path.clone());
        let before = fs::read_to_string(&path).unwrap();

        for (key, command) in [
            ("provider.api_key", "aidaemon keychain set api_key"),
            (
                "provider.fallbacks[1].api_key",
                "aidaemon keychain set provider_fallback_1_api_key",
            ),
            ("telegram.bot_token", "aidaemon keychain set <key>"),
        ] {
            let mut args = test_args();
            args.action = "set".into();
            args.key = key.into();
            args.value = "sk-synthetic-test-key".into();
            let reply = tool.change_value(&args).await.unwrap();
            assert!(
                !reply.text.contains("sk-synthetic-test-key"),
                "{}",
                reply.text
            );
            assert!(reply.text.contains(command), "{key}: {}", reply.text);
            assert!(matches!(
                tool.owner_approval(
                    &json!({"action": "set", "key": key, "value": "sk-synthetic-test-key"})
                )
                .await,
                OwnerApprovalNeed::NotNeeded
            ));
        }
        assert_eq!(fs::read_to_string(&path).unwrap(), before);

        let mut args = test_args();
        args.action = "set".into();
        args.key = "provider.api_key".into();
        args.value = "keychain".into();
        assert!(!matches!(
            tool.owner_approval(
                &json!({"action": "set", "key": "provider.api_key", "value": "keychain"})
            )
            .await,
            OwnerApprovalNeed::NotNeeded
        ));
        // Past the secret check (this test's config check does not model
        // the daemon's provider schema, so the write itself is not asserted).
        let reply = tool.change_value(&args).await.unwrap();
        assert!(
            !reply.text.contains("not accepted as tool arguments"),
            "{}",
            reply.text
        );
    }

    #[test]
    fn agent_config_changes_require_restart() {
        assert!(apply_instruction_for_key("agent.name").starts_with("Restart"));
        assert!(apply_instruction_for_key("agent.persona_file").starts_with("Restart"));
        assert!(apply_instruction_for_key("provider.models.default").starts_with("Run /reload"));
    }
}
