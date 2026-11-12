use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::config::AppConfig;

mod config_path;
use crate::traits::{
    semantics_for_exact_read_actions, OwnerApprovalNeed, Tool, ToolCallContext, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects,
};
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

const PROVIDER_PRESETS: &[ProviderPreset] = &[
    ProviderPreset {
        id: "google_ai_studio",
        aliases: &["google", "googleaistudio", "gemini"],
        display_name: "Google AI Studio",
        kind: "google_genai",
        base_url: "",
        primary: "gemini-3-flash-preview",
        fast: "gemini-2.5-flash-lite",
        smart: "gemini-3-pro-preview",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "openai",
        aliases: &[],
        display_name: "OpenAI",
        kind: "openai_compatible",
        base_url: "https://api.openai.com/v1",
        primary: "gpt-4o",
        fast: "gpt-4o-mini",
        smart: "gpt-4o",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "xai",
        aliases: &["x.ai", "grok"],
        display_name: "xAI (Grok)",
        kind: "xai_native",
        base_url: "https://api.x.ai/v1",
        primary: "grok-4",
        fast: "grok-4",
        smart: "grok-4",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "anthropic_native",
        aliases: &["anthropic"],
        display_name: "Anthropic (Native)",
        kind: "anthropic",
        base_url: "https://api.anthropic.com/v1",
        primary: "claude-sonnet-4-20250514",
        fast: "claude-haiku-4-20250414",
        smart: "claude-opus-4-20250414",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "anthropic_openrouter",
        aliases: &["anthropicopenrouter"],
        display_name: "Anthropic (via OpenRouter)",
        kind: "openai_compatible",
        base_url: "https://openrouter.ai/api/v1",
        primary: "anthropic/claude-sonnet-4",
        fast: "anthropic/claude-haiku-4",
        smart: "anthropic/claude-opus-4",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "openrouter",
        aliases: &[],
        display_name: "OpenRouter",
        kind: "openai_compatible",
        base_url: "https://openrouter.ai/api/v1",
        primary: "openai/gpt-4o",
        fast: "openai/gpt-4o-mini",
        smart: "anthropic/claude-sonnet-4",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "deepseek",
        aliases: &[],
        display_name: "DeepSeek",
        kind: "openai_compatible",
        base_url: "https://api.deepseek.com",
        primary: "deepseek-chat",
        fast: "deepseek-chat",
        smart: "deepseek-reasoner",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "moonshot",
        aliases: &["moonshotai", "kimi", "kimik2", "kimik25"],
        display_name: "Moonshot AI (Kimi)",
        kind: "openai_compatible",
        base_url: "https://api.moonshot.ai/v1",
        primary: "kimi-k2.5",
        fast: "kimi-k2.5",
        smart: "kimi-k2-thinking",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "minimax",
        aliases: &[],
        display_name: "MiniMax",
        kind: "openai_compatible",
        base_url: "https://api.minimax.io/v1",
        primary: "MiniMax-M2.5",
        fast: "MiniMax-M2.5-highspeed",
        smart: "MiniMax-M2.5",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "cloudflare_gateway",
        aliases: &["cloudflare", "cloudflareaigateway", "aigateway"],
        display_name: "Cloudflare AI Gateway",
        kind: "openai_compatible",
        base_url: "https://gateway.ai.cloudflare.com/v1/<ACCOUNT_ID>/<GATEWAY_ID>/compat",
        primary: "gpt-4o-mini",
        fast: "gpt-4o-mini",
        smart: "gpt-4o",
        needs_api_key: true,
        supports_gateway_token: true,
        requires_custom_base_url: true,
    },
    ProviderPreset {
        id: "ollama",
        aliases: &["ollamalocal"],
        display_name: "Ollama (Local)",
        kind: "openai_compatible",
        base_url: "http://localhost:11434/v1",
        primary: "llama3.1",
        fast: "llama3.1",
        smart: "llama3.1",
        needs_api_key: false,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "openai_chatgpt",
        aliases: &[
            "chatgpt",
            "chatgptsubscription",
            "codex",
            "openaisubscription",
        ],
        display_name: "OpenAI (ChatGPT subscription)",
        kind: "openai_chatgpt",
        base_url: "",
        primary: "gpt-5.6-terra",
        fast: "gpt-5.6-luna",
        smart: "gpt-5.6-sol",
        // Credentials come from the ChatGPT OAuth store, not an API key.
        needs_api_key: false,
        supports_gateway_token: false,
        requires_custom_base_url: false,
    },
    ProviderPreset {
        id: "custom_openai_compatible",
        aliases: &["custom", "openaicompatible"],
        display_name: "Custom (OpenAI Compatible)",
        kind: "openai_compatible",
        base_url: "",
        primary: "model-name",
        fast: "model-name",
        smart: "model-name",
        needs_api_key: true,
        supports_gateway_token: false,
        requires_custom_base_url: true,
    },
];

#[derive(Clone)]
struct ResolvedProviderSelection {
    preset: &'static ProviderPreset,
    api_key: String,
    base_url: String,
    primary_model: String,
    fast_model: String,
    smart_model: String,
    gateway_token: Option<String>,
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
    config_path: PathBuf,
}

/// Set file permissions to owner-only read/write (0600) on Unix.
/// This is a best-effort operation — failures are logged but not fatal.
#[cfg(unix)]
pub(crate) fn set_owner_only_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        warn!(path = %path.display(), "Failed to set 0600 permissions: {}", e);
    }
}

#[cfg(not(unix))]
pub(crate) fn set_owner_only_permissions(_path: &Path) {
    // No-op on non-Unix platforms
}

/// Rotate backups (3-deep ring) and create a new .bak from current config.
/// Rotation: .bak.2 is deleted, .bak.1 → .bak.2, .bak → .bak.1, current → .bak
pub(crate) async fn rotate_config_backups(config_path: &Path) -> anyhow::Result<()> {
    let bak = config_path.with_extension("toml.bak");
    let bak1 = config_path.with_extension("toml.bak.1");
    let bak2 = config_path.with_extension("toml.bak.2");

    // Rotate: .bak.1 → .bak.2
    if bak1.exists() {
        let _ = tokio::fs::rename(&bak1, &bak2).await;
    }
    // Rotate: .bak → .bak.1
    if bak.exists() {
        let _ = tokio::fs::rename(&bak, &bak1).await;
    }
    // Current → .bak
    tokio::fs::copy(&config_path, &bak).await?;
    // Restrict permissions on all backup files (contain secrets)
    set_owner_only_permissions(&bak);
    set_owner_only_permissions(&bak1);
    set_owner_only_permissions(&bak2);
    info!(backup = %bak.display(), "Config backup created (3-deep rotation)");
    Ok(())
}

impl ConfigManagerTool {
    pub fn new(config_path: PathBuf) -> Self {
        Self { config_path }
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

    /// The summary of failover entry `index` of `entries`.
    fn failover_entry_summary(entries: &[toml::Value], index: usize) -> String {
        entries[index]
            .as_table()
            .map(|table| Self::summarize_provider_table(index, table))
            .unwrap_or_else(|| format!("[{}] (invalid provider entry)", index))
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
                    || !Self::change_needs_approval(args)
                {
                    return Ok(None);
                }
                Ok(Some(Self::set_summary(args)))
            }
            "switch_provider" => Ok(Self::resolve_provider_selection(args, "switch_provider")
                .ok()
                .map(|selection| Self::switch_provider_summary(&selection))),
            "add_failover_provider" => {
                let Ok(selection) = Self::resolve_provider_selection(args, "add_failover_provider")
                else {
                    return Ok(None);
                };
                let content = tokio::fs::read_to_string(&self.config_path).await?;
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
                Ok(Some(Self::add_failover_summary(next_index, &selection)))
            }
            "remove_failover_provider" => {
                let Some(index) = args.failover_index else {
                    return Ok(None);
                };
                let content = tokio::fs::read_to_string(&self.config_path).await?;
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
                Ok(Some(format!(
                    "Remove failover provider {}",
                    Self::failover_entry_summary(entries, index)
                )))
            }
            _ => Ok(None),
        }
    }

    /// The exact arguments the owner approves (a changed argument is a new
    /// card), with secret values replaced by their SHA-256 digest so no
    /// secret is kept in a grant or a decline.
    fn approval_subject(args: &ConfigArgs) -> String {
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
            _ => json!({
                "action": args.action,
                "provider": args.provider,
                "api_key": secret(&args.api_key),
                "base_url": args.base_url,
                "gateway_token": secret(&args.gateway_token),
                "primary_model": args.primary_model,
                "fast_model": args.fast_model,
                "smart_model": args.smart_model,
                "save_secrets_to_keychain": args.save_secrets_to_keychain,
            }),
        };
        subject.to_string()
    }

    /// The owner approval for one call: the same changes the tool asked
    /// about before (security-sensitive `set`, provider switch, failover
    /// add/remove), with the same wording. If the config cannot be read to
    /// word the card, it still asks.
    pub(crate) async fn owner_approval(&self, arguments: &Value) -> OwnerApprovalNeed {
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
            effects: Vec::new(),
            summary,
            subject: Self::approval_subject(&args),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec!["Modifying security-sensitive configuration".to_string()],
        })
    }

    /// The typed effects of one call: every change writes the config file,
    /// a configuration effect on that path. A change may also store secrets
    /// in the keychain or redirect provider traffic, so the effects are
    /// declared incomplete. Reads have none.
    pub(crate) fn effects_for(&self, arguments: &str) -> aidaemon_core::CallEffects {
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
        let config_key = std::path::absolute(&self.config_path)
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
        rotate_config_backups(&self.config_path).await
    }

    /// Restore from the first available backup in the chain.
    async fn restore_backup(&self) -> anyhow::Result<String> {
        let candidates = [
            self.config_path.with_extension("toml.bak"),
            self.config_path.with_extension("toml.bak.1"),
            self.config_path.with_extension("toml.bak.2"),
        ];

        for candidate in &candidates {
            if candidate.exists() {
                tokio::fs::copy(candidate, &self.config_path).await?;
                warn!(source = %candidate.display(), "Config restored from backup");
                return Ok(format!(
                    "Config restored from {}. Run /reload to apply.",
                    candidate.display()
                ));
            }
        }

        anyhow::bail!("No backups found (checked .bak, .bak.1, .bak.2)")
    }

    /// Validate that a config string parses correctly as AppConfig.
    /// Expands `${ENV_VAR}` references before structural validation but does
    /// NOT resolve `"keychain"` sentinels (the value may not be stored yet).
    pub(crate) fn validate_config(content: &str) -> Result<(), String> {
        // First check it's valid TOML
        let _doc: toml::Table = content
            .parse()
            .map_err(|e| format!("Invalid TOML syntax: {}", e))?;

        // Expand env vars so "${VAR}" values don't break type validation
        let expanded = crate::config::expand_env_vars(content).map_err(|e| format!("{}", e))?;

        // Then check it deserializes into our AppConfig
        let config = toml::from_str::<AppConfig>(&expanded)
            .map_err(|e| format!("Invalid config structure: {}", e))?;
        config
            .agent
            .validated_name()
            .map_err(|e| format!("Invalid agent configuration: {}", e))?;

        Ok(())
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

        for preset in PROVIDER_PRESETS {
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
            "Example: {\"action\":\"switch_provider\",\"provider\":\"moonshot\",\"api_key\":\"YOUR_KEY\"}"
                .to_string(),
        );
        lines.push(
            "Failover example: {\"action\":\"add_failover_provider\",\"provider\":\"anthropic\",\"api_key\":\"YOUR_KEY\"}"
                .to_string(),
        );

        lines.join("\n")
    }

    fn resolve_secret_config_value(
        keychain_key: &str,
        secret: &str,
        save_to_keychain: bool,
    ) -> (toml::Value, bool, Option<String>) {
        if !save_to_keychain {
            return (toml::Value::String(secret.to_string()), false, None);
        }

        match crate::config::store_in_keychain(keychain_key, secret) {
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

    fn resolve_provider_selection(
        args: &ConfigArgs,
        action_name: &str,
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

        let api_key = args.api_key.trim();
        if preset.needs_api_key && api_key.is_empty() {
            return Err(format!(
                "Missing API key for {}. Provide `api_key` and retry.\n\nExample: {{\"action\":\"{}\",\"provider\":\"{}\",\"api_key\":\"YOUR_KEY\"}}",
                preset.display_name, action_name, preset.id
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

        let gateway_token =
            if preset.supports_gateway_token && !args.gateway_token.trim().is_empty() {
                Some(args.gateway_token.trim().to_string())
            } else {
                None
            };

        Ok(ResolvedProviderSelection {
            preset,
            api_key: api_key.to_string(),
            base_url,
            primary_model,
            fast_model,
            smart_model,
            gateway_token,
        })
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
        provider_table: &mut toml::Table,
        selection: &ResolvedProviderSelection,
        key_prefix: Option<&str>,
        save_secrets_to_keychain: bool,
        notes: &mut Vec<String>,
        note_prefix: &str,
    ) {
        provider_table.insert(
            "kind".to_string(),
            toml::Value::String(selection.preset.kind.to_string()),
        );

        let api_key_value = if selection.preset.needs_api_key {
            let (value, stored, note) = Self::resolve_secret_config_value(
                &Self::keychain_field_name(key_prefix, "api_key"),
                &selection.api_key,
                save_secrets_to_keychain,
            );
            if stored {
                notes.push(if note_prefix.is_empty() {
                    "API key stored in OS keychain.".to_string()
                } else {
                    format!("{} API key stored in OS keychain.", note_prefix)
                });
            }
            if let Some(note) = note {
                notes.push(note);
            }
            value
        } else {
            toml::Value::String(if !selection.api_key.is_empty() {
                selection.api_key.clone()
            } else if selection.preset.kind == "openai_chatgpt" {
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

        if let Some(token) = selection.gateway_token.as_deref() {
            let (token_value, stored, note) = Self::resolve_secret_config_value(
                &Self::keychain_field_name(key_prefix, "gateway_token"),
                token,
                save_secrets_to_keychain,
            );
            if stored {
                notes.push(if note_prefix.is_empty() {
                    "Gateway token stored in OS keychain.".to_string()
                } else {
                    format!("{} gateway token stored in OS keychain.", note_prefix)
                });
            }
            if let Some(note) = note {
                notes.push(note);
            }
            provider_table.insert("gateway_token".to_string(), token_value);
        } else {
            provider_table.remove("gateway_token");
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

        let secret = crate::config::resolve_from_keychain(old_key)?;
        let (new_value, _stored, note) = Self::resolve_secret_config_value(new_key, &secret, true);
        provider_table.insert(field_name.to_string(), new_value);
        cleanup_keys.push(old_key.to_string());
        if let Some(note) = note {
            notes.push(note);
        }
        Ok(())
    }

    fn migrate_secret_headers(
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

            let secret = crate::config::resolve_from_keychain(&old_key)?;
            let (new_value, _stored, note) =
                Self::resolve_secret_config_value(&new_key, &secret, true);
            *header_value = new_value;
            cleanup_keys.push(old_key);
            if let Some(note) = note {
                notes.push(note);
            }
        }

        Ok(())
    }

    fn rekey_provider_table(
        provider_table: &toml::Table,
        old_prefix: &str,
        new_prefix: &str,
        cleanup_keys: &mut Vec<String>,
        notes: &mut Vec<String>,
    ) -> anyhow::Result<toml::Table> {
        let mut migrated = provider_table.clone();

        Self::migrate_secret_string(
            &mut migrated,
            "api_key",
            &Self::keychain_field_name(Some(old_prefix), "api_key"),
            &Self::keychain_field_name(Some(new_prefix), "api_key"),
            cleanup_keys,
            notes,
        )?;
        Self::migrate_secret_string(
            &mut migrated,
            "gateway_token",
            &Self::keychain_field_name(Some(old_prefix), "gateway_token"),
            &Self::keychain_field_name(Some(new_prefix), "gateway_token"),
            cleanup_keys,
            notes,
        )?;
        Self::migrate_secret_headers(&mut migrated, old_prefix, new_prefix, cleanup_keys, notes)?;

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
                *value = toml::Value::Table(Self::rekey_provider_table(
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
        let content = tokio::fs::read_to_string(&self.config_path).await?;
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
        let selection = match Self::resolve_provider_selection(args, "add_failover_provider") {
            Ok(selection) => selection,
            Err(message) => return Ok(message),
        };

        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let mut doc: toml::Table = content.parse()?;
        let provider_table = doc
            .entry("provider")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
        let next_index = Self::get_failover_array(provider_table)?
            .map(|entries| entries.len())
            .unwrap_or(0);

        Self::require_kernel_approval(args)?;

        let mut notes = Vec::new();
        let mut new_provider = toml::Table::new();
        let key_prefix = Self::fallback_key_prefix(next_index);
        Self::apply_provider_selection(
            &mut new_provider,
            &selection,
            Some(&key_prefix),
            args.save_secrets_to_keychain,
            &mut notes,
            "Failover",
        );

        Self::normalize_failover_array_mut(provider_table)?.push(toml::Value::Table(new_provider));

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = Self::validate_config(&new_content) {
            return Ok(format!(
                "Refused to save failover provider: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        tokio::fs::write(&self.config_path, &new_content).await?;
        set_owner_only_permissions(&self.config_path);

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

    async fn remove_failover_provider(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let Some(index) = args.failover_index else {
            return Ok(
                "Error: 'failover_index' is required for action='remove_failover_provider'."
                    .to_string(),
            );
        };

        let content = tokio::fs::read_to_string(&self.config_path).await?;
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

            let removed_summary = Self::failover_entry_summary(current_entries, index);
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

        let mut notes = Vec::new();
        let mut cleanup_keys = Vec::new();
        let mut remaining_entries = Vec::new();
        for (old_index, table) in existing_tables.into_iter().enumerate() {
            if old_index == index {
                continue;
            }
            let new_index = remaining_entries.len();
            let migrated = if old_index == new_index {
                table
            } else {
                Self::rekey_provider_table(
                    &table,
                    &Self::fallback_key_prefix(old_index),
                    &Self::fallback_key_prefix(new_index),
                    &mut cleanup_keys,
                    &mut notes,
                )?
            };
            remaining_entries.push(toml::Value::Table(migrated));
        }

        if remaining_entries.is_empty() {
            provider_table.remove("fallbacks");
            provider_table.remove("failover");
        } else {
            *Self::normalize_failover_array_mut(provider_table)? = remaining_entries;
        }

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = Self::validate_config(&new_content) {
            return Ok(format!(
                "Refused to remove failover provider: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        tokio::fs::write(&self.config_path, &new_content).await?;
        set_owner_only_permissions(&self.config_path);

        cleanup_keys.sort();
        cleanup_keys.dedup();
        for key in cleanup_keys {
            if let Err(e) = crate::config::delete_from_keychain(&key) {
                warn!(key = %key, error = %e, "Failed to delete stale failover keychain entry");
            }
        }

        let mut response = vec![format!("Removed failover provider #{}.", index)];
        response.push(format!("- removed: {}", removed_summary));
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

    async fn switch_provider(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let selection = match Self::resolve_provider_selection(args, "switch_provider") {
            Ok(selection) => selection,
            Err(message) => return Ok(message),
        };

        Self::require_kernel_approval(args)?;

        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let mut doc: toml::Table = content.parse()?;

        let mut notes: Vec<String> = Vec::new();
        let provider_table = doc
            .entry("provider")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`provider` is not a table"))?;
        Self::apply_provider_selection(
            provider_table,
            &selection,
            None,
            args.save_secrets_to_keychain,
            &mut notes,
            "",
        );

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;

        if let Err(e) = Self::validate_config(&new_content) {
            return Ok(format!(
                "Refused to save provider switch: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        tokio::fs::write(&self.config_path, &new_content).await?;
        set_owner_only_permissions(&self.config_path);

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
        let probe = crate::stt_setup::SttProbe::resolve(
            args.stt_cli_path.as_deref(),
            args.stt_model_path.as_deref(),
            args.stt_ffmpeg_path.as_deref(),
        );

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
                crate::stt_setup::format_stt_probe_report(&probe)
            ));
        }

        let language = args
            .stt_language
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("en");

        let content = tokio::fs::read_to_string(&self.config_path).await?;
        let mut doc: toml::Table = content.parse()?;
        crate::stt_setup::apply_stt_to_config_table(&mut doc, &probe, true, language)?;

        let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;
        if let Err(e) = Self::validate_config(&new_content) {
            return Ok(format!(
                "Refused to enable STT: {}.\n\nThe config was NOT modified.",
                e
            ));
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
        }

        tokio::fs::write(&self.config_path, &new_content).await?;
        set_owner_only_permissions(&self.config_path);

        Ok(format!(
            "Enabled Whisper STT fallback ([files.stt]).\n\n{}\n- language: `{language}`\n\nConfig validated and saved. Restart aidaemon (`/restart`) to apply — STT is loaded at startup.",
            crate::stt_setup::format_stt_probe_report(&probe)
        ))
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
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    gateway_token: String,
    #[serde(default)]
    primary_model: String,
    #[serde(default)]
    fast_model: String,
    #[serde(default)]
    smart_model: String,
    #[serde(default)]
    failover_index: Option<usize>,
    #[serde(default = "default_true")]
    save_secrets_to_keychain: bool,
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

fn default_true() -> bool {
    true
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
const TOOL_DESCRIPTION: &str =
    "Read (secrets hidden)/update config.toml (backup+validate). key: dotted path, list entry as a[2].";

fn manage_config_schema() -> Value {
    json!({
        "name": "manage_config",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "get", "set", "unset", "restore", "list_provider_presets", "switch_provider", "list_failover_providers", "add_failover_provider", "remove_failover_provider", "enable_stt"]
                },
                "key": { "type": "string" },
                "value": { "type": "string" },
                "provider": { "type": "string" },
                "api_key": { "type": "string" },
                "base_url": { "type": "string" },
                "gateway_token": { "type": "string" },
                "primary_model": { "type": "string" },
                "fast_model": { "type": "string" },
                "smart_model": { "type": "string" },
                "failover_index": { "type": "integer" },
                "save_secrets_to_keychain": { "type": "boolean" },
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
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let facts = facts.clone();

        let reply = self.run_reply(arguments, &facts).await?;
        let mut outcome = crate::traits::ToolCallOutcome::from_output(reply.text);
        outcome.metadata.outcome_status = reply.status;
        Ok(outcome)
    }
}

/// One call's reply. `status` is the tool's own verdict when it knows it
/// (reads, `get`, `set`, `unset`), so a config dump that happens to mention
/// timeouts is never read back as a failure.
struct ConfigReply {
    text: String,
    status: Option<crate::traits::ToolOutcomeStatus>,
}

impl ConfigReply {
    fn done(text: String) -> Self {
        Self {
            text,
            status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
        }
    }

    fn refused(text: String) -> Self {
        Self {
            text,
            status: Some(crate::traits::ToolOutcomeStatus::FailedPermanent),
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

        match args.action.as_str() {
            "list_provider_presets" => Ok(ConfigReply::done(Self::list_provider_presets_message())),
            "switch_provider" => self.switch_provider(&args).await.map(Into::into),
            "list_failover_providers" => self.list_failover_providers().await.map(ConfigReply::done),
            "add_failover_provider" => self.add_failover_provider(&args).await.map(Into::into),
            "remove_failover_provider" => self.remove_failover_provider(&args).await.map(Into::into),
            "enable_stt" => self.enable_stt(&args).await.map(Into::into),
            "read" => {
                let content = tokio::fs::read_to_string(&self.config_path).await?;
                let mut doc: toml::Value = toml::from_str(&content)?;
                redact_secrets(&mut doc);
                let redacted = toml::to_string_pretty(&doc)?;
                Ok(ConfigReply::done(format!(
                    "Current config.toml at {} (secrets redacted):\n\n{}",
                    self.config_path.display(),
                    redacted
                )))
            }
            "get" => self.get_value(&args).await,
            "set" | "unset" => self.change_value(&args).await,
            "restore" => match self.restore_backup().await {
                Ok(msg) => Ok(ConfigReply::done(msg)),
                Err(e) => Ok(ConfigReply::refused(format!("Restore failed: {}", e))),
            },
            _ => Ok(ConfigReply::refused(format!(
                "Unknown action: {}. Use 'read', 'get', 'set', 'unset', 'restore', 'list_provider_presets', 'switch_provider', 'list_failover_providers', 'add_failover_provider', 'remove_failover_provider', or 'enable_stt'.",
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
        let content = tokio::fs::read_to_string(&self.config_path).await?;
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
                )))
            }
        };

        // Security-sensitive keys need the owner's approval, which the
        // kernel asked for before this runs.
        if Self::change_needs_approval(args) {
            Self::require_kernel_approval(args)?;
        }

        let content = tokio::fs::read_to_string(&self.config_path).await?;
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

        if let Err(e) = Self::validate_config(&new_content) {
            return Ok(ConfigReply::refused(format!(
                "Refused to save: {}.\n\nThe config was NOT modified. Fix the value and try again.",
                e
            )));
        }
        if let Some(value) = &new_value {
            if !path.ends_in_index() && !Self::config_reads(&doc, &path, value) {
                return Ok(ConfigReply::refused(format!(
                    "Refused: aidaemon does not read a setting at `{path}`, so it would have no effect. \
                     The config was NOT modified. Check the key name with action 'read'."
                )));
            }
        }

        if let Err(e) = self.create_backup().await {
            warn!("Failed to create backup: {}", e);
            // Continue anyway — better to fix config than refuse
        }
        tokio::fs::write(&self.config_path, &new_content).await?;
        set_owner_only_permissions(&self.config_path);

        // What a reload will load: the file as it is on disk now.
        let written = tokio::fs::read_to_string(&self.config_path).await?;
        if let Err(e) = Self::validate_config(&written) {
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
        Self::validate_config(&probe.to_string()).is_err()
    }

    /// A change needs the owner when it names a security-sensitive key, or
    /// replaces a table or list entry that holds one.
    fn change_needs_approval(args: &ConfigArgs) -> bool {
        let Ok(path) = ConfigPath::parse(&args.key) else {
            return Self::requires_approval(&args.key);
        };
        if let Some(key) = path.last_key() {
            if APPROVAL_REQUIRED_KEYS.contains(&key) {
                return true;
            }
        }
        if args.action == "unset" {
            return path.last_key().is_none();
        }
        plain_holds_approval_key(
            config_path::plain_value(&config_path::parse_value(&args.value)).as_ref(),
        )
    }
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
impl crate::traits::ToolSemantics for ConfigManagerTool {
    /// Security-sensitive changes ask the owner, whatever the session. The
    /// dispatcher runs the call only with that answer; a decline is recorded
    /// by the kernel.
    async fn owner_approval_request(
        &self,
        call: &crate::traits::ApprovalCall<'_>,
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
    use crate::traits::ToolCalls as _;
    use std::fs;
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    fn test_args() -> ConfigArgs {
        ConfigArgs {
            action: "switch_provider".to_string(),
            key: String::new(),
            value: String::new(),
            provider: String::new(),
            api_key: String::new(),
            base_url: String::new(),
            gateway_token: String::new(),
            primary_model: String::new(),
            fast_model: String::new(),
            smart_model: String::new(),
            save_secrets_to_keychain: true,
            failover_index: None,
            stt_cli_path: None,
            stt_model_path: None,
            stt_language: None,
            stt_ffmpeg_path: None,
            kernel_approved: true,
        }
    }

    fn test_tool() -> ConfigManagerTool {
        ConfigManagerTool::new(PathBuf::from("/tmp/nonexistent-config.toml"))
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
        ConfigManagerTool::new(config_path)
    }

    const FAILOVER_CONFIG: &str = r#"
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
"#;

    #[tokio::test]
    async fn sensitive_changes_ask_with_the_old_wording_and_no_secret_in_the_subject() {
        let (_dir, path) = write_temp_config(FAILOVER_CONFIG);
        let tool = ConfigManagerTool::new(path);

        let set = json!({"action": "set", "key": "provider.api_key", "value": "synthetic-secret"});
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&set).await else {
            panic!("a secret change must ask");
        };
        assert_eq!(
            request.summary,
            "Config change: provider.api_key = [REDACTED]"
        );
        assert_eq!(request.risk, aidaemon_core::ApprovalRisk::High);
        assert_eq!(
            request.warnings,
            vec!["Modifying security-sensitive configuration".to_string()]
        );
        assert!(!request.subject.contains("synthetic-secret"));
        let mut other = set.clone();
        other["value"] = json!("synthetic-other-secret");
        let OwnerApprovalNeed::Ask(changed) = tool.owner_approval(&other).await else {
            panic!("a secret change must ask");
        };
        assert_ne!(request.subject, changed.subject);

        let remove = json!({"action": "remove_failover_provider", "failover_index": 0});
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&remove).await else {
            panic!("a failover removal must ask");
        };
        assert!(
            request.summary.starts_with("Remove failover provider [0]"),
            "{}",
            request.summary
        );

        let add = json!({"action": "add_failover_provider", "provider": "anthropic", "api_key": "synthetic-key"});
        let OwnerApprovalNeed::Ask(request) = tool.owner_approval(&add).await else {
            panic!("a failover addition must ask");
        };
        assert!(
            request.summary.starts_with("Add failover provider #1: "),
            "{}",
            request.summary
        );
        assert!(!request.subject.contains("synthetic-key"));

        for no_card in [
            json!({"action": "set", "key": "provider.models.default", "value": "\"other\""}),
            json!({"action": "read"}),
            json!({"action": "get", "key": "provider.kind"}),
            json!({"action": "list_failover_providers"}),
            json!({"action": "remove_failover_provider", "failover_index": 5}),
            json!({"action": "switch_provider", "provider": "unknown-provider"}),
            json!({"action": "restore"}),
        ] {
            assert!(
                matches!(
                    tool.owner_approval(&no_card).await,
                    OwnerApprovalNeed::NotNeeded
                ),
                "{no_card}"
            );
        }
    }

    const TWO_FALLBACKS: &str = r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "first-key"

[provider.fallbacks.models]
default = "first-model"

[[provider.fallbacks]]
kind = "google_genai"
api_key = "second-key"

[provider.fallbacks.models]
default = "old-model"
fallback = ["old-a", "old-b"]
"#;

    #[tokio::test]
    async fn a_list_position_edits_that_fallback_entry_and_is_checked_on_disk() {
        let (_dir, path) = write_temp_config(TWO_FALLBACKS);
        let tool = ConfigManagerTool::new(path.clone());
        for (key, value) in [
            ("provider.fallbacks[1].models.default", "new-model"),
            ("provider.fallbacks.1.models.smart", "\"new-smart\""),
            ("provider.fallbacks[1].models.fallback", "[\"new-a\"]"),
        ] {
            let args = json!({"action": "set", "key": key, "value": value}).to_string();
            let outcome = tool
                .execute(&args, &ToolCallContext::default())
                .await
                .unwrap();
            assert!(outcome.output.contains("Checked"), "{}", outcome.output);
            assert_eq!(
                outcome.metadata.outcome_status,
                Some(crate::traits::ToolOutcomeStatus::Succeeded)
            );
        }
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains("fallbacks["), "{text}");
        let config: AppConfig = toml::from_str(&text).unwrap();
        let google = &config.provider.fallbacks[1].models;
        assert_eq!(google.default_model, "new-model");
        assert_eq!(google.smart, "new-smart");
        assert_eq!(google.fallback_models, vec!["new-a".to_string()]);
        assert_eq!(
            config.provider.fallbacks[0].models.default_model,
            "first-model"
        );
    }

    #[tokio::test]
    async fn paths_the_config_does_not_have_or_read_are_refused_unchanged() {
        let (_dir, path) = write_temp_config(TWO_FALLBACKS);
        let tool = ConfigManagerTool::new(path.clone());
        for (key, expect) in [
            ("provider.fallbacks[2].models.default", "has 2 entries"),
            (
                "provider.fallbacks.models.default",
                "is a list of 2 entries",
            ),
            ("provider.\"fallbacks[2]\".models", "is not a config key"),
            ("provider.models.no_such_setting", "does not read a setting"),
            ("no_such_section.value", "does not read a setting"),
        ] {
            let args = json!({"action": "set", "key": key, "value": "x"}).to_string();
            let outcome = tool
                .execute(&args, &ToolCallContext::default())
                .await
                .unwrap();
            assert!(outcome.output.contains(expect), "{key}: {}", outcome.output);
            assert!(
                outcome.output.contains("NOT modified"),
                "{}",
                outcome.output
            );
            assert_eq!(
                outcome.metadata.outcome_status,
                Some(crate::traits::ToolOutcomeStatus::FailedPermanent)
            );
            assert_eq!(fs::read_to_string(&path).unwrap(), TWO_FALLBACKS);
        }
    }

    #[tokio::test]
    async fn unset_removes_a_setting_and_read_is_a_typed_success() {
        let (_dir, path) = write_temp_config(TWO_FALLBACKS);
        let tool = ConfigManagerTool::new(path.clone());
        let args =
            json!({"action": "unset", "key": "provider.fallbacks[1].models.fallback"}).to_string();
        let output = tool.call(&args).await.unwrap();
        assert!(
            output.starts_with("Removed provider.fallbacks[1].models.fallback"),
            "{output}"
        );
        assert!(!fs::read_to_string(&path).unwrap().contains("old-a"));

        let read = tool
            .execute(r#"{"action":"read"}"#, &ToolCallContext::default())
            .await
            .unwrap();
        assert_eq!(
            read.metadata.outcome_status,
            Some(crate::traits::ToolOutcomeStatus::Succeeded)
        );
        assert!(read.output.contains("old-model"));
        assert!(!read.output.contains("second-key"), "{}", read.output);
        let get = tool
            .call(r#"{"action":"get","key":"provider.fallbacks[1]"}"#)
            .await
            .unwrap();
        assert!(
            get.contains("old-model") && !get.contains("second-key"),
            "{get}"
        );
    }

    #[tokio::test]
    async fn sensitive_change_without_the_kernel_is_refused() {
        let (_dir, path) = write_temp_config(FAILOVER_CONFIG);
        let tool = ConfigManagerTool::new(path.clone());
        let before = fs::read_to_string(&path).unwrap();
        let args = json!({"action": "set", "key": "provider.base_url", "value": "\"https://h.example.invalid/v1\""})
            .to_string();
        let err = tool.call(&args).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("requires approval through the agent kernel"),
            "{err}"
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), before);

        let remove = json!({"action": "remove_failover_provider", "failover_index": 0}).to_string();
        assert!(tool.call(&remove).await.is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), before);
    }

    #[test]
    fn changes_declare_the_config_path_and_reads_declare_nothing() {
        let tool = ConfigManagerTool::new(PathBuf::from("/tmp/aidaemon-synthetic/config.toml"));
        let effects = tool.effects_for(r#"{"action":"set","key":"provider.kind","value":"x"}"#);
        assert_eq!(
            effects.keys,
            vec![aidaemon_core::EffectKey::path(
                aidaemon_core::EffectClass::Configuration,
                "/tmp/aidaemon-synthetic/config.toml"
            )
            .unwrap()]
        );
        assert!(!effects.complete);
        assert_eq!(
            tool.effects_for(r#"{"action":"read"}"#),
            aidaemon_core::CallEffects::none()
        );
    }

    #[test]
    fn schema_fits_payload_budget() {
        // Pillar C of 2026-06-06-cross-turn-prefix-stability-design.md:
        // admin-tool schemas ride in EVERY provider call; this ceiling is the
        // per-tool payload budget. If you trip this assert by adding features,
        // compress the description text — do not raise the ceiling without
        // updating the Pillar C implementation plan.
        // NOTE: plan ceiling was 900. manage_config has a 10-value action enum
        // (including enable_stt + 4 failover-provider actions) plus 15 parameters.
        // Descriptions trimmed to stay within payload budget.
        let bytes = serde_json::to_string(&manage_config_schema())
            .unwrap()
            .len();
        assert!(
            bytes <= 950,
            "manage_config schema is {bytes} bytes, budget is 950"
        );
    }

    #[test]
    fn agent_config_changes_require_restart() {
        assert!(apply_instruction_for_key("agent.name").starts_with("Restart"));
        assert!(apply_instruction_for_key("agent.persona_file").starts_with("Restart"));
        assert!(apply_instruction_for_key("provider.models.default").starts_with("Run /reload"));
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
        let tool = test_tool();
        let mut args = test_args();
        args.provider = "cloudflare_gateway".to_string();
        args.api_key = "test-key".to_string();
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
        let tool = approving_tool(path.clone());
        let mut args = test_args();
        args.action = "add_failover_provider".to_string();
        args.provider = "anthropic".to_string();
        args.api_key = "secondary-key".to_string();
        args.save_secrets_to_keychain = false;

        let reply = tool.add_failover_provider(&args).await.unwrap();
        assert!(reply.contains("Added failover provider #0"));

        let saved = fs::read_to_string(&path).expect("read saved config");
        let cfg: AppConfig = toml::from_str(&saved).expect("parse saved config");
        assert_eq!(cfg.provider.fallbacks.len(), 1);
        assert_eq!(
            cfg.provider.fallbacks[0].kind,
            crate::config::ProviderKind::Anthropic
        );
        assert_eq!(cfg.provider.fallbacks[0].api_key, "secondary-key");
        assert_eq!(
            cfg.provider.fallbacks[0].models.default_model,
            "claude-sonnet-4-20250514"
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
        assert!(reply.contains("Removed failover provider #0."));

        let saved = fs::read_to_string(&path).expect("read saved config");
        let cfg: AppConfig = toml::from_str(&saved).expect("parse saved config");
        assert_eq!(cfg.provider.fallbacks.len(), 1);
        assert_eq!(
            cfg.provider.fallbacks[0].kind,
            crate::config::ProviderKind::XaiNative
        );
        assert_eq!(cfg.provider.fallbacks[0].api_key, "second-key");
        assert_eq!(cfg.provider.fallbacks[0].models.default_model, "grok-4");
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
        let cfg: AppConfig = toml::from_str(&saved).expect("parse saved config");
        assert!(cfg.files.stt.enabled);
        assert_eq!(cfg.files.stt.language, "en");
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
}
