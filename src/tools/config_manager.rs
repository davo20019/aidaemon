use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::AppConfig;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::traits::{Tool, ToolCapabilities};
use crate::types::ApprovalResponse;

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
    approval_tx: mpsc::Sender<ApprovalRequest>,
}

/// Set file permissions to owner-only read/write (0600) on Unix.
/// This is a best-effort operation — failures are logged but not fatal.
#[cfg(unix)]
fn set_owner_only_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        warn!(path = %path.display(), "Failed to set 0600 permissions: {}", e);
    }
}

#[cfg(not(unix))]
fn set_owner_only_permissions(_path: &Path) {
    // No-op on non-Unix platforms
}

impl ConfigManagerTool {
    pub fn new(config_path: PathBuf, approval_tx: mpsc::Sender<ApprovalRequest>) -> Self {
        Self {
            config_path,
            approval_tx,
        }
    }

    /// Check if a key path requires user approval to modify.
    fn requires_approval(key_path: &str) -> bool {
        let last = key_path.rsplit('.').next().unwrap_or(key_path);
        APPROVAL_REQUIRED_KEYS.contains(&last)
    }

    /// Request user approval for a config change.
    async fn request_approval(
        &self,
        session_id: &str,
        description: &str,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: description.to_string(),
                session_id: session_id.to_string(),
                risk_level: RiskLevel::High,
                warnings: vec!["Modifying security-sensitive configuration".to_string()],
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: Default::default(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;
        match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                tracing::warn!(description, "Approval response channel closed");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                tracing::warn!(
                    description,
                    "Approval request timed out (300s), auto-denying"
                );
                Ok(ApprovalResponse::Deny)
            }
        }
    }

    /// Rotate backups (3-deep ring) and create a new .bak from current config.
    /// Rotation: .bak.2 is deleted, .bak.1 → .bak.2, .bak → .bak.1, current → .bak
    async fn create_backup(&self) -> anyhow::Result<()> {
        let bak = self.config_path.with_extension("toml.bak");
        let bak1 = self.config_path.with_extension("toml.bak.1");
        let bak2 = self.config_path.with_extension("toml.bak.2");

        // Rotate: .bak.1 → .bak.2
        if bak1.exists() {
            let _ = tokio::fs::rename(&bak1, &bak2).await;
        }
        // Rotate: .bak → .bak.1
        if bak.exists() {
            let _ = tokio::fs::rename(&bak, &bak1).await;
        }
        // Current → .bak
        tokio::fs::copy(&self.config_path, &bak).await?;
        // Restrict permissions on all backup files (contain secrets)
        set_owner_only_permissions(&bak);
        set_owner_only_permissions(&bak1);
        set_owner_only_permissions(&bak2);
        info!(backup = %bak.display(), "Config backup created (3-deep rotation)");
        Ok(())
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
    fn validate_config(content: &str) -> Result<(), String> {
        // First check it's valid TOML
        let _doc: toml::Table = content
            .parse()
            .map_err(|e| format!("Invalid TOML syntax: {}", e))?;

        // Expand env vars so "${VAR}" values don't break type validation
        let expanded = crate::config::expand_env_vars(content).map_err(|e| format!("{}", e))?;

        // Then check it deserializes into our AppConfig
        toml::from_str::<AppConfig>(&expanded)
            .map_err(|e| format!("Invalid config structure: {}", e))?;

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
        format!("provider_fallback_{}", index)
    }

    fn keychain_field_name(prefix: Option<&str>, field: &str) -> String {
        prefix
            .map(|prefix| format!("{}_{}", prefix, field))
            .unwrap_or_else(|| field.to_string())
    }

    fn normalized_header_key(header_name: &str) -> String {
        header_name
            .to_ascii_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }

    fn header_keychain_field_name(prefix: Option<&str>, header_name: &str) -> String {
        let normalized = Self::normalized_header_key(header_name);
        prefix
            .map(|prefix| format!("{}_header_{}", prefix, normalized))
            .unwrap_or_else(|| format!("provider_header_{}", normalized))
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
            toml::Value::String(if selection.api_key.is_empty() {
                "ollama".to_string()
            } else {
                selection.api_key.clone()
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
                let old_nested_prefix = format!("{}_fallback_{}", old_prefix, idx);
                let new_nested_prefix = format!("{}_fallback_{}", new_prefix, idx);
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

        let approval_description = format!(
            "Add failover provider #{}: {} (kind={}, base_url={}, models={}/{}/{})",
            next_index,
            selection.preset.display_name,
            selection.preset.kind,
            if selection.base_url.is_empty() {
                "(not used)"
            } else {
                selection.base_url.as_str()
            },
            selection.primary_model,
            selection.fast_model,
            selection.smart_model
        );
        match self
            .request_approval(&args._session_id, &approval_description)
            .await
        {
            Ok(ApprovalResponse::AllowOnce)
            | Ok(ApprovalResponse::AllowSession)
            | Ok(ApprovalResponse::AllowAlways) => {}
            Ok(ApprovalResponse::Deny) => {
                return Ok("Failover provider addition denied by user.".to_string());
            }
            Err(e) => {
                return Ok(format!("Could not get approval: {}", e));
            }
        }

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

            let removed_summary = current_entries[index]
                .as_table()
                .map(|table| Self::summarize_provider_table(index, table))
                .unwrap_or_else(|| format!("[{}] (invalid provider entry)", index));
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
        let approval_description = format!("Remove failover provider {}", removed_summary);
        match self
            .request_approval(&args._session_id, &approval_description)
            .await
        {
            Ok(ApprovalResponse::AllowOnce)
            | Ok(ApprovalResponse::AllowSession)
            | Ok(ApprovalResponse::AllowAlways) => {}
            Ok(ApprovalResponse::Deny) => {
                return Ok("Failover provider removal denied by user.".to_string());
            }
            Err(e) => {
                return Ok(format!("Could not get approval: {}", e));
            }
        }

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

        let approval_description = format!(
            "Switch provider to {} (kind={}, base_url={}, models={}/{}/{})",
            selection.preset.display_name,
            selection.preset.kind,
            if selection.base_url.is_empty() {
                "(not used)"
            } else {
                selection.base_url.as_str()
            },
            selection.primary_model,
            selection.fast_model,
            selection.smart_model
        );
        match self
            .request_approval(&args._session_id, &approval_description)
            .await
        {
            Ok(ApprovalResponse::AllowOnce)
            | Ok(ApprovalResponse::AllowSession)
            | Ok(ApprovalResponse::AllowAlways) => {}
            Ok(ApprovalResponse::Deny) => {
                return Ok("Provider switch denied by user.".to_string());
            }
            Err(e) => {
                return Ok(format!("Could not get approval: {}", e));
            }
        }

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
    /// Session ID for routing approval requests (injected by agent).
    #[serde(default)]
    _session_id: String,
}

fn default_true() -> bool {
    true
}

#[async_trait]
impl Tool for ConfigManagerTool {
    fn name(&self) -> &str {
        "manage_config"
    }

    fn description(&self) -> &str {
        "Read or update aidaemon's own config.toml, including guided primary-provider and failover-provider changes. Automatically backs up before changes and validates before saving."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_config",
            "description": "Read or update aidaemon config.toml with backup and validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "get", "set", "restore", "list_provider_presets", "switch_provider", "list_failover_providers", "add_failover_provider", "remove_failover_provider"],
                        "description": "Config action"
                    },
                    "key": {
                        "type": "string",
                        "description": "TOML key path for get/set"
                    },
                    "value": {
                        "type": "string",
                        "description": "New TOML literal for set"
                    },
                    "provider": {
                        "type": "string",
                        "description": "Provider preset"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Provider API key"
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Optional provider base URL"
                    },
                    "gateway_token": {
                        "type": "string",
                        "description": "Optional gateway token"
                    },
                    "primary_model": {
                        "type": "string",
                        "description": "Optional primary model override"
                    },
                    "fast_model": {
                        "type": "string",
                        "description": "Optional fast model override"
                    },
                    "smart_model": {
                        "type": "string",
                        "description": "Optional smart model override"
                    },
                    "failover_index": {
                        "type": "integer",
                        "description": "Failover index for removal"
                    },
                    "save_secrets_to_keychain": {
                        "type": "boolean",
                        "description": "Store API keys in keychain"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ConfigArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "list_provider_presets" => Ok(Self::list_provider_presets_message()),
            "switch_provider" => self.switch_provider(&args).await,
            "list_failover_providers" => self.list_failover_providers().await,
            "add_failover_provider" => self.add_failover_provider(&args).await,
            "remove_failover_provider" => self.remove_failover_provider(&args).await,
            "read" => {
                let content = tokio::fs::read_to_string(&self.config_path).await?;
                let mut doc: toml::Value = content.parse()?;
                redact_secrets(&mut doc);
                let redacted = toml::to_string_pretty(&doc)?;
                Ok(format!(
                    "Current config.toml (secrets redacted):\n\n{}",
                    redacted
                ))
            }
            "get" => {
                if args.key.is_empty() {
                    return Ok("Error: 'key' is required for 'get' action.".to_string());
                }
                // Block direct reads of sensitive keys
                if is_sensitive_key(&args.key) {
                    return Ok(format!(
                        "{} = \"{}\" (redacted for security)",
                        args.key, REDACTED
                    ));
                }
                let content = tokio::fs::read_to_string(&self.config_path).await?;
                let doc: toml::Value = content.parse()?;
                let value = navigate_toml(&doc, &args.key);
                match value {
                    Some(v) => Ok(format!("{} = {}", args.key, v)),
                    None => Ok(format!("Key '{}' not found in config.", args.key)),
                }
            }
            "set" => {
                if args.key.is_empty() || args.value.is_empty() {
                    return Ok(
                        "Error: 'key' and 'value' are required for 'set' action.".to_string()
                    );
                }

                // Require user approval for security-sensitive keys
                if Self::requires_approval(&args.key) {
                    // Show redacted value for secrets, actual value for other sensitive keys
                    let display_value = if is_sensitive_key(&args.key) {
                        "[REDACTED]".to_string()
                    } else {
                        args.value.clone()
                    };
                    let description = format!("Config change: {} = {}", args.key, display_value);

                    match self.request_approval(&args._session_id, &description).await {
                        Ok(ApprovalResponse::AllowOnce)
                        | Ok(ApprovalResponse::AllowSession)
                        | Ok(ApprovalResponse::AllowAlways) => {
                            info!(key = %args.key, "Config change approved by user");
                        }
                        Ok(ApprovalResponse::Deny) => {
                            return Ok("Config change denied by user.".to_string());
                        }
                        Err(e) => {
                            return Ok(format!("Could not get approval: {}", e));
                        }
                    }
                }

                let content = tokio::fs::read_to_string(&self.config_path).await?;
                let mut doc: toml::Table = content.parse()?;

                // Parse the new value as TOML
                let new_value: toml::Value = args.value.parse().or_else(|_| {
                    let wrapped = format!("v = {}", args.value);
                    let table: toml::Table = wrapped.parse()?;
                    Ok::<toml::Value, toml::de::Error>(table["v"].clone())
                })?;

                set_toml_value(&mut doc, &args.key, new_value)?;

                let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;

                // Validate before writing
                if let Err(e) = Self::validate_config(&new_content) {
                    return Ok(format!(
                        "Refused to save: {}.\n\nThe config was NOT modified. Fix the value and try again.",
                        e
                    ));
                }

                // Backup current config before writing
                if let Err(e) = self.create_backup().await {
                    warn!("Failed to create backup: {}", e);
                    // Continue anyway — better to fix config than refuse
                }

                tokio::fs::write(&self.config_path, &new_content).await?;
                set_owner_only_permissions(&self.config_path);

                Ok(format!(
                    "Updated {} = {}\n\nBackup saved to config.toml.bak. Config validated and saved. Run /reload to apply changes.",
                    args.key, args.value
                ))
            }
            "restore" => match self.restore_backup().await {
                Ok(msg) => Ok(msg),
                Err(e) => Ok(format!("Restore failed: {}", e)),
            },
            _ => Ok(format!(
                "Unknown action: {}. Use 'read', 'get', 'set', 'restore', 'list_provider_presets', 'switch_provider', 'list_failover_providers', 'add_failover_provider', or 'remove_failover_provider'.",
                args.action
            )),
        }
    }
}

/// Navigate a TOML value by a dotted key path like "provider.models.primary".
fn navigate_toml<'a>(value: &'a toml::Value, path: &str) -> Option<&'a toml::Value> {
    let mut current = value;
    for key in path.split('.') {
        current = current.get(key)?;
    }
    Some(current)
}

/// Set a value in a TOML table at a dotted key path.
fn set_toml_value(table: &mut toml::Table, path: &str, value: toml::Value) -> anyhow::Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        anyhow::bail!("Empty key path");
    }

    let mut current = table;
    for &key in &parts[..parts.len() - 1] {
        current = current
            .entry(key)
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("'{}' is not a table", key))?;
    }

    let last_key = parts.last().unwrap();
    current.insert(last_key.to_string(), value);
    Ok(())
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
    let last = path.rsplit('.').next().unwrap_or(path);
    SENSITIVE_KEYS.contains(&last)
}

/// Recursively walk a TOML value and replace sensitive keys with a redacted placeholder.
fn redact_secrets(value: &mut toml::Value) {
    match value {
        toml::Value::Table(table) => {
            for (key, val) in table.iter_mut() {
                if SENSITIVE_KEYS.contains(&key.as_str()) {
                    *val = toml::Value::String(REDACTED.to_string());
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
    use tokio::sync::mpsc;

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
            _session_id: "test-session".to_string(),
        }
    }

    fn test_tool() -> ConfigManagerTool {
        let (tx, _rx) = mpsc::channel(1);
        ConfigManagerTool::new(PathBuf::from("/tmp/nonexistent-config.toml"), tx)
    }

    fn write_temp_config(contents: &str) -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, contents).expect("write temp config");
        (dir, path)
    }

    fn approving_tool(config_path: PathBuf) -> ConfigManagerTool {
        let (tx, mut rx) = mpsc::channel::<ApprovalRequest>(4);
        tokio::spawn(async move {
            while let Some(request) = rx.recv().await {
                let _ = request.response_tx.send(ApprovalResponse::AllowOnce);
            }
        });
        ConfigManagerTool::new(config_path, tx)
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
}
