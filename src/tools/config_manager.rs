use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::AppConfig;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::traits::Tool;
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
            "Use action='switch_provider' with provider + api_key (and base_url when needed)."
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

    async fn switch_provider(&self, args: &ConfigArgs) -> anyhow::Result<String> {
        let provider_name = args.provider.trim();
        if provider_name.is_empty() {
            return Ok(
                "Error: 'provider' is required for action='switch_provider'. Use action='list_provider_presets' to see valid options."
                    .to_string(),
            );
        }

        let Some(preset) = Self::find_provider_preset(provider_name) else {
            return Ok(format!(
                "Unknown provider '{}'. Use action='list_provider_presets' to see valid options.",
                provider_name
            ));
        };

        let api_key = args.api_key.trim();
        if preset.needs_api_key && api_key.is_empty() {
            return Ok(format!(
                "Missing API key for {}. Provide `api_key` and retry.\n\nExample: {{\"action\":\"switch_provider\",\"provider\":\"{}\",\"api_key\":\"YOUR_KEY\"}}",
                preset.display_name, preset.id
            ));
        }

        let mut base_url = args.base_url.trim().to_string();
        if base_url.is_empty() {
            base_url = preset.base_url.to_string();
        }
        if preset.requires_custom_base_url && base_url.trim().is_empty() {
            return Ok(format!(
                "Provider {} requires `base_url`. Example:\n{{\"action\":\"switch_provider\",\"provider\":\"{}\",\"base_url\":\"https://api.example.com/v1\",\"api_key\":\"YOUR_KEY\"}}",
                preset.display_name, preset.id
            ));
        }
        if preset.id == "cloudflare_gateway"
            && (base_url.contains("<ACCOUNT_ID>") || base_url.contains("<GATEWAY_ID>"))
        {
            return Ok(
                "Cloudflare AI Gateway requires your real gateway URL. Replace `<ACCOUNT_ID>` and `<GATEWAY_ID>` in `base_url` and retry."
                    .to_string(),
            );
        }

        let primary = if args.primary_model.trim().is_empty() {
            preset.primary.to_string()
        } else {
            args.primary_model.trim().to_string()
        };
        let fast = if args.fast_model.trim().is_empty() {
            preset.fast.to_string()
        } else {
            args.fast_model.trim().to_string()
        };
        let smart = if args.smart_model.trim().is_empty() {
            preset.smart.to_string()
        } else {
            args.smart_model.trim().to_string()
        };

        let gateway_token = if preset.supports_gateway_token && !args.gateway_token.trim().is_empty()
        {
            Some(args.gateway_token.trim().to_string())
        } else {
            None
        };

        let approval_description = format!(
            "Switch provider to {} (kind={}, base_url={}, models={}/{}/{})",
            preset.display_name,
            preset.kind,
            if base_url.is_empty() {
                "(not used)"
            } else {
                base_url.as_str()
            },
            primary,
            fast,
            smart
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

        let provider_api_key_value = if preset.needs_api_key {
            let (value, stored, note) = Self::resolve_secret_config_value(
                "api_key",
                api_key,
                args.save_secrets_to_keychain,
            );
            if stored {
                notes.push("API key stored in OS keychain.".to_string());
            }
            if let Some(n) = note {
                notes.push(n);
            }
            value
        } else {
            toml::Value::String(if api_key.is_empty() {
                "ollama".to_string()
            } else {
                api_key.to_string()
            })
        };

        set_toml_value(
            &mut doc,
            "provider.kind",
            toml::Value::String(preset.kind.to_string()),
        )?;
        set_toml_value(&mut doc, "provider.api_key", provider_api_key_value)?;

        if base_url.trim().is_empty() {
            remove_toml_value(&mut doc, "provider.base_url")?;
        } else {
            set_toml_value(
                &mut doc,
                "provider.base_url",
                toml::Value::String(base_url.clone()),
            )?;
        }

        set_toml_value(
            &mut doc,
            "provider.models.primary",
            toml::Value::String(primary.clone()),
        )?;
        set_toml_value(
            &mut doc,
            "provider.models.fast",
            toml::Value::String(fast.clone()),
        )?;
        set_toml_value(
            &mut doc,
            "provider.models.smart",
            toml::Value::String(smart.clone()),
        )?;

        if let Some(token) = gateway_token.as_deref() {
            let (token_value, stored, note) = Self::resolve_secret_config_value(
                "gateway_token",
                token,
                args.save_secrets_to_keychain,
            );
            if stored {
                notes.push("Gateway token stored in OS keychain.".to_string());
            }
            if let Some(n) = note {
                notes.push(n);
            }
            set_toml_value(&mut doc, "provider.gateway_token", token_value)?;
        } else {
            remove_toml_value(&mut doc, "provider.gateway_token")?;
        }

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
            format!("Switched provider to {}.", preset.display_name),
            format!("- kind: `{}`", preset.kind),
            format!(
                "- base_url: `{}`",
                if base_url.is_empty() {
                    "(not set)"
                } else {
                    &base_url
                }
            ),
            format!("- models: `{}` / `{}` / `{}`", primary, fast, smart),
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
        "Read or update aidaemon's own config.toml, including guided provider switching. Automatically backs up before changes and validates before saving."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_config",
            "description": "Read or update aidaemon's own config.toml. Backs up before writing and validates changes. Use 'restore' to rollback if something goes wrong. Use 'switch_provider' for guided provider changes. After updating, tell the user to run /reload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "get", "set", "restore", "list_provider_presets", "switch_provider"],
                        "description": "'read' = full config, 'get' = read key, 'set' = update key (auto-backup + validate), 'restore' = rollback to last backup, 'list_provider_presets' = show guided provider options, 'switch_provider' = guided provider switch with minimal fields"
                    },
                    "key": {
                        "type": "string",
                        "description": "TOML key path, e.g. 'provider.models.primary'. Required for get/set."
                    },
                    "value": {
                        "type": "string",
                        "description": "New value (TOML literal, e.g. '\"gemini-2.0-flash\"' or '[\"ls\", \"git\"]'). Required for set."
                    },
                    "provider": {
                        "type": "string",
                        "description": "Provider preset for switch_provider (e.g. moonshot, minimax, openai, cloudflare_gateway, custom_openai_compatible)."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "Provider API key for switch_provider."
                    },
                    "base_url": {
                        "type": "string",
                        "description": "Optional override base URL for switch_provider. Required for cloudflare_gateway and custom_openai_compatible."
                    },
                    "gateway_token": {
                        "type": "string",
                        "description": "Optional Cloudflare gateway token for switch_provider."
                    },
                    "primary_model": {
                        "type": "string",
                        "description": "Optional primary model override for switch_provider."
                    },
                    "fast_model": {
                        "type": "string",
                        "description": "Optional fast model override for switch_provider."
                    },
                    "smart_model": {
                        "type": "string",
                        "description": "Optional smart model override for switch_provider."
                    },
                    "save_secrets_to_keychain": {
                        "type": "boolean",
                        "description": "When true (default), API keys are stored in OS keychain and config uses \"keychain\" sentinel."
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ConfigArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "list_provider_presets" => Ok(Self::list_provider_presets_message()),
            "switch_provider" => self.switch_provider(&args).await,
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
                "Unknown action: {}. Use 'read', 'get', 'set', 'restore', 'list_provider_presets', or 'switch_provider'.",
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
    use std::path::PathBuf;

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
            _session_id: "test-session".to_string(),
        }
    }

    fn test_tool() -> ConfigManagerTool {
        let (tx, _rx) = mpsc::channel(1);
        ConfigManagerTool::new(PathBuf::from("/tmp/nonexistent-config.toml"), tx)
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
    }

    #[test]
    fn list_provider_presets_message_includes_expected_entries() {
        let msg = ConfigManagerTool::list_provider_presets_message();
        assert!(msg.contains("cloudflare_gateway"));
        assert!(msg.contains("custom_openai_compatible"));
        assert!(msg.contains("\"action\":\"switch_provider\""));
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
}
