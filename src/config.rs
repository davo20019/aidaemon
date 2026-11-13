use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::types::WorkspaceGrant;

/// Helper to format a redacted secret field for Debug output.
/// Shows "[REDACTED]" if the value is non-empty, "\"\"" if empty.
fn redact(secret: &str) -> &'static str {
    if secret.is_empty() {
        "\"\""
    } else {
        "[REDACTED]"
    }
}

/// Helper to format an optional redacted secret field for Debug output.
fn redact_option(secret: &Option<String>) -> &'static str {
    match secret {
        Some(s) if !s.is_empty() => "Some([REDACTED])",
        Some(_) => "Some(\"\")",
        None => "None",
    }
}

pub(crate) const KEYCHAIN_SERVICE: &str = "aidaemon";

/// Expand `${ENV_VAR}` references in a string.
/// Returns an error listing all undefined variables (does not fail on first miss).
pub fn expand_env_vars(content: &str) -> anyhow::Result<String> {
    let re = regex::Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}").unwrap();
    let mut missing: Vec<String> = Vec::new();
    let result = re.replace_all(content, |caps: &regex::Captures| {
        let var_name = &caps[1];
        match std::env::var(var_name) {
            Ok(val) => val,
            Err(_) => {
                missing.push(var_name.to_string());
                caps[0].to_string()
            }
        }
    });
    if !missing.is_empty() {
        anyhow::bail!(
            "Config references undefined environment variable(s): {}",
            missing.join(", ")
        );
    }
    Ok(result.into_owned())
}

/// The daemon's secret store: the OS keychain under the `aidaemon` service,
/// or the runtime env file while `AIDAEMON_NO_KEYCHAIN` is `1` or `true`.
pub fn secret_store() -> &'static std::sync::Arc<aidaemon_secrets::KeychainSecretStore> {
    static STORE: std::sync::OnceLock<std::sync::Arc<aidaemon_secrets::KeychainSecretStore>> =
        std::sync::OnceLock::new();
    STORE.get_or_init(|| {
        std::sync::Arc::new(aidaemon_secrets::KeychainSecretStore::new(
            aidaemon_secrets::KeychainSettings {
                service: KEYCHAIN_SERVICE.to_string(),
                disable_var: Some("AIDAEMON_NO_KEYCHAIN".to_string()),
                env_file_vars: vec![
                    crate::RUNTIME_ENV_FILE_ENV_KEY.to_string(),
                    "AIDAEMON_ENV_FILE".to_string(),
                ],
                default_env_file: PathBuf::from(".env"),
            },
        ))
    })
}

/// The daemon's secret store as a shared trait object, for the crates.
pub fn shared_secret_store() -> std::sync::Arc<dyn aidaemon_secrets::SecretStore> {
    secret_store().clone()
}

#[cfg(test)]
fn resolve_env_file_path() -> PathBuf {
    secret_store().env_store().env_file_path()
}

/// Read a secret from the daemon's secret store. The value is registered
/// for redaction on the way out (`aidaemon_secrets::Secret`).
pub fn resolve_from_keychain(field_name: &str) -> anyhow::Result<String> {
    aidaemon_secrets::SecretStore::get(secret_store().as_ref(), field_name)
        .map(aidaemon_secrets::Secret::into_exposed)
}

/// Store a secret in the OS keychain under the `aidaemon` service.
pub fn store_in_keychain(field_name: &str, value: &str) -> anyhow::Result<()> {
    aidaemon_secrets::SecretStore::set(secret_store().as_ref(), field_name, value)
}

/// Delete a secret from the OS keychain.
pub fn delete_from_keychain(field_name: &str) -> anyhow::Result<()> {
    aidaemon_secrets::SecretStore::delete(secret_store().as_ref(), field_name)
}

/// The daemon's check of a whole config text before a tool writes it: TOML
/// syntax, then `${ENV_VAR}` references expanded, then [`AppConfig`] and
/// the checks startup makes (agent name, exposure). `"keychain"` sentinels
/// are not resolved: the value may not be stored yet.
pub fn check_config_text(content: &str) -> Result<(), String> {
    let _doc: toml::Table = content
        .parse()
        .map_err(|e| format!("Invalid TOML syntax: {}", e))?;
    let expanded = expand_env_vars(content).map_err(|e| format!("{}", e))?;
    let config = toml::from_str::<AppConfig>(&expanded)
        .map_err(|e| format!("Invalid config structure: {}", e))?;
    config
        .agent
        .validated_name()
        .map_err(|e| format!("Invalid agent configuration: {}", e))?;
    config
        .exposure
        .validate(config.client_gateway.enabled, config.nodes.gateway.enabled)?;
    Ok(())
}

/// The daemon's config file as the config-editing tools see it.
pub fn config_file(path: impl Into<PathBuf>) -> aidaemon_tools::config_file::ConfigFile {
    aidaemon_tools::config_file::ConfigFile::new(path, std::sync::Arc::new(check_config_text))
}

/// Where to export run traces. Nothing is exported unless an endpoint is
/// set, and prompts and replies only with `capture_content`.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct TelemetryConfig {
    /// OTLP/HTTP base URL, for example `https://api.smith.langchain.com/otel`.
    #[serde(default)]
    pub otlp_endpoint: Option<String>,
    /// Request headers. A value of `keychain` is read from the OS keychain
    /// under `otlp_header_<name>`.
    #[serde(default)]
    pub otlp_headers: HashMap<String, String>,
    /// Also export prompts, replies and tool arguments, with secrets removed.
    #[serde(default)]
    pub capture_content: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub agent: AgentConfig,
    #[serde(default)]
    pub execution: ExecutionConfig,
    #[serde(default)]
    pub checkpoints: CheckpointConfig,
    pub provider: ProviderConfig,
    /// Legacy single Telegram bot config (for backward compatibility).
    #[serde(default)]
    pub telegram: Option<TelegramConfig>,
    /// Array of named Telegram bots (preferred for multi-bot setups).
    #[serde(default)]
    pub telegram_bots: Vec<TelegramBotConfig>,
    /// Global webhook defaults — auto-applied to any Telegram bot without
    /// an explicit `[webhook]` section with `enabled = true`.
    #[serde(default)]
    pub telegram_webhook_defaults: TelegramWebhookDefaults,
    #[cfg(feature = "discord")]
    #[serde(default)]
    pub discord: Option<DiscordConfig>,
    /// Array of named Discord bots (preferred for multi-bot setups).
    #[cfg(feature = "discord")]
    #[serde(default)]
    pub discord_bots: Vec<DiscordBotConfig>,
    #[cfg(feature = "slack")]
    #[serde(default)]
    pub slack: Option<SlackConfig>,
    /// Array of named Slack bots (preferred for multi-bot setups).
    #[cfg(feature = "slack")]
    #[serde(default)]
    pub slack_bots: Vec<SlackBotConfig>,
    #[serde(default)]
    pub state: StateConfig,
    #[serde(default)]
    pub terminal: TerminalConfig,
    /// Individually enrolled, policy-constrained Node participants.
    #[serde(default)]
    pub nodes: NodesConfig,
    /// The Client Gateway (`/client/v1`) for the owner's own apps. Off by
    /// default; loopback only.
    #[serde(default)]
    pub client_gateway: aidaemon_client_gateway::ClientGatewayConfig,
    /// Tunnels the daemon runs to publish the gateways (`[exposure.*]`).
    /// None: nothing is exposed. Read at startup.
    #[serde(default)]
    pub exposure: aidaemon_exposure::ExposureConfig,
    #[serde(default)]
    pub path_aliases: PathAliasConfig,
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub triggers: TriggersConfig,
    #[serde(default)]
    pub mcp: HashMap<String, McpServerConfig>,
    #[serde(default)]
    pub browser: BrowserConfig,
    #[serde(default)]
    pub computer_use: ComputerUseConfig,
    #[serde(default)]
    pub skills: SkillsConfig,
    #[serde(default)]
    pub subagents: SubagentsConfig,
    #[serde(default)]
    pub cli_agents: CliAgentsConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub files: FilesConfig,
    #[serde(default)]
    pub health: HealthConfig,
    #[serde(default)]
    pub updates: UpdateConfig,
    #[serde(default)]
    pub users: UsersConfig,
    #[serde(default)]
    pub people: PeopleConfig,
    #[serde(default)]
    pub oauth: OAuthConfig,
    #[serde(default)]
    pub http_auth: HashMap<String, HttpAuthProfile>,
    #[serde(default)]
    pub heartbeat: HeartbeatConfig,
    #[serde(default)]
    pub policy: PolicyConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub diagnostics: DiagnosticsConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub authority: AuthorityConfig,
    #[serde(default)]
    pub decisions: DecisionsConfig,
}

pub use aidaemon_runtime::settings::{DecisionMode, DecisionsConfig};

impl AppConfig {
    /// The API key of the first Gemini (`google_genai`) provider in the
    /// provider chain (primary, then fallbacks in order), so one Gemini key
    /// serves both model calls and the decision backend.
    pub fn gemini_provider_api_key(&self) -> Option<String> {
        fn find(provider: &ProviderConfig) -> Option<String> {
            if provider.kind == ProviderKind::GoogleGenai && !provider.api_key.trim().is_empty() {
                return Some(provider.api_key.trim().to_string());
            }
            provider.fallbacks.iter().find_map(find)
        }
        find(&self.provider)
    }
}

pub use aidaemon_runtime::authority::OwnerLevel;

/// `[authority]`: the explicit facts tool authority is resolved from.
#[derive(Debug, Deserialize, Clone, PartialEq, Eq, Default)]
pub struct AuthorityConfig {
    #[serde(default)]
    pub owner_default: OwnerLevel,
    /// Absolute directories whose files are the owner's own: reading them
    /// does not mark the run as having seen untrusted content. The turn's
    /// working directory and project scope are always trusted in addition.
    #[serde(default)]
    pub trusted_roots: Vec<String>,
}

/// Which long-term memory agent turns use.
#[derive(Debug, Deserialize, Clone)]
pub struct MemoryConfig {
    #[serde(default)]
    pub backend: MemoryBackend,
    /// SQLite file for the `sdk` backend.
    #[serde(default = "default_sdk_memory_path")]
    pub path: String,
    /// Let a replacing memory system learn from the retained conversations.
    /// It reads them once, in the background, and costs model calls.
    #[serde(default)]
    pub learn_from_history: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            backend: MemoryBackend::default(),
            path: default_sdk_memory_path(),
            learn_from_history: false,
        }
    }
}

fn default_sdk_memory_path() -> String {
    "aidaemon-memory.db".to_string()
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBackend {
    /// The daemon's own memory in its state database.
    #[default]
    Daemon,
    /// The `aidaemon-memory` crate (needs the `sdk-memory` cargo feature).
    Sdk,
}

pub use aidaemon_nodes::config::NodesConfig;

#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    /// User-facing assistant name used in the root identity prompt.
    #[serde(default = "default_agent_name")]
    pub name: String,
    /// Optional Markdown persona file. Relative paths resolve next to config.toml.
    #[serde(default)]
    pub persona_file: Option<PathBuf>,
    /// `[agent.engine]` settings for the SDK kernel that runs agent turns.
    #[serde(default)]
    pub engine: AgentEngineConfig,
}

/// `[agent.engine]`: configures the agent kernel.
///
/// ```toml
/// [agent.engine]
/// model = "gpt-6-sol"   # optional; pins every turn to a model the daemon's provider serves
/// ```
#[derive(Debug, Deserialize, Clone, Default)]
pub struct AgentEngineConfig {
    /// Ignored: the SDK kernel is the only agent loop. Kept so older configs
    /// with `kind = "legacy"` or `kind = "sdk"` still parse.
    #[serde(default)]
    pub kind: Option<String>,
    /// Pins every SDK turn to this model, served by the daemon's providers.
    #[serde(default)]
    pub model: Option<String>,
    /// Ignored: the SDK engine uses the daemon's configured providers. Kept so
    /// older configs still parse; when set, a `model` pin written for the
    /// removed Anthropic engine client is ignored unless the provider is
    /// Anthropic.
    #[serde(default)]
    pub api_key: String,
    /// Model turns per agent turn before the run stops.
    #[serde(default)]
    pub max_turns: Option<u32>,
}

impl AgentConfig {
    pub fn validated_name(&self) -> anyhow::Result<String> {
        let name = self.name.trim();
        if name.is_empty() {
            anyhow::bail!("agent.name cannot be empty");
        }
        if name.chars().count() > 40 {
            anyhow::bail!("agent.name must be 40 characters or fewer");
        }
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '\''))
        {
            anyhow::bail!(
                "agent.name may contain only letters, numbers, spaces, hyphens, underscores, and apostrophes"
            );
        }
        Ok(name.to_string())
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: default_agent_name(),
            persona_file: None,
            engine: AgentEngineConfig::default(),
        }
    }
}

fn default_agent_name() -> String {
    "aidaemon".to_string()
}

pub use aidaemon_tools::execution::ExecutionConfig;

pub use aidaemon_tools::checkpoints::CheckpointConfig;

#[derive(Deserialize, Clone)]
pub struct ProviderConfig {
    #[serde(default)]
    pub kind: ProviderKind,
    pub api_key: String,
    #[serde(default)]
    pub gateway_token: Option<String>,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub extra_headers: Option<HashMap<String, String>>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Reasoning effort level for thinking models. When set, includes the
    /// `reasoning` parameter in requests to enable thinking/reasoning tokens.
    /// Values: "low", "medium", "high"
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Enable llama.cpp chat-template thinking for models such as Gemma.
    /// Disabled by default so other OpenAI-compatible providers receive the
    /// same request shape used before this option was introduced.
    #[serde(default)]
    pub llama_cpp_thinking: bool,
    /// Opt-in SSE streaming transport (OpenAI-compatible providers).
    /// Responses are accumulated to the same shape as buffered calls; a
    /// stream that dies after partial text is recovered via the
    /// truncation-recovery path instead of failing the whole call.
    #[serde(default)]
    pub streaming: bool,
    /// Models whose final (no-tool-call) answers are re-issued through a
    /// tool-less, schema-constrained `{reasoning, response}` second pass
    /// before delivery. For models (e.g. local Gemma via llama.cpp) that emit
    /// raw undelimited chain-of-thought as their answer body under
    /// challenge/truncation-recovery turns. Matched by exact model name;
    /// empty (disabled) by default. OpenAI-compatible providers only.
    #[serde(default)]
    pub structured_answer_models: Vec<String>,
    #[serde(default)]
    pub models: ModelsConfig,
    /// Ordered cross-provider fallback chain.
    /// TOML key: `[[provider.fallbacks]]`
    #[serde(default, alias = "failover")]
    pub fallbacks: Vec<ProviderConfig>,
    /// Opt-in llama.cpp slot routing (`id_slot` pinning). Disabled by default.
    #[serde(default)]
    pub slot_routing: SlotRoutingConfig,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("api_key", &redact(&self.api_key))
            .field("gateway_token", &redact_option(&self.gateway_token))
            .field("base_url", &self.base_url)
            .field(
                "extra_headers",
                &self
                    .extra_headers
                    .as_ref()
                    .map(|h| h.keys().map(String::as_str).collect::<Vec<_>>()),
            )
            .field("max_tokens", &self.max_tokens)
            .field("models", &self.models)
            .field("fallback_provider_count", &self.fallbacks.len())
            .field("slot_routing", &self.slot_routing)
            .finish()
    }
}

impl ProviderConfig {
    pub fn apply_model_defaults_recursive(&mut self) {
        self.models.apply_defaults(&self.kind);
        for fallback in &mut self.fallbacks {
            fallback.apply_model_defaults_recursive();
        }
    }

    fn resolve_secrets_recursive(&mut self, key_prefix: Option<&str>) -> anyhow::Result<()> {
        use aidaemon_secrets::names;
        if self.api_key == "keychain" {
            self.api_key = resolve_from_keychain(&names::provider_field(key_prefix, "api_key"))?;
        }

        if self.gateway_token.as_deref() == Some("keychain") {
            self.gateway_token = Some(resolve_from_keychain(&names::provider_field(
                key_prefix,
                "gateway_token",
            ))?);
        }

        if let Some(headers) = self.extra_headers.as_mut() {
            for (name, value) in headers.iter_mut() {
                if value == "keychain" {
                    *value = resolve_from_keychain(&names::provider_header(key_prefix, name))?;
                }
            }
        }

        for (idx, fallback) in self.fallbacks.iter_mut().enumerate() {
            let nested_prefix = names::provider_fallback_prefix(key_prefix, idx);
            fallback.resolve_secrets_recursive(Some(&nested_prefix))?;
        }

        Ok(())
    }
}

pub use aidaemon_providers::routing::{ModelsConfig, ProviderKind};

fn default_true() -> bool {
    true
}

fn default_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

fn default_background_slot() -> u32 {
    1
}

/// Opt-in llama.cpp slot routing (`id_slot` pinning).
///
/// When a local llama.cpp server runs with `--parallel N` (N KV cache slots),
/// llama assigns each request to the best-prefix-match-or-LRU slot by default.
/// Constant background tasks (memory consolidation, summarization, extraction)
/// then evict the idle interactive conversation's slot, forcing a full
/// re-evaluation on the user's next turn.
///
/// When `enabled`, the interactive generation loop is pinned to
/// `interactive_slot` and every other call defaults to `background_slot`,
/// keeping the conversation's KV cache warm. The `id_slot` request field is
/// ONLY emitted when `enabled == true` — cloud APIs reject unknown params, so
/// the default (disabled) is safe for all providers.
#[derive(Debug, Deserialize, Clone)]
pub struct SlotRoutingConfig {
    /// Master switch. Default `false` (no `id_slot` ever sent).
    #[serde(default)]
    pub enabled: bool,
    /// Slot pinned for the interactive conversation. Default `0`.
    #[serde(default)]
    pub interactive_slot: u32,
    /// Slot used for all background/non-interactive calls. Default `1`.
    #[serde(default = "default_background_slot")]
    pub background_slot: u32,
}

impl Default for SlotRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interactive_slot: 0,
            background_slot: default_background_slot(),
        }
    }
}

/// Single Telegram bot configuration (legacy format).
#[derive(Deserialize, Clone)]
pub struct TelegramConfig {
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
    #[serde(default)]
    pub webhook: TelegramWebhookConfig,
}

impl fmt::Debug for TelegramConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TelegramConfig")
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
            .field("webhook", &self.webhook)
            .finish()
    }
}

/// Global defaults for Telegram webhook mode. When `enabled = true`, any
/// Telegram bot (config-based or dynamic) that does NOT have an explicit
/// `[webhook]` section with `enabled = true` will auto-derive webhook
/// settings from these defaults at startup.
#[derive(Debug, Deserialize, Clone)]
pub struct TelegramWebhookDefaults {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub base_domain: Option<String>,
    #[serde(default = "default_webhook_port_start")]
    pub port_start: u16,
    #[serde(default = "default_webhook_max_connections")]
    pub max_connections: u8,
    #[serde(default)]
    pub drop_pending_updates: bool,
    #[serde(default = "default_true")]
    pub bind_local_only: bool,
}

fn default_webhook_port_start() -> u16 {
    8443
}

fn default_webhook_max_connections() -> u8 {
    40
}

impl Default for TelegramWebhookDefaults {
    fn default() -> Self {
        Self {
            enabled: false,
            base_domain: None,
            port_start: 8443,
            max_connections: 40,
            drop_pending_updates: false,
            bind_local_only: true,
        }
    }
}

impl TelegramWebhookDefaults {
    /// Derive a full `TelegramWebhookConfig` for a bot given its slug and port.
    pub fn derive_webhook_config(&self, slug: &str, port: u16) -> TelegramWebhookConfig {
        let listen_host = if self.bind_local_only {
            "127.0.0.1"
        } else {
            "0.0.0.0"
        };
        let path = format!("/telegram/{}", slug);
        let public_url = self
            .base_domain
            .as_ref()
            .map(|domain| format!("https://{}.{}{}", slug, domain, path));
        TelegramWebhookConfig {
            enabled: true,
            public_url,
            listen_addr: Some(format!("{}:{}", listen_host, port)),
            path: Some(path),
            max_connections: Some(self.max_connections),
            drop_pending_updates: self.drop_pending_updates,
        }
    }

    /// Find the next available port starting from `port_start`, skipping occupied ones.
    pub fn next_available_port(&self, occupied: &mut std::collections::HashSet<u16>) -> u16 {
        let mut port = self.port_start;
        while occupied.contains(&port) {
            port = port.saturating_add(1);
        }
        occupied.insert(port);
        port
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TelegramWebhookConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub public_url: Option<String>,
    #[serde(default)]
    pub listen_addr: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub max_connections: Option<u8>,
    #[serde(default)]
    pub drop_pending_updates: bool,
}

/// Telegram bot configuration for multi-bot support.
/// Bot username is auto-detected from Telegram API at startup.
#[derive(Deserialize, Clone)]
pub struct TelegramBotConfig {
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
    #[serde(default)]
    pub webhook: TelegramWebhookConfig,
}

impl fmt::Debug for TelegramBotConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TelegramBotConfig")
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
            .field("webhook", &self.webhook)
            .finish()
    }
}

#[cfg(feature = "discord")]
#[derive(Deserialize, Clone, Default)]
pub struct DiscordConfig {
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
    #[serde(default)]
    pub guild_id: Option<u64>,
}

#[cfg(feature = "discord")]
impl fmt::Debug for DiscordConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscordConfig")
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
            .field("guild_id", &self.guild_id)
            .finish()
    }
}

/// Discord bot configuration for multi-bot support.
/// Bot username is auto-detected from Discord API at startup.
#[cfg(feature = "discord")]
#[derive(Deserialize, Clone)]
pub struct DiscordBotConfig {
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
    #[serde(default)]
    pub guild_id: Option<u64>,
}

#[cfg(feature = "discord")]
impl fmt::Debug for DiscordBotConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DiscordBotConfig")
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
            .field("guild_id", &self.guild_id)
            .finish()
    }
}

#[cfg(feature = "slack")]
#[derive(Deserialize, Clone, Default)]
pub struct SlackConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub app_token: String,
    #[serde(default)]
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
    #[serde(default = "default_slack_use_threads")]
    pub use_threads: bool,
}

#[cfg(feature = "slack")]
impl fmt::Debug for SlackConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlackConfig")
            .field("enabled", &self.enabled)
            .field("app_token", &redact(&self.app_token))
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
            .field("use_threads", &self.use_threads)
            .finish()
    }
}

#[cfg(feature = "slack")]
fn default_slack_use_threads() -> bool {
    true
}

/// Slack bot configuration for multi-bot support.
/// Bot name is auto-detected from Slack API at startup.
#[cfg(feature = "slack")]
#[derive(Deserialize, Clone)]
pub struct SlackBotConfig {
    pub app_token: String,
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<String>,
    #[serde(default = "default_slack_use_threads")]
    pub use_threads: bool,
}

#[cfg(feature = "slack")]
impl fmt::Debug for SlackBotConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlackBotConfig")
            .field("app_token", &redact(&self.app_token))
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
            .field("use_threads", &self.use_threads)
            .finish()
    }
}

#[derive(Deserialize, Clone)]
pub struct StateConfig {
    #[serde(default = "default_db_path")]
    pub db_path: String,
    #[serde(default = "default_working_memory_cap")]
    pub working_memory_cap: usize,
    #[serde(default = "default_consolidation_interval_hours")]
    pub consolidation_interval_hours: u64,
    /// Encryption key for SQLCipher database encryption.
    /// Requires the `encryption` cargo feature to be enabled.
    /// If set, all database contents are AES-256 encrypted at rest.
    #[serde(default)]
    pub encryption_key: Option<String>,
    /// Maximum number of facts to inject into the system prompt.
    /// Facts are ordered by most recently updated first; older facts
    /// remain in the database but are not included in the prompt.
    #[serde(default = "default_max_facts")]
    pub max_facts: usize,
    /// Ignored: a global daily token cap is not enforced (logged once at
    /// startup). Kept so older configs still parse. Per-goal daily budgets and
    /// the per-task token budget still apply; cap total spend at the provider.
    #[serde(default)]
    pub daily_token_budget: Option<u64>,
    /// Retention policies for automatic cleanup of old data.
    #[serde(default)]
    pub retention: RetentionConfig,
    /// Snapshots of the database kept from before an upgrade: when a new
    /// aidaemon version starts on an existing database, it first writes a
    /// consistent copy (`VACUUM INTO`, encrypted like the database) to
    /// `backups/` next to it and keeps this many of them, newest first.
    /// `0` turns it off.
    #[serde(default = "default_upgrade_backups")]
    pub upgrade_backups: usize,
    /// Context window management (trimming, summarization, progressive facts).
    #[serde(default)]
    pub context_window: ContextWindowConfig,
}

/// Retention policies for automatic cleanup of old data.
/// All fields are in days. Set to 0 to disable cleanup for that table.
#[derive(Debug, Deserialize, Clone)]
pub struct RetentionConfig {
    /// Delete consolidated messages older than N days (default: 90)
    #[serde(default = "default_retention_90")]
    pub messages_days: u32,
    /// Delete consolidated diagnostic/tool events older than N days (default: 7).
    /// This is intentionally independent of exact conversation retention.
    #[serde(default = "default_retention_7")]
    pub diagnostic_events_days: u32,
    /// Delete superseded fact versions older than N days (default: 90)
    #[serde(default = "default_retention_90")]
    pub superseded_facts_days: u32,
    /// Aggregate raw token_usage into daily summaries after N days (default: 30)
    #[serde(default = "default_retention_30")]
    pub token_usage_aggregate_days: u32,
    /// Delete episodes with recall_count=0 older than N days (default: 365)
    #[serde(default = "default_retention_365")]
    pub episodes_days: u32,
    /// Delete behavior patterns at confidence floor older than N days (default: 90)
    #[serde(default = "default_retention_90")]
    pub behavior_patterns_days: u32,
    /// Delete completed/abandoned goals older than N days (default: 180)
    #[serde(default = "default_retention_180")]
    pub goals_days: u32,
    /// Delete zero-success procedures older than N days (default: 180)
    #[serde(default = "default_retention_180")]
    pub procedures_days: u32,
    /// Delete net-negative error solutions older than N days (default: 90)
    #[serde(default = "default_retention_90")]
    pub error_solutions_days: u32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        let defaults = aidaemon_runtime::state::retention::RetentionPolicy::default();
        Self {
            messages_days: defaults.messages_days,
            diagnostic_events_days: defaults.diagnostic_events_days,
            superseded_facts_days: defaults.superseded_facts_days,
            token_usage_aggregate_days: defaults.token_usage_aggregate_days,
            episodes_days: defaults.episodes_days,
            behavior_patterns_days: defaults.behavior_patterns_days,
            goals_days: defaults.goals_days,
            procedures_days: defaults.procedures_days,
            error_solutions_days: defaults.error_solutions_days,
        }
    }
}

impl RetentionConfig {
    /// The retention policy these settings give.
    pub fn policy(&self) -> aidaemon_runtime::state::retention::RetentionPolicy {
        aidaemon_runtime::state::retention::RetentionPolicy {
            messages_days: self.messages_days,
            diagnostic_events_days: self.diagnostic_events_days,
            superseded_facts_days: self.superseded_facts_days,
            token_usage_aggregate_days: self.token_usage_aggregate_days,
            episodes_days: self.episodes_days,
            behavior_patterns_days: self.behavior_patterns_days,
            goals_days: self.goals_days,
            procedures_days: self.procedures_days,
            error_solutions_days: self.error_solutions_days,
        }
    }
}

fn default_retention_7() -> u32 {
    7
}

fn default_retention_30() -> u32 {
    30
}
fn default_retention_90() -> u32 {
    90
}
fn default_retention_180() -> u32 {
    180
}
fn default_retention_365() -> u32 {
    365
}

impl fmt::Debug for StateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateConfig")
            .field("db_path", &self.db_path)
            .field("working_memory_cap", &self.working_memory_cap)
            .field(
                "consolidation_interval_hours",
                &self.consolidation_interval_hours,
            )
            .field("encryption_key", &redact_option(&self.encryption_key))
            .field("max_facts", &self.max_facts)
            .field("daily_token_budget", &self.daily_token_budget)
            .field("retention", &self.retention)
            .field("upgrade_backups", &self.upgrade_backups)
            .field("context_window", &self.context_window)
            .finish()
    }
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            db_path: default_db_path(),
            working_memory_cap: default_working_memory_cap(),
            consolidation_interval_hours: default_consolidation_interval_hours(),
            encryption_key: None,
            max_facts: default_max_facts(),
            daily_token_budget: None,
            retention: RetentionConfig::default(),
            upgrade_backups: default_upgrade_backups(),
            context_window: ContextWindowConfig::default(),
        }
    }
}

fn default_upgrade_backups() -> usize {
    2
}

fn default_db_path() -> String {
    "aidaemon.db".to_string()
}
fn default_working_memory_cap() -> usize {
    50
}
fn default_consolidation_interval_hours() -> u64 {
    6
}
fn default_max_facts() -> usize {
    20
}

/// Permission mode re-exported for config deserialization.
pub use crate::tools::command_risk::PermissionMode;

#[derive(Deserialize, Clone)]
pub struct TerminalConfig {
    #[serde(default = "default_allowed_prefixes")]
    pub allowed_prefixes: Vec<String>,
    #[serde(default = "default_terminal_web_app_url")]
    pub web_app_url: String,
    #[serde(default = "default_terminal_bridge_enabled")]
    pub bridge_enabled: bool,
    #[serde(default = "default_terminal_bridge_ws_url")]
    pub daemon_ws_url: String,
    #[serde(default)]
    pub daemon_connect_token: Option<String>,
    #[serde(default)]
    pub allow_static_token_fallback: bool,
    #[serde(default)]
    pub daemon_user_id: Option<u64>,
    #[serde(default)]
    pub daemon_device_id: Option<String>,
    #[serde(default)]
    pub daemon_shell: Option<String>,
    #[serde(default = "default_initial_timeout_secs")]
    pub initial_timeout_secs: u64,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    /// Permission persistence mode:
    /// - "default": Critical commands per-session, others persist forever
    /// - "cautious": All approvals per-session only (resets on restart)
    /// - "yolo": All approvals persist forever, including critical commands
    #[serde(default)]
    pub permission_mode: PermissionMode,
    /// Where commands run: "host" (default) executes directly on this
    /// machine with the user's environment and credentials; "sandbox" wraps
    /// every command in the OS sandbox with task-scoped filesystem/network
    /// policy.
    #[serde(default)]
    pub confinement: crate::types::TerminalConfinement,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            allowed_prefixes: default_allowed_prefixes(),
            web_app_url: default_terminal_web_app_url(),
            bridge_enabled: default_terminal_bridge_enabled(),
            daemon_ws_url: default_terminal_bridge_ws_url(),
            daemon_connect_token: None,
            allow_static_token_fallback: false,
            daemon_user_id: None,
            daemon_device_id: None,
            daemon_shell: None,
            initial_timeout_secs: default_initial_timeout_secs(),
            max_output_chars: default_max_output_chars(),
            permission_mode: PermissionMode::default(),
            confinement: crate::types::TerminalConfinement::default(),
        }
    }
}

impl fmt::Debug for TerminalConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TerminalConfig")
            .field("allowed_prefixes", &self.allowed_prefixes)
            .field("web_app_url", &self.web_app_url)
            .field("bridge_enabled", &self.bridge_enabled)
            .field("daemon_ws_url", &self.daemon_ws_url)
            .field(
                "daemon_connect_token",
                &self.daemon_connect_token.as_ref().map(|v| {
                    if v.is_empty() {
                        "\"\""
                    } else {
                        "[REDACTED]"
                    }
                }),
            )
            .field(
                "allow_static_token_fallback",
                &self.allow_static_token_fallback,
            )
            .field("daemon_user_id", &self.daemon_user_id)
            .field("daemon_device_id", &self.daemon_device_id)
            .field("daemon_shell", &self.daemon_shell)
            .field("initial_timeout_secs", &self.initial_timeout_secs)
            .field("max_output_chars", &self.max_output_chars)
            .field("permission_mode", &self.permission_mode)
            .finish()
    }
}

impl TerminalConfig {
    /// Returns the effective Telegram terminal web app URL.
    /// Environment variable override is useful for local/staging validation.
    pub fn effective_web_app_url(&self) -> String {
        let from_env = std::env::var("AIDAEMON_TERMINAL_WEB_APP_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        from_env.unwrap_or_else(|| self.web_app_url.trim().to_string())
    }

    pub fn effective_bridge_enabled(&self) -> bool {
        let from_env = std::env::var("AIDAEMON_TERMINAL_BRIDGE_ENABLED")
            .ok()
            .and_then(|v| parse_bool_like(&v));
        from_env.unwrap_or(self.bridge_enabled)
    }

    pub fn effective_daemon_ws_url(&self) -> String {
        let from_env = std::env::var("AIDAEMON_TERMINAL_BROKER_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        from_env.unwrap_or_else(|| self.daemon_ws_url.trim().to_string())
    }

    pub fn effective_daemon_connect_token(&self) -> Option<String> {
        let from_env = std::env::var("AIDAEMON_TERMINAL_DAEMON_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        from_env.or_else(|| {
            self.daemon_connect_token
                .as_ref()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
    }

    pub fn effective_allow_static_token_fallback(&self) -> bool {
        let from_env = std::env::var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK")
            .ok()
            .and_then(|v| parse_bool_like(&v));
        from_env.unwrap_or(self.allow_static_token_fallback)
    }

    pub fn effective_daemon_user_id(&self) -> Option<u64> {
        let from_env = std::env::var("AIDAEMON_TERMINAL_USER_ID")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok());
        from_env.or(self.daemon_user_id)
    }

    pub fn effective_daemon_device_id(&self) -> Option<String> {
        let from_env = std::env::var("AIDAEMON_TERMINAL_DEVICE_ID")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        from_env.or_else(|| {
            self.daemon_device_id
                .as_ref()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
    }

    pub fn effective_daemon_shell(&self) -> Option<String> {
        let from_env = std::env::var("AIDAEMON_TERMINAL_SHELL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        from_env.or_else(|| {
            self.daemon_shell
                .as_ref()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
    }
}

pub use aidaemon_runtime::settings::PathAliasConfig;

fn default_initial_timeout_secs() -> u64 {
    30
}
fn default_max_output_chars() -> usize {
    4000
}
fn parse_bool_like(value: &str) -> Option<bool> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
fn default_terminal_web_app_url() -> String {
    "https://terminal.aidaemon.ai".to_string()
}
fn default_terminal_bridge_enabled() -> bool {
    true
}
fn default_terminal_bridge_ws_url() -> String {
    "wss://terminal.aidaemon.ai/v1/ws/daemon".to_string()
}

fn default_allowed_prefixes() -> Vec<String> {
    vec![
        "ls".into(),
        "cat".into(),
        "head".into(),
        "tail".into(),
        "echo".into(),
        "date".into(),
        "whoami".into(),
        "pwd".into(),
        "find".into(),
        "wc".into(),
        "grep".into(),
        "tree".into(),
        "file".into(),
        "stat".into(),
        "uname".into(),
        "df".into(),
        "du".into(),
        "ps".into(),
        "which".into(),
        "env".into(),
        "printenv".into(),
    ]
}

#[derive(Debug, Deserialize, Clone)]
pub struct WatchdogConfig {
    #[serde(default = "default_watchdog_enabled")]
    pub enabled: bool,
    /// Seconds of no heartbeat before declaring stuck (default: 300 = 5 min).
    #[serde(default = "default_watchdog_stale_secs")]
    pub stale_threshold_secs: u64,
    /// Per-LLM-call timeout in seconds (default: 300 = 5 min).
    #[serde(default = "default_llm_call_timeout_secs")]
    pub llm_call_timeout_secs: u64,
    /// Seconds of no task activity (tool or LLM call) before an orchestration
    /// task is considered hung. Distinct from `stale_threshold_secs`, which
    /// drives channel stale-watchdog behavior. Default: 300 = 5 min.
    #[serde(default = "default_task_inactivity_timeout_secs")]
    pub task_inactivity_timeout_secs: u64,
    /// Base seconds a notifier-active background terminal command may make no
    /// objective progress before the heartbeat marks it as a stall candidate.
    /// Progress includes CPU time, disk I/O, output growth, and process-tree
    /// change. A deterministic launch-time workload contract may extend this
    /// window, and the reaper normally requires a second conclusive sample.
    #[serde(default = "default_background_stall_secs")]
    pub background_stall_secs: u64,
    /// Hard maximum total runtime for generic notifier-active commands. Known
    /// long-silent workloads such as filesystem traversal, builds/tests, and
    /// network transfers treat this as a soft boundary while objective progress
    /// remains visible. Default: 1200 = 20 min.
    #[serde(default = "default_background_max_runtime_secs")]
    pub background_max_runtime_secs: u64,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            enabled: default_watchdog_enabled(),
            stale_threshold_secs: default_watchdog_stale_secs(),
            llm_call_timeout_secs: default_llm_call_timeout_secs(),
            task_inactivity_timeout_secs: default_task_inactivity_timeout_secs(),
            background_stall_secs: default_background_stall_secs(),
            background_max_runtime_secs: default_background_max_runtime_secs(),
        }
    }
}

fn default_watchdog_enabled() -> bool {
    true
}
fn default_watchdog_stale_secs() -> u64 {
    300
}
fn default_llm_call_timeout_secs() -> u64 {
    300
}
fn default_task_inactivity_timeout_secs() -> u64 {
    300
}
fn default_background_stall_secs() -> u64 {
    120
}
fn default_background_max_runtime_secs() -> u64 {
    1200
}

#[derive(Debug, Deserialize, Clone)]
pub struct DaemonConfig {
    #[serde(default = "default_health_port")]
    pub health_port: u16,
    /// IP address to bind the health server to (default: "127.0.0.1").
    /// Set to "0.0.0.0" to listen on all interfaces.
    #[serde(default = "default_health_bind")]
    pub health_bind: String,
    /// Enable the embedded web dashboard on the health server port (default: true).
    #[serde(default = "default_dashboard_enabled")]
    pub dashboard_enabled: bool,
    /// Watchdog configuration for detecting stuck agent tasks.
    #[serde(default)]
    pub watchdog: WatchdogConfig,
    /// Queue policy for approval/media/trigger pipelines.
    #[serde(default)]
    pub queue_policy: QueuePolicyConfig,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            health_port: default_health_port(),
            health_bind: default_health_bind(),
            dashboard_enabled: default_dashboard_enabled(),
            watchdog: WatchdogConfig::default(),
            queue_policy: QueuePolicyConfig::default(),
        }
    }
}

fn default_health_port() -> u16 {
    8080
}

fn default_health_bind() -> String {
    "127.0.0.1".to_string()
}

fn default_dashboard_enabled() -> bool {
    true
}

pub use aidaemon_channels::queue::QueuePolicyConfig;
#[cfg(test)]
use aidaemon_channels::queue::{QueueLanePoliciesConfig, QueueLanePolicyConfig};

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TriggersConfig {
    pub email: Option<EmailTriggerConfig>,
}

impl TriggersConfig {
    /// The trigger sources as `aidaemon_work::triggers` runs them.
    pub fn settings(&self) -> aidaemon_work::triggers::TriggerSettings {
        aidaemon_work::triggers::TriggerSettings {
            email: self
                .email
                .as_ref()
                .map(|email| aidaemon_work::triggers::EmailTriggerSettings {
                    host: email.host.clone(),
                    port: email.port,
                    username: email.username.clone(),
                    password: email.password.clone(),
                    folder: email.folder.clone(),
                }),
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct EmailTriggerConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    #[serde(default = "default_folder")]
    pub folder: String,
}

impl fmt::Debug for EmailTriggerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmailTriggerConfig")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("username", &self.username)
            .field("password", &redact(&self.password))
            .field("folder", &self.folder)
            .finish()
    }
}

fn default_folder() -> String {
    "INBOX".to_string()
}

pub use aidaemon_mcp::registry::McpServerConfig;

pub use aidaemon_tool_browser::SessionIsolation;

#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(not(feature = "browser"), allow(dead_code))]
pub struct BrowserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_headless")]
    pub headless: bool,
    #[serde(default = "default_screenshot_width")]
    pub screenshot_width: u32,
    #[serde(default = "default_screenshot_height")]
    pub screenshot_height: u32,
    /// Connect to an existing Chrome instance on this debugging port instead of launching a new one.
    /// Start Chrome with: --remote-debugging-port=9222
    /// This shares your existing login sessions, cookies, and tabs.
    pub remote_debugging_port: Option<u16>,
    /// Path to Chrome user data directory for persistent sessions/cookies.
    /// Defaults to "~/.aidaemon/chrome-profile" so logins survive restarts.
    #[serde(default = "default_browser_user_data_dir")]
    pub user_data_dir: Option<String>,
    /// Chrome profile directory name within user_data_dir (default: "Default").
    /// Other profiles are typically "Profile 1", "Profile 2", etc.
    pub profile: Option<String>,
    /// How browser sessions are isolated from one another.
    ///
    /// `None` (the default) means "auto": dedicated ephemeral browsing resolves
    /// to `browser_context` (per-session isolated cookies), while a shared
    /// persistent profile (`user_data_dir`) or an attached remote-debugging
    /// Chrome (`remote_debugging_port`) resolves to `page` (shared cookies).
    /// Set explicitly to fail fast on incompatible combinations rather than
    /// silently claiming isolation. See [`Self::resolve_session_isolation`].
    #[serde(default)]
    pub session_isolation: Option<SessionIsolation>,
    /// Maximum seconds to wait for a navigation/DOM-ready signal after `goto`
    /// (and for a click-triggered navigation to settle). Clamped to 1..=120 by
    /// [`Self::nav_timeout`]. Default: 30.
    #[serde(default = "default_nav_timeout_secs")]
    pub nav_timeout_secs: u64,
    /// Default maximum seconds the `wait` action polls for its element
    /// condition. Clamped to 1..=120 by [`Self::element_timeout`]. A per-call
    /// `timeout_secs` argument overrides this (also clamped). Default: 10.
    #[serde(default = "default_element_timeout_secs")]
    pub element_timeout_secs: u64,
    /// Overall per-action ceiling for the click nav-race (the short
    /// stable-DOM fallback for a non-navigating click is derived from this but
    /// kept small). Clamped to 1..=300 by [`Self::action_timeout`]. Default: 30.
    #[serde(default = "default_action_timeout_secs")]
    pub action_timeout_secs: u64,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            headless: default_headless(),
            screenshot_width: default_screenshot_width(),
            screenshot_height: default_screenshot_height(),
            remote_debugging_port: None,
            user_data_dir: default_browser_user_data_dir(),
            profile: None,
            session_isolation: None,
            nav_timeout_secs: default_nav_timeout_secs(),
            element_timeout_secs: default_element_timeout_secs(),
            action_timeout_secs: default_action_timeout_secs(),
        }
    }
}

pub use aidaemon_runtime::settings::ComputerUseConfig;

impl BrowserConfig {
    /// What the browser tool takes from `[browser]`.
    pub fn settings(&self) -> aidaemon_tool_browser::BrowserSettings {
        aidaemon_tool_browser::BrowserSettings {
            headless: self.headless,
            screenshot_width: self.screenshot_width,
            screenshot_height: self.screenshot_height,
            remote_debugging_port: self.remote_debugging_port,
            user_data_dir: self.user_data_dir.clone(),
            profile: self.profile.clone(),
            session_isolation: self.session_isolation,
            nav_timeout_secs: self.nav_timeout_secs,
            element_timeout_secs: self.element_timeout_secs,
            action_timeout_secs: self.action_timeout_secs,
        }
    }

    /// Resolved navigation/DOM-ready timeout (see [`aidaemon_tool_browser::BrowserSettings::nav_timeout`]).
    #[cfg_attr(not(feature = "browser"), allow(dead_code))]
    pub fn nav_timeout(&self) -> std::time::Duration {
        self.settings().nav_timeout()
    }

    /// Resolved default `wait`-action element-poll timeout.
    #[cfg_attr(not(feature = "browser"), allow(dead_code))]
    pub fn element_timeout(&self) -> std::time::Duration {
        self.settings().element_timeout()
    }

    /// Resolved overall per-action ceiling.
    #[cfg_attr(not(feature = "browser"), allow(dead_code))]
    pub fn action_timeout(&self) -> std::time::Duration {
        self.settings().action_timeout()
    }

    /// The effective session-isolation mode, refusing configurations that
    /// would falsely claim per-session cookie isolation.
    #[cfg_attr(not(feature = "browser"), allow(dead_code))]
    pub fn resolve_session_isolation(&self) -> Result<SessionIsolation, String> {
        self.settings().resolve_session_isolation()
    }
}

fn default_browser_user_data_dir() -> Option<String> {
    dirs::home_dir().map(|h| {
        h.join(".aidaemon")
            .join("chrome-profile")
            .to_string_lossy()
            .into_owned()
    })
}

fn default_headless() -> bool {
    true
}
fn default_screenshot_width() -> u32 {
    1280
}
fn default_screenshot_height() -> u32 {
    720
}
fn default_nav_timeout_secs() -> u64 {
    30
}
fn default_element_timeout_secs() -> u64 {
    10
}
fn default_action_timeout_secs() -> u64 {
    30
}

#[derive(Debug, Deserialize, Clone)]
pub struct SkillsConfig {
    #[serde(default = "default_skills_dir")]
    pub dir: String,
    #[serde(default = "default_skills_enabled")]
    pub enabled: bool,
    /// URLs of skill registries (JSON manifests) to browse and install from.
    #[serde(default)]
    pub registries: Vec<String>,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            dir: default_skills_dir(),
            enabled: default_skills_enabled(),
            registries: Vec::new(),
        }
    }
}

fn default_skills_dir() -> String {
    "skills".to_string()
}

fn default_skills_enabled() -> bool {
    true
}

pub use aidaemon_runtime::settings::IterationLimitConfig;

#[derive(Debug, Deserialize, Clone)]
pub struct SubagentsConfig {
    #[serde(default = "default_subagents_enabled")]
    pub enabled: bool,
    #[serde(default = "default_subagents_max_depth")]
    pub max_depth: usize,
    /// Legacy field - use iteration_limit instead. Kept for backward compatibility.
    #[serde(default = "default_subagents_max_iterations")]
    pub max_iterations: usize,
    /// Legacy field - use iteration_limit instead. Kept for backward compatibility.
    #[serde(default = "default_subagents_max_iterations_cap")]
    pub max_iterations_cap: usize,
    #[serde(default = "default_subagents_max_response_chars")]
    pub max_response_chars: usize,
    /// Seconds a sub-agent may go without progress (no tool call running,
    /// no model reply or other event of its run) before it is stopped.
    /// Never a limit on how long it works: a run is bounded by progress and
    /// cost, so a long install that keeps making calls runs to its end.
    #[serde(default = "default_subagents_timeout_secs")]
    pub timeout_secs: u64,
    /// Iteration limit configuration. Defaults to Unlimited.
    /// If not set but max_iterations/max_iterations_cap are set, auto-migrates to Hard mode.
    #[serde(default)]
    pub iteration_limit: IterationLimitConfig,
    /// Accepted and ignored: a run is bounded by progress and cost (the
    /// inactivity watchdog, per-call timeouts, limits and token budgets), not
    /// by elapsed time, so legitimate long work is never cut off. Setting it
    /// logs one warning at startup.
    #[serde(default)]
    pub task_timeout_secs: Option<u64>,
    /// Token budget per task. When exceeded, productive runs may auto-extend
    /// temporarily; otherwise the agent gracefully summarizes and stops.
    /// Defaults to 500,000. Set to 0 to disable (unlimited).
    #[serde(default = "default_task_token_budget")]
    pub task_token_budget: Option<u64>,
    /// Plain sub-agents one request may run at the same time, across its
    /// whole tree. Task-lead executors have their own per-task limits.
    #[serde(default = "default_subagents_max_concurrent")]
    pub max_concurrent: usize,
    /// Plain sub-agents one request may start in total, across its whole tree.
    #[serde(default = "default_subagents_max_per_request")]
    pub max_per_request: usize,
    /// Track what the agents of one request read and change, refuse writes
    /// made from a stale view, and let children ask the top-level agent.
    #[serde(default = "default_subagents_coordination")]
    pub coordination: bool,
    /// Questions one child may put to the top-level agent itself.
    #[serde(default = "default_subagents_max_escalations_per_child")]
    pub max_escalations_per_child: usize,
    /// Escalations of any kind one request may make.
    #[serde(default = "default_subagents_max_escalations_per_request")]
    pub max_escalations_per_request: usize,
    /// Optional directory of user-override specialist `.md` files.
    /// Default: `~/.aidaemon/specialists/` (resolved at construction time if absent).
    #[serde(default)]
    pub specialists_override_dir: Option<PathBuf>,
}

impl SubagentsConfig {
    /// Returns the effective iteration limit config, handling backward compatibility.
    /// If iteration_limit is Unlimited but legacy max_iterations fields are set to
    /// non-default values, automatically migrates to Hard mode.
    pub fn effective_iteration_limit(&self) -> IterationLimitConfig {
        // Check if using non-default legacy values
        let legacy_initial_changed = self.max_iterations != default_subagents_max_iterations();
        let legacy_cap_changed = self.max_iterations_cap != default_subagents_max_iterations_cap();

        match &self.iteration_limit {
            IterationLimitConfig::Unlimited if legacy_initial_changed || legacy_cap_changed => {
                // Auto-migrate legacy config to Hard mode
                IterationLimitConfig::Hard {
                    initial: self.max_iterations,
                    cap: self.max_iterations_cap,
                }
            }
            other => other.clone(),
        }
    }
}

impl Default for SubagentsConfig {
    fn default() -> Self {
        Self {
            enabled: default_subagents_enabled(),
            max_depth: default_subagents_max_depth(),
            max_iterations: default_subagents_max_iterations(),
            max_iterations_cap: default_subagents_max_iterations_cap(),
            max_response_chars: default_subagents_max_response_chars(),
            timeout_secs: default_subagents_timeout_secs(),
            iteration_limit: IterationLimitConfig::default(),
            task_timeout_secs: None,
            task_token_budget: default_task_token_budget(),
            max_concurrent: default_subagents_max_concurrent(),
            max_per_request: default_subagents_max_per_request(),
            coordination: default_subagents_coordination(),
            max_escalations_per_child: default_subagents_max_escalations_per_child(),
            max_escalations_per_request: default_subagents_max_escalations_per_request(),
            specialists_override_dir: None,
        }
    }
}

use aidaemon_runtime::authority::{
    DEFAULT_SUBAGENTS_MAX_CONCURRENT, DEFAULT_SUBAGENTS_MAX_ESCALATIONS_PER_CHILD,
    DEFAULT_SUBAGENTS_MAX_ESCALATIONS_PER_REQUEST, DEFAULT_SUBAGENTS_MAX_PER_REQUEST,
};

fn default_subagents_max_concurrent() -> usize {
    DEFAULT_SUBAGENTS_MAX_CONCURRENT
}
fn default_subagents_max_per_request() -> usize {
    DEFAULT_SUBAGENTS_MAX_PER_REQUEST
}
fn default_subagents_coordination() -> bool {
    true
}
fn default_subagents_max_escalations_per_child() -> usize {
    DEFAULT_SUBAGENTS_MAX_ESCALATIONS_PER_CHILD
}
fn default_subagents_max_escalations_per_request() -> usize {
    DEFAULT_SUBAGENTS_MAX_ESCALATIONS_PER_REQUEST
}
fn default_subagents_enabled() -> bool {
    true
}
fn default_subagents_max_depth() -> usize {
    3
}
fn default_subagents_max_iterations() -> usize {
    10
}
fn default_subagents_max_iterations_cap() -> usize {
    25
}
fn default_subagents_max_response_chars() -> usize {
    8000
}
fn default_subagents_timeout_secs() -> u64 {
    300
}
fn default_task_token_budget() -> Option<u64> {
    Some(500_000)
}

pub use aidaemon_runtime::tools::cli_agent::CliAgentsConfig;

#[allow(unused_imports)]
pub use aidaemon_tools::web_search::{SearchBackendKind, SearchConfig};

pub use aidaemon_work::health::HealthConfig;

#[derive(Debug, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UpdateMode {
    /// Automatically download and apply updates, then restart.
    Enable,
    /// Check for updates and notify the user; wait for confirmation before applying.
    #[default]
    CheckOnly,
    /// Do not check for updates at all.
    Disable,
}

#[derive(Debug, Deserialize, Clone)]
pub struct UpdateConfig {
    /// Update behavior mode: "enable", "check_only", or "disable".
    #[serde(default)]
    pub mode: UpdateMode,
    /// How often to check for updates, in hours. Default: 24.
    #[serde(default = "default_update_check_interval_hours")]
    pub check_interval_hours: u64,
    /// Specific UTC hour to check (0-23). If set, overrides interval-based checks.
    #[serde(default)]
    pub check_at_utc_hour: Option<u8>,
    /// How long to wait for user confirmation in check_only mode, in minutes. Default: 60.
    #[serde(default = "default_update_confirmation_timeout_mins")]
    pub confirmation_timeout_mins: u64,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            mode: UpdateMode::default(),
            check_interval_hours: default_update_check_interval_hours(),
            check_at_utc_hour: None,
            confirmation_timeout_mins: default_update_confirmation_timeout_mins(),
        }
    }
}

fn default_update_check_interval_hours() -> u64 {
    24
}

fn default_update_confirmation_timeout_mins() -> u64 {
    60
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct UsersConfig {
    #[serde(default)]
    pub owner_ids: HashMap<String, Vec<String>>,
    /// Explicit, expiring project grants created by an owner command.
    #[serde(default)]
    pub workspace_grants: Vec<WorkspaceGrant>,
}

pub use aidaemon_runtime::settings::{AudioConfig, FilesConfig, SttConfig, VisionConfig};

#[derive(Debug, Deserialize, Clone, Default)]
pub struct OAuthConfig {
    // Retained for backward-compatible config parsing; OAuth tooling is always available at runtime.
    #[allow(dead_code)]
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub callback_url: Option<String>,
    #[serde(default)]
    pub providers: HashMap<String, OAuthProviderConfig>,
}

pub use aidaemon_tools::services::oauth::OAuthProviderConfig;

#[derive(Debug, Deserialize, Clone)]
pub struct PeopleConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_people_auto_extract")]
    pub auto_extract: bool,
    #[serde(default = "default_people_auto_extract_categories")]
    pub auto_extract_categories: Vec<String>,
    #[serde(default = "default_people_restricted_categories")]
    pub restricted_categories: Vec<String>,
    #[serde(default = "default_people_fact_retention_days")]
    pub fact_retention_days: u32,
    #[serde(default = "default_people_reconnect_reminder_days")]
    pub reconnect_reminder_days: u32,
}

impl PeopleConfig {
    /// The people upkeep's settings.
    pub fn intelligence(&self) -> aidaemon_memory::people_intelligence::PeopleIntelligenceConfig {
        aidaemon_memory::people_intelligence::PeopleIntelligenceConfig {
            fact_retention_days: self.fact_retention_days,
            reconnect_reminder_days: self.reconnect_reminder_days,
        }
    }
}

impl Default for PeopleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_extract: default_people_auto_extract(),
            auto_extract_categories: default_people_auto_extract_categories(),
            restricted_categories: default_people_restricted_categories(),
            fact_retention_days: default_people_fact_retention_days(),
            reconnect_reminder_days: default_people_reconnect_reminder_days(),
        }
    }
}

fn default_people_auto_extract() -> bool {
    true
}
fn default_people_auto_extract_categories() -> Vec<String> {
    vec![
        "birthday".into(),
        "preference".into(),
        "interest".into(),
        "work".into(),
        "family".into(),
        "important_date".into(),
    ]
}
fn default_people_restricted_categories() -> Vec<String> {
    vec![
        "health".into(),
        "finance".into(),
        "political".into(),
        "religious".into(),
    ]
}
fn default_people_fact_retention_days() -> u32 {
    180
}
fn default_people_reconnect_reminder_days() -> u32 {
    30
}

pub use aidaemon_tools::http_request::HttpAuthProfile;

pub use aidaemon_runtime::settings::ContextWindowConfig;

fn default_heartbeat_tick() -> u64 {
    30
}

fn default_max_concurrent() -> usize {
    3
}

fn default_diagnostics_max_events() -> usize {
    200
}

fn default_disabled_tools() -> Vec<String> {
    vec![
        "git_info".to_string(),
        "git_commit".to_string(),
        "policy_metrics".to_string(),
        "service_status".to_string(),
        "project_inspect".to_string(),
        "read_channel_history".to_string(),
        "tool_trace".to_string(),
    ]
}

/// Startup tool registration controls.
#[derive(Debug, Deserialize, Clone)]
pub struct ToolsConfig {
    /// Built-in tool machine names to omit at daemon startup (requires `/restart`).
    /// Off by default: `git_info`, `git_commit` (use `run_command`/`terminal`), plus
    /// `policy_metrics`, `service_status`, `project_inspect` (use
    /// `read_file`/`search_files`/`terminal`), `read_channel_history` (Slack opt-in),
    /// `tool_trace` (use `goal_trace` with `action: "tool_trace"` instead). Set `disabled = []`
    /// to register everything. Example extras: `["goal_trace"]` to disable forensics entirely.
    #[serde(default = "default_disabled_tools")]
    pub disabled: Vec<String>,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            disabled: default_disabled_tools(),
        }
    }
}

impl ToolsConfig {
    pub fn is_enabled(&self, tool_name: &str) -> bool {
        !self.disabled.iter().any(|name| name == tool_name)
    }
}

fn default_diagnostics_enabled() -> bool {
    false
}

#[derive(Debug, Deserialize, Clone)]
pub struct DiagnosticsConfig {
    #[serde(default = "default_diagnostics_enabled")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub record_decision_points: bool,
    #[serde(default = "default_diagnostics_max_events")]
    pub max_events: usize,
    #[serde(default)]
    pub include_raw_tool_args: bool,
    #[serde(default)]
    pub harness_eval: DiagnosticsHarnessEvalConfig,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)] // warn_* thresholds used in Phase C diagnose/db_probe
pub struct DiagnosticsHarnessEvalConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub persist_on_task_end: bool,
    #[serde(default)]
    pub emit_separate_event: bool,
    #[serde(default = "default_harness_weight_routing")]
    pub weight_routing: f32,
    #[serde(default = "default_harness_weight_progress")]
    pub weight_progress: f32,
    #[serde(default = "default_harness_weight_quality")]
    pub weight_quality: f32,
    #[serde(default = "default_harness_weight_cost")]
    pub weight_cost: f32,
    #[serde(default = "default_harness_cost_tier_cheap")]
    pub cost_tier_cheap: f32,
    #[serde(default = "default_harness_cost_tier_balanced")]
    pub cost_tier_balanced: f32,
    #[serde(default = "default_harness_cost_tier_strong")]
    pub cost_tier_strong: f32,
    #[serde(default = "default_harness_cost_tier_unknown")]
    pub cost_tier_unknown: f32,
    #[serde(default = "default_harness_warn_overall")]
    pub warn_overall_below: f32,
    #[serde(default = "default_harness_warn_routing")]
    pub warn_routing_below: f32,
}

impl From<&DiagnosticsHarnessEvalConfig> for aidaemon_runtime::eval::HarnessEvalConfig {
    fn from(cfg: &DiagnosticsHarnessEvalConfig) -> Self {
        Self {
            enabled: cfg.enabled,
            weight_routing: cfg.weight_routing,
            weight_progress: cfg.weight_progress,
            weight_quality: cfg.weight_quality,
            weight_cost: cfg.weight_cost,
            cost_tier_cheap: cfg.cost_tier_cheap,
            cost_tier_balanced: cfg.cost_tier_balanced,
            cost_tier_strong: cfg.cost_tier_strong,
            cost_tier_unknown: cfg.cost_tier_unknown,
        }
    }
}

impl Default for DiagnosticsHarnessEvalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            persist_on_task_end: true,
            emit_separate_event: false,
            weight_routing: default_harness_weight_routing(),
            weight_progress: default_harness_weight_progress(),
            weight_quality: default_harness_weight_quality(),
            weight_cost: default_harness_weight_cost(),
            cost_tier_cheap: default_harness_cost_tier_cheap(),
            cost_tier_balanced: default_harness_cost_tier_balanced(),
            cost_tier_strong: default_harness_cost_tier_strong(),
            cost_tier_unknown: default_harness_cost_tier_unknown(),
            warn_overall_below: default_harness_warn_overall(),
            warn_routing_below: default_harness_warn_routing(),
        }
    }
}

fn default_harness_weight_routing() -> f32 {
    0.30
}
fn default_harness_weight_progress() -> f32 {
    0.25
}
fn default_harness_weight_quality() -> f32 {
    0.30
}
fn default_harness_weight_cost() -> f32 {
    0.15
}
fn default_harness_cost_tier_cheap() -> f32 {
    1.0
}
fn default_harness_cost_tier_balanced() -> f32 {
    2.5
}
fn default_harness_cost_tier_strong() -> f32 {
    5.0
}
fn default_harness_cost_tier_unknown() -> f32 {
    3.0
}
fn default_harness_warn_overall() -> f32 {
    0.6
}
fn default_harness_warn_routing() -> f32 {
    0.7
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: default_diagnostics_enabled(),
            record_decision_points: true,
            max_events: default_diagnostics_max_events(),
            include_raw_tool_args: false,
            harness_eval: DiagnosticsHarnessEvalConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct HeartbeatConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Tick interval in seconds (default: 30)
    #[serde(default = "default_heartbeat_tick")]
    pub tick_interval_secs: u64,
    /// Max concurrent LLM tasks spawned by heartbeat (default: 3)
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_llm_tasks: usize,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tick_interval_secs: 30,
            max_concurrent_llm_tasks: 3,
        }
    }
}

pub use aidaemon_runtime::settings::PolicyConfig;

impl AppConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let expanded = expand_env_vars(&content)?;
        let mut config: AppConfig = toml::from_str(&expanded)?;
        config.agent.validated_name()?;
        config.execution.normalized_backend()?;
        config.provider.apply_model_defaults_recursive();
        config.daemon.queue_policy = config.daemon.queue_policy.normalized();
        config.resolve_secrets()?;
        if config.client_gateway.enabled {
            config.client_gateway.bind_addr()?;
        }
        if config.nodes.gateway.enabled {
            let address: std::net::IpAddr = config
                .nodes
                .gateway
                .bind
                .parse()
                .map_err(|_| anyhow::anyhow!("nodes.gateway.bind must be an IP address"))?;
            anyhow::ensure!(
                address.is_loopback(),
                "nodes.gateway.bind must be loopback; publish it through a transport adapter"
            );
            anyhow::ensure!(
                config.nodes.gateway.port != 0,
                "nodes.gateway.port must not be zero"
            );
            anyhow::ensure!(
                config.nodes.gateway.advertised_endpoints.len() <= 3,
                "nodes.gateway.advertised_endpoints supports at most three origins"
            );
            let mut unique_endpoints = std::collections::HashSet::new();
            for endpoint in &config.nodes.gateway.advertised_endpoints {
                if let Err(error) = aidaemon_exposure::validate_https_origin(endpoint) {
                    anyhow::bail!("nodes.gateway.advertised_endpoints {error}");
                }
                anyhow::ensure!(
                    unique_endpoints.insert(endpoint.trim_end_matches('/')),
                    "nodes.gateway.advertised_endpoints must be unique"
                );
            }
        }
        config
            .exposure
            .validate(config.client_gateway.enabled, config.nodes.gateway.enabled)
            .map_err(|error| anyhow::anyhow!(error))?;
        if config.nodes.ota.enabled {
            anyhow::ensure!(
                config.nodes.gateway.enabled,
                "nodes.ota requires nodes.gateway.enabled"
            );
            anyhow::ensure!(
                !config.nodes.ota.manifest_path.trim().is_empty()
                    && !config.nodes.ota.image_path.trim().is_empty(),
                "nodes.ota requires manifest_path and image_path"
            );
            anyhow::ensure!(
                !config.nodes.ota.release_public_key_sec1.trim().is_empty(),
                "nodes.ota requires release_public_key_sec1"
            );
            anyhow::ensure!(
                (262_144..=2_621_440).contains(&config.nodes.ota.max_image_bytes),
                "nodes.ota.max_image_bytes must be between 262144 and the K10 OTA slot size 2621440"
            );
        }
        if config.nodes.announcements.enabled {
            anyhow::ensure!(
                config.nodes.gateway.enabled,
                "nodes.announcements requires nodes.gateway.enabled"
            );
            anyhow::ensure!(
                config.nodes.speech.enabled,
                "nodes.announcements requires nodes.speech.enabled"
            );
            anyhow::ensure!(
                (1..=500).contains(&config.nodes.announcements.max_text_chars),
                "nodes.announcements.max_text_chars must be between 1 and 500"
            );
            anyhow::ensure!(
                (1..=config.nodes.limits.response_audio_bytes)
                    .contains(&config.nodes.announcements.max_audio_bytes),
                "nodes.announcements.max_audio_bytes must not exceed nodes.limits.response_audio_bytes"
            );
            anyhow::ensure!(
                (30..=3_600).contains(&config.nodes.announcements.ttl_seconds),
                "nodes.announcements.ttl_seconds must be between 30 and 3600"
            );
            anyhow::ensure!(
                (2..=60).contains(&config.nodes.announcements.poll_interval_seconds),
                "nodes.announcements.poll_interval_seconds must be between 2 and 60"
            );
            anyhow::ensure!(
                config.nodes.announcements.ack_wait_seconds <= 60,
                "nodes.announcements.ack_wait_seconds must not exceed 60"
            );
            anyhow::ensure!(
                (1..=16).contains(&config.nodes.announcements.max_pending_per_node),
                "nodes.announcements.max_pending_per_node must be between 1 and 16"
            );
        }
        if config.nodes.monitoring.enabled {
            anyhow::ensure!(
                config.nodes.gateway.enabled,
                "nodes.monitoring requires nodes.gateway.enabled"
            );
            anyhow::ensure!(
                config.heartbeat.enabled,
                "nodes.monitoring requires heartbeat.enabled"
            );
            anyhow::ensure!(
                (1..=720).contains(&config.nodes.monitoring.history_hours),
                "nodes.monitoring.history_hours must be between 1 and 720"
            );
            anyhow::ensure!(
                (60..=100_000).contains(&config.nodes.monitoring.max_history_rows_per_capability),
                "nodes.monitoring.max_history_rows_per_capability must be between 60 and 100000"
            );
            anyhow::ensure!(
                (1..=365).contains(&config.nodes.monitoring.event_retention_days),
                "nodes.monitoring.event_retention_days must be between 1 and 365"
            );
            anyhow::ensure!(
                (100..=10_000).contains(&config.nodes.monitoring.max_events_per_monitor),
                "nodes.monitoring.max_events_per_monitor must be between 100 and 10000"
            );
            anyhow::ensure!(
                (5..=300).contains(&config.nodes.monitoring.scan_interval_seconds),
                "nodes.monitoring.scan_interval_seconds must be between 5 and 300"
            );
            anyhow::ensure!(
                (1..=256).contains(&config.nodes.monitoring.max_monitors),
                "nodes.monitoring.max_monitors must be between 1 and 256"
            );
            anyhow::ensure!(
                (1..=365).contains(&config.nodes.monitoring.max_duration_days),
                "nodes.monitoring.max_duration_days must be between 1 and 365"
            );
            anyhow::ensure!(
                (30..=86_400).contains(&config.nodes.monitoring.default_stale_after_seconds),
                "nodes.monitoring.default_stale_after_seconds must be between 30 and 86400"
            );
            anyhow::ensure!(
                (30..=86_400).contains(&config.nodes.monitoring.default_offline_after_seconds),
                "nodes.monitoring.default_offline_after_seconds must be between 30 and 86400"
            );
        }
        Ok(config)
    }

    /// The configured API keys, tokens and passwords (static config only;
    /// dynamically connected bots live in the state store). The kernel's
    /// outbound secret scan refuses calls whose arguments carry one.
    pub fn outbound_secret_values(&self) -> Vec<String> {
        let mut values = Vec::new();
        let mut providers = vec![&self.provider];
        providers.extend(self.provider.fallbacks.iter());
        for provider in providers {
            values.push(provider.api_key.clone());
            values.extend(provider.gateway_token.clone());
        }
        values.push(self.agent.engine.api_key.clone());
        values.push(self.search.api_key.clone());
        values.extend(
            self.all_telegram_bots()
                .into_iter()
                .map(|bot| bot.bot_token),
        );
        #[cfg(feature = "discord")]
        values.extend(self.all_discord_bots().into_iter().map(|bot| bot.bot_token));
        #[cfg(feature = "slack")]
        for bot in self.all_slack_bots() {
            values.push(bot.bot_token);
            values.push(bot.app_token);
        }
        if let Some(email) = &self.triggers.email {
            values.push(email.password.clone());
        }
        for profile in self.http_auth.values() {
            values.extend(profile.credential_values().into_iter().map(str::to_string));
        }
        values.retain(|value| !value.trim().is_empty() && value != "keychain");
        values
    }

    /// Get all Telegram bot configurations (merges legacy single + array).
    /// Legacy `[telegram]` becomes a bot named "default".
    pub fn all_telegram_bots(&self) -> Vec<TelegramBotConfig> {
        let mut bots = self.telegram_bots.clone();

        // Add legacy config as "default" if present and has a token
        if let Some(ref legacy) = self.telegram {
            if !legacy.bot_token.is_empty() {
                bots.push(TelegramBotConfig {
                    bot_token: legacy.bot_token.clone(),
                    allowed_user_ids: legacy.allowed_user_ids.clone(),
                    webhook: legacy.webhook.clone(),
                });
            }
        }

        bots
    }

    /// Get all Discord bot configurations (merges legacy single + array).
    #[cfg(feature = "discord")]
    pub fn all_discord_bots(&self) -> Vec<DiscordBotConfig> {
        let mut bots = self.discord_bots.clone();

        if let Some(ref legacy) = self.discord {
            if !legacy.bot_token.is_empty() {
                bots.push(DiscordBotConfig {
                    bot_token: legacy.bot_token.clone(),
                    allowed_user_ids: legacy.allowed_user_ids.clone(),
                    guild_id: legacy.guild_id,
                });
            }
        }

        bots
    }

    /// Get all Slack bot configurations (merges legacy single + array).
    #[cfg(feature = "slack")]
    pub fn all_slack_bots(&self) -> Vec<SlackBotConfig> {
        let mut bots = self.slack_bots.clone();

        if let Some(ref legacy) = self.slack {
            // Keep legacy `enabled` for backward compatibility in config parsing,
            // but activation is token-driven to match Telegram/Discord behavior.
            if !legacy.bot_token.is_empty() && !legacy.app_token.is_empty() {
                bots.push(SlackBotConfig {
                    app_token: legacy.app_token.clone(),
                    bot_token: legacy.bot_token.clone(),
                    allowed_user_ids: legacy.allowed_user_ids.clone(),
                    use_threads: legacy.use_threads,
                });
            }
        }

        bots
    }

    /// Resolve fields set to `"keychain"` by reading them from the OS credential store.
    fn resolve_secrets(&mut self) -> anyhow::Result<()> {
        self.provider.resolve_secrets_recursive(None)?;

        if self.agent.engine.api_key == "keychain" {
            self.agent.engine.api_key = resolve_from_keychain("agent_engine_api_key")?;
        }

        // Legacy telegram config
        if let Some(ref mut telegram) = self.telegram {
            if telegram.bot_token == "keychain" {
                telegram.bot_token = resolve_from_keychain("bot_token")?;
            }
        }

        // Telegram bots array
        for (i, bot) in self.telegram_bots.iter_mut().enumerate() {
            if bot.bot_token == "keychain" {
                let key = aidaemon_secrets::names::bot_token("bot_token", "telegram_bot_token", i);
                bot.bot_token = resolve_from_keychain(&key)?;
            }
        }

        if let Some(ref email) = self.triggers.email {
            if email.password == "keychain" {
                let password = resolve_from_keychain("email_password")?;
                let mut email = email.clone();
                email.password = password;
                self.triggers.email = Some(email);
            }
        }
        if let Some(ref key) = self.state.encryption_key {
            if key == "keychain" {
                self.state.encryption_key = Some(resolve_from_keychain("encryption_key")?);
            }
        }
        if self.search.api_key == "keychain" {
            self.search.api_key = resolve_from_keychain("search_api_key")?;
        }
        if self.decisions.api_key == "keychain" {
            // A missing key turns the optional decision layer off; it never
            // stops the daemon from starting.
            self.decisions.api_key =
                resolve_from_keychain("decisions_api_key").unwrap_or_else(|error| {
                    tracing::warn!(%error, "decisions api key not found; decision layer disabled");
                    String::new()
                });
        }
        if self.terminal.daemon_connect_token.as_deref() == Some("keychain") {
            self.terminal.daemon_connect_token =
                Some(resolve_from_keychain("terminal_daemon_connect_token")?);
        }

        #[cfg(feature = "discord")]
        {
            if let Some(ref mut discord) = self.discord {
                if discord.bot_token == "keychain" {
                    discord.bot_token = resolve_from_keychain("discord_bot_token")?;
                }
            }
            for (i, bot) in self.discord_bots.iter_mut().enumerate() {
                if bot.bot_token == "keychain" {
                    let key = aidaemon_secrets::names::bot_token(
                        "discord_bot_token",
                        "discord_bot_token",
                        i,
                    );
                    bot.bot_token = resolve_from_keychain(&key)?;
                }
            }
        }

        #[cfg(feature = "slack")]
        {
            if let Some(ref mut slack) = self.slack {
                if slack.app_token == "keychain" {
                    slack.app_token = resolve_from_keychain("slack_app_token")?;
                }
                if slack.bot_token == "keychain" {
                    slack.bot_token = resolve_from_keychain("slack_bot_token")?;
                }
            }
            for (i, bot) in self.slack_bots.iter_mut().enumerate() {
                if bot.app_token == "keychain" {
                    let key =
                        aidaemon_secrets::names::bot_token("slack_app_token", "slack_app_token", i);
                    bot.app_token = resolve_from_keychain(&key)?;
                }
                if bot.bot_token == "keychain" {
                    let key =
                        aidaemon_secrets::names::bot_token("slack_bot_token", "slack_bot_token", i);
                    bot.bot_token = resolve_from_keychain(&key)?;
                }
            }
        }

        // HTTP auth profiles
        for (name, profile) in self.http_auth.iter_mut() {
            let fields: &mut [(&str, &mut Option<String>)] = &mut [
                ("api_key", &mut profile.api_key),
                ("api_secret", &mut profile.api_secret),
                ("access_token", &mut profile.access_token),
                ("access_token_secret", &mut profile.access_token_secret),
                ("token", &mut profile.token),
                ("header_value", &mut profile.header_value),
                ("password", &mut profile.password),
            ];
            for (field, value) in fields.iter_mut() {
                if let Some(ref v) = value {
                    if v == "keychain" {
                        let keychain_key = aidaemon_secrets::names::http_auth(name, field);
                        **value = Some(resolve_from_keychain(&keychain_key)?);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::KeychainIsolation;
    use std::sync::LazyLock;

    /// A config written before a policy key was retired must still parse:
    /// `PolicyConfig` tolerates unknown keys rather than failing startup.
    #[test]
    fn policy_config_ignores_retired_keys() {
        let cfg: PolicyConfig = toml::from_str(
            r#"
policy_enforce = false
required_tool_choice_ignored_models = ["gemma-4-26b"]
"#,
        )
        .expect("retired policy key must not break config parsing");
        assert!(!cfg.policy_enforce);
    }

    #[test]
    fn watchdog_task_inactivity_timeout_defaults_to_300() {
        let cfg = WatchdogConfig::default();
        assert_eq!(cfg.task_inactivity_timeout_secs, 300);
    }

    #[test]
    fn context_budget_follows_config_then_known_window_then_default() {
        let mut config = ContextWindowConfig::default();
        assert_eq!(config.context_budget_for("gpt-6-sol"), 231_200);
        assert_eq!(config.context_budget_for("claude-opus-5-5"), 170_000);
        assert_eq!(config.context_budget_for("models/gemini-3-pro"), 850_000);
        assert_eq!(config.context_budget_for("Claude-Sonnet-5-5"), 170_000);
        assert_eq!(
            config.context_budget_for("my-local-model"),
            config.default_budget
        );
        config.model_budgets.insert("gpt-6-sol".to_string(), 50_000);
        assert_eq!(config.context_budget_for("gpt-6-sol"), 50_000);
    }

    /// Compaction follows each model's own window: a history that fits a
    /// large-window model is left alone, the same history is compacted for a
    /// small one, and a turn waits only near the window.
    #[test]
    fn compaction_thresholds_follow_each_models_window() {
        let mut config = ContextWindowConfig::default();
        assert_eq!(config.summarize_token_threshold_for("gpt-6-sol"), 138_720);
        assert_eq!(
            config.emergency_compaction_threshold_for("gpt-6-sol"),
            184_960
        );
        assert_eq!(config.summary_recent_tokens_for("gpt-6-sol"), 34_680);
        assert_eq!(
            config.summarize_token_threshold_for("claude-opus-5-5"),
            102_000
        );
        assert_eq!(
            config.emergency_compaction_threshold_for("claude-opus-5-5"),
            136_000
        );
        assert_eq!(
            config.summarize_token_threshold_for("models/gemini-3-pro"),
            510_000
        );
        config
            .model_budgets
            .insert("synthetic-local-32k".to_string(), 32_000);
        assert_eq!(
            config.summarize_token_threshold_for("synthetic-local-32k"),
            19_200
        );
        assert_eq!(
            config.emergency_compaction_threshold_for("synthetic-local-32k"),
            25_600
        );
        assert_eq!(
            config.summary_recent_tokens_for("synthetic-local-32k"),
            4_800
        );

        // An explicit cap lowers the background threshold, never the point at
        // which a turn waits.
        config.summarize_token_threshold = 12_000;
        config.summary_recent_tokens = 4_000;
        assert_eq!(config.summarize_token_threshold_for("gpt-6-sol"), 12_000);
        assert_eq!(config.summary_recent_tokens_for("gpt-6-sol"), 4_000);
        assert_eq!(
            config.emergency_compaction_threshold_for("gpt-6-sol"),
            184_960
        );
    }

    #[test]
    fn a_reported_overflow_lowers_the_budget_and_never_raises_it() {
        let config = ContextWindowConfig::default();
        let model = "gpt-6-synthetic-overflow";
        assert_eq!(config.context_budget_for(model), 231_200);
        // No stated window: three quarters of the rejected attempt.
        assert_eq!(
            aidaemon_runtime::settings::record_context_overflow(model, None, 200_000),
            150_000
        );
        assert_eq!(config.context_budget_for(model), 150_000);
        // A larger stated window cannot raise it back.
        assert_eq!(
            aidaemon_runtime::settings::record_context_overflow(model, Some(400_000), 0),
            150_000
        );
        assert_eq!(config.context_budget_for(model), 150_000);
    }

    #[test]
    fn tool_result_chars_for_unknown_model_uses_global_default() {
        let config = ContextWindowConfig::default();
        assert_eq!(
            config.tool_result_chars_for("some-model"),
            config.max_tool_result_chars
        );
    }

    #[test]
    fn tool_result_chars_for_explicit_override_wins() {
        let mut config = ContextWindowConfig::default();
        config.model_budgets.insert("gemma-3-4b".to_string(), 8_000);
        config
            .model_tool_result_chars
            .insert("gemma-3-4b".to_string(), 1_500);
        assert_eq!(config.tool_result_chars_for("gemma-3-4b"), 1_500);
    }

    #[test]
    fn tool_result_chars_for_scales_down_with_small_model_budget() {
        // Small local model: budget 8k tokens vs default 48k → tool results
        // shrink proportionally so a single read can't swamp the window.
        let mut config = ContextWindowConfig::default();
        config.model_budgets.insert("gemma-3-4b".to_string(), 8_000);
        let chars = config.tool_result_chars_for("gemma-3-4b");
        assert!(
            chars < config.max_tool_result_chars,
            "small budget should shrink the cap, got {chars}"
        );
        assert!(chars >= 1_000, "cap must not drop below usable floor");
    }

    #[test]
    fn tool_result_chars_for_scales_up_with_large_model_budget() {
        // Big cloud model: budget 192k tokens → tool results may grow so
        // powerful models see more of each file in one read.
        let mut config = ContextWindowConfig::default();
        config
            .model_budgets
            .insert("claude-fable-5".to_string(), 192_000);
        let chars = config.tool_result_chars_for("claude-fable-5");
        assert!(
            chars > config.max_tool_result_chars,
            "large budget should raise the cap, got {chars}"
        );
    }

    static ENV_EXPANSION_LOCK: LazyLock<std::sync::Mutex<()>> =
        LazyLock::new(|| std::sync::Mutex::new(()));
    static TERMINAL_WEB_APP_URL_ENV_LOCK: LazyLock<std::sync::Mutex<()>> =
        LazyLock::new(|| std::sync::Mutex::new(()));
    static TERMINAL_BRIDGE_ENABLED_ENV_LOCK: LazyLock<std::sync::Mutex<()>> =
        LazyLock::new(|| std::sync::Mutex::new(()));
    static TERMINAL_STATIC_FALLBACK_ENV_LOCK: LazyLock<std::sync::Mutex<()>> =
        LazyLock::new(|| std::sync::Mutex::new(()));

    #[test]
    fn cli_agents_config_defaults_to_disabled() {
        let config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test"
            "#,
        )
        .unwrap();
        assert!(!config.cli_agents.enabled);
    }

    #[test]
    fn retired_task_assessment_key_still_parses() {
        let config = "[provider]\napi_key = \"test\"\n\n[agent]\ntask_assessment = \"off\"\n";
        toml::from_str::<AppConfig>(config).expect("older configs keep parsing");
    }

    #[test]
    fn authority_owner_default_parses_and_defaults_to_full() {
        let config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test"
            "#,
        )
        .unwrap();
        assert_eq!(config.authority.owner_default, OwnerLevel::Full);
        let config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test"

            [authority]
            owner_default = "unrestricted"
            "#,
        )
        .unwrap();
        assert_eq!(config.authority.owner_default, OwnerLevel::Unrestricted);
    }

    #[test]
    fn health_config_defaults_to_disabled() {
        let config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test"
            "#,
        )
        .unwrap();
        assert!(!config.health.enabled);
    }

    #[test]
    fn diagnostics_config_defaults_to_disabled() {
        let config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test"
            "#,
        )
        .unwrap();
        assert!(!config.diagnostics.enabled);
        assert!(config.diagnostics.record_decision_points);
    }

    #[test]
    fn tools_config_defaults_disable_low_value_base_tools() {
        let config = ToolsConfig::default();
        for tool in [
            "git_info",
            "git_commit",
            "policy_metrics",
            "service_status",
            "project_inspect",
            "read_channel_history",
            "tool_trace",
        ] {
            assert!(
                !config.is_enabled(tool),
                "expected {tool} to be disabled by default"
            );
        }
        assert!(config.is_enabled("terminal"));
        assert!(config.is_enabled("check_environment"));
        assert!(config.is_enabled("goal_trace"));
        assert!(!config.is_enabled("tool_trace"));
        assert!(config.is_enabled("search_files"));
    }

    #[test]
    fn tools_disabled_list_parses_from_config() {
        let config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test"
            [tools]
            disabled = ["policy_metrics", "service_status"]
            "#,
        )
        .unwrap();
        assert_eq!(
            config.tools.disabled,
            vec!["policy_metrics".to_string(), "service_status".to_string()]
        );
        assert!(!config.tools.is_enabled("policy_metrics"));
        assert!(config.tools.is_enabled("terminal"));
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn expand_env_vars_replaces_set_variable() {
        let _guard = ENV_EXPANSION_LOCK.lock().unwrap();
        std::env::set_var("AIDAEMON_TEST_VAR", "hello");
        let result = expand_env_vars("key = \"${AIDAEMON_TEST_VAR}\"").unwrap();
        assert_eq!(result, "key = \"hello\"");
        std::env::remove_var("AIDAEMON_TEST_VAR");
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn expand_env_vars_errors_on_missing() {
        let _guard = ENV_EXPANSION_LOCK.lock().unwrap();
        std::env::remove_var("AIDAEMON_MISSING_VAR");
        let result = expand_env_vars("key = \"${AIDAEMON_MISSING_VAR}\"");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("AIDAEMON_MISSING_VAR"));
    }

    #[test]
    fn expand_env_vars_leaves_plain_strings() {
        let input = "key = \"sk-hardcoded-value\"";
        let result = expand_env_vars(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn expand_env_vars_handles_multiple() {
        let _guard = ENV_EXPANSION_LOCK.lock().unwrap();
        std::env::set_var("AIDAEMON_TEST_A", "aaa");
        std::env::set_var("AIDAEMON_TEST_B", "bbb");
        let result =
            expand_env_vars("a = \"${AIDAEMON_TEST_A}\"\nb = \"${AIDAEMON_TEST_B}\"").unwrap();
        assert_eq!(result, "a = \"aaa\"\nb = \"bbb\"");
        std::env::remove_var("AIDAEMON_TEST_A");
        std::env::remove_var("AIDAEMON_TEST_B");
    }

    #[test]
    fn expand_env_vars_ignores_bare_dollar() {
        let input = "path = \"$HOME/something\"";
        let result = expand_env_vars(input).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn store_in_keychain_writes_env_file_when_keychain_disabled() {
        let mut isolation = KeychainIsolation::acquire();
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(&env_path, "EXISTING=1\n").unwrap();
        isolation.unset(crate::RUNTIME_ENV_FILE_ENV_KEY);
        isolation.set("AIDAEMON_ENV_FILE", &env_path.to_string_lossy());

        store_in_keychain("oauth_twitter_access_token", "new token$1")
            .expect("write oauth token to env");

        let content = std::fs::read_to_string(&env_path).expect("read env file");
        assert!(content.contains("EXISTING=1"));
        assert!(content.contains("OAUTH_TWITTER_ACCESS_TOKEN=\"new token\\$1\""));
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn delete_from_keychain_removes_env_file_key_when_keychain_disabled() {
        let mut isolation = KeychainIsolation::acquire();
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(
            &env_path,
            "KEEP=1\nOAUTH_TWITTER_ACCESS_TOKEN=oldtoken\nANOTHER=2\n",
        )
        .unwrap();
        isolation.unset(crate::RUNTIME_ENV_FILE_ENV_KEY);
        isolation.set("AIDAEMON_ENV_FILE", &env_path.to_string_lossy());

        delete_from_keychain("oauth_twitter_access_token").expect("delete oauth token from env");

        let content = std::fs::read_to_string(&env_path).expect("read env file");
        assert!(content.contains("KEEP=1"));
        assert!(content.contains("ANOTHER=2"));
        assert!(!content.contains("OAUTH_TWITTER_ACCESS_TOKEN"));
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn resolve_from_keychain_reads_env_file_when_keychain_disabled() {
        let mut isolation = KeychainIsolation::acquire();
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(&env_path, "OAUTH_TWITTER_CLIENT_ID=client123\n").unwrap();
        isolation.unset(crate::RUNTIME_ENV_FILE_ENV_KEY);
        isolation.set("AIDAEMON_ENV_FILE", &env_path.to_string_lossy());
        isolation.unset("OAUTH_TWITTER_CLIENT_ID");

        let value = resolve_from_keychain("oauth_twitter_client_id").expect("resolve client id");
        assert_eq!(value, "client123");
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn resolve_from_keychain_tolerates_malformed_env_lines() {
        let mut isolation = KeychainIsolation::acquire();
        let tmp = tempfile::tempdir().unwrap();
        let env_path = tmp.path().join(".env.test");
        std::fs::write(
            &env_path,
            "BROKEN=\"unterminated\nOAUTH_TWITTER_CLIENT_ID=client123\n",
        )
        .unwrap();
        isolation.unset(crate::RUNTIME_ENV_FILE_ENV_KEY);
        isolation.set("AIDAEMON_ENV_FILE", &env_path.to_string_lossy());
        isolation.unset("OAUTH_TWITTER_CLIENT_ID");

        let value = resolve_from_keychain("oauth_twitter_client_id")
            .expect("resolve client id despite malformed env line");
        assert_eq!(value, "client123");
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn resolve_env_file_path_prefers_runtime_resolved_path() {
        // The runtime env-file key is owned by the keychain isolation guard;
        // override it through the guard so parallel secret-store tests keep
        // their redirect.
        let mut isolation = KeychainIsolation::acquire();
        isolation.set(crate::RUNTIME_ENV_FILE_ENV_KEY, "/daemon/root/.env");
        isolation.set("AIDAEMON_ENV_FILE", "/some/other/.env");

        assert_eq!(resolve_env_file_path(), PathBuf::from("/daemon/root/.env"));
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn expand_env_vars_reports_all_missing() {
        let _guard = ENV_EXPANSION_LOCK.lock().unwrap();
        std::env::remove_var("AIDAEMON_MISS_X");
        std::env::remove_var("AIDAEMON_MISS_Y");
        let result = expand_env_vars("a = \"${AIDAEMON_MISS_X}\"\nb = \"${AIDAEMON_MISS_Y}\"");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("AIDAEMON_MISS_X"));
        assert!(msg.contains("AIDAEMON_MISS_Y"));
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn terminal_web_app_url_defaults_to_hosted_terminal() {
        let _guard = TERMINAL_WEB_APP_URL_ENV_LOCK.lock().unwrap();
        let old = std::env::var("AIDAEMON_TERMINAL_WEB_APP_URL").ok();
        std::env::remove_var("AIDAEMON_TERMINAL_WEB_APP_URL");
        let cfg = TerminalConfig::default();
        assert_eq!(cfg.effective_web_app_url(), "https://terminal.aidaemon.ai");
        if let Some(old) = old {
            std::env::set_var("AIDAEMON_TERMINAL_WEB_APP_URL", old);
        }
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn terminal_web_app_url_env_override_wins() {
        let _guard = TERMINAL_WEB_APP_URL_ENV_LOCK.lock().unwrap();
        let old = std::env::var("AIDAEMON_TERMINAL_WEB_APP_URL").ok();
        std::env::set_var(
            "AIDAEMON_TERMINAL_WEB_APP_URL",
            "https://terminal.dev.aidaemon.ai",
        );
        let cfg = TerminalConfig {
            web_app_url: "https://terminal.aidaemon.ai".to_string(),
            ..TerminalConfig::default()
        };
        assert_eq!(
            cfg.effective_web_app_url(),
            "https://terminal.dev.aidaemon.ai"
        );
        if let Some(old) = old {
            std::env::set_var("AIDAEMON_TERMINAL_WEB_APP_URL", old);
        } else {
            std::env::remove_var("AIDAEMON_TERMINAL_WEB_APP_URL");
        }
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn terminal_bridge_enabled_defaults_to_true() {
        let _guard = TERMINAL_BRIDGE_ENABLED_ENV_LOCK.lock().unwrap();
        let old = std::env::var("AIDAEMON_TERMINAL_BRIDGE_ENABLED").ok();
        std::env::remove_var("AIDAEMON_TERMINAL_BRIDGE_ENABLED");
        let cfg = TerminalConfig::default();
        assert!(cfg.effective_bridge_enabled());
        if let Some(old) = old {
            std::env::set_var("AIDAEMON_TERMINAL_BRIDGE_ENABLED", old);
        }
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn terminal_bridge_enabled_env_override_wins() {
        let _guard = TERMINAL_BRIDGE_ENABLED_ENV_LOCK.lock().unwrap();
        let old = std::env::var("AIDAEMON_TERMINAL_BRIDGE_ENABLED").ok();
        std::env::set_var("AIDAEMON_TERMINAL_BRIDGE_ENABLED", "false");
        let cfg = TerminalConfig {
            bridge_enabled: true,
            ..TerminalConfig::default()
        };
        assert!(!cfg.effective_bridge_enabled());
        if let Some(old) = old {
            std::env::set_var("AIDAEMON_TERMINAL_BRIDGE_ENABLED", old);
        } else {
            std::env::remove_var("AIDAEMON_TERMINAL_BRIDGE_ENABLED");
        }
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn terminal_static_fallback_defaults_to_disabled() {
        let _guard = TERMINAL_STATIC_FALLBACK_ENV_LOCK.lock().unwrap();
        let old = std::env::var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK").ok();
        std::env::remove_var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK");
        let cfg = TerminalConfig::default();
        assert!(!cfg.effective_allow_static_token_fallback());
        if let Some(old) = old {
            std::env::set_var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK", old);
        }
    }

    #[test]
    #[serial_test::serial(env_vars)]
    fn terminal_static_fallback_env_override_wins() {
        let _guard = TERMINAL_STATIC_FALLBACK_ENV_LOCK.lock().unwrap();
        let old = std::env::var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK").ok();
        std::env::set_var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK", "true");
        let cfg = TerminalConfig {
            allow_static_token_fallback: false,
            ..TerminalConfig::default()
        };
        assert!(cfg.effective_allow_static_token_fallback());
        if let Some(old) = old {
            std::env::set_var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK", old);
        } else {
            std::env::remove_var("AIDAEMON_TERMINAL_ALLOW_STATIC_FALLBACK");
        }
    }

    #[test]
    fn provider_gateway_token_defaults_to_none() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gpt-4o"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.provider.gateway_token, None);
    }

    #[test]
    fn provider_gateway_token_parses_when_set() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"
gateway_token = "cf-gw-token"

[provider.models]
primary = "gpt-4o"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.provider.gateway_token.as_deref(), Some("cf-gw-token"));
    }

    #[test]
    fn provider_extra_headers_parse_when_set() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"
extra_headers = { "x-app" = "aidaemon", "x-env" = "dev" }

[provider.models]
primary = "gpt-4o"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        let headers = cfg.provider.extra_headers.expect("extra headers");
        assert_eq!(headers.get("x-app"), Some(&"aidaemon".to_string()));
        assert_eq!(headers.get("x-env"), Some(&"dev".to_string()));
    }

    #[test]
    fn provider_max_tokens_parses_when_set() {
        let toml = r#"
[provider]
kind = "anthropic"
api_key = "test-key"
max_tokens = 32768

[provider.models]
primary = "claude-sonnet-4-20250514"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.provider.max_tokens, Some(32768));
    }

    #[test]
    fn provider_llama_cpp_thinking_defaults_to_disabled() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gemma-4-26b"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert!(!cfg.provider.llama_cpp_thinking);
    }

    #[test]
    fn provider_llama_cpp_thinking_parses_when_enabled() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"
llama_cpp_thinking = true

[provider.models]
primary = "gemma-4-26b"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert!(cfg.provider.llama_cpp_thinking);
    }

    #[test]
    fn provider_structured_answer_models_defaults_to_empty() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gemma-4-26b"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert!(cfg.provider.structured_answer_models.is_empty());
    }

    #[test]
    fn provider_structured_answer_models_parses_when_set() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"
structured_answer_models = ["gemma-4-26b"]

[provider.models]
primary = "gemma-4-26b"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.provider.structured_answer_models, vec!["gemma-4-26b"]);
    }

    #[test]
    fn provider_kind_parses_xai_native() {
        let toml = r#"
[provider]
kind = "xai_native"
api_key = "test-key"

[provider.models]
primary = "grok-4"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.provider.kind, ProviderKind::XaiNative);
    }

    #[test]
    fn provider_models_support_default_and_fallback_keys() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
default = "kimi-k2.5"
fallback = ["mistral-nemo", "gpt-4o-mini"]
"#;
        let mut cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        cfg.provider.models.apply_defaults(&cfg.provider.kind);
        assert_eq!(cfg.provider.models.default_model, "kimi-k2.5");
        assert_eq!(
            cfg.provider.models.fallback_models,
            vec!["mistral-nemo".to_string(), "gpt-4o-mini".to_string()]
        );
        // Legacy compatibility mirrors default/fallback.
        assert_eq!(cfg.provider.models.primary, "kimi-k2.5");
    }

    #[test]
    fn provider_models_legacy_keys_derive_default_and_fallback() {
        let mut models = ModelsConfig {
            default_model: String::new(),
            fallback_models: Vec::new(),
            primary: "primary-model".to_string(),
            fast: "fast-model".to_string(),
            smart: "smart-model".to_string(),
        };
        models.apply_defaults(&ProviderKind::OpenaiCompatible);
        assert_eq!(models.default_model, "primary-model");
        assert_eq!(
            models.fallback_models,
            vec!["smart-model".to_string(), "fast-model".to_string()]
        );
    }

    #[test]
    fn provider_fallback_configs_parse() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"
fallback = ["primary-backup"]

[[provider.fallbacks]]
kind = "anthropic"
api_key = "secondary-key"

[provider.fallbacks.models]
default = "claude-sonnet-4-20250514"
fallback = ["claude-3-5-haiku-latest"]
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.provider.fallbacks.len(), 1);
        assert_eq!(cfg.provider.fallbacks[0].kind, ProviderKind::Anthropic);
        assert_eq!(cfg.provider.fallbacks[0].api_key, "secondary-key");
        assert_eq!(
            cfg.provider.fallbacks[0].models.default_model,
            "claude-sonnet-4-20250514"
        );
    }

    #[test]
    fn provider_failover_alias_still_parses() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
default = "primary-model"

[[provider.failover]]
kind = "xai_native"
api_key = "secondary-key"

[provider.failover.models]
default = "grok-4"
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.provider.fallbacks.len(), 1);
        assert_eq!(cfg.provider.fallbacks[0].kind, ProviderKind::XaiNative);
        assert_eq!(cfg.provider.fallbacks[0].models.default_model, "grok-4");
    }

    #[test]
    fn provider_fallback_model_defaults_apply_recursively() {
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "primary-key"

[provider.models]
primary = "primary-model"
fast = "primary-fast"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "secondary-key"

[provider.fallbacks.models]
primary = "claude-sonnet-4-20250514"
smart = "claude-3-5-haiku-latest"
"#;
        let mut cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        cfg.provider.apply_model_defaults_recursive();
        assert_eq!(cfg.provider.models.default_model, "primary-model");
        assert_eq!(
            cfg.provider.models.fallback_models,
            vec!["primary-fast".to_string()]
        );
        assert_eq!(
            cfg.provider.fallbacks[0].models.default_model,
            "claude-sonnet-4-20250514"
        );
        assert_eq!(
            cfg.provider.fallbacks[0].models.fallback_models,
            vec!["claude-3-5-haiku-latest".to_string()]
        );
    }

    #[test]
    fn iteration_limit_default_is_unlimited() {
        let config = SubagentsConfig::default();
        assert!(matches!(
            config.iteration_limit,
            IterationLimitConfig::Unlimited
        ));
    }

    #[test]
    fn iteration_limit_effective_returns_unlimited_when_defaults_unchanged() {
        let config = SubagentsConfig::default();
        assert!(matches!(
            config.effective_iteration_limit(),
            IterationLimitConfig::Unlimited
        ));
    }

    #[test]
    fn iteration_limit_effective_migrates_legacy_config() {
        // Simulate old config with custom values
        let config = SubagentsConfig {
            max_iterations: 15,
            max_iterations_cap: 30,
            ..SubagentsConfig::default()
        };

        match config.effective_iteration_limit() {
            IterationLimitConfig::Hard { initial, cap } => {
                assert_eq!(initial, 15);
                assert_eq!(cap, 30);
            }
            _ => panic!("Expected Hard mode migration"),
        }
    }

    #[test]
    fn iteration_limit_explicit_soft_preserved() {
        // Even with legacy values set, explicit config takes precedence
        let config = SubagentsConfig {
            iteration_limit: IterationLimitConfig::Soft {
                threshold: 50,
                warn_at: 40,
            },
            max_iterations: 15,
            max_iterations_cap: 30,
            ..SubagentsConfig::default()
        };

        match config.effective_iteration_limit() {
            IterationLimitConfig::Soft { threshold, warn_at } => {
                assert_eq!(threshold, 50);
                assert_eq!(warn_at, 40);
            }
            _ => panic!("Expected Soft mode to be preserved"),
        }
    }

    #[test]
    fn task_timeout_is_accepted_but_unset_by_default() {
        let config = SubagentsConfig::default();
        assert_eq!(config.task_timeout_secs, None);
        let parsed: SubagentsConfig = toml::from_str("task_timeout_secs = 600").unwrap();
        assert_eq!(parsed.task_timeout_secs, Some(600));
    }

    #[test]
    fn write_consistency_policy_defaults_are_applied() {
        let policy = PolicyConfig::default();
        assert_eq!(policy.write_consistency.max_abs_global_delta, 3);
        assert_eq!(policy.write_consistency.max_session_mismatch_count, 0);
        assert_eq!(policy.write_consistency.max_stale_task_starts, 0);
        assert_eq!(policy.write_consistency.max_missing_message_id_events, 0);
    }

    #[test]
    fn write_consistency_policy_can_be_overridden_in_toml() {
        let toml = r#"
[provider]
api_key = "test-key"
kind = "openai_compatible"

[provider.models]
primary = "gpt-4o"
fast = "gpt-4o-mini"
smart = "gpt-4o"

[terminal]
allowed_prefixes = ["ls"]

[policy.write_consistency]
max_abs_global_delta = 7
max_session_mismatch_count = 2
max_stale_task_starts = 1
max_missing_message_id_events = 4
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        assert_eq!(cfg.policy.write_consistency.max_abs_global_delta, 7);
        assert_eq!(cfg.policy.write_consistency.max_session_mismatch_count, 2);
        assert_eq!(cfg.policy.write_consistency.max_stale_task_starts, 1);
        assert_eq!(
            cfg.policy.write_consistency.max_missing_message_id_events,
            4
        );
    }

    #[test]
    fn queue_policy_normalizes_invalid_values() {
        let policy = QueuePolicyConfig {
            approval_capacity: 0,
            media_capacity: 0,
            trigger_event_capacity: 0,
            warning_ratio: f32::NAN,
            overload_ratio: f32::INFINITY,
            lanes: QueueLanePoliciesConfig::default(),
            adaptive_shedding: Some(true),
            fair_trigger_sessions: Some(true),
            fair_trigger_session_window_secs: Some(0),
            fair_trigger_max_events_per_session: Some(0),
        };
        let normalized = policy.normalized();
        assert_eq!(normalized.approval_capacity, 1);
        assert_eq!(normalized.media_capacity, 1);
        assert_eq!(normalized.trigger_event_capacity, 1);
        assert_eq!(normalized.warning_ratio, 0.75);
        assert_eq!(normalized.overload_ratio, 0.90);
        assert_eq!(normalized.lanes.approval.fair_session_window_secs, 1);
        assert_eq!(normalized.lanes.media.fair_session_window_secs, 1);
        assert_eq!(normalized.lanes.trigger.fair_session_window_secs, 1);
        assert_eq!(normalized.lanes.approval.fair_max_events_per_session, 1);
        assert_eq!(normalized.lanes.media.fair_max_events_per_session, 1);
        assert_eq!(normalized.lanes.trigger.fair_max_events_per_session, 1);
    }

    #[test]
    fn queue_policy_legacy_shared_knobs_apply_to_all_lanes() {
        let policy = QueuePolicyConfig {
            adaptive_shedding: Some(false),
            fair_trigger_sessions: Some(false),
            fair_trigger_session_window_secs: Some(13),
            fair_trigger_max_events_per_session: Some(2),
            ..QueuePolicyConfig::default()
        };
        let normalized = policy.normalized();
        for lane in [
            &normalized.lanes.approval,
            &normalized.lanes.media,
            &normalized.lanes.trigger,
        ] {
            assert!(!lane.adaptive_shedding);
            assert!(!lane.fair_sessions);
            assert_eq!(lane.fair_session_window_secs, 13);
            assert_eq!(lane.fair_max_events_per_session, 2);
        }
    }

    #[test]
    fn queue_policy_lane_specific_overrides_take_precedence() {
        let mut policy = QueuePolicyConfig {
            adaptive_shedding: Some(false),
            fair_trigger_sessions: Some(false),
            fair_trigger_session_window_secs: Some(13),
            fair_trigger_max_events_per_session: Some(2),
            ..QueuePolicyConfig::default()
        };
        policy.lanes.media = QueueLanePolicyConfig {
            adaptive_shedding: true,
            fair_sessions: true,
            fair_session_window_secs: 99,
            fair_max_events_per_session: 9,
        };
        let normalized = policy.normalized();
        assert!(!normalized.lanes.approval.adaptive_shedding);
        assert!(!normalized.lanes.trigger.adaptive_shedding);
        assert_eq!(normalized.lanes.approval.fair_session_window_secs, 13);
        assert_eq!(normalized.lanes.trigger.fair_session_window_secs, 13);
        assert!(normalized.lanes.media.adaptive_shedding);
        assert!(normalized.lanes.media.fair_sessions);
        assert_eq!(normalized.lanes.media.fair_session_window_secs, 99);
        assert_eq!(normalized.lanes.media.fair_max_events_per_session, 9);
    }

    #[test]
    fn queue_policy_lane_specific_toml_is_supported() {
        let toml = r#"
[provider]
api_key = "test-key"
kind = "openai_compatible"

[provider.models]
primary = "gpt-4o"
fast = "gpt-4o-mini"
smart = "gpt-4o"

[terminal]
allowed_prefixes = ["ls"]

[daemon.queue_policy.lanes.approval]
adaptive_shedding = false
fair_sessions = false
fair_session_window_secs = 45
fair_max_events_per_session = 3

[daemon.queue_policy.lanes.media]
adaptive_shedding = true
fair_sessions = true
fair_session_window_secs = 20
fair_max_events_per_session = 2
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        let queue = cfg.daemon.queue_policy.normalized();
        assert!(!queue.lanes.approval.adaptive_shedding);
        assert_eq!(queue.lanes.approval.fair_session_window_secs, 45);
        assert_eq!(queue.lanes.approval.fair_max_events_per_session, 3);
        assert!(queue.lanes.media.adaptive_shedding);
        assert_eq!(queue.lanes.media.fair_session_window_secs, 20);
        assert_eq!(queue.lanes.media.fair_max_events_per_session, 2);
    }

    #[test]
    fn all_telegram_bots_merges_legacy_and_preserves_webhook_config() {
        let toml = r#"
[provider]
api_key = "test-key"
kind = "openai_compatible"

[provider.models]
primary = "gpt-4o"
fast = "gpt-4o-mini"
smart = "gpt-4o"

[terminal]
allowed_prefixes = ["ls"]

[[telegram_bots]]
bot_token = "array-token"
allowed_user_ids = [111]

[telegram_bots.webhook]
enabled = true
public_url = "https://array.example.com/hook"
listen_addr = "0.0.0.0:8443"
path = "/array"
max_connections = 42
drop_pending_updates = true

[telegram]
bot_token = "legacy-token"
allowed_user_ids = [222]

[telegram.webhook]
enabled = true
public_url = "https://legacy.example.com/hook"
listen_addr = "127.0.0.1:9443"
path = "/legacy"
max_connections = 21
drop_pending_updates = false
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        let bots = cfg.all_telegram_bots();
        assert_eq!(bots.len(), 2);

        assert_eq!(bots[0].bot_token, "array-token");
        assert_eq!(bots[0].allowed_user_ids, vec![111]);
        assert!(bots[0].webhook.enabled);
        assert_eq!(
            bots[0].webhook.public_url.as_deref(),
            Some("https://array.example.com/hook")
        );
        assert_eq!(bots[0].webhook.listen_addr.as_deref(), Some("0.0.0.0:8443"));
        assert_eq!(bots[0].webhook.path.as_deref(), Some("/array"));
        assert_eq!(bots[0].webhook.max_connections, Some(42));
        assert!(bots[0].webhook.drop_pending_updates);

        assert_eq!(bots[1].bot_token, "legacy-token");
        assert_eq!(bots[1].allowed_user_ids, vec![222]);
        assert!(bots[1].webhook.enabled);
        assert_eq!(
            bots[1].webhook.public_url.as_deref(),
            Some("https://legacy.example.com/hook")
        );
        assert_eq!(
            bots[1].webhook.listen_addr.as_deref(),
            Some("127.0.0.1:9443")
        );
        assert_eq!(bots[1].webhook.path.as_deref(), Some("/legacy"));
        assert_eq!(bots[1].webhook.max_connections, Some(21));
        assert!(!bots[1].webhook.drop_pending_updates);
    }

    #[test]
    fn telegram_webhook_defaults_to_disabled_when_omitted() {
        let toml = r#"
[provider]
api_key = "test-key"
kind = "openai_compatible"

[provider.models]
primary = "gpt-4o"
fast = "gpt-4o-mini"
smart = "gpt-4o"

[terminal]
allowed_prefixes = ["ls"]

[telegram]
bot_token = "legacy-token"
allowed_user_ids = [333]
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        let bots = cfg.all_telegram_bots();
        assert_eq!(bots.len(), 1);
        assert_eq!(bots[0].bot_token, "legacy-token");
        assert!(!bots[0].webhook.enabled);
        assert_eq!(bots[0].webhook.public_url, None);
        assert_eq!(bots[0].webhook.listen_addr, None);
        assert_eq!(bots[0].webhook.path, None);
        assert_eq!(bots[0].webhook.max_connections, None);
        assert!(!bots[0].webhook.drop_pending_updates);
    }

    #[cfg(feature = "slack")]
    #[test]
    fn all_slack_bots_merges_legacy_without_enabled_gate() {
        let toml = r#"
[provider]
api_key = "test-key"
kind = "openai_compatible"

[provider.models]
primary = "gpt-4o"
fast = "gpt-4o-mini"
smart = "gpt-4o"

[terminal]
allowed_prefixes = ["ls"]

[slack]
enabled = false
app_token = "xapp-123"
bot_token = "xoxb-456"
allowed_user_ids = ["U123"]
"#;
        let cfg: AppConfig = toml::from_str(toml).expect("parse app config");
        let bots = cfg.all_slack_bots();
        assert_eq!(bots.len(), 1);
        assert_eq!(bots[0].app_token, "xapp-123");
        assert_eq!(bots[0].bot_token, "xoxb-456");
        assert_eq!(bots[0].allowed_user_ids, vec!["U123"]);
    }

    // === Browser session_isolation parsing + resolution ===

    /// A BrowserConfig with no profile and no remote-debugging port (truly
    /// ephemeral), letting us isolate the `session_isolation` field's behavior.
    fn ephemeral_browser_config(isolation: Option<SessionIsolation>) -> BrowserConfig {
        BrowserConfig {
            enabled: true,
            headless: true,
            screenshot_width: 1280,
            screenshot_height: 720,
            remote_debugging_port: None,
            user_data_dir: None,
            profile: None,
            session_isolation: isolation,
            ..BrowserConfig::default()
        }
    }

    #[test]
    fn parse_session_isolation_page() {
        let toml = r#"
enabled = true
user_data_dir = "/tmp/profile"
session_isolation = "page"
"#;
        let cfg: BrowserConfig = toml::from_str(toml).expect("parse browser config");
        assert_eq!(cfg.session_isolation, Some(SessionIsolation::Page));
    }

    #[test]
    fn parse_session_isolation_browser_context() {
        let toml = r#"
enabled = true
session_isolation = "browser_context"
"#;
        let cfg: BrowserConfig = toml::from_str(toml).expect("parse browser config");
        assert_eq!(
            cfg.session_isolation,
            Some(SessionIsolation::BrowserContext)
        );
    }

    #[test]
    fn parse_session_isolation_omitted_is_none() {
        let toml = r#"
enabled = true
"#;
        let cfg: BrowserConfig = toml::from_str(toml).expect("parse browser config");
        assert_eq!(cfg.session_isolation, None);
    }

    #[test]
    fn parse_session_isolation_invalid_value_rejected() {
        let toml = r#"
enabled = true
session_isolation = "incognito"
"#;
        let result: Result<BrowserConfig, _> = toml::from_str(toml);
        assert!(
            result.is_err(),
            "unknown session_isolation variant should fail to parse"
        );
    }

    #[test]
    fn browser_timeout_defaults_are_sane() {
        let cfg = BrowserConfig::default();
        assert_eq!(cfg.nav_timeout(), std::time::Duration::from_secs(30));
        assert_eq!(cfg.element_timeout(), std::time::Duration::from_secs(10));
        assert_eq!(cfg.action_timeout(), std::time::Duration::from_secs(30));
    }

    #[test]
    fn browser_timeouts_clamp_out_of_range_values() {
        // Zero is clamped up to the per-field minimum (1s) so navigation/wait
        // never gives up before it can start.
        let mut cfg = BrowserConfig {
            nav_timeout_secs: 0,
            element_timeout_secs: 0,
            action_timeout_secs: 0,
            ..Default::default()
        };
        assert_eq!(cfg.nav_timeout(), std::time::Duration::from_secs(1));
        assert_eq!(cfg.element_timeout(), std::time::Duration::from_secs(1));
        assert_eq!(cfg.action_timeout(), std::time::Duration::from_secs(1));

        // Absurdly large values are clamped down to the per-field maximum so a
        // typo can never hang the tool indefinitely.
        cfg.nav_timeout_secs = 1_000_000;
        cfg.element_timeout_secs = 1_000_000;
        cfg.action_timeout_secs = 1_000_000;
        assert_eq!(cfg.nav_timeout(), std::time::Duration::from_secs(120));
        assert_eq!(cfg.element_timeout(), std::time::Duration::from_secs(120));
        assert_eq!(cfg.action_timeout(), std::time::Duration::from_secs(300));
    }

    #[test]
    fn browser_timeouts_in_range_pass_through() {
        let cfg = BrowserConfig {
            nav_timeout_secs: 45,
            element_timeout_secs: 5,
            action_timeout_secs: 90,
            ..Default::default()
        };
        assert_eq!(cfg.nav_timeout(), std::time::Duration::from_secs(45));
        assert_eq!(cfg.element_timeout(), std::time::Duration::from_secs(5));
        assert_eq!(cfg.action_timeout(), std::time::Duration::from_secs(90));
    }

    #[test]
    fn resolve_auto_ephemeral_is_browser_context() {
        let cfg = ephemeral_browser_config(None);
        assert_eq!(
            cfg.resolve_session_isolation(),
            Ok(SessionIsolation::BrowserContext)
        );
    }

    #[test]
    fn resolve_auto_with_profile_is_page() {
        let mut cfg = ephemeral_browser_config(None);
        cfg.user_data_dir = Some("/tmp/profile".to_string());
        assert_eq!(cfg.resolve_session_isolation(), Ok(SessionIsolation::Page));
    }

    #[test]
    fn resolve_auto_with_remote_port_is_page() {
        let mut cfg = ephemeral_browser_config(None);
        cfg.remote_debugging_port = Some(9222);
        assert_eq!(cfg.resolve_session_isolation(), Ok(SessionIsolation::Page));
    }

    #[test]
    fn resolve_page_with_profile_is_ok_page() {
        let mut cfg = ephemeral_browser_config(Some(SessionIsolation::Page));
        cfg.user_data_dir = Some("/tmp/profile".to_string());
        assert_eq!(cfg.resolve_session_isolation(), Ok(SessionIsolation::Page));
    }

    #[test]
    fn resolve_browser_context_ephemeral_is_ok() {
        let cfg = ephemeral_browser_config(Some(SessionIsolation::BrowserContext));
        assert_eq!(
            cfg.resolve_session_isolation(),
            Ok(SessionIsolation::BrowserContext)
        );
    }

    #[test]
    fn resolve_browser_context_with_profile_is_err() {
        let mut cfg = ephemeral_browser_config(Some(SessionIsolation::BrowserContext));
        cfg.user_data_dir = Some("/tmp/profile".to_string());
        let err = cfg
            .resolve_session_isolation()
            .expect_err("browser_context + persistent profile must be rejected");
        assert!(
            err.contains("browser_context") && err.contains("user_data_dir"),
            "error should explain the incompatibility: {err}"
        );
    }

    #[test]
    fn resolve_browser_context_with_remote_port_is_err() {
        let mut cfg = ephemeral_browser_config(Some(SessionIsolation::BrowserContext));
        cfg.remote_debugging_port = Some(9222);
        let err = cfg
            .resolve_session_isolation()
            .expect_err("browser_context + remote-debugging Chrome must be rejected");
        assert!(
            err.contains("browser_context") && err.contains("remote_debugging_port"),
            "error should explain the incompatibility: {err}"
        );
    }

    #[test]
    fn shipped_default_config_resolves_without_error() {
        // The shipped default uses a persistent profile (user_data_dir set) and
        // session_isolation = None. It must resolve to Page, not error.
        let cfg = BrowserConfig::default();
        assert_eq!(cfg.session_isolation, None);
        assert_eq!(cfg.resolve_session_isolation(), Ok(SessionIsolation::Page));
    }

    #[test]
    fn config_with_removed_correction_sections_still_parses() {
        // The self-correction feature was removed. An existing config.toml that
        // still carries its `[self_correction]` section or the retention key for
        // its attempts table must keep loading; the keys are simply ignored.
        let toml = r#"
[provider]
kind = "openai_compatible"
api_key = "test-key"

[provider.models]
primary = "gpt-4o"

[self_correction]
enabled = true
correction_bypass_enabled = true
max_attempts = 5
shadow_mode = false

[state.retention]
self_correction_attempts_days = 30
"#;
        let app: AppConfig =
            toml::from_str(toml).expect("config with removed sections must still parse");
        assert_eq!(app.state.retention.error_solutions_days, 90);
    }
}
