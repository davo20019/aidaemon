use serde::Deserialize;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

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

const KEYCHAIN_SERVICE: &str = "aidaemon";

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

fn resolve_from_keychain(field_name: &str) -> anyhow::Result<String> {
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, field_name)?;
    entry
        .get_password()
        .map_err(|e| anyhow::anyhow!("Failed to read '{}' from OS keychain: {}", field_name, e))
}

/// Store a secret in the OS keychain under the `aidaemon` service.
pub fn store_in_keychain(field_name: &str, value: &str) -> anyhow::Result<()> {
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, field_name)?;
    entry.set_password(value)?;
    Ok(())
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub provider: ProviderConfig,
    /// Legacy single Telegram bot config (for backward compatibility).
    #[serde(default)]
    pub telegram: Option<TelegramConfig>,
    /// Array of named Telegram bots (preferred for multi-bot setups).
    #[serde(default)]
    pub telegram_bots: Vec<TelegramBotConfig>,
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
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub triggers: TriggersConfig,
    #[serde(default)]
    pub mcp: HashMap<String, McpServerConfig>,
    #[serde(default)]
    pub browser: BrowserConfig,
    #[serde(default)]
    pub skills: SkillsConfig,
    #[serde(default)]
    pub subagents: SubagentsConfig,
    #[serde(default)]
    pub cli_agents: CliAgentsConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    #[serde(default)]
    pub files: FilesConfig,
    #[serde(default)]
    pub health: HealthConfig,
    #[serde(default)]
    pub updates: UpdateConfig,
}

#[derive(Deserialize, Clone)]
pub struct ProviderConfig {
    #[serde(default)]
    pub kind: ProviderKind,
    pub api_key: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub models: ModelsConfig,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("api_key", &redact(&self.api_key))
            .field("base_url", &self.base_url)
            .field("models", &self.models)
            .finish()
    }
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    #[default]
    OpenaiCompatible,
    GoogleGenai,
    Anthropic,
}


fn default_base_url() -> String {
    "https://api.openai.com/v1".to_string()
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ModelsConfig {
    #[serde(default)]
    pub primary: String,
    #[serde(default)]
    pub fast: String,
    #[serde(default)]
    pub smart: String,
}

impl ModelsConfig {
    /// Fill in unset model tiers. `fast` and `smart` default to `primary`;
    /// `primary` defaults to a sensible model for the provider kind.
    pub fn apply_defaults(&mut self, provider_kind: &ProviderKind) {
        if self.primary.is_empty() {
            self.primary = match provider_kind {
                ProviderKind::GoogleGenai => "gemini-3-flash-preview".to_string(),
                ProviderKind::Anthropic => "claude-sonnet-4-20250514".to_string(),
                ProviderKind::OpenaiCompatible => "openai/gpt-4o".to_string(),
            };
        }
        if self.fast.is_empty() {
            self.fast = self.primary.clone();
        }
        if self.smart.is_empty() {
            self.smart = self.primary.clone();
        }
    }
}

/// Single Telegram bot configuration (legacy format).
#[derive(Deserialize, Clone)]
pub struct TelegramConfig {
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
}

impl fmt::Debug for TelegramConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TelegramConfig")
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
            .finish()
    }
}

/// Telegram bot configuration for multi-bot support.
/// Bot username is auto-detected from Telegram API at startup.
#[derive(Deserialize, Clone)]
pub struct TelegramBotConfig {
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
}

impl fmt::Debug for TelegramBotConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TelegramBotConfig")
            .field("bot_token", &redact(&self.bot_token))
            .field("allowed_user_ids", &self.allowed_user_ids)
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
    /// Optional daily token budget. When set, LLM calls will be rejected
    /// once cumulative daily usage exceeds this limit. Resets at midnight UTC.
    #[serde(default)]
    pub daily_token_budget: Option<u64>,
}

impl fmt::Debug for StateConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateConfig")
            .field("db_path", &self.db_path)
            .field("working_memory_cap", &self.working_memory_cap)
            .field("consolidation_interval_hours", &self.consolidation_interval_hours)
            .field("encryption_key", &redact_option(&self.encryption_key))
            .field("max_facts", &self.max_facts)
            .field("daily_token_budget", &self.daily_token_budget)
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
        }
    }
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
    100
}

/// Permission mode re-exported for config deserialization.
pub use crate::tools::command_risk::PermissionMode;

#[derive(Debug, Deserialize, Clone)]
pub struct TerminalConfig {
    #[serde(default = "default_allowed_prefixes")]
    pub allowed_prefixes: Vec<String>,
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
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            allowed_prefixes: default_allowed_prefixes(),
            initial_timeout_secs: default_initial_timeout_secs(),
            max_output_chars: default_max_output_chars(),
            permission_mode: PermissionMode::default(),
        }
    }
}

fn default_initial_timeout_secs() -> u64 {
    30
}
fn default_max_output_chars() -> usize {
    4000
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
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            health_port: default_health_port(),
            health_bind: default_health_bind(),
            dashboard_enabled: default_dashboard_enabled(),
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

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TriggersConfig {
    pub email: Option<EmailTriggerConfig>,
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

#[derive(Debug, Deserialize, Clone)]
pub struct McpServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

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
    /// Path to Chrome user data directory to reuse an existing profile's sessions/cookies.
    /// e.g. "~/Library/Application Support/Google/Chrome"
    pub user_data_dir: Option<String>,
    /// Chrome profile directory name within user_data_dir (default: "Default").
    /// Other profiles are typically "Profile 1", "Profile 2", etc.
    pub profile: Option<String>,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            headless: default_headless(),
            screenshot_width: default_screenshot_width(),
            screenshot_height: default_screenshot_height(),
            user_data_dir: None,
            profile: None,
        }
    }
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

/// Iteration limit configuration for agent loops.
/// Controls how iteration limits are enforced (or not enforced).
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum IterationLimitConfig {
    /// No iteration limit - run until natural completion, token budget, or timeout.
    /// This is the recommended default for complex tasks.
    #[default]
    Unlimited,
    /// Soft limit with warnings but no forced termination.
    /// Agent will be warned at `warn_at` iterations and continue until natural completion.
    Soft {
        threshold: usize,
        warn_at: usize,
    },
    /// Legacy hard limit behavior (backward compatibility).
    /// Agent will be forced to stop at `cap` iterations.
    Hard {
        initial: usize,
        cap: usize,
    },
}

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
    #[serde(default = "default_subagents_timeout_secs")]
    pub timeout_secs: u64,
    /// Iteration limit configuration. Defaults to Unlimited.
    /// If not set but max_iterations/max_iterations_cap are set, auto-migrates to Hard mode.
    #[serde(default)]
    pub iteration_limit: IterationLimitConfig,
    /// Optional time limit per task in seconds. Default: 1800 (30 minutes).
    #[serde(default = "default_task_timeout_secs")]
    pub task_timeout_secs: Option<u64>,
    /// Optional token budget per task. When exceeded, agent gracefully summarizes and stops.
    #[serde(default)]
    pub task_token_budget: Option<u64>,
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
            task_timeout_secs: default_task_timeout_secs(),
            task_token_budget: None,
        }
    }
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
fn default_task_timeout_secs() -> Option<u64> {
    Some(1800) // 30 minutes default
}

#[derive(Debug, Deserialize, Clone)]
pub struct CliAgentsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cli_agents_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_cli_agents_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default)]
    pub tools: HashMap<String, CliToolConfig>,
}

impl Default for CliAgentsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout_secs: default_cli_agents_timeout_secs(),
            max_output_chars: default_cli_agents_max_output_chars(),
            tools: HashMap::new(),
        }
    }
}

fn default_cli_agents_timeout_secs() -> u64 {
    600
}
fn default_cli_agents_max_output_chars() -> usize {
    16000
}

#[derive(Debug, Deserialize, Clone)]
pub struct CliToolConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub max_output_chars: Option<usize>,
}

#[derive(Deserialize, Clone, Default)]
pub struct SearchConfig {
    #[serde(default)]
    pub backend: SearchBackendKind,
    #[serde(default)]
    pub api_key: String,
}

impl fmt::Debug for SearchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchConfig")
            .field("backend", &self.backend)
            .field("api_key", &redact(&self.api_key))
            .finish()
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchBackendKind {
    #[default]
    DuckDuckGo,
    Brave,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct SchedulerConfig {
    #[serde(default = "default_scheduler_enabled")]
    pub enabled: bool,
    #[serde(default = "default_tick_interval_secs")]
    pub tick_interval_secs: u64,
    #[serde(default)]
    pub tasks: Vec<ScheduledTaskConfig>,
}

fn default_scheduler_enabled() -> bool {
    true
}

fn default_tick_interval_secs() -> u64 {
    30
}

#[derive(Debug, Deserialize, Clone)]
pub struct ScheduledTaskConfig {
    pub name: String,
    pub schedule: String,
    pub prompt: String,
    #[serde(default)]
    pub oneshot: bool,
    #[serde(default)]
    pub trusted: bool,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct HealthConfig {
    #[serde(default = "default_health_enabled")]
    pub enabled: bool,
    #[serde(default = "default_health_tick_interval_secs")]
    pub tick_interval_secs: u64,
    #[serde(default = "default_health_result_retention_days")]
    pub result_retention_days: u32,
    #[serde(default)]
    pub probes: Vec<HealthProbeConfig>,
}

fn default_health_enabled() -> bool {
    true
}

fn default_health_tick_interval_secs() -> u64 {
    30
}

fn default_health_result_retention_days() -> u32 {
    7
}

#[derive(Debug, Deserialize, Clone)]
pub struct HealthProbeConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub probe_type: String,
    pub target: String,
    pub schedule: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub config: HealthProbeConfigOptions,
    #[serde(default)]
    pub consecutive_failures_alert: Option<u32>,
    #[serde(default)]
    pub latency_threshold_ms: Option<u32>,
    #[serde(default)]
    pub alert_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct HealthProbeConfigOptions {
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub expected_status: Option<u16>,
    #[serde(default)]
    pub expected_body: Option<String>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub headers: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub max_age_secs: Option<u64>,
    #[serde(default)]
    pub expected_exit_code: Option<i32>,
}

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

#[derive(Debug, Deserialize, Clone)]
pub struct FilesConfig {
    #[serde(default = "default_files_enabled")]
    pub enabled: bool,
    #[serde(default = "default_inbox_dir")]
    pub inbox_dir: String,
    #[serde(default = "default_outbox_dirs")]
    pub outbox_dirs: Vec<String>,
    #[serde(default = "default_max_file_size_mb")]
    pub max_file_size_mb: u64,
    #[serde(default = "default_retention_hours")]
    pub retention_hours: u64,
}

impl Default for FilesConfig {
    fn default() -> Self {
        Self {
            enabled: default_files_enabled(),
            inbox_dir: default_inbox_dir(),
            outbox_dirs: default_outbox_dirs(),
            max_file_size_mb: default_max_file_size_mb(),
            retention_hours: default_retention_hours(),
        }
    }
}

fn default_files_enabled() -> bool {
    true
}
fn default_inbox_dir() -> String {
    "~/.aidaemon/files/inbox".to_string()
}
fn default_outbox_dirs() -> Vec<String> {
    vec!["~".to_string()]
}
fn default_max_file_size_mb() -> u64 {
    10
}
fn default_retention_hours() -> u64 {
    24
}

impl AppConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let expanded = expand_env_vars(&content)?;
        let mut config: AppConfig = toml::from_str(&expanded)?;
        config.provider.models.apply_defaults(&config.provider.kind);
        config.resolve_secrets()?;
        Ok(config)
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
            if legacy.enabled && !legacy.bot_token.is_empty() && !legacy.app_token.is_empty() {
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
        if self.provider.api_key == "keychain" {
            self.provider.api_key = resolve_from_keychain("api_key")?;
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
                let key = if i == 0 { "bot_token".to_string() } else { format!("telegram_bot_token_{}", i) };
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

        #[cfg(feature = "discord")]
        {
            if let Some(ref mut discord) = self.discord {
                if discord.bot_token == "keychain" {
                    discord.bot_token = resolve_from_keychain("discord_bot_token")?;
                }
            }
            for (i, bot) in self.discord_bots.iter_mut().enumerate() {
                if bot.bot_token == "keychain" {
                    let key = if i == 0 { "discord_bot_token".to_string() } else { format!("discord_bot_token_{}", i) };
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
                    let key = if i == 0 { "slack_app_token".to_string() } else { format!("slack_app_token_{}", i) };
                    bot.app_token = resolve_from_keychain(&key)?;
                }
                if bot.bot_token == "keychain" {
                    let key = if i == 0 { "slack_bot_token".to_string() } else { format!("slack_bot_token_{}", i) };
                    bot.bot_token = resolve_from_keychain(&key)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_env_vars_replaces_set_variable() {
        std::env::set_var("AIDAEMON_TEST_VAR", "hello");
        let result = expand_env_vars("key = \"${AIDAEMON_TEST_VAR}\"").unwrap();
        assert_eq!(result, "key = \"hello\"");
        std::env::remove_var("AIDAEMON_TEST_VAR");
    }

    #[test]
    fn expand_env_vars_errors_on_missing() {
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
    fn expand_env_vars_handles_multiple() {
        std::env::set_var("AIDAEMON_TEST_A", "aaa");
        std::env::set_var("AIDAEMON_TEST_B", "bbb");
        let result = expand_env_vars("a = \"${AIDAEMON_TEST_A}\"\nb = \"${AIDAEMON_TEST_B}\"").unwrap();
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
    fn expand_env_vars_reports_all_missing() {
        std::env::remove_var("AIDAEMON_MISS_X");
        std::env::remove_var("AIDAEMON_MISS_Y");
        let result = expand_env_vars("a = \"${AIDAEMON_MISS_X}\"\nb = \"${AIDAEMON_MISS_Y}\"");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("AIDAEMON_MISS_X"));
        assert!(msg.contains("AIDAEMON_MISS_Y"));
    }

    #[test]
    fn iteration_limit_default_is_unlimited() {
        let config = SubagentsConfig::default();
        assert!(matches!(config.iteration_limit, IterationLimitConfig::Unlimited));
    }

    #[test]
    fn iteration_limit_effective_returns_unlimited_when_defaults_unchanged() {
        let config = SubagentsConfig::default();
        assert!(matches!(config.effective_iteration_limit(), IterationLimitConfig::Unlimited));
    }

    #[test]
    fn iteration_limit_effective_migrates_legacy_config() {
        let mut config = SubagentsConfig::default();
        // Simulate old config with custom values
        config.max_iterations = 15;
        config.max_iterations_cap = 30;

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
        let mut config = SubagentsConfig::default();
        config.iteration_limit = IterationLimitConfig::Soft {
            threshold: 50,
            warn_at: 40,
        };
        // Even with legacy values set, explicit config takes precedence
        config.max_iterations = 15;
        config.max_iterations_cap = 30;

        match config.effective_iteration_limit() {
            IterationLimitConfig::Soft { threshold, warn_at } => {
                assert_eq!(threshold, 50);
                assert_eq!(warn_at, 40);
            }
            _ => panic!("Expected Soft mode to be preserved"),
        }
    }

    #[test]
    fn task_timeout_default_is_30_minutes() {
        let config = SubagentsConfig::default();
        assert_eq!(config.task_timeout_secs, Some(1800));
    }
}
