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

/// Returns true if keychain access is disabled via `AIDAEMON_NO_KEYCHAIN=1`.
fn keychain_disabled() -> bool {
    std::env::var("AIDAEMON_NO_KEYCHAIN").is_ok_and(|v| v == "1" || v == "true")
}

pub fn resolve_from_keychain(field_name: &str) -> anyhow::Result<String> {
    // Try env var first (uppercased key name, e.g. oauth_google_access_token → OAUTH_GOOGLE_ACCESS_TOKEN).
    // This allows .env-based credential storage for headless/remote operation.
    let env_key = field_name.to_uppercase();
    if let Ok(val) = std::env::var(&env_key) {
        if !val.is_empty() {
            return Ok(val);
        }
    }

    if keychain_disabled() {
        anyhow::bail!(
            "Keychain disabled and env var '{}' not set for '{}'",
            env_key,
            field_name
        );
    }
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, field_name)?;
    entry
        .get_password()
        .map_err(|e| anyhow::anyhow!("Failed to read '{}' from OS keychain: {}", field_name, e))
}

/// Store a secret in the OS keychain under the `aidaemon` service.
pub fn store_in_keychain(field_name: &str, value: &str) -> anyhow::Result<()> {
    if keychain_disabled() {
        anyhow::bail!(
            "Keychain disabled — set env var {}={} to persist this credential",
            field_name.to_uppercase(),
            "[REDACTED]"
        );
    }
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, field_name)?;
    entry.set_password(value)?;
    Ok(())
}

/// Delete a secret from the OS keychain.
pub fn delete_from_keychain(field_name: &str) -> anyhow::Result<()> {
    if keychain_disabled() {
        return Ok(());
    }
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, field_name)?;
    entry
        .delete_credential()
        .map_err(|e| anyhow::anyhow!("Failed to delete '{}' from OS keychain: {}", field_name, e))
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
    pub diagnostics: DiagnosticsConfig,
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

fn default_true() -> bool {
    true
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
    /// Retention policies for automatic cleanup of old data.
    #[serde(default)]
    pub retention: RetentionConfig,
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
        Self {
            messages_days: 90,
            superseded_facts_days: 90,
            token_usage_aggregate_days: 30,
            episodes_days: 365,
            behavior_patterns_days: 90,
            goals_days: 180,
            procedures_days: 180,
            error_solutions_days: 90,
        }
    }
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
            context_window: ContextWindowConfig::default(),
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
    20
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

#[derive(Debug, Deserialize, Clone, Default)]
pub struct PathAliasConfig {
    /// Optional alias roots for "projects/..." paths.
    /// Example:
    /// [path_aliases]
    /// projects = ["~/projects"]
    #[serde(default)]
    pub projects: Vec<String>,
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
pub struct WatchdogConfig {
    #[serde(default = "default_watchdog_enabled")]
    pub enabled: bool,
    /// Seconds of no heartbeat before declaring stuck (default: 300 = 5 min).
    #[serde(default = "default_watchdog_stale_secs")]
    pub stale_threshold_secs: u64,
    /// Per-LLM-call timeout in seconds (default: 300 = 5 min).
    #[serde(default = "default_llm_call_timeout_secs")]
    pub llm_call_timeout_secs: u64,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            enabled: default_watchdog_enabled(),
            stale_threshold_secs: default_watchdog_stale_secs(),
            llm_call_timeout_secs: default_llm_call_timeout_secs(),
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

fn default_queue_capacity_approval() -> usize {
    16
}

fn default_queue_capacity_media() -> usize {
    16
}

fn default_queue_capacity_trigger_events() -> usize {
    64
}

fn default_queue_warning_ratio() -> f32 {
    0.75
}

fn default_queue_overload_ratio() -> f32 {
    0.90
}

fn default_queue_adaptive_shedding() -> bool {
    true
}

fn default_queue_fair_trigger_sessions() -> bool {
    true
}

fn default_queue_fair_window_secs() -> u64 {
    60
}

fn default_queue_fair_max_events_per_session() -> u32 {
    4
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct QueueLanePolicyConfig {
    #[serde(default = "default_queue_adaptive_shedding")]
    pub adaptive_shedding: bool,
    #[serde(default = "default_queue_fair_trigger_sessions")]
    pub fair_sessions: bool,
    #[serde(default = "default_queue_fair_window_secs")]
    pub fair_session_window_secs: u64,
    #[serde(default = "default_queue_fair_max_events_per_session")]
    pub fair_max_events_per_session: u32,
}

impl Default for QueueLanePolicyConfig {
    fn default() -> Self {
        Self {
            adaptive_shedding: default_queue_adaptive_shedding(),
            fair_sessions: default_queue_fair_trigger_sessions(),
            fair_session_window_secs: default_queue_fair_window_secs(),
            fair_max_events_per_session: default_queue_fair_max_events_per_session(),
        }
    }
}

impl QueueLanePolicyConfig {
    pub fn normalized(&self) -> Self {
        let mut lane = self.clone();
        lane.fair_session_window_secs = lane.fair_session_window_secs.max(1);
        lane.fair_max_events_per_session = lane.fair_max_events_per_session.max(1);
        lane
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct QueueLanePoliciesConfig {
    #[serde(default)]
    pub approval: QueueLanePolicyConfig,
    #[serde(default)]
    pub media: QueueLanePolicyConfig,
    #[serde(default)]
    pub trigger: QueueLanePolicyConfig,
}

impl QueueLanePoliciesConfig {
    pub fn normalized(&self) -> Self {
        Self {
            approval: self.approval.normalized(),
            media: self.media.normalized(),
            trigger: self.trigger.normalized(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct QueuePolicyConfig {
    #[serde(default = "default_queue_capacity_approval")]
    pub approval_capacity: usize,
    #[serde(default = "default_queue_capacity_media")]
    pub media_capacity: usize,
    #[serde(default = "default_queue_capacity_trigger_events")]
    pub trigger_event_capacity: usize,
    #[serde(default = "default_queue_warning_ratio")]
    pub warning_ratio: f32,
    #[serde(default = "default_queue_overload_ratio")]
    pub overload_ratio: f32,
    #[serde(default)]
    pub lanes: QueueLanePoliciesConfig,

    // Legacy shared overload knobs retained for backward compatibility.
    #[serde(default)]
    pub adaptive_shedding: Option<bool>,
    #[serde(default)]
    pub fair_trigger_sessions: Option<bool>,
    #[serde(default)]
    pub fair_trigger_session_window_secs: Option<u64>,
    #[serde(default)]
    pub fair_trigger_max_events_per_session: Option<u32>,
}

impl Default for QueuePolicyConfig {
    fn default() -> Self {
        Self {
            approval_capacity: default_queue_capacity_approval(),
            media_capacity: default_queue_capacity_media(),
            trigger_event_capacity: default_queue_capacity_trigger_events(),
            warning_ratio: default_queue_warning_ratio(),
            overload_ratio: default_queue_overload_ratio(),
            lanes: QueueLanePoliciesConfig::default(),
            adaptive_shedding: None,
            fair_trigger_sessions: None,
            fair_trigger_session_window_secs: None,
            fair_trigger_max_events_per_session: None,
        }
    }
}

impl QueuePolicyConfig {
    pub fn normalized(&self) -> Self {
        let mut policy = self.clone();
        policy.approval_capacity = policy.approval_capacity.max(1);
        policy.media_capacity = policy.media_capacity.max(1);
        policy.trigger_event_capacity = policy.trigger_event_capacity.max(1);
        policy.warning_ratio = if policy.warning_ratio.is_finite() {
            policy.warning_ratio.clamp(0.0, 1.0)
        } else {
            default_queue_warning_ratio()
        };
        policy.overload_ratio = if policy.overload_ratio.is_finite() {
            policy.overload_ratio.clamp(policy.warning_ratio, 1.0)
        } else {
            default_queue_overload_ratio().clamp(policy.warning_ratio, 1.0)
        };

        let default_lane = QueueLanePolicyConfig::default().normalized();
        let legacy_lane = QueueLanePolicyConfig {
            adaptive_shedding: policy
                .adaptive_shedding
                .unwrap_or(default_queue_adaptive_shedding()),
            fair_sessions: policy
                .fair_trigger_sessions
                .unwrap_or(default_queue_fair_trigger_sessions()),
            fair_session_window_secs: policy
                .fair_trigger_session_window_secs
                .unwrap_or(default_queue_fair_window_secs()),
            fair_max_events_per_session: policy
                .fair_trigger_max_events_per_session
                .unwrap_or(default_queue_fair_max_events_per_session()),
        }
        .normalized();

        policy.lanes = policy.lanes.normalized();
        if policy.lanes.approval == default_lane {
            policy.lanes.approval = legacy_lane.clone();
        }
        if policy.lanes.media == default_lane {
            policy.lanes.media = legacy_lane.clone();
        }
        if policy.lanes.trigger == default_lane {
            policy.lanes.trigger = legacy_lane;
        }

        policy
    }
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
    #[serde(default)]
    pub env: HashMap<String, String>,
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
        }
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
    Soft { threshold: usize, warn_at: usize },
    /// Legacy hard limit behavior (backward compatibility).
    /// Agent will be forced to stop at `cap` iterations.
    Hard { initial: usize, cap: usize },
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
    /// Token budget per task. When exceeded, agent gracefully summarizes and stops.
    /// Defaults to 500,000. Set to 0 to disable (unlimited).
    #[serde(default = "default_task_token_budget")]
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
            task_token_budget: default_task_token_budget(),
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
fn default_task_token_budget() -> Option<u64> {
    Some(500_000)
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
            enabled: true,
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

#[derive(Debug, Deserialize, Clone, Default)]
pub struct UsersConfig {
    #[serde(default)]
    pub owner_ids: HashMap<String, Vec<String>>,
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

#[derive(Debug, Deserialize, Clone, Default)]
pub struct OAuthConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub callback_url: Option<String>,
    #[serde(default)]
    pub providers: HashMap<String, OAuthProviderConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OAuthProviderConfig {
    pub auth_type: String,
    pub authorize_url: String,
    pub token_url: String,
    #[serde(default)]
    pub scopes: Vec<String>,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
}

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

#[derive(Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum HttpAuthType {
    Oauth1a,
    Bearer,
    Header,
    Basic,
}

impl fmt::Debug for HttpAuthType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HttpAuthType::Oauth1a => write!(f, "oauth1a"),
            HttpAuthType::Bearer => write!(f, "bearer"),
            HttpAuthType::Header => write!(f, "header"),
            HttpAuthType::Basic => write!(f, "basic"),
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct HttpAuthProfile {
    pub auth_type: HttpAuthType,
    pub allowed_domains: Vec<String>,
    // OAuth 1.0a fields
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_secret: Option<String>,
    #[serde(default)]
    pub access_token: Option<String>,
    #[serde(default)]
    pub access_token_secret: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    // Bearer token
    #[serde(default)]
    pub token: Option<String>,
    // Header auth
    #[serde(default)]
    pub header_name: Option<String>,
    #[serde(default)]
    pub header_value: Option<String>,
    // Basic auth
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
}

impl fmt::Debug for HttpAuthProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpAuthProfile")
            .field("auth_type", &self.auth_type)
            .field("allowed_domains", &self.allowed_domains)
            .field("api_key", &redact_option(&self.api_key))
            .field("api_secret", &redact_option(&self.api_secret))
            .field("access_token", &redact_option(&self.access_token))
            .field(
                "access_token_secret",
                &redact_option(&self.access_token_secret),
            )
            .field("user_id", &self.user_id)
            .field("token", &redact_option(&self.token))
            .field("header_name", &self.header_name)
            .field("header_value", &redact_option(&self.header_value))
            .field("username", &self.username)
            .field("password", &redact_option(&self.password))
            .finish()
    }
}

impl HttpAuthProfile {
    /// Collect all credential values for this profile (for stripping from error messages).
    pub fn credential_values(&self) -> Vec<&str> {
        let mut vals = Vec::new();
        for v in [
            &self.api_key,
            &self.api_secret,
            &self.access_token,
            &self.access_token_secret,
            &self.token,
            &self.header_value,
            &self.password,
        ]
        .into_iter()
        .flatten()
        {
            if !v.is_empty() {
                vals.push(v.as_str());
            }
        }
        vals
    }
}

/// Context window management configuration.
/// Controls conversation history trimming, summarization, and progressive fact extraction.
#[derive(Debug, Deserialize, Clone)]
pub struct ContextWindowConfig {
    /// Enable context window management (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Default token budget for conversation history (default: 24000).
    #[serde(default = "default_context_budget")]
    pub default_budget: usize,
    /// Per-model token budgets (model name → budget). Overrides default_budget.
    #[serde(default)]
    pub model_budgets: HashMap<String, usize>,
    /// Maximum characters for tool results before truncation (default: 2000).
    #[serde(default = "default_tool_result_chars")]
    pub max_tool_result_chars: usize,
    /// Number of recent messages to always keep in the window (default: 6).
    #[serde(default = "default_summary_window")]
    pub summary_window: usize,
    /// Enable progressive fact extraction after each interaction (default: true).
    #[serde(default = "default_true")]
    pub progressive_facts: bool,
    /// Message count threshold before summarization kicks in (default: 12).
    #[serde(default = "default_summarize_threshold")]
    pub summarize_threshold: usize,
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_budget: default_context_budget(),
            model_budgets: HashMap::new(),
            max_tool_result_chars: default_tool_result_chars(),
            summary_window: default_summary_window(),
            progressive_facts: true,
            summarize_threshold: default_summarize_threshold(),
        }
    }
}

fn default_context_budget() -> usize {
    24000
}
fn default_tool_result_chars() -> usize {
    2000
}
fn default_summary_window() -> usize {
    6
}
fn default_summarize_threshold() -> usize {
    12
}

fn default_heartbeat_tick() -> u64 {
    30
}

fn default_max_concurrent() -> usize {
    3
}

fn default_policy_shadow_mode() -> bool {
    true
}
fn default_policy_enforce() -> bool {
    true
}
fn default_tool_filter_enforce() -> bool {
    true
}
fn default_uncertainty_clarify_enforce() -> bool {
    true
}
fn default_context_refresh_enforce() -> bool {
    true
}
fn default_learning_evidence_gate_enforce() -> bool {
    true
}
fn default_autotune_shadow() -> bool {
    true
}
fn default_autotune_enforce() -> bool {
    true
}
fn default_uncertainty_threshold() -> f32 {
    0.55
}
fn default_write_consistency_max_abs_global_delta() -> u64 {
    3
}
fn default_write_consistency_max_session_mismatch_count() -> u64 {
    0
}
fn default_write_consistency_max_stale_task_starts() -> u64 {
    0
}
fn default_write_consistency_max_missing_message_id_events() -> u64 {
    0
}
fn default_diagnostics_max_events() -> usize {
    200
}

#[derive(Debug, Deserialize, Clone)]
pub struct DiagnosticsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_true")]
    pub record_decision_points: bool,
    #[serde(default = "default_diagnostics_max_events")]
    pub max_events: usize,
    #[serde(default)]
    pub include_raw_tool_args: bool,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            record_decision_points: true,
            max_events: default_diagnostics_max_events(),
            include_raw_tool_args: false,
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

#[derive(Debug, Deserialize, Clone)]
pub struct WriteConsistencyConfig {
    #[serde(default = "default_write_consistency_max_abs_global_delta")]
    pub max_abs_global_delta: u64,
    #[serde(default = "default_write_consistency_max_session_mismatch_count")]
    pub max_session_mismatch_count: u64,
    #[serde(default = "default_write_consistency_max_stale_task_starts")]
    pub max_stale_task_starts: u64,
    #[serde(default = "default_write_consistency_max_missing_message_id_events")]
    pub max_missing_message_id_events: u64,
}

impl Default for WriteConsistencyConfig {
    fn default() -> Self {
        Self {
            max_abs_global_delta: default_write_consistency_max_abs_global_delta(),
            max_session_mismatch_count: default_write_consistency_max_session_mismatch_count(),
            max_stale_task_starts: default_write_consistency_max_stale_task_starts(),
            max_missing_message_id_events: default_write_consistency_max_missing_message_id_events(
            ),
        }
    }
}

impl WriteConsistencyConfig {
    pub fn thresholds(&self) -> crate::events::WriteConsistencyThresholds {
        crate::events::WriteConsistencyThresholds {
            max_abs_global_delta: self.max_abs_global_delta,
            max_session_mismatch_count: self.max_session_mismatch_count,
            max_stale_task_starts: self.max_stale_task_starts,
            max_missing_message_id_events: self.max_missing_message_id_events,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct PolicyConfig {
    #[serde(default = "default_policy_shadow_mode")]
    pub policy_shadow_mode: bool,
    /// Enforce policy-derived context budget caps. When false, message trimming
    /// only respects the model provider budget.
    #[serde(default = "default_policy_enforce")]
    pub policy_enforce: bool,
    #[serde(default = "default_tool_filter_enforce")]
    pub tool_filter_enforce: bool,
    #[serde(default = "default_uncertainty_clarify_enforce")]
    pub uncertainty_clarify_enforce: bool,
    #[serde(default = "default_context_refresh_enforce")]
    pub context_refresh_enforce: bool,
    #[serde(default = "default_learning_evidence_gate_enforce")]
    pub learning_evidence_gate_enforce: bool,
    #[serde(default = "default_autotune_shadow")]
    pub autotune_shadow: bool,
    #[serde(default = "default_autotune_enforce")]
    pub autotune_enforce: bool,
    #[serde(default = "default_uncertainty_threshold")]
    pub uncertainty_clarify_threshold: f32,
    #[serde(default)]
    pub write_consistency: WriteConsistencyConfig,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            policy_shadow_mode: default_policy_shadow_mode(),
            policy_enforce: default_policy_enforce(),
            tool_filter_enforce: default_tool_filter_enforce(),
            uncertainty_clarify_enforce: default_uncertainty_clarify_enforce(),
            context_refresh_enforce: default_context_refresh_enforce(),
            learning_evidence_gate_enforce: default_learning_evidence_gate_enforce(),
            autotune_shadow: default_autotune_shadow(),
            autotune_enforce: default_autotune_enforce(),
            uncertainty_clarify_threshold: default_uncertainty_threshold(),
            write_consistency: WriteConsistencyConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let expanded = expand_env_vars(&content)?;
        let mut config: AppConfig = toml::from_str(&expanded)?;
        config.provider.models.apply_defaults(&config.provider.kind);
        config.daemon.queue_policy = config.daemon.queue_policy.normalized();
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
                let key = if i == 0 {
                    "bot_token".to_string()
                } else {
                    format!("telegram_bot_token_{}", i)
                };
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
                    let key = if i == 0 {
                        "discord_bot_token".to_string()
                    } else {
                        format!("discord_bot_token_{}", i)
                    };
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
                    let key = if i == 0 {
                        "slack_app_token".to_string()
                    } else {
                        format!("slack_app_token_{}", i)
                    };
                    bot.app_token = resolve_from_keychain(&key)?;
                }
                if bot.bot_token == "keychain" {
                    let key = if i == 0 {
                        "slack_bot_token".to_string()
                    } else {
                        format!("slack_bot_token_{}", i)
                    };
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
                        let keychain_key = format!("http_auth_{}_{}", name, field);
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
    fn task_timeout_default_is_30_minutes() {
        let config = SubagentsConfig::default();
        assert_eq!(config.task_timeout_secs, Some(1800));
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
}
