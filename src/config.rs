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
    pub telegram: TelegramConfig,
    #[cfg(feature = "discord")]
    #[serde(default)]
    pub discord: DiscordConfig,
    #[cfg(feature = "slack")]
    #[serde(default)]
    pub slack: SlackConfig,
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

#[derive(Debug, Deserialize, Clone)]
pub struct ModelsConfig {
    #[serde(default)]
    pub primary: String,
    #[serde(default)]
    pub fast: String,
    #[serde(default)]
    pub smart: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            primary: String::new(),
            fast: String::new(),
            smart: String::new(),
        }
    }
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
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            dir: default_skills_dir(),
            enabled: default_skills_enabled(),
        }
    }
}

fn default_skills_dir() -> String {
    "skills".to_string()
}

fn default_skills_enabled() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone)]
pub struct SubagentsConfig {
    #[serde(default = "default_subagents_enabled")]
    pub enabled: bool,
    #[serde(default = "default_subagents_max_depth")]
    pub max_depth: usize,
    #[serde(default = "default_subagents_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_subagents_max_iterations_cap")]
    pub max_iterations_cap: usize,
    #[serde(default = "default_subagents_max_response_chars")]
    pub max_response_chars: usize,
    #[serde(default = "default_subagents_timeout_secs")]
    pub timeout_secs: u64,
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

    /// Resolve fields set to `"keychain"` by reading them from the OS credential store.
    fn resolve_secrets(&mut self) -> anyhow::Result<()> {
        if self.provider.api_key == "keychain" {
            self.provider.api_key = resolve_from_keychain("api_key")?;
        }
        if self.telegram.bot_token == "keychain" {
            self.telegram.bot_token = resolve_from_keychain("bot_token")?;
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
        if self.discord.bot_token == "keychain" {
            self.discord.bot_token = resolve_from_keychain("discord_bot_token")?;
        }
        #[cfg(feature = "slack")]
        {
            if self.slack.app_token == "keychain" {
                self.slack.app_token = resolve_from_keychain("slack_app_token")?;
            }
            if self.slack.bot_token == "keychain" {
                self.slack.bot_token = resolve_from_keychain("slack_bot_token")?;
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
}
