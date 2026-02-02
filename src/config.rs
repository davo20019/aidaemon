use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub provider: ProviderConfig,
    pub telegram: TelegramConfig,
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
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    #[serde(default)]
    pub kind: ProviderKind,
    pub api_key: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub models: ModelsConfig,
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
    #[serde(default = "default_primary_model")]
    pub primary: String,
    #[serde(default = "default_fast_model")]
    pub fast: String,
    #[serde(default = "default_smart_model")]
    pub smart: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            primary: default_primary_model(),
            fast: default_fast_model(),
            smart: default_smart_model(),
        }
    }
}

fn default_primary_model() -> String {
    "openai/gpt-4o".to_string()
}
fn default_fast_model() -> String {
    "openai/gpt-4o-mini".to_string()
}
fn default_smart_model() -> String {
    "openai/gpt-4o".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct TelegramConfig {
    pub bot_token: String,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StateConfig {
    #[serde(default = "default_db_path")]
    pub db_path: String,
    #[serde(default = "default_working_memory_cap")]
    pub working_memory_cap: usize,
    #[serde(default = "default_consolidation_interval_hours")]
    pub consolidation_interval_hours: u64,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            db_path: default_db_path(),
            working_memory_cap: default_working_memory_cap(),
            consolidation_interval_hours: default_consolidation_interval_hours(),
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

#[derive(Debug, Deserialize, Clone)]
pub struct TerminalConfig {
    #[serde(default = "default_allowed_prefixes")]
    pub allowed_prefixes: Vec<String>,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            allowed_prefixes: default_allowed_prefixes(),
        }
    }
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
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            health_port: default_health_port(),
        }
    }
}

fn default_health_port() -> u16 {
    8080
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TriggersConfig {
    pub email: Option<EmailTriggerConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EmailTriggerConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    #[serde(default = "default_folder")]
    pub folder: String,
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
fn default_subagents_max_response_chars() -> usize {
    8000
}
fn default_subagents_timeout_secs() -> u64 {
    300
}

impl AppConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    }
}
