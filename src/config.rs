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
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub models: ModelsConfig,
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
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            db_path: default_db_path(),
            working_memory_cap: default_working_memory_cap(),
        }
    }
}

fn default_db_path() -> String {
    "aidaemon.db".to_string()
}
fn default_working_memory_cap() -> usize {
    50
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

impl AppConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    }
}
