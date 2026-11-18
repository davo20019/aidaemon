//! How much a coding agent the daemon delegates to may do on its own, set
//! by the owner per agent (Settings, Coding agents).
//!
//! - [`AgentMode::Checked`]: confined to the project folder by the agent's
//!   own sandbox; whatever lies beyond it is asked first, and each request
//!   is judged by the run as the model's own calls are (`cli_host`,
//!   `codex_host`). The default for an agent the daemon can host.
//! - [`AgentMode::EveryCommand`]: on the machine itself, every command and
//!   file change (but the agent's own short list of read-only commands)
//!   asked first and judged the same way.
//! - [`AgentMode::FullAccess`]: on the machine itself, nothing asked; the
//!   run's safeguards never see what it does. The owner's choice, made
//!   with an approval.
//!
//! Checking needs an agent whose approval requests the daemon can answer
//! (Claude Code and Codex today). Any other agent offers only full access,
//! and runs no delegated work until the owner picks it.

use std::sync::{Arc, Mutex, OnceLock};

use aidaemon_tools::services::SettingsStore;

/// A coding agent's permission mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    Checked,
    EveryCommand,
    FullAccess,
}

impl AgentMode {
    /// Its stored name (a machine format of the daemon's own).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Checked => "checked",
            Self::EveryCommand => "every_command",
            Self::FullAccess => "full_access",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "checked" => Some(Self::Checked),
            "every_command" => Some(Self::EveryCommand),
            "full_access" => Some(Self::FullAccess),
            _ => None,
        }
    }

    /// Whether the agent's requests reach the run (it must be hosted).
    pub fn is_checked(self) -> bool {
        !matches!(self, Self::FullAccess)
    }
}

/// The program an agent launches, as far as its modes go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    ClaudeCode,
    Codex,
    /// One whose approval requests the daemon cannot answer.
    Other,
}

impl AgentKind {
    /// From the file name of the program an entry launches (`claude`,
    /// `codex`; a machine format: an executable's name).
    pub fn of_program(file_name: &str) -> Self {
        let name = file_name.to_ascii_lowercase();
        match name.trim_end_matches(".exe") {
            "claude" => Self::ClaudeCode,
            "codex" => Self::Codex,
            _ => Self::Other,
        }
    }

    /// Its name in plain words.
    pub fn program(self, fallback: &str) -> String {
        match self {
            Self::ClaudeCode => "Claude Code".to_string(),
            Self::Codex => "Codex".to_string(),
            Self::Other => fallback.to_string(),
        }
    }

    /// The modes it can run in, safest first.
    pub fn offered(self) -> &'static [AgentMode] {
        match self {
            Self::ClaudeCode | Self::Codex => &[
                AgentMode::Checked,
                AgentMode::EveryCommand,
                AgentMode::FullAccess,
            ],
            Self::Other => &[AgentMode::FullAccess],
        }
    }

    /// Its mode when the owner chose none: checked when it can be.
    pub fn default_mode(self) -> Option<AgentMode> {
        match self {
            Self::ClaudeCode | Self::Codex => Some(AgentMode::Checked),
            Self::Other => None,
        }
    }

    /// Why some modes are not offered; empty when all are.
    pub fn note(self) -> &'static str {
        match self {
            Self::ClaudeCode | Self::Codex => "",
            Self::Other => {
                "AIDaemon can't check this agent's actions, so it can only run with full access."
            }
        }
    }
}

fn setting_key(agent: &str) -> String {
    format!("cli_agent_mode:{agent}")
}

/// The mode `agent` runs in: the owner's choice when it is one its kind
/// offers, else its kind's default.
pub async fn mode_of(
    store: &(impl SettingsStore + ?Sized),
    agent: &str,
    kind: AgentKind,
) -> Option<AgentMode> {
    let stored = store
        .get_setting(&setting_key(agent))
        .await
        .ok()
        .flatten()
        .and_then(|value| AgentMode::parse(&value))
        .filter(|mode| kind.offered().contains(mode));
    stored.or_else(|| kind.default_mode())
}

/// Store the owner's choice of `mode` for `agent`. Refused when its kind
/// does not offer it.
pub async fn set_mode(
    store: &(impl SettingsStore + ?Sized),
    agent: &str,
    kind: AgentKind,
    mode: AgentMode,
) -> Result<(), String> {
    if !kind.offered().contains(&mode) {
        return Err(format!(
            "{} can't run in that mode: {}",
            kind.program(agent),
            kind.note()
        ));
    }
    store
        .set_setting(&setting_key(agent), mode.as_str())
        .await
        .map_err(|error| format!("The mode could not be saved: {error}"))
}

/// One agent the daemon can delegate to, as the Settings page shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentInfo {
    pub name: String,
    pub kind: AgentKind,
}

/// Lists one `cli_agent` tool's agents; empty once the tool is gone.
pub type Lister = Arc<dyn Fn() -> Vec<AgentInfo> + Send + Sync>;

fn registry() -> &'static Mutex<Vec<Lister>> {
    static REGISTRY: OnceLock<Mutex<Vec<Lister>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Add a tool's agents to the ones listed. Every live tool is read (the
/// daemon builds one; tests build many), so one never hides another's.
pub fn publish(lister: Lister) {
    if let Ok(mut registry) = registry().lock() {
        registry.push(lister);
    }
}

/// The agents the daemon can delegate to now, by name, each once.
pub fn agents() -> Vec<AgentInfo> {
    let listers = registry()
        .lock()
        .map(|registry| registry.clone())
        .unwrap_or_default();
    let mut agents: Vec<AgentInfo> = Vec::new();
    for lister in listers {
        for info in lister() {
            if !agents.iter().any(|known| known.name == info.name) {
                agents.push(info);
            }
        }
    }
    agents.sort_by(|a, b| a.name.cmp(&b.name));
    agents
}

/// The kind of the agent named `agent` now, when there is one.
pub fn kind_of(agent: &str) -> Option<AgentKind> {
    agents()
        .into_iter()
        .find(|info| info.name == agent)
        .map(|info| info.kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Settings(std::sync::Mutex<std::collections::HashMap<String, String>>);

    #[async_trait::async_trait]
    impl SettingsStore for Settings {
        async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_string(), value.to_string());
            Ok(())
        }
    }

    #[tokio::test]
    async fn an_agent_the_daemon_can_host_is_checked_until_the_owner_says_otherwise() {
        let store = Settings::default();
        assert_eq!(
            mode_of(&store, "codex", AgentKind::Codex).await,
            Some(AgentMode::Checked)
        );
        set_mode(&store, "codex", AgentKind::Codex, AgentMode::FullAccess)
            .await
            .unwrap();
        assert_eq!(
            mode_of(&store, "codex", AgentKind::Codex).await,
            Some(AgentMode::FullAccess)
        );
        // Each agent has its own.
        assert_eq!(
            mode_of(&store, "claude", AgentKind::ClaudeCode).await,
            Some(AgentMode::Checked)
        );
    }

    #[tokio::test]
    async fn an_agent_it_cannot_check_runs_nothing_until_given_full_access() {
        let store = Settings::default();
        assert_eq!(mode_of(&store, "aider", AgentKind::Other).await, None);
        assert!(
            set_mode(&store, "aider", AgentKind::Other, AgentMode::Checked)
                .await
                .is_err()
        );
        set_mode(&store, "aider", AgentKind::Other, AgentMode::FullAccess)
            .await
            .unwrap();
        assert_eq!(
            mode_of(&store, "aider", AgentKind::Other).await,
            Some(AgentMode::FullAccess)
        );
    }

    #[tokio::test]
    async fn a_stored_mode_its_kind_does_not_offer_is_ignored() {
        let store = Settings::default();
        store
            .set_setting(&setting_key("aider"), "checked")
            .await
            .unwrap();
        assert_eq!(mode_of(&store, "aider", AgentKind::Other).await, None);
    }

    #[test]
    fn the_kind_is_read_from_the_program_name() {
        assert_eq!(AgentKind::of_program("claude"), AgentKind::ClaudeCode);
        assert_eq!(AgentKind::of_program("codex.exe"), AgentKind::Codex);
        assert_eq!(AgentKind::of_program("aider"), AgentKind::Other);
    }
}
