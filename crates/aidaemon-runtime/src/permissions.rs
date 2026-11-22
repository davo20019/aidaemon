//! AIdaemon's permission level and each coding agent's: the owner's choices,
//! kept in `config.toml` as `[authority] owner_default` and
//! `[cli_agents.permissions]`.
//!
//! The file is their only store. A [`Permissions`] handle is their live
//! value: read from the file at start and refreshed from it whenever the
//! daemon writes one ([`save_level`], [`save_agent`], or any Settings change
//! through [`Permissions::reload`]), so a change takes effect at once and
//! nothing else keeps a copy.
//!
//! The level is the most anything AIdaemon runs may do:
//! - its own turns, and every sub-agent and specialist they start, through
//!   [`crate::authority`] (a child never gets more than its parent);
//! - the terminal: [`OwnerLevel::FullAccess`] runs every command unasked
//!   ([`Permissions::trust_all`]);
//! - each coding agent, capped at [`agent_cap`]: an agent can follow the
//!   level ("same as AIdaemon") or be stricter, never looser.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use aidaemon_coding_agents::{CodingAgent, PermissionMode};
use aidaemon_tools::config_file::ConfigFile;

pub use crate::authority::OwnerLevel;

/// What the owner chose for one coding agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentChoice {
    /// The most AIdaemon's level lets it do.
    Same,
    /// This mode, or the most the level allows when it allows less.
    Mode(PermissionMode),
}

impl AgentChoice {
    /// Its name in `config.toml`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Same => "same",
            Self::Mode(mode) => mode.as_str(),
        }
    }

    /// The choice a config value names.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "same" => Some(Self::Same),
            other => PermissionMode::parse(other).map(Self::Mode),
        }
    }
}

/// The most a coding agent may do under `level`: Read only lets it act
/// only through AIdaemon's own checks of every step, Standard lets it work
/// in its sandbox, Full access lets it do anything.
pub fn agent_cap(level: OwnerLevel) -> PermissionMode {
    match level {
        OwnerLevel::ReadOnly => PermissionMode::EveryCommand,
        OwnerLevel::Full => PermissionMode::Checked,
        OwnerLevel::FullAccess | OwnerLevel::Unrestricted => PermissionMode::FullAccess,
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct State {
    level: OwnerLevel,
    agents: HashMap<String, AgentChoice>,
}

/// The live permission choices. Clones share one value.
#[derive(Clone, Default)]
pub struct Permissions(Arc<RwLock<State>>);

impl std::fmt::Debug for Permissions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.read();
        f.debug_struct("Permissions")
            .field("level", &state.level)
            .field("agents", &state.agents)
            .finish()
    }
}

/// Two handles are equal when they share one value.
impl PartialEq for Permissions {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Permissions {}

impl Permissions {
    /// A value of its own at `level`, with no agent choices.
    pub fn new(level: OwnerLevel) -> Self {
        let permissions = Self::default();
        permissions.set_level(level);
        permissions
    }

    /// The choices a parsed `config.toml` holds.
    pub fn from_config(config: &toml::Table) -> Self {
        let permissions = Self::default();
        permissions.reload(config);
        permissions
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, State> {
        self.0
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, State> {
        self.0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Take the choices `config` (the file as written) holds.
    pub fn reload(&self, config: &toml::Table) {
        let level = config_level(config);
        let mut agents = HashMap::new();
        if let Some(table) = config
            .get("cli_agents")
            .and_then(|table| table.get("permissions"))
            .and_then(toml::Value::as_table)
        {
            for (name, value) in table {
                match value.as_str().and_then(AgentChoice::parse) {
                    Some(choice) => {
                        agents.insert(name.clone(), choice);
                    }
                    None => tracing::warn!(
                        agent = %name,
                        "[cli_agents.permissions] has a value this daemon does not know; \
                         the agent keeps its default"
                    ),
                }
            }
        }
        *self.write() = State { level, agents };
    }

    /// AIdaemon's level.
    pub fn level(&self) -> OwnerLevel {
        self.read().level
    }

    /// Set the level in this value only (the file is [`save_level`]'s).
    pub fn set_level(&self, level: OwnerLevel) {
        self.write().level = level;
    }

    /// Every terminal command runs unasked.
    pub fn trust_all(&self) -> bool {
        self.level() >= OwnerLevel::FullAccess
    }

    /// The owner's choice for the agent named `name` (`None`: its kind's
    /// default).
    pub fn choice(&self, name: &str) -> Option<AgentChoice> {
        self.read().agents.get(name).copied()
    }

    /// The mode the agent named `name` runs in now; `None`: it may not run
    /// delegated work (none of its modes fit under the level).
    pub fn agent_mode(&self, name: &str, agent: &dyn CodingAgent) -> Option<PermissionMode> {
        let state = self.read();
        let cap = agent_cap(state.level);
        match state.agents.get(name) {
            Some(AgentChoice::Same) => aidaemon_coding_agents::mode_within(agent, None, true, cap),
            Some(AgentChoice::Mode(mode)) => {
                aidaemon_coding_agents::mode_within(agent, Some(*mode), false, cap)
            }
            None => aidaemon_coding_agents::mode_within(agent, None, false, cap),
        }
    }

    /// Whether the owner may pick `mode` for `agent` now: one it offers,
    /// within the level. `Err`: why not, in plain words.
    pub fn may_pick(&self, agent: &dyn CodingAgent, mode: PermissionMode) -> Result<(), String> {
        if !agent.modes().contains(&mode) {
            let why = agent.note();
            return Err(if why.is_empty() {
                format!("{} can't run in that mode.", agent.name())
            } else {
                format!("{} can't run in that mode: {why}", agent.name())
            });
        }
        let level = self.level();
        if mode > agent_cap(level) {
            return Err(format!(
                "AIdaemon itself is on {}, and no coding agent may do more than AIdaemon. \
                 Raise AIdaemon's permission level first.",
                level.title()
            ));
        }
        Ok(())
    }
}

/// The level a parsed `config.toml` sets (`full` when it sets none or one
/// this daemon cannot read: the default, never wider).
fn config_level(config: &toml::Table) -> OwnerLevel {
    let Some(value) = config
        .get("authority")
        .and_then(|table| table.get("owner_default"))
    else {
        return OwnerLevel::default();
    };
    value.clone().try_into().unwrap_or_else(|error| {
        tracing::warn!(%error, "[authority] owner_default is not a level; using Standard");
        OwnerLevel::default()
    })
}

/// Change `config.toml` with `edit` (checked, backed up and written), then
/// take what the file now holds.
async fn edit_config(
    config: &ConfigFile,
    permissions: &Permissions,
    edit: impl FnOnce(&mut toml_edit::DocumentMut) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let text = config.read().await?;
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    edit(&mut doc)?;
    config.save(&doc.to_string()).await?;
    permissions.reload(&config.load_table().await?);
    Ok(())
}

/// `table` in `doc` at `path`, made (as a plain table) when missing.
fn table_at<'d>(
    doc: &'d mut toml_edit::DocumentMut,
    path: &[&str],
) -> anyhow::Result<&'d mut toml_edit::Table> {
    let mut table = doc.as_table_mut();
    for key in path {
        let item = table
            .entry(key)
            .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
        table = item
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`{key}` in config.toml is not a table"))?;
    }
    Ok(table)
}

/// Save AIdaemon's level to `config.toml` and take it.
pub async fn save_level(
    config: &ConfigFile,
    permissions: &Permissions,
    level: OwnerLevel,
) -> anyhow::Result<()> {
    edit_config(config, permissions, |doc| {
        table_at(doc, &["authority"])?.insert("owner_default", toml_edit::value(level.as_str()));
        Ok(())
    })
    .await
}

/// Save the owner's choice for the agent named `name` (`None`: back to its
/// kind's default) and take it.
pub async fn save_agent(
    config: &ConfigFile,
    permissions: &Permissions,
    name: &str,
    choice: Option<AgentChoice>,
) -> anyhow::Result<()> {
    edit_config(config, permissions, |doc| {
        let table = table_at(doc, &["cli_agents", "permissions"])?;
        match choice {
            Some(choice) => {
                table.insert(name, toml_edit::value(choice.as_str()));
            }
            None => {
                table.remove(name);
            }
        }
        Ok(())
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_coding_agents::{claude::ClaudeCode, plain::PlainAgent};

    fn config(text: &str) -> toml::Table {
        text.parse().unwrap()
    }

    #[test]
    fn the_level_and_agent_choices_come_from_the_file() {
        let permissions = Permissions::from_config(&config(
            "[authority]\nowner_default = \"full_access\"\n\
             [cli_agents.permissions]\nclaude = \"same\"\ncodex = \"every_command\"\n\
             other = \"sideways\"\n",
        ));
        assert_eq!(permissions.level(), OwnerLevel::FullAccess);
        assert!(permissions.trust_all());
        assert_eq!(permissions.choice("claude"), Some(AgentChoice::Same));
        assert_eq!(
            permissions.choice("codex"),
            Some(AgentChoice::Mode(PermissionMode::EveryCommand))
        );
        // An unknown value is no choice.
        assert_eq!(permissions.choice("other"), None);

        // Standard is the default, under either name.
        assert_eq!(
            Permissions::from_config(&config("")).level(),
            OwnerLevel::Full
        );
        let standard = config("[authority]\nowner_default = \"standard\"\n");
        assert_eq!(
            Permissions::from_config(&standard).level(),
            OwnerLevel::Full
        );
        let unknown = config("[authority]\nowner_default = \"everything\"\n");
        assert_eq!(Permissions::from_config(&unknown).level(), OwnerLevel::Full);
        assert!(!Permissions::from_config(&unknown).trust_all());
    }

    /// No coding agent runs above AIdaemon's level, whatever was picked;
    /// "same" follows the level.
    #[test]
    fn an_agent_never_runs_above_aidaemons_level() {
        let permissions = Permissions::from_config(&config(
            "[cli_agents.permissions]\nclaude = \"full_access\"\nsame = \"same\"\n",
        ));
        let claude = &ClaudeCode;
        // Standard: full access is capped to checked.
        assert_eq!(
            permissions.agent_mode("claude", claude),
            Some(PermissionMode::Checked)
        );
        assert_eq!(
            permissions.agent_mode("same", claude),
            Some(PermissionMode::Checked)
        );
        assert!(
            permissions
                .may_pick(claude, PermissionMode::FullAccess)
                .unwrap_err()
                .contains("Standard")
        );
        assert!(
            permissions
                .may_pick(claude, PermissionMode::Checked)
                .is_ok()
        );

        permissions.set_level(OwnerLevel::FullAccess);
        assert_eq!(
            permissions.agent_mode("claude", claude),
            Some(PermissionMode::FullAccess)
        );
        assert_eq!(
            permissions.agent_mode("same", claude),
            Some(PermissionMode::FullAccess)
        );
        // An agent with no choice keeps its kind's default: raising the
        // level never widens an agent nobody set.
        assert_eq!(
            permissions.agent_mode("unset", claude),
            Some(PermissionMode::Checked)
        );

        permissions.set_level(OwnerLevel::ReadOnly);
        assert_eq!(
            permissions.agent_mode("claude", claude),
            Some(PermissionMode::EveryCommand)
        );
        assert_eq!(permissions.agent_mode("unset", claude), None);
        let plain = PlainAgent::unknown("synthetic-coder");
        assert_eq!(permissions.agent_mode("synthetic-coder", &plain), None);
    }

    #[tokio::test]
    async fn saving_writes_the_file_and_takes_effect_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "# the owner's notes\n[authority]\ntrusted_roots = []\n",
        )
        .unwrap();
        let file = ConfigFile::new(
            &path,
            Arc::new(aidaemon_tools::config_file::toml_only)
                as Arc<dyn aidaemon_tools::config_file::ConfigCheck>,
        );
        let permissions = Permissions::from_config(&file.load_table().await.unwrap());
        save_level(&file, &permissions, OwnerLevel::FullAccess)
            .await
            .unwrap();
        assert!(permissions.trust_all());
        save_agent(&file, &permissions, "claude", Some(AgentChoice::Same))
            .await
            .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# the owner's notes"), "{text}");
        assert!(text.contains("owner_default = \"full_access\""), "{text}");
        assert!(text.contains("claude = \"same\""), "{text}");
        // A restart reads the same.
        let again = Permissions::from_config(&file.load_table().await.unwrap());
        assert_eq!(again.level(), OwnerLevel::FullAccess);
        assert_eq!(again.choice("claude"), Some(AgentChoice::Same));

        save_agent(&file, &permissions, "claude", None)
            .await
            .unwrap();
        save_level(&file, &permissions, OwnerLevel::Full)
            .await
            .unwrap();
        assert!(!permissions.trust_all());
        assert_eq!(permissions.choice("claude"), None);
    }
}
