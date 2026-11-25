//! AIdaemon's permission level and each coding agent's: the owner's choices,
//! kept in `config.toml` as `[authority] owner_default` and
//! `[cli_agents.permissions]`. The model the owner chose for a coding agent
//! (`[cli_agents.models]`) is kept and refreshed the same way.
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
use std::sync::{Arc, OnceLock, RwLock};

use aidaemon_coding_agents::{CodingAgent, PermissionMode, SignIn, SignInMethod};
use aidaemon_secrets::SecretStore;
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
    /// The model chosen per agent, by name (absent: its own default).
    models: HashMap<String, String>,
    /// How each agent signs in, by name (absent: its own sign-in).
    sign_ins: HashMap<String, SignInMethod>,
}

/// The live permission choices. Clones share one value.
#[derive(Clone, Default)]
pub struct Permissions(Arc<RwLock<State>>, Arc<OnceLock<Arc<dyn SecretStore>>>);

impl std::fmt::Debug for Permissions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.read();
        f.debug_struct("Permissions")
            .field("level", &state.level)
            .field("agents", &state.agents)
            .field("models", &state.models)
            .field("sign_ins", &state.sign_ins)
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
        let models = config
            .get("cli_agents")
            .and_then(|table| table.get("models"))
            .and_then(toml::Value::as_table)
            .into_iter()
            .flatten()
            .filter_map(|(name, value)| {
                let model = value.as_str()?.trim();
                (!model.is_empty()).then(|| (name.clone(), model.to_string()))
            })
            .collect();
        let sign_ins = config
            .get("cli_agents")
            .and_then(|table| table.get("sign_in"))
            .and_then(toml::Value::as_table)
            .into_iter()
            .flatten()
            .filter_map(|(name, value)| {
                let method = value.as_str().and_then(SignInMethod::parse);
                if method.is_none() {
                    tracing::warn!(
                        agent = %name,
                        "[cli_agents.sign_in] has a value this daemon does not know; \
                         the agent keeps its own sign-in"
                    );
                }
                Some((name.clone(), method?))
            })
            .collect();
        *self.write() = State {
            level,
            agents,
            models,
            sign_ins,
        };
    }

    /// Where the keys an agent signs in with are kept (set once, at start;
    /// until then a key the owner chose can't be read, so such an agent
    /// can't run).
    pub fn set_secrets(&self, secrets: Arc<dyn SecretStore>) {
        let _ = self.1.set(secrets);
    }

    /// How the owner chose the agent named `name` signs in.
    pub fn sign_in_method(&self, name: &str) -> SignInMethod {
        self.read().sign_ins.get(name).copied().unwrap_or_default()
    }

    /// Whether a key is saved for the agent named `name` to sign in with.
    pub fn has_sign_in_key(&self, name: &str) -> bool {
        self.1.get().is_some_and(|secrets| {
            secrets.contains(&aidaemon_secrets::names::cli_agent_sign_in_key(name))
        })
    }

    /// The sign-in the agent named `name` runs with, its key read now
    /// (`Err`: the owner chose a key and none is saved, in words for the
    /// owner).
    pub fn sign_in_of(&self, name: &str) -> Result<SignIn, String> {
        match self.sign_in_method(name) {
            SignInMethod::Own => Ok(SignIn::Own),
            SignInMethod::GeminiApiKey => self
                .1
                .get()
                .and_then(|secrets| {
                    secrets
                        .get(&aidaemon_secrets::names::cli_agent_sign_in_key(name))
                        .ok()
                })
                .map(|key| SignIn::GeminiApiKey(key.expose().to_string()))
                .ok_or_else(|| {
                    "it is set to sign in with a Gemini API key, and none is saved. Add one in \
                     Settings (Coding agents), or switch it back to its own sign-in."
                        .to_string()
                }),
        }
    }

    /// The model the owner chose for the agent named `name` (`None`: the
    /// agent's own default).
    pub fn model_of(&self, name: &str) -> Option<String> {
        self.read().models.get(name).cloned()
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

/// Save how the agent named `name` signs in ([`SignInMethod::Own`]: back
/// to its own sign-in) and take it.
pub async fn save_agent_sign_in(
    config: &ConfigFile,
    permissions: &Permissions,
    name: &str,
    method: SignInMethod,
) -> anyhow::Result<()> {
    edit_config(config, permissions, |doc| {
        let table = table_at(doc, &["cli_agents", "sign_in"])?;
        match method {
            SignInMethod::Own => {
                table.remove(name);
            }
            other => {
                table.insert(name, toml_edit::value(other.as_str()));
            }
        }
        Ok(())
    })
    .await
}

/// Save the model chosen for the agent named `name` (`None`: back to its
/// own default) and take it.
pub async fn save_agent_model(
    config: &ConfigFile,
    permissions: &Permissions,
    name: &str,
    model: Option<&str>,
) -> anyhow::Result<()> {
    edit_config(config, permissions, |doc| {
        let table = table_at(doc, &["cli_agents", "models"])?;
        match model {
            Some(model) => {
                table.insert(name, toml_edit::value(model));
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

        // A model chosen for an agent, then its own default again.
        save_agent_model(&file, &permissions, "agy", Some("synthetic-large-high"))
            .await
            .unwrap();
        assert_eq!(
            permissions.model_of("agy").as_deref(),
            Some("synthetic-large-high")
        );
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("agy = \"synthetic-large-high\""), "{text}");
        let again = Permissions::from_config(&file.load_table().await.unwrap());
        assert_eq!(
            again.model_of("agy").as_deref(),
            Some("synthetic-large-high")
        );
        save_agent_model(&file, &permissions, "agy", None)
            .await
            .unwrap();
        assert_eq!(permissions.model_of("agy"), None);
    }

    #[test]
    fn a_sign_in_choice_reads_its_key_from_the_secret_store() {
        let permissions = Permissions::from_config(&config(
            "[cli_agents.sign_in]\nagy = \"gemini_api_key\"\nother = \"unknown\"\n",
        ));
        assert_eq!(
            permissions.sign_in_method("agy"),
            SignInMethod::GeminiApiKey
        );
        assert_eq!(permissions.sign_in_method("other"), SignInMethod::Own);
        assert_eq!(permissions.sign_in_of("other"), Ok(SignIn::Own));
        // Chosen, and no key saved: it can't run, and says how to fix it.
        assert!(permissions.sign_in_of("agy").is_err());
        assert!(!permissions.has_sign_in_key("agy"));
        let secrets = Arc::new(aidaemon_secrets::MemorySecretStore::default());
        secrets
            .set(
                &aidaemon_secrets::names::cli_agent_sign_in_key("agy"),
                "synthetic-gemini-key-0001",
            )
            .unwrap();
        permissions.set_secrets(secrets);
        assert!(permissions.has_sign_in_key("agy"));
        assert_eq!(
            permissions.sign_in_of("agy"),
            Ok(SignIn::GeminiApiKey("synthetic-gemini-key-0001".into()))
        );
    }
}
