//! How much a coding agent may do on its own, set per agent by its owner.

use async_trait::async_trait;

use crate::CodingAgent;

/// How much a coding agent may do on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PermissionMode {
    /// Confined to the project folder by the agent's own sandbox; anything
    /// beyond it is asked first, and each request is judged.
    Checked,
    /// On the machine itself, with every command and file change (but the
    /// agent's own short list of read-only commands) asked first.
    EveryCommand,
    /// On the machine itself, nothing asked: what it does is never judged.
    FullAccess,
}

impl PermissionMode {
    /// Its stored name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Checked => "checked",
            Self::EveryCommand => "every_command",
            Self::FullAccess => "full_access",
        }
    }

    /// The mode a stored name names.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "checked" => Some(Self::Checked),
            "every_command" => Some(Self::EveryCommand),
            "full_access" => Some(Self::FullAccess),
            _ => None,
        }
    }
}

/// Where the owner's choices are kept: any key-value store of the embedding
/// agent.
#[async_trait]
pub trait ModeStore: Send + Sync {
    /// The value stored under `key`, if any.
    async fn load(&self, key: &str) -> Option<String>;
    /// Store `value` under `key`.
    async fn save(&self, key: &str, value: &str) -> Result<(), String>;
}

fn key(agent: &str) -> String {
    format!("cli_agent_mode:{agent}")
}

/// The mode the agent named `name` runs in: the owner's choice when its
/// kind offers it, else its kind's default (`None`: it may not run
/// delegated work until the owner picks one).
pub async fn mode_of(
    store: &(impl ModeStore + ?Sized),
    name: &str,
    agent: &dyn CodingAgent,
) -> Option<PermissionMode> {
    store
        .load(&key(name))
        .await
        .and_then(|value| PermissionMode::parse(&value))
        .filter(|mode| agent.modes().contains(mode))
        .or_else(|| agent.default_mode())
}

/// Keep the owner's choice of `mode` for the agent named `name`; refused
/// when its kind does not offer it.
pub async fn set_mode(
    store: &(impl ModeStore + ?Sized),
    name: &str,
    agent: &dyn CodingAgent,
    mode: PermissionMode,
) -> Result<(), String> {
    if !agent.modes().contains(&mode) {
        let why = agent.note();
        return Err(if why.is_empty() {
            format!("{} can't run in that mode.", agent.name())
        } else {
            format!("{} can't run in that mode: {why}", agent.name())
        });
    }
    store.save(&key(name), mode.as_str()).await
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;
    use crate::{claude::ClaudeCode, plain::PlainAgent};

    #[derive(Default)]
    struct Memory(Mutex<HashMap<String, String>>);

    #[async_trait]
    impl ModeStore for Memory {
        async fn load(&self, key: &str) -> Option<String> {
            self.0.lock().unwrap().get(key).cloned()
        }
        async fn save(&self, key: &str, value: &str) -> Result<(), String> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_string(), value.to_string());
            Ok(())
        }
    }

    #[tokio::test]
    async fn an_agent_that_can_be_checked_is_checked_until_the_owner_says_otherwise() {
        let store = Memory::default();
        assert_eq!(
            mode_of(&store, "claude", &ClaudeCode).await,
            Some(PermissionMode::Checked)
        );
        set_mode(&store, "claude", &ClaudeCode, PermissionMode::FullAccess)
            .await
            .unwrap();
        assert_eq!(
            mode_of(&store, "claude", &ClaudeCode).await,
            Some(PermissionMode::FullAccess)
        );
        // Each agent keeps its own.
        assert_eq!(
            mode_of(&store, "claude-work", &ClaudeCode).await,
            Some(PermissionMode::Checked)
        );
    }

    #[tokio::test]
    async fn an_agent_nobody_can_check_runs_nothing_until_given_full_access() {
        let store = Memory::default();
        let agent = PlainAgent::unknown("synthetic-coder");
        assert_eq!(mode_of(&store, "synthetic-coder", &agent).await, None);
        assert!(
            set_mode(&store, "synthetic-coder", &agent, PermissionMode::Checked)
                .await
                .is_err()
        );
        // A stored mode it does not offer is not read back either.
        store
            .save(&key("synthetic-coder"), "checked")
            .await
            .unwrap();
        assert_eq!(mode_of(&store, "synthetic-coder", &agent).await, None);
        set_mode(
            &store,
            "synthetic-coder",
            &agent,
            PermissionMode::FullAccess,
        )
        .await
        .unwrap();
        assert_eq!(
            mode_of(&store, "synthetic-coder", &agent).await,
            Some(PermissionMode::FullAccess)
        );
    }
}
