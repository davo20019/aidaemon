//! What a platform bot (the Telegram, Slack and Discord bots in their
//! adapter crates) runs on: the agent, sessions, the task registry and file
//! settings, as one bundle, and who counts as the owner.

use std::path::PathBuf;
use std::sync::Arc;

use aidaemon_tools::semantics::UserRole;

use crate::hub::{SessionChannelStore, SessionMap};
use crate::ports::ChannelAgentRuntime;
use crate::tasks::TaskRegistry;
use crate::turns::TurnDeps;

/// The services and settings every bot takes, so adding one does not
/// change each bot's constructor.
#[derive(Clone)]
pub struct BotDeps {
    pub agent: Arc<dyn ChannelAgentRuntime>,
    /// Which bot each session belongs to, for outbound routing.
    pub session_map: SessionMap,
    /// Where that mapping is persisted.
    pub sessions: Arc<dyn SessionChannelStore>,
    pub task_registry: Arc<TaskRegistry>,
    /// Whether inbound files are accepted.
    pub files_enabled: bool,
    /// Where inbound files are saved.
    pub inbox_dir: PathBuf,
    /// Largest inbound file accepted, in MB.
    pub max_file_size_mb: u64,
    /// Heartbeat silence (seconds) after which a turn is cancelled as hung;
    /// 0 never cancels.
    pub watchdog_stale_threshold_secs: u64,
}

impl BotDeps {
    /// What the shared turn runner needs from these.
    pub fn turn_deps(&self) -> TurnDeps {
        TurnDeps {
            agent: self.agent.clone(),
            session_map: self.session_map.clone(),
            sessions: self.sessions.clone(),
            task_registry: self.task_registry.clone(),
            watchdog_stale_threshold_secs: self.watchdog_stale_threshold_secs,
        }
    }
}

/// A sender's role: Owner when `owner_ids` is empty (every allowed user is
/// an owner) or lists them; Guest otherwise.
pub fn determine_role(owner_ids: &[u64], user_id: u64) -> UserRole {
    if owner_ids.is_empty() || owner_ids.contains(&user_id) {
        UserRole::Owner
    } else {
        UserRole::Guest
    }
}

/// The shared slash commands as the application runs them (its stores,
/// its extra commands).
#[async_trait::async_trait]
pub trait SharedCommandDispatch: Send + Sync {
    /// The reply to `cmd`; `None` when it is not a shared command.
    async fn dispatch(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
        user_role: UserRole,
    ) -> Option<String>;
}
