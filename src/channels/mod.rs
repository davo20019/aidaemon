#[cfg(test)]
pub(crate) use aidaemon_channels::approval_render;
#[cfg(test)]
pub(crate) use aidaemon_channels::attachments;
pub(crate) mod commands;
pub(crate) mod connect;
#[cfg(feature = "discord")]
pub(crate) mod discord;
#[cfg(test)]
pub(crate) use aidaemon_channels::live_status;
#[cfg(test)]
pub(crate) mod mock_api;
pub(crate) mod set_key;
#[cfg(feature = "slack")]
pub(crate) mod slack;
pub(crate) mod telegram;
pub(crate) use aidaemon_channel_telegram::bot_api as telegram_api;
#[cfg(test)]
pub(crate) use aidaemon_channels::turns;
#[cfg(test)]
mod turns_tests;

use std::path::PathBuf;
use std::sync::Arc;

pub use aidaemon_channels::hub::{ChannelHub, SessionMap};
#[cfg(feature = "discord")]
pub use discord::{DiscordChannel, spawn_discord_channel};
#[cfg(feature = "slack")]
pub use slack::{SlackChannel, spawn_slack_channel};
pub(crate) use telegram::{TelegramBot, new_telegram_bot};

/// Shared application services/configuration consumed by every chat transport.
/// Transport constructors take this named bundle plus protocol-specific values,
/// so adding an application dependency does not change every constructor.
#[derive(Clone)]
pub(crate) struct ChannelRuntimeDeps {
    pub agent: Arc<dyn crate::runtime_ports::DaemonAgentRuntime>,
    pub config_path: PathBuf,
    pub session_map: SessionMap,
    pub task_registry: Arc<crate::tasks::TaskRegistry>,
    pub files_enabled: bool,
    pub inbox_dir: PathBuf,
    pub max_file_size_mb: u64,
    pub state: Arc<dyn crate::traits::StateStore>,
    pub watchdog_stale_threshold_secs: u64,
    /// Who sees every turn: the Client Gateway, when it runs.
    pub watch: aidaemon_channels::turns::TurnWatch,
}

impl ChannelRuntimeDeps {
    /// What a bot in an adapter crate runs on.
    pub(crate) fn bot_deps(&self) -> aidaemon_channels::bot::BotDeps {
        aidaemon_channels::bot::BotDeps {
            agent: self.agent.clone(),
            session_map: self.session_map.clone(),
            sessions: self.state.clone(),
            task_registry: self.task_registry.clone(),
            files_enabled: self.files_enabled,
            inbox_dir: self.inbox_dir.clone(),
            max_file_size_mb: self.max_file_size_mb,
            watchdog_stale_threshold_secs: self.watchdog_stale_threshold_secs,
            watch: self.watch.clone(),
        }
    }

    /// The shared slash commands on the daemon's stores.
    pub(crate) fn command_context(&self) -> commands::CommandContext {
        commands::CommandContext {
            agent: self.agent.clone(),
            state: self.state.clone(),
            task_registry: self.task_registry.clone(),
            config_path: self.config_path.clone(),
        }
    }

    /// What the shared turn runner needs from these.
    #[cfg(test)]
    pub(crate) fn turn_deps(&self) -> turns::TurnDeps {
        turns::TurnDeps {
            agent: self.agent.clone(),
            session_map: self.session_map.clone(),
            sessions: self.state.clone(),
            task_registry: self.task_registry.clone(),
            watchdog_stale_threshold_secs: self.watchdog_stale_threshold_secs,
            watch: self.watch.clone(),
        }
    }
}
