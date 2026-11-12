#[cfg(test)]
pub(crate) use aidaemon_channels::approval_render;
pub(crate) use aidaemon_channels::attachments;
pub(crate) mod commands;
pub(crate) mod connect;
#[cfg(feature = "discord")]
mod discord;
#[cfg(test)]
pub(crate) use aidaemon_channels::live_status;
#[cfg(test)]
pub(crate) mod mock_api;
use aidaemon_channels::presentation;
pub(crate) use aidaemon_channels::prompts;
#[cfg(feature = "slack")]
use aidaemon_channels::rate_limit;
pub(crate) mod set_key;
#[cfg(feature = "slack")]
mod slack;
pub(crate) mod telegram;
pub(crate) use aidaemon_channel_telegram::bot_api as telegram_api;
mod telegram_bootstrap_signing;
pub(crate) use aidaemon_channels::turns;
#[cfg(test)]
mod turns_tests;

use std::path::PathBuf;
use std::sync::Arc;

pub use aidaemon_channels::hub::{ChannelHub, SessionMap};
#[cfg(feature = "discord")]
pub use discord::{spawn_discord_channel, DiscordChannel};
pub(crate) use presentation::{
    present_notification, present_scheduled_run_notification, ScheduledRunProgress,
};
#[cfg(feature = "slack")]
pub use slack::{spawn_slack_channel, SlackChannel};
pub(crate) use telegram::{new_telegram_bot, TelegramBot};

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
        }
    }

    /// What the shared turn runner needs from these.
    pub(crate) fn turn_deps(&self) -> turns::TurnDeps {
        turns::TurnDeps {
            agent: self.agent.clone(),
            session_map: self.session_map.clone(),
            sessions: self.state.clone(),
            task_registry: self.task_registry.clone(),
            watchdog_stale_threshold_secs: self.watchdog_stale_threshold_secs,
        }
    }
}
