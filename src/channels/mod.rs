mod approval_render;
pub(crate) mod attachments;
pub(crate) mod commands;
pub(crate) mod connect;
#[cfg(feature = "discord")]
mod discord;
mod formatting;
mod hub;
pub(crate) mod live_status;
mod presentation;
mod rate_limit;
#[cfg(feature = "slack")]
mod slack;
pub(crate) mod telegram;
mod telegram_bootstrap_signing;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "discord")]
pub use discord::{spawn_discord_channel, DiscordChannel};
pub use hub::{ChannelHub, SessionMap};
pub(crate) use presentation::{
    prepare_chat_message, present_notification, present_scheduled_run_notification,
};
#[cfg(feature = "slack")]
pub use slack::{spawn_slack_channel, SlackChannel};
pub use telegram::TelegramChannel;

/// Shared application services/configuration consumed by every chat transport.
/// Transport constructors take this named bundle plus protocol-specific values,
/// so adding an application dependency does not change every constructor.
#[derive(Clone)]
pub(crate) struct ChannelRuntimeDeps {
    pub agent: Arc<dyn crate::runtime_ports::ChannelAgentRuntime>,
    pub config_path: PathBuf,
    pub session_map: SessionMap,
    pub task_registry: Arc<crate::tasks::TaskRegistry>,
    pub files_enabled: bool,
    pub inbox_dir: PathBuf,
    pub max_file_size_mb: u64,
    pub state: Arc<dyn crate::traits::StateStore>,
    pub watchdog_stale_threshold_secs: u64,
}

/// Wait until the heartbeat becomes stale and return stale minutes.
///
/// `stale_threshold_secs` must be > 0.
pub(crate) async fn wait_for_stale_heartbeat(
    heartbeat: Arc<AtomicU64>,
    stale_threshold_secs: u64,
    check_interval_secs: u64,
) -> u64 {
    debug_assert!(stale_threshold_secs > 0);

    loop {
        tokio::time::sleep(Duration::from_secs(check_interval_secs.max(1))).await;
        let last_hb = heartbeat.load(Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if now.saturating_sub(last_hb) > stale_threshold_secs {
            return now.saturating_sub(last_hb) / 60;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::wait_for_stale_heartbeat;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[tokio::test]
    async fn stale_heartbeat_returns_minutes() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let heartbeat = Arc::new(AtomicU64::new(now.saturating_sub(125)));

        let mins = tokio::time::timeout(
            Duration::from_secs(3),
            wait_for_stale_heartbeat(heartbeat, 60, 1),
        )
        .await
        .expect("stale heartbeat should resolve quickly");

        assert!(mins >= 2);
    }
}
