#[cfg(feature = "discord")]
mod discord;
mod formatting;
mod hub;
#[cfg(feature = "slack")]
mod slack;
pub(crate) mod telegram;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "discord")]
pub use discord::{spawn_discord_channel, DiscordChannel};
pub use hub::{ChannelHub, SessionMap};
#[cfg(feature = "slack")]
pub use slack::{spawn_slack_channel, SlackChannel};
pub use telegram::TelegramChannel;

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
