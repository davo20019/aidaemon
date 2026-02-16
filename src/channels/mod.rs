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

pub(crate) const LIGHTWEIGHT_INTERJECTION_IGNORE_GRACE_SECS: u64 = 120;

/// Lightweight conversational check-ins that should not be queued while work is running.
pub(crate) fn is_lightweight_interjection(text: &str) -> bool {
    let cleaned = text
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_ascii_lowercase();
    if cleaned.is_empty() || cleaned.len() > 24 {
        return false;
    }
    matches!(
        cleaned.as_str(),
        "hey"
            | "hi"
            | "hello"
            | "yo"
            | "ok"
            | "okay"
            | "thanks"
            | "thank you"
            | "thx"
            | "got it"
            | "cool"
            | "sure"
            | "yep"
            | "yes"
    )
}

/// Ignore lightweight check-ins while a task is running, but only after daemon startup grace.
pub(crate) fn should_ignore_lightweight_interjection(text: &str, daemon_uptime: Duration) -> bool {
    is_lightweight_interjection(text)
        && daemon_uptime > Duration::from_secs(LIGHTWEIGHT_INTERJECTION_IGNORE_GRACE_SECS)
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
    use super::{
        is_lightweight_interjection, should_ignore_lightweight_interjection,
        wait_for_stale_heartbeat, LIGHTWEIGHT_INTERJECTION_IGNORE_GRACE_SECS,
    };
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

    #[test]
    fn lightweight_interjection_detects_short_checkins() {
        assert!(is_lightweight_interjection("hey"));
        assert!(is_lightweight_interjection("Thanks!"));
        assert!(is_lightweight_interjection("OK"));
        assert!(is_lightweight_interjection("  got it  "));
    }

    #[test]
    fn lightweight_interjection_ignores_substantive_requests() {
        assert!(!is_lightweight_interjection("can you send me my resume?"));
        assert!(!is_lightweight_interjection("please run the tests"));
        assert!(!is_lightweight_interjection("check logs and fix the error"));
    }

    #[test]
    fn lightweight_interjection_not_ignored_during_restart_grace_window() {
        let uptime = Duration::from_secs(30);
        assert!(!should_ignore_lightweight_interjection("hi", uptime));
    }

    #[test]
    fn lightweight_interjection_ignored_after_grace_window() {
        let uptime = Duration::from_secs(LIGHTWEIGHT_INTERJECTION_IGNORE_GRACE_SECS + 5);
        assert!(should_ignore_lightweight_interjection("hi", uptime));
    }
}
