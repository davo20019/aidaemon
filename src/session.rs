//! Session ID parsing utilities.
//!
//! The codebase historically used stringly-typed session IDs with different
//! per-channel formats (and optional bot-name prefixes). Centralizing parsing
//! here reduces duplicated heuristics and keeps behavior consistent.

/// Returns the first well-known platform marker found inside `session_id`.
///
/// For multi-bot formats like "{bot}:slack:C123", this finds "slack:".
fn find_platform_substring(session_id: &str) -> Option<(&'static str, &str)> {
    // Keep this list small and explicit: only markers that are actually used
    // in session IDs today.
    for marker in ["slack:", "discord:", "telegram:"].into_iter() {
        if let Some(idx) = session_id.rfind(marker) {
            return Some((marker, &session_id[idx..]));
        }
    }
    None
}

/// Extract a Telegram chat ID from a session ID.
///
/// Supports:
/// - "12345" (single-bot legacy/default)
/// - "-10012345" (Telegram groups/supergroups are negative)
/// - "telegram:12345"
/// - "{bot}:12345" (multi-bot Telegram sessions)
/// - "bot:telegram:12345" (legacy pattern)
pub fn telegram_chat_id_from_session(session_id: &str) -> Option<i64> {
    // Prefer explicit marker forms.
    if let Some((marker, rest)) = find_platform_substring(session_id) {
        if marker != "telegram:" {
            return None;
        }
        return rest
            .strip_prefix("telegram:")
            .and_then(|s| s.trim().parse::<i64>().ok());
    }

    // Legacy: "bot:telegram:12345" or "bot:12345"
    let stripped = session_id.strip_prefix("bot:").unwrap_or(session_id);

    // Plain numeric (legacy/default bot)
    if let Ok(chat_id) = stripped.trim().parse::<i64>() {
        return Some(chat_id);
    }

    // Heuristic: multi-bot sessions commonly look like "{username}:{chat_id}".
    // Only apply when no other platform marker exists.
    if stripped.contains("slack:") || stripped.contains("discord:") {
        return None;
    }

    let (prefix, suffix) = stripped.rsplit_once(':')?;
    if prefix.trim().is_empty() {
        return None;
    }
    suffix.trim().parse::<i64>().ok()
}

/// Derive a canonical `channel_id` (used for memory scoping) from a session ID.
///
/// Returns platform-scoped IDs:
/// - Slack:   "slack:{CHANNEL_ID}"
/// - Discord: "discord:dm:{ID}" / "discord:ch:{ID}"
/// - Telegram: "telegram:{CHAT_ID}"
pub fn derive_channel_id_from_session(session_id: &str) -> Option<String> {
    if let Some((marker, rest)) = find_platform_substring(session_id) {
        match marker {
            "slack:" => {
                // "slack:C123" or "slack:C123:ts" -> "slack:C123"
                let mut it = rest.splitn(3, ':');
                let _slack = it.next()?;
                let channel = it.next()?;
                if channel.is_empty() {
                    return None;
                }
                return Some(format!("slack:{}", channel));
            }
            "discord:" => {
                // Keep "discord:dm:..." / "discord:ch:..." as-is (strip bot prefix).
                return Some(rest.to_string());
            }
            "telegram:" => {
                let chat_id = telegram_chat_id_from_session(rest)?;
                return Some(format!("telegram:{}", chat_id));
            }
            _ => {}
        }
    }

    // Telegram fallback (numeric or "{bot}:{chat_id}")
    telegram_chat_id_from_session(session_id).map(|id| format!("telegram:{}", id))
}

/// True if a session corresponds to a shared/group channel (vs a 1:1 DM).
///
/// Used to suppress noisy progress updates in shared channels.
pub fn is_group_session(session_id: &str) -> bool {
    if let Some((marker, rest)) = find_platform_substring(session_id) {
        match marker {
            "discord:" => {
                // "discord:ch:{id}" indicates a guild channel
                return rest.contains("discord:ch:");
            }
            "slack:" => {
                // Slack public/private channels start with C or G. DMs start with D.
                // Format: "slack:{channel_id}" or "slack:{channel_id}:{thread_ts}"
                if let Some(after) = rest.strip_prefix("slack:") {
                    return after.starts_with('C') || after.starts_with('G');
                }
            }
            "telegram:" => {
                if let Some(chat_id) = telegram_chat_id_from_session(rest) {
                    return chat_id < 0;
                }
            }
            _ => {}
        }
    }

    // Telegram fallback: negative chat IDs are groups/supergroups.
    if let Some(chat_id) = telegram_chat_id_from_session(session_id) {
        return chat_id < 0;
    }

    false
}

/// Extract a Discord channel/DM ID from a session ID.
///
/// Supports:
/// - "discord:ch:12345" or "discord:dm:12345"
/// - "{bot}:discord:ch:12345"
#[cfg(feature = "discord")]
pub fn discord_channel_id_from_session(session_id: &str) -> Option<u64> {
    if let Some((marker, rest)) = find_platform_substring(session_id) {
        if marker != "discord:" {
            return None;
        }
        // rest looks like "discord:ch:12345" or "discord:dm:12345"
        let stripped = rest.strip_prefix("discord:")?;
        // stripped is "ch:12345" or "dm:12345"
        let id_str = stripped
            .strip_prefix("ch:")
            .or_else(|| stripped.strip_prefix("dm:"))?;
        return id_str.trim().parse::<u64>().ok();
    }
    None
}

/// Extract Slack channel ID and optional thread_ts from a session ID.
///
/// Supports:
/// - "slack:C12345"
/// - "slack:C12345:1234567890.123456"
/// - "{bot}:slack:C12345:ts"
#[cfg(feature = "slack")]
pub fn slack_channel_from_session(session_id: &str) -> (String, Option<String>) {
    if let Some((marker, rest)) = find_platform_substring(session_id) {
        if marker == "slack:" {
            let stripped = rest.strip_prefix("slack:").unwrap_or(rest);
            let parts: Vec<&str> = stripped.splitn(2, ':').collect();
            let channel_id = parts[0].to_string();
            let thread_ts = parts.get(1).map(|s| s.to_string());
            return (channel_id, thread_ts);
        }
    }
    // Fallback: return the whole session_id as channel, no thread
    (session_id.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telegram_chat_id_parses_default_numeric() {
        assert_eq!(telegram_chat_id_from_session("12345"), Some(12345));
        assert_eq!(telegram_chat_id_from_session("-10012345"), Some(-10012345));
    }

    #[test]
    fn telegram_chat_id_parses_multi_bot() {
        assert_eq!(telegram_chat_id_from_session("mybot:12345"), Some(12345));
        assert_eq!(
            telegram_chat_id_from_session("mybot:-10012345"),
            Some(-10012345)
        );
    }

    #[test]
    fn telegram_chat_id_parses_explicit_marker() {
        assert_eq!(telegram_chat_id_from_session("telegram:12345"), Some(12345));
        assert_eq!(
            telegram_chat_id_from_session("bot:telegram:12345"),
            Some(12345)
        );
    }

    #[test]
    fn derive_channel_id_handles_slack_and_discord_prefixes() {
        assert_eq!(
            derive_channel_id_from_session("mybot:slack:C123:123.45").as_deref(),
            Some("slack:C123")
        );
        assert_eq!(
            derive_channel_id_from_session("mybot:discord:dm:42").as_deref(),
            Some("discord:dm:42")
        );
    }

    #[test]
    fn derive_channel_id_handles_telegram_forms() {
        assert_eq!(
            derive_channel_id_from_session("12345").as_deref(),
            Some("telegram:12345")
        );
        assert_eq!(
            derive_channel_id_from_session("mybot:12345").as_deref(),
            Some("telegram:12345")
        );
    }

    #[test]
    fn group_session_detects_telegram_groups() {
        assert!(is_group_session("-10012345"));
        assert!(is_group_session("mybot:-10012345"));
        assert!(!is_group_session("12345"));
        assert!(!is_group_session("mybot:12345"));
    }
}
