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
fn telegram_session_namespace_and_chat_id(session_id: &str) -> Option<(Option<&str>, i64)> {
    if let Some((marker, rest)) = find_platform_substring(session_id) {
        if marker != "telegram:" {
            return None;
        }
        let marker_idx = session_id.rfind(marker)?;
        let prefix = session_id[..marker_idx].trim_end_matches(':').trim();
        let chat_id = rest
            .strip_prefix("telegram:")
            .and_then(|s| s.trim().parse::<i64>().ok())?;
        let namespace = if prefix.is_empty() {
            None
        } else {
            Some(prefix)
        };
        return Some((namespace, chat_id));
    }

    // Legacy: "bot:telegram:12345" or "bot:12345"
    let stripped = session_id.strip_prefix("bot:").unwrap_or(session_id);

    // Plain numeric (legacy/default bot)
    if let Ok(chat_id) = stripped.trim().parse::<i64>() {
        return Some((None, chat_id));
    }

    // Heuristic: multi-bot sessions commonly look like "{username}:{chat_id}".
    // Only apply when no other platform marker exists.
    if stripped.contains("slack:") || stripped.contains("discord:") {
        return None;
    }

    let (prefix, suffix) = stripped.rsplit_once(':')?;
    let prefix = prefix.trim();
    if prefix.is_empty() {
        return None;
    }
    let chat_id = suffix.trim().parse::<i64>().ok()?;
    Some((Some(prefix), chat_id))
}

pub fn telegram_chat_id_from_session(session_id: &str) -> Option<i64> {
    telegram_session_namespace_and_chat_id(session_id).map(|(_, chat_id)| chat_id)
}

fn telegram_channel_namespace_and_chat_id(channel_id: &str) -> Option<(Option<&str>, i64)> {
    let stripped = channel_id.strip_prefix("telegram:")?;
    if let Ok(chat_id) = stripped.trim().parse::<i64>() {
        return Some((None, chat_id));
    }

    let (namespace, suffix) = stripped.rsplit_once(':')?;
    let namespace = namespace.trim();
    if namespace.is_empty() {
        return None;
    }
    let chat_id = suffix.trim().parse::<i64>().ok()?;
    Some((Some(namespace), chat_id))
}

/// Returns true when a stored channel id should be considered the same channel
/// as the current one. This preserves reads for legacy Telegram facts/episodes
/// (`telegram:{chat_id}`) after introducing namespaced multi-bot ids.
pub fn stored_channel_matches_current(stored_channel_id: &str, current_channel_id: &str) -> bool {
    if stored_channel_id == current_channel_id {
        return true;
    }

    let Some((stored_namespace, stored_chat_id)) =
        telegram_channel_namespace_and_chat_id(stored_channel_id)
    else {
        return false;
    };
    let Some((current_namespace, current_chat_id)) =
        telegram_channel_namespace_and_chat_id(current_channel_id)
    else {
        return false;
    };

    stored_chat_id == current_chat_id && stored_namespace.is_none() && current_namespace.is_some()
}

/// Derive a Telegram channel_id that preserves multi-bot namespaces.
///
/// Returns:
/// - "telegram:{CHAT_ID}" for legacy/single-bot sessions
/// - "telegram:{BOT_NAMESPACE}:{CHAT_ID}" for multi-bot sessions
pub fn telegram_channel_id_from_session(session_id: &str) -> Option<String> {
    let (namespace, chat_id) = telegram_session_namespace_and_chat_id(session_id)?;
    match namespace {
        Some(prefix) => Some(format!("telegram:{}:{}", prefix, chat_id)),
        None => Some(format!("telegram:{}", chat_id)),
    }
}

/// Derive a canonical `channel_id` (used for memory scoping) from a session ID.
///
/// Returns platform-scoped IDs:
/// - Slack:   "slack:{CHANNEL_ID}"
/// - Discord: "discord:dm:{ID}" / "discord:ch:{ID}"
/// - Telegram: "telegram:{CHAT_ID}" or "telegram:{BOT_NAMESPACE}:{CHAT_ID}"
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
                return telegram_channel_id_from_session(session_id);
            }
            _ => {}
        }
    }

    // Telegram fallback (numeric or "{bot}:{chat_id}")
    telegram_channel_id_from_session(session_id)
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
            Some("telegram:mybot:12345")
        );
        assert_eq!(
            derive_channel_id_from_session("mybot:telegram:12345").as_deref(),
            Some("telegram:mybot:12345")
        );
    }

    #[test]
    fn stored_channel_match_restores_legacy_telegram_reads() {
        assert!(stored_channel_matches_current(
            "telegram:12345",
            "telegram:mybot:12345"
        ));
        assert!(!stored_channel_matches_current(
            "telegram:otherbot:12345",
            "telegram:mybot:12345"
        ));
        assert!(!stored_channel_matches_current(
            "telegram:mybot:12345",
            "telegram:12345"
        ));
    }

    #[test]
    fn group_session_detects_telegram_groups() {
        assert!(is_group_session("-10012345"));
        assert!(is_group_session("mybot:-10012345"));
        assert!(!is_group_session("12345"));
        assert!(!is_group_session("mybot:12345"));
    }
}
