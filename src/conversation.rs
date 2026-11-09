//! Conversation projection helpers.
//!
//! The canonical persistence path is event-sourced (`events` table). Event ->
//! turn projection now lives in `crate::events::turn_from_event`; this module
//! keeps runtime `traits::Message` compatibility helpers (including truncation).

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::traits::Message;

/// Project a single canonical event row into a `Message` (runtime conversation shape).
#[allow(dead_code)] // Transitional shim while runtime callers migrate to event-native turns.
pub fn message_from_event(
    event_id: i64,
    session_id: &str,
    event_type: &str,
    data: &Value,
    created_at: DateTime<Utc>,
) -> Option<Message> {
    crate::events::turn_from_event(event_id, session_id, event_type, data, created_at)
        .map(|turn| turn.into_message())
}

pub use aidaemon_events::conversation::truncate_with_anchor;

#[cfg(test)]
mod tests {
    use super::truncate_with_anchor;
    use crate::traits::Message;

    fn message(index: usize, role: &str) -> Message {
        let mut message = Message::new_runtime(format!("m{index}"), "s", role);
        message.content = Some(format!("{role} {index}"));
        message
    }

    #[test]
    fn pinned_anchor_keeps_the_newest_messages() {
        // user, then alternating assistant/user; the tail window would start
        // on an assistant message, so the first user message is pinned.
        let messages: Vec<Message> = (0..9)
            .map(|index| message(index, if index % 2 == 0 { "user" } else { "assistant" }))
            .collect();
        let truncated = truncate_with_anchor(messages, 4);
        let ids: Vec<&str> = truncated.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["m0", "m6", "m7", "m8"]);
    }
}
