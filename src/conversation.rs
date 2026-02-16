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

/// Truncate a conversation while preserving the first user message ("anchor") within the slice.
///
/// Some providers require assistant/tool turns to follow a user message. This helper
/// ensures the returned slice does not accidentally drop the first user message
/// when truncating to a tail window.
pub fn truncate_with_anchor(messages: Vec<Message>, limit: usize) -> Vec<Message> {
    if messages.len() <= limit {
        return messages;
    }

    // Find the first user message (anchor) *within* this slice.
    let anchor_idx = messages.iter().position(|m| m.role == "user");
    let skip = messages.len().saturating_sub(limit);

    if let Some(anchor) = anchor_idx {
        if skip > anchor {
            // We would skip past the anchor - preserve it
            let anchor_msg = messages[anchor].clone();
            let remaining: Vec<_> = messages.into_iter().skip(skip).collect();

            // Only prepend anchor if not already in remaining
            if remaining.first().map(|m| m.role.as_str()) != Some("user") {
                let mut result = vec![anchor_msg];
                result.extend(remaining.into_iter().take(limit.saturating_sub(1)));
                return result;
            }
            return remaining;
        }
    }

    // Normal case - just take last N
    messages.into_iter().skip(skip).collect()
}
