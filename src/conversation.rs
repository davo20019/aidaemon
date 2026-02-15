//! Conversation projection helpers.
//!
//! The canonical persistence path is event-sourced (`events` table). Multiple
//! call sites need to project those events into the legacy `traits::Message`
//! shape used by providers and tooling. Keeping this logic centralized avoids
//! subtle drift (ordering, tool_call serialization, truncation).

use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::traits::{Message, ToolCall};

fn message_id_from_event_data(data: &Value, fallback_event_id: i64) -> String {
    data.get("message_id")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| fallback_event_id.to_string())
}

fn tool_calls_json_from_assistant_response(data: &Value) -> Option<String> {
    let calls = data.get("tool_calls")?.as_array()?;
    let mapped: Vec<ToolCall> = calls
        .iter()
        .filter_map(|tc| {
            let id = tc.get("id")?.as_str()?;
            let name = tc.get("name")?.as_str()?;
            Some(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: tc
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}))
                    .to_string(),
                extra_content: tc.get("extra_content").cloned(),
            })
        })
        .collect();
    if mapped.is_empty() {
        return None;
    }
    serde_json::to_string(&mapped).ok()
}

/// Project a single canonical event row into a `Message` (legacy conversation shape).
pub fn message_from_event(
    event_id: i64,
    session_id: &str,
    event_type: &str,
    data: &Value,
    created_at: DateTime<Utc>,
) -> Option<Message> {
    let message_id = message_id_from_event_data(data, event_id);
    match event_type {
        "user_message" => Some(Message {
            id: message_id,
            session_id: session_id.to_string(),
            role: "user".to_string(),
            content: Some(
                data.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            ),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at,
            importance: 0.5,
            embedding: None,
        }),
        "assistant_response" => {
            let tool_calls_json = tool_calls_json_from_assistant_response(data);
            Some(Message {
                id: message_id,
                session_id: session_id.to_string(),
                role: "assistant".to_string(),
                content: data
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                tool_call_id: None,
                tool_name: None,
                tool_calls_json,
                created_at,
                importance: 0.5,
                embedding: None,
            })
        }
        "tool_result" => Some(Message {
            id: message_id,
            session_id: session_id.to_string(),
            role: "tool".to_string(),
            content: Some(
                data.get("result")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            ),
            tool_call_id: Some(
                data.get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("event-tool-{}", event_id)),
            ),
            tool_name: Some(
                data.get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("system")
                    .to_string(),
            ),
            tool_calls_json: None,
            created_at,
            importance: 0.5,
            embedding: None,
        }),
        _ => None,
    }
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
