use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::traits::{Message, ToolCall};

use super::ToolCallInfo;

/// Event-native conversation item projected from canonical conversation events.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub event_id: i64,
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub role: ConversationTurnRole,
    pub message_id: String,
    pub content: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub tool_calls: Option<Vec<ToolCallInfo>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationTurnRole {
    User,
    Assistant,
    Tool,
}

impl ConversationTurnRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ConversationTurnRole::User => "user",
            ConversationTurnRole::Assistant => "assistant",
            ConversationTurnRole::Tool => "tool",
        }
    }
}

fn message_id_from_event_data(data: &Value, fallback_event_id: i64) -> String {
    data.get("message_id")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| fallback_event_id.to_string())
}

fn tool_calls_from_assistant_response(data: &Value) -> Option<Vec<ToolCallInfo>> {
    let calls = data.get("tool_calls")?.as_array()?;
    let mapped: Vec<ToolCallInfo> = calls
        .iter()
        .filter_map(|tc| {
            let id = tc.get("id")?.as_str()?.to_string();
            let name = tc.get("name")?.as_str()?.to_string();
            let arguments = tc
                .get("arguments")
                .cloned()
                .and_then(|args| match args {
                    Value::String(raw) => serde_json::from_str::<Value>(&raw).ok(),
                    other => Some(other),
                })
                .unwrap_or_else(|| serde_json::json!({}));

            Some(ToolCallInfo {
                id,
                name,
                arguments,
                extra_content: tc.get("extra_content").cloned(),
            })
        })
        .collect();

    if mapped.is_empty() {
        return None;
    }
    Some(mapped)
}

/// Project a single canonical conversation event into an event-native turn.
pub fn turn_from_event(
    event_id: i64,
    session_id: &str,
    event_type: &str,
    data: &Value,
    created_at: DateTime<Utc>,
) -> Option<ConversationTurn> {
    let message_id = message_id_from_event_data(data, event_id);
    match event_type {
        "user_message" => Some(ConversationTurn {
            event_id,
            session_id: session_id.to_string(),
            created_at,
            role: ConversationTurnRole::User,
            message_id,
            content: Some(
                data.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            ),
            tool_call_id: None,
            tool_name: None,
            tool_calls: None,
        }),
        "assistant_response" => Some(ConversationTurn {
            event_id,
            session_id: session_id.to_string(),
            created_at,
            role: ConversationTurnRole::Assistant,
            message_id,
            content: data
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls: tool_calls_from_assistant_response(data),
        }),
        "tool_result" => Some(ConversationTurn {
            event_id,
            session_id: session_id.to_string(),
            created_at,
            role: ConversationTurnRole::Tool,
            message_id,
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
            tool_calls: None,
        }),
        _ => None,
    }
}

impl ConversationTurn {
    pub fn into_message(self) -> Message {
        let tool_calls_json = self.tool_calls.and_then(|calls| {
            let runtime_calls: Vec<ToolCall> = calls
                .into_iter()
                .map(|tc| ToolCall {
                    id: tc.id,
                    name: tc.name,
                    arguments: tc.arguments.to_string(),
                    extra_content: tc.extra_content,
                })
                .collect();
            if runtime_calls.is_empty() {
                return None;
            }
            serde_json::to_string(&runtime_calls).ok()
        });

        Message {
            id: self.message_id,
            session_id: self.session_id,
            role: self.role.as_str().to_string(),
            content: self.content,
            tool_call_id: self.tool_call_id,
            tool_name: self.tool_name,
            tool_calls_json,
            created_at: self.created_at,
            importance: 0.5,
            embedding: None,
        }
    }
}
