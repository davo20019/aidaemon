use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::traits::{Message, MessageAnnotation, MessageAttachment, ToolCall};

use super::{TaskStatus, ToolCallInfo};

/// A whole conversation turn fetched and grouped from canonical events.
///
/// Produced by the turn-anchored fetch (`EventStore::get_turns_from_anchor`
/// and `get_recent_turns_page`). All ordering is by `events.id` only —
/// `turn_seq` is the immutable `MIN(events.id)` over the turn's rows, and the
/// messages within are ordered by `msg_seq` (each event's own `id`).
#[derive(Debug, Clone)]
pub struct FetchedTurn {
    /// The turn's globally-unique UUID. `Some` for all reconstructed turns;
    /// legacy `turn_id IS NULL` rows are excluded from the fetch entirely.
    pub turn_id: Option<String>,
    /// `MIN(events.id)` over the turn (immutable once the opening event lands).
    pub turn_seq: i64,
    /// Hydrated messages, ordered by `msg_seq` (`events.id`) within the turn.
    pub messages: Vec<Message>,
    /// Latest `TaskEnd` status for the turn; `None` => renderer derives
    /// `Interrupted`.
    pub terminal_status: Option<TaskStatus>,
}

/// A single conversation row as projected from the turn-anchored fetch query.
/// Carries the per-turn invariants (`turn_seq`, latest terminal `status`) that
/// are constant within a turn group, alongside the hydrated message.
pub struct FetchedRow {
    pub turn_id: Option<String>,
    pub turn_seq: i64,
    pub terminal_status: Option<TaskStatus>,
    pub message: Message,
}

/// Group already-ordered conversation rows into whole turns.
///
/// Rows MUST arrive ordered by `(turn_seq ASC, msg_seq ASC)` — the queries
/// guarantee this — so a single linear pass groups consecutive rows sharing a
/// `turn_id`. `turn_seq` and `terminal_status` are constant within a group, so
/// the first row of each group fixes them. Turns may have no user message
/// (scheduled/background); grouping never synthesizes one.
pub fn group_rows_into_turns(rows: Vec<FetchedRow>) -> Vec<FetchedTurn> {
    let mut turns: Vec<FetchedTurn> = Vec::new();
    for row in rows {
        match turns.last_mut() {
            Some(last) if last.turn_id == row.turn_id && last.turn_seq == row.turn_seq => {
                last.messages.push(row.message);
            }
            _ => {
                turns.push(FetchedTurn {
                    turn_id: row.turn_id,
                    turn_seq: row.turn_seq,
                    messages: vec![row.message],
                    terminal_status: row.terminal_status,
                });
            }
        }
    }
    turns
}

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
    pub annotations: Vec<MessageAnnotation>,
    pub turn_id: Option<String>,
    pub attachments: Vec<MessageAttachment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ConversationTurnRole {
    User,
    Assistant,
    Tool,
}

impl ConversationTurnRole {
    pub fn as_str(self) -> &'static str {
        self.into()
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

fn annotations_from_event_data(data: &Value) -> Vec<MessageAnnotation> {
    data.get("annotations")
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<MessageAnnotation>>(value).ok())
        .unwrap_or_default()
}

fn attachments_from_event_data(data: &Value) -> Vec<MessageAttachment> {
    data.get("attachments")
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<MessageAttachment>>(value).ok())
        .unwrap_or_default()
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
    let turn_id = data
        .get("turn_id")
        .and_then(|v| v.as_str())
        .map(String::from);
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
            annotations: annotations_from_event_data(data),
            turn_id: turn_id.clone(),
            attachments: attachments_from_event_data(data),
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
            annotations: annotations_from_event_data(data),
            turn_id: turn_id.clone(),
            attachments: Vec::new(),
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
            annotations: annotations_from_event_data(data),
            turn_id: turn_id.clone(),
            attachments: attachments_from_event_data(data),
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
            annotations: self.annotations,
            importance: 0.5,
            embedding: None,
            // turn_id flows through from the event payload (Pillar B Task 2),
            // so messages reconstructed on hydrate carry the turn's opening
            // user-message UUID. Legacy events without it remain None.
            turn_id: self.turn_id,
            attachments: self.attachments,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn into_message_propagates_turn_id() {
        let data = json!({"message_id": "m1", "content": "hello", "turn_id": "turn-9"});
        let turn = turn_from_event(1, "sess", "user_message", &data, Utc::now()).unwrap();
        assert_eq!(turn.turn_id.as_deref(), Some("turn-9"));
        let msg = turn.into_message();
        assert_eq!(msg.turn_id.as_deref(), Some("turn-9"));
    }

    #[test]
    fn into_message_turn_id_none_for_legacy_event() {
        let data = json!({"message_id": "m1", "content": "hello"});
        let turn = turn_from_event(1, "sess", "user_message", &data, Utc::now()).unwrap();
        assert!(turn.into_message().turn_id.is_none());
    }

    #[test]
    fn tool_result_projection_round_trips_annotations() {
        let turn = turn_from_event(
            42,
            "session-1",
            "tool_result",
            &json!({
                "message_id": "msg-1",
                "tool_call_id": "call-1",
                "name": "terminal",
                "result": "cargo test\n\n[SYSTEM] Do not retry.",
                "annotations": [{"type": "appended_system_notice"}]
            }),
            Utc::now(),
        )
        .expect("tool_result should project");

        let msg = turn.into_message();
        assert_eq!(
            msg.annotations,
            vec![MessageAnnotation::AppendedSystemNotice]
        );
        assert_eq!(msg.primary_content().as_deref(), Some("cargo test"));
    }

    #[test]
    fn tool_result_projection_legacy_messages_infer_annotations() {
        let turn = turn_from_event(
            43,
            "session-1",
            "tool_result",
            &json!({
                "message_id": "msg-2",
                "tool_call_id": "call-2",
                "name": "system",
                "result": "[SYSTEM] Before executing tools, narrate the plan."
            }),
            Utc::now(),
        )
        .expect("tool_result should project");

        let msg = turn.into_message();
        assert!(
            msg.annotations.is_empty(),
            "legacy rows should not require backfill"
        );
        assert!(msg.is_structural_only());
    }
}
