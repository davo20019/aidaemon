//! An Agent Client Protocol agent's run read into [`AgentEvent`]s, from the
//! lines its host gives the run's reader ([`super::host`]): `acp.session`
//! (its session id, its model, and whether a later run can continue it:
//! the agent's own `loadSession`), `tool_call` (the call's id, its ACP `kind`, the paths
//! it names, its command or URL), `tool_call_update` (a call completed or
//! failed), `acp.plan` (its plan entries) and `result` (its turn's end,
//! with its final message as `final_message`: the protocol holds it to no
//! schema, so it is text alone).
//! A step's kind is the protocol's `kind` value, never its title.

use serde_json::Value;

use crate::events::{
    AgentEvent, AgentStep, PlanItem, PlanStatus, StepTool, host_of, program_of, text,
};

/// The events one line carries.
pub fn read_events(line: &str) -> Vec<AgentEvent> {
    let Ok(event) = serde_json::from_str::<Value>(line.trim()) else {
        return Vec::new();
    };
    match event.get("type").and_then(Value::as_str) {
        Some("acp.session") => text(&event, "/session_id")
            .map(|id| AgentEvent::Session {
                id,
                model: text(&event, "/model"),
                // The agent's word, given only once it runs (its
                // `loadSession` capability); absent: the launch decided.
                resumable: event.get("resumable").and_then(Value::as_bool),
            })
            .into_iter()
            .collect(),
        Some("tool_call") => {
            let path = text(&event, "/paths/0");
            // ACP's tool kinds (a protocol enum).
            let tool = match event.get("kind").and_then(Value::as_str) {
                Some("read") => StepTool::Read { path },
                Some("edit" | "delete" | "move") => StepTool::Edit { path },
                Some("search") => StepTool::Search { pattern: None },
                Some("execute") => StepTool::Run {
                    program: text(&event, "/command").as_deref().and_then(program_of),
                },
                Some("fetch") => StepTool::Fetch {
                    host: text(&event, "/url").as_deref().and_then(host_of),
                },
                // Its own thinking is not a step.
                Some("think") => return Vec::new(),
                Some(other) => StepTool::Other {
                    name: other.to_string(),
                },
                None => StepTool::Other {
                    name: "a tool".to_string(),
                },
            };
            vec![AgentEvent::Step(AgentStep {
                id: text(&event, "/id"),
                parent: None,
                tool,
            })]
        }
        Some("tool_call_update") => text(&event, "/id")
            .map(|id| AgentEvent::StepEnded {
                id,
                failed: text(&event, "/status").as_deref() == Some("failed"),
            })
            .into_iter()
            .collect(),
        Some("acp.plan") => vec![AgentEvent::Plan(
            event
                .get("entries")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|entry| {
                    Some(PlanItem {
                        text: text(entry, "/content")?,
                        status: match entry.get("status").and_then(Value::as_str) {
                            Some("completed") => PlanStatus::Done,
                            Some("in_progress") => PlanStatus::InProgress,
                            _ => PlanStatus::Pending,
                        },
                    })
                })
                .collect(),
        )],
        Some("result") => {
            let mut events: Vec<AgentEvent> = event
                .get("final_message")
                .and_then(Value::as_str)
                .map(|text| AgentEvent::Reported {
                    text: text.trim().to_string(),
                    report: None,
                })
                .into_iter()
                .collect();
            events.push(AgentEvent::TurnEnded {
                failed: event.get("is_error").and_then(Value::as_bool) == Some(true),
            });
            events
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_session_tool_calls_and_end_are_read_from_typed_fields() {
        assert_eq!(
            read_events(r#"{"type":"acp.session","session_id":"sess-1"}"#),
            vec![AgentEvent::Session {
                id: "sess-1".into(),
                model: None,
                resumable: None,
            }]
        );
        assert_eq!(
            read_events(
                r#"{"type":"acp.session","session_id":"sess-2","model":"synthetic-large","resumable":true}"#
            ),
            vec![AgentEvent::Session {
                id: "sess-2".into(),
                model: Some("synthetic-large".into()),
                resumable: Some(true),
            }]
        );
        assert!(matches!(
            &read_events(r#"{"type":"acp.session","session_id":"sess-3","resumable":false}"#)[0],
            AgentEvent::Session {
                resumable: Some(false),
                ..
            }
        ));
        assert_eq!(
            read_events(
                r#"{"type":"tool_call","id":"c1","title":"Delete everything","kind":"edit","paths":["/w/synthetic/notes.md"]}"#
            ),
            vec![AgentEvent::Step(AgentStep {
                id: Some("c1".into()),
                parent: None,
                tool: StepTool::Edit {
                    path: Some("/w/synthetic/notes.md".into())
                }
            })]
        );
        assert!(
            matches!(&read_events(r#"{"type":"tool_call","id":"c2","kind":"execute","command":"pytest -q"}"#)[0],
                AgentEvent::Step(step) if step.tool == StepTool::Run { program: Some("pytest".into()) })
        );
        assert!(read_events(r#"{"type":"tool_call","id":"c3","kind":"think"}"#).is_empty());
        assert_eq!(
            read_events(r#"{"type":"tool_call_update","id":"c1","status":"failed"}"#),
            vec![AgentEvent::StepEnded {
                id: "c1".into(),
                failed: true
            }]
        );
        assert_eq!(
            read_events(r#"{"type":"result","subtype":"success","result":"ok"}"#),
            vec![AgentEvent::TurnEnded { failed: false }]
        );
    }

    #[test]
    fn its_final_message_is_text_alone_even_when_it_is_a_report() {
        assert_eq!(
            read_events(
                r#"{"type":"result","subtype":"success","result":"Created note.txt.","final_message":"Created note.txt."}"#
            ),
            vec![
                AgentEvent::Reported {
                    text: "Created note.txt.".into(),
                    report: None
                },
                AgentEvent::TurnEnded { failed: false }
            ]
        );
        // Held to no schema: JSON it writes is its text, never a report.
        let json = r#"{"summary":"x","changed_files":[],"checks":[],"open_questions":[],"needs_owner":false}"#;
        let line = serde_json::json!({"type": "result", "subtype": "success", "result": json, "final_message": json});
        assert_eq!(
            read_events(&line.to_string())[0],
            AgentEvent::Reported {
                text: json.into(),
                report: None
            }
        );
        // A turn that stopped: its message, not the reason.
        assert_eq!(
            read_events(
                r#"{"type":"result","subtype":"error","is_error":true,"result":"The agent stopped (refusal). Half done.","final_message":"Half done."}"#
            ),
            vec![
                AgentEvent::Reported {
                    text: "Half done.".into(),
                    report: None
                },
                AgentEvent::TurnEnded { failed: true }
            ]
        );
    }
}
