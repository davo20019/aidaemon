//! Codex's JSON events read into [`AgentEvent`]s: what `codex exec --json`
//! prints, and what this crate's host prints in its place from the
//! app-server protocol ([`super::host`]). `thread.started` carries its
//! thread id; `item.started`/`item.completed` carry an item whose `type`
//! says what it is (`command_execution`, `file_change`, `web_search`,
//! `mcp_tool_call`, `todo_list`, ...); `turn.completed`/`turn.failed` end
//! its turn.

use serde_json::Value;

use crate::events::{AgentEvent, AgentStep, PlanItem, PlanStatus, StepTool, program_of, text};

/// The events one stdout line carries.
pub fn read_events(line: &str) -> Vec<AgentEvent> {
    let Ok(event) = serde_json::from_str::<Value>(line.trim()) else {
        return Vec::new();
    };
    match event.get("type").and_then(Value::as_str) {
        Some("thread.started") => text(&event, "/thread_id")
            .map(|id| AgentEvent::Session {
                id,
                model: None,
                resumable: None,
            })
            .into_iter()
            .collect(),
        Some(kind @ ("item.started" | "item.updated" | "item.completed")) => {
            let item = event.get("item").unwrap_or(&Value::Null);
            item_events(item, kind == "item.completed")
        }
        Some("turn.completed") => {
            let count = |key: &str| {
                event
                    .pointer(&format!("/usage/{key}"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            };
            vec![
                AgentEvent::Usage {
                    input_tokens: count("input_tokens"),
                    output_tokens: count("output_tokens"),
                    cost_usd: None,
                },
                AgentEvent::TurnEnded { failed: false },
            ]
        }
        Some("turn.failed") => vec![AgentEvent::TurnEnded { failed: true }],
        _ => Vec::new(),
    }
}

/// An item's step (when it is one), and its end once it completed. An item
/// seen only completed (a file change) is a step that started and ended;
/// a reader of these events keeps one step per id.
fn item_events(item: &Value, completed: bool) -> Vec<AgentEvent> {
    let id = text(item, "/id");
    let tool = match item.get("type").and_then(Value::as_str) {
        Some("command_execution") => StepTool::Run {
            program: text(item, "/command").as_deref().and_then(program_of),
        },
        Some("file_change") => StepTool::Edit {
            path: text(item, "/changes/0/path"),
        },
        Some("web_search") => StepTool::Fetch { host: None },
        Some("mcp_tool_call") => StepTool::Other {
            name: match (text(item, "/server"), text(item, "/tool")) {
                (Some(server), Some(tool)) => format!("{server}.{tool}"),
                (_, Some(tool)) => tool,
                _ => "a tool".to_string(),
            },
        },
        Some("todo_list") => {
            let items = item
                .get("items")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|todo| {
                    Some(PlanItem {
                        text: text(todo, "/text")?,
                        status: if todo.get("completed").and_then(Value::as_bool) == Some(true) {
                            PlanStatus::Done
                        } else {
                            PlanStatus::Pending
                        },
                    })
                })
                .collect();
            return vec![AgentEvent::Plan(items)];
        }
        // Its messages and reasoning are not steps.
        _ => return Vec::new(),
    };
    let mut events = vec![AgentEvent::Step(AgentStep {
        id: id.clone(),
        parent: None,
        tool,
    })];
    if completed && let Some(id) = id {
        let status = text(item, "/status");
        let exit = item.get("exit_code").and_then(Value::as_i64);
        events.push(AgentEvent::StepEnded {
            id,
            failed: status.as_deref() == Some("failed") || exit.is_some_and(|code| code != 0),
        });
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_thread_items_and_turn_are_read_from_typed_fields() {
        assert_eq!(
            read_events(r#"{"type":"thread.started","thread_id":"th-1"}"#),
            vec![AgentEvent::Session {
                id: "th-1".into(),
                model: None,
                resumable: None,
            }]
        );
        assert_eq!(
            read_events(
                r#"{"type":"item.started","item":{"id":"i1","type":"command_execution","command":"bash -lc 'npm test'","status":"in_progress"}}"#
            ),
            vec![AgentEvent::Step(AgentStep {
                id: Some("i1".into()),
                parent: None,
                tool: StepTool::Run {
                    program: Some("npm".into())
                }
            })]
        );
        let done = read_events(
            r#"{"type":"item.completed","item":{"id":"i1","type":"command_execution","command":"npm test","exit_code":1,"status":"completed"}}"#,
        );
        assert_eq!(
            done[1],
            AgentEvent::StepEnded {
                id: "i1".into(),
                failed: true
            }
        );
        let change = read_events(
            r#"{"type":"item.completed","item":{"id":"i2","type":"file_change","changes":[{"path":"/w/synthetic/a.rs","kind":"update"}],"status":"completed"}}"#,
        );
        assert!(
            matches!(&change[0], AgentEvent::Step(step) if step.tool == StepTool::Edit { path: Some("/w/synthetic/a.rs".into()) })
        );
        assert!(
            read_events(
                r#"{"type":"item.completed","item":{"id":"m","type":"agent_message","text":"done"}}"#
            )
            .is_empty()
        );
        assert_eq!(
            read_events(
                r#"{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":3}}"#
            )[1],
            AgentEvent::TurnEnded { failed: false }
        );
        assert_eq!(
            read_events(r#"{"type":"turn.failed","error":{"message":"x"}}"#),
            vec![AgentEvent::TurnEnded { failed: true }]
        );
    }
}
