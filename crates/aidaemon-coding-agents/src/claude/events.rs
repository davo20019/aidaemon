//! Claude Code's stream-json output read into [`AgentEvent`]s.
//!
//! Its lines are typed messages: `system`/`init` (its session id and
//! model), `assistant` (with `tool_use` blocks: a tool's name, input and
//! id; `parent_tool_use_id` when one of its own helpers runs it), `user`
//! (with `tool_result` blocks: the call's id and `is_error`) and `result`
//! (its turn's end, cost and usage). A tool is told apart by its name, the
//! protocol's identifier for it.

use serde_json::Value;

use crate::events::{
    AgentEvent, AgentStep, PlanItem, PlanStatus, StepTool, host_of, program_of, text,
};

/// The events one stdout line carries.
pub fn read_events(line: &str) -> Vec<AgentEvent> {
    let Ok(message) = serde_json::from_str::<Value>(line.trim()) else {
        return Vec::new();
    };
    let parent = text(&message, "/parent_tool_use_id");
    match message.get("type").and_then(Value::as_str) {
        Some("system") if text(&message, "/subtype").as_deref() == Some("init") => {
            text(&message, "/session_id")
                .map(|id| AgentEvent::Session {
                    id,
                    model: text(&message, "/model"),
                    resumable: None,
                })
                .into_iter()
                .collect()
        }
        Some("assistant") => blocks(&message)
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            .flat_map(|block| tool_use(block, parent.clone()))
            .collect(),
        Some("user") => blocks(&message)
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
            .filter_map(|block| {
                Some(AgentEvent::StepEnded {
                    id: text(block, "/tool_use_id")?,
                    failed: block.get("is_error").and_then(Value::as_bool) == Some(true),
                })
            })
            .collect(),
        Some("result") => {
            let mut events = Vec::new();
            let count = |key: &str| {
                message
                    .pointer(&format!("/usage/{key}"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
            };
            if message.get("usage").is_some() || message.get("total_cost_usd").is_some() {
                events.push(AgentEvent::Usage {
                    input_tokens: count("input_tokens")
                        + count("cache_read_input_tokens")
                        + count("cache_creation_input_tokens"),
                    output_tokens: count("output_tokens"),
                    cost_usd: message.get("total_cost_usd").and_then(Value::as_f64),
                });
            }
            events.push(AgentEvent::TurnEnded {
                failed: message.get("is_error").and_then(Value::as_bool) == Some(true),
            });
            events
        }
        _ => Vec::new(),
    }
}

fn blocks(message: &Value) -> impl Iterator<Item = &Value> {
    message
        .pointer("/message/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

/// A `tool_use` block's step, and its plan when it writes one. The tool's
/// name is Claude Code's identifier for it (a protocol value).
fn tool_use(block: &Value, parent: Option<String>) -> Vec<AgentEvent> {
    let name = text(block, "/name").unwrap_or_default();
    let input = block.get("input").unwrap_or(&Value::Null);
    let field = |key: &str| text(input, &format!("/{key}"));
    let mut events = Vec::new();
    let tool = match name.as_str() {
        "Read" | "NotebookRead" => StepTool::Read {
            path: field("file_path").or_else(|| field("notebook_path")),
        },
        "Grep" | "Glob" => StepTool::Search {
            pattern: field("pattern"),
        },
        "LS" => StepTool::Search { pattern: None },
        "Edit" | "MultiEdit" | "Write" | "NotebookEdit" => StepTool::Edit {
            path: field("file_path").or_else(|| field("notebook_path")),
        },
        "Bash" => StepTool::Run {
            program: field("command").as_deref().and_then(program_of),
        },
        "WebFetch" => StepTool::Fetch {
            host: field("url").as_deref().and_then(host_of),
        },
        "WebSearch" => StepTool::Fetch { host: None },
        "Task" | "Agent" => StepTool::Delegate {
            what: field("description"),
        },
        "TodoWrite" => {
            let items = input
                .get("todos")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|todo| {
                    Some(PlanItem {
                        text: text(todo, "/content")?,
                        status: match todo.get("status").and_then(Value::as_str) {
                            Some("completed") => PlanStatus::Done,
                            Some("in_progress") => PlanStatus::InProgress,
                            _ => PlanStatus::Pending,
                        },
                    })
                })
                .collect();
            events.push(AgentEvent::Plan(items));
            StepTool::Plan
        }
        "" => return events,
        other => StepTool::Other {
            name: other.to_string(),
        },
    };
    events.insert(
        0,
        AgentEvent::Step(AgentStep {
            id: text(block, "/id"),
            parent,
            tool,
        }),
    );
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn its_session_steps_and_end_are_read_from_typed_fields() {
        assert_eq!(
            read_events(
                r#"{"type":"system","subtype":"init","session_id":"s-1","model":"synthetic-model"}"#
            ),
            vec![AgentEvent::Session {
                id: "s-1".into(),
                model: Some("synthetic-model".into()),
                resumable: None,
            }]
        );
        let steps = read_events(
            r#"{"type":"assistant","parent_tool_use_id":"t-0","message":{"content":[{"type":"text","text":"Let me run the tests with my token"},{"type":"tool_use","id":"t-1","name":"Bash","input":{"command":"API_TOKEN=abc cargo test -q"}},{"type":"tool_use","id":"t-2","name":"Edit","input":{"file_path":"/w/synthetic/src/lib.rs","old_string":"a","new_string":"b"}}]}}"#,
        );
        assert_eq!(
            steps,
            vec![
                AgentEvent::Step(AgentStep {
                    id: Some("t-1".into()),
                    parent: Some("t-0".into()),
                    tool: StepTool::Run {
                        program: Some("cargo".into())
                    },
                }),
                AgentEvent::Step(AgentStep {
                    id: Some("t-2".into()),
                    parent: Some("t-0".into()),
                    tool: StepTool::Edit {
                        path: Some("/w/synthetic/src/lib.rs".into())
                    },
                }),
            ]
        );
        assert_eq!(
            read_events(
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t-1","is_error":true,"content":"failed"}]}}"#
            ),
            vec![AgentEvent::StepEnded {
                id: "t-1".into(),
                failed: true
            }]
        );
        assert_eq!(
            read_events(
                r#"{"type":"result","subtype":"success","is_error":false,"total_cost_usd":0.25,"usage":{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":5}}"#
            ),
            vec![
                AgentEvent::Usage {
                    input_tokens: 100,
                    output_tokens: 5,
                    cost_usd: Some(0.25)
                },
                AgentEvent::TurnEnded { failed: false }
            ]
        );
    }

    #[test]
    fn its_todo_list_is_its_plan_and_an_unknown_tool_keeps_its_name() {
        let events = read_events(
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"p","name":"TodoWrite","input":{"todos":[{"content":"Bump the version","status":"completed"},{"content":"Tag it","status":"in_progress"}]}}]}}"#,
        );
        assert!(matches!(&events[0], AgentEvent::Step(step) if step.tool == StepTool::Plan));
        assert_eq!(
            events[1],
            AgentEvent::Plan(vec![
                PlanItem {
                    text: "Bump the version".into(),
                    status: PlanStatus::Done
                },
                PlanItem {
                    text: "Tag it".into(),
                    status: PlanStatus::InProgress
                },
            ])
        );
        let other = read_events(
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"m","name":"mcp__synthetic__lookup","input":{}}]}}"#,
        );
        assert!(
            matches!(&other[0], AgentEvent::Step(step) if step.tool == StepTool::Other { name: "mcp__synthetic__lookup".into() })
        );
        assert!(read_events("plain text").is_empty());
    }
}
