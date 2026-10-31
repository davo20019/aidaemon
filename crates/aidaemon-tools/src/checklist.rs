//! `track_requirements`: the model's own checklist, plus a stop hook that
//! refuses to finish while items are open.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use aidaemon_core::{
    Hook, HookContext, StopDecision, Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult,
    ToolSpec,
};

use crate::util::{fail, parse_args};

/// Most items a checklist may hold.
pub const MAX_ITEMS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    Pending,
    InProgress,
    Done,
}

impl ItemStatus {
    fn marker(self) -> &'static str {
        match self {
            Self::Pending => "[ ]",
            Self::InProgress => "[~]",
            Self::Done => "[x]",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChecklistItem {
    pub text: String,
    pub status: ItemStatus,
}

/// Shared checklist state. Clone it to share between the tool, the gate and
/// your own code. Use one per run or session.
#[derive(Debug, Clone, Default)]
pub struct ChecklistState {
    items: Arc<Mutex<Vec<ChecklistItem>>>,
}

impl ChecklistState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn items(&self) -> Vec<ChecklistItem> {
        self.items.lock().expect("checklist poisoned").clone()
    }

    pub fn open_items(&self) -> Vec<ChecklistItem> {
        self.items()
            .into_iter()
            .filter(|i| i.status != ItemStatus::Done)
            .collect()
    }

    pub fn set(&self, items: Vec<ChecklistItem>) {
        *self.items.lock().expect("checklist poisoned") = items;
    }
}

fn render(items: &[ChecklistItem]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| format!("{}. {} {}", i + 1, item.status.marker(), item.text))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `track_requirements` tool. Each call replaces the whole list, so the
/// model always states the full current plan.
#[derive(Debug, Clone, Default)]
pub struct Checklist {
    state: ChecklistState,
}

impl Checklist {
    pub fn new(state: ChecklistState) -> Self {
        Self { state }
    }

    /// A tool and its matching stop gate over fresh shared state.
    pub fn with_gate() -> (Self, ChecklistGate) {
        let state = ChecklistState::new();
        (Self::new(state.clone()), ChecklistGate::new(state))
    }

    pub fn state(&self) -> &ChecklistState {
        &self.state
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChecklistArgs {
    items: Vec<ChecklistItem>,
}

#[async_trait]
impl Tool for Checklist {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "track_requirements".into(),
            description: "Declare or update your checklist for the current task. Send the \
                 complete list every time (it replaces the previous one), each item with a \
                 status: pending, in_progress or done. Use it for multi-step work: list every \
                 requirement up front, keep one item in_progress, and mark an item done only \
                 after it is actually finished and verified. You cannot finish while items \
                 are open; if one turns out to be unnecessary, remove it and say why."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "maxItems": MAX_ITEMS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "What must be true when this item is done."},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "done"]}
                            },
                            "required": ["text", "status"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        // Changes only agent-internal state, but two updates in one turn
        // must not race, so it is not read-only.
        ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: true,
        }
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let args: ChecklistArgs = match parse_args("track_requirements", arguments) {
            Ok(a) => a,
            Err(e) => return Ok(fail(e)),
        };
        if args.items.len() > MAX_ITEMS {
            return Ok(fail(format!(
                "too many items ({}); at most {MAX_ITEMS}",
                args.items.len()
            )));
        }
        let mut items = args.items;
        for item in &mut items {
            item.text = item.text.trim().to_string();
        }
        if items.iter().any(|i| i.text.is_empty()) {
            return Ok(fail("every item needs non-empty `text`"));
        }
        self.state.set(items.clone());
        let open = items
            .iter()
            .filter(|i| i.status != ItemStatus::Done)
            .count();
        let done = items.len() - open;
        let summary = if items.is_empty() {
            "Checklist cleared.".to_string()
        } else {
            format!(
                "Checklist updated: {done}/{} done.\n{}",
                items.len(),
                render(&items)
            )
        };
        Ok(ToolOutput::text(summary).with_details(json!({
            "total": items.len(),
            "done": done,
            "open": open,
        })))
    }
}

/// Stop hook: while any checklist item is not done, ask the model to keep
/// going, listing the open items. Accepts once all are done (or the list is
/// empty). The agent's turn limits still bound how long this can continue.
#[derive(Debug, Clone)]
pub struct ChecklistGate {
    state: ChecklistState,
}

impl ChecklistGate {
    pub fn new(state: ChecklistState) -> Self {
        Self { state }
    }
}

#[async_trait]
impl Hook for ChecklistGate {
    async fn on_stop(&self, _ctx: &HookContext<'_>, _final_text: &str) -> StopDecision {
        let items = self.state.items();
        let open: Vec<&ChecklistItem> = items
            .iter()
            .filter(|i| i.status != ItemStatus::Done)
            .collect();
        if open.is_empty() {
            return StopDecision::Accept;
        }
        let list = open
            .iter()
            .map(|i| format!("- {} {}", i.status.marker(), i.text))
            .collect::<Vec<_>>()
            .join("\n");
        StopDecision::Continue {
            message: format!(
                "Your checklist still has {} open item(s):\n{list}\n\nFinish them and mark \
                 them done with track_requirements before giving your final answer. If an item \
                 is no longer needed, remove it from the list and explain why.",
                open.len()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider};
    use aidaemon_core::{Agent, Role};

    #[tokio::test]
    async fn tool_replaces_list_and_validates() {
        let tool = Checklist::default();
        let out = tool
            .call(
                json!({"items": [
                    {"text": "write parser", "status": "done"},
                    {"text": "add tests", "status": "in_progress"}
                ]}),
                &crate::testutil::ctx(),
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("1/2 done"));
        assert!(out.content.contains("2. [~] add tests"));
        assert_eq!(tool.state().open_items().len(), 1);

        let out = tool
            .call(
                json!({"items": [{"text": "x", "status": "finished"}]}),
                &crate::testutil::ctx(),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        // A rejected update leaves the previous list in place.
        assert_eq!(tool.state().items().len(), 2);
    }

    #[tokio::test]
    async fn gate_blocks_until_items_are_done() {
        let (tool, gate) = Checklist::with_gate();
        let provider = ScriptedProvider::new([
            tool_calls([call(
                "c1",
                "track_requirements",
                json!({"items": [
                    {"text": "update config", "status": "in_progress"},
                    {"text": "run tests", "status": "pending"}
                ]}),
            )]),
            // Tries to finish early; the gate pushes back.
            text("All done!"),
            tool_calls([call(
                "c2",
                "track_requirements",
                json!({"items": [
                    {"text": "update config", "status": "done"},
                    {"text": "run tests", "status": "done"}
                ]}),
            )]),
            text("Config updated and tests pass."),
        ]);
        let state = tool.state().clone();
        let agent = Agent::builder("checklist-test")
            .model("scripted")
            .provider(provider)
            .tool(tool)
            .hook(gate)
            .build()
            .unwrap();
        let outcome = agent
            .run("test:synthetic-user-1", "Do the task")
            .await
            .unwrap();
        assert!(outcome.is_completed(), "{:?}", outcome.status);
        assert_eq!(
            outcome.final_text.as_deref(),
            Some("Config updated and tests pass.")
        );
        let nudge = outcome
            .state
            .messages
            .iter()
            .find(|m| m.role == Role::User && m.content.contains("open item"))
            .expect("gate message in transcript");
        assert!(nudge.content.contains("update config") && nudge.content.contains("run tests"));
        assert!(state.open_items().is_empty());
    }
}
