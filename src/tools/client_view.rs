//! `client_view`: the agent moves the owner's app view (the agent-driven
//! view of the Client Gateway). "Show me what's waiting" or "open
//! yesterday's deploy session" in the TUI or the web client, and the client
//! goes there with a short note of what moved and why.
//!
//! Offered only on a turn whose message came from an app connection that
//! declared the `view_actions` capability ([`ToolSemantics::offered_in_session`]
//! asks the gateway), and the action goes to that connection only.
//!
//! View actions only. The schema's `action` enum names places on screen;
//! nothing here can answer an approval, pair or revoke a device, send a
//! message or stop work: those stay commands the owner issues.

use std::sync::Arc;

use aidaemon_client_protocol::{Pane, ViewAction};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{
    Tool, ToolCallContext, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolRole,
    ToolSemantics,
};

/// Where view actions go: the running Client Gateway, or a test double.
pub trait ClientViews: Send + Sync {
    /// Whether a turn of `session_id` has an app connection it may move.
    fn can_show(&self, session_id: &str) -> bool;
    /// Move that connection's view.
    fn show(&self, session_id: &str, action: ViewAction, reason: &str) -> Result<(), String>;
}

pub struct ClientViewTool {
    views: Arc<dyn ClientViews>,
}

impl ClientViewTool {
    pub fn new(views: Arc<dyn ClientViews>) -> Self {
        Self { views }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    action: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    approval_id: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    pane: Option<Pane>,
    #[serde(default)]
    reason: String,
}

fn required(value: Option<String>, field: &str, action: &str) -> anyhow::Result<String> {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("`{action}` needs `{field}`"))
}

/// The view action the arguments name.
fn view_action(args: &Args) -> anyhow::Result<ViewAction> {
    Ok(match args.action.as_str() {
        "open_session" => ViewAction::OpenSession {
            session_id: required(args.session_id.clone(), "session_id", "open_session")?,
        },
        "show_approval" => ViewAction::ShowApproval {
            approval_id: required(args.approval_id.clone(), "approval_id", "show_approval")?,
        },
        "show_task" => ViewAction::ShowTask {
            task_id: required(args.task_id.clone(), "task_id", "show_task")?,
        },
        "focus_pane" => ViewAction::FocusPane {
            pane: match args.pane {
                Some(pane) if pane != Pane::Unknown => pane,
                _ => anyhow::bail!("`focus_pane` needs `pane`: sessions, chat, approvals or tasks"),
            },
        },
        "new_session" => ViewAction::NewSession,
        "show_devices" => ViewAction::ShowDevices,
        other => anyhow::bail!("`{other}` is not a view action"),
    })
}

#[async_trait]
impl Tool for ClientViewTool {
    fn name(&self) -> &str {
        "client_view"
    }

    fn description(&self) -> &str {
        "Move the owner's app view (the TUI or web client they are typing in): open a session, \
         bring a waiting approval's card into view, show a piece of running work, focus a pane, \
         start a new session or show the paired devices. Use it when the owner asks to see or go \
         somewhere in the app. It only changes what is on their screen: it cannot answer an \
         approval, send a message or stop anything."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "client_view",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open_session", "show_approval", "show_task", "focus_pane", "new_session", "show_devices"],
                        "description": "Where to take the view"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "For open_session: the session's id (app:..., telegram:...)"
                    },
                    "approval_id": {
                        "type": "string",
                        "description": "For show_approval: the waiting approval's id"
                    },
                    "task_id": {
                        "type": "string",
                        "description": "For show_task: the running-work entry's id"
                    },
                    "pane": {
                        "type": "string",
                        "enum": ["sessions", "chat", "approvals", "tasks"],
                        "description": "For focus_pane"
                    },
                    "reason": {
                        "type": "string",
                        "description": "A few words for the note the app shows, e.g. \"You asked what is waiting\""
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Args = serde_json::from_str(arguments)?;
        let action = view_action(&args)?;
        let session = if facts.session_id.is_empty() {
            facts.channel_id.clone().unwrap_or_default()
        } else {
            facts.session_id.clone()
        };
        self.views
            .show(&session, action, &args.reason)
            .map_err(|why| anyhow::anyhow!("The app's view could not be moved: {why}"))?;
        Ok(ToolCallOutcome::from_output(format!(
            "The app now shows it ({}). The owner sees a note saying so and can go back.",
            args.action
        )))
    }
}

impl ToolSemantics for ClientViewTool {
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::administrative()
    }

    /// A local, reversible view change on the owner's own client: no file,
    /// process, credential or outside host is touched, and the owner's next
    /// key undoes it. So it declares no effect a decline could bind.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::none()
    }

    fn call_annotations_in(
        &self,
        _arguments: &str,
        _roots: &aidaemon_tools::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        aidaemon_core::ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: true,
            egress: false,
        }
    }

    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &aidaemon_tools::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Trusted
    }

    fn offered_in_session(&self, session_id: &str) -> bool {
        self.views.can_show(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Views {
        target: Option<String>,
        shown: Mutex<Vec<(String, ViewAction, String)>>,
    }

    impl ClientViews for Views {
        fn can_show(&self, session_id: &str) -> bool {
            self.target.as_deref() == Some(session_id)
        }
        fn show(&self, session_id: &str, action: ViewAction, reason: &str) -> Result<(), String> {
            if !self.can_show(session_id) {
                return Err("no client".into());
            }
            self.shown
                .lock()
                .unwrap()
                .push((session_id.into(), action, reason.into()));
            Ok(())
        }
    }

    fn facts(session: &str) -> ToolCallContext {
        ToolCallContext {
            session_id: session.into(),
            ..ToolCallContext::default()
        }
    }

    #[tokio::test]
    async fn it_is_offered_only_where_a_client_can_be_moved_and_moves_only_it() {
        let views = Arc::new(Views {
            target: Some("app:s1".into()),
            ..Views::default()
        });
        let tool = ClientViewTool::new(views.clone());
        assert!(tool.offered_in_session("app:s1"));
        assert!(!tool.offered_in_session("telegram:synthetic-user-1"));
        tool.execute(
            r#"{"action":"show_approval","approval_id":"a1","reason":"You asked what is waiting"}"#,
            &facts("app:s1"),
        )
        .await
        .unwrap();
        assert_eq!(
            views.shown.lock().unwrap().as_slice(),
            &[(
                "app:s1".to_string(),
                ViewAction::ShowApproval {
                    approval_id: "a1".into()
                },
                "You asked what is waiting".to_string()
            )]
        );
        assert!(tool
            .execute(
                r#"{"action":"new_session"}"#,
                &facts("telegram:synthetic-user-1")
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn only_view_actions_are_accepted() {
        let tool = ClientViewTool::new(Arc::new(Views {
            target: Some("app:s1".into()),
            ..Views::default()
        }));
        for bad in [
            r#"{"action":"resolve_approval","approval_id":"a1"}"#,
            r#"{"action":"show_approval","approval_id":"a1","decision":"approve_once"}"#,
            r#"{"action":"send_message","text":"hi"}"#,
            r#"{"action":"open_session"}"#,
            r#"{"action":"focus_pane","pane":"holodeck"}"#,
        ] {
            assert!(
                tool.execute(bad, &facts("app:s1")).await.is_err(),
                "{bad} was accepted"
            );
        }
        // The schema offers view kinds and nothing else.
        let kinds: Vec<String> = tool.schema()["parameters"]["properties"]["action"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            kinds,
            [
                "open_session",
                "show_approval",
                "show_task",
                "focus_pane",
                "new_session",
                "show_devices"
            ]
        );
    }

    #[test]
    fn it_declares_a_local_view_change() {
        let tool = ClientViewTool::new(Arc::new(Views::default()));
        assert_eq!(
            tool.call_effects(r#"{"action":"new_session"}"#),
            aidaemon_core::CallEffects::none()
        );
        assert!(!tool.capabilities().external_side_effect);
    }
}
