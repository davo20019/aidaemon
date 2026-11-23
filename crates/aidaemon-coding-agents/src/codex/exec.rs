//! A `codex exec` run read through a host of its own, which answers
//! nothing: `exec` asks no one. It passes every line on as `exec` wrote it
//! and adds, to the line that ends the turn (`turn.completed`,
//! `turn.failed`), the turn's final message: the last `agent_message` item
//! it completed ([`super::events::FINAL_MESSAGE`]). A line alone can't say
//! which message is the last; the run's reader reads its report from there.
//!
//! The task stays its last argument. Its stdin is closed at once: `exec`
//! reads more of its task from a stdin that is not a terminal until it
//! ends.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;
use tokio::process::ChildStdin;

use crate::{ActionJudge, AgentHost, HostAnswers, HostStarter};

/// Starts an [`ExecHost`].
pub struct ExecHostStarter;

#[async_trait]
impl HostStarter for ExecHostStarter {
    /// Its stdin closes here; the task is already its last argument.
    async fn start(
        self: Box<Self>,
        stdin: ChildStdin,
        _judge: Arc<dyn ActionJudge>,
        _prompt: &str,
    ) -> Arc<dyn AgentHost> {
        drop(stdin);
        Arc::new(ExecHost::default())
    }
}

/// Reads one `codex exec --json` run's output.
#[derive(Default)]
pub struct ExecHost {
    /// The last message it completed (an `agent_message` item).
    last_message: Mutex<Option<String>>,
}

impl ExecHost {
    /// `line` as the run's reader gets it.
    pub fn read(&self, line: &str) -> String {
        let Ok(mut event) = serde_json::from_str::<Value>(line.trim()) else {
            return line.to_string();
        };
        // Its event and item types (a protocol's typed values).
        match event.get("type").and_then(Value::as_str) {
            Some("item.completed")
                if event.pointer("/item/type").and_then(Value::as_str) == Some("agent_message") =>
            {
                if let (Some(text), Ok(mut last)) = (
                    event.pointer("/item/text").and_then(Value::as_str),
                    self.last_message.lock(),
                ) {
                    *last = Some(text.to_string());
                }
                line.to_string()
            }
            Some("turn.completed" | "turn.failed") => {
                let message = self
                    .last_message
                    .lock()
                    .ok()
                    .and_then(|mut last| last.take())
                    .unwrap_or_default();
                if let Some(object) = event.as_object_mut() {
                    object.insert(
                        super::events::FINAL_MESSAGE.to_string(),
                        Value::String(message),
                    );
                }
                event.to_string()
            }
            _ => line.to_string(),
        }
    }
}

#[async_trait]
impl AgentHost for ExecHost {
    async fn on_line(&self, line: &str) -> Vec<String> {
        vec![self.read(line)]
    }

    fn answers(&self) -> HostAnswers {
        HostAnswers::default()
    }

    async fn close(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentEvent, Report};

    #[test]
    fn its_lines_pass_as_written_and_its_turns_end_carries_the_last_message() {
        let host = ExecHost::default();
        for line in [
            r#"{"type":"thread.started","thread_id":"th-1"}"#,
            "not json",
            r#"{"type":"item.completed","item":{"id":"i0","type":"agent_message","text":"I will list the files."}}"#,
            r#"{"type":"item.completed","item":{"id":"i1","type":"command_execution","command":"ls","exit_code":0,"status":"completed"}}"#,
            r#"{"type":"item.completed","item":{"id":"i2","type":"agent_message","text":"There are 4 files."}}"#,
        ] {
            assert_eq!(host.read(line), line);
        }
        let end: Value = serde_json::from_str(
            &host.read(r#"{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":3}}"#),
        )
        .unwrap();
        assert_eq!(end["final_message"], "There are 4 files.");
        assert_eq!(end["usage"]["input_tokens"], 7);
        let events = super::super::events::read_events(&end.to_string());
        assert_eq!(
            events[1],
            AgentEvent::Reported {
                text: "There are 4 files.".into(),
                report: None
            }
        );
    }

    #[test]
    fn a_turn_held_to_the_schema_ends_with_its_report() {
        let host = ExecHost::default();
        // As `codex exec --output-schema` writes it (live probe, 0.170.0).
        host.read(r#"{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"I will run the tests.\n"}}"#);
        host.read(r#"{"type":"item.completed","item":{"id":"item_3","type":"agent_message","text":"{\"summary\":\"Added a test.\",\"changed_files\":[\"tests/test_slug.py\"],\"checks\":[{\"command\":\"pytest -q\",\"passed\":true}],\"open_questions\":[],\"needs_owner\":false}"}}"#);
        let end = host.read(r#"{"type":"turn.completed","usage":{}}"#);
        let reported: Vec<AgentEvent> = super::super::events::read_events(&end)
            .into_iter()
            .filter(|event| matches!(event, AgentEvent::Reported { .. }))
            .collect();
        let AgentEvent::Reported { text, report } = &reported[0] else {
            unreachable!()
        };
        assert_eq!(reported.len(), 1);
        assert_eq!(text, "Added a test.");
        let report: &Report = report.as_ref().expect("a report");
        assert_eq!(report.changed_files, ["tests/test_slug.py"]);
        assert!(report.checks[0].passed);
        // A failed turn ends with what it last said, if anything.
        let failed = host.read(r#"{"type":"turn.failed","error":{"message":"x"}}"#);
        assert_eq!(
            super::super::events::read_events(&failed),
            vec![
                AgentEvent::Reported {
                    text: String::new(),
                    report: None
                },
                AgentEvent::TurnEnded { failed: true }
            ]
        );
    }
}
