//! A delegated coding agent's permission prompts, answered by the run that
//! delegated to it.
//!
//! Claude Code run with `--permission-prompt-tool stdio` and stream-json
//! input sends each permission prompt to its host as a typed
//! `control_request` (`subtype: "can_use_tool"`, the tool's name and input,
//! the path it would touch) and waits for a `control_response`. Each prompt
//! becomes a nested call of the daemon's own tool for that action (`Bash`
//! is `terminal`, `Write` is `write_file`, ...), judged by the run exactly
//! as the model's own calls are ([`aidaemon_core::NestedCalls`]): the
//! owner's grants answer what they cover, a card asks the rest, and a
//! decline binds the run. The prompt's own "remember this" suggestions are
//! never sent back: a grant stays the daemon's, never a rule written into
//! the agent's settings.
//!
//! Verified against Claude Code 2.1.327 (live probe, 2026-11-17).

use std::sync::{Arc, Mutex};

use aidaemon_core::{NestedCalls, NestedVerdict, ToolCall};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::process::ChildStdin;

/// The launch flags that make Claude Code ask its host: the prompt and the
/// host's answers come on stdin as stream-json, its prompts go out on
/// stdout.
pub const CLAUDE_HOST_FLAGS: [&str; 4] = [
    "--input-format",
    "stream-json",
    "--permission-prompt-tool",
    "stdio",
];

/// The first stdin line: the task, as a user message.
pub fn claude_prompt_line(prompt: &str) -> String {
    let mut line = json!({
        "type": "user",
        "message": {"role": "user", "content": prompt},
    })
    .to_string();
    line.push('\n');
    line
}

/// One permission prompt, read from a stdout line.
#[derive(Debug, Clone, PartialEq)]
pub struct PermissionPrompt {
    pub request_id: String,
    pub tool_name: String,
    pub input: Value,
}

/// What a stdout line means to the host.
#[derive(Debug, Clone, PartialEq)]
pub enum HostLine {
    /// A permission prompt to answer.
    Permission(PermissionPrompt),
    /// Another control request, which this host does not serve.
    OtherControl { request_id: String },
    /// The agent's final result: its turn is over.
    Result,
    /// Anything else (its own progress).
    Other,
}

/// Read one stdout line by its JSON fields.
pub fn read_line(line: &str) -> HostLine {
    let Ok(event) = serde_json::from_str::<Value>(line.trim()) else {
        return HostLine::Other;
    };
    match event.get("type").and_then(Value::as_str) {
        Some("control_request") => {
            let request_id = event
                .get("request_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let request = event.get("request");
            let subtype = request
                .and_then(|request| request.get("subtype"))
                .and_then(Value::as_str);
            match (subtype, request) {
                (Some("can_use_tool"), Some(request)) => HostLine::Permission(PermissionPrompt {
                    request_id,
                    tool_name: request
                        .get("tool_name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    input: request.get("input").cloned().unwrap_or(Value::Null),
                }),
                _ => HostLine::OtherControl { request_id },
            }
        }
        Some("result") => HostLine::Result,
        _ => HostLine::Other,
    }
}

/// The daemon call a prompt asks for: the daemon tool whose judgment
/// applies, with arguments its schema takes. `Err` names a prompt the
/// daemon has no tool for, which is refused.
pub fn daemon_call(
    prompt: &PermissionPrompt,
    working_dir: Option<&str>,
) -> Result<ToolCall, String> {
    let input = &prompt.input;
    let text = |key: &str| input.get(key).and_then(Value::as_str).map(str::to_string);
    let id = format!("cli-permission-{}", prompt.request_id);
    let missing = |field: &str| format!("its `{}` request names no {field}", prompt.tool_name);
    let arguments = match prompt.tool_name.as_str() {
        "Bash" => {
            let command = text("command").ok_or_else(|| missing("command"))?;
            let mut arguments = json!({"action": "run", "command": command});
            if let Some(dir) = working_dir {
                arguments["working_dir"] = json!(dir);
            }
            return Ok(ToolCall::new(id, "terminal", arguments));
        }
        "Write" => json!({
            "path": text("file_path").ok_or_else(|| missing("file"))?,
            "content": text("content").unwrap_or_default(),
        }),
        "Edit" => json!({
            "path": text("file_path").ok_or_else(|| missing("file"))?,
            "old_text": text("old_string").unwrap_or_default(),
            "new_text": text("new_string").unwrap_or_default(),
            "replace_all": input.get("replace_all").and_then(Value::as_bool).unwrap_or(false),
        }),
        "MultiEdit" => {
            let edits: Vec<Value> = input
                .get("edits")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|edit| {
                    json!({
                        "old_text": edit.get("old_string").and_then(Value::as_str).unwrap_or_default(),
                        "new_text": edit.get("new_string").and_then(Value::as_str).unwrap_or_default(),
                    })
                })
                .collect();
            json!({
                "path": text("file_path").ok_or_else(|| missing("file"))?,
                "edits": edits,
            })
        }
        "WebFetch" => json!({"url": text("url").ok_or_else(|| missing("URL"))?}),
        other => {
            return Err(format!(
                "AIDaemon has no rule for the coding agent's `{other}` tool, so it was not allowed"
            ));
        }
    };
    let tool = match prompt.tool_name.as_str() {
        "Write" => "write_file",
        "Edit" | "MultiEdit" => "edit_file",
        _ => "web_fetch",
    };
    Ok(ToolCall::new(id, tool, arguments))
}

/// Claude Code's answer line for `prompt`: allow with its input unchanged,
/// or deny with the run's reason. Never a permission update.
pub fn permission_response(prompt: &PermissionPrompt, verdict: &NestedVerdict) -> String {
    let answer = match verdict {
        NestedVerdict::Allow => json!({"behavior": "allow", "updatedInput": prompt.input}),
        other => json!({
            "behavior": "deny",
            "message": other.reason().unwrap_or("not allowed"),
        }),
    };
    response_line(
        &prompt.request_id,
        json!({"subtype": "success", "request_id": prompt.request_id, "response": answer}),
    )
}

/// The answer to a control request this host does not serve.
pub fn unsupported_response(request_id: &str) -> String {
    response_line(
        request_id,
        json!({"subtype": "error", "request_id": request_id, "error": "not supported by this host"}),
    )
}

fn response_line(_request_id: &str, response: Value) -> String {
    let mut line = json!({"type": "control_response", "response": response}).to_string();
    line.push('\n');
    line
}

/// What the host answered over a run, by the actions' typed names.
#[derive(Debug, Default, Clone)]
pub struct HostAnswers {
    /// Actions the owner declined.
    pub declined: Vec<String>,
}

/// Answers one Claude Code process's prompts for its run.
#[derive(Clone)]
pub struct ClaudeHost {
    stdin: Arc<tokio::sync::Mutex<Option<ChildStdin>>>,
    nested: NestedCalls,
    working_dir: Option<String>,
    answers: Arc<Mutex<HostAnswers>>,
}

impl ClaudeHost {
    pub fn new(stdin: ChildStdin, nested: NestedCalls, working_dir: Option<String>) -> Self {
        Self {
            stdin: Arc::new(tokio::sync::Mutex::new(Some(stdin))),
            nested,
            working_dir,
            answers: Arc::default(),
        }
    }

    /// What it answered so far.
    pub fn answers(&self) -> HostAnswers {
        self.answers.lock().map(|a| a.clone()).unwrap_or_default()
    }

    /// Handle one stdout line. A prompt is answered in its own task, so the
    /// agent's output keeps being read while the owner decides.
    pub async fn on_line(&self, line: &str) {
        match read_line(line) {
            HostLine::Permission(prompt) => {
                let host = self.clone();
                tokio::spawn(async move { host.answer(prompt).await });
            }
            HostLine::OtherControl { request_id } => {
                self.write(&unsupported_response(&request_id)).await;
            }
            // Its turn is over: closing stdin lets it exit.
            HostLine::Result => self.close().await,
            HostLine::Other => {}
        }
    }

    async fn answer(&self, prompt: PermissionPrompt) {
        let verdict = match daemon_call(&prompt, self.working_dir.as_deref()) {
            Ok(call) => self.nested.authorize(call).await,
            Err(reason) => NestedVerdict::Deny { reason },
        };
        if matches!(verdict, NestedVerdict::Declined { .. })
            && let Ok(mut answers) = self.answers.lock()
        {
            answers.declined.push(action_label(&prompt));
        }
        tracing::info!(
            tool = %prompt.tool_name,
            allowed = verdict.is_allowed(),
            "Answered a coding agent's permission prompt"
        );
        self.write(&permission_response(&prompt, &verdict)).await;
    }

    async fn write(&self, line: &str) {
        let mut stdin = self.stdin.lock().await;
        if let Some(pipe) = stdin.as_mut()
            && (pipe.write_all(line.as_bytes()).await.is_err() || pipe.flush().await.is_err())
        {
            *stdin = None;
        }
    }

    async fn close(&self) {
        self.stdin.lock().await.take();
    }
}

/// A prompt said as the owner reads it: the tool and what it targets.
fn action_label(prompt: &PermissionPrompt) -> String {
    let target = ["command", "file_path", "url"]
        .iter()
        .find_map(|key| prompt.input.get(*key).and_then(Value::as_str))
        .map(|target| target.split_whitespace().collect::<Vec<_>>().join(" "));
    match target {
        Some(target) if !target.is_empty() => format!(
            "{}: {}",
            prompt.tool_name,
            aidaemon_events::text::truncate_str(&target, 160)
        ),
        _ => prompt.tool_name.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prompt Claude Code 2.1.327 sent in the live probe.
    const PROBE: &str = r#"{"type":"control_request","request_id":"ae1fdf0e","request":{"subtype":"can_use_tool","tool_name":"Bash","display_name":"Bash","input":{"command":"touch probe.txt","description":"Create empty probe.txt"},"permission_suggestions":[{"type":"addRules","rules":[{"toolName":"Bash","ruleContent":"touch probe.txt"}],"behavior":"allow","destination":"localSettings"}],"blocked_path":"/w/synthetic/probe.txt","tool_use_id":"toolu_1"}}"#;

    fn probe() -> PermissionPrompt {
        let HostLine::Permission(prompt) = read_line(PROBE) else {
            panic!("a permission prompt");
        };
        prompt
    }

    #[test]
    fn a_permission_prompt_is_read_from_its_typed_fields() {
        let prompt = probe();
        assert_eq!(prompt.request_id, "ae1fdf0e");
        assert_eq!(prompt.tool_name, "Bash");
        assert_eq!(prompt.input["command"], "touch probe.txt");
        assert_eq!(
            read_line(
                r#"{"type":"control_request","request_id":"x","request":{"subtype":"hook_callback"}}"#
            ),
            HostLine::OtherControl {
                request_id: "x".into()
            }
        );
        assert_eq!(
            read_line(r#"{"type":"result","subtype":"success"}"#),
            HostLine::Result
        );
        assert_eq!(read_line(r#"{"type":"assistant"}"#), HostLine::Other);
        assert_eq!(read_line("may I run touch?"), HostLine::Other);
    }

    #[test]
    fn each_prompt_becomes_the_daemon_tool_whose_judgment_applies() {
        let bash = daemon_call(&probe(), Some("/w/synthetic")).unwrap();
        assert_eq!(bash.name, "terminal");
        assert_eq!(
            bash.arguments,
            json!({"action": "run", "command": "touch probe.txt", "working_dir": "/w/synthetic"})
        );
        let prompt = |tool: &str, input: Value| PermissionPrompt {
            request_id: "r".into(),
            tool_name: tool.into(),
            input,
        };
        let write = daemon_call(
            &prompt(
                "Write",
                json!({"file_path": "/w/synthetic/a.md", "content": "x"}),
            ),
            None,
        )
        .unwrap();
        assert_eq!(write.name, "write_file");
        assert_eq!(
            write.arguments,
            json!({"path": "/w/synthetic/a.md", "content": "x"})
        );
        let edit = daemon_call(
            &prompt(
                "Edit",
                json!({"file_path": "/w/synthetic/a.md", "old_string": "a", "new_string": "b"}),
            ),
            None,
        )
        .unwrap();
        assert_eq!(edit.name, "edit_file");
        assert_eq!(edit.arguments["old_text"], "a");
        assert_eq!(edit.arguments["new_text"], "b");
        let fetch = daemon_call(
            &prompt("WebFetch", json!({"url": "https://example.com"})),
            None,
        )
        .unwrap();
        assert_eq!(fetch.name, "web_fetch");
        assert!(daemon_call(&prompt("mcp__synthetic__send", json!({})), None).is_err());
        assert!(daemon_call(&prompt("Bash", json!({})), None).is_err());
    }

    #[test]
    fn an_answer_allows_or_denies_and_never_writes_a_rule() {
        let prompt = probe();
        let allow: Value =
            serde_json::from_str(permission_response(&prompt, &NestedVerdict::Allow).trim())
                .unwrap();
        assert_eq!(allow["type"], "control_response");
        assert_eq!(allow["response"]["request_id"], "ae1fdf0e");
        assert_eq!(allow["response"]["response"]["behavior"], "allow");
        assert_eq!(
            allow["response"]["response"]["updatedInput"]["command"],
            "touch probe.txt"
        );
        assert!(
            allow["response"]["response"]
                .get("updatedPermissions")
                .is_none()
        );
        let deny: Value = serde_json::from_str(
            permission_response(
                &prompt,
                &NestedVerdict::Declined {
                    reason: "[OWNER DECLINED] no".into(),
                },
            )
            .trim(),
        )
        .unwrap();
        assert_eq!(deny["response"]["response"]["behavior"], "deny");
        assert_eq!(
            deny["response"]["response"]["message"],
            "[OWNER DECLINED] no"
        );
    }

    #[test]
    fn the_prompt_goes_in_as_a_user_message() {
        let line: Value = serde_json::from_str(claude_prompt_line("Fix the tests").trim()).unwrap();
        assert_eq!(line["type"], "user");
        assert_eq!(line["message"]["content"], "Fix the tests");
    }
}
