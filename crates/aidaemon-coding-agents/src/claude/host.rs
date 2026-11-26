//! Claude Code's permission prompts, answered by its host.
//!
//! Run with `--permission-prompt-tool stdio` and stream-json input, Claude
//! Code sends each permission prompt as a typed `control_request`
//! (`subtype: "can_use_tool"`, the tool's name and input) and waits for a
//! `control_response`. Each prompt becomes an [`AgentAction`] for the
//! judge. The prompt's own "remember this" suggestions are never sent
//! back: a grant stays the judge's, never a rule written into the agent's
//! settings.
//!
//! Verified against Claude Code 2.1.327 (live probes, 2026-11-17/18).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::ChildStdin;

use crate::stdin::Stdin;
use crate::{ActionJudge, AgentAction, AgentHost, HostAnswers, HostStarter, TextEdit, Verdict};

/// The launch flags that make Claude Code ask its host: the task and the
/// host's answers come on stdin as stream-json, its prompts go out on
/// stdout.
pub const HOST_FLAGS: [&str; 4] = [
    "--input-format",
    "stream-json",
    "--permission-prompt-tool",
    "stdio",
];

/// The first stdin line: the task, as a user message.
pub fn prompt_line(prompt: &str) -> String {
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

/// The action a prompt asks for. `Err` names a prompt with no action this
/// crate knows, which is refused.
pub fn action(prompt: &PermissionPrompt, working_dir: Option<&str>) -> Result<AgentAction, String> {
    let input = &prompt.input;
    let text = |key: &str| input.get(key).and_then(Value::as_str).map(str::to_string);
    let missing = |field: &str| format!("its `{}` request names no {field}", prompt.tool_name);
    Ok(match prompt.tool_name.as_str() {
        "Bash" => {
            // Leaving its sandbox is never granted, whatever the command
            // (its launch settings forbid it too).
            if input
                .get("dangerouslyDisableSandbox")
                .and_then(Value::as_bool)
                == Some(true)
            {
                return Err(
                    "a command may not run outside the coding agent's sandbox here".to_string(),
                );
            }
            AgentAction::RunCommand {
                command: text("command").ok_or_else(|| missing("command"))?,
                working_dir: working_dir.map(str::to_string),
            }
        }
        "Write" => AgentAction::WriteFile {
            path: text("file_path").ok_or_else(|| missing("file"))?,
            content: text("content").unwrap_or_default(),
        },
        "Edit" => AgentAction::EditFile {
            path: text("file_path").ok_or_else(|| missing("file"))?,
            edits: vec![TextEdit {
                old_text: text("old_string").unwrap_or_default(),
                new_text: text("new_string").unwrap_or_default(),
            }],
            replace_all: input
                .get("replace_all")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        "MultiEdit" => AgentAction::EditFile {
            path: text("file_path").ok_or_else(|| missing("file"))?,
            edits: input
                .get("edits")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|edit| TextEdit {
                    old_text: edit
                        .get("old_string")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    new_text: edit
                        .get("new_string")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                })
                .collect(),
            replace_all: false,
        },
        "Read" => AgentAction::ReadFile {
            path: text("file_path").ok_or_else(|| missing("file"))?,
        },
        "Grep" => AgentAction::Search {
            pattern: Some(text("pattern").ok_or_else(|| missing("pattern"))?),
            glob: None,
            path: text("path"),
        },
        "Glob" => AgentAction::Search {
            pattern: None,
            glob: Some(text("pattern").ok_or_else(|| missing("pattern"))?),
            path: text("path"),
        },
        "WebFetch" => AgentAction::Fetch {
            url: text("url").ok_or_else(|| missing("URL"))?,
        },
        // Its sandbox asks before a command reaches a network host.
        "SandboxNetworkAccess" => AgentAction::ReachHost {
            host: text("host").ok_or_else(|| missing("host"))?,
        },
        other => {
            return Err(format!(
                "AIDaemon has no rule for the coding agent's `{other}` tool, so it was not allowed"
            ));
        }
    })
}

/// Claude Code's answer line for `prompt`: allow with its input unchanged,
/// or deny with the judge's reason. Never a permission update.
pub fn permission_response(prompt: &PermissionPrompt, verdict: &Verdict) -> String {
    let answer = match verdict {
        Verdict::Allow => json!({"behavior": "allow", "updatedInput": prompt.input}),
        other => json!({
            "behavior": "deny",
            "message": other.reason().unwrap_or("not allowed"),
        }),
    };
    response_line(json!({
        "subtype": "success",
        "request_id": prompt.request_id,
        "response": answer,
    }))
}

/// The answer to a control request this host does not serve.
pub fn unsupported_response(request_id: &str) -> String {
    response_line(json!({
        "subtype": "error",
        "request_id": request_id,
        "error": "not supported by this host",
    }))
}

fn response_line(response: Value) -> String {
    let mut line = json!({"type": "control_response", "response": response}).to_string();
    line.push('\n');
    line
}

/// A control request asking Claude Code to interrupt its active turn.
pub fn interrupt_line() -> String {
    let mut line = json!({
        "type": "control_request",
        "request_id": "aidaemon-interrupt",
        "request": {"subtype": "interrupt"},
    })
    .to_string();
    line.push('\n');
    line
}

/// Starts a [`ClaudeHost`] for a run in `working_dir`.
pub struct ClaudeHostStarter {
    pub working_dir: Option<String>,
    /// Host-injected MCP server names whose tools (`mcp__<server>__*`) are
    /// authorized directly by the host (the host's MCP server enforces its own
    /// authority and approval checks).
    pub trusted_mcp_servers: Vec<String>,
}

#[async_trait]
impl HostStarter for ClaudeHostStarter {
    /// The task goes in as the first stream-json message.
    async fn start(
        self: Box<Self>,
        stdin: ChildStdin,
        judge: Arc<dyn ActionJudge>,
        prompt: &str,
    ) -> Arc<dyn AgentHost> {
        let host = ClaudeHost::new(stdin, judge, self.working_dir)
            .with_trusted_mcp_servers(self.trusted_mcp_servers);
        host.stdin.write(&prompt_line(prompt)).await;
        Arc::new(host)
    }
}

/// Whether `tool_name` is an MCP tool (`mcp__<server>__<tool>`, Claude Code's
/// protocol naming) belonging to a host-injected server in `trusted`.
pub fn is_trusted_mcp_tool(tool_name: &str, trusted: &[String]) -> bool {
    let Some(rest) = tool_name.strip_prefix("mcp__") else {
        return false;
    };
    let Some((server, tool)) = rest.split_once("__") else {
        return false;
    };
    !tool.is_empty() && trusted.iter().any(|allowed| allowed == server)
}

/// Answers one Claude Code process's prompts.
#[derive(Clone)]
pub struct ClaudeHost {
    stdin: Stdin,
    judge: Arc<dyn ActionJudge>,
    working_dir: Option<String>,
    trusted_mcp_servers: Arc<[String]>,
    answers: Arc<Mutex<HostAnswers>>,
    /// Set once the turn's final `result` arrives or the host closes stdin.
    finished: Arc<AtomicBool>,
}

impl ClaudeHost {
    pub fn new(
        stdin: ChildStdin,
        judge: Arc<dyn ActionJudge>,
        working_dir: Option<String>,
    ) -> Self {
        Self {
            stdin: Stdin::new(stdin),
            judge,
            working_dir,
            trusted_mcp_servers: Arc::from(Vec::<String>::new()),
            answers: Arc::default(),
            finished: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Authorize MCP tool prompts belonging to host-injected servers.
    pub fn with_trusted_mcp_servers(mut self, servers: Vec<String>) -> Self {
        self.trusted_mcp_servers = Arc::from(servers);
        self
    }

    async fn answer(&self, prompt: PermissionPrompt) {
        let (label, verdict) = if is_trusted_mcp_tool(&prompt.tool_name, &self.trusted_mcp_servers)
        {
            (prompt.tool_name.clone(), Verdict::Allow)
        } else {
            match action(&prompt, self.working_dir.as_deref()) {
                Ok(action) => (action.label(), self.judge.judge(action).await),
                Err(reason) => (prompt.tool_name.clone(), Verdict::Deny { reason }),
            }
        };
        // Claude Code reports what it was refused itself
        // (`permission_denials`); a decline is kept here.
        if matches!(verdict, Verdict::Declined { .. })
            && let Ok(mut answers) = self.answers.lock()
        {
            answers.declined.push(label);
        }
        tracing::info!(
            tool = %prompt.tool_name,
            allowed = verdict.is_allowed(),
            "Answered a coding agent's permission prompt"
        );
        self.stdin
            .write(&permission_response(&prompt, &verdict))
            .await;
    }
}

#[async_trait]
impl AgentHost for ClaudeHost {
    async fn on_line(&self, line: &str) -> Vec<String> {
        match read_line(line) {
            HostLine::Permission(prompt) => {
                let host = self.clone();
                tokio::spawn(async move { host.answer(prompt).await });
            }
            HostLine::OtherControl { request_id } => {
                self.stdin.write(&unsupported_response(&request_id)).await;
            }
            // Its turn is over: closing stdin lets it exit.
            HostLine::Result => {
                self.finished.store(true, Ordering::Release);
                self.stdin.close().await;
            }
            HostLine::Other => {}
        }
        vec![line.to_string()]
    }

    fn answers(&self) -> HostAnswers {
        self.answers.lock().map(|a| a.clone()).unwrap_or_default()
    }

    fn accepts_mid_run(&self) -> bool {
        !self.finished.load(Ordering::Acquire)
    }

    async fn send_note(&self, note: &str) -> Result<(), String> {
        if self.finished.load(Ordering::Acquire) {
            return Err("Claude Code's turn has already finished".to_string());
        }
        if !self.stdin.try_write(&prompt_line(note)).await {
            self.finished.store(true, Ordering::Release);
            return Err("Claude Code's input stream is closed".to_string());
        }
        // Re-check after flushing: if `HostLine::Result` raced and arrived
        // while we were writing, report `Err` so the caller falls back to a
        // queued resumed follow-up rather than assuming the turn consumed it.
        if self.finished.load(Ordering::Acquire) {
            return Err("Claude Code's turn finished while sending the note".to_string());
        }
        Ok(())
    }

    async fn interrupt(&self) {
        self.stdin.write(&interrupt_line()).await;
    }

    async fn close(&self) {
        self.finished.store(true, Ordering::Release);
        self.stdin.close().await;
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

    fn prompt(tool: &str, input: Value) -> PermissionPrompt {
        PermissionPrompt {
            request_id: "r".into(),
            tool_name: tool.into(),
            input,
        }
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
    fn each_prompt_becomes_the_action_it_asks_for() {
        assert_eq!(
            action(&probe(), Some("/w/synthetic")).unwrap(),
            AgentAction::RunCommand {
                command: "touch probe.txt".into(),
                working_dir: Some("/w/synthetic".into()),
            }
        );
        assert_eq!(
            action(
                &prompt("Read", json!({"file_path": "/etc/synthetic.conf"})),
                None
            )
            .unwrap(),
            AgentAction::ReadFile {
                path: "/etc/synthetic.conf".into()
            }
        );
        assert_eq!(
            action(
                &prompt("Grep", json!({"pattern": "TODO", "path": "/w/other"})),
                None
            )
            .unwrap(),
            AgentAction::Search {
                pattern: Some("TODO".into()),
                glob: None,
                path: Some("/w/other".into()),
            }
        );
        assert_eq!(
            action(&prompt("Glob", json!({"pattern": "*.rs"})), None).unwrap(),
            AgentAction::Search {
                pattern: None,
                glob: Some("*.rs".into()),
                path: None,
            }
        );
        assert_eq!(
            action(
                &prompt(
                    "Write",
                    json!({"file_path": "/w/synthetic/a.md", "content": "x"})
                ),
                None
            )
            .unwrap(),
            AgentAction::WriteFile {
                path: "/w/synthetic/a.md".into(),
                content: "x".into(),
            }
        );
        let AgentAction::EditFile { edits, .. } = action(
            &prompt(
                "Edit",
                json!({"file_path": "/w/synthetic/a.md", "old_string": "a", "new_string": "b"}),
            ),
            None,
        )
        .unwrap() else {
            panic!("an edit");
        };
        assert_eq!(edits[0].old_text, "a");
        assert_eq!(edits[0].new_text, "b");
        assert_eq!(
            action(
                &prompt("WebFetch", json!({"url": "https://example.com"})),
                None
            )
            .unwrap(),
            AgentAction::Fetch {
                url: "https://example.com".into()
            }
        );
        assert_eq!(
            action(
                &prompt("SandboxNetworkAccess", json!({"host": "example.com"})),
                None
            )
            .unwrap(),
            AgentAction::ReachHost {
                host: "example.com".into()
            }
        );
        // Asking to leave its sandbox is refused before anything is judged.
        assert!(
            action(
                &prompt(
                    "Bash",
                    json!({"command": "touch /w/x", "dangerouslyDisableSandbox": true})
                ),
                None
            )
            .is_err()
        );
        assert!(action(&prompt("mcp__synthetic__send", json!({})), None).is_err());
        assert!(action(&prompt("Bash", json!({})), None).is_err());
    }

    #[test]
    fn an_answer_allows_or_denies_and_never_writes_a_rule() {
        let prompt = probe();
        let allow: Value =
            serde_json::from_str(permission_response(&prompt, &Verdict::Allow).trim()).unwrap();
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
                &Verdict::Declined {
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
        let line: Value = serde_json::from_str(prompt_line("Fix the tests").trim()).unwrap();
        assert_eq!(line["type"], "user");
        assert_eq!(line["message"]["content"], "Fix the tests");
    }

    /// Declines every command, allows the rest, and keeps what it saw.
    struct Judge(Mutex<Vec<AgentAction>>);

    #[async_trait]
    impl ActionJudge for Judge {
        async fn judge(&self, action: AgentAction) -> Verdict {
            let command = matches!(action, AgentAction::RunCommand { .. });
            self.0.lock().unwrap().push(action);
            if command {
                Verdict::Declined {
                    reason: "[OWNER DECLINED] not today".into(),
                }
            } else {
                Verdict::Allow
            }
        }
    }

    /// A stand-in for Claude Code over real pipes: it reads its task, asks
    /// once, echoes the host's answer, ends its turn, then reports whether
    /// its stdin closed.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_prompt_is_answered_over_the_pipe_and_the_turn_end_closes_stdin() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
        let script = r#"read task
echo '{"type":"control_request","request_id":"r1","request":{"subtype":"can_use_tool","tool_name":"Bash","input":{"command":"touch notes.txt"}}}'
read answer
printf '{"type":"echo","answer":%s}\n' "$answer"
echo '{"type":"result","subtype":"success"}'
if read more; then echo '{"type":"stdin_open"}'; else echo '{"type":"stdin_closed"}'; fi
"#;
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        stdin
            .write_all(prompt_line("Create notes.txt").as_bytes())
            .await
            .unwrap();
        let judge = Arc::new(Judge(Mutex::default()));
        let host = ClaudeHost::new(stdin, judge.clone(), Some("/w/synthetic".into()));
        let mut lines = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let mut seen = Vec::new();
        while let Some(line) = lines.next_line().await.unwrap() {
            assert_eq!(host.on_line(&line).await, vec![line.clone()]);
            seen.push(serde_json::from_str::<Value>(&line).unwrap());
        }
        child.wait().await.unwrap();
        let echo = seen.iter().find(|event| event["type"] == "echo").unwrap();
        assert_eq!(echo["answer"]["type"], "control_response");
        assert_eq!(echo["answer"]["response"]["request_id"], "r1");
        assert_eq!(echo["answer"]["response"]["response"]["behavior"], "deny");
        assert!(seen.iter().any(|event| event["type"] == "stdin_closed"));
        assert_eq!(
            judge.0.lock().unwrap()[0],
            AgentAction::RunCommand {
                command: "touch notes.txt".into(),
                working_dir: Some("/w/synthetic".into()),
            }
        );
        assert_eq!(
            host.answers().declined,
            vec!["command: touch notes.txt".to_string()]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn interrupt_writes_a_claude_interrupt_control_request() {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};
        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("cat");
        let host = ClaudeHost::new(child.stdin.take().unwrap(), Arc::new(crate::AllowAll), None);
        let mut sent = BufReader::new(child.stdout.take().unwrap()).lines();
        host.interrupt().await;
        let line: Value = serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(line["type"], "control_request");
        assert_eq!(line["request"]["subtype"], "interrupt");
        host.close().await;
        let _ = child.wait().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn send_note_writes_a_user_message_while_running_and_refuses_after_result() {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};
        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("cat");
        let host = ClaudeHost::new(child.stdin.take().unwrap(), Arc::new(crate::AllowAll), None);
        let mut sent = BufReader::new(child.stdout.take().unwrap()).lines();
        assert!(host.accepts_mid_run());
        host.send_note("also update the synthetic test")
            .await
            .expect("note accepted");
        let line: Value = serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(line["type"], "user");
        assert_eq!(line["message"]["role"], "user");
        assert_eq!(line["message"]["content"], "also update the synthetic test");

        // Once the turn finishes, mid-run notes are refused.
        host.on_line(r#"{"type":"result","result":"done"}"#).await;
        assert!(!host.accepts_mid_run());
        assert!(host.send_note("too late").await.is_err());
        let _ = child.wait().await;
    }
}
