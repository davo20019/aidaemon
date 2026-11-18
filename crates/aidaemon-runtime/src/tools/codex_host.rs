//! A delegated Codex run's approval requests, answered by the run that
//! delegated to it.
//!
//! `codex exec` cannot ask anyone: run confined with `approval_policy =
//! "never"`, a command at the edge of its sandbox just fails inside it, and
//! nothing typed says which action was refused. `codex app-server` is the
//! same agent behind a JSON-RPC protocol on stdin/stdout, and it sends its
//! host a typed request before it leaves its sandbox:
//! `item/commandExecution/requestApproval` (a command to run outside it, or
//! a network host), `item/fileChange/requestApproval` (a patch outside its
//! writable roots) and `item/permissions/requestApproval` (wider access for
//! the rest of the turn). Each becomes a nested call of the daemon's own
//! tool for that action, judged by the run exactly as the model's own calls
//! are ([`aidaemon_core::NestedCalls`]). A decision that would stay with the
//! agent ("for this session", a policy amendment) is never sent back: a
//! grant stays the daemon's.
//!
//! The process runs with its own `CODEX_HOME` ([`CodexHome`]), holding only
//! a link to the owner's sign-in: the owner's `config.toml` (its sandbox
//! mode and approval policy), its command rules, MCP servers, plugins and
//! trusted projects never reach a delegated run.
//!
//! Its notifications are translated into the `codex exec --json` events the
//! rest of `cli_agent` already reads ([`translate`]), so the run's result,
//! progress and end are judged the same way for both launch modes.
//!
//! Verified against Codex CLI 0.170.0 (live probes, 2026-11-18).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use aidaemon_core::{NestedCalls, NestedVerdict, ToolCall};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader};
use tokio::process::ChildStdin;

use crate::tools::cli_host::HostAnswers;

/// Features turned off in a hosted run: each reaches outside the sandbox
/// without a request this host sees (connectors, plugins, a browser, the
/// desktop, lifecycle hooks). Set with `-c features.<name>=false`, which a
/// Codex that does not know a name ignores (`--disable` would refuse to
/// start).
const DISABLED_FEATURES: [&str; 11] = [
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "in_app_browser",
    "in_app_local_automation",
    "hooks",
    "skill_mcp_dependency_install",
    "image_generation",
];

const INITIALIZE_ID: u64 = 1;
const THREAD_START_ID: u64 = 2;
const TURN_START_ID: u64 = 3;

/// The arguments that launch `codex app-server` for a run the adapters
/// already confined as `codex exec` (`args[start..]` are Codex's own; what
/// precedes them is an `env` launcher's). The confinement carries over
/// unchanged (`-c` profile entries, the sandbox mode, the model); the
/// approval policy becomes `on-request`, answered by this host. `Err` names
/// an argument with no app-server equivalent, and the run stays on `exec`.
pub fn app_server_args(args: &[String], start: usize) -> Result<Vec<String>, String> {
    let toml = |value: &str| serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string());
    let mut out: Vec<String> = args[..start.min(args.len())].to_vec();
    let mut saw_exec = false;
    let mut index = start;
    while index < args.len() {
        let argument = args[index].as_str();
        let value = args.get(index + 1).map(String::as_str);
        // Exact command-line flags of `codex exec` (machine format).
        match argument {
            "exec" if !saw_exec => {
                saw_exec = true;
                out.push("app-server".to_string());
            }
            "-c" | "--config" => {
                let value = value.ok_or("`-c` has no value")?;
                if !value.trim_start().starts_with("approval_policy") {
                    out.extend(["-c".to_string(), value.to_string()]);
                }
                index += 1;
            }
            "--sandbox" | "-s" => {
                let value = value.ok_or("`--sandbox` has no value")?;
                out.extend(["-c".to_string(), format!("sandbox_mode={}", toml(value))]);
                index += 1;
            }
            "--model" | "-m" => {
                let value = value.ok_or("`--model` has no value")?;
                out.extend(["-c".to_string(), format!("model={}", toml(value))]);
                index += 1;
            }
            "--color" => index += 1,
            "--json" | "--ephemeral" | "--skip-git-repo-check" | "--ignore-user-config" => {}
            other => {
                if let Some(value) = other.strip_prefix("--sandbox=") {
                    out.extend(["-c".to_string(), format!("sandbox_mode={}", toml(value))]);
                } else if let Some(value) = other
                    .strip_prefix("--model=")
                    .or_else(|| other.strip_prefix("-m="))
                {
                    out.extend(["-c".to_string(), format!("model={}", toml(value))]);
                } else if other.starts_with("--color=") {
                } else {
                    return Err(format!(
                        "`{other}` has no equivalent in Codex's app-server launch"
                    ));
                }
            }
        }
        index += 1;
    }
    if !saw_exec {
        return Err("the launch does not run `codex exec`".to_string());
    }
    out.extend([
        "-c".to_string(),
        "approval_policy=\"on-request\"".to_string(),
        "-c".to_string(),
        "approvals_reviewer=\"user\"".to_string(),
    ]);
    for feature in DISABLED_FEATURES {
        out.extend(["-c".to_string(), format!("features.{feature}=false")]);
    }
    Ok(out)
}

/// A run's own `CODEX_HOME`: a fresh folder holding only a link to the
/// owner's `auth.json`, removed when the run is over. The link (not a copy)
/// keeps a token Codex refreshes during the run the owner's.
#[derive(Debug)]
pub struct CodexHome {
    path: PathBuf,
}

impl CodexHome {
    /// `Err` when the owner's sign-in is not a file this can link (no
    /// `auth.json`, or links unsupported here); the run then stays on
    /// `exec`.
    pub fn prepare() -> Result<Self, String> {
        let source = std::env::var_os("CODEX_HOME")
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|home| home.join(".codex")))
            .ok_or("no home directory")?;
        let auth = source.join("auth.json");
        if !auth.is_file() {
            return Err(format!("no Codex sign-in at {}", auth.display()));
        }
        let path = std::env::temp_dir().join(format!("aidaemon-codex-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).map_err(|error| error.to_string())?;
        let home = Self { path };
        link(&auth, &home.path.join("auth.json")).map_err(|error| error.to_string())?;
        Ok(home)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

#[cfg(unix)]
fn link(source: &std::path::Path, target: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, target)
}

#[cfg(windows)]
fn link(source: &std::path::Path, target: &std::path::Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(source, target)
}

impl Drop for CodexHome {
    fn drop(&mut self) {
        // Removes the link, never the file it points to.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// The first stdin line: the protocol handshake.
pub fn initialize_line() -> String {
    request_line(
        INITIALIZE_ID,
        "initialize",
        json!({"clientInfo": {"name": "aidaemon", "version": env!("CARGO_PKG_VERSION")}}),
    )
}

fn request_line(id: u64, method: &str, params: Value) -> String {
    let mut line =
        json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
    line.push('\n');
    line
}

fn reply_line(id: &Value, result: Value) -> String {
    let mut line = json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
    line.push('\n');
    line
}

fn error_reply_line(id: &Value, message: &str) -> String {
    let mut line = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": -32601, "message": message},
    })
    .to_string();
    line.push('\n');
    line
}

fn event(value: Value) -> String {
    value.to_string()
}

/// The daemon call a command approval asks for: a network host is judged
/// as a fetch from it (an egress to it); a command as the daemon's own
/// `terminal` running it. `Err` is refused.
pub fn command_call(
    id: &str,
    params: &Value,
    working_dir: Option<&str>,
) -> Result<ToolCall, String> {
    if params.get("kind").and_then(Value::as_str) == Some("writeStdin") {
        return Err("input to a command already running is not forwarded".to_string());
    }
    if let Some(host) = params
        .pointer("/networkApprovalContext/host")
        .and_then(Value::as_str)
    {
        return Ok(ToolCall::new(
            id,
            "web_fetch",
            json!({"url": format!("https://{host}/")}),
        ));
    }
    let command = params
        .get("command")
        .and_then(Value::as_str)
        .filter(|command| !command.trim().is_empty())
        .ok_or("its approval request names no command")?;
    let mut arguments = json!({"action": "run", "command": unwrap_shell(command)});
    if let Some(dir) = params.get("cwd").and_then(Value::as_str).or(working_dir) {
        arguments["working_dir"] = json!(dir);
    }
    Ok(ToolCall::new(id, "terminal", arguments))
}

/// The script inside Codex's shell wrapper (`/bin/zsh -lc '<script>'`), so
/// the command judged is the one it runs. Machine format: a shell command
/// line, tokenized.
fn unwrap_shell(command: &str) -> String {
    if let Ok(words) = shell_words::split(command)
        && let [shell, flag, script] = words.as_slice()
        && matches!(
            std::path::Path::new(shell)
                .file_name()
                .and_then(|name| name.to_str()),
            Some("zsh" | "bash" | "sh")
        )
        && matches!(flag.as_str(), "-c" | "-lc")
    {
        return script.clone();
    }
    command.to_string()
}

/// One file a patch changes, from its `fileChange` item.
#[derive(Debug, Clone, PartialEq)]
struct FileChange {
    path: String,
    diff: String,
}

#[derive(Default)]
struct HostState {
    thread_id: Option<String>,
    usage: Option<Value>,
    /// The changes of each `fileChange` item, by item id: its approval
    /// request names only the item.
    changes: HashMap<String, Vec<FileChange>>,
    answers: HostAnswers,
    /// Its turn ended, and how: `Some(true)` completed.
    completed: Option<bool>,
}

/// Hosts one `codex app-server` process for its run: starts its thread and
/// turn, answers its approval requests, and translates what it reports.
#[derive(Clone)]
pub struct CodexHost {
    stdin: Arc<tokio::sync::Mutex<Option<ChildStdin>>>,
    nested: NestedCalls,
    working_dir: Option<String>,
    prompt: Arc<str>,
    state: Arc<Mutex<HostState>>,
}

impl CodexHost {
    pub fn new(
        stdin: ChildStdin,
        nested: NestedCalls,
        working_dir: Option<String>,
        prompt: &str,
    ) -> Self {
        Self {
            stdin: Arc::new(tokio::sync::Mutex::new(Some(stdin))),
            nested,
            working_dir,
            prompt: Arc::from(prompt),
            state: Arc::default(),
        }
    }

    /// What it answered so far.
    pub fn answers(&self) -> HostAnswers {
        self.state
            .lock()
            .map(|state| state.answers.clone())
            .unwrap_or_default()
    }

    /// Whether its turn completed (not failed, interrupted, or never
    /// reached): the process exits 0 either way once its stdin closes.
    pub fn turn_completed(&self) -> bool {
        self.state
            .lock()
            .is_ok_and(|state| state.completed == Some(true))
    }

    /// Handle one stdout line; returns the `codex exec --json` events it
    /// stands for (none for most). An approval request is answered in its
    /// own task, so its output keeps being read while the owner decides.
    pub async fn on_line(&self, line: &str) -> Vec<String> {
        let Ok(message) = serde_json::from_str::<Value>(line.trim()) else {
            return vec![line.to_string()];
        };
        let method = message.get("method").and_then(Value::as_str);
        let id = message.get("id");
        match (method, id) {
            (Some(method), Some(id)) => {
                let host = self.clone();
                let method = method.to_string();
                let id = id.clone();
                let params = message.get("params").cloned().unwrap_or(Value::Null);
                tokio::spawn(async move { host.answer(&method, &id, &params).await });
                Vec::new()
            }
            (None, Some(id)) => self.on_response(id, &message).await,
            (Some(method), None) => {
                let params = message.get("params").cloned().unwrap_or(Value::Null);
                self.on_notification(method, &params).await
            }
            (None, None) => Vec::new(),
        }
    }

    async fn on_response(&self, id: &Value, message: &Value) -> Vec<String> {
        if let Some(error) = message.get("error") {
            let text = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Codex refused the request");
            if let Ok(mut state) = self.state.lock() {
                state.completed = Some(false);
            }
            self.close().await;
            return vec![event(json!({"type": "error", "message": text}))];
        }
        match id.as_u64() {
            Some(INITIALIZE_ID) => {
                let mut lines = json!({"jsonrpc": "2.0", "method": "initialized"}).to_string();
                lines.push('\n');
                let mut params = json!({
                    "approvalPolicy": "on-request",
                    "approvalsReviewer": "user",
                    "ephemeral": true,
                });
                if let Some(dir) = self.working_dir.as_deref() {
                    params["cwd"] = json!(dir);
                }
                lines.push_str(&request_line(THREAD_START_ID, "thread/start", params));
                self.write(&lines).await;
            }
            Some(THREAD_START_ID) => {
                let thread_id = message
                    .pointer("/result/thread/id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if let Ok(mut state) = self.state.lock() {
                    state.thread_id = Some(thread_id.clone());
                }
                let params = json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": &*self.prompt}],
                });
                self.write(&request_line(TURN_START_ID, "turn/start", params))
                    .await;
            }
            _ => {}
        }
        Vec::new()
    }

    async fn on_notification(&self, method: &str, params: &Value) -> Vec<String> {
        let text = |pointer: &str| {
            params
                .pointer(pointer)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        match method {
            "thread/started" => vec![event(
                json!({"type": "thread.started", "thread_id": text("/thread/id")}),
            )],
            "turn/started" => vec![event(json!({"type": "turn.started"}))],
            "item/started" | "item/completed" => {
                let item = params.get("item").unwrap_or(&Value::Null);
                if item.get("type").and_then(Value::as_str) == Some("fileChange") {
                    self.remember_changes(item);
                }
                let kind = if method == "item/started" {
                    "item.started"
                } else {
                    "item.completed"
                };
                exec_item(item)
                    .map(|item| vec![event(json!({"type": kind, "item": item}))])
                    .unwrap_or_default()
            }
            "thread/tokenUsage/updated" => {
                if let Ok(mut state) = self.state.lock() {
                    state.usage = params.pointer("/tokenUsage/total").cloned();
                }
                Vec::new()
            }
            "error" if params.get("willRetry").and_then(Value::as_bool) != Some(true) => {
                vec![event(
                    json!({"type": "error", "message": text("/error/message")}),
                )]
            }
            "turn/completed" => {
                let status = text("/turn/status");
                let completed = status == "completed";
                let usage = {
                    let Ok(mut state) = self.state.lock() else {
                        return Vec::new();
                    };
                    state.completed = Some(completed);
                    state.usage.clone()
                };
                // Its turn is over: closing stdin lets it exit.
                self.close().await;
                if completed {
                    let usage = usage.unwrap_or(Value::Null);
                    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
                    vec![event(json!({
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": count("inputTokens"),
                            "cached_input_tokens": count("cachedInputTokens"),
                            "output_tokens": count("outputTokens"),
                        },
                    }))]
                } else {
                    let message = params
                        .pointer("/turn/error/message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("the turn ended {status}"));
                    vec![event(
                        json!({"type": "turn.failed", "error": {"message": message}}),
                    )]
                }
            }
            _ => Vec::new(),
        }
    }

    fn remember_changes(&self, item: &Value) {
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            return;
        };
        let changes: Vec<FileChange> = item
            .get("changes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|change| {
                Some(FileChange {
                    path: change.get("path")?.as_str()?.to_string(),
                    diff: change
                        .get("diff")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                })
            })
            .collect();
        if !changes.is_empty()
            && let Ok(mut state) = self.state.lock()
        {
            state.changes.insert(id.to_string(), changes);
        }
    }

    async fn answer(&self, method: &str, id: &Value, params: &Value) {
        let call_id = format!("codex-approval-{}", id.to_string().trim_matches('"'));
        let (label, verdict, reply) = match method {
            "item/commandExecution/requestApproval" => {
                let label = command_label(params);
                let verdict = match command_call(&call_id, params, self.working_dir.as_deref()) {
                    Ok(call) => self.nested.authorize(call).await,
                    Err(reason) => NestedVerdict::Deny { reason },
                };
                let decision = if verdict.is_allowed() {
                    "accept"
                } else {
                    "decline"
                };
                (label, verdict, json!({"decision": decision}))
            }
            "item/fileChange/requestApproval" => {
                let changes = self
                    .state
                    .lock()
                    .ok()
                    .and_then(|state| {
                        let item = params.get("itemId").and_then(Value::as_str)?;
                        state.changes.get(item).cloned()
                    })
                    .unwrap_or_default();
                let label = format!(
                    "file change: {}",
                    changes
                        .iter()
                        .map(|change| change.path.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                let verdict = self.authorize_changes(&call_id, &changes).await;
                let decision = if verdict.is_allowed() {
                    "accept"
                } else {
                    "decline"
                };
                (label, verdict, json!({"decision": decision}))
            }
            // Wider access for the rest of its turn is never granted here:
            // each action it then needs is asked one by one instead.
            "item/permissions/requestApproval" => (
                "wider access for the turn".to_string(),
                NestedVerdict::Deny {
                    reason: "a delegated run's access is not widened for its whole turn"
                        .to_string(),
                },
                json!({"permissions": {}, "scope": "turn"}),
            ),
            other => {
                tracing::info!(method = %other, "A coding agent's request this host does not serve");
                self.write(&error_reply_line(id, "not supported by this host"))
                    .await;
                return;
            }
        };
        if let Ok(mut state) = self.state.lock() {
            match verdict {
                NestedVerdict::Allow => {}
                NestedVerdict::Declined { .. } => state.answers.declined.push(label),
                _ => state.answers.refused.push(label),
            }
        }
        tracing::info!(
            request = %method,
            allowed = verdict.is_allowed(),
            "Answered a coding agent's permission prompt"
        );
        self.write(&reply_line(id, reply)).await;
    }

    /// Every file a patch changes is judged as the daemon's `write_file` of
    /// it; the patch is accepted only when all are allowed.
    async fn authorize_changes(&self, call_id: &str, changes: &[FileChange]) -> NestedVerdict {
        if changes.is_empty() {
            return NestedVerdict::Deny {
                reason: "its file-change request names no file".to_string(),
            };
        }
        for (index, change) in changes.iter().enumerate() {
            let call = ToolCall::new(
                format!("{call_id}-{index}"),
                "write_file",
                json!({"path": change.path, "content": change.diff}),
            );
            let verdict = self.nested.authorize(call).await;
            if !verdict.is_allowed() {
                return verdict;
            }
        }
        NestedVerdict::Allow
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

/// An app-server item as `codex exec --json` writes it, for the items the
/// rest of `cli_agent` reads; `None` for the others.
fn exec_item(item: &Value) -> Option<Value> {
    let id = item.get("id").cloned().unwrap_or(Value::Null);
    let status = item.get("status").cloned().unwrap_or(Value::Null);
    match item.get("type").and_then(Value::as_str)? {
        "agentMessage" => Some(json!({
            "id": id,
            "type": "agent_message",
            "text": item.get("text").and_then(Value::as_str).unwrap_or_default(),
        })),
        "commandExecution" => Some(json!({
            "id": id,
            "type": "command_execution",
            "command": item.get("command").and_then(Value::as_str).unwrap_or_default(),
            "aggregated_output": item.get("aggregatedOutput").and_then(Value::as_str).unwrap_or_default(),
            "exit_code": item.get("exitCode").cloned().unwrap_or(Value::Null),
            "status": status,
        })),
        "fileChange" => {
            let changes: Vec<Value> = item
                .get("changes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|change| {
                    json!({
                        "path": change.get("path").cloned().unwrap_or(Value::Null),
                        "kind": change.pointer("/kind/type").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect();
            Some(json!({"id": id, "type": "file_change", "changes": changes, "status": status}))
        }
        _ => None,
    }
}

/// A command approval said as the owner reads it.
fn command_label(params: &Value) -> String {
    if let Some(host) = params
        .pointer("/networkApprovalContext/host")
        .and_then(Value::as_str)
    {
        return format!("network: {host}");
    }
    let command = params
        .get("command")
        .and_then(Value::as_str)
        .map(unwrap_shell)
        .unwrap_or_default();
    let command = command.split_whitespace().collect::<Vec<_>>().join(" ");
    format!(
        "command: {}",
        aidaemon_events::text::truncate_str(&command, 160)
    )
}

/// The process's stdout, read by `host` and replaced by the `codex exec
/// --json` events it stands for. Its stdin closes at the end.
pub fn translate<R>(stdout: R, host: CodexHost) -> tokio::io::DuplexStream
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let (mut writer, reader) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            for mut out in host.on_line(&line).await {
                out.push('\n');
                if writer.write_all(out.as_bytes()).await.is_err() {
                    return;
                }
            }
        }
        host.close().await;
    });
    reader
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    fn config_values(args: &[String]) -> Vec<&str> {
        args.windows(2)
            .filter(|pair| pair[0] == "-c")
            .map(|pair| pair[1].as_str())
            .collect()
    }

    #[test]
    fn a_confined_exec_launch_becomes_an_app_server_launch_with_its_confinement() {
        let args = strings(&[
            "exec",
            "--json",
            "--model",
            "gpt-test",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "-c",
            "default_permissions=\"aidaemon-task\"",
            "-c",
            "permissions.aidaemon-task.network.enabled=false",
            "-c",
            "approval_policy=\"never\"",
        ]);
        let out = app_server_args(&args, 0).unwrap();
        assert_eq!(out[0], "app-server");
        let config = config_values(&out);
        assert!(config.contains(&"default_permissions=\"aidaemon-task\""));
        assert!(config.contains(&"permissions.aidaemon-task.network.enabled=false"));
        assert!(config.contains(&"model=\"gpt-test\""));
        // It asks now, and never by the owner's own policy.
        assert!(config.contains(&"approval_policy=\"on-request\""));
        assert!(!config.contains(&"approval_policy=\"never\""));
        assert!(config.contains(&"features.computer_use=false"));
        assert!(!out.iter().any(|arg| arg == "--json" || arg == "exec"));
    }

    #[test]
    fn an_env_launcher_is_kept_and_an_unknown_argument_keeps_the_run_on_exec() {
        let args = strings(&["FOO=1", "codex", "exec", "--sandbox", "read-only"]);
        let out = app_server_args(&args, 2).unwrap();
        assert_eq!(&out[..3], &strings(&["FOO=1", "codex", "app-server"])[..]);
        assert!(config_values(&out).contains(&"sandbox_mode=\"read-only\""));

        let args = strings(&["exec", "--output-schema", "s.json"]);
        assert!(app_server_args(&args, 0).is_err());
        assert!(app_server_args(&strings(&["--json"]), 0).is_err());
    }

    #[test]
    fn a_command_approval_is_the_terminal_command_it_runs_or_the_host_it_reaches() {
        let params = json!({
            "kind": "command",
            "command": "/bin/zsh -lc 'touch $HOME/probe.txt'",
            "cwd": "/w/project",
        });
        let call = command_call("a", &params, None).unwrap();
        assert_eq!(call.name, "terminal");
        assert_eq!(call.arguments["command"], "touch $HOME/probe.txt");
        assert_eq!(call.arguments["working_dir"], "/w/project");

        let params = json!({
            "command": "curl https://example.com",
            "networkApprovalContext": {"host": "example.com", "protocol": "https"},
        });
        let call = command_call("b", &params, Some("/w")).unwrap();
        assert_eq!(call.name, "web_fetch");
        assert_eq!(call.arguments["url"], "https://example.com/");

        assert!(command_call("c", &json!({"kind": "writeStdin"}), None).is_err());
        assert!(command_call("d", &json!({}), None).is_err());
    }

    #[test]
    fn items_are_written_the_way_codex_exec_writes_them() {
        let message = exec_item(&json!({
            "type": "agentMessage", "id": "m1", "text": "Done.", "phase": "final_answer"
        }))
        .unwrap();
        assert_eq!(
            message,
            json!({"id": "m1", "type": "agent_message", "text": "Done."})
        );
        let command = exec_item(&json!({
            "type": "commandExecution", "id": "c1", "command": "ls",
            "aggregatedOutput": "a\n", "exitCode": 0, "status": "completed",
            "commandActions": [], "cwd": "/w"
        }))
        .unwrap();
        assert_eq!(command["type"], "command_execution");
        assert_eq!(command["aggregated_output"], "a\n");
        assert_eq!(command["exit_code"], 0);
        let patch = exec_item(&json!({
            "type": "fileChange", "id": "f1", "status": "completed",
            "changes": [{"path": "/w/a.txt", "kind": {"type": "add"}, "diff": "+a"}]
        }))
        .unwrap();
        assert_eq!(
            patch["changes"][0],
            json!({"path": "/w/a.txt", "kind": "add"})
        );
        assert!(exec_item(&json!({"type": "reasoning", "id": "r"})).is_none());
    }

    /// A fake `codex app-server` on the other end of real pipes: the host
    /// starts the thread and turn, a command approval is judged by the run
    /// and answered, its result reads as `codex exec --json` events, and the
    /// end of the turn closes stdin.
    #[tokio::test]
    async fn a_run_is_started_answered_translated_and_closed_over_the_pipes() {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};

        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("cat");
        let stdin = child.stdin.take().unwrap();
        let echoed = child.stdout.take().unwrap();
        let (nested, asked) = aidaemon_core::testing::nested_calls(|call| {
            if call.arguments["command"] == "rm -rf /w/other" {
                NestedVerdict::Declined {
                    reason: format!("{} by the owner", aidaemon_core::OWNER_DECLINED),
                }
            } else {
                NestedVerdict::Allow
            }
        });
        let host = CodexHost::new(stdin, nested, Some("/w".to_string()), "fix it");

        // What the host writes to the agent comes back on `cat`'s stdout.
        let mut sent = BufReader::new(echoed).lines();
        assert!(host.on_line(r#"{"id":1,"result":{}}"#).await.is_empty());
        let initialized = sent.next_line().await.unwrap().unwrap();
        assert!(initialized.contains("\"initialized\""));
        let thread_start: Value =
            serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(thread_start["method"], "thread/start");
        assert_eq!(thread_start["params"]["approvalPolicy"], "on-request");
        assert_eq!(thread_start["params"]["cwd"], "/w");

        host.on_line(r#"{"id":2,"result":{"thread":{"id":"t1"}}}"#)
            .await;
        let turn_start: Value =
            serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(turn_start["params"]["threadId"], "t1");
        assert_eq!(turn_start["params"]["input"][0]["text"], "fix it");

        host.on_line(r#"{"id":0,"method":"item/commandExecution/requestApproval","params":{"kind":"command","command":"/bin/zsh -lc 'npm install'","cwd":"/w","itemId":"i1"}}"#).await;
        let reply: Value = serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(
            reply,
            json!({"jsonrpc": "2.0", "id": 0, "result": {"decision": "accept"}})
        );
        host.on_line(r#"{"id":1,"method":"item/commandExecution/requestApproval","params":{"command":"rm -rf /w/other","itemId":"i2"}}"#).await;
        let reply: Value = serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(reply["result"]["decision"], "decline");
        assert_eq!(asked.lock().unwrap()[0].arguments["command"], "npm install");

        let message = host
            .on_line(r#"{"method":"item/completed","params":{"item":{"type":"agentMessage","id":"m","text":"All set."}}}"#)
            .await;
        assert_eq!(message.len(), 1);
        assert!(message[0].contains("\"agent_message\""));
        host.on_line(r#"{"method":"thread/tokenUsage/updated","params":{"tokenUsage":{"total":{"inputTokens":10,"cachedInputTokens":4,"outputTokens":2}}}}"#).await;
        let end = host
            .on_line(
                r#"{"method":"turn/completed","params":{"turn":{"id":"u","status":"completed"}}}"#,
            )
            .await;
        let end: Value = serde_json::from_str(&end[0]).unwrap();
        assert_eq!(end["type"], "turn.completed");
        assert_eq!(end["usage"]["input_tokens"], 10);
        assert!(host.turn_completed());
        // The end of its turn closed stdin: `cat` reads end-of-file and exits.
        assert!(sent.next_line().await.unwrap().is_none());
        assert!(child.wait().await.unwrap().success());
        let answers = host.answers();
        assert_eq!(
            answers.declined,
            vec!["command: rm -rf /w/other".to_string()]
        );
        assert!(answers.refused.is_empty());
    }

    #[tokio::test]
    async fn a_file_change_is_judged_per_file_from_its_item() {
        use std::process::Stdio;
        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .expect("cat");
        let (nested, asked) = aidaemon_core::testing::nested_calls(|_| NestedVerdict::Deny {
            reason: "outside the task".to_string(),
        });
        let host = CodexHost::new(child.stdin.take().unwrap(), nested, None, "p");
        host.on_line(r#"{"method":"item/started","params":{"item":{"type":"fileChange","id":"f1","status":"inProgress","changes":[{"path":"/etc/hosts","kind":{"type":"update"},"diff":"+x"}]}}}"#).await;
        host.answer(
            "item/fileChange/requestApproval",
            &json!(7),
            &json!({"itemId": "f1"}),
        )
        .await;
        {
            let asked = asked.lock().unwrap();
            assert_eq!(asked.len(), 1);
            assert_eq!(asked[0].name, "write_file");
            assert_eq!(asked[0].arguments["path"], "/etc/hosts");
        }
        assert_eq!(
            host.answers().refused,
            vec!["file change: /etc/hosts".to_string()]
        );
        host.close().await;
        let _ = child.wait().await;
    }
}
