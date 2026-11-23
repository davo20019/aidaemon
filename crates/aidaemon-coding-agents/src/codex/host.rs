//! A hosted Codex: `codex app-server`, its approval requests answered by
//! the judge.
//!
//! `codex exec` cannot ask anyone. `codex app-server` is the same agent
//! behind a JSON-RPC protocol on stdin/stdout, and it sends its host a
//! typed request before it acts beyond what it may do on its own:
//! `item/commandExecution/requestApproval` (a command, or a network host),
//! `item/fileChange/requestApproval` (a patch) and
//! `item/permissions/requestApproval` (wider access for the rest of its
//! turn). Each becomes an [`AgentAction`] for the judge. A decision that
//! would stay with the agent ("for this session", a policy amendment) is
//! never sent back.
//!
//! The process runs with its own `CODEX_HOME` ([`CodexHome`]), holding only
//! a link to the owner's sign-in: the owner's `config.toml`, command rules,
//! MCP servers, plugins and trusted projects never reach a delegated run.
//!
//! Its notifications are translated into the `codex exec --json` events an
//! `exec` run writes, so the embedding agent reads both launches the same
//! way.
//!
//! Verified against Codex CLI 0.170.0 (live probes, 2026-11-18).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::ChildStdin;

use crate::stdin::Stdin;
use crate::{ActionJudge, AgentAction, AgentHost, HostAnswers, HostStarter, Verdict};

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

/// What the agent is told about its approvals. Codex's own model decides
/// when to ask to leave its sandbox, and without this it reads a sandbox
/// refusal as final (live 2026-11-18: a blocked `touch` outside the folder
/// and a blocked `curl` were reported as failures, never asked).
pub const HOST_INSTRUCTIONS: &str = "Your approval requests are answered by the \
owner's assistant that delegated this task, under the owner's own rules. When \
the sandbox blocks a command, file change or network access that the task \
needs, request escalated permissions for it (with a one-line justification) \
instead of reporting the failure. A declined request is final: do not retry \
it another way.";

/// When a hosted Codex asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Asks {
    /// Before it leaves its sandbox (the project folder): checked mode.
    /// Its network goes through Codex's own proxy, which lets no host
    /// through unasked, so a blocked host is a plain refusal it can ask
    /// past rather than a lookup that just fails.
    BeyondSandbox,
    /// Before every command and file change but its short list of
    /// read-only commands: every-command mode.
    EveryCommand,
}

impl Asks {
    /// Codex's name for the thread's policy (a machine format of its
    /// protocol). `untrusted` is accepted only here: Codex 0.170 refuses it
    /// as a launch setting ("no longer supported"), so the launch says
    /// `on-request` and the thread narrows it (live probe, 2026-11-18).
    fn policy(self) -> &'static str {
        match self {
            Self::BeyondSandbox => "on-request",
            Self::EveryCommand => "untrusted",
        }
    }
}

/// The arguments that launch `codex app-server` for a run already shaped
/// as `codex exec` (`args[start..]` are Codex's own; what precedes them is
/// a launcher's). The confinement carries over (`-c` profile entries, the
/// sandbox mode, the model); the approval policy becomes `asks`'s. `Err`
/// names an argument with no app-server equivalent.
pub fn app_server_args(args: &[String], start: usize, asks: Asks) -> Result<Vec<String>, String> {
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
                // Its own entries, by their config keys (machine format).
                let key = value.split('=').next().unwrap_or_default().trim();
                if key == "approval_policy" {
                } else if asks == Asks::BeyondSandbox && key.ends_with(".network.enabled") {
                    // Through the proxy below, which allows no host.
                    out.extend(["-c".to_string(), format!("{key}=true")]);
                } else {
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
    if asks == Asks::BeyondSandbox {
        out.extend([
            "-c".to_string(),
            "features.network_proxy.enabled=true".to_string(),
        ]);
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
    /// `auth.json`, or links unsupported here).
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

/// The action a command approval asks for: a network host to reach, or a
/// command to run. `Err` is refused.
pub fn command_action(params: &Value, working_dir: Option<&str>) -> Result<AgentAction, String> {
    if params.get("kind").and_then(Value::as_str) == Some("writeStdin") {
        return Err("input to a command already running is not forwarded".to_string());
    }
    if let Some(host) = params
        .pointer("/networkApprovalContext/host")
        .and_then(Value::as_str)
    {
        return Ok(AgentAction::ReachHost {
            host: host.to_string(),
        });
    }
    let command = params
        .get("command")
        .and_then(Value::as_str)
        .filter(|command| !command.trim().is_empty())
        .ok_or("its approval request names no command")?;
    Ok(AgentAction::RunCommand {
        command: unwrap_shell(command),
        working_dir: params
            .get("cwd")
            .and_then(Value::as_str)
            .or(working_dir)
            .map(str::to_string),
    })
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
    usage: Option<Value>,
    /// The changes of each `fileChange` item, by item id: its approval
    /// request names only the item.
    changes: HashMap<String, Vec<FileChange>>,
    answers: HostAnswers,
    /// Its turn ended, and how: `Some(true)` completed.
    completed: Option<bool>,
}

/// Starts a [`CodexHost`].
pub struct CodexHostStarter {
    pub working_dir: Option<String>,
    pub asks: Asks,
}

#[async_trait]
impl HostStarter for CodexHostStarter {
    /// The handshake goes in first; the task follows once its thread is
    /// started.
    async fn start(
        self: Box<Self>,
        stdin: ChildStdin,
        judge: Arc<dyn ActionJudge>,
        prompt: &str,
    ) -> Arc<dyn AgentHost> {
        let host = CodexHost::new(stdin, judge, self.working_dir, prompt, self.asks);
        host.stdin.write(&initialize_line()).await;
        Arc::new(host)
    }
}

/// Hosts one `codex app-server` process for its run: starts its thread and
/// turn, answers its approval requests, and translates what it reports.
#[derive(Clone)]
pub struct CodexHost {
    stdin: Stdin,
    judge: Arc<dyn ActionJudge>,
    working_dir: Option<String>,
    prompt: Arc<str>,
    asks: Asks,
    state: Arc<Mutex<HostState>>,
}

impl CodexHost {
    pub fn new(
        stdin: ChildStdin,
        judge: Arc<dyn ActionJudge>,
        working_dir: Option<String>,
        prompt: &str,
        asks: Asks,
    ) -> Self {
        Self {
            stdin: Stdin::new(stdin),
            judge,
            working_dir,
            prompt: Arc::from(prompt),
            asks,
            state: Arc::default(),
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
            self.stdin.close().await;
            return vec![json!({"type": "error", "message": text}).to_string()];
        }
        match id.as_u64() {
            Some(INITIALIZE_ID) => {
                let mut lines = json!({"jsonrpc": "2.0", "method": "initialized"}).to_string();
                lines.push('\n');
                let mut params = json!({
                    "approvalPolicy": self.asks.policy(),
                    "approvalsReviewer": "user",
                    "ephemeral": true,
                    "developerInstructions": HOST_INSTRUCTIONS,
                });
                if let Some(dir) = self.working_dir.as_deref() {
                    params["cwd"] = json!(dir);
                }
                lines.push_str(&request_line(THREAD_START_ID, "thread/start", params));
                self.stdin.write(&lines).await;
            }
            Some(THREAD_START_ID) => {
                let thread_id = message
                    .pointer("/result/thread/id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let params = json!({
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": &*self.prompt}],
                });
                self.stdin
                    .write(&request_line(TURN_START_ID, "turn/start", params))
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
        let event = |value: Value| vec![value.to_string()];
        match method {
            "thread/started" => {
                event(json!({"type": "thread.started", "thread_id": text("/thread/id")}))
            }
            "turn/started" => event(json!({"type": "turn.started"})),
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
                    .map(|item| event(json!({"type": kind, "item": item})))
                    .unwrap_or_default()
            }
            "thread/tokenUsage/updated" => {
                if let Ok(mut state) = self.state.lock() {
                    state.usage = params.pointer("/tokenUsage/total").cloned();
                }
                Vec::new()
            }
            "error" if params.get("willRetry").and_then(Value::as_bool) != Some(true) => {
                event(json!({"type": "error", "message": text("/error/message")}))
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
                self.stdin.close().await;
                if completed {
                    let usage = usage.unwrap_or(Value::Null);
                    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
                    event(json!({
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": count("inputTokens"),
                            "cached_input_tokens": count("cachedInputTokens"),
                            "output_tokens": count("outputTokens"),
                        },
                    }))
                } else {
                    let message = params
                        .pointer("/turn/error/message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("the turn ended {status}"));
                    event(json!({"type": "turn.failed", "error": {"message": message}}))
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
        let decision = |verdict: &Verdict| json!({"decision": if verdict.is_allowed() { "accept" } else { "decline" }});
        let (label, verdict, reply) = match method {
            "item/commandExecution/requestApproval" => {
                let (label, verdict) = match command_action(params, self.working_dir.as_deref()) {
                    Ok(action) => (action.label(), self.judge.judge(action).await),
                    Err(reason) => ("command".to_string(), Verdict::Deny { reason }),
                };
                let reply = decision(&verdict);
                (label, verdict, reply)
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
                    "write: {}",
                    changes
                        .iter()
                        .map(|change| change.path.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                let verdict = self.judge_changes(&changes).await;
                let reply = decision(&verdict);
                (label, verdict, reply)
            }
            // Wider access for the rest of its turn is never granted here:
            // each action it then needs is asked one by one instead.
            "item/permissions/requestApproval" => (
                "wider access for the turn".to_string(),
                Verdict::Deny {
                    reason: "a delegated run's access is not widened for its whole turn"
                        .to_string(),
                },
                json!({"permissions": {}, "scope": "turn"}),
            ),
            other => {
                tracing::info!(method = %other, "A coding agent's request this host does not serve");
                self.stdin
                    .write(&error_reply_line(id, "not supported by this host"))
                    .await;
                return;
            }
        };
        if let Ok(mut state) = self.state.lock() {
            match verdict {
                Verdict::Allow => {}
                Verdict::Declined { .. } => state.answers.declined.push(label),
                _ => state.answers.refused.push(label),
            }
        }
        tracing::info!(
            request = %method,
            allowed = verdict.is_allowed(),
            "Answered a coding agent's permission prompt"
        );
        self.stdin.write(&reply_line(id, reply)).await;
    }

    /// Every file a patch changes is judged as a write of it; the patch is
    /// accepted only when all are allowed.
    async fn judge_changes(&self, changes: &[FileChange]) -> Verdict {
        if changes.is_empty() {
            return Verdict::Deny {
                reason: "its file-change request names no file".to_string(),
            };
        }
        for change in changes {
            let verdict = self
                .judge
                .judge(AgentAction::WriteFile {
                    path: change.path.clone(),
                    content: change.diff.clone(),
                })
                .await;
            if !verdict.is_allowed() {
                return verdict;
            }
        }
        Verdict::Allow
    }
}

#[async_trait]
impl AgentHost for CodexHost {
    /// Returns the `codex exec --json` events a line stands for (none for
    /// most).
    async fn on_line(&self, line: &str) -> Vec<String> {
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

    fn answers(&self) -> HostAnswers {
        self.state
            .lock()
            .map(|state| state.answers.clone())
            .unwrap_or_default()
    }

    /// Its process exits 0 once its stdin closes, whatever its turn did.
    fn unfinished(&self) -> bool {
        !self
            .state
            .lock()
            .is_ok_and(|state| state.completed == Some(true))
    }

    async fn close(&self) {
        self.stdin.close().await;
    }
}

/// An app-server item as `codex exec --json` writes it, for the items an
/// `exec` run's reader uses; `None` for the others.
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
        "webSearch" => Some(json!({"id": id, "type": "web_search", "status": status})),
        "mcpToolCall" => Some(json!({
            "id": id,
            "type": "mcp_tool_call",
            "server": item.get("server").cloned().unwrap_or(Value::Null),
            "tool": item.get("tool").cloned().unwrap_or(Value::Null),
            "status": status,
        })),
        _ => None,
    }
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
        let out = app_server_args(&args, 0, Asks::BeyondSandbox).unwrap();
        assert_eq!(out[0], "app-server");
        let config = config_values(&out);
        assert!(config.contains(&"default_permissions=\"aidaemon-task\""));
        // Its network goes through Codex's proxy, which allows no host.
        assert!(config.contains(&"permissions.aidaemon-task.network.enabled=true"));
        assert!(config.contains(&"features.network_proxy.enabled=true"));
        assert!(config.contains(&"model=\"gpt-test\""));
        // It asks now, and never by the owner's own policy.
        assert!(config.contains(&"approval_policy=\"on-request\""));
        assert!(!config.contains(&"approval_policy=\"never\""));
        assert!(config.contains(&"features.computer_use=false"));
        assert!(!out.iter().any(|arg| arg == "--json" || arg == "exec"));
    }

    #[test]
    fn an_env_launcher_is_kept_and_an_unknown_argument_is_refused() {
        let args = strings(&["FOO=1", "codex", "exec", "--sandbox", "read-only"]);
        let out = app_server_args(&args, 2, Asks::EveryCommand).unwrap();
        assert_eq!(&out[..3], &strings(&["FOO=1", "codex", "app-server"])[..]);
        let config = config_values(&out);
        assert!(config.contains(&"sandbox_mode=\"read-only\""));
        // The launch says `on-request` (Codex refuses `untrusted` there);
        // the thread asks for every command. No proxy: its sandbox is
        // read-only.
        assert!(config.contains(&"approval_policy=\"on-request\""));
        assert!(!config.contains(&"features.network_proxy.enabled=true"));

        let args = strings(&["exec", "--output-schema", "s.json"]);
        assert!(app_server_args(&args, 0, Asks::BeyondSandbox).is_err());
        assert!(app_server_args(&strings(&["--json"]), 0, Asks::BeyondSandbox).is_err());
    }

    #[test]
    fn a_command_approval_is_the_command_it_runs_or_the_host_it_reaches() {
        let params = json!({
            "kind": "command",
            "command": "/bin/zsh -lc 'touch $HOME/probe.txt'",
            "cwd": "/w/project",
        });
        assert_eq!(
            command_action(&params, None).unwrap(),
            AgentAction::RunCommand {
                command: "touch $HOME/probe.txt".into(),
                working_dir: Some("/w/project".into()),
            }
        );
        let params = json!({
            "command": "curl https://example.com",
            "networkApprovalContext": {"host": "example.com", "protocol": "https"},
        });
        assert_eq!(
            command_action(&params, Some("/w")).unwrap(),
            AgentAction::ReachHost {
                host: "example.com".into()
            }
        );
        assert!(command_action(&json!({"kind": "writeStdin"}), None).is_err());
        assert!(command_action(&json!({}), None).is_err());
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

    /// Answers by a rule, and keeps what it was asked.
    struct Judge {
        rule: fn(&AgentAction) -> Verdict,
        asked: Mutex<Vec<AgentAction>>,
    }

    impl Judge {
        fn new(rule: fn(&AgentAction) -> Verdict) -> Arc<Self> {
            Arc::new(Self {
                rule,
                asked: Mutex::default(),
            })
        }
    }

    #[async_trait]
    impl ActionJudge for Judge {
        async fn judge(&self, action: AgentAction) -> Verdict {
            let verdict = (self.rule)(&action);
            self.asked.lock().unwrap().push(action);
            verdict
        }
    }

    /// A fake `codex app-server` on the other end of real pipes: the host
    /// starts the thread and turn, a command approval is judged and
    /// answered, its result reads as `codex exec --json` events, and the
    /// end of the turn closes stdin.
    #[cfg(unix)]
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
        let judge = Judge::new(|action| match action {
            AgentAction::RunCommand { command, .. } if command == "rm -rf /w/other" => {
                Verdict::Declined {
                    reason: format!("{} by the owner", aidaemon_core::OWNER_DECLINED),
                }
            }
            _ => Verdict::Allow,
        });
        let host = CodexHost::new(
            stdin,
            judge.clone(),
            Some("/w".to_string()),
            "fix it",
            Asks::BeyondSandbox,
        );

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
        assert_eq!(
            thread_start["params"]["developerInstructions"],
            HOST_INSTRUCTIONS
        );

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
        assert_eq!(
            judge.asked.lock().unwrap()[0],
            AgentAction::RunCommand {
                command: "npm install".into(),
                working_dir: Some("/w".into()),
            }
        );

        let message = host
            .on_line(r#"{"method":"item/completed","params":{"item":{"type":"agentMessage","id":"m","text":"All set."}}}"#)
            .await;
        assert_eq!(message.len(), 1);
        assert!(message[0].contains("\"agent_message\""));
        host.on_line(r#"{"method":"thread/tokenUsage/updated","params":{"tokenUsage":{"total":{"inputTokens":10,"cachedInputTokens":4,"outputTokens":2}}}}"#).await;
        assert!(host.unfinished());
        let end = host
            .on_line(
                r#"{"method":"turn/completed","params":{"turn":{"id":"u","status":"completed"}}}"#,
            )
            .await;
        let end: Value = serde_json::from_str(&end[0]).unwrap();
        assert_eq!(end["type"], "turn.completed");
        assert_eq!(end["usage"]["input_tokens"], 10);
        assert!(!host.unfinished());
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

    /// Every command is asked on the thread, where Codex accepts it.
    #[cfg(unix)]
    #[tokio::test]
    async fn every_command_asks_on_the_thread() {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};
        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("cat");
        let host = CodexHost::new(
            child.stdin.take().unwrap(),
            Judge::new(|_| Verdict::Allow),
            None,
            "p",
            Asks::EveryCommand,
        );
        let mut sent = BufReader::new(child.stdout.take().unwrap()).lines();
        host.on_line(r#"{"id":1,"result":{}}"#).await;
        sent.next_line().await.unwrap();
        let thread_start: Value =
            serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(thread_start["params"]["approvalPolicy"], "untrusted");
        host.close().await;
        let _ = child.wait().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_file_change_is_judged_per_file_from_its_item() {
        use std::process::Stdio;
        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .expect("cat");
        let judge = Judge::new(|_| Verdict::Deny {
            reason: "outside the task".to_string(),
        });
        let host = CodexHost::new(
            child.stdin.take().unwrap(),
            judge.clone(),
            None,
            "p",
            Asks::BeyondSandbox,
        );
        host.on_line(r#"{"method":"item/started","params":{"item":{"type":"fileChange","id":"f1","status":"inProgress","changes":[{"path":"/etc/hosts","kind":{"type":"update"},"diff":"+x"}]}}}"#).await;
        host.answer(
            "item/fileChange/requestApproval",
            &json!(7),
            &json!({"itemId": "f1"}),
        )
        .await;
        {
            let asked = judge.asked.lock().unwrap();
            assert_eq!(
                *asked,
                vec![AgentAction::WriteFile {
                    path: "/etc/hosts".into(),
                    content: "+x".into(),
                }]
            );
        }
        assert_eq!(
            host.answers().refused,
            vec!["write: /etc/hosts".to_string()]
        );
        host.close().await;
        let _ = child.wait().await;
    }
}
