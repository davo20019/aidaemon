//! The host side of the Agent Client Protocol: one session, one task, each
//! permission request answered by the judge.
//!
//! JSON-RPC over the agent's stdin and stdout: `initialize`, then
//! `session/new` in the run's folder, then (when the owner chose its model)
//! `session/set_model`, then `session/prompt` with the task.
//! While it works the agent sends `session/update` notifications (its
//! message, its tool calls) and, before an action that needs one,
//! `session/request_permission` with the tool call (its `kind`, raw input
//! and the files it touches) and the options it offers. Each becomes an
//! [`AgentAction`] for the judge; the answer is the agent's own "once"
//! option, never one that would keep a rule in the agent's settings. The
//! prompt's response ends the turn (its `stopReason`).
//!
//! A run that continues an earlier session sends `session/load` with that
//! session's id in place of `session/new`, which only an agent whose
//! `initialize` reply says `agentCapabilities.loadSession` offers; any
//! other agent's run ends there, before a prompt, with that said in plain
//! words. While it loads, the agent replays the session's history as
//! `session/update` notifications: they are what it did before, not what
//! it does now, so none of them reaches the run. Its reply to the load
//! means the session is ready, and the task follows as in a new one.
//! The run's reader is told the session's id, its model, and whether a
//! later run can continue it (`resumable`: the agent's `loadSession`).
//!
//! The run's reader is given a `{"type": "result"}` line with the agent's
//! final message, as a stream-json agent writes it, and that message alone
//! as `final_message` (its `result` says why a turn that failed stopped).
//!
//! The host advertises no file system or terminal of its own: the agent
//! acts with its own tools, after asking.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::ChildStdin;

use crate::stdin::Stdin;
use crate::{ActionJudge, AgentAction, AgentHost, HostAnswers, HostStarter, Verdict};

const INITIALIZE_ID: u64 = 1;
const SESSION_NEW_ID: u64 = 2;
const PROMPT_ID: u64 = 3;
const SET_MODEL_ID: u64 = 4;
const SESSION_LOAD_ID: u64 = 5;

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

fn error_line(id: &Value, message: &str) -> String {
    let mut line = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": -32601, "message": message},
    })
    .to_string();
    line.push('\n');
    line
}

/// The paths a tool call names (`locations[].path`).
fn locations(update: &Value) -> Vec<Value> {
    update
        .get("locations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|location| location.get("path").cloned())
        .collect()
}

/// The handshake: protocol 1, and no file system or terminal offered.
pub fn initialize_line() -> String {
    request_line(
        INITIALIZE_ID,
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": false, "writeTextFile": false},
                "terminal": false,
            },
            "clientInfo": {"name": "aidaemon", "version": env!("CARGO_PKG_VERSION")},
        }),
    )
}

/// The first string among `keys` in `value` (an agent's own field names
/// for the same thing differ: a machine format, read field by field).
/// The `rawInput` keys agents put a command and a URL under (ACP leaves
/// `rawInput` to each agent; Gemini's agents use `CommandLine`).
const COMMAND_KEYS: &[&str] = &["command", "CommandLine", "cmd"];
const URL_KEYS: &[&str] = &["url", "Url", "URL"];

/// The run's line for a `tool_call` update: its typed fields only.
fn tool_call_line(update: &Value) -> String {
    let input = update.get("rawInput");
    json!({
        "type": "tool_call",
        "id": update.get("toolCallId"),
        "title": update.get("title").and_then(Value::as_str).unwrap_or_default(),
        "kind": update.get("kind"),
        "status": update.get("status"),
        "paths": locations(update),
        "command": text_of(input, COMMAND_KEYS),
        "url": text_of(input, URL_KEYS),
    })
    .to_string()
}

fn text_of(value: Option<&Value>, keys: &[&str]) -> Option<String> {
    let value = value?;
    keys.iter().find_map(|key| match value.get(*key)? {
        Value::String(text) if !text.trim().is_empty() => Some(text.clone()),
        Value::Array(parts) => {
            let words: Vec<&str> = parts.iter().filter_map(Value::as_str).collect();
            (!words.is_empty()).then(|| words.join(" "))
        }
        _ => None,
    })
}

/// The actions a permission request's tool call asks for, by its `kind`:
/// one per file it touches for a change. `Err` is refused.
pub fn actions(tool_call: &Value, working_dir: Option<&str>) -> Result<Vec<AgentAction>, String> {
    let kind = tool_call
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("other");
    let input = tool_call.get("rawInput");
    let title = tool_call
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut paths: Vec<String> = tool_call
        .get("locations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|location| location.get("path").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    if paths.is_empty()
        && let Some(path) = text_of(
            input,
            &["path", "Path", "file_path", "filePath", "TargetFile"],
        )
    {
        paths.push(path);
    }
    // The new text a change writes, per file, from its `diff` content.
    let new_text = |path: &str| {
        tool_call
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|part| {
                part.get("type").and_then(Value::as_str) == Some("diff")
                    && part.get("path").and_then(Value::as_str) == Some(path)
            })
            .and_then(|part| part.get("newText").and_then(Value::as_str))
            .map(str::to_string)
            .or_else(|| text_of(input, &["content", "Content", "new_string", "newText"]))
            .unwrap_or_default()
    };
    let none = |what: &str| format!("its `{kind}` request names no {what}");
    Ok(match kind {
        "execute" => vec![AgentAction::RunCommand {
            command: text_of(input, COMMAND_KEYS)
                .or_else(|| (!title.is_empty()).then(|| title.to_string()))
                .ok_or_else(|| none("command"))?,
            working_dir: text_of(input, &["cwd", "Cwd", "directory", "workdir"])
                .or_else(|| working_dir.map(str::to_string)),
        }],
        "edit" | "delete" | "move" => {
            if paths.is_empty() {
                return Err(none("file"));
            }
            paths
                .iter()
                .map(|path| AgentAction::WriteFile {
                    content: new_text(path),
                    path: path.clone(),
                })
                .collect()
        }
        "read" => {
            if paths.is_empty() {
                return Err(none("file"));
            }
            paths
                .into_iter()
                .map(|path| AgentAction::ReadFile { path })
                .collect()
        }
        "search" => vec![AgentAction::Search {
            pattern: text_of(input, &["pattern", "query", "Pattern", "Query"])
                .or_else(|| (!title.is_empty()).then(|| title.to_string())),
            glob: text_of(input, &["glob", "Glob"]),
            path: paths.into_iter().next(),
        }],
        "fetch" => vec![AgentAction::Fetch {
            url: text_of(input, URL_KEYS).ok_or_else(|| none("URL"))?,
        }],
        other => {
            return Err(format!(
                "AIDaemon has no rule for the coding agent's `{other}` action, so it was not allowed"
            ));
        }
    })
}

/// The model a session works with, as `session/new` reports it: ACP's
/// `models.currentModelId`, else its `model` config option's value.
pub(crate) fn current_model(session: &Value) -> Option<String> {
    let id = session
        .pointer("/models/currentModelId")
        .and_then(Value::as_str)
        .or_else(|| {
            session
                .get("configOptions")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|option| option.get("category").and_then(Value::as_str) == Some("model"))
                .and_then(|option| option.get("currentValue").and_then(Value::as_str))
        })?;
    (!id.trim().is_empty()).then(|| id.to_string())
}

/// The models a session offers, as `session/new` lists them
/// (`models.availableModels`: each one's id and name).
pub(crate) fn available_models(session: &Value) -> Vec<crate::AgentModel> {
    session
        .pointer("/models/availableModels")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| {
            let id = model.get("modelId").and_then(Value::as_str)?.trim();
            if id.is_empty() {
                return None;
            }
            let name = model
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.trim().is_empty())
                .unwrap_or(id);
            Some(crate::AgentModel {
                id: id.to_string(),
                name: name.to_string(),
            })
        })
        .collect()
}

/// The answer to a permission request: the option of the kind wanted, the
/// once-only one (an "always" option would keep a rule in the agent).
pub fn outcome(options: &[Value], allowed: bool) -> Value {
    let wanted = if allowed { "allow_once" } else { "reject_once" };
    match options
        .iter()
        .find(|option| option.get("kind").and_then(Value::as_str) == Some(wanted))
        .and_then(|option| option.get("optionId").cloned())
    {
        Some(option_id) => json!({"outcome": {"outcome": "selected", "optionId": option_id}}),
        // No such option: the request is withdrawn, which allows nothing.
        None => json!({"outcome": {"outcome": "cancelled"}}),
    }
}

#[derive(Default)]
struct HostState {
    session_id: Option<String>,
    /// Its `initialize` reply said it can load an earlier session
    /// (`agentCapabilities.loadSession`).
    loads_sessions: bool,
    /// Between `session/load` and its reply: the updates are the session's
    /// history, replayed.
    loading: bool,
    /// Its message since its last tool call: the turn's final message.
    message: String,
    answers: HostAnswers,
    /// Its turn ended, and how: `Some(true)` with `end_turn`.
    completed: Option<bool>,
}

/// Starts an [`AcpHost`].
pub struct AcpHostStarter {
    /// The agent's name, for the owner's words.
    pub agent: String,
    pub working_dir: Option<String>,
    /// The model the owner chose (`None`: the agent's own).
    pub model: Option<String>,
    /// The earlier session to continue (`None`: a new one).
    pub resume: Option<String>,
}

#[async_trait]
impl HostStarter for AcpHostStarter {
    async fn start(
        self: Box<Self>,
        stdin: ChildStdin,
        judge: Arc<dyn ActionJudge>,
        prompt: &str,
    ) -> Arc<dyn AgentHost> {
        let mut host = AcpHost::new(stdin, judge, self.working_dir, prompt);
        host.agent = Arc::from(self.agent);
        host.model = self.model.map(Arc::from);
        host.resume = self.resume.map(Arc::from);
        host.stdin.write(&initialize_line()).await;
        Arc::new(host)
    }
}

/// Hosts one agent's session for its run.
#[derive(Clone)]
pub struct AcpHost {
    stdin: Stdin,
    judge: Arc<dyn ActionJudge>,
    /// The agent's name, for the owner's words.
    agent: Arc<str>,
    working_dir: Option<String>,
    prompt: Arc<str>,
    /// The model the owner chose, set before the task is sent.
    model: Option<Arc<str>>,
    /// The earlier session the run continues (`session/load`).
    resume: Option<Arc<str>>,
    state: Arc<Mutex<HostState>>,
}

impl AcpHost {
    pub fn new(
        stdin: ChildStdin,
        judge: Arc<dyn ActionJudge>,
        working_dir: Option<String>,
        prompt: &str,
    ) -> Self {
        Self {
            stdin: Stdin::new(stdin),
            judge,
            agent: Arc::from("The coding agent"),
            working_dir,
            prompt: Arc::from(prompt),
            model: None,
            resume: None,
            state: Arc::default(),
        }
    }

    async fn send_prompt(&self, session: &str) {
        let params = json!({
            "sessionId": session,
            "prompt": [{"type": "text", "text": &*self.prompt}],
        });
        self.stdin
            .write(&request_line(PROMPT_ID, "session/prompt", params))
            .await;
    }

    /// The run ends on a protocol error, in `text`: stdin closes, so the
    /// agent exits.
    async fn fail(&self, text: String) -> Vec<String> {
        self.end(false);
        self.stdin.close().await;
        vec![json!({"type": "error", "message": text}).to_string()]
    }

    /// Its session is open (new or loaded): the chosen model is set, else
    /// the task sent, and the run's reader told the session's id (what a
    /// later run continues), its model, and whether it can be continued.
    async fn opened(&self, session: String, result: &Value) -> Vec<String> {
        let resumable = match self.state.lock() {
            Ok(mut state) => {
                state.session_id = Some(session.clone());
                state.loads_sessions
            }
            Err(_) => false,
        };
        let model = match self.model.as_deref() {
            Some(model) => {
                let params = json!({"sessionId": session, "modelId": model});
                self.stdin
                    .write(&request_line(SET_MODEL_ID, "session/set_model", params))
                    .await;
                Some(model.to_string())
            }
            None => {
                self.send_prompt(&session).await;
                current_model(result)
            }
        };
        vec![
            json!({
                "type": "acp.session",
                "session_id": session,
                "model": model,
                "resumable": resumable,
            })
            .to_string(),
        ]
    }

    fn end(&self, completed: bool) -> String {
        let Ok(mut state) = self.state.lock() else {
            return String::new();
        };
        state.completed = Some(completed);
        state.message.trim().to_string()
    }

    async fn on_response(&self, id: &Value, message: &Value) -> Vec<String> {
        if let Some(error) = message.get("error") {
            let mut text = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("the agent refused the request")
                .to_string();
            match id.as_u64() {
                Some(SET_MODEL_ID) => {
                    text = format!(
                        "It could not switch to the model chosen for it ({}): {text}",
                        self.model.as_deref().unwrap_or_default()
                    );
                }
                Some(SESSION_LOAD_ID) => {
                    text = format!(
                        "{} could not load its earlier session ({}): {text}",
                        self.agent,
                        self.resume.as_deref().unwrap_or_default()
                    );
                }
                _ => {}
            }
            return self.fail(text).await;
        }
        let result = message.get("result").unwrap_or(&Value::Null);
        match id.as_u64() {
            Some(INITIALIZE_ID) => {
                let loads_sessions = result
                    .pointer("/agentCapabilities/loadSession")
                    .and_then(Value::as_bool)
                    == Some(true);
                let mut params = json!({"mcpServers": []});
                if let Some(dir) = self.working_dir.as_deref() {
                    params["cwd"] = json!(dir);
                }
                let Some(resume) = self.resume.as_deref() else {
                    if let Ok(mut state) = self.state.lock() {
                        state.loads_sessions = loads_sessions;
                    }
                    self.stdin
                        .write(&request_line(SESSION_NEW_ID, "session/new", params))
                        .await;
                    return Vec::new();
                };
                if !loads_sessions {
                    // Asked to continue a session it has no way to load: no
                    // prompt is sent, so nothing starts over in its place.
                    return self
                        .fail(format!(
                            "{} can't continue an earlier session: it doesn't offer to load one",
                            self.agent
                        ))
                        .await;
                }
                params["sessionId"] = json!(resume);
                if let Ok(mut state) = self.state.lock() {
                    state.loads_sessions = true;
                    state.loading = true;
                }
                self.stdin
                    .write(&request_line(SESSION_LOAD_ID, "session/load", params))
                    .await;
                Vec::new()
            }
            Some(SESSION_NEW_ID) => {
                let session = result
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.opened(session, result).await
            }
            // The load's reply names no session of its own (the protocol's
            // has none): it is the one asked for, its history replayed by now.
            Some(SESSION_LOAD_ID) => {
                if let Ok(mut state) = self.state.lock() {
                    state.loading = false;
                }
                let session = self.resume.as_deref().unwrap_or_default().to_string();
                self.opened(session, result).await
            }
            Some(SET_MODEL_ID) => {
                let session = self
                    .state
                    .lock()
                    .ok()
                    .and_then(|state| state.session_id.clone())
                    .unwrap_or_default();
                self.send_prompt(&session).await;
                Vec::new()
            }
            Some(PROMPT_ID) => {
                let stop = result
                    .get("stopReason")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let completed = stop == "end_turn";
                let message = self.end(completed);
                // Its turn is over: closing stdin lets it exit.
                self.stdin.close().await;
                if completed {
                    vec![
                        json!({
                            "type": "result",
                            "subtype": "success",
                            "result": message,
                            "final_message": message,
                        })
                        .to_string(),
                    ]
                } else {
                    vec![
                        json!({
                            "type": "result",
                            "subtype": "error",
                            "is_error": true,
                            "result": format!("The agent stopped ({stop}). {message}").trim().to_string(),
                            "final_message": message,
                        })
                        .to_string(),
                    ]
                }
            }
            _ => Vec::new(),
        }
    }

    fn on_update(&self, update: &Value) -> Vec<String> {
        let kind = update
            .get("sessionUpdate")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Ok(mut state) = self.state.lock() else {
            return Vec::new();
        };
        // A loading session's history, replayed: done before this run, so
        // neither a step nor its message.
        if state.loading {
            return Vec::new();
        }
        match kind {
            "agent_message_chunk" => {
                if let Some(text) = update.pointer("/content/text").and_then(Value::as_str) {
                    state.message.push_str(text);
                }
                Vec::new()
            }
            // What it says before a tool call is not its final message.
            // Each tool call goes on with its typed fields (its id, `kind`,
            // status, the paths it touches and the command it runs), so the
            // run's reader records the step without reading its title.
            "tool_call" => {
                state.message.clear();
                vec![tool_call_line(update)]
            }
            "tool_call_update" => match update.get("status").and_then(Value::as_str) {
                Some(status @ ("completed" | "failed")) => vec![
                    json!({
                        "type": "tool_call_update",
                        "id": update.get("toolCallId"),
                        "status": status,
                    })
                    .to_string(),
                ],
                _ => Vec::new(),
            },
            "plan" => {
                vec![json!({"type": "acp.plan", "entries": update.get("entries")}).to_string()]
            }
            _ => Vec::new(),
        }
    }

    async fn answer(&self, method: &str, id: &Value, params: &Value) {
        if method != "session/request_permission" {
            tracing::info!(%method, "A coding agent's request this host does not serve");
            self.stdin
                .write(&error_line(id, "not supported by this host"))
                .await;
            return;
        }
        let tool_call = params.get("toolCall").unwrap_or(&Value::Null);
        let options: Vec<Value> = params
            .get("options")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let (label, verdict) = match actions(tool_call, self.working_dir.as_deref()) {
            Ok(actions) => {
                let label = actions
                    .iter()
                    .map(AgentAction::label)
                    .collect::<Vec<_>>()
                    .join("; ");
                let mut verdict = Verdict::Allow;
                for action in actions {
                    verdict = self.judge.judge(action).await;
                    if !verdict.is_allowed() {
                        break;
                    }
                }
                (label, verdict)
            }
            Err(reason) => (
                tool_call
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("an action")
                    .to_string(),
                Verdict::Deny { reason },
            ),
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
        self.stdin
            .write(&reply_line(id, outcome(&options, verdict.is_allowed())))
            .await;
    }
}

#[async_trait]
impl AgentHost for AcpHost {
    async fn on_line(&self, line: &str) -> Vec<String> {
        let Ok(message) = serde_json::from_str::<Value>(line.trim()) else {
            return vec![line.to_string()];
        };
        let method = message.get("method").and_then(Value::as_str);
        match (method, message.get("id")) {
            (Some(method), Some(id)) => {
                let host = self.clone();
                let method = method.to_string();
                let id = id.clone();
                let params = message.get("params").cloned().unwrap_or(Value::Null);
                tokio::spawn(async move { host.answer(&method, &id, &params).await });
                Vec::new()
            }
            (None, Some(id)) => self.on_response(id, &message).await,
            (Some("session/update"), None) => message
                .pointer("/params/update")
                .map(|update| self.on_update(update))
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn answers(&self) -> HostAnswers {
        self.state
            .lock()
            .map(|state| state.answers.clone())
            .unwrap_or_default()
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// The requests `agy acp` 1.5.10 sent in the live probe (2026-11-18).
    fn write_request() -> Value {
        json!({
            "sessionUpdate": "tool_call", "toolCallId": "call_1", "name": "write_file",
            "status": "pending", "kind": "edit", "title": "File note.txt",
            "locations": [{"path": "/w/project/note.txt"}],
            "rawInput": {"Content": "ok", "Overwrite": true, "Path": "/w/project/note.txt"},
            "content": [{"type": "diff", "path": "/w/project/note.txt", "oldText": null, "newText": "ok"}]
        })
    }

    fn command_request() -> Value {
        json!({
            "sessionUpdate": "tool_call", "toolCallId": "call_2", "name": "run_command",
            "status": "pending", "kind": "execute",
            "title": "curl -sS https://example.com",
            "rawInput": {"CommandLine": "curl -sS https://example.com", "WaitMsBeforeAsync": 5000}
        })
    }

    #[test]
    fn each_request_is_the_action_its_kind_says() {
        assert_eq!(
            actions(&write_request(), Some("/w/project")).unwrap(),
            vec![AgentAction::WriteFile {
                path: "/w/project/note.txt".into(),
                content: "ok".into(),
            }]
        );
        assert_eq!(
            actions(&command_request(), Some("/w/project")).unwrap(),
            vec![AgentAction::RunCommand {
                command: "curl -sS https://example.com".into(),
                working_dir: Some("/w/project".into()),
            }]
        );
        assert_eq!(
            actions(
                &json!({"kind": "execute", "rawInput": {"command": ["git", "status"], "cwd": "/w/x"}}),
                None
            )
            .unwrap(),
            vec![AgentAction::RunCommand {
                command: "git status".into(),
                working_dir: Some("/w/x".into()),
            }]
        );
        assert_eq!(
            actions(
                &json!({"kind": "read", "locations": [{"path": "/etc/hosts"}]}),
                None
            )
            .unwrap(),
            vec![AgentAction::ReadFile {
                path: "/etc/hosts".into()
            }]
        );
        assert_eq!(
            actions(
                &json!({"kind": "fetch", "rawInput": {"url": "https://example.com"}}),
                None
            )
            .unwrap(),
            vec![AgentAction::Fetch {
                url: "https://example.com".into()
            }]
        );
        // Unknown kinds and changes naming no file are refused.
        assert!(actions(&json!({"kind": "other", "title": "send mail"}), None).is_err());
        assert!(actions(&json!({"kind": "edit"}), None).is_err());
    }

    #[test]
    fn a_tool_calls_line_carries_its_command_under_any_agents_key() {
        let line = tool_call_line(&json!({
            "sessionUpdate": "tool_call", "toolCallId": "call_9", "kind": "execute",
            "rawInput": {"CommandLine": "python3 -m pytest -q", "WaitMsBeforeAsync": 5000}
        }));
        let events = super::super::events::read_events(&line);
        assert!(matches!(&events[0], crate::AgentEvent::Step(step)
            if step.tool == crate::StepTool::Run { program: Some("python3".into()) }));
    }

    #[test]
    fn the_answer_is_the_agents_once_only_option() {
        let options = vec![
            json!({"optionId": "always", "name": "Always", "kind": "allow_always"}),
            json!({"optionId": "allow", "name": "Allow", "kind": "allow_once"}),
            json!({"optionId": "deny", "name": "Deny", "kind": "reject_once"}),
        ];
        assert_eq!(
            outcome(&options, true),
            json!({"outcome": {"outcome": "selected", "optionId": "allow"}})
        );
        assert_eq!(
            outcome(&options, false),
            json!({"outcome": {"outcome": "selected", "optionId": "deny"}})
        );
        // With no once-only option to refuse with, the request is withdrawn.
        assert_eq!(
            outcome(&options[..2], false),
            json!({"outcome": {"outcome": "cancelled"}})
        );
    }

    struct Judge(fn(&AgentAction) -> Verdict, Mutex<Vec<AgentAction>>);

    #[async_trait]
    impl ActionJudge for Judge {
        async fn judge(&self, action: AgentAction) -> Verdict {
            let verdict = (self.0)(&action);
            self.1.lock().unwrap().push(action);
            verdict
        }
    }

    /// A stand-in agent on the other end of real pipes: the host opens the
    /// session, sends the task, answers a permission request with the
    /// judge's verdict, and ends with the final message as a result line.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_session_is_opened_answered_and_closed_over_the_pipes() {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};

        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("cat");
        let judge = Arc::new(Judge(
            |action| match action {
                AgentAction::RunCommand { .. } => Verdict::Declined {
                    reason: "[OWNER DECLINED] not now".into(),
                },
                _ => Verdict::Allow,
            },
            Mutex::default(),
        ));
        let host = AcpHost::new(
            child.stdin.take().unwrap(),
            judge.clone(),
            Some("/w/project".into()),
            "make the note",
        );
        let mut sent = BufReader::new(child.stdout.take().unwrap()).lines();
        macro_rules! next {
            () => {
                serde_json::from_str::<Value>(&sent.next_line().await.unwrap().unwrap()).unwrap()
            };
        }

        host.on_line(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1}}"#)
            .await;
        let session_new = next!();
        assert_eq!(session_new["method"], "session/new");
        assert_eq!(session_new["params"]["cwd"], "/w/project");
        host.on_line(r#"{"jsonrpc":"2.0","id":2,"result":{"sessionId":"s1"}}"#)
            .await;
        let prompt = next!();
        assert_eq!(prompt["params"]["sessionId"], "s1");
        assert_eq!(prompt["params"]["prompt"][0]["text"], "make the note");

        let request = |id: u64, tool_call: Value| {
            json!({"jsonrpc": "2.0", "id": id, "method": "session/request_permission", "params": {
                "sessionId": "s1", "toolCall": tool_call,
                "options": [{"optionId": "allow", "kind": "allow_once"}, {"optionId": "deny", "kind": "reject_once"}]
            }})
            .to_string()
        };
        host.on_line(&request(7, write_request())).await;
        assert_eq!(next!()["result"]["outcome"]["optionId"], "allow");
        host.on_line(&request(8, command_request())).await;
        assert_eq!(next!()["result"]["outcome"]["optionId"], "deny");

        host.on_line(r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Created note.txt."}}}}"#).await;
        assert!(host.unfinished());
        let end = host
            .on_line(r#"{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}"#)
            .await;
        let end: Value = serde_json::from_str(&end[0]).unwrap();
        assert_eq!(end["type"], "result");
        assert_eq!(end["result"], "Created note.txt.");
        assert_eq!(
            super::super::events::read_events(&end.to_string())[0],
            crate::AgentEvent::Reported {
                text: "Created note.txt.".into(),
                report: None
            }
        );
        assert!(!host.unfinished());
        // The end of its turn closed stdin.
        assert!(sent.next_line().await.unwrap().is_none());
        let _ = child.wait().await;
        assert_eq!(
            host.answers().declined,
            vec!["command: curl -sS https://example.com".to_string()]
        );
        assert_eq!(judge.1.lock().unwrap().len(), 2);
    }

    /// With a model chosen, the host switches to it before sending the
    /// task; without one, the session's own model is what the run reports.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_chosen_model_is_set_before_the_task_and_reported() {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};

        let start = |model: Option<&str>| {
            let mut child = tokio::process::Command::new("cat")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("cat");
            let mut host = AcpHost::new(
                child.stdin.take().unwrap(),
                Arc::new(crate::AllowAll),
                None,
                "fix the test",
            );
            host.model = model.map(Arc::from);
            let sent = BufReader::new(child.stdout.take().unwrap()).lines();
            (child, host, sent)
        };
        let session = r#"{"jsonrpc":"2.0","id":2,"result":{"sessionId":"s1","models":{"currentModelId":"synthetic-large","availableModels":[{"modelId":"synthetic-large","name":"Synthetic Large"}]}}}"#;

        let (_child, host, mut sent) = start(Some("synthetic-small"));
        let line = host.on_line(session).await;
        let reported: Value = serde_json::from_str(&line[0]).unwrap();
        assert_eq!(reported["model"], "synthetic-small");
        let set: Value = serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(set["method"], "session/set_model");
        assert_eq!(set["params"]["modelId"], "synthetic-small");
        assert!(
            host.on_line(r#"{"jsonrpc":"2.0","id":4,"result":{}}"#)
                .await
                .is_empty()
        );
        let prompt: Value =
            serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(prompt["method"], "session/prompt");
        assert_eq!(prompt["params"]["sessionId"], "s1");

        // A model it does not know ends the run with its own reason.
        let (_child, host, _sent) = start(Some("synthetic-missing"));
        host.on_line(session).await;
        let refused = host
            .on_line(r#"{"jsonrpc":"2.0","id":4,"error":{"code":-32602,"message":"Unknown model ID: synthetic-missing"}}"#)
            .await;
        let refused: Value = serde_json::from_str(&refused[0]).unwrap();
        assert_eq!(refused["type"], "error");
        assert!(
            refused["message"]
                .as_str()
                .unwrap()
                .contains("Unknown model ID: synthetic-missing"),
            "{refused}"
        );

        let (_child, host, mut sent) = start(None);
        let line = host.on_line(session).await;
        let reported: Value = serde_json::from_str(&line[0]).unwrap();
        assert_eq!(reported["model"], "synthetic-large");
        let events = super::super::events::read_events(&line[0]);
        assert_eq!(
            events,
            vec![crate::AgentEvent::Session {
                id: "s1".into(),
                model: Some("synthetic-large".into()),
                // Its `initialize` reply (not sent here) offered no load.
                resumable: Some(false),
            }]
        );
        let prompt: Value =
            serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(prompt["method"], "session/prompt");
    }

    /// A host over `cat`, as the named agent, continuing `resume` when set.
    #[cfg(unix)]
    fn hosted(
        resume: Option<&str>,
    ) -> (
        tokio::process::Child,
        AcpHost,
        tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    ) {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};

        let mut child = tokio::process::Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("cat");
        let mut host = AcpHost::new(
            child.stdin.take().unwrap(),
            Arc::new(crate::AllowAll),
            Some("/w/project".into()),
            "go on with the note",
        );
        host.agent = Arc::from("Synthetic Agent");
        host.resume = resume.map(Arc::from);
        let sent = BufReader::new(child.stdout.take().unwrap()).lines();
        (child, host, sent)
    }

    const LOADS: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentCapabilities":{"loadSession":true}}}"#;
    const NO_LOAD: &str =
        r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1,"agentCapabilities":{}}}"#;

    /// Continuing an earlier session: it is loaded, its replayed history
    /// reaches the run as nothing (no step, plan or message), and the task
    /// goes to the same session once the load's reply comes.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_earlier_session_is_loaded_its_history_dropped_then_prompted() {
        let (_child, host, mut sent) = hosted(Some("synthetic-session-1"));
        macro_rules! next {
            () => {
                serde_json::from_str::<Value>(&sent.next_line().await.unwrap().unwrap()).unwrap()
            };
        }
        assert!(host.on_line(LOADS).await.is_empty());
        let load = next!();
        assert_eq!(load["method"], "session/load");
        assert_eq!(
            load["params"],
            json!({"sessionId": "synthetic-session-1", "cwd": "/w/project", "mcpServers": []})
        );

        let update = |update: Value| {
            json!({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "synthetic-session-1", "update": update,
            }})
            .to_string()
        };
        for replayed in [
            json!({"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "make the note"}}),
            json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Made it earlier."}}),
            write_request(),
            json!({"sessionUpdate": "tool_call_update", "toolCallId": "call_1", "status": "completed"}),
            json!({"sessionUpdate": "plan", "entries": [{"content": "Write the note", "status": "completed"}]}),
        ] {
            assert!(host.on_line(&update(replayed)).await.is_empty());
        }

        let opened = host
            .on_line(r#"{"jsonrpc":"2.0","id":5,"result":{"models":{"currentModelId":"synthetic-large"}}}"#)
            .await;
        assert_eq!(
            super::super::events::read_events(&opened[0]),
            vec![crate::AgentEvent::Session {
                id: "synthetic-session-1".into(),
                model: Some("synthetic-large".into()),
                resumable: Some(true),
            }]
        );
        let prompt = next!();
        assert_eq!(prompt["method"], "session/prompt");
        assert_eq!(prompt["params"]["sessionId"], "synthetic-session-1");
        assert_eq!(prompt["params"]["prompt"][0]["text"], "go on with the note");

        // What it does now is the run's again.
        assert_eq!(host.on_line(&update(command_request())).await.len(), 1);
        host.on_line(&update(json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Added a line."}})))
            .await;
        let end = host
            .on_line(r#"{"jsonrpc":"2.0","id":3,"result":{"stopReason":"end_turn"}}"#)
            .await;
        let end: Value = serde_json::from_str(&end[0]).unwrap();
        assert_eq!(end["result"], "Added a line.");
        assert!(!host.unfinished());
    }

    /// An agent that offers no load: asked to continue, its run ends with
    /// that said before any prompt; a new session it opens can't be
    /// continued later.
    #[cfg(unix)]
    #[tokio::test]
    async fn without_load_a_run_cannot_continue_and_a_new_session_is_not_resumable() {
        let (_child, host, mut sent) = hosted(Some("synthetic-session-1"));
        let refused = host.on_line(NO_LOAD).await;
        let refused: Value = serde_json::from_str(&refused[0]).unwrap();
        assert_eq!(refused["type"], "error");
        assert_eq!(
            refused["message"],
            "Synthetic Agent can't continue an earlier session: it doesn't offer to load one"
        );
        assert!(host.unfinished());
        // Nothing was sent, and stdin is closed.
        assert!(sent.next_line().await.unwrap().is_none());

        for (initialized, resumable) in [(NO_LOAD, false), (LOADS, true)] {
            let (_child, host, mut sent) = hosted(None);
            host.on_line(initialized).await;
            let new: Value =
                serde_json::from_str(&sent.next_line().await.unwrap().unwrap()).unwrap();
            assert_eq!(new["method"], "session/new");
            let opened = host
                .on_line(r#"{"jsonrpc":"2.0","id":2,"result":{"sessionId":"synthetic-session-2"}}"#)
                .await;
            assert!(matches!(
                &super::super::events::read_events(&opened[0])[0],
                crate::AgentEvent::Session { id, resumable: Some(kept), .. }
                    if id == "synthetic-session-2" && *kept == resumable
            ));
        }
    }

    /// A session the agent can't find ends the run with its own reason.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_failed_load_ends_the_run_with_its_reason() {
        let (_child, host, mut sent) = hosted(Some("synthetic-missing"));
        host.on_line(LOADS).await;
        assert!(sent.next_line().await.unwrap().is_some());
        let failed = host
            .on_line(r#"{"jsonrpc":"2.0","id":5,"error":{"code":-32002,"message":"Session not found: synthetic-missing"}}"#)
            .await;
        let failed: Value = serde_json::from_str(&failed[0]).unwrap();
        assert_eq!(failed["type"], "error");
        assert_eq!(
            failed["message"],
            "Synthetic Agent could not load its earlier session (synthetic-missing): Session not found: synthetic-missing"
        );
        assert!(sent.next_line().await.unwrap().is_none());
    }
}
