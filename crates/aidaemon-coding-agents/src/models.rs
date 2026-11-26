//! Which models an installed coding agent offers, asked of the agent itself
//! without a task and without a model call. An ACP agent says so when its
//! session opens (its readiness check reads it); Claude Code and Codex are
//! asked here ([`ModelListing`]), each over its own protocol:
//!
//! - Claude Code: the `initialize` control request of its stream-json
//!   protocol answers with `models` (each one's `value`, `displayName` and
//!   `resolvedModel`; `default` is the one it uses unless told otherwise).
//! - Codex: its app server's `model/list` (each one's `id`, `displayName`,
//!   `hidden`, `isDefault`) and `config/read` (the owner's `model`, which
//!   wins over the listed default).

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::{AgentModel, AgentModels};

/// How an agent is asked for its models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelListing {
    /// Claude Code's stream-json `initialize`.
    Claude,
    /// Codex's app server (`model/list`, `config/read`).
    Codex,
}

/// The longest an agent is given to list its models.
const ANSWER_WITHIN: Duration = Duration::from_secs(30);

/// Ask the agent launched as `command` (after a launcher's `prefix`) which
/// models it offers. `None`: it could not be asked, or did not say.
pub async fn list_models(
    listing: ModelListing,
    command: &str,
    prefix: &[String],
    cwd: Option<&Path>,
) -> Option<AgentModels> {
    tokio::time::timeout(ANSWER_WITHIN, ask(listing, command, prefix, cwd))
        .await
        .ok()
        .flatten()
}

async fn ask(
    listing: ModelListing,
    command: &str,
    prefix: &[String],
    cwd: Option<&Path>,
) -> Option<AgentModels> {
    let mut process = tokio::process::Command::new(command);
    process
        .args(prefix)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        // It must not see itself run inside another Claude Code session.
        .env_remove("CLAUDECODE");
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }
    let (args, requests): (&[&str], Vec<Value>) = match listing {
        ModelListing::Claude => (
            &[
                "-p",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
                "--verbose",
            ],
            vec![json!({
                "type": "control_request",
                "request_id": "models",
                "request": {"subtype": "initialize"},
            })],
        ),
        ModelListing::Codex => (
            &["app-server"],
            vec![
                json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "clientInfo": {"name": "aidaemon", "title": "AIdaemon", "version": env!("CARGO_PKG_VERSION")},
                }}),
                json!({"jsonrpc": "2.0", "method": "initialized"}),
                json!({"jsonrpc": "2.0", "id": 2, "method": "model/list", "params": {}}),
                json!({"jsonrpc": "2.0", "id": 3, "method": "config/read", "params": {}}),
            ],
        ),
    };
    let mut child = process.args(args).spawn().ok()?;
    let mut stdin = child.stdin.take()?;
    let stdout = child.stdout.take()?;
    for request in requests {
        let mut line = request.to_string();
        line.push('\n');
        stdin.write_all(line.as_bytes()).await.ok()?;
    }
    let mut lines = BufReader::new(stdout).lines();
    let (mut listed, mut configured) = (None, None);
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match listing {
            ModelListing::Claude => {
                // Only the answer to its own request (a protocol field).
                if message.get("type").and_then(Value::as_str) == Some("control_response")
                    && message
                        .pointer("/response/request_id")
                        .and_then(Value::as_str)
                        == Some("models")
                {
                    return claude_models(message.pointer("/response/response")?);
                }
            }
            ModelListing::Codex => match message.get("id").and_then(Value::as_u64) {
                Some(2) => listed = Some(message.get("result").cloned().unwrap_or_default()),
                Some(3) => {
                    configured = Some(message.get("result").cloned().unwrap_or_default());
                }
                _ => {}
            },
        }
        if let (Some(listed), Some(configured)) = (&listed, &configured) {
            return codex_models(listed, configured);
        }
    }
    None
}

fn named(id: &str, name: Option<&str>) -> AgentModel {
    AgentModel {
        id: id.to_string(),
        name: name
            .filter(|name| !name.trim().is_empty())
            .unwrap_or(id)
            .to_string(),
    }
}

/// Claude Code's `initialize` answer: its `models`. The `default` entry is
/// not a model of its own: the one it uses is the entry that resolves to
/// the same model (its alias, `opus`), else that model's id.
fn claude_models(answer: &Value) -> Option<AgentModels> {
    let entries = answer.get("models")?.as_array()?;
    let field = |entry: &Value, key: &str| {
        entry
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    let mut available = Vec::new();
    let mut default_resolves_to = None;
    for entry in entries {
        let Some(value) = field(entry, "value") else {
            continue;
        };
        if value == "default" {
            default_resolves_to = field(entry, "resolvedModel");
            continue;
        }
        available.push((
            named(&value, field(entry, "displayName").as_deref()),
            field(entry, "resolvedModel"),
        ));
    }
    let current = default_resolves_to.map(|resolved| {
        available
            .iter()
            .find(|(_, resolves)| resolves.as_deref() == Some(resolved.as_str()))
            .map_or(resolved, |(model, _)| model.id.clone())
    });
    let available: Vec<AgentModel> = available.into_iter().map(|(model, _)| model).collect();
    (current.is_some() || !available.is_empty()).then_some(AgentModels { current, available })
}

/// Codex's `model/list` and `config/read` answers: its shown models, and
/// the one it uses (the owner's configured model, else the listed default).
fn codex_models(listed: &Value, configured: &Value) -> Option<AgentModels> {
    let entries = listed.get("data").and_then(Value::as_array)?;
    let mut available = Vec::new();
    let mut listed_default = None;
    for entry in entries {
        if entry.get("hidden").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let Some(id) = entry
            .get("model")
            .or_else(|| entry.get("id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
        else {
            continue;
        };
        if entry.get("isDefault").and_then(Value::as_bool) == Some(true) {
            listed_default = Some(id.to_string());
        }
        available.push(named(id, entry.get("displayName").and_then(Value::as_str)));
    }
    let current = configured
        .pointer("/config/model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .or(listed_default);
    (current.is_some() || !available.is_empty()).then_some(AgentModels { current, available })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_codes_default_is_the_alias_it_resolves_to() {
        let answer = json!({"models": [
            {"value": "default", "resolvedModel": "synthetic-large-5", "displayName": "Default (recommended)"},
            {"value": "large", "resolvedModel": "synthetic-large-5", "displayName": "Large 5"},
            {"value": "small", "resolvedModel": "synthetic-small-5", "displayName": "Small 5"},
        ]});
        assert_eq!(
            claude_models(&answer),
            Some(AgentModels {
                current: Some("large".into()),
                available: vec![
                    AgentModel {
                        id: "large".into(),
                        name: "Large 5".into()
                    },
                    AgentModel {
                        id: "small".into(),
                        name: "Small 5".into()
                    },
                ],
            })
        );
        assert_eq!(claude_models(&json!({"commands": []})), None);
    }

    #[test]
    fn codexs_configured_model_wins_over_its_listed_default() {
        let listed = json!({"data": [
            {"id": "synthetic-sol", "model": "synthetic-sol", "displayName": "Synthetic Sol", "hidden": false, "isDefault": true},
            {"id": "synthetic-old", "model": "synthetic-old", "displayName": "Old", "hidden": true, "isDefault": false},
            {"id": "synthetic-astra", "model": "synthetic-astra", "displayName": "Synthetic Astra", "hidden": false, "isDefault": false},
        ]});
        let models =
            codex_models(&listed, &json!({"config": {"model": "synthetic-astra"}})).unwrap();
        assert_eq!(models.current.as_deref(), Some("synthetic-astra"));
        assert_eq!(
            models
                .available
                .iter()
                .map(|m| m.id.as_str())
                .collect::<Vec<_>>(),
            ["synthetic-sol", "synthetic-astra"]
        );
        let models = codex_models(&listed, &json!({"config": {"model": null}})).unwrap();
        assert_eq!(models.current.as_deref(), Some("synthetic-sol"));
    }

    /// A stand-in Codex app server answering its two questions.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_codex_app_server_is_asked_over_its_pipes() {
        let script = r#"read a; read b; read c; read d;
echo '{"id":1,"result":{}}'
echo '{"id":2,"result":{"data":[{"model":"synthetic-sol","displayName":"Synthetic Sol","isDefault":true}]}}'
echo '{"id":3,"result":{"config":{}}}'"#;
        let models = list_models(
            ModelListing::Codex,
            "sh",
            &["-c".into(), script.into(), "sh".into()],
            None,
        )
        .await
        .unwrap();
        assert_eq!(models.current.as_deref(), Some("synthetic-sol"));
    }
}

/// Asks the agents installed on this machine (run by hand:
/// `cargo test -p aidaemon-coding-agents models::live -- --ignored --nocapture`).
#[cfg(test)]
mod live {
    #[tokio::test]
    #[ignore = "asks the agents installed on this machine"]
    async fn the_installed_agents_list_their_models() {
        for agent in crate::CodingAgents::builtin().all() {
            let (Some(found), Some(listing)) = (agent.discovery(), agent.model_listing()) else {
                continue;
            };
            let models = super::list_models(listing, found.command, &[], None).await;
            println!("{}: {models:?}", agent.name());
        }
    }
}
