//! Whether an installed coding agent can do work now: signed in, served by
//! its provider. Each agent says how to ask it ([`Probe`]), without a task
//! and without a model call; [`check`] asks. An agent whose answer lists
//! its models (an ACP session) says which ones, and which it uses unless
//! told otherwise ([`check_models`]).

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Whether an agent can do work now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Readiness {
    /// It answered that it is ready.
    Ready,
    /// It answered that it is not: its own reason, and how its owner fixes
    /// it when that is known.
    NotUsable { reason: String, fix: Option<String> },
    /// It could not be asked, or did not say.
    Unchecked,
}

impl Readiness {
    pub fn is_usable(&self) -> bool {
        !matches!(self, Self::NotUsable { .. })
    }
}

/// One model an agent offers: its id (what selects it) and its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentModel {
    pub id: String,
    pub name: String,
}

/// The models an agent said it offers, as it answered a check.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentModels {
    /// The one it uses when it is given none (its own default).
    pub current: Option<String>,
    /// Every one it offers, in its order.
    pub available: Vec<AgentModel>,
}

/// An agent's answer to a check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checked {
    pub readiness: Readiness,
    /// Its models, when its answer listed them.
    pub models: Option<AgentModels>,
}

impl From<Readiness> for Checked {
    fn from(readiness: Readiness) -> Self {
        Self {
            readiness,
            models: None,
        }
    }
}

/// How an agent is asked.
#[derive(Debug, Clone)]
pub enum Probe {
    /// Run it with these arguments (in place of its own) and read the
    /// answer: its exit code and its stdout.
    Status {
        args: Vec<String>,
        read: fn(Option<i32>, &str) -> Readiness,
    },
    /// Open an Agent Client Protocol session (the arguments start its
    /// server): it opens, or the agent says why not. No prompt is sent.
    AcpSession { serve: Vec<String> },
}

/// The longest an agent is given to answer a check. A check is not work:
/// an agent that does not answer in this time is reported unchecked, never
/// unusable.
const ANSWER_WITHIN: Duration = Duration::from_secs(30);

/// Ask an agent launched as `command` with `prefix` (a launcher's own
/// arguments, `env FOO=1 ...`), in `cwd`.
pub async fn check(
    probe: &Probe,
    command: &str,
    prefix: &[String],
    cwd: Option<&Path>,
) -> Readiness {
    check_models(probe, command, prefix, cwd).await.readiness
}

/// [`check`], with the models the agent listed in its answer.
pub async fn check_models(
    probe: &Probe,
    command: &str,
    prefix: &[String],
    cwd: Option<&Path>,
) -> Checked {
    match tokio::time::timeout(ANSWER_WITHIN, ask(probe, command, prefix, cwd)).await {
        Ok(checked) => checked,
        Err(_) => Readiness::Unchecked.into(),
    }
}

async fn ask(probe: &Probe, command: &str, prefix: &[String], cwd: Option<&Path>) -> Checked {
    let mut process = tokio::process::Command::new(command);
    process
        .args(prefix)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        // It must not see itself run inside another Claude Code session.
        .env_remove("CLAUDECODE");
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }
    match probe {
        Probe::Status { args, read } => {
            let Ok(output) = process.args(args).stdin(Stdio::null()).output().await else {
                return Readiness::Unchecked.into();
            };
            read(
                output.status.code(),
                &String::from_utf8_lossy(&output.stdout),
            )
            .into()
        }
        Probe::AcpSession { serve } => {
            let Ok(mut child) = process.args(serve).stdin(Stdio::piped()).spawn() else {
                return Readiness::Unchecked.into();
            };
            let (Some(mut stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
                return Readiness::Unchecked.into();
            };
            let request = |id: u64, method: &str, params: Value| {
                let mut line =
                    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
                        .to_string();
                line.push('\n');
                line
            };
            let mut lines = BufReader::new(stdout).lines();
            let first = request(
                1,
                "initialize",
                json!({"protocolVersion": 1, "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false}, "terminal": false}}),
            );
            if stdin.write_all(first.as_bytes()).await.is_err() {
                return Readiness::Unchecked.into();
            }
            // A session needs a folder; with none given, the temp folder.
            let cwd = cwd
                .map(Path::to_path_buf)
                .unwrap_or_else(std::env::temp_dir)
                .to_string_lossy()
                .into_owned();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(message) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                // Only the answers to its two requests count (JSON-RPC ids).
                match (
                    message.get("id").and_then(Value::as_u64),
                    message.get("method"),
                ) {
                    (Some(1), None) => {
                        if let Some(error) = message.get("error") {
                            return not_usable(error).into();
                        }
                        let params = json!({"mcpServers": [], "cwd": cwd});
                        if stdin
                            .write_all(request(2, "session/new", params).as_bytes())
                            .await
                            .is_err()
                        {
                            return Readiness::Unchecked.into();
                        }
                    }
                    (Some(2), None) => {
                        return match message.get("error") {
                            Some(error) => not_usable(error).into(),
                            None => {
                                let session = message.get("result").unwrap_or(&Value::Null);
                                let available = crate::acp::host::available_models(session);
                                let current = crate::acp::host::current_model(session);
                                Checked {
                                    readiness: Readiness::Ready,
                                    models: (current.is_some() || !available.is_empty())
                                        .then_some(AgentModels { current, available }),
                                }
                            }
                        };
                    }
                    _ => {}
                }
            }
            Readiness::Unchecked.into()
        }
    }
}

/// The agent's own refusal, in its words.
fn not_usable(error: &Value) -> Readiness {
    Readiness::NotUsable {
        reason: error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("it refused to start a session")
            .to_string(),
        fix: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exit_code(code: Option<i32>, _stdout: &str) -> Readiness {
        if code == Some(0) {
            Readiness::Ready
        } else {
            Readiness::NotUsable {
                reason: "not signed in".into(),
                fix: Some("sign in".into()),
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_status_command_is_read_by_its_exit_code() {
        let probe = |code: &str| Probe::Status {
            args: vec!["-c".into(), format!("exit {code}")],
            read: exit_code,
        };
        assert_eq!(check(&probe("0"), "sh", &[], None).await, Readiness::Ready);
        assert!(!check(&probe("1"), "sh", &[], None).await.is_usable());
        // A program that cannot run is not known to be unusable.
        assert_eq!(
            check(&probe("0"), "/synthetic/missing-agent", &[], None).await,
            Readiness::Unchecked
        );
    }

    /// A stand-in agent that opens the protocol and then refuses or opens a
    /// session.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_acp_agent_is_ready_when_its_session_opens() {
        let agent = |session_reply: &str| {
            format!(
                r#"read init; echo '{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":1}}}}'; read new; echo '{session_reply}'"#
            )
        };
        let probe = |script: String| Probe::AcpSession {
            serve: vec!["-c".into(), script],
        };
        let ready = probe(agent(
            r#"{"jsonrpc":"2.0","id":2,"result":{"sessionId":"s"}}"#,
        ));
        assert_eq!(check(&ready, "sh", &[], None).await, Readiness::Ready);
        let listed = probe(agent(
            r#"{"jsonrpc":"2.0","id":2,"result":{"sessionId":"s","models":{"currentModelId":"synthetic-large","availableModels":[{"modelId":"synthetic-large","name":"Synthetic Large"},{"modelId":"synthetic-small","name":"Synthetic Small"}]}}}"#,
        ));
        assert_eq!(
            check_models(&listed, "sh", &[], None).await,
            Checked {
                readiness: Readiness::Ready,
                models: Some(AgentModels {
                    current: Some("synthetic-large".into()),
                    available: vec![
                        AgentModel {
                            id: "synthetic-large".into(),
                            name: "Synthetic Large".into(),
                        },
                        AgentModel {
                            id: "synthetic-small".into(),
                            name: "Synthetic Small".into(),
                        },
                    ],
                }),
            }
        );
        assert_eq!(check_models(&ready, "sh", &[], None).await.models, None);
        let refused = probe(agent(
            r#"{"jsonrpc":"2.0","id":2,"error":{"code":-32000,"message":"This client is no longer supported for this account."}}"#,
        ));
        assert_eq!(
            check(&refused, "sh", &[], None).await,
            Readiness::NotUsable {
                reason: "This client is no longer supported for this account.".into(),
                fix: None,
            }
        );
    }
}

/// Asks the agents installed on this machine (run by hand:
/// `cargo test -p aidaemon-coding-agents -- --ignored --nocapture`).
#[cfg(test)]
mod live {
    #[tokio::test]
    #[ignore = "asks the agents installed on this machine"]
    async fn the_installed_agents_say_whether_they_can_work() {
        for agent in crate::CodingAgents::builtin().all() {
            let (Some(found), Some(probe)) = (agent.discovery(), agent.readiness_probe()) else {
                continue;
            };
            let readiness = super::check(&probe, found.command, &[], None).await;
            println!("{}: {readiness:?}", agent.name());
        }
    }
}
