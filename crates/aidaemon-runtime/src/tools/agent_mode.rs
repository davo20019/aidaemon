//! The daemon's side of the coding agents it hands work to
//! ([`aidaemon_coding_agents`]): where each agent's permission mode is kept,
//! which agents the Settings page lists, and how a hosted agent's request
//! is judged.
//!
//! A request is judged as a call of the daemon's own tool for that action
//! (a command is `terminal`, a write `write_file`, a network host a fetch
//! from it), through the kernel's nested-call seam: the owner's grants
//! answer what they cover, a card asks the rest, and a decline binds the
//! run, exactly as for the model's own calls.

use std::sync::{Arc, Mutex, OnceLock};

use aidaemon_coding_agents::{ActionJudge, AgentAction, ModeStore, Verdict};
pub use aidaemon_coding_agents::{CodingAgent, CodingAgents, PermissionMode};
use aidaemon_core::{NestedCalls, ToolCall};
use aidaemon_tools::services::SettingsStore;
use async_trait::async_trait;
use serde_json::json;

/// The daemon's settings store, as the place modes are kept.
pub struct SettingsModes<'a, S: ?Sized>(pub &'a S);

#[async_trait]
impl<S: SettingsStore + ?Sized> ModeStore for SettingsModes<'_, S> {
    async fn load(&self, key: &str) -> Option<String> {
        self.0.get_setting(key).await.ok().flatten()
    }

    async fn save(&self, key: &str, value: &str) -> Result<(), String> {
        self.0
            .set_setting(key, value)
            .await
            .map_err(|error| format!("The mode could not be saved: {error}"))
    }
}

/// The mode the agent named `name` runs in.
pub async fn mode_of(
    store: &(impl SettingsStore + ?Sized),
    name: &str,
    agent: &dyn CodingAgent,
) -> Option<PermissionMode> {
    aidaemon_coding_agents::mode_of(&SettingsModes(store), name, agent).await
}

/// Keep the owner's choice of `mode` for the agent named `name`.
pub async fn set_mode(
    store: &(impl SettingsStore + ?Sized),
    name: &str,
    agent: &dyn CodingAgent,
    mode: PermissionMode,
) -> Result<(), String> {
    aidaemon_coding_agents::set_mode(&SettingsModes(store), name, agent, mode).await
}

/// One agent the daemon can hand work to, by its name among its tools.
#[derive(Clone)]
pub struct AgentInfo {
    pub name: String,
    pub agent: Arc<dyn CodingAgent>,
}

/// Lists one `cli_agent` tool's agents; empty once the tool is gone.
pub type Lister = Arc<dyn Fn() -> Vec<AgentInfo> + Send + Sync>;

fn registry() -> &'static Mutex<Vec<Lister>> {
    static REGISTRY: OnceLock<Mutex<Vec<Lister>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Add a tool's agents to the ones listed. Every live tool is read (the
/// daemon builds one; tests build many), so one never hides another's.
pub fn publish(lister: Lister) {
    if let Ok(mut registry) = registry().lock() {
        registry.push(lister);
    }
}

/// The agents the daemon can hand work to now, by name, each once.
pub fn agents() -> Vec<AgentInfo> {
    let listers = registry()
        .lock()
        .map(|registry| registry.clone())
        .unwrap_or_default();
    let mut agents: Vec<AgentInfo> = Vec::new();
    for lister in listers {
        for info in lister() {
            if !agents.iter().any(|known| known.name == info.name) {
                agents.push(info);
            }
        }
    }
    agents.sort_by(|a, b| a.name.cmp(&b.name));
    agents
}

/// The agent named `name` now, when there is one.
pub fn agent_named(name: &str) -> Option<Arc<dyn CodingAgent>> {
    agents()
        .into_iter()
        .find(|info| info.name == name)
        .map(|info| info.agent)
}

/// Judges a hosted agent's requests through the run's nested-call seam.
pub struct DaemonJudge {
    nested: NestedCalls,
}

impl DaemonJudge {
    pub fn new(nested: NestedCalls) -> Self {
        Self { nested }
    }
}

/// The daemon call an action is judged as: the daemon tool whose judgment
/// applies, with arguments its schema takes.
pub fn daemon_call(action: &AgentAction) -> ToolCall {
    let id = format!("coding-agent-{}", uuid::Uuid::new_v4());
    let (tool, arguments) = match action {
        AgentAction::RunCommand {
            command,
            working_dir,
        } => {
            let mut arguments = json!({"action": "run", "command": command});
            if let Some(dir) = working_dir {
                arguments["working_dir"] = json!(dir);
            }
            ("terminal", arguments)
        }
        AgentAction::WriteFile { path, content } => {
            ("write_file", json!({"path": path, "content": content}))
        }
        AgentAction::EditFile {
            path,
            edits,
            replace_all,
        } => match edits.as_slice() {
            [edit] => (
                "edit_file",
                json!({
                    "path": path,
                    "old_text": edit.old_text,
                    "new_text": edit.new_text,
                    "replace_all": replace_all,
                }),
            ),
            edits => (
                "edit_file",
                json!({
                    "path": path,
                    "edits": edits
                        .iter()
                        .map(|edit| json!({"old_text": edit.old_text, "new_text": edit.new_text}))
                        .collect::<Vec<_>>(),
                }),
            ),
        },
        AgentAction::ReadFile { path } => ("read_file", json!({"path": path})),
        AgentAction::Search {
            pattern,
            glob,
            path,
        } => {
            let mut arguments = json!({});
            if let Some(pattern) = pattern {
                arguments["pattern"] = json!(pattern);
            }
            if let Some(glob) = glob {
                arguments["glob"] = json!(glob);
            }
            if let Some(path) = path {
                arguments["path"] = json!(path);
            }
            ("search_files", arguments)
        }
        AgentAction::Fetch { url } => ("web_fetch", json!({"url": url})),
        // An egress to that host, judged as a fetch from it.
        AgentAction::ReachHost { host } => {
            ("web_fetch", json!({"url": format!("https://{host}/")}))
        }
    };
    ToolCall::new(id, tool, arguments)
}

#[async_trait]
impl ActionJudge for DaemonJudge {
    async fn judge(&self, action: AgentAction) -> Verdict {
        self.nested.authorize(daemon_call(&action)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_coding_agents::TextEdit;

    #[test]
    fn each_action_is_the_daemon_tool_whose_judgment_applies() {
        let call = daemon_call(&AgentAction::RunCommand {
            command: "touch probe.txt".into(),
            working_dir: Some("/w/synthetic".into()),
        });
        assert_eq!(call.name, "terminal");
        assert_eq!(
            call.arguments,
            json!({"action": "run", "command": "touch probe.txt", "working_dir": "/w/synthetic"})
        );
        let call = daemon_call(&AgentAction::WriteFile {
            path: "/w/a.md".into(),
            content: "x".into(),
        });
        assert_eq!(call.name, "write_file");
        assert_eq!(call.arguments, json!({"path": "/w/a.md", "content": "x"}));
        let call = daemon_call(&AgentAction::EditFile {
            path: "/w/a.md".into(),
            edits: vec![TextEdit {
                old_text: "a".into(),
                new_text: "b".into(),
            }],
            replace_all: false,
        });
        assert_eq!(call.name, "edit_file");
        assert_eq!(call.arguments["old_text"], "a");
        let call = daemon_call(&AgentAction::ReadFile {
            path: "/etc/synthetic.conf".into(),
        });
        assert_eq!(call.name, "read_file");
        let call = daemon_call(&AgentAction::Search {
            pattern: None,
            glob: Some("*.rs".into()),
            path: None,
        });
        assert_eq!(call.name, "search_files");
        assert_eq!(call.arguments, json!({"glob": "*.rs"}));
        let call = daemon_call(&AgentAction::ReachHost {
            host: "example.com".into(),
        });
        assert_eq!(call.name, "web_fetch");
        assert_eq!(call.arguments, json!({"url": "https://example.com/"}));
    }

    #[tokio::test]
    async fn the_judge_asks_the_run() {
        let (nested, asked) = aidaemon_core::testing::nested_calls(|call| {
            if call.name == "terminal" {
                Verdict::Declined {
                    reason: "[OWNER DECLINED] not today".into(),
                }
            } else {
                Verdict::Allow
            }
        });
        let judge = DaemonJudge::new(nested);
        let verdict = judge
            .judge(AgentAction::RunCommand {
                command: "rm -rf /w/x".into(),
                working_dir: None,
            })
            .await;
        assert!(matches!(verdict, Verdict::Declined { .. }));
        assert_eq!(asked.lock().unwrap()[0].name, "terminal");
    }
}
