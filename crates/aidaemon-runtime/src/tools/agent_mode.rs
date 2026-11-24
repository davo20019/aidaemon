//! The daemon's side of the coding agents it hands work to
//! ([`aidaemon_coding_agents`]): which agents the Settings page lists, and
//! how a hosted agent's request is judged. Each agent's permission mode is
//! the owner's choice in `config.toml`, capped by AIdaemon's level
//! ([`crate::permissions`]).
//!
//! A request is judged as a call of the daemon's own tool for that action
//! (a command is `terminal`, a write `write_file`, a network host a fetch
//! from it), through the kernel's nested-call seam: the owner's grants
//! answer what they cover, a card asks the rest, and a decline binds the
//! run, exactly as for the model's own calls.

use std::sync::{Arc, Mutex, OnceLock};

use aidaemon_coding_agents::{ActionJudge, AgentAction, Verdict};
pub use aidaemon_coding_agents::{
    AgentModel, AgentModels, Checked, CodingAgent, CodingAgents, PermissionMode, Readiness,
    modes_within,
};
use aidaemon_core::{NestedCalls, ToolCall};
use async_trait::async_trait;
use serde_json::json;

/// One agent the daemon can hand work to, by its name among its tools.
#[derive(Clone)]
pub struct AgentInfo {
    pub name: String,
    pub agent: Arc<dyn CodingAgent>,
    /// How it is launched: the command, and what precedes the agent's own
    /// arguments (a launcher's: `env FOO=1 codex`), for asking it whether
    /// it can work.
    pub command: String,
    pub prefix: Vec<String>,
    /// It runs on this machine (only such an agent is asked).
    pub local: bool,
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

/// The agents the daemon can hand work to now, preferred first (matching
/// [`CodingAgents::builtin`] preference order), then alphabetically by name.
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
    let preference: Vec<&'static str> = CodingAgents::builtin()
        .all()
        .iter()
        .filter_map(|agent| agent.discovery().map(|found| found.name))
        .collect();
    let rank = |name: &str| {
        preference
            .iter()
            .position(|preferred| preferred.eq_ignore_ascii_case(name))
            .unwrap_or(usize::MAX)
    };
    agents.sort_by(|a, b| {
        rank(&a.name)
            .cmp(&rank(&b.name))
            .then_with(|| a.name.cmp(&b.name))
    });
    agents
}

/// The agent named `name` now, when there is one.
pub fn agent_named(name: &str) -> Option<Arc<dyn CodingAgent>> {
    agents()
        .into_iter()
        .find(|info| info.name == name)
        .map(|info| info.agent)
}

fn readiness_cache() -> &'static Mutex<std::collections::HashMap<String, Readiness>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, Readiness>>> = OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

/// Whether the agent named `name` can do work, as it last answered
/// (`Unchecked` before it is asked).
pub fn readiness_of(name: &str) -> Readiness {
    readiness_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(name).cloned())
        .unwrap_or(Readiness::Unchecked)
}

fn remember(name: &str, readiness: Readiness) {
    tracing::info!(agent = %name, ?readiness, "A coding agent says whether it can work");
    if let Ok(mut cache) = readiness_cache().lock() {
        cache.insert(name.to_string(), readiness);
    }
}

fn models_cache() -> &'static Mutex<std::collections::HashMap<String, AgentModels>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, AgentModels>>> = OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

/// The models the agent named `name` listed when it was last asked, and
/// the one it uses unless told otherwise (`None`: it lists none, or was
/// not asked yet).
pub fn models_of(name: &str) -> Option<AgentModels> {
    models_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(name).cloned())
}

/// Keep an agent's answer: its readiness, and its models when it listed
/// them (an answer that lists none keeps the ones it listed before).
fn remember_checked(name: &str, checked: Checked) {
    if let Some(models) = checked.models
        && let Ok(mut cache) = models_cache().lock()
    {
        cache.insert(name.to_string(), models);
    }
    remember(name, checked.readiness);
}

/// Ask one agent now (its own probe; one that cannot be asked, or does not
/// run here, is unchecked).
async fn ask(info: &AgentInfo) -> Checked {
    let mut checked: Checked = match info.agent.readiness_probe() {
        Some(probe) if info.local => {
            aidaemon_coding_agents::check_models(&probe, &info.command, &info.prefix, None).await
        }
        _ => Readiness::Unchecked.into(),
    };
    // One whose check does not list its models is asked for them, unless
    // it said it cannot work.
    if checked.models.is_none()
        && info.local
        && checked.readiness.is_usable()
        && let Some(listing) = info.agent.model_listing()
    {
        checked.models =
            aidaemon_coding_agents::list_models(listing, &info.command, &info.prefix, None).await;
    }
    checked
}

/// Ask the agent named `name` again, and keep its answer.
pub async fn recheck(name: &str) -> Readiness {
    let Some(info) = agents().into_iter().find(|info| info.name == name) else {
        return Readiness::Unchecked;
    };
    let checked = ask(&info).await;
    let readiness = checked.readiness.clone();
    remember_checked(name, checked);
    readiness
}

/// Ask every agent again, all at once, and keep their answers.
pub async fn check_all() {
    let checks = agents().into_iter().map(|info| async move {
        let checked = ask(&info).await;
        remember_checked(&info.name, checked);
    });
    futures::future::join_all(checks).await;
}

/// Ask the agents not asked yet.
pub async fn ensure_checked() {
    let unasked: Vec<AgentInfo> = agents()
        .into_iter()
        .filter(|info| {
            readiness_cache()
                .lock()
                .is_ok_and(|cache| !cache.contains_key(&info.name))
        })
        .collect();
    let checks = unasked.into_iter().map(|info| async move {
        let checked = ask(&info).await;
        remember_checked(&info.name, checked);
    });
    futures::future::join_all(checks).await;
}

/// Its last run could not sign in (the run's own typed signal): not usable
/// until it is asked again and answers otherwise.
pub fn mark_not_usable(name: &str, reason: &str, fix: Option<&str>) {
    remember(
        name,
        Readiness::NotUsable {
            reason: reason.to_string(),
            fix: fix.map(str::to_string),
        },
    );
}

/// Judges a hosted agent's requests through the run's nested-call seam,
/// falling back to a detached safety judge when the launching turn's tool
/// batch has already returned (`async_mode`, queued runs, or timeout detach).
pub struct DaemonJudge {
    nested: Option<NestedCalls>,
    detached: Option<Arc<crate::approvals::DetachedNestedJudge>>,
    /// The run whose agent asks, named on any card its asks raise.
    asker: Option<Asker>,
    /// Cancels any in-flight detached approval card when the hosted run stops.
    cancel: tokio_util::sync::CancellationToken,
    /// Number of approval checks currently awaiting an answer (so the run's
    /// idle watchdog pauses while the owner is deciding).
    in_flight: Arc<std::sync::atomic::AtomicUsize>,
}

struct InFlightGuard(Arc<std::sync::atomic::AtomicUsize>);

impl InFlightGuard {
    fn enter(counter: &Arc<std::sync::atomic::AtomicUsize>) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self(counter.clone())
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl DaemonJudge {
    pub fn new(nested: NestedCalls) -> Self {
        Self {
            nested: Some(nested),
            detached: None,
            asker: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            in_flight: Arc::default(),
        }
    }

    /// Its asks name run `asker` on the cards they raise.
    pub fn for_run(nested: NestedCalls, asker: Asker) -> Self {
        Self {
            nested: Some(nested),
            detached: None,
            asker: Some(asker),
            cancel: tokio_util::sync::CancellationToken::new(),
            in_flight: Arc::default(),
        }
    }

    /// Judge via `nested` while the launching turn is active, falling back to
    /// `detached` once the launching tool batch has returned.
    pub fn for_run_with_detached(
        nested: Option<NestedCalls>,
        detached: Option<Arc<crate::approvals::DetachedNestedJudge>>,
        asker: Asker,
    ) -> Self {
        let cancel = detached
            .as_ref()
            .map(|d| d.child_token())
            .unwrap_or_default();
        Self {
            nested,
            detached,
            asker: Some(asker),
            cancel,
            in_flight: Arc::default(),
        }
    }

    /// Attach a per-run cancellation token and in-flight approval counter.
    pub fn with_controls(
        mut self,
        cancel: tokio_util::sync::CancellationToken,
        in_flight: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        self.cancel = cancel;
        self.in_flight = in_flight;
        self
    }

    async fn authorize(&self, call: ToolCall) -> Verdict {
        let _active = InFlightGuard::enter(&self.in_flight);
        if self.cancel.is_cancelled() {
            return Verdict::Deny {
                reason: "the run was stopped before this call could be checked".to_string(),
            };
        }
        if let Some(nested) = &self.nested {
            let verdict = tokio::select! {
                biased;
                () = self.cancel.cancelled() => Verdict::Deny {
                    reason: "the run was stopped before this call could be checked".to_string(),
                },
                verdict = nested.authorize(call.clone()) => verdict,
            };
            let run_gone = matches!(
                &verdict,
                Verdict::Deny { reason } if reason == "the run ended before this call could be checked"
            );
            if !run_gone || self.detached.is_none() {
                return verdict;
            }
        }
        if let Some(detached) = &self.detached {
            return detached.authorize_with_cancel(call, &self.cancel).await;
        }
        Verdict::Deny {
            reason: "the run ended before this call could be checked".to_string(),
        }
    }
}

/// A coding agent's run, as a card it raises names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asker {
    /// The run's id.
    pub run: String,
    /// Its agent ("Codex").
    pub agent: String,
}

/// The asks being judged now, by the id of the daemon call each is judged
/// as ([`daemon_call`] makes it unique): the card an ask raises reads its
/// run here ([`asker_of`]).
static ASKING: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, Asker>>> =
    std::sync::LazyLock::new(Default::default);

/// The run whose agent's ask is daemon call `call_id`, while it is judged.
pub fn asker_of(call_id: &str) -> Option<Asker> {
    ASKING
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(call_id)
        .cloned()
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
        let call = daemon_call(&action);
        let Some(asker) = &self.asker else {
            return self.authorize(call).await;
        };
        let id = call.id.clone();
        ASKING
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id.clone(), asker.clone());
        let verdict = self.authorize(call).await;
        ASKING.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
        verdict
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

    #[test]
    fn an_agent_is_unchecked_until_it_answers_and_a_failed_sign_in_marks_it() {
        assert_eq!(readiness_of("synthetic-never-asked"), Readiness::Unchecked);
        mark_not_usable(
            "synthetic-signed-out",
            "It couldn't sign in on its last run.",
            None,
        );
        assert!(!readiness_of("synthetic-signed-out").is_usable());
        remember("synthetic-signed-out", Readiness::Ready);
        assert!(readiness_of("synthetic-signed-out").is_usable());
    }

    /// A run's judge names that run while its ask is judged (the card it
    /// raises reads it), and only then.
    #[tokio::test]
    async fn an_ask_names_its_run_while_it_is_judged() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
        let during = seen.clone();
        let (nested, _asked) = aidaemon_core::testing::nested_calls(move |call| {
            *during.lock().unwrap() = Some((call.id.clone(), asker_of(&call.id)));
            Verdict::Allow
        });
        let judge = DaemonJudge::for_run(
            nested,
            Asker {
                run: "cli-agent-synthetic".into(),
                agent: "Codex".into(),
            },
        );
        let verdict = judge
            .judge(AgentAction::RunCommand {
                command: "pytest".into(),
                working_dir: None,
            })
            .await;
        assert!(verdict.is_allowed());
        let (id, asker) = seen.lock().unwrap().clone().expect("the ask was judged");
        assert_eq!(
            asker.map(|asker| asker.run),
            Some("cli-agent-synthetic".to_string())
        );
        assert_eq!(asker_of(&id), None, "gone once judged");
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

    /// When a hosted agent runs detached (`async_mode`, queued, or timeout
    /// fallback) and its launching tool batch has already returned (dropping
    /// `NestedCalls`'s receiver), `DaemonJudge` falls back to its owned
    /// `DetachedNestedJudge` instead of failing with `the run ended`.
    #[tokio::test]
    async fn detached_judge_handles_asks_after_nested_channel_closes() {
        use aidaemon_core as sdk;
        struct AllowTool;
        #[async_trait]
        impl sdk::Tool for AllowTool {
            fn spec(&self) -> sdk::ToolSpec {
                sdk::ToolSpec {
                    name: "terminal".into(),
                    description: "terminal".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                _ctx: &sdk::ToolContext,
            ) -> Result<sdk::ToolOutput, sdk::ToolError> {
                Ok(sdk::ToolOutput::text("ok"))
            }
        }
        let mut tools: std::collections::HashMap<String, Arc<dyn sdk::Tool>> =
            std::collections::HashMap::new();
        tools.insert("terminal".into(), Arc::new(AllowTool));
        let approvals = Arc::new(crate::approvals::OwnerApprovals::default());
        let rules = Arc::new(crate::approvals::AppApprovalRules::default());
        let detached = Arc::new(crate::approvals::DetachedNestedJudge::new(
            tools,
            Vec::new(),
            approvals,
            rules,
            sdk::SharedDeclines::default(),
            sdk::RunId::new(),
            "app:synthetic".into(),
            "app:synthetic".into(),
            None,
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
            false,
        ));
        let judge = DaemonJudge::for_run_with_detached(
            None,
            Some(detached),
            Asker {
                run: "cli-agent-detached".into(),
                agent: "Codex".into(),
            },
        );
        let verdict = judge
            .judge(AgentAction::RunCommand {
                command: "cargo test".into(),
                working_dir: Some("/w/synthetic".into()),
            })
            .await;
        assert!(
            verdict.is_allowed(),
            "detached judge should allow: {verdict:?}"
        );
    }

    /// Cancelling the run's token immediately aborts an in-flight approval
    /// wait and decrements the `in_flight` counter back to zero.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelling_daemon_judge_aborts_pending_approval_and_tracks_in_flight() {
        let (arrived_tx, arrived_rx) = tokio::sync::oneshot::channel::<()>();
        let arrived_tx = std::sync::Mutex::new(Some(arrived_tx));
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = std::sync::Mutex::new(release_rx);
        let (nested, _asked) = aidaemon_core::testing::nested_calls(move |_call| {
            if let Some(tx) = arrived_tx.lock().unwrap().take() {
                let _ = tx.send(());
            }
            let _ = release_rx
                .lock()
                .unwrap()
                .recv_timeout(std::time::Duration::from_secs(5));
            Verdict::Allow
        });
        let cancel = tokio_util::sync::CancellationToken::new();
        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let judge = DaemonJudge::new(nested).with_controls(cancel.clone(), in_flight.clone());

        let judge_task = tokio::spawn(async move {
            judge
                .judge(AgentAction::RunCommand {
                    command: "cargo test".into(),
                    working_dir: None,
                })
                .await
        });

        // Wait until the nested call arrives (so the judge is actively waiting).
        tokio::time::timeout(std::time::Duration::from_secs(2), arrived_rx)
            .await
            .expect("nested call should arrive")
            .expect("arrived sender open");
        assert_eq!(
            in_flight.load(std::sync::atomic::Ordering::Acquire),
            1,
            "in_flight should be 1 while waiting for owner approval"
        );

        // Cancel the run's token without ever answering the approval request.
        cancel.cancel();
        let verdict = tokio::time::timeout(std::time::Duration::from_secs(2), judge_task)
            .await
            .expect("cancelled judge should abort promptly")
            .unwrap();
        let _ = release_tx.send(());
        assert!(
            matches!(verdict, Verdict::Deny { ref reason } if reason.contains("stopped")),
            "cancelled judge should deny as stopped: {verdict:?}"
        );
        assert_eq!(
            in_flight.load(std::sync::atomic::Ordering::Acquire),
            0,
            "in_flight should return to 0 after abort"
        );
    }
}
