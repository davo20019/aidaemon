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
    AgentModel, AgentModels, Checked, CodingAgent, CodingAgents, PermissionMode, Readiness, SignIn,
    SignInMethod, modes_within,
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
    /// The sign-in it runs with, as the owner chose it (`Err`: it can't
    /// sign in that way now, in words for the owner).
    pub sign_in: Result<aidaemon_coding_agents::SignIn, String>,
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

/// An agent's last answer, and when it was given.
struct Answer {
    readiness: Readiness,
    at: std::time::Instant,
}

fn readiness_cache() -> &'static Mutex<std::collections::HashMap<String, Answer>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, Answer>>> = OnceLock::new();
    CACHE.get_or_init(Mutex::default)
}

/// How long an agent that could be asked but gave no answer (its check
/// timed out or it closed first, as a slow first start does) waits before
/// [`ensure_checked`] asks it again.
const ASK_UNANSWERED_AGAIN_AFTER: std::time::Duration = std::time::Duration::from_secs(60);

/// Whether the agent named `name` can do work, as it last answered
/// (`Unchecked` before it is asked).
pub fn readiness_of(name: &str) -> Readiness {
    readiness_cache()
        .lock()
        .ok()
        .and_then(|cache| cache.get(name).map(|answer| answer.readiness.clone()))
        .unwrap_or(Readiness::Unchecked)
}

fn remember(name: &str, readiness: Readiness) {
    tracing::info!(agent = %name, ?readiness, "A coding agent says whether it can work");
    if let Ok(mut cache) = readiness_cache().lock() {
        cache.insert(
            name.to_string(),
            Answer {
                readiness,
                at: std::time::Instant::now(),
            },
        );
    }
}

/// A run of the agent named `name` finished on its own with success: it
/// can work, whatever its check said (a check that timed out left it
/// unchecked).
pub fn mark_ready_after_run(name: &str) {
    if readiness_of(name) != Readiness::Ready {
        remember(name, Readiness::Ready);
    }
}

/// Whether [`ensure_checked`] asks `info` now: never asked, or it can be
/// asked (it runs here and has a check) yet its last check gave no answer,
/// long enough ago.
fn due_for_check(info: &AgentInfo, now: std::time::Instant) -> bool {
    let Ok(cache) = readiness_cache().lock() else {
        return false;
    };
    match cache.get(&info.name) {
        None => true,
        Some(answer) => {
            answer.readiness == Readiness::Unchecked
                && info.local
                && info.agent.readiness_probe().is_some()
                && now.saturating_duration_since(answer.at) >= ASK_UNANSWERED_AGAIN_AFTER
        }
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
    // Asked as a run launches it: with the sign-in the owner chose.
    let signed = match info
        .sign_in
        .clone()
        .and_then(|sign_in| info.agent.signed_in(&sign_in))
    {
        Ok(signed) => signed,
        Err(reason) => return Readiness::NotUsable { reason, fix: None }.into(),
    };
    let mut checked: Checked = match info.agent.readiness_probe() {
        Some(probe) if info.local => {
            aidaemon_coding_agents::check_models(
                &probe.signed(signed),
                &info.command,
                &info.prefix,
                None,
            )
            .await
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

/// Ask the agents not asked yet, and, without waiting for them, the ones
/// whose last check gave no answer a while ago (a slow first start must
/// not hide an agent until the daemon restarts, nor stall every caller on
/// one that hangs).
pub async fn ensure_checked() {
    let now = std::time::Instant::now();
    let (unasked, unanswered): (Vec<AgentInfo>, Vec<AgentInfo>) = agents()
        .into_iter()
        .filter(|info| due_for_check(info, now))
        .partition(|info| {
            readiness_cache()
                .lock()
                .is_ok_and(|cache| !cache.contains_key(&info.name))
        });
    for info in unanswered {
        // Asked once at a time: until its answer, it reads as asked now.
        remember(&info.name, Readiness::Unchecked);
        tokio::spawn(async move {
            let checked = ask(&info).await;
            remember_checked(&info.name, checked);
        });
    }
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

fn usage_limits() -> &'static Mutex<std::collections::HashMap<String, chrono::DateTime<chrono::Utc>>>
{
    static LIMITS: OnceLock<
        Mutex<std::collections::HashMap<String, chrono::DateTime<chrono::Utc>>>,
    > = OnceLock::new();
    LIMITS.get_or_init(Mutex::default)
}

/// The agent named `name` said its account is out of usage until
/// `resets_at` (its own typed answer, asked after a run of it failed).
pub fn mark_out_of_usage(name: &str, resets_at: chrono::DateTime<chrono::Utc>) {
    if let Ok(mut limits) = usage_limits().lock() {
        limits.insert(name.to_string(), resets_at);
    }
}

/// Until when the agent named `name` is out of usage, while it still is.
pub fn out_of_usage(name: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let mut limits = usage_limits().lock().ok()?;
    match limits.get(name).copied() {
        Some(until) if until > chrono::Utc::now() => Some(until),
        Some(_) => {
            limits.remove(name);
            None
        }
        None => None,
    }
}

/// Ask the agent named `name` whether its account is out of usage, when it
/// can be asked ([`aidaemon_coding_agents::CodingAgent::usage_check`]):
/// its name in plain words and when the usage resets. Remembered until
/// then ([`out_of_usage`]).
pub async fn check_usage(name: &str) -> Option<(String, Option<chrono::DateTime<chrono::Utc>>)> {
    let info = agents()
        .into_iter()
        .find(|info| info.name == name && info.local)?;
    let check = info.agent.usage_check()?;
    let limit = aidaemon_coding_agents::usage_limit(check, &info.command, &info.prefix).await?;
    let resets_at = limit
        .resets_at_ms
        .and_then(|ms| i64::try_from(ms).ok())
        .and_then(chrono::DateTime::<chrono::Utc>::from_timestamp_millis);
    if let Some(at) = resets_at {
        mark_out_of_usage(name, at);
    }
    Some((info.agent.name().to_string(), resets_at))
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

pub use crate::approvals::agent_asks::{Asker, asker_of};

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
        crate::approvals::agent_asks::begin(&id, asker.clone());
        let verdict = self.authorize(call).await;
        crate::approvals::agent_asks::end(&id);
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

    /// A check that gave no answer (timed out on a slow first start) is
    /// asked again after a while, not kept for the daemon's life; an agent
    /// with no check of its own is not; a finished run marks it ready.
    #[test]
    fn an_unanswered_check_is_asked_again_and_a_finished_run_marks_ready() {
        let info = |name: &str, local: bool| AgentInfo {
            name: name.to_string(),
            agent: Arc::new(aidaemon_coding_agents::acp::AcpAgent::custom(
                "Synthetic ACP",
                "synthetic-acp",
                &["acp"],
            )),
            command: "synthetic-acp".into(),
            prefix: Vec::new(),
            local,
            sign_in: Ok(SignIn::default()),
        };
        let slow = info("synthetic-slow-start", true);
        let now = std::time::Instant::now();
        assert!(due_for_check(&slow, now), "never asked");
        remember(&slow.name, Readiness::Unchecked);
        assert!(!due_for_check(&slow, now), "just asked");
        let later = now + ASK_UNANSWERED_AGAIN_AFTER + std::time::Duration::from_secs(1);
        assert!(due_for_check(&slow, later), "unanswered, a while ago");
        let remote = info("synthetic-remote", false);
        remember(&remote.name, Readiness::Unchecked);
        assert!(!due_for_check(&remote, later), "it can't be asked here");
        mark_ready_after_run(&slow.name);
        assert_eq!(readiness_of(&slow.name), Readiness::Ready);
        assert!(!due_for_check(&slow, later), "answered");
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
