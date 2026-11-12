//! Owner approvals on the SDK kernel: the host's approval rules, the
//! owner's session grants, and the resolver that asks the owner.
//!
//! The kernel's `ApprovalPolicy` (registered last, after `UntrustedHook`)
//! consults [`AppApprovalRules`] for the daemon's own facts and the chat's
//! [`SessionGrants`]; an ask becomes a typed
//! [`sdk::InterruptKind::OwnerApproval`] interruption, which
//! [`DaemonInterruptResolver`] settles by sending the owner a card through
//! the [`ApprovalBroker`]. A denial, a timeout and a send failure all deny,
//! and the kernel then refuses every later call of the run that overlaps the
//! declined effects.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core as sdk;
use aidaemon_policies::{ApprovalRules, RuleDecision, SessionGrants};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::oneshot;

use aidaemon_channels::hub::ApprovalRequest;
use aidaemon_events::{DecisionType, EventEmitter};
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{
    ApprovalCall, ApprovalKind, ApprovalResponse, OwnerApprovalNeed, ScopeEscalation,
    ToolCallContext, UserRole,
};
use aidaemon_tools::OutcomeTool as Tool;

use crate::host::DecisionRecorder;
use crate::turn::tool_dispatch::{access_manifest_scope_check, names_sensitive_path, ScopeCheck};
use crate::untrusted::UntrustedVerifier;

pub mod broker;

pub use broker::ApprovalBroker;

/// How long a card waits for the owner.
pub const OWNER_APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

/// How long a goal or Autopilot confirmation card waits for the owner (the
/// same window the channel's goal prompt keeps its buttons).
pub const GOAL_CONFIRMATION_TIMEOUT: Duration = Duration::from_secs(1800);

/// Reason recorded when the owner answers Deny.
const DECLINED_BY_OWNER: &str = "declined by the owner";
/// Reason recorded for a card that was lost when its run was cut off.
const RUN_ABANDONED: &str = "the run was cut off before the owner answered";
/// Reason recorded when the card could not be delivered.
const COULD_NOT_REACH: &str = "could not reach the owner";

/// The daemon's approval rules for one turn, built from the turn's facts.
///
/// The rules hold only the turn's facts and dispatch generically: each
/// call goes to its tool's `ToolSemantics::owner_approval_request` with a
/// `ToolCallContext` built from these facts (never from the call's
/// arguments). The tools own their ladders: `http_request` keeps H7,
/// `manage_api` asks about a live probe that may POST, and the terminal
/// runs its T8 ladder.
///
/// A tool whose `dispatch_facts().runs_as_judged` holds (the terminal) is
/// judged in the canonical form the dispatcher executes, with the scope the
/// dispatcher will check, and that judgment is recorded so the dispatcher
/// runs the call exactly as judged. A tool with no opinion defers to the
/// policy's mode (`Never`), so it runs as before.
#[derive(Debug, Clone, Default)]
pub struct AppApprovalRules {
    /// A trusted scheduled session (never under a mandate): the tools' own
    /// ladders skip their cards for it.
    pub trusted_session: bool,
    /// A mandate run: the dispatcher admits only calls its exact action
    /// grant covers, so the owner already approved them.
    pub mandate_preapproved: bool,
    /// The turn's session.
    pub session_id: String,
    /// The chat whose approvals cover the turn (a sub-agent's parent chat).
    pub grant_session: String,
    /// The role of the turn's user (`None` when unknown).
    pub user_role: Option<UserRole>,
    /// Visibility of the channel the turn came from (`None` when unknown),
    /// so a tool's judgment needs no card for a call its run refuses there.
    pub channel_visibility: Option<aidaemon_core::ChannelVisibility>,
    /// The session was started by an automated trigger (an email, an
    /// event), not by the owner: its terminal calls always ask.
    pub untrusted_trigger_session: bool,
    /// The turn's tool whose calls are judged in canonical form (the
    /// terminal), with the turn's canonicalizer.
    pub terminal: Option<TerminalApprovals>,
    /// The turn's tools, whose own approval requests the rules consult.
    pub tools: ApprovalTools,
    /// The folders the turn guessed from conversation text. A file write
    /// outside them asks the owner; empty when nobody can answer or a real
    /// grant bounds the turn.
    pub guessed_scopes: Vec<String>,
    /// The turn's project directory, which anchors relative targets.
    pub known_project_dir: Option<Arc<std::sync::Mutex<Option<String>>>>,
    /// How paths are classified for this daemon.
    pub paths: aidaemon_tools::path_class::DaemonPaths,
    /// The run's declines as the kernel last showed them to these rules
    /// (`HookContext::declined`, which every judged call refreshes): a
    /// child spawned from this turn starts with them.
    pub run_declines: RunDeclines,
}

/// The turn's tools as the approval rules consult them, by name.
#[derive(Clone, Default)]
pub struct ApprovalTools(Vec<Arc<dyn Tool>>);

impl ApprovalTools {
    pub fn new(tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        Self(tools.into_iter().collect())
    }

    fn find(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.0.iter().find(|tool| tool.name() == name)
    }
}

impl std::fmt::Debug for ApprovalTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.0.iter().map(|tool| tool.name().to_string()))
            .finish()
    }
}

/// The effects the owner declined in a run, as the kernel's hook context
/// reports them (`RunState::declined`). Declines only grow within a
/// daemon run, so the latest view holds every earlier one.
///
/// Once bound to its turn's run, it takes in that run's declines only: the
/// same hooks also judge a run a restart left unfinished, settled before
/// the turn, and what that run declined (with nobody asked) is not the
/// owner's answer in this one.
#[derive(Debug, Clone, Default)]
pub struct RunDeclines(Arc<Mutex<(Option<sdk::RunId>, Vec<sdk::EffectKey>)>>);

impl RunDeclines {
    /// Take in only `run`'s declines from now on.
    pub fn bind(&self, run: sdk::RunId) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).0 = Some(run);
    }

    /// Take in `run`'s declines as a hook context shows them.
    pub fn observe(&self, run: sdk::RunId, declined: &[sdk::EffectKey]) {
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let (bound, seen) = &mut *guard;
        if bound.is_some_and(|bound| bound != run) {
            return;
        }
        for key in declined {
            if !seen.contains(key) {
                seen.push(key.clone());
            }
        }
    }

    /// Every decline seen so far.
    pub fn snapshot(&self) -> Vec<sdk::EffectKey> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).1.clone()
    }
}

/// How the controller's filesystem scope check sees a terminal call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalScope {
    InScope,
    /// Exact out-of-scope targets the owner may authorize (a confined
    /// capability escalation).
    Escalate(ScopeEscalation),
    /// The dispatcher refuses the call (for this reason); nobody is asked.
    Refused(String),
}

impl TerminalScope {
    /// The scope check the dispatcher applies for a call judged this way:
    /// exactly the judged one, never a recomputation (the turn's known
    /// project directory may have moved since the judgment).
    pub fn as_scope_check(&self) -> Option<ScopeCheck> {
        match self {
            Self::InScope => None,
            Self::Escalate(escalation) => Some(ScopeCheck::Escalatable {
                reason: "the out-of-scope targets the owner was asked about".to_string(),
                escalation: escalation.clone(),
            }),
            Self::Refused(reason) => Some(ScopeCheck::Rejected(reason.clone())),
        }
    }
}

pub use crate::engine::tool_calls::CanonicalCall;

/// A terminal call as the kernel judged it.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgedTerminal {
    pub call: CanonicalCall,
    pub scope: TerminalScope,
}

/// Canonicalize and scope-check one call, by run and call id.
type Canonicalize = dyn Fn(sdk::RunId, &str, &serde_json::Value) -> JudgedTerminal + Send + Sync;

/// Test seams on the judgment path, by session: drop a judgment before the
/// dispatcher takes it, or make the kernel judge a call with a given scope.
#[cfg(any(test, feature = "testing"))]
#[derive(Debug, Clone)]
pub enum JudgmentSeam {
    DropJudgment,
    JudgedScope(TerminalScope),
}

#[cfg(any(test, feature = "testing"))]
pub static JUDGMENT_SEAMS: std::sync::LazyLock<Mutex<HashMap<String, JudgmentSeam>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(any(test, feature = "testing"))]
pub fn judgment_seam(session_id: &str) -> Option<JudgmentSeam> {
    JUDGMENT_SEAMS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(session_id)
        .cloned()
}

/// The canonical form of each terminal call the kernel judged, keyed by
/// run and call id (call ids are only unique within a run, and concurrent
/// runs share no entries): the dispatcher executes exactly these arguments.
#[derive(Debug, Clone, Default)]
pub struct JudgedCalls(Arc<Mutex<HashMap<(sdk::RunId, String), JudgedTerminal>>>);

impl JudgedCalls {
    /// The judged form of `call_id` in `run`, if the kernel judged it.
    pub fn get(&self, run: sdk::RunId, call_id: &str) -> Option<JudgedTerminal> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(run, call_id.to_string()))
            .cloned()
    }

    /// Remove and return the judged form of `call_id` in `run`.
    pub fn take(&self, run: sdk::RunId, call_id: &str) -> Option<JudgedTerminal> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(run, call_id.to_string()))
    }

    fn insert(&self, run: sdk::RunId, call_id: &str, judged: JudgedTerminal) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((run, call_id.to_string()), judged);
    }
}

/// The terminal whose approval ladder the kernel's rules run, how a call is
/// canonicalized and scope-checked for this turn, and the calls judged so
/// far (which the dispatcher then runs as judged).
#[derive(Clone)]
pub struct TerminalApprovals {
    tool: Arc<dyn Tool>,
    canonicalize: Option<Arc<Canonicalize>>,
    judged: JudgedCalls,
}

impl std::fmt::Debug for TerminalApprovals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalApprovals")
            .field("canonicalize", &self.canonicalize.is_some())
            .finish_non_exhaustive()
    }
}

impl TerminalApprovals {
    /// The tool among `tools` whose calls the kernel judges in canonical
    /// form (`dispatch_facts().runs_as_judged`), if there is one.
    pub fn find(tools: &[Arc<dyn Tool>]) -> Option<Self> {
        let tool = tools
            .iter()
            .find(|tool| tool.dispatch_facts().runs_as_judged)?
            .clone();
        Some(Self {
            tool,
            canonicalize: None,
            judged: JudgedCalls::default(),
        })
    }

    /// The terminal tool itself.
    pub fn tool(&self) -> Arc<dyn Tool> {
        self.tool.clone()
    }

    /// Canonicalize and scope-check calls as the turn's dispatcher does,
    /// recording each judged call in `judged` for the dispatcher.
    pub fn with_canonicalizer(
        mut self,
        canonicalize: impl Fn(sdk::RunId, &str, &serde_json::Value) -> JudgedTerminal
            + Send
            + Sync
            + 'static,
        judged: JudgedCalls,
    ) -> Self {
        self.canonicalize = Some(Arc::new(canonicalize));
        self.judged = judged;
        self
    }

    /// The call in the form the dispatcher will execute, with its scope.
    pub fn judge(&self, run: sdk::RunId, call_id: &str, raw: &serde_json::Value) -> JudgedTerminal {
        match &self.canonicalize {
            Some(canonicalize) => canonicalize(run, call_id, raw),
            None => JudgedTerminal {
                call: CanonicalCall::stripped(Some(self.tool.as_ref()), raw),
                scope: TerminalScope::InScope,
            },
        }
    }

    /// The judged form of the call `call_id` in `run`, if the kernel judged it.
    pub fn judged(&self, run: sdk::RunId, call_id: &str) -> Option<JudgedTerminal> {
        self.judged.get(run, call_id)
    }

    fn remember(&self, run: sdk::RunId, call_id: &str, call: JudgedTerminal) {
        self.judged.insert(run, call_id, call);
    }
}

/// The rules' decision for what a tool said about a call.
fn rule_decision(need: OwnerApprovalNeed) -> RuleDecision {
    match need {
        OwnerApprovalNeed::Ask(request) => RuleDecision::Ask(request),
        OwnerApprovalNeed::NotNeeded => RuleDecision::Allow,
        OwnerApprovalNeed::Unjudged => RuleDecision::Defer,
    }
}

impl AppApprovalRules {
    /// The dispatcher's facts for a call of this turn, with the scope
    /// escalation the dispatcher judged for it.
    fn call_facts(&self, escalation: Option<ScopeEscalation>) -> ToolCallContext {
        ToolCallContext {
            session_id: self.session_id.clone(),
            grant_session: self.grant_session.clone(),
            user_role: self.user_role,
            channel_visibility: self.channel_visibility,
            trusted_session: self.trusted_session,
            mandate_preapproved: self.mandate_preapproved,
            untrusted_source: self.untrusted_trigger_session,
            scope_escalation: escalation,
            ..Default::default()
        }
    }

    /// The judged-form tool if `name` is it.
    fn judged_tool(&self, name: &str) -> Option<&TerminalApprovals> {
        self.terminal
            .as_ref()
            .filter(|approvals| approvals.tool.name() == name)
    }

    fn tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.judged_tool(name)
            .map(|approvals| &approvals.tool)
            .or_else(|| self.tools.find(name))
    }

    /// The mode a card for `tool` is sent with: the tool's own
    /// (`approval_card_mode`), Cautious otherwise.
    pub fn card_permission_mode(&self, tool: &str) -> PermissionMode {
        self.tool(tool)
            .and_then(|tool| tool.approval_card_mode())
            .unwrap_or(PermissionMode::Cautious)
    }

    /// The kind of card `request` for `tool` is sent as: the tool's own
    /// (`approval_card_kind`), a standard command card otherwise.
    pub fn card_kind(&self, tool: &str, request: &sdk::OwnerApprovalRequest) -> ApprovalKind {
        self.tool(tool)
            .map(|tool| tool.approval_card_kind(request))
            .unwrap_or_default()
    }

    /// Whether `tool` keeps the owner's answers in its own grant stores
    /// (`dispatch_facts().keeps_own_grants`) instead of the session grants.
    pub fn keeps_own_grants(&self, tool: &str) -> bool {
        self.tool(tool)
            .is_some_and(|tool| tool.dispatch_facts().keeps_own_grants)
    }

    /// The card for a file write outside the folders the turn guessed. Asks
    /// nothing for a write the dispatcher refuses anyway, so the owner is
    /// never asked to approve a call that cannot run.
    fn write_outside_guess_request(
        &self,
        tool: &dyn Tool,
        arguments: &serde_json::Value,
    ) -> Option<sdk::OwnerApprovalRequest> {
        if self.guessed_scopes.is_empty() {
            return None;
        }
        let arguments = arguments.to_string();
        let mut manifest = aidaemon_tools::semantics::prepare_invocation(tool, &arguments)
            .map(|prepared| prepared.access_manifest)
            .unwrap_or_else(|_| tool.call_access_manifest(&arguments));
        manifest.read_targets.clear();
        if manifest.write_targets.is_empty() || names_sensitive_path(&manifest, &self.paths) {
            return None;
        }
        let known = self
            .known_project_dir
            .as_ref()
            .and_then(|known| known.lock().ok().and_then(|known| known.clone()));
        let check = |scopes: &[String]| {
            access_manifest_scope_check(
                tool.name(),
                &manifest,
                None,
                scopes,
                known.as_deref(),
                &self.paths,
            )
        };
        if check(&[]).is_some()
            || !matches!(check(&self.guessed_scopes), Some(ScopeCheck::Rejected(_)))
        {
            return None;
        }
        let paths = manifest
            .write_targets
            .iter()
            .map(|target| target.value.as_str())
            .collect::<Vec<_>>();
        let mut folders = paths
            .iter()
            .map(|path| {
                std::path::Path::new(path)
                    .parent()
                    .map_or_else(|| (*path).to_string(), |dir| dir.display().to_string())
            })
            .collect::<Vec<_>>();
        folders.sort();
        folders.dedup();
        Some(sdk::OwnerApprovalRequest {
            effects: Vec::new(),
            summary: format!(
                "Write {}? This task looks like it is about {}, and this is outside it.",
                paths.join(", "),
                self.guessed_scopes.join(", ")
            ),
            subject: format!("write outside guessed project: {}", folders.join(", ")),
            risk: sdk::ApprovalRisk::Medium,
            warnings: Vec::new(),
        })
    }

    /// Judge the call the dispatcher will execute (canonicalized), never
    /// the raw one, and record the judgment for the dispatcher.
    async fn judge_canonical(
        &self,
        approvals: &TerminalApprovals,
        run: sdk::RunId,
        call: &sdk::ToolCall,
    ) -> RuleDecision {
        let judged = approvals.judge(run, &call.id, &call.arguments);
        approvals.remember(run, &call.id, judged.clone());
        let (refused, escalation) = match judged.scope {
            // The dispatcher refuses it before the tool runs.
            TerminalScope::Refused(_) => (true, None),
            TerminalScope::Escalate(escalation) => (false, Some(escalation)),
            TerminalScope::InScope => (false, None),
        };
        let approval_call = ApprovalCall {
            arguments: &judged.call.arguments,
            as_sent: &call.arguments,
            refused,
        };
        rule_decision(
            approvals
                .tool
                .owner_approval_request(&approval_call, &self.call_facts(escalation))
                .await,
        )
    }

    /// Apply the owner's answer to a card to the tool's own grant stores
    /// (`None`: the card timed out or never reached the owner), for the
    /// call as it was judged when the card was asked.
    pub async fn record_owner_answer(
        &self,
        run: sdk::RunId,
        call: &sdk::ToolCall,
        response: Option<ApprovalResponse>,
    ) {
        if let Some(approvals) = self.judged_tool(&call.name) {
            let judged = approvals
                .judged(run, &call.id)
                .unwrap_or_else(|| approvals.judge(run, &call.id, &call.arguments));
            let (refused, escalation) = match judged.scope {
                TerminalScope::Refused(_) => (true, None),
                TerminalScope::Escalate(escalation) => (false, Some(escalation)),
                TerminalScope::InScope => (false, None),
            };
            let approval_call = ApprovalCall {
                arguments: &judged.call.arguments,
                as_sent: &call.arguments,
                refused,
            };
            approvals
                .tool
                .record_owner_answer(&approval_call, &self.call_facts(escalation), response)
                .await;
        } else if let Some(tool) = self.tools.find(&call.name) {
            tool.record_owner_answer(
                &ApprovalCall::as_sent(&call.arguments),
                &self.call_facts(None),
                response,
            )
            .await;
        }
    }
}

#[async_trait]
impl ApprovalRules for AppApprovalRules {
    async fn judge(
        &self,
        ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _info: &sdk::ToolInfo,
    ) -> RuleDecision {
        self.run_declines.observe(ctx.run_id, ctx.declined);
        if let Some(approvals) = self.judged_tool(&call.name) {
            return self.judge_canonical(approvals, ctx.run_id, call).await;
        }
        let Some(tool) = self.tools.find(&call.name) else {
            return RuleDecision::Defer;
        };
        let need = tool
            .owner_approval_request(
                &ApprovalCall::as_sent(&call.arguments),
                &self.call_facts(None),
            )
            .await;
        if !matches!(need, OwnerApprovalNeed::Ask(_)) {
            if let Some(request) = self.write_outside_guess_request(tool.as_ref(), &call.arguments)
            {
                return RuleDecision::Ask(request);
            }
        }
        rule_decision(need)
    }
}

/// The owner's approvals across turns: the card channel, one set of grants
/// per chat session, and the cards still waiting for an answer.
///
/// A chat's grants are shared with every child agent spawned from it (they
/// are keyed by [`grant_session`]): "Allow for session" on a request in a
/// chat also covers the identical request from that chat's children.
pub struct OwnerApprovals {
    broker: std::sync::RwLock<Option<ApprovalBroker>>,
    grants: Mutex<HashMap<String, Arc<SessionGrants>>>,
    /// Cards waiting for the owner. Dropping an entry (a restart loses them
    /// all) ends the wait as unanswered.
    pending: Mutex<HashMap<u64, oneshot::Sender<()>>>,
    next_card: AtomicU64,
    timeout: Duration,
    /// How long a goal or Autopilot confirmation waits: the owner reads a
    /// whole plan there, so it gets the channel's longer goal window.
    goal_timeout: Duration,
}

impl Default for OwnerApprovals {
    fn default() -> Self {
        Self {
            broker: std::sync::RwLock::new(None),
            grants: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
            next_card: AtomicU64::new(0),
            timeout: OWNER_APPROVAL_TIMEOUT,
            goal_timeout: GOAL_CONFIRMATION_TIMEOUT,
        }
    }
}

impl std::fmt::Debug for OwnerApprovals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnerApprovals")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// How the owner settled a card.
enum Answer {
    Granted(ApprovalResponse),
    Declined(String),
}

impl OwnerApprovals {
    /// Approvals whose cards wait `timeout` instead of five minutes.
    #[cfg(any(test, feature = "testing"))]
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout,
            goal_timeout: timeout,
            ..Self::default()
        }
    }

    /// The channel cards go out on (set once channels are up).
    pub fn set_broker(&self, broker: ApprovalBroker) {
        *self.broker.write().unwrap_or_else(|e| e.into_inner()) = Some(broker);
    }

    /// The grants of one chat session.
    pub fn grants_for(&self, session_id: &str) -> Arc<SessionGrants> {
        self.grants
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(session_id.to_string())
            .or_default()
            .clone()
    }

    /// The owner approved `request` for the rest of the session. For
    /// http_request, Always means Session, as it always did.
    pub fn record_grant(&self, session_id: &str, request: &sdk::OwnerApprovalRequest) {
        self.grants_for(session_id).grant(request);
    }

    /// Forget every waiting card, as a restart does.
    #[cfg(any(test, feature = "testing"))]
    pub fn abandon_pending(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Send the owner a card for `request` and wait for the answer.
    async fn ask(
        &self,
        card_session: &str,
        request: &sdk::OwnerApprovalRequest,
        permission_mode: PermissionMode,
        kind: ApprovalKind,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Answer {
        let broker = self
            .broker
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(broker) = broker else {
            return Answer::Declined(COULD_NOT_REACH.to_string());
        };
        // Time on this card is the owner's, not the turn's.
        let _waiting = aidaemon_channels::owner_wait::OwnerWaitClock::begin_waiting();
        let card = self.next_card.fetch_add(1, Ordering::Relaxed);
        let (abandon_tx, abandon_rx) = oneshot::channel::<()>();
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(card, abandon_tx);
        let (response_tx, response_rx) = oneshot::channel();
        let sent = broker
            .send(ApprovalRequest {
                command: request.summary.clone(),
                session_id: card_session.to_string(),
                risk_level: card_risk(request.risk),
                warnings: request.warnings.clone(),
                permission_mode,
                response_tx,
                kind,
            })
            .await;
        let wait = if matches!(
            kind,
            ApprovalKind::GoalConfirmation | ApprovalKind::AutopilotConfirmation
        ) {
            self.goal_timeout
        } else {
            self.timeout
        };
        let answer = if sent.is_err() {
            Answer::Declined(COULD_NOT_REACH.to_string())
        } else {
            tokio::select! {
                response = response_rx => match response {
                    Ok(response) => Answer::Granted(response),
                    // The channel dropped the card without an answer.
                    Err(_) => Answer::Declined(COULD_NOT_REACH.to_string()),
                },
                _ = tokio::time::sleep(wait) => {
                    Answer::Declined(no_answer_reason(wait))
                }
                // The waiting entry is gone (a restart): the card is stale.
                _ = abandon_rx => Answer::Declined(no_answer_reason(wait)),
                _ = cancel.cancelled() => {
                    Answer::Declined("the turn was cancelled before the owner answered".to_string())
                }
            }
        };
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&card);
        match answer {
            Answer::Granted(ApprovalResponse::Deny) => {
                Answer::Declined(DECLINED_BY_OWNER.to_string())
            }
            other => other,
        }
    }
}

/// The session whose grants a turn uses and adds to: the human-facing chat
/// a child agent was spawned from, else the turn's own session.
pub fn grant_session(approval_session_id: Option<&str>, session_id: &str) -> String {
    approval_session_id.unwrap_or(session_id).to_string()
}

/// The request as its card shows it: a sub-agent's card names the mission
/// it was delegated (the last step of its inherited lineage), so the owner
/// can tell concurrent sub-agents' cards apart. The request itself (subject,
/// effects, grants) is unchanged.
fn asked_by<'r>(
    mission: Option<&str>,
    request: &'r sdk::OwnerApprovalRequest,
) -> std::borrow::Cow<'r, sdk::OwnerApprovalRequest> {
    let mission = mission
        .map(|mission| mission.trim())
        .filter(|mission| !mission.is_empty());
    match mission {
        Some(mission) => {
            let mut card = request.clone();
            card.summary = sub_agent_card_summary(mission, &request.summary);
            std::borrow::Cow::Owned(card)
        }
        None => std::borrow::Cow::Borrowed(request),
    }
}

/// How many characters of a sub-agent's name a card shows.
const CARD_MISSION_CHARS: usize = 40;

/// A sub-agent's card opens with its short name in plain words ("Drupal
/// install asks:"): the title its parent gave it, else its mission's first
/// line cut at a whole word.
fn sub_agent_card_summary(mission: &str, summary: &str) -> String {
    let line = mission.lines().next().unwrap_or_default().trim();
    let line = line.trim_end_matches('.');
    let shown = if line.chars().count() <= CARD_MISSION_CHARS {
        line.to_string()
    } else {
        let head: String = line.chars().take(CARD_MISSION_CHARS - 1).collect();
        let kept = head
            .rfind(char::is_whitespace)
            .map_or(head.as_str(), |at| &head[..at]);
        format!("{}…", kept.trim_end())
    };
    format!("{shown} asks:\n{summary}")
}

/// The card's risk level for a kernel request.
/// The card's risk level for a kernel approval request: the inverse of
/// [`approval_risk`], so a level survives the round trip.
pub fn card_risk(risk: sdk::ApprovalRisk) -> RiskLevel {
    match risk {
        sdk::ApprovalRisk::Low => RiskLevel::Safe,
        sdk::ApprovalRisk::Medium => RiskLevel::Medium,
        sdk::ApprovalRisk::Critical => RiskLevel::Critical,
        _ => RiskLevel::High,
    }
}

/// A tool's risk level as a kernel approval request's risk.
pub fn approval_risk(risk: RiskLevel) -> sdk::ApprovalRisk {
    match risk {
        RiskLevel::Safe => sdk::ApprovalRisk::Low,
        RiskLevel::Medium => sdk::ApprovalRisk::Medium,
        RiskLevel::High => sdk::ApprovalRisk::High,
        RiskLevel::Critical => sdk::ApprovalRisk::Critical,
    }
}

fn risk_name(risk: sdk::ApprovalRisk) -> &'static str {
    match risk {
        sdk::ApprovalRisk::Low => "low",
        sdk::ApprovalRisk::Medium => "medium",
        sdk::ApprovalRisk::Critical => "critical",
        _ => "high",
    }
}

/// The decline reason for a card nobody answered within `timeout`.
pub fn no_answer_reason(timeout: Duration) -> String {
    let secs = timeout.as_secs_f64();
    let within = if timeout.as_secs() >= 60 && timeout.as_secs().is_multiple_of(60) {
        let minutes = timeout.as_secs() / 60;
        format!("{minutes} minute{}", if minutes == 1 { "" } else { "s" })
    } else {
        format!("{secs} seconds")
    };
    format!("no answer from the owner within {within}")
}

/// The daemon's configured API keys and tokens, refused in outbound calls
/// by the kernel's `EgressGuard`. Never printed.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct ConfiguredSecrets(Vec<String>);

impl ConfiguredSecrets {
    pub fn new(values: Vec<String>) -> Self {
        let mut values: Vec<String> = values
            .into_iter()
            .filter(|value| !value.trim().is_empty())
            .collect();
        values.sort();
        values.dedup();
        Self(values)
    }

    pub fn values(&self) -> &[String] {
        &self.0
    }
}

impl std::fmt::Debug for ConfiguredSecrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConfiguredSecrets({} values)", self.0.len())
    }
}

/// Asks the owner about the kernel's owner-approval interruptions.
pub struct OwnerApprovalResolver<'a> {
    /// Where the resolver's decision points go.
    pub decisions: &'a dyn DecisionRecorder,
    /// The mission a sub-agent was delegated (the last step of its inherited
    /// lineage), named on its cards; `None` for a top-level agent.
    pub mission: Option<&'a str>,
    pub approvals: Arc<OwnerApprovals>,
    /// The turn's rules: a terminal answer goes to the terminal's grants.
    pub rules: Arc<AppApprovalRules>,
    /// Where the card goes (the turn's session; a child's card is routed
    /// to its parent chat by the hub).
    pub card_session: &'a str,
    /// Whose grants an Allow Session adds to: the human-facing chat.
    pub grant_session: String,
    pub emitter: &'a EventEmitter,
    pub task_id: &'a str,
    pub cancel: tokio_util::sync::CancellationToken,
}

impl OwnerApprovalResolver<'_> {
    async fn record(
        &self,
        condition: &str,
        tool: &str,
        request: &sdk::OwnerApprovalRequest,
        reason: Option<&str>,
    ) {
        let effects: Vec<String> = request
            .effects
            .iter()
            .map(|key| format!("{} {}", key.class, key.target))
            .collect();
        self.decisions
            .record_decision_point(
                self.emitter,
                self.task_id,
                0,
                DecisionType::GateTelemetry,
                format!("owner approval: {condition}"),
                json!({
                    "condition": condition,
                    "tool": tool,
                    "effects": effects,
                    "risk": risk_name(request.risk),
                    "reason": reason,
                }),
            )
            .await;
    }

    /// Decline `request` without a card: its run outlived the conversation
    /// it belonged to.
    async fn decline_unasked(
        &self,
        run: sdk::RunId,
        interruption: &sdk::Interruption,
        request: &sdk::OwnerApprovalRequest,
    ) -> sdk::Resolution {
        let tool = interruption.call.name.as_str();
        if self.rules.keeps_own_grants(tool) {
            self.rules
                .record_owner_answer(run, &interruption.call, None)
                .await;
        }
        self.record(
            "owner_approval_declined",
            tool,
            request,
            Some(RUN_ABANDONED),
        )
        .await;
        sdk::Resolution::Deny {
            reason: RUN_ABANDONED.to_string(),
        }
    }

    async fn resolve(
        &self,
        run: sdk::RunId,
        interruption: &sdk::Interruption,
        request: &sdk::OwnerApprovalRequest,
    ) -> sdk::Resolution {
        let kind = self.rules.card_kind(&interruption.call.name, request);
        self.ask(run, interruption, request, kind).await
    }

    /// The call declares no effect beyond the agent's own state (its memory,
    /// people store, checklist): a refusal of it goes back to the model, not
    /// to the owner as a card. Such a call is never worth holding the turn
    /// for an answer, and refusing it is the stricter outcome.
    fn only_changes_agent_state(&self, call: &sdk::ToolCall) -> bool {
        self.rules.tool(&call.name).is_some_and(|tool| {
            effects_only_agent_state(&tool.call_effects(&call.arguments.to_string()))
        })
    }

    /// Put a call the clean check would not pass to the owner, for this one
    /// call only. The owner's Deny binds the run like any other decline. A
    /// card nobody answered leaves the check's own refusal standing: no owner
    /// declined anything.
    async fn ask_about_refused_call(
        &self,
        state: &sdk::RunState,
        interruption: &sdk::Interruption,
        verdict: &crate::untrusted::VerifierVerdict,
    ) -> sdk::Resolution {
        let call = &interruption.call;
        let effects = self
            .rules
            .tool(&call.name)
            .map(|tool| tool.call_effects(&call.arguments.to_string()).keys)
            .unwrap_or_default();
        let request = refused_call_request(&call.name, &call.arguments, effects, &verdict.reason);
        // The kernel only stops a retry by its effects; one declared by
        // subject alone must not bring the same card back.
        let declined = sdk::EffectKey::named(sdk::EffectClass::Call, &request.subject);
        if state.declined.contains(&declined) {
            return sdk::Resolution::OwnerDeclined {
                request,
                reason: DECLINED_BY_OWNER.to_string(),
            };
        }
        let kind = ApprovalKind::CommandOnce;
        match self.ask(state.run_id, interruption, &request, kind).await {
            sdk::Resolution::Deny { reason } if reason == DECLINED_BY_OWNER => {
                sdk::Resolution::OwnerDeclined { request, reason }
            }
            sdk::Resolution::Deny { .. } => {
                crate::untrusted::verdict_resolution(&call.name, verdict)
            }
            other => other,
        }
    }

    async fn ask(
        &self,
        run: sdk::RunId,
        interruption: &sdk::Interruption,
        request: &sdk::OwnerApprovalRequest,
        kind: ApprovalKind,
    ) -> sdk::Resolution {
        let tool = interruption.call.name.as_str();
        self.record("owner_approval_requested", tool, request, None)
            .await;
        // Several sub-agents can ask at once: the card says which one asks.
        let card = asked_by(self.mission, request);
        match self
            .approvals
            .ask(
                self.card_session,
                card.as_ref(),
                self.rules.card_permission_mode(tool),
                kind,
                &self.cancel,
            )
            .await
        {
            Answer::Granted(response) => {
                if self.rules.keeps_own_grants(tool) {
                    // The tool's grants live in its own stores (for the
                    // terminal: session and persisted prefixes, capability
                    // grants), consulted at its ladder's own rungs.
                    self.rules
                        .record_owner_answer(run, &interruption.call, Some(response))
                        .await;
                } else if matches!(
                    response,
                    ApprovalResponse::AllowSession | ApprovalResponse::AllowAlways
                ) {
                    self.approvals.record_grant(&self.grant_session, request);
                }
                self.record("owner_approval_granted", tool, request, None)
                    .await;
                sdk::Resolution::Approve
            }
            Answer::Declined(reason) => {
                if self.rules.keeps_own_grants(tool) {
                    let answer = (reason == DECLINED_BY_OWNER).then_some(ApprovalResponse::Deny);
                    self.rules
                        .record_owner_answer(run, &interruption.call, answer)
                        .await;
                }
                self.record("owner_approval_declined", tool, request, Some(&reason))
                    .await;
                sdk::Resolution::Deny { reason }
            }
        }
    }
}

/// Settles a run the daemon left unfinished, ahead of the owner's new message.
/// Nobody is asked: a card for the old request would answer the new message
/// with an unrelated question. Its owner approvals are declined and listed,
/// so the turn can say what was not done.
pub struct AbandonedRunResolver<'a, 'r> {
    pub turn: &'a DaemonInterruptResolver<'r>,
    pub not_done: Mutex<Vec<String>>,
}

#[async_trait]
impl sdk::InterruptResolver for AbandonedRunResolver<'_, '_> {
    async fn resolve(
        &self,
        state: &sdk::RunState,
        interruption: &sdk::Interruption,
    ) -> Option<sdk::Resolution> {
        let sdk::InterruptKind::OwnerApproval(request) = &interruption.kind else {
            return match &interruption.kind {
                // The clean check alone: a card here would be the same
                // unrelated question.
                sdk::InterruptKind::Untrusted => {
                    sdk::InterruptResolver::resolve(&self.turn.untrusted, state, interruption).await
                }
                _ => Some(unknown_interrupt_kind(interruption)),
            };
        };
        self.not_done
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(request.summary.clone());
        Some(
            self.turn
                .owner
                .decline_unasked(state.run_id, interruption, request)
                .await,
        )
    }
}

/// The turn's resolver: untrusted-content checks go to the clean verifier,
/// owner approvals to the owner.
pub struct DaemonInterruptResolver<'a> {
    pub untrusted: UntrustedVerifier<'a>,
    pub owner: OwnerApprovalResolver<'a>,
}

#[async_trait]
impl sdk::InterruptResolver for DaemonInterruptResolver<'_> {
    async fn resolve(
        &self,
        state: &sdk::RunState,
        interruption: &sdk::Interruption,
    ) -> Option<sdk::Resolution> {
        match &interruption.kind {
            sdk::InterruptKind::OwnerApproval(request) => Some(
                self.owner
                    .resolve(state.run_id, interruption, request)
                    .await,
            ),
            sdk::InterruptKind::Untrusted => {
                let verdict = self.untrusted.judge(interruption).await?;
                let owner_can_answer =
                    refused_check_may_ask_owner(&self.owner.rules, self.untrusted.origin_trusted)
                        && !self.untrusted.cancel.is_cancelled()
                        && !self.owner.only_changes_agent_state(&interruption.call);
                Some(if verdict.consistent || !owner_can_answer {
                    crate::untrusted::verdict_resolution(&interruption.call.name, &verdict)
                } else {
                    self.owner
                        .ask_about_refused_call(state, interruption, &verdict)
                        .await
                })
            }
            _ => Some(unknown_interrupt_kind(interruption)),
        }
    }
}

/// Declared as touching nothing outside the agent (`CallEffects::none()`);
/// undeclared effects (`unknown()`) may do anything and do not qualify.
fn effects_only_agent_state(effects: &aidaemon_core::CallEffects) -> bool {
    effects.complete && effects.keys.is_empty()
}

/// Whether a call the clean check would not pass can go to the owner: only
/// when the owner wrote the request and is in a private chat to answer.
/// Scheduled, mandate and triggered runs, and other people's turns, keep the
/// check's refusal.
fn refused_check_may_ask_owner(rules: &AppApprovalRules, origin_trusted: bool) -> bool {
    origin_trusted
        && rules.user_role == Some(UserRole::Owner)
        && rules.channel_visibility == Some(aidaemon_core::ChannelVisibility::Private)
        && !rules.trusted_session
        && !rules.mandate_preapproved
        && !rules.untrusted_trigger_session
}

/// How much of a call a card shows, head and tail around a marked cut.
const REFUSED_CALL_SHOWN_CHARS: usize = 1500;
const REFUSED_CALL_REASON_CHARS: usize = 400;

/// The card for a call the clean check would not pass. It says in plain
/// words what allowing it does and that the step waits meanwhile; the exact
/// command the owner would be allowing (not the model's account of it) is
/// under its details, never the headline.
fn refused_call_request(
    tool: &str,
    arguments: &Value,
    effects: Vec<sdk::EffectKey>,
    check_reason: &str,
) -> sdk::OwnerApprovalRequest {
    let command = ["command", "script"]
        .iter()
        .find_map(|key| arguments.get(key).and_then(Value::as_str));
    let shown = command.map_or_else(|| arguments.to_string(), str::to_string);
    // The owner decides on the call as it would run: secrets stay hidden,
    // the paths it touches stay readable.
    let shown = aidaemon_tools::sanitize::redact_secrets_only(&shown);
    let shown = crate::untrusted::cap_strings(&Value::from(shown), REFUSED_CALL_SHOWN_CHARS);
    let action = if command.is_some() {
        "Run a command on your computer?".to_string()
    } else {
        format!("Let me use {}?", tool.replace('_', " "))
    };
    sdk::OwnerApprovalRequest {
        effects,
        summary: format!(
            "{action} This work read content from outside, and the safety check couldn't \
             confirm this step fits your request. This step is paused until you answer."
        ),
        subject: format!("{tool} {arguments}"),
        risk: sdk::ApprovalRisk::High,
        warnings: vec![
            format!("Details: {}", shown.as_str().unwrap_or_default()),
            format!(
                "Why the check stopped it: {}",
                aidaemon_events::text::truncate_str(check_reason, REFUSED_CALL_REASON_CHARS)
            ),
        ],
    }
}

/// The refusal for an interruption of a kind this daemon cannot resolve:
/// fail closed rather than hand it to a resolver built for another kind.
fn unknown_interrupt_kind(interruption: &sdk::Interruption) -> sdk::Resolution {
    sdk::Resolution::Deny {
        reason: format!(
            "`{}` was not run: it is waiting on a check this daemon cannot resolve ({}).",
            interruption.call.name, interruption.reason
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core as sdk;
    use aidaemon_policies::{ApprovalRules, RuleDecision};
    use serde_json::json;

    /// Only the owner's own request in a private chat may reach a card.
    #[test]
    fn a_refused_check_reaches_the_owner_only_in_their_private_chat() {
        use aidaemon_core::ChannelVisibility;
        use aidaemon_tools::semantics::UserRole;
        let owner_dm = || AppApprovalRules {
            user_role: Some(UserRole::Owner),
            channel_visibility: Some(ChannelVisibility::Private),
            ..Default::default()
        };
        assert!(refused_check_may_ask_owner(&owner_dm(), true));
        // A refused memory write goes back to the model, never to a card;
        // a call with undeclared effects still may.
        assert!(effects_only_agent_state(&aidaemon_core::CallEffects::none()));
        assert!(!effects_only_agent_state(
            &aidaemon_core::CallEffects::unknown()
        ));
        assert!(!refused_check_may_ask_owner(&owner_dm(), false));
        let barred = [
            AppApprovalRules {
                user_role: Some(UserRole::Guest),
                ..owner_dm()
            },
            AppApprovalRules {
                user_role: None,
                ..owner_dm()
            },
            AppApprovalRules {
                channel_visibility: Some(ChannelVisibility::PrivateGroup),
                ..owner_dm()
            },
            AppApprovalRules {
                channel_visibility: Some(ChannelVisibility::Internal),
                ..owner_dm()
            },
            AppApprovalRules {
                channel_visibility: None,
                ..owner_dm()
            },
            AppApprovalRules {
                trusted_session: true,
                ..owner_dm()
            },
            AppApprovalRules {
                mandate_preapproved: true,
                ..owner_dm()
            },
            AppApprovalRules {
                untrusted_trigger_session: true,
                ..owner_dm()
            },
        ];
        for rules in &barred {
            assert!(!refused_check_may_ask_owner(rules, true), "{rules:?}");
        }
    }

    /// The card says in plain words what allowing does and what waits; the
    /// command is under its details, never the headline.
    #[test]
    fn a_refused_call_card_says_what_happens_and_shows_the_command_under_details() {
        let arguments =
            json!({"action": "run", "command": "rm -rf synthetic-dir", "network": true});
        let request = refused_call_request("terminal", &arguments, Vec::new(), "not requested");
        assert!(
            request.summary.starts_with("Run a command on your computer?"),
            "{}",
            request.summary
        );
        assert!(request.summary.contains("paused until you answer"));
        assert!(!request.summary.contains("rm -rf"));
        assert!(!request.summary.contains("terminal"));
        assert_eq!(request.subject, format!("terminal {arguments}"));
        assert_eq!(request.warnings[0], "Details: rm -rf synthetic-dir");
        assert!(request.warnings[1].contains("not requested"));
        let other = refused_call_request("http_request", &json!({"url": "x"}), Vec::new(), "r");
        assert!(other.summary.starts_with("Let me use http request?"));
    }

    /// The end of a long command is kept: that is where a payload hides.
    #[test]
    fn a_refused_call_card_keeps_both_ends_of_a_long_command() {
        let command = format!("start {} rm-tail", "x".repeat(5000));
        let request = refused_call_request(
            "terminal",
            &json!({"command": command}),
            Vec::new(),
            "not requested",
        );
        assert!(request.warnings[0].contains("start"));
        assert!(request.warnings[0].contains("rm-tail"));
        assert!(request.warnings[0].contains("chars cut"));
    }

    fn hook_ctx() -> sdk::HookContext<'static> {
        sdk::HookContext {
            run_id: sdk::RunId::new(),
            agent_id: "aidaemon",
            session_id: "telegram:synthetic-user-1",
            turn: 1,
            messages: &[],
            receipts: &[],
            usage: sdk::Usage::default(),
            run_usage: sdk::Usage::default(),
            untrusted_seen: false,
            declined: &[],
        }
    }

    /// An interruption of a kind the daemon does not know is refused, never
    /// handed to a resolver built for another kind. (`InterruptKind` is
    /// `non_exhaustive` with no third variant yet, so the refusal itself is
    /// pinned here.)
    #[test]
    fn unknown_interrupt_kind_fails_closed() {
        let interruption = sdk::Interruption {
            call: sdk::ToolCall::new("c1", "future_tool", json!({})),
            reason: "a future check".to_string(),
            kind: sdk::InterruptKind::Untrusted,
        };
        match unknown_interrupt_kind(&interruption) {
            sdk::Resolution::Deny { reason } => {
                assert!(reason.contains("future_tool"), "{reason}");
                assert!(reason.contains("not run"), "{reason}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    fn tool_info() -> sdk::ToolInfo {
        sdk::ToolInfo {
            annotations: sdk::ToolAnnotations::default(),
            access: sdk::AccessManifest::default(),
            effects: sdk::CallEffects::unknown(),
            outbound: Default::default(),
        }
    }

    /// A file tool that reads and writes the paths its arguments name.
    struct FileTool;

    #[async_trait]
    impl aidaemon_tools::semantics::ToolSemantics for FileTool {
        fn call_access_manifest(
            &self,
            arguments: &str,
        ) -> aidaemon_tools::semantics::ToolCallAccessManifest {
            use aidaemon_tools::semantics::{ToolTargetHint, ToolTargetHintKind};
            let args: serde_json::Value = serde_json::from_str(arguments).unwrap();
            let targets = |key: &str| {
                args[key]
                    .as_str()
                    .and_then(|path| ToolTargetHint::new(ToolTargetHintKind::Path, path))
                    .into_iter()
                    .collect::<Vec<_>>()
            };
            aidaemon_tools::semantics::ToolCallAccessManifest {
                read_targets: targets("read"),
                write_targets: targets("write"),
                ..Default::default()
            }
        }
    }

    #[async_trait]
    impl aidaemon_tools::OutcomeTool for FileTool {
        fn name(&self) -> &str {
            "file_tool"
        }
        fn description(&self) -> &str {
            "file tool"
        }
        fn schema(&self) -> serde_json::Value {
            json!({"name": "file_tool", "description": "", "parameters": {"type": "object"}})
        }
        async fn execute(
            &self,
            _arguments: &str,
            _facts: &aidaemon_tools::semantics::ToolCallContext,
        ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
            let output: anyhow::Result<String> =
                async move { anyhow::bail!("not run in these tests") }.await;
            output.map(aidaemon_tools::semantics::ToolCallOutcome::from_output)
        }
    }

    fn guessing_rules(scopes: &[&str]) -> AppApprovalRules {
        AppApprovalRules {
            tools: ApprovalTools::new([Arc::new(FileTool) as Arc<dyn aidaemon_tools::OutcomeTool>]),
            guessed_scopes: scopes.iter().map(|scope| scope.to_string()).collect(),
            ..AppApprovalRules::default()
        }
    }

    #[tokio::test]
    async fn a_write_outside_the_guess_asks_and_names_the_path() {
        let rules = guessing_rules(&["/tmp/synthetic-guess"]);
        let call = sdk::ToolCall::new(
            "c1",
            "file_tool",
            json!({"write": "/tmp/synthetic-real/app/settings.php"}),
        );
        match rules.judge(&hook_ctx(), &call, &tool_info()).await {
            RuleDecision::Ask(request) => {
                assert!(request
                    .summary
                    .contains("/tmp/synthetic-real/app/settings.php"));
                assert!(request.summary.contains("/tmp/synthetic-guess"));
                assert_eq!(
                    request.subject,
                    "write outside guessed project: /tmp/synthetic-real/app"
                );
            }
            other => panic!("expected a card, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn only_writes_outside_the_guess_ask() {
        let rules = guessing_rules(&["/tmp/synthetic-guess"]);
        for args in [
            json!({"write": "/tmp/synthetic-guess/inside.txt"}),
            json!({"read": "/tmp/synthetic-real/settings.php"}),
        ] {
            let call = sdk::ToolCall::new("c1", "file_tool", args.clone());
            assert!(
                !matches!(
                    rules.judge(&hook_ctx(), &call, &tool_info()).await,
                    RuleDecision::Ask(_)
                ),
                "{args}"
            );
        }
    }

    #[tokio::test]
    async fn no_guess_or_a_sensitive_path_sends_no_card() {
        let write =
            sdk::ToolCall::new("c1", "file_tool", json!({"write": "/tmp/synthetic-real/a"}));
        assert!(!matches!(
            guessing_rules(&[])
                .judge(&hook_ctx(), &write, &tool_info())
                .await,
            RuleDecision::Ask(_)
        ));
        let home = dirs::home_dir().unwrap();
        let sensitive = home.join(".ssh/synthetic_key").display().to_string();
        let write = sdk::ToolCall::new("c1", "file_tool", json!({"write": sensitive}));
        assert!(!matches!(
            guessing_rules(&["/tmp/synthetic-guess"])
                .judge(&hook_ctx(), &write, &tool_info())
                .await,
            RuleDecision::Ask(_)
        ));
    }

    /// A sub-agent's card opens with a short plain name, never "Sub-agent"
    /// or its whole mission in quotes.
    #[test]
    fn a_sub_agent_card_opens_with_its_short_name() {
        assert_eq!(
            super::sub_agent_card_summary("Drupal install", "Run a command?"),
            "Drupal install asks:\nRun a command?"
        );
        let card = super::sub_agent_card_summary(
            "Install a fresh WordPress site with DDEV for a Claude company demo.",
            "s",
        );
        assert_eq!(card, "Install a fresh WordPress site with… asks:\ns");
        assert!(!card.contains("Sub-agent") && !card.contains('"'));
        let long = "x".repeat(200);
        let card = super::sub_agent_card_summary(&long, "s");
        assert!(card.starts_with(&format!("{}…", "x".repeat(39))));
    }

    #[test]
    fn risks_map_to_card_levels() {
        use aidaemon_tools::command_risk::RiskLevel;
        assert_eq!(card_risk(sdk::ApprovalRisk::Medium), RiskLevel::Medium);
        assert_eq!(card_risk(sdk::ApprovalRisk::High), RiskLevel::High);
        assert_eq!(card_risk(sdk::ApprovalRisk::Critical), RiskLevel::Critical);
        // Every tool risk level survives the trip through the kernel.
        for level in [
            RiskLevel::Safe,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Critical,
        ] {
            assert_eq!(card_risk(approval_risk(level)), level);
        }
    }

    #[test]
    fn timeouts_read_as_minutes() {
        assert_eq!(
            no_answer_reason(std::time::Duration::from_secs(300)),
            "no answer from the owner within 5 minutes"
        );
        assert_eq!(
            no_answer_reason(std::time::Duration::from_millis(100)),
            "no answer from the owner within 0.1 seconds"
        );
    }

    #[tokio::test]
    async fn grants_are_per_session_and_always_is_session() {
        use aidaemon_policies::ApprovalGrants;
        let approvals = OwnerApprovals::default();
        let request = sdk::OwnerApprovalRequest {
            effects: vec![sdk::EffectKey::egress("https", "h.example.invalid", None)],
            summary: "POST https://h.example.invalid/x".to_string(),
            subject: "POST https://h.example.invalid/x".to_string(),
            risk: sdk::ApprovalRisk::High,
            warnings: Vec::new(),
        };
        approvals.record_grant("telegram:synthetic-user-1", &request);
        assert!(
            approvals
                .grants_for("telegram:synthetic-user-1")
                .covers(&request)
                .await
        );
        assert!(
            !approvals
                .grants_for("telegram:synthetic-user-2")
                .covers(&request)
                .await
        );
    }

    #[tokio::test]
    async fn a_chat_and_its_children_share_grants() {
        use aidaemon_policies::ApprovalGrants;
        let approvals = OwnerApprovals::default();
        let request = sdk::OwnerApprovalRequest {
            effects: vec![sdk::EffectKey::egress("https", "h.example.invalid", None)],
            summary: "POST https://h.example.invalid/x".to_string(),
            subject: "POST https://h.example.invalid/x".to_string(),
            risk: sdk::ApprovalRisk::High,
            warnings: Vec::new(),
        };
        let chat = "telegram:synthetic-user-1";
        // The owner allows it for the session in the chat.
        approvals.record_grant(&grant_session(None, chat), &request);
        // A child spawned from the chat uses the chat's grants.
        let child = grant_session(Some(chat), "sub-agent:synthetic-child-1");
        assert_eq!(child, chat);
        assert!(approvals.grants_for(&child).covers(&request).await);
        // Another chat's child does not.
        let other = grant_session(
            Some("telegram:synthetic-user-2"),
            "sub-agent:synthetic-child-2",
        );
        assert!(!approvals.grants_for(&other).covers(&request).await);
    }

    /// Every call the rules judge refreshes what they know of the run's
    /// declines, so a child spawned later in the run starts with them.
    #[tokio::test]
    async fn rules_observe_the_runs_declines() {
        let rules = AppApprovalRules::default();
        let declined = [sdk::EffectKey::egress("https", "h.example.invalid", None)];
        let ctx = sdk::HookContext {
            declined: &declined,
            ..hook_ctx()
        };
        let spawn = sdk::ToolCall::new(
            "c1",
            "spawn_agent",
            json!({"mission": "m", "task": "Deliver the synthetic report."}),
        );
        assert_eq!(
            rules.judge(&ctx, &spawn, &tool_info()).await,
            RuleDecision::Defer
        );
        assert_eq!(rules.run_declines.snapshot(), declined.to_vec());
        // A later view keeps every earlier decline.
        rules.run_declines.observe(ctx.run_id, &[]);
        assert_eq!(rules.run_declines.snapshot(), declined.to_vec());
        // Bound to its turn's run, it ignores another run's declines (a
        // run settled after a restart, with nobody asked).
        let turn = RunDeclines::default();
        turn.bind(sdk::RunId::new());
        turn.observe(ctx.run_id, &declined);
        assert!(turn.snapshot().is_empty());
    }

    #[test]
    fn configured_secrets_never_print() {
        let secrets = ConfiguredSecrets::new(vec!["synthetic-token-123456".to_string()]);
        assert!(!format!("{secrets:?}").contains("synthetic-token"));
    }
}
