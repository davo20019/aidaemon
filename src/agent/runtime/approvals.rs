//! Owner approvals on the SDK kernel: the daemon's approval rules, the
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
use serde_json::json;
use tokio::sync::oneshot;

use crate::agent::untrusted::UntrustedVerifier;
use crate::agent::Agent;
use crate::events::EventEmitter;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::ApprovalBroker;
use crate::types::ApprovalResponse;

/// How long a card waits for the owner.
pub(crate) const OWNER_APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

/// Reason recorded when the owner answers Deny.
const DECLINED_BY_OWNER: &str = "declined by the owner";
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
pub(crate) struct AppApprovalRules {
    /// A trusted scheduled session (never under a mandate): the tools' own
    /// ladders skip their cards for it.
    pub trusted_session: bool,
    /// A mandate run: the dispatcher admits only calls its exact action
    /// grant covers, so the owner already approved them.
    pub mandate_preapproved: bool,
    /// The turn's session.
    pub session_id: String,
    /// The role of the turn's user (`None` when unknown).
    pub user_role: Option<crate::types::UserRole>,
    /// The session was started by an automated trigger (an email, an
    /// event), not by the owner: its terminal calls always ask.
    pub untrusted_trigger_session: bool,
    /// The turn's tool whose calls are judged in canonical form (the
    /// terminal), with the turn's canonicalizer.
    pub terminal: Option<TerminalApprovals>,
    /// The turn's tools, whose own approval requests the rules consult.
    pub tools: ApprovalTools,
    /// The run's declines as the kernel last showed them to these rules
    /// (`HookContext::declined`, which every judged call refreshes): a
    /// child spawned from this turn starts with them.
    pub run_declines: RunDeclines,
}

/// The turn's tools as the approval rules consult them, by name.
#[derive(Clone, Default)]
pub(crate) struct ApprovalTools(Vec<Arc<dyn crate::traits::Tool>>);

impl ApprovalTools {
    pub(crate) fn new(tools: impl IntoIterator<Item = Arc<dyn crate::traits::Tool>>) -> Self {
        Self(tools.into_iter().collect())
    }

    fn find(&self, name: &str) -> Option<&Arc<dyn crate::traits::Tool>> {
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
#[derive(Debug, Clone, Default)]
pub(crate) struct RunDeclines(Arc<Mutex<Vec<sdk::EffectKey>>>);

impl RunDeclines {
    /// Take in the run's declines as a hook context shows them.
    pub(crate) fn observe(&self, declined: &[sdk::EffectKey]) {
        let mut seen = self.0.lock().unwrap_or_else(|e| e.into_inner());
        for key in declined {
            if !seen.contains(key) {
                seen.push(key.clone());
            }
        }
    }

    /// Every decline seen so far.
    pub(crate) fn snapshot(&self) -> Vec<sdk::EffectKey> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// How the controller's filesystem scope check sees a terminal call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TerminalScope {
    InScope,
    /// Exact out-of-scope targets the owner may authorize (a confined
    /// capability escalation).
    Escalate(crate::traits::ScopeEscalation),
    /// The dispatcher refuses the call (for this reason); nobody is asked.
    Refused(String),
}

impl TerminalScope {
    /// The scope check the dispatcher applies for a call judged this way:
    /// exactly the judged one, never a recomputation (the turn's known
    /// project directory may have moved since the judgment).
    pub(in crate::agent) fn as_scope_check(
        &self,
    ) -> Option<crate::agent::tool_execution::ScopeCheck> {
        match self {
            Self::InScope => None,
            Self::Escalate(escalation) => {
                Some(crate::agent::tool_execution::ScopeCheck::Escalatable {
                    reason: "the out-of-scope targets the owner was asked about".to_string(),
                    escalation: escalation.clone(),
                })
            }
            Self::Refused(reason) => Some(crate::agent::tool_execution::ScopeCheck::Rejected(
                reason.clone(),
            )),
        }
    }
}

/// A call as the dispatcher executes it: the model's `_` keys and unset
/// optional arguments dropped, the turn's project directory injected and
/// unauthorized read roots dropped (`sdk_engine::tool_calls::canonicalize_call`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CanonicalCall {
    pub arguments: serde_json::Value,
    /// The model's own explicit directory; it refines the turn's known
    /// project directory when the call is dispatched.
    pub explicit_dir: Option<String>,
    /// The project directory injected into the call, if any.
    pub injected_dir: Option<String>,
    /// Declared read roots dropped as unauthorized.
    pub stripped_read_roots: Vec<String>,
}

impl CanonicalCall {
    /// Only the model's `_` keys and unset optional arguments dropped (no
    /// turn to anchor against).
    pub(crate) fn stripped(
        tool: Option<&dyn crate::traits::Tool>,
        raw: &serde_json::Value,
    ) -> Self {
        let mut arguments = raw.clone();
        if let serde_json::Value::Object(map) = &mut arguments {
            crate::agent::tool_exec::strip_model_arguments(map, tool);
        }
        Self {
            arguments,
            explicit_dir: None,
            injected_dir: None,
            stripped_read_roots: Vec::new(),
        }
    }
}

/// A terminal call as the kernel judged it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JudgedTerminal {
    pub call: CanonicalCall,
    pub scope: TerminalScope,
}

/// Canonicalize and scope-check one call, by run and call id.
type Canonicalize = dyn Fn(sdk::RunId, &str, &serde_json::Value) -> JudgedTerminal + Send + Sync;

/// Test seams on the judgment path, by session: drop a judgment before the
/// dispatcher takes it, or make the kernel judge a call with a given scope.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) enum JudgmentSeam {
    DropJudgment,
    JudgedScope(TerminalScope),
}

#[cfg(test)]
pub(crate) static JUDGMENT_SEAMS: std::sync::LazyLock<Mutex<HashMap<String, JudgmentSeam>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
pub(crate) fn judgment_seam(session_id: &str) -> Option<JudgmentSeam> {
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
pub(crate) struct JudgedCalls(Arc<Mutex<HashMap<(sdk::RunId, String), JudgedTerminal>>>);

impl JudgedCalls {
    /// The judged form of `call_id` in `run`, if the kernel judged it.
    pub(crate) fn get(&self, run: sdk::RunId, call_id: &str) -> Option<JudgedTerminal> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(run, call_id.to_string()))
            .cloned()
    }

    /// Remove and return the judged form of `call_id` in `run`.
    pub(crate) fn take(&self, run: sdk::RunId, call_id: &str) -> Option<JudgedTerminal> {
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
pub(crate) struct TerminalApprovals {
    tool: Arc<dyn crate::traits::Tool>,
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
    pub(crate) fn find(tools: &[Arc<dyn crate::traits::Tool>]) -> Option<Self> {
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
    pub(crate) fn tool(&self) -> Arc<dyn crate::traits::Tool> {
        self.tool.clone()
    }

    /// Canonicalize and scope-check calls as the turn's dispatcher does,
    /// recording each judged call in `judged` for the dispatcher.
    pub(crate) fn with_canonicalizer(
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
    pub(crate) fn judge(
        &self,
        run: sdk::RunId,
        call_id: &str,
        raw: &serde_json::Value,
    ) -> JudgedTerminal {
        match &self.canonicalize {
            Some(canonicalize) => canonicalize(run, call_id, raw),
            None => JudgedTerminal {
                call: CanonicalCall::stripped(Some(self.tool.as_ref()), raw),
                scope: TerminalScope::InScope,
            },
        }
    }

    /// The judged form of the call `call_id` in `run`, if the kernel judged it.
    pub(crate) fn judged(&self, run: sdk::RunId, call_id: &str) -> Option<JudgedTerminal> {
        self.judged.get(run, call_id)
    }

    fn remember(&self, run: sdk::RunId, call_id: &str, call: JudgedTerminal) {
        self.judged.insert(run, call_id, call);
    }
}

/// The rules' decision for what a tool said about a call.
fn rule_decision(need: crate::traits::OwnerApprovalNeed) -> RuleDecision {
    match need {
        crate::traits::OwnerApprovalNeed::Ask(request) => RuleDecision::Ask(request),
        crate::traits::OwnerApprovalNeed::NotNeeded => RuleDecision::Allow,
        crate::traits::OwnerApprovalNeed::Unjudged => RuleDecision::Defer,
    }
}

impl AppApprovalRules {
    /// The dispatcher's facts for a call of this turn, with the scope
    /// escalation the dispatcher judged for it.
    fn call_facts(
        &self,
        escalation: Option<crate::traits::ScopeEscalation>,
    ) -> crate::traits::ToolCallContext {
        crate::traits::ToolCallContext {
            session_id: self.session_id.clone(),
            user_role: self.user_role,
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

    fn tool(&self, name: &str) -> Option<&Arc<dyn crate::traits::Tool>> {
        self.judged_tool(name)
            .map(|approvals| &approvals.tool)
            .or_else(|| self.tools.find(name))
    }

    /// The mode a card for `tool` is sent with: the tool's own
    /// (`approval_card_mode`), Cautious otherwise.
    pub(crate) fn card_permission_mode(&self, tool: &str) -> PermissionMode {
        self.tool(tool)
            .and_then(|tool| tool.approval_card_mode())
            .unwrap_or(PermissionMode::Cautious)
    }

    /// Whether `tool` keeps the owner's answers in its own grant stores
    /// (`dispatch_facts().keeps_own_grants`) instead of the session grants.
    pub(crate) fn keeps_own_grants(&self, tool: &str) -> bool {
        self.tool(tool)
            .is_some_and(|tool| tool.dispatch_facts().keeps_own_grants)
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
        let approval_call = crate::traits::ApprovalCall {
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
    pub(crate) async fn record_owner_answer(
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
            let approval_call = crate::traits::ApprovalCall {
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
                &crate::traits::ApprovalCall::as_sent(&call.arguments),
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
        self.run_declines.observe(ctx.declined);
        if let Some(approvals) = self.judged_tool(&call.name) {
            return self.judge_canonical(approvals, ctx.run_id, call).await;
        }
        let Some(tool) = self.tools.find(&call.name) else {
            return RuleDecision::Defer;
        };
        rule_decision(
            tool.owner_approval_request(
                &crate::traits::ApprovalCall::as_sent(&call.arguments),
                &self.call_facts(None),
            )
            .await,
        )
    }
}

/// The owner's approvals across turns: the card channel, one set of grants
/// per chat session, and the cards still waiting for an answer.
///
/// A chat's grants are shared with every child agent spawned from it (they
/// are keyed by [`grant_session`]): "Allow for session" on a request in a
/// chat also covers the identical request from that chat's children.
pub(crate) struct OwnerApprovals {
    broker: std::sync::RwLock<Option<ApprovalBroker>>,
    grants: Mutex<HashMap<String, Arc<SessionGrants>>>,
    /// Cards waiting for the owner. Dropping an entry (a restart loses them
    /// all) ends the wait as unanswered.
    pending: Mutex<HashMap<u64, oneshot::Sender<()>>>,
    next_card: AtomicU64,
    timeout: Duration,
}

impl Default for OwnerApprovals {
    fn default() -> Self {
        Self {
            broker: std::sync::RwLock::new(None),
            grants: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
            next_card: AtomicU64::new(0),
            timeout: OWNER_APPROVAL_TIMEOUT,
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
    #[cfg(test)]
    pub(crate) fn with_timeout(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Self::default()
        }
    }

    /// The channel cards go out on (set once channels are up).
    pub(crate) fn set_broker(&self, broker: ApprovalBroker) {
        *self.broker.write().unwrap_or_else(|e| e.into_inner()) = Some(broker);
    }

    /// The grants of one chat session.
    pub(crate) fn grants_for(&self, session_id: &str) -> Arc<SessionGrants> {
        self.grants
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(session_id.to_string())
            .or_default()
            .clone()
    }

    /// The owner approved `request` for the rest of the session. For
    /// http_request, Always means Session, as it always did.
    pub(crate) fn record_grant(&self, session_id: &str, request: &sdk::OwnerApprovalRequest) {
        self.grants_for(session_id).grant(request);
    }

    /// Forget every waiting card, as a restart does.
    #[cfg(test)]
    pub(crate) fn abandon_pending(&self) {
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
                kind: Default::default(),
            })
            .await;
        let answer = if sent.is_err() {
            Answer::Declined(COULD_NOT_REACH.to_string())
        } else {
            tokio::select! {
                response = response_rx => match response {
                    Ok(response) => Answer::Granted(response),
                    // The channel dropped the card without an answer.
                    Err(_) => Answer::Declined(COULD_NOT_REACH.to_string()),
                },
                _ = tokio::time::sleep(self.timeout) => {
                    Answer::Declined(no_answer_reason(self.timeout))
                }
                // The waiting entry is gone (a restart): the card is stale.
                _ = abandon_rx => Answer::Declined(no_answer_reason(self.timeout)),
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
pub(crate) fn grant_session(approval_session_id: Option<&str>, session_id: &str) -> String {
    approval_session_id.unwrap_or(session_id).to_string()
}

/// The card's risk level for a kernel request.
pub(crate) fn card_risk(risk: sdk::ApprovalRisk) -> RiskLevel {
    match risk {
        sdk::ApprovalRisk::Medium => RiskLevel::Medium,
        sdk::ApprovalRisk::Critical => RiskLevel::Critical,
        _ => RiskLevel::High,
    }
}

fn risk_name(risk: sdk::ApprovalRisk) -> &'static str {
    match risk {
        sdk::ApprovalRisk::Medium => "medium",
        sdk::ApprovalRisk::Critical => "critical",
        _ => "high",
    }
}

/// The decline reason for a card nobody answered within `timeout`.
pub(crate) fn no_answer_reason(timeout: Duration) -> String {
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
pub(crate) struct ConfiguredSecrets(Vec<String>);

impl ConfiguredSecrets {
    pub(crate) fn new(values: Vec<String>) -> Self {
        let mut values: Vec<String> = values
            .into_iter()
            .filter(|value| !value.trim().is_empty())
            .collect();
        values.sort();
        values.dedup();
        Self(values)
    }

    pub(crate) fn values(&self) -> &[String] {
        &self.0
    }
}

impl std::fmt::Debug for ConfiguredSecrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConfiguredSecrets({} values)", self.0.len())
    }
}

/// Asks the owner about the kernel's owner-approval interruptions.
pub(in crate::agent) struct OwnerApprovalResolver<'a> {
    pub agent: &'a Agent,
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
        self.agent
            .emit_decision_point(
                self.emitter,
                self.task_id,
                0,
                crate::events::DecisionType::GateTelemetry,
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

    async fn resolve(
        &self,
        run: sdk::RunId,
        interruption: &sdk::Interruption,
        request: &sdk::OwnerApprovalRequest,
    ) -> sdk::Resolution {
        let tool = interruption.call.name.as_str();
        self.record("owner_approval_requested", tool, request, None)
            .await;
        match self
            .approvals
            .ask(
                self.card_session,
                request,
                self.rules.card_permission_mode(tool),
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

/// The turn's resolver: untrusted-content checks go to the clean verifier,
/// owner approvals to the owner.
pub(in crate::agent) struct DaemonInterruptResolver<'a> {
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
                sdk::InterruptResolver::resolve(&self.untrusted, state, interruption).await
            }
            _ => Some(unknown_interrupt_kind(interruption)),
        }
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
        }
    }

    /// `manage_api`'s approval judgment without its stores (which need a
    /// database): the same `ManageApiTool::owner_approval` the tool runs.
    struct ManageApiApprovals;

    #[async_trait]
    impl crate::traits::ToolSemantics for ManageApiApprovals {
        async fn owner_approval_request(
            &self,
            call: &crate::traits::ApprovalCall<'_>,
            facts: &crate::traits::ToolCallContext,
        ) -> crate::traits::OwnerApprovalNeed {
            crate::tools::ManageApiTool::owner_approval(call.arguments, facts)
        }
    }

    #[async_trait]
    impl crate::traits::Tool for ManageApiApprovals {
        fn name(&self) -> &str {
            "manage_api"
        }
        fn description(&self) -> &str {
            "manage_api approvals"
        }
        fn schema(&self) -> serde_json::Value {
            json!({"name": "manage_api", "description": "", "parameters": {"type": "object"}})
        }
        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            anyhow::bail!("not run in these tests")
        }
    }

    /// `manage_mcp`'s approval judgment without its registry (which needs a
    /// database): the same `ManageMcpTool::owner_approval` the tool runs.
    struct ManageMcpApprovals;

    #[async_trait]
    impl crate::traits::ToolSemantics for ManageMcpApprovals {
        async fn owner_approval_request(
            &self,
            call: &crate::traits::ApprovalCall<'_>,
            _facts: &crate::traits::ToolCallContext,
        ) -> crate::traits::OwnerApprovalNeed {
            crate::tools::ManageMcpTool::owner_approval(call.arguments)
        }
    }

    #[async_trait]
    impl crate::traits::Tool for ManageMcpApprovals {
        fn name(&self) -> &str {
            "manage_mcp"
        }
        fn description(&self) -> &str {
            "manage_mcp approvals"
        }
        fn schema(&self) -> serde_json::Value {
            json!({"name": "manage_mcp", "description": "", "parameters": {"type": "object"}})
        }
        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            anyhow::bail!("not run in these tests")
        }
    }

    /// Rules over the turn's approval-judging tools: a real `http_request`,
    /// `manage_api`'s judgment and `manage_mcp`'s judgment.
    fn rules() -> AppApprovalRules {
        let http: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::HttpRequestTool::new(
            Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        ));
        AppApprovalRules {
            tools: ApprovalTools::new([
                http,
                Arc::new(ManageApiApprovals) as Arc<_>,
                Arc::new(ManageMcpApprovals) as Arc<_>,
            ]),
            ..AppApprovalRules::default()
        }
    }

    async fn judge(rules: &AppApprovalRules, args: serde_json::Value) -> RuleDecision {
        let call = sdk::ToolCall::new("c1", "http_request", args);
        rules.judge(&hook_ctx(), &call, &tool_info()).await
    }

    /// An MCP start asks the owner in every session, and a call the add path
    /// refuses needs no card (it never runs).
    #[tokio::test]
    async fn mcp_add_always_asks_before_dispatch() {
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                trusted_session: trusted,
                mandate_preapproved: trusted,
                ..rules()
            };
            let add = sdk::ToolCall::new(
                "c1",
                "manage_mcp",
                json!({"action":"add","name":"synthetic","command":"node","args":["server.js"]}),
            );
            assert!(
                matches!(
                    rules.judge(&hook_ctx(), &add, &tool_info()).await,
                    RuleDecision::Ask(_)
                ),
                "trusted={trusted}"
            );
            let refused = sdk::ToolCall::new(
                "c2",
                "manage_mcp",
                json!({"action":"add","name":"synthetic","command":"sh"}),
            );
            assert_eq!(
                rules.judge(&hook_ctx(), &refused, &tool_info()).await,
                RuleDecision::Allow
            );
            let list = sdk::ToolCall::new("c3", "manage_mcp", json!({"action":"list"}));
            assert_eq!(
                rules.judge(&hook_ctx(), &list, &tool_info()).await,
                RuleDecision::Allow
            );
        }
    }

    /// Exercise the real app rules and SDK policy with a process-effect
    /// probe. The alternate name proves declines follow effects, not names.
    #[tokio::test]
    async fn mcp_approval_binds_process_effects_across_tools() {
        use sdk::testing::{call, spec, text, tool_calls, ScriptedProvider};
        struct ProcessProbe(&'static str, Arc<AtomicU64>);
        #[async_trait]
        impl sdk::Tool for ProcessProbe {
            fn spec(&self) -> sdk::ToolSpec {
                spec(self.0, "synthetic process probe", &[])
            }
            fn annotations(&self) -> sdk::ToolAnnotations {
                sdk::ToolAnnotations::WRITE
            }
            fn effects(&self, args: &serde_json::Value) -> sdk::CallEffects {
                crate::tools::ManageMcpTool::effects_for(&args.to_string())
            }
            async fn call(&self, _: serde_json::Value, _: &sdk::ToolContext) -> sdk::ToolResult {
                self.1.fetch_add(1, Ordering::SeqCst);
                Ok(sdk::ToolOutput::text("started"))
            }
        }
        for approve in [false, true] {
            let executed = Arc::new(AtomicU64::new(0));
            let args =
                json!({"action":"add","name":"synthetic","command":"node","args":["server.js"]});
            let policy =
                aidaemon_policies::ApprovalPolicy::new(aidaemon_policies::ApprovalMode::Never)
                    .with_rules(Arc::new(rules()));
            let agent = sdk::Agent::builder("synthetic")
                .allow_unguarded()
                .model("mock")
                .provider(ScriptedProvider::new([
                    tool_calls([call("c1", "manage_mcp", args.clone())]),
                    tool_calls([call("c2", "alternate_process", args)]),
                    text("Finished considering the request."),
                ]))
                .tool(ProcessProbe("manage_mcp", executed.clone()))
                .tool(ProcessProbe("alternate_process", executed.clone()))
                .hook(policy)
                .build()
                .unwrap();
            let pending = agent
                .run("synthetic-session", "Add the server")
                .await
                .unwrap();
            assert!(matches!(pending.status, sdk::RunStatus::Interrupted { .. }));
            assert_eq!(executed.load(Ordering::SeqCst), 0);
            let resolution = if approve {
                sdk::Resolution::Approve
            } else {
                sdk::Resolution::Deny {
                    reason: "No".into(),
                }
            };
            let finished = agent.resume(pending.state, resolution).await.unwrap();
            assert!(matches!(finished.status, sdk::RunStatus::Completed));
            assert_eq!(executed.load(Ordering::SeqCst), if approve { 2 } else { 0 });
            if !approve {
                assert_eq!(
                    finished.state.declined,
                    vec![
                        sdk::EffectKey::named(sdk::EffectClass::Process, "node"),
                        sdk::EffectKey::named(sdk::EffectClass::Process, "mcp-server:synthetic"),
                    ]
                );
                assert!(finished
                    .receipts()
                    .iter()
                    .all(|r| r.status == sdk::ReceiptStatus::Denied));
            }
        }
    }

    #[tokio::test]
    async fn observation_methods_without_body_are_allowed() {
        for method in ["GET", "HEAD", "OPTIONS", "get"] {
            let decision = judge(
                &rules(),
                json!({"method": method, "url": "https://h.example.invalid/x"}),
            )
            .await;
            assert_eq!(decision, RuleDecision::Allow, "{method}");
        }
    }

    #[tokio::test]
    async fn writes_and_bodies_ask_the_owner() {
        let body = "b".repeat(700);
        for (method, body) in [
            ("POST", Some(body.as_str())),
            ("DELETE", None),
            ("GET", Some("q=1")),
        ] {
            let mut args = json!({"method": method, "url": "https://h.example.invalid/x"});
            if let Some(body) = body {
                args["body"] = json!(body);
            }
            let RuleDecision::Ask(request) = judge(&rules(), args).await else {
                panic!("{method} must ask");
            };
            assert_eq!(request.risk, sdk::ApprovalRisk::High);
            assert_eq!(
                request.subject,
                format!("{method} https://h.example.invalid/x")
            );
            match body {
                Some(body) => {
                    let shown: String = body.chars().take(500).collect();
                    assert_eq!(
                        request.summary,
                        format!("{method} https://h.example.invalid/x\nBody: {shown}")
                    );
                }
                None => assert_eq!(request.summary, request.subject),
            }
        }
    }

    #[tokio::test]
    async fn method_override_header_asks() {
        let decision = judge(
            &rules(),
            json!({
                "method": "GET",
                "url": "https://h.example.invalid/x",
                "headers": {"X-HTTP-Method-Override": "DELETE"},
            }),
        )
        .await;
        assert!(matches!(decision, RuleDecision::Ask(_)), "{decision:?}");
    }

    #[tokio::test]
    async fn trusted_session_and_mandate_preapproval_allow() {
        let post = json!({"method": "POST", "url": "https://h.example.invalid/x", "body": "x"});
        let trusted = AppApprovalRules {
            trusted_session: true,
            ..rules()
        };
        assert_eq!(judge(&trusted, post.clone()).await, RuleDecision::Allow);
        let mandate = AppApprovalRules {
            mandate_preapproved: true,
            ..rules()
        };
        assert_eq!(judge(&mandate, post).await, RuleDecision::Allow);
    }

    #[tokio::test]
    async fn other_tools_defer() {
        let call = sdk::ToolCall::new("c1", "write_file", json!({"path": "/tmp/aidaemon-test/x"}));
        let decision = rules().judge(&hook_ctx(), &call, &tool_info()).await;
        assert_eq!(decision, RuleDecision::Defer);
    }

    #[tokio::test]
    async fn asks_always_carry_a_subject() {
        // Even an unparseable URL names the call it asks about.
        let RuleDecision::Ask(request) =
            judge(&rules(), json!({"method": "POST", "url": "not a url"})).await
        else {
            panic!("must ask");
        };
        assert!(!request.subject.trim().is_empty());
    }

    /// Sharing a stored memory asks the owner in every session; a call
    /// naming no active fact changes nothing and needs no card, nor does a
    /// call the dispatcher refuses.
    #[tokio::test]
    async fn share_memory_asks_before_widening_a_memory() {
        use crate::traits::ToolSemantics;
        let tool = Arc::new(crate::tools::ShareMemoryTool::with_synthetic_fact(
            "preference",
            "report_format",
            "short summaries for Jordan Lee",
        ));
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                tools: ApprovalTools::new([tool.clone() as Arc<dyn crate::traits::Tool>]),
                trusted_session: trusted,
                mandate_preapproved: trusted,
                ..AppApprovalRules::default()
            };
            let share = sdk::ToolCall::new(
                "c1",
                "share_memory",
                json!({"category": "preference", "key": "report_format"}),
            );
            let RuleDecision::Ask(request) = rules.judge(&hook_ctx(), &share, &tool_info()).await
            else {
                panic!("sharing must ask (trusted={trusted})");
            };
            assert_eq!(request.risk, sdk::ApprovalRisk::High);
            let missing = sdk::ToolCall::new(
                "c2",
                "share_memory",
                json!({"category": "preference", "key": "absent"}),
            );
            assert_eq!(
                rules.judge(&hook_ctx(), &missing, &tool_info()).await,
                RuleDecision::Allow
            );
        }
        let args = json!({"category": "preference", "key": "report_format"});
        let refused = crate::traits::ApprovalCall {
            refused: true,
            ..crate::traits::ApprovalCall::as_sent(&args)
        };
        assert!(matches!(
            tool.owner_approval_request(&refused, &Default::default())
                .await,
            crate::traits::OwnerApprovalNeed::NotNeeded
        ));
    }

    #[test]
    fn risks_map_to_card_levels() {
        use crate::tools::command_risk::RiskLevel;
        assert_eq!(card_risk(sdk::ApprovalRisk::Medium), RiskLevel::Medium);
        assert_eq!(card_risk(sdk::ApprovalRisk::High), RiskLevel::High);
        assert_eq!(card_risk(sdk::ApprovalRisk::Critical), RiskLevel::Critical);
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

    #[test]
    fn manage_api_derived_probe_asks_at_the_outer_call() {
        let onboard = |extra: serde_json::Value| {
            let mut args = json!({
                "action": "onboard",
                "service": "synthetic-api",
                "learn_url": "https://docs.example.invalid/graphql",
            });
            for (key, value) in extra.as_object().unwrap() {
                args[key] = value.clone();
            }
            crate::tools::ManageApiTool::probe_approval_request(&args)
        };
        let request = onboard(json!({})).expect("a derived probe may POST");
        assert_eq!(
            request.subject,
            "manage_api probe POST https://docs.example.invalid/graphql"
        );
        assert_eq!(request.risk, sdk::ApprovalRisk::High);
        // A verify_url probe is GET or HEAD; an OpenAPI source probes with
        // GET or HEAD only.
        assert!(onboard(json!({"verify_url": "https://api.example.invalid/me"})).is_none());
        assert!(
            onboard(json!({"openapi_url": "https://api.example.invalid/openapi.json"})).is_none()
        );
    }

    #[tokio::test]
    async fn manage_api_rules_ask_unless_trusted() {
        let args = json!({
            "action": "onboard",
            "service": "synthetic-api",
            "learn_url": "https://docs.example.invalid/graphql",
        });
        let call = sdk::ToolCall::new("c1", "manage_api", args);
        let asked = rules().judge(&hook_ctx(), &call, &tool_info()).await;
        assert!(matches!(asked, RuleDecision::Ask(_)), "{asked:?}");
        let trusted = AppApprovalRules {
            trusted_session: true,
            ..rules()
        };
        assert_eq!(
            trusted.judge(&hook_ctx(), &call, &tool_info()).await,
            RuleDecision::Defer
        );
    }

    /// Rules over a real terminal with `allowed` prefixes and no semantic
    /// assessor (an unknown command then fails closed to a Critical ask).
    async fn terminal_rules(allowed: &[&str]) -> AppApprovalRules {
        let pool_file = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", pool_file.path().display()))
            .await
            .unwrap();
        std::mem::forget(pool_file);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let terminal: Arc<dyn crate::traits::Tool> = Arc::new(
            crate::tools::TerminalTool::new(
                crate::execution::active_execution_backend(),
                allowed.iter().map(|p| p.to_string()).collect(),
                crate::tools::ApprovalBroker::new(tx),
                1,
                1000,
                crate::types::PermissionMode::Default,
                pool,
            )
            .await,
        );
        AppApprovalRules {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_role: Some(crate::types::UserRole::Owner),
            terminal: TerminalApprovals::find(&[terminal]),
            ..rules()
        }
    }

    /// The terminal with every call judged to have `scope`.
    fn with_scope(terminal: TerminalApprovals, scope: TerminalScope) -> TerminalApprovals {
        terminal.with_canonicalizer(
            move |_, _, raw| JudgedTerminal {
                call: CanonicalCall::stripped(None, raw),
                scope: scope.clone(),
            },
            JudgedCalls::default(),
        )
    }

    async fn judge_terminal(rules: &AppApprovalRules, args: serde_json::Value) -> RuleDecision {
        let call = sdk::ToolCall::new("c1", "terminal", args);
        rules.judge(&hook_ctx(), &call, &tool_info()).await
    }

    fn run(command: &str) -> serde_json::Value {
        json!({"action": "run", "command": command})
    }

    fn asks_with(decision: &RuleDecision, warning: &str) -> bool {
        matches!(decision, RuleDecision::Ask(request)
            if request.warnings.iter().any(|w| w.contains(warning)))
    }

    /// The T8 ladder, in order: an untrusted trigger asks before the
    /// trusted-session and explicit-allow rungs; a trusted session and an
    /// explicit allow both come before the approval floor; an unknown
    /// command with no assessor fails closed, Critical.
    #[tokio::test]
    async fn terminal_ladder_keeps_its_order() {
        let base = terminal_rules(&["echo", "cat"]).await;
        let trusted = AppApprovalRules {
            trusted_session: true,
            ..base.clone()
        };
        let trigger = AppApprovalRules {
            untrusted_trigger_session: true,
            ..trusted.clone()
        };
        // Explicit allow.
        assert_eq!(
            judge_terminal(&base, run("echo hi")).await,
            RuleDecision::Allow
        );
        // Untrusted trigger beats both the trusted session and the allowlist.
        let asked = judge_terminal(&trigger, run("echo hi")).await;
        assert!(asks_with(&asked, "untrusted external trigger"), "{asked:?}");
        // A trusted session runs an unknown command without asking.
        assert_eq!(
            judge_terminal(&trusted, run("printf synthetic")).await,
            RuleDecision::Allow
        );
        // The approval floor precedes the explicit allow (owner ruling,
        // closing the T10 hole): an allowlisted command touching a
        // credential path asks; a plain allowlisted command does not.
        for sensitive in ["cat ~/.ssh/config", "cat ~/.ssh/id_rsa"] {
            let asked = judge_terminal(&base, run(sensitive)).await;
            assert!(
                matches!(asked, RuleDecision::Ask(_)),
                "{sensitive}: {asked:?}"
            );
        }
        let plain = terminal_rules(&["ls"]).await;
        assert_eq!(judge_terminal(&plain, run("ls")).await, RuleDecision::Allow);
        // The floor asks for a credential path that no grant covers.
        let floor = judge_terminal(&base, run("head ~/.ssh/config")).await;
        assert!(matches!(floor, RuleDecision::Ask(_)), "{floor:?}");
        // Unknown command, no assessor: Critical, failed closed.
        let unknown = judge_terminal(&base, run("printf synthetic")).await;
        let RuleDecision::Ask(request) = &unknown else {
            panic!("{unknown:?}");
        };
        assert_eq!(request.risk, sdk::ApprovalRisk::Critical);
        assert_eq!(request.subject, "printf synthetic");
        assert!(asks_with(&unknown, "not configured"), "{unknown:?}");
        // A novel detached command asks; an allowlisted one does not.
        let detached = json!({"action": "run", "command": "printf x", "detach": true});
        assert!(matches!(
            judge_terminal(&base, detached).await,
            RuleDecision::Ask(_)
        ));
        let allowed_detached = json!({"action": "run", "command": "echo x", "detach": true});
        assert_eq!(
            judge_terminal(&base, allowed_detached).await,
            RuleDecision::Allow
        );
    }

    /// Daemonization: the owner gets a Critical card; a non-owner's call or
    /// one without `detach=true` is refused by the tool's hard block, so the
    /// rules never ask about it.
    #[tokio::test]
    async fn terminal_daemonization_asks_the_owner_critical() {
        let owner = terminal_rules(&["*"]).await;
        let detached = json!({"action": "run", "command": "nohup sleep 1 &", "detach": true});
        let RuleDecision::Ask(request) = judge_terminal(&owner, detached.clone()).await else {
            panic!("the owner is asked even under trust-all");
        };
        assert_eq!(request.risk, sdk::ApprovalRisk::Critical);
        assert!(request
            .warnings
            .iter()
            .any(|w| w.contains("Daemonization primitives detected")));
        let guest = AppApprovalRules {
            user_role: Some(crate::types::UserRole::Guest),
            ..owner.clone()
        };
        assert_eq!(judge_terminal(&guest, detached).await, RuleDecision::Allow);
        let attached = json!({"action": "run", "command": "nohup sleep 1 &"});
        assert_eq!(judge_terminal(&owner, attached).await, RuleDecision::Allow);
    }

    /// Trust markers in a terminal call's arguments change nothing: the
    /// rules read the turn's facts only.
    #[tokio::test]
    async fn terminal_trust_arguments_bypass_nothing() {
        let rules = terminal_rules(&[]).await;
        let spoofed = json!({
            "action": "run",
            "command": "printf synthetic",
            "_trusted_session": true,
            "_user_role": "Owner",
        });
        assert!(matches!(
            judge_terminal(&rules, spoofed).await,
            RuleDecision::Ask(_)
        ));
    }

    /// Enabling trust-all always asks, Critical (until it is on).
    #[tokio::test]
    async fn terminal_trust_all_asks_critical() {
        let trusted = AppApprovalRules {
            trusted_session: true,
            ..terminal_rules(&[]).await
        };
        let RuleDecision::Ask(request) =
            judge_terminal(&trusted, json!({"action": "trust_all"})).await
        else {
            panic!("trust_all must ask");
        };
        assert_eq!(request.risk, sdk::ApprovalRisk::Critical);
        assert_eq!(request.subject, "ENABLE TRUST-ALL MODE");
        // Always asked, even when it is already on, and whatever the
        // model's `_` keys say.
        let on = terminal_rules(&["*"]).await;
        assert!(matches!(
            judge_terminal(&on, json!({"action": "trust_all", "_goal_id": 0})).await,
            RuleDecision::Ask(_)
        ));
    }

    /// A terminal call whose arguments cannot be read is never allowed: it
    /// asks, Critical, with the raw arguments on the card.
    #[tokio::test]
    async fn unreadable_terminal_call_asks_critical_with_raw_arguments() {
        let rules = terminal_rules(&["*"]).await;
        let raw = json!({"action": "run", "command": 5, "detach": "yes"});
        let RuleDecision::Ask(request) = judge_terminal(&rules, raw.clone()).await else {
            panic!("an unreadable call must ask");
        };
        assert_eq!(request.risk, sdk::ApprovalRisk::Critical);
        assert!(request.summary.contains(&raw.to_string()), "{request:?}");
        assert!(
            request
                .warnings
                .iter()
                .any(|w| w.contains("could not be read")),
            "{request:?}"
        );
        // The model's `_` keys of the wrong type are dropped first, as the
        // dispatcher drops them: the call is judged on what runs.
        let spoofed = json!({"action": "run", "command": "ls", "_session_id": 1});
        assert_eq!(judge_terminal(&rules, spoofed).await, RuleDecision::Allow);
    }

    /// Terminal cards go out in the terminal's configured mode (which picks
    /// the buttons); other tools' cards stay Cautious.
    #[tokio::test]
    async fn terminal_cards_use_the_terminal_permission_mode() {
        let rules = terminal_rules(&[]).await;
        assert_eq!(
            rules.card_permission_mode("terminal"),
            PermissionMode::Default
        );
        assert_eq!(
            rules.card_permission_mode("http_request"),
            PermissionMode::Cautious
        );
    }

    /// Allow Session adds the session prefix, as the in-tool prompt did:
    /// the next call runs without a card, but a trigger session still asks.
    #[tokio::test]
    async fn terminal_allow_session_adds_the_session_prefix() {
        let rules = terminal_rules(&[]).await;
        let call = sdk::ToolCall::new("c1", "terminal", run("printf first"));
        rules
            .record_owner_answer(
                sdk::RunId::new(),
                &call,
                Some(ApprovalResponse::AllowSession),
            )
            .await;
        assert_eq!(
            judge_terminal(&rules, run("printf second")).await,
            RuleDecision::Allow
        );
        let trigger = AppApprovalRules {
            untrusted_trigger_session: true,
            ..rules.clone()
        };
        assert!(matches!(
            judge_terminal(&trigger, run("printf second")).await,
            RuleDecision::Ask(_)
        ));
        // A decline grants nothing.
        let other = sdk::ToolCall::new("c2", "terminal", run("seq 1 2"));
        rules
            .record_owner_answer(sdk::RunId::new(), &other, Some(ApprovalResponse::Deny))
            .await;
        assert!(matches!(
            judge_terminal(&rules, run("seq 1 2")).await,
            RuleDecision::Ask(_)
        ));
    }

    /// The dispatcher refuses an out-of-scope call it cannot escalate, so
    /// the rules do not ask about it; on a confined terminal an escalatable
    /// target is asked about even for an allowlisted command, and an Allow
    /// Session grants that capability for the command.
    #[tokio::test]
    async fn terminal_scope_check_shapes_the_ask() {
        let refused = AppApprovalRules {
            terminal: terminal_rules(&["echo"])
                .await
                .terminal
                .map(|t| with_scope(t, TerminalScope::Refused("protected host data".to_string()))),
            ..terminal_rules(&[]).await
        };
        assert_eq!(
            judge_terminal(&refused, run("printf synthetic")).await,
            RuleDecision::Allow
        );

        let pool_file = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", pool_file.path().display()))
            .await
            .unwrap();
        std::mem::forget(pool_file);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let confined: Arc<dyn crate::traits::Tool> = Arc::new(
            crate::tools::TerminalTool::new(
                crate::execution::active_execution_backend(),
                vec!["cat".to_string()],
                crate::tools::ApprovalBroker::new(tx),
                1,
                1000,
                crate::types::PermissionMode::Default,
                pool,
            )
            .await
            .with_confinement(crate::types::TerminalConfinement::Sandbox),
        );
        let escalation = crate::traits::ScopeEscalation {
            read_paths: vec!["/tmp/aidaemon-test/sibling/notes.txt".to_string()],
            write_paths: Vec::new(),
        };
        let scoped = escalation.clone();
        let rules = AppApprovalRules {
            terminal: TerminalApprovals::find(&[confined])
                .map(|t| with_scope(t, TerminalScope::Escalate(scoped.clone()))),
            ..terminal_rules(&[]).await
        };
        let args = run("cat /tmp/aidaemon-test/sibling/notes.txt");
        let asked = judge_terminal(&rules, args.clone()).await;
        assert!(
            asks_with(
                &asked,
                "read outside task scope: /tmp/aidaemon-test/sibling/notes.txt"
            ),
            "{asked:?}"
        );
        let call = sdk::ToolCall::new("c1", "terminal", args.clone());
        rules
            .record_owner_answer(
                sdk::RunId::new(),
                &call,
                Some(ApprovalResponse::AllowSession),
            )
            .await;
        assert_eq!(judge_terminal(&rules, args).await, RuleDecision::Allow);
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
        rules.run_declines.observe(&[]);
        assert_eq!(rules.run_declines.snapshot(), declined.to_vec());
    }

    #[test]
    fn configured_secrets_never_print() {
        let secrets = ConfiguredSecrets::new(vec!["synthetic-token-123456".to_string()]);
        assert!(!format!("{secrets:?}").contains("synthetic-token"));
    }
}
