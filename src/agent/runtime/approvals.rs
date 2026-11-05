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
/// `http_request` keeps H7: a GET, HEAD or OPTIONS without a body (and
/// without a method or authority override header) runs; anything else asks
/// the owner, High risk, unless the turn is a trusted scheduled session or
/// a mandate run (whose exact action grant the dispatcher checks). Every
/// other tool defers to the policy's mode (`Never`), so it runs as before.
#[derive(Debug, Clone, Default)]
pub(crate) struct AppApprovalRules {
    /// A trusted scheduled session (never under a mandate): no card, as the
    /// tools skipped it for `_trusted_session`.
    pub trusted_session: bool,
    /// A mandate run: the dispatcher admits only calls its exact action
    /// grant covers, so the owner already approved them.
    pub mandate_preapproved: bool,
}

#[async_trait]
impl ApprovalRules for AppApprovalRules {
    async fn judge(
        &self,
        _ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _info: &sdk::ToolInfo,
    ) -> RuleDecision {
        match call.name.as_str() {
            "http_request" => {
                if self.trusted_session || self.mandate_preapproved {
                    return RuleDecision::Allow;
                }
                match crate::tools::HttpRequestTool::owner_approval_request(&call.arguments) {
                    Some(request) => RuleDecision::Ask(request),
                    None => RuleDecision::Allow,
                }
            }
            _ => RuleDecision::Defer,
        }
    }
}

/// The owner's approvals across turns: the card channel, one set of grants
/// per chat session, and the cards still waiting for an answer.
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
                permission_mode: PermissionMode::Cautious,
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
        interruption: &sdk::Interruption,
        request: &sdk::OwnerApprovalRequest,
    ) -> sdk::Resolution {
        let tool = interruption.call.name.as_str();
        self.record("owner_approval_requested", tool, request, None)
            .await;
        match self
            .approvals
            .ask(self.card_session, request, &self.cancel)
            .await
        {
            Answer::Granted(response) => {
                if matches!(
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
            sdk::InterruptKind::OwnerApproval(request) => {
                Some(self.owner.resolve(interruption, request).await)
            }
            _ => sdk::InterruptResolver::resolve(&self.untrusted, state, interruption).await,
        }
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

    fn tool_info() -> sdk::ToolInfo {
        sdk::ToolInfo {
            annotations: sdk::ToolAnnotations::default(),
            access: sdk::AccessManifest::default(),
            effects: sdk::CallEffects::unknown(),
        }
    }

    fn rules() -> AppApprovalRules {
        AppApprovalRules::default()
    }

    async fn judge(rules: &AppApprovalRules, args: serde_json::Value) -> RuleDecision {
        let call = sdk::ToolCall::new("c1", "http_request", args);
        rules.judge(&hook_ctx(), &call, &tool_info()).await
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

    #[test]
    fn configured_secrets_never_print() {
        let secrets = ConfiguredSecrets::new(vec!["synthetic-token-123456".to_string()]);
        assert!(!format!("{secrets:?}").contains("synthetic-token"));
    }
}
