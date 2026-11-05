//! Approval as a `before_tool` hook: pause for the owner (a typed
//! [`InterruptKind::OwnerApproval`] interruption) or refuse (Deny) based on
//! annotations, name rules, tokenized command rules, application rules and
//! standing grants.

use std::collections::HashSet;
use std::sync::Arc;

use aidaemon_core::{
    ApprovalRisk, EffectClass, Hook, HookContext, InterruptKind, OwnerApprovalRequest, Safeguard,
    ToolCall, ToolDecision, ToolInfo, OWNER_DECLINED,
};
use async_trait::async_trait;
use serde_json::Value;

use crate::shell::{catastrophic_reason, command_segments, segment_matches_prefix};

/// When to ask a human before a call runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalMode {
    /// Never ask (rules and hard blocks still apply).
    Never,
    /// Ask before tools annotated `destructive`.
    #[default]
    DestructiveOnly,
    /// Ask before anything not annotated `read_only`.
    AllWrites,
    /// Ask before every call.
    Always,
}

/// What an application's [`ApprovalRules`] decide about one call.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RuleDecision {
    /// Run it without asking.
    Allow,
    /// Refuse it.
    Deny { reason: String },
    /// Ask the owner with this request (unless a grant covers it). The
    /// policy fills `effects` from the call's declared effects.
    Ask(OwnerApprovalRequest),
    /// No opinion: the allow rules and the mode decide.
    Defer,
}

/// Application rules consulted after the hard blocks and deny rules and
/// before the allow rules and the mode (for example: a trusted scheduled
/// session needs no approval; a command needs one with these warnings).
#[async_trait]
pub trait ApprovalRules: Send + Sync {
    async fn judge(&self, ctx: &HookContext<'_>, call: &ToolCall, info: &ToolInfo) -> RuleDecision;
}

/// Approvals the owner already granted. An ask they cover is allowed.
#[async_trait]
pub trait ApprovalGrants: Send + Sync {
    async fn covers(&self, request: &OwnerApprovalRequest) -> bool;
}

/// Approval policy. Declares [`Safeguard::Approval`].
///
/// Evaluation order, first match wins:
/// 1. **Hard blocks** (`Deny`): catastrophic commands found by tokenizing
///    `access.commands` (see [`crate::catastrophic_reason`]). Cannot be
///    allowed by any rule.
/// 2. **Deny rules** (`Deny`): tool name in the deny list, or any command
///    segment starting with a denied prefix.
/// 3. **Application rules** ([`ApprovalPolicy::with_rules`]): `Allow`,
///    `Deny`, `Ask`, or `Defer` to the steps below.
/// 4. **Allow rules** (`Allow`): tool name in the allow list, or every
///    command segment of every declared command starts with an allowed
///    prefix.
/// 5. **Mode** (ask or `Allow`).
///
/// An ask that the [`ApprovalPolicy::with_grants`] grants cover is allowed;
/// otherwise it becomes an [`InterruptKind::OwnerApproval`] interruption
/// whose request carries the call's declared effects, so a denial binds the
/// rest of the run. After the owner declined something in the run, an
/// egress or destructive call whose effects the tool could not fully
/// declare is asked about even where it would be allowed: it might be the
/// declined action by another route.
///
/// Command prefixes are matched by tokens, not characters: `git push`
/// matches `git push origin` but not `git pushy`, and a prefix never
/// matches text inside quotes or in a later `&&` segment by accident.
#[derive(Clone, Default)]
pub struct ApprovalPolicy {
    mode: ApprovalMode,
    allow_tools: HashSet<String>,
    deny_tools: HashSet<String>,
    allow_commands: Vec<Vec<String>>,
    deny_commands: Vec<Vec<String>>,
    rules: Option<Arc<dyn ApprovalRules>>,
    grants: Option<Arc<dyn ApprovalGrants>>,
}

impl std::fmt::Debug for ApprovalPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApprovalPolicy")
            .field("mode", &self.mode)
            .field("allow_tools", &self.allow_tools)
            .field("deny_tools", &self.deny_tools)
            .field("allow_commands", &self.allow_commands)
            .field("deny_commands", &self.deny_commands)
            .field("rules", &self.rules.is_some())
            .field("grants", &self.grants.is_some())
            .finish()
    }
}

/// Where the policy stands after the rules that need no context.
enum Step {
    Decided(ToolDecision),
    Ask(OwnerApprovalRequest),
}

impl ApprovalPolicy {
    pub fn new(mode: ApprovalMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    pub fn allow_tool(mut self, name: impl Into<String>) -> Self {
        self.allow_tools.insert(name.into());
        self
    }

    pub fn deny_tool(mut self, name: impl Into<String>) -> Self {
        self.deny_tools.insert(name.into());
        self
    }

    /// Allow commands starting with these tokens (e.g. `"git status"`).
    pub fn allow_command(mut self, prefix: &str) -> Self {
        if let Some(tokens) = command_segments(prefix).into_iter().next() {
            self.allow_commands.push(tokens);
        }
        self
    }

    /// Deny commands starting with these tokens (e.g. `"git push"`).
    pub fn deny_command(mut self, prefix: &str) -> Self {
        if let Some(tokens) = command_segments(prefix).into_iter().next() {
            self.deny_commands.push(tokens);
        }
        self
    }

    /// Consult `rules` after the hard blocks and deny rules.
    pub fn with_rules(mut self, rules: Arc<dyn ApprovalRules>) -> Self {
        self.rules = Some(rules);
        self
    }

    /// Allow asks that `grants` cover.
    pub fn with_grants(mut self, grants: Arc<dyn ApprovalGrants>) -> Self {
        self.grants = Some(grants);
        self
    }

    pub fn mode(&self) -> ApprovalMode {
        self.mode
    }

    /// The decision for one call without a hook context, application rules
    /// or grants (handy in tests and for previewing a policy). Mode asks are
    /// typed owner approvals with the call's declared effects.
    pub fn decide(&self, tool_name: &str, info: &ToolInfo) -> ToolDecision {
        if let Some(decision) = self.blocked(tool_name, info) {
            return decision;
        }
        match self.allowed_or_mode(tool_name, &Value::Null, info) {
            Step::Decided(decision) => decision,
            Step::Ask(request) => interrupt(request, info),
        }
    }

    /// Steps 1 and 2: hard blocks and deny rules.
    fn blocked(&self, tool_name: &str, info: &ToolInfo) -> Option<ToolDecision> {
        for command in &info.access.commands {
            if let Some(reason) = catastrophic_reason(command) {
                return Some(ToolDecision::Deny {
                    reason: format!("blocked by policy: {reason} (command: `{command}`)"),
                });
            }
        }

        if self.deny_tools.contains(tool_name) {
            return Some(ToolDecision::Deny {
                reason: format!("blocked by policy: the tool `{tool_name}` is denied"),
            });
        }
        for segment in self.segments(info) {
            if let Some(prefix) = self
                .deny_commands
                .iter()
                .find(|p| segment_matches_prefix(&segment, p))
            {
                return Some(ToolDecision::Deny {
                    reason: format!(
                        "blocked by policy: commands starting with `{}` are denied",
                        prefix.join(" ")
                    ),
                });
            }
        }
        None
    }

    fn segments(&self, info: &ToolInfo) -> Vec<Vec<String>> {
        info.access
            .commands
            .iter()
            .flat_map(|c| command_segments(c))
            .collect()
    }

    /// Steps 4 and 5: allow rules, then the mode.
    fn allowed_or_mode(&self, tool_name: &str, arguments: &Value, info: &ToolInfo) -> Step {
        if self.allow_tools.contains(tool_name) {
            return Step::Decided(ToolDecision::Allow);
        }
        let segments = self.segments(info);
        if !segments.is_empty()
            && segments.iter().all(|segment| {
                self.allow_commands
                    .iter()
                    .any(|p| segment_matches_prefix(segment, p))
            })
        {
            return Step::Decided(ToolDecision::Allow);
        }

        let ann = info.annotations;
        let why = match self.mode {
            ApprovalMode::Never => None,
            ApprovalMode::DestructiveOnly => ann
                .destructive
                .then(|| format!("`{tool_name}` is marked destructive")),
            ApprovalMode::AllWrites => {
                (!ann.read_only).then(|| format!("`{tool_name}` is not read-only"))
            }
            ApprovalMode::Always => Some(format!("every call needs approval (`{tool_name}`)")),
        };
        match why {
            Some(why) => Step::Ask(call_request(tool_name, arguments, info, why)),
            None => Step::Decided(ToolDecision::Allow),
        }
    }

    /// An ask, unless the owner already declined this exact request or a
    /// grant covers it.
    async fn ask(
        &self,
        ctx: &HookContext<'_>,
        request: OwnerApprovalRequest,
        info: &ToolInfo,
    ) -> ToolDecision {
        let request = with_effects(request, info);
        if let Some(refusal) = declined_subject(ctx, &request) {
            return refusal;
        }
        if let Some(grants) = &self.grants {
            if grants.covers(&request).await {
                return ToolDecision::Allow;
            }
        }
        interrupt(request, info)
    }
}

/// The request for a call the policy itself asks about: the tool name and
/// arguments, High risk, or Critical when the tool is destructive.
fn call_request(
    tool_name: &str,
    arguments: &Value,
    info: &ToolInfo,
    why: String,
) -> OwnerApprovalRequest {
    let subject = if arguments.is_null() {
        tool_name.to_string()
    } else {
        format!("{tool_name} {arguments}")
    };
    OwnerApprovalRequest {
        effects: info.effects.keys.clone(),
        summary: format!("Run {subject}: {why}"),
        subject,
        risk: if info.annotations.destructive {
            ApprovalRisk::Critical
        } else {
            ApprovalRisk::High
        },
        warnings: Vec::new(),
    }
}

/// Requests carry the call's declared effects, whoever wrote them.
fn with_effects(mut request: OwnerApprovalRequest, info: &ToolInfo) -> OwnerApprovalRequest {
    request.effects = info.effects.keys.clone();
    request
}

fn interrupt(request: OwnerApprovalRequest, info: &ToolInfo) -> ToolDecision {
    let request = with_effects(request, info);
    ToolDecision::Interrupt {
        reason: format!("approval required: {}", request.summary),
        kind: InterruptKind::OwnerApproval(request),
    }
}

/// The refusal for a request whose subject the owner declined in this run
/// (recorded by the kernel as an [`EffectClass::Call`] key when the declined
/// request declared no effects), so the identical request gets no new card.
/// A blank subject names no call and matches nothing.
fn declined_subject(ctx: &HookContext<'_>, request: &OwnerApprovalRequest) -> Option<ToolDecision> {
    if request.subject.trim().is_empty() {
        return None;
    }
    let declined = ctx
        .declined
        .iter()
        .find(|key| key.class == EffectClass::Call && key.target == request.subject)?;
    Some(ToolDecision::Deny {
        reason: format!(
            "{OWNER_DECLINED} The owner declined {} {} earlier in this run. Do not attempt it \
             another way; tell the owner it was not done.",
            declined.class, declined.target
        ),
    })
}

/// After a decline in this run, an egress or destructive call whose effects
/// are not fully declared might be the declined action by another route.
fn needs_reask(ctx: &HookContext<'_>, info: &ToolInfo) -> bool {
    !ctx.declined.is_empty()
        && !info.effects.complete
        && (info.annotations.egress
            || info.annotations.destructive
            || info
                .effects
                .keys
                .iter()
                .any(|key| matches!(key.class, EffectClass::Egress | EffectClass::Delete)))
}

#[async_trait]
impl Hook for ApprovalPolicy {
    fn safeguards(&self) -> &'static [Safeguard] {
        &[Safeguard::Approval]
    }

    async fn before_tool(
        &self,
        ctx: &HookContext<'_>,
        call: &ToolCall,
        info: &ToolInfo,
    ) -> ToolDecision {
        if let Some(decision) = self.blocked(&call.name, info) {
            return decision;
        }
        let rule = match &self.rules {
            Some(rules) => rules.judge(ctx, call, info).await,
            None => RuleDecision::Defer,
        };
        let step = match rule {
            RuleDecision::Allow => Step::Decided(ToolDecision::Allow),
            RuleDecision::Deny { reason } => return ToolDecision::Deny { reason },
            RuleDecision::Ask(request) => Step::Ask(request),
            RuleDecision::Defer => self.allowed_or_mode(&call.name, &call.arguments, info),
        };
        let decision = match step {
            Step::Decided(decision) => decision,
            Step::Ask(request) => self.ask(ctx, request, info).await,
        };
        if decision == ToolDecision::Allow && needs_reask(ctx, info) {
            // Never silently allow an undeterminable call after a decline;
            // a grant does not cover it either.
            let why = "the owner declined an action earlier in this run, and what this call \
                       does cannot be fully determined"
                .to_string();
            let request = with_effects(call_request(&call.name, &call.arguments, info, why), info);
            if let Some(refusal) = declined_subject(ctx, &request) {
                return refusal;
            }
            return interrupt(request, info);
        }
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::{AccessManifest, ToolAnnotations};

    fn info(annotations: ToolAnnotations, commands: &[&str]) -> ToolInfo {
        ToolInfo {
            annotations,
            access: AccessManifest {
                commands: commands.iter().map(|c| c.to_string()).collect(),
                ..AccessManifest::default()
            },
            effects: aidaemon_core::CallEffects::unknown(),
        }
    }

    fn interrupts(d: &ToolDecision) -> bool {
        matches!(d, ToolDecision::Interrupt { .. })
    }

    #[test]
    fn modes_follow_annotations() {
        let ro = info(ToolAnnotations::READ_ONLY, &[]);
        let w = info(ToolAnnotations::WRITE, &[]);
        let d = info(ToolAnnotations::DESTRUCTIVE, &[]);

        let never = ApprovalPolicy::new(ApprovalMode::Never);
        assert_eq!(never.decide("t", &d), ToolDecision::Allow);

        let destructive = ApprovalPolicy::new(ApprovalMode::DestructiveOnly);
        assert_eq!(destructive.decide("t", &w), ToolDecision::Allow);
        assert!(interrupts(&destructive.decide("t", &d)));

        let writes = ApprovalPolicy::new(ApprovalMode::AllWrites);
        assert_eq!(writes.decide("t", &ro), ToolDecision::Allow);
        assert!(interrupts(&writes.decide("t", &w)));

        let always = ApprovalPolicy::new(ApprovalMode::Always);
        assert!(interrupts(&always.decide("t", &ro)));
    }

    #[test]
    fn deny_wins_over_allow() {
        let policy = ApprovalPolicy::new(ApprovalMode::Always)
            .allow_tool("shell")
            .deny_tool("shell");
        assert!(matches!(
            policy.decide("shell", &info(ToolAnnotations::READ_ONLY, &[])),
            ToolDecision::Deny { .. }
        ));

        let policy = ApprovalPolicy::new(ApprovalMode::Always)
            .allow_command("git")
            .deny_command("git push");
        let shell = |cmd: &str| info(ToolAnnotations::DESTRUCTIVE, &[cmd]);
        assert!(matches!(
            policy.decide("shell", &shell("git push origin main")),
            ToolDecision::Deny { .. }
        ));
        assert_eq!(
            policy.decide("shell", &shell("git status")),
            ToolDecision::Allow
        );
    }

    #[test]
    fn command_allow_needs_every_segment() {
        let policy = ApprovalPolicy::new(ApprovalMode::DestructiveOnly).allow_command("git status");
        let shell = |cmd: &str| info(ToolAnnotations::DESTRUCTIVE, &[cmd]);
        assert_eq!(
            policy.decide("shell", &shell("git status --short")),
            ToolDecision::Allow
        );
        assert!(interrupts(&policy.decide(
            "shell",
            &shell("git status && curl https://example.invalid")
        )));
        assert!(interrupts(&policy.decide("shell", &shell("git statusx"))));
    }

    #[test]
    fn hard_blocks_cannot_be_allowed() {
        let policy = ApprovalPolicy::new(ApprovalMode::Never)
            .allow_tool("shell")
            .allow_command("rm");
        let decision = policy.decide("shell", &info(ToolAnnotations::WRITE, &["rm -rf ~"]));
        match decision {
            ToolDecision::Deny { reason } => assert!(reason.contains("never allowed"), "{reason}"),
            other => panic!("expected deny, got {other:?}"),
        }
        // The same words quoted as data are not a command.
        assert_eq!(
            policy.decide("shell", &info(ToolAnnotations::WRITE, &["echo 'rm -rf ~'"])),
            ToolDecision::Allow
        );
    }
}

#[cfg(test)]
mod rules_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::grants::SessionGrants;
    use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
    use aidaemon_core::{
        is_untrusted_egress, AccessManifest, Agent, ApprovalRisk, CallEffects, EffectClass,
        EffectKey, FnTool, OwnerApprovalRequest, Resolution, RunId, RunStatus, SourceTrust,
        ToolAnnotations, ToolOutput, UntrustedHook, UntrustedMode, Usage,
    };
    use serde_json::{json, Value};

    const SEND: ToolAnnotations = ToolAnnotations {
        egress: true,
        ..ToolAnnotations::WRITE
    };

    fn hook_ctx(declined: &[EffectKey]) -> HookContext<'_> {
        HookContext {
            run_id: RunId::new(),
            agent_id: "a",
            session_id: "telegram:synthetic-user-1",
            turn: 1,
            messages: &[],
            receipts: &[],
            usage: Usage::default(),
            run_usage: Usage::default(),
            untrusted_seen: false,
            declined,
        }
    }

    fn egress_key() -> EffectKey {
        EffectKey::egress("https", "h.example.invalid", None)
    }

    fn send_info(effects: CallEffects) -> ToolInfo {
        ToolInfo {
            annotations: SEND,
            access: AccessManifest::default(),
            effects,
        }
    }

    fn complete(keys: Vec<EffectKey>) -> CallEffects {
        CallEffects {
            keys,
            complete: true,
        }
    }

    fn post() -> ToolCall {
        ToolCall::new(
            "c1",
            "http_request",
            json!({"method": "POST", "url": "https://h.example.invalid/report"}),
        )
    }

    fn request() -> OwnerApprovalRequest {
        OwnerApprovalRequest {
            // The policy replaces these with the call's declared effects.
            effects: Vec::new(),
            summary: "Post a report for Alice Rivera to h.example.invalid".to_string(),
            subject: "POST https://h.example.invalid/report".to_string(),
            risk: ApprovalRisk::High,
            warnings: Vec::new(),
        }
    }

    /// A rule with a fixed answer that counts how often it was asked.
    struct Fixed {
        decision: RuleDecision,
        asked: AtomicUsize,
    }

    impl Fixed {
        fn new(decision: RuleDecision) -> Arc<Self> {
            Arc::new(Self {
                decision,
                asked: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl ApprovalRules for Fixed {
        async fn judge(
            &self,
            _ctx: &HookContext<'_>,
            _call: &ToolCall,
            _info: &ToolInfo,
        ) -> RuleDecision {
            self.asked.fetch_add(1, Ordering::SeqCst);
            self.decision.clone()
        }
    }

    fn owner_request(decision: &ToolDecision) -> &OwnerApprovalRequest {
        match decision {
            ToolDecision::Interrupt {
                kind: InterruptKind::OwnerApproval(req),
                ..
            } => req,
            other => panic!("expected an owner approval, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rules_ask_is_typed_owner_approval() {
        let rules = Fixed::new(RuleDecision::Ask(request()));
        let policy = ApprovalPolicy::new(ApprovalMode::Never).with_rules(rules.clone());
        let info = send_info(complete(vec![egress_key()]));
        let decision = policy.before_tool(&hook_ctx(&[]), &post(), &info).await;

        let req = owner_request(&decision);
        assert_eq!(req.effects, vec![egress_key()]);
        assert_eq!(req.subject, "POST https://h.example.invalid/report");
        let ToolDecision::Interrupt { reason, .. } = &decision else {
            unreachable!()
        };
        assert!(reason.contains("Alice Rivera"), "{reason}");
        assert_eq!(rules.asked.load(Ordering::SeqCst), 1);

        // Mode asks are typed too: High, or Critical when destructive.
        let always = ApprovalPolicy::new(ApprovalMode::Always);
        let write = always.before_tool(&hook_ctx(&[]), &post(), &info).await;
        let req = owner_request(&write);
        assert_eq!(req.risk, ApprovalRisk::High);
        assert_eq!(req.effects, vec![egress_key()]);
        assert!(req.subject.contains("http_request"), "{}", req.subject);
        assert!(req.subject.contains("h.example.invalid"), "{}", req.subject);
        let destructive = ToolInfo {
            annotations: ToolAnnotations::DESTRUCTIVE,
            ..info.clone()
        };
        let delete = always
            .before_tool(&hook_ctx(&[]), &post(), &destructive)
            .await;
        assert_eq!(owner_request(&delete).risk, ApprovalRisk::Critical);
    }

    #[tokio::test]
    async fn rules_allow_deny_and_defer() {
        let info = send_info(complete(vec![egress_key()]));
        let ask_mode = ApprovalPolicy::new(ApprovalMode::Always);

        let allow = ask_mode.clone().with_rules(Fixed::new(RuleDecision::Allow));
        assert_eq!(
            allow.before_tool(&hook_ctx(&[]), &post(), &info).await,
            ToolDecision::Allow
        );

        let deny =
            ApprovalPolicy::new(ApprovalMode::Never).with_rules(Fixed::new(RuleDecision::Deny {
                reason: "not for Acme Corp".to_string(),
            }));
        assert_eq!(
            deny.before_tool(&hook_ctx(&[]), &post(), &info).await,
            ToolDecision::Deny {
                reason: "not for Acme Corp".to_string()
            }
        );

        // Defer falls through to the allow rules and the mode.
        let defer = ask_mode.clone().with_rules(Fixed::new(RuleDecision::Defer));
        owner_request(&defer.before_tool(&hook_ctx(&[]), &post(), &info).await);
        let defer_allowed = defer.allow_tool("http_request");
        assert_eq!(
            defer_allowed
                .before_tool(&hook_ctx(&[]), &post(), &info)
                .await,
            ToolDecision::Allow
        );
    }

    #[tokio::test]
    async fn granted_ask_is_allowed() {
        let info = send_info(complete(vec![egress_key()]));
        let grants = Arc::new(SessionGrants::new());
        let policy = ApprovalPolicy::new(ApprovalMode::Never)
            .with_rules(Fixed::new(RuleDecision::Ask(request())))
            .with_grants(grants.clone());

        let first = policy.before_tool(&hook_ctx(&[]), &post(), &info).await;
        let asked = owner_request(&first).clone();
        grants.grant(&asked);
        assert_eq!(
            policy.before_tool(&hook_ctx(&[]), &post(), &info).await,
            ToolDecision::Allow
        );

        // A grant is for that subject and those effects only.
        let other = send_info(complete(vec![EffectKey::egress(
            "https",
            "other.example.invalid",
            None,
        )]));
        owner_request(&policy.before_tool(&hook_ctx(&[]), &post(), &other).await);

        // Mode asks are covered by grants too.
        let mode = ApprovalPolicy::new(ApprovalMode::Always).with_grants(grants.clone());
        let ask = mode.before_tool(&hook_ctx(&[]), &post(), &info).await;
        grants.grant(owner_request(&ask));
        assert_eq!(
            mode.before_tool(&hook_ctx(&[]), &post(), &info).await,
            ToolDecision::Allow
        );
    }

    #[tokio::test]
    async fn incomplete_effects_after_decline_reask() {
        let declined = [egress_key()];
        let opaque = send_info(CallEffects::unknown());
        let never = ApprovalPolicy::new(ApprovalMode::Never);

        // Without a decline, an opaque egress call follows the mode.
        assert_eq!(
            never.before_tool(&hook_ctx(&[]), &post(), &opaque).await,
            ToolDecision::Allow
        );
        // After one, it is asked about, even where it would be allowed.
        owner_request(
            &never
                .before_tool(&hook_ctx(&declined), &post(), &opaque)
                .await,
        );
        let allowed = never
            .clone()
            .allow_tool("http_request")
            .with_rules(Fixed::new(RuleDecision::Allow));
        owner_request(
            &allowed
                .before_tool(&hook_ctx(&declined), &post(), &opaque)
                .await,
        );

        // Complete effects are judged by the kernel's decline check, not
        // re-asked here.
        let known = send_info(complete(vec![EffectKey::egress(
            "https",
            "other.example.invalid",
            None,
        )]));
        assert_eq!(
            never
                .before_tool(&hook_ctx(&declined), &post(), &known)
                .await,
            ToolDecision::Allow
        );
        // A local, non-destructive write is not egress.
        let local = ToolInfo {
            annotations: ToolAnnotations {
                destructive: false,
                ..ToolAnnotations::WRITE
            },
            access: AccessManifest::default(),
            effects: CallEffects::unknown(),
        };
        assert_eq!(
            never
                .before_tool(&hook_ctx(&declined), &post(), &local)
                .await,
            ToolDecision::Allow
        );
    }

    #[tokio::test]
    async fn catastrophic_still_denied_before_rules() {
        let rules = Fixed::new(RuleDecision::Allow);
        let policy = ApprovalPolicy::new(ApprovalMode::Never)
            .allow_tool("shell")
            .deny_command("git push")
            .with_rules(rules.clone());
        let shell = |cmd: &str| ToolInfo {
            annotations: ToolAnnotations::DESTRUCTIVE,
            access: AccessManifest {
                commands: vec![cmd.to_string()],
                ..AccessManifest::default()
            },
            effects: CallEffects::unknown(),
        };
        let run = ToolCall::new("c1", "shell", json!({}));
        let blocked = policy
            .before_tool(&hook_ctx(&[]), &run, &shell("rm -rf ~"))
            .await;
        assert!(
            matches!(&blocked, ToolDecision::Deny { reason } if reason.contains("never allowed")),
            "{blocked:?}"
        );
        let denied = policy
            .before_tool(&hook_ctx(&[]), &run, &shell("git push origin main"))
            .await;
        assert!(matches!(denied, ToolDecision::Deny { .. }), "{denied:?}");
        assert_eq!(rules.asked.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn existing_mode_tests_still_pass() {
        // `decide` keeps its signature and rules for callers without a hook
        // context, and its asks are typed owner approvals.
        let policy = ApprovalPolicy::new(ApprovalMode::DestructiveOnly);
        let d = ToolInfo {
            annotations: ToolAnnotations::DESTRUCTIVE,
            access: AccessManifest::default(),
            effects: CallEffects::unknown(),
        };
        let decision = policy.decide("delete_file", &d);
        assert_eq!(owner_request(&decision).risk, ApprovalRisk::Critical);
        let ToolDecision::Interrupt { reason, .. } = &decision else {
            unreachable!()
        };
        assert!(reason.contains("destructive"), "{reason}");
        assert_eq!(policy.safeguards(), &[Safeguard::Approval]);
    }

    #[tokio::test]
    async fn approval_runs_after_untrusted_verifier() {
        fn mark(tool: &str, content: &str) -> String {
            format!("<untrusted tool=\"{tool}\">{content}</untrusted>")
        }
        let sent = Arc::new(AtomicUsize::new(0));
        let counter = sent.clone();
        let agent = Agent::builder("a")
            .allow_unguarded()
            .model("m")
            .provider(ScriptedProvider::new([
                tool_calls([call("c1", "fetch", json!({}))]),
                tool_calls([call("c2", "send", json!({}))]),
                text("done"),
            ]))
            .tool(FnTool::new(
                spec("fetch", "fetches", &[]),
                ToolAnnotations::READ_ONLY,
                |_: Value| std::future::ready(Ok(ToolOutput::text("page body"))),
            ))
            .tool(
                FnTool::new(
                    spec("send", "sends", &[]),
                    ToolAnnotations {
                        egress: true,
                        ..ToolAnnotations::DESTRUCTIVE
                    },
                    move |_: Value| {
                        counter.fetch_add(1, Ordering::SeqCst);
                        std::future::ready(Ok(ToolOutput::text("sent")))
                    },
                )
                .with_trust(SourceTrust::Trusted),
            )
            .hook(UntrustedHook::new(UntrustedMode::Enforce, mark))
            .hook(ApprovalPolicy::new(ApprovalMode::DestructiveOnly))
            .build()
            .unwrap();

        let out = agent.run("s", "go").await.unwrap();
        let RunStatus::Interrupted { interruption } = &out.status else {
            panic!("{:?}", out.status)
        };
        assert!(is_untrusted_egress(interruption), "{interruption:?}");

        let out = agent.resume(out.state, Resolution::Approve).await.unwrap();
        let RunStatus::Interrupted { interruption } = &out.status else {
            panic!("{:?}", out.status)
        };
        assert!(
            matches!(interruption.kind, InterruptKind::OwnerApproval(_)),
            "{interruption:?}"
        );
        assert_eq!(sent.load(Ordering::SeqCst), 0);

        let out = agent.resume(out.state, Resolution::Approve).await.unwrap();
        assert!(out.is_completed(), "{:?}", out.status);
        assert_eq!(sent.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn mode_ask_summary_names_the_tool() {
        let policy = ApprovalPolicy::new(ApprovalMode::Always);
        let info = send_info(complete(vec![EffectKey::named(
            EffectClass::Process,
            "git",
        )]));
        let decision = policy.decide("shell", &info);
        assert!(owner_request(&decision).summary.contains("shell"));
    }

    /// An opaque destructive tool (no declared effects), asked about by the
    /// mode.
    fn opaque_agent(replies: Vec<aidaemon_core::ModelResponse>) -> (Agent, Arc<AtomicUsize>) {
        let ran = Arc::new(AtomicUsize::new(0));
        let counter = ran.clone();
        let agent = Agent::builder("a")
            .allow_unguarded()
            .model("m")
            .provider(ScriptedProvider::new(replies))
            .tool(
                FnTool::new(
                    spec("terminal", "runs a command", &[]),
                    ToolAnnotations::DESTRUCTIVE,
                    move |_: Value| {
                        counter.fetch_add(1, Ordering::SeqCst);
                        std::future::ready(Ok(ToolOutput::text("ran")))
                    },
                )
                .with_trust(SourceTrust::Trusted),
            )
            .hook(ApprovalPolicy::new(ApprovalMode::DestructiveOnly))
            .build()
            .unwrap();
        (agent, ran)
    }

    /// Denies every interruption and counts the cards.
    #[derive(Default)]
    struct CountingDeny(AtomicUsize);

    #[async_trait]
    impl aidaemon_core::InterruptResolver for CountingDeny {
        async fn resolve(
            &self,
            _: &aidaemon_core::RunState,
            _: &aidaemon_core::Interruption,
        ) -> Option<Resolution> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Some(Resolution::Deny {
                reason: "Jordan Lee said no".to_string(),
            })
        }
    }

    fn run_script(script: &str) -> Value {
        json!({"command": format!("python3 /tmp/aidaemon-test/{script}")})
    }

    async fn run_denying(agent: &Agent) -> (aidaemon_core::RunOutcome, usize) {
        let resolver = CountingDeny::default();
        let out = agent
            .run_resolving(
                aidaemon_core::RunState::new("a", "s", "go"),
                aidaemon_core::CancellationToken::new(),
                &resolver,
            )
            .await
            .unwrap();
        (out, resolver.0.load(Ordering::SeqCst))
    }

    #[tokio::test]
    async fn declined_request_without_effects_blocks_identical_retry() {
        let (agent, ran) = opaque_agent(vec![
            tool_calls([call("c1", "terminal", run_script("run.py"))]),
            tool_calls([call("c2", "terminal", run_script("run.py"))]),
            text("done"),
        ]);
        let (out, cards) = run_denying(&agent).await;
        assert!(out.is_completed(), "{:?}", out.status);
        assert_eq!(cards, 1, "no second card");
        let retry = out.receipts()[1].reason.clone().unwrap();
        assert!(
            retry.starts_with("[OWNER DECLINED] The owner declined call "),
            "{retry}"
        );
        assert!(retry.contains("tell the owner it was not done"), "{retry}");
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn declined_request_without_effects_still_asks_other_subject() {
        let (agent, ran) = opaque_agent(vec![
            tool_calls([call("c1", "terminal", run_script("run.py"))]),
            tool_calls([call("c2", "terminal", run_script("other.py"))]),
            text("done"),
        ]);
        let (_, cards) = run_denying(&agent).await;
        assert_eq!(cards, 2);
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn blank_subject_is_never_a_declined_subject() {
        // A blank subject names no call, so a blank `Call` key (from a run
        // saved before the kernel stopped recording them) refuses nothing.
        let declined = [EffectKey::named(EffectClass::Call, "   ")];
        let mut blank = request();
        blank.subject = "   ".to_string();
        let rules = Fixed::new(RuleDecision::Ask(blank));
        let policy = ApprovalPolicy::new(ApprovalMode::Never).with_rules(rules);
        let info = send_info(complete(Vec::new()));
        let decision = policy
            .before_tool(&hook_ctx(&declined), &post(), &info)
            .await;
        assert!(
            matches!(decision, ToolDecision::Interrupt { .. }),
            "{decision:?}"
        );
    }
}
