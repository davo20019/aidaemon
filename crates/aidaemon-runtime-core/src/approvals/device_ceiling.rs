//! The device ceiling as a kernel hook: in a run capped at Medium (a
//! message from an e-ink device started or steered it, see
//! [`crate::device_cap`]), a call above Medium waits for the owner's answer
//! on a card only the phone or the computer can give, once.
//!
//! "Above Medium" is judged from the call's typed facts only, never from
//! its tool's name or its wording:
//!
//! - it sends data off the machine: the `egress` annotation, an egress
//!   effect, a declared host, or anything a read carries out (`Outbound`);
//! - it destroys data or cannot be undone: the `destructive` annotation or
//!   a delete effect, or it changes configuration or uses a credential;
//! - its own tool rated it High or above when its approval rule asked
//!   (whether or not a grant then covered the ask);
//! - its tool cannot say what it does (effects neither complete nor
//!   understood): a delegated coding agent or an opaque script may do any
//!   of the above.
//!
//! The hook sits after `ApprovalPolicy` in the turn's chain
//! (`engine::turn_engine::turn_hooks::with_device_ceiling`). The kernel asks every `before_tool` hook in
//! order and runs a call only when all of them allow it: an allow by a
//! permission mode, an allow rule or a session grant in `ApprovalPolicy`
//! is only that hook's answer, so it never stands in for this one. A hook
//! the owner answered is waived for that exact call only, and a rewrite
//! restarts every judgment.
//!
//! The card carries risk High (Critical for a destructive call), which the
//! client gateway lets no e-ink device answer, and the resolver offers it
//! only Allow once and Deny and records no lasting grant
//! ([`card_is_capped`]). A High card some other rule raised for the same
//! call in a capped run is treated the same way, and its Allow is a pass
//! for that exact call ([`call_key`]), so the owner answers once.

use aidaemon_core as sdk;
use async_trait::async_trait;

use crate::device_cap::{DeviceCap, above_medium as risk_above_medium, risk_rank};

/// The key the cap's records use for one call: its id, tool and exact
/// arguments.
pub fn call_key(call: &sdk::ToolCall) -> String {
    format!("{}\u{0}{} {}", call.id, call.name, call.arguments)
}

/// Why a call is above the Medium ceiling, from its typed facts and the
/// rating its own tool gave it (`rated`); `None` for a call at or below it
/// (a read that sends nothing out, a local change its tool fully declares).
pub fn above_medium(
    info: &sdk::ToolInfo,
    rated: Option<sdk::ApprovalRisk>,
) -> Option<&'static str> {
    let annotations = info.annotations;
    let keys = &info.effects.keys;
    let has = |class: sdk::EffectClass| keys.iter().any(|key| key.class == class);
    if annotations.egress
        || has(sdk::EffectClass::Egress)
        || !info.access.hosts.is_empty()
        || info.outbound != sdk::Outbound::Nothing
    {
        return Some("sends data off this computer");
    }
    if annotations.destructive || has(sdk::EffectClass::Delete) {
        return Some("deletes or overwrites data, or cannot be undone");
    }
    if has(sdk::EffectClass::Configuration) || has(sdk::EffectClass::Credential) {
        return Some("changes settings or uses a stored credential");
    }
    if rated.is_some_and(risk_above_medium) {
        return Some("is rated high risk by its own tool");
    }
    if !info.effects.complete && !info.effects.understood {
        return Some("may do things its tool cannot fully declare");
    }
    None
}

/// What the owner's card says about why it is asked.
const CAPPED_WARNING: &str = "Asked from a device that may not allow this (an e-ink reader): \
     answer it on your phone or computer. Allowing it covers this one call only.";

/// The kernel hook that holds a capped run's calls above Medium for the
/// owner. Declares [`sdk::Safeguard::Approval`].
pub struct DeviceCeilingHook {
    cap: DeviceCap,
}

impl DeviceCeilingHook {
    pub fn new(cap: DeviceCap) -> Self {
        Self { cap }
    }
}

#[async_trait]
impl sdk::Hook for DeviceCeilingHook {
    fn name(&self) -> &'static str {
        "device_ceiling"
    }

    fn safeguards(&self) -> &'static [sdk::Safeguard] {
        &[sdk::Safeguard::Approval]
    }

    async fn before_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        info: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        if !self.cap.is_capped() {
            return sdk::ToolDecision::Allow;
        }
        let key = call_key(call);
        let Some(why) = above_medium(info, self.cap.rating(ctx.run_id, &key)) else {
            return sdk::ToolDecision::Allow;
        };
        // The owner already allowed this exact call on a card the capped
        // device could not answer (another rule's High card).
        if self.cap.take_pass(ctx.run_id, &key) {
            return sdk::ToolDecision::Allow;
        }
        let mut request = sdk::OwnerApprovalRequest::for_call(call, info, "");
        // `for_call` asks at High, Critical for a destructive call: never
        // lower, so no Medium-ceilinged device can answer it.
        if !risk_above_medium(request.risk) {
            request.risk = sdk::ApprovalRisk::High;
        }
        request.warnings.push(CAPPED_WARNING.to_string());
        // A request that declares no effects is declined by its subject: the
        // identical one is not put to the owner again.
        if request.effects.is_empty()
            && ctx
                .declined
                .iter()
                .any(|key| key.class == sdk::EffectClass::Call && key.target == request.subject)
        {
            return sdk::ToolDecision::Deny {
                reason: format!(
                    "{} The owner declined this call earlier in this run. Do not attempt it \
                     another way; tell the owner it was not done.",
                    sdk::OWNER_DECLINED
                ),
            };
        }
        sdk::ToolDecision::Interrupt {
            reason: format!(
                "approval required: {}: the run is capped at Medium by the device that sent its \
                 request, and this call {why}",
                request.summary
            ),
            kind: sdk::InterruptKind::OwnerApproval(request),
        }
    }
}

/// Whether a card for `request` is a capped card: the run is capped and
/// the card is above Medium. Such a card offers only Allow once and Deny,
/// never records a lasting grant, and its Allow passes the cap for that
/// call.
pub fn card_is_capped(cap: &DeviceCap, request: &sdk::OwnerApprovalRequest) -> bool {
    cap.is_capped() && risk_rank(request.risk) > risk_rank(sdk::ApprovalRisk::Medium)
}

/// The card kind a capped card is sent as: a command card with only Allow
/// once and Deny. Kinds that already offer only a one-time answer keep
/// their own buttons.
pub fn once_only(
    kind: aidaemon_tools::semantics::ApprovalKind,
) -> aidaemon_tools::semantics::ApprovalKind {
    use aidaemon_tools::semantics::ApprovalKind;
    match kind {
        ApprovalKind::Command => ApprovalKind::CommandOnce,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::{
        AccessManifest, CallEffects, EffectClass, EffectKey, Hook, Outbound, RunId,
        ToolAnnotations, ToolCall, ToolDecision, ToolInfo,
    };
    use aidaemon_tools::semantics::ApprovalKind;

    fn info(annotations: ToolAnnotations, effects: CallEffects) -> ToolInfo {
        ToolInfo {
            annotations,
            access: AccessManifest::default(),
            effects,
            outbound: Outbound::Nothing,
        }
    }

    fn local_write() -> ToolInfo {
        info(
            ToolAnnotations::WRITE,
            CallEffects {
                keys: vec![
                    EffectKey::path(EffectClass::Write, "/tmp/aidaemon-test/notes.md").unwrap(),
                ],
                complete: true,
                understood: true,
            },
        )
    }

    fn read() -> ToolInfo {
        info(ToolAnnotations::READ_ONLY, CallEffects::none())
    }

    fn post() -> ToolInfo {
        let mut post = info(
            ToolAnnotations {
                egress: true,
                ..ToolAnnotations::WRITE
            },
            CallEffects {
                keys: vec![EffectKey::egress("https", "api.example.invalid", None)],
                complete: true,
                understood: true,
            },
        );
        post.access.hosts = vec!["api.example.invalid".to_string()];
        post
    }

    fn call(name: &str) -> ToolCall {
        ToolCall::new(
            format!("call-{name}"),
            name,
            serde_json::json!({"to": "Jordan Lee"}),
        )
    }

    fn ctx(run_id: RunId, declined: &[EffectKey]) -> sdk::HookContext<'_> {
        sdk::HookContext {
            run_id,
            agent_id: "aidaemon",
            session_id: "app:synthetic-session-1",
            turn: 1,
            messages: &[],
            receipts: &[],
            usage: Default::default(),
            run_usage: Default::default(),
            untrusted_seen: false,
            declined,
        }
    }

    fn asked(decision: &ToolDecision) -> Option<&sdk::OwnerApprovalRequest> {
        match decision {
            ToolDecision::Interrupt {
                kind: sdk::InterruptKind::OwnerApproval(request),
                ..
            } => Some(request),
            _ => None,
        }
    }

    #[test]
    fn above_medium_is_judged_from_typed_facts() {
        assert_eq!(above_medium(&read(), None), None);
        assert_eq!(above_medium(&local_write(), None), None);
        assert!(above_medium(&post(), None).is_some());
        // A read whose query leaves the machine.
        let search = ToolInfo {
            outbound: Outbound::Composed,
            ..read()
        };
        assert!(above_medium(&search, None).is_some());
        let delete = info(
            ToolAnnotations {
                destructive: true,
                egress: false,
                ..ToolAnnotations::WRITE
            },
            CallEffects {
                keys: vec![EffectKey::path(EffectClass::Delete, "/tmp/aidaemon-test/old").unwrap()],
                complete: true,
                understood: true,
            },
        );
        assert!(above_medium(&delete, None).is_some());
        // An opaque call (a delegated agent, a script) may do anything.
        assert!(
            above_medium(&info(ToolAnnotations::WRITE, CallEffects::unknown()), None).is_some()
        );
        assert!(
            above_medium(
                &info(ToolAnnotations::READ_ONLY, CallEffects::unknown()),
                None
            )
            .is_some()
        );
        // The tool's own rating counts; a Medium one does not.
        assert!(above_medium(&local_write(), Some(sdk::ApprovalRisk::High)).is_some());
        assert_eq!(
            above_medium(&local_write(), Some(sdk::ApprovalRisk::Medium)),
            None
        );
    }

    #[tokio::test]
    async fn an_uncapped_run_is_untouched() {
        let hook = DeviceCeilingHook::new(DeviceCap::new());
        let run = RunId::new();
        for info in [post(), read(), local_write()] {
            assert_eq!(
                hook.before_tool(&ctx(run, &[]), &call("send"), &info).await,
                ToolDecision::Allow
            );
        }
    }

    #[tokio::test]
    async fn a_capped_run_asks_on_a_high_card_for_calls_above_medium() {
        let cap = DeviceCap::new();
        cap.raise();
        let hook = DeviceCeilingHook::new(cap.clone());
        let run = RunId::new();
        // Reads and local medium work run as before.
        assert_eq!(
            hook.before_tool(&ctx(run, &[]), &call("read"), &read())
                .await,
            ToolDecision::Allow
        );
        assert_eq!(
            hook.before_tool(&ctx(run, &[]), &call("write"), &local_write())
                .await,
            ToolDecision::Allow
        );
        // Egress waits for the owner on a High card, offered once only.
        let decision = hook
            .before_tool(&ctx(run, &[]), &call("send"), &post())
            .await;
        let request = asked(&decision).expect("asked");
        assert!(risk_above_medium(request.risk));
        assert!(card_is_capped(&cap, request));
        assert!(matches!(
            once_only(ApprovalKind::Command),
            ApprovalKind::CommandOnce
        ));
        assert!(matches!(
            once_only(ApprovalKind::GoalConfirmation),
            ApprovalKind::GoalConfirmation
        ));
        // A destructive call is Critical.
        let delete = info(ToolAnnotations::DESTRUCTIVE, CallEffects::unknown());
        let decision = hook.before_tool(&ctx(run, &[]), &call("rm"), &delete).await;
        assert_eq!(asked(&decision).unwrap().risk, sdk::ApprovalRisk::Critical);
        // A local change its tool rated High asks too.
        let rated = call("write-high");
        cap.rate(run, &call_key(&rated), sdk::ApprovalRisk::High);
        assert!(
            asked(
                &hook
                    .before_tool(&ctx(run, &[]), &rated, &local_write())
                    .await
            )
            .is_some()
        );
    }

    #[tokio::test]
    async fn an_owner_allow_on_a_high_card_passes_that_call_once() {
        let cap = DeviceCap::new();
        cap.raise();
        let hook = DeviceCeilingHook::new(cap.clone());
        let run = RunId::new();
        let send = call("send");
        cap.allow_call(run, &call_key(&send));
        assert_eq!(
            hook.before_tool(&ctx(run, &[]), &send, &post()).await,
            ToolDecision::Allow
        );
        // Once: the same call again asks again, and so do other arguments.
        assert!(asked(&hook.before_tool(&ctx(run, &[]), &send, &post()).await).is_some());
        let mut other = call("send");
        other.arguments = serde_json::json!({"to": "Alice Rivera"});
        cap.allow_call(run, &call_key(&send));
        assert!(asked(&hook.before_tool(&ctx(run, &[]), &other, &post()).await).is_some());
    }

    #[tokio::test]
    async fn a_declined_effectless_call_is_refused_not_asked_again() {
        let cap = DeviceCap::new();
        cap.raise();
        let hook = DeviceCeilingHook::new(cap);
        let run = RunId::new();
        let opaque = info(ToolAnnotations::WRITE, CallEffects::unknown());
        let decision = hook.before_tool(&ctx(run, &[]), &call("x"), &opaque).await;
        let subject = asked(&decision).unwrap().subject.clone();
        let declined = [EffectKey::named(EffectClass::Call, &subject)];
        assert!(matches!(
            hook.before_tool(&ctx(run, &declined), &call("x"), &opaque)
                .await,
            ToolDecision::Deny { .. }
        ));
    }
}
