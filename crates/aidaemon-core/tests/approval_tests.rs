//! Typed call effects, owner-approval interrupts and the run's memory of
//! declined effects.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, ApprovalRisk, CallEffects, CancellationToken, EffectClass, EffectKey, FnTool, Hook,
    HookContext, InterruptKind, InterruptResolver, Interruption, OwnerApprovalRequest,
    ReceiptStatus, Resolution, RunState, SourceTrust, Tool, ToolAnnotations, ToolCall,
    ToolDecision, ToolInfo, ToolOutput,
};
use async_trait::async_trait;
use serde_json::{json, Value};

const SEND: ToolAnnotations = ToolAnnotations {
    egress: true,
    ..ToolAnnotations::WRITE
};

fn key(class: EffectClass, target: &str) -> EffectKey {
    EffectKey {
        class,
        target: target.to_string(),
    }
}

fn egress_key() -> EffectKey {
    key(EffectClass::Egress, "h.example.invalid:443")
}

fn owner_request(effects: Vec<EffectKey>) -> OwnerApprovalRequest {
    OwnerApprovalRequest {
        effects,
        summary: "Send a report to h.example.invalid for Alice Rivera".to_string(),
        subject: "POST https://h.example.invalid/report".to_string(),
        risk: ApprovalRisk::High,
        warnings: Vec::new(),
    }
}

/// Interrupts every `send` call with a fixed kind, and records what
/// `HookContext::declined` and `ToolInfo::effects` held when it was asked.
struct AskBeforeSend {
    kind: InterruptKind,
    seen_declined: Mutex<Vec<Vec<EffectKey>>>,
    seen_effects: Mutex<Vec<CallEffects>>,
}

impl AskBeforeSend {
    fn new(kind: InterruptKind) -> Arc<Self> {
        Arc::new(Self {
            kind,
            seen_declined: Mutex::default(),
            seen_effects: Mutex::default(),
        })
    }
}

#[async_trait]
impl Hook for AskBeforeSend {
    async fn before_tool(
        &self,
        ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        self.seen_declined
            .lock()
            .unwrap()
            .push(ctx.declined.to_vec());
        self.seen_effects.lock().unwrap().push(tool.effects.clone());
        if call.name == "send" {
            ToolDecision::Interrupt {
                reason: "the owner must confirm this send".to_string(),
                kind: self.kind.clone(),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

/// A send tool that declares its egress effect.
struct Send {
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for Send {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec("send", "sends", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        SEND
    }
    fn source_trust(&self, _: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }
    fn effects(&self, _args: &Value) -> CallEffects {
        CallEffects {
            keys: vec![egress_key()],
            complete: true,
        }
    }
    async fn call(
        &self,
        _args: Value,
        _ctx: &aidaemon_core::ToolContext,
    ) -> aidaemon_core::ToolResult {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(ToolOutput::text("sent"))
    }
}

struct Fixed(Resolution);

#[async_trait]
impl InterruptResolver for Fixed {
    async fn resolve(&self, _: &RunState, _: &Interruption) -> Option<Resolution> {
        Some(self.0.clone())
    }
}

fn deny() -> Fixed {
    Fixed(Resolution::Deny {
        reason: "Alice Rivera said no".to_string(),
    })
}

fn agent(
    hook: Arc<AskBeforeSend>,
    replies: Vec<aidaemon_core::ModelResponse>,
) -> (Agent, Arc<AtomicUsize>) {
    let sent = Arc::new(AtomicUsize::new(0));
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(replies))
        .tool(Send {
            count: sent.clone(),
        })
        .hook_arc(hook)
        .build()
        .unwrap();
    (agent, sent)
}

fn send_once() -> Vec<aidaemon_core::ModelResponse> {
    vec![tool_calls([call("c1", "send", json!({}))]), text("done")]
}

#[tokio::test]
async fn owner_approval_deny_records_declined_effects() {
    let hook = AskBeforeSend::new(InterruptKind::OwnerApproval(owner_request(vec![
        egress_key(),
    ])));
    let (a, sent) = agent(hook.clone(), send_once());
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &deny(),
        )
        .await
        .unwrap();

    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(out.state.declined, vec![egress_key()]);
    let receipt = &out.receipts()[0];
    assert_eq!(receipt.status, ReceiptStatus::Denied);
    let reason = receipt.reason.as_deref().unwrap();
    assert!(reason.starts_with("[OWNER DECLINED] "), "{reason}");
    assert!(reason.contains("Alice Rivera said no"), "{reason}");
    assert_eq!(sent.load(Ordering::SeqCst), 0);
    // The hook saw the tool's declared effects.
    assert_eq!(
        hook.seen_effects.lock().unwrap()[0],
        CallEffects {
            keys: vec![egress_key()],
            complete: true,
        }
    );
}

#[tokio::test]
async fn owner_approval_deny_dedups_and_hooks_see_declined() {
    let hook = AskBeforeSend::new(InterruptKind::OwnerApproval(owner_request(vec![
        egress_key(),
    ])));
    let (a, _sent) = agent(
        hook.clone(),
        vec![
            tool_calls([call("c1", "send", json!({}))]),
            tool_calls([call("c2", "send", json!({}))]),
            text("done"),
        ],
    );
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &deny(),
        )
        .await
        .unwrap();

    assert_eq!(out.state.declined, vec![egress_key()]);
    let seen = hook.seen_declined.lock().unwrap();
    assert!(seen[0].is_empty());
    assert_eq!(seen[1], vec![egress_key()]);
}

#[tokio::test]
async fn untrusted_deny_records_nothing() {
    let hook = AskBeforeSend::new(InterruptKind::Untrusted);
    let (a, _sent) = agent(hook, send_once());
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &deny(),
        )
        .await
        .unwrap();

    assert!(out.is_completed(), "{:?}", out.status);
    assert!(out.state.declined.is_empty());
    let receipt = &out.receipts()[0];
    assert_eq!(receipt.status, ReceiptStatus::Denied);
    assert_eq!(receipt.reason.as_deref(), Some("Alice Rivera said no"));
}

#[tokio::test]
async fn declined_survives_serde_and_segments() {
    let first = AskBeforeSend::new(InterruptKind::OwnerApproval(owner_request(vec![
        egress_key(),
    ])));
    let (a, _) = agent(first, send_once());
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &deny(),
        )
        .await
        .unwrap();
    assert_eq!(out.state.declined, vec![egress_key()]);

    let json = serde_json::to_string(&out.state).unwrap();
    let restored: RunState = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, out.state);
    assert_eq!(restored.declined, vec![egress_key()]);

    // A second segment, interrupted and settled again, keeps the memory and
    // adds to it.
    let other = key(EffectClass::Delete, "/tmp/aidaemon-test/report");
    let second = AskBeforeSend::new(InterruptKind::OwnerApproval(owner_request(vec![
        other.clone()
    ])));
    let (b, _) = agent(second.clone(), send_once());
    let mut state = restored;
    state.messages.push(aidaemon_core::Message::user("again"));
    let out = b
        .run_resolving(state, CancellationToken::new(), &deny())
        .await
        .unwrap();
    assert_eq!(out.state.declined, vec![egress_key(), other]);
    assert_eq!(second.seen_declined.lock().unwrap()[0], vec![egress_key()]);
}

#[test]
fn old_state_and_interruptions_read_without_the_new_fields() {
    let state = RunState::new("a", "s", "go");
    let mut value = serde_json::to_value(&state).unwrap();
    value.as_object_mut().unwrap().remove("declined");
    let restored: RunState = serde_json::from_value(value).unwrap();
    assert!(restored.declined.is_empty());

    let interruption: Interruption = serde_json::from_value(json!({
        "call": {"id": "c1", "name": "send", "arguments": {}},
        "reason": "[UNTRUSTED EGRESS] old",
    }))
    .unwrap();
    assert_eq!(interruption.kind, InterruptKind::Untrusted);
}

#[test]
fn a_new_run_starts_with_no_declines() {
    let mut state = RunState::new("a", "s", "go");
    state.declined.push(egress_key());
    let next = state.with_user_message("next request");
    assert!(next.declined.is_empty());
}

#[test]
fn effect_key_overlap_rules() {
    let a443 = key(EffectClass::Egress, "a:443");
    assert!(a443.overlaps(&key(EffectClass::Egress, "a:443")));
    assert!(!a443.overlaps(&key(EffectClass::Egress, "a:80")));

    let del = key(EffectClass::Delete, "/tmp/x");
    assert!(del.overlaps(&key(EffectClass::Delete, "/tmp/x/y")));
    assert!(key(EffectClass::Delete, "/tmp/x/y").overlaps(&del));
    assert!(del.overlaps(&key(EffectClass::Delete, "/tmp/x")));
    assert!(!del.overlaps(&key(EffectClass::Delete, "/tmp/xy")));
    assert!(del.overlaps(&key(EffectClass::Delete, "/tmp")));
    assert!(key(EffectClass::Delete, "/tmp").overlaps(&del));
    assert!(key(EffectClass::Write, "/").overlaps(&key(EffectClass::Write, "/etc/hosts")));
    assert!(key(EffectClass::Configuration, "/etc/x/")
        .overlaps(&key(EffectClass::Configuration, "/etc/x/y")));

    assert!(!key(EffectClass::Write, "/tmp/x").overlaps(&del));

    // Non-path classes match exactly; "below" means nothing for them.
    assert!(key(EffectClass::Process, "git").overlaps(&key(EffectClass::Process, "git")));
    assert!(!key(EffectClass::Process, "git").overlaps(&key(EffectClass::Process, "git/x")));
    assert!(
        !key(EffectClass::Credential, "acme").overlaps(&key(EffectClass::Credential, "acme/dev"))
    );
}

#[test]
fn call_effects_constructors() {
    assert_eq!(
        CallEffects::unknown(),
        CallEffects {
            keys: Vec::new(),
            complete: false,
        }
    );
    assert_eq!(
        CallEffects::none(),
        CallEffects {
            keys: Vec::new(),
            complete: true,
        }
    );
}

#[test]
fn default_effects_follow_read_only() {
    let reader = FnTool::new(
        spec("read", "reads", &[]),
        ToolAnnotations::READ_ONLY,
        |_: Value| std::future::ready(Ok(ToolOutput::text("x"))),
    );
    let writer = FnTool::new(
        spec("write", "writes", &[]),
        ToolAnnotations::WRITE,
        |_: Value| std::future::ready(Ok(ToolOutput::text("x"))),
    );
    assert_eq!(reader.effects(&json!({})), CallEffects::none());
    assert_eq!(writer.effects(&json!({})), CallEffects::unknown());
}

#[test]
fn interrupt_kind_defaults_to_untrusted() {
    assert_eq!(InterruptKind::default(), InterruptKind::Untrusted);
}
