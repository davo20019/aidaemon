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
    asks: &'static [&'static str],
    seen_declined: Mutex<Vec<Vec<EffectKey>>>,
    seen_effects: Mutex<Vec<CallEffects>>,
}

impl AskBeforeSend {
    fn new(kind: InterruptKind) -> Arc<Self> {
        Self::asking(kind, &["send"])
    }

    /// Interrupts calls of every tool in `asks`.
    fn asking(kind: InterruptKind, asks: &'static [&'static str]) -> Arc<Self> {
        Arc::new(Self {
            kind,
            asks,
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
        if self.asks.contains(&call.name.as_str()) {
            ToolDecision::Interrupt {
                reason: "the owner must confirm this send".to_string(),
                kind: self.kind.clone(),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

/// A send tool that declares one effect.
struct Send {
    count: Arc<AtomicUsize>,
    effect: EffectKey,
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
            keys: vec![self.effect.clone()],
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
    agent_with_effect(hook, replies, egress_key())
}

fn agent_with_effect(
    hook: Arc<AskBeforeSend>,
    replies: Vec<aidaemon_core::ModelResponse>,
    effect: EffectKey,
) -> (Agent, Arc<AtomicUsize>) {
    let sent = Arc::new(AtomicUsize::new(0));
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(replies))
        .tool(Send {
            count: sent.clone(),
            effect,
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
    // The same effect denied twice (`send`, then `note`, which declares no
    // effects of its own so the kernel lets it reach the hook) is
    // remembered once, and the second ask sees the first decline.
    let hook = AskBeforeSend::asking(
        InterruptKind::OwnerApproval(owner_request(vec![egress_key()])),
        &["send", "note"],
    );
    let sent = Arc::new(AtomicUsize::new(0));
    let a = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("c1", "send", json!({}))]),
            tool_calls([call("c2", "note", json!({}))]),
            text("done"),
        ]))
        .tool(Send {
            count: sent.clone(),
            effect: egress_key(),
        })
        .tool(effect_tool("note", CallEffects::none(), Arc::default()))
        .hook_arc(hook.clone())
        .build()
        .unwrap();
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
    assert_eq!(seen.len(), 2, "the hook was asked about both calls");
    assert!(seen[0].is_empty());
    assert_eq!(seen[1], vec![egress_key()]);
    let reasons: Vec<_> = out
        .receipts()
        .iter()
        .map(|r| r.reason.clone().unwrap_or_default())
        .collect();
    assert!(
        reasons.iter().all(|r| r.starts_with("[OWNER DECLINED]")),
        "{reasons:?}"
    );
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
    let (b, _) = agent_with_effect(second.clone(), send_once(), other.clone());
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

// ---------- Task 2: a decline binds the run before any hook ----------

/// A tool with fixed effects that counts its calls.
struct EffectTool {
    name: String,
    effects: CallEffects,
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for EffectTool {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec(&self.name, "acts", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        SEND
    }
    fn source_trust(&self, _: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }
    fn effects(&self, _args: &Value) -> CallEffects {
        self.effects.clone()
    }
    async fn call(
        &self,
        _args: Value,
        _ctx: &aidaemon_core::ToolContext,
    ) -> aidaemon_core::ToolResult {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(ToolOutput::text("acted"))
    }
}

fn effect_tool(name: &str, effects: CallEffects, count: Arc<AtomicUsize>) -> EffectTool {
    EffectTool {
        name: name.to_string(),
        effects,
        count,
    }
}

fn egress(target: &str) -> CallEffects {
    CallEffects {
        keys: vec![key(EffectClass::Egress, target)],
        complete: true,
    }
}

/// Records every call it is asked about; asks the owner before `http`,
/// with the call's declared effects.
#[derive(Default)]
struct AskBeforeHttp {
    seen: Mutex<Vec<String>>,
}

#[async_trait]
impl Hook for AskBeforeHttp {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        self.seen.lock().unwrap().push(call.name.clone());
        if call.name == "http" {
            ToolDecision::Interrupt {
                reason: "the owner must confirm this request".to_string(),
                kind: InterruptKind::OwnerApproval(owner_request(tool.effects.keys.clone())),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

/// Denies every interruption and counts how often it was asked.
#[derive(Default)]
struct CountingDeny {
    asked: AtomicUsize,
}

#[async_trait]
impl InterruptResolver for CountingDeny {
    async fn resolve(&self, _: &RunState, _: &Interruption) -> Option<Resolution> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        Some(Resolution::Deny {
            reason: "Alice Rivera said no".to_string(),
        })
    }
}

fn http_then(
    second: &str,
    second_effects: CallEffects,
) -> (Agent, Arc<AskBeforeHttp>, Arc<AtomicUsize>) {
    let hook = Arc::new(AskBeforeHttp::default());
    let ran = Arc::new(AtomicUsize::new(0));
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("c1", "http", json!({}))]),
            tool_calls([call("c2", second, json!({}))]),
            text("done"),
        ]))
        .tool(effect_tool(
            "http",
            egress("h.example.invalid:443"),
            ran.clone(),
        ))
        .tool(effect_tool(second, second_effects, ran.clone()))
        .hook_arc(hook.clone())
        .build()
        .unwrap();
    (agent, hook, ran)
}

#[tokio::test]
async fn declined_effect_blocks_other_tool_in_same_run() {
    let (a, hook, ran) = http_then("shell", egress("h.example.invalid:443"));
    let resolver = CountingDeny::default();
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &resolver,
        )
        .await
        .unwrap();

    assert!(out.is_completed(), "{:?}", out.status);
    let shell = &out.receipts()[1];
    assert_eq!(shell.tool, "shell");
    assert_eq!(shell.status, ReceiptStatus::Denied);
    let reason = shell.reason.as_deref().unwrap();
    assert!(reason.starts_with("[OWNER DECLINED]"), "{reason}");
    assert!(reason.contains("egress h.example.invalid:443"), "{reason}");
    assert!(reason.contains("Do not attempt it another way"), "{reason}");
    assert_eq!(*hook.seen.lock().unwrap(), vec!["http".to_string()]);
    assert_eq!(resolver.asked.load(Ordering::SeqCst), 1);
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn decline_does_not_block_other_targets() {
    let (a, hook, ran) = http_then("shell", egress("other.example.invalid:443"));
    let resolver = CountingDeny::default();
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &resolver,
        )
        .await
        .unwrap();

    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(out.receipts()[1].status, ReceiptStatus::Succeeded);
    assert_eq!(
        *hook.seen.lock().unwrap(),
        vec!["http".to_string(), "shell".to_string()]
    );
    assert_eq!(ran.load(Ordering::SeqCst), 1);
}

/// Declares the safeguards it is built with.
struct Guards(&'static [aidaemon_core::Safeguard]);

impl Hook for Guards {
    fn safeguards(&self) -> &'static [aidaemon_core::Safeguard] {
        self.0
    }
}

#[test]
fn build_requires_approval_and_egress_safeguards() {
    use aidaemon_core::{BuildError, Safeguard};
    assert_eq!(Safeguard::REQUIRED.len(), 5);
    assert!(Safeguard::REQUIRED.contains(&Safeguard::Approval));
    assert!(Safeguard::REQUIRED.contains(&Safeguard::Egress));

    let base = || {
        Agent::builder("a")
            .model("m")
            .provider(ScriptedProvider::new([]))
            .tool(effect_tool("t", CallEffects::none(), Arc::default()))
            .hook(Guards(&[
                Safeguard::Authority,
                Safeguard::Scope,
                Safeguard::UntrustedOutput,
            ]))
    };
    assert!(matches!(
        base().build(),
        Err(BuildError::MissingSafeguard(Safeguard::Approval))
    ));
    assert!(matches!(
        base().attest(Safeguard::Approval).build(),
        Err(BuildError::MissingSafeguard(Safeguard::Egress))
    ));
    assert!(base()
        .attest(Safeguard::Approval)
        .hook(Guards(&[Safeguard::Egress]))
        .build()
        .is_ok());
    assert_eq!(Safeguard::Approval.to_string(), "approval");
    assert_eq!(Safeguard::Egress.to_string(), "egress");
}

// ---------- normalizing constructors ----------

#[test]
fn path_constructor_normalizes() {
    let p = |raw: &str| EffectKey::path(EffectClass::Delete, raw).map(|k| k.target);
    assert_eq!(
        p("/tmp/aidaemon-test/a"),
        Some("/tmp/aidaemon-test/a".into())
    );
    assert_eq!(
        p("/tmp//aidaemon-test/./a/"),
        Some("/tmp/aidaemon-test/a".into())
    );
    assert_eq!(
        p("/tmp/aidaemon-test/x/../a"),
        Some("/tmp/aidaemon-test/a".into())
    );
    assert_eq!(p("/"), Some("/".into()));
    assert_eq!(p("//"), Some("/".into()));
    assert_eq!(p("/tmp/.."), Some("/".into()));
    assert_eq!(p("/.."), None);
    assert_eq!(p("/tmp/../../etc"), None);
    let k = EffectKey::path(EffectClass::Write, "/etc/hosts").unwrap();
    assert_eq!(k.class, EffectClass::Write);
}

#[test]
fn relative_path_rejected() {
    assert_eq!(EffectKey::path(EffectClass::Write, "notes.txt"), None);
    assert_eq!(EffectKey::path(EffectClass::Write, "./a"), None);
    assert_eq!(EffectKey::path(EffectClass::Write, "~/a"), None);
    assert_eq!(EffectKey::path(EffectClass::Write, ""), None);
}

#[test]
fn egress_default_ports() {
    let t = |scheme, host, port| EffectKey::egress(scheme, host, port).target;
    assert_eq!(
        t("https", "H.Example.Invalid", None),
        "h.example.invalid:443"
    );
    assert_eq!(t("http", "h.example.invalid", None), "h.example.invalid:80");
    assert_eq!(t("ssh", "h.example.invalid", None), "h.example.invalid:22");
    assert_eq!(
        t("HTTPS", "h.example.invalid", None),
        "h.example.invalid:443"
    );
    assert_eq!(
        t("https", "h.example.invalid", Some(8443)),
        "h.example.invalid:8443"
    );
    assert_eq!(
        t("ftp", "h.example.invalid", Some(21)),
        "h.example.invalid:21"
    );
    assert_eq!(t("ftp", "h.example.invalid", None), "h.example.invalid:0");
    assert_eq!(
        EffectKey::egress("https", "h", None).class,
        EffectClass::Egress
    );
}

#[test]
fn host_and_host_443_overlap_via_constructor() {
    let bare = EffectKey::egress("https", "H.example.invalid", None);
    let explicit = EffectKey::egress("https", "h.example.invalid", Some(443));
    assert!(bare.overlaps(&explicit));
    assert!(!bare.overlaps(&EffectKey::egress("http", "h.example.invalid", None)));

    let named = EffectKey::named(EffectClass::Credential, "acme-dev");
    assert_eq!(named, key(EffectClass::Credential, "acme-dev"));
    assert_eq!(
        EffectKey::named(EffectClass::Process, "git"),
        key(EffectClass::Process, "git")
    );
}

// ---------- every way of denying records the decline ----------

fn interrupted_send() -> (Agent, RunState) {
    let hook = AskBeforeSend::new(InterruptKind::OwnerApproval(owner_request(vec![
        egress_key(),
    ])));
    let (a, _) = agent(hook, send_once());
    let state = futures_block(a.run("s", "go")).unwrap().state;
    assert!(state.pending.is_some());
    (a, state)
}

fn futures_block<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

fn no() -> Resolution {
    Resolution::Deny {
        reason: "Alice Rivera said no".to_string(),
    }
}

#[test]
fn resume_deny_records_decline() {
    let (a, state) = interrupted_send();
    let out = futures_block(a.resume(state, no())).unwrap();
    assert_eq!(out.state.declined, vec![egress_key()]);
    let reason = out.receipts()[0].reason.clone().unwrap();
    assert!(reason.starts_with("[OWNER DECLINED]"), "{reason}");
}

#[test]
fn resume_with_cancel_deny_records_decline() {
    let (a, state) = interrupted_send();
    let out = futures_block(a.resume_with_cancel(state, no(), CancellationToken::new())).unwrap();
    assert_eq!(out.state.declined, vec![egress_key()]);
}

#[test]
fn deny_after_cancel_records_decline() {
    // The run is cancelled while the owner is being asked; the denial that
    // arrives afterwards still binds the run.
    let (a, state) = interrupted_send();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let out = futures_block(a.resume_with_cancel(state, no(), cancel)).unwrap();
    assert_eq!(out.state.declined, vec![egress_key()]);
}

#[test]
fn untrusted_egress_is_judged_by_kind() {
    use aidaemon_core::{is_untrusted_egress, UNTRUSTED_EGRESS_INTERRUPT};
    let reason = format!("{UNTRUSTED_EGRESS_INTERRUPT} `send` acts outside the workspace");
    let untrusted = Interruption {
        call: call("c1", "send", json!({})),
        reason: reason.clone(),
        kind: InterruptKind::Untrusted,
    };
    assert!(is_untrusted_egress(&untrusted));
    let owner = Interruption {
        kind: InterruptKind::OwnerApproval(owner_request(vec![egress_key()])),
        ..untrusted
    };
    assert!(!is_untrusted_egress(&owner));
}

// ---------- fix round 1 ----------

#[tokio::test]
async fn trailing_dot_host_is_the_same_declined_host() {
    let (a, hook, ran) = http_then(
        "shell",
        CallEffects {
            keys: vec![EffectKey::egress("https", "H.example.invalid.", None)],
            complete: true,
        },
    );
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &CountingDeny::default(),
        )
        .await
        .unwrap();
    let shell = &out.receipts()[1];
    assert_eq!(shell.status, ReceiptStatus::Denied);
    assert!(shell
        .reason
        .as_deref()
        .unwrap()
        .starts_with("[OWNER DECLINED]"));
    assert_eq!(*hook.seen.lock().unwrap(), vec!["http".to_string()]);
    assert_eq!(ran.load(Ordering::SeqCst), 0);
    assert_eq!(
        EffectKey::egress("https", "h.example.invalid.", None).target,
        "h.example.invalid:443"
    );
}

#[tokio::test]
async fn decline_without_effects_records_the_call_subject() {
    let hook = AskBeforeSend::new(InterruptKind::OwnerApproval(owner_request(Vec::new())));
    let (a, _) = agent(hook, send_once());
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &deny(),
        )
        .await
        .unwrap();
    assert_eq!(
        out.state.declined,
        vec![EffectKey::named(
            EffectClass::Call,
            "POST https://h.example.invalid/report"
        )]
    );
    assert_eq!(EffectClass::Call.to_string(), "call");
}

#[tokio::test]
async fn decline_binds_a_later_call_in_the_same_batch() {
    let hook = Arc::new(AskBeforeHttp::default());
    let ran = Arc::new(AtomicUsize::new(0));
    let a = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([
                call("c1", "http", json!({})),
                call("c2", "shell", json!({})),
            ]),
            text("done"),
        ]))
        .tool(effect_tool(
            "http",
            egress("h.example.invalid:443"),
            ran.clone(),
        ))
        .tool(effect_tool(
            "shell",
            egress("h.example.invalid:443"),
            ran.clone(),
        ))
        .hook_arc(hook.clone())
        .build()
        .unwrap();
    let out = a
        .run_resolving(
            RunState::new("a", "s", "go"),
            CancellationToken::new(),
            &CountingDeny::default(),
        )
        .await
        .unwrap();
    let r = out.receipts();
    assert_eq!(r[1].tool, "shell");
    assert!(r[1]
        .reason
        .as_deref()
        .unwrap()
        .starts_with("[OWNER DECLINED] The owner declined"));
    assert_eq!(*hook.seen.lock().unwrap(), vec!["http".to_string()]);
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}
