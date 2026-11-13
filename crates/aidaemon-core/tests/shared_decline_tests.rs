//! Declines shared between a run and the child runs it starts
//! (`SharedDeclines`): a decline in either reaches the other before its next
//! call, without asking the owner again.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, ApprovalRisk, CallEffects, CancellationToken, EffectClass, EffectKey, Hook, HookContext,
    InterruptKind, InterruptResolver, Interruption, OwnerApprovalRequest, ReceiptStatus,
    Resolution, RunState, RunStatus, SharedDeclines, SourceTrust, Tool, ToolAnnotations, ToolCall,
    ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult, OWNER_DECLINED,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Notify;

const SEND: ToolAnnotations = ToolAnnotations {
    egress: true,
    ..ToolAnnotations::WRITE
};

fn egress(host: &str) -> EffectKey {
    EffectKey {
        class: EffectClass::Egress,
        target: format!("{host}:443"),
    }
}

fn report_host() -> EffectKey {
    egress("reports.example.invalid")
}

fn other_host() -> EffectKey {
    egress("status.example.invalid")
}

/// Asks the owner before every call of the tools in `asks`, and counts
/// how often it was consulted about each tool.
struct AskOwner {
    asks: &'static [&'static str],
    consulted: Mutex<Vec<String>>,
}

impl AskOwner {
    fn new(asks: &'static [&'static str]) -> Arc<Self> {
        Arc::new(Self {
            asks,
            consulted: Mutex::default(),
        })
    }

    fn consulted_about(&self, tool: &str) -> usize {
        self.consulted
            .lock()
            .unwrap()
            .iter()
            .filter(|name| *name == tool)
            .count()
    }
}

#[async_trait]
impl Hook for AskOwner {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        self.consulted.lock().unwrap().push(call.name.clone());
        if self.asks.contains(&call.name.as_str()) {
            ToolDecision::Interrupt {
                reason: "the owner must confirm this send".to_string(),
                kind: InterruptKind::OwnerApproval(OwnerApprovalRequest {
                    card: None,
                    effects: tool.effects.keys.clone(),
                    summary: "Send the weekly report for Alice Rivera".to_string(),
                    subject: format!("send via {}", call.name),
                    risk: ApprovalRisk::High,
                    warnings: Vec::new(),
                }),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

/// A send tool with one declared egress effect.
struct Sender {
    name: &'static str,
    effect: EffectKey,
    sent: Arc<AtomicUsize>,
}

impl Sender {
    fn new(name: &'static str, effect: EffectKey) -> (Self, Arc<AtomicUsize>) {
        let sent = Arc::new(AtomicUsize::new(0));
        (
            Self {
                name,
                effect,
                sent: sent.clone(),
            },
            sent,
        )
    }
}

#[async_trait]
impl Tool for Sender {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec(self.name, "sends", &[])
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
            understood: false,
        }
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        self.sent.fetch_add(1, Ordering::SeqCst);
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

/// Starts a child run that shares `declines` and whose owner declines its
/// `send` call, the way an application's spawn tool would.
struct SpawnChild {
    declines: SharedDeclines,
    child_sent: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for SpawnChild {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec("spawn", "starts a child agent", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }
    fn source_trust(&self, _: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }
    fn effects(&self, _args: &Value) -> CallEffects {
        CallEffects::none()
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        let (send, _) = Sender::new("send", report_host());
        let child_sent = self.child_sent.clone();
        let child = Agent::builder("child")
            .allow_unguarded()
            .model("m")
            .provider(ScriptedProvider::new(vec![
                tool_calls([call("k1", "send", json!({}))]),
                text("could not send"),
            ]))
            .tool(Sender {
                sent: child_sent,
                ..send
            })
            .hook_arc(AskOwner::new(&["send"]))
            .build()
            .unwrap();
        let state = RunState::new("child", "child-session", "send the report")
            .share_declines(self.declines.clone());
        let out = child
            .run_resolving(state, CancellationToken::new(), &deny())
            .await
            .unwrap();
        assert_eq!(out.state.declined, vec![report_host()]);
        Ok(ToolOutput::text("the child finished"))
    }
}

#[tokio::test]
async fn a_childs_decline_binds_its_parent_without_a_new_card() {
    let declines = SharedDeclines::new();
    let child_sent = Arc::new(AtomicUsize::new(0));
    let (send, sent) = Sender::new("send", report_host());
    let (post, posted) = Sender::new("post", other_host());
    // The parent's own policy would ask the owner before every send.
    let parent_hook = AskOwner::new(&["send"]);
    let parent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("p1", "spawn", json!({}))]),
            tool_calls([call("p2", "send", json!({}))]),
            tool_calls([call("p3", "post", json!({}))]),
            text("done"),
        ]))
        .tool(SpawnChild {
            declines: declines.clone(),
            child_sent: child_sent.clone(),
        })
        .tool(send)
        .tool(post)
        .hook_arc(parent_hook.clone())
        .build()
        .unwrap();
    // An approval is never granted: a card would show up as a send.
    let state =
        RunState::new("parent", "parent-session", "report").share_declines(declines.clone());
    let out = parent
        .run_resolving(state, CancellationToken::new(), &Fixed(Resolution::Approve))
        .await
        .unwrap();

    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(child_sent.load(Ordering::SeqCst), 0);
    // The parent's send was refused by the kernel before its policy (and so
    // the owner) was consulted.
    assert_eq!(sent.load(Ordering::SeqCst), 0);
    assert_eq!(parent_hook.consulted_about("send"), 0);
    let refused = out.receipts().iter().find(|r| r.call_id == "p2").unwrap();
    assert_eq!(refused.status, ReceiptStatus::Denied);
    assert!(
        refused
            .reason
            .as_deref()
            .unwrap()
            .starts_with(OWNER_DECLINED),
        "{:?}",
        refused.reason
    );
    // An effect nobody declined still runs.
    assert_eq!(posted.load(Ordering::SeqCst), 1);
    // The child's decline is part of the parent's record.
    assert_eq!(out.state.declined, vec![report_host()]);
    assert!(!out.is_succeeded());
    assert_eq!(declines.snapshot(), vec![report_host()]);
}

#[tokio::test]
async fn a_parent_never_succeeds_after_its_child_was_declined() {
    let declines = SharedDeclines::new();
    let parent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("p1", "spawn", json!({}))]),
            text("the child is done"),
        ]))
        .tool(SpawnChild {
            declines: declines.clone(),
            child_sent: Arc::new(AtomicUsize::new(0)),
        })
        .build()
        .unwrap();
    let out = parent
        .run_from(
            RunState::new("parent", "parent-session", "report").share_declines(declines),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(out.state.declined, vec![report_host()]);
    assert!(!out.is_succeeded());
}

/// Blocks until released, so the parent can decline while the child runs.
struct Wait {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl Tool for Wait {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec("wait", "waits", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }
    fn source_trust(&self, _: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }
    fn effects(&self, _args: &Value) -> CallEffects {
        CallEffects::none()
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        self.started.notify_one();
        self.release.notified().await;
        Ok(ToolOutput::text("waited"))
    }
}

#[tokio::test]
async fn a_parents_later_decline_reaches_a_running_child() {
    let declines = SharedDeclines::new();
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());

    // The child is already running (inside `wait`) when the parent's owner
    // declines; its policy would let every call through.
    let (child_send, child_sent) = Sender::new("send", report_host());
    let (child_post, child_posted) = Sender::new("post", other_host());
    let child_hook = AskOwner::new(&[]);
    let child = Agent::builder("child")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("k1", "wait", json!({}))]),
            tool_calls([call("k2", "send", json!({})), call("k3", "post", json!({}))]),
            text("child done"),
        ]))
        .tool(Wait {
            started: started.clone(),
            release: release.clone(),
        })
        .tool(child_send)
        .tool(child_post)
        .hook_arc(child_hook.clone())
        .build()
        .unwrap();
    let child_state =
        RunState::new("child", "child-session", "send the report").share_declines(declines.clone());
    let child_run = tokio::spawn(async move {
        child
            .run_from(child_state, CancellationToken::new())
            .await
            .unwrap()
    });
    started.notified().await;

    let (send, sent) = Sender::new("send", report_host());
    let parent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("p1", "send", json!({}))]),
            text("parent done"),
        ]))
        .tool(send)
        .hook_arc(AskOwner::new(&["send"]))
        .build()
        .unwrap();
    let parent_out = parent
        .run_resolving(
            RunState::new("parent", "parent-session", "report").share_declines(declines.clone()),
            CancellationToken::new(),
            &deny(),
        )
        .await
        .unwrap();
    assert_eq!(sent.load(Ordering::SeqCst), 0);
    assert_eq!(parent_out.state.declined, vec![report_host()]);

    release.notify_one();
    let child_out = child_run.await.unwrap();
    assert!(child_out.is_completed(), "{:?}", child_out.status);
    // The overlapping call is refused before the child's policy sees it.
    assert_eq!(child_sent.load(Ordering::SeqCst), 0);
    assert_eq!(child_hook.consulted_about("send"), 0);
    let refused = child_out
        .receipts()
        .iter()
        .find(|r| r.call_id == "k2")
        .unwrap();
    assert_eq!(refused.status, ReceiptStatus::Denied);
    assert!(refused
        .reason
        .as_deref()
        .unwrap()
        .starts_with(OWNER_DECLINED));
    // A call with another effect still runs.
    assert_eq!(child_posted.load(Ordering::SeqCst), 1);
    assert_eq!(child_out.state.declined, vec![report_host()]);
    assert!(!child_out.is_succeeded());
}

#[tokio::test]
async fn a_resumed_run_shares_its_declines_once_a_set_is_reattached() {
    let (send, _) = Sender::new("send", report_host());
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("c1", "send", json!({}))]),
            text("done"),
        ]))
        .tool(send)
        .hook_arc(AskOwner::new(&["send"]))
        .build()
        .unwrap();
    let first = SharedDeclines::new();
    let out = agent
        .run_from(
            RunState::new("a", "s", "report").share_declines(first.clone()),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(matches!(out.status, RunStatus::Interrupted { .. }));
    assert!(out.state.shared_declines().is_some());

    // Saved and loaded elsewhere: the handle is not part of the data.
    let saved = serde_json::to_string(&out.state).unwrap();
    assert!(!saved.contains("shared_declines"), "{saved}");
    let loaded: RunState = serde_json::from_str(&saved).unwrap();
    assert!(loaded.shared_declines().is_none());

    // The application reattaches the family's set and the owner declines.
    let family = SharedDeclines::new();
    let resumed = agent
        .resume(
            loaded.share_declines(family.clone()),
            Resolution::Deny {
                reason: "Alice Rivera said no".to_string(),
            },
        )
        .await
        .unwrap();
    assert_eq!(resumed.state.declined, vec![report_host()]);
    assert_eq!(family.snapshot(), vec![report_host()]);
    assert!(first.snapshot().is_empty());

    // A new run in the session starts with no declines and no set.
    let next = resumed.state.with_user_message("something else");
    assert!(next.declined.is_empty());
    assert!(next.shared_declines().is_none());
}

#[test]
fn attaching_a_set_publishes_the_runs_declines_and_takes_in_the_sets() {
    let set = SharedDeclines::new();
    set.record(&[other_host()]);
    let mut state = RunState::new("a", "s", "go");
    state.declined = vec![report_host()];
    let state = state.share_declines(set.clone());
    assert_eq!(state.declined, vec![report_host(), other_host()]);
    assert_eq!(set.snapshot(), vec![other_host(), report_host()]);
    assert!(state.shared_declines().unwrap().same_set(&set));
}
