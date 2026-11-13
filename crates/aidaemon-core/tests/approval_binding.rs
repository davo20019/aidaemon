//! An owner's approval answers one question about one call: tampering with
//! the saved state, swapping the call, replaying the answer or a gate that
//! changed its mind before the answer came must not run anything the owner
//! did not approve.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, ApprovalRisk, CoreError, FnTool, Hook, HookContext, InterruptKind, OwnerApprovalRequest,
    ReceiptStatus, Resolution, RunState, RunStatus, ToolAnnotations, ToolCall, ToolDecision,
    ToolInfo, ToolOutput, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};

/// A gate before the approval hook (authority, scope): denies paths with
/// `outside` in them, and every call once `closed`.
struct Gate {
    closed: Arc<AtomicBool>,
}

#[async_trait]
impl Hook for Gate {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        call: &ToolCall,
        _: &ToolInfo,
    ) -> ToolDecision {
        if self.closed.load(Ordering::SeqCst) {
            return ToolDecision::Deny {
                reason: "the gate is closed".into(),
            };
        }
        if call.arguments.to_string().contains("outside") {
            return ToolDecision::Deny {
                reason: "outside the root".into(),
            };
        }
        ToolDecision::Allow
    }
}

/// Asks the owner before every call.
struct AskOwner;

#[async_trait]
impl Hook for AskOwner {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        call: &ToolCall,
        _: &ToolInfo,
    ) -> ToolDecision {
        ToolDecision::Interrupt {
            reason: format!("approve {}", call.name),
            kind: InterruptKind::OwnerApproval(OwnerApprovalRequest {
                card: None,
                effects: Vec::new(),
                summary: format!("Run {} {}", call.name, call.arguments),
                subject: format!("{} {}", call.name, call.arguments),
                risk: ApprovalRisk::High,
                warnings: Vec::new(),
            }),
        }
    }
}

type Ran = Arc<Mutex<Vec<String>>>;

fn recording_tool(name: &str, ran: &Ran) -> impl aidaemon_core::Tool + use<> {
    let ran = ran.clone();
    let tool_name = name.to_string();
    FnTool::new(
        ToolSpec {
            name: name.into(),
            description: "test tool".into(),
            parameters: json!({"type": "object"}),
        },
        ToolAnnotations::WRITE,
        move |args: Value| {
            let ran = ran.clone();
            let tool_name = tool_name.clone();
            async move {
                ran.lock().unwrap().push(format!("{tool_name} {args}"));
                Ok(ToolOutput::text("ok"))
            }
        },
    )
}

struct Setup {
    agent: Agent,
    ran: Ran,
    closed: Arc<AtomicBool>,
}

fn setup(script: Vec<aidaemon_core::ModelResponse>) -> Setup {
    let ran: Ran = Arc::default();
    let closed = Arc::new(AtomicBool::new(false));
    let agent = Agent::builder("bound")
        .model("m")
        .provider(ScriptedProvider::new(script))
        .tool(recording_tool("write_note", &ran))
        .tool(recording_tool("deploy", &ran))
        .hook(Gate {
            closed: closed.clone(),
        })
        .hook(AskOwner)
        .allow_unguarded()
        .build()
        .unwrap();
    Setup { agent, ran, closed }
}

fn write_final() -> Vec<aidaemon_core::ModelResponse> {
    vec![
        tool_calls([call("c1", "write_note", json!({"path": "final.txt"}))]),
        text("done"),
    ]
}

fn tampered(state: &RunState, from: &str, to: &str) -> RunState {
    let raw = serde_json::to_string(state).unwrap().replace(from, to);
    serde_json::from_str(&raw).unwrap()
}

#[tokio::test]
async fn an_approval_runs_its_call_once() {
    let s = setup(write_final());
    let out = s.agent.run("s", "go").await.unwrap();
    assert!(matches!(out.status, RunStatus::Interrupted { .. }));
    let out = s
        .agent
        .resume(out.state, Resolution::Approve)
        .await
        .unwrap();
    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(
        *s.ran.lock().unwrap(),
        vec![r#"write_note {"path":"final.txt"}"#]
    );
}

#[tokio::test]
async fn an_edited_saved_call_is_judged_again_from_the_first_gate() {
    let s = setup(write_final());
    let out = s.agent.run("s", "go").await.unwrap();
    let state = tampered(&out.state, "final.txt", "../outside/evil.txt");
    let out = s.agent.resume(state, Resolution::Approve).await.unwrap();
    assert!(
        s.ran.lock().unwrap().is_empty(),
        "{:?}",
        s.ran.lock().unwrap()
    );
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
}

#[tokio::test]
async fn an_approval_for_one_call_never_runs_another() {
    let s = setup(write_final());
    let out = s.agent.run("s", "go").await.unwrap();
    let state = tampered(
        &tampered(&out.state, "\"write_note\"", "\"deploy\""),
        r#"{"path":"final.txt"}"#,
        r#"{"env":"prod"}"#,
    );
    let out = s.agent.resume(state, Resolution::Approve).await.unwrap();
    assert!(
        s.ran.lock().unwrap().is_empty(),
        "{:?}",
        s.ran.lock().unwrap()
    );
    // The swapped call is a new question for the owner.
    let RunStatus::Interrupted { interruption } = &out.status else {
        panic!("expected a new question, got {:?}", out.status);
    };
    assert_eq!(interruption.call.name, "deploy");
}

#[tokio::test]
async fn an_approval_answers_once() {
    let s = setup(write_final());
    let out = s.agent.run("s", "go").await.unwrap();
    let copy = out.state.clone();
    s.agent
        .resume(out.state, Resolution::Approve)
        .await
        .unwrap();
    let again = s.agent.resume(copy, Resolution::Approve).await;
    assert!(
        matches!(again, Err(CoreError::ApprovalAlreadyUsed)),
        "{again:?}"
    );
    assert_eq!(s.ran.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn gates_before_the_approval_judge_the_call_again() {
    let s = setup(write_final());
    let out = s.agent.run("s", "go").await.unwrap();
    // Authority was withdrawn while the owner was deciding.
    s.closed.store(true, Ordering::SeqCst);
    let out = s
        .agent
        .resume(out.state, Resolution::Approve)
        .await
        .unwrap();
    assert!(s.ran.lock().unwrap().is_empty());
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
    assert_eq!(
        out.receipts()[0].reason.as_deref(),
        Some("the gate is closed")
    );
}

#[tokio::test]
async fn the_interruption_carries_the_digest_of_its_call() {
    let s = setup(write_final());
    let out = s.agent.run("s", "go").await.unwrap();
    let RunStatus::Interrupted { interruption } = &out.status else {
        panic!()
    };
    let digest = interruption.digest.clone().expect("a digest");
    assert_eq!(digest.len(), 64, "{digest}");
    // Another agent in the same process honors it: the key is the
    // process's, not the agent value's.
    let again = setup(vec![text("done")]);
    let out = again
        .agent
        .resume(out.state, Resolution::Approve)
        .await
        .unwrap();
    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(again.ran.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_denial_needs_no_binding() {
    let s = setup(write_final());
    let out = s.agent.run("s", "go").await.unwrap();
    let copy = out.state.clone();
    let counter = AtomicUsize::new(0);
    for state in [out.state, copy] {
        let out = s
            .agent
            .resume(
                state,
                Resolution::Deny {
                    reason: "no".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(out.receipts()[0].status, ReceiptStatus::Denied);
        counter.fetch_add(1, Ordering::SeqCst);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 2);
    assert!(s.ran.lock().unwrap().is_empty());
}
