//! Calls a running tool makes on its own (a coding agent it drives asks to
//! act), judged by the run's own safeguards (`ToolContext::nested`): hooks
//! decide, an interruption is settled by the run's resolver while the tool
//! waits, and an owner's decline binds the rest of the run.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{ScriptedProvider, call, spec, text, tool_calls};
use aidaemon_core::{
    Agent, ApprovalRisk, CallEffects, CancellationToken, EffectClass, EffectKey, Hook, HookContext,
    InterruptKind, InterruptResolver, Interruption, NestedVerdict, OWNER_DECLINED,
    OWNER_UNANSWERED, OwnerApprovalRequest, ReceiptStatus, Resolution, RunState, SourceTrust, Tool,
    ToolAnnotations, ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use serde_json::{Value, json};

fn write(path: &str) -> EffectKey {
    EffectKey {
        class: EffectClass::Write,
        target: path.to_string(),
    }
}

/// A file tool: its call writes `path`.
struct WriteFile {
    ran: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec("write_file", "writes a file", &["path"])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }
    fn source_trust(&self, _: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }
    fn effects(&self, args: &Value) -> CallEffects {
        CallEffects {
            keys: vec![write(args["path"].as_str().unwrap_or_default())],
            complete: true,
            understood: true,
        }
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        self.ran.fetch_add(1, Ordering::SeqCst);
        Ok(ToolOutput::text("written"))
    }
}

/// Drives a "coding agent" that asks to write each path in its arguments,
/// and records what its run answered.
struct DriveAgent {
    verdicts: Arc<Mutex<Vec<NestedVerdict>>>,
}

#[async_trait]
impl Tool for DriveAgent {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        spec("drive_agent", "runs a coding agent", &["paths"])
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
    async fn call(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let nested = ctx
            .nested
            .clone()
            .expect("a running batch judges nested calls");
        let paths = args["paths"].as_str().unwrap_or_default();
        for (index, path) in paths.split(',').enumerate() {
            let verdict = nested
                .authorize(ToolCall::new(
                    format!("nested-{index}"),
                    "write_file",
                    json!({"path": path}),
                ))
                .await;
            self.verdicts.lock().unwrap().push(verdict);
        }
        Ok(ToolOutput::text("the agent finished"))
    }
}

/// Asks the owner before writing anything under `/protected`.
struct AskForProtected {
    asked: Arc<AtomicUsize>,
}

#[async_trait]
impl Hook for AskForProtected {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        let protected = tool
            .effects
            .keys
            .iter()
            .any(|key| key.target.starts_with("/protected"));
        if call.name == "write_file" && protected {
            self.asked.fetch_add(1, Ordering::SeqCst);
            ToolDecision::Interrupt {
                reason: "writing here needs the owner".to_string(),
                kind: InterruptKind::OwnerApproval(OwnerApprovalRequest {
                    card: None,
                    effects: tool.effects.keys.clone(),
                    summary: "Write a protected file".to_string(),
                    subject: format!("write {}", call.arguments["path"]),
                    risk: ApprovalRisk::Medium,
                    warnings: Vec::new(),
                }),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

/// Answers every interruption the same way, and keeps what it was asked.
struct Answer {
    resolution: Option<Resolution>,
    seen: Mutex<Vec<Interruption>>,
}

impl Answer {
    fn new(resolution: Option<Resolution>) -> Self {
        Self {
            resolution,
            seen: Mutex::default(),
        }
    }
}

#[async_trait]
impl InterruptResolver for Answer {
    async fn resolve(&self, _: &RunState, interruption: &Interruption) -> Option<Resolution> {
        self.seen.lock().unwrap().push(interruption.clone());
        self.resolution.clone()
    }
}

struct Run {
    verdicts: Vec<NestedVerdict>,
    asked: usize,
    written_by_model: usize,
    outcome: aidaemon_core::RunOutcome,
}

/// The model drives the agent once over `paths`, then (when `then_write`
/// names a path) calls `write_file` itself.
async fn run(paths: &[&str], then_write: Option<&str>, resolver: &Answer) -> Run {
    let verdicts = Arc::new(Mutex::new(Vec::new()));
    let asked = Arc::new(AtomicUsize::new(0));
    let ran = Arc::new(AtomicUsize::new(0));
    let mut script = vec![tool_calls([call(
        "c1",
        "drive_agent",
        json!({"paths": paths.join(",")}),
    )])];
    if let Some(path) = then_write {
        script.push(tool_calls([call(
            "c2",
            "write_file",
            json!({"path": path}),
        )]));
    }
    script.push(text("done"));
    let agent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(script))
        .tool(DriveAgent {
            verdicts: verdicts.clone(),
        })
        .tool(WriteFile { ran: ran.clone() })
        .hook_arc(Arc::new(AskForProtected {
            asked: asked.clone(),
        }))
        .build()
        .unwrap();
    let outcome = agent
        .run_resolving(
            RunState::new("parent", "s", "fix the project"),
            CancellationToken::new(),
            resolver,
        )
        .await
        .unwrap();
    let verdicts = verdicts.lock().unwrap().clone();
    Run {
        verdicts,
        asked: asked.load(Ordering::SeqCst),
        written_by_model: ran.load(Ordering::SeqCst),
        outcome,
    }
}

#[tokio::test]
async fn a_nested_call_the_hooks_allow_is_allowed_without_asking() {
    let resolver = Answer::new(None);
    let out = run(&["/work/notes.txt"], None, &resolver).await;
    assert_eq!(out.verdicts, vec![NestedVerdict::Allow]);
    assert_eq!(out.asked, 0);
    assert!(resolver.seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_nested_call_the_owner_approves_is_allowed_and_never_run_by_the_kernel() {
    let resolver = Answer::new(Some(Resolution::Approve));
    let out = run(&["/protected/a.txt"], None, &resolver).await;
    assert_eq!(out.verdicts, vec![NestedVerdict::Allow]);
    assert_eq!(out.asked, 1);
    // The owner was shown the nested call itself.
    let seen = resolver.seen.lock().unwrap();
    assert_eq!(seen[0].call.name, "write_file");
    assert_eq!(seen[0].call.arguments, json!({"path": "/protected/a.txt"}));
    // Judged only: the kernel never ran `write_file` for it.
    assert_eq!(out.written_by_model, 0);
}

/// The owner's decline of a nested call binds the rest of the run: the
/// agent's second ask for the same file and the model's own later call
/// are refused without a new card.
#[tokio::test]
async fn an_owners_decline_of_a_nested_call_binds_the_run() {
    let resolver = Answer::new(Some(Resolution::Deny {
        reason: "Alice Rivera said no".to_string(),
    }));
    let out = run(
        &["/protected/a.txt", "/protected/a.txt"],
        Some("/protected/a.txt"),
        &resolver,
    )
    .await;
    assert_eq!(out.verdicts.len(), 2);
    // The first is the owner's answer; the second is refused because of it.
    let NestedVerdict::Declined { reason } = &out.verdicts[0] else {
        panic!("{:?}", out.verdicts[0]);
    };
    assert!(reason.starts_with(OWNER_DECLINED), "{reason}");
    let NestedVerdict::Deny { reason } = &out.verdicts[1] else {
        panic!("{:?}", out.verdicts[1]);
    };
    assert!(reason.starts_with(OWNER_DECLINED), "{reason}");
    assert_eq!(out.asked, 1, "one card, then the decline binds");
    assert_eq!(out.written_by_model, 0);
    let refused = out
        .outcome
        .state
        .receipts
        .iter()
        .find(|receipt| receipt.call_id == "c2")
        .unwrap();
    assert_ne!(refused.status, ReceiptStatus::Succeeded);
    assert!(
        out.outcome
            .state
            .declined
            .contains(&write("/protected/a.txt"))
    );
}

#[tokio::test]
async fn a_nested_call_nobody_answers_is_unanswered_and_not_remembered() {
    let resolver = Answer::new(Some(Resolution::Unanswered {
        reason: "the card expired".to_string(),
    }));
    let out = run(&["/protected/a.txt"], None, &resolver).await;
    let [NestedVerdict::Unanswered { reason }] = out.verdicts.as_slice() else {
        panic!("{:?}", out.verdicts);
    };
    assert!(reason.starts_with(OWNER_UNANSWERED), "{reason}");
    assert!(out.outcome.state.declined.is_empty());
}

#[tokio::test]
async fn a_nested_call_of_a_tool_the_run_lacks_is_refused() {
    let verdicts = Arc::new(Mutex::new(Vec::new()));
    struct AskUnknown(Arc<Mutex<Vec<NestedVerdict>>>);
    #[async_trait]
    impl Tool for AskUnknown {
        fn spec(&self) -> aidaemon_core::ToolSpec {
            spec("drive_agent", "runs a coding agent", &[])
        }
        fn annotations(&self) -> ToolAnnotations {
            ToolAnnotations::WRITE
        }
        fn source_trust(&self, _: &Value) -> SourceTrust {
            SourceTrust::Trusted
        }
        async fn call(&self, _args: Value, ctx: &ToolContext) -> ToolResult {
            let verdict = ctx
                .nested
                .as_ref()
                .unwrap()
                .authorize(ToolCall::new("n", "format_disk", json!({})))
                .await;
            self.0.lock().unwrap().push(verdict);
            Ok(ToolOutput::text("ok"))
        }
    }
    let agent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new(vec![
            tool_calls([call("c1", "drive_agent", json!({}))]),
            text("done"),
        ]))
        .tool(AskUnknown(verdicts.clone()))
        .build()
        .unwrap();
    agent.run("s", "go").await.unwrap();
    let verdicts = verdicts.lock().unwrap();
    assert!(matches!(
        verdicts.as_slice(),
        [NestedVerdict::Deny { reason }] if reason.contains("format_disk")
    ));
}
