//! Behavioral tests for the kernel loop, driven by a scripted provider.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, CoreError, Event, FnTool, Hook, HookContext, InvocationStage, LimitKind, Limits,
    MemoryStore, ProviderError, Receipt, ReceiptStatus, Resolution, Retry, Role, RunSnapshot,
    RunState, RunStatus, RunStore, StopDecision, StuckConfig, StuckDecision, StuckSignal, Tool,
    ToolAnnotations, ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult,
    ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Barrier;
use tokio_util::sync::CancellationToken;

// ---------- helpers ----------

fn echo_tool() -> impl Tool {
    FnTool::new(
        spec("echo", "Echo the text back", &["text"]),
        ToolAnnotations::READ_ONLY,
        |args: Value| async move {
            Ok(ToolOutput::text(
                args["text"].as_str().unwrap_or_default().to_string(),
            ))
        },
    )
}

/// Counts calls so tests can prove a tool did or did not run.
struct CountingTool {
    name: &'static str,
    annotations: ToolAnnotations,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CountingTool {
    fn spec(&self) -> ToolSpec {
        spec(self.name, "counts calls", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        self.annotations
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(ToolOutput::text(format!("call #{n}")))
    }
}

fn counting(name: &'static str, annotations: ToolAnnotations) -> (CountingTool, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        CountingTool {
            name,
            annotations,
            calls: calls.clone(),
        },
        calls,
    )
}

fn base(provider: ScriptedProvider) -> aidaemon_core::AgentBuilder {
    Agent::builder("test-agent")
        .model("scripted")
        .instructions("You are a test agent.")
        .provider(provider)
}

fn tool_messages(state: &RunState) -> Vec<(String, String)> {
    state
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| {
            (
                m.tool_call_id.clone().unwrap_or_default(),
                m.content.clone(),
            )
        })
        .collect()
}

// ---------- the basic loop ----------

#[tokio::test]
async fn completes_when_the_model_answers_without_tools() {
    let provider = Arc::new(ScriptedProvider::new([text("hello")]));
    let agent = Agent::builder("a")
        .model("m")
        .instructions("be brief")
        .provider(provider.clone())
        .build()
        .unwrap();

    let out = agent.run("s1", "hi").await.unwrap();

    assert_eq!(out.status, RunStatus::Completed);
    assert_eq!(out.final_text.as_deref(), Some("hello"));
    assert_eq!(out.turns(), 1);
    assert_eq!(out.usage().input_tokens, 100);
    // Instructions are sent first on every request but never stored.
    let req = &provider.requests()[0];
    assert_eq!(req.messages[0].role, Role::System);
    assert_eq!(req.messages[0].content, "be brief");
    assert!(out.state.messages.iter().all(|m| m.role != Role::System));
}

#[tokio::test]
async fn tool_results_are_appended_and_receipted() {
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "echo", json!({"text": "ping"}))]),
        text("got ping"),
    ]))
    .tool(echo_tool())
    .build()
    .unwrap();

    let out = agent.run("s1", "echo ping").await.unwrap();

    assert!(out.is_completed());
    assert_eq!(out.turns(), 2);
    assert_eq!(
        tool_messages(&out.state),
        vec![("c1".into(), "ping".into())]
    );
    let r = &out.receipts()[0];
    assert_eq!(r.status, ReceiptStatus::Succeeded);
    assert_eq!(r.stage, InvocationStage::Dispatched);
    assert!(r.read_only);
}

#[tokio::test]
async fn missing_call_ids_are_filled_in() {
    let agent = base(ScriptedProvider::new([
        tool_calls([call("", "echo", json!({"text": "x"}))]),
        text("ok"),
    ]))
    .tool(echo_tool())
    .build()
    .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(out.receipts()[0].call_id, "call_1_0");
}

// ---------- errors become receipts, never crashes ----------

#[tokio::test]
async fn unknown_tool_is_reported_to_the_model() {
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "rm_rf", json!({}))]),
        text("sorry"),
    ]))
    .tool(echo_tool())
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert!(out.is_completed());
    let r = &out.receipts()[0];
    assert_eq!(r.status, ReceiptStatus::UnknownTool);
    assert_eq!(r.stage, InvocationStage::RejectedBeforeDispatch);
    let (_, content) = &tool_messages(&out.state)[0];
    assert!(content.contains("unknown tool `rm_rf`"), "{content}");
    assert!(content.contains("echo"), "lists available tools: {content}");
}

#[tokio::test]
async fn missing_required_arguments_are_rejected_before_policy() {
    struct PanicsIfAsked;
    #[async_trait]
    impl Hook for PanicsIfAsked {
        async fn before_tool(
            &self,
            _: &HookContext<'_>,
            _: &ToolCall,
            _: &ToolInfo,
        ) -> ToolDecision {
            panic!("malformed calls must not reach policy hooks");
        }
    }
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "echo", json!({"wrong": 1}))]),
        text("ok"),
    ]))
    .tool(echo_tool())
    .hook(PanicsIfAsked)
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(out.receipts()[0].status, ReceiptStatus::InvalidArguments);
    assert!(tool_messages(&out.state)[0].1.contains("text"));
}

#[tokio::test]
async fn tool_errors_and_error_outputs_are_failed_receipts() {
    let failing = FnTool::new(
        spec("fail", "always fails", &[]),
        ToolAnnotations::WRITE,
        |_args: Value| async move { Err("disk full".into()) },
    );
    let negative = FnTool::new(
        spec("grep", "search", &[]),
        ToolAnnotations::READ_ONLY,
        |_args: Value| async move {
            Ok(ToolOutput::error("no matches").with_details(json!({"exit": 1})))
        },
    );
    let agent = base(ScriptedProvider::new([
        tool_calls([call("a", "fail", json!({})), call("b", "grep", json!({}))]),
        text("done"),
    ]))
    .tool(failing)
    .tool(negative)
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    let r = out.receipts();
    assert_eq!(r[0].status, ReceiptStatus::Failed);
    assert_eq!(r[0].reason.as_deref(), Some("disk full"));
    assert_eq!(r[1].status, ReceiptStatus::Failed);
    assert_eq!(r[1].details, json!({"exit": 1}));
    assert!(out.is_completed(), "tool failures never end the run");
}

#[tokio::test]
async fn slow_tools_time_out() {
    let slow = FnTool::new(
        spec("slow", "sleeps", &[]),
        ToolAnnotations::READ_ONLY,
        |_args: Value| async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok(ToolOutput::text("never"))
        },
    );
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "slow", json!({}))]),
        text("ok"),
    ]))
    .tool(slow)
    .limits(Limits {
        tool_timeout: Duration::from_millis(50),
        ..Limits::default()
    })
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(out.receipts()[0].status, ReceiptStatus::TimedOut);
}

#[tokio::test]
async fn long_output_is_truncated_head_and_tail() {
    let big = FnTool::new(
        spec("big", "big output", &[]),
        ToolAnnotations::READ_ONLY,
        |_args: Value| async move { Ok(ToolOutput::text(format!("START{}END", "x".repeat(10_000)))) },
    );
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "big", json!({}))]),
        text("ok"),
    ]))
    .tool(big)
    .limits(Limits {
        max_tool_output_chars: 300,
        ..Limits::default()
    })
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    let (_, content) = &tool_messages(&out.state)[0];
    assert!(content.starts_with("START"));
    assert!(content.ends_with("END"));
    assert!(content.contains("characters omitted"));
    assert!(content.len() < 500);
}

// ---------- scheduling ----------

#[tokio::test]
async fn read_only_calls_run_concurrently() {
    // Each call waits for the other at a barrier: sequential execution would
    // deadlock, so finishing at all proves concurrency.
    let barrier = Arc::new(Barrier::new(2));
    let b = barrier.clone();
    let rendezvous = FnTool::new(
        spec("meet", "wait for a peer", &["who"]),
        ToolAnnotations::READ_ONLY,
        move |args: Value| {
            let b = b.clone();
            async move {
                b.wait().await;
                Ok(ToolOutput::text(format!(
                    "met {}",
                    args["who"].as_str().unwrap()
                )))
            }
        },
    );
    let agent = base(ScriptedProvider::new([
        tool_calls([
            call("c1", "meet", json!({"who": "a"})),
            call("c2", "meet", json!({"who": "b"})),
        ]),
        text("done"),
    ]))
    .tool(rendezvous)
    .build()
    .unwrap();

    let out = tokio::time::timeout(Duration::from_secs(5), agent.run("s", "go"))
        .await
        .expect("read-only calls should not run one at a time")
        .unwrap();
    // Results stay in call order.
    assert_eq!(
        tool_messages(&out.state),
        vec![("c1".into(), "met a".into()), ("c2".into(), "met b".into())]
    );
}

#[tokio::test]
async fn mutating_calls_run_one_at_a_time() {
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let (a, p) = (active.clone(), peak.clone());
    let writer = FnTool::new(
        spec("write", "writes", &[]),
        ToolAnnotations::WRITE,
        move |_args: Value| {
            let (a, p) = (a.clone(), p.clone());
            async move {
                let now = a.fetch_add(1, Ordering::SeqCst) + 1;
                p.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                a.fetch_sub(1, Ordering::SeqCst);
                Ok(ToolOutput::text("ok"))
            }
        },
    );
    let agent = base(ScriptedProvider::new([
        tool_calls((0..3).map(|i| call(&format!("c{i}"), "write", json!({})))),
        text("done"),
    ]))
    .tool(writer)
    .build()
    .unwrap();

    agent.run("s", "go").await.unwrap();
    assert_eq!(peak.load(Ordering::SeqCst), 1);
}

// ---------- hooks ----------

struct DenyDestructive;
#[async_trait]
impl Hook for DenyDestructive {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        _: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        if tool.annotations.destructive {
            ToolDecision::Deny {
                reason: "destructive tools are disabled".into(),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

#[tokio::test]
async fn deny_hook_blocks_the_tool_and_tells_the_model_why() {
    let (nuke, calls) = counting("nuke", ToolAnnotations::DESTRUCTIVE);
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "nuke", json!({}))]),
        text("ok, I won't"),
    ]))
    .tool(nuke)
    .hook(DenyDestructive)
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let r = &out.receipts()[0];
    assert_eq!(r.status, ReceiptStatus::Denied);
    assert_eq!(r.stage, InvocationStage::RejectedBeforeDispatch);
    assert!(tool_messages(&out.state)[0]
        .1
        .contains("destructive tools are disabled"));
}

#[tokio::test]
async fn rewrite_hook_changes_arguments() {
    struct ForcePrefix;
    #[async_trait]
    impl Hook for ForcePrefix {
        async fn before_tool(
            &self,
            _: &HookContext<'_>,
            call: &ToolCall,
            _: &ToolInfo,
        ) -> ToolDecision {
            let text = call.arguments["text"].as_str().unwrap_or_default();
            ToolDecision::Rewrite {
                arguments: json!({"text": format!("[safe] {text}")}),
            }
        }
    }
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "echo", json!({"text": "hi"}))]),
        text("ok"),
    ]))
    .tool(echo_tool())
    .hook(ForcePrefix)
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(tool_messages(&out.state)[0].1, "[safe] hi");
    assert_eq!(out.receipts()[0].arguments, json!({"text": "[safe] hi"}));
}

#[tokio::test]
async fn after_tool_hook_can_redact_output() {
    struct Redact;
    #[async_trait]
    impl Hook for Redact {
        async fn after_tool(&self, _: &HookContext<'_>, _: &Receipt, output: &mut ToolOutput) {
            output.content = output.content.replace("sk-secret", "[redacted]");
        }
    }
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "echo", json!({"text": "key=sk-secret"}))]),
        text("ok"),
    ]))
    .tool(echo_tool())
    .hook(Redact)
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(tool_messages(&out.state)[0].1, "key=[redacted]");
}

#[tokio::test]
async fn before_model_hook_can_edit_the_request() {
    struct AddNote;
    #[async_trait]
    impl Hook for AddNote {
        async fn before_model(
            &self,
            _: &HookContext<'_>,
            request: &mut aidaemon_core::ModelRequest,
        ) {
            request.messages.push(aidaemon_core::Message::user(
                "[memory] user prefers metric units",
            ));
        }
    }
    let provider = Arc::new(ScriptedProvider::new([text("ok")]));
    let agent = Agent::builder("a")
        .model("m")
        .provider(provider.clone())
        .hook(AddNote)
        .build()
        .unwrap();

    let out = agent.run("s", "hi").await.unwrap();
    let sent = &provider.requests()[0].messages;
    assert!(sent.last().unwrap().content.contains("metric units"));
    // Hook additions affect the request only, not the stored transcript.
    assert!(out
        .state
        .messages
        .iter()
        .all(|m| !m.content.contains("metric")));
}

/// A completion check built from receipts: the run may only finish after at
/// least one successful tool call. The shape of aidaemon's evidence closeout.
struct RequireEvidence;
#[async_trait]
impl Hook for RequireEvidence {
    async fn on_stop(&self, ctx: &HookContext<'_>, _final_text: &str) -> StopDecision {
        if ctx.receipts.iter().any(Receipt::succeeded) {
            StopDecision::Accept
        } else {
            StopDecision::Continue {
                message: "You have not run any tools yet. Do the work before answering.".into(),
            }
        }
    }
}

#[tokio::test]
async fn stop_hook_can_reject_an_unsupported_done() {
    let agent = base(ScriptedProvider::new([
        text("All done! (fabricated)"),
        tool_calls([call("c1", "echo", json!({"text": "real work"}))]),
        text("Done, with evidence."),
    ]))
    .tool(echo_tool())
    .hook(RequireEvidence)
    .build()
    .unwrap();

    let out = agent.run("s", "do the thing").await.unwrap();

    assert!(out.is_completed());
    assert_eq!(out.final_text.as_deref(), Some("Done, with evidence."));
    assert_eq!(out.turns(), 3);
    assert!(out
        .state
        .messages
        .iter()
        .any(|m| m.role == Role::User && m.content.contains("not run any tools")));
}

// ---------- interrupt / resume ----------

struct AskBeforeDestructive;
#[async_trait]
impl Hook for AskBeforeDestructive {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        call: &ToolCall,
        tool: &ToolInfo,
    ) -> ToolDecision {
        if tool.annotations.destructive {
            ToolDecision::Interrupt {
                reason: format!("approve {}?", call.name),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

fn interrupting_agent(provider: ScriptedProvider) -> (Agent, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let (reader, reads) = counting("read", ToolAnnotations::READ_ONLY);
    let (deleter, deletes) = counting("delete", ToolAnnotations::DESTRUCTIVE);
    let agent = base(provider)
        .tool(reader)
        .tool(deleter)
        .hook(AskBeforeDestructive)
        .build()
        .unwrap();
    (agent, reads, deletes)
}

#[tokio::test]
async fn interrupt_pauses_and_approve_resumes_from_serialized_state() {
    let (agent, reads, deletes) = interrupting_agent(ScriptedProvider::new([
        tool_calls([
            call("c1", "read", json!({})),
            call("c2", "delete", json!({})),
            call("c3", "read", json!({})),
        ]),
        text("cleaned up"),
    ]));

    let paused = agent.run("s", "clean up").await.unwrap();
    let RunStatus::Interrupted { interruption } = &paused.status else {
        panic!("expected interruption, got {:?}", paused.status);
    };
    assert_eq!(interruption.call.id, "c2");
    assert_eq!(
        reads.load(Ordering::SeqCst),
        1,
        "calls before the pause ran"
    );
    assert_eq!(
        deletes.load(Ordering::SeqCst),
        0,
        "the paused call did not run"
    );

    // Simulate another process: state goes through JSON.
    let json = serde_json::to_string(&paused.state).unwrap();
    let restored: RunState = serde_json::from_str(&json).unwrap();

    let done = agent.resume(restored, Resolution::Approve).await.unwrap();
    assert!(done.is_completed());
    assert_eq!(deletes.load(Ordering::SeqCst), 1);
    assert_eq!(
        reads.load(Ordering::SeqCst),
        2,
        "calls after the pause ran too"
    );
    let ids: Vec<String> = tool_messages(&done.state)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(ids, vec!["c1", "c2", "c3"], "every call answered, in order");
    assert!(done.state.pending.is_none());
}

#[tokio::test]
async fn deny_on_resume_records_a_denied_receipt() {
    let (agent, _reads, deletes) = interrupting_agent(ScriptedProvider::new([
        tool_calls([call("c1", "delete", json!({}))]),
        text("ok, leaving it"),
    ]));

    let paused = agent.run("s", "go").await.unwrap();
    let done = agent
        .resume(
            paused.state,
            Resolution::Deny {
                reason: "not today".into(),
            },
        )
        .await
        .unwrap();

    assert!(done.is_completed());
    assert_eq!(deletes.load(Ordering::SeqCst), 0);
    assert_eq!(done.receipts()[0].status, ReceiptStatus::Denied);
    assert!(tool_messages(&done.state)[0].1.contains("not today"));
}

#[tokio::test]
async fn resume_rules_are_enforced() {
    let (agent, _, _) = interrupting_agent(ScriptedProvider::new([
        tool_calls([call("c1", "delete", json!({}))]),
        text("x"),
    ]));
    let paused = agent.run("s", "go").await.unwrap();

    // Another agent id cannot pick up this state.
    let other = Agent::builder("other-agent")
        .model("m")
        .provider(ScriptedProvider::new([]))
        .build()
        .unwrap();
    assert!(matches!(
        other
            .resume(paused.state.clone(), Resolution::Approve)
            .await,
        Err(CoreError::AgentMismatch { .. })
    ));

    // An interrupted run cannot be continued without settling it.
    assert!(matches!(
        agent
            .run_from(paused.state.clone(), CancellationToken::new())
            .await,
        Err(CoreError::PendingInterruption)
    ));

    // A finished run has nothing to resume.
    let done = agent
        .resume(paused.state, Resolution::Approve)
        .await
        .unwrap();
    assert!(matches!(
        agent.resume(done.state, Resolution::Approve).await,
        Err(CoreError::NotInterrupted)
    ));
}

#[tokio::test]
async fn finished_runs_continue_with_a_new_message() {
    let agent = base(ScriptedProvider::new([text("first"), text("second")]))
        .build()
        .unwrap();
    let first = agent.run("s", "one").await.unwrap();
    let next = first.state.clone().with_user_message("two");
    assert_ne!(next.run_id, first.state.run_id);

    let second = agent
        .run_from(next, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(second.final_text.as_deref(), Some("second"));
    let users: Vec<&str> = second
        .state
        .messages
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(users, vec!["one", "two"]);
    assert_eq!(
        second.usage().output_tokens,
        40,
        "usage accumulates per session"
    );
}

// ---------- limits, stop conditions, stuck detection, cancellation ----------

#[tokio::test]
async fn turn_limit_ends_the_run_with_a_typed_status() {
    let agent = base(ScriptedProvider::new([]).then_repeat(tool_calls([call(
        "",
        "echo",
        json!({"text": "again"}),
    )])))
    .tool(echo_tool())
    .limits(Limits {
        max_turns: 3,
        ..Limits::default()
    })
    .stuck_detection(StuckConfig {
        repeat_threshold: 0,
        error_threshold: 0,
        ping_pong_window: 0,
        max_nudges: 0,
    })
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(
        out.status,
        RunStatus::LimitReached {
            limit: LimitKind::Turns
        }
    );
    assert_eq!(out.turns(), 3);
}

#[tokio::test]
async fn token_budget_is_enforced() {
    let agent = base(ScriptedProvider::new([]).then_repeat(tool_calls([call(
        "",
        "echo",
        json!({"text": "x"}),
    )])))
    .tool(echo_tool())
    .limits(Limits {
        max_tokens: Some(250), // each scripted turn bills 120
        ..Limits::default()
    })
    .stuck_detection(StuckConfig {
        repeat_threshold: 0,
        ..StuckConfig::default()
    })
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(
        out.status,
        RunStatus::LimitReached {
            limit: LimitKind::Tokens
        }
    );
    assert_eq!(out.turns(), 3);
}

#[tokio::test]
async fn custom_stop_condition() {
    let agent = base(ScriptedProvider::new([]).then_repeat(tool_calls([call(
        "",
        "echo",
        json!({"text": "x"}),
    )])))
    .tool(echo_tool())
    .stuck_detection(StuckConfig {
        repeat_threshold: 0,
        ..StuckConfig::default()
    })
    .stop_when(|s: &RunSnapshot| (s.tool_calls >= 2).then(|| "enough tool calls".to_string()))
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(
        out.status,
        RunStatus::Stopped {
            reason: "enough tool calls".into()
        }
    );
}

#[tokio::test]
async fn stuck_detector_nudges_then_stops() {
    let agent = base(ScriptedProvider::new([]).then_repeat(tool_calls([call(
        "",
        "echo",
        json!({"text": "same"}),
    )])))
    .tool(echo_tool())
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    let RunStatus::Stuck { signal } = &out.status else {
        panic!("expected stuck, got {:?}", out.status);
    };
    assert!(matches!(signal, StuckSignal::RepeatedCall { .. }));
    let nudges = out
        .state
        .messages
        .iter()
        .filter(|m| m.role == Role::User && m.content.starts_with("[harness]"))
        .count();
    assert_eq!(
        nudges, 2,
        "default policy nudges max_nudges times, then stops"
    );
    assert_eq!(out.turns(), 9);
}

#[tokio::test]
async fn hooks_can_declare_polling_is_not_stuck() {
    struct PollingIsFine;
    #[async_trait]
    impl Hook for PollingIsFine {
        async fn on_stuck(&self, _: &HookContext<'_>, _: &StuckSignal) -> Option<StuckDecision> {
            Some(StuckDecision::Ignore)
        }
    }
    let agent = base(ScriptedProvider::new([]).then_repeat(tool_calls([call(
        "",
        "echo",
        json!({"text": "same"}),
    )])))
    .tool(echo_tool())
    .hook(PollingIsFine)
    .limits(Limits {
        max_turns: 12,
        ..Limits::default()
    })
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(
        out.status,
        RunStatus::LimitReached {
            limit: LimitKind::Turns
        }
    );
}

#[tokio::test]
async fn cancellation_is_a_status_and_stops_running_tools() {
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    let waits_forever = FnTool::new(
        spec("wait", "waits", &[]),
        ToolAnnotations::READ_ONLY,
        move |_args: Value| {
            let trigger = trigger.clone();
            async move {
                trigger.cancel();
                std::future::pending::<()>().await;
                Ok(ToolOutput::text("never"))
            }
        },
    );
    let agent = base(ScriptedProvider::new([tool_calls([call(
        "c1",
        "wait",
        json!({}),
    )])]))
    .tool(waits_forever)
    .build()
    .unwrap();

    let out = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run_with_cancel("s", "go", cancel),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(out.status, RunStatus::Cancelled);
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Cancelled);
}

#[tokio::test]
async fn provider_failure_is_a_status_not_an_error() {
    let agent = base(ScriptedProvider::with_results([Err(ProviderError::Auth(
        "bad key".into(),
    ))]))
    .build()
    .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    assert!(matches!(out.status, RunStatus::ProviderFailed { .. }));
}

#[tokio::test]
async fn retry_layer_recovers_from_transient_errors() {
    let flaky = ScriptedProvider::with_results([
        Err(ProviderError::Transient("502".into())),
        Err(ProviderError::RateLimited {
            retry_after: Some(Duration::from_millis(1)),
        }),
        Ok(text("finally")),
    ]);
    let agent = Agent::builder("a")
        .model("m")
        .provider(Retry::new(flaky).base_delay(Duration::from_millis(1)))
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    assert_eq!(out.final_text.as_deref(), Some("finally"));
}

// ---------- events and store ----------

#[tokio::test]
async fn events_are_emitted_and_state_is_checkpointed() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink_seen = seen.clone();
    let store = Arc::new(MemoryStore::new());
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "echo", json!({"text": "x"}))]),
        text("done"),
    ]))
    .tool(echo_tool())
    .store(store.clone())
    .event_sink(move |_run, e: &Event| {
        let name = serde_json::to_value(e).unwrap()["type"]
            .as_str()
            .unwrap()
            .to_string();
        sink_seen.lock().unwrap().push(name);
    })
    .build()
    .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            "run_started",
            "turn_started",
            "model_responded",
            "tool_started",
            "tool_finished",
            "turn_started",
            "model_responded",
            "finished",
        ]
    );
    assert_eq!(store.events(out.state.run_id).len(), 8);
    let saved = store
        .load_state(out.state.run_id)
        .await
        .unwrap()
        .expect("final state saved");
    assert_eq!(saved, out.state);
}

#[tokio::test]
async fn tools_get_trusted_context_separately_from_arguments() {
    let seen: Arc<Mutex<Option<(String, String)>>> = Arc::default();
    struct Spy(Arc<Mutex<Option<(String, String)>>>);
    #[async_trait]
    impl Tool for Spy {
        fn spec(&self) -> ToolSpec {
            spec("spy", "spy", &[])
        }
        fn annotations(&self) -> ToolAnnotations {
            ToolAnnotations::READ_ONLY
        }
        async fn call(&self, args: Value, ctx: &ToolContext) -> ToolResult {
            *self.0.lock().unwrap() = Some((
                ctx.session_id.clone(),
                args["session_id"].as_str().unwrap_or_default().to_string(),
            ));
            Ok(ToolOutput::text("ok"))
        }
    }
    let agent = base(ScriptedProvider::new([
        tool_calls([call("c1", "spy", json!({"session_id": "telegram:spoofed"}))]),
        text("ok"),
    ]))
    .tool(Spy(seen.clone()))
    .build()
    .unwrap();

    agent.run("telegram:synthetic-user-1", "go").await.unwrap();
    let (trusted, claimed) = seen.lock().unwrap().clone().unwrap();
    assert_eq!(trusted, "telegram:synthetic-user-1");
    assert_eq!(
        claimed, "telegram:spoofed",
        "model text never overrides ctx"
    );
}

// ---------- builder validation ----------

#[test]
fn builder_rejects_bad_configuration() {
    use aidaemon_core::BuildError;
    let missing_provider = Agent::builder("a").model("m").build();
    assert!(matches!(missing_provider, Err(BuildError::MissingProvider)));

    let dup = Agent::builder("a")
        .model("m")
        .provider(ScriptedProvider::new([]))
        .tool(echo_tool())
        .tool(echo_tool())
        .build();
    assert!(matches!(dup, Err(BuildError::DuplicateTool(name)) if name == "echo"));

    let bad_name = Agent::builder("a")
        .model("m")
        .provider(ScriptedProvider::new([]))
        .tool(FnTool::new(
            spec("bad name!", "x", &[]),
            ToolAnnotations::READ_ONLY,
            |_args: Value| async move { Ok(ToolOutput::text("")) },
        ))
        .build();
    assert!(matches!(bad_name, Err(BuildError::InvalidToolName(_))));
}
