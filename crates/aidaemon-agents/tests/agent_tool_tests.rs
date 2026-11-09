//! Agent-as-tool behavior against the real loop, with scripted providers.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_agents::{AgentExt, AgentTool, ChildBudget, TranscriptTap};
use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, AgentBuilder, CancellationToken, FnTool, Hook, HookContext, InterruptResolver,
    Interruption, ReceiptStatus, Resolution, Role, RunState, RunStatus, Tool, ToolAnnotations,
    ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::{Barrier, Notify};

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

fn child_builder(provider: Arc<ScriptedProvider>) -> AgentBuilder {
    Agent::builder("researcher")
        .allow_unguarded()
        .model("test-model")
        .instructions("You research things.")
        .provider(provider)
}

fn parent(provider: Arc<ScriptedProvider>, tool: AgentTool) -> Agent {
    Agent::builder("orchestrator")
        .allow_unguarded()
        .model("test-model")
        .provider(provider)
        .tool(tool)
        .build()
        .unwrap()
}

fn delegate(id: &str, task: &str) -> ToolCall {
    call(id, "research", json!({"task": task}))
}

/// The tool message the parent's model saw for `call_id`.
fn tool_message(provider: &ScriptedProvider, call_id: &str) -> String {
    provider
        .requests()
        .iter()
        .flat_map(|r| r.messages.clone())
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some(call_id))
        .map(|m| m.content)
        .expect("tool result present")
}

#[tokio::test]
async fn completed_child_result_is_condensed_with_details() {
    let long: String = format!("START{}END", "x".repeat(10_000));
    let child_provider = Arc::new(ScriptedProvider::new([text(long)]));
    let tool = AgentTool::from_builder(
        child_builder(child_provider.clone()),
        ChildBudget::default(),
    )
    .unwrap()
    .name("research")
    .max_result_chars(300);
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "Find the release date")]),
        text("done"),
    ]));
    let out = parent(parent_provider.clone(), tool)
        .run("telegram:synthetic-user-1", "go")
        .await
        .unwrap();
    assert!(out.is_completed());

    let seen = tool_message(&parent_provider, "c1");
    assert!(seen.len() < 500, "{} chars", seen.len());
    assert!(seen.starts_with("START"));
    assert!(seen.ends_with("END"));
    assert!(seen.contains("characters of sub-agent output omitted"));

    let receipt = &out.receipts()[0];
    assert_eq!(receipt.status, ReceiptStatus::Succeeded);
    assert_eq!(receipt.details["status"]["status"], "completed");
    assert_eq!(receipt.details["turns"], 1);
    assert_eq!(receipt.details["succeeded"], true);
    assert_eq!(
        receipt.details["child_session_id"],
        "telegram:synthetic-user-1/sub/c1"
    );
    assert!(receipt.details["child_run_id"].is_string());

    // The child saw only the task (Brief mode), in its own session.
    let child_request = &child_provider.requests()[0];
    let user = child_request
        .messages
        .iter()
        .find(|m| m.role == Role::User)
        .unwrap();
    assert_eq!(user.content, "Find the release date");
}

#[tokio::test]
async fn child_that_stops_with_unmet_requirements_reports_them() {
    struct Unmet;
    #[async_trait]
    impl Hook for Unmet {
        async fn on_stop(&self, _: &HookContext<'_>, _: &str) -> aidaemon_core::StopDecision {
            aidaemon_core::StopDecision::AcceptUnmet {
                unmet: vec!["checklist: [-] cite a primary source (deferred: paywalled)".into()],
            }
        }
    }
    let child_provider = Arc::new(ScriptedProvider::new([text("It shipped in May.")]));
    let tool = AgentTool::from_builder(
        child_builder(child_provider).hook(Unmet),
        ChildBudget::default(),
    )
    .unwrap()
    .name("research");
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "Find the release date with a source")]),
        text("done"),
    ]));
    let out = parent(parent_provider.clone(), tool)
        .run("telegram:synthetic-user-1", "go")
        .await
        .unwrap();

    let seen = tool_message(&parent_provider, "c1");
    assert!(seen.contains("stopped without finishing"), "{seen}");
    assert!(seen.contains("cite a primary source"));
    assert!(seen.contains("It shipped in May."));
    let receipt = &out.receipts()[0];
    assert_eq!(receipt.details["succeeded"], false);
    assert_eq!(receipt.details["unmet"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn child_usage_rolls_up_into_parent_outcome() {
    let child_provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("k1", "echo", json!({"text": "fact"}))]),
        text("found: fact"),
    ]));
    let child = child_builder(child_provider)
        .tool(echo_tool())
        .build()
        .unwrap();
    let tool = child.as_tool("research", "Research a question");
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([call(
            "c1",
            "research",
            json!({"task": "look it up", "context": "Alice Rivera asked"}),
        )]),
        text("done"),
    ]));
    let out = parent(parent_provider, tool).run("s", "go").await.unwrap();
    assert!(out.is_completed());
    // Parent: 2 turns x 120; child: 2 turns x 120.
    assert_eq!(out.usage().billable_tokens(), 480);
    let receipt = &out.receipts()[0];
    assert_eq!(receipt.details["usage"]["input_tokens"], 200);
    assert_eq!(
        receipt.details["receipts"],
        json!([{"tool": "echo", "status": "succeeded"}])
    );
}

#[tokio::test]
async fn parent_token_budget_covers_child_spend() {
    let child_provider = Arc::new(ScriptedProvider::new([]).then_repeat(text("child answer")));
    let tool = AgentTool::from_builder(child_builder(child_provider), ChildBudget::default())
        .unwrap()
        .name("research");
    let parent_provider =
        Arc::new(ScriptedProvider::new([]).then_repeat(tool_calls([delegate("", "again")])));
    let agent = Agent::builder("orchestrator")
        .allow_unguarded()
        .model("m")
        .provider(parent_provider)
        .tool(tool)
        .limits(aidaemon_core::Limits {
            max_tokens: Some(400),
            ..Default::default()
        })
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    // Each parent turn costs 120 + 120 for the child: the cap trips after 2.
    assert!(matches!(out.status, RunStatus::LimitReached { .. }));
    assert_eq!(out.turns(), 2);
}

async fn run_failing_child(child: AgentTool) -> (String, aidaemon_core::Receipt) {
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "do it")]),
        text("handled"),
    ]));
    let out = parent(parent_provider.clone(), child.name("research"))
        .run("s", "go")
        .await
        .unwrap();
    assert!(
        out.is_completed(),
        "the parent keeps going: {:?}",
        out.status
    );
    (
        tool_message(&parent_provider, "c1"),
        out.receipts()[0].clone(),
    )
}

#[tokio::test]
async fn limit_reached_child_is_an_error() {
    let child_provider = Arc::new(ScriptedProvider::new([]).then_repeat(tool_calls([call(
        "k",
        "echo",
        json!({"text": "a"}),
    )])));
    let budget = ChildBudget {
        max_turns: 2,
        ..ChildBudget::default()
    };
    let tool =
        AgentTool::from_builder(child_builder(child_provider).tool(echo_tool()), budget).unwrap();
    let (seen, receipt) = run_failing_child(tool).await;
    assert_eq!(receipt.status, ReceiptStatus::Failed);
    assert!(
        seen.contains("did not complete: limit reached (turns)"),
        "{seen}"
    );
    assert_eq!(receipt.details["status"]["status"], "limit_reached");
}

#[tokio::test]
async fn provider_failed_child_is_an_error() {
    let child_provider = Arc::new(ScriptedProvider::new([])); // exhausted at once
    let tool =
        AgentTool::from_builder(child_builder(child_provider), ChildBudget::default()).unwrap();
    let (seen, receipt) = run_failing_child(tool).await;
    assert_eq!(receipt.status, ReceiptStatus::Failed);
    assert!(seen.contains("provider failed"), "{seen}");
}

#[tokio::test]
async fn stuck_child_is_an_error() {
    let child_provider = Arc::new(ScriptedProvider::new([]).then_repeat(tool_calls([call(
        "k",
        "echo",
        json!({"text": "same"}),
    )])));
    let tool = AgentTool::from_builder(
        child_builder(child_provider).tool(echo_tool()),
        ChildBudget::default(),
    )
    .unwrap();
    let (seen, receipt) = run_failing_child(tool).await;
    assert_eq!(receipt.status, ReceiptStatus::Failed);
    assert!(seen.contains("stuck"), "{seen}");
    assert_eq!(receipt.details["status"]["status"], "stuck");
}

struct AskFirst;

#[async_trait]
impl Hook for AskFirst {
    async fn before_tool(&self, _: &HookContext<'_>, _: &ToolCall, _: &ToolInfo) -> ToolDecision {
        ToolDecision::Interrupt {
            reason: "needs approval".into(),
            kind: aidaemon_core::InterruptKind::default(),
        }
    }
}

#[tokio::test]
async fn interrupted_child_explains_sync_mode_limitation() {
    let child_provider = Arc::new(ScriptedProvider::new([tool_calls([call(
        "k",
        "echo",
        json!({"text": "a"}),
    )])]));
    let tool = AgentTool::from_builder(
        child_builder(child_provider)
            .tool(echo_tool())
            .hook(AskFirst),
        ChildBudget::default(),
    )
    .unwrap();
    let (seen, receipt) = run_failing_child(tool).await;
    assert_eq!(receipt.status, ReceiptStatus::Failed);
    assert!(seen.contains("not supported in synchronous"), "{seen}");
    assert!(seen.contains("`echo`"), "{seen}");
}

/// Answers every interruption the same way, and records whose it was.
struct Answers {
    answer: Option<Resolution>,
    sessions: Mutex<Vec<String>>,
}

impl Answers {
    fn new(answer: Option<Resolution>) -> Arc<Self> {
        Arc::new(Self {
            answer,
            sessions: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl InterruptResolver for Answers {
    async fn resolve(&self, state: &RunState, _: &Interruption) -> Option<Resolution> {
        self.sessions.lock().unwrap().push(state.session_id.clone());
        self.answer.clone()
    }
}

/// A child that asks before `echo`, then replies; and how often `echo` ran.
fn asking_child(reply: &str) -> (AgentTool, Arc<ScriptedProvider>, Arc<Mutex<u32>>) {
    child_replying([
        tool_calls([call("k", "echo", json!({"text": "a"}))]),
        text(reply),
    ])
}

/// The same child as a process rebuilds it to answer an interruption: its
/// first model turn is already in the saved state.
fn resumed_child(reply: &str) -> (AgentTool, Arc<ScriptedProvider>, Arc<Mutex<u32>>) {
    child_replying([text(reply)])
}

fn child_replying<const N: usize>(
    replies: [aidaemon_core::ModelResponse; N],
) -> (AgentTool, Arc<ScriptedProvider>, Arc<Mutex<u32>>) {
    let runs = Arc::new(Mutex::new(0));
    let counted = {
        let runs = runs.clone();
        FnTool::new(
            spec("echo", "Echo the text back", &["text"]),
            ToolAnnotations::READ_ONLY,
            move |_args: Value| {
                let runs = runs.clone();
                async move {
                    *runs.lock().unwrap() += 1;
                    Ok(ToolOutput::text("echoed"))
                }
            },
        )
    };
    let provider = Arc::new(ScriptedProvider::new(replies));
    let tool = AgentTool::from_builder(
        child_builder(provider.clone()).tool(counted).hook(AskFirst),
        ChildBudget::default(),
    )
    .unwrap();
    (tool, provider, runs)
}

#[tokio::test]
async fn an_approved_child_keeps_going_inside_the_parents_call() {
    let (tool, _, runs) = asking_child("Echo finished for Alice Rivera.");
    let answers = Answers::new(Some(Resolution::Approve));
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "do it")]),
        text("handled"),
    ]));
    let out = parent(
        parent_provider.clone(),
        tool.name("research").interrupt_resolver(answers.clone()),
    )
    .run("s", "go")
    .await
    .unwrap();

    assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded);
    assert_eq!(*runs.lock().unwrap(), 1);
    assert!(tool_message(&parent_provider, "c1").contains("Echo finished for Alice Rivera."));
    assert_eq!(*answers.sessions.lock().unwrap(), vec!["s/sub/c1"]);
}

#[tokio::test]
async fn a_denied_child_is_told_and_its_call_does_not_run() {
    let (tool, child_provider, runs) = asking_child("I did not echo.");
    let answers = Answers::new(Some(Resolution::Deny {
        reason: "Jordan Lee said no".into(),
    }));
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "do it")]),
        text("handled"),
    ]));
    let out = parent(
        parent_provider.clone(),
        tool.name("research").interrupt_resolver(answers),
    )
    .run("s", "go")
    .await
    .unwrap();

    assert!(out.is_completed());
    assert_eq!(*runs.lock().unwrap(), 0);
    assert!(tool_message(&child_provider, "k").contains("Jordan Lee said no"));
    assert!(tool_message(&parent_provider, "c1").contains("I did not echo."));
}

#[tokio::test]
async fn a_child_left_unsettled_is_an_error_and_its_call_does_not_run() {
    let (tool, _, runs) = asking_child("never");
    let (seen, receipt) = run_failing_child(tool.interrupt_resolver(Answers::new(None))).await;
    assert_eq!(receipt.status, ReceiptStatus::Failed);
    assert!(seen.contains("it was not settled"), "{seen}");
    assert!(seen.contains("`echo`"), "{seen}");
    assert_eq!(*runs.lock().unwrap(), 0);
}

/// Save the run and read it back, as a process that exits and restarts does.
fn through_json(state: &RunState) -> RunState {
    serde_json::from_str(&serde_json::to_string(state).unwrap()).unwrap()
}

#[tokio::test]
async fn a_suspending_child_pauses_the_parent_and_a_later_answer_finishes_it() {
    let (tool, _, runs) = asking_child("unused");
    let out = parent(
        Arc::new(ScriptedProvider::new([tool_calls([delegate(
            "c1", "do it",
        )])])),
        tool.name("research").suspend_on_interrupt(),
    )
    .run("s", "go")
    .await
    .unwrap();

    let RunStatus::Interrupted { interruption } = &out.status else {
        panic!("{:?}", out.status);
    };
    assert_eq!(interruption.call.name, "echo");
    assert_eq!(*runs.lock().unwrap(), 0);
    let saved = through_json(&out.state);
    assert_eq!(saved.waiting_run().unwrap().session_id, "s/sub/c1");

    let (tool, _, runs) = resumed_child("Echo finished for Alice Rivera.");
    let parent_provider = Arc::new(ScriptedProvider::new([text("handled")]));
    let agent = parent(
        parent_provider.clone(),
        tool.name("research").suspend_on_interrupt(),
    );
    let out = agent.resume(saved, Resolution::Approve).await.unwrap();

    assert!(out.is_completed(), "{:?}", out.status);
    assert_eq!(*runs.lock().unwrap(), 1);
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded);
    assert_eq!(out.receipts()[0].details["child_session_id"], "s/sub/c1");
    assert!(tool_message(&parent_provider, "c1").contains("Echo finished for Alice Rivera."));
}

#[tokio::test]
async fn a_suspended_childs_decline_is_told_to_the_child() {
    let (tool, _, runs) = asking_child("unused");
    let out = parent(
        Arc::new(ScriptedProvider::new([tool_calls([delegate(
            "c1", "do it",
        )])])),
        tool.name("research").suspend_on_interrupt(),
    )
    .run("s", "go")
    .await
    .unwrap();
    let saved = through_json(&out.state);

    let (tool, child_provider, _) = resumed_child("I did not echo.");
    let parent_provider = Arc::new(ScriptedProvider::new([text("handled")]));
    let agent = parent(
        parent_provider.clone(),
        tool.name("research").suspend_on_interrupt(),
    );
    let out = agent
        .resume(
            saved,
            Resolution::Deny {
                reason: "Jordan Lee said no".into(),
            },
        )
        .await
        .unwrap();

    assert_eq!(*runs.lock().unwrap(), 0);
    assert!(tool_message(&child_provider, "k").contains("Jordan Lee said no"));
    assert!(tool_message(&parent_provider, "c1").contains("I did not echo."));
    assert!(out.is_completed());
}

#[tokio::test]
async fn the_live_resolver_still_settles_first_and_only_what_it_leaves_suspends() {
    // Settled live: no pause, the parent's call just returns.
    let (tool, _, runs) = asking_child("Finished live.");
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "do it")]),
        text("handled"),
    ]));
    let out = parent(
        parent_provider.clone(),
        tool.name("research")
            .interrupt_resolver(Answers::new(Some(Resolution::Approve)))
            .suspend_on_interrupt(),
    )
    .run("s", "go")
    .await
    .unwrap();
    assert!(out.is_completed());
    assert_eq!(*runs.lock().unwrap(), 1);
    assert!(tool_message(&parent_provider, "c1").contains("Finished live."));

    // Left unsettled: the parent pauses rather than reporting an error.
    let (tool, _, runs) = asking_child("unused");
    let out = parent(
        Arc::new(ScriptedProvider::new([tool_calls([delegate(
            "c1", "do it",
        )])])),
        tool.name("research")
            .interrupt_resolver(Answers::new(None))
            .suspend_on_interrupt(),
    )
    .run("s", "go")
    .await
    .unwrap();
    assert!(matches!(out.status, RunStatus::Interrupted { .. }));
    assert!(out.state.pending.as_ref().unwrap().child.is_some());
    assert_eq!(*runs.lock().unwrap(), 0);
}

/// Never answers.
struct NeverAnswers;

#[async_trait]
impl InterruptResolver for NeverAnswers {
    async fn resolve(&self, _: &RunState, _: &Interruption) -> Option<Resolution> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn a_cancelled_parent_stops_waiting_for_an_answer() {
    let (tool, _, runs) = asking_child("never");
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "do it")]),
        text("unused"),
    ]));
    let agent = parent(
        parent_provider,
        tool.name("research")
            .interrupt_resolver(Arc::new(NeverAnswers)),
    );
    let cancel = CancellationToken::new();
    let run = tokio::spawn({
        let cancel = cancel.clone();
        async move { agent.run_with_cancel("s", "go", cancel).await }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancel.cancel();
    let out = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("the parent stops")
        .unwrap()
        .unwrap();
    assert!(
        matches!(out.status, RunStatus::Cancelled),
        "{:?}",
        out.status
    );
    assert_eq!(*runs.lock().unwrap(), 0);
}

#[tokio::test]
async fn depth_limit_refuses_runaway_recursion() {
    let child_provider = Arc::new(ScriptedProvider::new([text("never")]));
    let tool = AgentTool::from_builder(
        child_builder(child_provider.clone()),
        ChildBudget::default(),
    )
    .unwrap()
    .name("research")
    .max_depth(2);
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c3", "deeper")]),
        text("ok"),
    ]));
    let out = parent(parent_provider.clone(), tool)
        .run("s/sub/c1/sub/c2", "go")
        .await
        .unwrap();
    let receipt = &out.receipts()[0];
    assert_eq!(receipt.status, ReceiptStatus::Failed);
    assert_eq!(receipt.details["refused"], "max_depth");
    assert!(tool_message(&parent_provider, "c3").contains("depth 3"));
    assert!(child_provider.requests().is_empty(), "child never ran");
}

#[tokio::test]
async fn read_only_research_children_run_concurrently() {
    let barrier = Arc::new(Barrier::new(2));
    let b = barrier.clone();
    let meet = FnTool::new(
        spec("meet", "wait for a peer", &[]),
        ToolAnnotations::READ_ONLY,
        move |_args: Value| {
            let b = b.clone();
            async move {
                b.wait().await;
                Ok(ToolOutput::text("met"))
            }
        },
    );
    // Both children call `meet` before either can finish, so the script
    // order is deterministic: two tool turns, then two answers.
    let child_provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("m", "meet", json!({}))]),
        tool_calls([call("m", "meet", json!({}))]),
        text("finding"),
        text("finding"),
    ]));
    let tool = AgentTool::from_builder(
        child_builder(child_provider).tool(meet),
        ChildBudget::default(),
    )
    .unwrap()
    .name("research")
    .read_only();
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "topic A"), delegate("c2", "topic B")]),
        text("merged"),
    ]));
    let agent = parent(parent_provider, tool);
    let out = tokio::time::timeout(Duration::from_secs(5), agent.run("s", "go"))
        .await
        .expect("read-only children should run concurrently")
        .unwrap();
    assert!(out.is_completed());
    assert!(out.receipts().iter().all(|r| r.succeeded()));
}

#[test]
fn children_are_not_read_only_by_default() {
    let agent = child_builder(Arc::new(ScriptedProvider::new([])))
        .build()
        .unwrap();
    assert_eq!(AgentTool::new(agent).annotations(), ToolAnnotations::WRITE);
}

/// Parks until cancelled, recording the token it was given.
struct Park {
    seen: Arc<Mutex<Option<CancellationToken>>>,
    started: Arc<Notify>,
}

#[async_trait]
impl Tool for Park {
    fn spec(&self) -> ToolSpec {
        spec("park", "wait", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }
    async fn call(&self, _args: Value, ctx: &ToolContext) -> ToolResult {
        *self.seen.lock().unwrap() = Some(ctx.cancel.clone());
        self.started.notify_one();
        ctx.cancel.cancelled().await;
        Ok(ToolOutput::text("woke"))
    }
}

#[tokio::test]
async fn cancellation_propagates_to_the_child() {
    let seen = Arc::new(Mutex::new(None));
    let started = Arc::new(Notify::new());
    let child_provider = Arc::new(ScriptedProvider::new([tool_calls([call(
        "p",
        "park",
        json!({}),
    )])]));
    let tool = AgentTool::from_builder(
        child_builder(child_provider).tool(Park {
            seen: seen.clone(),
            started: started.clone(),
        }),
        ChildBudget::default(),
    )
    .unwrap()
    .name("research");
    let parent_provider = Arc::new(ScriptedProvider::new([tool_calls([delegate(
        "c1", "wait",
    )])]));
    let agent = parent(parent_provider, tool);
    let cancel = CancellationToken::new();
    let run = tokio::spawn({
        let cancel = cancel.clone();
        async move { agent.run_with_cancel("s", "go", cancel).await }
    });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("child tool started");
    cancel.cancel();
    let out = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("parent stops promptly")
        .unwrap()
        .unwrap();
    assert_eq!(out.status, RunStatus::Cancelled);
    let token = seen.lock().unwrap().clone().expect("child tool ran");
    assert!(
        token.is_cancelled(),
        "the child's tool saw the cancellation"
    );
}

#[tokio::test]
async fn fork_mode_passes_a_caller_supplied_transcript_summary() {
    let child_provider = Arc::new(ScriptedProvider::new([text("ok")]));
    let tap = TranscriptTap::default();
    let tool = AgentTool::from_builder(
        child_builder(child_provider.clone()),
        ChildBudget::default(),
    )
    .unwrap()
    .name("research")
    .fork_from(tap.source());
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "summarize the plan")]),
        text("done"),
    ]));
    let agent = Agent::builder("orchestrator")
        .allow_unguarded()
        .model("m")
        .provider(parent_provider)
        .hook(tap)
        .tool(tool)
        .build()
        .unwrap();
    let out = agent
        .run("s", "Plan a trip for Jordan Lee to Lisbon")
        .await
        .unwrap();
    assert!(out.is_completed());
    let input = child_provider.requests()[0]
        .messages
        .iter()
        .find(|m| m.role == Role::User)
        .unwrap()
        .content
        .clone();
    assert!(
        input.contains("user: Plan a trip for Jordan Lee to Lisbon"),
        "{input}"
    );
    assert!(input.contains("[Task]\nsummarize the plan"), "{input}");
}

#[tokio::test]
async fn invalid_arguments_are_rejected_before_the_child_runs() {
    let child_provider = Arc::new(ScriptedProvider::new([text("never")]));
    let tool = AgentTool::from_builder(
        child_builder(child_provider.clone()),
        ChildBudget::default(),
    )
    .unwrap()
    .name("research");
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "research", json!({"task": "  "}))]),
        text("ok"),
    ]));
    let out = parent(parent_provider, tool).run("s", "go").await.unwrap();
    assert_eq!(out.receipts()[0].status, ReceiptStatus::InvalidArguments);
    assert!(child_provider.requests().is_empty());
}

/// A child a tool finished: the finish reply is the answer, partial work
/// says so, and a blocked child is an error the parent reacts to.
#[tokio::test]
async fn finished_child_returns_its_finish_reply() {
    use aidaemon_core::Finish;
    for (finish, error, expected) in [
        (
            Finish::succeeded("Filed the synthetic report."),
            false,
            "Filed the synthetic report.",
        ),
        (
            Finish::partial("Filed half; the rest runs later."),
            false,
            "finished part of the task:\nFiled half; the rest runs later.",
        ),
        (
            Finish::blocked("Needs the owner's sign-off."),
            true,
            "is blocked:\nNeeds the owner's sign-off.",
        ),
    ] {
        let finisher = FnTool::new(
            spec("file_report", "Files the report", &[]),
            ToolAnnotations::READ_ONLY,
            move |_: Value| {
                let finish = finish.clone();
                async move { Ok(ToolOutput::text("filed").finish(finish)) }
            },
        );
        let child_provider = Arc::new(ScriptedProvider::new([tool_calls([call(
            "f1",
            "file_report",
            json!({}),
        )])]));
        let tool = AgentTool::from_builder(
            child_builder(child_provider).tool(finisher),
            ChildBudget::default(),
        )
        .unwrap();
        let parent_provider = Arc::new(ScriptedProvider::new([
            tool_calls([delegate("c1", "file it")]),
            text("handled"),
        ]));
        let out = parent(parent_provider.clone(), tool.name("research"))
            .run("s", "go")
            .await
            .unwrap();
        assert!(out.is_completed());
        let seen = tool_message(&parent_provider, "c1");
        assert!(seen.contains(expected), "{seen}");
        let receipt = &out.receipts()[0];
        assert_eq!(
            receipt.status == ReceiptStatus::Failed,
            error,
            "{:?}",
            receipt.status
        );
    }
}

// ---------- what a child inherits ----------

fn write_tool(name: &'static str, calls: Arc<std::sync::atomic::AtomicUsize>) -> impl Tool {
    FnTool::new(
        spec(name, "Changes something", &[]),
        ToolAnnotations::default(),
        move |_: Value| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(ToolOutput::text("written"))
            }
        },
    )
}

#[tokio::test]
async fn a_child_cannot_do_what_its_parent_may_not() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let child = Arc::new(ScriptedProvider::new([
        tool_calls([call("w1", "write", json!({}))]),
        text("tried"),
    ]));
    let tool = AgentTool::new(
        child_builder(child)
            .tool(write_tool("write", calls.clone()))
            .build()
            .unwrap(),
    )
    .name("research")
    .description("Research")
    .read_only();
    let parent_provider = Arc::new(ScriptedProvider::new([
        tool_calls([delegate("c1", "change it")]),
        text("done"),
    ]));
    let agent = Agent::builder("orchestrator")
        .allow_unguarded()
        .model("test-model")
        .provider(parent_provider)
        .hook(aidaemon_core::AuthorityHook::new(
            aidaemon_core::Authority::read_only(),
        ))
        .tool(tool)
        .build()
        .unwrap();

    agent.run("s", "go").await.unwrap();

    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_child_names_the_run_that_started_it() {
    let child_events: Arc<Mutex<Vec<aidaemon_core::Event>>> = Arc::default();
    let sink = child_events.clone();
    let child = Arc::new(ScriptedProvider::new([text("found")]));
    let tool = AgentTool::new(
        child_builder(child)
            .event_sink(move |_, e: &aidaemon_core::Event| sink.lock().unwrap().push(e.clone()))
            .build()
            .unwrap(),
    )
    .name("research")
    .description("Research");
    let agent = parent(
        Arc::new(ScriptedProvider::new([
            tool_calls([delegate("c1", "look")]),
            text("done"),
        ])),
        tool,
    );

    let out = agent.run("s", "go").await.unwrap();

    let parent_run = out.state.run_id;
    assert!(child_events.lock().unwrap().iter().any(|e| matches!(
        e,
        aidaemon_core::Event::RunLinked { parent_run: p } if *p == parent_run
    )));
}

struct DeclinesFromInside;
#[async_trait]
impl Tool for DeclinesFromInside {
    fn spec(&self) -> ToolSpec {
        spec("decline", "Records a decline", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }
    async fn call(&self, _args: Value, ctx: &ToolContext) -> ToolResult {
        let effect = aidaemon_core::EffectKey::named(aidaemon_core::EffectClass::Call, "x");
        ctx.declines.as_ref().unwrap().record(&[effect]);
        Ok(ToolOutput::text("declined"))
    }
}

#[tokio::test]
async fn a_tool_holds_the_runs_live_decline_set() {
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("test-model")
        .provider(Arc::new(ScriptedProvider::new([
            tool_calls([call("c1", "decline", json!({}))]),
            text("done"),
        ])))
        .tool(DeclinesFromInside)
        .build()
        .unwrap();

    let out = agent.run("s", "go").await.unwrap();

    assert_eq!(out.state.declined.len(), 1);
}
