//! Agent-as-tool behavior against the real loop, with scripted providers.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_agents::{AgentExt, AgentTool, ChildBudget, TranscriptTap};
use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, AgentBuilder, CancellationToken, FnTool, Hook, HookContext, ReceiptStatus, Role,
    RunStatus, Tool, ToolAnnotations, ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput,
    ToolResult, ToolSpec,
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
        .model("test-model")
        .instructions("You research things.")
        .provider(provider)
}

fn parent(provider: Arc<ScriptedProvider>, tool: AgentTool) -> Agent {
    Agent::builder("orchestrator")
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
