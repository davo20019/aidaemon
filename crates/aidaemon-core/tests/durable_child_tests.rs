//! A parent run paused durably for its child's approval: the snapshot goes
//! through JSON into agents built afresh, as after a restart.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aidaemon_core::testing::{ScriptedProvider, call, spec, text, tool_calls};
use aidaemon_core::{
    Agent, ApprovalRisk, Authority, AuthorityHook, CallEffects, CoreError, EffectClass, EffectKey,
    Hook, HookContext, InterruptKind, MemoryStore, ModelResponse, OWNER_DECLINED,
    OwnerApprovalRequest, RUN_STATE_SCHEMA, ReceiptStatus, Resolution, Role, RunOutcome, RunState,
    RunStatus, RunStore, SharedDeclines, SourceTrust, Tool, ToolAnnotations, ToolCall, ToolContext,
    ToolDecision, ToolError, ToolInfo, ToolOutput, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Notify;

const SESSION: &str = "telegram:synthetic-user-1";

fn egress() -> EffectKey {
    EffectKey {
        class: EffectClass::Egress,
        target: "h.example.invalid:443".to_string(),
    }
}

/// Asks the owner before every `send`.
struct AskBeforeSend;

#[async_trait]
impl Hook for AskBeforeSend {
    fn name(&self) -> &'static str {
        "ask_before_send"
    }

    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        _tool: &ToolInfo,
    ) -> ToolDecision {
        if call.name != "send" {
            return ToolDecision::Allow;
        }
        ToolDecision::Interrupt {
            reason: "the owner must confirm this send".to_string(),
            kind: InterruptKind::OwnerApproval(OwnerApprovalRequest {
                card: None,
                effects: vec![egress()],
                summary: "Send a report for Alice Rivera".to_string(),
                subject: "POST https://h.example.invalid/report".to_string(),
                risk: ApprovalRisk::High,
                warnings: Vec::new(),
            }),
        }
    }
}

/// A tool that counts its runs and declares the egress effect.
struct Counted {
    name: &'static str,
    runs: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for Counted {
    fn spec(&self) -> ToolSpec {
        spec(self.name, "synthetic", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }
    fn source_trust(&self, _: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }
    fn effects(&self, _: &Value) -> CallEffects {
        if self.name == "send" {
            CallEffects {
                keys: vec![egress()],
                complete: true,
                understood: false,
            }
        } else {
            CallEffects::none()
        }
    }
    async fn call(&self, _: Value, _: &ToolContext) -> ToolResult {
        self.runs.fetch_add(1, Ordering::SeqCst);
        Ok(ToolOutput::text(format!("{} ok", self.name)))
    }
}

/// The smallest agent-as-tool, built only on the public contract.
struct Delegate {
    name: &'static str,
    child: Agent,
    annotations: ToolAnnotations,
}

impl Delegate {
    fn new(name: &'static str, child: Agent) -> Self {
        Self {
            name,
            child,
            annotations: ToolAnnotations::WRITE,
        }
    }

    fn scoped(&self, ctx: &ToolContext) -> Agent {
        match &ctx.authority {
            Some(authority) => self.child.with_hook(AuthorityHook::new(authority.clone())),
            None => self.child.clone(),
        }
    }

    fn conclude(outcome: RunOutcome) -> ToolResult {
        match outcome.status {
            RunStatus::Interrupted { .. } => Ok(ToolOutput::suspended(outcome.state)),
            RunStatus::Completed => Ok(ToolOutput::text(format!(
                "child said: {}",
                outcome.final_text.unwrap_or_default()
            ))),
            status => Ok(ToolOutput::error(format!("child ended as {status:?}"))),
        }
    }
}

#[async_trait]
impl Tool for Delegate {
    fn spec(&self) -> ToolSpec {
        spec(self.name, "delegates", &["task"])
    }
    fn annotations(&self) -> ToolAnnotations {
        self.annotations
    }
    fn source_trust(&self, _: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }
    fn suspends(&self) -> bool {
        true
    }
    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        let session = format!("{}/sub/{}", ctx.session_id, ctx.call_id);
        let task = arguments["task"].as_str().unwrap_or_default();
        let agent = self.scoped(ctx);
        let mut state = RunState::new(agent.id(), &session, task).with_parent(ctx.run_id);
        if let Some(declines) = &ctx.declines {
            state = state.share_declines(declines.clone());
        }
        let outcome = agent
            .run_from(state, ctx.cancel.clone())
            .await
            .map_err(|e| ToolError(e.to_string()))?;
        Self::conclude(outcome)
    }
    async fn resume_child(
        &self,
        mut child: RunState,
        resolution: Resolution,
        ctx: &ToolContext,
    ) -> ToolResult {
        if let Some(declines) = &ctx.declines {
            child = child.share_declines(declines.clone());
        }
        let outcome = self
            .scoped(ctx)
            .resume_with_cancel(child, resolution, ctx.cancel.clone())
            .await
            .map_err(|e| ToolError(e.to_string()))?;
        Self::conclude(outcome)
    }
}

fn delegate(id: &str) -> ToolCall {
    call(id, "delegate", json!({"task": "send the report"}))
}

fn scripted(replies: Vec<ModelResponse>) -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::new(replies))
}

/// Agents as a process would rebuild them: same ids and tools, its own
/// providers, and counters kept across "restarts" by the test.
#[derive(Clone, Default)]
struct Counters {
    send: Arc<AtomicUsize>,
    ping: Arc<AtomicUsize>,
}

fn leaf_agent(id: &str, provider: Arc<ScriptedProvider>, counters: &Counters) -> Agent {
    Agent::builder(id)
        .allow_unguarded()
        .model("m")
        .provider(provider)
        .tool(Counted {
            name: "send",
            runs: counters.send.clone(),
        })
        .tool(Counted {
            name: "ping",
            runs: counters.ping.clone(),
        })
        .hook(AskBeforeSend)
        .build()
        .unwrap()
}

fn parent_agent(provider: Arc<ScriptedProvider>, child: Agent, counters: &Counters) -> Agent {
    Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(provider)
        .tool(Delegate::new("delegate", child))
        .tool(Counted {
            name: "send",
            runs: counters.send.clone(),
        })
        .tool(Counted {
            name: "ping",
            runs: counters.ping.clone(),
        })
        .build()
        .unwrap()
}

/// Save the run and read it back, as a process that exits and restarts does.
fn through_json(state: &RunState) -> RunState {
    serde_json::from_str(&serde_json::to_string(state).unwrap()).unwrap()
}

fn tool_result(provider: &ScriptedProvider, call_id: &str) -> String {
    provider
        .requests()
        .iter()
        .flat_map(|r| r.messages.clone())
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some(call_id))
        .map(|m| m.content)
        .expect("tool result present")
}

fn approve() -> Resolution {
    Resolution::Approve
}

fn decline() -> Resolution {
    Resolution::Deny {
        reason: "Jordan Lee said no".to_string(),
    }
}

/// The parent and child of the common case, paused on the child's `send`.
async fn paused_on_child_send(counters: &Counters) -> RunState {
    let child = leaf_agent(
        "child",
        scripted(vec![tool_calls([call("k1", "send", json!({}))])]),
        counters,
    );
    let parent = parent_agent(
        scripted(vec![tool_calls([delegate("c1")])]),
        child,
        counters,
    );
    let out = parent.run(SESSION, "go").await.unwrap();
    assert!(
        matches!(out.status, RunStatus::Interrupted { .. }),
        "{:?}",
        out.status
    );
    out.state
}

#[tokio::test]
async fn an_answer_after_a_restart_resumes_the_child_and_completes_the_parents_call() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;

    // The parent's interruption shows what would really run.
    let pending = state.pending.as_ref().unwrap();
    assert_eq!(pending.interruption.call.name, "send");
    assert_eq!(pending.calls[pending.next].name, "delegate");
    assert!(matches!(
        pending.interruption.kind,
        InterruptKind::OwnerApproval(_)
    ));
    let waiting = state.waiting_run().unwrap();
    assert_eq!(waiting.agent_id, "child");
    assert_eq!(waiting.session_id, format!("{SESSION}/sub/c1"));
    assert_eq!(waiting.parent_run, Some(state.run_id));
    assert_eq!(state.schema(), RUN_STATE_SCHEMA);
    assert_eq!(state.tool_calls, 0, "the delegating call has no result yet");
    assert_eq!(counters.send.load(Ordering::SeqCst), 0);

    let saved = through_json(&state);
    assert_eq!(saved, through_json(&saved), "the snapshot is stable");

    let child_provider = scripted(vec![text("Report sent for Alice Rivera.")]);
    let parent_provider = scripted(vec![text("All done.")]);
    let parent = parent_agent(
        parent_provider.clone(),
        leaf_agent("child", child_provider.clone(), &counters),
        &counters,
    );
    let out = parent.resume(saved, approve()).await.unwrap();

    assert!(out.is_succeeded(), "{:?}", out.status);
    assert_eq!(out.final_text.as_deref(), Some("All done."));
    assert_eq!(counters.send.load(Ordering::SeqCst), 1);
    assert_eq!(
        tool_result(&parent_provider, "c1"),
        "child said: Report sent for Alice Rivera."
    );
    assert_eq!(out.receipts().len(), 1);
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded);
    assert_eq!(out.state.pending, None);
    assert_eq!(
        tool_result(&child_provider, "k1"),
        "send ok",
        "the child's model sees its approved call"
    );
}

#[tokio::test]
async fn an_answer_after_a_restart_reaches_a_grandchild() {
    let counters = Counters::default();
    let leaf = leaf_agent(
        "leaf",
        scripted(vec![tool_calls([call("k1", "send", json!({}))])]),
        &counters,
    );
    let mid = Agent::builder("mid")
        .allow_unguarded()
        .model("m")
        .provider(scripted(vec![tool_calls([delegate("m1")])]))
        .tool(Delegate::new("delegate", leaf))
        .build()
        .unwrap();
    let parent = parent_agent(scripted(vec![tool_calls([delegate("c1")])]), mid, &counters);

    let out = parent.run(SESSION, "go").await.unwrap();
    assert!(matches!(out.status, RunStatus::Interrupted { .. }));
    let saved = through_json(&out.state);
    let waiting = saved.waiting_run().unwrap();
    assert_eq!(waiting.agent_id, "leaf");
    assert_eq!(waiting.session_id, format!("{SESSION}/sub/c1/sub/m1"));
    let mid_state = saved.pending.as_ref().unwrap().child.as_ref().unwrap();
    assert_eq!(mid_state.agent_id, "mid");
    assert_eq!(mid_state.waiting_run().unwrap().agent_id, "leaf");

    let leaf_provider = scripted(vec![text("sent")]);
    let mid_provider = scripted(vec![text("mid finished")]);
    let parent_provider = scripted(vec![text("top finished")]);
    let mid = Agent::builder("mid")
        .allow_unguarded()
        .model("m")
        .provider(mid_provider.clone())
        .tool(Delegate::new(
            "delegate",
            leaf_agent("leaf", leaf_provider, &counters),
        ))
        .build()
        .unwrap();
    let parent = parent_agent(parent_provider.clone(), mid, &counters);
    let out = parent.resume(saved, approve()).await.unwrap();

    assert!(out.is_succeeded(), "{:?}", out.status);
    assert_eq!(counters.send.load(Ordering::SeqCst), 1);
    assert_eq!(tool_result(&mid_provider, "m1"), "child said: sent");
    assert_eq!(
        tool_result(&parent_provider, "c1"),
        "child said: mid finished"
    );
}

#[tokio::test]
async fn a_child_that_asks_again_pauses_the_parent_again() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;

    let parent = parent_agent(
        scripted(vec![]),
        leaf_agent(
            "child",
            scripted(vec![tool_calls([call("k2", "send", json!({}))])]),
            &counters,
        ),
        &counters,
    );
    let out = parent
        .resume(through_json(&state), approve())
        .await
        .unwrap();
    assert!(matches!(out.status, RunStatus::Interrupted { .. }));
    assert_eq!(counters.send.load(Ordering::SeqCst), 1);
    assert_eq!(
        out.state.pending.as_ref().unwrap().interruption.call.id,
        "k2"
    );
    assert_eq!(out.state.waiting_run().unwrap().run_receipts().count(), 1);

    let parent_provider = scripted(vec![text("done")]);
    let parent = parent_agent(
        parent_provider.clone(),
        leaf_agent("child", scripted(vec![text("twice")]), &counters),
        &counters,
    );
    let done = parent
        .resume(through_json(&out.state), approve())
        .await
        .unwrap();
    assert!(done.is_succeeded(), "{:?}", done.status);
    assert_eq!(
        counters.send.load(Ordering::SeqCst),
        2,
        "the first approved send did not run again"
    );
    assert_eq!(tool_result(&parent_provider, "c1"), "child said: twice");
}

#[tokio::test]
async fn a_decline_fails_the_childs_call_and_binds_the_parent() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;

    let child_provider = scripted(vec![text("I did not send it.")]);
    // The parent tries the declined send itself afterwards.
    let parent_provider = scripted(vec![
        tool_calls([call("c2", "send", json!({}))]),
        text("Not sent."),
    ]);
    let parent = parent_agent(
        parent_provider.clone(),
        leaf_agent("child", child_provider.clone(), &counters),
        &counters,
    );
    let out = parent
        .resume(through_json(&state), decline())
        .await
        .unwrap();

    assert_eq!(counters.send.load(Ordering::SeqCst), 0);
    let seen = tool_result(&child_provider, "k1");
    assert!(seen.starts_with("Denied:"), "{seen}");
    assert!(seen.contains(OWNER_DECLINED), "{seen}");
    assert!(seen.contains("Jordan Lee said no"), "{seen}");
    assert_eq!(
        tool_result(&parent_provider, "c1"),
        "child said: I did not send it."
    );

    assert_eq!(out.state.declined, vec![egress()]);
    assert_eq!(out.receipts()[1].status, ReceiptStatus::Denied);
    assert!(tool_result(&parent_provider, "c2").contains(OWNER_DECLINED));
    assert!(
        matches!(out.status, RunStatus::Completed) && !out.is_succeeded(),
        "a run with a decline never ends succeeded: {:?}",
        out.status
    );
}

#[tokio::test]
async fn a_decline_in_one_run_of_the_family_reaches_the_resumed_child() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;
    // The parent's record holds a decline from before the pause.
    let mut saved = through_json(&state);
    saved.declined.push(egress());

    let parent = parent_agent(
        scripted(vec![text("handled")]),
        leaf_agent("child", scripted(vec![text("could not")]), &counters),
        &counters,
    );
    let out = parent.resume(saved, approve()).await.unwrap();

    assert_eq!(
        counters.send.load(Ordering::SeqCst),
        0,
        "an approval does not get past a decline the family already made"
    );
    assert!(!out.is_succeeded());
}

#[tokio::test]
async fn a_resumed_child_stays_under_the_parents_current_authority() {
    let counters = Counters::default();
    // Started under full authority, paused on the child's send.
    let child = leaf_agent(
        "child",
        scripted(vec![tool_calls([call("k1", "send", json!({}))])]),
        &counters,
    );
    let parent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(scripted(vec![tool_calls([delegate("c1")])]))
        .tool(Delegate::new("delegate", child))
        .hook(AuthorityHook::new(Authority::full()))
        .build()
        .unwrap();
    let out = parent.run(SESSION, "go").await.unwrap();
    assert!(matches!(out.status, RunStatus::Interrupted { .. }));

    // After the restart the parent may only read. The approval settles the
    // hook that asked, and the authority stacked after it still judges.
    let child_provider = scripted(vec![text("not allowed")]);
    let parent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(scripted(vec![text("handled")]))
        .tool(Delegate::new(
            "delegate",
            leaf_agent("child", child_provider.clone(), &counters),
        ))
        .hook(AuthorityHook::new(Authority::read_only()))
        .build()
        .unwrap();
    let out = parent
        .resume(through_json(&out.state), approve())
        .await
        .unwrap();

    assert_eq!(counters.send.load(Ordering::SeqCst), 0);
    assert!(tool_result(&child_provider, "k1").starts_with("Denied:"));
    assert!(matches!(out.status, RunStatus::Completed));
}

#[tokio::test]
async fn the_childs_recorded_calls_do_not_run_again_and_the_parents_batch_continues() {
    let counters = Counters::default();
    let child = leaf_agent(
        "child",
        scripted(vec![tool_calls([
            call("k0", "ping", json!({})),
            call("k1", "send", json!({})),
        ])]),
        &counters,
    );
    let parent = parent_agent(
        scripted(vec![tool_calls([
            call("c0", "ping", json!({})),
            delegate("c1"),
            call("c2", "ping", json!({})),
        ])]),
        child,
        &counters,
    );
    let out = parent.run(SESSION, "go").await.unwrap();
    assert!(matches!(out.status, RunStatus::Interrupted { .. }));
    assert_eq!(
        counters.ping.load(Ordering::SeqCst),
        2,
        "c0 and the child's k0 ran; c2 waits"
    );

    let parent_provider = scripted(vec![text("done")]);
    let parent = parent_agent(
        parent_provider.clone(),
        leaf_agent("child", scripted(vec![text("sent")]), &counters),
        &counters,
    );
    let out = parent
        .resume(through_json(&out.state), approve())
        .await
        .unwrap();

    assert!(out.is_succeeded(), "{:?}", out.status);
    assert_eq!(counters.send.load(Ordering::SeqCst), 1);
    assert_eq!(counters.ping.load(Ordering::SeqCst), 3, "only c2 is new");
    let order: Vec<&str> = out.receipts().iter().map(|r| r.call_id.as_str()).collect();
    assert_eq!(order, ["c0", "c1", "c2"]);
    assert_eq!(tool_result(&parent_provider, "c2"), "ping ok");
}

/// A send that never returns: the process dies inside it.
struct HangingSend {
    entered: Arc<Notify>,
    runs: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for HangingSend {
    fn spec(&self) -> ToolSpec {
        spec("send", "synthetic", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::WRITE
    }
    async fn call(&self, _: Value, _: &ToolContext) -> ToolResult {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.entered.notify_one();
        std::future::pending().await
    }
}

#[tokio::test]
async fn a_crash_after_the_answer_does_not_replay_the_answer_or_the_child() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;

    let store = Arc::new(MemoryStore::new());
    let entered = Arc::new(Notify::new());
    let child = Agent::builder("child")
        .allow_unguarded()
        .model("m")
        .provider(scripted(vec![text("never")]))
        .tool(HangingSend {
            entered: entered.clone(),
            runs: counters.send.clone(),
        })
        .hook(AskBeforeSend)
        .build()
        .unwrap();
    let parent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(scripted(vec![]))
        .tool(Delegate::new("delegate", child))
        .store(store.clone())
        .build()
        .unwrap();
    let run_id = state.run_id;
    let resumed = tokio::spawn({
        let saved = through_json(&state);
        async move { parent.resume(saved, approve()).await }
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .expect("the approved call started");
    resumed.abort();
    let _ = resumed.await;

    // What the process left on disk: the answer is consumed.
    let saved = store.load_state(run_id).await.unwrap().unwrap();
    assert!(saved.pending.is_none());
    let in_flight = saved.in_flight.as_ref().unwrap();
    assert_eq!(in_flight.started, ["c1"]);
    assert!(in_flight.done.is_empty());

    let parent_provider = scripted(vec![text("I will check whether it was sent.")]);
    let child_provider = scripted(vec![text("never")]);
    let parent = parent_agent(
        parent_provider.clone(),
        leaf_agent("child", child_provider.clone(), &counters),
        &counters,
    );
    let err = parent
        .resume(through_json(&saved), approve())
        .await
        .unwrap_err();
    assert!(
        matches!(err, CoreError::NotInterrupted),
        "the answer applies once: {err:?}"
    );

    let out = parent
        .run_from(
            through_json(&saved),
            aidaemon_core::CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(
        matches!(out.status, RunStatus::Completed),
        "{:?}",
        out.status
    );
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Indeterminate);
    assert!(tool_result(&parent_provider, "c1").contains("may or may not have completed"));
    assert_eq!(
        counters.send.load(Ordering::SeqCst),
        1,
        "the child's call did not run a second time"
    );
    assert!(child_provider.requests().is_empty(), "the child is over");
}

#[tokio::test]
async fn a_tool_that_does_not_declare_suspension_fails_closed() {
    struct Undeclared(RunState);
    #[async_trait]
    impl Tool for Undeclared {
        fn spec(&self) -> ToolSpec {
            spec("sneaky", "synthetic", &[])
        }
        async fn call(&self, _: Value, _: &ToolContext) -> ToolResult {
            Ok(ToolOutput::suspended(self.0.clone()))
        }
    }

    let counters = Counters::default();
    let paused = paused_on_child_send(&counters).await;
    let provider = scripted(vec![
        tool_calls([call("c1", "sneaky", json!({}))]),
        text("noted"),
    ]);
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(provider.clone())
        .tool(Undeclared(paused))
        .build()
        .unwrap();
    let out = agent.run(SESSION, "go").await.unwrap();
    assert!(matches!(out.status, RunStatus::Completed));
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Failed);
    assert!(out.state.pending.is_none());
}

#[tokio::test]
async fn a_suspension_without_a_paused_run_is_a_failed_call() {
    struct NotPaused;
    #[async_trait]
    impl Tool for NotPaused {
        fn spec(&self) -> ToolSpec {
            spec("delegate", "synthetic", &[])
        }
        fn suspends(&self) -> bool {
            true
        }
        async fn call(&self, _: Value, _: &ToolContext) -> ToolResult {
            Ok(ToolOutput::suspended(RunState::new("c", "s", "x")))
        }
    }
    let agent = Agent::builder("a")
        .allow_unguarded()
        .model("m")
        .provider(scripted(vec![
            tool_calls([call("c1", "delegate", json!({}))]),
            text("noted"),
        ]))
        .tool(NotPaused)
        .build()
        .unwrap();
    let out = agent.run(SESSION, "go").await.unwrap();
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Failed);
    assert!(matches!(out.status, RunStatus::Completed));
}

#[tokio::test]
async fn a_tool_gone_after_the_restart_is_a_failed_call() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;
    let agent = Agent::builder("parent")
        .allow_unguarded()
        .model("m")
        .provider(scripted(vec![text("handled")]))
        .tool(Counted {
            name: "ping",
            runs: counters.ping.clone(),
        })
        .build()
        .unwrap();
    let out = agent.resume(through_json(&state), approve()).await.unwrap();
    assert_eq!(out.receipts()[0].status, ReceiptStatus::UnknownTool);
    assert_eq!(counters.send.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_child_saved_by_a_newer_kernel_is_refused_before_anything_runs() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;
    let mut json = serde_json::to_value(&state).unwrap();
    json["pending"]["child"]["schema"] = json!(RUN_STATE_SCHEMA + 1);
    let saved: RunState = serde_json::from_value(json).unwrap();

    let parent = parent_agent(
        scripted(vec![]),
        leaf_agent("child", scripted(vec![]), &counters),
        &counters,
    );
    let err = parent.resume(saved, approve()).await.unwrap_err();
    assert!(
        matches!(err, CoreError::UnsupportedRunState { found, .. } if found == RUN_STATE_SCHEMA + 1),
        "{err:?}"
    );
    assert_eq!(counters.send.load(Ordering::SeqCst), 0);
}

/// A run interrupted before `Pending::child` existed, as schema 1 wrote it.
const SCHEMA_1_SNAPSHOT: &str = r#"{
  "schema": 1,
  "run_id": "5a6fb472-9745-4963-b4ba-8cdfe101a252",
  "agent_id": "a",
  "session_id": "telegram:synthetic-user-1",
  "messages": [
    {"role": "user", "content": "Send the report.", "origin": "user"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {"id": "c0", "name": "ping", "arguments": {}},
        {"id": "c1", "name": "send", "arguments": {}}
      ]
    },
    {"role": "tool", "content": "ping ok", "tool_call_id": "c0"}
  ],
  "receipts": [
    {
      "run_id": "5a6fb472-9745-4963-b4ba-8cdfe101a252",
      "call_id": "c0",
      "tool": "ping",
      "arguments": {},
      "turn": 1,
      "stage": "dispatched",
      "status": "succeeded",
      "read_only": false,
      "duration_ms": 0,
      "source": "trusted",
      "after_untrusted": false
    }
  ],
  "usage": {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 0, "cache_creation_tokens": 0},
  "run_usage": {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 0, "cache_creation_tokens": 0},
  "turn": 1,
  "tool_calls": 1,
  "elapsed_ms": 0,
  "pending": {
    "calls": [
      {"id": "c0", "name": "ping", "arguments": {}},
      {"id": "c1", "name": "send", "arguments": {}}
    ],
    "next": 1,
    "interruption": {
      "call": {"id": "c1", "name": "send", "arguments": {}},
      "reason": "the owner must confirm this send",
      "kind": {
        "owner_approval": {
          "effects": [{"class": "egress", "target": "h.example.invalid:443"}],
          "summary": "Send a report for Alice Rivera",
          "subject": "POST https://h.example.invalid/report",
          "risk": "high"
        }
      }
    },
    "hook_index": 0,
    "hook_chain": ["ask_before_send"]
  },
  "nudges": 0,
  "untrusted_seen": false,
  "declined": [],
  "limit_grants": {"turns": 0, "tokens": 0, "tool_calls": 0, "wall_clock_ms": 0},
  "idle_continuations": 0,
  "continued_at_successes": 0
}"#;

#[tokio::test]
async fn a_snapshot_saved_at_the_old_schema_still_resumes() {
    let saved: RunState = serde_json::from_str(SCHEMA_1_SNAPSHOT).unwrap();
    assert_eq!(saved.schema(), 1);
    let pending = saved.pending.as_ref().unwrap();
    assert_eq!(pending.child, None);
    assert_eq!(saved.waiting_run().unwrap().run_id, saved.run_id);

    let counters = Counters::default();
    let provider = scripted(vec![text("Sent for Alice Rivera.")]);
    let agent = leaf_agent("a", provider.clone(), &counters);
    // Saved before approvals carried a digest: nothing binds the approval
    // to the call, so the call is judged again and the owner asked again.
    let out = agent.resume(saved, approve()).await.unwrap();
    let RunStatus::Interrupted { interruption } = &out.status else {
        panic!("an unbound approval is asked again: {:?}", out.status);
    };
    assert_eq!(interruption.call.id, "c1");
    assert!(interruption.digest.is_some());
    assert_eq!(counters.send.load(Ordering::SeqCst), 0);
    let out = agent.resume(out.state, approve()).await.unwrap();

    assert!(out.is_succeeded(), "{:?}", out.status);
    assert_eq!(out.final_text.as_deref(), Some("Sent for Alice Rivera."));
    assert_eq!(counters.send.load(Ordering::SeqCst), 1);
    assert_eq!(
        counters.ping.load(Ordering::SeqCst),
        0,
        "the earlier call in the batch already has its result"
    );
    assert_eq!(out.state.schema(), RUN_STATE_SCHEMA, "written back current");
}

#[tokio::test]
async fn shared_declines_are_not_part_of_the_snapshot() {
    let counters = Counters::default();
    let state = paused_on_child_send(&counters).await;
    let child = state.pending.as_ref().unwrap().child.as_ref().unwrap();
    assert_eq!(child.shared_declines(), None::<&SharedDeclines>);
}
