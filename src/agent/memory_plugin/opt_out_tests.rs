//! "Don't remember this" end to end: what an opted-out turn writes, what it
//! still reads, and what a replacing memory system is told.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_policies::{Learning, MemoryParts, MemorySystem, MemoryTurn, TurnLearning};
use async_trait::async_trait;

use super::opt_out::SKIP_LEARNING_TOOL;
use super::{set_next_message_private, SystemMemory};
use crate::runtime_ports::{AgentIngress, InboundMessageRequest};
use crate::testing::{MockProvider, TestHarness};
use crate::traits::store_prelude::*;
use crate::types::{ChannelContext, FactPrivacy, UserRole};

const SESSION: &str = "telegram:synthetic-user-1";

const MEMORY_TABLES: &[&str] = &[
    "facts",
    "episodes",
    "people",
    "procedures",
    "expertise",
    "error_solutions",
    "behavior_patterns",
    "memory_claims",
];

fn request(text: &str) -> InboundMessageRequest {
    InboundMessageRequest {
        session_id: SESSION.to_string(),
        user_text: text.to_string(),
        attachments: Vec::new(),
        status_tx: None,
        user_role: UserRole::Owner,
        channel_ctx: ChannelContext::private("telegram"),
        heartbeat: None,
        ingress_timing: None,
    }
}

/// A turn that calls a read-only tool twice, so a normal turn learns a
/// procedure from it; `opt_out` makes the model call `skip_learning` first.
fn script(opt_out: bool) -> MockProvider {
    let mut responses = Vec::new();
    if opt_out {
        responses.push(MockProvider::tool_call_response(SKIP_LEARNING_TOOL, "{}"));
    }
    responses.extend([
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response("Your machine looks fine. I'll keep your teal in mind."),
    ]);
    MockProvider::with_responses(responses)
}

async fn rows(harness: &TestHarness, table: &str) -> i64 {
    sqlx::query_scalar(&format!("SELECT COUNT(*) FROM {table}"))
        .fetch_one(&harness.state.pool())
        .await
        .unwrap()
}

async fn opted_out_markers(harness: &TestHarness) -> i64 {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM events WHERE event_type = 'memory_policy_compiled'
           AND json_extract(data, '$.access') = 'suppressed'
           AND json_extract(data, '$.reason_code') = 'learning_opted_out'",
    )
    .fetch_one(&harness.state.pool())
    .await
    .unwrap()
}

async fn run_turn(opt_out: bool) -> TestHarness {
    let harness = crate::testing::setup_test_agent_root(script(opt_out))
        .await
        .unwrap();
    harness
        .agent
        .handle_inbound_message(request(
            "Check my system, and note my favorite color is teal.",
        ))
        .await
        .unwrap();
    // Learning runs in the background after a turn.
    tokio::time::sleep(Duration::from_millis(300)).await;
    harness
}

#[tokio::test]
async fn a_turn_the_model_opts_out_writes_no_memory() {
    let normal = run_turn(false).await;
    assert!(
        rows(&normal, "procedures").await > 0,
        "a normal turn learns"
    );
    assert_eq!(opted_out_markers(&normal).await, 0);

    let opted_out = run_turn(true).await;
    for table in MEMORY_TABLES {
        assert_eq!(rows(&opted_out, table).await, 0, "{table} was written");
    }
    assert_eq!(opted_out_markers(&opted_out).await, 1);
}

#[tokio::test]
async fn private_opts_out_the_next_message_and_recall_still_works() {
    let mut harness = crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
        MockProvider::text_response("Noted, Alice."),
        MockProvider::text_response("Hi again."),
    ]))
    .await
    .unwrap();
    let writes = Arc::new(super::writer::tests::RecordingWrites::default());
    harness.agent.set_test_transcript_writes(writes.clone());
    harness
        .state
        .upsert_fact(
            "user",
            "name",
            "Alice Rivera",
            "user",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    set_next_message_private(harness.state.as_ref(), SESSION, true)
        .await
        .unwrap();

    harness
        .agent
        .handle_inbound_message(request("My salary is synthetic. What's my name?"))
        .await
        .unwrap();
    assert!(writes.0.lock().unwrap().learned.is_empty());
    assert_eq!(opted_out_markers(&harness).await, 1);
    let log = harness.provider.call_log.lock().await;
    let first = serde_json::to_string(&log[0].messages).unwrap();
    assert!(
        first.contains("Alice Rivera"),
        "recall on the opted-out turn"
    );
    let offered = serde_json::to_string(&log[0].tools).unwrap();
    assert!(
        !offered.contains(SKIP_LEARNING_TOOL),
        "already opted out: no tool"
    );
    drop(log);

    // `/private` covers one message.
    harness
        .agent
        .handle_inbound_message(request("Thanks, that's all."))
        .await
        .unwrap();
    assert_eq!(writes.0.lock().unwrap().learned.len(), 1);
    assert_eq!(opted_out_markers(&harness).await, 1);
}

/// Records the learning handle each turn is given.
#[derive(Default)]
struct WatchingMemory(Mutex<Vec<TurnLearning>>);

#[async_trait]
impl MemorySystem for WatchingMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        self.0.lock().unwrap().push(turn.learning.clone());
        MemoryParts::default()
    }
}

#[tokio::test]
async fn a_replacing_memory_system_is_told_the_turn_is_opted_out() {
    let mut harness = crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
        MockProvider::tool_call_response(SKIP_LEARNING_TOOL, "{}"),
        MockProvider::text_response("I won't keep that."),
        MockProvider::text_response("Hello."),
        MockProvider::text_response("Hello again."),
    ]))
    .await
    .unwrap();
    let system = Arc::new(WatchingMemory::default());
    harness
        .agent
        .set_memory_plugin(Arc::new(SystemMemory::new(system.clone())));

    // The model opts out during the turn: the system's handle follows.
    harness
        .agent
        .handle_inbound_message(request("Please don't remember this one."))
        .await
        .unwrap();
    // A normal turn.
    harness
        .agent
        .handle_inbound_message(request("Hello."))
        .await
        .unwrap();
    // `/private` before the message: opted out from the start.
    set_next_message_private(harness.state.as_ref(), SESSION, true)
        .await
        .unwrap();
    harness
        .agent
        .handle_inbound_message(request("Hello once more."))
        .await
        .unwrap();

    let seen: Vec<Learning> = system.0.lock().unwrap().iter().map(|l| l.get()).collect();
    assert_eq!(
        seen,
        [Learning::OptedOut, Learning::Allowed, Learning::OptedOut]
    );
    let log = harness.provider.call_log.lock().await;
    let offered = serde_json::to_string(&log[0].tools).unwrap();
    assert!(
        offered.contains(SKIP_LEARNING_TOOL),
        "offered under any memory system"
    );
}

/// Whether any later model request carries the private-turn refusal.
async fn model_was_told_private(harness: &TestHarness) -> bool {
    let log = harness.provider.call_log.lock().await;
    log.iter().skip(1).any(|call| {
        serde_json::to_string(&call.messages)
            .unwrap()
            .contains(super::opt_out::PRIVATE_TURN_STORE_REFUSAL)
    })
}

async fn remember_after(private: bool) -> TestHarness {
    let harness = crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"personal","key":"test_lizard_name","value":"Quibbit-9"}"#,
        ),
        MockProvider::text_response("This message is private, so I won't store it."),
    ]))
    .await
    .unwrap();
    if private {
        set_next_message_private(harness.state.as_ref(), SESSION, true)
            .await
            .unwrap();
    }
    harness
        .agent
        .handle_inbound_message(request("My synthetic lizard is named Quibbit-9."))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    harness
}

async fn lizard_facts(harness: &TestHarness) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM facts WHERE value LIKE '%Quibbit-9%'")
        .fetch_one(&harness.state.pool())
        .await
        .unwrap()
}

/// `/private` covers the explicit memory tools too (live 2026-11-12: the
/// model stored the fact with `remember_fact` and recalled it after
/// `/clear`). The kernel refuses the store with a reason the model relays.
#[tokio::test]
async fn an_opted_out_turn_cannot_store_through_a_memory_tool() {
    let normal = remember_after(false).await;
    assert_eq!(lizard_facts(&normal).await, 1, "a normal turn stores");
    assert!(!model_was_told_private(&normal).await);

    let private = remember_after(true).await;
    assert_eq!(lizard_facts(&private).await, 0, "the private turn stored");
    assert!(model_was_told_private(&private).await);
}

/// A memory system's tool, counting its calls.
struct CountingMemoryTool {
    name: &'static str,
    read_only: bool,
    calls: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl aidaemon_core::Tool for CountingMemoryTool {
    fn spec(&self) -> aidaemon_core::ToolSpec {
        aidaemon_core::ToolSpec {
            name: self.name.into(),
            description: "Synthetic memory tool".into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    fn annotations(&self) -> aidaemon_core::ToolAnnotations {
        if self.read_only {
            aidaemon_core::ToolAnnotations::READ_ONLY
        } else {
            aidaemon_core::ToolAnnotations::WRITE
        }
    }

    async fn call(
        &self,
        _arguments: serde_json::Value,
        _ctx: &aidaemon_core::ToolContext,
    ) -> aidaemon_core::ToolResult {
        self.calls.lock().unwrap().push(self.name);
        Ok(aidaemon_core::ToolOutput::text("ok"))
    }
}

/// A memory system that offers one store and one recall tool.
struct ToolMemory(Arc<Mutex<Vec<&'static str>>>);

#[async_trait]
impl MemorySystem for ToolMemory {
    async fn turn(&self, _turn: &MemoryTurn) -> MemoryParts {
        MemoryParts {
            hooks: Vec::new(),
            tools: vec![
                Arc::new(CountingMemoryTool {
                    name: "synthetic_memory_store",
                    read_only: false,
                    calls: self.0.clone(),
                }),
                Arc::new(CountingMemoryTool {
                    name: "synthetic_memory_recall",
                    read_only: true,
                    calls: self.0.clone(),
                }),
            ],
        }
    }
}

#[tokio::test]
async fn an_opted_out_turn_cannot_store_through_a_memory_systems_tool() {
    let mut harness = crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
        MockProvider::tool_call_response("synthetic_memory_recall", "{}"),
        MockProvider::tool_call_response("synthetic_memory_store", "{}"),
        MockProvider::text_response("This message is private, so I won't store it."),
    ]))
    .await
    .unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    harness
        .agent
        .set_memory_plugin(Arc::new(SystemMemory::new(Arc::new(ToolMemory(
            calls.clone(),
        )))));
    set_next_message_private(harness.state.as_ref(), SESSION, true)
        .await
        .unwrap();
    harness
        .agent
        .handle_inbound_message(request("Remember my synthetic code word."))
        .await
        .unwrap();

    assert_eq!(*calls.lock().unwrap(), ["synthetic_memory_recall"]);
    assert!(model_was_told_private(&harness).await);
}
