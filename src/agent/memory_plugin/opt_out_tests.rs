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
    sqlx::query_scalar(sqlx::AssertSqlSafe(format!("SELECT COUNT(*) FROM {table}")))
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
    set_next_message_private(harness.state.as_ref(), SESSION, None, true)
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
    set_next_message_private(harness.state.as_ref(), SESSION, None, true)
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
        set_next_message_private(harness.state.as_ref(), SESSION, None, true)
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
    set_next_message_private(harness.state.as_ref(), SESSION, None, true)
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

/// What the private turn said: it may live on in the stored history, but
/// once the conversation moves on no reader may hand it to a prompt.
const PRIVATE_DETAIL: &str = "Quillo-11";

#[tokio::test]
async fn after_clear_a_private_turn_reaches_no_prompt_or_history_search() {
    let harness = crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
        MockProvider::text_response("I can use that here, but I won't save it."),
        MockProvider::text_response("I don't know your test gecko's name."),
        MockProvider::text_response("Your test hamster is Plimbo-31."),
    ]))
    .await
    .unwrap();
    set_next_message_private(harness.state.as_ref(), SESSION, None, true)
        .await
        .unwrap();
    harness
        .agent
        .handle_inbound_message(request(&format!(
            "My test gecko is named {PRIVATE_DETAIL}. Please remember it."
        )))
        .await
        .unwrap();
    // A normal turn after it, which history search may still return.
    harness
        .agent
        .handle_inbound_message(request("My test hamster is named Plimbo-31."))
        .await
        .unwrap();
    harness.agent.clear_session_context(SESSION).await.unwrap();
    harness
        .agent
        .handle_inbound_message(request("Do you know my test gecko's name?"))
        .await
        .unwrap();

    let log = harness.provider.call_log.lock().await;
    let last = serde_json::to_string(&log.last().unwrap().messages).unwrap();
    assert!(
        !last.contains(PRIVATE_DETAIL),
        "the private turn reached a later prompt"
    );
    drop(log);

    // The model's history search, as it runs in that later turn.
    let history = crate::tools::SearchHistoryTool::new(harness.state.clone(), 90);
    let facts = crate::traits::ToolCallContext {
        session_id: SESSION.to_string(),
        user_role: Some(UserRole::Owner),
        channel_visibility: Some(crate::types::ChannelVisibility::Private),
        ..Default::default()
    };
    for query in ["gecko", PRIVATE_DETAIL] {
        let args =
            serde_json::json!({"action": "search", "query": query, "limit": 10, "context": 1});
        let found = crate::testing::call_with_facts(&history, &args.to_string(), facts.clone())
            .await
            .unwrap();
        // The query is echoed back; the private message must not be.
        assert!(
            !found.contains(&format!("named {PRIVATE_DETAIL}")),
            "history search: {found}"
        );
    }
    let args = serde_json::json!({"action": "search", "query": "hamster", "limit": 10});
    let found = crate::testing::call_with_facts(&history, &args.to_string(), facts)
        .await
        .unwrap();
    assert!(
        found.contains("Plimbo-31"),
        "a normal turn stays searchable: {found}"
    );
}

/// Fails its first mark, then succeeds; counts each attempt.
#[derive(Default)]
struct FlakyRecorder(Mutex<u32>);

#[async_trait]
impl aidaemon_policies::OptOutRecorder for FlakyRecorder {
    async fn record_opt_out(&self) -> Result<(), String> {
        let mut attempts = self.0.lock().unwrap();
        *attempts += 1;
        if *attempts == 1 {
            Err("synthetic storage fault".to_string())
        } else {
            Ok(())
        }
    }
}

/// The turn's opt-out flag flips once, but its mark is retried until it is
/// recorded: a mark that failed once must not leave the events unmarked.
#[tokio::test]
async fn a_failed_opt_out_mark_is_retried_until_recorded() {
    use aidaemon_core::Tool;

    let recorder = Arc::new(FlakyRecorder::default());
    let learning = TurnLearning::default();
    let tool = super::opt_out::SkipLearningTool::with_recorder(learning.clone(), recorder.clone());
    let ctx = aidaemon_core::ToolContext::new(SESSION, "call-1");

    assert!(tool.call(serde_json::json!({}), &ctx).await.is_err());
    assert!(!learning.allowed(), "the turn is opted out even so");
    assert!(tool.call(serde_json::json!({}), &ctx).await.is_ok());
    assert_eq!(
        *recorder.0.lock().unwrap(),
        2,
        "the failed mark was retried"
    );
    assert!(tool.call(serde_json::json!({}), &ctx).await.is_ok());
    assert_eq!(
        *recorder.0.lock().unwrap(),
        2,
        "a recorded mark is not repeated"
    );
}

/// Policy events of every turn so far: (access, reason).
async fn policy_events(harness: &TestHarness) -> Vec<(String, String)> {
    sqlx::query_as(
        "SELECT json_extract(data, '$.access'), json_extract(data, '$.reason_code')
           FROM events WHERE event_type = 'memory_policy_compiled' ORDER BY id",
    )
    .fetch_all(&harness.state.pool())
    .await
    .unwrap()
}

/// A `/private` turn records one policy event, already suppressed with the
/// opt-out's reason: no window where background intake sees it as allowed.
#[tokio::test]
async fn a_private_turn_records_one_suppressed_policy_event() {
    let harness = crate::testing::setup_test_agent_root(MockProvider::with_responses(vec![
        MockProvider::text_response("I won't keep that."),
    ]))
    .await
    .unwrap();
    set_next_message_private(harness.state.as_ref(), SESSION, None, true)
        .await
        .unwrap();
    harness
        .agent
        .handle_inbound_message(request("My synthetic code word is Zephyr-4."))
        .await
        .unwrap();
    assert_eq!(
        policy_events(&harness).await,
        [(
            "suppressed".to_string(),
            super::opt_out::OPTED_OUT_REASON.to_string()
        )]
    );
}

/// A stand-in `spawn_agent` that records the learning handle the
/// dispatcher hands a child in the parent authority fact.
struct LearningProbe(Arc<Mutex<Vec<Option<TurnLearning>>>>);

#[async_trait]
impl crate::traits::Tool for LearningProbe {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Test probe: records a delegation."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "name": "spawn_agent",
            "description": "Test probe: records a delegation.",
            "parameters": {
                "type": "object",
                "properties": {"mission": {"type": "string"}, "task": {"type": "string"}}
            }
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        self.0.lock().unwrap().push(
            crate::agent::authority::parent_authority(facts)
                .and_then(|parent| parent.learning.clone()),
        );
        Ok(crate::traits::ToolCallOutcome::from_output(
            "delegated".to_string(),
        ))
    }
}

impl crate::traits::ToolSemantics for LearningProbe {
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            spawns_child_agents: true,
            ..Default::default()
        }
    }

    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: true,
            idempotent: true,
            ..Default::default()
        }
    }
}

/// A sub-agent started during a private turn is opted out too: the
/// dispatcher hands it the parent turn's learning, and a turn run with that
/// learning learns nothing and marks its own events.
#[tokio::test]
async fn a_sub_agent_of_a_private_turn_is_opted_out_too() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let harness = crate::testing::setup_test_agent_root_with_extra_tools_and_llm_timeout(
        MockProvider::with_responses(vec![
            MockProvider::tool_call_response(
                "spawn_agent",
                r#"{"mission":"Look","task":"Look at the synthetic notes."}"#,
            ),
            MockProvider::text_response("Delegated."),
        ]),
        vec![Arc::new(LearningProbe(seen.clone()))],
        None,
    )
    .await
    .unwrap();
    set_next_message_private(harness.state.as_ref(), SESSION, None, true)
        .await
        .unwrap();
    harness
        .agent
        .handle_inbound_message(request("Have a helper look at my synthetic notes."))
        .await
        .unwrap();
    let handed = seen.lock().unwrap().clone();
    assert_eq!(handed.len(), 1, "spawn_agent ran once");
    let handed = handed[0]
        .clone()
        .expect("the child gets the turn's learning");
    assert!(!handed.allowed(), "the child is opted out");

    // The child's own turn, run with the handle it was given: it would learn
    // a procedure from two tool calls, and does not.
    let mut child = crate::testing::setup_test_agent_root(script(false))
        .await
        .unwrap();
    child
        .agent
        .set_test_inherited_authority(crate::agent::authority::InheritedAuthority {
            authority: aidaemon_core::Authority::full(),
            owner_level: Some(crate::config::OwnerLevel::Full),
            task_access: None,
            mode_session: Some(SESSION.to_string()),
            untrusted_context: false,
            origin_request: None,
            origin_trusted: true,
            trusted_roots: Vec::new(),
            declined: Vec::new(),
            shared_declines: None,
            learning: Some(handed),
            spawn_family: None,
            child_declaration: None,
            parent_run: None,
            known_resources: Default::default(),
            card_title: None,
            card_steps: Vec::new(),
            lineage: Vec::new(),
            parent_task: None,
        });
    child
        .agent
        .handle_inbound_message(request(
            "Check my system, and note my favorite color is teal.",
        ))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    for table in MEMORY_TABLES {
        assert_eq!(rows(&child, table).await, 0, "{table} was written");
    }
    assert_eq!(opted_out_markers(&child).await, 1);
}
