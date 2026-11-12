//! Typed structured output against the scripted provider: the schema sent,
//! the one repair attempt, and giving up after it.

use aidaemon_sdk::structured::{complete_typed, AgentStructuredExt, StructuredError};
use aidaemon_sdk::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_sdk::{
    Agent, FnTool, Message, ModelRequest, ResponseFormat, Role, ToolAnnotations, ToolOutput,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

/// How a support ticket should be handled.
#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Triage {
    /// 1 is the most urgent.
    priority: u8,
    refund: bool,
}

fn request() -> ModelRequest {
    ModelRequest {
        model: "scripted".into(),
        messages: vec![
            Message::system("You triage tickets."),
            Message::user("The jar of order ORD-1001 arrived cracked."),
        ],
        ..Default::default()
    }
}

#[tokio::test]
async fn one_call_returns_the_typed_value_and_sends_the_schema() {
    let provider = ScriptedProvider::new([text(r#"{"priority": 2, "refund": true}"#)]);
    let typed = complete_typed::<Triage>(&provider, request())
        .await
        .unwrap();
    assert_eq!(
        typed.value,
        Triage {
            priority: 2,
            refund: true
        }
    );
    assert!(!typed.repaired);

    let sent = &provider.requests()[0];
    let ResponseFormat::JsonSchema {
        name,
        schema,
        strict,
    } = &sent.options.response_format
    else {
        panic!("no schema sent: {:?}", sent.options.response_format);
    };
    assert_eq!(name, "Triage");
    assert!(strict);
    assert_eq!(
        schema["properties"]["priority"]["description"],
        "1 is the most urgent."
    );
    // Repeated in the system prompt for providers that do not enforce it.
    assert_eq!(sent.messages[0].role, Role::System);
    assert!(sent.messages[0].content.starts_with("You triage tickets."));
    assert!(sent.messages[0].content.contains("\"priority\""));
}

#[tokio::test]
async fn a_bad_reply_gets_one_repair_with_the_parse_error() {
    let provider = ScriptedProvider::new([
        text(r#"{"priority": "high"}"#),
        text(r#"{"priority": 1, "refund": false}"#),
    ]);
    let typed = complete_typed::<Triage>(&provider, request())
        .await
        .unwrap();
    assert!(typed.repaired);
    assert_eq!(typed.value.priority, 1);
    assert_eq!(typed.usage.output_tokens, 40, "both calls counted");

    let repair = &provider.requests()[1];
    let last = repair.messages.last().unwrap();
    assert_eq!(last.role, Role::User);
    assert!(last.content.contains("invalid type"), "{}", last.content);
    let previous = &repair.messages[repair.messages.len() - 2];
    assert_eq!(previous.role, Role::Assistant);
    assert_eq!(previous.content, r#"{"priority": "high"}"#);
}

#[tokio::test]
async fn a_second_bad_reply_gives_up() {
    let provider = ScriptedProvider::new([
        text("I think it is urgent."),
        text("Still prose."),
        text(r#"{"priority": 1, "refund": false}"#),
    ]);
    let err = complete_typed::<Triage>(&provider, request())
        .await
        .unwrap_err();
    let StructuredError::Invalid { text, outcome, .. } = err else {
        panic!("{err:?}");
    };
    assert_eq!(text, "Still prose.");
    assert!(outcome.is_none());
    assert_eq!(provider.remaining(), 1, "exactly one repair attempt");
}

fn agent(provider: ScriptedProvider) -> Agent {
    let lookup = FnTool::new(
        spec("lookup_order", "Read an order", &["order_id"]),
        ToolAnnotations::READ_ONLY,
        |_args| async move {
            Ok(ToolOutput::text(
                "Acme Blender, 89.00, delivered (synthetic)",
            ))
        },
    );
    Agent::builder("triage")
        .allow_unguarded()
        .model("scripted")
        .instructions("You triage tickets.")
        .provider(provider)
        .tool(lookup)
        .build()
        .unwrap()
}

#[tokio::test]
async fn an_agent_final_answer_is_typed_after_tool_use() {
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "lookup_order", json!({"order_id": "ORD-1001"}))]),
        text("```json\n{\"priority\": 2, \"refund\": true}\n```"),
    ]));
    let agent = agent_shared(provider.clone());
    let typed = agent
        .run_typed::<Triage>("support:synthetic-1", "Cracked jar on ORD-1001")
        .await
        .unwrap();
    assert!(typed.value.refund);
    assert!(!typed.repaired);
    assert_eq!(typed.outcome.receipts().len(), 1);
    for sent in provider.requests() {
        assert!(matches!(
            sent.options.response_format,
            ResponseFormat::JsonSchema { .. }
        ));
    }
}

#[tokio::test]
async fn an_agent_answer_that_does_not_parse_is_repaired_once() {
    let provider = Arc::new(ScriptedProvider::new([
        text("Refund it, it is urgent."),
        text(r#"{"priority": 1, "refund": true}"#),
    ]));
    let agent = agent_shared(provider.clone());
    let typed = agent
        .run_typed::<Triage>("support:synthetic-2", "Cracked jar")
        .await
        .unwrap();
    assert!(typed.repaired);
    assert_eq!(typed.value.priority, 1);
    let repair = provider.requests().pop().unwrap();
    let asked = repair
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .unwrap();
    assert!(
        asked.content.contains("could not be read"),
        "{}",
        asked.content
    );
}

#[tokio::test]
async fn an_agent_gives_up_after_the_repair() {
    let agent = agent(ScriptedProvider::new([text("nope"), text("still nope")]));
    let err = agent
        .run_typed::<Triage>("support:synthetic-3", "Cracked jar")
        .await
        .unwrap_err();
    let StructuredError::Invalid { text, outcome, .. } = err else {
        panic!("{err:?}");
    };
    assert_eq!(text, "still nope");
    assert!(outcome.unwrap().is_completed());
}

#[tokio::test]
async fn an_interrupted_run_is_not_finished() {
    use aidaemon_sdk::{Hook, HookContext, InterruptKind, ToolCall, ToolDecision, ToolInfo};
    struct AlwaysAsk;
    #[async_trait::async_trait]
    impl Hook for AlwaysAsk {
        async fn before_tool(
            &self,
            _ctx: &HookContext<'_>,
            _call: &ToolCall,
            _tool: &ToolInfo,
        ) -> ToolDecision {
            ToolDecision::Interrupt {
                reason: "ask".into(),
                kind: InterruptKind::Untrusted,
            }
        }
    }
    let provider = ScriptedProvider::new([
        tool_calls([call("c1", "lookup_order", json!({"order_id": "ORD-1"}))]),
        text(r#"{"priority": 3, "refund": false}"#),
    ]);
    let agent = agent(provider).with_hook(AlwaysAsk);
    let err = agent
        .run_typed::<Triage>("support:synthetic-4", "Where is it?")
        .await
        .unwrap_err();
    let StructuredError::NotFinished(outcome) = err else {
        panic!("{err:?}");
    };
    let typed = agent
        .resume_typed::<Triage>(outcome.state, aidaemon_sdk::Resolution::Approve)
        .await
        .unwrap();
    assert_eq!(typed.value.priority, 3);
}

fn agent_shared(provider: Arc<ScriptedProvider>) -> Agent {
    let lookup = FnTool::new(
        spec("lookup_order", "Read an order", &["order_id"]),
        ToolAnnotations::READ_ONLY,
        |_args| async move { Ok(ToolOutput::text("Acme Blender, 89.00 (synthetic)")) },
    );
    Agent::builder("triage")
        .allow_unguarded()
        .model("scripted")
        .provider(provider)
        .tool(lookup)
        .build()
        .unwrap()
}
