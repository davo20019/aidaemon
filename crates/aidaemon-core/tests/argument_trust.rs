//! What the model sends is checked before any hook or person sees it:
//! `_`-prefixed keys are dropped (they are the dispatcher's, never the
//! model's), and arguments must match the types the tool declared.

use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, FnTool, Hook, HookContext, ReceiptStatus, Tool, ToolAnnotations, ToolCall, ToolDecision,
    ToolInfo, ToolOutput, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};

type Seen = Arc<Mutex<Vec<Value>>>;

/// Records every call a `before_tool` hook is shown.
struct Watch(Seen);

#[async_trait]
impl Hook for Watch {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        call: &ToolCall,
        _: &ToolInfo,
    ) -> ToolDecision {
        self.0.lock().unwrap().push(call.arguments.clone());
        ToolDecision::Allow
    }
}

fn write_tool(received: &Seen) -> impl Tool {
    let received = received.clone();
    FnTool::new(
        ToolSpec {
            name: "write_note".into(),
            description: "Write a note".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "lines": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["append", "replace"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "meta": {
                        "type": "object",
                        "properties": {"author": {"type": "string"}},
                        "required": ["author"]
                    }
                },
                "required": ["path", "content"]
            }),
        },
        ToolAnnotations::WRITE,
        move |args: Value| {
            let received = received.clone();
            async move {
                received.lock().unwrap().push(args);
                Ok(ToolOutput::text("ok"))
            }
        },
    )
}

async fn run(arguments: Value) -> (aidaemon_core::RunOutcome, Seen, Seen) {
    let hooked: Seen = Arc::default();
    let received: Seen = Arc::default();
    let agent = Agent::builder("args")
        .model("m")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "write_note", arguments)]),
            text("done"),
        ]))
        .tool(write_tool(&received))
        .hook(Watch(hooked.clone()))
        .allow_unguarded()
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    (out, hooked, received)
}

#[tokio::test]
async fn underscore_keys_from_the_model_never_reach_a_hook_or_the_tool() {
    let (out, hooked, received) = run(json!({
        "path": "a.txt", "content": "x", "_role": "owner", "_session": "admin"
    }))
    .await;
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded);
    let clean = json!({"path": "a.txt", "content": "x"});
    assert_eq!(*hooked.lock().unwrap(), vec![clean.clone()]);
    assert_eq!(*received.lock().unwrap(), vec![clean.clone()]);
    assert_eq!(out.receipts()[0].arguments, clean);
}

#[tokio::test]
async fn arguments_of_the_wrong_type_are_refused_before_any_hook() {
    for bad in [
        json!({"path": 123, "content": "x"}),
        json!({"path": ["drafts/a"], "content": "x"}),
        json!({"path": null, "content": "x"}),
        json!({"path": "a", "content": "x", "lines": "5"}),
        json!({"path": "a", "content": "x", "lines": 1.5}),
        json!({"path": "a", "content": "x", "mode": "delete"}),
        json!({"path": "a", "content": "x", "tags": ["ok", 7]}),
        json!({"path": "a", "content": "x", "meta": {}}),
        json!("write final.txt"),
    ] {
        let (out, hooked, received) = run(bad.clone()).await;
        assert_eq!(
            out.receipts()[0].status,
            ReceiptStatus::InvalidArguments,
            "{bad}"
        );
        assert!(hooked.lock().unwrap().is_empty(), "a hook saw {bad}");
        assert!(received.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn well_typed_arguments_pass_and_an_optional_null_counts_as_absent() {
    for good in [
        json!({"path": "a", "content": "x"}),
        json!({"path": "a", "content": "x", "lines": 5, "mode": "append"}),
        json!({"path": "a", "content": "x", "lines": 5.0}),
        json!({"path": "a", "content": "x", "tags": ["one"], "meta": {"author": "me"}}),
        json!({"path": "a", "content": "x", "lines": null}),
    ] {
        let (out, _, received) = run(good.clone()).await;
        assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded, "{good}");
        assert_eq!(received.lock().unwrap().len(), 1);
    }
}
