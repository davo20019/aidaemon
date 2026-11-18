//! One redaction pass at the kernel boundary: a registered secret never
//! reaches the model, the event stream, a receipt (arguments and details
//! included), the question put to the owner or the saved state.

use std::sync::{Arc, Mutex};

use aidaemon_core::redaction::{RedactionScope, SecretTooShort};
use aidaemon_core::testing::{ScriptedProvider, call, text, tool_calls};
use aidaemon_core::{
    Agent, ApprovalRisk, Event, FnTool, Hook, HookContext, InterruptKind, MemoryStore,
    OwnerApprovalRequest, Resolution, RunId, RunStatus, RunStore, ToolAnnotations, ToolCall,
    ToolDecision, ToolInfo, ToolOutput, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{Value, json};

/// Asks the owner before every `write` call, quoting its arguments.
struct AskBeforeWrite;

#[async_trait]
impl Hook for AskBeforeWrite {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        call: &ToolCall,
        _: &ToolInfo,
    ) -> ToolDecision {
        if call.name != "write" {
            return ToolDecision::Allow;
        }
        ToolDecision::Interrupt {
            reason: format!("approve {} {}", call.name, call.arguments),
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

fn tool(name: &str, secret: &'static str) -> impl aidaemon_core::Tool + use<> {
    let spec = ToolSpec {
        name: name.into(),
        description: "test".into(),
        parameters: json!({"type": "object"}),
    };
    FnTool::new(spec, ToolAnnotations::WRITE, move |_args: Value| {
        std::future::ready(Ok(ToolOutput::text(format!("Authorization: {secret}"))
            .with_details(json!({"auth_used": secret}))))
    })
}

#[tokio::test]
async fn a_registered_secret_is_redacted_everywhere_it_leaves_the_kernel() {
    const SECRET: &str = "tok_live_boundary_9f8e7d6c5b4a";
    let scope = RedactionScope::isolated();
    scope.register(SECRET).unwrap();
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([
            call("c1", "fetch", json!({})),
            call(
                "c2",
                "write",
                json!({"path": "notes.txt", "content": SECRET}),
            ),
        ]),
        text(format!("done, the token was {SECRET}")),
    ]));
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = events.clone();
    let store = Arc::new(MemoryStore::new());
    let agent = Agent::builder("redact")
        .model("m")
        .provider(provider.clone())
        .tool(tool("fetch", SECRET))
        .tool(tool("write", SECRET))
        .hook(AskBeforeWrite)
        .event_sink(move |_: RunId, event: &Event| sink.lock().unwrap().push(format!("{event:?}")))
        .store(store.clone())
        .redaction_scope(scope)
        .allow_unguarded()
        .build()
        .unwrap();

    let out = agent
        .run("s", format!("use the key {SECRET} please"))
        .await
        .unwrap();
    let RunStatus::Interrupted { interruption } = &out.status else {
        panic!("{:?}", out.status);
    };
    assert!(
        !format!("{interruption:?}").contains(SECRET),
        "{interruption:?}"
    );
    // The paused call keeps its exact arguments so it can run; everything
    // else in the saved state is clean.
    let mut saved = serde_json::to_value(&out.state).unwrap();
    saved["pending"]["calls"] = Value::Null;
    assert!(!saved.to_string().contains(SECRET), "{saved}");

    let out = agent.resume(out.state, Resolution::Approve).await.unwrap();
    assert!(out.is_completed(), "{:?}", out.status);

    for request in provider.requests() {
        assert!(!format!("{request:?}").contains(SECRET), "model saw it");
    }
    for event in events.lock().unwrap().iter() {
        assert!(!event.contains(SECRET), "event: {event}");
    }
    for receipt in out.receipts() {
        assert!(!format!("{receipt:?}").contains(SECRET), "{receipt:?}");
    }
    assert!(!serde_json::to_string(&out.state).unwrap().contains(SECRET));
    let stored = store.load_state(out.state.run_id).await.unwrap().unwrap();
    assert!(!serde_json::to_string(&stored).unwrap().contains(SECRET));
    for event in store.events(out.state.run_id) {
        assert!(!format!("{event:?}").contains(SECRET));
    }
    assert!(!out.final_text.unwrap_or_default().contains(SECRET));
}

#[test]
fn a_secret_too_short_to_redact_is_refused_not_ignored() {
    let scope = RedactionScope::isolated();
    assert_eq!(scope.register("12345"), Err(SecretTooShort { chars: 5 }));
    assert!(scope.is_empty());
    scope.register("123456-registered").unwrap();
    assert_eq!(
        scope.redact("code 123456-registered end"),
        "code [REDACTED:secret] end"
    );
}

/// An agent given its own scope redacts what is registered there and
/// nothing else: another agent's secret, or the process-wide set, never
/// garbles its output, and its secret never reaches theirs. A test agent
/// gets a clean slate however many tests registered secrets before it.
#[tokio::test]
async fn an_isolated_scope_neither_leaks_in_nor_out() {
    const MINE: &str = "tok_scope_mine_0a1b2c3d";
    const THEIRS: &str = "tok_scope_theirs_4e5f6a7b";
    let mine = RedactionScope::isolated();
    let theirs = RedactionScope::isolated();
    mine.register(MINE).unwrap();
    theirs.register(THEIRS).unwrap();
    aidaemon_core::redaction::register_secret("tok_scope_process_8c9d0e1f").unwrap();

    let run = |scope: RedactionScope| async move {
        let provider = Arc::new(ScriptedProvider::new([text(format!(
            "{MINE} {THEIRS} tok_scope_process_8c9d0e1f"
        ))]));
        let agent = Agent::builder("scoped")
            .model("m")
            .provider(provider)
            .redaction_scope(scope)
            .allow_unguarded()
            .build()
            .unwrap();
        agent.run("s", "go").await.unwrap().final_text.unwrap()
    };
    assert_eq!(
        run(mine.clone()).await,
        format!("[REDACTED:secret] {THEIRS} tok_scope_process_8c9d0e1f")
    );
    assert_eq!(
        run(theirs).await,
        format!("{MINE} [REDACTED:secret] tok_scope_process_8c9d0e1f")
    );
    assert!(!mine.is_process());
    assert!(RedactionScope::process().is_process());
    assert_eq!(
        aidaemon_core::redaction::redact_registered(MINE),
        MINE,
        "an isolated scope's secret stays out of the process-wide set"
    );
}
