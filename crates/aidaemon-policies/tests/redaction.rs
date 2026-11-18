//! The `Redactor`'s literals and credential shapes cover everything that
//! leaves the kernel, not only tool output.

use std::sync::{Arc, Mutex};

use aidaemon_core::testing::{ScriptedProvider, call, text, tool_calls};
use aidaemon_core::{Agent, Event, FnTool, RunId, ToolAnnotations, ToolOutput, ToolSpec};
use aidaemon_policies::Redactor;
use serde_json::{Value, json};

const TOKEN: &str = "tok_live_policies_1a2b3c4d5e6f";
const SHAPED: &str = "sk-proj-AAAABBBBCCCCDDDDEEEEFFFF1234";

#[tokio::test]
async fn redactor_literals_and_shapes_reach_receipts_events_and_state() {
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call(
            "c1",
            "call_api",
            json!({"endpoint": "/x", "key": SHAPED}),
        )]),
        text("done"),
    ]));
    let tool = FnTool::new(
        ToolSpec {
            name: "call_api".into(),
            description: "test".into(),
            parameters: json!({"type": "object"}),
        },
        ToolAnnotations::READ_ONLY,
        |_args: Value| async move {
            Ok(ToolOutput::text(format!("debug: Authorization: {TOKEN}"))
                .with_details(json!({"auth_used": TOKEN})))
        },
    );
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = events.clone();
    let agent = Agent::builder("r")
        .model("m")
        .provider(provider.clone())
        .tool(tool)
        .hook(Redactor::default().secret(TOKEN).unwrap())
        .event_sink(move |_: RunId, e: &Event| sink.lock().unwrap().push(format!("{e:?}")))
        .allow_unguarded()
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    let everything = format!(
        "{:?}\n{:?}\n{}\n{}",
        provider.requests(),
        out.receipts(),
        serde_json::to_string(&out.state).unwrap(),
        events.lock().unwrap().join("\n")
    );
    for secret in [TOKEN, SHAPED] {
        assert!(
            !everything.contains(secret),
            "{secret} leaked:\n{everything}"
        );
    }
}

#[test]
fn a_too_short_literal_is_an_error_not_a_silent_no_op() {
    assert_eq!(
        Redactor::default().secret("12345").unwrap_err(),
        aidaemon_core::redaction::SecretTooShort { chars: 5 }
    );
    let r = Redactor::default().secret("123456").unwrap();
    assert_eq!(r.redact("code 123456 end"), "code [REDACTED:secret] end");
}

/// A credential a tool printed is registered in the scope the redactor was
/// given (the agent's own), not the process-wide set.
#[tokio::test]
async fn a_printed_credential_is_registered_in_the_redactors_scope() {
    const PRINTED: &str = "Scope-Printed-Pass-58";
    let scope = aidaemon_core::RedactionScope::isolated();
    let tool = FnTool::new(
        ToolSpec {
            name: "install".into(),
            description: "test".into(),
            parameters: json!({"type": "object"}),
        },
        ToolAnnotations::READ_ONLY,
        |_args: Value| {
            std::future::ready(Ok(ToolOutput::text(format!(
                "[success] User password: {PRINTED}"
            ))))
        },
    );
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "install", json!({}))]),
        text(format!("Your login is admin / {PRINTED}")),
    ]));
    let agent = Agent::builder("r")
        .model("m")
        .provider(provider)
        .tool(tool)
        .hook(Redactor::default().redaction_scope(scope.clone()))
        .redaction_scope(scope.clone())
        .allow_unguarded()
        .build()
        .unwrap();
    let out = agent.run("s", "go").await.unwrap();
    let final_text = out.final_text.unwrap();
    assert!(!final_text.contains(PRINTED), "{final_text}");
    assert_eq!(scope.redact(PRINTED), "[REDACTED:secret]");
    assert_eq!(
        aidaemon_core::redaction::redact_registered(PRINTED),
        PRINTED
    );
}
