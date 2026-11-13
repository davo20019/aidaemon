//! The API an outside developer meets first: a provider's default model, a
//! scripted provider shared with the test, each call's result on the
//! outcome, asking the owner from a hook, and run time on Tokio's clock.

use std::time::Duration;

use aidaemon_core::testing::{SCRIPTED_MODEL, ScriptedProvider, call, spec, text, tool_calls};
use aidaemon_core::{
    Agent, ApprovalRisk, BuildError, CallEffects, Event, FnTool, Hook, HookContext, InterruptKind,
    OwnerApprovalRequest, Provider, ReceiptStatus, RunId, RunStatus, ToolAnnotations, ToolCall,
    ToolDecision, ToolError, ToolInfo, ToolOutput,
};
use async_trait::async_trait;
use serde_json::json;

fn echo() -> impl aidaemon_core::Tool {
    FnTool::new(
        spec("echo", "Echo the text back", &["text"]),
        ToolAnnotations::READ_ONLY,
        |args| async move {
            match args["text"].as_str() {
                Some("fail") => Err(ToolError::new("synthetic failure")),
                Some(text) => Ok(ToolOutput::text(format!("echo: {text}"))),
                None => Err(ToolError::new("no text")),
            }
        },
    )
}

#[test]
fn a_provider_with_a_default_model_needs_no_model_call() {
    let model = ScriptedProvider::new([text("hi")]);
    assert_eq!(model.default_model(), Some(SCRIPTED_MODEL));
    let agent = Agent::builder("a").provider(model).build();
    assert!(agent.is_ok(), "{:?}", agent.err());
}

struct Nameless;

#[async_trait]
impl Provider for Nameless {
    fn name(&self) -> &str {
        "nameless"
    }
    async fn complete(
        &self,
        _request: &aidaemon_core::ModelRequest,
    ) -> Result<aidaemon_core::ModelResponse, aidaemon_core::ProviderError> {
        Ok(text("hi"))
    }
}

#[test]
fn a_missing_model_says_which_call_sets_it() {
    let error = Agent::builder("a")
        .provider(Nameless)
        .build()
        .err()
        .unwrap();
    assert_eq!(error, BuildError::MissingModel);
    assert!(error.to_string().contains(".model(..)"), "{error}");
}

#[tokio::test]
async fn a_clone_of_the_scripted_provider_sees_every_request() {
    let model = ScriptedProvider::new([
        tool_calls([
            call("c1", "echo", json!({"text": "hello"})),
            call("c2", "echo", json!({"text": "fail"})),
        ]),
        text("done"),
    ]);
    let seen = model.clone();
    let agent = Agent::builder("a")
        .allow_unguarded()
        .provider(model)
        .tool(echo())
        .build()
        .unwrap();
    let outcome = agent.run("test:s", "go").await.unwrap();
    assert!(outcome.is_completed());
    assert_eq!(seen.requests().len(), 2);
    assert_eq!(seen.requests()[0].model, SCRIPTED_MODEL);
    assert_eq!(seen.remaining(), 0);

    let results = outcome.tool_results();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].receipt.call_id, "c1");
    assert!(results[0].succeeded());
    assert_eq!(results[0].text, Some("echo: hello"));
    assert_eq!(results[1].receipt.status, ReceiptStatus::Failed);
    assert!(results[1].text.unwrap().contains("synthetic failure"));
}

/// Asks the owner before every `echo`, the way the approval policy would.
struct AskFirst;

#[async_trait]
impl Hook for AskFirst {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        info: &ToolInfo,
    ) -> ToolDecision {
        ToolDecision::ask(OwnerApprovalRequest::for_call(call, info, "it is a test"))
    }
}

#[tokio::test]
async fn a_hook_asks_the_owner_with_a_request_built_for_the_call() {
    let agent = Agent::builder("a")
        .allow_unguarded()
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "echo", json!({"text": "hi"}))]),
            text("done"),
        ]))
        .tool(echo())
        .hook(AskFirst)
        .build()
        .unwrap();
    let outcome = agent.run("test:s", "go").await.unwrap();
    let RunStatus::Interrupted { interruption } = &outcome.status else {
        panic!("expected an interruption, got {:?}", outcome.status);
    };
    let InterruptKind::OwnerApproval(request) = &interruption.kind else {
        panic!("expected an owner approval");
    };
    assert_eq!(request.subject, r#"echo {"text":"hi"}"#);
    assert_eq!(request.summary, r#"Run echo {"text":"hi"}: it is a test"#);
    assert_eq!(request.risk, ApprovalRisk::High);
    assert!(request.card.is_some());
    assert_eq!(
        interruption.reason,
        format!("approval required: {}", request.summary)
    );
}

#[test]
fn complete_effects_without_paths_vouch_for_no_files() {
    assert!(CallEffects::none().touches_no_paths());
    assert!(!CallEffects::unknown().touches_no_paths());
    let write = CallEffects {
        keys: vec![
            aidaemon_core::EffectKey::path(aidaemon_core::EffectClass::Write, "/tmp/x").unwrap(),
        ],
        complete: true,
        understood: true,
    };
    assert!(!write.touches_no_paths());
}

#[tokio::test(start_paused = true)]
async fn run_time_follows_the_runtime_clock() {
    let slow = FnTool::new(
        spec("slow", "Wait a little", &[]),
        ToolAnnotations::READ_ONLY,
        |_| async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(ToolOutput::text("waited"))
        },
    );
    let finished = std::sync::Arc::new(std::sync::Mutex::new(None));
    let seen = finished.clone();
    let agent = Agent::builder("a")
        .allow_unguarded()
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "slow", json!({}))]),
            text("done"),
        ]))
        .tool(slow)
        .event_sink(move |_run: RunId, event: &Event| {
            if let Event::Finished { elapsed_ms, .. } = event {
                *seen.lock().unwrap() = Some(*elapsed_ms);
            }
        })
        .build()
        .unwrap();
    let outcome = agent.run("test:s", "go").await.unwrap();
    assert_eq!(outcome.elapsed(), Duration::from_millis(50));
    assert_eq!(*finished.lock().unwrap(), Some(50));
}
