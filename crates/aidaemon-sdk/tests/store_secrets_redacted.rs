//! A tool that reads a token from a secret store and leaks it (in its
//! output and its receipt details) leaks nothing: the store registered the
//! value, and the kernel redacts it with no `Redactor::secret` call.

use std::sync::Arc;

use aidaemon_sdk::secrets::{MemorySecretStore, SecretStore};
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use aidaemon_sdk::{Agent, FnTool, ToolAnnotations, ToolOutput, ToolSpec};
use serde_json::{Value, json};

const TOKEN: &str = "tok_live_9f8e7d6c5b4a3210abcd";

#[tokio::test]
async fn a_token_from_the_store_never_reaches_the_model_or_the_receipts() {
    let store = Arc::new(MemorySecretStore::new());
    store.set("deploy_token", TOKEN).unwrap();
    let tool_store = store.clone();
    let tool = FnTool::new(
        ToolSpec {
            name: "call_api".into(),
            description: "Call the deploy API".into(),
            parameters: json!({"type": "object"}),
        },
        ToolAnnotations::READ_ONLY,
        move |_args: Value| {
            let store = tool_store.clone();
            async move {
                let token = store.get("deploy_token").map_err(|e| e.to_string())?;
                Ok(
                    ToolOutput::text(format!("200 OK; debug: Authorization: {}", token.expose()))
                        .with_details(json!({"auth_used": token.expose()})),
                )
            }
        },
    );
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call("c1", "call_api", json!({}))]),
        text("done"),
    ]));
    let agent = Agent::builder("store-secrets")
        .model("m")
        .provider(provider.clone())
        .tool(tool)
        .allow_unguarded()
        .build()
        .unwrap();
    let out = agent.run("s", "deploy").await.unwrap();
    assert!(out.is_completed(), "{:?}", out.status);
    assert!(!format!("{:?}", provider.requests()).contains(TOKEN));
    assert!(!format!("{:?}", out.receipts()).contains(TOKEN));
    assert!(!serde_json::to_string(&out.state).unwrap().contains(TOKEN));
}
