//! End to end: a scripted model edits a file through the real agent loop,
//! then reads it back to verify.

use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider};
use aidaemon_core::{Agent, ReceiptStatus, Role};
use aidaemon_tools::{default_tools, Workspace};
use serde_json::json;

#[tokio::test]
async fn agent_edits_then_verifies_with_read() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("greeting.txt"),
        "Hello, Alice Rivera!\nWelcome to Acme Corp.\n",
    )
    .unwrap();
    let workspace = Workspace::new(dir.path()).unwrap();

    let provider = ScriptedProvider::new([
        tool_calls([call(
            "c1",
            "grep",
            json!({"pattern": "Acme", "output_mode": "content"}),
        )]),
        tool_calls([call(
            "c2",
            "edit_file",
            json!({
                "path": "greeting.txt",
                "old_string": "Welcome to Acme Corp.",
                "new_string": "Welcome to Acme Labs."
            }),
        )]),
        tool_calls([call("c3", "read_file", json!({"path": "greeting.txt"}))]),
        text("Renamed Acme Corp to Acme Labs in greeting.txt and verified the file."),
    ]);

    let mut builder = Agent::builder("e2e-editor")
        .model("scripted")
        .provider(provider);
    for tool in default_tools(workspace) {
        builder = builder.tool_arc(tool);
    }
    let agent = builder.build().unwrap();

    let outcome = agent
        .run("test:synthetic-user-1", "Rename Acme Corp to Acme Labs")
        .await
        .unwrap();
    assert!(outcome.is_completed(), "{:?}", outcome.status);

    let statuses: Vec<(String, ReceiptStatus)> = outcome
        .receipts()
        .iter()
        .map(|r| (r.tool.clone(), r.status))
        .collect();
    assert_eq!(
        statuses,
        vec![
            ("grep".to_string(), ReceiptStatus::Succeeded),
            ("edit_file".to_string(), ReceiptStatus::Succeeded),
            ("read_file".to_string(), ReceiptStatus::Succeeded),
        ]
    );
    assert_eq!(outcome.receipts()[1].details["replacements"], 1);

    let tool_results: Vec<&str> = outcome
        .state
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(tool_results[0], "greeting.txt:2:Welcome to Acme Corp.");
    assert!(tool_results[2].contains("     2\tWelcome to Acme Labs."));

    assert_eq!(
        std::fs::read_to_string(dir.path().join("greeting.txt")).unwrap(),
        "Hello, Alice Rivera!\nWelcome to Acme Labs.\n"
    );
}
