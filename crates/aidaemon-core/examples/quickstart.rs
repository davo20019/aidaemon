//! Runnable tour of the kernel with a scripted model:
//! read-only tools run in parallel, a destructive tool pauses for approval,
//! the paused state survives a JSON round trip, and the run resumes.
//!
//! cargo run -p aidaemon-core --example quickstart --features testing

use std::sync::Arc;

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, Event, FnTool, Hook, HookContext, MemoryStore, Resolution, RunState, RunStatus,
    ToolAnnotations, ToolCall, ToolDecision, ToolOutput,
};
use async_trait::async_trait;
use serde_json::json;

/// Policy as a hook: ask a human before anything destructive.
struct AskBeforeDestructive;

#[async_trait]
impl Hook for AskBeforeDestructive {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        annotations: ToolAnnotations,
    ) -> ToolDecision {
        if annotations.destructive {
            ToolDecision::Interrupt {
                reason: format!("`{}` is destructive; needs approval", call.name),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The "model": list two files in parallel, delete one, then answer.
    let model = ScriptedProvider::new([
        tool_calls([
            call("c1", "read_file", json!({"path": "notes/a.txt"})),
            call("c2", "read_file", json!({"path": "notes/b.txt"})),
        ]),
        tool_calls([call("c3", "delete_file", json!({"path": "notes/b.txt"}))]),
        text("b.txt duplicated a.txt, so I deleted it."),
    ]);

    let read_file = FnTool::new(
        spec("read_file", "Read a UTF-8 text file", &["path"]),
        ToolAnnotations::READ_ONLY,
        |args| async move {
            let path = args["path"].as_str().unwrap_or_default().to_string();
            Ok(ToolOutput::text(format!("(contents of {path})")))
        },
    );
    let delete_file = FnTool::new(
        spec("delete_file", "Delete a file", &["path"]),
        ToolAnnotations::DESTRUCTIVE,
        |args| async move {
            let path = args["path"].as_str().unwrap_or_default().to_string();
            Ok(ToolOutput::text(format!("deleted {path}")).with_details(json!({"path": path})))
        },
    );

    let store = Arc::new(MemoryStore::new());
    let agent = Agent::builder("file-tidier")
        .model("scripted-model")
        .instructions("You tidy up notes folders. Explain what you changed.")
        .provider(model)
        .tool(read_file)
        .tool(delete_file)
        .hook(AskBeforeDestructive)
        .store(store.clone())
        .event_sink(|_run, event: &Event| match event {
            Event::ToolFinished { receipt } => {
                println!("  receipt: {} -> {:?}", receipt.tool, receipt.status)
            }
            Event::Interrupted { interruption } => {
                println!("  paused: {}", interruption.reason)
            }
            _ => {}
        })
        .build()?;

    println!("run 1");
    let outcome = agent
        .run("cli:synthetic-user-1", "Remove duplicate notes")
        .await?;
    assert!(matches!(outcome.status, RunStatus::Interrupted { .. }));

    // Persist the paused run as JSON, as if a human answers tomorrow.
    let saved = serde_json::to_string(&outcome.state)?;
    let restored: RunState = serde_json::from_str(&saved)?;

    println!("resume (approved)");
    let done = agent.resume(restored, Resolution::Approve).await?;
    println!("status: {:?}", done.status);
    println!("answer: {}", done.final_text.as_deref().unwrap_or(""));
    println!(
        "turns: {}, receipts: {}, billable tokens: {}",
        done.turns(),
        done.receipts().len(),
        done.usage().billable_tokens()
    );
    println!(
        "journal events stored for the resumed run: {}",
        store.events(done.state.run_id).len()
    );
    Ok(())
}
