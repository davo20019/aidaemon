//! Memory swap: the same agent with two long-term memory systems. One is
//! `aidaemon-memory` (facts, episodes, hybrid recall); the other is a
//! notebook written here in a few lines. Anything implementing
//! `MemorySystem` plugs in the same way: a vector database, a hosted
//! memory service.
//!
//! Live with ANTHROPIC_API_KEY or OPENAI_API_KEY, otherwise scripted:
//! cargo run -p aidaemon-sdk --example memory_swap --features "testing memory"
use std::sync::{Arc, Mutex};

use aidaemon_sdk::FnTool;
use aidaemon_sdk::memory::{HashEmbedder, InMemoryMemoryStore, Memory, ScopedMemory};
use aidaemon_sdk::policies::{MemoryInjection, MemoryParts, MemorySystem, MemoryTurn, Recall};
use aidaemon_sdk::prelude::*;
use aidaemon_sdk::testing::{ScriptedProvider, call, text, tool_calls};
use serde_json::json;

/// A notebook: a `remember` tool writes a line, and every line is shown to
/// the model before it answers.
#[derive(Clone, Default)]
struct Notebook(Arc<Mutex<Vec<String>>>);

#[async_trait::async_trait]
impl Recall for Notebook {
    async fn recall(&self, _session_id: &str, _query: &str, limit: usize) -> Vec<String> {
        let lines = self.0.lock().unwrap();
        lines.iter().rev().take(limit).cloned().collect()
    }
}

#[async_trait::async_trait]
impl MemorySystem for Notebook {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        let mut parts = MemoryParts::default();
        if turn.read {
            parts
                .hooks
                .push(Arc::new(MemoryInjection::new(self.clone())));
        }
        if turn.write {
            let notebook = self.clone();
            let spec = aidaemon_sdk::ToolSpec {
                name: "remember".into(),
                description: "Write a fact about the user in the notebook.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "key": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["category", "key", "value"],
                    "additionalProperties": false
                }),
            };
            let remember = FnTool::new(spec, ToolAnnotations::WRITE, move |args| {
                let notebook = notebook.clone();
                async move {
                    let line = format!(
                        "{}: {}",
                        args["key"].as_str().unwrap_or_default(),
                        args["value"].as_str().unwrap_or_default()
                    );
                    notebook.0.lock().unwrap().push(line);
                    Ok(ToolOutput::text("written"))
                }
            })
            .with_trust(aidaemon_sdk::SourceTrust::Trusted);
            parts.tools.push(Arc::new(remember));
        }
        parts
    }

    fn guidance(&self) -> Option<String> {
        Some("Save lasting facts about the user with `remember`.".into())
    }
}

async fn chat(memory: Arc<dyn MemorySystem>) -> Result<(), Box<dyn std::error::Error>> {
    let script = ScriptedProvider::new([
        tool_calls([call(
            "c1",
            "remember",
            json!({"category": "preference", "key": "favorite_color", "value": "teal"}),
        )]),
        text("Noted: teal."),
        text("Your favorite color is teal."),
    ]);
    let turn = MemoryTurn {
        session_id: "cli:synthetic-user-1".into(),
        scope: "owner".into(),
        read: true,
        write: true,
        explicit_writes_only: true,
        learning: Default::default(),
        host: Default::default(),
    };
    let agent = aidaemon_sdk::preset("assistant")
        .provider_from_env_or(script)
        .instructions("You are a helpful assistant. Keep answers short.")
        .memory(memory, turn)
        .build()
        .await?;
    agent
        .run("cli:synthetic-user-1", "My favorite color is teal.")
        .await?;
    let answer = agent
        .run("cli:synthetic-user-1", "What's my favorite color?")
        .await?;
    println!("  {}", answer.final_text.unwrap_or_default());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("aidaemon-memory:");
    let store = Arc::new(InMemoryMemoryStore::new());
    let builder = Memory::builder(store, Arc::new(HashEmbedder::default()));
    chat(Arc::new(ScopedMemory::new(builder))).await?;

    println!("notebook:");
    let notebook = Notebook::default();
    chat(Arc::new(notebook.clone())).await?;
    println!("  notebook holds: {:?}", notebook.0.lock().unwrap());
    Ok(())
}
