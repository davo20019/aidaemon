//! A memory system need not resemble the daemon's. This one keeps a journal:
//! no facts, no people, no episodes, one tool.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::{Tool, ToolContext, ToolResult, ToolSpec};
use aidaemon_policies::{MemoryParts, MemorySystem, MemoryTurn};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::{MemoryPlugin, SystemMemory, DAEMON_MEMORY_TOOLS};
use crate::testing::{setup_test_agent, MockProvider};
use crate::types::{ChannelContext, UserRole};

type Journal = Arc<Mutex<Vec<String>>>;

struct JournalNote(Journal);

#[async_trait]
impl Tool for JournalNote {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "journal_note".into(),
            description: "Add a line to the journal.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"line": {"type": "string"}},
                "required": ["line"],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let line = arguments["line"].as_str().unwrap_or_default().to_string();
        self.0.lock().unwrap().push(line);
        Ok(aidaemon_core::ToolOutput::text("noted"))
    }
}

struct JournalMemory(Journal);

#[async_trait]
impl MemorySystem for JournalMemory {
    async fn turn(&self, _turn: &MemoryTurn) -> MemoryParts {
        MemoryParts {
            tools: vec![Arc::new(JournalNote(self.0.clone()))],
            ..MemoryParts::default()
        }
    }
}

fn offered_names(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|tool| {
            tool.pointer("/function/name")
                .or_else(|| tool.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

#[tokio::test]
async fn a_memory_system_of_another_shape_runs_a_whole_turn() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("journal_note", r#"{"line":"synthetic entry"}"#),
        MockProvider::text_response("Done."),
    ]);
    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();
    let journal = Journal::default();
    let plugin = SystemMemory::new(Arc::new(JournalMemory(journal.clone())));
    assert!(!plugin.is_daemons_own());
    harness.agent.set_memory_plugin(Arc::new(plugin));

    harness
        .agent
        .handle_message(
            "journal_session",
            "Note this in the journal: synthetic entry. My favorite color is teal.",
            None,
            UserRole::Owner,
            ChannelContext::private("telegram"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(*journal.lock().unwrap(), ["synthetic entry"]);
    let facts = super::memory_tool_facts("journal_note").expect("described by the turn");
    assert_eq!(
        facts.scope,
        Some(crate::traits::ToolSemanticScope::UserMemory)
    );

    let calls = harness.provider.call_log.lock().await;
    let first = calls.first().expect("the model was called");
    let offered = offered_names(&first.tools);
    assert!(
        offered.iter().any(|name| name == "journal_note"),
        "{offered:?}"
    );
    for owned in DAEMON_MEMORY_TOOLS
        .iter()
        .filter(|name| **name != "manage_memories")
    {
        assert!(!offered.iter().any(|name| name == owned), "{owned} offered");
    }
    let tools = serde_json::to_string(&first.tools).unwrap();
    assert!(!tools.contains("search_episodes"), "memory actions offered");

    let system = first
        .messages
        .iter()
        .filter(|message| message["role"] == "system")
        .map(|message| message.to_string())
        .collect::<String>();
    for assumption in [
        "`remember_fact`",
        "Proactively store personal information",
        "Quote stored fact values",
        "your memory recall tool",
    ] {
        assert!(!system.contains(assumption), "prompt assumes: {assumption}");
    }
    drop(calls);

    // Learning runs in the background after a turn.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let pool = harness.state.pool();
    for table in [
        "facts",
        "episodes",
        "people",
        "procedures",
        "expertise",
        "error_solutions",
        "behavior_patterns",
        "memory_claims",
    ] {
        let rows: i64 =
            sqlx::query_scalar(sqlx::AssertSqlSafe(format!("SELECT COUNT(*) FROM {table}")))
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(rows, 0, "{table} was written");
    }
}
