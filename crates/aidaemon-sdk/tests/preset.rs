//! The preset: a working agent from a few lines, every part replaceable.

use std::sync::Arc;

use aidaemon_sdk::policies::{ApprovalMode, MemoryParts, MemorySystem, MemoryTurn};
use aidaemon_sdk::testing::{call, text, tool_calls, ScriptedProvider};
use aidaemon_sdk::{
    preset, tool, BuildError, MemoryStore, PresetError, RunStatus, RunStore, Safeguard, Tool,
};
use serde_json::json;

/// Read a synthetic note.
#[tool(read_only)]
fn read_note(name: String) -> String {
    format!("{name}: buy oat milk (synthetic)")
}

/// Delete a synthetic note.
#[tool(destructive)]
fn delete_note(name: String) -> String {
    format!("deleted {name}")
}

#[tokio::test]
async fn a_preset_agent_runs_with_the_required_safeguards() {
    // Tools on an agent need every required safeguard: the preset's hooks
    // cover them, or `build` would fail.
    let agent = preset("notes")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "read_note", json!({"name": "groceries"}))]),
            text("You need oat milk."),
        ]))
        .model("scripted")
        .tool(ReadNote)
        .build()
        .await
        .unwrap();
    let outcome = agent
        .run("test:synthetic-user", "What do I need?")
        .await
        .unwrap();
    assert!(outcome.is_succeeded(), "{:?}", outcome.status);
    assert_eq!(outcome.final_text.as_deref(), Some("You need oat milk."));
}

#[tokio::test]
async fn a_destructive_call_waits_for_the_owner_by_default() {
    let store = Arc::new(MemoryStore::new());
    let agent = preset("notes")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "delete_note", json!({"name": "old"}))]),
            text("Deleted."),
        ]))
        .model("scripted")
        .tool(DeleteNote)
        .store(store.clone())
        .build()
        .await
        .unwrap();
    let outcome = agent
        .run("test:synthetic-user", "Delete old")
        .await
        .unwrap();
    assert!(
        matches!(outcome.status, RunStatus::Interrupted { .. }),
        "{:?} {:?}",
        outcome.status,
        outcome.receipts()
    );
    assert!(
        !store.events(outcome.state.run_id).is_empty(),
        "the given store is used"
    );

    // Approvals can be turned off like any other part.
    let agent = preset("notes")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "delete_note", json!({"name": "old"}))]),
            text("Deleted."),
        ]))
        .model("scripted")
        .approvals(ApprovalMode::Never)
        .tool(DeleteNote)
        .build()
        .await
        .unwrap();
    let outcome = agent
        .run("test:synthetic-user", "Delete old")
        .await
        .unwrap();
    assert!(outcome.is_completed(), "{:?}", outcome.status);
}

#[tokio::test]
async fn without_a_provider_or_the_default_hooks_building_fails() {
    let err = preset("x").model("m").build().await.err().unwrap();
    assert!(matches!(err, PresetError::NoProvider), "{err}");

    let err = preset("x")
        .provider(ScriptedProvider::new([]))
        .model("m")
        .without_default_hooks()
        .tool(ReadNote)
        .build()
        .await
        .err()
        .unwrap();
    assert!(
        matches!(
            err,
            PresetError::Build(BuildError::MissingSafeguard(Safeguard::Authority))
        ),
        "{err}"
    );
}

#[tokio::test]
async fn a_scripted_fallback_is_used_when_no_key_is_set() {
    if std::env::var("ANTHROPIC_API_KEY").is_ok() || std::env::var("OPENAI_API_KEY").is_ok() {
        return; // a live key in the environment wins, by design
    }
    let agent = preset("x")
        .provider_from_env_or(ScriptedProvider::new([text("hello")]))
        .build()
        .await
        .unwrap();
    let outcome = agent.run("test:synthetic-user", "hi").await.unwrap();
    assert_eq!(outcome.final_text.as_deref(), Some("hello"));
}

#[tokio::test]
async fn a_workspace_brings_the_default_tools_and_sqlite_keeps_runs() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("runs.db");
    let agent = preset("coder")
        .provider(ScriptedProvider::new([text("ok")]))
        .model("scripted")
        .workspace(dir.path())
        .sqlite(&db)
        .tool(ReadNote)
        .build()
        .await
        .unwrap();
    let names: Vec<&str> = agent.tool_specs().iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"read_file"), "{names:?}");
    assert!(names.contains(&"read_note"), "{names:?}");
    let outcome = agent.run("test:synthetic-user", "hi").await.unwrap();
    assert!(outcome.is_completed());
    drop(agent);
    let store = aidaemon_sdk::store::SqliteStore::open(&db).await.unwrap();
    assert!(store
        .load_state(outcome.state.run_id)
        .await
        .unwrap()
        .is_some());
}

/// A synthetic memory system: one recall tool and its own guidance.
struct NotebookMemory;

#[async_trait::async_trait]
impl MemorySystem for NotebookMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        assert_eq!(turn.scope, "owner");
        MemoryParts {
            hooks: Vec::new(),
            tools: vec![Arc::new(ReadNote) as Arc<dyn Tool>],
        }
    }

    fn guidance(&self) -> Option<String> {
        Some("Notes live in the notebook; read them with read_note.".into())
    }
}

#[tokio::test]
async fn any_memory_system_plugs_in() {
    let provider = Arc::new(ScriptedProvider::new([text("ok")]));
    let agent = preset("assistant")
        .provider(provider.clone())
        .model("scripted")
        .instructions("Be brief.")
        .memory(
            Arc::new(NotebookMemory),
            MemoryTurn {
                session_id: "test:synthetic-user".into(),
                scope: "owner".into(),
                read: true,
                write: true,
                explicit_writes_only: false,
                learning: Default::default(),
                host: Default::default(),
            },
        )
        .build()
        .await
        .unwrap();
    assert_eq!(agent.tool_specs()[0].name, "read_note");
    agent.run("test:synthetic-user", "hi").await.unwrap();
    let system = &provider.requests()[0].messages[0].content;
    assert!(system.starts_with("Be brief."), "{system}");
    assert!(system.contains("notebook"), "{system}");
}
