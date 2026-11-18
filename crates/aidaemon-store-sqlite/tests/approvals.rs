#![cfg(feature = "channels")]

use std::sync::Arc;

use aidaemon_channels::approval_conformance::check_approval_store;
use aidaemon_channels::channel::{Conversation, ConversationKind, MessageRef};
use aidaemon_channels::{ApprovalStore, StoredApproval};
use aidaemon_core::RunId;
use aidaemon_store_sqlite::{SqliteApprovalStore, SqliteStore};

fn approval(prompt_id: &str, conversation: &str) -> StoredApproval {
    StoredApproval {
        prompt_id: prompt_id.to_string(),
        session_id: format!("telegram:{conversation}"),
        channel: "telegram".to_string(),
        conversation: Conversation {
            id: conversation.to_string(),
            kind: ConversationKind::Direct,
        },
        reply_to: Some("m-1".to_string()),
        run_id: RunId::new(),
        prompt_text: "Allow the synthetic deploy?".to_string(),
        prompt_ref: Some(MessageRef {
            conversation_id: conversation.to_string(),
            message_id: "m-2".to_string(),
        }),
        expires_at_ms: 4_000_000_000_000,
    }
}

#[tokio::test]
async fn the_sqlite_store_passes_the_kit() {
    check_approval_store(|| async {
        let store = SqliteStore::open_in_memory().await.unwrap();
        Arc::new(SqliteApprovalStore::new(&store)) as Arc<dyn ApprovalStore>
    })
    .await
    .assert_all_passed();
}

#[tokio::test]
async fn pending_approvals_survive_a_restart_and_settle_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("agent.db");
    let kept = approval("p-keep", "chat-1");
    let other = approval("p-other", "chat-2");
    {
        let store = SqliteStore::open(&path).await.unwrap();
        let approvals = SqliteApprovalStore::new(&store);
        approvals.save(&kept).await.unwrap();
        approvals.save(&other).await.unwrap();
    } // everything dropped: the "process" ends

    let store = SqliteStore::open(&path).await.unwrap();
    let approvals = SqliteApprovalStore::new(&store);
    let mut loaded = approvals.load_all().await.unwrap();
    loaded.sort_by(|a, b| a.prompt_id.cmp(&b.prompt_id));
    assert_eq!(loaded, vec![kept.clone(), other.clone()]);

    // A double click settles it once; the other conversation's is untouched.
    approvals.remove("p-keep").await.unwrap();
    approvals.remove("p-keep").await.unwrap();
    assert_eq!(approvals.load_all().await.unwrap(), vec![other]);
}

#[tokio::test]
async fn an_approval_expired_while_stopped_is_still_loaded_for_the_runner_to_deny() {
    let store = SqliteStore::open_in_memory().await.unwrap();
    let approvals = SqliteApprovalStore::new(&store);
    let mut old = approval("p-old", "chat-1");
    old.expires_at_ms = 1;
    approvals.save(&old).await.unwrap();
    assert_eq!(approvals.load_all().await.unwrap(), vec![old]);
}

#[tokio::test]
async fn racing_removes_and_saves_do_not_error() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(dir.path().join("agent.db"))
        .await
        .unwrap();
    let approvals = Arc::new(SqliteApprovalStore::new(&store));
    approvals.save(&approval("p", "chat-1")).await.unwrap();
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let a = approvals.clone();
        tasks.push(tokio::spawn(async move { a.remove("p").await }));
    }
    for t in tasks {
        t.await.unwrap().unwrap();
    }
    assert!(approvals.load_all().await.unwrap().is_empty());
}

#[tokio::test]
async fn the_schema_version_covers_approvals() {
    let store = SqliteStore::open_in_memory().await.unwrap();
    assert_eq!(
        store.schema_version().await.unwrap(),
        aidaemon_store_sqlite::latest_version()
    );
    assert!(aidaemon_store_sqlite::latest_version() >= 2);
}
