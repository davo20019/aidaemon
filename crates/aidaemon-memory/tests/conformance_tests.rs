//! The memory store conformance kit against both bundled stores and the
//! user-defined Vec-backed store, plus the tool kit over the memory tools.

mod common;

use std::sync::Arc;

use aidaemon_core::conformance::{check_tool, ConformanceReport};
use aidaemon_memory::conformance::{check_store, store_checks};
use aidaemon_memory::{
    HashEmbedder, InMemoryMemoryStore, Memory, MemoryStore, RecallTool, RememberTool,
    SqliteMemoryStore,
};
use common::VecStore;

#[tokio::test]
async fn in_memory_store_conforms() {
    let report =
        check_store(|| async { Arc::new(InMemoryMemoryStore::new()) as Arc<dyn MemoryStore> })
            .await;
    report.assert_all_passed();
    // Both bundled stores rank through the scorer (no native search).
    assert_eq!(report.notes.len(), 2, "{report}");
}

#[tokio::test]
async fn sqlite_store_conforms() {
    let report = check_store(|| async {
        Arc::new(SqliteMemoryStore::open_in_memory().await.unwrap()) as Arc<dyn MemoryStore>
    })
    .await;
    report.assert_all_passed();
    // Both bundled stores rank through the scorer (no native search).
    assert_eq!(report.notes.len(), 2, "{report}");
}

#[tokio::test]
async fn sqlite_file_store_conforms() {
    let dir = tempfile::tempdir().unwrap();
    let counter = std::sync::atomic::AtomicUsize::new(0);
    let report = check_store(|| {
        let path = dir.path().join(format!(
            "memory-{}.db",
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        async move { Arc::new(SqliteMemoryStore::open(path).await.unwrap()) as Arc<dyn MemoryStore> }
    })
    .await;
    report.assert_all_passed();
}

#[tokio::test]
async fn vec_store_conforms() {
    let report =
        check_store(|| async { Arc::new(VecStore::default()) as Arc<dyn MemoryStore> }).await;
    report.assert_all_passed();
    // It keeps the trait's default decay, search and access levels.
    assert!(report.passed_check(store_checks::DECAY_FOLLOWS_RETENTION));
    assert_eq!(report.notes.len(), 4, "{report}");
}

#[test]
fn memory_tools_pass_the_tool_kit() {
    let memory = Memory::new(InMemoryMemoryStore::new(), HashEmbedder::default());
    let mut all = ConformanceReport::new("aidaemon-memory tools");
    all.merge(check_tool(&RememberTool::new(memory.clone())));
    all.merge(check_tool(&RecallTool::new(memory)));
    all.assert_all_passed();
    assert_eq!(all.passed.len(), 22, "{all}");
}

/// The owner's own memory store is not someone else's content: reading or
/// writing it does not taint a run.
#[test]
fn memory_tools_return_trusted_output() {
    use aidaemon_core::{SourceTrust, Tool};
    let memory = Memory::new(InMemoryMemoryStore::new(), HashEmbedder::default());
    let args = serde_json::json!({});
    assert_eq!(
        RememberTool::new(memory.clone()).source_trust(&args),
        SourceTrust::Trusted
    );
    assert_eq!(
        RecallTool::new(memory).source_trust(&args),
        SourceTrust::Trusted
    );
}
