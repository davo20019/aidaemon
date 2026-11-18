//! The shared behavior suite against the default preset (aidaemon's full
//! memory system), one test per case.

use aidaemon_sdk::testing::behavior::{Subject, Verdict, cases, run_case};

fn default_preset() -> Subject {
    Subject::new("default preset", |parts| async move {
        let preset = aidaemon_sdk::preset(parts.id())
            .instructions("You are a helpful assistant. Keep answers short.")
            .memory_path(parts.memory_dir().join("memory.db"));
        Ok(parts.finish(preset).build().await?)
    })
    .full_memory()
}

async fn check(name: &str) {
    let verdict = run_case(&default_preset(), name).await;
    assert_eq!(verdict, Verdict::Passed, "{name}");
}

#[tokio::test]
async fn memory_write() {
    check("memory_write").await;
}

#[tokio::test]
async fn memory_recall() {
    check("memory_recall").await;
}

#[tokio::test]
async fn tool_use() {
    check("tool_use").await;
}

#[tokio::test]
async fn approvals() {
    check("approvals").await;
}

#[tokio::test]
async fn sub_agents() {
    check("sub_agents").await;
}

#[tokio::test]
async fn channel_round_trip() {
    check("channel_round_trip").await;
}

#[tokio::test]
async fn full_memory_tools() {
    check("full_memory_tools").await;
}

#[tokio::test]
async fn full_memory_manage() {
    check("full_memory_manage").await;
}

#[test]
fn every_case_has_a_test_here() {
    let here = include_str!("behavior.rs");
    for case in cases() {
        assert!(
            here.contains(&format!("async fn {}()", case.name)),
            "{}",
            case.name
        );
    }
}
