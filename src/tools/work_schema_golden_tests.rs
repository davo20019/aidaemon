//! What the model sees of each work tool -- name, description and parameter
//! schema -- pinned byte for byte, so moving a tool into `aidaemon-work`
//! (Phase 9 step 3) cannot change it unnoticed.
//!
//! Regenerate on purpose with `UPDATE_TOOL_GOLDENS=1 cargo test work_schema_golden`.

use std::path::PathBuf;

use crate::testing::{setup_test_agent, MockProvider};
use crate::traits::Tool;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/tools/schema_goldens")
}

fn check_golden(tool: &dyn Tool) {
    let value = serde_json::json!({
        "name": tool.name(),
        "description": tool.description(),
        "schema": tool.schema(),
    });
    let rendered = format!("{}\n", serde_json::to_string_pretty(&value).unwrap());
    let path = goldens_dir().join(format!("{}.golden", tool.name()));
    if std::env::var_os("UPDATE_TOOL_GOLDENS").is_some() {
        std::fs::create_dir_all(goldens_dir()).unwrap();
        std::fs::write(&path, &rendered).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing golden {}", path.display()));
    assert_eq!(
        rendered,
        expected,
        "{} schema changed; if on purpose, rerun with UPDATE_TOOL_GOLDENS=1",
        tool.name()
    );
}

#[tokio::test]
async fn work_schema_golden_manage_mandates() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    check_golden(&crate::tools::ManageMandatesTool::new(harness.state));
}

/// New with the one automation list (2026-11-12): the owner's automations of
/// both kinds, and pause/resume/cancel by name.
#[tokio::test]
async fn work_schema_golden_manage_automations() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    let memory = harness.state.memory_stores();
    check_golden(&crate::tools::ManageAutomationsTool::new(
        harness.state,
        memory,
    ));
}
