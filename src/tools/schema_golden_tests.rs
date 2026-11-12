//! What the model sees of each general tool -- name, description and
//! parameter schema -- pinned byte for byte, so moving a tool into a crate
//! (Phase 9 step 4) cannot change it unnoticed.
//!
//! Regenerate on purpose with `UPDATE_TOOL_GOLDENS=1 cargo test schema_golden`.

use std::path::PathBuf;

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

#[test]
fn web_search_schema_golden() {
    check_golden(&crate::tools::WebSearchTool::new(
        &crate::config::SearchConfig::default(),
    ));
}

#[test]
fn http_request_schema_golden() {
    check_golden(&crate::tools::HttpRequestTool::new(std::sync::Arc::new(
        tokio::sync::RwLock::new(std::collections::HashMap::new()),
    )));
}

#[cfg(feature = "browser")]
#[tokio::test]
async fn browser_schema_golden() {
    let (media_tx, _media_rx) = tokio::sync::mpsc::channel(1);
    let tool = crate::tools::BrowserTool::new(
        crate::config::BrowserConfig::default().settings(),
        media_tx,
        std::env::temp_dir().join("aidaemon-golden-inbox"),
        crate::execution::active_execution_backend(),
    )
    .expect("browser tool");
    check_golden(&tool);
}

#[test]
fn send_file_schema_golden() {
    let (media_tx, _media_rx) = tokio::sync::mpsc::channel(1);
    check_golden(&crate::tools::SendFileTool::new(
        crate::execution::active_execution_backend(),
        media_tx,
        &[],
        "/tmp/aidaemon-golden-inbox",
    ));
}

#[test]
fn dev_tool_schema_goldens() {
    let backend = crate::execution::active_execution_backend;
    check_golden(&crate::tools::SystemInfoTool::new(backend()));
    check_golden(&crate::tools::GitInfoTool::new(backend()));
    check_golden(&crate::tools::GitCommitTool::new(backend()));
    check_golden(&crate::tools::ProjectInspectTool::new(backend()));
    check_golden(&crate::tools::CheckEnvironmentTool::new(backend()));
    check_golden(&crate::tools::ServiceStatusTool::new(backend()));
}

#[test]
fn list_checkpoints_schema_golden() {
    check_golden(&crate::tools::ListCheckpointsTool);
}
