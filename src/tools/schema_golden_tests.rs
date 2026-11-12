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
