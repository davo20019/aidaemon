//! Every bundled tool passes the core tool conformance kit.

use std::sync::Arc;

use aidaemon_core::conformance::{check_tool, check_tool_with, ConformanceReport, ToolKitOptions};
use aidaemon_core::Tool;
use aidaemon_tools::{default_tools, Checklist, Shell, Workspace};
use serde_json::json;

#[test]
fn bundled_tools_pass_the_tool_kit() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = Workspace::new(dir.path()).unwrap();

    let mut tools: Vec<Arc<dyn Tool>> = default_tools(workspace.clone());
    tools.push(Arc::new(Checklist::default()));
    tools.push(Arc::new(Shell::new(
        workspace,
        Arc::new(aidaemon_tools::LocalBackend),
    )));

    let mut all = ConformanceReport::new("aidaemon-tools");
    for tool in &tools {
        let report = match tool.spec().name.as_str() {
            // `validate` parses the URL, which the schema cannot express.
            "web_fetch" => check_tool_with(
                tool.as_ref(),
                ToolKitOptions::default().valid_arguments(json!({"url": "https://example.com/"})),
            ),
            _ => check_tool(tool.as_ref()),
        };
        all.merge(report);
    }
    all.assert_all_passed();
    // 8 tool instances x 10 checks each.
    assert_eq!(all.passed.len(), tools.len() * 10, "{all}");
}

#[test]
fn web_fetch_generated_sample_is_rejected_without_a_supplied_call() {
    // Documents why web_fetch needs `valid_arguments`: the generated string
    // is not a URL, and the kit says so instead of passing silently.
    let report = check_tool(&aidaemon_tools::WebFetch::new());
    let reason = report
        .failure("tool.validate_accepts_minimal_call")
        .expect("generated sample should be rejected");
    assert!(reason.contains("valid_arguments"), "{reason}");
}
