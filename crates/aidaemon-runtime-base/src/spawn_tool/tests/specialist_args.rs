use super::*;
use serde_json::Value;

fn schema_props() -> Value {
    let tool = SpawnAgentTool::new_deferred(8192, 60);
    let schema = tool.schema();
    schema
        .get("parameters")
        .and_then(|p| p.get("properties"))
        .cloned()
        .expect("schema has properties")
}

#[test]
fn schema_advertises_specialist_arg_with_enum() {
    let props = schema_props();
    let specialist = props
        .get("specialist")
        .expect("specialist property declared in schema");
    let kinds = specialist
        .get("enum")
        .and_then(|v| v.as_array())
        .expect("specialist has enum array");
    let names: Vec<&str> = kinds.iter().filter_map(|v| v.as_str()).collect();
    for expected in [
        "code",
        "browser_verifier",
        "artifact_writer",
        "research",
        "review",
        "comms_draft",
        "executor",
        "generic",
    ] {
        assert!(
            names.contains(&expected),
            "missing {} from enum: {:?}",
            expected,
            names
        );
    }
    // task_lead is NOT a parent-LLM-selectable value (role-typed only).
    assert!(!names.contains(&"task_lead"));
}

#[test]
fn schema_falls_back_to_static_description_when_agent_unwired() {
    let tool = SpawnAgentTool::new_deferred(8192, 60);
    let schema = tool.schema();
    let desc = schema
        .get("parameters")
        .and_then(|p| p.get("properties"))
        .and_then(|p| p.get("specialist"))
        .and_then(|s| s.get("description"))
        .and_then(|d| d.as_str())
        .expect("specialist description string");
    assert_eq!(desc, STATIC_SPECIALIST_ARG_DESCRIPTION);
}
