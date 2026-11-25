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
fn format_specialist_arg_description_lists_every_kind_with_text() {
    let registry = crate::specialists::SpecialistRegistry::load(None);
    let entries = registry.llm_visible_kinds();
    let desc = format_specialist_arg_description(&entries);

    // Each non-task_lead kind appears as a `- name:` bullet, and its
    // bundled frontmatter description text is included verbatim.
    for kind in [
        "code",
        "browser_verifier",
        "artifact_writer",
        "research",
        "review",
        "comms_draft",
        "executor",
        "generic",
    ] {
        let marker = format!("- {}:", kind);
        assert!(
            desc.contains(&marker),
            "specialist description missing bullet for {}: {}",
            kind,
            desc
        );
    }

    // task_lead must NOT appear — it's role-typed, not LLM-selectable.
    assert!(!desc.contains("- task_lead:"));
    assert!(!desc.contains("task_lead:"));

    // Spot-check the actual frontmatter descriptions surface through.
    // (Test asserts the registry's description text reached the formatter.)
    let code_desc = entries
        .iter()
        .find(|(n, _)| *n == "code")
        .map(|(_, d)| d.clone())
        .expect("code kind present");
    assert!(
        desc.contains(&code_desc),
        "code description not surfaced: {}",
        desc
    );

    // Closing line tells the model omission is allowed.
    assert!(desc.contains("Omit `specialist`"));
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
