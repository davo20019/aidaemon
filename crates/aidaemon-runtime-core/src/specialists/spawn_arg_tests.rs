//! The `spawn_agent` tool's `specialist` argument lists the registry's kinds.

use crate::spawn_tool::format_specialist_arg_description;

#[test]
fn format_specialist_arg_description_lists_every_kind_with_text() {
    let registry = super::SpecialistRegistry::load(None);
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
