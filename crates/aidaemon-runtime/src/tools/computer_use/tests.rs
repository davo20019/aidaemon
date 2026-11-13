use aidaemon_tools::testing::ToolCalls as _;
use serde_json::json;
use tempfile::TempDir;

use super::test_tool;
use crate::settings::ComputerUseConfig;
use crate::tools::traits::Tool;

#[test]
fn lock_banner_stamps_observations_when_locked() {
    // Live 2026-07-12: observations succeed while the screen is locked, so the
    // model only discovered the lock when input bounced — then flailed via
    // AppleScript instead of telling the user.
    let stamped =
        super::ComputerUseTool::apply_lock_banner("Running apps:\n- Calculator".to_string(), true);
    assert!(stamped.contains("SCREEN IS LOCKED"), "got: {stamped}");
    assert!(stamped.contains("Running apps:"));
    let clean =
        super::ComputerUseTool::apply_lock_banner("Running apps:\n- Calculator".to_string(), false);
    assert_eq!(clean, "Running apps:\n- Calculator");
}

#[test]
fn bounds_match_reidentifies_control_across_index_renumber() {
    use crate::tools::computer_use::types::ElementBounds;
    let b = |x: f64, y: f64| ElementBounds {
        x,
        y,
        width: 40.0,
        height: 40.0,
    };
    // Same on-screen spot (within tolerance) → same control, even though its
    // index renumbered after the re-render. This is what makes the post-click
    // "did it change?" check fire when a Like button is still "no reaction".
    assert!(super::bounds_match(
        Some(b(100.0, 200.0)),
        Some(b(108.0, 205.0))
    ));
    // A different post's like button (far away) is NOT the same control.
    assert!(!super::bounds_match(
        Some(b(100.0, 200.0)),
        Some(b(100.0, 600.0))
    ));
    // Missing bounds conservatively match (prefer flagging a possible no-op).
    assert!(super::bounds_match(None, Some(b(100.0, 200.0))));
}

#[tokio::test]
async fn schema_includes_action_enum() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let schema = tool.schema();
    let actions = schema["parameters"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap();
    assert!(actions.iter().any(|v| v == "get_app_state"));
}

/// The dispatcher's facts for a computer_use call on `task`: the session
/// and the live vision-capable model.
fn cu_facts(task: &str) -> crate::tools::traits::ToolCallContext {
    crate::tools::traits::ToolCallContext {
        task_id: Some(task.to_string()),
        model: Some("gpt-4o".to_string()),
        model_chain: Some(vec!["gpt-4o".to_string()]),
        provider_kind: Some("OpenaiCompatible".to_string()),
        ..aidaemon_tools::testing::call_facts("telegram:1")
    }
}

#[tokio::test]
async fn mock_get_app_state_returns_generation_and_attachment() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    let outcome = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-1"))
        .await
        .unwrap();
    assert!(outcome.output.contains("snapshot_generation="));
    assert_eq!(outcome.metadata.attachments.len(), 1);
    assert_eq!(
        outcome.metadata.attachments[0].source_tool.as_deref(),
        Some("computer_use")
    );
}

#[tokio::test]
async fn stale_generation_is_rejected() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let inspect = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&inspect.to_string(), None, cu_facts("task-1"))
        .await
        .unwrap();
    tool.call_with_execution_context(&inspect.to_string(), None, cu_facts("task-1"))
        .await
        .unwrap();
    let click = json!({
        "action": "click",
        "app": "Calculator",
        "snapshot_generation": 1,
        "element_index": 1
    });
    let outcome = tool
        .call_with_execution_context(&click.to_string(), None, cu_facts("task-1"))
        .await
        .unwrap();
    assert!(outcome.output.contains("Stale snapshot_generation"));
}

#[tokio::test]
async fn coordinate_click_flags_unverified_and_observation_clears_it() {
    use super::pin_registry::ComputerUsePinRegistry;
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    // Unique task id: the pin registry is process-wide (shared OnceLock).
    let task = "task-coord-verify-1";
    let reg = ComputerUsePinRegistry::shared();

    // Pin the task + get a generation.
    let inspect = json!({
        "action": "get_app_state", "app": "Calculator"
    });
    tool.call_with_execution_context(&inspect.to_string(), None, cu_facts(task))
        .await
        .unwrap();
    assert!(!reg.has_unverified_coordinate_click(task).await);

    // A coordinate click: flags the result unverified and marks the task.
    let click = json!({
        "action": "click", "app": "Calculator",
        "snapshot_generation": 1, "x": 500.0, "y": 500.0
    });
    let outcome = tool
        .call_with_execution_context(&click.to_string(), None, cu_facts(task))
        .await
        .unwrap();
    assert!(
        outcome.output.contains("[UNVERIFIED]"),
        "coordinate click must carry the unverified notice: {}",
        outcome.output
    );
    assert!(
        reg.has_unverified_coordinate_click(task).await,
        "coordinate click must mark the task unverified"
    );

    // A deliberate follow-up observation clears the flag.
    tool.call_with_execution_context(&inspect.to_string(), None, cu_facts(task))
        .await
        .unwrap();
    assert!(
        !reg.has_unverified_coordinate_click(task).await,
        "a verifying observation must clear the flag"
    );

    reg.clear_task(task).await;
}

#[tokio::test]
async fn list_apps_works_without_session() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({ "action": "list_apps" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Calculator"),
        "unexpected list_apps output: {out:?}"
    );
}

#[tokio::test]
async fn model_pin_is_set_on_first_gui_action() {
    use super::pin_registry::ComputerUsePinRegistry;

    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&args.to_string(), None, cu_facts("task-pin"))
        .await
        .unwrap();
    let pinned = ComputerUsePinRegistry::shared()
        .get("task-pin")
        .await
        .expect("expected model pin");
    assert_eq!(pinned, "gpt-4o");
}

#[test]
fn consequential_subject_binds_the_exact_action() {
    let one = super::consequential_subject(
        "com.example.App",
        "Type text into Example",
        &json!({"action": "type_text", "text": "one"}),
    );
    let two = super::consequential_subject(
        "com.example.App",
        "Type text into Example",
        &json!({"action": "type_text", "text": "two"}),
    );
    assert_ne!(one, two);
    assert!(!one.contains("one"), "{one}");
    // Key order in the arguments does not matter.
    assert_eq!(
        one,
        super::consequential_subject(
            "com.example.App",
            "Type text into Example",
            &json!({"text": "one", "action": "type_text"}),
        )
    );
}

#[tokio::test]
async fn app_control_is_asked_through_the_kernel() {
    use crate::tools::traits::{ApprovalCall, OwnerApprovalNeed, ToolSemantics};
    use aidaemon_tools::semantics::{ApprovalKind, ApprovalResponse};

    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    assert!(tool.dispatch_facts().keeps_own_grants);
    // `test_tool` approves Calculator for telegram:1 only.
    let facts = crate::tools::traits::ToolCallContext {
        session_id: "telegram:2".into(),
        ..cu_facts("task-kernel")
    };
    let args = json!({"action": "get_app_state", "app": "Calculator"});
    let raw = args.to_string();

    // Never approved: refused, not asked about inside the tool.
    let out = aidaemon_tools::testing::call_with_facts(&tool, &raw, facts.clone())
        .await
        .unwrap();
    assert!(out.contains("through the agent kernel"), "{out}");

    let OwnerApprovalNeed::Ask(request) = tool
        .owner_approval_request(&ApprovalCall::as_sent(&args), &facts)
        .await
    else {
        panic!("an app not approved in the session must ask");
    };
    assert!(
        request.summary.contains("com.apple.calculator"),
        "{}",
        request.summary
    );
    assert!(matches!(
        tool.approval_card_kind(&request),
        ApprovalKind::Command
    ));

    // Allow Once covers one call.
    let run = aidaemon_tools::testing::call_through_kernel(
        &tool,
        &raw,
        facts.clone(),
        Some(ApprovalResponse::AllowOnce),
    )
    .await
    .unwrap();
    assert!(run.asked.is_some());
    assert!(
        !run.output.contains("through the agent kernel"),
        "{}",
        run.output
    );
    let out = aidaemon_tools::testing::call_with_facts(&tool, &raw, facts.clone())
        .await
        .unwrap();
    assert!(out.contains("through the agent kernel"), "{out}");

    // A decline keeps nothing.
    let run = aidaemon_tools::testing::call_through_kernel(
        &tool,
        &raw,
        facts.clone(),
        Some(ApprovalResponse::Deny),
    )
    .await
    .unwrap();
    assert!(run.asked.is_some());
    assert!(run.output.contains("Denied"), "{}", run.output);

    // Allow Session covers the session's later calls without a card.
    let run = aidaemon_tools::testing::call_through_kernel(
        &tool,
        &raw,
        facts.clone(),
        Some(ApprovalResponse::AllowSession),
    )
    .await
    .unwrap();
    assert!(run.asked.is_some());
    let run = aidaemon_tools::testing::call_through_kernel(
        &tool,
        &raw,
        facts.clone(),
        Some(ApprovalResponse::Deny),
    )
    .await
    .unwrap();
    assert!(run.asked.is_none(), "an approved session needs no card");
    assert!(
        !run.output.contains("through the agent kernel"),
        "{}",
        run.output
    );
}

#[tokio::test]
async fn activate_app_without_generation_succeeds() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "activate_app",
        "app": "Calculator"
    });
    let outcome = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-activate"))
        .await
        .unwrap();
    assert!(
        !outcome.output.starts_with("Error:"),
        "activate_app should not require snapshot_generation: {}",
        outcome.output
    );
    assert!(outcome.output.contains("Calculator"));
}

#[tokio::test]
async fn activate_app_treats_zero_generation_as_omitted() {
    // Repro from live telemetry (2026-07-30): the model adapter serialized an
    // optional snapshot_generation as 0. Activation then tried to validate a
    // nonexistent generation and failed before it could recover a windowless
    // Calculator process.
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "activate_app",
        "app": "Calculator",
        "snapshot_generation": 0
    });
    let outcome = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-activate-zero"))
        .await
        .unwrap();
    assert!(
        !outcome.output.starts_with("Error:"),
        "zero should mean no optional generation for activate_app: {}",
        outcome.output
    );
    assert!(outcome.output.contains("Calculator"));
}

#[tokio::test]
async fn missing_app_error_is_instructional() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "click",
        "snapshot_generation": 1,
        "element_index": 2
    });
    let outcome = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-noapp"))
        .await
        .unwrap();
    assert!(
        outcome.output.contains("repeat the same call with app set"),
        "missing-app error should tell the model the literal next step: {}",
        outcome.output
    );
}

#[tokio::test]
async fn missing_generation_error_is_instructional() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "click",
        "app": "Calculator",
        "element_index": 2
    });
    let outcome = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-nogen"))
        .await
        .unwrap();
    assert!(
        outcome
            .output
            .contains("copy the snapshot_generation value"),
        "missing-generation error should tell the model the literal next step: {}",
        outcome.output
    );
}

#[tokio::test]
async fn schema_documents_requirements_and_verification() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let schema = tool.schema();
    let description = schema["description"].as_str().unwrap();
    assert!(
        description.contains("before reporting success"),
        "tool description should require verifying state before claiming success: {description}"
    );
    let props = &schema["parameters"]["properties"];
    let app_desc = props["app"]["description"].as_str().unwrap();
    assert!(
        app_desc.contains("Required for every action except list_apps"),
        "app param should document when it is required: {app_desc}"
    );
    let gen_desc = props["snapshot_generation"]["description"]
        .as_str()
        .unwrap();
    assert!(
        gen_desc.contains("optional for activate_app"),
        "snapshot_generation should document the activate_app exemption: {gen_desc}"
    );
}

#[tokio::test]
async fn schema_includes_launch_app_action() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let schema = tool.schema();
    let actions = schema["parameters"]["properties"]["action"]["enum"]
        .as_array()
        .unwrap();
    assert!(
        actions.iter().any(|v| v == "launch_app"),
        "launch_app should be a documented action: {actions:?}"
    );
}

#[tokio::test]
async fn not_running_app_error_points_to_launch_app() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    // The mock only knows Calculator, so any other app resolves as "not running".
    let args = json!({
        "action": "get_app_state",
        "app": "Slack"
    });
    let outcome = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-notrunning"))
        .await
        .unwrap();
    assert!(
        outcome.output.contains("launch_app"),
        "a not-running app should tell the model to launch it instead of dead-ending: {}",
        outcome.output
    );
}

#[tokio::test]
async fn launch_app_returns_tree_for_running_app() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "launch_app",
        "app": "Calculator"
    });
    let outcome = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-launch"))
        .await
        .unwrap();
    assert!(
        !outcome.output.starts_with("Error:"),
        "launch_app should succeed for a known app: {}",
        outcome.output
    );
    assert!(
        outcome.output.contains("snapshot_generation="),
        "launch_app should return an inspectable tree with a generation: {}",
        outcome.output
    );
    assert_eq!(outcome.metadata.attachments.len(), 1);
}

#[tokio::test]
async fn screenshot_does_not_invalidate_generation() {
    // Repro of the real failure: get_app_state -> screenshot -> type_text reusing
    // the generation get_app_state reported. A screenshot must NOT advance the
    // generation, so the mutation must not be rejected as stale.
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;

    // Fresh cache: the first get_app_state stores generation 1.
    let gs = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    let gs_out = tool
        .call_with_execution_context(&gs.to_string(), None, cu_facts("task-shot"))
        .await
        .unwrap();
    assert!(gs_out.output.contains("snapshot_generation=1"));

    // A screenshot in between must report the current generation and not bump it.
    let shot = json!({
        "action": "screenshot",
        "app": "Calculator"
    });
    let shot_out = tool
        .call_with_execution_context(&shot.to_string(), None, cu_facts("task-shot"))
        .await
        .unwrap();
    assert!(
        !shot_out.output.starts_with("Error:"),
        "screenshot should succeed: {}",
        shot_out.output
    );
    assert!(
        shot_out.output.contains("snapshot_generation=1"),
        "screenshot should report the still-current generation: {}",
        shot_out.output
    );

    // Reusing generation 1 (what get_app_state reported) must still be valid.
    let tt = json!({
        "action": "type_text",
        "app": "Calculator",
        "snapshot_generation": 1,
        "text": "hello"
    });
    let tt_out = tool
        .call_with_execution_context(&tt.to_string(), None, cu_facts("task-shot"))
        .await
        .unwrap();
    assert!(
        !tt_out.output.contains("Stale"),
        "a screenshot must not invalidate the working generation: {}",
        tt_out.output
    );
}

#[tokio::test]
async fn type_text_by_descriptor_resolves_and_focuses() {
    // get_app_state then type_text targeting a field by element_title (not index).
    // The mock validates the resolved index exists, so success means the
    // descriptor resolved to a real element.
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let gs = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&gs.to_string(), None, cu_facts("task-desc"))
        .await
        .unwrap();

    // Mock snapshot has buttons titled "7" (index 1) and "+" (index 2).
    let tt = json!({
        "action": "type_text",
        "app": "Calculator",
        "snapshot_generation": 1,
        "element_title": "+",
        "text": "hi"
    });
    let out = tool
        .call_with_execution_context(&tt.to_string(), None, cu_facts("task-desc"))
        .await
        .unwrap();
    assert!(
        !out.output.starts_with("Error:"),
        "type_text by descriptor should resolve to a real element: {}",
        out.output
    );
}

#[tokio::test]
async fn descriptor_with_no_match_is_actionable() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let gs = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&gs.to_string(), None, cu_facts("task-nomatch"))
        .await
        .unwrap();

    let click = json!({
        "action": "click",
        "app": "Calculator",
        "snapshot_generation": 1,
        "element_title": "Nonexistent Button"
    });
    let out = tool
        .call_with_execution_context(&click.to_string(), None, cu_facts("task-nomatch"))
        .await
        .unwrap();
    assert!(
        out.output.contains("No interactive element matching"),
        "an unmatched descriptor should give an actionable error: {}",
        out.output
    );
}

#[tokio::test]
async fn schema_documents_descriptor_targeting() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let schema = tool.schema();
    let props = &schema["parameters"]["properties"];
    assert!(props.get("element_title").is_some(), "element_title param");
    assert!(props.get("element_role").is_some(), "element_role param");
    assert!(props.get("occurrence").is_some(), "occurrence param");
    let text_desc = props["text"]["description"].as_str().unwrap();
    assert!(
        text_desc.contains("element_title") || text_desc.contains("focus"),
        "text description should explain focusing the target field: {text_desc}"
    );
}

#[tokio::test]
async fn scroll_without_a_target_element_succeeds() {
    // "scroll the page" — no element_index/descriptor — must work now.
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let gs = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&gs.to_string(), None, cu_facts("task-scroll"))
        .await
        .unwrap();

    let scroll = json!({
        "action": "scroll",
        "app": "Calculator",
        "snapshot_generation": 1,
        "direction": "down"
    });
    let out = tool
        .call_with_execution_context(&scroll.to_string(), None, cu_facts("task-scroll"))
        .await
        .unwrap();
    assert!(
        !out.output.contains("requires element_index"),
        "scroll should no longer require an element target: {}",
        out.output
    );
    assert!(!out.output.starts_with("Error:"), "{}", out.output);
}

#[tokio::test]
async fn click_with_no_target_change_is_flagged_for_verification() {
    // The mock click never changes the clicked element's role/title, so the
    // element-specific verifier should warn that the target did not change —
    // exactly the signal that catches a web Like that silently did nothing.
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let gs = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&gs.to_string(), None, cu_facts("task-verify"))
        .await
        .unwrap();

    let click = json!({
        "action": "click",
        "app": "Calculator",
        "snapshot_generation": 1,
        "element_index": 1
    });
    let out = tool
        .call_with_execution_context(&click.to_string(), None, cu_facts("task-verify"))
        .await
        .unwrap();
    assert!(
        out.output.contains("[VERIFY]"),
        "a click whose target element is unchanged should be flagged: {}",
        out.output
    );
}

#[tokio::test]
async fn repeated_identical_click_is_cautioned() {
    // Clicking the same element twice with no get_app_state in between (the
    // pattern that can toggle a Like back off) appends a caution.
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let mk = |generation: u64| {
        json!({
            "action": "click",
            "app": "Calculator",
            "snapshot_generation": generation,
            "element_index": 1
        })
    };
    let gs = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&gs.to_string(), None, cu_facts("task-dup"))
        .await
        .unwrap();

    let first = tool
        .call_with_execution_context(&mk(1).to_string(), None, cu_facts("task-dup"))
        .await
        .unwrap();
    assert!(
        !first.output.contains("[NOTE]"),
        "first click should not be cautioned: {}",
        first.output
    );
    let second = tool
        .call_with_execution_context(&mk(2).to_string(), None, cu_facts("task-dup"))
        .await
        .unwrap();
    assert!(
        second.output.contains("[NOTE]") && second.output.contains("UNDO"),
        "repeating the same click should be cautioned: {}",
        second.output
    );
}

#[tokio::test]
async fn results_carry_the_gui_task_anchor() {
    // Every computer_use result reminds the model to emit a computer_use call
    // (not narrate it in a file) — the fix for the observed derailment.
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let args = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    let out = tool
        .call_with_execution_context(&args.to_string(), None, cu_facts("task-anchor"))
        .await
        .unwrap();
    assert!(
        out.output.contains("[NEXT]") && out.output.contains("computer_use call"),
        "result should carry the GUI task anchor: {}",
        out.output
    );
}

#[tokio::test]
async fn mutation_budget_blocks_after_limit() {
    let dir = TempDir::new().unwrap();
    let tool = test_tool(
        ComputerUseConfig {
            enabled: true,
            max_mutating_actions: 1,
            ..Default::default()
        },
        dir.path().to_path_buf(),
    )
    .await;
    let inspect = json!({
        "action": "get_app_state",
        "app": "Calculator"
    });
    tool.call_with_execution_context(&inspect.to_string(), None, cu_facts("task-budget"))
        .await
        .unwrap();

    let click = |generation: u64, index: u32| {
        json!({
            "action": "click",
            "app": "Calculator",
            "snapshot_generation": generation,
            "element_index": index
        })
    };

    let first = tool
        .call_with_execution_context(&click(1, 1).to_string(), None, cu_facts("task-budget"))
        .await
        .unwrap();
    assert!(!first.output.contains("budget exceeded"));

    let second = tool
        .call_with_execution_context(&click(2, 2).to_string(), None, cu_facts("task-budget"))
        .await
        .unwrap();
    assert!(second.output.contains("budget exceeded"));
}
