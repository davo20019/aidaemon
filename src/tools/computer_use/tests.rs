use serde_json::json;
use tempfile::TempDir;

use super::test_tool;
use crate::config::ComputerUseConfig;
use crate::traits::Tool;

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

fn test_model_args() -> serde_json::Value {
    json!({
        "_model": "gpt-4o",
        "_model_chain": ["gpt-4o"],
        "_provider_kind": "OpenaiCompatible"
    })
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
    let mut args = json!({
        "action": "get_app_state",
        "app": "Calculator",
        "_session_id": "telegram:1",
        "_task_id": "task-1"
    });
    if let Some(obj) = args.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    let outcome = tool
        .call_with_status_outcome(&args.to_string(), None)
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
    let mut inspect = json!({
        "action": "get_app_state",
        "app": "Calculator",
        "_session_id": "telegram:1",
        "_task_id": "task-1"
    });
    if let Some(obj) = inspect.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    tool.call_with_status_outcome(&inspect.to_string(), None)
        .await
        .unwrap();
    tool.call_with_status_outcome(&inspect.to_string(), None)
        .await
        .unwrap();
    let mut click = json!({
        "action": "click",
        "app": "Calculator",
        "snapshot_generation": 1,
        "element_index": 1,
        "_session_id": "telegram:1",
        "_task_id": "task-1"
    });
    if let Some(obj) = click.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    let outcome = tool
        .call_with_status_outcome(&click.to_string(), None)
        .await
        .unwrap();
    assert!(outcome.output.contains("Stale snapshot_generation"));
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
    let mut args = json!({
        "action": "get_app_state",
        "app": "Calculator",
        "_session_id": "telegram:1",
        "_task_id": "task-pin"
    });
    if let Some(obj) = args.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    tool.call_with_status_outcome(&args.to_string(), None)
        .await
        .unwrap();
    let pinned = ComputerUsePinRegistry::shared()
        .get("task-pin")
        .await
        .expect("expected model pin");
    assert_eq!(pinned, "gpt-4o");
}

#[tokio::test]
async fn consequential_allow_always_proceeds_as_one_time_allow() {
    use super::approvals::ApprovalState;
    use crate::tools::ApprovalBroker;
    use crate::types::ApprovalResponse;
    use tokio::sync::mpsc;

    let (tx, mut rx) = mpsc::channel(1);
    let broker = ApprovalBroker::new(tx);
    let state = ApprovalState::new();
    let responder = tokio::spawn(async move {
        let req = rx.recv().await.expect("approval request");
        let _ = req.response_tx.send(ApprovalResponse::AllowAlways);
    });
    let result = state
        .ensure_consequential(&broker, "telegram:1", "task-1", "Click 'Delete'")
        .await;
    responder.await.unwrap();
    assert!(
        result.is_ok(),
        "AllowAlways on a consequential action should proceed as a one-time allow: {result:?}"
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
    let mut args = json!({
        "action": "activate_app",
        "app": "Calculator",
        "_session_id": "telegram:1",
        "_task_id": "task-activate"
    });
    if let Some(obj) = args.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    let outcome = tool
        .call_with_status_outcome(&args.to_string(), None)
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
    let mut args = json!({
        "action": "click",
        "snapshot_generation": 1,
        "element_index": 2,
        "_session_id": "telegram:1",
        "_task_id": "task-noapp"
    });
    if let Some(obj) = args.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    let outcome = tool
        .call_with_status_outcome(&args.to_string(), None)
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
    let mut args = json!({
        "action": "click",
        "app": "Calculator",
        "element_index": 2,
        "_session_id": "telegram:1",
        "_task_id": "task-nogen"
    });
    if let Some(obj) = args.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    let outcome = tool
        .call_with_status_outcome(&args.to_string(), None)
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
    let mut args = json!({
        "action": "get_app_state",
        "app": "Slack",
        "_session_id": "telegram:1",
        "_task_id": "task-notrunning"
    });
    if let Some(obj) = args.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    let outcome = tool
        .call_with_status_outcome(&args.to_string(), None)
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
    let mut args = json!({
        "action": "launch_app",
        "app": "Calculator",
        "_session_id": "telegram:1",
        "_task_id": "task-launch"
    });
    if let Some(obj) = args.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    let outcome = tool
        .call_with_status_outcome(&args.to_string(), None)
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
    let mut inspect = json!({
        "action": "get_app_state",
        "app": "Calculator",
        "_session_id": "telegram:1",
        "_task_id": "task-budget"
    });
    if let Some(obj) = inspect.as_object_mut() {
        obj.extend(test_model_args().as_object().unwrap().clone());
    }
    tool.call_with_status_outcome(&inspect.to_string(), None)
        .await
        .unwrap();

    let click = |generation: u64, index: u32| {
        let mut args = json!({
            "action": "click",
            "app": "Calculator",
            "snapshot_generation": generation,
            "element_index": index,
            "_session_id": "telegram:1",
            "_task_id": "task-budget"
        });
        if let Some(obj) = args.as_object_mut() {
            obj.extend(test_model_args().as_object().unwrap().clone());
        }
        args
    };

    let first = tool
        .call_with_status_outcome(&click(1, 1).to_string(), None)
        .await
        .unwrap();
    assert!(!first.output.contains("budget exceeded"));

    let second = tool
        .call_with_status_outcome(&click(2, 2).to_string(), None)
        .await
        .unwrap();
    assert!(second.output.contains("budget exceeded"));
}
