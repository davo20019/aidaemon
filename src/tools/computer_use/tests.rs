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
