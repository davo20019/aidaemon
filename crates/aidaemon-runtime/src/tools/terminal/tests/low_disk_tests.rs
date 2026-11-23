use super::*;

#[test]
fn a_failure_on_a_nearly_full_disk_names_the_free_space() {
    let gib = 1024 * 1024 * 1024;
    let path = std::path::Path::new("/Users/synthetic-user");
    let note = low_disk_message(path, 2 * gib, 460 * gib).expect("low disk is reported");
    assert!(note.contains("only 2.1 GB free of 494 GB"), "{note}");
    assert!(low_disk_message(path, 50 * gib, 460 * gib).is_none());
    assert!(low_disk_message(path, 0, 0).is_none());
    assert!(disk_space(std::path::Path::new("/")).is_some_and(|(_, total)| total > 0));
}
#[tokio::test]
async fn a_run_with_nothing_to_run_is_invalid_before_any_approval() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let pool = SqlitePool::connect(&format!("sqlite:{}", db_file.path().display()))
        .await
        .unwrap();
    let (_approval_tx, approval_rx) = mpsc::channel::<ApprovalRequest>(1);
    drop(approval_rx);
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec![],
        1,
        1000,
        PermissionMode::Default,
        pool,
    )
    .await;
    let refused = aidaemon_tools::outcome::check_argument_contract(&tool, &serde_json::json!({}))
        .unwrap_err();
    assert!(refused.contains("requires command or script"), "{refused}");
    aidaemon_tools::outcome::check_argument_contract(&tool, &serde_json::json!({"command": "ls"}))
        .unwrap();
}
