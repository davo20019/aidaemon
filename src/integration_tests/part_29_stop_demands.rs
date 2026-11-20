// ==================== Stop demands, end to end ====================
//
// Each `StopDemand` is judged by the pure verdict function (unit tests in
// `turn::closeout`); these drive the real turn so the demand's wiring is
// covered too: what records the need, the one bounded ask, what clears it,
// and that an ignored ask still lets the run stop.

fn owner_channel() -> ChannelContext {
    ChannelContext {
        visibility: ChannelVisibility::Private,
        platform: "telegram".to_string(),
        channel_name: None,
        channel_id: None,
        workspace_id: None,
        sender_name: Some("Alice".to_string()),
        sender_id: Some("telegram:synthetic-owner".to_string()),
        channel_member_names: vec![],
        user_id_map: std::collections::HashMap::new(),
        workspace_grant: None,
        trusted: false,
    }
}

/// The `stop_demand` verdicts recorded for `demand` in this session, in order.
async fn stop_demand_verdicts(
    harness: &crate::testing::FullStackTestHarness,
    session_id: &str,
    demand: &str,
) -> Vec<String> {
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    rows.iter()
        .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .filter(|data| {
            data["metadata"]["condition"] == "stop_demand" && data["metadata"]["demand"] == demand
        })
        .filter_map(|data| data["metadata"]["verdict"].as_str().map(str::to_string))
        .collect()
}

fn write_file_tool() -> Arc<dyn crate::traits::Tool> {
    Arc::new(crate::tools::WriteFileTool::new(
        crate::execution::active_execution_backend(),
    ))
}

/// Every decision-point condition this session recorded, for failures.
async fn all_conditions(harness: &crate::testing::FullStackTestHarness, session_id: &str) -> Vec<String> {
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE session_id = ? AND event_type = 'decision_point' ORDER BY id",
    )
    .bind(session_id)
    .fetch_all(&harness.state.pool())
    .await
    .unwrap();
    rows.iter()
        .filter_map(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .map(|data| format!("{}:{}", data["metadata"]["condition"], data["metadata"]["demand"]))
        .collect()
}

/// A synthetic repository with one source file to change.
fn synthetic_repo() -> (tempfile::TempDir, String) {
    let workspace = tempfile::tempdir().expect("workspace");
    std::fs::create_dir_all(workspace.path().join(".git")).unwrap();
    std::fs::create_dir_all(workspace.path().join("src")).unwrap();
    let file = workspace
        .path()
        .join("src/lib.rs")
        .to_string_lossy()
        .to_string();
    std::fs::write(&file, "pub fn one() -> u32 { 1 }\n").unwrap();
    (workspace, file)
}

/// Two stop checks want the same unchecked write: the ledger wants it
/// observed and the code check wants a command after it. They compose into
/// one extra pass, not two: the ledger asks first, the command the model
/// runs answers both, and the code check never has to ask.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell command; the demands are platform-neutral")]
async fn one_command_answers_both_the_ledger_and_the_code_check() {
    let (workspace, file) = synthetic_repo();
    let cwd = workspace.path().to_string_lossy().to_string();
    let write_args = json!({"path": file, "content": "pub fn one() -> u32 { 1 }\npub fn two() -> u32 { 2 }\n"}).to_string();
    let check_args = json!({
        "command": format!("cat '{file}'"),
        "working_dir": cwd,
        "read_paths": [file],
    })
    .to_string();
    let early = "phase=EARLY; added two().";
    let checked = "phase=CHECKED; added two() and read it back.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("write_file", &write_args),
        MockProvider::text_response(early),
        MockProvider::tool_call_response("terminal", &check_args),
        MockProvider::text_response(checked),
        MockProvider::text_response(checked),
    ]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, vec![write_file_tool()])
        .await
        .unwrap();
    let session_id = "stop_demand_code_check";
    let response = harness
        .agent
        .handle_message(
            session_id,
            &format!("Add a two() function next to one() in {file}."),
            None,
            UserRole::Owner,
            owner_channel(),
            None,
        )
        .await
        .unwrap();

    assert!(
        std::fs::read_to_string(&file)
            .unwrap()
            .contains("fn two()"),
        "the change must have been written"
    );
    assert!(
        response.contains(checked) && !response.contains(early),
        "the unchecked reply must not ship: {response}"
    );
    let conditions = all_conditions(&harness, session_id).await;
    assert_eq!(
        conditions
            .iter()
            .filter(|c| c.starts_with("\"ledger_expectations_required\""))
            .count(),
        1,
        "one extra pass: {conditions:?}"
    );
    assert!(
        stop_demand_verdicts(&harness, session_id, "code_change_unchecked")
            .await
            .iter()
            .all(|v| v != "demand"),
        "the command cleared the code check before it could ask: {conditions:?}"
    );
}

/// The ask is bounded: a model that answers it with prose instead of a
/// command still stops after the one ask, never in a loop.
#[tokio::test]
async fn an_ignored_code_check_ask_stops_after_its_one_ask() {
    let (_workspace, file) = synthetic_repo();
    let write_args = json!({"path": file, "content": "pub fn one() -> u32 { 11 }\n"}).to_string();
    let reply = "phase=PROSE; changed one() to return 11. No check can run here.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("write_file", &write_args),
        MockProvider::text_response(reply),
        MockProvider::text_response(reply),
        MockProvider::text_response(reply),
        MockProvider::text_response(reply),
    ]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, vec![write_file_tool()])
        .await
        .unwrap();
    let session_id = "stop_demand_code_check_ignored";
    let response = harness
        .agent
        .handle_message(
            session_id,
            &format!("Make one() in {file} return 11."),
            None,
            UserRole::Owner,
            owner_channel(),
            None,
        )
        .await
        .unwrap();

    assert!(response.contains("phase=PROSE"), "{response}");
    let verdicts = stop_demand_verdicts(&harness, session_id, "code_change_unchecked").await;
    assert_eq!(
        verdicts.iter().filter(|v| *v == "demand").count(),
        1,
        "one ask, never a loop: {verdicts:?}"
    );
    assert!(
        verdicts.len() <= 1 || verdicts[1..].iter().all(|v| v == "exhausted"),
        "after its ask the demand is spent: {verdicts:?}"
    );
}
