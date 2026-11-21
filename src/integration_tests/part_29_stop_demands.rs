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

/// An empty final reply after real work is sent back once for a written
/// answer; the rewritten reply ships.
#[tokio::test]
async fn an_empty_reply_after_work_is_rewritten_once() {
    let written = "phase=WRITTEN; the system info is in.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(""),
        MockProvider::text_response(written),
        MockProvider::text_response(written),
    ]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, vec![])
        .await
        .unwrap();
    let session_id = "stop_demand_empty_reply";
    let response = harness
        .agent
        .handle_message(
            session_id,
            "What system is this running on?",
            None,
            UserRole::Owner,
            owner_channel(),
            None,
        )
        .await
        .unwrap();
    assert!(response.contains("phase=WRITTEN"), "{response}");
    assert_eq!(
        stop_demand_verdicts(&harness, session_id, "final_reply_defect").await,
        vec!["demand"],
        "one rewrite: {:?}",
        all_conditions(&harness, session_id).await
    );
}

/// A model that answers the rewrite with another empty reply does not
/// loop: the demand is spent after its one ask and the run ends.
#[tokio::test]
async fn a_second_empty_reply_does_not_loop() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::text_response(""),
        MockProvider::text_response(""),
        MockProvider::text_response(""),
        MockProvider::text_response(""),
    ]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, vec![])
        .await
        .unwrap();
    let session_id = "stop_demand_empty_twice";
    harness
        .agent
        .handle_message(
            session_id,
            "What system is this running on?",
            None,
            UserRole::Owner,
            owner_channel(),
            None,
        )
        .await
        .unwrap();
    let verdicts = stop_demand_verdicts(&harness, session_id, "final_reply_defect").await;
    assert_eq!(
        verdicts.iter().filter(|v| *v == "demand").count(),
        1,
        "one ask, never a loop: {verdicts:?}"
    );
}

/// One turn that declares a write, makes it through a wrapper the terminal
/// cannot see into (here `sh -c`, live `docker exec`), and replies; returns
/// the reply and the ledger's demands.
async fn opaque_write_turn(session_id: &str, claim: bool) -> (String, Vec<String>) {
    wrapped_turn(session_id, claim, "pending", "sh -c 'exit 0'").await
}

/// [`opaque_write_turn`] with the item's declared status and the command.
async fn wrapped_turn(
    session_id: &str,
    claim: bool,
    status: &str,
    command: &str,
) -> (String, Vec<String>) {
    let checklist = json!({"items": [{
        "text": "write the synthetic report inside the container",
        "status": status,
        "mutation_effects": ["local_workspace_write"]
    }]})
    .to_string();
    let mut run = json!({"action": "run", "command": command});
    if claim {
        run["checklist_items"] = json!([0]);
    }
    let done = "phase=DONE; the report is written in the container.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &checklist),
        MockProvider::tool_call_response("terminal", &run.to_string()),
        MockProvider::text_response(done),
        MockProvider::text_response(done),
        MockProvider::text_response(done),
        MockProvider::text_response(done),
    ]);
    let plan_pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let plan_store = Arc::new(crate::plans::PlanStore::new(plan_pool).await.unwrap());
    let track = Arc::new(crate::tools::track_requirements::TrackRequirementsTool::new(
        plan_store,
    ));
    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![track as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();
    let response = harness
        .agent
        .handle_message(
            session_id,
            "Write the synthetic report inside the container.",
            None,
            UserRole::Owner,
            owner_channel(),
            None,
        )
        .await
        .unwrap();
    let demands = all_conditions(&harness, session_id)
        .await
        .into_iter()
        .filter(|c| {
            c.starts_with("\"ledger_expectations_required\"")
                || c.starts_with("\"declared_work_unattempted\"")
        })
        .collect();
    (response, demands)
}

/// A wrapped command that says it does the declared item is the item's
/// proof: the ledger does not send the run back to redo finished work.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell wrapper; the typing is platform-neutral")]
async fn a_claimed_change_through_a_wrapper_closes_its_item() {
    let (response, demands) = opaque_write_turn("opaque_change_claimed", true).await;
    assert!(response.contains("phase=DONE"), "{response}");
    assert!(demands.is_empty(), "no extra pass: {demands:?}");
}

/// Without the claim, nothing ties the wrapped command to the item: the
/// ledger still asks, as before.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell wrapper; the typing is platform-neutral")]
async fn an_unclaimed_wrapped_command_does_not_close_the_item() {
    let (_, demands) = opaque_write_turn("opaque_change_unclaimed", false).await;
    assert!(!demands.is_empty(), "the item stays open without a claim");
}

/// A change through a wrapper that names no item, made while the model has
/// exactly one change item in progress, is that item's work.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell wrapper; the typing is platform-neutral")]
async fn an_unnamed_change_goes_to_the_item_in_progress() {
    let dir = tempfile::tempdir().expect("container stand-in");
    let file = dir.path().join("report.txt").to_string_lossy().to_string();
    let command = format!("sh -c 'touch {file}'");
    let (response, demands) =
        wrapped_turn("unnamed_change_in_progress", false, "in_progress", &command).await;
    assert!(response.contains("phase=DONE"), "{response}");
    assert!(demands.is_empty(), "no extra pass: {demands:?}");
}

/// A command whose change cannot be recognized is not credited without a
/// claim: the item stays open rather than being closed by a guess.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell wrapper; the typing is platform-neutral")]
async fn an_unrecognized_command_is_not_credited_to_the_item_in_progress() {
    let (_, demands) =
        wrapped_turn("unrecognized_in_progress", false, "in_progress", "sh -c 'exit 0'").await;
    assert!(!demands.is_empty(), "no credit from an unrecognized command");
}

/// A look is never the work: a read-only command made while an item is in
/// progress does not close it.
#[tokio::test]
async fn a_look_while_an_item_is_in_progress_does_not_close_it() {
    let (_, demands) =
        wrapped_turn("look_in_progress", false, "in_progress", "ls /").await;
    assert!(!demands.is_empty(), "the item stays open after a look");
}

/// A recognized change made before any checklist exists is work done: the
/// run is not sent back as if nothing had been attempted.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell wrapper; the typing is platform-neutral")]
async fn a_recognized_change_before_any_checklist_counts_as_attempted() {
    let dir = tempfile::tempdir().expect("container stand-in");
    let file = dir.path().join("report.txt").to_string_lossy().to_string();
    let run = json!({"action": "run", "command": format!("sh -c 'touch {file}'")}).to_string();
    let done = "phase=DONE; the report is written in the container.";
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("terminal", &run),
        MockProvider::text_response(done),
        MockProvider::text_response(done),
        MockProvider::text_response(done),
    ])
    .with_owner_units(vec![MockProvider::owner_units(&[(
        "Write the synthetic report inside the container",
        true,
    )])]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, vec![])
        .await
        .unwrap();
    let session_id = "recognized_change_no_checklist";
    let response = harness
        .agent
        .handle_message(
            session_id,
            "Write the synthetic report inside the container.",
            None,
            UserRole::Owner,
            owner_channel(),
            None,
        )
        .await
        .unwrap();
    assert!(response.contains("phase=DONE"), "{response}");
    let conditions = all_conditions(&harness, session_id).await;
    assert!(
        !conditions
            .iter()
            .any(|c| c.starts_with("\"declared_work_unattempted\"")),
        "{conditions:?}"
    );
    assert!(std::path::Path::new(&file).exists());
}
