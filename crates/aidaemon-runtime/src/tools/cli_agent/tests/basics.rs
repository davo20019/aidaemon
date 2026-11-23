//! Running, listing and cancelling, schema and argument aliases.

use super::*;

/// The model-facing descriptions must not promise an auto-approve mode.
#[test]
fn default_descriptions_do_not_claim_auto_approve() {
    for (name, _, _, description) in default_tool_definitions() {
        assert!(
            !description.to_ascii_lowercase().contains("auto-approve"),
            "{name}: {description}"
        );
    }
}
#[tokio::test]
async fn typed_workspace_mode_controls_delegation_semantics() {
    let (tool, _db_file) = setup_echo_tool().await;
    let read_only = tool.call_semantics(
        r#"{"action":"run","workspace_mode":"read_only","working_dir":"/tmp","prompt":"inspect"}"#,
    );
    let read_write = tool.call_semantics(
        r#"{"action":"run","workspace_mode":"read_write","working_dir":"/tmp","prompt":"change"}"#,
    );

    assert!(read_only.observes_state());
    assert!(!read_only.mutates_state());
    assert!(read_write.observes_state());
    assert!(read_write.mutates_state());

    let schema = tool.schema();
    assert_eq!(
        schema["parameters"]["properties"]["workspace_mode"]["enum"],
        json!(["read_only", "read_write"])
    );
    assert!(
        schema["parameters"]["properties"]["write_paths"]["description"]
            .as_str()
            .is_some_and(|description| description.contains("Required and non-empty"))
    );
}

#[tokio::test]
async fn test_run_echo_returns_output() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool
        .call(r#"{"action":"run","tool":"echo","prompt":"hello world"}"#)
        .await
        .unwrap();
    assert!(
        result.contains("hello world"),
        "Expected 'hello world' in output, got: {}",
        result
    );
}

#[tokio::test]
async fn test_run_bash_script_returns_output() {
    let (tool, _db) = setup_bash_tool().await;
    let result = tool
        .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo 'test output 42'"}"#)
        .await
        .unwrap();
    assert!(
        result.contains("test output 42"),
        "Expected 'test output 42' in output, got: {}",
        result
    );
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[tokio::test]
async fn a_process_the_agent_leaves_running_is_reported_not_stopped() {
    // A double-forked sleep outlives the agent. On macOS `/bin/sleep`
    // hides its environment, so this also proves the process-group
    // path; on Linux both its tag and its group find it.
    let (tool, _db) = setup_bash_tool().await;
    let seconds = 3000 + (std::process::id() % 900);
    let prompt = format!("(sleep {seconds} >/dev/null 2>&1 &); echo started");
    let result = tool
        .call(&json!({"action": "run", "tool": "bash-agent", "prompt": prompt}).to_string())
        .await
        .unwrap();
    let pids: Vec<u32> = result
        .lines()
        .filter(|line| line.ends_with(&format!("sleep {seconds}")))
        .filter_map(|line| line.strip_prefix("- pid ")?.split(':').next()?.parse().ok())
        .collect();
    // It is still alive: reported, not stopped.
    let alive: Vec<bool> = pids
        .iter()
        // SAFETY: signal 0 only checks that the pid exists.
        .map(|pid| unsafe { libc::kill(*pid as libc::pid_t, 0) } == 0)
        .collect();
    for pid in &pids {
        // SAFETY: kills only the sleep this test's agent started.
        unsafe { libc::kill(*pid as libc::pid_t, libc::SIGKILL) };
    }
    assert!(result.contains("## Left running"), "{result}");
    assert_eq!(pids.len(), 1, "{result}");
    assert_eq!(alive, vec![true]);
}

#[tokio::test]
async fn a_prompt_naming_tmux_or_nohup_is_not_judged_by_its_words() {
    // The delegated agent's prompt is prose: only commands the daemon
    // itself runs are scanned (terminal), so the run starts for a guest
    // and asks nobody.
    use crate::tools::traits::{ApprovalCall, OwnerApprovalNeed};
    let (tool, _db) = setup_echo_tool().await;
    let raw =
        r#"{"action":"run","tool":"echo","prompt":"Explain how screen, tmux and nohup & differ"}"#;
    let facts = crate::tools::traits::ToolCallContext {
        session_id: "sess1".into(),
        user_role: Some(aidaemon_tools::semantics::UserRole::Guest),
        ..Default::default()
    };
    let result = tool
        .call_with_status_under_contract(raw, None, false, false, &facts)
        .await
        .unwrap();
    assert!(!result.contains("Blocked"), "{result}");
    let args: Value = serde_json::from_str(raw).unwrap();
    assert!(!matches!(
        tool.owner_approval_request(&ApprovalCall::as_sent(&args), &facts)
            .await,
        OwnerApprovalNeed::Ask(_)
    ));
}

#[tokio::test]
async fn test_run_captures_exit_code_failure() {
    let (tool, _db) = setup_bash_tool().await;
    let result = tool
        .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo 'failing' >&2; exit 1"}"#)
        .await
        .unwrap();
    assert!(
        result.contains("ERROR"),
        "Expected ERROR in output for exit code 1, got: {}",
        result
    );
    assert!(
        result.contains("failing"),
        "Expected stderr in error output, got: {}",
        result
    );
}

#[tokio::test]
async fn test_run_unavailable_requested_tool_uses_configured_fallback() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool
        .call(r#"{"action":"run","tool":"nonexistent","prompt":"test"}"#)
        .await
        .unwrap();
    assert!(result.contains("test"), "got: {result}");
}

/// After the picked agent is quarantined, an enforced read/write run
/// does not fall back to an agent without a native split adapter (it
/// would be refused on start); a read-only run may use any agent.
#[tokio::test]
async fn fallback_after_unavailable_agent_keeps_the_adapter_rule() {
    let (tool, _db) = setup_echo_tool().await;
    assert_eq!(tool.fallback_tool_name(true, false), None);
    assert_eq!(tool.fallback_tool_name(true, true).as_deref(), Some("echo"));
    assert_eq!(
        tool.fallback_tool_name(false, false).as_deref(),
        Some("echo")
    );
}

#[tokio::test]
async fn test_run_missing_tool_param() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool
        .call(r#"{"action":"run","prompt":"test"}"#)
        .await
        .unwrap();
    assert!(
        result.contains("test"),
        "Expected default tool fallback output, got: {}",
        result
    );
}

#[tokio::test]
async fn test_run_missing_prompt_param() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool.call(r#"{"action":"run","tool":"echo"}"#).await;
    assert!(result.is_err());
}

// -----------------------------------------------------------------------
// Stdin hang prevention test (the critical fix)
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_process_completes_without_hanging_on_stdin() {
    // This is THE critical test: before the fix, piped stdin would cause
    // CLI agents to hang waiting for EOF. With stdin set to null, the
    // process should complete quickly.
    let (tool, _db) = setup_bash_tool().await;

    let start = Instant::now();
    let result = tool
        .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo 'done'; exit 0"}"#)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        result.contains("done"),
        "Expected 'done' in output, got: {}",
        result
    );
    // Should complete in well under 5 seconds (the hang was 5+ minutes)
    assert!(
        elapsed < Duration::from_secs(5),
        "Process took {:?} — likely hanging on stdin",
        elapsed
    );
}

#[tokio::test]
async fn test_cat_stdin_completes_quickly() {
    // `cat` without args reads from stdin — with Stdio::null() it should
    // get immediate EOF and exit. With Stdio::piped() it would hang forever.
    let (tool, _db) = setup_bash_tool().await;

    let start = Instant::now();
    let result = tool
        .call(r#"{"action":"run","tool":"bash-agent","prompt":"cat; echo 'cat done'"}"#)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        result.contains("cat done"),
        "Expected 'cat done' in output, got: {}",
        result
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "`cat` took {:?} — stdin not null?",
        elapsed
    );
}

// -----------------------------------------------------------------------
// Concurrent limit test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_cancel_nonexistent_task() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool
        .call(r#"{"action":"cancel","task_id":"nonexistent"}"#)
        .await
        .unwrap();
    assert!(
        result.contains("No running CLI agent"),
        "Expected not found message, got: {}",
        result
    );
}

// -----------------------------------------------------------------------
// List test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_list_empty() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool.call(r#"{"action":"list"}"#).await.unwrap();
    assert!(
        result.contains("No CLI agents currently running"),
        "Expected empty list message, got: {}",
        result
    );
}

#[tokio::test]
async fn test_run_accepts_mission_and_task_aliases() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool
            .call(
                r#"{"action":"run","tool":"echo","mission":"Email helper","task":"Open Gmail and summarize the inbox"}"#,
            )
            .await
            .unwrap();
    assert!(
        result.contains("Mission: Email helper"),
        "Expected synthesized mission in output, got: {}",
        result
    );
    assert!(
        result.contains("Task: Open Gmail and summarize the inbox"),
        "Expected synthesized task in output, got: {}",
        result
    );
}

#[tokio::test]
async fn test_run_recovers_prompt_from_task_and_goal_context() {
    let (tool, _db) = setup_echo_tool().await;
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let working_dir = tmp_dir.path().to_string_lossy().to_string();
    let goal = Goal::new_finite(
        "Review the latest aidaemon service logs and make the smallest safe fix",
        "sess-ctx",
    );
    tool.state.create_goal(&goal).await.unwrap();

    let task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: format!(
            "Inspect the recent log failures and patch the root cause in {}",
            working_dir
        ),
        status: "claimed".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    tool.state.create_task(&task).await.unwrap();

    let result = tool
        .call_with_status_under_contract(
            &format!(
                r#"{{"action":"run","tool":"echo","working_dir":"{}"}}"#,
                working_dir
            ),
            None,
            false,
            false,
            &ToolCallContext {
                goal_id: Some(goal.id.clone()),
                task_id: Some(task.id.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    assert!(
        result.contains(
            "Mission: Review the latest aidaemon service logs and make the smallest safe fix"
        ),
        "Expected synthesized mission from goal context, got: {}",
        result
    );
    assert!(
        result.contains(&format!(
            "Task: Inspect the recent log failures and patch the root cause in {}",
            working_dir
        )),
        "Expected synthesized task from task context, got: {}",
        result
    );

    let updated_task = tool.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated_task.status, "completed");
    let context: serde_json::Value =
        serde_json::from_str(updated_task.context.as_deref().unwrap()).unwrap();
    assert_eq!(
        context["executor_result"]["task_outcome"].as_str(),
        Some("task_done")
    );
    let stored_scope = context["executor_handoff"]["target_scope"]["allowed_targets"][0]["value"]
        .as_str()
        .unwrap();
    assert!(
        stored_scope.ends_with(tmp_dir.path().to_string_lossy().as_ref()),
        "expected stored scope '{}' to end with '{}'",
        stored_scope,
        tmp_dir.path().to_string_lossy()
    );
}

#[tokio::test]
async fn test_run_accepts_command_and_description_aliases() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool
        .call(r#"{"action":"run","tool":"echo","description":"Check test output","command":"pwd"}"#)
        .await
        .unwrap();
    assert!(
        result.contains("Check test output"),
        "Expected synthesized description in output, got: {}",
        result
    );
    assert!(
        result.contains(
            "Run this exact shell command in the working directory and report the result:\npwd"
        ),
        "Expected synthesized command guidance in output, got: {}",
        result
    );
}

// -----------------------------------------------------------------------
// Schema test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_schema_includes_registered_tools() {
    let (tool, _db) = setup_echo_tool().await;
    let schema = tool.schema();

    // Must have name and description (critical gotcha from CLAUDE.md)
    assert_eq!(schema["name"], "cli_agent");
    assert!(schema["description"].as_str().unwrap().contains("echo"));
    assert!(
        schema["parameters"]["properties"]["tool"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("echo"))
    );
}

#[tokio::test]
async fn test_schema_updates_after_dynamic_add() {
    let (tool, _db) = setup_echo_tool().await;

    // Add a new agent directly to the map
    {
        let mut tools = tool.tools.write().unwrap();
        tools.insert(
            "new-tool".to_string(),
            CliToolEntry {
                command: "echo".to_string(),
                args: vec![],
                description: "Newly added".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10000,
                is_dynamic: true,
            },
        );
        let mut names = tool.tool_names.write().unwrap();
        names.push("new-tool".to_string());
        names.sort();
    }

    let schema = tool.schema();
    let enum_vals = schema["parameters"]["properties"]["tool"]["enum"]
        .as_array()
        .unwrap();
    assert!(
        enum_vals.contains(&json!("echo")) && enum_vals.contains(&json!("new-tool")),
        "Schema should include both tools: {:?}",
        enum_vals
    );
}

// -----------------------------------------------------------------------
// Dynamic agent management tests
// -----------------------------------------------------------------------

#[test]
fn the_cli_agent_description_states_what_its_sandbox_cannot_do() {
    let described = CLI_SANDBOX.describe();
    assert!(described.contains("reach the network"), "{described}");
    assert!(described.contains("spawn_agent"), "{described}");
}

// -----------------------------------------------------------------------
// Git diff capture test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_unknown_action() {
    let (tool, _db) = setup_echo_tool().await;
    let result = tool.call(r#"{"action":"invalid_action"}"#).await.unwrap();
    assert!(result.contains("Unknown action"));
}

// -----------------------------------------------------------------------
// has_tools test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_user_prompt_website_about_cars() {
    // This replicates the exact prompt that caused a 5-minute hang.
    // The fix: stdin is now Stdio::null() so processes get EOF immediately.
    let (tool, _db) = setup_bash_tool().await;

    let user_prompt = "I need to create a new website about cars. We should push it to cars.example.com. make it modern.";

    let start = Instant::now();
    let args = json!({
        "action": "run",
        "tool": "bash-agent",
        "prompt": format!("echo 'Received prompt: {}'; echo 'Task complete'", user_prompt),
    });
    let result = tool
        .call_with_status_under_contract(
            &args.to_string(),
            None,
            false,
            false,
            &aidaemon_tools::testing::call_facts("telegram_12345"),
        )
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "Took {:?} — should complete quickly, not hang",
        elapsed
    );
    assert!(
        result.contains("Task complete"),
        "Expected output, got: {}",
        result
    );
}

#[tokio::test]
async fn test_user_prompt_with_system_instruction() {
    // Tests the orchestrator flow: LLM sends system_instruction to shape
    // the CLI agent into a specialist, which triggers build_enriched_prompt.
    let (tool, _db) = setup_bash_tool().await;

    let start = Instant::now();
    let args = json!({
        "action": "run",
        "tool": "bash-agent",
        "prompt": "echo 'Building website...'; echo 'Created index.html'; echo 'Done'",
        "system_instruction": "You are a senior web developer. Create a modern, responsive website.",
    });
    let result = tool
        .call_with_status_under_contract(
            &args.to_string(),
            None,
            false,
            false,
            &aidaemon_tools::testing::call_facts("telegram_12345"),
        )
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "Enriched prompt flow took {:?}",
        elapsed
    );
    assert!(result.contains("Done"), "Expected output, got: {}", result);
}

// -----------------------------------------------------------------------
// Claude Code stream-json output parsing
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_stderr_captured_in_error_output() {
    let (tool, _db) = setup_bash_tool().await;

    let args = json!({
        "action": "run",
        "tool": "bash-agent",
        "prompt": "echo 'some stdout'; echo 'error detail 1' >&2; echo 'error detail 2' >&2; exit 1"
    });
    let result = tool.call(&args.to_string()).await.unwrap();

    assert!(result.contains("ERROR"));
    assert!(
        result.contains("error detail 1"),
        "Should capture stderr, got: {}",
        result
    );
}

// -----------------------------------------------------------------------
// Working dir with real CLI agent simulation
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_run_with_working_dir() {
    let (tool, _db) = setup_bash_tool().await;
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir_path = tmp_dir.path().to_str().unwrap();

    let args = json!({
        "action": "run",
        "tool": "bash-agent",
        "prompt": "pwd",
        "working_dir": dir_path
    });
    let result = tool.call(&args.to_string()).await.unwrap();

    assert!(
        result.contains(dir_path),
        "CLI agent should run in specified working dir, got: {}",
        result
    );
}

// -----------------------------------------------------------------------
// Prompt similarity tests
// -----------------------------------------------------------------------
