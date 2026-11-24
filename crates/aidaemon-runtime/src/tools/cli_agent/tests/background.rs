//! Background runs, the invocation log, checks and run cards.

use super::*;

#[test]
fn background_cli_completion_reengages_the_original_outcome() {
    let followup = CliAgentTool::build_background_reengagement_followup(
        "codex",
        "abc12345",
        "Investigate the failure",
        "completed",
        "The root cause is in scheduler.rs",
    );

    assert!(followup.starts_with("[Background command completed]"));
    assert!(followup.contains("Runtime task ID: abc12345"));
    assert!(followup.contains("previous unfinished request"));
    assert!(followup.contains("complete every remaining implementation"));
    assert!(followup.contains("The root cause is in scheduler.rs"));
}

#[tokio::test]
async fn test_async_mode_returns_immediately() {
    let (tool, _db) = setup_bash_tool().await;

    let start = Instant::now();
    let result = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"sleep 2; echo async-done","async_mode":true}"#,
            )
            .await
            .unwrap();
    let elapsed = start.elapsed();

    // Should return immediately (< 1s) with a task_id
    assert!(
        elapsed < Duration::from_secs(1),
        "Async mode took {:?} — not returning immediately",
        elapsed
    );
    assert!(
        result.contains("started in background"),
        "Expected background message, got: {}",
        result
    );
    assert!(
        result.contains("task_id="),
        "Expected task_id in response, got: {}",
        result
    );
}

#[tokio::test]
async fn test_async_mode_check_shows_result() {
    let (tool, _db) = setup_bash_tool().await;

    // Start a longer async task so it's still running when we check
    let result = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"echo async-check-test; sleep 5","async_mode":true}"#,
            )
            .await
            .unwrap();

    // Extract task_id from "task_id=XXXX)"
    let task_id = result
        .split("task_id=")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();

    // Give it a moment to produce output
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check on the task — should still be running
    let check_args = format!(r#"{{"action":"check","task_id":"{}"}}"#, task_id);
    let check_result = tool.call(&check_args).await.unwrap();

    assert!(
        check_result.contains("async-check-test")
            || check_result.contains("still running")
            || check_result.contains("finished"),
        "Expected task output or status, got: {}",
        check_result
    );

    // Cancel to clean up
    let cancel_args = format!(r#"{{"action":"cancel","task_id":"{}"}}"#, task_id);
    tool.call(&cancel_args).await.unwrap();
}

// -----------------------------------------------------------------------
// Cancel test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_run_logs_invocation() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(TestStore::open(db_file.path().to_str().unwrap()).await);
    let state_clone = state.clone();

    let mut tools_map = HashMap::new();
    tools_map.insert(
        "echo".to_string(),
        CliToolEntry {
            command: "echo".to_string(),
            args: vec![],
            description: "".to_string(),
            timeout: Duration::from_secs(10),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );

    let tool = CliAgentTool {
        context_facts: None,
        project_notes: None,
        backend: aidaemon_tools::testing::active_execution_backend(),
        tools: Arc::new(std::sync::RwLock::new(tools_map)),
        tool_names: Arc::new(std::sync::RwLock::new(vec!["echo".to_string()])),
        running: Arc::new(Mutex::new(HashMap::new())),
        completed: Arc::new(Mutex::new(HashMap::new())),
        working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
        state: state as Arc<dyn CliAgentStore>,
        default_timeout: Duration::from_secs(10),
        default_max_output: 10000,
        max_concurrent: 3,
        concurrency_limiter: Arc::new(Semaphore::new(3)),
        hub: OnceLock::new(),
        runs: OnceLock::new(),
        plan_store: OnceLock::new(),
        agent: OnceLock::new(),
        reengagements: Arc::new(Mutex::new(HashMap::new())),
        coding_agents: aidaemon_coding_agents::CodingAgents::builtin(),
        permissions: Arc::new(OnceLock::new()),
    };

    // Run a command
    tool.call_with_status_under_contract(
        r#"{"action":"run","tool":"echo","prompt":"log test"}"#,
        None,
        false,
        false,
        &aidaemon_tools::testing::call_facts("sess1"),
    )
    .await
    .unwrap();

    // Check invocations were logged
    let invocations = state_clone.get_cli_agent_invocations(10).await.unwrap();
    assert!(
        !invocations.is_empty(),
        "Expected at least one invocation logged"
    );
    assert_eq!(invocations[0].agent_name, "echo");
    assert!(invocations[0].prompt_summary.contains("log test"));
    assert_eq!(invocations[0].success, Some(true));
    assert!(invocations[0].duration_secs.is_some());
}

#[tokio::test]
async fn test_timeout_run_still_logs_completion() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(TestStore::open(db_file.path().to_str().unwrap()).await);
    let state_clone = state.clone();

    let mut tools_map = HashMap::new();
    tools_map.insert(
        "bash".to_string(),
        CliToolEntry {
            command: "bash".to_string(),
            args: vec!["-c".to_string()],
            description: "".to_string(),
            timeout: Duration::from_millis(50),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );

    let tool = CliAgentTool {
        context_facts: None,
        project_notes: None,
        backend: aidaemon_tools::testing::active_execution_backend(),
        tools: Arc::new(std::sync::RwLock::new(tools_map)),
        tool_names: Arc::new(std::sync::RwLock::new(vec!["bash".to_string()])),
        running: Arc::new(Mutex::new(HashMap::new())),
        completed: Arc::new(Mutex::new(HashMap::new())),
        working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
        state: state as Arc<dyn CliAgentStore>,
        default_timeout: Duration::from_secs(10),
        default_max_output: 10000,
        max_concurrent: 3,
        concurrency_limiter: Arc::new(Semaphore::new(3)),
        hub: OnceLock::new(),
        runs: OnceLock::new(),
        plan_store: OnceLock::new(),
        agent: OnceLock::new(),
        reengagements: Arc::new(Mutex::new(HashMap::new())),
        coding_agents: aidaemon_coding_agents::CodingAgents::builtin(),
        permissions: Arc::new(OnceLock::new()),
    };

    // Run a command that will exceed the short timeout and be moved to background,
    // but should still eventually be logged as completed in the DB.
    let resp = tool
        .call_with_status_under_contract(
            r#"{"action":"run","tool":"bash","prompt":"sleep 0.2; echo done"}"#,
            None,
            false,
            false,
            &aidaemon_tools::testing::call_facts("sess1"),
        )
        .await
        .unwrap();
    assert!(
        resp.contains("Moved to background") || resp.contains("still running"),
        "expected background/timeout response, got: {}",
        resp
    );

    // Allow the child to finish and the completion logger to flush to SQLite.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let invocations = state_clone.get_cli_agent_invocations(10).await.unwrap();
    assert!(!invocations.is_empty(), "Expected invocation logged");
    assert_eq!(invocations[0].agent_name, "bash");
    assert_eq!(
        invocations[0].success,
        Some(true),
        "Expected background invocation to be marked successful"
    );
    assert!(
        invocations[0].completed_at.is_some(),
        "Expected completed_at to be set"
    );
    assert!(
        invocations[0].duration_secs.unwrap_or(0.0) > 0.0,
        "Expected a positive duration"
    );
}

#[tokio::test]
async fn test_background_delegated_run_persists_structured_task_result() {
    let (tool, _db) = setup_bash_tool().await;
    {
        let mut tools = tool.tools.write().unwrap();
        tools.get_mut("bash-agent").unwrap().timeout = Duration::from_millis(50);
    }

    let goal = Goal::new_finite("Patch the current repo safely", "sess-bg");
    tool.state.create_goal(&goal).await.unwrap();

    let task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Run a delegated CLI task in the background".to_string(),
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

    let resp = tool
            .call_with_status_under_contract(r#"{"action":"run","tool":"bash-agent","prompt":"sleep 0.2; echo delegated-background-ok"}"#, None, false, false, &ToolCallContext {
                    goal_id: Some(goal.id.clone()),
                    task_id: Some(task.id.clone()),
                    ..aidaemon_tools::testing::call_facts("sess-bg")
                })
            .await
            .unwrap();
    assert!(
        resp.contains("Moved to background") || resp.contains("still running"),
        "expected timeout/background handoff, got: {}",
        resp
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    let updated_task = loop {
        let current = tool.state.get_task(&task.id).await.unwrap().unwrap();
        if current.status != "running" && current.status != "claimed" {
            break current;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "delegated task did not persist a terminal outcome before the deadline"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert_eq!(updated_task.status, "completed");
    let context: serde_json::Value =
        serde_json::from_str(updated_task.context.as_deref().unwrap()).unwrap();
    assert_eq!(
        context["executor_result"]["task_outcome"].as_str(),
        Some("task_done")
    );
    assert!(
        updated_task
            .result
            .as_deref()
            .unwrap_or("")
            .contains("delegated-background-ok"),
        "expected delegated background result to be persisted, got: {:?}",
        updated_task.result
    );
}

#[tokio::test]
async fn test_background_cli_check_returns_result_after_reap() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state = Arc::new(TestStore::open(db_file.path().to_str().unwrap()).await);

    let mut tools_map = HashMap::new();
    tools_map.insert(
        "bash".to_string(),
        CliToolEntry {
            command: "bash".to_string(),
            args: vec!["-c".to_string()],
            description: "".to_string(),
            timeout: Duration::from_millis(50),
            max_output_chars: 10000,
            is_dynamic: false,
        },
    );

    let tool = CliAgentTool {
        context_facts: None,
        project_notes: None,
        backend: aidaemon_tools::testing::active_execution_backend(),
        tools: Arc::new(std::sync::RwLock::new(tools_map)),
        tool_names: Arc::new(std::sync::RwLock::new(vec!["bash".to_string()])),
        running: Arc::new(Mutex::new(HashMap::new())),
        completed: Arc::new(Mutex::new(HashMap::new())),
        working_dir_claims: Arc::new(std::sync::Mutex::new(HashMap::new())),
        state: state as Arc<dyn CliAgentStore>,
        default_timeout: Duration::from_secs(10),
        default_max_output: 10000,
        max_concurrent: 3,
        concurrency_limiter: Arc::new(Semaphore::new(3)),
        hub: OnceLock::new(),
        runs: OnceLock::new(),
        plan_store: OnceLock::new(),
        agent: OnceLock::new(),
        reengagements: Arc::new(Mutex::new(HashMap::new())),
        coding_agents: aidaemon_coding_agents::CodingAgents::builtin(),
        permissions: Arc::new(OnceLock::new()),
    };

    let resp = tool
        .call_with_status_under_contract(
            r#"{"action":"run","tool":"bash","prompt":"sleep 0.2; echo cli-reap-ok"}"#,
            None,
            false,
            false,
            &aidaemon_tools::testing::call_facts("sess1"),
        )
        .await
        .unwrap();
    let task_id = extract_task_id_from_background_message(&resp);

    // Let it finish so next call reaps before check.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let check = tool
        .call_with_status_under_contract(
            &format!(r#"{{"action":"check","task_id":"{}"}}"#, task_id),
            None,
            false,
            false,
            &aidaemon_tools::testing::call_facts("sess1"),
        )
        .await
        .unwrap();
    assert!(check.contains("cli-reap-ok"));
    assert!(check.contains("finished"));
}

#[tokio::test]
async fn check_with_wrong_id_recovers_only_session_task() {
    let (tool, _db) = setup_bash_tool().await;
    let started = tool
            .call_with_status_under_contract(r#"{"action":"run","tool":"bash-agent","prompt":"echo recovered-result","async_mode":true}"#, None, false, false, &aidaemon_tools::testing::call_facts("session-recover"))
            .await
            .unwrap();
    let real_task_id = extract_task_id_from_background_message(&started);
    tokio::time::sleep(Duration::from_millis(300)).await;

    let recovered = tool
        .call_with_status_under_contract(
            r#"{"action":"check","task_id":"provider-thread-uuid"}"#,
            None,
            false,
            false,
            &aidaemon_tools::testing::call_facts("session-recover"),
        )
        .await
        .unwrap();

    assert!(recovered.contains("Recovered the only recent CLI task"));
    assert!(recovered.contains(&format!("task_id={real_task_id}")));
    assert!(recovered.contains("recovered-result"));
}

/// The outcome comes from what the call did: a finished agent whose own
/// output says "still running" or "moved to background" is not a
/// background handoff, and an async start is one.
#[tokio::test]
async fn background_outcome_is_typed_not_read_from_the_output() {
    let (tool, _db) = setup_bash_tool().await;
    let finished = tool
            .outcome_with_facts(
                r#"{"action":"run","tool":"bash-agent","prompt":"echo 'the server is still running. Moved to background earlier.'"}"#,
                None,
                false,
                &aidaemon_tools::testing::call_facts("session-typed-finish"),
            )
            .await
            .unwrap();
    assert!(
        finished.output.contains("still running"),
        "{}",
        finished.output
    );
    assert_ne!(
        finished.metadata.outcome_status,
        Some(ToolOutcomeStatus::Backgrounded),
        "{}",
        finished.output
    );
    assert!(!finished.metadata.background_started);

    let started = tool
            .outcome_with_facts(
                r#"{"action":"run","tool":"bash-agent","prompt":"sleep 0.2; echo later","async_mode":true}"#,
                None,
                false,
                &aidaemon_tools::testing::call_facts("session-typed-async"),
            )
            .await
            .unwrap();
    assert_eq!(
        started.metadata.outcome_status,
        Some(ToolOutcomeStatus::Backgrounded)
    );
    assert!(started.metadata.background_started);
}

/// A run that changed its write roots is an untyped change: the items
/// it names are settled by it, and without a claim the runtime may give
/// it to the one item in progress.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell agent; the typing is platform-neutral")]
async fn a_run_that_changed_its_folder_settles_the_items_it_names() {
    let (tool, _db) = setup_bash_tool().await;
    for (claim, session) in [
        (true, "session-cli-claim"),
        (false, "session-cli-unclaimed"),
    ] {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().to_string_lossy().to_string();
        let mut args = json!({
            "action": "run",
            "tool": "bash-agent",
            "prompt": format!("echo site > {root}/index.html"),
            "working_dir": root,
            "workspace_mode": "read_write",
            "write_paths": [root],
        });
        if claim {
            args["checklist_items"] = json!([1]);
        }
        let outcome = tool
            .outcome_with_facts(
                &args.to_string(),
                None,
                false,
                &aidaemon_tools::testing::call_facts(session),
            )
            .await
            .unwrap();
        assert!(dir.path().join("index.html").exists(), "{}", outcome.output);
        let semantics = &outcome.metadata.semantics;
        assert!(semantics.mutates_state(), "{}", outcome.output);
        if claim {
            assert_eq!(outcome.metadata.checklist_items, vec![1]);
            assert!(
                semantics
                    .mutation_effects
                    .intersects(aidaemon_tools::semantics::ToolMutationEffects::UNSPECIFIED)
            );
        } else {
            assert!(outcome.metadata.checklist_items.is_empty());
            assert!(outcome.metadata.changes_untyped);
        }
    }
}

fn started_task_id(output: &str) -> String {
    output
        .split("task_id=")
        .nth(1)
        .unwrap()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .next()
        .unwrap()
        .to_string()
}

/// A check waits for a task about to finish and returns its result, typed
/// as settling the launch; a check of a task that outlasts the wait stays
/// a nonterminal background observation that settles nothing.
#[tokio::test]
async fn check_waits_for_the_task_and_types_its_end() {
    let (tool, _db) = setup_bash_tool().await;
    let facts = aidaemon_tools::testing::call_facts("session-check-wait");
    let started = tool
            .outcome_with_facts(
                r#"{"action":"run","tool":"bash-agent","prompt":"sleep 0.1; echo waited-result","async_mode":true}"#,
                None,
                false,
                &facts,
            )
            .await
            .unwrap();
    assert!(started.metadata.background_started);
    assert!(!started.metadata.background_settled);
    let task_id = started_task_id(&started.output);

    let check = tool
        .outcome_with_facts(
            &format!(r#"{{"action":"check","task_id":"{task_id}"}}"#),
            None,
            false,
            &facts,
        )
        .await
        .unwrap();
    assert!(check.output.contains("waited-result"), "{}", check.output);
    assert!(check.metadata.background_settled, "{}", check.output);
    assert!(!check.metadata.background_started);

    let slow = tool
            .outcome_with_facts(
                r#"{"action":"run","tool":"bash-agent","prompt":"sleep 5; echo never","async_mode":true}"#,
                None,
                false,
                &facts,
            )
            .await
            .unwrap();
    let slow_id = started_task_id(&slow.output);
    let waited = Instant::now();
    let running = tool
        .outcome_with_facts(
            &format!(r#"{{"action":"check","task_id":"{slow_id}"}}"#),
            None,
            false,
            &facts,
        )
        .await
        .unwrap();
    assert!(
        waited.elapsed() >= CHECK_WAIT,
        "check returned before its wait"
    );
    assert!(
        running.output.contains("still running"),
        "{}",
        running.output
    );
    assert!(!running.metadata.background_settled);
    assert!(!running.metadata.completion_notifications_enabled);

    let cancel = tool
        .outcome_with_facts(
            &format!(r#"{{"action":"cancel","task_id":"{slow_id}"}}"#),
            None,
            false,
            &facts,
        )
        .await
        .unwrap();
    assert!(cancel.metadata.background_settled, "{}", cancel.output);
}

#[tokio::test]
async fn missing_check_is_a_typed_negative_result() {
    let (tool, _db) = setup_echo_tool().await;
    let outcome = tool
        .outcome_with_facts(
            r#"{"action":"check","task_id":"missing"}"#,
            None,
            false,
            &aidaemon_tools::testing::call_facts("empty-session"),
        )
        .await
        .unwrap();

    assert_eq!(
        outcome.metadata.outcome_status,
        Some(ToolOutcomeStatus::CompletedWithNegativeResult)
    );
    assert!(
        outcome
            .output
            .contains("do not treat this lookup miss as completion")
    );
}

#[tokio::test]
async fn async_run_reports_background_metadata_and_notifications() {
    let (tool, _db) = setup_bash_tool().await;
    let outcome = tool
        .outcome_with_facts(
            r#"{"action":"run","tool":"bash-agent","prompt":"sleep 2","async_mode":true}"#,
            None,
            false,
            &aidaemon_tools::testing::call_facts("session-background"),
        )
        .await
        .unwrap();

    assert_eq!(
        outcome.metadata.outcome_status,
        Some(ToolOutcomeStatus::Backgrounded)
    );
    assert!(outcome.metadata.background_started);
    assert!(outcome.metadata.completion_notifications_enabled);

    tool.handle_cancel_all("session-background").await.unwrap();
}

/// A background CLI run is listed for its chat's running work, under
/// the turn that started it, and stopping it reads the process back as
/// ended.
#[tokio::test]
async fn a_background_cli_run_is_listed_for_its_chat_and_stops() {
    use crate::tools::traits::ToolSemantics as _;
    let (tool, _db) = setup_bash_tool().await;
    let facts = ToolCallContext {
        task_id: Some("turn-cli".to_string()),
        ..aidaemon_tools::testing::call_facts("session-cli-jobs")
    };
    tool.outcome_with_facts(
        r#"{"action":"run","tool":"bash-agent","prompt":"sleep 30","async_mode":true}"#,
        None,
        false,
        &facts,
    )
    .await
    .unwrap();
    let jobs = tool.background_jobs("session-cli-jobs").await;
    assert_eq!(jobs.len(), 1, "{jobs:?}");
    // Named by what it is, never by its prompt.
    assert_eq!(jobs[0].title, "bash-agent coding agent", "{jobs:?}");
    assert_eq!(jobs[0].task_id.as_deref(), Some("turn-cli"));
    assert!(tool.background_jobs("another-session").await.is_empty());
    assert!(tool.stop_background_job(&jobs[0].key).await);
    assert!(tool.background_jobs("session-cli-jobs").await.is_empty());
}

/// A continuation runtime that counts how often a run's end wakes it.
#[derive(Default)]
struct CountingRuntime(std::sync::atomic::AtomicUsize);

#[async_trait::async_trait]
impl ConversationRuntime for CountingRuntime {
    async fn continue_conversation(
        &self,
        _request: ConversationRequest,
    ) -> anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        anyhow::bail!("not answered in this test")
    }

    async fn record_continuation_delivery(
        &self,
        _session_id: &str,
        _delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Live 2026-11-23: the owner pressed Stop on a background Codex run in
/// the app; its card said "Stopped by you", then its chat woke up and
/// started the work again. A run the owner stopped ends on its card and
/// wakes nothing.
#[tokio::test]
async fn a_run_the_owner_stops_does_not_wake_its_chat() {
    let (tool, _db) = setup_bash_tool().await;
    let runs = Arc::new(crate::run_state::RunStates::default());
    tool.set_runs(runs.clone());
    let counting = Arc::new(CountingRuntime::default());
    let runtime: Arc<dyn ConversationRuntime> = counting.clone();
    tool.set_agent(Arc::downgrade(&runtime));
    let facts = aidaemon_tools::testing::call_facts("app:synthetic-stop");
    tool.outcome_with_facts(
        r#"{"action":"run","tool":"bash-agent","prompt":"sleep 30","async_mode":true}"#,
        None,
        false,
        &facts,
    )
    .await
    .unwrap();
    let jobs = tool.background_jobs("app:synthetic-stop").await;
    assert_eq!(jobs.len(), 1, "{jobs:?}");
    let run_id = runs
        .for_chat("app:synthetic-stop")
        .into_iter()
        .next()
        .expect("its run")
        .id;
    // The app's Stop: the run is marked as the owner's stop, then ended.
    runs.owner_stopping(&run_id);
    assert!(tool.stop_background_job(&jobs[0].key).await);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::time::Instant::now() < deadline
        && !runs
            .for_chat("app:synthetic-stop")
            .iter()
            .any(crate::run_state::RunState::stopped_by_owner)
    {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Long enough for a continuation to have started.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(counting.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}

/// A CLI agent left running in the background is a run on its turn's
/// one card: the reply names it in plain words (never the prompt), its
/// latest output line is its step (never "(N min)"), it settles to its
/// final header, and its result follows without the header.
#[tokio::test]
async fn a_background_agent_shows_one_card_that_settles() {
    let (tool, _db) = setup_bash_tool().await;
    let chat = aidaemon_channels::testing::CardChat::new("telegram:synthetic-agent").await;
    tool.set_hub(chat.outbound());
    let facts = aidaemon_tools::testing::call_facts("telegram:synthetic-agent");
    let output = chat
            .turn(
                |tx| {
                    tool.call_with_status_under_contract(
                        r#"{"action":"run","tool":"bash-agent","prompt":"echo working-on-it; sleep 4; echo agent-done","description":"tidy the notes","async_mode":true}"#,
                        Some(tx),
                        false,
                        false,
                        &facts,
                    )
                },
                |output| {
                    output
                        .as_ref()
                        .map(|output| output.lines().next().unwrap_or("").to_string())
                        .unwrap_or_default()
                },
            )
            .await
            .unwrap();
    assert!(
        output.starts_with("Started in the background: tidy the notes."),
        "{output}"
    );
    assert!(
        chat.wait_for(Duration::from_secs(10), |text| text
            .starts_with("⏳ Running in the background: tidy the notes")
            && text.contains("of output so far"))
            .await,
        "{:#?}",
        chat.shown().await
    );
    assert!(
        chat.wait_for(Duration::from_secs(20), |text| text.contains("agent-done")
            && !text.contains("in the background:"))
            .await,
        "{:#?}",
        chat.shown().await
    );
    chat.assert_one_settled_card(
        "✅ Finished in the background: tidy the notes",
        "agent-done",
    )
    .await;
    for text in chat.shown().await {
        assert!(
            !text.contains("sleep 4"),
            "the prompt is never echoed: {text}"
        );
    }
}

/// A continuation that never gets its slot (another one holds it, as
/// live 2026-11-13).
struct PendingRuntime;

#[async_trait::async_trait]
impl ConversationRuntime for PendingRuntime {
    async fn continue_conversation(
        &self,
        _request: ConversationRequest,
    ) -> anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope> {
        std::future::pending().await
    }

    async fn record_continuation_delivery(
        &self,
        _session_id: &str,
        _delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// A launched agent's output is read into typed events: its session id
/// is kept, each event is kept in order with each permission decision,
/// and its latest step is its run's activity, in words that never carry
/// a command line.
#[tokio::test]
async fn a_coding_agents_events_are_kept_and_its_step_is_its_runs_activity() {
    let db = tempfile::NamedTempFile::new().unwrap();
    let store = Arc::new(TestStore::open(db.path().to_str().unwrap()).await);
    let runs = Arc::new(crate::run_state::RunStates::default());
    runs.begin("cli-agent-r1", "app:synthetic-events", "Claude Code", None);
    let agent = aidaemon_coding_agents::CodingAgents::builtin().for_program("claude");
    let mut recorder = crate::tools::cli_agent_events::AgentEventRecorder::new(
        agent,
        store.clone() as Arc<dyn CliAgentStore>,
        7,
        runs.clone(),
        "cli-agent-r1".to_string(),
    );
    recorder
        .read(r#"{"type":"system","subtype":"init","session_id":"sess-synthetic","model":"m"}"#)
        .await;
    recorder
            .read(r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"SECRET=abc cargo test -q"}}]}}"#)
            .await;
    assert_eq!(
        runs.get("cli-agent-r1").unwrap().activity.as_deref(),
        Some("Running cargo")
    );
    let judge = recorder.judging(Arc::new(aidaemon_coding_agents::AllowAll));
    judge
        .judge(aidaemon_coding_agents::AgentAction::ReachHost {
            host: "github.com".into(),
        })
        .await;
    recorder
            .read(r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","is_error":false}]}}"#)
            .await;
    recorder
            .read(r#"{"type":"result","is_error":false,"total_cost_usd":0.1,"usage":{"input_tokens":1,"output_tokens":2}}"#)
            .await;
    assert_eq!(
        *store.sessions.lock().unwrap(),
        vec![(7, "sess-synthetic".to_string())]
    );
    let events = store.events.lock().unwrap().clone();
    let kinds: Vec<(i64, &str)> = events
        .iter()
        .map(|(_, seq, kind, _)| (*seq, kind.as_str()))
        .collect();
    assert_eq!(
        kinds,
        vec![
            (1, "session"),
            (2, "step"),
            (3, "permission"),
            (4, "step_ended"),
            (5, "usage"),
            (6, "reported"),
            (7, "turn_ended"),
        ]
    );
    assert!(events.iter().all(|(invocation, ..)| *invocation == 7));
    assert!(!events[1].3.contains("SECRET"), "{}", events[1].3);
    assert!(
        events[2].3.contains("\"outcome\":\"allowed\""),
        "{}",
        events[2].3
    );
}

/// A run its turn waits on has a record like a background one: it is
/// listed in its chat while it runs and ends there with how it ended.
#[tokio::test]
async fn a_waited_on_cli_run_has_a_record_in_its_chat() {
    let (tool, _db) = setup_bash_tool().await;
    let runs = Arc::new(crate::run_state::RunStates::default());
    tool.set_runs(runs.clone());
    let facts = aidaemon_tools::testing::call_facts("app:synthetic-waited");
    let out = tool
        .call_with_status_under_contract(
            r#"{"action":"run","tool":"bash-agent","prompt":"echo reviewed"}"#,
            None,
            false,
            false,
            &facts,
        )
        .await
        .unwrap();
    assert!(out.contains("reviewed"), "{out}");
    let mut chat = Vec::new();
    for _ in 0..100 {
        chat = runs.for_chat("app:synthetic-waited");
        if chat.iter().any(|run| run.phase.is_terminal()) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert_eq!(chat.len(), 1, "{chat:?}");
    assert!(!chat[0].background);
    assert_eq!(chat[0].phase, crate::run_state::RunPhase::Succeeded);
}

/// Live 2026-11-13: two codex runs, one refused `ddev` by its sandbox,
/// both read "Done" under "All 2 done". A run that changed nothing it
/// was sent to change ends Failed with a typed reason; the card's
/// header and total come from the runs' phases; each is named from its
/// task; and the card settles the moment the programs end, though the
/// parent's continuation turn never gets its slot.
#[tokio::test]
async fn a_refused_cli_run_ends_failed_and_the_card_agrees_at_once() {
    let (tool, _db) = setup_bash_tool().await;
    let chat = aidaemon_channels::testing::CardChat::new("telegram:synthetic-refused").await;
    tool.set_hub(chat.outbound());
    let runtime: Arc<dyn ConversationRuntime> = Arc::new(PendingRuntime);
    tool.set_agent(Arc::downgrade(&runtime));
    let projects = tempfile::TempDir::new().unwrap();
    let drupal = projects.path().join("claude-drupal-7");
    let wordpress = projects.path().join("claude-wordpress-7");
    let facts = aidaemon_tools::testing::call_facts("telegram:synthetic-refused");
    let installs = json!({
        "action": "run",
        "tool": "bash-agent",
        "title": format!("Install Drupal in {}", drupal.display()),
        "prompt": "echo '<h1>Claude</h1>' > index.html; echo installed",
        "new_folders": [drupal.display().to_string()],
        "async_mode": true
    })
    .to_string();
    let refused = json!({
        "action": "run",
        "tool": "bash-agent",
        "title": "Install WordPress",
        "prompt": "echo 'zsh:1: operation not permitted: ddev'; echo 'Installation blocked.'",
        "new_folders": [wordpress.display().to_string()],
        "async_mode": true
    })
    .to_string();
    let outputs = chat
        .turn(
            |tx| async {
                let first = tool
                    .call_with_status_under_contract(
                        &installs,
                        Some(tx.clone()),
                        false,
                        false,
                        &facts,
                    )
                    .await
                    .unwrap();
                let second = tool
                    .call_with_status_under_contract(&refused, Some(tx), false, false, &facts)
                    .await
                    .unwrap();
                (first, second)
            },
            |_| "Both are running.".to_string(),
        )
        .await;
    assert!(
        outputs
            .0
            .starts_with("Started in the background: Install Drupal in claude-drupal-7."),
        "{outputs:?}"
    );
    assert!(
        outputs
            .1
            .starts_with("Started in the background: Install WordPress."),
        "{outputs:?}"
    );
    let settled = |text: &str| {
        text.contains("✅ Finished in the background: Install Drupal in claude-drupal-7")
            && text.contains("❌ Failed in the background: Install WordPress")
            && text.contains("The last of them ended")
            && text.contains("1 failed")
    };
    assert!(
        chat.wait_for(Duration::from_secs(20), settled).await,
        "{:#?}",
        chat.shown().await
    );
    for text in chat.shown().await {
        assert!(!text.contains("All 2 done"), "{text}");
        assert!(!text.contains("coding agent"), "{text}");
    }
    // The run worked in the folder made for it; the refused run's
    // folder, still empty, was given back.
    assert!(drupal.join("index.html").exists());
    assert!(!wordpress.exists());
    drop(runtime);
}

/// A finished sync run that did not do its work is a failed call.
#[tokio::test]
async fn a_sync_cli_run_that_changed_nothing_is_a_failed_call() {
    let (tool, _db) = setup_bash_tool().await;
    let projects = tempfile::TempDir::new().unwrap();
    let site = projects.path().join("site");
    let args = json!({
        "action": "run",
        "tool": "bash-agent",
        "title": "Install the site",
        "prompt": "echo 'Installation blocked by the sandbox.'",
        "new_folders": [site.display().to_string()]
    })
    .to_string();
    let outcome = tool.call_with_status_outcome(&args, None).await.unwrap();
    assert!(
        outcome
            .output
            .starts_with("ERROR: CLI agent 'bash-agent' did not do its work."),
        "{}",
        outcome.output
    );
    assert!(
        outcome.output.contains("nothing was changed"),
        "{}",
        outcome.output
    );
    assert_eq!(
        outcome.metadata.outcome_status,
        Some(ToolOutcomeStatus::FailedPermanent)
    );
}

/// A run's answer is what its agent's adapter read from its typed final
/// event: Claude Code's report in `structured_output`, as the report's
/// text, never the JSON it also writes as `result`.
#[tokio::test]
async fn a_runs_answer_is_its_adapters_typed_report() {
    let db = tempfile::NamedTempFile::new().unwrap();
    let store = Arc::new(TestStore::open(db.path().to_str().unwrap()).await);
    let runs = Arc::new(crate::run_state::RunStates::default());
    runs.begin("cli-agent-r2", "app:synthetic-report", "Claude Code", None);
    let agent = aidaemon_coding_agents::CodingAgents::builtin().for_program("claude");
    let mut recorder = crate::tools::cli_agent_events::AgentEventRecorder::new(
        agent,
        store.clone() as Arc<dyn CliAgentStore>,
        8,
        runs.clone(),
        "cli-agent-r2".to_string(),
    );
    assert_eq!(recorder.reported_text(), None);
    let report = serde_json::json!({
        "summary": "Renamed the button to Send.",
        "changed_files": ["form.html"],
        "checks": [],
        "open_questions": [],
        "needs_owner": false,
    });
    let last = serde_json::json!({
        "type": "result",
        "result": report.to_string(),
        "structured_output": report,
    });
    recorder.read(&last.to_string()).await;
    assert_eq!(
        recorder.reported_text().as_deref(),
        Some("Renamed the button to Send.\n\nChanged files:\n- form.html")
    );
}

/// A run that may write anywhere in its folder (no write paths, as in a
/// conversation set to the agent) is watched over that folder: a file it
/// writes is seen as its change, and one that only answers is not a run
/// that fell short.
#[tokio::test]
#[cfg_attr(windows, ignore = "Unix shell agent; the typing is platform-neutral")]
async fn a_whole_folder_run_is_watched_over_its_folder() {
    let (tool, _db) = setup_bash_tool().await;
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path().to_string_lossy().to_string();
    let wrote = json!({
        "action": "run",
        "tool": "bash-agent",
        "prompt": format!("echo ok > {root}/heron.txt"),
        "working_dir": root,
        "workspace_mode": "read_write",
    });
    let outcome = tool
        .outcome_with_facts(
            &wrote.to_string(),
            None,
            false,
            &aidaemon_tools::testing::call_facts("session-cli-whole"),
        )
        .await
        .unwrap();
    assert!(dir.path().join("heron.txt").exists(), "{}", outcome.output);
    assert!(
        outcome.metadata.semantics.mutates_state(),
        "{}",
        outcome.output
    );

    let answered = json!({
        "action": "run",
        "tool": "bash-agent",
        "prompt": "echo 'Nothing to change.'",
        "working_dir": root,
        "workspace_mode": "read_write",
    });
    let outcome = tool
        .outcome_with_facts(
            &answered.to_string(),
            None,
            false,
            &aidaemon_tools::testing::call_facts("session-cli-whole-answer"),
        )
        .await
        .unwrap();
    assert!(!outcome.output.starts_with("ERROR:"), "{}", outcome.output);
}

#[tokio::test]
async fn a_check_on_a_failed_background_run_reads_as_failed_with_why() {
    let (tool, _db) = setup_bash_tool().await;
    let started = tool
        .call(
            r#"{"action":"run","tool":"bash-agent","prompt":"echo synthetic-failure-note >&2; exit 3","async_mode":true}"#,
        )
        .await
        .unwrap();
    let task_id = started_task_id(&started);
    let check = format!(r#"{{"action":"check","task_id":"{task_id}"}}"#);
    let mut result = String::new();
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        result = tool.call(&check).await.unwrap();
        if !result.contains("still running") {
            break;
        }
    }
    assert!(
        result.contains("failed after") && !result.contains("finished after"),
        "a failed run reads as failed: {result}"
    );
}
