use crate::traits::AgentRole;
use aidaemon_runtime::engine::chat_work::ChatWorkControl as _;

fn extract_tool_names(defs: &[serde_json::Value]) -> Vec<String> {
    defs.iter()
        .filter_map(|def| def.get("function"))
        .filter_map(|f| f.get("name"))
        .filter_map(|n| n.as_str())
        .map(ToString::to_string)
        .collect()
}

struct HighImpactCliAgentMock;

struct StructuredPatchTool;

#[async_trait::async_trait]
impl crate::traits::Tool for StructuredPatchTool {
    fn name(&self) -> &str {
        "synthetic_patch"
    }

    fn description(&self) -> &str {
        "Return a typed local-source-write receipt for a synthetic patch"
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "synthetic_patch",
            "description": "Apply a synthetic scoped patch",
            "parameters": {
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> =
            async move { Ok(format!("Synthetic patch applied: {arguments}")) }.await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for StructuredPatchTool {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> crate::traits::ToolCallSemantics {
        let path = serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|args| args["path"].as_str().map(str::to_string))
            .unwrap_or_default();
        crate::traits::ToolCallSemantics::mutation_with(
            crate::traits::ToolMutationEffects::LOCAL_SOURCE_WRITE,
        )
        .with_target_hint(crate::traits::ToolTargetHintKind::Path, path)
    }
}

#[async_trait::async_trait]
impl crate::traits::Tool for HighImpactCliAgentMock {
    fn name(&self) -> &str {
        "cli_agent"
    }

    fn description(&self) -> &str {
        "high-impact delegation tool for policy-filter regression tests"
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "cli_agent",
            "description": "high-impact delegation tool for policy-filter regression tests",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": { "type": "string" }
                },
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        _arguments: &str,
        _facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let output: anyhow::Result<String> =
            async move { Ok("cli agent executed".to_string()) }.await;
        output.map(crate::traits::ToolCallOutcome::from_output)
    }
}

impl crate::traits::ToolSemantics for HighImpactCliAgentMock {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }
}

#[tokio::test]
async fn test_delegation_executor_hides_competing_tools_when_cli_agent_available() {
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("executor finished")]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("browser", "browser tool", "ok")) as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("run_command", "run command tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, extra_tools)
        .await
        .unwrap();
    let provider_log = harness.provider.clone();
    let agent = Arc::new(harness.agent);

    let response = agent
        .spawn_child(
            "executor test",
            "inspect workspace",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "executor finished");

    let calls = provider_log.call_log.lock().await;
    assert!(!calls.is_empty(), "Expected at least one LLM call");
    let names = extract_tool_names(&calls.last().unwrap().tools);

    assert!(names.contains(&"cli_agent".to_string()));
    assert!(!names.contains(&"terminal".to_string()));
    assert!(!names.contains(&"browser".to_string()));
    assert!(!names.contains(&"run_command".to_string()));
}

#[tokio::test]
async fn test_delegation_executor_keeps_competing_tools_when_cli_agent_unavailable() {
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("executor finished")]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok").with_availability(false))
            as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("browser", "browser tool", "ok")) as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("run_command", "run command tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, extra_tools)
        .await
        .unwrap();
    let provider_log = harness.provider.clone();
    let agent = Arc::new(harness.agent);

    let response = agent
        .spawn_child(
            "executor test",
            "inspect workspace",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "executor finished");

    let calls = provider_log.call_log.lock().await;
    assert!(!calls.is_empty(), "Expected at least one LLM call");
    let names = extract_tool_names(&calls.last().unwrap().tools);

    assert!(!names.contains(&"cli_agent".to_string()));
    assert!(names.contains(&"terminal".to_string()));
    assert!(names.contains(&"browser".to_string()));
    assert!(names.contains(&"run_command".to_string()));
}

#[tokio::test]
async fn test_spawn_child_task_lead_scopes_tools_via_shared_builder() {
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("task lead finished")]);
    let extra_tools = vec![
        Arc::new(MockTool::new("cli_agent", "delegation tool", "ok"))
            as Arc<dyn crate::traits::Tool>,
        Arc::new(MockTool::new("browser", "browser tool", "ok")) as Arc<dyn crate::traits::Tool>,
    ];
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, extra_tools)
        .await
        .unwrap();
    let provider_log = harness.provider.clone();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("audit workspace", "task_lead_test_session");
    harness.state.create_goal(&goal).await.unwrap();

    let response = agent
        .spawn_child(
            "goal orchestration",
            "audit workspace",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::TaskLead),
            Some(goal.id.as_str()),
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "task lead finished");

    let calls = provider_log.call_log.lock().await;
    assert!(!calls.is_empty(), "Expected at least one LLM call");
    let names = extract_tool_names(&calls.last().unwrap().tools);

    assert!(
        names.contains(&"manage_goal_tasks".to_string()),
        "tool names: {:?}",
        names
    );
    assert!(
        names.contains(&"cli_agent".to_string()),
        "tool names: {:?}",
        names
    );
    // TaskLeads now include essential Action tools as fallback for when delegation fails.
    assert!(
        names.contains(&"terminal".to_string()),
        "TaskLead should have terminal as fallback: {:?}",
        names
    );
    assert!(
        !names.contains(&"browser".to_string()),
        "browser should NOT be in essential action tools: {:?}",
        names
    );
}

#[tokio::test]
async fn test_advisory_policy_filter_keeps_registered_tool_visible() {
    let provider = MockProvider::with_responses(vec![
        ProviderResponse {
            content: None,
            tool_calls: vec![crate::traits::ToolCall {
                id: "call_hidden_cli_agent".to_string(),
                name: "cli_agent".to_string(),
                arguments: r#"{"task":"inspect workspace"}"#.to_string(),
                extra_content: None,
            }],
            usage: None,
            thinking: None,
            response_note: None,
        },
        MockProvider::text_response("Used the currently exposed tools instead."),
    ]);
    let extra_tools = vec![Arc::new(HighImpactCliAgentMock) as Arc<dyn crate::traits::Tool>];
    let policy_config = crate::config::PolicyConfig {
        tool_filter_enforce: false,
        ..Default::default()
    };
    let harness =
        crate::testing::setup_test_agent_with_policy(provider, policy_config, extra_tools)
            .await
            .unwrap();

    let response = harness
        .agent
        .handle_message(
            "hidden_tool_guess_test",
            "check the system specs on this machine",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert_eq!(response, "Used the currently exposed tools instead.");

    let calls = harness.provider.call_log.lock().await;
    assert!(
        calls.len() >= 2,
        "expected at least two LLM calls, got {}",
        calls.len()
    );

    let first_call_tool_names = extract_tool_names(&calls[0].tools);
    assert!(
        first_call_tool_names.contains(&"cli_agent".to_string()),
        "an advisory policy score must not hide a registered tool, got {:?}",
        first_call_tool_names
    );

    let second_call_has_tool_result = calls[1].messages.iter().any(|message| {
        message.get("role").and_then(|role| role.as_str()) == Some("tool")
            && message.get("name").and_then(|name| name.as_str()) == Some("cli_agent")
            && message
                .get("content")
                .and_then(|content| content.as_str())
                .is_some_and(|content| content.contains("cli agent executed"))
    });
    assert!(
        second_call_has_tool_result,
        "second call messages did not contain the execution result: {:?}",
        calls[1].messages
    );
}

#[tokio::test]
async fn test_executor_spawn_persists_structured_handoff_and_result_on_task() {
    let workspace = tempfile::tempdir().expect("synthetic executor workspace");
    let workspace_root = workspace
        .path()
        .canonicalize()
        .expect("canonical workspace");
    let workspace_path = workspace_root.to_string_lossy().to_string();
    let file_path = workspace_root.join("src/main.rs");
    let final_summary = format!(
        "Updated {} and reran the scoped checks successfully.",
        file_path.display()
    );
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "synthetic_patch",
            &serde_json::json!({"path": file_path}).to_string(),
        ),
        MockProvider::text_response(&final_summary),
    ]);
    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![Arc::new(StructuredPatchTool) as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Patch the regression", "delegation-task-context");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-structured-001".to_string(),
        goal_id: goal.id.clone(),
        description: format!("Patch {}", file_path.display()),
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
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&task).await.unwrap();

    let response = agent
        .spawn_child(
            &format!("Patch the scoped regression in {workspace_path}"),
            &format!("Patch {}", file_path.display()),
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(task.id.as_str()),
            Some(&workspace_path),
            None,
        )
        .await
        .unwrap();

    assert!(
        !response.trim().is_empty(),
        "executor should return a result"
    );

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "completed");
    let context: serde_json::Value = serde_json::from_str(updated.context.as_deref().unwrap())
        .expect("task context should be json");
    assert_eq!(
        context["executor_handoff"]["task_id"].as_str(),
        Some(task.id.as_str())
    );
    assert_eq!(
        context["executor_result"]["task_outcome"].as_str(),
        Some("task_done")
    );
    assert_eq!(
        context["executor_handoff"]["target_scope"]["allowed_targets"][0]["value"].as_str(),
        Some(workspace_path.as_str())
    );
}

#[tokio::test]
async fn operational_blocker_without_recovery_evidence_requires_recovery_receipt() {
    let workspace = tempfile::tempdir().expect("synthetic recovery workspace");
    let workspace_path = workspace.path().to_string_lossy().to_string();
    let file_path = workspace.path().join("post.md");
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "report_blocker",
            r#"{"reason":"The build result contradicts a direct parser check","blocker_class":"recovery_exhausted","external_effect_state":"none","recovery_attempts":[{"action":"Parsed the current file","outcome":"The id is valid","evidence":"The parser returned integer 108"}],"outcome":"blocked"}"#,
        ),
        MockProvider::tool_call_response(
            "synthetic_patch",
            &serde_json::json!({"path": file_path}).to_string(),
        ),
        MockProvider::text_response(
            "Refreshed the workspace, reran the original build, and completed the task successfully.",
        ),
    ]);
    let harness = setup_full_stack_test_agent_with_extra_tools(
        provider,
        vec![Arc::new(StructuredPatchTool) as Arc<dyn crate::traits::Tool>],
    )
    .await
    .unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Build the current blog post", "delegation-recovery-gate");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-recovery-gate-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Build the current blog post".to_string(),
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
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&task).await.unwrap();

    let response = agent
        .spawn_child(
            "Build the current blog post",
            "Build the current blog post",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(task.id.as_str()),
            Some(&workspace_path),
            None,
        )
        .await
        .unwrap();

    assert!(response.contains("reran the original build"), "{response}");
    assert_eq!(
        harness.provider.call_count().await,
        3,
        "a rejected blocker must return control to the executor until a receipt exists"
    );
    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "completed");
    assert!(updated.blocker.is_none());
}

#[tokio::test]
async fn test_report_blocker_terminates_executor_loop() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response(
            "report_blocker",
            r#"{"reason":"Docker daemon is not reachable","blocker_class":"external_dependency","external_effect_state":"none","recovery_attempts":[],"outcome":"blocked","exact_need":"Start Docker so ddev can connect.","next_step":"Re-run ddev composer update once Docker is up."}"#,
        ),
        MockProvider::text_response("SHOULD NOT BE REACHED - loop must end at report_blocker."),
    ]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-blocker-terminal");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-blocker-terminal-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
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
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&task).await.unwrap();

    let response = agent
        .spawn_child(
            "Run the ddev composer update for the Drupal site",
            "Run ddev composer update",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(task.id.as_str()),
            Some("/tmp/demo"),
            None,
        )
        .await
        .unwrap();

    // The executor's final response is the structured blocker summary,
    // not a follow-up LLM reply.
    assert!(
        response.contains("Executor outcome: blocked"),
        "expected blocker summary as final response, got: {response}"
    );
    assert!(
        !response.contains("SHOULD NOT BE REACHED"),
        "loop continued past report_blocker"
    );

    // Exactly one LLM call: the one that emitted report_blocker. No
    // reconciliation/verification iterations after a declared blocker.
    assert_eq!(
        harness.provider.call_count().await,
        1,
        "report_blocker must terminate the executor loop without further LLM calls"
    );

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "blocked");

    // The executor's run ends completed with partial work.
    let ends: Vec<String> = sqlx::query_scalar(
        "SELECT data FROM events WHERE event_type = 'task_end' ORDER BY id DESC",
    )
    .fetch_all(&agent.event_store().pool())
    .await
    .unwrap();
    let end: crate::events::TaskEndData =
        serde_json::from_str(ends.first().expect("executor task end")).unwrap();
    assert_eq!(end.status, crate::events::TaskStatus::Completed);
    assert_eq!(end.effective_outcome(), crate::events::TaskOutcome::Partial);
}

#[tokio::test]
async fn test_spawn_timeout_salvages_persisted_executor_outcome() {
    // Child future will be cancelled instantly (timeout_secs = 0), simulating
    // the parent timing out while the executor had already persisted a
    // terminal outcome via report_blocker.
    let provider = MockProvider::with_responses(vec![MockProvider::text_response(
        "child reply that the timeout will discard",
    )]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-timeout-salvage");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-timeout-salvage-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
        status: "blocked".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: Some(
            "Executor outcome: blocked\nSummary: Docker daemon is not reachable.".to_string(),
        ),
        error: None,
        blocker: Some("BLOCKED: Docker daemon is not reachable".to_string()),
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    harness.state.create_task(&task).await.unwrap();

    let runtime: Arc<dyn crate::runtime_ports::ChildAgentRuntime> = agent.clone();
    let spawn_tool = crate::tools::spawn::SpawnAgentTool::new(Arc::downgrade(&runtime), 4000, 0);
    let result = crate::traits::ToolCalls::call(
        &spawn_tool,
        &json!({
            "mission": "Run the ddev composer update",
            "task": "Run ddev composer update",
            "task_id": task.id,
        })
        .to_string(),
    )
    .await
    .unwrap();

    assert!(
        result.contains("Docker daemon is not reachable"),
        "timeout must salvage the persisted executor outcome, got: {result}"
    );
    assert!(
        !result.contains("timed out after"),
        "salvaged outcome must replace the generic timeout error, got: {result}"
    );

    // The persisted terminal status must survive the timeout handling.
    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "blocked");
}

#[tokio::test]
async fn test_background_spawn_timeout_salvages_persisted_executor_outcome() {
    use crate::traits::NotificationStore;

    let provider = MockProvider::new();
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-bg-salvage");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-bg-salvage-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
        status: "blocked".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: Some(
            "Executor outcome: blocked\nSummary: Docker daemon is not reachable.".to_string(),
        ),
        error: None,
        blocker: Some("BLOCKED: Docker daemon is not reachable".to_string()),
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    harness.state.create_task(&task).await.unwrap();

    let runtime: Arc<dyn crate::runtime_ports::ChildAgentRuntime> = agent.clone();
    let spawn_tool = crate::tools::spawn::SpawnAgentTool::new(Arc::downgrade(&runtime), 4000, 0)
        .with_state(harness.state.clone() as Arc<dyn crate::traits::StateStore>);
    let ack = crate::testing::call_with_facts(
        &spawn_tool,
        &json!({
            "mission": "Run the ddev composer update",
            "task": "Run ddev composer update",
            "task_id": task.id,
            "background": true,
        })
        .to_string(),
        crate::traits::ToolCallContext {
            goal_id: Some(goal.id.clone()),
            ..crate::testing::call_facts("delegation-bg-salvage")
        },
    )
    .await
    .unwrap();
    assert!(ack.contains("background"));

    let mut completion_message = None;
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let pending = harness.state.get_pending_notifications(20).await.unwrap();
        if let Some(entry) = pending
            .iter()
            .find(|n| n.message.contains("in the background: Run the ddev composer update"))
        {
            completion_message = Some(entry.message.clone());
            break;
        }
    }
    let message = completion_message.expect("background completion notification should be queued");
    assert!(
        message.contains("Docker daemon is not reachable"),
        "background timeout must salvage the persisted executor outcome, got: {message}"
    );
    assert!(
        !message.contains("timed out"),
        "salvaged outcome must replace the generic timeout message, got: {message}"
    );
}

#[tokio::test]
async fn test_executor_timeout_does_not_clobber_terminal_task_status() {
    let provider = MockProvider::new();
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Update Drupal modules", "delegation-timeout-clobber");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-timeout-clobber-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Run ddev composer update".to_string(),
        status: "blocked".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("task-lead".to_string()),
        context: None,
        result: Some(
            "Executor outcome: blocked\nSummary: Docker daemon is not reachable.".to_string(),
        ),
        error: None,
        blocker: Some("BLOCKED: Docker daemon is not reachable".to_string()),
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: Some(chrono::Utc::now().to_rfc3339()),
    };
    harness.state.create_task(&task).await.unwrap();

    agent.engine().mark_executor_task_timeout(&task.id, 300).await;

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(
        updated.status, "blocked",
        "timeout finalization must not overwrite an already-terminal task status"
    );
    assert!(
        updated
            .blocker
            .as_deref()
            .unwrap_or_default()
            .contains("Docker daemon is not reachable"),
        "persisted blocker must be preserved"
    );
}

#[tokio::test]
async fn test_executor_spawn_persists_needs_approval_blocker_result() {
    let provider = MockProvider::with_responses(vec![MockProvider::tool_call_response(
        "report_blocker",
        r#"{"reason":"Need approval to rotate the production credentials","blocker_class":"missing_authority","external_effect_state":"none","recovery_attempts":[],"outcome":"needs_approval","partial_work":"Validated the rotation script and staged the rollout notes","exact_need":"Owner approval to rotate the production credentials.","next_step":"Run the approved credential rotation and verify the service health.","target":"production credentials","dependency_repair":false}"#,
    )]);
    let harness = setup_full_stack_test_agent(provider).await.unwrap();
    let agent = Arc::new(harness.agent);

    let goal = Goal::new_finite("Rotate production credentials", "delegation-approval");
    harness.state.create_goal(&goal).await.unwrap();
    let task = crate::traits::Task {
        id: "task-approval-001".to_string(),
        goal_id: goal.id.clone(),
        description: "Rotate the production credentials".to_string(),
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
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&task).await.unwrap();

    let response = agent
        .spawn_child(
            "Rotate the production credentials safely",
            "Rotate the production credentials",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(task.id.as_str()),
            Some("/tmp/demo"),
            None,
        )
        .await
        .unwrap();

    // report_blocker is terminal: the structured summary is the final response.
    assert!(response.contains("Executor outcome: needs_approval"));

    let updated = harness.state.get_task(&task.id).await.unwrap().unwrap();
    assert_eq!(updated.status, "blocked");
    assert!(updated
        .result
        .as_deref()
        .unwrap_or_default()
        .contains("Executor outcome: needs_approval"));
    let context: serde_json::Value = serde_json::from_str(updated.context.as_deref().unwrap())
        .expect("task context should be json");
    assert_eq!(
        context["executor_result"]["task_outcome"].as_str(),
        Some("needs_approval")
    );
}

async fn plain_spawn_call(specialist: Option<&str>) -> (Vec<String>, String) {
    let provider =
        MockProvider::with_responses(vec![MockProvider::text_response("child finished")]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, Vec::new())
        .await
        .unwrap();
    let provider_log = harness.provider.clone();
    let agent = Arc::new(harness.agent);

    agent
        .spawn_child(
            "synthetic mission",
            "synthetic task",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            None,
            None,
            None,
            None,
            specialist,
        )
        .await
        .unwrap();

    let calls = provider_log.call_log.lock().await;
    let last = calls.last().expect("the child called the model");
    (
        extract_tool_names(&last.tools),
        format!("{:?}", last.messages),
    )
}

#[tokio::test]
async fn test_plain_spawn_without_specialist_keeps_the_parent_prompt_alone() {
    let (_, prompt) = plain_spawn_call(None).await;
    assert!(!prompt.contains("## Specialist Profile"));
}

#[tokio::test]
async fn test_plain_spawn_with_specialist_uses_its_prompt() {
    let (_, prompt) = plain_spawn_call(Some("code")).await;
    assert!(prompt.contains("You are a Code specialist."));
}

#[tokio::test]
async fn test_review_specialist_prompt_reaches_a_plain_spawn() {
    let (_, prompt) = plain_spawn_call(Some("review")).await;
    assert!(prompt.contains("You are a Review specialist."));
}

async fn run_limited_child(provider: MockProvider) -> crate::runtime_ports::ChildTimedOut {
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, Vec::new())
        .await
        .unwrap();
    let agent = Arc::new(harness.agent);
    let error = agent
        .spawn_child_with_outcome(
            "synthetic mission",
            "synthetic task",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(crate::runtime_ports::ChildTimeLimit {
                limit: std::time::Duration::from_millis(1_500),
            }),
            None,
        )
        .await
        .expect_err("the child runs past its limit");
    error
        .downcast::<crate::runtime_ports::ChildTimedOut>()
        .expect("a typed timeout")
}

#[tokio::test]
async fn test_child_past_its_limit_reports_the_actions_it_completed() {
    let timed_out = run_limited_child(MockProvider::with_delayed_responses(
        vec![
            MockProvider::tool_call_response("terminal", r#"{"command":"echo synthetic"}"#),
            MockProvider::text_response("never reached"),
        ],
        vec![
            std::time::Duration::ZERO,
            std::time::Duration::from_secs(30),
        ],
    ))
    .await;
    assert!(
        timed_out
            .report
            .changed
            .iter()
            .any(|item| item.starts_with("terminal:")),
        "{:?}",
        timed_out.report
    );
    assert!(timed_out.render().contains("Do NOT run it again"));
}

#[tokio::test]
async fn test_child_past_its_limit_with_no_actions_is_safe_to_rerun() {
    let timed_out = run_limited_child(MockProvider::with_delayed_responses(
        vec![MockProvider::text_response("never reached")],
        vec![std::time::Duration::from_secs(30)],
    ))
    .await;
    assert!(!timed_out.report.may_have_changed_something());
}

/// A child is bounded by progress, not elapsed time (Decision 1): one that
/// keeps working well past its idle limit, with a tool call longer than the
/// limit itself (an install that takes minutes), is never stopped.
#[tokio::test]
async fn test_a_slow_but_progressing_child_is_never_stopped() {
    let step = |seconds: &str| {
        MockProvider::tool_call_response(
            "terminal",
            &serde_json::json!({"command": format!("sleep {seconds} && echo synthetic-step")})
                .to_string(),
        )
    };
    let provider = MockProvider::with_responses(vec![
        step("1"),
        step("1"),
        step("2.5"),
        MockProvider::text_response("synthetic install finished"),
    ]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, Vec::new())
        .await
        .unwrap();
    let agent = Arc::new(harness.agent);
    let begun = std::time::Instant::now();
    let running = agent.clone();
    let run = tokio::spawn(async move {
        running
            .spawn_child_with_outcome(
                "Install a synthetic site\nwith every step",
                "synthetic task",
                None,
                ChannelContext::private("test"),
                UserRole::Owner,
                None,
                None,
                None,
                None,
                None,
                Some("telegram:synthetic-chat"),
                None,
                Some(crate::runtime_ports::ChildTimeLimit {
                    limit: std::time::Duration::from_millis(1_500),
                }),
                None,
            )
            .await
    });
    // While it runs, the chat's `/status` and `/tasks` list it by its title.
    let listed = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let children = aidaemon_channels::ports::ChannelAgentRuntime::running_children(
                agent.as_ref(),
                "telegram:synthetic-chat",
            )
            .await;
            if !children.is_empty() {
                break children;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the running child is listed");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].label, "Install a synthetic site");
    let run = run
        .await
        .unwrap()
        .expect("a progressing child runs to its end");
    assert!(
        begun.elapsed() > std::time::Duration::from_secs(4),
        "the child worked well past its idle limit: {:?}",
        begun.elapsed()
    );
    assert!(run.response.contains("synthetic install finished"), "{}", run.response);
    assert!(
        aidaemon_channels::ports::ChannelAgentRuntime::running_children(
            agent.as_ref(),
            "telegram:synthetic-chat"
        )
        .await
        .is_empty(),
        "a finished child is no longer listed"
    );
}

/// A child is listed for the chat from the moment it is created, before its
/// setup and first model call, and its start reaches the chat's card as its
/// own line (live 2026-11-12: a status read while two helpers were starting
/// said none ran).
#[tokio::test]
async fn test_a_child_is_listed_from_the_moment_it_is_created() {
    let provider = MockProvider::with_delayed_responses(
        vec![MockProvider::text_response("synthetic count done")],
        vec![std::time::Duration::from_millis(300)],
    );
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, Vec::new())
        .await
        .unwrap();
    let agent = Arc::new(harness.agent);
    let (status_tx, mut status_rx) = tokio::sync::mpsc::channel(64);
    let running = agent.clone();
    let run = tokio::spawn(async move {
        running
            .spawn_child_with_outcome(
                "Counter A",
                "synthetic count",
                Some(status_tx),
                ChannelContext::private("test"),
                UserRole::Owner,
                None,
                None,
                None,
                None,
                None,
                Some("telegram:synthetic-chat"),
                None,
                None,
                None,
            )
            .await
    });
    let first = tokio::time::timeout(std::time::Duration::from_secs(5), status_rx.recv())
        .await
        .expect("a status update arrives")
        .expect("the channel is open");
    assert!(
        matches!(
            &first,
            crate::types::StatusUpdate::Helper(view)
                if view.label == "Counter A"
                    && view.phase == aidaemon_core::card::RunPhase::Running
        ),
        "{first:?}"
    );
    // The start line is sent once the child is listed, so it is listed now.
    let listed = aidaemon_channels::ports::ChannelAgentRuntime::running_children(
        agent.as_ref(),
        "telegram:synthetic-chat",
    )
    .await;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].label, "Counter A");
    run.await.unwrap().expect("the child runs to its end");
}

/// A child charges the tree it belongs to, so a request's budget covers all
/// of its sub-agents.
#[tokio::test]
async fn test_child_tokens_count_against_the_requests_tree() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("done")]);
    let mut harness = setup_full_stack_test_agent_with_extra_tools(provider, Vec::new())
        .await
        .unwrap();
    let family = crate::agent::authority::SpawnFamily::new(Default::default());
    harness
        .agent
        .set_test_inherited_authority(crate::agent::authority::InheritedAuthority {
            authority: aidaemon_core::Authority::full(),
            owner_level: Some(crate::config::OwnerLevel::Full),
            task_access: None,
            mode_session: None,
            untrusted_context: false,
            origin_request: Some("synthetic request".to_string()),
            origin_trusted: true,
            trusted_roots: Vec::new(),
            declined: Vec::new(),
            shared_declines: None,
            learning: None,
            spawn_family: Some(family.clone()),
            child_declaration: None,
            parent_run: None,
            known_resources: Default::default(),
            card_title: None,
            card_steps: Vec::new(),
            lineage: Vec::new(),
            parent_task: None,
        });
    harness
        .agent
        .handle_message(
            "sub_agent_synthetic_child",
            "synthetic task",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(family.tokens().load(std::sync::atomic::Ordering::Relaxed) > 0);
}

/// The owner's cancel reaches a session's background sub-agents.
#[tokio::test]
async fn test_cancel_stops_a_sessions_background_children() {
    use crate::runtime_ports::ChildAgentRuntime;
    let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), Vec::new())
        .await
        .unwrap();
    let agent = Arc::new(harness.agent);
    harness
        .state
        .record_background_spawn(&crate::traits::BackgroundSpawn {
            id: "background_spawn:1".to_string(),
            session_id: "telegram:synthetic-user-1".to_string(),
            mission: "Publish the synthetic report".to_string(),
            started_at: chrono::Utc::now().to_rfc3339(),
        })
        .await
        .unwrap();
    let token = agent
        .background_cancel_token("background_spawn:1")
        .await
        .expect("the agent can cancel background work");

    let cancelled = agent
        .cancel_active_finite_work_for_session("telegram:synthetic-user-1")
        .await;

    assert!(token.is_cancelled());
    assert_eq!(
        cancelled,
        vec!["background task: Publish the synthetic report".to_string()]
    );
}

/// A stop or /clear ends work in flight, not a reminder the owner scheduled:
/// a finite goal waiting on its schedule survives, an unscheduled one does not.
#[tokio::test]
async fn cancel_keeps_a_scheduled_reminder_that_is_not_running() {
    let harness = setup_full_stack_test_agent_with_extra_tools(MockProvider::new(), Vec::new())
        .await
        .unwrap();
    let session = "telegram:synthetic-user-1";
    let reminder = crate::traits::Goal::new_finite("Remind me to water the plants", session);
    let in_flight = crate::traits::Goal::new_finite("Summarize the synthetic report", session);
    harness.state.create_goal(&reminder).await.unwrap();
    harness.state.create_goal(&in_flight).await.unwrap();
    let now = chrono::Utc::now();
    harness
        .state
        .create_goal_schedule(&crate::traits::GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: reminder.id.clone(),
            cron_expr: "30 15 12 11 *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("in one hour".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: true,
            is_paused: false,
            last_run_at: None,
            next_run_at: (now + chrono::Duration::hours(1)).to_rfc3339(),
            created_at: now.to_rfc3339(),
            updated_at: now.to_rfc3339(),
        })
        .await
        .unwrap();

    let cancelled = harness
        .agent
        .cancel_active_finite_work_for_session(session)
        .await;

    assert_eq!(cancelled, vec!["Summarize the synthetic report".to_string()]);
    let kept = harness.state.get_goal(&reminder.id).await.unwrap().unwrap();
    assert_eq!(kept.status, reminder.status);
    assert_eq!(
        harness.state.get_schedules_for_goal(&reminder.id).await.unwrap().len(),
        1
    );
}

/// Through the real agent: a spawn whose setup fails (here, a task lead
/// with no goal) gives its place back, and a child that ran keeps it.
#[tokio::test]
async fn only_a_child_that_reached_its_run_counts_toward_the_requests_total() {
    use crate::agent::authority::{within_permit, SpawnFamily, SpawnFamilyLimits};
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("done")]);
    let harness = setup_full_stack_test_agent_with_extra_tools(provider, Vec::new())
        .await
        .unwrap();
    let agent = Arc::new(harness.agent);
    let family = SpawnFamily::new(SpawnFamilyLimits {
        max_concurrent: 5,
        max_total: 1,
        ..Default::default()
    });
    let permit = family.try_admit().unwrap();
    within_permit(
        Some(&permit),
        agent.spawn_child(
            "synthetic mission",
            "synthetic task",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            Some(AgentRole::TaskLead),
            None,
            None,
            None,
            None,
        ),
    )
    .await
    .expect_err("a task lead with no goal stops during setup");
    drop(permit);

    let permit = family.try_admit().expect("the failed setup used no place");
    within_permit(
        Some(&permit),
        agent.spawn_child(
            "synthetic mission",
            "synthetic task",
            None,
            ChannelContext::private("test"),
            UserRole::Owner,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    .await
    .unwrap();
    drop(permit);

    assert!(family.try_admit().is_err(), "the child that ran kept it");
}

/// What a chat receives once a background helper with `script` as its
/// model ends, waiting at most a few seconds and sending nothing in.
async fn background_finish_reaches_chat(script: Vec<crate::traits::ProviderResponse>) -> Vec<String> {
    let session = "telegram:synthetic-chat-1";
    let harness = crate::testing::setup_test_agent_root_with_extra_tools_and_llm_timeout(
        MockProvider::with_responses(script),
        vec![],
        None,
    )
    .await
    .unwrap();
    let channel = Arc::new(crate::testing::TestChannel::new());
    let session_map: crate::channels::SessionMap = Arc::new(tokio::sync::RwLock::new(
        std::collections::HashMap::from([(session.to_string(), "test".to_string())]),
    ));
    let hub = Arc::new(crate::channels::ChannelHub::new(
        vec![channel.clone() as Arc<dyn crate::traits::HubChannel>],
        session_map,
    ));
    let outbound: Arc<dyn crate::runtime_ports::OutboundRouter> = hub.clone();
    let (agent, files) = harness.into_agent();
    let agent = Arc::new(agent);
    let runtime: Arc<dyn crate::runtime_ports::ChildAgentRuntime> = agent.clone();
    let spawn_tool = crate::tools::spawn::SpawnAgentTool::new(Arc::downgrade(&runtime), 4000, 30)
        .with_state(files.state.clone() as Arc<dyn crate::traits::StateStore>);
    spawn_tool.set_hub(Arc::downgrade(&outbound));
    let started = crate::testing::call_with_facts(
        &spawn_tool,
        &json!({
            "title": "Slow count",
            "task": "Count to 3.",
            "background": true,
        })
        .to_string(),
        crate::testing::call_facts(session),
    )
    .await
    .unwrap();
    assert!(started.contains("Started in the background"), "{started}");
    for _ in 0..200 {
        let sent = channel.messages_for(session).await;
        if sent.iter().any(|text| text.contains("in the background: Slow count")) {
            return sent;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    channel.messages_for(session).await
}

/// A background helper's result reaches its chat the moment it ends, with
/// nothing sent in meanwhile (live 2026-11-13: a finished count's result
/// was looked for until the owner's next message): after a plain final
/// reply, and after a final reply the engine rejected and retried once.
#[tokio::test]
async fn a_background_helpers_result_reaches_the_chat_when_it_ends() {
    let plain = background_finish_reaches_chat(vec![MockProvider::text_response(
        "Done. I counted to 3.",
    )])
    .await;
    assert!(
        plain.iter().any(|text| text.contains("in the background: Slow count")),
        "{plain:?}"
    );

    let checklist = r#"{"items":[
        {"text":"Count to 3","status":"in_progress","mutation_effects":["process_state"]}
    ]}"#;
    let retried = background_finish_reaches_chat(vec![
        MockProvider::tool_call_response("track_requirements", checklist),
        MockProvider::text_response("Done. I counted to 3."),
        MockProvider::text_response("Done. I counted to 3."),
    ])
    .await;
    assert!(
        retried.iter().any(|text| text.contains("in the background: Slow count")),
        "{retried:?}"
    );
}

/// Live 2026-11-13: a Drupal helper took 7m 54s, the parent then repaired
/// the homepage itself and answered that the total "wasn't measured",
/// though the owner had asked how long it takes. A request the parent
/// finishes after a helper it waited on ends with what the runtime
/// measured, under the final answer: the parent's own work after the helper
/// and the request's total from the owner's message to the answer.
#[tokio::test]
async fn a_request_finished_by_the_parent_after_its_helper_shows_its_total() {
    use crate::runtime_ports::AgentIngress;
    let script = vec![
        MockProvider::tool_call_response(
            "spawn_agent",
            &json!({"title": "Drupal install", "mission": "Install Drupal", "task": "Install Drupal."})
                .to_string(),
        ),
        MockProvider::text_response("Installed Drupal."),
        MockProvider::tool_call_response(
            "synthetic_patch",
            &json!({"path": "/tmp/aidaemon-test/drupal/web/index.php"}).to_string(),
        ),
        MockProvider::text_response("Drupal is installed and its homepage is repaired."),
    ];
    let spawn_tool = Arc::new(crate::tools::spawn::SpawnAgentTool::new_deferred(4000, 60));
    let tools: Vec<Arc<dyn crate::traits::Tool>> = vec![
        Arc::new(StructuredPatchTool),
        Arc::new(OfferedSpawnAgent(spawn_tool.clone())),
    ];
    let mut harness = setup_test_agent_with_extra_tools_and_llm_timeout(
        MockProvider::with_responses(script),
        tools,
        None,
    )
    .await
    .unwrap();
    harness.agent.set_test_orchestrator_mode();
    let agent = Arc::new(harness.agent);
    let runtime: Arc<dyn crate::runtime_ports::ChildAgentRuntime> = agent.clone();
    spawn_tool.set_agent(Arc::downgrade(&runtime));
    let reply = agent
        .handle_inbound_message(crate::runtime_ports::InboundMessageRequest {
            session_id: "telegram:synthetic-total".to_string(),
            user_text: "Install Drupal with a helper and tell me how long it takes.".to_string(),
            attachments: Vec::new(),
            status_tx: None,
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("telegram"),
            heartbeat: None,
            ingress_timing: None,
        })
        .await
        .unwrap()
        .text;
    let measured = aidaemon_runtime::turn::sliding_window::MEASURED_TIMING_MARK;
    assert!(
        reply.contains(&format!("{measured}Finished by me after the helper: Drupal install")),
        "{reply}"
    );
    let total = reply
        .lines()
        .find(|line| line.starts_with(&format!("{measured}Finished ")) && line.contains("after your request ("))
        .unwrap_or_else(|| panic!("the request's total under the answer: {reply}"));
    assert!(total.ends_with("ET)."), "{total}");
    assert!(!reply.contains("wasn't measured"), "{reply}");
}
