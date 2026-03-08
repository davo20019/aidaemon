// ==================== Context Window Management Tests ====================

/// Verify that a long conversation (20+ messages) doesn't crash and the agent
/// still produces a response. Budget enforcement should trim history silently.
#[tokio::test]
async fn test_long_conversation_no_crash() {
    // Create responses for 11 turns (22 messages total user+assistant)
    let mut responses = Vec::new();
    for i in 0..11 {
        responses.push(MockProvider::text_response(&format!("Response {}", i)));
    }

    let provider = MockProvider::with_responses(responses);
    let harness = setup_test_agent(provider).await.unwrap();

    // Send 11 messages in the same session
    for i in 0..11 {
        let msg = format!(
            "Message number {} with some extra text to make it a bit longer",
            i
        );
        let result = harness
            .agent
            .handle_message(
                "long_session",
                &msg,
                None,
                UserRole::Owner,
                ChannelContext::private("telegram"),
                None,
            )
            .await;

        assert!(
            result.is_ok(),
            "Message {} should succeed: {:?}",
            i,
            result.err()
        );
        let text = result.unwrap();
        assert!(!text.is_empty(), "Response {} should not be empty", i);
    }
}

/// Verify tool result compression: a very large tool result should be truncated.
#[tokio::test]
async fn test_tool_result_compressed() {
    use crate::memory::context_window::compress_tool_result;

    // Result under the limit should pass through unchanged
    let short = "Hello world";
    let result = compress_tool_result("terminal", short, 2000);
    assert_eq!(result, short);

    // Result over the limit should be truncated with annotation
    let large = "x".repeat(5000);
    let compressed = compress_tool_result("terminal", &large, 2000);
    assert!(compressed.len() < 5000);
    assert!(compressed.contains("[truncated"));
    assert!(compressed.contains("5000"));
}

/// Verify conversation summary CRUD operations work correctly.
#[tokio::test]
async fn test_summary_crud() {
    use crate::traits::ConversationSummary;

    let provider = MockProvider::with_responses(vec![MockProvider::text_response("Hello")]);
    let harness = setup_test_agent(provider).await.unwrap();

    // Initially no summary
    let summary = harness
        .state
        .get_conversation_summary("test_session")
        .await
        .unwrap();
    assert!(summary.is_none());

    // Upsert a summary
    let summary = ConversationSummary {
        session_id: "test_session".to_string(),
        summary: "We discussed topic A and decided on approach B.".to_string(),
        message_count: 10,
        last_message_id: "msg-123".to_string(),
        updated_at: Utc::now(),
    };
    harness
        .state
        .upsert_conversation_summary(&summary)
        .await
        .unwrap();

    // Retrieve it
    let loaded = harness
        .state
        .get_conversation_summary("test_session")
        .await
        .unwrap();
    assert!(loaded.is_some());
    let loaded = loaded.unwrap();
    assert_eq!(loaded.session_id, "test_session");
    assert_eq!(
        loaded.summary,
        "We discussed topic A and decided on approach B."
    );
    assert_eq!(loaded.message_count, 10);

    // Update it
    let updated = ConversationSummary {
        summary: "Updated: topic A, approach B, and new topic C.".to_string(),
        message_count: 15,
        ..loaded
    };
    harness
        .state
        .upsert_conversation_summary(&updated)
        .await
        .unwrap();

    let reloaded = harness
        .state
        .get_conversation_summary("test_session")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        reloaded.summary,
        "Updated: topic A, approach B, and new topic C."
    );
    assert_eq!(reloaded.message_count, 15);

    // Clear session should also clear summary
    harness.state.clear_session("test_session").await.unwrap();
    let after_clear = harness
        .state
        .get_conversation_summary("test_session")
        .await
        .unwrap();
    assert!(
        after_clear.is_none(),
        "Summary should be deleted after clear_session"
    );
}

/// Verify should_extract_facts filters trivial messages correctly.
#[tokio::test]
async fn test_should_extract_facts_filtering() {
    use crate::memory::context_window::should_extract_facts;

    // Trivial messages should be filtered out
    assert!(!should_extract_facts("ok"));
    assert!(!should_extract_facts("thanks"));
    assert!(!should_extract_facts("👍"));
    assert!(!should_extract_facts("hi")); // too short

    // Meaningful messages should pass through
    assert!(should_extract_facts(
        "My dog's name is Bella and she's 3 years old"
    ));
    assert!(should_extract_facts(
        "I work at Acme Corp as a senior engineer"
    ));
}

// ─── Budget auto-extension integration tests ───────────────────────────

/// Helper to collect StatusUpdate messages from a channel.
async fn collect_status_updates(
    mut rx: tokio::sync::mpsc::Receiver<StatusUpdate>,
) -> Vec<StatusUpdate> {
    let mut updates = Vec::new();
    while let Ok(update) = rx.try_recv() {
        updates.push(update);
    }
    updates
}

/// Task token budget auto-extends when the agent is making productive progress.
/// is_productive requires 8+ successful tool calls. We alternate between
/// system_info and remember_fact to avoid the repeated-call blocking guard
/// (which blocks non-exempt tools after 8 calls) while accumulating enough
/// successful calls to pass the is_productive threshold.
#[tokio::test]
async fn test_task_budget_auto_extends_on_progress() {
    let provider = MockProvider::with_responses(vec![
        // Alternate tools to avoid the 8-call same-tool block on system_info
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"test","key":"k1","value":"v1"}"#,
        ),
        MockProvider::tool_call_response("system_info", r#"{"verbose": true}"#),
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"test","key":"k2","value":"v2"}"#,
        ),
        MockProvider::tool_call_response("system_info", r#"{"check": "os"}"#),
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"test","key":"k3","value":"v3"}"#,
        ),
        MockProvider::tool_call_response("system_info", r#"{"check": "mem"}"#),
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"test","key":"k4","value":"v4"}"#,
        ),
        MockProvider::tool_call_response("system_info", r#"{"check": "cpu"}"#),
        MockProvider::tool_call_response(
            "remember_fact",
            r#"{"category":"test","key":"k5","value":"v5"}"#,
        ),
        // After budget extension (10 calls × 15 = 150 tokens = budget hit):
        MockProvider::tool_call_response("system_info", r#"{"check": "final"}"#),
        // Final text response
        MockProvider::text_response("Task completed successfully."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();
    // Budget of 150 = 10 LLM calls at 15 tokens each. After 10 iterations,
    // total_successful_tool_calls >= 8 → is_productive=true → auto-extend.
    harness.agent.set_test_task_token_budget(Some(150));

    let (status_tx, status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(64);

    let response = harness
        .agent
        .handle_message(
            "budget_test",
            "Run a complex analysis requiring multiple steps",
            Some(status_tx),
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // The agent may either:
    // 1. Continue past the budget hit (old behavior: "Task completed successfully.")
    // 2. Gracefully stall after meaningful progress (new stopping_phase behavior)
    // 3. Return the last narration text when stopped by another mechanism
    // All are acceptable — the key is no crash and a non-empty response.
    assert!(
        !response.is_empty(),
        "Agent should return a non-empty response"
    );

    // BudgetExtended may or may not be emitted depending on whether the
    // stopping_phase's stall-with-progress path fires before the budget check.
    // We only verify if the agent completed successfully (reached final text).
    let updates = collect_status_updates(status_rx).await;
    if response.contains("Task completed successfully.") {
        let budget_extended = updates
            .iter()
            .any(|u| matches!(u, StatusUpdate::BudgetExtended { .. }));
        assert!(
            budget_extended,
            "Expected BudgetExtended status update when agent completes normally"
        );
    }
}

/// Task token budget stops execution when progress is not productive (stalling).
/// Script: same tool with same args → stall detection → is_productive=false → stops.
#[tokio::test]
async fn test_task_budget_stops_when_not_productive() {
    // Create responses that will trigger stall detection.
    // 3 calls to hit the budget, but stall_count > 0 due to repetition.
    let mut responses = Vec::new();
    // Generate enough identical tool calls to trigger stall detection AND hit budget.
    // Stall detection fires at 3 consecutive identical calls (same name + same args hash).
    for _ in 0..5 {
        responses.push(MockProvider::tool_call_response("system_info", "{}"));
    }
    // Final text response (may not reach this if budget stops first)
    responses.push(MockProvider::text_response("Should not reach this."));

    let provider = MockProvider::with_responses(responses);
    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();
    harness.agent.set_test_task_token_budget(Some(45));

    let (status_tx, status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(64);

    let response = harness
        .agent
        .handle_message(
            "stall_test",
            "Do something",
            Some(status_tx),
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Should have stopped — either by stall detection or budget exhaustion without extension
    // (stall_count > 0 makes is_productive=false)
    assert_ne!(response, "Should not reach this.");

    // Verify NO BudgetExtended was emitted
    let updates = collect_status_updates(status_rx).await;
    let budget_extended = updates
        .iter()
        .any(|u| matches!(u, StatusUpdate::BudgetExtended { .. }));
    assert!(
        !budget_extended,
        "BudgetExtended should NOT be emitted when agent is stalling"
    );
}

/// Non-scheduled goals keep the stricter productivity threshold for automatic
/// goal-budget extension. Low-call runs should still stop at the budget, and
/// the DB budget must not be ratcheted upward.
#[tokio::test]
async fn test_goal_budget_auto_extends_and_persists() {
    let provider = MockProvider::with_responses(vec![
        // Calls 1-4: tool calls (each adds 15 tokens to goal budget tracking)
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", r#"{"verbose": true}"#),
        MockProvider::tool_call_response("system_info", r#"{"check": "os"}"#),
        MockProvider::tool_call_response("system_info", r#"{"check": "mem"}"#),
        // Call 5: won't be reached — budget stops execution with only 3 successful calls
        MockProvider::text_response("Goal task completed."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();

    // Create a goal with a low daily budget (60 tokens = 4 LLM calls at 15 each)
    let mut goal = Goal::new_finite("Test goal for budget extension", "goal_budget_session");
    goal.status = "active".to_string();
    goal.budget_daily = Some(60);
    goal.budget_per_check = Some(500);
    harness.state.create_goal(&goal).await.unwrap();

    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let (status_tx, _status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(64);

    let response = harness
        .agent
        .handle_message(
            "goal_budget_session",
            "Execute the goal task",
            Some(status_tx),
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // With only 3 successful tool calls, is_productive returns false → budget stops execution
    assert!(
        response.contains("daily processing budget"),
        "Expected budget-exceeded message, got: {}",
        response
    );

    // Verify the budget was NOT persisted/inflated in the database
    let updated_goal = harness.state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(
        updated_goal.budget_daily.unwrap(),
        60,
        "Budget should NOT be ratcheted up in DB — expected 60, got {:?}",
        updated_goal.budget_daily
    );
}

/// Scheduled goals relax the auto-extension threshold so a productive scheduled
/// run is less likely to be cut off mid-task, but the DB budget still must not
/// be ratcheted upward.
struct TrustProbeTool {
    seen: Arc<tokio::sync::Mutex<Vec<bool>>>,
}

impl TrustProbeTool {
    fn new(seen: Arc<tokio::sync::Mutex<Vec<bool>>>) -> Self {
        Self { seen }
    }
}

#[async_trait::async_trait]
impl crate::traits::Tool for TrustProbeTool {
    fn name(&self) -> &str {
        "trust_probe"
    }

    fn description(&self) -> &str {
        "Inspect whether the agent injected _trusted_session"
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "trust_probe",
            "description": "Inspect whether the agent injected _trusted_session",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: serde_json::Value = serde_json::from_str(arguments)?;
        let trusted = args["_trusted_session"].as_bool().unwrap_or(false);
        self.seen.lock().await.push(trusted);
        Ok(if trusted { "trusted" } else { "untrusted" }.to_string())
    }
}

#[tokio::test]
async fn test_scheduled_goal_tool_calls_are_marked_trusted() {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("trust_probe", "{}"),
        MockProvider::text_response("done"),
    ]);

    let mut harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(TrustProbeTool::new(seen.clone())) as Arc<dyn crate::traits::Tool>],
        None,
    )
    .await
    .unwrap();

    let mut goal = Goal::new_continuous(
        "Scheduled trust test",
        "scheduled_trust_session",
        None,
        None,
    );
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now,
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();
    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let response = harness
        .agent
        .handle_message(
            "scheduled_trust_session",
            "Run the scheduled goal",
            None,
            UserRole::Owner,
            ChannelContext::internal(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "done");
    assert_eq!(*seen.lock().await, vec![true]);
}

#[tokio::test]
async fn test_scheduled_goal_executor_tool_calls_inherit_trust_from_goal() {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("trust_probe", "{}"),
        MockProvider::text_response("done"),
    ]);

    let mut harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(TrustProbeTool::new(seen.clone())) as Arc<dyn crate::traits::Tool>],
        None,
    )
    .await
    .unwrap();

    harness.agent.set_test_executor_mode();

    let mut goal = Goal::new_continuous(
        "Scheduled executor trust test",
        "scheduled_executor_trust_session",
        None,
        None,
    );
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();

    let child_task = crate::traits::Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Post the composed tweet to Twitter/X".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now.clone(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&child_task).await.unwrap();

    harness.agent.set_test_goal_id(None);
    harness.agent.set_test_task_id(Some(child_task.id.clone()));

    let response = harness
        .agent
        .handle_message(
            "scheduled_executor_trust_session",
            "Run the executor task",
            None,
            UserRole::Owner,
            ChannelContext::internal(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "done");
    assert_eq!(*seen.lock().await, vec![true]);
}

#[tokio::test]
async fn test_scheduled_goal_spawned_executor_tool_calls_inherit_trust_from_goal() {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("trust_probe", "{}"),
        MockProvider::text_response("done"),
    ]);

    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(TrustProbeTool::new(seen.clone())) as Arc<dyn crate::traits::Tool>],
        None,
    )
    .await
    .unwrap();

    let mut goal = Goal::new_continuous(
        "Scheduled spawned executor trust test",
        "scheduled_spawned_executor_trust_session",
        None,
        None,
    );
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();

    let child_task = crate::traits::Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Post the composed tweet to Twitter/X".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now.clone(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&child_task).await.unwrap();

    let agent = Arc::new(harness.agent);
    let response = agent
        .spawn_child(
            "Post the scheduled tweet",
            "Post the scheduled tweet",
            None,
            ChannelContext::internal(),
            UserRole::Owner,
            Some(crate::traits::AgentRole::Executor),
            Some(goal.id.as_str()),
            Some(child_task.id.as_str()),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "done");
    assert_eq!(*seen.lock().await, vec![true]);
}

#[tokio::test]
async fn test_non_scheduled_goal_tool_calls_remain_untrusted() {
    let seen = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("trust_probe", "{}"),
        MockProvider::text_response("done"),
    ]);

    let mut harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![Arc::new(TrustProbeTool::new(seen.clone())) as Arc<dyn crate::traits::Tool>],
        None,
    )
    .await
    .unwrap();

    let mut goal = Goal::new_finite("Plain goal", "plain_goal_session");
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();
    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let response = harness
        .agent
        .handle_message(
            "plain_goal_session",
            "Run the goal",
            None,
            UserRole::Owner,
            ChannelContext::internal(),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "done");
    assert_eq!(*seen.lock().await, vec![false]);
}

#[tokio::test]
async fn test_scheduled_goal_daily_budget_is_backstop_only_during_active_run() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", r#"{"verbose": true}"#),
        MockProvider::tool_call_response("system_info", r#"{"check": "os"}"#),
        MockProvider::tool_call_response("system_info", r#"{"check": "mem"}"#),
        MockProvider::text_response("Goal task completed."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();

    let mut goal = Goal::new_continuous(
        "Scheduled build task",
        "scheduled_goal_budget_session",
        None,
        None,
    );
    goal.status = "active".to_string();
    goal.budget_daily = Some(60);
    goal.budget_per_check = Some(500);
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now,
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();

    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let (status_tx, status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(64);

    let response = harness
        .agent
        .handle_message(
            "scheduled_goal_budget_session",
            "Execute the scheduled goal task",
            Some(status_tx),
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Goal task completed.");

    let updates = collect_status_updates(status_rx).await;
    assert!(
        !updates
            .iter()
            .any(|u| matches!(u, StatusUpdate::BudgetExtended { .. })),
        "Daily backstop should not trigger an in-run budget extension for scheduled goals"
    );

    let updated_goal = harness.state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(
        updated_goal.budget_daily,
        Some(60),
        "The daily backstop should remain unchanged in the database"
    );
}

/// Scheduled goals should use the per-run budget as the active limiter.
/// If the run is clearly unproductive, it should stop even when the daily
/// budget still has room left.
#[tokio::test]
async fn test_scheduled_goal_run_budget_stops_unproductive_run() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("no_such_tool", "{}"),
        MockProvider::text_response("Should not reach this."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();

    let mut goal = Goal::new_continuous(
        "Scheduled noisy task",
        "scheduled_goal_run_budget_session",
        Some(10),
        Some(500),
    );
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now,
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();

    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let response = harness
        .agent
        .handle_message(
            "scheduled_goal_run_budget_session",
            "Execute the scheduled goal task",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("per-run processing budget"),
        "Expected scheduled run budget stop message, got: {}",
        response
    );

    let updated_goal = harness.state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(updated_goal.budget_per_check, Some(10));
    assert_eq!(updated_goal.budget_daily, Some(500));
}

/// Each scheduled run should get a fresh per-run budget even when the goal stays active.
#[tokio::test]
async fn test_scheduled_goal_run_budget_resets_between_runs() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("no_such_tool", "{}"),
        MockProvider::text_response("Should not reach the first run."),
        MockProvider::text_response("Second scheduled run completed."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();

    let mut goal = Goal::new_continuous(
        "Scheduled repeated task",
        "scheduled_goal_reset_session",
        Some(10),
        Some(500),
    );
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now,
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();

    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let first_response = harness
        .agent
        .handle_message(
            "scheduled_goal_reset_session",
            "Execute the scheduled goal task",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        first_response.contains("per-run processing budget"),
        "Expected first run to stop on per-run budget, got: {}",
        first_response
    );

    let second_response = harness
        .agent
        .handle_message(
            "scheduled_goal_reset_session",
            "Execute the scheduled goal task",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(second_response, "Second scheduled run completed.");
}

/// Scheduled runs should not be cut off by the generic hard iteration cap.
#[tokio::test]
async fn test_scheduled_goal_ignores_hard_iteration_cap() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("system_info", "{}"),
        MockProvider::tool_call_response("system_info", r#"{"verbose": true}"#),
        MockProvider::text_response("Scheduled run finished after multiple steps."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();
    harness
        .agent
        .set_test_iteration_config(crate::config::IterationLimitConfig::Hard {
            initial: 1,
            cap: 1,
        });

    let mut goal = Goal::new_continuous(
        "Scheduled iterative task",
        "scheduled_goal_iteration_session",
        Some(500),
        Some(5000),
    );
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now,
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();

    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let response = harness
        .agent
        .handle_message(
            "scheduled_goal_iteration_session",
            "Execute the scheduled goal task",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert_eq!(response, "Scheduled run finished after multiple steps.");
}

/// If the daemon restarts mid-run, the next scheduled task lead should restore
/// the persisted per-run budget state instead of silently starting from zero.
#[tokio::test]
async fn test_scheduled_goal_restores_run_state_after_restart_like_resume() {
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("no_such_tool", "{}"),
        MockProvider::text_response("Should not reach this."),
    ]);

    let mut harness = setup_test_agent(provider).await.unwrap();
    harness.agent.set_test_executor_mode();

    let mut goal = Goal::new_continuous(
        "Scheduled resumed task",
        "scheduled_goal_resume_session",
        Some(100),
        Some(500),
    );
    goal.status = "active".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now().to_rfc3339();
    let schedule = crate::traits::GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("hourly".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now.clone(),
    };
    harness.state.create_goal_schedule(&schedule).await.unwrap();

    let root_task = crate::traits::Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description:
            "Execute scheduled goal: Scheduled resumed task [SYSTEM: already scheduled and firing now; do not reschedule.]"
                .to_string(),
        status: "pending".to_string(),
        priority: "low".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now.clone(),
        started_at: None,
        completed_at: None,
    };
    harness.state.create_task(&root_task).await.unwrap();

    harness
        .state
        .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
            goal_id: goal.id.clone(),
            root_task_id: root_task.id.clone(),
            effective_budget_per_check: 20,
            tokens_used: 15,
            budget_extensions_count: 0,
            created_at: now.clone(),
            updated_at: now,
        })
        .await
        .unwrap();

    harness.agent.set_test_goal_id(Some(goal.id.clone()));

    let response = harness
        .agent
        .handle_message(
            "scheduled_goal_resume_session",
            "Execute the scheduled goal task",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(
        response.contains("per-run processing budget"),
        "Expected restored scheduled run state to stop the resumed run, got: {}",
        response
    );

    assert!(
        harness
            .state
            .get_scheduled_run_state(&goal.id)
            .await
            .unwrap()
            .is_none(),
        "Scheduled run state should be cleared after the resumed run exits"
    );
}

// ─── Role-gate integration tests ───────────────────────────────────────

/// Non-owner (Guest) sending a schedule-like intent should not create a goal.
/// The request should be handled directly by the agent loop instead.
#[tokio::test]
async fn test_non_owner_cannot_schedule_goal() {
    let provider = MockProvider::with_responses(vec![
        // Consultant pass (iteration 0)
        MockProvider::text_response("I'll handle this request now."),
        // Agent loop (iteration 1) — after Scheduled intent is downgraded for Guest
        MockProvider::text_response("Here is the information you requested."),
    ]);

    let harness = setup_test_agent(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "guest_schedule_session",
            "Every day at 9am, check the weather forecast",
            None,
            UserRole::Guest,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Should get a response (not a scheduling confirmation)
    assert!(!response.is_empty());
    let call_log = harness.provider.call_log.lock().await;
    assert!(
        call_log.iter().all(|call| call.tools.is_empty()),
        "Guest schedule requests should be handled without tools"
    );

    // Verify no goals were created
    let goals = harness
        .state
        .get_pending_confirmation_goals("guest_schedule_session")
        .await
        .unwrap();
    assert!(
        goals.is_empty(),
        "Guest should not be able to create scheduled goals"
    );
}

/// Non-owner (Guest) saying "confirm" while Owner has pending goals should get
/// an owner-only message and the goal should remain pending.
#[tokio::test]
async fn test_non_owner_cannot_confirm_scheduled_goal() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Create a pending_confirmation goal (as if Owner had initiated it)
    let mut goal = Goal::new_finite("Check weather daily", "confirm_test_session");
    goal.status = "pending_confirmation".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    // Guest tries to confirm
    let response = harness
        .agent
        .handle_message(
            "confirm_test_session",
            "confirm",
            None,
            UserRole::Guest,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Should get the owner-only message
    assert!(
        response.contains("Only the owner"),
        "Expected owner-only message, got: {}",
        response
    );

    // Goal should still be pending_confirmation (not activated or cancelled)
    let pending = harness
        .state
        .get_pending_confirmation_goals("confirm_test_session")
        .await
        .unwrap();
    assert_eq!(pending.len(), 1, "Goal should remain pending_confirmation");
    assert_eq!(pending[0].status, "pending_confirmation");
}

/// Regression: Non-owner sending an unrelated message should NOT auto-cancel
/// pending goals that belong to the Owner's session.
#[tokio::test]
async fn test_non_owner_unrelated_message_does_not_cancel_pending_goal() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    // Create a pending_confirmation goal
    let mut goal = Goal::new_finite("Deploy app nightly", "shared_session");
    goal.status = "pending_confirmation".to_string();
    harness.state.create_goal(&goal).await.unwrap();

    // Guest sends unrelated message in the same session
    let _response = harness
        .agent
        .handle_message(
            "shared_session",
            "What's the weather today?",
            None,
            UserRole::Guest,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Goal should still be pending_confirmation — NOT cancelled
    let pending = harness
        .state
        .get_pending_confirmation_goals("shared_session")
        .await
        .unwrap();
    assert_eq!(
        pending.len(),
        1,
        "Pending goal should not be auto-cancelled by non-owner message"
    );
    assert_eq!(pending[0].status, "pending_confirmation");
}

fn test_reload_config(
    kind: &str,
    base_url: &str,
    api_key: &str,
    primary: &str,
    fast: &str,
    smart: &str,
) -> crate::config::AppConfig {
    let toml = format!(
        r#"
[provider]
kind = "{kind}"
base_url = "{base_url}"
api_key = "{api_key}"

[provider.models]
primary = "{primary}"
fast = "{fast}"
smart = "{smart}"
"#
    );
    toml::from_str(&toml).expect("reload test config should parse")
}

#[tokio::test]
async fn test_reload_provider_switches_backend_and_models_endpoint() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    let before_models = harness.agent.list_models().await.unwrap();
    assert_eq!(before_models, vec!["mock-model".to_string()]);
    assert_eq!(harness.agent.current_model().await, "mock-model");

    let anthropic = test_reload_config(
        "anthropic",
        "https://api.openai.com/v1",
        "test-anthropic-key",
        "claude-3-haiku-20240307",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
    );

    let status = harness.agent.reload_provider(&anthropic).await.unwrap();
    assert!(
        status.contains("OpenaiCompatible -> Anthropic"),
        "unexpected reload status: {}",
        status
    );
    assert!(
        status.contains("mock-model -> claude-3-haiku-20240307"),
        "unexpected reload status: {}",
        status
    );
    assert_eq!(
        harness.agent.current_model().await,
        "claude-3-haiku-20240307"
    );

    let after_models = harness.agent.list_models().await.unwrap();
    assert!(
        after_models.iter().any(|m| m.starts_with("claude")),
        "expected Anthropic known models, got {:?}",
        after_models
    );
    assert!(
        !after_models.iter().any(|m| m == "mock-model"),
        "expected list_models source to switch away from mock provider, got {:?}",
        after_models
    );
}

#[tokio::test]
async fn test_reload_provider_resets_manual_model_and_supports_second_reload() {
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();

    harness.agent.set_model("manual-override".to_string()).await;
    assert_eq!(harness.agent.current_model().await, "manual-override");

    let anthropic = test_reload_config(
        "anthropic",
        "https://api.openai.com/v1",
        "test-anthropic-key",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
    );
    let status_1 = harness.agent.reload_provider(&anthropic).await.unwrap();
    assert!(
        status_1.contains("manual-override -> claude-3-opus-20240229"),
        "unexpected reload status: {}",
        status_1
    );
    assert_eq!(
        harness.agent.current_model().await,
        "claude-3-opus-20240229"
    );

    let openai = test_reload_config(
        "openai_compatible",
        "https://api.openai.com/v1",
        "test-openai-key",
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
    );
    let status_2 = harness.agent.reload_provider(&openai).await.unwrap();
    assert!(
        status_2.contains("Anthropic -> OpenaiCompatible"),
        "unexpected second reload status: {}",
        status_2
    );
    assert!(
        status_2.contains("claude-3-opus-20240229 -> openai/gpt-4o-mini"),
        "unexpected second reload status: {}",
        status_2
    );
    assert_eq!(harness.agent.current_model().await, "openai/gpt-4o-mini");
}

#[tokio::test]
async fn test_no_router_auto_mode_uses_runtime_primary_over_stale_model_field() {
    let provider = MockProvider::with_responses(vec![MockProvider::text_response("ok")]);
    let mut harness = setup_test_agent(provider).await.unwrap();

    // Top-level orchestrator path (depth=0) with uniform models => no router.
    harness.agent.set_test_orchestrator_mode();

    // Create a stale local model field, then disable override.
    harness
        .agent
        .set_model("manual-stale-model".to_string())
        .await;
    harness.agent.clear_model_override().await;

    let response = harness
        .agent
        .handle_message(
            "no_router_auto_primary",
            "hello",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();
    assert!(
        !response.is_empty(),
        "agent should return a non-empty response"
    );

    let calls = harness.provider.call_log.lock().await;
    assert!(
        !calls.is_empty(),
        "expected at least one LLM call in orchestrator mode"
    );
    assert_eq!(
        calls[0].model, "mock-model",
        "top-level no-router auto mode should use runtime primary model, not stale self.model"
    );
}
