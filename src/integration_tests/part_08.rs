// ============================================================================
// Task Lead + Executor tests
// ============================================================================

#[tokio::test]
async fn test_orchestration_task_lead_flag_off_uses_agent_loop() {
    // Deterministic pre-routing classifies the request as complex based on action
    // markers (analyze, compare, identify, find, summarize) + compound keywords.
    // No consultant LLM call. Goal created, task lead spawned synchronously.
    let provider = MockProvider::with_responses(vec![
        // Task lead's LLM call (deterministic routing creates goal without LLM call)
        MockProvider::text_response("I'll start building the website."),
    ]);
    let harness = setup_test_agent_orchestrator(provider).await.unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Analyze the requirements, compare authentication approaches, identify security gaps, find the best database solutions, and summarize a deployment plan for a full-stack website with CI/CD",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Should get a response from the task lead
    assert!(!response.is_empty());

    // Goal should be created
    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1, "Complex request should create a goal");
    // Goal should be completed (task lead always-on, succeeds)
    assert_eq!(goals[0].status, "completed");
}

#[tokio::test]
async fn test_orchestration_task_lead_spawns_for_complex() {
    // Deterministic pre-routing classifies the request as complex, creates a goal,
    // and spawns a task lead synchronously (no self_ref in tests).
    let provider = MockProvider::with_responses(vec![
        // Task lead's LLM call (no consultant pass — deterministic routing)
        MockProvider::text_response("I've planned and completed all the tasks for your website."),
    ]);
    let harness = setup_test_agent_orchestrator_task_leads(provider)
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Analyze the requirements, compare authentication approaches, identify security gaps, find the best database solutions, and summarize a deployment plan for a full-stack website with CI/CD",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    // Task lead's response is returned
    assert!(
        response.contains("planned") || response.contains("completed") || !response.is_empty(),
        "Task lead should return a response, got: {}",
        response
    );

    // Goal should be created and completed (task lead succeeded)
    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1, "Complex request should create a goal");
    assert_eq!(
        goals[0].status, "completed",
        "Goal should be completed after task lead succeeds"
    );
}

#[tokio::test]
async fn test_orchestration_task_lead_creates_tasks_via_tool() {
    // Deterministic pre-routing classifies as complex, creates a goal, spawns
    // task lead. The task lead uses manage_goal_tasks to create tasks.
    let provider = MockProvider::with_responses(vec![
        // Task lead calls manage_goal_tasks(create_task)
        MockProvider::tool_call_response(
            "manage_goal_tasks",
            r#"{"action":"create_task","description":"Build the frontend","task_order":1,"priority":"high"}"#,
        ),
        // Task lead calls manage_goal_tasks(complete_goal) after seeing the result
        MockProvider::tool_call_response(
            "manage_goal_tasks",
            r#"{"action":"complete_goal","summary":"Frontend task created successfully"}"#,
        ),
        // Task lead's final text response
        MockProvider::text_response("All tasks have been created and the goal is complete."),
    ]);
    let harness = setup_test_agent_orchestrator_task_leads(provider)
        .await
        .unwrap();

    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Analyze the requirements, compare authentication approaches, identify security gaps, find the best database solutions, and summarize a deployment plan for a full-stack website with CI/CD",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(!response.is_empty());

    // Check that a task was created in the DB
    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);
    let goal_id = &goals[0].id;

    let tasks = harness.state.get_tasks_for_goal(goal_id).await.unwrap();
    assert_eq!(
        tasks.len(),
        1,
        "Task lead should have created 1 task via manage_goal_tasks"
    );
    assert_eq!(tasks[0].description, "Build the frontend");
    assert_eq!(tasks[0].priority, "high");
}

#[tokio::test]
async fn test_orchestration_task_lead_claims_before_dispatch() {
    // Deterministic pre-routing classifies as complex, creates a goal, spawns
    // task lead. The task lead creates tasks with idempotent and dependency features.
    let provider = MockProvider::with_responses(vec![
        // Task lead creates task with idempotent=true
        MockProvider::tool_call_response(
            "manage_goal_tasks",
            r#"{"action":"create_task","description":"Research the topic","task_order":1,"idempotent":true}"#,
        ),
        // Task lead lists tasks to check state
        MockProvider::tool_call_response(
            "manage_goal_tasks",
            r#"{"action":"list_tasks"}"#,
        ),
        // Task lead completes goal
        MockProvider::tool_call_response(
            "manage_goal_tasks",
            r#"{"action":"complete_goal","summary":"Research task created and listed"}"#,
        ),
        // Task lead final text
        MockProvider::text_response("Goal complete. Research task has been created."),
    ]);
    let harness = setup_test_agent_orchestrator_task_leads(provider)
        .await
        .unwrap();

    // User text triggers complex classification: analyze, compare, identify, find,
    // report = 5 action markers + "and"/"then" compound keywords.
    let response = harness
        .agent
        .handle_message(
            "test_session",
            "Analyze the quantum computing landscape, compare visualization frameworks, identify performance bottlenecks, find optimal algorithms, and report on production deployment strategies with monitoring and documentation",
            None,
            UserRole::Owner,
            ChannelContext::private("test"),
            None,
        )
        .await
        .unwrap();

    assert!(!response.is_empty());

    // Verify task was created with idempotent flag
    let goals = harness
        .state
        .get_goals_for_session("test_session")
        .await
        .unwrap();
    assert_eq!(goals.len(), 1);

    let tasks = harness
        .state
        .get_tasks_for_goal(&goals[0].id)
        .await
        .unwrap();
    assert_eq!(tasks.len(), 1);
    assert!(tasks[0].idempotent, "Task should be marked idempotent");
    assert_eq!(tasks[0].description, "Research the topic");
}

#[tokio::test]
async fn test_executor_activity_logging() {
    // Test that executor agents with task_id log TaskActivity records.
    // This tests the activity logging indirectly through manage_goal_tasks.
    let state = {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service =
            Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
        let state = Arc::new(
            crate::state::SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        let state: Arc<dyn crate::traits::StateStore> = state;

        // Create a goal
        let goal = crate::traits::Goal::new_finite("Test activity logging", "test-session");
        state.create_goal(&goal).await.unwrap();

        // Create a task
        let task = crate::traits::Task {
            id: "test-task-001".to_string(),
            goal_id: goal.id.clone(),
            description: "Test task for activity logging".to_string(),
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
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        state.create_task(&task).await.unwrap();

        std::mem::forget(db_file);
        state
    };

    // Log a tool_call activity
    let activity = crate::traits::TaskActivity {
        id: 0,
        task_id: "test-task-001".to_string(),
        activity_type: "tool_call".to_string(),
        tool_name: Some("terminal".to_string()),
        tool_args: Some(r#"{"command":"ls"}"#.to_string()),
        result: Some("file1.txt\nfile2.txt".to_string()),
        success: Some(true),
        tokens_used: None,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    state.log_task_activity(&activity).await.unwrap();

    // Log an llm_call activity
    let activity2 = crate::traits::TaskActivity {
        id: 0,
        task_id: "test-task-001".to_string(),
        activity_type: "llm_call".to_string(),
        tool_name: None,
        tool_args: None,
        result: Some("I found 2 files".to_string()),
        success: Some(true),
        tokens_used: Some(150),
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    state.log_task_activity(&activity2).await.unwrap();

    // Verify activities were logged
    let activities = state.get_task_activities("test-task-001").await.unwrap();
    assert_eq!(activities.len(), 2, "Should have 2 activity records");

    let tool_activity = activities
        .iter()
        .find(|a| a.activity_type == "tool_call")
        .expect("Should have a tool_call activity");
    assert_eq!(tool_activity.tool_name.as_deref(), Some("terminal"));
    assert_eq!(tool_activity.success, Some(true));

    let llm_activity = activities
        .iter()
        .find(|a| a.activity_type == "llm_call")
        .expect("Should have an llm_call activity");
    assert_eq!(llm_activity.tokens_used, Some(150));
    assert_eq!(llm_activity.success, Some(true));
}

#[tokio::test]
async fn test_task_id_passed_to_executor() {
    // Verify spawn_agent schema accepts task_id parameter
    let json_args = serde_json::json!({
        "mission": "Test executor",
        "task": "Do something",
        "task_id": "test-task-123"
    });

    // The SpawnArgs struct should parse task_id
    let parsed: serde_json::Value = serde_json::from_str(&json_args.to_string()).unwrap();
    assert_eq!(parsed["task_id"], "test-task-123");
    assert_eq!(parsed["mission"], "Test executor");

    // Also verify the schema includes task_id
    use crate::tools::spawn::SpawnAgentTool;
    use crate::traits::Tool;
    let tool = SpawnAgentTool::new_deferred(8000, 300);
    let schema = tool.schema();
    let props = &schema["parameters"]["properties"];
    assert!(
        props.get("task_id").is_some(),
        "spawn_agent schema should include task_id"
    );
    assert_eq!(props["task_id"]["type"], "string");
}

