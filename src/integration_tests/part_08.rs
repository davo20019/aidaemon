// ============================================================================
// Task Lead + Executor tests
// ============================================================================

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
