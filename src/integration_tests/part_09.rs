// ======================== Phase 4: Learning Integration Tests ========================

#[tokio::test]
async fn test_orchestration_goal_context_feed_forward() {
    // Verify that when facts exist in the state, a goal created for a complex
    // request gets relevant knowledge injected into its context field.
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::types::FactPrivacy;

    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().to_str().unwrap().to_string();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(&db_path, 100, None, embedding_service)
            .await
            .unwrap(),
    );

    // Pre-populate facts
    state
        .upsert_fact(
            "technical",
            "build_full_stack_website_deploy_aws_target",
            "AWS us-east-1 deployment target for full stack website",
            "manual",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();
    state
        .upsert_fact(
            "project",
            "full_stack_website_framework",
            "Uses React and Node.js for full stack website deploy flow to AWS",
            "manual",
            None,
            FactPrivacy::Global,
        )
        .await
        .unwrap();

    // Create a goal and simulate what the agent does: query facts and inject into context
    let mut goal = Goal::new_finite(
        "Build a full-stack website and deploy to AWS",
        "test-session",
    );

    let relevant_facts = state
        .get_relevant_facts("Build a full-stack website and deploy to AWS", 10)
        .await
        .unwrap_or_default();

    let relevant_procedures = state
        .get_relevant_procedures("Build a full-stack website and deploy to AWS", 5)
        .await
        .unwrap_or_default();

    if !relevant_facts.is_empty() || !relevant_procedures.is_empty() {
        let ctx = serde_json::json!({
            "relevant_facts": relevant_facts.iter().map(|f| {
                serde_json::json!({"category": f.category, "key": f.key, "value": f.value})
            }).collect::<Vec<_>>(),
            "relevant_procedures": relevant_procedures.iter().map(|p| {
                serde_json::json!({"name": p.name, "trigger": p.trigger_pattern, "steps": p.steps})
            }).collect::<Vec<_>>(),
            "task_results": [],
        });
        goal.context = Some(serde_json::to_string(&ctx).unwrap_or_default());
    }

    state.create_goal(&goal).await.unwrap();

    // Verify goal was created with context
    let stored_goal = state.get_goal(&goal.id).await.unwrap().unwrap();
    assert!(
        stored_goal.context.is_some(),
        "Goal should have context with relevant facts"
    );

    let ctx: serde_json::Value =
        serde_json::from_str(stored_goal.context.as_deref().unwrap()).unwrap();

    // Should have the expected structure
    assert!(ctx.get("task_results").is_some());
    assert!(ctx.get("relevant_facts").is_some());

    let facts = ctx["relevant_facts"].as_array().unwrap();
    assert!(
        facts.len() >= 2,
        "Should have at least 2 relevant facts, got {}",
        facts.len()
    );

    // Verify fact structure
    assert!(facts[0].get("category").is_some());
    assert!(facts[0].get("key").is_some());
    assert!(facts[0].get("value").is_some());

    std::mem::forget(db_file);
}

#[tokio::test]
async fn test_orchestration_context_accumulation_end_to_end() {
    // Create a goal and tasks, complete tasks, verify context accumulation
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::tools::manage_goal_tasks::ManageGoalTasksTool;
    use crate::traits::Tool;

    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().to_str().unwrap().to_string();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(&db_path, 100, None, embedding_service)
            .await
            .unwrap(),
    );

    // Create a goal
    let goal = Goal::new_finite("Build and deploy website", "test-session");
    let goal_id = goal.id.clone();
    state.create_goal(&goal).await.unwrap();

    let tool = ManageGoalTasksTool::new(goal_id.clone(), state.clone());

    // Create tasks
    tool.call(
        &serde_json::json!({
            "action": "create_task",
            "description": "Set up database schema",
            "task_order": 1
        })
        .to_string(),
    )
    .await
    .unwrap();

    tool.call(
        &serde_json::json!({
            "action": "create_task",
            "description": "Build API endpoints",
            "task_order": 2
        })
        .to_string(),
    )
    .await
    .unwrap();

    let tasks = state.get_tasks_for_goal(&goal_id).await.unwrap();
    assert_eq!(tasks.len(), 2);

    // Complete first task
    tool.call(
        &serde_json::json!({
            "action": "update_task",
            "task_id": tasks[0].id,
            "status": "completed",
            "result": "Created users, posts, and comments tables in PostgreSQL"
        })
        .to_string(),
    )
    .await
    .unwrap();

    // Verify context after first task
    let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
    let ctx: serde_json::Value = serde_json::from_str(goal.context.as_deref().unwrap()).unwrap();
    let results = ctx["task_results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0]["result_summary"]
        .as_str()
        .unwrap()
        .contains("PostgreSQL"));

    // Complete second task
    tool.call(
        &serde_json::json!({
            "action": "update_task",
            "task_id": tasks[1].id,
            "status": "completed",
            "result": "Built REST API with CRUD for users, posts, comments"
        })
        .to_string(),
    )
    .await
    .unwrap();

    // Verify both results in context
    let goal = state.get_goal(&goal_id).await.unwrap().unwrap();
    let ctx: serde_json::Value = serde_json::from_str(goal.context.as_deref().unwrap()).unwrap();
    let results = ctx["task_results"].as_array().unwrap();
    assert_eq!(results.len(), 2, "Both task results should be accumulated");

    std::mem::forget(db_file);
}

