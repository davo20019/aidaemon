use super::*;
use serde_json::json;

fn sample_task(status: &str, description: &str, result: Option<&str>, error: Option<&str>) -> Task {
    Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: "goal-1".to_string(),
        description: description.to_string(),
        status: status.to_string(),
        priority: "medium".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: result.map(ToOwned::to_owned),
        error: error.map(ToOwned::to_owned),
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: "2026-03-08T13:00:00Z".to_string(),
        started_at: Some("2026-03-08T13:00:05Z".to_string()),
        completed_at: Some("2026-03-08T13:00:10Z".to_string()),
    }
}

#[test]
fn build_goal_failure_summary_prefers_persisted_summary() {
    let mut goal = Goal::new_finite("Post tweet", "session-1");
    goal.context = Some(json!({ "failure_summary": "Request denied by user" }).to_string());

    let summary = build_goal_failure_summary(Some(&goal), &[], None, None);

    assert_eq!(summary, "Request denied by user");
}

#[test]
fn build_goal_failure_summary_falls_back_to_failed_task_details() {
    let goal = Goal::new_finite("Post tweet", "session-1");
    let tasks = vec![sample_task(
        "failed",
        "Post to Twitter/X",
        Some("Request denied by user"),
        None,
    )];

    let summary = build_goal_failure_summary(Some(&goal), &tasks, None, None);

    assert!(summary.contains("Post to Twitter/X"));
    assert!(summary.contains("Request denied by user"));
}
