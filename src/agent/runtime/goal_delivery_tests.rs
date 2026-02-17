use super::*;
use crate::traits::Task;

fn completed_task(
    id: &str,
    description: &str,
    result: &str,
    completed_at: &str,
    task_order: i32,
) -> Task {
    Task {
        id: id.to_string(),
        goal_id: "goal_123".to_string(),
        description: description.to_string(),
        status: "completed".to_string(),
        priority: "medium".to_string(),
        task_order,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: Some(result.to_string()),
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: completed_at.to_string(),
        started_at: None,
        completed_at: Some(completed_at.to_string()),
    }
}

#[test]
fn low_signal_task_lead_reply_detects_synth_done() {
    assert!(is_low_signal_task_lead_reply("Done."));
    assert!(is_low_signal_task_lead_reply(
        "Done — Check the disk usage of my projects directory..."
    ));
}

#[test]
fn low_signal_task_lead_reply_detects_brief_complete_goal_echo() {
    assert!(is_low_signal_task_lead_reply(
        "Goal goal_123 completed: Goal completed successfully"
    ));
}

#[test]
fn low_signal_task_lead_reply_allows_concrete_multiline_results() {
    let concrete = "Goal goal_123 completed: Finished disk usage audit\n\nFinal task result:\n1.2G /Users/me/projects/aidaemon/target";
    assert!(!is_low_signal_task_lead_reply(concrete));
}

#[test]
fn goal_task_results_summary_includes_recent_tasks_and_omission_count() {
    let tasks = vec![
        completed_task("task_1", "Step 1", "Result one", "2026-02-17T09:00:00Z", 1),
        completed_task("task_2", "Step 2", "Result two", "2026-02-17T09:01:00Z", 2),
        completed_task(
            "task_3",
            "Step 3",
            "Result three",
            "2026-02-17T09:02:00Z",
            3,
        ),
        completed_task("task_4", "Step 4", "Result four", "2026-02-17T09:03:00Z", 4),
    ];

    let summary = build_goal_task_results_summary(&tasks, "fallback");
    assert!(summary.contains("4/4 tasks completed."));
    assert!(summary.contains("Step 4"));
    assert!(summary.contains("Step 3"));
    assert!(summary.contains("Step 2"));
    assert!(summary.contains("(+1 earlier completed task result omitted)"));
}

#[test]
fn goal_task_results_summary_returns_fallback_when_no_results() {
    let mut task = completed_task("task_1", "Step 1", " ", "2026-02-17T09:00:00Z", 1);
    task.result = Some("   ".to_string());
    let summary = build_goal_task_results_summary(&[task], "fallback summary");
    assert_eq!(summary, "fallback summary");
}
