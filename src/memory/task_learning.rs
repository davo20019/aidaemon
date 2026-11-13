//! Knowledge extraction from completed tasks.
//!
//! After an executor completes a task, this module reads activity logs and uses
//! an LLM to extract durable knowledge (facts, procedures, error solutions) that
//! can benefit future tasks.

use std::sync::Arc;

use tracing::info;

use crate::traits::{StateStore, Task};

/// Maximum size of activity log text sent to the extraction prompt.
const MAX_ACTIVITY_LOG_CHARS: usize = 4096;

pub use aidaemon_runtime::turn::memory_writer::RunTaint;

/// Extract durable knowledge from a completed task's activity log.
///
/// Reads the task's activity records, builds a prompt asking the LLM to extract
/// facts, procedures, and error solutions, then stores them in the state store.
/// A tainted run teaches nothing: content someone other than the owner wrote
/// must not become durable memory, so no model call is made and nothing is
/// stored.
pub async fn extract_task_knowledge(
    state: Arc<dyn StateStore>,
    store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    side_calls: crate::llm_runtime::gateway::SideCallGateway,
    model: String,
    task: Task,
    taint: RunTaint,
) -> anyhow::Result<()> {
    if taint == RunTaint::Tainted {
        info!(task_id = %task.id, "Skipping task knowledge extraction: the run read untrusted content");
        return Ok(());
    }

    // Derive channel provenance from the parent goal (tasks do not carry session_id).
    let goal = state.get_goal(&task.goal_id).await.ok().flatten();
    let derived_channel_id: Option<String> = goal
        .as_ref()
        .and_then(|g| crate::memory::derive_channel_id_from_session(&g.session_id));
    // A recurring goal re-runs the same brief on every fire, so its "facts"
    // are restatements of its own instructions and run state. Its procedures
    // and error solutions still carry over; owner facts do not.
    let learn_facts = goal.as_ref().is_none_or(|g| g.goal_type != "continuous");

    // Fetch activity log for this task
    let activities = state.get_task_activities(&task.id).await?;
    if activities.is_empty() {
        info!(task_id = %task.id, "No activities to extract knowledge from");
        return Ok(());
    }

    // Build activity log text (truncated)
    let mut activity_log = String::new();
    for act in &activities {
        let line = match act.activity_type.as_str() {
            "tool_call" => {
                let name = act.tool_name.as_deref().unwrap_or("unknown");
                let args_preview = act
                    .tool_args
                    .as_deref()
                    .map(|a| {
                        let end = crate::utils::floor_char_boundary(a, 200);
                        &a[..end]
                    })
                    .unwrap_or("");
                format!("TOOL_CALL: {} args={}\n", name, args_preview)
            }
            "tool_result" => {
                let name = act.tool_name.as_deref().unwrap_or("unknown");
                let success = act.success.unwrap_or(false);
                let result_preview = act
                    .result
                    .as_deref()
                    .map(|r| {
                        let end = crate::utils::floor_char_boundary(r, 300);
                        &r[..end]
                    })
                    .unwrap_or("");
                format!(
                    "TOOL_RESULT: {} success={} result={}\n",
                    name, success, result_preview
                )
            }
            "status_change" => {
                let result_text = act.result.as_deref().unwrap_or("");
                format!("STATUS: {}\n", result_text)
            }
            _ => {
                format!(
                    "{}: {}\n",
                    act.activity_type,
                    act.result.as_deref().unwrap_or("")
                )
            }
        };

        if activity_log.len() + line.len() > MAX_ACTIVITY_LOG_CHARS {
            activity_log.push_str("... (truncated)\n");
            break;
        }
        activity_log.push_str(&line);
    }

    let provider = side_calls.provider(crate::llm_runtime::side_calls::SideCall::TaskLearning);
    aidaemon_memory::task_learning::TaskLearner::new(provider, model)
        .learn(
            store.as_ref(),
            &aidaemon_memory::task_learning::FinishedTask {
                description: &task.description,
                result: task.result.as_deref(),
                activity_log: &activity_log,
                channel_id: derived_channel_id.as_deref(),
                learn_facts,
            },
        )
        .await
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_runtime::gateway::test_support::{
        failed_calls, gateway_with, ok_calls, provider_answering, provider_failing,
    };
    use crate::state::SqliteStateStore;
    use crate::traits::store_prelude::*;

    /// A state store holding one finished task with one activity, and its
    /// memory store.
    async fn stores_with_finished_task(
        task_id: &str,
    ) -> (
        Arc<SqliteStateStore>,
        Arc<dyn aidaemon_memory::stores::MemoryStores>,
    ) {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string();
        // The store keeps the file open; the file outlives the test.
        std::mem::forget(file);
        let embeddings = Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&path, 100, None, embeddings)
                .await
                .unwrap(),
        );
        let goal = crate::traits::Goal::new_finite("Ship the Acme Corp report", "session_1");
        state.create_goal(&goal).await.unwrap();
        let mut finished = task(task_id);
        finished.goal_id = goal.id.clone();
        state.create_task(&finished).await.unwrap();
        state
            .log_task_activity(&crate::traits::TaskActivity {
                id: 0,
                task_id: task_id.to_string(),
                activity_type: "tool_call".to_string(),
                tool_name: Some("terminal".to_string()),
                tool_args: Some("{\"command\":\"make report\"}".to_string()),
                result: Some("report.pdf written".to_string()),
                success: Some(true),
                tokens_used: None,
                created_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();
        let store = state.memory_stores();
        (state, store)
    }

    fn task(id: &str) -> Task {
        let now = chrono::Utc::now().to_rfc3339();
        Task {
            id: id.to_string(),
            goal_id: "goal-1".to_string(),
            description: "Build the Acme Corp report".to_string(),
            status: "completed".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: Some("Report built".to_string()),
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 3,
            created_at: now.clone(),
            started_at: Some(now.clone()),
            completed_at: Some(now),
        }
    }

    /// Facts, procedures and error solutions in the store.
    async fn count_learned(state: &SqliteStateStore) -> i64 {
        let mut learned = 0;
        for table in ["facts", "procedures", "error_solutions"] {
            let (count,): (i64,) =
                sqlx::query_as(sqlx::AssertSqlSafe(format!("SELECT COUNT(*) FROM {table}")))
                    .fetch_one(&state.pool())
                    .await
                    .unwrap();
            learned += count;
        }
        learned
    }

    #[tokio::test]
    async fn a_failed_learning_call_writes_nothing_and_is_recorded() {
        let (gateway, events) = gateway_with(provider_failing("503")).await;
        let (state, store) = stores_with_finished_task("task-1").await;
        let task = state.get_task("task-1").await.unwrap().unwrap();
        let result = extract_task_knowledge(
            state.clone(),
            store,
            gateway,
            "m".into(),
            task,
            RunTaint::Clean,
        )
        .await;
        assert!(result.is_err());
        assert_eq!(count_learned(&state).await, 0);
        assert!(failed_calls(&events, "task_learning").await >= 1);
        assert_eq!(ok_calls(&events, "task_learning").await, 0);
    }

    #[tokio::test]
    async fn a_learning_call_is_recorded_once() {
        let (gateway, events) = gateway_with(provider_answering(
            r#"{"facts": [], "procedures": [], "error_solutions": []}"#,
        ))
        .await;
        let (state, store) = stores_with_finished_task("task-1").await;
        let task = state.get_task("task-1").await.unwrap().unwrap();
        extract_task_knowledge(state, store, gateway, "m".into(), task, RunTaint::Clean)
            .await
            .unwrap();
        assert_eq!(ok_calls(&events, "task_learning").await, 1);
        assert_eq!(failed_calls(&events, "task_learning").await, 0);
    }

    /// An extraction that teaches one fact and one procedure.
    const TEACHES_SOMETHING: &str = r#"{
        "facts": [{"category": "project", "key": "acme_report_build", "value": "The Acme Corp report is built with make report"}],
        "procedures": [{"name": "build the Acme Corp report", "trigger_pattern": "build report", "steps": ["make report"]}],
        "error_solutions": []
    }"#;

    #[tokio::test]
    async fn a_tainted_run_teaches_nothing_and_makes_no_model_call() {
        let (gateway, events) = gateway_with(provider_answering(TEACHES_SOMETHING)).await;
        let (state, store) = stores_with_finished_task("task-1").await;
        let task = state.get_task("task-1").await.unwrap().unwrap();
        extract_task_knowledge(
            state.clone(),
            store,
            gateway,
            "m".into(),
            task,
            RunTaint::Tainted,
        )
        .await
        .unwrap();
        assert_eq!(count_learned(&state).await, 0);
        assert_eq!(ok_calls(&events, "task_learning").await, 0);
        assert_eq!(failed_calls(&events, "task_learning").await, 0);
    }

    #[tokio::test]
    async fn a_clean_run_still_learns() {
        let (gateway, _events) = gateway_with(provider_answering(TEACHES_SOMETHING)).await;
        let (state, store) = stores_with_finished_task("task-1").await;
        let task = state.get_task("task-1").await.unwrap().unwrap();
        extract_task_knowledge(
            state.clone(),
            store,
            gateway,
            "m".into(),
            task,
            RunTaint::Clean,
        )
        .await
        .unwrap();
        assert!(count_learned(&state).await > 0);
    }
}
