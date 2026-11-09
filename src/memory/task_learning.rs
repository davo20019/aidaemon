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

/// Extract durable knowledge from a completed task's activity log.
///
/// Reads the task's activity records, builds a prompt asking the LLM to extract
/// facts, procedures, and error solutions, then stores them in the state store.
pub async fn extract_task_knowledge(
    state: Arc<dyn StateStore>,
    store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    event_store: Arc<crate::events::EventStore>,
    llm_runtime: crate::llm_runtime::SharedLlmRuntime,
    model: String,
    task: Task,
) -> anyhow::Result<()> {
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

    let provider = crate::agent::RuntimeProvider::with_accounting(
        llm_runtime,
        state.clone(),
        event_store,
        "task_learning",
    );
    aidaemon_memory::task_learning::TaskLearner::new(Arc::new(provider), model)
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
