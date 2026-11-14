//! Resuming interrupted work: the checkpoint an interrupted task leaves in
//! its events, the prompt section that tells the model where it stopped, the
//! words that ask to resume, and closing the interrupted task as partial.

use aidaemon_events::traits::ToolOutcomeStatus;
use std::collections::HashSet;

use chrono::Utc;

use aidaemon_events::{
    AssistantResponseData, DecisionPointData, DecisionType, ErrorData, EventType, TaskEndData,
    TaskStartData, TaskStatus, ThinkingStartData, ToolReceiptV1, ToolResultData,
};
use tracing::warn;

use crate::host::TurnHost;
use crate::turn::execution_state::StepExecutionOutcome;
use crate::turn::helpers::truncate_for_resume;

#[derive(Debug, Clone)]
pub struct ResumeCheckpoint {
    pub task_id: String,
    pub description: String,
    pub original_user_message: Option<String>,
    pub elapsed_secs: u64,
    pub last_iteration: u32,
    pub tool_results_count: u32,
    pub pending_tool_call_ids: Vec<String>,
    pub last_assistant_summary: Option<String>,
    pub last_tool_summary: Option<String>,
    pub last_error: Option<String>,
    pub execution_snapshot: Option<ResumeExecutionSnapshot>,
    /// The interrupted task's ORIGINAL turn_id (from its TaskStart event).
    /// Recovery TaskEnd MUST use this, never the new resume turn. None for
    /// legacy tasks whose TaskStart predates turn_id persistence.
    pub turn_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResumeExecutionSnapshot {
    pub execution_id: String,
    pub current_step_id: Option<String>,
    pub current_tool: Option<String>,
    pub current_target: Option<String>,
    pub last_outcome: Option<StepExecutionOutcome>,
    pub background_handoff_active: bool,
    pub idempotency_key: Option<String>,
}

impl ResumeCheckpoint {
    pub fn render_prompt_section(&self) -> String {
        let mut lines = vec![
            "## Resume Checkpoint".to_string(),
            "The user explicitly asked to continue prior in-progress work. Resume from this checkpoint and avoid restarting completed steps."
                .to_string(),
            format!("- Previous task_id: {}", self.task_id),
            format!("- Original task: {}", self.description),
            format!("- Elapsed before interruption: {}s", self.elapsed_secs),
            format!("- Last completed iteration: {}", self.last_iteration),
            format!("- Completed tool results: {}", self.tool_results_count),
            format!(
                "- Pending unresolved tool calls: {}",
                self.pending_tool_call_ids.len()
            ),
        ];

        if !self.pending_tool_call_ids.is_empty() {
            let pending = self
                .pending_tool_call_ids
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("- Pending tool call IDs: {}", pending));
        }
        if let Some(msg) = &self.original_user_message {
            lines.push(format!(
                "- Original user request: {}",
                truncate_for_resume(msg, 180)
            ));
        }
        if let Some(summary) = &self.last_assistant_summary {
            lines.push(format!("- Last assistant output: {}", summary));
        }
        if let Some(summary) = &self.last_tool_summary {
            lines.push(format!("- Last tool result: {}", summary));
        }
        if let Some(err) = &self.last_error {
            lines.push(format!("- Last error: {}", err));
        }
        if let Some(snapshot) = &self.execution_snapshot {
            lines.push(format!("- Execution id: {}", snapshot.execution_id));
            if let Some(step_id) = &snapshot.current_step_id {
                lines.push(format!("- Last execution step: {}", step_id));
            }
            if let Some(tool) = &snapshot.current_tool {
                lines.push(format!("- Last execution tool: {}", tool));
            }
            if let Some(target) = &snapshot.current_target {
                lines.push(format!("- Last execution target: {}", target));
            }
            if let Some(outcome) = snapshot.last_outcome {
                lines.push(format!("- Last execution outcome: {:?}", outcome));
            }
            if snapshot.background_handoff_active {
                lines.push("- Background execution was active before interruption.".to_string());
            }
            if let Some(key) = &snapshot.idempotency_key {
                lines.push(format!(
                    "- Replay/idempotency key: {}",
                    truncate_for_resume(key, 120)
                ));
            }
        }
        lines.push(
            "Resume from the next concrete step immediately. Re-run tools only if needed to verify or recover."
                .to_string(),
        );
        lines.join("\n")
    }
}

pub fn normalize_for_resume_intent(text: &str) -> String {
    text.split_whitespace()
        .map(|part| part.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn is_resume_request(text: &str) -> bool {
    let normalized = normalize_for_resume_intent(text);
    if normalized.is_empty() {
        return false;
    }

    const EXACT: &[&str] = &[
        "continue",
        "resume",
        "keep going",
        "go on",
        "carry on",
        "next phase",
        "next step",
    ];
    if EXACT.contains(&normalized.as_str()) {
        return true;
    }

    normalized.starts_with("continue ")
        || normalized.starts_with("resume ")
        || normalized.starts_with("keep going ")
        || normalized.starts_with("go on ")
        || normalized.starts_with("carry on ")
        || normalized.starts_with("next phase ")
        || normalized.starts_with("next step ")
}

pub async fn build_resume_checkpoint(
    host: &dyn TurnHost,
    session_id: &str,
) -> anyhow::Result<Option<ResumeCheckpoint>> {
    let Some(active_task_event) = host.events().get_active_task(session_id).await? else {
        return Ok(None);
    };
    let Some(task_id) = active_task_event.task_id.clone() else {
        return Ok(None);
    };

    let start_data = active_task_event.parse_data::<TaskStartData>().ok();
    let description = start_data
        .as_ref()
        .map(|d| d.description.clone())
        .unwrap_or_else(|| "in-progress task".to_string());
    let original_user_message = start_data.and_then(|d| d.user_message);

    let events = host
        .events()
        .query_task_events_for_session(session_id, &task_id)
        .await
        .unwrap_or_default();

    let mut last_iteration: u32 = 0;
    let mut tool_results_count: u32 = 0;
    let mut pending_tool_calls: HashSet<String> = HashSet::new();
    let mut last_assistant_summary: Option<String> = None;
    let mut last_tool_summary: Option<String> = None;
    let mut last_error: Option<String> = None;
    let mut execution_snapshot: Option<ResumeExecutionSnapshot> = None;
    let mut last_typed_receipt: Option<(String, String, ToolReceiptV1)> = None;

    for event in &events {
        match event.event_type {
            EventType::ThinkingStart => {
                if let Ok(data) = event.parse_data::<ThinkingStartData>() {
                    last_iteration = last_iteration.max(data.iteration);
                }
            }
            EventType::AssistantResponse => {
                if let Ok(data) = event.parse_data::<AssistantResponseData>() {
                    if let Some(calls) = data.tool_calls.as_ref() {
                        for call in calls {
                            pending_tool_calls.insert(call.id.clone());
                        }
                    }
                    if let Some(content) = data.content.as_deref() {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() {
                            last_assistant_summary = Some(truncate_for_resume(trimmed, 180));
                        }
                    }
                }
            }
            EventType::ToolResult => {
                if let Ok(data) = event.parse_data::<ToolResultData>() {
                    tool_results_count = tool_results_count.saturating_add(1);
                    pending_tool_calls.remove(&data.tool_call_id);
                    let detail = if !data.failed() {
                        data.result
                    } else {
                        data.error.unwrap_or(data.result)
                    };
                    let detail = detail.trim();
                    if !detail.is_empty() {
                        last_tool_summary = Some(truncate_for_resume(detail, 180));
                    }
                    if let Some(receipt) = data.receipt {
                        last_typed_receipt = Some((data.tool_call_id, data.name, receipt));
                    }
                }
            }
            EventType::Error => {
                if let Ok(data) = event.parse_data::<ErrorData>() {
                    last_error = Some(truncate_for_resume(&data.message, 180));
                }
            }
            EventType::DecisionPoint => {
                let Ok(data) = event.parse_data::<DecisionPointData>() else {
                    continue;
                };
                if data.decision_type != DecisionType::ExecutionStateSnapshot {
                    continue;
                }
                let Some(state) = data
                    .metadata
                    .get("execution_state")
                    .and_then(|value| value.as_object())
                else {
                    continue;
                };
                let execution_id = state
                    .get("execution_id")
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let current_step = state
                    .get("current_step")
                    .and_then(|value| value.as_object());
                let last_outcome = state
                    .get("last_outcome")
                    .cloned()
                    .and_then(|value| serde_json::from_value::<StepExecutionOutcome>(value).ok());
                let background_handoff_active = state
                    .get("background_handoff_active")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let current_step_id = current_step
                    .and_then(|step| step.get("step_id"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let current_tool = current_step
                    .and_then(|step| step.get("primary_tool"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let current_target = current_step
                    .and_then(|step| step.get("expected_targets"))
                    .and_then(|value| value.as_array())
                    .and_then(|targets| targets.first())
                    .and_then(|target| target.get("value"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .or_else(|| {
                        current_step
                            .and_then(|step| step.get("target_scope"))
                            .and_then(|value| value.get("allowed_targets"))
                            .and_then(|value| value.as_array())
                            .and_then(|targets| targets.first())
                            .and_then(|target| target.get("value"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string)
                    });
                let idempotency_key = current_step
                    .and_then(|step| step.get("idempotency_key"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let Some(execution_id) = execution_id else {
                    continue;
                };
                execution_snapshot = Some(ResumeExecutionSnapshot {
                    execution_id,
                    current_step_id,
                    current_tool,
                    current_target,
                    last_outcome,
                    background_handoff_active,
                    idempotency_key,
                });
            }
            _ => {}
        }
    }

    // The versioned receipt is the producer-owned execution fact. Snapshot
    // JSON remains a compatibility/reconciliation view, but it must not
    // override a later typed result when recovering after a crash.
    if let (Some(snapshot), Some((tool_call_id, tool_name, receipt))) =
        (&mut execution_snapshot, last_typed_receipt)
    {
        let call_matches_step = snapshot
            .current_step_id
            .as_deref()
            .is_some_and(|step_id| step_id.ends_with(&format!("-{tool_call_id}")));
        let key_matches_step = receipt.idempotency_key.is_some()
            && receipt.idempotency_key == snapshot.idempotency_key;
        let receipt_matches_step = snapshot.current_tool.as_deref() == Some(tool_name.as_str())
            && (call_matches_step || key_matches_step);
        if receipt_matches_step {
            snapshot.current_tool.get_or_insert(tool_name);
            if receipt.idempotency_key.is_some() {
                snapshot.idempotency_key = receipt.idempotency_key.clone();
            }
            snapshot.background_handoff_active =
                receipt.outcome_status == ToolOutcomeStatus::Backgrounded;
            snapshot.last_outcome = Some(match receipt.outcome_status {
                ToolOutcomeStatus::Succeeded => StepExecutionOutcome::Progress,
                ToolOutcomeStatus::CompletedWithNegativeResult => StepExecutionOutcome::NoProgress,
                ToolOutcomeStatus::FailedRetryable => StepExecutionOutcome::RecoverableFailure,
                ToolOutcomeStatus::FailedPermanent | ToolOutcomeStatus::Blocked => {
                    StepExecutionOutcome::NonrecoverableFailure
                }
                ToolOutcomeStatus::Backgrounded => StepExecutionOutcome::BackgroundDetached,
            });
        }
    }

    let elapsed_secs = (Utc::now() - active_task_event.created_at)
        .num_seconds()
        .max(0) as u64;
    let mut pending_tool_call_ids: Vec<String> = pending_tool_calls.into_iter().collect();
    pending_tool_call_ids.sort();

    Ok(Some(ResumeCheckpoint {
        task_id,
        description,
        original_user_message,
        elapsed_secs,
        last_iteration,
        tool_results_count,
        pending_tool_call_ids,
        last_assistant_summary,
        last_tool_summary,
        last_error,
        execution_snapshot,
        // Carry the interrupted task's original turn_id (legacy = None) so the
        // recovery TaskEnd attributes to the correct turn, not the resume turn.
        turn_id: active_task_event.turn_id.clone(),
    }))
}

pub async fn mark_task_interrupted_for_resume(
    host: &dyn TurnHost,
    session_id: &str,
    checkpoint: &ResumeCheckpoint,
    resumed_task_id: &str,
) {
    // Best-effort: if task already has task_end, skip.
    let already_ended = host
        .events()
        .query_task_events_for_session(session_id, &checkpoint.task_id)
        .await
        .ok()
        .is_some_and(|events| events.iter().any(|e| e.event_type == EventType::TaskEnd));
    if already_ended {
        return;
    }

    let resume_emitter =
        aidaemon_events::EventEmitter::new(host.event_log(), session_id.to_string())
            .with_task_id(checkpoint.task_id.clone());
    let summary = format!(
        "Agent process interrupted before completion. Resumed in task {}.",
        resumed_task_id
    );
    let _ = resume_emitter
        .emit(
            EventType::TaskEnd,
            TaskEndData {
                task_id: checkpoint.task_id.clone(),
                status: TaskStatus::Interrupted,
                outcome: Some(aidaemon_events::TaskOutcome::Partial),
                duration_secs: checkpoint.elapsed_secs,
                iterations: checkpoint.last_iteration,
                tool_calls_count: checkpoint.tool_results_count,
                error: None,
                summary: Some(summary),
                efficiency: None,
                // Recovery MUST use the checkpoint's original turn_id, never
                // current_turn_ids (which points at the new resume turn).
                // Legacy (TaskStart.turn_id == None) => None here.
                turn_id: checkpoint.turn_id.clone(),
                completion_proof: None,
                harness_eval: None,
            },
        )
        .await;
    if let Err(err) = crate::turn::dialogue_state::record_dialogue_task_end(
        host,
        session_id,
        &checkpoint.task_id,
        TaskStatus::Interrupted,
        aidaemon_events::TaskOutcome::Partial,
    )
    .await
    {
        warn!(
            session_id,
            task_id = %checkpoint.task_id,
            error = %err,
            "Failed to record dialogue task end for stale checkpoint"
        );
    }
    crate::turn::task_end::run_task_end_tool_hooks(host, &checkpoint.task_id, session_id).await;
}
