//! Step and task validation outcomes, approval state, and the executor
//! handoff and step-result records kept in a task's context.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use aidaemon_events::traits::ToolTargetHint;
use aidaemon_work::goals::Task;

use crate::turn::execution_state::TargetScope;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationOutcome {
    StepDone,
    VerifyAgain,
    NeedsApproval,
    PartialDoneBlocked,
    ReduceScope,
    Abandon,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalState {
    NotNeeded,
    Required,
    Requested,
    Granted,
    Denied,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartialResult {
    pub completed_work_summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<String>,
    pub blocker: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remaining_work: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HumanInterventionRequest {
    pub outcome: ValidationOutcome,
    pub approval_state: ApprovalState,
    pub action_requested: String,
    pub target: Option<String>,
    pub reason: String,
    pub exact_need: String,
    pub next_step: String,
    pub consequence_if_not_provided: Option<String>,
    pub partial_result: Option<PartialResult>,
    /// False for an agent-side terminal execution failure. Missing evidence is
    /// not automatically a request for the user to manufacture a receipt.
    #[serde(default = "default_user_action_required")]
    pub user_action_required: bool,
}

fn default_user_action_required() -> bool {
    true
}

impl HumanInterventionRequest {
    pub fn to_inline_approval_prompt(&self) -> (String, Vec<String>) {
        let mut description = format!("Approval needed: {}", self.action_requested);
        if let Some(target) = &self.target {
            description.push_str(&format!(" (target: {})", target));
        }

        let mut warnings = vec![self.reason.clone(), self.exact_need.clone()];
        if let Some(partial_result) = &self.partial_result {
            warnings.push(format!(
                "Completed work so far: {}",
                partial_result.completed_work_summary
            ));
        }
        warnings.push(format!("Next step after approval: {}", self.next_step));
        if let Some(consequence) = &self.consequence_if_not_provided {
            warnings.push(consequence.clone());
        }

        (description, warnings)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StepValidationOutcome {
    StepDone,
    RetryStep,
    UseFallbackTool,
    VerifyAgain,
    ReplanTask,
    NeedsApproval,
    PartialDoneBlocked,
    ReduceScope,
    Abandon,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskValidationOutcome {
    TaskDone,
    ContinueWithNextStep,
    VerifyAgain,
    ReplanTask,
    NeedsApproval,
    PartialDoneBlocked,
    ReduceScope,
    Abandon,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutorHandoff {
    pub task_id: String,
    pub mission: String,
    pub task_description: String,
    pub target_scope: TargetScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_targets: Vec<ToolTargetHint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
}

impl ExecutorHandoff {
    pub fn render_prompt_section(&self) -> String {
        let mut lines = vec![
            "## Task Contract".to_string(),
            format!("- task_id: {}", self.task_id),
        ];

        if !self.target_scope.allowed_targets.is_empty() {
            let allowed = self
                .target_scope
                .allowed_targets
                .iter()
                .map(|target| target.value.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("- allowed targets (hard boundary): {allowed}"));
        }

        if !self.expected_targets.is_empty() {
            let expected = self
                .expected_targets
                .iter()
                .map(|target| target.value.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("- expected targets (execution hint): {expected}"));
        }

        if let Some(allowed_tools) = &self.allowed_tools
            && !allowed_tools.is_empty()
        {
            lines.push(format!(
                "- allowed tools for this executor: {}",
                allowed_tools.join(", ")
            ));
        }

        lines.push(
            "- If finishing requires acting outside the allowed targets, stop and use report_blocker."
                .to_string(),
        );
        lines.push(
            "- You may complete this single task only. Do not claim the overall goal is done."
                .to_string(),
        );

        lines.join("\n")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutorStepResult {
    pub task_id: String,
    pub step_outcome: StepValidationOutcome,
    pub task_outcome: TaskValidationOutcome,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocker: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_need: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_step: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval_request: Option<HumanInterventionRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_result: Option<PartialResult>,
}

impl ExecutorStepResult {
    pub fn render_task_lead_summary(&self) -> String {
        let mut lines = vec![
            format!("Executor outcome: {:?}", self.task_outcome)
                .replace("TaskDone", "task_done")
                .replace("ContinueWithNextStep", "continue_with_next_step")
                .replace("VerifyAgain", "verify_again")
                .replace("ReplanTask", "replan_task")
                .replace("NeedsApproval", "needs_approval")
                .replace("PartialDoneBlocked", "partial_done_blocked")
                .replace("ReduceScope", "reduce_scope")
                .replace("Abandon", "abandon")
                .replace("Blocked", "blocked"),
            format!("Summary: {}", self.summary),
        ];

        if !self.artifacts.is_empty() {
            lines.push(format!("Artifacts: {}", self.artifacts.join(", ")));
        }
        if let Some(blocker) = &self.blocker {
            lines.push(format!("Blocker: {}", blocker));
        }
        if let Some(exact_need) = &self.exact_need {
            lines.push(format!("Need: {}", exact_need));
        }
        if let Some(next_step) = &self.next_step {
            lines.push(format!("Next step: {}", next_step));
        }
        if let Some(partial) = &self.partial_result {
            lines.push(format!(
                "Completed work so far: {}",
                partial.completed_work_summary
            ));
        }

        lines.join("\n")
    }
}

pub fn extract_executor_handoff_context(context: Option<&str>) -> Option<ExecutorHandoff> {
    context
        .and_then(parse_context_object)
        .and_then(|context| context.get("executor_handoff").cloned())
        .and_then(|value| serde_json::from_value(value).ok())
}

pub fn extract_executor_result_context(context: Option<&str>) -> Option<ExecutorStepResult> {
    context
        .and_then(parse_context_object)
        .and_then(|context| context.get("executor_result").cloned())
        .and_then(|value| serde_json::from_value(value).ok())
}

pub fn persist_executor_handoff_context(
    existing_context: Option<&str>,
    handoff: &ExecutorHandoff,
) -> Result<String, serde_json::Error> {
    merge_context_entry(
        existing_context,
        "executor_handoff",
        serde_json::to_value(handoff)?,
    )
}

pub fn persist_executor_result_context(
    existing_context: Option<&str>,
    result: &ExecutorStepResult,
) -> Result<String, serde_json::Error> {
    merge_context_entry(
        existing_context,
        "executor_result",
        serde_json::to_value(result)?,
    )
}

pub fn derive_executor_step_result(
    task_id: &str,
    task: Option<&Task>,
    response: Option<&str>,
    error: Option<&str>,
) -> ExecutorStepResult {
    if let Some(task) = task
        && let Some(existing) = extract_executor_result_context(task.context.as_deref())
    {
        return existing;
    }

    let handoff = task.and_then(|task| extract_executor_handoff_context(task.context.as_deref()));
    let artifacts = handoff
        .as_ref()
        .map(|handoff| {
            handoff
                .expected_targets
                .iter()
                .map(|target| target.value.clone())
                .collect::<Vec<_>>()
        })
        .filter(|targets| !targets.is_empty())
        .unwrap_or_default();

    let task_response = response
        .map(str::trim)
        .filter(|response| !response.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            task.and_then(|task| {
                task.result
                    .as_deref()
                    .map(str::trim)
                    .filter(|result| !result.is_empty())
                    .map(ToOwned::to_owned)
            })
        });

    if let Some(error) = error {
        let partial_summary = task
            .and_then(|task| task.result.clone())
            .filter(|result| !result.trim().is_empty());
        return ExecutorStepResult {
            task_id: task_id.to_string(),
            step_outcome: if partial_summary.is_some() {
                StepValidationOutcome::PartialDoneBlocked
            } else {
                StepValidationOutcome::Abandon
            },
            task_outcome: if partial_summary.is_some() {
                TaskValidationOutcome::PartialDoneBlocked
            } else {
                TaskValidationOutcome::Abandon
            },
            summary: format!("Executor run failed: {error}"),
            artifacts,
            blocker: Some(error.to_string()),
            exact_need: Some(if partial_summary.is_some() {
                "Resolve the blocker or narrow the task before retrying.".to_string()
            } else {
                "A new plan or a narrower replacement task before trying again.".to_string()
            }),
            next_step: Some(if partial_summary.is_some() {
                "Resume from the partial progress with a narrower follow-up step.".to_string()
            } else {
                "Abandon this execution path and create a fresh replacement task.".to_string()
            }),
            approval_request: None,
            partial_result: partial_summary.map(|completed_work_summary| PartialResult {
                completed_work_summary,
                artifacts: Vec::new(),
                blocker: error.to_string(),
                remaining_work: vec![
                    "Resolve the blocker and resume with a narrower task.".to_string(),
                ],
            }),
        };
    }

    // A blocker is a durable protocol outcome, not a keyword found in otherwise
    // successful executor prose.  Executors can legitimately describe work on
    // "blocked tasks" or say that a fix "unblocked" a queue.  Inferring state
    // from those words turned verified success into a blocked task and could
    // schedule a duplicate retry after an external side effect.  Explicit
    // `report_blocker` state (or the `error` branch above) remains authoritative.
    let blocker = task.and_then(|task| task.blocker.clone());
    let summary = task_response
        .clone()
        .or_else(|| blocker.clone())
        .unwrap_or_else(|| "Executor finished without a detailed summary.".to_string());

    if let Some(blocker) = blocker {
        let partial_summary = task
            .and_then(|task| task.result.clone())
            .filter(|result| !result.trim().is_empty());
        let partial_result = partial_summary
            .as_ref()
            .map(|partial_summary| PartialResult {
                completed_work_summary: partial_summary.clone(),
                artifacts: artifacts.clone(),
                blocker: blocker.clone(),
                remaining_work: vec!["Resolve the blocker and resume the task.".to_string()],
            });

        // Approval is a typed executor outcome persisted by `report_blocker`.
        // A legacy/free-form blocker cannot be upgraded to an approval request
        // by words in its summary: that would let prose manufacture authority
        // and change retry behavior. Treat it as the blocker state it actually
        // is; callers that need approval must persist `executor_result`.
        let exact_need = if let Some(partial_result) = &partial_result {
            format!(
                "Resolve the blocker after '{}'.",
                partial_result.completed_work_summary
            )
        } else {
            "Clarify the blocker or provide the missing dependency.".to_string()
        };
        let next_step = "Continue the task after the blocker is resolved.".to_string();

        return ExecutorStepResult {
            task_id: task_id.to_string(),
            step_outcome: if partial_result.is_some() {
                StepValidationOutcome::PartialDoneBlocked
            } else {
                StepValidationOutcome::Blocked
            },
            task_outcome: if partial_result.is_some() {
                TaskValidationOutcome::PartialDoneBlocked
            } else {
                TaskValidationOutcome::Blocked
            },
            summary,
            artifacts,
            blocker: Some(blocker),
            exact_need: Some(exact_need),
            next_step: Some(next_step),
            approval_request: None,
            partial_result,
        };
    }

    // Reaching this branch means the invocation completed without a typed
    // error, blocker, or persisted executor result. The response is a summary,
    // not a second protocol: words such as "partial" or "not verified" must
    // not overturn the structured invocation outcome.
    ExecutorStepResult {
        task_id: task_id.to_string(),
        step_outcome: StepValidationOutcome::StepDone,
        task_outcome: TaskValidationOutcome::TaskDone,
        summary,
        artifacts,
        blocker: None,
        exact_need: None,
        next_step: None,
        approval_request: None,
        partial_result: None,
    }
}

fn parse_context_object(existing_context: &str) -> Option<Value> {
    let parsed = serde_json::from_str::<Value>(existing_context).ok()?;
    parsed.is_object().then_some(parsed)
}

fn merge_context_entry(
    existing_context: Option<&str>,
    key: &str,
    value: Value,
) -> Result<String, serde_json::Error> {
    let mut context = existing_context
        .and_then(parse_context_object)
        .unwrap_or_else(|| {
            existing_context
                .filter(|context| !context.trim().is_empty())
                .map(|context| json!({ "prior_context_raw": context }))
                .unwrap_or_else(|| json!({}))
        });
    if let Some(object) = context.as_object_mut() {
        object.insert(key.to_string(), value);
    }
    serde_json::to_string(&context)
}

pub fn build_needs_approval_request(
    action_requested: impl Into<String>,
    target: Option<String>,
    reason: impl Into<String>,
    exact_need: impl Into<String>,
    next_step: impl Into<String>,
    partial_result: Option<PartialResult>,
) -> HumanInterventionRequest {
    HumanInterventionRequest {
        outcome: ValidationOutcome::NeedsApproval,
        approval_state: ApprovalState::Required,
        action_requested: action_requested.into(),
        target,
        reason: reason.into(),
        exact_need: exact_need.into(),
        next_step: next_step.into(),
        consequence_if_not_provided: Some(
            "I will stop here and return the current best progress without taking the requested action."
                .to_string(),
        ),
        partial_result,
        user_action_required: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::Utc;

    fn executor_task(blocker: Option<&str>) -> Task {
        Task {
            id: "executor-task".to_string(),
            goal_id: "goal".to_string(),
            description: "Complete one bounded step".to_string(),
            status: "running".to_string(),
            priority: "medium".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: blocker.map(ToOwned::to_owned),
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: Utc::now().to_rfc3339(),
            started_at: Some(Utc::now().to_rfc3339()),
            completed_at: None,
        }
    }

    #[test]
    fn successful_executor_prose_about_blocked_tasks_is_not_a_blocker() {
        let task = executor_task(None);
        let response = "I fixed the scheduler: old blocked tasks no longer prevent new runs. The deployment and read-back verification both succeeded.";

        let result = derive_executor_step_result(&task.id, Some(&task), Some(response), None);

        assert_eq!(result.step_outcome, StepValidationOutcome::StepDone);
        assert_eq!(result.task_outcome, TaskValidationOutcome::TaskDone);
        assert!(result.blocker.is_none());
    }

    #[test]
    fn legacy_blocker_prose_cannot_manufacture_approval_state() {
        let task = executor_task(Some("Production approval is required."));

        let result = derive_executor_step_result(
            &task.id,
            Some(&task),
            Some("I completed the safe preflight work."),
            None,
        );

        assert_eq!(result.step_outcome, StepValidationOutcome::Blocked);
        assert_eq!(result.task_outcome, TaskValidationOutcome::Blocked);
        assert!(result.approval_request.is_none());
        assert_eq!(
            result.blocker.as_deref(),
            Some("Production approval is required.")
        );
    }

    #[test]
    fn needs_approval_request_formats_inline_prompt() {
        let request = build_needs_approval_request(
            "extend the task budget from 10_000 to 20_000 tokens",
            Some("task budget".to_string()),
            "The current task budget is exhausted.",
            "Explicit owner approval to continue spending tokens for this run.",
            "Continue the current task inside the extended budget.",
            None,
        );
        let (description, warnings) = request.to_inline_approval_prompt();

        assert_eq!(request.outcome, ValidationOutcome::NeedsApproval);
        assert_eq!(request.approval_state, ApprovalState::Required);
        assert!(description.contains("Approval needed: extend the task budget"));
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("Explicit owner approval"))
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("Next step after approval"))
        );
    }
}
