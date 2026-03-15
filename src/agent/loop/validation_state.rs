use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::agent::{post_task, LearningContext, TurnContext};
use crate::traits::{Task, ToolTargetHint};

use super::execution_state::{ExecutionState, LinearIntentPlan, OutcomeEntry, TargetScope};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutedAction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationFailure {
    MissingEvidence,
    ContradictoryEvidence,
    VerificationPending,
    BudgetExhausted,
    ApprovalRequired,
    ScopeViolation,
    PlanRejected,
    CritiqueRejected,
    SuccessCriteriaUnmatched,
    NonrecoverableFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoopRepetitionReason {
    MissingEvidence,
    ContradictoryEvidence,
    VerificationPending,
    PlanRejected,
    CritiqueRejected,
    RetryStep,
    ReplanRequired,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_action: Option<ExecutedAction>,
    #[serde(default)]
    pub last_result_had_evidence: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched_success_criteria: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_checks: Vec<ValidationFailure>,
    #[serde(default)]
    pub replan_count: usize,
    #[serde(default)]
    pub retry_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loop_repetition_reason: Option<LoopRepetitionReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_plan_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub active_success_criteria: Vec<String>,
}

impl ValidationState {
    pub fn set_plan(&mut self, plan_version: u32, success_criteria: &[String]) {
        if plan_version > 0 {
            self.active_plan_version = Some(plan_version);
        }
        self.active_success_criteria = success_criteria.to_vec();
        self.refresh_success_criteria_matches("");
    }

    pub fn record_action(
        &mut self,
        tool_name: Option<&str>,
        target: Option<String>,
        last_result_had_evidence: bool,
    ) {
        self.last_action = Some(ExecutedAction {
            tool_name: tool_name.map(str::to_string),
            target,
        });
        self.last_result_had_evidence = last_result_had_evidence;
    }

    pub fn record_failure(&mut self, failure: ValidationFailure) {
        if !self.failed_checks.contains(&failure) {
            self.failed_checks.push(failure);
        }
    }

    pub fn note_retry(&mut self, reason: LoopRepetitionReason) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.loop_repetition_reason = Some(reason);
    }

    pub fn note_replan(&mut self) {
        self.replan_count = self.replan_count.saturating_add(1);
        self.loop_repetition_reason = Some(LoopRepetitionReason::ReplanRequired);
    }

    pub fn note_replan_for(&mut self, reason: LoopRepetitionReason) {
        self.replan_count = self.replan_count.saturating_add(1);
        self.loop_repetition_reason = Some(reason);
    }

    pub fn clear_loop_repetition_reason(&mut self) {
        self.loop_repetition_reason = None;
    }

    pub fn refresh_success_criteria_matches(&mut self, text: &str) {
        let normalized_text = normalize_success_text(text);
        self.matched_success_criteria = self
            .active_success_criteria
            .iter()
            .filter(|criterion| !criterion.trim().is_empty())
            .filter(|criterion| {
                let normalized = normalize_success_text(criterion);
                !normalized.is_empty() && normalized_text.contains(&normalized)
            })
            .cloned()
            .collect();
    }
}

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
}

impl HumanInterventionRequest {
    pub fn render_user_message(&self) -> String {
        let mut lines = Vec::new();
        lines.push("I'm blocked from safely finishing this request.".to_string());

        if let Some(partial_result) = &self.partial_result {
            lines.push(String::new());
            lines.push(format!(
                "Completed work so far: {}",
                partial_result.completed_work_summary
            ));
            if !partial_result.artifacts.is_empty() {
                lines.push(format!(
                    "Relevant target: {}",
                    partial_result.artifacts.join(", ")
                ));
            }
            lines.push(format!("Current blocker: {}", partial_result.blocker));
            if !partial_result.remaining_work.is_empty() {
                lines.push(format!(
                    "Remaining work: {}",
                    partial_result.remaining_work.join("; ")
                ));
            }
        }

        lines.push(String::new());
        lines.push(format!("What I need from you: {}", self.exact_need));
        lines.push(format!("What I will do next: {}", self.next_step));
        if let Some(consequence) = &self.consequence_if_not_provided {
            lines.push(format!("If not provided: {}", consequence));
        }

        lines.join("\n")
    }

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

        if let Some(allowed_tools) = &self.allowed_tools {
            if !allowed_tools.is_empty() {
                lines.push(format!(
                    "- allowed tools for this executor: {}",
                    allowed_tools.join(", ")
                ));
            }
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
    if let Some(task) = task {
        if let Some(existing) = extract_executor_result_context(task.context.as_deref()) {
            return existing;
        }
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
                    "Resolve the blocker and resume with a narrower task.".to_string()
                ],
            }),
        };
    }

    let blocker = task
        .and_then(|task| task.blocker.clone())
        .or_else(|| task_response.clone().filter(|text| mentions_blocker(text)));
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

        let needs_approval =
            mentions_approval(&blocker) || task_response.as_deref().is_some_and(mentions_approval);
        let exact_need = if needs_approval {
            "Explicit approval or permission to perform the blocked action.".to_string()
        } else if let Some(partial_result) = &partial_result {
            format!(
                "Resolve the blocker after '{}'.",
                partial_result.completed_work_summary
            )
        } else {
            "Clarify the blocker or provide the missing dependency.".to_string()
        };
        let next_step = if needs_approval {
            "Resume the blocked action once approval is granted.".to_string()
        } else {
            "Continue the task after the blocker is resolved.".to_string()
        };
        let approval_request = needs_approval.then(|| HumanInterventionRequest {
            outcome: ValidationOutcome::NeedsApproval,
            approval_state: ApprovalState::Required,
            action_requested: blocker.clone(),
            target: artifacts.first().cloned(),
            reason: blocker.clone(),
            exact_need: exact_need.clone(),
            next_step: next_step.clone(),
            consequence_if_not_provided: Some(
                "The task will remain blocked and the executor will stop without taking the gated action."
                    .to_string(),
            ),
            partial_result: partial_result.clone(),
        });

        return ExecutorStepResult {
            task_id: task_id.to_string(),
            step_outcome: if needs_approval {
                StepValidationOutcome::NeedsApproval
            } else if partial_result.is_some() {
                StepValidationOutcome::PartialDoneBlocked
            } else {
                StepValidationOutcome::Blocked
            },
            task_outcome: if needs_approval {
                TaskValidationOutcome::NeedsApproval
            } else if partial_result.is_some() {
                TaskValidationOutcome::PartialDoneBlocked
            } else {
                TaskValidationOutcome::Blocked
            },
            summary,
            artifacts,
            blocker: Some(blocker),
            exact_need: Some(exact_need),
            next_step: Some(next_step),
            approval_request,
            partial_result,
        };
    }

    if task_response
        .as_deref()
        .is_some_and(super::goal_completion_response_indicates_incomplete_work)
    {
        let blocker =
            "The executor reported progress but did not verify the final outcome.".to_string();
        return ExecutorStepResult {
            task_id: task_id.to_string(),
            step_outcome: StepValidationOutcome::VerifyAgain,
            task_outcome: TaskValidationOutcome::PartialDoneBlocked,
            summary,
            artifacts: artifacts.clone(),
            blocker: Some(blocker.clone()),
            exact_need: Some(
                "A fresh verification step before treating the task as complete.".to_string(),
            ),
            next_step: Some(
                "Run the remaining verification or follow-up check, then update the task."
                    .to_string(),
            ),
            approval_request: None,
            partial_result: Some(PartialResult {
                completed_work_summary: task_response
                    .unwrap_or_else(|| "The executor reported partial progress.".to_string()),
                artifacts,
                blocker,
                remaining_work: vec![
                    "Verify the final state before marking the task complete.".to_string()
                ],
            }),
        };
    }

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

fn mentions_blocker(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    super::contains_keyword_as_words(&lower, "blocked")
        || super::contains_keyword_as_words(&lower, "blocker")
        || super::contains_keyword_as_words(&lower, "cannot proceed")
}

fn mentions_approval(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    super::contains_keyword_as_words(&lower, "approval")
        || super::contains_keyword_as_words(&lower, "permission")
        || super::contains_keyword_as_words(&lower, "approve")
}

fn normalize_success_text(text: &str) -> String {
    text.to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
    }
}

pub fn build_partial_done_blocked_request(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    blocker: impl Into<String>,
    exact_need: impl Into<String>,
    next_step: impl Into<String>,
) -> HumanInterventionRequest {
    build_partial_done_blocked_request_with_plan(
        turn_context,
        learning_ctx,
        None,
        blocker,
        exact_need,
        next_step,
    )
}

pub fn build_partial_done_blocked_request_with_plan(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    execution_state: Option<&ExecutionState>,
    blocker: impl Into<String>,
    exact_need: impl Into<String>,
    next_step: impl Into<String>,
) -> HumanInterventionRequest {
    let blocker = blocker.into();
    let partial_result = summarize_partial_result_with_plan(
        turn_context,
        learning_ctx,
        execution_state,
        blocker.clone(),
    );

    HumanInterventionRequest {
        outcome: ValidationOutcome::PartialDoneBlocked,
        approval_state: ApprovalState::NotNeeded,
        action_requested: "provide the missing verification input so I can finish safely"
            .to_string(),
        target: turn_context.completion_contract.primary_target_hint(),
        reason: blocker.clone(),
        exact_need: exact_need.into(),
        next_step: next_step.into(),
        consequence_if_not_provided: Some(
            "I will stop short of claiming success because the outcome is not yet verified."
                .to_string(),
        ),
        partial_result: Some(partial_result),
    }
}

pub fn build_reduce_scope_request(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    blocker: impl Into<String>,
    exact_need: impl Into<String>,
    next_step: impl Into<String>,
) -> HumanInterventionRequest {
    build_reduce_scope_request_with_plan(
        turn_context,
        learning_ctx,
        None,
        blocker,
        exact_need,
        next_step,
    )
}

pub fn build_reduce_scope_request_with_plan(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    execution_state: Option<&ExecutionState>,
    blocker: impl Into<String>,
    exact_need: impl Into<String>,
    next_step: impl Into<String>,
) -> HumanInterventionRequest {
    let blocker = blocker.into();
    HumanInterventionRequest {
        outcome: ValidationOutcome::ReduceScope,
        approval_state: ApprovalState::NotNeeded,
        action_requested: "confirm a narrower scope so I can finish safely".to_string(),
        target: turn_context.completion_contract.primary_target_hint(),
        reason: blocker.clone(),
        exact_need: exact_need.into(),
        next_step: next_step.into(),
        consequence_if_not_provided: Some(
            "I will stop here instead of silently continuing beyond the current safe scope."
                .to_string(),
        ),
        partial_result: Some(summarize_partial_result_with_plan(
            turn_context,
            learning_ctx,
            execution_state,
            blocker,
        )),
    }
}

pub fn build_abandon_request(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    blocker: impl Into<String>,
    exact_need: impl Into<String>,
    next_step: impl Into<String>,
) -> HumanInterventionRequest {
    let blocker = blocker.into();
    HumanInterventionRequest {
        outcome: ValidationOutcome::Abandon,
        approval_state: ApprovalState::NotNeeded,
        action_requested: "replace the current approach with a new plan".to_string(),
        target: turn_context.completion_contract.primary_target_hint(),
        reason: blocker.clone(),
        exact_need: exact_need.into(),
        next_step: next_step.into(),
        consequence_if_not_provided: Some(
            "I will abandon this execution path rather than repeatedly retrying a broken approach."
                .to_string(),
        ),
        partial_result: (!learning_ctx.tool_calls.is_empty())
            .then(|| summarize_partial_result(turn_context, learning_ctx, blocker)),
    }
}

fn summarize_partial_result(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    blocker: String,
) -> PartialResult {
    summarize_partial_result_with_plan(turn_context, learning_ctx, None, blocker)
}

pub(crate) fn summarize_partial_result_with_plan(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    execution_state: Option<&ExecutionState>,
    blocker: String,
) -> PartialResult {
    let completed_work_summary =
        if let Some(plan) = execution_state.and_then(|es| es.active_linear_intent_plan.as_ref()) {
            format_plan_for_blocked_message(plan, execution_state.unwrap())
        } else if let Some(reconciliation) =
            execution_state.and_then(|es| es.build_reconciliation_overview())
        {
            reconciliation.summary
        } else if !learning_ctx.tool_calls.is_empty() {
            let summary = post_task::categorize_tool_calls(&learning_ctx.tool_calls);
            if summary.trim().is_empty() {
                format!(
                    "Completed {} tool action(s).",
                    learning_ctx.tool_calls.len()
                )
            } else {
                summary.trim().to_string()
            }
        } else {
            "No executable work completed yet.".to_string()
        };

    let artifacts = turn_context
        .completion_contract
        .primary_target_hint()
        .into_iter()
        .collect();

    let remaining_work =
        if let Some(plan) = execution_state.and_then(|es| es.active_linear_intent_plan.as_ref()) {
            plan.steps
                .iter()
                .filter(|s| s.step_index > plan.current_step_cursor)
                .map(|s| s.description.clone())
                .take(3)
                .collect()
        } else if turn_context.completion_contract.requires_observation {
            vec!["Run the final verification step and confirm the result.".to_string()]
        } else if learning_ctx.tool_calls.is_empty() {
            vec!["Resume the next concrete step once the blocker is resolved.".to_string()]
        } else {
            vec!["Synthesize the observed result into the final user-facing answer.".to_string()]
        };

    PartialResult {
        completed_work_summary,
        artifacts,
        blocker,
        remaining_work,
    }
}

fn format_plan_for_blocked_message(
    plan: &LinearIntentPlan,
    execution_state: &ExecutionState,
) -> String {
    let has_outcomes = execution_state
        .outcome_ledger
        .iter()
        .any(|e| e.planned_step_id.is_some());

    let mut lines = Vec::new();

    if has_outcomes {
        for step in &plan.steps {
            let outcomes: Vec<&OutcomeEntry> = execution_state
                .outcome_ledger
                .iter()
                .filter(|e| e.planned_step_id.as_deref() == Some(&step.step_id))
                .collect();

            if outcomes.is_empty() {
                if step.step_index <= plan.current_step_cursor {
                    lines.push(format!(
                        "\u{2705} Step {}: {}",
                        step.step_index, step.description
                    ));
                } else {
                    lines.push(format!(
                        "\u{2b1c} Step {}: {}",
                        step.step_index, step.description
                    ));
                }
            } else if let Some(last) = outcomes.last() {
                if last.success {
                    lines.push(format!(
                        "\u{2705} Step {}: {}",
                        step.step_index, step.description
                    ));
                } else {
                    let err = last.error_summary.as_deref().unwrap_or("unknown error");
                    let err = crate::utils::truncate_str(err, 100);
                    let attempts = outcomes.len();
                    lines.push(format!(
                        "\u{274c} Step {}: {} \u{2014} {} ({} attempt{})",
                        step.step_index,
                        step.description,
                        err,
                        attempts,
                        if attempts == 1 { "" } else { "s" }
                    ));
                }
            }
        }
    } else {
        lines.push("Plan:".to_string());
        for step in &plan.steps {
            lines.push(format!("{}. {}", step.step_index, step.description));
        }
        let total = execution_state.outcome_ledger.len();
        let succeeded = execution_state
            .outcome_ledger
            .iter()
            .filter(|e| e.success)
            .count();
        if total > 0 {
            lines.push(format!(
                "\nTool results: {} tool calls completed ({} succeeded, {} failed).",
                total,
                succeeded,
                total - succeeded
            ));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{
        history::CompletionTaskKind, CompletionContract, VerificationTarget, VerificationTargetKind,
    };
    use chrono::Utc;

    #[test]
    fn partial_done_blocked_request_is_specific() {
        let turn_context = TurnContext {
            completion_contract: CompletionContract {
                task_kind: CompletionTaskKind::Diagnose,
                requires_observation: true,
                verification_targets: vec![VerificationTarget {
                    kind: VerificationTargetKind::Url,
                    value: "https://example.com/health".to_string(),
                }],
                ..CompletionContract::default()
            },
            ..TurnContext::default()
        };
        let learning_ctx = LearningContext {
            user_text: "deploy it".to_string(),
            intent_domains: Vec::new(),
            tool_calls: vec![
                "terminal(npm run build)".to_string(),
                "terminal(cloudflare deploy)".to_string(),
            ],
            errors: Vec::new(),
            first_error: None,
            recovery_actions: Vec::new(),
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            replay_notes: Vec::new(),
        };

        let request = build_partial_done_blocked_request(
            &turn_context,
            &learning_ctx,
            "I still need a live verification check.",
            "A fresh read-only verification against the deployed URL.",
            "I will run the final verification check and then confirm the deployment state.",
        );
        let rendered = request.render_user_message();

        assert_eq!(request.outcome, ValidationOutcome::PartialDoneBlocked);
        assert!(rendered.contains("Completed work so far:"));
        assert!(rendered.contains("https://example.com/health"));
        assert!(rendered.contains("What I need from you:"));
        assert!(rendered.contains("What I will do next:"));
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
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("Explicit owner approval")));
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("Next step after approval")));
    }

    #[test]
    fn reduce_scope_summary_for_non_observation_work_does_not_claim_verification() {
        let turn_context = TurnContext {
            completion_contract: CompletionContract {
                task_kind: CompletionTaskKind::Answer,
                requires_observation: false,
                ..CompletionContract::default()
            },
            ..TurnContext::default()
        };
        let learning_ctx = LearningContext {
            user_text: "find trials".to_string(),
            intent_domains: Vec::new(),
            tool_calls: vec![
                "http_request(GET https://clinicaltrials.gov/api/v2/studies)".to_string(),
            ],
            errors: Vec::new(),
            first_error: None,
            recovery_actions: Vec::new(),
            start_time: Utc::now(),
            completed_naturally: false,
            explicit_positive_signals: 0,
            explicit_negative_signals: 0,
            replay_notes: Vec::new(),
        };

        let request = build_reduce_scope_request(
            &turn_context,
            &learning_ctx,
            "Budget exhausted before I could finish the summary.",
            "A narrower target.",
            "I will continue on the narrowed scope.",
        );
        let partial = request.partial_result.expect("expected partial result");
        assert_eq!(
            partial.remaining_work,
            vec!["Synthesize the observed result into the final user-facing answer.".to_string()]
        );
    }
}
