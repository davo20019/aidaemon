use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::info;

use crate::tools::traits::{
    HandoffArtifact, ReportBlockerStore, TaskAttempt, TaskAttemptPatch, TaskHandoff, Tool,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolRole,
};
use crate::turn::validation_state::{
    ExecutorStepResult, PartialResult, StepValidationOutcome, TaskValidationOutcome,
    build_needs_approval_request, extract_executor_handoff_context,
    persist_executor_result_context,
};

/// Tool for executors to report they are blocked and cannot proceed.
///
pub struct ReportBlockerTool {
    task_id: String,
    state: Arc<dyn ReportBlockerStore>,
    attempt: Option<TaskAttempt>,
    mandate_execution: bool,
}

impl ReportBlockerTool {
    pub fn new(task_id: String, state: Arc<dyn ReportBlockerStore>) -> Self {
        Self {
            task_id,
            state,
            attempt: None,
            mandate_execution: false,
        }
    }

    pub fn for_attempt(
        task_id: String,
        state: Arc<dyn ReportBlockerStore>,
        attempt: TaskAttempt,
        mandate_execution: bool,
    ) -> Self {
        Self {
            task_id,
            state,
            attempt: Some(attempt),
            mandate_execution,
        }
    }

    /// Prove that a requested local repair is both inside the executor's
    /// durable target scope and causally linked to an observed failed tool
    /// result. This deliberately uses exact paths and typed task state rather
    /// than interpreting words such as "pre-existing" or "unrelated".
    async fn causal_local_repair_is_proven(&self, args: &ReportBlockerArgs) -> bool {
        if args.dependency_repair != Some(true) {
            return false;
        }
        let Some(target) = args.target.as_deref().map(str::trim) else {
            return false;
        };
        let Ok(canonical_target) = std::fs::canonicalize(target) else {
            return false;
        };
        let Ok(Some(task)) = self.state.get_task(&self.task_id).await else {
            return false;
        };
        let Some(handoff) = extract_executor_handoff_context(task.context.as_deref()) else {
            return false;
        };
        let target_is_allowed = handoff
            .target_scope
            .allowed_targets
            .iter()
            .filter(|allowed| {
                matches!(
                    allowed.kind,
                    crate::tools::traits::ToolTargetHintKind::Path
                        | crate::tools::traits::ToolTargetHintKind::ProjectScope
                )
            })
            .filter_map(|allowed| std::fs::canonicalize(&allowed.value).ok())
            .any(|allowed| canonical_target == allowed || canonical_target.starts_with(allowed));
        if !target_is_allowed {
            return false;
        }

        self.state
            .get_task_activities(&self.task_id)
            .await
            .unwrap_or_default()
            .iter()
            .any(|activity| {
                activity.success == Some(false)
                    && activity
                        .result
                        .as_deref()
                        .is_some_and(|result| result.contains(target))
            })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BlockerClass {
    OwnerInput,
    MissingAuthority,
    ExternalDependency,
    AmbiguousExternalEffect,
    SafetyBoundary,
    RecoveryExhausted,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ExternalEffectState {
    None,
    ConfirmedNoEffect,
    ConfirmedEffect,
    Ambiguous,
}

#[derive(Clone, Debug, Deserialize)]
struct RecoveryAttempt {
    action: String,
    outcome: String,
    evidence: String,
}

#[derive(Deserialize)]
struct ReportBlockerArgs {
    reason: String,
    blocker_class: BlockerClass,
    external_effect_state: ExternalEffectState,
    recovery_attempts: Vec<RecoveryAttempt>,
    #[serde(default)]
    outcome: Option<String>,
    #[serde(default)]
    partial_work: Option<String>,
    #[serde(default)]
    exact_need: Option<String>,
    #[serde(default)]
    next_step: Option<String>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    dependency_repair: Option<bool>,
    #[serde(default)]
    consequence_if_not_provided: Option<String>,
    #[serde(default)]
    artifacts: Option<Vec<String>>,
    #[serde(default)]
    options: Option<Vec<String>>,
}

fn sanitize_mandate_text(value: &str, max_chars: usize) -> String {
    value
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .take(max_chars)
        .collect::<String>()
        .trim()
        .to_string()
}

fn sanitize_mandate_blocker_args(args: &mut ReportBlockerArgs) {
    args.reason = sanitize_mandate_text(&args.reason, 500);
    for value in [
        &mut args.partial_work,
        &mut args.exact_need,
        &mut args.next_step,
        &mut args.target,
        &mut args.consequence_if_not_provided,
    ] {
        if let Some(text) = value.as_mut() {
            *text = sanitize_mandate_text(text, 500);
        }
    }
    for values in [&mut args.artifacts, &mut args.options] {
        if let Some(values) = values.as_mut() {
            values.truncate(8);
            for value in values {
                *value = sanitize_mandate_text(value, 300);
            }
        }
    }
    args.recovery_attempts.truncate(5);
    for attempt in &mut args.recovery_attempts {
        attempt.action = sanitize_mandate_text(&attempt.action, 300);
        attempt.outcome = sanitize_mandate_text(&attempt.outcome, 300);
        attempt.evidence = sanitize_mandate_text(&attempt.evidence, 500);
    }
}

fn suppresses_immediate_blocker_notification(context: Option<&str>) -> bool {
    context
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|value| value.get("terminal_recovery").and_then(Value::as_bool))
        .unwrap_or(false)
}

fn blocker_notification_excerpt(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }

    let mut excerpt = compact.chars().take(max_chars).collect::<String>();
    if let Some(last_space) = excerpt.rfind(' ') {
        excerpt.truncate(last_space);
    }
    excerpt.push('…');
    excerpt
}

fn build_blocker_notification(
    reason: &str,
    task_description: &str,
    exact_need: Option<&str>,
    task_id: &str,
) -> String {
    const MAX_STEP_CHARS: usize = 140;

    let mut message = format!(
        "⚠️ **Action needed**\n\n**Blocked:** {}",
        blocker_notification_excerpt(reason, 500)
    );
    let compact_step = blocker_notification_excerpt(task_description, MAX_STEP_CHARS);
    if task_description.chars().count() <= MAX_STEP_CHARS
        && !reason
            .to_ascii_lowercase()
            .contains(&task_description.to_ascii_lowercase())
    {
        message.push_str(&format!("\n\n**Step:** {compact_step}"));
    }
    if let Some(need) = exact_need.map(str::trim).filter(|need| !need.is_empty()) {
        message.push_str(&format!(
            "\n\n**Needed:** {}",
            blocker_notification_excerpt(need, 360)
        ));
    }
    let short_id = task_id.chars().take(8).collect::<String>();
    message.push_str(&format!(
        "\n\n**Resume:** `/work unblock {short_id} <resolution>`"
    ));
    message
}

fn validate_blocker_boundary(
    args: &ReportBlockerArgs,
    causal_local_repair_is_proven: bool,
) -> Result<(), String> {
    let exact_need = args
        .exact_need
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if args.blocker_class == BlockerClass::AmbiguousExternalEffect
        && args.external_effect_state != ExternalEffectState::Ambiguous
    {
        return Err(
            "ambiguous_external_effect requires external_effect_state=ambiguous".to_string(),
        );
    }
    if args.external_effect_state == ExternalEffectState::Ambiguous
        && args.blocker_class != BlockerClass::AmbiguousExternalEffect
    {
        return Err(
            "an ambiguous external effect must use blocker_class=ambiguous_external_effect"
                .to_string(),
        );
    }

    match args.blocker_class {
        BlockerClass::RecoveryExhausted => {
            if args.external_effect_state == ExternalEffectState::Ambiguous {
                return Err(
                    "ambiguous external effects require reconciliation, not operational retry"
                        .to_string(),
                );
            }
            if args.recovery_attempts.len() < 2 {
                return Err(
                    "operational recovery is not exhausted: record at least two bounded in-scope recovery attempts, then rerun the original verification"
                        .to_string(),
                );
            }
            if args.recovery_attempts.iter().any(|attempt| {
                attempt.action.trim().is_empty()
                    || attempt.outcome.trim().is_empty()
                    || attempt.evidence.trim().is_empty()
            }) {
                return Err(
                    "every recovery attempt must include its action, outcome, and concrete evidence"
                        .to_string(),
                );
            }
        }
        BlockerClass::OwnerInput
        | BlockerClass::ExternalDependency
        | BlockerClass::AmbiguousExternalEffect
        | BlockerClass::SafetyBoundary => {
            if exact_need.is_none() {
                return Err(
                    "this blocker class requires exact_need to identify the unavailable external input, authority, dependency, reconciliation, or safety judgment"
                        .to_string(),
                );
            }
        }
        BlockerClass::MissingAuthority => {
            if exact_need.is_none() {
                return Err(
                    "this blocker class requires exact_need to identify the unavailable authority"
                        .to_string(),
                );
            }
            let Some(_dependency_repair) = args.dependency_repair else {
                return Err(
                    "missing_authority requires dependency_repair so a minimal local repair of a proven task dependency is not escalated as unrelated work"
                        .to_string(),
                );
            };
            if causal_local_repair_is_proven {
                return Err(
                    "durable tool evidence proves that this target is inside the delegated project and directly blocked the required workflow. The proposed minimal reversible local repair is already in scope. Inspect the exact invariant, preserve substantive content, apply only that repair, rerun the failed workflow, and continue"
                        .to_string(),
                );
            }
        }
    }

    if args.outcome.as_deref() == Some("needs_approval")
        && !matches!(
            args.blocker_class,
            BlockerClass::OwnerInput
                | BlockerClass::MissingAuthority
                | BlockerClass::SafetyBoundary
        )
    {
        return Err(
            "outcome=needs_approval requires an owner_input, missing_authority, or safety_boundary blocker"
                .to_string(),
        );
    }

    Ok(())
}

#[async_trait]
impl Tool for ReportBlockerTool {
    fn name(&self) -> &str {
        "report_blocker"
    }

    fn description(&self) -> &str {
        "Report a genuine external boundary or operational failure that remains unresolved after \
         bounded recovery. A command or tool failure alone is not a blocker: inspect current state, \
         try safe in-scope recovery, and rerun the original verification first."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "report_blocker",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string"
                    },
                    "blocker_class": {
                        "type": "string",
                        "enum": ["owner_input", "missing_authority", "external_dependency", "ambiguous_external_effect", "safety_boundary", "recovery_exhausted"]
                    },
                    "external_effect_state": {
                        "type": "string",
                        "enum": ["none", "confirmed_no_effect", "confirmed_effect", "ambiguous"]
                    },
                    "recovery_attempts": {
                        "type": "array",
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": { "type": "string" },
                                "outcome": { "type": "string" },
                                "evidence": { "type": "string" }
                            },
                            "required": ["action", "outcome", "evidence"],
                            "additionalProperties": false
                        }
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["blocked", "partial_done_blocked", "needs_approval", "reduce_scope", "abandon"]
                    },
                    "partial_work": {
                        "type": "string"
                    },
                    "exact_need": {
                        "type": "string"
                    },
                    "next_step": {
                        "type": "string"
                    },
                    "target": {
                        "type": "string"
                    },
                    "dependency_repair": {
                        "type": "boolean",
                        "description": "For missing_authority: true only for a minimal reversible local repair of a path named by failed required-workflow evidence."
                    },
                    "consequence_if_not_provided": {
                        "type": "string"
                    },
                    "artifacts": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "options": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["reason", "blocker_class", "external_effect_state", "recovery_attempts"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &crate::tools::traits::ToolCallContext,
    ) -> anyhow::Result<crate::tools::traits::ToolCallOutcome> {
        self.report(arguments).await
    }
}

impl ReportBlockerTool {
    /// Record the blocker. A recorded blocker ends the executor's run as
    /// partial work, its summary the reply; a rejected report changes nothing
    /// and the run goes on.
    async fn report(&self, arguments: &str) -> anyhow::Result<ToolCallOutcome> {
        let mut args: ReportBlockerArgs = serde_json::from_str(arguments)?;
        if self.mandate_execution {
            anyhow::ensure!(
                self.attempt.is_some(),
                "A mandate blocker requires an exact durable task-attempt fence"
            );
            sanitize_mandate_blocker_args(&mut args);
        }

        let causal_local_repair_is_proven = self.causal_local_repair_is_proven(&args).await;
        if let Err(reason) = validate_blocker_boundary(&args, causal_local_repair_is_proven) {
            return Ok(ToolCallOutcome::from_output(format!(
                "Error: blocker report rejected because {reason}. No task state changed. Continue the task autonomously: inspect current state, choose a safe in-scope RETRY, REPAIR, SUBSTITUTE, or RECONCILE action, and rerun the original verification before reporting a blocker."
            )));
        }

        let outcome = classify_blocker_outcome(&args);
        let partial_result = args
            .partial_work
            .as_ref()
            .map(|partial_work| PartialResult {
                completed_work_summary: partial_work.clone(),
                artifacts: args.artifacts.clone().unwrap_or_default(),
                blocker: args.reason.clone(),
                remaining_work: args.options.clone().unwrap_or_default(),
            });
        let exact_need = args.exact_need.clone().or_else(|| {
            args.options.as_ref().map(|options| {
                if options.is_empty() {
                    "Resolve the blocker and resume the task.".to_string()
                } else {
                    format!("Choose one of: {}", options.join(", "))
                }
            })
        });
        let next_step = args
            .next_step
            .clone()
            .unwrap_or_else(|| "Resume the task after the blocker is resolved.".to_string());
        let approval_request = (outcome == TaskValidationOutcome::NeedsApproval).then(|| {
            let mut request = build_needs_approval_request(
                args.reason.clone(),
                args.target.clone(),
                args.reason.clone(),
                exact_need
                    .clone()
                    .unwrap_or_else(|| "Explicit approval to continue.".to_string()),
                next_step.clone(),
                partial_result.clone(),
            );
            request.consequence_if_not_provided = args
                .consequence_if_not_provided
                .clone()
                .or(request.consequence_if_not_provided.clone());
            request
        });
        let executor_result = ExecutorStepResult {
            task_id: self.task_id.clone(),
            step_outcome: match outcome {
                TaskValidationOutcome::NeedsApproval => StepValidationOutcome::NeedsApproval,
                TaskValidationOutcome::PartialDoneBlocked => {
                    StepValidationOutcome::PartialDoneBlocked
                }
                TaskValidationOutcome::ReduceScope => StepValidationOutcome::ReduceScope,
                TaskValidationOutcome::Abandon => StepValidationOutcome::Abandon,
                TaskValidationOutcome::Blocked => StepValidationOutcome::Blocked,
                TaskValidationOutcome::VerifyAgain => StepValidationOutcome::VerifyAgain,
                TaskValidationOutcome::ReplanTask => StepValidationOutcome::ReplanTask,
                TaskValidationOutcome::TaskDone | TaskValidationOutcome::ContinueWithNextStep => {
                    StepValidationOutcome::Blocked
                }
            },
            task_outcome: outcome.clone(),
            summary: args
                .partial_work
                .clone()
                .unwrap_or_else(|| args.reason.clone()),
            artifacts: args.artifacts.clone().unwrap_or_default(),
            blocker: Some(args.reason.clone()),
            exact_need: exact_need.clone(),
            next_step: Some(next_step.clone()),
            approval_request,
            partial_result,
        };

        // Build blocker details
        let mut blocker = format!("BLOCKED: {}", args.reason);
        if let Some(partial) = &args.partial_work {
            blocker.push_str(&format!("\nPartial work: {}", partial));
        }
        if let Some(options) = &args.options {
            blocker.push_str(&format!("\nPossible resolutions: {}", options.join(", ")));
        }

        // Persist the blocker through the current fenced attempt. The
        // compatibility path is retained for direct unit tests and older
        // callers that do not execute under a durable claim.
        if let Ok(Some(mut task)) = self.state.get_task(&self.task_id).await {
            // A fenced executor is subordinate to a parent task lead. Its
            // blocker is a durable handoff for that parent to reconcile, not a
            // terminal owner escalation. The parent may resolve the blocker,
            // retry safely, or eventually surface its own terminal outcome.
            // Publishing here races that reconciliation and can tell the owner
            // to intervene even though the harness is still recovering.
            let parent_owns_terminal_decision = self.attempt.is_some();
            let suppress_immediate_notification = parent_owns_terminal_decision
                || self.mandate_execution
                || suppresses_immediate_blocker_notification(task.context.as_deref());
            let result_summary = if task
                .result
                .as_deref()
                .is_none_or(|result| result.trim().is_empty())
            {
                Some(executor_result.render_task_lead_summary())
            } else {
                task.result.clone()
            };
            let context =
                persist_executor_result_context(task.context.as_deref(), &executor_result).ok();
            if let Some(attempt) = &self.attempt {
                let handoff = TaskHandoff {
                    id: uuid::Uuid::new_v4().to_string(),
                    task_id: self.task_id.clone(),
                    attempt_id: attempt.id.clone(),
                    summary: executor_result.summary.clone(),
                    artifacts: executor_result
                        .artifacts
                        .iter()
                        .map(|reference| HandoffArtifact {
                            kind: if reference.starts_with("http://")
                                || reference.starts_with("https://")
                            {
                                "url".to_string()
                            } else {
                                "path".to_string()
                            },
                            reference: reference.clone(),
                            digest: None,
                            metadata: None,
                        })
                        .collect(),
                    verification: Vec::new(),
                    remaining_risk: Some(args.reason.clone()),
                    next_step: Some(next_step.clone()),
                    created_at: chrono::Utc::now().to_rfc3339(),
                };
                let patch = TaskAttemptPatch {
                    status: "blocked".to_string(),
                    result: result_summary,
                    error: None,
                    blocker: Some(blocker.clone()),
                    context,
                    handoff: Some(handoff),
                };
                let updated = self
                    .state
                    .patch_task_from_attempt(&attempt.id, &attempt.lease_token, &patch)
                    .await?;
                if !updated {
                    anyhow::bail!(
                        "The execution lease is no longer current; the blocker was not applied"
                    );
                }
            } else {
                task.status = "blocked".to_string();
                task.blocker = Some(blocker.clone());
                task.result = result_summary;
                task.context = context;
                task.completed_at = Some(chrono::Utc::now().to_rfc3339());
                self.state.update_task(&task).await?;
            }
            info!(task_id = %self.task_id, reason = %args.reason, "Executor reported blocker");

            // Surface the blocker to the user right away through the
            // notification queue (delivered on the next heartbeat tick)
            // instead of waiting for the goal wrap-up summary. A blocker is
            // usually actionable by the user (start a service, grant access),
            // so minutes of silence here cost real wall-clock time.
            if !suppress_immediate_notification {
                if let Ok(Some(goal)) = self.state.get_goal(&task.goal_id).await {
                    let message = build_blocker_notification(
                        &args.reason,
                        &task.description,
                        exact_need.as_deref(),
                        &self.task_id,
                    );
                    let entry = crate::tools::traits::NotificationEntry::new(
                        &goal.id,
                        &goal.session_id,
                        "escalation",
                        &message,
                    )
                    .with_task(&self.task_id);
                    if let Err(e) = self.state.enqueue_notification(&entry).await {
                        info!(task_id = %self.task_id, error = %e, "Failed to enqueue blocker notification");
                    }
                }
            } else {
                info!(
                    task_id = %self.task_id,
                    parent_owns_terminal_decision,
                    mandate_execution = self.mandate_execution,
                    "Suppressed immediate blocker escalation for controlled task finalization"
                );
            }
        }

        // The dispatcher ships the summary the model saw as the reply.
        Ok(
            ToolCallOutcome::from_output(executor_result.render_task_lead_summary()).with_finish(
                aidaemon_core::Finish::new(aidaemon_core::FinishOutcome::Partial, None),
            ),
        )
    }
}

impl crate::tools::traits::ToolSemantics for ReportBlockerTool {
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::administrative()
    }

    /// Records the blocker in the agent's own task state; nothing else.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::none()
    }
}

fn classify_blocker_outcome(args: &ReportBlockerArgs) -> TaskValidationOutcome {
    match args.outcome.as_deref() {
        Some("needs_approval") => TaskValidationOutcome::NeedsApproval,
        Some("partial_done_blocked") => TaskValidationOutcome::PartialDoneBlocked,
        Some("reduce_scope") => TaskValidationOutcome::ReduceScope,
        Some("abandon") => TaskValidationOutcome::Abandon,
        Some("blocked") => TaskValidationOutcome::Blocked,
        _ if args.partial_work.is_some() => TaskValidationOutcome::PartialDoneBlocked,
        _ => TaskValidationOutcome::Blocked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocker_notification_omits_a_long_task_prompt() {
        let long_task = "Create and publish exactly one diary post. ".repeat(12);

        let message = build_blocker_notification(
            "The deploy command failed because a post ID was invalid.",
            &long_task,
            Some("Correct the repository build inconsistency."),
            "60559600-abcd",
        );

        assert!(!message.contains("Create and publish"));
        assert!(message.contains("**Blocked:** The deploy command failed"));
        assert!(message.contains("**Needed:** Correct the repository"));
        assert!(message.contains("`/work unblock 60559600 <resolution>`"));
    }
}
