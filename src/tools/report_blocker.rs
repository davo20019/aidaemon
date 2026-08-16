use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::agent::{
    build_needs_approval_request, extract_executor_handoff_context,
    persist_executor_result_context, ExecutorStepResult, PartialResult, StepValidationOutcome,
    TaskValidationOutcome,
};
use crate::traits::{
    HandoffArtifact, ReportBlockerStore, StateStore, TaskAttempt, TaskAttemptPatch, TaskHandoff,
    Tool, ToolCallSemantics, ToolCapabilities, ToolRole,
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
    #[cfg(test)]
    pub fn new(task_id: String, state: Arc<dyn StateStore>) -> Self {
        Self {
            task_id,
            state: state as Arc<dyn ReportBlockerStore>,
            attempt: None,
            mandate_execution: false,
        }
    }

    pub fn for_attempt(
        task_id: String,
        state: Arc<dyn StateStore>,
        attempt: TaskAttempt,
        mandate_execution: bool,
    ) -> Self {
        Self {
            task_id,
            state: state as Arc<dyn ReportBlockerStore>,
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
                    crate::traits::ToolTargetHintKind::Path
                        | crate::traits::ToolTargetHintKind::ProjectScope
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
            "description": "Report an unresolved external boundary or exhausted recovery.",
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
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
            return Ok(format!(
                "Error: blocker report rejected because {reason}. No task state changed. Continue the task autonomously: inspect current state, choose a safe in-scope RETRY, REPAIR, SUBSTITUTE, or RECONCILE action, and rerun the original verification before reporting a blocker."
            ));
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
                    let entry = crate::traits::NotificationEntry::new(
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

        Ok(executor_result.render_task_lead_summary())
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
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::store_prelude::*;
    use crate::traits::{Goal, Task};

    async fn setup_test_state() -> (Arc<dyn StateStore>, String, String) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );

        let goal = Goal::new_finite("Test goal", "test-session");
        state.create_goal(&goal).await.unwrap();

        let now = chrono::Utc::now().to_rfc3339();
        let task = Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            description: "Test task".to_string(),
            status: "running".to_string(),
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
            created_at: now,
            started_at: None,
            completed_at: None,
        };
        state.create_task(&task).await.unwrap();

        std::mem::forget(db_file);
        (state as Arc<dyn StateStore>, goal.id, task.id)
    }

    #[tokio::test]
    async fn test_report_blocker_updates_task() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "reason": "Missing API credentials",
                    "blocker_class": "external_dependency",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "exact_need": "Provide credentials for the requested API."
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Executor outcome: blocked"));
        assert!(result.contains("Summary: Missing API credentials"));

        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(task
            .blocker
            .as_deref()
            .unwrap()
            .contains("Missing API credentials"));
        assert!(task
            .context
            .as_deref()
            .unwrap()
            .contains("\"executor_result\""));
    }

    #[tokio::test]
    async fn test_report_blocker_enqueues_user_notification() {
        let (state, goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        tool.call(
            &json!({
                "reason": "Docker daemon is not reachable",
                "blocker_class": "external_dependency",
                "external_effect_state": "none",
                "recovery_attempts": [],
                "exact_need": "Start Docker, then ask me to retry."
            })
            .to_string(),
        )
        .await
        .unwrap();

        let pending = state.get_pending_notifications(10).await.unwrap();
        let entry = pending
            .iter()
            .find(|n| n.goal_id == goal_id)
            .expect("blocker should queue an immediate user notification");
        assert_eq!(entry.session_id, "test-session");
        assert_eq!(entry.notification_type, "escalation");
        assert!(entry.message.contains("Docker daemon is not reachable"));
        assert!(entry
            .message
            .contains("Start Docker, then ask me to retry."));
        assert!(entry.message.starts_with("⚠️ **Action needed**"));
        assert!(entry.message.contains("**Blocked:**"));
        assert!(entry.message.contains("**Needed:**"));
        assert!(entry.message.contains("**Resume:** `/work unblock"));
    }

    #[tokio::test]
    async fn fenced_executor_blocker_is_handoff_not_owner_escalation() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let mut task = state.get_task(&task_id).await.unwrap().unwrap();
        task.status = "pending".to_string();
        task.started_at = None;
        task.completed_at = None;
        state.update_task(&task).await.unwrap();
        let attempt = state
            .claim_task_with_lease(&task_id, "synthetic-executor", None, 180)
            .await
            .unwrap()
            .expect("executor should claim the task");
        let tool = ReportBlockerTool::for_attempt(task_id.clone(), state.clone(), attempt, false);

        tool.call(
            &json!({
                "reason": "Current source state contradicts the earlier deployment receipt.",
                "blocker_class": "ambiguous_external_effect",
                "external_effect_state": "ambiguous",
                "recovery_attempts": [],
                "outcome": "partial_done_blocked",
                "partial_work": "The deployment completed and returned a public URL.",
                "exact_need": "Reconcile the deployment receipt with current source state.",
                "next_step": "Parent task lead should inspect current state and choose the safe recovery path."
            })
            .to_string(),
        )
        .await
        .unwrap();

        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(task.context.as_deref().unwrap().contains("executor_result"));
        assert!(
            state
                .get_pending_notifications(10)
                .await
                .unwrap()
                .is_empty(),
            "the parent task lead owns reconciliation and terminal notification"
        );
    }

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

    #[tokio::test]
    async fn operational_failure_without_recovery_evidence_is_not_escalated_to_the_user() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());
        let original_status = state.get_task(&task_id).await.unwrap().unwrap().status;

        let result = tool
            .call(
                &json!({
                    "reason": "The build reports invalid frontmatter while a direct parser check reports a valid integer.",
                    "blocker_class": "recovery_exhausted",
                    "external_effect_state": "none",
                    "recovery_attempts": [{
                        "action": "Parsed the current file directly",
                        "outcome": "The id is a valid integer",
                        "evidence": "Number.isInteger(data.id) returned true"
                    }]
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.starts_with("Error: blocker report rejected"));
        assert!(result.contains("at least two bounded in-scope recovery attempts"));
        assert_eq!(
            state.get_task(&task_id).await.unwrap().unwrap().status,
            original_status
        );
        assert!(state
            .get_pending_notifications(10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn terminal_recovery_blocker_uses_parent_terminal_notification_only() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let mut task = state.get_task(&task_id).await.unwrap().unwrap();
        task.context = Some(json!({ "terminal_recovery": true }).to_string());
        state.update_task(&task).await.unwrap();
        let tool = ReportBlockerTool::new(task_id, state.clone());

        tool.call(
            &json!({
                "reason": "Browser verification is unavailable",
                "blocker_class": "recovery_exhausted",
                "external_effect_state": "none",
                "recovery_attempts": [
                    {
                        "action": "Tried the browser verifier",
                        "outcome": "Browser session was unavailable",
                        "evidence": "Browser tool returned an unavailable-session error"
                    },
                    {
                        "action": "Tried an authorized HTTP verification",
                        "outcome": "No HTTP-capable tool was present",
                        "evidence": "Executor tool inventory contained no HTTP client"
                    }
                ],
                "exact_need": "Use another verification path."
            })
            .to_string(),
        )
        .await
        .unwrap();

        assert!(state
            .get_pending_notifications(10)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn report_blocker_has_administrative_semantics() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id, state);

        let semantics = tool.call_semantics(
            r#"{"reason":"OAuth authorization required","blocker_class":"external_dependency","external_effect_state":"none","recovery_attempts":[]}"#,
        );

        assert_eq!(
            semantics.effect,
            crate::traits::ToolCallEffect::Administrative
        );
    }

    #[tokio::test]
    async fn test_report_blocker_with_partial_work() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "reason": "Need clarification on API version",
                    "blocker_class": "owner_input",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "outcome": "partial_done_blocked",
                    "partial_work": "Set up project structure and dependencies",
                    "exact_need": "Choose between the v1 and v2 API contract.",
                    "next_step": "Resume the client implementation once the API version is confirmed.",
                    "artifacts": ["/tmp/demo/Cargo.toml"],
                    "options": ["Use v1 API", "Use v2 API"]
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Executor outcome: partial_done_blocked"));
        assert!(result.contains("Completed work so far: Set up project structure and dependencies"));

        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert_eq!(task.status, "blocked");
        assert!(task
            .blocker
            .as_deref()
            .unwrap()
            .contains("Need clarification"));
        assert!(task
            .blocker
            .as_deref()
            .unwrap()
            .contains("Possible resolutions"));
        assert!(task
            .context
            .as_deref()
            .unwrap()
            .contains("\"partial_done_blocked\""));
    }

    #[tokio::test]
    async fn test_report_blocker_supports_needs_approval() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());

        let result = tool
            .call(
                &json!({
                    "reason": "Need approval to rotate the production credentials",
                    "blocker_class": "missing_authority",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "outcome": "needs_approval",
                    "partial_work": "Validated the pending rotation script and staged the change plan",
                    "exact_need": "Owner approval to rotate the credentials in production.",
                    "next_step": "Run the approved credential rotation and verify the service health.",
                    "target": "production credentials",
                    "dependency_repair": false
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Executor outcome: needs_approval"));
        let task = state.get_task(&task_id).await.unwrap().unwrap();
        assert!(task
            .context
            .as_deref()
            .unwrap()
            .contains("\"needs_approval\""));
    }

    #[tokio::test]
    async fn proven_causal_local_repair_is_not_accepted_as_missing_authority() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let project = tempfile::tempdir().unwrap();
        let target = project.path().join("content/posts/synthetic-post.md");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(
            &target,
            "---\ntitle: Synthetic\n---\n\nBody stays intact.\n",
        )
        .unwrap();

        let mut task = state.get_task(&task_id).await.unwrap().unwrap();
        let handoff = crate::agent::ExecutorHandoff {
            task_id: task_id.clone(),
            mission: "Build and publish the synthetic site.".to_string(),
            task_description: "Repair the failed build and continue the workflow.".to_string(),
            target_scope: crate::agent::TargetScope {
                allowed_targets: vec![crate::traits::ToolTargetHint::new(
                    crate::traits::ToolTargetHintKind::ProjectScope,
                    project.path().to_string_lossy(),
                )
                .unwrap()],
                hard_fail_outside_scope: true,
            },
            expected_targets: Vec::new(),
            allowed_tools: None,
        };
        task.context = Some(
            crate::agent::persist_executor_handoff_context(task.context.as_deref(), &handoff)
                .unwrap(),
        );
        state.update_task(&task).await.unwrap();
        state
            .log_task_activity(&crate::traits::TaskActivity {
                id: 0,
                task_id: task_id.clone(),
                activity_type: "tool_call".to_string(),
                tool_name: Some("run_command".to_string()),
                tool_args: Some("{\"command\":\"npm run build\"}".to_string()),
                result: Some(format!(
                    "Build failed: {} lacks the required numeric frontmatter id.",
                    target.display()
                )),
                success: Some(false),
                tokens_used: None,
                created_at: chrono::Utc::now().to_rfc3339(),
            })
            .await
            .unwrap();

        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());
        let result = tool
            .call(
                &json!({
                    "reason": "The pre-existing post needs a mechanical frontmatter repair.",
                    "blocker_class": "missing_authority",
                    "external_effect_state": "none",
                    "recovery_attempts": [],
                    "outcome": "needs_approval",
                    "exact_need": "Authority to add the required numeric id.",
                    "next_step": "Add only the missing id, rerun the build, and continue.",
                    "target": target.to_string_lossy(),
                    "dependency_repair": true
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("already in scope"), "result: {result}");
        assert!(result.contains("durable tool evidence"), "result: {result}");
        assert_eq!(
            state.get_task(&task_id).await.unwrap().unwrap().status,
            "running"
        );
    }
}
