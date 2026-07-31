use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::agent::{
    build_needs_approval_request, persist_executor_result_context, ExecutorStepResult,
    PartialResult, StepValidationOutcome, TaskValidationOutcome,
};
use crate::traits::{
    HandoffArtifact, StateStore, TaskAttempt, TaskAttemptPatch, TaskHandoff, Tool,
    ToolCallSemantics, ToolCapabilities, ToolRole,
};

/// Tool for executors to report they are blocked and cannot proceed.
///
pub struct ReportBlockerTool {
    task_id: String,
    state: Arc<dyn StateStore>,
    attempt: Option<TaskAttempt>,
}

impl ReportBlockerTool {
    #[cfg(test)]
    pub fn new(task_id: String, state: Arc<dyn StateStore>) -> Self {
        Self {
            task_id,
            state,
            attempt: None,
        }
    }

    pub fn for_attempt(task_id: String, state: Arc<dyn StateStore>, attempt: TaskAttempt) -> Self {
        Self {
            task_id,
            state,
            attempt: Some(attempt),
        }
    }
}

#[derive(Deserialize)]
struct ReportBlockerArgs {
    reason: String,
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
    consequence_if_not_provided: Option<String>,
    #[serde(default)]
    artifacts: Option<Vec<String>>,
    #[serde(default)]
    options: Option<Vec<String>>,
}

fn suppresses_immediate_blocker_notification(context: Option<&str>) -> bool {
    context
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|value| value.get("terminal_recovery").and_then(Value::as_bool))
        .unwrap_or(false)
}

fn is_self_resolvable_execution_preflight(args: &ReportBlockerArgs) -> bool {
    let combined = format!(
        "{} {}",
        args.reason,
        args.exact_need.as_deref().unwrap_or_default()
    )
    .to_ascii_lowercase();
    combined.contains("refusing an unbounded checkpoint")
        || combined.contains("outside checkpoint scope")
        || combined.contains("checkpoint for a command spanning multiple project roots")
        || (combined.contains("terminal call limit")
            && (combined.contains("workspace container")
                || combined.contains("bounded subdirectory")))
}

#[async_trait]
impl Tool for ReportBlockerTool {
    fn name(&self) -> &str {
        "report_blocker"
    }

    fn description(&self) -> &str {
        "Report that you are blocked and cannot proceed. Use this instead of guessing \
         when you encounter ambiguity, missing information, or an obstacle you cannot resolve."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "report_blocker",
            "description": "Report that you are blocked and cannot proceed. Use this instead of guessing when you encounter ambiguity, missing information, or an obstacle you cannot resolve.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you are blocked"
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["blocked", "partial_done_blocked", "needs_approval", "reduce_scope", "abandon"],
                        "description": "Structured blocker outcome. Use partial_done_blocked when some work is complete, or needs_approval when a gated action requires permission."
                    },
                    "partial_work": {
                        "type": "string",
                        "description": "What you completed so far"
                    },
                    "exact_need": {
                        "type": "string",
                        "description": "The exact input, approval, permission, or dependency needed to unblock the task"
                    },
                    "next_step": {
                        "type": "string",
                        "description": "What should happen immediately after the blocker is resolved"
                    },
                    "target": {
                        "type": "string",
                        "description": "The target path, URL, system, or task artifact affected by the blocker"
                    },
                    "consequence_if_not_provided": {
                        "type": "string",
                        "description": "What will happen if the missing input or approval is not provided"
                    },
                    "artifacts": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Relevant artifacts or target paths already touched before the blocker"
                    },
                    "options": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Possible resolutions (if any)"
                    }
                },
                "required": ["reason"],
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
        let args: ReportBlockerArgs = serde_json::from_str(arguments)?;

        if is_self_resolvable_execution_preflight(&args) {
            return Ok(
                "Error: this is a recoverable internal execution preflight, not a human blocker. No task state changed and no process was started. Select one explicit bounded project directory, retry the command from there, and use another available inspection tool if needed."
                    .to_string(),
            );
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
            let suppress_immediate_notification =
                suppresses_immediate_blocker_notification(task.context.as_deref());
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
                    let mut message = format!(
                        "\u{26a0}\u{fe0f} A step is blocked: {}\nStep: {}",
                        args.reason, task.description
                    );
                    if let Some(need) = &exact_need {
                        message.push_str(&format!("\nNeeded to continue: {}", need));
                    }
                    let short_id = self.task_id.chars().take(8).collect::<String>();
                    message.push_str(&format!(
                        "\nReply with /work unblock {} <resolution> after resolving it.",
                        short_id
                    ));
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
                    "Suppressed duplicate blocker escalation for terminal recovery task"
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
                    "reason": "Missing API credentials"
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
    }

    #[tokio::test]
    async fn recoverable_checkpoint_preflight_is_not_escalated_to_the_user() {
        let (state, _goal_id, task_id) = setup_test_state().await;
        let tool = ReportBlockerTool::new(task_id.clone(), state.clone());
        let original_status = state.get_task(&task_id).await.unwrap().unwrap().status;

        let result = tool
            .call(
                &json!({
                    "reason": "The command surface refused an unbounded checkpoint at workspace container /Users/example/projects and the terminal call limit was reached.",
                    "exact_need": "A bounded subdirectory under /Users/example/projects."
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.starts_with("Error: this is a recoverable internal execution preflight"));
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

        let semantics = tool.call_semantics(r#"{"reason":"OAuth authorization required"}"#);

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
                    "outcome": "needs_approval",
                    "partial_work": "Validated the pending rotation script and staged the change plan",
                    "exact_need": "Owner approval to rotate the credentials in production.",
                    "next_step": "Run the approved credential rotation and verify the service health.",
                    "target": "production credentials"
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
}
