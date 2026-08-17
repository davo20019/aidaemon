#![allow(clippy::items_after_test_module)]

use super::completion_checks::*;
use super::response_phase::ResponsePhaseOutcome;
use super::validation_state::HumanInterventionRequest;
use super::*;
use crate::events::TaskOutcome;
use crate::execution_policy::PolicyBundle;
use crate::llm_markers::INTENT_GATE_MARKER;
use crate::traits::ProviderResponse;

const MAX_VERIFICATION_ATTEMPTS: usize = 2;

pub(super) struct CompletionCtx<'a> {
    pub resp: &'a mut ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub user_text: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a mut LearningContext,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub tool_defs: &'a mut Vec<Value>,
    pub base_tool_defs: &'a mut Vec<Value>,
    pub available_capabilities: &'a mut HashMap<String, ToolCapabilities>,
    pub policy_bundle: &'a mut PolicyBundle,
    pub llm_provider: Arc<dyn ModelProvider>,
    pub llm_router: Option<Router>,
    pub model: &'a mut String,
    pub channel_ctx: ChannelContext,
    pub user_role: UserRole,
    pub total_successful_tool_calls: usize,
    pub stall_count: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
    pub deferred_no_tool_streak: &'a mut usize,
    pub deferred_no_tool_model_switches: &'a mut usize,
    pub fallback_expanded_once: &'a mut bool,
    pub empty_response_retry_used: &'a mut bool,
    pub empty_response_retry_pending: &'a mut bool,
    pub empty_response_retry_note: &'a mut Option<String>,
    pub identity_prefill_text: &'a mut Option<String>,
    pub pending_background_ack: &'a mut Option<String>,
    pub pending_external_action_ack: &'a mut Option<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
    pub completion_progress: &'a mut CompletionProgress,
    pub turn_context: &'a TurnContext,
    pub execution_requirement: &'a ExecutionRequirement,
    pub force_text_response: &'a mut bool,
    pub execution_state: &'a mut ExecutionState,
    pub validation_state: &'a mut ValidationState,
}

/// Owned completion-phase state accumulated while the ordered completion gates
/// run. Applying this delta is deliberately separate from evaluating the gates:
/// every successful, retry, and error exit must restore the state moved out of
/// [`CompletionCtx`] exactly once.
struct CompletionStateDelta {
    tool_defs: Vec<Value>,
    model: String,
    stall_count: usize,
    consecutive_clean_iterations: usize,
    deferred_no_tool_streak: usize,
    deferred_no_tool_model_switches: usize,
    fallback_expanded_once: bool,
    empty_response_retry_used: bool,
    empty_response_retry_pending: bool,
    empty_response_retry_note: Option<String>,
    identity_prefill_text: Option<String>,
    pending_background_ack: Option<String>,
    pending_external_action_ack: Option<String>,
    require_file_recheck_before_answer: bool,
    completion_progress: CompletionProgress,
    force_text_response: bool,
    validation_state: ValidationState,
}

/// Narrow destination view for the mutable state owned by the caller. Keeping
/// this separate from `CompletionCtx` makes the application boundary testable
/// without constructing provider, policy, or event services.
struct CompletionStateTargets<'a> {
    tool_defs: &'a mut Vec<Value>,
    model: &'a mut String,
    stall_count: &'a mut usize,
    consecutive_clean_iterations: &'a mut usize,
    deferred_no_tool_streak: &'a mut usize,
    deferred_no_tool_model_switches: &'a mut usize,
    fallback_expanded_once: &'a mut bool,
    empty_response_retry_used: &'a mut bool,
    empty_response_retry_pending: &'a mut bool,
    empty_response_retry_note: &'a mut Option<String>,
    identity_prefill_text: &'a mut Option<String>,
    pending_background_ack: &'a mut Option<String>,
    pending_external_action_ack: &'a mut Option<String>,
    require_file_recheck_before_answer: &'a mut bool,
    completion_progress: &'a mut CompletionProgress,
    force_text_response: &'a mut bool,
    validation_state: &'a mut ValidationState,
}

impl CompletionStateDelta {
    fn apply(self, targets: CompletionStateTargets<'_>) {
        *targets.tool_defs = self.tool_defs;
        *targets.model = self.model;
        *targets.stall_count = self.stall_count;
        *targets.consecutive_clean_iterations = self.consecutive_clean_iterations;
        *targets.deferred_no_tool_streak = self.deferred_no_tool_streak;
        *targets.deferred_no_tool_model_switches = self.deferred_no_tool_model_switches;
        *targets.fallback_expanded_once = self.fallback_expanded_once;
        *targets.empty_response_retry_used = self.empty_response_retry_used;
        *targets.empty_response_retry_pending = self.empty_response_retry_pending;
        *targets.empty_response_retry_note = self.empty_response_retry_note;
        *targets.identity_prefill_text = self.identity_prefill_text;
        *targets.pending_background_ack = self.pending_background_ack;
        *targets.pending_external_action_ack = self.pending_external_action_ack;
        *targets.require_file_recheck_before_answer = self.require_file_recheck_before_answer;
        *targets.completion_progress = self.completion_progress;
        *targets.force_text_response = self.force_text_response;
        *targets.validation_state = self.validation_state;
    }
}

fn finish_completion_phase<T>(
    outcome: anyhow::Result<T>,
    delta: CompletionStateDelta,
    targets: CompletionStateTargets<'_>,
) -> anyhow::Result<T> {
    delta.apply(targets);
    outcome
}

fn build_final_sanitization_gate_telemetry(
    original_chars: usize,
    final_chars: usize,
    collapsed_repetition: bool,
    gutted_by_sanitization: bool,
) -> (String, Value) {
    let changed = original_chars != final_chars || collapsed_repetition || gutted_by_sanitization;
    let action = if gutted_by_sanitization {
        "recovered_from_gutted"
    } else if changed {
        "sanitized"
    } else {
        "passed"
    };
    let summary = if changed {
        format!(
            "Final sanitization gate changed final reply: {original_chars} -> {final_chars} chars"
        )
    } else {
        format!("Final sanitization gate passed final reply unchanged: {final_chars} chars")
    };
    (
        summary,
        json!({
            "condition": "final_reply_sanitization",
            "gate_family": "final_reply",
            "action": action,
            "original_chars": original_chars,
            "final_chars": final_chars,
            "delta_chars": (final_chars as i64) - (original_chars as i64),
            "collapsed_repetition": collapsed_repetition,
            "gutted_by_sanitization": gutted_by_sanitization,
        }),
    )
}

fn build_terminal_verification_request(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    execution_state: Option<&ExecutionState>,
    concrete_progress: bool,
    partial_blocker: impl Into<String>,
    exact_need: impl Into<String>,
    next_step: impl Into<String>,
) -> (HumanInterventionRequest, Option<TaskTerminalCause>) {
    let exact_need = exact_need.into();
    let next_step = next_step.into();
    if concrete_progress {
        let mut request = build_partial_done_blocked_request_with_plan(
            turn_context,
            learning_ctx,
            execution_state,
            partial_blocker,
            exact_need,
            next_step,
        );
        request.user_action_required = false;
        request.consequence_if_not_provided = None;
        return (request, None);
    }

    let mut request = build_abandon_request(
        turn_context,
        learning_ctx,
        "No concrete tool or verification step completed, so there is no partial result to preserve.",
        exact_need,
        next_step,
    );
    request.user_action_required = false;
    request.consequence_if_not_provided = None;
    (request, Some(TaskTerminalCause::HardFailure))
}

fn build_agent_side_partial_failure(
    turn_context: &TurnContext,
    learning_ctx: &LearningContext,
    execution_state: Option<&ExecutionState>,
    blocker: impl Into<String>,
    next_step: impl Into<String>,
) -> HumanInterventionRequest {
    let mut request = build_partial_done_blocked_request_with_plan(
        turn_context,
        learning_ctx,
        execution_state,
        blocker,
        "No user input is required for this agent-side execution failure.",
        next_step,
    );
    request.user_action_required = false;
    request.consequence_if_not_provided = None;
    request
}

/// Decide whether ordinary conversational memory maintenance may run after a
/// completed turn. Mandate workers have their own bounded, typed continuity;
/// allowing either background path here would both leak mandate content into
/// global owner memory and make unmetered provider calls outside the durable
/// per-run token lease.
fn post_completion_memory_plan(
    mandate_execution: bool,
    memory_pipeline_allowed: bool,
    progressive_facts_enabled: bool,
    fact_extraction_eligible: bool,
    summarization_enabled: bool,
) -> (bool, bool) {
    if mandate_execution || !memory_pipeline_allowed {
        return (false, false);
    }
    (
        progressive_facts_enabled && fact_extraction_eligible,
        summarization_enabled,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MandateDecisionCommitState {
    NotApplicable,
    Missing,
    Committed,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MandateDecisionCompletionAction {
    Allow,
    Retry,
    Reject,
}

async fn mandate_decision_commit_state(
    agent: &Agent,
) -> anyhow::Result<MandateDecisionCommitState> {
    let Some(fence) = agent.mandate_execution.as_ref() else {
        return Ok(MandateDecisionCommitState::NotApplicable);
    };
    if fence.worker_task_id != fence.root_task_id {
        return Ok(MandateDecisionCommitState::NotApplicable);
    }
    let Some(decision) = agent
        .state
        .get_mandate_decision_for_run(&fence.goal_run_id)
        .await?
    else {
        return Ok(MandateDecisionCommitState::Missing);
    };
    if decision.goal_run_id == fence.goal_run_id
        && decision.mandate_id == fence.mandate_id
        && decision.mandate_version == fence.mandate_version
    {
        Ok(MandateDecisionCommitState::Committed)
    } else {
        Ok(MandateDecisionCommitState::Invalid)
    }
}

fn mandate_decision_completion_action(
    state: MandateDecisionCommitState,
    retries_used: usize,
    decision_tool_available: bool,
) -> MandateDecisionCompletionAction {
    match state {
        MandateDecisionCommitState::NotApplicable | MandateDecisionCommitState::Committed => {
            MandateDecisionCompletionAction::Allow
        }
        MandateDecisionCommitState::Missing if retries_used == 0 && decision_tool_available => {
            MandateDecisionCompletionAction::Retry
        }
        MandateDecisionCommitState::Missing | MandateDecisionCommitState::Invalid => {
            MandateDecisionCompletionAction::Reject
        }
    }
}

fn tool_definition_is_named(definition: &Value, expected: &str) -> bool {
    definition
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        == Some(expected)
}

#[cfg(test)]
mod gate_telemetry_tests {
    use super::*;

    #[test]
    fn final_sanitization_telemetry_marks_changed_reply_and_repetition_collapse() {
        let (summary, metadata) = build_final_sanitization_gate_telemetry(120, 80, true, false);

        assert!(
            summary.contains("changed final reply"),
            "summary: {summary}"
        );
        assert_eq!(metadata["condition"], "final_reply_sanitization");
        assert_eq!(metadata["action"], "sanitized");
        assert_eq!(metadata["collapsed_repetition"], true);
        assert_eq!(metadata["gutted_by_sanitization"], false);
        assert_eq!(metadata["original_chars"], 120);
        assert_eq!(metadata["final_chars"], 80);
    }

    #[test]
    fn mandate_completion_disables_fact_extraction_and_summarization() {
        assert_eq!(
            post_completion_memory_plan(true, true, true, true, true),
            (false, false)
        );
        assert_eq!(
            post_completion_memory_plan(false, true, true, true, true),
            (true, true)
        );
        assert_eq!(
            post_completion_memory_plan(false, false, true, true, true),
            (false, false),
            "a task-local no-memory policy suppresses every post-turn pipeline"
        );
    }

    #[test]
    fn mandate_decision_completion_retries_once_then_rejects_prose() {
        assert_eq!(
            mandate_decision_completion_action(MandateDecisionCommitState::Missing, 0, true),
            MandateDecisionCompletionAction::Retry
        );
        assert_eq!(
            mandate_decision_completion_action(MandateDecisionCommitState::Missing, 1, true),
            MandateDecisionCompletionAction::Reject
        );
        assert_eq!(
            mandate_decision_completion_action(MandateDecisionCommitState::Missing, 0, false),
            MandateDecisionCompletionAction::Reject
        );
        assert_eq!(
            mandate_decision_completion_action(MandateDecisionCommitState::Committed, 1, true),
            MandateDecisionCompletionAction::Allow
        );
    }

    #[tokio::test]
    async fn mandate_prose_closeout_retries_and_corrected_typed_wait_commits() {
        use crate::testing::{setup_test_agent_with_mandates, MockProvider};
        use crate::traits::store_prelude::*;
        use crate::traits::{Goal, Mandate, MandateAuthority, MandateDecisionOutcome, Task};

        let provider = MockProvider::with_responses(vec![
            MockProvider::text_response("Decision: WAIT. No worthwhile action is available."),
            MockProvider::tool_call_response(
                "manage_mandates",
                &json!({
                    "action": "record_decision",
                    "outcome": "wait",
                    "activity_level": "quiet",
                    "rationale": "No worthwhile action is available in this review.",
                    "reconsider_minutes": 180,
                    "learning_note": "",
                    "learning_evidence_receipt_ids": [],
                    "strategy_key": "",
                    "strategy_kind": "explore",
                    "strategy_confidence_bps": 0
                })
                .to_string(),
            ),
        ]);
        let mut harness = setup_test_agent_with_mandates(provider).await.unwrap();
        let goal = Goal::new_continuous(
            "Review a synthetic bounded source",
            "owner-session",
            Some(250_000),
            Some(2_000_000),
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Review a synthetic bounded source",
            "owner-session",
            MandateAuthority::default(),
            3_600,
            21_600,
            10_800,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        let leased = harness
            .state
            .claim_due_mandates(1, "prose-closeout-test", 300)
            .await
            .unwrap()
            .pop()
            .expect("one due mandate");
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let root_task = Task {
            id: root_task_id.clone(),
            goal_id: goal.id.clone(),
            description: "Record one bounded decision".to_string(),
            status: "pending".to_string(),
            priority: "high".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: false,
            retry_count: 0,
            max_retries: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: None,
        };
        let run_id = uuid::Uuid::new_v4().to_string();
        let run = harness
            .state
            .create_mandate_review_run(
                &mandate.id,
                leased.review_lease_token.as_deref().unwrap(),
                &run_id,
                &root_task,
            )
            .await
            .unwrap();
        let attempt = harness
            .state
            .claim_task_with_lease(&root_task_id, "prose-closeout-task-lead", None, 300)
            .await
            .unwrap()
            .unwrap();
        harness.agent.set_test_mandate_execution(
            &mandate.id,
            mandate.version,
            mandate.authority.clone(),
            &goal.id,
            &root_task_id,
            &attempt.id,
            &attempt,
        );

        let response = harness
            .agent
            .handle_message(
                "synthetic-mandate-session",
                "Run one bounded autonomous review and record exactly one decision.",
                None,
                crate::types::UserRole::Owner,
                crate::types::ChannelContext::internal(),
                None,
            )
            .await
            .unwrap();

        assert!(!response.contains("Decision: WAIT"));
        assert_eq!(harness.provider.call_count().await, 2);
        let decision = harness
            .state
            .get_mandate_decision_for_run(&run.id)
            .await
            .unwrap()
            .expect("corrected durable decision");
        assert_eq!(decision.outcome, MandateDecisionOutcome::Wait);
        let calls = harness.provider.call_log.lock().await;
        let retry_messages = &calls[1].messages;
        assert!(retry_messages.iter().any(|message| {
            message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("no durable decision"))
        }));
    }

    #[test]
    fn completion_state_delta_is_applied_before_error_propagation() {
        let mut tool_defs = vec![serde_json::json!({"name": "old"})];
        let mut model = "old-model".to_string();
        let mut stall_count = 0;
        let mut consecutive_clean_iterations = 0;
        let mut deferred_no_tool_streak = 0;
        let mut deferred_no_tool_model_switches = 0;
        let mut fallback_expanded_once = false;
        let mut empty_response_retry_used = false;
        let mut empty_response_retry_pending = false;
        let mut empty_response_retry_note = Some("old-retry".to_string());
        let mut identity_prefill_text = Some("old-prefill".to_string());
        let mut pending_background_ack = Some("old-background".to_string());
        let mut pending_external_action_ack = None;
        let mut require_file_recheck_before_answer = false;
        let mut completion_progress = CompletionProgress::default();
        let mut force_text_response = false;
        let mut validation_state = ValidationState::default();

        let updated_progress = CompletionProgress {
            quality_nudge_count: 4,
            ..CompletionProgress::default()
        };
        let mut updated_validation = ValidationState::default();
        updated_validation.note_retry(LoopRepetitionReason::RetryStep);

        let outcome: anyhow::Result<()> = finish_completion_phase(
            Err(anyhow::anyhow!("synthetic phase failure")),
            CompletionStateDelta {
                tool_defs: vec![serde_json::json!({"name": "new"})],
                model: "new-model".to_string(),
                stall_count: 7,
                consecutive_clean_iterations: 3,
                deferred_no_tool_streak: 2,
                deferred_no_tool_model_switches: 1,
                fallback_expanded_once: true,
                empty_response_retry_used: true,
                empty_response_retry_pending: true,
                empty_response_retry_note: Some("new-retry".to_string()),
                identity_prefill_text: None,
                pending_background_ack: None,
                pending_external_action_ack: Some("external-ack".to_string()),
                require_file_recheck_before_answer: true,
                completion_progress: updated_progress,
                force_text_response: true,
                validation_state: updated_validation,
            },
            CompletionStateTargets {
                tool_defs: &mut tool_defs,
                model: &mut model,
                stall_count: &mut stall_count,
                consecutive_clean_iterations: &mut consecutive_clean_iterations,
                deferred_no_tool_streak: &mut deferred_no_tool_streak,
                deferred_no_tool_model_switches: &mut deferred_no_tool_model_switches,
                fallback_expanded_once: &mut fallback_expanded_once,
                empty_response_retry_used: &mut empty_response_retry_used,
                empty_response_retry_pending: &mut empty_response_retry_pending,
                empty_response_retry_note: &mut empty_response_retry_note,
                identity_prefill_text: &mut identity_prefill_text,
                pending_background_ack: &mut pending_background_ack,
                pending_external_action_ack: &mut pending_external_action_ack,
                require_file_recheck_before_answer: &mut require_file_recheck_before_answer,
                completion_progress: &mut completion_progress,
                force_text_response: &mut force_text_response,
                validation_state: &mut validation_state,
            },
        );

        assert!(outcome.is_err());
        assert_eq!(tool_defs, vec![serde_json::json!({"name": "new"})]);
        assert_eq!(model, "new-model");
        assert_eq!(stall_count, 7);
        assert_eq!(consecutive_clean_iterations, 3);
        assert_eq!(deferred_no_tool_streak, 2);
        assert_eq!(deferred_no_tool_model_switches, 1);
        assert!(fallback_expanded_once);
        assert!(empty_response_retry_used);
        assert!(empty_response_retry_pending);
        assert_eq!(empty_response_retry_note.as_deref(), Some("new-retry"));
        assert_eq!(identity_prefill_text, None);
        assert_eq!(pending_background_ack, None);
        assert_eq!(pending_external_action_ack.as_deref(), Some("external-ack"));
        assert!(require_file_recheck_before_answer);
        assert_eq!(completion_progress.quality_nudge_count, 4);
        assert!(force_text_response);
        assert_eq!(validation_state.retry_count, 1);
        assert_eq!(
            validation_state.loop_repetition_reason,
            Some(LoopRepetitionReason::RetryStep)
        );
    }
}

pub(super) async fn run_completion_phase(
    services: &super::services::AgentServices<'_>,
    ctx: &mut CompletionCtx<'_>,
) -> anyhow::Result<Option<ResponsePhaseOutcome>> {
    let agent = services.agent;
    let resp = &mut *ctx.resp;
    let emitter = ctx.emitter;
    let task_id = ctx.task_id;
    let session_id = ctx.session_id;
    let user_text = ctx.user_text;
    let iteration = ctx.iteration;
    let task_start = ctx.task_start;
    let learning_ctx = &mut *ctx.learning_ctx;
    let pending_system_messages = &mut *ctx.pending_system_messages;
    let mut tool_defs = std::mem::take(ctx.tool_defs);
    let base_tool_defs = &*ctx.base_tool_defs;
    let available_capabilities = &*ctx.available_capabilities;
    let policy_bundle = &mut *ctx.policy_bundle;
    let llm_provider = ctx.llm_provider.clone();
    let llm_router = ctx.llm_router.clone();
    let mut model = ctx.model.clone();
    let channel_ctx = ctx.channel_ctx.clone();
    let user_role = ctx.user_role;
    let total_successful_tool_calls = ctx.total_successful_tool_calls;
    let mut stall_count = *ctx.stall_count;
    let mut consecutive_clean_iterations = *ctx.consecutive_clean_iterations;
    let mut deferred_no_tool_streak = *ctx.deferred_no_tool_streak;
    let mut deferred_no_tool_model_switches = *ctx.deferred_no_tool_model_switches;
    let mut fallback_expanded_once = *ctx.fallback_expanded_once;
    let mut empty_response_retry_used = *ctx.empty_response_retry_used;
    let mut empty_response_retry_pending = *ctx.empty_response_retry_pending;
    let mut empty_response_retry_note = ctx.empty_response_retry_note.clone();
    let mut identity_prefill_text = ctx.identity_prefill_text.clone();
    let mut pending_background_ack = std::mem::take(ctx.pending_background_ack);
    let mut pending_external_action_ack = std::mem::take(ctx.pending_external_action_ack);
    let mut require_file_recheck_before_answer = *ctx.require_file_recheck_before_answer;
    let mut completion_progress = ctx.completion_progress.clone();
    let mut validation_state = ctx.validation_state.clone();
    let mut terminal_cause = None;
    let turn_context = ctx.turn_context;
    let execution_requirement = ctx.execution_requirement;
    // Execution obligations are lifecycle invariants, not model-quality
    // heuristics. Autonomous models choose their own strategy, but a first-pass
    // text denial is not evidence that an actionable request was resolved.
    let execution_tool_recovery = execution_requirement.requires_execution();
    let force_text_allowed = completion_contract_allows_force_text(
        &turn_context.completion_contract,
        &completion_progress,
    );
    let mut force_text_response = *ctx.force_text_response && force_text_allowed;
    let mut force_text_fast_path_accepted = false;
    let execution_state = &mut *ctx.execution_state;
    #[cfg(feature = "computer_use")]
    let computer_use_pin_active =
        crate::agent::computer_use::task_has_computer_use_pin(task_id).await;
    #[cfg(not(feature = "computer_use"))]
    let computer_use_pin_active = false;

    let outcome: anyhow::Result<Option<ResponsePhaseOutcome>> = async {
        // === NATURAL COMPLETION: No tool calls ===
        if resp.tool_calls.is_empty() {
        let mut reply = resp
            .content
            .clone()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_default();
        // Snapshot of what the MODEL authored this iteration. Recovery paths
        // below may replace `reply` with daemon-built text (tool-output
        // excerpts, fallback summaries); the tool-output-paste guard must only
        // bounce model-authored pastes — a daemon-built reply is deliberate,
        // already policy-gated at its build site.
        let model_authored_reply = reply.clone();

        let missing_response_fields = turn_context
            .completion_contract
            .required_response_fields
            .iter()
            .filter(|field| {
                !reply
                    .to_lowercase()
                    .contains(&field.trim().to_lowercase())
            })
            .cloned()
            .collect::<Vec<_>>();
        let context_floor_result = emitter.session_context_boundary().await;
        let context_floor_event_id = context_floor_result.as_ref().ok().copied().flatten();
        let outstanding_requirement_indices = turn_context
            .completion_contract
            .evidence_requirements
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                completion_progress
                    .evidence_obligation_ids
                    .get(index)
                    .filter(|obligation_id| {
                        completion_progress.proof_graph.state(obligation_id)
                            != Some(crate::execution_graph::ExecutionNodeState::Satisfied)
                    })
                    .map(|_| index)
            })
            .collect::<Vec<_>>();
        agent
            .emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::FinalizationStateSnapshot,
                "Captured authoritative state before finalization gates".to_string(),
                json!({
                    "condition": "finalization_state_snapshot",
                    "schema_version": 1,
                    "candidate_source": "current_model_response",
                    "candidate_chars": reply.chars().count(),
                    "contract_scope_task_id": turn_context.completion_contract.scope_task_id,
                    "contract_adopted_from_task_ids": turn_context.completion_contract.adopted_from_task_ids,
                    "completion_task_kind": format!("{:?}", turn_context.completion_contract.task_kind).to_lowercase(),
                    "context_floor_event_id": context_floor_event_id,
                    "context_floor_available": context_floor_result.is_ok(),
                    "required_response_fields": turn_context.completion_contract.required_response_fields,
                    "missing_response_fields": missing_response_fields,
                    "primary_target_hint": turn_context.completion_contract.primary_target_hint(),
                    "evidence_requirements_total": turn_context.completion_contract.evidence_requirements.len(),
                    "evidence_requirements_satisfied": completion_progress.satisfied_evidence_requirements(),
                    "outstanding_requirement_indices": outstanding_requirement_indices,
                    "selected_receipt_ids": completion_progress.satisfying_receipt_ids(),
                    "verification_pending": completion_progress.verification_pending,
                    "mutation_attempt_count": completion_progress.mutation_attempt_count,
                    "indeterminate_mutation_count": completion_progress.indeterminate_mutation_count,
                    "mutation_count": completion_progress.mutation_count,
                    "observation_count": completion_progress.observation_count,
                    "verification_count": completion_progress.verification_count,
                    "verification_attempt_count": completion_progress.verification_attempt_count,
                    "verification_block_count": completion_progress.verification_block_count,
                    "decision": "pending_ordered_finalization_gates",
                    "rejection_reason": Value::Null,
                }),
            )
            .await;
        if !missing_response_fields.is_empty() {
            if completion_progress.response_contract_retry_count == 0 {
                completion_progress.response_contract_retry_count = 1;
                pending_system_messages.push(SystemDirective::OutputContractIncomplete {
                    missing_fields: missing_response_fields.clone(),
                });
                agent
                    .emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Draft omitted grounded response-contract fields".to_string(),
                        json!({
                            "condition": "response_contract_incomplete",
                            "contract_scope_task_id": turn_context.completion_contract.scope_task_id,
                            "contract_adopted_from_task_ids": turn_context.completion_contract.adopted_from_task_ids,
                            "missing_fields": missing_response_fields,
                            "retry": 1,
                        }),
                    )
                    .await;
                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }

            validation_state.record_failure(ValidationFailure::SuccessCriteriaUnmatched);
            let unavailable = missing_response_fields
                .iter()
                .map(|field| format!("{}: unavailable in the generated answer", field.trim()))
                .collect::<Vec<_>>()
                .join("\n");
            reply = if reply.trim().is_empty() {
                unavailable
            } else {
                format!("{unavailable}\n\n{reply}")
            };
        }

        // A mandate task lead's prose is never the authoritative decision.
        // Give a missing typed commit one bounded correction turn, then fail
        // closed so background finalization can safely schedule a retry.
        let mandate_commit_state = mandate_decision_commit_state(agent).await?;
        let decision_tool_available = tool_defs
            .iter()
            .any(|definition| tool_definition_is_named(definition, "manage_mandates"));
        match mandate_decision_completion_action(
            mandate_commit_state,
            completion_progress.mandate_decision_retry_count,
            decision_tool_available,
        ) {
            MandateDecisionCompletionAction::Allow => {}
            MandateDecisionCompletionAction::Retry => {
                completion_progress.mandate_decision_retry_count += 1;
                force_text_response = false;
                stall_count = 0;
                consecutive_clean_iterations = 0;
                pending_system_messages.push(SystemDirective::MandateDecisionCommitRequired);
                agent
                    .emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Blocked mandate prose completion without a durable decision".to_string(),
                        json!({
                            "condition": "mandate_decision_commit_missing",
                            "goal_run_id": agent
                                .mandate_execution
                                .as_ref()
                                .map(|fence| fence.goal_run_id.as_str()),
                            "retry": completion_progress.mandate_decision_retry_count,
                        }),
                    )
                    .await;
                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }
            MandateDecisionCompletionAction::Reject => {
                let error_message = match mandate_commit_state {
                    MandateDecisionCommitState::Invalid => {
                        "Mandate deliberator found a decision for the current run with mismatched immutable identity."
                    }
                    _ => {
                        "Mandate deliberator returned without committing an exact durable decision after its bounded correction turn."
                    }
                }
                .to_string();
                agent
                    .emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Rejected mandate prose completion without a durable decision".to_string(),
                        json!({
                            "condition": "mandate_decision_commit_rejected",
                            "goal_run_id": agent
                                .mandate_execution
                                .as_ref()
                                .map(|fence| fence.goal_run_id.as_str()),
                            "commit_state": format!("{mandate_commit_state:?}"),
                            "retries_used": completion_progress.mandate_decision_retry_count,
                            "decision_tool_available": decision_tool_available,
                        }),
                    )
                    .await;
                agent
                    .emit_task_end(
                        emitter,
                        task_id,
                        TaskStatus::Failed,
                        TaskOutcome::Failed,
                        task_start,
                        iteration,
                        learning_ctx.tool_calls.len(),
                        Some(error_message.clone()),
                        None,
                    )
                    .await;
                learning_ctx.completed_naturally = false;
                learning_ctx.task_outcome = Some(TaskOutcome::Failed);
                return Ok(Some(ResponsePhaseOutcome::Return(Err(anyhow::anyhow!(
                    error_message
                )))));
            }
        }

        // If we used an identity-attack prefill, prepend it so the user
        // sees the full decline (the API only returns continuation tokens).
        let used_identity_prefill = identity_prefill_text.is_some();
        if let Some(ref prefill) = identity_prefill_text {
            if reply.is_empty() {
                reply = prefill.clone();
            } else {
                reply = format!("{} {}", prefill, reply.trim_start());
            }
            identity_prefill_text = None;
        }

        // Deterministic cross-model behavior: once a long-running tool detaches
        // to background, do not rely on model compliance for the handoff text.
        if agent.depth == 0 {
            if let Some(background_ack) = pending_background_ack.take() {
                info!(
                    session_id,
                    iteration, "Background detach acknowledgement enforced"
                );
                let assistant_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "assistant".to_string(),
                    content: Some(background_ack.clone()),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.4,
                    ..Message::runtime_defaults()
                };
                agent
                    .append_assistant_message_with_event(
                        emitter,
                        &assistant_msg,
                        "system",
                        None,
                        None,
                    )
                    .await?;

                return Ok(Some(ResponsePhaseOutcome::Return(Ok(background_ack))));
            }
        }

        let has_uncorrected = execution_state.has_uncorrected_failed_external_mutations();
        if agent.depth == 0
            && !completion_verification_still_required(
                turn_context,
                &completion_progress,
                has_uncorrected,
            )
            && should_recover_completion_from_tool_output(
                &reply,
                agent.depth,
                total_successful_tool_calls,
            )
        {
            // Only use the external-action ack for truly empty replies.
            // For low-signal but non-empty replies (e.g., "Done."), the
            // ack may still be a better outcome — but for compound tasks
            // the LLM's reply often carries valuable content (memory
            // recall, explanations) that the ack would obliterate because
            // it only echoes the last tool result.
            let reply_is_truly_empty = reply.trim().is_empty();
            if reply_is_truly_empty {
                if let Some(external_action_ack) = pending_external_action_ack.take() {
                    info!(
                        session_id,
                        iteration,
                        "Successful external-action acknowledgement enforced (empty reply)"
                    );
                    reply = external_action_ack;
                }
            }
        }

        if agent.depth == 0
            && force_text_response
            && learning_ctx
                .tool_calls
                .iter()
                .any(|call| call.starts_with("send_file("))
            && (reply.trim().is_empty() || is_low_signal_task_lead_reply(&reply))
        {
            reply = super::stopping_phase::send_file_completion_reply().to_string();
            info!(
                session_id,
                iteration, "Force-text send_file completion upgraded to shared closeout"
            );
        }

        // Force-text fast-path: when the model can't use tools, all guards
        // that require tool execution (file-recheck, tool-required, deferred-
        // action) are pointless — they would block the reply and return
        // ContinueLoop, but the next iteration strips tools again, creating
        // a deadlock.  Skip directly to completion.  If the reply is empty or
        // low-signal, upgrade it to an activity summary.
        if force_text_response
            && agent.depth == 0
            && total_successful_tool_calls >= 3
            && !completion_verification_still_required(
                turn_context,
                &completion_progress,
                has_uncorrected,
            )
        {
            if reply.trim().is_empty()
                || is_low_signal_task_lead_reply(&reply)
                || looks_like_deferred_action_response(&reply)
                || looks_like_recovery_message_with_trivial_content(&reply)
            {
                let actions: Vec<&str> =
                    learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                if !actions.is_empty() {
                    let candidate =
                        latest_task_tool_result_for_completion(agent, session_id, task_id, 2500)
                            .await;
                    reply = build_completion_fallback_reply(
                        candidate.as_ref(),
                        &actions,
                        learning_ctx.tool_calls.len(),
                    );
                }
            }
            require_file_recheck_before_answer = false;
            force_text_fast_path_accepted = true;
            info!(
                session_id,
                iteration,
                total_successful_tool_calls,
                reply_len = reply.len(),
                "Force-text fast-path: bypassing all tool-requiring guards"
            );
            // Fall through to the normal completion path (sanitize + return)
        } else if should_enforce_no_tool_text_when_tools_required(
            &reply,
            execution_tool_recovery,
            learning_ctx
                .tool_calls
                .len()
                .saturating_add(validation_state.failed_checks.len()),
            agent.depth,
        ) {
            if tool_defs.is_empty() || force_text_response {
                if !force_text_response {
                    // Only show the "no tools available" message when tools are genuinely
                    // absent. In force-text mode the model already has a reply — let it through.
                    reply = "I can't complete that request in this context because it requires running tools, but no tools are currently available. Please retry in a tool-enabled context."
                            .to_string();
                }
                warn!(
                    session_id,
                    iteration,
                    force_text_response,
                    "Tool-required response bypassed: tools unavailable or force-text active"
                );
            } else {
                deferred_no_tool_streak = deferred_no_tool_streak.saturating_add(1);
                stall_count = 0;
                consecutive_clean_iterations = 0;

                // Early acceptance: after enough retries, if the model's text is
                // substantive (not just "I'll do X"), accept it instead of looping
                // forever.  This prevents stalls on queries the intent gate
                // classified as needing tools but the model can answer directly
                // (e.g., "Tell me a joke in Spanish", "List your capabilities").
                if deferred_no_tool_streak >= DEFERRED_NO_TOOL_ACCEPT_THRESHOLD
                    && is_substantive_text_response(&reply, 15)
                {
                    info!(
                            session_id,
                            iteration,
                            deferred_no_tool_streak,
                            reply_len = reply.len(),
                            "Accepting substantive text-only response after repeated tool-required retries"
                        );
                    deferred_no_tool_streak = 0;
                    // Fall through to normal completion path
                } else {
                    pending_system_messages
                        .push(SystemDirective::ExecutionResolutionEvidenceRequired);
                    agent.emit_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::IntentGate,
                            "Execution obligation enforced: evidence-seeking pass required before text completion"
                                .to_string(),
                            json!({
                                "condition":"tools_required_no_tool_response",
                                "reply_len": reply.len(),
                                "deferred_no_tool_streak": deferred_no_tool_streak
                            }),
                        )
                        .await;
                    warn!(
                        session_id,
                        iteration,
                        deferred_no_tool_streak,
                        "Blocked no-tool completion because current turn requires tools"
                    );
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }
            }
        }

        let has_tool_attempts = !learning_ctx.tool_calls.is_empty();
        let assistant_claimed_mutation = claims_completed_side_effect(&reply);
        let assistant_claimed_delegation = claims_delegation_started(&reply);
        let typed_mutation_fulfilled =
            mutation_contract_fulfilled(&turn_context.completion_contract, &completion_progress);
        let authored_artifact_undelivered = authored_artifact_still_needs_delivery_recovery(
            &turn_context.completion_contract,
            &completion_progress,
        );

        // A local artifact is progress toward a delivery, not delivery itself.
        // Give this recoverable state one explicit strategy-pivot pass even
        // when the model honestly admits the converter failed. Missing-file
        // requests never enter this branch because they have no local write.
        if authored_artifact_undelivered
            && !completion_progress.undelivered_artifact_recovery_attempted
            && agent.depth == 0
            && !force_text_response
            && !tool_defs.is_empty()
        {
            completion_progress.mark_undelivered_artifact_recovery_attempted();
            execution_state.record_validation_round();
            consecutive_clean_iterations = 0;
            pending_system_messages.push(SystemDirective::UndeliveredArtifactRecoveryRequired);
            agent
                .emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::PostExecutionValidation,
                    "Blocked completion: authored artifact was not delivered".to_string(),
                    json!({
                        "condition": "authored_artifact_undelivered",
                        "required_mutation_effects": turn_context.completion_contract.required_mutation_effects,
                        "observed_mutation_effects": completion_progress.observed_mutation_effects,
                    }),
                )
                .await;
            warn!(
                session_id,
                iteration,
                "Authored artifact remains undelivered — one recovery pass before reporting"
            );
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }
        let unbacked_mutation_claim = assistant_claimed_mutation
            && if turn_context.completion_contract.expects_mutation {
                !typed_mutation_fulfilled
            } else {
                completion_progress.mutation_count == 0
            };
        let concrete_mutation_required =
            super::completion_checks::contract_has_concrete_mutation_target(
                &turn_context.completion_contract,
            );
        let unfulfilled_concrete_mutation = concrete_mutation_required && !typed_mutation_fulfilled;
        // The completion contract is an intent hint, not proof of a required
        // side effect. Hard enforcement starts when the assistant itself
        // claims that a mutation happened; that claim can be checked exactly
        // against mutation semantics recorded in the tool ledger.
        let mutation_gate_relevant = assistant_claimed_mutation || concrete_mutation_required;
        if mutation_gate_relevant {
            let mutation_gate_block_condition = !force_text_fast_path_accepted
                && !force_text_response
                && agent.depth == 0
                && (unbacked_mutation_claim
                    || (unfulfilled_concrete_mutation && has_tool_attempts))
                && completion_progress.mutation_claim_nudge_count == 0;
            let zero_tool_claim_condition = !has_tool_attempts && unbacked_mutation_claim;
            let (outcome, skip_reason) = if force_text_fast_path_accepted {
                ("skipped_force_text_fast_path", Some("force_text_fast_path"))
            } else if force_text_response {
                ("skipped_force_text_response", Some("force_text_response"))
            } else if zero_tool_claim_condition {
                ("blocked_claimed_mutation_without_tool", None)
            } else if agent.depth != 0 {
                ("skipped_non_root_agent", Some("non_root_agent"))
            } else if typed_mutation_fulfilled {
                ("passed", None)
            } else if mutation_gate_block_condition {
                ("blocked_unsatisfied_after_tools", None)
            } else {
                ("pending_no_mutation_tool", None)
            };
            let metadata = json!({
                "condition": "expects_mutation_gate_evaluated",
                "expects_mutation": turn_context.completion_contract.expects_mutation,
                "assistant_claimed_mutation": assistant_claimed_mutation,
                "tool_calls_count": resp.tool_calls.len(),
                "mutation_tool_calls_count": completion_progress.mutation_count,
                "required_mutation_effects": turn_context.completion_contract.required_mutation_effects,
                "observed_mutation_effects": completion_progress.observed_mutation_effects,
                "total_successful_tool_calls": total_successful_tool_calls,
                "has_tool_attempts": has_tool_attempts,
                "outcome": outcome,
                "skip_reason": skip_reason,
                "stall_count": stall_count,
            });
            if matches!(
                outcome,
                "blocked_claimed_mutation_without_tool" | "blocked_unsatisfied_after_tools"
            ) {
                agent
                    .with_harness_eval(|eval| eval.record_post_exec_validation_failure())
                    .await;
                agent
                    .emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Mutation expectation gate flagged an unsafe completion".to_string(),
                        metadata,
                    )
                    .await;
            } else {
                agent
                    .emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Mutation expectation gate evaluated".to_string(),
                        metadata,
                    )
                    .await;
            }
        }

        // GUI coordinate-click verification gate: a coordinate click targets a
        // raw point with no element identity, so the harness cannot tell a hit
        // from a miss — only a deliberate follow-up observation can. Don't let a
        // GUI success claim ship while such a click is unconfirmed (2026-07-12:
        // an unverified coordinate click at a guessed heart position was
        // reported as a successful Like). Re-fires each iteration UNTIL the
        // model runs a verifying get_app_state/screenshot (which clears the
        // flag) — the block IS the "verify before you claim" invariant, not a
        // one-shot nudge; the loop is bounded because any observation clears it.
        #[cfg(feature = "computer_use")]
        if computer_use_pin_active
            && agent.depth == 0
            && assistant_claimed_mutation
            && !force_text_response
            && crate::agent::computer_use::task_has_unverified_coordinate_click(task_id).await
        {
            consecutive_clean_iterations = 0;
            pending_system_messages.push(SystemDirective::GuiCoordinateClickUnverified);
            agent
                .emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::PostExecutionValidation,
                    "GUI success claim blocked: unverified coordinate click".to_string(),
                    json!({ "condition": "gui_coordinate_click_unverified" }),
                )
                .await;
            warn!(
                session_id,
                iteration, "Blocked GUI success claim: unverified coordinate click outstanding"
            );
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }

        if agent.depth == 0
            && total_successful_tool_calls == 0
            && execution_requirement.requires_execution()
            && !used_identity_prefill
            && looks_like_deferred_action_response(&reply)
            && !is_substantive_text_response(&reply, 200)
            && agent
                .supervision_gate_enforced(
                    "deferred_action_block",
                    &model,
                    emitter,
                    task_id,
                    iteration,
                )
                .await
        {
            if tool_defs.is_empty() {
                warn!(
                    session_id,
                    iteration,
                    "Deferred-action reply with no available tools; returning explicit blocker"
                );
                reply = "I wasn't able to complete that request because no execution tools are available in this context. Please try again in a context with tool access."
                        .to_string();
            } else if deferred_no_tool_streak >= DEFERRED_NO_TOOL_ACCEPT_THRESHOLD
                && is_substantive_text_response(&reply, 50)
            {
                info!(
                        session_id,
                        iteration,
                        deferred_no_tool_streak,
                        reply_len = reply.len(),
                        "Accepting substantive text-only response after repeated deferred-no-tool retries"
                    );
                deferred_no_tool_streak = 0;
            } else {
                deferred_no_tool_streak = deferred_no_tool_streak.saturating_add(1);
                agent
                    .with_harness_eval(|eval| eval.record_deferred_no_tool_event())
                    .await;
                consecutive_clean_iterations = 0;
                pending_system_messages.push(SystemDirective::DeferredToolCallRequired);
                warn!(
                    session_id,
                    iteration,
                    deferred_no_tool_streak,
                    "Deferred-action reply before first tool call; continuing loop"
                );

                if deferred_no_tool_streak >= DEFERRED_NO_TOOL_SWITCH_THRESHOLD
                    && deferred_no_tool_model_switches < MAX_DEFERRED_NO_TOOL_MODEL_SWITCHES
                    && !computer_use_pin_active
                {
                    if let Some(next_model) = agent
                        .pick_fallback_excluding(&model, &[], llm_router.as_ref())
                        .await
                    {
                        info!(
                            session_id,
                            iteration,
                            from_model = %model,
                            to_model = %next_model,
                            "Deferred/no-tool recovery: switching model for one retry window"
                        );
                        model = next_model;
                        deferred_no_tool_model_switches += 1;
                        POLICY_METRICS
                            .deferred_no_tool_model_switch_total
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }

                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }
        }

        let false_capability_denial = looks_like_false_capability_denial_after_tool_success(&reply);

        if false_capability_denial
            && agent
                .supervision_gate_enforced_with_context(
                    "false_capability_denial_rewrite",
                    &model,
                    emitter,
                    task_id,
                    iteration,
                    json!({
                        "successful_tool_calls": total_successful_tool_calls,
                        "reply_preview": reply.chars().take(180).collect::<String>(),
                    }),
                )
                .await
        {
            if !force_text_response && !tool_defs.is_empty() && stall_count == 0 {
                stall_count = stall_count.saturating_add(1);
                consecutive_clean_iterations = 0;
                pending_system_messages.push(SystemDirective::SuccessfulToolEvidenceMustBeUsed);
                agent
                    .with_harness_eval(|eval| eval.record_stall_guard())
                    .await;
                warn!(
                    session_id,
                    iteration,
                    reply_preview = %reply.chars().take(180).collect::<String>(),
                    "Rejected completion that denied live capabilities after successful tool use"
                );
                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }

            let mut recovered = false;
            let candidate =
                latest_task_tool_result_for_completion(agent, session_id, task_id, 2500).await;
            if let Some(candidate) = candidate.as_ref() {
                if candidate.tool_name == "send_file" {
                    reply = super::stopping_phase::send_file_completion_reply().to_string();
                    recovered = true;
                } else if let Some(tool_reply) = build_tool_output_completion_reply(
                    &candidate.tool_name,
                    &candidate.tool_output,
                    candidate.artifact_delivered,
                ) {
                    reply = tool_reply;
                    recovered = true;
                } else if let Some(tool_reply) = build_structured_tool_output_completion_reply(
                    &candidate.tool_name,
                    &candidate.tool_output,
                    candidate.artifact_delivered,
                ) {
                    reply = tool_reply;
                    recovered = true;
                }
            }
            if !recovered && !learning_ctx.tool_calls.is_empty() {
                let actions: Vec<&str> = learning_ctx
                    .tool_calls
                    .iter()
                    .map(|call| call.as_str())
                    .collect();
                reply = build_completion_fallback_reply(
                    candidate.as_ref(),
                    &actions,
                    learning_ctx.tool_calls.len(),
                );
            }
            info!(
                session_id,
                iteration,
                recovered,
                "Recovered false capability-denial completion after successful tools"
            );
        }

        let low_signal_completion = is_low_signal_task_lead_reply(&reply);
        let idle_reengagement_completion = looks_like_idle_reengagement_reply(&reply);
        let was_truly_empty = reply.trim().is_empty();
        if !force_text_fast_path_accepted
            && (should_recover_completion_from_tool_output(
                &reply,
                agent.depth,
                total_successful_tool_calls,
            ) || idle_reengagement_completion)
        {
            let mut recovered = false;
            let mut candidate_requires_synthesis = false;
            let mut synthesis_retry_scheduled = false;
            let candidate =
                latest_task_tool_result_for_completion(agent, session_id, task_id, 2500).await;
            if let Some(candidate) = candidate.as_ref() {
                candidate_requires_synthesis = tool_output_requires_final_synthesis(
                    &candidate.tool_name,
                    &candidate.tool_output,
                );
                if candidate.tool_name == "send_file" {
                    reply = super::stopping_phase::send_file_completion_reply().to_string();
                    recovered = true;
                    info!(
                        session_id,
                        iteration,
                        "Recovered completion reply after send_file with shared closeout"
                    );
                } else if candidate.tool_name == "read_file"
                    && learning_ctx.tool_calls.len() > 1
                    && !candidate.artifact_delivered
                {
                    // When the latest tool is read_file and there were multiple tool
                    // calls, the activity summary is more useful than a raw file dump.
                    // Build the activity summary directly here instead of relying on
                    // the fallback branch (which is gated on !was_truly_empty and
                    // would be skipped when the LLM returned a truly empty response).
                    let actions: Vec<&str> =
                        learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                    reply = build_completion_fallback_reply(
                        Some(candidate),
                        &actions,
                        learning_ctx.tool_calls.len(),
                    );
                    if !reply.is_empty() {
                        recovered = true;
                    }
                    info!(
                        session_id,
                        iteration,
                        tool_call_count = learning_ctx.tool_calls.len(),
                        recovered,
                        "Built activity summary instead of read_file output recovery"
                    );
                } else if !tool_output_paste_recovery_allowed(
                    turn_context.completion_contract.expects_mutation,
                    usize::from(mutation_contract_fulfilled(
                        &turn_context.completion_contract,
                        &completion_progress,
                    )),
                ) {
                    // An unfulfilled mutation contract: never substitute a
                    // tool-output paste for the missing action ("Send me X" →
                    // empty reply must not ship an `ls` of personal files).
                    // Fall through to the activity-summary fallback below.
                    info!(
                        session_id,
                        iteration,
                        "Skipping tool-output paste recovery: mutation contract unfulfilled"
                    );
                } else if let Some(tool_reply) = build_tool_output_completion_reply(
                    &candidate.tool_name,
                    &candidate.tool_output,
                    candidate.artifact_delivered,
                ) {
                    // When there were multiple successful tool calls but the
                    // latest tool output is trivially uninformative (e.g.,
                    // "(no output)" from a memory tool), prefer the activity
                    // summary which lists what was actually accomplished.
                    let tool_output_trivial = is_trivial_tool_output(candidate.tool_output.trim());
                    if tool_output_trivial && learning_ctx.tool_calls.len() > 2 {
                        info!(
                            session_id,
                            iteration,
                            tool = %candidate.tool_name,
                            tool_call_count = learning_ctx.tool_calls.len(),
                            "Latest tool output trivial with multiple tool calls — deferring to activity summary"
                        );
                        // Don't mark as recovered — let the activity summary
                        // branch below handle it.
                    } else {
                        reply = tool_reply;
                        recovered = true;
                        info!(
                            session_id,
                            iteration,
                            low_signal_completion,
                            idle_reengagement_completion,
                            "Recovered completion reply from latest tool output"
                        );
                    }
                } else if candidate_requires_synthesis {
                    if !empty_response_retry_used {
                        empty_response_retry_used = true;
                        empty_response_retry_pending = true;
                        empty_response_retry_note =
                            Some("structured_tool_output_requires_synthesis".to_string());
                        pending_system_messages
                            .push(structured_result_synthesis_directive(candidate));
                        synthesis_retry_scheduled = true;
                    } else if let Some(tool_reply) = build_structured_tool_output_completion_reply(
                        &candidate.tool_name,
                        &candidate.tool_output,
                        candidate.artifact_delivered,
                    ) {
                        reply = tool_reply;
                        recovered = true;
                    }
                    if !recovered {
                        reply.clear();
                        info!(
                            session_id,
                            iteration,
                            tool = %candidate.tool_name,
                            retry_scheduled = synthesis_retry_scheduled,
                            "Deferring structured tool output to synthesis recovery or deterministic fallback"
                        );
                    }
                }
            }
            // If tool output was trivial/empty and the LLM returned a truly empty
            // response (not just low-signal), don't build an activity summary —
            // leave reply empty so the empty-response retry mechanism kicks in
            // and gives the model another chance to complete the task properly.
            if !recovered
                && !was_truly_empty
                && !learning_ctx.tool_calls.is_empty()
                && !synthesis_retry_scheduled
            {
                let actions: Vec<&str> =
                    learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                reply = build_completion_fallback_reply(
                    candidate.as_ref(),
                    &actions,
                    learning_ctx.tool_calls.len(),
                );
                info!(
                        session_id,
                        iteration,
                        tool_call_count = learning_ctx.tool_calls.len(),
                        candidate_requires_synthesis,
                        "Built deterministic completion fallback from latest tool result or activity summary"
                    );
            } else if !recovered && was_truly_empty {
                info!(
                        session_id,
                        iteration,
                        "Empty LLM response with no recoverable tool output — deferring to empty-response retry"
                    );
            }

            // When synthesis retry was scheduled above (structured tool
            // output like web_fetch/web_search needs a follow-up LLM call
            // to produce a human-readable summary), continue the loop
            // immediately so the model processes the synthesis directive.
            // Without this, the empty reply would fall through to the
            // deterministic "couldn't recover" fallback — which is the
            // wrong outcome when the data IS available but needs synthesis.
            if synthesis_retry_scheduled && empty_response_retry_pending {
                info!(
                    session_id,
                    iteration,
                    "Synthesis retry scheduled — continuing loop for structured output synthesis"
                );
                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }
        }

        if reply.is_empty()
            && agent.depth == 0
            && force_text_response
            && learning_ctx
                .tool_calls
                .iter()
                .any(|call| call.starts_with("send_file("))
        {
            reply = super::stopping_phase::send_file_completion_reply().to_string();
            info!(
                session_id,
                iteration, "Recovered empty force-text completion with shared send_file closeout"
            );
        } else if reply.is_empty() && total_successful_tool_calls > 0 && agent.depth == 0 {
            reply = "I executed the requested tools, but no usable output snapshot was retained, so I couldn't verify the result.".to_string();
            info!(
                session_id,
                iteration, "Tool execution completed but no output snapshot was available"
            );
        }

        if reply.is_empty() {
            // User-facing empty response: never return silence.
            // Retry once; if the model remains empty, return an explicit fallback.
            if !is_trigger_session(session_id) {
                if !empty_response_retry_used {
                    empty_response_retry_used = true;
                    empty_response_retry_pending = true;
                    empty_response_retry_note = resp
                        .response_note
                        .as_deref()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string);

                    stall_count += 1;
                    consecutive_clean_iterations = 0;

                    // Retry once with a stronger model profile to avoid repeated empties,
                    // unless the user explicitly pinned a model override.
                    let is_override = match tokio::time::timeout(
                        Duration::from_secs(2),
                        agent.model_override.read(),
                    )
                    .await
                    {
                        Ok(guard) => *guard,
                        Err(_) => {
                            warn!(
                                        session_id,
                                        iteration,
                                        "Timed out acquiring model_override lock during empty-response recovery"
                                    );
                            false
                        }
                    };
                    if !is_override
                        && agent.trust_tier_for_model(&model)
                            == crate::agent::trust_tier::ModelTrustTier::Guided
                    {
                        let reason = format!("empty_response(iter={},model={})", iteration, model);
                        if policy_bundle.policy.escalate(reason.clone()) {
                            POLICY_METRICS
                                .escalation_total
                                .fetch_add(1, Ordering::Relaxed);
                            if let Some(ref router) = llm_router {
                                let next_model = router
                                    .select_for_profile(policy_bundle.policy.model_profile)
                                    .to_string();
                                if next_model != model && !computer_use_pin_active {
                                    info!(
                                        session_id,
                                        iteration,
                                        reason = %reason,
                                        from_model = %model,
                                        to_model = %next_model,
                                        "Empty-response recovery: escalated model for retry"
                                    );
                                    model = next_model;
                                }
                            }
                        }
                    }

                    info!(
                        session_id,
                        iteration,
                        response_note = ?resp.response_note,
                        "Empty-response recovery: issuing one retry before fallback"
                    );

                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }

                let response_note = if empty_response_retry_pending {
                    resp.response_note
                        .as_deref()
                        .or(empty_response_retry_note.as_deref())
                } else {
                    resp.response_note.as_deref()
                };
                let fallback = build_empty_response_fallback(response_note);
                info!(
                    session_id,
                    iteration,
                    response_note = ?resp.response_note,
                    retry_response_note = ?empty_response_retry_note,
                    "Agent completed with no work done — LLM returned empty with tools available"
                );
                let assistant_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "assistant".to_string(),
                    content: Some(fallback.clone()),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.5,
                    ..Message::runtime_defaults()
                };
                agent
                    .append_assistant_message_with_event(
                        emitter,
                        &assistant_msg,
                        &model,
                        resp.usage.as_ref().map(|u| u.input_tokens),
                        resp.usage.as_ref().map(|u| u.output_tokens),
                    )
                    .await?;

                let has_unrecovered_errors = learning_ctx.has_unrecovered_model_error();
                let outcome = TaskOutcomeDerivation::from_completion_state(
                    &validation_state,
                    execution_state,
                    ctx.completion_progress,
                    &turn_context.completion_contract,
                    response_has_user_value(&fallback, total_successful_tool_calls),
                    has_unrecovered_errors,
                    None,
                )
                .derive_outcome();
                agent
                    .emit_task_end(
                        emitter,
                        task_id,
                        TaskStatus::Completed,
                        outcome,
                        task_start,
                        iteration,
                        learning_ctx.tool_calls.len(),
                        None,
                        Some(fallback.chars().take(200).collect()),
                    )
                    .await;
                learning_ctx.completed_naturally = true;
                learning_ctx.task_outcome = Some(outcome);
                if agent.mandate_execution.is_none() {
                    let learning_ctx_for_task = learning_ctx.clone();
                    let state = agent.state.clone();
                    tokio::spawn(async move {
                        if let Err(e) =
                            post_task::process_learning(&state, learning_ctx_for_task).await
                        {
                            warn!("Learning failed: {}", e);
                        }
                    });
                }

                return Ok(Some(ResponsePhaseOutcome::Return(Ok(fallback))));
            }
            // First iteration or sub-agent — stay silent
            info!(session_id, iteration, "Agent completed with empty response");
            return Ok(Some(ResponsePhaseOutcome::Return(Ok(String::new()))));
        }

        if require_file_recheck_before_answer {
            if tool_defs.is_empty() || force_text_response {
                warn!(
                        session_id,
                        iteration,
                        force_text_response,
                        "File re-check required but tools unavailable (empty or force-text); clearing guard"
                    );
                // In force-text mode the model can't use tools, so blocking
                // on file re-check is a deadlock. Clear the guard and let
                // the response through.
                require_file_recheck_before_answer = false;
            } else {
                execution_state.record_validation_round();
                validation_state.record_failure(ValidationFailure::ContradictoryEvidence);
                validation_state.note_retry(LoopRepetitionReason::ContradictoryEvidence);
                learning_ctx.record_replay_note(
                        ReplayNoteCategory::ValidationFailure,
                        "contradictory_file_evidence",
                        "Blocked completion because current file evidence contradicted an earlier read."
                            .to_string(),
                        true,
                    );
                learning_ctx.record_replay_note(
                    ReplayNoteCategory::RetryReason,
                    "contradictory_evidence",
                    "Retried because contradictory file evidence required a fresh re-check."
                        .to_string(),
                    true,
                );
                execution_state.mark_persisted_now();
                agent
                    .emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::PostExecutionValidation,
                        "Blocked completion until contradictory file evidence is rechecked"
                            .to_string(),
                        json!({
                            "outcome": ValidationOutcome::VerifyAgain,
                            "reason": "contradictory_file_evidence",
                            "loop_repetition_reason": validation_state.loop_repetition_reason,
                            "target_hint": turn_context.completion_contract.primary_target_hint(),
                            "completed_tool_calls": learning_ctx.tool_calls.len(),
                        }),
                    )
                    .await;
                stall_count = stall_count.saturating_add(1);
                consecutive_clean_iterations = 0;
                pending_system_messages
                    .push(SystemDirective::ContradictoryFileEvidenceRecheckRequired);
                warn!(
                    session_id,
                    iteration,
                    stall_count,
                    "Blocking completion until required file re-check is performed"
                );
                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }
        }

        if completion_verification_still_required(
            turn_context,
            &completion_progress,
            has_uncorrected,
        ) {
            // Handle failed external mutations first — independent of observation contract.
            // Only enter reconciliation if there are *uncorrected* failures (not
            // ones the agent subsequently retried successfully).
            if completion_progress.failed_external_mutation_count > 0 && has_uncorrected {
                let reconciliation_overview = execution_state.build_reconciliation_overview();
                let reconciliation = reconciliation_overview
                        .as_ref()
                        .map(|overview| overview.summary.clone())
                        .or_else(|| execution_state.build_attempt_reconciliation_summary())
                        .unwrap_or_else(|| {
                            "[SYSTEM] External mutation attempt reconciliation: one or more attempts failed."
                                .to_string()
                        });

                // Recovery pass (before any reporting): the model is trying
                // to finish with unfixed failed mutations. Give it ONE
                // evidence-fed chance to fix the failure with a different
                // approach — the immediate ExternalMutationFailed nudge fires
                // per-failure, but an exit-0-with-error-in-output retry breaks
                // the failure streak and the model declares itself done (live
                // repro 2026-07-03, task 2e87a458: python one-liner quoting
                // failure "succeeded" with a traceback in stdout; the model
                // answered instead of pivoting; the user got an honest report
                // when a script-file retry would likely have worked).
                if !completion_progress.external_mutation_recovery_attempted {
                    pending_system_messages.push(
                        SystemDirective::ExternalMutationRecoveryRequired(reconciliation.clone()),
                    );
                    completion_progress.mark_external_mutation_recovery_attempted();
                    execution_state.record_validation_round();
                    warn!(
                        session_id,
                        iteration,
                        "Unfixed failed external mutations at completion — one recovery pass before reporting"
                    );
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }

                // First pass: send the verified reconciliation facts back through the LLM.
                if !completion_progress.external_mutation_reconciliation_attempted {
                    pending_system_messages.push(SystemDirective::OutcomeReconciliation(
                        reconciliation.clone(),
                    ));
                    completion_progress.mark_external_mutation_reconciliation_attempted();
                    execution_state.record_validation_round();
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }

                // Second pass: the reply was generated with reconciliation context.
                // Validate that reply against the ledger.
                if reconciliation_overview.as_ref().is_some_and(|overview| {
                    !reply_acknowledges_outcome_reconciliation(&reply, overview)
                }) {
                    reply = build_outcome_reconciliation_fallback_reply(&reconciliation);
                }

                completion_progress.clear_failed_external_mutation_gate();
                pending_external_action_ack = None;
            }

            // If the standard observation contract is also still pending, handle it
            if completion_progress.verification_pending
                && turn_context.completion_contract.requires_observation
            {
                let has_concrete_progress = super::stopping_progress::has_any_concrete_execution(
                    turn_context,
                    &completion_progress,
                    false,
                    total_successful_tool_calls,
                );
                let outstanding_requirements = completion_progress
                    .outstanding_evidence_requirements(&turn_context.completion_contract)
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>();
                let outstanding_evidence = outstanding_requirements
                    .iter()
                    .map(crate::agent::inquiry::describe_requirement)
                    .collect::<Vec<_>>();
                let visible_tool_names = tool_defs
                    .iter()
                    .filter_map(|definition| {
                        definition
                            .get("function")
                            .and_then(|function| function.get("name"))
                            .and_then(serde_json::Value::as_str)
                    })
                    .collect::<Vec<_>>();
                let candidate_verification_tools =
                    crate::agent::inquiry::candidate_tools_for_requirements(
                        &outstanding_requirements,
                        visible_tool_names.iter().copied(),
                    );
                let visible_evidence_catalog_is_complete = visible_tool_names
                    .iter()
                    .all(|name| crate::agent::inquiry::has_static_evidence_model(name));
                let typed_verification_unavailable = !outstanding_requirements.is_empty()
                    && candidate_verification_tools.is_empty()
                    && visible_evidence_catalog_is_complete;
                let missing_evidence_requirement = if outstanding_evidence.is_empty() {
                    "A successful read-only verification against the requested target or output."
                        .to_string()
                } else {
                    format!(
                        "Compatible evidence is still required for: {}",
                        outstanding_evidence.join("; ")
                    )
                };
                execution_state.record_validation_round();
                validation_state.record_failure(ValidationFailure::VerificationPending);
                execution_state.mark_persisted_now();
                if matches!(
                    execution_state.exhausted_limit(0, task_start.elapsed()),
                    Some(ExecutionBudgetLimit::ValidationRounds)
                ) {
                    validation_state.record_failure(ValidationFailure::BudgetExhausted);
                    learning_ctx.record_replay_note(
                            ReplayNoteCategory::ValidationFailure,
                            "validation_budget_exhausted",
                            "Stopped final verification because the current validation budget was exhausted."
                                .to_string(),
                            true,
                        );
                    let request = if has_concrete_progress {
                        build_reduce_scope_request_with_plan(
                                turn_context,
                                learning_ctx,
                                Some(execution_state),
                                "I used the current validation budget and still do not have a confirmed final result.",
                                "Confirm the narrower scope or exact verification target I should spend the next pass on.",
                                "I will spend the next validation pass on the reduced scope and then report the confirmed outcome.",
                            )
                    } else {
                        let (request, cause) = build_terminal_verification_request(
                                turn_context,
                                learning_ctx,
                                Some(execution_state),
                                false,
                                "I used the current validation budget and still do not have a confirmed final result.",
                                &missing_evidence_requirement,
                                "Start a new attempt with a concrete re-check before reporting the outcome.",
                            );
                        terminal_cause = cause;
                        request
                    };
                    agent.emit_warning_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::PostExecutionValidation,
                            if has_concrete_progress {
                                "Surfacing partial result because validation budget is exhausted"
                            } else {
                                "Failing verification because validation budget was exhausted before concrete progress"
                            }
                            .to_string(),
                            json!({
                                "condition": "validation_budget_exhausted",
                                "outcome": request.outcome.clone(),
                                "approval_state": request.approval_state.clone(),
                                "validation_state": validation_state.clone(),
                                "request": request.clone(),
                                "validation_rounds_used": execution_state.validation_rounds_used,
                                "validation_round_budget": execution_state.budget.max_validation_rounds,
                                "execution_id": execution_state.execution_id,
                                "outstanding_evidence": outstanding_evidence,
                            }),
                        )
                        .await;
                    reply = request.render_user_message();
                    pending_external_action_ack = None;
                } else if completion_progress.verification_attempt_count
                    >= MAX_VERIFICATION_ATTEMPTS
                {
                    // A bounded retry may end in an honest partial result only
                    // when concrete work exists. With no successful execution,
                    // the typed terminal cause must remain a failure.
                    learning_ctx.record_replay_note(
                            ReplayNoteCategory::ValidationFailure,
                            "verification_stall_escape",
                            if has_concrete_progress {
                                "Two compatible verification calls completed without satisfying the obligation; returning an honest partial result."
                            } else {
                                "Two compatible verification calls completed without concrete progress; failing the attempt."
                            }
                            .to_string(),
                            true,
                        );
                    agent.emit_warning_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::PostExecutionValidation,
                            if has_concrete_progress {
                                "Verification retries exhausted — returning an honest partial result"
                            } else {
                                "Verification retries exhausted without concrete progress — failing the attempt"
                            }
                            .to_string(),
                            json!({
                                "verification_block_count": completion_progress.verification_block_count,
                                "verification_attempt_count": completion_progress.verification_attempt_count,
                                "stall_count": stall_count,
                            }),
                        )
                        .await;
                    warn!(
                        session_id,
                        iteration,
                        verification_block_count = completion_progress.verification_block_count,
                        verification_attempt_count = completion_progress.verification_attempt_count,
                        "Verification retries exhausted; suppressing unverified completion claim"
                    );
                    let (request, cause) = build_terminal_verification_request(
                        turn_context,
                        learning_ctx,
                        Some(execution_state),
                        has_concrete_progress,
                        "I completed part of the request, but I could not obtain the required final verification receipt.",
                        &missing_evidence_requirement,
                        "Start a new attempt and run a compatible verification check before reporting success.",
                    );
                    terminal_cause = cause;
                    reply = request.render_user_message();
                    pending_external_action_ack = None;
                } else if tool_defs.is_empty()
                    || force_text_response
                    || typed_verification_unavailable
                {
                    validation_state.note_retry(LoopRepetitionReason::VerificationPending);
                    learning_ctx.record_replay_note(
                            ReplayNoteCategory::ValidationFailure,
                            "verification_unavailable_in_phase",
                            "Verification was still required, but this phase could not run the needed read-only checks."
                                .to_string(),
                            true,
                        );
                        learning_ctx.record_replay_note(
                            ReplayNoteCategory::RetryReason,
                            "verification_pending",
                            "Retried because verification was still pending at completion time."
                                .to_string(),
                            true,
                        );
                    let (request, cause) = build_terminal_verification_request(
                        turn_context,
                        learning_ctx,
                        Some(execution_state),
                        has_concrete_progress,
                        "I completed part of the request, but the final outcome still needs a read-only verification step.",
                        &missing_evidence_requirement,
                        "Start a new attempt when a compatible verification check is available.",
                    );
                    terminal_cause = cause;
                    agent.emit_warning_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::PostExecutionValidation,
                            if has_concrete_progress {
                                "Surfacing partial result because post-execution verification cannot run in this phase"
                            } else {
                                "Failing the attempt because verification cannot run and no concrete work completed"
                            }
                            .to_string(),
                            json!({
                                "outcome": request.outcome.clone(),
                                "approval_state": request.approval_state.clone(),
                                "validation_state": validation_state.clone(),
                                "request": request.clone(),
                                "force_text_response": force_text_response,
                                "tools_available": !tool_defs.is_empty(),
                                "compatible_verification_tools": candidate_verification_tools,
                                "visible_evidence_catalog_is_complete": visible_evidence_catalog_is_complete,
                                "stall_count": stall_count,
                                "outstanding_evidence": outstanding_evidence,
                            }),
                        )
                        .await;
                        warn!(
                            session_id,
                            iteration,
                            stall_count,
                            force_text_response,
                            "Completion evidence remained open with tools unavailable; returning a typed partial or failure"
                        );
                    reply = request.render_user_message();
                    pending_external_action_ack = None;
                } else {
                    validation_state.note_retry(LoopRepetitionReason::VerificationPending);
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::ValidationFailure,
                        "verification_pending",
                        "Blocked completion until the final verification step could run."
                            .to_string(),
                        true,
                    );
                    learning_ctx.record_replay_note(
                        ReplayNoteCategory::RetryReason,
                        "verification_pending",
                        "Retried because verification was still pending at completion time."
                            .to_string(),
                        true,
                    );
                    agent.emit_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::PostExecutionValidation,
                            "Post-execution verification required before completion".to_string(),
                            json!({
                                "outcome": ValidationOutcome::VerifyAgain,
                                "reason": "verification_pending",
                                "loop_repetition_reason": validation_state.loop_repetition_reason,
                                "target_hint": turn_context.completion_contract.primary_target_hint(),
                                "completed_tool_calls": learning_ctx.tool_calls.len(),
                                "verification_pending": completion_progress.verification_pending,
                                "verification_block_count": completion_progress.verification_block_count,
                                "verification_attempt_count": completion_progress.verification_attempt_count,
                            }),
                        )
                        .await;
                    stall_count = stall_count.saturating_add(1);
                    completion_progress.verification_block_count = completion_progress
                        .verification_block_count
                        .saturating_add(1);

                    // A blocked completion is not a verification attempt. Keep
                    // directing the model toward compatible evidence until an
                    // actual attempt budget or the validation-round budget is
                    // exhausted.
                    consecutive_clean_iterations = 0;
                    if outstanding_requirements.is_empty() {
                        pending_system_messages.push(
                            SystemDirective::CompletionVerificationRequired {
                                target_hint: turn_context
                                    .completion_contract
                                    .primary_target_hint(),
                            },
                        );
                    } else {
                        pending_system_messages.push(SystemDirective::InquiryEvidenceRequired {
                            outstanding_needs: outstanding_requirements
                                .iter()
                                .map(crate::agent::inquiry::describe_requirement)
                                .collect(),
                            candidate_tools: candidate_verification_tools,
                        });
                    }
                    warn!(
                        session_id,
                        iteration,
                        stall_count,
                        verification_block_count = completion_progress.verification_block_count,
                        verification_attempt_count = completion_progress.verification_attempt_count,
                        "Blocking completion until request outcome verification is performed"
                    );
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }
            }
        }

        // Evidence-first mutation guard: if the assistant claims a completed
        // side effect but no mutation tool actually ran, nudge the model to
        // perform the action or report the limitation honestly. The inferred
        // completion contract remains advisory and cannot arm this hard gate.
        //
        // An honest admission that the deliverable could not be produced or
        // found is NOT a fabricated completion — blocking it only forces the
        // model to re-generate the same truthful answer (observed live: a
        // "couldn't find that file" reply blocked 3 times, ~5 minutes of
        // re-decode). The gate targets false success claims and content
        // dumps, both of which lack such an admission.
        if !force_text_fast_path_accepted
            && agent.depth == 0
            && (unbacked_mutation_claim || (unfulfilled_concrete_mutation && has_tool_attempts))
            && completion_progress.mutation_claim_nudge_count == 0
            && !super::completion_checks::reply_admits_unfulfilled_request(&reply)
            && agent
                .supervision_gate_enforced_with_context(
                    "mutation_contract_block",
                    &model,
                    emitter,
                    task_id,
                    iteration,
                    json!({
                        "assistant_claimed_mutation": assistant_claimed_mutation,
                        "concrete_mutation_required": concrete_mutation_required,
                        "mutation_count": completion_progress.mutation_count,
                        "tool_attempt_count": learning_ctx.tool_calls.len(),
                    }),
                )
                .await
        {
            completion_progress.mutation_claim_nudge_count = completion_progress
                .mutation_claim_nudge_count
                .saturating_add(1);
            stall_count = stall_count.saturating_add(1);
            consecutive_clean_iterations = 0;
            pending_system_messages.push(if has_tool_attempts {
                SystemDirective::MutationStillRequired
            } else {
                // A zero-tool side-effect claim has no adapter identity yet.
                // Require execution generically instead of steering every
                // mutation toward file-edit tools.
                SystemDirective::DeferredToolCallRequired
            });
            agent
                .emit_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::PostExecutionValidation,
                    "Blocked completion: side-effect claim has no mutation evidence".to_string(),
                    json!({
                        "condition": "claimed_mutation_without_evidence",
                        "expects_mutation_hint": turn_context.completion_contract.expects_mutation,
                        "assistant_claimed_mutation": true,
                        "mutation_count": completion_progress.mutation_count,
                        "total_successful_tool_calls": total_successful_tool_calls,
                        "stall_count": stall_count,
                    }),
                )
                .await;
            warn!(
                session_id,
                iteration,
                stall_count,
                total_successful_tool_calls,
                "Blocked completion: side-effect claim has mutation_count=0"
            );
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }

        if !force_text_fast_path_accepted
            && agent.depth == 0
            && (unbacked_mutation_claim || (unfulfilled_concrete_mutation && has_tool_attempts))
            && completion_progress.mutation_claim_nudge_count > 0
            && !super::completion_checks::reply_admits_unfulfilled_request(&reply)
        {
            let request = build_agent_side_partial_failure(
                turn_context,
                learning_ctx,
                Some(execution_state),
                "I do not have a successful typed receipt for the requested side effect, so I cannot claim it completed.",
                "Start a new attempt, retry the action, and verify its resulting state before reporting completion.",
            );
            reply = request.render_user_message();
            pending_external_action_ack = None;
            // Preserve the unresolved proof in task-outcome derivation.
            completion_progress.verification_pending = true;
            warn!(
                session_id,
                iteration,
                "Suppressed side-effect success claim after bounded receipt retry"
            );
        }

        // Guardrail: don't accept "I'll do X" / workflow narration as
        // completion text. Either keep the loop alive (if tools exist)
        // or return an explicit blocker (if no tools are available).
        // When tools have already succeeded: allow ONE retry (the agent may
        // produce a better response), but if the guard fires a second time,
        // accept the reply to avoid "Stuck" loops (e.g., after remember_fact
        // the LLM says "I'll remember that" — a confirmation, not a real deferral).
        // Substantive-response fast path: if the model produced a long,
        // content-rich answer (≥200 chars after stripping deferred-action
        // lines) AND it doesn't contain leaked structural markers
        // ([tool_use:], [INTENT_GATE], etc.), accept it immediately even
        // if it opens with an action-promise phrase like "I'll recall…".
        // This prevents recall/informational queries from being rejected
        // and forced through unnecessary tool-call loops.
        let has_structural_markers = {
            let lower = reply.trim().to_ascii_lowercase();
            lower.contains("[consultation]")
                || lower.contains(&INTENT_GATE_MARKER.to_ascii_lowercase())
                || lower.contains("[tool_use:")
                || lower.contains("[tool_call:")
        };
        let reply_is_substantive =
            !has_structural_markers && is_substantive_text_response(&reply, 200);
        // Fabricated-action guard: a reply that claims a completed side
        // effect ("I have deleted the folder") in a task that made ZERO
        // tool calls cannot be truthful. Treat it like a deferred action so the
        // no-tool recovery machinery (hard tool-call nudge, fallback
        // expansion, model switch) gets a chance to make it real. The
        // substantive-text bypass must not rescue such replies either —
        // length is no evidence of truth.
        let claims_unfulfilled_mutation = !has_tool_attempts && unbacked_mutation_claim;
        let claims_unfulfilled_delegation = !has_tool_attempts && assistant_claimed_delegation;
        // False in-progress status: the reply asserts work is happening RIGHT
        // NOW ("I'm searching the API now...") while the task made zero tool
        // calls — nothing is running, nothing was spawned, and the task is
        // about to end. This remains a useful Guided-tier recovery signal, but
        // the phrase detector itself is not exact enough to override an
        // Autonomous model's completion.
        // Present-progressive false status at COMPLETION is dishonest
        // regardless of tool history: the task is ending, so "I am currently
        // refining the query..." is false even after earlier attempts (3rd
        // live incident 2026-07-03, task e5a505af: shipped as the final
        // answer of an ended task WITH failed tool attempts — the earlier
        // zero-tool scoping let it through). Background handoffs are exempt:
        // their in-progress claim has real pending work.
        let claims_false_in_progress =
            crate::agent::response_analysis::reply_is_unbacked_action_promise(&reply)
                && !crate::agent::is_friendly_background_handoff(&reply);
        // Terminal unbacked plan: the FINAL reply promises future work ("I'll
        // try a web search instead") while the task produced zero successful
        // tool calls and schedules nothing. Harness-composed background
        // handoffs are exempt; otherwise the text classifier remains Guided
        // supervision while exact tool/receipt state is retained as telemetry.
        // 4th live incident (2026-07-03, task ending 18:30): a 750-char
        // progress narration ending in "I will present the trials shortly"
        // shipped as the final answer of an ended task — evading BOTH the
        // detector's short-reply cap and this gate's zero-success scoping.
        // The scoping goes: successful earlier calls don't make a terminal
        // promise of FUTURE agent action true; only pending work does.
        let terminal_unbacked_plan = (looks_like_deferred_action_response(&reply)
            || crate::agent::response_analysis::reply_ends_with_unfulfilled_promise(&reply))
            && !crate::agent::is_friendly_background_handoff(&reply);
        if !used_identity_prefill
            && !force_text_fast_path_accepted
            && (looks_like_deferred_action_response(&reply)
                || claims_unfulfilled_mutation
                || claims_unfulfilled_delegation
                || claims_false_in_progress
                || terminal_unbacked_plan)
            && (!reply_is_substantive
                || claims_unfulfilled_mutation
                || claims_unfulfilled_delegation
                || claims_false_in_progress
                || terminal_unbacked_plan)
            // Structural protocol markers are deterministic malformed output.
            // All natural-language claim/deferral detectors remain behavioral
            // supervision, even when compared with the tool ledger: the ledger
            // is exact, but the claim classifier is not.
            && (has_structural_markers
                || agent
                    .supervision_gate_enforced_with_context(
                        "deferred_action_guard",
                        &model,
                        emitter,
                        task_id,
                        iteration,
                        json!({
                            "claims_unfulfilled_mutation": claims_unfulfilled_mutation,
                            "claims_unfulfilled_delegation": claims_unfulfilled_delegation,
                            "claims_false_in_progress": claims_false_in_progress,
                            "terminal_unbacked_plan": terminal_unbacked_plan,
                        }),
                    )
                    .await)
        {
            // Post-tool-success: if we've already caught one deferral after tools
            // succeeded, accept this reply instead of stalling further.
            // Exception: when force_text is active (tools stripped), a deferred
            // reply like "Let me examine..." is useless — the model can't act.
            // Replace it with an activity summary of what was actually done.
            if has_tool_attempts
                && stall_count >= 1
                && !claims_false_in_progress
            {
                if force_text_response && !learning_ctx.tool_calls.is_empty() {
                    let mut recovered_tool_output = false;
                    let mut needs_synthesis_retry = false;
                    let candidate =
                        latest_task_tool_result_for_completion(agent, session_id, task_id, 2500)
                            .await;
                    if let Some(candidate) = candidate.as_ref() {
                        if let Some(tool_reply) = build_force_text_deferred_completion_reply(
                            candidate,
                            learning_ctx.tool_calls.len(),
                        ) {
                            reply = tool_reply;
                            recovered_tool_output = true;
                        } else if tool_output_requires_final_synthesis(
                            &candidate.tool_name,
                            &candidate.tool_output,
                        ) && !empty_response_retry_used
                        {
                            empty_response_retry_used = true;
                            empty_response_retry_pending = true;
                            empty_response_retry_note =
                                Some("structured_tool_output_requires_synthesis".to_string());
                            pending_system_messages
                                .push(structured_result_synthesis_directive(candidate));
                            consecutive_clean_iterations = 0;
                            info!(
                                session_id,
                                iteration,
                                tool = %candidate.tool_name,
                                "Force-text active: retrying once so the model synthesizes the structured tool result"
                            );
                            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                        } else if let Some(tool_reply) =
                            build_structured_tool_output_completion_reply(
                                &candidate.tool_name,
                                &candidate.tool_output,
                                candidate.artifact_delivered,
                            )
                        {
                            reply = tool_reply;
                            recovered_tool_output = true;
                        } else {
                            needs_synthesis_retry = true;
                        }
                    }
                    if !recovered_tool_output {
                        let actions: Vec<&str> =
                            learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                        reply = build_completion_fallback_reply(
                            candidate.as_ref(),
                            &actions,
                            learning_ctx.tool_calls.len(),
                        );
                    }
                    info!(
                            session_id,
                            iteration,
                            total_successful_tool_calls,
                            stall_count,
                            recovered = recovered_tool_output,
                            needs_synthesis_retry,
                            "Force-text active: replaced deferred reply with recovered tool result or activity summary"
                        );
                } else {
                    info!(
                        session_id,
                        iteration,
                        total_successful_tool_calls,
                        stall_count,
                        "Accepting deferred-looking reply as completion after tool progress"
                    );
                }
                // Fall through to the normal completion path below
            } else if tool_defs.is_empty() {
                warn!(
                    session_id,
                    iteration,
                    "Deferred-action reply with no available tools; returning explicit blocker"
                );
                reply = "I wasn't able to complete that request because no execution tools are available in this context. Please try again in a context with tool access."
                    .to_string();
            } else if !has_tool_attempts
                && deferred_no_tool_streak >= DEFERRED_NO_TOOL_ACCEPT_THRESHOLD
                && is_substantive_text_response(&reply, 50)
                && !claims_unfulfilled_mutation
            {
                // Early acceptance: the model keeps producing deferred-action text
                // but the underlying content is substantive (e.g., a greeting,
                // explanation, joke, or capability listing).  Queries that genuinely
                // don't need tools should not stall for 6 retries.
                info!(
                        session_id,
                        iteration,
                        deferred_no_tool_streak,
                        reply_len = reply.len(),
                        "Accepting substantive text-only response after repeated deferred-no-tool retries"
                    );
                deferred_no_tool_streak = 0;
                // Fall through to the normal completion path below
            } else {
                let mut recovered_post_tool_deferral = false;
                // Pre-execution deferrals ("I'll do X") should not consume the
                // main stall budget. Reserve stall_count for post-tool loops so
                // we don't fail as "stuck" before any tool ever executes.
                if !has_tool_attempts {
                    deferred_no_tool_streak = deferred_no_tool_streak.saturating_add(1);
                    POLICY_METRICS
                        .deferred_no_tool_deferral_detected_total
                        .fetch_add(1, Ordering::Relaxed);
                    agent
                        .with_harness_eval(|eval| eval.record_deferred_no_tool_event())
                        .await;
                    consecutive_clean_iterations = 0;
                } else {
                    // First post-tool deferral: try recovering from the latest tool
                    // result before burning another LLM iteration on "I'll do X" text.
                    // NOT when the contract expects a mutation that hasn't happened:
                    // pasting whatever the last read produced is never the requested
                    // action (live 2026-07-12: "Send me my resume" → deferral →
                    // recovery pasted an `ls -R` of personal PDFs while the send_file
                    // stayed undone). Let the stall path nudge the model instead.
                    let paste_recovery_allowed = tool_output_paste_recovery_allowed(
                        turn_context.completion_contract.expects_mutation,
                        usize::from(mutation_contract_fulfilled(
                            &turn_context.completion_contract,
                            &completion_progress,
                        )),
                    );
                    let candidate = if paste_recovery_allowed {
                        latest_task_tool_result_for_completion(agent, session_id, task_id, 2500)
                            .await
                    } else {
                        None
                    };
                    if let Some(candidate) = candidate.as_ref() {
                        if candidate.tool_name == "send_file" {
                            reply = super::stopping_phase::send_file_completion_reply().to_string();
                            recovered_post_tool_deferral = true;
                        } else if let Some(tool_reply) = build_tool_output_completion_reply(
                            &candidate.tool_name,
                            &candidate.tool_output,
                            candidate.artifact_delivered,
                        ) {
                            reply = tool_reply;
                            recovered_post_tool_deferral = true;
                        } else if let Some(tool_reply) =
                            build_structured_tool_output_completion_reply(
                                &candidate.tool_name,
                                &candidate.tool_output,
                                candidate.artifact_delivered,
                            )
                        {
                            reply = tool_reply;
                            recovered_post_tool_deferral = true;
                        }
                    }
                    if recovered_post_tool_deferral {
                        info!(
                            session_id,
                            iteration,
                            total_successful_tool_calls,
                            "Recovered first post-tool deferred reply from successful tool output"
                        );
                    } else {
                        stall_count = stall_count.saturating_add(1);
                        deferred_no_tool_streak = 0;
                        agent
                            .with_harness_eval(|eval| eval.record_stall_guard())
                            .await;
                        consecutive_clean_iterations = 0;
                    }
                }

                if recovered_post_tool_deferral {
                    // Fall through to the normal completion path below.
                } else {
                    // Hard escape: when force_text is active (tools stripped) and we
                    // have tool history, deferred-action ContinueLoop is a dead end —
                    // the model cannot use tools. Build a fallback reply immediately
                    // instead of looping forever.
                    if force_text_response
                        && has_tool_attempts
                        && !learning_ctx.tool_calls.is_empty()
                    {
                        let actions: Vec<&str> =
                            learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                        let candidate = latest_task_tool_result_for_completion(
                            agent, session_id, task_id, 2500,
                        )
                        .await;
                        reply = build_completion_fallback_reply(
                            candidate.as_ref(),
                            &actions,
                            learning_ctx.tool_calls.len(),
                        );
                        info!(
                            session_id,
                            iteration,
                            stall_count,
                            total_successful_tool_calls,
                            "Force-text deferred-action hard escape: replaced with fallback reply"
                        );
                        // Fall through to normal completion path (no ContinueLoop)
                    } else {
                        warn!(
                            session_id,
                            iteration,
                            stall_count,
                            deferred_no_tool_streak,
                            total_successful_tool_calls,
                            has_tool_attempts,
                            "Deferred-action reply without concrete results; continuing loop"
                        );

                        // Check if the deferred-action reply itself contains an
                        // INTENT_GATE marker claiming needs_tools:true — i.e. the model
                        // explicitly told us it needs tool access to fulfil this request.
                        // This is more reliable than `expects_mutation` which also matches
                        // pure text-generation tasks ("write a tweet").
                        let response_claims_needs_tools = {
                            let lower_reply = reply.to_ascii_lowercase();
                            lower_reply.contains(&INTENT_GATE_MARKER.to_ascii_lowercase())
                                && lower_reply.contains("\"needs_tools\":true")
                        };
                        let deferred_nudge = if !has_tool_attempts {
                            // A claimed-but-unexecuted mutation always needs a
                            // tool call — never downgrade it to plain-text mode,
                            // which would accept the fabrication next iteration.
                            if execution_requirement.requires_execution()
                                || response_claims_needs_tools
                                || claims_unfulfilled_mutation
                                || claims_unfulfilled_delegation
                            {
                                SystemDirective::DeferredToolCallRequired
                            } else if force_text_allowed {
                                force_text_response = true;
                                SystemDirective::ToolModeDisabledPlainText
                            } else {
                                SystemDirective::DeferredToolCallRequired
                            }
                        } else {
                            SystemDirective::DeferredProvideConcreteResults
                        };

                        pending_system_messages.push(deferred_nudge);

                        // Fallback expansion: widen tool set once after exactly two
                        // no-progress iterations, even in no-tool-call paths.
                        let fallback_trigger = if !has_tool_attempts {
                            deferred_no_tool_streak == 2
                        } else {
                            stall_count == 2
                        };
                        if fallback_trigger && !fallback_expanded_once {
                            fallback_expanded_once = true;
                            let previous_count = tool_defs.len();
                            let widened = agent.filter_tool_definitions_for_policy(
                                base_tool_defs,
                                available_capabilities,
                                &policy_bundle.policy,
                                policy_bundle.risk_score,
                                true,
                            );
                            if widened.len() > previous_count {
                                POLICY_METRICS
                                    .fallback_expansion_total
                                    .fetch_add(1, Ordering::Relaxed);
                                tool_defs = widened;
                                info!(
                                    session_id,
                                    iteration,
                                    previous_count,
                                    widened_count = tool_defs.len(),
                                    "No-progress fallback expansion applied (deferred-action path)"
                                );
                            }
                        }

                        if !has_tool_attempts
                            && deferred_no_tool_streak >= DEFERRED_NO_TOOL_SWITCH_THRESHOLD
                            && deferred_no_tool_model_switches < MAX_DEFERRED_NO_TOOL_MODEL_SWITCHES
                            && !computer_use_pin_active
                        {
                            if let Some(next_model) = agent
                                .pick_fallback_excluding(&model, &[], llm_router.as_ref())
                                .await
                            {
                                info!(
                                    session_id,
                                    iteration,
                                    from_model = %model,
                                    to_model = %next_model,
                                    "Deferred/no-tool recovery: switching model for one retry window"
                                );
                                model = next_model;
                                deferred_no_tool_model_switches += 1;
                                POLICY_METRICS
                                    .deferred_no_tool_model_switch_total
                                    .fetch_add(1, Ordering::Relaxed);
                                // Strategy changed, give the new model a fresh stall budget.
                                stall_count = 0;
                                pending_system_messages
                                    .push(SystemDirective::RecoveryModeModelSwitch);
                            }
                        }

                        if !has_tool_attempts
                            && deferred_no_tool_streak >= MAX_STALL_ITERATIONS
                            && !learning_ctx
                                .errors
                                .iter()
                                .any(|(e, _)| e == DEFERRED_NO_TOOL_ERROR_MARKER)
                        {
                            learning_ctx
                                .errors
                                .push((DEFERRED_NO_TOOL_ERROR_MARKER.to_string(), false));
                            POLICY_METRICS
                                .deferred_no_tool_error_marker_total
                                .fetch_add(1, Ordering::Relaxed);
                            warn!(
                                session_id,
                                iteration,
                                deferred_no_tool_streak,
                                "Deferred/no-tool recovery exhausted: recording terminal marker"
                            );
                        }

                        return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                    } // end force_text hard escape else
                } // end recovered_post_tool_deferral else
            }
        }

        validation_state.refresh_success_criteria_matches(&reply);
        if !validation_state.active_success_criteria.is_empty()
            && validation_state.matched_success_criteria.is_empty()
        {
            validation_state.record_failure(ValidationFailure::SuccessCriteriaUnmatched);
        }
        validation_state.clear_loop_repetition_reason();
        let reply_is_model_authored = reply == model_authored_reply;
        // Degeneration guard: collapse runaway repetition loops before anything
        // else. Models (especially local ones) sometimes collapse into emitting
        // the same sentence or line many times in a row once their context fills
        // with repetitive content; without this the user sees a wall of
        // duplicated text split across many chunked messages.
        let original_final_reply_chars = reply.trim().chars().count();
        let (mut reply, collapsed_repetition) =
            crate::tools::sanitize::collapse_degenerate_repetition(&reply);
        if collapsed_repetition {
            warn!(
                session_id,
                iteration,
                original_len = original_final_reply_chars,
                collapsed_len = reply.trim().chars().count(),
                "Collapsed degenerate repetition loop in final reply"
            );
        }

        let leaked_internal_identifiers = if reply_is_model_authored {
            execution_state.unrequested_internal_identifiers(&reply, user_text)
        } else {
            Vec::new()
        };
        if !leaked_internal_identifiers.is_empty() && !empty_response_retry_used {
            warn!(
                session_id,
                iteration,
                identifier_count = leaked_internal_identifiers.len(),
                "Natural outcome reply exposed internal identifiers — retrying presentation once"
            );
            empty_response_retry_used = true;
            empty_response_retry_pending = true;
            empty_response_retry_note =
                Some("natural_outcome_exposed_internal_identifiers".to_string());
            pending_system_messages.push(SystemDirective::NaturalToolOutcomePresentation);
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }
        if !leaked_internal_identifiers.is_empty() {
            reply = crate::tools::sanitize::remove_lines_with_identifiers(
                &reply,
                &leaked_internal_identifiers,
            );
        }

        // Sanitize user-facing output before any channel-specific redaction.
        let pre_sanitize_chars = reply.trim().chars().count();
        let sanitized_reply = crate::tools::sanitize::sanitize_user_facing_reply(&reply);
        let gutted_by_sanitization = crate::tools::sanitize::reply_gutted_by_sanitization(
            pre_sanitize_chars,
            &sanitized_reply,
        );

        // Safety net for a gutted reply (sanitization stripped a non-empty
        // draft to empty or a dangling lead-in stub like "Here are the
        // results:"). The draft was scaffolding, but the model usually still
        // holds a real conclusion — give it ONE tool-less restatement retry
        // before shipping a deterministic activity dump (observed live: the
        // dump lost an honest "couldn't find that file" conclusion).
        if gutted_by_sanitization && !empty_response_retry_used {
            warn!(
                session_id,
                iteration, "Sanitization gutted reply — retrying final answer once"
            );
            empty_response_retry_used = true;
            empty_response_retry_pending = true;
            empty_response_retry_note = Some("final_reply_gutted_by_sanitization".to_string());
            pending_system_messages.push(SystemDirective::FinalAnswerRejectedInternalMarkers);
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }
        // The model shipped a recent tool result verbatim instead of answering.
        // Two shape-specific checks (a `File: … (lines …)` read_file page, a
        // large JSON/bracket blob) plus a general check: does the reply's own
        // content substantially duplicate the latest tool output — regardless of
        // shape (a search_files path list, raw terminal output, …)? The general
        // check needs the tool output, so it only runs (one DB read) when the
        // cheap shape checks miss and we still have a retry to spend. One-shot
        // retry with a directive to answer from the data; shares the same retry
        // budget as the gutted case above.
        let reply_is_shape_paste = reply_is_model_authored
            && (reply_is_pasted_file_page(&sanitized_reply)
                || crate::agent::response_analysis::reply_is_raw_data_dump(&sanitized_reply));
        let mut reply_pastes_tool_output = false;
        if reply_is_model_authored && !reply_is_shape_paste && !empty_response_retry_used {
            if let Some(latest) =
                latest_task_tool_result_for_completion(agent, session_id, task_id, 2500).await
            {
                reply_pastes_tool_output =
                    crate::agent::response_analysis::reply_duplicates_tool_output(
                        &sanitized_reply,
                        &latest.tool_output,
                    );
            }
        }
        if (reply_is_shape_paste || reply_pastes_tool_output) && !empty_response_retry_used {
            warn!(
                session_id,
                iteration,
                shape_paste = reply_is_shape_paste,
                tool_output_paste = reply_pastes_tool_output,
                "Final reply pastes tool output instead of answering — retrying final answer once"
            );
            empty_response_retry_used = true;
            empty_response_retry_pending = true;
            empty_response_retry_note = Some("final_reply_tool_output_paste".to_string());
            pending_system_messages.push(SystemDirective::FinalAnswerWasFilePaste);
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }
        let reply = if gutted_by_sanitization {
            warn!(
                session_id,
                iteration, "Sanitization gutted reply — falling back to activity summary"
            );
            if !learning_ctx.tool_calls.is_empty() {
                let refs: Vec<&str> = learning_ctx.tool_calls.iter().map(|s| s.as_str()).collect();
                build_activity_summary_reply(&refs)
            } else {
                // No tools ran and both composition attempts gutted: the
                // model kept echoing scaffolding-wrapped content. A bare
                // "Done." here is dishonest — nothing was done (live repro
                // 2026-07-03: "give me the trial details" → "Done."). Say
                // what happened and offer a recovery path instead.
                "I ran into a formatting problem composing that answer and couldn't recover \
                 it. Ask me again and I'll re-gather the details."
                    .to_string()
            }
        } else {
            sanitized_reply
        };

        let mut reply = match channel_ctx.visibility {
            ChannelVisibility::Public | ChannelVisibility::PublicExternal => {
                let (sanitized, had_redactions) = crate::tools::sanitize::sanitize_output(&reply);
                if had_redactions && channel_ctx.visibility == ChannelVisibility::PublicExternal {
                    format!("{}\n\n(Some content was filtered for security)", sanitized)
                } else {
                    sanitized
                }
            }
            _ => reply,
        };
        // Diagnostic: warn when completing with zero tool calls and deferred-action
        // text. This catches cases where the agent promises future work ("I'll search
        // for TODOs...") but never actually executes any tools (G2 stall pattern).
        if total_successful_tool_calls == 0
            && !reply.trim().is_empty()
            && looks_like_deferred_action_response(&reply)
        {
            warn!(
                session_id,
                iteration,
                reply_preview = &reply.chars().take(200).collect::<String>() as &str,
                "Zero-tool completion with deferred-action text detected — possible stall pattern"
            );
        }

        // The model asked the user to upload/provide a file it can locate
        // itself (small models default to chat behavior on fresh contexts).
        // Force one retry with an explicit lookup directive instead of
        // accepting the punt. Fires once per turn to prevent loops.
        if total_successful_tool_calls == 0
            && completion_progress.file_access_retry_count == 0
            && crate::agent::response_analysis::reply_defers_file_access(&reply)
            && crate::agent::response_analysis::user_text_references_file(user_text)
            && agent
                .supervision_gate_enforced(
                    "file_access_deferral_retry",
                    &model,
                    emitter,
                    task_id,
                    iteration,
                )
                .await
        {
            // Increment the LOCAL copy; the single phase boundary applies it
            // over ctx, so a direct ctx increment would be clobbered and the
            // "fires once" guarantee silently lost.
            completion_progress.file_access_retry_count += 1;
            let hint = user_text.chars().take(300).collect::<String>();
            pending_system_messages.push(SystemDirective::LocateFileInsteadOfAsking {
                user_text_hint: hint,
            });
            warn!(
                session_id,
                iteration,
                reply_preview = &reply.chars().take(200).collect::<String>() as &str,
                "Reply defers file access to user despite available lookup tools — forcing retry"
            );
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }

        // Quality guard: reject canned ack responses and low-quality replies.
        // Canned acks ("The requested action completed successfully") are NEVER
        // appropriate as final user-facing responses — they lack explanation of
        // what was done. Always nudge for a proper response regardless of whether
        // the request looks multi-part.
        // Only fires once (quality_nudge_count == 0) to prevent infinite loops.
        let is_canned_ack_reply = reply.starts_with("The requested action completed successfully")
            || reply.starts_with("The requested action finished with errors");
        let is_low_quality_multipart = !is_canned_ack_reply
            && reply.len() < 400
            && total_successful_tool_calls >= 4
            && looks_like_multi_part_request(user_text);
        // Canned ack is always low quality when there was significant tool work
        let is_canned_with_work = is_canned_ack_reply && total_successful_tool_calls >= 3;
        let is_plain_text_tool_call = response_looks_like_plain_text_tool_call(&reply);
        if (is_canned_with_work || is_low_quality_multipart || is_plain_text_tool_call)
            && completion_progress.quality_nudge_count == 0
            && agent
                .supervision_gate_enforced_with_context(
                    "response_quality_nudge",
                    &model,
                    emitter,
                    task_id,
                    iteration,
                    json!({
                        "canned_with_work": is_canned_with_work,
                        "low_quality_multipart": is_low_quality_multipart,
                        "plain_text_tool_call": is_plain_text_tool_call,
                    }),
                )
                .await
        {
            // Local copy, not ctx — see file_access_retry_count above.
            completion_progress.quality_nudge_count += 1;
            let hint = user_text.chars().take(300).collect::<String>();
            pending_system_messages.push(SystemDirective::ResponseQualityNudge {
                user_text_hint: hint,
            });
            warn!(
                session_id,
                iteration,
                reply_len = reply.len(),
                total_successful_tool_calls,
                is_plain_text_tool_call,
                "Response quality too low for multi-part request — nudging for better response"
            );
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }

        // Grounding guard: a reply that enumerates name-like list entries
        // absent from every tool output this turn is fabricating list content
        // (e.g. inventing roster members that search snippets never showed).
        // The user's own message also counts as evidence. Fires once per turn;
        // skipped when the evidence buffer overflowed (incomplete evidence
        // would flag legitimately-observed entries).
        if completion_progress.grounding_nudge_count == 0
            && total_successful_tool_calls > 0
            && !execution_state.tool_output_evidence_overflow
            && !execution_state.tool_output_evidence.is_empty()
        {
            let ungrounded = super::answer_grounding::find_ungrounded_list_entities(
                &reply,
                &[execution_state.tool_output_evidence.as_str(), user_text],
            );
            if !ungrounded.is_empty()
                && agent
                    .supervision_gate_enforced_with_context(
                        "ungrounded_list_rewrite",
                        &model,
                        emitter,
                        task_id,
                        iteration,
                        json!({ "ungrounded_count": ungrounded.len() }),
                    )
                    .await
            {
                completion_progress.grounding_nudge_count += 1;
                warn!(
                    session_id,
                    iteration,
                    ungrounded_count = ungrounded.len(),
                    ungrounded_preview = %ungrounded.join(", "),
                    "Final reply enumerates entities absent from all tool outputs — forcing grounded rewrite"
                );
                pending_system_messages.push(SystemDirective::UngroundedListEntities {
                    entities: ungrounded,
                });
                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }
        }

        // Search-before-deny gate: the reply denies or asserts a specific
        // personal fact about an entity the user named (or an owner-relative
        // such as "my dad"), but no memory lookup grounded it this turn.
        // Classifier-gated and owner-DM-gated; bounded to one fire per turn so
        // it can never loop indefinitely.
        // Only fires in private DMs (owner 1-on-1) — never in public channels,
        // group chats, or sub-agent internal sessions.
        let is_owner_dm = user_role == UserRole::Owner
            && channel_ctx.visibility == crate::types::ChannelVisibility::Private;
        if agent.mandate_execution.is_none()
            && completion_progress.denial_gate_count == 0
            && is_owner_dm
            && !completion_progress.coreference_fired
            && !execution_state.tool_output_evidence_overflow
            && super::answer_grounding::reply_contains_unsearched_denial_phrase(&reply)
        {
            let memory_lookup_fired_this_turn = learning_ctx.tool_calls.iter().any(|call| {
                call.starts_with("manage_memories(") || call.starts_with("manage_people(")
            });
            // Scope the denial gate to a specific named-person or owner-relative
            // relational query, where a no-search denial is unambiguous.
            if !memory_lookup_fired_this_turn
                && crate::agent::relational_prefilter::user_text_is_named_person_relational_query(
                    user_text,
                )
            {
                let fast_model_for_denial = llm_router
                    .as_ref()
                    .map(|r| r.select(crate::router::Tier::Fast).to_string())
                    .unwrap_or_else(|| model.clone());
                let intent = crate::agent::llm_classifier::classify_relational_intent(
                    llm_provider.as_ref(),
                    &fast_model_for_denial,
                    user_text,
                )
                .await;
                if !intent.entities.is_empty() {
                    // Pass only tool output as evidence — NOT user_text.
                    // Including user_text would cause entity names from the
                    // user's own question to appear "grounded" even though no
                    // memory search was done. We only want to flag entities
                    // that were not found in actual memory lookup results.
                    let unsearched = super::answer_grounding::find_unsearched_denials(
                        &reply,
                        &intent.entities,
                        &[execution_state.tool_output_evidence.as_str()],
                    );
                    if !unsearched.is_empty()
                        && agent
                            .supervision_gate_enforced_with_context(
                                "unsearched_personal_fact_retry",
                                &model,
                                emitter,
                                task_id,
                                iteration,
                                json!({ "entity_count": unsearched.len() }),
                            )
                            .await
                    {
                        completion_progress.denial_gate_count += 1;
                        warn!(
                            target: "memory_recall",
                            session_id,
                            iteration,
                            entities = %unsearched.join(", "),
                            "Reply denies/asserts an unsearched entity — forcing a memory lookup"
                        );
                        pending_system_messages.push(SystemDirective::UnsearchedEntityDenial {
                            entities: unsearched,
                        });
                        return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                    }
                }
            } else if memory_lookup_fired_this_turn
                && crate::agent::relational_prefilter::user_text_is_named_person_relational_query(
                    user_text,
                )
                && !execution_state.tool_output_evidence.is_empty()
            {
                // Observation only (no intervention): the model DID search memory
                // and got non-empty results, yet still denied a named-person
                // relational query — a possible "searched-but-denied" reasoning
                // miss (the connecting facts may have been present in the results
                // but not used). Logged, not acted on, so we can measure whether
                // this failure mode actually occurs before deciding to build a
                // gate for it. See CHANGELOG / the relational-recall design notes.
                tracing::info!(
                    target: "memory_recall",
                    session_id,
                    iteration,
                    query = %crate::utils::truncate_str(user_text, 120),
                    evidence_len = execution_state.tool_output_evidence.len(),
                    "relational denial despite memory results (possible reasoning miss; observation only)"
                );
            }
        }

        // Exact-history honesty gate. Immediate context may answer the question
        // directly; this only intervenes when the draft claims the wording is
        // unavailable and canonical history has not been queried.
        if turn_context.completion_contract.requires_exact_history
            && crate::agent::response_analysis::reply_claims_history_unavailable(&reply)
            && !execution_state
                .outcome_ledger
                .iter()
                .any(|entry| entry.tool_name == "search_history" && entry.success)
        {
            if completion_progress.history_lookup_nudge_count == 0
                && !force_text_response
                && tool_defs
                    .iter()
                    .any(|schema| tool_definition_is_named(schema, "search_history"))
            {
                completion_progress.history_lookup_nudge_count = completion_progress
                    .history_lookup_nudge_count
                    .saturating_add(1);
                pending_system_messages.push(SystemDirective::ExactHistoryLookupRequired);
                return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
            }
            let request = build_agent_side_partial_failure(
                turn_context,
                learning_ctx,
                Some(execution_state),
                "I could not establish the exact earlier wording from immediate context or a successful retained-history lookup, so I will not guess it.",
                "Start a new attempt when retained-history lookup is available.",
            );
            reply = request.render_user_message();
            completion_progress.verification_pending = true;
            pending_external_action_ack = None;
        }

        // Explicit source contract: both successful source-page reads and
        // direct citations in the candidate answer are required. Approval
        // prompts cannot count as evidence because they produce neither.
        if turn_context.completion_contract.minimum_sources > 0 {
            let required = turn_context.completion_contract.minimum_sources;
            let sources_read = execution_state.web_source_urls.len();
            let sources_cited = execution_state.cited_web_source_count(&reply);
            if sources_read < required || sources_cited < required {
                if completion_progress.source_evidence_nudge_count == 0
                    && !force_text_response
                    && !tool_defs.is_empty()
                {
                    completion_progress.source_evidence_nudge_count = completion_progress
                        .source_evidence_nudge_count
                        .saturating_add(1);
                    pending_system_messages.push(SystemDirective::SourceEvidenceRequired {
                        required,
                        sources_read,
                        sources_cited,
                        primary: turn_context.completion_contract.requires_primary_sources,
                    });
                    warn!(
                        session_id,
                        iteration,
                        required,
                        sources_read,
                        sources_cited,
                        "Blocking completion until explicit source requirement is met"
                    );
                    return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
                }

                let request = build_agent_side_partial_failure(
                    turn_context,
                    learning_ctx,
                    Some(execution_state),
                    format!(
                        "I could not satisfy the requested evidence threshold: {sources_read} qualifying source page(s) were read and {sources_cited} were cited, out of {required} required."
                    ),
                    "Start a new attempt when the required sources can be retrieved.",
                );
                reply = request.render_user_message();
                completion_progress.verification_pending = true;
                pending_external_action_ack = None;
            }
        }

        // Corroboration guard: an enumeration-style answer produced from web
        // research must rest on at least two successfully read source pages —
        // snippets-only or single-page answers get one chance to fetch a
        // second independent source or explicitly caveat the single-sourcing.
        // Fires once per turn.
        if completion_progress.corroboration_nudge_count == 0
            && execution_state.web_search_used
            && execution_state.web_source_domains.len() < 2
            && super::answer_grounding::count_list_name_entities(&reply)
                >= super::answer_grounding::MIN_LIST_ENTITIES
            && agent
                .supervision_gate_enforced_with_context(
                    "single_source_enumeration_retry",
                    &model,
                    emitter,
                    task_id,
                    iteration,
                    json!({
                        "sources_read": execution_state.web_source_domains.len(),
                    }),
                )
                .await
        {
            completion_progress.corroboration_nudge_count += 1;
            warn!(
                session_id,
                iteration,
                sources_read = execution_state.web_source_domains.len(),
                "Enumeration answer rests on <2 read sources — requesting corroboration or explicit caveat"
            );
            pending_system_messages.push(SystemDirective::SingleSourceEnumeration {
                sources_read: execution_state.web_source_domains.len(),
            });
            return Ok(Some(ResponsePhaseOutcome::ContinueLoop));
        }

        // Approval-producing tools normally await their resolution before
        // returning. This durable task-local projection is a defense-in-depth
        // gate against any transport/restart race: pending authority can never
        // be reported as completed or verified.
        match agent
            .event_store
            .get_pending_task_interactions(task_id)
            .await
        {
            Ok(pending) if !pending.is_empty() => {
                let request = build_partial_done_blocked_request_with_plan(
                    turn_context,
                    learning_ctx,
                    Some(execution_state),
                    "The requested action is still awaiting approval, so it has not been performed or verified.",
                    "Resolution of the pending approval request.",
                    "After approval is resolved, I will execute the action and report only the resulting receipts.",
                );
                reply = request.render_user_message();
                completion_progress.verification_pending = true;
                pending_external_action_ack = None;
                warn!(
                    session_id,
                    iteration,
                    pending_approvals = pending.len(),
                    "Suppressed completion while task approval remains unresolved"
                );
            }
            Ok(_) => {}
            Err(error) => {
                warn!(%error, task_id, "Could not query task approval projection at completion");
                let request = build_agent_side_partial_failure(
                    turn_context,
                    learning_ctx,
                    Some(execution_state),
                    "I could not verify whether the task still has a pending approval, so I cannot safely report the action as completed.",
                    "Start a new attempt after approval state can be read reliably.",
                );
                reply = request.render_user_message();
                completion_progress.verification_pending = true;
                pending_external_action_ack = None;
            }
        }

        let has_unrecovered_errors = learning_ctx.has_unrecovered_model_error();
        let (sanitization_summary, sanitization_metadata) = build_final_sanitization_gate_telemetry(
            original_final_reply_chars,
            reply.trim().chars().count(),
            collapsed_repetition,
            gutted_by_sanitization,
        );
        agent
            .emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::GateTelemetry,
                sanitization_summary,
                sanitization_metadata,
            )
            .await;

        // Finalization is a single commit boundary: persist exactly the text
        // that will be returned to ingress after every rewrite, safety gate,
        // channel redaction, and fallback has finished. Response identity is
        // therefore stable across persistence and delivery without recovering
        // it by comparing transformed prose.
        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(reply.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        };
        agent
            .append_assistant_message_with_event(
                emitter,
                &assistant_msg,
                &model,
                resp.usage.as_ref().map(|usage| usage.input_tokens),
                resp.usage.as_ref().map(|usage| usage.output_tokens),
            )
            .await?;

        let fast_model = llm_router
            .as_ref()
            .map(|router| router.select(crate::router::Tier::Fast).to_string())
            .unwrap_or_else(|| model.clone());
        let (spawn_progressive_facts, spawn_summary_maintenance) = post_completion_memory_plan(
            agent.mandate_execution.is_some(),
            learning_ctx.memory_persistence_allowed,
            agent.context_window_config.progressive_facts,
            crate::memory::context_window::should_extract_facts(user_text),
            agent.context_window_config.enabled,
        );
        if spawn_progressive_facts {
            crate::memory::context_window::spawn_progressive_extraction(
                llm_provider.clone(),
                fast_model.clone(),
                agent.state.clone(),
                agent.event_store.clone(),
                task_id.to_string(),
                user_text.to_string(),
                reply.clone(),
                channel_ctx.channel_id.clone(),
                channel_ctx.visibility,
                user_role,
            );
        }
        if spawn_summary_maintenance {
            let summary_token_threshold = agent
                .context_window_config
                .summarize_token_threshold_for(&fast_model);
            let summary_recent_tokens = agent
                .context_window_config
                .summary_recent_tokens_for(&fast_model);
            crate::memory::context_window::spawn_incremental_summarization(
                llm_provider.clone(),
                fast_model,
                agent.state.clone(),
                agent.event_store.clone(),
                session_id.to_string(),
                summary_token_threshold,
                summary_recent_tokens,
                user_role,
            );
        }

        let outcome = TaskOutcomeDerivation::from_completion_state(
            &validation_state,
            execution_state,
            ctx.completion_progress,
            &turn_context.completion_contract,
            response_has_user_value(&reply, total_successful_tool_calls),
            has_unrecovered_errors,
            terminal_cause,
        )
        .derive_outcome();

        agent
            .emit_task_end(
                emitter,
                task_id,
                TaskStatus::Completed,
                outcome,
                task_start,
                iteration,
                learning_ctx.tool_calls.len(),
                None,
                Some(reply.chars().take(200).collect()),
            )
            .await;

        learning_ctx.completed_naturally = true;
        learning_ctx.task_outcome = Some(outcome);
        if agent.mandate_execution.is_none() {
            let learning_ctx_for_task = learning_ctx.clone();
            let state = agent.state.clone();
            tokio::spawn(async move {
                if let Err(e) = post_task::process_learning(&state, learning_ctx_for_task).await {
                    warn!("Learning failed: {}", e);
                }
            });
        }

        info!(
            session_id,
            iteration,
            reply_len = reply.len(),
            reply_empty = reply.trim().is_empty(),
            reply_preview = &reply.chars().take(120).collect::<String>() as &str,
            outcome = outcome.as_str(),
            "Agent completed naturally"
        );
            return Ok(Some(ResponsePhaseOutcome::Return(Ok(reply))));
        }

        Ok(None)
    }
    .await;

    finish_completion_phase(
        outcome,
        CompletionStateDelta {
            tool_defs,
            model,
            stall_count,
            consecutive_clean_iterations,
            deferred_no_tool_streak,
            deferred_no_tool_model_switches,
            fallback_expanded_once,
            empty_response_retry_used,
            empty_response_retry_pending,
            empty_response_retry_note,
            identity_prefill_text,
            pending_background_ack,
            pending_external_action_ack,
            require_file_recheck_before_answer,
            completion_progress,
            force_text_response,
            validation_state,
        },
        CompletionStateTargets {
            tool_defs: ctx.tool_defs,
            model: ctx.model,
            stall_count: ctx.stall_count,
            consecutive_clean_iterations: ctx.consecutive_clean_iterations,
            deferred_no_tool_streak: ctx.deferred_no_tool_streak,
            deferred_no_tool_model_switches: ctx.deferred_no_tool_model_switches,
            fallback_expanded_once: ctx.fallback_expanded_once,
            empty_response_retry_used: ctx.empty_response_retry_used,
            empty_response_retry_pending: ctx.empty_response_retry_pending,
            empty_response_retry_note: ctx.empty_response_retry_note,
            identity_prefill_text: ctx.identity_prefill_text,
            pending_background_ack: ctx.pending_background_ack,
            pending_external_action_ack: ctx.pending_external_action_ack,
            require_file_recheck_before_answer: ctx.require_file_recheck_before_answer,
            completion_progress: ctx.completion_progress,
            force_text_response: ctx.force_text_response,
            validation_state: ctx.validation_state,
        },
    )
}
