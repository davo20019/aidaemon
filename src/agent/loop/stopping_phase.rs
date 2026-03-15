use super::stopping_conditions::{
    LoopControlDecision, LoopControlInputs, PureStoppingInputs, StoppingCondition,
};
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ConversationSummary;
use crate::utils::truncate_with_note;

pub(super) enum StoppingPhaseOutcome {
    ContinueLoop,
    Return(anyhow::Result<String>),
    Proceed,
}

pub(super) struct StoppingPhaseCtx<'a> {
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a mut LearningContext,
    pub hard_cap: Option<usize>,
    pub effective_task_timeout: Option<Duration>,
    pub task_tokens_used: u64,
    pub effective_task_budget: &'a mut Option<u64>,
    pub budget_warning_sent: &'a mut bool,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub budget_extensions_count: &'a mut usize,
    pub user_role: UserRole,
    pub evidence_gain_count: usize,
    pub stall_count: usize,
    pub deferred_no_tool_streak: usize,
    pub consecutive_same_tool: &'a (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a HashSet<u64>,
    pub total_successful_tool_calls: usize,
    pub pending_background_ack: &'a mut Option<String>,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub resolved_goal_id: &'a Option<String>,
    pub is_scheduled_goal: bool,
    pub effective_daily_budget: &'a mut Option<u64>,
    pub effective_goal_daily_budget: &'a mut Option<i64>,
    pub successful_send_file_keys: &'a HashSet<String>,
    pub model: &'a mut String,
    pub soft_threshold: Option<usize>,
    pub soft_warn_at: Option<usize>,
    pub soft_limit_warned: &'a mut bool,
    pub last_progress_summary: &'a mut Instant,
    pub tool_failure_count: &'a HashMap<String, usize>,
    pub session_summary: &'a mut Option<ConversationSummary>,
    pub policy_bundle: &'a mut PolicyBundle,
    pub user_text: &'a str,
    pub available_capabilities: &'a HashMap<String, ToolCapabilities>,
    pub llm_router: &'a Option<Router>,
    pub last_escalation_iteration: &'a mut Option<usize>,
    pub consecutive_clean_iterations: &'a mut usize,
    pub max_budget_extensions: usize,
    pub hard_token_cap: i64,
    pub execution_state: &'a mut ExecutionState,
    pub force_text_response: &'a mut bool,
    pub completion_progress: &'a mut CompletionProgress,
    pub turn_context: &'a TurnContext,
    pub validation_state: &'a mut ValidationState,
}

fn turn_contract_is_text_only(turn_context: &TurnContext) -> bool {
    !turn_context.completion_contract.expects_mutation
        && !turn_context.completion_contract.requires_observation
}

fn has_task_relevant_progress(
    turn_context: &TurnContext,
    completion_progress: &CompletionProgress,
) -> bool {
    (turn_context.completion_contract.expects_mutation && completion_progress.mutation_count > 0)
        || completion_progress.observation_count > 0
        || completion_progress.verification_count > 0
}

fn has_any_concrete_execution(
    turn_context: &TurnContext,
    completion_progress: &CompletionProgress,
    recoverable_tool_snapshot_present: bool,
    total_successful_tool_calls: usize,
) -> bool {
    has_task_relevant_progress(turn_context, completion_progress)
        || recoverable_tool_snapshot_present
        // Any successfully completed tool call counts as concrete work,
        // even if its semantics did not classify as observation/mutation
        // (common for MCP tools with Unknown/Administrative effects).
        // This prevents the harsh "abandon" path when tools DID execute.
        || total_successful_tool_calls > 0
}

fn only_final_response_remains(
    turn_context: &TurnContext,
    completion_progress: &CompletionProgress,
    recoverable_tool_snapshot_present: bool,
    total_successful_tool_calls: usize,
) -> bool {
    has_any_concrete_execution(
        turn_context,
        completion_progress,
        recoverable_tool_snapshot_present,
        total_successful_tool_calls,
    ) && !completion_progress.verification_pending
}

impl Agent {
    pub(super) fn send_file_completion_reply() -> &'static str {
        "I've sent the requested file. If you want any changes or another file, tell me exactly what to send."
    }

    pub(super) async fn latest_non_system_tool_result(
        &self,
        session_id: &str,
        max_chars: usize,
    ) -> Option<(String, String)> {
        let history = match tokio::time::timeout(
            Duration::from_secs(5),
            self.state.get_history(session_id, 80),
        )
        .await
        {
            Ok(Ok(history)) => history,
            Ok(Err(_)) => return None,
            Err(_) => {
                warn!(
                    session_id,
                    "Timed out while loading history for stall output excerpt"
                );
                return None;
            }
        };

        // Low-information tool names whose output is rarely useful as a user-facing
        // summary (e.g., "File written to /path, 200 bytes").  We still fall back to
        // them if nothing better is available.
        const LOW_INFO_TOOLS: &[&str] = &[
            "write_file",
            "edit_file",
            "manage_memories",
            "manage_people",
            "remember_fact",
            "check_environment", // diagnostic: lists installed tools, never a task result
        ];

        let clean_tool_content = |msg: &crate::traits::Message| -> Option<(String, String)> {
            if msg.role != "tool" {
                return None;
            }
            let cleaned = msg.primary_content()?;
            let cleaned = cleaned.trim();
            if cleaned.is_empty() {
                return None;
            }
            Some((
                msg.tool_name.clone().unwrap_or_default(),
                truncate_with_note(cleaned, max_chars),
            ))
        };

        // Only return output from informative tools (terminal, search, etc.).
        // State-changing tools (remember_fact, manage_memories, write_file, etc.)
        // produce confirmations ("Remembered: ...", "Forgotten: ...") that are
        // self-documenting.  Wrapping them with "Here is the latest tool output:"
        // creates confusing debugging-style messages.  When only low-info tools
        // ran, return None so the LLM's natural response passes through instead.
        //
        // IMPORTANT: Stop at the first `user` message boundary to avoid leaking
        // tool results from previous interactions into the current response.
        let mut hit_user_boundary = false;
        for msg in history.iter().rev() {
            if msg.role == "user" {
                hit_user_boundary = true;
            }
            if hit_user_boundary && msg.role == "tool" {
                // This tool result is from a previous interaction — stop.
                break;
            }
            let tool_name = msg.tool_name.as_deref().unwrap_or("");
            if LOW_INFO_TOOLS.contains(&tool_name) {
                continue;
            }
            if let Some(result) = clean_tool_content(msg) {
                return Some(result);
            }
        }
        None
    }

    pub(super) async fn latest_non_system_tool_output_excerpt(
        &self,
        session_id: &str,
        max_chars: usize,
    ) -> Option<String> {
        self.latest_non_system_tool_result(session_id, max_chars)
            .await
            .map(|(_, content)| content)
    }

    pub(super) async fn run_stopping_phase(
        &self,
        ctx: &mut StoppingPhaseCtx<'_>,
    ) -> anyhow::Result<StoppingPhaseOutcome> {
        let emitter = ctx.emitter;
        let task_id = ctx.task_id;
        let session_id = ctx.session_id;
        let iteration = ctx.iteration;
        let task_start = ctx.task_start;
        let learning_ctx = &mut *ctx.learning_ctx;
        let hard_cap = ctx.hard_cap;
        let task_tokens_used = ctx.task_tokens_used;
        let mut effective_task_budget = *ctx.effective_task_budget;
        let mut budget_warning_sent = *ctx.budget_warning_sent;
        let pending_system_messages = &mut *ctx.pending_system_messages;
        let mut budget_extensions_count = *ctx.budget_extensions_count;
        let user_role = ctx.user_role;
        let evidence_gain_count = ctx.evidence_gain_count;
        let stall_count = ctx.stall_count;
        let deferred_no_tool_streak = ctx.deferred_no_tool_streak;
        let consecutive_same_tool = ctx.consecutive_same_tool;
        let consecutive_same_tool_arg_hashes = ctx.consecutive_same_tool_arg_hashes;
        let total_successful_tool_calls = ctx.total_successful_tool_calls;
        let mut pending_background_ack = std::mem::take(ctx.pending_background_ack);
        let status_tx = ctx.status_tx;
        let resolved_goal_id = ctx.resolved_goal_id;
        let is_scheduled_goal = ctx.is_scheduled_goal;
        let mut effective_daily_budget = *ctx.effective_daily_budget;
        let mut effective_goal_daily_budget = *ctx.effective_goal_daily_budget;
        let successful_send_file_keys = ctx.successful_send_file_keys;
        let mut model = ctx.model.clone();
        let soft_threshold = ctx.soft_threshold;
        let soft_warn_at = ctx.soft_warn_at;
        let mut soft_limit_warned = *ctx.soft_limit_warned;
        let mut last_progress_summary = *ctx.last_progress_summary;
        let tool_failure_count = ctx.tool_failure_count;
        let mut session_summary = ctx.session_summary.clone();
        let mut policy_bundle = ctx.policy_bundle.clone();
        let user_text = ctx.user_text;
        let available_capabilities = ctx.available_capabilities;
        let llm_router = ctx.llm_router;
        let mut last_escalation_iteration = *ctx.last_escalation_iteration;
        let mut consecutive_clean_iterations = *ctx.consecutive_clean_iterations;
        let max_budget_extensions = ctx.max_budget_extensions;
        let hard_token_cap = ctx.hard_token_cap;
        let execution_state = &mut *ctx.execution_state;
        let mut force_text_response = *ctx.force_text_response;
        let completion_progress = &mut *ctx.completion_progress;
        let turn_context = ctx.turn_context;
        let mut validation_state = ctx.validation_state.clone();

        macro_rules! commit_state {
            () => {
                *ctx.effective_task_budget = effective_task_budget;
                *ctx.budget_warning_sent = budget_warning_sent;
                *ctx.budget_extensions_count = budget_extensions_count;
                *ctx.effective_daily_budget = effective_daily_budget;
                *ctx.effective_goal_daily_budget = effective_goal_daily_budget;
                *ctx.pending_background_ack = pending_background_ack.clone();
                *ctx.model = model.clone();
                *ctx.soft_limit_warned = soft_limit_warned;
                *ctx.last_progress_summary = last_progress_summary;
                *ctx.session_summary = session_summary.clone();
                *ctx.policy_bundle = policy_bundle.clone();
                *ctx.last_escalation_iteration = last_escalation_iteration;
                *ctx.consecutive_clean_iterations = consecutive_clean_iterations;
                *ctx.force_text_response = force_text_response;
                *ctx.validation_state = validation_state.clone();
            };
        }

        if let Some(limit) = execution_state.exhausted_limit(task_tokens_used, task_start.elapsed())
        {
            let text_only_turn = turn_contract_is_text_only(turn_context);
            let recoverable_tool_snapshot_present = if total_successful_tool_calls > 0
                && !has_task_relevant_progress(turn_context, completion_progress)
            {
                self.latest_non_system_tool_result(session_id, 1200)
                    .await
                    .is_some()
            } else {
                false
            };
            let made_progress = has_task_relevant_progress(turn_context, completion_progress)
                || recoverable_tool_snapshot_present;
            let has_executed_concrete_work = has_any_concrete_execution(
                turn_context,
                completion_progress,
                recoverable_tool_snapshot_present,
                total_successful_tool_calls,
            );
            let final_response_only = only_final_response_remains(
                turn_context,
                completion_progress,
                recoverable_tool_snapshot_present,
                total_successful_tool_calls,
            );
            let can_shift_to_final_response_closeout =
                final_response_only && !execution_state.final_response_closeout_active;

            if can_shift_to_final_response_closeout {
                validation_state.record_failure(ValidationFailure::BudgetExhausted);
                force_text_response = true;
                pending_system_messages.push(SystemDirective::DeferredProvideConcreteResults);
                execution_state.suspend_budget_for_final_response();
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::ExecutionStateSnapshot,
                    "Suspending execution budget after concrete progress so the agent can finish the final answer"
                        .to_string(),
                    json!({
                        "condition": "execution_budget_shifted_to_final_response_closeout",
                        "budget_limit": limit,
                        "execution_state": execution_state.clone(),
                        "validation_state": validation_state.clone(),
                        "observational_progress": completion_progress.observation_count,
                        "mutation_progress": completion_progress.mutation_count,
                        "verification_progress": completion_progress.verification_count,
                        "recoverable_tool_snapshot_present": recoverable_tool_snapshot_present,
                    }),
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Proceed);
            }

            let closeout_grace_available = !validation_state
                .failed_checks
                .contains(&ValidationFailure::BudgetExhausted)
                && made_progress
                && matches!(
                    execution_state.last_outcome,
                    Some(StepExecutionOutcome::Progress | StepExecutionOutcome::BackgroundDetached)
                );
            if closeout_grace_available {
                validation_state.record_failure(ValidationFailure::BudgetExhausted);
                force_text_response = true;
                pending_system_messages.push(SystemDirective::DeferredProvideConcreteResults);
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::ExecutionStateSnapshot,
                    "Allowing one force-text closeout after execution budget exhaustion"
                        .to_string(),
                    json!({
                        "condition": "execution_budget_exhausted_closeout_grace",
                        "budget_limit": limit,
                        "execution_state": execution_state.clone(),
                        "validation_state": validation_state.clone(),
                    }),
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Proceed);
            }

            let plain_text_recovery_available = text_only_turn
                && !force_text_response
                && !validation_state
                    .failed_checks
                    .contains(&ValidationFailure::BudgetExhausted);
            if plain_text_recovery_available {
                validation_state.record_failure(ValidationFailure::BudgetExhausted);
                force_text_response = true;
                pending_system_messages.push(SystemDirective::ToolModeDisabledPlainText);
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::ExecutionStateSnapshot,
                    "Allowing one plain-text recovery after execution budget exhaustion"
                        .to_string(),
                    json!({
                        "condition": "execution_budget_exhausted_plain_text_recovery",
                        "budget_limit": limit,
                        "execution_state": execution_state.clone(),
                        "validation_state": validation_state.clone(),
                    }),
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Proceed);
            }

            validation_state.record_failure(ValidationFailure::BudgetExhausted);
            let request = if made_progress {
                build_reduce_scope_request_with_plan(
                    turn_context,
                    learning_ctx,
                    Some(execution_state),
                    format!(
                        "I hit the current execution budget limit ({}) before I could safely continue.",
                        limit.as_str()
                    ),
                    "Confirm the reduced scope or next concrete target I should spend the remaining effort on.",
                    "I will continue only on that narrowed scope and then report what changed.",
                )
            } else if matches!(
                execution_state.last_outcome,
                Some(StepExecutionOutcome::NonrecoverableFailure)
            ) {
                validation_state.record_failure(ValidationFailure::NonrecoverableFailure);
                build_abandon_request(
                    turn_context,
                    learning_ctx,
                    format!(
                        "The current execution path failed nonrecoverably before making progress, and I also hit the {} limit.",
                        limit.as_str()
                    ),
                    "A different plan, target, or operator intervention before I attempt this again.",
                    "I will abandon this path and wait for a revised instruction instead of retrying the same broken approach.",
                )
            } else if !has_executed_concrete_work {
                build_abandon_request(
                    turn_context,
                    learning_ctx,
                    format!(
                        "I hit the current execution budget limit ({}) while planning or retrying, before any concrete tool or verification step could complete.",
                        limit.as_str()
                    ),
                    "A narrower target, a revised approach, or explicit permission to spend more execution budget on a new attempt.",
                    "I will stop this execution path here instead of pretending partial work exists when no concrete step completed.",
                )
            } else {
                build_partial_done_blocked_request_with_plan(
                    turn_context,
                    learning_ctx,
                    Some(execution_state),
                    format!(
                        "I hit the current execution budget limit ({}) before I could safely continue.",
                        limit.as_str()
                    ),
                    "A narrower scope or explicit approval to continue beyond the current execution envelope.",
                    "I will either continue with the reduced scope or spend the additional budget on the next concrete step.",
                )
            };
            learning_ctx.record_replay_note(
                ReplayNoteCategory::ValidationFailure,
                "execution_budget_exhausted",
                format!(
                    "Stopped because execution budget limit {} was exhausted before the task could finish safely.",
                    limit.as_str()
                ),
                true,
            );
            let retry_code = match request.outcome {
                ValidationOutcome::ReduceScope => "reduce_scope",
                ValidationOutcome::Abandon => "abandon",
                _ => "execution_budget_blocked",
            };
            learning_ctx.record_replay_note(
                ReplayNoteCategory::RetryReason,
                retry_code,
                format!(
                    "Budget exhaustion forced {:?} instead of continuing the same execution path.",
                    request.outcome
                ),
                true,
            );
            warn!(
                session_id,
                task_id,
                iteration,
                outcome = ?request.outcome,
                budget_limit = %limit.as_str(),
                tool_calls = learning_ctx.tool_calls.len(),
                "Execution budget exhausted — rendering HumanInterventionRequest"
            );
            self.emit_warning_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::ExecutionStateSnapshot,
                "Stopping because execution budget is exhausted".to_string(),
                json!({
                    "condition": "execution_budget_exhausted",
                    "budget_limit": limit,
                    "execution_state": execution_state.clone(),
                    "validation_state": validation_state.clone(),
                    "request": request.clone(),
                }),
            )
            .await;
            let reply = request.render_user_message();
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
            self.append_assistant_message_with_event(emitter, &assistant_msg, &model, None, None)
                .await?;
            self.emit_task_end(
                emitter,
                task_id,
                TaskStatus::Completed,
                task_start,
                iteration,
                learning_ctx.tool_calls.len(),
                None,
                Some(reply.chars().take(200).collect()),
            )
            .await;
            commit_state!();
            return Ok(StoppingPhaseOutcome::Return(Ok(reply)));
        }

        if self.depth == 0 {
            if let Some(background_ack) = pending_background_ack.take() {
                info!(
                    session_id,
                    iteration,
                    total_successful_tool_calls,
                    tool_calls = learning_ctx.tool_calls.len(),
                    "Background handoff: stopping loop and returning summary"
                );
                self.emit_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Stopping condition fired: deterministic background handoff".to_string(),
                    json!({
                        "condition":"background_detach_handoff",
                        "total_successful_tool_calls": total_successful_tool_calls
                    }),
                )
                .await;

                // Build a richer response that includes an activity summary
                // so the user knows what was accomplished before the background
                // task was started, not just the technical "moved to background" text.
                // NOTE: We use display_tool_call() to convert "tool_name(args)" to
                // a user-friendly format that won't be stripped by
                // strip_tool_name_references() (which replaces raw tool_name(...)
                // patterns with "that").
                let reply = if learning_ctx.tool_calls.is_empty() {
                    background_ack.clone()
                } else {
                    let mut summary =
                        String::from("Here's what I did before the background task started:\n");
                    for (i, call) in learning_ctx.tool_calls.iter().enumerate() {
                        summary.push_str(&format!(
                            "{}. {}\n",
                            i + 1,
                            post_task::display_tool_call(call)
                        ));
                    }
                    summary.push_str(&format!("\n{}", background_ack));
                    summary
                };

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
                self.append_assistant_message_with_event(
                    emitter,
                    &assistant_msg,
                    &model,
                    None,
                    None,
                )
                .await?;
                self.emit_task_end(
                    emitter,
                    task_id,
                    TaskStatus::Completed,
                    task_start,
                    iteration,
                    learning_ctx.tool_calls.len(),
                    None,
                    Some(reply.chars().take(200).collect()),
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Return(Ok(reply)));
            }
        }

        // === STOPPING CONDITIONS ===

        // 1-2. Pure stopping conditions (hard cap + timeout) in precedence order.
        let elapsed_secs = task_start.elapsed().as_secs();
        if let Some(condition) = (PureStoppingInputs {
            iteration,
            hard_cap,
            timeout_secs: ctx.effective_task_timeout.map(|timeout| timeout.as_secs()),
            elapsed_secs,
            task_token_budget: None,
            task_tokens_used,
            stall_count: 0,
            max_stall_iterations: MAX_STALL_ITERATIONS,
        })
        .evaluate()
        {
            match condition {
                StoppingCondition::HardIterationCap { cap, .. } => {
                    warn!(session_id, iteration, cap, "Hard iteration cap reached");
                    self.emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::StoppingCondition,
                        "Stopping condition fired: hard iteration cap".to_string(),
                        json!({"condition":"hard_iteration_cap","cap":cap,"iteration":iteration}),
                    )
                    .await;
                    let result = self
                        .graceful_cap_response(emitter, session_id, learning_ctx, iteration)
                        .await;
                    let (status, error, summary) = match &result {
                        Ok(reply) => (
                            TaskStatus::Completed,
                            None,
                            Some(reply.chars().take(200).collect()),
                        ),
                        Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                    };
                    if status == TaskStatus::Failed {
                        record_failed_task_tokens(task_tokens_used);
                    }
                    self.emit_task_end(
                        emitter,
                        task_id,
                        status,
                        task_start,
                        iteration,
                        learning_ctx.tool_calls.len(),
                        error,
                        summary,
                    )
                    .await;
                    commit_state!();
                    return Ok(StoppingPhaseOutcome::Return(result));
                }
                StoppingCondition::TaskTimeout {
                    timeout_secs,
                    elapsed_secs,
                } => {
                    let elapsed = Duration::from_secs(elapsed_secs);
                    warn!(session_id, elapsed_secs, "Task timeout reached");
                    self.emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::StoppingCondition,
                        "Stopping condition fired: task timeout".to_string(),
                        json!({
                            "condition":"task_timeout",
                            "timeout_secs": timeout_secs,
                            "elapsed_secs": elapsed_secs
                        }),
                    )
                    .await;
                    let result = self
                        .graceful_timeout_response(emitter, session_id, learning_ctx, elapsed)
                        .await;
                    let (status, error, summary) = match &result {
                        Ok(reply) => (
                            TaskStatus::Completed,
                            None,
                            Some(reply.chars().take(200).collect()),
                        ),
                        Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                    };
                    if status == TaskStatus::Failed {
                        record_failed_task_tokens(task_tokens_used);
                    }
                    self.emit_task_end(
                        emitter,
                        task_id,
                        status,
                        task_start,
                        iteration,
                        learning_ctx.tool_calls.len(),
                        error,
                        summary,
                    )
                    .await;
                    commit_state!();
                    return Ok(StoppingPhaseOutcome::Return(result));
                }
                _ => {}
            }
        }

        // 3. Task token budget (if configured)
        let scheduled_run_budget_active = if is_scheduled_goal {
            if let (Some(goal_id), Some(registry)) = (
                resolved_goal_id.as_deref(),
                self.goal_token_registry.as_ref(),
            ) {
                registry.get_run_budget(goal_id).await.is_some()
            } else {
                false
            }
        } else {
            false
        };

        if let Some(budget) = effective_task_budget.filter(|_| !scheduled_run_budget_active) {
            // One-time warning at 80% of budget
            if budget > 0
                && !budget_warning_sent
                && task_tokens_used >= budget.saturating_mul(80) / 100
                && task_tokens_used < budget
            {
                budget_warning_sent = true;
                let pct = task_tokens_used.saturating_mul(100) / budget;
                warn!(
                    session_id,
                    tokens_used = task_tokens_used,
                    budget,
                    pct,
                    "Task token budget at 80%"
                );
                let task_hint = super::loop_utils::build_task_boundary_hint(user_text, 150);
                let task_anchor = if task_hint.is_empty() {
                    String::new()
                } else {
                    format!(" Current task: {}", task_hint)
                };
                pending_system_messages.push(SystemDirective::TaskTokenBudgetWarning {
                    used: task_tokens_used,
                    budget,
                    pct,
                    task_anchor,
                });
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Task token budget warning threshold reached".to_string(),
                    json!({
                        "condition":"task_token_budget_warning",
                        "budget": budget,
                        "task_tokens_used": task_tokens_used,
                        "pct": pct
                    }),
                )
                .await;
            }

            if budget > 0 && task_tokens_used >= budget {
                // Try auto-extending if productive
                let cap_u64 = hard_token_cap as u64;
                let new_budget_u64 = budget
                    .saturating_mul(2)
                    .max(task_tokens_used.saturating_add(budget / 2))
                    .min(cap_u64);
                let productive = Self::has_meaningful_budget_progress(
                    evidence_gain_count,
                    total_successful_tool_calls,
                ) && post_task::is_productive(
                    learning_ctx,
                    stall_count,
                    consecutive_same_tool.1,
                    consecutive_same_tool_arg_hashes.len(),
                    total_successful_tool_calls,
                );
                if budget_extensions_count < max_budget_extensions
                    && budget < cap_u64
                    && new_budget_u64 > task_tokens_used
                    && user_role == UserRole::Owner
                    && productive
                {
                    let old_budget_i64 = budget as i64;
                    let new_budget_i64 = new_budget_u64 as i64;
                    budget_extensions_count += 1;
                    effective_task_budget = Some(new_budget_u64);
                    budget_warning_sent = false;
                    info!(
                        session_id,
                        old_budget = old_budget_i64,
                        new_budget = new_budget_i64,
                        extension = budget_extensions_count,
                        "Auto-extended task token budget"
                    );
                    pending_system_messages.push(SystemDirective::TaskBudgetAutoExtended {
                        old_budget: old_budget_i64,
                        new_budget: new_budget_i64,
                        extension: budget_extensions_count,
                        max_extensions: max_budget_extensions,
                    });
                    send_status(
                        status_tx,
                        StatusUpdate::BudgetExtended {
                            old_budget: old_budget_i64,
                            new_budget: new_budget_i64,
                            extension: budget_extensions_count,
                            max_extensions: max_budget_extensions,
                        },
                    );
                    self.emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::BudgetAutoExtension,
                        "Auto-extended task token budget on productive progress".to_string(),
                        json!({
                            "condition": "task_token_budget_extension",
                            "old_budget": old_budget_i64,
                            "new_budget": new_budget_i64,
                            "extension": budget_extensions_count,
                            "max_extensions": max_budget_extensions,
                            "total_successful_tool_calls": total_successful_tool_calls,
                        }),
                    )
                    .await;
                    commit_state!();
                    return Ok(StoppingPhaseOutcome::ContinueLoop);
                }

                if budget < cap_u64
                    && new_budget_u64 > task_tokens_used
                    && self
                        .request_budget_continue_approval(
                            emitter,
                            task_id,
                            iteration,
                            session_id,
                            user_role,
                            "task",
                            task_tokens_used as i64,
                            budget as i64,
                            new_budget_u64 as i64,
                        )
                        .await
                {
                    effective_task_budget = Some(new_budget_u64);
                    budget_warning_sent = false;
                    pending_system_messages.push(SystemDirective::TaskBudgetExtensionApproved {
                        old_budget: budget as i64,
                        new_budget: new_budget_u64 as i64,
                    });
                    self.emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::BudgetAutoExtension,
                        "Extended task token budget via owner approval".to_string(),
                        json!({
                            "condition": "task_token_budget_extension_manual",
                            "approval_state": ApprovalState::Granted,
                            "old_budget": budget,
                            "new_budget": new_budget_u64,
                            "task_tokens_used": task_tokens_used,
                        }),
                    )
                    .await;
                    commit_state!();
                    return Ok(StoppingPhaseOutcome::ContinueLoop);
                }

                warn!(
                    session_id,
                    tokens_used = task_tokens_used,
                    budget,
                    "Task token budget exhausted"
                );
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Stopping condition fired: task token budget exhausted".to_string(),
                    json!({
                        "condition":"task_token_budget",
                        "budget": budget,
                        "task_tokens_used": task_tokens_used
                    }),
                )
                .await;
                let alert_msg = format!(
                        "Token alert: execution in session '{}' hit task token budget (used {} / limit {}). The run was stopped to prevent overspending.",
                        session_id,
                        task_tokens_used,
                        budget
                    );
                self.fanout_token_alert(
                    self.goal_id.as_deref(),
                    session_id,
                    &alert_msg,
                    Some(session_id),
                )
                .await;
                let result = self
                    .graceful_budget_response(emitter, session_id, learning_ctx, task_tokens_used)
                    .await;
                let (status, error, summary) = match &result {
                    Ok(reply) => (
                        TaskStatus::Completed,
                        None,
                        Some(reply.chars().take(200).collect()),
                    ),
                    Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                };
                if status == TaskStatus::Failed {
                    record_failed_task_tokens(task_tokens_used);
                }
                self.emit_task_end(
                    emitter,
                    task_id,
                    status,
                    task_start,
                    iteration,
                    learning_ctx.tool_calls.len(),
                    error,
                    summary,
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Return(result));
            }
        }

        // 4. Goal budget controls.
        // Scheduled goals are admitted by their daily budget before a new run
        // starts, then governed by their per-run budget while active.
        if let Some(ref goal_id) = resolved_goal_id {
            if is_scheduled_goal {
                if let Some(run_budget_status) = if let Some(registry) = &self.goal_token_registry {
                    registry
                        .update_run_health(
                            goal_id,
                            Self::scheduled_run_health_snapshot(
                                learning_ctx,
                                evidence_gain_count,
                                stall_count,
                                consecutive_same_tool.1,
                                consecutive_same_tool_arg_hashes.len(),
                                total_successful_tool_calls,
                            ),
                        )
                        .await
                } else {
                    None
                } {
                    persist_scheduled_run_state(&self.state, goal_id, None, &run_budget_status)
                        .await;
                    let mut run_budget_ctx = graceful::ScheduledRunBudgetControlCtx {
                        emitter,
                        task_id,
                        session_id,
                        iteration,
                        goal_id,
                        status: &run_budget_status,
                        user_role,
                        status_tx,
                        max_budget_extensions,
                        hard_token_cap,
                    };
                    if let graceful::ScheduledRunBudgetControlOutcome::Exhausted {
                        tokens_used,
                        budget_per_check,
                    } = self
                        .enforce_scheduled_run_budget_control(&mut run_budget_ctx)
                        .await
                    {
                        warn!(
                            session_id,
                            iteration,
                            goal_id = %goal_id,
                            tokens_used,
                            budget_per_check,
                            "Scheduled run budget exhausted"
                        );
                        self.emit_warning_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::StoppingCondition,
                            "Stopping condition fired: scheduled run budget exhausted".to_string(),
                            json!({
                                "condition":"scheduled_run_budget",
                                "goal_id": goal_id,
                                "budget_per_check": budget_per_check,
                                "tokens_used": tokens_used
                            }),
                        )
                        .await;
                        let alert_msg = format!(
                            "Token alert: scheduled run for goal '{}' hit per-run budget (used {} / limit {}). Execution was stopped because the run no longer appeared productive.",
                            goal_id, tokens_used, budget_per_check
                        );
                        self.fanout_token_alert(
                            Some(goal_id.as_str()),
                            session_id,
                            &alert_msg,
                            Some(session_id),
                        )
                        .await;
                        let result = self
                            .graceful_scheduled_run_budget_response(
                                emitter,
                                session_id,
                                learning_ctx,
                                tokens_used,
                                budget_per_check,
                            )
                            .await;
                        let (status, error, summary) = match &result {
                            Ok(reply) => (
                                TaskStatus::Completed,
                                None,
                                Some(reply.chars().take(200).collect()),
                            ),
                            Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                        };
                        if status == TaskStatus::Failed {
                            record_failed_task_tokens(task_tokens_used);
                        }
                        self.emit_task_end(
                            emitter,
                            task_id,
                            status,
                            task_start,
                            iteration,
                            learning_ctx.tool_calls.len(),
                            error,
                            summary,
                        )
                        .await;
                        commit_state!();
                        return Ok(StoppingPhaseOutcome::Return(result));
                    }
                }
            } else {
                match self
                    .state
                    .add_goal_tokens_and_get_budget_status(goal_id, 0)
                    .await
                {
                    Ok(Some(status)) => {
                        let mut goal_budget_ctx = graceful::GoalBudgetControlCtx {
                            emitter,
                            task_id,
                            session_id,
                            iteration,
                            goal_id,
                            status: &status,
                            user_role,
                            learning_ctx,
                            evidence_gain_count,
                            stall_count,
                            consecutive_same_tool_count: consecutive_same_tool.1,
                            consecutive_same_tool_unique_args: consecutive_same_tool_arg_hashes
                                .len(),
                            total_successful_tool_calls,
                            pending_system_messages,
                            status_tx,
                            is_scheduled_goal,
                            effective_goal_daily_budget: &mut effective_goal_daily_budget,
                            budget_extensions_count: &mut budget_extensions_count,
                            max_budget_extensions,
                            hard_token_cap,
                            source: graceful::GoalBudgetCheckSource::PreCheck,
                        };
                        match self
                            .enforce_goal_daily_budget_control(&mut goal_budget_ctx)
                            .await
                        {
                            graceful::GoalBudgetControlOutcome::Continue => {}
                            graceful::GoalBudgetControlOutcome::Exhausted {
                                tokens_used_today,
                                budget_daily,
                            } => {
                                warn!(
                                    session_id,
                                    iteration,
                                    goal_id = %goal_id,
                                    tokens_used_today,
                                    budget_daily,
                                    "Goal daily token budget exhausted"
                                );
                                self.emit_warning_decision_point(
                                    emitter,
                                    task_id,
                                    iteration,
                                    DecisionType::StoppingCondition,
                                    "Stopping condition fired: goal daily token budget exhausted"
                                        .to_string(),
                                    json!({
                                        "condition":"goal_daily_token_budget",
                                        "goal_id": goal_id,
                                        "budget_daily": budget_daily,
                                        "tokens_used_today": tokens_used_today
                                    }),
                                )
                                .await;
                                let alert_msg = format!(
                                    "Token alert: goal '{}' hit daily token budget (used {} / limit {}). Execution was stopped to prevent overspending.",
                                    goal_id, tokens_used_today, budget_daily
                                );
                                self.fanout_token_alert(
                                    Some(goal_id.as_str()),
                                    session_id,
                                    &alert_msg,
                                    Some(session_id),
                                )
                                .await;
                                let result = self
                                    .graceful_goal_daily_budget_response(
                                        emitter,
                                        session_id,
                                        learning_ctx,
                                        tokens_used_today,
                                        budget_daily,
                                        is_scheduled_goal,
                                    )
                                    .await;
                                let (status, error, summary) = match &result {
                                    Ok(reply) => (
                                        TaskStatus::Completed,
                                        None,
                                        Some(reply.chars().take(200).collect()),
                                    ),
                                    Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                                };
                                if status == TaskStatus::Failed {
                                    record_failed_task_tokens(task_tokens_used);
                                }
                                self.emit_task_end(
                                    emitter,
                                    task_id,
                                    status,
                                    task_start,
                                    iteration,
                                    learning_ctx.tool_calls.len(),
                                    error,
                                    summary,
                                )
                                .await;
                                commit_state!();
                                return Ok(StoppingPhaseOutcome::Return(result));
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(
                            session_id,
                            iteration,
                            goal_id = %goal_id,
                            error = %e,
                            "Failed to check goal daily token budget"
                        );
                    }
                }
            }
        }

        // 5. Daily token budget (existing global limit)
        if let Some(daily_budget) = effective_daily_budget {
            let today_start = Utc::now().format("%Y-%m-%d 00:00:00").to_string();
            if let Ok(records) = self.state.get_token_usage_since(&today_start).await {
                let total: u64 = records
                    .iter()
                    .map(|r| (r.input_tokens + r.output_tokens) as u64)
                    .sum();
                if total >= daily_budget {
                    let cap_u64 = hard_token_cap as u64;
                    let new_daily_budget = daily_budget
                        .saturating_mul(2)
                        .max(total.saturating_add(daily_budget / 2))
                        .min(cap_u64);
                    let productive = Self::has_meaningful_budget_progress(
                        evidence_gain_count,
                        total_successful_tool_calls,
                    ) && post_task::is_productive(
                        learning_ctx,
                        stall_count,
                        consecutive_same_tool.1,
                        consecutive_same_tool_arg_hashes.len(),
                        total_successful_tool_calls,
                    );
                    if budget_extensions_count < max_budget_extensions
                        && daily_budget < cap_u64
                        && new_daily_budget > total
                        && user_role == UserRole::Owner
                        && productive
                    {
                        budget_extensions_count += 1;
                        effective_daily_budget = Some(new_daily_budget);
                        pending_system_messages.push(
                            SystemDirective::GlobalDailyBudgetAutoExtended {
                                old_budget: daily_budget as i64,
                                new_budget: new_daily_budget as i64,
                                extension: budget_extensions_count,
                                max_extensions: max_budget_extensions,
                            },
                        );
                        send_status(
                            status_tx,
                            StatusUpdate::BudgetExtended {
                                old_budget: daily_budget as i64,
                                new_budget: new_daily_budget as i64,
                                extension: budget_extensions_count,
                                max_extensions: max_budget_extensions,
                            },
                        );
                        self.emit_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::BudgetAutoExtension,
                            "Auto-extended global daily token budget on productive progress"
                                .to_string(),
                            json!({
                                "condition":"daily_token_budget_extension",
                                "old_budget": daily_budget,
                                "new_budget": new_daily_budget,
                                "extension": budget_extensions_count,
                                "max_extensions": max_budget_extensions,
                                "total_today": total
                            }),
                        )
                        .await;
                        commit_state!();
                        return Ok(StoppingPhaseOutcome::ContinueLoop);
                    }
                    if daily_budget < cap_u64
                        && new_daily_budget > total
                        && self
                            .request_budget_continue_approval(
                                emitter,
                                task_id,
                                iteration,
                                session_id,
                                user_role,
                                "global daily",
                                total as i64,
                                daily_budget as i64,
                                new_daily_budget as i64,
                            )
                            .await
                    {
                        effective_daily_budget = Some(new_daily_budget);
                        pending_system_messages.push(
                            SystemDirective::GlobalDailyBudgetExtensionApproved {
                                old_budget: daily_budget as i64,
                                new_budget: new_daily_budget as i64,
                            },
                        );
                        self.emit_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::BudgetAutoExtension,
                            "Extended global daily token budget via owner approval".to_string(),
                            json!({
                                "condition":"daily_token_budget_extension_manual",
                                "approval_state": ApprovalState::Granted,
                                "old_budget": daily_budget,
                                "new_budget": new_daily_budget,
                                "total_today": total
                            }),
                        )
                        .await;
                        commit_state!();
                        return Ok(StoppingPhaseOutcome::ContinueLoop);
                    }

                    self.emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::StoppingCondition,
                        "Stopping condition fired: daily token budget exhausted".to_string(),
                        json!({
                            "condition":"daily_token_budget",
                            "daily_budget": daily_budget,
                            "total_today": total
                        }),
                    )
                    .await;
                    let alert_msg = format!(
                            "Token alert: global daily token budget was exceeded (used {} / limit {}) while running session '{}'.",
                            total,
                            daily_budget,
                            session_id
                        );
                    self.fanout_token_alert(self.goal_id.as_deref(), session_id, &alert_msg, None)
                        .await;
                    let error_msg = format!(
                        "Daily token budget of {} exceeded (used: {}). Resets at midnight UTC.",
                        daily_budget, total
                    );
                    record_failed_task_tokens(task_tokens_used);
                    self.emit_task_end(
                        emitter,
                        task_id,
                        TaskStatus::Failed,
                        task_start,
                        iteration,
                        learning_ctx.tool_calls.len(),
                        Some(error_msg.clone()),
                        None,
                    )
                    .await;
                    commit_state!();
                    return Ok(StoppingPhaseOutcome::Return(Err(anyhow::anyhow!(
                        error_msg
                    ))));
                }
            }
        }

        // 6. Pre-execution deferral guard — the model keeps narrating
        // planned actions without issuing any tool calls.
        const MAX_PRE_TOOL_DEFERRALS: usize = 6;
        let loop_control_decision = LoopControlInputs {
            iteration,
            hard_cap: None,
            timeout_secs: None,
            elapsed_secs: 0,
            stall_count,
            max_stall_iterations: MAX_STALL_ITERATIONS,
            deferred_no_tool_streak,
            deferred_no_tool_switch_threshold: DEFERRED_NO_TOOL_SWITCH_THRESHOLD,
            deferred_no_tool_error_marker: DEFERRED_NO_TOOL_ERROR_MARKER,
            max_pre_tool_deferrals: MAX_PRE_TOOL_DEFERRALS,
            total_successful_tool_calls,
            recent_errors: &learning_ctx.errors,
        }
        .evaluate();

        if let Some(LoopControlDecision::PreToolDeferral {
            deferred_no_tool_streak: decision_streak,
            max_pre_tool_deferrals,
        }) = loop_control_decision
        {
            warn!(
                session_id,
                decision_streak, "Pre-tool deferral threshold reached"
            );
            self.emit_warning_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::StoppingCondition,
                "Stopping condition fired: repeated pre-tool deferrals".to_string(),
                json!({
                    "condition":"pre_tool_deferral_stall",
                    "deferred_no_tool_streak": decision_streak,
                    "max_pre_tool_deferrals": max_pre_tool_deferrals
                }),
            )
            .await;
            let reply = "I'm having trouble processing this request. Could you try rephrasing it or breaking it into smaller steps?"
                .to_string();
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
            self.append_assistant_message_with_event(emitter, &assistant_msg, &model, None, None)
                .await?;

            self.emit_task_end(
                emitter,
                task_id,
                TaskStatus::Failed,
                task_start,
                iteration,
                learning_ctx.tool_calls.len(),
                Some("Repeated pre-tool deferrals".to_string()),
                Some(reply.chars().take(200).collect()),
            )
            .await;
            record_failed_task_tokens(task_tokens_used);
            commit_state!();
            return Ok(StoppingPhaseOutcome::Return(Ok(reply)));
        }

        // 7. Stall detection — agent spinning without progress
        if let Some(LoopControlDecision::Stall {
            stall_count: detected_stall_count,
            max_stall_iterations: stall_limit,
            mode,
        }) = loop_control_decision
        {
            let stall_mode = mode.as_code();
            if !successful_send_file_keys.is_empty() && learning_ctx.errors.is_empty() {
                let reply = Self::send_file_completion_reply().to_string();
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Stopping condition fired after successful send_file; resolving as completed"
                        .to_string(),
                    json!({
                        "condition":"post_send_file_stall",
                        "stall_count": detected_stall_count,
                        "max_stall_iterations": stall_limit,
                        "stall_mode": stall_mode,
                        "successful_send_file_count": successful_send_file_keys.len()
                    }),
                )
                .await;

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
                self.append_assistant_message_with_event(
                    emitter,
                    &assistant_msg,
                    &model,
                    None,
                    None,
                )
                .await?;

                self.emit_task_end(
                    emitter,
                    task_id,
                    TaskStatus::Completed,
                    task_start,
                    iteration,
                    learning_ctx.tool_calls.len(),
                    None,
                    Some(reply.chars().take(200).collect()),
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Return(Ok(reply)));
            }

            let unrecovered_errors = learning_ctx
                .errors
                .iter()
                .filter(|(_, recovered)| !recovered)
                .count();
            let meaningful_progress = (total_successful_tool_calls >= 3
                || evidence_gain_count >= 2)
                && total_successful_tool_calls > unrecovered_errors;
            if meaningful_progress {
                warn!(
                    session_id,
                    detected_stall_count,
                    total_successful_tool_calls,
                    unrecovered_errors,
                    "Agent stalled after meaningful progress"
                );
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Stopping condition fired: stall after meaningful progress".to_string(),
                    json!({
                        "condition":"stall_with_progress",
                        "stall_count": detected_stall_count,
                        "max_stall_iterations": stall_limit,
                        "stall_mode": stall_mode,
                        "total_successful_tool_calls": total_successful_tool_calls,
                        "unrecovered_errors": unrecovered_errors
                    }),
                )
                .await;

                // Prefer surfacing the latest tool output directly when the work
                // was done but the model failed to compose a summary. This gives
                // the user concrete results instead of a generic canned message.
                if unrecovered_errors == 0 {
                    if let Some(tool_output) = self
                        .latest_non_system_tool_output_excerpt(session_id, 2500)
                        .await
                    {
                        let activity = post_task::categorize_tool_calls(&learning_ctx.tool_calls);
                        let mut reply =
                            String::from("Here's a summary of what was accomplished:\n\n");
                        if !activity.is_empty() {
                            reply.push_str(&activity);
                        }
                        reply.push_str("Latest output:\n\n");
                        reply.push_str(&tool_output);

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
                        self.append_assistant_message_with_event(
                            emitter,
                            &assistant_msg,
                            &model,
                            None,
                            None,
                        )
                        .await?;

                        self.emit_task_end(
                            emitter,
                            task_id,
                            TaskStatus::Completed,
                            task_start,
                            iteration,
                            learning_ctx.tool_calls.len(),
                            None,
                            Some(reply.chars().take(200).collect()),
                        )
                        .await;
                        commit_state!();
                        return Ok(StoppingPhaseOutcome::Return(Ok(reply)));
                    }
                }

                let result = self
                    .graceful_partial_stall_response(
                        emitter,
                        session_id,
                        learning_ctx,
                        !successful_send_file_keys.is_empty(),
                        tool_failure_count,
                    )
                    .await;
                let (status, error, summary) = match &result {
                    Ok(reply) => (
                        TaskStatus::Completed,
                        None,
                        Some(reply.chars().take(200).collect()),
                    ),
                    Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
                };
                if status == TaskStatus::Failed {
                    record_failed_task_tokens(task_tokens_used);
                }
                self.emit_task_end(
                    emitter,
                    task_id,
                    status,
                    task_start,
                    iteration,
                    learning_ctx.tool_calls.len(),
                    error,
                    summary,
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Return(result));
            }

            // Last-resort deterministic fallback: if tools succeeded and there were
            // no unrecovered errors, surface the latest tool output directly instead
            // of returning a generic "Stuck" message.
            if total_successful_tool_calls > 0 && unrecovered_errors == 0 {
                if let Some(tool_output) = self
                    .latest_non_system_tool_output_excerpt(session_id, 2500)
                    .await
                {
                    let reply = format!("Done. Here is the output:\n\n{}", tool_output);
                    self.emit_warning_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::StoppingCondition,
                        "Stopping condition fired: stall recovered by last tool output fallback"
                            .to_string(),
                        json!({
                            "condition":"stall_with_tool_output_fallback",
                            "stall_count": detected_stall_count,
                            "max_stall_iterations": stall_limit,
                            "stall_mode": stall_mode,
                            "total_successful_tool_calls": total_successful_tool_calls
                        }),
                    )
                    .await;

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
                    self.append_assistant_message_with_event(
                        emitter,
                        &assistant_msg,
                        &model,
                        None,
                        None,
                    )
                    .await?;

                    self.emit_task_end(
                        emitter,
                        task_id,
                        TaskStatus::Completed,
                        task_start,
                        iteration,
                        learning_ctx.tool_calls.len(),
                        None,
                        Some(reply.chars().take(200).collect()),
                    )
                    .await;
                    commit_state!();
                    return Ok(StoppingPhaseOutcome::Return(Ok(reply)));
                }
            }

            warn!(
                session_id,
                detected_stall_count, "Agent stalled - no progress detected"
            );
            self.emit_warning_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::StoppingCondition,
                "Stopping condition fired: stall threshold reached".to_string(),
                json!({
                    "condition":"stall",
                    "stall_count": detected_stall_count,
                    "max_stall_iterations": stall_limit,
                    "stall_mode": stall_mode
                }),
            )
            .await;
            // Before giving up, try a knowledge-only fallback (no tools).
            // If the model can answer from training knowledge, return that
            // instead of the unhelpful "I wasn't able to complete" message.
            let error_summary = if learning_ctx.errors.is_empty() {
                "tools produced no results".to_string()
            } else {
                learning_ctx
                    .errors
                    .iter()
                    .take(2)
                    .map(|(msg, _)| msg.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            };
            if let Some(result) = self
                .graceful_knowledge_fallback(
                    emitter,
                    session_id,
                    &learning_ctx.user_text,
                    &error_summary,
                )
                .await
            {
                info!(session_id, "Knowledge fallback succeeded after tool stall");
                self.emit_task_end(
                    emitter,
                    task_id,
                    TaskStatus::Completed,
                    task_start,
                    iteration,
                    learning_ctx.tool_calls.len(),
                    None,
                    result
                        .as_ref()
                        .ok()
                        .map(|r| r.chars().take(200).collect::<String>()),
                )
                .await;
                commit_state!();
                return Ok(StoppingPhaseOutcome::Return(result));
            }

            let result = self
                .graceful_stall_response(
                    emitter,
                    session_id,
                    learning_ctx,
                    !successful_send_file_keys.is_empty(),
                    tool_failure_count,
                )
                .await;
            let (status, error, summary) = match &result {
                Ok(reply) => (
                    TaskStatus::Failed,
                    Some("Agent stalled".to_string()),
                    Some(reply.chars().take(200).collect()),
                ),
                Err(e) => (TaskStatus::Failed, Some(e.to_string()), None),
            };
            if status == TaskStatus::Failed {
                record_failed_task_tokens(task_tokens_used);
            }
            self.emit_task_end(
                emitter,
                task_id,
                status,
                task_start,
                iteration,
                learning_ctx.tool_calls.len(),
                error,
                summary,
            )
            .await;
            commit_state!();
            return Ok(StoppingPhaseOutcome::Return(result));
        }

        // 6. Soft limit warning (warnings only, no forced stop)
        if let (Some(threshold), Some(warn_at)) = (soft_threshold, soft_warn_at) {
            if iteration >= warn_at && !soft_limit_warned {
                soft_limit_warned = true;
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Soft iteration warning threshold reached".to_string(),
                    json!({
                        "condition":"soft_iteration_warning",
                        "warn_at": warn_at,
                        "threshold": threshold,
                        "iteration": iteration
                    }),
                )
                .await;
                send_status(
                    status_tx,
                    StatusUpdate::IterationWarning {
                        current: iteration,
                        threshold,
                    },
                );
                info!(
                    session_id,
                    iteration, threshold, "Soft iteration limit warning"
                );
            }
        }

        // 7. Progress summary for long-running tasks (every 5 minutes)
        if last_progress_summary.elapsed() >= PROGRESS_SUMMARY_INTERVAL {
            let elapsed_mins = task_start.elapsed().as_secs() / 60;
            let last_tool_info = learning_ctx
                .tool_calls
                .last()
                .map(|tc| {
                    // Extract tool name from "tool_name(args)" format
                    tc.split('(').next().unwrap_or(tc).to_string()
                })
                .unwrap_or_default();
            let summary = if last_tool_info.is_empty() {
                format!(
                    "Working... {} iterations, {} tool calls, {} mins elapsed",
                    iteration,
                    learning_ctx.tool_calls.len(),
                    elapsed_mins
                )
            } else {
                format!(
                    "Working... {} iterations, {} tool calls, {} mins elapsed (last: {})",
                    iteration,
                    learning_ctx.tool_calls.len(),
                    elapsed_mins,
                    last_tool_info
                )
            };
            send_status(
                status_tx,
                StatusUpdate::ProgressSummary {
                    elapsed_mins,
                    summary,
                },
            );
            last_progress_summary = Instant::now();
        }

        // 8. Mid-loop adaptation: refresh + bounded escalation/de-escalation
        if self.policy_config.context_refresh_enforce {
            let max_same_tool_failures = tool_failure_count.values().copied().max().unwrap_or(0);
            let should_refresh =
                iteration >= 5 && (stall_count >= 1 || max_same_tool_failures >= 2);

            if should_refresh {
                POLICY_METRICS
                    .context_refresh_total
                    .fetch_add(1, Ordering::Relaxed);
                // Refresh summary context and re-score policy with fresh failure signal.
                if self.context_window_config.enabled {
                    session_summary = match tokio::time::timeout(
                        Duration::from_secs(5),
                        self.state.get_conversation_summary(session_id),
                    )
                    .await
                    {
                        Ok(Ok(summary)) => summary,
                        Ok(Err(e)) => {
                            warn!(
                                session_id,
                                iteration,
                                error = %e,
                                "Failed to refresh conversation summary"
                            );
                            None
                        }
                        Err(_) => {
                            warn!(
                                session_id,
                                iteration, "Timed out refreshing conversation summary"
                            );
                            None
                        }
                    };
                }
                policy_bundle = build_policy_bundle(user_text, available_capabilities, true);

                let can_escalate = last_escalation_iteration
                    .is_none_or(|last| iteration >= last.saturating_add(2));
                if can_escalate {
                    let reason = format!(
                        "refresh_trigger(iter={},stall={},same_tool_failures={})",
                        iteration, stall_count, max_same_tool_failures
                    );
                    if policy_bundle.policy.escalate(reason.clone()) {
                        POLICY_METRICS
                            .escalation_total
                            .fetch_add(1, Ordering::Relaxed);
                        last_escalation_iteration = Some(iteration);
                        if let Some(ref router) = llm_router {
                            let next_model = router
                                .select_for_profile(policy_bundle.policy.model_profile)
                                .to_string();
                            if next_model != model {
                                info!(
                                    session_id,
                                    iteration,
                                    reason = %reason,
                                    from_model = %model,
                                    to_model = %next_model,
                                    "Escalated model profile mid-loop"
                                );
                                model = next_model;
                            }
                        }
                    }
                }
                consecutive_clean_iterations = 0;
            } else if consecutive_clean_iterations >= 2 {
                // Bounded de-escalation only after a stable clean window.
                if policy_bundle.policy.deescalate() {
                    if let Some(ref router) = llm_router {
                        let next_model = router
                            .select_for_profile(policy_bundle.policy.model_profile)
                            .to_string();
                        if next_model != model {
                            info!(
                                session_id,
                                iteration,
                                from_model = %model,
                                to_model = %next_model,
                                "De-escalated model profile after stable window"
                            );
                            model = next_model;
                        }
                    }
                }
                consecutive_clean_iterations = 0;
            }
        }

        commit_state!();
        Ok(StoppingPhaseOutcome::Proceed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recoverable_tool_snapshot_counts_as_concrete_progress() {
        let turn_context = TurnContext::default();
        let completion_progress = CompletionProgress::default();

        assert!(has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
        assert!(only_final_response_remains(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
    }

    #[test]
    fn successful_tool_calls_count_as_concrete_work() {
        let turn_context = TurnContext::default();
        let completion_progress = CompletionProgress::default();

        // No snapshot, no progress, but successful tool calls → concrete work
        assert!(has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            false,
            1,
        ));
        // Zero successful tool calls and no other progress → not concrete
        assert!(!has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            false,
            0,
        ));
    }

    #[test]
    fn verification_pending_prevents_final_response_closeout_even_with_snapshot() {
        let turn_context = TurnContext {
            completion_contract: CompletionContract {
                requires_observation: true,
                ..CompletionContract::default()
            },
            ..TurnContext::default()
        };
        let completion_progress = CompletionProgress {
            verification_pending: true,
            ..CompletionProgress::default()
        };

        assert!(has_any_concrete_execution(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
        assert!(!only_final_response_remains(
            &turn_context,
            &completion_progress,
            true,
            0,
        ));
    }
}
