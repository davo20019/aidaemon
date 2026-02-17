use super::bootstrap_phase::{BootstrapCtx, BootstrapData, BootstrapOutcome};
use super::consultant_phase::{ConsultantPhaseCtx, ConsultantPhaseOutcome};
use super::llm_phase::{LlmPhaseCtx, LlmPhaseOutcome};
use super::message_build_phase::{MessageBuildCtx, MessageBuildData};
use super::stopping_phase::{StoppingPhaseCtx, StoppingPhaseOutcome};
use super::tool_execution_phase::{ToolExecutionCtx, ToolExecutionOutcome};
use super::tool_prelude_phase::{ToolPreludeCtx, ToolPreludeOutcome};
use super::*;

impl Agent {
    /// Run the agentic loop for a user message in the given session.
    /// Returns the final assistant text response.
    /// `heartbeat` is an optional atomic timestamp updated on each activity point.
    /// Channels pass `Some(heartbeat)` so the typing indicator can detect stalls;
    /// sub-agents, triggers, and tests pass `None`.
    pub(super) async fn handle_message_impl(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String> {
        touch_heartbeat(&heartbeat);

        let bootstrap_outcome = self
            .run_bootstrap_phase(&BootstrapCtx {
                session_id,
                user_text,
                status_tx: status_tx.clone(),
                user_role,
                channel_ctx: &channel_ctx,
            })
            .await?;
        let BootstrapData {
            task_id,
            emitter,
            mut learning_ctx,
            is_personal_memory_recall_turn,
            is_reaffirmation_challenge_turn,
            requests_external_verification,
            restrict_to_personal_memory_tools,
            personal_memory_tool_call_cap,
            tools_allowed_for_user,
            mut available_capabilities,
            mut base_tool_defs,
            mut tool_defs,
            mut policy_bundle,
            llm_provider,
            llm_router,
            mut model,
            mut consultant_pass_active,
            route_failsafe_active,
            system_prompt,
            pinned_memories,
            mut session_summary,
        } = match bootstrap_outcome {
            BootstrapOutcome::Return(result) => return result,
            BootstrapOutcome::Continue(data) => *data,
        };
        let turn_context = self
            .build_turn_context_from_recent_history(session_id, user_text)
            .await;
        let followup_mode = turn_context
            .followup_mode
            .map(|mode| mode.as_str())
            .unwrap_or("unknown");
        let turn_context_reasons: Vec<&'static str> = turn_context
            .reasons
            .iter()
            .map(|reason| reason.as_code())
            .collect();
        info!(
            session_id,
            followup_mode,
            reasons = ?turn_context_reasons,
            primary_project_scope = ?turn_context.primary_project_scope,
            allow_multi_project_scope = turn_context.allow_multi_project_scope,
            "Turn context resolved"
        );
        // 3. Agentic loop — runs until natural completion or safety limits
        let task_start = Instant::now();
        let mut last_progress_summary = Instant::now();
        let mut iteration: usize = 0;
        let mut stall_count: usize = 0;
        let mut deferred_no_tool_streak: usize = 0;
        let mut deferred_no_tool_model_switches: usize = 0;
        let mut total_successful_tool_calls: usize = 0;
        let mut total_tool_calls_attempted: usize = 0;
        let mut task_tokens_used: u64 = 0;
        let mut tool_failure_count: HashMap<String, usize> = HashMap::new();
        let mut tool_call_count: HashMap<String, usize> = HashMap::new();
        let mut personal_memory_tool_calls: usize = 0;
        let mut no_evidence_result_streak: usize = 0;
        let mut no_evidence_tools_seen: HashSet<String> = HashSet::new();
        let mut evidence_gain_count: usize = 0;
        // Track which error solutions were injected so we can credit them on recovery.
        let mut pending_error_solution_ids: Vec<i64> = Vec::new();
        // In-session error learning: track repeated failures by (tool, normalized error pattern).
        let mut tool_failure_patterns: HashMap<(String, String), usize> = HashMap::new();
        let mut last_tool_failure: Option<(String, String)> = None;
        let mut in_session_learned: HashSet<(String, String)> = HashSet::new();
        let mut unknown_tools: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut recent_tool_calls: VecDeque<u64> = VecDeque::with_capacity(RECENT_CALLS_WINDOW);
        // Tracks consecutive calls to the same tool name, plus the set of
        // unique argument hashes seen during the streak.  When every call in
        // the streak has unique args the agent is likely making progress (e.g.
        // running different terminal commands), so we only trigger the stall
        // guard when the ratio of unique args is low.
        let mut consecutive_same_tool: (String, usize) = (String::new(), 0);
        let mut consecutive_same_tool_arg_hashes: HashSet<u64> = HashSet::new();
        let mut soft_limit_warned = false;
        // Force-stop flag: when true, strip tools from next LLM call to force
        // a text response. Activated after too many tool calls without settling.
        let mut force_text_response = false;
        let mut budget_warning_sent = false;
        let mut effective_task_budget = self.task_token_budget;
        let mut effective_daily_budget = self.daily_token_budget;
        // Runtime-only override for goal daily budget extensions.
        // We intentionally do NOT persist auto-extensions to DB to avoid ratcheting.
        let mut effective_goal_daily_budget: Option<i64> = None;
        let mut budget_extensions_count: usize = 0;
        const MAX_BUDGET_EXTENSIONS: usize = 3;
        const HARD_TOKEN_CAP: i64 = 2_000_000;
        let mut pending_system_messages: Vec<String> = Vec::new();
        if route_failsafe_active {
            consultant_pass_active = false;
            pending_system_messages.push(
                "[SYSTEM] Route fail-safe is active for this session. Use explicit tools/results, avoid direct-return shortcuts, and prioritize concrete execution evidence."
                    .to_string(),
            );
        }
        // Track recent tool names for alternating pattern detection (A-B-A-B cycles)
        let mut recent_tool_names: VecDeque<String> = VecDeque::new();
        // Mid-loop adaptation and fallback expansion controls.
        let mut last_escalation_iteration: Option<usize> = None;
        let mut consecutive_clean_iterations: usize = 0;
        let mut fallback_expanded_once = false;
        // One-shot recovery for empty execution responses (no text + no tool calls).
        let mut empty_response_retry_used = false;
        let mut empty_response_retry_pending = false;
        let mut empty_response_retry_note: Option<String> = None;
        // Idempotency guard for send_file within a single task execution.
        let mut successful_send_file_keys: HashSet<String> = HashSet::new();
        // Inject cli_agent completion nudges at most once per phase
        // (consecutive cli_agent completions), then reset after a
        // successful non-cli_agent tool call.
        let mut cli_agent_boundary_injected = false;
        // Deterministic top-level acknowledgement when a tool detaches to background.
        let mut pending_background_ack: Option<String> = None;
        // Track identity-attack prefill so we can prepend it to the final reply.
        let mut identity_prefill_text: Option<String> = None;
        // Best-effort project directory hint (seeded from user text, refined by tool calls).
        let mut known_project_dir = turn_context
            .primary_project_scope
            .clone()
            .or_else(|| super::tool_execution_phase::extract_project_dir_hint(user_text));
        // Cross-iteration directory evidence tracking for contradiction detection.
        let mut dirs_with_project_inspect_file_evidence: HashSet<String> = HashSet::new();
        let mut dirs_with_search_no_matches: HashSet<String> = HashSet::new();
        // When true, the assistant must run at least one file re-check before finalizing text.
        let mut require_file_recheck_before_answer = false;
        // Route fail-safe bypasses consultant pass; seed tools-required state so
        // text-only completions cannot bypass execution in this mode.
        let mut needs_tools_for_turn = route_failsafe_active;

        // Determine iteration limit behavior
        let (hard_cap, soft_threshold, soft_warn_at) = match &self.iteration_config {
            IterationLimitConfig::Unlimited => (Some(HARD_ITERATION_CAP), None, None),
            IterationLimitConfig::Soft { threshold, warn_at } => {
                (Some(HARD_ITERATION_CAP), Some(*threshold), Some(*warn_at))
            }
            IterationLimitConfig::Hard { initial: _, cap } => (Some(*cap), None, None),
        };

        // Resolve goal_id once for per-goal token budget enforcement.
        // Executors currently carry only task_id, so we may need to lookup goal_id via task.
        let resolved_goal_id: Option<String> = if let Some(gid) = self.goal_id.clone() {
            Some(gid)
        } else if let Some(ref tid) = self.task_id {
            match self.state.get_task(tid).await {
                Ok(Some(task)) => Some(task.goal_id),
                Ok(None) => {
                    warn!(
                        session_id,
                        task_id = %tid,
                        "Task not found while resolving goal_id; goal budget enforcement disabled for this run"
                    );
                    None
                }
                Err(e) => {
                    warn!(
                        session_id,
                        task_id = %tid,
                        error = %e,
                        "Failed to resolve goal_id from task; goal budget enforcement disabled for this run"
                    );
                    None
                }
            }
        } else {
            None
        };

        loop {
            iteration += 1;
            touch_heartbeat(&heartbeat);

            // Check for cancellation (cascades via token hierarchy)
            if let Some(ref ct) = self.cancel_token {
                if ct.is_cancelled() {
                    info!(session_id, iteration, "Task cancelled by parent");
                    self.emit_decision_point(
                        &emitter,
                        &task_id,
                        iteration,
                        DecisionType::StoppingCondition,
                        "Stopping condition fired: cancellation token set".to_string(),
                        json!({"condition":"cancelled"}),
                    )
                    .await;

                    // Mark remaining tasks as cancelled.
                    if let Some(ref gid) = self.goal_id {
                        if let Ok(tasks) = self.state.get_tasks_for_goal(gid).await {
                            for task in &tasks {
                                if task.status != "completed"
                                    && task.status != "failed"
                                    && task.status != "cancelled"
                                {
                                    let mut ct = task.clone();
                                    ct.status = "cancelled".to_string();
                                    let _ = self.state.update_task(&ct).await;
                                }
                            }
                        }
                    }

                    let cancel_reply = "Task cancelled.".to_string();
                    let assistant_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "assistant".to_string(),
                        content: Some(cancel_reply.clone()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.3,
                        embedding: None,
                    };
                    let _ = self
                        .append_assistant_message_with_event(
                            &emitter,
                            &assistant_msg,
                            "system",
                            None,
                            None,
                        )
                        .await;

                    self.emit_task_end(
                        &emitter,
                        &task_id,
                        TaskStatus::Cancelled,
                        task_start,
                        iteration,
                        0,
                        None,
                        Some(cancel_reply.clone()),
                    )
                    .await;
                    return Ok(cancel_reply);
                }
            }

            info!(
                iteration,
                session_id,
                model = %model,
                depth = self.depth,
                policy_profile = ?policy_bundle.policy.model_profile,
                verify_level = ?policy_bundle.policy.verify_level,
                approval_mode = ?policy_bundle.policy.approval_mode,
                context_budget = policy_bundle.policy.context_budget,
                tool_budget = policy_bundle.policy.tool_budget,
                policy_rev = policy_bundle.policy.policy_rev,
                risk_score = policy_bundle.risk_score,
                uncertainty_score = policy_bundle.uncertainty_score,
                "Agent loop iteration"
            );

            // Emit ThinkingStart event
            let _ = emitter
                .emit(
                    EventType::ThinkingStart,
                    ThinkingStartData {
                        iteration: iteration as u32,
                        task_id: task_id.clone(),
                        total_tool_calls: learning_ctx.tool_calls.len() as u32,
                    },
                )
                .await;

            let stopping_outcome = self
                .run_stopping_phase(&mut StoppingPhaseCtx {
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx: &mut learning_ctx,
                    hard_cap,
                    task_tokens_used,
                    effective_task_budget: &mut effective_task_budget,
                    budget_warning_sent: &mut budget_warning_sent,
                    pending_system_messages: &mut pending_system_messages,
                    budget_extensions_count: &mut budget_extensions_count,
                    user_role,
                    evidence_gain_count,
                    stall_count,
                    deferred_no_tool_streak,
                    consecutive_same_tool: &consecutive_same_tool,
                    consecutive_same_tool_arg_hashes: &consecutive_same_tool_arg_hashes,
                    total_successful_tool_calls,
                    pending_background_ack: &mut pending_background_ack,
                    status_tx: &status_tx,
                    resolved_goal_id: &resolved_goal_id,
                    effective_daily_budget: &mut effective_daily_budget,
                    effective_goal_daily_budget: &mut effective_goal_daily_budget,
                    successful_send_file_keys: &successful_send_file_keys,
                    model: &mut model,
                    soft_threshold,
                    soft_warn_at,
                    soft_limit_warned: &mut soft_limit_warned,
                    last_progress_summary: &mut last_progress_summary,
                    tool_failure_count: &tool_failure_count,
                    session_summary: &mut session_summary,
                    policy_bundle: &mut policy_bundle,
                    user_text,
                    available_capabilities: &available_capabilities,
                    llm_router: &llm_router,
                    last_escalation_iteration: &mut last_escalation_iteration,
                    consecutive_clean_iterations: &mut consecutive_clean_iterations,
                    max_budget_extensions: MAX_BUDGET_EXTENSIONS,
                    hard_token_cap: HARD_TOKEN_CAP,
                })
                .await?;
            match stopping_outcome {
                StoppingPhaseOutcome::ContinueLoop => continue,
                StoppingPhaseOutcome::Return(result) => return result,
                StoppingPhaseOutcome::Proceed => {}
            }
            let MessageBuildData { mut messages } = self
                .run_message_build_phase(&mut MessageBuildCtx {
                    session_id,
                    iteration,
                    user_text,
                    model: &model,
                    system_prompt: &system_prompt,
                    consultant_pass_active,
                    pinned_memories: &pinned_memories,
                    tool_defs: &tool_defs,
                    policy_bundle: &policy_bundle,
                    session_summary: &session_summary,
                    pending_system_messages: &mut pending_system_messages,
                    empty_response_retry_pending,
                    status_tx: &status_tx,
                })
                .await?;

            let mut resp = match self
                .run_llm_phase(&mut LlmPhaseCtx {
                    messages: &mut messages,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    user_text,
                    iteration,
                    consultant_pass_active,
                    force_text_response,
                    task_start,
                    task_tokens_used: &mut task_tokens_used,
                    learning_ctx: &mut learning_ctx,
                    pending_system_messages: &mut pending_system_messages,
                    llm_provider: llm_provider.clone(),
                    llm_router: llm_router.clone(),
                    model: &model,
                    user_role,
                    tool_defs: &tool_defs,
                    status_tx: &status_tx,
                    resolved_goal_id: &resolved_goal_id,
                    effective_goal_daily_budget: &mut effective_goal_daily_budget,
                    budget_extensions_count: &mut budget_extensions_count,
                    evidence_gain_count,
                    stall_count: &mut stall_count,
                    consecutive_same_tool: &consecutive_same_tool,
                    consecutive_same_tool_arg_hashes: &consecutive_same_tool_arg_hashes,
                    total_successful_tool_calls,
                    heartbeat: &heartbeat,
                    empty_response_retry_pending: &mut empty_response_retry_pending,
                    empty_response_retry_note: &mut empty_response_retry_note,
                    identity_prefill_text: &mut identity_prefill_text,
                    deferred_no_tool_streak,
                    max_budget_extensions: MAX_BUDGET_EXTENSIONS,
                    hard_token_cap: HARD_TOKEN_CAP,
                })
                .await?
            {
                LlmPhaseOutcome::ContinueLoop => continue,
                LlmPhaseOutcome::Return(result) => return result,
                LlmPhaseOutcome::Proceed(resp) => resp,
            };

            let consultant_outcome = self
                .run_consultant_phase(&mut ConsultantPhaseCtx {
                    resp: &mut resp,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    user_text,
                    iteration,
                    consultant_pass_active,
                    task_start,
                    task_tokens_used,
                    learning_ctx: &mut learning_ctx,
                    pending_system_messages: &mut pending_system_messages,
                    tool_defs: &mut tool_defs,
                    base_tool_defs: &mut base_tool_defs,
                    available_capabilities: &mut available_capabilities,
                    policy_bundle: &mut policy_bundle,
                    tools_allowed_for_user,
                    restrict_to_personal_memory_tools,
                    is_personal_memory_recall_turn,
                    is_reaffirmation_challenge_turn,
                    requests_external_verification,
                    llm_provider: llm_provider.clone(),
                    llm_router: llm_router.clone(),
                    model: &mut model,
                    user_role,
                    channel_ctx: channel_ctx.clone(),
                    status_tx: status_tx.clone(),
                    total_successful_tool_calls,
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
                    require_file_recheck_before_answer: &mut require_file_recheck_before_answer,
                    turn_context: &turn_context,
                    needs_tools_for_turn: &mut needs_tools_for_turn,
                })
                .await?;
            match consultant_outcome {
                ConsultantPhaseOutcome::ContinueLoop => continue,
                ConsultantPhaseOutcome::Return(result) => return result,
                ConsultantPhaseOutcome::ProceedToToolExecution => {}
            }
            // === EXECUTE TOOL CALLS ===
            let tool_prelude_outcome = self
                .run_tool_prelude_phase(&ToolPreludeCtx {
                    resp: &resp,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    model: &model,
                    iteration,
                    task_start,
                    learning_ctx: &learning_ctx,
                    user_text,
                    policy_bundle: &policy_bundle,
                    available_capabilities: &available_capabilities,
                })
                .await?;
            match tool_prelude_outcome {
                ToolPreludeOutcome::ContinueLoop => continue,
                ToolPreludeOutcome::Return(result) => return result,
                ToolPreludeOutcome::Proceed => {}
            }

            let tool_execution_outcome = self
                .run_tool_execution_phase(&mut ToolExecutionCtx {
                    resp: &resp,
                    emitter: &emitter,
                    task_id: &task_id,
                    session_id,
                    iteration,
                    task_start,
                    learning_ctx: &mut learning_ctx,
                    task_tokens_used,
                    user_text,
                    restrict_to_personal_memory_tools,
                    is_reaffirmation_challenge_turn,
                    personal_memory_tool_call_cap,
                    base_tool_defs: &base_tool_defs,
                    available_capabilities: &available_capabilities,
                    policy_bundle: &policy_bundle,
                    status_tx: status_tx.clone(),
                    channel_ctx: &channel_ctx,
                    user_role,
                    heartbeat: &heartbeat,
                    tool_defs: &mut tool_defs,
                    total_tool_calls_attempted: &mut total_tool_calls_attempted,
                    total_successful_tool_calls: &mut total_successful_tool_calls,
                    tool_failure_count: &mut tool_failure_count,
                    tool_call_count: &mut tool_call_count,
                    personal_memory_tool_calls: &mut personal_memory_tool_calls,
                    no_evidence_result_streak: &mut no_evidence_result_streak,
                    no_evidence_tools_seen: &mut no_evidence_tools_seen,
                    evidence_gain_count: &mut evidence_gain_count,
                    pending_error_solution_ids: &mut pending_error_solution_ids,
                    tool_failure_patterns: &mut tool_failure_patterns,
                    last_tool_failure: &mut last_tool_failure,
                    in_session_learned: &mut in_session_learned,
                    unknown_tools: &mut unknown_tools,
                    recent_tool_calls: &mut recent_tool_calls,
                    consecutive_same_tool: &mut consecutive_same_tool,
                    consecutive_same_tool_arg_hashes: &mut consecutive_same_tool_arg_hashes,
                    force_text_response: &mut force_text_response,
                    pending_system_messages: &mut pending_system_messages,
                    recent_tool_names: &mut recent_tool_names,
                    successful_send_file_keys: &mut successful_send_file_keys,
                    cli_agent_boundary_injected: &mut cli_agent_boundary_injected,
                    pending_background_ack: &mut pending_background_ack,
                    stall_count: &mut stall_count,
                    deferred_no_tool_streak: &mut deferred_no_tool_streak,
                    consecutive_clean_iterations: &mut consecutive_clean_iterations,
                    fallback_expanded_once: &mut fallback_expanded_once,
                    known_project_dir: &mut known_project_dir,
                    dirs_with_project_inspect_file_evidence:
                        &mut dirs_with_project_inspect_file_evidence,
                    dirs_with_search_no_matches: &mut dirs_with_search_no_matches,
                    require_file_recheck_before_answer: &mut require_file_recheck_before_answer,
                    turn_context: &turn_context,
                    resolved_goal_id: resolved_goal_id.as_deref(),
                })
                .await?;
            match tool_execution_outcome {
                ToolExecutionOutcome::Return(result) => return result,
                ToolExecutionOutcome::NextIteration => {}
            }
        }
    }
}

#[cfg(test)]
#[path = "characterization_tests.rs"]
mod characterization_tests;
