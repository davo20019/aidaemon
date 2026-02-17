use super::stopping_conditions::{PureStoppingInputs, StoppingCondition};
use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ConversationSummary;

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
    pub task_tokens_used: u64,
    pub effective_task_budget: &'a mut Option<u64>,
    pub budget_warning_sent: &'a mut bool,
    pub pending_system_messages: &'a mut Vec<String>,
    pub budget_extensions_count: &'a mut usize,
    pub user_role: UserRole,
    pub evidence_gain_count: usize,
    pub stall_count: usize,
    pub deferred_no_tool_streak: usize,
    pub consecutive_same_tool: &'a (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a HashSet<u64>,
    pub total_successful_tool_calls: usize,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub resolved_goal_id: &'a Option<String>,
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
}

impl Agent {
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
        let status_tx = ctx.status_tx;
        let resolved_goal_id = ctx.resolved_goal_id;
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

        macro_rules! commit_state {
            () => {
                *ctx.effective_task_budget = effective_task_budget;
                *ctx.budget_warning_sent = budget_warning_sent;
                *ctx.budget_extensions_count = budget_extensions_count;
                *ctx.effective_goal_daily_budget = effective_goal_daily_budget;
                *ctx.model = model.clone();
                *ctx.soft_limit_warned = soft_limit_warned;
                *ctx.last_progress_summary = last_progress_summary;
                *ctx.session_summary = session_summary.clone();
                *ctx.policy_bundle = policy_bundle.clone();
                *ctx.last_escalation_iteration = last_escalation_iteration;
                *ctx.consecutive_clean_iterations = consecutive_clean_iterations;
            };
        }

        // === STOPPING CONDITIONS ===

        // 1-2. Pure stopping conditions (hard cap + timeout) in precedence order.
        let elapsed_secs = task_start.elapsed().as_secs();
        if let Some(condition) = (PureStoppingInputs {
            iteration,
            hard_cap,
            timeout_secs: self.task_timeout.map(|timeout| timeout.as_secs()),
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
                    self.emit_decision_point(
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
                    self.emit_decision_point(
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
        if let Some(budget) = effective_task_budget {
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
                pending_system_messages.push(format!(
                    "[SYSTEM] TOKEN BUDGET WARNING: You have used {} of {} tokens ({}%). \
                         You are approaching the task token limit. Wrap up your work and \
                         respond to the user immediately.",
                    task_tokens_used, budget, pct
                ));
            }

            if budget > 0 && task_tokens_used >= budget {
                // Try auto-extending if productive
                let cap_u64 = hard_token_cap as u64;
                let new_budget_u64 = budget
                    .saturating_mul(2)
                    .max(task_tokens_used.saturating_add(budget / 2))
                    .min(cap_u64);
                if budget_extensions_count < max_budget_extensions
                    && budget < cap_u64
                    && new_budget_u64 > task_tokens_used
                    && user_role == UserRole::Owner
                    && evidence_gain_count >= 2
                    && post_task::is_productive(
                        learning_ctx,
                        stall_count,
                        consecutive_same_tool.1,
                        consecutive_same_tool_arg_hashes.len(),
                        total_successful_tool_calls,
                    )
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
                    pending_system_messages.push(format!(
                        "[SYSTEM] Token budget auto-extended from {} to {} ({}/{} extensions). \
                             Continue working.",
                        old_budget_i64,
                        new_budget_i64,
                        budget_extensions_count,
                        max_budget_extensions
                    ));
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

                warn!(
                    session_id,
                    tokens_used = task_tokens_used,
                    budget,
                    "Task token budget exhausted"
                );
                self.emit_decision_point(
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

        // 4. Goal daily token budget (per-goal limit)
        if let Some(ref goal_id) = resolved_goal_id {
            match self
                .state
                .add_goal_tokens_and_get_budget_status(goal_id, 0)
                .await
            {
                Ok(Some(status)) => {
                    if let Some(db_budget_daily) = status.budget_daily {
                        let budget_daily = effective_goal_daily_budget.unwrap_or(db_budget_daily);
                        if budget_daily > 0 && status.tokens_used_today >= budget_daily {
                            // Try auto-extending goal daily budget if productive
                            let old_gbudget = budget_daily;
                            let new_gbudget = old_gbudget
                                .saturating_mul(2)
                                .max(status.tokens_used_today.saturating_add(old_gbudget / 2))
                                .min(hard_token_cap);
                            if budget_extensions_count < max_budget_extensions
                                && old_gbudget < hard_token_cap
                                && new_gbudget > status.tokens_used_today
                                && evidence_gain_count >= 2
                                && post_task::is_productive(
                                    learning_ctx,
                                    stall_count,
                                    consecutive_same_tool.1,
                                    consecutive_same_tool_arg_hashes.len(),
                                    total_successful_tool_calls,
                                )
                            {
                                budget_extensions_count += 1;
                                effective_goal_daily_budget = Some(new_gbudget);
                                // NOTE: Do NOT persist the extended budget to DB.
                                // Persisting causes permanent budget ratcheting — once
                                // doubled, the inflated budget becomes the baseline for
                                // all future runs, eventually reaching the 2M hard cap.
                                // The extension is in-memory only for this run.
                                info!(
                                    session_id,
                                    goal_id = %goal_id,
                                    old_budget = old_gbudget,
                                    new_budget = new_gbudget,
                                    extension = budget_extensions_count,
                                    "Auto-extended goal daily token budget in-memory (pre-check)"
                                );
                                pending_system_messages.push(format!(
                                        "[SYSTEM] Goal daily token budget auto-extended from {} to {} ({}/{} extensions). \
                                         Continue working.",
                                        old_gbudget, new_gbudget, budget_extensions_count, max_budget_extensions
                                    ));
                                send_status(
                                    status_tx,
                                    StatusUpdate::BudgetExtended {
                                        old_budget: old_gbudget,
                                        new_budget: new_gbudget,
                                        extension: budget_extensions_count,
                                        max_extensions: max_budget_extensions,
                                    },
                                );
                                self.emit_decision_point(
                                    emitter,
                                    task_id,
                                    iteration,
                                    DecisionType::BudgetAutoExtension,
                                    "Auto-extended goal daily token budget on productive progress"
                                        .to_string(),
                                    json!({
                                        "condition": "goal_daily_budget_extension",
                                        "goal_id": goal_id,
                                        "old_budget": old_gbudget,
                                        "new_budget": new_gbudget,
                                        "extension": budget_extensions_count,
                                        "max_extensions": max_budget_extensions,
                                    }),
                                )
                                .await;
                                commit_state!();
                                return Ok(StoppingPhaseOutcome::ContinueLoop);
                            }

                            warn!(
                                session_id,
                                iteration,
                                goal_id = %goal_id,
                                tokens_used_today = status.tokens_used_today,
                                budget_daily,
                                "Goal daily token budget exhausted"
                            );
                            self.emit_decision_point(
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
                                    "tokens_used_today": status.tokens_used_today
                                }),
                            )
                            .await;
                            let alert_msg = format!(
                                    "Token alert: goal '{}' hit daily token budget (used {} / limit {}). Execution was stopped to prevent overspending.",
                                    goal_id, status.tokens_used_today, budget_daily
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
                                    status.tokens_used_today,
                                    budget_daily,
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

        // 5. Daily token budget (existing global limit)
        if let Some(daily_budget) = self.daily_token_budget {
            let today_start = Utc::now().format("%Y-%m-%d 00:00:00").to_string();
            if let Ok(records) = self.state.get_token_usage_since(&today_start).await {
                let total: u64 = records
                    .iter()
                    .map(|r| (r.input_tokens + r.output_tokens) as u64)
                    .sum();
                if total >= daily_budget {
                    self.emit_decision_point(
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
        if total_successful_tool_calls == 0 && deferred_no_tool_streak >= MAX_PRE_TOOL_DEFERRALS {
            warn!(
                session_id,
                deferred_no_tool_streak, "Pre-tool deferral threshold reached"
            );
            self.emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::StoppingCondition,
                "Stopping condition fired: repeated pre-tool deferrals".to_string(),
                json!({
                    "condition":"pre_tool_deferral_stall",
                    "deferred_no_tool_streak": deferred_no_tool_streak,
                    "max_pre_tool_deferrals": MAX_PRE_TOOL_DEFERRALS
                }),
            )
            .await;
            let reply = "I’m still getting planning-only outputs and haven’t started tool execution yet. Please resend the request and I’ll retry with stricter tool-call mode."
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
                embedding: None,
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
        let stall_detected = matches!(
            PureStoppingInputs {
                iteration,
                hard_cap: None,
                timeout_secs: None,
                elapsed_secs: 0,
                task_token_budget: None,
                task_tokens_used: 0,
                stall_count,
                max_stall_iterations: MAX_STALL_ITERATIONS,
            }
            .evaluate(),
            Some(StoppingCondition::Stall { .. })
        );
        if stall_detected {
            if !successful_send_file_keys.is_empty() && learning_ctx.errors.is_empty() {
                let reply = "I already sent the requested file. If you want any changes or another file, tell me exactly what to send.".to_string();
                self.emit_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Stopping condition fired after successful send_file; resolving as completed"
                        .to_string(),
                    json!({
                        "condition":"post_send_file_stall",
                        "stall_count": stall_count,
                        "max_stall_iterations": MAX_STALL_ITERATIONS,
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
                    embedding: None,
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
                    stall_count,
                    total_successful_tool_calls,
                    unrecovered_errors,
                    "Agent stalled after meaningful progress"
                );
                self.emit_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::StoppingCondition,
                    "Stopping condition fired: stall after meaningful progress".to_string(),
                    json!({
                        "condition":"stall_with_progress",
                        "stall_count": stall_count,
                        "max_stall_iterations": MAX_STALL_ITERATIONS,
                        "total_successful_tool_calls": total_successful_tool_calls,
                        "unrecovered_errors": unrecovered_errors
                    }),
                )
                .await;
                let result = self
                    .graceful_partial_stall_response(
                        emitter,
                        session_id,
                        learning_ctx,
                        !successful_send_file_keys.is_empty(),
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

            warn!(
                session_id,
                stall_count, "Agent stalled - no progress detected"
            );
            self.emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::StoppingCondition,
                "Stopping condition fired: stall threshold reached".to_string(),
                json!({
                    "condition":"stall",
                    "stall_count": stall_count,
                    "max_stall_iterations": MAX_STALL_ITERATIONS
                }),
            )
            .await;
            let result = self
                .graceful_stall_response(
                    emitter,
                    session_id,
                    learning_ctx,
                    !successful_send_file_keys.is_empty(),
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
                self.emit_decision_point(
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
                    session_summary = self
                        .state
                        .get_conversation_summary(session_id)
                        .await
                        .ok()
                        .flatten();
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
