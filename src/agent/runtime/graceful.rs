use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) enum GoalBudgetCheckSource {
    PreCheck,
    PostLlm,
}

pub(super) struct GoalBudgetControlCtx<'a> {
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub iteration: usize,
    pub goal_id: &'a str,
    pub status: &'a crate::traits::GoalTokenBudgetStatus,
    pub user_role: UserRole,
    pub learning_ctx: &'a LearningContext,
    pub evidence_gain_count: usize,
    pub stall_count: usize,
    pub consecutive_same_tool_count: usize,
    pub consecutive_same_tool_unique_args: usize,
    pub total_successful_tool_calls: usize,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub is_scheduled_goal: bool,
    pub effective_goal_daily_budget: &'a mut Option<i64>,
    pub budget_extensions_count: &'a mut usize,
    pub max_budget_extensions: usize,
    pub hard_token_cap: i64,
    pub source: GoalBudgetCheckSource,
}

pub(super) enum GoalBudgetControlOutcome {
    Continue,
    Exhausted {
        tokens_used_today: i64,
        budget_daily: i64,
    },
}

struct DecisionPointEmission {
    decision_type: DecisionType,
    severity: crate::events::DiagnosticSeverity,
    summary: String,
    metadata: Value,
}

pub(super) struct ScheduledRunBudgetControlCtx<'a> {
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub iteration: usize,
    pub goal_id: &'a str,
    pub status: &'a crate::goal_tokens::GoalRunBudgetStatus,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub max_budget_extensions: usize,
    pub hard_token_cap: i64,
}

pub(super) enum ScheduledRunBudgetControlOutcome {
    Continue,
    Exhausted {
        tokens_used: i64,
        budget_per_check: i64,
    },
}

pub(super) struct ScheduledRunActivityMetrics {
    pub evidence_gain_count: usize,
    pub stall_count: usize,
    pub consecutive_same_tool_count: usize,
    pub consecutive_same_tool_unique_args: usize,
    pub total_successful_tool_calls: usize,
}

// impl-Agent justification: graceful shutdown, budget progress, and task-end hooks over state/event_store.
impl Agent {
    pub(super) fn has_meaningful_budget_progress(
        evidence_gain_count: usize,
        _total_successful_tool_calls: usize,
    ) -> bool {
        // Transport-level success and administrative calls are not progress.
        // Only a mutation receipt or result-content verification can justify
        // spending the run's single autonomous extension.
        evidence_gain_count > 0
    }

    pub(super) fn scheduled_run_health_snapshot(
        learning_ctx: &LearningContext,
        metrics: ScheduledRunActivityMetrics,
        completion_contract: &CompletionContract,
        completion_progress: &CompletionProgress,
    ) -> crate::traits::ScheduledRunHealth {
        let required_mutation_progress = completion_contract.expects_mutation
            && completion_progress.mutation_count > 0
            && (completion_contract.required_mutation_effects.is_empty()
                || completion_progress
                    .observed_mutation_effects
                    .intersects(completion_contract.required_mutation_effects));
        crate::traits::ScheduledRunHealth {
            evidence_gain_count: metrics.evidence_gain_count,
            total_successful_tool_calls: metrics.total_successful_tool_calls,
            completion_requires_mutation: completion_contract.expects_mutation,
            required_mutation_progress,
            completion_requires_observation: completion_contract.requires_observation,
            verification_progress: completion_progress.verification_count > 0,
            stall_count: metrics.stall_count,
            consecutive_same_tool_count: metrics.consecutive_same_tool_count,
            consecutive_same_tool_unique_args: metrics.consecutive_same_tool_unique_args,
            unrecovered_error_count: learning_ctx
                .errors
                .iter()
                .filter(|(_, recovered)| !recovered)
                .count(),
        }
    }

    pub(super) fn scheduled_run_has_structural_progress(
        health: &crate::traits::ScheduledRunHealth,
    ) -> bool {
        if health.completion_requires_mutation {
            return health.required_mutation_progress;
        }
        if health.completion_requires_observation {
            return health.verification_progress;
        }
        // Text-only scheduled work has no mutation/verification receipt to
        // satisfy, so bounded result evidence remains its progress signal.
        health.evidence_gain_count > 0
    }

    pub(super) fn scheduled_run_metrics_are_clearly_unproductive(
        health: &crate::traits::ScheduledRunHealth,
    ) -> bool {
        if health.stall_count > 1 {
            return true;
        }

        let diverse_limit = MAX_CONSECUTIVE_SAME_TOOL + 4;
        if health.consecutive_same_tool_count >= diverse_limit {
            return true;
        }
        if health.consecutive_same_tool_count >= MAX_CONSECUTIVE_SAME_TOOL {
            let is_diverse =
                health.consecutive_same_tool_unique_args * 2 > health.consecutive_same_tool_count;
            if !is_diverse {
                return true;
            }
        }

        if health.total_successful_tool_calls == 0 {
            return health.unrecovered_error_count > 0 && health.evidence_gain_count == 0;
        }

        health.unrecovered_error_count >= health.total_successful_tool_calls
    }

    pub(super) fn scheduled_run_auto_extension_candidate(
        status: &crate::goal_tokens::GoalRunBudgetStatus,
        max_budget_extensions: usize,
        hard_token_cap: i64,
    ) -> Option<i64> {
        let old_budget = status.effective_budget_per_check;
        let new_budget = old_budget
            .saturating_mul(2)
            .max(status.tokens_used.saturating_add(old_budget / 2))
            .min(hard_token_cap);

        let has_meaningful_progress = Self::scheduled_run_has_structural_progress(&status.health);
        let clearly_unproductive =
            Self::scheduled_run_metrics_are_clearly_unproductive(&status.health);

        if status.budget_extensions_count < max_budget_extensions
            && old_budget < hard_token_cap
            && new_budget > status.tokens_used
            && has_meaningful_progress
            && !clearly_unproductive
        {
            Some(new_budget)
        } else {
            None
        }
    }

    pub(super) fn scheduled_run_budget_pressure_pct(
        status: &crate::goal_tokens::GoalRunBudgetStatus,
        warning_already_sent: bool,
    ) -> Option<i64> {
        let budget = status.effective_budget_per_check;
        if warning_already_sent || budget <= 0 || status.tokens_used >= budget {
            return None;
        }
        let warning_threshold = budget.saturating_mul(80) / 100;
        (status.tokens_used >= warning_threshold)
            .then(|| status.tokens_used.saturating_mul(100) / budget)
    }

    pub(super) async fn run_task_end_tool_hooks(&self, task_id: &str, session_id: &str) {
        for tool in &self.tools {
            if let Err(e) = tool.on_task_end(task_id, session_id).await {
                warn!(
                    task_id,
                    session_id,
                    tool = tool.name(),
                    error = %e,
                    "Task-end cleanup hook failed"
                );
            }
        }
        if let Some(manager) = crate::checkpoints::active_manager() {
            if let Err(error) = manager.finalize_task(task_id, session_id).await {
                warn!(
                    task_id,
                    session_id,
                    error = %error,
                    "Failed to finalize filesystem checkpoint"
                );
            }
        }
    }

    /// Ask the owner to approve a one-time budget extension for the current run.
    ///
    /// Returns true only when the owner explicitly approves.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn request_budget_continue_approval(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        session_id: &str,
        user_role: UserRole,
        scope_label: &str,
        used_tokens: i64,
        current_budget: i64,
        proposed_budget: i64,
    ) -> bool {
        if user_role != UserRole::Owner {
            return false;
        }
        if proposed_budget <= current_budget {
            return false;
        }

        let hub_weak = match tokio::time::timeout(Duration::from_secs(2), self.hub.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!(
                    session_id,
                    scope = scope_label,
                    "Timed out acquiring hub lock for budget extension approval"
                );
                return false;
            }
        };
        let Some(hub_weak) = hub_weak else {
            return false;
        };
        let Some(hub_arc) = hub_weak.upgrade() else {
            return false;
        };

        let approval_request = build_needs_approval_request(
            format!(
                "extend the {} token budget from {} to {} and continue execution",
                scope_label, current_budget, proposed_budget
            ),
            Some(format!("{} token budget", scope_label)),
            format!(
                "Current usage is {} tokens, which exhausted the {} budget.",
                used_tokens, scope_label
            ),
            "Explicit owner approval is required before spending more tokens on this run.",
            format!(
                "If approved, I will continue the current work inside the extended {} budget.",
                scope_label
            ),
            None,
        );
        let (approval_desc, warnings) = approval_request.to_inline_approval_prompt();
        self.emit_decision_point(
            emitter,
            task_id,
            iteration,
            DecisionType::BudgetAutoExtension,
            format!(
                "Requested owner approval for {} budget extension",
                scope_label
            ),
            json!({
                "condition": "budget_extension_manual_request",
                "scope_label": scope_label,
                "approval_state": ApprovalState::Requested,
                "used_tokens": used_tokens,
                "current_budget": current_budget,
                "proposed_budget": proposed_budget,
            }),
        )
        .await;

        match hub_arc
            .request_inline_approval(
                session_id,
                &approval_desc,
                RiskLevel::High,
                &warnings,
                PermissionMode::Cautious,
            )
            .await
        {
            Ok(ApprovalResponse::AllowOnce)
            | Ok(ApprovalResponse::AllowSession)
            | Ok(ApprovalResponse::AllowAlways) => true,
            Ok(ApprovalResponse::Deny) => {
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::BudgetAutoExtension,
                    format!("Owner denied {} budget extension", scope_label),
                    json!({
                        "condition": "budget_extension_manual_denied",
                        "scope_label": scope_label,
                        "approval_state": ApprovalState::Denied,
                        "used_tokens": used_tokens,
                        "current_budget": current_budget,
                        "proposed_budget": proposed_budget,
                    }),
                )
                .await;
                false
            }
            Err(e) => {
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::BudgetAutoExtension,
                    format!("Approval unavailable for {} budget extension", scope_label),
                    json!({
                        "condition": "budget_extension_manual_unavailable",
                        "scope_label": scope_label,
                        "approval_state": ApprovalState::Denied,
                        "used_tokens": used_tokens,
                        "current_budget": current_budget,
                        "proposed_budget": proposed_budget,
                        "error": e.to_string(),
                    }),
                )
                .await;
                warn!(
                    session_id,
                    scope = scope_label,
                    error = %e,
                    "Budget extension approval unavailable"
                );
                false
            }
        }
    }

    pub(super) async fn enforce_goal_daily_budget_control(
        &self,
        ctx: &mut GoalBudgetControlCtx<'_>,
    ) -> GoalBudgetControlOutcome {
        let Some(db_budget_daily) = ctx.status.budget_daily else {
            return GoalBudgetControlOutcome::Continue;
        };
        let shared_budget_daily = if let Some(registry) = &self.goal_token_registry {
            registry.get_effective_daily_budget(ctx.goal_id).await
        } else {
            None
        };
        let durable_override = crate::goal_tokens::load_goal_daily_budget_override(
            self.state.as_ref(),
            ctx.goal_id,
            db_budget_daily,
            ctx.hard_token_cap,
        )
        .await;
        if let Some(durable) = durable_override.as_ref() {
            *ctx.budget_extensions_count = (*ctx.budget_extensions_count)
                .max(durable.extensions_count.min(ctx.max_budget_extensions));
            if let Some(registry) = &self.goal_token_registry {
                registry
                    .set_effective_daily_budget(ctx.goal_id, durable.budget_daily)
                    .await;
            }
        }
        let budget_daily = [
            Some(db_budget_daily),
            *ctx.effective_goal_daily_budget,
            shared_budget_daily,
            durable_override.map(|value| value.budget_daily),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(db_budget_daily);
        *ctx.effective_goal_daily_budget = Some(budget_daily);
        if budget_daily <= 0 || ctx.status.tokens_used_today < budget_daily {
            return GoalBudgetControlOutcome::Continue;
        }

        let old_gbudget = budget_daily;
        let new_gbudget = old_gbudget
            .saturating_mul(2)
            .max(ctx.status.tokens_used_today.saturating_add(old_gbudget / 2))
            .min(ctx.hard_token_cap);

        let productive = if ctx.is_scheduled_goal {
            // Active scheduled runs are governed by their shared per-run
            // budget above. Keep this fail-closed for any legacy caller that
            // reaches daily control directly.
            false
        } else {
            Self::has_meaningful_budget_progress(
                ctx.evidence_gain_count,
                ctx.total_successful_tool_calls,
            ) && post_task::is_productive(
                ctx.learning_ctx,
                ctx.stall_count,
                ctx.consecutive_same_tool_count,
                ctx.consecutive_same_tool_unique_args,
                ctx.total_successful_tool_calls,
            )
        };

        let (auto_condition, manual_condition, source_label) = match ctx.source {
            GoalBudgetCheckSource::PreCheck => (
                "goal_daily_budget_extension",
                "goal_daily_budget_extension_manual",
                "pre-check",
            ),
            GoalBudgetCheckSource::PostLlm => (
                "goal_daily_budget_extension_post_llm",
                "goal_daily_budget_extension_manual_post_llm",
                "post-LLM",
            ),
        };

        if *ctx.budget_extensions_count < ctx.max_budget_extensions
            && old_gbudget < ctx.hard_token_cap
            && new_gbudget > ctx.status.tokens_used_today
            && productive
        {
            *ctx.budget_extensions_count += 1;
            *ctx.effective_goal_daily_budget = Some(new_gbudget);
            if let Some(registry) = &self.goal_token_registry {
                registry
                    .set_effective_daily_budget(ctx.goal_id, new_gbudget)
                    .await;
            }
            if let Err(error) = crate::goal_tokens::persist_goal_daily_budget_override(
                self.state.as_ref(),
                ctx.goal_id,
                new_gbudget,
                *ctx.budget_extensions_count,
            )
            .await
            {
                warn!(
                    goal_id = %ctx.goal_id,
                    %error,
                    "Failed to persist same-day goal budget extension"
                );
            }
            info!(
                ctx.session_id,
                goal_id = %ctx.goal_id,
                old_budget = old_gbudget,
                new_budget = new_gbudget,
                extension = *ctx.budget_extensions_count,
                source = source_label,
                "Auto-extended goal daily token budget in-memory"
            );
            ctx.pending_system_messages
                .push(SystemDirective::GoalDailyBudgetAutoExtended {
                    old_budget: old_gbudget,
                    new_budget: new_gbudget,
                    extension: *ctx.budget_extensions_count,
                    max_extensions: ctx.max_budget_extensions,
                });
            if !ctx.is_scheduled_goal {
                send_status(
                    ctx.status_tx,
                    StatusUpdate::BudgetExtended {
                        old_budget: old_gbudget,
                        new_budget: new_gbudget,
                        extension: *ctx.budget_extensions_count,
                        max_extensions: ctx.max_budget_extensions,
                    },
                );
            }
            self.emit_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::BudgetAutoExtension,
                "Auto-extended goal daily token budget on productive progress".to_string(),
                json!({
                    "condition": auto_condition,
                    "goal_id": ctx.goal_id,
                    "old_budget": old_gbudget,
                    "new_budget": new_gbudget,
                    "extension": *ctx.budget_extensions_count,
                    "max_extensions": ctx.max_budget_extensions,
                }),
            )
            .await;
            return GoalBudgetControlOutcome::Continue;
        }

        // Scheduled work has an owner-confirmed unattended authority envelope.
        // It may use the bounded autonomous extension above, but it must never
        // convert resource management into a live approval interruption. Once
        // that envelope is spent, stop this cycle cleanly at the hard boundary.
        if ctx.is_scheduled_goal {
            return GoalBudgetControlOutcome::Exhausted {
                tokens_used_today: ctx.status.tokens_used_today,
                budget_daily,
            };
        }

        let approved_extension =
            if old_gbudget < ctx.hard_token_cap && new_gbudget > ctx.status.tokens_used_today {
                self.request_budget_continue_approval(
                    ctx.emitter,
                    ctx.task_id,
                    ctx.iteration,
                    ctx.session_id,
                    ctx.user_role,
                    "goal daily",
                    ctx.status.tokens_used_today,
                    old_gbudget,
                    new_gbudget,
                )
                .await
            } else {
                false
            };

        if approved_extension {
            *ctx.effective_goal_daily_budget = Some(new_gbudget);
            if let Some(registry) = &self.goal_token_registry {
                registry
                    .set_effective_daily_budget(ctx.goal_id, new_gbudget)
                    .await;
            }
            if let Err(error) = crate::goal_tokens::persist_goal_daily_budget_override(
                self.state.as_ref(),
                ctx.goal_id,
                new_gbudget,
                (*ctx.budget_extensions_count).saturating_add(1),
            )
            .await
            {
                warn!(
                    goal_id = %ctx.goal_id,
                    %error,
                    "Failed to persist owner-approved same-day goal budget extension"
                );
            }
            ctx.pending_system_messages
                .push(SystemDirective::GoalDailyBudgetExtensionApproved {
                    old_budget: old_gbudget,
                    new_budget: new_gbudget,
                });
            self.emit_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::BudgetAutoExtension,
                "Extended goal daily token budget via owner approval".to_string(),
                json!({
                    "condition": manual_condition,
                    "goal_id": ctx.goal_id,
                    "approval_state": ApprovalState::Granted,
                    "old_budget": old_gbudget,
                    "new_budget": new_gbudget,
                    "tokens_used_today": ctx.status.tokens_used_today,
                }),
            )
            .await;
            return GoalBudgetControlOutcome::Continue;
        }

        GoalBudgetControlOutcome::Exhausted {
            tokens_used_today: ctx.status.tokens_used_today,
            budget_daily,
        }
    }

    pub(super) async fn enforce_scheduled_run_budget_control(
        &self,
        ctx: &mut ScheduledRunBudgetControlCtx<'_>,
    ) -> ScheduledRunBudgetControlOutcome {
        let budget_per_check = ctx.status.effective_budget_per_check;
        if budget_per_check <= 0 || ctx.status.tokens_used < budget_per_check {
            return ScheduledRunBudgetControlOutcome::Continue;
        }

        let old_budget = budget_per_check;
        if let Some(new_budget) = Self::scheduled_run_auto_extension_candidate(
            ctx.status,
            ctx.max_budget_extensions,
            ctx.hard_token_cap,
        ) {
            if let Some(registry) = &self.goal_token_registry {
                let updated = registry
                    .auto_extend_run_budget(ctx.goal_id, new_budget)
                    .await;
                if let Some(status) = updated.as_ref() {
                    persist_scheduled_run_state(&self.state, ctx.goal_id, None, status).await;
                }
                let extension = updated
                    .as_ref()
                    .map(|status| status.budget_extensions_count)
                    .unwrap_or_else(|| ctx.status.budget_extensions_count.saturating_add(1));
                info!(
                    ctx.session_id,
                    goal_id = %ctx.goal_id,
                    old_budget,
                    new_budget,
                    extension,
                    "Auto-extended scheduled run budget"
                );
                // This is routine internal adaptation, not an owner-attention
                // event. Keep it in the directive/event ledger instead of the
                // channel status stream so an unattended run cannot wake the
                // owner merely because it resized its own budget.
                ctx.pending_system_messages.push(
                    SystemDirective::ScheduledRunBudgetAdaptationRequired {
                        old_budget,
                        new_budget,
                        extension,
                        max_extensions: ctx.max_budget_extensions,
                    },
                );
                self.emit_decision_point(
                    ctx.emitter,
                    ctx.task_id,
                    ctx.iteration,
                    DecisionType::BudgetAutoExtension,
                    "Auto-extended scheduled run budget on continued progress".to_string(),
                    json!({
                        "condition": "scheduled_run_budget_extension",
                        "goal_id": ctx.goal_id,
                        "old_budget": old_budget,
                        "new_budget": new_budget,
                        "extension": extension,
                        "max_extensions": ctx.max_budget_extensions,
                        "tokens_used": ctx.status.tokens_used,
                    }),
                )
                .await;
                return ScheduledRunBudgetControlOutcome::Continue;
            }
        }

        ScheduledRunBudgetControlOutcome::Exhausted {
            tokens_used: ctx.status.tokens_used,
            budget_per_check,
        }
    }

    async fn append_graceful_assistant_summary(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        summary: String,
    ) -> anyhow::Result<String> {
        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(summary.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        };
        self.append_assistant_message_with_event(emitter, &assistant_msg, "system", None, None)
            .await?;
        Ok(summary)
    }

    /// Graceful response when task timeout is reached.
    pub(super) async fn graceful_timeout_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        elapsed: Duration,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_timeout_response(learning_ctx, elapsed);
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    /// Graceful response when task token budget is exhausted.
    pub(super) async fn graceful_budget_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        tokens_used: u64,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_budget_response(learning_ctx, tokens_used);
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    /// Graceful response when a scheduled run hits its per-run budget.
    pub(super) async fn graceful_scheduled_run_budget_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        tokens_used: i64,
        budget_per_check: i64,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_scheduled_run_budget_response(
            learning_ctx,
            tokens_used,
            budget_per_check,
        );
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    /// Graceful response when a goal hits its daily token budget.
    pub(super) async fn graceful_goal_daily_budget_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        tokens_used_today: i64,
        budget_daily: i64,
        is_scheduled_goal: bool,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_goal_daily_budget_response(
            learning_ctx,
            tokens_used_today,
            budget_daily,
            is_scheduled_goal,
        );
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    fn dedupe_alert_sessions(sessions: Vec<String>) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for session in sessions {
            let trimmed = session.trim();
            if trimmed.is_empty() {
                continue;
            }
            if seen.insert(trimmed.to_string()) {
                out.push(trimmed.to_string());
            }
        }
        out
    }

    fn sanitize_alert_scope(scope: &str) -> String {
        scope
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }

    async fn load_default_alert_sessions(&self) -> Vec<String> {
        match self.state.get_setting("default_alert_sessions").await {
            Ok(Some(raw)) => match serde_json::from_str::<Vec<String>>(&raw) {
                Ok(sessions) => Self::dedupe_alert_sessions(sessions),
                Err(e) => {
                    warn!(error = %e, "Invalid default_alert_sessions setting");
                    Vec::new()
                }
            },
            Ok(None) => Vec::new(),
            Err(e) => {
                warn!(error = %e, "Failed to read default_alert_sessions setting");
                Vec::new()
            }
        }
    }

    fn same_alert_destination(left: &str, right: &str) -> bool {
        if left == right {
            return true;
        }
        let is_non_telegram = |value: &str| {
            value.contains("slack:")
                || value.contains("discord:")
                || value.starts_with("specialist:")
        };
        if is_non_telegram(left) || is_non_telegram(right) {
            return false;
        }
        match (
            crate::session::telegram_chat_id_from_session(left),
            crate::session::telegram_chat_id_from_session(right),
        ) {
            (Some(left), Some(right)) => left == right,
            _ => false,
        }
    }

    /// Fan-out token alerts to owner sessions plus the triggering session.
    pub(super) async fn fanout_token_alert(
        &self,
        goal_id: Option<&str>,
        trigger_session_id: &str,
        message: &str,
        suppress_session_id: Option<&str>,
    ) {
        // Background goal runs deliver one consolidated terminal outcome from
        // the task lead. Child and task-lead budget checks can fire at nearly
        // the same instant; fanning out here produced duplicate alerts, often
        // addressed to internal specialist sessions. Defer those alerts to the
        // run finalizer, which has the originating user session and full task
        // outcome.
        if goal_id.is_some() && self.depth > 0 {
            info!(
                depth = self.depth,
                goal_id = goal_id.unwrap_or_default(),
                "Deferring background goal budget alert to terminal run notification"
            );
            return;
        }

        let mut targets = self.load_default_alert_sessions().await;
        let goal_session = if let Some(goal_id) = goal_id {
            self.state
                .get_goal(goal_id)
                .await
                .ok()
                .flatten()
                .map(|goal| goal.session_id)
        } else {
            None
        };
        let primary_session = goal_session.as_deref().unwrap_or(trigger_session_id);
        targets.retain(|target| !Self::same_alert_destination(target, primary_session));
        targets.push(primary_session.to_string());
        targets = Self::dedupe_alert_sessions(targets);

        let goal_ref = goal_id.map(ToString::to_string).unwrap_or_else(|| {
            format!(
                "token-budget:{}",
                Self::sanitize_alert_scope(trigger_session_id)
            )
        });

        let hub = match tokio::time::timeout(Duration::from_secs(2), self.hub.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!(
                    trigger_session_id,
                    "Timed out acquiring hub lock while faning out token alert"
                );
                None
            }
        };
        for target in targets {
            let entry =
                crate::traits::NotificationEntry::new(&goal_ref, &target, "token_alert", message);

            if let Err(e) = self.state.enqueue_notification(&entry).await {
                warn!(
                    session_id = %target,
                    goal_id = %goal_ref,
                    error = %e,
                    "Failed to enqueue token alert"
                );
                continue;
            }

            if suppress_session_id == Some(target.as_str()) {
                let _ = self.state.mark_notification_delivered(&entry.id).await;
                continue;
            }

            if let Some(hub_weak) = &hub {
                if let Some(hub_arc) = hub_weak.upgrade() {
                    if hub_arc.send_text(&target, message).await.is_ok() {
                        let _ = self.state.mark_notification_delivered(&entry.id).await;
                    }
                }
            }
        }
    }

    /// Test-only wrapper around `post_task::classify_stall`.
    ///
    /// Production flow should call the `post_task` function with the real
    /// `tool_failure_count` map so lockout classification remains available.
    #[allow(dead_code)] // Used in tests; production path delegates through post_task.
    pub(super) fn classify_stall(learning_ctx: &LearningContext) -> (&'static str, &'static str) {
        let empty_tool_failure_count: HashMap<String, usize> = HashMap::new();
        post_task::classify_stall(
            learning_ctx,
            DEFERRED_NO_TOOL_ERROR_MARKER,
            &empty_tool_failure_count,
        )
    }

    /// Graceful response when agent is stalled (no progress).
    pub(super) async fn graceful_stall_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        sent_file_successfully: bool,
        tool_failure_count: &HashMap<String, usize>,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_stall_response(
            learning_ctx,
            sent_file_successfully,
            DEFERRED_NO_TOOL_ERROR_MARKER,
            tool_failure_count,
        );
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    /// Graceful response when agent stalled after making meaningful progress.
    pub(super) async fn graceful_partial_stall_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        sent_file_successfully: bool,
        tool_failure_count: &HashMap<String, usize>,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_partial_stall_response(
            learning_ctx,
            sent_file_successfully,
            DEFERRED_NO_TOOL_ERROR_MARKER,
            tool_failure_count,
        );
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    /// Attempt a knowledge-only fallback when tools have failed.
    ///
    /// Makes one LLM call WITHOUT tools, asking the model to answer from
    /// its training knowledge. Returns `Some(answer)` if the model gives a
    /// substantive response (>30 chars), `None` otherwise.
    pub(super) async fn try_knowledge_fallback(
        &self,
        user_text: &str,
        error_summary: &str,
    ) -> Option<String> {
        if self.mandate_execution.is_some() {
            return None;
        }
        let system = format!(
            "The user asked a question but all tool-based approaches failed ({}).\n\
             Answer the question from your training knowledge if possible.\n\
             If you genuinely cannot answer without tools, say so briefly.\n\
             Do NOT mention tool failures or activity summaries.",
            error_summary
        );
        let messages = vec![
            serde_json::json!({"role": "system", "content": system}),
            serde_json::json!({"role": "user", "content": user_text}),
        ];
        let provider = self.llm_runtime.provider();
        let model = match tokio::time::timeout(Duration::from_secs(2), self.model.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring model lock during knowledge fallback");
                return None;
            }
        };
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            provider.chat(&model, &messages, &[]),
        )
        .await
        {
            Ok(Ok(resp)) => {
                let text = resp.content.unwrap_or_default();
                if text.trim().len() > 30 {
                    Some(text.trim().to_string())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Attempt a knowledge-only fallback and, if successful, append the
    /// answer as an assistant message.  Returns `Some(answer)` on success.
    pub(super) async fn graceful_knowledge_fallback(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        user_text: &str,
        error_summary: &str,
    ) -> Option<anyhow::Result<String>> {
        let answer = self
            .try_knowledge_fallback(user_text, error_summary)
            .await?;
        Some(
            self.append_graceful_assistant_summary(emitter, session_id, answer)
                .await,
        )
    }

    /// Graceful response when repetitive tool calls are detected.
    pub(super) async fn graceful_repetitive_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        tool_name: &str,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_repetitive_response(learning_ctx, tool_name);
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    /// Graceful response when hard iteration cap is reached (legacy mode).
    pub(super) async fn graceful_cap_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        iterations: usize,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_cap_response(learning_ctx, iterations);
        self.append_graceful_assistant_summary(emitter, session_id, summary)
            .await
    }

    /// Emit TaskEnd after recording an exact bootstrap protocol direct-return.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn emit_direct_return_task_end(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        status: TaskStatus,
        outcome: crate::events::TaskOutcome,
        task_start: Instant,
        iteration: usize,
        tool_calls_count: usize,
        error: Option<String>,
        summary: Option<String>,
        direct_return_succeeded: bool,
    ) {
        if self.harness_eval_enabled() {
            self.with_harness_eval(|eval| eval.record_direct_return(true, direct_return_succeeded))
                .await;
        }
        self.emit_task_end(
            emitter,
            task_id,
            status,
            outcome,
            task_start,
            iteration,
            tool_calls_count,
            error,
            summary,
        )
        .await;
    }

    /// Emit a TaskEnd event. Called from every exit path in the agent loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn emit_task_end(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        status: TaskStatus,
        outcome: crate::events::TaskOutcome,
        task_start: Instant,
        iteration: usize,
        tool_calls_count: usize,
        error: Option<String>,
        summary: Option<String>,
    ) {
        // The persisted receipt graph is the final authority for an
        // invocation-bound task. Individual exit paths can only propose an
        // outcome from in-memory state; reconcile it here so a provider
        // timeout or finalizer fallback cannot record `succeeded` while a
        // durable required invocation remains unsatisfied.
        let receipt_closure = self
            .event_store
            .task_receipt_closure(emitter.session_id(), task_id)
            .await
            .unwrap_or_default();
        let effective_outcome = if receipt_closure.contract_present {
            if !receipt_closure.fulfilled() && outcome == crate::events::TaskOutcome::Succeeded {
                crate::events::TaskOutcome::Partial
            } else if receipt_closure.fulfilled()
                && outcome == crate::events::TaskOutcome::Partial
                && summary
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty())
            {
                // In-memory verification can lag a durable receipt when a
                // provider/finalizer fails immediately after dispatch. The
                // persisted proof graph is authoritative for obligation
                // closure; promote only a present user-facing closeout, never
                // a hard failure or an empty response.
                crate::events::TaskOutcome::Succeeded
            } else {
                outcome
            }
        } else {
            outcome
        };
        let durable_tool_calls_count = self
            .event_store
            .task_event_count(
                emitter.session_id(),
                task_id,
                crate::events::EventType::ToolCall,
            )
            .await
            .unwrap_or(tool_calls_count as u32);
        let durable_duration_secs = self
            .event_store
            .task_elapsed_secs(emitter.session_id(), task_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| task_start.elapsed().as_secs());
        // Every terminalization path carries a task-level cost aggregate,
        // including cancellation/watchdog exits with zero provider calls.
        // A missing row is a real zero, not an unknown lifecycle state.
        let efficiency = Some(self.task_efficiency_data(task_id).await.unwrap_or_default());
        // Task outcome is the authoritative failure signal.  Some bounded
        // closeout paths intentionally return a user-facing fallback with a
        // `Completed` transport status while the semantic outcome is
        // `Failed`; recording only from the transport status silently drops
        // their token cost from policy telemetry.  Existing `Failed` paths
        // retain their legacy pre-boundary increment, so this branch covers
        // only the previously uncounted status projection.
        if !effective_outcome.task_success() && status != TaskStatus::Failed {
            let tokens = efficiency
                .as_ref()
                .map(|data| data.input_tokens.saturating_add(data.output_tokens))
                .unwrap_or_default();
            super::policy_metrics::record_failed_task_tokens(tokens);
        }
        let harness_eval_snapshot = if self.harness_eval_config.enabled {
            let acc = self.harness_eval.write().await.take();
            acc.map(|accumulator| {
                accumulator.finalize(
                    effective_outcome,
                    iteration as u32,
                    durable_tool_calls_count,
                    efficiency.as_ref(),
                )
            })
        } else {
            None
        };
        // Stamp the active turn so the normal TaskEnd carries turn identity
        // (recovery TaskEnd uses the checkpoint's original turn_id instead).
        let turn_id = self
            .current_turn_ids
            .read()
            .await
            .get(emitter.session_id())
            .cloned();
        if let Some(ref snapshot) = harness_eval_snapshot {
            super::policy_metrics::record_harness_eval_task(snapshot);
        }
        self.emit_decision_point(
            emitter,
            task_id,
            iteration,
            crate::events::DecisionType::PostExecutionValidation,
            "Task finalization reconciled against durable receipt state",
            serde_json::json!({
                "condition": "task_finalization_reconciled",
                "requested_outcome": outcome,
                "effective_outcome": effective_outcome,
                "status": status,
                "receipt_contract_present": receipt_closure.contract_present,
                "required_receipts": receipt_closure.required,
                "satisfied_receipts": receipt_closure.satisfied,
                "receipt_cardinality_violations": receipt_closure.cardinality_violations,
                "persisted_receipt_count": receipt_closure.receipt_count,
                "required_mutation_effects": receipt_closure
                    .required_mutation_effects
                    .telemetry_value(),
                "observed_mutation_effects": receipt_closure
                    .observed_mutation_effects
                    .telemetry_value(),
                "evidence_required": receipt_closure.evidence_required,
                "evidence_satisfied": receipt_closure.evidence_satisfied,
                "observation_required": receipt_closure.observation_required,
                "observation_satisfied": receipt_closure.observation_satisfied,
                "tool_calls_count": durable_tool_calls_count,
            }),
        )
        .await;
        let completion_proof = crate::events::TaskCompletionProofData {
            schema_version: 1,
            task_id: task_id.to_string(),
            request_turn_id: turn_id.clone(),
            response_message_ids: self
                .event_store
                .task_response_message_ids(task_id)
                .await
                .unwrap_or_default(),
            receipt_refs: self
                .event_store
                .task_completion_proof_references(task_id)
                .await
                .unwrap_or_default(),
            closed_at: chrono::Utc::now().to_rfc3339(),
        };
        let _ = emitter
            .emit(
                EventType::TaskEnd,
                TaskEndData {
                    task_id: task_id.to_string(),
                    status,
                    outcome: Some(effective_outcome),
                    duration_secs: durable_duration_secs,
                    iterations: iteration as u32,
                    tool_calls_count: durable_tool_calls_count,
                    error,
                    summary,
                    efficiency,
                    turn_id,
                    completion_proof: Some(completion_proof),
                    harness_eval: harness_eval_snapshot,
                },
            )
            .await;
        // Counters remain lock-free on hot paths. Checkpoint their cumulative
        // value at the authoritative task boundary so restarts do not erase
        // longitudinal policy telemetry. Readers select the latest row per
        // boot_id rather than summing cumulative rows.
        let _ = emitter
            .emit(
                EventType::PolicyMetricsSnapshot,
                super::policy_metrics::durable_policy_metrics_snapshot(),
            )
            .await;
        if let Err(err) = super::dialogue_state::record_dialogue_task_end(
            self,
            emitter.session_id(),
            task_id,
            status,
            effective_outcome,
        )
        .await
        {
            warn!(
                session_id = emitter.session_id(),
                task_id,
                error = %err,
                "Failed to record dialogue task end"
            );
        }
        self.run_task_end_tool_hooks(task_id, emitter.session_id())
            .await;
        self.emit_turn_efficiency_signal(emitter, task_id, iteration)
            .await;
    }

    /// Tier 2 reflection signal: roll up this turn's `LlmCall` telemetry and
    /// log it. When the turn looks inefficient (fallbacks, retries, heavy
    /// iteration loops, or large token-estimate drift), also emit a warning
    /// `DecisionPoint` so the agent's own `self_diagnose` surfaces it.
    pub(in crate::agent) async fn task_efficiency_data(
        &self,
        task_id: &str,
    ) -> Option<crate::events::TaskEfficiencyData> {
        let summary = self.event_store.get_task_llm_stats(task_id).await.ok()?;
        if summary.total_calls == 0 {
            return None;
        }
        let drift = summary.est_input_drift();
        let reasons = Self::task_efficiency_reasons(&summary, drift);
        Some(crate::events::TaskEfficiencyData {
            llm_calls: summary.total_calls,
            attempts: summary.total_attempts,
            fell_back_count: summary.fell_back_count,
            p95_latency_ms: summary.p95_latency_ms,
            max_latency_ms: summary.max_latency_ms,
            max_latency_iteration: summary.max_latency_iteration,
            input_tokens: summary.total_input_tokens,
            output_tokens: summary.total_output_tokens,
            failed_est_input_tokens: (summary.failed_est_input_tokens > 0)
                .then_some(summary.failed_est_input_tokens),
            cached_input_tokens: if summary.cached_input_token_samples > 0 {
                Some(summary.total_cached_input_tokens)
            } else {
                None
            },
            cache_creation_input_tokens: if summary.cache_creation_input_token_samples > 0 {
                Some(summary.total_cache_creation_input_tokens)
            } else {
                None
            },
            fresh_input_tokens: if summary.cached_input_token_samples > 0 {
                Some(
                    summary
                        .total_input_tokens
                        .saturating_sub(summary.total_cached_input_tokens),
                )
            } else {
                None
            },
            est_input_drift: drift,
            final_model: summary.final_model,
            reasons,
        })
    }

    fn task_efficiency_reasons(summary: &crate::events::TaskLlmSummary, drift: i64) -> Vec<String> {
        let mut reasons = Vec::new();
        if summary.fell_back_count > 0 {
            reasons.push(format!("{} fallback(s)", summary.fell_back_count));
        }
        if summary.total_attempts > summary.total_calls {
            reasons.push(format!(
                "{} retry attempt(s) over {} call(s)",
                summary.total_attempts - summary.total_calls,
                summary.total_calls
            ));
        }
        if summary.total_calls >= 8 {
            reasons.push(format!("{} LLM calls (heavy loop)", summary.total_calls));
        }
        if summary.est_samples > 0 && summary.actual_input_tokens_with_est > 0 {
            let pct = (drift.abs() as f64 / summary.actual_input_tokens_with_est as f64) * 100.0;
            if pct >= 30.0 {
                reasons.push(format!("token estimate off by {pct:.0}%"));
            }
        }
        reasons
    }

    async fn emit_turn_efficiency_signal(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
    ) {
        let summary = match self.event_store.get_task_llm_stats(task_id).await {
            Ok(s) if s.total_calls > 0 => s,
            _ => return,
        };
        let drift = summary.est_input_drift();
        info!(
            task_id,
            session_id = emitter.session_id(),
            llm_calls = summary.total_calls,
            attempts = summary.total_attempts,
            fell_back = summary.fell_back_count,
            p95_latency_ms = summary.p95_latency_ms,
            max_latency_ms = summary.max_latency_ms,
            input_tokens = summary.total_input_tokens,
            output_tokens = summary.total_output_tokens,
            est_input_drift = drift,
            final_model = summary.final_model.as_deref().unwrap_or("?"),
            "Turn efficiency summary"
        );

        if !summary.is_inefficient() {
            return;
        }
        let reasons = Self::task_efficiency_reasons(&summary, drift);
        let summary_text = format!("Inefficient turn: {}", reasons.join(", "));
        self.emit_warning_decision_point(
            emitter,
            task_id,
            iteration,
            DecisionType::LlmEfficiencyAlert,
            summary_text,
            json!({
                "reason": reasons.join(", "),
                "llm_calls": summary.total_calls,
                "attempts": summary.total_attempts,
                "fell_back_count": summary.fell_back_count,
                "p95_latency_ms": summary.p95_latency_ms,
                "max_latency_ms": summary.max_latency_ms,
                "max_latency_iteration": summary.max_latency_iteration,
                "input_tokens": summary.total_input_tokens,
                "output_tokens": summary.total_output_tokens,
                "est_input_drift": drift,
                "final_model": summary.final_model,
            }),
        )
        .await;
    }

    pub(in crate::agent) async fn emit_decision_point(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: DecisionType,
        summary: impl Into<String>,
        metadata: Value,
    ) {
        self.emit_decision_point_with_severity(
            emitter,
            task_id,
            iteration,
            DecisionPointEmission {
                decision_type,
                severity: crate::events::DiagnosticSeverity::Info,
                summary: summary.into(),
                metadata,
            },
        )
        .await;
    }

    pub(super) async fn emit_warning_decision_point(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: DecisionType,
        summary: impl Into<String>,
        metadata: Value,
    ) {
        self.emit_decision_point_with_severity(
            emitter,
            task_id,
            iteration,
            DecisionPointEmission {
                decision_type,
                severity: crate::events::DiagnosticSeverity::Warning,
                summary: summary.into(),
                metadata,
            },
        )
        .await;
    }

    async fn emit_decision_point_with_severity(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        emission: DecisionPointEmission,
    ) {
        if !self.record_decision_points {
            return;
        }
        let code = emission
            .metadata
            .as_object()
            .and_then(|obj| {
                ["condition", "route_reason", "reason"]
                    .iter()
                    .find_map(|key| obj.get(*key).and_then(Value::as_str))
            })
            .map(|value| value.to_string())
            .or_else(|| Some(emission.decision_type.as_str().to_string()));
        let _ = emitter
            .emit(
                EventType::DecisionPoint,
                DecisionPointData {
                    decision_type: emission.decision_type,
                    task_id: task_id.to_string(),
                    iteration: iteration.min(u32::MAX as usize) as u32,
                    severity: emission.severity,
                    code,
                    metadata: emission.metadata,
                    summary: emission.summary,
                },
            )
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meaningful_budget_progress_accepts_evidence_gain() {
        assert!(Agent::has_meaningful_budget_progress(1, 0));
    }

    #[test]
    fn meaningful_budget_progress_rejects_transport_success_without_evidence() {
        assert!(!Agent::has_meaningful_budget_progress(0, 30));
    }

    #[test]
    fn meaningful_budget_progress_rejects_shallow_runs_without_evidence() {
        assert!(!Agent::has_meaningful_budget_progress(0, 2));
    }

    #[test]
    fn scheduled_run_metrics_detect_unproductive_snapshot() {
        assert!(Agent::scheduled_run_metrics_are_clearly_unproductive(
            &crate::traits::ScheduledRunHealth {
                evidence_gain_count: 0,
                total_successful_tool_calls: 0,
                stall_count: 0,
                consecutive_same_tool_count: 0,
                consecutive_same_tool_unique_args: 0,
                unrecovered_error_count: 1,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn scheduled_run_auto_extension_candidate_requires_health() {
        assert_eq!(
            Agent::scheduled_run_auto_extension_candidate(
                &crate::goal_tokens::GoalRunBudgetStatus {
                    effective_budget_per_check: 100,
                    tokens_used: 100,
                    budget_extensions_count: 0,
                    health: crate::traits::ScheduledRunHealth::default(),
                },
                12,
                1_000,
            ),
            None
        );
    }

    #[test]
    fn scheduled_run_auto_extension_candidate_accepts_required_mutation_receipt() {
        assert_eq!(
            Agent::scheduled_run_auto_extension_candidate(
                &crate::goal_tokens::GoalRunBudgetStatus {
                    effective_budget_per_check: 100,
                    tokens_used: 100,
                    budget_extensions_count: 0,
                    health: crate::traits::ScheduledRunHealth {
                        evidence_gain_count: 1,
                        total_successful_tool_calls: 3,
                        completion_requires_mutation: true,
                        required_mutation_progress: true,
                        stall_count: 0,
                        consecutive_same_tool_count: 1,
                        consecutive_same_tool_unique_args: 1,
                        unrecovered_error_count: 0,
                        ..Default::default()
                    },
                },
                12,
                1_000,
            ),
            Some(200)
        );
    }

    #[test]
    fn scheduled_run_auto_extension_rejects_read_only_activity_for_mutation_task() {
        let status = crate::goal_tokens::GoalRunBudgetStatus {
            effective_budget_per_check: 400_000,
            tokens_used: 400_000,
            budget_extensions_count: 0,
            health: crate::traits::ScheduledRunHealth {
                evidence_gain_count: 28,
                total_successful_tool_calls: 30,
                completion_requires_mutation: true,
                required_mutation_progress: false,
                stall_count: 0,
                consecutive_same_tool_count: 1,
                consecutive_same_tool_unique_args: 1,
                unrecovered_error_count: 0,
                ..Default::default()
            },
        };

        assert_eq!(
            Agent::scheduled_run_auto_extension_candidate(&status, 1, 2_000_000),
            None
        );
    }

    #[test]
    fn scheduled_run_auto_extension_accepts_verified_research_progress() {
        let status = crate::goal_tokens::GoalRunBudgetStatus {
            effective_budget_per_check: 100,
            tokens_used: 100,
            budget_extensions_count: 0,
            health: crate::traits::ScheduledRunHealth {
                evidence_gain_count: 2,
                total_successful_tool_calls: 2,
                completion_requires_observation: true,
                verification_progress: true,
                ..Default::default()
            },
        };

        assert_eq!(
            Agent::scheduled_run_auto_extension_candidate(&status, 1, 1_000),
            Some(200)
        );
    }

    #[test]
    fn scheduled_run_auto_extension_candidate_respects_autonomous_extension_limit() {
        assert_eq!(
            Agent::scheduled_run_auto_extension_candidate(
                &crate::goal_tokens::GoalRunBudgetStatus {
                    effective_budget_per_check: 200,
                    tokens_used: 200,
                    budget_extensions_count: SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS,
                    health: crate::traits::ScheduledRunHealth {
                        evidence_gain_count: 2,
                        total_successful_tool_calls: 4,
                        completion_requires_mutation: true,
                        required_mutation_progress: true,
                        stall_count: 0,
                        consecutive_same_tool_count: 1,
                        consecutive_same_tool_unique_args: 1,
                        unrecovered_error_count: 0,
                        ..Default::default()
                    },
                },
                SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS,
                SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
            ),
            None
        );
    }

    #[test]
    fn scheduled_run_budget_pressure_fires_once_before_exhaustion() {
        let status = crate::goal_tokens::GoalRunBudgetStatus {
            effective_budget_per_check: 400_000,
            tokens_used: 320_000,
            budget_extensions_count: 0,
            health: crate::traits::ScheduledRunHealth::default(),
        };
        assert_eq!(
            Agent::scheduled_run_budget_pressure_pct(&status, false),
            Some(80)
        );
        assert_eq!(
            Agent::scheduled_run_budget_pressure_pct(&status, true),
            None
        );

        let exhausted = crate::goal_tokens::GoalRunBudgetStatus {
            tokens_used: 400_000,
            ..status
        };
        assert_eq!(
            Agent::scheduled_run_budget_pressure_pct(&exhausted, false),
            None
        );
    }

    #[test]
    fn alert_destination_dedupes_legacy_and_namespaced_telegram_sessions() {
        assert!(Agent::same_alert_destination(
            "301753035",
            "aidaemon_coding_bot:301753035"
        ));
        assert!(!Agent::same_alert_destination(
            "slack:U018EAFV5QR",
            "aidaemon_coding_bot:301753035"
        ));
    }
}
