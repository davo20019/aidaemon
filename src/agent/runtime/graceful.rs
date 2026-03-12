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
    pub user_role: UserRole,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
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

impl Agent {
    pub(super) fn has_meaningful_budget_progress(
        evidence_gain_count: usize,
        total_successful_tool_calls: usize,
    ) -> bool {
        // A single evidence gain is enough to show the run produced something
        // concrete; otherwise require at least a few successful tool calls so
        // we do not auto-extend pure narration or shallow retries.
        evidence_gain_count > 0 || total_successful_tool_calls >= 3
    }

    pub(super) fn scheduled_run_health_snapshot(
        learning_ctx: &LearningContext,
        evidence_gain_count: usize,
        stall_count: usize,
        consecutive_same_tool_count: usize,
        consecutive_same_tool_unique_args: usize,
        total_successful_tool_calls: usize,
    ) -> crate::traits::ScheduledRunHealth {
        crate::traits::ScheduledRunHealth {
            evidence_gain_count,
            total_successful_tool_calls,
            stall_count,
            consecutive_same_tool_count,
            consecutive_same_tool_unique_args,
            unrecovered_error_count: learning_ctx
                .errors
                .iter()
                .filter(|(_, recovered)| !recovered)
                .count(),
        }
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

        let has_meaningful_progress = Self::has_meaningful_budget_progress(
            status.health.evidence_gain_count,
            status.health.total_successful_tool_calls,
        );
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
        let budget_daily = (*ctx.effective_goal_daily_budget)
            .or(shared_budget_daily)
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
            Self::has_meaningful_budget_progress(
                ctx.evidence_gain_count,
                ctx.total_successful_tool_calls,
            ) && ctx.stall_count == 0
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
            send_status(
                ctx.status_tx,
                StatusUpdate::BudgetExtended {
                    old_budget: old_gbudget,
                    new_budget: new_gbudget,
                    extension: *ctx.budget_extensions_count,
                    max_extensions: ctx.max_budget_extensions,
                },
            );
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
        let proposed_budget = old_budget
            .saturating_mul(2)
            .max(ctx.status.tokens_used.saturating_add(old_budget / 2))
            .min(ctx.hard_token_cap);
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
                send_status(
                    ctx.status_tx,
                    StatusUpdate::BudgetExtended {
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

        let approved_extension =
            if old_budget < ctx.hard_token_cap && proposed_budget > ctx.status.tokens_used {
                self.request_budget_continue_approval(
                    ctx.emitter,
                    ctx.task_id,
                    ctx.iteration,
                    ctx.session_id,
                    ctx.user_role,
                    "scheduled run",
                    ctx.status.tokens_used,
                    old_budget,
                    proposed_budget,
                )
                .await
            } else {
                false
            };

        if approved_extension {
            if let Some(registry) = &self.goal_token_registry {
                if let Some(status) = registry.set_run_budget(ctx.goal_id, proposed_budget).await {
                    persist_scheduled_run_state(&self.state, ctx.goal_id, None, &status).await;
                }
            }
            self.emit_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::BudgetAutoExtension,
                "Extended scheduled run budget via owner approval".to_string(),
                json!({
                    "condition": "scheduled_run_budget_extension_manual",
                    "goal_id": ctx.goal_id,
                    "approval_state": ApprovalState::Granted,
                    "old_budget": old_budget,
                    "new_budget": proposed_budget,
                    "tokens_used": ctx.status.tokens_used,
                }),
            )
            .await;
            return ScheduledRunBudgetControlOutcome::Continue;
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

    /// Fan-out token alerts to owner sessions plus the triggering session.
    pub(super) async fn fanout_token_alert(
        &self,
        goal_id: Option<&str>,
        trigger_session_id: &str,
        message: &str,
        suppress_session_id: Option<&str>,
    ) {
        let mut targets = self.load_default_alert_sessions().await;
        targets.push(trigger_session_id.to_string());
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

    /// Emit a TaskEnd event. Called from every exit path in the agent loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn emit_task_end(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        status: TaskStatus,
        task_start: Instant,
        iteration: usize,
        tool_calls_count: usize,
        error: Option<String>,
        summary: Option<String>,
    ) {
        let _ = emitter
            .emit(
                EventType::TaskEnd,
                TaskEndData {
                    task_id: task_id.to_string(),
                    status,
                    duration_secs: task_start.elapsed().as_secs(),
                    iterations: iteration as u32,
                    tool_calls_count: tool_calls_count as u32,
                    error,
                    summary,
                },
            )
            .await;
        self.run_task_end_tool_hooks(task_id, emitter.session_id())
            .await;
    }

    pub(super) async fn emit_decision_point(
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
    fn meaningful_budget_progress_accepts_three_successful_calls() {
        assert!(Agent::has_meaningful_budget_progress(0, 3));
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
    fn scheduled_run_auto_extension_candidate_accepts_productive_snapshot() {
        assert_eq!(
            Agent::scheduled_run_auto_extension_candidate(
                &crate::goal_tokens::GoalRunBudgetStatus {
                    effective_budget_per_check: 100,
                    tokens_used: 100,
                    budget_extensions_count: 0,
                    health: crate::traits::ScheduledRunHealth {
                        evidence_gain_count: 1,
                        total_successful_tool_calls: 3,
                        stall_count: 0,
                        consecutive_same_tool_count: 1,
                        consecutive_same_tool_unique_args: 1,
                        unrecovered_error_count: 0,
                    },
                },
                12,
                1_000,
            ),
            Some(200)
        );
    }
}
