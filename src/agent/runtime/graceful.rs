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
    pub pending_system_messages: &'a mut Vec<String>,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
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

impl Agent {
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
    pub(super) async fn request_budget_continue_approval(
        &self,
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

        let approval_desc = format!(
            "Extend {} token budget from {} to {} and continue?",
            scope_label, current_budget, proposed_budget
        );
        let warnings = vec![
            format!("Current usage: {} tokens.", used_tokens),
            "This may increase spend for this run.".to_string(),
        ];

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
            Ok(ApprovalResponse::Deny) => false,
            Err(e) => {
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
        let budget_daily = ctx.effective_goal_daily_budget.unwrap_or(db_budget_daily);
        if budget_daily <= 0 || ctx.status.tokens_used_today < budget_daily {
            return GoalBudgetControlOutcome::Continue;
        }

        let old_gbudget = budget_daily;
        let new_gbudget = old_gbudget
            .saturating_mul(2)
            .max(ctx.status.tokens_used_today.saturating_add(old_gbudget / 2))
            .min(ctx.hard_token_cap);

        let productive = ctx.evidence_gain_count >= 2
            && post_task::is_productive(
                ctx.learning_ctx,
                ctx.stall_count,
                ctx.consecutive_same_tool_count,
                ctx.consecutive_same_tool_unique_args,
                ctx.total_successful_tool_calls,
            );

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
            info!(
                ctx.session_id,
                goal_id = %ctx.goal_id,
                old_budget = old_gbudget,
                new_budget = new_gbudget,
                extension = *ctx.budget_extensions_count,
                source = source_label,
                "Auto-extended goal daily token budget in-memory"
            );
            ctx.pending_system_messages.push(format!(
                "[SYSTEM] Goal daily token budget auto-extended from {} to {} ({}/{} extensions). \
                 Continue working.",
                old_gbudget, new_gbudget, *ctx.budget_extensions_count, ctx.max_budget_extensions
            ));
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
            ctx.pending_system_messages.push(format!(
                "[SYSTEM] Goal daily token budget extension approved by owner: {} -> {}. \
                 Continue working.",
                old_gbudget, new_gbudget
            ));
            self.emit_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::BudgetAutoExtension,
                "Extended goal daily token budget via owner approval".to_string(),
                json!({
                    "condition": manual_condition,
                    "goal_id": ctx.goal_id,
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
            embedding: None,
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

    /// Graceful response when a goal hits its daily token budget.
    pub(super) async fn graceful_goal_daily_budget_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        tokens_used_today: i64,
        budget_daily: i64,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_goal_daily_budget_response(
            learning_ctx,
            tokens_used_today,
            budget_daily,
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
        if !self.record_decision_points {
            return;
        }
        let _ = emitter
            .emit(
                EventType::DecisionPoint,
                DecisionPointData {
                    decision_type,
                    task_id: task_id.to_string(),
                    iteration: iteration.min(u32::MAX as usize) as u32,
                    metadata,
                    summary: summary.into(),
                },
            )
            .await;
    }
}
