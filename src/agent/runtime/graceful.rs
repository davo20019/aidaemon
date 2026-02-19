use super::*;

impl Agent {
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

        let hub_weak = self.hub.read().await.clone();
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

        let hub = self.hub.read().await.clone();
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

    /// Classify the stall cause from recent errors for actionable guidance.
    #[allow(dead_code)] // Used in tests; production path delegates through post_task.
    pub(super) fn classify_stall(learning_ctx: &LearningContext) -> (&'static str, &'static str) {
        post_task::classify_stall(learning_ctx, DEFERRED_NO_TOOL_ERROR_MARKER)
    }

    /// Graceful response when agent is stalled (no progress).
    pub(super) async fn graceful_stall_response(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        learning_ctx: &LearningContext,
        sent_file_successfully: bool,
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_stall_response(
            learning_ctx,
            sent_file_successfully,
            DEFERRED_NO_TOOL_ERROR_MARKER,
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
    ) -> anyhow::Result<String> {
        let summary = post_task::graceful_partial_stall_response(
            learning_ctx,
            sent_file_successfully,
            DEFERRED_NO_TOOL_ERROR_MARKER,
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
        let model = self.model.read().await.clone();
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
