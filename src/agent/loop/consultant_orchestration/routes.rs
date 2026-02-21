use super::types::ConsultantOrchestrationCtx;
use crate::agent::consultant_direct_return::consultant_direct_return_ok;
use crate::agent::consultant_fallthrough::consultant_fallthrough;
use crate::agent::consultant_phase::ConsultantPhaseOutcome;
use crate::agent::recall_guardrails::{
    filter_tool_defs_for_personal_memory, is_personal_memory_tool,
};
use crate::agent::*;

impl Agent {
    fn extract_labeled_block(text: &str, label: &str, end_labels: &[&str]) -> Option<String> {
        let start = text.find(label)?;
        let after = &text[start + label.len()..];
        let mut end = after.len();
        for end_label in end_labels {
            if let Some(idx) = after.find(end_label) {
                end = end.min(idx);
            }
        }
        let block = after[..end].trim();
        if block.is_empty() {
            None
        } else {
            Some(block.to_string())
        }
    }

    fn build_scheduled_goal_description(current_user_text: &str, goal_user_text: &str) -> String {
        let current = current_user_text.trim();
        let composed = goal_user_text.trim();
        if composed.is_empty() {
            return current.to_string();
        }
        if !composed.contains("Original request:") && !composed.contains("Follow-up:") {
            return composed.to_string();
        }

        let original = Self::extract_labeled_block(
            composed,
            "Original request:",
            &["Assistant asked:", "Follow-up:"],
        );
        let followup = Self::extract_labeled_block(composed, "Follow-up:", &[]);
        let mut pieces = Vec::new();
        if let Some(original_text) = original {
            pieces.push(original_text);
        }
        if let Some(followup_text) = followup {
            let duplicate = pieces
                .iter()
                .any(|piece| piece.eq_ignore_ascii_case(&followup_text));
            if !duplicate {
                pieces.push(followup_text);
            }
        }
        if !pieces.is_empty() {
            return pieces.join(" | ");
        }

        let flattened = composed
            .replace("Original request:", "")
            .replace("Assistant asked:", "")
            .replace("Follow-up:", "")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !flattened.is_empty() {
            flattened
        } else {
            current.to_string()
        }
    }

    async fn confirm_scheduled_goal_activation(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
        goal: &Goal,
        schedule: &crate::traits::GoalSchedule,
        tz_label: &str,
        completion_note: &str,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        let activation_msg = match self.state.activate_goal(&goal.id).await {
            Ok(true) => {
                if let Some(ref registry) = self.goal_token_registry {
                    registry.register(&goal.id).await;
                }
                let next_run = chrono::DateTime::parse_from_rfc3339(&schedule.next_run_at)
                    .ok()
                    .map(|dt| {
                        dt.with_timezone(&chrono::Local)
                            .format("%Y-%m-%d %H:%M %Z")
                            .to_string()
                    })
                    .unwrap_or_else(|| "n/a".to_string());
                format!(
                    "Scheduled: {} (next: {}). I'll execute it when the time comes. System timezone: {}.",
                    goal.description, next_run, tz_label
                )
            }
            Ok(false) => {
                "I couldn't activate that scheduled goal because it is no longer pending confirmation."
                    .to_string()
            }
            Err(e) => {
                format!("I couldn't activate the scheduled goal: {}", e)
            }
        };
        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: ctx.session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(activation_msg.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.append_assistant_message_with_event(ctx.emitter, &assistant_msg, "system", None, None)
            .await?;
        self.emit_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some(completion_note.to_string()),
        )
        .await;
        Ok(consultant_direct_return_ok(activation_msg))
    }

    async fn ensure_orchestrator_tools_loaded(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
    ) -> anyhow::Result<()> {
        if !ctx.tool_defs.is_empty() || !ctx.tools_allowed_for_user {
            return Ok(());
        }

        let (mut defs, mut base_defs, mut caps) = self
            .load_policy_tool_set(
                ctx.user_text,
                ctx.channel_ctx.visibility,
                &ctx.policy_bundle.policy,
                ctx.policy_bundle.risk_score,
                self.policy_config.tool_filter_enforce,
            )
            .await;

        if ctx.restrict_to_personal_memory_tools {
            defs = filter_tool_defs_for_personal_memory(&defs);
            base_defs = filter_tool_defs_for_personal_memory(&base_defs);
            caps.retain(|name, _| is_personal_memory_tool(name));
        }

        *ctx.tool_defs = defs;
        *ctx.base_tool_defs = base_defs;
        *ctx.available_capabilities = caps;
        Ok(())
    }

    pub(super) async fn maybe_handle_generic_cancel_request(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
    ) -> anyhow::Result<Option<ConsultantPhaseOutcome>> {
        // Check for cancel/stop intent before routing
        let lower_trimmed = ctx.user_text.trim().to_lowercase();
        let explicit_cancel_command =
            lower_trimmed == "/cancel" || lower_trimmed.starts_with("/cancel ");
        let model_requests_generic_cancel = ctx.intent_gate.cancel_intent.unwrap_or(false)
            && ctx.intent_gate.cancel_scope.as_deref() == Some("generic");
        let generic_cancel_request = explicit_cancel_command || model_requests_generic_cancel;

        // Only auto-cancel on generic stop/cancel commands.
        // Targeted requests ("cancel this goal: X") should flow
        // through normal tool routing so selection can be explicit.
        if !generic_cancel_request {
            return Ok(None);
        }

        let active_goals = self
            .state
            .get_goals_for_session(ctx.session_id)
            .await
            .unwrap_or_default();
        let active: Vec<&Goal> = active_goals
            .iter()
            .filter(|g| {
                g.status == "active" || g.status == "pending" || g.status == "pending_confirmation"
            })
            .collect();

        if active.is_empty() {
            return Ok(None);
        }

        let mut cancelled = Vec::new();
        for goal in &active {
            // Cancel via token hierarchy (cascades to task lead + executors)
            if let Some(ref registry) = self.goal_token_registry {
                registry.cancel(&goal.id).await;
            }

            // Update goal DB status
            let mut updated = (*goal).clone();
            updated.status = "cancelled".to_string();
            updated.updated_at = chrono::Utc::now().to_rfc3339();
            let _ = self.state.update_goal(&updated).await;

            // Best-effort cleanup: cancelled goals should not retain schedules.
            if let Ok(schedules) = self.state.get_schedules_for_goal(&updated.id).await {
                for s in &schedules {
                    let _ = self.state.delete_goal_schedule(&s.id).await;
                }
            }

            // Cancel all remaining tasks for this goal
            if let Ok(tasks) = self.state.get_tasks_for_goal(&goal.id).await {
                for task in &tasks {
                    if task.status != "completed"
                        && task.status != "failed"
                        && task.status != "cancelled"
                    {
                        let mut cancelled_task = task.clone();
                        cancelled_task.status = "cancelled".to_string();
                        let _ = self.state.update_task(&cancelled_task).await;
                    }
                }
            }

            cancelled.push(goal.description.chars().take(100).collect::<String>());
        }

        info!(
            ctx.session_id,
            count = cancelled.len(),
            "Cancelled active goals"
        );

        let msg = if cancelled.len() == 1 {
            format!("Cancelled: {}", cancelled[0])
        } else {
            format!(
                "Cancelled {} goals:\n{}",
                cancelled.len(),
                cancelled
                    .iter()
                    .map(|d| format!("- {}", d))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };

        self.emit_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some(msg.clone()),
        )
        .await;

        Ok(Some(consultant_direct_return_ok(msg)))
    }

    async fn handle_scheduled_missing_timing_intent(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        // Fall through to the full agent loop instead of
        // giving up. The LLM with tools can ask for timing
        // or infer it from context.
        info!(
            ctx.session_id,
            "ScheduledMissingTiming — falling through to agent loop"
        );
        self.ensure_orchestrator_tools_loaded(ctx).await?;
        Ok(consultant_fallthrough())
    }

    async fn handle_scheduled_intent(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
        mut schedule_raw: String,
        schedule_cron: Option<String>,
        is_one_shot: bool,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        if ctx.user_role != UserRole::Owner {
            // Non-owners cannot create scheduled goals — load tools and
            // fall through to agent loop so the request is handled directly.
            self.ensure_orchestrator_tools_loaded(ctx).await?;
            ctx.pending_system_messages.push(
                "[SYSTEM] Scheduling goals is owner-only. Handle this request directly without creating a goal.".to_string(),
            );
            return Ok(consultant_fallthrough());
        }

        let mut cron_expr = schedule_cron
            .as_ref()
            .filter(|candidate| {
                let parts: Vec<&str> = candidate.split_whitespace().collect();
                parts.len() == 5
            })
            .and_then(|candidate| candidate.parse::<Cron>().ok().map(|_| candidate.clone()))
            .or_else(|| {
                if schedule_cron.is_some() {
                    warn!(
                        ctx.session_id,
                        schedule_raw = %schedule_raw,
                        schedule_cron = ?schedule_cron,
                        "INTENT_GATE provided invalid schedule_cron; falling back to parser"
                    );
                }
                crate::cron_utils::parse_schedule(&schedule_raw).ok()
            });

        // E2E guardrail: if the model provided a malformed schedule string
        // (e.g. "2 minutes" instead of "in 2 minutes"), retry parsing using
        // heuristic extraction from the user text.
        if cron_expr.is_none() && ENABLE_SCHEDULE_HEURISTICS {
            if let Some((heuristic_raw, _)) =
                intent_routing::detect_schedule_heuristic(ctx.user_text)
            {
                if heuristic_raw != schedule_raw {
                    cron_expr = crate::cron_utils::parse_schedule(&heuristic_raw).ok();
                    if cron_expr.is_some() {
                        warn!(
                            ctx.session_id,
                            schedule_raw = %schedule_raw,
                            heuristic_raw = %heuristic_raw,
                            "Schedule parse failed; recovered using heuristic schedule"
                        );
                        schedule_raw = heuristic_raw;
                    }
                }
            }
        }

        let cron_expr = match cron_expr {
            Some(expr) => expr,
            None => {
                // Schedule parse failed — fall through to the
                // full agent loop instead of giving up. The LLM
                // with tools can handle the request directly.
                warn!(
                    ctx.session_id,
                    schedule_raw = %schedule_raw,
                    "Schedule parse failed — falling through to agent loop"
                );
                self.ensure_orchestrator_tools_loaded(ctx).await?;
                return Ok(consultant_fallthrough());
            }
        };

        // Cron-shape heuristic override: if the cron pins a specific
        // day+month it is clearly one-shot regardless of what the LLM
        // classified (LLMs often return "recurring" for "in 2 minutes").
        let cron_looks_one_shot = crate::cron_utils::is_one_shot_schedule(&cron_expr);
        let actually_one_shot = cron_looks_one_shot || is_one_shot;

        if is_internal_maintenance_intent(ctx.user_text) {
            let msg = "Memory maintenance already runs via built-in background jobs (embeddings, consolidation, decay, retention). I won't create a scheduled goal for that.".to_string();
            let assistant_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: ctx.session_id.to_string(),
                role: "assistant".to_string(),
                content: Some(msg.clone()),
                tool_call_id: None,
                tool_name: None,
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                embedding: None,
            };
            self.append_assistant_message_with_event(
                ctx.emitter,
                &assistant_msg,
                "system",
                None,
                None,
            )
            .await?;
            self.emit_task_end(
                ctx.emitter,
                ctx.task_id,
                TaskStatus::Completed,
                ctx.task_start,
                ctx.iteration,
                0,
                None,
                Some(msg.chars().take(200).collect()),
            )
            .await;
            return Ok(consultant_direct_return_ok(msg));
        }

        let goal_user_text = ctx.turn_context.goal_user_text.clone();
        let goal_description =
            Self::build_scheduled_goal_description(ctx.user_text, &goal_user_text);

        let mut goal = if actually_one_shot {
            Goal::new_deferred_finite(&goal_description, ctx.session_id)
        } else {
            Goal::new_continuous_pending(&goal_description, ctx.session_id, None, None)
        };

        if let Some(goal_context) = self
            .build_goal_feed_forward_context(
                ctx.session_id,
                &goal_user_text,
                &ctx.turn_context.recent_messages,
                &ctx.turn_context.project_hints,
            )
            .await
        {
            goal.context = Some(goal_context);
        }

        self.state.create_goal(&goal).await?;

        let now = chrono::Utc::now().to_rfc3339();
        let next_run_local = crate::cron_utils::compute_next_run_local(&cron_expr)?;
        let schedule = crate::traits::GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: cron_expr.clone(),
            tz: "local".to_string(),
            original_schedule: Some(schedule_raw.clone()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: actually_one_shot,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run_local.with_timezone(&chrono::Utc).to_rfc3339(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        self.state.create_goal_schedule(&schedule).await?;

        let tz_label = crate::cron_utils::system_timezone_display();
        let schedule_desc = if actually_one_shot {
            next_run_local.format("%Y-%m-%d %H:%M %Z").to_string()
        } else {
            let next_local = next_run_local.format("%Y-%m-%d %H:%M %Z").to_string();
            format!("{} (next: {})", schedule_raw, next_local)
        };
        let schedule_kind = if actually_one_shot {
            "one-time"
        } else {
            "recurring"
        };

        let already_approved = {
            let approved = self.schedule_approved_sessions.read().await;
            approved.contains(ctx.session_id)
        };
        if already_approved {
            return self
                .confirm_scheduled_goal_activation(
                    ctx,
                    &goal,
                    &schedule,
                    &tz_label,
                    "Scheduled goal auto-confirmed from prior session approval.",
                )
                .await;
        }

        // Prefer inline goal confirmation buttons for schedule confirmation
        // (Telegram/Discord/Slack). Shows Confirm ✅ / Cancel ❌ buttons.
        // Non-inline channels keep the existing text confirm/cancel fallback.
        let inline_confirmation = {
            let hub_weak = self.hub.read().await.clone();
            if let Some(hub_weak) = hub_weak {
                if let Some(hub_arc) = hub_weak.upgrade() {
                    let confirmation_desc = format!(
                        "Schedule {} goal ({}): {}",
                        schedule_kind, schedule_desc, goal.description
                    );
                    let details = vec![
                        format!("{} schedule", schedule_kind),
                        format!("Next: {}", schedule_desc),
                        format!("System timezone: {}", tz_label),
                    ];
                    Some(
                        hub_arc
                            .request_inline_goal_confirmation(
                                ctx.session_id,
                                &confirmation_desc,
                                &details,
                            )
                            .await,
                    )
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(confirmation_result) = inline_confirmation {
            match confirmation_result {
                Ok(true) => {
                    return self
                        .confirm_scheduled_goal_activation(
                            ctx,
                            &goal,
                            &schedule,
                            &tz_label,
                            "Scheduled goal confirmed via inline approval.",
                        )
                        .await;
                }
                Ok(false) => {
                    let now = chrono::Utc::now().to_rfc3339();
                    goal.status = "cancelled".to_string();
                    goal.completed_at = Some(now.clone());
                    goal.updated_at = now;
                    let _ = self.state.update_goal(&goal).await;
                    // Best-effort cleanup: schedules were created pre-confirmation.
                    if let Ok(schedules) = self.state.get_schedules_for_goal(&goal.id).await {
                        for s in &schedules {
                            let _ = self.state.delete_goal_schedule(&s.id).await;
                        }
                    }

                    let cancel_msg = "OK, cancelled the scheduled goal.".to_string();
                    let assistant_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: ctx.session_id.to_string(),
                        role: "assistant".to_string(),
                        content: Some(cancel_msg.clone()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.5,
                        embedding: None,
                    };
                    self.append_assistant_message_with_event(
                        ctx.emitter,
                        &assistant_msg,
                        "system",
                        None,
                        None,
                    )
                    .await?;
                    self.emit_task_end(
                        ctx.emitter,
                        ctx.task_id,
                        TaskStatus::Completed,
                        ctx.task_start,
                        ctx.iteration,
                        0,
                        None,
                        Some("Scheduled goal cancelled via inline approval.".to_string()),
                    )
                    .await;
                    return Ok(consultant_direct_return_ok(cancel_msg));
                }
                Err(e) => {
                    warn!(
                        ctx.session_id,
                        error = %e,
                        "Inline goal confirmation unavailable; falling back to text confirmation"
                    );
                }
            }
        }

        let confirmation = format!(
            "I'll schedule this as a {} task ({}):\n> {}\nSystem timezone: {}.\nReply **confirm** to proceed or **cancel** to discard.",
            schedule_kind, schedule_desc, goal.description, tz_label
        );

        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: ctx.session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(confirmation.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.append_assistant_message_with_event(ctx.emitter, &assistant_msg, "system", None, None)
            .await?;
        self.emit_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some("Scheduled goal awaiting text confirmation.".to_string()),
        )
        .await;
        Ok(consultant_direct_return_ok(confirmation))
    }

    async fn handle_knowledge_intent(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        self.ensure_orchestrator_tools_loaded(ctx).await?;
        ctx.pending_system_messages.push(
            "[SYSTEM] Consultant classified this turn as knowledge. Provide the best direct answer now. \
             Use tools only if needed to verify or retrieve missing facts."
                .to_string(),
        );
        info!(
            ctx.session_id,
            "Knowledge intent — classifier-only pass; continuing to execution loop"
        );
        Ok(consultant_fallthrough())
    }

    async fn handle_simple_intent(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        self.ensure_orchestrator_tools_loaded(ctx).await?;
        if !ctx.tool_defs.is_empty() {
            info!(
                ctx.session_id,
                tool_count = ctx.tool_defs.len(),
                "Simple intent — loaded tools for orchestrator"
            );
        }
        info!(
            ctx.session_id,
            "Simple intent — continuing to full agent loop"
        );

        // Skip to next iteration where the full agent loop
        // runs with all tools and full context.
        Ok(consultant_fallthrough())
    }

    async fn handle_complex_intent(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        if ctx.user_role != UserRole::Owner {
            // Non-owners cannot create complex goals — load tools and
            // fall through to agent loop so the request is handled directly.
            self.ensure_orchestrator_tools_loaded(ctx).await?;
            ctx.pending_system_messages.push(
                "[SYSTEM] Creating goals is owner-only. Handle this request directly without creating a goal."
                    .to_string(),
            );
            return Ok(consultant_fallthrough());
        }

        if is_internal_maintenance_intent(ctx.user_text) {
            let msg = "Memory maintenance already runs via built-in background jobs (embeddings, consolidation, decay, retention). I won't create a goal for that.".to_string();
            let assistant_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: ctx.session_id.to_string(),
                role: "assistant".to_string(),
                content: Some(msg.clone()),
                tool_call_id: None,
                tool_name: None,
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                embedding: None,
            };
            self.append_assistant_message_with_event(
                ctx.emitter,
                &assistant_msg,
                "system",
                None,
                None,
            )
            .await?;
            self.emit_task_end(
                ctx.emitter,
                ctx.task_id,
                TaskStatus::Completed,
                ctx.task_start,
                ctx.iteration,
                0,
                None,
                Some(msg.chars().take(200).collect()),
            )
            .await;
            return Ok(consultant_direct_return_ok(msg));
        }

        // Create goal
        let goal_user_text = ctx.turn_context.goal_user_text.clone();
        let mut goal = Goal::new_finite(&goal_user_text, ctx.session_id);

        // Feed-forward relevant knowledge into goal context.
        if let Some(goal_context) = self
            .build_goal_feed_forward_context(
                ctx.session_id,
                &goal_user_text,
                &ctx.turn_context.recent_messages,
                &ctx.turn_context.project_hints,
            )
            .await
        {
            goal.context = Some(goal_context);
        }

        self.state.create_goal(&goal).await?;

        // Register cancellation token for this goal.
        if let Some(ref registry) = self.goal_token_registry {
            registry.register(&goal.id).await;
        }

        info!(
            ctx.session_id,
            goal_id = %goal.id,
            "Created goal for complex request, spawning task lead in background"
        );

        // Upgrade weak self-reference to Arc for background spawning.
        let self_arc = {
            let self_ref = self.self_ref.read().await;
            self_ref.as_ref().and_then(|w| w.upgrade())
        };

        if let Some(agent_arc) = self_arc {
            // Spawn the task lead in the background — user gets immediate response.
            let bg_hub = self.hub.read().await.clone();
            spawn_background_task_lead(
                agent_arc,
                goal.clone(),
                goal_user_text.clone(),
                ctx.session_id.to_string(),
                ctx.channel_ctx.clone(),
                ctx.user_role,
                self.state.clone(),
                bg_hub,
                self.goal_token_registry.clone(),
                None,
            );
        } else {
            // No self_ref available (sub-agent or test) — fall back to sync.
            warn!("No self_ref available, running task lead synchronously");
            let result = self
                .spawn_task_lead(
                    &goal.id,
                    &goal.description,
                    &goal_user_text,
                    ctx.status_tx.clone(),
                    ctx.channel_ctx.clone(),
                    ctx.user_role,
                )
                .await;

            match result {
                Ok(response) => {
                    let mut updated_goal = goal.clone();
                    updated_goal.status = "completed".to_string();
                    updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    let _ = self.state.update_goal(&updated_goal).await;

                    let assistant_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: ctx.session_id.to_string(),
                        role: "assistant".to_string(),
                        content: Some(response.clone()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.5,
                        embedding: None,
                    };
                    let _ = self
                        .append_assistant_message_with_event(
                            ctx.emitter,
                            &assistant_msg,
                            ctx.model,
                            None,
                            None,
                        )
                        .await;

                    self.emit_task_end(
                        ctx.emitter,
                        ctx.task_id,
                        TaskStatus::Completed,
                        ctx.task_start,
                        ctx.iteration,
                        0,
                        None,
                        Some(response.chars().take(200).collect()),
                    )
                    .await;
                    return Ok(consultant_direct_return_ok(response));
                }
                Err(e) => {
                    let mut updated_goal = goal.clone();
                    updated_goal.status = "failed".to_string();
                    let _ = self.state.update_goal(&updated_goal).await;
                    let err_reply = format!(
                        "I encountered an issue while working on your request: {}",
                        e
                    );

                    let assistant_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: ctx.session_id.to_string(),
                        role: "assistant".to_string(),
                        content: Some(err_reply.clone()),
                        tool_call_id: None,
                        tool_name: None,
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.5,
                        embedding: None,
                    };
                    let _ = self
                        .append_assistant_message_with_event(
                            ctx.emitter,
                            &assistant_msg,
                            ctx.model,
                            None,
                            None,
                        )
                        .await;

                    record_failed_task_tokens(ctx.task_tokens_used);
                    self.emit_task_end(
                        ctx.emitter,
                        ctx.task_id,
                        TaskStatus::Failed,
                        ctx.task_start,
                        ctx.iteration,
                        0,
                        Some(e.to_string()),
                        None,
                    )
                    .await;
                    return Ok(consultant_direct_return_ok(err_reply));
                }
            }
        }

        // Run progressive extraction on the Goal path so facts
        // and conversation summaries don't become stale when most
        // interactions route through Goals.
        let desc_preview: String = goal.description.chars().take(500).collect();
        let ellipsis = if goal.description.chars().count() > 500 {
            "..."
        } else {
            ""
        };
        let goal_response = format!(
            "On it. I'll plan this out and get started. Goal: {}{}",
            desc_preview, ellipsis
        );
        if self.context_window_config.progressive_facts
            && crate::memory::context_window::should_extract_facts(ctx.user_text)
        {
            let fast_model = ctx
                .llm_router
                .as_ref()
                .map(|r| r.select(crate::router::Tier::Fast).to_string())
                .unwrap_or_else(|| ctx.model.clone());
            crate::memory::context_window::spawn_progressive_extraction(
                ctx.llm_provider.clone(),
                fast_model.clone(),
                self.state.clone(),
                ctx.user_text.to_string(),
                goal_response.clone(),
                ctx.channel_ctx.channel_id.clone(),
                ctx.channel_ctx.visibility,
            );

            if self.context_window_config.enabled {
                crate::memory::context_window::spawn_incremental_summarization(
                    ctx.llm_provider.clone(),
                    fast_model,
                    self.state.clone(),
                    ctx.session_id.to_string(),
                    self.context_window_config.summarize_threshold,
                    self.context_window_config.summary_window,
                );
            }
        }

        // Persist the goal acknowledgment reply.
        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: ctx.session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(goal_response.clone()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        let _ = self
            .append_assistant_message_with_event(ctx.emitter, &assistant_msg, ctx.model, None, None)
            .await;

        // Return immediately — user doesn't wait for task lead.
        self.emit_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some("Goal created, working in background.".to_string()),
        )
        .await;
        Ok(consultant_direct_return_ok(goal_response))
    }

    pub(super) async fn route_consultant_complexity(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
        complexity: IntentComplexity,
    ) -> anyhow::Result<ConsultantPhaseOutcome> {
        match complexity {
            IntentComplexity::ScheduledMissingTiming => {
                self.handle_scheduled_missing_timing_intent(ctx).await
            }
            IntentComplexity::Scheduled {
                schedule_raw,
                schedule_cron,
                is_one_shot,
                schedule_type_explicit: _,
            } => {
                self.handle_scheduled_intent(ctx, schedule_raw, schedule_cron, is_one_shot)
                    .await
            }
            IntentComplexity::Knowledge => self.handle_knowledge_intent(ctx).await,
            IntentComplexity::Simple => self.handle_simple_intent(ctx).await,
            IntentComplexity::Complex => self.handle_complex_intent(ctx).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_scheduled_goal_description_empty_composed_returns_current() {
        let result = Agent::build_scheduled_goal_description("current task", "");
        assert_eq!(result, "current task");
    }

    #[test]
    fn build_scheduled_goal_description_plain_text_returns_composed() {
        let result =
            Agent::build_scheduled_goal_description("current task", "say hello every morning");
        assert_eq!(result, "say hello every morning");
    }

    #[test]
    fn build_scheduled_goal_description_original_request_only() {
        let composed = "Original request: send a daily weather update";
        let result = Agent::build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "send a daily weather update");
    }

    #[test]
    fn build_scheduled_goal_description_original_and_followup() {
        let composed =
            "Original request: check server health\nAssistant asked: which server?\nFollow-up: the production server";
        let result = Agent::build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "check server health | the production server");
    }

    #[test]
    fn build_scheduled_goal_description_deduplicates_followup() {
        let composed = "Original request: say hello\nFollow-up: say hello";
        let result = Agent::build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "say hello");
    }

    #[test]
    fn build_scheduled_goal_description_deduplicates_case_insensitive() {
        let composed = "Original request: Say Hello\nFollow-up: say hello";
        let result = Agent::build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "Say Hello");
    }

    #[test]
    fn build_scheduled_goal_description_fallback_strips_labels() {
        // Labels present but blocks are empty — should fall through to flattened path
        let composed = "Original request:\nAssistant asked:\nFollow-up:\nsome leftover text";
        let result = Agent::build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "some leftover text");
    }

    #[test]
    fn build_scheduled_goal_description_trims_whitespace() {
        let composed = "  Original request:   remind me to stretch   ";
        let result = Agent::build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "remind me to stretch");
    }
}
