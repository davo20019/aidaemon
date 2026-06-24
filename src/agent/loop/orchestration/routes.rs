use super::types::OrchestrationCtx;
use crate::agent::direct_return::direct_return_ok;
use crate::agent::fallthrough::fallthrough;
use crate::agent::recall_guardrails::{
    filter_tool_defs_for_delegation, filter_tool_defs_for_personal_memory,
    is_delegation_blocked_tool, is_personal_memory_tool,
};
use crate::agent::response_phase::ResponsePhaseOutcome;
use crate::agent::*;
use crate::events::TaskOutcome;

const MAX_SCHEDULE_SEGMENTS_PER_MESSAGE: usize = 10;

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

    let original = extract_labeled_block(
        composed,
        "Original request:",
        &["Assistant asked:", "Follow-up:"],
    );
    let followup = extract_labeled_block(composed, "Follow-up:", &[]);
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

fn looks_like_schedule_only_description(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if crate::cron_utils::parse_schedule(trimmed).is_ok() {
        return true;
    }
    if let Some((detected, _)) = crate::cron_utils::extract_schedule_from_text(trimmed) {
        return detected.trim().eq_ignore_ascii_case(trimmed);
    }
    false
}

async fn emit_direct_reply(
    agent: &Agent,
    ctx: &OrchestrationCtx<'_>,
    message: String,
    completion_note: &str,
) -> anyhow::Result<ResponsePhaseOutcome> {
    let assistant_msg = Message {
        id: Uuid::new_v4().to_string(),
        session_id: ctx.session_id.to_string(),
        role: "assistant".to_string(),
        content: Some(message.clone()),
        tool_call_id: None,
        tool_name: None,
        tool_calls_json: None,
        created_at: Utc::now(),
        importance: 0.5,
        ..Message::runtime_defaults()
    };
    agent
        .append_assistant_message_with_event(ctx.emitter, &assistant_msg, "system", None, None)
        .await?;
    agent
        .emit_direct_return_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some(completion_note.to_string()),
            true,
        )
        .await;
    Ok(direct_return_ok(message))
}

async fn confirm_scheduled_goal_activation(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
    goal: &Goal,
    schedule: &crate::traits::GoalSchedule,
    completion_note: &str,
) -> anyhow::Result<ResponsePhaseOutcome> {
    let activation_msg = match agent.state.activate_goal(&goal.id).await {
        Ok(true) => {
            if let Some(ref registry) = agent.goal_token_registry {
                registry.register(&goal.id).await;
            }
            let next_run = chrono::DateTime::parse_from_rfc3339(&schedule.next_run_at)
                .ok()
                .map(|dt| crate::cron_utils::humanize_run_time(dt.with_timezone(&chrono::Local)))
                .unwrap_or_else(|| "n/a".to_string());
            format!(
                "✅ Scheduled: {} — next run {}.",
                goal.description, next_run
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
        ..Message::runtime_defaults()
    };
    agent
        .append_assistant_message_with_event(ctx.emitter, &assistant_msg, "system", None, None)
        .await?;
    agent
        .emit_direct_return_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some(completion_note.to_string()),
            true,
        )
        .await;
    Ok(direct_return_ok(activation_msg))
}

async fn confirm_scheduled_goal_activation_batch(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
    goals_and_schedules: &[(Goal, crate::traits::GoalSchedule)],
    completion_note: &str,
) -> anyhow::Result<ResponsePhaseOutcome> {
    let mut activated = Vec::new();
    let mut activation_errors = Vec::new();

    for (goal, schedule) in goals_and_schedules {
        match agent.state.activate_goal(&goal.id).await {
            Ok(true) => {
                if let Some(ref registry) = agent.goal_token_registry {
                    registry.register(&goal.id).await;
                }
                let next_run = chrono::DateTime::parse_from_rfc3339(&schedule.next_run_at)
                    .ok()
                    .map(|dt| {
                        crate::cron_utils::humanize_run_time(dt.with_timezone(&chrono::Local))
                    })
                    .unwrap_or_else(|| "n/a".to_string());
                activated.push(format!("{} (next run {})", goal.description, next_run));
            }
            Ok(false) => {}
            Err(e) => activation_errors.push(e.to_string()),
        }
    }

    let activation_msg = if !activated.is_empty() && activation_errors.is_empty() {
        if activated.len() == 1 {
            format!("✅ Scheduled: {}.", activated[0])
        } else {
            format!(
                "✅ Scheduled {} goals:\n- {}",
                activated.len(),
                activated.join("\n- ")
            )
        }
    } else if !activated.is_empty() {
        format!(
            "Scheduled {} goals:\n- {}\nBut {} could not be activated: {}",
            activated.len(),
            activated.join("\n- "),
            activation_errors.len(),
            activation_errors.join("; ")
        )
    } else {
        format!(
            "I couldn't activate scheduled goals: {}",
            activation_errors.join("; ")
        )
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
        ..Message::runtime_defaults()
    };
    agent
        .append_assistant_message_with_event(ctx.emitter, &assistant_msg, "system", None, None)
        .await?;
    let outcome = if !activated.is_empty() {
        TaskOutcome::Succeeded
    } else {
        TaskOutcome::Failed
    };
    agent
        .emit_direct_return_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            outcome,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some(completion_note.to_string()),
            outcome == TaskOutcome::Succeeded,
        )
        .await;
    Ok(direct_return_ok(activation_msg))
}

async fn cancel_scheduled_goals_before_confirmation(
    agent: &Agent,
    goals: &[Goal],
) -> anyhow::Result<usize> {
    let mut cancelled = 0usize;
    for goal in goals {
        let now = chrono::Utc::now().to_rfc3339();
        let mut updated = goal.clone();
        updated.status = "cancelled".to_string();
        updated.completed_at = Some(now.clone());
        updated.updated_at = now;
        if agent.state.update_goal(&updated).await.is_ok() {
            cancelled += 1;
        }
        if let Ok(schedules) = agent.state.get_schedules_for_goal(&goal.id).await {
            for schedule in &schedules {
                let _ = agent.state.delete_goal_schedule(&schedule.id).await;
            }
        }
    }
    Ok(cancelled)
}

async fn ensure_orchestrator_tools_loaded(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
) -> anyhow::Result<()> {
    if !ctx.tool_defs.is_empty() || !ctx.tools_allowed_for_user {
        return Ok(());
    }

    let (mut defs, mut base_defs, mut caps) = agent
        .load_policy_tool_set(
            ctx.user_text,
            ctx.channel_ctx.visibility,
            &ctx.policy_bundle.policy,
            ctx.policy_bundle.risk_score,
            agent.policy_config.tool_filter_enforce,
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
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
) -> anyhow::Result<Option<ResponsePhaseOutcome>> {
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

    let active_goals = agent
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
        if let Some(ref registry) = agent.goal_token_registry {
            registry.cancel(&goal.id).await;
        }

        // Update goal DB status
        let mut updated = (*goal).clone();
        updated.status = "cancelled".to_string();
        updated.updated_at = chrono::Utc::now().to_rfc3339();
        let _ = agent.state.update_goal(&updated).await;

        // Best-effort cleanup: cancelled goals should not retain schedules.
        if let Ok(schedules) = agent.state.get_schedules_for_goal(&updated.id).await {
            for s in &schedules {
                let _ = agent.state.delete_goal_schedule(&s.id).await;
            }
        }

        // Cancel all remaining tasks for this goal
        if let Ok(tasks) = agent.state.get_tasks_for_goal(&goal.id).await {
            for task in &tasks {
                if task.status != "completed"
                    && task.status != "failed"
                    && task.status != "cancelled"
                {
                    let mut cancelled_task = task.clone();
                    cancelled_task.status = "cancelled".to_string();
                    let _ = agent.state.update_task(&cancelled_task).await;
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

    agent
        .emit_direct_return_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some(msg.clone()),
            true,
        )
        .await;

    Ok(Some(direct_return_ok(msg)))
}

async fn handle_scheduled_missing_timing_intent(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
) -> anyhow::Result<ResponsePhaseOutcome> {
    // Fall through to the full agent loop instead of
    // giving up. The LLM with tools can ask for timing
    // or infer it from context.
    info!(
        ctx.session_id,
        "ScheduledMissingTiming — falling through to agent loop"
    );
    ensure_orchestrator_tools_loaded(agent, ctx).await?;
    Ok(fallthrough())
}

async fn handle_scheduled_intent(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
    mut schedule_raw: String,
    is_one_shot: bool,
) -> anyhow::Result<ResponsePhaseOutcome> {
    if ctx.user_role != UserRole::Owner {
        // Non-owners cannot create scheduled goals — load tools and
        // fall through to agent loop so the request is handled directly.
        ensure_orchestrator_tools_loaded(agent, ctx).await?;
        ctx.pending_system_messages
            .push(SystemDirective::SchedulingOwnerOnly);
        return Ok(fallthrough());
    }

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
            ..Message::runtime_defaults()
        };
        agent
            .append_assistant_message_with_event(ctx.emitter, &assistant_msg, "system", None, None)
            .await?;
        agent
            .emit_direct_return_task_end(
                ctx.emitter,
                ctx.task_id,
                TaskStatus::Completed,
                TaskOutcome::Succeeded,
                ctx.task_start,
                ctx.iteration,
                0,
                None,
                Some(msg.chars().take(200).collect()),
                true,
            )
            .await;
        return Ok(direct_return_ok(msg));
    }

    let goal_user_text = ctx.turn_context.goal_user_text.clone();

    // Multi-schedule path: parse all segments before creating any DB rows.
    let extracted_segments = crate::cron_utils::extract_schedule_segments(ctx.user_text);
    if extracted_segments.len() > MAX_SCHEDULE_SEGMENTS_PER_MESSAGE {
        let msg = format!(
                "I can schedule up to {} goals per message. Please split this into smaller batches and try again.",
                MAX_SCHEDULE_SEGMENTS_PER_MESSAGE
            );
        return emit_direct_reply(
            agent,
            ctx,
            msg,
            "Rejected oversized multi-schedule request.",
        )
        .await;
    }
    if extracted_segments.len() > 1 {
        let mut prepared_segments: Vec<(
            String,
            String,
            String,
            bool,
            chrono::DateTime<chrono::Local>,
        )> = Vec::new();
        for segment in extracted_segments {
            let cron_expr = match crate::cron_utils::parse_schedule(&segment.schedule_raw) {
                Ok(expr) => expr,
                Err(e) => {
                    warn!(
                        ctx.session_id,
                        schedule_raw = %segment.schedule_raw,
                        error = %e,
                        "Multi-schedule parse failed — rejecting batch"
                    );
                    let msg = format!(
                            "I couldn't parse one of the schedules ({}). Please resend with valid schedules so I can create them together.",
                            segment.schedule_raw
                        );
                    return emit_direct_reply(
                        agent,
                        ctx,
                        msg,
                        "Rejected multi-schedule request with invalid segment.",
                    )
                    .await;
                }
            };
            let cron_looks_one_shot = crate::cron_utils::is_one_shot_schedule(&cron_expr);
            let actually_one_shot = cron_looks_one_shot || segment.is_one_shot;
            let next_run_local = match crate::cron_utils::compute_next_run_local(&cron_expr) {
                Ok(next) => next,
                Err(e) => {
                    warn!(
                        ctx.session_id,
                        schedule_raw = %segment.schedule_raw,
                        error = %e,
                        "Multi-schedule next-run computation failed — rejecting batch"
                    );
                    let msg = format!(
                            "I couldn't compute the next run for one schedule ({}). Please resend with valid schedules so I can create them together.",
                            segment.schedule_raw
                        );
                    return emit_direct_reply(
                        agent,
                        ctx,
                        msg,
                        "Rejected multi-schedule request with invalid segment.",
                    )
                    .await;
                }
            };
            prepared_segments.push((
                segment.description,
                segment.schedule_raw,
                cron_expr,
                actually_one_shot,
                next_run_local,
            ));
        }

        let goal_context = super::run::build_goal_feed_forward_context(
            agent,
            ctx.session_id,
            &goal_user_text,
            &ctx.turn_context.recent_messages,
            &ctx.turn_context.project_hints,
        )
        .await;

        let mut created = Vec::<(Goal, crate::traits::GoalSchedule, String, String)>::new();
        let mut created_goals_for_cleanup = Vec::<Goal>::new();
        for (description, segment_schedule_raw, cron_expr, actually_one_shot, next_run_local) in
            prepared_segments
        {
            let mut goal = if actually_one_shot {
                Goal::new_deferred_finite(&description, ctx.session_id)
            } else {
                Goal::new_continuous_pending(&description, ctx.session_id, None, None)
            };
            if let Some(ref context) = goal_context {
                goal.context = Some(context.clone());
            }

            if let Err(e) = agent.state.create_goal(&goal).await {
                let _ =
                    cancel_scheduled_goals_before_confirmation(agent, &created_goals_for_cleanup)
                        .await;
                return Err(e);
            }
            created_goals_for_cleanup.push(goal.clone());

            let now = chrono::Utc::now().to_rfc3339();
            let schedule = crate::traits::GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: cron_expr.clone(),
                tz: "local".to_string(),
                original_schedule: Some(segment_schedule_raw.clone()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: actually_one_shot,
                is_paused: false,
                last_run_at: None,
                next_run_at: next_run_local.with_timezone(&chrono::Utc).to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            };
            if let Err(e) = agent.state.create_goal_schedule(&schedule).await {
                let _ =
                    cancel_scheduled_goals_before_confirmation(agent, &created_goals_for_cleanup)
                        .await;
                return Err(e);
            }

            let schedule_kind = if actually_one_shot {
                "one-time".to_string()
            } else {
                "recurring".to_string()
            };
            let schedule_desc = if actually_one_shot {
                crate::cron_utils::humanize_run_time(next_run_local)
            } else {
                format!(
                    "{} (next: {})",
                    segment_schedule_raw,
                    crate::cron_utils::humanize_run_time(next_run_local)
                )
            };
            created.push((goal, schedule, schedule_kind, schedule_desc));
        }

        let tz_label = crate::cron_utils::system_timezone_display();
        let goals_and_schedules = created
            .iter()
            .map(|(goal, schedule, _, _)| (goal.clone(), schedule.clone()))
            .collect::<Vec<_>>();

        let already_approved = {
            match tokio::time::timeout(
                Duration::from_secs(2),
                agent.schedule_approved_sessions.read(),
            )
            .await
            {
                Ok(approved) => approved.contains(ctx.session_id),
                Err(_) => {
                    warn!(
                        ctx.session_id,
                        "Timed out acquiring schedule_approved_sessions lock"
                    );
                    false
                }
            }
        };
        if already_approved {
            return confirm_scheduled_goal_activation_batch(
                agent,
                ctx,
                &goals_and_schedules,
                "Scheduled goals auto-confirmed from prior session approval.",
            )
            .await;
        }

        let inline_confirmation = {
            let hub_weak =
                match tokio::time::timeout(Duration::from_secs(2), agent.hub.read()).await {
                    Ok(guard) => guard.clone(),
                    Err(_) => {
                        warn!(
                            ctx.session_id,
                            "Timed out acquiring hub lock for inline schedule confirmation"
                        );
                        None
                    }
                };
            if let Some(hub_weak) = hub_weak {
                if let Some(hub_arc) = hub_weak.upgrade() {
                    let confirmation_desc =
                        format!("Confirm {} scheduled goals", goals_and_schedules.len());
                    let mut details = created
                        .iter()
                        .enumerate()
                        .map(|(idx, (goal, _, kind, schedule_desc))| {
                            format!(
                                "{}. [{}] {} ({})",
                                idx + 1,
                                kind,
                                goal.description,
                                schedule_desc
                            )
                        })
                        .collect::<Vec<_>>();
                    details.push(format!("System timezone: {}", tz_label));
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
                    return confirm_scheduled_goal_activation_batch(
                        agent,
                        ctx,
                        &goals_and_schedules,
                        "Scheduled goals confirmed via inline approval.",
                    )
                    .await;
                }
                Ok(false) => {
                    let goals = created
                        .iter()
                        .map(|(goal, _, _, _)| goal.clone())
                        .collect::<Vec<_>>();
                    let cancelled = cancel_scheduled_goals_before_confirmation(agent, &goals)
                        .await
                        .unwrap_or(0);
                    let cancel_msg = if cancelled == 1 {
                        "OK, cancelled the scheduled goal.".to_string()
                    } else {
                        format!(
                            "OK, cancelled {} scheduled goals.",
                            cancelled.max(goals.len())
                        )
                    };
                    return emit_direct_reply(
                        agent,
                        ctx,
                        cancel_msg,
                        "Scheduled goals cancelled via inline approval.",
                    )
                    .await;
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

        let summary_lines = created
            .iter()
            .enumerate()
            .map(|(idx, (goal, _, kind, schedule_desc))| {
                format!(
                    "{}. [{}] {} ({})",
                    idx + 1,
                    kind,
                    goal.description,
                    schedule_desc
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let confirmation = format!(
                "I'll schedule these {} goals:\n{}\nSystem timezone: {}.\nReply **confirm** to proceed or **cancel** to discard.",
                created.len(),
                summary_lines,
                tz_label
            );
        return emit_direct_reply(
            agent,
            ctx,
            confirmation,
            "Scheduled goals awaiting text confirmation.",
        )
        .await;
    }

    let mut cron_expr = crate::cron_utils::parse_schedule(&schedule_raw).ok();

    // E2E guardrail: if the model provided a malformed schedule string
    // (e.g. "2 minutes" instead of "in 2 minutes"), retry parsing using
    // heuristic extraction from the user text.
    if cron_expr.is_none() && ENABLE_SCHEDULE_HEURISTICS {
        if let Some((heuristic_raw, _)) = intent_routing::detect_schedule_heuristic(ctx.user_text) {
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
            warn!(
                ctx.session_id,
                schedule_raw = %schedule_raw,
                "Schedule parse failed — falling through to agent loop"
            );
            ensure_orchestrator_tools_loaded(agent, ctx).await?;
            return Ok(fallthrough());
        }
    };

    let cron_looks_one_shot = crate::cron_utils::is_one_shot_schedule(&cron_expr);
    let actually_one_shot = cron_looks_one_shot || is_one_shot;

    let current_turn_description =
        crate::cron_utils::clean_task_description(ctx.user_text, &schedule_raw);
    let mut goal_description = if !looks_like_schedule_only_description(&current_turn_description) {
        current_turn_description
    } else {
        let composed = build_scheduled_goal_description(ctx.user_text, &goal_user_text);
        let cleaned_composed = crate::cron_utils::clean_task_description(&composed, &schedule_raw);
        if looks_like_schedule_only_description(&cleaned_composed) {
            composed
        } else {
            cleaned_composed
        }
    };
    if goal_description.trim().is_empty() {
        goal_description = ctx.user_text.trim().to_string();
    }

    // Reminder-shaped requests: clean_task_description may strip the leading
    // "remind me to", leaving a bare imperative ("Check logs"). Restore the
    // canonical reminder form so the auto-confirm fast path below and the
    // fire-time fast path in heartbeat both recognize it.
    let user_text_lower = ctx.user_text.to_lowercase();
    if crate::reminders::parse_reminder(&goal_description).is_none()
        && (intent_routing::contains_keyword_as_words(&user_text_lower, "remind me")
            || intent_routing::contains_keyword_as_words(&user_text_lower, "remind us"))
        && !looks_like_schedule_only_description(&goal_description)
    {
        goal_description = crate::reminders::canonical_description(&goal_description);
    }

    // Duplicate detection: check for existing goals with matching description + schedule.
    let target_desc_canonical = goal_description
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let target_cron = cron_expr.trim().to_ascii_lowercase();
    if let Ok(existing_goals) = agent.state.get_scheduled_goals().await {
        for existing in &existing_goals {
            if existing.session_id != ctx.session_id {
                continue;
            }
            if !matches!(
                existing.status.as_str(),
                "active" | "pending_confirmation" | "paused"
            ) {
                continue;
            }
            let existing_desc = existing
                .description
                .trim()
                .to_ascii_lowercase()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if existing_desc != target_desc_canonical {
                continue;
            }
            if let Ok(schedules) = agent.state.get_schedules_for_goal(&existing.id).await {
                let has_matching_cron = schedules
                    .iter()
                    .any(|s| s.cron_expr.trim().to_ascii_lowercase() == target_cron);
                if has_matching_cron {
                    let msg = format!(
                            "A similar scheduled goal already exists ({}). Use \"list my scheduled goals\" to inspect existing goals.",
                            &existing.id[..8]
                        );
                    return emit_direct_reply(
                        agent,
                        ctx,
                        msg,
                        "Duplicate scheduled goal detected in fast-path.",
                    )
                    .await;
                }
            }
        }
    }

    let mut goal = if actually_one_shot {
        Goal::new_deferred_finite(&goal_description, ctx.session_id)
    } else {
        Goal::new_continuous_pending(&goal_description, ctx.session_id, None, None)
    };

    if let Some(goal_context) = super::run::build_goal_feed_forward_context(
        agent,
        ctx.session_id,
        &goal_user_text,
        &ctx.turn_context.recent_messages,
        &ctx.turn_context.project_hints,
    )
    .await
    {
        goal.context = Some(goal_context);
    }

    agent.state.create_goal(&goal).await?;

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
    agent.state.create_goal_schedule(&schedule).await?;

    // Plain one-shot reminders are low-risk — the only action at fire time is
    // a message back to the user — so skip the approval gate entirely and
    // confirm in a single friendly line. The user can still say "cancel" to
    // remove it.
    if actually_one_shot {
        if let Some(reminder) = crate::reminders::parse_reminder(&goal.description) {
            if let Ok(true) = agent.state.activate_goal(&goal.id).await {
                if let Some(ref registry) = agent.goal_token_registry {
                    registry.register(&goal.id).await;
                }
                let when = crate::cron_utils::humanize_run_time(next_run_local);
                let msg = crate::reminders::confirmation_message(&reminder, &when);
                return emit_direct_reply(
                    agent,
                    ctx,
                    msg,
                    "Plain one-shot reminder auto-confirmed (low-risk).",
                )
                .await;
            }
        }
    }

    let tz_label = crate::cron_utils::system_timezone_display();
    let schedule_desc = if actually_one_shot {
        crate::cron_utils::humanize_run_time(next_run_local)
    } else {
        let next_local = crate::cron_utils::humanize_run_time(next_run_local);
        format!("{} (next: {})", schedule_raw, next_local)
    };
    let schedule_kind = if actually_one_shot {
        "one-time"
    } else {
        "recurring"
    };

    let already_approved = {
        match tokio::time::timeout(
            Duration::from_secs(2),
            agent.schedule_approved_sessions.read(),
        )
        .await
        {
            Ok(approved) => approved.contains(ctx.session_id),
            Err(_) => {
                warn!(
                    ctx.session_id,
                    "Timed out acquiring schedule_approved_sessions lock"
                );
                false
            }
        }
    };
    if already_approved {
        return confirm_scheduled_goal_activation(
            agent,
            ctx,
            &goal,
            &schedule,
            "Scheduled goal auto-confirmed from prior session approval.",
        )
        .await;
    }

    // Prefer inline goal confirmation buttons for schedule confirmation
    // (Telegram/Discord/Slack). Shows Confirm ✅ / Cancel ❌ buttons.
    // Non-inline channels keep the existing text confirm/cancel fallback.
    let inline_confirmation = {
        let hub_weak = match tokio::time::timeout(Duration::from_secs(2), agent.hub.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!(
                    ctx.session_id,
                    "Timed out acquiring hub lock for inline goal confirmation"
                );
                None
            }
        };
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
                return confirm_scheduled_goal_activation(
                    agent,
                    ctx,
                    &goal,
                    &schedule,
                    "Scheduled goal confirmed via inline approval.",
                )
                .await;
            }
            Ok(false) => {
                let _ = cancel_scheduled_goals_before_confirmation(agent, &[goal.clone()]).await;

                let cancel_msg = "OK, cancelled the scheduled goal.".to_string();
                return emit_direct_reply(
                    agent,
                    ctx,
                    cancel_msg,
                    "Scheduled goal cancelled via inline approval.",
                )
                .await;
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

    emit_direct_reply(
        agent,
        ctx,
        confirmation,
        "Scheduled goal awaiting text confirmation.",
    )
    .await
}

async fn handle_simple_intent(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
) -> anyhow::Result<ResponsePhaseOutcome> {
    ensure_orchestrator_tools_loaded(agent, ctx).await?;
    if !ctx.tool_defs.is_empty() {
        info!(
            ctx.session_id,
            tool_count = ctx.tool_defs.len(),
            "Simple intent — loaded tools for orchestrator"
        );
    }

    // Inject plan suggestion when heuristics detect a multi-step task.
    let plan_trigger = crate::plans::detection::should_create_plan(ctx.user_text);
    if let Some(hint) = crate::plans::detection::get_plan_suggestion_prompt(&plan_trigger) {
        info!(
            ctx.session_id,
            reason = ?plan_trigger.reason(),
            "Plan detection triggered for simple intent"
        );
        ctx.pending_system_messages
            .push(SystemDirective::PlanSuggestion { hint });
        // Telemetry: this turn is plan-worthy, so the model is expected to call
        // track_requirements. The denominator for checklist capture rate (join
        // to a later checklist_nudge/checklist_complete event by task_id).
        agent
            .emit_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                crate::events::DecisionType::GateTelemetry,
                "checklist expected (plan-worthy turn)",
                serde_json::json!({ "event": "checklist_expected" }),
            )
            .await;
    }

    info!(
        ctx.session_id,
        "Simple intent — continuing to full agent loop"
    );

    // Skip to next iteration where the full agent loop
    // runs with all tools and full context.
    Ok(fallthrough())
}

async fn handle_complex_intent(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
) -> anyhow::Result<ResponsePhaseOutcome> {
    if ctx.user_role != UserRole::Owner {
        // Non-owners cannot create complex goals — load tools and
        // fall through to agent loop so the request is handled directly.
        ensure_orchestrator_tools_loaded(agent, ctx).await?;
        let cli_agent_in_defs = ctx.tool_defs.iter().any(|def| {
            def.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                == Some("cli_agent")
        });
        if agent.has_cli_agents_available() && cli_agent_in_defs {
            *ctx.tool_defs = filter_tool_defs_for_delegation(ctx.tool_defs);
            *ctx.base_tool_defs = filter_tool_defs_for_delegation(ctx.base_tool_defs);
            ctx.available_capabilities
                .retain(|name, _| !is_delegation_blocked_tool(name));
            ctx.pending_system_messages
                .push(SystemDirective::DelegationModeActive);
            info!(
                ctx.session_id,
                tool_count = ctx.tool_defs.len(),
                "Complex non-owner request: filtered competing execution tools for delegation mode"
            );
        }
        // Inject plan suggestion for non-owner complex intents going through direct loop.
        let plan_trigger = crate::plans::detection::should_create_plan(ctx.user_text);
        if let Some(hint) = crate::plans::detection::get_plan_suggestion_prompt(&plan_trigger) {
            ctx.pending_system_messages
                .push(SystemDirective::PlanSuggestion { hint });
            agent
                .emit_decision_point(
                    ctx.emitter,
                    ctx.task_id,
                    ctx.iteration,
                    crate::events::DecisionType::GateTelemetry,
                    "checklist expected (plan-worthy turn)",
                    serde_json::json!({ "event": "checklist_expected" }),
                )
                .await;
        }
        ctx.pending_system_messages
            .push(SystemDirective::GoalCreationOwnerOnly);
        return Ok(fallthrough());
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
            ..Message::runtime_defaults()
        };
        agent
            .append_assistant_message_with_event(ctx.emitter, &assistant_msg, "system", None, None)
            .await?;
        agent
            .emit_direct_return_task_end(
                ctx.emitter,
                ctx.task_id,
                TaskStatus::Completed,
                TaskOutcome::Succeeded,
                ctx.task_start,
                ctx.iteration,
                0,
                None,
                Some(msg.chars().take(200).collect()),
                true,
            )
            .await;
        return Ok(direct_return_ok(msg));
    }

    // Create goal
    let goal_user_text = ctx.turn_context.goal_user_text.clone();
    let mut goal = Goal::new_finite(&goal_user_text, ctx.session_id);

    // Feed-forward relevant knowledge into goal context.
    if let Some(goal_context) = super::run::build_goal_feed_forward_context(
        agent,
        ctx.session_id,
        &goal_user_text,
        &ctx.turn_context.recent_messages,
        &ctx.turn_context.project_hints,
    )
    .await
    {
        goal.context = Some(goal_context);
    }

    agent.state.create_goal(&goal).await?;

    // Register cancellation token for this goal.
    if let Some(ref registry) = agent.goal_token_registry {
        registry.register(&goal.id).await;
    }

    info!(
        ctx.session_id,
        goal_id = %goal.id,
        "Created goal for complex request, spawning task lead in background"
    );

    // Upgrade weak self-reference to Arc for background spawning.
    let self_arc = {
        match tokio::time::timeout(Duration::from_secs(2), agent.self_ref.read()).await {
            Ok(self_ref) => self_ref.as_ref().and_then(|w| w.upgrade()),
            Err(_) => {
                warn!(
                    ctx.session_id,
                    "Timed out acquiring self_ref lock for background task-lead spawn"
                );
                None
            }
        }
    };

    if let Some(agent_arc) = self_arc {
        // Spawn the task lead in the background — user gets immediate response.
        let bg_hub = match tokio::time::timeout(Duration::from_secs(2), agent.hub.read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!(
                    ctx.session_id,
                    "Timed out acquiring hub lock for background task-lead spawn"
                );
                None
            }
        };
        spawn_background_task_lead(
            agent_arc,
            goal.clone(),
            goal_user_text.clone(),
            ctx.session_id.to_string(),
            ctx.channel_ctx.clone(),
            ctx.user_role,
            agent.state.clone(),
            bg_hub,
            agent.goal_token_registry.clone(),
            None,
        );
    } else {
        // No self_ref available (sub-agent or test) — fall back to sync.
        warn!("No self_ref available, running task lead synchronously");
        let result = agent
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
                let tasks = agent
                    .state
                    .get_tasks_for_goal(&goal.id)
                    .await
                    .unwrap_or_default();
                let tasks_fully_done = !tasks.is_empty()
                    && tasks
                        .iter()
                        .all(|task| matches!(task.status.as_str(), "completed" | "skipped"));
                let should_auto_complete = tasks_fully_done
                    && !goal_completion_response_indicates_incomplete_work(&response);

                if should_auto_complete {
                    let mut updated_goal = goal.clone();
                    updated_goal.status = "completed".to_string();
                    updated_goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    let _ = agent.state.update_goal(&updated_goal).await;
                }

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
                    ..Message::runtime_defaults()
                };
                let _ = agent
                    .append_assistant_message_with_event(
                        ctx.emitter,
                        &assistant_msg,
                        ctx.model,
                        None,
                        None,
                    )
                    .await;

                let task_outcome = if response_has_user_value(&response, 0) {
                    TaskOutcome::Succeeded
                } else {
                    TaskOutcome::Failed
                };
                agent
                    .emit_direct_return_task_end(
                        ctx.emitter,
                        ctx.task_id,
                        TaskStatus::Completed,
                        task_outcome,
                        ctx.task_start,
                        ctx.iteration,
                        0,
                        None,
                        Some(response.chars().take(200).collect()),
                        task_outcome == TaskOutcome::Succeeded,
                    )
                    .await;
                return Ok(direct_return_ok(response));
            }
            Err(e) => {
                let mut updated_goal = goal.clone();
                updated_goal.status = "failed".to_string();
                let _ = agent.state.update_goal(&updated_goal).await;
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
                    ..Message::runtime_defaults()
                };
                let _ = agent
                    .append_assistant_message_with_event(
                        ctx.emitter,
                        &assistant_msg,
                        ctx.model,
                        None,
                        None,
                    )
                    .await;

                record_failed_task_tokens(ctx.task_tokens_used);
                agent
                    .emit_direct_return_task_end(
                        ctx.emitter,
                        ctx.task_id,
                        TaskStatus::Failed,
                        TaskOutcome::Failed,
                        ctx.task_start,
                        ctx.iteration,
                        0,
                        Some(e.to_string()),
                        None,
                        false,
                    )
                    .await;
                return Ok(direct_return_ok(err_reply));
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
    if agent.context_window_config.progressive_facts
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
            agent.state.clone(),
            agent.event_store.clone(),
            ctx.user_text.to_string(),
            goal_response.clone(),
            ctx.channel_ctx.channel_id.clone(),
            ctx.channel_ctx.visibility,
            ctx.user_role,
        );

        if agent.context_window_config.enabled {
            crate::memory::context_window::spawn_incremental_summarization(
                ctx.llm_provider.clone(),
                fast_model,
                agent.state.clone(),
                agent.event_store.clone(),
                ctx.session_id.to_string(),
                agent.context_window_config.summarize_threshold,
                agent.context_window_config.summary_window,
                ctx.user_role,
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
        ..Message::runtime_defaults()
    };
    let _ = agent
        .append_assistant_message_with_event(ctx.emitter, &assistant_msg, ctx.model, None, None)
        .await;

    // Return immediately — user doesn't wait for task lead.
    agent
        .emit_direct_return_task_end(
            ctx.emitter,
            ctx.task_id,
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            ctx.task_start,
            ctx.iteration,
            0,
            None,
            Some("Goal created, working in background.".to_string()),
            true,
        )
        .await;
    Ok(direct_return_ok(goal_response))
}

pub(super) async fn route_orchestration_complexity(
    agent: &Agent,
    ctx: &mut OrchestrationCtx<'_>,
    complexity: IntentComplexity,
) -> anyhow::Result<ResponsePhaseOutcome> {
    match complexity {
        IntentComplexity::ScheduledMissingTiming => {
            handle_scheduled_missing_timing_intent(agent, ctx).await
        }
        IntentComplexity::Scheduled {
            schedule_raw,
            is_one_shot,
        } => handle_scheduled_intent(agent, ctx, schedule_raw, is_one_shot).await,
        IntentComplexity::Simple => handle_simple_intent(agent, ctx).await,
        IntentComplexity::Complex => handle_complex_intent(agent, ctx).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::recall_guardrails::filter_tool_defs_for_delegation;
    use serde_json::json;

    #[test]
    fn build_scheduled_goal_description_empty_composed_returns_current() {
        let result = build_scheduled_goal_description("current task", "");
        assert_eq!(result, "current task");
    }

    #[test]
    fn build_scheduled_goal_description_plain_text_returns_composed() {
        let result = build_scheduled_goal_description("current task", "say hello every morning");
        assert_eq!(result, "say hello every morning");
    }

    #[test]
    fn build_scheduled_goal_description_original_request_only() {
        let composed = "Original request: send a daily weather update";
        let result = build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "send a daily weather update");
    }

    #[test]
    fn build_scheduled_goal_description_original_and_followup() {
        let composed =
            "Original request: check server health\nAssistant asked: which server?\nFollow-up: the production server";
        let result = build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "check server health | the production server");
    }

    #[test]
    fn build_scheduled_goal_description_deduplicates_followup() {
        let composed = "Original request: say hello\nFollow-up: say hello";
        let result = build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "say hello");
    }

    #[test]
    fn build_scheduled_goal_description_deduplicates_case_insensitive() {
        let composed = "Original request: Say Hello\nFollow-up: say hello";
        let result = build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "Say Hello");
    }

    #[test]
    fn build_scheduled_goal_description_fallback_strips_labels() {
        // Labels present but blocks are empty — should fall through to flattened path
        let composed = "Original request:\nAssistant asked:\nFollow-up:\nsome leftover text";
        let result = build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "some leftover text");
    }

    #[test]
    fn build_scheduled_goal_description_trims_whitespace() {
        let composed = "  Original request:   remind me to stretch   ";
        let result = build_scheduled_goal_description("ignored", composed);
        assert_eq!(result, "remind me to stretch");
    }

    #[test]
    fn delegation_filter_requires_cli_agent_in_defs() {
        let defs = [
            json!({"function":{"name":"terminal"}}),
            json!({"function":{"name":"run_command"}}),
            json!({"function":{"name":"web_search"}}),
        ];
        let cli_agent_in_defs = defs.iter().any(|d| {
            d.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                == Some("cli_agent")
        });
        assert!(!cli_agent_in_defs);
    }

    #[test]
    fn delegation_filter_applies_when_cli_agent_present() {
        let defs = vec![
            json!({"function":{"name":"terminal"}}),
            json!({"function":{"name":"cli_agent"}}),
            json!({"function":{"name":"run_command"}}),
            json!({"function":{"name":"web_search"}}),
            json!({"function":{"name":"browser"}}),
        ];
        let filtered = filter_tool_defs_for_delegation(&defs);
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|d| d.get("function"))
            .filter_map(|f| f.get("name"))
            .filter_map(|n| n.as_str())
            .collect();
        assert!(names.contains(&"cli_agent"));
        assert!(names.contains(&"web_search"));
        assert!(!names.contains(&"terminal"));
        assert!(!names.contains(&"browser"));
        assert!(!names.contains(&"run_command"));
    }
}
