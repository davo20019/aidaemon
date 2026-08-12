use crate::agent::*;
use crate::events::TaskOutcome;

#[allow(clippy::too_many_arguments)]
pub(super) async fn maybe_handle_stop_command(
    agent: &Agent,
    session_id: &str,
    user_text: &str,
    user_role: UserRole,
    channel_ctx: &ChannelContext,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    task_id: &str,
    emitter: &crate::events::EventEmitter,
) -> anyhow::Result<Option<String>> {
    let lower_trimmed = user_text.trim().to_ascii_lowercase();
    let is_stop_command = matches!(lower_trimmed.as_str(), "stop" | "cancel" | "abort");
    if !is_stop_command {
        return Ok(None);
    }

    let early_task_start = Instant::now();
    if user_role != UserRole::Owner {
        let reply = "Only the owner can cancel running work in this session.";
        let reply = emit_bootstrap_direct_reply(
            agent,
            emitter,
            task_id,
            session_id,
            early_task_start,
            reply,
        )
        .await?;
        return Ok(Some(reply));
    }

    let cancel_result = agent
        .execute_tool_with_watchdog(
            "cli_agent",
            r#"{"action": "cancel_all"}"#,
            &tool_exec::ToolExecCtx {
                session_id,
                task_id: Some(task_id),
                status_tx,
                channel_visibility: channel_ctx.visibility,
                channel_id: channel_ctx.channel_id.as_deref(),
                project_scope: None,
                trusted: channel_ctx.trusted,
                user_role,
                workspace_grant: channel_ctx.active_workspace_grant(user_role),
                correction_preapproved: false,
                suppress_trusted_session: false,
                mandate_authority: None,
                mandate_tool_call_id: None,
            },
        )
        .await;
    let cli_cancel_msg = cancel_result.ok();

    // Cancel any active goals for this session as well (background task leads/executors).
    let cancelled_goals = agent.cancel_active_goals_for_session(session_id).await;

    let cli_cancelled_any = cli_cancel_msg
        .as_deref()
        .is_some_and(|m| !m.contains("No running CLI agents"));

    let reply = if cli_cancelled_any || !cancelled_goals.is_empty() {
        let mut reply = String::new();
        if cli_cancelled_any {
            reply.push_str(cli_cancel_msg.as_deref().unwrap_or_default());
        }
        if !cancelled_goals.is_empty() {
            if !reply.is_empty() {
                reply.push('\n');
                reply.push('\n');
            }
            if cancelled_goals.len() == 1 {
                reply.push_str(&format!("cancelled goal: {}", cancelled_goals[0]));
            } else {
                reply.push_str(&format!(
                    "cancelled {} goals:\n{}",
                    cancelled_goals.len(),
                    cancelled_goals
                        .iter()
                        .map(|d| format!("- {}", d))
                        .collect::<Vec<_>>()
                        .join("\n")
                ));
            }
        }
        info!(session_id, "Cancelled work on stop command");
        reply
    } else {
        "No running task to cancel.".to_string()
    };

    let reply = emit_bootstrap_direct_reply(
        agent,
        emitter,
        task_id,
        session_id,
        early_task_start,
        &reply,
    )
    .await?;
    Ok(Some(reply))
}

pub(super) async fn emit_bootstrap_direct_reply(
    agent: &Agent,
    emitter: &crate::events::EventEmitter,
    task_id: &str,
    session_id: &str,
    task_start: Instant,
    reply: &str,
) -> anyhow::Result<String> {
    let reply_text = reply.to_string();
    let assistant_msg = Message {
        id: Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        role: "assistant".to_string(),
        content: Some(reply_text.clone()),
        tool_call_id: None,
        tool_name: None,
        tool_calls_json: None,
        created_at: Utc::now(),
        importance: 0.5,
        ..Message::runtime_defaults()
    };
    agent
        .append_assistant_message_with_event(emitter, &assistant_msg, "system", None, None)
        .await?;

    if agent.harness_eval_enabled() {
        let turn_id = agent.current_turn_ids.read().await.get(session_id).cloned();
        let mut harness_eval = HarnessEvalAccumulator::new(HarnessEvalSeed {
            task_id: task_id.to_string(),
            turn_id,
            depth: agent.depth as u32,
            parent_task_id: agent.task_id.clone(),
            goal_id: agent.goal_id.clone(),
            durable_task_id: agent.task_id.clone(),
            completion_task_kind: "conversational".to_string(),
            followup_mode: None,
            config: agent.harness_eval_config.clone(),
        });
        harness_eval.record_bootstrap("bootstrap_direct_return", vec![], None, false);
        agent.install_harness_eval(harness_eval).await;
    }

    agent
        .emit_decision_point(
            emitter,
            task_id,
            0,
            DecisionType::GateTelemetry,
            "Bootstrap direct-return gate answered without LLM loop".to_string(),
            json!({
                "condition": "bootstrap_direct_return",
                "gate_family": "bootstrap",
                "action": "returned",
                "reply_chars": reply_text.trim().chars().count(),
                "agent_depth": agent.depth,
            }),
        )
        .await;

    agent
        .emit_direct_return_task_end(
            emitter,
            task_id,
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            task_start,
            0,
            0,
            None,
            Some(reply_text.chars().take(200).collect()),
            true,
        )
        .await;

    Ok(reply_text)
}
