//! Background completion delivery, turn-group joining, and coordinator
//! re-engagement for finished CLI agent runs.

use super::*;

/// What the model is told of a waited run's end that the owner already sees:
/// the run is drawn as its agent's message, with its report and where its
/// record disagrees. With nothing to disagree with and nothing else to do,
/// the model's turn may end without a reply of its own
/// ([`crate::run_state::RunStates::agents_answered`]).
pub(super) fn shown_note(run: &crate::run_state::RunState) -> String {
    use aidaemon_channels::turns::RunMismatch;
    let agent = run.agent.as_deref().unwrap_or("The agent");
    let Some(result) = &run.result else {
        return format!("[The owner sees this run as {agent}'s message.]");
    };
    if result.claims.is_none() {
        return format!(
            "[The owner sees this run as {agent}'s message. It reported in text only: \
             nothing in it was checked against the run's record.]"
        );
    }
    if result.mismatches.is_empty() {
        return format!(
            "[The owner sees this run as {agent}'s message, with its report. The run's record \
             bears out every claim in it.]"
        );
    }
    let disagrees: Vec<String> = result
        .mismatches
        .iter()
        .map(|mismatch| match mismatch {
            RunMismatch::ChangeNotSeen { path } => {
                format!("it says it changed {path}; no change seen")
            }
            RunMismatch::ChangeNotClaimed { path } => {
                format!("it changed {path} without saying so")
            }
            RunMismatch::CheckNotRun { command } => {
                format!("it says it ran `{command}`; no run seen")
            }
            RunMismatch::CheckFailed { command } => {
                format!("it says `{command}` passed; it failed")
            }
        })
        .collect();
    format!(
        "[The owner sees this run as {agent}'s message, with its report. The run's record \
         disagrees with it: {}.]",
        disagrees.join("; ")
    )
}

/// How a background run ended, for its chat ([`deliver_end`]).
pub(super) struct EndDelivery<'a> {
    pub(super) phase: &'a crate::run_state::RunEnd,
    pub(super) evidence: &'a crate::run_state::RunEvidence,
    pub(super) success: bool,
    /// The agent's own report.
    pub(super) report: &'a str,
    /// Its report with what it left running: what the parent reads.
    pub(super) persisted_output: &'a str,
    pub(super) left: &'a left_running::LeftRunning,
    pub(super) diff_dir: Option<&'a super::prompt::DiffBase>,
    pub(super) max_output: usize,
    /// Whether a run that ends alone (in no group of its turn's runs)
    /// answers its chat. A queued run stopped before it started does not:
    /// nothing ran, and whoever stopped it knows.
    pub(super) answer_alone: bool,
}

/// Deliver a background run's end: settle its card, enter its result in
/// the chat's history, and answer the chat (re-engaging the coordinator)
/// once the last run of its turn's group ended.
pub(super) async fn deliver_end(
    handles: &RunHandles,
    call: &RunCall,
    shared: &RunShared,
    end: EndDelivery<'_>,
) {
    let run = shared.run_slot.get().cloned();
    let phase = end.phase.clone();
    let success = end.success;
    let status_word = if success { "completed" } else { "failed" };
    let task_id_for_notify = &call.task_id;
    let notify_session_id = call.session_id.clone();
    let task_context = &call.label;
    let persisted_output = end.persisted_output;
    let ended = match run.as_ref() {
        Some(run) => Some(run.finish(phase, end.evidence.clone()).await),
        None => None,
    };
    handles.claims_changed.notify_waiters();

    // Child peer runs spawned via the MCP coordination server settle their run
    // card above and hand their output back to their spawning peer via
    // `PeerTree`, never posting standalone replies into the owner's chat.
    if call.is_peer_run {
        return;
    }
    // Its conversation was erased after it began (`/wipe`, deleted): its
    // end has nowhere to go, and posting it would bring part of it back.
    if handles.runs.erased_after(&call.session_id, call.began) {
        info!(
            task_id = %call.task_id,
            session_id = %call.session_id,
            "A run of an erased conversation ended: its result is not posted"
        );
        return;
    }

    let direct = !notify_session_id.trim().is_empty()
        && crate::session_model::get(handles.state.as_ref(), &notify_session_id)
            .await
            .agent
            .is_some();
    let notification_summary: String = if direct {
        end.report.to_string()
    } else {
        end.report.chars().take(500).collect()
    };
    let diff_section = if direct {
        None
    } else if let Some(base) = end.diff_dir {
        CliAgentTool::capture_git_diff(&handles.backend, base)
            .await
            .map(|diff| format!("\n\n{}", diff))
    } else {
        None
    };
    let body = format!(
        "{}{}{}",
        notification_summary.trim(),
        diff_section.unwrap_or_default(),
        end.left
            .owner_line()
            .map(|line| format!("\n\n{line}"))
            .unwrap_or_default()
    );
    let notification_type = if success { "completed" } else { "failed" };

    let continuity_output = format!(
        "Background CLI task {status_word}.\nTask ID: {task_id_for_notify}\nTask: {task_context}\nResult:\n{persisted_output}"
    );
    let continuity_message = Message {
        content: Some(aidaemon_tools::sanitize::wrap_untrusted_output(
            "cli_agent",
            &continuity_output,
        )),
        importance: 0.8,
        annotations: vec![
            aidaemon_events::conversation::MessageAnnotation::InternalContinuation,
            aidaemon_events::conversation::MessageAnnotation::WrappedUntrustedExternalData {
                source_tool: Some("cli_agent".to_string()),
            },
        ],
        ..Message::new_runtime(
            Uuid::new_v4().to_string(),
            notify_session_id.clone(),
            "assistant",
        )
    };
    if let Err(e) = handles.state.append_message(&continuity_message).await {
        warn!(
            task_id = %task_id_for_notify,
            session_id = %notify_session_id,
            error = %e,
            "cli_agent background completion could not enter session history"
        );
    }

    let group = match (run.as_ref(), ended.as_ref()) {
        (Some(run), Some((Some(state), on_card))) => run.end_in_group(
            state,
            crate::run_state::RunReport {
                text: persisted_output.to_string(),
                untrusted: true,
            },
            run.result(Some(state), &body, *on_card),
            end.max_output,
        ),
        _ => crate::background_run::RunsTurn::Alone,
    };
    let group_answer = match group {
        crate::background_run::RunsTurn::Waiting => {
            info!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                "A CLI agent's result waits for the rest of its turn's runs"
            );
            return;
        }
        crate::background_run::RunsTurn::Stopped => return,
        crate::background_run::RunsTurn::Answer(answer) => Some(answer),
        crate::background_run::RunsTurn::Alone
            if !end.answer_alone
                || call.owner_follow_up
                || end.phase.stopped_by_owner()
                || ended
                    .as_ref()
                    .and_then(|(state, _)| state.as_ref())
                    .is_some_and(crate::run_state::RunState::stopped_by_owner) =>
        {
            return;
        }
        crate::background_run::RunsTurn::Alone => None,
    };

    let mut reply: Option<(
        String,
        Option<aidaemon_channels::ports::AgentResponseEnvelope>,
    )> = None;
    let goal_scoped = call
        .goal_id
        .as_deref()
        .is_some_and(|goal_id| !goal_id.is_empty());
    if direct {
        info!(
            session_id = %notify_session_id,
            "Coding agent conversation: delivering the agent's result without a model turn"
        );
    }
    if !direct && !goal_scoped && !notify_session_id.trim().is_empty() {
        let reengagement_allowed = {
            let mut log = handles.reengagements.lock().await;
            crate::tools::terminal::reengagement_allowed(
                &mut log,
                &notify_session_id,
                &format!("cli_agent {}", call.tool_name),
                Instant::now(),
            )
        };
        if !reengagement_allowed {
            warn!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                "cli_agent completion re-engagement budget exhausted; delivering raw result"
            );
        } else if let Some(ref agent) = handles.agent {
            let ending = match end.phase.line(end.evidence) {
                Some(why) => format!("{status_word}. {why}"),
                None => status_word.to_string(),
            };
            let (followup, parent_task_id) = match group_answer.as_ref() {
                Some(answer) => (answer.followup.clone(), answer.parent_task_id.clone()),
                None => (
                    CliAgentTool::build_background_reengagement_followup(
                        &call.tool_name,
                        task_id_for_notify,
                        task_context,
                        &ending,
                        persisted_output,
                    ),
                    None,
                ),
            };
            let followup = match &handles.hub {
                Some(hub) if hub.shows_runs(&notify_session_id).await => format!(
                    "{followup}\n\n{}",
                    crate::helper_run::card::ROWS_SHOW_THE_END
                ),
                _ => followup,
            };
            info!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                "Re-engaging agent loop after CLI agent completion"
            );
            if let Some(run) = run.as_ref() {
                run.step(Step::Reporting).await;
            }
            // A request its device capped stays capped in the turn that
            // answers it (`crate::device_cap`).
            let capped = call.device_cap.as_ref().is_some_and(|cap| cap.is_capped());
            crate::owed_reports::owe(
                &*handles.state,
                crate::owed_reports::OwedReport {
                    run: task_id_for_notify.to_string(),
                    chat: notify_session_id.clone(),
                    parent_task_id: parent_task_id.clone(),
                    followup: followup.clone(),
                    capped,
                },
            )
            .await;
            let (continued, _followup_end) = crate::tools::terminal::run_background_continuation(
                agent.as_ref(),
                ConversationRequest {
                    session_id: notify_session_id.clone(),
                    user_text: followup,
                    status_tx: None,
                    user_role: aidaemon_tools::semantics::UserRole::Owner,
                    channel_ctx: crate::device_cap::continuation_context(capped),
                    heartbeat: None,
                    parent_task_id,
                    parent_tool_call_id: None,
                    parent_result_id: None,
                    clock: match group_answer.as_ref() {
                        Some(answer) => answer.clock,
                        None => ended
                            .as_ref()
                            .and_then(|(state, _)| state.as_ref())
                            .and_then(crate::run_state::RequestClock::of_run),
                    },
                },
            )
            .await;
            match continued {
                Ok(envelope) if !envelope.text.trim().is_empty() => {
                    let text = aidaemon_tools::sanitize::sanitize_user_facing_reply(&envelope.text);
                    reply = Some((text, Some(envelope)));
                }
                Ok(_) => warn!(
                    task_id = %task_id_for_notify,
                    session_id = %notify_session_id,
                    "cli_agent completion re-engagement returned an empty reply"
                ),
                Err(e) => warn!(
                    task_id = %task_id_for_notify,
                    session_id = %notify_session_id,
                    error = %e,
                    "cli_agent completion re-engagement failed"
                ),
            }
        }
    }

    let present = |body: &str| match (run.as_ref(), ended.as_ref()) {
        (Some(run), Some((state, on_card))) => run.result(state.as_ref(), body, *on_card),
        _ => aidaemon_channels::presentation::present_notification(notification_type, body),
    };
    let (text, envelope) = match (reply, group_answer) {
        (Some((text, envelope)), Some(_)) => (text, envelope),
        (Some((text, envelope)), None) => (present(&text), envelope),
        (None, Some(answer)) => (answer.messages.join("\n\n"), None),
        (None, None) => (present(&body), None),
    };

    let mut delivered = false;
    let on_card = direct
        && envelope.is_none()
        && run.is_some()
        && match handles.hub.as_ref() {
            Some(hub) => hub.shows_runs(&notify_session_id).await,
            None => false,
        };
    if on_card {
        info!(
            task_id = %task_id_for_notify,
            session_id = %notify_session_id,
            "Coding agent conversation: the run's card carries its answer"
        );
        delivered = true;
    } else if let Some(ref hub) = handles.hub {
        if let (Some(envelope), Some(agent)) = (envelope.as_ref(), handles.agent.as_ref()) {
            let _ = agent
                .record_continuation_delivery(
                    &notify_session_id,
                    envelope.delivery(
                        "background_router",
                        aidaemon_events::ResponseDeliveryState::Queued,
                        Vec::new(),
                        None,
                    ),
                )
                .await;
        }
        match hub.send_text_tracked(&notify_session_id, &text).await {
            Ok(platform_id) => {
                delivered = true;
                if envelope.is_none()
                    && let Some(agent) = handles.agent.as_ref()
                    && let Err(error) = agent
                        .record_delivered_result(
                            &notify_session_id,
                            &text,
                            aidaemon_core::SourceTrust::Untrusted,
                        )
                        .await
                {
                    warn!(
                        task_id = %task_id_for_notify,
                        session_id = %notify_session_id,
                        %error,
                        "cli_agent result was delivered but not kept in history"
                    );
                }
                if let (Some(envelope), Some(agent)) = (envelope.as_ref(), handles.agent.as_ref()) {
                    let _ = agent
                        .record_continuation_delivery(
                            &notify_session_id,
                            envelope.delivery(
                                "background_router",
                                aidaemon_events::ResponseDeliveryState::PlatformAcknowledged,
                                platform_id.into_iter().collect(),
                                None,
                            ),
                        )
                        .await;
                }
            }
            Err(e) => warn!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                error = %e,
                "cli_agent background completion direct hub delivery failed"
            ),
        }
    }

    if !delivered {
        let queue_goal_id = match call.goal_id.as_deref() {
            Some(goal_id) if !goal_id.trim().is_empty() => goal_id,
            _ => "global",
        };
        let entry = crate::tools::traits::NotificationEntry::new(
            queue_goal_id,
            &notify_session_id,
            notification_type,
            &text,
        );
        if let Err(e) = handles.state.enqueue_notification(&entry).await {
            warn!(
                task_id = %task_id_for_notify,
                session_id = %notify_session_id,
                error = %e,
                "cli_agent background completion enqueue failed"
            );
        }
    }
    crate::owed_reports::paid(&*handles.state, task_id_for_notify).await;
}
