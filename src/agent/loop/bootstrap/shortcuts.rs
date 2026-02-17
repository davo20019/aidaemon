use crate::agent::*;

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn maybe_handle_stop_command(
        &self,
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
            let reply = self
                .emit_bootstrap_direct_reply(emitter, task_id, session_id, early_task_start, reply)
                .await?;
            return Ok(Some(reply));
        }

        let cancel_result = self
            .execute_tool_with_watchdog(
                "cli_agent",
                r#"{"action": "cancel_all"}"#,
                &tool_exec::ToolExecCtx {
                    session_id,
                    task_id: Some(task_id),
                    status_tx,
                    channel_visibility: channel_ctx.visibility,
                    channel_id: channel_ctx.channel_id.as_deref(),
                    trusted: channel_ctx.trusted,
                    user_role,
                },
            )
            .await;
        let cli_cancel_msg = cancel_result.ok();

        // Cancel any active goals for this session as well (background task leads/executors).
        let cancelled_goals = self.cancel_active_goals_for_session(session_id).await;

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

        let reply = self
            .emit_bootstrap_direct_reply(emitter, task_id, session_id, early_task_start, &reply)
            .await?;
        Ok(Some(reply))
    }

    /// Detect explicit "pivot" turns (e.g. "wait stop... actually ... instead")
    /// and cancel any in-flight background work before continuing with the new
    /// instruction in the same turn.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn maybe_cancel_work_for_mid_task_pivot(
        &self,
        session_id: &str,
        user_text: &str,
        user_role: UserRole,
        channel_ctx: &ChannelContext,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        task_id: &str,
    ) {
        if self.depth != 0 || user_role != UserRole::Owner || !looks_like_mid_task_pivot(user_text)
        {
            return;
        }

        let cancel_result = self
            .execute_tool_with_watchdog(
                "cli_agent",
                r#"{"action": "cancel_all"}"#,
                &tool_exec::ToolExecCtx {
                    session_id,
                    task_id: Some(task_id),
                    status_tx,
                    channel_visibility: channel_ctx.visibility,
                    channel_id: channel_ctx.channel_id.as_deref(),
                    trusted: channel_ctx.trusted,
                    user_role,
                },
            )
            .await;

        let cli_cancel_msg = match cancel_result {
            Ok(msg) => Some(msg),
            Err(e) => {
                warn!(
                    session_id,
                    task_id = %task_id,
                    error = %e,
                    "Failed to cancel in-flight cli_agent work during pivot"
                );
                None
            }
        };

        let cancelled_goals = self.cancel_active_goals_for_session(session_id).await;
        let cli_cancelled_any = cli_cancel_msg
            .as_deref()
            .is_some_and(|m| !m.contains("No running CLI agents"));

        if cli_cancelled_any || !cancelled_goals.is_empty() {
            info!(
                session_id,
                task_id = %task_id,
                cli_cancelled_any,
                cancelled_goals = cancelled_goals.len(),
                "Detected mid-task pivot; cancelled in-flight session work"
            );
        }
    }

    pub(super) async fn maybe_handle_pending_goal_confirmation(
        &self,
        session_id: &str,
        user_text: &str,
        user_role: UserRole,
        task_id: &str,
        emitter: &crate::events::EventEmitter,
    ) -> anyhow::Result<Option<String>> {
        let early_task_start = Instant::now();
        let pending_goals = self
            .state
            .get_pending_confirmation_goals(session_id)
            .await
            .unwrap_or_default();

        if pending_goals.is_empty() {
            return Ok(None);
        }

        if user_role == UserRole::Owner {
            let lower_trimmed = user_text.trim().to_lowercase();
            let is_confirm = ["confirm", "yes", "go ahead", "schedule it", "do it"]
                .iter()
                .any(|kw| contains_keyword_as_words(&lower_trimmed, kw));
            let is_reject = ["no", "cancel", "never mind", "nevermind"]
                .iter()
                .any(|kw| contains_keyword_as_words(&lower_trimmed, kw));

            if is_confirm {
                let mut activated = Vec::new();
                let mut activation_errors = Vec::new();
                let tz_label = crate::cron_utils::system_timezone_display();

                for goal in &pending_goals {
                    match self.state.activate_goal(&goal.id).await {
                        Ok(true) => {
                            if let Some(ref registry) = self.goal_token_registry {
                                registry.register(&goal.id).await;
                            }
                            let schedules = self
                                .state
                                .get_schedules_for_goal(&goal.id)
                                .await
                                .unwrap_or_default();
                            let next_run = schedules
                                .iter()
                                .filter_map(|s| {
                                    chrono::DateTime::parse_from_rfc3339(&s.next_run_at).ok()
                                })
                                .min_by_key(|dt| dt.timestamp())
                                .map(|dt| {
                                    dt.with_timezone(&chrono::Local)
                                        .format("%Y-%m-%d %H:%M %Z")
                                        .to_string()
                                })
                                .unwrap_or_else(|| "unscheduled".to_string());
                            activated.push(format!("{} (next: {})", goal.description, next_run));
                        }
                        Ok(false) => {}
                        Err(e) => activation_errors.push(e.to_string()),
                    }
                }

                let msg = if !activated.is_empty() && activation_errors.is_empty() {
                    if activated.len() == 1 {
                        format!(
                            "Scheduled: {}. I'll execute it when the time comes. System timezone: {}.",
                            activated[0], tz_label
                        )
                    } else {
                        format!(
                            "Scheduled {} goals:\n- {}\nSystem timezone: {}.",
                            activated.len(),
                            activated.join("\n- "),
                            tz_label
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

                let msg = self
                    .emit_bootstrap_direct_reply(
                        emitter,
                        task_id,
                        session_id,
                        early_task_start,
                        &msg,
                    )
                    .await?;
                return Ok(Some(msg));
            }

            if is_reject {
                let mut cancelled = 0usize;
                for goal in &pending_goals {
                    let mut updated = goal.clone();
                    updated.status = "cancelled".to_string();
                    updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    updated.updated_at = chrono::Utc::now().to_rfc3339();
                    if self.state.update_goal(&updated).await.is_ok() {
                        cancelled += 1;
                    }
                    // Best-effort cleanup: schedules were created before confirmation.
                    // Cancelled goals should not retain schedules.
                    if let Ok(schedules) = self.state.get_schedules_for_goal(&updated.id).await {
                        for s in &schedules {
                            let _ = self.state.delete_goal_schedule(&s.id).await;
                        }
                    }
                }

                let msg = if cancelled == 1 {
                    "OK, cancelled the scheduled goal.".to_string()
                } else {
                    format!("OK, cancelled {} scheduled goals.", cancelled)
                };

                let msg = self
                    .emit_bootstrap_direct_reply(
                        emitter,
                        task_id,
                        session_id,
                        early_task_start,
                        &msg,
                    )
                    .await?;
                return Ok(Some(msg));
            }

            // User moved on without explicit confirmation/rejection.
            // Auto-cancel pending confirmations to avoid stale intents.
            for goal in &pending_goals {
                let mut updated = goal.clone();
                updated.status = "cancelled".to_string();
                updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
                updated.updated_at = chrono::Utc::now().to_rfc3339();
                let _ = self.state.update_goal(&updated).await;
                // Best-effort cleanup: remove any schedules created pre-confirmation.
                if let Ok(schedules) = self.state.get_schedules_for_goal(&updated.id).await {
                    for s in &schedules {
                        let _ = self.state.delete_goal_schedule(&s.id).await;
                    }
                }
            }
            return Ok(None);
        }

        // Non-owner: if they typed confirm/reject keywords,
        // return owner-only message immediately (no LLM call).
        let lower_trimmed = user_text.trim().to_lowercase();
        let is_confirm_or_reject = [
            "confirm",
            "yes",
            "go ahead",
            "schedule it",
            "do it",
            "no",
            "cancel",
            "never mind",
            "nevermind",
        ]
        .iter()
        .any(|kw| contains_keyword_as_words(&lower_trimmed, kw));
        if is_confirm_or_reject {
            let msg = "Only the owner can confirm or cancel scheduled goals.";
            let msg = self
                .emit_bootstrap_direct_reply(emitter, task_id, session_id, early_task_start, msg)
                .await?;
            return Ok(Some(msg));
        }

        // Otherwise: ignore pending goals, proceed normally.
        // Don't confirm, reject, or auto-cancel.
        Ok(None)
    }

    pub(super) async fn maybe_handle_trivial_ack_shortcut(
        &self,
        session_id: &str,
        user_text: &str,
        task_id: &str,
        emitter: &crate::events::EventEmitter,
    ) -> anyhow::Result<Option<String>> {
        // Cheap local acknowledgment shortcut: avoid an LLM call for trivial turns like
        // "thanks" or a single emoji reaction. Keep this conservative to avoid eating
        // genuine requests.
        if self.depth != 0 {
            return Ok(None);
        }

        let trimmed = user_text.trim();
        let normalized = trimmed
            .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
            .to_ascii_lowercase();
        let is_thanks = matches!(normalized.as_str(), "thanks" | "thank you" | "thx");
        let is_ok = matches!(normalized.as_str(), "ok" | "okay");
        let is_single_emoji_reaction = {
            let char_count = trimmed.chars().count();
            char_count > 0
                && char_count <= 4
                && !trimmed.is_ascii()
                && trimmed
                    .chars()
                    .all(|c| !c.is_ascii_alphanumeric() && !c.is_ascii_whitespace())
        };

        // If the previous assistant turn ended with a question, treat "ok/okay" as non-terminal
        // and let the LLM re-ask for missing info rather than replying "Got it.".
        let ok_is_safe_to_short_circuit = if is_ok {
            let history = self
                .state
                .get_history(session_id, 12)
                .await
                .unwrap_or_default();
            let last_assistant = history.iter().rev().find(|m| m.role == "assistant");
            !last_assistant
                .and_then(|m| m.content.as_deref())
                .is_some_and(|c| c.contains('?'))
        } else {
            true
        };

        let trivial_reply = if is_thanks {
            Some("You're welcome.".to_string())
        } else if is_single_emoji_reaction || (is_ok && ok_is_safe_to_short_circuit) {
            Some("Got it.".to_string())
        } else {
            None
        };

        if let Some(reply) = trivial_reply {
            let reply = self
                .emit_bootstrap_direct_reply(emitter, task_id, session_id, Instant::now(), &reply)
                .await?;
            return Ok(Some(reply));
        }

        Ok(None)
    }

    pub(super) async fn maybe_handle_time_query_shortcut(
        &self,
        session_id: &str,
        user_text: &str,
        task_id: &str,
        emitter: &crate::events::EventEmitter,
    ) -> anyhow::Result<Option<String>> {
        // Cheap local time shortcut: avoid an LLM call for "what time is it?" style requests.
        // Keep this strict (exact-match after normalization) so we don't mis-handle timezone
        // or location-specific queries (e.g., "what time is it in Tokyo?").
        if self.depth != 0 {
            return Ok(None);
        }

        let trimmed = user_text.trim();
        let normalized = trimmed
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c.is_whitespace() {
                    c.to_ascii_lowercase()
                } else {
                    ' '
                }
            })
            .collect::<String>();
        let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
        let is_time_query = matches!(
            normalized.as_str(),
            "what time is it"
                | "what time is it now"
                | "what time is it right now"
                | "what is the time"
                | "what s the time"
                | "whats the time"
                | "current time"
                | "time now"
                | "time"
        );

        if !is_time_query {
            return Ok(None);
        }

        let now = chrono::Local::now();
        let reply = format!("It is {}.", now.format("%Y-%m-%d %H:%M:%S %Z (UTC%:z)"));
        let reply = self
            .emit_bootstrap_direct_reply(emitter, task_id, session_id, Instant::now(), &reply)
            .await?;
        Ok(Some(reply))
    }

    pub(super) async fn emit_bootstrap_direct_reply(
        &self,
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
            embedding: None,
        };
        self.append_assistant_message_with_event(emitter, &assistant_msg, "system", None, None)
            .await?;

        self.emit_task_end(
            emitter,
            task_id,
            TaskStatus::Completed,
            task_start,
            0,
            0,
            None,
            Some(reply_text.chars().take(200).collect()),
        )
        .await;

        Ok(reply_text)
    }
}

fn looks_like_mid_task_pivot(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    // Exact stop/cancel/abort is handled by maybe_handle_stop_command and should
    // return immediately to the user, not continue as a pivot.
    if matches!(lower.as_str(), "stop" | "cancel" | "abort") {
        return false;
    }

    let has_cancel_cue = [
        "stop",
        "cancel",
        "abort",
        "scratch that",
        "forget that",
        "never mind",
        "nevermind",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    if !has_cancel_cue {
        return false;
    }

    let has_pivot_cue = [
        "actually",
        "instead",
        "rather",
        "new plan",
        "change of plan",
        "let's",
        "lets",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    has_pivot_cue && lower.split_whitespace().count() >= 5
}

#[cfg(test)]
mod tests {
    use super::looks_like_mid_task_pivot;

    #[test]
    fn test_looks_like_mid_task_pivot_detects_explicit_pivot() {
        assert!(looks_like_mid_task_pivot(
            "Wait stop. Actually scratch React and do plain HTML/CSS/JS instead."
        ));
        assert!(looks_like_mid_task_pivot(
            "Cancel that and instead generate a static page."
        ));
    }

    #[test]
    fn test_looks_like_mid_task_pivot_ignores_plain_stop() {
        assert!(!looks_like_mid_task_pivot("stop"));
        assert!(!looks_like_mid_task_pivot("cancel"));
        assert!(!looks_like_mid_task_pivot("abort"));
    }

    #[test]
    fn test_looks_like_mid_task_pivot_requires_both_cancel_and_pivot_cues() {
        assert!(!looks_like_mid_task_pivot(
            "Actually create a static page instead."
        ));
        assert!(!looks_like_mid_task_pivot("Never mind."));
    }
}
