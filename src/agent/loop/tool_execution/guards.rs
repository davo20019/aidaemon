use super::types::ToolExecutionOutcome;
use crate::agent::*;

pub(super) enum LoopPatternGuardOutcome {
    ContinueLoop,
    Return(ToolExecutionOutcome),
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn maybe_handle_loop_pattern_guards(
        &self,
        tc: &ToolCall,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        session_id: &str,
        iteration: usize,
        task_start: Instant,
        task_tokens_used: u64,
        learning_ctx: &LearningContext,
        recent_tool_calls: &mut VecDeque<u64>,
        recent_tool_names: &mut VecDeque<String>,
        consecutive_same_tool: &mut (String, usize),
        consecutive_same_tool_arg_hashes: &mut HashSet<u64>,
    ) -> anyhow::Result<Option<LoopPatternGuardOutcome>> {
        // Check for repetitive behavior (same tool call hash appearing too often)
        let call_hash = hash_tool_call(&tc.name, &tc.arguments);
        recent_tool_calls.push_back(call_hash);
        if recent_tool_calls.len() > RECENT_CALLS_WINDOW {
            recent_tool_calls.pop_front();
        }

        // Count how many of the recent calls match this one
        let repetitive_count = recent_tool_calls
            .iter()
            .filter(|&&h| h == call_hash)
            .count();
        let repetitive_redirect_threshold =
            repetitive_redirect_threshold_for_call(&tc.name, &tc.arguments);

        // Soft redirect: skip execution and coach the LLM to adapt.
        // This fires BEFORE the hard stall, giving the agent a chance
        // to change approach instead of just giving up.
        if (repetitive_redirect_threshold..MAX_REPETITIVE_CALLS).contains(&repetitive_count) {
            warn!(
                session_id,
                tool = %tc.name,
                repetitive_count,
                "Redirecting repetitive tool call — coaching agent to adapt"
            );
            self.emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::RepetitiveCallDetection,
                format!(
                    "Repetitive tool call redirected for {} (count={})",
                    tc.name, repetitive_count
                ),
                json!({
                    "tool": tc.name,
                    "count": repetitive_count,
                    "action": "redirect"
                }),
            )
            .await;
            let redirect_msg = format!(
                "[SYSTEM] BLOCKED: You already called `{}` with these exact same arguments {} times \
                         and got the same result. Repeating it will NOT produce a different outcome.\n\n\
                         You MUST change your approach. Options:\n\
                         - Use DIFFERENT arguments or a different command\n\
                         - If you're missing information (URL, credentials, deployment method), \
                         ASK the user instead of guessing\n\
                         - If this sub-task is blocked, skip it and tell the user what you \
                         accomplished and what still needs their input",
                tc.name, repetitive_count
            );
            let tool_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
                role: "tool".to_string(),
                content: Some(redirect_msg),
                tool_call_id: Some(tc.id.clone()),
                tool_name: Some(tc.name.clone()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.3,
                embedding: None,
            };
            self.append_tool_message_with_result_event(
                emitter,
                &tool_msg,
                true,
                0,
                None,
                Some(task_id),
            )
            .await?;
            return Ok(Some(LoopPatternGuardOutcome::ContinueLoop));
        }

        if repetitive_count >= MAX_REPETITIVE_CALLS {
            warn!(
                session_id,
                tool = %tc.name,
                repetitive_count,
                "Repetitive tool call detected - agent may be stuck"
            );
            self.emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::RepetitiveCallDetection,
                format!(
                    "Repetitive tool call hard-stopped for {} (count={})",
                    tc.name, repetitive_count
                ),
                json!({
                    "tool": tc.name,
                    "count": repetitive_count,
                    "action": "hard_stop"
                }),
            )
            .await;
            let result = self
                .graceful_repetitive_response(emitter, session_id, learning_ctx, &tc.name)
                .await;
            let (status, error, summary) = match &result {
                Ok(reply) => (
                    TaskStatus::Failed,
                    Some("Repetitive tool calls".to_string()),
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
            return Ok(Some(LoopPatternGuardOutcome::Return(
                ToolExecutionOutcome::Return(result),
            )));
        }

        // Check for consecutive same-tool-name loop.
        // Track unique argument hashes within the streak so we can
        // distinguish productive work (many different commands) from
        // an actual loop (few unique args recycled over and over).
        if tc.name == consecutive_same_tool.0 {
            consecutive_same_tool.1 += 1;
            consecutive_same_tool_arg_hashes.insert(call_hash);
        } else {
            *consecutive_same_tool = (tc.name.clone(), 1);
            consecutive_same_tool_arg_hashes.clear();
            consecutive_same_tool_arg_hashes.insert(call_hash);
        }
        if consecutive_same_tool.1 >= MAX_CONSECUTIVE_SAME_TOOL {
            let total = consecutive_same_tool.1;
            let unique = consecutive_same_tool_arg_hashes.len();
            // Diverse args get a small bonus (+4), not a full bypass.
            // Even with different commands, 20+ consecutive same-tool
            // calls without switching tools indicates a stuck loop.
            let is_diverse = unique * 2 > total;
            let diverse_limit = MAX_CONSECUTIVE_SAME_TOOL + 4;
            self.emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::ConsecutiveSameToolDetection,
                format!(
                    "Consecutive same-tool detection for {} (total={}, unique_args={})",
                    tc.name, total, unique
                ),
                json!({
                    "tool": tc.name,
                    "consecutive_count": total,
                    "unique_args": unique,
                    "is_diverse": is_diverse
                }),
            )
            .await;
            if !is_diverse || total >= diverse_limit {
                warn!(
                    session_id,
                    tool = %tc.name,
                    consecutive = total,
                    unique_args = unique,
                    "Same tool called too many consecutive times - agent is looping"
                );
                let result = self
                    .graceful_repetitive_response(emitter, session_id, learning_ctx, &tc.name)
                    .await;
                let (status, error, summary) = match &result {
                    Ok(reply) => (
                        TaskStatus::Failed,
                        Some("Consecutive same-tool loop".to_string()),
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
                return Ok(Some(LoopPatternGuardOutcome::Return(
                    ToolExecutionOutcome::Return(result),
                )));
            }
        }

        // Check for alternating tool patterns (A-B-A-B cycles)
        // Only detects when exactly 2 different tools alternate — a
        // single tool used repeatedly is handled by consecutive-same-tool
        // detection above (which has proper argument diversity checks).
        recent_tool_names.push_back(tc.name.clone());
        if recent_tool_names.len() > ALTERNATING_PATTERN_WINDOW {
            recent_tool_names.pop_front();
        }
        if recent_tool_names.len() >= ALTERNATING_PATTERN_WINDOW {
            let unique_tools: HashSet<&String> = recent_tool_names.iter().collect();
            // Only fire for exactly 2 unique tools (true A-B-A-B pattern).
            // A single tool (unique_tools.len() == 1) is NOT an alternating
            // pattern — it's a legitimate streak of varied commands (e.g.
            // terminal with different args) and is guarded by the
            // consecutive-same-tool check instead.
            if unique_tools.len() == 2 {
                // Additional diversity check: if the argument hashes in the
                // window are mostly unique, the agent may be doing real work
                // that happens to bounce between two tools (e.g. terminal +
                // web_search).  Only trigger when diversity is low.
                let recent_hashes: HashSet<&u64> = recent_tool_calls.iter().collect();
                let diversity_ratio = if recent_tool_calls.is_empty() {
                    1.0
                } else {
                    recent_hashes.len() as f64 / recent_tool_calls.len() as f64
                };
                // High diversity (>60% unique calls) → productive work, skip
                if diversity_ratio <= 0.6 {
                    let tool_names: Vec<String> =
                        unique_tools.iter().map(|t| (*t).clone()).collect();
                    self.emit_decision_point(
                        emitter,
                        task_id,
                        iteration,
                        DecisionType::AlternatingPatternDetection,
                        "Alternating A-B loop detected".to_string(),
                        json!({
                            "tools": tool_names,
                            "diversity_ratio": diversity_ratio
                        }),
                    )
                    .await;
                    warn!(
                        session_id,
                        tools = ?unique_tools,
                        window = ALTERNATING_PATTERN_WINDOW,
                        diversity_ratio,
                        "Alternating tool pattern detected - agent is looping"
                    );
                    let result = self
                        .graceful_repetitive_response(emitter, session_id, learning_ctx, &tc.name)
                        .await;
                    let (status, error, summary) = match &result {
                        Ok(reply) => (
                            TaskStatus::Failed,
                            Some("Alternating tool loop".to_string()),
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
                    return Ok(Some(LoopPatternGuardOutcome::Return(
                        ToolExecutionOutcome::Return(result),
                    )));
                }
            }
        }

        Ok(None)
    }
}

fn is_background_status_poll(tool_name: &str, arguments: &str) -> bool {
    if !matches!(tool_name, "terminal" | "cli_agent" | "spawn_agent") {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| {
            v.get("action")
                .and_then(|a| a.as_str())
                .map(|s| s.eq_ignore_ascii_case("check"))
        })
        .unwrap_or(false)
}

fn repetitive_redirect_threshold_for_call(tool_name: &str, arguments: &str) -> usize {
    if is_background_status_poll(tool_name, arguments) {
        // Background status checks can consume budget quickly with little progress.
        2
    } else {
        REPETITIVE_REDIRECT_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::{is_background_status_poll, repetitive_redirect_threshold_for_call};
    use crate::agent::REPETITIVE_REDIRECT_THRESHOLD;

    #[test]
    fn detects_terminal_check_action() {
        assert!(is_background_status_poll(
            "terminal",
            r#"{"action":"check","pid":123}"#
        ));
    }

    #[test]
    fn ignores_terminal_run_action() {
        assert!(!is_background_status_poll(
            "terminal",
            r#"{"action":"run","command":"ls"}"#
        ));
    }

    #[test]
    fn detects_cli_agent_check_action() {
        assert!(is_background_status_poll(
            "cli_agent",
            r#"{"action":"check","task_id":"abc"}"#
        ));
    }

    #[test]
    fn lowers_redirect_threshold_for_background_polls() {
        assert_eq!(
            repetitive_redirect_threshold_for_call("terminal", r#"{"action":"check","pid":7}"#),
            2
        );
        assert_eq!(
            repetitive_redirect_threshold_for_call("cli_agent", r#"{"action":"check"}"#),
            2
        );
        assert_eq!(
            repetitive_redirect_threshold_for_call("terminal", r#"{"action":"run"}"#),
            REPETITIVE_REDIRECT_THRESHOLD
        );
    }
}
