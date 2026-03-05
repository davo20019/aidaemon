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
        tool_result_cache: &HashMap<u64, String>,
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
            let is_read_only = matches!(tc.name.as_str(), "read_file" | "search_files");
            let redirect_msg = if is_read_only {
                // For read-only tools, replay cached content so the model can
                // act on it despite context truncation dropping earlier results.
                if let Some(cached) = tool_result_cache.get(&call_hash) {
                    format!(
                        "[SYSTEM] You already read this content {} times. Here it is again — \
                         do NOT read it again. Use `write_file` to apply your fixes NOW.\n\n\
                         --- CACHED CONTENT ---\n{}\n--- END CACHED CONTENT ---\n\n\
                         IMPORTANT: You have the file content above. Analyze the bugs and \
                         write the corrected version using `write_file`. Do not call `{}` again.",
                        repetitive_count, cached, tc.name
                    )
                } else {
                    format!(
                        "[SYSTEM] BLOCKED: You already called `{}` with these exact same arguments {} times. \
                         The file content should be in your conversation history. \
                         Use `write_file` to apply your fixes based on what you already know.\n\n\
                         You MUST use `write_file` now — do not try to read the file again.",
                        tc.name, repetitive_count
                    )
                }
            } else {
                // Check if the agent has been editing files — if so, guide toward
                // edit_file rather than generic "change approach" advice.
                let has_edited = recent_tool_names.iter().any(|n| n == "edit_file");
                if has_edited && tc.name == "terminal" {
                    format!(
                        "[SYSTEM] BLOCKED: You already ran this exact terminal command {} times. \
                         Re-running tests will NOT fix bugs — you must edit the code first.\n\n\
                         NEXT STEP: Use `edit_file` to fix the next bug, THEN run tests once to verify.\n\
                         Look at the test failure output you already have and identify which file and line needs fixing.",
                        repetitive_count
                    )
                } else {
                    format!(
                        "[SYSTEM] BLOCKED: You already called `{}` with these exact same arguments {} times \
                                 and got the same result. Repeating it will NOT produce a different outcome.\n\n\
                                 You MUST change your approach. Options:\n\
                                 - Use DIFFERENT arguments or a different command\n\
                                 - If you're missing information (URL, credentials, deployment method), \
                                 ASK the user instead of guessing\n\
                                 - If this sub-task is blocked, skip it and tell the user what you \
                                 accomplished and what still needs their input",
                        tc.name, repetitive_count
                    )
                }
            };
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
            // Wrap the entire graceful shutdown in a timeout to prevent
            // indefinite hangs from SQLite pool exhaustion or deadlocks.
            let graceful_fut = async {
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
                self.graceful_repetitive_response(emitter, session_id, learning_ctx, &tc.name)
                    .await
            };
            let result = match tokio::time::timeout(Duration::from_secs(10), graceful_fut).await {
                Ok(result) => result,
                Err(_) => {
                    warn!(
                        session_id,
                        tool = %tc.name,
                        "Graceful repetitive response timed out after 10s — using fallback"
                    );
                    Ok(crate::agent::post_task::graceful_repetitive_response(
                        learning_ctx,
                        &tc.name,
                    ))
                }
            };
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
            // Also timeout the task_end emit to prevent hangs there too.
            let _ = tokio::time::timeout(
                Duration::from_secs(5),
                self.emit_task_end(
                    emitter,
                    task_id,
                    status,
                    task_start,
                    iteration,
                    learning_ctx.tool_calls.len(),
                    error,
                    summary,
                ),
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
        // Read-only tools get a higher threshold instead of complete exemption.
        // A 3-file debugging task may need ~5 reads, but 10+ reads in a row
        // with low argument diversity is a stuck loop, not productive work.
        let read_only_higher_threshold = matches!(
            tc.name.as_str(),
            "read_file" | "search_files" | "check_environment"
        );
        let effective_same_tool_limit = if read_only_higher_threshold {
            MAX_CONSECUTIVE_SAME_TOOL + 4 // 12 for read tools
        } else {
            MAX_CONSECUTIVE_SAME_TOOL
        };
        if consecutive_same_tool.1 >= effective_same_tool_limit {
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
        //
        // Read-like terminal commands (cat, head, tail) are tagged as
        // "terminal_read" so the read-saturation escalation counts them
        // as consecutive reads. Without this, the agent alternates
        // read_file + terminal(cat), resetting the read counter each time.
        let effective_tool_name = if tc.name == "terminal" {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(&tc.arguments) {
                let cmd = args.get("command").and_then(|c| c.as_str()).unwrap_or("");
                let first_word = cmd.split_whitespace().next().unwrap_or("");
                if matches!(
                    first_word,
                    "cat" | "head" | "tail" | "less" | "more" | "bat"
                ) {
                    "terminal_read".to_string()
                } else {
                    tc.name.clone()
                }
            } else {
                tc.name.clone()
            }
        } else {
            tc.name.clone()
        };
        recent_tool_names.push_back(effective_tool_name);
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
                // Skip if one of the tools is a read-only tool. The agent
                // legitimately alternates read_file/search_files with terminal
                // when gathering information, especially during multi-file
                // debugging where context truncation forces re-reads.
                let has_read_only_tool = unique_tools.iter().any(|t| {
                    let s = t.as_str();
                    matches!(s, "read_file" | "search_files" | "terminal_read")
                        || s.ends_with("__read_file")
                        || s.ends_with("__search_files")
                });
                if has_read_only_tool {
                    // Don't trigger alternating pattern for read-only tools
                } else {
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
                            .graceful_repetitive_response(
                                emitter,
                                session_id,
                                learning_ctx,
                                &tc.name,
                            )
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
                } // close else branch for non-read-only tools
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
    } else if matches!(tool_name, "read_file" | "search_files") {
        // Read-only tools need a higher threshold because context truncation
        // in long sessions drops their results, forcing legitimate re-reads.
        // Blocking at 3 prevents the agent from ever reaching the write phase
        // in multi-file debugging tasks.
        6
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
    fn raises_redirect_threshold_for_read_only_tools() {
        assert_eq!(
            repetitive_redirect_threshold_for_call("read_file", r#"{"path":"/tmp/foo.py"}"#),
            6
        );
        assert_eq!(
            repetitive_redirect_threshold_for_call("search_files", r#"{"query":"bug"}"#),
            6
        );
        // Other tools use default threshold
        assert_eq!(
            repetitive_redirect_threshold_for_call("write_file", r#"{"path":"/tmp/foo.py"}"#),
            REPETITIVE_REDIRECT_THRESHOLD
        );
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
