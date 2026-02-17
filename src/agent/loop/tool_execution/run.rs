use super::budget_blocking::{DuplicateSendFileNoopCtx, ToolBudgetBlockCtx};
use super::execution_io::ToolExecutionIoCtx;
use super::guards::LoopPatternGuardOutcome;
use super::project_dir::{
    extract_project_dir_hint, extract_project_dirs_from_tool_args, is_file_recheck_tool,
    maybe_inject_project_dir_into_tool_args, project_dir_from_tool_args, scope_allows_project_dir,
    tool_call_includes_project_path,
};
use super::result_learning::{ResultLearningEnv, ResultLearningState};
use super::types::{ToolExecutionCtx, ToolExecutionOutcome};
use crate::agent::recall_guardrails::is_personal_memory_tool;
use crate::agent::*;

fn raw_internal_scope_violation(
    raw_arguments: &str,
    session_id: &str,
    resolved_goal_id: Option<&str>,
) -> Option<String> {
    let parsed = serde_json::from_str::<Value>(raw_arguments).ok()?;
    let map = parsed.as_object()?;

    if let Some(candidate_session_id) = map.get("_session_id").and_then(|v| v.as_str()) {
        if candidate_session_id != session_id {
            return Some(format!(
                "_session_id mismatch (expected `{}`, got `{}`)",
                session_id, candidate_session_id
            ));
        }
    }

    if let Some(candidate_goal_id) = map.get("_goal_id").and_then(|v| v.as_str()) {
        match resolved_goal_id {
            Some(expected_goal_id) if candidate_goal_id != expected_goal_id => {
                return Some(format!(
                    "_goal_id mismatch (expected `{}`, got `{}`)",
                    expected_goal_id, candidate_goal_id
                ));
            }
            None => {
                return Some(format!(
                    "_goal_id `{}` provided but no goal scope is active",
                    candidate_goal_id
                ));
            }
            _ => {}
        }
    }

    None
}

fn project_scope_violation_for_tool_call(
    tool_name: &str,
    effective_arguments: &str,
    allowed_scope: Option<&str>,
    allow_multi_project_scope: bool,
) -> Option<String> {
    if allow_multi_project_scope {
        return None;
    }

    let allowed_scope = allowed_scope?;
    let candidate_dirs = extract_project_dirs_from_tool_args(tool_name, effective_arguments);
    if candidate_dirs.is_empty() {
        return None;
    }

    let allow_scaffold_parent_dir = |candidate: &str| -> bool {
        if tool_name != "run_command" {
            return false;
        }
        let Some(scope_path) = crate::tools::fs_utils::validate_path(allowed_scope).ok() else {
            return false;
        };
        if scope_path.is_dir() {
            return false;
        }
        let Some(candidate_path) = crate::tools::fs_utils::validate_path(candidate).ok() else {
            return false;
        };
        scope_path
            .parent()
            .is_some_and(|parent| parent.is_dir() && candidate_path == parent)
    };

    let violations: Vec<String> = candidate_dirs
        .iter()
        .filter(|dir| {
            !scope_allows_project_dir(allowed_scope, dir) && !allow_scaffold_parent_dir(dir)
        })
        .cloned()
        .collect();
    if violations.is_empty() {
        None
    } else {
        Some(format!(
            "project scope lock violation (allowed scope `{}`, requested path(s): {})",
            allowed_scope,
            violations.join(", ")
        ))
    }
}

fn is_hard_policy_tool_budget_reached(
    total_tool_calls_attempted: usize,
    policy_tool_budget: usize,
) -> bool {
    policy_tool_budget > 0 && total_tool_calls_attempted >= policy_tool_budget
}

fn tool_result_indicates_background_detach(tool_name: &str, result_text: &str) -> bool {
    let _ = tool_name;
    result_text.contains("Moved to background")
        || result_text.contains("started in background")
        || result_text.contains("spawned in background")
}

fn build_background_detach_ack(tool_name: &str, result_text: &str) -> String {
    let default_prefix = match tool_name {
        "terminal" => "The command is running in the background.",
        "cli_agent" => "The CLI agent task is running in the background.",
        "spawn_agent" => "The spawned sub-agent is running in the background.",
        _ => "The task is running in the background.",
    };
    let first_line = result_text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("[SYSTEM]"))
        .unwrap_or(default_prefix);
    format!(
        "{} Completion notifications are enabled, and the final result will be sent automatically when it finishes.",
        first_line
    )
}

fn run_command_policy_block_requires_terminal(result_text: &str) -> bool {
    let lower = result_text.to_ascii_lowercase();
    lower.contains("safe command list")
        || lower.contains("use 'terminal' for this command")
        || lower.contains("use `terminal`")
        || lower.contains("shell operators")
        || lower.contains("daemonization primitives are blocked in run_command")
}

fn shell_single_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('\'');
    for ch in value.chars() {
        if ch == '\'' {
            quoted.push_str("'\"'\"'");
        } else {
            quoted.push(ch);
        }
    }
    quoted.push('\'');
    quoted
}

fn build_terminal_fallback_arguments_from_run_command(raw_arguments: &str) -> Option<String> {
    let args = serde_json::from_str::<Value>(raw_arguments).ok()?;
    let map = args.as_object()?;
    let command = map.get("command").and_then(|v| v.as_str())?.trim();
    if command.is_empty() {
        return None;
    }
    let working_dir = map
        .get("working_dir")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty());
    let terminal_command = if let Some(dir) = working_dir {
        format!("cd {} && {}", shell_single_quote(dir), command)
    } else {
        command.to_string()
    };
    Some(
        json!({
            "action": "run",
            "command": terminal_command,
        })
        .to_string(),
    )
}

impl Agent {
    pub(in crate::agent) async fn run_tool_execution_phase(
        &self,
        ctx: &mut ToolExecutionCtx<'_>,
    ) -> anyhow::Result<ToolExecutionOutcome> {
        let resp = ctx.resp;
        let emitter = ctx.emitter;
        let task_id = ctx.task_id;
        let session_id = ctx.session_id;
        let iteration = ctx.iteration;
        let task_start = ctx.task_start;
        let learning_ctx = &mut *ctx.learning_ctx;
        let task_tokens_used = ctx.task_tokens_used;
        let _user_text = ctx.user_text;
        let restrict_to_personal_memory_tools = ctx.restrict_to_personal_memory_tools;
        let is_reaffirmation_challenge_turn = ctx.is_reaffirmation_challenge_turn;
        let personal_memory_tool_call_cap = ctx.personal_memory_tool_call_cap;
        let base_tool_defs = ctx.base_tool_defs;
        let available_capabilities = ctx.available_capabilities;
        let policy_bundle = ctx.policy_bundle;
        let status_tx = ctx.status_tx.clone();
        let channel_ctx = ctx.channel_ctx;
        let user_role = ctx.user_role;
        let heartbeat = ctx.heartbeat;
        let turn_context = ctx.turn_context;
        let resolved_goal_id = ctx.resolved_goal_id;

        let mut tool_defs = std::mem::take(ctx.tool_defs);
        let mut total_tool_calls_attempted = *ctx.total_tool_calls_attempted;
        let mut total_successful_tool_calls = *ctx.total_successful_tool_calls;
        let mut tool_failure_count = std::mem::take(ctx.tool_failure_count);
        let mut tool_transient_failure_count = std::mem::take(ctx.tool_transient_failure_count);
        let mut tool_cooldown_until_iteration = std::mem::take(ctx.tool_cooldown_until_iteration);
        let mut tool_call_count = std::mem::take(ctx.tool_call_count);
        let mut personal_memory_tool_calls = *ctx.personal_memory_tool_calls;
        let mut no_evidence_result_streak = *ctx.no_evidence_result_streak;
        let mut no_evidence_tools_seen = std::mem::take(ctx.no_evidence_tools_seen);
        let mut evidence_gain_count = *ctx.evidence_gain_count;
        let mut pending_error_solution_ids = std::mem::take(ctx.pending_error_solution_ids);
        let mut tool_failure_patterns = std::mem::take(ctx.tool_failure_patterns);
        let mut last_tool_failure = std::mem::take(ctx.last_tool_failure);
        let mut in_session_learned = std::mem::take(ctx.in_session_learned);
        let mut unknown_tools = std::mem::take(ctx.unknown_tools);
        let mut recent_tool_calls = std::mem::take(ctx.recent_tool_calls);
        let mut consecutive_same_tool = std::mem::take(ctx.consecutive_same_tool);
        let mut consecutive_same_tool_arg_hashes =
            std::mem::take(ctx.consecutive_same_tool_arg_hashes);
        let mut force_text_response = *ctx.force_text_response;
        let mut pending_system_messages = std::mem::take(ctx.pending_system_messages);
        let mut recent_tool_names = std::mem::take(ctx.recent_tool_names);
        let mut successful_send_file_keys = std::mem::take(ctx.successful_send_file_keys);
        let mut cli_agent_boundary_injected = *ctx.cli_agent_boundary_injected;
        let mut pending_background_ack = std::mem::take(ctx.pending_background_ack);
        let mut stall_count = *ctx.stall_count;
        let mut deferred_no_tool_streak = *ctx.deferred_no_tool_streak;
        let mut consecutive_clean_iterations = *ctx.consecutive_clean_iterations;
        let mut fallback_expanded_once = *ctx.fallback_expanded_once;
        let mut known_project_dir = std::mem::take(ctx.known_project_dir);
        let mut dirs_with_project_inspect_file_evidence =
            std::mem::take(ctx.dirs_with_project_inspect_file_evidence);
        let mut dirs_with_search_no_matches = std::mem::take(ctx.dirs_with_search_no_matches);
        let mut require_file_recheck_before_answer = *ctx.require_file_recheck_before_answer;

        macro_rules! commit_state {
            () => {
                *ctx.tool_defs = tool_defs;
                *ctx.total_tool_calls_attempted = total_tool_calls_attempted;
                *ctx.total_successful_tool_calls = total_successful_tool_calls;
                *ctx.tool_failure_count = tool_failure_count;
                *ctx.tool_transient_failure_count = tool_transient_failure_count;
                *ctx.tool_cooldown_until_iteration = tool_cooldown_until_iteration;
                *ctx.tool_call_count = tool_call_count;
                *ctx.personal_memory_tool_calls = personal_memory_tool_calls;
                *ctx.no_evidence_result_streak = no_evidence_result_streak;
                *ctx.no_evidence_tools_seen = no_evidence_tools_seen;
                *ctx.evidence_gain_count = evidence_gain_count;
                *ctx.pending_error_solution_ids = pending_error_solution_ids;
                *ctx.tool_failure_patterns = tool_failure_patterns;
                *ctx.last_tool_failure = last_tool_failure;
                *ctx.in_session_learned = in_session_learned;
                *ctx.unknown_tools = unknown_tools;
                *ctx.recent_tool_calls = recent_tool_calls;
                *ctx.consecutive_same_tool = consecutive_same_tool;
                *ctx.consecutive_same_tool_arg_hashes = consecutive_same_tool_arg_hashes;
                *ctx.force_text_response = force_text_response;
                *ctx.pending_system_messages = pending_system_messages;
                *ctx.recent_tool_names = recent_tool_names;
                *ctx.successful_send_file_keys = successful_send_file_keys;
                *ctx.cli_agent_boundary_injected = cli_agent_boundary_injected;
                *ctx.pending_background_ack = pending_background_ack;
                *ctx.stall_count = stall_count;
                *ctx.deferred_no_tool_streak = deferred_no_tool_streak;
                *ctx.consecutive_clean_iterations = consecutive_clean_iterations;
                *ctx.fallback_expanded_once = fallback_expanded_once;
                *ctx.known_project_dir = known_project_dir;
                *ctx.dirs_with_project_inspect_file_evidence =
                    dirs_with_project_inspect_file_evidence;
                *ctx.dirs_with_search_no_matches = dirs_with_search_no_matches;
                *ctx.require_file_recheck_before_answer = require_file_recheck_before_answer;
            };
        }

        if known_project_dir.is_none() {
            known_project_dir = extract_project_dir_hint(_user_text);
        }

        let mut successful_tool_calls = 0;
        let mut iteration_had_tool_failures = false;
        for tc in &resp.tool_calls {
            let policy_tool_budget = policy_bundle.policy.tool_budget;
            if self.policy_config.policy_enforce
                && is_hard_policy_tool_budget_reached(
                    total_tool_calls_attempted,
                    policy_tool_budget,
                )
            {
                force_text_response = true;
                pending_system_messages.push(format!(
                    "[SYSTEM] Hard tool budget reached ({} calls). Stop calling tools and answer with the evidence already collected.",
                    policy_tool_budget
                ));
                self.emit_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::ToolBudgetBlock,
                    format!(
                        "Blocked tool {} because hard tool budget was reached",
                        tc.name
                    ),
                    json!({
                        "tool": tc.name,
                        "policy_tool_budget": policy_tool_budget,
                        "total_tool_calls_attempted": total_tool_calls_attempted,
                        "reason": "hard_policy_tool_budget_reached"
                    }),
                )
                .await;
                let result_text = format!(
                    "[SYSTEM] Hard tool budget reached: {} calls allowed per turn for this policy profile. \
                     This call to `{}` was blocked. Synthesize and answer now.",
                    policy_tool_budget, tc.name
                );
                let tool_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "tool".to_string(),
                    content: Some(result_text),
                    tool_call_id: Some(tc.id.clone()),
                    tool_name: Some(tc.name.clone()),
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.2,
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
                continue;
            }
            total_tool_calls_attempted = total_tool_calls_attempted.saturating_add(1);
            let send_file_key = if tc.name == "send_file" {
                extract_send_file_dedupe_key_from_args(&tc.arguments)
            } else {
                None
            };
            let is_personal_memory_tool_call = is_personal_memory_tool(&tc.name);

            if restrict_to_personal_memory_tools {
                if !is_personal_memory_tool_call {
                    let result_text = format!(
                            "[SYSTEM] Personal-memory recall should only use `manage_people` / `manage_memories` \
                             unless the user explicitly requested broader verification. \
                             Do not call `{}` for this query.",
                            tc.name
                        );
                    let tool_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "tool".to_string(),
                        content: Some(result_text),
                        tool_call_id: Some(tc.id.clone()),
                        tool_name: Some(tc.name.clone()),
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.1,
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
                    continue;
                }

                if personal_memory_tool_calls >= personal_memory_tool_call_cap {
                    force_text_response = true;
                    pending_system_messages.push(
                        "[SYSTEM] You already performed the allowed targeted memory re-check(s). \
                             Stop calling tools and answer directly with what you know."
                            .to_string(),
                    );
                    let result_text =
                            "Targeted personal-memory re-check limit reached. No further tool calls are allowed for this question."
                                .to_string();
                    let tool_msg = Message {
                        id: Uuid::new_v4().to_string(),
                        session_id: session_id.to_string(),
                        role: "tool".to_string(),
                        content: Some(result_text),
                        tool_call_id: Some(tc.id.clone()),
                        tool_name: Some(tc.name.clone()),
                        tool_calls_json: None,
                        created_at: Utc::now(),
                        importance: 0.2,
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
                    continue;
                }

                personal_memory_tool_calls = personal_memory_tool_calls.saturating_add(1);
            }

            let mut effective_arguments = tc.arguments.clone();
            let mut injected_project_dir: Option<String> = None;
            if let Some(explicit_dir) = project_dir_from_tool_args(&tc.name, &effective_arguments) {
                known_project_dir = Some(explicit_dir);
            }
            if let Some((updated_args, injected)) = maybe_inject_project_dir_into_tool_args(
                &tc.name,
                &effective_arguments,
                known_project_dir.as_deref(),
            ) {
                effective_arguments = updated_args;
                injected_project_dir = Some(injected.clone());
                let preserve_existing_target_scope = tc.name == "run_command"
                    && known_project_dir
                        .as_deref()
                        .is_some_and(|known| known != injected.as_str());
                if !preserve_existing_target_scope {
                    known_project_dir = Some(injected);
                }
            }
            let attempted_required_file_recheck = require_file_recheck_before_answer
                && is_file_recheck_tool(&tc.name)
                && tool_call_includes_project_path(&tc.name, &effective_arguments);

            let internal_scope_violation =
                raw_internal_scope_violation(&tc.arguments, session_id, resolved_goal_id);
            let allowed_project_scope = turn_context
                .primary_project_scope
                .as_deref()
                .or(known_project_dir.as_deref());
            let project_scope_violation = project_scope_violation_for_tool_call(
                &tc.name,
                &effective_arguments,
                allowed_project_scope,
                turn_context.allow_multi_project_scope,
            );
            if let Some(scope_reason) = internal_scope_violation.or(project_scope_violation) {
                POLICY_METRICS
                    .cross_scope_blocked_total
                    .fetch_add(1, Ordering::Relaxed);
                *tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;
                let result_text = format!(
                    "[SYSTEM] Scope lock blocked `{}`: {}. Continue with tools that stay inside the active request scope.",
                    tc.name, scope_reason
                );
                let tool_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "tool".to_string(),
                    content: Some(result_text.clone()),
                    tool_call_id: Some(tc.id.clone()),
                    tool_name: Some(tc.name.clone()),
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.2,
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
                pending_system_messages.push(format!(
                    "[SYSTEM] The previous `{}` tool call was blocked by deterministic scope locks ({}). Use paths/tool args aligned with the current request scope.",
                    tc.name, scope_reason
                ));
                continue;
            }

            if self
                .maybe_block_tool_by_budget(
                    tc,
                    &mut ToolBudgetBlockCtx {
                        emitter,
                        task_id,
                        session_id,
                        iteration,
                        tool_failure_count: &tool_failure_count,
                        tool_transient_failure_count: &tool_transient_failure_count,
                        tool_cooldown_until_iteration: &mut tool_cooldown_until_iteration,
                        tool_call_count: &tool_call_count,
                        unknown_tools: &unknown_tools,
                    },
                )
                .await?
            {
                continue;
            }

            // Budget/unknown-tool blocks are deterministic hard gates.
            // They must run BEFORE loop-pattern guards so blocked calls
            // do not inflate repetitive/same-tool counters and trigger
            // false "agent is looping" failures.
            if let Some(guard_outcome) = self
                .maybe_handle_loop_pattern_guards(
                    tc,
                    emitter,
                    task_id,
                    session_id,
                    iteration,
                    task_start,
                    task_tokens_used,
                    learning_ctx,
                    &mut recent_tool_calls,
                    &mut recent_tool_names,
                    &mut consecutive_same_tool,
                    &mut consecutive_same_tool_arg_hashes,
                )
                .await?
            {
                match guard_outcome {
                    LoopPatternGuardOutcome::ContinueLoop => continue,
                    LoopPatternGuardOutcome::Return(outcome) => {
                        commit_state!();
                        return Ok(outcome);
                    }
                }
            }

            if self
                .maybe_handle_duplicate_send_file_noop(
                    tc,
                    &mut DuplicateSendFileNoopCtx {
                        send_file_key: send_file_key.as_ref(),
                        successful_send_file_keys: &successful_send_file_keys,
                        session_id,
                        iteration,
                        effective_arguments: &effective_arguments,
                        successful_tool_calls: &mut successful_tool_calls,
                        total_successful_tool_calls: &mut total_successful_tool_calls,
                        tool_call_count: &mut tool_call_count,
                        learning_ctx,
                        emitter,
                        task_id,
                        policy_bundle,
                    },
                )
                .await?
            {
                continue;
            }

            let io = self
                .execute_tool_call_io(
                    tc,
                    &ToolExecutionIoCtx {
                        effective_arguments: &effective_arguments,
                        injected_project_dir: injected_project_dir.as_deref(),
                        session_id,
                        task_id,
                        status_tx: &status_tx,
                        channel_ctx,
                        user_role,
                        heartbeat,
                        emitter,
                        policy_bundle,
                    },
                )
                .await;
            let mut result_text = io.result_text;
            let mut tool_duration_ms = io.tool_duration_ms;
            if tc.name == "run_command" && run_command_policy_block_requires_terminal(&result_text)
            {
                if let Some(terminal_args) =
                    build_terminal_fallback_arguments_from_run_command(&effective_arguments)
                {
                    let fallback_started = Instant::now();
                    let terminal_result = self
                        .execute_tool_with_watchdog(
                            "terminal",
                            &terminal_args,
                            &tool_exec::ToolExecCtx {
                                session_id,
                                task_id: Some(task_id),
                                status_tx: status_tx.clone(),
                                channel_visibility: channel_ctx.visibility,
                                channel_id: channel_ctx.channel_id.as_deref(),
                                trusted: channel_ctx.trusted,
                                user_role,
                            },
                        )
                        .await;
                    let fallback_duration =
                        fallback_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                    tool_duration_ms = tool_duration_ms.saturating_add(fallback_duration);
                    let fallback_note =
                        "[SYSTEM] run_command was blocked by policy; auto-routed to `terminal`.";
                    result_text = match terminal_result {
                        Ok(text) => format!("{}\n\n{}", text, fallback_note),
                        Err(e) => format!("Error: {}\n\n{}", e, fallback_note),
                    };
                    if self.context_window_config.enabled {
                        result_text = crate::memory::context_window::compress_tool_result(
                            "terminal",
                            &result_text,
                            self.context_window_config.max_tool_result_chars,
                        );
                    }
                }
            }
            let background_detached =
                tool_result_indicates_background_detach(&tc.name, &result_text);

            if background_detached {
                pending_background_ack = Some(build_background_detach_ack(&tc.name, &result_text));
                force_text_response = true;
                let system_msg = "[SYSTEM] A background task is now running and completion notifications are enabled. \
Do NOT call additional tools or poll status in this turn. Reply to the user now that work continues in background and results will be sent automatically."
                    .to_string();
                pending_system_messages.push(system_msg.clone());
                result_text = format!("{}\n\n{}", result_text, system_msg);
            }

            // Track total calls per tool
            *tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;

            // Track tool call for learning
            let tool_summary = format!(
                "{}({})",
                tc.name,
                summarize_tool_args(&tc.name, &effective_arguments)
            );
            learning_ctx.tool_calls.push(tool_summary.clone());

            // Track tool failures across iterations using structured detection
            // (prefixes, JSON error payloads, HTTP statuses, non-zero exit codes).
            let failure_class = classify_tool_result_failure(&tc.name, &result_text);
            let is_error = failure_class.is_some();

            let learning_env = ResultLearningEnv {
                attempted_required_file_recheck,
                send_file_key,
                restrict_to_personal_memory_tools,
                is_reaffirmation_challenge_turn,
                session_id,
                task_id,
                emitter,
                task_start,
                iteration,
                tool_summary: &tool_summary,
            };
            let mut learning_state = ResultLearningState {
                learning_ctx,
                no_evidence_result_streak: &mut no_evidence_result_streak,
                iteration_had_tool_failures: &mut iteration_had_tool_failures,
                no_evidence_tools_seen: &mut no_evidence_tools_seen,
                evidence_gain_count: &mut evidence_gain_count,
                unknown_tools: &mut unknown_tools,
                tool_failure_count: &mut tool_failure_count,
                tool_transient_failure_count: &mut tool_transient_failure_count,
                tool_cooldown_until_iteration: &mut tool_cooldown_until_iteration,
                pending_error_solution_ids: &mut pending_error_solution_ids,
                tool_failure_patterns: &mut tool_failure_patterns,
                last_tool_failure: &mut last_tool_failure,
                in_session_learned: &mut in_session_learned,
                force_text_response: &mut force_text_response,
                pending_system_messages: &mut pending_system_messages,
                successful_tool_calls: &mut successful_tool_calls,
                total_successful_tool_calls: &mut total_successful_tool_calls,
                successful_send_file_keys: &mut successful_send_file_keys,
                cli_agent_boundary_injected: &mut cli_agent_boundary_injected,
                recent_tool_calls: &mut recent_tool_calls,
                consecutive_same_tool: &mut consecutive_same_tool,
                consecutive_same_tool_arg_hashes: &mut consecutive_same_tool_arg_hashes,
                recent_tool_names: &mut recent_tool_names,
                require_file_recheck_before_answer: &mut require_file_recheck_before_answer,
                known_project_dir: &mut known_project_dir,
                dirs_with_project_inspect_file_evidence:
                    &mut dirs_with_project_inspect_file_evidence,
                dirs_with_search_no_matches: &mut dirs_with_search_no_matches,
            };
            if let Some(outcome) = self
                .apply_result_learning(
                    tc,
                    &mut result_text,
                    is_error,
                    failure_class,
                    &learning_env,
                    &mut learning_state,
                )
                .await?
            {
                commit_state!();
                return Ok(outcome);
            }

            let tool_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
                role: "tool".to_string(),
                content: Some(result_text.clone()),
                tool_call_id: Some(tc.id.clone()),
                tool_name: Some(tc.name.clone()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.3, // Tool outputs default to lower importance
                embedding: None,
            };
            self.append_tool_message_with_result_event(
                emitter,
                &tool_msg,
                !is_error,
                tool_duration_ms,
                if is_error {
                    Some(result_text.clone())
                } else {
                    None
                },
                Some(task_id),
            )
            .await?;

            // Emit Error event if tool failed
            if is_error {
                let _ = emitter
                    .emit(
                        EventType::Error,
                        ErrorData::tool_error(
                            tc.name.clone(),
                            result_text.clone(),
                            Some(task_id.to_string()),
                        ),
                    )
                    .await;
            }

            // Log tool activity for executor agents
            if let Some(ref tid) = self.task_id {
                let activity = TaskActivity {
                    id: 0,
                    task_id: tid.clone(),
                    activity_type: "tool_call".to_string(),
                    tool_name: Some(tc.name.clone()),
                    tool_args: Some(effective_arguments.chars().take(1000).collect()),
                    result: Some(result_text.chars().take(2000).collect()),
                    success: Some(!is_error),
                    tokens_used: None,
                    created_at: chrono::Utc::now().to_rfc3339(),
                };
                if let Err(e) = self.state.log_task_activity(&activity).await {
                    warn!(task_id = %tid, error = %e, "Failed to log task activity");
                }
            }

            if background_detached {
                info!(
                    session_id,
                    iteration,
                    tool = %tc.name,
                    "Background task detached; ending tool execution phase early and forcing text response"
                );
                break;
            }
        }

        self.apply_post_tool_iteration_controls(
            session_id,
            iteration,
            task_tokens_used,
            successful_tool_calls,
            iteration_had_tool_failures,
            restrict_to_personal_memory_tools,
            base_tool_defs,
            available_capabilities,
            policy_bundle,
            total_tool_calls_attempted,
            resolved_goal_id.is_some(),
            &mut total_successful_tool_calls,
            &mut force_text_response,
            &mut pending_system_messages,
            &mut tool_defs,
            &mut stall_count,
            &mut deferred_no_tool_streak,
            &mut consecutive_clean_iterations,
            &mut fallback_expanded_once,
        );
        commit_state!();
        Ok(ToolExecutionOutcome::NextIteration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_scope_violation_detects_session_mismatch() {
        let raw = r#"{"_session_id":"other-session"}"#;
        let violation = raw_internal_scope_violation(raw, "expected-session", None);
        assert!(violation.is_some());
        let message = violation.unwrap_or_default();
        assert!(message.contains("_session_id mismatch"));
    }

    #[test]
    fn internal_scope_violation_detects_goal_mismatch() {
        let raw = r#"{"_goal_id":"goal-2"}"#;
        let violation = raw_internal_scope_violation(raw, "s", Some("goal-1"));
        assert!(violation.is_some());
        let message = violation.unwrap_or_default();
        assert!(message.contains("_goal_id mismatch"));
    }

    #[test]
    fn project_scope_violation_flags_out_of_scope_path() {
        let args = r#"{"path":"/tmp/project-b/src"}"#;
        let violation = project_scope_violation_for_tool_call(
            "search_files",
            args,
            Some("/tmp/project-a"),
            false,
        );
        assert!(violation.is_some());
    }

    #[test]
    fn project_scope_violation_allows_multi_project_requests() {
        let args = r#"{"path":"/tmp/project-b/src"}"#;
        let violation = project_scope_violation_for_tool_call(
            "search_files",
            args,
            Some("/tmp/project-a"),
            true,
        );
        assert!(violation.is_none());
    }

    #[test]
    fn run_command_policy_block_requires_terminal_detects_policy_errors() {
        assert!(run_command_policy_block_requires_terminal(
            "Error: Command 'npm install' is not in the safe command list for run_command. Use 'terminal' for this command."
        ));
        assert!(!run_command_policy_block_requires_terminal(
            "$ cargo test (exit: 0, 22ms)"
        ));
    }

    #[test]
    fn build_terminal_fallback_arguments_preserves_working_dir() {
        let args = r#"{"command":"npm create vite@latest whatsapp-site -- --template react","working_dir":"/tmp/my folder"}"#;
        let terminal_args = build_terminal_fallback_arguments_from_run_command(args)
            .expect("fallback args expected");
        let parsed: Value = serde_json::from_str(&terminal_args).expect("valid json");
        assert_eq!(parsed["action"], "run");
        assert_eq!(
            parsed["command"],
            "cd '/tmp/my folder' && npm create vite@latest whatsapp-site -- --template react"
        );
    }

    #[test]
    fn build_terminal_fallback_arguments_escapes_single_quotes() {
        let args = r#"{"command":"npm create vite@latest whatsapp-site -- --template react","working_dir":"/tmp/david's projects"}"#;
        let terminal_args = build_terminal_fallback_arguments_from_run_command(args)
            .expect("fallback args expected");
        let parsed: Value = serde_json::from_str(&terminal_args).expect("valid json");
        assert_eq!(
            parsed["command"],
            "cd '/tmp/david'\"'\"'s projects' && npm create vite@latest whatsapp-site -- --template react"
        );
    }

    #[test]
    fn project_scope_violation_allows_run_command_parent_dir_for_new_project_scaffolding() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let parent = tmp.path().join("projects");
        std::fs::create_dir_all(&parent).expect("create parent");
        let target = parent.join("new-site");
        let args = format!(
            r#"{{"command":"pwd","working_dir":"{}"}}"#,
            parent.to_string_lossy()
        );
        let violation = project_scope_violation_for_tool_call(
            "run_command",
            &args,
            Some(target.to_string_lossy().as_ref()),
            false,
        );
        assert!(violation.is_none());
    }

    #[test]
    fn hard_policy_tool_budget_reached_when_attempts_hit_limit() {
        assert!(is_hard_policy_tool_budget_reached(6, 6));
        assert!(is_hard_policy_tool_budget_reached(7, 6));
        assert!(!is_hard_policy_tool_budget_reached(5, 6));
        assert!(!is_hard_policy_tool_budget_reached(10, 0));
    }

    #[test]
    fn detects_background_detach_markers_for_supported_tools() {
        assert!(tool_result_indicates_background_detach(
            "terminal",
            "Command still running after 30s. Moved to background (pid=123)."
        ));
        assert!(tool_result_indicates_background_detach(
            "cli_agent",
            "CLI agent 'x' started in background (task_id=abc)."
        ));
        assert!(tool_result_indicates_background_detach(
            "spawn_agent",
            "Sub-agent spawned in background for mission: \"...\""
        ));
        assert!(tool_result_indicates_background_detach(
            "web_search",
            "Moved to background (pid=1)"
        ));
        assert!(!tool_result_indicates_background_detach(
            "terminal",
            "Process finished normally."
        ));
    }

    #[test]
    fn builds_deterministic_background_ack_from_tool_result() {
        let ack = build_background_detach_ack(
            "terminal",
            "Command still running after 30s. Moved to background (pid=123).\n\n[SYSTEM] ...",
        );
        assert!(ack.contains("Moved to background (pid=123)"));
        assert!(ack.contains("final result will be sent automatically"));
    }
}
