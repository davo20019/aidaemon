use super::budget_blocking::{DuplicateSendFileNoopCtx, ToolBudgetBlockCtx};
use super::execution_io::ToolExecutionIoCtx;
use super::guards::LoopPatternGuardOutcome;
use super::project_dir::{
    extract_project_dir_hint, is_file_recheck_tool, maybe_inject_project_dir_into_tool_args,
    project_dir_from_tool_args, tool_call_includes_project_path,
};
use super::result_learning::{ResultLearningEnv, ResultLearningState};
use super::types::{ToolExecutionCtx, ToolExecutionOutcome};
use crate::agent::recall_guardrails::is_personal_memory_tool;
use crate::agent::*;

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

        let mut tool_defs = std::mem::take(ctx.tool_defs);
        let mut total_tool_calls_attempted = *ctx.total_tool_calls_attempted;
        let mut total_successful_tool_calls = *ctx.total_successful_tool_calls;
        let mut tool_failure_count = std::mem::take(ctx.tool_failure_count);
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
                known_project_dir = Some(injected);
            }
            let attempted_required_file_recheck = require_file_recheck_before_answer
                && is_file_recheck_tool(&tc.name)
                && tool_call_includes_project_path(&tc.name, &effective_arguments);

            if self
                .maybe_block_tool_by_budget(
                    tc,
                    &ToolBudgetBlockCtx {
                        emitter,
                        task_id,
                        session_id,
                        iteration,
                        tool_failure_count: &tool_failure_count,
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
            let tool_duration_ms = io.tool_duration_ms;

            // Track total calls per tool
            *tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;

            // Track tool call for learning
            let tool_summary = format!(
                "{}({})",
                tc.name,
                summarize_tool_args(&tc.name, &effective_arguments)
            );
            learning_ctx.tool_calls.push(tool_summary.clone());

            // Track tool failures across iterations (actual errors only)
            let is_error = result_text.starts_with("ERROR:")
                || result_text.starts_with("Error:")
                || result_text.starts_with("Failed to ");

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
