use super::project_dir::{
    extract_project_dir_from_project_inspect_output, extract_search_files_scanned_dir,
    project_inspect_reports_file_entries, search_files_result_no_matches,
};
use super::reflection::{record_tool_error, PendingReflectionRecovery, SemanticFailureInfo};
use super::types::ToolExecutionOutcome;
use crate::agent::loop_utils;
use crate::agent::recall_guardrails::tool_result_indicates_no_evidence;
use crate::agent::*;
use once_cell::sync::Lazy;
use regex::Regex;

fn append_tool_result_notice(result_text: &mut String, notice: &ToolResultNotice) {
    result_text.push_str("\n\n");
    result_text.push_str(&notice.render());
}

pub(super) struct ResultLearningEnv<'a> {
    pub attempted_required_file_recheck: bool,
    pub send_file_key: Option<String>,
    pub restrict_to_personal_memory_tools: bool,
    pub is_reaffirmation_challenge_turn: bool,
    pub session_id: &'a str,
    pub task_id: &'a str,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_start: Instant,
    pub iteration: usize,
    pub tool_arguments: &'a str,
    pub tool_summary: &'a str,
}

pub(super) struct ResultLearningState<'a> {
    pub learning_ctx: &'a mut LearningContext,
    pub no_evidence_result_streak: &'a mut usize,
    pub iteration_had_tool_failures: &'a mut bool,
    pub no_evidence_tools_seen: &'a mut HashSet<String>,
    pub evidence_gain_count: &'a mut usize,
    pub unknown_tools: &'a mut HashSet<String>,
    pub tool_failure_count: &'a mut HashMap<String, usize>,
    pub tool_failure_signatures: &'a mut HashMap<(String, String), usize>,
    pub tool_transient_failure_count: &'a mut HashMap<String, usize>,
    pub tool_cooldown_until_iteration: &'a mut HashMap<String, usize>,
    pub pending_error_solution_ids: &'a mut Vec<i64>,
    pub tool_error_history:
        &'a mut HashMap<(String, String), Vec<super::reflection::ToolErrorEntry>>,
    pub pending_reflection_recoveries: &'a mut HashMap<String, PendingReflectionRecovery>,
    pub tool_failure_patterns: &'a mut HashMap<(String, String), usize>,
    pub last_tool_failure: &'a mut Option<(String, String)>,
    pub in_session_learned: &'a mut HashSet<(String, String)>,
    pub force_text_response: &'a mut bool,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub successful_tool_calls: &'a mut usize,
    pub total_successful_tool_calls: &'a mut usize,
    pub successful_send_file_keys: &'a mut HashSet<String>,
    pub cli_agent_boundary_injected: &'a mut bool,
    pub recent_tool_calls: &'a mut VecDeque<u64>,
    pub consecutive_same_tool: &'a mut (String, usize),
    pub consecutive_same_tool_arg_hashes: &'a mut HashSet<u64>,
    pub recent_tool_names: &'a mut VecDeque<String>,
    pub require_file_recheck_before_answer: &'a mut bool,
    pub known_project_dir: &'a mut Option<String>,
    pub dirs_with_project_inspect_file_evidence: &'a mut HashSet<String>,
    pub dirs_with_search_no_matches: &'a mut HashSet<String>,
}

pub(super) struct ResultLearningOutcome {
    pub control_flow: Option<ToolExecutionOutcome>,
    pub semantic_failure: Option<SemanticFailureInfo>,
}

fn cli_result_is_substantive(result_text: &str) -> bool {
    let cleaned = strip_appended_diagnostics(result_text).trim().to_string();
    !cleaned.is_empty()
        && cleaned.chars().count() >= 500
        && !cleaned.to_ascii_lowercase().contains("error:")
        && !cleaned.to_ascii_lowercase().contains("failed")
}

fn looks_like_missing_goal_id_error(tool_name: &str, error_text: &str) -> bool {
    if !matches!(
        tool_name,
        "scheduled_goal_runs" | "manage_memories" | "goal_trace" | "tool_trace"
    ) {
        return false;
    }
    let lower = error_text.to_ascii_lowercase();
    lower.contains("'goal_id' is required")
        || lower.contains("\"goal_id\" is required")
        || lower.contains("goal_id is required")
}

fn user_looks_like_fact_storage_request(user_text: &str) -> bool {
    let lower = user_text.to_ascii_lowercase();
    contains_keyword_as_words(&lower, "learn this")
        || contains_keyword_as_words(&lower, "learn these")
        || contains_keyword_as_words(&lower, "remember this")
        || contains_keyword_as_words(&lower, "remember these")
        || contains_keyword_as_words(&lower, "remember the following")
        || contains_keyword_as_words(&lower, "remember this for later")
        || contains_keyword_as_words(&lower, "remember these for later")
        || contains_keyword_as_words(&lower, "store this")
        || contains_keyword_as_words(&lower, "store these")
        || contains_keyword_as_words(&lower, "save this")
        || contains_keyword_as_words(&lower, "save these")
        || contains_keyword_as_words(&lower, "note this down")
        || contains_keyword_as_words(&lower, "note these down")
        || contains_keyword_as_words(&lower, "keep in mind")
        || contains_keyword_as_words(&lower, "i need you to know")
}

static ABS_UNIX_PATH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"/[^\s"'`)\]}\{:,;]+"#).expect("absolute unix path regex must compile")
});
static ABS_WINDOWS_PATH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"[A-Z]:[\\/][^\s"'`)\]}\{:,;]+"#)
        .expect("absolute windows path regex must compile")
});
static LINE_COL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r":\d+:\d+").expect("line:column regex must compile"));
static ATTEMPT_NUM_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\battempt\s+\d+\b").expect("attempt regex must compile"));
static PID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\bpid\s*[=:]\s*\d+\b").expect("pid regex must compile"));
static EXIT_CODE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bexit(?:\s+code)?\s*[:=]?\s*-?\d+\b").expect("exit code regex must compile")
});
static WHITESPACE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s+").expect("whitespace regex must compile"));

fn path_tail_for_signature(path: &str) -> String {
    let tail = path
        .trim_matches(|ch| matches!(ch, '"' | '\'' | '`'))
        .rsplit(['/', '\\'])
        .find(|segment| !segment.is_empty())
        .unwrap_or(path);
    format!("<path:{}>", tail)
}

fn normalize_error_line_for_signature(line: &str) -> String {
    let mut normalized = line.to_ascii_lowercase();
    normalized = LINE_COL_RE.replace_all(&normalized, ":<line>").to_string();
    normalized = ATTEMPT_NUM_RE
        .replace_all(&normalized, "attempt <n>")
        .to_string();
    normalized = PID_RE.replace_all(&normalized, "pid=<n>").to_string();
    normalized = EXIT_CODE_RE
        .replace_all(&normalized, "exit <n>")
        .to_string();
    normalized = ABS_UNIX_PATH_RE
        .replace_all(&normalized, |caps: &regex::Captures<'_>| {
            path_tail_for_signature(caps.get(0).map(|m| m.as_str()).unwrap_or_default())
        })
        .to_string();
    normalized = ABS_WINDOWS_PATH_RE
        .replace_all(&normalized, |caps: &regex::Captures<'_>| {
            path_tail_for_signature(caps.get(0).map(|m| m.as_str()).unwrap_or_default())
        })
        .to_string();
    WHITESPACE_RE
        .replace_all(&normalized, " ")
        .trim()
        .to_string()
}

fn should_skip_transient_cooldown(tool_name: &str, error_text: &str) -> bool {
    let lower = error_text.to_ascii_lowercase();
    loop_utils::is_file_lookup_miss_for_tool(tool_name, &lower)
}

fn derive_failure_signature(error_text: &str) -> String {
    let key_line = extract_key_error_line(error_text);
    let normalized = normalize_error_line_for_signature(&key_line);
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        return "unclassified error".to_string();
    }
    trimmed.chars().take(160).collect()
}

fn record_semantic_failure_signature(
    tool_failure_count: &mut HashMap<String, usize>,
    tool_failure_signatures: &mut HashMap<(String, String), usize>,
    tool_name: &str,
    error_text: &str,
) -> SemanticFailureInfo {
    let signature = derive_failure_signature(error_text);
    let count = tool_failure_signatures
        .entry((tool_name.to_string(), signature.clone()))
        .or_insert(0);
    *count += 1;
    let repeated_count = *count;
    let per_tool = tool_failure_count
        .entry(tool_name.to_string())
        .or_insert(repeated_count);
    if repeated_count > *per_tool {
        *per_tool = repeated_count;
    }
    SemanticFailureInfo {
        signature,
        count: repeated_count,
    }
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn apply_result_learning(
        &self,
        tc: &ToolCall,
        result_text: &mut String,
        is_error: bool,
        failure_class: Option<ToolFailureClass>,
        execution_failure_kind: Option<ExecutionFailureKind>,
        env: &ResultLearningEnv<'_>,
        state: &mut ResultLearningState<'_>,
    ) -> anyhow::Result<ResultLearningOutcome> {
        if is_error {
            *state.no_evidence_result_streak = 0;
            *state.iteration_had_tool_failures = true;

            // Track hallucinated tool names so they're blocked on next attempt
            if result_text.contains("Unknown tool '") {
                state.unknown_tools.insert(tc.name.clone());
            }

            let base_error = strip_appended_diagnostics(result_text).to_string();
            let failure_class = failure_class.unwrap_or(ToolFailureClass::Semantic);
            let mut semantic_failure = None;
            match execution_failure_kind {
                Some(ExecutionFailureKind::ToolContractFailure) => append_tool_result_notice(
                    result_text,
                    &ToolResultNotice::ToolContractFailureRetry {
                        tool_name: tc.name.clone(),
                    },
                ),
                Some(ExecutionFailureKind::EnvironmentFailure) => append_tool_result_notice(
                    result_text,
                    &ToolResultNotice::EnvironmentFailureGuidance {
                        tool_name: tc.name.clone(),
                    },
                ),
                Some(ExecutionFailureKind::LogicFailure) => append_tool_result_notice(
                    result_text,
                    &ToolResultNotice::LogicFailureReplan {
                        tool_name: tc.name.clone(),
                    },
                ),
                Some(ExecutionFailureKind::ToolInvocationFailure) | None => {}
            }
            if matches!(failure_class, ToolFailureClass::Transient) {
                state.pending_error_solution_ids.clear();
                let transient_count = state
                    .tool_transient_failure_count
                    .entry(tc.name.clone())
                    .or_insert(0);
                *transient_count += 1;
                if should_skip_transient_cooldown(&tc.name, &base_error) {
                    state.tool_cooldown_until_iteration.remove(&tc.name);
                    append_tool_result_notice(
                        result_text,
                        &ToolResultNotice::RecoverableFilePathMiss {
                            tool_name: tc.name.clone(),
                        },
                    );
                } else {
                    let cooldown_iters = 2usize;
                    let cooldown_until = env.iteration.saturating_add(cooldown_iters);
                    state
                        .tool_cooldown_until_iteration
                        .insert(tc.name.clone(), cooldown_until);
                    append_tool_result_notice(
                        result_text,
                        &ToolResultNotice::TransientFailureCooldown {
                            tool_name: tc.name.clone(),
                            cooldown_until,
                            cooldown_iters,
                        },
                    );
                }
            } else {
                let failure = record_semantic_failure_signature(
                    state.tool_failure_count,
                    state.tool_failure_signatures,
                    &tc.name,
                    &base_error,
                );
                let semantic_count = failure.count;
                semantic_failure = Some(failure.clone());
                clear_pending_reflection_recovery_on_failure(
                    state.pending_reflection_recoveries,
                    &tc.name,
                    &failure.signature,
                    env.iteration,
                );
                record_tool_error(
                    state.tool_error_history,
                    &tc.name,
                    &failure.signature,
                    env.iteration,
                    env.tool_arguments,
                    &base_error,
                );

                if semantic_count == 1 && looks_like_missing_goal_id_error(&tc.name, &base_error) {
                    let likely_fact_storage =
                        user_looks_like_fact_storage_request(&state.learning_ctx.user_text);
                    let coach = if likely_fact_storage {
                        ToolResultNotice::OffTargetFactStorageRequest
                    } else if tc.name == "manage_memories" {
                        ToolResultNotice::MissingGoalIdManageMemories
                    } else {
                        ToolResultNotice::MissingGoalIdGeneric
                    };
                    append_tool_result_notice(result_text, &coach);
                }

                // write_file JSON escaping recovery: when write_file fails with
                // JSON parsing errors, the LLM's tool call JSON had improperly escaped
                // content (common with code containing backslashes like JSON parsers,
                // regex engines, etc.). Direct the LLM to use terminal heredoc instead.
                if tc.name == "write_file"
                    && (base_error.contains("EOF while parsing")
                        || base_error.contains("trailing characters")
                        || base_error.contains("invalid escape")
                        || base_error.contains("control character"))
                {
                    let write_path = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                        .ok()
                        .and_then(|v| {
                            v.get("path")
                                .and_then(|p| p.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| "<target file>".to_string());
                    append_tool_result_notice(
                        result_text,
                        &ToolResultNotice::WriteFileJsonRecovery { path: write_path },
                    );
                }

                if tc.name == "edit_file" {
                    let edit_path = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                        .ok()
                        .and_then(|v| {
                            v.get("path")
                                .and_then(|p| p.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| "<same file>".to_string());
                    // edit_file JSON escaping recovery: when edit_file fails with
                    // JSON parsing errors, the old_text/new_text content broke the
                    // tool call JSON (backslashes, quotes, unicode escapes).
                    // Direct to terminal sed for small fixes, or heredoc for rewrites.
                    if base_error.contains("EOF while parsing")
                        || base_error.contains("trailing characters")
                        || base_error.contains("invalid escape")
                        || base_error.contains("control character")
                    {
                        append_tool_result_notice(
                            result_text,
                            &ToolResultNotice::EditFileJsonRecovery {
                                path: edit_path.clone(),
                            },
                        );
                    } else if base_error.contains("Text not found in ") {
                        append_tool_result_notice(
                            result_text,
                            &ToolResultNotice::EditFileTextNotFoundRecovery {
                                path: edit_path.clone(),
                            },
                        );
                    } else if base_error.contains("Set replace_all=true")
                        || base_error.contains("occurrences of the text")
                    {
                        append_tool_result_notice(
                            result_text,
                            &ToolResultNotice::EditFileReplaceAllRecovery {
                                path: edit_path.clone(),
                            },
                        );
                    }
                }
                let err_pattern = crate::memory::procedures::extract_error_pattern(&base_error);
                if !err_pattern.trim().is_empty() {
                    let key = (tc.name.clone(), err_pattern.clone());
                    let pattern_count = state.tool_failure_patterns.entry(key.clone()).or_insert(0);
                    *pattern_count += 1;
                    *state.last_tool_failure = Some(key);

                    // Persist repeated dead-end workflows as explicit failure patterns.
                    if *pattern_count >= 3 {
                        let state_store = self.state.clone();
                        let tool_name = tc.name.clone();
                        let error_pattern = err_pattern.clone();
                        let observed_count = *pattern_count;
                        tokio::spawn(async move {
                            let description = format!(
                                "Repeated {} failures for {} on '{}'; pivot to a different approach earlier.",
                                observed_count, tool_name, error_pattern
                            );
                            let confidence = (0.5 + (observed_count as f32 * 0.05)).min(0.9);
                            if let Err(e) = state_store
                                .record_behavior_pattern(
                                    "failure",
                                    &description,
                                    Some(&tool_name),
                                    Some("pivot to alternate tool/strategy"),
                                    confidence,
                                    1,
                                )
                                .await
                            {
                                warn!(
                                    tool = %tool_name,
                                    error = %e,
                                    "Failed to record failure behavior pattern"
                                );
                            }
                        });
                    }
                }

                // DIAGNOSTIC LOOP: On first semantic failure, query memory for similar errors.
                if semantic_count == 1 {
                    state.pending_error_solution_ids.clear();
                    if let Ok(solutions) = self
                        .state
                        .get_relevant_error_solutions(&base_error, 3)
                        .await
                    {
                        if !solutions.is_empty() {
                            *state.pending_error_solution_ids =
                                solutions.first().map(|s| s.id).into_iter().collect();
                            let diagnostic_hints: Vec<String> = solutions
                                .iter()
                                .map(|s| {
                                    if let Some(ref steps) = s.solution_steps {
                                        format!(
                                            "- {}\n  Steps: {}",
                                            s.solution_summary,
                                            steps.join(" -> ")
                                        )
                                    } else {
                                        format!("- {}", s.solution_summary)
                                    }
                                })
                                .collect();
                            *result_text = format!(
                                "{}\n\n[DIAGNOSTIC] Similar errors resolved before:\n{}",
                                result_text,
                                diagnostic_hints.join("\n")
                            );
                            info!(
                                tool = %tc.name,
                                solutions_found = solutions.len(),
                                "Diagnostic loop: injected error solutions"
                            );
                        }
                    }

                    // Inline tool failure stats to help the model decide whether to retry or pivot.
                    // Bounded query (LIMIT 500) and guarded for graceful degradation.
                    let since = Utc::now() - chrono::Duration::hours(24);
                    if let Ok(stats) = self.event_store.get_tool_stats(&tc.name, since).await {
                        if stats.total_calls >= 3 {
                            // total_calls is bounded by the underlying LIMIT and guarded above.
                            let failure_pct = ((stats.failed * 100) + (stats.total_calls / 2))
                                / stats.total_calls;
                            let mut lines = Vec::new();
                            lines.push(format!(
                                "[TOOL STATS] {} (24h): {} calls, {} failed ({}%), avg {}ms",
                                tc.name,
                                stats.total_calls,
                                stats.failed,
                                failure_pct,
                                stats.avg_duration_ms
                            ));
                            for (pattern, count) in stats.common_errors.into_iter().take(2) {
                                let limit = 100usize;
                                let head: String = pattern.chars().take(limit).collect();
                                let preview = if pattern.chars().count() > limit {
                                    format!("{}...", head)
                                } else {
                                    head
                                };
                                lines.push(format!("  - {}x: {}", count, preview));
                            }
                            *result_text = format!("{}\n\n{}", result_text, lines.join("\n"));
                        }
                    }

                    // Explicit error-reading coaching: quote the key error back to the LLM
                    // so it can't miss it, and tell it to adapt.
                    let key_line = extract_key_error_line(&base_error);
                    if let Some(coaching) = format_error_coaching(&key_line) {
                        append_tool_result_notice(result_text, &coaching);
                    }
                }

                if semantic_count >= 2 {
                    // Hint was shown but the same tool failed again: record a miss.
                    if let Some(solution_id) = state.pending_error_solution_ids.first().copied() {
                        state.pending_error_solution_ids.clear();
                        let state_store = self.state.clone();
                        tokio::spawn(async move {
                            if let Err(e) = state_store
                                .update_error_solution_outcome(solution_id, false)
                                .await
                            {
                                warn!(
                                    solution_id,
                                    error = %e,
                                    "Failed to record error solution failure"
                                );
                            }
                        });
                    }

                    let key_line = extract_key_error_line(&base_error);
                    let coaching = format_semantic_failure_coaching(semantic_count, &key_line);
                    append_tool_result_notice(result_text, &coaching);
                }
            }

            // Track error for learning
            if state.learning_ctx.first_error.is_none() {
                state.learning_ctx.first_error = Some(base_error);
            }
            state.learning_ctx.errors.push((result_text.clone(), false));
            return Ok(ResultLearningOutcome {
                control_flow: None,
                semantic_failure,
            });
        }

        if env.attempted_required_file_recheck {
            *state.require_file_recheck_before_answer = false;
            append_tool_result_notice(result_text, &ToolResultNotice::RequiredFileRecheckCompleted);
        }

        if !env.attempted_required_file_recheck {
            let mut contradiction_dir: Option<String> = None;
            if tc.name == "project_inspect" {
                if let Some(dir) = extract_project_dir_from_project_inspect_output(result_text) {
                    *state.known_project_dir = Some(dir.clone());
                    if project_inspect_reports_file_entries(result_text) {
                        state
                            .dirs_with_project_inspect_file_evidence
                            .insert(dir.clone());
                        if state.dirs_with_search_no_matches.contains(&dir) {
                            contradiction_dir = Some(dir);
                        }
                    }
                }
            } else if tc.name == "search_files" {
                if let Some(dir) = extract_search_files_scanned_dir(result_text) {
                    *state.known_project_dir = Some(dir.clone());
                    if search_files_result_no_matches(result_text) {
                        state.dirs_with_search_no_matches.insert(dir.clone());
                        if state.dirs_with_project_inspect_file_evidence.contains(&dir) {
                            contradiction_dir = Some(dir);
                        }
                    }
                }
            }
            if let Some(dir) = contradiction_dir {
                *state.require_file_recheck_before_answer = true;
                let guardrail =
                    SystemDirective::ContradictoryFileEvidenceExplicitPath { dir: dir.clone() };
                state.pending_system_messages.push(guardrail.clone());
                *result_text = format!("{}\n\n{}", result_text, guardrail.render());
            }
        }

        let no_evidence_result =
            tool_result_indicates_no_evidence(&strip_appended_diagnostics(result_text));
        if no_evidence_result {
            *state.no_evidence_result_streak = state.no_evidence_result_streak.saturating_add(1);
            state.no_evidence_tools_seen.insert(tc.name.clone());
        } else {
            *state.no_evidence_result_streak = 0;
            state.no_evidence_tools_seen.clear();
            *state.evidence_gain_count = state.evidence_gain_count.saturating_add(1);
        }

        if env.restrict_to_personal_memory_tools
            && (env.is_reaffirmation_challenge_turn
                || state.no_evidence_tools_seen.len() >= 2
                || *state.no_evidence_result_streak >= 3)
            && no_evidence_result
        {
            let reaffirmation = if env.is_reaffirmation_challenge_turn {
                "I checked again in your stored people/memory records, and I still do not have that information saved. If you want, share it and I will remember it.".to_string()
            } else {
                "I checked your stored people/memory records, and I do not have that information saved yet. If you share it, I can remember it for next time.".to_string()
            };
            let assistant_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: env.session_id.to_string(),
                role: "assistant".to_string(),
                content: Some(reaffirmation.clone()),
                tool_call_id: None,
                tool_name: None,
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            };
            self.append_assistant_message_with_event(
                env.emitter,
                &assistant_msg,
                "system",
                None,
                None,
            )
            .await?;
            self.emit_task_end(
                env.emitter,
                env.task_id,
                TaskStatus::Completed,
                env.task_start,
                env.iteration,
                state.learning_ctx.tool_calls.len(),
                None,
                Some(reaffirmation.chars().take(200).collect()),
            )
            .await;
            return Ok(ResultLearningOutcome {
                control_flow: Some(ToolExecutionOutcome::Return(Ok(reaffirmation))),
                semantic_failure: None,
            });
        }

        if !env.restrict_to_personal_memory_tools
            && no_evidence_result
            && state.no_evidence_tools_seen.len() >= 5
        {
            *state.force_text_response = true;
            state
                .pending_system_messages
                .push(SystemDirective::NoEvidenceRespondKnownUnknown);
        }

        *state.successful_tool_calls += 1;
        *state.total_successful_tool_calls += 1;
        if tc.name == "send_file" {
            if let Some(key) = env.send_file_key.as_ref().cloned() {
                state.successful_send_file_keys.insert(key);
            }
            // Strongly bias the model to finish immediately after a
            // successful file delivery instead of continuing to
            // explore and risking follow-up path drift errors.
            append_tool_result_notice(
                result_text,
                &ToolResultNotice::SendFileSucceededStopAndReply,
            );
        }

        if tc.name != "cli_agent" {
            *state.cli_agent_boundary_injected = false;
        }

        // After a cli_agent call completes, reset stall detection
        // counters — the follow-up work (e.g. git push, deploy) is
        // a fresh phase and shouldn't inherit stall state.
        if tc.name == "cli_agent" {
            state.recent_tool_calls.clear();
            *state.consecutive_same_tool = (String::new(), 0);
            state.consecutive_same_tool_arg_hashes.clear();
            state.recent_tool_names.clear();

            if cli_result_is_substantive(result_text) {
                let present_results_msg = SystemDirective::CliAgentPresentResults;
                state
                    .pending_system_messages
                    .push(present_results_msg.clone());
                *result_text = format!("{}\n\n{}", result_text, present_results_msg.render());
            }

            if self.depth == 0 && !*state.cli_agent_boundary_injected {
                let task_hint = build_task_boundary_hint(&state.learning_ctx.user_text, 120);
                append_tool_result_notice(
                    result_text,
                    &ToolResultNotice::CliAgentInlineBoundary {
                        task_hint: task_hint.clone(),
                    },
                );
                state
                    .pending_system_messages
                    .push(SystemDirective::CliAgentTaskBoundary { task_hint });
                *state.cli_agent_boundary_injected = true;
            }
        }

        if !state.learning_ctx.errors.is_empty() {
            // Credit any injected diagnostic hints if we recovered after they were shown.
            if let Some(solution_id) = state.pending_error_solution_ids.first().copied() {
                state.pending_error_solution_ids.clear();
                let state_store = self.state.clone();
                tokio::spawn(async move {
                    if let Err(e) = state_store
                        .update_error_solution_outcome(solution_id, true)
                        .await
                    {
                        warn!(solution_id, error = %e, "Failed to record error solution success");
                    }
                });
            }

            let reflection_solution_ids = take_pending_reflection_solution_ids_for_recovery(
                state.pending_reflection_recoveries,
                &tc.name,
                env.iteration,
            );
            if !reflection_solution_ids.is_empty() {
                let state_store = self.state.clone();
                tokio::spawn(async move {
                    for solution_id in reflection_solution_ids {
                        if let Err(e) = state_store
                            .update_error_solution_outcome(solution_id, true)
                            .await
                        {
                            warn!(
                                solution_id,
                                error = %e,
                                "Failed to record reflection learning success"
                            );
                        }
                    }
                });
            }

            // In-session mini learning: if we saw repeated failures for a tool+pattern
            // and then recovered via a different tool, persist the workaround now.
            if let Some((failed_tool, failed_pattern)) = state.last_tool_failure.take() {
                let key = (failed_tool.clone(), failed_pattern.clone());
                let failures = state.tool_failure_patterns.get(&key).copied().unwrap_or(0);
                if failures >= 3 && tc.name != failed_tool && state.in_session_learned.insert(key) {
                    let recovery_tool = tc.name.clone();
                    let solution = crate::memory::procedures::create_error_solution(
                        failed_pattern,
                        Some(failed_tool.clone()),
                        format!(
                            "After {} errors, recovered via {}",
                            failed_tool, recovery_tool
                        ),
                        Some(vec![env.tool_summary.to_string()]),
                    );
                    let state_store = self.state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = state_store.insert_error_solution(&solution).await {
                            warn!(
                                error_pattern = %solution.error_pattern,
                                error = %e,
                                "Failed to save in-session error solution"
                            );
                        }
                    });
                    info!(
                        failed_tool = %failed_tool,
                        recovery_tool = %recovery_tool,
                        failures,
                        "In-session error solution learned"
                    );
                }
            }

            // Successful action after an error - this is recovery
            state
                .learning_ctx
                .recovery_actions
                .push(env.tool_summary.to_string());
            // Mark the last error as recovered
            if let Some((_, recovered)) = state.learning_ctx.errors.last_mut() {
                *recovered = true;
            }
        }

        Ok(ResultLearningOutcome {
            control_flow: None,
            semantic_failure: None,
        })
    }
}

/// Build the first-failure coaching message that quotes the key error line.
/// Returns `None` if key_line is empty (no extractable error).
fn format_error_coaching(key_line: &str) -> Option<ToolResultNotice> {
    if key_line.is_empty() {
        return None;
    }
    let lower = key_line.to_ascii_lowercase();
    // Permission-denied errors get specific guidance instead of generic coaching.
    if lower.contains("permission denied")
        || lower.contains("access denied")
        || lower.contains("operation not permitted")
        || lower.contains("read-only file system")
    {
        return Some(ToolResultNotice::PermissionDeniedCoaching {
            key_line: key_line.to_string(),
        });
    }
    Some(ToolResultNotice::ErrorCoaching {
        key_line: key_line.to_string(),
    })
}

/// Build the semantic-failure coaching message (tool errored N times).
/// Includes the error context when key_line is non-empty.
fn format_semantic_failure_coaching(semantic_count: usize, key_line: &str) -> ToolResultNotice {
    ToolResultNotice::SemanticFailureCoaching {
        semantic_count,
        key_line: key_line.to_string(),
    }
}

pub(super) fn expire_stale_pending_reflection_recoveries(
    pending_reflection_recoveries: &mut HashMap<String, PendingReflectionRecovery>,
    current_iteration: usize,
) {
    pending_reflection_recoveries
        .retain(|_, pending| pending.verify_on_iteration >= current_iteration);
}

fn clear_pending_reflection_recovery_on_failure(
    pending_reflection_recoveries: &mut HashMap<String, PendingReflectionRecovery>,
    tool_name: &str,
    current_signature: &str,
    current_iteration: usize,
) {
    if pending_reflection_recoveries
        .get(tool_name)
        .is_some_and(|pending| {
            pending.signature != current_signature
                || pending.verify_on_iteration <= current_iteration
        })
    {
        pending_reflection_recoveries.remove(tool_name);
    }
}

fn take_pending_reflection_solution_ids_for_recovery(
    pending_reflection_recoveries: &mut HashMap<String, PendingReflectionRecovery>,
    tool_name: &str,
    current_iteration: usize,
) -> Vec<i64> {
    let Some(verify_on_iteration) = pending_reflection_recoveries
        .get(tool_name)
        .map(|pending| pending.verify_on_iteration)
    else {
        return Vec::new();
    };
    if verify_on_iteration != current_iteration {
        if verify_on_iteration < current_iteration {
            pending_reflection_recoveries.remove(tool_name);
        }
        return Vec::new();
    }
    pending_reflection_recoveries
        .remove(tool_name)
        .map(|pending| pending.solution_ids)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_result_substantive_detection_prefers_large_non_error_payloads() {
        let payload = "x".repeat(600);
        assert!(cli_result_is_substantive(&payload));
        assert!(!cli_result_is_substantive("ERROR: agent failed to run"));
        assert!(!cli_result_is_substantive("short output"));
    }

    #[test]
    fn missing_goal_id_error_detection_is_tool_scoped() {
        assert!(looks_like_missing_goal_id_error(
            "scheduled_goal_runs",
            "'goal_id' is required for run_history"
        ));
        assert!(!looks_like_missing_goal_id_error(
            "remember_fact",
            "'goal_id' is required for run_history"
        ));
    }

    #[test]
    fn detects_fact_storage_language() {
        assert!(user_looks_like_fact_storage_request(
            "Please remember these for later"
        ));
        assert!(user_looks_like_fact_storage_request("learn this about me"));
        assert!(user_looks_like_fact_storage_request("note this down"));
        assert!(user_looks_like_fact_storage_request(
            "I need you to know this"
        ));
        assert!(!user_looks_like_fact_storage_request(
            "run scheduled goals now"
        ));
    }

    #[test]
    fn test_repeated_signature_persists_without_reset() {
        let mut counts = HashMap::new();
        let mut signatures = HashMap::new();
        let first = record_semantic_failure_signature(
            &mut counts,
            &mut signatures,
            "read_file",
            "Error: missing required field `path`",
        );
        // Simulate an unrelated success path by intentionally not mutating the
        // signature maps between repeated errors.
        let second = record_semantic_failure_signature(
            &mut counts,
            &mut signatures,
            "read_file",
            "Error: missing required field `path`",
        );
        assert_eq!(first.count, 1);
        assert_eq!(second.count, 2);
        assert_eq!(counts.get("read_file").copied(), Some(2));
    }

    #[test]
    fn test_signature_counter_only_rises_on_repeated_same_error() {
        let mut counts = HashMap::new();
        let mut signatures = HashMap::new();

        let first = record_semantic_failure_signature(
            &mut counts,
            &mut signatures,
            "read_file",
            "Error: missing required field `path`",
        );
        let second_unique = record_semantic_failure_signature(
            &mut counts,
            &mut signatures,
            "read_file",
            "Error: permission denied",
        );
        let third_repeat = record_semantic_failure_signature(
            &mut counts,
            &mut signatures,
            "read_file",
            "Error: missing required field `path`",
        );

        assert_eq!(first.count, 1);
        assert_eq!(second_unique.count, 1);
        assert_eq!(third_repeat.count, 2);
    }

    #[test]
    fn reflection_verification_uses_current_pending_recovery_for_tool() {
        let mut pending = HashMap::from([(
            "http_request".to_string(),
            PendingReflectionRecovery {
                signature: "http 401 unauthorized".to_string(),
                solution_ids: vec![22, 23],
                verify_on_iteration: 4,
            },
        )]);

        let ids =
            take_pending_reflection_solution_ids_for_recovery(&mut pending, "http_request", 4);

        assert_eq!(ids, vec![22, 23]);
        assert!(!pending.contains_key("http_request"));
    }

    #[test]
    fn reflection_verification_skips_later_successes_outside_recovery_turn() {
        let mut pending = HashMap::from([(
            "http_request".to_string(),
            PendingReflectionRecovery {
                signature: "http 401 unauthorized".to_string(),
                solution_ids: vec![22, 23],
                verify_on_iteration: 4,
            },
        )]);

        let ids =
            take_pending_reflection_solution_ids_for_recovery(&mut pending, "http_request", 5);

        assert!(ids.is_empty());
        assert!(!pending.contains_key("http_request"));
    }

    #[test]
    fn changing_failure_signature_clears_pending_reflection_recovery() {
        let mut pending = HashMap::from([(
            "http_request".to_string(),
            PendingReflectionRecovery {
                signature: "http 401 unauthorized".to_string(),
                solution_ids: vec![22, 23],
                verify_on_iteration: 4,
            },
        )]);

        clear_pending_reflection_recovery_on_failure(
            &mut pending,
            "http_request",
            "http 404 not found",
            3,
        );

        assert!(!pending.contains_key("http_request"));
    }

    #[test]
    fn failed_recovery_attempt_clears_pending_reflection_recovery_even_with_same_signature() {
        let mut pending = HashMap::from([(
            "http_request".to_string(),
            PendingReflectionRecovery {
                signature: "http 401 unauthorized".to_string(),
                solution_ids: vec![22, 23],
                verify_on_iteration: 4,
            },
        )]);

        clear_pending_reflection_recovery_on_failure(
            &mut pending,
            "http_request",
            "http 401 unauthorized",
            4,
        );

        assert!(!pending.contains_key("http_request"));
    }

    #[test]
    fn test_signature_distinguishes_different_paths() {
        let left =
            derive_failure_signature("Error: Text not found in /Users/alice/project/src/a.rs:12:8");
        let right =
            derive_failure_signature("Error: Text not found in /Users/alice/project/src/b.rs:59:2");
        assert_ne!(left, right);
        assert!(left.contains("<path:a.rs>"));
        assert!(right.contains("<path:b.rs>"));
    }

    #[test]
    fn test_signature_normalizes_line_numbers_and_attempt_ids() {
        let a = derive_failure_signature(
            "Attempt 1: Error: command failed in /tmp/test/main.rs:10:2 (exit code: 1, pid=923)",
        );
        let b = derive_failure_signature(
            "Attempt 7: Error: command failed in /tmp/test/main.rs:999:88 (exit code: 42, pid=11)",
        );
        assert_eq!(a, b);
        assert!(a.contains("attempt <n>"));
        assert!(a.contains(":<line>"));
        assert!(a.contains("exit <n>"));
        assert!(a.contains("pid=<n>"));
    }

    #[test]
    fn test_file_lookup_miss_skips_transient_cooldown() {
        assert!(should_skip_transient_cooldown(
            "read_file",
            "Error: ENOENT: no such file or directory, open '/tmp/missing.txt'",
        ));
        assert!(!should_skip_transient_cooldown(
            "terminal",
            "Error: ENOENT: no such file or directory, open '/tmp/missing.txt'",
        ));
    }

    #[test]
    fn test_error_coaching_quotes_key_line() {
        let coaching = format_error_coaching("command not found: drush");
        assert!(coaching.is_some());
        let msg = coaching.unwrap().render();
        assert!(msg.contains("[SYSTEM] IMPORTANT"));
        assert!(msg.contains("command not found: drush"));
        assert!(msg.contains("DIFFERENT approach"));
    }

    #[test]
    fn test_error_coaching_returns_none_for_empty_key_line() {
        assert!(format_error_coaching("").is_none());
    }

    #[test]
    fn test_semantic_failure_coaching_includes_error_context() {
        let msg = format_semantic_failure_coaching(3, "Error: ENOENT").render();
        assert!(msg.contains("errored 3 semantic times"));
        assert!(msg.contains("The error was: \"Error: ENOENT\"."));
        assert!(msg.contains("DIFFERENT tool"));
    }

    #[test]
    fn test_semantic_failure_coaching_omits_context_when_empty() {
        let msg = format_semantic_failure_coaching(2, "").render();
        assert!(msg.contains("errored 2 semantic times"));
        assert!(!msg.contains("The error was"));
    }
}
