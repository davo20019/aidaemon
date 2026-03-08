use super::budget_blocking::{DuplicateSendFileNoopCtx, ToolBlockKind, ToolBudgetBlockCtx};
use super::execution_io::ToolExecutionIoCtx;
use super::guards::LoopPatternGuardOutcome;
use super::project_dir::{
    extract_project_dir_hint_with_aliases, extract_project_dirs_from_tool_args,
    is_file_recheck_tool, is_recognized_project_root, maybe_inject_project_dir_into_tool_args,
    project_dir_from_tool_args, scope_allows_project_dir, tool_call_includes_project_path,
};
use super::result_learning::{ResultLearningEnv, ResultLearningState};
use super::types::{ToolExecutionCtx, ToolExecutionOutcome};
use crate::agent::recall_guardrails::is_personal_memory_tool;
use crate::agent::*;
use crate::traits::{ToolCallSemantics, ToolTargetHint, ToolTargetHintKind};
use crate::utils::{truncate_str, truncate_with_note};

const TOOL_COMPLETE_SUMMARY_MAX_CHARS: usize = 140;
const EXTERNAL_ACTION_ACK_MAX_CHARS: usize = 500;

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

struct DeterministicToolContractViolation {
    reason: String,
    coaching: String,
}

fn scheduled_goal_runs_missing_goal_id_violation(
    raw_arguments: &str,
) -> Option<DeterministicToolContractViolation> {
    let parsed = serde_json::from_str::<Value>(raw_arguments).ok()?;
    let map = parsed.as_object()?;
    let has_goal_id = map
        .get("goal_id")
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.trim().is_empty());
    if has_goal_id {
        None
    } else {
        let action = map
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("<missing>");
        Some(DeterministicToolContractViolation {
            reason: format!(
                "action `{}` requires `goal_id` for `scheduled_goal_runs`",
                action
            ),
            coaching: "If the user asked to learn/remember/save facts, use `remember_fact`. \
If they asked about scheduled-goal runs, first get IDs with \
`manage_memories(action='list_scheduled')`, then call `scheduled_goal_runs` with `goal_id`."
                .to_string(),
        })
    }
}

fn deterministic_tool_contract_violation(
    tool_name: &str,
    raw_arguments: &str,
) -> Option<DeterministicToolContractViolation> {
    match tool_name {
        "scheduled_goal_runs" => scheduled_goal_runs_missing_goal_id_violation(raw_arguments),
        _ => None,
    }
}

fn tool_is_currently_exposed(tool_defs: &[Value], tool_name: &str) -> bool {
    tool_defs.iter().any(|def| {
        def.get("function")
            .and_then(|function| function.get("name"))
            .and_then(|name| name.as_str())
            .is_some_and(|exposed_name| exposed_name == tool_name)
    })
}

fn blocked_for_untrusted_external_reference_message(
    tool_name: &str,
    active_skills: &[String],
) -> String {
    let scope = if active_skills.is_empty() {
        "an untrusted external API guide reference".to_string()
    } else {
        format!(
            "untrusted external API guide skill(s): {}",
            active_skills.join(", ")
        )
    };
    format!(
        "Blocked: `{}` is unavailable while using {}. \
Use API/auth tools directly, or ask explicitly for local file or repository inspection if you want me to read local files or inspect the local environment.",
        tool_name, scope
    )
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
            !scope_allows_project_dir(allowed_scope, dir)
                && !allow_scaffold_parent_dir(dir)
                && !is_recognized_project_root(dir)
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

fn tool_result_indicates_background_detach(
    tool_name: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> bool {
    let _ = tool_name;
    if metadata.background_started {
        return true;
    }
    result_text.contains("Moved to background")
        || result_text.contains("started in background")
        || result_text.contains("spawned in background")
}

fn build_background_detach_ack(
    tool_name: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> String {
    let default_prefix = match tool_name {
        "terminal" => "The command is running in the background.",
        "cli_agent" => "The CLI agent task is running in the background.",
        "spawn_agent" => "The spawned sub-agent is running in the background.",
        _ => "The task is running in the background.",
    };
    let first_line = crate::traits::first_primary_message_line(result_text, &[])
        .unwrap_or(default_prefix.to_string());
    // Use structured tool metadata rather than inferring notification semantics
    // from rendered tool output text.
    let notifications_active = metadata.completion_notifications_enabled;
    if notifications_active {
        format!(
            "{} Completion notifications are enabled, and the final result will be sent automatically when it finishes.",
            first_line
        )
    } else {
        first_line.to_string()
    }
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

fn is_trivial_success_excerpt(s: &str) -> bool {
    let lower = s.trim().to_ascii_lowercase();
    lower.is_empty()
        || lower == "ok"
        || lower == "done"
        || lower == "success"
        || lower == "completed"
        || lower == "completed successfully"
        || lower == "request completed successfully"
}

fn summarize_completed_tool_result(result_text: &str) -> String {
    let summary = crate::traits::first_primary_message_line(result_text, &[])
        .filter(|line| !line.trim().is_empty())
        .unwrap_or_else(|| "Completed".to_string());
    truncate_str(summary.trim(), TOOL_COMPLETE_SUMMARY_MAX_CHARS)
}

fn build_external_action_completion_ack(result_text: &str) -> String {
    let primary = crate::traits::extract_primary_message_content(result_text, &[]);
    let excerpt = primary.trim();
    if excerpt.is_empty() || is_trivial_success_excerpt(excerpt) {
        "The requested action completed successfully.".to_string()
    } else {
        format!(
            "The requested action completed successfully.\n\nLatest result:\n{}",
            truncate_with_note(excerpt, EXTERNAL_ACTION_ACK_MAX_CHARS)
        )
    }
}

fn should_build_external_action_ack(result_text: &str) -> bool {
    let primary = crate::traits::extract_primary_message_content(result_text, &[]);
    let lower = primary.trim_start().to_ascii_lowercase();
    !lower.starts_with("request blocked:")
        && !lower.starts_with("blocked:")
        && !lower.starts_with("[system] blocked:")
        && !lower.starts_with("error:")
        && !lower.starts_with("failed to ")
}

fn tool_result_contains_verifiable_evidence(
    semantics: &ToolCallSemantics,
    result_text: &str,
) -> bool {
    if !semantics.can_verify_with_result_content() {
        return false;
    }

    let primary = crate::traits::extract_primary_message_content(result_text, &[]);
    let primary = primary.trim();
    !primary.is_empty()
        && !matches!(
            primary.to_ascii_lowercase().as_str(),
            "ok" | "done" | "success" | "completed" | "completed successfully"
        )
}

fn normalized_target_value(value: &str) -> String {
    value
        .trim()
        .trim_end_matches('/')
        .trim_end_matches(['.', ',', ';'])
        .to_ascii_lowercase()
}

fn tool_target_hint_matches_contract_target(
    target_hint: &ToolTargetHint,
    contract_target: &VerificationTarget,
) -> bool {
    let compatible_kind = matches!(
        (target_hint.kind, contract_target.kind),
        (ToolTargetHintKind::Url, VerificationTargetKind::Url)
            | (ToolTargetHintKind::Path, VerificationTargetKind::Path)
            | (
                ToolTargetHintKind::ProjectScope,
                VerificationTargetKind::ProjectScope
            )
            | (
                ToolTargetHintKind::Path,
                VerificationTargetKind::ProjectScope
            )
            | (
                ToolTargetHintKind::ProjectScope,
                VerificationTargetKind::Path
            )
    );
    if !compatible_kind {
        return false;
    }

    let hint = normalized_target_value(&target_hint.value);
    let contract = normalized_target_value(&contract_target.value);
    if hint.is_empty() || contract.is_empty() {
        return false;
    }

    hint == contract || hint.contains(&contract) || contract.contains(&hint)
}

fn verification_target_matches_haystack(target: &VerificationTarget, haystack: &str) -> bool {
    let haystack = haystack.to_ascii_lowercase();
    let needle = normalized_target_value(&target.value);
    if needle.is_empty() {
        return false;
    }

    if haystack.contains(&needle) {
        return true;
    }

    match target.kind {
        VerificationTargetKind::ProjectScope | VerificationTargetKind::Path => target
            .value
            .rsplit(['/', '\\'])
            .find(|segment| !segment.is_empty())
            .map(normalized_target_value)
            .is_some_and(|tail| !tail.is_empty() && haystack.contains(&tail)),
        VerificationTargetKind::Url => false,
    }
}

fn observation_matches_completion_contract(
    contract: &CompletionContract,
    semantics: &ToolCallSemantics,
    raw_arguments: &str,
    result_text: &str,
) -> bool {
    if contract.verification_targets.is_empty() {
        return true;
    }

    if semantics.target_hints.iter().any(|hint| {
        contract
            .verification_targets
            .iter()
            .any(|target| tool_target_hint_matches_contract_target(hint, target))
    }) {
        return true;
    }

    let mut haystacks = vec![
        raw_arguments.to_string(),
        crate::traits::extract_primary_message_content(result_text, &[]).to_string(),
        result_text.to_string(),
    ];
    if let Some(command) = extract_command_from_args(raw_arguments) {
        haystacks.push(command);
    }

    contract.verification_targets.iter().any(|target| {
        haystacks
            .iter()
            .any(|haystack| verification_target_matches_haystack(target, haystack))
    })
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
        let active_untrusted_external_reference_skills =
            ctx.active_untrusted_external_reference_skills;
        let restrict_untrusted_external_reference_tools =
            ctx.restrict_untrusted_external_reference_tools;
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
        let mut tool_failure_signatures = std::mem::take(ctx.tool_failure_signatures);
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
        let mut pending_external_action_ack: Option<String> = None;
        let mut stall_count = *ctx.stall_count;
        let mut deferred_no_tool_streak = *ctx.deferred_no_tool_streak;
        let mut consecutive_clean_iterations = *ctx.consecutive_clean_iterations;
        let mut fallback_expanded_once = *ctx.fallback_expanded_once;
        let mut known_project_dir = std::mem::take(ctx.known_project_dir);
        let mut dirs_with_project_inspect_file_evidence =
            std::mem::take(ctx.dirs_with_project_inspect_file_evidence);
        let mut dirs_with_search_no_matches = std::mem::take(ctx.dirs_with_search_no_matches);
        let mut require_file_recheck_before_answer = *ctx.require_file_recheck_before_answer;
        let mut completion_progress = ctx.completion_progress.clone();
        let mut tool_result_cache = std::mem::take(ctx.tool_result_cache);

        macro_rules! commit_state {
            () => {
                *ctx.tool_defs = tool_defs;
                *ctx.total_tool_calls_attempted = total_tool_calls_attempted;
                *ctx.total_successful_tool_calls = total_successful_tool_calls;
                *ctx.tool_failure_count = tool_failure_count;
                *ctx.tool_failure_signatures = tool_failure_signatures;
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
                *ctx.pending_external_action_ack = pending_external_action_ack;
                *ctx.stall_count = stall_count;
                *ctx.deferred_no_tool_streak = deferred_no_tool_streak;
                *ctx.consecutive_clean_iterations = consecutive_clean_iterations;
                *ctx.fallback_expanded_once = fallback_expanded_once;
                *ctx.known_project_dir = known_project_dir;
                *ctx.dirs_with_project_inspect_file_evidence =
                    dirs_with_project_inspect_file_evidence;
                *ctx.dirs_with_search_no_matches = dirs_with_search_no_matches;
                *ctx.require_file_recheck_before_answer = require_file_recheck_before_answer;
                *ctx.completion_progress = completion_progress.clone();
                *ctx.tool_result_cache = tool_result_cache;
            };
        }

        if known_project_dir.is_none() {
            known_project_dir =
                extract_project_dir_hint_with_aliases(_user_text, &self.path_aliases.projects);
        }

        let mut successful_tool_calls = 0;
        let mut iteration_had_tool_failures = false;
        info!(
            session_id,
            iteration,
            tool_count = resp.tool_calls.len(),
            total_successful_tool_calls,
            "Tool execution phase starting"
        );
        for tc in &resp.tool_calls {
            let policy_tool_budget = policy_bundle.policy.tool_budget;
            if self.policy_config.policy_enforce
                && is_hard_policy_tool_budget_reached(
                    total_tool_calls_attempted,
                    policy_tool_budget,
                )
            {
                force_text_response = true;
                pending_system_messages
                    .push(SystemDirective::HardPolicyToolBudgetReached { policy_tool_budget });
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
                let result_text = ToolResultNotice::HardPolicyToolBudgetBlocked {
                    policy_tool_budget,
                    tool_name: tc.name.clone(),
                }
                .render();
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
                    ..Message::runtime_defaults()
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
                    let result_text = ToolResultNotice::PersonalMemoryToolsOnly {
                        tool_name: tc.name.clone(),
                    }
                    .render();
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
                        ..Message::runtime_defaults()
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
                    pending_system_messages
                        .push(SystemDirective::PersonalMemoryRecheckLimitReached);
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
                        ..Message::runtime_defaults()
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

            if restrict_untrusted_external_reference_tools
                && crate::agent::is_untrusted_external_reference_blocked_tool(&tc.name)
            {
                let result_text = blocked_for_untrusted_external_reference_message(
                    &tc.name,
                    active_untrusted_external_reference_skills,
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
                    importance: 0.15,
                    ..Message::runtime_defaults()
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
                iteration_had_tool_failures = true;
                continue;
            }

            let tool_is_known_but_hidden = !tool_is_currently_exposed(&tool_defs, &tc.name)
                && (tool_is_currently_exposed(base_tool_defs, &tc.name)
                    || self.has_registered_tool(&tc.name));
            if tool_is_known_but_hidden {
                *tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;
                self.emit_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::ToolBudgetBlock,
                    format!(
                        "Blocked tool {} because it is not currently exposed",
                        tc.name
                    ),
                    json!({
                        "tool": tc.name,
                        "reason": "tool_not_currently_exposed",
                    }),
                )
                .await;
                let result_text = ToolResultNotice::ToolNotCurrentlyExposed {
                    tool_name: tc.name.clone(),
                }
                .render();
                let tool_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "tool".to_string(),
                    content: Some(result_text),
                    tool_call_id: Some(tc.id.clone()),
                    tool_name: Some(tc.name.clone()),
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.15,
                    ..Message::runtime_defaults()
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
                iteration_had_tool_failures = true;
                continue;
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
                let result_text = ToolResultNotice::ScopeLockBlockedResult {
                    tool_name: tc.name.clone(),
                    reason: scope_reason.clone(),
                }
                .render();
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
                    ..Message::runtime_defaults()
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
                pending_system_messages.push(SystemDirective::ScopeLockBlocked {
                    tool_name: tc.name.clone(),
                    reason: scope_reason,
                });
                iteration_had_tool_failures = true;
                continue;
            }

            if let Some(contract_violation) =
                deterministic_tool_contract_violation(&tc.name, &effective_arguments)
            {
                *tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;
                let result_text = ToolResultNotice::DeterministicArgumentContractBlocked {
                    tool_name: tc.name.clone(),
                    reason: contract_violation.reason.clone(),
                }
                .render();
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
                    ..Message::runtime_defaults()
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
                pending_system_messages.push(SystemDirective::ArgumentContractBlocked {
                    tool_name: tc.name.clone(),
                    reason: contract_violation.reason.to_string(),
                    coaching: contract_violation.coaching.to_string(),
                });
                iteration_had_tool_failures = true;
                continue;
            }
            match self
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
                ToolBlockKind::NotBlocked => {}
                ToolBlockKind::Cooldown => {
                    // Cooldown blocks are temporary — the tool will be available
                    // again in a few iterations. Do NOT set force_text_response;
                    // let the agent try other tools or wait for cooldown to expire.
                    continue;
                }
                ToolBlockKind::HardBlock => {
                    // Permanent block (semantic failure limit, unknown tool, call
                    // count limit). Activate force-text so the next LLM call
                    // strips all tools and forces a text response. Without this,
                    // weak models keep retrying the same blocked tool indefinitely.
                    force_text_response = true;
                    pending_system_messages.push(SystemDirective::HardToolLimitReached);
                    continue;
                }
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
                    &tool_result_cache,
                )
                .await?
            {
                match guard_outcome {
                    LoopPatternGuardOutcome::ContinueLoop => {
                        continue;
                    }
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
                        force_text_response: &mut force_text_response,
                        pending_system_messages: &mut pending_system_messages,
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
                        project_scope: allowed_project_scope,
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
            let mut result_metadata = io.result_metadata;
            if tc.name == "run_command" && run_command_policy_block_requires_terminal(&result_text)
            {
                if let Some(terminal_args) =
                    build_terminal_fallback_arguments_from_run_command(&effective_arguments)
                {
                    let fallback_started = Instant::now();
                    let terminal_result = self
                        .execute_tool_with_watchdog_outcome(
                            "terminal",
                            &terminal_args,
                            &tool_exec::ToolExecCtx {
                                session_id,
                                task_id: Some(task_id),
                                status_tx: status_tx.clone(),
                                channel_visibility: channel_ctx.visibility,
                                channel_id: channel_ctx.channel_id.as_deref(),
                                project_scope: allowed_project_scope,
                                trusted: channel_ctx.trusted,
                                user_role,
                            },
                        )
                        .await;
                    let fallback_duration =
                        fallback_started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                    tool_duration_ms = tool_duration_ms.saturating_add(fallback_duration);
                    let fallback_note = ToolResultNotice::RunCommandPolicyAutoRoutedToTerminal;
                    result_text = match terminal_result {
                        Ok(outcome) => {
                            result_metadata = outcome.metadata;
                            format!("{}\n\n{}", outcome.output, fallback_note.render())
                        }
                        Err(e) => {
                            result_metadata.transport_error = Some(e.to_string());
                            format!("Error: {}\n\n{}", e, fallback_note.render())
                        }
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
                tool_result_indicates_background_detach(&tc.name, &result_text, &result_metadata);

            if background_detached {
                pending_background_ack = Some(build_background_detach_ack(
                    &tc.name,
                    &result_text,
                    &result_metadata,
                ));
                force_text_response = true;
                let notifications_active = result_metadata.completion_notifications_enabled;
                let system_msg = SystemDirective::BackgroundHandoff {
                    notifications_active,
                };
                pending_system_messages.push(system_msg.clone());
                result_text = format!("{}\n\n{}", result_text, system_msg.render());
            }

            // Track total calls per tool
            *tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;

            // Cache successful read_file/search_files results so the repetitive
            // redirect can replay them instead of sending a generic "BLOCKED"
            // message.  This solves the lost-context problem: when context
            // truncation drops earlier read results, the model re-reads the same
            // file, gets redirected, and receives the cached content + coaching
            // to write fixes instead of reading again.
            if matches!(tc.name.as_str(), "read_file" | "search_files") {
                let cache_hash = hash_tool_call(&tc.name, &tc.arguments);
                // Cap cached content at 8KB to avoid bloating the redirect msg
                let max_cache_chars = 8000;
                let primary_result_text =
                    crate::traits::extract_primary_message_content(&result_text, &[]);
                if primary_result_text.len() <= max_cache_chars
                    && !result_text.starts_with("Error")
                    && !crate::traits::message_content_is_structural_only(&result_text, &[])
                {
                    tool_result_cache.insert(cache_hash, primary_result_text.into_owned());
                } else if primary_result_text.len() > max_cache_chars {
                    // Store a truncated version rather than nothing
                    let mut boundary = max_cache_chars;
                    while boundary > 0 && !primary_result_text.is_char_boundary(boundary) {
                        boundary -= 1;
                    }
                    tool_result_cache.insert(
                        cache_hash,
                        format!(
                            "{}…\n[truncated — {} total chars]",
                            &primary_result_text[..boundary],
                            primary_result_text.len()
                        ),
                    );
                }
                // Bound the cache size to prevent unbounded growth
                const MAX_CACHE_ENTRIES: usize = 20;
                if tool_result_cache.len() > MAX_CACHE_ENTRIES {
                    // Remove the oldest entry (arbitrary, but bounded)
                    if let Some(key) = tool_result_cache.keys().next().copied() {
                        tool_result_cache.remove(&key);
                    }
                }
            }

            // Track tool call for learning
            let tool_summary = format!(
                "{}({})",
                tc.name,
                summarize_tool_args(&tc.name, &effective_arguments)
            );
            learning_ctx.tool_calls.push(tool_summary.clone());

            // Track tool failures across iterations using structured detection
            // (prefixes, JSON error payloads, HTTP statuses, non-zero exit codes).
            let failure_class = classify_tool_result_failure_with_context(
                &tc.name,
                &result_text,
                Some(&effective_arguments),
                Some(&result_metadata),
            );
            let is_error = failure_class.is_some();
            info!(
                session_id,
                iteration,
                tool = %tc.name,
                is_error,
                result_len = result_text.len(),
                result_preview = &result_text.chars().take(80).collect::<String>() as &str,
                "Tool execution completed"
            );

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
                tool_failure_signatures: &mut tool_failure_signatures,
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

            if !is_error {
                send_status(
                    &status_tx,
                    StatusUpdate::ToolComplete {
                        name: tc.name.clone(),
                        summary: summarize_completed_tool_result(&result_text),
                    },
                );
                let caps = available_capabilities
                    .get(&tc.name)
                    .copied()
                    .unwrap_or_default();
                let semantics = &result_metadata.semantics;
                if semantics.mutates_state() {
                    completion_progress.mark_mutation(&turn_context.completion_contract);
                }
                if semantics.observes_state() {
                    let can_verify =
                        tool_result_contains_verifiable_evidence(semantics, &result_text);
                    let matched_contract = observation_matches_completion_contract(
                        &turn_context.completion_contract,
                        semantics,
                        &effective_arguments,
                        &result_text,
                    );
                    completion_progress.mark_observation(
                        &turn_context.completion_contract,
                        can_verify && matched_contract,
                    );
                }
                if completion_progress.verification_pending {
                    pending_external_action_ack = None;
                } else if !background_detached
                    && semantics.mutates_state()
                    && caps.external_side_effect
                    && should_build_external_action_ack(&result_text)
                {
                    pending_external_action_ack =
                        Some(build_external_action_completion_ack(&result_text));
                }
            } else {
                pending_external_action_ack = None;
            }

            let tool_msg = Message {
                content: Some(result_text.clone()),
                tool_call_id: Some(tc.id.clone()),
                tool_name: Some(tc.name.clone()),
                importance: 0.3, // Tool outputs default to lower importance
                ..Message::new_runtime(Uuid::new_v4().to_string(), session_id, "tool")
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

        info!(
            session_id,
            iteration,
            successful_tool_calls,
            iteration_had_tool_failures,
            total_successful_tool_calls,
            stall_count,
            "Tool execution phase completed, entering post-loop"
        );

        self.apply_post_tool_iteration_controls(
            super::post_loop::PostToolIterationInputs {
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
                has_active_goal: resolved_goal_id.is_some(),
                completed_tool_calls: &learning_ctx.tool_calls,
                recent_tool_names: &recent_tool_names,
                user_text: _user_text,
            },
            super::post_loop::PostToolIterationState {
                total_successful_tool_calls: &mut total_successful_tool_calls,
                force_text_response: &mut force_text_response,
                pending_system_messages: &mut pending_system_messages,
                tool_defs: &mut tool_defs,
                stall_count: &mut stall_count,
                deferred_no_tool_streak: &mut deferred_no_tool_streak,
                consecutive_clean_iterations: &mut consecutive_clean_iterations,
                fallback_expanded_once: &mut fallback_expanded_once,
            },
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
    fn deterministic_tool_contract_violation_blocks_scheduled_goal_runs_without_goal_id() {
        let args = r#"{"action":"run_history"}"#;
        let violation = deterministic_tool_contract_violation("scheduled_goal_runs", args);
        assert!(violation.is_some());
        let violation = violation.expect("violation expected");
        assert!(violation.reason.contains("requires `goal_id`"));
        assert!(violation.coaching.contains("remember_fact"));
    }

    #[test]
    fn deterministic_tool_contract_violation_allows_scheduled_goal_runs_with_goal_id() {
        let args = r#"{"action":"run_history","goal_id":"goal-123"}"#;
        let violation = deterministic_tool_contract_violation("scheduled_goal_runs", args);
        assert!(violation.is_none());
    }

    #[test]
    fn tool_is_currently_exposed_matches_current_tool_defs() {
        let tool_defs = vec![
            json!({
                "type": "function",
                "function": {
                    "name": "system_info",
                    "description": "demo",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "remember_fact",
                    "description": "demo",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }),
        ];

        assert!(tool_is_currently_exposed(&tool_defs, "system_info"));
        assert!(!tool_is_currently_exposed(&tool_defs, "cli_agent"));
    }

    #[test]
    fn result_content_verification_requires_semantics_opt_in() {
        let verifyable = ToolCallSemantics::observation()
            .with_verification_mode(crate::traits::ToolVerificationMode::ResultContent);
        let non_verifyable = ToolCallSemantics::observation();
        assert!(tool_result_contains_verifiable_evidence(
            &verifyable,
            "Latest post title: Scheduled reflection"
        ));
        assert!(!tool_result_contains_verifiable_evidence(
            &non_verifyable,
            "Latest post title: Scheduled reflection"
        ));
    }

    #[test]
    fn semantics_target_hints_match_contract_targets() {
        let contract = CompletionContract {
            requires_observation: true,
            verification_targets: vec![VerificationTarget {
                kind: VerificationTargetKind::Url,
                value: "https://blog.aidaemon.ai".to_string(),
            }],
            ..CompletionContract::default()
        };
        let semantics = ToolCallSemantics::observation()
            .with_verification_mode(crate::traits::ToolVerificationMode::ResultContent)
            .with_target_hint(ToolTargetHintKind::Url, "https://blog.aidaemon.ai");
        assert!(observation_matches_completion_contract(
            &contract,
            &semantics,
            "{}",
            "Latest post title: Scheduled reflection"
        ));
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
        let none = crate::traits::ToolCallMetadata::default();
        let flagged = crate::traits::ToolCallMetadata {
            background_started: true,
            ..Default::default()
        };
        assert!(tool_result_indicates_background_detach(
            "terminal",
            "Process finished normally.",
            &flagged
        ));
        assert!(tool_result_indicates_background_detach(
            "terminal",
            "Command still running after 30s. Moved to background (pid=123).",
            &none
        ));
        assert!(tool_result_indicates_background_detach(
            "cli_agent",
            "CLI agent 'x' started in background (task_id=abc).",
            &none
        ));
        assert!(tool_result_indicates_background_detach(
            "spawn_agent",
            "Sub-agent spawned in background for mission: \"...\"",
            &none
        ));
        assert!(tool_result_indicates_background_detach(
            "web_search",
            "Moved to background (pid=1)",
            &none
        ));
        assert!(!tool_result_indicates_background_detach(
            "terminal",
            "Process finished normally.",
            &none
        ));
    }

    #[test]
    fn builds_deterministic_background_ack_from_tool_result() {
        let with_notify = crate::traits::ToolCallMetadata {
            completion_notifications_enabled: true,
            ..Default::default()
        };
        let without_notify = crate::traits::ToolCallMetadata::default();

        // With notifications enabled — should promise automatic delivery.
        let ack = build_background_detach_ack(
            "terminal",
            "Command still running after 30s. Moved to background (pid=123).\n\nCompletion notifications are enabled. The user will be notified when this process finishes.\n\n[SYSTEM] ...",
            &with_notify,
        );
        assert!(ack.contains("Moved to background (pid=123)"));
        assert!(ack.contains("final result will be sent automatically"));

        // Without notifications — should NOT promise automatic delivery.
        let ack_no_notify = build_background_detach_ack(
            "terminal",
            "Command still running after 30s. Moved to background (pid=456).\n\nThis process is task-owned and will be auto-killed when the current task ends.",
            &without_notify,
        );
        assert!(ack_no_notify.contains("Moved to background (pid=456)"));
        assert!(!ack_no_notify.contains("final result will be sent automatically"));
    }

    #[test]
    fn background_ack_uses_structured_notification_metadata_not_text() {
        let with_notify = crate::traits::ToolCallMetadata {
            completion_notifications_enabled: true,
            ..Default::default()
        };
        let without_notify = crate::traits::ToolCallMetadata::default();

        let ack = build_background_detach_ack(
            "terminal",
            "Command still running after 30s. Moved to background (pid=123).",
            &with_notify,
        );
        assert!(ack.contains("final result will be sent automatically"));

        let ack_no_notify = build_background_detach_ack(
            "terminal",
            "Command still running after 30s. Moved to background (pid=456).\n\nCompletion notifications are enabled. The user will be notified when this process finishes.",
            &without_notify,
        );
        assert!(!ack_no_notify.contains("final result will be sent automatically"));
    }

    #[test]
    fn blocked_for_untrusted_external_reference_message_mentions_skill_names() {
        let message = blocked_for_untrusted_external_reference_message(
            "read_file",
            &["widgets-api".to_string(), "linear-api".to_string()],
        );
        assert!(message.contains("read_file"));
        assert!(message.contains("widgets-api"));
        assert!(message.contains("linear-api"));
        assert!(message.contains("explicitly for local file or repository inspection"));
    }
}
