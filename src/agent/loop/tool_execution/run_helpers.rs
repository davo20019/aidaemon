//! Pure helper functions and predicates supporting `run_tool_execution_phase`.
//!
//! This module was split out of `run.rs` to keep that file under the size
//! budget. All items are moved verbatim — no logic changes. Visibility is
//! `pub(super)` so `run.rs` can call them via `super::run_helpers::*`.

use super::project_dir::scope_allows_project_dir;
use crate::agent::execution_state::{extract_target_hints_from_arguments, StepExecutionPlan};
use crate::agent::prefix_fingerprint;
use crate::agent::*;
use crate::events::{Event, EventType, ToolCallData, ToolResultData};
use crate::traits::{
    RequestEvidenceRequirement, RequestVerificationTargetKind, ToolCallSemantics,
    ToolSemanticScope, ToolTargetHint, ToolTargetHintKind,
};
use crate::utils::{truncate_str, truncate_with_note};

const TOOL_COMPLETE_SUMMARY_MAX_CHARS: usize = 140;
const EXTERNAL_ACTION_ACK_MAX_CHARS: usize = 500;

/// Extract a short error summary line from tool result text.
pub(super) fn extract_error_summary_line(result_text: &str) -> Option<String> {
    let mut lines = Vec::new();
    let mut in_recovery_section = false;
    for raw in result_text.lines() {
        if crate::utils::is_internal_scaffolding_line(raw) {
            continue;
        }
        let line = raw.trim();
        let label = line.trim_start_matches('#').trim().to_ascii_lowercase();
        if line.starts_with('#') {
            in_recovery_section =
                matches!(label.as_str(), "recovery options" | "safe recovery options");
            continue;
        }
        if in_recovery_section
            || line.is_empty()
            || matches!(
                label.as_str(),
                "error output" | "failure details" | "recovery options" | "safe recovery options"
            )
        {
            continue;
        }
        lines.push(line);
    }

    // Prefer the concrete OS/API diagnostic over a wrapper such as
    // "ERROR: CLI agent failed". In the live exit-127 incident, selecting the
    // wrapper's own `## Error Output` heading erased the only actionable line.
    const HIGH_SIGNAL: &[&str] = &[
        "no such file or directory",
        "command not found",
        "permission denied",
        "access denied",
        "unauthorized",
        "forbidden",
        "authentication",
        "timed out",
        "connection refused",
        "fatal:",
        "panicked at",
    ];
    if let Some(line) = lines.iter().find(|line| {
        let lower = line.to_ascii_lowercase();
        HIGH_SIGNAL.iter().any(|signal| lower.contains(signal))
    }) {
        return Some(line.chars().take(200).collect());
    }

    lines
        .into_iter()
        .find(|line| {
            line.contains("API ERROR")
                || line.contains("Error")
                || line.contains("error")
                || line.contains("Failed")
                || line.contains("failed")
                || line.contains("BLOCKED")
        })
        .map(|line| line.chars().take(200).collect())
}

pub(super) fn raw_internal_scope_violation(
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

pub(in crate::agent) fn fallback_tool_semantic_scope(tool_name: &str) -> Option<ToolSemanticScope> {
    match tool_name {
        "web_search" | "web_fetch" | "http_request" | "browser" => {
            Some(ToolSemanticScope::ExternalRemote)
        }
        "read_file" | "search_files" | "edit_file" | "write_file" | "terminal"
        | "project_inspect" => Some(ToolSemanticScope::LocalWorkspace),
        "read_channel_history" | "search_history" => Some(ToolSemanticScope::ConversationHistory),
        "remember_fact" | "manage_memories" | "share_memory" => Some(ToolSemanticScope::UserMemory),
        "scheduled_goals"
        | "scheduled_goal_runs"
        | "manage_goal_tasks"
        | "goal_trace"
        | "tool_trace"
        | "manage_mandates" => Some(ToolSemanticScope::GoalState),
        "system_info" | "check_environment" => Some(ToolSemanticScope::HostLocal),
        _ => None,
    }
}

pub(super) fn duplicate_successful_tool_result_count(
    events: &[Event],
    tool_name: &str,
    arguments_json: &str,
    result_text: &str,
) -> usize {
    let argument_hash = canonical_tool_arguments_hash(arguments_json);
    let current_result = normalized_tool_result_for_duplicate_detection(result_text);
    let mut matching_call_ids = HashSet::new();
    let mut duplicate_count = 0usize;

    for event in events {
        match event.event_type {
            EventType::ToolCall => {
                let Ok(call) = event.parse_data::<ToolCallData>() else {
                    continue;
                };
                if call.name == tool_name
                    && prefix_fingerprint::hash_canonical(&call.arguments) == argument_hash
                {
                    matching_call_ids.insert(call.tool_call_id);
                }
            }
            EventType::ToolResult => {
                let Ok(result) = event.parse_data::<ToolResultData>() else {
                    continue;
                };
                if result.success
                    && result.name == tool_name
                    && matching_call_ids.contains(&result.tool_call_id)
                    && normalized_tool_result_for_duplicate_detection(&result.result)
                        == current_result
                {
                    duplicate_count += 1;
                }
            }
            _ => {}
        }
    }

    duplicate_count
}

pub(super) fn canonical_tool_arguments_hash(arguments_json: &str) -> String {
    let value = serde_json::from_str::<Value>(arguments_json)
        .unwrap_or_else(|_| Value::String(arguments_json.to_string()));
    prefix_fingerprint::hash_canonical(&value)
}

fn normalized_tool_result_for_duplicate_detection(result_text: &str) -> String {
    crate::traits::extract_primary_message_content(result_text, &[])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn semantic_scope_blocks_tool(
    active_scope: Option<ToolSemanticScope>,
    tool_scope: Option<ToolSemanticScope>,
) -> bool {
    matches!(
        (active_scope, tool_scope),
        (
            Some(ToolSemanticScope::GoalState),
            Some(ToolSemanticScope::ExternalRemote)
        )
    )
}

pub(super) fn is_scheduled_goal_execution_text(text: &str) -> bool {
    text.to_ascii_lowercase()
        .contains("[system: already scheduled and firing now; do not reschedule.]")
}

pub(super) fn effective_dialogue_scope_for_tool_execution(
    active_scope: Option<ToolSemanticScope>,
    user_text: &str,
    is_scheduled_goal: bool,
) -> Option<ToolSemanticScope> {
    // Scheduled automation is already authorized and scoped by durable goal,
    // run, task, role, and per-tool policy state. Do not let a keyword-derived
    // dialogue label override that provenance. This matters for executor child
    // tasks, whose model-written step descriptions do not retain the root
    // task's scheduled-execution marker.
    if is_scheduled_goal || is_scheduled_goal_execution_text(user_text) {
        None
    } else {
        active_scope
    }
}

pub(super) struct DeterministicToolContractViolation {
    pub(super) reason: String,
    pub(super) coaching: String,
}

pub(super) fn scheduled_goal_runs_missing_goal_id_violation(
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

pub(super) fn manage_memories_missing_goal_id_violation(
    raw_arguments: &str,
) -> Option<DeterministicToolContractViolation> {
    const ACTIONS_REQUIRING_GOAL_ID: &[&str] = &[
        "add_schedule",
        "cancel_scheduled",
        "pause_scheduled",
        "resume_scheduled",
        "retry_scheduled",
        "diagnose_scheduled",
    ];

    let parsed = serde_json::from_str::<Value>(raw_arguments).ok()?;
    let map = parsed.as_object()?;
    let action = map.get("action").and_then(|v| v.as_str())?;
    if !ACTIONS_REQUIRING_GOAL_ID.contains(&action) {
        return None;
    }

    let has_goal_id = map
        .get("goal_id")
        .and_then(|v| v.as_str())
        .is_some_and(|v| !v.trim().is_empty());
    if has_goal_id {
        return None;
    }

    Some(DeterministicToolContractViolation {
        reason: format!(
            "action `{}` requires `goal_id` for `manage_memories`",
            action
        ),
        coaching: "If the user asked about scheduled goals, first get IDs with \
`manage_memories(action='list_scheduled')`, then retry the intended \
`manage_memories` action with a concrete `goal_id`."
            .to_string(),
    })
}

pub(super) fn deterministic_tool_contract_violation(
    tool_name: &str,
    raw_arguments: &str,
) -> Option<DeterministicToolContractViolation> {
    match tool_name {
        "scheduled_goal_runs" => scheduled_goal_runs_missing_goal_id_violation(raw_arguments),
        "manage_memories" => manage_memories_missing_goal_id_violation(raw_arguments),
        "terminal" => {
            let parsed = serde_json::from_str::<Value>(raw_arguments).ok()?;
            let command = parsed.get("command").and_then(Value::as_str)?;
            let has_write_paths = parsed
                .get("write_paths")
                .and_then(Value::as_array)
                .is_some_and(|paths| {
                    paths
                        .iter()
                        .any(|path| path.as_str().is_some_and(|path| !path.trim().is_empty()))
                });
            (crate::tools::command_semantics::classify_shell_command(command).mutates_state()
                && !has_write_paths)
                .then(|| DeterministicToolContractViolation {
                    reason: "mutating terminal runs require an explicit non-empty write_paths access manifest".to_string(),
                    coaching: "Retry the same operation with execution working_dir kept separate from the exact write_paths it may change.".to_string(),
                })
        }
        "cli_agent" => {
            let parsed = serde_json::from_str::<Value>(raw_arguments).ok()?;
            let is_read_write = parsed.get("action").and_then(Value::as_str) == Some("run")
                && parsed.get("workspace_mode").and_then(Value::as_str) == Some("read_write");
            let has_write_paths = parsed
                .get("write_paths")
                .and_then(Value::as_array)
                .is_some_and(|paths| {
                    paths
                        .iter()
                        .any(|path| path.as_str().is_some_and(|path| !path.trim().is_empty()))
                });
            (is_read_write && !has_write_paths).then(|| DeterministicToolContractViolation {
                reason: "read-write CLI delegation requires an explicit non-empty write_paths access manifest".to_string(),
                coaching: "Retry with working_dir/read_paths for readable context and exact write_paths for mutation authority.".to_string(),
            })
        }
        _ => None,
    }
}

/// Keep only the most-specific exact paths. If both a cwd ancestor and an
/// artifact below it were named, the ancestor is execution context rather than
/// an implicit blanket write grant.
pub(super) fn narrowest_authorized_path_scopes(scopes: &[String]) -> Vec<String> {
    let mut result = scopes
        .iter()
        .filter(|scope| !scope.trim().is_empty())
        .filter(|scope| {
            !scopes.iter().any(|other| {
                scope != &other
                    && std::path::Path::new(other).starts_with(std::path::Path::new(scope))
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    result.sort();
    result.dedup();
    result
}

pub(super) fn tool_is_currently_exposed(tool_defs: &[Value], tool_name: &str) -> bool {
    tool_defs.iter().any(|def| {
        def.get("function")
            .and_then(|function| function.get("name"))
            .and_then(|name| name.as_str())
            .is_some_and(|exposed_name| exposed_name == tool_name)
    })
}

pub(super) fn blocked_for_untrusted_external_reference_message(
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

pub(super) fn allow_scaffold_parent_dir_for_target(
    tool_name: &str,
    allowed_target: &ToolTargetHint,
    candidate_target: &ToolTargetHint,
) -> bool {
    if tool_name != "run_command" {
        return false;
    }
    if crate::execution::active_execution_backend().kind() != crate::execution::BackendKind::Local {
        let Some(scope_path) =
            crate::execution::normalize_active_path_lexically(&allowed_target.value).ok()
        else {
            return false;
        };
        let Some(candidate_path) =
            crate::execution::normalize_active_path_lexically(&candidate_target.value).ok()
        else {
            return false;
        };
        return scope_path.parent().as_ref() == Some(&candidate_path);
    }
    let Some(scope_path) = crate::tools::fs_utils::validate_path(&allowed_target.value).ok() else {
        return false;
    };
    if scope_path.is_dir() {
        return false;
    }
    let Some(candidate_path) = crate::tools::fs_utils::validate_path(&candidate_target.value).ok()
    else {
        return false;
    };
    scope_path
        .parent()
        .is_some_and(|parent| parent.is_dir() && candidate_path == parent)
}

pub(super) fn target_hint_allowed_for_step(
    tool_name: &str,
    allowed_target: &ToolTargetHint,
    candidate_target: &ToolTargetHint,
) -> bool {
    match (&allowed_target.kind, &candidate_target.kind) {
        (ToolTargetHintKind::ResourceId, ToolTargetHintKind::ResourceId) => {
            allowed_target.value == candidate_target.value
        }
        (ToolTargetHintKind::Url, ToolTargetHintKind::Url) => allowed_target
            .value
            .eq_ignore_ascii_case(&candidate_target.value),
        (
            ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope,
            ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope,
        ) => {
            allowed_target.value == candidate_target.value
                || scope_allows_project_dir(&allowed_target.value, &candidate_target.value)
                || allow_scaffold_parent_dir_for_target(tool_name, allowed_target, candidate_target)
        }
        _ => false,
    }
}

pub(super) fn json_contains_string_value(value: &Value, expected: &str) -> bool {
    match value {
        Value::String(candidate) => candidate.eq_ignore_ascii_case(expected),
        Value::Array(items) => items
            .iter()
            .any(|item| json_contains_string_value(item, expected)),
        Value::Object(map) => map
            .values()
            .any(|value| json_contains_string_value(value, expected)),
        _ => false,
    }
}

pub(super) fn linear_intent_step_matches_tool_call(
    step: &crate::agent::execution_state::LinearIntentStep,
    tool_name: &str,
    effective_arguments: &str,
) -> bool {
    if !step.tool.eq_ignore_ascii_case(tool_name) {
        return false;
    }
    if step.target.is_empty() {
        return true;
    }

    let candidate_targets = extract_target_hints_from_arguments(effective_arguments);
    if candidate_targets
        .iter()
        .any(|target| target.value.eq_ignore_ascii_case(&step.target))
    {
        return true;
    }

    serde_json::from_str::<Value>(effective_arguments)
        .ok()
        .is_some_and(|value| json_contains_string_value(&value, &step.target))
}

pub(super) fn target_scope_violation_for_tool_call(
    tool_name: &str,
    effective_arguments: &str,
    step_plan: &StepExecutionPlan,
) -> Option<String> {
    if !step_plan.target_scope.hard_fail_outside_scope
        || step_plan.target_scope.allowed_targets.is_empty()
    {
        return None;
    }

    // The compiled plan owns the prepared write set. Re-extracting every path
    // argument here would turn execution cwd and readable context into implied
    // mutation targets. Legacy/test plans without a prepared set retain the
    // conservative argument fallback.
    let candidate_targets = if step_plan.expected_targets.is_empty() {
        extract_target_hints_from_arguments(effective_arguments)
    } else {
        step_plan.expected_targets.clone()
    };

    let violations: Vec<String> = candidate_targets
        .iter()
        .filter(|candidate_target| {
            !step_plan
                .target_scope
                .allowed_targets
                .iter()
                .any(|allowed_target| {
                    target_hint_allowed_for_step(tool_name, allowed_target, candidate_target)
                })
        })
        .map(|target| target.value.clone())
        .collect();
    if violations.is_empty() {
        None
    } else {
        let allowed_targets = step_plan
            .target_scope
            .allowed_targets
            .iter()
            .map(|target| target.value.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!(
            "target scope lock violation (allowed target(s): {}, requested target(s): {})",
            allowed_targets,
            violations.join(", ")
        ))
    }
}

/// Verify a call's declared read/write capability request against the task's
/// authority before I/O. Mixed operations must satisfy both sets; execution
/// cwd is readable context and is never an implicit write grant.
pub(super) fn access_manifest_scope_violation(
    _tool_name: &str,
    call: &crate::traits::ToolCallAccessManifest,
    task: Option<&crate::traits::ToolCallAccessManifest>,
    fallback_scopes: &[String],
) -> Option<String> {
    // This layer attenuates an existing capability; it does not create the
    // authority boundary itself. If neither the semantic task contract nor a
    // channel workspace supplied a boundary, leave authorization to the
    // ordinary tool policy/approval layer. Treating an absent optional
    // manifest as an empty grant turns this composable check into deny-all.
    if task.is_none() && fallback_scopes.is_empty() {
        return None;
    }

    let fallback = fallback_scopes
        .iter()
        .filter_map(|scope| ToolTargetHint::new(ToolTargetHintKind::ProjectScope, scope.clone()))
        .collect::<Vec<_>>();
    let (read_grants, write_grants) = if let Some(task) = task {
        let mut reads = task.read_targets.clone();
        if let Some(cwd) = task.execution_cwd.as_deref() {
            if let Some(cwd) = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, cwd) {
                if !reads.contains(&cwd) {
                    reads.push(cwd);
                }
            }
        }
        (reads, task.write_targets.clone())
    } else {
        (fallback.clone(), fallback)
    };

    // Filesystem capabilities are monotone: a directory grant may authorize
    // descendants, while an exact-path grant never authorizes its parent or a
    // sibling. Project discovery has a separate near-ancestor convenience
    // rule; reusing it here would widen write authority.
    let capability_grant_allows =
        |grant: &ToolTargetHint, candidate: &ToolTargetHint| match (&grant.kind, &candidate.kind) {
            (ToolTargetHintKind::ResourceId, ToolTargetHintKind::ResourceId) => {
                grant.value == candidate.value
            }
            (ToolTargetHintKind::Url, ToolTargetHintKind::Url) => {
                grant.value.eq_ignore_ascii_case(&candidate.value)
            }
            (ToolTargetHintKind::Path, ToolTargetHintKind::Path) => grant.value == candidate.value,
            (
                ToolTargetHintKind::ProjectScope,
                ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope,
            ) => {
                let grant = std::path::Path::new(&grant.value);
                let candidate = std::path::Path::new(&candidate.value);
                candidate == grant || candidate.starts_with(grant)
            }
            _ => false,
        };
    let outside = |candidate: &ToolTargetHint, grants: &[ToolTargetHint]| {
        grants.is_empty()
            || !grants
                .iter()
                .any(|grant| capability_grant_allows(grant, candidate))
    };
    let invalid_reads = call
        .read_targets
        .iter()
        .filter(|candidate| outside(candidate, &read_grants))
        .map(|target| target.value.clone())
        .collect::<Vec<_>>();
    let invalid_writes = call
        .write_targets
        .iter()
        .filter(|candidate| outside(candidate, &write_grants))
        .map(|target| target.value.clone())
        .collect::<Vec<_>>();
    if invalid_reads.is_empty() && invalid_writes.is_empty() {
        return None;
    }
    Some(format!(
        "task filesystem capability violation (unauthorized reads: {}; unauthorized writes: {})",
        if invalid_reads.is_empty() {
            "none".to_string()
        } else {
            invalid_reads.join(", ")
        },
        if invalid_writes.is_empty() {
            "none".to_string()
        } else {
            invalid_writes.join(", ")
        }
    ))
}

pub(super) fn is_hard_policy_tool_budget_reached(
    total_tool_calls_attempted: usize,
    policy_tool_budget: usize,
) -> bool {
    policy_tool_budget > 0 && total_tool_calls_attempted >= policy_tool_budget
}

pub(super) fn tool_result_indicates_background_detach(
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

pub(super) fn build_background_detach_ack(
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

pub(super) fn run_command_policy_block_requires_terminal(result_text: &str) -> bool {
    let lower = result_text.to_ascii_lowercase();
    lower.contains("safe command list")
        || lower.contains("use 'terminal' for this command")
        || lower.contains("use `terminal`")
        || lower.contains("shell operators")
        || lower.contains("daemonization primitives are blocked in run_command")
}

pub(super) fn build_terminal_fallback_arguments_from_run_command(
    raw_arguments: &str,
) -> Option<String> {
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
    let mut terminal_args = json!({
        "action": "run",
        "command": command,
    });
    if let Some(working_dir) = working_dir {
        terminal_args["working_dir"] = json!(working_dir);
    }
    Some(terminal_args.to_string())
}

pub(super) fn is_trivial_success_excerpt(s: &str) -> bool {
    let lower = s.trim().to_ascii_lowercase();
    lower.is_empty()
        || lower == "ok"
        || lower == "done"
        || lower == "success"
        || lower == "completed"
        || lower == "completed successfully"
        || lower == "request completed successfully"
}

pub(super) fn summarize_completed_tool_result(result_text: &str) -> String {
    let summary = crate::traits::first_primary_message_line(result_text, &[])
        .filter(|line| !line.trim().is_empty())
        .unwrap_or_else(|| "Completed".to_string());
    truncate_str(summary.trim(), TOOL_COMPLETE_SUMMARY_MAX_CHARS)
}

pub(super) fn has_nonzero_exit_code(text: &str) -> bool {
    // Detect "[exit code: N]" where N != 0.
    if let Some(pos) = text.to_ascii_lowercase().find("[exit code:") {
        let after = &text[pos + 11..];
        let code_str: String = after
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ' ')
            .collect();
        if let Ok(code) = code_str.trim().parse::<i32>() {
            return code != 0;
        }
    }
    false
}

pub(super) fn build_external_action_completion_ack(result_text: &str) -> String {
    let primary = crate::traits::extract_primary_message_content(result_text, &[]);
    let filtered: String = primary
        .lines()
        .filter(|l| !crate::utils::is_internal_scaffolding_line(l))
        .collect::<Vec<_>>()
        .join("\n");
    let excerpt = filtered.trim();
    let has_error = has_nonzero_exit_code(excerpt);
    let status = if has_error {
        "The requested action finished with errors."
    } else {
        "The requested action completed successfully."
    };
    if excerpt.is_empty() || is_trivial_success_excerpt(excerpt) {
        status.to_string()
    } else {
        format!(
            "{}\n\nLatest result:\n{}",
            status,
            truncate_with_note(excerpt, EXTERNAL_ACTION_ACK_MAX_CHARS)
        )
    }
}

/// Derive the USER-facing form of an external-action ack. The full ack (with
/// its "Latest result:" excerpt) is model-facing context; when an LLM timeout
/// forces the ack to ship verbatim as the reply, the excerpt must not be a
/// raw data dump (observed live: 500 chars of clinical-trials JSON). Keep the
/// excerpt only when it reads like a short prose result; otherwise replace it
/// with an honest offer to summarize on request.
pub(crate) fn user_facing_external_action_ack(ack: &str) -> String {
    const MARKER: &str = "\n\nLatest result:\n";
    let Some((status, excerpt)) = ack.split_once(MARKER) else {
        return ack.to_string();
    };
    let excerpt = excerpt.trim();
    let short_prose = crate::agent::response_analysis::is_short_prose_excerpt(excerpt);
    if short_prose {
        return ack.to_string();
    }
    format!(
        "{} I wasn't able to finish writing up the details — ask me and I'll summarize the result.",
        status.trim()
    )
}

pub(super) fn should_build_external_action_ack(result_text: &str) -> bool {
    let primary = crate::traits::extract_primary_message_content(result_text, &[]);
    let lower = primary.trim_start().to_ascii_lowercase();
    !lower.starts_with("request blocked:")
        && !lower.starts_with("blocked:")
        && !lower.starts_with("[system] blocked:")
        && !lower.starts_with("error:")
        && !lower.starts_with("failed to ")
}

pub(super) fn should_refresh_external_action_ack(
    background_detached: bool,
    semantics: &ToolCallSemantics,
    tool_has_external_side_effect: bool,
    verified_observation: bool,
    successful_external_mutation_count: usize,
    failed_external_mutation_count: usize,
    result_text: &str,
) -> bool {
    !background_detached
        && ((semantics.mutates_state() && tool_has_external_side_effect)
            || (verified_observation
                && successful_external_mutation_count > 0
                && failed_external_mutation_count == 0))
        && should_build_external_action_ack(result_text)
}

pub(in crate::agent) fn tool_result_contains_verifiable_evidence(
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

/// A completed observation can be evidenced by either substantive returned
/// content or authoritative structured outcome metadata. The latter matters
/// for intentionally negative observations such as a command exiting 1 with
/// empty stdout/stderr: absence of stream bytes must not erase the exit
/// receipt, while generic prose alone still cannot manufacture evidence.
pub(in crate::agent) fn tool_result_or_metadata_contains_verifiable_evidence(
    semantics: &ToolCallSemantics,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> bool {
    tool_result_contains_verifiable_evidence(semantics, result_text)
        || (semantics.can_verify_with_result_content()
            && metadata.outcome_status.is_some()
            && (metadata.exit_code.is_some()
                || metadata.http_status.is_some()
                || metadata.contract_rejected))
}

/// Complete adapter-supplied result metadata with the exact semantics compiled
/// for the dispatched call. Adapters own domain outcomes; the central runtime
/// owns evidence routing, so neither side may accidentally erase the other.
pub(in crate::agent) fn complete_tool_result_semantics(
    tool_name: &str,
    arguments: &str,
    registered_call_semantics: &ToolCallSemantics,
    metadata: &mut crate::traits::ToolCallMetadata,
) {
    metadata
        .semantics
        .merge_missing_from(registered_call_semantics.clone());
    if metadata.semantics.evidence.is_empty() {
        metadata.semantics.evidence =
            crate::agent::inquiry::evidence_capabilities_for_tool_call(tool_name, arguments);
    }
    if metadata.semantics.evidence.is_empty() && metadata.semantics.observes_state() {
        metadata.semantics.evidence =
            crate::agent::inquiry::evidence_capabilities_from_target_hints(
                &metadata.semantics.target_hints,
            );
    }
    if metadata.semantics.observes_state()
        && !metadata.semantics.evidence.is_empty()
        && metadata.semantics.verification_mode == crate::traits::ToolVerificationMode::None
    {
        metadata.semantics.verification_mode = crate::traits::ToolVerificationMode::ResultContent;
    }
}

fn normalized_path_value(value: &str) -> Option<String> {
    let raw = value.trim().replace('\\', "/");
    if raw.is_empty() {
        return None;
    }
    let absolute = raw.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for part in raw.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            value => parts.push(value),
        }
    }
    let joined = parts.join("/");
    Some(if absolute {
        format!("/{joined}")
    } else {
        joined
    })
}

fn normalized_url_value(value: &str) -> Option<String> {
    let mut url = reqwest::Url::parse(value.trim()).ok()?;
    url.set_fragment(None);
    if url.path() != "/" {
        let path = url.path().trim_end_matches('/').to_string();
        url.set_path(&path);
    }
    Some(url.to_string())
}

pub(super) fn tool_target_hint_matches_contract_target(
    target_hint: &ToolTargetHint,
    contract_target: &VerificationTarget,
) -> bool {
    let compatible_kind = matches!(
        (target_hint.kind, contract_target.kind),
        (ToolTargetHintKind::Url, VerificationTargetKind::Url)
            | (ToolTargetHintKind::Path, VerificationTargetKind::Path)
            | (
                ToolTargetHintKind::ProjectScope,
                VerificationTargetKind::Path
            )
    );
    if !compatible_kind {
        return false;
    }

    match (target_hint.kind, contract_target.kind) {
        (ToolTargetHintKind::Url, VerificationTargetKind::Url) => {
            normalized_url_value(&target_hint.value) == normalized_url_value(&contract_target.value)
        }
        (ToolTargetHintKind::Path, VerificationTargetKind::Path) => {
            normalized_path_value(&target_hint.value)
                == normalized_path_value(&contract_target.value)
        }
        // A project-inspection tool may report its exact directory target as a
        // project scope. It verifies a path target only when the identities are
        // equal; observing an ancestor project root is still insufficient.
        (ToolTargetHintKind::ProjectScope, VerificationTargetKind::Path) => {
            normalized_path_value(&target_hint.value)
                == normalized_path_value(&contract_target.value)
        }
        (ToolTargetHintKind::ResourceId, _) => false,
        _ => false,
    }
}

pub(super) fn verification_target_matches_haystack(
    target: &VerificationTarget,
    haystack: &str,
) -> bool {
    let haystack = haystack.replace('\\', "/");
    let needle = match target.kind {
        VerificationTargetKind::Url => normalized_url_value(&target.value),
        VerificationTargetKind::Path => normalized_path_value(&target.value),
    };
    let Some(needle) = needle else {
        return false;
    };
    if needle.is_empty() {
        return false;
    }
    if haystack.contains(&needle) {
        return true;
    }

    // Completion targets are normalized to absolute paths, while shell calls
    // commonly preserve an equivalent home-relative spelling. Treat those
    // spellings as the same target so a successful command such as
    // `find "$HOME/projects" ...` can satisfy a contract inferred from
    // `~/projects`.
    if target.kind == VerificationTargetKind::Path {
        let Some(home) = dirs::home_dir() else {
            return false;
        };
        let Some(home) = normalized_path_value(&home.to_string_lossy()) else {
            return false;
        };
        let Some(relative) = needle
            .strip_prefix(&home)
            .and_then(|suffix| suffix.strip_prefix('/'))
            .filter(|suffix| !suffix.is_empty())
        else {
            return false;
        };
        return [
            format!("$HOME/{relative}"),
            format!("${{HOME}}/{relative}"),
            format!("~/{relative}"),
        ]
        .iter()
        .any(|spelling| haystack.contains(spelling));
    }

    false
}

pub(in crate::agent) fn observation_matches_completion_contract(
    contract: &CompletionContract,
    semantics: &ToolCallSemantics,
    raw_arguments: &str,
    _result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
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

    // A tool may be called with a workspace-relative spelling while the
    // structural contract stores the resolved absolute identity. The typed
    // read receipt is the authoritative reconciliation point; do not infer it
    // from the returned prose or from the daemon process cwd.
    if let Some(read) = metadata.read_file.as_ref() {
        let receipt_targets = [&read.canonical_path, &read.display_path];
        if receipt_targets.iter().any(|value| {
            ToolTargetHint::new(ToolTargetHintKind::Path, value.as_str()).is_some_and(|hint| {
                contract
                    .verification_targets
                    .iter()
                    .any(|target| tool_target_hint_matches_contract_target(&hint, target))
            })
        }) {
            return true;
        }
    }

    // Arguments are harness-controlled structured evidence. Free-form result
    // prose is not used for target identity because it may mention another
    // file/URL with the same basename or quote the requested target without
    // actually observing it.
    let mut haystacks = vec![raw_arguments.to_string()];
    if let Some(command) = extract_command_from_args(raw_arguments) {
        haystacks.push(command);
    }

    contract.verification_targets.iter().any(|target| {
        haystacks
            .iter()
            .any(|haystack| verification_target_matches_haystack(target, haystack))
    })
}

fn requirement_target_matches(
    requirement: &RequestEvidenceRequirement,
    semantics: &ToolCallSemantics,
    raw_arguments: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> bool {
    let Some(target) = requirement.target.as_ref() else {
        return true;
    };
    let contract_target = VerificationTarget {
        kind: match target.kind {
            RequestVerificationTargetKind::Url => VerificationTargetKind::Url,
            RequestVerificationTargetKind::Path => VerificationTargetKind::Path,
        },
        value: target.value.clone(),
    };
    if semantics
        .target_hints
        .iter()
        .any(|hint| tool_target_hint_matches_contract_target(hint, &contract_target))
    {
        return true;
    }
    if let Some(read) = metadata.read_file.as_ref() {
        if [&read.canonical_path, &read.display_path]
            .iter()
            .any(|value| {
                ToolTargetHint::new(ToolTargetHintKind::Path, value.as_str()).is_some_and(|hint| {
                    tool_target_hint_matches_contract_target(&hint, &contract_target)
                })
            })
        {
            return true;
        }
    }

    let mut haystacks = vec![raw_arguments.to_string()];
    if let Some(command) = extract_command_from_args(raw_arguments) {
        haystacks.push(command);
    }
    haystacks
        .iter()
        .any(|haystack| verification_target_matches_haystack(&contract_target, haystack))
}

/// Return the exact machine-checkable evidence obligations supported by one
/// successful observation receipt. Free-form subject words never decide task
/// completion; only typed capabilities, receipt fields, and structural target
/// identity participate here.
pub(in crate::agent) fn matching_evidence_requirement_indices(
    contract: &CompletionContract,
    requested_tool_name: &str,
    semantics: &ToolCallSemantics,
    raw_arguments: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> Vec<usize> {
    let read_receipt_is_compatible = metadata
        .read_file
        .as_ref()
        .is_none_or(|read| read_receipt_covers_requested_selection(raw_arguments, read));
    contract
        .evidence_requirements
        .iter()
        .enumerate()
        .filter_map(|(index, requirement)| {
            let capability_matches =
                crate::agent::inquiry::requirement_is_exact_invocation(requirement)
                    || semantics.evidence.iter().any(|capability| {
                        crate::agent::inquiry::capability_supports_requirement(
                            capability,
                            requirement,
                        )
                    });
            let receipt_matches =
                receipt_predicate_matches(requirement, requested_tool_name, result_text, metadata);
            (read_receipt_is_compatible
                && capability_matches
                && receipt_matches
                && requirement_target_matches(requirement, semantics, raw_arguments, metadata))
            .then_some(index)
        })
        .collect()
}

/// Associate a nonterminal invocation with the obligations it may eventually
/// close. This deliberately ignores terminal outcome fields (exit/status,
/// output, rejection disposition), because those facts do not exist yet, but
/// still requires exact tool identity, typed evidence capability, and target
/// compatibility. The returned IDs are ownership edges, never completion.
pub(in crate::agent) fn pending_evidence_requirement_indices(
    contract: &CompletionContract,
    requested_tool_name: &str,
    semantics: &ToolCallSemantics,
    raw_arguments: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> Vec<usize> {
    contract
        .evidence_requirements
        .iter()
        .enumerate()
        .filter_map(|(index, requirement)| {
            let capability_matches =
                crate::agent::inquiry::requirement_is_exact_invocation(requirement)
                    || semantics.evidence.iter().any(|capability| {
                        crate::agent::inquiry::capability_supports_requirement(
                            capability,
                            requirement,
                        )
                    });
            let tool_matches = requirement.receipt.as_ref().is_none_or(|receipt| {
                receipt.tool_names.is_empty()
                    || receipt
                        .tool_names
                        .iter()
                        .any(|name| name == requested_tool_name)
            });
            (capability_matches
                && tool_matches
                && requirement_target_matches(requirement, semantics, raw_arguments, metadata))
            .then_some(index)
        })
        .collect()
}

fn receipt_predicate_matches(
    requirement: &RequestEvidenceRequirement,
    requested_tool_name: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> bool {
    let Some(receipt) = requirement.receipt.as_ref() else {
        return true;
    };
    let tool_matches = receipt.tool_names.is_empty()
        || receipt.tool_names.iter().any(|name| {
            name == requested_tool_name || metadata.effective_tool_name.as_deref() == Some(name)
        });
    // Exit codes are meaningful only for process receipts. An assessor adding
    // a process-only field to a management/API receipt must not make an
    // otherwise valid named invocation impossible to prove.
    let exit_matches = receipt.exit_codes.is_empty()
        || metadata.receipt_kind != crate::traits::ToolReceiptKind::Process
        || metadata
            .exit_code
            .is_some_and(|actual| receipt.exit_codes.contains(&actual));
    let outcome_matches = receipt.outcome_statuses.is_empty()
        || metadata
            .outcome_status
            .is_some_and(|actual| receipt.outcome_statuses.contains(&actual));
    let rejection_matches = receipt
        .contract_rejected
        .is_none_or(|expected| metadata.contract_rejected == expected);
    let has_output = metadata
        .result_provenance
        .as_ref()
        .is_some_and(|provenance| provenance.authoritative_chars > 0)
        || !result_text.trim().is_empty();
    let output_matches = !receipt.requires_output || has_output;

    tool_matches && exit_matches && outcome_matches && rejection_matches && output_matches
}

pub(in crate::agent) fn evidence_requirement_accepts_nonstandard_outcome(
    requirement: &RequestEvidenceRequirement,
    metadata: &crate::traits::ToolCallMetadata,
) -> bool {
    requirement.receipt.as_ref().is_some_and(|receipt| {
        metadata.outcome_status.is_some_and(|actual| {
            receipt.outcome_statuses.contains(&actual)
                || (metadata.contract_rejected && receipt.contract_rejected == Some(true))
        })
    })
}

/// Compatibility shim for persisted schema-v7 marker state. Content markers
/// are advisory and cannot close a lifecycle obligation.
pub(in crate::agent) fn accumulate_evidence_requirement_marker_matches(
    _contract: &CompletionContract,
    _progress: &mut CompletionProgress,
    _semantics: &ToolCallSemantics,
    _raw_arguments: &str,
    _result_text: &str,
    _metadata: &crate::traits::ToolCallMetadata,
) -> Vec<usize> {
    Vec::new()
}

/// A read receipt may close a content obligation only when the tool actually
/// returned the range selected by the call. This is intentionally based on
/// typed request/result metadata; a path mention or prose summary is never
/// enough to turn a partial/tail read into evidence for a bounded range.
fn read_receipt_covers_requested_selection(
    raw_arguments: &str,
    read: &crate::traits::ReadFileResultMetadata,
) -> bool {
    use crate::traits::ReadFileSelectionMetadata;

    if read.truncated || !read_receipt_has_complete_content(read) {
        return false;
    }
    let Ok(args) = serde_json::from_str::<serde_json::Value>(raw_arguments) else {
        return false;
    };
    let requested_start = args.get("start_line").and_then(serde_json::Value::as_u64);
    let requested_end = args.get("end_line").and_then(serde_json::Value::as_u64);
    let requested_tail = args
        .get("tail_lines")
        .or_else(|| args.get("last_lines"))
        .or_else(|| args.get("last_n_lines"))
        .and_then(serde_json::Value::as_u64);

    if requested_tail.is_some() && (requested_start.is_some() || requested_end.is_some()) {
        return false;
    }

    match (
        requested_start,
        requested_end,
        requested_tail,
        &read.selection,
    ) {
        (None, None, None, ReadFileSelectionMetadata::Full) => {
            read.total_lines == 0
                || read.returned_start_line == Some(1)
                    && read.returned_end_line == Some(read.total_lines)
        }
        (
            start,
            end,
            None,
            ReadFileSelectionMetadata::BoundedRange {
                start_line,
                end_line,
            },
        ) => {
            let requested_start = start.unwrap_or(1) as usize;
            let requested_end = end.unwrap_or(u64::MAX) as usize;
            *start_line == requested_start
                && *end_line == requested_end
                && read
                    .returned_start_line
                    .is_some_and(|line| line <= requested_start)
                && read
                    .returned_end_line
                    .is_some_and(|line| line >= requested_end)
        }
        (start, None, None, ReadFileSelectionMetadata::OpenEndedRange { start_line }) => {
            let requested_start = start.unwrap_or(1) as usize;
            *start_line == requested_start
                && read
                    .returned_start_line
                    .is_some_and(|line| line <= requested_start)
                && read.returned_end_line == Some(read.total_lines)
        }
        (None, None, Some(count), ReadFileSelectionMetadata::Tail { requested_lines }) => {
            *requested_lines == count as usize
                && (read.total_lines == 0
                    || read.returned_end_line == Some(read.total_lines)
                        && read
                            .returned_start_line
                            .zip(read.returned_end_line)
                            .is_some_and(|(start, end)| {
                                end.saturating_sub(start).saturating_add(1)
                                    == (count as usize).min(read.total_lines)
                            }))
        }
        _ => false,
    }
}

fn read_receipt_has_complete_content(read: &crate::traits::ReadFileResultMetadata) -> bool {
    if read.total_lines == 0 {
        return read.selected_lines.is_empty()
            && read.returned_start_line.is_none()
            && read.returned_end_line.is_none();
    }
    read.returned_start_line
        .zip(read.returned_end_line)
        .is_some_and(|(start, end)| {
            start > 0
                && end >= start
                && end <= read.total_lines
                && read.selected_lines.len() == end - start + 1
        })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_outcome_metadata_retains_registered_observation_semantics() {
        let arguments = r#"{"action":"search","query":"synthetic pets"}"#;
        let registered = crate::traits::semantics_for_exact_read_actions(
            arguments,
            &["search"],
            crate::traits::ToolMutationEffects::NONE,
        );
        let mut metadata = crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult),
            ..crate::traits::ToolCallMetadata::default()
        };

        complete_tool_result_semantics("manage_memories", arguments, &registered, &mut metadata);

        assert!(metadata.semantics.observes_state());
        assert!(metadata.semantics.evidence.iter().any(|capability| {
            capability.scope == crate::traits::ToolSemanticScope::UserMemory
        }));
        assert!(tool_result_or_metadata_contains_verifiable_evidence(
            &metadata.semantics,
            "No memories matching 'synthetic pets'.",
            &metadata,
        ));
    }

    #[test]
    fn structured_negative_process_outcome_is_evidence_without_stream_bytes() {
        let semantics = crate::tools::command_semantics::classify_shell_command("/usr/bin/false");
        let metadata = crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult),
            exit_code: Some(1),
            semantics: semantics.clone(),
            ..crate::traits::ToolCallMetadata::default()
        };

        assert!(tool_result_or_metadata_contains_verifiable_evidence(
            &semantics, "", &metadata,
        ));
    }

    #[test]
    fn typed_process_outcome_is_not_vetoed_by_reply_format_markers() {
        use crate::traits::{
            EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
            RequestReceiptPredicate, ToolOutcomeStatus, ToolSemanticScope,
        };
        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![RequestEvidenceRequirement {
                summary: "Observe the process outcome".to_string(),
                acceptable_scopes: vec![ToolSemanticScope::HostLocal],
                purpose: EvidencePurpose::Outcome,
                minimum_authority: EvidenceAuthority::Direct,
                temporal_scope: EvidenceTemporalScope::Current,
                required_content_markers: Vec::new(),
                receipt: Some(RequestReceiptPredicate {
                    tool_names: vec!["run_command".to_string()],
                    exit_codes: vec![1],
                    outcome_statuses: vec![ToolOutcomeStatus::CompletedWithNegativeResult],
                    requires_output: false,
                    contract_rejected: Some(false),
                }),
                target: None,
            }],
            ..CompletionContract::default()
        };
        let raw_arguments = r#"{"command":"/usr/bin/false","working_dir":"/tmp"}"#;
        let registered = crate::tools::command_semantics::classify_shell_command("/usr/bin/false");
        let mut metadata = crate::traits::ToolCallMetadata {
            receipt_kind: crate::traits::ToolReceiptKind::Process,
            outcome_status: Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult),
            exit_code: Some(1),
            ..crate::traits::ToolCallMetadata::default()
        };
        complete_tool_result_semantics("terminal", raw_arguments, &registered, &mut metadata);

        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "run_command",
                &metadata.semantics,
                raw_arguments,
                "",
                &metadata,
            ),
            [0]
        );

        let mut mismatched = contract.clone();
        mismatched.evidence_requirements[0]
            .receipt
            .as_mut()
            .unwrap()
            .exit_codes = vec![0];
        assert!(matching_evidence_requirement_indices(
            &mismatched,
            "run_command",
            &metadata.semantics,
            raw_arguments,
            "untrusted process text says exit=0",
            &metadata,
        )
        .is_empty());
    }

    #[test]
    fn command_identity_and_output_presence_are_typed_receipt_fields() {
        use crate::traits::{
            EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
            RequestReceiptPredicate, ToolOutcomeStatus, ToolSemanticScope,
        };
        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![RequestEvidenceRequirement {
                summary: "Observe the current process working directory".to_string(),
                acceptable_scopes: vec![ToolSemanticScope::HostLocal],
                purpose: EvidencePurpose::CurrentState,
                minimum_authority: EvidenceAuthority::Direct,
                temporal_scope: EvidenceTemporalScope::Current,
                required_content_markers: Vec::new(),
                receipt: Some(RequestReceiptPredicate {
                    tool_names: vec!["run_command".to_string()],
                    exit_codes: vec![0],
                    outcome_statuses: vec![ToolOutcomeStatus::Succeeded],
                    requires_output: true,
                    contract_rejected: Some(false),
                }),
                target: None,
            }],
            ..CompletionContract::default()
        };
        let semantics = ToolCallSemantics::observation().with_evidence(vec![
            crate::traits::ToolEvidenceCapability::new(
                ToolSemanticScope::HostLocal,
                &[EvidencePurpose::CurrentState],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            ),
        ]);
        let metadata = crate::traits::ToolCallMetadata {
            receipt_kind: crate::traits::ToolReceiptKind::Process,
            outcome_status: Some(crate::traits::ToolOutcomeStatus::Succeeded),
            exit_code: Some(0),
            result_provenance: Some(crate::traits::ToolResultProvenance {
                authoritative_chars: 35,
                ..crate::traits::ToolResultProvenance::default()
            }),
            semantics: semantics.clone(),
            ..crate::traits::ToolCallMetadata::default()
        };

        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "run_command",
                &semantics,
                r#"{"command":"/bin/pwd","working_dir":"/synthetic/project"}"#,
                "/synthetic/project",
                &metadata,
            ),
            [0]
        );
    }

    #[test]
    fn process_only_predicate_fields_do_not_reject_nonprocess_receipts() {
        use crate::traits::{
            EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
            RequestReceiptPredicate, ToolOutcomeStatus, ToolReceiptKind, ToolSemanticScope,
        };
        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![RequestEvidenceRequirement {
                summary: "Read canonical mandate state".to_string(),
                acceptable_scopes: vec![ToolSemanticScope::GoalState],
                purpose: EvidencePurpose::CurrentState,
                minimum_authority: EvidenceAuthority::Canonical,
                temporal_scope: EvidenceTemporalScope::Current,
                required_content_markers: Vec::new(),
                receipt: Some(RequestReceiptPredicate {
                    tool_names: vec!["manage_mandates".to_string()],
                    exit_codes: vec![0],
                    outcome_statuses: vec![ToolOutcomeStatus::Succeeded],
                    requires_output: true,
                    contract_rejected: Some(false),
                }),
                target: None,
            }],
            ..CompletionContract::default()
        };
        let semantics = ToolCallSemantics::observation().with_evidence(vec![
            crate::traits::ToolEvidenceCapability::new(
                ToolSemanticScope::GoalState,
                &[EvidencePurpose::CurrentState],
                EvidenceAuthority::Canonical,
                EvidenceTemporalScope::Current,
            ),
        ]);
        let metadata = crate::traits::ToolCallMetadata {
            receipt_kind: ToolReceiptKind::Generic,
            outcome_status: Some(ToolOutcomeStatus::Succeeded),
            result_provenance: Some(crate::traits::ToolResultProvenance {
                authoritative_chars: 42,
                ..crate::traits::ToolResultProvenance::default()
            }),
            semantics: semantics.clone(),
            ..crate::traits::ToolCallMetadata::default()
        };

        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "manage_mandates",
                &semantics,
                r#"{"action":"get","id":"synthetic"}"#,
                "opaque canonical state",
                &metadata,
            ),
            [0]
        );
    }
    use crate::agent::execution_state::{
        default_execution_budget, BudgetTier, ExecutionPersistence, ExecutionState, RetryPolicy,
    };
    use crate::traits::ToolCallEffect;

    #[test]
    fn typed_receipt_matching_is_independent_of_result_prose() {
        use crate::traits::{
            EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
            ToolSemanticScope,
        };

        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![
                RequestEvidenceRequirement {
                    summary: "Observe current remote state".to_string(),
                    acceptable_scopes: vec![ToolSemanticScope::ExternalRemote],
                    purpose: EvidencePurpose::CurrentState,
                    minimum_authority: EvidenceAuthority::Direct,
                    temporal_scope: EvidenceTemporalScope::Current,
                    required_content_markers: vec!["state_key".to_string()],
                    receipt: None,
                    target: None,
                },
                RequestEvidenceRequirement {
                    summary: "Determine who executed the earlier action".to_string(),
                    acceptable_scopes: vec![ToolSemanticScope::GoalState],
                    purpose: EvidencePurpose::Attribution,
                    minimum_authority: EvidenceAuthority::Canonical,
                    temporal_scope: EvidenceTemporalScope::Historical,
                    required_content_markers: Vec::new(),
                    receipt: None,
                    target: None,
                },
            ],
            ..CompletionContract::default()
        };

        let external = ToolCallSemantics::observation().with_evidence(
            crate::agent::inquiry::evidence_capabilities_for_tool_call(
                "http_request",
                r#"{"method":"GET","url":"https://example.test/feed"}"#,
            ),
        );
        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "http_request",
                &external,
                "{}",
                "synthetic state_key current state",
                &crate::traits::ToolCallMetadata::default(),
            ),
            [0]
        );
        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "http_request",
                &external,
                "{}",
                "synthetic response without the requested field",
                &crate::traits::ToolCallMetadata::default(),
            ),
            [0],
            "typed scope, purpose, authority, and time decide evidence routing"
        );

        let trace = ToolCallSemantics::observation().with_evidence(
            crate::agent::inquiry::evidence_capabilities_for_tool_call(
                "goal_trace",
                r#"{"action":"tool_trace","task_id":"synthetic-task"}"#,
            ),
        );
        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "goal_trace",
                &trace,
                "{}",
                "synthetic attribution record",
                &crate::traits::ToolCallMetadata::default(),
            ),
            [1]
        );
    }

    #[test]
    fn typed_evidence_scope_closes_without_matching_response_prose() {
        use crate::traits::{
            EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
            ToolSemanticScope,
        };
        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![RequestEvidenceRequirement {
                summary: "Build the requested management audit".to_string(),
                acceptable_scopes: vec![ToolSemanticScope::GoalState],
                purpose: EvidencePurpose::Content,
                minimum_authority: EvidenceAuthority::Canonical,
                temporal_scope: EvidenceTemporalScope::Both,
                required_content_markers: vec![
                    "next_run".to_string(),
                    "objective_control".to_string(),
                ],
                receipt: None,
                target: None,
            }],
            ..CompletionContract::default()
        };
        let semantics = ToolCallSemantics::observation().with_evidence(vec![
            crate::traits::ToolEvidenceCapability {
                scope: ToolSemanticScope::GoalState,
                purposes: vec![EvidencePurpose::Content],
                authority: EvidenceAuthority::Canonical,
                temporal_scope: EvidenceTemporalScope::Both,
            },
        ]);
        let metadata = crate::traits::ToolCallMetadata::default();

        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "manage_goal_tasks",
                &semantics,
                "{}",
                "opaque authoritative payload",
                &metadata,
            ),
            [0]
        );
    }

    fn read_receipt(
        selection: crate::traits::ReadFileSelectionMetadata,
        start: usize,
        end: usize,
        total: usize,
        truncated: bool,
    ) -> crate::traits::ReadFileResultMetadata {
        crate::traits::ReadFileResultMetadata {
            display_path: "/tmp/synthetic.toml".to_string(),
            canonical_path: "/tmp/synthetic.toml".to_string(),
            selection,
            returned_start_line: Some(start),
            returned_end_line: Some(end),
            total_lines: total,
            file_size: 100,
            modified: None,
            selected_lines: (start..=end)
                .map(|line| format!("synthetic line {line}"))
                .collect(),
            truncated,
        }
    }

    #[test]
    fn canonical_read_receipt_reconciles_relative_call_with_absolute_contract() {
        let contract = CompletionContract {
            requires_observation: true,
            verification_targets: vec![VerificationTarget {
                kind: VerificationTargetKind::Path,
                value: "/tmp/synthetic.toml".to_string(),
            }],
            ..CompletionContract::default()
        };
        let arguments = r#"{"path":"synthetic.toml","start_line":1,"end_line":12}"#;
        let semantics = ToolCallSemantics::observation()
            .with_verification_mode(crate::traits::ToolVerificationMode::ResultContent)
            .with_target_hint(ToolTargetHintKind::Path, "synthetic.toml");
        let metadata = crate::traits::ToolCallMetadata {
            read_file: Some(read_receipt(
                crate::traits::ReadFileSelectionMetadata::BoundedRange {
                    start_line: 1,
                    end_line: 12,
                },
                1,
                12,
                40,
                false,
            )),
            ..crate::traits::ToolCallMetadata::default()
        };

        assert!(observation_matches_completion_contract(
            &contract,
            &semantics,
            arguments,
            "synthetic content",
            &metadata,
        ));
    }

    #[test]
    fn rejected_invocation_can_prove_its_typed_outcome_without_proving_content() {
        use crate::traits::{
            EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
            RequestReceiptPredicate, ToolSemanticScope,
        };
        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![RequestEvidenceRequirement {
                summary: "Observe parameter-contract outcome".to_string(),
                acceptable_scopes: vec![ToolSemanticScope::HostLocal],
                purpose: EvidencePurpose::Outcome,
                minimum_authority: EvidenceAuthority::Direct,
                temporal_scope: EvidenceTemporalScope::Current,
                required_content_markers: Vec::new(),
                receipt: Some(RequestReceiptPredicate {
                    tool_names: vec!["read_file".to_string()],
                    contract_rejected: Some(true),
                    ..RequestReceiptPredicate::default()
                }),
                target: None,
            }],
            ..CompletionContract::default()
        };
        let arguments = r#"{"path":"synthetic.toml","start_line":1,"end_line":12,"tail_lines":1}"#;
        let semantics = ToolCallSemantics::observation().with_evidence(
            crate::agent::inquiry::evidence_capabilities_for_tool_call("read_file", arguments),
        );
        let metadata = crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult),
            contract_rejected: true,
            ..crate::traits::ToolCallMetadata::default()
        };

        assert_eq!(
            matching_evidence_requirement_indices(
                &contract,
                "read_file",
                &semantics,
                arguments,
                "range modes are mutually exclusive",
                &metadata,
            ),
            [0]
        );
    }

    #[test]
    fn read_receipt_must_cover_the_exact_requested_range() {
        use crate::traits::ReadFileSelectionMetadata;

        let bounded = read_receipt(
            ReadFileSelectionMetadata::BoundedRange {
                start_line: 1,
                end_line: 12,
            },
            1,
            12,
            40,
            false,
        );
        assert!(read_receipt_covers_requested_selection(
            r#"{"path":"/tmp/synthetic.toml","start_line":1,"end_line":12}"#,
            &bounded,
        ));
        assert!(!read_receipt_covers_requested_selection(
            r#"{"path":"/tmp/synthetic.toml","start_line":1,"end_line":12,"tail_lines":1}"#,
            &bounded,
        ));
        let mut unavailable_content = bounded.clone();
        unavailable_content.selected_lines.clear();
        assert!(!read_receipt_covers_requested_selection(
            r#"{"path":"/tmp/synthetic.toml","start_line":1,"end_line":12}"#,
            &unavailable_content,
        ));

        let tail = read_receipt(
            ReadFileSelectionMetadata::Tail { requested_lines: 1 },
            40,
            40,
            40,
            false,
        );
        assert!(!read_receipt_covers_requested_selection(
            r#"{"path":"/tmp/synthetic.toml","start_line":1,"end_line":12}"#,
            &tail,
        ));

        let truncated = read_receipt(
            ReadFileSelectionMetadata::BoundedRange {
                start_line: 1,
                end_line: 12,
            },
            1,
            12,
            40,
            true,
        );
        assert!(!read_receipt_covers_requested_selection(
            r#"{"path":"/tmp/synthetic.toml","start_line":1,"end_line":12}"#,
            &truncated,
        ));
    }

    #[test]
    fn error_summary_skips_harness_truncation_notice() {
        // Live repro (task 29a8c716): the truncation notice's own wording
        // ("inventing the omitted content is an error") matched the keyword
        // scan, so the harness-injected notice — not the real error — became
        // the ledger's error_summary and was later shipped to the user via
        // the reconciliation fallback reply.
        let result_text = format!(
            "{}\nsome command output line\nError: No such file or directory\n",
            crate::utils::truncation_notice(4000, 4265)
        );
        let summary = extract_error_summary_line(&result_text);
        assert_eq!(summary.as_deref(), Some("Error: No such file or directory"));
    }

    #[test]
    fn error_summary_line_on_clean_content_is_the_real_error() {
        // After the loop-render migration, tool content reaching the ledger
        // never embeds a truncation notice — confirm extraction still finds
        // the real error line on such clean content.
        let result_text = "partial output\nError: disk full";
        assert_eq!(
            extract_error_summary_line(result_text).as_deref(),
            Some("Error: disk full")
        );
    }

    #[test]
    fn error_summary_prefers_cli_stderr_over_wrapper_heading() {
        let result_text = "[UNTRUSTED EXTERNAL DATA from 'cli_agent']\n\
ERROR: CLI agent 'claude' failed (exit code 127).\n\n\
## Failure Details\n\
[stderr] env: claude: No such file or directory\n\n\
## Safe Recovery Options\n\
- Verify the configured executable path\n\
[END UNTRUSTED EXTERNAL DATA]";
        assert_eq!(
            extract_error_summary_line(result_text).as_deref(),
            Some("[stderr] env: claude: No such file or directory")
        );
    }

    #[test]
    fn error_summary_never_returns_an_error_section_heading() {
        let result_text = "## Error Output\n\nERROR: process failed without diagnostics";
        assert_eq!(
            extract_error_summary_line(result_text).as_deref(),
            Some("ERROR: process failed without diagnostics")
        );
    }

    #[test]
    fn error_summary_ignores_recovery_advice_after_failure_details() {
        let result_text = "ERROR: CLI agent failed (exit code 1).\n\n\
## Failure Details\nNo diagnostic output was captured.\n\n\
## Safe Recovery Options\n- Verify authentication";
        assert_eq!(
            extract_error_summary_line(result_text).as_deref(),
            Some("ERROR: CLI agent failed (exit code 1).")
        );
    }

    #[test]
    fn error_summary_is_none_when_only_scaffolding_matches() {
        let result_text = format!(
            "{}\nplain output with no failures\n[SYSTEM] IMPORTANT — The error says: \"something\"\n",
            crate::utils::truncation_notice(4000, 4265)
        );
        assert_eq!(extract_error_summary_line(&result_text), None);
    }

    #[test]
    fn external_action_ack_excludes_rendered_truncation_notice() {
        // Live risk: run.rs now renders the truncation notice into
        // `result_text` before this ack is built. If the ack's embedded
        // excerpt isn't filtered the same way `extract_error_summary_line`
        // is, the raw model-facing notice ships verbatim as the user-facing
        // reply on a post-action provider failure.
        let result_text = format!(
            "Deployed 3 services successfully.\n{}",
            crate::utils::truncation_notice(4000, 4265)
        );
        let ack = build_external_action_completion_ack(&result_text);
        assert!(ack.contains("Deployed 3 services successfully."));
        assert!(!ack.contains("OUTPUT TRUNCATED"));
    }

    #[test]
    fn verified_observation_restores_external_action_ack() {
        let observation = ToolCallSemantics::observation();
        assert!(should_refresh_external_action_ack(
            false,
            &observation,
            false,
            true,
            1,
            0,
            "HTTP 200 OK",
        ));
        assert!(!should_refresh_external_action_ack(
            false,
            &observation,
            false,
            false,
            1,
            0,
            "HTTP 200 OK",
        ));
        assert!(!should_refresh_external_action_ack(
            false,
            &observation,
            false,
            true,
            1,
            1,
            "HTTP 200 OK",
        ));
    }

    #[test]
    fn user_facing_ack_drops_structured_data_dumps() {
        // Live repro (2026-07-02): an LLM timeout after an http_request shipped
        // the model-facing ack verbatim — "The requested action completed
        // successfully.\n\nLatest result:\n[ {\"nctId\": ...500 chars of
        // JSON...]" — straight to the user.
        let json_result = "Fetched trial data.\n[\n  {\n    \"nctId\": \"NCT00000000\",\n    \"title\": \"Synthetic Trial Study of Compound X in Participants With Condition Y\",\n    \"status\": \"COMPLETED\",\n    \"conditions\": [\"Condition Y\"]\n  }\n]";
        let ack = build_external_action_completion_ack(json_result);
        assert!(ack.contains("Latest result:"), "model-facing keeps excerpt");
        let user = user_facing_external_action_ack(&ack);
        assert!(
            !user.contains("nctId") && !user.contains("Latest result:"),
            "user version must not dump structured data: {user}"
        );
        assert!(user.contains("completed successfully"));
        assert!(user.contains("summarize"), "offers a follow-up: {user}");
    }

    #[test]
    fn user_facing_ack_keeps_short_prose_results() {
        let ack = build_external_action_completion_ack("Tweet posted. id=1234567890");
        let user = user_facing_external_action_ack(&ack);
        assert!(
            user.contains("Tweet posted"),
            "short prose results stay: {user}"
        );
    }

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
    fn deterministic_tool_contract_violation_blocks_manage_memories_without_goal_id() {
        let args = r#"{"action":"diagnose_scheduled"}"#;
        let violation = deterministic_tool_contract_violation("manage_memories", args);
        assert!(violation.is_some());
        let violation = violation.expect("violation expected");
        assert!(violation.reason.contains("requires `goal_id`"));
        assert!(violation.coaching.contains("list_scheduled"));
    }

    #[test]
    fn deterministic_tool_contract_violation_allows_manage_memories_with_goal_id() {
        let args = r#"{"action":"diagnose_scheduled","goal_id":"goal-123"}"#;
        let violation = deterministic_tool_contract_violation("manage_memories", args);
        assert!(violation.is_none());
    }

    #[test]
    fn scheduled_execution_text_clears_stale_goal_state_scope() {
        let scope = effective_dialogue_scope_for_tool_execution(
            Some(ToolSemanticScope::GoalState),
            "Scheduled check: Post daily update [SYSTEM: already scheduled and firing now; do not reschedule.]",
            false,
        );
        assert_eq!(scope, None);
    }

    #[test]
    fn finite_scheduled_execution_text_clears_stale_goal_state_scope() {
        let scope = effective_dialogue_scope_for_tool_execution(
            Some(ToolSemanticScope::GoalState),
            "Execute scheduled goal: Publish queued API update [SYSTEM: already scheduled and firing now; do not reschedule.]",
            false,
        );
        assert_eq!(scope, None);
    }

    #[test]
    fn scheduled_goal_question_keeps_goal_state_scope() {
        let scope = effective_dialogue_scope_for_tool_execution(
            Some(ToolSemanticScope::GoalState),
            "What times does the scheduled tweet goal run?",
            false,
        );
        assert_eq!(scope, Some(ToolSemanticScope::GoalState));
    }

    #[test]
    fn scheduled_executor_provenance_clears_keyword_derived_goal_state_scope() {
        let scope = effective_dialogue_scope_for_tool_execution(
            Some(ToolSemanticScope::GoalState),
            "Use the scheduler-admission fix as the topic, then POST once to /2/tweets.",
            true,
        );
        assert_eq!(scope, None);
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
            "Latest post title: Scheduled reflection",
            &crate::traits::ToolCallMetadata::default(),
        ));
    }

    #[test]
    fn same_basename_in_different_directory_does_not_verify_target() {
        let target = VerificationTarget {
            kind: VerificationTargetKind::Path,
            value: "/tmp/project-a/config.toml".to_string(),
        };
        let hint =
            ToolTargetHint::new(ToolTargetHintKind::Path, "/tmp/project-b/config.toml").unwrap();
        assert!(!tool_target_hint_matches_contract_target(&hint, &target));
        assert!(!verification_target_matches_haystack(
            &target,
            "read /tmp/project-b/config.toml"
        ));
    }

    #[test]
    fn shell_home_reference_verifies_normalized_home_target() {
        let home = dirs::home_dir().expect("home directory");
        let target = VerificationTarget {
            kind: VerificationTargetKind::Path,
            value: home.join("projects").to_string_lossy().into_owned(),
        };
        let contract = CompletionContract {
            requires_observation: true,
            verification_targets: vec![target],
            ..CompletionContract::default()
        };
        let semantics = ToolCallSemantics::observation()
            .with_verification_mode(crate::traits::ToolVerificationMode::ResultContent);
        let arguments = serde_json::json!({
            "action": "run",
            "command": "find \"$HOME/projects\" -mindepth 1 -maxdepth 1 -type d | wc -l"
        })
        .to_string();

        assert!(observation_matches_completion_contract(
            &contract,
            &semantics,
            &arguments,
            "213",
            &crate::traits::ToolCallMetadata::default(),
        ));
    }

    #[test]
    fn project_scope_hint_cannot_verify_a_specific_file() {
        let target = VerificationTarget {
            kind: VerificationTargetKind::Path,
            value: "/tmp/project-a/config.toml".to_string(),
        };
        let hint = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, "/tmp/project-a").unwrap();
        assert!(!tool_target_hint_matches_contract_target(&hint, &target));
    }

    #[test]
    fn exact_project_scope_hint_verifies_the_same_directory_path() {
        let target = VerificationTarget {
            kind: VerificationTargetKind::Path,
            value: "/tmp/project-a".to_string(),
        };
        let hint = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, "/tmp/project-a").unwrap();
        assert!(tool_target_hint_matches_contract_target(&hint, &target));
    }

    #[test]
    fn target_scope_violation_flags_out_of_scope_mutation_path() {
        let step_plan = StepExecutionPlan {
            step_id: "step-1".to_string(),
            description: "Edit a scoped file".to_string(),
            plan_version: 1,
            primary_tool: Some("edit_file".to_string()),
            expected_effect: ToolCallEffect::Mutation,
            target_scope: TargetScope {
                allowed_targets: vec![ToolTargetHint::new(
                    ToolTargetHintKind::ProjectScope,
                    "/tmp/project-a",
                )
                .expect("scope target")],
                hard_fail_outside_scope: true,
            },
            expected_targets: Vec::new(),
            retry_policy: RetryPolicy {
                max_attempts: 1,
                allow_tool_invocation_retry: false,
            },
            approval_requirement: ApprovalRequirement::NotNeeded,
            idempotency_key: None,
        };
        let args = r#"{"path":"/tmp/project-b/src/main.rs"}"#;
        let violation = target_scope_violation_for_tool_call("edit_file", args, &step_plan);
        assert!(violation.is_some());
    }

    #[test]
    fn target_scope_violation_blocks_cli_agent_from_switching_projects() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project_a = dir.path().join("project-a");
        let project_b = dir.path().join("project-b");
        std::fs::create_dir_all(&project_a).expect("create project-a");
        std::fs::create_dir_all(&project_b).expect("create project-b");
        std::fs::write(project_a.join("Cargo.toml"), "[package]\nname = \"a\"\n")
            .expect("project-a marker");
        std::fs::write(project_b.join("package.json"), r#"{"name":"b"}"#)
            .expect("project-b marker");

        let step_plan = StepExecutionPlan {
            step_id: "step-1".to_string(),
            description: "Modify only the selected project".to_string(),
            plan_version: 1,
            primary_tool: Some("cli_agent".to_string()),
            expected_effect: ToolCallEffect::Mutation,
            target_scope: TargetScope {
                allowed_targets: vec![ToolTargetHint::new(
                    ToolTargetHintKind::ProjectScope,
                    project_a.to_string_lossy(),
                )
                .expect("scope target")],
                hard_fail_outside_scope: true,
            },
            expected_targets: Vec::new(),
            retry_policy: RetryPolicy {
                max_attempts: 1,
                allow_tool_invocation_retry: false,
            },
            approval_requirement: ApprovalRequirement::Required {
                reason: "cross-project mutation".to_string(),
            },
            idempotency_key: None,
        };
        let args = serde_json::json!({
            "action": "run",
            "working_dir": project_b,
            "prompt": "Edit and deploy the project"
        })
        .to_string();

        assert!(target_scope_violation_for_tool_call("cli_agent", &args, &step_plan).is_some());
    }

    #[test]
    fn mixed_access_manifest_checks_read_and_write_grants_independently() {
        let hint = |kind, value| ToolTargetHint::new(kind, value).expect("target");
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![hint(
                ToolTargetHintKind::Path,
                "/workspace/project/Cargo.toml",
            )],
            write_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/synthetic-result.txt")],
        };
        let valid = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![
                hint(ToolTargetHintKind::ProjectScope, "/tmp"),
                hint(ToolTargetHintKind::Path, "/workspace/project/Cargo.toml"),
            ],
            write_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/synthetic-result.txt")],
        };
        assert!(access_manifest_scope_violation("cli_agent", &valid, Some(&task), &[]).is_none());

        let invalid = crate::traits::ToolCallAccessManifest {
            write_targets: vec![hint(ToolTargetHintKind::Path, "/workspace/project")],
            ..valid
        };
        let violation = access_manifest_scope_violation("cli_agent", &invalid, Some(&task), &[])
            .expect("write must be rejected");
        assert!(violation.contains("/workspace/project"));
    }

    #[test]
    fn execution_cwd_is_readable_but_never_an_implicit_write_grant() {
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: Vec::new(),
            write_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::Path, "/tmp/result.txt").expect("target"),
            ],
        };
        let call = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::ProjectScope, "/tmp").expect("cwd"),
            ],
            write_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::ProjectScope, "/tmp").expect("cwd"),
            ],
        };
        assert!(access_manifest_scope_violation("terminal", &call, Some(&task), &[]).is_some());
    }

    #[test]
    fn absent_access_boundary_does_not_invent_a_global_denial() {
        let call = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::Path, "/tmp/input.txt").expect("read"),
            ],
            write_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::Path, "/tmp/output.txt").expect("write"),
            ],
        };

        assert!(access_manifest_scope_violation("synthetic", &call, None, &[]).is_none());
    }

    #[test]
    fn target_scope_violation_skips_non_hard_fail_observation_steps() {
        let step_plan = StepExecutionPlan {
            step_id: "step-1".to_string(),
            description: "Inspect a path".to_string(),
            plan_version: 1,
            primary_tool: Some("search_files".to_string()),
            expected_effect: ToolCallEffect::Observation,
            target_scope: TargetScope {
                allowed_targets: vec![ToolTargetHint::new(
                    ToolTargetHintKind::ProjectScope,
                    "/tmp/project-a",
                )
                .expect("scope target")],
                hard_fail_outside_scope: false,
            },
            expected_targets: Vec::new(),
            retry_policy: RetryPolicy {
                max_attempts: 1,
                allow_tool_invocation_retry: true,
            },
            approval_requirement: ApprovalRequirement::NotNeeded,
            idempotency_key: None,
        };
        let args = r#"{"path":"/tmp/project-b/src"}"#;
        let violation = target_scope_violation_for_tool_call("search_files", args, &step_plan);
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
        // Routing cannot create a working directory as an incidental effect.
        // A command that needs to create it must carry that mutation itself.
        let args = r#"{"command":"npm create vite@latest whatsapp-site -- --template react","working_dir":"/tmp/my folder"}"#;
        let terminal_args = build_terminal_fallback_arguments_from_run_command(args)
            .expect("fallback args expected");
        let parsed: Value = serde_json::from_str(&terminal_args).expect("valid json");
        assert_eq!(parsed["action"], "run");
        assert_eq!(
            parsed["command"],
            "npm create vite@latest whatsapp-site -- --template react"
        );
        assert_eq!(parsed["working_dir"], "/tmp/my folder");
    }

    #[test]
    fn build_terminal_fallback_arguments_keeps_cwd_out_of_shell_program() {
        let args = r#"{"command":"npm create vite@latest whatsapp-site -- --template react","working_dir":"/tmp/david's projects"}"#;
        let terminal_args = build_terminal_fallback_arguments_from_run_command(args)
            .expect("fallback args expected");
        let parsed: Value = serde_json::from_str(&terminal_args).expect("valid json");
        assert_eq!(parsed["working_dir"], "/tmp/david's projects");
        assert!(!parsed["command"]
            .as_str()
            .unwrap_or_default()
            .contains("cd "));
    }

    #[test]
    fn target_scope_violation_allows_run_command_parent_dir_for_new_project_scaffolding() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let parent = tmp.path().join("projects");
        std::fs::create_dir_all(&parent).expect("create parent");
        let target = parent.join("new-site");
        let step_plan = StepExecutionPlan {
            step_id: "step-1".to_string(),
            description: "Scaffold a project".to_string(),
            plan_version: 1,
            primary_tool: Some("run_command".to_string()),
            expected_effect: ToolCallEffect::Mutation,
            target_scope: TargetScope {
                allowed_targets: vec![ToolTargetHint::new(
                    ToolTargetHintKind::ProjectScope,
                    target.to_string_lossy().to_string(),
                )
                .expect("scope target")],
                hard_fail_outside_scope: true,
            },
            expected_targets: Vec::new(),
            retry_policy: RetryPolicy {
                max_attempts: 1,
                allow_tool_invocation_retry: false,
            },
            approval_requirement: ApprovalRequirement::NotNeeded,
            idempotency_key: None,
        };
        let args = format!(
            r#"{{"command":"pwd","working_dir":"{}"}}"#,
            parent.to_string_lossy()
        );
        let violation = target_scope_violation_for_tool_call("run_command", &args, &step_plan);
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

    #[test]
    fn linear_intent_step_match_requires_declared_target_when_present() {
        let step = crate::agent::execution_state::LinearIntentStep {
            step_id: "plan-v1-step-2".to_string(),
            step_index: 2,
            tool: "http_request".to_string(),
            target: "https://api.example.com/tweet-2".to_string(),
            description: "Post tweet 2".to_string(),
            tool_calls_on_step: 0,
            completed: false,
            completion_evidence: None,
            last_evaluated_at: None,
        };

        assert!(!linear_intent_step_matches_tool_call(
            &step,
            "http_request",
            r#"{"url":"https://api.example.com/tweet-3","method":"POST"}"#
        ));
        assert!(linear_intent_step_matches_tool_call(
            &step,
            "http_request",
            r#"{"url":"https://api.example.com/tweet-2","method":"POST"}"#
        ));
    }

    #[test]
    fn unmatched_success_does_not_advance_linear_intent_cursor() {
        let mut execution_state = ExecutionState::new(
            BudgetTier::None,
            default_execution_budget(BudgetTier::None),
            ExecutionPersistence::Ephemeral,
        );
        execution_state.install_linear_intent_plan(
            1,
            vec![
                crate::agent::execution_state::LinearIntentStep {
                    step_id: "plan-v1-step-1".to_string(),
                    step_index: 1,
                    tool: "http_request".to_string(),
                    target: "https://api.example.com/tweet-1".to_string(),
                    description: "Post tweet 1".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
                crate::agent::execution_state::LinearIntentStep {
                    step_id: "plan-v1-step-2".to_string(),
                    step_index: 2,
                    tool: "http_request".to_string(),
                    target: "https://api.example.com/tweet-2".to_string(),
                    description: "Post tweet 2".to_string(),
                    tool_calls_on_step: 0,
                    completed: false,
                    completion_evidence: None,
                    last_evaluated_at: None,
                },
            ],
        );

        let planned_step = execution_state
            .current_linear_intent_step()
            .filter(|step| {
                linear_intent_step_matches_tool_call(
                    step,
                    "http_request",
                    r#"{"url":"https://api.example.com/tweet-2","method":"POST"}"#,
                )
            })
            .cloned();
        if planned_step.is_some() {
            execution_state.advance_linear_intent_step_after_external_success();
        }

        assert_eq!(
            execution_state
                .current_linear_intent_step()
                .expect("step should remain active")
                .step_index,
            1
        );
    }

    #[test]
    fn duplicate_successful_tool_result_requires_same_tool_args_and_result() {
        let events = vec![
            crate::events::Event::new(
                "s1",
                crate::events::EventType::ToolCall,
                serde_json::json!({
                    "tool_call_id": "call-1",
                    "name": "terminal",
                    "arguments": {"command": "wc -w file.txt"},
                    "task_id": "task-1"
                }),
            ),
            crate::events::Event::new(
                "s1",
                crate::events::EventType::ToolResult,
                serde_json::json!({
                    "tool_call_id": "call-1",
                    "name": "terminal",
                    "result": "64 file.txt\n",
                    "success": true,
                    "duration_ms": 10,
                    "task_id": "task-1"
                }),
            ),
            crate::events::Event::new(
                "s1",
                crate::events::EventType::ToolCall,
                serde_json::json!({
                    "tool_call_id": "call-2",
                    "name": "terminal",
                    "arguments": {"command": "wc -w other.txt"},
                    "task_id": "task-1"
                }),
            ),
            crate::events::Event::new(
                "s1",
                crate::events::EventType::ToolResult,
                serde_json::json!({
                    "tool_call_id": "call-2",
                    "name": "terminal",
                    "result": "12 other.txt\n",
                    "success": true,
                    "duration_ms": 10,
                    "task_id": "task-1"
                }),
            ),
        ];

        assert_eq!(
            duplicate_successful_tool_result_count(
                &events,
                "terminal",
                r#"{"command": "wc -w file.txt"}"#,
                "  64   file.txt\n",
            ),
            1
        );
        assert_eq!(
            duplicate_successful_tool_result_count(
                &events,
                "terminal",
                r#"{"command": "wc -w file.txt"}"#,
                "65 file.txt\n",
            ),
            0
        );
    }
}
