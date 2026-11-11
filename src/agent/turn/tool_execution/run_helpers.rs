//! Pure helper functions and predicates supporting `run_tool_execution_phase`.
//!
//! This module was split out of `run.rs` to keep that file under the size
//! budget. All items are moved verbatim — no logic changes. Visibility is
//! `pub(super)` so `run.rs` can call them via `super::run_helpers::*`.

use crate::agent::*;
use crate::traits::{
    RequestEvidenceRequirement, RequestVerificationTargetKind, ToolCallSemantics,
    ToolSemanticScope, ToolTargetHint, ToolTargetHintKind,
};
use crate::utils::{truncate_str, truncate_with_note};

const TOOL_COMPLETE_SUMMARY_MAX_CHARS: usize = 140;
const EXTERNAL_ACTION_ACK_MAX_CHARS: usize = 500;

pub(in crate::agent) fn fallback_tool_semantic_scope(tool_name: &str) -> Option<ToolSemanticScope> {
    if let Some(facts) = crate::agent::memory_plugin::memory_tool_facts(tool_name) {
        return facts.scope;
    }
    match tool_name {
        "web_search" | "web_fetch" | "http_request" | "browser" => {
            Some(ToolSemanticScope::ExternalRemote)
        }
        "read_file" | "search_files" | "edit_file" | "write_file" | "terminal"
        | "project_inspect" => Some(ToolSemanticScope::LocalWorkspace),
        "read_channel_history" | "search_history" => Some(ToolSemanticScope::ConversationHistory),
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

/// Return the typed write capabilities available for a mutation call when
/// the semantic manifest did not include a separate write target list.
///
/// A project path used only for read/cwd confinement must never become a write
/// grant by fallback. The promotion is allowed only for a contract that
/// explicitly expects mutation and has not installed a hard read-only fence.
/// Declared `read_roots` are a request for sandbox read authority, not a read.
/// A root the task is not authorized for must not veto the whole operation
/// when the operation's exact reads are authorized: strip it so the sandbox
/// is prepared with the authorized subset only, and let the process fail
/// with a typed sandbox denial if it genuinely needed the wider root. Write
/// roots are never attenuated; an unauthorized write stays a violation.
/// Returns the stripped roots (empty when nothing changed).
pub(in crate::agent) fn attenuate_unauthorized_read_roots(
    arguments: &mut Value,
    task: Option<&crate::traits::ToolCallAccessManifest>,
    fallback_scopes: &[String],
    execution_cwd: Option<&str>,
    paths: &crate::tools::DaemonPaths,
) -> Vec<String> {
    let Some(object) = arguments.as_object_mut() else {
        return Vec::new();
    };
    let Some(roots) = object.get("read_roots").and_then(Value::as_array).cloned() else {
        return Vec::new();
    };
    if roots.is_empty() || (task.is_none() && fallback_scopes.is_empty()) {
        return Vec::new();
    }
    let mut kept = Vec::new();
    let mut stripped = Vec::new();
    for root in roots {
        let Some(value) = root.as_str() else {
            continue;
        };
        let Some(hint) = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, value) else {
            continue;
        };
        let probe = crate::traits::ToolCallAccessManifest {
            execution_cwd: execution_cwd.map(str::to_string),
            read_targets: vec![hint],
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        if access_manifest_scope_violation(
            "terminal",
            &probe,
            task,
            fallback_scopes,
            execution_cwd,
            paths,
        )
        .is_none()
        {
            kept.push(Value::String(value.to_string()));
        } else {
            stripped.push(value.to_string());
        }
    }
    if !stripped.is_empty() {
        object.insert("read_roots".to_string(), Value::Array(kept));
    }
    stripped
}

pub(in crate::agent) fn blocked_for_untrusted_external_reference_message(
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

/// Verify a call's declared read/write capability request against the task's
/// authority before I/O. Mixed operations must satisfy both sets; execution
/// cwd selects process location and is never implicit data authority.
/// Outcome of checking a call's access manifest against task authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) enum ScopeCheck {
    /// The call must not run: protected host data, a credential store, or an
    /// undeclared write outside the task boundary.
    Rejected(String),
    /// The call declared exact targets outside the task boundary that a user
    /// may legitimately authorize (a tool's own config dir, a sibling
    /// project). The tool must obtain that authority before running.
    Escalatable {
        reason: String,
        escalation: crate::traits::ScopeEscalation,
    },
}

impl ScopeCheck {
    pub(super) fn reason(&self) -> &str {
        match self {
            ScopeCheck::Rejected(reason) | ScopeCheck::Escalatable { reason, .. } => reason,
        }
    }
}

/// Tools that can carry an out-of-scope target to a user approval prompt
/// instead of failing closed. Only the confined terminal has an approval
/// ladder that renders the exact capability being requested.
/// Whether a call names a path the guessed scope still protects: credential
/// stores and the like stay refused outside it, as they always were.
pub(in crate::agent) fn names_sensitive_path(
    manifest: &crate::traits::ToolCallAccessManifest,
    paths: &crate::tools::DaemonPaths,
) -> bool {
    manifest
        .read_targets
        .iter()
        .chain(&manifest.write_targets)
        .any(|target| paths.is_sensitive(std::path::Path::new(&target.value)))
}

pub(in crate::agent) fn tool_supports_scope_escalation(tool_name: &str) -> bool {
    tool_name == "terminal"
}

pub(super) fn access_manifest_scope_violation(
    tool_name: &str,
    call: &crate::traits::ToolCallAccessManifest,
    task: Option<&crate::traits::ToolCallAccessManifest>,
    fallback_scopes: &[String],
    execution_cwd: Option<&str>,
    paths: &crate::tools::DaemonPaths,
) -> Option<String> {
    access_manifest_scope_check(tool_name, call, task, fallback_scopes, execution_cwd, paths)
        .map(|check| check.reason().to_string())
}

/// A path as the filesystem resolves it, for scope containment: the
/// longest prefix that exists is canonicalized (symlinks and `..` resolved
/// physically), and the rest is resolved lexically (`.` dropped, `..` pops).
/// A relative path is only resolved lexically; callers anchor relative
/// targets to the execution cwd first.
pub(in crate::agent) fn normalize_scope_path(raw: &str) -> std::path::PathBuf {
    use std::path::{Component, Path, PathBuf};
    let path = Path::new(raw);
    let components: Vec<Component<'_>> = path.components().collect();
    let mut start = 0;
    let mut resolved = PathBuf::new();
    if path.is_absolute() {
        // The longest existing prefix, resolved physically.
        for end in (1..=components.len()).rev() {
            let prefix: PathBuf = components[..end].iter().collect();
            if let Ok(real) = std::fs::canonicalize(&prefix) {
                resolved = real;
                start = end;
                break;
            }
        }
    }
    for component in &components[start..] {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !resolved.pop() && !path.is_absolute() {
                    resolved.push("..");
                }
            }
            other => resolved.push(other.as_os_str()),
        }
    }
    resolved
}

pub(in crate::agent) fn access_manifest_scope_check(
    tool_name: &str,
    call: &crate::traits::ToolCallAccessManifest,
    task: Option<&crate::traits::ToolCallAccessManifest>,
    fallback_scopes: &[String],
    execution_cwd: Option<&str>,
    paths: &crate::tools::DaemonPaths,
) -> Option<ScopeCheck> {
    // Tools resolve relative targets against the execution cwd, so the
    // capability check must compare the same absolute path a directory grant
    // covers. A relative `Cargo.toml` inside an authorized workspace is not a
    // capability escape; `../outside` still resolves outside and is rejected.
    let resolve_candidate = |candidate: &ToolTargetHint| -> ToolTargetHint {
        if !matches!(
            candidate.kind,
            ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
        ) {
            return candidate.clone();
        }
        let path = std::path::Path::new(&candidate.value);
        if path.is_absolute() {
            return candidate.clone();
        }
        let Some(cwd) = execution_cwd.filter(|cwd| std::path::Path::new(cwd).is_absolute()) else {
            return candidate.clone();
        };
        let mut resolved = std::path::PathBuf::new();
        for component in std::path::Path::new(cwd).join(path).components() {
            match component {
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    resolved.pop();
                }
                other => resolved.push(other.as_os_str()),
            }
        }
        ToolTargetHint::new(candidate.kind, resolved.to_string_lossy().to_string())
            .unwrap_or_else(|| candidate.clone())
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
            (ToolTargetHintKind::Path, ToolTargetHintKind::Path) => {
                grant.value == candidate.value
                    || normalize_scope_path(&grant.value) == normalize_scope_path(&candidate.value)
            }
            (
                ToolTargetHintKind::ProjectScope,
                ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope,
            ) => {
                // Both sides normalized first: `..` and symlinks are
                // resolved, so `<root>/../x` and a link out of the root are
                // not "inside" it.
                let grant = normalize_scope_path(&grant.value);
                let candidate = normalize_scope_path(&candidate.value);
                candidate == grant || candidate.starts_with(&grant)
            }
            _ => false,
        };
    let covered_by = |candidate: &ToolTargetHint, grants: &[ToolTargetHint]| {
        grants
            .iter()
            .any(|grant| capability_grant_allows(grant, candidate))
    };
    let protected_reads = call
        .read_targets
        .iter()
        .filter(|target| {
            matches!(
                target.kind,
                ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
            ) && (crate::tools::fs_utils::is_protected_host_data_path(std::path::Path::new(
                &target.value,
            )) || crate::tools::fs_utils::is_protected_host_data_path(&normalize_scope_path(
                &target.value,
            ))) && !covered_by(target, &call.adapter_read_targets)
        })
        .map(|target| target.value.clone())
        .collect::<Vec<_>>();
    let protected_writes = call
        .write_targets
        .iter()
        .filter(|target| {
            matches!(
                target.kind,
                ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
            ) && (crate::tools::fs_utils::is_protected_host_data_path(std::path::Path::new(
                &target.value,
            )) || crate::tools::fs_utils::is_protected_host_data_path(&normalize_scope_path(
                &target.value,
            )))
        })
        .map(|target| target.value.clone())
        .collect::<Vec<_>>();
    if !protected_reads.is_empty() || !protected_writes.is_empty() {
        return Some(ScopeCheck::Rejected(format!(
            "protected host-data capability violation (protected reads: {}; protected writes: {})",
            if protected_reads.is_empty() {
                "none".to_string()
            } else {
                protected_reads.join(", ")
            },
            if protected_writes.is_empty() {
                "none".to_string()
            } else {
                protected_writes.join(", ")
            }
        )));
    }

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
        (task.read_targets.clone(), task.write_targets.clone())
    } else {
        (fallback.clone(), fallback)
    };
    let outside = |candidate: &ToolTargetHint, grants: &[ToolTargetHint]| {
        grants.is_empty() || !covered_by(candidate, grants)
    };
    // A write grant always implies read-back: a task authorized to create
    // `/tmp/x/result.txt` must be able to observe it, whether or not the
    // producer also declared explicit read grants. Read authority is the
    // union of both; writes are checked against write grants only.
    let mut read_authorities = read_grants.clone();
    for grant in &write_grants {
        if !read_authorities.contains(grant) {
            read_authorities.push(grant.clone());
        }
    }
    let invalid_reads = call
        .read_targets
        .iter()
        .filter(|candidate| {
            let resolved = resolve_candidate(candidate);
            (outside(candidate, &read_authorities) && outside(&resolved, &read_authorities))
                && outside(candidate, &call.adapter_read_targets)
        })
        .map(|target| target.value.clone())
        .collect::<Vec<_>>();
    let invalid_writes = call
        .write_targets
        .iter()
        .filter(|candidate| {
            let resolved = resolve_candidate(candidate);
            outside(candidate, &write_grants) && outside(&resolved, &write_grants)
        })
        .map(|target| target.value.clone())
        .collect::<Vec<_>>();
    if invalid_reads.is_empty() && invalid_writes.is_empty() {
        return None;
    }
    let reason = format!(
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
    );
    // An out-of-scope target the user can legitimately authorize (a deploy
    // tool's config directory, a sibling project) is escalated to the tool's
    // approval prompt rather than failed closed. Credential stores and other
    // sensitive paths never escalate: the model cannot ask its way into them.
    let sensitive = |path: &String| paths.is_sensitive(std::path::Path::new(path));
    if tool_supports_scope_escalation(tool_name)
        && !invalid_reads.iter().any(sensitive)
        && !invalid_writes.iter().any(sensitive)
    {
        return Some(ScopeCheck::Escalatable {
            reason,
            escalation: crate::traits::ScopeEscalation {
                read_paths: invalid_reads,
                write_paths: invalid_writes,
            },
        });
    }
    Some(ScopeCheck::Rejected(reason))
}

/// The reply for a turn a tool handed to the background without completion
/// notifications: the tool's own first line.
pub(in crate::agent) fn build_background_detach_ack(
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> String {
    let first_line = crate::traits::first_primary_message_line(result_text, &[])
        .unwrap_or_else(|| "The task is running in the background.".to_string());
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

pub(in crate::agent) fn summarize_completed_tool_result(result_text: &str) -> String {
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

pub(in crate::agent) fn build_external_action_completion_ack(result_text: &str) -> String {
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

pub(in crate::agent) fn should_build_external_action_ack(result_text: &str) -> bool {
    let primary = crate::traits::extract_primary_message_content(result_text, &[]);
    let lower = primary.trim_start().to_ascii_lowercase();
    !lower.starts_with("request blocked:")
        && !lower.starts_with("blocked:")
        && !lower.starts_with("[system] blocked:")
        && !lower.starts_with("error:")
        && !lower.starts_with("failed to ")
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
    if !metadata.invocation_stage.reached_dispatch() {
        // Rejection receipts describe the validation/policy boundary only.
        // Intended operation semantics are proposal data and cannot become
        // observed evidence or effects when dispatch never occurred.
        metadata.access_manifest = None;
        return;
    }
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
                ToolTargetHintKind::ResourceId,
                VerificationTargetKind::ResourceId
            )
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
        (ToolTargetHintKind::ResourceId, VerificationTargetKind::ResourceId) => {
            target_hint.value == contract_target.value
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
        // Opaque identities are adapter assertions. Merely echoing one in
        // model-supplied arguments or a shell command is not evidence that the
        // adapter observed that resource.
        VerificationTargetKind::ResourceId => None,
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
            RequestVerificationTargetKind::ResourceId => VerificationTargetKind::ResourceId,
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

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::agent) struct ReceiptPredicateEvaluation {
    pub has_predicate: bool,
    pub tool_compatible: bool,
    pub exit_compatible: bool,
    pub outcome_compatible: bool,
    pub rejection_compatible: bool,
    pub output_compatible: bool,
}

impl ReceiptPredicateEvaluation {
    fn matched(&self) -> bool {
        self.tool_compatible
            && self.exit_compatible
            && self.outcome_compatible
            && self.rejection_compatible
            && self.output_compatible
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::agent) struct EvidenceRequirementEvaluation {
    pub requirement_index: usize,
    pub matched: bool,
    pub exact_invocation: bool,
    pub read_selection_compatible: bool,
    pub capability_compatible: bool,
    pub receipt: ReceiptPredicateEvaluation,
    pub target_compatible: bool,
}

fn evaluate_receipt_predicate(
    requirement: &RequestEvidenceRequirement,
    requested_tool_name: &str,
    _result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> ReceiptPredicateEvaluation {
    let Some(receipt) = requirement.receipt.as_ref() else {
        return ReceiptPredicateEvaluation {
            has_predicate: false,
            tool_compatible: true,
            exit_compatible: true,
            outcome_compatible: true,
            // A generic subject requirement can never be closed by a
            // validation/policy rejection. Only an explicit typed receipt
            // predicate may request that disposition.
            rejection_compatible: metadata.invocation_stage.reached_dispatch(),
            output_compatible: true,
        };
    };
    let tool_compatible = receipt.tool_names.is_empty()
        || receipt.tool_names.iter().any(|name| {
            name == requested_tool_name || metadata.effective_tool_name.as_deref() == Some(name)
        });
    let exit_compatible = receipt.exit_matches(metadata.receipt_kind, metadata.exit_code);
    let outcome_compatible = metadata.outcome_status.is_some_and(|outcome| {
        receipt.outcome_matches(
            outcome,
            metadata.invocation_stage,
            metadata.contract_rejected,
            metadata.receipt_kind,
            metadata.exit_code,
        )
    });
    let rejection_compatible = receipt.rejection_matches(metadata.contract_rejected);
    let has_output = metadata
        .result_provenance
        .as_ref()
        .is_some_and(|provenance| provenance.authoritative_chars > 0);
    ReceiptPredicateEvaluation {
        has_predicate: true,
        tool_compatible,
        exit_compatible,
        outcome_compatible,
        rejection_compatible,
        output_compatible: !receipt.requires_output || has_output,
    }
}

/// Evaluate every lifecycle obligation against one typed receipt. The result
/// is both the authoritative match input and causal telemetry: consumers no
/// longer have to infer why a receipt failed from a generic validation loop.
pub(in crate::agent) fn evaluate_evidence_requirements(
    contract: &CompletionContract,
    requested_tool_name: &str,
    semantics: &ToolCallSemantics,
    raw_arguments: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> Vec<EvidenceRequirementEvaluation> {
    let read_selection_compatible = metadata
        .read_file
        .as_ref()
        .is_none_or(|read| read_receipt_covers_requested_selection(raw_arguments, read));
    contract
        .evidence_requirements
        .iter()
        .enumerate()
        .map(|(requirement_index, requirement)| {
            let exact_invocation =
                crate::agent::inquiry::requirement_is_exact_invocation(requirement);
            let capability_compatible = exact_invocation
                || semantics.evidence.iter().any(|capability| {
                    crate::agent::inquiry::capability_supports_requirement(capability, requirement)
                });
            let receipt =
                evaluate_receipt_predicate(requirement, requested_tool_name, result_text, metadata);
            let target_compatible =
                requirement_target_matches(requirement, semantics, raw_arguments, metadata);
            EvidenceRequirementEvaluation {
                requirement_index,
                matched: read_selection_compatible
                    && capability_compatible
                    && receipt.matched()
                    && target_compatible,
                exact_invocation,
                read_selection_compatible,
                capability_compatible,
                receipt,
                target_compatible,
            }
        })
        .collect()
}

/// Return the exact machine-checkable evidence obligations supported by one
/// observation receipt. Free-form subject words never decide task completion;
/// only typed capabilities, receipt fields, and structural target identity do.
pub(in crate::agent) fn matching_evidence_requirement_indices(
    contract: &CompletionContract,
    requested_tool_name: &str,
    semantics: &ToolCallSemantics,
    raw_arguments: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
) -> Vec<usize> {
    evaluate_evidence_requirements(
        contract,
        requested_tool_name,
        semantics,
        raw_arguments,
        result_text,
        metadata,
    )
    .into_iter()
    .filter_map(|evaluation| evaluation.matched.then_some(evaluation.requirement_index))
    .collect()
}

pub(in crate::agent) fn evidence_requirement_accepts_nonstandard_outcome(
    requirement: &RequestEvidenceRequirement,
    metadata: &crate::traits::ToolCallMetadata,
) -> bool {
    requirement.receipt.as_ref().is_some_and(|receipt| {
        metadata.outcome_status.is_some_and(|actual| {
            actual != crate::traits::ToolOutcomeStatus::Succeeded
                && receipt.outcome_matches(
                    actual,
                    metadata.invocation_stage,
                    metadata.contract_rejected,
                    metadata.receipt_kind,
                    metadata.exit_code,
                )
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
    use crate::traits::ToolVerificationMode;

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
            invocation_stage: crate::traits::ToolInvocationStage::Dispatched,
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
        let semantics = ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
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
                temporal_scope: EvidenceTemporalScope::Historical,
                required_content_markers: Vec::new(),
                receipt: Some(RequestReceiptPredicate {
                    tool_names: vec!["run_command".to_string()],
                    exit_codes: vec![1],
                    outcome_statuses: vec![ToolOutcomeStatus::CompletedWithNegativeResult],
                    outcome_condition: None,
                    requires_output: false,
                    contract_rejected: Some(false),
                    min_invocations: None,
                    max_invocations: None,
                }),
                target: None,
            }],
            ..CompletionContract::default()
        };
        let raw_arguments = r#"{"command":"/usr/bin/false","working_dir":"/tmp"}"#;
        let registered = ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
        let mut metadata = crate::traits::ToolCallMetadata {
            receipt_kind: crate::traits::ToolReceiptKind::Process,
            outcome_status: Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult),
            exit_code: Some(1),
            invocation_stage: crate::traits::ToolInvocationStage::Dispatched,
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
        let evaluations = evaluate_evidence_requirements(
            &mismatched,
            "run_command",
            &metadata.semantics,
            raw_arguments,
            "untrusted process text says exit=0",
            &metadata,
        );
        assert_eq!(evaluations.len(), 1);
        assert!(!evaluations[0].matched);
        assert!(!evaluations[0].receipt.exit_compatible);
        assert!(evaluations[0].receipt.tool_compatible);
        assert!(evaluations[0].capability_compatible);
        assert!(evaluations[0].target_compatible);
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
                    outcome_condition: None,
                    requires_output: true,
                    contract_rejected: Some(false),
                    min_invocations: None,
                    max_invocations: None,
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
                    outcome_condition: None,
                    requires_output: true,
                    contract_rejected: Some(false),
                    min_invocations: None,
                    max_invocations: None,
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
                &crate::traits::ToolCallMetadata {
                    invocation_stage: crate::traits::ToolInvocationStage::Dispatched,
                    ..crate::traits::ToolCallMetadata::default()
                },
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
                &crate::traits::ToolCallMetadata {
                    invocation_stage: crate::traits::ToolInvocationStage::Dispatched,
                    ..crate::traits::ToolCallMetadata::default()
                },
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
                &crate::traits::ToolCallMetadata {
                    invocation_stage: crate::traits::ToolInvocationStage::Dispatched,
                    ..crate::traits::ToolCallMetadata::default()
                },
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
        let metadata = crate::traits::ToolCallMetadata {
            invocation_stage: crate::traits::ToolInvocationStage::Dispatched,
            ..crate::traits::ToolCallMetadata::default()
        };

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
            outcome_status: Some(crate::traits::ToolOutcomeStatus::Blocked),
            contract_rejected: true,
            invocation_stage: crate::traits::ToolInvocationStage::RejectedBeforeIo,
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
    fn rejected_invocation_cannot_close_a_generic_evidence_requirement() {
        use crate::traits::{
            EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
            ToolSemanticScope,
        };
        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![RequestEvidenceRequirement {
                summary: "Observe current synthetic state".to_string(),
                acceptable_scopes: vec![ToolSemanticScope::HostLocal],
                purpose: EvidencePurpose::CurrentState,
                minimum_authority: EvidenceAuthority::Direct,
                temporal_scope: EvidenceTemporalScope::Current,
                required_content_markers: Vec::new(),
                receipt: None,
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
            outcome_status: Some(crate::traits::ToolOutcomeStatus::Blocked),
            contract_rejected: true,
            invocation_stage: crate::traits::ToolInvocationStage::RejectedBeforeIo,
            semantics: semantics.clone(),
            ..crate::traits::ToolCallMetadata::default()
        };

        assert!(matching_evidence_requirement_indices(
            &contract,
            "synthetic_observer",
            &semantics,
            "{}",
            "validation rejected",
            &metadata,
        )
        .is_empty());
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
    fn opaque_resource_id_cannot_be_verified_by_argument_text() {
        let target = VerificationTarget {
            kind: VerificationTargetKind::ResourceId,
            value: "objective:sha256:synthetic".to_string(),
        };
        assert!(!verification_target_matches_haystack(
            &target,
            r#"{"subject":"objective:sha256:synthetic"}"#
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
    fn mixed_access_manifest_checks_read_and_write_grants_independently() {
        let hint = |kind, value| ToolTargetHint::new(kind, value).expect("target");
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![
                hint(ToolTargetHintKind::ProjectScope, "/tmp"),
                hint(ToolTargetHintKind::Path, "/workspace/project/Cargo.toml"),
            ],
            write_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/synthetic-result.txt")],
            adapter_read_targets: Vec::new(),
        };
        let valid = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![
                hint(ToolTargetHintKind::ProjectScope, "/tmp"),
                hint(ToolTargetHintKind::Path, "/workspace/project/Cargo.toml"),
            ],
            write_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/synthetic-result.txt")],
            adapter_read_targets: Vec::new(),
        };
        assert!(access_manifest_scope_violation(
            "cli_agent",
            &valid,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_none());

        let invalid = crate::traits::ToolCallAccessManifest {
            write_targets: vec![hint(ToolTargetHintKind::Path, "/workspace/project")],
            ..valid
        };
        let violation = access_manifest_scope_violation(
            "cli_agent",
            &invalid,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default(),
        )
        .expect("write must be rejected");
        assert!(violation.contains("/workspace/project"));
    }

    /// Live 2026-11-11: a sub-agent confined to its project took a browser
    /// screenshot (saved in the daemon inbox) and `send_file` was refused as
    /// an "unauthorized write" of that screenshot. Sending a file reads it;
    /// the inbox is the adapter's own; a file elsewhere stays out of scope.
    #[test]
    fn a_confined_child_can_send_its_own_screenshot_but_nothing_else() {
        use crate::traits::ToolSemantics as _;
        let tmp = tempfile::tempdir().expect("tempdir");
        let inbox = tmp.path().join("files/inbox");
        let project = tmp.path().join("projects/acme-site");
        let elsewhere = tmp.path().join("private");
        for dir in [&inbox, &project, &elsewhere] {
            std::fs::create_dir_all(dir).expect("dir");
        }
        let (media_tx, _media_rx) = tokio::sync::mpsc::channel(1);
        let tool = crate::tools::SendFileTool::new(
            crate::execution::active_execution_backend(),
            media_tx,
            &[tmp.path().to_string_lossy().to_string()],
            &inbox.to_string_lossy(),
        );
        let task = crate::traits::ToolCallAccessManifest {
            read_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::ProjectScope,
                project.to_string_lossy(),
            )
            .unwrap()],
            write_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::ProjectScope,
                project.to_string_lossy(),
            )
            .unwrap()],
            ..Default::default()
        };
        let check = |path: &std::path::Path| {
            let args = serde_json::json!({ "file_path": path.to_string_lossy() }).to_string();
            let mut manifest = tool.call_access_manifest(&args);
            assert!(manifest.write_targets.is_empty(), "sending writes nothing");
            manifest.adapter_read_targets = tool
                .adapter_owned_access_manifest(&args)
                .adapter_read_targets;
            access_manifest_scope_violation(
                "send_file",
                &manifest,
                Some(&task),
                &[],
                None,
                &crate::tools::DaemonPaths::default(),
            )
        };
        assert_eq!(check(&inbox.join("5a947c87_screenshot.png")), None);
        assert_eq!(check(&project.join("homepage.png")), None);
        let refused = check(&elsewhere.join("notes.txt")).expect("outside the scope");
        assert!(refused.contains("unauthorized reads"), "{refused}");
        assert!(check(&inbox.join("../private/notes.txt")).is_some());
    }

    #[test]
    fn adapter_runtime_reads_are_not_task_data_authority() {
        let hint = |kind, value| ToolTargetHint::new(kind, value).expect("target");
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![hint(ToolTargetHintKind::ProjectScope, "/tmp")],
            write_targets: Vec::new(),
            adapter_read_targets: vec![hint(ToolTargetHintKind::ProjectScope, "/usr")],
        };
        let call = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![hint(ToolTargetHintKind::Path, "/usr/bin/env")],
            write_targets: Vec::new(),
            adapter_read_targets: vec![hint(ToolTargetHintKind::ProjectScope, "/usr")],
        };
        assert!(
            access_manifest_scope_violation(
                "terminal",
                &call,
                Some(&task),
                &[],
                None,
                &crate::tools::DaemonPaths::default()
            )
            .is_none(),
            "adapter-owned runtime reads must be independently grantable"
        );

        let undeclared = crate::traits::ToolCallAccessManifest {
            read_targets: vec![hint(ToolTargetHintKind::Path, "/etc/hosts")],
            ..call
        };
        assert!(access_manifest_scope_violation(
            "terminal",
            &undeclared,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_some());
    }

    #[test]
    fn semantic_task_grant_cannot_widen_into_protected_host_data() {
        let hint = |kind, value| ToolTargetHint::new(kind, value).expect("target");
        let task = crate::traits::ToolCallAccessManifest {
            read_targets: vec![hint(ToolTargetHintKind::Path, "/etc/hosts")],
            ..crate::traits::ToolCallAccessManifest::default()
        };
        let call = task.clone();
        let violation = access_manifest_scope_violation(
            "terminal",
            &call,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default(),
        )
        .expect("host data remains protected independently of task assessment");
        assert!(violation.contains("protected host-data capability violation"));

        let adapter_only = crate::traits::ToolCallAccessManifest {
            adapter_read_targets: vec![hint(ToolTargetHintKind::ProjectScope, "/usr")],
            ..crate::traits::ToolCallAccessManifest::default()
        };
        assert!(
            access_manifest_scope_violation(
                "terminal",
                &adapter_only,
                None,
                &[],
                None,
                &crate::tools::DaemonPaths::default()
            )
            .is_none(),
            "adapter runtime capability is not task data"
        );
    }

    #[test]
    fn directory_write_grant_covers_future_descendants_for_write_and_read_back() {
        let hint = |kind, value| ToolTargetHint::new(kind, value).expect("target");
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/input.txt")],
            write_targets: vec![hint(ToolTargetHintKind::ProjectScope, "/tmp/output-root")],
            adapter_read_targets: Vec::new(),
        };
        let call = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/input.txt")],
            write_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/output-root/.keep")],
            adapter_read_targets: Vec::new(),
        };
        assert!(access_manifest_scope_violation(
            "write_file",
            &call,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_none());

        let read_escape = crate::traits::ToolCallAccessManifest {
            read_targets: vec![hint(ToolTargetHintKind::Path, "/tmp/elsewhere/secret")],
            ..call
        };
        assert!(access_manifest_scope_violation(
            "write_file",
            &read_escape,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_some());
    }

    #[test]
    fn execution_cwd_is_neither_an_implicit_read_nor_write_grant() {
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: Vec::new(),
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        let call = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::ProjectScope, "/tmp").expect("cwd"),
            ],
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        assert!(access_manifest_scope_violation(
            "terminal",
            &call,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_some());
    }

    #[test]
    fn execution_cwd_is_strategy_and_does_not_restrict_data_capabilities() {
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp/expected".to_string()),
            read_targets: Vec::new(),
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        let call = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp/different".to_string()),
            read_targets: Vec::new(),
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        assert!(access_manifest_scope_violation(
            "terminal",
            &call,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_none());
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
            adapter_read_targets: Vec::new(),
        };

        assert!(access_manifest_scope_violation(
            "synthetic",
            &call,
            None,
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_none());
    }

    #[test]
    fn relative_targets_resolve_against_execution_cwd_before_capability_check() {
        let workspace = "/tmp/synthetic-workspace";
        let call = |value: &str| crate::traits::ToolCallAccessManifest {
            execution_cwd: Some(workspace.to_string()),
            read_targets: vec![ToolTargetHint::new(ToolTargetHintKind::Path, value).expect("read")],
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        let scopes = vec![workspace.to_string()];
        // Inside the authorized workspace: `Cargo.toml` and `./src/../Cargo.toml`.
        assert!(access_manifest_scope_violation(
            "read_file",
            &call("Cargo.toml"),
            None,
            &scopes,
            Some(workspace),
            &crate::tools::DaemonPaths::default()
        )
        .is_none());
        assert!(access_manifest_scope_violation(
            "read_file",
            &call("./src/../Cargo.toml"),
            None,
            &scopes,
            Some(workspace),
            &crate::tools::DaemonPaths::default()
        )
        .is_none());
        // Escapes still resolve outside and are rejected.
        assert!(access_manifest_scope_violation(
            "read_file",
            &call("../outside.txt"),
            None,
            &scopes,
            Some(workspace),
            &crate::tools::DaemonPaths::default()
        )
        .is_some());
        // Without a cwd a relative target cannot be proven inside any grant.
        assert!(access_manifest_scope_violation(
            "read_file",
            &call("Cargo.toml"),
            None,
            &scopes,
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_some());
    }

    #[test]
    fn directory_write_grant_authorizes_reading_the_same_directory() {
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: None,
            read_targets: Vec::new(),
            write_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::ProjectScope,
                "/tmp/synthetic-disposable",
            )
            .expect("write")],
            adapter_read_targets: Vec::new(),
        };
        let read_back = crate::traits::ToolCallAccessManifest {
            execution_cwd: None,
            read_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::Path,
                "/tmp/synthetic-disposable/result.txt",
            )
            .expect("read")],
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        assert!(access_manifest_scope_violation(
            "terminal",
            &read_back,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_none());
        // Explicit read grants do not remove the write-implies-read-back grant.
        let explicit_reads = crate::traits::ToolCallAccessManifest {
            read_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::Path,
                "/tmp/synthetic-input.txt",
            )
            .expect("read")],
            ..task.clone()
        };
        assert!(access_manifest_scope_violation(
            "terminal",
            &read_back,
            Some(&explicit_reads),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_none());
        // The implication is one-way: a read grant never authorizes a write.
        let read_only_task = crate::traits::ToolCallAccessManifest {
            execution_cwd: None,
            read_targets: task.write_targets.clone(),
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        let write_attempt = crate::traits::ToolCallAccessManifest {
            execution_cwd: None,
            read_targets: Vec::new(),
            write_targets: read_back.read_targets.clone(),
            adapter_read_targets: Vec::new(),
        };
        assert!(access_manifest_scope_violation(
            "terminal",
            &write_attempt,
            Some(&read_only_task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_some());
    }

    #[test]
    fn explicit_read_grants_still_cover_paths_the_task_may_write() {
        use crate::agent::tool_execution::{access_manifest_scope_check, ScopeCheck};
        let hint = |kind, value: &str| ToolTargetHint::new(kind, value).expect("target");
        let task = crate::traits::ToolCallAccessManifest {
            read_targets: vec![hint(
                ToolTargetHintKind::Path,
                "/tmp/aidaemon-test/data/prices.csv",
            )],
            write_targets: vec![hint(
                ToolTargetHintKind::Path,
                "/tmp/aidaemon-test/data/avg.txt",
            )],
            ..Default::default()
        };
        let call = |path: &str| crate::traits::ToolCallAccessManifest {
            read_targets: vec![hint(ToolTargetHintKind::Path, path)],
            ..Default::default()
        };
        assert!(access_manifest_scope_check(
            "read_file",
            &call("/tmp/aidaemon-test/data/avg.txt"),
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_none());
        assert!(matches!(
            access_manifest_scope_check(
                "read_file",
                &call("/tmp/aidaemon-test/data/other.txt"),
                Some(&task),
                &[],
                None,
                &crate::tools::DaemonPaths::default()
            ),
            Some(ScopeCheck::Rejected(_))
        ));
    }

    #[test]
    fn unauthorized_declared_read_roots_are_stripped_not_fatal() {
        let task = crate::traits::ToolCallAccessManifest {
            execution_cwd: Some("/tmp".to_string()),
            read_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::Path,
                "/tmp/synthetic-out/result.txt",
            )
            .expect("read")],
            write_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::ProjectScope,
                "/tmp/synthetic-out",
            )
            .expect("write")],
            adapter_read_targets: Vec::new(),
        };
        let mut arguments = serde_json::json!({
            "action": "run",
            "command": "cat /tmp/synthetic-out/result.txt",
            "read_paths": ["/tmp/synthetic-out/result.txt"],
            "read_roots": ["/tmp", "/tmp/synthetic-out"],
            "write_roots": ["/tmp/synthetic-out"],
        });
        let stripped = attenuate_unauthorized_read_roots(
            &mut arguments,
            Some(&task),
            &[],
            Some("/tmp"),
            &crate::tools::DaemonPaths::default(),
        );
        // The write root is itself readable (write implies read-back), so only
        // the broader `/tmp` root is stripped.
        assert_eq!(stripped, vec!["/tmp".to_string()]);
        assert_eq!(
            arguments["read_roots"],
            serde_json::json!(["/tmp/synthetic-out"])
        );
        // Exact reads and write roots are untouched: the capability check
        // still evaluates them.
        assert_eq!(
            arguments["read_paths"],
            serde_json::json!(["/tmp/synthetic-out/result.txt"])
        );
        assert_eq!(
            arguments["write_roots"],
            serde_json::json!(["/tmp/synthetic-out"])
        );

        // An authorized root is kept.
        let scoped = crate::traits::ToolCallAccessManifest {
            execution_cwd: None,
            read_targets: vec![ToolTargetHint::new(
                ToolTargetHintKind::ProjectScope,
                "/tmp/synthetic-out",
            )
            .expect("read")],
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        };
        let mut arguments = serde_json::json!({ "read_roots": ["/tmp/synthetic-out", "/etc"] });
        let stripped = attenuate_unauthorized_read_roots(
            &mut arguments,
            Some(&scoped),
            &[],
            None,
            &crate::tools::DaemonPaths::default(),
        );
        assert_eq!(stripped, vec!["/etc".to_string()]);
        assert_eq!(
            arguments["read_roots"],
            serde_json::json!(["/tmp/synthetic-out"])
        );

        // Without any authority boundary nothing is attenuated.
        let mut arguments = serde_json::json!({ "read_roots": ["/tmp"] });
        assert!(attenuate_unauthorized_read_roots(
            &mut arguments,
            None,
            &[],
            None,
            &crate::tools::DaemonPaths::default()
        )
        .is_empty());
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
            "Command still running after 30s. Moved to background (pid=123).\n\nCompletion notifications are enabled. The user will be notified when this process finishes.\n\n[SYSTEM] ...",
            &with_notify,
        );
        assert!(ack.contains("Moved to background (pid=123)"));
        assert!(ack.contains("final result will be sent automatically"));

        // Without notifications — should NOT promise automatic delivery.
        let ack_no_notify = build_background_detach_ack(
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
            "Command still running after 30s. Moved to background (pid=123).",
            &with_notify,
        );
        assert!(ack.contains("final result will be sent automatically"));

        let ack_no_notify = build_background_detach_ack(
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

#[cfg(test)]
mod scope_escalation_tests {
    use super::*;
    use crate::traits::{ToolCallAccessManifest, ToolTargetHint, ToolTargetHintKind};

    fn task_scope(root: &str) -> ToolCallAccessManifest {
        let hint = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, root).unwrap();
        ToolCallAccessManifest {
            execution_cwd: Some(root.to_string()),
            read_targets: vec![hint.clone()],
            write_targets: vec![hint],
            adapter_read_targets: Vec::new(),
        }
    }

    fn call_reading(root: &str, extra: &str) -> ToolCallAccessManifest {
        ToolCallAccessManifest {
            execution_cwd: Some(root.to_string()),
            read_targets: vec![
                ToolTargetHint::new(ToolTargetHintKind::Path, root).unwrap(),
                ToolTargetHint::new(ToolTargetHintKind::Path, extra).unwrap(),
            ],
            write_targets: Vec::new(),
            adapter_read_targets: Vec::new(),
        }
    }

    #[test]
    fn terminal_out_of_scope_config_dir_escalates_instead_of_rejecting() {
        let root = "/Users/alice/projects/site";
        let task = task_scope(root);
        let call = call_reading(root, "/Users/alice/Library/Preferences/.wrangler");
        match access_manifest_scope_check(
            "terminal",
            &call,
            Some(&task),
            &[],
            Some(root),
            &crate::tools::DaemonPaths::default(),
        ) {
            Some(ScopeCheck::Escalatable { escalation, reason }) => {
                assert_eq!(
                    escalation.read_paths,
                    vec!["/Users/alice/Library/Preferences/.wrangler".to_string()]
                );
                assert!(escalation.write_paths.is_empty());
                assert!(reason.contains("unauthorized reads"));
            }
            other => panic!("expected escalation, got {other:?}"),
        }
        // The string-returning wrapper still reports the violation for
        // callers that only need the reason.
        assert!(access_manifest_scope_violation(
            "terminal",
            &call,
            Some(&task),
            &[],
            Some(root),
            &crate::tools::DaemonPaths::default()
        )
        .is_some());
    }

    /// A fresh project under `/tmp/aidaemon-test/<uuid>`, with a sibling.
    fn temp_project() -> (std::path::PathBuf, String) {
        let base =
            std::path::Path::new("/tmp/aidaemon-test").join(uuid::Uuid::new_v4().to_string());
        let project = base.join("project");
        std::fs::create_dir_all(project.join("a")).unwrap();
        std::fs::create_dir_all(base.join("sibling")).unwrap();
        std::fs::write(base.join("sibling").join("x"), "synthetic").unwrap();
        let root = project.to_string_lossy().to_string();
        (base, root)
    }

    /// `..` is resolved before the containment test: a path that walks out
    /// of the project is out of scope however it is spelled.
    #[test]
    fn dotdot_escape_from_the_project_is_out_of_scope() {
        let (base, root) = temp_project();
        let task = task_scope(&root);
        for escape in [format!("{root}/../sibling/x"), "../sibling/x".to_string()] {
            let call = call_reading(&root, &escape);
            assert!(
                matches!(
                    access_manifest_scope_check(
                        "terminal",
                        &call,
                        Some(&task),
                        &[],
                        Some(&root),
                        &crate::tools::DaemonPaths::default()
                    ),
                    Some(ScopeCheck::Escalatable { .. })
                ),
                "{escape} must be out of scope"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn inner_dotdot_stays_in_scope() {
        let (base, root) = temp_project();
        let task = task_scope(&root);
        let call = call_reading(&root, &format!("{root}/a/../b"));
        assert_eq!(
            access_manifest_scope_check(
                "terminal",
                &call,
                Some(&task),
                &[],
                Some(&root),
                &crate::tools::DaemonPaths::default()
            ),
            None
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A symlink inside the project that points outside it is judged by its
    /// target, not by where it sits.
    #[cfg(unix)]
    #[test]
    fn symlink_out_of_the_project_is_out_of_scope() {
        let (base, root) = temp_project();
        let task = task_scope(&root);
        let link = std::path::Path::new(&root).join("link");
        std::os::unix::fs::symlink(base.join("sibling"), &link).unwrap();
        for read in [format!("{root}/link/x"), format!("{root}/link/../x")] {
            let call = call_reading(&root, &read);
            assert!(
                matches!(
                    access_manifest_scope_check(
                        "terminal",
                        &call,
                        Some(&task),
                        &[],
                        Some(&root),
                        &crate::tools::DaemonPaths::default()
                    ),
                    Some(ScopeCheck::Escalatable { .. })
                ),
                "{read} must be out of scope"
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn credential_stores_never_escalate() {
        let root = "/Users/alice/projects/site";
        let task = task_scope(root);
        for secret in [
            "/Users/alice/.ssh",
            "/Users/alice/.aws/credentials",
            "/Users/alice/.npmrc",
        ] {
            let call = call_reading(root, secret);
            assert!(
                matches!(
                    access_manifest_scope_check(
                        "terminal",
                        &call,
                        Some(&task),
                        &[],
                        Some(root),
                        &crate::tools::DaemonPaths::default()
                    ),
                    Some(ScopeCheck::Rejected(_))
                ),
                "{secret} must be rejected, not escalated"
            );
        }
    }

    #[test]
    fn only_the_terminal_carries_escalations() {
        let root = "/Users/alice/projects/site";
        let task = task_scope(root);
        let call = call_reading(root, "/Users/alice/Library/Preferences/.wrangler");
        assert!(matches!(
            access_manifest_scope_check(
                "read_file",
                &call,
                Some(&task),
                &[],
                Some(root),
                &crate::tools::DaemonPaths::default()
            ),
            Some(ScopeCheck::Rejected(_))
        ));
    }

    #[test]
    fn protected_host_data_is_rejected_before_escalation() {
        let root = "/Users/alice/projects/site";
        let task = task_scope(root);
        let call = call_reading(root, "/private/etc/hosts");
        assert!(matches!(
            access_manifest_scope_check(
                "terminal",
                &call,
                Some(&task),
                &[],
                Some(root),
                &crate::tools::DaemonPaths::default()
            ),
            Some(ScopeCheck::Rejected(_))
        ));
    }
}
