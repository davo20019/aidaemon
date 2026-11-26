//! A tool call's declared targets checked against the task's filesystem
//! authority: protected host data is refused, targets outside the task's
//! grants are refused or escalated to the owner.

use aidaemon_events::traits::{ToolTargetHint, ToolTargetHintKind};

/// Verify a call's declared read/write capability request against the task's
/// authority before I/O. Mixed operations must satisfy both sets; execution
/// cwd selects process location and is never implicit data authority.
/// Outcome of checking a call's access manifest against task authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeCheck {
    /// The call must not run: protected host data, a credential store, or an
    /// undeclared write outside the task boundary.
    Rejected(String),
    /// The call declared exact targets outside the task boundary that a user
    /// may legitimately authorize (a tool's own config dir, a sibling
    /// project). The tool must obtain that authority before running.
    Escalatable {
        reason: String,
        escalation: aidaemon_events::traits::ScopeEscalation,
    },
}

impl ScopeCheck {
    pub fn reason(&self) -> &str {
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
pub fn names_sensitive_path(
    manifest: &aidaemon_events::traits::ToolCallAccessManifest,
    paths: &aidaemon_tools::path_class::DaemonPaths,
) -> bool {
    manifest
        .read_targets
        .iter()
        .chain(&manifest.write_targets)
        .any(|target| paths.is_sensitive(std::path::Path::new(&target.value)))
}

pub fn tool_supports_scope_escalation(tool_name: &str) -> bool {
    tool_name == "terminal"
}

pub fn access_manifest_scope_violation(
    tool_name: &str,
    call: &aidaemon_events::traits::ToolCallAccessManifest,
    task: Option<&aidaemon_events::traits::ToolCallAccessManifest>,
    fallback_scopes: &[String],
    execution_cwd: Option<&str>,
    paths: &aidaemon_tools::path_class::DaemonPaths,
) -> Option<String> {
    access_manifest_scope_check(tool_name, call, task, fallback_scopes, execution_cwd, paths)
        .map(|check| check.reason().to_string())
}

/// A path as the filesystem resolves it, for scope containment: the
/// longest prefix that exists is canonicalized (symlinks and `..` resolved
/// physically), and the rest is resolved lexically (`.` dropped, `..` pops).
/// A relative path is only resolved lexically; callers anchor relative
/// targets to the execution cwd first.
pub fn normalize_scope_path(raw: &str) -> std::path::PathBuf {
    use std::path::{Component, Path, PathBuf};
    let path = Path::new(raw);
    let components: Vec<Component<'_>> = path.components().collect();
    let mut start = 0;
    let mut resolved = PathBuf::new();
    if path.is_absolute() {
        // The longest existing prefix, resolved physically.
        for end in (1..=components.len()).rev() {
            let prefix: PathBuf = components[..end].iter().collect();
            if let Ok(real) = dunce::canonicalize(&prefix) {
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

pub fn access_manifest_scope_check(
    tool_name: &str,
    call: &aidaemon_events::traits::ToolCallAccessManifest,
    task: Option<&aidaemon_events::traits::ToolCallAccessManifest>,
    fallback_scopes: &[String],
    execution_cwd: Option<&str>,
    paths: &aidaemon_tools::path_class::DaemonPaths,
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
            ) && (aidaemon_tools::fs_utils::is_protected_host_data_path(std::path::Path::new(
                &target.value,
            )) || aidaemon_tools::fs_utils::is_protected_host_data_path(
                &normalize_scope_path(&target.value),
            )) && !covered_by(target, &call.adapter_read_targets)
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
            ) && (aidaemon_tools::fs_utils::is_protected_host_data_path(std::path::Path::new(
                &target.value,
            )) || aidaemon_tools::fs_utils::is_protected_host_data_path(
                &normalize_scope_path(&target.value),
            ))
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
            escalation: aidaemon_events::traits::ScopeEscalation {
                read_paths: invalid_reads,
                write_paths: invalid_writes,
            },
        });
    }
    Some(ScopeCheck::Rejected(reason))
}
