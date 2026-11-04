//! Tool authority, resolved once per turn from explicit facts: who is asking
//! (role and workspace grant), the owner's configured level, the session mode
//! the owner chose, and a scheduled goal's bound workspace. The request
//! assessment never widens or narrows it.
//!
//! The result is the only authority the tool gate reads: the kernel's
//! `AuthorityHook` enforces its level and tool set per call, and the daemon's
//! scope check confines filesystem access to its `task_access`.

use crate::config::OwnerLevel;
use crate::traits::{ToolCallAccessManifest, ToolTargetHint, ToolTargetHintKind};
use crate::types::{UserRole, WorkspaceAccessLevel, WorkspaceGrant};

/// A session mode the owner chose with an explicit command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionMode {
    ReadOnly,
    Full,
    Unrestricted,
}

/// How long a `/readonly` or `/mode unrestricted` lasts.
const SESSION_MODE_TTL_HOURS: i64 = 4;

fn session_mode_key(session_id: &str) -> String {
    format!("authority_mode:{session_id}")
}

/// Persist the owner's session mode and return when it expires. `Full` is the
/// default, so it clears the setting (stored as an empty value, which reads as
/// no mode) instead of recording an expiry.
pub(crate) async fn set_session_mode(
    state: &dyn crate::traits::StateStore,
    session_id: &str,
    mode: SessionMode,
    now: chrono::DateTime<chrono::Utc>,
) -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
    let expires_at = now + chrono::Duration::hours(SESSION_MODE_TTL_HOURS);
    let key = session_mode_key(session_id);
    let value = match mode {
        SessionMode::Full => String::new(),
        SessionMode::ReadOnly | SessionMode::Unrestricted => serde_json::json!({
            "mode": if mode == SessionMode::ReadOnly { "read_only" } else { "unrestricted" },
            "expires_at": expires_at.to_rfc3339(),
        })
        .to_string(),
    };
    state.set_setting(&key, &value).await?;
    Ok(expires_at)
}

/// The session mode in force at `now`, with its expiry. Expired, cleared or
/// unreadable entries read as no mode (the configured default applies).
pub(crate) async fn session_mode_with_expiry(
    state: &dyn crate::traits::StateStore,
    session_id: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<(SessionMode, chrono::DateTime<chrono::Utc>)> {
    let raw = state
        .get_setting(&session_mode_key(session_id))
        .await
        .ok()
        .flatten()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let expires_at = chrono::DateTime::parse_from_rfc3339(value["expires_at"].as_str()?)
        .ok()?
        .with_timezone(&chrono::Utc);
    if expires_at <= now {
        return None;
    }
    let mode = match value["mode"].as_str()? {
        "read_only" => SessionMode::ReadOnly,
        "unrestricted" => SessionMode::Unrestricted,
        "full" => SessionMode::Full,
        _ => return None,
    };
    Some((mode, expires_at))
}

pub(crate) async fn session_mode(
    state: &dyn crate::traits::StateStore,
    session_id: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<SessionMode> {
    session_mode_with_expiry(state, session_id, now)
        .await
        .map(|(mode, _)| mode)
}

/// The explicit facts authority is resolved from.
pub(in crate::agent) struct AuthorityFacts<'a> {
    pub user_role: UserRole,
    pub owner_default: OwnerLevel,
    pub session_mode: Option<SessionMode>,
    pub workspace_grant: Option<&'a WorkspaceGrant>,
    pub bound_workspace: Option<&'a str>,
}

/// What this turn may do.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::agent) struct ResolvedAuthority {
    /// Level and tool set, enforced per call by the kernel hook.
    pub authority: aidaemon_core::Authority,
    /// The owner's level (`None` for guests and the public).
    #[allow(dead_code)] // Read by the untrusted-content rule (Phase 2).
    pub owner_level: Option<OwnerLevel>,
    /// Filesystem confinement the scope check applies (`None`: none beyond
    /// the turn's ordinary scope lock and a guest's workspace grant).
    pub task_access: Option<ToolCallAccessManifest>,
}

pub(in crate::agent) fn resolve_authority(facts: &AuthorityFacts<'_>) -> ResolvedAuthority {
    use aidaemon_core::Authority;
    match facts.user_role {
        UserRole::Owner => {
            let owner_level = Some(match facts.session_mode {
                Some(SessionMode::Full) => OwnerLevel::Full,
                Some(SessionMode::Unrestricted) => OwnerLevel::Unrestricted,
                Some(SessionMode::ReadOnly) | None => facts.owner_default,
            });
            if facts.session_mode == Some(SessionMode::ReadOnly) {
                return ResolvedAuthority {
                    authority: Authority::read_only(),
                    owner_level,
                    task_access: None,
                };
            }
            ResolvedAuthority {
                authority: Authority::full(),
                owner_level,
                task_access: facts.bound_workspace.and_then(bound_workspace_access),
            }
        }
        // A guest's grant keeps its own scoping in `tool_exec.rs`; without
        // one, every tool is refused there too.
        UserRole::Guest => ResolvedAuthority {
            authority: match facts.workspace_grant.map(|grant| grant.access) {
                Some(WorkspaceAccessLevel::Edit) => Authority::full(),
                Some(WorkspaceAccessLevel::Read) | None => Authority::read_only(),
            },
            owner_level: None,
            task_access: None,
        },
        UserRole::Public => ResolvedAuthority {
            authority: Authority::no_tools(),
            owner_level: None,
            task_access: None,
        },
    }
}

/// A scheduled goal bound to a workspace writes only inside it and runs
/// there; reads stay free (a grant on the filesystem root the workspace
/// lives under), as do network, commands and credentials.
fn bound_workspace_access(workspace: &str) -> Option<ToolCallAccessManifest> {
    let workspace = workspace.trim();
    if workspace.is_empty() {
        return None;
    }
    let write = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, workspace.to_string())?;
    let root = std::path::Path::new(workspace)
        .ancestors()
        .last()
        .map(|root| root.to_string_lossy().into_owned())
        .filter(|root| !root.is_empty())?;
    let read = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, root)?;
    Some(ToolCallAccessManifest {
        execution_cwd: Some(workspace.to_string()),
        read_targets: vec![read],
        write_targets: vec![write],
        adapter_read_targets: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::Authority;

    async fn temp_store(path: &std::path::Path) -> crate::state::SqliteStateStore {
        let embedding = std::sync::Arc::new(
            crate::memory::embeddings::EmbeddingService::new().expect("embedding service"),
        );
        crate::state::SqliteStateStore::new(path.to_str().unwrap(), 100, None, embedding)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn session_mode_expires() {
        let dir = tempfile::tempdir().unwrap();
        let store = temp_store(&dir.path().join("a.db")).await;
        let t = chrono::Utc::now();
        let expires = set_session_mode(&store, "synthetic-session", SessionMode::ReadOnly, t)
            .await
            .unwrap();
        assert_eq!(expires, t + chrono::Duration::hours(4));
        let before = t + chrono::Duration::minutes(3 * 60 + 59);
        let after = t + chrono::Duration::minutes(4 * 60 + 1);
        assert_eq!(
            session_mode(&store, "synthetic-session", before).await,
            Some(SessionMode::ReadOnly)
        );
        assert_eq!(session_mode(&store, "synthetic-session", after).await, None);
        assert_eq!(session_mode(&store, "other-session", t).await, None);
    }

    #[tokio::test]
    async fn session_mode_full_clears_the_mode() {
        let dir = tempfile::tempdir().unwrap();
        let store = temp_store(&dir.path().join("a.db")).await;
        let t = chrono::Utc::now();
        set_session_mode(&store, "s", SessionMode::Unrestricted, t)
            .await
            .unwrap();
        assert_eq!(
            session_mode(&store, "s", t).await,
            Some(SessionMode::Unrestricted)
        );
        set_session_mode(&store, "s", SessionMode::Full, t)
            .await
            .unwrap();
        assert_eq!(session_mode(&store, "s", t).await, None);
    }

    #[tokio::test]
    async fn session_mode_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("restart.db");
        let t = chrono::Utc::now();
        {
            let store = temp_store(&path).await;
            set_session_mode(&store, "synthetic-session", SessionMode::ReadOnly, t)
                .await
                .unwrap();
        }
        let reopened = temp_store(&path).await;
        assert_eq!(
            session_mode(&reopened, "synthetic-session", t).await,
            Some(SessionMode::ReadOnly)
        );
    }

    fn facts(role: UserRole) -> AuthorityFacts<'static> {
        AuthorityFacts {
            user_role: role,
            owner_default: OwnerLevel::Full,
            session_mode: None,
            workspace_grant: None,
            bound_workspace: None,
        }
    }

    fn grant(access: WorkspaceAccessLevel) -> WorkspaceGrant {
        WorkspaceGrant {
            platform: "telegram".to_string(),
            workspace_id: "synthetic-workspace".to_string(),
            channel_id: "synthetic-channel".to_string(),
            user_id: "telegram:synthetic-user-1".to_string(),
            project_root: "/tmp/aidaemon-test/grant".to_string(),
            access,
            expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
        }
    }

    #[test]
    fn resolve_authority_owner_read_only_mode_is_read_only() {
        let resolved = resolve_authority(&AuthorityFacts {
            session_mode: Some(SessionMode::ReadOnly),
            owner_default: OwnerLevel::Unrestricted,
            bound_workspace: Some("/tmp/aidaemon-test/ws"),
            ..facts(UserRole::Owner)
        });
        assert_eq!(resolved.authority, Authority::read_only());
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Unrestricted));
        assert!(resolved.task_access.is_none());
    }

    #[test]
    fn resolve_authority_owner_without_mode_is_full_at_the_configured_level() {
        let resolved = resolve_authority(&facts(UserRole::Owner));
        assert_eq!(resolved.authority, Authority::full());
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Full));
        assert!(resolved.task_access.is_none());

        let resolved = resolve_authority(&AuthorityFacts {
            owner_default: OwnerLevel::Unrestricted,
            ..facts(UserRole::Owner)
        });
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Unrestricted));
    }

    #[test]
    fn resolve_authority_owner_mode_sets_the_level() {
        for (mode, level) in [
            (SessionMode::Full, OwnerLevel::Full),
            (SessionMode::Unrestricted, OwnerLevel::Unrestricted),
        ] {
            let resolved = resolve_authority(&AuthorityFacts {
                session_mode: Some(mode),
                owner_default: if level == OwnerLevel::Full {
                    OwnerLevel::Unrestricted
                } else {
                    OwnerLevel::Full
                },
                ..facts(UserRole::Owner)
            });
            assert_eq!(resolved.authority, Authority::full());
            assert_eq!(resolved.owner_level, Some(level));
            assert!(resolved.task_access.is_none());
        }
    }

    #[test]
    fn resolve_authority_owner_bound_workspace_confines_writes_not_reads() {
        let resolved = resolve_authority(&AuthorityFacts {
            bound_workspace: Some("/tmp/aidaemon-test/ws"),
            ..facts(UserRole::Owner)
        });
        assert_eq!(resolved.authority, Authority::full());
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Full));
        let access = resolved.task_access.expect("bound workspace manifest");
        assert_eq!(
            access.execution_cwd.as_deref(),
            Some("/tmp/aidaemon-test/ws")
        );
        assert_eq!(
            access.write_targets,
            [ToolTargetHint::new(
                ToolTargetHintKind::ProjectScope,
                "/tmp/aidaemon-test/ws".to_string()
            )
            .unwrap()]
        );
        // Reads anywhere: the scope check admits them; writes outside do not.
        let call = |reads: &[&str], writes: &[&str]| ToolCallAccessManifest {
            read_targets: reads
                .iter()
                .map(|p| ToolTargetHint::new(ToolTargetHintKind::Path, p.to_string()).unwrap())
                .collect(),
            write_targets: writes
                .iter()
                .map(|p| ToolTargetHint::new(ToolTargetHintKind::Path, p.to_string()).unwrap())
                .collect(),
            ..ToolCallAccessManifest::default()
        };
        let check = |manifest: &ToolCallAccessManifest| {
            crate::agent::tool_execution::access_manifest_scope_check(
                "write_file",
                manifest,
                Some(&access),
                &[],
                access.execution_cwd.as_deref(),
            )
        };
        assert!(check(&call(&["/tmp/aidaemon-test/elsewhere/notes.txt"], &[])).is_none());
        assert!(check(&call(&[], &["/tmp/aidaemon-test/ws/a.txt"])).is_none());
        assert!(check(&call(&[], &["a.txt"])).is_none());
        assert!(matches!(
            check(&call(&[], &["/tmp/aidaemon-test/outside.txt"])),
            Some(crate::agent::tool_execution::ScopeCheck::Rejected(_))
        ));
    }

    #[test]
    fn resolve_authority_guest_read_grant_is_read_only() {
        let grant = grant(WorkspaceAccessLevel::Read);
        let resolved = resolve_authority(&AuthorityFacts {
            workspace_grant: Some(&grant),
            ..facts(UserRole::Guest)
        });
        assert_eq!(resolved.authority, Authority::read_only());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }

    #[test]
    fn resolve_authority_guest_edit_grant_is_full() {
        let grant = grant(WorkspaceAccessLevel::Edit);
        let resolved = resolve_authority(&AuthorityFacts {
            workspace_grant: Some(&grant),
            ..facts(UserRole::Guest)
        });
        assert_eq!(resolved.authority, Authority::full());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }

    #[test]
    fn resolve_authority_guest_without_grant_is_read_only() {
        let resolved = resolve_authority(&AuthorityFacts {
            session_mode: Some(SessionMode::Unrestricted),
            bound_workspace: Some("/tmp/aidaemon-test/ws"),
            ..facts(UserRole::Guest)
        });
        assert_eq!(resolved.authority, Authority::read_only());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }

    /// The integration harness cannot show this: a guest without a grant is
    /// offered no tool, so the kernel refuses an unknown tool before the
    /// dispatcher. Here the dispatcher itself still refuses it.
    #[tokio::test]
    async fn guest_without_grant_still_gets_no_active_workspace_grant() {
        let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
            crate::testing::MockProvider::new(),
            vec![std::sync::Arc::new(crate::tools::WriteFileTool)],
            None,
        )
        .await
        .unwrap();
        let ctx = crate::agent::tool_exec::ToolExecCtx {
            session_id: "guest-session",
            task_id: None,
            status_tx: None,
            channel_visibility: crate::types::ChannelVisibility::PrivateGroup,
            channel_id: Some("telegram:synthetic-group"),
            project_scope: None,
            trusted: false,
            user_role: UserRole::Guest,
            workspace_grant: None,
            mandate_authority: None,
            tool_call_id: None,
            mutation_forbidden: false,
            scope_escalation: None,
        };
        let error = harness
            .agent
            .execute_tool_with_watchdog_outcome(
                "write_file",
                r#"{"path":"/tmp/aidaemon-test/guest.txt","content":"x"}"#,
                &ctx,
            )
            .await
            .expect_err("a guest without a grant is refused");
        assert!(
            error.to_string().contains("no active workspace grant"),
            "{error}"
        );
    }

    #[test]
    fn resolve_authority_public_has_no_tools() {
        let resolved = resolve_authority(&AuthorityFacts {
            session_mode: Some(SessionMode::Full),
            ..facts(UserRole::Public)
        });
        assert_eq!(resolved.authority, Authority::no_tools());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }
}
