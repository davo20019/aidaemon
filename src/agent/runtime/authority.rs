//! Tool authority lives in `aidaemon-runtime`; the old paths stay.

pub(crate) use aidaemon_runtime::authority::*;

/// Every `InheritedAuthority` a child agent was created with (test-only
/// capture of the real spawn path).
#[cfg(test)]
pub(crate) static CREATED_CHILD_INHERITANCE: std::sync::Mutex<Vec<InheritedAuthority>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ToolCallAccessManifest, ToolTargetHint, ToolTargetHintKind};
    use crate::types::UserRole;
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
                &crate::tools::DaemonPaths::default(),
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

    /// The integration harness cannot show this: a guest without a grant is
    /// offered no tool, so the kernel refuses an unknown tool before the
    /// dispatcher. Here the dispatcher itself still refuses it.
    #[tokio::test]
    async fn guest_without_grant_still_gets_no_active_workspace_grant() {
        let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
            crate::testing::MockProvider::new(),
            vec![std::sync::Arc::new(crate::tools::WriteFileTool::new(
                crate::execution::active_execution_backend(),
            ))],
            None,
        )
        .await
        .unwrap();
        let ctx = crate::agent::tool_exec::ToolExecCtx {
            file_fence: None,
            mcp_tool: None,
            owner_approval_by_kernel: false,
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
            turn_authority: None,
            trust_roots: None,
            lone_root_call: false,
        };
        use aidaemon_runtime::engine::dispatcher::ToolDispatch as _;
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
}
