//! Tool dispatch (`aidaemon_runtime::engine::dispatcher`) on the daemon's
//! agent and test harness.

use super::*;
use crate::types::WorkspaceGrant;
pub(super) use aidaemon_runtime::engine::dispatcher::{
    sanitize_workspace_tool_outcome, ToolDispatch as _,
};
pub(super) use aidaemon_runtime::engine::tool_exec::{ToolExecCtx, UnknownToolError};

impl Agent {
    /// One tool call through the dispatcher's watchdog, its output only.
    pub(super) async fn execute_tool_with_watchdog(
        &self,
        name: &str,
        arguments: &str,
        ctx: &ToolExecCtx<'_>,
    ) -> anyhow::Result<String> {
        self.execute_tool_with_watchdog_outcome(name, arguments, ctx)
            .await
            .map(|outcome| outcome.output)
    }
}

#[path = "tool_watchdog_tests.rs"]
mod tool_watchdog_tests;

#[path = "tool_call_context_tests.rs"]
mod tool_call_context_tests;

mod scoped_workspace_tests {
    use super::*;
    use crate::types::WorkspaceAccessLevel;

    async fn scoped_workspace_arguments(
        name: &str,
        arguments: &str,
        grant: &WorkspaceGrant,
        paths: &crate::tools::DaemonPaths,
    ) -> anyhow::Result<Value> {
        aidaemon_runtime::engine::dispatcher::scoped_workspace_arguments(
            &*crate::execution::active_execution_backend(),
            name,
            arguments,
            grant,
            paths,
        )
        .await
    }

    fn grant(root: &std::path::Path) -> WorkspaceGrant {
        std::fs::write(root.join("package.json"), r#"{"name":"test-project"}"#).unwrap();
        WorkspaceGrant {
            platform: "slack".to_string(),
            workspace_id: "T_TEST".to_string(),
            channel_id: "C_PRIVATE".to_string(),
            user_id: "U_GUEST".to_string(),
            project_root: std::fs::canonicalize(root)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            access: WorkspaceAccessLevel::Edit,
            expires_at: Utc::now() + chrono::Duration::hours(1),
        }
    }

    #[test]
    fn scoped_outcomes_hide_absolute_roots_and_secret_values() {
        let root = tempfile::tempdir().unwrap();
        let grant = grant(root.path());
        let secret = "sk-abcdefghijklmnopqrstuvwxyz123456";
        let outcome = crate::traits::ToolCallOutcome::from_output(format!(
            "Read {}/src/main.rs\nAPI_KEY={secret}",
            grant.project_root
        ));

        let sanitized = sanitize_workspace_tool_outcome(outcome, &grant);
        assert!(sanitized.output.contains("./src/main.rs"));
        assert!(!sanitized.output.contains(&grant.project_root));
        assert!(!sanitized.output.contains(secret));
        assert!(sanitized.output.contains("[REDACTED:API key]"));
    }

    #[tokio::test]
    async fn scoped_arguments_root_relative_paths_and_block_escape() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("index.html"), "hello").unwrap();
        std::fs::write(root.path().join(".env.local"), "TOKEN=nope").unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let grant = grant(root.path());

        let inside = scoped_workspace_arguments(
            "read_file",
            r#"{"path":"index.html"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::canonicalize(inside["path"].as_str().unwrap()).unwrap(),
            std::fs::canonicalize(root.path().join("index.html")).unwrap()
        );

        let traversal = scoped_workspace_arguments(
            "read_file",
            r#"{"path":"../outside.txt"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(traversal.is_err());

        let absolute_outside = scoped_workspace_arguments(
            "read_file",
            &json!({"path": outside.path()}).to_string(),
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(absolute_outside.is_err());

        let sensitive = scoped_workspace_arguments(
            "read_file",
            r#"{"path":".env.local"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(sensitive.is_err());

        let dangerous = scoped_workspace_arguments(
            "terminal",
            r#"{"command":"pwd"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(dangerous.is_err());

        let aliased = scoped_workspace_arguments(
            "read_file",
            r#"{"file_path":"index.html","filename":"../../outside"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await
        .unwrap();
        assert!(aliased.get("file_path").is_none());
        assert!(aliased.get("filename").is_none());
        assert_eq!(
            std::fs::canonicalize(aliased["path"].as_str().unwrap()).unwrap(),
            std::fs::canonicalize(root.path().join("index.html")).unwrap()
        );
    }

    #[tokio::test]
    async fn scoped_write_requires_existing_parent() {
        let root = tempfile::tempdir().unwrap();
        let grant = grant(root.path());

        let allowed = scoped_workspace_arguments(
            "write_file",
            r#"{"path":"new.txt","content":"ok"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await
        .unwrap();
        assert!(std::path::Path::new(allowed["path"].as_str().unwrap())
            .starts_with(std::path::Path::new(&grant.project_root)));

        let missing_parent = scoped_workspace_arguments(
            "write_file",
            r#"{"path":"missing/new.txt","content":"no"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(missing_parent.is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn scoped_read_blocks_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        symlink(outside.path(), root.path().join("link.txt")).unwrap();
        let grant = grant(root.path());

        let result = scoped_workspace_arguments(
            "read_file",
            r#"{"path":"link.txt"}"#,
            &grant,
            &crate::tools::DaemonPaths::default(),
        )
        .await;
        assert!(result.is_err());
    }

    /// The dispatcher's per-tool treatment comes from facts the tools
    /// declare, never their names: each fact is pinned to the tools that
    /// hold it.
    #[tokio::test]
    async fn dispatch_treatment_is_declared_by_the_tools() {
        use crate::traits::{GoalContext, Tool};
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state: std::sync::Arc<dyn crate::traits::StateStore> = harness.state.clone();
        let backend = crate::execution::active_execution_backend;
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(crate::tools::WriteFileTool::new(backend())),
            Box::new(crate::tools::EditFileTool::new(backend())),
            Box::new(crate::tools::RunCommandTool::new(backend())),
            Box::new(crate::tools::spawn::SpawnAgentTool::new_deferred(4000, 60)),
            Box::new(crate::tools::ManagePeopleTool::new(
                state.clone(),
                harness.state.memory_stores(),
            )),
            Box::new(crate::tools::ManageMemoriesTool::new(
                state.clone(),
                harness.state.memory_stores(),
            )),
            Box::new(crate::tools::ManageGoalTasksTool::new(
                "goal-synthetic-1".to_string(),
                state.clone(),
            )),
        ];
        for tool in &tools {
            let facts = tool.dispatch_facts();
            let name = tool.name();
            assert_eq!(
                facts.checkpoints_writes,
                matches!(name, "write_file" | "edit_file" | "run_command"),
                "{name}"
            );
            assert_eq!(
                facts.status_from_arguments,
                matches!(name, "manage_people" | "manage_memories"),
                "{name}"
            );
            let goal = match name {
                "spawn_agent" => GoalContext::Goal,
                "manage_goal_tasks" => GoalContext::MandateRun,
                _ => GoalContext::None,
            };
            assert_eq!(facts.goal_context, goal, "{name}");
            assert_eq!(facts.manages_own_timeout, name == "spawn_agent", "{name}");
            assert!(!facts.records_launch_trust, "{name}");
            assert!(!facts.needs_live_model, "{name}");
        }
    }
}
