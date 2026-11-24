//! The enriched prompt, git diff capture and worktree scope.

use super::*;

#[tokio::test]
async fn test_capture_git_diff_no_repo() {
    let tmp = tempfile::TempDir::new().unwrap();
    let result = CliAgentTool::capture_git_diff(
        &aidaemon_tools::testing::active_execution_backend(),
        &crate::tools::cli_agent::prompt::DiffBase {
            dir: tmp.path().to_str().unwrap().to_string(),
            head: None,
        },
    )
    .await;
    assert!(result.is_none(), "Non-git directory should return None");
}

#[tokio::test]
async fn trusted_cli_delegation_requires_structural_worktree_scope() {
    let (tool, _db) = setup_echo_tool().await;
    for prompt in [
        "Build, deploy, and verify the exact public URL",
        "Build locally but do not deploy",
        "Inspect the code and return a report",
    ] {
        let args = json!({
            "action": "run",
            "tool": "echo",
            "prompt": prompt,
        });
        // A trusted session, as the dispatcher's typed context says
        // (`ToolCallContext::trusted_session`).
        let result = tool
            .call_with_status_under_contract(
                &args.to_string(),
                None,
                false,
                false,
                &ToolCallContext {
                    trusted_session: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(
            result.contains("requires an explicit working_dir"),
            "prompt wording must not narrow an opaque unattended delegation: {result}"
        );
    }
}

/// Trust comes only from the dispatcher's typed context: a
/// `_trusted_session` argument makes the run neither trusted nor
/// unattended.
#[tokio::test]
async fn trusted_session_argument_bypasses_nothing() {
    let (tool, _db) = setup_echo_tool().await;
    let args = json!({
        "action": "run",
        "tool": "echo",
        "prompt": "Inspect the code and return a report",
        "_trusted_session": true
    });
    let result = tool.call(&args.to_string()).await.unwrap();
    assert!(
        !result.contains("unattended CLI delegation"),
        "an argument is not the dispatcher's trust: {result}"
    );
}

#[tokio::test]
async fn test_capture_git_diff_with_changes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().to_str().unwrap();

    // Initialize a git repo with a commit
    tokio::process::Command::new("git")
        .args(["init"])
        .current_dir(dir)
        .output()
        .await
        .unwrap();
    tokio::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(dir)
        .output()
        .await
        .unwrap();
    tokio::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(dir)
        .output()
        .await
        .unwrap();

    // Create and commit a file
    std::fs::write(tmp.path().join("file.txt"), "initial").unwrap();
    tokio::process::Command::new("git")
        .args(["add", "."])
        .current_dir(dir)
        .output()
        .await
        .unwrap();
    tokio::process::Command::new("git")
        .args(["commit", "-m", "initial"])
        .current_dir(dir)
        .output()
        .await
        .unwrap();

    // Modify the file (uncommitted change)
    std::fs::write(tmp.path().join("file.txt"), "modified content").unwrap();

    assert_eq!(
        CliAgentTool::dirty_worktree_entry_count(
            &aidaemon_tools::testing::active_execution_backend(),
            dir
        )
        .await,
        Some(1)
    );

    let base = crate::tools::cli_agent::prompt::DiffBase {
        dir: dir.to_string(),
        head: None,
    };
    let result =
        CliAgentTool::capture_git_diff(&aidaemon_tools::testing::active_execution_backend(), &base)
            .await;
    assert!(result.is_some(), "Should capture uncommitted changes");
    let diff = result.unwrap();
    assert!(
        diff.contains("modified content") || diff.contains("file.txt"),
        "Diff should mention the changed file, got: {}",
        diff
    );
}

// -----------------------------------------------------------------------
// Unknown action test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_build_enriched_prompt_basic() {
    let (tool, _db) = setup_echo_tool().await;

    let prompt = tool
        .build_enriched_prompt(
            "test-session",
            "You are a security auditor",
            "Audit this codebase",
            None,
            None,
        )
        .await;

    assert!(prompt.contains("You are a security auditor"));
    assert!(prompt.contains("Audit this codebase"));
    assert!(prompt.contains("## Task"));
    assert!(prompt.contains("## Instructions"));
}

#[tokio::test]
async fn test_build_enriched_prompt_no_instruction() {
    let (tool, _db) = setup_echo_tool().await;

    let prompt = tool
        .build_enriched_prompt("test-session", "", "Just do the task", None, None)
        .await;

    // Empty instruction should not appear
    assert!(prompt.contains("Just do the task"));
    assert!(prompt.contains("## Task"));
}

#[tokio::test]
async fn test_enriched_prompt_uses_scoped_agents_instead_of_readme() {
    let (tool, _db) = setup_echo_tool().await;
    let project = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(project.path().join(".git")).unwrap();
    std::fs::write(
        project.path().join("AGENTS.md"),
        "RUN_THE_SCOPED_AGENT_TEST",
    )
    .unwrap();
    std::fs::write(
        project.path().join("README.md"),
        "README_MUST_NOT_BECOME_INSTRUCTIONS",
    )
    .unwrap();

    let prompt = tool
        .build_enriched_prompt(
            "test-session",
            "You are a code reviewer",
            "Review this codebase",
            project.path().to_str(),
            None,
        )
        .await;

    assert!(prompt.contains("## Project Instructions"));
    assert!(prompt.contains("RUN_THE_SCOPED_AGENT_TEST"));
    assert!(!prompt.contains("README_MUST_NOT_BECOME_INSTRUCTIONS"));
}

// -----------------------------------------------------------------------
// Scenario replication: exact user prompt that caused the hang
// -----------------------------------------------------------------------

/// A run reports only commits made since it started: the folder's last
/// commit from before the run is someone else's work.
#[tokio::test]
async fn a_commit_from_before_the_run_is_not_reported_as_its_change() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().to_str().unwrap();
    let git = |args: &'static [&'static str]| {
        let dir = dir.to_string();
        async move {
            tokio::process::Command::new("git")
                .args(args)
                .current_dir(dir)
                .output()
                .await
                .unwrap()
        }
    };
    git(&["init"]).await;
    git(&["config", "user.email", "synthetic@example.com"]).await;
    git(&["config", "user.name", "Synthetic Tester"]).await;
    std::fs::write(tmp.path().join("old.txt"), "before the run").unwrap();
    git(&["add", "."]).await;
    git(&["commit", "-m", "earlier work"]).await;
    let backend = aidaemon_tools::testing::active_execution_backend();
    let base = crate::tools::cli_agent::prompt::DiffBase {
        dir: dir.to_string(),
        head: CliAgentTool::git_head(&backend, dir).await,
    };
    assert!(base.head.is_some());
    assert!(
        CliAgentTool::capture_git_diff(&backend, &base)
            .await
            .is_none(),
        "the earlier commit is not the run's"
    );
    std::fs::write(tmp.path().join("new.txt"), "made by the run").unwrap();
    git(&["add", "."]).await;
    git(&["commit", "-m", "the run's work"]).await;
    let diff = CliAgentTool::capture_git_diff(&backend, &base)
        .await
        .unwrap();
    assert!(diff.contains("new.txt"), "{diff}");
    assert!(!diff.contains("old.txt"), "{diff}");
}
