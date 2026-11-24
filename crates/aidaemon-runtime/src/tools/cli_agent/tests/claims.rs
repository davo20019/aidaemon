//! Concurrency limits, working-directory claims and deduplication.

use super::*;

#[tokio::test]
async fn test_concurrent_limit_enforced() {
    let (tool, _db) = setup_bash_tool().await;

    // Start long-running async tasks to fill the concurrent limit
    // (real processes so reap_finished won't clean them up)
    for _ in 0..tool.max_concurrent {
        tool.call(r#"{"action":"run","tool":"bash-agent","prompt":"sleep 30","async_mode":true}"#)
            .await
            .unwrap();
    }

    // The next one is refused
    let result = tool
        .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo should-not-run"}"#)
        .await
        .unwrap();
    assert!(
        result.contains(&format!(
            "Maximum {} CLI agents already running",
            tool.max_concurrent
        )) && result.contains("NOT started"),
        "Expected concurrent limit message, got: {}",
        result
    );

    // Clean up: cancel all
    tool.handle_cancel_all("").await.unwrap();
}

// -----------------------------------------------------------------------
// Working directory claim tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_working_dir_conflict_detection() {
    let (tool, _db) = setup_bash_tool().await;

    // Seed a claim as if another task already owns this directory.
    lock_claims(&tool.working_dir_claims).insert(
        "/tmp/project".to_string(),
        vec![WorkingDirClaim {
            task_id: "abc12345".to_string(),
            tool_name: "bash-agent".to_string(),
            prompt_summary: "sleep 60".to_string(),
            writes: WriteScope::Whole,
            state: ClaimState::Active,
        }],
    );

    let result = tool
            .call(
                r#"{"action":"run","tool":"bash-agent","prompt":"echo test","working_dir":"/tmp/project"}"#,
            )
            .await
            .unwrap();
    assert!(
        result.contains("BLOCKED") && result.contains("Another CLI agent"),
        "Expected working dir conflict BLOCKED message, got: {}",
        result
    );
}

#[tokio::test]
async fn test_working_dir_lock_released_after_completion() {
    let (tool, _db) = setup_bash_tool().await;
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir_path = tmp_dir.path().to_str().unwrap();

    let args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"echo done","working_dir":"{}"}}"#,
        dir_path
    );
    let result = tool.call(&args).await.unwrap();
    assert!(result.contains("done"));

    // Working-dir claim should be released after completion.
    let claims = lock_claims(&tool.working_dir_claims);
    assert!(
        !claims.contains_key(dir_path),
        "Working-dir claim not released after completion"
    );
}

#[tokio::test]
async fn test_working_dir_aliases_conflict_after_normalization() {
    let (tool, _db) = setup_bash_tool().await;
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir = tmp_dir.path().to_string_lossy().to_string();
    let dir_with_slash = format!("{}/", dir.trim_end_matches('/'));

    let first_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 5","working_dir":"{}","async_mode":true}}"#,
        dir
    );
    let first = tool.call(&first_args).await.unwrap();
    assert!(
        first.contains("started in background"),
        "Expected first task to start in background, got: {}",
        first
    );

    let second_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"echo test","working_dir":"{}"}}"#,
        dir_with_slash
    );
    let second = tool.call(&second_args).await.unwrap();
    assert!(
        second.contains("BLOCKED") && second.contains("Another CLI agent"),
        "Expected normalized path conflict BLOCKED message, got: {}",
        second
    );

    tool.handle_cancel_all("").await.unwrap();
}

#[tokio::test]
async fn test_working_dir_lock_released_on_spawn_failure() {
    let (tool, _db) = setup_bash_tool().await;
    let missing_dir = "/tmp/aidaemon-cli-agent-missing-dir-lock-test";
    let _ = std::fs::remove_dir_all(missing_dir);
    let normalized = missing_dir.to_string();

    let args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"echo should-fail","working_dir":"{}"}}"#,
        missing_dir
    );
    let result = tool.call(&args).await;
    assert!(
        result.is_err(),
        "Expected spawn failure for missing current_dir, got: {:?}",
        result
    );

    let claims = lock_claims(&tool.working_dir_claims);
    assert!(
        !claims.contains_key(&normalized),
        "Working-dir claim should be released after spawn failure"
    );
}

#[tokio::test]
async fn test_concurrent_limit_includes_sync_runs() {
    let (mut tool, _db) = setup_bash_tool().await;
    tool.max_concurrent = 1;
    tool.concurrency_limiter = Arc::new(Semaphore::new(1));
    let tool = Arc::new(tool);

    let tool_for_first = Arc::clone(&tool);
    let first = tokio::spawn(async move {
        tool_for_first
            .call(r#"{"action":"run","tool":"bash-agent","prompt":"sleep 2; echo first"}"#)
            .await
    });

    tokio::time::sleep(Duration::from_millis(150)).await;

    let second = tool
        .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo second"}"#)
        .await
        .unwrap();
    assert!(
        second.contains("Maximum 1 CLI agents already running"),
        "Expected concurrency rejection while sync run is active, got: {}",
        second
    );

    let first_result = first.await.unwrap().unwrap();
    assert!(
        first_result.contains("first"),
        "Expected first sync run output, got: {}",
        first_result
    );
}

#[tokio::test]
async fn test_sync_inflight_conflict_reports_claim_metadata() {
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir = tmp_dir.path().to_string_lossy().to_string();

    let first_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 2; echo first","working_dir":"{}"}}"#,
        dir
    );
    let tool_for_first = Arc::clone(&tool);
    let first = tokio::spawn(async move { tool_for_first.call(&first_args).await.unwrap() });

    tokio::time::sleep(Duration::from_millis(150)).await;

    let second_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"echo second","working_dir":"{}"}}"#,
        dir
    );
    let second = tool.call(&second_args).await.unwrap();
    assert!(
        second.contains("BLOCKED")
            && second.contains("task_id=")
            && second.contains("agent=bash-agent"),
        "Expected conflict response with claim metadata, got: {}",
        second
    );

    let first_result = first.await.unwrap();
    assert!(
        first_result.contains("first"),
        "Expected first sync run output, got: {}",
        first_result
    );
}

#[tokio::test]
async fn test_sync_inflight_duplicate_prompt_blocked() {
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir = tmp_dir.path().to_string_lossy().to_string();

    let first_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"echo refactor the auth module to use OAuth tokens; sleep 2","working_dir":"{}"}}"#,
        dir
    );
    let tool_for_first = Arc::clone(&tool);
    let first = tokio::spawn(async move { tool_for_first.call(&first_args).await.unwrap() });

    tokio::time::sleep(Duration::from_millis(150)).await;

    let second_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"echo refactor the auth module to use OAuth tokens","working_dir":"{}"}}"#,
        dir
    );
    let second = tool.call(&second_args).await.unwrap();
    assert!(
        second.contains("BLOCKED") && second.contains("Another CLI agent"),
        "Expected a run changing the same folder to be blocked, got: {}",
        second
    );

    let _ = first.await.unwrap();
}

#[tokio::test]
async fn test_working_dir_claim_released_on_semaphore_reject() {
    let (mut tool, _db) = setup_bash_tool().await;
    tool.max_concurrent = 0;
    tool.concurrency_limiter = Arc::new(Semaphore::new(0));

    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir_path = tmp_dir.path().to_str().unwrap();
    let normalized = CliAgentTool::normalize_working_dir(
        &aidaemon_tools::testing::active_execution_backend(),
        dir_path,
    )
    .await
    .unwrap();
    let args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"echo blocked","working_dir":"{}"}}"#,
        dir_path
    );
    let result = tool.call(&args).await.unwrap();
    assert!(
        result.contains("Maximum 0 CLI agents already running"),
        "Expected semaphore rejection, got: {}",
        result
    );

    let claims = lock_claims(&tool.working_dir_claims);
    assert!(
        !claims.contains_key(&normalized),
        "Working-dir claim should be released after semaphore rejection"
    );
}

/// Regression test: a sync-mode dispatch future dropped mid-await (the
/// calling session was cancelled or timed out) must release its
/// working-dir claim. Before WorkingDirClaimGuard, this leaked the claim
/// permanently and blocked all future dispatches to the directory.
#[tokio::test]
async fn test_working_dir_claim_released_when_dispatch_future_dropped() {
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir = tmp_dir.path().to_string_lossy().to_string();
    let normalized = CliAgentTool::normalize_working_dir(
        &aidaemon_tools::testing::active_execution_backend(),
        &dir,
    )
    .await
    .unwrap();

    let args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 15","working_dir":"{}"}}"#,
        dir
    );
    let tool_for_task = Arc::clone(&tool);
    let handle = tokio::spawn(async move { tool_for_task.call(&args).await });

    // Wait until the dispatch has registered its claim.
    let mut claimed = false;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if lock_claims(&tool.working_dir_claims).contains_key(&normalized) {
            claimed = true;
            break;
        }
    }
    assert!(claimed, "Working-dir claim was never registered");

    // Abort the dispatching future mid-await (simulates the calling
    // agent session being cancelled while waiting for completion).
    handle.abort();
    let _ = handle.await;

    assert!(
        !lock_claims(&tool.working_dir_claims).contains_key(&normalized),
        "Working-dir claim should be released when the dispatch future is dropped"
    );
}

/// The guard must NOT release the claim once disarmed (background
/// handoff transfers ownership to the reaper / cancel paths).
#[tokio::test]
async fn test_claim_guard_disarm_keeps_claim() {
    let claims: Arc<WorkingDirClaims> = Arc::new(std::sync::Mutex::new(HashMap::new()));
    lock_claims(&claims).insert(
        "/tmp/project".to_string(),
        vec![WorkingDirClaim {
            task_id: "abc12345".to_string(),
            tool_name: "bash-agent".to_string(),
            prompt_summary: "sleep 60".to_string(),
            writes: WriteScope::Whole,
            state: ClaimState::Active,
        }],
    );

    let notify = Arc::new(tokio::sync::Notify::new());
    let mut guard = WorkingDirClaimGuard::new(
        Arc::clone(&claims),
        Arc::clone(&notify),
        "/tmp/project".to_string(),
        "abc12345".to_string(),
    );
    guard.disarm();
    drop(guard);
    assert!(
        lock_claims(&claims).contains_key("/tmp/project"),
        "Disarmed guard must not release the claim"
    );

    // An armed guard owned by a DIFFERENT task must not steal the claim.
    let other_guard = WorkingDirClaimGuard::new(
        Arc::clone(&claims),
        Arc::clone(&notify),
        "/tmp/project".to_string(),
        "other999".to_string(),
    );
    drop(other_guard);
    assert!(
        lock_claims(&claims).contains_key("/tmp/project"),
        "Guard for a different task must not release someone else's claim"
    );

    // The owning guard releases it.
    let owner_guard = WorkingDirClaimGuard::new(
        Arc::clone(&claims),
        Arc::clone(&notify),
        "/tmp/project".to_string(),
        "abc12345".to_string(),
    );
    drop(owner_guard);
    assert!(
        !lock_claims(&claims).contains_key("/tmp/project"),
        "Owning guard must release the claim on drop"
    );
}

// -----------------------------------------------------------------------
// Async mode test
// -----------------------------------------------------------------------

#[test]
fn write_scopes_overlap_only_where_both_may_change_the_same_files() {
    let paths = |p: &[&str]| WriteScope::Paths(p.iter().map(|p| p.to_string()).collect());
    let reviews = paths(&["/w/synthetic/reviews/claude.md"]);
    assert!(!reviews.overlaps(&paths(&["/w/synthetic/reviews/codex.md"])));
    assert!(reviews.overlaps(&paths(&["/w/synthetic/reviews"])));
    assert!(reviews.overlaps(&WriteScope::Whole));
    assert!(!WriteScope::ReadOnly.overlaps(&WriteScope::Whole));
    // A sibling whose name starts the same is not inside it.
    assert!(!paths(&["/w/synthetic/src"]).overlaps(&paths(&["/w/synthetic/src2/a.rs"])));
}

// -----------------------------------------------------------------------
// Task deduplication test
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_duplicate_task_detection() {
    let (tool, _db) = setup_bash_tool().await;
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir = tmp_dir.path().to_string_lossy().to_string();

    // Start a background task
    let first_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"refactor the auth module to use JWT tokens","working_dir":"{}","async_mode":true}}"#,
        dir
    );
    let first = tool.call(&first_args).await.unwrap();
    assert!(
        first.contains("started in background"),
        "Expected first task to start, got: {}",
        first
    );

    // The same task again, changing the same folder: blocked.
    let second_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"refactor the auth module to use JWT tokens","working_dir":"{}"}}"#,
        dir
    );
    let second = tool.call(&second_args).await.unwrap();
    assert!(
        second.contains("BLOCKED") && second.contains("Another CLI agent"),
        "Expected the same folder's writer to be blocked, got: {}",
        second
    );

    tool.handle_cancel_all("").await.unwrap();
}

/// Agents given the same task side by side, each writing its own file,
/// all start: what they may change does not overlap.
#[tokio::test]
async fn runs_that_change_different_files_in_one_folder_start_side_by_side() {
    let (tool, _db) = setup_bash_tool().await;
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let dir = tmp_dir.path().to_string_lossy().to_string();
    for name in ["a", "b"] {
        let args = format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 5; review the module","working_dir":"{dir}","write_paths":["{dir}/reviews/{name}.md"],"async_mode":true}}"#
        );
        let started = tool.call(&args).await.unwrap();
        assert!(
            started.contains("started in background"),
            "Expected run {name} to start, got: {started}"
        );
    }
    // A third that may change anything in the folder waits for them.
    let whole = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"review the module","working_dir":"{dir}"}}"#
    );
    assert!(tool.call(&whole).await.unwrap().contains("BLOCKED"));

    tool.handle_cancel_all("").await.unwrap();
}

/// A run's write paths that are files name what it wrote, however it wrote
/// them: a file created or changed is listed, one left alone is not.
#[tokio::test]
async fn the_files_a_run_wrote_are_found_by_their_marks() {
    let (tool, _db) = setup_bash_tool().await;
    let dir = tempfile::TempDir::new().unwrap();
    let written = dir.path().join("review.md").to_string_lossy().to_string();
    let kept = dir.path().join("notes.md").to_string_lossy().to_string();
    std::fs::write(&kept, "kept").unwrap();
    let roots = vec![written.clone(), kept.clone()];
    let before = file_marks(&tool.backend, &roots).await;
    std::fs::write(&written, "a synthetic review").unwrap();
    assert_eq!(
        changed_files(&tool.backend, &before, &roots).await,
        vec![written]
    );
}

/// A write path that is a folder names each file the run changed, created
/// or removed in it, however it did (live 2026-11-23: Codex appended a line
/// with a shell redirect and the run read as changing nothing); its
/// repository's own records are not read file by file.
#[tokio::test]
async fn the_files_a_run_changed_in_its_folder_are_found() {
    let (tool, _db) = setup_bash_tool().await;
    let dir = tempfile::TempDir::new().unwrap();
    let path = |name: &str| dir.path().join(name).to_string_lossy().to_string();
    std::fs::write(path("note.md"), "hello\n").unwrap();
    std::fs::write(path("gone.md"), "x").unwrap();
    std::fs::write(path("kept.md"), "kept").unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(path(".git/index"), "a").unwrap();
    let roots = vec![dir.path().to_string_lossy().to_string()];
    let before = file_marks(&tool.backend, &roots).await;
    std::fs::write(path("note.md"), "hello\nworld\n").unwrap();
    std::fs::write(path("new.md"), "new").unwrap();
    std::fs::remove_file(path("gone.md")).unwrap();
    std::fs::write(path(".git/index"), "changed").unwrap();
    let mut changed = changed_files(&tool.backend, &before, &roots).await;
    changed.sort();
    let mut expected = vec![path("gone.md"), path("new.md"), path("note.md")];
    expected.sort();
    assert_eq!(changed, expected);
}

#[test]
fn runs_in_different_folders_collide_where_their_roots_meet() {
    let paths = |p: &[&str]| WriteScope::Paths(p.iter().map(|p| p.to_string()).collect());
    // A run over a whole repository and one in its src folder.
    assert!(WriteScope::Whole.collides("/w/synthetic", &WriteScope::Whole, "/w/synthetic/src"));
    assert!(paths(&["/w/synthetic/src"]).collides(
        "/w/synthetic",
        &paths(&["/w/synthetic/src/lib.rs"]),
        "/w/synthetic/src"
    ));
    // Two folders' runs given the same write path.
    assert!(paths(&["/w/shared/out.md"]).collides(
        "/w/synthetic-a",
        &paths(&["/w/shared/out.md"]),
        "/w/synthetic-b"
    ));
    // Sibling folders, and a read-only run, do not.
    assert!(!WriteScope::Whole.collides("/w/synthetic-a", &WriteScope::Whole, "/w/synthetic-b"));
    assert!(!WriteScope::ReadOnly.collides("/w/synthetic", &WriteScope::Whole, "/w/synthetic/src"));
}

#[test]
fn a_claim_in_an_enclosing_folder_conflicts_unless_its_run_is_gone() {
    let mut claims: HashMap<String, Vec<WorkingDirClaim>> = HashMap::new();
    claims.insert(
        "/w/synthetic".to_string(),
        vec![WorkingDirClaim {
            task_id: "outer-run".to_string(),
            tool_name: "bash-agent".to_string(),
            prompt_summary: "refactor".to_string(),
            writes: WriteScope::Whole,
            state: ClaimState::Active,
        }],
    );
    let (dir, claim) =
        conflicting(&claims, "/w/synthetic/src", &WriteScope::Whole, |_| false).unwrap();
    assert_eq!((dir, claim.task_id.as_str()), ("/w/synthetic", "outer-run"));
    // A run that already finished holds its claim until the reaper runs:
    // it no longer counts.
    assert!(
        conflicting(&claims, "/w/synthetic/src", &WriteScope::Whole, |id| id
            == "outer-run")
        .is_none()
    );
}

#[tokio::test]
async fn exactly_one_of_a_runs_end_and_its_timed_out_caller_answers_for_it() {
    // The run ends first: its caller, still waiting, answers.
    let shared = RunShared::default();
    assert_eq!(shared.ends_by(), EndsBy::Caller);
    assert!(shared.move_to_background().is_none());
    // The caller's wait runs out first: the run delivers its own end, once
    // the caller has set up its background run.
    let shared = RunShared::default();
    let done = shared.move_to_background().expect("it moves");
    assert_eq!(shared.ends_by(), EndsBy::Moved);
    let waiter = {
        let moved = shared.moved.clone();
        tokio::spawn(async move { moved.notified().await })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!waiter.is_finished(), "its end waits for the move");
    drop(done);
    tokio::time::timeout(Duration::from_secs(1), waiter)
        .await
        .expect("the move's end wakes it")
        .unwrap();
    // A background run always delivers its own end.
    let shared = RunShared::default();
    shared.starts_in_background();
    assert_eq!(shared.ends_by(), EndsBy::Background);
}

/// Working-dir conflicts (`BLOCKED: ...`) and concurrency-cap refusals
/// (`Maximum ...`) are typed as retryable administrative failures, whereas
/// policy refusals (`Blocked: ...`) are permanent administrative failures.
#[tokio::test]
async fn prelaunch_refusals_have_typed_retryable_and_permanent_outcomes() {
    let (mut tool, _db) = setup_bash_tool().await;
    tool.max_concurrent = 1;
    tool.concurrency_limiter = Arc::new(Semaphore::new(1));
    let dir = tempfile::TempDir::new().unwrap();
    let dir_path = dir.path().to_string_lossy().to_string();

    // Start one background run in `dir_path`.
    let first = tool
        .call(&format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 30","working_dir":"{dir_path}","async_mode":true}}"#
        ))
        .await
        .unwrap();
    assert!(first.contains("started in background"), "{first}");

    // 1. Working-dir conflict -> FailedRetryable + administrative.
    let conflict = tool
        .call_with_status_outcome(
            &format!(
                r#"{{"action":"run","tool":"bash-agent","prompt":"echo conflict","working_dir":"{dir_path}"}}"#
            ),
            None,
        )
        .await
        .unwrap();
    assert!(conflict.output.contains("BLOCKED"), "{}", conflict.output);
    assert_eq!(
        conflict.metadata.outcome_status,
        Some(ToolOutcomeStatus::FailedRetryable),
        "working-dir conflict must be FailedRetryable, not Succeeded"
    );
    assert_eq!(
        conflict.metadata.semantics.effect,
        aidaemon_tools::semantics::ToolCallEffect::Administrative
    );

    // 2. Foreground concurrency limit (different directory) -> FailedRetryable + administrative.
    let other_dir = tempfile::TempDir::new().unwrap();
    let other_path = other_dir.path().to_string_lossy().to_string();
    let capped = tool
        .call_with_status_outcome(
            &format!(
                r#"{{"action":"run","tool":"bash-agent","prompt":"echo capped","working_dir":"{other_path}"}}"#
            ),
            None,
        )
        .await
        .unwrap();
    assert!(capped.output.contains("Maximum 1"), "{}", capped.output);
    assert_eq!(
        capped.metadata.outcome_status,
        Some(ToolOutcomeStatus::FailedRetryable),
        "concurrency limit refusal must be FailedRetryable, not Succeeded"
    );
    assert_eq!(
        capped.metadata.semantics.effect,
        aidaemon_tools::semantics::ToolCallEffect::Administrative
    );

    // 3. Policy refusal (`Blocked: ...`) -> FailedPermanent + administrative.
    let policy = super::super::tool_impl::cli_agent_outcome_with_refusal(
        "run",
        "Blocked: CLI delegation was not started because policy forbids it.".to_string(),
        false,
        false,
        false,
        Some(PrelaunchRefusal::Policy),
    );
    assert_eq!(
        policy.metadata.outcome_status,
        Some(ToolOutcomeStatus::FailedPermanent)
    );
    assert_eq!(
        policy.metadata.semantics.effect,
        aidaemon_tools::semantics::ToolCallEffect::Administrative
    );

    tool.handle_cancel_all("").await.unwrap();
}

/// A read-write run with `working_dir` omitted but explicit `write_paths`
/// still acquires a claim and blocks overlapping concurrent writes.
#[tokio::test]
async fn explicit_write_paths_without_working_dir_enforce_claims() {
    let (tool, _db) = setup_bash_tool().await;
    let dir = tempfile::TempDir::new().unwrap();
    let target = dir.path().join("notes.md").to_string_lossy().to_string();

    let first = tool
        .call(&format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"sleep 30","write_paths":["{target}"],"async_mode":true}}"#
        ))
        .await
        .unwrap();
    assert!(first.contains("started in background"), "{first}");

    let second = tool
        .call_with_status_outcome(
            &format!(
                r#"{{"action":"run","tool":"bash-agent","prompt":"echo overlap","write_paths":["{target}"]}}"#
            ),
            None,
        )
        .await
        .unwrap();
    assert!(
        second.output.contains("BLOCKED"),
        "overlapping write_paths without working_dir must be blocked: {}",
        second.output
    );
    assert_eq!(
        second.metadata.outcome_status,
        Some(ToolOutcomeStatus::FailedRetryable)
    );

    tool.handle_cancel_all("").await.unwrap();
}

/// Reserved claims (queued / starting) say "starting in" instead of claiming
/// the agent is already running.
#[test]
fn reserved_claim_conflict_message_distinguishes_starting_from_running() {
    let reserved = WorkingDirClaim {
        task_id: "res-1".to_string(),
        tool_name: "codex".to_string(),
        prompt_summary: "edit lib.rs".to_string(),
        writes: WriteScope::Whole,
        state: ClaimState::Reserved,
    };
    let active = WorkingDirClaim {
        task_id: "res-1".to_string(),
        tool_name: "codex".to_string(),
        prompt_summary: "edit lib.rs".to_string(),
        writes: WriteScope::Whole,
        state: ClaimState::Active,
    };
    let reserved_msg = conflict_message("/w/synthetic", &reserved);
    let active_msg = conflict_message("/w/synthetic", &active);
    assert!(reserved_msg.contains("queued (reserved)"), "{reserved_msg}");
    assert!(active_msg.contains("is already working in"), "{active_msg}");
}

/// Single-pass `snapshot_roots` skips `UNMARKED` subdirectories consistently
/// for both the aggregate receipt and per-file marks, so internal `.git` or
/// `target` churn does not flip `changed_something`.
#[tokio::test]
async fn single_pass_snapshot_skips_unmarked_dirs_consistently() {
    let (tool, _db) = setup_bash_tool().await;
    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path().to_string_lossy().to_string();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::create_dir(dir.path().join("target")).unwrap();
    std::fs::write(dir.path().join(".git/index"), "v1").unwrap();
    std::fs::write(dir.path().join("target/cache"), "v1").unwrap();
    std::fs::write(dir.path().join("lib.rs"), "fn main() {}\n").unwrap();

    let before = snapshot_roots(&tool.backend, &[root.clone()]).await;
    // Mutate only unmarked directories.
    std::fs::write(dir.path().join(".git/index"), "v2-longer").unwrap();
    std::fs::write(dir.path().join("target/new_artifact"), "bytes").unwrap();
    let after_unmarked = snapshot_roots(&tool.backend, &[root.clone()]).await;
    assert_eq!(
        before.receipt, after_unmarked.receipt,
        "unmarked directory churn must not alter TreeReceipt"
    );
    assert!(
        changed_files_from(&before.marks, &after_unmarked.marks).is_empty(),
        "unmarked directory churn must not appear in changed_files"
    );

    // Now mutate a tracked source file.
    std::fs::write(
        dir.path().join("lib.rs"),
        "fn main() { println!(\"hi\"); }\n",
    )
    .unwrap();
    let after_edit = snapshot_roots(&tool.backend, &[root]).await;
    assert_ne!(before.receipt, after_edit.receipt);
    assert_eq!(
        changed_files_from(&before.marks, &after_edit.marks),
        vec![dir.path().join("lib.rs").to_string_lossy().to_string()]
    );
}
