//! Concurrency limits, working-directory claims and deduplication.

use super::*;

#[tokio::test]
async fn test_concurrent_limit_enforced() {
    let (tool, _db) = setup_bash_tool().await;

    // Start 3 long-running async tasks to fill the concurrent limit
    // (real processes so reap_finished won't clean them up)
    for _ in 0..3 {
        tool.call(r#"{"action":"run","tool":"bash-agent","prompt":"sleep 30","async_mode":true}"#)
            .await
            .unwrap();
    }

    // The 4th should be rejected
    let result = tool
        .call(r#"{"action":"run","tool":"bash-agent","prompt":"echo should-not-run"}"#)
        .await
        .unwrap();
    assert!(
        result.contains("Maximum 3 CLI agents already running"),
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
        WorkingDirClaim {
            task_id: "abc12345".to_string(),
            tool_name: "bash-agent".to_string(),
            prompt_summary: "sleep 60".to_string(),
            dedup_prompt: make_dedup_prompt("sleep 60"),
        },
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
        second.contains("BLOCKED") && second.contains("similar task"),
        "Expected duplicate-task block for sync in-flight run, got: {}",
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
        WorkingDirClaim {
            task_id: "abc12345".to_string(),
            tool_name: "bash-agent".to_string(),
            prompt_summary: "sleep 60".to_string(),
            dedup_prompt: make_dedup_prompt("sleep 60"),
        },
    );

    let mut guard = WorkingDirClaimGuard::new(
        Arc::clone(&claims),
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
fn test_prompt_similarity_scores() {
    // Identical prompts → 1.0
    assert!(
        (prompt_similarity("refactor the auth module", "refactor the auth module") - 1.0).abs()
            < f64::EPSILON
    );

    // Very similar prompts → high similarity
    let sim = prompt_similarity(
        "refactor the auth module to use JWT",
        "refactor the auth module to use OAuth",
    );
    assert!(
        sim > 0.5,
        "Similar prompts should score > 0.5, got: {:.3}",
        sim
    );

    // Completely different prompts → low similarity
    let sim = prompt_similarity("refactor the auth module", "deploy to production server");
    assert!(
        sim < 0.3,
        "Different prompts should score < 0.3, got: {:.3}",
        sim
    );

    // Empty prompts
    assert!((prompt_similarity("", "") - 1.0).abs() < f64::EPSILON);
    assert!((prompt_similarity("hello", "") - 0.0).abs() < f64::EPSILON);

    // Single-word prompts (unigram fallback)
    assert!((prompt_similarity("deploy", "deploy") - 1.0).abs() < f64::EPSILON);
    assert!((prompt_similarity("deploy", "refactor") - 0.0).abs() < f64::EPSILON);
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

    // Try a very similar prompt to the same dir → should be blocked
    let second_args = format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"refactor the auth module to use OAuth tokens","working_dir":"{}"}}"#,
        dir
    );
    let second = tool.call(&second_args).await.unwrap();
    assert!(
        second.contains("BLOCKED") && second.contains("similar task"),
        "Expected duplicate task BLOCKED message, got: {}",
        second
    );

    tool.handle_cancel_all("").await.unwrap();
}
