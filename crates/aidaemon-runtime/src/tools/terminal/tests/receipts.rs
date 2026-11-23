//! Run receipts, background hand-off and the running-work list.

use super::*;

#[test]
fn extract_terminal_exit_code_parses_marker() {
    assert_eq!(
        extract_terminal_exit_code(
            "[Process pid=123 finished after 2s]\nall done\n[exit code: 42]"
        ),
        Some(42)
    );
}

#[test]
fn deletion_observation_candidates_nominate_declared_targets_and_operands() {
    // Declared write targets and parsed deletion operands are nominated
    // for pre/post observation; syntax alone never types the receipt.
    let candidates = deletion_observation_candidates(
        "rm -rf /tmp/x/old",
        Some("/tmp"),
        &[],
        &["/tmp/x".to_string()],
    );
    assert!(candidates.contains(&"/tmp/x".to_string()));
    assert!(candidates.contains(&"/tmp/x/old".to_string()));

    // Relative deletion operands resolve against the working directory.
    let relative = deletion_observation_candidates("rmdir old-dir", Some("/tmp/work"), &[], &[]);
    assert!(relative.contains(&"/tmp/work/old-dir".to_string()));

    // A deletion program named in quoted DATA is not an invocation and
    // nominates nothing beyond the declared targets.
    let quoted = deletion_observation_candidates(
        "echo 'rm -rf /' > /tmp/x/notes.txt",
        Some("/tmp"),
        &["/tmp/x/notes.txt".to_string()],
        &[],
    );
    assert_eq!(quoted, vec!["/tmp/x/notes.txt".to_string()]);

    // Globs are never stat candidates; find only nominates with -delete.
    let globbed = deletion_observation_candidates("rm -rf /tmp/x/*", Some("/tmp"), &[], &[]);
    assert!(globbed.is_empty());
    let find_plain = deletion_observation_candidates("find /tmp/x -name f", Some("/tmp"), &[], &[]);
    assert!(find_plain.is_empty());
    let find_delete =
        deletion_observation_candidates("find /tmp/x -name f -delete", Some("/tmp"), &[], &[]);
    assert!(find_delete.contains(&"/tmp/x".to_string()));
}

#[tokio::test]
async fn deletion_run_receipt_carries_destructive_effect() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await;

    // Under /tmp like the live disposable-workflow paths; the platform
    // temp dir (/var/folders/…) trips the broad-path delete guard.
    let root = std::path::PathBuf::from(format!("/tmp/aid-del-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&root).expect("create disposable dir");
    let args = serde_json::json!({
        "action": "run",
        "command": format!("rm -rf {}", root.to_string_lossy()),
        "write_roots": [root.to_string_lossy()],
    });
    let outcome = tool
        .call_with_execution_context(
            &args.to_string(),
            None,
            aidaemon_tools::testing::call_facts("sess_del_receipt"),
        )
        .await
        .unwrap();
    assert_eq!(outcome.metadata.exit_code, Some(0), "{}", outcome.output);
    assert!(!root.exists());
    // The receipt is the only thing that can close a checklist obligation
    // the executor declared `destructive`.
    assert!(
        outcome
            .metadata
            .semantics
            .mutation_effects
            .contains(ToolMutationEffects::DESTRUCTIVE)
    );
    assert!(
        outcome
            .metadata
            .semantics
            .mutation_effects
            .contains(ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
    );

    // A run that leaves its declared target in place gains no
    // destructive typing: the effect comes from the observed transition,
    // not from any command vocabulary.
    let keep = std::path::PathBuf::from(format!("/tmp/aid-keep-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&keep).expect("create persistent dir");
    let args = serde_json::json!({
        "action": "run",
        "command": format!("ls {}", keep.to_string_lossy()),
        "write_roots": [keep.to_string_lossy()],
    });
    let outcome = tool
        .call_with_execution_context(
            &args.to_string(),
            None,
            aidaemon_tools::testing::call_facts("sess_del_receipt"),
        )
        .await
        .unwrap();
    assert_eq!(outcome.metadata.exit_code, Some(0), "{}", outcome.output);
    assert!(
        !outcome
            .metadata
            .semantics
            .mutation_effects
            .contains(ToolMutationEffects::DESTRUCTIVE)
    );
    let _ = std::fs::remove_dir_all(&keep);
}

/// A successful fetch-only command names the URL it reached on its
/// receipt, as written; a failed fetch, or one a later step could mask,
/// names none.
#[tokio::test]
async fn fetch_run_receipt_names_the_url_only_when_the_fetch_is_proven() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let (_approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        state.pool(),
    )
    .await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        while let Ok((mut socket, _)) = listener.accept().await {
            let mut request = [0u8; 1024];
            let _ = socket.read(&mut request).await;
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .await;
        }
    });
    let url = format!("http://127.0.0.1:{port}/synthetic");
    let run = |command: String| {
        let tool = &tool;
        async move {
            let args = serde_json::json!({
                "action": "run",
                "command": command,
                "network": true,
            });
            tool.call_with_execution_context(
                &args.to_string(),
                None,
                aidaemon_tools::testing::call_facts("sess_fetch_receipt"),
            )
            .await
            .unwrap()
        }
    };
    let url_hints = |outcome: &ToolCallOutcome| -> Vec<String> {
        outcome
            .metadata
            .semantics
            .target_hints
            .iter()
            .filter(|hint| hint.kind == ToolTargetHintKind::Url)
            .map(|hint| hint.value.clone())
            .collect()
    };

    let fetched = run(format!("curl -sSf {url}")).await;
    assert_eq!(fetched.metadata.exit_code, Some(0), "{}", fetched.output);
    assert_eq!(url_hints(&fetched), vec![url.clone()]);
    assert!(fetched.metadata.semantics.observes_state());

    let masked = run(format!("curl -sSf {url} | head -c 1")).await;
    assert_eq!(masked.metadata.exit_code, Some(0), "{}", masked.output);
    assert!(url_hints(&masked).is_empty());

    // Nothing listens on port 9 of the loopback: curl fails.
    let failed = run("curl -sSf http://127.0.0.1:9/synthetic".to_string()).await;
    assert_ne!(failed.metadata.exit_code, Some(0), "{}", failed.output);
    assert!(url_hints(&failed).is_empty());
}

#[test]
fn background_report_metadata_never_claims_launch() {
    // Completion and observation receipts describe an already-tracked
    // process; only the launch receipt may set background_started.
    let completed = background_completion_metadata(Some(0));
    assert!(!completed.background_started);
    assert!(!completed.timed_out);
    assert!(!completed.detached);
    assert_eq!(completed.outcome_status, Some(ToolOutcomeStatus::Succeeded));
    assert_eq!(
        background_completion_metadata(Some(2)).outcome_status,
        Some(ToolOutcomeStatus::CompletedWithNegativeResult)
    );

    let observed = background_observation_metadata(ToolOutcomeStatus::Backgrounded, true, false);
    assert!(!observed.background_started);
    assert!(!observed.timed_out);
    assert!(observed.detached);
}

#[tokio::test]
async fn timed_out_background_run_sets_notification_metadata_when_available() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_stores(state.clone(), state);

    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 2; echo notify-meta"}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_meta"),
        )
        .await
        .unwrap();
    assert!(outcome.output.contains("Moved to background (pid="));
    assert!(outcome.metadata.background_started);
    assert!(outcome.metadata.timed_out);
    assert!(!outcome.metadata.detached);
    assert!(outcome.metadata.completion_notifications_enabled);
}

/// A finite job the model asks to run in the background (`background`)
/// returns after the startup check as a task job that reports back and
/// hands the run off, not as a detached service the run keeps watching
/// (live 2026-11-12: a count run with `detach` kept the run going and
/// the model started a `sleep 42` job to wait for it).
#[tokio::test]
async fn background_run_hands_off_at_launch_without_detaching() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_stores(state.clone(), state);

    let started = Instant::now();
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 20; echo counted","background":true}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_background_job"),
        )
        .await
        .unwrap();
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(outcome.metadata.background_started);
    assert!(!outcome.metadata.detached);
    assert!(outcome.metadata.completion_notifications_enabled);
    assert!(outcome.metadata.background_handoff);
    assert!(outcome.finish.is_some());
    let _ = tool
        .call_with_execution_context(
            &format!(
                r#"{{"action":"kill","pid":{}}}"#,
                outcome
                    .output
                    .split("pid=")
                    .nth(1)
                    .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
                    .unwrap()
            ),
            None,
            aidaemon_tools::testing::call_facts("sess_background_job"),
        )
        .await;
}

/// A sub-agent's long command keeps its call: one loop printing a count
/// is one call that finishes, never a background job the sub-agent then
/// waits on or kills; a chat's command and any explicit background
/// launch keep their own waits.
#[test]
fn a_sub_agents_command_keeps_its_call() {
    let initial = Duration::from_secs(30);
    assert_eq!(
        foreground_wait(initial, false, false),
        SUB_AGENT_FOREGROUND_WAIT
    );
    assert!(SUB_AGENT_FOREGROUND_WAIT >= Duration::from_secs(120));
    assert_eq!(foreground_wait(initial, false, true), initial);
    assert_eq!(foreground_wait(initial, true, false), DETACH_STARTUP_GRACE);
}

/// A helper's long command shows its latest output line as it runs,
/// whether the call waits for it or it was moved to the background
/// (live 2026-11-13: "count slowly to 30" showed only its start line
/// until the end).
#[tokio::test]
async fn a_helpers_long_command_reports_its_output_as_it_runs() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let (_approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        state.pool(),
    )
    .await
    .with_stores(state.clone(), state);
    let helper = || ToolCallContext {
        grant_session: "telegram:synthetic-chat".to_string(),
        task_id: Some("helper-task".to_string()),
        ..aidaemon_tools::testing::call_facts("specialist:executor:synthetic-count")
    };
    let lines = |rx: &mut mpsc::Receiver<StatusUpdate>| {
        let mut lines = Vec::new();
        while let Ok(update) = rx.try_recv() {
            if let StatusUpdate::CommandOutput { lines: count, .. } = update {
                lines.push(count);
            }
        }
        lines
    };

    // One call that waits for the whole count.
    let (tx, mut rx) = mpsc::channel(32);
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"for i in 1 2 3 4 5; do echo $i; sleep 1; done"}"#,
            Some(tx),
            helper(),
        )
        .await
        .unwrap();
    assert!(!outcome.metadata.background_started);
    let shown = lines(&mut rx);
    assert!(
        shown.iter().any(|count| (1..=4).contains(count)),
        "{shown:?}"
    );

    // A count moved to the background keeps reporting after the call.
    let (tx, mut rx) = mpsc::channel(32);
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"for i in 1 2 3 4 5 6; do echo $i; sleep 1; done","background":true}"#,
            Some(tx),
            helper(),
        )
        .await
        .unwrap();
    assert!(outcome.metadata.background_started);
    tokio::time::sleep(Duration::from_secs(4)).await;
    let shown = lines(&mut rx);
    assert!(
        shown.iter().any(|count| (3..=5).contains(count)),
        "{shown:?}"
    );
    while tool.jobs_ended_by_task_end("helper-task").await > 0 {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// A sub-agent's job that outlives the foreground wait does not end the
/// sub-agent's run: its session has no reader, so the notifier would
/// deliver nowhere while the parent got "still on it" as the child's
/// answer (live 2026-11-12). The run keeps going and reads the result.
#[tokio::test]
async fn sub_agent_background_run_keeps_the_run() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_stores(state.clone(), state);

    let child = ToolCallContext {
        grant_session: "telegram:synthetic-chat".to_string(),
        ..aidaemon_tools::testing::call_facts("specialist:executor:synthetic-child")
    };
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 20; echo built","background":true}"#,
            None,
            child.clone(),
        )
        .await
        .unwrap();
    assert!(outcome.metadata.background_started);
    assert!(!outcome.metadata.completion_notifications_enabled);
    assert!(!outcome.metadata.background_handoff);
    assert!(outcome.finish.is_none());
    assert!(outcome.output.contains("action=\"check\""));
    let pid = outcome
        .output
        .split("pid=")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
        .unwrap()
        .to_string();
    let _ = tool
        .call_with_execution_context(&format!(r#"{{"action":"kill","pid":{pid}}}"#), None, child)
        .await;
}

/// A chat's background command is listed for the chat's running work
/// under a plain title, and stopping it reads the process back as gone.
#[tokio::test]
async fn the_chat_lists_its_background_command_and_stops_it() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_stores(state.clone(), state);

    let chat = ToolCallContext {
        task_id: Some("turn-count".to_string()),
        ..aidaemon_tools::testing::call_facts("telegram:synthetic-count")
    };
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"for i in $(seq 1 300); do sleep 2; done","background":true}"#,
            None,
            chat,
        )
        .await
        .unwrap();
    assert!(outcome.metadata.background_started);
    let jobs = tool.background_jobs("telegram:synthetic-count").await;
    assert_eq!(jobs.len(), 1, "{jobs:?}");
    assert_eq!(jobs[0].title, "Background command: a shell loop");
    assert_eq!(jobs[0].task_id.as_deref(), Some("turn-count"));
    assert!(!jobs[0].title.contains("pid"));
    assert!(
        tool.background_jobs("telegram:another-chat")
            .await
            .is_empty()
    );

    assert!(tool.stop_background_job(&jobs[0].key).await);
    assert!(
        tool.background_jobs("telegram:synthetic-count")
            .await
            .is_empty()
    );
    // Gone already: stopping it again reads as stopped.
    assert!(tool.stop_background_job(&jobs[0].key).await);
}

/// A sub-agent's job (nothing reports its result) is one its task's end
/// would stop, so the run is held while it runs; once it is killed it no
/// longer counts.
#[tokio::test]
async fn a_sub_agent_job_counts_as_ended_by_its_task_end() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_stores(state.clone(), state);

    let child = ToolCallContext {
        grant_session: "telegram:synthetic-chat".to_string(),
        task_id: Some("child-task".to_string()),
        ..aidaemon_tools::testing::call_facts("specialist:executor:synthetic-count")
    };
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 30; echo counted","background":true}"#,
            None,
            child,
        )
        .await
        .unwrap();
    assert!(!outcome.metadata.completion_notifications_enabled);
    assert!(!outcome.output.contains("No completion notification"));
    // The run wakes the sub-agent when the job ends: it must not poll.
    assert!(
        outcome
            .output
            .contains("the run holds it until this command ends")
    );
    assert!(outcome.output.contains("no loop polling its pid"));
    assert_eq!(tool.jobs_ended_by_task_end("child-task").await, 1);
    assert_eq!(tool.jobs_ended_by_task_end("other-task").await, 0);
    let jobs = tool
        .background_jobs("specialist:executor:synthetic-count")
        .await;
    assert_eq!(jobs.len(), 1);
    assert!(tool.stop_background_job(&jobs[0].key).await);
    assert_eq!(tool.jobs_ended_by_task_end("child-task").await, 0);
}

#[test]
fn notifications_reach_the_chat_or_a_goal_session_only() {
    let chat = aidaemon_tools::testing::call_facts("telegram:synthetic-chat");
    assert!(notifications_reach(&chat, "telegram:synthetic-chat"));
    let child = ToolCallContext {
        grant_session: "telegram:synthetic-chat".to_string(),
        ..aidaemon_tools::testing::call_facts("specialist:executor:synthetic-child")
    };
    assert!(!notifications_reach(
        &child,
        "specialist:executor:synthetic-child"
    ));
    // A child working for a goal notifies the goal's origin session.
    assert!(notifications_reach(
        &child,
        "telegram:synthetic-goal-origin"
    ));
}

/// A background job hands the run off only when something will report
/// back: without a notifier the task end would kill the job and lose its
/// output, so the run keeps going and the output says how to read the
/// result instead of promising a notification.
#[tokio::test]
async fn background_run_without_notifier_keeps_the_run() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    // No state queue and no hub: no completion notifier can start.
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        state.pool(),
    )
    .await;

    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 20; echo counted","background":true}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_background_no_notifier"),
        )
        .await
        .unwrap();
    assert!(outcome.metadata.background_started);
    assert!(!outcome.metadata.completion_notifications_enabled);
    assert!(!outcome.metadata.background_handoff);
    assert!(outcome.finish.is_none());
    assert!(!outcome.output.contains("do NOT wait or repeatedly check"));
    assert!(outcome.output.contains("action=\"check\""));
    let pid = outcome
        .output
        .split("pid=")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
        .unwrap()
        .to_string();
    let _ = tool
        .call_with_execution_context(
            &format!(r#"{{"action":"kill","pid":{pid}}}"#),
            None,
            aidaemon_tools::testing::call_facts("sess_background_no_notifier"),
        )
        .await;
}

/// A service that logs its URL on stderr (cloudflared does) must show it
/// in the launch result, and the session's next turn must see the running
/// process so a follow-up checks it rather than relaunching it.
#[tokio::test]
async fn detached_service_shows_stderr_startup_and_is_listed_for_its_session() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await;

    let outcome = tool
        .call_with_execution_context(r#"{"action":"run","command":"echo 'ready at https://synthetic-tunnel.example.test' >&2; sleep 30","detach":true}"#, None, aidaemon_tools::testing::call_facts("sess_service"))
        .await
        .unwrap();
    assert!(outcome.metadata.detached);
    assert!(
        outcome
            .output
            .contains("ready at https://synthetic-tunnel.example.test"),
        "{}",
        outcome.output
    );
    assert!(
        !outcome.output.contains("do NOT wait or repeatedly check"),
        "{}",
        outcome.output
    );

    let note = tool
        .session_context_note("sess_service")
        .await
        .expect("a running detached process is listed for its session");
    assert!(note.contains("detached, runs until killed"), "{note}");
    assert!(note.contains("sleep 30"), "{note}");
    assert!(tool.session_context_note("sess_other").await.is_none());

    let pid = tool
        .running
        .lock()
        .await
        .keys()
        .copied()
        .next()
        .expect("tracked pid");
    tool.call_with_execution_context(
        &format!(r#"{{"action":"kill","pid":{pid}}}"#),
        None,
        aidaemon_tools::testing::call_facts("sess_service"),
    )
    .await
    .unwrap();
    assert!(tool.session_context_note("sess_service").await.is_none());
}

#[tokio::test]
async fn detached_run_launches_and_returns_after_startup_grace() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    // A 30s foreground window: without launch-and-return, `sleep 5`
    // would run to completion and be reported synchronously.
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await;

    let started = Instant::now();
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 5","detach":true}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_detach"),
        )
        .await
        .unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "detach must not wait for the command: {:?}",
        started.elapsed()
    );
    assert!(
        outcome.output.contains("Detached launch"),
        "{}",
        outcome.output
    );
    assert!(outcome.metadata.background_started);
    assert!(outcome.metadata.detached);
    assert!(!outcome.metadata.timed_out);
    assert_eq!(
        outcome.metadata.outcome_status,
        Some(ToolOutcomeStatus::Backgrounded)
    );
    assert!(
        outcome
            .metadata
            .semantics
            .mutation_effects
            .satisfies(ToolMutationEffects::PROCESS_STATE)
    );

    // An immediate failure inside the startup check is still synchronous.
    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"exit 3","detach":true}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_detach"),
        )
        .await
        .unwrap();
    assert!(!outcome.metadata.background_started);
    assert_eq!(outcome.metadata.exit_code, Some(3));
}

#[tokio::test]
async fn checking_finished_background_process_returns_completion_receipt() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await;

    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 2; echo done-marker"}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_check_done"),
        )
        .await
        .unwrap();
    let pid = extract_pid_from_background_message(&outcome.output);

    // Poll until the process has finished and the check returns the
    // terminal result.
    let mut finished = None;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let check = tool.handle_check(pid).await.unwrap();
        if check.output.contains("finished after") {
            finished = Some(check);
            break;
        }
    }
    let finished = finished.expect("background process should finish");

    // A receipt for a FINISHED process is a completion receipt, not a
    // launch receipt: it must never re-trigger the background-detach
    // handoff that ends the task with "Still on it".
    assert!(
        !finished.metadata.background_started,
        "finished-process check receipt must not claim background_started"
    );
    assert!(!finished.metadata.timed_out);
    assert_eq!(finished.metadata.exit_code, Some(0));
    assert_eq!(
        finished.metadata.outcome_status,
        Some(ToolOutcomeStatus::Succeeded)
    );
    assert!(finished.output.contains("done-marker"));
}

#[tokio::test]
async fn checking_reaped_background_process_returns_completion_receipt() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await;

    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 2; echo reap-marker"}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_reap_done"),
        )
        .await
        .unwrap();
    let pid = extract_pid_from_background_message(&outcome.output);

    // Wait for process exit, then force the reap path so the completed-map
    // receipt (the one served to a later check) is the one under test.
    let mut reaped = false;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        tool.reap_finished().await;
        if !tool.running.lock().await.contains_key(&pid) {
            reaped = true;
            break;
        }
    }
    assert!(reaped, "background process should be reaped");

    let check = tool.handle_check(pid).await.unwrap();
    assert!(check.output.contains("finished after"), "{}", check.output);
    assert!(
        !check.metadata.background_started,
        "reaped-process receipt must not claim background_started"
    );
    assert!(!check.metadata.timed_out);
    assert_eq!(check.metadata.exit_code, Some(0));
    assert_eq!(
        check.metadata.outcome_status,
        Some(ToolOutcomeStatus::Succeeded)
    );
}

#[tokio::test]
async fn still_running_check_and_kill_receipts_are_not_background_launches() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await;

    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 30"}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_check_kill"),
        )
        .await
        .unwrap();
    let pid = extract_pid_from_background_message(&outcome.output);

    // Observing a still-running process is a finished observation, not
    // a fresh detach or a second in-flight launch: only the launch
    // receipt is provisional, and the check changes nothing.
    let check = tool.handle_check(pid).await.unwrap();
    assert!(check.output.contains("still running"), "{}", check.output);
    assert!(!check.metadata.background_started);
    assert!(!check.metadata.timed_out);
    assert_eq!(
        check.metadata.outcome_status,
        Some(ToolOutcomeStatus::Succeeded)
    );
    assert!(!check.metadata.semantics.mutates_state());

    let kill = tool.handle_kill(pid).await.unwrap();
    assert!(
        !kill.metadata.background_started,
        "kill receipt must not claim background_started"
    );
    assert!(!kill.metadata.timed_out);
    assert_eq!(
        kill.metadata.outcome_status,
        Some(ToolOutcomeStatus::Succeeded)
    );
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn confined_command_can_create_its_own_declared_write_root() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    let root = std::env::temp_dir().join(format!("aid-own-root-{}", uuid::Uuid::new_v4()));
    let root_str = root.to_string_lossy().to_string();
    // The declared write-root grant itself authorizes creating the root:
    // a plain `mkdir <root>` (the canonical first step of a "create a
    // workspace" request) must succeed, not collide with a runtime
    // pre-created directory and fail with EEXIST.
    let args = serde_json::json!({
        "action": "run",
        "command": format!("mkdir {}", root_str),
        "write_roots": [root_str],
    });
    let outcome = tool
        .call_with_execution_context(
            &args.to_string(),
            None,
            aidaemon_tools::testing::call_facts("sess_own_root"),
        )
        .await
        .unwrap();
    assert!(
        !outcome.output.contains("File exists"),
        "runtime must not pre-create the command's own target: {}",
        outcome.output
    );
    assert_eq!(outcome.metadata.exit_code, Some(0), "{}", outcome.output);
    assert!(
        root.is_dir(),
        "command should have created its declared root"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn future_write_root_used_as_working_dir_is_prepared_for_spawn() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        30,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    let root = std::env::temp_dir().join(format!("aid-cwd-root-{}", uuid::Uuid::new_v4()));
    let root_str = root.to_string_lossy().to_string();
    // A declared future root that is also the working directory must
    // exist before exec; this is the one preparation the runtime still
    // owns.
    let args = serde_json::json!({
        "action": "run",
        "command": "pwd",
        "working_dir": root_str,
        "write_roots": [root_str],
    });
    let outcome = tool
        .call_with_execution_context(
            &args.to_string(),
            None,
            aidaemon_tools::testing::call_facts("sess_cwd_root"),
        )
        .await
        .unwrap();
    assert_eq!(outcome.metadata.exit_code, Some(0), "{}", outcome.output);
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn timed_out_background_run_clears_notification_metadata_when_unavailable() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let state = Arc::new(work_store(&db_path).await);
    let pool = state.pool();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    let outcome = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 2; echo no-notify-meta"}"#,
            None,
            aidaemon_tools::testing::call_facts("sess_meta2"),
        )
        .await
        .unwrap();
    assert!(outcome.output.contains("Moved to background (pid="));
    assert!(outcome.metadata.background_started);
    assert!(outcome.metadata.timed_out);
    assert!(!outcome.metadata.detached);
    assert!(!outcome.metadata.completion_notifications_enabled);
}
