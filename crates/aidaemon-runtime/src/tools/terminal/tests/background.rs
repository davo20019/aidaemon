//! Background lifecycle: notifications, cards, reaping, task-end cleanup.

use super::*;

#[tokio::test]
async fn test_daemonization_requires_owner_role() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        1000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    let response = tool
        .call_with(
            r#"{"action":"run","command":"nohup sleep 1 &"}"#,
            aidaemon_tools::testing::call_facts("s1"),
        )
        .await
        .unwrap();
    assert!(response.contains("only owners can approve"));
}

#[tokio::test]
async fn test_terminal_blocks_unscoped_recursive_grep() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        1000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    let response = tool
        .call_with(
            r#"{"action":"run","command":"grep -rc \"async fn\" ."}"#,
            aidaemon_tools::testing::call_facts("s1"),
        )
        .await
        .unwrap();
    assert!(response.contains("Blocked: broad recursive `grep`"));
    assert!(response.contains("search_files"));
    assert!(response.contains("rg -n --glob"));
}

#[tokio::test]
async fn test_background_terminal_completion_enqueues_notification() {
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
    .with_stores(state.clone(), state.clone());

    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 2; echo terminal-notify-ok"}"#,
            aidaemon_tools::testing::call_facts("sess_notify"),
        )
        .await
        .unwrap();
    assert!(response.contains("Moved to background (pid="));

    let mut found = false;
    for _ in 0..40 {
        let pending = state.get_pending_notifications(20).await.unwrap();
        if pending.iter().any(|entry| {
            entry.session_id == "sess_notify"
                && entry.notification_type == "progress"
                && entry.message.contains("terminal-notify-ok")
        }) {
            found = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    assert!(
        found,
        "expected background completion notification to be enqueued"
    );
}

/// A command moved to the background is a run on its turn's one card:
/// the turn's reply names it, its latest output line is its step, it
/// settles to its final header when it ends, and the result follows in
/// its own message without the header (live 2026-11-13: "Still on it …
/// Request: <the whole prompt>", then "Preparing your result", removed
/// with no final header).
#[tokio::test]
async fn a_background_command_shows_one_card_that_settles() {
    let chat = aidaemon_channels::testing::CardChat::new("telegram:synthetic-card").await;
    let tool = card_chat_terminal(&chat).await;
    let outcome = chat
        .turn(
            |tx| {
                tool.call_with_execution_context(
                    r#"{"action":"run","command":"echo step-one; sleep 5; echo all-done","description":"counting script","background":true}"#,
                    Some(tx),
                    aidaemon_tools::testing::call_facts("telegram:synthetic-card"),
                )
            },
            |outcome| {
                outcome
                    .as_ref()
                    .map(|outcome| outcome.output.lines().next().unwrap_or("").to_string())
                    .unwrap_or_default()
            },
        )
        .await
        .unwrap();
    assert!(outcome.metadata.background_handoff);
    assert!(
        outcome
            .output
            .starts_with("Started in the background: counting script.")
    );
    assert!(
        chat.wait_for(Duration::from_secs(10), |text| text
            .starts_with("⏳ Running in the background: counting script")
            && text.contains("1 line of output so far"))
            .await,
        "{:#?}",
        chat.shown().await
    );
    assert!(
        chat.wait_for(Duration::from_secs(20), |text| text.contains("all-done")
            && !text.contains("in the background:"))
            .await,
        "{:#?}",
        chat.shown().await
    );
    chat.assert_one_settled_card("✅ Finished in the background: counting script", "all-done")
        .await;
}

/// Live 2026-11-13: a helper's card showed `drush site:install`'s
/// "User password: …" line as its "last output". A command whose output
/// carries a password shows no line of it on its card, and none of the
/// chat's messages carries the value.
#[tokio::test]
async fn a_password_a_command_prints_never_reaches_the_chat() {
    let chat = aidaemon_channels::testing::CardChat::new("telegram:synthetic-secret").await;
    let tool = card_chat_terminal(&chat).await;
    chat.turn(
        |tx| {
            tool.call_with_execution_context(
                r#"{"action":"run","command":"echo '[success] User name: admin  User password: Card-Synthetic-73'; sleep 5; echo installed","description":"site install","background":true}"#,
                Some(tx),
                aidaemon_tools::testing::call_facts("telegram:synthetic-secret"),
            )
        },
        |_| String::new(),
    )
    .await
    .unwrap();
    assert!(
        chat.wait_for(Duration::from_secs(10), |text| text
            .contains("1 line of output so far"))
            .await,
        "{:#?}",
        chat.shown().await
    );
    assert!(
        chat.wait_for(Duration::from_secs(20), |text| text
            .starts_with("✅ Finished in the background"))
            .await,
        "{:#?}",
        chat.shown().await
    );
    for text in chat.shown().await {
        assert!(!text.contains("Card-Synthetic-73"), "{text}");
    }
}

/// A command that prints nothing has no step on its card, and its end
/// settles the card with nothing else to say: no message of its own.
#[tokio::test]
async fn a_quiet_background_command_settles_its_card_and_says_nothing() {
    let chat = aidaemon_channels::testing::CardChat::new("telegram:synthetic-quiet").await;
    let tool = card_chat_terminal(&chat).await;
    chat.turn(
        |tx| {
            tool.call_with_execution_context(
                r#"{"action":"run","command":"sleep 3","background":true}"#,
                Some(tx),
                aidaemon_tools::testing::call_facts("telegram:synthetic-quiet"),
            )
        },
        |_| String::new(),
    )
    .await
    .unwrap();
    assert!(
        chat.wait_for(Duration::from_secs(15), |text| text
            .starts_with("✅ Finished in the background"))
            .await,
        "{:#?}",
        chat.shown().await
    );
    let shown = chat.shown().await;
    assert_eq!(shown.len(), 1, "only the card: {shown:#?}");
    assert!(!shown[0].contains("of output so far"), "{shown:#?}");
}

#[test]
fn test_should_idle_reap_policy() {
    let stall = Duration::from_secs(120);
    let max_runtime = Duration::from_secs(1200);
    let no = Duration::ZERO;

    // No progress past the stall threshold (well under max runtime) → reap.
    assert!(should_idle_reap(
        true,
        false,
        Duration::from_secs(121),
        Duration::from_secs(200),
        stall,
        max_runtime,
        true,
    ));
    // Exactly at the stall threshold → reap (>=).
    assert!(should_idle_reap(
        true,
        false,
        Duration::from_secs(120),
        Duration::from_secs(200),
        stall,
        max_runtime,
        true,
    ));
    // Progress recent (no_progress below stall) AND under max runtime → keep.
    assert!(!should_idle_reap(
        true,
        false,
        Duration::from_secs(119),
        Duration::from_secs(200),
        stall,
        max_runtime,
        true,
    ));
    // Busy-loop: progress recent (no_progress=0) but total runtime hit the
    // max-runtime backstop → reap.
    assert!(should_idle_reap(
        true,
        false,
        no,
        Duration::from_secs(1200),
        stall,
        max_runtime,
        true,
    ));
    assert!(should_idle_reap(
        true,
        false,
        no,
        Duration::from_secs(5000),
        stall,
        max_runtime,
        true,
    ));
    // Context-recognized long work treats max runtime as a soft boundary:
    // elapsed wall time alone is not a reason to kill an active command.
    assert!(!should_idle_reap(
        true,
        false,
        no,
        Duration::from_secs(5000),
        stall,
        max_runtime,
        false,
    ));
    // Detached (dev server) → never reaped, even when long idle / long-running.
    assert!(!should_idle_reap(
        true,
        true,
        Duration::from_secs(100_000),
        Duration::from_secs(100_000),
        stall,
        max_runtime,
        true,
    ));
    // Not notifier-active (task-owned, no promise to deliver) → not reaped here.
    assert!(!should_idle_reap(
        false,
        false,
        Duration::from_secs(100_000),
        Duration::from_secs(100_000),
        stall,
        max_runtime,
        true,
    ));
}

#[test]
fn test_progress_contract_is_structural_and_context_aware() {
    let broad_scan = progress_contract_for_command(
        "du -sh /Users/synthetic/{Library,Documents,projects} 2>/dev/null",
    );
    assert_eq!(
        broad_scan.workload,
        BackgroundWorkload::FilesystemTraversal {
            broad_or_multi_target: true
        }
    );
    assert_eq!(broad_scan.stall_multiplier, 5);
    assert!(!broad_scan.hard_max_runtime);

    let bounded_scan = progress_contract_for_command("find /tmp/synthetic -type f -size +1G");
    assert_eq!(
        bounded_scan.workload,
        BackgroundWorkload::FilesystemTraversal {
            broad_or_multi_target: false
        }
    );
    assert_eq!(bounded_scan.stall_multiplier, 3);

    let build = progress_contract_for_command("cargo test --all-features");
    assert_eq!(build.workload, BackgroundWorkload::BuildOrTest);
    assert!(!build.hard_max_runtime);

    let prose_only = progress_contract_for_command("printf 'find du cargo'");
    assert_eq!(prose_only, BackgroundProgressContract::default());
}

#[test]
fn test_process_made_progress_any_signal_grows() {
    let flat = ProcessResourceSample {
        cpu_ms: 100,
        io_bytes: 2_000,
        tree_fingerprint: 7,
        process_count: 1,
        runnable_count: 0,
    };
    let previous = |cpu_ms, io_bytes, tree_fingerprint| ProcessResourceSample {
        cpu_ms,
        io_bytes,
        tree_fingerprint,
        process_count: 1,
        runnable_count: 0,
    };
    // CPU advanced (silent busy scan statting files) → progress.
    assert!(process_made_progress(
        previous(100, 0, 7),
        0,
        0,
        ProcessResourceSample {
            cpu_ms: 150,
            ..flat
        }
    ));
    // Disk I/O advanced (silent scan reading directory entries) → progress.
    assert!(process_made_progress(
        previous(0, 1_000, 7),
        0,
        0,
        ProcessResourceSample { cpu_ms: 0, ..flat }
    ));
    // Output grew (streaming) → progress.
    assert!(process_made_progress(previous(0, 0, 7), 10, 25, flat));
    // Process-tree churn is additional evidence.
    assert!(process_made_progress(
        previous(100, 2_000, 7),
        25,
        25,
        ProcessResourceSample {
            tree_fingerprint: 8,
            ..flat
        }
    ));
    // A point-in-time runnable state is useful telemetry, but not proof of
    // progress: macOS can report a sleeping child this way between sweeps.
    assert!(!process_made_progress(
        previous(100, 2_000, 7),
        25,
        25,
        ProcessResourceSample {
            runnable_count: 1,
            ..flat
        }
    ));
    // Nothing advanced (truly stalled) → no progress.
    assert!(!process_made_progress(
        previous(100, 2_000, 7),
        25,
        25,
        flat
    ));
    // A carried-forward (equal) signal alone is not progress; any OTHER
    // advancing signal still wins.
    let zero = ProcessResourceSample {
        cpu_ms: 100,
        io_bytes: 0,
        tree_fingerprint: 7,
        process_count: 1,
        runnable_count: 0,
    };
    assert!(!process_made_progress(previous(100, 0, 7), 0, 0, zero));
    assert!(process_made_progress(previous(100, 0, 7), 0, 1, zero));
}

#[test]
fn test_sum_subtree_resources_includes_children() {
    // sh wrapper (100) is idle; its du child (101) is churning, with a
    // grandchild (102). The busy descendants must count toward the tracked
    // wrapper pid so a working pipeline is not false-reaped.
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    children.insert(100, vec![101]);
    children.insert(101, vec![102]);
    let mut per_pid: HashMap<u32, (u64, u64, bool)> = HashMap::new();
    per_pid.insert(100, (0, 0, false));
    per_pid.insert(101, (5_000, 2_000_000, true));
    per_pid.insert(102, (10, 50, false));

    let out = sum_subtree_resources(&[100], &children, &per_pid);
    let sample = out.get(&100).expect("root sample");
    assert_eq!(sample.cpu_ms, 5_010);
    assert_eq!(sample.io_bytes, 2_000_050);
    assert_eq!(sample.process_count, 3);
    assert_eq!(sample.runnable_count, 1);
}

#[test]
fn test_sum_subtree_resources_isolated_root() {
    // A root with no children sums to just its own values.
    let children: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut per_pid: HashMap<u32, (u64, u64, bool)> = HashMap::new();
    per_pid.insert(42, (123, 456, false));

    let out = sum_subtree_resources(&[42], &children, &per_pid);
    let sample = out.get(&42).expect("root sample");
    assert_eq!((sample.cpu_ms, sample.io_bytes), (123, 456));
}

#[test]
fn test_sum_subtree_resources_handles_cycle_safely() {
    // A malformed parent->child map containing a cycle (100->101->100)
    // must not infinite-loop; the visited set bounds the traversal and each
    // node is counted at most once.
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    children.insert(100, vec![101]);
    children.insert(101, vec![100]); // cycle back to root
    let mut per_pid: HashMap<u32, (u64, u64, bool)> = HashMap::new();
    per_pid.insert(100, (1, 2, false));
    per_pid.insert(101, (3, 4, false));

    let out = sum_subtree_resources(&[100], &children, &per_pid);
    let sample = out.get(&100).expect("root sample");
    assert_eq!((sample.cpu_ms, sample.io_bytes), (4, 6));
}

#[test]
fn test_sum_subtree_missing_pid() {
    // A root absent from per_pid (process exited between snapshot and
    // lookup) contributes nothing → no map entry. The caller carries
    // forward the previous sample, so an absent entry is safe.
    let children: HashMap<u32, Vec<u32>> = HashMap::new();
    let per_pid: HashMap<u32, (u64, u64, bool)> = HashMap::new();

    let out = sum_subtree_resources(&[999], &children, &per_pid);
    assert_eq!(out.get(&999), None);
    assert!(out.is_empty());
}

/// A disowned, no-output background command (the `du -ah ~` failure mode) is
/// stopped by the idle reaper, removed from tracking, and the user is told.
#[tokio::test]
async fn test_idle_reap_stops_hung_background_process() {
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
    .with_stores(state.clone(), state.clone());

    // No-output, long-running command — stands in for a whole-disk scan that
    // never exits. It is moved to background after the 1s initial timeout.
    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 120"}"#,
            aidaemon_tools::testing::call_facts("sess_reap"),
        )
        .await
        .unwrap();
    assert!(response.contains("Moved to background (pid="));

    // Threshold near-zero: the process produced no output, so the reaper sees
    // it as immediately idle and stops it.
    let mut reaped = 0;
    for _ in 0..40 {
        reaped = tool
            .reap_stale_background_processes(Duration::from_millis(1))
            .await;
        if reaped > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        reaped, 1,
        "expected the hung background process to be reaped"
    );

    // It must be gone from the tracking map (no leak).
    assert!(
        tool.running.lock().await.is_empty(),
        "reaped process should be removed from the running map"
    );

    // The user must be told why it stopped (the screenshot's failure was the
    // bot waiting forever in silence).
    let mut saw_notice = false;
    for _ in 0..20 {
        let pending = state.get_pending_notifications(50).await.unwrap();
        if pending.iter().any(|entry| {
            entry.session_id == "sess_reap"
                && entry.message.contains("stopped a background command")
        }) {
            saw_notice = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        saw_notice,
        "expected a user-facing notice that the hung command was stopped"
    );
}

/// A background command that keeps streaming output is NOT reaped: each new
/// byte refreshes its idle clock, so a healthy long-running process survives.
#[tokio::test]
async fn test_idle_reap_spares_streaming_process() {
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
    .with_stores(state.clone(), state.clone());

    // Emits a line roughly every 0.2s for a while — a stand-in for a dev server
    // or a scan that is genuinely making progress.
    let response = tool
        .call_with(r#"{"action":"run","command":"for i in $(seq 1 50); do echo line-$i; sleep 0.2; done"}"#, aidaemon_tools::testing::call_facts("sess_stream"))
        .await
        .unwrap();
    assert!(response.contains("Moved to background (pid="));

    // Prime the baseline, then sweep again after fresh output has arrived. With
    // a 1s threshold and ~0.2s output cadence, the idle clock keeps resetting,
    // so nothing is reaped.
    let _ = tool
        .reap_stale_background_processes(Duration::from_secs(1))
        .await;
    tokio::time::sleep(Duration::from_millis(600)).await;
    let reaped = tool
        .reap_stale_background_processes(Duration::from_secs(1))
        .await;
    assert_eq!(reaped, 0, "a streaming process must not be idle-reaped");

    // Clean up the still-running process.
    let pids: Vec<u32> = tool.running.lock().await.keys().copied().collect();
    for pid in pids {
        let _ = tool.handle_kill(pid).await;
    }
}

#[test]
fn test_is_short_complete_output_classification() {
    // Short, self-contained results → delivered directly.
    assert!(is_short_complete_output("42"));
    assert!(is_short_complete_output("207"));
    assert!(is_short_complete_output(
        "/Users/jordan/projects/resume/google"
    ));
    assert!(is_short_complete_output("Build complete"));
    assert!(is_short_complete_output("a\nb\nc")); // 3 lines, tiny

    // Long / multi-line output → still routed through re-engagement.
    let long_line = "x".repeat(SHORT_OUTPUT_DIRECT_DELIVERY_MAX_CHARS + 1);
    assert!(!is_short_complete_output(&long_line));
    let many_lines = "l\n".repeat(SHORT_OUTPUT_DIRECT_DELIVERY_MAX_LINES + 1);
    assert!(!is_short_complete_output(&many_lines));
}

#[test]
fn test_format_short_background_result_message() {
    // One-liner → inline code, no pid, no raw command.
    let one = format_short_background_result("207");
    assert_eq!(one, "Result: `207`");
    assert!(!one.contains("pid"));
    assert!(!one.contains("find"));

    // Multi-line → fenced block.
    let multi = format_short_background_result("a\nb");
    assert!(multi.contains("```"));
    assert!(multi.contains("a\nb"));
}

/// A background command whose answer is a short string (a `wc -l` count, a
/// numeric total, a one-word status) must still deliver that answer to the
/// user. Regression test for the bug where outputs under 5 chars were
/// classified as "trivial" and silently dropped — the user asked "how many
/// resumes?", the count came back short, and only the bare "finished" ping
/// reached the chat with no number. The short result is delivered directly
/// (as `Result: ...`), never re-fed into the agent loop.
#[tokio::test]
async fn test_background_terminal_short_output_is_delivered() {
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
    .with_stores(state.clone(), state.clone());

    // Output "42" (2 chars) — the command text contains 40 and 2 but not 42,
    // so finding "42" in a notification proves the OUTPUT was delivered.
    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 2; echo $((40 + 2))"}"#,
            aidaemon_tools::testing::call_facts("sess_short"),
        )
        .await
        .unwrap();
    assert!(response.contains("Moved to background (pid="));

    let mut saw_output = false;
    for _ in 0..60 {
        let pending = state.get_pending_notifications(50).await.unwrap();
        if pending.iter().any(|entry| {
            entry.session_id == "sess_short"
                && entry.notification_type == "progress"
                && entry.message.contains("42")
                // Delivered directly as a short result, NOT via the
                // re-engagement verbatim fallback (which exposes the raw
                // command) and NOT dropped as trivial.
                && entry.message.contains("Result:")
                && !entry.message.contains("Output from")
        }) {
            saw_output = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        saw_output,
        "short background command output (the count the user asked for) must be delivered directly as a short result, not dropped or re-engaged"
    );
}

#[tokio::test]
async fn test_background_terminal_duplicate_result_is_suppressed() {
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
    .with_stores(state.clone(), state.clone());

    let first = tool
        .call_with(
            r#"{"action":"run","command":"sleep 2; echo duplicate-result"}"#,
            aidaemon_tools::testing::call_facts("sess_dupe_result"),
        )
        .await
        .unwrap();
    assert!(first.contains("Moved to background (pid="));

    let second = tool
        .call_with(
            r#"{"action":"run","command":"sleep 2; printf '%s\n' duplicate-result"}"#,
            aidaemon_tools::testing::call_facts("sess_dupe_result"),
        )
        .await
        .unwrap();
    assert!(second.contains("Moved to background (pid="));

    for _ in 0..60 {
        let pending = state.get_pending_notifications(50).await.unwrap();
        let result_count = pending
            .iter()
            .filter(|entry| {
                entry.session_id == "sess_dupe_result"
                    && entry.notification_type == "progress"
                    && entry.message.contains("Result: `duplicate-result`")
            })
            .count();
        if result_count >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let pending = state.get_pending_notifications(50).await.unwrap();
    let result_count = pending
        .iter()
        .filter(|entry| {
            entry.session_id == "sess_dupe_result"
                && entry.notification_type == "progress"
                && entry.message.contains("Result: `duplicate-result`")
        })
        .count();
    assert_eq!(
        result_count,
        1,
        "duplicate background completions should not deliver the same result twice; messages={:?}",
        pending
            .iter()
            .filter(|entry| entry.session_id == "sess_dupe_result")
            .map(|entry| entry.message.as_str())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_background_terminal_kill_suppresses_completion_notification() {
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
    .with_stores(state.clone(), state.clone());

    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 10"}"#,
            aidaemon_tools::testing::call_facts("sess_kill"),
        )
        .await
        .unwrap();
    let pid = extract_pid_from_background_message(&response);

    let kill_response = tool
        .call_with(
            &format!(r#"{{"action":"kill","pid":{}}}"#, pid),
            aidaemon_tools::testing::call_facts("sess_kill"),
        )
        .await
        .unwrap();
    assert!(kill_response.contains("stopped"));

    tokio::time::sleep(Duration::from_millis(500)).await;
    let pending = state.get_pending_notifications(20).await.unwrap();
    assert!(
        !pending.iter().any(|entry| {
            entry.session_id == "sess_kill"
                && entry.notification_type == "progress"
                && entry.message.contains("Background command finished")
        }),
        "kill action should suppress background completion notification"
    );
}

#[tokio::test]
async fn test_background_terminal_check_returns_result_after_reap() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
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

    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 2; echo post-reap-ok"}"#,
            aidaemon_tools::testing::call_facts("s1"),
        )
        .await
        .unwrap();
    let pid = extract_pid_from_background_message(&response);
    tokio::time::sleep(Duration::from_secs(3)).await;

    // This call will reap finished processes first; check must still return final output.
    let check = tool
        .call_with(
            &format!(r#"{{"action":"check","pid":{}}}"#, pid),
            aidaemon_tools::testing::call_facts("s1"),
        )
        .await
        .unwrap();
    assert!(check.contains("post-reap-ok"));
    assert!(check.contains(&format!("pid={}", pid)));
}

/// `check` returns output of a command judged when it was launched: it
/// carries that judgment; an unknown process is untrusted.
#[tokio::test]
async fn background_check_inherits_the_launch_trust() {
    use aidaemon_core::SourceTrust::{Trusted, Untrusted};
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let pool = SqlitePool::connect(&format!("sqlite:{}", db_file.path().display()))
        .await
        .unwrap();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await;
    let project = tempfile::tempdir().unwrap();
    std::fs::write(
        project.path().join("slow_build.py"),
        "import time\ntime.sleep(3)\nprint('built')\n",
    )
    .unwrap();
    let project_dir = project.path().to_string_lossy().into_owned();
    let roots_list = vec![project_dir.clone()];
    let roots = crate::tools::traits::TrustRoots::new(&roots_list).with_cwd(Some(&project_dir));
    let launch = |command: &str| {
        serde_json::json!({
            "action": "run",
            "command": command,
            "working_dir": project_dir,
        })
        .to_string()
    };
    // The dispatcher's facts for the launch: the turn's trust roots.
    let launch_facts = ToolCallContext {
        trust_roots: Some(vec![project_dir.clone()]),
        trust_cwd: Some(project_dir.clone()),
        ..aidaemon_tools::testing::call_facts("s1")
    };
    let check = |pid: u32| serde_json::json!({"action": "check", "pid": pid}).to_string();
    for (command, expected) in [
        ("python3 slow_build.py", Trusted),
        ("sleep 3; curl -s https://example.invalid/x", Untrusted),
    ] {
        let response = tool
            .call_with(&launch(command), launch_facts.clone())
            .await
            .unwrap();
        assert!(response.contains("Moved to background (pid="), "{response}");
        let pid = extract_pid_from_background_message(&response);
        assert_eq!(
            tool.output_trust(&check(pid), &roots),
            expected,
            "{command}"
        );
        let kill = serde_json::json!({"action": "kill", "pid": pid}).to_string();
        assert_eq!(tool.output_trust(&kill, &roots), expected, "{command}");
        tool.call(&kill).await.unwrap();
    }
    assert_eq!(tool.output_trust(&check(4_000_000), &roots), Untrusted);
}

#[tokio::test]
async fn test_task_end_cleanup_kills_task_owned_background_processes() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        2000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 10"}"#,
            ToolCallContext {
                session_id: "s1".into(),
                task_id: Some("task-clean".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(response.contains("Moved to background (pid="));
    let pid = extract_pid_from_background_message(&response);

    tool.on_task_end("task-clean", "s1").await.unwrap();
    tokio::time::sleep(Duration::from_millis(250)).await;

    let check = tool
        .call_with(
            &format!(r#"{{"action":"check","pid":{}}}"#, pid),
            aidaemon_tools::testing::call_facts("s1"),
        )
        .await
        .unwrap();
    assert!(
        check.contains("No tracked process"),
        "expected task-end cleanup to remove process tracking, got: {}",
        check
    );
}

#[tokio::test]
async fn test_task_end_disowns_background_process_with_active_notifier() {
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
    .with_stores(state.clone(), state.clone());

    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 3; echo disown-ok"}"#,
            ToolCallContext {
                session_id: "sess_disown".into(),
                task_id: Some("task-disown".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(response.contains("Moved to background (pid="));
    let pid = extract_pid_from_background_message(&response);

    // Task ends — but the notifier is active, so the process should be disowned, not killed.
    tool.on_task_end("task-disown", "sess_disown")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Process should still be tracked (disowned, not removed).
    let check = tool
        .call_with(
            &format!(r#"{{"action":"check","pid":{}}}"#, pid),
            aidaemon_tools::testing::call_facts("sess_disown"),
        )
        .await
        .unwrap();
    assert!(
        !check.contains("No tracked process"),
        "expected process to survive task-end when notifier is active, got: {}",
        check
    );

    // Wait for the process to complete and the notification to be enqueued.
    let mut found = false;
    for _ in 0..50 {
        let pending = state.get_pending_notifications(20).await.unwrap();
        if pending
            .iter()
            .any(|entry| entry.session_id == "sess_disown" && entry.message.contains("disown-ok"))
        {
            found = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    assert!(
        found,
        "expected background completion notification after task-end disown"
    );
}

#[tokio::test]
async fn test_duplicate_background_run_is_suppressed_within_goal_scope() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        2000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    let first = tool
        .call_with(
            r#"{"action":"run","command":"sleep 5"}"#,
            ToolCallContext {
                session_id: "sub-a".into(),
                task_id: Some("task-a".into()),
                goal_id: Some("goal-1".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let pid = extract_pid_from_background_message(&first);

    let second = tool
        .call_with(
            r#"{"action":"run","command":"sleep   5"}"#,
            ToolCallContext {
                session_id: "sub-b".into(),
                task_id: Some("task-b".into()),
                goal_id: Some("goal-1".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(
        second.contains("Equivalent command is already running"),
        "expected duplicate suppression, got: {}",
        second
    );
    assert!(
        second.contains(&format!("pid={}", pid)),
        "expected duplicate response to reference original pid {}, got: {}",
        pid,
        second
    );

    tool.on_task_end("task-a", "sub-a").await.unwrap();
}

#[tokio::test]
async fn test_detached_background_process_survives_task_end_cleanup() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let (approval_tx_raw, mut approval_rx) = mpsc::channel::<ApprovalRequest>(8);
    let _approval_tx = crate::approvals::broker::ApprovalBroker::new(approval_tx_raw);
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        2000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    tokio::spawn(async move {
        while let Some(req) = approval_rx.recv().await {
            let _ = req.response_tx.send(ApprovalResponse::AllowOnce);
        }
    });

    let response = tool
        .call_with(
            r#"{"action":"run","command":"sleep 3","detach":true}"#,
            ToolCallContext {
                session_id: "s1".into(),
                task_id: Some("task-detach".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let pid = extract_pid_from_background_message(&response);
    assert!(response.contains("Detached launch"), "{response}");
    assert!(response.contains("Detached mode is enabled"));

    tool.on_task_end("task-detach", "s1").await.unwrap();
    let check = tool
        .call_with(
            &format!(r#"{{"action":"check","pid":{}}}"#, pid),
            aidaemon_tools::testing::call_facts("s1"),
        )
        .await
        .unwrap();
    assert!(
        !check.contains("No tracked process"),
        "detached process should not be cleaned by task-end hook"
    );

    let _ = tool
        .call_with(
            &format!(r#"{{"action":"kill","pid":{}}}"#, pid),
            aidaemon_tools::testing::call_facts("s1"),
        )
        .await;
}

#[tokio::test]
async fn test_daemon_command_returns_immediately_without_background_tracking() {
    // The macOS seatbelt path intentionally keeps the detached child's
    // inherited pipe open, which exercises the daemon early-return branch.
    // Codex's Linux bwrap adapter closes that pipe and reports status 3;
    // detached-process semantics are covered portably by the neighboring
    // `detached_run_launches_and_returns_after_startup_grace` test.
    if !cfg!(target_os = "macos") {
        return;
    }
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let (approval_tx_raw, mut approval_rx) = mpsc::channel::<ApprovalRequest>(1);
    let _approval_tx = crate::approvals::broker::ApprovalBroker::new(approval_tx_raw);
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        2, // 2 second timeout
        4000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    // Auto-approve the daemonization approval request
    tokio::spawn(async move {
        if let Some(req) = approval_rx.recv().await {
            let _ = req.response_tx.send(ApprovalResponse::AllowOnce);
        }
    });

    let start = Instant::now();
    // The owner approved the daemonization card in the kernel.
    let response = tool
        .call_with_execution_context(
            r#"{"action":"run","command":"sleep 5 & echo $!","detach":true}"#,
            None,
            ToolCallContext {
                session_id: "s1".into(),
                ..kernel_owner_ctx()
            },
        )
        .await
        .unwrap()
        .output;
    let elapsed = start.elapsed();

    // Should return promptly (within the timeout + small margin) rather than
    // entering the infinite background tracking loop.
    assert!(
        elapsed < Duration::from_secs(5),
        "daemon command should return within timeout, not stall; took {:?}",
        elapsed
    );
    assert!(
        response.contains("Detached background command launched"),
        "expected daemon early-return message, got: {}",
        response
    );
    assert!(
        response.contains("pid="),
        "expected pid in response, got: {}",
        response
    );
}

#[tokio::test]
async fn test_large_heredoc_soft_blocked() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        1000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    // Build a command >500 chars with UNQUOTED heredoc (should be soft-blocked)
    let large_content = "x".repeat(600);
    let command = format!("cat > /tmp/test.html << EOF\n{}\nEOF", large_content);
    let args = serde_json::json!({
        "action": "run",
        "command": command});

    let response = tool
        .call_with(&args.to_string(), aidaemon_tools::testing::call_facts("s1"))
        .await
        .unwrap();
    assert!(
        response.contains("write_file"),
        "expected heredoc soft-block to recommend write_file, got: {}",
        response
    );
    assert!(
        response.contains("unreliable"),
        "expected heredoc soft-block message, got: {}",
        response
    );
}

#[tokio::test]
async fn test_large_quoted_heredoc_allowed() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        1000,
        PermissionMode::Yolo,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);

    // Build a command >500 chars with QUOTED heredoc (should be allowed)
    let large_content = "echo 'hello'";
    let command = format!(
        "cat > /tmp/test.py << 'PYEOF'\n{}\nPYEOF",
        large_content.repeat(50)
    );
    let args = serde_json::json!({
        "action": "run",
        "command": command});

    let response = tool
        .call_with(&args.to_string(), aidaemon_tools::testing::call_facts("s1"))
        .await
        .unwrap();
    assert!(
        !response.contains("unreliable"),
        "quoted heredoc should NOT be soft-blocked, got: {}",
        response
    );
}
