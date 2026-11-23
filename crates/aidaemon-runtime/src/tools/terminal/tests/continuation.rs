//! Background follow-up turns, re-engagement and deliverables.

use super::*;

#[test]
fn system_events_ui_scripting_detected() {
    // Live 2026-07-12: lock-screen flailing via osascript after computer_use
    // correctly refused input.
    assert!(is_system_events_ui_scripting(
        r#"osascript -e 'tell application "System Events" to tell process "Calculator" to click button "2" of group 1 of window 1'"#
    ));
    assert!(is_system_events_ui_scripting(
        r#"osascript -e 'tell application "System Events" to keystroke "12+3" & return'"#
    ));
    assert!(is_system_events_ui_scripting(
        r#"osascript -e 'tell application "System Events" to key code 36'"#
    ));
}

#[test]
fn system_events_scripting_is_read_from_the_command_tokens() {
    assert!(is_system_events_ui_scripting(
        r#"bash -c "osascript -e 'tell application \"System Events\" to keystroke \"a\"'""#
    ));
    // Text that only names osascript is not a call to it.
    assert!(!is_system_events_ui_scripting(
        r#"echo "osascript can tell System Events to click""#
    ));
}

#[test]
fn system_events_reads_and_notifications_allowed() {
    assert!(!is_system_events_ui_scripting(
        r#"osascript -e 'tell application "System Events" to get name of every process'"#
    ));
    assert!(!is_system_events_ui_scripting(
        r#"osascript -e 'display notification "build done"'"#
    ));
    assert!(!is_system_events_ui_scripting("ls -la ~/projects"));
}

/// A follow-up still making progress is not cut off by the idle limit
/// (live 2026-11-15: a fix-and-retest follow-up was dropped at ten
/// minutes and its raw test output posted instead).
#[tokio::test]
async fn a_continuation_that_keeps_making_progress_runs_past_its_idle_limit() {
    let request = ConversationRequest {
        session_id: "app:synthetic-busy".to_string(),
        user_text: "[Background command finished]".to_string(),
        status_tx: None,
        user_role: aidaemon_tools::semantics::UserRole::Owner,
        channel_ctx: aidaemon_channels::context::ChannelContext::internal(),
        heartbeat: None,
        parent_task_id: None,
        parent_tool_call_id: None,
        parent_result_id: None,
        clock: None,
    };
    let (result, _end) = run_background_continuation_with_timeout(
        &BusyConversationRuntime,
        request,
        Duration::from_secs(2),
    )
    .await;
    assert_eq!(result.unwrap().text, "Drupal is installed.");
}

/// A background job's follow-up is a turn the app sees live: it starts
/// (with no owner message), reports its work, and ends before its answer
/// is delivered as the conversation's message (live 2026-11-15: the
/// app showed the conversation idle until the final message).
#[tokio::test]
async fn a_background_follow_up_is_reported_as_a_live_turn() {
    let watch = aidaemon_channels::turns::TurnWatch::default();
    let recorded = Arc::new(Recorded::default());
    watch.set(recorded.clone());
    let envelope = run_background_continuation_with_timeout(
        &ReportingConversationRuntime(watch),
        ConversationRequest {
            session_id: "app:synthetic-follow-up".to_string(),
            user_text: "[Background command finished]".to_string(),
            status_tx: None,
            user_role: aidaemon_tools::semantics::UserRole::Owner,
            channel_ctx: aidaemon_channels::context::ChannelContext::internal(),
            heartbeat: None,
            parent_task_id: None,
            parent_tool_call_id: None,
            parent_result_id: None,
            clock: None,
        },
        Duration::from_secs(5),
    )
    .await;
    let (envelope, end) = (envelope.0.unwrap(), envelope.1);
    assert_eq!(envelope.text, "Drupal is installed.");
    // Its end is told once its answer is delivered: when the caller
    // drops what it was handed.
    assert_eq!(recorded.0.lock().unwrap().len(), 2);
    drop(end);
    assert_eq!(
        *recorded.0.lock().unwrap(),
        vec![
            format!("started app:synthetic-follow-up: {FOLLOWUP_DESCRIPTION}"),
            "thinking".to_string(),
            "finished Delivered reply=\"\" handed_off=false".to_string(),
        ]
    );
}

/// A request answered once its background runs ended gets its total
/// under the final answer, labelled as the answer's time, only when it
/// adds to the runs' own "ended … after your request": several runs or
/// a notable gap after the last one. One run that ended just before the
/// answer, or no request time, adds nothing (live 2026-11-13: "ended 16s
/// after your request" then "Finished 24s after your request").
#[tokio::test]
async fn a_continuations_final_answer_carries_the_answer_time_when_it_adds_something() {
    let request = |clock| ConversationRequest {
        session_id: "telegram:synthetic-total".to_string(),
        user_text: "[Background helpers finished]".to_string(),
        status_tx: None,
        user_role: aidaemon_tools::semantics::UserRole::Owner,
        channel_ctx: aidaemon_channels::context::ChannelContext::internal(),
        heartbeat: None,
        parent_task_id: Some("synthetic-parent".to_string()),
        parent_tool_call_id: None,
        parent_result_id: None,
        clock,
    };
    let answer = |clock| async move {
        run_background_continuation_with_timeout(
            &AnsweringConversationRuntime,
            request(clock),
            Duration::from_secs(5),
        )
        .await
        .0
        .unwrap()
        .text
    };
    let now = chrono::Utc::now();
    let requested = now - chrono::Duration::seconds(612);
    let clock = |runs, ended_ago| crate::run_state::RequestClock {
        requested,
        runs,
        last_run_ended: Some(now - chrono::Duration::seconds(ended_ago)),
    };

    let several = answer(Some(clock(2, 5))).await;
    let (body, footer) = several.split_once("\n\n").unwrap();
    assert_eq!(body, "Drupal is installed.");
    assert!(
        footer.starts_with(&format!(
            "{}Answer sent 10m 1",
            crate::turn::sliding_window::MEASURED_TIMING_MARK
        )),
        "{footer}"
    );
    assert!(footer.contains("after your request ("), "{footer}");

    let late = answer(Some(clock(1, 120))).await;
    assert!(late.contains("Answer sent 10m 1"), "{late}");

    assert_eq!(
        answer(Some(clock(1, 8))).await,
        "Drupal is installed.",
        "one run that just ended: its own end already says it"
    );
    assert_eq!(answer(None).await, "Drupal is installed.");
}

#[tokio::test]
async fn background_continuation_has_a_wall_clock_bound() {
    let request = ConversationRequest {
        session_id: "telegram:synthetic-timeout".to_string(),
        user_text: "synthetic completion".to_string(),
        status_tx: None,
        user_role: aidaemon_tools::semantics::UserRole::Owner,
        channel_ctx: aidaemon_channels::context::ChannelContext::internal(),
        heartbeat: None,
        parent_task_id: Some("synthetic-parent".to_string()),
        parent_tool_call_id: Some("synthetic-call".to_string()),
        parent_result_id: None,
        clock: None,
    };
    let error = run_background_continuation_with_timeout(
        &PendingConversationRuntime,
        request,
        Duration::from_millis(20),
    )
    .await
    .0
    .expect_err("pending continuation must time out");
    assert!(error.to_string().contains("timed out"), "{error}");
}

#[tokio::test]
async fn reengagement_slots_serialize_concurrent_completions() {
    // Live 2026-07-12: two background completions re-engaged the agent
    // loop concurrently — two racing loops on the same session, duplicate
    // find sweeps, duplicate "Done" pings. Slots must serialize.
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static MAX_ACTIVE: AtomicUsize = AtomicUsize::new(0);
    let mut handles = Vec::new();
    for _ in 0..4 {
        handles.push(tokio::spawn(async {
            let _slot = acquire_reengagement_slot().await;
            let now = ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
            MAX_ACTIVE.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            ACTIVE.fetch_sub(1, Ordering::SeqCst);
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
    assert_eq!(
        MAX_ACTIVE.load(Ordering::SeqCst),
        1,
        "re-engagements must not run concurrently"
    );
}

#[test]
fn deliverable_caption_has_no_command() {
    let cap = build_deliverable_caption("netprobe_results3.txt", None);
    assert!(
        !cap.contains("cd "),
        "caption must not contain a shell command"
    );
    assert!(
        !cap.contains('`'),
        "caption must not contain backticked command"
    );
    assert!(cap.contains("netprobe_results3.txt"));
}

#[test]
fn small_text_deliverable_caption_shows_its_content() {
    let dir = tempfile::tempdir().unwrap();
    let small = dir.path().join("bg.txt");
    std::fs::write(&small, "build-ok\n").unwrap();
    let preview = small_text_preview(&small);
    assert_eq!(preview.as_deref(), Some("build-ok"));
    let cap = build_deliverable_caption(
        "bg.txt",
        Some(&format!("Done. bg.txt contains:\n{}", preview.unwrap())),
    );
    assert!(cap.contains("build-ok"), "{cap}");

    let large = dir.path().join("big.txt");
    std::fs::write(&large, "x".repeat(SMALL_TEXT_PREVIEW_BYTES + 1)).unwrap();
    assert_eq!(small_text_preview(&large), None);
    let binary = dir.path().join("img.png");
    std::fs::write(&binary, [0x89, b'P', b'N', b'G', 0, 1, 2]).unwrap();
    assert_eq!(small_text_preview(&binary), None);
    let empty = dir.path().join("empty.txt");
    std::fs::write(&empty, "  \n").unwrap();
    assert_eq!(small_text_preview(&empty), None);
}

#[test]
fn reengagement_followup_uses_typed_parent_edge_not_session_plan() {
    let followup = build_background_reengagement_followup(
        "python3 /tmp/ping_latency.py",
        "latency results written to /tmp/latency.txt",
    );
    // A check that found something to fix may run again after the fix.
    assert!(followup.contains("after you changed files is a new check"));
    assert!(
        followup.contains("attached runtime continuation edge"),
        "follow-up must name the typed ownership boundary: {followup}"
    );
    assert!(
        followup.contains("latency results written to /tmp/latency.txt"),
        "follow-up must include the command output: {followup}"
    );
    assert!(
        followup.contains("python3 /tmp/ping_latency.py"),
        "follow-up must include the command: {followup}"
    );
    // The self-reported lifecycle flapped between typed values and
    // "unknown" across identical contexts; the continuation must state
    // the typed facts so the model reads them instead of guessing.
    assert!(
        followup.contains("nonterminal background handoff"),
        "follow-up must state the typed handoff disposition: {followup}"
    );
    assert!(
        followup.contains("recorded as backgrounded"),
        "follow-up must state the typed terminal result: {followup}"
    );
    assert!(
        followup.contains("queued for delivery through this continuation"),
        "follow-up must state the typed delivery state: {followup}"
    );
}

#[test]
fn reengagement_followup_cannot_import_unrelated_checklist_text() {
    let followup = build_background_reengagement_followup(
        "python3 /tmp/ping.py",
        "latency written to /tmp/latency.txt",
    );
    assert!(
        followup.contains("/tmp/latency.txt"),
        "must include the command output: {followup}"
    );
    assert!(!followup.contains("UNCHECKED"));
    assert!(!followup.contains("track_requirements"));
}

#[test]
fn test_reengagement_allowed_caps_a_repeating_job_not_progress() {
    let mut log = HashMap::new();
    let t0 = Instant::now();

    // The same job re-engages twice, then its output goes out raw.
    for _ in 0..MAX_REENGAGEMENTS_PER_JOB {
        assert!(reengagement_allowed(&mut log, "s", "find ~ -name x", t0));
    }
    assert!(!reengagement_allowed(&mut log, "s", "find  ~ -name x", t0));

    // Other jobs in the same session are progress: build, test, check.
    assert!(reengagement_allowed(&mut log, "s", "cargo build", t0));
    assert!(reengagement_allowed(&mut log, "s", "cargo test -q", t0));
    assert!(reengagement_allowed(&mut log, "s", "cargo clippy", t0));

    // A different session has its own budget.
    assert!(reengagement_allowed(
        &mut log,
        "other",
        "find ~ -name x",
        t0
    ));

    // After the window elapses, the repeating job's budget refills.
    let later = t0 + REENGAGE_WINDOW + Duration::from_secs(1);
    assert!(reengagement_allowed(&mut log, "s", "find ~ -name x", later));
}

#[test]
fn test_reengagement_allowed_has_a_session_backstop() {
    let mut log = HashMap::new();
    let t0 = Instant::now();
    for n in 0..MAX_REENGAGEMENTS_PER_WINDOW {
        assert!(reengagement_allowed(&mut log, "s", &format!("job {n}"), t0));
    }
    assert!(!reengagement_allowed(&mut log, "s", "one more job", t0));
    let later = t0 + REENGAGE_WINDOW + Duration::from_secs(1);
    assert!(reengagement_allowed(&mut log, "s", "one more job", later));
}
