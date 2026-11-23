//! Replay skipping and connection notices.

use super::*;

#[test]
fn test_should_skip_stdout_replay_when_gap_detected() {
    let frames = vec![(25, "partial output".to_string())];
    assert!(should_skip_stdout_replay(3, 25, &frames));
}

#[test]
fn test_should_skip_stdout_replay_for_noisy_fresh_attach() {
    let frames = (1..=20)
        .map(|seq| (seq, format!("progress {}\r", seq)))
        .collect::<Vec<_>>();
    assert!(should_skip_stdout_replay(0, 1, &frames));
}

#[test]
fn test_should_not_skip_stdout_replay_for_small_clean_resume() {
    let frames = vec![(8, "line one\n".to_string()), (9, "line two\n".to_string())];
    assert!(!should_skip_stdout_replay(7, 8, &frames));
}

#[test]
fn test_build_skipped_stdout_replay_status_message_omits_clean_resume_noise() {
    let frames = vec![(8, "line one\n".to_string()), (9, "line two\n".to_string())];
    assert_eq!(
        build_skipped_stdout_replay_status_message(7, 8, &frames),
        None
    );
    assert_eq!(build_skipped_stdout_replay_status_message(7, 8, &[]), None);
}

#[test]
fn test_build_skipped_stdout_replay_status_message_mentions_recovery_only_when_skipped() {
    let frames = (1..=20)
        .map(|seq| (seq, format!("progress {}\r", seq)))
        .collect::<Vec<_>>();
    let message =
        build_skipped_stdout_replay_status_message(0, 1, &frames).expect("skipped replay status");
    assert!(message.contains("Connection recovered."));
    assert!(message.contains("Skipped replaying 20 buffered output frame(s)"));
    assert!(message.contains("If the terminal appears blank"));
}

#[test]
fn test_build_connection_notice_payload_marks_out_of_band_notice() {
    let payload = build_connection_notice_payload(
        "Connection recovered.".to_string(),
        "warn",
        "terminal",
        5200,
    );
    assert_eq!(payload["kind"].as_str(), Some("connection_notice"));
    assert_eq!(payload["tone"].as_str(), Some("warn"));
    assert_eq!(payload["scope"].as_str(), Some("terminal"));
    assert_eq!(payload["ttl_ms"].as_u64(), Some(5200));
}

#[test]
fn test_should_skip_review_stream_replay_when_gap_detected() {
    assert!(should_skip_review_stream_replay(4, 20, 3));
}

#[test]
fn test_should_not_skip_review_stream_replay_for_small_resume() {
    assert!(!should_skip_review_stream_replay(12, 13, 4));
}

#[test]
fn test_build_skipped_review_stream_replay_status_message_omits_clean_resume_noise() {
    assert_eq!(
        build_skipped_review_stream_replay_status_message(12, 13, 4),
        None
    );
    assert_eq!(
        build_skipped_review_stream_replay_status_message(12, 13, 0),
        None
    );
}

#[test]
fn test_build_skipped_review_stream_replay_status_message_mentions_recovery_only_when_skipped() {
    let message = build_skipped_review_stream_replay_status_message(4, 20, 3)
        .expect("skipped review replay status");
    assert!(message.contains("Connection recovered."));
    assert!(message.contains("Skipped replaying 3 buffered review stream chunk(s)"));
}
