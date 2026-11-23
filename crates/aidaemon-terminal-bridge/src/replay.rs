//! Replaying output to a reattaching client, and connection notices.

use super::*;

pub(crate) fn split_utf8_chunks(s: &str, max_bytes: usize) -> Vec<String> {
    if s.len() <= max_bytes {
        return vec![s.to_string()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in s.chars() {
        let next_len = current.len() + ch.len_utf8();
        if next_len > max_bytes && !current.is_empty() {
            out.push(current);
            current = String::new();
        }
        current.push(ch);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

pub(crate) fn replay_frames_total_bytes(frames: &[(u64, String)]) -> usize {
    frames.iter().map(|(_, chunk)| chunk.len()).sum()
}

pub(crate) fn replay_frames_look_interactive(frames: &[(u64, String)]) -> bool {
    frames
        .iter()
        .any(|(_, chunk)| chunk.contains('\r') || chunk.contains('\u{001b}'))
}

pub(crate) fn should_skip_stdout_replay(
    resume_from_seq: u64,
    oldest_seq: u64,
    frames: &[(u64, String)],
) -> bool {
    if frames.is_empty() {
        return false;
    }
    if resume_from_seq.saturating_add(1) < oldest_seq {
        // We already lost continuity; replay from an arbitrary tail often renders as
        // visual garbage for interactive CLIs, so prefer a clean live resume.
        return true;
    }
    if resume_from_seq != 0 {
        return false;
    }

    let total_bytes = replay_frames_total_bytes(frames);
    if frames.len() > REATTACH_REPLAY_HARD_CAP_FRAMES
        || total_bytes > REATTACH_REPLAY_HARD_CAP_BYTES
    {
        return true;
    }

    replay_frames_look_interactive(frames)
        && (frames.len() > REATTACH_REPLAY_INTERACTIVE_CAP_FRAMES
            || total_bytes > REATTACH_REPLAY_INTERACTIVE_CAP_BYTES)
}

pub(crate) fn should_skip_review_stream_replay(
    resume_from_review_stream_seq: u64,
    oldest_review_stream_seq: u64,
    replay_len: usize,
) -> bool {
    if replay_len == 0 {
        return false;
    }
    if resume_from_review_stream_seq.saturating_add(1) < oldest_review_stream_seq {
        return true;
    }
    resume_from_review_stream_seq == 0 && replay_len > REATTACH_REVIEW_REPLAY_CAP_FRAMES
}

pub(crate) fn build_skipped_stdout_replay_status_message(
    resume_from_seq: u64,
    oldest_seq: u64,
    frames: &[(u64, String)],
) -> Option<String> {
    if !should_skip_stdout_replay(resume_from_seq, oldest_seq, frames) {
        return None;
    }
    let blank_terminal_hint =
        "If the terminal appears blank, press Enter or send a command to refresh live output.";
    let reason = if resume_from_seq.saturating_add(1) < oldest_seq {
        "buffered history was unavailable"
    } else {
        "buffered output was too noisy to replay cleanly"
    };
    Some(format!(
        "Connection recovered. Skipped replaying {} buffered output frame(s) because {}. Session is still active. {}",
        frames.len(),
        reason,
        blank_terminal_hint
    ))
}

pub(crate) fn build_skipped_review_stream_replay_status_message(
    resume_from_review_stream_seq: u64,
    oldest_review_stream_seq: u64,
    replay_len: usize,
) -> Option<String> {
    if !should_skip_review_stream_replay(
        resume_from_review_stream_seq,
        oldest_review_stream_seq,
        replay_len,
    ) {
        return None;
    }
    let reason = if resume_from_review_stream_seq.saturating_add(1) < oldest_review_stream_seq {
        "buffered review history was unavailable"
    } else {
        "the buffered review stream was too large to replay cleanly"
    };
    Some(format!(
        "Connection recovered. Skipped replaying {} buffered review stream chunk(s) because {}.",
        replay_len, reason
    ))
}

pub(crate) fn build_connection_notice_payload(
    message: String,
    tone: &str,
    scope: &str,
    ttl_ms: u64,
) -> Value {
    json!({
        "kind": "connection_notice",
        "message": message,
        "tone": tone,
        "scope": scope,
        "ttl_ms": ttl_ms,
    })
}
