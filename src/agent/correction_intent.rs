//! Pure helper that reconstructs a [`CorrectionSubjectContext`] from session
//! history, so the autonomous-correction bridge knows what the original request
//! was when it retries a failed or idle-reaped command.
//!
//! **Pure module** — no async, no I/O, no database access.  The caller fetches
//! `history` (e.g. via `EventStore::get_conversation_history`) and passes it
//! in here.

use std::path::{Path, PathBuf};

use crate::agent::correction_sandbox::{CorrectionSubjectContext, IntendedAccount};
use crate::traits::{Message, SelfCorrectionSubjectKind};

/// Maximum characters of `failed_command` included in the synthesized
/// `completion_contract_summary` line.
const MAX_COMMAND_CHARS_IN_SUMMARY: usize = 120;

// ── Working-dir safety ────────────────────────────────────────────────────────

/// Returns `true` when `dir` is genuinely-invalid for unattended autonomous
/// correction and must be refused.
///
/// As of Plan 3c, broad scopes like `/` and the home directory are
/// **intentionally allowed**: read-only remediation may legitimately answer
/// whole-computer questions (e.g. "what's the biggest file?"). The safety for
/// those scopes is the sandbox's read-only allowlist + sensitive-file guards —
/// NOT the working_dir boundary. This guard therefore refuses ONLY
/// genuinely-invalid scopes:
///
/// - Empty path (no directory at all).
/// - A non-absolute / relative path. The bridge canonicalizes the derived
///   working_dir before this guard sees it; a relative path that survives
///   canonicalization signals a derivation failure, so it is refused.
///
/// Any absolute directory — including `/`, `$HOME`, and a literal expanded home
/// path — is allowed and protected downstream by the sandbox.
///
/// The caller (the bridge, P3b) must call this **before** launching a
/// correction session and refuse if it returns `true`.
/// [`reconstruct_subject_context`] does **not** rewrite the directory — it
/// stays pure/honest and leaves the safety decision to the caller.
#[allow(dead_code)]
pub fn is_unsafe_correction_working_dir(dir: &Path) -> bool {
    // Empty path — no directory at all.
    if dir.as_os_str().is_empty() {
        return true;
    }

    // Non-absolute / relative path. After the bridge canonicalizes the derived
    // scope, a relative path indicates derivation failure — refuse it. This also
    // catches the literal unexpanded tilde `~`, which is relative.
    if !dir.is_absolute() {
        return true;
    }

    // Any absolute dir — including `/` and the home directory — is allowed;
    // the sandbox's read-only + sensitive-file guards provide the safety.
    false
}

// ── Context reconstruction ────────────────────────────────────────────────────

/// Reconstructs a [`CorrectionSubjectContext`] from already-fetched session
/// history.
///
/// ## Arguments
///
/// * `history` — conversation messages, **newest last** (the ordering returned
///   by `EventStore::get_conversation_history`).
/// * `session_id` — session the correction belongs to.
/// * `subject_id` — unique ID for the correction subject (task ID, goal ID, …).
/// * `subject_kind` — what kind of subject is being corrected.
/// * `working_dir` — project / task directory for the retry.  This value is
///   passed through **unchanged**; the caller must validate it with
///   [`is_unsafe_correction_working_dir`] before launching correction.
/// * `failed_command` — the shell command (or descriptive label) that was
///   idle-reaped or failed, used to synthesize the
///   `completion_contract_summary`.
///
/// ## Behavior
///
/// * `original_request`: the most recent `role == "user"` message text found
///   in `history`.  If no user message exists, falls back to the safe generic
///   `"(original request unavailable)"`.
/// * `completion_contract_summary`: a one-liner synthesized from
///   `failed_command` (truncated to [`MAX_COMMAND_CHARS_IN_SUMMARY`] chars).
/// * `intended_accounts`: always `vec![]` — no external accounts for
///   unattended correction.
/// * `allowed_external_targets`: always `vec![]`.
/// * `working_dir`: the passed-in value, not rewritten.
#[allow(dead_code)]
pub fn reconstruct_subject_context(
    history: &[Message],
    session_id: &str,
    subject_id: &str,
    subject_kind: SelfCorrectionSubjectKind,
    working_dir: PathBuf,
    failed_command: &str,
) -> CorrectionSubjectContext {
    // Walk history newest-last to find the most recent user message.
    let original_request = history
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "(original request unavailable)".to_string());

    // Truncate the command if it would make the summary unwieldy.
    // Use the UTF-8-safe helper so multi-byte chars straddling the byte boundary
    // don't cause a panic (the systemic bug class documented in utils.rs).
    let truncated_command =
        crate::utils::truncate_str(failed_command, MAX_COMMAND_CHARS_IN_SUMMARY);

    let completion_contract_summary = format!(
        "Re-attempt the goal behind the failed command `{}` with a faster, scoped approach.",
        truncated_command
    );

    CorrectionSubjectContext {
        subject_id: subject_id.to_string(),
        subject_kind,
        session_id: session_id.to_string(),
        original_request,
        completion_contract_summary,
        intended_accounts: Vec::<IntendedAccount>::new(),
        allowed_external_targets: Vec::<String>::new(),
        working_dir,
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::SelfCorrectionSubjectKind;
    use chrono::Utc;

    /// Build a minimal [`Message`] for tests — only `role` and `content` matter
    /// here; all other fields use harmless defaults.
    fn make_message(role: &str, content: Option<&str>) -> Message {
        Message {
            id: "test-id".to_string(),
            session_id: "test-session".to_string(),
            role: role.to_string(),
            content: content.map(|s| s.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            annotations: vec![],
            importance: 0.5,
            embedding: None,
            turn_id: None,
            attachments: vec![],
        }
    }

    // ── P3a.3 TDD tests ──────────────────────────────────────────────────────

    /// RED → GREEN: history with a user message followed by assistant replies
    /// must produce `original_request` equal to the user message text.
    #[test]
    fn test_reconstruct_extracts_latest_user_request() {
        let history = vec![
            make_message("user", Some("what's the biggest file?")),
            make_message("assistant", Some("Let me check…")),
            make_message("assistant", Some("Here is the result.")),
        ];

        let ctx = reconstruct_subject_context(
            &history,
            "session-1",
            "subject-1",
            SelfCorrectionSubjectKind::BackgroundCommand,
            PathBuf::from("/tmp/myproject"),
            "find / -type f",
        );

        assert_eq!(ctx.original_request, "what's the biggest file?");
    }

    /// RED → GREEN: when history is empty the function must not panic and must
    /// return the safe generic fallback string.
    #[test]
    fn test_reconstruct_empty_history_safe_fallback() {
        let ctx = reconstruct_subject_context(
            &[],
            "session-2",
            "subject-2",
            SelfCorrectionSubjectKind::Task,
            PathBuf::from("/tmp/project"),
            "cargo build",
        );

        assert_eq!(ctx.original_request, "(original request unavailable)");
        // Must not panic — that's the primary contract of this test.
    }

    /// RED → GREEN: `intended_accounts` and `allowed_external_targets` must
    /// always be empty for unattended correction.
    #[test]
    fn test_reconstruct_sets_empty_accounts_and_targets() {
        let history = vec![make_message("user", Some("run the suite"))];

        let ctx = reconstruct_subject_context(
            &history,
            "session-3",
            "subject-3",
            SelfCorrectionSubjectKind::Goal,
            PathBuf::from("/tmp/repo"),
            "cargo test",
        );

        assert!(
            ctx.intended_accounts.is_empty(),
            "intended_accounts must be empty"
        );
        assert!(
            ctx.allowed_external_targets.is_empty(),
            "allowed_external_targets must be empty"
        );
    }

    /// RED → GREEN (3c scope relax): is_unsafe_correction_working_dir now refuses
    /// ONLY genuinely-invalid scopes — empty and relative/uncanonicalized paths.
    /// Broad-but-absolute scopes (`/`, the actual $HOME) are intentionally
    /// ALLOWED: read-only remediation over the whole computer is protected by the
    /// sandbox's read-only allowlist + sensitive-file guards, not by the
    /// working_dir boundary.
    #[test]
    fn test_is_unsafe_correction_working_dir() {
        // Filesystem root — now ALLOWED (sandbox protects it).
        assert!(
            !is_unsafe_correction_working_dir(Path::new("/")),
            "/ must now be allowed (false)"
        );

        // $HOME from the environment — now ALLOWED.
        if let Ok(home) = std::env::var("HOME") {
            assert!(
                !is_unsafe_correction_working_dir(Path::new(&home)),
                "$HOME must now be allowed (false)"
            );
        }

        // A concrete absolute project subdirectory — allowed.
        assert!(
            !is_unsafe_correction_working_dir(Path::new("/tmp/myproject/src")),
            "/tmp/myproject/src must be allowed (false)"
        );

        // /tmp — allowed (absolute).
        assert!(
            !is_unsafe_correction_working_dir(Path::new("/tmp")),
            "/tmp must be allowed (false)"
        );

        // Empty path — still refused (no directory at all).
        assert!(
            is_unsafe_correction_working_dir(Path::new("")),
            "empty path must be refused (true)"
        );

        // Relative / uncanonicalized path — refused (post-canonicalization a
        // relative path signals a derivation failure).
        assert!(
            is_unsafe_correction_working_dir(Path::new("relative/dir")),
            "relative path must be refused (true)"
        );

        // Literal tilde — relative (not absolute), so refused.
        assert!(
            is_unsafe_correction_working_dir(Path::new("~")),
            "literal ~ is non-absolute → refused (true)"
        );
    }

    // ── Extra coverage ────────────────────────────────────────────────────────

    /// The most-recent user message wins when there are multiple user turns.
    #[test]
    fn test_reconstruct_picks_latest_user_message() {
        let history = vec![
            make_message("user", Some("first question")),
            make_message("assistant", Some("reply 1")),
            make_message("user", Some("what's the biggest file?")),
            make_message("assistant", Some("reply 2")),
        ];

        let ctx = reconstruct_subject_context(
            &history,
            "s",
            "sub",
            SelfCorrectionSubjectKind::BackgroundCommand,
            PathBuf::from("/home/user/project"),
            "ls -lh",
        );

        assert_eq!(ctx.original_request, "what's the biggest file?");
    }

    /// History with only assistant messages (no user turn) falls back safely.
    #[test]
    fn test_reconstruct_no_user_message_in_history() {
        let history = vec![make_message("assistant", Some("some assistant text"))];

        let ctx = reconstruct_subject_context(
            &history,
            "s",
            "sub",
            SelfCorrectionSubjectKind::Task,
            PathBuf::from("/tmp/p"),
            "echo hi",
        );

        assert_eq!(ctx.original_request, "(original request unavailable)");
    }

    /// Long commands are truncated in the summary but the full text appears in
    /// original_request.
    #[test]
    fn test_reconstruct_long_command_truncated_in_summary() {
        let long_cmd = "a".repeat(200);

        let ctx = reconstruct_subject_context(
            &[make_message("user", Some("do something"))],
            "s",
            "sub",
            SelfCorrectionSubjectKind::Task,
            PathBuf::from("/tmp/p"),
            &long_cmd,
        );

        // Summary must not contain the full 200-char command.
        assert!(
            ctx.completion_contract_summary.len() < long_cmd.len() + 100,
            "summary should be reasonably short"
        );
        assert!(
            ctx.completion_contract_summary.contains("..."),
            "summary must contain ellipsis when command is truncated"
        );
    }

    /// A long command containing multi-byte UTF-8 characters positioned to
    /// straddle byte 120 must not panic and must produce a non-empty summary.
    #[test]
    fn test_reconstruct_multibyte_command_no_panic() {
        // "find ~ " is 7 bytes; each "日本語" is 9 bytes.
        // Repeating 50 times gives 457 bytes total — well past the 120-byte limit.
        // The 3-byte chars straddle the 120-byte boundary, triggering the old
        // raw-byte-index panic.
        let multibyte_cmd = "find ~ ".to_string() + &"日本語".repeat(50);
        assert!(multibyte_cmd.len() > MAX_COMMAND_CHARS_IN_SUMMARY);

        // Must not panic.
        let ctx = reconstruct_subject_context(
            &[make_message("user", Some("search for large files"))],
            "s",
            "sub",
            SelfCorrectionSubjectKind::BackgroundCommand,
            PathBuf::from("/tmp/project"),
            &multibyte_cmd,
        );

        // Summary must be non-empty and must contain the truncation marker.
        assert!(
            !ctx.completion_contract_summary.is_empty(),
            "summary must not be empty for multibyte command"
        );
        assert!(
            ctx.completion_contract_summary.contains("..."),
            "summary must contain ellipsis when multibyte command is truncated"
        );
    }

    /// scalar fields are threaded through correctly.
    #[test]
    fn test_reconstruct_scalar_fields() {
        let ctx = reconstruct_subject_context(
            &[make_message("user", Some("ping"))],
            "my-session",
            "my-subject",
            SelfCorrectionSubjectKind::Goal,
            PathBuf::from("/srv/project"),
            "ping localhost",
        );

        assert_eq!(ctx.session_id, "my-session");
        assert_eq!(ctx.subject_id, "my-subject");
        assert_eq!(ctx.subject_kind, SelfCorrectionSubjectKind::Goal);
        assert_eq!(ctx.working_dir, PathBuf::from("/srv/project"));
    }

    /// working_dir is passed through unchanged even if it is unsafe — the
    /// bridge is responsible for checking is_unsafe_correction_working_dir.
    #[test]
    fn test_reconstruct_does_not_rewrite_unsafe_working_dir() {
        let ctx = reconstruct_subject_context(
            &[make_message("user", Some("go"))],
            "s",
            "sub",
            SelfCorrectionSubjectKind::Task,
            PathBuf::from("/"),
            "rm -rf /",
        );

        // Working dir passed through unchanged — the bridge decides what to do.
        assert_eq!(ctx.working_dir, PathBuf::from("/"));
    }
}
