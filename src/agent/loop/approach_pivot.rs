//! Approach-pivot retries: persistence instead of giving up at a stall.
//!
//! When the loop reaches a stall give-up point while the current approach is
//! demonstrably failing (unrecovered errors, or no successful tool calls at
//! all), the harness does not end the task. It builds a deterministic
//! failure record of the attempt — what was tried, what errored, what was
//! already changed in the world — injects it as a pivot directive, resets
//! the stall trackers for fresh runway, and lets the loop continue with an
//! explicit instruction to try a fundamentally different approach.
//!
//! Design constraints (from review):
//! - The record is built deterministically from the ledger, never by the
//!   (possibly stuck) model — error lines are copied verbatim, not
//!   paraphrased.
//! - The raw failed attempt stays in context/history; the record is a
//!   *reference*, not a replacement. Nothing is discarded.
//! - Pivots are bounded; hard caps (iterations/tokens/wall-clock) still
//!   bound everything. After pivots are exhausted, the existing graceful
//!   exits apply unchanged.

/// Maximum approach pivots per task (total approaches = pivots + 1).
pub(in crate::agent) const MAX_APPROACH_PIVOTS: usize = 2;

/// Whether the current approach is demonstrably failing and worth pivoting away
/// from. Pivot only when there is something to pivot *from* (at least one tool
/// attempt) and the approach is failing (unrecovered errors, or zero successful
/// calls). A stall after clean meaningful progress is a summarization problem,
/// not an approach problem — no pivot. The pivot *budget* (how many pivots are
/// allowed) is enforced separately by `SelfCorrectionController`.
pub(in crate::agent) fn approach_is_failing(
    tool_attempts: usize,
    unrecovered_errors: usize,
    total_successful_tool_calls: usize,
) -> bool {
    tool_attempts > 0 && (unrecovered_errors > 0 || total_successful_tool_calls == 0)
}

/// Build the deterministic failure record for the pivot directive.
///
/// Includes: attempt number, the tool calls tried, every unrecovered error
/// line verbatim (deduplicated, bounded), and the count of mutations the
/// attempt already performed (so the next approach knows the world changed).
pub(in crate::agent) fn build_failure_record(
    attempt_number: usize,
    tool_calls: &[String],
    errors: &[(String, bool)],
    mutation_count: usize,
) -> String {
    const MAX_LISTED_CALLS: usize = 10;
    const MAX_LISTED_ERRORS: usize = 5;
    const MAX_ERROR_LINE_CHARS: usize = 300;

    let mut record = format!("Failed approach record (attempt #{attempt_number}):\n");

    record.push_str("Tool calls tried:\n");
    for call in tool_calls.iter().take(MAX_LISTED_CALLS) {
        record.push_str("- ");
        record.push_str(crate::utils::truncate_str(call, MAX_ERROR_LINE_CHARS).as_ref());
        record.push('\n');
    }
    if tool_calls.len() > MAX_LISTED_CALLS {
        record.push_str(&format!(
            "- … and {} more\n",
            tool_calls.len() - MAX_LISTED_CALLS
        ));
    }

    // Unrecovered errors, verbatim (truncated for length only) and deduped
    // in first-seen order.
    let mut seen = std::collections::HashSet::new();
    let unrecovered: Vec<&str> = errors
        .iter()
        .filter(|(_, recovered)| !recovered)
        .map(|(error, _)| error.as_str())
        .filter(|error| seen.insert(*error))
        .collect();
    if !unrecovered.is_empty() {
        record.push_str("Unrecovered errors (verbatim):\n");
        for error in unrecovered.iter().take(MAX_LISTED_ERRORS) {
            record.push_str("- ");
            record.push_str(crate::utils::truncate_str(error, MAX_ERROR_LINE_CHARS).as_ref());
            record.push('\n');
        }
        if unrecovered.len() > MAX_LISTED_ERRORS {
            record.push_str(&format!(
                "- … and {} more distinct errors\n",
                unrecovered.len() - MAX_LISTED_ERRORS
            ));
        }
    }

    if mutation_count > 0 {
        record.push_str(&format!(
            "Already performed: {} state-changing action{} — do not blindly repeat them.\n",
            mutation_count,
            if mutation_count == 1 { "" } else { "s" }
        ));
    }

    record
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failing_when_errors_or_zero_success_and_something_tried() {
        // Unrecovered errors with attempts → failing.
        assert!(approach_is_failing(4, 3, 1));
        // Zero successes also qualifies even without recorded errors
        // (e.g. every call was guard-blocked).
        assert!(approach_is_failing(2, 0, 0));
    }

    #[test]
    fn not_failing_when_nothing_tried_or_clean_progress() {
        // No tool attempts at all — nothing to pivot from.
        assert!(!approach_is_failing(0, 0, 0));
        // Clean progress, no unrecovered errors — summarize, don't pivot.
        assert!(!approach_is_failing(6, 0, 6));
    }

    #[test]
    fn failure_record_is_deterministic_and_verbatim() {
        let tool_calls = vec![
            "terminal(`npm run deploy`)".to_string(),
            "terminal(`npm run deploy --force`)".to_string(),
        ];
        let errors = vec![
            (
                "Error: EACCES permission denied /usr/lib".to_string(),
                false,
            ),
            ("transient timeout".to_string(), true), // recovered → excluded
            (
                "Error: EACCES permission denied /usr/lib".to_string(),
                false,
            ), // dup → once
        ];
        let record = build_failure_record(2, &tool_calls, &errors, 1);

        assert!(record.contains("attempt #2"), "record: {record}");
        assert!(record.contains("npm run deploy"));
        assert!(
            record.contains("Error: EACCES permission denied /usr/lib"),
            "error lines must be verbatim"
        );
        assert_eq!(
            record.matches("EACCES").count(),
            1,
            "duplicate errors collapse to one"
        );
        assert!(
            !record.contains("transient timeout"),
            "recovered errors are not part of the failure reference"
        );
        assert!(
            record.contains("1 state-changing action"),
            "must warn the next approach that the world already changed"
        );
        // Deterministic: same inputs, same record.
        assert_eq!(record, build_failure_record(2, &tool_calls, &errors, 1));
    }

    #[test]
    fn failure_record_bounds_runaway_inputs() {
        let tool_calls: Vec<String> = (0..50).map(|i| format!("terminal(cmd-{i})")).collect();
        let errors: Vec<(String, bool)> = (0..50)
            .map(|i| (format!("Error: failure {i} {}", "x".repeat(500)), false))
            .collect();
        let record = build_failure_record(1, &tool_calls, &errors, 0);
        assert!(
            record.len() < 4000,
            "record must stay compact, got {} chars",
            record.len()
        );
    }
}
