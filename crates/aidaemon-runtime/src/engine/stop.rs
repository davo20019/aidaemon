//! What the host's stop check reads: reply defects, the reply that ships,
//! and when the ledger closeout may demand more work.

/// Extra passes the ledger may demand in one run.
pub const MAX_CLOSEOUT_DEMANDS: u32 = 2;

/// A final reply that must not ship as is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyDefect {
    /// No text at all.
    Empty,
    /// Only internal markers or tool envelopes; sanitizing leaves nothing.
    InternalMarkers,
    /// A pasted file page, raw data dump, or a copy of the latest tool output.
    ToolOutputPaste,
    /// A well-formed reply written after a closeout demand that ended unmet
    /// (`closeout_ended_unmet`): judged from the run's state, never from the
    /// reply's words, it answers the daemon's demand rather than the user.
    AnswersUnmetDemand,
}

/// Reply shape checks, all structural (sanitizer
/// output, harness page headers, data-dump shape, overlap with the latest
/// tool output), never wording.
pub fn reply_defect(text: &str, latest_tool_output: Option<&str>) -> Option<ReplyDefect> {
    if text.trim().is_empty() {
        return Some(ReplyDefect::Empty);
    }
    let sanitized = aidaemon_tools::sanitize::sanitize_user_facing_reply(text);
    if aidaemon_tools::sanitize::reply_gutted_by_sanitization(
        text.trim().chars().count(),
        &sanitized,
    ) || crate::turn::task_outcome::response_looks_like_plain_text_tool_call(text)
        || crate::turn::reply_checks::reply_has_protocol_markers(text)
    {
        return Some(ReplyDefect::InternalMarkers);
    }
    let pasted = crate::turn::reply_checks::reply_is_pasted_file_page(&sanitized)
        || crate::turn::reply_checks::reply_is_raw_data_dump(&sanitized)
        || latest_tool_output.is_some_and(|output| {
            crate::turn::reply_checks::reply_duplicates_tool_output(&sanitized, output)
        });
    pasted.then_some(ReplyDefect::ToolOutputPaste)
}

/// The reply that ships for a completed run: sanitized, and an honest
/// fallback instead of an empty or gutted answer (never a bare "Done.").
/// After tool work the fallback describes that work: the receipts, not a
/// silent model, say whether anything happened.
pub fn finalize_reply(text: &str, tool_calls: &[String], response_note: Option<&str>) -> String {
    if text.trim().is_empty() {
        if !tool_calls.is_empty() {
            let refs: Vec<&str> = tool_calls.iter().map(String::as_str).collect();
            return crate::turn::completion_checks::build_activity_summary_reply(&refs);
        }
        return crate::turn::completion_checks::build_empty_response_fallback(response_note);
    }
    let sanitized = aidaemon_tools::sanitize::sanitize_user_facing_reply(text);
    if !aidaemon_tools::sanitize::reply_gutted_by_sanitization(
        text.trim().chars().count(),
        &sanitized,
    ) {
        return sanitized;
    }
    if tool_calls.is_empty() {
        "I ran into a formatting problem composing that answer and couldn't recover it. Ask me \
         again and I'll re-gather the details."
            .to_string()
    } else {
        let refs: Vec<&str> = tool_calls.iter().map(String::as_str).collect();
        crate::turn::completion_checks::build_activity_summary_reply(&refs)
    }
}

/// What the ledger says when the model wants to stop.
pub struct StopFacts {
    /// Expectations an admissible tool could still satisfy.
    pub reachable_obligation_ids: Vec<String>,
    /// Operations of this run that completed.
    pub completed_results: usize,
    /// `completed_results` when the executor's checklist was last demanded.
    pub executor_demanded_at: Option<usize>,
    pub terminal_policy_denial: bool,
    /// A reachable observation obligation remains after a successful change
    /// and has not been demanded yet (the owner's verify-once rule).
    pub verification_after_mutation: bool,
    /// The run's latest call failed and nothing succeeded, and its recovery
    /// pass has not been used.
    pub unrecovered_failure: bool,
}

/// When the closeout may demand more work, read from receipts:
/// contract expectations are demanded only while nothing has completed; the
/// executor's own open checklist items are demanded again whenever completed
/// work changed since the last demand; nothing is demanded after a terminal
/// policy denial.
pub fn closeout_demands_work(facts: &StopFacts) -> bool {
    if facts.terminal_policy_denial {
        return false;
    }
    if facts.unrecovered_failure && facts.completed_results == 0 {
        return true;
    }
    if facts.reachable_obligation_ids.is_empty() {
        return false;
    }
    if facts.verification_after_mutation {
        return true;
    }
    let executor_demands = facts
        .reachable_obligation_ids
        .iter()
        .any(|id| id.contains("/obligation:checklist:"))
        && facts.executor_demanded_at != Some(facts.completed_results);
    facts.completed_results == 0 || executor_demands
}

/// Whether a stop the closeout accepted ends a demand unmet: the ledger
/// demanded more work earlier in the run and still holds expectations an
/// admissible tool could satisfy. The model's reply then answers the
/// daemon's demand, not the user's request.
pub fn closeout_ended_unmet(facts: &StopFacts, demands_made: u32) -> bool {
    demands_made > 0 && !facts.reachable_obligation_ids.is_empty()
}
