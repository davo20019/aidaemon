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
pub fn finalize_reply(
    text: &str,
    tool_calls: &[crate::turn::post_task::LearnedCall],
    response_note: Option<&str>,
) -> String {
    if text.trim().is_empty() {
        if !tool_calls.is_empty() {
            return crate::turn::completion_checks::build_activity_summary_reply(tool_calls);
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
        crate::turn::completion_checks::build_activity_summary_reply(tool_calls)
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

/// The typed reason a declared item ends with when nothing took it on.
pub const UNATTEMPTED_WORK_UNMET: &str = "not attempted";

/// The reply's line for declared steps the run ended without attempting,
/// in the steps' own words.
pub fn not_attempted_note(steps: &[String]) -> Option<String> {
    let steps: Vec<&str> = steps
        .iter()
        .map(|step| step.trim().trim_end_matches('.'))
        .filter(|step| !step.is_empty())
        .collect();
    match steps.as_slice() {
        [] => None,
        [one] => Some(format!("I didn't start this part: {one}.")),
        many => Some(format!("I didn't start these parts: {}.", many.join("; "))),
    }
}

/// What the closeout does with a run about to stop, or hand off, with
/// declared work nothing took on ([`aidaemon_events::run_aggregate::RunAggregate::declared_work_unattempted`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnattemptedWorkVerdict {
    /// Nothing to hold: work was attempted, none was declared, the request
    /// forbids changes, the run is not about to stop, or the owner was asked.
    NotApplicable,
    /// Send the run back once: do the work, or ask the owner.
    Demand,
    /// The pass was spent and the work still was not attempted: the run
    /// stops unmet, with [`UNATTEMPTED_WORK_UNMET`] as each item's reason.
    StopUnmet,
}

/// Judge a stop from typed facts only: whether the declared work was
/// attempted, the agent's verdict, whether the closeout would let the run
/// stop, whether the request permits changes, and whether the one pass was
/// used. An unfinished verdict or a deferral is honoured once work was tried
/// or something typed stood in the way; with nothing tried and nothing in
/// the way, the request is carried out or the owner is asked
/// (`needs_owner`), never closed on a report of earlier work.
pub fn unattempted_work_verdict(
    work_unattempted: bool,
    claim: Option<aidaemon_events::ExecutorClaimedOutcome>,
    closeout_lets_stop: bool,
    changes_permitted: bool,
    demanded: bool,
) -> UnattemptedWorkVerdict {
    let asks_owner = claim == Some(aidaemon_events::ExecutorClaimedOutcome::NeedsOwner);
    if !work_unattempted || !changes_permitted || asks_owner || !closeout_lets_stop {
        return UnattemptedWorkVerdict::NotApplicable;
    }
    if demanded {
        UnattemptedWorkVerdict::StopUnmet
    } else {
        UnattemptedWorkVerdict::Demand
    }
}

#[cfg(test)]
mod unattempted_work_tests {
    use super::*;
    use aidaemon_events::ExecutorClaimedOutcome as Claim;

    /// Live 2026-11-13: two installs declared, only a listing run, then a
    /// `blocked` verdict. The run goes back once, then stops unmet.
    #[test]
    fn an_unfinished_verdict_on_unattempted_work_is_sent_back_once() {
        assert_eq!(
            unattempted_work_verdict(true, Some(Claim::Blocked), true, true, false),
            UnattemptedWorkVerdict::Demand
        );
        assert_eq!(
            unattempted_work_verdict(true, Some(Claim::Partial), true, true, true),
            UnattemptedWorkVerdict::StopUnmet
        );
        // A deferral with no verdict is the same stop.
        assert_eq!(
            unattempted_work_verdict(true, None, true, true, false),
            UnattemptedWorkVerdict::Demand
        );
    }

    #[test]
    fn asking_the_owner_attempted_work_and_read_only_requests_are_honoured() {
        assert_eq!(
            unattempted_work_verdict(true, Some(Claim::NeedsOwner), true, true, false),
            UnattemptedWorkVerdict::NotApplicable
        );
        assert_eq!(
            unattempted_work_verdict(false, Some(Claim::Blocked), true, true, false),
            UnattemptedWorkVerdict::NotApplicable
        );
        assert_eq!(
            unattempted_work_verdict(true, Some(Claim::Blocked), true, false, false),
            UnattemptedWorkVerdict::NotApplicable
        );
        // Reachable work is demanded by the ledger closeout itself.
        assert_eq!(
            unattempted_work_verdict(true, None, false, true, false),
            UnattemptedWorkVerdict::NotApplicable
        );
    }
}
