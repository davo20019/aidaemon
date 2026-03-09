use super::response_phase::ResponsePhaseOutcome;
use super::POLICY_METRICS;
use std::sync::atomic::Ordering;

pub(super) fn fallthrough() -> ResponsePhaseOutcome {
    POLICY_METRICS
        .response_fallthrough_total
        .fetch_add(1, Ordering::Relaxed);
    ResponsePhaseOutcome::ContinueLoop
}
