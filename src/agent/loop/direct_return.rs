use super::response_phase::ResponsePhaseOutcome;
use super::POLICY_METRICS;
use std::sync::atomic::Ordering;

pub(super) fn direct_return_ok(reply: String) -> ResponsePhaseOutcome {
    POLICY_METRICS
        .response_direct_return_total
        .fetch_add(1, Ordering::Relaxed);
    ResponsePhaseOutcome::Return(Ok(reply))
}
