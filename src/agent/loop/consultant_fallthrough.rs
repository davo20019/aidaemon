use super::consultant_pass::mark_consultant_fallthrough;
use super::consultant_phase::ConsultantPhaseOutcome;

pub(super) fn consultant_fallthrough() -> ConsultantPhaseOutcome {
    mark_consultant_fallthrough();
    ConsultantPhaseOutcome::ContinueLoop
}
