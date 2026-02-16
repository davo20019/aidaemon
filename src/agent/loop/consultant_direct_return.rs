use super::consultant_pass::mark_consultant_direct_return;
use super::consultant_phase::ConsultantPhaseOutcome;

pub(super) fn consultant_direct_return_ok(reply: String) -> ConsultantPhaseOutcome {
    mark_consultant_direct_return();
    ConsultantPhaseOutcome::Return(Ok(reply))
}
