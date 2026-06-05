mod budget;
mod counters;
mod directives;
mod evidence;
mod failure;
mod recovery;
mod reflection;
mod stall;

pub(super) use budget::{BudgetTracker, IterationLimitSettings};
pub(super) use counters::LoopCounters;
pub(super) use directives::PendingDirectives;
pub(super) use evidence::EvidenceLedger;
pub(super) use failure::FailureLedger;
pub(super) use recovery::RecoveryState;
pub(super) use reflection::ReflectionState;
pub(super) use stall::StallTracker;

#[derive(Debug, Default)]
pub(super) struct TurnState {
    pub stall: StallTracker,
    pub failures: FailureLedger,
    pub recovery: RecoveryState,
    pub budget: BudgetTracker,
    pub evidence: EvidenceLedger,
    pub reflection: ReflectionState,
    pub directives: PendingDirectives,
    pub counters: LoopCounters,
}
