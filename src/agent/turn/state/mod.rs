mod budget;
mod counters;
mod directives;
mod evidence;
mod failure;
mod read_file_tracker;
mod recovery;
mod reflection;
mod stall;

pub(super) use budget::{BudgetTracker, IterationLimitSettings};
pub(super) use counters::LoopCounters;
pub(super) use directives::PendingDirectives;
pub(super) use evidence::EvidenceLedger;
pub(super) use failure::FailureLedger;
pub(super) use read_file_tracker::{
    canonical_path_from_arguments, LineInterval, ReadDecision, ReadFileObservationTracker,
    ReadRequest,
};
pub(super) use recovery::RecoveryState;
pub(super) use reflection::ReflectionState;
pub(super) use stall::StallTracker;

use crate::agent::eval::{HarnessEvalAccumulator, HarnessEvalHandle};

/// Per-turn loop state. `harness_eval` shares the agent's accumulator handle
/// (single source of truth for phase instrumentation through TaskEnd finalize).
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
    pub read_files: ReadFileObservationTracker,
    pub harness_eval: Option<HarnessEvalHandle>,
    /// Alias for plan terminology — same handle as `harness_eval`.
    pub eval: Option<HarnessEvalHandle>,
}

impl TurnState {
    pub(super) fn attach_harness_eval(&mut self, handle: HarnessEvalHandle) {
        self.harness_eval = Some(handle.clone());
        self.eval = Some(handle);
    }

    pub(super) async fn with_harness_eval<R>(
        &self,
        f: impl FnOnce(&mut HarnessEvalAccumulator) -> R,
    ) -> Option<R> {
        if let Some(handle) = &self.harness_eval {
            handle.write().await.as_mut().map(f)
        } else {
            None
        }
    }
}
