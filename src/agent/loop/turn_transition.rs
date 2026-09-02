/// Why the per-turn driver is starting another iteration.
///
/// Phase-local outcome enums remain intentionally small, while this shared
/// vocabulary gives `main_loop` one typed control-flow boundary.  The reason is
/// control state only; it is never added to the model transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TurnRestartReason {
    StoppingPhaseControl,
    ApproachPivot {
        failure_record: String,
    },
    /// A stall after clean progress hands the model one tool-less closeout
    /// pass; the stall evidence belonged to the tool-calling approach that
    /// this pass ends.
    StallForceTextCloseout,
    LlmPhaseRecovery,
    ResponsePhaseRecovery,
    ToolPreludeRecovery,
    ToolExecutionCompleted,
}

/// Common transition shape produced by each phase at the turn-driver boundary.
pub(super) enum TurnTransition<T> {
    Restart(TurnRestartReason),
    Finish(anyhow::Result<String>),
    Advance(T),
}
