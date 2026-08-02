/// Why the per-turn driver is starting another iteration.
///
/// Phase-local outcome enums remain intentionally small, while this shared
/// vocabulary gives `main_loop` one typed control-flow boundary.  The reason is
/// control state only; it is never added to the model transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TurnRestartReason {
    StoppingPhaseControl,
    ApproachPivot { failure_record: String },
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
