//! Where a turn's decision points go, below the host and the modules that
//! record one.

/// Where a turn's decision points go: the record of why the runtime chose
/// what it did, kept with the turn's events when the host records them.
#[async_trait::async_trait]
pub trait DecisionRecorder: Send + Sync {
    /// Record one decision point of `task_id` at loop `iteration`.
    async fn record_decision_point(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: aidaemon_events::DecisionType,
        summary: String,
        metadata: serde_json::Value,
    );
}
