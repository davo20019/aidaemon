//! The turn's task end and decision points, recorded through
//! [`crate::turn::task_end`] on this engine's host.

use super::*;
use crate::turn::task_end;
use aidaemon_events::{DecisionType, EventEmitter, TaskOutcome, TaskStatus};

impl TurnEngine<'_> {
    /// Emit a TaskEnd event. Called from every exit path in the turn.
    #[allow(clippy::too_many_arguments)]
    pub async fn emit_task_end(
        &self,
        emitter: &EventEmitter,
        task_id: &str,
        status: TaskStatus,
        outcome: TaskOutcome,
        task_start: Instant,
        iteration: usize,
        tool_calls_count: usize,
        error: Option<String>,
        summary: Option<String>,
    ) {
        task_end::emit_task_end(
            self.host,
            emitter,
            task_id,
            status,
            outcome,
            task_start,
            iteration,
            tool_calls_count,
            error,
            summary,
        )
        .await;
    }

    /// This task's model-call cost and efficiency.
    pub async fn task_efficiency_data(
        &self,
        task_id: &str,
    ) -> Option<aidaemon_events::TaskEfficiencyData> {
        task_end::task_efficiency_data(self.host, task_id).await
    }

    pub async fn emit_decision_point(
        &self,
        emitter: &EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: DecisionType,
        summary: impl Into<String>,
        metadata: Value,
    ) {
        task_end::emit_decision_point(
            self.host,
            emitter,
            task_id,
            iteration,
            decision_type,
            summary,
            metadata,
        )
        .await;
    }

    pub async fn emit_warning_decision_point(
        &self,
        emitter: &EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: DecisionType,
        summary: impl Into<String>,
        metadata: Value,
    ) {
        task_end::emit_warning_decision_point(
            self.host,
            emitter,
            task_id,
            iteration,
            decision_type,
            summary,
            metadata,
        )
        .await;
    }
}
