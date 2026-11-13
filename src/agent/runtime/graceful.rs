use super::*;
#[cfg(test)]
use aidaemon_runtime::turn::budget;
use aidaemon_runtime::turn::task_end;
pub(in crate::agent) use aidaemon_runtime::turn::task_end::emit_decision_point_event;
#[cfg(test)]
use aidaemon_runtime::turn::task_end::task_end_decision_and_outcome;

// impl-Agent justification: graceful shutdown, budget progress, and task-end hooks over state/event_store.
impl Agent {
    pub(super) async fn run_task_end_tool_hooks(&self, task_id: &str, session_id: &str) {
        task_end::run_task_end_tool_hooks(self, task_id, session_id).await;
    }

    /// Test-only wrapper around `post_task::classify_stall`.
    ///
    /// Production flow should call the `post_task` function with the real
    /// `tool_failure_count` map so lockout classification remains available.
    #[allow(dead_code)] // Used in tests; production path delegates through post_task.
    pub(super) fn classify_stall(learning_ctx: &LearningContext) -> (&'static str, &'static str) {
        let empty_tool_failure_count: HashMap<String, usize> = HashMap::new();
        post_task::classify_stall(
            learning_ctx,
            DEFERRED_NO_TOOL_ERROR_MARKER,
            &empty_tool_failure_count,
        )
    }

    /// Emit TaskEnd after recording an exact bootstrap protocol direct-return.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn emit_direct_return_task_end(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        status: TaskStatus,
        outcome: crate::events::TaskOutcome,
        task_start: Instant,
        iteration: usize,
        tool_calls_count: usize,
        error: Option<String>,
        summary: Option<String>,
        direct_return_succeeded: bool,
    ) {
        task_end::emit_direct_return_task_end(
            self,
            emitter,
            task_id,
            status,
            outcome,
            task_start,
            iteration,
            tool_calls_count,
            error,
            summary,
            direct_return_succeeded,
        )
        .await;
    }

    /// Emit a TaskEnd event. Called from every exit path in the agent loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn emit_task_end(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        status: TaskStatus,
        outcome: crate::events::TaskOutcome,
        task_start: Instant,
        iteration: usize,
        tool_calls_count: usize,
        error: Option<String>,
        summary: Option<String>,
    ) {
        task_end::emit_task_end(
            self,
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
    pub(in crate::agent) async fn task_efficiency_data(
        &self,
        task_id: &str,
    ) -> Option<crate::events::TaskEfficiencyData> {
        task_end::task_efficiency_data(self, task_id).await
    }

    pub(in crate::agent) async fn emit_decision_point(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: DecisionType,
        summary: impl Into<String>,
        metadata: Value,
    ) {
        task_end::emit_decision_point(
            self,
            emitter,
            task_id,
            iteration,
            decision_type,
            summary,
            metadata,
        )
        .await;
    }

    pub(super) async fn emit_warning_decision_point(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        decision_type: DecisionType,
        summary: impl Into<String>,
        metadata: Value,
    ) {
        task_end::emit_warning_decision_point(
            self,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meaningful_budget_progress_accepts_evidence_gain() {
        assert!(budget::has_meaningful_budget_progress(1, 0));
    }

    #[test]
    fn meaningful_budget_progress_rejects_transport_success_without_evidence() {
        assert!(!budget::has_meaningful_budget_progress(0, 30));
    }

    #[test]
    fn meaningful_budget_progress_rejects_shallow_runs_without_evidence() {
        assert!(!budget::has_meaningful_budget_progress(0, 2));
    }

    #[test]
    fn scheduled_run_metrics_detect_unproductive_snapshot() {
        assert!(budget::scheduled_run_metrics_are_clearly_unproductive(
            &crate::traits::ScheduledRunHealth {
                evidence_gain_count: 0,
                total_successful_tool_calls: 0,
                stall_count: 0,
                consecutive_same_tool_count: 0,
                consecutive_same_tool_unique_args: 0,
                unrecovered_error_count: 1,
                ..Default::default()
            }
        ));
    }

    #[test]
    fn scheduled_run_auto_extension_candidate_requires_health() {
        assert_eq!(
            budget::scheduled_run_auto_extension_candidate(
                &crate::goal_tokens::GoalRunBudgetStatus {
                    effective_budget_per_check: 100,
                    tokens_used: 100,
                    budget_extensions_count: 0,
                    health: crate::traits::ScheduledRunHealth::default(),
                },
                12,
                1_000,
            ),
            None
        );
    }

    #[test]
    fn scheduled_run_auto_extension_candidate_accepts_required_mutation_receipt() {
        assert_eq!(
            budget::scheduled_run_auto_extension_candidate(
                &crate::goal_tokens::GoalRunBudgetStatus {
                    effective_budget_per_check: 100,
                    tokens_used: 100,
                    budget_extensions_count: 0,
                    health: crate::traits::ScheduledRunHealth {
                        evidence_gain_count: 1,
                        total_successful_tool_calls: 3,
                        completion_requires_mutation: true,
                        required_mutation_progress: true,
                        stall_count: 0,
                        consecutive_same_tool_count: 1,
                        consecutive_same_tool_unique_args: 1,
                        unrecovered_error_count: 0,
                        ..Default::default()
                    },
                },
                12,
                1_000,
            ),
            Some(200)
        );
    }

    #[test]
    fn scheduled_run_auto_extension_rejects_read_only_activity_for_mutation_task() {
        let status = crate::goal_tokens::GoalRunBudgetStatus {
            effective_budget_per_check: 400_000,
            tokens_used: 400_000,
            budget_extensions_count: 0,
            health: crate::traits::ScheduledRunHealth {
                evidence_gain_count: 28,
                total_successful_tool_calls: 30,
                completion_requires_mutation: true,
                required_mutation_progress: false,
                stall_count: 0,
                consecutive_same_tool_count: 1,
                consecutive_same_tool_unique_args: 1,
                unrecovered_error_count: 0,
                ..Default::default()
            },
        };

        assert_eq!(
            budget::scheduled_run_auto_extension_candidate(&status, 1, 2_000_000),
            None
        );
    }

    #[test]
    fn scheduled_run_auto_extension_accepts_verified_research_progress() {
        let status = crate::goal_tokens::GoalRunBudgetStatus {
            effective_budget_per_check: 100,
            tokens_used: 100,
            budget_extensions_count: 0,
            health: crate::traits::ScheduledRunHealth {
                evidence_gain_count: 2,
                total_successful_tool_calls: 2,
                completion_requires_observation: true,
                verification_progress: true,
                ..Default::default()
            },
        };

        assert_eq!(
            budget::scheduled_run_auto_extension_candidate(&status, 1, 1_000),
            Some(200)
        );
    }

    #[test]
    fn scheduled_run_auto_extension_candidate_respects_autonomous_extension_limit() {
        assert_eq!(
            budget::scheduled_run_auto_extension_candidate(
                &crate::goal_tokens::GoalRunBudgetStatus {
                    effective_budget_per_check: 200,
                    tokens_used: 200,
                    budget_extensions_count: SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS,
                    health: crate::traits::ScheduledRunHealth {
                        evidence_gain_count: 2,
                        total_successful_tool_calls: 4,
                        completion_requires_mutation: true,
                        required_mutation_progress: true,
                        stall_count: 0,
                        consecutive_same_tool_count: 1,
                        consecutive_same_tool_unique_args: 1,
                        unrecovered_error_count: 0,
                        ..Default::default()
                    },
                },
                SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS,
                SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
            ),
            None
        );
    }

    #[test]
    fn scheduled_run_budget_pressure_fires_once_before_exhaustion() {
        let status = crate::goal_tokens::GoalRunBudgetStatus {
            effective_budget_per_check: 400_000,
            tokens_used: 320_000,
            budget_extensions_count: 0,
            health: crate::traits::ScheduledRunHealth::default(),
        };
        assert_eq!(
            budget::scheduled_run_budget_pressure_pct(&status, false),
            Some(80)
        );
        assert_eq!(
            budget::scheduled_run_budget_pressure_pct(&status, true),
            None
        );

        let exhausted = crate::goal_tokens::GoalRunBudgetStatus {
            tokens_used: 400_000,
            ..status
        };
        assert_eq!(
            budget::scheduled_run_budget_pressure_pct(&exhausted, false),
            None
        );
    }

    #[test]
    fn alert_destination_dedupes_legacy_and_namespaced_telegram_sessions() {
        assert!(budget::same_alert_destination(
            "301753035",
            "aidaemon_coding_bot:301753035"
        ));
        assert!(!budget::same_alert_destination(
            "slack:U018EAFV5QR",
            "aidaemon_coding_bot:301753035"
        ));
    }

    /// A run with no contract whose only operation was a typed policy
    /// denial (nothing dispatched).
    fn denial_only_aggregate() -> crate::events::RunAggregate {
        let mut aggregate = crate::events::RunAggregate::new("task-denied");
        let operation = serde_json::from_value(serde_json::json!({
            "operation_id": "op-denied",
            "tool_name": "write_file",
            "policy_denied": true,
            "mutating": true
        }))
        .expect("operation");
        aggregate
            .operations
            .insert("op-denied".to_string(), operation);
        assert!(aggregate.terminal_policy_denial());
        assert_eq!(
            aggregate.terminal_decision(),
            crate::events::RunTerminalDecision::Unspecified
        );
        aggregate
    }

    #[test]
    fn policy_denial_closes_an_uncredited_success_at_the_boundary() {
        let (decision, outcome) = task_end_decision_and_outcome(
            &denial_only_aggregate(),
            TaskStatus::Completed,
            crate::events::TaskOutcome::Succeeded,
            false,
        );
        assert_eq!(
            decision,
            crate::events::RunTerminalDecision::ClosedByPolicyDenial
        );
        assert_eq!(outcome, crate::events::TaskOutcome::Partial);
    }

    #[test]
    fn refused_work_beside_a_fulfilled_contract_is_not_a_success() {
        // The contract's only obligation was satisfied by a read; the refused
        // call was never in the contract. The run's decision is therefore
        // `Succeeded`, and the refusal must still keep the request open.
        let mut aggregate = denial_only_aggregate();
        aggregate.contract_present = true;
        let obligation = serde_json::from_value(serde_json::json!({
            "id": "obligation-read",
            "class": "perform",
            "state": "satisfied"
        }))
        .expect("obligation");
        aggregate
            .obligations
            .insert("obligation-read".to_string(), obligation);
        assert_eq!(
            aggregate.terminal_decision(),
            crate::events::RunTerminalDecision::Succeeded
        );
        let (_, outcome) = task_end_decision_and_outcome(
            &aggregate,
            TaskStatus::Completed,
            crate::events::TaskOutcome::Succeeded,
            false,
        );
        assert_eq!(outcome, crate::events::TaskOutcome::Partial);
    }

    #[test]
    fn refused_work_beside_dispatched_work_is_not_a_success() {
        // The refused call is owed work; the dispatched read is not proof that
        // it happened. The run must not close as an answered success.
        let mut aggregate = denial_only_aggregate();
        let read = serde_json::from_value(serde_json::json!({
            "operation_id": "op-read",
            "tool_name": "search_memories",
            "dispatched": true
        }))
        .expect("operation");
        aggregate.operations.insert("op-read".to_string(), read);
        let (decision, outcome) = task_end_decision_and_outcome(
            &aggregate,
            TaskStatus::Completed,
            crate::events::TaskOutcome::Succeeded,
            false,
        );
        assert_eq!(
            decision,
            crate::events::RunTerminalDecision::Unspecified,
            "a mixed run is not closed by the denial-only boundary rule"
        );
        assert_eq!(outcome, crate::events::TaskOutcome::Partial);
    }

    #[test]
    fn policy_denial_does_not_turn_a_failed_run_into_success() {
        let aggregate = denial_only_aggregate();
        for (status, proposed) in [
            (TaskStatus::Failed, crate::events::TaskOutcome::Failed),
            (TaskStatus::Completed, crate::events::TaskOutcome::Failed),
            (TaskStatus::Completed, crate::events::TaskOutcome::Partial),
            (TaskStatus::Interrupted, crate::events::TaskOutcome::Partial),
        ] {
            let (decision, outcome) =
                task_end_decision_and_outcome(&aggregate, status, proposed, false);
            assert_eq!(
                decision,
                crate::events::RunTerminalDecision::Unspecified,
                "{status:?}/{proposed:?}"
            );
            assert_eq!(outcome, proposed, "{status:?}/{proposed:?}");
        }
    }

    #[test]
    fn policy_denial_does_not_close_a_cancelled_run() {
        let (decision, outcome) = task_end_decision_and_outcome(
            &denial_only_aggregate(),
            TaskStatus::Cancelled,
            crate::events::TaskOutcome::Succeeded,
            false,
        );
        assert_ne!(
            decision,
            crate::events::RunTerminalDecision::ClosedByPolicyDenial
        );
        assert_eq!(outcome, crate::events::TaskOutcome::Failed);
    }
}
