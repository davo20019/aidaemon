//! The SQLite implementation of [`MandateStore`]: each method's body lives
//! in the child module for its concern, and the shared row decoding,
//! validation and transitions in their own.

use super::*;
use crate::traits::{
    GoalRun, Intention, IntentionStatus, Mandate, MandateActionOutcome, MandateActivityLevel,
    MandateAuthority, MandateAutonomyMode, MandateDecisionCycle, MandateDecisionOrigin,
    MandateDecisionOutcome, MandateDispatchGate, MandateFinalizationRejectReason,
    MandateFinalizationStaleReason, MandateLearningNote, MandateMutationAttempt,
    MandateMutationAttemptStatus, MandateMutationDispatchClaim, MandateMutationEvidence,
    MandateMutationQuotaBlockReason, MandateMutationQuotaState, MandateMutationReservation,
    MandateObjectiveMeasurement, MandateOperatingUpdates, MandateReconciliationReason,
    MandateReconciliationResolution, MandateRunFinalizationRequest, MandateRunFinalizationResult,
    MandateRunProofCounts, MandateStatus, MandateStore, MandateSuspension, MandateSuspensionKind,
    MandateTerminationKind, MandateWakeSignal, Task,
};
use sha2::{Digest, Sha256};

mod controller;
mod decisions;
mod finalize;
mod measurements;
mod mutations;
mod recovery;
mod reviews;
mod rows;
mod status;
mod tasks;
#[cfg(test)]
mod tests;
mod transitions;
mod validate;

use mutations::*;
use recovery::*;
use rows::*;
use transitions::*;
use validate::*;

#[async_trait]
impl<S: WorkDatabase> MandateStore for S {
    async fn create_mandate_controller(
        &self,
        goal: &Goal,
        mandate: &Mandate,
    ) -> anyhow::Result<()> {
        controller::create_mandate_controller(self, goal, mandate).await
    }

    async fn get_mandate(&self, id: &str) -> anyhow::Result<Option<Mandate>> {
        controller::get_mandate(self, id).await
    }

    async fn get_mandate_for_goal(&self, goal_id: &str) -> anyhow::Result<Option<Mandate>> {
        controller::get_mandate_for_goal(self, goal_id).await
    }

    async fn list_mandates(
        &self,
        session_id: Option<&str>,
        include_terminal: bool,
    ) -> anyhow::Result<Vec<Mandate>> {
        controller::list_mandates(self, session_id, include_terminal).await
    }

    async fn is_mandate_session_authorized(
        &self,
        mandate_id: &str,
        session_id: &str,
    ) -> anyhow::Result<bool> {
        controller::is_mandate_session_authorized(self, mandate_id, session_id).await
    }

    async fn update_mandate(&self, mandate: &Mandate) -> anyhow::Result<()> {
        controller::update_mandate(self, mandate).await
    }

    async fn confirm_mandate(
        &self,
        mandate_id: &str,
        expected_version: i64,
        activation_duration_secs: Option<i64>,
    ) -> anyhow::Result<bool> {
        controller::confirm_mandate(self, mandate_id, expected_version, activation_duration_secs)
            .await
    }

    async fn transition_mandate_status(
        &self,
        mandate_id: &str,
        from_status: MandateStatus,
        to_status: MandateStatus,
    ) -> anyhow::Result<bool> {
        status::transition_mandate_status(self, mandate_id, from_status, to_status).await
    }

    async fn resume_mandate_with_context(
        &self,
        mandate_id: &str,
        from_status: MandateStatus,
        expected_version: i64,
        controller_context: Option<&str>,
    ) -> anyhow::Result<bool> {
        status::resume_mandate_with_context(
            self,
            mandate_id,
            from_status,
            expected_version,
            controller_context,
        )
        .await
    }

    async fn record_active_mandate_owner_guidance(
        &self,
        mandate_id: &str,
        expected_version: i64,
        owner_session: &str,
        controller_context: &str,
    ) -> anyhow::Result<bool> {
        status::record_active_mandate_owner_guidance(
            self,
            mandate_id,
            expected_version,
            owner_session,
            controller_context,
        )
        .await
    }

    async fn keep_mandate_controller_active(
        &self,
        mandate_id: &str,
        expected_version: i64,
    ) -> anyhow::Result<bool> {
        status::keep_mandate_controller_active(self, mandate_id, expected_version).await
    }

    async fn wake_mandates_for_signal(
        &self,
        signal: &MandateWakeSignal,
    ) -> anyhow::Result<Vec<String>> {
        status::wake_mandates_for_signal(self, signal).await
    }

    async fn resolve_mandate_suspension(
        &self,
        mandate_id: &str,
        expected_version: i64,
        expected_kind: MandateSuspensionKind,
        controller_context: Option<&str>,
        reconciliation_resolution: Option<MandateReconciliationResolution>,
        owner_guidance: &str,
        owner_session: &str,
    ) -> anyhow::Result<bool> {
        status::resolve_mandate_suspension(
            self,
            mandate_id,
            expected_version,
            expected_kind,
            controller_context,
            reconciliation_resolution,
            owner_guidance,
            owner_session,
        )
        .await
    }

    async fn record_mandate_objective_measurement(
        &self,
        measurement: &MandateObjectiveMeasurement,
    ) -> anyhow::Result<()> {
        measurements::record_mandate_objective_measurement(self, measurement).await
    }

    async fn list_mandate_objective_measurements(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateObjectiveMeasurement>> {
        measurements::list_mandate_objective_measurements(self, mandate_id, limit).await
    }

    async fn record_mandate_learning_note(&self, note: &MandateLearningNote) -> anyhow::Result<()> {
        measurements::record_mandate_learning_note(self, note).await
    }

    async fn list_mandate_learning_notes(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateLearningNote>> {
        measurements::list_mandate_learning_notes(self, mandate_id, limit).await
    }

    async fn list_mandate_action_outcomes(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateActionOutcome>> {
        decisions::list_mandate_action_outcomes(self, mandate_id, limit).await
    }

    async fn create_mandate_review_run(
        &self,
        mandate_id: &str,
        review_lease_token: &str,
        goal_run_id: &str,
        root_task: &Task,
    ) -> anyhow::Result<GoalRun> {
        reviews::create_mandate_review_run(
            self,
            mandate_id,
            review_lease_token,
            goal_run_id,
            root_task,
        )
        .await
    }

    async fn claim_due_mandates(
        &self,
        limit: i64,
        lease_owner: &str,
        lease_secs: i64,
    ) -> anyhow::Result<Vec<Mandate>> {
        reviews::claim_due_mandates(self, limit, lease_owner, lease_secs).await
    }

    async fn release_mandate_review_lease(
        &self,
        mandate_id: &str,
        lease_token: &str,
        retry_at: &str,
    ) -> anyhow::Result<bool> {
        reviews::release_mandate_review_lease(self, mandate_id, lease_token, retry_at).await
    }

    async fn create_mandate_task_from_attempt(
        &self,
        task: &Task,
        mandate_id: &str,
        mandate_version: i64,
        goal_run_id: &str,
        root_task_attempt_id: &str,
        max_non_root_tasks: i64,
    ) -> anyhow::Result<bool> {
        tasks::create_mandate_task_from_attempt(
            self,
            task,
            mandate_id,
            mandate_version,
            goal_run_id,
            root_task_attempt_id,
            max_non_root_tasks,
        )
        .await
    }

    async fn claim_mandate_task_from_attempt(
        &self,
        task_id: &str,
        worker_instance_id: &str,
        mandate_id: &str,
        mandate_version: i64,
        goal_run_id: &str,
        root_task_attempt_id: &str,
        lease_secs: i64,
    ) -> anyhow::Result<Option<crate::traits::TaskAttempt>> {
        tasks::claim_mandate_task_from_attempt(
            self,
            task_id,
            worker_instance_id,
            mandate_id,
            mandate_version,
            goal_run_id,
            root_task_attempt_id,
            lease_secs,
        )
        .await
    }

    async fn record_mandate_decision(
        &self,
        decision: &MandateDecisionCycle,
        intention: Option<&Intention>,
        task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        decisions::record_mandate_decision(self, decision, intention, task_attempt_id).await
    }

    async fn record_mandate_decision_with_updates(
        &self,
        decision: &MandateDecisionCycle,
        intention: Option<&Intention>,
        operating_updates: Option<&MandateOperatingUpdates>,
        task_attempt_id: Option<&str>,
    ) -> anyhow::Result<()> {
        decisions::record_mandate_decision_with_updates(
            self,
            decision,
            intention,
            operating_updates,
            task_attempt_id,
        )
        .await
    }

    async fn get_mandate_decision_for_run(
        &self,
        goal_run_id: &str,
    ) -> anyhow::Result<Option<MandateDecisionCycle>> {
        decisions::get_mandate_decision_for_run(self, goal_run_id).await
    }

    async fn current_run_evidence_receipt_ids(
        &self,
        goal_run_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<String>> {
        decisions::current_run_evidence_receipt_ids(self, goal_run_id, limit).await
    }

    async fn list_mandate_decisions(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<MandateDecisionCycle>> {
        decisions::list_mandate_decisions(self, mandate_id, limit).await
    }

    async fn list_intentions(
        &self,
        mandate_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<Intention>> {
        decisions::list_intentions(self, mandate_id, limit).await
    }

    async fn reserve_mandate_action_attempt(
        &self,
        reservation: &MandateMutationReservation,
    ) -> anyhow::Result<Option<MandateMutationAttempt>> {
        mutations::reserve_mandate_action_attempt(self, reservation).await
    }

    async fn claim_mandate_mutation_dispatch(
        &self,
        claim: &MandateMutationDispatchClaim,
    ) -> anyhow::Result<bool> {
        mutations::claim_mandate_mutation_dispatch(self, claim).await
    }

    async fn get_mandate_mutation_quota_state(
        &self,
        mandate_id: &str,
        as_of: &str,
    ) -> anyhow::Result<Option<MandateMutationQuotaState>> {
        mutations::get_mandate_mutation_quota_state(self, mandate_id, as_of).await
    }

    async fn list_mandate_mutation_attempts_for_run(
        &self,
        goal_run_id: &str,
    ) -> anyhow::Result<Vec<MandateMutationAttempt>> {
        mutations::list_mandate_mutation_attempts_for_run(self, goal_run_id).await
    }

    async fn finalize_mandate_run_from_proof(
        &self,
        request: &MandateRunFinalizationRequest,
    ) -> anyhow::Result<MandateRunFinalizationResult> {
        finalize::finalize_mandate_run_from_proof(self, request).await
    }
}
