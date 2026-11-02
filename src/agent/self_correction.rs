use std::sync::Arc;

use crate::traits::{attempt_status, SelfCorrectionAttempt, SelfCorrectionSubjectKind, StateStore};

/// Outcome of asking the controller whether a correction attempt may proceed.
#[derive(Debug, PartialEq, Eq)]
pub enum AttemptDecision {
    /// Proceed; this is attempt number `attempt_index` (1-based) for the subject.
    Proceed { attempt_index: usize },
    /// This exact approach already failed/was blocked — do not retry it.
    StopRepeat,
    /// K distinct approaches already failed — stop and give up.
    StopBudget,
}

/// Pure attempt policy. See module docs: repeat-block known-bad approaches,
/// then enforce the K distinct-failure budget, else proceed.
#[allow(dead_code)] // Used in Task 5+
pub fn decide_attempt(
    prior: &[SelfCorrectionAttempt],
    signature: &str,
    k: usize,
) -> AttemptDecision {
    let is_failed = |s: &str| s == attempt_status::FAILED || s == attempt_status::BLOCKED;

    if prior
        .iter()
        .any(|a| a.approach_signature == signature && is_failed(&a.status))
    {
        return AttemptDecision::StopRepeat;
    }

    let mut distinct_failures = std::collections::HashSet::new();
    for a in prior.iter().filter(|a| is_failed(&a.status)) {
        distinct_failures.insert(a.approach_signature.as_str());
    }
    if distinct_failures.len() >= k {
        return AttemptDecision::StopBudget;
    }

    AttemptDecision::Proceed {
        attempt_index: prior.len() + 1,
    }
}

/// Bounded, durable policy engine for self-correction. Library only: it decides
/// whether an attempt may proceed and persists attempt outcomes; it never spawns
/// or executes anything.
pub struct SelfCorrectionController {
    state: Arc<dyn StateStore>,
    max_attempts: usize,
}

impl SelfCorrectionController {
    pub fn new(state: Arc<dyn StateStore>, max_attempts: usize) -> Self {
        Self {
            state,
            max_attempts,
        }
    }

    /// Decide whether a correction attempt with `signature` may proceed for the
    /// subject. Pure policy applied over the durable attempt history.
    #[allow(dead_code)] // Used in Task 5+
    pub async fn attempt(
        &self,
        subject_id: &str,
        _kind: SelfCorrectionSubjectKind,
        signature: &str,
    ) -> anyhow::Result<AttemptDecision> {
        let prior = self.state.get_self_correction_attempts(subject_id).await?;
        Ok(decide_attempt(&prior, signature, self.max_attempts))
    }

    #[allow(clippy::too_many_arguments)]
    async fn record(
        &self,
        subject_id: &str,
        kind: SelfCorrectionSubjectKind,
        signature: &str,
        attempt_index: usize,
        status: &str,
        blocked_reason: Option<&str>,
        evidence_ref: Option<&str>,
    ) -> anyhow::Result<()> {
        let attempt = SelfCorrectionAttempt {
            id: 0,
            subject_id: subject_id.to_string(),
            subject_kind: kind.as_str().to_string(),
            approach_signature: signature.to_string(),
            attempt_index: attempt_index as i64,
            status: status.to_string(),
            blocked_reason: blocked_reason.map(str::to_string),
            evidence_ref: evidence_ref.map(str::to_string),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        self.state.record_self_correction_attempt(&attempt).await
    }

    pub async fn record_failure(
        &self,
        subject_id: &str,
        kind: SelfCorrectionSubjectKind,
        signature: &str,
        attempt_index: usize,
        evidence_ref: Option<&str>,
    ) -> anyhow::Result<()> {
        let sig = crate::tools::sanitize::redact_secrets(signature);
        let eref = evidence_ref.map(crate::tools::sanitize::redact_secrets);
        self.record(
            subject_id,
            kind,
            &sig,
            attempt_index,
            attempt_status::FAILED,
            None,
            eref.as_deref(),
        )
        .await
    }

    #[allow(dead_code)] // Used in Task 5+
    pub async fn record_success(
        &self,
        subject_id: &str,
        kind: SelfCorrectionSubjectKind,
        signature: &str,
        attempt_index: usize,
    ) -> anyhow::Result<()> {
        let sig = crate::tools::sanitize::redact_secrets(signature);
        self.record(
            subject_id,
            kind,
            &sig,
            attempt_index,
            attempt_status::VERIFIED_SUCCESS,
            None,
            None,
        )
        .await
    }

    #[allow(dead_code)] // Used in Task 5+
    pub async fn record_blocked(
        &self,
        subject_id: &str,
        kind: SelfCorrectionSubjectKind,
        signature: &str,
        attempt_index: usize,
        blocked_reason: &str,
    ) -> anyhow::Result<()> {
        let sig = crate::tools::sanitize::redact_secrets(signature);
        let reason = crate::tools::sanitize::redact_secrets(blocked_reason);
        self.record(
            subject_id,
            kind,
            &sig,
            attempt_index,
            attempt_status::BLOCKED,
            Some(&reason),
            None,
        )
        .await
    }

    #[allow(dead_code)] // Used in Task 5+
    pub async fn record_executed(
        &self,
        subject_id: &str,
        kind: SelfCorrectionSubjectKind,
        signature: &str,
        attempt_index: usize,
        evidence_ref: Option<&str>,
    ) -> anyhow::Result<()> {
        let sig = crate::tools::sanitize::redact_secrets(signature);
        let eref = evidence_ref.map(crate::tools::sanitize::redact_secrets);
        self.record(
            subject_id,
            kind,
            &sig,
            attempt_index,
            attempt_status::EXECUTED,
            None,
            eref.as_deref(),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::traits::SelfCorrectionSubjectKind;
    use std::sync::Arc;

    async fn temp_state() -> Arc<dyn crate::traits::StateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding = Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
        let store = crate::state::SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding,
        )
        .await
        .unwrap();
        // Keep the temp file alive for the test's duration by leaking it; the OS
        // reclaims on process exit. Tests are short-lived.
        std::mem::forget(db_file);
        Arc::new(store)
    }

    #[tokio::test]
    async fn controller_proceeds_then_blocks_repeat_then_exhausts_budget() {
        let ctrl = SelfCorrectionController::new(temp_state().await, 3);
        let k = SelfCorrectionSubjectKind::Task;

        // 1st distinct approach proceeds.
        assert_eq!(
            ctrl.attempt("t1", k, "sig-a").await.unwrap(),
            AttemptDecision::Proceed { attempt_index: 1 }
        );
        ctrl.record_failure("t1", k, "sig-a", 1, None)
            .await
            .unwrap();

        // Repeating the failed approach is blocked.
        assert_eq!(
            ctrl.attempt("t1", k, "sig-a").await.unwrap(),
            AttemptDecision::StopRepeat
        );

        // Two more distinct failures exhaust the K=3 budget.
        ctrl.record_failure("t1", k, "sig-b", 2, None)
            .await
            .unwrap();
        ctrl.record_failure("t1", k, "sig-c", 3, None)
            .await
            .unwrap();
        assert_eq!(
            ctrl.attempt("t1", k, "sig-d").await.unwrap(),
            AttemptDecision::StopBudget
        );
    }

    fn attempt(sig: &str, status: &str) -> SelfCorrectionAttempt {
        SelfCorrectionAttempt {
            id: 0,
            subject_id: "s".to_string(),
            subject_kind: "task".to_string(),
            approach_signature: sig.to_string(),
            attempt_index: 1,
            status: status.to_string(),
            blocked_reason: None,
            evidence_ref: None,
            created_at: "2026-06-23 00:00:00".to_string(),
        }
    }

    #[test]
    fn first_attempt_proceeds() {
        assert_eq!(
            decide_attempt(&[], "terminal:du -ah ~", 3),
            AttemptDecision::Proceed { attempt_index: 1 }
        );
    }

    #[test]
    fn repeat_of_failed_signature_is_blocked() {
        let prior = vec![attempt("terminal:du -ah ~", attempt_status::FAILED)];
        assert_eq!(
            decide_attempt(&prior, "terminal:du -ah ~", 3),
            AttemptDecision::StopRepeat
        );
    }

    #[test]
    fn distinct_new_approach_proceeds_until_k() {
        let prior = vec![
            attempt("a", attempt_status::FAILED),
            attempt("b", attempt_status::FAILED),
        ];
        assert_eq!(
            decide_attempt(&prior, "c", 3),
            AttemptDecision::Proceed { attempt_index: 3 }
        );
    }

    #[test]
    fn k_distinct_failures_exhausts_budget() {
        let prior = vec![
            attempt("a", attempt_status::FAILED),
            attempt("b", attempt_status::FAILED),
            attempt("c", attempt_status::FAILED),
        ];
        assert_eq!(decide_attempt(&prior, "d", 3), AttemptDecision::StopBudget);
    }

    #[test]
    fn success_rows_do_not_count_toward_budget_or_repeat() {
        let prior = vec![
            attempt("a", attempt_status::VERIFIED_SUCCESS),
            attempt("b", attempt_status::VERIFIED_SUCCESS),
            attempt("c", attempt_status::VERIFIED_SUCCESS),
        ];
        // No failed approaches → budget intact, and retrying "a" is fine.
        assert_eq!(
            decide_attempt(&prior, "a", 3),
            AttemptDecision::Proceed { attempt_index: 4 }
        );
    }

    #[test]
    fn blocked_signature_is_repeat_blocked() {
        let prior = vec![attempt("x", attempt_status::BLOCKED)];
        assert_eq!(decide_attempt(&prior, "x", 3), AttemptDecision::StopRepeat);
    }

    #[test]
    fn gave_up_rows_do_not_count_toward_budget_or_repeat() {
        let prior = vec![
            attempt("a", attempt_status::GAVE_UP),
            attempt("b", attempt_status::GAVE_UP),
            attempt("c", attempt_status::GAVE_UP),
        ];
        // gave_up rows are neither failures nor successes → budget intact, retrying "a" is fine.
        assert_eq!(
            decide_attempt(&prior, "a", 3),
            AttemptDecision::Proceed { attempt_index: 4 }
        );
    }

    // ── P1.3 tests ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_record_blocked_persists_blocked_status_and_reason() {
        let state = temp_state().await;
        let ctrl = SelfCorrectionController::new(state.clone(), 3);
        let k = SelfCorrectionSubjectKind::Task;

        ctrl.record_blocked("b1", k, "sig-blocked", 1, "policy denied")
            .await
            .unwrap();

        let rows = state.get_self_correction_attempts("b1").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, crate::traits::attempt_status::BLOCKED);
        assert_eq!(
            rows[0].blocked_reason.as_deref(),
            Some("policy denied"),
            "blocked_reason must be persisted"
        );
        assert_eq!(rows[0].approach_signature, "sig-blocked");
    }

    #[tokio::test]
    async fn test_record_executed_advances_attempt_index_without_counting_as_failure() {
        let state = temp_state().await;
        let ctrl = SelfCorrectionController::new(state.clone(), 3);
        let k = SelfCorrectionSubjectKind::Task;

        ctrl.record_executed("e1", k, "sig-exec", 1, Some("evt-42"))
            .await
            .unwrap();

        let rows = state.get_self_correction_attempts("e1").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, crate::traits::attempt_status::EXECUTED);
        assert_eq!(rows[0].evidence_ref.as_deref(), Some("evt-42"));

        // The EXECUTED row must not count as a failure: budget is still intact.
        let decision = ctrl.attempt("e1", k, "sig-new").await.unwrap();
        assert_eq!(
            decision,
            AttemptDecision::Proceed { attempt_index: 2 },
            "EXECUTED must not consume the failure budget"
        );
    }

    #[tokio::test]
    async fn test_executed_same_signature_does_not_stop_repeat() {
        let state = temp_state().await;
        let ctrl = SelfCorrectionController::new(state.clone(), 3);
        let k = SelfCorrectionSubjectKind::Task;

        // Record an EXECUTED row for sig-exec.
        ctrl.record_executed("e2", k, "sig-exec", 1, None)
            .await
            .unwrap();

        // Attempting the same signature again must NOT trigger StopRepeat.
        let decision = ctrl.attempt("e2", k, "sig-exec").await.unwrap();
        assert_eq!(
            decision,
            AttemptDecision::Proceed { attempt_index: 2 },
            "EXECUTED must not trigger StopRepeat on the same signature"
        );
    }

    #[tokio::test]
    async fn test_executed_does_not_increment_distinct_failure_budget() {
        let state = temp_state().await;
        let ctrl = SelfCorrectionController::new(state.clone(), 2);
        let k = SelfCorrectionSubjectKind::Task;

        // Fill budget with one real failure plus one EXECUTED — should NOT exhaust.
        ctrl.record_failure("e3", k, "sig-fail", 1, None)
            .await
            .unwrap();
        ctrl.record_executed("e3", k, "sig-exec", 2, None)
            .await
            .unwrap();

        // With k=2 and only 1 distinct failure, the budget is not exhausted.
        let decision = ctrl.attempt("e3", k, "sig-new").await.unwrap();
        assert_eq!(
            decision,
            AttemptDecision::Proceed { attempt_index: 3 },
            "EXECUTED must not count toward the distinct-failure budget"
        );
    }

    #[tokio::test]
    async fn test_blocked_reason_redacts_secrets() {
        let state = temp_state().await;
        let ctrl = SelfCorrectionController::new(state.clone(), 3);
        let k = SelfCorrectionSubjectKind::Task;

        // A blocked_reason containing an API-key-shaped secret.
        ctrl.record_blocked(
            "r1",
            k,
            "sig-redact",
            1,
            "blocked because token=sk-ABCDEF1234567890abcdef leaked",
        )
        .await
        .unwrap();

        let rows = state.get_self_correction_attempts("r1").await.unwrap();
        let reason = rows[0].blocked_reason.as_deref().unwrap_or("");
        assert!(
            !reason.contains("sk-ABCDEF1234567890abcdef"),
            "secret must be redacted in persisted blocked_reason: {reason}"
        );
        assert!(
            reason.contains("[REDACTED:API key]"),
            "expected [REDACTED:API key] in persisted reason: {reason}"
        );
    }

    // ── P2.4 carry-forward redaction test ──────────────────────────────────

    #[tokio::test]
    async fn test_secret_bearing_repeat_triggers_stop_repeat() {
        // Prove that like-for-like redacted comparison fires StopRepeat even when
        // the raw signature going in on both sides contains a secret.
        //
        // Real gate flow (P2.4):
        //   1. Gate calls `normalized_attempt_signature(&action)` → redacted string.
        //   2. Gate calls `controller.attempt(subject, kind, &redacted)` — attempt()
        //      receives an *already-redacted* prospective signature and compares it
        //      against stored rows unchanged.
        //   3. On failure: gate calls `controller.record_failure(subject, kind, raw_sig, ...)`
        //      — record_failure redacts before storing.
        //
        // This test replicates that exact flow:
        //   - "raw" prospective sig has the secret; we redact it with `redact_secrets`
        //     (mirroring what `normalized_attempt_signature` does) before passing to
        //     `attempt()`.
        //   - `record_failure` receives the raw sig and redacts it before storing.
        //   - Both stored and prospective strings are "[REDACTED:API key]"-bearing, so
        //     they compare equal → StopRepeat fires.
        //
        // This is NOT a tautology: if `record_failure` forgot to redact the stored sig
        // (storing the raw value), the comparison with the pre-redacted prospective sig
        // would produce a *mismatch* and StopRepeat would never fire — the secret-bearing
        // approach could be retried indefinitely.
        let ctrl = SelfCorrectionController::new(temp_state().await, 3);
        let k = SelfCorrectionSubjectKind::Task;
        let subject = "p2-4-test";

        // The raw signature the gate would produce for a terminal command embedding
        // an API key (sk- prefix, ≥20 alphanum chars → matches SECRET_PATTERNS).
        let raw_sig = "tool=terminal cmd=curl -H Authorization: Bearer sk-ABCDEF1234567890XYZ0 https://api.example.com method= host=api.example.com auth_profile=false auth_header=false detach=false path_scope=none mutating=false external=false high_impact=false";

        // Step 1: gate pre-redacts (mirrors normalized_attempt_signature's redact_secrets call).
        let prospective = crate::tools::sanitize::redact_secrets(raw_sig);

        // First attempt must Proceed (no prior failures yet).
        assert_eq!(
            ctrl.attempt(subject, k, &prospective).await.unwrap(),
            AttemptDecision::Proceed { attempt_index: 1 },
            "first attempt must Proceed"
        );

        // Step 2: record_failure with the raw sig — internally it redacts before storing.
        ctrl.record_failure(subject, k, raw_sig, 1, None)
            .await
            .unwrap();

        // Step 3: second attempt with the same pre-redacted prospective sig must trigger
        // StopRepeat, proving stored-redacted == prospective-redacted comparison holds.
        assert_eq!(
            ctrl.attempt(subject, k, &prospective).await.unwrap(),
            AttemptDecision::StopRepeat,
            "secret-bearing approach must trigger StopRepeat on second attempt (redacted \
             comparison is like-for-like)"
        );
    }
}
