use std::sync::Arc;

use crate::traits::{attempt_status, SelfCorrectionAttempt, SelfCorrectionSubjectKind, StateStore};

/// Outcome of asking the controller whether a correction attempt may proceed.
#[allow(dead_code)] // Used in Task 5+; integration into agent loop pending
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
#[allow(dead_code)] // Used in Task 5+; integration into agent loop pending
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
#[allow(dead_code)] // Used in future plans; integration into agent loop pending
pub struct SelfCorrectionController {
    state: Arc<dyn StateStore>,
    max_attempts: usize,
}

#[allow(dead_code)] // Used in future plans; integration into agent loop pending
impl SelfCorrectionController {
    pub fn new(state: Arc<dyn StateStore>, max_attempts: usize) -> Self {
        Self {
            state,
            max_attempts,
        }
    }

    /// Decide whether a correction attempt with `signature` may proceed for the
    /// subject. Pure policy applied over the durable attempt history.
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
        self.record(
            subject_id,
            kind,
            signature,
            attempt_index,
            attempt_status::FAILED,
            None,
            evidence_ref,
        )
        .await
    }

    pub async fn record_success(
        &self,
        subject_id: &str,
        kind: SelfCorrectionSubjectKind,
        signature: &str,
        attempt_index: usize,
    ) -> anyhow::Result<()> {
        self.record(
            subject_id,
            kind,
            signature,
            attempt_index,
            attempt_status::VERIFIED_SUCCESS,
            None,
            None,
        )
        .await
    }

    /// Honest summary of what was tried, or `None` if nothing failed. Persists a
    /// single terminal `gave_up` row (idempotent per subject).
    pub async fn give_up_report(&self, subject_id: &str) -> anyhow::Result<Option<String>> {
        let prior = self.state.get_self_correction_attempts(subject_id).await?;
        let is_failed = |s: &str| s == attempt_status::FAILED || s == attempt_status::BLOCKED;

        let mut distinct = Vec::new();
        for a in prior.iter().filter(|a| is_failed(&a.status)) {
            if !distinct.contains(&a.approach_signature) {
                distinct.push(a.approach_signature.clone());
            }
        }
        if distinct.is_empty() {
            return Ok(None);
        }

        let mut report = format!(
            "I tried {} different approach{} and none worked:\n",
            distinct.len(),
            if distinct.len() == 1 { "" } else { "es" }
        );
        for sig in &distinct {
            report.push_str("- ");
            report.push_str(sig);
            report.push('\n');
        }

        // Idempotent terminal marker: only one gave_up row per subject.
        let already_gave_up = prior.iter().any(|a| a.status == attempt_status::GAVE_UP);
        if !already_gave_up {
            // Reconstruct the subject kind from history (all rows share it).
            let kind = prior
                .first()
                .and_then(|a| SelfCorrectionSubjectKind::from_str(&a.subject_kind))
                .unwrap_or(SelfCorrectionSubjectKind::Task);
            self.record(
                subject_id,
                kind,
                "<gave_up>",
                prior.len() + 1,
                attempt_status::GAVE_UP,
                None,
                None,
            )
            .await?;
        }

        Ok(Some(report))
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

    #[tokio::test]
    async fn give_up_report_is_none_without_failures_and_summarizes_after() {
        let ctrl = SelfCorrectionController::new(temp_state().await, 3);
        let k = SelfCorrectionSubjectKind::Task;
        assert!(ctrl.give_up_report("t2").await.unwrap().is_none());

        ctrl.record_failure("t2", k, "du -ah ~", 1, None)
            .await
            .unwrap();
        ctrl.record_failure("t2", k, "find ~ -size +500M", 2, None)
            .await
            .unwrap();
        let report = ctrl.give_up_report("t2").await.unwrap().unwrap();
        assert!(report.contains("du -ah ~"));
        assert!(report.contains("find ~ -size +500M"));
        assert!(report.contains("2")); // tried 2 approaches
    }

    #[tokio::test]
    async fn record_success_does_not_count_as_failure() {
        let ctrl = SelfCorrectionController::new(temp_state().await, 3);
        let k = SelfCorrectionSubjectKind::Task;
        ctrl.record_success("t3", k, "sig-ok", 1).await.unwrap();
        // A success leaves the budget intact: the next attempt still proceeds.
        assert_eq!(
            ctrl.attempt("t3", k, "sig-next").await.unwrap(),
            AttemptDecision::Proceed { attempt_index: 2 }
        );
        assert!(ctrl.give_up_report("t3").await.unwrap().is_none());
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
}
