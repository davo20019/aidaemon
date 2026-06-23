use crate::traits::{attempt_status, SelfCorrectionAttempt};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
