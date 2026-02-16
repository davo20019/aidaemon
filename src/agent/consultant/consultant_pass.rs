//! Consultant pass extraction scaffolding.
//!
//! This module defines typed outcomes for consultant routing so the large
//! orchestrator branch in `main_loop.rs` can be moved behind a narrower API.
#![allow(dead_code)]

use std::sync::atomic::Ordering;

use super::POLICY_METRICS;

/// High-level result of consultant-pass routing.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsultantPassOutcome {
    /// Return text directly to the user without entering the tool loop.
    DirectReturn {
        reason: ConsultantDirectReason,
        response_text: String,
    },
    /// Continue into the full agent loop (tool-enabled execution).
    Fallthrough {
        reason: ConsultantFallthroughReason,
        /// Optional advisory text to carry into iteration 2 as a system hint.
        carry_system_hint: Option<String>,
    },
}

/// Direct-return reasons currently present in main-loop consultant logic.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsultantDirectReason {
    ClarificationRequired,
    QuestionAnswered,
    AcknowledgmentOrCorrection,
    GenericCancelHandled,
    ScheduledInternalMaintenanceDeclined,
    ScheduledGoalConfirmedInline,
    ScheduledGoalCancelledInline,
    ScheduledGoalAwaitingTextConfirmation,
    KnowledgeIntentAnswered,
    ComplexInternalMaintenanceDeclined,
    ComplexGoalSpawnSyncSuccess,
    ComplexGoalSpawnSyncFailure,
    ComplexGoalSpawnBackgroundAck,
}

/// Fallthrough reasons currently present in main-loop consultant logic.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsultantFallthroughReason {
    ScheduledMissingTiming,
    ScheduledNonOwner,
    ScheduledParseFailed,
    SimpleIntent,
    ComplexNonOwner,
}

impl ConsultantPassOutcome {
    /// Returns true when this outcome should increment
    /// `consultant_direct_return_total`.
    pub fn is_direct_return(&self) -> bool {
        matches!(self, Self::DirectReturn { .. })
    }

    /// Returns true when this outcome should increment
    /// `consultant_fallthrough_total`.
    pub fn is_fallthrough(&self) -> bool {
        matches!(self, Self::Fallthrough { .. })
    }
}

/// Increment consultant pass direct-return metric.
pub fn mark_consultant_direct_return() {
    POLICY_METRICS
        .consultant_direct_return_total
        .fetch_add(1, Ordering::Relaxed);
}

/// Increment consultant pass fallthrough metric.
pub fn mark_consultant_fallthrough() {
    POLICY_METRICS
        .consultant_fallthrough_total
        .fetch_add(1, Ordering::Relaxed);
}

/// Record consultant metrics based on a typed outcome.
pub fn record_consultant_outcome_metrics(outcome: &ConsultantPassOutcome) {
    if outcome.is_direct_return() {
        mark_consultant_direct_return();
    } else if outcome.is_fallthrough() {
        mark_consultant_fallthrough();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_return_flag_matches_variant() {
        let outcome = ConsultantPassOutcome::DirectReturn {
            reason: ConsultantDirectReason::QuestionAnswered,
            response_text: "answer".to_string(),
        };
        assert!(outcome.is_direct_return());
        assert!(!outcome.is_fallthrough());
    }

    #[test]
    fn fallthrough_flag_matches_variant() {
        let outcome = ConsultantPassOutcome::Fallthrough {
            reason: ConsultantFallthroughReason::SimpleIntent,
            carry_system_hint: Some("hint".to_string()),
        };
        assert!(outcome.is_fallthrough());
        assert!(!outcome.is_direct_return());
    }
}
