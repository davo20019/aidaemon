use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::config::QueueLanePolicyConfig;
use crate::queue_telemetry::QueuePressure;

const MAX_SESSION_BUDGET_ENTRIES: usize = 2048;

pub type SessionFairnessBudget = HashMap<String, (Instant, u32)>;

pub fn allow_overload_session(
    lane_policy: &QueueLanePolicyConfig,
    fair_session_budget: &mut SessionFairnessBudget,
    session_id: &str,
) -> bool {
    if !lane_policy.fair_sessions {
        return false;
    }

    let now = Instant::now();
    let window = Duration::from_secs(lane_policy.fair_session_window_secs);
    let max_events = lane_policy.fair_max_events_per_session;
    let budget = fair_session_budget
        .entry(session_id.to_string())
        .or_insert((now, 0));
    if now.duration_since(budget.0) >= window {
        *budget = (now, 0);
    }

    let mut allow = false;
    if budget.1 < max_events {
        budget.1 += 1;
        allow = true;
    }

    // Keep map bounded over long runtimes.
    if fair_session_budget.len() > MAX_SESSION_BUDGET_ENTRIES {
        fair_session_budget.retain(|_, (started_at, _)| now.duration_since(*started_at) < window);
    }

    allow
}

pub fn should_shed_due_to_overload(
    lane_policy: &QueueLanePolicyConfig,
    pressure: QueuePressure,
    fair_session_budget: &mut SessionFairnessBudget,
    session_id: &str,
) -> bool {
    lane_policy.adaptive_shedding
        && pressure == QueuePressure::Overload
        && !allow_overload_session(lane_policy, fair_session_budget, session_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fairness_budget_enforces_per_session_limit() {
        let policy = QueueLanePolicyConfig {
            fair_sessions: true,
            fair_session_window_secs: 60,
            fair_max_events_per_session: 2,
            ..QueueLanePolicyConfig::default()
        };
        let mut budget: SessionFairnessBudget = HashMap::new();

        assert!(allow_overload_session(&policy, &mut budget, "sess-1"));
        assert!(allow_overload_session(&policy, &mut budget, "sess-1"));
        assert!(!allow_overload_session(&policy, &mut budget, "sess-1"));

        // A different session gets its own fair share.
        assert!(allow_overload_session(&policy, &mut budget, "sess-2"));
    }

    #[test]
    fn fairness_budget_can_be_disabled() {
        let policy = QueueLanePolicyConfig {
            fair_sessions: false,
            ..QueueLanePolicyConfig::default()
        };
        let mut budget: SessionFairnessBudget = HashMap::new();
        assert!(!allow_overload_session(&policy, &mut budget, "sess-1"));
    }

    #[test]
    fn overload_shedding_only_triggers_when_overloaded() {
        let policy = QueueLanePolicyConfig {
            adaptive_shedding: true,
            fair_sessions: false,
            ..QueueLanePolicyConfig::default()
        };
        let mut budget: SessionFairnessBudget = HashMap::new();
        assert!(!should_shed_due_to_overload(
            &policy,
            QueuePressure::Normal,
            &mut budget,
            "sess-1"
        ));
        assert!(should_shed_due_to_overload(
            &policy,
            QueuePressure::Overload,
            &mut budget,
            "sess-1"
        ));
    }
}
