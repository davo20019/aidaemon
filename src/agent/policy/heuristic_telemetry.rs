//! Shadow-mode fire counters for loop supervision heuristics.
//!
//! Every supervision gate (deferred-action blocking, prelude gates, budget
//! blocks) reports its fires here, tagged with the model and trust tier and
//! whether the gate actually enforced or was skipped in shadow mode. The
//! counters answer the tuning question the loop has historically lacked
//! data for: which heuristics fire, on which models, and how often —
//! so false-positive-prone gates can be found and pruned from evidence
//! instead of anecdote.
//!
//! Audit invariant: any decision sourced from keyword/regex matching, prose
//! shape, or an auxiliary intent classifier must enter through
//! `supervision_gate_enforced[_with_context]` before it can block, redirect,
//! retry, rewrite, or narrow tools. Hard enforcement is reserved for durable
//! authorization and structured runtime evidence (role/capability policy,
//! explicit global read-only scope, target identity, approvals, idempotency,
//! budgets, typed tool receipts, and exact protocol validity).

use std::collections::HashMap;
use std::sync::Mutex;

/// Per-(heuristic, model) fire counts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HeuristicFireStats {
    pub enforced: u64,
    pub shadow_skipped: u64,
}

impl HeuristicFireStats {
    /// Diagnostics read API — consumed by tests today.
    #[allow(dead_code)]
    pub fn total(self) -> u64 {
        self.enforced + self.shadow_skipped
    }
}

/// In-memory fire-counter registry. One instance lives on the Agent;
/// gates record through [`crate::agent::Agent::supervision_gate_enforced`].
#[derive(Debug, Default)]
pub struct HeuristicTelemetry {
    counters: Mutex<HashMap<(String, String), HeuristicFireStats>>,
}

impl HeuristicTelemetry {
    /// Counts for one (heuristic, model) pair. Zero stats if never fired.
    /// Diagnostics read API — consumed by tests today; the tracing events
    /// under the `heuristic_telemetry` target are the production output.
    #[allow(dead_code)]
    pub fn stats_for(&self, heuristic: &str, model: &str) -> HeuristicFireStats {
        let counters = match self.counters.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        counters
            .get(&(heuristic.to_string(), model.to_string()))
            .copied()
            .unwrap_or_default()
    }

    /// All counters, sorted by heuristic then model, for diagnostics dumps.
    /// Diagnostics read API — consumed by tests today; the tracing events
    /// under the `heuristic_telemetry` target are the production output.
    #[allow(dead_code)]
    pub fn snapshot(&self) -> Vec<(String, String, HeuristicFireStats)> {
        let counters = match self.counters.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut entries: Vec<_> = counters
            .iter()
            .map(|((heuristic, model), stats)| (heuristic.clone(), model.clone(), *stats))
            .collect();
        entries.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        entries
    }
}

impl crate::agent::Agent {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_for_unknown_pair_is_zero() {
        let t = HeuristicTelemetry::default();
        assert_eq!(
            t.stats_for("never_fired", "any-model"),
            HeuristicFireStats::default()
        );
    }
}
