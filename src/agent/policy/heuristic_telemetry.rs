//! Shadow-mode fire counters for loop supervision heuristics.
//!
//! Every supervision gate (deferred-action blocking, prelude gates, budget
//! blocks) reports its fires here, tagged with the model and trust tier and
//! whether the gate actually enforced or was skipped in shadow mode. The
//! counters answer the tuning question the loop has historically lacked
//! data for: which heuristics fire, on which models, and how often —
//! so false-positive-prone gates can be found and pruned from evidence
//! instead of anecdote.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use super::trust_tier::ModelTrustTier;

/// Process-wide registry. Parent and child agents share one set of counters
/// so per-model fire rates aggregate across the whole loop.
static GLOBAL: LazyLock<HeuristicTelemetry> = LazyLock::new(HeuristicTelemetry::default);

pub fn global() -> &'static HeuristicTelemetry {
    &GLOBAL
}

/// What the gate did when it fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeuristicAction {
    /// The gate blocked/redirected the model (Guided tier behavior).
    Enforced,
    /// The gate matched but was skipped — telemetry only (Autonomous tier).
    ShadowSkipped,
}

impl HeuristicAction {
    pub fn as_str(self) -> &'static str {
        match self {
            HeuristicAction::Enforced => "enforced",
            HeuristicAction::ShadowSkipped => "shadow_skipped",
        }
    }
}

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
    /// Record one gate fire and emit a structured tracing event under the
    /// stable `heuristic_telemetry` target for offline analysis.
    pub fn record(
        &self,
        heuristic: &str,
        model: &str,
        tier: ModelTrustTier,
        action: HeuristicAction,
    ) {
        tracing::info!(
            target: "heuristic_telemetry",
            heuristic,
            model,
            tier = tier.as_str(),
            action = action.as_str(),
            "supervision heuristic fired"
        );
        let mut counters = match self.counters.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let stats = counters
            .entry((heuristic.to_string(), model.to_string()))
            .or_default();
        match action {
            HeuristicAction::Enforced => stats.enforced += 1,
            HeuristicAction::ShadowSkipped => stats.shadow_skipped += 1,
        }
    }

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

/// Pure decision used by gate sites: Guided tiers enforce, Autonomous tiers
/// shadow-skip. Returned action is what should be recorded.
pub fn gate_action_for_tier(tier: ModelTrustTier) -> HeuristicAction {
    match tier {
        ModelTrustTier::Guided => HeuristicAction::Enforced,
        ModelTrustTier::Autonomous => HeuristicAction::ShadowSkipped,
    }
}

impl crate::agent::Agent {
    /// Evaluate a supervision gate for the active model: records the fire in
    /// telemetry and returns whether the gate should actually block.
    ///
    /// Guided tier → records `Enforced`, returns true (gate blocks as today).
    /// Autonomous tier → records `ShadowSkipped`, returns false (gate is
    /// telemetry-only; the model proceeds).
    pub(crate) fn supervision_gate_enforced(&self, heuristic: &'static str, model: &str) -> bool {
        let tier = self.trust_tier_for_model(model);
        let action = gate_action_for_tier(tier);
        global().record(heuristic, model, tier, action);
        matches!(action, HeuristicAction::Enforced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accumulates_per_heuristic_and_model() {
        let t = HeuristicTelemetry::default();
        t.record(
            "uncertainty_gate",
            "gemma-3-27b-it",
            ModelTrustTier::Guided,
            HeuristicAction::Enforced,
        );
        t.record(
            "uncertainty_gate",
            "gemma-3-27b-it",
            ModelTrustTier::Guided,
            HeuristicAction::Enforced,
        );
        t.record(
            "uncertainty_gate",
            "claude-opus-4-8",
            ModelTrustTier::Autonomous,
            HeuristicAction::ShadowSkipped,
        );

        let gemma = t.stats_for("uncertainty_gate", "gemma-3-27b-it");
        assert_eq!(gemma.enforced, 2);
        assert_eq!(gemma.shadow_skipped, 0);

        let claude = t.stats_for("uncertainty_gate", "claude-opus-4-8");
        assert_eq!(claude.enforced, 0);
        assert_eq!(claude.shadow_skipped, 1);
        assert_eq!(claude.total(), 1);
    }

    #[test]
    fn stats_for_unknown_pair_is_zero() {
        let t = HeuristicTelemetry::default();
        assert_eq!(
            t.stats_for("never_fired", "any-model"),
            HeuristicFireStats::default()
        );
    }

    #[test]
    fn snapshot_is_sorted_and_complete() {
        let t = HeuristicTelemetry::default();
        t.record(
            "b_gate",
            "model-1",
            ModelTrustTier::Guided,
            HeuristicAction::Enforced,
        );
        t.record(
            "a_gate",
            "model-2",
            ModelTrustTier::Autonomous,
            HeuristicAction::ShadowSkipped,
        );

        let snap = t.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].0, "a_gate");
        assert_eq!(snap[1].0, "b_gate");
    }

    #[test]
    fn gate_action_follows_tier() {
        assert_eq!(
            gate_action_for_tier(ModelTrustTier::Guided),
            HeuristicAction::Enforced
        );
        assert_eq!(
            gate_action_for_tier(ModelTrustTier::Autonomous),
            HeuristicAction::ShadowSkipped
        );
    }
}
