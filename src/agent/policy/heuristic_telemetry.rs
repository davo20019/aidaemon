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
    ///
    /// Each fire is also persisted as a queryable `GateTelemetry` decision
    /// point tagged with `task_id` + `iteration`, so efficacy analysis (which
    /// gates help vs hurt, within-task repeat loops, join to the task's
    /// `TaskEnd` outcome) is a query against the event store rather than
    /// log archaeology over the in-memory counter.
    pub(crate) async fn supervision_gate_enforced(
        &self,
        heuristic: &'static str,
        model: &str,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
    ) -> bool {
        self.supervision_gate_enforced_with_context(
            heuristic,
            model,
            emitter,
            task_id,
            iteration,
            serde_json::Value::Null,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn supervision_gate_enforced_with_context(
        &self,
        heuristic: &'static str,
        model: &str,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        context: serde_json::Value,
    ) -> bool {
        let tier = self.trust_tier_for_model(model);
        let action = gate_action_for_tier(tier);
        self.persist_gate_fire_with_context(
            emitter, task_id, iteration, heuristic, model, tier, action, context,
        )
        .await;
        matches!(action, HeuristicAction::Enforced)
    }

    /// Record a gate fire in the in-memory counter (+ `heuristic_telemetry`
    /// tracing event) and persist it as a queryable `GateTelemetry` decision
    /// point. Shared by tier-gated supervision gates and always-enforced
    /// hard-cap blocks so every gate fire is analyzable from the event store.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn persist_gate_fire(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        heuristic: &str,
        model: &str,
        tier: ModelTrustTier,
        action: HeuristicAction,
    ) {
        self.persist_gate_fire_with_context(
            emitter,
            task_id,
            iteration,
            heuristic,
            model,
            tier,
            action,
            serde_json::Value::Null,
        )
        .await;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn persist_gate_fire_with_context(
        &self,
        emitter: &crate::events::EventEmitter,
        task_id: &str,
        iteration: usize,
        heuristic: &str,
        model: &str,
        tier: ModelTrustTier,
        action: HeuristicAction,
        context: serde_json::Value,
    ) {
        global().record(heuristic, model, tier, action);
        self.emit_decision_point(
            emitter,
            task_id,
            iteration,
            crate::events::DecisionType::GateTelemetry,
            format!("Supervision gate '{heuristic}': {}", action.as_str()),
            serde_json::json!({
                "code": "supervision_gate_fire",
                "heuristic": heuristic,
                "action": action.as_str(),
                "tier": tier.as_str(),
                "model": model,
                "context": context,
            }),
        )
        .await;
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

    /// A supervision gate fire must be persisted as a queryable
    /// `GateTelemetry` decision point carrying the heuristic name, action,
    /// tier, model, and the task_id + iteration that make within-task repeat
    /// detection a query rather than a stored counter.
    #[tokio::test]
    async fn gate_fire_is_persisted_as_decision_point() {
        use crate::events::{DecisionPointData, DecisionType, EventEmitter, EventStore};

        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .expect("test agent");
        let session_id = "gate_telemetry_session";
        let emitter = EventEmitter::new(harness.agent.event_store.clone(), session_id.to_string());

        // Guided-tier model → gate enforces (returns true).
        let enforced = harness
            .agent
            .supervision_gate_enforced(
                "uncertainty_clarify_gate",
                "gemma-3-27b-it",
                &emitter,
                "task-guided",
                3,
            )
            .await;
        assert!(enforced, "Guided-tier model should enforce the gate");

        // Autonomous-tier model → gate shadow-skips (returns false).
        let shadow = harness
            .agent
            .supervision_gate_enforced(
                "uncertainty_clarify_gate",
                "claude-opus-4-8",
                &emitter,
                "task-auto",
                1,
            )
            .await;
        assert!(!shadow, "Autonomous-tier model should shadow-skip the gate");

        let store = EventStore::new(harness.state.pool())
            .await
            .expect("event store");
        let events = store
            .query_recent_events(session_id, 200)
            .await
            .expect("recent events");

        let gate_fires: Vec<DecisionPointData> = events
            .iter()
            .filter_map(|e| e.parse_data::<DecisionPointData>().ok())
            .filter(|d| d.decision_type == DecisionType::GateTelemetry)
            .filter(|d| {
                d.metadata.get("code").and_then(serde_json::Value::as_str)
                    == Some("supervision_gate_fire")
            })
            .collect();
        assert_eq!(gate_fires.len(), 2, "expected one persisted event per fire");

        let guided = gate_fires
            .iter()
            .find(|d| d.task_id == "task-guided")
            .expect("guided fire persisted");
        assert_eq!(guided.iteration, 3);
        assert_eq!(
            guided
                .metadata
                .get("heuristic")
                .and_then(serde_json::Value::as_str),
            Some("uncertainty_clarify_gate")
        );
        assert_eq!(
            guided
                .metadata
                .get("action")
                .and_then(serde_json::Value::as_str),
            Some("enforced")
        );
        assert_eq!(
            guided
                .metadata
                .get("tier")
                .and_then(serde_json::Value::as_str),
            Some("guided")
        );
        assert_eq!(
            guided
                .metadata
                .get("model")
                .and_then(serde_json::Value::as_str),
            Some("gemma-3-27b-it")
        );

        let auto = gate_fires
            .iter()
            .find(|d| d.task_id == "task-auto")
            .expect("autonomous fire persisted");
        assert_eq!(
            auto.metadata
                .get("action")
                .and_then(serde_json::Value::as_str),
            Some("shadow_skipped")
        );
        assert_eq!(
            auto.metadata
                .get("tier")
                .and_then(serde_json::Value::as_str),
            Some("autonomous")
        );
    }
}
