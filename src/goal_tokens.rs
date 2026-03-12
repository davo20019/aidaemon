//! GoalTokenRegistry — cancellation token hierarchy for goal orchestration.
//!
//! Each active goal gets a `CancellationToken`. Task leads and executors
//! derive child tokens so cancelling a goal cascades to all its agents.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::traits::{Goal, ScheduledRunHealth};

/// Registry of cancellation tokens keyed by goal ID.
///
/// Clone is cheap (Arc-based). Created once in `core.rs`, shared with
/// `HeartbeatCoordinator` and `Agent`.
#[derive(Clone)]
pub struct GoalTokenRegistry {
    tokens: Arc<RwLock<HashMap<String, CancellationToken>>>,
    budget_overrides: Arc<RwLock<HashMap<String, GoalBudgetOverride>>>,
    run_budgets: Arc<RwLock<HashMap<String, GoalRunBudgetState>>>,
    // Best-effort in-memory guard to avoid spawning duplicate task leads/heartbeats
    // for the same goal_id within a single daemon process.
    active_runs: Arc<Mutex<HashSet<String>>>,
}

#[derive(Clone)]
struct GoalBudgetOverride {
    budget_daily: i64,
    day_anchor: String,
}

#[derive(Clone)]
struct GoalRunBudgetState {
    effective_budget_per_check: i64,
    tokens_used: i64,
    budget_extensions_count: usize,
    health: ScheduledRunHealth,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalRunBudgetStatus {
    pub effective_budget_per_check: i64,
    pub tokens_used: i64,
    pub budget_extensions_count: usize,
    pub health: ScheduledRunHealth,
}

/// Guard object returned by `GoalTokenRegistry::try_acquire_run`.
/// Releases the run lock when dropped.
pub struct GoalRunGuard {
    registry: GoalTokenRegistry,
    goal_id: String,
}

impl Drop for GoalRunGuard {
    fn drop(&mut self) {
        if let Ok(mut runs) = self.registry.active_runs.lock() {
            runs.remove(&self.goal_id);
        }
    }
}

impl GoalTokenRegistry {
    fn run_budget_status_from_state(state: &GoalRunBudgetState) -> GoalRunBudgetStatus {
        GoalRunBudgetStatus {
            effective_budget_per_check: state.effective_budget_per_check,
            tokens_used: state.tokens_used,
            budget_extensions_count: state.budget_extensions_count,
            health: state.health.clone(),
        }
    }

    pub fn new() -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
            budget_overrides: Arc::new(RwLock::new(HashMap::new())),
            run_budgets: Arc::new(RwLock::new(HashMap::new())),
            active_runs: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Try to acquire an in-process "run lock" for a goal. Returns None if another
    /// run is already active for this goal_id.
    pub fn try_acquire_run(&self, goal_id: &str) -> Option<GoalRunGuard> {
        let mut runs = self.active_runs.lock().ok()?;
        if runs.contains(goal_id) {
            return None;
        }
        runs.insert(goal_id.to_string());
        Some(GoalRunGuard {
            registry: self.clone(),
            goal_id: goal_id.to_string(),
        })
    }

    /// Register a new cancellation token for a goal. Returns the token.
    pub async fn register(&self, goal_id: &str) -> CancellationToken {
        let token = CancellationToken::new();
        self.tokens
            .write()
            .await
            .insert(goal_id.to_string(), token.clone());
        token
    }

    /// Cancel a goal's token (cascades to all child tokens). Returns true if found.
    pub async fn cancel(&self, goal_id: &str) -> bool {
        let tokens = self.tokens.read().await;
        if let Some(token) = tokens.get(goal_id) {
            token.cancel();
            true
        } else {
            false
        }
    }

    /// Get a child token for a goal (for task leads / executors).
    /// Returns None if the goal has no registered token.
    pub async fn child_token(&self, goal_id: &str) -> Option<CancellationToken> {
        let tokens = self.tokens.read().await;
        tokens.get(goal_id).map(|t| t.child_token())
    }

    /// Remove a goal's token (after goal completes/fails).
    pub async fn remove(&self, goal_id: &str) {
        self.tokens.write().await.remove(goal_id);
        self.budget_overrides.write().await.remove(goal_id);
        self.run_budgets.write().await.remove(goal_id);
    }

    /// Rebuild registry from a list of active goals (for startup recovery).
    pub async fn rebuild_from_goals(&self, goals: &[Goal]) {
        let mut tokens = self.tokens.write().await;
        for goal in goals {
            if !tokens.contains_key(&goal.id) {
                tokens.insert(goal.id.clone(), CancellationToken::new());
            }
        }
    }

    /// Persist a runtime-only daily budget override for the current UTC day.
    ///
    /// This is shared across task-leads/executors for the same goal, but never
    /// written to SQLite, so manual budgets do not ratchet upward permanently.
    pub async fn set_effective_daily_budget(&self, goal_id: &str, budget_daily: i64) {
        let day_anchor = chrono::Utc::now().date_naive().to_string();
        self.budget_overrides.write().await.insert(
            goal_id.to_string(),
            GoalBudgetOverride {
                budget_daily,
                day_anchor,
            },
        );
    }

    /// Get the runtime-only daily budget override for the current UTC day.
    pub async fn get_effective_daily_budget(&self, goal_id: &str) -> Option<i64> {
        let day_anchor = chrono::Utc::now().date_naive().to_string();
        let overrides = self.budget_overrides.read().await;
        overrides.get(goal_id).and_then(|entry| {
            if entry.day_anchor == day_anchor {
                Some(entry.budget_daily)
            } else {
                None
            }
        })
    }

    /// Initialize the runtime-only per-run budget for an active scheduled run.
    pub async fn start_run_budget(&self, goal_id: &str, budget_per_check: Option<i64>) {
        let mut budgets = self.run_budgets.write().await;
        if let Some(budget_per_check) = budget_per_check.filter(|b| *b > 0) {
            budgets.insert(
                goal_id.to_string(),
                GoalRunBudgetState {
                    effective_budget_per_check: budget_per_check,
                    tokens_used: 0,
                    budget_extensions_count: 0,
                    health: ScheduledRunHealth::default(),
                },
            );
        } else {
            budgets.remove(goal_id);
        }
    }

    pub async fn restore_run_budget(
        &self,
        goal_id: &str,
        effective_budget_per_check: i64,
        tokens_used: i64,
        budget_extensions_count: usize,
        health: ScheduledRunHealth,
    ) -> Option<GoalRunBudgetStatus> {
        if effective_budget_per_check <= 0 {
            self.run_budgets.write().await.remove(goal_id);
            return None;
        }

        self.run_budgets.write().await.insert(
            goal_id.to_string(),
            GoalRunBudgetState {
                effective_budget_per_check,
                tokens_used: tokens_used.max(0),
                budget_extensions_count,
                health,
            },
        );

        self.get_run_budget(goal_id).await
    }

    pub async fn get_run_budget(&self, goal_id: &str) -> Option<GoalRunBudgetStatus> {
        let budgets = self.run_budgets.read().await;
        budgets.get(goal_id).map(Self::run_budget_status_from_state)
    }

    pub async fn add_run_tokens(
        &self,
        goal_id: &str,
        delta_tokens: i64,
    ) -> Option<GoalRunBudgetStatus> {
        let mut budgets = self.run_budgets.write().await;
        let state = budgets.get_mut(goal_id)?;
        state.tokens_used = state.tokens_used.saturating_add(delta_tokens).max(0);
        Some(Self::run_budget_status_from_state(state))
    }

    pub async fn auto_extend_run_budget(
        &self,
        goal_id: &str,
        new_budget: i64,
    ) -> Option<GoalRunBudgetStatus> {
        let mut budgets = self.run_budgets.write().await;
        let state = budgets.get_mut(goal_id)?;
        state.effective_budget_per_check = new_budget;
        state.budget_extensions_count = state.budget_extensions_count.saturating_add(1);
        Some(Self::run_budget_status_from_state(state))
    }

    pub async fn set_run_budget(
        &self,
        goal_id: &str,
        new_budget: i64,
    ) -> Option<GoalRunBudgetStatus> {
        let mut budgets = self.run_budgets.write().await;
        let state = budgets.get_mut(goal_id)?;
        state.effective_budget_per_check = new_budget;
        Some(Self::run_budget_status_from_state(state))
    }

    pub async fn clear_run_budget(&self, goal_id: &str) {
        self.run_budgets.write().await.remove(goal_id);
    }

    pub async fn update_run_health(
        &self,
        goal_id: &str,
        health: ScheduledRunHealth,
    ) -> Option<GoalRunBudgetStatus> {
        let mut budgets = self.run_budgets.write().await;
        let state = budgets.get_mut(goal_id)?;
        state.health.evidence_gain_count = state
            .health
            .evidence_gain_count
            .max(health.evidence_gain_count);
        state.health.total_successful_tool_calls = state
            .health
            .total_successful_tool_calls
            .max(health.total_successful_tool_calls);
        state.health.stall_count = health.stall_count;
        state.health.consecutive_same_tool_count = health.consecutive_same_tool_count;
        state.health.consecutive_same_tool_unique_args = health.consecutive_same_tool_unique_args;
        state.health.unrecovered_error_count = health.unrecovered_error_count;
        Some(Self::run_budget_status_from_state(state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_cancel() {
        let registry = GoalTokenRegistry::new();
        let token = registry.register("goal-1").await;
        assert!(!token.is_cancelled());

        let cancelled = registry.cancel("goal-1").await;
        assert!(cancelled);
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn test_cancel_nonexistent() {
        let registry = GoalTokenRegistry::new();
        let cancelled = registry.cancel("nope").await;
        assert!(!cancelled);
    }

    #[tokio::test]
    async fn test_child_token_cascades() {
        let registry = GoalTokenRegistry::new();
        let _parent = registry.register("goal-2").await;

        let child = registry.child_token("goal-2").await.unwrap();
        assert!(!child.is_cancelled());

        registry.cancel("goal-2").await;
        assert!(child.is_cancelled());
    }

    #[tokio::test]
    async fn test_child_token_nonexistent() {
        let registry = GoalTokenRegistry::new();
        assert!(registry.child_token("nope").await.is_none());
    }

    #[tokio::test]
    async fn test_remove() {
        let registry = GoalTokenRegistry::new();
        registry.register("goal-3").await;
        registry.set_effective_daily_budget("goal-3", 123_456).await;
        registry.remove("goal-3").await;
        assert!(registry.child_token("goal-3").await.is_none());
        assert!(registry
            .get_effective_daily_budget("goal-3")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_clear_run_budget() {
        let registry = GoalTokenRegistry::new();
        registry.start_run_budget("goal-4", Some(50)).await;
        assert!(registry.get_run_budget("goal-4").await.is_some());

        registry.clear_run_budget("goal-4").await;
        assert!(registry.get_run_budget("goal-4").await.is_none());
    }

    #[tokio::test]
    async fn test_rebuild_from_goals() {
        let registry = GoalTokenRegistry::new();
        let goals = vec![
            Goal::new_finite("Goal A", "session-1"),
            Goal::new_finite("Goal B", "session-2"),
        ];
        registry.rebuild_from_goals(&goals).await;

        // Both should have tokens
        assert!(registry.child_token(&goals[0].id).await.is_some());
        assert!(registry.child_token(&goals[1].id).await.is_some());

        // Rebuild again shouldn't duplicate
        registry.rebuild_from_goals(&goals).await;
        assert!(registry.child_token(&goals[0].id).await.is_some());
    }

    #[test]
    fn test_try_acquire_run_is_exclusive_and_released_on_drop() {
        let registry = GoalTokenRegistry::new();

        let g1 = registry
            .try_acquire_run("goal-1")
            .expect("first acquire should succeed");
        assert!(
            registry.try_acquire_run("goal-1").is_none(),
            "second acquire should fail while guard is held"
        );

        drop(g1);
        assert!(
            registry.try_acquire_run("goal-1").is_some(),
            "acquire should succeed again after guard drop"
        );
    }

    #[tokio::test]
    async fn test_runtime_daily_budget_override_round_trips() {
        let registry = GoalTokenRegistry::new();
        assert!(registry
            .get_effective_daily_budget("goal-4")
            .await
            .is_none());

        registry.set_effective_daily_budget("goal-4", 400_000).await;

        assert_eq!(
            registry.get_effective_daily_budget("goal-4").await,
            Some(400_000)
        );
    }

    #[tokio::test]
    async fn test_run_budget_round_trips_and_extends() {
        let registry = GoalTokenRegistry::new();
        assert!(registry.get_run_budget("goal-5").await.is_none());

        registry.start_run_budget("goal-5", Some(100_000)).await;
        assert_eq!(
            registry.get_run_budget("goal-5").await,
            Some(GoalRunBudgetStatus {
                effective_budget_per_check: 100_000,
                tokens_used: 0,
                budget_extensions_count: 0,
                health: ScheduledRunHealth::default(),
            })
        );

        let after_tokens = registry.add_run_tokens("goal-5", 12_345).await.unwrap();
        assert_eq!(after_tokens.tokens_used, 12_345);

        let after_health = registry
            .update_run_health(
                "goal-5",
                ScheduledRunHealth {
                    evidence_gain_count: 1,
                    total_successful_tool_calls: 3,
                    stall_count: 0,
                    consecutive_same_tool_count: 1,
                    consecutive_same_tool_unique_args: 1,
                    unrecovered_error_count: 0,
                },
            )
            .await
            .unwrap();
        assert_eq!(after_health.health.evidence_gain_count, 1);
        assert_eq!(after_health.health.total_successful_tool_calls, 3);

        let after_extend = registry
            .auto_extend_run_budget("goal-5", 180_000)
            .await
            .unwrap();
        assert_eq!(after_extend.effective_budget_per_check, 180_000);
        assert_eq!(after_extend.budget_extensions_count, 1);

        let after_manual = registry.set_run_budget("goal-5", 220_000).await.unwrap();
        assert_eq!(after_manual.effective_budget_per_check, 220_000);
        assert_eq!(after_manual.budget_extensions_count, 1);

        registry.start_run_budget("goal-5", Some(90_000)).await;
        assert_eq!(
            registry.get_run_budget("goal-5").await,
            Some(GoalRunBudgetStatus {
                effective_budget_per_check: 90_000,
                tokens_used: 0,
                budget_extensions_count: 0,
                health: ScheduledRunHealth::default(),
            })
        );

        let restored = registry
            .restore_run_budget(
                "goal-5",
                180_000,
                77_000,
                2,
                ScheduledRunHealth {
                    evidence_gain_count: 2,
                    total_successful_tool_calls: 4,
                    stall_count: 0,
                    consecutive_same_tool_count: 1,
                    consecutive_same_tool_unique_args: 1,
                    unrecovered_error_count: 0,
                },
            )
            .await
            .unwrap();
        assert_eq!(restored.effective_budget_per_check, 180_000);
        assert_eq!(restored.tokens_used, 77_000);
        assert_eq!(restored.budget_extensions_count, 2);
        assert_eq!(restored.health.total_successful_tool_calls, 4);
    }
}
