//! GoalTokenRegistry — cancellation token hierarchy for goal orchestration.
//!
//! Each active goal gets a `CancellationToken`. Task leads and executors
//! derive child tokens so cancelling a goal cascades to all its agents.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::traits::Goal;

/// Registry of cancellation tokens keyed by goal ID.
///
/// Clone is cheap (Arc-based). Created once in `core.rs`, shared with
/// `HeartbeatCoordinator` and `Agent`.
#[derive(Clone)]
pub struct GoalTokenRegistry {
    tokens: Arc<RwLock<HashMap<String, CancellationToken>>>,
    // Best-effort in-memory guard to avoid spawning duplicate task leads/heartbeats
    // for the same goal_id within a single daemon process.
    active_runs: Arc<Mutex<HashSet<String>>>,
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
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
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
        registry.remove("goal-3").await;
        assert!(registry.child_token("goal-3").await.is_none());
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
}
