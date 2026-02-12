//! GoalTokenRegistry — cancellation token hierarchy for V3 goals.
//!
//! Each active goal gets a `CancellationToken`. Task leads and executors
//! derive child tokens so cancelling a goal cascades to all its agents.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::traits::GoalV3;

/// Registry of cancellation tokens keyed by goal ID.
///
/// Clone is cheap (Arc-based). Created once in `core.rs`, shared with
/// `HeartbeatCoordinator` and `Agent`.
#[derive(Clone)]
pub struct GoalTokenRegistry {
    tokens: Arc<RwLock<HashMap<String, CancellationToken>>>,
}

impl GoalTokenRegistry {
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
        }
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
    pub async fn rebuild_from_goals(&self, goals: &[GoalV3]) {
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
            GoalV3::new_finite("Goal A", "session-1"),
            GoalV3::new_finite("Goal B", "session-2"),
        ];
        registry.rebuild_from_goals(&goals).await;

        // Both should have tokens
        assert!(registry.child_token(&goals[0].id).await.is_some());
        assert!(registry.child_token(&goals[1].id).await.is_some());

        // Rebuild again shouldn't duplicate
        registry.rebuild_from_goals(&goals).await;
        assert!(registry.child_token(&goals[0].id).await.is_some());
    }
}
