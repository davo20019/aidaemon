//! Per-task state for active `computer_use` GUI loops: the pinned vision model
//! and whether an unverified coordinate click is outstanding.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use tokio::sync::RwLock;

#[derive(Clone)]
struct TaskState {
    model: String,
    /// A coordinate click landed but has not been verified by a deliberate
    /// follow-up observation. Coordinate clicks target a raw (x, y) with no
    /// element identity, so the harness cannot auto-detect a hit vs a miss the
    /// way it does for element clicks — only a subsequent look (the vision
    /// model re-reading the screen) can confirm the intended change. Until then
    /// a "done / I clicked it" claim is unverified (2026-07-12: an unverified
    /// coordinate click at a guessed heart position was reported as a
    /// successful Like).
    unverified_coordinate_click: bool,
}

#[derive(Clone, Default)]
pub struct ComputerUsePinRegistry {
    inner: Arc<RwLock<HashMap<String, TaskState>>>,
}

impl ComputerUsePinRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process-wide registry shared by the tool and agent loop.
    pub fn shared() -> Self {
        static REGISTRY: OnceLock<ComputerUsePinRegistry> = OnceLock::new();
        REGISTRY.get_or_init(ComputerUsePinRegistry::new).clone()
    }

    /// The pinned vision model for this task, if any.
    pub async fn get(&self, task_id: &str) -> Option<String> {
        self.inner
            .read()
            .await
            .get(task_id)
            .map(|s| s.model.clone())
    }

    pub async fn pin(&self, task_id: impl Into<String>, model: impl Into<String>) {
        let model = model.into();
        self.inner
            .write()
            .await
            .entry(task_id.into())
            .and_modify(|s| s.model = model.clone())
            .or_insert(TaskState {
                model,
                unverified_coordinate_click: false,
            });
    }

    /// Record that a coordinate click landed without verification. No-op if the
    /// task is not pinned (coordinate clicks only run inside a pinned GUI loop).
    pub async fn mark_unverified_coordinate_click(&self, task_id: &str) {
        if let Some(s) = self.inner.write().await.get_mut(task_id) {
            s.unverified_coordinate_click = true;
        }
    }

    /// Clear the flag after a deliberate verifying observation (get_app_state /
    /// screenshot the model chose to run *after* the click).
    pub async fn clear_unverified_coordinate_click(&self, task_id: &str) {
        if let Some(s) = self.inner.write().await.get_mut(task_id) {
            s.unverified_coordinate_click = false;
        }
    }

    pub async fn has_unverified_coordinate_click(&self, task_id: &str) -> bool {
        self.inner
            .read()
            .await
            .get(task_id)
            .is_some_and(|s| s.unverified_coordinate_click)
    }

    pub async fn clear_task(&self, task_id: &str) {
        self.inner.write().await.remove(task_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn coordinate_click_flag_set_and_cleared_by_verification() {
        let reg = ComputerUsePinRegistry::new();
        reg.pin("t1", "gemini-3.5-flash").await;
        assert!(!reg.has_unverified_coordinate_click("t1").await);

        // A coordinate click leaves an unverified mutation outstanding.
        reg.mark_unverified_coordinate_click("t1").await;
        assert!(reg.has_unverified_coordinate_click("t1").await);
        // Pinned model is unchanged.
        assert_eq!(reg.get("t1").await.as_deref(), Some("gemini-3.5-flash"));

        // A deliberate verifying observation clears it.
        reg.clear_unverified_coordinate_click("t1").await;
        assert!(!reg.has_unverified_coordinate_click("t1").await);
    }

    #[tokio::test]
    async fn flag_is_noop_for_unpinned_task_and_survives_repin() {
        let reg = ComputerUsePinRegistry::new();
        // No pin yet: marking is a no-op (coordinate clicks only run pinned).
        reg.mark_unverified_coordinate_click("t2").await;
        assert!(!reg.has_unverified_coordinate_click("t2").await);

        reg.pin("t2", "m").await;
        reg.mark_unverified_coordinate_click("t2").await;
        // Re-pinning (model fallback mid-task) must not silently clear the flag.
        reg.pin("t2", "m2").await;
        assert!(reg.has_unverified_coordinate_click("t2").await);
        assert_eq!(reg.get("t2").await.as_deref(), Some("m2"));

        reg.clear_task("t2").await;
        assert!(reg.get("t2").await.is_none());
    }
}
