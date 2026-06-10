//! Per-task model pinning for active `computer_use` GUI loops.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use tokio::sync::RwLock;

#[derive(Clone, Default)]
pub struct ComputerUsePinRegistry {
    inner: Arc<RwLock<HashMap<String, String>>>,
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

    pub async fn get(&self, task_id: &str) -> Option<String> {
        self.inner.read().await.get(task_id).cloned()
    }

    pub async fn pin(&self, task_id: impl Into<String>, model: impl Into<String>) {
        self.inner
            .write()
            .await
            .insert(task_id.into(), model.into());
    }

    pub async fn clear_task(&self, task_id: &str) {
        self.inner.write().await.remove(task_id);
    }
}
