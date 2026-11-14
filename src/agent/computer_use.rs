//! Agent-loop integration for per-task `computer_use` model pinning.

#[cfg(feature = "computer_use")]
use crate::tools::computer_use::pin_registry::ComputerUsePinRegistry;

#[cfg(feature = "computer_use")]
pub(in crate::agent) async fn pinned_model_for_task(task_id: &str) -> Option<String> {
    ComputerUsePinRegistry::shared().get(task_id).await
}

/// The computer-use pin registry as the turn engine's computer-use state.
#[cfg(feature = "computer_use")]
pub(in crate::agent) struct ComputerUsePins;

#[cfg(feature = "computer_use")]
#[async_trait::async_trait]
impl aidaemon_runtime::host::ComputerUseState for ComputerUsePins {
    async fn pinned_model(&self, task_id: &str) -> Option<String> {
        pinned_model_for_task(task_id).await
    }

    async fn has_unverified_coordinate_click(&self, task_id: &str) -> bool {
        ComputerUsePinRegistry::shared()
            .has_unverified_coordinate_click(task_id)
            .await
    }
}
