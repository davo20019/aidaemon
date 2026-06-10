//! Agent-loop integration for per-task `computer_use` model pinning.

#[cfg(feature = "computer_use")]
use crate::tools::computer_use::pin_registry::ComputerUsePinRegistry;

#[cfg(feature = "computer_use")]
pub(in crate::agent) async fn pinned_model_for_task(task_id: &str) -> Option<String> {
    ComputerUsePinRegistry::shared().get(task_id).await
}

#[cfg(feature = "computer_use")]
pub(in crate::agent) async fn resolve_model_for_task(task_id: &str, model: &str) -> String {
    pinned_model_for_task(task_id)
        .await
        .unwrap_or_else(|| model.to_string())
}

#[cfg(feature = "computer_use")]
pub(in crate::agent) async fn task_has_computer_use_pin(task_id: &str) -> bool {
    pinned_model_for_task(task_id).await.is_some()
}
