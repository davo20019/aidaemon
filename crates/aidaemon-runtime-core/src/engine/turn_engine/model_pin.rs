//! The owner's `/model` pin: the model this agent runs on, switching it, and
//! handing routing back.

use std::time::Duration;

use tracing::{info, warn};

use super::TurnEngine;

/// How long a model lock is waited for before giving up.
const LOCK_WAIT: Duration = Duration::from_secs(2);

impl TurnEngine<'_> {
    /// The model this agent runs on (the primary model when its lock is
    /// held too long).
    pub async fn current_model(&self) -> String {
        match tokio::time::timeout(LOCK_WAIT, self.model_lock().read()).await {
            Ok(guard) => guard.clone(),
            Err(_) => {
                warn!("Timed out acquiring model lock for current_model()");
                self.llm_runtime().snapshot().primary_model()
            }
        }
    }

    /// Switch the active model, keeping the old one as the fallback, and pin
    /// it: routing leaves it until [`Self::clear_model_override`].
    pub async fn set_model(&self, model: String) {
        let mut current = match tokio::time::timeout(LOCK_WAIT, self.model_lock().write()).await {
            Ok(guard) => guard,
            Err(_) => {
                warn!("Timed out acquiring model write lock for set_model()");
                return;
            }
        };
        let mut fallback =
            match tokio::time::timeout(LOCK_WAIT, self.fallback_model_lock().write()).await {
                Ok(guard) => guard,
                Err(_) => {
                    warn!("Timed out acquiring fallback_model write lock for set_model()");
                    return;
                }
            };
        info!(old = %*current, new = %model, "Model switched");
        *fallback = current.clone();
        *current = model;
        match tokio::time::timeout(LOCK_WAIT, self.model_override_lock().write()).await {
            Ok(mut guard) => *guard = true,
            Err(_) => warn!("Timed out acquiring model_override write lock for set_model()"),
        }
    }

    /// Re-enable auto-routing after a manual model pin.
    pub async fn clear_model_override(&self) {
        match tokio::time::timeout(LOCK_WAIT, self.model_override_lock().write()).await {
            Ok(mut guard) => *guard = false,
            Err(_) => {
                warn!("Timed out acquiring model_override write lock for clear_model_override()");
                return;
            }
        }
        info!("Model override cleared, auto-routing re-enabled");
    }
}
