//! Runtime-managed bots (`/connect`): checking for, saving and listing the
//! bots kept in a [`DynamicBotStore`].

use tracing::warn;

use super::{DynamicBot, DynamicBotStore};

/// Check if a bot with the given `channel_type` and `bot_token` already exists.
pub async fn check_bot_exists(
    state: &dyn DynamicBotStore,
    channel_type: &str,
    bot_token: &str,
) -> Result<bool, String> {
    match state.get_dynamic_bots().await {
        Ok(bots) => Ok(bots
            .iter()
            .any(|b| b.channel_type == channel_type && b.bot_token == bot_token)),
        Err(e) => {
            warn!("Failed to check existing bots: {}", e);
            // Non-fatal: allow the caller to proceed (same behaviour as before).
            Ok(false)
        }
    }
}

/// Persist a new dynamic bot to the database. Returns the database row ID.
pub async fn persist_dynamic_bot(
    state: &dyn DynamicBotStore,
    new_bot: &DynamicBot,
) -> Result<i64, String> {
    state
        .add_dynamic_bot(new_bot)
        .await
        .map_err(|e| format!("Failed to save bot configuration: {}", e))
}

/// List all dynamic bots from the database.
pub async fn list_dynamic_bots(state: &dyn DynamicBotStore) -> Result<Vec<DynamicBot>, String> {
    state
        .get_dynamic_bots()
        .await
        .map_err(|e| format!("Failed to list bots: {}", e))
}
