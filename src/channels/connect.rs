//! Shared validation and persistence helpers for the `/connect` command.
//!
//! These functions are channel-agnostic: they validate tokens and manage the
//! `dynamic_bots` table without touching channel-specific spawning logic.

use teloxide::prelude::*;
use tracing::warn;

use crate::traits::{DynamicBot, StateStore};

// ---------------------------------------------------------------------------
// Telegram
// ---------------------------------------------------------------------------

/// Validate a Telegram bot token by calling `getMe`.
/// Returns the bot username on success.
pub(crate) async fn validate_telegram_token(token: &str) -> Result<String, String> {
    let test_bot = Bot::new(token);
    let me = test_bot.get_me().await.map_err(|e| {
        format!(
            "Invalid token: {}\n\nMake sure you copied the full token from @BotFather.",
            e
        )
    })?;

    Ok(me.username.clone().unwrap_or_else(|| "unknown".to_string()))
}

// ---------------------------------------------------------------------------
// Discord
// ---------------------------------------------------------------------------

/// Validate a Discord bot token by calling the Discord API.
/// Returns the bot name on success.
#[cfg(feature = "discord")]
pub(crate) async fn validate_discord_token(token: &str) -> Result<String, String> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://discord.com/api/v10/users/@me")
        .header("Authorization", format!("Bot {}", token))
        .send()
        .await
        .map_err(|e| format!("Failed to validate token: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Invalid Discord token (HTTP {}). Make sure you copied the bot token from Discord Developer Portal.",
            response.status()
        ));
    }

    match response.json::<serde_json::Value>().await {
        Ok(json) => Ok(json["username"].as_str().unwrap_or("unknown").to_string()),
        Err(_) => Ok("unknown".to_string()),
    }
}

#[cfg(not(feature = "discord"))]
pub(crate) async fn validate_discord_token(_token: &str) -> Result<String, String> {
    Err("Discord support is not enabled in this build.".to_string())
}

// ---------------------------------------------------------------------------
// Slack
// ---------------------------------------------------------------------------

/// Validate Slack tokens by calling `auth.test`.
/// Returns `(bot_name, team_name)` on success.
#[cfg(feature = "slack")]
pub(crate) async fn validate_slack_tokens(
    bot_token: &str,
    _app_token: &str,
) -> Result<(String, String), String> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://slack.com/api/auth.test")
        .header("Authorization", format!("Bearer {}", bot_token))
        .send()
        .await
        .map_err(|e| format!("Failed to validate Slack token: {}", e))?;

    let json: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Slack response: {}", e))?;

    if json["ok"].as_bool() != Some(true) {
        return Err(format!(
            "Invalid Slack token: {}\n\nMake sure you have the correct bot token.",
            json["error"].as_str().unwrap_or("unknown error")
        ));
    }

    Ok((
        json["user"].as_str().unwrap_or("unknown").to_string(),
        json["team"].as_str().unwrap_or("unknown").to_string(),
    ))
}

#[cfg(not(feature = "slack"))]
pub(crate) async fn validate_slack_tokens(
    _bot_token: &str,
    _app_token: &str,
) -> Result<(String, String), String> {
    Err("Slack support is not enabled in this build.".to_string())
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

/// Check if a bot with the given `channel_type` and `bot_token` already exists.
pub(crate) async fn check_bot_exists(
    state: &dyn StateStore,
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
pub(crate) async fn persist_dynamic_bot(
    state: &dyn StateStore,
    new_bot: &DynamicBot,
) -> Result<i64, String> {
    state
        .add_dynamic_bot(new_bot)
        .await
        .map_err(|e| format!("Failed to save bot configuration: {}", e))
}

/// List all dynamic bots from the database.
pub(crate) async fn list_dynamic_bots(state: &dyn StateStore) -> Result<Vec<DynamicBot>, String> {
    state
        .get_dynamic_bots()
        .await
        .map_err(|e| format!("Failed to list bots: {}", e))
}
