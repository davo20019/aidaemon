//! `/connect` token checks, by platform, as this build has them. Saving and
//! listing runtime-managed bots is `aidaemon_channels::hub::bots`.

pub(crate) use aidaemon_channels::hub::bots::{
    check_bot_exists, list_dynamic_bots, persist_dynamic_bot,
};

/// Validate a Telegram bot token by calling `getMe`.
/// Returns the bot username on success.
pub(crate) async fn validate_telegram_token(token: &str) -> Result<String, String> {
    aidaemon_channel_telegram::bot_api::validate_token(token).await
}

/// Validate a Discord bot token by calling the Discord API.
/// Returns the bot name on success.
#[cfg(feature = "discord")]
pub(crate) async fn validate_discord_token(token: &str) -> Result<String, String> {
    aidaemon_channel_discord::validate_token(token).await
}

#[cfg(not(feature = "discord"))]
pub(crate) async fn validate_discord_token(_token: &str) -> Result<String, String> {
    Err("Discord support is not enabled in this build.".to_string())
}

/// Validate Slack tokens by calling `auth.test`.
/// Returns `(bot_name, team_name)` on success.
#[cfg(feature = "slack")]
pub(crate) async fn validate_slack_tokens(
    bot_token: &str,
    app_token: &str,
) -> Result<(String, String), String> {
    aidaemon_channel_slack::validate_tokens(bot_token, app_token).await
}

#[cfg(not(feature = "slack"))]
pub(crate) async fn validate_slack_tokens(
    _bot_token: &str,
    _app_token: &str,
) -> Result<(String, String), String> {
    Err("Slack support is not enabled in this build.".to_string())
}
