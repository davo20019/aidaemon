//! The Discord bot ([`DiscordChannel`], `aidaemon_channel_discord::bot`) on
//! the daemon's services: its shared slash commands and dynamic-bot store.

use std::sync::Arc;

pub use aidaemon_channel_discord::bot::{spawn_discord_channel, DiscordBot as DiscordChannel};

use crate::channels::ChannelRuntimeDeps;

/// A Discord bot on the daemon's services.
pub(crate) fn new_discord_bot(
    bot_token: &str,
    allowed_user_ids: Vec<u64>,
    owner_user_ids: Vec<u64>,
    guild_id: Option<u64>,
    runtime: ChannelRuntimeDeps,
) -> DiscordChannel {
    DiscordChannel::new(
        bot_token,
        allowed_user_ids,
        owner_user_ids,
        guild_id,
        runtime.bot_deps(),
        Arc::new(runtime.command_context()),
        runtime.state.clone(),
    )
}

#[cfg(test)]
#[path = "discord_tests.rs"]
mod tests;
