#[cfg(feature = "discord")]
mod discord;
mod formatting;
mod hub;
#[cfg(feature = "slack")]
mod slack;
pub(crate) mod telegram;

#[cfg(feature = "discord")]
pub use discord::{spawn_discord_channel, DiscordChannel};
pub use hub::{ChannelHub, SessionMap};
#[cfg(feature = "slack")]
pub use slack::{spawn_slack_channel, SlackChannel};
pub use telegram::TelegramChannel;
