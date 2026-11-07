//! # aidaemon-channel-telegram (experimental)
//!
//! A Telegram transport for `aidaemon_channels`: long polling over the Bot
//! HTTP API, Markdown rendered as Telegram HTML, inline-keyboard approvals.
//! The adapter holds no agent logic; the channel runner owns sessions,
//! authorization and approvals.
//!
//! ```no_run
//! use aidaemon_channel_telegram::TelegramChannel;
//! use aidaemon_channels::Channel;
//!
//! let channel = TelegramChannel::new("000000:SYNTHETIC").poll_timeout_secs(50);
//! assert_eq!(channel.name(), "telegram");
//! ```

mod api;
mod callback;
mod channel;
pub mod format;
mod mapping;
#[cfg(feature = "webhook")]
mod webhook;

pub use callback::MAX_CALLBACK_DATA;
pub use channel::{TelegramChannel, MAX_DOWNLOAD_BYTES, TELEGRAM_MAX_MESSAGE_LEN};
#[cfg(feature = "webhook")]
pub use webhook::WebhookConfig;
