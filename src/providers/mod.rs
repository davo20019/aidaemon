//! Provider construction from the daemon's config. Calling providers with
//! chat-completions JSON and the classified provider errors live in
//! `aidaemon-runtime`; the old paths stay.

pub mod chatgpt_login;
pub(crate) mod factory;

#[cfg(test)]
pub(crate) use aidaemon_runtime::chat_bridge::streaming;
pub(crate) use aidaemon_runtime::chat_bridge::{build_http_client, sdk_backed};
#[cfg(test)]
pub use aidaemon_runtime::chat_bridge::{ProviderError, ProviderErrorKind};
