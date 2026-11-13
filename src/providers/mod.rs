//! Provider construction from the daemon's config. Calling providers with
//! chat-completions JSON and the classified provider errors live in
//! `aidaemon-runtime`; the old paths stay.

pub(crate) mod factory;

#[cfg(test)]
pub(crate) use aidaemon_runtime::chat_bridge::streaming;
#[cfg(test)]
pub use aidaemon_runtime::chat_bridge::ModelFailure;
pub(crate) use aidaemon_runtime::chat_bridge::{build_http_client, describe_reset, sdk_backed};
pub use aidaemon_runtime::chat_bridge::{ProviderError, ProviderErrorKind};
