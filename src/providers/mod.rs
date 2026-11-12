//! Provider construction from the daemon's config. Calling providers with
//! chat-completions JSON and the classified provider errors live in
//! `aidaemon-runtime`; the old paths stay.

pub(crate) mod factory;

pub(crate) use aidaemon_runtime::chat_bridge::{
    build_http_client, context_overflow, describe_reset, sdk_backed, streaming,
    ProviderRecoveryRoute,
};
pub use aidaemon_runtime::chat_bridge::{
    MalformedResponseReason, ModelFailure, ProviderError, ProviderErrorKind,
};
