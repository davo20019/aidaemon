//! OAuth lives in `aidaemon_tools::services::oauth`; the daemon keeps its old
//! paths to it, builds its gateway over the daemon's secret store, and keeps
//! the ChatGPT subscription login.

pub mod chatgpt_codex;

pub use aidaemon_tools::services::oauth::*;

/// The daemon's gateway: tokens in the daemon's secret store, and a missing
/// client credential named with the `aidaemon keychain set` command.
pub fn daemon_gateway(
    state_store: std::sync::Arc<dyn crate::traits::OAuthGatewayStore>,
    http_profiles: SharedHttpProfiles,
    callback_base_url: String,
) -> OAuthGateway {
    OAuthGateway::new(
        state_store,
        crate::config::shared_secret_store(),
        http_profiles,
        callback_base_url,
    )
    .with_secret_set_command("aidaemon keychain set")
}
