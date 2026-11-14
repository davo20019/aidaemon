//! The daemon's ChatGPT subscription login: one process-wide
//! [`ChatGptCredentialManager`] over the daemon's secret store, so every
//! subscription-backed adapter shares one refresh-token rotation. The login
//! itself is `aidaemon_providers::chatgpt_login`.

use std::sync::{Arc, OnceLock};

pub use aidaemon_providers::chatgpt_login::*;

/// Shared credential owner used by every ChatGPT subscription-backed adapter.
pub fn shared_credential_manager() -> Arc<ChatGptCredentialManager> {
    static SHARED: OnceLock<Arc<ChatGptCredentialManager>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            let mut settings = LoginSettings::new("aidaemon", "aidaemon auth login openai");
            // A public client id; the variable overrides it if OpenAI rotates it.
            if let Some(client_id) = std::env::var("AIDAEMON_CHATGPT_CLIENT_ID")
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
            {
                settings.client_id = client_id;
            }
            Arc::new(ChatGptCredentialManager::new(
                crate::config::shared_secret_store(),
                settings,
            ))
        })
        .clone()
}

/// Load stored credentials, if a login has happened on this machine.
pub fn load_credentials() -> Option<ChatGptCredentials> {
    shared_credential_manager().load_credentials()
}

/// Whether a ChatGPT subscription login exists on this machine.
pub fn is_connected() -> bool {
    shared_credential_manager().is_connected()
}

/// Remove stored credentials and best-effort revoke the refresh token.
pub async fn logout(client: &reqwest::Client) -> anyhow::Result<()> {
    shared_credential_manager().logout(client).await
}

/// Run the full login flow and persist the resulting credentials.
pub async fn login(
    client: &reqwest::Client,
    mode: CallbackMode,
    announce: impl FnOnce(&str),
) -> anyhow::Result<ChatGptCredentials> {
    shared_credential_manager()
        .login(client, mode, announce)
        .await
}
