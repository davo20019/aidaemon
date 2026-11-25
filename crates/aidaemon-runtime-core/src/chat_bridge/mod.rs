//! Calling an SDK provider with the daemon's chat-completions JSON, and the
//! daemon's classification of provider errors for its recovery decisions.

mod error;
pub mod sdk_backed;
pub mod streaming;

use std::time::Duration;

use reqwest::Client;
use tracing::warn;

pub use error::{MalformedResponseReason, ModelFailure, ProviderError, ProviderErrorKind};
pub use error::{ProviderRecoveryRoute, describe_reset};
pub use sdk_backed::ChatJson;

/// Build an HTTP client with a panic-safe fallback when system proxy discovery
/// is unavailable in the runtime environment.
pub fn build_http_client(timeout: Duration) -> Result<Client, String> {
    // Test environments (and some constrained runtimes) can panic inside
    // macOS system proxy discovery. Skip that code path entirely for tests.
    if cfg!(test)
        || matches!(
            std::env::var("AIDAEMON_DISABLE_SYSTEM_PROXY_DISCOVERY").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    {
        return Client::builder()
            .timeout(timeout)
            .no_proxy()
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e));
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Client::builder().timeout(timeout).build()
    })) {
        Ok(Ok(client)) => return Ok(client),
        Ok(Err(e)) => {
            warn!(
                error = %e,
                "HTTP client build with system proxy support failed; retrying with proxy discovery disabled"
            );
        }
        Err(_) => {
            warn!(
                "HTTP client build panicked during system proxy discovery; retrying with proxy discovery disabled"
            );
        }
    }

    Client::builder()
        .timeout(timeout)
        .no_proxy()
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}
