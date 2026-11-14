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

/// Whether a provider error says the request exceeded the model's context
/// window, and the window size when the error states it. Matches the
/// providers' own error codes and fixed messages (OpenAI
/// `context_length_exceeded`, Anthropic "prompt is too long", Gemini's
/// token-count error), not free text.
pub fn context_overflow(error: &str) -> Option<Option<usize>> {
    let lower = error.to_ascii_lowercase();
    let overflow = [
        "context_length_exceeded",
        "maximum context length",
        "exceeds the context window",
        "prompt is too long",
        "input token count",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    if !overflow {
        return None;
    }
    let number_after = |marker: &str| -> Option<usize> {
        let start = lower.find(marker)? + marker.len();
        let digits: String = lower[start..]
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit() || *c == ',')
            .filter(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    };
    // OpenAI: "maximum context length is 128000 tokens"; Anthropic:
    // "210000 tokens > 200000 maximum"; Gemini: "allowed (1048576)".
    let limit = number_after("maximum context length is")
        .or_else(|| number_after("tokens >"))
        .or_else(|| number_after("allowed ("))
        .filter(|limit| *limit >= 1_000);
    Some(limit)
}

#[cfg(test)]
mod context_overflow_tests {
    use super::context_overflow;

    #[test]
    fn provider_overflow_errors_are_recognized_with_their_limit() {
        assert_eq!(
            context_overflow(
                "400: This model's maximum context length is 128000 tokens. However, your \
                 messages resulted in 130512 tokens."
            ),
            Some(Some(128_000))
        );
        assert_eq!(
            context_overflow(
                "invalid_request_error: prompt is too long: 210000 tokens > 200000 maximum"
            ),
            Some(Some(200_000))
        );
        assert_eq!(
            context_overflow(
                "The input token count (1100000) exceeds the maximum number of tokens allowed (1048576)."
            ),
            Some(Some(1_048_576))
        );
        // The Responses API states no number.
        assert_eq!(
            context_overflow(
                r#"{"error":{"code":"context_length_exceeded","message":"Your input exceeds the context window of this model."}}"#
            ),
            Some(None)
        );
        assert_eq!(context_overflow("429: rate limited"), None);
    }
}
