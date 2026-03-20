use std::fmt;

/// Classified provider error — tells the caller *why* the LLM call failed
/// so it can pick the right recovery strategy.
#[derive(Debug, Clone)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub status: Option<u16>,
    pub message: String,
    pub malformed_reason: Option<MalformedResponseReason>,
    /// Seconds to wait before retrying (from 429 Retry-After header or body).
    pub retry_after_secs: Option<u64>,
    /// For 402 Billing errors: the max tokens the account can actually afford,
    /// parsed from messages like "can only afford 6917".
    pub affordable_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MalformedResponseReason {
    Parse,
    Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    /// 401/403 — bad API key or permissions.
    Auth,
    /// 402 — billing/quota exhausted.
    Billing,
    /// 429 — rate limited; check retry_after_secs.
    RateLimit,
    /// 400 — malformed request (e.g. missing thought_signature, invalid schema).
    BadRequest,
    /// 404 or "model not found" — bad model name.
    NotFound,
    /// 408, request timeout, or provider took too long.
    Timeout,
    /// Connection refused, DNS failure, reset, etc.
    Network,
    /// 500/502/503/504 — provider-side outage.
    ServerError,
    /// Provider returned a malformed success payload.
    /// Recovery is reason-aware (parse may be transient; shape is often deterministic).
    MalformedResponse,
    /// Anything else.
    Unknown,
}

impl ProviderError {
    pub fn from_status(status: u16, body: &str) -> Self {
        let kind = match status {
            400 => ProviderErrorKind::BadRequest,
            401 | 403 => ProviderErrorKind::Auth,
            402 => ProviderErrorKind::Billing,
            404 => ProviderErrorKind::NotFound,
            408 => ProviderErrorKind::Timeout,
            429 => ProviderErrorKind::RateLimit,
            500 | 502 | 503 | 504 => ProviderErrorKind::ServerError,
            _ => ProviderErrorKind::Unknown,
        };

        // Try to extract retry_after from JSON body for 429s
        let retry_after_secs = if kind == ProviderErrorKind::RateLimit {
            extract_retry_after(body)
        } else {
            None
        };

        // For 402 billing errors, parse affordable token count from the message.
        // OpenRouter format: "can only afford 6917"
        let affordable_tokens = if kind == ProviderErrorKind::Billing {
            extract_affordable_tokens(body)
        } else {
            None
        };

        Self {
            kind,
            status: Some(status),
            message: truncate_body(body),
            malformed_reason: None,
            retry_after_secs,
            affordable_tokens,
        }
    }

    pub fn timeout_msg(message: impl Into<String>) -> Self {
        Self {
            kind: ProviderErrorKind::Timeout,
            status: None,
            message: message.into(),
            malformed_reason: None,
            retry_after_secs: None,
            affordable_tokens: None,
        }
    }

    pub fn network(err: &reqwest::Error) -> Self {
        let kind = if err.is_timeout() {
            ProviderErrorKind::Timeout
        } else {
            ProviderErrorKind::Network
        };
        Self {
            kind,
            status: None,
            message: err.to_string(),
            malformed_reason: None,
            retry_after_secs: None,
            affordable_tokens: None,
        }
    }

    pub fn malformed_parse(message: impl Into<String>) -> Self {
        Self {
            kind: ProviderErrorKind::MalformedResponse,
            status: Some(200),
            message: message.into(),
            malformed_reason: Some(MalformedResponseReason::Parse),
            retry_after_secs: None,
            affordable_tokens: None,
        }
    }

    pub fn malformed_shape(message: impl Into<String>) -> Self {
        Self {
            kind: ProviderErrorKind::MalformedResponse,
            status: Some(200),
            message: message.into(),
            malformed_reason: Some(MalformedResponseReason::Shape),
            retry_after_secs: None,
            affordable_tokens: None,
        }
    }

    /// User-facing summary suitable for sending back via Telegram.
    pub fn user_message(&self) -> String {
        match self.kind {
            ProviderErrorKind::Auth => {
                "LLM API authentication failed. Check your API key in config.toml.".to_string()
            }
            ProviderErrorKind::Billing => {
                "LLM API billing error — your account quota may be exhausted.".to_string()
            }
            ProviderErrorKind::RateLimit => {
                if let Some(secs) = self.retry_after_secs {
                    format!("Rate limited. Retrying in {}s...", secs)
                } else {
                    "Rate limited. Retrying shortly...".to_string()
                }
            }
            ProviderErrorKind::NotFound => {
                "Model not found. Falling back to previous model.".to_string()
            }
            ProviderErrorKind::Timeout => "LLM request timed out. Retrying...".to_string(),
            ProviderErrorKind::Network => {
                "Cannot reach LLM provider (network error). Will retry.".to_string()
            }
            ProviderErrorKind::ServerError => {
                "LLM provider is experiencing issues (server error). Will retry.".to_string()
            }
            ProviderErrorKind::MalformedResponse => {
                format!(
                    "LLM provider returned a malformed response. This may be a provider bug. Details: {}",
                    self.message
                )
            }
            ProviderErrorKind::BadRequest => {
                format!("LLM request was malformed (400). This may be a bug — please report it. Details: {}", self.message)
            }
            ProviderErrorKind::Unknown => format!("LLM error: {}", self.message),
        }
    }

    /// User-facing summary for cases where recovery has already failed and
    /// there are no more retries/fallbacks left to attempt.
    pub fn recovery_failed_message(&self) -> String {
        match self.kind {
            ProviderErrorKind::RateLimit => {
                "The LLM provider remained rate limited during recovery. Try again shortly."
                    .to_string()
            }
            ProviderErrorKind::NotFound => {
                "The configured LLM model could not be used, and fallback recovery did not succeed. Check model settings."
                    .to_string()
            }
            ProviderErrorKind::Timeout => {
                "LLM requests kept timing out during recovery. Try again shortly.".to_string()
            }
            ProviderErrorKind::Network => {
                "Could not reach the LLM provider during recovery. Check connectivity or try again shortly."
                    .to_string()
            }
            ProviderErrorKind::ServerError => {
                "The LLM provider kept returning server errors during recovery. Try again later or switch providers."
                    .to_string()
            }
            _ => self.user_message(),
        }
    }

    /// Whether this error is worth retrying (same request, same model).
    #[allow(dead_code)]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.kind,
            ProviderErrorKind::RateLimit
                | ProviderErrorKind::Timeout
                | ProviderErrorKind::Network
                | ProviderErrorKind::ServerError
        )
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(status) = self.status {
            write!(
                f,
                "Provider error ({}, {:?}): {}",
                status, self.kind, self.message
            )
        } else {
            write!(f, "Provider error ({:?}): {}", self.kind, self.message)
        }
    }
}

impl std::error::Error for ProviderError {}

/// Try to parse retry_after from a JSON response body.
/// Handles: {"error": {"retry_after": 5}} and {"retry_after": 5}
fn extract_retry_after(body: &str) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    v["error"]["retry_after"]
        .as_u64()
        .or_else(|| v["retry_after"].as_u64())
        .or_else(|| {
            // Some providers use a float
            v["error"]["retry_after"]
                .as_f64()
                .or_else(|| v["retry_after"].as_f64())
                .map(|f| f.ceil() as u64)
        })
}

/// Parse affordable token count from a 402 billing error message.
/// Handles OpenRouter format: "can only afford 6917"
fn extract_affordable_tokens(body: &str) -> Option<u32> {
    // Try JSON first: {"error":{"message":"...can only afford 6917..."}}
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        let msg = v["error"]["message"]
            .as_str()
            .or_else(|| v["message"].as_str())
            .unwrap_or("");
        if let Some(n) = parse_affordable_from_text(msg) {
            return Some(n);
        }
    }
    // Fallback: search the raw body text
    parse_affordable_from_text(body)
}

fn parse_affordable_from_text(text: &str) -> Option<u32> {
    // Pattern: "can only afford <number>"
    let marker = "can only afford ";
    let pos = text.find(marker)?;
    let after = &text[pos + marker.len()..];
    let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    num_str.parse::<u32>().ok().filter(|&n| n > 0)
}

/// Truncate a string to at most `max_len` bytes, respecting UTF-8 char boundaries.
/// Avoids panicking on multi-byte characters.
fn truncate_body(body: &str) -> String {
    const MAX_LEN: usize = 300;
    if body.len() <= MAX_LEN {
        return body.to_string();
    }
    // Find a valid UTF-8 char boundary at or before MAX_LEN
    let mut end = MAX_LEN;
    while end > 0 && !body.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &body[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_server_error_message_mentions_retry() {
        let err = ProviderError::from_status(
            500,
            "{\"error\":{\"message\":\"Internal Server Error\",\"code\":500}}",
        );
        assert_eq!(
            err.user_message(),
            "LLM provider is experiencing issues (server error). Will retry."
        );
    }

    #[test]
    fn terminal_server_error_message_does_not_promise_retry() {
        let err = ProviderError::from_status(
            500,
            "{\"error\":{\"message\":\"Internal Server Error\",\"code\":500}}",
        );
        let msg = err.recovery_failed_message();
        assert!(msg.contains("server errors during recovery"));
        assert!(!msg.contains("Will retry"));
    }

    #[test]
    fn terminal_rate_limit_message_does_not_promise_retry() {
        let err = ProviderError::from_status(429, "{\"error\":{\"retry_after\":5}}");
        let msg = err.recovery_failed_message();
        assert!(msg.contains("remained rate limited during recovery"));
        assert!(!msg.contains("Retrying"));
    }

    #[test]
    fn billing_402_parses_affordable_tokens_from_openrouter() {
        let body = r#"{"error":{"message":"This request requires more credits, or fewer max_tokens. You requested up to 16384 tokens, but can only afford 6917. To increase, visit https://openrouter.ai/settings/credits","code":402}}"#;
        let err = ProviderError::from_status(402, body);
        assert_eq!(err.kind, ProviderErrorKind::Billing);
        assert_eq!(err.affordable_tokens, Some(6917));
    }

    #[test]
    fn billing_402_no_affordable_tokens_when_missing() {
        let body = r#"{"error":{"message":"Insufficient credits","code":402}}"#;
        let err = ProviderError::from_status(402, body);
        assert_eq!(err.kind, ProviderErrorKind::Billing);
        assert_eq!(err.affordable_tokens, None);
    }

    #[test]
    fn billing_402_affordable_zero_returns_none() {
        let body = r#"{"error":{"message":"can only afford 0 tokens","code":402}}"#;
        let err = ProviderError::from_status(402, body);
        assert_eq!(err.affordable_tokens, None);
    }
}
