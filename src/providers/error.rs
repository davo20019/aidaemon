use std::fmt;

/// Classified provider error — tells the caller *why* the LLM call failed
/// so it can pick the right recovery strategy.
#[derive(Debug)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub status: Option<u16>,
    pub message: String,
    /// Seconds to wait before retrying (from 429 Retry-After header or body).
    pub retry_after_secs: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    /// 401/403 — bad API key or permissions.
    Auth,
    /// 402 — billing/quota exhausted.
    Billing,
    /// 429 — rate limited; check retry_after_secs.
    RateLimit,
    /// 404 or "model not found" — bad model name.
    NotFound,
    /// 408, request timeout, or provider took too long.
    Timeout,
    /// Connection refused, DNS failure, reset, etc.
    Network,
    /// 500/502/503/504 — provider-side outage.
    ServerError,
    /// Anything else.
    Unknown,
}

impl ProviderError {
    pub fn from_status(status: u16, body: &str) -> Self {
        let kind = match status {
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

        Self {
            kind,
            status: Some(status),
            message: truncate_body(body),
            retry_after_secs,
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
            retry_after_secs: None,
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
            ProviderErrorKind::Unknown => format!("LLM error: {}", self.message),
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
            write!(f, "Provider error ({}, {:?}): {}", status, self.kind, self.message)
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

fn truncate_body(body: &str) -> String {
    if body.len() > 300 {
        format!("{}...", &body[..300])
    } else {
        body.to_string()
    }
}
