use std::fmt;

pub use aidaemon_events::provider::ProviderErrorKind;

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
    /// For a used-up quota: when the provider says it resets.
    pub resets_at: Option<std::time::SystemTime>,
    /// What each other model answered when this error ended a failover chain.
    pub fallback_failures: Vec<ModelFailure>,
    /// The provider refused the call before doing any work, so it used no
    /// tokens (`aidaemon_core::ProviderError::proves_nothing_spent`). False
    /// whenever that is not proven.
    pub spent_nothing: bool,
}

/// One model's failure during a failover chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFailure {
    pub model: String,
    pub kind: ProviderErrorKind,
    pub status: Option<u16>,
}

impl ModelFailure {
    fn reason(&self) -> String {
        let reason = match self.kind {
            ProviderErrorKind::Auth => "API key rejected",
            ProviderErrorKind::Billing => "out of credits",
            ProviderErrorKind::QuotaExhausted => "usage limit reached",
            ProviderErrorKind::RateLimit => "rate limited",
            ProviderErrorKind::NotFound => "model not found",
            ProviderErrorKind::Timeout => "timed out",
            ProviderErrorKind::Network => "could not be reached",
            ProviderErrorKind::ServerError => "provider error",
            _ => "request failed",
        };
        match self.status {
            Some(status) => format!("{reason} ({status})"),
            None => reason.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MalformedResponseReason {
    Parse,
    Shape,
}

/// Typed first transition in provider recovery. Retry loops implement bounded
/// waiting inside a state; this enum decides which state is reachable from a
/// classified outcome, independent of error-message wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderRecoveryRoute {
    Fail,
    RetryReducedBudgetThenFallback,
    AdaptRequestOnce,
    RetryRateLimitThenFallback,
    RetryTransientThenFallback,
    RetryMalformedThenFallback,
    RetryMalformedThenFail,
    Fallback,
}

impl ProviderError {
    pub fn recovery_route(&self, single_attempt_fail_closed: bool) -> ProviderRecoveryRoute {
        use ProviderRecoveryRoute::*;

        if single_attempt_fail_closed {
            return Fail;
        }
        match self.kind {
            ProviderErrorKind::Unknown => Fail,
            // The account's other models are skipped without a request
            // (each backend sits behind a `QuotaGuard`); another provider
            // may still answer.
            ProviderErrorKind::Auth => Fallback,
            ProviderErrorKind::Billing => RetryReducedBudgetThenFallback,
            ProviderErrorKind::BadRequest => AdaptRequestOnce,
            ProviderErrorKind::RateLimit => RetryRateLimitThenFallback,
            ProviderErrorKind::Timeout
            | ProviderErrorKind::Network
            | ProviderErrorKind::ServerError => RetryTransientThenFallback,
            ProviderErrorKind::MalformedResponse
                if self.malformed_reason == Some(MalformedResponseReason::Parse) =>
            {
                RetryMalformedThenFallback
            }
            ProviderErrorKind::MalformedResponse => RetryMalformedThenFail,
            ProviderErrorKind::NotFound | ProviderErrorKind::QuotaExhausted => Fallback,
        }
    }

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
            resets_at: None,
            fallback_failures: Vec::new(),
            spent_nothing: false,
        }
    }

    #[cfg(test)]
    pub fn malformed_parse(message: impl Into<String>) -> Self {
        Self {
            kind: ProviderErrorKind::MalformedResponse,
            status: Some(200),
            message: message.into(),
            malformed_reason: Some(MalformedResponseReason::Parse),
            retry_after_secs: None,
            affordable_tokens: None,
            resets_at: None,
            fallback_failures: Vec::new(),
            spent_nothing: false,
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
            resets_at: None,
            fallback_failures: Vec::new(),
            spent_nothing: false,
        }
    }

    /// Why each other model failed, grouped by reason, for a reply about the
    /// whole chain failing. `None` when no other model was tried.
    pub fn fallback_failures_note(&self) -> Option<String> {
        if self.fallback_failures.is_empty() {
            return None;
        }
        let mut grouped: Vec<(String, Vec<&str>)> = Vec::new();
        for failure in &self.fallback_failures {
            let reason = failure.reason();
            match grouped.iter_mut().find(|(known, _)| *known == reason) {
                Some((_, models)) => models.push(&failure.model),
                None => grouped.push((reason, vec![&failure.model])),
            }
        }
        let lines = grouped
            .iter()
            .map(|(reason, models)| format!("- {}: {reason}", models.join(", ")))
            .collect::<Vec<_>>()
            .join("\n");
        let hint = if self
            .fallback_failures
            .iter()
            .any(|failure| failure.kind == ProviderErrorKind::Auth)
        {
            "\nOn Telegram, /setkey replaces a rejected key."
        } else {
            ""
        };
        Some(format!("Other models also failed:\n{lines}{hint}"))
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
            ProviderErrorKind::QuotaExhausted => match self.resets_at {
                Some(at) => format!(
                    "The LLM plan's usage limit has been reached; it resets {}.",
                    describe_reset(at, std::time::SystemTime::now())
                ),
                None => "The LLM plan's usage limit has been reached; the provider did not say when it resets."
                    .to_string(),
            },
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

/// When a quota resets, for the user: the UTC time and how long until then
/// ("at 14:05 UTC (in 2h 10m)"), with the date when it is not today.
pub fn describe_reset(at: std::time::SystemTime, now: std::time::SystemTime) -> String {
    let when = chrono::DateTime::<chrono::Utc>::from(at);
    let today = chrono::DateTime::<chrono::Utc>::from(now).date_naive();
    let clock = if when.date_naive() == today {
        when.format("at %H:%M UTC").to_string()
    } else {
        when.format("on %b %-d at %H:%M UTC").to_string()
    };
    match at.duration_since(now) {
        Ok(left) => {
            let minutes = left.as_secs().div_ceil(60);
            let (days, hours, mins) = (minutes / 1440, minutes % 1440 / 60, minutes % 60);
            let left = match (days, hours) {
                (0, 0) => format!("{mins}m"),
                (0, _) => format!("{hours}h {mins}m"),
                _ => format!("{days}d {hours}h"),
            };
            format!("{clock} (in {left})")
        }
        Err(_) => clock,
    }
}

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
    fn recovery_routes_are_typed_and_single_attempt_fails_closed() {
        let transient = ProviderError::from_status(503, "busy");
        assert_eq!(
            transient.recovery_route(false),
            ProviderRecoveryRoute::RetryTransientThenFallback
        );
        assert_eq!(transient.recovery_route(true), ProviderRecoveryRoute::Fail);

        let parse = ProviderError::malformed_parse("truncated JSON");
        let shape = ProviderError::malformed_shape("missing choices");
        assert_eq!(
            parse.recovery_route(false),
            ProviderRecoveryRoute::RetryMalformedThenFallback
        );
        assert_eq!(
            shape.recovery_route(false),
            ProviderRecoveryRoute::RetryMalformedThenFail
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

    #[test]
    fn infrastructure_classification_is_typed() {
        assert!(ProviderErrorKind::Auth.is_infrastructure_failure());
        assert!(ProviderErrorKind::ServerError.is_infrastructure_failure());
        assert!(!ProviderErrorKind::BadRequest.is_infrastructure_failure());
        assert!(!ProviderErrorKind::MalformedResponse.is_infrastructure_failure());
        assert!(!ProviderErrorKind::Unknown.is_infrastructure_failure());
    }
}
