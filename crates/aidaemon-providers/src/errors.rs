//! Reading provider error bodies into typed [`ProviderError`]s.

use aidaemon_core::ProviderError;

/// Whether an error text says the request exceeded the model's context
/// window, and the window when the text states it. Matches the providers'
/// own error codes and fixed messages (OpenAI `context_length_exceeded`,
/// Anthropic "prompt is too long", Gemini's token-count error), not free
/// text.
pub fn context_overflow(error: &str) -> Option<Option<u64>> {
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
    let number_after = |marker: &str| -> Option<u64> {
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

/// The output tokens a 402 says the account could still pay for
/// (OpenRouter: "can only afford 6917").
pub fn affordable_tokens(body: &str) -> Option<u64> {
    let marker = "can only afford ";
    let at = body.find(marker)? + marker.len();
    body[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
        .filter(|&n| n > 0)
}

/// The typed error for a non-success HTTP status and its body.
pub fn error_for_status(status: u16, body: &str, detail: String) -> ProviderError {
    if let Some(limit) = context_overflow(body) {
        return ProviderError::ContextOverflow {
            message: detail,
            limit,
        };
    }
    match status {
        401 | 403 => ProviderError::Auth(detail),
        402 => ProviderError::Billing {
            message: detail,
            affordable_tokens: affordable_tokens(body),
        },
        404 => ProviderError::NotFound(detail),
        408 => ProviderError::Transient(detail),
        400..=499 => ProviderError::InvalidRequest(detail),
        500..=599 => ProviderError::Transient(detail),
        _ => ProviderError::Other(detail),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_reads_each_providers_wording_and_limit() {
        assert_eq!(
            context_overflow("This model's maximum context length is 128,000 tokens."),
            Some(Some(128_000))
        );
        assert_eq!(
            context_overflow("prompt is too long: 210000 tokens > 200000 maximum"),
            Some(Some(200_000))
        );
        assert_eq!(
            context_overflow("The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."),
            Some(Some(1_048_576))
        );
        assert_eq!(
            context_overflow("code: context_length_exceeded"),
            Some(None)
        );
        assert_eq!(context_overflow("rate limit"), None);
    }

    #[test]
    fn statuses_map_to_typed_errors() {
        assert!(matches!(
            error_for_status(
                402,
                r#"{"error":{"message":"can only afford 6917"}}"#,
                "d".into()
            ),
            ProviderError::Billing {
                affordable_tokens: Some(6917),
                ..
            }
        ));
        assert!(matches!(
            error_for_status(404, "no such model", "d".into()),
            ProviderError::NotFound(_)
        ));
        assert!(matches!(
            error_for_status(400, "context_length_exceeded", "d".into()),
            ProviderError::ContextOverflow { limit: None, .. }
        ));
        assert!(matches!(
            error_for_status(400, "bad tool schema", "d".into()),
            ProviderError::InvalidRequest(_)
        ));
        assert!(matches!(
            error_for_status(503, "", "d".into()),
            ProviderError::Transient(_)
        ));
    }
}
