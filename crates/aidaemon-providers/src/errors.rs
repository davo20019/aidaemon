//! Reading provider error bodies into typed [`ProviderError`]s.

use aidaemon_core::ProviderError;
use serde_json::Value;

/// A string field of the body's `error` object (`error.code`,
/// `error.type`, `error.status`, `error.message`), or of the body itself
/// when it has no `error` object (Anthropic nests it under `error` too).
fn error_field<'a>(value: &'a Value, name: &str) -> Option<&'a str> {
    value
        .pointer(&format!("/error/{name}"))
        .or_else(|| value.get(name))
        .and_then(Value::as_str)
}

/// The provider's message for a rejection: `error.message` when the body
/// is JSON, else the whole body (a proxy's plain-text answer).
fn error_message(value: Option<&Value>, body: &str) -> String {
    value
        .and_then(|value| error_field(value, "message"))
        .unwrap_or(body)
        .to_string()
}

/// Whether a rejection says the request exceeded the model's context
/// window, and the window when the provider states it.
///
/// Decided from the provider's structured fields first: OpenAI and
/// OpenAI-compatible APIs send `error.code` (or `error.type`)
/// `context_length_exceeded`. Only a 400 or 413 is read further, and then
/// only the provider's own error message, for providers that send no code
/// for it: Anthropic (`error.type` is the generic `invalid_request_error`,
/// message "prompt is too long: N tokens > M maximum"), Gemini
/// (`error.status` `INVALID_ARGUMENT`, "The input token count (N) exceeds
/// the maximum number of tokens allowed (M)") and OpenRouter (a numeric
/// `error.code`, "maximum context length is M tokens"). Those fixed
/// messages are the only signal those APIs give; they stay here until the
/// providers send a code.
pub fn context_overflow(status: u16, body: &str) -> Option<Option<u64>> {
    let value: Option<Value> = serde_json::from_str(body).ok();
    let message = error_message(value.as_ref(), body);
    let coded = value.as_ref().is_some_and(|value| {
        error_field(value, "code") == Some("context_length_exceeded")
            || error_field(value, "type") == Some("context_length_exceeded")
    });
    if !coded {
        if !matches!(status, 400 | 413) {
            return None;
        }
        let lower = message.to_ascii_lowercase();
        let stated = [
            "prompt is too long",
            "input token count",
            "maximum context length",
        ]
        .iter()
        .any(|marker| lower.contains(marker));
        if !stated {
            return None;
        }
    }
    Some(stated_context_limit(&message))
}

/// The window a provider's overflow message states, when it states one:
/// OpenAI/OpenRouter "maximum context length is 128000 tokens", Anthropic
/// "210000 tokens > 200000 maximum", Gemini "allowed (1048576)". A number,
/// read only after the rejection was classified as an overflow.
fn stated_context_limit(message: &str) -> Option<u64> {
    let lower = message.to_ascii_lowercase();
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
    number_after("maximum context length is")
        .or_else(|| number_after("tokens >"))
        .or_else(|| number_after("allowed ("))
        .filter(|limit| *limit >= 1_000)
}

/// The output tokens a 402 says the account could still pay for. Read only
/// after the 402 classified the rejection as billing; OpenRouter states the
/// number in its message alone ("can only afford 6917"), with no field.
pub fn affordable_tokens(body: &str) -> Option<u64> {
    let value: Option<Value> = serde_json::from_str(body).ok();
    let message = error_message(value.as_ref(), body);
    let marker = "can only afford ";
    let at = message.find(marker)? + marker.len();
    message[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
        .filter(|&n| n > 0)
}

/// Whether a rejection says the account cannot pay, by the provider's own
/// error code: OpenAI's `insufficient_quota` (sent as a 429) and
/// Anthropic's `billing_error`. Neither is cured by retrying, and neither
/// is the request's fault.
///
/// Anthropic also answers a used-up credit balance with a 400 whose
/// `error.type` is the generic `invalid_request_error` and whose only
/// signal is its fixed message ("Your credit balance is too low ..."); that
/// one message is read, for that status and type only.
pub fn out_of_credit(status: u16, body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let code = error_field(&value, "code");
    let kind = error_field(&value, "type");
    if code == Some("insufficient_quota")
        || kind == Some("insufficient_quota")
        || kind == Some("billing_error")
    {
        return true;
    }
    status == 400
        && kind == Some("invalid_request_error")
        && error_field(&value, "message")
            .is_some_and(|message| message.starts_with("Your credit balance is too low"))
}

/// The typed error for a non-success HTTP status and its body.
pub fn error_for_status(status: u16, body: &str, detail: String) -> ProviderError {
    if out_of_credit(status, body) {
        return ProviderError::Billing {
            message: detail,
            affordable_tokens: None,
        };
    }
    if let Some(limit) = context_overflow(status, body) {
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
            context_overflow(
                400,
                r#"{"error":{"code":400,"message":"This endpoint's maximum context length is 128,000 tokens."}}"#
            ),
            Some(Some(128_000))
        );
        assert_eq!(
            context_overflow(
                400,
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}"#
            ),
            Some(Some(200_000))
        );
        assert_eq!(
            context_overflow(
                400,
                r#"{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."}}"#
            ),
            Some(Some(1_048_576))
        );
        assert_eq!(context_overflow(429, "rate limit"), None);
    }

    /// The provider's code decides, whatever its message says: a reworded
    /// message with `context_length_exceeded` is still an overflow, and
    /// the overflow wording on a status other than 400/413 is not.
    #[test]
    fn overflow_is_decided_by_the_code_before_any_wording() {
        let reworded = r#"{"error":{"code":"context_length_exceeded","message":"Your request is bigger than this model accepts."}}"#;
        assert_eq!(context_overflow(400, reworded), Some(None));
        assert_eq!(context_overflow(500, reworded), Some(None));
        let echoed =
            r#"{"error":{"type":"server_error","message":"upstream said: prompt is too long"}}"#;
        assert_eq!(context_overflow(500, echoed), None);
        assert_eq!(context_overflow(400, "bad tool schema"), None);
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
            error_for_status(
                400,
                r#"{"error":{"code":"context_length_exceeded","message":"too big"}}"#,
                "d".into()
            ),
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

    #[test]
    fn an_account_out_of_credit_is_billing_whatever_its_status() {
        let anthropic = r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}"#;
        assert!(matches!(
            error_for_status(400, anthropic, "d".into()),
            ProviderError::Billing {
                affordable_tokens: None,
                ..
            }
        ));
        let openai = r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","code":"insufficient_quota"}}"#;
        assert!(out_of_credit(429, openai));
        assert!(!out_of_credit(
            429,
            r#"{"error":{"type":"rate_limit_error"}}"#
        ));
        assert!(out_of_credit(
            402,
            r#"{"type":"error","error":{"type":"billing_error","message":"Payment needed"}}"#
        ));
        // The fixed message counts only on Anthropic's 400 with its type,
        // never echoed inside some other failure.
        let echoed =
            r#"{"error":{"type":"server_error","message":"Your credit balance is too low"}}"#;
        assert!(!out_of_credit(500, echoed));
        assert!(!out_of_credit(400, "Your credit balance is too low"));
    }
}
