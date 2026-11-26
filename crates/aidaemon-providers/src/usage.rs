//! What a provider account has used and has left, where the provider says:
//! an OpenRouter key's spend and balance ([`openrouter_usage`]). Other
//! providers report no public balance/window endpoint (Anthropic only
//! per-minute rate limits, ChatGPT plan usage lives in ChatGPT settings,
//! Gemini nothing), so they have no reader here.

use std::time::{Duration, SystemTime};

use aidaemon_core::ProviderError;
use serde_json::Value;

use crate::transport::{build_client, map_reqwest_error, map_status};

const USAGE_TIMEOUT: Duration = Duration::from_secs(10);

/// What an account has used and has left.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AccountUsage {
    /// The plan, as the provider names it ("plus", "pro"), when it says.
    pub plan: Option<String>,
    /// Rolling limits, shortest window first.
    pub windows: Vec<UsageWindow>,
    /// Money spent and left, for an account billed from a balance.
    pub balance: Option<Balance>,
}

/// One rolling usage limit.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageWindow {
    /// How long the window is.
    pub length: Duration,
    /// How much of it is used, 0 to 100.
    pub used_percent: f64,
    /// When it starts over, when the provider says.
    pub resets_at: Option<SystemTime>,
}

/// Money spent and left, in US dollars.
#[derive(Debug, Clone, PartialEq)]
pub struct Balance {
    /// Spent so far.
    pub used: f64,
    /// The most that may be spent; `None`: no cap.
    pub limit: Option<f64>,
    /// Left to spend; `None`: no cap.
    pub remaining: Option<f64>,
}

/// An OpenRouter key's spend and cap as `GET /key` answers:
/// `{"data": {"usage", "limit", "limit_remaining"}}` (dollars; a `null`
/// limit is no cap).
pub fn parse_openrouter_key(body: &str) -> Option<Balance> {
    let value: Value = serde_json::from_str(body).ok()?;
    let data = value.get("data")?;
    let number = |name: &str| {
        data.get(name)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
    };
    Some(Balance {
        used: number("usage")?,
        limit: number("limit"),
        remaining: number("limit_remaining").map(|v| v.max(0.0)),
    })
}

/// An OpenRouter account's credits as `GET /credits` answers:
/// `{"data": {"total_credits", "total_usage"}}` (dollars).
pub fn parse_openrouter_credits(body: &str) -> Option<Balance> {
    let value: Value = serde_json::from_str(body).ok()?;
    let data = value.get("data")?;
    let number = |name: &str| {
        data.get(name)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
    };
    let total = number("total_credits")?;
    let used = number("total_usage")?;
    Some(Balance {
        used,
        limit: Some(total),
        remaining: Some((total - used).max(0.0)),
    })
}

/// An OpenRouter key's balance: the key's own cap when it has one, else
/// the account's credits. `base_url` is the API root
/// (`https://openrouter.ai/api/v1`).
pub async fn openrouter_usage(
    base_url: &str,
    api_key: &str,
) -> Result<AccountUsage, ProviderError> {
    let base = base_url.trim().trim_end_matches('/');
    let client =
        build_client(USAGE_TIMEOUT).map_err(|e| ProviderError::Transient(e.to_string()))?;
    let get = |path: &'static str| {
        let client = client.clone();
        async move {
            let response = client
                .get(format!("{base}{path}"))
                .timeout(USAGE_TIMEOUT)
                .bearer_auth(api_key)
                .send()
                .await
                .map_err(map_reqwest_error)?;
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.text().await.unwrap_or_default();
            if status.is_success() {
                Ok(body)
            } else {
                Err(map_status(status, &headers, &body))
            }
        }
    };
    let key = parse_openrouter_key(&get("/key").await?).ok_or_else(|| {
        ProviderError::InvalidRequest("OpenRouter's key answer had no usage".into())
    })?;
    // A key with no cap of its own draws on the account: its credits say
    // what is left. A key that may not read them still shows its spend.
    let balance = if key.limit.is_some() {
        key
    } else {
        match get("/credits")
            .await
            .ok()
            .and_then(|b| parse_openrouter_credits(&b))
        {
            Some(credits) => credits,
            None => key,
        }
    };
    Ok(AccountUsage {
        plan: None,
        windows: Vec::new(),
        balance: Some(balance),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_openrouter_key_reads_its_cap_or_none() {
        let capped = parse_openrouter_key(
            r#"{"data": {"label": "k", "usage": 3.5, "limit": 10, "limit_remaining": 6.5}}"#,
        )
        .unwrap();
        assert_eq!(
            capped,
            Balance {
                used: 3.5,
                limit: Some(10.0),
                remaining: Some(6.5)
            }
        );
        let open = parse_openrouter_key(
            r#"{"data": {"usage": 1.25, "limit": null, "limit_remaining": null}}"#,
        )
        .unwrap();
        assert_eq!(open.limit, None);
        assert_eq!(open.remaining, None);
    }

    #[test]
    fn openrouter_credits_say_what_is_left() {
        let credits =
            parse_openrouter_credits(r#"{"data": {"total_credits": 20, "total_usage": 7.5}}"#)
                .unwrap();
        assert_eq!(credits.remaining, Some(12.5));
        assert_eq!(credits.limit, Some(20.0));
    }
}
