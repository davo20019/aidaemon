//! What a provider account has used and has left, where the provider says:
//! a ChatGPT plan's rolling limits ([`ChatGpt::usage`](crate::ChatGpt::usage))
//! and an OpenRouter key's spend and balance ([`openrouter_usage`]).
//! Other providers report no such thing to an API key (Anthropic only
//! per-minute rate limits, Gemini nothing), so they have no reader here.

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

/// The `wham/usage` address for a Codex base URL
/// (`https://chatgpt.com/backend-api/codex` → `.../backend-api/wham/usage`).
pub(crate) fn chatgpt_usage_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let base = base.strip_suffix("/codex").unwrap_or(base);
    format!("{base}/wham/usage")
}

/// A ChatGPT plan's usage as `wham/usage` answers:
/// `{"plan_type", "rate_limit": {"primary_window": {"used_percent",
/// "limit_window_seconds", "reset_at" | "reset_after_seconds"},
/// "secondary_window": ..}}`. `now` dates a reset given as a delay.
pub fn parse_chatgpt_usage(body: &str, now: SystemTime) -> Option<AccountUsage> {
    let value: Value = serde_json::from_str(body).ok()?;
    let limits = value.get("rate_limit");
    let mut windows: Vec<UsageWindow> = ["primary_window", "secondary_window"]
        .iter()
        .filter_map(|name| window(limits?.get(*name)?, now))
        .collect();
    windows.sort_by_key(|w| w.length);
    let plan = value
        .get("plan_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|plan| !plan.is_empty())
        .map(str::to_string);
    (plan.is_some() || !windows.is_empty()).then_some(AccountUsage {
        plan,
        windows,
        balance: None,
    })
}

fn window(value: &Value, now: SystemTime) -> Option<UsageWindow> {
    let number = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v >= 0.0)
    };
    let used_percent = number("used_percent")?.min(100.0);
    let length = Duration::from_secs_f64(number("limit_window_seconds")?);
    let resets_at = number("reset_at")
        .filter(|secs| *secs > 0.0)
        .map(|secs| SystemTime::UNIX_EPOCH + Duration::from_secs_f64(secs))
        .or_else(|| number("reset_after_seconds").map(|secs| now + Duration::from_secs_f64(secs)));
    Some(UsageWindow {
        length,
        used_percent,
        resets_at,
    })
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

    fn at(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
    }

    #[test]
    fn the_usage_address_sits_beside_the_codex_base() {
        assert_eq!(
            chatgpt_usage_url("https://chatgpt.com/backend-api/codex"),
            "https://chatgpt.com/backend-api/wham/usage"
        );
        assert_eq!(
            chatgpt_usage_url("https://example.test/backend-api/"),
            "https://example.test/backend-api/wham/usage"
        );
    }

    #[test]
    fn a_chatgpt_plan_reads_its_windows_shortest_first() {
        let body = r#"{
            "plan_type": "plus",
            "rate_limit": {
                "allowed": true,
                "limit_reached": false,
                "secondary_window": {"used_percent": 18, "limit_window_seconds": 604800,
                                     "reset_after_seconds": 3600},
                "primary_window": {"used_percent": 42.5, "limit_window_seconds": 18000,
                                   "reset_at": 1700000100}
            },
            "credits": {"has_credits": false, "unlimited": false, "balance": null}
        }"#;
        let usage = parse_chatgpt_usage(body, at(1_000)).unwrap();
        assert_eq!(usage.plan.as_deref(), Some("plus"));
        assert_eq!(
            usage.windows,
            vec![
                UsageWindow {
                    length: Duration::from_secs(18_000),
                    used_percent: 42.5,
                    resets_at: Some(at(1_700_000_100)),
                },
                UsageWindow {
                    length: Duration::from_secs(604_800),
                    used_percent: 18.0,
                    resets_at: Some(at(4_600)),
                },
            ]
        );
    }

    #[test]
    fn a_chatgpt_answer_without_limits_or_plan_is_none() {
        assert_eq!(parse_chatgpt_usage(r#"{"rate_limit": null}"#, at(0)), None);
        assert_eq!(parse_chatgpt_usage("not json", at(0)), None);
        // A window missing its length is skipped, the plan kept.
        let usage = parse_chatgpt_usage(
            r#"{"plan_type": "pro", "rate_limit": {"primary_window": {"used_percent": 5}}}"#,
            at(0),
        )
        .unwrap();
        assert!(usage.windows.is_empty());
    }

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
