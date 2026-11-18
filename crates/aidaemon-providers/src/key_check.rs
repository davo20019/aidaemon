//! A read-only check of a candidate API key against its provider: one
//! request that authenticates with the key, and what the answer says about
//! it. Applications use it to test a key before saving it.

use std::time::Duration;

use serde_json::Value;

use crate::OPENAI_DEFAULT_BASE_URL;
use crate::routing::ProviderKind;

const CHECK_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_KEY_CHARS: usize = 1024;
/// Sent to endpoints that may not check keys at all, to find out.
const CONTROL_KEY: &str = "aidaemon-control-key-not-valid";

/// A provider endpoint that authenticates with an API key.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyEndpoint {
    kind: ProviderKind,
    base_url: String,
}

/// What a provider said about a candidate key.
#[derive(Debug, PartialEq, Eq)]
pub enum KeyCheck {
    /// The provider accepted the key.
    Accepted,
    /// The provider refused the key (the detail says why).
    Rejected(String),
    /// The provider gave no answer about the key.
    Unverified(String),
}

impl KeyEndpoint {
    /// `None` for providers that log in some other way.
    pub fn new(kind: ProviderKind, base_url: &str) -> Option<Self> {
        (kind != ProviderKind::OpenaiChatgpt).then(|| Self {
            kind,
            base_url: base_url.trim().trim_end_matches('/').to_string(),
        })
    }

    /// The configured URL when it is not the serde default every entry gets.
    fn base_override(&self) -> Option<&str> {
        (!self.base_url.is_empty() && self.base_url != OPENAI_DEFAULT_BASE_URL)
            .then_some(self.base_url.as_str())
    }

    fn host(&self) -> Option<String> {
        reqwest::Url::parse(&self.base_url)
            .ok()?
            .host_str()
            .map(str::to_string)
    }

    fn is_openrouter(&self) -> bool {
        self.host().as_deref() == Some("openrouter.ai")
    }

    /// The provider's name, for the owner.
    pub fn provider_name(&self) -> String {
        match self.kind {
            ProviderKind::Anthropic => "Anthropic".to_string(),
            ProviderKind::GoogleGenai => "Google".to_string(),
            ProviderKind::XaiNative => "xAI".to_string(),
            _ if self.is_openrouter() => "OpenRouter".to_string(),
            _ if self.base_url == OPENAI_DEFAULT_BASE_URL => "OpenAI".to_string(),
            _ => self.host().unwrap_or_else(|| self.base_url.clone()),
        }
    }

    /// A read-only request that authenticates with `key`.
    fn check_request(&self, client: &reqwest::Client, key: &str) -> reqwest::RequestBuilder {
        match self.kind {
            ProviderKind::Anthropic => {
                let base = self.base_override().unwrap_or("https://api.anthropic.com");
                client
                    .get(format!(
                        "{}/v1/models?limit=1",
                        base.trim_end_matches("/v1")
                    ))
                    .header("x-api-key", key)
                    .header("anthropic-version", "2023-06-01")
            }
            ProviderKind::GoogleGenai => {
                let base = self
                    .base_override()
                    .unwrap_or("https://generativelanguage.googleapis.com");
                let base = base.trim_end_matches("/v1beta").trim_end_matches("/v1");
                client
                    .get(format!("{base}/v1beta/models?pageSize=1"))
                    .header("x-goog-api-key", key)
            }
            ProviderKind::XaiNative => {
                let base = self.base_override().unwrap_or("https://api.x.ai/v1");
                client.get(format!("{base}/models")).bearer_auth(key)
            }
            _ => {
                // OpenRouter's model list is public; its key endpoint is not.
                let path = if self.is_openrouter() {
                    "key"
                } else {
                    "models"
                };
                client
                    .get(format!("{}/{path}", self.base_url))
                    .bearer_auth(key)
            }
        }
    }

    /// Whether a success could just mean the endpoint ignores keys.
    fn needs_control_check(&self) -> bool {
        matches!(self.kind, ProviderKind::OpenaiCompatible) && !self.is_openrouter()
    }
}

/// At most `max` characters of `text`, ending in "..." when cut.
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max.saturating_sub(3)).collect();
    format!("{kept}...")
}

/// The provider's own error text, without the key if it echoed it back.
fn provider_detail(status: u16, body: &str, key: &str) -> String {
    let message = serde_json::from_str::<Value>(body).ok().and_then(|value| {
        let error = value.get("error")?;
        error
            .get("message")
            .or(Some(error))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    match message {
        Some(message) => format!(
            "HTTP {status}: {}",
            truncate(&message.replace(key, "[key]"), 160)
        ),
        None => format!("HTTP {status}"),
    }
}

/// Ask the provider whether it accepts `key`, with one read-only request.
pub async fn check_key(target: &KeyEndpoint, key: &str) -> KeyCheck {
    if key.is_empty()
        || key.chars().count() > MAX_KEY_CHARS
        || key.chars().any(|c| c.is_whitespace() || c.is_control())
    {
        return KeyCheck::Rejected("that does not look like an API key".to_string());
    }
    let Ok(client) = reqwest::Client::builder().timeout(CHECK_TIMEOUT).build() else {
        return KeyCheck::Unverified("could not start the check".to_string());
    };
    let send = |candidate: &str| target.check_request(&client, candidate).send();
    let response = match send(key).await {
        Ok(response) => response,
        Err(error) if error.is_timeout() => {
            return KeyCheck::Unverified("the provider timed out".to_string());
        }
        Err(_) => return KeyCheck::Unverified("could not reach the provider".to_string()),
    };
    let status = response.status();
    if status.is_success() {
        if target.needs_control_check()
            && send(CONTROL_KEY)
                .await
                .is_ok_and(|control| control.status().is_success())
        {
            return KeyCheck::Unverified(
                "this endpoint accepts any key, so it cannot confirm this one".to_string(),
            );
        }
        return KeyCheck::Accepted;
    }
    let body = response.text().await.unwrap_or_default();
    let detail = provider_detail(status.as_u16(), &body, key);
    if matches!(status.as_u16(), 401 | 403) {
        KeyCheck::Rejected(detail)
    } else {
        KeyCheck::Unverified(detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// Answers each request by the key it carried.
    fn server(answer: fn(&str) -> (u16, String)) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut buffer = [0u8; 4096];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]).to_string();
                let (status, body) = answer(&request);
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        base
    }

    fn target(kind: ProviderKind, base_url: &str) -> KeyEndpoint {
        KeyEndpoint::new(kind, base_url).unwrap()
    }

    #[tokio::test]
    async fn a_rejected_key_is_reported_without_echoing_it() {
        let base = server(|_| {
            (
                401,
                r#"{"error":{"message":"bad key synthetic-key-1"}}"#.to_string(),
            )
        });
        let check = check_key(&target(ProviderKind::Anthropic, &base), "synthetic-key-1").await;
        assert_eq!(
            check,
            KeyCheck::Rejected("HTTP 401: bad key [key]".to_string())
        );
    }

    #[tokio::test]
    async fn an_endpoint_that_accepts_anything_cannot_confirm_a_key() {
        let base = server(|_| (200, "{}".to_string()));
        let check = check_key(
            &target(ProviderKind::OpenaiCompatible, &base),
            "synthetic-key",
        )
        .await;
        assert!(matches!(check, KeyCheck::Unverified(_)), "{check:?}");
    }

    #[tokio::test]
    async fn a_key_only_counts_when_the_control_is_refused() {
        let base = server(|request| {
            if request.contains(CONTROL_KEY) {
                (401, "{}".to_string())
            } else {
                (200, "{}".to_string())
            }
        });
        let check = check_key(
            &target(ProviderKind::OpenaiCompatible, &base),
            "synthetic-key",
        )
        .await;
        assert_eq!(check, KeyCheck::Accepted);
    }

    #[tokio::test]
    async fn a_provider_outage_is_not_a_verdict_on_the_key() {
        let base = server(|_| (503, "{}".to_string()));
        let check = check_key(&target(ProviderKind::Anthropic, &base), "synthetic-key").await;
        assert_eq!(check, KeyCheck::Unverified("HTTP 503".to_string()));
    }

    #[tokio::test]
    async fn text_that_cannot_be_a_key_is_refused_without_a_request() {
        let check = check_key(
            &target(ProviderKind::Anthropic, "http://127.0.0.1:1"),
            "hello there",
        )
        .await;
        assert!(matches!(check, KeyCheck::Rejected(_)), "{check:?}");
    }
}
