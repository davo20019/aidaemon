//! Minimal Discord REST client with one internal 429 retry.

use std::time::Duration;

use aidaemon_channels::ChannelError;
use aidaemon_channels::download::{DownloadClient, DownloadRequest};
use reqwest::{Method, StatusCode, Url};
use serde_json::Value;

/// Longest `retry_after` honoured internally; longer waits surface as
/// [`ChannelError::RateLimited`] right away.
const MAX_INTERNAL_RETRY_WAIT: Duration = Duration::from_secs(30);

const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// Domains Discord serves attachments from.
const DISCORD_FILE_DOMAINS: [&str; 3] = ["discord.com", "discordapp.com", "discordapp.net"];

const USER_AGENT: &str = concat!(
    "DiscordBot (https://github.com/davo20019/aidaemon, ",
    env!("CARGO_PKG_VERSION"),
    ")"
);

#[derive(Clone)]
pub(crate) struct Rest {
    pub(crate) http: reqwest::Client,
    pub(crate) api_base: String,
    pub(crate) token: String,
    /// Follows no redirect without checking it first.
    pub(crate) files: DownloadClient,
}

fn retry_after(headers: &reqwest::header::HeaderMap, body: &Value) -> Option<f64> {
    body.get("retry_after").and_then(Value::as_f64).or_else(|| {
        headers
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<f64>().ok())
    })
}

impl Rest {
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_base.trim_end_matches('/'), path)
    }

    /// Send a request; returns the JSON body (`Null` for empty bodies).
    pub(crate) async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, ChannelError> {
        // The one redaction boundary every request to the chat passes.
        let body = body.map(aidaemon_channels::outbound::redact_body);
        self.send_with(method, path, |req| match &body {
            Some(b) => req.json(b),
            None => req.body(Vec::<u8>::new()),
        })
        .await
    }

    /// POST a message with one file (`multipart/form-data`): the JSON
    /// payload as `payload_json` and the file as `files[0]`.
    pub(crate) async fn upload(
        &self,
        path: &str,
        payload: &Value,
        file_name: &str,
        bytes: &[u8],
        mime_type: Option<&str>,
    ) -> Result<Value, ChannelError> {
        let payload = &aidaemon_channels::outbound::redact_body(payload);
        self.send_with(Method::POST, path, |req| {
            let mut part =
                reqwest::multipart::Part::bytes(bytes.to_vec()).file_name(file_name.to_string());
            if let Some(mime) = mime_type {
                part = part.mime_str(mime).unwrap_or_else(|_| {
                    reqwest::multipart::Part::bytes(bytes.to_vec()).file_name(file_name.to_string())
                });
            }
            let form = reqwest::multipart::Form::new()
                .text("payload_json", payload.to_string())
                .part("files[0]", part);
            req.multipart(form)
        })
        .await
    }

    /// Whether `url` is one of Discord's file hosts (or the configured API
    /// host). The gateway names these URLs, but a file is only ever fetched
    /// from Discord.
    fn discord_host(&self, url: &Url) -> bool {
        let Some(host) = url.host_str() else {
            return false;
        };
        let on_discord = url.scheme() == "https"
            && DISCORD_FILE_DOMAINS
                .iter()
                .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")));
        let configured = Url::parse(&self.api_base)
            .ok()
            .and_then(|base| base.host_str().map(str::to_string));
        on_discord || configured.as_deref() == Some(host)
    }

    /// Fetch a file from Discord's CDN. Attachment URLs need no token, and
    /// none is sent.
    pub(crate) async fn download(
        &self,
        url: &str,
        max_bytes: u64,
    ) -> Result<Vec<u8>, ChannelError> {
        self.files
            .fetch(DownloadRequest {
                url,
                bearer_token: None,
                allowed: &|target| self.discord_host(target),
                max_bytes,
                timeout: DOWNLOAD_TIMEOUT,
                user_agent: Some(USER_AGENT),
            })
            .await
            .map(|downloaded| downloaded.bytes)
    }

    /// Send a request built by `body` (called again on the one retry).
    async fn send_with(
        &self,
        method: Method,
        path: &str,
        body: impl Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
    ) -> Result<Value, ChannelError> {
        let mut retried = false;
        loop {
            let req = self
                .http
                .request(method.clone(), self.url(path))
                .header("Authorization", format!("Bot {}", self.token))
                .header("User-Agent", USER_AGENT);
            let req = body(req);
            let resp = req
                .send()
                .await
                .map_err(|e| ChannelError::Transport(e.to_string()))?;
            let status = resp.status();
            let headers = resp.headers().clone();
            let bytes = resp
                .bytes()
                .await
                .map_err(|e| ChannelError::Transport(e.to_string()))?;
            let json: Value = if bytes.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&bytes).unwrap_or(Value::Null)
            };

            if status.is_success() {
                return Ok(json);
            }
            if status == StatusCode::TOO_MANY_REQUESTS {
                let wait = retry_after(&headers, &json);
                let wait_dur = wait.map(|s| Duration::from_secs_f64(s.max(0.0)));
                match wait_dur {
                    Some(d) if !retried && d <= MAX_INTERNAL_RETRY_WAIT => {
                        tracing::debug!(path, wait_secs = d.as_secs_f64(), "discord 429, retrying");
                        retried = true;
                        tokio::time::sleep(d).await;
                        continue;
                    }
                    _ => {
                        return Err(ChannelError::RateLimited {
                            retry_after_secs: wait.map(|s| s.max(0.0).ceil() as u64),
                        });
                    }
                }
            }
            let message = json
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
            return Err(match status {
                StatusCode::UNAUTHORIZED => ChannelError::Auth(message),
                s if s.is_server_error() => {
                    ChannelError::Transport(format!("discord {method} {path}: {s}: {message}"))
                }
                s => ChannelError::Other(format!("discord {method} {path}: {s}: {message}")),
            });
        }
    }
}

/// Check a bot token against `users/@me`; returns the bot's name.
pub async fn validate_token(token: &str) -> Result<String, String> {
    let client = reqwest::Client::new();
    let response = client
        .get("https://discord.com/api/v10/users/@me")
        .header("Authorization", format!("Bot {}", token))
        .send()
        .await
        .map_err(|e| format!("Failed to validate token: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Invalid Discord token (HTTP {}). Make sure you copied the bot token from Discord Developer Portal.",
            response.status()
        ));
    }

    match response.json::<serde_json::Value>().await {
        Ok(json) => Ok(json["username"].as_str().unwrap_or("unknown").to_string()),
        Err(_) => Ok("unknown".to_string()),
    }
}
