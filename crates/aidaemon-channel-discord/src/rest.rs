//! Minimal Discord REST client with one internal 429 retry.

use std::time::Duration;

use aidaemon_channels::ChannelError;
use reqwest::{Method, StatusCode};
use serde_json::Value;

/// Longest `retry_after` honoured internally; longer waits surface as
/// [`ChannelError::RateLimited`] right away.
const MAX_INTERNAL_RETRY_WAIT: Duration = Duration::from_secs(30);

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
        self.send_with(method, path, |req| match body {
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

    /// Fetch a file from Discord's CDN (attachment URLs need no token).
    pub(crate) async fn download(&self, url: &str) -> Result<Vec<u8>, ChannelError> {
        let resp = self
            .http
            .get(url)
            .header("User-Agent", USER_AGENT)
            .send()
            .await
            .map_err(|e| ChannelError::Transport(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(ChannelError::Other(format!(
                "discord: attachment download failed: {status}"
            )));
        }
        resp.bytes()
            .await
            .map(|bytes| bytes.to_vec())
            .map_err(|e| ChannelError::Transport(e.to_string()))
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
                        })
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
