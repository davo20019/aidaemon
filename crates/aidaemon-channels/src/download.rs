//! Bounded file downloads for adapters (feature `http`).
//!
//! A platform file URL needs a credential: a bearer header (Slack) or a
//! token in the URL itself (Telegram). The shared rules live here:
//!
//! - the size is bounded while reading, so an oversized file is never
//!   buffered;
//! - every hop, including the first, must be on a host the adapter allows,
//!   so a redirect cannot carry the credential elsewhere;
//! - no `Referer` is sent, and no error mentions a URL or a response body.

use std::time::Duration;

use reqwest::header::{CONTENT_TYPE, LOCATION, RETRY_AFTER};
use reqwest::redirect::Policy;
use reqwest::{StatusCode, Url};

use crate::channel::ChannelError;

/// Redirects followed before giving up.
const MAX_REDIRECTS: usize = 3;

/// Says whether a URL may be fetched with this request's credential.
pub type HostCheck<'a> = &'a (dyn Fn(&Url) -> bool + Send + Sync);

/// One file to fetch.
pub struct DownloadRequest<'a> {
    /// May hold a token: it is never logged or put in an error.
    pub url: &'a str,
    /// Sent as `Authorization: Bearer` on every allowed hop.
    pub bearer_token: Option<&'a str>,
    /// Applied to the URL and to every redirect target.
    pub allowed: HostCheck<'a>,
    pub max_bytes: u64,
    /// For the whole transfer, per hop.
    pub timeout: Duration,
    pub user_agent: Option<&'a str>,
}

/// What a download returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Downloaded {
    pub bytes: Vec<u8>,
    /// The `Content-Type` the server sent.
    pub content_type: Option<String>,
}

/// An HTTP client that follows no redirects by itself, so that
/// [`DownloadClient::fetch`] can judge each one. A client passed in from
/// outside could not be trusted to do that, so it takes a builder.
#[derive(Clone)]
pub struct DownloadClient {
    client: Option<reqwest::Client>,
}

impl Default for DownloadClient {
    fn default() -> Self {
        Self::from_builder(reqwest::Client::builder())
    }
}

impl DownloadClient {
    /// Whatever `builder` sets (proxies, TLS roots) is kept. Its redirect
    /// and referer settings are replaced.
    pub fn from_builder(builder: reqwest::ClientBuilder) -> Self {
        Self {
            client: builder.redirect(Policy::none()).referer(false).build().ok(),
        }
    }

    pub async fn fetch(&self, request: DownloadRequest<'_>) -> Result<Downloaded, ChannelError> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| ChannelError::Other("download: no HTTP client".into()))?;
        let mut url = Url::parse(request.url)
            .map_err(|_| ChannelError::Other("download: invalid URL".into()))?;
        if !(request.allowed)(&url) {
            return Err(ChannelError::Other(
                "download refused: not one of the platform's hosts".into(),
            ));
        }
        for _ in 0..=MAX_REDIRECTS {
            let mut get = client.get(url.clone()).timeout(request.timeout);
            if let Some(token) = request.bearer_token {
                get = get.bearer_auth(token);
            }
            if let Some(agent) = request.user_agent {
                get = get.header(reqwest::header::USER_AGENT, agent);
            }
            let response = get.send().await.map_err(transport_error)?;
            let status = response.status();
            if is_followed_redirect(status) {
                url = redirect_target(&url, &response)?;
                if !(request.allowed)(&url) {
                    return Err(ChannelError::Other(
                        "download refused: redirected off the platform's hosts".into(),
                    ));
                }
                continue;
            }
            if !status.is_success() {
                return Err(status_error(status, &response));
            }
            return read_bounded(response, request.max_bytes).await;
        }
        Err(ChannelError::Other("download: too many redirects".into()))
    }
}

fn is_followed_redirect(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    )
}

fn redirect_target(from: &Url, response: &reqwest::Response) -> Result<Url, ChannelError> {
    response
        .headers()
        .get(LOCATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|location| from.join(location).ok())
        .ok_or_else(|| ChannelError::Other("download: unusable redirect".into()))
}

fn status_error(status: StatusCode, response: &reqwest::Response) -> ChannelError {
    let text = format!("download: HTTP {}", status.as_u16());
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ChannelError::Auth(text),
        StatusCode::TOO_MANY_REQUESTS => ChannelError::RateLimited {
            retry_after_secs: response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse().ok()),
        },
        status if status.is_server_error() => ChannelError::Transport(text),
        _ => ChannelError::Other(text),
    }
}

async fn read_bounded(
    mut response: reqwest::Response,
    max_bytes: u64,
) -> Result<Downloaded, ChannelError> {
    let too_large = || ChannelError::TooLarge {
        limit_bytes: max_bytes,
    };
    if response.content_length().is_some_and(|len| len > max_bytes) {
        return Err(too_large());
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
        if bytes.len() as u64 + chunk.len() as u64 > max_bytes {
            return Err(too_large());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(Downloaded {
        bytes,
        content_type,
    })
}

fn transport_error(error: reqwest::Error) -> ChannelError {
    ChannelError::Transport(format!("download: {}", error.without_url()))
}
