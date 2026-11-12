//! Webhook mode: Telegram POSTs each update to a URL instead of the bot
//! long-polling for it.

use std::net::SocketAddr;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use serde_json::Value;

/// The header Telegram sends the webhook's secret token in.
const SECRET_HEADER: &str = "x-telegram-bot-api-secret-token";

/// Where Telegram delivers updates.
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// The local address the listener binds.
    pub listen: SocketAddr,
    /// The public HTTPS URL Telegram posts to, path included. The listener
    /// serves that path.
    pub url: String,
    /// The secret Telegram echoes on every delivery; updates without it are
    /// refused. Generated when unset.
    pub secret: Option<String>,
    /// Concurrent deliveries Telegram may open (1-100).
    pub max_connections: Option<u8>,
    /// Drop updates queued while the bot was away.
    pub drop_pending_updates: bool,
}

impl WebhookConfig {
    pub fn new(listen: SocketAddr, url: impl Into<String>) -> Self {
        Self {
            listen,
            url: url.into(),
            secret: None,
            max_connections: None,
            drop_pending_updates: false,
        }
    }

    /// The path of `url` the listener serves (`/` when it has none).
    pub(crate) fn path(&self) -> String {
        let without_scheme = self
            .url
            .split_once("://")
            .map_or(&*self.url, |(_, rest)| rest);
        match without_scheme.find('/') {
            Some(i) => {
                let path = &without_scheme[i..];
                path.split(['?', '#']).next().unwrap_or("/").to_string()
            }
            None => "/".to_string(),
        }
    }
}

/// A secret Telegram accepts: 1-256 characters from `[A-Za-z0-9_-]`.
pub(crate) fn generate_secret() -> String {
    use rand::RngExt;
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";
    let mut rng = rand::rng();
    (0..48)
        .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())] as char)
        .collect()
}

#[derive(Clone)]
struct Listener {
    secret: String,
    updates: mpsc::Sender<Value>,
}

async fn deliver(
    State(listener): State<Listener>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> StatusCode {
    let presented = headers
        .get(SECRET_HEADER)
        .and_then(|value| value.to_str().ok());
    if presented != Some(listener.secret.as_str()) {
        return StatusCode::UNAUTHORIZED;
    }
    let Ok(update) = serde_json::from_slice::<Value>(&body) else {
        // Telegram retries non-2xx deliveries; a body this adapter cannot
        // read would be retried forever, so it is accepted and dropped.
        tracing::debug!("telegram webhook: unreadable update dropped");
        return StatusCode::OK;
    };
    match listener.updates.send(update).await {
        Ok(()) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// Bind the listener. A bind failure is returned so the caller can fall
/// back to long polling.
pub(crate) async fn bind(config: &WebhookConfig) -> std::io::Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(config.listen).await
}

/// Serve deliveries on `listener` until `shutdown`, forwarding each
/// authenticated update to `updates`.
pub(crate) async fn serve(
    listener: tokio::net::TcpListener,
    path: String,
    secret: String,
    updates: mpsc::Sender<Value>,
    shutdown: CancellationToken,
) -> std::io::Result<()> {
    let app = axum::Router::new()
        .route(&path, post(deliver))
        .with_state(Listener { secret, updates });
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_listener_serves_the_urls_path() {
        let listen: SocketAddr = "127.0.0.1:8443".parse().unwrap();
        assert_eq!(
            WebhookConfig::new(listen, "https://bot.example.test/hooks/tg?x=1").path(),
            "/hooks/tg"
        );
        assert_eq!(
            WebhookConfig::new(listen, "https://bot.example.test").path(),
            "/"
        );
    }

    #[test]
    fn generated_secrets_are_valid_for_telegram() {
        let secret = generate_secret();
        assert_eq!(secret.len(), 48);
        assert!(secret
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
        assert_ne!(secret, generate_secret());
    }
}
