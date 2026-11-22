//! Remote access over iroh: the owner's apps reach the daemon's loopback
//! gateway from anywhere by the daemon's endpoint key, with no tunnel,
//! account or domain.
//!
//! Connections go directly between the devices when the network allows and
//! through a relay otherwise; either way they are QUIC with TLS 1.3 and raw
//! public keys, so a relay sees only ciphertext. Each bidirectional stream
//! carries one HTTP/1.1 connection (with upgrades, so WebSockets work),
//! served by the gateway's own [`axum::Router`].
//!
//! The transport authenticates the peer's key; it never authorizes. The
//! gateway's tokens and signed challenges still decide what a caller may do.
//! [`Access`] only narrows which keys may open a connection, and handlers
//! see the peer's key as a [`RemotePeer`] request extension.

use std::fmt;
use std::sync::Arc;

use anyhow::Context as _;
use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use iroh::endpoint::{Connection, RecvStream, SendStream, presets};
use iroh::{Endpoint, RelayMode};
use tokio_tungstenite::WebSocketStream;
use tracing::debug;

pub use iroh::{EndpointAddr, EndpointId, RelayUrl, SecretKey};

/// The protocol both ends name in the handshake: the client gateway's
/// HTTP, one HTTP/1.1 connection per stream.
pub const ALPN: &[u8] = b"aidaemon/client/1";

/// The `Host` a request carries when it names none; the gateway serves the
/// same routes whatever the host.
const HOST: &str = "aidaemon.remote";

/// One stream as a byte pipe (both halves).
pub type RemoteStream = tokio::io::Join<RecvStream, SendStream>;

/// Which relays an endpoint uses to be found and to fall back on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Relays {
    /// n0's public relays.
    #[default]
    N0,
    /// Your own `iroh-relay` servers.
    Custom(Vec<RelayUrl>),
    /// None: direct addresses only (the local network, tests).
    Disabled,
}

impl Relays {
    fn mode(&self) -> RelayMode {
        match self {
            Self::N0 => RelayMode::Default,
            Self::Custom(urls) => RelayMode::custom(urls.iter().cloned()),
            Self::Disabled => RelayMode::Disabled,
        }
    }
}

/// Which peers may open a connection.
#[derive(Clone, Default)]
pub enum Access {
    /// Any key; the gateway's own sign-in still applies.
    #[default]
    Any,
    /// Only keys the check accepts, e.g. paired devices.
    Only(Arc<dyn Fn(&EndpointId) -> bool + Send + Sync>),
}

impl Access {
    fn allows(&self, peer: &EndpointId) -> bool {
        match self {
            Self::Any => true,
            Self::Only(check) => check(peer),
        }
    }
}

impl fmt::Debug for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Any => "Access::Any",
            Self::Only(_) => "Access::Only(..)",
        })
    }
}

/// The endpoint key kept at `path` (hex, owner-only), made on first use.
/// The key is the daemon's address for life: a new one unpairs every app.
pub fn load_or_create_key(path: &std::path::Path) -> anyhow::Result<SecretKey> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let key: SecretKey = text
                .trim()
                .parse()
                .with_context(|| format!("{} holds no endpoint key", path.display()))?;
            Ok(key)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let key = SecretKey::generate();
            write_owner_only(path, &hex(&key.to_bytes())).with_context(|| {
                format!("could not save the endpoint key at {}", path.display())
            })?;
            Ok(key)
        }
        Err(error) => Err(error)
            .with_context(|| format!("could not read the endpoint key at {}", path.display())),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn write_owner_only(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    // Created owner-only, never widened and narrowed after.
    let _ = std::fs::remove_file(&tmp);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    {
        use std::io::Write;
        let mut file = options.open(&tmp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// The key of the peer a request came from, as a request extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemotePeer(pub EndpointId);

async fn bind(secret: SecretKey, relays: &Relays) -> anyhow::Result<Endpoint> {
    // Minimal: no address publishing to a DNS server; peers dial the
    // address they were given (key, relay, direct addresses).
    Endpoint::builder(presets::Minimal)
        .secret_key(secret)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(relays.mode())
        .bind()
        .await
        .context("could not bind the iroh endpoint")
}

/// The daemon's side: serves a router to the peers [`Access`] lets in.
pub struct RemoteListener {
    endpoint: Endpoint,
    accept: tokio::task::JoinHandle<()>,
}

impl RemoteListener {
    pub async fn bind(
        secret: SecretKey,
        relays: &Relays,
        access: Access,
        app: axum::Router,
    ) -> anyhow::Result<Self> {
        let endpoint = bind(secret, relays).await?;
        let accept = tokio::spawn(accept_loop(endpoint.clone(), access, app));
        Ok(Self { endpoint, accept })
    }

    /// The daemon's key: its address for life.
    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// Where peers dial now: the key, the home relay once connected, and
    /// direct addresses.
    pub fn addr(&self) -> EndpointAddr {
        self.endpoint.addr()
    }

    /// Wait until the endpoint is connected to its home relay.
    pub async fn online(&self) {
        self.endpoint.online().await;
    }

    pub async fn close(self) {
        self.accept.abort();
        self.endpoint.close().await;
    }
}

async fn accept_loop(endpoint: Endpoint, access: Access, app: axum::Router) {
    while let Some(incoming) = endpoint.accept().await {
        let access = access.clone();
        let app = app.clone();
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(conn) => conn,
                Err(error) => {
                    debug!(%error, "Remote connection failed during the handshake");
                    return;
                }
            };
            let peer = conn.remote_id();
            if !access.allows(&peer) {
                debug!(peer = %peer.fmt_short(), "Remote connection refused: key not allowed");
                conn.close(1u32.into(), b"not allowed");
                return;
            }
            debug!(peer = %peer.fmt_short(), "Remote connection accepted");
            serve_connection(conn, peer, app).await;
        });
    }
}

async fn serve_connection(conn: Connection, peer: EndpointId, app: axum::Router) {
    let app = app.layer(axum::Extension(RemotePeer(peer)));
    while let Ok((send, recv)) = conn.accept_bi().await {
        let service = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            let io = TokioIo::new(tokio::io::join(recv, send));
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .with_upgrades()
                .await
            {
                debug!(%error, "Remote stream ended with an error");
            }
        });
    }
}

/// An app's side: dials daemons by address.
pub struct RemoteClient {
    endpoint: Endpoint,
}

impl RemoteClient {
    pub async fn bind(secret: SecretKey, relays: &Relays) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: bind(secret, relays).await?,
        })
    }

    /// This device's key, as the daemon sees it.
    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// Connect to a daemon. The handshake fails unless the peer holds the
    /// key in `daemon`.
    pub async fn connect(
        &self,
        daemon: impl Into<EndpointAddr>,
    ) -> anyhow::Result<RemoteConnection> {
        let conn = self
            .endpoint
            .connect(daemon, ALPN)
            .await
            .context("could not connect to the daemon")?;
        Ok(RemoteConnection { conn })
    }

    pub async fn close(&self) {
        self.endpoint.close().await;
    }
}

/// A connection to one daemon; each request or socket opens its own stream.
#[derive(Clone)]
pub struct RemoteConnection {
    conn: Connection,
}

impl RemoteConnection {
    /// The daemon's key (verified in the handshake).
    pub fn daemon(&self) -> EndpointId {
        self.conn.remote_id()
    }

    /// A fresh stream, for a caller that speaks HTTP itself.
    pub async fn open_stream(&self) -> anyhow::Result<RemoteStream> {
        let (send, recv) = self
            .conn
            .open_bi()
            .await
            .context("could not open a stream")?;
        Ok(tokio::io::join(recv, send))
    }

    /// Send one HTTP request and read the whole response.
    pub async fn send(
        &self,
        mut request: http::Request<Full<Bytes>>,
    ) -> anyhow::Result<http::Response<Bytes>> {
        if !request.headers().contains_key(http::header::HOST) {
            request
                .headers_mut()
                .insert(http::header::HOST, http::HeaderValue::from_static(HOST));
        }
        let io = TokioIo::new(self.open_stream().await?);
        let (mut sender, connection) = hyper::client::conn::http1::handshake(io).await?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                debug!(%error, "Remote request stream ended with an error");
            }
        });
        let response = sender.send_request(request).await?;
        let (parts, body) = response.into_parts();
        let body = body.collect().await?.to_bytes();
        Ok(http::Response::from_parts(parts, body))
    }

    /// Open a WebSocket at `path` (e.g. `/client/v1/ws`).
    pub async fn websocket(&self, path: &str) -> anyhow::Result<WebSocketStream<RemoteStream>> {
        let stream = self.open_stream().await?;
        let (socket, _) = tokio_tungstenite::client_async(format!("ws://{HOST}{path}"), stream)
            .await
            .context("the WebSocket handshake failed")?;
        Ok(socket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_is_made_once_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("remote-key");
        let first = load_or_create_key(&path).unwrap();
        let again = load_or_create_key(&path).unwrap();
        assert_eq!(first.public(), again.public());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn a_damaged_key_file_is_an_error_not_a_new_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("remote-key");
        std::fs::write(&path, "not a key").unwrap();
        assert!(load_or_create_key(&path).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "not a key");
    }
}
