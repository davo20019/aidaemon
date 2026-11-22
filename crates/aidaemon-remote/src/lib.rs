//! Remote access over iroh: the owner's apps reach the daemon's loopback
//! gateway from anywhere by the daemon's endpoint key, with no tunnel,
//! account or domain.
//!
//! Connections go directly between the devices when the network allows and
//! through a relay otherwise; either way they are QUIC with TLS 1.3 and raw
//! public keys, so a relay sees only ciphertext. Each bidirectional stream
//! carries one HTTP/1.1 connection (with upgrades, so WebSockets work).
//!
//! A daemon's address is a [`RemoteUrl`], `iroh://<endpoint-id>@<relay>`,
//! which clients keep where they would keep an `https://` gateway address.
//!
//! The daemon's side ([`server`], feature `server`) serves the gateway's
//! own router; the transport authenticates the peer's key and never
//! authorizes, so the gateway's sign-in still decides every request. An
//! app's side is [`RemoteClient`].

use std::fmt;

use anyhow::Context as _;
use bytes::Bytes;
use http_body_util::{BodyExt as _, Full};
use hyper_util::rt::TokioIo;
use iroh::endpoint::{Connection, RecvStream, SendStream, presets};
use iroh::{Endpoint, RelayMode, TransportAddr};
use sha2::{Digest, Sha256};
use tokio_tungstenite::WebSocketStream;
use tracing::debug;

pub use iroh::{EndpointAddr, EndpointId, RelayUrl, SecretKey};

#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "server")]
pub use server::{Access, RemoteListener, RemotePeer, load_or_create_key};

/// The protocol both ends name in the handshake: the client gateway's
/// HTTP, one HTTP/1.1 connection per stream.
pub const ALPN: &[u8] = b"aidaemon/client/1";

/// The URL scheme of a [`RemoteUrl`].
pub const SCHEME: &str = "iroh";

/// The `Host` a request carries when it names none; the gateway serves the
/// same routes whatever the host.
const HOST: &str = "aidaemon.remote";

/// One stream as a byte pipe (both halves).
pub type RemoteStream = tokio::io::Join<RecvStream, SendStream>;

/// Which relays an endpoint uses to be found and to fall back on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
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

    /// Every relay in the set.
    pub fn urls(&self) -> Vec<RelayUrl> {
        self.mode().relay_map().urls()
    }
}

/// A daemon's address as clients keep it: `iroh://<endpoint-id>`, with its
/// home relay as `@<host>[:port]` (an `https` relay). The key is the
/// address; the relay is a hint, and dialing also tries every relay of the
/// dialer's own set, so a daemon that moved to another relay of that set is
/// still found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteUrl {
    pub id: EndpointId,
    pub relay: Option<RelayUrl>,
}

impl RemoteUrl {
    /// Whether `url` names a daemon by key (`iroh://…`).
    pub fn is_remote(url: &str) -> bool {
        url.trim()
            .strip_prefix(SCHEME)
            .is_some_and(|rest| rest.starts_with("://"))
    }

    /// Read `iroh://<id>[@<relay-host>][/path…]`; a path is ignored, so a
    /// socket URL reads as its daemon.
    pub fn parse(url: &str) -> Option<Self> {
        let rest = url.trim().strip_prefix(SCHEME)?.strip_prefix("://")?;
        let authority = rest.split(['/', '?', '#']).next()?;
        let (id, relay) = match authority.split_once('@') {
            Some((id, host)) => (id, Some(host)),
            None => (authority, None),
        };
        let id: EndpointId = id.parse().ok()?;
        let relay = match relay.filter(|host| !host.is_empty()) {
            Some(host) => Some(format!("https://{host}/").parse::<RelayUrl>().ok()?),
            None => None,
        };
        Some(Self { id, relay })
    }

    /// The relays a client reaching this daemon should use: n0's when the
    /// daemon's home relay is one of them or unknown, else only the
    /// daemon's own, so a daemon on self-hosted relays never puts the
    /// devices that reach it on n0's.
    pub fn dial_relays(&self) -> Relays {
        fn host(url: &RelayUrl) -> Option<String> {
            url.host_str()
                .map(|host| host.trim_end_matches('.').to_ascii_lowercase())
        }
        match &self.relay {
            Some(relay) if !Relays::N0.urls().iter().any(|n0| host(n0) == host(relay)) => {
                Relays::Custom(vec![relay.clone()])
            }
            _ => Relays::N0,
        }
    }

    /// Where to dial: the key, the relay hint, and every relay in `relays`.
    pub fn addr(&self, relays: &Relays) -> EndpointAddr {
        let mut addr = EndpointAddr::new(self.id);
        for relay in self.relay.iter().cloned().chain(relays.urls()) {
            addr.addrs.insert(TransportAddr::Relay(relay));
        }
        addr
    }
}

impl fmt::Display for RemoteUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{SCHEME}://{}", self.id)?;
        if let Some(host) = self.relay.as_ref().and_then(|relay| relay.host_str()) {
            write!(f, "@{host}")?;
            if let Some(port) = self.relay.as_ref().and_then(|relay| relay.port()) {
                write!(f, ":{port}")?;
            }
        }
        Ok(())
    }
}

/// An endpoint key as 64 hex digits: how a gateway that does not link iroh
/// records it.
pub fn endpoint_hex(id: &EndpointId) -> String {
    id.as_bytes().iter().map(|b| format!("{b:02x}")).collect()
}

/// An endpoint key derived from `seed` (a device's own secret key): the
/// same seed always gives the same key, so a paired device needs nothing
/// more kept to have a stable endpoint identity.
pub fn derive_key(seed: &[u8]) -> SecretKey {
    let mut hasher = Sha256::new();
    hasher.update(b"aidaemon/remote/device-key/1\0");
    hasher.update(seed);
    SecretKey::from_bytes(&hasher.finalize().into())
}

async fn bind(secret: SecretKey, relays: &Relays) -> anyhow::Result<Endpoint> {
    // Minimal: nothing published to a DNS server; peers dial the address
    // they were given (key, relays, direct addresses).
    Endpoint::builder(presets::Minimal)
        .secret_key(secret)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(relays.mode())
        .bind()
        .await
        .context("could not bind the iroh endpoint")
}

/// An app's side: dials daemons by address.
pub struct RemoteClient {
    endpoint: Endpoint,
    relays: Relays,
}

impl RemoteClient {
    pub async fn bind(secret: SecretKey, relays: &Relays) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: bind(secret, relays).await?,
            relays: relays.clone(),
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

    /// Connect to the daemon at `url`, trying this client's relays too.
    pub async fn connect_url(&self, url: &RemoteUrl) -> anyhow::Result<RemoteConnection> {
        self.connect(url.addr(&self.relays)).await
    }

    pub async fn close(&self) {
        self.endpoint.close().await;
    }
}

/// Which way a connection's traffic goes now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    /// Straight between the two devices.
    Direct,
    /// Through a relay, which passes it on encrypted.
    Relay,
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

    /// The path the connection uses now, `None` before one is chosen. It
    /// changes as hole punching succeeds or the network moves.
    pub fn path(&self) -> Option<PathKind> {
        let paths = self.conn.paths();
        let selected = paths.iter().find(|path| path.is_selected())?;
        Some(if selected.is_relay() {
            PathKind::Relay
        } else {
            PathKind::Direct
        })
    }

    /// Whether the connection has ended (closed, timed out, lost).
    pub fn is_closed(&self) -> bool {
        self.conn.close_reason().is_some()
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

    /// Open a WebSocket at `path` (e.g. `/client/v1/ws`) with `headers`
    /// (e.g. `authorization`).
    pub async fn websocket(
        &self,
        path: &str,
        headers: &http::HeaderMap,
    ) -> Result<WebSocketStream<RemoteStream>, WebSocketError> {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
        let stream = self
            .open_stream()
            .await
            .map_err(|e| WebSocketError::Stream(format!("{e:#}")))?;
        let mut request = format!("ws://{HOST}{path}")
            .into_client_request()
            .map_err(WebSocketError::Handshake)?;
        for (name, value) in headers {
            request.headers_mut().insert(name.clone(), value.clone());
        }
        let (socket, _) = tokio_tungstenite::client_async(request, stream)
            .await
            .map_err(WebSocketError::Handshake)?;
        Ok(socket)
    }
}

/// Why a WebSocket over iroh did not open.
#[derive(Debug)]
pub enum WebSocketError {
    /// No stream to the daemon (the connection is gone).
    Stream(String),
    /// The daemon answered the upgrade with a refusal or nonsense; a 401 is
    /// `Handshake(Error::Http(response))`.
    Handshake(tokio_tungstenite::tungstenite::Error),
}

impl fmt::Display for WebSocketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stream(why) => write!(f, "no stream to the daemon: {why}"),
            Self::Handshake(error) => write!(f, "the WebSocket handshake failed: {error}"),
        }
    }
}

impl std::error::Error for WebSocketError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_self_hosted_relay_keeps_clients_off_n0() {
        let id = SecretKey::generate().public();
        let own = RemoteUrl::parse(&format!("iroh://{id}@relay.example.test")).unwrap();
        match own.dial_relays() {
            Relays::Custom(urls) => {
                assert_eq!(urls.len(), 1);
                assert_eq!(urls[0].host_str(), Some("relay.example.test"));
            }
            other => panic!("expected the daemon's own relay, got {other:?}"),
        }
        let bare = RemoteUrl::parse(&format!("iroh://{id}")).unwrap();
        assert_eq!(bare.dial_relays(), Relays::N0);
        let n0 = Relays::N0.urls()[0].clone();
        let on_n0 = RemoteUrl::parse(&format!(
            "iroh://{id}@{}",
            n0.host_str().unwrap().trim_end_matches('.')
        ))
        .unwrap();
        assert_eq!(on_n0.dial_relays(), Relays::N0);
    }

    #[test]
    fn a_remote_url_reads_back_and_ignores_its_path() {
        let id = SecretKey::generate().public();
        let url = RemoteUrl {
            id,
            relay: Some("https://relay.example.invalid./".parse().unwrap()),
        };
        let text = url.to_string();
        assert_eq!(text, format!("iroh://{id}@relay.example.invalid."));
        assert_eq!(RemoteUrl::parse(&text), Some(url.clone()));
        assert_eq!(RemoteUrl::parse(&format!("{text}/client/v1/ws")), Some(url));
        let bare = RemoteUrl::parse(&format!("iroh://{id}")).unwrap();
        assert_eq!(bare.relay, None);
        assert!(RemoteUrl::is_remote(&text));
        assert!(!RemoteUrl::is_remote("https://relay.example.invalid"));
        assert!(RemoteUrl::parse("iroh://not-a-key@relay.example.invalid").is_none());
    }

    #[test]
    fn a_custom_relay_keeps_its_port() {
        let id = SecretKey::generate().public();
        let url = RemoteUrl::parse(&format!("iroh://{id}@relay.example.invalid:8443")).unwrap();
        assert_eq!(
            url.to_string(),
            format!("iroh://{id}@relay.example.invalid:8443")
        );
    }

    #[test]
    fn dialing_tries_the_hint_and_the_set() {
        let id = SecretKey::generate().public();
        let hint: RelayUrl = "https://hint.example.invalid./".parse().unwrap();
        let own: RelayUrl = "https://own.example.invalid./".parse().unwrap();
        let addr = RemoteUrl {
            id,
            relay: Some(hint.clone()),
        }
        .addr(&Relays::Custom(vec![own.clone()]));
        let relays: Vec<_> = addr.relay_urls().cloned().collect();
        assert!(
            relays.contains(&hint) && relays.contains(&own),
            "{relays:?}"
        );
    }

    #[test]
    fn a_derived_key_is_stable_and_seed_bound() {
        assert_eq!(
            derive_key(b"seed-1").public(),
            derive_key(b"seed-1").public()
        );
        assert_ne!(
            derive_key(b"seed-1").public(),
            derive_key(b"seed-2").public()
        );
    }
}
