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
use iroh::address_lookup::pkarr::{PkarrPublisher, PkarrResolver};
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

/// The aidaemon project's relays: `iroh-relay` servers with QUIC address
/// discovery on its default port.
pub const AIDAEMON_RELAYS: &[&str] = &["https://relay.aidaemon.ai/"];

/// The project's address lookup (a pkarr relay): a daemon on the project's
/// relays publishes which relay it is on, signed with its key (the relay URL
/// only, never an IP address), and an app looks the daemon's key up when it
/// dials, so a daemon that moved relays is still found from an old address.
pub const AIDAEMON_LOOKUP: &str = "https://dns.aidaemon.ai/pkarr";

/// Which relays an endpoint uses to be found and to fall back on.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub enum Relays {
    /// The aidaemon project's relays ([`AIDAEMON_RELAYS`]).
    #[default]
    Aidaemon,
    /// n0's public relays, which daemons used before the project ran its
    /// own; a device paired then still names one in its address.
    N0,
    /// Your own `iroh-relay` servers.
    Custom(Vec<RelayUrl>),
    /// None: direct addresses only (the local network, tests).
    Disabled,
}

impl Relays {
    fn mode(&self) -> RelayMode {
        match self {
            Self::Aidaemon => RelayMode::custom(aidaemon_relays()),
            Self::N0 => RelayMode::Default,
            Self::Custom(urls) => RelayMode::custom(urls.iter().cloned()),
            Self::Disabled => RelayMode::Disabled,
        }
    }

    /// The address lookup that goes with these relays: the project's for
    /// its relays; none for your own (your devices dial the relay they
    /// paired with) or for none.
    fn lookup(&self) -> Option<url::Url> {
        match self {
            Self::Aidaemon => Some(
                AIDAEMON_LOOKUP
                    .parse()
                    .expect("AIDAEMON_LOOKUP is a valid URL"),
            ),
            Self::N0 | Self::Custom(_) | Self::Disabled => None,
        }
    }

    /// Every relay in the set.
    pub fn urls(&self) -> Vec<RelayUrl> {
        self.mode().relay_map().urls()
    }
}

fn aidaemon_relays() -> impl Iterator<Item = RelayUrl> {
    AIDAEMON_RELAYS
        .iter()
        .map(|url| url.parse().expect("AIDAEMON_RELAYS are valid relay URLs"))
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

    /// The relays a client reaching this daemon should use: the project's
    /// (and its address lookup) when the daemon's home relay is one of them,
    /// unknown, or an n0 relay from before the project ran its own (the
    /// hint stays in [`Self::addr`], so a daemon still there is reached
    /// there and one that moved is found by its key); else only the
    /// daemon's own, so a daemon on self-hosted relays never puts the
    /// devices that reach it on anyone else's.
    pub fn dial_relays(&self) -> Relays {
        fn host(url: &RelayUrl) -> Option<String> {
            url.host_str()
                .map(|host| host.trim_end_matches('.').to_ascii_lowercase())
        }
        let among =
            |set: &Relays, relay: &RelayUrl| set.urls().iter().any(|u| host(u) == host(relay));
        match &self.relay {
            Some(relay) if !among(&Relays::Aidaemon, relay) && !among(&Relays::N0, relay) => {
                Relays::Custom(vec![relay.clone()])
            }
            _ => Relays::Aidaemon,
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

/// Which end of a connection an endpoint is: a daemon publishes where it
/// is, an app only looks daemons up (so no lookup learns of the device).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    Daemon,
    App,
}

async fn bind(secret: SecretKey, relays: &Relays, side: Side) -> anyhow::Result<Endpoint> {
    // Minimal: no n0 services. Peers dial the address they were given
    // (key, relays, direct addresses), plus the address lookup that goes
    // with the relay set, if any.
    let mut builder = Endpoint::builder(presets::Minimal)
        .secret_key(secret)
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(relays.mode());
    if let Some(lookup) = relays.lookup() {
        builder = match side {
            Side::Daemon => builder.address_lookup(PkarrPublisher::builder(lookup)),
            Side::App => builder.address_lookup(PkarrResolver::builder(lookup)),
        };
    }
    builder
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
            endpoint: bind(secret, relays, Side::App).await?,
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
        assert_eq!(bare.dial_relays(), Relays::Aidaemon);
        let ours = RemoteUrl::parse(&format!("iroh://{id}@relay.aidaemon.ai")).unwrap();
        assert_eq!(ours.dial_relays(), Relays::Aidaemon);
    }

    #[test]
    fn an_address_from_n0_days_dials_the_projects_relays_and_keeps_its_hint() {
        let id = SecretKey::generate().public();
        let n0 = Relays::N0.urls()[0].clone();
        let on_n0 = RemoteUrl::parse(&format!(
            "iroh://{id}@{}",
            n0.host_str().unwrap().trim_end_matches('.')
        ))
        .unwrap();
        let relays = on_n0.dial_relays();
        assert_eq!(relays, Relays::Aidaemon);
        assert!(relays.lookup().is_some());
        let tried: Vec<_> = on_n0.addr(&relays).relay_urls().cloned().collect();
        let bare = |u: &RelayUrl| u.host_str().map(|h| h.trim_end_matches('.').to_owned());
        assert!(tried.iter().any(|u| bare(u) == bare(&n0)));
        assert!(
            tried
                .iter()
                .any(|u| u.host_str() == Some("relay.aidaemon.ai"))
        );
    }

    #[test]
    fn only_the_projects_relays_come_with_its_lookup() {
        assert_eq!(Relays::Aidaemon.lookup().unwrap().as_str(), AIDAEMON_LOOKUP);
        let own = Relays::Custom(vec!["https://relay.example.test/".parse().unwrap()]);
        assert_eq!(own.lookup(), None);
        assert_eq!(Relays::N0.lookup(), None);
        assert_eq!(Relays::Disabled.lookup(), None);
    }

    #[test]
    fn the_default_relays_are_the_projects() {
        let urls = Relays::default().urls();
        assert_eq!(urls.len(), AIDAEMON_RELAYS.len());
        assert_eq!(urls[0].host_str(), Some("relay.aidaemon.ai"));
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
