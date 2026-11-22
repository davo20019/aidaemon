//! Dialing a daemon by key over iroh: an address `iroh://<key>@<relay>`
//! where others are `https://…`, kept in a [`StoredDevice`] like any
//! gateway address. Sign-in, pairing and the link go over it, directly
//! between the devices when the network allows and through a relay (which
//! sees only ciphertext) otherwise.
//!
//! A paired device dials with an endpoint key derived from its own device
//! key, so the daemon ties each session it grants to that endpoint; while
//! pairing, before there is a device key, it dials with a throwaway one.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use aidaemon_remote::{
    EndpointId, Relays, RemoteClient, RemoteConnection, RemoteStream, SecretKey, WebSocketError,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use bytes::Bytes;
use http_body_util::Full;
use tokio::sync::Mutex;
use tokio_tungstenite::WebSocketStream;

pub use aidaemon_remote::RemoteUrl;

use crate::keystore::StoredDevice;
use crate::signin::{Http, HttpReply, SignInError};

/// The longest reaching a daemon may take before it reads as unreachable.
const DIAL_TIMEOUT: Duration = Duration::from_secs(20);

/// Whether `url` names a daemon by key (`iroh://…`).
pub fn is_remote(url: &str) -> bool {
    RemoteUrl::is_remote(url)
}

/// Which endpoint key dials.
#[derive(Clone)]
pub enum Dialer {
    /// A paired device: the key derived from its device key.
    Device(Box<SecretKey>),
    /// Pairing, before there is a device key: one throwaway key per run.
    Pairing,
}

impl Dialer {
    /// The dialer of a paired `device`.
    pub fn for_device(device: &StoredDevice) -> Result<Self, SignInError> {
        let seed = URL_SAFE_NO_PAD
            .decode(device.secret_key.trim())
            .map_err(|_| SignInError::Refused("the stored device key is damaged".into()))?;
        Ok(Self::Device(Box::new(aidaemon_remote::derive_key(&seed))))
    }

    fn key(&self) -> SecretKey {
        match self {
            Self::Device(key) => (**key).clone(),
            Self::Pairing => {
                static THROWAWAY: OnceLock<SecretKey> = OnceLock::new();
                THROWAWAY.get_or_init(SecretKey::generate).clone()
            }
        }
    }
}

type Key = [u8; 32];

/// One endpoint per key.
fn clients() -> &'static Mutex<HashMap<Key, Arc<RemoteClient>>> {
    static CLIENTS: OnceLock<Mutex<HashMap<Key, Arc<RemoteClient>>>> = OnceLock::new();
    CLIENTS.get_or_init(Mutex::default)
}

/// One connection per key and daemon, kept while it is open.
fn connections() -> &'static Mutex<HashMap<(Key, EndpointId), RemoteConnection>> {
    static CONNECTIONS: OnceLock<Mutex<HashMap<(Key, EndpointId), RemoteConnection>>> =
        OnceLock::new();
    CONNECTIONS.get_or_init(Mutex::default)
}

fn parse(url: &str) -> Result<RemoteUrl, SignInError> {
    RemoteUrl::parse(url)
        .ok_or_else(|| SignInError::Refused(format!("not a daemon address: {url}")))
}

/// The open connection to `url`'s daemon, dialing it when there is none.
async fn connection(url: &RemoteUrl, dialer: &Dialer) -> Result<RemoteConnection, SignInError> {
    let key = dialer.key();
    let slot = (key.to_bytes(), url.id);
    let mut open = connections().lock().await;
    if let Some(conn) = open.get(&slot)
        && !conn.is_closed()
    {
        return Ok(conn.clone());
    }
    let client = {
        let mut clients = clients().lock().await;
        match clients.get(&slot.0) {
            Some(client) => client.clone(),
            None => {
                let client = Arc::new(
                    RemoteClient::bind(key, &Relays::N0)
                        .await
                        .map_err(|e| SignInError::Failed(format!("{e:#}")))?,
                );
                clients.insert(slot.0, client.clone());
                client
            }
        }
    };
    let conn = tokio::time::timeout(DIAL_TIMEOUT, client.connect_url(url))
        .await
        .map_err(|_| SignInError::Unreachable("reaching the daemon over iroh timed out".into()))?
        .map_err(|e| SignInError::Unreachable(format!("{e:#}")))?;
    open.insert(slot, conn.clone());
    Ok(conn)
}

/// Forget the connection to `url`'s daemon, so the next call dials again.
async fn forget(url: &RemoteUrl, dialer: &Dialer) {
    connections()
        .lock()
        .await
        .remove(&(dialer.key().to_bytes(), url.id));
}

/// The path and query of `url` (`iroh://key@relay/client/v1/…`).
fn path_of(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    rest.find('/').map_or("/", |at| &rest[at..])
}

/// Sign-in and pairing's HTTP, over iroh to one daemon.
#[derive(Clone)]
pub struct RemoteHttp {
    url: RemoteUrl,
    dialer: Dialer,
}

impl RemoteHttp {
    /// HTTP to the daemon at `gateway` (`iroh://…`) as `dialer`.
    pub fn new(gateway: &str, dialer: Dialer) -> Result<Self, SignInError> {
        Ok(Self {
            url: parse(gateway)?,
            dialer,
        })
    }

    async fn call(&self, request: http::Request<Full<Bytes>>) -> Result<HttpReply, SignInError> {
        let conn = connection(&self.url, &self.dialer).await?;
        match conn.send(request).await {
            Ok(response) => Ok(HttpReply {
                status: response.status().as_u16(),
                body: String::from_utf8_lossy(response.body()).into_owned(),
            }),
            Err(error) => {
                forget(&self.url, &self.dialer).await;
                Err(SignInError::Unreachable(format!("{error:#}")))
            }
        }
    }
}

impl Http for RemoteHttp {
    async fn post_json(&self, url: &str, body: String) -> Result<HttpReply, SignInError> {
        let request = http::Request::post(path_of(url))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body)))
            .map_err(|e| SignInError::Failed(e.to_string()))?;
        self.call(request).await
    }

    async fn get(&self, url: &str) -> Result<HttpReply, SignInError> {
        let request = http::Request::get(path_of(url))
            .body(Full::default())
            .map_err(|e| SignInError::Failed(e.to_string()))?;
        self.call(request).await
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}

/// Open the client socket at `url` (`iroh://…/client/v1/ws`) with `token`.
/// A refused token is [`SignInError::Refused`].
pub async fn websocket(
    url: &str,
    token: &str,
    dialer: &Dialer,
) -> Result<WebSocketStream<RemoteStream>, SignInError> {
    let daemon = parse(url)?;
    let conn = connection(&daemon, dialer).await?;
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::AUTHORIZATION,
        format!("Bearer {token}")
            .parse()
            .map_err(|_| SignInError::Failed("the token is not a header value".into()))?,
    );
    match conn.websocket(path_of(url), &headers).await {
        Ok(socket) => Ok(socket),
        Err(WebSocketError::Handshake(tokio_tungstenite::tungstenite::Error::Http(response)))
            if response.status().as_u16() == 401 =>
        {
            Err(SignInError::Refused(
                "the daemon refused this client's token".into(),
            ))
        }
        Err(error) => {
            forget(&daemon, dialer).await;
            Err(SignInError::Unreachable(error.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_path_follows_the_authority() {
        assert_eq!(
            path_of("iroh://k@relay.example.invalid/client/v1/ws"),
            "/client/v1/ws"
        );
        assert_eq!(path_of("iroh://k@relay.example.invalid"), "/");
        assert_eq!(
            path_of("iroh://k/client/v1/enrollments/of-1"),
            "/client/v1/enrollments/of-1"
        );
    }

    #[test]
    fn a_device_dials_with_a_key_of_its_own() {
        let device = |secret: &str| StoredDevice {
            gateway: "iroh://unused".into(),
            device_id: "dev-synthetic-1".into(),
            secret_key: secret.into(),
        };
        let a = crate::signin::DeviceKey::generate().secret_b64();
        let b = crate::signin::DeviceKey::generate().secret_b64();
        let key = |secret: &str| Dialer::for_device(&device(secret)).unwrap().key().public();
        assert_eq!(key(&a), key(&a));
        assert_ne!(key(&a), key(&b));
        assert!(Dialer::for_device(&device("!!")).is_err());
    }
}
