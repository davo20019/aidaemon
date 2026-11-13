//! The connection to the gateway: signs in, connects, says hello with the
//! last `seq` it delivered, keeps the socket alive with pings, and on any
//! failure reconnects and resumes. Commands not yet answered are sent again
//! with their own `request_id` after a reconnect, so the gateway answers
//! each exactly once.
//!
//! The link always says what state it is in ([`Update`]): connecting,
//! connected, or dropped with why and when it tries again, so a client can
//! tell "the daemon is not running" from "this device was refused".

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use aidaemon_client_protocol::{
    Capability, ClientFrame, ClientInfo, ClientKind, Command, ErrorCode, ProtocolVersion,
    ServerFrame,
};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

use crate::keystore::KeyStore;
use crate::signin::{self, GrantCache, SignInError};

const PING_EVERY: Duration = Duration::from_secs(25);
/// Two missed pongs.
const SILENT_LIMIT: Duration = Duration::from_secs(60);
const RETRY_MIN: Duration = Duration::from_millis(500);
const RETRY_MAX: Duration = Duration::from_secs(10);

/// How the link proves who it is.
#[derive(Clone)]
pub enum Auth {
    /// The owner's local token (`aidaemon client-token`), as a client on the
    /// daemon's own machine uses.
    LocalToken(String),
    /// A paired device: a key in `store`, traded for a short-lived grant on
    /// each connection.
    Device(Arc<dyn KeyStore>),
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalToken(_) => f.write_str("LocalToken(<redacted>)"),
            Self::Device(_) => f.write_str("Device(..)"),
        }
    }
}

/// What the client says it is in `hello`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientIdentity {
    pub kind: ClientKind,
    /// The program's name (`aidaemon-tui`).
    pub name: String,
    pub version: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LinkConfig {
    /// `ws://127.0.0.1:8788/client/v1/ws`
    pub url: String,
    pub auth: Auth,
    pub client: ClientIdentity,
    /// The device's name as the owner knows it.
    pub device: Option<String>,
    pub capabilities: Vec<Capability>,
}

impl LinkConfig {
    /// The gateway's HTTP base for the socket address: `ws` becomes `http`,
    /// `wss` becomes `https`, the path is dropped.
    pub fn http_base(&self) -> String {
        http_base(&self.url)
    }
}

/// `ws://host:port/client/v1/ws` as `http://host:port`.
pub fn http_base(url: &str) -> String {
    let (scheme, rest) = match url.split_once("://") {
        Some(("wss", rest)) => ("https", rest),
        Some(("ws", rest)) => ("http", rest),
        Some((other, rest)) => (other, rest),
        None => ("http", url),
    };
    let host = rest.split('/').next().unwrap_or(rest);
    format!("{scheme}://{host}")
}

/// Why a connection ended, as far as a client should tell the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropKind {
    /// Nothing answered at the address: the daemon (or its gateway) is not
    /// running.
    DaemonNotRunning,
    /// The token or device was refused; retrying alone will not fix it.
    Unauthorized,
    /// The daemon speaks another major protocol version.
    VersionUnsupported,
    /// The connection was lost; it comes back by itself.
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disconnect {
    pub kind: DropKind,
    pub reason: String,
    /// When the link tries again.
    pub retry_in: Duration,
}

/// What the link tells the app.
#[allow(clippy::large_enum_variant)] // Frames dominate; boxing buys nothing here.
#[derive(Debug, Clone, PartialEq)]
pub enum Update {
    /// Signing in and opening the socket.
    Connecting,
    /// The socket is open and `hello` is sent.
    Connected,
    Disconnected(Disconnect),
    Frame(ServerFrame),
}

/// Start the link; send commands on the returned sender.
pub fn spawn(
    config: LinkConfig,
    updates: mpsc::UnboundedSender<Update>,
) -> mpsc::UnboundedSender<(String, Command)> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(run(config, updates, rx));
    tx
}

/// A new request id.
pub fn request_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

struct Link {
    config: LinkConfig,
    updates: mpsc::UnboundedSender<Update>,
    commands: mpsc::UnboundedReceiver<(String, Command)>,
    /// The last durable `seq` delivered to the app.
    cursor: Option<u64>,
    /// Commands sent but not answered, by request id.
    pending: BTreeMap<String, Command>,
    http: reqwest::Client,
    grant: GrantCache,
    /// The gateway refused the token on the last socket.
    refused: Option<String>,
}

async fn run(
    config: LinkConfig,
    updates: mpsc::UnboundedSender<Update>,
    commands: mpsc::UnboundedReceiver<(String, Command)>,
) {
    let mut link = Link {
        config,
        updates,
        commands,
        cursor: None,
        pending: BTreeMap::new(),
        http: reqwest::Client::new(),
        grant: GrantCache::default(),
        refused: None,
    };
    let mut wait = RETRY_MIN;
    loop {
        if link.updates.send(Update::Connecting).is_err() {
            return;
        }
        let (kind, reason) = match link.session().await {
            Ok(Ended::AppGone) => return,
            Ok(Ended::Dropped(kind, reason)) => {
                if kind == DropKind::Dropped {
                    wait = RETRY_MIN;
                }
                (kind, reason)
            }
            Err(failure) => failure,
        };
        if kind == DropKind::Unauthorized {
            link.grant.drop_grant();
        }
        let disconnect = Disconnect {
            kind,
            reason,
            retry_in: wait,
        };
        if link.updates.send(Update::Disconnected(disconnect)).is_err() {
            return;
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(RETRY_MAX);
    }
}

enum Ended {
    AppGone,
    Dropped(DropKind, String),
}

type Failure = (DropKind, String);

fn failed(error: impl std::fmt::Display) -> Failure {
    (DropKind::Dropped, error.to_string())
}

/// What a failed WebSocket connect means for the owner.
fn connect_failure(error: tokio_tungstenite::tungstenite::Error) -> Failure {
    use tokio_tungstenite::tungstenite::Error;
    match &error {
        Error::Io(io) if io.kind() == std::io::ErrorKind::ConnectionRefused => (
            DropKind::DaemonNotRunning,
            "nothing is listening at the daemon's address".into(),
        ),
        Error::Http(response) if response.status().as_u16() == 401 => (
            DropKind::Unauthorized,
            "the daemon refused this client's token".into(),
        ),
        _ => failed(error),
    }
}

impl Link {
    fn deliver(&mut self, frame: ServerFrame) -> bool {
        match &frame {
            ServerFrame::Event { seq: Some(seq), .. } | ServerFrame::Snapshot { seq, .. } => {
                self.cursor = Some(*seq);
            }
            ServerFrame::Ack { request_id, .. }
            | ServerFrame::Error {
                request_id: Some(request_id),
                ..
            } => {
                self.pending.remove(request_id);
            }
            ServerFrame::Error {
                request_id: None,
                error,
            } if matches!(
                error.code,
                ErrorCode::Unauthorized | ErrorCode::VersionUnsupported
            ) =>
            {
                self.refused = Some(error.message.clone());
            }
            _ => {}
        }
        self.updates.send(Update::Frame(frame)).is_ok()
    }

    /// The token for this connection.
    async fn token(&mut self) -> Result<String, Failure> {
        match &self.config.auth {
            Auth::LocalToken(token) => Ok(token.clone()),
            Auth::Device(store) => {
                let device = store
                    .load()
                    .map_err(failed)?
                    .ok_or((DropKind::Unauthorized, "this device is not paired".into()))?;
                self.grant
                    .token(&self.http, &device, signin::now_ms())
                    .await
                    .map_err(|error| match error {
                        SignInError::Unreachable(why) => (DropKind::DaemonNotRunning, why),
                        SignInError::Refused(why) => (DropKind::Unauthorized, why),
                        SignInError::Failed(why) => (DropKind::Dropped, why),
                    })
            }
        }
    }

    async fn session(&mut self) -> Result<Ended, Failure> {
        let token = self.token().await?;
        let mut request = self
            .config
            .url
            .as_str()
            .into_client_request()
            .map_err(failed)?;
        request.headers_mut().insert(
            "authorization",
            format!("Bearer {token}").parse().map_err(failed)?,
        );
        let (mut socket, _) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(connect_failure)?;
        let hello = ClientFrame::Hello {
            version: ProtocolVersion::CURRENT,
            client: ClientInfo {
                kind: self.config.client.kind,
                name: self.config.client.name.clone(),
                version: self.config.client.version.clone(),
                device: self.config.device.clone(),
            },
            capabilities: self.config.capabilities.clone(),
            resume_from: self.cursor,
            token: None,
        };
        let send = |frame: &ClientFrame| {
            serde_json::to_string(frame)
                .map(Message::text)
                .map_err(failed)
        };
        socket.send(send(&hello)?).await.map_err(failed)?;
        if self.updates.send(Update::Connected).is_err() {
            return Ok(Ended::AppGone);
        }
        self.refused = None;
        for (request_id, command) in self.pending.clone() {
            let frame = ClientFrame::Command {
                request_id,
                command,
            };
            socket.send(send(&frame)?).await.map_err(failed)?;
        }
        let mut ping = tokio::time::interval(PING_EVERY);
        ping.tick().await;
        let mut last_heard = tokio::time::Instant::now();
        let closed = |refused: &Option<String>, fallback: String| match refused {
            Some(why) => Ended::Dropped(DropKind::Unauthorized, why.clone()),
            None => Ended::Dropped(DropKind::Dropped, fallback),
        };
        loop {
            tokio::select! {
                incoming = socket.next() => {
                    let message = match incoming {
                        Some(Ok(message)) => message,
                        Some(Err(error)) => return Ok(closed(&self.refused, error.to_string())),
                        None => return Ok(closed(&self.refused, "the gateway closed the connection".into())),
                    };
                    last_heard = tokio::time::Instant::now();
                    match message {
                        Message::Text(text) => {
                            let Ok(frame) = serde_json::from_str::<ServerFrame>(&text) else {
                                continue;
                            };
                            if !self.deliver(frame) {
                                return Ok(Ended::AppGone);
                            }
                        }
                        Message::Close(_) => {
                            return Ok(closed(&self.refused, "the gateway closed the connection".into()))
                        }
                        _ => {}
                    }
                }
                command = self.commands.recv() => {
                    let Some((request_id, command)) = command else {
                        return Ok(Ended::AppGone);
                    };
                    self.pending.insert(request_id.clone(), command.clone());
                    let frame = ClientFrame::Command { request_id, command };
                    socket.send(send(&frame)?).await.map_err(failed)?;
                }
                _ = ping.tick() => {
                    if last_heard.elapsed() > SILENT_LIMIT {
                        return Ok(Ended::Dropped(DropKind::Dropped, "no answer from the gateway".into()));
                    }
                    socket.send(send(&ClientFrame::Ping)?).await.map_err(failed)?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_http_base_follows_the_socket_scheme() {
        assert_eq!(
            http_base("ws://127.0.0.1:8788/client/v1/ws"),
            "http://127.0.0.1:8788"
        );
        assert_eq!(
            http_base("wss://daemon.example.invalid/client/v1/ws"),
            "https://daemon.example.invalid"
        );
    }

    #[tokio::test]
    async fn nothing_listening_reads_as_the_daemon_not_running() {
        // Bind and drop to get a port nothing listens on.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let _commands = spawn(
            LinkConfig {
                url: format!("ws://127.0.0.1:{port}/client/v1/ws"),
                auth: Auth::LocalToken("acg_synthetic".into()),
                client: ClientIdentity {
                    kind: ClientKind::Tui,
                    name: "test".into(),
                    version: None,
                },
                device: None,
                capabilities: Vec::new(),
            },
            tx,
        );
        assert_eq!(rx.recv().await, Some(Update::Connecting));
        let Some(Update::Disconnected(disconnect)) = rx.recv().await else {
            panic!("a disconnect");
        };
        assert_eq!(disconnect.kind, DropKind::DaemonNotRunning);
    }
}
