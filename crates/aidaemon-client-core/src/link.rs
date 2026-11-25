//! The connection to the gateway: signs in, connects, says hello with the
//! last `seq` it delivered, keeps the socket alive with pings, and on any
//! failure reconnects and resumes. Commands not yet answered are sent again
//! with their own `request_id` after a reconnect, so the gateway answers
//! each exactly once.
//!
//! The link always says what state it is in ([`Update`]): connecting,
//! connected, or dropped with why and when it tries again, so a client can
//! tell "the daemon is not running" from "this device was refused".

use std::sync::Arc;

use aidaemon_client_protocol::{Capability, ClientFrame, Command};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

use crate::keystore::KeyStore;
pub use crate::machine::{
    ClientIdentity, Disconnect, DropKind, LinkMachine, Tick, Update, http_base, request_id,
};
use crate::signin::{self, GrantCache, SignInError};

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

/// The longest signing in (a challenge and a grant) may take.
const SIGN_IN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// The longest opening the socket may take.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Start the link; send commands on the returned sender.
pub fn spawn(
    config: LinkConfig,
    updates: mpsc::UnboundedSender<Update>,
) -> mpsc::UnboundedSender<(String, Command)> {
    spawn_wakeable(config, updates).0
}

/// Tells the link the device woke (the app came back to the foreground, the
/// network returned): a link waiting to try again tries now, and a quiet
/// open socket is pinged and dropped when it does not answer within a few
/// of its measured round trips ([`LinkMachine::wake_answer`],
/// [`LinkMachine::woke`]).
#[derive(Clone, Default)]
pub struct Waker(Arc<tokio::sync::Notify>);

impl Waker {
    pub fn wake(&self) {
        self.0.notify_one();
    }
}

/// [`spawn`], with the [`Waker`] to call when the device wakes.
pub fn spawn_wakeable(
    config: LinkConfig,
    updates: mpsc::UnboundedSender<Update>,
) -> (mpsc::UnboundedSender<(String, Command)>, Waker) {
    let (tx, rx) = mpsc::unbounded_channel();
    let waker = Waker::default();
    tokio::spawn(run(config, updates, rx, waker.clone()));
    (tx, waker)
}

struct Link {
    config: LinkConfig,
    updates: mpsc::UnboundedSender<Update>,
    commands: mpsc::UnboundedReceiver<(String, Command)>,
    machine: LinkMachine,
    http: reqwest::Client,
    grant: GrantCache,
    started: tokio::time::Instant,
    waker: Waker,
}

async fn run(
    config: LinkConfig,
    updates: mpsc::UnboundedSender<Update>,
    commands: mpsc::UnboundedReceiver<(String, Command)>,
    waker: Waker,
) {
    let machine = LinkMachine::new(
        config.client.clone(),
        config.device.clone(),
        config.capabilities.clone(),
    );
    let mut link = Link {
        config,
        updates,
        commands,
        machine,
        http: crate::tls::http_client(),
        grant: GrantCache::default(),
        started: tokio::time::Instant::now(),
        waker,
    };
    loop {
        if link.updates.send(Update::Connecting).is_err() {
            return;
        }
        let disconnect = match link.session().await {
            Ok(Ended::AppGone) => return,
            Ok(Ended::Closed(disconnect)) => disconnect,
            Err((kind, reason)) => link.machine.failed(kind, reason),
        };
        if disconnect.kind == DropKind::Unauthorized {
            link.grant.drop_grant();
        }
        let wait = disconnect.retry_in;
        if link.updates.send(Update::Disconnected(disconnect)).is_err() {
            return;
        }
        // A wake (the app back in front) cuts the wait short.
        let woke = link.waker.0.clone();
        tokio::select! {
            _ = tokio::time::sleep(wait) => {}
            _ = woke.notified() => {
                let now = link.now_ms();
                link.machine.woke(now);
            }
        }
    }
}

enum Ended {
    AppGone,
    Closed(Disconnect),
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

/// What a failed sign-in means for the owner.
fn sign_in_failure(error: SignInError) -> Failure {
    match error {
        SignInError::Unreachable(why) => (DropKind::DaemonNotRunning, why),
        SignInError::Refused(why) => (DropKind::Unauthorized, why),
        SignInError::Failed(why) => (DropKind::Dropped, why),
    }
}

impl Link {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
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
                #[cfg(feature = "remote")]
                if crate::remote::is_remote(&device.gateway) {
                    let dialer =
                        crate::remote::Dialer::for_device(&device).map_err(sign_in_failure)?;
                    let http = crate::remote::RemoteHttp::new(&device.gateway, dialer)
                        .map_err(sign_in_failure)?;
                    return self
                        .grant
                        .token_with(&http, &device, signin::now_ms())
                        .await
                        .map_err(sign_in_failure);
                }
                self.grant
                    .token(&self.http, &device, signin::now_ms())
                    .await
                    .map_err(sign_in_failure)
            }
        }
    }

    async fn session(&mut self) -> Result<Ended, Failure> {
        // A network that swallows packets would hold the link in
        // "connecting" for good: give up, say why, and try again.
        let token = tokio::time::timeout(SIGN_IN_TIMEOUT, self.token())
            .await
            .map_err(|_| {
                (
                    DropKind::DaemonNotRunning,
                    "signing in to the daemon timed out".to_string(),
                )
            })??;
        #[cfg(feature = "remote")]
        if crate::remote::is_remote(&self.config.url) {
            let dialer = match &self.config.auth {
                Auth::Device(store) => {
                    let device = store
                        .load()
                        .map_err(failed)?
                        .ok_or((DropKind::Unauthorized, "this device is not paired".into()))?;
                    crate::remote::Dialer::for_device(&device).map_err(sign_in_failure)?
                }
                Auth::LocalToken(_) => {
                    return Err((
                        DropKind::Unauthorized,
                        "the local token is for the daemon's own machine; pair this device".into(),
                    ));
                }
            };
            let socket = tokio::time::timeout(
                CONNECT_TIMEOUT,
                crate::remote::websocket(&self.config.url, &token, &dialer),
            )
            .await
            .map_err(|_| {
                (
                    DropKind::DaemonNotRunning,
                    "connecting to the daemon timed out".to_string(),
                )
            })?
            .map_err(sign_in_failure)?;
            return self.drive(socket).await;
        }
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
        let tls = crate::tls::client_config().map_err(failed)?;
        let (socket, _) = tokio::time::timeout(
            CONNECT_TIMEOUT,
            tokio_tungstenite::connect_async_tls_with_config(
                request,
                None,
                false,
                Some(tokio_tungstenite::Connector::Rustls(tls)),
            ),
        )
        .await
        .map_err(|_| {
            (
                DropKind::DaemonNotRunning,
                "connecting to the daemon timed out".to_string(),
            )
        })?
        .map_err(connect_failure)?;
        self.drive(socket).await
    }

    /// Say hello on an open socket, then pass frames both ways until it
    /// ends.
    async fn drive<S>(
        &mut self,
        mut socket: tokio_tungstenite::WebSocketStream<S>,
    ) -> Result<Ended, Failure>
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        let send = |frame: &ClientFrame| {
            serde_json::to_string(frame)
                .map(Message::text)
                .map_err(failed)
        };
        let opening = self.machine.opened(self.now_ms());
        let mut frames = opening.iter();
        if let Some(hello) = frames.next() {
            socket.send(send(hello)?).await.map_err(failed)?;
        }
        if self.updates.send(Update::Connected).is_err() {
            return Ok(Ended::AppGone);
        }
        for frame in frames {
            socket.send(send(frame)?).await.map_err(failed)?;
        }
        let mut ticks = tokio::time::interval(crate::machine::TICK_EVERY);
        ticks.tick().await;
        let woke = self.waker.0.clone();
        loop {
            tokio::select! {
                _ = woke.notified() => {
                    if let Some(ping) = self.machine.woke(self.now_ms()) {
                        socket.send(send(&ping)?).await.map_err(failed)?;
                    }
                }
                incoming = socket.next() => {
                    let message = match incoming {
                        Some(Ok(message)) => message,
                        Some(Err(error)) => return Ok(Ended::Closed(self.machine.closed(error.to_string()))),
                        None => return Ok(Ended::Closed(self.machine.closed("the gateway closed the connection"))),
                    };
                    let now = self.now_ms();
                    match message {
                        Message::Text(text) => {
                            if let Some(frame) = self.machine.received(&text, now)
                                && self.updates.send(Update::Frame(frame)).is_err()
                            {
                                return Ok(Ended::AppGone);
                            }
                        }
                        Message::Close(_) => {
                            return Ok(Ended::Closed(self.machine.closed("the gateway closed the connection")))
                        }
                        _ => self.machine.heard(now),
                    }
                }
                command = self.commands.recv() => {
                    let Some((request_id, command)) = command else {
                        return Ok(Ended::AppGone);
                    };
                    if let Some(frame) = self.machine.command(request_id, command) {
                        socket.send(send(&frame)?).await.map_err(failed)?;
                    }
                }
                _ = ticks.tick() => {
                    match self.machine.tick(self.now_ms()) {
                        Tick::Wait => {}
                        Tick::Ping(frame) => socket.send(send(&frame)?).await.map_err(failed)?,
                        Tick::Dead(disconnect) => return Ok(Ended::Closed(disconnect)),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    kind: aidaemon_client_protocol::ClientKind::Tui,
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

    #[tokio::test]
    async fn a_wake_ends_the_wait_before_the_next_try() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (_commands, waker) = spawn_wakeable(
            LinkConfig {
                url: format!("ws://127.0.0.1:{port}/client/v1/ws"),
                auth: Auth::LocalToken("acg_synthetic".into()),
                client: ClientIdentity {
                    kind: aidaemon_client_protocol::ClientKind::Mobile,
                    name: "test".into(),
                    version: None,
                },
                device: None,
                capabilities: Vec::new(),
            },
            tx,
        );
        // Fail until the backoff is long, as in the background.
        let wait = loop {
            if let Some(Update::Disconnected(disconnect)) = rx.recv().await
                && disconnect.retry_in >= std::time::Duration::from_secs(2)
            {
                break disconnect.retry_in;
            }
        };
        let woken = std::time::Instant::now();
        waker.wake();
        assert_eq!(rx.recv().await, Some(Update::Connecting));
        assert!(woken.elapsed() < wait / 2, "tried again at once");
    }
}
