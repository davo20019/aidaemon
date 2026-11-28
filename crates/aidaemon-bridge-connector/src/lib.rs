//! The daemon's side of e-ink access.
//!
//! The connector dials out to an `aidaemon-bridge` (so the daemon needs no
//! open port or tunnel), logs in with its own login key, and for each
//! browser the bridge joins to it completes the handshake of
//! `aidaemon-bridge-proto` (`SPEC.md`), then serves the device's requests
//! through the gateway's bridge router
//! ([`aidaemon_client_gateway::bridge_router`]): sign-in, commands and
//! message files only, behind the same peer guard as remote access, with
//! the device key the handshake proved as the peer.
//!
//! Nothing here decides what a device may do: the gateway does, from the
//! device's record. The connector only proves who is on the other end and
//! keeps the bytes private from the bridge.

mod code;
mod keys;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_bridge_proto::frame::{self, MAX_DATA_FRAME, MAX_RESPONSE};
use aidaemon_bridge_proto::handshake::{self, DaemonChannel, Hello};
use aidaemon_bridge_proto::{ids, message};
use aidaemon_client_gateway::{BridgePairer, ClientGateway};
use aidaemon_client_protocol::{ErrorCode, ProtocolError};
use futures::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;
use tower::ServiceExt;
use tracing::{info, warn};

pub use code::{PairingCode, parse_code};
pub use keys::BridgeKeys;

/// How long a handshake may take, end to end.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a paired device's connection may sit idle.
const IDLE_TIMEOUT: Duration = Duration::from_secs(600);
/// How long the owner's pairing waits for the device.
const PAIR_TIMEOUT: Duration = Duration::from_secs(120);
/// Frames waiting for one device's connection task.
const CONN_QUEUE: usize = 64;
/// Frames waiting to go out on the bridge socket.
const OUT_QUEUE: usize = 256;
/// Browser connections served at once: the project's bridge allows 8 per
/// daemon; this bounds what a misbehaving bridge can make the daemon hold.
const MAX_CONNS: usize = 32;

/// The device key a handshake proved (`device_id`, 32 bytes): the peer the
/// gateway's guard binds sign-in tokens to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BridgePeer(pub [u8; 32]);

impl AsRef<[u8]> for BridgePeer {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

enum Out {
    Text(String),
    Frame(u32, Vec<u8>),
}

struct PendingJoin {
    code: PairingCode,
    done: oneshot::Sender<Result<String, ProtocolError>>,
}

/// The connector: one per daemon while e-ink access is on.
pub struct Connector {
    gateway: Arc<ClientGateway>,
    url: String,
    keys: BridgeKeys,
    router: axum::Router,
    route_id: [u8; 32],
    daemon_id: [u8; 32],
    link: Mutex<Option<mpsc::Sender<Out>>>,
    joins: Mutex<HashMap<String, PendingJoin>>,
    /// The task that keeps the bridge connection; [`Self::stop`] ends it.
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl Connector {
    /// Start dialing `url` (`wss://<bridge>/v1/daemon`) and register with
    /// the gateway: devices may then sign in to `bridge:<daemon id>`, and
    /// the owner may pair one. Runs until [`Self::stop`] or the process
    /// ends.
    pub fn start(gateway: Arc<ClientGateway>, url: String, keys: BridgeKeys) -> Arc<Self> {
        let route_id = ids::route_id(&keys.login_pub());
        let daemon_id = ids::daemon_id(&keys.identity_pub());
        let router = aidaemon_client_gateway::bridge_router::<BridgePeer>(gateway.clone());
        let connector = Arc::new(Self {
            gateway: gateway.clone(),
            url,
            keys,
            router,
            route_id,
            daemon_id,
            link: Mutex::new(None),
            joins: Mutex::new(HashMap::new()),
            task: Mutex::new(None),
        });
        let pairing = gateway.pairing();
        pairing.set_bridge_id(Some(hex::encode(daemon_id)));
        pairing.set_bridge_pairer(Some(connector.clone()));
        let running = connector.clone();
        let task = tokio::spawn(async move { running.run().await });
        *connector.task.lock().unwrap_or_else(|p| p.into_inner()) = Some(task);
        connector
    }

    /// E-ink access turned off: end the bridge connection (and with it
    /// every device's connection through it, and a pairing under way), and
    /// take the bridge's sign-in and pairing away from the gateway. Paired
    /// devices stay listed; they connect again once a connector starts.
    pub fn stop(&self) {
        if let Some(task) = self.task.lock().unwrap_or_else(|p| p.into_inner()).take() {
            // Dropping the session drops each device connection's queue,
            // so their tasks end and the writer closes the socket.
            task.abort();
        }
        *self.link.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self.fail_joins("E-ink access was turned off");
        let pairing = self.gateway.pairing();
        pairing.set_bridge_id(None);
        pairing.set_bridge_pairer(None);
        info!("E-ink access off: the bridge connection is closed");
    }

    /// Whether the connection task still runs (not stopped).
    pub fn is_running(&self) -> bool {
        self.task
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .is_some_and(|task| !task.is_finished())
    }

    /// The route id browsers connect to (`?route=`), as hex.
    pub fn route_id_hex(&self) -> String {
        hex::encode(self.route_id)
    }

    /// The daemon id devices sign in to (`bridge:<id>`), as hex.
    pub fn daemon_id_hex(&self) -> String {
        hex::encode(self.daemon_id)
    }

    async fn run(self: Arc<Self>) {
        let mut backoff = Duration::from_secs(1);
        loop {
            match self.clone().session().await {
                Ok(()) => backoff = Duration::from_secs(1),
                Err(error) => warn!(%error, "The e-ink bridge connection ended"),
            }
            *self.link.lock().unwrap_or_else(|p| p.into_inner()) = None;
            self.fail_joins("The bridge connection dropped");
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(60));
        }
    }

    async fn session(self: Arc<Self>) -> anyhow::Result<()> {
        let (socket, _) = tokio_tungstenite::connect_async_tls_with_config(
            self.url.as_str(),
            None,
            false,
            Some(tokio_tungstenite::Connector::Rustls(tls_config()?)),
        )
        .await?;
        let (mut sink, mut stream) = socket.split();

        // Login: sign the bridge's nonce with the login key (and nothing else).
        let challenge = next_json(&mut stream).await?;
        anyhow::ensure!(challenge["type"] == "challenge", "expected a challenge");
        let nonce: [u8; 32] = hex::decode(challenge["nonce"].as_str().unwrap_or_default())?
            .try_into()
            .map_err(|_| anyhow::anyhow!("bad nonce"))?;
        let signature = aidaemon_bridge_proto::login::sign_login(&self.keys.login, &nonce);
        let login = serde_json::json!({
            "type": "login",
            "login_pub": hex::encode(self.keys.login_pub()),
            "sig": hex::encode(signature),
        });
        sink.send(Message::text(login.to_string())).await?;
        let ready = next_json(&mut stream).await?;
        anyhow::ensure!(
            ready["type"] == "ready" && ready["route_id"] == hex::encode(self.route_id),
            "the bridge did not accept the login"
        );
        info!("Connected to the e-ink bridge");

        let (out_tx, mut out_rx) = mpsc::channel::<Out>(OUT_QUEUE);
        let writer = tokio::spawn(async move {
            while let Some(out) = out_rx.recv().await {
                let message = match out {
                    Out::Text(text) => Message::text(text),
                    Out::Frame(conn, frame) => {
                        let mut bytes = Vec::with_capacity(4 + frame.len());
                        bytes.extend_from_slice(&conn.to_be_bytes());
                        bytes.extend_from_slice(&frame);
                        Message::binary(bytes)
                    }
                };
                if sink.send(message).await.is_err() {
                    break;
                }
            }
        });
        *self.link.lock().unwrap_or_else(|p| p.into_inner()) = Some(out_tx.clone());

        let mut conns: HashMap<u32, mpsc::Sender<Vec<u8>>> = HashMap::new();
        while let Some(message) = stream.next().await {
            match message? {
                Message::Text(text) => {
                    let control: serde_json::Value = serde_json::from_str(&text)?;
                    match control["type"].as_str() {
                        Some("open") => {
                            let conn = u32::try_from(control["conn"].as_u64().unwrap_or(u64::MAX))?;
                            let nameplate = control["nameplate"].as_str().map(str::to_string);
                            // Connections whose task ended are gone.
                            conns.retain(|_, tx| !tx.is_closed());
                            if conns.len() >= MAX_CONNS && !conns.contains_key(&conn) {
                                warn!("E-ink bridge connection refused: too many at once");
                                let _ = out_tx.try_send(Out::Text(close_json(conn)));
                                continue;
                            }
                            let (tx, rx) = mpsc::channel(CONN_QUEUE);
                            conns.insert(conn, tx);
                            let me = self.clone();
                            let out = out_tx.clone();
                            tokio::spawn(
                                async move { me.connection(conn, nameplate, rx, out).await },
                            );
                        }
                        Some("close") => {
                            let conn = control["conn"].as_u64().unwrap_or(u64::MAX);
                            if let Ok(conn) = u32::try_from(conn) {
                                conns.remove(&conn);
                            }
                        }
                        Some("join_failed") => {
                            let nameplate = control["nameplate"].as_str().unwrap_or_default();
                            self.fail_join(
                                nameplate,
                                "No Kindle is waiting with that code; check it and try again",
                            );
                        }
                        _ => anyhow::bail!("unexpected control message from the bridge"),
                    }
                }
                Message::Binary(bytes) => {
                    if bytes.len() < 4 {
                        anyhow::bail!("short frame from the bridge");
                    }
                    let conn = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    if let Some(tx) = conns.get(&conn)
                        && tx.try_send(bytes[4..].to_vec()).is_err()
                    {
                        // Full or gone: close it rather than buffer.
                        conns.remove(&conn);
                        let _ = out_tx.try_send(Out::Text(close_json(conn)));
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
        writer.abort();
        Ok(())
    }

    /// One browser connection: the handshake, then its requests.
    async fn connection(
        self: Arc<Self>,
        conn: u32,
        nameplate: Option<String>,
        mut rx: mpsc::Receiver<Vec<u8>>,
        out: mpsc::Sender<Out>,
    ) {
        let result = match nameplate {
            None => self.session_handshake(conn, &mut rx, &out).await,
            Some(nameplate) => {
                self.pairing_handshake(conn, &nameplate, &mut rx, &out)
                    .await
            }
        };
        let mut channel = match result {
            Ok(channel) => channel,
            Err(error) => {
                info!(%error, "An e-ink handshake was refused");
                let _ = out.send(Out::Frame(conn, frame::abort())).await;
                let _ = out.send(Out::Text(close_json(conn))).await;
                return;
            }
        };
        let peer = BridgePeer(channel.device_id);
        // A revocation (from any device) closes this connection at once
        // when it names this device.
        let mut revoked = self.gateway.pairing().revocations_seen();
        loop {
            let frame = tokio::select! {
                frame = tokio::time::timeout(IDLE_TIMEOUT, rx.recv()) => match frame {
                    Ok(Some(frame)) => frame,
                    _ => break,
                },
                changed = revoked.changed() => {
                    if changed.is_err() || !self.still_paired(&channel.device_id).await {
                        break;
                    }
                    continue;
                }
            };
            let message = match channel.opener.open(&frame) {
                Ok(Some(message)) => message,
                Ok(None) => continue,
                Err(_) => break,
            };
            let Ok(request) = message::Request::decode(&message) else {
                break;
            };
            let response = self.serve(&peer, request).await;
            let Ok(encoded) = response.encode() else {
                break;
            };
            let Ok(frames) = channel.sealer.seal_message(&encoded) else {
                break;
            };
            for frame in frames {
                if out.send(Out::Frame(conn, frame)).await.is_err() {
                    return;
                }
            }
        }
        let _ = out.send(Out::Text(close_json(conn))).await;
    }

    /// Whether the device with `device_id` is still paired (not revoked).
    async fn still_paired(&self, device_id: &[u8; 32]) -> bool {
        self.gateway
            .pairing()
            .active_bridge_keys()
            .await
            .iter()
            .any(|key| ids::device_id(key) == *device_id)
    }

    async fn session_handshake(
        &self,
        conn: u32,
        rx: &mut mpsc::Receiver<Vec<u8>>,
        out: &mpsc::Sender<Out>,
    ) -> anyhow::Result<DaemonChannel> {
        let deadline = tokio::time::Instant::now() + HANDSHAKE_TIMEOUT;
        let hello = recv_until(rx, deadline).await?;
        let Hello::Session(hello) = handshake::parse_hello(&hello)? else {
            anyhow::bail!("a pairing hello on a connection that did not pair");
        };
        let keys = self.gateway.pairing().active_bridge_keys().await;
        let device_pub = keys
            .into_iter()
            .find(|key| ids::device_id(key) == *hello.device_id());
        let (session, reply) = hello.respond(&self.keys.identity, device_pub.as_deref())?;
        out.send(Out::Frame(conn, reply)).await?;
        let finish = recv_until(rx, deadline).await?;
        let mut channel = session.finish(&finish)?;
        out.send(Out::Text(promote_json(conn))).await?;
        let ready = channel.ready_frame()?;
        out.send(Out::Frame(conn, ready)).await?;
        Ok(channel)
    }

    async fn pairing_handshake(
        &self,
        conn: u32,
        nameplate: &str,
        rx: &mut mpsc::Receiver<Vec<u8>>,
        out: &mpsc::Sender<Out>,
    ) -> anyhow::Result<DaemonChannel> {
        let Some(pending) = self
            .joins
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(nameplate)
        else {
            anyhow::bail!("no pairing was asked for on this nameplate");
        };
        let outcome = self.pair_over(conn, &pending.code, rx, out).await;
        let answer = match &outcome {
            Ok((_, name)) => Ok(name.clone()),
            Err(_) => Err(ProtocolError::new(
                ErrorCode::Invalid,
                "Pairing failed: check the code on the Kindle and start again there",
            )),
        };
        let _ = pending.done.send(answer);
        outcome.map(|(channel, _)| channel)
    }

    async fn pair_over(
        &self,
        conn: u32,
        code: &PairingCode,
        rx: &mut mpsc::Receiver<Vec<u8>>,
        out: &mpsc::Sender<Out>,
    ) -> anyhow::Result<(DaemonChannel, String)> {
        let deadline = tokio::time::Instant::now() + HANDSHAKE_TIMEOUT;
        let hello = recv_until(rx, deadline).await?;
        let Hello::Pairing(hello) = handshake::parse_hello(&hello)? else {
            anyhow::bail!("a session hello on a pairing connection");
        };
        let (pairing, reply) = hello.respond(
            &self.keys.identity,
            code.secret.as_bytes(),
            &code.device_key_hash,
        )?;
        out.send(Out::Frame(conn, reply)).await?;
        let finish = recv_until(rx, deadline).await?;
        let mut paired = pairing.finish(&finish)?;
        let name = paired.device_name.clone();
        let device_id = self
            .gateway
            .pairing()
            .add_bridge_device(&name, &paired.channel.device_pub)
            .await
            .map_err(|error| anyhow::anyhow!(error.message))?;
        out.send(Out::Text(promote_json(conn))).await?;
        let control = paired.paired_control(&self.route_id, &device_id);
        for frame in paired.channel.sealer.seal_message(&control)? {
            out.send(Out::Frame(conn, frame)).await?;
        }
        Ok((paired.channel, name))
    }

    /// One request through the bridge router, as the device the handshake
    /// proved.
    async fn serve(&self, peer: &BridgePeer, request: message::Request) -> message::Response {
        let request_id = request.request_id;
        let refused = |status: u16, text: &str| message::Response {
            request_id,
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: serde_json::json!({ "error": text })
                .to_string()
                .into_bytes(),
        };
        let mut builder = http::Request::builder()
            .method(request.method.as_str())
            .uri(request.path_and_query.as_str())
            // The page is a web client; what it may do comes from its record.
            .header("x-aidaemon-client", "web");
        for (name, value) in &request.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        let Ok(mut http_request) = builder.body(axum::body::Body::from(request.body)) else {
            return refused(400, "Malformed request");
        };
        http_request.extensions_mut().insert(peer.clone());
        let Ok(response) = self.router.clone().oneshot(http_request).await;
        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let body = match http_body_util::Limited::new(response.into_body(), MAX_RESPONSE - 1024)
            .collect()
            .await
        {
            Ok(collected) => collected.to_bytes().to_vec(),
            Err(_) => return refused(502, "That reply is too large for this device"),
        };
        message::Response {
            request_id,
            status,
            headers: content_type
                .map(|value| vec![("content-type".to_string(), value)])
                .unwrap_or_default(),
            body,
        }
    }

    fn fail_join(&self, nameplate: &str, why: &str) {
        if let Some(pending) = self
            .joins
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(nameplate)
        {
            let _ = pending
                .done
                .send(Err(ProtocolError::new(ErrorCode::NotFound, why)));
        }
    }

    fn fail_joins(&self, why: &str) {
        let pending: Vec<_> = self
            .joins
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .drain()
            .collect();
        for (_, join) in pending {
            let _ = join
                .done
                .send(Err(ProtocolError::new(ErrorCode::Conflict, why)));
        }
    }
}

#[async_trait::async_trait]
impl BridgePairer for Connector {
    async fn pair(&self, code: &str) -> Result<String, ProtocolError> {
        let code = parse_code(code).ok_or_else(|| {
            ProtocolError::new(
                ErrorCode::Invalid,
                "That code is not a Kindle pairing code: type it exactly as the Kindle shows it",
            )
        })?;
        let link = self
            .link
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or_else(|| {
                ProtocolError::new(ErrorCode::Conflict, "Not connected to the e-ink bridge yet")
            })?;
        let (done, answer) = oneshot::channel();
        let nameplate = code.nameplate.clone();
        {
            let mut joins = self.joins.lock().unwrap_or_else(|p| p.into_inner());
            if joins.contains_key(&nameplate) {
                return Err(ProtocolError::new(
                    ErrorCode::Conflict,
                    "That Kindle is already being paired",
                ));
            }
            joins.insert(nameplate.clone(), PendingJoin { code, done });
        }
        let join = serde_json::json!({ "type": "join", "nameplate": nameplate });
        if link.send(Out::Text(join.to_string())).await.is_err() {
            self.fail_join(&nameplate, "The bridge connection dropped");
        }
        match tokio::time::timeout(PAIR_TIMEOUT, answer).await {
            Ok(Ok(result)) => result,
            _ => {
                self.fail_join(&nameplate, "Timed out");
                Err(ProtocolError::new(
                    ErrorCode::Conflict,
                    "The Kindle did not finish pairing in time: start again on the Kindle",
                ))
            }
        }
    }
}

/// TLS for the bridge socket: rustls on aws-lc-rs, checked by the OS trust
/// store. The provider is named because the daemon links both rustls
/// backends, and rustls panics when it has to choose.
fn tls_config() -> anyhow::Result<Arc<rustls::ClientConfig>> {
    use rustls_platform_verifier::BuilderVerifierExt;
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .with_platform_verifier()?
        .with_no_client_auth();
    Ok(Arc::new(config))
}

fn close_json(conn: u32) -> String {
    serde_json::json!({ "type": "close", "conn": conn }).to_string()
}

fn promote_json(conn: u32) -> String {
    serde_json::json!({ "type": "promote", "conn": conn }).to_string()
}

async fn recv_until(
    rx: &mut mpsc::Receiver<Vec<u8>>,
    deadline: tokio::time::Instant,
) -> anyhow::Result<Vec<u8>> {
    let frame = tokio::time::timeout_at(deadline, rx.recv())
        .await
        .map_err(|_| anyhow::anyhow!("handshake timed out"))?
        .ok_or_else(|| anyhow::anyhow!("the browser left"))?;
    anyhow::ensure!(frame.len() <= MAX_DATA_FRAME, "frame too large");
    Ok(frame)
}

async fn next_json<S>(stream: &mut S) -> anyhow::Result<serde_json::Value>
where
    S: futures::Stream<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => return Ok(serde_json::from_str(&text)?),
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(_)) => anyhow::bail!("expected a control message"),
            Some(Err(error)) => return Err(error.into()),
            None => anyhow::bail!("the bridge closed the connection"),
        }
    }
}
