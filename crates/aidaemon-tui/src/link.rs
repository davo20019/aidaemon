//! The connection to the gateway: connects, says hello with the last `seq`
//! it delivered, keeps the socket alive with pings, and on any failure
//! reconnects and resumes. Commands not yet answered are sent again with
//! their own `request_id` after a reconnect, so the gateway answers each
//! exactly once.

use std::collections::BTreeMap;
use std::time::Duration;

use aidaemon_client_protocol::{
    Capability, ClientFrame, ClientInfo, ClientKind, Command, ProtocolVersion, ServerFrame,
};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

const PING_EVERY: Duration = Duration::from_secs(25);
/// Two missed pongs.
const SILENT_LIMIT: Duration = Duration::from_secs(60);
const RETRY_MIN: Duration = Duration::from_millis(500);
const RETRY_MAX: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub struct LinkConfig {
    /// `ws://127.0.0.1:8788/client/v1/ws`
    pub url: String,
    pub token: String,
    pub device: Option<String>,
    pub capabilities: Vec<Capability>,
}

/// What the link tells the app.
#[allow(clippy::large_enum_variant)] // Frames dominate; boxing buys nothing here.
#[derive(Debug, Clone, PartialEq)]
pub enum Update {
    Connected,
    Disconnected(String),
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
    };
    let mut wait = RETRY_MIN;
    loop {
        let reason = match link.session().await {
            Ok(Ended::AppGone) => return,
            Ok(Ended::Dropped(reason)) => {
                wait = RETRY_MIN;
                reason
            }
            Err(error) => error.to_string(),
        };
        if link.updates.send(Update::Disconnected(reason)).is_err() {
            return;
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(RETRY_MAX);
    }
}

enum Ended {
    AppGone,
    Dropped(String),
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
            _ => {}
        }
        self.updates.send(Update::Frame(frame)).is_ok()
    }

    async fn session(&mut self) -> anyhow::Result<Ended> {
        let mut request = self.config.url.as_str().into_client_request()?;
        request.headers_mut().insert(
            "authorization",
            format!("Bearer {}", self.config.token).parse()?,
        );
        let (mut socket, _) = tokio_tungstenite::connect_async(request).await?;
        let hello = ClientFrame::Hello {
            version: ProtocolVersion::CURRENT,
            client: ClientInfo {
                kind: ClientKind::Tui,
                name: "aidaemon-tui".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
                device: self.config.device.clone(),
            },
            capabilities: self.config.capabilities.clone(),
            resume_from: self.cursor,
            token: None,
        };
        socket
            .send(Message::text(serde_json::to_string(&hello)?))
            .await?;
        if self.updates.send(Update::Connected).is_err() {
            return Ok(Ended::AppGone);
        }
        for (request_id, command) in self.pending.clone() {
            let frame = ClientFrame::Command {
                request_id,
                command,
            };
            socket
                .send(Message::text(serde_json::to_string(&frame)?))
                .await?;
        }
        let mut ping = tokio::time::interval(PING_EVERY);
        ping.tick().await;
        let mut last_heard = tokio::time::Instant::now();
        loop {
            tokio::select! {
                incoming = socket.next() => {
                    let message = match incoming {
                        Some(Ok(message)) => message,
                        Some(Err(error)) => return Ok(Ended::Dropped(error.to_string())),
                        None => return Ok(Ended::Dropped("the gateway closed the connection".into())),
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
                            return Ok(Ended::Dropped("the gateway closed the connection".into()))
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
                    socket.send(Message::text(serde_json::to_string(&frame)?)).await?;
                }
                _ = ping.tick() => {
                    if last_heard.elapsed() > SILENT_LIMIT {
                        return Ok(Ended::Dropped("no answer from the gateway".into()));
                    }
                    socket.send(Message::text(serde_json::to_string(&ClientFrame::Ping)?)).await?;
                }
            }
        }
    }
}
