//! Conversation shells through the gateway (protocol 1.42): the shell
//! commands, the per-terminal socket and the shell events.
//!
//! `open_shell` and `close_shell` run on the main socket as ordinary
//! commands, acked and scope-checked. A shell's output runs on a socket of
//! its own, `/client/v1/shells/{shell_id}/ws`, so it never waits behind
//! chat events or enters the resume log. That socket opens with a
//! single-use ticket from `shell_opened`, bound to the device and the
//! shell, sent as its first message (never in the URL).
//!
//! A shell runs anything on this computer, so only a device that may also
//! answer every approval (scope `control` and `approve`, no risk ceiling)
//! may open one. Nothing here logs what is typed or what a shell writes.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_client_protocol::shell::{MAX_CLIENT_MESSAGE, ShellClientMessage, ShellServerMessage};
use aidaemon_client_protocol::{
    self as wire, CommandResult, ErrorCode, Event, ProtocolError, ShellTransport,
};
use aidaemon_shells::{ShellChange, ShellError, ShellId, ShellInfo, ShellRegistry};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as UrlPath, State};
use axum::response::Response;
use tokio::sync::broadcast;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::pairing::Principal;
use crate::{ClientGateway, Publisher};

/// How long a ticket from `shell_opened` waits to be used.
const TICKET_TTL: Duration = Duration::from_secs(60);
/// How long the socket waits for its ticket.
const AUTH_WITHIN: Duration = Duration::from_secs(10);
/// A socket that hears nothing (not even a pong) for this long closes.
const QUIET_LIMIT: Duration = Duration::from_secs(60);
/// How often the server pings the socket.
const PING_EVERY: Duration = Duration::from_secs(20);
/// The most output one message carries when the shell writes faster than
/// the socket sends.
const MAX_OUTPUT_MESSAGE: usize = 64 * 1024;
/// The device id the owner's local token acts as.
const LOCAL_DEVICE: &str = "local";

struct Ticket {
    /// The ticket's hash; the ticket itself is never kept.
    hash: String,
    shell_id: String,
    expires: Instant,
    revoked: CancellationToken,
}

/// The gateway's side of conversation shells.
pub struct ShellDesk {
    registry: ShellRegistry,
    tickets: Mutex<Vec<Ticket>>,
}

use crate::util::sha;

use crate::util::lock;

fn refusal(error: ShellError) -> ProtocolError {
    let code = match &error {
        ShellError::NotFound | ShellError::Ended => ErrorCode::NotFound,
        ShellError::Disabled => ErrorCode::Forbidden,
        ShellError::TooMany => ErrorCode::RateLimited,
        ShellError::NoFolder => ErrorCode::Invalid,
        ShellError::Failed(_) => ErrorCode::Internal,
    };
    ProtocolError::new(code, error.to_string())
}

/// A shell as the protocol shows it.
pub fn wire_shell(info: &ShellInfo) -> wire::ShellInfo {
    wire::ShellInfo {
        id: info.id.to_string(),
        session_id: info.session_id.clone(),
        folder: info.folder.display().to_string(),
        title: info.title.clone(),
        cols: info.cols,
        rows: info.rows,
        exit_code: info.exit_code,
        started_at_ms: info.started_ms,
    }
}

impl ShellDesk {
    pub fn new(registry: ShellRegistry) -> Arc<Self> {
        Arc::new(Self {
            registry,
            tickets: Mutex::new(Vec::new()),
        })
    }

    pub fn registry(&self) -> &ShellRegistry {
        &self.registry
    }

    /// Whether clients are offered shells at all.
    pub fn enabled(&self) -> bool {
        self.registry.enabled()
    }

    pub fn list(&self) -> Vec<wire::ShellInfo> {
        self.registry.list().iter().map(wire_shell).collect()
    }

    /// `open_shell`: the conversation's shell, started in `folder` when
    /// none runs, and a ticket for its socket.
    pub fn open(
        &self,
        principal: &Principal,
        session_id: &str,
        folder: &Path,
        cols: u16,
        rows: u16,
    ) -> Result<CommandResult, ProtocolError> {
        let shell = self
            .registry
            .open(session_id, folder, cols, rows)
            .map_err(refusal)?;
        let ticket = format!("tkt_{}", uuid::Uuid::new_v4().simple());
        lock(&self.tickets).push(Ticket {
            hash: sha(&ticket),
            shell_id: shell.id.to_string(),
            expires: Instant::now() + TICKET_TTL,
            revoked: principal.revoked.clone(),
        });
        info!(
            shell = %shell.id,
            device = principal.device_id.as_deref().unwrap_or(LOCAL_DEVICE),
            "Conversation shell opened from an app"
        );
        Ok(CommandResult::ShellOpened {
            transport: ShellTransport {
                url: format!("/client/v1/shells/{}/ws", shell.id),
                ticket,
            },
            shell: wire_shell(&shell),
        })
    }

    /// `close_shell`.
    pub fn close(&self, shell_id: &str) -> Result<CommandResult, ProtocolError> {
        self.registry
            .close(&ShellId::from_string(shell_id))
            .map_err(refusal)?;
        Ok(CommandResult::Reply {
            text: "The shell was closed".into(),
        })
    }

    fn redeem(&self, shell_id: &str, ticket: &str) -> Option<CancellationToken> {
        let hash = sha(ticket);
        let now = Instant::now();
        let mut tickets = lock(&self.tickets);
        tickets.retain(|t| t.expires > now);
        let index = tickets
            .iter()
            .position(|t| t.hash == hash && t.shell_id == shell_id)?;
        let ticket = tickets.swap_remove(index);
        (!ticket.revoked.is_cancelled()).then_some(ticket.revoked)
    }

    /// Publish the registry's changes as events.
    pub(crate) fn spawn_pump(self: &Arc<Self>, publisher: Publisher) {
        let desk = Arc::downgrade(self);
        let mut changes = self.registry.subscribe();
        tokio::spawn(async move {
            loop {
                let change = changes.recv().await;
                let Some(desk) = desk.upgrade() else { return };
                match change {
                    Ok(ShellChange::Updated(info)) => publisher.publish(
                        Some(info.session_id.clone()),
                        Event::ShellUpdated {
                            shell: wire_shell(&info),
                        },
                    ),
                    Ok(ShellChange::Closed { id, session_id }) => publisher.publish(
                        Some(session_id.clone()),
                        Event::ShellClosed {
                            shell_id: id.to_string(),
                            session_id,
                        },
                    ),
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        for info in desk.registry.list() {
                            publisher.publish(
                                Some(info.session_id.clone()),
                                Event::ShellUpdated {
                                    shell: wire_shell(&info),
                                },
                            );
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        });
    }
}

/// `GET /client/v1/shells/{shell_id}/ws`.
pub(crate) async fn shell_socket(
    State(gateway): State<Arc<ClientGateway>>,
    UrlPath(shell_id): UrlPath<String>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade
        .max_message_size(MAX_CLIENT_MESSAGE)
        .on_upgrade(move |socket| async move {
            let Some(desk) = gateway.shell_desk() else {
                return;
            };
            if let Err(error) = run_shell(desk, socket, shell_id).await {
                debug!(%error, "Shell socket closed");
            }
        })
}

async fn send(socket: &mut WebSocket, message: &ShellServerMessage) -> anyhow::Result<()> {
    socket
        .send(Message::Text(serde_json::to_string(message)?.into()))
        .await?;
    Ok(())
}

async fn close_with(socket: &mut WebSocket, reason: &str) -> anyhow::Result<()> {
    let _ = send(
        socket,
        &ShellServerMessage::Closed {
            reason: reason.to_string(),
        },
    )
    .await;
    let _ = socket.send(Message::Close(None)).await;
    Ok(())
}

async fn run_shell(
    desk: Arc<ShellDesk>,
    mut socket: WebSocket,
    shell_id: String,
) -> anyhow::Result<()> {
    let auth = match tokio::time::timeout(AUTH_WITHIN, socket.recv()).await {
        Ok(Some(Ok(Message::Text(text)))) => serde_json::from_str::<ShellClientMessage>(&text).ok(),
        _ => None,
    };
    let Some(ShellClientMessage::Auth { ticket }) = auth else {
        return close_with(&mut socket, "Send the ticket first").await;
    };
    let Some(revoked) = desk.redeem(&shell_id, &ticket) else {
        return close_with(&mut socket, "That ticket is not valid").await;
    };
    let id = ShellId::from_string(shell_id);
    let Some(shell) = desk.registry.get(&id) else {
        return close_with(&mut socket, "That shell has ended").await;
    };
    send(
        &mut socket,
        &ShellServerMessage::Ready {
            shell: wire_shell(&shell),
        },
    )
    .await?;
    let reason = pump(&desk, &mut socket, &id, revoked).await?;
    close_with(&mut socket, reason).await
}

/// The shell's output to the socket and the owner's typing to the shell,
/// until one side goes. Returns why it ended.
async fn pump(
    desk: &ShellDesk,
    socket: &mut WebSocket,
    id: &ShellId,
    revoked: CancellationToken,
) -> anyhow::Result<&'static str> {
    let mut changes = desk.registry.subscribe();
    let mut ping = tokio::time::interval_at(Instant::now() + PING_EVERY, PING_EVERY);
    let mut heard = Instant::now();
    let mut ended_sent = false;
    'attach: loop {
        let Ok(mut attached) = desk.registry.attach(id) else {
            return Ok("The shell was closed");
        };
        if !attached.backlog.is_empty() {
            send(
                socket,
                &ShellServerMessage::Output {
                    data: std::mem::take(&mut attached.backlog),
                },
            )
            .await?;
        }
        if !ended_sent && let Some(info) = desk.registry.get(id).filter(|i| !i.running()) {
            ended_sent = true;
            send(
                socket,
                &ShellServerMessage::Ended {
                    exit_code: info.exit_code,
                },
            )
            .await?;
        }
        loop {
            tokio::select! {
                () = revoked.cancelled() => return Ok("This device was signed out"),
                out = attached.output.recv() => match out {
                    Ok(text) => {
                        // Gather what else is waiting into one message.
                        let mut data = text.to_string();
                        while data.len() < MAX_OUTPUT_MESSAGE {
                            match attached.output.try_recv() {
                                Ok(more) => data.push_str(&more),
                                Err(_) => break,
                            }
                        }
                        send(socket, &ShellServerMessage::Output { data }).await?;
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        send(socket, &ShellServerMessage::Reset).await?;
                        continue 'attach;
                    }
                    // The shell is gone; its `Closed` change says so too.
                    Err(broadcast::error::RecvError::Closed) => {
                        return Ok("The shell was closed");
                    }
                },
                change = changes.recv() => match change {
                    Ok(ShellChange::Updated(info)) if info.id == *id && !info.running() => {
                        if !ended_sent {
                            ended_sent = true;
                            send(socket, &ShellServerMessage::Ended { exit_code: info.exit_code })
                                .await?;
                        }
                    }
                    Ok(ShellChange::Closed { id: gone, .. }) if gone == *id => {
                        return Ok("The shell was closed");
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => return Ok("The daemon stopped"),
                },
                message = socket.recv() => {
                    heard = Instant::now();
                    let text = match message {
                        Some(Ok(Message::Text(text))) => text,
                        Some(Ok(Message::Close(_))) | None | Some(Err(_)) => {
                            return Ok("The app closed the shell");
                        }
                        Some(Ok(_)) => continue,
                    };
                    match serde_json::from_str::<ShellClientMessage>(&text) {
                        Ok(ShellClientMessage::Input { data }) => {
                            if let Err(error) = desk.registry.write(id, &data) {
                                debug!(%error, "Typing into a shell failed");
                            }
                        }
                        Ok(ShellClientMessage::Resize { cols, rows }) => {
                            let _ = desk.registry.resize(id, cols, rows);
                        }
                        Ok(_) | Err(_) => {}
                    }
                }
                _ = ping.tick() => {
                    if heard.elapsed() > QUIET_LIMIT {
                        return Ok("The app stopped answering");
                    }
                    socket.send(Message::Ping(Vec::new().into())).await?;
                }
            }
        }
    }
}
