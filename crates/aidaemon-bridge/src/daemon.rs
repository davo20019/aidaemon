//! A daemon's session (SPEC §3): challenge and login, then one task that
//! owns this daemon's browser connections and multiplexes them over its
//! WebSocket (`u32 conn ‖ frame` binary, JSON control text).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aidaemon_bridge_proto::ids::route_id;
use aidaemon_bridge_proto::login::verify_login;
use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{MissedTickBehavior, interval_at, timeout};
use tracing::{debug, info};

use crate::browser::{GoneGuard, Link, ToBrowser};
use crate::limits::Budget;
use crate::wire::{self, DaemonMessage, Nameplate};
use crate::{Bridge, close, code, send, send_text};

/// A browser asking to join this daemon; `None` in the reply refuses it.
pub(crate) struct Admit {
    pub reply: oneshot::Sender<Option<Link>>,
}

pub(crate) async fn run(bridge: Arc<Bridge>, mut socket: WebSocket) {
    let config = bridge.config.clone();
    let mut nonce = [0u8; 32];
    if getrandom::fill(&mut nonce).is_err() {
        close(&mut socket, code::TRY_AGAIN, config.send_timeout).await;
        return;
    }
    if !send_text(&mut socket, wire::challenge(&nonce), config.send_timeout).await {
        return;
    }
    let login = match timeout(config.login_timeout, socket.recv()).await {
        Ok(Some(Ok(Message::Text(text)))) => wire::parse_login(&text),
        _ => None,
    };
    let Some((login_pub, sig)) = login else {
        info!("daemon login refused: malformed or late");
        close(&mut socket, code::POLICY, config.send_timeout).await;
        return;
    };
    if !verify_login(&login_pub, &nonce, &sig) {
        info!("daemon login refused: bad key or signature");
        close(&mut socket, code::POLICY, config.send_timeout).await;
        return;
    }
    let route = route_id(&login_pub);
    let Some(registration) = bridge.register_daemon(route) else {
        info!("daemon login refused: bridge full");
        close(&mut socket, code::TRY_AGAIN, config.send_timeout).await;
        return;
    };
    let session_id = registration.session;
    if send_text(&mut socket, wire::ready(&route), config.send_timeout).await {
        info!("daemon logged in");
        Session::new(bridge.clone(), registration.budget)
            .run(socket, registration.admit, registration.shutdown)
            .await;
        info!("daemon disconnected");
    }
    bridge.unregister_daemon(&route, session_id);
}

struct Conn {
    tx: mpsc::Sender<ToBrowser>,
    handshaking: bool,
}

struct Session {
    bridge: Arc<Bridge>,
    budget: Arc<Mutex<Budget>>,
    conns: HashMap<u32, Conn>,
    handshaking: usize,
    next_conn: u32,
    frames_tx: mpsc::Sender<(u32, Bytes)>,
    frames_rx: mpsc::Receiver<(u32, Bytes)>,
    gone_tx: mpsc::UnboundedSender<u32>,
    gone_rx: mpsc::UnboundedReceiver<u32>,
}

/// Keep going, or end the session (having sent the daemon a close frame
/// when that is the reason).
type Flow = bool;

impl Session {
    fn new(bridge: Arc<Bridge>, budget: Arc<Mutex<Budget>>) -> Self {
        let (frames_tx, frames_rx) = mpsc::channel(bridge.config.daemon_queue);
        // One message per admitted browser, so bounded by the connection cap.
        let (gone_tx, gone_rx) = mpsc::unbounded_channel();
        Self {
            bridge,
            budget,
            conns: HashMap::new(),
            handshaking: 0,
            next_conn: 0,
            frames_tx,
            frames_rx,
            gone_tx,
            gone_rx,
        }
    }

    async fn run(
        mut self,
        mut socket: WebSocket,
        mut admit: mpsc::Receiver<Admit>,
        mut shutdown: oneshot::Receiver<()>,
    ) {
        let config = self.bridge.config.clone();
        let mut keepalive = interval_at(
            tokio::time::Instant::now() + config.keepalive,
            config.keepalive,
        );
        keepalive.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut last_heard = Instant::now();
        loop {
            let flow = tokio::select! {
                _ = &mut shutdown => {
                    info!("daemon session closed: replaced by a new login");
                    close(&mut socket, code::GOING_AWAY, config.send_timeout).await;
                    false
                }
                message = socket.recv() => {
                    last_heard = Instant::now();
                    match message {
                        Some(Ok(Message::Text(text))) => self.on_text(&mut socket, &text).await,
                        Some(Ok(Message::Binary(frame))) => self.on_binary(&mut socket, frame).await,
                        Some(Ok(Message::Ping(_) | Message::Pong(_))) => true,
                        Some(Ok(Message::Close(_))) | None => false,
                        Some(Err(_)) => {
                            info!("daemon session closed: transport error or oversize frame");
                            false
                        }
                    }
                }
                Some((conn, frame)) = self.frames_rx.recv() => {
                    self.forward_to_daemon(&mut socket, conn, frame).await
                }
                Some(conn) = self.gone_rx.recv() => self.browser_gone(&mut socket, conn).await,
                Some(request) = admit.recv() => self.admit_connect(&mut socket, request).await,
                _ = keepalive.tick() => {
                    if last_heard.elapsed() > config.keepalive * 3 {
                        info!("daemon session closed: silent");
                        false
                    } else {
                        send(&mut socket, Message::Ping(Bytes::new()), config.send_timeout).await
                    }
                }
            };
            if !flow {
                break;
            }
        }
        // Dropping the connections' senders closes every browser.
    }

    async fn refuse(&self, socket: &mut WebSocket, why: &'static str) -> Flow {
        info!(reason = why, "daemon session closed: protocol violation");
        close(socket, code::POLICY, self.bridge.config.send_timeout).await;
        false
    }

    async fn on_text(&mut self, socket: &mut WebSocket, text: &str) -> Flow {
        let Some(message) = wire::parse_daemon(text) else {
            return self.refuse(socket, "unknown control message").await;
        };
        match message {
            DaemonMessage::Promote { conn } => {
                // A connection already gone (a race with its close) is ignored.
                let Some(entry) = self.conns.get_mut(&conn) else {
                    return true;
                };
                if !entry.handshaking {
                    return true;
                }
                entry.handshaking = false;
                self.handshaking -= 1;
                if entry.tx.try_send(ToBrowser::Promote).is_err() {
                    info!("connection closed: browser too slow");
                    return self.drop_conn(socket, conn).await;
                }
                true
            }
            DaemonMessage::Close { conn } => {
                // Frames already queued still reach the browser, then it closes.
                self.forget(conn);
                true
            }
            DaemonMessage::Join { nameplate } => match wire::nameplate(&nameplate) {
                Some(nameplate) => self.join(socket, nameplate).await,
                None => self.refuse(socket, "malformed nameplate").await,
            },
        }
    }

    async fn on_binary(&mut self, socket: &mut WebSocket, frame: Bytes) -> Flow {
        if frame.len() < 4 {
            return self
                .refuse(socket, "binary frame without a connection id")
                .await;
        }
        let conn = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]);
        let payload = frame.slice(4..);
        let Some(entry) = self.conns.get(&conn) else {
            // Racing a close the daemon has not seen yet.
            return true;
        };
        if !self.charge(payload.len()) {
            info!("connection closed: daily byte budget spent");
            return self.drop_conn(socket, conn).await;
        }
        if entry.tx.try_send(ToBrowser::Frame(payload)).is_err() {
            info!("connection closed: browser too slow");
            return self.drop_conn(socket, conn).await;
        }
        true
    }

    async fn forward_to_daemon(&mut self, socket: &mut WebSocket, conn: u32, frame: Bytes) -> Flow {
        if !self.conns.contains_key(&conn) {
            return true;
        }
        if !self.charge(frame.len()) {
            info!("connection closed: daily byte budget spent");
            return self.drop_conn(socket, conn).await;
        }
        let mut prefixed = Vec::with_capacity(4 + frame.len());
        prefixed.extend_from_slice(&conn.to_be_bytes());
        prefixed.extend_from_slice(&frame);
        let sent = send(
            socket,
            Message::Binary(prefixed.into()),
            self.bridge.config.send_timeout,
        )
        .await;
        if !sent {
            info!("daemon session closed: daemon too slow");
        }
        sent
    }

    async fn browser_gone(&mut self, socket: &mut WebSocket, conn: u32) -> Flow {
        if self.forget(conn) {
            debug!("browser connection ended");
            return send_text(socket, wire::close(conn), self.bridge.config.send_timeout).await;
        }
        true
    }

    async fn admit_connect(&mut self, socket: &mut WebSocket, request: Admit) -> Flow {
        let Some(link) = self.admit() else {
            let _ = request.reply.send(None);
            return true;
        };
        let conn = link.conn;
        if request.reply.send(Some(link)).is_err() {
            // The browser left while waiting; the daemon never heard of it.
            self.forget(conn);
            return true;
        }
        send_text(socket, wire::open(conn), self.bridge.config.send_timeout).await
    }

    async fn join(&mut self, socket: &mut WebSocket, nameplate: Nameplate) -> Flow {
        let send_timeout = self.bridge.config.send_timeout;
        let Some(link) = self.admit() else {
            info!("pairing join refused: connection limit");
            return send_text(socket, wire::join_failed(&nameplate), send_timeout).await;
        };
        let conn = link.conn;
        let Some(waiter) = self.bridge.take_waiter(&nameplate) else {
            self.forget(conn);
            info!("pairing join failed: no browser waiting");
            return send_text(socket, wire::join_failed(&nameplate), send_timeout).await;
        };
        if waiter.send(link).is_err() {
            self.forget(conn);
            info!("pairing join failed: browser left");
            return send_text(socket, wire::join_failed(&nameplate), send_timeout).await;
        }
        info!("pairing browser joined");
        send_text(socket, wire::open(conn), send_timeout).await
    }

    /// Take a new browser connection if this daemon's caps and budget allow.
    fn admit(&mut self) -> Option<Link> {
        let config = &self.bridge.config;
        let exhausted = self
            .budget
            .lock()
            .map(|mut budget| budget.exhausted(config.daily_bytes_per_route, Instant::now()))
            .unwrap_or(true);
        if exhausted
            || self.conns.len() >= config.max_conns_per_daemon
            || self.handshaking >= config.max_handshaking_per_daemon
        {
            return None;
        }
        let conn = loop {
            self.next_conn = self.next_conn.wrapping_add(1);
            if !self.conns.contains_key(&self.next_conn) {
                break self.next_conn;
            }
        };
        let (tx, rx) = mpsc::channel(config.browser_queue);
        self.conns.insert(
            conn,
            Conn {
                tx,
                handshaking: true,
            },
        );
        self.handshaking += 1;
        Some(Link {
            conn,
            rx,
            frames: self.frames_tx.clone(),
            _gone: GoneGuard {
                conn,
                tx: self.gone_tx.clone(),
            },
        })
    }

    /// Drop a connection without telling the daemon; true if it existed.
    fn forget(&mut self, conn: u32) -> bool {
        match self.conns.remove(&conn) {
            Some(entry) => {
                if entry.handshaking {
                    self.handshaking -= 1;
                }
                true
            }
            None => false,
        }
    }

    /// Close a connection from the bridge's side and tell the daemon.
    async fn drop_conn(&mut self, socket: &mut WebSocket, conn: u32) -> Flow {
        if self.forget(conn) {
            return send_text(socket, wire::close(conn), self.bridge.config.send_timeout).await;
        }
        true
    }

    fn charge(&self, bytes: usize) -> bool {
        let limit = self.bridge.config.daily_bytes_per_route;
        self.budget
            .lock()
            .map(|mut budget| budget.charge(bytes as u64, limit, Instant::now()))
            .unwrap_or(false)
    }
}
