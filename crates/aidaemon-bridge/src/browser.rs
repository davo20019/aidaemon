//! A browser's side (SPEC §3): join a logged-in daemon or wait on a
//! nameplate, then relay raw binary frames under the handshake and data
//! limits (§7).

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, sleep, sleep_until, timeout};
use tracing::info;

use crate::daemon::Admit;
use crate::{Bridge, RouteId, WaitGuard, close, code, send};

/// What the daemon's session sends one browser. The sender being dropped
/// closes the browser (after the frames already queued).
pub(crate) enum ToBrowser {
    Frame(Bytes),
    Promote,
}

/// A browser's place in a daemon session.
pub(crate) struct Link {
    pub conn: u32,
    pub rx: mpsc::Receiver<ToBrowser>,
    pub frames: mpsc::Sender<(u32, Bytes)>,
    pub _gone: GoneGuard,
}

/// Tells the daemon session the browser is gone, however its task ends.
pub(crate) struct GoneGuard {
    pub conn: u32,
    pub tx: mpsc::UnboundedSender<u32>,
}

impl Drop for GoneGuard {
    fn drop(&mut self) {
        let _ = self.tx.send(self.conn);
    }
}

/// `/v1/connect`: ask the daemon logged in under `route` to take us.
pub(crate) async fn connect(bridge: Arc<Bridge>, mut socket: WebSocket, route: RouteId) {
    let send_timeout = bridge.config.send_timeout;
    let Some(admit) = bridge.daemon_admit(&route) else {
        info!("connection refused: no daemon");
        close(&mut socket, code::TRY_AGAIN, send_timeout).await;
        return;
    };
    let (reply, answer) = oneshot::channel();
    if admit.try_send(Admit { reply }).is_err() {
        info!("connection refused: daemon busy");
        close(&mut socket, code::TRY_AGAIN, send_timeout).await;
        return;
    }
    drop(admit);
    match timeout(send_timeout, answer).await {
        Ok(Ok(Some(link))) => {
            info!("browser connected");
            relay(&bridge, socket, link).await;
        }
        _ => {
            info!("connection refused: daemon connection limit");
            close(&mut socket, code::TRY_AGAIN, send_timeout).await;
        }
    }
}

/// `/v1/wait`: hold the nameplate until a daemon joins it or time runs out.
pub(crate) async fn wait(
    bridge: Arc<Bridge>,
    mut socket: WebSocket,
    guard: WaitGuard,
    mut link: oneshot::Receiver<Link>,
) {
    let config = bridge.config.clone();
    let deadline = sleep(config.wait_timeout);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            joined = &mut link => {
                drop(guard);
                if let Ok(link) = joined {
                    relay(&bridge, socket, link).await;
                } else {
                    close(&mut socket, code::TRY_AGAIN, config.send_timeout).await;
                }
                return;
            }
            message = socket.recv() => match message {
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                Some(Ok(Message::Text(_) | Message::Binary(_))) => {
                    info!("pairing wait closed: browser sent before a daemon joined");
                    close(&mut socket, code::POLICY, config.send_timeout).await;
                    return;
                }
                _ => return,
            },
            _ = &mut deadline => {
                info!("pairing wait closed: no daemon joined in time");
                close(&mut socket, code::NORMAL, config.send_timeout).await;
                return;
            }
        }
    }
}

/// Forward frames between the browser and its daemon session until either
/// side closes or a limit is hit.
async fn relay(bridge: &Bridge, mut socket: WebSocket, mut link: Link) {
    let config = &bridge.config;
    let send_timeout = config.send_timeout;
    let mut limit = config.handshake_frame_max;
    let mut promoted = false;
    let handshake = sleep(config.handshake_timeout);
    tokio::pin!(handshake);
    let idle_deadline = |now: Instant| now + config.idle_timeout;
    let mut idle_at = idle_deadline(Instant::now());
    loop {
        tokio::select! {
            message = socket.recv() => match message {
                Some(Ok(Message::Binary(frame))) => {
                    if frame.len() > limit {
                        info!("browser closed: frame over the limit");
                        close(&mut socket, code::TOO_BIG, send_timeout).await;
                        return;
                    }
                    idle_at = idle_deadline(Instant::now());
                    if !hand_off(&link, frame, send_timeout).await {
                        info!("browser closed: daemon too slow");
                        close(&mut socket, code::TRY_AGAIN, send_timeout).await;
                        return;
                    }
                }
                Some(Ok(Message::Text(_))) => {
                    info!("browser closed: text frame");
                    close(&mut socket, code::UNSUPPORTED, send_timeout).await;
                    return;
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) | None => return,
                Some(Err(_)) => {
                    info!("browser closed: transport error or oversize frame");
                    return;
                }
            },
            out = link.rx.recv() => match out {
                Some(ToBrowser::Frame(frame)) => {
                    idle_at = idle_deadline(Instant::now());
                    if !send(&mut socket, Message::Binary(frame), send_timeout).await {
                        info!("browser closed: too slow");
                        return;
                    }
                }
                Some(ToBrowser::Promote) => {
                    promoted = true;
                    limit = config.data_frame_max;
                }
                None => {
                    close(&mut socket, code::NORMAL, send_timeout).await;
                    return;
                }
            },
            _ = &mut handshake, if !promoted => {
                info!("browser closed: handshake not finished in time");
                close(&mut socket, code::POLICY, send_timeout).await;
                return;
            }
            _ = sleep_until(idle_at) => {
                info!("browser closed: idle");
                close(&mut socket, code::NORMAL, send_timeout).await;
                return;
            }
        }
    }
}

async fn hand_off(link: &Link, frame: Bytes, limit: Duration) -> bool {
    matches!(
        timeout(limit, link.frames.send((link.conn, frame))).await,
        Ok(Ok(()))
    )
}
