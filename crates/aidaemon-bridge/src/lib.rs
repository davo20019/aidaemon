//! `aidaemon-bridge`: the public relay of the aidaemon bridge protocol
//! (`aidaemon-bridge-proto`'s `SPEC.md`, §3 transport, §7 limits).
//!
//! A daemon logs in on `/v1/daemon` with its login key and carries every
//! browser routed to it over that one WebSocket; browsers join it on
//! `/v1/connect?route=…` or, while pairing, wait on `/v1/wait?nameplate=…`
//! until the daemon joins. The bridge forwards binary frames it cannot
//! read (the page and the daemon encrypt end to end) and enforces size,
//! count, time, rate and byte limits so one party cannot starve the rest.
//!
//! Logging rule: the bridge logs events and refusals only, never frame
//! contents, frame sizes, nameplates, route ids or client addresses.

pub mod config;

mod browser;
mod daemon;
mod limits;
mod wire;

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

pub use config::Config;

use limits::{Budget, Kind, RateLimiter};
use wire::Nameplate;

/// `H("aidaemon-bridge-route-v1" ‖ login_pub)`, the daemon's address.
pub(crate) type RouteId = [u8; 32];

/// WebSocket close codes the bridge uses.
pub(crate) mod code {
    pub const NORMAL: u16 = 1000;
    pub const GOING_AWAY: u16 = 1001;
    pub const UNSUPPORTED: u16 = 1003;
    pub const POLICY: u16 = 1008;
    pub const TOO_BIG: u16 = 1009;
    pub const TRY_AGAIN: u16 = 1013;
}

/// The relay: who is logged in, who is waiting, and the limits' counters.
pub struct Bridge {
    config: Config,
    state: Mutex<BridgeState>,
}

#[derive(Default)]
struct BridgeState {
    daemons: HashMap<RouteId, DaemonEntry>,
    waiting: HashMap<Nameplate, Waiter>,
    budgets: HashMap<RouteId, Arc<Mutex<Budget>>>,
    rates: RateLimiter,
    next_id: u64,
}

struct DaemonEntry {
    session: u64,
    admit: mpsc::Sender<daemon::Admit>,
    /// Dropped when another login replaces this one: the session ends.
    _shutdown: oneshot::Sender<()>,
}

struct Waiter {
    id: u64,
    link: oneshot::Sender<browser::Link>,
}

/// What a daemon session gets when its login is accepted.
pub(crate) struct Registration {
    pub session: u64,
    pub admit: mpsc::Receiver<daemon::Admit>,
    pub shutdown: oneshot::Receiver<()>,
    pub budget: Arc<Mutex<Budget>>,
}

/// Admissions waiting for one daemon session to take them.
const ADMIT_QUEUE: usize = 16;

impl Bridge {
    pub fn new(config: Config) -> Arc<Self> {
        Arc::new(Self {
            config,
            state: Mutex::default(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The bridge's routes. Serve it with
    /// `into_make_service_with_connect_info::<SocketAddr>()` (see [`Bridge::serve`]).
    pub fn router(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .route("/v1/daemon", get(daemon_route))
            .route("/v1/connect", get(connect_route))
            .route("/v1/wait", get(wait_route))
            .with_state(self.clone())
    }

    /// Serve the bridge on `listener` until the task is dropped.
    pub async fn serve(self: Arc<Self>, listener: TcpListener) -> std::io::Result<()> {
        axum::serve(
            listener,
            self.router()
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
    }

    fn state(&self) -> MutexGuard<'_, BridgeState> {
        // A panic while holding the lock leaves plain maps behind; keep going.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The client's address: the trusted proxy's header when configured
    /// (its last entry, the address the proxy itself saw), else the socket.
    fn client_ip(&self, peer: SocketAddr, headers: &HeaderMap) -> IpAddr {
        if let Some(name) = &self.config.trusted_proxy_header
            && let Some(ip) = headers
                .get_all(name.as_str())
                .iter()
                .next_back()
                .and_then(|value| value.to_str().ok())
                // Machine format: a comma-separated address list.
                .and_then(|value| value.rsplit(',').next())
                .and_then(|value| value.trim().parse().ok())
        {
            return ip;
        }
        peer.ip()
    }

    fn allow(&self, kind: Kind, ip: IpAddr) -> bool {
        let limit = match kind {
            Kind::Connect => self.config.connects_per_ip_per_minute,
            Kind::Login => self.config.logins_per_ip_per_minute,
            Kind::Wait => self.config.waits_per_ip_per_minute,
            Kind::Join => self.config.joins_per_ip_per_minute,
        };
        self.state().rates.allow(kind, ip, limit, Instant::now())
    }

    /// Log a daemon in under `route`, replacing an earlier login with the
    /// same route. `None` when the bridge is full.
    pub(crate) fn register_daemon(&self, route: RouteId) -> Option<Registration> {
        let now = Instant::now();
        let mut state = self.state();
        if !state.daemons.contains_key(&route) && state.daemons.len() >= self.config.max_daemons {
            return None;
        }
        let budgets_max = self.config.max_daemons.saturating_mul(2);
        if !state.budgets.contains_key(&route) && state.budgets.len() >= budgets_max {
            state.budgets.retain(|_, budget| {
                Arc::strong_count(budget) > 1
                    || !budget
                        .lock()
                        .map(|budget| budget.expired(now))
                        .unwrap_or(true)
            });
            // Still full of today's routes (many fresh keys): a new one
            // waits, so the map, and each login's scan of it, stay bounded.
            if state.budgets.len() >= budgets_max {
                return None;
            }
        }
        let budget = state
            .budgets
            .entry(route)
            .or_insert_with(|| Arc::new(Mutex::new(Budget::new(now))))
            .clone();
        state.next_id += 1;
        let session = state.next_id;
        let (admit_tx, admit) = mpsc::channel(ADMIT_QUEUE);
        let (shutdown_tx, shutdown) = oneshot::channel();
        let replaced = state.daemons.insert(
            route,
            DaemonEntry {
                session,
                admit: admit_tx,
                _shutdown: shutdown_tx,
            },
        );
        drop(state);
        if replaced.is_some() {
            info!("daemon login replaced an earlier one");
        }
        Some(Registration {
            session,
            admit,
            shutdown,
            budget,
        })
    }

    pub(crate) fn unregister_daemon(&self, route: &RouteId, session: u64) {
        let mut state = self.state();
        if state
            .daemons
            .get(route)
            .is_some_and(|entry| entry.session == session)
        {
            state.daemons.remove(route);
        }
    }

    fn daemon_admit(&self, route: &RouteId) -> Option<mpsc::Sender<daemon::Admit>> {
        self.state()
            .daemons
            .get(route)
            .map(|entry| entry.admit.clone())
    }

    fn start_wait(
        self: &Arc<Self>,
        nameplate: Nameplate,
    ) -> Result<(WaitGuard, oneshot::Receiver<browser::Link>), StatusCode> {
        let mut state = self.state();
        if let Some(waiter) = state.waiting.get(&nameplate) {
            if !waiter.link.is_closed() {
                return Err(StatusCode::CONFLICT);
            }
            state.waiting.remove(&nameplate);
        }
        if state.waiting.len() >= self.config.max_waiting {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        state.next_id += 1;
        let id = state.next_id;
        let (link, rx) = oneshot::channel();
        state.waiting.insert(nameplate, Waiter { id, link });
        Ok((
            WaitGuard {
                bridge: self.clone(),
                nameplate,
                id,
            },
            rx,
        ))
    }

    pub(crate) fn take_waiter(
        &self,
        nameplate: &Nameplate,
    ) -> Option<oneshot::Sender<browser::Link>> {
        self.state()
            .waiting
            .remove(nameplate)
            .map(|waiter| waiter.link)
    }
}

/// Removes its nameplate from the waiting list when the wait ends, unless a
/// join already took it (and a new wait may hold it now).
pub(crate) struct WaitGuard {
    bridge: Arc<Bridge>,
    nameplate: Nameplate,
    id: u64,
}

impl Drop for WaitGuard {
    fn drop(&mut self) {
        let mut state = self.bridge.state();
        if state
            .waiting
            .get(&self.nameplate)
            .is_some_and(|waiter| waiter.id == self.id)
        {
            state.waiting.remove(&self.nameplate);
        }
    }
}

/// The value of the only query parameter `name`, or `None` for anything else.
fn only_param<'a>(query: &'a Option<String>, name: &str) -> Option<&'a str> {
    // Machine format: a URL query string; exactly `name=value` is accepted.
    query.as_deref()?.strip_prefix(name)?.strip_prefix('=')
}

async fn daemon_route(
    State(bridge): State<Arc<Bridge>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let ip = bridge.client_ip(peer, &headers);
    if !bridge.allow(Kind::Login, ip) {
        info!("daemon login refused: rate limit");
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    }
    let max = bridge.config.daemon_frame_max();
    ws.max_message_size(max)
        .max_frame_size(max)
        .on_upgrade(move |socket| daemon::run(bridge, socket, ip))
}

async fn connect_route(
    State(bridge): State<Arc<Bridge>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    RawQuery(query): RawQuery,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(route) = only_param(&query, "route").and_then(wire::lower_hex::<32>) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if !bridge.allow(Kind::Connect, bridge.client_ip(peer, &headers)) {
        info!("connection refused: rate limit");
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    }
    if bridge.daemon_admit(&route).is_none() {
        info!("connection refused: no daemon");
        return StatusCode::NOT_FOUND.into_response();
    }
    let max = bridge.config.data_frame_max;
    ws.max_message_size(max)
        .max_frame_size(max)
        .on_upgrade(move |socket| browser::connect(bridge, socket, route))
}

async fn wait_route(
    State(bridge): State<Arc<Bridge>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    RawQuery(query): RawQuery,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(nameplate) = only_param(&query, "nameplate").and_then(wire::nameplate) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if !bridge.allow(Kind::Wait, bridge.client_ip(peer, &headers)) {
        info!("pairing wait refused: rate limit");
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    }
    let (guard, link) = match bridge.start_wait(nameplate) {
        Ok(wait) => wait,
        Err(status) => {
            info!("pairing wait refused: nameplate taken or bridge full");
            return status.into_response();
        }
    };
    let max = bridge.config.data_frame_max;
    ws.max_message_size(max)
        .max_frame_size(max)
        .on_upgrade(move |socket| browser::wait(bridge, socket, guard, link))
}

/// Send one message, giving up (false) after `limit`: a peer that does not
/// take its frames is closed rather than buffered for.
pub(crate) async fn send(socket: &mut WebSocket, message: Message, limit: Duration) -> bool {
    matches!(
        tokio::time::timeout(limit, socket.send(message)).await,
        Ok(Ok(()))
    )
}

pub(crate) async fn send_text(socket: &mut WebSocket, text: String, limit: Duration) -> bool {
    send(socket, Message::Text(text.into()), limit).await
}

/// Send a close frame (best effort); the caller then drops the socket.
pub(crate) async fn close(socket: &mut WebSocket, code: u16, limit: Duration) {
    let frame = CloseFrame {
        code,
        reason: Utf8Bytes::from_static(""),
    };
    send(socket, Message::Close(Some(frame)), limit).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", value.parse().unwrap());
        headers
    }

    #[test]
    fn client_ip_believes_the_header_only_when_trusted() {
        let peer: SocketAddr = "10.0.0.1:5000".parse().unwrap();
        let untrusted = Bridge::new(Config::default());
        assert_eq!(untrusted.client_ip(peer, &headers("192.0.2.7")), peer.ip());
        let trusted = Bridge::new(Config {
            trusted_proxy_header: Some("X-Forwarded-For".into()),
            ..Config::default()
        });
        // The last entry is the one the trusted proxy appended.
        assert_eq!(
            trusted.client_ip(peer, &headers("203.0.113.9, 192.0.2.7")),
            "192.0.2.7".parse::<IpAddr>().unwrap()
        );
        assert_eq!(trusted.client_ip(peer, &headers("junk")), peer.ip());
        assert_eq!(trusted.client_ip(peer, &HeaderMap::new()), peer.ip());
    }
}
