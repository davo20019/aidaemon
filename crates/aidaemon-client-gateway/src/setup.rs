//! The gateway while the daemon is not set up yet: the same address and
//! protocol, with no agent behind it. A client signs in (the owner's local
//! token, or a device paired with the code printed at start), reads where
//! setup stands and takes its steps; everything else waits for setup to
//! finish. The app's web build is served, so a headless install is set up
//! from a browser on another device.
//!
//! Pairing here needs no chat to confirm in: the code was printed on the
//! daemon's own console, so whoever redeems it first, before any owner
//! device exists, becomes the owner device. Once one exists (a paired
//! device, or a client signed in with the local token) the code is dead.

// Handlers return a refusal as the response itself (`Result<_, Response>`).
#![allow(clippy::result_large_err)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use aidaemon_client_protocol::{
    ChallengeRequest, ClientFrame, CommandResult, ErrorCode, Event, OwnerInfo, PairingOffer,
    ProtocolError, ProtocolVersion, RedeemRequest, ServerFrame, SessionRequest, SetupAction,
    SetupState,
};
use aidaemon_store_sqlite::SqliteClientEventLog;
use async_trait::async_trait;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::pairing::{Pairing, Principal};
use crate::server::{
    MAX_BODY, app_root, bearer, body, error_response, refusal, refuse, reply, send, serve_app_file,
    web_page_for,
};
use crate::{ClientGatewayConfig, PublicUrl};

const HELLO_WITHIN: Duration = Duration::from_secs(10);
const IDLE_LIMIT: Duration = Duration::from_secs(60);

/// What only the daemon knows about setting itself up.
#[async_trait]
pub trait SetupHost: Send + Sync {
    /// Where setup stands, for a client on this machine (`local`) or on
    /// another.
    async fn state(&self, local: bool) -> SetupState;
    /// Take one step. `Err`: nothing changed, and why, in plain words.
    async fn step(&self, action: SetupAction, local: bool) -> Result<SetupState, String>;
}

/// What the setup gateway runs on.
pub struct SetupDeps {
    pub host: Arc<dyn SetupHost>,
    /// Where setup stands as it moves on by itself (a sign-in finishing in
    /// the browser, a test call returning): every client is told.
    pub updates: watch::Receiver<SetupState>,
    /// The client log on the daemon's database: devices paired here are
    /// the devices of the daemon once it is set up.
    pub log: SqliteClientEventLog,
    pub config: ClientGatewayConfig,
    /// The local token's hash.
    pub token_file: PathBuf,
    /// Cancelled when setup ends: open sockets close, so the daemon can
    /// serve the full gateway on the same address.
    pub stop: CancellationToken,
}

pub struct SetupGateway {
    host: Arc<dyn SetupHost>,
    updates: watch::Receiver<SetupState>,
    pairing: Arc<Pairing>,
    config: ClientGatewayConfig,
    token_file: PathBuf,
    /// An owner device exists: the printed code no longer pairs.
    owner_seen: AtomicBool,
    stop: CancellationToken,
}

impl SetupGateway {
    pub async fn start(deps: SetupDeps) -> anyhow::Result<Arc<Self>> {
        let pairing = Arc::new(Pairing::new(
            deps.log,
            None,
            PublicUrl {
                endpoints: aidaemon_core::EndpointSource::fixed(Vec::new()),
                fallback: deps.config.public_url(),
            },
            deps.config.watch_risk_ceiling,
            crate::token::install_fingerprint(&deps.token_file),
        ));
        let seen = pairing
            .devices()
            .await
            .map(|devices| devices.iter().any(|d| !d.revoked))
            .unwrap_or(false);
        Ok(Arc::new(Self {
            host: deps.host,
            updates: deps.updates,
            pairing,
            config: deps.config,
            token_file: deps.token_file,
            owner_seen: AtomicBool::new(seen),
            stop: deps.stop,
        }))
    }

    /// A one-time pairing offer for the console of a headless install;
    /// `None` once an owner device exists.
    pub async fn console_offer(&self) -> Option<PairingOffer> {
        if self.owner_seen.load(Ordering::SeqCst) {
            return None;
        }
        self.pairing.offer(None).await.ok()
    }

    /// The web app's address with `offer` filled in.
    pub fn web_link(&self, offer: &PairingOffer) -> String {
        format!(
            "{}{}#offer={}&code={}&fp={}",
            self.config.public_url(),
            self.config.web_app_path(),
            offer.offer_id,
            offer.code,
            offer.fingerprint
        )
    }

    fn principal(&self, token: &str) -> Option<Principal> {
        if crate::token::verify(&self.token_file, token) {
            self.owner_seen.store(true, Ordering::SeqCst);
            return Some(Principal::owner());
        }
        self.pairing.principal(token)
    }

    async fn answer(&self, request_id: String, frame: ClientFrame, local: bool) -> ServerFrame {
        let ClientFrame::Command { command, .. } = frame else {
            return ServerFrame::Error {
                request_id: None,
                error: ProtocolError::new(ErrorCode::Invalid, "Send a command frame"),
            };
        };
        let result = match command {
            aidaemon_client_protocol::Command::GetSetup => Ok(self.host.state(local).await),
            aidaemon_client_protocol::Command::Setup { action } => self
                .host
                .step(action, local)
                .await
                .map_err(|message| ProtocolError::new(ErrorCode::Invalid, message)),
            _ => Err(ProtocolError::new(
                ErrorCode::Forbidden,
                "This daemon is not set up yet: finish setup first",
            )),
        };
        match result {
            Ok(setup) => ServerFrame::Ack {
                request_id,
                result: CommandResult::Setup { setup },
            },
            Err(error) => ServerFrame::Error {
                request_id: Some(request_id),
                error,
            },
        }
    }
}

/// The setup gateway's routes, all under `/client/v1`.
pub fn router(gateway: Arc<SetupGateway>) -> Router {
    Router::new()
        .route("/client/v1/ws", get(websocket))
        .route("/client/v1/commands", post(http_command))
        .route("/client/v1/enrollments/redeem", post(redeem))
        .route("/client/v1/enrollments/{offer_id}", get(enrollment))
        .route("/client/v1/sessions/challenge", post(challenge))
        .route("/client/v1/sessions", post(open_session))
        .route("/client/v1/app", get(web_page))
        .route("/client/v1/app/", get(web_page))
        .route("/client/v1/next", get(app_root))
        .route("/client/v1/next/", get(app_index))
        .route("/client/v1/next/{*path}", get(app_file))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY))
        .with_state(gateway)
}

/// Take the configured loopback address, so a clash is known before
/// anything is printed about it.
pub async fn bind(gateway: &SetupGateway) -> anyhow::Result<tokio::net::TcpListener> {
    let addr = gateway.config.bind_addr()?;
    tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        anyhow::anyhow!("could not listen on {addr} ({e}); is another AIDaemon running?")
    })
}

/// Serve on `listener` ([`bind`]) until setup stops.
pub async fn serve(
    gateway: Arc<SetupGateway>,
    listener: tokio::net::TcpListener,
) -> anyhow::Result<()> {
    let stop = gateway.stop.clone();
    info!(addr = ?listener.local_addr().ok(), "Client Gateway listening for setup");
    axum::serve(listener, router(gateway))
        .with_graceful_shutdown(async move { stop.cancelled().await })
        .await?;
    Ok(())
}

async fn http_command(
    State(gateway): State<Arc<SetupGateway>>,
    headers: HeaderMap,
    bytes: axum::body::Bytes,
) -> Response {
    let Some(principal) = bearer(&headers).and_then(|t| gateway.principal(t)) else {
        return error_response(ErrorCode::Unauthorized, "A valid client token is required");
    };
    let frame: ClientFrame = match body(&bytes) {
        Ok(frame) => frame,
        Err(refused) => return refused,
    };
    let request_id = match &frame {
        ClientFrame::Command { request_id, .. } => request_id.clone(),
        _ => return error_response(ErrorCode::Invalid, "Send a command frame"),
    };
    Json(
        gateway
            .answer(request_id, frame, principal.is_owner())
            .await,
    )
    .into_response()
}

/// A device redeems the printed code: it becomes the owner device at once,
/// unless one already exists.
async fn redeem(State(gateway): State<Arc<SetupGateway>>, bytes: axum::body::Bytes) -> Response {
    let request: RedeemRequest = match body(&bytes) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    if gateway.owner_seen.load(Ordering::SeqCst) {
        return refusal(ProtocolError::new(
            ErrorCode::Unauthorized,
            "This daemon is already being set up from another device",
        ));
    }
    let offer_id = request.offer_id.clone();
    if let Err(error) = gateway.pairing.redeem(request).await {
        return refusal(error);
    }
    // Only one device may win the race to the printed code.
    if gateway.owner_seen.swap(true, Ordering::SeqCst) {
        let _ = gateway.pairing.decide(&offer_id, false).await;
        return refusal(ProtocolError::new(
            ErrorCode::Unauthorized,
            "This daemon is already being set up from another device",
        ));
    }
    match gateway.pairing.decide(&offer_id, true).await {
        Ok(enrollment) => {
            info!("Paired the owner device during setup");
            (StatusCode::ACCEPTED, Json(enrollment)).into_response()
        }
        Err(error) => {
            gateway.owner_seen.store(false, Ordering::SeqCst);
            refusal(error)
        }
    }
}

async fn enrollment(
    State(gateway): State<Arc<SetupGateway>>,
    Path(offer_id): Path<String>,
) -> Response {
    reply(gateway.pairing.enrollment(&offer_id).await)
}

async fn challenge(State(gateway): State<Arc<SetupGateway>>, bytes: axum::body::Bytes) -> Response {
    let request: ChallengeRequest = match body(&bytes) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    reply(
        gateway
            .pairing
            .challenge(&request.device_id, request.endpoint.as_deref())
            .await,
    )
}

async fn open_session(
    State(gateway): State<Arc<SetupGateway>>,
    bytes: axum::body::Bytes,
) -> Response {
    let request: SessionRequest = match body(&bytes) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    reply(
        gateway
            .pairing
            // Setup mode listens on loopback only: every request is local.
            .open_session(
                &request.challenge_id,
                &request.signature,
                request.audience.as_deref(),
                true,
            )
            .await,
    )
}

async fn web_page(State(gateway): State<Arc<SetupGateway>>) -> Response {
    web_page_for(&gateway.config)
}

async fn app_index(State(gateway): State<Arc<SetupGateway>>, headers: HeaderMap) -> Response {
    serve_app_file(&gateway.config, "index.html", &headers).await
}

async fn app_file(
    State(gateway): State<Arc<SetupGateway>>,
    Path(path): Path<String>,
    headers: HeaderMap,
) -> Response {
    serve_app_file(&gateway.config, &path, &headers).await
}

async fn websocket(
    State(gateway): State<Arc<SetupGateway>>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let by_header = match bearer(&headers) {
        Some(token) => match gateway.principal(token) {
            Some(principal) => Some(principal),
            None => {
                return error_response(ErrorCode::Unauthorized, "A valid client token is required");
            }
        },
        None => None,
    };
    upgrade
        .max_message_size(MAX_BODY)
        .on_upgrade(move |socket| async move {
            if let Err(error) = run_socket(gateway, socket, by_header).await {
                debug!(%error, "Setup socket closed");
            }
        })
}

async fn run_socket(
    gateway: Arc<SetupGateway>,
    mut socket: WebSocket,
    by_header: Option<Principal>,
) -> anyhow::Result<()> {
    let hello = match tokio::time::timeout(HELLO_WITHIN, socket.recv()).await {
        Ok(Some(Ok(Message::Text(text)))) => serde_json::from_str::<ClientFrame>(&text).ok(),
        _ => None,
    };
    let Some(ClientFrame::Hello { version, token, .. }) = hello else {
        return refuse(&mut socket, ErrorCode::Invalid, "Say hello first").await;
    };
    let Some(principal) = by_header.or_else(|| token.and_then(|t| gateway.principal(&t))) else {
        return refuse(
            &mut socket,
            ErrorCode::Unauthorized,
            "A valid client token is required",
        )
        .await;
    };
    if !ProtocolVersion::CURRENT.compatible_with(version) {
        return refuse(
            &mut socket,
            ErrorCode::VersionUnsupported,
            &format!(
                "This daemon speaks protocol {}.x",
                ProtocolVersion::CURRENT.major
            ),
        )
        .await;
    }
    let local = principal.is_owner();
    let mut updates = gateway.updates.clone();
    send(
        &mut socket,
        &ServerFrame::Welcome {
            version: ProtocolVersion::CURRENT,
            server_capabilities: Vec::new(),
            owner: OwnerInfo::default(),
            latest_seq: 0,
            setup: Some(gateway.host.state(local).await),
        },
    )
    .await?;
    let revoked = principal.revoked.clone();
    let mut last_heard = tokio::time::Instant::now();
    loop {
        let idle = tokio::time::sleep_until(last_heard + IDLE_LIMIT);
        tokio::select! {
            incoming = socket.recv() => {
                let Some(Ok(message)) = incoming else { return Ok(()) };
                last_heard = tokio::time::Instant::now();
                let text = match message {
                    Message::Text(text) => text,
                    Message::Close(_) => return Ok(()),
                    _ => continue,
                };
                match serde_json::from_str::<ClientFrame>(&text) {
                    Ok(ClientFrame::Ping) => send(&mut socket, &ServerFrame::Pong).await?,
                    Ok(frame @ ClientFrame::Command { .. }) => {
                        let ClientFrame::Command { request_id, .. } = &frame else { continue };
                        let request_id = request_id.clone();
                        let reply = gateway.answer(request_id, frame, local).await;
                        send(&mut socket, &reply).await?;
                    }
                    Ok(ClientFrame::Hello { .. }) => {
                        send(&mut socket, &ServerFrame::Error {
                            request_id: None,
                            error: ProtocolError::new(ErrorCode::Invalid, "Already said hello"),
                        }).await?;
                    }
                    Err(error) => {
                        send(&mut socket, &ServerFrame::Error {
                            request_id: None,
                            error: ProtocolError::new(ErrorCode::Invalid, error.to_string()),
                        }).await?;
                    }
                }
            }
            changed = updates.changed() => {
                if changed.is_err() {
                    return Ok(());
                }
                // Each client is told in its own terms (this machine or not).
                let _ = updates.borrow_and_update();
                let setup = gateway.host.state(local).await;
                send(&mut socket, &ServerFrame::Event {
                    seq: None,
                    at_ms: crate::bus::now_ms(),
                    session_id: None,
                    event: Event::SetupUpdated { setup },
                }).await?;
            }
            _ = idle => {
                let _ = socket.send(Message::Close(None)).await;
                return Ok(());
            }
            _ = revoked.cancelled() => return Ok(()),
            _ = gateway.stop.cancelled() => {
                let _ = socket.send(Message::Close(None)).await;
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// Every route sits under `/client/v1`, as on the full gateway.
    #[test]
    fn every_route_is_under_the_service_prefix() {
        let source = include_str!("setup.rs");
        let body = source
            .split("pub fn router(gateway: Arc<SetupGateway>) -> Router {")
            .nth(1)
            .expect("router() is in this file")
            .split("\n}\n")
            .next()
            .unwrap();
        let routes: Vec<&str> = body
            .split(".route(")
            .skip(1)
            .map(|rest| rest.trim_start().trim_start_matches('"'))
            .collect();
        assert!(!routes.is_empty());
        for route in routes {
            assert!(route.starts_with("/client/v1/"), "{route}");
        }
    }
}
