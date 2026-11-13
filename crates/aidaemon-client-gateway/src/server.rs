//! The gateway's routes: the WebSocket at `/client/v1/ws`, and the HTTP
//! mode for clients that cannot hold a socket (watches, scripts):
//! `POST /client/v1/commands` and the long-poll `GET /client/v1/events`.

// Handlers return a refusal as the response itself (`Result<_, Response>`).
#![allow(clippy::result_large_err)]

use std::sync::Arc;
use std::time::Duration;

use aidaemon_client_protocol::{
    Capability, ChallengeRequest, ClientFrame, ClientKind, ErrorCode, Event, EventsPage,
    OfferRequest, PairingDecision, ProtocolError, ProtocolVersion, RedeemRequest, Scope,
    ServerFrame, SessionRequest,
};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info};

use crate::pairing::Principal;
use crate::{Caller, ClientGateway};

/// The largest frame or request body accepted.
const MAX_BODY: usize = 256 * 1024;
/// How long a socket may take to say hello.
const HELLO_WITHIN: Duration = Duration::from_secs(10);
/// A socket that says nothing for this long (clients ping every 25 s) is
/// closed; the client reconnects with `resume_from`.
const IDLE_LIMIT: Duration = Duration::from_secs(60);
/// Log rows read per page when replaying.
const REPLAY_PAGE: u32 = 500;
/// The longest a long-poll waits.
const MAX_WAIT: Duration = Duration::from_secs(25);

/// The gateway's routes.
pub fn router(gateway: Arc<ClientGateway>) -> Router {
    Router::new()
        .route("/client/v1/ws", get(websocket))
        .route("/client/v1/commands", post(http_command))
        .route("/client/v1/events", get(http_events))
        .route("/client/v1/pairing/offers", post(create_offer))
        .route("/client/v1/pairing/offers/{offer_id}", get(offer_status))
        .route(
            "/client/v1/pairing/offers/{offer_id}/decision",
            post(decide_offer),
        )
        .route("/client/v1/enrollments/redeem", post(redeem))
        .route("/client/v1/enrollments/{offer_id}", get(enrollment))
        .route("/client/v1/sessions/challenge", post(challenge))
        .route("/client/v1/sessions", post(open_session))
        .route("/client/v1/devices", get(devices))
        .route("/client/v1/devices/{device_id}/revoke", post(revoke_device))
        .route("/client/v1/app", get(web_page))
        .route("/client/v1/app/", get(web_page))
        .route("/client/v1/app/{module}", get(web_script))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY))
        .with_state(gateway)
}

/// Listen on the configured loopback address until the process ends.
pub async fn serve(gateway: Arc<ClientGateway>) -> anyhow::Result<()> {
    let addr = gateway.config.bind_addr()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(%addr, "Client Gateway listening");
    axum::serve(listener, router(gateway)).await?;
    Ok(())
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::trim)
}

fn principal(gateway: &ClientGateway, token: Option<&str>) -> Option<Principal> {
    gateway.principal(token?)
}

/// The caller with this token, or the response refusing it.
fn authenticate(gateway: &ClientGateway, headers: &HeaderMap) -> Result<Principal, Response> {
    principal(gateway, bearer(headers))
        .ok_or_else(|| error_response(ErrorCode::Unauthorized, "A valid client token is required"))
}

fn status_of(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::Unauthorized => StatusCode::UNAUTHORIZED,
        ErrorCode::Forbidden => StatusCode::FORBIDDEN,
        ErrorCode::NotFound => StatusCode::NOT_FOUND,
        ErrorCode::Conflict => StatusCode::CONFLICT,
        ErrorCode::RateLimited => StatusCode::TOO_MANY_REQUESTS,
        ErrorCode::Invalid | ErrorCode::VersionUnsupported => StatusCode::BAD_REQUEST,
        ErrorCode::Internal | ErrorCode::Unknown => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn error_response(code: ErrorCode, message: &str) -> Response {
    (
        status_of(code),
        Json(ServerFrame::Error {
            request_id: None,
            error: ProtocolError::new(code, message),
        }),
    )
        .into_response()
}

/// A frame as `caller` gets it: compact clients get no turn streaming and
/// approvals without their detail. `None`: not sent at all.
fn fit(frame: &ServerFrame, compact: bool) -> Option<ServerFrame> {
    if !compact {
        return Some(frame.clone());
    }
    use aidaemon_client_protocol::CommandResult;
    let mut frame = frame.clone();
    match &mut frame {
        ServerFrame::Event { event, .. } => match event {
            Event::TurnDelta { .. } | Event::TurnStatus { .. } => return None,
            Event::ApprovalRequested { approval } => {
                *approval = approval.clone().compact();
            }
            _ => {}
        },
        ServerFrame::Snapshot { state, .. }
        | ServerFrame::Ack {
            result: CommandResult::Snapshot { state, .. },
            ..
        } => {
            state.pending_approvals = std::mem::take(&mut state.pending_approvals)
                .into_iter()
                .map(|a| a.compact())
                .collect();
        }
        _ => {}
    }
    Some(frame)
}

/// The caller an HTTP request names: `X-Aidaemon-Client` (its kind),
/// `X-Aidaemon-Device` and `X-Aidaemon-Capabilities` (comma-separated).
/// A watch is compact unless it says otherwise.
fn http_caller(headers: &HeaderMap, principal: Principal) -> Caller {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    let kind = header("x-aidaemon-client")
        .and_then(|kind| serde_json::from_value(serde_json::Value::String(kind)).ok());
    let capabilities: Vec<Capability> = header("x-aidaemon-capabilities")
        .map(|list| {
            list.split(',')
                .filter_map(|c| {
                    serde_json::from_value(serde_json::Value::String(c.trim().to_string())).ok()
                })
                .collect()
        })
        .unwrap_or_default();
    caller_of(kind, header("x-aidaemon-device"), &capabilities, principal)
}

fn caller_of(
    kind: Option<ClientKind>,
    device: Option<String>,
    capabilities: &[Capability],
    principal: Principal,
) -> Caller {
    Caller {
        compact: capabilities.contains(&Capability::Compact) || kind == Some(ClientKind::Watch),
        kind,
        device,
        principal,
        view_connection: None,
    }
}

fn refusal(error: ProtocolError) -> Response {
    (
        status_of(error.code),
        Json(ServerFrame::Error {
            request_id: None,
            error,
        }),
    )
        .into_response()
}

fn reply<T: serde::Serialize>(result: Result<T, ProtocolError>) -> Response {
    match result {
        Ok(value) => Json(value).into_response(),
        Err(error) => refusal(error),
    }
}

/// Only the owner's local token may pair and revoke devices.
fn owner_only(gateway: &ClientGateway, headers: &HeaderMap) -> Result<(), Response> {
    let principal = authenticate(gateway, headers)?;
    if principal.is_owner() {
        Ok(())
    } else {
        Err(error_response(
            ErrorCode::Forbidden,
            "Only the owner's local token may manage devices",
        ))
    }
}

fn body<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, Response> {
    serde_json::from_slice(bytes)
        .map_err(|error| error_response(ErrorCode::Invalid, &error.to_string()))
}

async fn create_offer(
    State(gateway): State<Arc<ClientGateway>>,
    headers: HeaderMap,
    bytes: axum::body::Bytes,
) -> Response {
    if let Err(refused) = owner_only(&gateway, &headers) {
        return refused;
    }
    let request: OfferRequest = if bytes.is_empty() {
        OfferRequest::default()
    } else {
        match body(&bytes) {
            Ok(request) => request,
            Err(refused) => return refused,
        }
    };
    reply(gateway.pairing.offer(request.confirm_in_session).await)
}

async fn offer_status(
    State(gateway): State<Arc<ClientGateway>>,
    headers: HeaderMap,
    Path(offer_id): Path<String>,
) -> Response {
    if let Err(refused) = owner_only(&gateway, &headers) {
        return refused;
    }
    reply(gateway.pairing.enrollment(&offer_id).await)
}

async fn decide_offer(
    State(gateway): State<Arc<ClientGateway>>,
    headers: HeaderMap,
    Path(offer_id): Path<String>,
    bytes: axum::body::Bytes,
) -> Response {
    if let Err(refused) = owner_only(&gateway, &headers) {
        return refused;
    }
    let decision: PairingDecision = match body(&bytes) {
        Ok(decision) => decision,
        Err(refused) => return refused,
    };
    reply(gateway.pairing.decide(&offer_id, decision.approve).await)
}

async fn redeem(State(gateway): State<Arc<ClientGateway>>, bytes: axum::body::Bytes) -> Response {
    let request: RedeemRequest = match body(&bytes) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    match gateway.pairing.redeem(request).await {
        Ok(enrollment) => (StatusCode::ACCEPTED, Json(enrollment)).into_response(),
        Err(error) => refusal(error),
    }
}

async fn enrollment(
    State(gateway): State<Arc<ClientGateway>>,
    Path(offer_id): Path<String>,
) -> Response {
    reply(gateway.pairing.enrollment(&offer_id).await)
}

async fn challenge(
    State(gateway): State<Arc<ClientGateway>>,
    bytes: axum::body::Bytes,
) -> Response {
    let request: ChallengeRequest = match body(&bytes) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    reply(gateway.pairing.challenge(&request.device_id).await)
}

async fn open_session(
    State(gateway): State<Arc<ClientGateway>>,
    bytes: axum::body::Bytes,
) -> Response {
    let request: SessionRequest = match body(&bytes) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    reply(
        gateway
            .pairing
            .open_session(&request.challenge_id, &request.signature)
            .await,
    )
}

async fn devices(State(gateway): State<Arc<ClientGateway>>, headers: HeaderMap) -> Response {
    if let Err(refused) = owner_only(&gateway, &headers) {
        return refused;
    }
    reply(gateway.pairing.devices().await)
}

async fn revoke_device(
    State(gateway): State<Arc<ClientGateway>>,
    headers: HeaderMap,
    Path(device_id): Path<String>,
) -> Response {
    if let Err(refused) = owner_only(&gateway, &headers) {
        return refused;
    }
    if let Err(error) = gateway.pairing.revoke(&device_id).await {
        return refusal(error);
    }
    reply(gateway.pairing.devices().await)
}

/// The web client: one page and its modules, built from `clients/web`.
const WEB_PAGE: &str = include_str!("../web/index.html");
const WEB_MODULES: &[(&str, &str)] = &[
    ("app.js", include_str!("../web/app.js")),
    ("link.js", include_str!("../web/link.js")),
    ("platform.js", include_str!("../web/platform.js")),
    ("state.js", include_str!("../web/state.js")),
    ("ui.js", include_str!("../web/ui.js")),
];

async fn web_page() -> Response {
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (
                axum::http::header::CONTENT_SECURITY_POLICY,
                "default-src 'self'; connect-src 'self' ws: wss:; style-src 'self' 'unsafe-inline'",
            ),
        ],
        WEB_PAGE,
    )
        .into_response()
}

async fn web_script(Path(module): Path<String>) -> Response {
    let Some((_, script)) = WEB_MODULES.iter().find(|(name, _)| *name == module) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/javascript; charset=utf-8",
        )],
        *script,
    )
        .into_response()
}

async fn http_command(
    State(gateway): State<Arc<ClientGateway>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let principal = match authenticate(&gateway, &headers) {
        Ok(principal) => principal,
        Err(refused) => return refused,
    };
    let frame: ClientFrame = match serde_json::from_slice(&body) {
        Ok(frame) => frame,
        Err(error) => return error_response(ErrorCode::Invalid, &error.to_string()),
    };
    let ClientFrame::Command {
        request_id,
        command,
    } = frame
    else {
        return error_response(ErrorCode::Invalid, "Send a command frame");
    };
    let caller = http_caller(&headers, principal);
    let reply = gateway.command(&request_id, command, &caller).await;
    let reply = fit(&reply, caller.compact).unwrap_or(reply);
    Json(reply).into_response()
}

#[derive(Debug, Deserialize)]
struct EventsQuery {
    /// The last `seq` applied; absent: start from a snapshot.
    since: Option<u64>,
    /// Seconds to wait for an event when there is none (at most 25).
    #[serde(default)]
    wait: u64,
    #[serde(default = "default_page")]
    limit: u32,
}

fn default_page() -> u32 {
    REPLAY_PAGE
}

async fn http_events(
    State(gateway): State<Arc<ClientGateway>>,
    headers: HeaderMap,
    Query(query): Query<EventsQuery>,
) -> Response {
    let principal = match authenticate(&gateway, &headers) {
        Ok(principal) => principal,
        Err(refused) => return refused,
    };
    if !principal.may(Scope::Observe) {
        return error_response(ErrorCode::Forbidden, "This device may not read events");
    }
    let caller = http_caller(&headers, principal);
    let limit = query.limit.clamp(1, REPLAY_PAGE);
    let since = match query.since {
        Some(since) if gateway.bus.can_resume(since).await => since,
        _ => {
            let (seq, state) = gateway.snapshot().await;
            let frame = ServerFrame::Snapshot { seq, state };
            let frames = fit(&frame, caller.compact).into_iter().collect();
            return Json(EventsPage {
                frames,
                next_since: seq,
            })
            .into_response();
        }
    };
    // Subscribe before reading, so an event logged in between wakes us.
    let mut live = gateway.bus.subscribe();
    let mut frames = gateway.bus.after(since, limit).await.unwrap_or_default();
    if frames.is_empty() && query.wait > 0 {
        let wait = Duration::from_secs(query.wait).min(MAX_WAIT);
        let _ = tokio::time::timeout(wait, async {
            loop {
                match live.recv().await {
                    Ok(frame) if matches!(*frame, ServerFrame::Event { seq: Some(_), .. }) => {
                        return
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return,
                }
            }
        })
        .await;
        frames = gateway.bus.after(since, limit).await.unwrap_or_default();
    }
    let next_since = frames
        .iter()
        .filter_map(|frame| match frame {
            ServerFrame::Event { seq, .. } => *seq,
            _ => None,
        })
        .max()
        .unwrap_or(since);
    let frames = frames
        .iter()
        .filter_map(|frame| fit(frame, caller.compact))
        .collect();
    Json(EventsPage { frames, next_since }).into_response()
}

async fn websocket(
    State(gateway): State<Arc<ClientGateway>>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    // A token in the header is checked now; without one, `hello` must
    // carry it.
    let by_header = match bearer(&headers) {
        Some(token) => match principal(&gateway, Some(token)) {
            Some(principal) => Some(principal),
            None => {
                return error_response(ErrorCode::Unauthorized, "A valid client token is required")
            }
        },
        None => None,
    };
    upgrade
        .max_message_size(MAX_BODY)
        .on_upgrade(move |socket| async move {
            if let Err(error) = run_socket(gateway, socket, by_header).await {
                debug!(%error, "Client socket closed");
            }
        })
}

async fn send(socket: &mut WebSocket, frame: &ServerFrame) -> anyhow::Result<()> {
    let text = serde_json::to_string(frame)?;
    socket.send(Message::Text(text.into())).await?;
    Ok(())
}

async fn refuse(socket: &mut WebSocket, code: ErrorCode, message: &str) -> anyhow::Result<()> {
    send(
        socket,
        &ServerFrame::Error {
            request_id: None,
            error: ProtocolError::new(code, message),
        },
    )
    .await?;
    let _ = socket.send(Message::Close(None)).await;
    anyhow::bail!("{message}")
}

async fn run_socket(
    gateway: Arc<ClientGateway>,
    mut socket: WebSocket,
    by_header: Option<Principal>,
) -> anyhow::Result<()> {
    let hello = match tokio::time::timeout(HELLO_WITHIN, socket.recv()).await {
        Ok(Some(Ok(Message::Text(text)))) => serde_json::from_str::<ClientFrame>(&text).ok(),
        _ => None,
    };
    let Some(ClientFrame::Hello {
        version,
        client,
        capabilities,
        resume_from,
        token,
    }) = hello
    else {
        return refuse(&mut socket, ErrorCode::Invalid, "Say hello first").await;
    };
    let Some(principal) = by_header.or_else(|| principal(&gateway, token.as_deref())) else {
        return refuse(
            &mut socket,
            ErrorCode::Unauthorized,
            "A valid client token is required",
        )
        .await;
    };
    if !principal.may(Scope::Observe) {
        return refuse(
            &mut socket,
            ErrorCode::Forbidden,
            "This device may not read events",
        )
        .await;
    }
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
    let revoked = principal.revoked.clone();
    let mut caller = caller_of(
        Some(client.kind),
        client.device.clone(),
        &capabilities,
        principal,
    );
    // A socket that can be moved gets its own outbox for view actions.
    let mut view = capabilities
        .contains(&Capability::ViewActions)
        .then(|| gateway.views.connect());
    caller.view_connection = view.as_ref().map(|view| view.id);

    // Subscribe before reading the log, so nothing falls between them.
    let mut live = gateway.bus.subscribe();
    let latest = gateway.bus.latest().await;
    send(
        &mut socket,
        &ServerFrame::Welcome {
            version: ProtocolVersion::CURRENT,
            server_capabilities: vec![Capability::Compact, Capability::ViewActions],
            owner: gateway.owner(),
            latest_seq: latest,
        },
    )
    .await?;

    // Catch up: everything after the client's seq, or a snapshot.
    let mut sent = match resume_from {
        Some(seq) if gateway.bus.can_resume(seq).await => seq,
        _ => {
            let (seq, state) = gateway.snapshot().await;
            if let Some(frame) = fit(&ServerFrame::Snapshot { seq, state }, caller.compact) {
                send(&mut socket, &frame).await?;
            }
            seq
        }
    };
    sent = replay(&gateway, &mut socket, sent, caller.compact).await?;

    let (replies_tx, mut replies) = mpsc::channel::<ServerFrame>(64);
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
                    Ok(ClientFrame::Command { request_id, command }) => {
                        // Commands run beside the event stream; each sends
                        // exactly one reply.
                        let gateway = gateway.clone();
                        let caller = caller.clone();
                        let replies = replies_tx.clone();
                        tokio::spawn(async move {
                            let reply = gateway.command(&request_id, command, &caller).await;
                            let _ = replies.send(reply).await;
                        });
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
            Some(frame) = async {
                match view.as_mut() {
                    Some(view) => view.outbox.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                send(&mut socket, &frame).await?;
            }
            Some(reply) = replies.recv() => {
                if let Some(reply) = fit(&reply, caller.compact) {
                    send(&mut socket, &reply).await?;
                }
            }
            frame = live.recv() => match frame {
                Ok(frame) => {
                    match &*frame {
                        // Already sent from the log.
                        ServerFrame::Event { seq: Some(seq), .. } if *seq <= sent => continue,
                        // A gap means frames were missed: read them from the log.
                        ServerFrame::Event { seq: Some(seq), .. } if *seq > sent + 1 => {
                            sent = replay(&gateway, &mut socket, sent, caller.compact).await?;
                            continue;
                        }
                        ServerFrame::Event { seq: Some(seq), .. } => sent = *seq,
                        _ => {}
                    }
                    if let Some(frame) = fit(&frame, caller.compact) {
                        send(&mut socket, &frame).await?;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    sent = replay(&gateway, &mut socket, sent, caller.compact).await?;
                }
                Err(broadcast::error::RecvError::Closed) => return Ok(()),
            },
            _ = idle => {
                let _ = socket.send(Message::Close(None)).await;
                return Ok(());
            }
            _ = revoked.cancelled() => {
                send(&mut socket, &ServerFrame::Error {
                    request_id: None,
                    error: ProtocolError::new(ErrorCode::Unauthorized, "This device was revoked"),
                }).await?;
                let _ = socket.send(Message::Close(None)).await;
                return Ok(());
            }
        }
    }
}

/// Send every logged event after `sent`; returns the last `seq` sent.
async fn replay(
    gateway: &ClientGateway,
    socket: &mut WebSocket,
    mut sent: u64,
    compact: bool,
) -> anyhow::Result<u64> {
    loop {
        let frames = gateway.bus.after(sent, REPLAY_PAGE).await?;
        if frames.is_empty() {
            return Ok(sent);
        }
        for frame in &frames {
            if let ServerFrame::Event { seq: Some(seq), .. } = frame {
                sent = *seq;
            }
            if let Some(frame) = fit(frame, compact) {
                send(socket, &frame).await?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_client_protocol::{Approval, ApprovalDecision, ApprovalKind, Risk, TurnStatus};

    fn approval() -> Approval {
        Approval {
            id: "a1".into(),
            session_id: "app:s".into(),
            kind: ApprovalKind::Command,
            risk: Risk::Medium,
            summary: "Run the synthetic report".into(),
            detail: Some("report --all".into()),
            card: None,
            warnings: Vec::new(),
            options: vec![ApprovalDecision::ApproveOnce, ApprovalDecision::Deny],
            requested_at_ms: 1,
            expires_at_ms: None,
            channel: None,
        }
    }

    fn event(event: Event) -> ServerFrame {
        ServerFrame::Event {
            seq: Some(1),
            at_ms: 1,
            session_id: None,
            event,
        }
    }

    #[test]
    fn compact_drops_streaming_and_approval_detail() {
        let status = ServerFrame::Event {
            seq: None,
            at_ms: 1,
            session_id: None,
            event: Event::TurnStatus {
                turn_id: "t".into(),
                status: TurnStatus::Thinking,
            },
        };
        assert!(fit(&status, true).is_none());
        assert!(fit(&status, false).is_some());
        let asked = event(Event::ApprovalRequested {
            approval: approval(),
        });
        let Some(ServerFrame::Event {
            event: Event::ApprovalRequested { approval },
            ..
        }) = fit(&asked, true)
        else {
            panic!("kept");
        };
        assert_eq!(approval.detail, None);
        assert_eq!(approval.summary, "Run the synthetic report");
    }

    #[test]
    fn a_watch_is_compact_by_default() {
        let mut headers = HeaderMap::new();
        headers.insert("x-aidaemon-client", "watch".parse().unwrap());
        headers.insert("x-aidaemon-device", "Synthetic Watch".parse().unwrap());
        let caller = http_caller(&headers, Principal::owner());
        assert!(caller.compact);
        assert_eq!(caller.kind, Some(ClientKind::Watch));
        assert_eq!(caller.device.as_deref(), Some("Synthetic Watch"));
        let mut headers = HeaderMap::new();
        headers.insert("x-aidaemon-capabilities", "compact, goals".parse().unwrap());
        assert!(http_caller(&headers, Principal::owner()).compact);
        assert!(!http_caller(&HeaderMap::new(), Principal::owner()).compact);
    }
}
