use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, time::Instant};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::channels::ChannelHub;
use crate::config::NodesConfig;

use super::domain::NODE_CHANNEL_NAME;
use super::protocol::*;
use super::service::{NodeConversationIngress, NodeService};

#[derive(Clone)]
pub struct NodeGatewayState {
    pub service: Arc<NodeService>,
    pub ingress: Arc<dyn NodeConversationIngress>,
    pub hub: Arc<ChannelHub>,
    pub config: NodesConfig,
    pub rate_limiter: Arc<NodeRateLimiter>,
}

#[derive(Default)]
pub struct NodeRateLimiter {
    windows: tokio::sync::Mutex<HashMap<(String, &'static str), RateWindow>>,
}

struct RateWindow {
    started: Instant,
    units: u64,
}

impl NodeRateLimiter {
    const MAX_WINDOWS: usize = 4096;

    async fn consume(
        &self,
        identity: &str,
        bucket: &'static str,
        units: u64,
        limit: u64,
    ) -> Result<(), GatewayError> {
        if limit == 0 {
            return Err(GatewayError::rate_limited());
        }
        let mut windows = self.windows.lock().await;
        let key = (identity.to_string(), bucket);
        if windows.len() >= Self::MAX_WINDOWS && !windows.contains_key(&key) {
            windows.retain(|_, window| window.started.elapsed() < Duration::from_secs(60));
            if windows.len() >= Self::MAX_WINDOWS {
                return Err(GatewayError::rate_limited());
            }
        }
        let window = windows.entry(key).or_insert_with(|| RateWindow {
            started: Instant::now(),
            units: 0,
        });
        if window.started.elapsed() >= Duration::from_secs(60) {
            window.started = Instant::now();
            window.units = 0;
        }
        if window.units.saturating_add(units) > limit {
            return Err(GatewayError::rate_limited());
        }
        window.units = window.units.saturating_add(units);
        Ok(())
    }
}

async fn rate_limit_authenticated_request(
    state: &NodeGatewayState,
    node_id: &str,
) -> Result<(), GatewayError> {
    state
        .rate_limiter
        .consume(
            node_id,
            "requests",
            1,
            u64::from(state.config.limits.requests_per_minute),
        )
        .await
}

#[derive(Debug)]
struct GatewayError {
    status: StatusCode,
    code: &'static str,
    message: String,
    retryable: bool,
}

impl GatewayError {
    fn bad_request(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request",
            message: error.to_string(),
            retryable: false,
        }
    }
    fn unauthorized(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "authentication_failed",
            message: error.to_string(),
            retryable: false,
        }
    }
    fn forbidden(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "not_authorized",
            message: error.to_string(),
            retryable: false,
        }
    }
    fn internal(error: impl std::fmt::Display) -> Self {
        tracing::warn!(error = %error, "Node Gateway request failed");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal_error",
            message: "The Node Gateway could not complete the request".to_string(),
            retryable: true,
        }
    }
    fn rate_limited() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "rate_limited",
            message: "Node request rate exceeded the server limit".to_string(),
            retryable: true,
        }
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ProtocolErrorBody {
                error: ProtocolError {
                    code: self.code.to_string(),
                    message: self.message,
                    retryable: self.retryable,
                    request_id: None,
                },
            }),
        )
            .into_response()
    }
}

pub fn router(state: NodeGatewayState) -> Router {
    let limit = state.config.limits.control_message_bytes;
    Router::new()
        .route("/node/v1/enrollments/redeem", post(redeem))
        .route("/node/v1/sessions/challenge", post(challenge))
        .route("/node/v1/sessions", post(open_session))
        .route(
            "/node/v1/credentials/rotation-challenge",
            post(credential_rotation_challenge),
        )
        .route("/node/v1/credentials/rotate", post(rotate_credential))
        .route("/node/v1/heartbeats", post(heartbeat))
        .route("/node/v1/sensor-readings", post(report_sensor_readings))
        .route("/node/v1/outbox", get(node_outbox))
        .route("/node/v1/outbox/{cursor}/ack", post(ack_node_outbox))
        .route("/node/v1/turns", post(create_turn))
        .route("/node/v1/firmware/{release_id}", get(download_firmware))
        .route("/node/v1/uploads/{slot_id}", put(upload_media))
        .route("/node/v1/turns/{turn_id}/commit", post(commit_turn))
        .route("/node/v1/media/{media_id}", get(download_media))
        .route("/node/v1/turns/{turn_id}/events", get(turn_events))
        .route("/node/v1/turns/{turn_id}/cancel", post(cancel_turn))
        .layer(DefaultBodyLimit::max(
            limit
                .max(state.config.limits.audio_upload_bytes)
                .max(state.config.limits.image_upload_bytes),
        ))
        .layer(middleware::from_fn(security_headers))
        .with_state(state)
}

pub async fn serve(state: NodeGatewayState) -> anyhow::Result<()> {
    let bind: std::net::IpAddr = state.config.gateway.bind.parse()?;
    anyhow::ensure!(
        bind.is_loopback(),
        "Node Gateway bind address must be loopback"
    );
    let address = std::net::SocketAddr::new(bind, state.config.gateway.port);
    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!(%address, "Node Gateway listening");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

async fn security_headers(request: Request<Body>, next: Next) -> Response {
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    response
        .headers_mut()
        .insert("x-content-type-options", "nosniff".parse().unwrap());
    response
}

async fn redeem(
    State(state): State<NodeGatewayState>,
    Json(request): Json<RedeemEnrollmentRequest>,
) -> Result<Json<RedeemEnrollmentResponse>, GatewayError> {
    state
        .rate_limiter
        .consume(&request.offer_id, "enrollment", 1, 10)
        .await?;
    state
        .service
        .redeem(request)
        .await
        .map(Json)
        .map_err(GatewayError::bad_request)
}

async fn challenge(
    State(state): State<NodeGatewayState>,
    Json(request): Json<SessionChallengeRequest>,
) -> Result<Json<SessionChallengeResponse>, GatewayError> {
    state
        .rate_limiter
        .consume(&request.credential_id, "authentication", 1, 30)
        .await?;
    state
        .service
        .challenge(request)
        .await
        .map(Json)
        .map_err(GatewayError::bad_request)
}

async fn open_session(
    State(state): State<NodeGatewayState>,
    Json(request): Json<OpenSessionRequest>,
) -> Result<Json<OpenSessionResponse>, GatewayError> {
    state
        .rate_limiter
        .consume(&request.credential_id, "authentication", 1, 30)
        .await?;
    state
        .service
        .open_session(request)
        .await
        .map(Json)
        .map_err(GatewayError::unauthorized)
}

async fn credential_rotation_challenge(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Json(request): Json<CredentialRotationChallengeRequest>,
) -> Result<Json<CredentialRotationChallengeResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    state
        .service
        .create_credential_rotation_challenge(&context, request)
        .await
        .map(Json)
        .map_err(GatewayError::bad_request)
}

async fn rotate_credential(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Json(request): Json<RotateCredentialRequest>,
) -> Result<Json<RotateCredentialResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    state
        .service
        .rotate_credential(&context, request)
        .await
        .map(Json)
        .map_err(GatewayError::unauthorized)
}

fn bearer(headers: &HeaderMap) -> Result<&str, GatewayError> {
    let value = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| GatewayError::unauthorized("missing bearer token"))?;
    value
        .strip_prefix("Bearer ")
        .filter(|token| !token.is_empty())
        .ok_or_else(|| GatewayError::unauthorized("invalid bearer token"))
}

async fn heartbeat(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Json(request): Json<HeartbeatRequest>,
) -> Result<Json<HeartbeatResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    state
        .rate_limiter
        .consume(
            &context.node_id,
            "requests",
            1,
            u64::from(state.config.limits.requests_per_minute),
        )
        .await?;
    state
        .service
        .heartbeat(&context, request)
        .await
        .map(Json)
        .map_err(GatewayError::forbidden)
}

async fn report_sensor_readings(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Json(request): Json<ReportSensorReadingsRequest>,
) -> Result<Json<ReportSensorReadingsResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    state
        .service
        .report_sensor_readings(&context, request)
        .await
        .map(Json)
        .map_err(GatewayError::forbidden)
}

#[derive(Debug, Deserialize)]
struct NodeOutboxQuery {
    #[serde(default)]
    after: u64,
    #[serde(default)]
    wait_seconds: u64,
}

async fn node_outbox(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Query(query): Query<NodeOutboxQuery>,
) -> Result<Json<NodeOutboxResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(query.wait_seconds.min(25));
    loop {
        let response = state
            .service
            .node_outbox(&context, query.after)
            .await
            .map_err(GatewayError::forbidden)?;
        if !response.events.is_empty() || tokio::time::Instant::now() >= deadline {
            return Ok(Json(response));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn ack_node_outbox(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Path(cursor): Path<u64>,
    Json(request): Json<AckNodeOutboxRequest>,
) -> Result<Json<AckNodeOutboxResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    state
        .service
        .acknowledge_node_outbox(&context, cursor, request)
        .await
        .map(Json)
        .map_err(GatewayError::forbidden)
}

async fn create_turn(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Json(request): Json<CreateTurnRequest>,
) -> Result<(StatusCode, Json<CreateTurnResponse>), GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    state
        .rate_limiter
        .consume(
            &context.node_id,
            "requests",
            1,
            u64::from(state.config.limits.requests_per_minute),
        )
        .await?;
    state
        .rate_limiter
        .consume(
            &context.node_id,
            "turns",
            1,
            u64::from(state.config.limits.turns_per_minute),
        )
        .await?;
    let idempotency_key = headers
        .get("idempotency-key")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| GatewayError::bad_request("Idempotency-Key is required"))?;
    state.hub.session_map().write().await.insert(
        context.conversation_session_id.clone(),
        NODE_CHANNEL_NAME.to_string(),
    );
    let (outcome, upload_slots) = match request {
        CreateTurnRequest::Text { request_id, text } => {
            let outcome = state
                .service
                .create_text_turn(&context, idempotency_key, &request_id, &text)
                .await
                .map_err(GatewayError::forbidden)?;
            if !outcome.duplicate {
                let service = state.service.clone();
                let ingress = state.ingress.clone();
                let turn_id = outcome.turn_id.clone();
                tokio::spawn(async move {
                    service.process_text_turn(context, turn_id, ingress).await;
                });
            }
            (outcome, Vec::new())
        }
        media => {
            let (outcome, slot) = state
                .service
                .create_media_turn(&context, idempotency_key, media)
                .await
                .map_err(GatewayError::forbidden)?;
            (outcome, vec![slot])
        }
    };
    let status = if outcome.duplicate {
        StatusCode::OK
    } else {
        StatusCode::ACCEPTED
    };
    Ok((
        status,
        Json(CreateTurnResponse {
            turn_id: outcome.turn_id,
            state: outcome.state,
            cursor: outcome.cursor,
            upload_slots,
        }),
    ))
}

async fn upload_media(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Path(slot_id): Path<String>,
    body: Bytes,
) -> Result<Json<UploadMediaResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    state
        .rate_limiter
        .consume(
            &context.node_id,
            "requests",
            1,
            u64::from(state.config.limits.requests_per_minute),
        )
        .await?;
    state
        .rate_limiter
        .consume(
            &context.node_id,
            "media_bytes",
            body.len() as u64,
            state.config.limits.media_bytes_per_minute as u64,
        )
        .await?;
    let upload = state
        .service
        .store()
        .upload_slot(&context.node_id, &slot_id)
        .await
        .map_err(GatewayError::bad_request)?;
    if upload.state != "pending" {
        return Err(GatewayError::bad_request("upload slot is not pending"));
    }
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if content_type != upload.content_type {
        return Err(GatewayError::bad_request(
            "upload Content-Type does not match slot",
        ));
    }
    if body.len() as u64 != upload.expected_bytes {
        return Err(GatewayError::bad_request(
            "upload size does not match declaration",
        ));
    }
    let digest = format!("{:x}", Sha256::digest(&body));
    if digest != upload.expected_sha256 {
        return Err(GatewayError::bad_request(
            "upload digest does not match declaration",
        ));
    }
    let directory = std::path::PathBuf::from(
        shellexpand::tilde(&state.config.retention.media_dir).into_owned(),
    );
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(GatewayError::internal)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))
            .await
            .map_err(GatewayError::internal)?;
    }
    let partial = directory.join(format!("{slot_id}.part"));
    let final_path = directory.join(format!("{slot_id}.bin"));
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .await
        .map_err(GatewayError::bad_request)?;
    file.write_all(&body)
        .await
        .map_err(GatewayError::internal)?;
    file.sync_all().await.map_err(GatewayError::internal)?;
    drop(file);
    tokio::fs::rename(&partial, &final_path)
        .await
        .map_err(GatewayError::internal)?;
    if let Err(error) = state
        .service
        .store()
        .mark_slot_uploaded(&context.node_id, &slot_id, &final_path.to_string_lossy())
        .await
    {
        let _ = tokio::fs::remove_file(&final_path).await;
        return Err(GatewayError::bad_request(error));
    }
    Ok(Json(UploadMediaResponse {
        slot_id,
        received_bytes: body.len() as u64,
        sha256: digest,
    }))
}

async fn commit_turn(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Path(turn_id): Path<String>,
    Json(request): Json<CommitTurnRequest>,
) -> Result<Json<CommitTurnResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    let upload = state
        .service
        .store()
        .upload_slot(&context.node_id, &request.slot_id)
        .await
        .map_err(GatewayError::bad_request)?;
    if upload.turn_id != turn_id {
        return Err(GatewayError::bad_request(
            "upload slot does not belong to turn",
        ));
    }
    if upload.state != "uploaded" {
        return Err(GatewayError::bad_request("media upload is not complete"));
    }
    let service = state.service.clone();
    let ingress = state.ingress.clone();
    let context_for_task = context;
    tokio::spawn(async move {
        service
            .process_media_turn(context_for_task, upload, ingress)
            .await;
    });
    Ok(Json(CommitTurnResponse {
        turn_id,
        state: super::domain::NodeTurnState::Accepted,
    }))
}

async fn download_media(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Path(media_id): Path<String>,
) -> Result<Response, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    let (path, content_type, expected_bytes) = state
        .service
        .store()
        .response_media(&context.node_id, &media_id)
        .await
        .map_err(GatewayError::bad_request)?;
    if expected_bytes > state.config.limits.response_audio_bytes as u64 {
        return Err(GatewayError::bad_request("response media exceeds limit"));
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(GatewayError::internal)?;
    if bytes.len() as u64 != expected_bytes {
        return Err(GatewayError::internal("response media size changed"));
    }
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .header("content-length", expected_bytes)
        .header("cache-control", "private, no-store")
        .body(Body::from(bytes))
        .map_err(GatewayError::internal)
}

async fn download_firmware(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Path(release_id): Path<String>,
) -> Result<Response, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    let image = state
        .service
        .firmware_image(&context, &release_id)
        .await
        .map_err(GatewayError::forbidden)?;
    let digest = format!("{:x}", Sha256::digest(image.as_ref()));
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/octet-stream")
        .header("content-length", image.len())
        .header("x-content-sha256", digest)
        .header("cache-control", "private, no-store")
        .body(Body::from(image.to_vec()))
        .map_err(GatewayError::internal)
}

#[derive(Debug, Deserialize)]
struct EventQuery {
    #[serde(default)]
    after: u64,
    #[serde(default = "default_wait_seconds")]
    wait_seconds: u64,
}
fn default_wait_seconds() -> u64 {
    20
}

async fn turn_events(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Path(turn_id): Path<String>,
    Query(query): Query<EventQuery>,
) -> Result<Json<TurnEventsResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(query.wait_seconds.min(25));
    loop {
        let events = state
            .service
            .store()
            .events_after(&context.node_id, &turn_id, query.after, 50)
            .await
            .map_err(GatewayError::bad_request)?;
        if !events.is_empty() || tokio::time::Instant::now() >= deadline {
            let next_cursor = events
                .last()
                .map(|event| event.cursor)
                .unwrap_or(query.after);
            return Ok(Json(TurnEventsResponse {
                events,
                next_cursor,
            }));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn cancel_turn(
    State(state): State<NodeGatewayState>,
    headers: HeaderMap,
    Path(turn_id): Path<String>,
) -> Result<Json<CancelTurnResponse>, GatewayError> {
    let context = state
        .service
        .authenticate(bearer(&headers)?)
        .await
        .map_err(GatewayError::unauthorized)?;
    rate_limit_authenticated_request(&state, &context.node_id).await?;
    let state_value = state
        .service
        .store()
        .cancel_turn(&context.node_id, &turn_id)
        .await
        .map_err(GatewayError::bad_request)?;
    Ok(Json(CancelTurnResponse {
        turn_id,
        state: state_value,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::collections::HashMap;

    use crate::channels::SessionMap;
    use crate::nodes::domain::{AuthenticatedNodeContext, CHILD_COMPANION_POLICY};
    use crate::nodes::service::NodeConversationIngress;
    use crate::nodes::simulator::NodeSimulator;
    use crate::nodes::store::NodeStore;
    use crate::nodes::tool::{ReadNodeHealthTool, ReadNodeSensorsTool};
    use crate::traits::Tool;

    struct EchoIngress;
    #[async_trait]
    impl NodeConversationIngress for EchoIngress {
        async fn respond(
            &self,
            _context: &AuthenticatedNodeContext,
            text: &str,
        ) -> anyhow::Result<String> {
            Ok(format!("Companion heard: {text}"))
        }
    }

    #[tokio::test]
    async fn rate_limiter_enforces_limits_and_bounds_untrusted_identities() {
        let limiter = NodeRateLimiter::default();
        limiter.consume("node-1", "turns", 1, 1).await.unwrap();
        let error = limiter.consume("node-1", "turns", 1, 1).await.unwrap_err();
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);

        for index in 0..NodeRateLimiter::MAX_WINDOWS + 100 {
            let _ = limiter
                .consume(&format!("untrusted-{index}"), "authentication", 1, 30)
                .await;
        }
        assert!(limiter.windows.lock().await.len() <= NodeRateLimiter::MAX_WINDOWS);
    }

    #[tokio::test]
    async fn enrolled_node_authenticates_and_completes_idempotent_turn() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::nodes::store::migrate(&pool).await.unwrap();
        let store = Arc::new(NodeStore::new(pool, [7_u8; 32]));
        let media_dir = tempfile::tempdir().unwrap();
        let mut config = NodesConfig::default();
        config.retention.media_dir = media_dir.path().to_string_lossy().into_owned();
        config.announcements.enabled = true;
        config.monitoring.enabled = true;
        config.ota.enabled = true;
        config.gateway.advertised_endpoints = vec![
            "https://node.example.test".to_string(),
            "https://node-lan.example.test".to_string(),
        ];
        let firmware_image = b"synthetic signed K10 application image";
        let firmware_release =
            crate::nodes::ota::FirmwareRelease::signed_for_test(firmware_image, "0.4.0", 1);
        let service = Arc::new(
            NodeService::new(store.clone(), config.clone()).with_firmware_release(firmware_release),
        );
        let offer = service
            .create_pairing_offer(
                "parent",
                "simulator",
                "Test Companion",
                CHILD_COMPANION_POLICY,
            )
            .await
            .unwrap();
        let sessions: SessionMap = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
        let hub = Arc::new(ChannelHub::new(Vec::new(), sessions));
        let state = NodeGatewayState {
            service: service.clone(),
            ingress: Arc::new(EchoIngress),
            hub,
            config,
            rate_limiter: Arc::new(NodeRateLimiter::default()),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router(state)).await.unwrap();
        });

        let mut simulator = NodeSimulator::new(format!("http://{address}")).unwrap();
        let enrollment = simulator
            .enroll(&offer.offer_id, &offer.offer_secret)
            .await
            .unwrap();
        store
            .set_authorization(
                &enrollment.node_id,
                crate::nodes::domain::NodeAction::SubmitAudioTurn,
                true,
                serde_json::json!({"max_duration_ms":10000}),
            )
            .await
            .unwrap();
        store
            .set_authorization(
                &enrollment.node_id,
                crate::nodes::domain::NodeAction::ReportSensor,
                true,
                serde_json::json!({"capabilities":[
                    "sensor.environment.temperature",
                    "sensor.environment.humidity"
                ]}),
            )
            .await
            .unwrap();
        simulator.open_session().await.unwrap();
        let heartbeat_without_audio_authorization = simulator
            .heartbeat(vec![
                CapabilityObservation {
                    capability_id: "sensor.environment.temperature".to_string(),
                    version: 1,
                    limits: serde_json::json!({"unit":"celsius"}),
                },
                CapabilityObservation {
                    capability_id: "sensor.environment.humidity".to_string(),
                    version: 1,
                    limits: serde_json::json!({"unit":"percent_rh"}),
                },
                CapabilityObservation {
                    capability_id: "output.audio".to_string(),
                    version: 1,
                    limits: serde_json::json!({"content_type":"audio/wav","max_bytes":1048576}),
                },
                CapabilityObservation {
                    capability_id: "firmware.ota".to_string(),
                    version: 1,
                    limits: serde_json::json!({"board":"unihiker_k10","slot_bytes":2621440}),
                },
            ])
            .await
            .unwrap();
        assert_eq!(heartbeat_without_audio_authorization.outbox_poll_seconds, 0);
        assert_eq!(
            heartbeat_without_audio_authorization.gateway_endpoints,
            vec![
                "https://node.example.test".to_string(),
                "https://node-lan.example.test".to_string(),
            ]
        );
        assert!(heartbeat_without_audio_authorization
            .firmware_update
            .is_none());
        store
            .set_authorization(
                &enrollment.node_id,
                crate::nodes::domain::NodeAction::ReceiveAudio,
                true,
                serde_json::json!({"content_types":["audio/wav"]}),
            )
            .await
            .unwrap();
        store
            .set_authorization(
                &enrollment.node_id,
                crate::nodes::domain::NodeAction::ReceiveOta,
                true,
                serde_json::json!({"board":"unihiker_k10","channel":"stable"}),
            )
            .await
            .unwrap();
        simulator.open_session().await.unwrap();
        let heartbeat_with_audio_authorization = simulator
            .heartbeat(vec![
                CapabilityObservation {
                    capability_id: "sensor.environment.temperature".to_string(),
                    version: 1,
                    limits: serde_json::json!({"unit":"celsius"}),
                },
                CapabilityObservation {
                    capability_id: "sensor.environment.humidity".to_string(),
                    version: 1,
                    limits: serde_json::json!({"unit":"percent_rh"}),
                },
                CapabilityObservation {
                    capability_id: "output.audio".to_string(),
                    version: 1,
                    limits: serde_json::json!({"content_type":"audio/wav","max_bytes":1048576}),
                },
                CapabilityObservation {
                    capability_id: "firmware.ota".to_string(),
                    version: 1,
                    limits: serde_json::json!({"board":"unihiker_k10","slot_bytes":2621440}),
                },
            ])
            .await
            .unwrap();
        assert_eq!(heartbeat_with_audio_authorization.outbox_poll_seconds, 3);
        let firmware_offer = heartbeat_with_audio_authorization
            .firmware_update
            .as_ref()
            .expect("authorized OTA-capable Node should receive the release");
        assert_eq!(firmware_offer.version, "0.4.0");
        assert_eq!(firmware_offer.sequence, 1);
        assert_eq!(
            simulator
                .download_firmware(&firmware_offer.download_path)
                .await
                .unwrap(),
            firmware_image
        );
        let heartbeat_at_release_sequence = simulator
            .heartbeat(vec![
                CapabilityObservation {
                    capability_id: "sensor.environment.temperature".to_string(),
                    version: 1,
                    limits: serde_json::json!({"unit":"celsius"}),
                },
                CapabilityObservation {
                    capability_id: "sensor.environment.humidity".to_string(),
                    version: 1,
                    limits: serde_json::json!({"unit":"percent_rh"}),
                },
                CapabilityObservation {
                    capability_id: "output.audio".to_string(),
                    version: 1,
                    limits: serde_json::json!({
                        "content_type":"audio/wav",
                        "max_bytes":1048576
                    }),
                },
                CapabilityObservation {
                    capability_id: "firmware.ota".to_string(),
                    version: 1,
                    limits: serde_json::json!({
                        "board":"unihiker_k10",
                        "slot_bytes":2621440,
                        "sequence":1
                    }),
                },
            ])
            .await
            .unwrap();
        assert!(
            heartbeat_at_release_sequence.firmware_update.is_none(),
            "the Gateway must not re-offer a release at the Device's current or rejected sequence"
        );
        let sensor_response = simulator
            .report_sensor_readings(vec![
                SensorReading {
                    capability_id: "sensor.environment.temperature".to_string(),
                    capability_version: 1,
                    value: 22.75,
                    unit: "celsius".to_string(),
                },
                SensorReading {
                    capability_id: "sensor.environment.humidity".to_string(),
                    capability_version: 1,
                    value: 48.5,
                    unit: "percent_rh".to_string(),
                },
            ])
            .await
            .unwrap();
        assert!(sensor_response.accepted);
        let history_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM node_sensor_readings_history WHERE node_id = ?",
        )
        .bind(&enrollment.node_id)
        .fetch_one(store.pool())
        .await
        .unwrap();
        assert_eq!(history_count, 2);
        let tool = ReadNodeSensorsTool::new(store.pool().clone());
        let tool_output = tool.call(r#"{"_user_role":"Owner"}"#).await.unwrap();
        assert!(tool_output.contains("22.75"));
        assert!(tool_output.contains("72.95"));
        let health_tool = ReadNodeHealthTool::new(store.pool().clone());
        let health_output = health_tool.call(r#"{"_user_role":"Owner"}"#).await.unwrap();
        assert!(health_output.contains("recently_connected"));
        assert!(health_output.contains("aidaemon.runtime.recovery.v1"));
        assert!(health_output.contains("test_restart"));
        assert!(health_output.contains("reported_capabilities"));
        assert!(health_output.contains("output.audio"));
        assert!(health_output.contains("current_authorizations"));
        assert!(health_output.contains("receive_audio"));
        assert!(
            !health_output.contains("output.audio.volume"),
            "the capability snapshot must not invent an unreported volume control"
        );
        let events = simulator.text_turn("hello").await.unwrap();
        assert!(events
            .iter()
            .any(|event| event.event_type == "turn.thinking"));
        let complete = events
            .iter()
            .find(|event| event.event_type == "turn.complete")
            .unwrap();
        assert_eq!(complete.payload["text"], "Companion heard: hello");
        let media_events = simulator
            .media_turn(b"bounded fake wav fixture", "audio/wav", Some(800))
            .await
            .unwrap();
        assert!(media_events
            .iter()
            .any(|event| event.event_type == "turn.complete"));
        assert_eq!(
            std::fs::read_dir(media_dir.path()).unwrap().count(),
            0,
            "raw media should be deleted after processing"
        );

        let audio_bytes = b"RIFF bounded synthetic WAVE fixture";
        let announcement_path = media_dir.path().join("announcement.wav");
        tokio::fs::write(&announcement_path, audio_bytes)
            .await
            .unwrap();
        let target = store
            .resolve_audio_announcement_target(Some("Test Companion"))
            .await
            .unwrap();
        let digest = format!("{:x}", Sha256::digest(audio_bytes));
        let queued = store
            .queue_audio_announcement(
                &target,
                "audio/wav",
                audio_bytes.len() as u64,
                &digest,
                &announcement_path.to_string_lossy(),
                120,
                4,
            )
            .await
            .unwrap();
        let outbox = simulator.outbox(0).await.unwrap();
        assert_eq!(outbox.events.len(), 1);
        assert_eq!(outbox.events[0].event_type, "channel.audio");
        assert!(outbox.events[0].payload.get("text").is_none());
        let download_path = outbox.events[0].payload["audio"]["download_path"]
            .as_str()
            .unwrap();
        assert_eq!(
            simulator.download_media(download_path).await.unwrap(),
            audio_bytes
        );
        let acknowledgement = simulator
            .acknowledge_outbox(queued.cursor, NodeOutboxAckStatus::Played, None)
            .await
            .unwrap();
        assert!(acknowledgement.accepted);
        assert_eq!(acknowledgement.status, NodeOutboxAckStatus::Played);
        assert!(!announcement_path.exists());
        assert!(simulator.outbox(0).await.unwrap().events.is_empty());
        assert_eq!(
            store
                .node_outbox_receipt(&enrollment.node_id, queued.cursor)
                .await
                .unwrap()
                .unwrap()
                .status,
            "played"
        );

        let context = service
            .authenticate(simulator.identity.access_token.as_deref().unwrap())
            .await
            .unwrap();
        assert!(service
            .authorize(
                &context,
                crate::nodes::domain::NodeAction::ReceiveActuatorCommand
            )
            .await
            .is_err());
        let old_token = simulator.identity.access_token.clone().unwrap();
        let rotated = simulator.rotate_credential().await.unwrap();
        assert!(rotated.sessions_closed);
        assert!(service.authenticate(&old_token).await.is_err());
        simulator.open_session().await.unwrap();
        assert!(service
            .authenticate(simulator.identity.access_token.as_deref().unwrap())
            .await
            .is_ok());
        store.revoke_node(&context.node_id).await.unwrap();
        assert!(service
            .authenticate(simulator.identity.access_token.as_deref().unwrap())
            .await
            .is_err());
        server.abort();
    }
}
