use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use serde::Deserialize;
use serde_json::json;
use sqlx::SqlitePool;
use std::time::Instant;
use tracing::{info, warn};

use crate::config::ModelsConfig;

const DASHBOARD_HTML: &str = include_str!("dashboard.html");
const KEYCHAIN_FIELD: &str = "dashboard_token";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DashboardState {
    pub pool: SqlitePool,
    pub provider_kind: String,
    pub models: ModelsConfig,
    pub started_at: Instant,
    pub dashboard_token: String,
    pub daily_token_budget: Option<u64>,
}

// ---------------------------------------------------------------------------
// Token management
// ---------------------------------------------------------------------------

pub fn get_or_create_dashboard_token() -> anyhow::Result<String> {
    match keyring::Entry::new("aidaemon", KEYCHAIN_FIELD) {
        Ok(entry) => match entry.get_password() {
            Ok(tok) if !tok.is_empty() => {
                info!("Dashboard token loaded from keychain");
                Ok(tok)
            }
            _ => {
                let tok = uuid::Uuid::new_v4().to_string();
                if let Err(e) = entry.set_password(&tok) {
                    warn!("Could not store dashboard token in keychain: {e}");
                }
                // Only log a prefix to avoid exposing the full token in logs
                let prefix = tok.get(..8).unwrap_or("????????");
                info!("Dashboard token created (prefix: {}...)", prefix);
                Ok(tok)
            }
        },
        Err(e) => {
            warn!("Keychain unavailable for dashboard token: {e}");
            let tok = uuid::Uuid::new_v4().to_string();
            // Only log a prefix to avoid exposing the full token in logs
            let prefix = tok.get(..8).unwrap_or("????????");
            info!("Ephemeral dashboard token created (prefix: {}..., not persisted)", prefix);
            Ok(tok)
        }
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn build_router(state: DashboardState) -> Router {
    let api = Router::new()
        .route("/api/status", get(api_status))
        .route("/api/usage", get(api_usage))
        .route("/api/sessions", get(api_sessions))
        .route("/api/tasks", get(api_tasks))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .route("/health", get(health_handler))
        .route("/", get(index_handler))
        .merge(api)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

async fn auth_middleware(
    State(state): State<DashboardState>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Result<impl IntoResponse, StatusCode> {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");

    if token != state.dashboard_token {
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(next.run(request).await)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn index_handler() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn api_status(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let uptime = state.started_at.elapsed().as_secs();
    Json(json!({
        "provider": state.provider_kind,
        "models": {
            "primary": state.models.primary,
            "fast": state.models.fast,
            "smart": state.models.smart,
        },
        "uptime_secs": uptime,
        "version": env!("CARGO_PKG_VERSION"),
        "daily_token_budget": state.daily_token_budget,
    }))
}

#[derive(Deserialize)]
struct UsageQuery {
    #[serde(default = "default_usage_days")]
    days: u32,
}

fn default_usage_days() -> u32 {
    7
}

async fn api_usage(
    State(state): State<DashboardState>,
    Query(q): Query<UsageQuery>,
) -> Json<serde_json::Value> {
    let days = q.days.min(90);
    let rows = sqlx::query_as::<_, UsageRow>(
        "SELECT date(created_at) as day, model, \
         SUM(input_tokens) as input_tokens, SUM(output_tokens) as output_tokens, \
         COUNT(*) as request_count \
         FROM token_usage \
         WHERE created_at >= datetime('now', '-' || ? || ' days') \
         GROUP BY day, model ORDER BY day DESC, model",
    )
    .bind(days)
    .fetch_all(&state.pool)
    .await
    .unwrap_or_default();

    let vals: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "day": r.day,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "request_count": r.request_count,
            })
        })
        .collect();

    Json(serde_json::Value::Array(vals))
}

#[derive(sqlx::FromRow)]
struct UsageRow {
    day: Option<String>,
    model: String,
    input_tokens: i64,
    output_tokens: i64,
    request_count: i64,
}

#[derive(Deserialize)]
struct SessionsQuery {
    #[serde(default = "default_sessions_limit")]
    limit: u32,
}

fn default_sessions_limit() -> u32 {
    20
}

async fn api_sessions(
    State(state): State<DashboardState>,
    Query(q): Query<SessionsQuery>,
) -> Json<serde_json::Value> {
    let limit = q.limit.min(100);
    let rows = sqlx::query_as::<_, SessionRow>(
        "SELECT session_id, MAX(created_at) as last_activity, \
         COUNT(*) as message_count, MIN(created_at) as first_message \
         FROM messages GROUP BY session_id \
         ORDER BY last_activity DESC LIMIT ?",
    )
    .bind(limit)
    .fetch_all(&state.pool)
    .await
    .unwrap_or_default();

    let vals: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "session_id": r.session_id,
                "last_activity": r.last_activity,
                "message_count": r.message_count,
                "first_message": r.first_message,
            })
        })
        .collect();

    Json(serde_json::Value::Array(vals))
}

#[derive(sqlx::FromRow)]
struct SessionRow {
    session_id: String,
    last_activity: Option<String>,
    message_count: i64,
    first_message: Option<String>,
}

async fn api_tasks(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let rows = sqlx::query_as::<_, TaskRow>(
        "SELECT id, name, original_schedule, cron_expr, prompt, source, \
         is_oneshot, is_paused, is_trusted, last_run_at, next_run_at \
         FROM scheduled_tasks ORDER BY next_run_at ASC",
    )
    .fetch_all(&state.pool)
    .await
    .unwrap_or_default();

    let vals: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|r| {
            json!({
                "id": r.id,
                "name": r.name,
                "original_schedule": r.original_schedule,
                "cron_expr": r.cron_expr,
                "prompt": r.prompt,
                "source": r.source,
                "is_oneshot": r.is_oneshot != 0,
                "is_paused": r.is_paused != 0,
                "is_trusted": r.is_trusted != 0,
                "last_run_at": r.last_run_at,
                "next_run_at": r.next_run_at,
            })
        })
        .collect();

    Json(serde_json::Value::Array(vals))
}

#[derive(sqlx::FromRow)]
struct TaskRow {
    id: String,
    name: String,
    original_schedule: String,
    cron_expr: String,
    prompt: String,
    source: String,
    is_oneshot: i64,
    is_paused: i64,
    is_trusted: i64,
    last_run_at: Option<String>,
    next_run_at: String,
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

pub async fn start_dashboard_server(
    state: DashboardState,
    port: u16,
    bind_addr: &str,
) -> anyhow::Result<()> {
    let app = build_router(state);

    let ip: std::net::IpAddr = bind_addr
        .parse()
        .unwrap_or_else(|_| std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = std::net::SocketAddr::new(ip, port);
    info!("Dashboard server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
