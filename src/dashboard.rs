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
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use crate::config::ModelsConfig;
use crate::health::HealthProbeStore;

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
    pub health_store: Option<Arc<HealthProbeStore>>,
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
        .route("/api/health/probes", get(api_health_probes))
        .route("/api/health/history", get(api_health_history))
        .route("/api/health/summary", get(api_health_summary))
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
// Health Probe API Endpoints
// ---------------------------------------------------------------------------

async fn api_health_probes(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let Some(store) = &state.health_store else {
        return Json(json!({
            "error": "Health probes not enabled",
            "probes": []
        }));
    };

    let probes = store.list_probes().await.unwrap_or_default();

    let mut probe_data = Vec::new();
    for probe in probes {
        let latest = store.get_latest_result(&probe.id).await.unwrap_or(None);
        let consecutive_failures = store
            .count_consecutive_failures(&probe.id)
            .await
            .unwrap_or(0);

        probe_data.push(json!({
            "id": probe.id,
            "name": probe.name,
            "description": probe.description,
            "type": probe.probe_type.as_str(),
            "target": probe.target,
            "schedule": probe.schedule,
            "source": probe.source,
            "is_paused": probe.is_paused,
            "consecutive_failures_alert": probe.consecutive_failures_alert,
            "latency_threshold_ms": probe.latency_threshold_ms,
            "last_run_at": probe.last_run_at.map(|t| t.to_rfc3339()),
            "next_run_at": probe.next_run_at.to_rfc3339(),
            "last_status": latest.as_ref().map(|r| r.status.as_str()),
            "last_latency_ms": latest.as_ref().and_then(|r| r.latency_ms),
            "last_checked": latest.as_ref().map(|r| r.checked_at.to_rfc3339()),
            "consecutive_failures": consecutive_failures,
        }));
    }

    Json(serde_json::Value::Array(probe_data))
}

#[derive(Deserialize)]
struct HealthHistoryQuery {
    probe: String,
    #[serde(default = "default_history_hours")]
    hours: u32,
}

fn default_history_hours() -> u32 {
    24
}

async fn api_health_history(
    State(state): State<DashboardState>,
    Query(q): Query<HealthHistoryQuery>,
) -> Json<serde_json::Value> {
    let Some(store) = &state.health_store else {
        return Json(json!({
            "error": "Health probes not enabled",
            "results": []
        }));
    };

    // Find probe by ID or name
    let probe = store.get_probe(&q.probe).await.ok().flatten()
        .or_else(|| {
            // Try by name synchronously is tricky; we'll use a future
            None
        });

    let probe_id = if let Some(p) = probe {
        p.id
    } else {
        // Try by name
        match store.get_probe_by_name(&q.probe).await {
            Ok(Some(p)) => p.id,
            _ => {
                return Json(json!({
                    "error": format!("Probe not found: {}", q.probe),
                    "results": []
                }));
            }
        }
    };

    let hours = q.hours.min(168); // Max 7 days
    let end = chrono::Utc::now();
    let start = end - chrono::Duration::hours(hours as i64);

    let results = store
        .get_results_in_range(&probe_id, start, end)
        .await
        .unwrap_or_default();

    let result_data: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "status": r.status.as_str(),
                "latency_ms": r.latency_ms,
                "error_message": r.error_message,
                "checked_at": r.checked_at.to_rfc3339(),
            })
        })
        .collect();

    Json(json!({
        "probe_id": probe_id,
        "hours": hours,
        "count": result_data.len(),
        "results": result_data
    }))
}

async fn api_health_summary(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let Some(store) = &state.health_store else {
        return Json(json!({
            "error": "Health probes not enabled",
            "total": 0,
            "healthy": 0,
            "unhealthy": 0,
            "paused": 0,
            "probes": []
        }));
    };

    let probes = store.list_probes().await.unwrap_or_default();
    let stats_map = store.get_all_probe_stats(24).await.unwrap_or_default();

    let mut total = 0u32;
    let mut healthy = 0u32;
    let mut unhealthy = 0u32;
    let mut paused = 0u32;

    let mut probe_summaries = Vec::new();

    for probe in probes {
        total += 1;

        if probe.is_paused {
            paused += 1;
            probe_summaries.push(json!({
                "id": probe.id,
                "name": probe.name,
                "status": "paused",
                "health_score": null,
            }));
            continue;
        }

        let latest = store.get_latest_result(&probe.id).await.unwrap_or(None);
        let is_healthy = latest.as_ref().map(|r| r.status.is_healthy()).unwrap_or(true);

        if is_healthy {
            healthy += 1;
        } else {
            unhealthy += 1;
        }

        let stats = stats_map.get(&probe.id);
        let health_score = stats.map(|s| crate::health::TrendAnalyzer::health_score(s));
        let uptime = stats.map(|s| s.uptime_percent);
        let avg_latency = stats.and_then(|s| s.avg_latency_ms);

        probe_summaries.push(json!({
            "id": probe.id,
            "name": probe.name,
            "status": if is_healthy { "healthy" } else { "unhealthy" },
            "health_score": health_score,
            "uptime_percent": uptime,
            "avg_latency_ms": avg_latency,
        }));
    }

    // Calculate overall health score
    let overall_score = if total > paused {
        let active = total - paused;
        ((healthy as f64 / active as f64) * 100.0) as u32
    } else {
        100 // No active probes = healthy
    };

    Json(json!({
        "total": total,
        "healthy": healthy,
        "unhealthy": unhealthy,
        "paused": paused,
        "overall_health_score": overall_score,
        "probes": probe_summaries
    }))
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
