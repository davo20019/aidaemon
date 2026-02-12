use axum::{
    extract::{ConnectInfo, Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use serde::Deserialize;
use serde_json::json;
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::agent;
use crate::config::ModelsConfig;
use crate::events::EventStore;
use crate::health::HealthProbeStore;
use crate::heartbeat::HeartbeatTelemetry;
use crate::oauth::OAuthGateway;

const DASHBOARD_HTML: &str = include_str!("dashboard.html");
const KEYCHAIN_FIELD: &str = "dashboard_token";
/// Dashboard token TTL: 24 hours.
const TOKEN_TTL_SECS: u64 = 86400;
/// Max failed auth attempts before rate limiting kicks in.
const MAX_FAILED_ATTEMPTS: u32 = 10;
/// Rate limit window: 15 minutes.
const RATE_LIMIT_WINDOW_SECS: u64 = 900;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct DashboardState {
    pub pool: SqlitePool,
    pub event_store: Option<Arc<EventStore>>,
    pub provider_kind: String,
    pub models: ModelsConfig,
    pub started_at: Instant,
    pub dashboard_token: String,
    pub token_created_at: Instant,
    pub daily_token_budget: Option<u64>,
    pub health_store: Option<Arc<HealthProbeStore>>,
    pub heartbeat_telemetry: Option<Arc<HeartbeatTelemetry>>,
    pub oauth_gateway: Option<OAuthGateway>,
    pub policy_window_days: u32,
    pub policy_max_divergence: f64,
    pub policy_uncertainty_threshold: f32,
    /// Rate limiter: maps IP address to (failure_count, first_failure_time).
    pub auth_failures: Arc<Mutex<HashMap<String, (u32, Instant)>>>,
}

// ---------------------------------------------------------------------------
// Token management
// ---------------------------------------------------------------------------

/// Token with creation timestamp for expiration checks.
pub struct DashboardToken {
    pub token: String,
    pub created_at: Instant,
}

pub fn get_or_create_dashboard_token() -> anyhow::Result<DashboardToken> {
    // Always generate a fresh token on startup (enforces TTL on restart)
    let tok = uuid::Uuid::new_v4().to_string();
    if let Err(e) = crate::config::store_in_keychain(KEYCHAIN_FIELD, &tok) {
        warn!("Could not store dashboard token in keychain: {e}");
    }
    let prefix = tok.get(..8).unwrap_or("????????");
    info!(
        "Dashboard token created (prefix: {}..., expires in 24h)",
        prefix
    );
    Ok(DashboardToken {
        token: tok,
        created_at: Instant::now(),
    })
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
        .route("/api/heartbeat/jobs", get(api_heartbeat_jobs))
        .route("/api/policy/metrics", get(api_policy_metrics))
        .route("/api/health/probes", get(api_health_probes))
        .route("/api/health/history", get(api_health_history))
        .route("/api/health/summary", get(api_health_summary))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .route("/health", get(health_handler))
        .route("/oauth/callback", get(oauth_callback_handler))
        .route("/", get(index_handler))
        .merge(api)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

async fn auth_middleware(
    State(state): State<DashboardState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Result<impl IntoResponse, StatusCode> {
    let client_ip = addr.ip().to_string();

    // Rate limiting: check if this IP has too many recent failures
    {
        let mut failures = state.auth_failures.lock().await;
        if let Some((count, first_failure)) = failures.get(&client_ip) {
            if first_failure.elapsed().as_secs() < RATE_LIMIT_WINDOW_SECS
                && *count >= MAX_FAILED_ATTEMPTS
            {
                warn!(ip = %client_ip, "Rate limited: too many failed auth attempts");
                return Err(StatusCode::TOO_MANY_REQUESTS);
            }
            // Reset window if expired
            if first_failure.elapsed().as_secs() >= RATE_LIMIT_WINDOW_SECS {
                failures.remove(&client_ip);
            }
        }
    }

    // Check token expiration
    if state.token_created_at.elapsed().as_secs() > TOKEN_TTL_SECS {
        warn!("Dashboard token expired (>24h). Restart the daemon to generate a new token.");
        return Err(StatusCode::UNAUTHORIZED);
    }

    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");

    if !constant_time_eq(token.as_bytes(), state.dashboard_token.as_bytes()) {
        // Track failed attempt
        let mut failures = state.auth_failures.lock().await;
        let entry = failures
            .entry(client_ip.clone())
            .or_insert((0, Instant::now()));
        entry.0 += 1;
        warn!(ip = %client_ip, attempts = entry.0, "Failed dashboard auth attempt");
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

#[derive(Deserialize)]
struct OAuthCallbackParams {
    state: Option<String>,
    code: Option<String>,
    error: Option<String>,
}

async fn oauth_callback_handler(
    State(state): State<DashboardState>,
    Query(params): Query<OAuthCallbackParams>,
) -> Html<String> {
    let gateway = match &state.oauth_gateway {
        Some(g) => g,
        None => {
            return Html("<html><body><h2>OAuth not enabled</h2><p>OAuth is not configured on this daemon.</p></body></html>".to_string());
        }
    };

    let state_param = match &params.state {
        Some(s) => s.as_str(),
        None => {
            return Html(
                "<html><body><h2>Error</h2><p>Missing state parameter.</p></body></html>"
                    .to_string(),
            );
        }
    };

    match gateway
        .handle_callback(state_param, params.code.as_deref(), params.error.as_deref())
        .await
    {
        Ok(msg) => {
            let safe_msg = html_escape(&msg);
            let is_error =
                msg.contains("denied") || msg.contains("failed") || msg.contains("expired");
            let (icon, title) = if is_error {
                ("&#10060;", "OAuth Error")
            } else {
                ("&#9989;", "Connected!")
            };
            Html(format!(
                "<html><head><title>{title}</title></head>\
                 <body style=\"font-family:sans-serif;text-align:center;padding:60px\">\
                 <h1>{icon}</h1><h2>{title}</h2><p>{safe_msg}</p>\
                 <p style=\"color:#888\">You can close this tab and return to your chat.</p>\
                 </body></html>"
            ))
        }
        Err(e) => {
            warn!("OAuth callback error: {}", e);
            let safe_err = html_escape(&e.to_string());
            Html(format!(
                "<html><head><title>OAuth Error</title></head>\
                 <body style=\"font-family:sans-serif;text-align:center;padding:60px\">\
                 <h1>&#10060;</h1><h2>OAuth Error</h2><p>{safe_err}</p>\
                 <p style=\"color:#888\">Please try again.</p>\
                 </body></html>"
            ))
        }
    }
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

/// Escape HTML special characters to prevent XSS in rendered pages.
/// Constant-time byte comparison to prevent timing side-channel attacks
/// on dashboard bearer token authentication.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
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
            // Truncate prompt to prevent leaking sensitive task instructions
            // through the dashboard API. Full prompt visible via direct DB only.
            let truncated_prompt = if r.prompt.len() > 100 {
                format!("{}...", &r.prompt[..100])
            } else {
                r.prompt.clone()
            };
            json!({
                "id": r.id,
                "name": r.name,
                "original_schedule": r.original_schedule,
                "cron_expr": r.cron_expr,
                "prompt": truncated_prompt,
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

async fn api_heartbeat_jobs(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let Some(telemetry) = &state.heartbeat_telemetry else {
        return Json(json!({
            "error": "Heartbeat telemetry unavailable",
            "jobs": [],
            "maintenance_jobs": []
        }));
    };

    let jobs = telemetry.snapshots();
    let maintenance_names = [
        "embeddings",
        "consolidation",
        "memory_decay",
        "event_pruning",
        "retention_cleanup",
    ];
    let maintenance_jobs: Vec<_> = jobs
        .iter()
        .filter(|job| maintenance_names.contains(&job.name.as_str()))
        .cloned()
        .collect();

    Json(json!({
        "jobs": jobs,
        "maintenance_jobs": maintenance_jobs
    }))
}

async fn api_policy_metrics(State(state): State<DashboardState>) -> Json<serde_json::Value> {
    let metrics = agent::policy_metrics_snapshot();
    let autotune = agent::policy_autotune_snapshot(state.policy_uncertainty_threshold);
    let divergence_rate = if metrics.router_shadow_total > 0 {
        metrics.router_shadow_diverged as f64 / metrics.router_shadow_total as f64
    } else {
        0.0
    };
    let avg_tools_before = if metrics.tool_exposure_samples > 0 {
        metrics.tool_exposure_before_sum as f64 / metrics.tool_exposure_samples as f64
    } else {
        0.0
    };
    let avg_tools_after = if metrics.tool_exposure_samples > 0 {
        metrics.tool_exposure_after_sum as f64 / metrics.tool_exposure_samples as f64
    } else {
        0.0
    };
    let mut graduation = serde_json::json!({
        "window_days": state.policy_window_days,
        "observed_days": 0.0,
        "total_decisions": 0,
        "diverged_decisions": 0,
        "divergence_rate": 0.0,
        "gate_passed": false,
    });
    if let Some(store) = &state.event_store {
        match store
            .policy_graduation_report(state.policy_window_days)
            .await
        {
            Ok(report) => {
                graduation = serde_json::json!({
                    "window_days": report.window_days,
                    "observed_days": report.observed_days,
                    "total_decisions": report.total_decisions,
                    "diverged_decisions": report.diverged_decisions,
                    "divergence_rate": report.divergence_rate,
                    "gate_passed": report.gate_passes(state.policy_max_divergence),
                    "max_divergence": state.policy_max_divergence
                });
            }
            Err(e) => {
                warn!(error = %e, "Failed to load policy graduation report");
            }
        }
    }
    Json(json!({
        "router_shadow_total": metrics.router_shadow_total,
        "router_shadow_diverged": metrics.router_shadow_diverged,
        "router_divergence_rate": divergence_rate,
        "tool_exposure_samples": metrics.tool_exposure_samples,
        "avg_tools_before_filter": avg_tools_before,
        "avg_tools_after_filter": avg_tools_after,
        "ambiguity_detected_total": metrics.ambiguity_detected_total,
        "uncertainty_clarify_total": metrics.uncertainty_clarify_total,
        "uncertainty_threshold": autotune.uncertainty_threshold,
        "context_refresh_total": metrics.context_refresh_total,
        "escalation_total": metrics.escalation_total,
        "fallback_expansion_total": metrics.fallback_expansion_total,
        "graduation": graduation
    }))
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
    let probe = store.get_probe(&q.probe).await.ok().flatten().or({
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
        let is_healthy = latest
            .as_ref()
            .map(|r| r.status.is_healthy())
            .unwrap_or(true);

        if is_healthy {
            healthy += 1;
        } else {
            unhealthy += 1;
        }

        let stats = stats_map.get(&probe.id);
        let health_score = stats.map(crate::health::TrendAnalyzer::health_score);
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
    let app = build_router(state).into_make_service_with_connect_info::<SocketAddr>();

    let ip: std::net::IpAddr = bind_addr
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = std::net::SocketAddr::new(ip, port);
    info!("Dashboard server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
