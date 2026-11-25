use axum::{
    Json, Router,
    extract::{ConnectInfo, Query, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse},
    routing::{get, post},
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

use crate::events::EventStore;
use crate::oauth::{OAuthCallbackOutcome, OAuthGateway};
use crate::traits::{MandateWakeSignal, StateStore};

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
    pub mandate_store: Arc<dyn StateStore>,
    pub heartbeat_wake_tx: tokio::sync::mpsc::Sender<()>,
    pub event_store: Option<Arc<EventStore>>,
    pub dashboard_token: String,
    pub token_created_at: Instant,
    pub oauth_gateway: Option<OAuthGateway>,
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
    info!("Dashboard token created and stored (expires in 24h)");
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
        .route("/api/mandates/signals", post(api_mandate_signal))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .route("/health", get(health_handler))
        .route("/oauth/callback", get(oauth_callback_handler))
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

async fn health_handler(State(state): State<DashboardState>) -> impl IntoResponse {
    crate::daemon::health_response(&state.pool, state.event_store.as_deref()).await
}

/// Authenticated, content-free webhook ingress for proactive mandate wake-ups.
/// The store performs exact account/URL-scope matching and durable deduplication.
async fn api_mandate_signal(
    State(state): State<DashboardState>,
    Json(signal): Json<MandateWakeSignal>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, Json<serde_json::Value>)> {
    signal
        .validate()
        .map_err(|error| (StatusCode::BAD_REQUEST, Json(json!({"error": error}))))?;
    let awakened = state
        .mandate_store
        .wake_mandates_for_signal(&signal)
        .await
        .map_err(|error| {
            tracing::error!(%error, "Failed to route mandate wake signal");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "mandate signal routing failed"})),
            )
        })?;
    if !awakened.is_empty() {
        let _ = state.heartbeat_wake_tx.try_send(());
    }
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({"awakened_count": awakened.len(), "mandate_ids": awakened})),
    ))
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
        Ok(outcome) => Html(oauth_callback_page(&outcome)),
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

/// The page a finished OAuth callback shows; success or error styling comes
/// from the outcome's variant.
fn oauth_callback_page(outcome: &OAuthCallbackOutcome) -> String {
    let safe_msg = html_escape(outcome.message());
    let (icon, title) = if outcome.is_connected() {
        ("&#9989;", "Connected!")
    } else {
        ("&#10060;", "OAuth Error")
    };
    format!(
        "<html><head><title>{title}</title></head>\
         <body style=\"font-family:sans-serif;text-align:center;padding:60px\">\
         <h1>{icon}</h1><h2>{title}</h2><p>{safe_msg}</p>\
         <p style=\"color:#888\">You can close this tab and return to your chat.</p>\
         </body></html>"
    )
}

/// Constant-time byte comparison, so a token check leaks nothing through
/// timing (`subtle`; unequal lengths are unequal).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    subtle::ConstantTimeEq::ct_eq(a, b).into()
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_callback_page_styles_from_the_outcome_variant() {
        let connected = oauth_callback_page(&OAuthCallbackOutcome::Connected {
            message: "Connected to Linear".to_string(),
        });
        assert!(connected.contains("Connected!"));
        assert!(!connected.contains("OAuth Error"));

        // A failure whose text names none of the old keywords is still an error.
        let failed = oauth_callback_page(&OAuthCallbackOutcome::Failed {
            message: "No authorization code in callback".to_string(),
        });
        assert!(failed.contains("OAuth Error"));
        assert!(!failed.contains("Connected!"));

        // A success whose text happens to contain "failed" stays a success.
        let worded = oauth_callback_page(&OAuthCallbackOutcome::Connected {
            message: "Connected; an earlier attempt failed".to_string(),
        });
        assert!(worded.contains("Connected!"));
    }

    #[test]
    fn oauth_callback_page_escapes_the_message() {
        let page = oauth_callback_page(&OAuthCallbackOutcome::Failed {
            message: "<script>x</script>".to_string(),
        });
        assert!(!page.contains("<script>"));
    }
}
