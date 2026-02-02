use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tracing::{info, warn};
use std::sync::Arc;

use crate::agent::Agent;
use crate::config::DaemonConfig;

#[derive(Clone)]
struct AppState {
    agent: Arc<Agent>,
    auth_token: Option<String>,
}

/// Start the HTTP server (health check + agent API).
///
/// Binds to `bind_addr` (default "127.0.0.1") to avoid exposing the
/// endpoint on all interfaces. Set to "0.0.0.0" in config if external
/// access is needed.
pub async fn start_server(
    config: DaemonConfig,
    agent: Arc<Agent>,
) -> anyhow::Result<()> {
    let state = AppState {
        agent,
        auth_token: config.auth_token,
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/agent/message", post(agent_message_handler))
        .with_state(state);

    let ip: std::net::IpAddr = config.health_bind
        .parse()
        .unwrap_or_else(|_| std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = std::net::SocketAddr::new(ip, config.health_port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn agent_message_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    // 1. Auth check
    if let Some(expected_token) = &state.auth_token {
        let auth_header = headers.get("Authorization")
            .and_then(|h| h.to_str().ok());

        let valid = match auth_header {
            Some(h) if h.starts_with("Bearer ") => &h[7..] == expected_token,
            _ => false,
        };

        if !valid {
            warn!("Unauthorized access attempt to /agent/message");
            return Err((StatusCode::UNAUTHORIZED, "Unauthorized".to_string()));
        }
    }

    // 2. Parse payload
    let session_id = payload["session_id"].as_str()
        .ok_or((StatusCode::BAD_REQUEST, "Missing session_id".to_string()))?;
    let message = payload["message"].as_str()
        .ok_or((StatusCode::BAD_REQUEST, "Missing message".to_string()))?;

    // 3. Call agent
    match state.agent.handle_message(session_id, message).await {
        Ok(response) => Ok(Json(json!({ "response": response }))),
        Err(e) => {
            tracing::error!("Agent error handling remote message: {}", e);
            Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Agent error: {}", e)))
        }
    }
}

/// Generate and write a systemd service file (Linux).
#[cfg(target_os = "linux")]
pub fn install_service() -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let working_dir = std::env::current_dir()?;

    let unit = format!(
        r#"[Unit]
Description=aidaemon - AI personal daemon
After=network.target

[Service]
Type=simple
ExecStart={}
WorkingDirectory={}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"#,
        exe.display(),
        working_dir.display()
    );

    let path = "/etc/systemd/system/aidaemon.service";
    std::fs::write(path, unit)?;
    println!("Service file written to {}", path);
    println!("Run: sudo systemctl daemon-reload && sudo systemctl enable --now aidaemon");
    Ok(())
}

/// Generate and write a launchd plist file (macOS).
#[cfg(target_os = "macos")]
pub fn install_service() -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let working_dir = std::env::current_dir()?;

    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.aidaemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/aidaemon.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/aidaemon.stderr.log</string>
</dict>
</plist>
"#,
        exe.display(),
        working_dir.display()
    );

    let home = std::env::var("HOME")?;
    let path = format!("{}/Library/LaunchAgents/ai.aidaemon.plist", home);
    std::fs::write(&path, plist)?;
    println!("Plist written to {}", path);
    println!("Run: launchctl load {}", path);
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn install_service() -> anyhow::Result<()> {
    anyhow::bail!("Service installation is only supported on Linux and macOS");
}
