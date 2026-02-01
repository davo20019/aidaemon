use axum::{routing::get, Json, Router};
use serde_json::json;
use tracing::info;

/// Start the health check HTTP server.
pub async fn start_health_server(port: u16) -> anyhow::Result<()> {
    let app = Router::new().route("/health", get(health_handler));

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    info!("Health server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
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
