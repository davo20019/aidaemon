use axum::{routing::get, Json, Router};
use serde_json::json;
use tracing::info;

/// Start the health check HTTP server.
///
/// Binds to `bind_addr` (default "127.0.0.1") to avoid exposing the
/// endpoint on all interfaces. Set to "0.0.0.0" in config if external
/// access is needed.
pub async fn start_health_server(port: u16, bind_addr: &str) -> anyhow::Result<()> {
    let app = Router::new().route("/health", get(health_handler));

    let ip: std::net::IpAddr = bind_addr
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = std::net::SocketAddr::new(ip, port);
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

/// Bundle identifier used for the macOS `.app` and TCC permission grants.
///
/// macOS ties Accessibility / Screen Recording grants to a *stable code
/// identity*. A bare Mach-O binary launched by launchd has no durable identity
/// (ad-hoc signatures change on every rebuild and background agents can't be
/// granted reliably), which forces the `computer_use` permissions to be
/// re-granted constantly. Packaging the daemon as a signed `.app` with this
/// fixed bundle id makes grants persist across rebuilds — see
/// `COMPUTER_USE_MACOS.md`.
#[cfg(target_os = "macos")]
pub const MACOS_BUNDLE_ID: &str = "ai.aidaemon";

/// Generate and write a launchd plist file (macOS).
///
/// On macOS the daemon is installed as a signed `~/Applications/aidaemon.app`
/// bundle and launchd is pointed at the binary *inside* the bundle, so macOS
/// associates the running process with the bundle identifier and the
/// `computer_use` permission grants survive rebuilds.
#[cfg(target_os = "macos")]
pub fn install_service() -> anyhow::Result<()> {
    use std::path::PathBuf;
    use std::process::Command;

    let exe = std::env::current_exe()?;
    let working_dir = std::env::current_dir()?;
    let home = std::env::var("HOME")?;

    // Use user-private log directory instead of world-readable /tmp
    let log_dir = format!("{home}/Library/Logs/aidaemon");
    std::fs::create_dir_all(&log_dir)?;

    // 1. Build the .app bundle skeleton.
    let app_dir = PathBuf::from(&home).join("Applications/aidaemon.app");
    let macos_dir = app_dir.join("Contents/MacOS");
    std::fs::create_dir_all(&macos_dir)?;
    let bundle_bin = macos_dir.join("aidaemon");
    std::fs::copy(&exe, &bundle_bin)?;

    let info_plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>{bundle_id}</string>
    <key>CFBundleName</key><string>aidaemon</string>
    <key>CFBundleExecutable</key><string>aidaemon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>{version}</string>
    <key>CFBundleShortVersionString</key><string>{version}</string>
    <key>LSUIElement</key><true/>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
"#,
        bundle_id = MACOS_BUNDLE_ID,
        version = env!("CARGO_PKG_VERSION"),
    );
    std::fs::write(app_dir.join("Contents/Info.plist"), info_plist)?;

    // 2. Code-sign the bundle. Prefer a stable self-signed identity named
    //    "aidaemon-dev" (created by scripts/create-signing-identity.sh) so the
    //    designated requirement is identity-based and survives rebuilds. Fall
    //    back to ad-hoc signing with the fixed bundle id — durable for install-
    //    once users; developers who rebuild should create the identity.
    let have_identity = Command::new("security")
        .args(["find-identity", "-v", "-p", "codesigning"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("aidaemon-dev"))
        .unwrap_or(false);
    let sign_arg = if have_identity { "aidaemon-dev" } else { "-" };
    let sign = Command::new("codesign")
        .args(["-f", "-s", sign_arg, "--identifier", MACOS_BUNDLE_ID])
        .arg("--timestamp=none")
        .arg(&app_dir)
        .output()?;
    if !sign.status.success() {
        anyhow::bail!(
            "codesign failed: {}",
            String::from_utf8_lossy(&sign.stderr).trim()
        );
    }

    // 3. Point launchd at the binary inside the bundle (NOT wrapped in
    //    caffeinate — a wrapper process muddies the TCC identity). Idle-sleep
    //    prevention is handled by the daemon itself at startup (see
    //    `spawn_macos_keep_awake`).
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{wd}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log}/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{log}/stderr.log</string>
</dict>
</plist>
"#,
        label = MACOS_BUNDLE_ID,
        bin = bundle_bin.display(),
        wd = working_dir.display(),
        log = log_dir,
    );

    let path = format!("{home}/Library/LaunchAgents/ai.aidaemon.plist");
    std::fs::write(&path, plist)?;

    println!("Installed signed app bundle: {}", app_dir.display());
    println!(
        "Signed with {}.",
        if have_identity {
            "stable identity 'aidaemon-dev' (grants survive rebuilds)"
        } else {
            "ad-hoc signature (run scripts/create-signing-identity.sh for rebuild-durable grants)"
        }
    );
    println!("Plist written to {path}");
    println!("Logs will be written to {log_dir}/");
    println!("Run: launchctl bootstrap gui/$(id -u) {path}");
    println!("Then grant Accessibility + Screen Recording to aidaemon — see COMPUTER_USE_MACOS.md");
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn install_service() -> anyhow::Result<()> {
    anyhow::bail!("Service installation is only supported on Linux and macOS");
}

/// Prevent macOS idle sleep for the lifetime of the daemon.
///
/// The launchd plist launches the daemon binary directly (not wrapped in
/// `caffeinate`, which would muddy the TCC code identity needed for
/// `computer_use` permissions). To preserve the old keep-awake behaviour, the
/// daemon spawns its own `caffeinate -i -w <pid>` child that exits when the
/// daemon does. Best-effort: a missing `caffeinate` or spawn failure is logged
/// and ignored.
#[cfg(target_os = "macos")]
pub fn spawn_macos_keep_awake() {
    let pid = std::process::id();
    match std::process::Command::new("/usr/bin/caffeinate")
        .args(["-i", "-w", &pid.to_string()])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(_) => info!("macOS keep-awake active (caffeinate -i -w {pid})"),
        Err(e) => tracing::warn!("Could not start caffeinate keep-awake: {e}"),
    }
}
